"""Paired SM90 MegaMoE benchmark across weight formats: fp8 / mxfp4 / nvfp4.

Arms run in one process on identical shapes, identical routing and the same
underlying BF16 weights, so the only difference is the quantized weight format
and the kernel consuming it. Arms are interleaved and repeated because
throughput on these nodes drifts several percent between runs; a single
A-then-B pass cannot separate a real delta from that drift.

Reachability on a given device is not uniform, and this matters more than the
numbers:

  fp8    generic -- no shape or SM lock.
  mxfp4  shape-locked to 8 ranks / 384 experts / topk 8 / hidden 6144 / ih 2048
         (SM90MXFP4H200FusedShape::is_supported_shape), but its kernel takes
         kNumSMs as a template parameter, so it runs on any SM90 device.
  nvfp4  same shape lock AND num_sms == 132, because 132 is baked into the
         kernel body rather than templated. It therefore cannot run on H20
         (78 SMs) at all.

fp8 launches two kernels (l1, l2); mxfp4/nvfp4 are fused single kernels. The
fp8 time reported is the sum of both.
"""
import argparse
import os
import random
import statistics
import sys
from typing import Tuple

import torch
import torch.distributed as dist

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

import deep_gemm
from deep_gemm.quantization_mxfp4 import quantize_to_mxfp4
from deep_gemm.utils import per_token_cast_to_fp8
from deep_gemm.utils.dist import dist_print, init_dist, uneven_all_gather
from deep_gemm.testing import bench_kineto, get_arch_major

FP8_KERNELS = ('sm90_fp8_mega_moe_l1_impl', 'sm90_fp8_mega_moe_l2_impl')


def _quantize_grouped_fp8_block_128_128(weights: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    num_groups, n, k = weights.shape
    assert n % 128 == 0 and k % 128 == 0
    weights_fp8 = torch.empty_like(weights, dtype=torch.float8_e4m3fn)
    scales = torch.empty((num_groups, n // 128, k // 128), dtype=torch.float, device=weights.device)
    for start in range(0, num_groups, 4):
        end = min(start + 4, num_groups)
        block = weights[start:end].view(end - start, n // 128, 128, k // 128, 128).float()
        bs = block.abs().amax(dim=(-1, -3)).clamp(1e-4) / 448.0
        weights_fp8[start:end].copy_(
            (block / bs.unsqueeze(-1).unsqueeze(-3)).to(torch.float8_e4m3fn).view(end - start, n, k))
        scales[start:end].copy_(bs)
    return weights_fp8, scales.contiguous()


class Arm:
    """One weight format: its symm buffer, transformed weights and runner.

    ``sched`` is an optional ``{env: value}`` overlay applied around this arm's
    launch only. The SM90 MXFP4 selector reads its DG_MXFP4_* overrides through
    getenv on every launch, so several schedules can be compared as separate
    arms of one process -- which is the only way to compare them at all, since
    a fresh process draws a different router and therefore a different number
    of routed tokens.
    """

    def __init__(self, name, group, num_experts, cap, num_topk, hidden, ih,
                 l1_bf, l2_bf, x_fp8, x_sf, topk_idx, topk_w, cum_stats,
                 num_tokens, activation_clamp, fast_math, sched=None):
        self.sched = sched or {}
        self.fmt = name.split('[')[0]
        self.name = name
        self.num_tokens = num_tokens
        if self.fmt == 'fp8':
            self.buffer = deep_gemm.get_symm_buffer_for_sm90_mega_moe(
                group, num_experts, cap, num_topk, hidden, ih)
            l1 = _quantize_grouped_fp8_block_128_128(l1_bf)
            l2 = _quantize_grouped_fp8_block_128_128(l2_bf)
            self.t1, self.t2 = deep_gemm.transform_weights_for_mega_moe_sm90(l1, l2)
            self.kernels = FP8_KERNELS
            self.scale_bytes_per_k = 0.0   # block-(128,128) FP32 SF, negligible
        else:
            self.buffer = deep_gemm.get_symm_buffer_for_mega_moe(
                group, num_experts, cap, num_topk, hidden, ih)
            if self.fmt == 'mxfp4':
                gs = 32
                l1, l2 = quantize_to_mxfp4(l1_bf, group_size=gs), quantize_to_mxfp4(l2_bf, group_size=gs)
                self.t1, self.t2 = deep_gemm.transform_mxfp4_weights_for_mega_moe_sm90(l1, l2)
                self.kernels = 'sm90_mxfp4_mega_moe'
            else:
                from deep_gemm.quantization_nvfp4 import quantize_to_nvfp4
                gs = 16
                l1, l2 = quantize_to_nvfp4(l1_bf, group_size=gs), quantize_to_nvfp4(l2_bf, group_size=gs)
                self.t1, self.t2 = deep_gemm.transform_nvfp4_weights_for_mega_moe_sm90(l1, l2)
                self.kernels = 'sm90_nvfp4_mega_moe'
            self.scale_bytes_per_k = 1.0 / gs
        self._x_fp8, self._x_sf = x_fp8, x_sf
        self._topk_idx, self._topk_w = topk_idx, topk_w
        self._cum, self._hidden = cum_stats, hidden
        self._clamp, self._fm = activation_clamp, fast_math

    def run(self):
        prev = {k: os.environ.get(k) for k in self.sched}
        os.environ.update({k: str(v) for k, v in self.sched.items()})
        try:
            return self._run()
        finally:
            for k, v in prev.items():
                if v is None:
                    os.environ.pop(k, None)
                else:
                    os.environ[k] = v

    def _run(self):
        n = self.num_tokens
        b = self.buffer
        b.x[:n].copy_(self._x_fp8)
        b.x_sf[:n].copy_(self._x_sf)
        b.topk_idx[:n].copy_(self._topk_idx)
        b.topk_weights[:n].copy_(self._topk_w)
        y = torch.empty((n, self._hidden), dtype=torch.bfloat16, device='cuda')
        if self.fmt == 'fp8':
            deep_gemm.fp8_mega_moe(
                y, self.t1, self.t2, b,
                cumulative_local_expert_recv_stats=self._cum,
                recipe=(128, 128, 128), activation='swiglu',
                activation_clamp=self._clamp, fast_math=self._fm)
        else:
            entry = deep_gemm.mxfp4_mega_moe if self.fmt == 'mxfp4' else deep_gemm.nvfp4_mega_moe
            entry(y, self.t1, self.t2, b,
                  cumulative_local_expert_recv_stats=self._cum,
                  activation_clamp=self._clamp, fast_math=self._fm)
        return y

    def time_once(self, num_tests, show_kineto):
        t = bench_kineto(self.run, self.kernels, barrier=dist.barrier,
                         num_tests=num_tests, suppress_kineto_output=not show_kineto)
        # fp8 is two kernels; bench_kineto returns a tuple for tuple input.
        return sum(t) if isinstance(t, (tuple, list)) else t


def _run_one_config(args, num_tokens, cap, hidden, ih, num_experts, num_topk,
                    num_ranks, rank_idx, group, activation_clamp, fast_math):
    num_experts_per_rank = num_experts // num_ranks
    assert num_tokens <= cap

    x_bf = torch.randn((num_tokens, hidden), dtype=torch.bfloat16, device='cuda')
    l1_bf = torch.randn((num_experts_per_rank, ih * 2, hidden), dtype=torch.bfloat16, device='cuda') * 0.05
    l2_bf = torch.randn((num_experts_per_rank, hidden, ih), dtype=torch.bfloat16, device='cuda') * 0.05
    scores = torch.randn((num_tokens, num_experts), dtype=torch.float, device='cuda')
    topk_w, topk_idx = torch.topk(scores, num_topk, dim=-1, largest=True, sorted=False)
    if args.masked_ratio > 0:
        m = torch.rand_like(topk_idx, dtype=torch.float) < args.masked_ratio
        topk_idx.masked_fill_(m, -1)
        topk_w.masked_fill_(m, 0)

    x_fp8, x_sf = per_token_cast_to_fp8(x_bf, use_ue8m0=False, gran_k=128, use_packed_ue8m0=False)
    cum = torch.zeros(num_experts_per_rank, dtype=torch.int, device='cuda')

    arms = {}
    for name in args.arms:
        arms[name] = Arm(name, group, num_experts, cap, num_topk, hidden, ih,
                         l1_bf, l2_bf, x_fp8, x_sf, topk_idx, topk_w, cum,
                         num_tokens, activation_clamp, fast_math)
    for spec in args.mxfp4_scheds:
        fields = [int(v) for v in spec.split(',')]
        bm, st, epw = fields[:3]
        # Whether one or both dispatch warps stay active is a shared-memory
        # decision as much as a routing one: the second warp's send buffer is
        # another `hidden` bytes, which is what decides how deep the pipeline
        # can go. Default to the table's own choice unless the spec pins it.
        single = fields[3] if len(fields) > 3 else None
        rs = fields[4] if len(fields) > 4 else None
        label = (f'mxfp4[{bm}/{st}/{epw}'
                 + (f'/s{single}' if single is not None else '')
                 + (f'/rs{rs}' if rs is not None else '') + ']')
        # A duplicate label would drop the earlier Arm on the floor, and its
        # symmetric buffer would then be torn down mid-run by the collector.
        assert label not in arms, f'duplicate schedule arm {label}'
        arms[label] = Arm(label, group, num_experts, cap, num_topk, hidden, ih,
                          l1_bf, l2_bf, x_fp8, x_sf, topk_idx, topk_w, cum,
                          num_tokens, activation_clamp, fast_math,
                          sched={'DG_MXFP4_BLOCK_M': bm, 'DG_MXFP4_STAGES': st,
                                 'DG_MXFP4_EPW': epw,
                                 # swapAB is a property of the tile, not of the
                                 # batch: the transposed path packs tokens into
                                 # WGMMA N and so needs BLOCK_M <= 24, while the
                                 # straight path needs a full M64 tile. Deriving
                                 # it here keeps every variant self-consistent.
                                 'DG_MXFP4_SWAP_AB': 1 if bm <= 24 else 0,
                                 **({} if single is None
                                    else {'DG_MXFP4_SINGLE_DISPATCH': single}),
                                 **({} if rs is None
                                    else {'DG_MXFP4_RS': rs})})
    order = list(arms)
    del l1_bf, l2_bf, x_bf, scores

    for a in arms.values():
        a.run()
    dist.barrier()

    show_kineto = os.environ.get('DG_SHOW_KINETO', '0') != '0'
    samples = {n: [] for n in order}
    for _ in range(args.reps):              # interleaved, so drift hits all arms alike
        for n in order:
            samples[n].append(arms[n].time_once(args.num_tests, show_kineto))

    med = {n: statistics.median(samples[n]) for n in order}

    gathered = uneven_all_gather(topk_idx, group=group)
    gathered[(gathered < rank_idx * num_experts_per_rank) |
             (gathered >= (rank_idx + 1) * num_experts_per_rank)] = -1
    num_recv = (gathered != -1).sum().item()
    touched = max(0, torch.unique(gathered.flatten()).numel() - 1)

    sd = lambda a, b: float('nan') if not b else a / b
    tf = lambda t: sd(2 * num_recv * (hidden * ih * 3) / 1e12, t)

    parts = []
    for n in order:
        parts.append(f'{n}={med[n]*1e6:7.1f}us({tf(med[n]):5.1f}TF)')
    base = args.baseline if args.baseline in med else None
    delta = ''
    if base:
        for n in order:
            if n == base:
                continue
            d = (med[n] - med[base]) / med[base] * 100
            delta += f'  {n} vs {base} = {d:+6.2f}%'
    dist_print(f' M={num_tokens:4d} recv={num_recv:5d} exp={touched:3d} | ' +
               ' '.join(parts) + ' |' + delta, once_in_node=True)
    if args.verbose:
        for n in order:
            dist_print(f'    raw {n}={[round(v*1e6, 1) for v in samples[n]]}', once_in_node=True)
    dist.barrier()
    for a in arms.values():
        a.buffer.destroy()


def test(local_rank: int, num_local_ranks: int, args: argparse.Namespace):
    rank_idx, num_ranks, group = init_dist(local_rank, num_local_ranks)
    torch.manual_seed(args.seed + rank_idx)
    random.seed(args.seed + rank_idx)

    if get_arch_major() != 9:
        dist_print(f'[SKIP] requires SM90, got SM{get_arch_major()}0', once_in_node=True)
        dist.destroy_process_group()
        return

    num_sms = torch.cuda.get_device_properties(0).multi_processor_count
    if 'nvfp4' in args.arms and num_sms != 132:
        dist_print(f'[SKIP nvfp4] kernel body hardcodes 132 SMs; this device has {num_sms}. '
                   f'Dropping the nvfp4 arm.', once_in_node=True)
        args.arms = [a for a in args.arms if a != 'nvfp4']

    batches = args.batches if args.batches is not None else [1, 8, 16, 32, 64, 128, 256]
    dist_print(
        f'SM90 MegaMoE format bench: dev={torch.cuda.get_device_name(0)} sms={num_sms} '
        f'ranks={num_ranks} hidden={args.hidden} ih={args.intermediate_hidden} '
        f'experts={args.num_experts} topk={args.num_topk} arms={args.arms} '
        f'baseline={args.baseline} reps={args.reps} num_tests={args.num_tests}',
        once_in_node=True)

    cap = args.num_max_tokens_per_rank or max(batches)
    if cap < max(batches):
        raise ValueError(f'num_max_tokens_per_rank={cap} < max batch {max(batches)}')
    for n in batches:
        _run_one_config(args, n, cap, args.hidden, args.intermediate_hidden,
                        args.num_experts, args.num_topk, num_ranks, rank_idx, group,
                        activation_clamp=args.activation_clamp, fast_math=bool(args.fast_math))
    dist.barrier()
    dist.destroy_process_group()


if __name__ == '__main__':
    p = argparse.ArgumentParser(description='SM90 MegaMoE fp8/mxfp4/nvfp4 paired benchmark')
    p.add_argument('--num-processes', type=int, default=8)
    p.add_argument('--local-rank-idx', type=int, default=None)
    p.add_argument('--arms', nargs='+', default=['fp8', 'mxfp4'], choices=['fp8', 'mxfp4', 'nvfp4'])
    p.add_argument('--baseline', default='fp8')
    # Extra MXFP4 arms that differ only in schedule, e.g. --mxfp4-scheds 24,3,48 24,7,48
    p.add_argument('--mxfp4-scheds', nargs='*', default=[],
                   metavar='BLOCK_M,STAGES,EPW[,SINGLE_DISPATCH[,RS]]')
    p.add_argument('--batches', type=int, nargs='+', default=None)
    # Defaults are the ONLY shape the mxfp4/nvfp4 kernels accept.
    p.add_argument('--hidden', type=int, default=6144)
    p.add_argument('--intermediate-hidden', type=int, default=2048)
    p.add_argument('--num-experts', type=int, default=384)
    p.add_argument('--num-topk', type=int, default=8)
    p.add_argument('--activation-clamp', type=float, default=10.0)
    p.add_argument('--masked-ratio', type=float, default=0.0)
    p.add_argument('--fast-math', type=int, default=1)
    p.add_argument('--num-tests', type=int, default=20)
    p.add_argument('--reps', type=int, default=3)
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--verbose', action='store_true')
    p.add_argument('--num-max-tokens-per-rank', type=int, default=None)
    args = p.parse_args()

    if args.local_rank_idx is not None:
        test(args.local_rank_idx, args.num_processes, args)
    else:
        torch.multiprocessing.spawn(test, args=(args.num_processes, args), nprocs=args.num_processes)
