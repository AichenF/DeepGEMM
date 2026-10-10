"""Dynamic EPLB gate for the SM90 MXFP4 MegaMoE kernel.

Three checks per batch, all under deliberately skewed routing:
  1. the device planner agrees with the Python reference planner;
  2. the remapped routing table matches the plan (counts per physical slot);
  3. the kernel output on physical slots matches the logical-expert reference.
Then a paired benchmark of the same routing with EPLB on and off.

Usage: python tests/test_mxfp4_mega_moe_sm90_eplb.py --num-processes 4
"""
import argparse
import os
import sys
import time

import torch
import torch.distributed as dist

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

import deep_gemm
from deep_gemm.mega.eplb_planner import plan_reference
from deep_gemm.quantization_mxfp4 import dequantize_mxfp4_to_fp32, quantize_to_mxfp4
from deep_gemm.testing import bench_kineto, get_arch_major
from deep_gemm.utils import per_token_cast_to_fp8
from deep_gemm.utils.dist import init_dist


def _silu(x):
    return x * torch.sigmoid(x)


def _skewed_topk(m_tokens, num_experts, num_topk, hot_experts, hot_share, generator):
    """Top-k routing where a share of all slots lands on a few hot experts."""
    scores = torch.rand((m_tokens, num_experts), generator=generator, device='cuda')
    hot = torch.tensor(hot_experts, device='cuda')
    mask = torch.rand((m_tokens, len(hot_experts)), generator=generator, device='cuda') < hot_share
    boost = torch.zeros_like(scores)
    boost[:, hot] = mask.float() * 10.0
    topk_weights, topk_idx = torch.topk(scores + boost, num_topk, dim=-1)
    topk_weights = torch.softmax(topk_weights, dim=-1)
    return topk_idx, topk_weights


def _all_gather_counts(topk_idx, num_experts, group):
    counts = torch.bincount(topk_idx.flatten(), minlength=num_experts).to(torch.int64)
    gathered = [torch.empty_like(counts) for _ in range(dist.get_world_size(group))]
    dist.all_gather(gathered, counts, group=group)
    return [g.cpu().tolist() for g in gathered]


def _reference_output(x_fp8, x_sf, topk_idx, topk_weights, l1_dequant, l2_dequant,
                      l1_gs, l2_gs, rank_idx, num_local_experts, clamp, group):
    m_tokens, hidden = x_fp8.shape
    x_ref = (x_fp8.float().view(m_tokens, hidden // 128, 128) * x_sf.float().unsqueeze(2)).view(m_tokens, hidden)
    y_ref = torch.zeros((m_tokens, hidden), device='cuda', dtype=torch.float32)
    topk_idx_cpu = topk_idx.cpu()
    topk_w_cpu = topk_weights.cpu()
    # Group by local expert so each expert is one GEMM.
    for local in range(num_local_experts):
        expert = rank_idx * num_local_experts + local
        hits = (topk_idx_cpu == expert).nonzero()
        if hits.numel() == 0:
            continue
        tok = hits[:, 0].cuda()
        w = topk_w_cpu[hits[:, 0], hits[:, 1]].cuda().float()
        l1_out = x_ref[tok] @ l1_dequant[local].float().t() * l1_gs[local]
        inter = l1_out.shape[1] // 2
        gate = l1_out[:, :inter].clamp(max=clamp)
        up = l1_out[:, inter:].clamp(min=-clamp, max=clamp)
        h = _silu(gate) * up * w[:, None]
        y_ref.index_add_(0, tok, (h @ l2_dequant[local].float().t()) * l2_gs[local])
    dist.all_reduce(y_ref, op=dist.ReduceOp.SUM, group=group)
    return y_ref


def _check_plan(eplb, counts, block_m, prev_slot_expert, rank_idx):
    ref = plan_reference(counts, eplb.num_helper_slots, block_m,
                         prev_slot_expert=prev_slot_expert, open_gate_blocks=eplb.open_gate_blocks)
    dev_slots = eplb.slot_expert.cpu().tolist()
    assert dev_slots == ref.slot_expert, f'rank {rank_idx}: slot table differs\n dev={dev_slots}\n ref={ref.slot_expert}'
    dev_copies = eplb.copies.cpu().tolist()
    for e in range(eplb.num_experts):
        got = []
        for rank, cum in dev_copies[e]:
            if rank < 0:
                break
            got.append((rank, cum))
        assert got == ref.copies[e], f'rank {rank_idx}: copies for expert {e} differ: dev={got} ref={ref.copies[e]}'
    return ref


def _check_remap(eplb, ref, counts, logical_rows, physical_rows, rank_idx):
    """Every physical slot refers to a slot that holds the expert, and the
    per-slot counts equal the reference remap exactly."""
    slots = eplb.slots_per_rank
    H = eplb.home_experts
    seen = {}
    for row_l, row_p in zip(logical_rows, physical_rows):
        for e, p in zip(row_l, row_p):
            dest, local = divmod(p, slots)
            if local < H:
                assert dest * H + local == e, f'rank {rank_idx}: home slot mismatch {p} for expert {e}'
            else:
                assert ref.slot_expert[dest][local - H] == e, \
                    f'rank {rank_idx}: helper slot {p} does not hold expert {e}'
            seen[(dest, e)] = seen.get((dest, e), 0) + 1
    return seen


def _run_case(args, m_tokens, rank_idx, group, eplb_state, check=True):
    num_ranks = dist.get_world_size(group)
    gen = torch.Generator(device='cuda')
    gen.manual_seed(args.seed + m_tokens)
    hidden, ih = args.hidden, args.intermediate_hidden
    E, K, S = args.num_experts, args.num_topk, args.num_helper_slots
    H = E // num_ranks

    eplb, l1_dequant, l2_dequant, l1_gs, l2_gs = eplb_state
    hot = [h % E for h in args.hot_experts]
    topk_idx, topk_weights = _skewed_topk(m_tokens, E, K, hot, args.hot_share, gen)
    x_bf = torch.randn((m_tokens, hidden), dtype=torch.bfloat16, device='cuda', generator=gen)
    x_fp8, x_sf = per_token_cast_to_fp8(x_bf, use_ue8m0=False, gran_k=128)

    if not check:
        return topk_idx, topk_weights, x_fp8, x_sf
    counts = _all_gather_counts(topk_idx, E, group)
    prev_slot_expert = eplb.slot_expert.cpu().tolist()

    # Stage by stage so each one can be checked.
    b = eplb.symm
    b.x[:m_tokens].copy_(x_fp8)
    b.x_sf[:m_tokens].copy_(x_sf)
    b.topk_idx[:m_tokens].copy_(topk_idx.to(torch.int64))
    b.topk_weights[:m_tokens].copy_(topk_weights.to(torch.float32))
    block_m = eplb.plan(m_tokens)
    torch.cuda.synchronize()
    ref = _check_plan(eplb, counts, block_m, prev_slot_expert, rank_idx)
    copies = int(eplb.copy_count.item())
    eplb.copy_weights()
    eplb.remap(m_tokens, block_m)
    torch.cuda.synchronize()
    physical = b.topk_idx[:m_tokens].cpu().tolist()
    seen = _check_remap(eplb, ref, counts, topk_idx.cpu().tolist(), physical, rank_idx)
    # Per-(rank, expert) counts across all ranks must equal the plan's blocks.
    local_counts = torch.zeros((num_ranks, E), dtype=torch.int64, device='cuda')
    for (dest, e), n in seen.items():
        local_counts[dest, e] += n
    dist.all_reduce(local_counts, group=group)
    local_counts = local_counts.cpu().tolist()
    for e in range(E):
        for r in range(num_ranks):
            n, blocks = local_counts[r][e], ref.blocks[e][r]
            assert n <= blocks * block_m and (blocks == 0) == (n == 0) and n > (blocks - 1) * block_m, \
                f'rank {rank_idx}: expert {e} on rank {r} got {n} tokens for {blocks} blocks of {block_m}'

    y = torch.empty((m_tokens, hidden), dtype=torch.bfloat16, device='cuda')
    deep_gemm.mxfp4_mega_moe(
        y, eplb.l1_weights, eplb.l2_weights, b,
        cumulative_local_expert_recv_stats=eplb.cumulative_slot_recv_stats,
        l1_global_scales=eplb.l1_global_scales, l2_global_scales=eplb.l2_global_scales,
        activation_clamp=args.activation_clamp, fast_math=bool(args.fast_math))
    torch.cuda.synchronize()
    dist.barrier(group=group)

    y_ref = _reference_output(x_fp8, x_sf, topk_idx, topk_weights, l1_dequant, l2_dequant,
                              l1_gs, l2_gs, rank_idx, H, args.activation_clamp, group)
    finite = torch.isfinite(y).all().item()
    cosine = torch.nn.functional.cosine_similarity(y.float(), y_ref, dim=-1)
    cos_min, cos_mean = cosine.min().item(), cosine.mean().item()
    norm_ratio = (torch.linalg.vector_norm(y.float()) / torch.linalg.vector_norm(y_ref).clamp_min(1e-30)).item()
    baseline = max(sum(-(-sum(counts[r][e] for r in range(num_ranks)) // block_m)
                       for e in range(rr * H, (rr + 1) * H)) for rr in range(num_ranks))
    if rank_idx == 0:
        replicas = sum(1 for row in ref.slot_expert for e in row if e >= 0)
        print(f'M={m_tokens} block_m={block_m} replicas={replicas} copies(rank0)={copies} '
              f'max_load {baseline} -> {max(ref.loads)} blocks | '
              f'cos min={cos_min:.4f} mean={cos_mean:.4f} norm={norm_ratio:.4f} finite={finite}', flush=True)
    assert finite, 'non-finite output'
    assert cos_mean >= args.cosine_mean_threshold, f'cosine mean {cos_mean}'
    assert cos_min >= args.cosine_min_threshold, f'cosine min {cos_min}'
    assert args.norm_ratio_min <= norm_ratio <= args.norm_ratio_max, f'norm ratio {norm_ratio}'
    return topk_idx, topk_weights, x_fp8, x_sf


def _bench(args, m_tokens, rank_idx, group, eplb, routing, baseline_buffer, baseline_weights):
    topk_idx, topk_weights, x_fp8, x_sf = routing
    hidden = args.hidden

    def run_eplb():
        return eplb.forward(x_fp8, x_sf, topk_idx, topk_weights,
                            activation_clamp=args.activation_clamp, fast_math=bool(args.fast_math))

    def run_fixed():
        b = baseline_buffer
        n = m_tokens
        b.x[:n].copy_(x_fp8)
        b.x_sf[:n].copy_(x_sf)
        b.topk_idx[:n].copy_(topk_idx.to(torch.int64))
        b.topk_weights[:n].copy_(topk_weights.to(torch.float32))
        y = torch.empty((n, hidden), dtype=torch.bfloat16, device='cuda')
        deep_gemm.mxfp4_mega_moe(y, baseline_weights[0], baseline_weights[1], b,
                                 activation_clamp=args.activation_clamp, fast_math=bool(args.fast_math))
        return y

    torch.cuda.synchronize()
    dist.barrier(group=group)
    # Cold copy: every helper slot changes, so this is the full weight transfer
    # the steady-state loop below never pays again. The first pass only builds
    # the kernel; the second one is timed.
    b = eplb.symm
    for _ in range(2):
        eplb.slot_expert.fill_(-1)
        b.topk_idx[:m_tokens].copy_(topk_idx.to(torch.int64))
        eplb.plan(m_tokens)
        torch.cuda.synchronize()
        dist.barrier(group=group)
        copies = int(eplb.copy_count.item())
        start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        start.record()
        eplb.copy_weights()
        end.record()
        torch.cuda.synchronize()
    cold_us = torch.tensor([start.elapsed_time(end) * 1e3], device='cuda')
    dist.all_reduce(cold_us, op=dist.ReduceOp.MAX, group=group)
    bytes_per_slot = sum(spec[1] for spec in eplb.arena.plane_specs(eplb._plane_names))
    if rank_idx == 0:
        print(f'[cold copy M={m_tokens}] {copies} slots x {bytes_per_slot / 2**20:.1f} MiB on rank 0, '
              f'rank-max {cold_us.item():.1f}us', flush=True)

    kernels = ('eplb_plan_kernel', 'eplb_copy_weights_kernel', 'eplb_remap_kernel', 'sm90_mxfp4_mega_moe')
    # The fixed arm is a separate kernel instantiation. Build it on every rank
    # before the paired loop: the planner waits on its peers in-kernel and
    # would otherwise time out behind another rank's JIT compile.
    run_fixed()
    torch.cuda.synchronize()
    dist.barrier(group=group)
    for _ in range(3):
        run_eplb()
        run_fixed()
    torch.cuda.synchronize()
    dist.barrier(group=group)
    # Every rank must issue the same number of plans, so the debug profile runs
    # everywhere and only rank 0 prints it.
    debug = bool(os.environ.get('DG_EPLB_DEBUG'))
    quiet = not (debug and rank_idx == 0)
    if debug:
        with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CUDA]) as prof:
            run_eplb()
            torch.cuda.synchronize()
        if rank_idx == 0:
            print(prof.key_averages().table(sort_by='cuda_time_total', row_limit=25, max_name_column_width=120), flush=True)
    # The planner and the kernel both wait on their peers in-kernel with a
    # 30 s budget; a profiler session can leave the hosts skewed by more than
    # that, so resynchronize before every segment that launches them.
    def resync():
        torch.cuda.synchronize()
        dist.barrier(group=group)

    resync()
    t_eplb = bench_kineto(run_eplb, kernels, barrier=dist.barrier, num_tests=args.bench_tests,
                          suppress_kineto_output=quiet)
    resync()
    t_fixed = bench_kineto(run_fixed, 'sm90_mxfp4_mega_moe', barrier=dist.barrier, num_tests=args.bench_tests,
                           suppress_kineto_output=quiet)
    resync()
    t_eplb = [t * 1e6 for t in t_eplb]
    t_fixed_us = t_fixed * 1e6
    # Rank-maximum latency is what the layer waits for.
    local = torch.tensor([sum(t_eplb), t_eplb[3], t_fixed_us] + t_eplb[:3], device='cuda')
    dist.all_reduce(local, op=dist.ReduceOp.MAX, group=group)
    total, kernel_only, fixed, plan_us, copy_us, remap_us = local.tolist()
    if rank_idx == 0:
        ratio = lambda a, b: f'{a / b:.3f}x' if b > 0 else 'n/a'
        print(f'[bench M={m_tokens}] fixed={fixed:.1f}us  eplb total={total:.1f}us '
              f'(kernel {kernel_only:.1f} + plan {plan_us:.1f} + copy {copy_us:.1f} + remap {remap_us:.1f})  '
              f'speedup total={ratio(fixed, total)} kernel-only={ratio(fixed, kernel_only)}', flush=True)


def _worker(local_rank, num_local_ranks, args):
    rank_idx, _, group = init_dist(local_rank, num_local_ranks)
    try:
        if get_arch_major() != 9:
            if rank_idx == 0:
                print(f'[SKIP] requires SM90, got SM{get_arch_major()}0', flush=True)
            return
        num_ranks = dist.get_world_size(group)
        E, H = args.num_experts, args.num_experts // num_ranks
        torch.manual_seed(args.seed + rank_idx)
        l1_bf = torch.randn((H, 2 * args.intermediate_hidden, args.hidden), dtype=torch.bfloat16, device='cuda') * args.weight_scale
        l2_bf = torch.randn((H, args.hidden, args.intermediate_hidden), dtype=torch.bfloat16, device='cuda') * args.weight_scale
        l1_packed, l1_scale = quantize_to_mxfp4(l1_bf, group_size=32)
        l2_packed, l2_scale = quantize_to_mxfp4(l2_bf, group_size=32)
        l1_dequant = dequantize_mxfp4_to_fp32(l1_packed, l1_scale, group_size=32)
        l2_dequant = dequantize_mxfp4_to_fp32(l2_packed, l2_scale, group_size=32)
        t1, t2 = deep_gemm.transform_mxfp4_weights_for_mega_moe_sm90((l1_packed, l1_scale), (l2_packed, l2_scale))
        l1_gs = torch.linspace(0.73, 1.37, H, dtype=torch.float32, device='cuda')
        l2_gs = torch.linspace(1.31, 0.67, H, dtype=torch.float32, device='cuda')

        eplb = deep_gemm.DynamicEPLB(group, E, args.num_helper_slots, args.num_max_tokens_per_rank,
                                     args.num_topk, args.hidden, args.intermediate_hidden,
                                     open_gate_blocks=args.open_gate_blocks)
        eplb.set_home_weights(t1, t2, l1_gs, l2_gs)
        state = (eplb, l1_dequant, l2_dequant, l1_gs.cpu().tolist(), l2_gs.cpu().tolist())

        routings = {}
        for m_tokens in args.batches:
            routings[m_tokens] = _run_case(args, m_tokens, rank_idx, group, state,
                                           check=bool(args.correctness))
            if args.correctness:
                # Second pass on the same routing exercises the slot cache.
                _run_case(args, m_tokens, rank_idx, group, state)
        if rank_idx == 0 and args.correctness:
            print('correctness: PASS', flush=True)

        if args.bench:
            baseline_buffer = deep_gemm.get_symm_buffer_for_mega_moe(
                group, E, args.num_max_tokens_per_rank, args.num_topk, args.hidden, args.intermediate_hidden,
                use_fp8_dispatch=True, activation='swiglu')
            for m_tokens in args.batches:
                _bench(args, m_tokens, rank_idx, group, eplb, routings[m_tokens], baseline_buffer, (t1, t2))
            baseline_buffer.destroy()
        eplb.destroy()
    finally:
        dist.destroy_process_group()


def _parse_args():
    p = argparse.ArgumentParser(description='SM90 MXFP4 MegaMoE dynamic EPLB gate')
    p.add_argument('--batches', nargs='+', type=int, default=[64, 256, 1024])
    p.add_argument('--hidden', type=int, default=6144)
    p.add_argument('--intermediate-hidden', type=int, default=2048)
    p.add_argument('--num-experts', type=int, default=384)
    p.add_argument('--num-topk', type=int, default=8)
    p.add_argument('--num-helper-slots', type=int, default=4)
    p.add_argument('--open-gate-blocks', type=int, default=0)
    p.add_argument('--hot-experts', nargs='+', type=int, default=[3, 7, 11])
    p.add_argument('--hot-share', type=float, default=0.5)
    p.add_argument('--num-max-tokens-per-rank', type=int, default=4096)
    p.add_argument('--num-processes', type=int, default=8)
    p.add_argument('--activation-clamp', type=float, default=10.0)
    p.add_argument('--fast-math', type=int, default=1)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--weight-scale', type=float, default=0.05)
    p.add_argument('--cosine-mean-threshold', type=float, default=0.995)
    p.add_argument('--cosine-min-threshold', type=float, default=0.99)
    p.add_argument('--norm-ratio-min', type=float, default=0.99)
    p.add_argument('--norm-ratio-max', type=float, default=1.01)
    p.add_argument('--bench', type=int, default=1)
    p.add_argument('--correctness', type=int, default=1)
    p.add_argument('--bench-tests', type=int, default=20)
    return p.parse_args()


if __name__ == '__main__':
    parsed = _parse_args()
    torch.multiprocessing.spawn(_worker, args=(parsed.num_processes, parsed), nprocs=parsed.num_processes, join=True)
