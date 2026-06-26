"""SM90 (Hopper) NVFP4 MegaMoE benchmark / NCU-profile harness.

This harness drives the packed NVFP4 MegaMoE path only.

In normal (non-NCU) mode it sweeps a list of ``num_tokens`` values (default:
1, 2, 4, 8, 16, 32) and reports per-call kernel time via the same
``bench_kineto`` helper used by the SM100 perf test, plus a rough TFLOPS /
HBM GB/s figure useful for tracking optimisation deltas.
"""

import argparse
import os
import random
import sys
import torch
import torch.distributed as dist

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

import deep_gemm
from deep_gemm.quantization_nvfp4 import quantize_to_nvfp4
from deep_gemm.utils import per_token_cast_to_fp8
from deep_gemm.utils.dist import dist_print, init_dist, uneven_all_gather
from deep_gemm.testing import bench_kineto, get_arch_major


def _nvfp4_bn256_fused_m(num_tokens: int) -> bool:
    return num_tokens <= int(os.environ.get("DG_SM90_NVFP4_BN256_FUSED_MAX_M", "511"))


def _run_one_config(args, num_tokens, num_max_tokens_per_rank,
                    hidden, intermediate_hidden,
                    num_experts, num_topk, num_ranks, rank_idx, group,
                    activation_clamp, fast_math,
                    print_perf=True):
    num_experts_per_rank = num_experts // num_ranks
    assert num_tokens <= num_max_tokens_per_rank

    # Symmetric buffer (one per config: cheaper to recreate than to keep max-size)
    buffer = deep_gemm.get_symm_buffer_for_mega_moe(
        group, num_experts,
        num_max_tokens_per_rank, num_topk,
        hidden, intermediate_hidden,
    )

    # Inputs (bf16, then quantised)
    x_bf = torch.randn((num_tokens, hidden), dtype=torch.bfloat16, device='cuda')
    l1_bf = torch.randn(
        (num_experts_per_rank, intermediate_hidden * 2, hidden),
        dtype=torch.bfloat16, device='cuda') * 0.05
    l2_bf = torch.randn(
        (num_experts_per_rank, hidden, intermediate_hidden),
        dtype=torch.bfloat16, device='cuda') * 0.05
    scores = torch.randn((num_tokens, num_experts), dtype=torch.float, device='cuda')
    topk_w, topk_idx = torch.topk(scores, num_topk, dim=-1, largest=True, sorted=False)
    if args.masked_ratio > 0:
        rand_mask = torch.rand_like(topk_idx, dtype=torch.float)
        topk_idx.masked_fill_(rand_mask < args.masked_ratio, -1)
        topk_w.masked_fill_(topk_idx < 0, 0)

    x_fp8, x_sf = per_token_cast_to_fp8(x_bf, use_ue8m0=False, gran_k=128,
                                        use_packed_ue8m0=False)
    # NVFP4: quantize real BF16 weights directly to packed NVFP4 + per-16 UE4M3 SF.
    # Do not route through block-FP8 weight quantization here; that produces
    # byte-domain FP8 values and requires a separate block SF that this kernel
    # ABI does not consume.
    l1_packed, l1_scale = quantize_to_nvfp4(l1_bf, group_size=16)
    l2_packed, l2_scale = quantize_to_nvfp4(l2_bf, group_size=16)
    # The SM90 NVFP4 wrapper uses the BN256 fused path for the compact bucket up
    # to the configurable small/mid-M cutoff; larger M keeps the BN128 split path.
    nvfp4_use_bn256 = _nvfp4_bn256_fused_m(num_tokens)
    nvfp4_default_block_n = 256 if nvfp4_use_bn256 else 128
    nvfp4_block_n = int(os.environ.get("DG_SM90_NVFP4_BLOCK_N", nvfp4_default_block_n))
    nvfp4_fused_b_scale_env = os.environ.get("DG_SM90_NVFP4_FUSED_B_SCALE")
    nvfp4_fused_b_scale = None if nvfp4_fused_b_scale_env is not None else (True if nvfp4_use_bn256 else None)
    transformed_l1, transformed_l2 = deep_gemm.transform_nvfp4_weights_for_mega_moe_sm90(
        (l1_packed, l1_scale), (l2_packed, l2_scale),
        block_n=nvfp4_block_n, fused_b_scale=nvfp4_fused_b_scale,
    )
    kernel_name = 'sm90_nvfp4_mega_moe'

    phase_profile_enabled = os.environ.get('DG_SM90_MOE_PHASE_PROFILE', '0') != '0'
    phase_profile_ints = 96 if phase_profile_enabled else 0
    cum_stats = torch.zeros(num_experts_per_rank + phase_profile_ints, dtype=torch.int, device='cuda')

    # Stage inputs once; bench-loop re-copies them each call (bench helper expects
    # an idempotent ``fn``).
    def run():
        buffer.x[:num_tokens].copy_(x_fp8)
        buffer.x_sf[:num_tokens].copy_(x_sf)
        buffer.topk_idx[:num_tokens].copy_(topk_idx)
        buffer.topk_weights[:num_tokens].copy_(topk_w)
        y = torch.empty((num_tokens, hidden), dtype=torch.bfloat16, device='cuda')
        deep_gemm.nvfp4_mega_moe(
            y, transformed_l1, transformed_l2, buffer,
            cumulative_local_expert_recv_stats=cum_stats,
            recipe=(128, 128, 128),
            activation='swiglu',
            activation_clamp=activation_clamp,
            fast_math=fast_math,
        )
        return y

    if args.ncu_profile_only:
        dist_print(f'[NCU] tokens={num_tokens} hidden={hidden} ih={intermediate_hidden}',
                   once_in_node=True)
        run()
        torch.cuda.synchronize()
        dist.barrier()
        buffer.destroy()
        return

    # Warm up + benchmark
    run()
    dist.barrier()
    if phase_profile_enabled:
        cum_stats.zero_()
        torch.cuda.synchronize()
        dist.barrier()
    # NSYS MULTI-ITER (aichenf): N timed iters with barrier+sleep between them.
    # bench_kineto returns 1 under DG_USE_NVIDIA_TOOLS=1, but this loop puts
    # multiple mega_moe instances on the nsys timeline so we can measure variance.
    import os as _os
    _nsys_iters = int(_os.environ.get('NSYS_ITERS', '0'))
    if _nsys_iters > 0:
        for _it in range(_nsys_iters):
            torch.cuda.synchronize()
            dist.barrier()
            torch.cuda._sleep(int(2e7))  # 10ms gap between iters
            dist.barrier()
            run()
        torch.cuda.synchronize()
        dist.barrier()
    show_kineto = os.environ.get('DG_SHOW_KINETO', '0') != '0'
    split_env = os.environ.get('DG_SM90_MOE_SPLIT_L1_L2')
    if split_env is None:
        fused_bn256_default = nvfp4_block_n == 256 and _nvfp4_bn256_fused_m(num_tokens)
        split_l1_l2 = not fused_bn256_default
    else:
        split_l1_l2 = split_env != '0'
    # The profiler table exposes the generated CUDA function name, not the
    # JIT build name, so split L1/L2 launches cannot be matched separately
    # by "_l1" / "_l2" suffix. With one substring and
    # with_multiple_kernels=True, bench_kineto returns the per-kernel
    # average across both split launches; multiply by two to estimate one
    # end-to-end MoE call.
    t_nvfp4 = bench_kineto(run, kernel_name,
                           barrier=lambda: dist.barrier(),
                           num_tests=args.num_tests,
                           suppress_kineto_output=not show_kineto,
                           with_multiple_kernels=split_l1_l2)
    if split_l1_l2:
        t_nvfp4 *= 2

    t_rank = torch.tensor([t_nvfp4], dtype=torch.float64, device="cuda")
    t_rank_max = t_rank.clone()
    t_rank_min = t_rank.clone()
    t_rank_sum = t_rank.clone()
    dist.all_reduce(t_rank_max, op=dist.ReduceOp.MAX)
    dist.all_reduce(t_rank_min, op=dist.ReduceOp.MIN)
    dist.all_reduce(t_rank_sum, op=dist.ReduceOp.SUM)
    t_nvfp4_rank_max = float(t_rank_max.item())
    t_nvfp4_rank_min = float(t_rank_min.item())
    t_nvfp4_rank_mean = float(t_rank_sum.item()) / num_ranks

    # Count tokens that landed on this rank for stats
    gathered_topk_idx = uneven_all_gather(topk_idx, group=group)
    gathered_topk_idx[(gathered_topk_idx < rank_idx * num_experts_per_rank) |
                      (gathered_topk_idx >= (rank_idx + 1) * num_experts_per_rank)] = -1
    num_recv_tokens = (gathered_topk_idx != -1).sum().item()

    safe_div = lambda a, b: float('nan') if b == 0 else a / b
    tflops_nvfp4 = safe_div(2 * num_recv_tokens * (hidden * intermediate_hidden * 3) / 1e12, t_nvfp4)
    tflops = tflops_nvfp4  # legacy alias for the rest of the print logic
    num_touched_experts = max(0, torch.unique(gathered_topk_idx.flatten()).numel() - 1)
    # NVFP4 weights = 0.5 byte/value plus 1 UE4M3 scale byte per 16 K values.
    num_hbm_bytes = (
        num_touched_experts * intermediate_hidden * 2 * (hidden // 2 + hidden // 16) +
        num_touched_experts * hidden * (intermediate_hidden // 2 + intermediate_hidden // 16) +
        num_recv_tokens * hidden +                                  # L1 acts read
        num_recv_tokens * intermediate_hidden +                     # L1 out write
        num_recv_tokens * intermediate_hidden +                     # L2 acts read
        num_recv_tokens * hidden * 2                                # L2 out write
    )
    hbm_gbs = safe_div(num_hbm_bytes / 1e9, t_nvfp4)

    if print_perf:
        dist_print(
            f' tokens={num_tokens:4d}  recv={num_recv_tokens:5d}  experts={num_touched_experts:4d}  '
            f'nvfp4={t_nvfp4 * 1e6:7.1f}us '
            f'mean_rank={t_nvfp4_rank_mean * 1e6:7.1f}us max_rank={t_nvfp4_rank_max * 1e6:7.1f}us '
            f'({tflops_nvfp4:5.1f}TF, {hbm_gbs:4.0f}GB/s)  (rank{rank_idx})',
            once_in_node=True,
        )
        if phase_profile_enabled:
            torch.cuda.synchronize()
            names = [
                'dispatch_total', 'dispatch_pull', 'math_loop', 'combine_barrier',
                'combine_reduce', 'gemm_core', 'l1_epilogue', 'l2_epilogue',
                'loader_dequant', 'math_dequant_wait', 'l1_tma_wait',
                'l1_ready_notify', 'l2_ready_wait', 'l2_scatter',
            ]
            num_phase_metrics = len(names)
            profile = cum_stats[
                num_experts_per_rank:num_experts_per_rank + 3 * num_phase_metrics * 2
            ].view(torch.int64).cpu().tolist()
            for i, name in enumerate(names):
                total = profile[i]
                max_v = profile[num_phase_metrics + i]
                count = profile[2 * num_phase_metrics + i]
                avg = float(total) / count if count else 0.0
                dist_print(
                    f'   phase {name:16s} avg={avg:10.0f} max={max_v:10d} count={count}',
                    once_in_node=True,
                )

    dist.barrier()
    buffer.destroy()


def test(local_rank: int, num_local_ranks: int, args: argparse.Namespace):
    rank_idx, num_ranks, group = init_dist(local_rank, num_local_ranks)
    forced_num_sms = int(os.environ.get('DG_SM90_MOE_SET_NUM_SMS', '0'))
    if forced_num_sms > 0:
        deep_gemm.set_num_sms(forced_num_sms)
    torch.manual_seed(rank_idx)
    random.seed(rank_idx)

    if get_arch_major() != 9:
        dist_print(f'[SKIP] requires SM90, got SM{get_arch_major()}0', once_in_node=True)
        dist.destroy_process_group()
        return

    if args.batches is None:
        batches = [1, 2, 4, 8, 16, 32]
    else:
        batches = args.batches

    dist_print(
        f'SM90 MegaMoE bench: ranks={num_ranks} hidden={args.hidden} '
        f'ih={args.intermediate_hidden} experts={args.num_experts} topk={args.num_topk} '
        f'masked_ratio={args.masked_ratio} fast_math={bool(args.fast_math)}',
        once_in_node=True,
    )

    # In NCU mode we run only one batch (the first one in `batches`) so that
    # ncu's `--launch-count 1` is unambiguous.
    if args.ncu_profile_only:
        batches = batches[:1]

    num_max_tokens_per_rank = max(batches)
    for num_tokens in batches:
        _run_one_config(
            args, num_tokens, num_max_tokens_per_rank,
            args.hidden, args.intermediate_hidden,
            args.num_experts, args.num_topk,
            num_ranks, rank_idx, group,
            activation_clamp=args.activation_clamp,
            fast_math=bool(args.fast_math),
        )

    dist.barrier()
    dist.destroy_process_group()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='SM90 MegaMoE benchmark')

    parser.add_argument('--ncu-profile-only', action='store_true')
    parser.add_argument('--num-processes', type=int, default=8)
    parser.add_argument('--local-rank-idx', type=int, default=None)

    parser.add_argument('--batches', type=int, nargs='+', default=None,
                        help='List of num_tokens to sweep (default: 1 2 4 8 16 32)')
    parser.add_argument('--hidden', type=int, default=7168)
    parser.add_argument('--intermediate-hidden', type=int, default=2048)
    parser.add_argument('--num-experts', type=int, default=256)
    parser.add_argument('--num-topk', type=int, default=8)
    parser.add_argument('--activation-clamp', type=float, default=10.0)
    parser.add_argument('--masked-ratio', type=float, default=0.0)
    parser.add_argument('--fast-math', type=int, default=1)
    parser.add_argument('--num-tests', type=int, default=20)
    args = parser.parse_args()

    if args.local_rank_idx is not None:
        test(args.local_rank_idx, args.num_processes, args)
    else:
        np = args.num_processes
        torch.multiprocessing.spawn(test, args=(np, args), nprocs=np)
