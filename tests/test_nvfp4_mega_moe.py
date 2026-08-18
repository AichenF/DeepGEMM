import argparse
import os
import random
import torch
import torch.distributed as dist
from typing import Optional, Tuple

import deep_gemm
from deep_gemm.utils import cast_back_from_fp4
from deep_gemm.utils.dist import dist_print, init_dist
from deep_gemm.testing import bench_kineto, calc_diff


def _quantize_to_fp4_e2m1_rne(x: torch.Tensor) -> torch.Tensor:
    # Match `cvt.rn.satfinite.e2m1x2.f32` on the E2M1 grid.
    ax = x.abs()
    code = torch.zeros_like(x, dtype=torch.uint8)
    for boundary in (0.25, 1.25, 2.5, 5.0):
        code += (ax > boundary).to(torch.uint8)
    for boundary in (0.75, 1.75, 3.5):
        code += (ax >= boundary).to(torch.uint8)
    code |= (((x < 0) & (code != 0)).to(torch.uint8) << 3)
    return code.view(torch.int8)


def _cast_to_nvfp4(x: torch.Tensor, global_scale: Optional[float] = None,
                    pack_sf: bool = False) -> Tuple[torch.Tensor, torch.Tensor]:
    gran_k = 16
    m, n = x.shape
    assert n % (gran_k * 4) == 0
    blocks = x.float().view(m, n // gran_k, gran_k)
    inv_global = 1.0 if global_scale is None else 1.0 / global_scale
    sf_e4m3 = (blocks.abs().amax(dim=2) / 6.0 * inv_global).clamp_min(2.0 ** -9).to(torch.float8_e4m3fn)
    sf = sf_e4m3.float()
    elem_scale = sf if global_scale is None else sf * global_scale
    codes = _quantize_to_fp4_e2m1_rne(blocks / elem_scale.unsqueeze(2)).view(m, n // 2, 2)
    packed = ((codes[:, :, 0] & 0x0F) | ((codes[:, :, 1] & 0x0F) << 4)).contiguous()
    return (packed, sf_e4m3.view(torch.uint8).contiguous().view(torch.int32)) if pack_sf else (packed, sf)


def _unpack_sf(packed_sf: torch.Tensor) -> torch.Tensor:
    return packed_sf.contiguous().view(torch.uint8).view(torch.float8_e4m3fn).float()


def _quantize_weights_to_nvfp4(bf16_weights: torch.Tensor) -> Tuple[Tuple[torch.Tensor, torch.Tensor], torch.Tensor]:
    # Cast per-expert weights into packed E2M1 with per-16-element E4M3 SF.
    # Returns `((packed_weights, mn_major_packed_sf), dequantized_weights)`
    num_groups, n, k = bf16_weights.shape
    w = torch.empty((num_groups, n, k // 2), device='cuda', dtype=torch.int8)
    w_sf = torch.empty((num_groups, n, k // 64), device='cuda', dtype=torch.int32)
    w_dq = torch.empty((num_groups, n, k), device='cuda', dtype=torch.float)
    for i in range(num_groups):
        w[i], w_sf[i] = _cast_to_nvfp4(bf16_weights[i], pack_sf=True)
        w_dq[i] = cast_back_from_fp4(w[i], _unpack_sf(w_sf[i]).view(n, k // 16), gran_k=16)
    # The kernel expects TMA-aligned MN-major SF
    w_sf = w_sf.transpose(-1, -2).contiguous().transpose(-1, -2)
    return (w, w_sf), w_dq


def _reference_expert_ffn(x_dq: torch.Tensor,
                          w1_dq: torch.Tensor, w2_dq: torch.Tensor,
                          topk_weights: torch.Tensor,
                          intermediate_hidden: int,
                          activation_clamp: float,
                          l1_alpha: torch.Tensor, l2_alpha: torch.Tensor,
                          use_l2_input_scale: bool = False) -> Tuple[torch.Tensor, float]:
    # Mirror the kernel semantics: FP32 GEMM -> BF16 round -> clamp -> SwiGLU (FP32)
    # -> top-k weight -> NVFP4 quantization -> FP32 GEMM -> BF16 partials.
    # Returns `(partial, a2_scale)` where `a2_scale` is the down-proj input scale used
    # for the intermediate NVFP4 requant (1.0 when disabled).
    h1 = x_dq @ w1_dq.t()
    gate = (h1[:, :intermediate_hidden] * l1_alpha[0]).bfloat16()
    up = (h1[:, intermediate_hidden:] * l1_alpha[1]).bfloat16()
    if activation_clamp is not None and activation_clamp != float('inf'):
        gate = gate.clamp_max(activation_clamp)
        up = up.clamp(-activation_clamp, activation_clamp)
    gate = gate.float()
    # Unweighted intermediate: the top-k routing weight is applied AFTER L2 (at combine),
    # not folded in before the NVFP4 requant. Folding it in before would couple the small
    # per-token weight into the E4M3 block-SF rounding (a systematic error); the kernel and
    # a standard/flashinfer MoE both apply it post-L2.
    act = (gate / (1.0 + torch.exp(-gate))) * up.float()

    # Per-expert down-proj input scale (modelopt's `input_scale`), calibrated from the
    # intermediate amax: normalizes the block SF into E4M3's well-represented range.
    if use_l2_input_scale:
        act_amax = act.abs().amax().item()
        a2_scale = act_amax / (448.0 * 6.0) if act_amax > 0 else 1.0
    else:
        a2_scale = None

    # NVFP4 quantization of the intermediate activations (as done by the L1 epilogue)
    act_packed, act_sf = _cast_to_nvfp4(act, global_scale=a2_scale)
    act_dq = cast_back_from_fp4(act_packed, act_sf, gran_k=16)
    if a2_scale is not None:
        act_dq = act_dq * a2_scale  # undo the block-SF normalization (folded into L2 alpha in-kernel)

    # L2 GEMM -> BF16 partial (fc2 * l2_alpha), THEN apply the top-k weight post-L2 in FP32
    # -> BF16 (mirrors the kernel's combine-write scaling); the combine sums in FP32.
    partial = ((act_dq @ w2_dq.t()) * l2_alpha).bfloat16()
    weighted = (partial.float() * topk_weights.unsqueeze(1)).bfloat16()
    return weighted, (a2_scale if a2_scale is not None else 1.0)


# noinspection PyShadowingNames
def test(local_rank: int, num_local_ranks: int, args: argparse.Namespace):
    rank_idx, num_ranks, group = init_dist(local_rank, num_local_ranks)
    torch.manual_seed(args.seed + rank_idx)
    random.seed(args.seed + rank_idx)

    # Settings (defaults follow `nvidia/GLM-5.2-NVFP4` MoE shapes)
    num_max_tokens_per_rank = args.num_max_tokens_per_rank
    num_tokens = args.num_max_tokens_per_rank - random.randint(0, args.num_max_removed_tokens) \
        if args.num_tokens < 0 else args.num_tokens
    if args.zero_token_rank == rank_idx:
        num_tokens = 0
    hidden, intermediate_hidden = args.hidden, args.intermediate_hidden
    num_experts, num_topk = args.num_experts, args.num_topk
    num_experts_per_rank = num_experts // num_ranks
    activation_clamp = args.activation_clamp
    assert num_tokens <= num_max_tokens_per_rank

    # Allocate symmetric memory
    num_shared_experts = args.num_shared_experts
    buffer = deep_gemm.get_symm_buffer_for_mega_moe(
        group, num_experts,
        num_max_tokens_per_rank, num_topk,
        hidden, intermediate_hidden,
        num_shared_experts=num_shared_experts,
        mma_type='fp4xfp4'
    )

    dist_print('Config:', once_in_node=True)
    dist_print(f' > MMA: fp4xfp4 (NVFP4 x NVFP4)', once_in_node=True)
    dist_print(f' > Tokens: {num_tokens}/{num_max_tokens_per_rank}', once_in_node=True)
    dist_print(f' > Hidden: {hidden}', once_in_node=True)
    dist_print(f' > Intermediate: {intermediate_hidden}', once_in_node=True)
    dist_print(f' > Experts: {num_topk}/{num_experts}', once_in_node=True)
    dist_print(f' > Alphas: {"per-expert" if args.per_expert_alphas else "none"}', once_in_node=True)
    dist_print(f' > Shared experts: {num_shared_experts} (BF16)', once_in_node=True)
    dist_print(f' > Buffer: {buffer.buffer.nbytes / 2 ** 30:.3f} GiB', once_in_node=True)
    dist_print(once_in_node=True)

    # Create inputs
    x = args.input_scale * torch.randn(
        (num_tokens, hidden), dtype=torch.bfloat16, device='cuda')
    l1_weights_bf16 = args.routed_weight_scale * torch.randn(
        (num_experts_per_rank, intermediate_hidden * 2, hidden), dtype=torch.bfloat16, device='cuda')
    l2_weights_bf16 = args.routed_weight_scale * torch.randn(
        (num_experts_per_rank, hidden, intermediate_hidden), dtype=torch.bfloat16, device='cuda')
    scores = torch.randn((num_tokens, num_experts), dtype=torch.float, device='cuda')
    topk_weights, topk_idx = torch.topk(scores, num_topk, dim=-1, largest=True, sorted=False)
    topk_weights = topk_weights.softmax(dim=-1)
    if args.masked_ratio > 0:
        rand_mask = torch.rand_like(topk_idx, dtype=torch.float)
        topk_idx.masked_fill_(rand_mask < args.masked_ratio, -1)
        topk_weights.masked_fill_(topk_idx < 0, 0)

    # Quantize inputs and weights into NVFP4
    x_packed, x_sf_packed = _cast_to_nvfp4(x, pack_sf=True)
    x_dq = cast_back_from_fp4(x_packed, _unpack_sf(x_sf_packed).view(num_tokens, hidden // 16), gran_k=16)
    l1_weights, l1_weights_dq = _quantize_weights_to_nvfp4(l1_weights_bf16)
    l2_weights, l2_weights_dq = _quantize_weights_to_nvfp4(l2_weights_bf16)
    transformed_l1_weights, transformed_l2_weights = (
        deep_gemm.transform_weights_for_mega_moe(l1_weights, l2_weights))

    # Optional per-local-expert scales (mirroring modelopt's `weight_scale_2`)
    if args.per_expert_alphas:
        l1_alphas = (0.5 + torch.rand((num_experts_per_rank, 2), dtype=torch.float, device='cuda'))
        l2_alphas = (0.5 + torch.rand((num_experts_per_rank, ), dtype=torch.float, device='cuda'))
    else:
        l1_alphas, l2_alphas = None, None

    # Optional BF16 shared expert (not quantized; folded on N for L1, K for L2)
    if num_shared_experts > 0:
        shared_intermediate_hidden = intermediate_hidden * num_shared_experts
        shared_l1_weights = args.shared_weight_scale * torch.randn(
            (shared_intermediate_hidden * 2, hidden), dtype=torch.bfloat16, device='cuda')
        shared_l2_weights = args.shared_weight_scale * torch.randn(
            (hidden, shared_intermediate_hidden), dtype=torch.bfloat16, device='cuda')
        transformed_shared_l1_weights, transformed_shared_l2_weights = (
            deep_gemm.transform_weights_for_mega_moe(shared_l1_weights, shared_l2_weights))
    else:
        shared_l1_weights, shared_l2_weights = None, None
        transformed_shared_l1_weights, transformed_shared_l2_weights = None, None

    cumulative_local_expert_recv_stats_fused = torch.randint(
        0, 100, (num_experts_per_rank, ), dtype=torch.int, device='cuda')
    cumulative_stats_initial = cumulative_local_expert_recv_stats_fused.clone()

    # Per-expert down-proj input scale (`a2_scale`) for the intermediate NVFP4 requant.
    # Computed by the reference from the intermediate amax, then fed to the kernel so both
    # use identical scales (mirroring a calibrated static `input_scale`).
    a2_scales_for_kernel = None

    # Run fused mega MoE
    # NOTES: copy inputs into the buffer before each call because debug mode zeros the entire buffer
    def run_fused():
        buffer.x[:num_tokens].copy_(x_packed.view(torch.uint8))
        buffer.x_sf[:num_tokens].copy_(x_sf_packed)
        buffer.topk_idx[:num_tokens].copy_(topk_idx)
        buffer.topk_weights[:num_tokens].copy_(topk_weights)

        y = torch.empty((num_tokens, hidden), dtype=torch.bfloat16, device='cuda')
        deep_gemm.fp4_fp4_mega_moe(
            y=y, l1_weights=transformed_l1_weights, l2_weights=transformed_l2_weights,
            sym_buffer=buffer,
            shared_l1_weights=transformed_shared_l1_weights,
            shared_l2_weights=transformed_shared_l2_weights,
            x_bf16=x if num_shared_experts > 0 else None,
            cumulative_local_expert_recv_stats=cumulative_local_expert_recv_stats_fused,
            activation_clamp=activation_clamp,
            fast_math=bool(args.fast_math),
            l1_alphas=l1_alphas, l2_alphas=l2_alphas,
            a2_scales=a2_scales_for_kernel,
            routed_scaling_factor=args.routed_scaling_factor)
        return y

    # Torch reference: gather all ranks' tokens, compute the local experts' contributions,
    # and reduce the (disjoint) BF16 partials across ranks in FP32 (exactly as combine does)
    def run_reference():
        nonlocal a2_scales_for_kernel
        # Gather token counts and inputs
        counts = torch.zeros(num_ranks, dtype=torch.long, device='cuda')
        counts[rank_idx] = num_tokens
        dist.all_reduce(counts, group=group)
        counts = counts.tolist()
        offsets = [0]
        for c in counts:
            offsets.append(offsets[-1] + c)
        num_global_tokens = offsets[-1]

        def gather(t, pad_rows):
            padded = torch.zeros((pad_rows, ) + tuple(t.shape[1:]), dtype=t.dtype, device='cuda')
            padded[:t.size(0)] = t
            out = [torch.empty_like(padded) for _ in range(num_ranks)]
            dist.all_gather(out, padded, group=group)
            return torch.cat([o[:counts[i]] for i, o in enumerate(out)], dim=0)

        pad_rows = max(counts)
        g_x_dq = gather(x_dq, pad_rows)
        g_topk_idx = gather(topk_idx, pad_rows)
        g_topk_weights = gather(topk_weights, pad_rows)

        # Each (token, slot) pair is written by exactly one rank, so a FP32 all-reduce
        # sums disjoint contributions exactly
        partials = torch.zeros((num_global_tokens, num_topk, hidden), dtype=torch.float, device='cuda')
        num_recv_per_expert = torch.zeros((num_experts_per_rank, ), dtype=torch.int, device='cuda')
        use_a2 = bool(args.per_expert_a2)
        a2_scales = torch.ones((num_experts_per_rank, ), dtype=torch.float, device='cuda') if use_a2 else None
        for local_e in range(num_experts_per_rank):
            global_e = rank_idx * num_experts_per_rank + local_e
            token_sel, slot_sel = (g_topk_idx == global_e).nonzero(as_tuple=True)
            num_recv_per_expert[local_e] = token_sel.numel()
            if token_sel.numel() == 0:
                continue
            ones2 = torch.ones(2, device='cuda')
            partial, a2 = _reference_expert_ffn(
                g_x_dq[token_sel],
                l1_weights_dq[local_e], l2_weights_dq[local_e],
                g_topk_weights[token_sel, slot_sel],
                intermediate_hidden, activation_clamp,
                l1_alphas[local_e] if l1_alphas is not None else ones2,
                l2_alphas[local_e] if l2_alphas is not None else ones2[0],
                use_l2_input_scale=use_a2)
            partials[token_sel, slot_sel] = partial.float()
            if use_a2:
                a2_scales[local_e] = a2
        dist.all_reduce(partials, group=group)
        a2_scales_for_kernel = a2_scales

        # Match vLLM's serial ordering exactly: reduce routed BF16 partials in
        # FP32, cast routed output to BF16, apply routed scaling and round to
        # BF16 again, then add the BF16 shared result.
        y_local = partials[
            offsets[rank_idx]: offsets[rank_idx] + num_tokens
        ].sum(dim=1).bfloat16()
        if args.routed_scaling_factor != 1.0:
            y_local = (
                y_local.float() * args.routed_scaling_factor
            ).bfloat16()
        y_local = y_local.float()

        # BF16 shared expert on the local tokens (mirror the kernel: FP32 GEMM accumulate
        # -> BF16 gate/up round -> clamp -> SwiGLU in FP32 -> BF16 intermediate
        # -> FP32 GEMM accumulate -> BF16 partial summed by the combine in FP32)
        if num_shared_experts > 0:
            sih = intermediate_hidden * num_shared_experts
            h1 = x.float() @ shared_l1_weights.float().t()
            gate, up = h1[:, :sih].bfloat16(), h1[:, sih:].bfloat16()
            if activation_clamp is not None and activation_clamp != float('inf'):
                gate = gate.clamp_max(activation_clamp)
                up = up.clamp(-activation_clamp, activation_clamp)
            gate = gate.float()
            act = ((gate / (1.0 + torch.exp(-gate))) * up.float()).bfloat16()
            shared_partial = (act.float() @ shared_l2_weights.float().t()).bfloat16()
            y_local = y_local + shared_partial.float()

        y_ref = y_local.bfloat16()
        return y_ref, num_recv_per_expert

    # Correctness
    dist_print('Running correctness tests:', once_in_node=True)
    y_ref, num_recv_per_expert = run_reference()
    for i in range(args.num_correctness_tests):
        y_fused = run_fused()
        if y_ref.numel() == 0:
            assert y_fused.shape == y_ref.shape
            diff = max_abs_diff = relative_l2 = 0.0
            error = y_fused.float() - y_ref.float()
        else:
            diff = calc_diff(y_fused, y_ref)
            error = y_fused.float() - y_ref.float()
            max_abs_diff = error.abs().max().item()
            relative_l2 = (
                torch.linalg.vector_norm(error)
                / torch.linalg.vector_norm(y_ref.float()).clamp_min(1e-12)
            ).item()
        if diff >= 1e-3 and int(os.getenv('DG_TEST_DEBUG', '0')):
            err = (y_fused.float() - y_ref.float()).abs().max(dim=1).values
            bad = (err > 1.0).nonzero().flatten()
            print(f'DBG rank {rank_idx}: bad tokens {bad.numel()}/{num_tokens}, '
                  f'first {bad[:24].tolist()}', flush=True)
            if bad.numel() > 0:
                t = int(bad[0].item())
                col_err = (y_fused[t].float() - y_ref[t].float()).abs()
                bad_cols = (col_err > 1.0).nonzero().flatten()
                print(f'DBG rank {rank_idx}: token {t} bad cols {bad_cols.numel()}/{hidden}, '
                      f'first {bad_cols[:16].tolist()}', flush=True)
        assert diff < 1e-3, f'Rank {rank_idx}: diff {diff} is too large (max abs {max_abs_diff})'
        assert relative_l2 < 2e-2, (
            f'Rank {rank_idx}: relative L2 {relative_l2} is too large '
            f'(max abs {max_abs_diff})')
        dist.barrier()
        if i == 0:
            dist_print(f' > Output diff: {diff:.3e} (relative L2: {relative_l2:.3e}, '
                       f'max abs diff: {max_abs_diff:.3e})')

    # Check cumulative stats: `num_correctness_tests` accumulations over the initial values
    expected_stats = cumulative_stats_initial + args.num_correctness_tests * num_recv_per_expert
    assert torch.equal(cumulative_local_expert_recv_stats_fused, expected_stats), \
        f'Rank {rank_idx}: cumulative stats mismatch'
    dist_print(' > All correctness tests passed', once_in_node=True)
    dist_print(once_in_node=True)

    # Benchmark
    t_fused = bench_kineto(run_fused, 'mega_moe', barrier=lambda: dist.barrier())

    # Count locally received tokens for FLOP statistics
    num_recv_tokens = num_recv_per_expert.sum().item()
    safe_div = lambda a, b: float('nan') if b == 0 else a / b
    tflops = safe_div(2 * num_recv_tokens * (hidden * intermediate_hidden * 3) / 1e12, t_fused)
    dist_print('Performance:', once_in_node=True)
    dist_print(f' > EP: {rank_idx:2}/{num_ranks} | {tflops:4.0f} TFLOPS | {t_fused * 1e6:4.0f} us')

    # Exit
    dist.barrier()
    buffer.destroy()
    dist.destroy_process_group()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Test NVFP4 x NVFP4 Mega MoE')

    # Resource settings
    parser.add_argument('--num-processes', type=int, default=4, help='Number of processes to spawn (default: 4)')

    # Model settings: defaults follow `nvidia/GLM-5.2-NVFP4`
    # (hidden 6144, MoE intermediate 2048, 256 routed experts, top-8, NVFP4 with group size 16)
    parser.add_argument('--num-max-tokens-per-rank', type=int, default=64, help='Number of maximum tokens per rank')
    parser.add_argument('--num-tokens', type=int, default=16,
                        help='Number of tokens per rank (negative follows max minus removed)')
    parser.add_argument('--zero-token-rank', type=int, default=-1,
                        help='Force one EP rank to receive zero tokens (-1 disables)')
    parser.add_argument('--num-max-removed-tokens', type=int, default=0, help='Maximum number of tokens to remove')
    parser.add_argument('--hidden', type=int, default=6144, help='Hidden size')
    parser.add_argument('--intermediate-hidden', type=int, default=2048, help='MoE intermediate hidden size')
    parser.add_argument('--activation-clamp', type=float, default=10, help='Clamp value for activation')
    parser.add_argument('--num-experts', type=int, default=256, help='Number of experts')
    parser.add_argument('--num-topk', type=int, default=8, help='Number of expert selections')
    parser.add_argument('--masked-ratio', type=float, default=0.0, help='Mask some expert selections')
    parser.add_argument('--fast-math', type=int, default=0, help='Enable fast math (0 or 1, default: 0 for exactness)')
    parser.add_argument('--per-expert-alphas', type=int, default=1,
                        help='Test per-local-expert L1/L2 scales (0 or 1)')
    parser.add_argument('--per-expert-a2', type=int, default=1,
                        help='Test per-local-expert down-proj input scale for the intermediate '
                             'NVFP4 requant (0 or 1)')
    parser.add_argument('--num-shared-experts', type=int, default=1,
                        help='Number of fused BF16 shared experts (0 disables)')
    parser.add_argument('--seed', type=int, default=0, help='Base random seed')
    parser.add_argument('--input-scale', type=float, default=1.0,
                        help='Scale applied to BF16 input samples')
    parser.add_argument('--routed-weight-scale', type=float, default=0.1,
                        help='Scale applied to routed-expert weight samples')
    parser.add_argument('--shared-weight-scale', type=float, default=0.1,
                        help='Scale applied to shared-expert weight samples')
    parser.add_argument('--routed-scaling-factor', type=float, default=0.75,
                        help='Scale routed BF16 output before adding shared output')

    # Test settings
    parser.add_argument('--num-correctness-tests', type=int, default=2, help='Number of correctness test rounds')
    args = parser.parse_args()

    torch.multiprocessing.spawn(test, args=(args.num_processes, args), nprocs=args.num_processes)
