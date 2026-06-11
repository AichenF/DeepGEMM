"""Single-rank correctness gate for the SM90 NVFP4 MegaMoE kernel."""

import argparse
import os
import sys

import torch
import torch.distributed as dist
from torch.utils.cpp_extension import load_inline

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

import deep_gemm
from deep_gemm.quantization_nvfp4 import (
    FP4_VALUES,
    dequantize_nvfp4_to_fp32,
    nvfp4_scale_to_tile_major,
    quantize_to_nvfp4,
    ue4m3_to_fp32,
)
from deep_gemm.testing import get_arch_major
from deep_gemm.utils import per_token_cast_to_fp8
from deep_gemm.utils.dist import init_dist


def _interleave_l1_n(tensor: torch.Tensor, gran: int = 8) -> torch.Tensor:
    groups, n_cols, *rest = tensor.shape
    half = n_cols // 2
    gate = tensor[:, :half].reshape(groups, half // gran, gran, *rest)
    up = tensor[:, half:].reshape(groups, half // gran, gran, *rest)
    return torch.empty_like(tensor).copy_(torch.stack([gate, up], dim=2).reshape(groups, n_cols, *rest))


def _pack_nvfp4_marlin(nibbles: torch.Tensor) -> torch.Tensor:
    *outer_shape, k = nibbles.shape
    assert k % 8 == 0
    chunks = nibbles.view(*outer_shape, k // 8, 8).to(torch.int16)
    return (chunks[..., 4:8] | (chunks[..., 0:4] << 4)).to(torch.uint8).view(*outer_shape, k // 2).contiguous()


_CUDA_DEQUANT_EXT = None


def _load_cuda_dequant_ext():
    global _CUDA_DEQUANT_EXT
    if _CUDA_DEQUANT_EXT is not None:
        return _CUDA_DEQUANT_EXT

    cuda_src = r"""
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <deep_gemm/quantization/nvfp4_dequant.cuh>

__global__ void nvfp4_lut_bytes_kernel(uint8_t* out) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= 128 * 16)
        return;

    const uint32_t scale = static_cast<uint32_t>(idx / 16);
    const uint32_t nibble = static_cast<uint32_t>(idx & 15);
    const uint32_t packed_byte = (nibble << 4) | nibble;
    const uint32_t q = packed_byte * 0x01010101u;
    const uint2 lut = deep_gemm::nvfp4::load_e2m1_ue4m3_lut(scale);
    const uint2 deq = deep_gemm::nvfp4::dequant_nvfp4_to_fp8_pair_with_lut(q, lut);

    const uint8_t* hi = reinterpret_cast<const uint8_t*>(&deq.x);
    const uint8_t* lo = reinterpret_cast<const uint8_t*>(&deq.y);
    out[idx * 2] = hi[0];
    out[idx * 2 + 1] = lo[0];
}

torch::Tensor nvfp4_lut_bytes_cuda() {
    auto out = torch::empty({128, 16, 2}, torch::device(torch::kCUDA).dtype(torch::kUInt8));
    nvfp4_lut_bytes_kernel<<<8, 256>>>(out.data_ptr<uint8_t>());
    return out;
}
"""
    cpp_src = "torch::Tensor nvfp4_lut_bytes_cuda();"
    _CUDA_DEQUANT_EXT = load_inline(
        name="deepgemm_nvfp4_dequant_lut_test",
        cpp_sources=cpp_src,
        cuda_sources=cuda_src,
        functions=["nvfp4_lut_bytes_cuda"],
        extra_include_paths=[os.path.join(REPO_ROOT, "deep_gemm", "include")],
        extra_cuda_cflags=["--expt-relaxed-constexpr"],
        verbose=False,
    )
    return _CUDA_DEQUANT_EXT


def _run_cuda_dequant_lut_unit_test() -> None:
    ext = _load_cuda_dequant_ext()
    got = ext.nvfp4_lut_bytes_cuda().cpu()

    scales = torch.arange(128, dtype=torch.uint8, device="cpu")
    nibbles = torch.arange(16, dtype=torch.uint8, device="cpu")
    mag = FP4_VALUES[(nibbles & 0x7).long()].view(1, 16)
    signed = torch.where(((nibbles >> 3) & 0x1).bool().view(1, 16), -mag, mag)
    expected = (signed * ue4m3_to_fp32(scales).view(128, 1)).to(torch.float8_e4m3fn).view(torch.uint8)
    expected = torch.where(
        ((expected & 0x7F) == 0) & (((nibbles >> 3) & 0x1).bool().view(1, 16)),
        expected | 0x80,
        expected,
    )
    expected = expected.unsqueeze(-1).expand(128, 16, 2).contiguous()
    torch.testing.assert_close(got, expected, rtol=0, atol=0)
    print("NVFP4 CUDA dequant LUT unit test: PASS", flush=True)


def _run_dequant_unit_test() -> None:
    scales = torch.tensor([0x00, 0x01, 0x07, 0x08, 0x38, 0x3F, 0x7E, 0x7F], dtype=torch.uint8)
    nibbles = torch.arange(16, dtype=torch.uint8).view(1, 1, 16).expand(scales.numel(), 1, 16).clone()
    packed = _pack_nvfp4_marlin(nibbles)
    got = dequantize_nvfp4_to_fp32(packed, scales.view(-1, 1, 1), group_size=16)

    mag = FP4_VALUES.to(nibbles.device)[(nibbles & 0x7).long()]
    signed = torch.where(((nibbles >> 3) & 0x1).bool(), -mag, mag)
    expected = signed * ue4m3_to_fp32(scales).view(-1, 1, 1)
    torch.testing.assert_close(got, expected, rtol=0, atol=0)

    tile_weight = torch.randn(2, 256, 256, dtype=torch.bfloat16) * 0.1
    tile_packed, tile_scale = quantize_to_nvfp4(tile_weight, group_size=16)
    tile_scale_tm = nvfp4_scale_to_tile_major(tile_scale)
    torch.testing.assert_close(
        dequantize_nvfp4_to_fp32(tile_packed, tile_scale, group_size=16),
        dequantize_nvfp4_to_fp32(tile_packed, tile_scale_tm, group_size=16),
        rtol=0,
        atol=0,
    )

    l1_weight = torch.randn(2, 512, 256, dtype=torch.bfloat16) * 0.1
    l2_weight = torch.randn(2, 256, 256, dtype=torch.bfloat16) * 0.1
    l1_packed, l1_scale = quantize_to_nvfp4(l1_weight, group_size=16)
    l2_packed, l2_scale = quantize_to_nvfp4(l2_weight, group_size=16)
    transformed_l1, transformed_l2 = deep_gemm.transform_nvfp4_weights_for_mega_moe_sm90(
        (l1_packed, l1_scale), (l2_packed, l2_scale),
    )
    torch.testing.assert_close(
        dequantize_nvfp4_to_fp32(transformed_l1[0], transformed_l1[1], group_size=16),
        _interleave_l1_n(dequantize_nvfp4_to_fp32(l1_packed, l1_scale, group_size=16)),
        rtol=0,
        atol=0,
    )
    torch.testing.assert_close(
        dequantize_nvfp4_to_fp32(transformed_l2[0], transformed_l2[1], group_size=16),
        dequantize_nvfp4_to_fp32(l2_packed, l2_scale, group_size=16),
        rtol=0,
        atol=0,
    )
    print('NVFP4 dequant unit test: PASS', flush=True)


def _silu(x: torch.Tensor) -> torch.Tensor:
    return x * torch.sigmoid(x)


def _run_case(args: argparse.Namespace, m_tokens: int, weight_scale: float,
              rank_idx: int, group: dist.ProcessGroup) -> None:
    torch.manual_seed(args.seed + m_tokens + int(weight_scale * 1000000))

    hidden = args.hidden
    intermediate_hidden = args.intermediate_hidden
    num_experts = args.num_experts
    num_topk = args.num_topk
    num_ranks = dist.get_world_size(group)
    num_local_experts = num_experts // num_ranks
    num_max_tokens_per_rank = args.num_max_tokens_per_rank or max(32, m_tokens)
    if num_max_tokens_per_rank < m_tokens:
        raise ValueError(
            f"num_max_tokens_per_rank={num_max_tokens_per_rank} is smaller than M={m_tokens}"
        )

    if rank_idx == 0:
        print(
            f"=== NVFP4 single-rank correctness M={m_tokens}, "
            f"NE={num_experts}, NL={num_local_experts}, NK={num_topk}, "
            f"NMT={num_max_tokens_per_rank}, weight_scale={weight_scale:g}, "
            f"reference={args.reference_mode}, weight_mode={args.weight_mode}, "
            f"unit_weight_scale={args.unit_weight_scale} ===",
            flush=True,
        )

    buffer = deep_gemm.get_symm_buffer_for_mega_moe(
        group,
        num_experts,
        num_max_tokens_per_rank,
        num_topk,
        hidden,
        intermediate_hidden,
        use_fp8_dispatch=True,
        activation="swiglu",
    )

    x_bf = torch.randn((m_tokens, hidden), dtype=torch.bfloat16, device="cuda")
    l1_bf = torch.randn(
        (num_local_experts, 2 * intermediate_hidden, hidden),
        dtype=torch.bfloat16,
        device="cuda",
    ) * weight_scale
    l2_bf = torch.randn(
        (num_local_experts, hidden, intermediate_hidden),
        dtype=torch.bfloat16,
        device="cuda",
    ) * weight_scale
    scores = torch.randn((m_tokens, num_experts), dtype=torch.float, device="cuda")
    topk_weights, topk_idx = torch.topk(scores, num_topk, dim=-1)

    x_fp8, x_sf = per_token_cast_to_fp8(x_bf, use_ue8m0=False, gran_k=128)

    l1_packed, l1_scale = quantize_to_nvfp4(l1_bf, group_size=16)
    l2_packed, l2_scale = quantize_to_nvfp4(l2_bf, group_size=16)
    l1_dequant = dequantize_nvfp4_to_fp32(l1_packed, l1_scale, group_size=16)
    l2_dequant = dequantize_nvfp4_to_fp32(l2_packed, l2_scale, group_size=16)
    if args.reference_mode == "fp8-bridge":
        l1_dequant = l1_dequant.to(torch.float8_e4m3fn).float()
        l2_dequant = l2_dequant.to(torch.float8_e4m3fn).float()

    if args.weight_mode == "fp8-shadow":
        transformed_l1, transformed_l2 = deep_gemm.materialize_nvfp4_fp8_shadow_for_mega_moe_sm90(
            (l1_packed, l1_scale), (l2_packed, l2_scale),
        )
        kernel_fn = deep_gemm.fp8_mega_moe
    else:
        transformed_l1, transformed_l2 = deep_gemm.transform_nvfp4_weights_for_mega_moe_sm90(
            (l1_packed, l1_scale), (l2_packed, l2_scale),
        )
        kernel_fn = deep_gemm.nvfp4_mega_moe

    cumulative_stats = torch.zeros(num_local_experts, dtype=torch.int, device="cuda")
    buffer.x[:m_tokens].copy_(x_fp8)
    buffer.x_sf[:m_tokens].copy_(x_sf)
    buffer.topk_idx[:m_tokens].copy_(topk_idx.to(torch.int32))
    buffer.topk_weights[:m_tokens].copy_(topk_weights.to(torch.float32))

    y_kernel = torch.zeros((m_tokens, hidden), dtype=torch.bfloat16, device="cuda")
    kernel_fn(
        y_kernel,
        transformed_l1,
        transformed_l2,
        buffer,
        cumulative_local_expert_recv_stats=cumulative_stats,
        recipe=(128, 128, 128),
        activation="swiglu",
        activation_clamp=args.activation_clamp,
        fast_math=bool(args.fast_math),
    )
    torch.cuda.synchronize()
    dist.barrier(group=group)

    if args.reference_mode == "fp8-bridge":
        x_ref = (x_fp8.float().view(m_tokens, hidden // 128, 128) * x_sf.unsqueeze(-1)).view(m_tokens, hidden)
    else:
        x_ref = x_bf.float()
    y_ref = torch.zeros((m_tokens, hidden), device="cuda", dtype=torch.float32)
    for token_idx in range(m_tokens):
        for topk_i in range(num_topk):
            expert_idx = topk_idx[token_idx, topk_i].item()
            route_weight = topk_weights[token_idx, topk_i].item()
            l1_out = l1_dequant[expert_idx].float() @ x_ref[token_idx]
            gate, up = l1_out[:intermediate_hidden], l1_out[intermediate_hidden:]
            gate = gate.clamp(max=args.activation_clamp)
            up = up.clamp(min=-args.activation_clamp, max=args.activation_clamp)
            intermediate = _silu(gate) * up * route_weight
            y_ref[token_idx] += l2_dequant[expert_idx].float() @ intermediate

    finite = torch.isfinite(y_kernel).all().item()
    diff = y_kernel.float() - y_ref
    cosine = torch.nn.functional.cosine_similarity(y_kernel.float(), y_ref, dim=-1)
    cosine_min = cosine.min().item()
    cosine_mean = cosine.mean().item()
    norm_ratio = (
        torch.linalg.vector_norm(y_kernel.float()) /
        torch.linalg.vector_norm(y_ref).clamp_min(1e-30)
    ).item()

    if rank_idx == 0:
        print(f"cum_stats: {cumulative_stats.cpu().tolist()}", flush=True)
        print(
            f"Kernel: abs_max={y_kernel.abs().max().item():.4e} "
            f"abs_mean={y_kernel.abs().mean().item():.4e} finite={finite}",
            flush=True,
        )
        print(
            f"Ref:    abs_max={y_ref.abs().max().item():.4e} "
            f"abs_mean={y_ref.abs().mean().item():.4e}",
            flush=True,
        )
        print(
            f"Diff: max_abs={diff.abs().max().item():.4e} "
            f"mean_abs={diff.abs().mean().item():.4e}",
            flush=True,
        )
        print(
            f"Per-token cosine sim: min={cosine_min:.4f} mean={cosine_mean:.4f}",
            flush=True,
        )
        print(f"Norm ratio: kernel/ref={norm_ratio:.4f}", flush=True)

    if not finite:
        raise AssertionError(f"M={m_tokens}, weight_scale={weight_scale:g}: kernel produced non-finite values")
    if cosine_mean < args.cosine_mean_threshold:
        raise AssertionError(
            f"M={m_tokens}, weight_scale={weight_scale:g}: cosine_mean={cosine_mean:.4f} < {args.cosine_mean_threshold:.4f}"
        )
    if cosine_min < args.cosine_min_threshold:
        raise AssertionError(
            f"M={m_tokens}, weight_scale={weight_scale:g}: cosine_min={cosine_min:.4f} < {args.cosine_min_threshold:.4f}"
        )
    if not (args.norm_ratio_min <= norm_ratio <= args.norm_ratio_max):
        raise AssertionError(
            f"M={m_tokens}, weight_scale={weight_scale:g}: norm_ratio={norm_ratio:.4f} "
            f"outside [{args.norm_ratio_min:.4f}, {args.norm_ratio_max:.4f}]"
        )

    if rank_idx == 0:
        print(
            f"PASS M={m_tokens} weight_scale={weight_scale:g}: "
            f"cosine_min={cosine_min:.4f} cosine_mean={cosine_mean:.4f}",
            flush=True,
        )

    buffer.destroy()


def _worker(local_rank: int, num_local_ranks: int, args: argparse.Namespace) -> None:
    rank_idx, _, group = init_dist(local_rank, num_local_ranks)
    try:
        if rank_idx == 0:
            _run_dequant_unit_test()
            _run_cuda_dequant_lut_unit_test()
        dist.barrier(group=group)
        if get_arch_major() != 9:
            if rank_idx == 0:
                print(f"[SKIP] requires SM90, got SM{get_arch_major()}0", flush=True)
            return
        for weight_scale in args.weight_scales:
            for m_tokens in args.batches:
                _run_case(args, m_tokens, weight_scale, rank_idx, group)
    finally:
        dist.destroy_process_group()


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="SM90 NVFP4 MegaMoE correctness gate")
    parser.add_argument("--batches", nargs="+", type=int, default=[32, 256])
    parser.add_argument("--hidden", type=int, default=7168)
    parser.add_argument("--intermediate-hidden", type=int, default=2048)
    parser.add_argument("--num-experts", type=int, default=8)
    parser.add_argument("--num-topk", type=int, default=4)
    parser.add_argument("--num-max-tokens-per-rank", type=int, default=0)
    parser.add_argument("--activation-clamp", type=float, default=10.0)
    parser.add_argument("--fast-math", type=int, default=1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--weight-scales", nargs="+", type=float, default=[0.05])
    parser.add_argument("--reference-mode", choices=["fp8-bridge", "exact-nvfp4"], default="fp8-bridge")
    parser.add_argument("--weight-mode", choices=["compact", "fp8-shadow"], default="compact",
                        help="compact: run NVFP4 fused kernel; fp8-shadow: materialize NVFP4 weights to FP8 once and run PR323 FP8 fused kernel")
    parser.add_argument("--no-unit-weight-scale", action="store_true",
                        help="debug only: do not specialize fp8-shadow for all-one FP8 weight scales")
    parser.add_argument("--cosine-mean-threshold", type=float, default=0.9)
    parser.add_argument("--cosine-min-threshold", type=float, default=0.9)
    parser.add_argument("--norm-ratio-min", type=float, default=0.5)
    parser.add_argument("--norm-ratio-max", type=float, default=2.0)
    args = parser.parse_args()
    if args.weight_mode == "fp8-shadow" and args.reference_mode != "fp8-bridge":
        parser.error("--weight-mode fp8-shadow requires --reference-mode fp8-bridge")
    args.unit_weight_scale = args.weight_mode == "fp8-shadow" and not args.no_unit_weight_scale
    return args


if __name__ == "__main__":
    args = _parse_args()
    if args.unit_weight_scale:
        os.environ["DG_SM90_FP8_UNIT_WEIGHT_SCALE"] = "1"
    torch.multiprocessing.spawn(_worker, args=(1, args), nprocs=1, join=True)
