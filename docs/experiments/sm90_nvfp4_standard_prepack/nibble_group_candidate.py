"""Experimental row-major two-seed NVFP4 prepack and launch wrapper."""

from typing import Optional, Tuple

import torch

from deep_gemm import _C
from deep_gemm.mega import SymmBuffer
from deep_gemm.mega import transform_nvfp4_weights_for_mega_moe_sm90 as _stable_transform
from deep_gemm.quantization_nvfp4 import ue4m3_to_fp32


_TWO_SEED_TENSOR_SCALES = {}
_TWO_SEED_COMBINED_SCALES = {}


@torch.inference_mode()
def _pack_two_seed_row_layout(
    fused_weight: torch.Tensor,
    scale_tile_major: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor, int, int, int]:
    """Replace each 64B-FP4/8B-SF row with 64B braided FP4/16B seeds.

    Each BK32 slice pairs its two K16 scale groups. Four FP4 words hold four
    values from each group, while four metadata bytes hold Q(scale/2) and
    Q(3*scale/2) for both groups. The public 80-byte row ABI is unchanged.
    """
    assert fused_weight.dtype == torch.uint8 and fused_weight.dim() == 3
    assert scale_tile_major.dtype == torch.uint8 and scale_tile_major.dim() == 5
    experts, rows, storage_k = fused_weight.shape
    experts_s, n_blocks, k_blocks, block_n, scale_groups = scale_tile_major.shape
    assert experts == experts_s and block_n == 256 and scale_groups == 8
    assert rows == n_blocks * block_n and storage_k == k_blocks * 80

    fused_rows = fused_weight.view(experts, rows, k_blocks, 80)
    marlin = fused_rows[..., :64].view(experts, rows, k_blocks, 16, 4)
    # Stable Marlin bytes store K[0:4] in high nibbles and K[4:8] in low.
    codes = torch.cat(((marlin >> 4) & 0xF, marlin & 0xF), dim=-1).reshape(
        experts, rows, k_blocks, 128
    )

    scale_rows = (
        scale_tile_major.permute(0, 1, 3, 2, 4)
        .contiguous()
        .view(experts, rows, k_blocks, 8)
    )
    scale_fp32 = ue4m3_to_fp32(scale_rows)
    max_scale = scale_fp32.amax(dim=(1, 2, 3)).clamp_min(1.0e-30)
    # A power-of-two tensor scale keeps representable UE4M3 ratios exact while
    # putting max(scale_ratio * E2M1_MAX) inside finite E4M3 range.
    min_tensor_scale = max_scale * (6.0 / 448.0)
    tensor_scale = torch.exp2(torch.ceil(torch.log2(min_tensor_scale)))
    ratio = scale_fp32 / tensor_scale[:, None, None, None]
    ratio_fp8 = ratio.to(torch.float8_e4m3fn)
    ratio_code = ratio_fp8.view(torch.uint8)
    ratio_rounded = ratio_fp8.float()
    seed_half = (ratio_rounded * 0.5).to(torch.float8_e4m3fn).view(torch.uint8)
    seed_one_half = (
        (ratio_rounded * 1.5).to(torch.float8_e4m3fn).view(torch.uint8)
    )

    # Exponent-byte construction is exact only away from E4M3 subnormal and
    # saturation boundaries. Flag other groups for the existing LUT fallback.
    seed_fast = (ratio_code >= 16) & (ratio_code <= 105)
    seed0 = torch.where(seed_fast, seed_half, ratio_code | 0x80)
    seed1 = torch.where(seed_fast, seed_one_half, torch.zeros_like(seed_one_half))

    # (BK32, K16-half, four K4 chunks, four values) ->
    # (BK32, chunk, [first K16 values, second K16 values]).
    paired = (
        codes.view(experts, rows, k_blocks, 4, 2, 4, 4)
        .permute(0, 1, 2, 3, 5, 4, 6)
        .contiguous()
        .view(experts, rows, k_blocks, 4, 4, 8)
        .to(torch.int32)
    )
    magnitudes = paired & 0x7
    signs = (
        (paired >> 3)
        .view(experts, rows, k_blocks, 4, 4, 2, 4)
        .transpose(-1, -2)
        .reshape(experts, rows, k_blocks, 4, 4, 8)
    )
    braided = magnitudes | (signs << 3)
    shifts = torch.arange(0, 32, 4, device=codes.device, dtype=torch.int32)
    words = (braided << shifts).sum(dim=-1).to(torch.int32)
    packed_bytes = words.contiguous().view(torch.uint8).reshape(
        experts, rows, k_blocks, 64
    )

    seed0_pairs = seed0.view(experts, rows, k_blocks, 4, 2)
    seed1_pairs = seed1.view(experts, rows, k_blocks, 4, 2)
    metadata = torch.stack(
        (
            seed0_pairs[..., 0],
            seed1_pairs[..., 0],
            seed0_pairs[..., 1],
            seed1_pairs[..., 1],
        ),
        dim=-1,
    ).reshape(experts, rows, k_blocks, 16)
    packed_output = torch.cat((packed_bytes, metadata), dim=-1).reshape(
        experts, rows, storage_k
    ).contiguous()

    fallback_count = int((~seed_fast).sum().item())
    exact_scale_count = int(
        ((ratio_rounded * tensor_scale[:, None, None, None]) == scale_fp32)
        .sum()
        .item()
    )
    return (
        packed_output,
        tensor_scale.contiguous(),
        fallback_count,
        seed_fast.numel(),
        exact_scale_count,
    )


def transform_nvfp4_weights_for_mega_moe_sm90_nibble_group(
    l1_weights: Tuple[torch.Tensor, torch.Tensor],
    l2_weights: Tuple[torch.Tensor, torch.Tensor],
    block_n: int = 256,
    block_k: int = 128,
    group_size: int = 16,
) -> Tuple[Tuple[torch.Tensor, torch.Tensor], Tuple[torch.Tensor, torch.Tensor]]:
    assert block_n == 256
    transformed_l1, transformed_l2 = _stable_transform(
        l1_weights,
        l2_weights,
        block_n=block_n,
        block_k=block_k,
        group_size=group_size,
    )
    l1_result = _pack_two_seed_row_layout(transformed_l1[0], transformed_l1[1])
    l2_result = _pack_two_seed_row_layout(transformed_l2[0], transformed_l2[1])
    l1_packed, l1_tensor_scale, l1_fallback, l1_groups, l1_exact = l1_result
    l2_packed, l2_tensor_scale, l2_fallback, l2_groups, l2_exact = l2_result
    key = (int(l1_packed.data_ptr()), int(l2_packed.data_ptr()))
    _TWO_SEED_TENSOR_SCALES[key] = (l1_tensor_scale, l2_tensor_scale)
    print(
        "NVFP4_SHARED_TWO_SEED_PREPACK "
        f"l1_fallback={l1_fallback}/{l1_groups} l1_exact={l1_exact}/{l1_groups} "
        f"l2_fallback={l2_fallback}/{l2_groups} l2_exact={l2_exact}/{l2_groups}",
        flush=True,
    )
    return (l1_packed, transformed_l1[1]), (l2_packed, transformed_l2[1])


def nvfp4_nibble_group_mega_moe(
    y: torch.Tensor,
    l1_weights: Tuple[torch.Tensor, torch.Tensor],
    l2_weights: Tuple[torch.Tensor, torch.Tensor],
    sym_buffer: SymmBuffer,
    cumulative_local_expert_recv_stats: Optional[torch.Tensor] = None,
    l1_global_scales: Optional[torch.Tensor] = None,
    l2_global_scales: Optional[torch.Tensor] = None,
    recipe: Tuple[int, int, int] = (128, 128, 128),
    activation: str = "swiglu",
    activation_clamp: Optional[float] = None,
    fast_math: bool = True,
) -> None:
    key = (int(l1_weights[0].data_ptr()), int(l2_weights[0].data_ptr()))
    tensor_scales = _TWO_SEED_TENSOR_SCALES.get(key)
    if tensor_scales is None:
        raise RuntimeError("two-seed weights were not produced by this transform")

    combined_key = (
        key,
        0 if l1_global_scales is None else int(l1_global_scales.data_ptr()),
        0 if l2_global_scales is None else int(l2_global_scales.data_ptr()),
    )
    effective_scales = _TWO_SEED_COMBINED_SCALES.get(combined_key)
    if effective_scales is None:
        effective_scales = (
            tensor_scales[0]
            if l1_global_scales is None
            else tensor_scales[0] * l1_global_scales,
            tensor_scales[1]
            if l2_global_scales is None
            else tensor_scales[1] * l2_global_scales,
        )
        _TWO_SEED_COMBINED_SCALES[combined_key] = effective_scales

    _C.nvfp4_nibble_group_mega_moe(
        y,
        l1_weights,
        l2_weights,
        cumulative_local_expert_recv_stats,
        effective_scales[0],
        effective_scales[1],
        sym_buffer.buffer,
        sym_buffer.handle.buffer_ptrs,
        sym_buffer.group.rank(),
        sym_buffer.num_max_tokens_per_rank,
        sym_buffer.num_experts,
        sym_buffer.num_topk,
        recipe,
        activation,
        activation_clamp,
        fast_math,
    )
