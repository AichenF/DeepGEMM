"""Experimental two-seed register-weight NVFP4 prepack and launch wrapper."""

from typing import Optional, Tuple

import torch

from deep_gemm import _C
from deep_gemm.mega import SymmBuffer
from deep_gemm.mega import transform_nvfp4_weights_for_mega_moe_sm90 as _stable_transform
from deep_gemm.quantization_nvfp4 import ue4m3_to_fp32


_TWO_SEED_TENSOR_SCALES = {}
_TWO_SEED_COMBINED_SCALES = {}


@torch.inference_mode()
def _pack_two_seed_rs_layout(
    fused_weight: torch.Tensor,
    scale_tile_major: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor, int, int, int]:
    """Pack every BN256/BK128 tile as sixteen lane-native RS fragments.

    One 64x32 fragment stores 1024 bytes of E2M1 plus 256 bytes of two-seed
    metadata.  Sixteen fragments therefore occupy exactly the existing
    256x80-byte tile, so the TMA descriptor and public tensor shape stay
    unchanged even though the payload is no longer logical-row-major.
    """
    assert fused_weight.dtype == torch.uint8 and fused_weight.dim() == 3
    assert scale_tile_major.dtype == torch.uint8 and scale_tile_major.dim() == 5
    experts, rows, storage_k = fused_weight.shape
    experts_s, n_blocks, k_blocks, block_n, scale_groups = scale_tile_major.shape
    assert experts == experts_s and block_n == 256 and scale_groups == 8
    assert rows == n_blocks * block_n and storage_k == k_blocks * 80

    fused_tiles = (
        fused_weight.view(experts, n_blocks, block_n, k_blocks, 80)
        .permute(0, 1, 3, 2, 4)
        .contiguous()
    )
    marlin = fused_tiles[..., :64].view(
        experts, n_blocks, k_blocks, block_n, 16, 4
    )
    codes = torch.cat(((marlin >> 4) & 0xF, marlin & 0xF), dim=-1).reshape(
        experts, n_blocks, k_blocks, block_n, 128
    )

    scale_fp32 = ue4m3_to_fp32(scale_tile_major)
    max_scale = scale_fp32.amax(dim=(1, 2, 3, 4)).clamp_min(1.0e-30)
    # Preserve the UE4M3 mantissa exactly by normalizing with a power of two.
    min_tensor_scale = max_scale * (6.0 / 448.0)
    tensor_scale = torch.exp2(torch.ceil(torch.log2(min_tensor_scale)))
    ratio = scale_fp32 / tensor_scale[:, None, None, None, None]
    ratio_fp8 = ratio.to(torch.float8_e4m3fn)
    ratio_code = ratio_fp8.view(torch.uint8)
    ratio_rounded = ratio_fp8.float()
    ratio_mismatch_count = int((ratio_rounded != ratio).sum().item())
    seed_half = (ratio_rounded * 0.5).to(torch.float8_e4m3fn).view(torch.uint8)
    seed_one_half = (ratio_rounded * 1.5).to(torch.float8_e4m3fn).view(torch.uint8)

    # The byte-add exponent construction is exact in this interior range.  A
    # flagged raw ratio code preserves correctness for rare subnormal groups;
    # the device decoder sends only those pairs through the existing LUT.
    seed_fast = (ratio_code >= 16) & (ratio_code <= 105)
    seed0 = torch.where(seed_fast, seed_half, ratio_code | 0x80)
    seed1 = torch.where(seed_fast, seed_one_half, torch.zeros_like(seed_one_half))

    tid = torch.arange(128, device=fused_weight.device, dtype=torch.long)
    row0 = (tid // 4) % 8 + 16 * (tid // 32)
    row1 = row0 + 8
    k0 = 4 * (tid % 4)
    k_lane = k0[:, None] + torch.arange(4, device=fused_weight.device)
    meta_group = torch.arange(32, device=fused_weight.device, dtype=torch.long)
    meta_row0 = meta_group % 8 + 16 * (meta_group // 8)
    meta_row1 = meta_row0 + 8
    sign_order = torch.tensor(
        [0, 4, 1, 5, 2, 6, 3, 7],
        device=fused_weight.device,
        dtype=torch.long,
    )
    nibble_shifts = torch.arange(
        0, 32, 4, device=fused_weight.device, dtype=torch.int32
    )

    fragments = []
    for n64 in range(4):
        n_base = n64 * 64
        for k32 in range(4):
            k_base = k32 * 32
            tile_codes = codes[
                ..., n_base : n_base + 64, k_base : k_base + 32
            ].to(torch.int32)
            groups = torch.stack(
                (
                    tile_codes[..., row0[:, None], k_lane],
                    tile_codes[..., row1[:, None], k_lane],
                    tile_codes[..., row0[:, None], k_lane + 16],
                    tile_codes[..., row1[:, None], k_lane + 16],
                ),
                dim=-2,
            )

            packed_pairs = []
            for pair_idx in range(2):
                values = torch.cat(
                    (groups[..., pair_idx * 2, :], groups[..., pair_idx * 2 + 1, :]),
                    dim=-1,
                )
                magnitudes = values & 0x7
                signs = (values >> 3).index_select(-1, sign_order)
                nibbles = magnitudes | (signs << 3)
                packed_pairs.append(
                    (nibbles << nibble_shifts).sum(dim=-1).to(torch.int32)
                )
            packed_words = torch.stack(packed_pairs, dim=-1).contiguous()
            packed_bytes = packed_words.view(torch.uint8).reshape(
                experts, n_blocks, k_blocks, 1024
            )

            scale_k0 = k32 * 2
            metadata = torch.stack(
                (
                    seed0[..., n_base + meta_row0, scale_k0],
                    seed1[..., n_base + meta_row0, scale_k0],
                    seed0[..., n_base + meta_row1, scale_k0],
                    seed1[..., n_base + meta_row1, scale_k0],
                    seed0[..., n_base + meta_row0, scale_k0 + 1],
                    seed1[..., n_base + meta_row0, scale_k0 + 1],
                    seed0[..., n_base + meta_row1, scale_k0 + 1],
                    seed1[..., n_base + meta_row1, scale_k0 + 1],
                ),
                dim=-1,
            ).reshape(experts, n_blocks, k_blocks, 256)
            fragments.append(torch.cat((packed_bytes, metadata), dim=-1))

    packed_tiles = torch.cat(fragments, dim=-1)
    assert packed_tiles.shape[-1] == block_n * 80
    packed_output = (
        packed_tiles.view(experts, n_blocks, k_blocks, block_n, 80)
        .permute(0, 1, 3, 2, 4)
        .reshape(experts, rows, storage_k)
        .contiguous()
    )
    fallback_count = int((~seed_fast).sum().item())
    return (
        packed_output,
        tensor_scale.contiguous(),
        fallback_count,
        seed_fast.numel(),
        ratio_mismatch_count,
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
        l1_weights, l2_weights,
        block_n=block_n, block_k=block_k, group_size=group_size,
    )
    (
        l1_packed,
        l1_tensor_scale,
        l1_fallback,
        l1_groups,
        l1_ratio_mismatch,
    ) = _pack_two_seed_rs_layout(transformed_l1[0], transformed_l1[1])
    (
        l2_packed,
        l2_tensor_scale,
        l2_fallback,
        l2_groups,
        l2_ratio_mismatch,
    ) = _pack_two_seed_rs_layout(transformed_l2[0], transformed_l2[1])
    key = (int(l1_packed.data_ptr()), int(l2_packed.data_ptr()))
    _TWO_SEED_TENSOR_SCALES[key] = (l1_tensor_scale, l2_tensor_scale)
    print(
        "NVFP4_RS_TWO_SEED_PREPACK "
        f"l1_fallback={l1_fallback}/{l1_groups} "
        f"l2_fallback={l2_fallback}/{l2_groups} "
        f"l1_ratio_mismatch={l1_ratio_mismatch} "
        f"l2_ratio_mismatch={l2_ratio_mismatch}",
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
        raise RuntimeError("two-seed weights were not produced by this candidate transform")

    combined_key = (
        key,
        0 if l1_global_scales is None else int(l1_global_scales.data_ptr()),
        0 if l2_global_scales is None else int(l2_global_scales.data_ptr()),
    )
    effective_scales = _TWO_SEED_COMBINED_SCALES.get(combined_key)
    if effective_scales is None:
        effective_scales = (
            tensor_scales[0] if l1_global_scales is None
            else tensor_scales[0] * l1_global_scales,
            tensor_scales[1] if l2_global_scales is None
            else tensor_scales[1] * l2_global_scales,
        )
        _TWO_SEED_COMBINED_SCALES[combined_key] = effective_scales

    _C.nvfp4_nibble_group_mega_moe(
        y, l1_weights, l2_weights,
        cumulative_local_expert_recv_stats,
        effective_scales[0], effective_scales[1],
        sym_buffer.buffer,
        sym_buffer.handle.buffer_ptrs, sym_buffer.group.rank(),
        sym_buffer.num_max_tokens_per_rank,
        sym_buffer.num_experts, sym_buffer.num_topk,
        recipe, activation, activation_clamp, fast_math,
    )
