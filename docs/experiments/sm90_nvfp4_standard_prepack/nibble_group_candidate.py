"""Experimental lossless nibble-group NVFP4 prepack and launch wrapper."""

from typing import Optional, Tuple

import torch

from deep_gemm import _C
from deep_gemm.mega import SymmBuffer
from deep_gemm.mega import transform_nvfp4_weights_for_mega_moe_sm90 as _stable_transform

_LAYOUT_ATTR = "_deep_gemm_nvfp4_nibble_layout"


def _group_nibbles_by_half(fused_weight: torch.Tensor) -> torch.Tensor:
    assert fused_weight.dtype == torch.uint8 and fused_weight.dim() == 3
    experts, rows, storage_k = fused_weight.shape
    assert storage_k % 80 == 0
    k_blocks = storage_k // 80
    fused_rows = fused_weight.view(experts, rows, k_blocks, 80).clone()
    q_bytes = fused_rows[..., :64].view(
        experts, rows, k_blocks, 16, 4
    ).to(torch.int32)

    def pack_nibbles(nibbles: torch.Tensor) -> torch.Tensor:
        return (
            nibbles[..., 0]
            | (nibbles[..., 1] << 4)
            | (nibbles[..., 2] << 8)
            | (nibbles[..., 3] << 12)
        )

    high = pack_nibbles((q_bytes >> 4) & 0xf)
    low = pack_nibbles(q_bytes & 0xf)
    grouped = (high | (low << 16)).to(torch.int32)
    fused_rows[..., :64] = grouped.contiguous().view(torch.uint8).view(
        experts, rows, k_blocks, 64
    )
    return fused_rows.view(experts, rows, storage_k).contiguous()


def _braid_mode2_signs(fused_weight: torch.Tensor) -> torch.Tensor:
    assert fused_weight.dtype == torch.uint8 and fused_weight.dim() == 3
    experts, rows, storage_k = fused_weight.shape
    assert storage_k % 80 == 0
    k_blocks = storage_k // 80
    fused_rows = fused_weight.view(experts, rows, k_blocks, 80).clone()
    packed = fused_rows[..., :64].view(experts, rows, k_blocks, 16, 4)
    codes = torch.cat(((packed >> 4) & 0x0F, packed & 0x0F), dim=-1)
    magnitudes = codes & 0x07
    signs = codes >> 3
    # Keep the four high-nibble magnitudes followed by the four low-nibble
    # magnitudes, but braid their sign bits into the byte positions consumed
    # by the two output words.  This is Humming/Shawn's mode-2 sign transport:
    # output 0 can OR `packed` directly and output 1 can OR `packed << 4`.
    braided_signs = torch.stack(
        (
            signs[..., 4], signs[..., 0], signs[..., 5], signs[..., 1],
            signs[..., 6], signs[..., 2], signs[..., 7], signs[..., 3],
        ),
        dim=-1,
    )
    braided_nibbles = magnitudes | (braided_signs << 3)
    braided_bytes = (
        braided_nibbles[..., 0::2] |
        (braided_nibbles[..., 1::2] << 4)
    )
    fused_rows[..., :64] = braided_bytes.reshape(
        experts, rows, k_blocks, 64
    )
    return fused_rows.view(experts, rows, storage_k).contiguous()


def _use_h200_mimo_mode2(
    l1_fused_weight: torch.Tensor,
    l2_fused_weight: torch.Tensor,
) -> bool:
    props = torch.cuda.get_device_properties(l1_fused_weight.device)
    return (
        props.multi_processor_count >= 132
        and tuple(l1_fused_weight.shape) == (48, 4096, 3840)
        and tuple(l2_fused_weight.shape) == (48, 6144, 1280)
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
    mode2_nibble_weights = _use_h200_mimo_mode2(
        transformed_l1[0], transformed_l2[0]
    )
    transform = (
        _braid_mode2_signs
        if mode2_nibble_weights
        else _group_nibbles_by_half
    )
    l1_fused_weight = transform(transformed_l1[0])
    l2_fused_weight = transform(transformed_l2[0])
    layout = "mode2" if mode2_nibble_weights else "grouped"
    setattr(l1_fused_weight, _LAYOUT_ATTR, layout)
    setattr(l2_fused_weight, _LAYOUT_ATTR, layout)
    return (
        l1_fused_weight,
        transformed_l1[1],
    ), (
        l2_fused_weight,
        transformed_l2[1],
    )


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
    mode2_nibble_weights: Optional[bool] = None,
) -> None:
    l1_layout = getattr(l1_weights[0], _LAYOUT_ATTR, None)
    l2_layout = getattr(l2_weights[0], _LAYOUT_ATTR, None)
    if l1_layout != l2_layout:
        raise ValueError(f"L1/L2 NVFP4 layouts differ: {l1_layout} vs {l2_layout}")
    inferred_mode2 = l1_layout == "mode2"
    if mode2_nibble_weights is None:
        mode2_nibble_weights = inferred_mode2
    elif l1_layout is not None and mode2_nibble_weights != inferred_mode2:
        raise ValueError(
            "mode2_nibble_weights does not match the prepacked weight layout"
        )
    _C.nvfp4_nibble_group_mega_moe(
        y, l1_weights, l2_weights,
        cumulative_local_expert_recv_stats,
        l1_global_scales, l2_global_scales,
        sym_buffer.buffer,
        sym_buffer.handle.buffer_ptrs, sym_buffer.group.rank(),
        sym_buffer.num_max_tokens_per_rank,
        sym_buffer.num_experts, sym_buffer.num_topk,
        recipe, activation, activation_clamp, fast_math,
        mode2_nibble_weights,
    )
