"""Experimental lossless nibble-group NVFP4 prepack and launch wrapper."""

from typing import Optional, Tuple

import torch

from deep_gemm import _C
from deep_gemm.mega import SymmBuffer
from deep_gemm.mega import transform_nvfp4_weights_for_mega_moe_sm90 as _stable_transform


def _group_nibbles_by_half(fused_weight: torch.Tensor) -> torch.Tensor:
    assert fused_weight.dtype == torch.uint8 and fused_weight.dim() == 3
    experts, rows, storage_k = fused_weight.shape
    assert storage_k % 80 == 0
    k_blocks = storage_k // 80
    fused_rows = fused_weight.view(experts, rows, k_blocks, 80).clone()
    q_bytes = fused_rows[..., :64].view(experts, rows, k_blocks, 16, 4).to(torch.int32)

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
    return (
        _group_nibbles_by_half(transformed_l1[0]),
        transformed_l1[1],
    ), (
        _group_nibbles_by_half(transformed_l2[0]),
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
) -> None:
    _C.nvfp4_nibble_group_mega_moe(
        y, l1_weights, l2_weights,
        cumulative_local_expert_recv_stats,
        l1_global_scales, l2_global_scales,
        sym_buffer.buffer,
        sym_buffer.handle.buffer_ptrs, sym_buffer.group.rank(),
        sym_buffer.num_max_tokens_per_rank,
        sym_buffer.num_experts, sym_buffer.num_topk,
        recipe, activation, activation_clamp, fast_math,
    )
