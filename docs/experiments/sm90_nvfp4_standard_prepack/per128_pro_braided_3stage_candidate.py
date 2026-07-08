"""Experimental per-128 three-stage Pro braided NVFP4 launch wrapper."""

from typing import Optional, Tuple

import torch

from deep_gemm import _C
from deep_gemm.mega import SymmBuffer
from pro_braided_candidate import (
    transform_nvfp4_weights_for_mega_moe_sm90_pro_braided,
)


def nvfp4_per128_pro_braided_3stage_mega_moe(
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
    _C.nvfp4_per128_pro_braided_3stage_mega_moe(
        y,
        l1_weights,
        l2_weights,
        cumulative_local_expert_recv_stats,
        l1_global_scales,
        l2_global_scales,
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
