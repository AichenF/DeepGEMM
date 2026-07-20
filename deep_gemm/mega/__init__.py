import torch
from typing import Tuple, Optional
from ..utils.math import align

# noinspection PyBroadException
try:
    # noinspection PyProtectedMember
    import torch.distributed._symmetric_memory as symm_mem
    import torch.distributed as dist
except Exception as exception:
    print(f'Failed to load mega kernels, please check your PyTorch version: {exception}')

from .. import _C


_SM90_NVFP4_H200_MIMO_LAYOUT_ATTR = "_deep_gemm_nvfp4_h200_mimo_layout"
_SM90_NVFP4_H200_MIMO_LAYOUT = "mode2_braided"


class SymmBuffer:
    def __init__(self, group: dist.ProcessGroup,
                 # MoE arguments
                 num_experts: int,
                 num_max_tokens_per_rank: int, num_topk: int,
                 hidden: int, intermediate_hidden: int,
                 use_fp8_dispatch: bool = True,
                 activation: str = 'swiglu'):
        self.group = group
        self.num_experts = num_experts
        self.num_max_tokens_per_rank = num_max_tokens_per_rank
        self.num_topk = num_topk
        self.hidden = hidden
        self.intermediate_hidden = intermediate_hidden

        # Allocate a symmetric buffer
        num_bytes, slice_input_buffers = _C.get_symm_buffer_size_for_mega_moe(
            group.size(), num_experts,
            num_max_tokens_per_rank, num_topk,
            hidden, intermediate_hidden,
            use_fp8_dispatch, activation
        )
        self.buffer = symm_mem.empty(num_bytes, dtype=torch.int8, device='cuda')
        self.handle = symm_mem.rendezvous(self.buffer, group=group)
        self.buffer.zero_()
        self.group.barrier()
        torch.cuda.synchronize()

        # Create input buffer views
        (self.x, self.x_sf,
         self.topk_idx, self.topk_weights,
         self.l1_acts, self.l1_acts_sf,
         self.l2_acts, self.l2_acts_sf) = slice_input_buffers(self.buffer)

    def destroy(self):
        self.handle = None
        self.buffer = None
        self.group = None
        self.x = None
        self.x_sf = None


def get_symm_buffer_for_mega_moe(group: dist.ProcessGroup,
                                 num_experts: int,
                                 num_max_tokens_per_rank: int, num_topk: int,
                                 hidden: int, intermediate_hidden: int,
                                 use_fp8_dispatch: bool = True,
                                 activation: str = 'swiglu') -> SymmBuffer:
    # Token count must be aligned to block sizes
    num_max_tokens_per_rank = align(num_max_tokens_per_rank, _C.get_token_alignment_for_mega_moe())

    return SymmBuffer(
        group, num_experts,
        num_max_tokens_per_rank, num_topk,
        hidden, intermediate_hidden,
        use_fp8_dispatch, activation
    )


def _interleave_l1_weights(l1_weights: Tuple[torch.Tensor, torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor]:
    # [gate: 0..7, up: 0..7, gate: 8..15, up: 8..15, ...] instead of [gate | up]
    def interleave(t, gran: int = 8) -> torch.Tensor:
        g, n, *rest = t.shape
        half = n // 2
        gate = t[:, :half].reshape(g, half // gran, gran, *rest)
        up = t[:, half:].reshape(g, half // gran, gran, *rest)
        return torch.empty_like(t).copy_(torch.stack([gate, up], dim=2).reshape(g, n, *rest))

    return interleave(l1_weights[0]), interleave(l1_weights[1])


def _transpose_sf_for_utccp(sf: torch.Tensor) -> torch.Tensor:
    num_groups, mn, packed_sf_k = sf.shape
    assert sf.dtype == torch.int and mn % 128 == 0
    result = (sf.reshape(num_groups, -1, 4, 32, packed_sf_k)
                .transpose(2, 3)
                .reshape(num_groups, mn, packed_sf_k))
    return torch.empty_like(sf).copy_(result)


def transform_weights_for_mega_moe(
    l1_weights: Tuple[torch.Tensor, torch.Tensor],
    l2_weights: Tuple[torch.Tensor, torch.Tensor]
) -> Tuple[Tuple[torch.Tensor, torch.Tensor], Tuple[torch.Tensor, torch.Tensor]]:
    # L1: interleave gate/up, then transpose SF for UTCCP
    l1_interleaved = _interleave_l1_weights(l1_weights)
    l1_weights = (l1_interleaved[0], _transpose_sf_for_utccp(l1_interleaved[1]))
    # L2: only transpose SF for UTCCP
    l2_weights = (l2_weights[0], _transpose_sf_for_utccp(l2_weights[1]))
    return l1_weights, l2_weights


def _braid_nvfp4_mode2_signs(fused_weight: torch.Tensor) -> torch.Tensor:
    """Arrange FP4 sign bits for the fused Mode2 row/LUT decoders."""
    if fused_weight.dtype != torch.uint8 or fused_weight.dim() != 3:
        raise ValueError("fused NVFP4 weight must be a 3-D uint8 tensor")
    experts, rows, storage_k = fused_weight.shape
    if storage_k % 80 != 0:
        raise ValueError("fused NVFP4 K storage must contain 80-byte BK128 tiles")

    fused_rows = fused_weight.view(experts, rows, storage_k // 80, 80).clone()
    packed = fused_rows[..., :64].view(experts, rows, storage_k // 80, 16, 4)
    codes = torch.cat(((packed >> 4) & 0x0f, packed & 0x0f), dim=-1)
    magnitudes = codes & 0x07
    signs = codes >> 3
    braided_signs = torch.stack(
        (
            signs[..., 4], signs[..., 0], signs[..., 5], signs[..., 1],
            signs[..., 6], signs[..., 2], signs[..., 7], signs[..., 3],
        ),
        dim=-1,
    )
    braided_nibbles = magnitudes | (braided_signs << 3)
    fused_rows[..., :64] = (
        braided_nibbles[..., 0::2] | (braided_nibbles[..., 1::2] << 4)
    ).reshape(experts, rows, storage_k // 80, 64)
    return fused_rows.view(experts, rows, storage_k).contiguous()


def transform_nvfp4_weights_for_mega_moe_sm90(
    l1_weights: Tuple[torch.Tensor, torch.Tensor],
    l2_weights: Tuple[torch.Tensor, torch.Tensor],
) -> Tuple[Tuple[torch.Tensor, torch.Tensor], Tuple[torch.Tensor, torch.Tensor]]:
    """Prepack weights for the H200 MiMo small-M fused kernel.

    Input scale tensors are row-major ``(E, N, K/16)`` UE4M3. Returned scale
    tensors are BN256 tile-major. Packed E2M1 values use the single Mode2
    Braided physical layout consumed by both the row and LUT-window decoders.
    Cache the result at weight-load time rather than rebuilding it per request.
    """
    block_n, block_k, group_size = 256, 128, 16
    from ..quantization_nvfp4 import (
        nvfp4_fuse_packed_with_scale_tile_major,
        nvfp4_scale_to_tile_major,
    )
    l1_packed, l1_scale = l1_weights
    l2_packed, l2_scale = l2_weights
    assert l1_packed.dtype == torch.uint8 and l2_packed.dtype == torch.uint8
    assert l1_scale.dtype == torch.uint8 and l2_scale.dtype == torch.uint8
    assert l1_packed.dim() == 3 and l2_packed.dim() == 3
    assert l1_scale.dim() == 3 and l2_scale.dim() == 3

    l1_packed_il, l1_scale_il = _interleave_l1_weights((l1_packed, l1_scale))
    l1_scale_tm = nvfp4_scale_to_tile_major(l1_scale_il, block_n=block_n, block_k=block_k, group_size=group_size)
    l2_scale_tm = nvfp4_scale_to_tile_major(l2_scale, block_n=block_n, block_k=block_k, group_size=group_size)
    l1_packed_out = _braid_nvfp4_mode2_signs(nvfp4_fuse_packed_with_scale_tile_major(
        l1_packed_il.contiguous(), l1_scale_tm, block_k=block_k)
    )
    l2_packed_out = _braid_nvfp4_mode2_signs(nvfp4_fuse_packed_with_scale_tile_major(
        l2_packed.contiguous(), l2_scale_tm, block_k=block_k)
    )
    setattr(l1_packed_out, _SM90_NVFP4_H200_MIMO_LAYOUT_ATTR,
            _SM90_NVFP4_H200_MIMO_LAYOUT)
    setattr(l2_packed_out, _SM90_NVFP4_H200_MIMO_LAYOUT_ATTR,
            _SM90_NVFP4_H200_MIMO_LAYOUT)
    return (
        l1_packed_out,
        l1_scale_tm,
    ), (
        l2_packed_out,
        l2_scale_tm,
    )

def fp8_fp4_mega_moe(y: torch.Tensor,
                     l1_weights: Tuple[torch.Tensor, torch.Tensor],
                     l2_weights: Tuple[torch.Tensor, torch.Tensor],
                     sym_buffer: SymmBuffer,
                     cumulative_local_expert_recv_stats: Optional[torch.Tensor] = None,
                     recipe: Tuple[int, int, int] = (1, 1, 32),
                     activation: str = 'swiglu',
                     activation_clamp: Optional[float] = None,
                     fast_math: bool = True):
    _C.fp8_fp4_mega_moe(
        y,
        l1_weights, l2_weights,
        cumulative_local_expert_recv_stats,
        sym_buffer.buffer,
        sym_buffer.handle.buffer_ptrs, sym_buffer.group.rank(),
        sym_buffer.num_max_tokens_per_rank,
        sym_buffer.num_experts, sym_buffer.num_topk,
        recipe,
        activation, activation_clamp,
        fast_math
    )


def nvfp4_mega_moe(y: torch.Tensor,
                  l1_weights: Tuple[torch.Tensor, torch.Tensor],
                  l2_weights: Tuple[torch.Tensor, torch.Tensor],
                  sym_buffer: SymmBuffer,
                  cumulative_local_expert_recv_stats: Optional[torch.Tensor] = None,
                  l1_global_scales: Optional[torch.Tensor] = None,
                  l2_global_scales: Optional[torch.Tensor] = None,
                  recipe: Tuple[int, int, int] = (128, 128, 128),
                  activation: str = 'swiglu',
                  activation_clamp: Optional[float] = None,
                  fast_math: bool = True):
    """H200 MiMo small-M NVFP4 MegaMoE entry.

    Weights must come directly from
    ``transform_nvfp4_weights_for_mega_moe_sm90`` so their sign bits match the
    fused Mode2 Braided decoder. There is no generic or split-kernel fallback.
    """
    l1_layout = getattr(l1_weights[0], _SM90_NVFP4_H200_MIMO_LAYOUT_ATTR, None)
    l2_layout = getattr(l2_weights[0], _SM90_NVFP4_H200_MIMO_LAYOUT_ATTR, None)
    if l1_layout != _SM90_NVFP4_H200_MIMO_LAYOUT or l2_layout != l1_layout:
        raise ValueError(
            "NVFP4 weights must use the Mode2 Braided layout produced by "
            "transform_nvfp4_weights_for_mega_moe_sm90"
        )
    _C.nvfp4_mega_moe(
        y, l1_weights, l2_weights,
        cumulative_local_expert_recv_stats,
        l1_global_scales, l2_global_scales,
        sym_buffer.buffer,
        sym_buffer.handle.buffer_ptrs, sym_buffer.group.rank(),
        sym_buffer.num_max_tokens_per_rank,
        sym_buffer.num_experts, sym_buffer.num_topk,
        recipe, activation, activation_clamp, fast_math,
    )
