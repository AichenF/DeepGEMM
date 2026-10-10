"""Dynamic expert load balancing for the SM90 MXFP4 MegaMoE kernel.

Each rank keeps its ``H`` home experts in local slots ``[0, H)`` and ``S``
helper slots in ``[H, H + S)``. Every iteration the planner (one kernel per
rank, exchanging histograms over symmetric memory) decides which experts the
helper slots hold and how each expert's token blocks split across its copies;
changed slots receive their weights from the owner over NVLink and the
routing table is rewritten to physical slot ids. The MegaMoE kernel itself
runs unchanged on ``R * (H + S)`` physical slots. Correctness never depends
on the plan: any slot holding an expert's weights computes the same output.
"""
from typing import Dict, List, Optional, Sequence, Tuple

import torch
import torch.distributed as dist

from .. import _C
from . import (
    SymmBuffer,
    _SM90_MXFP4_H20_FUSED_LAYOUT,
    _SM90_MXFP4_H20_FUSED_LAYOUT_ATTR,
    get_symm_buffer_for_mega_moe,
    mxfp4_mega_moe,
)

try:
    # noinspection PyProtectedMember
    import torch.distributed._symmetric_memory as symm_mem
except Exception as exception:  # pragma: no cover
    print(f'Failed to load symmetric memory, please check your PyTorch version: {exception}')


class _SymmArena:
    """One symmetric allocation holding several per-slot planes.

    ``planes`` maps a name to ``(shape_per_slot, dtype)``; every plane gets
    ``num_slots`` slots laid out contiguously, 256-byte aligned.
    """

    def __init__(self, group: dist.ProcessGroup, num_slots: int,
                 planes: Dict[str, Tuple[Tuple[int, ...], torch.dtype]]):
        self.group = group
        self.num_slots = num_slots
        self.layout: Dict[str, Tuple[int, int]] = {}
        offset = 0
        for name, (shape, dtype) in planes.items():
            bytes_per_slot = int(torch.empty(shape, dtype=dtype, device='meta').nbytes)
            assert bytes_per_slot % 4 == 0, f'{name}: per-slot bytes must be a multiple of 4'
            offset = (offset + 255) // 256 * 256
            self.layout[name] = (offset, bytes_per_slot)
            offset += bytes_per_slot * num_slots
        self.buffer = symm_mem.empty(max(offset, 256), dtype=torch.int8, device='cuda')
        self.handle = symm_mem.rendezvous(self.buffer, group=group)
        self.buffer.zero_()
        self.views: Dict[str, torch.Tensor] = {}
        for name, (shape, dtype) in planes.items():
            offset, bytes_per_slot = self.layout[name]
            flat = self.buffer[offset:offset + bytes_per_slot * num_slots]
            self.views[name] = flat.view(dtype).view(num_slots, *shape)
        group.barrier()
        torch.cuda.synchronize()

    def plane_specs(self, names: Sequence[str]) -> List[Tuple[int, int]]:
        return [self.layout[name] for name in names]

    def destroy(self):
        self.views = {}
        self.handle = None
        self.buffer = None


class DynamicEPLB:
    """Per-layer state for dynamic expert load balancing.

    Typical use::

        eplb = DynamicEPLB(group, num_experts, num_helper_slots, ...)
        eplb.set_home_weights(l1, l2, l1_global_scales, l2_global_scales)
        ...
        y = eplb.forward(x_fp8, x_sf, topk_idx, topk_weights)

    ``topk_idx`` is logical. ``forward`` plans, copies weights, remaps and
    launches the kernel on the current stream; the three stages are also
    exposed separately for callers that overlap them with other work.
    """

    def __init__(self, group: dist.ProcessGroup,
                 num_experts: int, num_helper_slots: int,
                 num_max_tokens_per_rank: int, num_topk: int,
                 hidden: int, intermediate_hidden: int,
                 open_gate_blocks: int = 0,
                 activation: str = 'swiglu'):
        assert num_helper_slots > 0
        self.group = group
        self.num_ranks = group.size()
        self.rank = group.rank()
        assert num_experts % self.num_ranks == 0
        self.num_experts = num_experts
        self.num_helper_slots = num_helper_slots
        self.home_experts = num_experts // self.num_ranks
        self.slots_per_rank = self.home_experts + num_helper_slots
        self.num_slots = self.slots_per_rank * self.num_ranks
        self.num_topk = num_topk
        self.hidden = hidden
        self.intermediate_hidden = intermediate_hidden
        self.open_gate_blocks = open_gate_blocks
        self.generation = 0

        # The MegaMoE symmetric buffer is sized for physical slots.
        self.symm: SymmBuffer = get_symm_buffer_for_mega_moe(
            group, self.num_slots, num_max_tokens_per_rank, num_topk,
            hidden, intermediate_hidden, True, activation)
        self.num_max_tokens_per_rank = self.symm.num_max_tokens_per_rank

        # Planner workspace, also symmetric: histograms travel through it.
        layout = _C.get_mega_moe_eplb_workspace_layout(
            self.num_ranks, num_experts, num_helper_slots)
        self.ws_layout = layout
        self.ws = symm_mem.empty(int(layout['num_bytes']), dtype=torch.int8, device='cuda')
        self.ws_handle = symm_mem.rendezvous(self.ws, group=group)
        self.ws.zero_()
        words = self.ws.view(torch.int32)
        self.slot_expert = words[layout['slot_expert']:
                                 layout['slot_expert'] + self.num_ranks * num_helper_slots] \
            .view(self.num_ranks, num_helper_slots)
        self.copies = words[layout['copies']:
                            layout['copies'] + 2 * num_experts * self.num_ranks] \
            .view(num_experts, self.num_ranks, 2)
        self.copy_list = words[layout['copy_list']:
                               layout['copy_list'] + 2 * num_helper_slots].view(num_helper_slots, 2)
        self.copy_count = words[layout['copy_count']:layout['copy_count'] + 1]
        # Empty helper slots are -1; the buffer was zeroed.
        self.slot_expert.fill_(-1)
        group.barrier()
        torch.cuda.synchronize()

        self.arena: Optional[_SymmArena] = None
        self.l1_weights = None
        self.l2_weights = None
        self.l1_global_scales = None
        self.l2_global_scales = None
        self.cumulative_slot_recv_stats = torch.zeros(
            self.slots_per_rank, dtype=torch.int, device='cuda')

    def set_home_weights(self,
                         l1_weights: Tuple[torch.Tensor, torch.Tensor],
                         l2_weights: Tuple[torch.Tensor, torch.Tensor],
                         l1_global_scales: Optional[torch.Tensor] = None,
                         l2_global_scales: Optional[torch.Tensor] = None):
        """Install this rank's transformed MXFP4 weights (`H` experts each).

        Takes the output of ``transform_mxfp4_weights_for_mega_moe_sm90``.
        The arena is allocated on first call; later calls refresh the home
        slots in place and invalidate every helper slot.
        """
        l1_packed, l1_sf = l1_weights
        l2_packed, l2_sf = l2_weights
        if getattr(l1_packed, _SM90_MXFP4_H20_FUSED_LAYOUT_ATTR, None) != _SM90_MXFP4_H20_FUSED_LAYOUT:
            raise ValueError('weights must come from transform_mxfp4_weights_for_mega_moe_sm90')
        H = self.home_experts
        for t in (l1_packed, l1_sf, l2_packed, l2_sf):
            assert t.shape[0] == H and t.is_contiguous() and t.dtype == torch.uint8
        use_global = l1_global_scales is not None or l2_global_scales is not None
        if use_global:
            assert l1_global_scales is not None and l2_global_scales is not None
            assert l1_global_scales.numel() == H and l2_global_scales.numel() == H
        if self.arena is None:
            planes = {
                'l1_packed': (tuple(l1_packed.shape[1:]), torch.uint8),
                'l1_sf': (tuple(l1_sf.shape[1:]), torch.uint8),
                'l2_packed': (tuple(l2_packed.shape[1:]), torch.uint8),
                'l2_sf': (tuple(l2_sf.shape[1:]), torch.uint8),
                'l1_global_scale': ((1,), torch.float32),
                'l2_global_scale': ((1,), torch.float32),
            }
            self.arena = _SymmArena(self.group, self.slots_per_rank, planes)
            v = self.arena.views
            self.l1_weights = (v['l1_packed'], v['l1_sf'])
            self.l2_weights = (v['l2_packed'], v['l2_sf'])
            setattr(v['l1_packed'], _SM90_MXFP4_H20_FUSED_LAYOUT_ATTR, _SM90_MXFP4_H20_FUSED_LAYOUT)
            setattr(v['l2_packed'], _SM90_MXFP4_H20_FUSED_LAYOUT_ATTR, _SM90_MXFP4_H20_FUSED_LAYOUT)
            self.l1_global_scales = v['l1_global_scale'].view(-1)
            self.l2_global_scales = v['l2_global_scale'].view(-1)
            self._plane_names = list(planes.keys())
        v = self.arena.views
        v['l1_packed'][:H].copy_(l1_packed)
        v['l1_sf'][:H].copy_(l1_sf)
        v['l2_packed'][:H].copy_(l2_packed)
        v['l2_sf'][:H].copy_(l2_sf)
        self.l1_global_scales.fill_(1.0)
        self.l2_global_scales.fill_(1.0)
        if use_global:
            self.l1_global_scales[:H].copy_(l1_global_scales.float())
            self.l2_global_scales[:H].copy_(l2_global_scales.float())
        self.use_global_scales = use_global
        # Helper slots now hold stale bytes: force a re-copy on the next plan.
        self.slot_expert.fill_(-1)
        torch.cuda.synchronize()
        self.group.barrier()

    def block_m(self, num_tokens: int) -> int:
        return _C.sm90_mxfp4_mega_moe_block_m(
            self.num_ranks, self.num_slots, self.num_topk,
            self.hidden, self.intermediate_hidden,
            self.num_max_tokens_per_rank, num_tokens)

    def plan(self, num_tokens: int, block_m: Optional[int] = None) -> int:
        """Plan from the logical ``topk_idx`` already in ``self.symm.topk_idx``.

        All ranks must call this the same number of times. Returns the block
        size used, which ``remap`` needs.
        """
        if block_m is None:
            block_m = self.block_m(num_tokens)
        self.generation += 1
        _C.mega_moe_eplb_plan(
            self.ws, self.ws_handle.buffer_ptrs, self.rank,
            self.symm.topk_idx[:num_tokens],
            self.num_experts, self.num_helper_slots,
            block_m, self.open_gate_blocks, self.generation)
        return block_m

    def copy_weights(self):
        assert self.arena is not None, 'set_home_weights first'
        _C.mega_moe_eplb_copy_weights(
            self.ws, self.arena.buffer, self.arena.handle.buffer_ptrs, self.rank,
            self.num_experts, self.num_helper_slots,
            self.arena.plane_specs(self._plane_names))

    def remap(self, num_tokens: int, block_m: int):
        _C.mega_moe_eplb_remap(
            self.ws, self.rank, self.num_ranks,
            self.symm.topk_idx[:num_tokens],
            self.num_experts, self.num_helper_slots, block_m)

    def prepare(self, num_tokens: int, block_m: Optional[int] = None) -> int:
        block_m = self.plan(num_tokens, block_m)
        self.copy_weights()
        self.remap(num_tokens, block_m)
        return block_m

    def forward(self, x_fp8: torch.Tensor, x_sf: torch.Tensor,
                topk_idx: torch.Tensor, topk_weights: torch.Tensor,
                y: Optional[torch.Tensor] = None,
                activation_clamp: Optional[float] = None,
                fast_math: bool = True,
                block_m: Optional[int] = None) -> torch.Tensor:
        num_tokens = x_fp8.shape[0]
        assert num_tokens <= self.num_max_tokens_per_rank
        b = self.symm
        b.x[:num_tokens].copy_(x_fp8)
        b.x_sf[:num_tokens].copy_(x_sf)
        b.topk_idx[:num_tokens].copy_(topk_idx.to(torch.int64))
        b.topk_weights[:num_tokens].copy_(topk_weights.to(torch.float32))
        self.prepare(num_tokens, block_m)
        if y is None:
            y = torch.empty((num_tokens, self.hidden), dtype=torch.bfloat16, device=x_fp8.device)
        mxfp4_mega_moe(
            y, self.l1_weights, self.l2_weights, b,
            cumulative_local_expert_recv_stats=self.cumulative_slot_recv_stats,
            l1_global_scales=self.l1_global_scales if self.use_global_scales else None,
            l2_global_scales=self.l2_global_scales if self.use_global_scales else None,
            activation_clamp=activation_clamp, fast_math=fast_math)
        return y

    def destroy(self):
        if self.arena is not None:
            self.arena.destroy()
            self.arena = None
        self.symm.destroy()
        self.ws_handle = None
        self.ws = None
