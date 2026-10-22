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
import os
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


# Planner phase stamps and loop counters, in the order the kernel records
# them (``DG_EPLB_PROFILE=1`` when the kernels are built).
PLAN_STAMPS = ('entry', 'histogram', 'arrived', 'published', 'wait', 'setup',
               'greedy', 'prune', 'slots', 'tables', 'remap', 'end')
PLAN_COUNTERS = ('visits', 'picks', 'moves', 'prune_steps', 'extreme_moves', 'grid')


def plan_profile_enabled() -> bool:
    return os.environ.get('DG_EPLB_PROFILE', '0') == '1'


def format_plan_profile(prof: dict) -> str:
    ph = prof['phases']
    body = ' + '.join(f'{k} {ph[k]:.1f}' for k in PLAN_STAMPS[1:] if ph[k] >= 0.05)
    c = prof['counters']
    return (f'total {prof["total_us"]:.1f} us = {body} | compute (total - wait) '
            f'{prof["total_us"] - ph["wait"]:.1f} us | picks={c["picks"]} moves={c["moves"]} '
            f'prune={c["prune_steps"]} extreme={c["extreme_moves"]} grid={c["grid"]} @ {prof["ghz"]:.2f} GHz')


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
                 activation: str = 'swiglu',
                 symm: Optional[SymmBuffer] = None,
                 fuse_copy: Optional[bool] = None):
        """``fuse_copy`` (default on; env ``DG_EPLB_FUSE_COPY=0`` turns it off)
        makes the MegaMoE kernel copy the changed helper slots itself:
        ``prepare`` then runs only the planner and the remap, and ``forward``
        passes the copy list to the kernel, whose dispatch warps move the
        weights during the GEMM phase and whose weight loader waits per slot.
        Off, a separate copy kernel runs between the plan and the remap (the
        measurement mode). It can be toggled per call through the attribute
        of the same name."""
        assert num_helper_slots > 0
        assert num_helper_slots <= 32, 'the fused copy tracks helper slots in a 32-bit mask'
        if fuse_copy is None:
            fuse_copy = bool(int(os.environ.get('DG_EPLB_FUSE_COPY', '1')))
        self.fuse_copy = fuse_copy
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

        # The MegaMoE symmetric buffer is sized for physical slots. A model
        # with many MoE layers passes one shared buffer (`symm`) instead of
        # paying for a buffer per layer; the planner workspace and the weight
        # arena stay per instance.
        self._owns_symm = symm is None
        if symm is None:
            symm = get_symm_buffer_for_mega_moe(
                group, self.num_slots, num_max_tokens_per_rank, num_topk,
                hidden, intermediate_hidden, True, activation)
        assert symm.num_experts == self.num_slots, 'shared buffer must be sized for the physical slots'
        self.symm: SymmBuffer = symm
        self.num_max_tokens_per_rank = self.symm.num_max_tokens_per_rank

        # Planner workspace, also symmetric: histograms travel through it.
        layout = _C.get_mega_moe_eplb_workspace_layout(
            self.num_ranks, num_experts, num_helper_slots)
        self.ws_layout = layout
        # The device headers may be newer than the host extension (JIT kernels
        # pick up `layout/mega_moe_eplb.cuh` immediately, `_C` only after a
        # rebuild): derive the trailing words the kernels expect from the
        # offsets the extension reports and size the buffer for the larger
        # of the two.
        copy_done_words = 2 * num_helper_slots
        copy_done = int(layout.get('copy_done', (int(layout['copy_count']) + 1 + 1) // 2 * 2))
        remap_done = copy_done + copy_done_words
        num_words = (remap_done + 1 + 3) // 4 * 4
        layout = dict(layout, copy_done=copy_done, remap_done=remap_done,
                      num_bytes=max(int(layout['num_bytes']), num_words * 4))
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
        self.copy_done = words[layout['copy_done']:
                               layout['copy_done'] + 2 * num_helper_slots].view(num_helper_slots, 2)
        self.local_counter = words[layout['local_counter']:layout['local_counter'] + num_experts]
        # Per-forward host work is kept to the two launches: the symmetric
        # pointer lists (a property call into the handle each time), the
        # ``topk_idx[:n]`` views (one ``aten::slice`` per call) and the fused
        # copy argument tuple are all fixed after construction, so they are
        # built once and reused.
        self._ws_ptrs = list(self.ws_handle.buffer_ptrs)
        self._topk_views: Dict[int, torch.Tensor] = {}
        self._arena_ptrs = None
        self._fused_copy_args = None
        # Empty helper slots are -1; the buffer was zeroed. Before the first
        # plan every expert has exactly one copy, its home rank, with an
        # unbounded block budget, so a remap without a plan sends everything
        # home.
        self.slot_expert.fill_(-1)
        self.copies.fill_(-1)
        self.copies[:, 0, 0] = torch.arange(num_experts, device='cuda', dtype=torch.int32) // self.home_experts
        self.copies[:, 0, 1] = 0x7fffffff
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
            # Plane groups for the fused copy: what the L1 GEMM reads, then
            # what the L2 GEMM reads; each group has its own readiness counter.
            self._l1_plane_names = ['l1_packed', 'l1_sf', 'l1_global_scale']
            self._l2_plane_names = ['l2_packed', 'l2_sf', 'l2_global_scale']
            self._arena_ptrs = list(self.arena.handle.buffer_ptrs)
            self._fused_copy_args = None
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
        # The heuristic is pure arithmetic on fixed shapes; cache it per token
        # count so the per-layer host path does not re-enter the extension.
        cache = self.__dict__.setdefault('_block_m_cache', {})
        bm = cache.get(num_tokens)
        if bm is None:
            bm = cache[num_tokens] = _C.sm90_mxfp4_mega_moe_block_m(
                self.num_ranks, self.num_slots, self.num_topk,
                self.hidden, self.intermediate_hidden,
                self.num_max_tokens_per_rank, num_tokens)
        return bm

    def topk_view(self, num_tokens: int) -> torch.Tensor:
        """``self.symm.topk_idx[:num_tokens]``, cached per token count."""
        v = self._topk_views.get(num_tokens)
        if v is None:
            v = self._topk_views[num_tokens] = self.symm.topk_idx[:num_tokens]
        return v

    def plan(self, num_tokens: int, block_m: Optional[int] = None,
             topk_idx: Optional[torch.Tensor] = None) -> int:
        """Plan from the logical ``topk_idx`` already in ``self.symm.topk_idx``,
        or from ``topk_idx`` (a private int64 copy of the logical routing) so a
        caller can plan on a side stream after the shared buffer has moved on.

        All ranks must call this the same number of times. Returns the block
        size used, which ``remap`` needs.
        """
        if block_m is None:
            block_m = self.block_m(num_tokens)
        if topk_idx is None:
            topk_idx = self.topk_view(num_tokens)
        elif topk_idx.shape[0] != num_tokens:
            topk_idx = topk_idx[:num_tokens]
        self.generation += 1
        _C.mega_moe_eplb_plan(
            self.ws, self._ws_ptrs, self.rank,
            topk_idx,
            self.num_experts, self.num_helper_slots,
            block_m, self.open_gate_blocks, self.generation)
        return block_m

    def copy_weights(self):
        assert self.arena is not None, 'set_home_weights first'
        _C.mega_moe_eplb_copy_weights(
            self.ws, self.arena.buffer, self._arena_ptrs, self.rank,
            self.num_experts, self.num_helper_slots,
            self.arena.plane_specs(self._plane_names))

    def fused_copy_args(self) -> tuple:
        """The extra arguments ``mxfp4_mega_moe`` needs to perform this step's
        weight copy itself (``eplb_fused_copy=``). The kernel consumes the
        planner's copy list; a list already applied by ``copy_weights`` must
        be cleared first (``copy_count.zero_()``) or it is copied again."""
        assert self.arena is not None, 'set_home_weights first'
        args = self._fused_copy_args
        if args is None:
            # The kernel binding sees the physical slot count as its expert
            # count, so the logical count is passed alongside. Everything in
            # the tuple is fixed once the arena exists.
            args = self._fused_copy_args = (
                self.ws, self.arena.buffer, self._arena_ptrs,
                self.num_experts, self.num_helper_slots,
                self.arena.plane_specs(self._l1_plane_names),
                self.arena.plane_specs(self._l2_plane_names))
        return args

    def remap(self, num_tokens: int, block_m: int):
        """Rewrite ``self.symm.topk_idx[:num_tokens]`` from logical experts
        to physical slots using the current placement. Standalone (without a
        fresh plan, e.g. decode steps) it reuses the last plan's copies and
        block split, which is correct for any placement. The ordinal counters
        are left at zero by both the planner and the remap kernel itself (its
        last CTA clears them), so no fill launch precedes the remap."""
        _C.mega_moe_eplb_remap(
            self.ws, self.rank, self.num_ranks,
            self.topk_view(num_tokens),
            self.num_experts, self.num_helper_slots, block_m)

    def prepare(self, num_tokens: int, block_m: Optional[int] = None) -> int:
        """Plan and remap in one launch, then copy the weights unless the
        kernel does it (``fuse_copy``). The synchronous path of a large
        forward; a plan issued early on a side stream uses ``plan`` and
        ``remap`` separately."""
        if block_m is None:
            block_m = self.block_m(num_tokens)
        self.generation += 1
        _C.mega_moe_eplb_plan_and_remap(
            self.ws, self._ws_ptrs, self.rank,
            self.topk_view(num_tokens),
            self.num_experts, self.num_helper_slots,
            block_m, self.open_gate_blocks, self.generation)
        if not self.fuse_copy:
            self.copy_weights()
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
            activation_clamp=activation_clamp, fast_math=fast_math,
            eplb_fused_copy=self.fused_copy_args() if self.fuse_copy else None)
        return y

    def plan_profile(self, generation: Optional[int] = None) -> Optional[dict]:
        """The planner's timing record of ``generation`` (default: the last
        plan): phase durations in microseconds (SM clock cycles converted
        with the globaltimer span of the same launch), the globaltimer stamps
        in ns and the loop counters. Needs ``DG_EPLB_PROFILE=1`` when the
        kernels were built and a ``_C`` that reports the ``profile`` region;
        returns None otherwise. Reads the workspace back (synchronises)."""
        layout = self.ws_layout
        if 'profile' not in layout:
            return None
        gen = self.generation if generation is None else generation
        slots, words = int(layout['num_profile_slots']), int(layout['num_profile_words'])
        base = int(layout['profile']) + (gen % slots) * words
        rec = self.ws.view(torch.int32)[base:base + words].cpu().contiguous()
        u64 = rec.view(torch.int64)
        n = len(PLAN_STAMPS)
        clocks = u64[:n].tolist()
        t_entry, t_peers, t_end = u64[n].item(), u64[n + 1].item(), u64[n + 2].item()
        counters = rec[2 * n + 6:2 * n + 6 + len(PLAN_COUNTERS)].tolist()
        span_cycles, span_ns = clocks[-1] - clocks[0], t_end - t_entry
        if span_cycles <= 0 or span_ns <= 0:
            return None
        ghz = span_cycles / span_ns
        phases = {PLAN_STAMPS[i]: (clocks[i] - clocks[i - 1]) / ghz / 1e3 for i in range(1, n)}
        return dict(phases=phases, total_us=span_cycles / ghz / 1e3, ghz=ghz,
                    timer_entry_ns=t_entry, timer_peers_ns=t_peers, timer_end_ns=t_end,
                    counters=dict(zip(PLAN_COUNTERS, counters)))

    def set_replicas(self, slot_expert) -> int:
        """Install a static replica placement.

        ``slot_expert[r][s]`` is the logical expert held by helper slot ``s``
        of rank ``r`` (-1 = empty); every rank must pass the same table. The
        changed slots of this rank fetch their weights from the owners, and
        the routing falls back to "every expert at its owner" until the next
        ``plan``. Returns the number of slots this rank copied.
        """
        table = torch.as_tensor(slot_expert, dtype=torch.int32).view(self.num_ranks, self.num_helper_slots)
        prev = self.slot_expert.cpu()
        changed = [(s, int(table[self.rank, s])) for s in range(self.num_helper_slots)
                   if int(table[self.rank, s]) >= 0 and int(table[self.rank, s]) != int(prev[self.rank, s])]
        self.slot_expert.copy_(table.to(self.slot_expert.device))
        self.copy_list.zero_()
        for i, (s, e) in enumerate(changed):
            self.copy_list[i, 0] = s
            self.copy_list[i, 1] = e
        self.copy_count.fill_(len(changed))
        self.copies.fill_(-1)
        self.copies[:, 0, 0] = torch.arange(self.num_experts, device='cuda', dtype=torch.int32) // self.home_experts
        self.copies[:, 0, 1] = 0x7fffffff
        if changed:
            self.copy_weights()
        # The list is applied; a fused-copy kernel launch must not redo it.
        self.copy_count.zero_()
        self.generation += 1
        torch.cuda.synchronize()
        self.group.barrier()
        return len(changed)

    def destroy(self):
        if self.arena is not None:
            self.arena.destroy()
            self.arena = None
        if self._owns_symm:
            self.symm.destroy()
        self.ws_handle = None
        self.ws = None
