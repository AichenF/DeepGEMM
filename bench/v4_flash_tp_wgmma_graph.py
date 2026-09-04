#!/usr/bin/env python3
"""CUDA-Graph benchmark for the route-aware V4 Flash TP WGMMA pipeline."""

from __future__ import annotations

import argparse
import atexit
import json
import logging
import os
import statistics
from dataclasses import dataclass
from typing import Any

import torch
import torch.distributed as dist
from triton import runtime as triton_runtime

import sglang.srt.distributed.parallel_state as ps
from humming import ops as humming_ops
try:
    from sglang.jit_kernel.mp import register_comm_cleanup
except ImportError:  # Compatibility with the older benchmark checkout.
    from sglang.kernels.ops.communication.mp import register_comm_cleanup
from sglang.kernels.ops.moe.moe_fused_mul_sum import moe_fused_mul_sum
from sglang.srt.distributed.device_communicators.custom_all_reduce_v2 import (
    CustomAllReduceV2,
)
from sglang.srt.layers.moe.fused_moe_triton import moe_align_block_size

import v4_flash_tp_wgmma as kernel


HIDDEN = 4096
INTERMEDIATE = 2048
NUM_EXPERTS = 256
TOP_K = 6
ROUTED_SCALING_FACTOR = 1.5
DEFAULT_MS = (8, 16, 32, 64, 128)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ms", default=",".join(map(str, DEFAULT_MS)))
    parser.add_argument(
        "--route-pattern",
        choices=("random", "balanced", "skew"),
        default="random",
        help=(
            "Precomputed route distribution. random follows DeepGEMM MegaMoE's "
            "random-scores/top-k construction; router computation is not timed."
        ),
    )
    parser.add_argument("--outer", type=int, default=7)
    parser.add_argument("--replays", type=int, default=100)
    parser.add_argument("--warmup-replays", type=int, default=10)
    parser.add_argument(
        "--profile-once",
        action="store_true",
        help="Expose one explicitly cold graph replay to CUDA profiler APIs.",
    )
    parser.add_argument("--seed", type=int, default=20260902)
    args = parser.parse_args()
    args.ms = tuple(int(value) for value in args.ms.split(",") if value)
    if not args.ms or any(value <= 0 for value in args.ms):
        parser.error("--ms must contain positive integers")
    if args.outer < 1 or args.replays < 1 or args.warmup_replays < 1:
        parser.error("timing loop counts must be positive")
    if args.profile_once and len(args.ms) != 1:
        parser.error("--profile-once requires exactly one M value")
    return args


def init_distributed() -> tuple[int, int, torch.device, dist.ProcessGroup]:
    local_rank = int(os.environ["LOCAL_RANK"])
    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    if world_size not in (4, 8):
        raise ValueError(f"Expected TP4 or TP8, got TP{world_size}")
    torch.cuda.set_device(local_rank)
    dist.init_process_group(backend="gloo")
    atexit.register(dist.destroy_process_group)
    ps._WORLD = coordinator = ps.init_world_group(
        ranks=list(range(world_size)), local_rank=local_rank, backend="nccl"
    )
    cpu_group = coordinator.cpu_group
    if not isinstance(cpu_group, dist.ProcessGroup):
        raise RuntimeError("SGLang did not create the CPU process group")
    device = torch.device(f"cuda:{local_rank}")
    stream = torch.cuda.Stream(device=device)
    torch.cuda.set_stream(stream)
    logging.disable(logging.INFO)
    return rank, world_size, device, cpu_group


def make_routes(
    m: int, pattern: str, device: torch.device, seed: int
) -> tuple[torch.Tensor, torch.Tensor]:
    if pattern == "random":
        # Match DeepGEMM's MegaMoE benchmark route construction, but use an
        # isolated CPU generator so every TP rank receives identical routes.
        generator = torch.Generator(device="cpu")
        generator.manual_seed(seed)
        scores = torch.randn(
            (m, NUM_EXPERTS), dtype=torch.float32, generator=generator
        )
        ids = torch.topk(
            scores, TOP_K, dim=-1, largest=True, sorted=False
        ).indices.to(torch.int32)
    elif pattern == "balanced":
        ids = torch.arange(m * TOP_K, dtype=torch.int32).view(m, TOP_K)
        ids.remainder_(NUM_EXPERTS)
    else:
        ids = torch.arange(TOP_K, dtype=torch.int32).repeat(m, 1)
    weights = torch.arange(1, TOP_K + 1, dtype=torch.float32).repeat(m, 1)
    weights /= weights.sum(dim=1, keepdim=True)
    return ids.to(device), weights.to(device)


def make_fp8_input(
    m: int, device: torch.device, seed: int
) -> tuple[torch.Tensor, torch.Tensor]:
    """Create replicated TP input and quantize it before the timed graph."""
    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed + m)
    x_bf16 = (
        torch.randn((m, HIDDEN), dtype=torch.float32, generator=generator)
        .mul_(0.1)
        .to(torch.bfloat16)
        .to(device)
    )
    qx = torch.empty(
        (m, HIDDEN), dtype=torch.float8_e4m3fn, device=device
    )
    qx, x_scale = humming_ops.quant_input(
        inputs=x_bf16,
        outputs=qx,
        dtype="float8e4m3",
        group_size=128,
        m_major_scale=False,
        scale_dtype="float32",
    )
    return qx, x_scale


def make_weights(
    intermediate_per_rank: int,
    device: torch.device,
    include_native: bool = False,
) -> tuple[torch.Tensor, ...]:
    n13 = 2 * intermediate_per_rank
    w13 = torch.randint(
        0,
        256,
        (NUM_EXPERTS, n13, HIDDEN // 2),
        dtype=torch.uint8,
        device=device,
    )
    s13 = torch.randint(
        125,
        129,
        (NUM_EXPERTS, n13, HIDDEN // 32),
        dtype=torch.uint8,
        device=device,
    )
    w2 = torch.randint(
        0,
        256,
        (NUM_EXPERTS, HIDDEN, intermediate_per_rank // 2),
        dtype=torch.uint8,
        device=device,
    )
    s2 = torch.randint(
        125,
        129,
        (NUM_EXPERTS, HIDDEN, intermediate_per_rank // 32),
        dtype=torch.uint8,
        device=device,
    )
    native_weights: tuple[torch.Tensor, ...] = ()
    if include_native:
        import v4_flash_tp_native_megamoe as native_kernel

        native_weights = native_kernel.transform_weights(w13, s13, w2, s2)
    # The checkpoint/Humming contract is canonical Marlin K8.  The inherited
    # custom route-GEMM core uses its older group32 nibble order internally.
    w13 = kernel.marlin_to_legacy_mxfp4(w13)
    w2 = kernel.marlin_to_legacy_mxfp4(w2)
    g13 = torch.empty(0, dtype=torch.float32, device=device)
    g2 = torch.empty(0, dtype=torch.float32, device=device)
    if kernel.NORMALIZED_WEIGHT_SCALE:
        s13, g13 = kernel.normalize_mxfp4_weight_scales_(w13, s13)
        s2, g2 = kernel.normalize_mxfp4_weight_scales_(w2, s2)
    if kernel.W13_PAIRED_WG:
        w13, s13 = kernel.pair_gate_up_weight_layout(w13, s13)
    if kernel.MODE2_BRAID:
        kernel.braid_mode2_(w13)
        kernel.braid_mode2_(w2)
    if kernel.TILED_WEIGHT_LAYOUT:
        w13, s13 = kernel.tile_mxfp4_weight_layout(w13, s13)
        w2, s2 = kernel.tile_mxfp4_weight_layout(w2, s2)
    return w13, s13, g13, w2, s2, g2, *native_weights


@dataclass
class CapturedCase:
    m: int
    qx: torch.Tensor
    x_scale: torch.Tensor
    topk_ids: torch.Tensor
    topk_weights: torch.Tensor
    w13: torch.Tensor
    s13: torch.Tensor
    g13: torch.Tensor
    w2: torch.Tensor
    s2: torch.Tensor
    g2: torch.Tensor
    lut: torch.Tensor
    intermediate_per_rank: int
    native_w13: torch.Tensor | None = None
    native_w2: torch.Tensor | None = None

    def __post_init__(self) -> None:
        device = self.qx.device
        if self.qx.dtype != torch.float8_e4m3fn or self.qx.shape != (
            self.m,
            HIDDEN,
        ):
            raise ValueError("qx must be FP8-E4M3 [M,4096]")
        if self.x_scale.dtype != torch.float32 or self.x_scale.shape != (
            self.m,
            HIDDEN // 128,
        ):
            raise ValueError("x_scale must be FP32 [M,32]")
        routes = self.m * TOP_K
        n13 = 2 * self.intermediate_per_rank
        max_padded = (
            routes * 8
            if routes < NUM_EXPERTS + 1
            else routes + (NUM_EXPERTS + 1) * 7
        )
        max_mblocks = (max_padded + 7) // 8
        w2_activation_rows = max_mblocks * 8 if kernel.W2_SORTED_ACT else routes
        self.partials = torch.empty(
            (kernel.W13_MAX_SPLITS, routes, n13),
            dtype=torch.float32,
            device=device,
        )
        self.paired_raw = torch.empty(0, dtype=torch.float32, device=device)
        self.activation = (
            torch.empty(0, dtype=torch.bfloat16, device=device)
            if kernel.FUSED_ACT_QUANT
            else torch.empty(
                (routes, self.intermediate_per_rank),
                dtype=torch.bfloat16,
                device=device,
            )
        )
        self.qactivation = torch.empty(
            (w2_activation_rows, self.intermediate_per_rank),
            dtype=torch.float8_e4m3fn,
            device=device,
        )
        self.local_float = (
            None
            if kernel.W2_ROUTE_OUTPUT
            else torch.empty((self.m, HIDDEN), dtype=torch.float32, device=device)
        )
        self.down = (
            torch.empty((routes, HIDDEN), dtype=torch.bfloat16, device=device)
            if kernel.W2_ROUTE_OUTPUT
            else None
        )
        self.local_bf16 = torch.empty(
            (self.m, HIDDEN), dtype=torch.bfloat16, device=device
        )
        self.fused_graph_output = torch.empty_like(self.local_bf16)
        self.fused_push_symm = None
        self.fused_push_workspaces = None
        self.fused_push_counter = None
        self.fused_push_rank = -1
        self.fused_push_stride = 0
        self.fused_push_mc_ptr = 0
        self.fused_pull_output = None
        self.fused_pull_mc_ptr = 0
        self.fused_pull_sem_local = None
        self.fused_pull_sem_mc_ptr = 0
        self.symm_route_input = None
        self.symm_route_mc_ptr = 0
        self.symm_route_sem_local = None
        self.fused_k6_push_active = False
        self.fused_k6_ar_mode = "stock"
        self.pipeline_stream = torch.cuda.Stream(device=device)
        self.pipeline_start_event = torch.cuda.Event()
        self.pipeline_chunk_events = tuple(
            torch.cuda.Event() for _ in range(8)
        )
        self.pipeline_done_event = torch.cuda.Event()
        # Routes are fixed benchmark inputs.  Inspect them once before graph
        # capture; this synchronization and split-K policy selection are not
        # timed.  Schedule-4 state sizing also needs the selected split here.
        self.active_experts = int(torch.unique(self.topk_ids).numel())
        self.w13_split_k = (
            1
            if kernel.W13_PAIRED_WG
            else kernel.select_w13_split_k(routes, self.active_experts)
        )
        # One allocation is reset by a single captured memset.  Layout is
        # one direct marker per route/N128 tile, followed by task-done and
        # worker-done diagnostic scalars.
        self.w2_progress_state = torch.empty(
            (self.m * TOP_K * 32 + 2,), dtype=torch.int32, device=device
        )
        # Four count/epoch pairs and five uint64 device timestamps precede the
        # optional scheduler suffix.  The suffix is reset inside the same
        # business kernel and stores the
        # W13->activation->W2 task-DAG counters/readiness queues.  No captured
        # memset or additional launch is part of the single-launch path.
        oversubscribed_grid = (
            max_mblocks * 8 * self.w13_split_k
            + routes * 4
            + max_mblocks * 32
            + (64 if self.m == 128 else 78)
        )
        scheduler_words = (
            (484 if kernel.SINGLE_LAUNCH_SHARDED_TURNOVER else 16)
            + oversubscribed_grid
            if kernel.SINGLE_LAUNCH_OVERSUBSCRIBED
            else 8 + 3 * max_mblocks
            if kernel.SINGLE_LAUNCH_INTERLEAVED
            else 624
            if kernel.SINGLE_LAUNCH_GROUPED_W13_ACT
            else 78
            if kernel.SINGLE_LAUNCH_TAIL_OVERLAP
            else 0
        )
        hierarchical_words = (
            4 * 78 * 2 if kernel.SINGLE_LAUNCH_HIERARCHICAL_GRID else 0
        )
        self.single_launch_barrier_state = torch.zeros(
            (18 + hierarchical_words + scheduler_words,),
            dtype=torch.int32,
            device=device,
        )
        self.sorted_ids = torch.empty(
            (max_padded,), dtype=torch.int32, device=device
        )
        self.expert_ids = torch.empty(
            (max_mblocks,), dtype=torch.int32, device=device
        )
        self.route_to_sorted = torch.empty(
            (routes if kernel.W2_NEEDS_ROUTE_MAP else 0,),
            dtype=torch.int32,
            device=device,
        )
        self.num_tokens_padded = torch.empty(
            (1,), dtype=torch.int32, device=device
        )
        self.activation_scale: torch.Tensor | None = (
            torch.empty(
                (
                    (max_mblocks, self.intermediate_per_rank // 128, 8)
                    if kernel.W2_SORTED_ACT or kernel.W2_MBLOCK_SCALE
                    else (routes, self.intermediate_per_rank // 128)
                ),
                dtype=torch.float32,
                device=device,
            )
            if kernel.FUSED_ACT_QUANT
            else None
        )
        self.graph_output: torch.Tensor | None = None
        self.native_workspace = None
        self.native_local_output = None
        if (self.native_w13 is None) != (self.native_w2 is None):
            raise ValueError("native W13 and W2 must be provided together")
        if self.native_w13 is not None:
            import v4_flash_tp_native_megamoe as native_kernel

            self.native_workspace = native_kernel.allocate_workspace(
                self.intermediate_per_rank, device
            )
            self.native_workspace.load_inputs(
                self.qx, self.x_scale, self.topk_ids, self.topk_weights
            )
            self.native_local_output = torch.empty(
                (self.m, HIDDEN), dtype=torch.bfloat16, device=device
            )
        self.tiled_k6_reduce_mode = kernel.select_tiled_k6_reduce_mode(self.m)

    @property
    def routes(self) -> int:
        return self.m * TOP_K

    def run_before_w2(self) -> None:
        if kernel.FUSED_ROUTE_ALIGN:
            kernel.route_align(
                self.topk_ids,
                self.sorted_ids,
                self.expert_ids,
                self.num_tokens_padded,
                self.route_to_sorted,
            )
        else:
            (
                self.sorted_ids,
                self.expert_ids,
                self.num_tokens_padded,
            ) = moe_align_block_size(
                topk_ids=self.topk_ids,
                block_size=8,
                num_experts=NUM_EXPERTS,
                ignore_invalid_expert=True,
            )
        if kernel.W13_PAIRED_WG:
            assert self.activation_scale is not None
            kernel.run_w13_paired(
                self.w13,
                self.g13,
                self.qx.view(torch.uint8),
                self.x_scale,
                self.sorted_ids,
                self.expert_ids,
                self.num_tokens_padded,
                self.paired_raw,
                self.activation,
                self.qactivation.view(torch.uint8),
                self.activation_scale,
                self.intermediate_per_rank,
            )
        else:
            kernel.run_w13(
                self.w13,
                self.s13,
                self.g13,
                self.qx.view(torch.uint8),
                self.x_scale,
                self.sorted_ids,
                self.expert_ids,
                self.num_tokens_padded,
                self.partials,
                self.lut,
                self.intermediate_per_rank,
                self.w13_split_k,
            )
            if kernel.FUSED_ACT_QUANT:
                assert self.activation_scale is not None
                kernel.reduce_swiglu_quant(
                    self.partials,
                    self.activation,
                    self.qactivation.view(torch.uint8),
                    self.activation_scale,
                    self.intermediate_per_rank,
                    self.w13_split_k,
                    self.route_to_sorted,
                    self.topk_ids,
                    self.g2,
                )
            else:
                kernel.reduce_swiglu(
                    self.partials,
                    self.activation,
                    self.intermediate_per_rank,
                    self.w13_split_k,
                )
                self.qactivation, self.activation_scale = humming_ops.quant_input(
                    inputs=self.activation,
                    outputs=self.qactivation,
                    dtype="float8e4m3",
                    group_size=128,
                    m_major_scale=False,
                    scale_dtype="float32",
                )

    def run_w2_full(self) -> None:
        if self.local_float is not None:
            self.local_float.zero_()
        w2_output = self.down if kernel.W2_ROUTE_OUTPUT else self.local_float
        assert w2_output is not None
        kernel.run_w2(
            self.w2,
            self.s2,
            self.g2,
            self.qactivation.view(torch.uint8),
            self.activation_scale,
            self.sorted_ids,
            self.expert_ids,
            self.num_tokens_padded,
            self.topk_weights,
            w2_output,
            self.lut,
            self.intermediate_per_rank,
        )

    def run_before_local_reduce(self) -> None:
        self.run_before_w2()
        self.run_w2_full()

    def reduce_local(self) -> torch.Tensor:
        if kernel.W2_ROUTE_OUTPUT:
            assert self.down is not None
            if self.tiled_k6_reduce_mode:
                kernel.tiled_k6_reduce(
                    self.down,
                    self.topk_weights,
                    self.local_bf16,
                    self.tiled_k6_reduce_mode,
                )
            else:
                moe_fused_mul_sum(
                    inputs=self.down.view(self.m, TOP_K, HIDDEN),
                    topk_weights=self.topk_weights,
                    topk_ids=self.topk_ids,
                    is_ep=False,
                    routed_scaling_factor=ROUTED_SCALING_FACTOR,
                    outputs=self.local_bf16,
                )
        else:
            assert self.local_float is not None
            kernel.cast_bf16(self.local_float, self.local_bf16)
        return self.local_bf16

    def run_local(self) -> torch.Tensor:
        self.run_before_local_reduce()
        return self.reduce_local()

    def prepare_fused_push(self, comm: CustomAllReduceV2) -> None:
        if self.fused_push_workspaces is not None:
            return
        from torch._C._distributed_c10d import _SymmetricMemory

        symm = _SymmetricMemory.rendezvous(comm._symm_tensor)
        total_bytes = comm._symm_tensor.numel()
        workspaces = tuple(
            symm.get_buffer(peer, [total_bytes], torch.uint8)
            for peer in range(comm.world_size)
        )
        if len(workspaces) != 4:
            raise RuntimeError("fused k6 push all-reduce requires TP4")
        self.fused_push_symm = symm
        self.fused_push_workspaces = workspaces
        self.fused_push_counter = comm._push_counter
        self.fused_push_rank = comm.rank
        self.fused_push_stride = comm.max_push_size
        self.fused_push_mc_ptr = int(symm.multicast_ptr)

    def run_pipelined_w2_mc_push(self, comm: CustomAllReduceV2) -> torch.Tensor:
        self.prepare_fused_push(comm)
        assert self.down is not None
        assert self.fused_push_workspaces is not None
        assert self.fused_push_counter is not None
        if not self.fused_push_mc_ptr:
            raise RuntimeError("pipelined W2 requires multicast symmetric memory")

        self.run_before_w2()
        main_stream = torch.cuda.current_stream(self.qx.device)
        self.pipeline_start_event.record(main_stream)
        with torch.cuda.stream(self.pipeline_stream):
            self.pipeline_stream.wait_event(self.pipeline_start_event)

        for chunk_idx in range(kernel.PIPELINE_CHUNKS):
            kernel.run_w2_chunk(
                self.w2,
                self.s2,
                self.g2,
                self.qactivation.view(torch.uint8),
                self.activation_scale,
                self.sorted_ids,
                self.expert_ids,
                self.num_tokens_padded,
                self.topk_weights,
                self.down,
                self.lut,
                self.intermediate_per_rank,
                kernel.PIPELINE_CHUNKS,
                chunk_idx,
            )
            self.pipeline_chunk_events[chunk_idx].record(main_stream)
            with torch.cuda.stream(self.pipeline_stream):
                self.pipeline_stream.wait_event(
                    self.pipeline_chunk_events[chunk_idx]
                )
                kernel.fused_k6_push_ar_tp4_chunk(
                    self.down,
                    self.topk_weights,
                    self.fused_graph_output,
                    self.fused_push_counter,
                    self.fused_push_workspaces,
                    self.fused_push_rank,
                    self.fused_push_stride,
                    self.fused_push_mc_ptr,
                    kernel.PIPELINE_CHUNKS,
                    chunk_idx,
                    kernel.PIPELINE_AR_BLOCKS,
                )

        with torch.cuda.stream(self.pipeline_stream):
            self.pipeline_done_event.record(self.pipeline_stream)
        main_stream.wait_event(self.pipeline_done_event)
        self.fused_k6_push_active = True
        self.fused_k6_ar_mode = "pipelined_w2_multicast_push"
        self.graph_output = self.fused_graph_output
        return self.graph_output

    def run_progress_w2_mc_push(self, comm: CustomAllReduceV2) -> torch.Tensor:
        self.prepare_fused_push(comm)
        assert self.down is not None
        assert self.activation_scale is not None
        assert self.fused_push_workspaces is not None
        assert self.fused_push_counter is not None
        if not self.fused_push_mc_ptr:
            raise RuntimeError("W2 progress path requires multicast memory")

        self.run_before_w2()
        self.w2_progress_state.zero_()
        main_stream = torch.cuda.current_stream(self.qx.device)
        self.pipeline_start_event.record(main_stream)
        kernel.run_w2_progress(
            self.w2,
            self.s2,
            self.g2,
            self.qactivation.view(torch.uint8),
            self.activation_scale,
            self.sorted_ids,
            self.expert_ids,
            self.num_tokens_padded,
            self.topk_weights,
            self.down,
            self.lut,
            self.w2_progress_state,
            self.intermediate_per_rank,
        )
        # Submit the finite W2 producer before the persistent polling
        # consumers.  The consumers depend only on the pre-W2 start event, so
        # they can overlap once scheduled without blocking producer admission.
        with torch.cuda.stream(self.pipeline_stream):
            self.pipeline_stream.wait_event(self.pipeline_start_event)
            kernel.progress_k6_mc_push_tp4(
                self.down,
                self.topk_weights,
                self.fused_graph_output,
                self.w2_progress_state,
                self.fused_push_counter,
                self.fused_push_workspaces,
                self.fused_push_mc_ptr,
                self.fused_push_rank,
                self.fused_push_stride,
                kernel.W2_PROGRESS_WORKERS,
            )
        with torch.cuda.stream(self.pipeline_stream):
            self.pipeline_done_event.record(self.pipeline_stream)
        main_stream.wait_event(self.pipeline_done_event)
        if not kernel.W2_PROGRESS_INLINE_FINISH:
            kernel.progress_mc_push_finish_tp4(
                self.fused_graph_output,
                self.fused_push_counter,
                self.fused_push_workspaces,
                self.fused_push_rank,
                self.fused_push_stride,
            )
        self.fused_k6_push_active = True
        self.fused_k6_ar_mode = (
            "w2_progress_inline_finish_multicast_push"
            if kernel.W2_PROGRESS_INLINE_FINISH
            else "w2_progress_multicast_push"
        )
        self.graph_output = self.fused_graph_output
        return self.graph_output

    def prepare_fused_pull(self, comm: CustomAllReduceV2) -> None:
        if self.fused_pull_output is not None:
            return
        self.prepare_fused_push(comm)
        assert self.fused_push_workspaces is not None
        pull_offset = 2 * comm.world_size * comm.max_push_size
        nbytes = self.m * HIDDEN * torch.bfloat16.itemsize
        if nbytes > comm.max_pull_size:
            raise RuntimeError("fused pull output exceeds symmetric workspace")
        local_slab = self.fused_push_workspaces[comm.rank]
        pull_bytes = local_slab[pull_offset : pull_offset + nbytes]
        self.fused_pull_output = pull_bytes.view(torch.bfloat16).view(
            self.m, HIDDEN
        )
        self.fused_pull_mc_ptr = self.fused_push_mc_ptr + pull_offset
        sem_offset = pull_offset + comm.max_pull_size
        sem_nbytes = comm.config.num_pull_blocks * 128
        self.fused_pull_sem_local = local_slab[
            sem_offset : sem_offset + sem_nbytes
        ]
        self.fused_pull_sem_mc_ptr = self.fused_push_mc_ptr + sem_offset

    def run_fused_k6_nvls_pull(self, comm: CustomAllReduceV2) -> torch.Tensor:
        self.prepare_fused_pull(comm)
        assert self.down is not None
        assert self.fused_pull_output is not None
        assert self.fused_pull_sem_local is not None
        if not self.fused_pull_mc_ptr or not self.fused_pull_sem_mc_ptr:
            raise RuntimeError("fused k6 NVLS pull requires multicast memory")

        self.run_before_local_reduce()
        kernel.fused_k6_nvls_pull_tp4(
            self.down,
            self.topk_weights,
            self.fused_pull_output,
            self.fused_graph_output,
            self.fused_pull_sem_local,
            self.fused_pull_mc_ptr,
            self.fused_pull_sem_mc_ptr,
            kernel.K6_NVLS_PULL_BLOCKS,
        )
        self.fused_k6_push_active = True
        self.fused_k6_ar_mode = "fused_k6_nvls_one_shot_pull"
        self.graph_output = self.fused_graph_output
        return self.graph_output

    def prepare_symm_route_pull(self, comm: CustomAllReduceV2) -> None:
        if self.symm_route_input is not None:
            return
        self.prepare_fused_push(comm)
        assert self.fused_push_workspaces is not None
        pull_offset = 2 * comm.world_size * comm.max_push_size
        route_nbytes = self.routes * HIDDEN * torch.bfloat16.itemsize
        if route_nbytes > comm.max_pull_size:
            raise RuntimeError("symmetric route tensor exceeds pull workspace")
        local_slab = self.fused_push_workspaces[comm.rank]
        self.symm_route_input = local_slab[
            pull_offset : pull_offset + route_nbytes
        ].view(torch.bfloat16).view(self.routes, HIDDEN)
        self.symm_route_mc_ptr = self.fused_push_mc_ptr + pull_offset
        sem_offset = pull_offset + comm.max_pull_size
        sem_nbytes = comm.config.num_pull_blocks * 128
        self.symm_route_sem_local = local_slab[
            sem_offset : sem_offset + sem_nbytes
        ]
        self.fused_pull_sem_mc_ptr = self.fused_push_mc_ptr + sem_offset

    def run_rank_route_mc_pull(self, comm: CustomAllReduceV2) -> torch.Tensor:
        self.prepare_symm_route_pull(comm)
        assert self.symm_route_input is not None
        assert self.symm_route_sem_local is not None
        if not self.symm_route_mc_ptr or not self.fused_pull_sem_mc_ptr:
            raise RuntimeError("rank-route pull requires multicast symmetric memory")

        self.run_before_w2()
        kernel.run_w2(
            self.w2,
            self.s2,
            self.g2,
            self.qactivation.view(torch.uint8),
            self.activation_scale,
            self.sorted_ids,
            self.expert_ids,
            self.num_tokens_padded,
            self.topk_weights,
            self.symm_route_input,
            self.lut,
            self.intermediate_per_rank,
        )
        kernel.fused_rank_route_mc_pull_tp4(
            self.symm_route_input,
            self.topk_weights,
            self.fused_graph_output,
            self.symm_route_sem_local,
            self.symm_route_mc_ptr,
            self.fused_pull_sem_mc_ptr,
            kernel.RANK_ROUTE_PULL_BLOCKS,
        )
        self.fused_k6_push_active = True
        self.fused_k6_ar_mode = "rank_route_multicast_pull"
        self.graph_output = self.fused_graph_output
        return self.graph_output

    def reduce_local_to(self, output: torch.Tensor) -> torch.Tensor:
        assert self.down is not None
        if self.tiled_k6_reduce_mode:
            kernel.tiled_k6_reduce(
                self.down,
                self.topk_weights,
                output,
                self.tiled_k6_reduce_mode,
            )
        else:
            moe_fused_mul_sum(
                inputs=self.down.view(self.m, TOP_K, HIDDEN),
                topk_weights=self.topk_weights,
                topk_ids=self.topk_ids,
                is_ep=False,
                routed_scaling_factor=ROUTED_SCALING_FACTOR,
                outputs=output,
            )
        return output

    def run_tp4_single_launch(self, comm: CustomAllReduceV2) -> torch.Tensor:
        if self.native_workspace is not None:
            return self.run_native_tp4_single_launch(comm)
        # The single entry selects push for M<128 and the communicator's
        # multicast-bound pull slab for M128.  Prepare both ABIs outside the
        # timed graph; no allocation or registration occurs during replay.
        self.prepare_fused_pull(comm)
        assert self.down is not None
        assert self.activation_scale is not None
        assert self.fused_push_workspaces is not None
        assert self.fused_push_counter is not None
        assert self.fused_pull_output is not None
        assert self.fused_pull_sem_local is not None
        if (
            comm.world_size != 4
            or not self.fused_push_mc_ptr
            or not self.fused_pull_mc_ptr
            or not self.fused_pull_sem_mc_ptr
        ):
            raise RuntimeError(
                "single-launch bring-up requires TP4 NVLS multicast memory"
            )
        kernel.run_tp4_megamoe_single_launch(
            self.w13,
            self.s13,
            self.g13,
            self.w2,
            self.s2,
            self.g2,
            self.qx,
            self.x_scale,
            self.topk_ids,
            self.topk_weights,
            self.sorted_ids,
            self.expert_ids,
            self.num_tokens_padded,
            self.partials,
            self.activation,
            self.qactivation,
            self.activation_scale,
            self.down,
            self.lut,
            self.single_launch_barrier_state,
            self.route_to_sorted,
            self.fused_graph_output,
            self.fused_push_counter,
            self.fused_push_workspaces,
            self.fused_pull_output,
            self.fused_pull_sem_local,
            self.fused_push_rank,
            self.fused_push_stride,
            self.fused_push_mc_ptr,
            self.fused_pull_mc_ptr,
            self.fused_pull_sem_mc_ptr,
            self.w13_split_k,
        )
        self.fused_k6_push_active = True
        self.fused_k6_ar_mode = (
            "single_launch_nvls_pull"
            if self.m == 128
            else "single_launch_multicast_push"
        )
        self.graph_output = self.fused_graph_output
        return self.graph_output

    def run_native_tp4_single_launch(
        self, comm: CustomAllReduceV2
    ) -> torch.Tensor:
        import v4_flash_tp_native_megamoe as native_kernel

        self.prepare_fused_pull(comm)
        assert self.native_workspace is not None
        assert self.native_w13 is not None and self.native_w2 is not None
        assert self.native_local_output is not None
        assert self.fused_push_workspaces is not None
        assert self.fused_push_counter is not None
        assert self.fused_pull_output is not None
        assert self.fused_pull_sem_local is not None
        if (
            comm.world_size != 4
            or not self.fused_push_mc_ptr
            or not self.fused_pull_mc_ptr
            or not self.fused_pull_sem_mc_ptr
        ):
            raise RuntimeError(
                "native single-launch bring-up requires TP4 NVLS multicast memory"
            )
        native_kernel.run_tp4(
            self.native_workspace,
            self.native_w13,
            self.native_w2,
            self.native_local_output,
            self.fused_graph_output,
            self.fused_push_counter,
            self.fused_push_workspaces,
            self.fused_pull_output,
            self.fused_pull_sem_local,
            self.fused_push_mc_ptr,
            self.fused_pull_mc_ptr,
            self.fused_pull_sem_mc_ptr,
            self.fused_push_rank,
            self.fused_push_stride,
            self.m,
        )
        self.fused_k6_push_active = True
        self.fused_k6_ar_mode = (
            "native_single_launch_nvls_pull"
            if self.m == 128
            else "native_single_launch_multicast_push"
        )
        self.graph_output = self.fused_graph_output
        return self.graph_output

    def run_full(self, comm: CustomAllReduceV2) -> torch.Tensor:
        if kernel.SINGLE_LAUNCH_TP4:
            return self.run_tp4_single_launch(comm)
        use_w2_progress = (
            kernel.W2_PROGRESS_MC_PUSH_AR
            and comm.world_size == 4
            and self.m <= 128
            and kernel.W2_ROUTE_OUTPUT
        )
        if use_w2_progress:
            return self.run_progress_w2_mc_push(comm)
        use_fused_k6_nvls_pull = (
            kernel.FUSED_K6_NVLS_PULL_AR
            and comm.world_size == 4
            and self.m <= 128
            and kernel.W2_ROUTE_OUTPUT
        )
        if use_fused_k6_nvls_pull:
            return self.run_fused_k6_nvls_pull(comm)
        use_rank_route_pull = (
            kernel.FUSED_RANK_ROUTE_MC_PULL_AR
            and comm.world_size == 4
            and self.m == 128
            and kernel.W2_ROUTE_OUTPUT
        )
        if use_rank_route_pull:
            return self.run_rank_route_mc_pull(comm)
        use_pipeline = (
            kernel.PIPELINED_W2_MC_PUSH_AR
            and comm.world_size == 4
            and self.m == 128
            and kernel.W2_ROUTE_OUTPUT
        )
        if use_pipeline:
            return self.run_pipelined_w2_mc_push(comm)
        use_mc_pull = (
            kernel.FUSED_K6_MC_PULL_AR
            and comm.world_size == 4
            and self.m <= 128
        )
        if use_mc_pull:
            self.prepare_fused_pull(comm)
            assert self.fused_pull_output is not None
            self.run_before_local_reduce()
            self.reduce_local_to(self.fused_pull_output)
            from sglang.kernels.ops.kimi_k3.all_reduce import (
                all_reduce_pull_res,
            )

            all_reduce_pull_res(
                comm.world_size,
                self.fused_pull_output,
                input_mc_ptr=self.fused_pull_mc_ptr,
                num_blocks=kernel.MC_PULL_BLOCKS or None,
                unroll=kernel.MC_PULL_UNROLL or None,
            )
            self.fused_k6_push_active = True
            self.fused_k6_ar_mode = "multicast_pull"
            self.graph_output = self.fused_pull_output
            return self.graph_output
        use_mc_push = (
            kernel.FUSED_K6_MC_PUSH_AR
            and comm.world_size == 4
            and self.m <= kernel.FUSED_K6_MC_PUSH_MAX_M
        )
        use_unicast_push = (
            kernel.FUSED_K6_PUSH_AR
            and comm.world_size == 4
            and self.m <= 32
        )
        if use_mc_push or use_unicast_push:
            self.prepare_fused_push(comm)
            assert self.down is not None
            assert self.fused_push_workspaces is not None
            assert self.fused_push_counter is not None
            self.run_before_local_reduce()
            kernel.fused_k6_push_ar_tp4(
                self.down,
                self.topk_weights,
                self.fused_graph_output,
                self.fused_push_counter,
                self.fused_push_workspaces,
                self.fused_push_rank,
                self.fused_push_stride,
                self.fused_push_mc_ptr if use_mc_push else 0,
            )
            self.fused_k6_push_active = True
            self.fused_k6_ar_mode = (
                "multicast_push" if use_mc_push else "unicast_push"
            )
            self.graph_output = self.fused_graph_output
            return self.graph_output
        self.fused_k6_push_active = False
        self.fused_k6_ar_mode = "stock"
        self.graph_output = comm.custom_all_reduce(self.run_local())
        return self.graph_output

    def make_reference_case(self) -> "CapturedCase":
        return CapturedCase(
            m=self.m,
            qx=self.qx,
            x_scale=self.x_scale,
            topk_ids=self.topk_ids,
            topk_weights=self.topk_weights,
            w13=self.w13,
            s13=self.s13,
            g13=self.g13,
            w2=self.w2,
            s2=self.s2,
            g2=self.g2,
            lut=self.lut,
            intermediate_per_rank=self.intermediate_per_rank,
        )


def reduce_rank_metric(
    value: float,
    op: dist.ReduceOp,
    device: torch.device,
    group: dist.ProcessGroup,
) -> float:
    tensor = torch.tensor(value, dtype=torch.float64, device=device)
    dist.all_reduce(tensor, op=op, group=group)
    return float(tensor.item())


def correctness_metrics(
    case: CapturedCase,
    graph: torch.cuda.CUDAGraph,
    nccl_group: dist.ProcessGroup,
    device: torch.device,
) -> dict[str, float | bool]:
    graph.replay()
    torch.cuda.synchronize(device)
    assert case.graph_output is not None
    actual = case.graph_output.clone()
    reference = case.make_reference_case().run_local().clone()
    dist.all_reduce(reference, group=nccl_group)
    torch.cuda.synchronize(device)

    actual_f = actual.double()
    reference_f = reference.double()
    diff = actual_f - reference_f
    cosine = float(
        torch.nn.functional.cosine_similarity(
            actual_f.flatten(), reference_f.flatten(), dim=0
        ).item()
    )
    rel_l2 = float(
        (torch.linalg.vector_norm(diff)
         / torch.linalg.vector_norm(reference_f).clamp_min(1e-40)).item()
    )
    cosine_min = reduce_rank_metric(
        cosine, dist.ReduceOp.MIN, device, nccl_group
    )
    rel_l2_max = reduce_rank_metric(
        rel_l2, dist.ReduceOp.MAX, device, nccl_group
    )
    finite = float(
        bool(torch.isfinite(actual).all()) and bool(torch.isfinite(reference).all())
    )
    finite_all = bool(
        reduce_rank_metric(finite, dist.ReduceOp.MIN, device, nccl_group)
    )
    ref_max = reference_f.abs().max().clamp_min(1e-40)
    max_abs = reduce_rank_metric(
        float(diff.abs().max()), dist.ReduceOp.MAX, device, nccl_group
    )
    max_abs_ratio = reduce_rank_metric(
        float(diff.abs().max() / ref_max),
        dist.ReduceOp.MAX,
        device,
        nccl_group,
    )
    return {
        "cosine_min_rank": cosine_min,
        "rel_l2_max_rank": rel_l2_max,
        "max_abs_max_rank": max_abs,
        "max_abs_over_ref_max_rank": max_abs_ratio,
        "finite_all_ranks": finite_all,
        "allreduce_ok": bool(
            finite_all and cosine_min >= 0.999 and rel_l2_max <= 0.02
        ),
    }


def time_graph(
    graph: torch.cuda.CUDAGraph,
    outer: int,
    replays: int,
    cpu_group: dist.ProcessGroup,
    nccl_group: dist.ProcessGroup,
    device: torch.device,
    l2_flush_buffer: torch.Tensor,
) -> tuple[list[float], list[float]]:
    samples: list[float] = []
    batch_medians: list[float] = []
    driver = triton_runtime.driver.active
    for _ in range(outer):
        dist.barrier(group=cpu_group)
        starts = [torch.cuda.Event(enable_timing=True) for _ in range(replays)]
        ends = [torch.cuda.Event(enable_timing=True) for _ in range(replays)]
        for replay_idx in range(replays):
            driver.clear_cache(l2_flush_buffer)
            starts[replay_idx].record()
            graph.replay()
            ends[replay_idx].record()
        ends[-1].synchronize()
        local_times = torch.tensor(
            [start.elapsed_time(end) for start, end in zip(starts, ends)],
            dtype=torch.float64,
            device=device,
        )
        dist.all_reduce(local_times, op=dist.ReduceOp.MAX, group=nccl_group)
        batch = [float(value) for value in local_times.cpu().tolist()]
        samples.extend(batch)
        batch_medians.append(statistics.median(batch))
    return samples, batch_medians


@torch.inference_mode()
def main() -> None:
    args = parse_args()
    rank, world_size, device, cpu_group = init_distributed()
    nccl_group = ps._WORLD.device_group
    if not isinstance(nccl_group, dist.ProcessGroup):
        raise RuntimeError("SGLang did not create the NCCL process group")
    props = torch.cuda.get_device_properties(device)
    if props.major != 9:
        raise RuntimeError("This WGMMA kernel requires Hopper/sm90")

    intermediate_per_rank = INTERMEDIATE // world_size
    torch.manual_seed(args.seed + rank)
    torch.cuda.manual_seed(args.seed + rank)
    w13, s13, g13, w2, s2, g2 = make_weights(intermediate_per_rank, device)
    lut = kernel.make_e2m1_e8m0_lut(device)

    comm = CustomAllReduceV2(cpu_group, device)
    if comm.disabled:
        raise RuntimeError("SGLang CustomAllReduceV2 is disabled")
    register_comm_cleanup(comm)
    l2_flush_buffer = triton_runtime.driver.active.get_empty_cache_for_benchmark()
    if l2_flush_buffer.nbytes < 2 * props.L2_cache_size:
        raise RuntimeError(
            f"L2 flush buffer ({l2_flush_buffer.nbytes}) is smaller than 2x "
            f"L2 ({props.L2_cache_size})"
        )

    if rank == 0:
        print(
            "CUSTOM_ENV "
            + json.dumps(
                {
                    "benchmark": "v4_flash_tp_route_wgmma_cuda_graph",
                    "gpu": props.name,
                    "sm_count": props.multi_processor_count,
                    "capability": f"{props.major}.{props.minor}",
                    "world_size": world_size,
                    "route_pattern": args.route_pattern,
                    "m_values": args.ms,
                    "outer": args.outer,
                    "replays_per_outer": args.replays,
                    "warmup_replays": args.warmup_replays,
                    "l2_policy": "cold; 256MiB Triton clear before every replay, clear excluded from events",
                    "l2_cache_bytes": props.L2_cache_size,
                    "l2_flush_bytes": l2_flush_buffer.nbytes,
                    "w13_split_policy": (
                        f"{kernel.W13_SPLIT_MODE}; routed_rows<=192 or "
                        "active_experts<=96 -> 4, else 2; selected before capture"
                        if kernel.W13_SPLIT_MODE == "auto"
                        else kernel.W13_SPLIT_MODE
                    ),
                    "output_tile_channels": kernel.WOUT,
                    "mxfp4_lut_rows": kernel.LUT_ROWS,
                    "scale_quad_reuse": kernel.SCALE_QUAD_REUSE,
                    "scale_buffers": kernel.SCALE_BUFFERS,
                    "weight_stages": kernel.WEIGHT_STAGES,
                    "weight_swizzle_bytes": kernel.WEIGHT_SWIZZLE,
                    "weight_common_address": kernel.WEIGHT_COMMON_ADDRESS,
                    "dequant_dp4a_hi": kernel.DEQUANT_DP4A_HI,
                    "dequant_dp4a_lo": kernel.DEQUANT_DP4A_LO,
                    "dequant_synth_lut": kernel.DEQUANT_SYNTH_LUT,
                    "normalized_weight_scale": kernel.NORMALIZED_WEIGHT_SCALE,
                    "normalized_shared_lut": kernel.NORMALIZED_SHARED_LUT,
                    "activation_evict_last": kernel.ACTIVATION_EVICT_LAST,
                    "predicated_padded_activation": (
                        kernel.PREDICATED_PADDED_ACTIVATION
                    ),
                    "tiled_weight_layout": kernel.TILED_WEIGHT_LAYOUT,
                    "bulk_weight_copy": kernel.BULK_WEIGHT_COPY,
                    "tma_cta_scope": kernel.TMA_CTA_SCOPE,
                    "weight_evict_first": kernel.WEIGHT_EVICT_FIRST,
                    "weight_policy_hoist": kernel.WEIGHT_POLICY_HOIST,
                    "weight_policy_constant": kernel.WEIGHT_POLICY_CONSTANT,
                    "w2_no_weight_evict_first": (
                        kernel.W2_NO_WEIGHT_EVICT_FIRST
                    ),
                    "interleaved_bulk_copy": kernel.INTERLEAVED_BULK_COPY,
                    "compact_interleaved_scale": (
                        kernel.COMPACT_INTERLEAVED_SCALE
                    ),
                    "mode2_braid": kernel.MODE2_BRAID,
                    "fused_activation_quant": kernel.FUSED_ACT_QUANT,
                    "input_contract": "FP8-E4M3 group128; input quant untimed",
                    "fused_route_align": kernel.FUSED_ROUTE_ALIGN,
                    "w2_sorted_activation": kernel.W2_SORTED_ACT,
                    "w2_mblock_scale": kernel.W2_MBLOCK_SCALE,
                    "w2_fold_global_scale": kernel.W2_FOLD_GLOBAL_SCALE,
                    "w2_coalesced_store": kernel.W2_COALESCED_STORE,
                    "w13_paired_wg": kernel.W13_PAIRED_WG,
                    "w2_global_lut": kernel.W2_GLOBAL_LUT,
                    "w2_s2r_prefetch": kernel.W2_S2R_PREFETCH,
                    "w13_s2r_prefetch": kernel.W13_S2R_PREFETCH,
                    "leader_mbar_wait": kernel.LEADER_MBAR_WAIT,
                    "direct_barrier_addr": kernel.DIRECT_BARRIER_ADDR,
                    "route_k_unroll2": kernel.ROUTE_K_UNROLL2,
                    "route_k_unroll4": kernel.ROUTE_K_UNROLL4,
                    "route_k_unroll8": kernel.ROUTE_K_UNROLL8,
                    "route_k_unroll8_split2": kernel.ROUTE_K_UNROLL8_SPLIT2,
                    "w13_k_unroll8_split2": kernel.W13_K_UNROLL8_SPLIT2,
                    "w13_k_unroll16_split2": kernel.W13_K_UNROLL16_SPLIT2,
                    "w13_distributed_prep": kernel.W13_DISTRIBUTED_PREP,
                    "w2_distributed_prep": kernel.W2_DISTRIBUTED_PREP,
                    "w13_merged_wgmma_group": (
                        kernel.W13_MERGED_WGMMA_GROUP
                    ),
                    "w13_dual_wg_split": kernel.W13_DUAL_WG_SPLIT,
                    "w13_launch_bound_10": kernel.W13_LAUNCH_BOUND_10,
                    "w13_max_smem_carveout": (
                        kernel.W13_MAX_SMEM_CARVEOUT
                    ),
                    "tiled_k6_reduce_policy": kernel.TILED_K6_REDUCE_POLICY,
                    "fused_k6_push_ar": kernel.FUSED_K6_PUSH_AR,
                    "fused_k6_mc_push_ar": kernel.FUSED_K6_MC_PUSH_AR,
                    "fused_k6_mc_push_max_m": kernel.FUSED_K6_MC_PUSH_MAX_M,
                    "fused_k6_mc_pull_ar": kernel.FUSED_K6_MC_PULL_AR,
                    "pipelined_w2_mc_push_ar": kernel.PIPELINED_W2_MC_PUSH_AR,
                    "pipeline_chunks": kernel.PIPELINE_CHUNKS,
                    "pipeline_ar_blocks": kernel.PIPELINE_AR_BLOCKS,
                    "fused_rank_route_mc_pull_ar": (
                        kernel.FUSED_RANK_ROUTE_MC_PULL_AR
                    ),
                    "rank_route_pull_blocks": kernel.RANK_ROUTE_PULL_BLOCKS,
                    "fused_k6_nvls_pull_ar": kernel.FUSED_K6_NVLS_PULL_AR,
                    "single_launch_tp4": kernel.SINGLE_LAUNCH_TP4,
                    "single_launch_schedule": (
                        kernel.SINGLE_LAUNCH_SCHEDULE
                    ),
                    "single_launch_noinline_gemm": (
                        kernel.SINGLE_LAUNCH_NOINLINE_GEMM
                    ),
                    "single_launch_min_blocks": (
                        kernel.SINGLE_LAUNCH_MIN_BLOCKS
                    ),
                    "single_launch_m128_bound9": (
                        kernel.SINGLE_LAUNCH_M128_BOUND9
                    ),
                    "single_launch_ctas_per_sm": (
                        kernel.SINGLE_LAUNCH_CTAS_PER_SM
                    ),
                    "single_launch_hierarchical_grid": (
                        kernel.SINGLE_LAUNCH_HIERARCHICAL_GRID
                    ),
                    "single_launch_grid_poll_sleep_ns": (
                        kernel.SINGLE_LAUNCH_GRID_POLL_SLEEP_NS
                    ),
                    "single_launch_phase_stamps": (
                        kernel.SINGLE_LAUNCH_PHASE_STAMPS
                    ),
                    "single_launch_packed_grid_barrier": (
                        kernel.SINGLE_LAUNCH_PACKED_GRID_BARRIER
                    ),
                    "single_launch_balanced_workers": (
                        kernel.SINGLE_LAUNCH_BALANCED_WORKERS
                    ),
                    "single_launch_skip_final_cta_sync": (
                        kernel.SINGLE_LAUNCH_SKIP_FINAL_CTA_SYNC
                    ),
                    "k6_nvls_pull_blocks": kernel.K6_NVLS_PULL_BLOCKS,
                    "mc_pull_blocks": kernel.MC_PULL_BLOCKS or "default",
                    "mc_pull_unroll": kernel.MC_PULL_UNROLL or "default",
                    "w2_epilogue": (
                        "BF16 route output + fixed tiled CUDA k6 mode 4 at "
                        "M<=16, SGLang moe_fused_mul_sum otherwise"
                        if kernel.TILED_K6_REDUCE_POLICY == "auto"
                        else (
                            "BF16 route output + fixed tiled CUDA k6 reduce "
                            f"mode {kernel.TILED_K6_REDUCE_POLICY}"
                            if kernel.TILED_K6_REDUCE_POLICY != "0"
                            else "BF16 route output + sglang moe_fused_mul_sum"
                        )
                        if kernel.W2_ROUTE_OUTPUT
                        else "FP32 weighted atomic scatter + BF16 cast"
                    ),
                    "min_blocks_per_sm": kernel.MIN_BLOCKS_PER_SM,
                    "weight_dtype": "OCP MXFP4 E2M1",
                    "weight_scale": "E8M0 group32",
                    "activation_dtype": "FP8 E4M3 group128",
                    "output_dtype": "BF16",
                    "timed_allreduce": "sglang CustomAllReduceV2 default graph heuristic",
                },
                sort_keys=True,
            ),
            flush=True,
        )
        print(
            "CUSTOM_SHAPES "
            + json.dumps(
                {
                    "H": HIDDEN,
                    "I": INTERMEDIATE,
                    "I_per_rank": intermediate_per_rank,
                    "experts": NUM_EXPERTS,
                    "top_k": TOP_K,
                    "W13_per_rank": list(w13.shape),
                    "W2_per_rank": list(w2.shape),
                    "W13_bytes_with_scale": w13.nbytes + s13.nbytes,
                    "W2_bytes_with_scale": w2.nbytes + s2.nbytes,
                },
                sort_keys=True,
            ),
            flush=True,
        )

    cases: list[CapturedCase] = []
    graphs: list[torch.cuda.CUDAGraph] = []
    records: list[dict[str, Any]] = []
    for m in args.ms:
        topk_ids, topk_weights = make_routes(
            m, args.route_pattern, device, args.seed
        )
        qx, x_scale = make_fp8_input(m, device, args.seed)
        case = CapturedCase(
            m=m,
            qx=qx,
            x_scale=x_scale,
            topk_ids=topk_ids,
            topk_weights=topk_weights,
            w13=w13,
            s13=s13,
            g13=g13,
            w2=w2,
            s2=s2,
            g2=g2,
            lut=lut,
            intermediate_per_rank=intermediate_per_rank,
        )

        for _ in range(2):
            case.run_full(comm)
        torch.cuda.synchronize(device)
        dist.barrier(group=cpu_group)

        graph = torch.cuda.CUDAGraph()
        with comm.capture():
            with torch.cuda.graph(graph):
                case.run_full(comm)
        torch.cuda.synchronize(device)
        if args.profile_once:
            dist.barrier(group=cpu_group)
            torch.cuda.cudart().cudaProfilerStart()
            triton_runtime.driver.active.clear_cache(l2_flush_buffer)
            graph.replay()
            torch.cuda.synchronize(device)
            torch.cuda.cudart().cudaProfilerStop()
            dist.barrier(group=cpu_group)
            if rank == 0:
                print(
                    "CUSTOM_PROFILE_REPLAY "
                    + json.dumps(
                        {
                            "m": m,
                            "l2_policy": "cold; 256MiB clear immediately before replay",
                        },
                        sort_keys=True,
                    ),
                    flush=True,
                )
            cases.append(case)
            graphs.append(graph)
            continue
        for _ in range(args.warmup_replays):
            triton_runtime.driver.active.clear_cache(l2_flush_buffer)
            graph.replay()
        torch.cuda.synchronize(device)

        check = correctness_metrics(case, graph, nccl_group, device)
        samples, batch_medians = time_graph(
            graph,
            args.outer,
            args.replays,
            cpu_group,
            nccl_group,
            device,
            l2_flush_buffer,
        )
        nbytes = m * HIDDEN * torch.tensor([], dtype=torch.bfloat16).element_size()
        ar_algo, ar_mode = comm._pick_algo(nbytes, can_use_graph=True)
        assert case.num_tokens_padded is not None
        padded_rows = int(case.num_tokens_padded.item())
        record: dict[str, Any] = {
            "m": m,
            "route_pattern": args.route_pattern,
            "active_experts": case.active_experts,
            "routed_rows": m * TOP_K,
            "padded_rows": padded_rows,
            "padding_ratio": padded_rows / (m * TOP_K),
            "w13_split_k": case.w13_split_k,
            "tiled_k6_reduce_mode": case.tiled_k6_reduce_mode,
            "fused_k6_push_ar": case.fused_k6_push_active,
            "fused_k6_ar_mode": case.fused_k6_ar_mode,
            "allreduce_bytes": nbytes,
            "allreduce_algo": None if ar_algo is None else ar_algo.name,
            "allreduce_mode": ar_mode.name,
            "latency_ms_min": min(samples),
            "latency_ms_median": statistics.median(samples),
            "latency_ms_max": max(samples),
            "cold_samples": len(samples),
            "batch_medians_ms_max_rank": batch_medians,
            **check,
        }
        records.append(record)
        if rank == 0:
            print("CUSTOM_RESULT " + json.dumps(record, sort_keys=True), flush=True)
        cases.append(case)
        graphs.append(graph)

    if rank == 0 and not args.profile_once:
        medians = [float(record["latency_ms_median"]) for record in records]
        print(
            "CUSTOM_SUMMARY "
            + json.dumps(
                {
                    "world_size": world_size,
                    "route_pattern": args.route_pattern,
                    "m_values": args.ms,
                    "median_ms": medians,
                    "geometric_mean_median_ms": statistics.geometric_mean(medians),
                    "correctness": "custom-AR graph output vs NCCL sum of independent custom local recompute",
                },
                sort_keys=True,
            ),
            flush=True,
        )
    dist.barrier(group=cpu_group)


if __name__ == "__main__":
    main()
