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


def make_weights(
    intermediate_per_rank: int, device: torch.device
) -> tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
]:
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
    g13 = torch.empty(0, dtype=torch.float32, device=device)
    g2 = torch.empty(0, dtype=torch.float32, device=device)
    if kernel.NORMALIZED_WEIGHT_SCALE:
        s13, g13 = kernel.normalize_mxfp4_weight_scales_(w13, s13)
        s2, g2 = kernel.normalize_mxfp4_weight_scales_(w2, s2)
    if kernel.MODE2_BRAID:
        kernel.braid_mode2_(w13)
        kernel.braid_mode2_(w2)
    if kernel.TILED_WEIGHT_LAYOUT:
        w13, s13 = kernel.tile_mxfp4_weight_layout(w13, s13)
        w2, s2 = kernel.tile_mxfp4_weight_layout(w2, s2)
    return w13, s13, g13, w2, s2, g2


@dataclass
class CapturedCase:
    m: int
    x: torch.Tensor
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

    def __post_init__(self) -> None:
        device = self.x.device
        routes = self.m * TOP_K
        n13 = 2 * self.intermediate_per_rank
        self.qx = torch.empty(
            (self.m, HIDDEN), dtype=torch.float8_e4m3fn, device=device
        )
        self.partials = torch.empty(
            (kernel.W13_MAX_SPLITS, routes, n13),
            dtype=torch.float32,
            device=device,
        )
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
            (routes, self.intermediate_per_rank),
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
        max_padded = (
            routes * 8
            if routes < NUM_EXPERTS + 1
            else routes + (NUM_EXPERTS + 1) * 7
        )
        self.sorted_ids = torch.empty(
            (max_padded,), dtype=torch.int32, device=device
        )
        self.expert_ids = torch.empty(
            ((max_padded + 7) // 8,), dtype=torch.int32, device=device
        )
        self.num_tokens_padded = torch.empty(
            (1,), dtype=torch.int32, device=device
        )
        self.x_scale = torch.empty(
            (self.m, HIDDEN // 128), dtype=torch.float32, device=device
        )
        self.activation_scale: torch.Tensor | None = (
            torch.empty(
                (routes, self.intermediate_per_rank // 128),
                dtype=torch.float32,
                device=device,
            )
            if kernel.FUSED_ACT_QUANT
            else None
        )
        self.graph_output: torch.Tensor | None = None
        # Routes are fixed benchmark inputs.  Inspect them once before graph
        # capture; this synchronization and policy selection are not timed.
        self.active_experts = int(torch.unique(self.topk_ids).numel())
        self.w13_split_k = kernel.select_w13_split_k(
            routes, self.active_experts
        )

    @property
    def routes(self) -> int:
        return self.m * TOP_K

    def run_local(self) -> torch.Tensor:
        if kernel.FUSED_ROUTE_QUANT:
            kernel.fused_route_quant(
                self.topk_ids,
                self.x,
                self.sorted_ids,
                self.expert_ids,
                self.num_tokens_padded,
                self.qx.view(torch.uint8),
                self.x_scale,
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
            self.qx, self.x_scale = humming_ops.quant_input(
                inputs=self.x,
                outputs=self.qx,
                dtype="float8e4m3",
                group_size=128,
                m_major_scale=False,
                scale_dtype="float32",
            )
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
        if kernel.W2_ROUTE_OUTPUT:
            assert self.down is not None
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

    def run_full(self, comm: CustomAllReduceV2) -> torch.Tensor:
        self.graph_output = comm.custom_all_reduce(self.run_local())
        return self.graph_output

    def make_reference_case(self) -> "CapturedCase":
        return CapturedCase(
            m=self.m,
            x=self.x,
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
                    "tiled_weight_layout": kernel.TILED_WEIGHT_LAYOUT,
                    "bulk_weight_copy": kernel.BULK_WEIGHT_COPY,
                    "interleaved_bulk_copy": kernel.INTERLEAVED_BULK_COPY,
                    "mode2_braid": kernel.MODE2_BRAID,
                    "fused_activation_quant": kernel.FUSED_ACT_QUANT,
                    "fused_route_quant": kernel.FUSED_ROUTE_QUANT,
                    "w2_global_lut": kernel.W2_GLOBAL_LUT,
                    "w2_s2r_prefetch": kernel.W2_S2R_PREFETCH,
                    "w13_s2r_prefetch": kernel.W13_S2R_PREFETCH,
                    "leader_mbar_wait": kernel.LEADER_MBAR_WAIT,
                    "single_imad_lut": kernel.SINGLE_IMAD_LUT,
                    "w2_epilogue": (
                        "BF16 route output + sglang moe_fused_mul_sum"
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
        x = torch.randn((m, HIDDEN), dtype=torch.bfloat16, device=device) * 0.1
        case = CapturedCase(
            m=m,
            x=x,
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
