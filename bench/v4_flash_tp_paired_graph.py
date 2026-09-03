#!/usr/bin/env python3
"""Interleaved cold-L2 TP comparison of Humming and the custom V4 MoE path."""

from __future__ import annotations

import argparse
import json
import statistics
from typing import Any

import torch
import torch.distributed as dist
from triton import runtime as triton_runtime

import sglang.srt.distributed.parallel_state as ps
from humming.config import GemmType
from humming.layer import HummingMethod
from sglang.kernels.ops.communication.mp import register_comm_cleanup
from sglang.srt.distributed.device_communicators.custom_all_reduce_v2 import (
    CustomAllReduceV2,
)

import v4_flash_tp_humming_graph as humming
import v4_flash_tp_wgmma as kernel
import v4_flash_tp_wgmma_graph as custom


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ms", default="8,16,32,64,128")
    parser.add_argument(
        "--route-pattern", choices=("random", "balanced", "skew"), default="random"
    )
    parser.add_argument("--outer", type=int, default=10)
    parser.add_argument("--replays", type=int, default=200)
    parser.add_argument("--warmup-replays", type=int, default=10)
    parser.add_argument(
        "--pair-granularity",
        choices=("batch", "replay"),
        default="batch",
        help=(
            "batch preserves each implementation's repeated-execution TLB "
            "state; replay is a cross-weight-set TLB stress test"
        ),
    )
    parser.add_argument("--seed", type=int, default=20260902)
    args = parser.parse_args()
    args.ms = tuple(int(value) for value in args.ms.split(",") if value)
    if not args.ms or any(value <= 0 for value in args.ms):
        parser.error("--ms must contain positive integers")
    if args.outer < 1 or args.replays < 1 or args.warmup_replays < 1:
        parser.error("timing loop counts must be positive")
    if args.pair_granularity == "batch" and args.outer % 2:
        parser.error("batch pairing requires an even --outer for balanced AB/BA")
    return args


def capture_graph(
    case: Any,
    comm: CustomAllReduceV2,
    cpu_group: dist.ProcessGroup,
    device: torch.device,
) -> torch.cuda.CUDAGraph:
    for _ in range(2):
        case.run_full(comm)
    torch.cuda.synchronize(device)
    dist.barrier(group=cpu_group)
    graph = torch.cuda.CUDAGraph()
    with comm.capture():
        with torch.cuda.graph(graph):
            case.run_full(comm)
    torch.cuda.synchronize(device)
    dist.barrier(group=cpu_group)
    return graph


def time_graph_pair(
    humming_graph: torch.cuda.CUDAGraph,
    custom_graph: torch.cuda.CUDAGraph,
    outer: int,
    replays: int,
    cpu_group: dist.ProcessGroup,
    nccl_group: dist.ProcessGroup,
    device: torch.device,
    l2_flush_buffer: torch.Tensor,
    pair_granularity: str,
) -> tuple[list[float], list[float], list[float], list[float]]:
    humming_samples: list[float] = []
    custom_samples: list[float] = []
    humming_batch_medians: list[float] = []
    custom_batch_medians: list[float] = []
    driver = triton_runtime.driver.active

    for outer_idx in range(outer):
        dist.barrier(group=cpu_group)
        humming_starts = [
            torch.cuda.Event(enable_timing=True) for _ in range(replays)
        ]
        humming_ends = [
            torch.cuda.Event(enable_timing=True) for _ in range(replays)
        ]
        custom_starts = [
            torch.cuda.Event(enable_timing=True) for _ in range(replays)
        ]
        custom_ends = [
            torch.cuda.Event(enable_timing=True) for _ in range(replays)
        ]

        def replay_one(
            graph: torch.cuda.CUDAGraph,
            start: torch.cuda.Event,
            end: torch.cuda.Event,
        ) -> None:
            driver.clear_cache(l2_flush_buffer)
            start.record()
            graph.replay()
            end.record()

        if pair_granularity == "replay":
            for replay_idx in range(replays):
                # Fine-grained AB/BA is useful as a TLB-stress diagnostic.
                if (outer_idx + replay_idx) & 1:
                    replay_one(
                        custom_graph,
                        custom_starts[replay_idx],
                        custom_ends[replay_idx],
                    )
                    replay_one(
                        humming_graph,
                        humming_starts[replay_idx],
                        humming_ends[replay_idx],
                    )
                else:
                    replay_one(
                        humming_graph,
                        humming_starts[replay_idx],
                        humming_ends[replay_idx],
                    )
                    replay_one(
                        custom_graph,
                        custom_starts[replay_idx],
                        custom_ends[replay_idx],
                    )
        else:
            # Alternate complete batches to cancel long clock/temperature
            # drift without evicting one implementation's weight-page TLB
            # entries before every replay.  Use an even outer count for an
            # exactly balanced number of AB and BA batches.
            order = (
                (
                    (humming_graph, humming_starts, humming_ends),
                    (custom_graph, custom_starts, custom_ends),
                )
                if outer_idx % 2 == 0
                else (
                    (custom_graph, custom_starts, custom_ends),
                    (humming_graph, humming_starts, humming_ends),
                )
            )
            for graph, starts, ends in order:
                for replay_idx in range(replays):
                    replay_one(graph, starts[replay_idx], ends[replay_idx])
        torch.cuda.synchronize(device)

        humming_times = torch.tensor(
            [
                start.elapsed_time(end)
                for start, end in zip(humming_starts, humming_ends)
            ],
            dtype=torch.float64,
            device=device,
        )
        custom_times = torch.tensor(
            [
                start.elapsed_time(end)
                for start, end in zip(custom_starts, custom_ends)
            ],
            dtype=torch.float64,
            device=device,
        )
        dist.all_reduce(humming_times, op=dist.ReduceOp.MAX, group=nccl_group)
        dist.all_reduce(custom_times, op=dist.ReduceOp.MAX, group=nccl_group)
        humming_batch = [float(value) for value in humming_times.cpu().tolist()]
        custom_batch = [float(value) for value in custom_times.cpu().tolist()]
        humming_samples.extend(humming_batch)
        custom_samples.extend(custom_batch)
        humming_batch_medians.append(statistics.median(humming_batch))
        custom_batch_medians.append(statistics.median(custom_batch))

    return (
        humming_samples,
        custom_samples,
        humming_batch_medians,
        custom_batch_medians,
    )


@torch.inference_mode()
def main() -> None:
    args = parse_args()
    rank, world_size, device, cpu_group = custom.init_distributed()
    nccl_group = ps._WORLD.device_group
    if not isinstance(nccl_group, dist.ProcessGroup):
        raise RuntimeError("SGLang did not create the NCCL process group")
    props = torch.cuda.get_device_properties(device)
    intermediate_per_rank = custom.INTERMEDIATE // world_size

    torch.manual_seed(args.seed + rank)
    torch.cuda.manual_seed(args.seed + rank)
    (
        custom_w13,
        custom_s13,
        custom_g13,
        custom_w2,
        custom_s2,
        custom_g2,
    ) = custom.make_weights(
        intermediate_per_rank, device
    )
    lut = kernel.make_e2m1_e8m0_lut(device)
    humming_w13 = humming.make_layer(
        2 * intermediate_per_rank, humming.HIDDEN, device
    )
    humming_w2 = humming.make_layer(
        humming.HIDDEN, intermediate_per_rank, device
    )
    humming_w13_tuning = HummingMethod.get_default_tuning_configs(
        layer=humming_w13, use_f16_accum=False, gemm_type=GemmType.INDEXED
    )
    humming_w2_tuning = HummingMethod.get_default_tuning_configs(
        layer=humming_w2, use_f16_accum=False, gemm_type=GemmType.INDEXED
    )

    comm = CustomAllReduceV2(cpu_group, device)
    if comm.disabled:
        raise RuntimeError("SGLang CustomAllReduceV2 is disabled")
    register_comm_cleanup(comm)
    l2_flush_buffer = triton_runtime.driver.active.get_empty_cache_for_benchmark()
    if l2_flush_buffer.nbytes < 2 * props.L2_cache_size:
        raise RuntimeError("benchmark cache buffer is smaller than twice L2")

    if rank == 0:
        print(
            "PAIRED_ENV "
            + json.dumps(
                {
                    "benchmark": "interleaved_v4_flash_tp_humming_vs_custom",
                    "gpu": props.name,
                    "sm_count": props.multi_processor_count,
                    "world_size": world_size,
                    "m_values": args.ms,
                    "route_pattern": args.route_pattern,
                    "outer": args.outer,
                    "replays_per_outer_per_impl": args.replays,
                    "warmup_replays": args.warmup_replays,
                    "pair_order": (
                        "AB/BA alternated every replay and outer batch"
                        if args.pair_granularity == "replay"
                        else "AB/BA alternated by complete outer batch"
                    ),
                    "pair_granularity": args.pair_granularity,
                    "l2_policy": (
                        "cold; separate 256MiB Triton clear before every impl "
                        "replay, clear excluded from events"
                    ),
                    "l2_cache_bytes": props.L2_cache_size,
                    "l2_flush_bytes": l2_flush_buffer.nbytes,
                    "custom_mode2_braid": kernel.MODE2_BRAID,
                    "custom_fused_activation_quant": kernel.FUSED_ACT_QUANT,
                    "custom_fused_route_quant": kernel.FUSED_ROUTE_QUANT,
                    "custom_fused_k6_mc_push_ar": (
                        kernel.FUSED_K6_MC_PUSH_AR
                    ),
                    "custom_w2_global_lut": kernel.W2_GLOBAL_LUT,
                    "custom_w2_s2r_prefetch": kernel.W2_S2R_PREFETCH,
                    "custom_w13_s2r_prefetch": kernel.W13_S2R_PREFETCH,
                    "custom_leader_mbar_wait": kernel.LEADER_MBAR_WAIT,
                    "custom_w13_distributed_prep": (
                        kernel.W13_DISTRIBUTED_PREP
                    ),
                    "custom_w13_dual_wg_split": kernel.W13_DUAL_WG_SPLIT,
                    "custom_w13_launch_bound_10": (
                        kernel.W13_LAUNCH_BOUND_10
                    ),
                    "custom_w13_max_smem_carveout": (
                        kernel.W13_MAX_SMEM_CARVEOUT
                    ),
                    "custom_w13_early_stage_refill": (
                        kernel.W13_EARLY_STAGE_REFILL
                    ),
                    "custom_w2_distributed_prep": (
                        kernel.W2_DISTRIBUTED_PREP
                    ),
                    "custom_normalized_weight_scale": (
                        kernel.NORMALIZED_WEIGHT_SCALE
                    ),
                    "custom_activation_evict_last": (
                        kernel.ACTIVATION_EVICT_LAST
                    ),
                    "custom_tiled_weight_layout": kernel.TILED_WEIGHT_LAYOUT,
                    "custom_bulk_weight_copy": kernel.BULK_WEIGHT_COPY,
                    "custom_tma_cta_scope": kernel.TMA_CTA_SCOPE,
                    "custom_weight_evict_first": kernel.WEIGHT_EVICT_FIRST,
                    "custom_weight_policy_hoist": kernel.WEIGHT_POLICY_HOIST,
                    "custom_weight_policy_constant": (
                        kernel.WEIGHT_POLICY_CONSTANT
                    ),
                    "custom_w2_no_weight_evict_first": (
                        kernel.W2_NO_WEIGHT_EVICT_FIRST
                    ),
                    "custom_interleaved_bulk_copy": (
                        kernel.INTERLEAVED_BULK_COPY
                    ),
                    "custom_compact_interleaved_scale": (
                        kernel.COMPACT_INTERLEAVED_SCALE
                    ),
                    "timed_allreduce": "same SGLang CustomAllReduceV2 instance",
                },
                sort_keys=True,
            ),
            flush=True,
        )

    records: list[dict[str, Any]] = []
    for m in args.ms:
        topk_ids, topk_weights = custom.make_routes(
            m, args.route_pattern, device, args.seed
        )
        x = torch.randn((m, custom.HIDDEN), dtype=torch.bfloat16, device=device) * 0.1

        selected_humming_w13 = humming.select_tuning_config(
            humming_w13_tuning, m * custom.TOP_K
        )
        humming_case = humming.CapturedCase(
            m=m,
            x=x,
            topk_ids=topk_ids,
            topk_weights=topk_weights,
            w13=humming_w13,
            w2=humming_w2,
            w13_tuning=humming_w13_tuning,
            w2_tuning=humming_w2_tuning,
            w13_block_m=int(selected_humming_w13["block_shape"][0]),
            intermediate_per_rank=intermediate_per_rank,
        )
        custom_case = custom.CapturedCase(
            m=m,
            x=x,
            topk_ids=topk_ids,
            topk_weights=topk_weights,
            w13=custom_w13,
            s13=custom_s13,
            g13=custom_g13,
            w2=custom_w2,
            s2=custom_s2,
            g2=custom_g2,
            lut=lut,
            intermediate_per_rank=intermediate_per_rank,
        )

        humming_graph = capture_graph(humming_case, comm, cpu_group, device)
        custom_graph = capture_graph(custom_case, comm, cpu_group, device)
        humming_check = humming.correctness_metrics(
            humming_case, humming_graph, nccl_group, device
        )
        custom_check = custom.correctness_metrics(
            custom_case, custom_graph, nccl_group, device
        )
        if not humming_check["allreduce_ok"] or not custom_check["allreduce_ok"]:
            raise RuntimeError(f"correctness failure at M={m}")

        for warmup_idx in range(args.warmup_replays):
            if warmup_idx & 1:
                custom_graph.replay()
                humming_graph.replay()
            else:
                humming_graph.replay()
                custom_graph.replay()
        torch.cuda.synchronize(device)
        (
            humming_samples,
            custom_samples,
            humming_batch_medians,
            custom_batch_medians,
        ) = time_graph_pair(
            humming_graph,
            custom_graph,
            args.outer,
            args.replays,
            cpu_group,
            nccl_group,
            device,
            l2_flush_buffer,
            args.pair_granularity,
        )

        humming_median = statistics.median(humming_samples)
        custom_median = statistics.median(custom_samples)
        record = {
            "m": m,
            "routed_rows": m * custom.TOP_K,
            "active_experts": custom_case.active_experts,
            "padded_rows": int(custom_case.num_tokens_padded.item()),
            "w13_split_k": custom_case.w13_split_k,
            "cold_samples_per_impl": len(humming_samples),
            "humming_latency_ms_min": min(humming_samples),
            "humming_latency_ms_median": humming_median,
            "humming_latency_ms_max": max(humming_samples),
            "custom_latency_ms_min": min(custom_samples),
            "custom_latency_ms_median": custom_median,
            "custom_latency_ms_max": max(custom_samples),
            "custom_over_humming": custom_median / humming_median,
            "speedup_humming_over_custom": humming_median / custom_median,
            "humming_batch_medians_ms_max_rank": humming_batch_medians,
            "custom_batch_medians_ms_max_rank": custom_batch_medians,
            "humming_correctness": humming_check,
            "custom_correctness": custom_check,
        }
        records.append(record)
        if rank == 0:
            print("PAIRED_RESULT " + json.dumps(record, sort_keys=True), flush=True)

    if rank == 0:
        humming_geomean = statistics.geometric_mean(
            record["humming_latency_ms_median"] for record in records
        )
        custom_geomean = statistics.geometric_mean(
            record["custom_latency_ms_median"] for record in records
        )
        print(
            "PAIRED_SUMMARY "
            + json.dumps(
                {
                    "m_values": list(args.ms),
                    "humming_geometric_mean_ms": humming_geomean,
                    "custom_geometric_mean_ms": custom_geomean,
                    "custom_over_humming": custom_geomean / humming_geomean,
                    "speedup_humming_over_custom": humming_geomean / custom_geomean,
                    "samples_per_m_per_impl": args.outer * args.replays,
                },
                sort_keys=True,
            ),
            flush=True,
        )


if __name__ == "__main__":
    main()
