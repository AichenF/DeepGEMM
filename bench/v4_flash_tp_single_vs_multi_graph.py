#!/usr/bin/env python3
"""Paired cold-L2 comparison of single- and multi-kernel TP MegaMoE."""

from __future__ import annotations

import argparse
import json
import statistics
from typing import Any

import torch
import torch.distributed as dist
from triton import runtime as triton_runtime

import sglang.srt.distributed.parallel_state as ps
from sglang.kernels.ops.communication.mp import register_comm_cleanup
from sglang.srt.distributed.device_communicators.custom_all_reduce_v2 import (
    CustomAllReduceV2,
)

import v4_flash_tp_wgmma as kernel
import v4_flash_tp_wgmma_graph as custom
from v4_flash_tp_paired_graph import capture_graph, time_graph_pair


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ms", default="8,16,32,64,128")
    parser.add_argument(
        "--route-pattern", choices=("random", "balanced", "skew"), default="random"
    )
    parser.add_argument("--outer", type=int, default=10)
    parser.add_argument("--replays", type=int, default=200)
    parser.add_argument("--warmup-replays", type=int, default=20)
    parser.add_argument(
        "--pair-granularity", choices=("batch", "replay"), default="batch"
    )
    parser.add_argument("--seed", type=int, default=20260902)
    args = parser.parse_args()
    args.ms = tuple(int(value) for value in args.ms.split(",") if value)
    if not args.ms or any(value <= 0 for value in args.ms):
        parser.error("--ms must contain positive integers")
    if args.outer < 1 or args.replays < 1 or args.warmup_replays < 1:
        parser.error("timing loop counts must be positive")
    if args.pair_granularity == "batch" and args.outer % 2:
        parser.error("batch pairing requires an even --outer")
    return args


def make_case(
    m: int,
    qx: torch.Tensor,
    x_scale: torch.Tensor,
    topk_ids: torch.Tensor,
    topk_weights: torch.Tensor,
    weights: tuple[torch.Tensor, ...],
    lut: torch.Tensor,
    intermediate_per_rank: int,
) -> custom.CapturedCase:
    w13, s13, g13, w2, s2, g2 = weights
    return custom.CapturedCase(
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


@torch.inference_mode()
def main() -> None:
    args = parse_args()
    rank, world_size, device, cpu_group = custom.init_distributed()
    if world_size != 4:
        raise RuntimeError("single-vs-multi performance harness currently requires TP4")
    nccl_group = ps._WORLD.device_group
    if not isinstance(nccl_group, dist.ProcessGroup):
        raise RuntimeError("SGLang did not create the NCCL process group")

    props = torch.cuda.get_device_properties(device)
    intermediate_per_rank = custom.INTERMEDIATE // world_size
    torch.manual_seed(args.seed + rank)
    torch.cuda.manual_seed(args.seed + rank)
    weights = custom.make_weights(intermediate_per_rank, device)
    lut = kernel.make_e2m1_e8m0_lut(device)
    comm = CustomAllReduceV2(cpu_group, device)
    if comm.disabled:
        raise RuntimeError("SGLang CustomAllReduceV2 is disabled")
    register_comm_cleanup(comm)
    l2_flush_buffer = triton_runtime.driver.active.get_empty_cache_for_benchmark()
    if l2_flush_buffer.nbytes < 2 * props.L2_cache_size:
        raise RuntimeError("benchmark cache buffer is smaller than twice L2")

    if rank == 0:
        print(
            "SINGLE_MULTI_ENV "
            + json.dumps(
                {
                    "benchmark": "v4_flash_tp_single_vs_multi_cuda_graph",
                    "gpu": props.name,
                    "sm_count": props.multi_processor_count,
                    "world_size": world_size,
                    "m_values": args.ms,
                    "route_pattern": args.route_pattern,
                    "outer": args.outer,
                    "replays_per_outer_per_impl": args.replays,
                    "warmup_replays": args.warmup_replays,
                    "pair_granularity": args.pair_granularity,
                    "l2_policy": (
                        "cold; separate 256MiB Triton clear immediately before "
                        "every implementation replay, clear excluded from events"
                    ),
                    "input_contract": (
                        "shared FP8-E4M3 X + FP32 group128 scale; "
                        "BF16-to-FP8 quantization outside timed graphs"
                    ),
                    "control": "selected multi-kernel path from the same source",
                    "candidate": "one tp4_megamoe_single_launch_kernel graph node",
                },
                sort_keys=True,
            ),
            flush=True,
        )

    records: list[dict[str, Any]] = []
    keepalive: list[Any] = []
    driver = triton_runtime.driver.active
    for m in args.ms:
        topk_ids, topk_weights = custom.make_routes(
            m, args.route_pattern, device, args.seed
        )
        qx, x_scale = custom.make_fp8_input(m, device, args.seed)
        control_case = make_case(
            m,
            qx,
            x_scale,
            topk_ids,
            topk_weights,
            weights,
            lut,
            intermediate_per_rank,
        )
        candidate_case = make_case(
            m,
            qx,
            x_scale,
            topk_ids,
            topk_weights,
            weights,
            lut,
            intermediate_per_rank,
        )

        kernel.SINGLE_LAUNCH_TP4 = False
        control_graph = capture_graph(control_case, comm, cpu_group, device)
        kernel.SINGLE_LAUNCH_TP4 = True
        candidate_graph = capture_graph(candidate_case, comm, cpu_group, device)

        control_check = custom.correctness_metrics(
            control_case, control_graph, nccl_group, device
        )
        candidate_check = custom.correctness_metrics(
            candidate_case, candidate_graph, nccl_group, device
        )
        if not control_check["allreduce_ok"] or not candidate_check["allreduce_ok"]:
            raise RuntimeError(f"correctness failure at M={m}")

        for warmup_idx in range(args.warmup_replays):
            order = (
                (candidate_graph, control_graph)
                if warmup_idx & 1
                else (control_graph, candidate_graph)
            )
            for graph in order:
                driver.clear_cache(l2_flush_buffer)
                graph.replay()
        torch.cuda.synchronize(device)

        (
            control_samples,
            candidate_samples,
            control_batch_medians,
            candidate_batch_medians,
        ) = time_graph_pair(
            control_graph,
            candidate_graph,
            args.outer,
            args.replays,
            cpu_group,
            nccl_group,
            device,
            l2_flush_buffer,
            args.pair_granularity,
        )
        control_median = statistics.median(control_samples)
        candidate_median = statistics.median(candidate_samples)
        record: dict[str, Any] = {
            "m": m,
            "active_experts": candidate_case.active_experts,
            "routed_rows": m * custom.TOP_K,
            "control_padded_rows": int(control_case.num_tokens_padded.item()),
            "candidate_padded_rows": int(candidate_case.num_tokens_padded.item()),
            "w13_split_k": candidate_case.w13_split_k,
            "control_ar_mode": control_case.fused_k6_ar_mode,
            "candidate_ar_mode": candidate_case.fused_k6_ar_mode,
            "cold_samples_per_impl": len(control_samples),
            "control_latency_ms_min": min(control_samples),
            "control_latency_ms_median": control_median,
            "control_latency_ms_max": max(control_samples),
            "candidate_latency_ms_min": min(candidate_samples),
            "candidate_latency_ms_median": candidate_median,
            "candidate_latency_ms_max": max(candidate_samples),
            "candidate_over_control": candidate_median / control_median,
            "speedup_control_over_candidate": control_median / candidate_median,
            "control_batch_medians_ms_max_rank": control_batch_medians,
            "candidate_batch_medians_ms_max_rank": candidate_batch_medians,
            "control_correctness": control_check,
            "candidate_correctness": candidate_check,
        }
        records.append(record)
        if rank == 0:
            print(
                "SINGLE_MULTI_RESULT " + json.dumps(record, sort_keys=True),
                flush=True,
            )
        keepalive.extend((control_case, candidate_case, control_graph, candidate_graph))

    if rank == 0:
        control_geomean = statistics.geometric_mean(
            float(record["control_latency_ms_median"]) for record in records
        )
        candidate_geomean = statistics.geometric_mean(
            float(record["candidate_latency_ms_median"]) for record in records
        )
        print(
            "SINGLE_MULTI_SUMMARY "
            + json.dumps(
                {
                    "m_values": list(args.ms),
                    "control_geometric_mean_ms": control_geomean,
                    "candidate_geometric_mean_ms": candidate_geomean,
                    "candidate_over_control": candidate_geomean / control_geomean,
                    "speedup_control_over_candidate": control_geomean / candidate_geomean,
                    "samples_per_m_per_impl": args.outer * args.replays,
                },
                sort_keys=True,
            ),
            flush=True,
        )
    dist.barrier(group=cpu_group)


if __name__ == "__main__":
    main()
