#!/usr/bin/env python3
"""Same-process cold-L2 A/B audit of fused k6+push AR versus stock CARv2."""

from __future__ import annotations

import argparse
import json
import statistics

import torch
import torch.distributed as dist
from triton import runtime as triton_runtime

import sglang.srt.distributed.parallel_state as ps
from sglang.kernels.ops.communication.mp import register_comm_cleanup
from sglang.srt.distributed.device_communicators.custom_all_reduce_v2 import (
    CustomAllReduceV2,
)

import v4_flash_tp_paired_graph as paired
import v4_flash_tp_wgmma as kernel
import v4_flash_tp_wgmma_graph as custom


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ms", default="8,16,32")
    parser.add_argument(
        "--candidate",
        choices=(
            "unicast",
            "multicast",
            "multicast_pull",
            "pipeline",
            "progress",
            "rank_route_pull",
            "k6_nvls_pull",
        ),
        default="multicast",
    )
    parser.add_argument(
        "--route-pattern", choices=("random", "balanced", "skew"), default="random"
    )
    parser.add_argument("--outer", type=int, default=6)
    parser.add_argument("--replays", type=int, default=200)
    parser.add_argument("--warmup-replays", type=int, default=10)
    parser.add_argument("--pull-blocks", type=int, default=0)
    parser.add_argument("--pull-unroll", type=int, choices=(0, 2, 4, 8, 16), default=0)
    parser.add_argument("--pipeline-chunks", type=int, choices=(2, 4, 8), default=4)
    parser.add_argument(
        "--pipeline-ar-blocks",
        type=int,
        choices=(1, 2, 4, 8, 16, 32, 78),
        default=8,
    )
    parser.add_argument(
        "--progress-workers",
        type=int,
        choices=(1, 2, 4, 8, 16, 32),
        default=8,
    )
    parser.add_argument(
        "--rank-route-pull-blocks",
        type=int,
        choices=(1, 2, 4, 8, 16, 32, 64),
        default=16,
    )
    parser.add_argument(
        "--k6-nvls-pull-blocks",
        type=int,
        choices=(1, 2, 4, 8, 16, 32, 64),
        default=16,
    )
    parser.add_argument("--seed", type=int, default=20260902)
    args = parser.parse_args()
    args.ms = tuple(int(value) for value in args.ms.split(",") if value)
    if args.candidate in ("pipeline", "rank_route_pull"):
        supported = (128,)
    elif args.candidate == "progress":
        supported = (8, 16, 32, 64, 128)
    elif args.candidate == "k6_nvls_pull":
        supported = (8, 16, 32, 64, 128)
    elif args.candidate == "multicast_pull":
        supported = (8, 16, 32, 64, 128)
    elif args.candidate == "multicast":
        supported = (8, 16, 32, 64)
    else:
        supported = (8, 16, 32)
    if not args.ms or any(value not in supported for value in args.ms):
        parser.error(f"--ms must be a nonempty subset of {supported}")
    if args.outer < 2 or args.outer % 2:
        parser.error("--outer must be positive and even for balanced AB/BA")
    if args.replays < 1 or args.warmup_replays < 1:
        parser.error("replay counts must be positive")
    if args.pull_blocks < 0:
        parser.error("--pull-blocks must be nonnegative")
    return args


def make_case(
    m: int,
    x: torch.Tensor,
    topk_ids: torch.Tensor,
    topk_weights: torch.Tensor,
    weights: tuple[torch.Tensor, ...],
    lut: torch.Tensor,
    intermediate_per_rank: int,
) -> custom.CapturedCase:
    w13, s13, g13, w2, s2, g2 = weights
    return custom.CapturedCase(
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


@torch.inference_mode()
def main() -> None:
    args = parse_args()
    rank, world_size, device, cpu_group = custom.init_distributed()
    if world_size != 4:
        raise RuntimeError("fused push audit requires TP4")
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
        raise RuntimeError("cache-clear buffer is smaller than twice L2")

    if rank == 0:
        print(
            "FUSED_AB_ENV "
            + json.dumps(
                {
                    "benchmark": "same-process fused-k6-push versus stock CARv2",
                    "gpu": props.name,
                    "sm_count": props.multi_processor_count,
                    "world_size": world_size,
                    "m_values": args.ms,
                    "candidate": args.candidate,
                    "pull_blocks": args.pull_blocks or "default",
                    "pull_unroll": args.pull_unroll or "default",
                    "pipeline_chunks": args.pipeline_chunks,
                    "pipeline_ar_blocks": args.pipeline_ar_blocks,
                    "progress_workers": args.progress_workers,
                    "rank_route_pull_blocks": args.rank_route_pull_blocks,
                    "k6_nvls_pull_blocks": args.k6_nvls_pull_blocks,
                    "route_pattern": args.route_pattern,
                    "outer": args.outer,
                    "replays_per_outer_per_impl": args.replays,
                    "pair_order": "fused/control then control/fused by batch",
                    "l2_policy": (
                        "cold; separate 256MiB clear immediately before every "
                        "graph replay, clear excluded from CUDA events"
                    ),
                    "same_communicator": True,
                    "same_weights_and_inputs": True,
                },
                sort_keys=True,
            ),
            flush=True,
        )

    records: list[dict[str, object]] = []
    for m in args.ms:
        topk_ids, topk_weights = custom.make_routes(
            m, args.route_pattern, device, args.seed
        )
        x = torch.randn((m, custom.HIDDEN), dtype=torch.bfloat16, device=device) * 0.1
        fused_case = make_case(
            m, x, topk_ids, topk_weights, weights, lut, intermediate_per_rank
        )
        control_case = make_case(
            m, x, topk_ids, topk_weights, weights, lut, intermediate_per_rank
        )

        kernel.MC_PULL_BLOCKS = args.pull_blocks
        kernel.MC_PULL_UNROLL = args.pull_unroll
        kernel.PIPELINE_CHUNKS = args.pipeline_chunks
        kernel.PIPELINE_AR_BLOCKS = args.pipeline_ar_blocks
        kernel.W2_PROGRESS_WORKERS = args.progress_workers
        kernel.RANK_ROUTE_PULL_BLOCKS = args.rank_route_pull_blocks
        kernel.K6_NVLS_PULL_BLOCKS = args.k6_nvls_pull_blocks
        kernel.FUSED_K6_PUSH_AR = args.candidate == "unicast"
        kernel.FUSED_K6_MC_PUSH_AR = args.candidate == "multicast"
        kernel.FUSED_K6_MC_PULL_AR = args.candidate == "multicast_pull"
        kernel.PIPELINED_W2_MC_PUSH_AR = args.candidate == "pipeline"
        kernel.W2_PROGRESS_MC_PUSH_AR = args.candidate == "progress"
        kernel.FUSED_RANK_ROUTE_MC_PULL_AR = (
            args.candidate == "rank_route_pull"
        )
        kernel.FUSED_K6_NVLS_PULL_AR = args.candidate == "k6_nvls_pull"
        fused_graph = paired.capture_graph(fused_case, comm, cpu_group, device)
        if not fused_case.fused_k6_push_active:
            raise RuntimeError(f"fused graph did not select fused AR at M={m}")
        kernel.FUSED_K6_PUSH_AR = False
        kernel.FUSED_K6_MC_PUSH_AR = False
        kernel.FUSED_K6_MC_PULL_AR = False
        kernel.PIPELINED_W2_MC_PUSH_AR = False
        kernel.W2_PROGRESS_MC_PUSH_AR = False
        kernel.FUSED_RANK_ROUTE_MC_PULL_AR = False
        kernel.FUSED_K6_NVLS_PULL_AR = False
        control_graph = paired.capture_graph(control_case, comm, cpu_group, device)
        if control_case.fused_k6_push_active:
            raise RuntimeError(f"control graph selected fused AR at M={m}")

        fused_check = custom.correctness_metrics(
            fused_case, fused_graph, nccl_group, device
        )
        control_check = custom.correctness_metrics(
            control_case, control_graph, nccl_group, device
        )
        if not fused_check["allreduce_ok"] or not control_check["allreduce_ok"]:
            raise RuntimeError(f"correctness failure at M={m}")

        fused_graph.replay()
        control_graph.replay()
        torch.cuda.synchronize(device)
        graph_max_abs = torch.max(
            torch.abs(
                fused_case.graph_output.float() - control_case.graph_output.float()
            )
        )
        dist.all_reduce(graph_max_abs, op=dist.ReduceOp.MAX, group=nccl_group)

        for warmup_idx in range(args.warmup_replays):
            if warmup_idx & 1:
                control_graph.replay()
                fused_graph.replay()
            else:
                fused_graph.replay()
                control_graph.replay()
        torch.cuda.synchronize(device)

        (
            fused_samples,
            control_samples,
            fused_batch_medians,
            control_batch_medians,
        ) = paired.time_graph_pair(
            fused_graph,
            control_graph,
            args.outer,
            args.replays,
            cpu_group,
            nccl_group,
            device,
            l2_flush_buffer,
            "batch",
        )
        fused_median = statistics.median(fused_samples)
        control_median = statistics.median(control_samples)
        record = {
            "m": m,
            "cold_samples_per_impl": len(fused_samples),
            "fused_latency_ms_min": min(fused_samples),
            "fused_latency_ms_median": fused_median,
            "fused_latency_ms_max": max(fused_samples),
            "control_latency_ms_min": min(control_samples),
            "control_latency_ms_median": control_median,
            "control_latency_ms_max": max(control_samples),
            "speedup_control_over_fused": control_median / fused_median,
            "fused_batch_medians_ms_max_rank": fused_batch_medians,
            "control_batch_medians_ms_max_rank": control_batch_medians,
            "fused_vs_control_max_abs": float(graph_max_abs.item()),
            "fused_correctness": fused_check,
            "control_correctness": control_check,
        }
        records.append(record)
        if rank == 0:
            print("FUSED_AB_RESULT " + json.dumps(record, sort_keys=True), flush=True)

    if rank == 0:
        fused_geomean = statistics.geometric_mean(
            float(record["fused_latency_ms_median"]) for record in records
        )
        control_geomean = statistics.geometric_mean(
            float(record["control_latency_ms_median"]) for record in records
        )
        print(
            "FUSED_AB_SUMMARY "
            + json.dumps(
                {
                    "m_values": list(args.ms),
                    "fused_geometric_mean_ms": fused_geomean,
                    "control_geometric_mean_ms": control_geomean,
                    "speedup_control_over_fused": control_geomean / fused_geomean,
                },
                sort_keys=True,
            ),
            flush=True,
        )


if __name__ == "__main__":
    main()
