#!/usr/bin/env python3
"""Cold-L2 TP4 decomposition of the exact Humming baseline's all-reduce.

For each M this benchmark captures three graphs over identical shapes:

  * full:  Humming local MoE pipeline + SGLang CustomAllReduceV2
  * local: the same Humming local MoE pipeline without all-reduce
  * ar:    CustomAllReduceV2 alone on a stable zero BF16 [M, H] input

The primary all-reduce contribution is the paired ``full - local`` delta.
The independently timed AR-only graph is reported as a diagnostic because
adding isolated stage medians need not reproduce an end-to-end graph median.
"""

from __future__ import annotations

import argparse
import json
import math
import statistics
from dataclasses import dataclass

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
import v4_flash_tp_paired_graph as paired


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ms", default="8,16,32,64,128")
    parser.add_argument(
        "--route-pattern", choices=("random", "balanced", "skew"), default="random"
    )
    parser.add_argument("--outer", type=int, default=10)
    parser.add_argument("--replays", type=int, default=200)
    parser.add_argument("--warmup-replays", type=int, default=20)
    parser.add_argument("--seed", type=int, default=20260902)
    args = parser.parse_args()
    args.ms = tuple(int(value) for value in args.ms.split(",") if value)
    if not args.ms or any(value <= 0 for value in args.ms):
        parser.error("--ms must contain positive integers")
    if args.outer < 2 or args.outer % 2:
        parser.error("--outer must be positive and even for paired AB/BA timing")
    if args.replays < 1 or args.warmup_replays < 1:
        parser.error("replay counts must be positive")
    return args


def make_case(
    m: int,
    x: torch.Tensor,
    topk_ids: torch.Tensor,
    topk_weights: torch.Tensor,
    w13,
    w2,
    w13_tuning,
    w2_tuning,
    intermediate_per_rank: int,
) -> humming.CapturedCase:
    selected_w13 = humming.select_tuning_config(w13_tuning, m * humming.TOP_K)
    return humming.CapturedCase(
        m=m,
        x=x,
        topk_ids=topk_ids,
        topk_weights=topk_weights,
        w13=w13,
        w2=w2,
        w13_tuning=w13_tuning,
        w2_tuning=w2_tuning,
        w13_block_m=int(selected_w13["block_shape"][0]),
        intermediate_per_rank=intermediate_per_rank,
    )


def capture_local(
    case: humming.CapturedCase,
    cpu_group: dist.ProcessGroup,
    device: torch.device,
) -> torch.cuda.CUDAGraph:
    for _ in range(2):
        case.run_local()
    torch.cuda.synchronize(device)
    dist.barrier(group=cpu_group)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        case.run_local()
    torch.cuda.synchronize(device)
    dist.barrier(group=cpu_group)
    return graph


@dataclass
class AllReduceOnlyCase:
    input: torch.Tensor
    graph_output: torch.Tensor | None = None

    def run_full(self, comm: CustomAllReduceV2) -> torch.Tensor:
        self.graph_output = comm.custom_all_reduce(self.input)
        return self.graph_output


def warm_pair(
    first: torch.cuda.CUDAGraph,
    second: torch.cuda.CUDAGraph,
    replays: int,
    device: torch.device,
) -> None:
    for replay in range(replays):
        if replay & 1:
            second.replay()
            first.replay()
        else:
            first.replay()
            second.replay()
    torch.cuda.synchronize(device)


def geometric_mean(values: list[float]) -> float:
    return math.exp(sum(math.log(value) for value in values) / len(values))


@torch.inference_mode()
def main() -> None:
    args = parse_args()
    rank, world_size, device, cpu_group = humming.init_distributed()
    if world_size != 4:
        raise RuntimeError(f"this decomposition is defined for TP4, got TP{world_size}")
    nccl_group = ps._WORLD.device_group
    if not isinstance(nccl_group, dist.ProcessGroup):
        raise RuntimeError("SGLang did not create the NCCL process group")
    props = torch.cuda.get_device_properties(device)
    intermediate_per_rank = humming.INTERMEDIATE // world_size

    torch.manual_seed(args.seed + rank)
    torch.cuda.manual_seed(args.seed + rank)
    w13 = humming.make_layer(2 * intermediate_per_rank, humming.HIDDEN, device)
    w2 = humming.make_layer(humming.HIDDEN, intermediate_per_rank, device)
    w13_tuning = HummingMethod.get_default_tuning_configs(
        layer=w13, use_f16_accum=False, gemm_type=GemmType.INDEXED
    )
    w2_tuning = HummingMethod.get_default_tuning_configs(
        layer=w2, use_f16_accum=False, gemm_type=GemmType.INDEXED
    )
    comm = CustomAllReduceV2(cpu_group, device)
    if comm.disabled:
        raise RuntimeError("SGLang CustomAllReduceV2 is disabled")
    register_comm_cleanup(comm)

    l2_flush_buffer = triton_runtime.driver.active.get_empty_cache_for_benchmark()
    if l2_flush_buffer.nbytes < 2 * props.L2_cache_size:
        raise RuntimeError("cache-clear buffer is smaller than twice L2")

    if rank == 0:
        print(
            "AR_BREAKDOWN_ENV "
            + json.dumps(
                {
                    "benchmark": "exact_humming_full_vs_local_vs_carv2_only",
                    "gpu": props.name,
                    "sm_count": props.multi_processor_count,
                    "world_size": world_size,
                    "m_values": args.ms,
                    "route_pattern": args.route_pattern,
                    "outer": args.outer,
                    "replays_per_outer_per_graph": args.replays,
                    "warmup_replays": args.warmup_replays,
                    "pair_order": "complete-batch AB/BA",
                    "l2_policy": (
                        "cold; separate 256MiB Triton clear immediately before "
                        "every graph replay, clear excluded from CUDA events"
                    ),
                    "primary_share": "(median(full)-median(local))/median(full)",
                    "diagnostic_share": "median(ar_only)/median(paired_full)",
                },
                sort_keys=True,
            ),
            flush=True,
        )

    records: list[dict[str, object]] = []
    for m in args.ms:
        topk_ids, topk_weights = humming.make_routes(
            m, args.route_pattern, device, args.seed
        )
        x = (
            torch.randn((m, humming.HIDDEN), dtype=torch.bfloat16, device=device)
            * 0.1
        )
        full_case = make_case(
            m,
            x,
            topk_ids,
            topk_weights,
            w13,
            w2,
            w13_tuning,
            w2_tuning,
            intermediate_per_rank,
        )
        local_case = make_case(
            m,
            x,
            topk_ids,
            topk_weights,
            w13,
            w2,
            w13_tuning,
            w2_tuning,
            intermediate_per_rank,
        )

        full_graph = paired.capture_graph(full_case, comm, cpu_group, device)
        local_graph = capture_local(local_case, cpu_group, device)
        full_check = humming.correctness_metrics(
            full_case, full_graph, nccl_group, device
        )
        if not full_check["allreduce_ok"]:
            raise RuntimeError(f"full baseline correctness failure at M={m}")

        # Zero is stable even when graph-mode 2-shot pull reuses its input in
        # place, so every AR-only replay has identical numerical contents.
        ar_case = AllReduceOnlyCase(
            torch.zeros((m, humming.HIDDEN), dtype=torch.bfloat16, device=device)
        )
        ar_graph = paired.capture_graph(ar_case, comm, cpu_group, device)
        ar_graph.replay()
        torch.cuda.synchronize(device)
        assert ar_case.graph_output is not None
        ar_nonzero = torch.count_nonzero(ar_case.graph_output).to(torch.int64)
        dist.all_reduce(ar_nonzero, op=dist.ReduceOp.MAX, group=nccl_group)
        if int(ar_nonzero.item()) != 0:
            raise RuntimeError(f"AR-only zero correctness failure at M={m}")

        warm_pair(full_graph, local_graph, args.warmup_replays, device)
        (
            full_samples,
            local_samples,
            full_batch_medians,
            local_batch_medians,
        ) = paired.time_graph_pair(
            full_graph,
            local_graph,
            args.outer,
            args.replays,
            cpu_group,
            nccl_group,
            device,
            l2_flush_buffer,
            "batch",
        )

        warm_pair(full_graph, ar_graph, args.warmup_replays, device)
        (
            paired_full_samples,
            ar_samples,
            paired_full_batch_medians,
            ar_batch_medians,
        ) = paired.time_graph_pair(
            full_graph,
            ar_graph,
            args.outer,
            args.replays,
            cpu_group,
            nccl_group,
            device,
            l2_flush_buffer,
            "batch",
        )

        full_median = statistics.median(full_samples)
        local_median = statistics.median(local_samples)
        paired_full_median = statistics.median(paired_full_samples)
        ar_median = statistics.median(ar_samples)
        nbytes = m * humming.HIDDEN * torch.bfloat16.itemsize
        algo, mode = comm._pick_algo(nbytes, can_use_graph=True)
        record: dict[str, object] = {
            "m": m,
            "active_experts": int(torch.unique(topk_ids).numel()),
            "allreduce_bytes": nbytes,
            "allreduce_algo": None if algo is None else algo.name,
            "allreduce_mode": mode.name,
            "cold_samples_per_graph_per_pair": len(full_samples),
            "full_ms_min": min(full_samples),
            "full_ms_median": full_median,
            "full_ms_max": max(full_samples),
            "local_ms_min": min(local_samples),
            "local_ms_median": local_median,
            "local_ms_max": max(local_samples),
            "incremental_ar_ms": full_median - local_median,
            "incremental_ar_share": (full_median - local_median) / full_median,
            "paired_full_for_ar_ms_median": paired_full_median,
            "ar_only_ms_min": min(ar_samples),
            "ar_only_ms_median": ar_median,
            "ar_only_ms_max": max(ar_samples),
            "ar_only_share": ar_median / paired_full_median,
            "full_batch_medians_ms": full_batch_medians,
            "local_batch_medians_ms": local_batch_medians,
            "paired_full_for_ar_batch_medians_ms": paired_full_batch_medians,
            "ar_only_batch_medians_ms": ar_batch_medians,
            "full_correctness": full_check,
            "ar_only_zero_ok": True,
        }
        records.append(record)
        if rank == 0:
            print("AR_BREAKDOWN_RESULT " + json.dumps(record, sort_keys=True), flush=True)

    if rank == 0:
        full_values = [float(record["full_ms_median"]) for record in records]
        local_values = [float(record["local_ms_median"]) for record in records]
        paired_full_values = [
            float(record["paired_full_for_ar_ms_median"]) for record in records
        ]
        ar_values = [float(record["ar_only_ms_median"]) for record in records]
        full_gm = geometric_mean(full_values)
        local_gm = geometric_mean(local_values)
        paired_full_gm = geometric_mean(paired_full_values)
        ar_gm = geometric_mean(ar_values)
        print(
            "AR_BREAKDOWN_SUMMARY "
            + json.dumps(
                {
                    "m_values": args.ms,
                    "full_geometric_mean_ms": full_gm,
                    "local_geometric_mean_ms": local_gm,
                    "incremental_ar_geometric_mean_ms": full_gm - local_gm,
                    "incremental_ar_share_of_geometric_mean": (
                        full_gm - local_gm
                    )
                    / full_gm,
                    "paired_full_for_ar_geometric_mean_ms": paired_full_gm,
                    "ar_only_geometric_mean_ms": ar_gm,
                    "ar_only_share_of_paired_full_geometric_mean": (
                        ar_gm / paired_full_gm
                    ),
                },
                sort_keys=True,
            ),
            flush=True,
        )


if __name__ == "__main__":
    main()
