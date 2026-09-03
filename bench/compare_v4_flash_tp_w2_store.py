#!/usr/bin/env python3
"""Paired cold-L2 TP comparison of one compile-time kernel flag."""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import statistics
import sys
from pathlib import Path
from types import ModuleType

import torch
import torch.distributed as dist
from triton import runtime as triton_runtime

# Import the reusable graph harness with the requested control setting.
COMPARE_FLAG = os.environ.get(
    "V4_COMPARE_FLAG", "V4_W2_COALESCED_STORE"
)
if COMPARE_FLAG not in {
    "V4_W2_COALESCED_STORE",
    "V4_W2_MBLOCK_SCALE",
    "V4_W2_SORTED_ACT",
    "V4_W2_FOLD_GLOBAL_SCALE",
    "V4_W13_DISTRIBUTED_PREP",
    "V4_W13_DUAL_WG_SPLIT",
    "V4_W2_DISTRIBUTED_PREP",
    "V4_W13_MERGED_WGMMA_GROUP",
    "V4_NORMALIZED_SHARED_LUT",
    "V4_TMA_CTA_SCOPE",
    "V4_ACTIVATION_EVICT_LAST",
    "V4_WEIGHT_EVICT_FIRST",
    "V4_WEIGHT_POLICY_HOIST",
    "V4_WEIGHT_POLICY_CONSTANT",
    "V4_W2_NO_WEIGHT_EVICT_FIRST",
    "V4_EXACT_ROUTE_CAPACITY",
}:
    raise ValueError(f"unsupported V4_COMPARE_FLAG={COMPARE_FLAG}")
if COMPARE_FLAG != "V4_EXACT_ROUTE_CAPACITY":
    os.environ[COMPARE_FLAG] = "0"
import v4_flash_tp_wgmma as control_kernel  # noqa: E402
import v4_flash_tp_wgmma_graph as bench  # noqa: E402
from sglang.kernels.ops.communication.mp import (  # noqa: E402
    register_comm_cleanup,
)
from sglang.srt.distributed.device_communicators.custom_all_reduce_v2 import (  # noqa: E402
    CustomAllReduceV2,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--m", type=int, default=128)
    parser.add_argument("--route-pattern", choices=("random", "balanced", "skew"), default="random")
    parser.add_argument("--outer", type=int, default=5)
    parser.add_argument("--replays", type=int, default=100)
    parser.add_argument("--warmup-replays", type=int, default=10)
    parser.add_argument("--seed", type=int, default=20260902)
    args = parser.parse_args()
    if args.m <= 0 or args.outer <= 0 or args.replays <= 0:
        parser.error("M and timing loop counts must be positive")
    return args


def load_candidate() -> ModuleType:
    if COMPARE_FLAG == "V4_EXACT_ROUTE_CAPACITY":
        return control_kernel
    os.environ[COMPARE_FLAG] = "1"
    source = Path(control_kernel.__file__).resolve()
    name = "v4_flash_tp_wgmma_candidate_" + COMPARE_FLAG.lower()
    spec = importlib.util.spec_from_file_location(name, source)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load candidate module from {source}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    os.environ[COMPARE_FLAG] = "0"
    return module


def capture_case(
    kernel: ModuleType,
    comm: CustomAllReduceV2,
    m: int,
    x: torch.Tensor,
    topk_ids: torch.Tensor,
    topk_weights: torch.Tensor,
    weights: tuple[torch.Tensor, ...],
    lut: torch.Tensor,
    intermediate_per_rank: int,
    device: torch.device,
    cpu_group: dist.ProcessGroup,
    exact_route_capacity: bool = False,
) -> tuple[bench.CapturedCase, torch.cuda.CUDAGraph]:
    bench.kernel = kernel
    w13, s13, g13, w2, s2, g2 = weights
    case = bench.CapturedCase(
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
    if exact_route_capacity:
        counts = torch.bincount(
            topk_ids.flatten().to(torch.int64), minlength=bench.NUM_EXPERTS
        )
        exact_padded = int((((counts + 7) // 8) * 8).sum().item())
        case.sorted_ids = torch.empty(
            exact_padded, dtype=torch.int32, device=device
        )
        case.expert_ids = torch.empty(
            exact_padded // 8, dtype=torch.int32, device=device
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
    return case, graph


def reduce_rank_max(
    values: list[float], device: torch.device, group: dist.ProcessGroup
) -> list[float]:
    tensor = torch.tensor(values, dtype=torch.float64, device=device)
    dist.all_reduce(tensor, op=dist.ReduceOp.MAX, group=group)
    return [float(value) for value in tensor.cpu().tolist()]


@torch.inference_mode()
def main() -> None:
    args = parse_args()
    rank, world_size, device, cpu_group = bench.init_distributed()
    nccl_group = bench.ps._WORLD.device_group
    if not isinstance(nccl_group, dist.ProcessGroup):
        raise RuntimeError("SGLang did not create the NCCL group")
    candidate_kernel = load_candidate()
    intermediate_per_rank = bench.INTERMEDIATE // world_size

    torch.manual_seed(args.seed + rank)
    torch.cuda.manual_seed(args.seed + rank)
    bench.kernel = control_kernel
    weights = bench.make_weights(intermediate_per_rank, device)
    lut = control_kernel.make_e2m1_e8m0_lut(device)
    topk_ids, topk_weights = bench.make_routes(
        args.m, args.route_pattern, device, args.seed
    )
    x = torch.randn(
        (args.m, bench.HIDDEN), dtype=torch.bfloat16, device=device
    ) * 0.1

    comm = CustomAllReduceV2(cpu_group, device)
    if comm.disabled:
        raise RuntimeError("SGLang CustomAllReduceV2 is disabled")
    register_comm_cleanup(comm)
    l2_flush_buffer = triton_runtime.driver.active.get_empty_cache_for_benchmark()
    props = torch.cuda.get_device_properties(device)
    if l2_flush_buffer.nbytes < 2 * props.L2_cache_size:
        raise RuntimeError("cold-L2 buffer is smaller than twice the GPU L2")

    control_case, control_graph = capture_case(
        control_kernel,
        comm,
        args.m,
        x,
        topk_ids,
        topk_weights,
        weights,
        lut,
        intermediate_per_rank,
        device,
        cpu_group,
        False,
    )
    candidate_case, candidate_graph = capture_case(
        candidate_kernel,
        comm,
        args.m,
        x,
        topk_ids,
        topk_weights,
        weights,
        lut,
        intermediate_per_rank,
        device,
        cpu_group,
        COMPARE_FLAG == "V4_EXACT_ROUTE_CAPACITY",
    )

    driver = triton_runtime.driver.active
    for replay in range(args.warmup_replays):
        order = (
            (control_graph, candidate_graph)
            if replay % 2 == 0
            else (candidate_graph, control_graph)
        )
        for graph in order:
            driver.clear_cache(l2_flush_buffer)
            graph.replay()
    torch.cuda.synchronize(device)

    # Alternating graphs must remain semantically interchangeable after
    # repeated replay of the shared CustomAllReduceV2 communicator.
    control_graph.replay()
    candidate_graph.replay()
    torch.cuda.synchronize(device)
    assert control_case.graph_output is not None
    assert candidate_case.graph_output is not None
    exact_local = torch.equal(
        control_case.graph_output, candidate_case.graph_output
    )
    exact_tensor = torch.tensor(int(exact_local), dtype=torch.int32, device=device)
    dist.all_reduce(exact_tensor, op=dist.ReduceOp.MIN, group=nccl_group)
    exact_all_ranks = bool(exact_tensor.item())
    if not exact_all_ranks:
        raise RuntimeError("control/candidate graph outputs differ")

    control_samples: list[float] = []
    candidate_samples: list[float] = []
    control_batches: list[float] = []
    candidate_batches: list[float] = []
    for outer in range(args.outer):
        dist.barrier(group=cpu_group)
        control_events: list[tuple[torch.cuda.Event, torch.cuda.Event]] = []
        candidate_events: list[tuple[torch.cuda.Event, torch.cuda.Event]] = []
        for replay in range(args.replays):
            pairs = (
                (("control", control_graph), ("candidate", candidate_graph))
                if (outer * args.replays + replay) % 2 == 0
                else (("candidate", candidate_graph), ("control", control_graph))
            )
            for label, graph in pairs:
                start = torch.cuda.Event(enable_timing=True)
                end = torch.cuda.Event(enable_timing=True)
                driver.clear_cache(l2_flush_buffer)
                start.record()
                graph.replay()
                end.record()
                if label == "control":
                    control_events.append((start, end))
                else:
                    candidate_events.append((start, end))
        torch.cuda.synchronize(device)
        local_control = [a.elapsed_time(b) for a, b in control_events]
        local_candidate = [a.elapsed_time(b) for a, b in candidate_events]
        batch_control = reduce_rank_max(local_control, device, nccl_group)
        batch_candidate = reduce_rank_max(local_candidate, device, nccl_group)
        control_samples.extend(batch_control)
        candidate_samples.extend(batch_candidate)
        control_batches.append(statistics.median(batch_control))
        candidate_batches.append(statistics.median(batch_candidate))

    control_median = statistics.median(control_samples)
    candidate_median = statistics.median(candidate_samples)
    record = {
        "compare_flag": COMPARE_FLAG,
        "m": args.m,
        "world_size": world_size,
        "route_pattern": args.route_pattern,
        "samples_per_variant": len(control_samples),
        "l2_policy": "cold; separate 256MiB clear immediately before every graph replay; clear excluded from events",
        "order_policy": "per-sample alternating A/B then B/A",
        "control": {
            "flag_value": 0,
            "min_ms": min(control_samples),
            "median_ms": control_median,
            "max_ms": max(control_samples),
            "batch_medians_ms": control_batches,
        },
        "candidate": {
            "flag_value": 1,
            "min_ms": min(candidate_samples),
            "median_ms": candidate_median,
            "max_ms": max(candidate_samples),
            "batch_medians_ms": candidate_batches,
        },
        "control_over_candidate": control_median / candidate_median,
        "exact_all_ranks": exact_all_ranks,
    }
    if rank == 0:
        print("PAIRED_W2_STORE_RESULT " + json.dumps(record, sort_keys=True), flush=True)
    dist.barrier(group=cpu_group)


if __name__ == "__main__":
    main()
