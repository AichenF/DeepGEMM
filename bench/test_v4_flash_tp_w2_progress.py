#!/usr/bin/env python3
"""Single-rank publication/count gate for the W2 progress prototype."""

from __future__ import annotations

import argparse
import json
import statistics

import torch
from triton import runtime as triton_runtime

import profile_v4_flash_tp_local as factory
import v4_flash_tp_wgmma as kernel


@torch.inference_mode()
def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--m", type=int, default=8)
    parser.add_argument(
        "--route-pattern", choices=("random", "balanced", "skew"),
        default="random"
    )
    parser.add_argument("--bench-samples", type=int, default=0)
    parser.add_argument("--bench-outer", type=int, default=4)
    args = parser.parse_args()
    if args.bench_samples < 0:
        parser.error("--bench-samples must be nonnegative")
    if args.bench_outer < 2 or args.bench_outer % 2:
        parser.error("--bench-outer must be positive and even")

    torch.cuda.set_device(0)
    device = torch.device("cuda:0")
    case = factory.make_custom_case(
        args.m, 4, args.route_pattern, 20260902, device
    )
    case.run_before_w2()
    case.w2_progress_state.zero_()
    assert case.down is not None
    assert case.activation_scale is not None
    kernel.run_w2_progress(
        case.w2,
        case.s2,
        case.g2,
        case.qactivation.view(torch.uint8),
        case.activation_scale,
        case.sorted_ids,
        case.expert_ids,
        case.num_tokens_padded,
        case.topk_weights,
        case.down,
        case.lut,
        case.w2_progress_state,
        case.intermediate_per_rank,
    )
    progress_output = case.down.clone()
    kernel.run_w2(
        case.w2,
        case.s2,
        case.g2,
        case.qactivation.view(torch.uint8),
        case.activation_scale,
        case.sorted_ids,
        case.expert_ids,
        case.num_tokens_padded,
        case.topk_weights,
        case.down,
        case.lut,
        case.intermediate_per_rank,
    )
    torch.cuda.synchronize(device)

    state = case.w2_progress_state.cpu()
    marker_end = args.m * 6 * 32
    markers = state[:marker_end]
    task_done = int(state[marker_end].item())
    worker_done = int(state[marker_end + 1].item())
    result = {
        "m": args.m,
        "route_pattern": args.route_pattern,
        "marker_min": int(markers.min().item()),
        "marker_max": int(markers.max().item()),
        "marker_sum": int(markers.sum().item()),
        "task_done": task_done,
        "worker_done": worker_done,
        "w2_bitwise_equal": bool(torch.equal(progress_output, case.down)),
        "finite": bool(torch.isfinite(progress_output.float()).all().item()),
    }
    print("W2_PROGRESS_PROBE " + json.dumps(result, sort_keys=True), flush=True)
    if not (
        result["marker_min"] == result["marker_max"] == 1
        and result["marker_sum"] == marker_end
        and task_done == 0
        and worker_done == 0
        and result["w2_bitwise_equal"]
        and result["finite"]
    ):
        raise SystemExit("W2_PROGRESS_PROBE_WRONG")
    print("W2_PROGRESS_PROBE_OK", flush=True)

    if not args.bench_samples:
        return

    # Capture the same state reset in both graphs so the measured difference
    # is the W2 publication protocol itself, not an extra memset node.
    def run_control() -> None:
        case.w2_progress_state.zero_()
        kernel.run_w2(
            case.w2,
            case.s2,
            case.g2,
            case.qactivation.view(torch.uint8),
            case.activation_scale,
            case.sorted_ids,
            case.expert_ids,
            case.num_tokens_padded,
            case.topk_weights,
            case.down,
            case.lut,
            case.intermediate_per_rank,
        )

    def run_progress() -> None:
        case.w2_progress_state.zero_()
        kernel.run_w2_progress(
            case.w2,
            case.s2,
            case.g2,
            case.qactivation.view(torch.uint8),
            case.activation_scale,
            case.sorted_ids,
            case.expert_ids,
            case.num_tokens_padded,
            case.topk_weights,
            case.down,
            case.lut,
            case.w2_progress_state,
            case.intermediate_per_rank,
        )

    for _ in range(2):
        run_control()
        run_progress()
    torch.cuda.synchronize(device)

    control_graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(control_graph):
        run_control()
    progress_graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(progress_graph):
        run_progress()
    torch.cuda.synchronize(device)

    control_graph.replay()
    torch.cuda.synchronize(device)
    control_output = case.down.clone()
    progress_graph.replay()
    torch.cuda.synchronize(device)
    graph_bitwise_equal = bool(torch.equal(control_output, case.down))
    if not graph_bitwise_equal:
        raise SystemExit("W2_PROGRESS_GRAPH_WRONG")

    props = torch.cuda.get_device_properties(device)
    flush = triton_runtime.driver.active.get_empty_cache_for_benchmark()
    if flush.nbytes < 2 * props.L2_cache_size:
        raise RuntimeError("cache-clear buffer is smaller than twice L2")
    for _ in range(5):
        triton_runtime.driver.active.clear_cache(flush)
        control_graph.replay()
        triton_runtime.driver.active.clear_cache(flush)
        progress_graph.replay()
    torch.cuda.synchronize(device)

    samples = {"control": [], "progress": []}
    for outer in range(args.bench_outer):
        order = (
            (("progress", progress_graph), ("control", control_graph))
            if outer % 2 == 0
            else (("control", control_graph), ("progress", progress_graph))
        )
        for name, graph in order:
            for _ in range(args.bench_samples):
                triton_runtime.driver.active.clear_cache(flush)
                start = torch.cuda.Event(enable_timing=True)
                end = torch.cuda.Event(enable_timing=True)
                start.record()
                graph.replay()
                end.record()
                end.synchronize()
                samples[name].append(start.elapsed_time(end) * 1000.0)

    control_median = statistics.median(samples["control"])
    progress_median = statistics.median(samples["progress"])
    print(
        "W2_PROGRESS_AB "
        + json.dumps(
            {
                "m": args.m,
                "route_pattern": args.route_pattern,
                "samples_per_impl": len(samples["control"]),
                "l2_policy": (
                    "cold; separate 256MiB clear before every graph replay, "
                    "clear excluded from CUDA events"
                ),
                "same_state_reset": True,
                "graph_bitwise_equal": graph_bitwise_equal,
                "control_us_min": min(samples["control"]),
                "control_us_median": control_median,
                "control_us_max": max(samples["control"]),
                "progress_us_min": min(samples["progress"]),
                "progress_us_median": progress_median,
                "progress_us_max": max(samples["progress"]),
                "control_over_progress": control_median / progress_median,
            },
            sort_keys=True,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
