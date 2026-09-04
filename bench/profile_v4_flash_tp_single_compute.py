#!/usr/bin/env python3
"""Profile the single-launch TP MegaMoE compute body without its collective."""

from __future__ import annotations

import argparse
import json

import torch
from triton import runtime as triton_runtime

import v4_flash_tp_wgmma as kernel
import v4_flash_tp_wgmma_graph as bench


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--m", type=int, choices=(8, 16, 32, 64, 128), default=128)
    parser.add_argument(
        "--route-pattern", choices=("random", "balanced", "skew"), default="random"
    )
    parser.add_argument("--seed", type=int, default=20260902)
    return parser.parse_args()


def compare(actual: torch.Tensor, expected: torch.Tensor) -> dict[str, float | bool]:
    actual_f = actual.double()
    expected_f = expected.double()
    diff = actual_f - expected_f
    return {
        "cosine": float(
            torch.nn.functional.cosine_similarity(
                actual_f.flatten(), expected_f.flatten(), dim=0
            ).item()
        ),
        "rel_l2": float(
            (
                torch.linalg.vector_norm(diff)
                / torch.linalg.vector_norm(expected_f).clamp_min(1e-40)
            ).item()
        ),
        "finite": bool(torch.isfinite(actual).all()),
    }


@torch.inference_mode()
def main() -> None:
    args = parse_args()
    device = torch.device("cuda:0")
    torch.cuda.set_device(device)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)

    intermediate = bench.INTERMEDIATE // 4
    weights = bench.make_weights(intermediate, device)
    topk_ids, topk_weights = bench.make_routes(
        args.m, args.route_pattern, device, args.seed
    )
    qx, x_scale = bench.make_fp8_input(args.m, device, args.seed)
    case = bench.CapturedCase(
        m=args.m,
        qx=qx,
        x_scale=x_scale,
        topk_ids=topk_ids,
        topk_weights=topk_weights,
        w13=weights[0],
        s13=weights[1],
        g13=weights[2],
        w2=weights[3],
        s2=weights[4],
        g2=weights[5],
        lut=kernel.make_e2m1_e8m0_lut(device),
        intermediate_per_rank=intermediate,
    )
    assert case.down is not None
    assert case.activation_scale is not None

    reference = case.make_reference_case()
    reference.run_before_local_reduce()
    assert reference.down is not None
    expected_down = reference.down.clone()
    torch.cuda.synchronize(device)

    output = torch.empty((args.m, bench.HIDDEN), dtype=torch.bfloat16, device=device)
    push_counter = torch.empty((1,), dtype=torch.int32, device=device)
    push_workspaces = tuple(
        torch.empty((1,), dtype=torch.uint8, device=device) for _ in range(4)
    )
    pull_input = torch.empty_like(output)
    pull_sem_local = torch.empty(
        (kernel.K6_NVLS_PULL_BLOCKS * 128,), dtype=torch.uint8, device=device
    )

    def run() -> None:
        kernel.run_tp4_megamoe_single_launch(
            case.w13,
            case.s13,
            case.g13,
            case.w2,
            case.s2,
            case.g2,
            case.qx,
            case.x_scale,
            case.topk_ids,
            case.topk_weights,
            case.sorted_ids,
            case.expert_ids,
            case.num_tokens_padded,
            case.partials,
            case.activation,
            case.qactivation,
            case.activation_scale,
            case.down,
            case.lut,
            case.single_launch_barrier_state,
            case.route_to_sorted,
            output,
            push_counter,
            push_workspaces,
            pull_input,
            pull_sem_local,
            -1,
            0,
            0,
            0,
            0,
            case.w13_split_k,
            enable_tp_collective=False,
        )

    run()
    torch.cuda.synchronize(device)
    props = torch.cuda.get_device_properties(device)
    flush = triton_runtime.driver.active.get_empty_cache_for_benchmark()
    if flush.nbytes < 2 * props.L2_cache_size:
        raise RuntimeError("cache-clear buffer is smaller than twice L2")

    torch.cuda.cudart().cudaProfilerStart()
    triton_runtime.driver.active.clear_cache(flush)
    run()
    torch.cuda.synchronize(device)
    torch.cuda.cudart().cudaProfilerStop()

    check = compare(case.down, expected_down)
    stamps = case.single_launch_barrier_state[8:18].view(torch.int64)
    durations_us = (stamps[1:] - stamps[:-1]).double().cpu() / 1000.0
    phases = dict(
        zip(
            ("route", "w13", "activation_requant", "w2"),
            (float(value) for value in durations_us.tolist()),
            strict=True,
        )
    )
    accepted = bool(check["finite"] and check["cosine"] >= 0.999)
    print(
        "SINGLE_COMPUTE_PROFILE "
        + json.dumps(
            {
                "m": args.m,
                "route_pattern": args.route_pattern,
                "input_contract": "prequantized FP8-E4M3 X plus FP32 group128 scale",
                "tp_collective_executed": False,
                "sm_count": props.multi_processor_count,
                "l2_policy": "cold 256MiB clear outside profiled kernel",
                "w13_split_k": case.w13_split_k,
                "padded_rows": int(case.num_tokens_padded.item()),
                "phase_us": phases,
                "down_check": check,
                "accepted": accepted,
            },
            sort_keys=True,
        ),
        flush=True,
    )
    if not accepted:
        raise RuntimeError("single-launch compute-only output failed correctness")


if __name__ == "__main__":
    main()
