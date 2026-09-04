#!/usr/bin/env python3
"""Stress the packed single-launch grid barrier under CUDA Graph replay.

This is a warm-cache correctness/liveness test, not a performance benchmark.
It deliberately performs no L2 eviction and records no timing events.
"""

from __future__ import annotations

import argparse
import json

import torch

import v4_flash_tp_wgmma as kernel
import v4_flash_tp_wgmma_graph as bench


COUNT_BITS = 10
COUNT_MASK = (1 << COUNT_BITS) - 1
GENERATION_BITS = 32 - COUNT_BITS
GENERATION_MODULUS = 1 << GENERATION_BITS


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ms", default="8,128")
    parser.add_argument("--replays", type=int, default=10_000)
    parser.add_argument("--checkpoint", type=int, default=1_000)
    parser.add_argument("--seed", type=int, default=20260904)
    args = parser.parse_args()
    args.ms = tuple(int(value) for value in args.ms.split(",") if value)
    if not args.ms or any(value not in bench.DEFAULT_MS for value in args.ms):
        parser.error(f"--ms must be drawn from {bench.DEFAULT_MS}")
    if args.replays < 2 or args.replays % 2:
        parser.error("--replays must be an even integer >= 2")
    if args.checkpoint < 1 or args.replays % args.checkpoint:
        parser.error("--checkpoint must divide --replays")
    if (args.replays // 2) % args.checkpoint:
        parser.error("--checkpoint must also divide --replays/2")
    if args.replays // 2 >= GENERATION_MODULUS:
        parser.error("--replays/2 must fit in the packed generation field")
    return args


def signed_i32(value: int) -> int:
    value &= 0xFFFF_FFFF
    return value if value < (1 << 31) else value - (1 << 32)


def compare(actual: torch.Tensor, expected: torch.Tensor) -> dict[str, float | bool]:
    actual_f = actual.double()
    expected_f = expected.double()
    diff = actual_f - expected_f
    return {
        "bitwise_equal": bool(torch.equal(actual, expected)),
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
def stress_shape(m: int, args: argparse.Namespace, device: torch.device) -> dict:
    intermediate = bench.INTERMEDIATE // 4
    weights = bench.make_weights(intermediate, device)
    topk_ids, topk_weights = bench.make_routes(m, "random", device, args.seed)
    qx, x_scale = bench.make_fp8_input(m, device, args.seed)
    case = bench.CapturedCase(
        m=m,
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

    output = torch.empty((m, bench.HIDDEN), dtype=torch.bfloat16, device=device)
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

    # Compile and initialize all lazy runtime state before graph capture.
    run()
    torch.cuda.synchronize(device)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        run()
    torch.cuda.synchronize(device)

    wrap_after = args.replays // 2
    seed_generation = GENERATION_MODULUS - wrap_after
    seed_word = signed_i32(seed_generation << COUNT_BITS)
    case.single_launch_barrier_state.zero_()
    case.single_launch_barrier_state[:8:2].fill_(seed_word)
    torch.cuda.synchronize(device)

    checkpoints = []
    expected_down = None
    for completed in range(1, args.replays + 1):
        if completed == args.replays:
            # Keep graph addresses fixed while changing the public FP8 input.
            new_qx, new_x_scale = bench.make_fp8_input(
                m, device, args.seed + 1_000_000
            )
            case.qx.copy_(new_qx)
            case.x_scale.copy_(new_x_scale)
            reference = case.make_reference_case()
            reference.run_before_local_reduce()
            assert reference.down is not None
            expected_down = reference.down.clone()
            case.down.fill_(float("nan"))

        graph.replay()
        if completed % args.checkpoint == 0:
            torch.cuda.synchronize(device)
            packed_signed = [
                int(value)
                for value in case.single_launch_barrier_state[:8:2]
                .cpu()
                .tolist()
            ]
            packed_unsigned = [value & 0xFFFF_FFFF for value in packed_signed]
            counts = [value & COUNT_MASK for value in packed_unsigned]
            generations = [value >> COUNT_BITS for value in packed_unsigned]
            expected_generation = (seed_generation + completed) % GENERATION_MODULUS
            legacy_words = [
                int(value)
                for value in case.single_launch_barrier_state[1:8:2]
                .cpu()
                .tolist()
            ]
            state_ok = (
                counts == [0, 0, 0, 0]
                and generations == [expected_generation] * 4
                and legacy_words == [0, 0, 0, 0]
            )
            checkpoints.append(
                {
                    "completed": completed,
                    "packed_words": packed_signed,
                    "generations": generations,
                    "counts": counts,
                    "legacy_words": legacy_words,
                    "expected_generation": expected_generation,
                    "state_ok": state_ok,
                }
            )
            if not state_ok:
                raise RuntimeError(
                    f"M={m} packed barrier state mismatch after {completed} replays"
                )

    assert expected_down is not None
    torch.cuda.synchronize(device)
    output_check = compare(case.down, expected_down)
    midpoint = next(item for item in checkpoints if item["completed"] == wrap_after)
    final = checkpoints[-1]
    accepted = bool(
        midpoint["packed_words"] == [0, 0, 0, 0]
        and final["packed_words"] == [wrap_after << COUNT_BITS] * 4
        and output_check["finite"]
        and output_check["bitwise_equal"]
        and output_check["cosine"] >= 0.999
    )
    return {
        "m": m,
        "w13_split_k": case.w13_split_k,
        "padded_rows": int(case.num_tokens_padded.item()),
        "replays": args.replays,
        "barriers_per_replay": 4,
        "seed_generation": seed_generation,
        "seed_packed_word": seed_word,
        "wrap_after": wrap_after,
        "checkpoint_states": checkpoints,
        "final_output_check": output_check,
        "accepted": accepted,
    }


@torch.inference_mode()
def main() -> None:
    args = parse_args()
    if not kernel.SINGLE_LAUNCH_PACKED_GRID_BARRIER:
        raise RuntimeError("stress requires V4_SINGLE_LAUNCH_PACKED_GRID_BARRIER=1")
    if not kernel.SINGLE_LAUNCH_RELEASE_GRID_ARRIVAL:
        raise RuntimeError("stress requires V4_SINGLE_LAUNCH_RELEASE_GRID_ARRIVAL=1")
    if kernel.SINGLE_LAUNCH_PHASE_STAMPS:
        raise RuntimeError("stress requires V4_SINGLE_LAUNCH_PHASE_STAMPS=0")

    device = torch.device("cuda:0")
    torch.cuda.set_device(device)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    results = [stress_shape(m, args, device) for m in args.ms]
    accepted = all(result["accepted"] for result in results)
    print(
        "SINGLE_GRAPH_BARRIER_STRESS "
        + json.dumps(
            {
                "input_contract": (
                    "prequantized FP8-E4M3 X plus FP32 group128 scale"
                ),
                "tp_collective_executed": False,
                "cuda_graph_capture_per_shape": 1,
                "l2_policy": "warm-cache liveness test; no performance timing",
                "packed_grid_barrier": kernel.SINGLE_LAUNCH_PACKED_GRID_BARRIER,
                "release_grid_arrival": kernel.SINGLE_LAUNCH_RELEASE_GRID_ARRIVAL,
                "results": results,
                "accepted": accepted,
            },
            sort_keys=True,
        ),
        flush=True,
    )
    if not accepted:
        raise RuntimeError("single-launch graph barrier stress failed")


if __name__ == "__main__":
    main()
