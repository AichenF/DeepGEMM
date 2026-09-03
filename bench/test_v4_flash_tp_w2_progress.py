#!/usr/bin/env python3
"""Single-rank publication/count gate for the W2 progress prototype."""

from __future__ import annotations

import argparse
import json

import torch

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
    args = parser.parse_args()

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
    tile_end = args.m * 32
    chunk_end = tile_end + args.m * 4
    queue_end = chunk_end + args.m * 4
    valid_end = queue_end + args.m * 4
    tile = state[:tile_end]
    chunk = state[tile_end:chunk_end]
    queue = state[chunk_end:queue_end]
    valid = state[queue_end:valid_end]
    tail = int(state[valid_end].item())
    claim = int(state[valid_end + 1].item())
    worker_done = int(state[valid_end + 2].item())
    expected_tasks = args.m * 4
    queue_values = sorted(int(value) for value in queue[:tail].tolist())
    result = {
        "m": args.m,
        "route_pattern": args.route_pattern,
        "tile_min": int(tile.min().item()),
        "tile_max": int(tile.max().item()),
        "chunk_min": int(chunk.min().item()),
        "chunk_max": int(chunk.max().item()),
        "queue_tail": tail,
        "valid_sum": int(valid.sum().item()),
        "worker_claim": claim,
        "worker_done": worker_done,
        "queue_is_permutation": queue_values == list(range(expected_tasks)),
        "w2_bitwise_equal": bool(torch.equal(progress_output, case.down)),
        "finite": bool(torch.isfinite(progress_output.float()).all().item()),
    }
    print("W2_PROGRESS_PROBE " + json.dumps(result, sort_keys=True), flush=True)
    if not (
        result["tile_min"] == result["tile_max"] == 6
        and result["chunk_min"] == result["chunk_max"] == 8
        and tail == expected_tasks
        and result["valid_sum"] == expected_tasks
        and claim == 0
        and worker_done == 0
        and result["queue_is_permutation"]
        and result["w2_bitwise_equal"]
        and result["finite"]
    ):
        raise SystemExit("W2_PROGRESS_PROBE_WRONG")
    print("W2_PROGRESS_PROBE_OK", flush=True)


if __name__ == "__main__":
    main()
