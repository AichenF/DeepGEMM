#!/usr/bin/env python3
"""Correctness and graph-mutation gate for the TP interleaved scheduler."""

from __future__ import annotations

import json

import torch

import v4_flash_tp_wgmma as kernel


HIDDEN = 4096
EXPERTS = 256
TOP_K = 6
W13_TILES_TP4 = 4
W2_TILES = 16


def make_routes(m: int, pattern: str, device: torch.device) -> torch.Tensor:
    if pattern == "balanced":
        ids = torch.arange(m * TOP_K, dtype=torch.int32).view(m, TOP_K)
        ids %= EXPERTS
    elif pattern == "skew":
        ids = torch.arange(TOP_K, dtype=torch.int32).repeat(m, 1)
    elif pattern == "random":
        generator = torch.Generator(device="cpu")
        generator.manual_seed(20260903 + m)
        scores = torch.randn((m, EXPERTS), generator=generator)
        ids = torch.topk(scores, TOP_K, dim=1).indices.to(torch.int32)
    else:
        raise ValueError(pattern)
    return ids.to(device)


def max_padded_for(routes: int) -> int:
    return routes * 8 if routes < EXPERTS + 1 else routes + (EXPERTS + 1) * 7


class ProbeCase:
    def __init__(self, m: int, device: torch.device) -> None:
        self.m = m
        self.routes = m * TOP_K
        self.max_padded = max_padded_for(self.routes)
        self.max_mblocks = (self.max_padded + 7) // 8
        self.topk_ids = torch.empty((m, TOP_K), dtype=torch.int32, device=device)
        self.x = torch.randn((m, HIDDEN), dtype=torch.bfloat16, device=device)
        self.sorted_ids = torch.empty(
            self.max_padded, dtype=torch.int32, device=device
        )
        self.expert_ids = torch.empty(
            self.max_mblocks, dtype=torch.int32, device=device
        )
        self.num_tokens_padded = torch.empty(1, dtype=torch.int32, device=device)
        self.qx = torch.empty_like(self.x, dtype=torch.float8_e4m3fn)
        self.x_scale = torch.empty(
            (m, HIDDEN // 128), dtype=torch.float32, device=device
        )
        self.counters = torch.empty(8, dtype=torch.int32, device=device)
        self.readiness = torch.empty(
            self.max_mblocks, dtype=torch.int32, device=device
        )
        self.ready_queue = torch.empty_like(self.readiness)
        self.ready_valid = torch.empty_like(self.readiness)
        self.w13_owner = torch.empty(
            self.max_mblocks * W13_TILES_TP4,
            dtype=torch.int32,
            device=device,
        )
        self.w13_order = torch.empty_like(self.w13_owner)
        self.w2_owner = torch.empty(
            self.max_mblocks * W2_TILES, dtype=torch.int32, device=device
        )
        self.w2_mblock = torch.empty_like(self.w2_owner)
        self.w2_order = torch.empty_like(self.w2_owner)

    def run(self, num_sms: int) -> None:
        self.counters.zero_()
        self.readiness.zero_()
        self.ready_queue.fill_(-1)
        self.ready_valid.zero_()
        self.w13_owner.fill_(-1)
        self.w13_order.fill_(-1)
        self.w2_owner.fill_(-1)
        self.w2_mblock.fill_(-1)
        self.w2_order.fill_(-1)
        kernel.fused_route_quant(
            self.topk_ids,
            self.x,
            self.sorted_ids,
            self.expert_ids,
            self.num_tokens_padded,
            self.qx.view(torch.uint8),
            self.x_scale,
        )
        kernel.interleaved_scheduler_probe(
            self.expert_ids,
            self.num_tokens_padded,
            self.counters,
            self.readiness,
            self.ready_queue,
            self.ready_valid,
            self.w13_owner,
            self.w13_order,
            self.w2_owner,
            self.w2_mblock,
            self.w2_order,
            W13_TILES_TP4,
            W2_TILES,
            num_sms,
        )


def expected_mblocks(topk_ids: torch.Tensor) -> int:
    counts = torch.bincount(topk_ids.flatten().cpu(), minlength=EXPERTS)
    return int(((counts + 7) // 8).sum().item())


def validate(case: ProbeCase, pattern: str, num_sms: int) -> dict[str, object]:
    torch.cuda.synchronize()
    expected_blocks = expected_mblocks(case.topk_ids)
    actual_blocks = int(case.num_tokens_padded.item()) // 8
    counters = case.counters.cpu().tolist()
    total_w13 = expected_blocks * W13_TILES_TP4
    total_w2 = expected_blocks * W2_TILES

    queue = case.ready_queue[:expected_blocks].cpu().tolist()
    w2_mblocks = case.w2_mblock[:total_w2].cpu().tolist()
    expected_w2_pairs = [
        (mblock, n_tile)
        for mblock in range(expected_blocks)
        for n_tile in range(W2_TILES)
    ]
    actual_w2_pairs = sorted(
        (mblock, index % W2_TILES)
        for index, mblock in enumerate(w2_mblocks)
    )
    w13_orders = case.w13_order[:total_w13].cpu()
    w2_orders = case.w2_order[:total_w2].cpu()
    overlap = bool(
        total_w13
        and total_w2
        and int(w2_orders.min()) < int(w13_orders.max())
    )
    valid = all(
        (
            actual_blocks == expected_blocks,
            counters[2] == expected_blocks,
            counters[4] == total_w13,
            counters[5] == total_w2,
            counters[6] == 0,
            counters[7] == expected_blocks,
            sorted(queue) == list(range(expected_blocks)),
            torch.equal(
                case.readiness[:expected_blocks].cpu(),
                torch.full((expected_blocks,), W13_TILES_TP4, dtype=torch.int32),
            ),
            bool((case.ready_valid[:expected_blocks] == 1).all()),
            bool((case.w13_owner[:total_w13] >= 0).all()),
            bool((case.w13_owner[:total_w13] < num_sms).all()),
            bool((case.w2_owner[:total_w2] >= 0).all()),
            bool((case.w2_owner[:total_w2] < num_sms).all()),
            actual_w2_pairs == expected_w2_pairs,
        )
    )
    result = {
        "m": case.m,
        "pattern": pattern,
        "expected_mblocks": expected_blocks,
        "actual_mblocks": actual_blocks,
        "w13_tasks": total_w13,
        "w2_tasks": total_w2,
        "w13_claim_cursor": counters[0],
        "w2_claim_cursor": counters[1],
        "interleaved_observed": overlap,
        "violations": counters[6],
        "valid": valid,
    }
    print("SCHED_PROBE " + json.dumps(result, sort_keys=True), flush=True)
    if not valid:
        raise RuntimeError(f"scheduler probe failed: {result}")
    return result


@torch.inference_mode()
def main() -> None:
    torch.cuda.set_device(0)
    device = torch.device("cuda:0")
    props = torch.cuda.get_device_properties(device)
    if props.multi_processor_count != 78:
        raise RuntimeError(f"expected H20 with 78 SMs, got {props.multi_processor_count}")

    # Eager coverage spans low/high task counts and route distributions.
    for m in (8, 32, 128):
        for pattern in ("balanced", "skew", "random"):
            case = ProbeCase(m, device)
            case.topk_ids.copy_(make_routes(m, pattern, device))
            case.run(props.multi_processor_count)
            validate(case, pattern, props.multi_processor_count)

    # One captured graph must consume changed route IDs on each replay.
    case = ProbeCase(32, device)
    case.topk_ids.copy_(make_routes(32, "balanced", device))
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        case.run(props.multi_processor_count)
    mutation_blocks: list[int] = []
    for pattern in ("balanced", "skew", "random", "balanced"):
        case.topk_ids.copy_(make_routes(32, pattern, device))
        graph.replay()
        result = validate(case, f"graph-{pattern}", props.multi_processor_count)
        mutation_blocks.append(int(result["actual_mblocks"]))
    if len(set(mutation_blocks)) < 2:
        raise RuntimeError("route mutation did not change device task bounds")
    print(
        "SCHED_PROBE_OK "
        + json.dumps(
            {"graph_mutation_mblocks": mutation_blocks, "num_sms": 78},
            sort_keys=True,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
