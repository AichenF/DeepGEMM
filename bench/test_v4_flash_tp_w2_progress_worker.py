#!/usr/bin/env python3
"""TP4 gate for W2 ready-queue consumers and multicast slot coverage."""

from __future__ import annotations

import argparse
import json
import os

import torch
import torch.distributed as dist

from sglang.kernels.ops.communication.mp import register_comm_cleanup
from sglang.srt.distributed.device_communicators.custom_all_reduce_v2 import (
    CustomAllReduceV2,
)

import profile_v4_flash_tp_local as factory
import v4_flash_tp_wgmma as kernel
import v4_flash_tp_wgmma_graph as custom


@torch.inference_mode()
def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--m", type=int, default=8)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--chunks", type=int, choices=(2, 4, 8), default=4)
    parser.add_argument(
        "--launch-mode", choices=("sequential", "concurrent"),
        default="sequential"
    )
    parser.add_argument(
        "--route-pattern", choices=("random", "balanced", "skew"),
        default="random"
    )
    args = parser.parse_args()
    kernel.W2_PROGRESS_CHUNKS = args.chunks

    rank, world_size, device, cpu_group = custom.init_distributed()
    if world_size != 4:
        raise RuntimeError("worker probe requires TP4")
    torch.manual_seed(20260902 + rank)
    torch.cuda.manual_seed(20260902 + rank)
    case = factory.make_custom_case(
        args.m, 4, args.route_pattern, 20260902, device
    )
    comm = CustomAllReduceV2(cpu_group, device)
    if comm.disabled:
        raise RuntimeError("SGLang CustomAllReduceV2 is disabled")
    register_comm_cleanup(comm)
    case.prepare_fused_push(comm)
    assert case.down is not None
    assert case.activation_scale is not None
    assert case.fused_push_counter is not None
    assert case.fused_push_workspaces is not None
    if not case.fused_push_mc_ptr:
        raise RuntimeError("worker probe requires multicast memory")
    comm._symm_tensor.zero_()
    case.fused_push_counter.zero_()
    torch.cuda.synchronize(device)
    dist.barrier(group=cpu_group)
    if rank == 0:
        print("W2_PROGRESS_STAGE setup_complete", flush=True)

    case.run_before_w2()
    case.w2_progress_state.zero_()
    main_stream = torch.cuda.current_stream(device)
    phase = int(case.fused_push_counter[0].item()) & 1
    if args.launch_mode == "concurrent":
        case.pipeline_start_event.record(main_stream)
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
    if args.launch_mode == "concurrent":
        with torch.cuda.stream(case.pipeline_stream):
            case.pipeline_stream.wait_event(case.pipeline_start_event)
            kernel.progress_k6_mc_push_tp4(
                case.down,
                case.topk_weights,
                case.w2_progress_state,
                case.fused_push_counter,
                case.fused_push_mc_ptr,
                case.fused_push_rank,
                case.fused_push_stride,
                args.workers,
            )
    if args.launch_mode == "sequential":
        torch.cuda.synchronize(device)
        dist.barrier(group=cpu_group)
        if rank == 0:
            print("W2_PROGRESS_STAGE publication_complete", flush=True)
        kernel.progress_k6_mc_push_tp4(
            case.down,
            case.topk_weights,
            case.w2_progress_state,
            case.fused_push_counter,
            case.fused_push_mc_ptr,
            case.fused_push_rank,
            case.fused_push_stride,
            args.workers,
        )
    else:
        with torch.cuda.stream(case.pipeline_stream):
            case.pipeline_done_event.record(case.pipeline_stream)
        main_stream.wait_event(case.pipeline_done_event)
    if rank == 0:
        print(
            f"W2_PROGRESS_STAGE launches_submitted mode={args.launch_mode}",
            flush=True,
        )
    torch.cuda.synchronize(device)
    dist.barrier(group=cpu_group)
    if rank == 0:
        print("W2_PROGRESS_STAGE worker_complete", flush=True)

    state = case.w2_progress_state.cpu()
    marker_end = args.m * 6 * 32
    total_tasks = args.m * args.chunks
    local_state = {
        "rank": rank,
        "launch_mode": args.launch_mode,
        "chunks": args.chunks,
        "marker_min": int(state[:marker_end].min().item()),
        "marker_max": int(state[:marker_end].max().item()),
        "marker_sum": int(state[:marker_end].sum().item()),
        "task_done": int(state[marker_end].item()),
        "worker_done": int(state[marker_end + 1].item()),
    }

    nbytes = args.m * 4096 * torch.bfloat16.itemsize
    phase_base = phase * case.fused_push_stride * world_size
    local_workspace = case.fused_push_workspaces[rank]
    source_nonzero_words = []
    for source in range(world_size):
        begin = phase_base + source * case.fused_push_stride
        words = local_workspace[begin : begin + nbytes].view(torch.int32)
        source_nonzero_words.append(int(torch.count_nonzero(words).item()))
    local_state["phase"] = phase
    local_state["source_nonzero_words"] = source_nonzero_words
    local_state["expected_words_per_source"] = nbytes // 4
    local_state["pass"] = bool(
        local_state["marker_min"] == local_state["marker_max"] == 1
        and local_state["marker_sum"] == marker_end
        and local_state["task_done"] == total_tasks
        and local_state["worker_done"] == args.workers
        and all(count == nbytes // 4 for count in source_nonzero_words)
    )
    gathered = [None for _ in range(world_size)] if rank == 0 else None
    dist.gather_object(local_state, gathered, dst=0, group=cpu_group)
    if rank == 0:
        assert gathered is not None
        print(
            "W2_PROGRESS_WORKER "
            + json.dumps({"ranks": gathered}, sort_keys=True),
            flush=True,
        )
        if not all(bool(record["pass"]) for record in gathered):
            raise SystemExit("W2_PROGRESS_WORKER_WRONG")
        print("W2_PROGRESS_WORKER_OK", flush=True)
    dist.barrier(group=cpu_group)
    os._exit(0)


if __name__ == "__main__":
    main()
