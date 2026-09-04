#!/usr/bin/env python3
"""Cold-L2 TP4 A/B sweep of CARv2 graph algorithms and pull-block counts."""

from __future__ import annotations

import argparse
import json
import statistics

import torch
import torch.distributed as dist
from triton import runtime as triton_runtime

import sglang.srt.distributed.parallel_state as ps
from sglang.kernels.ops.communication.all_reduce import AllReduceAlgo
try:
    from sglang.jit_kernel.mp import register_comm_cleanup
except ImportError:  # Compatibility with the older benchmark checkout.
    from sglang.kernels.ops.communication.mp import register_comm_cleanup
from sglang.srt.distributed.device_communicators.custom_all_reduce_v2 import (
    CustomAllReduceV2,
)

import v4_flash_tp_paired_graph as paired
import v4_flash_tp_wgmma as kernel
import v4_flash_tp_wgmma_graph as custom


_ALGOS = {
    "1shot_pull": AllReduceAlgo.ONE_SHOT_PULL,
    "2shot_pull": AllReduceAlgo.TWO_SHOT_PULL,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ms", default="64,128")
    parser.add_argument("--algos", default="1shot_pull,2shot_pull")
    parser.add_argument("--pull-blocks", default="24,32,40,48,56,64")
    parser.add_argument(
        "--route-pattern", choices=("random", "balanced", "skew"), default="random"
    )
    parser.add_argument("--outer", type=int, default=4)
    parser.add_argument("--replays", type=int, default=100)
    parser.add_argument("--warmup-replays", type=int, default=6)
    parser.add_argument("--seed", type=int, default=20260902)
    args = parser.parse_args()
    args.ms = tuple(int(value) for value in args.ms.split(",") if value)
    args.algos = tuple(value for value in args.algos.split(",") if value)
    args.pull_blocks = tuple(
        int(value) for value in args.pull_blocks.split(",") if value
    )
    if not args.ms or any(value not in (64, 128) for value in args.ms):
        parser.error("--ms must be a nonempty subset of 64,128")
    if not args.algos or any(value not in _ALGOS for value in args.algos):
        parser.error(f"--algos must contain values from {tuple(_ALGOS)}")
    if not args.pull_blocks or any(value <= 0 for value in args.pull_blocks):
        parser.error("--pull-blocks must contain positive integers")
    if args.outer < 2 or args.outer % 2:
        parser.error("--outer must be positive and even for balanced AB/BA")
    if args.replays < 1 or args.warmup_replays < 1:
        parser.error("replay counts must be positive")
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


def set_pull_policy(
    comm: CustomAllReduceV2,
    algo: AllReduceAlgo | None,
    blocks: int,
) -> None:
    comm.override_algo = algo
    comm.obj.config(num_pull_blocks=blocks)


@torch.inference_mode()
def main() -> None:
    args = parse_args()
    rank, world_size, device, cpu_group = custom.init_distributed()
    if world_size != 4:
        raise RuntimeError("CAR policy audit is specialized for TP4")
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
    allocated_pull_blocks = comm.config.num_pull_blocks
    if any(value > allocated_pull_blocks for value in args.pull_blocks):
        raise RuntimeError(
            f"requested pull blocks exceed allocated semaphore rows "
            f"({allocated_pull_blocks})"
        )
    l2_flush_buffer = triton_runtime.driver.active.get_empty_cache_for_benchmark()
    if l2_flush_buffer.nbytes < 2 * props.L2_cache_size:
        raise RuntimeError("cache-clear buffer is smaller than twice L2")

    # Keep the accepted M<=32 fused multicast specialization out of this
    # experiment. M64/M128 must call stock CARv2 for both graph variants.
    kernel.FUSED_K6_MC_PUSH_MAX_M = 32

    if rank == 0:
        print(
            "CAR_POLICY_AB_ENV "
            + json.dumps(
                {
                    "benchmark": "same-process CARv2 policy versus stock graph CARv2",
                    "gpu": props.name,
                    "sm_count": props.multi_processor_count,
                    "world_size": world_size,
                    "m_values": args.ms,
                    "candidate_algorithms": args.algos,
                    "candidate_pull_blocks": args.pull_blocks,
                    "control": {
                        "algorithm": "stock graph heuristic (2shot_pull)",
                        "pull_blocks": allocated_pull_blocks,
                    },
                    "route_pattern": args.route_pattern,
                    "outer": args.outer,
                    "replays_per_outer_per_impl": args.replays,
                    "pair_order": "candidate/control then control/candidate by batch",
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
        nbytes = m * custom.HIDDEN * torch.empty((), dtype=torch.bfloat16).element_size()
        default_algo, default_mode = comm._pick_algo(nbytes, can_use_graph=True)
        if default_algo != AllReduceAlgo.TWO_SHOT_PULL:
            raise RuntimeError(
                f"expected stock graph 2shot_pull at M={m}, got "
                f"{default_algo}/{default_mode}"
            )
        topk_ids, topk_weights = custom.make_routes(
            m, args.route_pattern, device, args.seed
        )
        x = torch.randn((m, custom.HIDDEN), dtype=torch.bfloat16, device=device) * 0.1

        control_case = make_case(
            m, x, topk_ids, topk_weights, weights, lut, intermediate_per_rank
        )
        set_pull_policy(comm, None, allocated_pull_blocks)
        control_graph = paired.capture_graph(control_case, comm, cpu_group, device)
        if control_case.fused_k6_push_active:
            raise RuntimeError(f"control unexpectedly selected fused AR at M={m}")
        control_check = custom.correctness_metrics(
            control_case, control_graph, nccl_group, device
        )
        if not control_check["allreduce_ok"]:
            raise RuntimeError(f"stock control correctness failure at M={m}")

        for algo_name in args.algos:
            for blocks in args.pull_blocks:
                if algo_name == "2shot_pull" and blocks == allocated_pull_blocks:
                    continue
                candidate_case = make_case(
                    m,
                    x,
                    topk_ids,
                    topk_weights,
                    weights,
                    lut,
                    intermediate_per_rank,
                )
                set_pull_policy(comm, _ALGOS[algo_name], blocks)
                candidate_graph = paired.capture_graph(
                    candidate_case, comm, cpu_group, device
                )
                set_pull_policy(comm, None, allocated_pull_blocks)
                if candidate_case.fused_k6_push_active:
                    raise RuntimeError(
                        f"candidate unexpectedly selected fused AR at M={m}"
                    )
                candidate_check = custom.correctness_metrics(
                    candidate_case, candidate_graph, nccl_group, device
                )
                if not candidate_check["allreduce_ok"]:
                    raise RuntimeError(
                        f"candidate correctness failure at M={m}, "
                        f"algo={algo_name}, blocks={blocks}"
                    )

                candidate_graph.replay()
                control_graph.replay()
                torch.cuda.synchronize(device)
                graph_max_abs = torch.max(
                    torch.abs(
                        candidate_case.graph_output.float()
                        - control_case.graph_output.float()
                    )
                )
                dist.all_reduce(graph_max_abs, op=dist.ReduceOp.MAX, group=nccl_group)

                for warmup_idx in range(args.warmup_replays):
                    if warmup_idx & 1:
                        control_graph.replay()
                        candidate_graph.replay()
                    else:
                        candidate_graph.replay()
                        control_graph.replay()
                torch.cuda.synchronize(device)

                (
                    candidate_samples,
                    control_samples,
                    candidate_batch_medians,
                    control_batch_medians,
                ) = paired.time_graph_pair(
                    candidate_graph,
                    control_graph,
                    args.outer,
                    args.replays,
                    cpu_group,
                    nccl_group,
                    device,
                    l2_flush_buffer,
                    "batch",
                )
                candidate_median = statistics.median(candidate_samples)
                control_median = statistics.median(control_samples)
                wins = sum(
                    candidate < control
                    for candidate, control in zip(
                        candidate_batch_medians, control_batch_medians
                    )
                )
                record: dict[str, object] = {
                    "m": m,
                    "candidate_algorithm": algo_name,
                    "candidate_pull_blocks": blocks,
                    "cold_samples_per_impl": len(candidate_samples),
                    "candidate_latency_ms_min": min(candidate_samples),
                    "candidate_latency_ms_median": candidate_median,
                    "candidate_latency_ms_max": max(candidate_samples),
                    "control_latency_ms_min": min(control_samples),
                    "control_latency_ms_median": control_median,
                    "control_latency_ms_max": max(control_samples),
                    "speedup_control_over_candidate": (
                        control_median / candidate_median
                    ),
                    "candidate_batch_wins": wins,
                    "batch_count": len(candidate_batch_medians),
                    "candidate_batch_medians_ms_max_rank": (
                        candidate_batch_medians
                    ),
                    "control_batch_medians_ms_max_rank": control_batch_medians,
                    "candidate_vs_control_max_abs": float(graph_max_abs.item()),
                    "candidate_correctness": candidate_check,
                    "control_correctness": control_check,
                }
                records.append(record)
                if rank == 0:
                    print(
                        "CAR_POLICY_AB_RESULT "
                        + json.dumps(record, sort_keys=True),
                        flush=True,
                    )

    if rank == 0:
        best_by_m: dict[int, dict[str, object]] = {}
        for m in args.ms:
            m_records = [record for record in records if record["m"] == m]
            best_by_m[m] = max(
                m_records,
                key=lambda record: float(record["speedup_control_over_candidate"]),
            )
        print(
            "CAR_POLICY_AB_SUMMARY "
            + json.dumps(
                {
                    "best_by_m": best_by_m,
                    "candidate_count": len(records),
                },
                sort_keys=True,
            ),
            flush=True,
        )


if __name__ == "__main__":
    main()
