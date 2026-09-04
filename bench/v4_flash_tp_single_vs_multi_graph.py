#!/usr/bin/env python3
"""Paired cold-L2 comparison of single- and multi-kernel TP MegaMoE."""

from __future__ import annotations

import argparse
import json
import statistics
from typing import Any

import torch
import torch.distributed as dist
from triton import runtime as triton_runtime

import sglang.srt.distributed.parallel_state as ps
from sglang.kernels.ops.communication.mp import register_comm_cleanup
from sglang.srt.distributed.device_communicators.custom_all_reduce_v2 import (
    CustomAllReduceV2,
)

import v4_flash_tp_wgmma as kernel
import v4_flash_tp_wgmma_graph as custom
from v4_flash_tp_paired_graph import capture_graph, time_graph_pair


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ms", default="8,16,32,64,128")
    parser.add_argument(
        "--route-pattern", choices=("random", "balanced", "skew"), default="random"
    )
    parser.add_argument("--outer", type=int, default=10)
    parser.add_argument("--replays", type=int, default=200)
    parser.add_argument("--warmup-replays", type=int, default=20)
    parser.add_argument(
        "--pair-granularity", choices=("batch", "replay"), default="batch"
    )
    parser.add_argument("--seed", type=int, default=20260902)
    args = parser.parse_args()
    args.ms = tuple(int(value) for value in args.ms.split(",") if value)
    if not args.ms or any(value <= 0 for value in args.ms):
        parser.error("--ms must contain positive integers")
    if args.outer < 1 or args.replays < 1 or args.warmup_replays < 1:
        parser.error("timing loop counts must be positive")
    if args.pair_granularity == "batch" and args.outer % 2:
        parser.error("batch pairing requires an even --outer")
    return args


def make_case(
    m: int,
    qx: torch.Tensor,
    x_scale: torch.Tensor,
    topk_ids: torch.Tensor,
    topk_weights: torch.Tensor,
    weights: tuple[torch.Tensor, ...],
    lut: torch.Tensor,
    intermediate_per_rank: int,
    use_native: bool = False,
) -> custom.CapturedCase:
    w13, s13, g13, w2, s2, g2 = weights[:6]
    native_w13, native_w2 = weights[6:] if use_native else (None, None)
    return custom.CapturedCase(
        m=m,
        qx=qx,
        x_scale=x_scale,
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
        native_w13=native_w13,
        native_w2=native_w2,
    )


def tensor_comparison_metrics(
    actual: torch.Tensor,
    reference: torch.Tensor,
    nccl_group: dist.ProcessGroup,
    device: torch.device,
) -> dict[str, float | bool]:
    """Compare rank-local tensors while reporting the worst TP rank."""
    actual_f = actual.double()
    reference_f = reference.double()
    diff = actual_f - reference_f
    cosine = float(
        torch.nn.functional.cosine_similarity(
            actual_f.flatten(), reference_f.flatten(), dim=0
        ).item()
    )
    rel_l2 = float(
        (
            torch.linalg.vector_norm(diff)
            / torch.linalg.vector_norm(reference_f).clamp_min(1e-40)
        ).item()
    )
    finite = float(
        bool(torch.isfinite(actual).all())
        and bool(torch.isfinite(reference).all())
    )
    return {
        "cosine_min_rank": custom.reduce_rank_metric(
            cosine, dist.ReduceOp.MIN, device, nccl_group
        ),
        "rel_l2_max_rank": custom.reduce_rank_metric(
            rel_l2, dist.ReduceOp.MAX, device, nccl_group
        ),
        "finite_all_ranks": bool(
            custom.reduce_rank_metric(
                finite, dist.ReduceOp.MIN, device, nccl_group
            )
        ),
    }


@torch.inference_mode()
def main() -> None:
    args = parse_args()
    rank, world_size, device, cpu_group = custom.init_distributed()
    if world_size != 4:
        raise RuntimeError("single-vs-multi performance harness currently requires TP4")
    nccl_group = ps._WORLD.device_group
    if not isinstance(nccl_group, dist.ProcessGroup):
        raise RuntimeError("SGLang did not create the NCCL process group")

    props = torch.cuda.get_device_properties(device)
    intermediate_per_rank = custom.INTERMEDIATE // world_size
    torch.manual_seed(args.seed + rank)
    torch.cuda.manual_seed(args.seed + rank)
    weights = custom.make_weights(
        intermediate_per_rank, device, include_native=True
    )
    lut = kernel.make_e2m1_e8m0_lut(device)
    comm = CustomAllReduceV2(cpu_group, device)
    if comm.disabled:
        raise RuntimeError("SGLang CustomAllReduceV2 is disabled")
    register_comm_cleanup(comm)
    l2_flush_buffer = triton_runtime.driver.active.get_empty_cache_for_benchmark()
    if l2_flush_buffer.nbytes < 2 * props.L2_cache_size:
        raise RuntimeError("benchmark cache buffer is smaller than twice L2")

    if rank == 0:
        print(
            "SINGLE_MULTI_ENV "
            + json.dumps(
                {
                    "benchmark": "v4_flash_tp_single_vs_multi_cuda_graph",
                    "gpu": props.name,
                    "sm_count": props.multi_processor_count,
                    "world_size": world_size,
                    "m_values": args.ms,
                    "route_pattern": args.route_pattern,
                    "outer": args.outer,
                    "replays_per_outer_per_impl": args.replays,
                    "warmup_replays": args.warmup_replays,
                    "pair_granularity": args.pair_granularity,
                    "l2_policy": (
                        "cold; separate 256MiB Triton clear immediately before "
                        "every implementation replay, clear excluded from events"
                    ),
                    "input_contract": (
                        "shared FP8-E4M3 X + FP32 group128 scale; "
                        "BF16-to-FP8 quantization outside timed graphs"
                    ),
                    "native_megamoe": True,
                    "single_launch_interleaved": (
                        kernel.SINGLE_LAUNCH_INTERLEAVED
                    ),
                    "single_launch_schedule": kernel.SINGLE_LAUNCH_SCHEDULE,
                    "single_launch_noinline_gemm": (
                        kernel.SINGLE_LAUNCH_NOINLINE_GEMM
                    ),
                    "single_launch_min_blocks": (
                        kernel.SINGLE_LAUNCH_MIN_BLOCKS
                    ),
                    "single_launch_ctas_per_sm": (
                        kernel.SINGLE_LAUNCH_CTAS_PER_SM
                    ),
                    "control": "selected multi-kernel path from the same source",
                    "candidate": "one tp4_megamoe_single_launch_kernel graph node",
                },
                sort_keys=True,
            ),
            flush=True,
        )

    records: list[dict[str, Any]] = []
    keepalive: list[Any] = []
    driver = triton_runtime.driver.active
    for m in args.ms:
        topk_ids, topk_weights = custom.make_routes(
            m, args.route_pattern, device, args.seed
        )
        qx, x_scale = custom.make_fp8_input(m, device, args.seed)
        control_case = make_case(
            m,
            qx,
            x_scale,
            topk_ids,
            topk_weights,
            weights,
            lut,
            intermediate_per_rank,
        )
        candidate_case = make_case(
            m,
            qx,
            x_scale,
            topk_ids,
            topk_weights,
            weights,
            lut,
            intermediate_per_rank,
            use_native=True,
        )

        kernel.SINGLE_LAUNCH_TP4 = False
        control_graph = capture_graph(control_case, comm, cpu_group, device)
        kernel.SINGLE_LAUNCH_TP4 = True
        candidate_graph = capture_graph(candidate_case, comm, cpu_group, device)

        control_check = custom.correctness_metrics(
            control_case, control_graph, nccl_group, device
        )
        candidate_check = custom.correctness_metrics(
            candidate_case, candidate_graph, nccl_group, device
        )
        assert candidate_case.native_local_output is not None
        native_local_raw = candidate_case.native_local_output.clone()
        native_local_scaled = (
            native_local_raw.float() * custom.ROUTED_SCALING_FACTOR
        ).to(torch.bfloat16)
        local_reference = candidate_case.make_reference_case().run_local().clone()
        torch.cuda.synchronize(device)
        native_local_raw_check = tensor_comparison_metrics(
            native_local_raw, local_reference, nccl_group, device
        )
        native_local_scaled_check = tensor_comparison_metrics(
            native_local_scaled, local_reference, nccl_group, device
        )
        assert candidate_case.graph_output is not None
        native_nccl_reference = native_local_raw.clone()
        dist.all_reduce(native_nccl_reference, group=nccl_group)
        native_nccl_reference = (
            native_nccl_reference.float() * custom.ROUTED_SCALING_FACTOR
        ).to(torch.bfloat16)
        native_embedded_comm_check = tensor_comparison_metrics(
            candidate_case.graph_output,
            native_nccl_reference,
            nccl_group,
            device,
        )
        native_l2_check = None
        native_l2_unweighted_check = None
        native_l2_scale_check = None
        native_l2_byte_mismatches = None
        native_l2_cross_route_rank0 = None
        native_combine_route_check = None
        native_combine_sum_check = None
        if args.route_pattern == "balanced" and m * custom.TOP_K <= custom.NUM_EXPERTS:
            assert candidate_case.native_workspace is not None
            assert control_case.activation_scale is not None
            route_indices = torch.arange(m * custom.TOP_K, device=device)
            pool_rows = route_indices * 8
            native_l2_q = candidate_case.native_workspace.l2_acts.index_select(
                0, pool_rows
            )
            control_l2_q = control_case.qactivation[: m * custom.TOP_K]
            scale_groups = intermediate_per_rank // 128
            native_l2_scale = (
                candidate_case.native_workspace.l2_acts_sf[
                    :scale_groups, pool_rows
                ]
                .T.contiguous()
            )
            control_l2_scale = control_case.activation_scale[
                : m * custom.TOP_K
            ].reshape(m * custom.TOP_K, scale_groups)
            native_l2 = native_l2_q.float() * native_l2_scale.repeat_interleave(
                128, dim=1
            )
            control_l2 = control_l2_q.float() * control_l2_scale.repeat_interleave(
                128, dim=1
            )
            native_l2_unweighted_check = tensor_comparison_metrics(
                native_l2, control_l2, nccl_group, device
            )
            route_weights = topk_weights.reshape(-1, 1)
            native_l2_check = tensor_comparison_metrics(
                native_l2, control_l2 * route_weights, nccl_group, device
            )
            native_l2_scale_check = tensor_comparison_metrics(
                native_l2_scale, control_l2_scale, nccl_group, device
            )
            byte_mismatches = float(
                (native_l2_q.view(torch.uint8) != control_l2_q.view(torch.uint8))
                .sum()
                .item()
            )
            native_l2_byte_mismatches = custom.reduce_rank_metric(
                byte_mismatches, dist.ReduceOp.MAX, device, nccl_group
            )
            if rank == 0:
                native_rows = torch.nn.functional.normalize(
                    native_l2.double(), dim=1
                )
                control_rows = torch.nn.functional.normalize(
                    control_l2.double(), dim=1
                )
                route_cosine = native_rows @ control_rows.T
                best_cosine, best_route = route_cosine.max(dim=1)
                native_l2_cross_route_rank0 = {
                    "diagonal_cosine_mean": float(
                        route_cosine.diagonal().mean().item()
                    ),
                    "best_cosine_mean": float(best_cosine.mean().item()),
                    "best_cosine_max": float(best_cosine.max().item()),
                    "best_cosine_min": float(best_cosine.min().item()),
                    "best_route_is_diagonal_fraction": float(
                        (best_route == route_indices).double().mean().item()
                    ),
                    "best_route_first_16": best_route[:16].tolist(),
                }
            if control_case.down is not None:
                native_combine = (
                    candidate_case.native_workspace.combine[
                        : custom.TOP_K, :m
                    ]
                    .permute(1, 0, 2)
                    .reshape(m * custom.TOP_K, custom.HIDDEN)
                )
                control_weighted_routes = (
                    control_case.down.float() * route_weights
                )
                native_combine_route_check = tensor_comparison_metrics(
                    native_combine,
                    control_weighted_routes,
                    nccl_group,
                    device,
                )
                native_combine_sum = (
                    native_combine.float()
                    .view(m, custom.TOP_K, custom.HIDDEN)
                    .sum(dim=1)
                    .to(torch.bfloat16)
                )
                native_combine_sum_check = tensor_comparison_metrics(
                    native_local_raw,
                    native_combine_sum,
                    nccl_group,
                    device,
                )
        if rank == 0:
            print(
                "SINGLE_MULTI_CORRECTNESS "
                + json.dumps(
                    {
                        "m": m,
                        "control_final": control_check,
                        "candidate_final": candidate_check,
                        "candidate_local_raw": native_local_raw_check,
                        "candidate_local_scaled_1p5": native_local_scaled_check,
                        "candidate_embedded_comm_vs_native_nccl": (
                            native_embedded_comm_check
                        ),
                        "candidate_l2_dequant": native_l2_check,
                        "candidate_l2_dequant_vs_unweighted_control": (
                            native_l2_unweighted_check
                        ),
                        "candidate_l2_scale": native_l2_scale_check,
                        "candidate_l2_fp8_byte_mismatches_max_rank": (
                            native_l2_byte_mismatches
                        ),
                        "candidate_l2_cross_route_rank0": (
                            native_l2_cross_route_rank0
                        ),
                        "candidate_combine_routes": native_combine_route_check,
                        "candidate_local_vs_own_combine_sum": (
                            native_combine_sum_check
                        ),
                    },
                    sort_keys=True,
                ),
                flush=True,
            )
        if not control_check["allreduce_ok"] or not candidate_check["allreduce_ok"]:
            raise RuntimeError(f"correctness failure at M={m}")

        for warmup_idx in range(args.warmup_replays):
            order = (
                (candidate_graph, control_graph)
                if warmup_idx & 1
                else (control_graph, candidate_graph)
            )
            for graph in order:
                driver.clear_cache(l2_flush_buffer)
                graph.replay()
        torch.cuda.synchronize(device)

        (
            control_samples,
            candidate_samples,
            control_batch_medians,
            candidate_batch_medians,
        ) = time_graph_pair(
            control_graph,
            candidate_graph,
            args.outer,
            args.replays,
            cpu_group,
            nccl_group,
            device,
            l2_flush_buffer,
            args.pair_granularity,
        )
        control_median = statistics.median(control_samples)
        candidate_median = statistics.median(candidate_samples)
        phase_us: dict[str, float] = {}
        record: dict[str, Any] = {
            "m": m,
            "active_experts": candidate_case.active_experts,
            "routed_rows": m * custom.TOP_K,
            "control_padded_rows": int(control_case.num_tokens_padded.item()),
            "candidate_padded_rows": int(candidate_case.num_tokens_padded.item()),
            "w13_split_k": candidate_case.w13_split_k,
            "control_ar_mode": control_case.fused_k6_ar_mode,
            "candidate_ar_mode": candidate_case.fused_k6_ar_mode,
            "cold_samples_per_impl": len(control_samples),
            "control_latency_ms_min": min(control_samples),
            "control_latency_ms_median": control_median,
            "control_latency_ms_max": max(control_samples),
            "candidate_latency_ms_min": min(candidate_samples),
            "candidate_latency_ms_median": candidate_median,
            "candidate_latency_ms_max": max(candidate_samples),
            "candidate_over_control": candidate_median / control_median,
            "speedup_control_over_candidate": control_median / candidate_median,
            "control_batch_medians_ms_max_rank": control_batch_medians,
            "candidate_batch_medians_ms_max_rank": candidate_batch_medians,
            "candidate_device_phase_us_rank_max": phase_us,
            "control_correctness": control_check,
            "candidate_correctness": candidate_check,
        }
        records.append(record)
        if rank == 0:
            print(
                "SINGLE_MULTI_RESULT " + json.dumps(record, sort_keys=True),
                flush=True,
            )
        keepalive.extend((control_case, candidate_case, control_graph, candidate_graph))

    if rank == 0:
        control_geomean = statistics.geometric_mean(
            float(record["control_latency_ms_median"]) for record in records
        )
        candidate_geomean = statistics.geometric_mean(
            float(record["candidate_latency_ms_median"]) for record in records
        )
        print(
            "SINGLE_MULTI_SUMMARY "
            + json.dumps(
                {
                    "m_values": list(args.ms),
                    "control_geometric_mean_ms": control_geomean,
                    "candidate_geometric_mean_ms": candidate_geomean,
                    "candidate_over_control": candidate_geomean / control_geomean,
                    "speedup_control_over_candidate": control_geomean / candidate_geomean,
                    "samples_per_m_per_impl": args.outer * args.replays,
                },
                sort_keys=True,
            ),
            flush=True,
        )
    dist.barrier(group=cpu_group)


if __name__ == "__main__":
    main()
