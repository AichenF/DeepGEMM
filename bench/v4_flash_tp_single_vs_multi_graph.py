#!/usr/bin/env python3
"""Paired cold-L2 comparison of single- and multi-kernel TP MegaMoE."""

from __future__ import annotations

import argparse
import json
import os
import statistics
from typing import Any

import torch
import torch.distributed as dist
from triton import runtime as triton_runtime

import sglang.srt.distributed.parallel_state as ps
try:
    from sglang.jit_kernel.mp import register_comm_cleanup
except ImportError:  # Compatibility with the older benchmark checkout.
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
    parser.add_argument(
        "--candidate",
        choices=("tp", "native"),
        default="tp",
        help="single-launch implementation to compare against the multi-kernel control",
    )
    parser.add_argument("--outer", type=int, default=10)
    parser.add_argument("--replays", type=int, default=200)
    parser.add_argument("--warmup-replays", type=int, default=20)
    parser.add_argument(
        "--diagnose-output",
        action="store_true",
        help="Print repeat/chunk error localization before performance timing.",
    )
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
    native_kernel_module: Any | None = None,
) -> custom.CapturedCase:
    w13, s13, g13, w2, s2, g2 = weights[:6]
    native_w13, native_w2, native_g13, native_g2, native_s13, native_s2 = (
        weights[6:] if use_native else (None, None, None, None, None, None)
    )
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
        native_g13=native_g13,
        native_g2=native_g2,
        native_s13=native_s13,
        native_s2=native_s2,
        native_kernel_module=native_kernel_module,
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
    use_native = args.candidate == "native"
    native_kernel = None
    if use_native:
        import v4_flash_tp_native_megamoe as native_kernel

    weights = custom.make_weights(
        intermediate_per_rank, device, include_native=use_native
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
                    "native_megamoe": use_native,
                    "native_register_dequant": bool(
                        native_kernel
                        and native_kernel.NATIVE_REGISTER_DEQUANT
                    ),
                    "native_rs_k128_batch": bool(
                        native_kernel
                        and native_kernel.NATIVE_RS_K128_BATCH
                    ),
                    "native_two_cta_per_sm": bool(
                        native_kernel
                        and native_kernel.NATIVE_TWO_CTA_PER_SM
                    ),
                    "native_skip_cleanup_grid_sync": bool(
                        native_kernel
                        and native_kernel.NATIVE_SKIP_CLEANUP_GRID_SYNC
                    ),
                    "native_rs_half_prefetch": bool(
                        native_kernel
                        and native_kernel.NATIVE_RS_HALF_PREFETCH
                    ),
                    "native_normalized_weight_scale": bool(
                        native_kernel
                        and native_kernel.NATIVE_NORMALIZED_WEIGHT_SCALE
                    ),
                    "native_rs_scale_word_cache": bool(
                        native_kernel
                        and native_kernel.NATIVE_RS_SCALE_WORD_CACHE
                    ),
                    "native_split_weight_scale_tma": bool(
                        native_kernel
                        and native_kernel.NATIVE_SPLIT_WEIGHT_SCALE_TMA
                    ),
                    "native_tile_weight_scale_tma": bool(
                        native_kernel
                        and native_kernel.NATIVE_TILE_WEIGHT_SCALE_TMA
                    ),
                    "native_single_l1_warmup_wave": bool(
                        native_kernel
                        and native_kernel.NATIVE_SINGLE_L1_WARMUP_WAVE
                    ),
                    "native_tp_local_barrier_fastpath": bool(
                        native_kernel
                        and native_kernel.NATIVE_TP_LOCAL_BARRIER_FASTPATH
                    ),
                    "native_tp_local_route_build": bool(
                        native_kernel
                        and native_kernel.NATIVE_TP_LOCAL_ROUTE_BUILD
                    ),
                    "native_tp_local_parallel_combine_chunks": bool(
                        native_kernel
                        and native_kernel.NATIVE_TP_LOCAL_PARALLEL_COMBINE_CHUNKS
                    ),
                    "single_launch_interleaved": (
                        kernel.SINGLE_LAUNCH_INTERLEAVED
                    ),
                    "single_launch_schedule": kernel.SINGLE_LAUNCH_SCHEDULE,
                    "single_launch_noinline_gemm": (
                        kernel.SINGLE_LAUNCH_NOINLINE_GEMM
                    ),
                    "single_launch_w13_phase_noinline": (
                        kernel.SINGLE_LAUNCH_W13_PHASE_NOINLINE
                    ),
                    "single_launch_w13_phase_compact_abi": (
                        kernel.SINGLE_LAUNCH_W13_PHASE_COMPACT_ABI
                    ),
                    "single_launch_w13_wave_rotate": (
                        kernel.SINGLE_LAUNCH_W13_WAVE_ROTATE
                    ),
                    "single_launch_w2_wave_rotate": (
                        kernel.SINGLE_LAUNCH_W2_WAVE_ROTATE
                    ),
                    "single_launch_w2_phase_noinline": (
                        kernel.SINGLE_LAUNCH_W2_PHASE_NOINLINE
                    ),
                    "single_launch_min_blocks": (
                        kernel.SINGLE_LAUNCH_MIN_BLOCKS
                    ),
                    "single_launch_route_dynamic_smem": (
                        kernel.SINGLE_LAUNCH_ROUTE_DYNAMIC_SMEM
                    ),
                    "single_launch_m128_bound9": (
                        kernel.SINGLE_LAUNCH_M128_BOUND9
                    ),
                    "single_launch_m128_unbounded": (
                        kernel.SINGLE_LAUNCH_M128_UNBOUNDED
                    ),
                    "single_launch_compact_w13_bundle": (
                        kernel.SINGLE_LAUNCH_COMPACT_W13_BUNDLE
                    ),
                    "single_launch_persistent_gemm_state": (
                        kernel.SINGLE_LAUNCH_PERSISTENT_GEMM_STATE
                    ),
                    "single_launch_w13_next_task_prefetch": (
                        kernel.SINGLE_LAUNCH_W13_NEXT_TASK_PREFETCH
                    ),
                    "single_launch_w13_tail_split4": (
                        kernel.SINGLE_LAUNCH_W13_TAIL_SPLIT4
                    ),
                    "single_launch_w2_next_task_prefetch": (
                        kernel.SINGLE_LAUNCH_W2_NEXT_TASK_PREFETCH
                    ),
                    "single_launch_assume_valid_gemm_tasks": (
                        kernel.SINGLE_LAUNCH_ASSUME_VALID_GEMM_TASKS
                    ),
                    "single_launch_w2_chunk_major": (
                        kernel.SINGLE_LAUNCH_W2_CHUNK_MAJOR
                    ),
                    "single_launch_w2_chunk_ar_overlap": (
                        kernel.SINGLE_LAUNCH_W2_CHUNK_AR_OVERLAP
                    ),
                    "single_launch_w2_chunk_ar_dedicated": (
                        kernel.SINGLE_LAUNCH_W2_CHUNK_AR_DEDICATED
                    ),
                    "single_launch_w2_chunk_ar_wait_only": (
                        kernel.SINGLE_LAUNCH_W2_CHUNK_AR_WAIT_ONLY
                    ),
                    "single_launch_w2_chunk_ar_helper_stage": (
                        kernel.SINGLE_LAUNCH_W2_CHUNK_AR_HELPER_STAGE
                    ),
                    "single_launch_w2_chunk_ar_strong_producer_fence": (
                        kernel.SINGLE_LAUNCH_W2_CHUNK_AR_STRONG_PRODUCER_FENCE
                    ),
                    "single_launch_w2_chunk_ar_l2_load": (
                        kernel.SINGLE_LAUNCH_W2_CHUNK_AR_L2_LOAD
                    ),
                    "single_launch_w2_chunk_ar_post": (
                        kernel.SINGLE_LAUNCH_W2_CHUNK_AR_POST
                    ),
                    "single_launch_w2_chunk_ar_post_concurrent": (
                        kernel.SINGLE_LAUNCH_W2_CHUNK_AR_POST_CONCURRENT
                    ),
                    "single_launch_w2_bulk_reduce_combine": (
                        kernel.SINGLE_LAUNCH_W2_BULK_REDUCE_COMBINE
                    ),
                    "single_launch_w2_bulk_reduce_routes": (
                        kernel.SINGLE_LAUNCH_W2_BULK_REDUCE_ROUTES
                    ),
                    "single_launch_w2_producer_atomic_combine": (
                        kernel.SINGLE_LAUNCH_W2_PRODUCER_ATOMIC_COMBINE
                    ),
                    "single_launch_w2_f16_wgmma_accum": (
                        kernel.SINGLE_LAUNCH_W2_F16_WGMMA_ACCUM
                    ),
                    "single_launch_w2_f16_pair_inline_asm": (
                        kernel.SINGLE_LAUNCH_W2_F16_PAIR_INLINE_ASM
                    ),
                    "single_launch_cooperative_grid": (
                        kernel.SINGLE_LAUNCH_COOPERATIVE_GRID
                    ),
                    "single_launch_relaxed_grid_poll": (
                        kernel.SINGLE_LAUNCH_RELAXED_GRID_POLL
                    ),
                    "single_launch_hierarchical_grid": (
                        kernel.SINGLE_LAUNCH_HIERARCHICAL_GRID
                    ),
                    "single_launch_grid_poll_sleep_ns": (
                        kernel.SINGLE_LAUNCH_GRID_POLL_SLEEP_NS
                    ),
                    "single_launch_grid_barrier_poll_warp": (
                        kernel.SINGLE_LAUNCH_GRID_BARRIER_POLL_WARP
                    ),
                    "single_launch_adaptive_grid_poll": (
                        kernel.SINGLE_LAUNCH_ADAPTIVE_GRID_POLL
                    ),
                    "single_launch_adaptive_grid_poll_max_ns": (
                        kernel.SINGLE_LAUNCH_ADAPTIVE_GRID_POLL_MAX_NS
                    ),
                    "single_launch_phase_stamps": (
                        kernel.SINGLE_LAUNCH_PHASE_STAMPS
                    ),
                    "single_launch_packed_grid_barrier": (
                        kernel.SINGLE_LAUNCH_PACKED_GRID_BARRIER
                    ),
                    "single_launch_release_grid_arrival": (
                        kernel.SINGLE_LAUNCH_RELEASE_GRID_ARRIVAL
                    ),
                    "single_launch_balanced_workers": (
                        kernel.SINGLE_LAUNCH_BALANCED_WORKERS
                    ),
                    "single_launch_balanced_activation_workers": (
                        kernel.SINGLE_LAUNCH_BALANCED_ACTIVATION_WORKERS
                    ),
                    "single_launch_balanced_w2_workers": (
                        kernel.SINGLE_LAUNCH_BALANCED_W2_WORKERS
                    ),
                    "single_launch_w2_n64_tail": (
                        kernel.SINGLE_LAUNCH_W2_N64_TAIL
                    ),
                    "single_launch_skip_final_cta_sync": (
                        kernel.SINGLE_LAUNCH_SKIP_FINAL_CTA_SYNC
                    ),
                    "single_launch_grid_barrier_no_entry_sync": (
                        kernel.SINGLE_LAUNCH_GRID_BARRIER_NO_ENTRY_SYNC
                    ),
                    "single_launch_skip_activation_task_sync": (
                        kernel.SINGLE_LAUNCH_SKIP_ACTIVATION_TASK_SYNC
                    ),
                    "single_launch_tail_overlap": (
                        kernel.SINGLE_LAUNCH_TAIL_OVERLAP
                    ),
                    "single_launch_tail_act_only": (
                        kernel.SINGLE_LAUNCH_TAIL_ACT_ONLY
                    ),
                    "single_launch_tail_group_ctas": (
                        kernel.SINGLE_LAUNCH_TAIL_GROUP_CTAS
                    ),
                    "single_launch_grouped_w13_act": (
                        kernel.SINGLE_LAUNCH_GROUPED_W13_ACT
                    ),
                    "single_launch_act_w2_cohort": (
                        kernel.SINGLE_LAUNCH_ACT_W2_COHORT
                    ),
                    "single_launch_w13_completion_act": (
                        kernel.SINGLE_LAUNCH_W13_COMPLETION_ACT
                    ),
                    "single_launch_w13_act_tail_pipe": (
                        kernel.SINGLE_LAUNCH_W13_ACT_TAIL_PIPE
                    ),
                    "single_launch_w13_n64_tail": (
                        kernel.SINGLE_LAUNCH_W13_N64_TAIL
                    ),
                    "single_launch_cluster_w13_act": (
                        kernel.SINGLE_LAUNCH_CLUSTER_W13_ACT
                    ),
                    "single_launch_dual_wg_phases": (
                        kernel.SINGLE_LAUNCH_DUAL_WG_PHASES
                    ),
                    "single_launch_78cta_8wg": (
                        kernel.SINGLE_LAUNCH_78CTA_8WG
                    ),
                    "single_launch_156cta_4wg": (
                        kernel.SINGLE_LAUNCH_156CTA_4WG
                    ),
                    "single_launch_78cta_wg_dag": (
                        kernel.SINGLE_LAUNCH_78CTA_WG_DAG
                    ),
                    "single_launch_78cta_local_w13": (
                        kernel.SINGLE_LAUNCH_78CTA_LOCAL_W13
                    ),
                    "single_launch_dual_wg_ctas_per_sm": (
                        kernel.SINGLE_LAUNCH_DUAL_WG_CTAS_PER_SM
                    ),
                    "single_launch_dual_wg_private_act": (
                        kernel.SINGLE_LAUNCH_DUAL_WG_PRIVATE_ACT
                    ),
                    "single_launch_w2_unroll2_bound9": (
                        kernel.SINGLE_LAUNCH_W2_UNROLL2_BOUND9
                    ),
                    "single_launch_ctas_per_sm": (
                        kernel.SINGLE_LAUNCH_CTAS_PER_SM
                    ),
                    "single_launch_p2p_two_shot": (
                        kernel.SINGLE_LAUNCH_P2P_TWO_SHOT
                    ),
                    "single_launch_p2p_two_shot_blocks": (
                        kernel.SINGLE_LAUNCH_P2P_TWO_SHOT_BLOCKS
                    ),
                    "control": "selected multi-kernel path from the same source",
                    "candidate": (
                        "native Hopper MegaMoE kernel"
                        if use_native
                        else "TP-specialized MegaMoE single kernel"
                    ),
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
            use_native=use_native,
        )

        kernel.SINGLE_LAUNCH_TP4 = False
        control_graph = capture_graph(control_case, comm, cpu_group, device)
        kernel.SINGLE_LAUNCH_TP4 = True
        candidate_graph = capture_graph(candidate_case, comm, cpu_group, device)

        if args.diagnose_output:
            assert candidate_case.down is not None
            assert candidate_case.fused_pull_output is not None
            assert candidate_case.fused_pull_sem_local is not None
            if rank == 0:
                def tensor_range(tensor: torch.Tensor) -> dict[str, int]:
                    begin = int(tensor.data_ptr())
                    nbytes = int(tensor.numel() * tensor.element_size())
                    return {"begin": begin, "end": begin + nbytes,
                            "bytes": nbytes}

                print(
                    "SINGLE_MULTI_POINTER_DIAG "
                    + json.dumps(
                        {
                            "down": tensor_range(candidate_case.down),
                            "pull_output": tensor_range(
                                candidate_case.fused_pull_output
                            ),
                            "pull_sem": tensor_range(
                                candidate_case.fused_pull_sem_local
                            ),
                        },
                        sort_keys=True,
                    ),
                    flush=True,
                )
            control_graph.replay()
            torch.cuda.synchronize(device)
            assert control_case.down is not None
            control_down = control_case.down.clone()
            candidate_snapshots: list[torch.Tensor] = []
            candidate_down_snapshots: list[torch.Tensor] = []
            candidate_chunk_state: list[list[int]] = []
            for _ in range(4):
                candidate_graph.replay()
                torch.cuda.synchronize(device)
                assert candidate_case.graph_output is not None
                assert candidate_case.down is not None
                candidate_snapshots.append(candidate_case.graph_output.clone())
                candidate_down_snapshots.append(candidate_case.down.clone())
                packed_words = [
                    int(value) & 0xFFFFFFFF
                    for value in candidate_case.single_launch_barrier_state[
                        18:22
                    ].cpu().tolist()
                ]
                candidate_chunk_state.append(packed_words)
            diagnostic_reference = (
                candidate_case.make_reference_case().run_local().clone()
            )
            dist.all_reduce(diagnostic_reference, group=nccl_group)
            torch.cuda.synchronize(device)
            repeat_metrics: list[dict[str, Any]] = []
            for repeat, snapshot in enumerate(candidate_snapshots):
                chunk_metrics: list[dict[str, float | int]] = []
                for chunk in range(4):
                    begin = chunk * custom.HIDDEN // 4
                    end = (chunk + 1) * custom.HIDDEN // 4
                    actual_chunk = snapshot[:, begin:end].double()
                    reference_chunk = diagnostic_reference[:, begin:end].double()
                    chunk_diff = actual_chunk - reference_chunk
                    chunk_metrics.append(
                        {
                            "chunk": chunk,
                            "max_abs_rank_max": custom.reduce_rank_metric(
                                float(chunk_diff.abs().max()),
                                dist.ReduceOp.MAX,
                                device,
                                nccl_group,
                            ),
                            "rel_l2_rank_max": custom.reduce_rank_metric(
                                float(
                                    torch.linalg.vector_norm(chunk_diff)
                                    / torch.linalg.vector_norm(
                                        reference_chunk
                                    ).clamp_min(1e-40)
                                ),
                                dist.ReduceOp.MAX,
                                device,
                                nccl_group,
                            ),
                            "bf16_mismatches_rank_max": int(
                                custom.reduce_rank_metric(
                                    float(
                                        (snapshot[:, begin:end]
                                         != diagnostic_reference[:, begin:end])
                                        .sum()
                                        .item()
                                    ),
                                    dist.ReduceOp.MAX,
                                    device,
                                    nccl_group,
                                )
                            ),
                        }
                    )
                repeat_metrics.append(
                    {"repeat": repeat, "chunks": chunk_metrics}
                )
            repeat_max_abs = []
            for repeat in range(1, len(candidate_snapshots)):
                repeat_max_abs.append(
                    custom.reduce_rank_metric(
                        float(
                            (candidate_snapshots[repeat]
                             - candidate_snapshots[0]).abs().max()
                        ),
                        dist.ReduceOp.MAX,
                        device,
                        nccl_group,
                    )
                )
            if rank == 0:
                token_max = (
                    candidate_snapshots[-1].double()
                    - diagnostic_reference.double()
                ).abs().amax(dim=1)
                top_values, top_tokens = torch.topk(
                    token_max, k=min(8, token_max.numel())
                )
                print(
                    "SINGLE_MULTI_OUTPUT_DIAG "
                    + json.dumps(
                        {
                            "m": m,
                            "repeat_metrics": repeat_metrics,
                            "repeat_vs_first_max_abs_rank_max": repeat_max_abs,
                            "worst_tokens_rank0": top_tokens.cpu().tolist(),
                            "worst_token_max_abs_rank0": (
                                top_values.cpu().tolist()
                            ),
                        },
                        sort_keys=True,
                    ),
                    flush=True,
                )
            down_repeat_metrics: list[dict[str, Any]] = []
            for repeat, snapshot in enumerate(candidate_down_snapshots):
                chunk_metrics = []
                for chunk in range(4):
                    begin = chunk * custom.HIDDEN // 4
                    end = (chunk + 1) * custom.HIDDEN // 4
                    actual_chunk = snapshot[:, begin:end].double()
                    control_chunk = control_down[:, begin:end].double()
                    chunk_diff = actual_chunk - control_chunk
                    chunk_metrics.append(
                        {
                            "chunk": chunk,
                            "max_abs_rank_max": custom.reduce_rank_metric(
                                float(chunk_diff.abs().max()),
                                dist.ReduceOp.MAX,
                                device,
                                nccl_group,
                            ),
                            "rel_l2_rank_max": custom.reduce_rank_metric(
                                float(
                                    torch.linalg.vector_norm(chunk_diff)
                                    / torch.linalg.vector_norm(
                                        control_chunk
                                    ).clamp_min(1e-40)
                                ),
                                dist.ReduceOp.MAX,
                                device,
                                nccl_group,
                            ),
                            "bf16_mismatches_rank_max": int(
                                custom.reduce_rank_metric(
                                    float(
                                        (snapshot[:, begin:end]
                                         != control_down[:, begin:end])
                                        .sum()
                                        .item()
                                    ),
                                    dist.ReduceOp.MAX,
                                    device,
                                    nccl_group,
                                )
                            ),
                        }
                    )
                down_repeat_metrics.append(
                    {"repeat": repeat, "chunks": chunk_metrics}
                )
            down_repeat_max_abs = []
            for repeat in range(1, len(candidate_down_snapshots)):
                down_repeat_max_abs.append(
                    custom.reduce_rank_metric(
                        float(
                            (candidate_down_snapshots[repeat]
                             - candidate_down_snapshots[0]).abs().max()
                        ),
                        dist.ReduceOp.MAX,
                        device,
                        nccl_group,
                    )
                )
            if rank == 0:
                last_down = candidate_down_snapshots[-1]
                down_mismatch = torch.nonzero(
                    last_down != control_down, as_tuple=False
                )
                padded_rows = int(candidate_case.num_tokens_padded.item())
                sorted_rows = candidate_case.sorted_ids[:padded_rows].long()
                route_mblock = torch.full(
                    (m * custom.TOP_K,), -1, dtype=torch.long, device=device
                )
                sorted_positions = torch.arange(
                    padded_rows, dtype=torch.long, device=device
                )
                valid_sorted = (
                    (sorted_rows >= 0) & (sorted_rows < m * custom.TOP_K)
                )
                route_mblock[sorted_rows[valid_sorted]] = (
                    sorted_positions[valid_sorted] // 8
                )
                task_owners: list[dict[str, int | bool]] = []
                top_cta_counts: list[dict[str, int]] = []
                comm_owned_elements = 0
                noncomm_owned_elements = 0
                if down_mismatch.numel():
                    mismatch_routes = down_mismatch[:, 0]
                    mismatch_ntiles = down_mismatch[:, 1] // 128
                    mismatch_mblocks = route_mblock[mismatch_routes]
                    mismatch_chunks = mismatch_ntiles // 8
                    chunk_tasks = padded_rows
                    mismatch_logical = (
                        mismatch_chunks * chunk_tasks
                        + mismatch_mblocks * 8
                        + mismatch_ntiles % 8
                    )
                    mismatch_ctas = mismatch_logical % (78 * 8)
                    comm_owned = (
                        (mismatch_ctas >= (78 * 8 - 64))
                        & (((mismatch_ctas - (78 * 8 - 64)) & 3)
                           == mismatch_chunks)
                    )
                    comm_owned_elements = int(comm_owned.sum().item())
                    noncomm_owned_elements = int(
                        comm_owned.numel() - comm_owned.sum().item()
                    )
                    task_keys = mismatch_mblocks * 32 + mismatch_ntiles
                    unique_tasks, task_counts = torch.unique(
                        task_keys, return_counts=True
                    )
                    top_count, top_index = torch.topk(
                        task_counts,
                        k=min(16, task_counts.numel()),
                    )
                    for key, count in zip(
                        unique_tasks[top_index].cpu().tolist(),
                        top_count.cpu().tolist(),
                    ):
                        mblock = int(key) // 32
                        n_tile = int(key) % 32
                        chunk = n_tile // 8
                        logical = (
                            chunk * chunk_tasks
                            + mblock * 8 + n_tile % 8
                        )
                        owner_cta = logical % (78 * 8)
                        task_owners.append(
                            {
                                "mblock": mblock,
                                "n_tile": n_tile,
                                "chunk": chunk,
                                "owner_cta": owner_cta,
                                "owner_is_chunk_comm": bool(
                                    owner_cta >= (78 * 8 - 64)
                                    and ((owner_cta - (78 * 8 - 64)) & 3)
                                        == chunk
                                ),
                                "mismatches": int(count),
                            }
                        )
                    unique_ctas, cta_counts = torch.unique(
                        mismatch_ctas, return_counts=True
                    )
                    top_count, top_index = torch.topk(
                        cta_counts, k=min(16, cta_counts.numel())
                    )
                    top_cta_counts = [
                        {"cta": int(cta), "mismatches": int(count)}
                        for cta, count in zip(
                            unique_ctas[top_index].cpu().tolist(),
                            top_count.cpu().tolist(),
                        )
                    ]
                print(
                    "SINGLE_MULTI_DOWN_DIAG "
                    + json.dumps(
                        {
                            "m": m,
                            "candidate_vs_control": down_repeat_metrics,
                            "repeat_vs_first_max_abs_rank_max": (
                                down_repeat_max_abs
                            ),
                            "chunk_ready_counts_rank0": [
                                [word & 0x3FF for word in words]
                                for words in candidate_chunk_state
                            ],
                            "chunk_ready_generations_rank0": [
                                [word >> 10 for word in words]
                                for words in candidate_chunk_state
                            ],
                            "rank0_owner_summary": {
                                "total_mismatches": int(
                                    down_mismatch.shape[0]
                                ),
                                "comm_owned_elements": comm_owned_elements,
                                "noncomm_owned_elements": (
                                    noncomm_owned_elements
                                ),
                                "top_tasks": task_owners,
                                "top_ctas": top_cta_counts,
                            },
                        },
                        sort_keys=True,
                    ),
                    flush=True,
                )

        control_check = custom.correctness_metrics(
            control_case, control_graph, nccl_group, device
        )
        candidate_check = custom.correctness_metrics(
            candidate_case, candidate_graph, nccl_group, device
        )
        native_local_raw = None
        native_local_raw_check = None
        native_local_scaled_check = None
        native_embedded_comm_check = None
        native_l2_check = None
        native_l2_unweighted_check = None
        native_l2_scale_check = None
        native_l2_byte_mismatches = None
        native_l2_cross_route_rank0 = None
        native_combine_route_check = None
        native_combine_sum_check = None
        candidate_accept = bool(candidate_check["allreduce_ok"])
        if use_native:
            assert candidate_case.native_local_output is not None
            native_local_raw = candidate_case.native_local_output.clone()
            native_local_scaled = (
                native_local_raw.float() * custom.ROUTED_SCALING_FACTOR
            ).to(torch.bfloat16)
            local_reference = (
                candidate_case.make_reference_case().run_local().clone()
            )
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
            candidate_accept = bool(
                candidate_check["finite_all_ranks"]
                and candidate_check["cosine_min_rank"] >= 0.999
                and candidate_check["rel_l2_max_rank"] <= 0.05
                and native_local_scaled_check["finite_all_ranks"]
                and native_local_scaled_check["cosine_min_rank"] >= 0.999
                and native_local_scaled_check["rel_l2_max_rank"] <= 0.05
                and native_embedded_comm_check["finite_all_ranks"]
                and native_embedded_comm_check["cosine_min_rank"] >= 0.999
                and native_embedded_comm_check["rel_l2_max_rank"] <= 0.01
            )
        if (
            use_native
            and args.route_pattern == "balanced"
            and m * custom.TOP_K <= custom.NUM_EXPERTS
        ):
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
                        "candidate_accept": candidate_accept,
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
        if not control_check["allreduce_ok"] or not candidate_accept:
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
        if not use_native:
            phase_stamps = candidate_case.single_launch_barrier_state[
                8:18
            ].view(torch.int64)
            phase_durations_ns = phase_stamps[1:] - phase_stamps[:-1]
            dist.all_reduce(
                phase_durations_ns, op=dist.ReduceOp.MAX, group=nccl_group
            )
            phase_us = {
                name: float(value) / 1000.0
                for name, value in zip(
                    ("route", "w13", "activation_requant", "w2"),
                    phase_durations_ns.cpu().tolist(),
                    strict=True,
                )
            }
        record: dict[str, Any] = {
            "m": m,
            "active_experts": candidate_case.active_experts,
            "routed_rows": m * custom.TOP_K,
            "control_padded_rows": int(control_case.num_tokens_padded.item()),
            "candidate_padded_rows": (
                None
                if use_native
                else int(candidate_case.num_tokens_padded.item())
            ),
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
