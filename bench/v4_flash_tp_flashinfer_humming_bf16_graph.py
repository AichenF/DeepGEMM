#!/usr/bin/env python3
"""Cold-L2 TP4 serving-layer comparison against FlashInfer CUTLASS Humming.

Both implementations receive BF16 activations and precomputed top-k routes.
The timed CUDA graphs include activation quantization, FC1, activation, FC2,
route reduction, and TP all-reduce.  Model-load weight conversion, routing,
autotuning, graph capture, warmup, and L2 eviction are excluded.
"""

from __future__ import annotations

import argparse
import gc
import json
import statistics
from dataclasses import dataclass
from typing import Any

import torch
import torch.distributed as dist
from triton import runtime as triton_runtime

import flashinfer
from flashinfer.autotuner import AutoTuner, autotune
from flashinfer.fused_moe import (
    BackendOptions,
    CutlassHummingConfig,
    CutlassHummingRunner,
    ExecutionConfig,
    ExpertConfig,
    MoEActivationPack,
    MoEConfig,
    MoEWeightPack,
    QuantConfig,
    QuantVariant,
    RoutingConfig,
    SwiGLU,
    preprocess_moe_weights_for_sm90_mixed_gemm_humming,
)
from humming import ops as humming_ops
from humming.ops.input import _quant_tensor_kernel
import sglang.srt.distributed.parallel_state as ps
try:
    from sglang.jit_kernel.mp import register_comm_cleanup
except ImportError:
    from sglang.kernels.ops.communication.mp import register_comm_cleanup
from sglang.srt.distributed.device_communicators.custom_all_reduce_v2 import (
    CustomAllReduceV2,
)

import v4_flash_tp_paired_graph as paired
import v4_flash_tp_wgmma as kernel
import v4_flash_tp_wgmma_graph as custom


@dataclass
class PreparedWeights:
    custom_w13: torch.Tensor
    custom_s13: torch.Tensor
    custom_g13: torch.Tensor
    custom_w2: torch.Tensor
    custom_s2: torch.Tensor
    custom_g2: torch.Tensor
    flashinfer_pack: MoEWeightPack


@dataclass
class CustomBF16Case:
    x_bf16: torch.Tensor
    inner: custom.CapturedCase

    def run_full(self, comm: CustomAllReduceV2) -> torch.Tensor:
        quant_group128_into(self.x_bf16, self.inner.qx, self.inner.x_scale)
        return self.inner.run_full(comm)


@dataclass
class FlashInferBF16Case:
    runner: CutlassHummingRunner
    inputs: list[torch.Tensor]
    tactic: Any
    graph_output: torch.Tensor | None = None

    def run_full(self, comm: CustomAllReduceV2) -> torch.Tensor:
        local = self.runner.forward(self.inputs, tactic=self.tactic)
        self.graph_output = comm.custom_all_reduce(local)
        return self.graph_output


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ms", default="8,16,32,64,128")
    parser.add_argument(
        "--route-pattern", choices=("random", "balanced", "skew"), default="random"
    )
    parser.add_argument("--outer", type=int, default=10)
    parser.add_argument("--replays", type=int, default=200)
    parser.add_argument("--warmup-replays", type=int, default=10)
    parser.add_argument(
        "--pair-granularity", choices=("batch", "replay"), default="batch"
    )
    parser.add_argument("--seed", type=int, default=20260908)
    parser.add_argument(
        "--no-autotune",
        action="store_true",
        help="Use CUTLASS fallback tactics; intended only for bring-up.",
    )
    args = parser.parse_args()
    args.ms = tuple(int(value) for value in args.ms.split(",") if value)
    if not args.ms or any(value <= 0 for value in args.ms):
        parser.error("--ms must contain positive integers")
    if args.outer < 1 or args.replays < 1 or args.warmup_replays < 1:
        parser.error("timing loop counts must be positive")
    if args.pair_granularity == "batch" and args.outer % 2:
        parser.error("batch pairing requires an even --outer")
    return args


def marlin_to_linear_mxfp4(weight: torch.Tensor) -> torch.Tensor:
    """Convert canonical Marlin-K8 packed nibbles to adjacent-pair packing."""
    if weight.dtype is not torch.uint8 or weight.ndim != 3:
        raise TypeError("weight must be rank-three uint8")
    *leading, half_k = weight.shape
    if half_k % 4:
        raise ValueError("packed K must contain complete Marlin K8 chunks")
    chunks = weight.view(*leading, half_k // 4, 4)
    logical = torch.cat((chunks >> 4, chunks & 0x0F), dim=-1).reshape(
        *leading, half_k * 2
    )
    return (logical[..., 0::2] | (logical[..., 1::2] << 4)).contiguous()


def prepare_custom_layer(
    raw_marlin: torch.Tensor,
    raw_scale: torch.Tensor,
    *,
    is_w13: bool,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    weight = kernel.marlin_to_legacy_mxfp4(raw_marlin)
    scale = raw_scale
    global_scale = torch.empty(0, dtype=torch.float32, device=weight.device)
    if kernel.NORMALIZED_WEIGHT_SCALE:
        scale, global_scale = kernel.normalize_mxfp4_weight_scales_(weight, scale)
    if is_w13 and kernel.W13_PAIRED_WG:
        weight, scale = kernel.pair_gate_up_weight_layout(weight, scale)
    if kernel.MODE2_BRAID:
        kernel.braid_mode2_(weight)
    if kernel.TILED_WEIGHT_LAYOUT:
        weight, scale = kernel.tile_mxfp4_weight_layout(weight, scale)
    return weight, scale, global_scale


def prepare_common_weights(
    intermediate_per_rank: int,
    device: torch.device,
    seed: int,
) -> PreparedWeights:
    """Build both physical layouts from one canonical MXFP4 checkpoint image."""
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)

    def one_layer(
        rows: int, cols: int, *, is_w13: bool
    ) -> tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
    ]:
        raw_marlin = torch.randint(
            0,
            256,
            (custom.NUM_EXPERTS, rows, cols // 2),
            dtype=torch.uint8,
            device=device,
        )
        raw_scale = torch.randint(
            125,
            129,
            (custom.NUM_EXPERTS, rows, cols // 32),
            dtype=torch.uint8,
            device=device,
        )
        linear = marlin_to_linear_mxfp4(raw_marlin)
        fi_weight, fi_scale, fi_residual = (
            preprocess_moe_weights_for_sm90_mixed_gemm_humming(
                linear, raw_scale
            )
        )
        del linear
        c_weight, c_scale, c_global = prepare_custom_layer(
            raw_marlin, raw_scale, is_w13=is_w13
        )
        del raw_marlin, raw_scale
        return c_weight, c_scale, c_global, fi_weight, fi_scale, fi_residual

    n13 = 2 * intermediate_per_rank
    c_w13, c_s13, c_g13, fi_w13, fi_s13, fi_r13 = one_layer(
        n13, custom.HIDDEN, is_w13=True
    )
    c_w2, c_s2, c_g2, fi_w2, fi_s2, fi_r2 = one_layer(
        custom.HIDDEN, intermediate_per_rank, is_w13=False
    )
    view = {
        "fc1_expert_weights": fi_w13,
        "fc2_expert_weights": fi_w2,
        "fc1_expert_scales": fi_s13,
        "fc2_expert_scales": fi_s2,
        "fc1_residual_scale": (fi_r13 * 64.0).contiguous(),
        "fc2_residual_scale": (fi_r2 * 64.0).contiguous(),
        "fc2_act_global": torch.ones((), dtype=torch.float32, device=device),
    }
    fi_pack = MoEWeightPack()
    fi_pack.prepare_for("cutlass_humming", view)
    return PreparedWeights(c_w13, c_s13, c_g13, c_w2, c_s2, c_g2, fi_pack)


def make_bf16_input(m: int, device: torch.device, seed: int) -> torch.Tensor:
    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed + m)
    return (
        torch.randn((m, custom.HIDDEN), generator=generator, dtype=torch.float32)
        .mul_(0.1)
        .to(torch.bfloat16)
        .to(device)
    )


def quant_group128_into(
    x_bf16: torch.Tensor, qx: torch.Tensor, scale: torch.Tensor
) -> None:
    """The exact Humming group-128 quant kernel used by the custom serving path."""
    inputs = x_bf16.view(-1, x_bf16.shape[-1])
    group_size = 128
    num_blocks = inputs.numel() // group_size
    _quant_tensor_kernel[(num_blocks,)](
        inputs,
        qx,
        scale,
        inputs.stride(0),
        num_blocks,
        True,
        inputs.shape[1],
        group_size,
        group_size,
        1,
        "float8e4m3",
        inputs.shape[0],
        False,
        "float32",
        False,
        None,
        False,
        num_warps=1,
        num_stages=1,
    )


def make_flashinfer_runner(
    intermediate_per_rank: int, device: torch.device
) -> CutlassHummingRunner:
    config = MoEConfig(
        routing=RoutingConfig(num_experts=custom.NUM_EXPERTS, top_k=custom.TOP_K),
        quant=QuantConfig(variant=QuantVariant.Humming),
        experts=ExpertConfig(intermediate_size=intermediate_per_rank),
        activation=SwiGLU(),
        backend=BackendOptions((CutlassHummingConfig(),)),
        execution=ExecutionConfig(enable_pdl=False, tune_max_num_tokens=128),
    )
    runner = CutlassHummingRunner(config, device)
    runner.check_support()
    runner.build()
    return runner


def choose_tactic(
    runner: CutlassHummingRunner,
    inputs: list[torch.Tensor],
    m: int,
    rank: int,
    no_autotune: bool,
) -> Any:
    if no_autotune:
        return -1
    with autotune(True):
        _, tactic = AutoTuner.get().choose_one(
            f"bf16_serving_cutlass_humming_tp4_m{m}_rank{rank}",
            [runner],
            runner.tuning_config,
            inputs,
        )
    if not isinstance(tactic, tuple) or len(tactic) != 2:
        raise RuntimeError(f"unexpected Humming tactic {tactic!r}")
    return tactic


def metrics(
    actual: torch.Tensor,
    reference: torch.Tensor,
    nccl_group: dist.ProcessGroup,
    device: torch.device,
) -> dict[str, float | bool]:
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
        bool(torch.isfinite(actual).all()) and bool(torch.isfinite(reference).all())
    )
    return {
        "cosine_min_rank": custom.reduce_rank_metric(
            cosine, dist.ReduceOp.MIN, device, nccl_group
        ),
        "rel_l2_max_rank": custom.reduce_rank_metric(
            rel_l2, dist.ReduceOp.MAX, device, nccl_group
        ),
        "max_abs_max_rank": custom.reduce_rank_metric(
            float(diff.abs().max()), dist.ReduceOp.MAX, device, nccl_group
        ),
        "finite_all_ranks": bool(
            custom.reduce_rank_metric(finite, dist.ReduceOp.MIN, device, nccl_group)
        ),
    }


def flashinfer_allreduce_correctness(
    case: FlashInferBF16Case,
    graph: torch.cuda.CUDAGraph,
    nccl_group: dist.ProcessGroup,
    device: torch.device,
) -> dict[str, float | bool]:
    graph.replay()
    torch.cuda.synchronize(device)
    assert case.graph_output is not None
    actual = case.graph_output.clone()
    reference = case.runner.forward(case.inputs, tactic=case.tactic).clone()
    dist.all_reduce(reference, group=nccl_group)
    torch.cuda.synchronize(device)
    result = metrics(actual, reference, nccl_group, device)
    result["allreduce_ok"] = bool(
        result["finite_all_ranks"]
        and result["cosine_min_rank"] >= 0.999
        and result["rel_l2_max_rank"] <= 0.02
    )
    return result


def quant_correctness(
    x: torch.Tensor, qx: torch.Tensor, scale: torch.Tensor
) -> dict[str, float | bool]:
    expected_q = torch.empty_like(qx)
    expected_q, expected_scale = humming_ops.quant_input(
        inputs=x,
        outputs=expected_q,
        dtype="float8e4m3",
        group_size=128,
        m_major_scale=False,
        scale_dtype="float32",
    )
    quant_group128_into(x, qx, scale)
    torch.cuda.synchronize(x.device)
    result = {
        "fp8_bytes_exact": bool(torch.equal(qx.view(torch.uint8), expected_q.view(torch.uint8))),
        "scale_max_abs": float((scale - expected_scale).abs().max()),
    }
    del expected_q, expected_scale
    return result


@torch.inference_mode()
def main() -> None:
    args = parse_args()
    rank, world_size, device, cpu_group = custom.init_distributed()
    if world_size != 4:
        raise ValueError(f"this comparison is the TP4 target, got TP{world_size}")
    nccl_group = ps._WORLD.device_group
    if not isinstance(nccl_group, dist.ProcessGroup):
        raise RuntimeError("missing NCCL process group")
    props = torch.cuda.get_device_properties(device)
    intermediate_per_rank = custom.INTERMEDIATE // world_size

    prepared = prepare_common_weights(
        intermediate_per_rank, device, args.seed + 1000 * rank
    )
    lut = kernel.make_e2m1_e8m0_lut(device)
    fi_runner = make_flashinfer_runner(intermediate_per_rank, device)
    comm = CustomAllReduceV2(cpu_group, device)
    if comm.disabled:
        raise RuntimeError("SGLang CustomAllReduceV2 is disabled")
    register_comm_cleanup(comm)
    l2_flush_buffer = triton_runtime.driver.active.get_empty_cache_for_benchmark()
    if l2_flush_buffer.nbytes < 2 * props.L2_cache_size:
        raise RuntimeError("L2 eviction buffer is smaller than twice physical L2")

    if rank == 0:
        print(
            "BF16_SERVING_ENV "
            + json.dumps(
                {
                    "benchmark": "flashinfer_cutlass_humming_vs_custom_tp4_bf16_serving",
                    "torch": torch.__version__,
                    "flashinfer": flashinfer.__version__,
                    "flashinfer_file": flashinfer.__file__,
                    "gpu": props.name,
                    "sm_count": props.multi_processor_count,
                    "world_size": world_size,
                    "m_values": args.ms,
                    "shape": {
                        "experts": custom.NUM_EXPERTS,
                        "hidden": custom.HIDDEN,
                        "global_intermediate": custom.INTERMEDIATE,
                        "local_intermediate": intermediate_per_rank,
                        "top_k": custom.TOP_K,
                    },
                    "route_pattern": args.route_pattern,
                    "outer": args.outer,
                    "replays": args.replays,
                    "warmup_replays": args.warmup_replays,
                    "pair_granularity": args.pair_granularity,
                    "autotune": not args.no_autotune,
                    "l2_policy": "256MiB clear before every replay; clear excluded from CUDA events",
                    "l2_cache_bytes": props.L2_cache_size,
                    "l2_flush_bytes": l2_flush_buffer.nbytes,
                    "timed_contract": "BF16 input -> online FP8 quant -> W13/SwiGLU/W2/route reduce -> TP all-reduce",
                    "routing_timed": False,
                    "weight_preprocess_timed": False,
                    "custom_input_quant": "Humming Triton FP8-E4M3 group128, one quant per original token",
                    "flashinfer_input_quant": "CutlassHummingConfig native rowwise FP8-E4M3 after route expansion",
                    "common_weights": "same canonical Marlin-K8 MXFP4 payload and E8M0 scales; independently transformed at model load",
                    "route_scale": "custom applies 1.5 in fused reduction; FlashInfer receives topk weights pre-multiplied by 1.5",
                    "flashinfer_allreduce": "stock SGLang custom_all_reduce_v2",
                    "custom_allreduce": "selected fused k6 multicast push for M<=32; stock custom_all_reduce_v2 for M>=64",
                    "custom_single_launch_tp4": kernel.SINGLE_LAUNCH_TP4,
                    "custom_fused_k6_mc_push_ar": kernel.FUSED_K6_MC_PUSH_AR,
                    "custom_fused_k6_mc_push_max_m": kernel.FUSED_K6_MC_PUSH_MAX_M,
                },
                sort_keys=True,
            ),
            flush=True,
        )

    records: list[dict[str, Any]] = []
    for m in args.ms:
        topk_ids, topk_weights = custom.make_routes(
            m, args.route_pattern, device, args.seed
        )
        x_bf16 = make_bf16_input(m, device, args.seed)
        qx = torch.empty((m, custom.HIDDEN), dtype=torch.float8_e4m3fn, device=device)
        x_scale = torch.empty(
            (m, custom.HIDDEN // 128), dtype=torch.float32, device=device
        )
        quant_check = quant_correctness(x_bf16, qx, x_scale)

        inner = custom.CapturedCase(
            m=m,
            qx=qx,
            x_scale=x_scale,
            topk_ids=topk_ids,
            topk_weights=topk_weights,
            w13=prepared.custom_w13,
            s13=prepared.custom_s13,
            g13=prepared.custom_g13,
            w2=prepared.custom_w2,
            s2=prepared.custom_s2,
            g2=prepared.custom_g2,
            lut=lut,
            intermediate_per_rank=intermediate_per_rank,
        )
        custom_case = CustomBF16Case(x_bf16, inner)

        fi_ids = topk_ids.clone()
        fi_weights = (topk_weights * custom.ROUTED_SCALING_FACTOR).contiguous()
        fi_act = MoEActivationPack(x_bf16, None, fi_ids, fi_weights)
        fi_inputs = fi_runner.pack_inputs(fi_act, prepared.flashinfer_pack)
        tactic = choose_tactic(fi_runner, fi_inputs, m, rank, args.no_autotune)
        # CUTLASS's tuning input hook fills synthetic balanced routes. Restore
        # the serving inputs before graph capture.
        fi_inputs[2].copy_(topk_ids)
        fi_inputs[3].copy_(topk_weights * custom.ROUTED_SCALING_FACTOR)
        fi_case = FlashInferBF16Case(fi_runner, fi_inputs, tactic)

        fi_graph = paired.capture_graph(fi_case, comm, cpu_group, device)
        custom_graph = paired.capture_graph(custom_case, comm, cpu_group, device)
        fi_check = flashinfer_allreduce_correctness(
            fi_case, fi_graph, nccl_group, device
        )
        custom_check = custom.correctness_metrics(
            inner, custom_graph, nccl_group, device
        )
        if not fi_check["allreduce_ok"] or not custom_check["allreduce_ok"]:
            raise RuntimeError(
                f"communication correctness failure at M={m}: "
                f"flashinfer={fi_check}, custom={custom_check}"
            )

        fi_graph.replay()
        custom_graph.replay()
        torch.cuda.synchronize(device)
        assert fi_case.graph_output is not None and inner.graph_output is not None
        cross = metrics(
            inner.graph_output.clone(),
            fi_case.graph_output.clone(),
            nccl_group,
            device,
        )

        for warmup_idx in range(args.warmup_replays):
            if warmup_idx & 1:
                custom_graph.replay()
                fi_graph.replay()
            else:
                fi_graph.replay()
                custom_graph.replay()
        torch.cuda.synchronize(device)

        fi_samples, custom_samples, fi_batches, custom_batches = paired.time_graph_pair(
            fi_graph,
            custom_graph,
            args.outer,
            args.replays,
            cpu_group,
            nccl_group,
            device,
            l2_flush_buffer,
            args.pair_granularity,
        )
        fi_median = statistics.median(fi_samples)
        custom_median = statistics.median(custom_samples)
        tactics = [None for _ in range(world_size)]
        dist.all_gather_object(tactics, tactic, group=cpu_group)
        record = {
            "m": m,
            "routed_rows": m * custom.TOP_K,
            "active_experts": inner.active_experts,
            "padded_rows": int(inner.num_tokens_padded.item()),
            "w13_split_k": inner.w13_split_k,
            "cold_samples_per_impl": len(fi_samples),
            "flashinfer_latency_ms_min": min(fi_samples),
            "flashinfer_latency_ms_median": fi_median,
            "flashinfer_latency_ms_max": max(fi_samples),
            "custom_latency_ms_min": min(custom_samples),
            "custom_latency_ms_median": custom_median,
            "custom_latency_ms_max": max(custom_samples),
            "speedup_flashinfer_over_custom": fi_median / custom_median,
            "custom_over_flashinfer": custom_median / fi_median,
            "flashinfer_batch_medians_ms_max_rank": fi_batches,
            "custom_batch_medians_ms_max_rank": custom_batches,
            "flashinfer_tactics_by_rank": tactics,
            "quant_correctness": quant_check,
            "flashinfer_allreduce_correctness": fi_check,
            "custom_allreduce_correctness": custom_check,
            "cross_backend_same_checkpoint": cross,
            "custom_allreduce_mode": inner.fused_k6_ar_mode,
        }
        records.append(record)
        if rank == 0:
            print("BF16_SERVING_RESULT " + json.dumps(record, sort_keys=True), flush=True)

        del fi_graph, custom_graph, fi_case, custom_case, inner, fi_inputs
        gc.collect()
        torch.cuda.empty_cache()
        dist.barrier(group=cpu_group)

    if rank == 0:
        fi_geomean = statistics.geometric_mean(
            record["flashinfer_latency_ms_median"] for record in records
        )
        custom_geomean = statistics.geometric_mean(
            record["custom_latency_ms_median"] for record in records
        )
        print(
            "BF16_SERVING_SUMMARY "
            + json.dumps(
                {
                    "m_values": list(args.ms),
                    "flashinfer_geometric_mean_ms": fi_geomean,
                    "custom_geometric_mean_ms": custom_geomean,
                    "speedup_flashinfer_over_custom": fi_geomean / custom_geomean,
                    "custom_over_flashinfer": custom_geomean / fi_geomean,
                    "samples_per_m_per_impl": args.outer * args.replays,
                },
                sort_keys=True,
            ),
            flush=True,
        )


if __name__ == "__main__":
    main()
