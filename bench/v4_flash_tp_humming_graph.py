#!/usr/bin/env python3
"""CUDA-Graph baseline for DeepSeek-V4-Flash TP MXFP4 MoE.

The timed graph follows SGLang's standard (non-EP) Humming path:

  route align -> BF16/FP8 quant -> Humming MXFP4 W13 -> SwiGLU
  -> BF16/FP8 quant -> Humming MXFP4 W2 -> local top-k weighted sum
  -> SGLang custom_all_reduce_v2

Router/top-k selection, weight construction/transform, allocations, JIT, and graph
capture are deliberately outside the measured region.  ``topk_ids`` and
``topk_weights`` are static graph inputs, matching serving CUDA-graph replay.
"""

from __future__ import annotations

import argparse
import atexit
import json
import logging
import os
import statistics
from dataclasses import dataclass
from typing import Any

import torch
import torch.distributed as dist
from triton import runtime as triton_runtime

import sglang.srt.distributed.parallel_state as ps
from humming.config import GemmType
from humming.layer import HummingLayer, HummingMethod
try:
    from sglang.jit_kernel.mp import register_comm_cleanup
except ImportError:  # Compatibility with the older benchmark checkout.
    from sglang.kernels.ops.communication.mp import register_comm_cleanup
from sglang.kernels.ops.moe.moe_fused_mul_sum import moe_fused_mul_sum
from sglang.srt.distributed.device_communicators.custom_all_reduce_v2 import (
    CustomAllReduceV2,
)
from sglang.srt.layers.moe.fused_moe_triton import moe_align_block_size
from sgl_kernel import silu_and_mul


HIDDEN = 4096
INTERMEDIATE = 2048
NUM_EXPERTS = 256
TOP_K = 6
ROUTED_SCALING_FACTOR = 1.5
DEFAULT_MS = (8, 16, 32, 64, 128)

WEIGHT_CONFIG = {
    "dtype": "float4e2m1",
    "group_size": 32,
    "scale_dtype": "float8e8m0",
    "has_zero_point": False,
    "is_fp_zero_point": False,
}
INPUT_CONFIG = {"dtype": "float8e4m3", "group_size": 128}
COMPUTE_CONFIG = {"use_f16_accum": False, "gemm_type": "indexed"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--ms",
        default=",".join(map(str, DEFAULT_MS)),
        help="Comma-separated token counts (default: 8,16,32,64,128).",
    )
    parser.add_argument(
        "--route-pattern",
        choices=("random", "balanced", "skew"),
        default="random",
        help=(
            "Precomputed route distribution. random follows DeepGEMM MegaMoE's "
            "random-scores/top-k construction; router computation is not timed."
        ),
    )
    parser.add_argument("--outer", type=int, default=7)
    parser.add_argument("--replays", type=int, default=100)
    parser.add_argument("--warmup-replays", type=int, default=10)
    parser.add_argument(
        "--profile-once",
        action="store_true",
        help="Expose one explicitly cold graph replay to CUDA profiler APIs.",
    )
    parser.add_argument("--seed", type=int, default=20260902)
    args = parser.parse_args()
    args.ms = tuple(int(value) for value in args.ms.split(",") if value)
    if not args.ms or any(value <= 0 for value in args.ms):
        parser.error("--ms must contain positive integers")
    if args.outer < 1 or args.replays < 1 or args.warmup_replays < 1:
        parser.error("--outer, --replays, and --warmup-replays must be positive")
    if args.profile_once and len(args.ms) != 1:
        parser.error("--profile-once requires exactly one M value")
    return args


def init_distributed() -> tuple[int, int, torch.device, dist.ProcessGroup]:
    local_rank = int(os.environ["LOCAL_RANK"])
    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    if world_size not in (4, 8):
        raise ValueError(f"This benchmark supports TP4/TP8, got world_size={world_size}")

    torch.cuda.set_device(local_rank)
    dist.init_process_group(backend="gloo")
    # Registered before communicator cleanup, so LIFO atexit ordering closes
    # the custom communicator while process groups are still live.
    atexit.register(dist.destroy_process_group)
    ps._WORLD = coordinator = ps.init_world_group(
        ranks=list(range(world_size)),
        local_rank=local_rank,
        backend="nccl",
    )
    cpu_group = coordinator.cpu_group
    if not isinstance(cpu_group, dist.ProcessGroup):
        raise RuntimeError("SGLang world coordinator did not create a CPU group")

    device = torch.device(f"cuda:{local_rank}")
    stream = torch.cuda.Stream(device=device)
    torch.cuda.set_stream(stream)
    logging.disable(logging.INFO)
    return rank, world_size, device, cpu_group


def make_layer(shape_n: int, shape_k: int, device: torch.device) -> HummingLayer:
    layer = HummingLayer(
        shape_n=shape_n,
        shape_k=shape_k,
        num_experts=NUM_EXPERTS,
        weight_config=WEIGHT_CONFIG.copy(),
        input_config=INPUT_CONFIG.copy(),
        torch_dtype=torch.bfloat16,
    ).to(device)
    # Keep E8M0 scales in a numerically ordinary range.  Humming's generic
    # random helper intentionally fills arbitrary bit patterns, which is fine
    # for throughput but can make a BF16 reduction's FP32 diagnostic norm
    # overflow.  Packed FP4 codes remain fully random; scale=1 only changes
    # data values, never the measured instruction/memory path.
    for name, parameter in layer.named_parameters():
        if "scale" in name:
            parameter.fill_(1.0)
        elif parameter.dtype == torch.uint8:
            parameter.random_(0, 256)
        elif parameter.dtype == torch.int32:
            parameter.random_(-(2**31), 2**31 - 1)
        elif not parameter.dtype.is_floating_point:
            parameter.zero_()
        else:
            parameter.normal_(std=0.01)
    layer.transform()
    return layer


def make_routes(
    m: int, route_pattern: str, device: torch.device, seed: int
) -> tuple[torch.Tensor, torch.Tensor]:
    if route_pattern == "random":
        # DeepGEMM MegaMoE uses random scores followed by top-k.  Generate on
        # CPU with an isolated seed so all TP ranks replay exactly one route.
        generator = torch.Generator(device="cpu")
        generator.manual_seed(seed)
        scores = torch.randn(
            (m, NUM_EXPERTS), dtype=torch.float32, generator=generator
        )
        ids = torch.topk(
            scores, TOP_K, dim=-1, largest=True, sorted=False
        ).indices.to(torch.int32)
    elif route_pattern == "balanced":
        # Consecutive routes are unique within every token and cover experts as
        # evenly as possible.  This is deterministic and makes active-expert
        # count explicit rather than inferring it from an unrelated G knob.
        ids = torch.arange(m * TOP_K, dtype=torch.int32).reshape(m, TOP_K)
        ids.remainder_(NUM_EXPERTS)
    else:
        # Maximal legal skew: every token selects the same six distinct experts.
        ids = torch.arange(TOP_K, dtype=torch.int32).repeat(m, 1)

    # Fixed, normalized positive weights.  Routing itself remains precomputed.
    row = torch.arange(1, TOP_K + 1, dtype=torch.float32)
    row /= row.sum()
    weights = row.repeat(m, 1)
    return ids.to(device), weights.to(device)


def select_tuning_config(configs: list[Any], valid_shape_m: int) -> dict[str, Any]:
    for min_shape_m, max_shape_m, config in configs:
        if valid_shape_m > min_shape_m and valid_shape_m <= max_shape_m:
            return config
    raise ValueError(f"No Humming indexed config covers M={valid_shape_m}")


@dataclass
class CapturedCase:
    m: int
    x: torch.Tensor
    topk_ids: torch.Tensor
    topk_weights: torch.Tensor
    w13: HummingLayer
    w2: HummingLayer
    w13_tuning: list[Any]
    w2_tuning: list[Any]
    w13_block_m: int
    intermediate_per_rank: int

    def __post_init__(self) -> None:
        device = self.x.device
        routed_m = self.m * TOP_K
        self.qx = torch.empty(
            (self.m, HIDDEN), dtype=torch.float8_e4m3fn, device=device
        )
        self.gate_up = torch.empty(
            (routed_m, 2 * self.intermediate_per_rank),
            dtype=torch.bfloat16,
            device=device,
        )
        self.activation = torch.empty(
            (routed_m, self.intermediate_per_rank),
            dtype=torch.bfloat16,
            device=device,
        )
        self.qdown = torch.empty(
            (routed_m, self.intermediate_per_rank),
            dtype=torch.float8_e4m3fn,
            device=device,
        )
        self.down = torch.empty(
            (routed_m, HIDDEN), dtype=torch.bfloat16, device=device
        )
        self.local_output = torch.empty(
            (self.m, HIDDEN), dtype=torch.bfloat16, device=device
        )
        self.graph_output: torch.Tensor | None = None
        self.sorted_ids: torch.Tensor | None = None
        self.expert_ids: torch.Tensor | None = None
        self.num_tokens_padded: torch.Tensor | None = None
        self.x_scale: torch.Tensor | None = None
        self.down_scale: torch.Tensor | None = None

    @property
    def valid_shape_m(self) -> int:
        return self.m * TOP_K

    def run_local(self) -> torch.Tensor:
        # Keep graph-captured allocation results alive on the case object.  The
        # allocation calls themselves are not replayed; only their GPU kernels
        # are in the graph timing region.
        (
            self.sorted_ids,
            self.expert_ids,
            self.num_tokens_padded,
        ) = moe_align_block_size(
            topk_ids=self.topk_ids,
            block_size=self.w13_block_m,
            num_experts=NUM_EXPERTS,
            ignore_invalid_expert=True,
        )

        qx, self.x_scale = HummingMethod.may_quant_input(
            layer=self.w13,
            inputs=self.x,
            quanted_input=self.qx,
        )
        HummingMethod.forward_layer(
            layer=self.w13,
            inputs=qx,
            input_scale=self.x_scale,
            outputs=self.gate_up,
            sorted_ids=self.sorted_ids,
            expert_ids=self.expert_ids,
            num_tokens_padded=self.num_tokens_padded,
            top_k=TOP_K,
            valid_shape_m=self.valid_shape_m,
            compute_config=COMPUTE_CONFIG,
            tuning_config=self.w13_tuning,
        )

        silu_and_mul(self.gate_up, self.activation)

        qdown, self.down_scale = HummingMethod.may_quant_input(
            layer=self.w2,
            inputs=self.activation,
            quanted_input=self.qdown,
        )
        HummingMethod.forward_layer(
            layer=self.w2,
            inputs=qdown,
            input_scale=self.down_scale,
            outputs=self.down,
            sorted_ids=self.sorted_ids,
            expert_ids=self.expert_ids,
            num_tokens_padded=self.num_tokens_padded,
            top_k=1,
            valid_shape_m=self.valid_shape_m,
            compute_config=COMPUTE_CONFIG,
            tuning_config=self.w2_tuning,
        )

        moe_fused_mul_sum(
            inputs=self.down.view(self.m, TOP_K, HIDDEN),
            topk_weights=self.topk_weights,
            topk_ids=self.topk_ids,
            is_ep=False,
            routed_scaling_factor=ROUTED_SCALING_FACTOR,
            outputs=self.local_output,
        )
        return self.local_output

    def run_full(self, comm: CustomAllReduceV2) -> torch.Tensor:
        self.graph_output = comm.custom_all_reduce(self.run_local())
        return self.graph_output


def max_across_ranks(value: float, device: torch.device, group: dist.ProcessGroup) -> float:
    tensor = torch.tensor(value, dtype=torch.float64, device=device)
    dist.all_reduce(tensor, op=dist.ReduceOp.MAX, group=group)
    return float(tensor.item())


def min_across_ranks(value: float, device: torch.device, group: dist.ProcessGroup) -> float:
    tensor = torch.tensor(value, dtype=torch.float64, device=device)
    dist.all_reduce(tensor, op=dist.ReduceOp.MIN, group=group)
    return float(tensor.item())


def correctness_metrics(
    case: CapturedCase,
    graph: torch.cuda.CUDAGraph,
    nccl_group: dist.ProcessGroup,
    device: torch.device,
) -> dict[str, float | bool]:
    graph.replay()
    torch.cuda.synchronize(device)
    assert case.graph_output is not None
    actual = case.graph_output.clone()

    # TWO_SHOT_PULL is allowed to reuse/overwrite the all-reduce input.  Build
    # the NCCL reference from a separate untimed local pipeline so that we do
    # not accidentally reduce an already-reduced buffer (which would produce
    # an exact world-size factor error).  Separate intermediates also keep the
    # pointers retained by the captured graph untouched.
    reference_case = CapturedCase(
        m=case.m,
        x=case.x,
        topk_ids=case.topk_ids,
        topk_weights=case.topk_weights,
        w13=case.w13,
        w2=case.w2,
        w13_tuning=case.w13_tuning,
        w2_tuning=case.w2_tuning,
        w13_block_m=case.w13_block_m,
        intermediate_per_rank=case.intermediate_per_rank,
    )
    reference = reference_case.run_local().clone()
    dist.all_reduce(reference, group=nccl_group)
    torch.cuda.synchronize(device)

    # FP64 diagnostics are outside the timed graph.  They remain stable even
    # when valid quantized values have a wide dynamic range.
    actual_f = actual.double()
    reference_f = reference.double()
    diff = actual_f - reference_f
    cosine = torch.nn.functional.cosine_similarity(
        actual_f.flatten(), reference_f.flatten(), dim=0
    )
    denom = torch.linalg.vector_norm(reference_f).clamp_min(1e-40)
    rel_l2 = torch.linalg.vector_norm(diff) / denom
    ref_max = reference_f.abs().max().clamp_min(1e-40)
    finite = bool(
        torch.isfinite(actual).all().item()
        and torch.isfinite(reference).all().item()
    )
    cosine_value = float(cosine.item())
    rel_l2_value = float(rel_l2.item())
    cosine_min_rank = min_across_ranks(cosine_value, device, nccl_group)
    rel_l2_max_rank = max_across_ranks(rel_l2_value, device, nccl_group)
    finite_all_ranks = bool(min_across_ranks(float(finite), device, nccl_group))
    metrics = {
        "cosine_min_rank": cosine_min_rank,
        "rel_l2_max_rank": rel_l2_max_rank,
        "max_abs_max_rank": max_across_ranks(
            float(diff.abs().max().item()), device, nccl_group
        ),
        "max_abs_over_ref_max_rank": max_across_ranks(
            float((diff.abs().max() / ref_max).item()), device, nccl_group
        ),
        "finite_all_ranks": finite_all_ranks,
        "allreduce_ok": bool(
            finite_all_ranks
            and cosine_min_rank >= 0.999
            and rel_l2_max_rank <= 0.02
        ),
    }
    return metrics


def time_graph(
    graph: torch.cuda.CUDAGraph,
    outer: int,
    replays: int,
    cpu_group: dist.ProcessGroup,
    nccl_group: dist.ProcessGroup,
    device: torch.device,
    l2_flush_buffer: torch.Tensor,
) -> tuple[list[float], list[float]]:
    samples: list[float] = []
    batch_medians: list[float] = []
    driver = triton_runtime.driver.active
    for _ in range(outer):
        dist.barrier(group=cpu_group)
        starts = [torch.cuda.Event(enable_timing=True) for _ in range(replays)]
        ends = [torch.cuda.Event(enable_timing=True) for _ in range(replays)]
        for replay_idx in range(replays):
            # Same 256-MiB cold-L2 protocol as triton.testing.do_bench.  The
            # clear is ordered before the start event on the same stream, so it
            # evicts cache contents but is excluded from the measured interval.
            driver.clear_cache(l2_flush_buffer)
            starts[replay_idx].record()
            graph.replay()
            ends[replay_idx].record()
        ends[-1].synchronize()
        local_times = torch.tensor(
            [start.elapsed_time(end) for start, end in zip(starts, ends)],
            dtype=torch.float64,
            device=device,
        )
        dist.all_reduce(local_times, op=dist.ReduceOp.MAX, group=nccl_group)
        batch = [float(value) for value in local_times.cpu().tolist()]
        samples.extend(batch)
        batch_medians.append(statistics.median(batch))
    return samples, batch_medians


@torch.inference_mode()
def main() -> None:
    args = parse_args()
    rank, world_size, device, cpu_group = init_distributed()
    nccl_group = ps._WORLD.device_group
    if not isinstance(nccl_group, dist.ProcessGroup):
        raise RuntimeError("SGLang world coordinator did not create an NCCL group")

    props = torch.cuda.get_device_properties(device)
    if props.major != 9:
        raise RuntimeError(f"Expected Hopper/sm90, got capability {props.major}.{props.minor}")

    intermediate_per_rank = INTERMEDIATE // world_size
    torch.manual_seed(args.seed + rank)
    torch.cuda.manual_seed(args.seed + rank)

    # Pure TP: every rank owns all expert IDs, but only its shard of I.
    w13 = make_layer(2 * intermediate_per_rank, HIDDEN, device)
    w2 = make_layer(HIDDEN, intermediate_per_rank, device)
    w13_tuning = HummingMethod.get_default_tuning_configs(
        layer=w13, use_f16_accum=False, gemm_type=GemmType.INDEXED
    )
    w2_tuning = HummingMethod.get_default_tuning_configs(
        layer=w2, use_f16_accum=False, gemm_type=GemmType.INDEXED
    )

    comm = CustomAllReduceV2(cpu_group, device)
    if comm.disabled:
        raise RuntimeError("SGLang CustomAllReduceV2 is disabled on this topology")
    register_comm_cleanup(comm)
    l2_flush_buffer = triton_runtime.driver.active.get_empty_cache_for_benchmark()
    if l2_flush_buffer.nbytes < 2 * props.L2_cache_size:
        raise RuntimeError(
            f"L2 flush buffer ({l2_flush_buffer.nbytes}) is smaller than 2x "
            f"L2 ({props.L2_cache_size})"
        )

    if rank == 0:
        print(
            "BASELINE_ENV "
            + json.dumps(
                {
                    "benchmark": "v4_flash_tp_humming_mxfp4_cuda_graph",
                    "gpu": props.name,
                    "sm_count": props.multi_processor_count,
                    "capability": f"{props.major}.{props.minor}",
                    "world_size": world_size,
                    "route_pattern": args.route_pattern,
                    "m_values": args.ms,
                    "outer": args.outer,
                    "replays_per_outer": args.replays,
                    "warmup_replays": args.warmup_replays,
                    "l2_policy": "cold; 256MiB Triton clear before every replay, clear excluded from events",
                    "l2_cache_bytes": props.L2_cache_size,
                    "l2_flush_bytes": l2_flush_buffer.nbytes,
                    "humming_gemm_type": GemmType.INDEXED.value,
                    "weight_dtype": "OCP MXFP4 E2M1",
                    "weight_scale": "E8M0 group32",
                    "activation_dtype": "FP8 E4M3 group128",
                    "output_dtype": "BF16",
                    "timed_allreduce": "sglang CustomAllReduceV2 default graph heuristic",
                },
                sort_keys=True,
            ),
            flush=True,
        )
        print(
            "BASELINE_SHAPES "
            + json.dumps(
                {
                    "H": HIDDEN,
                    "I": INTERMEDIATE,
                    "I_per_rank": intermediate_per_rank,
                    "experts": NUM_EXPERTS,
                    "top_k": TOP_K,
                    "W13_per_rank": [NUM_EXPERTS, 2 * intermediate_per_rank, HIDDEN],
                    "W2_per_rank": [NUM_EXPERTS, HIDDEN, intermediate_per_rank],
                    "W13_transformed_bytes": w13.humming_config.weight_nbytes,
                    "W2_transformed_bytes": w2.humming_config.weight_nbytes,
                },
                sort_keys=True,
            ),
            flush=True,
        )

    cases: list[CapturedCase] = []
    graphs: list[torch.cuda.CUDAGraph] = []
    results: list[dict[str, Any]] = []

    for m in args.ms:
        topk_ids, topk_weights = make_routes(
            m, args.route_pattern, device, args.seed
        )
        x = torch.randn((m, HIDDEN), dtype=torch.bfloat16, device=device) * 0.1
        selected_w13 = select_tuning_config(w13_tuning, m * TOP_K)
        selected_w2 = select_tuning_config(w2_tuning, m * TOP_K)
        block_m = int(selected_w13["block_shape"][0])
        case = CapturedCase(
            m=m,
            x=x,
            topk_ids=topk_ids,
            topk_weights=topk_weights,
            w13=w13,
            w2=w2,
            w13_tuning=w13_tuning,
            w2_tuning=w2_tuning,
            w13_block_m=block_m,
            intermediate_per_rank=intermediate_per_rank,
        )

        # Eager warm-up resolves every JIT/load path before capture.  This call
        # is collective because it includes CustomAllReduceV2.
        for _ in range(2):
            case.run_full(comm)
        torch.cuda.synchronize(device)
        dist.barrier(group=cpu_group)

        graph = torch.cuda.CUDAGraph()
        with comm.capture():
            with torch.cuda.graph(graph):
                case.run_full(comm)
        torch.cuda.synchronize(device)

        if args.profile_once:
            dist.barrier(group=cpu_group)
            torch.cuda.cudart().cudaProfilerStart()
            triton_runtime.driver.active.clear_cache(l2_flush_buffer)
            graph.replay()
            torch.cuda.synchronize(device)
            torch.cuda.cudart().cudaProfilerStop()
            dist.barrier(group=cpu_group)
            if rank == 0:
                print(
                    "BASELINE_PROFILE_REPLAY "
                    + json.dumps(
                        {
                            "m": m,
                            "l2_policy": "cold; 256MiB clear immediately before replay",
                        },
                        sort_keys=True,
                    ),
                    flush=True,
                )
            cases.append(case)
            graphs.append(graph)
            continue

        for _ in range(args.warmup_replays):
            triton_runtime.driver.active.clear_cache(l2_flush_buffer)
            graph.replay()
        torch.cuda.synchronize(device)

        check = correctness_metrics(case, graph, nccl_group, device)
        samples, batch_medians = time_graph(
            graph,
            outer=args.outer,
            replays=args.replays,
            cpu_group=cpu_group,
            nccl_group=nccl_group,
            device=device,
            l2_flush_buffer=l2_flush_buffer,
        )

        active_experts = int(torch.unique(topk_ids).numel())
        assert case.num_tokens_padded is not None
        padded_rows = int(case.num_tokens_padded.item())
        nbytes = m * HIDDEN * torch.tensor([], dtype=torch.bfloat16).element_size()
        ar_algo, ar_mode = comm._pick_algo(nbytes, can_use_graph=True)
        record: dict[str, Any] = {
            "m": m,
            "route_pattern": args.route_pattern,
            "active_experts": active_experts,
            "routed_rows": m * TOP_K,
            "padded_rows": padded_rows,
            "padding_ratio": padded_rows / (m * TOP_K),
            "w13_block_shape": selected_w13["block_shape"],
            "w2_block_shape": selected_w2["block_shape"],
            "allreduce_bytes": nbytes,
            "allreduce_algo": None if ar_algo is None else ar_algo.name,
            "allreduce_mode": ar_mode.name,
            "latency_ms_min": min(samples),
            "latency_ms_median": statistics.median(samples),
            "latency_ms_max": max(samples),
            "cold_samples": len(samples),
            "batch_medians_ms_max_rank": batch_medians,
            **check,
        }
        results.append(record)
        if rank == 0:
            print("BASELINE_RESULT " + json.dumps(record, sort_keys=True), flush=True)

        # Keep captured tensors and graph-private pools alive until all cases
        # complete; CustomAllReduceV2's graph pointer table references them.
        cases.append(case)
        graphs.append(graph)

    if rank == 0 and not args.profile_once:
        medians = [float(result["latency_ms_median"]) for result in results]
        geometric_mean_ms = statistics.geometric_mean(medians)
        print(
            "BASELINE_SUMMARY "
            + json.dumps(
                {
                    "world_size": world_size,
                    "route_pattern": args.route_pattern,
                    "m_values": args.ms,
                    "median_ms": medians,
                    "geometric_mean_median_ms": geometric_mean_ms,
                    "correctness": "CustomAllReduceV2 graph output vs NCCL sum of captured local output",
                },
                sort_keys=True,
            ),
            flush=True,
        )

    dist.barrier(group=cpu_group)


if __name__ == "__main__":
    main()
