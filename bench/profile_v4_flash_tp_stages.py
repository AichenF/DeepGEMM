#!/usr/bin/env python3
"""Cold-L2 CUDA-event stage breakdown for one TP-rank local MoE path."""

from __future__ import annotations

import argparse
import json
import os
import statistics

import torch
from triton import runtime as triton_runtime

import profile_v4_flash_tp_local as factory


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--impl", choices=("humming", "custom"), required=True)
    parser.add_argument("--m", type=int, default=32)
    parser.add_argument("--tp", type=int, choices=(4, 8), default=4)
    parser.add_argument(
        "--route-pattern", choices=("random", "balanced", "skew"), default="random"
    )
    parser.add_argument("--samples", type=int, default=100)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--mode", choices=("graph", "eager"), default="graph")
    parser.add_argument("--seed", type=int, default=20260902)
    args = parser.parse_args()
    if args.m <= 0 or args.samples <= 0 or args.warmup <= 0:
        parser.error("m, samples, and warmup must be positive")
    return args


def custom_stages(case) -> tuple[tuple[str, ...], tuple[callable, ...]]:
    import v4_flash_tp_wgmma as kernel
    import v4_flash_tp_wgmma_graph as bench

    def align() -> None:
        case.sorted_ids, case.expert_ids, case.num_tokens_padded = (
            bench.moe_align_block_size(
                topk_ids=case.topk_ids,
                block_size=8,
                num_experts=bench.NUM_EXPERTS,
                ignore_invalid_expert=True,
            )
        )

    def quant_x() -> None:
        case.qx, case.x_scale = bench.humming_ops.quant_input(
            inputs=case.x,
            outputs=case.qx,
            dtype="float8e4m3",
            group_size=128,
            m_major_scale=False,
            scale_dtype="float32",
        )

    def w13() -> None:
        kernel.run_w13(
            case.w13,
            case.s13,
            case.g13,
            case.qx.view(torch.uint8),
            case.x_scale,
            case.sorted_ids,
            case.expert_ids,
            case.num_tokens_padded,
            case.partials,
            case.lut,
            case.intermediate_per_rank,
            case.w13_split_k,
        )

    def activation() -> None:
        if kernel.FUSED_ACT_QUANT:
            kernel.reduce_swiglu_quant(
                case.partials,
                case.activation,
                case.qactivation.view(torch.uint8),
                case.activation_scale,
                case.intermediate_per_rank,
                case.w13_split_k,
            )
        else:
            kernel.reduce_swiglu(
                case.partials,
                case.activation,
                case.intermediate_per_rank,
                case.w13_split_k,
            )
            case.qactivation, case.activation_scale = bench.humming_ops.quant_input(
                inputs=case.activation,
                outputs=case.qactivation,
                dtype="float8e4m3",
                group_size=128,
                m_major_scale=False,
                scale_dtype="float32",
            )

    def w2() -> None:
        w2_output = case.down if kernel.W2_ROUTE_OUTPUT else case.local_float
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
            w2_output,
            case.lut,
            case.intermediate_per_rank,
        )

    def local_reduce() -> None:
        if kernel.W2_ROUTE_OUTPUT:
            bench.moe_fused_mul_sum(
                inputs=case.down.view(case.m, bench.TOP_K, bench.HIDDEN),
                topk_weights=case.topk_weights,
                topk_ids=case.topk_ids,
                is_ep=False,
                routed_scaling_factor=bench.ROUTED_SCALING_FACTOR,
                outputs=case.local_bf16,
            )
        else:
            kernel.cast_bf16(case.local_float, case.local_bf16)

    return (
        ("align", "quant_x", "w13", "activation_quant", "w2", "local_reduce"),
        (align, quant_x, w13, activation, w2, local_reduce),
    )


def humming_stages(case) -> tuple[tuple[str, ...], tuple[callable, ...]]:
    import v4_flash_tp_humming_graph as bench

    def align() -> None:
        case.sorted_ids, case.expert_ids, case.num_tokens_padded = (
            bench.moe_align_block_size(
                topk_ids=case.topk_ids,
                block_size=case.w13_block_m,
                num_experts=bench.NUM_EXPERTS,
                ignore_invalid_expert=True,
            )
        )

    def quant_x() -> None:
        case.qx, case.x_scale = bench.HummingMethod.may_quant_input(
            layer=case.w13, inputs=case.x, quanted_input=case.qx
        )

    def w13() -> None:
        bench.HummingMethod.forward_layer(
            layer=case.w13,
            inputs=case.qx,
            input_scale=case.x_scale,
            outputs=case.gate_up,
            sorted_ids=case.sorted_ids,
            expert_ids=case.expert_ids,
            num_tokens_padded=case.num_tokens_padded,
            top_k=bench.TOP_K,
            valid_shape_m=case.valid_shape_m,
            compute_config=bench.COMPUTE_CONFIG,
            tuning_config=case.w13_tuning,
        )

    def activation() -> None:
        bench.silu_and_mul(case.gate_up, case.activation)

    def quant_activation() -> None:
        case.qdown, case.down_scale = bench.HummingMethod.may_quant_input(
            layer=case.w2, inputs=case.activation, quanted_input=case.qdown
        )

    def w2() -> None:
        bench.HummingMethod.forward_layer(
            layer=case.w2,
            inputs=case.qdown,
            input_scale=case.down_scale,
            outputs=case.down,
            sorted_ids=case.sorted_ids,
            expert_ids=case.expert_ids,
            num_tokens_padded=case.num_tokens_padded,
            top_k=1,
            valid_shape_m=case.valid_shape_m,
            compute_config=bench.COMPUTE_CONFIG,
            tuning_config=case.w2_tuning,
        )

    def local_reduce() -> None:
        bench.moe_fused_mul_sum(
            inputs=case.down.view(case.m, bench.TOP_K, bench.HIDDEN),
            topk_weights=case.topk_weights,
            topk_ids=case.topk_ids,
            is_ep=False,
            routed_scaling_factor=bench.ROUTED_SCALING_FACTOR,
            outputs=case.local_output,
        )

    return (
        ("align", "quant_x", "w13", "activation", "quant_activation", "w2", "local_reduce"),
        (align, quant_x, w13, activation, quant_activation, w2, local_reduce),
    )


@torch.inference_mode()
def main() -> None:
    args = parse_args()
    torch.cuda.set_device(0)
    device = torch.device("cuda:0")
    stream = torch.cuda.Stream(device=device)
    torch.cuda.set_stream(stream)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)

    if args.impl == "custom":
        case = factory.make_custom_case(
            args.m, args.tp, args.route_pattern, args.seed, device
        )
        names, functions = custom_stages(case)
    else:
        case = factory.make_humming_case(
            args.m, args.tp, args.route_pattern, args.seed, device
        )
        names, functions = humming_stages(case)

    # Materialize allocations and dispatch caches before either timing mode.
    for _ in range(2):
        for function in functions:
            function()
    torch.cuda.synchronize(device)

    props = torch.cuda.get_device_properties(device)
    flush = triton_runtime.driver.active.get_empty_cache_for_benchmark()
    if flush.nbytes < 2 * props.L2_cache_size:
        raise RuntimeError("cache-clear buffer is smaller than twice L2")

    samples = {name: [] for name in names}
    samples["total"] = []
    if args.mode == "graph":
        events = [
            torch.cuda.Event(enable_timing=True, external=True)
            for _ in range(len(functions) + 1)
        ]
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            events[0].record()
            for index, function in enumerate(functions):
                function()
                events[index + 1].record()
        for _ in range(args.warmup):
            triton_runtime.driver.active.clear_cache(flush)
            graph.replay()
        events[-1].synchronize()
        for _ in range(args.samples):
            triton_runtime.driver.active.clear_cache(flush)
            graph.replay()
            events[-1].synchronize()
            for index, name in enumerate(names):
                samples[name].append(
                    events[index].elapsed_time(events[index + 1])
                )
            samples["total"].append(events[0].elapsed_time(events[-1]))
    else:
        all_events: list[list[torch.cuda.Event]] = []
        for _ in range(args.samples):
            triton_runtime.driver.active.clear_cache(flush)
            events = [
                torch.cuda.Event(enable_timing=True)
                for _ in range(len(functions) + 1)
            ]
            events[0].record()
            for index, function in enumerate(functions):
                function()
                events[index + 1].record()
            all_events.append(events)
        all_events[-1][-1].synchronize()
        for events in all_events:
            for index, name in enumerate(names):
                samples[name].append(
                    events[index].elapsed_time(events[index + 1])
                )
            samples["total"].append(events[0].elapsed_time(events[-1]))

    summary = {
        name: {
            "min_us": min(values) * 1000.0,
            "median_us": statistics.median(values) * 1000.0,
            "max_us": max(values) * 1000.0,
        }
        for name, values in samples.items()
    }
    print(
        "STAGE_PROFILE "
        + json.dumps(
            {
                "impl": args.impl,
                "custom_tiled_weight_layout": (
                    os.environ.get("V4_TILED_WEIGHT_LAYOUT", "1") == "1"
                    if args.impl == "custom"
                    else None
                ),
                "custom_wout": (
                    int(os.environ.get("V4_WOUT", "128"))
                    if args.impl == "custom"
                    else None
                ),
                "custom_w13_split_mode": (
                    os.environ.get("V4_W13_SPLIT_K", "auto")
                    if args.impl == "custom"
                    else None
                ),
                "m": args.m,
                "tp": args.tp,
                "route_pattern": args.route_pattern,
                "samples": args.samples,
                "mode": args.mode,
                "l2_policy": (
                    "cold; separate 256MiB clear before every local pipeline, "
                    "clear excluded from events"
                ),
                "l2_cache_bytes": props.L2_cache_size,
                "l2_flush_bytes": flush.nbytes,
                "stages": summary,
            },
            sort_keys=True,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
