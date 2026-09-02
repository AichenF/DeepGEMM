#!/usr/bin/env python3
"""Expose one cold per-rank V4 Flash MoE pipeline to Nsight Compute."""

from __future__ import annotations

import argparse
import json
import os

import torch
from triton import runtime as triton_runtime


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--impl", choices=("humming", "custom"), required=True)
    parser.add_argument("--m", type=int, default=32)
    parser.add_argument("--tp", type=int, choices=(4, 8), default=4)
    parser.add_argument(
        "--route-pattern",
        choices=("random", "balanced", "skew"),
        default="random",
    )
    parser.add_argument("--seed", type=int, default=20260902)
    args = parser.parse_args()
    if args.m <= 0:
        parser.error("--m must be positive")
    return args


def make_humming_case(
    m: int, tp: int, route_pattern: str, seed: int, device: torch.device
):
    import v4_flash_tp_humming_graph as bench
    from humming.config import GemmType
    from humming.layer import HummingMethod

    intermediate_per_rank = bench.INTERMEDIATE // tp
    w13 = bench.make_layer(2 * intermediate_per_rank, bench.HIDDEN, device)
    w2 = bench.make_layer(bench.HIDDEN, intermediate_per_rank, device)
    w13_tuning = HummingMethod.get_default_tuning_configs(
        layer=w13, use_f16_accum=False, gemm_type=GemmType.INDEXED
    )
    w2_tuning = HummingMethod.get_default_tuning_configs(
        layer=w2, use_f16_accum=False, gemm_type=GemmType.INDEXED
    )
    topk_ids, topk_weights = bench.make_routes(
        m, route_pattern, device, seed
    )
    x = torch.randn((m, bench.HIDDEN), dtype=torch.bfloat16, device=device) * 0.1
    selected_w13 = bench.select_tuning_config(w13_tuning, m * bench.TOP_K)
    return bench.CapturedCase(
        m=m,
        x=x,
        topk_ids=topk_ids,
        topk_weights=topk_weights,
        w13=w13,
        w2=w2,
        w13_tuning=w13_tuning,
        w2_tuning=w2_tuning,
        w13_block_m=int(selected_w13["block_shape"][0]),
        intermediate_per_rank=intermediate_per_rank,
    )


def make_custom_case(
    m: int, tp: int, route_pattern: str, seed: int, device: torch.device
):
    import v4_flash_tp_wgmma_graph as bench
    import v4_flash_tp_wgmma as kernel

    intermediate_per_rank = bench.INTERMEDIATE // tp
    w13, s13, w2, s2 = bench.make_weights(intermediate_per_rank, device)
    topk_ids, topk_weights = bench.make_routes(
        m, route_pattern, device, seed
    )
    x = torch.randn((m, bench.HIDDEN), dtype=torch.bfloat16, device=device) * 0.1
    return bench.CapturedCase(
        m=m,
        x=x,
        topk_ids=topk_ids,
        topk_weights=topk_weights,
        w13=w13,
        s13=s13,
        w2=w2,
        s2=s2,
        lut=kernel.make_e2m1_e8m0_lut(device),
        intermediate_per_rank=intermediate_per_rank,
    )


@torch.inference_mode()
def main() -> None:
    args = parse_args()
    device = torch.device("cuda:0")
    torch.cuda.set_device(device)
    stream = torch.cuda.Stream(device=device)
    torch.cuda.set_stream(stream)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)

    if args.impl == "humming":
        case = make_humming_case(
            args.m, args.tp, args.route_pattern, args.seed, device
        )
    else:
        case = make_custom_case(
            args.m, args.tp, args.route_pattern, args.seed, device
        )

    for _ in range(2):
        case.run_local()
    torch.cuda.synchronize(device)

    props = torch.cuda.get_device_properties(device)
    l2_flush_buffer = triton_runtime.driver.active.get_empty_cache_for_benchmark()
    if l2_flush_buffer.nbytes < 2 * props.L2_cache_size:
        raise RuntimeError("benchmark cache buffer is smaller than twice L2")

    torch.cuda.cudart().cudaProfilerStart()
    triton_runtime.driver.active.clear_cache(l2_flush_buffer)
    case.run_local()
    torch.cuda.synchronize(device)
    torch.cuda.cudart().cudaProfilerStop()

    print(
        "LOCAL_PROFILE_REPLAY "
        + json.dumps(
            {
                "impl": args.impl,
                "m": args.m,
                "tp": args.tp,
                "route_pattern": args.route_pattern,
                "w13_split_policy": os.environ.get("V4_W13_SPLIT_K", "auto")
                if args.impl == "custom"
                else None,
                "output_tile_channels": int(os.environ.get("V4_WOUT", "128"))
                if args.impl == "custom"
                else None,
                "w2_route_output": os.environ.get("V4_W2_ROUTE_OUTPUT", "1") == "1"
                if args.impl == "custom"
                else None,
                "min_blocks_per_sm": int(
                    os.environ.get("V4_MIN_BLOCKS_PER_SM", "0")
                )
                if args.impl == "custom"
                else None,
                "mxfp4_lut_rows": int(os.environ.get("V4_LUT_ROWS", "256"))
                if args.impl == "custom"
                else None,
                "scale_quad_reuse": int(
                    os.environ.get("V4_SCALE_QUAD_REUSE", "4")
                )
                if args.impl == "custom"
                else None,
                "scale_buffers": int(os.environ.get("V4_SCALE_BUFFERS", "2"))
                if args.impl == "custom"
                else None,
                "weight_swizzle_bytes": int(
                    os.environ.get("V4_WEIGHT_SWIZZLE", "64")
                )
                if args.impl == "custom"
                else None,
                "l2_cache_bytes": props.L2_cache_size,
                "l2_flush_bytes": l2_flush_buffer.nbytes,
                "l2_policy": "cold; 256MiB clear immediately before pipeline",
            },
            sort_keys=True,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
