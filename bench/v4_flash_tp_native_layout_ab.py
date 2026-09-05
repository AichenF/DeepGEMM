#!/usr/bin/env python3
"""Same-process cold-L2 A/B for two native MegaMoE variants."""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import statistics
import sys
from pathlib import Path
from types import ModuleType
from typing import Any

import torch
import torch.distributed as dist
from triton import runtime as triton_runtime

import sglang.srt.distributed.parallel_state as ps
try:
    from sglang.jit_kernel.mp import register_comm_cleanup
except ImportError:
    from sglang.kernels.ops.communication.mp import register_comm_cleanup
from sglang.srt.distributed.device_communicators.custom_all_reduce_v2 import (
    CustomAllReduceV2,
)

import v4_flash_tp_wgmma as kernel
import v4_flash_tp_wgmma_graph as custom
from v4_flash_tp_paired_graph import capture_graph, time_graph_pair
from v4_flash_tp_single_vs_multi_graph import make_case


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ms", default="8,128")
    parser.add_argument("--outer", type=int, default=4)
    parser.add_argument("--replays", type=int, default=50)
    parser.add_argument("--warmup-replays", type=int, default=5)
    parser.add_argument("--seed", type=int, default=20260902)
    parser.add_argument(
        "--experiment",
        choices=("tile_tma", "single_l1_warmup", "dual_dispatch"),
        default="tile_tma",
    )
    args = parser.parse_args()
    args.ms = tuple(int(value) for value in args.ms.split(",") if value)
    if not args.ms or any(value not in (8, 16, 32, 64, 128) for value in args.ms):
        parser.error("--ms must contain values from 8,16,32,64,128")
    if args.outer < 2 or args.outer % 2:
        parser.error("--outer must be a positive even number >= 2")
    if args.replays < 1 or args.warmup_replays < 1:
        parser.error("replay counts must be positive")
    return args


def load_native_variant(
    alias: str,
    *,
    tile_tma: bool = False,
    single_l1_warmup_wave: bool = False,
    dual_active_dispatch: bool = False,
) -> ModuleType:
    source = Path(__file__).resolve().parents[1] / "v4_flash_tp_native_megamoe.py"
    saved = {
        name: os.environ.get(name)
        for name in (
            "V4_NATIVE_TILE_WEIGHT_SCALE_TMA",
            "V4_NATIVE_SPLIT_WEIGHT_SCALE_TMA",
            "V4_NATIVE_SINGLE_L1_WARMUP_WAVE",
            "V4_NATIVE_DUAL_ACTIVE_DISPATCH",
        )
    }
    try:
        os.environ["V4_NATIVE_TILE_WEIGHT_SCALE_TMA"] = str(int(tile_tma))
        os.environ["V4_NATIVE_SPLIT_WEIGHT_SCALE_TMA"] = "0"
        os.environ["V4_NATIVE_SINGLE_L1_WARMUP_WAVE"] = str(
            int(single_l1_warmup_wave)
        )
        os.environ["V4_NATIVE_DUAL_ACTIVE_DISPATCH"] = str(
            int(dual_active_dispatch)
        )
        spec = importlib.util.spec_from_file_location(alias, source)
        if spec is None or spec.loader is None:
            raise RuntimeError(f"cannot load native variant from {source}")
        module = importlib.util.module_from_spec(spec)
        sys.modules[alias] = module
        spec.loader.exec_module(module)
        return module
    finally:
        for name, value in saved.items():
            if value is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = value


def make_variant_weights(
    module: ModuleType,
    intermediate_per_rank: int,
    device: torch.device,
    seed: int,
    rank: int,
) -> tuple[torch.Tensor, ...]:
    torch.manual_seed(seed + rank)
    torch.cuda.manual_seed(seed + rank)
    return custom.make_weights(
        intermediate_per_rank,
        device,
        include_native=True,
        native_kernel_module=module,
    )


def rank_metrics(
    control: torch.Tensor,
    candidate: torch.Tensor,
    nccl_group: dist.ProcessGroup,
    device: torch.device,
) -> dict[str, float | int | bool]:
    control_f = control.double()
    candidate_f = candidate.double()
    cosine = float(
        torch.nn.functional.cosine_similarity(
            control_f.flatten(), candidate_f.flatten(), dim=0
        ).item()
    )
    rel_l2 = float(
        (
            torch.linalg.vector_norm(candidate_f - control_f)
            / torch.linalg.vector_norm(control_f).clamp_min(1e-40)
        ).item()
    )
    mismatches = float((candidate != control).sum().item())
    finite = float(bool(torch.isfinite(control).all() and torch.isfinite(candidate).all()))
    return {
        "cosine_min_rank": custom.reduce_rank_metric(
            cosine, dist.ReduceOp.MIN, device, nccl_group
        ),
        "rel_l2_max_rank": custom.reduce_rank_metric(
            rel_l2, dist.ReduceOp.MAX, device, nccl_group
        ),
        "bf16_mismatches_max_rank": int(
            custom.reduce_rank_metric(
                mismatches, dist.ReduceOp.MAX, device, nccl_group
            )
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
        raise RuntimeError("native-variant A/B currently requires TP4")
    nccl_group = ps._WORLD.device_group
    if not isinstance(nccl_group, dist.ProcessGroup):
        raise RuntimeError("SGLang did not create the NCCL process group")

    props = torch.cuda.get_device_properties(device)
    intermediate_per_rank = custom.INTERMEDIATE // world_size
    control_module = load_native_variant(
        "v4_native_variant_control",
        tile_tma=False,
        single_l1_warmup_wave=False,
        dual_active_dispatch=False,
    )
    if args.experiment == "tile_tma":
        candidate_module = load_native_variant(
            "v4_native_variant_tile_tma",
            tile_tma=True,
            single_l1_warmup_wave=False,
            dual_active_dispatch=False,
        )
        benchmark_name = "native_80b_vs_single_tile_tma"
    elif args.experiment == "single_l1_warmup":
        candidate_module = load_native_variant(
            "v4_native_variant_single_l1_warmup",
            tile_tma=False,
            single_l1_warmup_wave=True,
            dual_active_dispatch=False,
        )
        benchmark_name = "native_two_vs_one_l1_warmup_wave"
    else:
        candidate_module = load_native_variant(
            "v4_native_variant_dual_dispatch",
            tile_tma=False,
            single_l1_warmup_wave=False,
            dual_active_dispatch=True,
        )
        benchmark_name = "native_single_vs_dual_active_dispatch"
    control_weights = make_variant_weights(
        control_module, intermediate_per_rank, device, args.seed, rank
    )
    candidate_weights = make_variant_weights(
        candidate_module, intermediate_per_rank, device, args.seed, rank
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
            "NATIVE_VARIANT_ENV "
            + json.dumps(
                {
                    "benchmark": benchmark_name,
                    "experiment": args.experiment,
                    "gpu": props.name,
                    "sm_count": props.multi_processor_count,
                    "world_size": world_size,
                    "m_values": list(args.ms),
                    "outer": args.outer,
                    "replays": args.replays,
                    "warmup_replays": args.warmup_replays,
                    "pair_order": "balanced whole-batch AB/BA",
                    "l2_policy": (
                        "separate 256MiB clear immediately before every replay; "
                        "clear excluded from CUDA events"
                    ),
                    "control_tile_tma": control_module.NATIVE_TILE_WEIGHT_SCALE_TMA,
                    "candidate_tile_tma": (
                        candidate_module.NATIVE_TILE_WEIGHT_SCALE_TMA
                    ),
                    "control_single_l1_warmup_wave": (
                        control_module.NATIVE_SINGLE_L1_WARMUP_WAVE
                    ),
                    "candidate_single_l1_warmup_wave": (
                        candidate_module.NATIVE_SINGLE_L1_WARMUP_WAVE
                    ),
                    "control_dual_active_dispatch": (
                        control_module.NATIVE_DUAL_ACTIVE_DISPATCH
                    ),
                    "candidate_dual_active_dispatch": (
                        candidate_module.NATIVE_DUAL_ACTIVE_DISPATCH
                    ),
                },
                sort_keys=True,
            ),
            flush=True,
        )

    records: list[dict[str, Any]] = []
    keepalive: list[Any] = []
    driver = triton_runtime.driver.active
    kernel.SINGLE_LAUNCH_TP4 = True
    for m in args.ms:
        topk_ids, topk_weights = custom.make_routes(
            m, "random", device, args.seed
        )
        qx, x_scale = custom.make_fp8_input(m, device, args.seed)
        control_case = make_case(
            m,
            qx,
            x_scale,
            topk_ids,
            topk_weights,
            control_weights,
            lut,
            intermediate_per_rank,
            use_native=True,
            native_kernel_module=control_module,
        )
        candidate_case = make_case(
            m,
            qx,
            x_scale,
            topk_ids,
            topk_weights,
            candidate_weights,
            lut,
            intermediate_per_rank,
            use_native=True,
            native_kernel_module=candidate_module,
        )
        control_graph = capture_graph(
            control_case, comm, cpu_group, device
        )
        candidate_graph = capture_graph(
            candidate_case, comm, cpu_group, device
        )

        control_graph.replay()
        torch.cuda.synchronize(device)
        assert control_case.graph_output is not None
        control_output = control_case.graph_output.clone()
        candidate_graph.replay()
        torch.cuda.synchronize(device)
        assert candidate_case.graph_output is not None
        candidate_output = candidate_case.graph_output.clone()
        correctness = rank_metrics(
            control_output, candidate_output, nccl_group, device
        )
        if rank == 0:
            print(
                "NATIVE_VARIANT_CORRECTNESS "
                + json.dumps({"m": m, **correctness}, sort_keys=True),
                flush=True,
            )
        if not correctness["finite_all_ranks"] or correctness[
            "bf16_mismatches_max_rank"
        ]:
            raise RuntimeError(f"native variant mismatch at M={m}")

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
            "batch",
        )
        control_median = statistics.median(control_samples)
        candidate_median = statistics.median(candidate_samples)
        record = {
            "m": m,
            "samples_per_variant": len(control_samples),
            "control_min_ms": min(control_samples),
            "control_median_ms": control_median,
            "control_max_ms": max(control_samples),
            "candidate_min_ms": min(candidate_samples),
            "candidate_median_ms": candidate_median,
            "candidate_max_ms": max(candidate_samples),
            "control_batch_medians_ms": control_batch_medians,
            "candidate_batch_medians_ms": candidate_batch_medians,
            "speedup_control_over_candidate": (
                control_median / candidate_median
            ),
            "candidate_over_control": candidate_median / control_median,
        }
        records.append(record)
        if rank == 0:
            print(
                "NATIVE_VARIANT_RESULT "
                + json.dumps(record, sort_keys=True),
                flush=True,
            )
        keepalive.extend(
            (control_case, candidate_case, control_graph, candidate_graph)
        )

    if rank == 0:
        control_gm = statistics.geometric_mean(
            float(record["control_median_ms"]) for record in records
        )
        candidate_gm = statistics.geometric_mean(
            float(record["candidate_median_ms"]) for record in records
        )
        print(
            "NATIVE_VARIANT_SUMMARY "
            + json.dumps(
                {
                    "m_values": list(args.ms),
                    "control_geometric_mean_ms": control_gm,
                    "candidate_geometric_mean_ms": candidate_gm,
                    "speedup_control_over_candidate": (
                        control_gm / candidate_gm
                    ),
                    "candidate_over_control": candidate_gm / control_gm,
                    "samples_per_m_per_layout": args.outer * args.replays,
                },
                sort_keys=True,
            ),
            flush=True,
        )
    dist.barrier(group=cpu_group)


if __name__ == "__main__":
    main()
