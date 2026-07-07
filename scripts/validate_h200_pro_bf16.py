#!/usr/bin/env python3
"""Validate H200 global-BF16 MegaMoE candidates on identical inputs.

For every requested ``(seed, M)`` pair this runner compares:

1. the retained configuration with FP32 scaled accumulation;
2. the same configuration with global packed-BF16 scaled accumulation;
3. a distributed FP32 PyTorch golden reference.

The golden reference computes only each rank's local experts and uses a
reduce-scatter for the per-top-k contributions.  This keeps the actual Pro
shape tractable without gathering all 384 experts' weights onto every GPU.
"""

import argparse
import json
import os
import random
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import torch
import torch.distributed as dist
import torch.multiprocessing as mp


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import deep_gemm
from deep_gemm.testing import calc_diff
from deep_gemm.utils import per_token_cast_to_fp8
from deep_gemm.utils.dist import init_dist
from tests.test_mega_moe_sm90 import (
    _dequant_block_128_128,
    _dequant_per_token_per_128_k,
    _quantize_grouped_fp8_block_128_128,
    _swiglu_fp32,
)


SHAPES = {
    "flash": {
        "hidden": 4096,
        "intermediate": 2048,
        "num_experts": 256,
        "num_topk": 6,
    },
    "pro": {
        "hidden": 7168,
        "intermediate": 3072,
        "num_experts": 384,
        "num_topk": 6,
    },
}

HIDDEN = SHAPES["pro"]["hidden"]
INTERMEDIATE = SHAPES["pro"]["intermediate"]
NUM_EXPERTS = SHAPES["pro"]["num_experts"]
NUM_TOPK = SHAPES["pro"]["num_topk"]
ACTIVATION_CLAMP = 10.0
CAPACITY = 8192


def _all_gather_equal(tensor: torch.Tensor, num_ranks: int) -> torch.Tensor:
    output = torch.empty(
        (tensor.shape[0] * num_ranks, *tensor.shape[1:]),
        dtype=tensor.dtype,
        device=tensor.device,
    )
    dist.all_gather_into_tensor(output, tensor.contiguous())
    return output


def _distributed_reference(
    x_fp8: torch.Tensor,
    x_sf: torch.Tensor,
    topk_idx: torch.Tensor,
    topk_weights: torch.Tensor,
    l1_w_fp8: torch.Tensor,
    l1_w_sf: torch.Tensor,
    l2_w_fp8: torch.Tensor,
    l2_w_sf: torch.Tensor,
    rank_idx: int,
    num_ranks: int,
) -> torch.Tensor:
    """Return the local-rank golden output for the actual Pro shape."""
    m = x_fp8.shape[0]
    x_fp8_global = _all_gather_equal(x_fp8, num_ranks)
    x_sf_global = _all_gather_equal(x_sf, num_ranks)
    topk_idx_global = _all_gather_equal(topk_idx, num_ranks)
    topk_weights_global = _all_gather_equal(topk_weights, num_ranks)
    x_fp32_global = _dequant_per_token_per_128_k(x_fp8_global, x_sf_global)

    # Each expert is owned by exactly one rank, so summing these sparse slots
    # across ranks reconstructs the full combine input without weight gathers.
    combine_partial = torch.zeros(
        (m * num_ranks, NUM_TOPK, HIDDEN),
        dtype=torch.bfloat16,
        device="cuda",
    )
    first_expert = rank_idx * l1_w_fp8.shape[0]
    for local_expert in range(l1_w_fp8.shape[0]):
        expert = first_expert + local_expert
        assignments = (topk_idx_global == expert).nonzero(as_tuple=False)
        if assignments.numel() == 0:
            continue
        token_indices = assignments[:, 0]
        topk_slots = assignments[:, 1]
        x_selected = x_fp32_global.index_select(0, token_indices)

        l1_weight = _dequant_block_128_128(
            l1_w_fp8[local_expert], l1_w_sf[local_expert]
        )
        gate_up = torch.matmul(x_selected, l1_weight.t())
        del l1_weight
        intermediate = _swiglu_fp32(gate_up, ACTIVATION_CLAMP)
        intermediate.mul_(topk_weights_global[token_indices, topk_slots].unsqueeze(-1))
        del gate_up

        # Match the fused L1 epilogue: per-row/per-64 E4M3 quantize/dequantize.
        intermediate_view = intermediate.view(-1, INTERMEDIATE // 64, 64)
        intermediate_amax = intermediate_view.abs().amax(dim=-1).clamp(1e-4)
        intermediate_sf = intermediate_amax / 448.0
        intermediate_fp8 = (
            intermediate_view / intermediate_sf.unsqueeze(-1)
        ).to(torch.float8_e4m3fn)
        l2_input = (
            intermediate_fp8.float() * intermediate_sf.unsqueeze(-1)
        ).view(-1, INTERMEDIATE)
        del intermediate, intermediate_view, intermediate_amax, intermediate_sf
        del intermediate_fp8

        l2_weight = _dequant_block_128_128(
            l2_w_fp8[local_expert], l2_w_sf[local_expert]
        )
        contribution = torch.matmul(l2_input, l2_weight.t())
        del l2_input, l2_weight
        combine_partial[token_indices, topk_slots] = contribution.to(torch.bfloat16)
        del contribution, assignments, token_indices, topk_slots, x_selected

    local_slots = torch.empty(
        (m, NUM_TOPK, HIDDEN), dtype=torch.bfloat16, device="cuda"
    )
    dist.reduce_scatter_tensor(local_slots, combine_partial, op=dist.ReduceOp.SUM)
    result = local_slots.sum(dim=1).to(torch.bfloat16)
    del combine_partial, local_slots
    del x_fp8_global, x_sf_global, topk_idx_global, topk_weights_global
    del x_fp32_global
    return result


def _launch(
    buffer,
    transformed_l1,
    transformed_l2,
    x_fp8: torch.Tensor,
    x_sf: torch.Tensor,
    topk_idx: torch.Tensor,
    topk_weights: torch.Tensor,
    use_bf16_accum: bool,
    fp8_combine: str,
    num_experts_per_rank: int,
) -> torch.Tensor:
    if fp8_combine == "auto":
        os.environ.pop("DG_SM90_MOE_FP8_COMBINE", None)
    else:
        os.environ["DG_SM90_MOE_FP8_COMBINE"] = fp8_combine
    os.environ["DG_SM90_MOE_BF16_SCALED_ACCUM"] = "1" if use_bf16_accum else "0"
    m = x_fp8.shape[0]
    buffer.x[:m].copy_(x_fp8)
    buffer.x_sf[:m].copy_(x_sf)
    buffer.topk_idx[:m].copy_(topk_idx)
    buffer.topk_weights[:m].copy_(topk_weights)
    cumulative_stats = torch.zeros(
        num_experts_per_rank, dtype=torch.int, device="cuda"
    )
    output = torch.empty((m, HIDDEN), dtype=torch.bfloat16, device="cuda")
    dist.barrier()
    deep_gemm.fp8_mega_moe(
        output,
        transformed_l1,
        transformed_l2,
        buffer,
        cumulative_local_expert_recv_stats=cumulative_stats,
        recipe=(128, 128, 128),
        activation="swiglu",
        activation_clamp=ACTIVATION_CLAMP,
        fast_math=True,
    )
    torch.cuda.synchronize()
    dist.barrier()
    return output


def _diff(lhs: torch.Tensor, rhs: torch.Tensor) -> float:
    value = calc_diff(lhs, rhs)
    return float(value.item() if isinstance(value, torch.Tensor) else value)


def _collect_case_metrics(
    fp32_output: torch.Tensor,
    bf16_output: torch.Tensor,
    golden: torch.Tensor,
    num_ranks: int,
) -> Tuple[List[Dict[str, float]], Dict[str, float]]:
    local = torch.tensor(
        [
            float(torch.isfinite(fp32_output).all()),
            float(torch.isfinite(bf16_output).all()),
            float(torch.isfinite(golden).all()),
            _diff(fp32_output, golden),
            _diff(bf16_output, golden),
            _diff(bf16_output, fp32_output),
            float((bf16_output.float() - fp32_output.float()).abs().max().item()),
        ],
        dtype=torch.float64,
        device="cuda",
    )
    gathered = [torch.empty_like(local) for _ in range(num_ranks)]
    dist.all_gather(gathered, local)
    rows = []
    for rank, values in enumerate(gathered):
        values = values.cpu().tolist()
        rows.append(
            {
                "rank": rank,
                "fp32_finite": bool(values[0]),
                "bf16_finite": bool(values[1]),
                "golden_finite": bool(values[2]),
                "fp32_vs_golden": values[3],
                "bf16_vs_golden": values[4],
                "bf16_vs_fp32": values[5],
                "bf16_vs_fp32_max_abs": values[6],
            }
        )
    summary = {
        "max_fp32_vs_golden": max(row["fp32_vs_golden"] for row in rows),
        "max_bf16_vs_golden": max(row["bf16_vs_golden"] for row in rows),
        "max_bf16_vs_fp32": max(row["bf16_vs_fp32"] for row in rows),
        "max_bf16_vs_fp32_abs": max(row["bf16_vs_fp32_max_abs"] for row in rows),
        "all_finite": all(
            row["fp32_finite"] and row["bf16_finite"] and row["golden_finite"]
            for row in rows
        ),
    }
    return rows, summary


def _make_weights(seed: int, rank_idx: int, num_experts_per_rank: int):
    torch.manual_seed(seed * 1_000_003 + rank_idx * 10_007 + 17)
    random.seed(seed * 1_000_003 + rank_idx * 10_007 + 17)
    l1_bf16 = torch.randn(
        (num_experts_per_rank, INTERMEDIATE * 2, HIDDEN),
        dtype=torch.bfloat16,
        device="cuda",
    ).mul_(0.05)
    l1_fp8, l1_sf = _quantize_grouped_fp8_block_128_128(l1_bf16)
    del l1_bf16
    l2_bf16 = torch.randn(
        (num_experts_per_rank, HIDDEN, INTERMEDIATE),
        dtype=torch.bfloat16,
        device="cuda",
    ).mul_(0.05)
    l2_fp8, l2_sf = _quantize_grouped_fp8_block_128_128(l2_bf16)
    del l2_bf16
    transformed_l1, transformed_l2 = deep_gemm.transform_weights_for_mega_moe_sm90(
        (l1_fp8, l1_sf), (l2_fp8, l2_sf)
    )
    return l1_fp8, l1_sf, l2_fp8, l2_sf, transformed_l1, transformed_l2


def _make_case(seed: int, m: int, rank_idx: int):
    torch.manual_seed(seed * 1_000_003 + m * 1_009 + rank_idx)
    random.seed(seed * 1_000_003 + m * 1_009 + rank_idx)
    x_bf16 = torch.randn((m, HIDDEN), dtype=torch.bfloat16, device="cuda")
    x_fp8, x_sf = per_token_cast_to_fp8(
        x_bf16, use_ue8m0=False, gran_k=128, use_packed_ue8m0=False
    )
    scores = torch.randn((m, NUM_EXPERTS), dtype=torch.float, device="cuda")
    topk_weights, topk_idx = torch.topk(
        scores, NUM_TOPK, dim=-1, largest=True, sorted=False
    )
    return x_fp8, x_sf, topk_idx, topk_weights


def worker(local_rank: int, num_processes: int, cfg: Dict):
    global HIDDEN, INTERMEDIATE, NUM_EXPERTS, NUM_TOPK
    shape = SHAPES[cfg["shape"]]
    HIDDEN = shape["hidden"]
    INTERMEDIATE = shape["intermediate"]
    NUM_EXPERTS = shape["num_experts"]
    NUM_TOPK = shape["num_topk"]

    rank_idx, num_ranks, group = init_dist(local_rank, num_processes)
    assert num_ranks == 8, f"expected 8 ranks, got {num_ranks}"
    assert "H200" in torch.cuda.get_device_name(), torch.cuda.get_device_name()
    torch.set_float32_matmul_precision("highest")
    num_experts_per_rank = NUM_EXPERTS // num_ranks
    failures = []
    case_summaries = []

    for seed in cfg["seeds"]:
        if rank_idx == 0:
            print(f"VALIDATION_STAGE seed={seed} stage=make_weights", flush=True)
        weights = _make_weights(seed, rank_idx, num_experts_per_rank)
        l1_fp8, l1_sf, l2_fp8, l2_sf, transformed_l1, transformed_l2 = weights
        buffer = deep_gemm.get_symm_buffer_for_mega_moe(
            group,
            NUM_EXPERTS,
            CAPACITY,
            NUM_TOPK,
            HIDDEN,
            INTERMEDIATE,
        )

        for m in cfg["ms"]:
            if rank_idx == 0:
                print(f"VALIDATION_STAGE seed={seed} m={m} stage=inputs", flush=True)
            x_fp8, x_sf, topk_idx, topk_weights = _make_case(seed, m, rank_idx)
            fp32_output = _launch(
                buffer,
                transformed_l1,
                transformed_l2,
                x_fp8,
                x_sf,
                topk_idx,
                topk_weights,
                False,
                cfg["fp8_combine"],
                num_experts_per_rank,
            )
            bf16_output = _launch(
                buffer,
                transformed_l1,
                transformed_l2,
                x_fp8,
                x_sf,
                topk_idx,
                topk_weights,
                True,
                cfg["fp8_combine"],
                num_experts_per_rank,
            )
            if rank_idx == 0:
                print(f"VALIDATION_STAGE seed={seed} m={m} stage=golden", flush=True)
            golden = _distributed_reference(
                x_fp8,
                x_sf,
                topk_idx,
                topk_weights,
                l1_fp8,
                l1_sf,
                l2_fp8,
                l2_sf,
                rank_idx,
                num_ranks,
            )
            rows, summary = _collect_case_metrics(
                fp32_output, bf16_output, golden, num_ranks
            )
            passed = (
                summary["all_finite"]
                and summary["max_fp32_vs_golden"] < cfg["diff_tol"]
                and summary["max_bf16_vs_golden"] < cfg["diff_tol"]
                and summary["max_bf16_vs_fp32"] < cfg["diff_tol"]
            )
            if rank_idx == 0:
                print(
                    "VALIDATION_JSON "
                    + json.dumps(
                        {
                            "shape": cfg["shape"],
                            "fp8_combine": cfg["fp8_combine"],
                            "seed": seed,
                            "m": m,
                            "diff_tol": cfg["diff_tol"],
                            "passed": passed,
                            "summary": summary,
                            "ranks": rows,
                        },
                        sort_keys=True,
                    ),
                    flush=True,
                )
            if not passed:
                failures.append((seed, m, summary))
            case_summaries.append({"seed": seed, "m": m, **summary})
            del x_fp8, x_sf, topk_idx, topk_weights
            del fp32_output, bf16_output, golden
            torch.cuda.empty_cache()
            dist.barrier()

        buffer.destroy()
        del buffer, weights, l1_fp8, l1_sf, l2_fp8, l2_sf
        del transformed_l1, transformed_l2
        torch.cuda.empty_cache()
        dist.barrier()

    failure_count = torch.tensor([len(failures)], dtype=torch.int, device="cuda")
    dist.all_reduce(failure_count, op=dist.ReduceOp.MAX)
    if rank_idx == 0:
        worst = {}
        for metric in (
            "max_fp32_vs_golden",
            "max_bf16_vs_golden",
            "max_bf16_vs_fp32",
            "max_bf16_vs_fp32_abs",
        ):
            row = max(case_summaries, key=lambda item: item[metric])
            worst[metric] = {
                "value": row[metric],
                "seed": row["seed"],
                "m": row["m"],
            }
        print(
            "VALIDATION_FINAL_JSON "
            + json.dumps(
                {
                    "shape": cfg["shape"],
                    "fp8_combine": cfg["fp8_combine"],
                    "seeds": cfg["seeds"],
                    "ms": cfg["ms"],
                    "num_cases": len(cfg["seeds"]) * len(cfg["ms"]),
                    "passed": failure_count.item() == 0,
                    "failure_count": failure_count.item(),
                    "worst": worst,
                },
                sort_keys=True,
            ),
            flush=True,
        )
    dist.barrier()
    dist.destroy_process_group()
    if failure_count.item():
        raise SystemExit(1)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--shape", choices=tuple(SHAPES), default="pro")
    parser.add_argument("--fp8-combine", choices=("auto", "0", "1"), default="auto")
    parser.add_argument("--seeds", type=int, nargs="+", default=[7, 23, 101, 509])
    parser.add_argument(
        "--ms", type=int, nargs="+", default=[128, 1024, 2048, 4096, 8192]
    )
    parser.add_argument("--diff-tol", type=float, default=0.01)
    parser.add_argument("--num-processes", type=int, default=8)
    args = parser.parse_args()
    if any(m <= 0 or m > CAPACITY for m in args.ms):
        parser.error(f"all M values must be in [1, {CAPACITY}]")
    mp.spawn(worker, args=(args.num_processes, vars(args)), nprocs=args.num_processes)


if __name__ == "__main__":
    main()
