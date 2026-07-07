#!/usr/bin/env python3
"""Standalone SM90 DeepEP low-latency + masked grouped-FP8 benchmark."""

import argparse
import json
import math
import os
import random
import statistics
import sys
from typing import Tuple

import torch
import torch.distributed as dist
import triton
import triton.language as tl


REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

import deep_gemm
from deep_gemm.testing import get_arch_major
from deep_gemm.utils.dist import init_dist


SHAPES = {
    "flash": {
        "hidden": 4096,
        "intermediate_hidden": 2048,
        "num_experts": 256,
        "num_topk": 6,
    },
    "pro": {
        "hidden": 7168,
        "intermediate_hidden": 3072,
        "num_experts": 384,
        "num_topk": 6,
    },
    "mimo_pro": {
        "hidden": 6144,
        "intermediate_hidden": 2048,
        "num_experts": 384,
        "num_topk": 8,
    },
}

FP8_E4M3_MAX = 448.0
_FP8_E4M3_MAX_TL = tl.constexpr(448.0)
ACT_SF_GRAN = 128


try:
    import deep_ep as _deep_ep

    _DEEP_EP_IMPORT_ERROR = None
except Exception as exc:  # pragma: no cover - exercised on the GPU host
    _deep_ep = None
    _DEEP_EP_IMPORT_ERROR = exc


def quantize_grouped_fp8_block_128_128(
    w: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Convert (G, N, K) BF16 weights to E4M3 plus FP32 128x128 scales."""
    g, n, k = w.shape
    assert w.is_cuda and w.dtype == torch.bfloat16
    assert n % 128 == 0 and k % 128 == 0
    w_fp8 = torch.empty_like(w, dtype=torch.float8_e4m3fn)
    sf = torch.empty((g, n // 128, k // 128), dtype=torch.float32, device=w.device)
    for start in range(0, g, 4):
        end = min(start + 4, g)
        view = w[start:end].view(
            end - start, n // 128, 128, k // 128, 128
        ).float()
        sf_chunk = view.abs().amax(dim=(-1, -3)).clamp(1e-4) / FP8_E4M3_MAX
        w_fp8[start:end].copy_(
            (view / sf_chunk.unsqueeze(-1).unsqueeze(-3))
            .to(torch.float8_e4m3fn)
            .view(end - start, n, k)
        )
        sf[start:end].copy_(sf_chunk)
    return w_fp8, sf.contiguous()


@triton.jit
def _swiglu_masked_post_quant_kernel(
    x_ptr,
    stride_x_e,
    stride_x_m,
    stride_x_n,
    y_ptr,
    stride_y_e,
    stride_y_m,
    stride_y_n,
    y_sf_ptr,
    stride_sf_e,
    stride_sf_m,
    stride_sf_k,
    masked_m_ptr,
    H,
    clamp_value,
    HAS_CLAMP: tl.constexpr,
    USE_UE8M0_SCALE: tl.constexpr,
    BLOCK_K: tl.constexpr,
    NUM_STAGES: tl.constexpr,
):
    pid_k = tl.program_id(0)
    pid_m = tl.program_id(1)
    pid_e = tl.program_id(2)
    num_token_stripes = tl.num_programs(1)
    num_valid_tokens = tl.load(masked_m_ptr + pid_e)
    offs_k = pid_k * BLOCK_K + tl.arange(0, BLOCK_K)

    x_base = x_ptr + pid_e * stride_x_e + offs_k * stride_x_n
    y_base = y_ptr + pid_e * stride_y_e + offs_k * stride_y_n
    sf_base = y_sf_ptr + pid_e * stride_sf_e + pid_k * stride_sf_k

    for token in tl.range(
        pid_m, num_valid_tokens, num_token_stripes, num_stages=NUM_STAGES
    ):
        gate = tl.load(x_base + token * stride_x_m).to(tl.float32)
        up = tl.load(x_base + token * stride_x_m + H * stride_x_n).to(tl.float32)
        if HAS_CLAMP:
            gate = tl.minimum(gate, clamp_value)
            up = tl.minimum(tl.maximum(up, -clamp_value), clamp_value)
        y = gate * tl.sigmoid(gate) * up
        amax = tl.max(tl.abs(y))
        sf = tl.maximum(amax / _FP8_E4M3_MAX_TL, 1.0e-30)
        if USE_UE8M0_SCALE:
            sf = tl.exp2(tl.ceil(tl.log2(sf)))
        tl.store(y_base + token * stride_y_m, (y / sf).to(tl.float8e4nv))
        tl.store(sf_base + token * stride_sf_m, sf)


def swiglu_masked_post_quant_to_fp8(
    x: torch.Tensor,
    masked_m: torch.Tensor,
    clamp_value: float | None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    assert x.is_cuda and x.dtype == torch.bfloat16 and x.is_contiguous()
    assert x.dim() == 3 and x.shape[-1] % 2 == 0
    num_experts, max_m, two_h = x.shape
    hidden = two_h // 2
    assert hidden % ACT_SF_GRAN == 0
    assert masked_m.shape == (num_experts,)

    y = torch.empty(
        (num_experts, max_m, hidden),
        dtype=torch.float8_e4m3fn,
        device=x.device,
    )
    y_sf = torch.empty(
        (num_experts, max_m, hidden // ACT_SF_GRAN),
        dtype=torch.float32,
        device=x.device,
    )
    token_stripes = 64 if num_experts < 4 else 32
    grid = (hidden // ACT_SF_GRAN, token_stripes, num_experts)
    _swiglu_masked_post_quant_kernel[grid](
        x,
        x.stride(0),
        x.stride(1),
        x.stride(2),
        y,
        y.stride(0),
        y.stride(1),
        y.stride(2),
        y_sf,
        y_sf.stride(0),
        y_sf.stride(1),
        y_sf.stride(2),
        masked_m,
        hidden,
        float(clamp_value) if clamp_value is not None else 0.0,
        HAS_CLAMP=clamp_value is not None,
        USE_UE8M0_SCALE=False,
        BLOCK_K=ACT_SF_GRAN,
        NUM_STAGES=4,
        num_warps=1,
    )
    return y, y_sf


def _expected_local_routes(
    topk_idx: torch.Tensor,
    num_experts: int,
    rank: int,
    num_ranks: int,
    group,
) -> torch.Tensor:
    counts = torch.bincount(topk_idx.reshape(-1), minlength=num_experts)
    dist.all_reduce(counts, op=dist.ReduceOp.SUM, group=group)
    experts_per_rank = num_experts // num_ranks
    return counts[
        rank * experts_per_rank : (rank + 1) * experts_per_rank
    ].to(torch.int)


def _worker(local_rank: int, num_processes: int, cfg: dict):
    if _deep_ep is None:
        raise RuntimeError(f"DeepEP import failed: {_DEEP_EP_IMPORT_ERROR}")

    rank, num_ranks, group = init_dist(local_rank, num_processes)
    buffer = None
    completed = False
    try:
        if get_arch_major() != 9:
            raise RuntimeError(f"SM90 required, got SM{get_arch_major()}0")

        shape = SHAPES[cfg["shape"]]
        m = cfg["m"]
        cap = cfg["cap"]
        hidden = shape["hidden"]
        intermediate = shape["intermediate_hidden"]
        num_experts = shape["num_experts"]
        num_topk = shape["num_topk"]
        experts_per_rank = num_experts // num_ranks
        assert num_experts % num_ranks == 0 and m <= cap

        torch.manual_seed(cfg["seed"] + rank)
        random.seed(cfg["seed"] + rank)
        x_bf16 = torch.randn((m, hidden), dtype=torch.bfloat16, device="cuda")
        l1_bf16 = (
            torch.randn(
                (experts_per_rank, intermediate * 2, hidden),
                dtype=torch.bfloat16,
                device="cuda",
            )
            * 0.05
        )
        l2_bf16 = (
            torch.randn(
                (experts_per_rank, hidden, intermediate),
                dtype=torch.bfloat16,
                device="cuda",
            )
            * 0.05
        )
        scores = torch.randn((m, num_experts), dtype=torch.float, device="cuda")
        topk_weights, topk_idx = torch.topk(
            scores, num_topk, dim=-1, largest=True, sorted=False
        )
        topk_idx = topk_idx.to(torch.int64)
        expected_routes = _expected_local_routes(
            topk_idx, num_experts, rank, num_ranks, group
        )

        l1_weights = quantize_grouped_fp8_block_128_128(l1_bf16)
        l2_weights = quantize_grouped_fp8_block_128_128(l2_bf16)
        del l1_bf16, l2_bf16, scores
        torch.cuda.empty_cache()

        num_rdma_bytes = _deep_ep.Buffer.get_low_latency_rdma_size_hint(
            cap, hidden, num_ranks, num_experts
        )
        buffer = _deep_ep.Buffer(
            group,
            num_nvl_bytes=0,
            num_rdma_bytes=num_rdma_bytes,
            low_latency_mode=True,
            num_qps_per_rank=experts_per_rank,
            allow_nvlink_for_low_latency_mode=True,
            explicitly_destroy=True,
        )
        buffer.clean_low_latency_buffer(cap, hidden, num_experts)
        torch.cuda.synchronize()
        dist.barrier()

        max_m = cap * num_ranks
        expected_m = max(
            1,
            (cap * num_ranks * num_topk + num_experts - 1) // num_experts,
        )
        l1_y = torch.empty(
            (experts_per_rank, max_m, intermediate * 2),
            dtype=torch.bfloat16,
            device="cuda",
        )
        l2_y = torch.empty(
            (experts_per_rank, max_m, hidden),
            dtype=torch.bfloat16,
            device="cuda",
        )
        combined = torch.empty((m, hidden), dtype=torch.bfloat16, device="cuda")
        clamp = cfg["activation_clamp"]
        clamp_arg = clamp if math.isfinite(clamp) else None
        last_masked_m = [None]

        def run_low_latency():
            (recv_x_data, recv_x_sf), masked_m, handle, _, _ = (
                buffer.low_latency_dispatch(
                    x_bf16,
                    topk_idx,
                    cap,
                    num_experts,
                    use_fp8=True,
                    round_scale=False,
                    use_ue8m0=False,
                    async_finish=False,
                    return_recv_hook=False,
                )
            )
            last_masked_m[0] = masked_m
            deep_gemm.m_grouped_fp8_gemm_nt_masked(
                (recv_x_data, recv_x_sf),
                l1_weights,
                l1_y,
                masked_m,
                expected_m,
                disable_ue8m0_cast=True,
            )
            l2_x = swiglu_masked_post_quant_to_fp8(
                l1_y, masked_m, clamp_arg
            )
            deep_gemm.m_grouped_fp8_gemm_nt_masked(
                l2_x,
                l2_weights,
                l2_y,
                masked_m,
                expected_m,
                disable_ue8m0_cast=True,
            )
            output, _, _ = buffer.low_latency_combine(
                l2_y,
                topk_idx,
                topk_weights,
                handle,
                use_logfmt=False,
                zero_copy=False,
                async_finish=False,
                return_recv_hook=False,
                out=combined,
            )
            return output

        output = run_low_latency()
        torch.cuda.synchronize()
        dist.barrier()
        if output.shape != (m, hidden) or output.dtype != torch.bfloat16:
            raise RuntimeError(
                f"invalid output shape/dtype: {tuple(output.shape)}, {output.dtype}"
            )
        if not bool(torch.isfinite(output).all().item()):
            raise RuntimeError("low-latency output contains non-finite values")
        route_counts = last_masked_m[0].to(torch.int).clone()
        if not torch.equal(route_counts, expected_routes):
            raise RuntimeError(
                f"route mismatch: actual={route_counts.tolist()}, "
                f"expected={expected_routes.tolist()}"
            )

        free_bytes, _ = torch.cuda.mem_get_info()
        requested_flush = max(0, int(cfg["flush_l2_bytes"]))
        actual_flush = min(requested_flush, int(free_bytes * 0.5))
        actual_flush -= actual_flush % 4
        flush = (
            torch.empty(actual_flush // 4, dtype=torch.int, device="cuda")
            if actual_flush
            else None
        )

        route_list = route_counts.tolist()
        for observation in range(1, cfg["observations"] + 1):
            for _ in range(cfg["warmups"]):
                run_low_latency()
            torch.cuda.synchronize()
            dist.barrier()

            samples_us = []
            for _ in range(cfg["samples"]):
                if flush is not None:
                    flush.zero_()
                torch.cuda.synchronize()
                dist.barrier()
                start = torch.cuda.Event(enable_timing=True)
                end = torch.cuda.Event(enable_timing=True)
                start.record()
                run_low_latency()
                end.record()
                end.synchronize()
                samples_us.append(start.elapsed_time(end) * 1000.0)

            observed_routes = last_masked_m[0].to(torch.int).tolist()
            if observed_routes != route_list:
                raise RuntimeError("low-latency routes changed across calls")
            returned_us = statistics.median(samples_us)
            local_row = {
                "shape": cfg["shape"],
                "m": m,
                "cap": cap,
                "seed": cfg["seed"],
                "observation": observation,
                "rank": rank,
                "num_samples": len(samples_us),
                "returned_us": returned_us,
                "mean_us": statistics.mean(samples_us),
                "min_us": min(samples_us),
                "max_us": max(samples_us),
                "samples_us": samples_us,
                "max_m": max_m,
                "expected_m": expected_m,
                "route_counts": route_list,
                "expected_route_counts": expected_routes.tolist(),
                "route_total": sum(route_list),
                "touched_experts": sum(value > 0 for value in route_list),
                "requested_flush_l2_bytes": requested_flush,
                "actual_flush_l2_bytes": actual_flush,
            }
            print(
                "LOW_LATENCY_STAT_JSON " + json.dumps(local_row, sort_keys=True),
                flush=True,
            )

            gathered = [None] * num_ranks
            dist.all_gather_object(gathered, local_row)
            if rank == 0:
                max_row = max(gathered, key=lambda row: row["returned_us"])
                aggregate = {
                    "shape": cfg["shape"],
                    "m": m,
                    "seed": cfg["seed"],
                    "observation": observation,
                    "num_samples": cfg["samples"],
                    "max_rank": max_row["rank"],
                    "max_rank_us": max_row["returned_us"],
                    "per_rank_us": [
                        row["returned_us"]
                        for row in sorted(gathered, key=lambda row: row["rank"])
                    ],
                    "routes": [
                        {
                            "rank": row["rank"],
                            "route_counts": row["route_counts"],
                            "route_total": row["route_total"],
                            "touched_experts": row["touched_experts"],
                        }
                        for row in sorted(gathered, key=lambda row: row["rank"])
                    ],
                    "requested_flush_l2_bytes": requested_flush,
                    "actual_flush_l2_bytes_min": min(
                        row["actual_flush_l2_bytes"] for row in gathered
                    ),
                }
                print(
                    "LOW_LATENCY_OBSERVATION_JSON "
                    + json.dumps(aggregate, sort_keys=True),
                    flush=True,
                )
            dist.barrier()

        del flush
        completed = True
    finally:
        if buffer is not None:
            buffer.destroy()
        if dist.is_initialized():
            if completed:
                dist.barrier()
            dist.destroy_process_group()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--shape", required=True, choices=tuple(SHAPES))
    parser.add_argument("--m", required=True, type=int)
    parser.add_argument("--cap", required=True, type=int)
    parser.add_argument("--seed", type=int, default=101)
    parser.add_argument("--num-processes", type=int, default=8)
    parser.add_argument("--observations", type=int, default=3)
    parser.add_argument("--warmups", type=int, default=5)
    parser.add_argument("--samples", type=int, default=20)
    parser.add_argument("--flush-l2-bytes", type=int, default=8_000_000_000)
    parser.add_argument("--activation-clamp", type=float, default=10.0)
    args = parser.parse_args()
    if args.cap < args.m:
        parser.error("--cap must be at least --m")
    cfg = vars(args).copy()
    torch.multiprocessing.spawn(
        _worker,
        args=(args.num_processes, cfg),
        nprocs=args.num_processes,
    )


if __name__ == "__main__":
    main()
