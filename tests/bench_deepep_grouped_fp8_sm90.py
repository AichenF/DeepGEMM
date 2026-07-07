#!/usr/bin/env python3
"""Baseline-only SM90 MoE benchmark.

The timed path is exactly:

    DeepEP dispatch
      -> contiguous grouped-FP8 L1
      -> Triton SwiGLU, top-k weighting, and E4M3 quantization
      -> contiguous grouped-FP8 L2
      -> DeepEP combine

This file intentionally contains no fused MoE setup, warmup, or invocation.
"""

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
from deep_gemm.utils import per_token_cast_to_fp8
from deep_gemm.utils.dist import init_dist, uneven_all_gather
from deep_gemm.testing import get_arch_major


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


@triton.jit
def _swiglu_apply_weight_to_fp8_kernel(
    x_ptr,
    topk_w_ptr,
    y_ptr,
    y_sf_ptr,
    M,
    H,
    stride_xm,
    stride_xn,
    stride_ym,
    stride_yn,
    stride_sfm,
    stride_sfk,
    clamp_value,
    HAS_CLAMP: tl.constexpr,
    USE_UE8M0_SCALE: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_k = tl.program_id(1)
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_k = pid_k * BLOCK_K + tl.arange(0, BLOCK_K)
    mask_m = offs_m < M

    gate_ptrs = x_ptr + offs_m[:, None] * stride_xm + offs_k[None, :] * stride_xn
    up_ptrs = (
        x_ptr
        + offs_m[:, None] * stride_xm
        + (H + offs_k[None, :]) * stride_xn
    )
    gate = tl.load(gate_ptrs, mask=mask_m[:, None], other=0.0).to(tl.float32)
    up = tl.load(up_ptrs, mask=mask_m[:, None], other=0.0).to(tl.float32)

    if HAS_CLAMP:
        gate = tl.minimum(gate, clamp_value)
        up = tl.minimum(tl.maximum(up, -clamp_value), clamp_value)

    y = gate * tl.sigmoid(gate) * up
    topk_w = tl.load(topk_w_ptr + offs_m, mask=mask_m, other=1.0)
    y *= topk_w[:, None]

    amax = tl.max(tl.abs(y), axis=1)
    sf = tl.maximum(amax / _FP8_E4M3_MAX_TL, 1.0e-30)
    if USE_UE8M0_SCALE:
        sf = tl.exp2(tl.ceil(tl.log2(sf)))
    y_fp8 = (y / sf[:, None]).to(tl.float8e4nv)

    y_ptrs = y_ptr + offs_m[:, None] * stride_ym + offs_k[None, :] * stride_yn
    tl.store(y_ptrs, y_fp8, mask=mask_m[:, None])
    sf_ptrs = y_sf_ptr + offs_m * stride_sfm + pid_k * stride_sfk
    tl.store(sf_ptrs, sf, mask=mask_m)


def swiglu_apply_weight_to_fp8(
    x: torch.Tensor,
    topk_weights: torch.Tensor,
    clamp_value: float | None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    assert x.is_cuda and x.dtype == torch.bfloat16 and x.is_contiguous()
    m, two_h = x.shape
    h = two_h // 2
    assert two_h == h * 2 and h % ACT_SF_GRAN == 0
    topk_weights = topk_weights.reshape(-1)
    if topk_weights.numel() != m:
        raise RuntimeError(
            f"SwiGLU weight rows mismatch: x_rows={m}, "
            f"topk_weights={topk_weights.numel()}"
        )

    y = torch.empty((m, h), dtype=torch.float8_e4m3fn, device=x.device)
    y_sf = torch.empty((m, h // ACT_SF_GRAN), dtype=torch.float32, device=x.device)
    block_m = 16
    grid = (triton.cdiv(m, block_m), h // ACT_SF_GRAN)
    _swiglu_apply_weight_to_fp8_kernel[grid](
        x,
        topk_weights,
        y,
        y_sf,
        m,
        h,
        x.stride(0),
        x.stride(1),
        y.stride(0),
        y.stride(1),
        y_sf.stride(0),
        y_sf.stride(1),
        float(clamp_value) if clamp_value is not None else 0.0,
        HAS_CLAMP=clamp_value is not None,
        USE_UE8M0_SCALE=True,
        BLOCK_M=block_m,
        BLOCK_K=ACT_SF_GRAN,
    )
    return y, y_sf


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
        view = w[start:end].view(end - start, n // 128, 128, k // 128, 128).float()
        sf_chunk = view.abs().amax(dim=(-1, -3)).clamp(1e-4) / FP8_E4M3_MAX
        w_fp8[start:end].copy_(
            (view / sf_chunk.unsqueeze(-1).unsqueeze(-3))
            .to(torch.float8_e4m3fn)
            .view(end - start, n, k)
        )
        sf[start:end].copy_(sf_chunk)
    return w_fp8, sf.contiguous()


class _DeepEPHandle:
    def __init__(self, raw_handle, psum_num_recv_tokens_per_expert: torch.Tensor):
        self.raw_handle = raw_handle
        self.psum_num_recv_tokens_per_expert = psum_num_recv_tokens_per_expert


class _DeepEPBufferCompat:
    """Compatibility shim for DeepEP versions exposing Buffer only."""

    def __init__(self, deep_ep, group, num_nvl_bytes: int):
        self.buffer = deep_ep.Buffer(
            group,
            num_nvl_bytes=num_nvl_bytes,
            num_rdma_bytes=0,
            explicitly_destroy=True,
        )

    def dispatch(
        self,
        x,
        *,
        topk_idx,
        topk_weights,
        num_experts: int,
        expert_alignment: int,
        **_,
    ):
        (
            num_tokens_per_rank,
            num_tokens_per_rdma_rank,
            num_tokens_per_expert,
            is_token_in_rank,
            event,
        ) = self.buffer.get_dispatch_layout(topk_idx, num_experts)
        (
            recv_x,
            recv_topk_idx,
            recv_topk_weights,
            num_recv_tokens_per_expert,
            raw_handle,
            event,
        ) = self.buffer.dispatch(
            x,
            num_tokens_per_rank=num_tokens_per_rank,
            num_tokens_per_rdma_rank=num_tokens_per_rdma_rank,
            is_token_in_rank=is_token_in_rank,
            num_tokens_per_expert=num_tokens_per_expert,
            topk_idx=topk_idx,
            topk_weights=topk_weights,
            expert_alignment=expert_alignment,
        )
        if os.environ.get("DG_BASELINE_DEBUG_SHAPES") == "1":
            print(
                "LEGACY_DISPATCH_SHAPE_JSON "
                + json.dumps(
                    {
                        "rank": dist.get_rank(),
                        "recv_x": [list(t.shape) for t in recv_x],
                        "recv_topk_idx": list(recv_topk_idx.shape),
                        "recv_topk_weights": list(recv_topk_weights.shape),
                        "recv_per_expert": list(num_recv_tokens_per_expert),
                    },
                    sort_keys=True,
                ),
                flush=True,
            )
        psum = torch.tensor(
            num_recv_tokens_per_expert,
            dtype=torch.int,
            device=topk_idx.device,
        ).cumsum(dim=0, dtype=torch.int)
        return recv_x, None, recv_topk_weights, _DeepEPHandle(raw_handle, psum), event

    def combine(self, x, *, handle):
        raw_handle = handle.raw_handle if isinstance(handle, _DeepEPHandle) else handle
        return self.buffer.combine(x, handle=raw_handle)

    def barrier(self):
        torch.cuda.synchronize()
        dist.barrier()

    def destroy(self):
        self.buffer.destroy()


def _make_deep_ep_buffer(
    deep_ep,
    group,
    num_ranks: int,
    num_experts: int,
    cap: int,
    hidden: int,
    intermediate_hidden: int,
    num_topk: int,
    buffer_api: str,
):
    if buffer_api == "elastic":
        if not hasattr(deep_ep, "ElasticBuffer"):
            raise RuntimeError("DeepEP ElasticBuffer was requested but is unavailable")
        return deep_ep.ElasticBuffer(
            group,
            num_max_tokens_per_rank=cap,
            hidden=hidden,
            num_topk=num_topk,
            use_fp8_dispatch=True,
            explicitly_destroy=True,
            allow_multiple_reduction=False,
            num_gpu_timeout_secs=10,
            num_cpu_timeout_secs=30,
        )

    if buffer_api != "legacy":
        raise ValueError(f"unsupported DeepEP buffer API: {buffer_api}")

    # Legacy Buffer is the normal single-node NVLink path and does not require
    # the NCCL GIN transport used by ElasticBuffer.
    num_bytes, _ = deep_gemm._C.get_symm_buffer_size_for_mega_moe(
        num_ranks,
        num_experts,
        cap,
        num_topk,
        hidden,
        intermediate_hidden,
        True,
        "swiglu",
    )
    alignment = 2 * 1024 * 1024
    num_nvl_bytes = ((int(num_bytes) + alignment - 1) // alignment) * alignment
    return _DeepEPBufferCompat(deep_ep, group, num_nvl_bytes)


def _barrier(ep_buffer):
    if isinstance(ep_buffer, _DeepEPBufferCompat):
        ep_buffer.barrier()
    else:
        ep_buffer.barrier(use_comm_stream=False)


def _expected_route_stats(topk_idx, num_experts, rank, num_ranks, group):
    gathered = uneven_all_gather(topk_idx, group=group)
    experts_per_rank = num_experts // num_ranks
    local = gathered[
        (gathered >= rank * experts_per_rank)
        & (gathered < (rank + 1) * experts_per_rank)
    ]
    return int(local.numel()), int(torch.unique(local).numel())


def _worker(local_rank: int, num_processes: int, cfg: dict):
    if _deep_ep is None:
        raise RuntimeError(f"DeepEP import failed: {_DEEP_EP_IMPORT_ERROR}")

    rank, num_ranks, group = init_dist(local_rank, num_processes)
    shape = SHAPES[cfg["shape"]]
    m = cfg["m"]
    cap = cfg["cap"]
    hidden = shape["hidden"]
    intermediate_hidden = shape["intermediate_hidden"]
    num_experts = shape["num_experts"]
    num_topk = shape["num_topk"]
    experts_per_rank = num_experts // num_ranks

    try:
        if get_arch_major() != 9:
            raise RuntimeError(f"SM90 required, got SM{get_arch_major()}0")
        assert num_experts % num_ranks == 0 and m <= cap

        # Match the established H200 matrix runner: each point consumes seed+rank
        # from a fresh RNG state and creates tensors in x/W1/W2/scores order.
        torch.manual_seed(cfg["seed"] + rank)
        random.seed(cfg["seed"] + rank)
        x_bf16 = torch.randn((m, hidden), dtype=torch.bfloat16, device="cuda")
        l1_bf16 = (
            torch.randn(
                (experts_per_rank, intermediate_hidden * 2, hidden),
                dtype=torch.bfloat16,
                device="cuda",
            )
            * 0.05
        )
        l2_bf16 = (
            torch.randn(
                (experts_per_rank, hidden, intermediate_hidden),
                dtype=torch.bfloat16,
                device="cuda",
            )
            * 0.05
        )
        scores = torch.randn((m, num_experts), dtype=torch.float, device="cuda")
        topk_weights, topk_idx = torch.topk(
            scores, num_topk, dim=-1, largest=True, sorted=False
        )

        x_fp8 = per_token_cast_to_fp8(
            x_bf16,
            use_ue8m0=False,
            gran_k=128,
            use_packed_ue8m0=False,
        )
        l1_weights = quantize_grouped_fp8_block_128_128(l1_bf16)
        l2_weights = quantize_grouped_fp8_block_128_128(l2_bf16)
        del x_bf16, l1_bf16, l2_bf16, scores
        torch.cuda.empty_cache()

        alignment = deep_gemm.get_theoretical_mk_alignment_for_contiguous_layout()
        deep_gemm.set_mk_alignment_for_contiguous_layout(alignment)
        ep_buffer = _make_deep_ep_buffer(
            _deep_ep,
            group,
            num_ranks,
            num_experts,
            cap,
            hidden,
            intermediate_hidden,
            num_topk,
            cfg["deepep_buffer"],
        )
        cumulative_recv = torch.zeros(
            (experts_per_rank,), dtype=torch.int, device="cuda"
        )
        clamp = cfg["activation_clamp"]
        clamp_arg = clamp if math.isfinite(clamp) else None
        last_buffer_rows = [0]
        last_handle = [None]

        def run_baseline():
            recv_x, _, recv_topk_weights, handle, _ = ep_buffer.dispatch(
                x_fp8,
                topk_idx=topk_idx,
                topk_weights=topk_weights,
                cumulative_local_expert_recv_stats=cumulative_recv,
                num_experts=num_experts,
                expert_alignment=alignment,
                do_cpu_sync=bool(cfg["do_cpu_sync"]),
                do_handle_copy=False,
                do_expand=True,
                use_tma_aligned_col_major_sf=False,
            )
            n = recv_x[0].size(0)
            last_buffer_rows[0] = n
            last_handle[0] = handle
            l1_y = torch.empty(
                (n, intermediate_hidden * 2),
                dtype=torch.bfloat16,
                device="cuda",
            )
            deep_gemm.m_grouped_fp8_gemm_nt_contiguous(
                recv_x,
                l1_weights,
                l1_y,
                handle.psum_num_recv_tokens_per_expert,
                use_psum_layout=True,
                disable_ue8m0_cast=True,
            )
            l2_x = swiglu_apply_weight_to_fp8(
                l1_y,
                recv_topk_weights,
                clamp_arg,
            )
            l2_y = torch.empty((n, hidden), dtype=torch.bfloat16, device="cuda")
            deep_gemm.m_grouped_fp8_gemm_nt_contiguous(
                l2_x,
                l2_weights,
                l2_y,
                handle.psum_num_recv_tokens_per_expert,
                use_psum_layout=True,
                disable_ue8m0_cast=True,
            )
            return ep_buffer.combine(l2_y, handle=handle)[0]

        expected_recv, touched_experts = _expected_route_stats(
            topk_idx, num_experts, rank, num_ranks, group
        )
        y = run_baseline()
        torch.cuda.synchronize()
        _barrier(ep_buffer)
        expert_ends = [
            int(value)
            for value in last_handle[0].psum_num_recv_tokens_per_expert.tolist()
        ]
        last_expert_end = expert_ends[-1]
        actual_expert_tokens = 0
        expert_start = 0
        for expert_end in expert_ends:
            actual_expert_tokens += expert_end - expert_start
            expert_start = (
                (expert_end + alignment - 1) // alignment * alignment
            )
        padded_expert_rows = sum(last_handle[0].num_recv_tokens_per_expert_list)
        if last_expert_end > last_buffer_rows[0]:
            raise RuntimeError(
                f"expanded rows exceed buffer: {last_expert_end} > {last_buffer_rows[0]}"
            )
        if y.shape != (m, hidden) or y.dtype != torch.bfloat16:
            raise RuntimeError(
                f"invalid output shape/dtype: shape={tuple(y.shape)}, dtype={y.dtype}"
            )
        if not bool(torch.isfinite(y).all().item()):
            raise RuntimeError("baseline output contains non-finite values")

        free_bytes, _ = torch.cuda.mem_get_info()
        requested_flush = max(0, int(cfg["flush_l2_bytes"]))
        actual_flush = min(requested_flush, int(free_bytes * 0.5))
        actual_flush -= actual_flush % 4
        flush = (
            torch.empty(actual_flush // 4, dtype=torch.int, device="cuda")
            if actual_flush
            else None
        )

        for observation in range(1, cfg["observations"] + 1):
            for _ in range(cfg["warmups"]):
                run_baseline()
            torch.cuda.synchronize()
            _barrier(ep_buffer)

            samples_us = []
            for _ in range(cfg["samples"]):
                if flush is not None:
                    flush.zero_()
                torch.cuda.synchronize()
                _barrier(ep_buffer)
                start = torch.cuda.Event(enable_timing=True)
                end = torch.cuda.Event(enable_timing=True)
                start.record()
                run_baseline()
                end.record()
                end.synchronize()
                samples_us.append(start.elapsed_time(end) * 1000.0)

            returned_us = statistics.median(samples_us)
            local_row = {
                "shape": cfg["shape"],
                "m": m,
                "seed": cfg["seed"],
                "observation": observation,
                "rank": rank,
                "num_samples": len(samples_us),
                "returned_us": returned_us,
                "mean_us": statistics.mean(samples_us),
                "min_us": min(samples_us),
                "max_us": max(samples_us),
                "samples_us": samples_us,
                "cap": cap,
                "do_cpu_sync": bool(cfg["do_cpu_sync"]),
                "deepep_buffer": cfg["deepep_buffer"],
                "expected_recv_tokens": expected_recv,
                "actual_expert_tokens": actual_expert_tokens,
                "padded_expert_rows": padded_expert_rows,
                "last_expert_end": last_expert_end,
                "expanded_buffer_rows": last_buffer_rows[0],
                "expert_alignment": alignment,
                "touched_experts": touched_experts,
                "requested_flush_l2_bytes": requested_flush,
                "actual_flush_l2_bytes": actual_flush,
            }
            print("BASELINE_STAT_JSON " + json.dumps(local_row, sort_keys=True), flush=True)

            gathered_rows = [None] * num_ranks
            dist.all_gather_object(gathered_rows, local_row)
            if rank == 0:
                max_row = max(gathered_rows, key=lambda row: row["returned_us"])
                print(
                    "BASELINE_OBSERVATION_JSON "
                    + json.dumps(
                        {
                            "shape": cfg["shape"],
                            "m": m,
                            "seed": cfg["seed"],
                            "observation": observation,
                            "num_samples": cfg["samples"],
                            "max_rank": max_row["rank"],
                            "max_rank_us": max_row["returned_us"],
                            "per_rank_us": [
                                row["returned_us"]
                                for row in sorted(gathered_rows, key=lambda row: row["rank"])
                            ],
                            "routes": [
                                {
                                    "rank": row["rank"],
                                    "expected_recv_tokens": row["expected_recv_tokens"],
                                    "actual_expert_tokens": row["actual_expert_tokens"],
                                    "padded_expert_rows": row["padded_expert_rows"],
                                    "last_expert_end": row["last_expert_end"],
                                    "expanded_buffer_rows": row["expanded_buffer_rows"],
                                    "expert_alignment": row["expert_alignment"],
                                    "touched_experts": row["touched_experts"],
                                }
                                for row in sorted(gathered_rows, key=lambda row: row["rank"])
                            ],
                            "requested_flush_l2_bytes": requested_flush,
                            "actual_flush_l2_bytes_min": min(
                                row["actual_flush_l2_bytes"] for row in gathered_rows
                            ),
                        },
                        sort_keys=True,
                    ),
                    flush=True,
                )
            dist.barrier()

        del flush
        ep_buffer.destroy()
        dist.barrier()
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--shape", required=True, choices=tuple(SHAPES))
    parser.add_argument("--m", required=True, type=int)
    parser.add_argument("--cap", type=int, default=8192)
    parser.add_argument("--seed", type=int, default=101)
    parser.add_argument("--num-processes", type=int, default=8)
    parser.add_argument("--observations", type=int, default=3)
    parser.add_argument("--warmups", type=int, default=5)
    parser.add_argument("--samples", type=int, default=20)
    parser.add_argument("--flush-l2-bytes", type=int, default=8_000_000_000)
    parser.add_argument(
        "--deepep-buffer",
        choices=("legacy", "elastic"),
        default="elastic",
        help="DeepEP transport API; legacy is the intranode NVLink path",
    )
    parser.add_argument(
        "--do-cpu-sync",
        type=int,
        choices=(0, 1),
        default=1,
        help="request exact dynamic receive extents from DeepEP dispatch",
    )
    parser.add_argument("--activation-clamp", type=float, default=10.0)
    args = parser.parse_args()
    if args.observations <= 0 or args.warmups < 0 or args.samples <= 0:
        parser.error("observations/samples must be positive and warmups non-negative")

    torch.multiprocessing.spawn(
        _worker,
        args=(args.num_processes, vars(args)),
        nprocs=args.num_processes,
    )


if __name__ == "__main__":
    main()
