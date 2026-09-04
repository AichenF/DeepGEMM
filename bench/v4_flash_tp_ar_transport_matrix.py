#!/usr/bin/env python3
"""Cold-L2 TP4 matrix for all-reduce algorithm and transport choices.

The benchmark deliberately has two scopes:

* ``ar`` restores one random nonzero BF16 ``[M, 4096]`` input per rank,
  evicts L2, and times only one captured all-reduce.  The restore and cache
  clear are outside the CUDA-event interval.  This isolates the collective.
* ``full`` captures the unchanged local TP-MoE pipeline followed by the same
  five collective variants.  This measures whether a transport win survives
  in the real graph without mixing in W2 chunking or producer overlap.

The five variants separate algorithm from transport:

* p2p_1shot_push: generic CARv2, ordinary peer stores plus local polling.
* p2p_1shot_pull: generic CARv2, graph-mapped ordinary peer loads.
* p2p_2shot_pull: generic CARv2 reduce-scatter/all-gather with peer pointers.
* nvls_1shot_push: K3 one-shot push using one ``multimem.st`` per vector.
* nvls_2shot_pull: K3 direct-symmetric two-shot using
  ``multimem.ld_reduce`` plus ``multimem.st``.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import statistics
from dataclasses import dataclass
from typing import Callable

import torch
import torch.distributed as dist
from triton import runtime as triton_runtime

import sglang.srt.distributed.parallel_state as ps
from sglang.kernels.ops.communication.all_reduce import AllReduceAlgo
try:
    from sglang.jit_kernel.mp import register_comm_cleanup
except ImportError:  # Compatibility with the older benchmark checkout.
    from sglang.kernels.ops.communication.mp import register_comm_cleanup
from sglang.kernels.ops.kimi_k3.all_reduce import (
    all_reduce_pull_res,
    all_reduce_push_res,
    register_comm,
)
from sglang.srt.distributed.device_communicators.custom_all_reduce_v2 import (
    CustomAllReduceV2,
)

import v4_flash_tp_wgmma as kernel
import v4_flash_tp_wgmma_graph as custom


VARIANTS = (
    "p2p_1shot_push",
    "p2p_1shot_pull",
    "p2p_2shot_pull",
    "nvls_1shot_push",
    "nvls_2shot_pull",
)
P2P_ALGOS = {
    "p2p_1shot_push": AllReduceAlgo.ONE_SHOT_PUSH,
    "p2p_1shot_pull": AllReduceAlgo.ONE_SHOT_PULL,
    "p2p_2shot_pull": AllReduceAlgo.TWO_SHOT_PULL,
}
VARIANT_METADATA = {
    "p2p_1shot_push": {
        "algorithm": "one_shot_push",
        "transport": "ordinary_peer_global_store_load",
        "input": "ordinary_cuda_tensor",
    },
    "p2p_1shot_pull": {
        "algorithm": "one_shot_pull",
        "transport": "ordinary_graph_mapped_peer_global_load",
        "input": "ordinary_cuda_tensor",
    },
    "p2p_2shot_pull": {
        "algorithm": "two_shot_pull",
        "transport": "ordinary_graph_mapped_peer_global_load_store",
        "input": "ordinary_cuda_tensor",
    },
    "nvls_1shot_push": {
        "algorithm": "one_shot_push",
        "transport": "nvls_multimem_store_then_local_poll",
        "input": "ordinary_cuda_tensor",
    },
    "nvls_2shot_pull": {
        "algorithm": "two_shot_pull",
        "transport": "nvls_multimem_load_reduce_and_store",
        "input": "multicast_bound_symmetric_tensor",
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ms", default="64,128")
    parser.add_argument("--scopes", default="ar,full")
    parser.add_argument("--variants", default=",".join(VARIANTS))
    parser.add_argument(
        "--route-pattern", choices=("random", "balanced", "skew"), default="random"
    )
    parser.add_argument("--outer", type=int, default=10)
    parser.add_argument("--replays", type=int, default=200)
    parser.add_argument("--warmup-replays", type=int, default=10)
    parser.add_argument(
        "--pair-granularity", choices=("batch", "replay"), default="batch"
    )
    parser.add_argument("--p2p-pull-blocks", type=int, default=64)
    parser.add_argument("--nvls-pull-blocks", type=int, default=16)
    parser.add_argument("--nvls-pull-unroll", type=int, choices=(2, 4, 8, 16), default=8)
    parser.add_argument("--seed", type=int, default=20260902)
    args = parser.parse_args()
    args.ms = tuple(int(value) for value in args.ms.split(",") if value)
    args.scopes = tuple(value for value in args.scopes.split(",") if value)
    args.variants = tuple(value for value in args.variants.split(",") if value)
    if not args.ms or any(value <= 0 for value in args.ms):
        parser.error("--ms must contain positive integers")
    if not args.scopes or any(value not in ("ar", "full") for value in args.scopes):
        parser.error("--scopes must contain ar and/or full")
    if not args.variants or any(value not in VARIANTS for value in args.variants):
        parser.error(f"--variants must come from {VARIANTS}")
    if "p2p_2shot_pull" not in args.variants:
        parser.error("p2p_2shot_pull is the required control")
    if args.outer < 2 or args.outer % 2:
        parser.error("--outer must be positive and even")
    if args.replays < 1 or args.warmup_replays < 1:
        parser.error("replay counts must be positive")
    if not 1 <= args.p2p_pull_blocks <= 64:
        parser.error("--p2p-pull-blocks must be in [1, 64]")
    if not 1 <= args.nvls_pull_blocks <= 64:
        parser.error("--nvls-pull-blocks must be in [1, 64]")
    return args


def percentile(values: list[float], q: float) -> float:
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    position = q * (len(ordered) - 1)
    lo = math.floor(position)
    hi = math.ceil(position)
    if lo == hi:
        return ordered[lo]
    alpha = position - lo
    return ordered[lo] * (1.0 - alpha) + ordered[hi] * alpha


def reduce_rank_metric(
    value: float, op: dist.ReduceOp, device: torch.device, group: dist.ProcessGroup
) -> float:
    tensor = torch.tensor(value, dtype=torch.float64, device=device)
    dist.all_reduce(tensor, op=op, group=group)
    return float(tensor.item())


def correctness_metrics(
    actual: torch.Tensor,
    reference: torch.Tensor,
    device: torch.device,
    group: dist.ProcessGroup,
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
    max_abs = float(diff.abs().max().item())
    finite = float(bool(torch.isfinite(actual).all()))
    cosine_min = reduce_rank_metric(cosine, dist.ReduceOp.MIN, device, group)
    rel_l2_max = reduce_rank_metric(rel_l2, dist.ReduceOp.MAX, device, group)
    max_abs_max = reduce_rank_metric(max_abs, dist.ReduceOp.MAX, device, group)
    finite_all = bool(reduce_rank_metric(finite, dist.ReduceOp.MIN, device, group))
    return {
        "cosine_min_rank": cosine_min,
        "rel_l2_max_rank": rel_l2_max,
        "max_abs_max_rank": max_abs_max,
        "finite_all_ranks": finite_all,
        "allreduce_ok": bool(finite_all and cosine_min >= 0.999 and rel_l2_max <= 0.02),
    }


def run_p2p(
    comm: CustomAllReduceV2, value: torch.Tensor, algo: AllReduceAlgo
) -> torch.Tensor:
    previous = comm.override_algo
    comm.override_algo = algo
    try:
        return comm.custom_all_reduce(value)
    finally:
        comm.override_algo = previous


def run_variant(
    name: str,
    comm: CustomAllReduceV2,
    value: torch.Tensor,
    value_mc_ptr: int,
    nvls_pull_blocks: int,
    nvls_pull_unroll: int,
) -> torch.Tensor:
    if name in P2P_ALGOS:
        return run_p2p(comm, value, P2P_ALGOS[name])
    if name == "nvls_1shot_push":
        return all_reduce_push_res(
            comm.world_size, value, ws_mc_base=comm.mc_base_ptr
        )
    if name == "nvls_2shot_pull":
        if value_mc_ptr == 0:
            raise RuntimeError("NVLS two-shot requires a multicast input pointer")
        return all_reduce_pull_res(
            comm.world_size,
            value,
            input_mc_ptr=value_mc_ptr,
            num_blocks=nvls_pull_blocks,
            unroll=nvls_pull_unroll,
        )
    raise AssertionError(name)


@dataclass
class ARCase:
    name: str
    source: torch.Tensor
    work: torch.Tensor
    work_mc_ptr: int
    nvls_pull_blocks: int
    nvls_pull_unroll: int
    graph_output: torch.Tensor | None = None

    def reset(self) -> None:
        self.work.copy_(self.source)

    def run_full(self, comm: CustomAllReduceV2) -> torch.Tensor:
        self.graph_output = run_variant(
            self.name,
            comm,
            self.work,
            self.work_mc_ptr,
            self.nvls_pull_blocks,
            self.nvls_pull_unroll,
        )
        return self.graph_output


@dataclass
class FullCase:
    name: str
    base: custom.CapturedCase
    nvls_pull_blocks: int
    nvls_pull_unroll: int
    graph_output: torch.Tensor | None = None

    def run_full(self, comm: CustomAllReduceV2) -> torch.Tensor:
        if self.name == "nvls_2shot_pull":
            self.base.prepare_fused_pull(comm)
            assert self.base.fused_pull_output is not None
            self.base.run_before_local_reduce()
            self.base.reduce_local_to(self.base.fused_pull_output)
            self.graph_output = run_variant(
                self.name,
                comm,
                self.base.fused_pull_output,
                self.base.fused_pull_mc_ptr,
                self.nvls_pull_blocks,
                self.nvls_pull_unroll,
            )
        else:
            local = self.base.run_local()
            self.graph_output = run_variant(
                self.name,
                comm,
                local,
                0,
                self.nvls_pull_blocks,
                self.nvls_pull_unroll,
            )
        return self.graph_output


def make_compute_case(
    m: int,
    x: torch.Tensor,
    topk_ids: torch.Tensor,
    topk_weights: torch.Tensor,
    weights: tuple[torch.Tensor, ...],
    lut: torch.Tensor,
    intermediate_per_rank: int,
) -> custom.CapturedCase:
    w13, s13, g13, w2, s2, g2 = weights
    return custom.CapturedCase(
        m=m,
        x=x,
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
    )


def symmetric_pull_view(
    comm: CustomAllReduceV2, m: int
) -> tuple[torch.Tensor, int, object]:
    from torch._C._distributed_c10d import _SymmetricMemory

    symm = _SymmetricMemory.rendezvous(comm._symm_tensor)
    total_bytes = comm._symm_tensor.numel()
    local_slab = symm.get_buffer(comm.rank, [total_bytes], torch.uint8)
    pull_offset = 2 * comm.world_size * comm.max_push_size
    nbytes = m * custom.HIDDEN * torch.bfloat16.itemsize
    if nbytes > comm.max_pull_size:
        raise RuntimeError("AR input exceeds symmetric pull workspace")
    view = local_slab[pull_offset : pull_offset + nbytes].view(torch.bfloat16)
    view = view.view(m, custom.HIDDEN)
    mc_ptr = int(symm.multicast_ptr) + pull_offset
    return view, mc_ptr, symm


def capture_case(
    case: ARCase | FullCase,
    comm: CustomAllReduceV2,
    cpu_group: dist.ProcessGroup,
    device: torch.device,
) -> torch.cuda.CUDAGraph:
    for _ in range(2):
        if isinstance(case, ARCase):
            case.reset()
        case.run_full(comm)
    torch.cuda.synchronize(device)
    dist.barrier(group=cpu_group)
    if isinstance(case, ARCase):
        case.reset()
        torch.cuda.synchronize(device)
    graph = torch.cuda.CUDAGraph()
    with comm.capture():
        with torch.cuda.graph(graph):
            case.run_full(comm)
    torch.cuda.synchronize(device)
    dist.barrier(group=cpu_group)
    return graph


def balanced_order(names: tuple[str, ...], outer_idx: int) -> tuple[str, ...]:
    shift = (outer_idx // 2) % len(names)
    rotated = names[shift:] + names[:shift]
    return tuple(reversed(rotated)) if outer_idx & 1 else rotated


def time_graph_matrix(
    graphs: dict[str, torch.cuda.CUDAGraph],
    resets: dict[str, Callable[[], None] | None],
    names: tuple[str, ...],
    outer: int,
    replays: int,
    cpu_group: dist.ProcessGroup,
    nccl_group: dist.ProcessGroup,
    device: torch.device,
    l2_flush_buffer: torch.Tensor,
    pair_granularity: str,
) -> tuple[dict[str, list[float]], dict[str, list[float]], list[list[str]]]:
    samples = {name: [] for name in names}
    batch_medians = {name: [] for name in names}
    orders: list[list[str]] = []
    driver = triton_runtime.driver.active

    for outer_idx in range(outer):
        dist.barrier(group=cpu_group)
        order = balanced_order(names, outer_idx)
        orders.append(list(order))
        events: dict[str, tuple[list[torch.cuda.Event], list[torch.cuda.Event]]] = {}
        for name in names:
            events[name] = (
                [torch.cuda.Event(enable_timing=True) for _ in range(replays)],
                [torch.cuda.Event(enable_timing=True) for _ in range(replays)],
            )

        def replay_one(name: str, replay_idx: int) -> None:
            starts, ends = events[name]
            reset = resets[name]
            if reset is not None:
                reset()
            driver.clear_cache(l2_flush_buffer)
            starts[replay_idx].record()
            graphs[name].replay()
            ends[replay_idx].record()

        if pair_granularity == "replay":
            for replay_idx in range(replays):
                replay_order = order if replay_idx % 2 == 0 else tuple(reversed(order))
                for name in replay_order:
                    replay_one(name, replay_idx)
        else:
            for name in order:
                for replay_idx in range(replays):
                    replay_one(name, replay_idx)
        torch.cuda.synchronize(device)

        for name in names:
            starts, ends = events[name]
            rank_times = torch.tensor(
                [start.elapsed_time(end) for start, end in zip(starts, ends)],
                dtype=torch.float64,
                device=device,
            )
            dist.all_reduce(rank_times, op=dist.ReduceOp.MAX, group=nccl_group)
            batch = [float(value) for value in rank_times.cpu().tolist()]
            samples[name].extend(batch)
            batch_medians[name].append(statistics.median(batch))

    return samples, batch_medians, orders


def summarize_variant(
    name: str,
    samples: list[float],
    batch_medians: list[float],
    control_batch_medians: list[float],
    correctness: dict[str, float | bool],
) -> dict[str, object]:
    median = statistics.median(samples)
    control_wins = sum(
        candidate < control
        for candidate, control in zip(batch_medians, control_batch_medians)
    )
    return {
        "variant": name,
        **VARIANT_METADATA[name],
        "cold_samples": len(samples),
        "latency_ms_min": min(samples),
        "latency_ms_p05": percentile(samples, 0.05),
        "latency_ms_median": median,
        "latency_ms_p95": percentile(samples, 0.95),
        "latency_ms_max": max(samples),
        "batch_medians_ms": batch_medians,
        "batch_wins_vs_p2p_2shot": control_wins,
        "batch_count": len(batch_medians),
        "correctness": correctness,
    }


def comparison_summary(records: dict[str, dict[str, object]]) -> dict[str, float]:
    medians = {
        name: float(record["latency_ms_median"]) for name, record in records.items()
    }
    result: dict[str, float] = {}
    required = {
        "speedup_nvls_1shot_over_p2p_1shot": (
            "p2p_1shot_push",
            "nvls_1shot_push",
        ),
        "speedup_p2p_2shot_over_p2p_1shot": (
            "p2p_1shot_push",
            "p2p_2shot_pull",
        ),
        "speedup_nvls_2shot_over_p2p_2shot": (
            "p2p_2shot_pull",
            "nvls_2shot_pull",
        ),
        "speedup_nvls_2shot_over_nvls_1shot": (
            "nvls_1shot_push",
            "nvls_2shot_pull",
        ),
        "speedup_p2p_1shot_pull_over_p2p_1shot_push": (
            "p2p_1shot_push",
            "p2p_1shot_pull",
        ),
    }
    for label, (baseline, candidate) in required.items():
        if baseline in medians and candidate in medians:
            result[label] = medians[baseline] / medians[candidate]
            result[label.replace("speedup", "delta_us")] = (
                medians[candidate] - medians[baseline]
            ) * 1000.0
    return result


def warm_graphs(
    graphs: dict[str, torch.cuda.CUDAGraph],
    resets: dict[str, Callable[[], None] | None],
    names: tuple[str, ...],
    replays: int,
    device: torch.device,
) -> None:
    for replay in range(replays):
        for name in balanced_order(names, replay):
            reset = resets[name]
            if reset is not None:
                reset()
            graphs[name].replay()
    torch.cuda.synchronize(device)


@torch.inference_mode()
def main() -> None:
    args = parse_args()
    rank, world_size, device, cpu_group = custom.init_distributed()
    if world_size != 4:
        raise RuntimeError("transport matrix is specialized for TP4")
    nccl_group = ps._WORLD.device_group
    if not isinstance(nccl_group, dist.ProcessGroup):
        raise RuntimeError("SGLang did not create the NCCL process group")
    props = torch.cuda.get_device_properties(device)
    intermediate_per_rank = custom.INTERMEDIATE // world_size
    max_message_bytes = max(args.ms) * custom.HIDDEN * torch.bfloat16.itemsize

    torch.manual_seed(args.seed + rank)
    torch.cuda.manual_seed(args.seed + rank)
    weights = custom.make_weights(intermediate_per_rank, device)
    lut = kernel.make_e2m1_e8m0_lut(device)
    comm = CustomAllReduceV2(
        cpu_group,
        device,
        max_pull_size=max_message_bytes,
        max_push_size=max_message_bytes,
    )
    if comm.disabled:
        raise RuntimeError("SGLang CustomAllReduceV2 is disabled")
    register_comm_cleanup(comm)
    comm.obj.config(num_pull_blocks=args.p2p_pull_blocks)
    register_comm(comm.obj, pull_sem_mc_ptr=comm.pull_sem_mc_ptr)
    if comm.mc_base_ptr == 0 or comm.pull_sem_mc_ptr == 0:
        raise RuntimeError("NVLS multicast mappings are unavailable")

    l2_flush_buffer = triton_runtime.driver.active.get_empty_cache_for_benchmark()
    if l2_flush_buffer.nbytes < 2 * props.L2_cache_size:
        raise RuntimeError("cache-clear buffer is smaller than twice L2")

    if rank == 0:
        print(
            "AR_TRANSPORT_ENV "
            + json.dumps(
                {
                    "benchmark": "TP4 one/two-shot and NVLS/P2P transport matrix",
                    "gpu": props.name,
                    "sm_count": props.multi_processor_count,
                    "physical_cuda_visible_devices": os.environ.get(
                        "CUDA_VISIBLE_DEVICES", ""
                    ),
                    "world_size": world_size,
                    "m_values": args.ms,
                    "scopes": args.scopes,
                    "variants": args.variants,
                    "route_pattern": args.route_pattern,
                    "outer": args.outer,
                    "replays_per_outer_per_variant": args.replays,
                    "warmup_replays": args.warmup_replays,
                    "p2p_pull_blocks": args.p2p_pull_blocks,
                    "nvls_pull_blocks": args.nvls_pull_blocks,
                    "nvls_pull_unroll": args.nvls_pull_unroll,
                    "pair_granularity": args.pair_granularity,
                    "max_push_bytes": comm.max_push_size,
                    "max_pull_bytes": comm.max_pull_size,
                    "multicast_ptr_nonzero": bool(comm.mc_base_ptr),
                    "l2_bytes": props.L2_cache_size,
                    "l2_clear_bytes": l2_flush_buffer.nbytes,
                    "l2_policy": (
                        "cold; AR input restore then separate 256MiB clear; "
                        "restore and clear excluded from CUDA events"
                    ),
                    "timing": "CUDA Graph replay, TP4 max rank, balanced rotation/reversal",
                    "same_communicator": True,
                },
                sort_keys=True,
            ),
            flush=True,
        )

    all_results: list[dict[str, object]] = []
    for m in args.ms:
        nbytes = m * custom.HIDDEN * torch.bfloat16.itemsize
        stock_algo, stock_mode = comm._pick_algo(nbytes, can_use_graph=True)

        # Rank-local random sources are intentionally different so correctness
        # checks prove that every collective really sums all four ranks.
        torch.manual_seed(args.seed + 100_000 + rank * 1009 + m)
        torch.cuda.manual_seed(args.seed + 100_000 + rank * 1009 + m)
        ar_source = torch.randn(
            (m, custom.HIDDEN), dtype=torch.bfloat16, device=device
        )
        ar_reference = ar_source.clone()
        dist.all_reduce(ar_reference, group=nccl_group)
        # Keep the rendezvous handle live for the lifetime of every graph that
        # dereferences its multicast VA.
        symm_work, symm_mc_ptr, symm_handle = symmetric_pull_view(comm, m)

        if "ar" in args.scopes:
            ar_cases: dict[str, ARCase] = {}
            ar_graphs: dict[str, torch.cuda.CUDAGraph] = {}
            for name in args.variants:
                if name == "nvls_2shot_pull":
                    work = symm_work
                    work_mc_ptr = symm_mc_ptr
                else:
                    work = torch.empty_like(ar_source)
                    work_mc_ptr = 0
                case = ARCase(
                    name,
                    ar_source,
                    work,
                    work_mc_ptr,
                    args.nvls_pull_blocks,
                    args.nvls_pull_unroll,
                )
                case.reset()
                ar_cases[name] = case
                ar_graphs[name] = capture_case(case, comm, cpu_group, device)

            ar_correctness: dict[str, dict[str, float | bool]] = {}
            ar_actuals: dict[str, torch.Tensor] = {}
            for name in args.variants:
                case = ar_cases[name]
                case.reset()
                ar_graphs[name].replay()
                torch.cuda.synchronize(device)
                assert case.graph_output is not None
                actual = case.graph_output.clone()
                ar_actuals[name] = actual
                check = correctness_metrics(actual, ar_reference, device, nccl_group)
                if not check["allreduce_ok"]:
                    raise RuntimeError(f"AR-only correctness failed for M={m}, {name}")
                ar_correctness[name] = check

            control_actual = ar_actuals["p2p_2shot_pull"]
            ar_cross_max_abs: dict[str, float] = {}
            for name, actual in ar_actuals.items():
                value = float((actual.float() - control_actual.float()).abs().max().item())
                ar_cross_max_abs[name] = reduce_rank_metric(
                    value, dist.ReduceOp.MAX, device, nccl_group
                )

            ar_resets: dict[str, Callable[[], None] | None] = {
                name: ar_cases[name].reset for name in args.variants
            }
            warm_graphs(
                ar_graphs,
                ar_resets,
                args.variants,
                args.warmup_replays,
                device,
            )
            ar_samples, ar_batches, ar_orders = time_graph_matrix(
                ar_graphs,
                ar_resets,
                args.variants,
                args.outer,
                args.replays,
                cpu_group,
                nccl_group,
                device,
                l2_flush_buffer,
                args.pair_granularity,
            )
            ar_control_batches = ar_batches["p2p_2shot_pull"]
            ar_records = {
                name: summarize_variant(
                    name,
                    ar_samples[name],
                    ar_batches[name],
                    ar_control_batches,
                    ar_correctness[name],
                )
                for name in args.variants
            }
            for name in args.variants:
                ar_records[name]["max_abs_vs_p2p_2shot"] = ar_cross_max_abs[name]
            result = {
                "scope": "ar_only",
                "m": m,
                "message_bytes": nbytes,
                "stock_graph_pick": None if stock_algo is None else stock_algo.name,
                "stock_graph_mode": stock_mode.name,
                "batch_orders": ar_orders,
                "variants": ar_records,
                "comparisons": comparison_summary(ar_records),
            }
            all_results.append(result)
            if rank == 0:
                print("AR_TRANSPORT_RESULT " + json.dumps(result, sort_keys=True), flush=True)

        # Deliberately retain the handle through both scopes for this M.
        _ = symm_handle

        if "full" in args.scopes:
            topk_ids, topk_weights = custom.make_routes(
                m, args.route_pattern, device, args.seed
            )
            # TP ranks consume the same activation and routing metadata.
            torch.manual_seed(args.seed + 200_000 + m)
            torch.cuda.manual_seed(args.seed + 200_000 + m)
            x = torch.randn(
                (m, custom.HIDDEN), dtype=torch.bfloat16, device=device
            ) * 0.1
            full_cases: dict[str, FullCase] = {}
            full_graphs: dict[str, torch.cuda.CUDAGraph] = {}
            for name in args.variants:
                base = make_compute_case(
                    m,
                    x,
                    topk_ids,
                    topk_weights,
                    weights,
                    lut,
                    intermediate_per_rank,
                )
                case = FullCase(
                    name,
                    base,
                    args.nvls_pull_blocks,
                    args.nvls_pull_unroll,
                )
                full_cases[name] = case
                full_graphs[name] = capture_case(case, comm, cpu_group, device)

            reference_case = make_compute_case(
                m,
                x,
                topk_ids,
                topk_weights,
                weights,
                lut,
                intermediate_per_rank,
            )
            full_reference = reference_case.run_local().clone()
            dist.all_reduce(full_reference, group=nccl_group)
            torch.cuda.synchronize(device)

            full_correctness: dict[str, dict[str, float | bool]] = {}
            full_actuals: dict[str, torch.Tensor] = {}
            for name in args.variants:
                full_graphs[name].replay()
                torch.cuda.synchronize(device)
                case = full_cases[name]
                assert case.graph_output is not None
                actual = case.graph_output.clone()
                full_actuals[name] = actual
                check = correctness_metrics(actual, full_reference, device, nccl_group)
                if not check["allreduce_ok"]:
                    raise RuntimeError(f"full correctness failed for M={m}, {name}")
                full_correctness[name] = check

            control_actual = full_actuals["p2p_2shot_pull"]
            full_cross_max_abs: dict[str, float] = {}
            for name, actual in full_actuals.items():
                value = float((actual.float() - control_actual.float()).abs().max().item())
                full_cross_max_abs[name] = reduce_rank_metric(
                    value, dist.ReduceOp.MAX, device, nccl_group
                )

            full_resets = {name: None for name in args.variants}
            warm_graphs(
                full_graphs,
                full_resets,
                args.variants,
                args.warmup_replays,
                device,
            )
            full_samples, full_batches, full_orders = time_graph_matrix(
                full_graphs,
                full_resets,
                args.variants,
                args.outer,
                args.replays,
                cpu_group,
                nccl_group,
                device,
                l2_flush_buffer,
                args.pair_granularity,
            )
            full_control_batches = full_batches["p2p_2shot_pull"]
            full_records = {
                name: summarize_variant(
                    name,
                    full_samples[name],
                    full_batches[name],
                    full_control_batches,
                    full_correctness[name],
                )
                for name in args.variants
            }
            for name in args.variants:
                full_records[name]["max_abs_vs_p2p_2shot"] = full_cross_max_abs[name]
            result = {
                "scope": "full_tp_moe",
                "m": m,
                "message_bytes": nbytes,
                "active_experts": int(torch.unique(topk_ids).numel()),
                "stock_graph_pick": None if stock_algo is None else stock_algo.name,
                "stock_graph_mode": stock_mode.name,
                "batch_orders": full_orders,
                "variants": full_records,
                "comparisons": comparison_summary(full_records),
            }
            all_results.append(result)
            if rank == 0:
                print("AR_TRANSPORT_RESULT " + json.dumps(result, sort_keys=True), flush=True)

    if rank == 0:
        print(
            "AR_TRANSPORT_SUMMARY "
            + json.dumps(
                {
                    "result_count": len(all_results),
                    "results": all_results,
                    "interpretation_rule": (
                        "select only direction-consistent paired batches; pooled "
                        "sub-percent differences are diagnostic"
                    ),
                },
                sort_keys=True,
            ),
            flush=True,
        )


if __name__ == "__main__":
    main()
