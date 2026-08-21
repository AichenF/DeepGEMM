#!/usr/bin/env python3
"""Run and validate the SM120 MegaMoE G8 full-chain benchmark."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import statistics
import subprocess
import sys
from pathlib import Path
from typing import Any

_RANK_PREFIX = "READY_CHUNK8_PERF_RANK_JSON="
_SUMMARY_PREFIX = "READY_CHUNK8_PERF_RESULT_JSON="
_AGGREGATE_PREFIX = "READY_CHUNK8_PERF_AGGREGATE_JSON="
_SOURCE_SHA256 = "dffd9acdc32716a523a05d8970035fef203b97084b1d5bc1b03db018e5ab0cae"
_MATRIX_SCHEMA = "cake-sm120-canonical-fused-ready-chunk8-selected-matrix-v1"
_EXPECTED_CASES = (
    "world1-r1-distinct-balanced-c110",
    "world2-r1-distinct-balanced-c110",
    "world2-r1-zero-balanced-mask1-c110",
    "world2-r17-analytic-balanced-mask7-c110",
    "world2-r17-analytic-empty-mask0-c110",
    "world2-r17-analytic-skewed-mask0-c110",
    "world4-r128-distinct-balanced-mask0-c110",
    "world8-r113-distinct-balanced-mask0-c110",
    "world8-r2048-distinct-balanced-mask0-c110",
)
_ZERO_FIELDS = (
    "precheck_failures",
    "postcheck_failures",
    "precheck_protocol_error",
    "precheck_owner_mismatches",
    "precheck_counter_mismatches",
    "precheck_signal_mismatches",
    "precheck_ack_signal_mismatches",
    "precheck_ready_audit_mismatches",
    "precheck_w1_bf16_mismatches",
    "precheck_requant_fp8_sf_mismatches",
    "precheck_w2_bf16_partial_mismatches",
    "precheck_output_mismatches",
    "precheck_output_guard_mismatches",
    "precheck_ring_mismatches",
    "precheck_launch_mismatches",
    "protocol_error",
    "owner_mismatches",
    "counter_mismatches",
    "signal_mismatches",
    "ack_signal_mismatches",
    "ready_audit_mismatches",
    "w1_bf16_mismatches",
    "requant_fp8_sf_mismatches",
    "w2_bf16_partial_mismatches",
    "output_mismatches",
    "output_guard_mismatches",
    "ring_mismatches",
    "launch_mismatches",
)


def validate_matrix_receipt(path: Path) -> tuple[dict[str, Any], str]:
    """Require the exact nine-case, 31-rank correctness matrix."""

    raw = path.read_bytes()
    receipt = json.loads(raw)
    if not isinstance(receipt, dict):
        raise ValueError("matrix receipt must be a JSON object")
    required = {
        "schema": _MATRIX_SCHEMA,
        "chunk8_generated_source_sha256": _SOURCE_SHA256,
        "status": "pass",
        "case_count": len(_EXPECTED_CASES),
        "rank_record_count": 31,
        "all_case_exit_zero": True,
        "all_rank_status_pass": True,
        "all_exact_bf16": True,
        "all_fail_fields_zero": True,
        "epoch_slots": [0, 1, 0],
    }
    for field, expected in required.items():
        if receipt.get(field) != expected:
            raise ValueError(
                f"matrix receipt {field} mismatch: "
                f"{receipt.get(field)!r} != {expected!r}"
            )
    if tuple(receipt.get("cases", ())) != _EXPECTED_CASES:
        raise ValueError("matrix receipt case set/order mismatch")
    return receipt, hashlib.sha256(raw).hexdigest()


def parse_rank_records(output: str) -> list[dict[str, Any]]:
    records = []
    for line in output.splitlines():
        if line.startswith(_RANK_PREFIX):
            payload = json.loads(line[len(_RANK_PREFIX) :])
            if not isinstance(payload, dict):
                raise ValueError("rank payload must be an object")
            records.append(payload)
    return records


def parse_summary(output: str) -> dict[str, Any]:
    summaries = []
    for line in output.splitlines():
        if line.startswith(_SUMMARY_PREFIX):
            payload = json.loads(line[len(_SUMMARY_PREFIX) :])
            if not isinstance(payload, dict):
                raise ValueError("summary payload must be an object")
            summaries.append(payload)
    if len(summaries) != 1:
        raise ValueError(f"expected one summary, observed {len(summaries)}")
    return summaries[0]


def _percentile(samples: list[float], quantile: float) -> float:
    ordered = sorted(samples)
    position = quantile * (len(ordered) - 1)
    lower = math.floor(position)
    upper = min(lower + 1, len(ordered) - 1)
    fraction = position - lower
    return ordered[lower] * (1.0 - fraction) + ordered[upper] * fraction


def aggregate_rank_records(
    records: list[dict[str, Any]], *, expected_world_size: int
) -> dict[str, Any]:
    """Validate every rank, then take the aligned max for each sample."""

    if len(records) != expected_world_size:
        raise ValueError(
            f"expected {expected_world_size} rank records, observed {len(records)}"
        )
    by_rank: dict[int, dict[str, Any]] = {}
    for record in records:
        rank = int(record["rank"])
        if rank in by_rank:
            raise ValueError(f"duplicate rank {rank}")
        by_rank[rank] = record
    if sorted(by_rank) != list(range(expected_world_size)):
        raise ValueError(f"rank set is not contiguous: {sorted(by_rank)}")

    first = by_rank[0]
    shared_fields = (
        "world_size",
        "active_rows",
        "oracle",
        "route_mode",
        "mask_period",
        "epoch_base",
        "warmup_launches",
        "repeat_launches",
        "ready_ctas",
        "threads_per_cta",
        "dynamic_smem_bytes",
        "kernel_count",
        "kernel_count_per_sample",
        "epochs_per_sample",
        "static_generated_grid_sync_sites",
    )
    required_true = (
        "same_process",
        "same_communicator",
        "single_launch_full_chain",
        "ready_driven",
        "early_signal_base_hb",
        "post_combine_ack",
        "zero_flush",
        "selected_full_matrix_required_before_execution",
        "chunked_task_claim",
        "task_major_chunk_issuance",
        "forced_w1_opportunity_after_each_early_w2_chunk",
        "rank_rendezvous_outside_event",
        "initialization_reference_outside_event",
    )
    required_false = (
        "barrier_ordered",
        "formal_functional_qualified",
        "runtime_register_repartition_qualified",
        "resource_qualified",
        "performance_qualified",
        "production_compute_comparable",
        "diagnostic_oracle_in_timing",
        "phase_clock64_in_timing",
    )
    exact_scalars = {
        "audit_epochs_before": 3,
        "audit_epochs_after": 3,
        "kernel_count": 1,
        "kernel_count_per_sample": 1,
        "epochs_per_sample": 1,
        "static_generated_grid_sync_sites": 15,
        "ready_ctas": 110,
        "threads_per_cta": 384,
        "dynamic_smem_bytes": 94208,
        "chunk_physical_n128_tiles": 8,
        "w1_chunks_per_task": 6,
        "w2_chunks_per_task": 7,
        "w1_warmup_tasks": 27,
        "early_w2_worker_limit": 27,
    }
    for rank, record in by_rank.items():
        if record.get("correctness_status") != "pass":
            raise ValueError(f"rank {rank} correctness did not pass")
        if record.get("exact_bf16_equal") is not True:
            raise ValueError(f"rank {rank} is not exact BF16")
        for field in _ZERO_FIELDS:
            if int(record.get(field, -1)) != 0:
                raise ValueError(f"rank {rank} failed {field}")
        if any(record.get(field) is not True for field in required_true):
            raise ValueError(f"rank {rank} weakened a required true contract")
        if any(record.get(field) is not False for field in required_false):
            raise ValueError(f"rank {rank} weakened a required false contract")
        if record.get("event_boundary") != "chunk8_ready_kernel_only":
            raise ValueError(f"rank {rank} has the wrong event boundary")
        if record.get("credit_mechanism") != "two_slot_strong_result_post_combine_ack":
            raise ValueError(f"rank {rank} changed the slot-credit mechanism")
        for field, expected in exact_scalars.items():
            if int(record.get(field, -1)) != expected:
                raise ValueError(f"rank {rank} changed {field}")
        if int(record.get("ready_capacity", -1)) < 110:
            raise ValueError(f"rank {rank} lacks cooperative capacity")
        for field in shared_fields:
            if record[field] != first[field]:
                raise ValueError(f"rank {rank} disagrees on {field}")

    repeat = int(first["repeat_launches"])
    warmup = int(first["warmup_launches"])
    if warmup != 5 or repeat not in (20, 100):
        raise ValueError("evidence requires five warmups and 20 or 100 repeats")
    epoch_base = int(first["epoch_base"])
    expected_launches = 3 + warmup + repeat + 3
    expected_pre_slots = [(epoch_base + offset) & 1 for offset in range(3)]
    post_base = epoch_base + 3 + warmup + repeat
    expected_post_slots = [(post_base + offset) & 1 for offset in range(3)]
    samples_by_rank = []
    for rank in range(expected_world_size):
        record = by_rank[rank]
        expected_routes = int(record.get("expected_received_routes", -1))
        if expected_routes < 0:
            raise ValueError(f"rank {rank} has invalid expected routes")
        if int(record.get("total_kernel_launches", -1)) != expected_launches:
            raise ValueError(f"rank {rank} has the wrong launch count")
        if record.get("precheck_slots") != expected_pre_slots:
            raise ValueError(f"rank {rank} failed precheck slots")
        if record.get("postcheck_slots") != expected_post_slots:
            raise ValueError(f"rank {rank} failed postcheck slots")
        if record.get("precheck_route_totals") != [expected_routes] * 3:
            raise ValueError(f"rank {rank} failed precheck routes")
        if record.get("postcheck_route_totals") != [expected_routes] * 3:
            raise ValueError(f"rank {rank} failed postcheck routes")
        samples = [float(value) for value in record.get("samples_ms", [])]
        if len(samples) != repeat:
            raise ValueError(f"rank {rank} has {len(samples)} samples")
        if not all(math.isfinite(value) and value > 0.0 for value in samples):
            raise ValueError(f"rank {rank} has invalid samples")
        samples_by_rank.append(samples)

    distributed_samples = []
    critical_ranks = []
    for sample_index in range(repeat):
        rank_values = [samples[sample_index] for samples in samples_by_rank]
        critical_rank = max(range(expected_world_size), key=rank_values.__getitem__)
        distributed_samples.append(rank_values[critical_rank])
        critical_ranks.append(critical_rank)
    mean_ms = statistics.fmean(distributed_samples)
    total_flops = sum(
        int(by_rank[rank]["tensor_flops_per_epoch"])
        for rank in range(expected_world_size)
    )
    return {
        "schema_version": 1,
        "kind": "cake_sm120_canonical_fused_ready_chunk8_perf_aggregate",
        "world_size": expected_world_size,
        "active_rows": int(first["active_rows"]),
        "oracle": str(first["oracle"]),
        "route_mode": str(first["route_mode"]),
        "mask_period": int(first["mask_period"]),
        "epoch_base": epoch_base,
        "warmup_launches": warmup,
        "repeat_launches": repeat,
        "kernel_count": 1,
        "kernel_count_per_sample": 1,
        "single_launch_full_chain": True,
        "formal_functional_qualified": False,
        "resource_qualified": False,
        "performance_qualified": False,
        "production_compute_comparable": False,
        "distributed_samples_ms": distributed_samples,
        "critical_rank_by_sample": critical_ranks,
        "distributed_mean_ms": mean_ms,
        "distributed_p50_ms": _percentile(distributed_samples, 0.50),
        "distributed_p95_ms": _percentile(distributed_samples, 0.95),
        "distributed_min_ms": min(distributed_samples),
        "distributed_max_ms": max(distributed_samples),
        "aggregate_tensor_tflops": total_flops / (mean_ms * 1.0e9),
        "per_rank_samples_ms": samples_by_rank,
        "per_rank": [by_rank[rank] for rank in range(expected_world_size)],
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--binary", type=Path, required=True)
    parser.add_argument("--matrix-receipt", type=Path, required=True)
    parser.add_argument("--world-size", type=int, choices=(1, 2, 4, 8), default=8)
    parser.add_argument("--active-rows", type=int, default=2048)
    parser.add_argument("--warmup", type=int, choices=(5,), default=5)
    parser.add_argument("--repeat", type=int, choices=(20, 100), default=20)
    parser.add_argument(
        "--oracle", choices=("zero", "analytic", "distinct_k32"), default="distinct_k32"
    )
    parser.add_argument(
        "--route-mode", choices=("balanced", "skewed", "empty"), default="balanced"
    )
    parser.add_argument("--mask-period", type=int, default=0)
    parser.add_argument("--epoch-base", type=int, default=0)
    parser.add_argument("--timeout-s", type=float, default=14400.0)
    parser.add_argument("--output", type=Path)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if not 1 <= args.active_rows <= 2048:
        raise SystemExit("--active-rows must be in [1,2048]")
    if not 0 <= args.mask_period <= 2048 * 6:
        raise SystemExit("--mask-period must be in [0,12288]")
    if not 0 <= args.epoch_base <= 1_000_000_000:
        raise SystemExit("--epoch-base must be in [0,1000000000]")
    if args.timeout_s <= 0.0:
        raise SystemExit("--timeout-s must be positive")
    binary = args.binary.resolve()
    if not binary.is_file():
        raise SystemExit(f"benchmark binary does not exist: {binary}")
    receipt_path = args.matrix_receipt.resolve()
    if not receipt_path.is_file():
        raise SystemExit(f"matrix receipt does not exist: {receipt_path}")
    _, receipt_sha = validate_matrix_receipt(receipt_path)

    env = os.environ.copy()
    for key in (
        "RANK",
        "WORLD_SIZE",
        "SLURM_PROCID",
        "SLURM_NTASKS",
        "NCCL_UNIQUE_ID_FILE",
        "LOCAL_DEVICE",
    ):
        env.pop(key, None)
    env.update(
        {
            "NTHREADS": str(args.world_size),
            "CAKE_ACTIVE_ROWS": str(args.active_rows),
            "CAKE_MASK_PERIOD": str(args.mask_period),
            "CAKE_ROUTE_MODE": args.route_mode,
            "CAKE_ORACLE": args.oracle,
            "CAKE_WARMUP": str(args.warmup),
            "CAKE_REPEAT": str(args.repeat),
            "CAKE_EPOCH_BASE": str(args.epoch_base),
        }
    )
    completed = subprocess.run(
        [str(binary)],
        check=False,
        capture_output=True,
        text=True,
        timeout=args.timeout_s,
        env=env,
    )
    sys.stdout.write(completed.stdout)
    sys.stderr.write(completed.stderr)
    if completed.returncode != 0:
        return completed.returncode

    summary = parse_summary(completed.stdout)
    if int(summary.get("world_size", -1)) != args.world_size:
        raise ValueError("completion summary has the wrong world size")
    for field in ("same_process", "same_communicator"):
        if summary.get(field) is not True:
            raise ValueError(f"completion summary failed {field}")
    if int(summary.get("kernel_count", -1)) != 1:
        raise ValueError("completion summary did not use one kernel entry")
    if int(summary.get("failures", -1)) != 0 or summary.get("status") != "pass":
        raise ValueError("completion summary failed")

    aggregate = aggregate_rank_records(
        parse_rank_records(completed.stdout), expected_world_size=args.world_size
    )
    aggregate["selected_matrix_receipt_sha256"] = receipt_sha
    aggregate["selected_matrix_gate_passed"] = True
    aggregate["command"] = [str(binary)]
    aggregate["environment"] = {
        "NTHREADS": str(args.world_size),
        "CAKE_ACTIVE_ROWS": str(args.active_rows),
        "CAKE_MASK_PERIOD": str(args.mask_period),
        "CAKE_ROUTE_MODE": args.route_mode,
        "CAKE_ORACLE": args.oracle,
        "CAKE_WARMUP": str(args.warmup),
        "CAKE_REPEAT": str(args.repeat),
        "CAKE_EPOCH_BASE": str(args.epoch_base),
    }
    rendered = json.dumps(aggregate, sort_keys=True, separators=(",", ":"))
    print(f"{_AGGREGATE_PREFIX}{rendered}")
    if args.output is not None:
        output = args.output.resolve()
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(rendered + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
