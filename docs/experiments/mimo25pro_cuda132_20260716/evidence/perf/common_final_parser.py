#!/usr/bin/env python3
"""Strict raw-log comparison for final MiMo versus the frozen Aichen baseline.

The formal final run contains one excluded complete warmup followed by thirty
fresh complete processes.  Every process runs the same ordered eleven-M,
eight-rank MiMo benchmark with ``--num-tests 20``.  The comparison is rebuilt
from the thirty raw baseline logs and the thirty raw final logs; precomputed
baseline summary values are never used as inputs.

Performance is deliberately observational.  A slower or faster result never
changes this parser's exit status.  Nonzero exit is reserved for malformed,
missing, additional, retried, mutated, or otherwise untrustworthy evidence.
"""

from __future__ import annotations

import argparse
import contextlib
import csv
import hashlib
import json
import math
import re
import statistics
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterable


EXPECTED_M = (8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192)
EXPECTED_ROUNDS = 30
NUM_TESTS = 20
MEDIAN_CI_COVERAGE_PCT = 95.7226
IMAGEPERF_RANK0_US = (
    491.4, 536.4, 554.7, 568.8, 524.4, 540.2,
    1002.2, 1500.1, 2852.3, 5337.9, 10398.5,
)

DEFAULT_BASELINE_DIR = Path(
    "/home/xuechengw/MegaMoe/artifacts/"
    "aichen_imageperf_three_models_cuda132_20260716/"
    "job3119858_20260716_034631"
)

# These bind the comparison to the already completed new-node baseline.
PINNED_BASELINE_SHA256 = {
    "artifact_sha256.txt": "e29d50eec961c77adbaedeb60f7906d41a2d70d1a1e738cf2ca3873e99b19071",
    "comparison_30.csv": "a32bde987175dacf60173207565b550398b1b91d6a68f1e462eaec95daea70ef",
    "individual_runs_30.csv": "ca6395379429eaf8bcc1a579de0b960f2bf68d02d8560348e234cf3db38de05c",
    "process_ledger.tsv": "5cda919490917371551f5e979eb63323ec302d6ebfb8b7df279248e3d13fb8f0",
    "identity_manifest.txt": "b479f131ea42644bd86a05234109465c48bf3e7e876559e2746cf0c9638febfa",
    "run_three_models_baseline_cuda132_30.snapshot.sh": "d5f2c7319740aa981fe5198d1fdc0a9d5926475efbf77e2eaf4fca05108ecfc1",
    "summarize_three_models_30.snapshot.py": "eced94465b860a0073bc6951632484f135529ddad8ba847f98f169af63a5c29f",
}


@dataclass(frozen=True)
class ModelSpec:
    key: str
    display_name: str
    hidden: int
    intermediate_hidden: int
    num_experts: int
    num_topk: int
    recv: tuple[int, ...]
    touched_experts: tuple[int, ...]

    @property
    def header(self) -> str:
        return (
            "SM90 MegaMoE bench: ranks=8 "
            f"hidden={self.hidden} ih={self.intermediate_hidden} "
            f"experts={self.num_experts} topk={self.num_topk} "
            "masked_ratio=0.0 fast_math=True"
        )

    @property
    def recv_by_m(self) -> dict[int, int]:
        return dict(zip(EXPECTED_M, self.recv, strict=True))

    @property
    def touched_by_m(self) -> dict[int, int]:
        return dict(zip(EXPECTED_M, self.touched_experts, strict=True))


MODEL_SPECS = {
    "flash": ModelSpec(
        "flash", "Flash", 4096, 2048, 256, 6,
        (46, 92, 170, 378, 784, 1568, 3147, 6221, 12365, 24547, 49065),
        (24, 30, 32, 32, 32, 32, 32, 32, 32, 32, 32),
    ),
    "pro": ModelSpec(
        "pro", "Pro", 7168, 3072, 384, 6,
        (50, 89, 187, 348, 789, 1574, 3089, 6102, 12294, 24323, 49566),
        (33, 39, 47, 48, 48, 48, 48, 48, 48, 48, 48),
    ),
    "mimo": ModelSpec(
        "mimo", "MiMo-V2.5-Pro", 6144, 2048, 384, 8,
        (64, 118, 241, 546, 984, 2037, 4097, 8140, 16409, 32686, 65538),
        (39, 45, 48, 48, 48, 48, 48, 48, 48, 48, 48),
    ),
}
MIMO = MODEL_SPECS["mimo"]

ROW_RE = re.compile(
    r"tokens=\s*(?P<m>\d+)\s+"
    r"recv=\s*(?P<recv>\d+)\s+"
    r"experts=\s*(?P<experts>\d+)\s+"
    r"nvfp4=\s*(?P<rank0>[0-9]+(?:\.[0-9]+)?)us\s+"
    r"mean_rank=\s*(?P<mean_rank>[0-9]+(?:\.[0-9]+)?)us\s+"
    r"max_rank=\s*(?P<max_rank>[0-9]+(?:\.[0-9]+)?)us\s+"
    r"\(\s*(?P<tflops>[0-9]+(?:\.[0-9]+)?)TF,\s*"
    r"(?P<hbm_gbs>[0-9]+(?:\.[0-9]+)?)GB/s\)\s+\(rank0\)"
)
HEADER_RE = re.compile(r"^SM90 MegaMoE bench:.*$", re.MULTILINE)
FAILURE_MARKERS = (
    "Traceback (most recent call last)",
    "AssertionError",
    "Segmentation fault",
    "CUDA error:",
    "NCCL error",
    "DistBackendError",
    "ProcessExitedException",
    "ChildFailedError",
    "RuntimeError:",
    "OutOfMemoryError",
    "CUBLAS_STATUS_",
    "NCCL WARN",
)
COMPILE_MARKERS = ("Running NVCC command:", "Compiling JIT runtime")
LEDGER_HEADER = (
    "label", "kind", "started_at", "ended_at", "exit_code", "log_sha256"
)


def fail(message: str) -> None:
    raise ValueError(message)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def require_file(root: Path, name: str) -> Path:
    path = root / name
    if not path.is_file():
        fail(f"missing required file: {path}")
    return path


def validate_pinned_baseline(root: Path) -> None:
    root = root.resolve()
    if root != DEFAULT_BASELINE_DIR.resolve():
        fail(f"baseline directory is not the frozen run: {root}")
    for name, expected_sha in PINNED_BASELINE_SHA256.items():
        path = require_file(root, name)
        observed = sha256_file(path)
        if observed != expected_sha:
            fail(
                f"pinned baseline hash mismatch for {name}: "
                f"expected {expected_sha}, got {observed}"
            )

    # Recheck every file named by the baseline run's own artifact manifest.
    manifest = require_file(root, "artifact_sha256.txt")
    checked = 0
    for line_number, line in enumerate(
        manifest.read_text(encoding="utf-8").splitlines(), 1
    ):
        match = re.fullmatch(r"([0-9a-f]{64})  (.+)", line)
        if match is None:
            fail(f"{manifest}: malformed line {line_number}")
        expected_sha, raw_path = match.groups()
        path = Path(raw_path)
        try:
            path.resolve().relative_to(root)
        except ValueError as error:
            raise ValueError(
                f"{manifest}: path escapes frozen baseline: {path}"
            ) from error
        if not path.is_file():
            fail(f"{manifest}: missing listed file: {path}")
        if sha256_file(path) != expected_sha:
            fail(f"{manifest}: listed file hash mismatch: {path}")
        checked += 1
    if checked < 200:
        fail(f"{manifest}: unexpectedly short artifact manifest ({checked} files)")


def baseline_round_order(round_index: int) -> tuple[str, ...]:
    if not 1 <= round_index <= EXPECTED_ROUNDS:
        fail(f"round outside 1..30: {round_index}")
    order = ("flash", "pro", "mimo")
    shift = (round_index - 1) % 3
    return order[shift:] + order[:shift]


def expected_baseline_entries() -> list[tuple[str, str, str, int | None]]:
    entries = [
        (f"warmup_{model}", "excluded_warmup", model, None)
        for model in ("flash", "pro", "mimo")
    ]
    for round_index in range(1, EXPECTED_ROUNDS + 1):
        for model in baseline_round_order(round_index):
            entries.append(
                (f"round_{round_index:02d}_{model}", "measured", model, round_index)
            )
    return entries


def expected_final_entries() -> list[tuple[str, str, str, int | None]]:
    return [("warmup_mimo", "excluded_warmup", "mimo", None)] + [
        (f"round_{round_index:02d}_mimo", "measured", "mimo", round_index)
        for round_index in range(1, EXPECTED_ROUNDS + 1)
    ]


def parse_benchmark_log(
    path: Path, model: str = "mimo", *, forbid_compile: bool = False
) -> list[dict[str, Any]]:
    spec = MODEL_SPECS[model]
    text = path.read_text(encoding="utf-8", errors="replace")
    headers = HEADER_RE.findall(text)
    if headers != [spec.header]:
        fail(
            f"{path}: expected exactly one {model} header {spec.header!r}; "
            f"got {headers!r}"
        )
    for marker in FAILURE_MARKERS:
        if marker in text:
            fail(f"{path}: failure marker found: {marker}")
    if forbid_compile:
        for marker in COMPILE_MARKERS:
            if marker in text:
                fail(f"{path}: measured process unexpectedly compiled JIT: {marker}")

    rows: list[dict[str, Any]] = []
    for match in ROW_RE.finditer(text):
        row: dict[str, Any] = {
            "m": int(match.group("m")),
            "recv": int(match.group("recv")),
            "experts": int(match.group("experts")),
            "rank0_us": float(match.group("rank0")),
            "mean_rank_us": float(match.group("mean_rank")),
            "max_rank_us": float(match.group("max_rank")),
            "tflops": float(match.group("tflops")),
            "hbm_gbs": float(match.group("hbm_gbs")),
        }
        for key in ("rank0_us", "mean_rank_us", "max_rank_us", "tflops", "hbm_gbs"):
            if not math.isfinite(row[key]) or row[key] <= 0:
                fail(f"{path}: M={row['m']} invalid {key}={row[key]!r}")
        # Printed timings have 0.1 us precision.
        if row["max_rank_us"] + 0.11 < row["rank0_us"]:
            fail(f"{path}: M={row['m']} max_rank is below rank0")
        if row["max_rank_us"] + 0.11 < row["mean_rank_us"]:
            fail(f"{path}: M={row['m']} max_rank is below mean_rank")
        rows.append(row)

    actual_m = [row["m"] for row in rows]
    if actual_m != list(EXPECTED_M):
        fail(f"{path}: ordered M mismatch; expected {list(EXPECTED_M)}, got {actual_m}")
    for row in rows:
        m = row["m"]
        if row["recv"] != spec.recv_by_m[m]:
            fail(
                f"{path}: {model} M={m} recv mismatch; "
                f"expected {spec.recv_by_m[m]}, got {row['recv']}"
            )
        if row["experts"] != spec.touched_by_m[m]:
            fail(
                f"{path}: {model} M={m} touched-experts mismatch; "
                f"expected {spec.touched_by_m[m]}, got {row['experts']}"
            )
    return rows


def read_and_validate_ledger(
    root: Path,
    expected: list[tuple[str, str, str, int | None]],
) -> list[tuple[str, str, str, int | None, Path, str]]:
    ledger = require_file(root, "process_ledger.tsv")
    with ledger.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        if tuple(reader.fieldnames or ()) != LEDGER_HEADER:
            fail(f"{ledger}: ledger header mismatch")
        rows = list(reader)
    if len(rows) != len(expected):
        fail(f"{ledger}: expected {len(expected)} rows, got {len(rows)}")

    validated: list[tuple[str, str, str, int | None, Path, str]] = []
    for index, (row, expected_entry) in enumerate(zip(rows, expected, strict=True), 1):
        label, kind, model, round_index = expected_entry
        if row["label"] != label or row["kind"] != kind:
            fail(
                f"{ledger}: row {index} process order/class mismatch; "
                f"expected {label}/{kind}, got {row!r}"
            )
        if row["exit_code"] != "0":
            fail(f"{ledger}: row {index} has nonzero exit")
        if not row["started_at"] or not row["ended_at"]:
            fail(f"{ledger}: row {index} has empty timestamps")
        log_sha = row["log_sha256"]
        if re.fullmatch(r"[0-9a-f]{64}", log_sha) is None:
            fail(f"{ledger}: row {index} malformed log hash")
        log_path = require_file(root, f"{label}.log")
        if sha256_file(log_path) != log_sha:
            fail(f"{ledger}: log hash mismatch for {log_path.name}")
        validated.append((label, kind, model, round_index, log_path, log_sha))

    actual_logs = {
        path.name
        for pattern in ("warmup_*.log", "round_*.log", "sweep_*.log")
        for path in root.glob(pattern)
    }
    expected_logs = {f"{entry[0]}.log" for entry in expected}
    if actual_logs != expected_logs:
        fail(
            "protocol log set mismatch; "
            f"missing={sorted(expected_logs - actual_logs)}, "
            f"extra={sorted(actual_logs - expected_logs)}"
        )
    return validated


def load_baseline_runs(
    root: Path, *, enforce_pinned_identity: bool
) -> list[list[dict[str, Any]]]:
    root = root.resolve()
    if enforce_pinned_identity:
        validate_pinned_baseline(root)
    entries = read_and_validate_ledger(root, expected_baseline_entries())
    runs: list[list[dict[str, Any]]] = []
    next_round = 1
    for label, kind, model, round_index, log_path, ledger_sha in entries:
        if model != "mimo":
            continue
        rows = parse_benchmark_log(
            log_path, model, forbid_compile=(kind == "measured")
        )
        if sha256_file(log_path) != ledger_sha:
            fail(f"{log_path}: log changed while being parsed")
        if kind == "measured":
            if round_index != next_round:
                fail(f"{label}: expected MiMo round {next_round}, got {round_index}")
            runs.append(rows)
            next_round += 1
    if len(runs) != EXPECTED_ROUNDS:
        fail(f"baseline: expected 30 MiMo measured processes, got {len(runs)}")
    return runs


def load_final_runs(root: Path) -> list[list[dict[str, Any]]]:
    root = root.resolve()
    entries = read_and_validate_ledger(root, expected_final_entries())
    runs: list[list[dict[str, Any]]] = []
    saw_warmup = False
    for label, kind, model, round_index, log_path, ledger_sha in entries:
        rows = parse_benchmark_log(
            log_path, model, forbid_compile=(kind == "measured")
        )
        if sha256_file(log_path) != ledger_sha:
            fail(f"{log_path}: log changed while being parsed")
        if kind == "excluded_warmup":
            if saw_warmup or round_index is not None:
                fail("final: duplicate or misclassified warmup")
            saw_warmup = True
        else:
            if round_index != len(runs) + 1:
                fail(f"{label}: non-contiguous final round")
            runs.append(rows)
    if not saw_warmup or len(runs) != EXPECTED_ROUNDS:
        fail("final: expected one warmup and 30 measured processes")
    return runs


def percentile(values: list[float], fraction: float) -> float:
    if not 0 <= fraction <= 1:
        fail(f"invalid percentile: {fraction}")
    ordered = sorted(values)
    position = (len(ordered) - 1) * fraction
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return float(ordered[lower])
    weight = position - lower
    return float(ordered[lower] * (1 - weight) + ordered[upper] * weight)


def distribution(values: Iterable[float]) -> dict[str, float]:
    materialized = list(values)
    if len(materialized) != EXPECTED_ROUNDS:
        fail(f"distribution requires 30 values, got {len(materialized)}")
    if any(not math.isfinite(value) or value <= 0 for value in materialized):
        fail("distribution contains invalid values")
    ordered = sorted(materialized)
    return {
        "median": float(statistics.median(materialized)),
        "p10": percentile(materialized, 0.10),
        "p90": percentile(materialized, 0.90),
        "min": float(ordered[0]),
        "max": float(ordered[-1]),
        "median_ci_x10_low": float(ordered[9]),
        "median_ci_x21_high": float(ordered[20]),
    }


def metric_distributions(
    runs: list[list[dict[str, Any]]], row_index: int
) -> dict[str, dict[str, float]]:
    return {
        "rank0": distribution(run[row_index]["rank0_us"] for run in runs),
        "mean_rank": distribution(run[row_index]["mean_rank_us"] for run in runs),
        "max_rank": distribution(run[row_index]["max_rank_us"] for run in runs),
    }


def layout_for(source: str, m: int) -> str:
    if source == "final":
        return "BN256-grouped"
    return "BN256-standard" if m <= 1024 else "BN128-split"


def make_results(
    baseline_runs: list[list[dict[str, Any]]],
    final_runs: list[list[dict[str, Any]]],
) -> list[dict[str, Any]]:
    results: list[dict[str, Any]] = []
    for row_index, m in enumerate(EXPECTED_M):
        baseline = metric_distributions(baseline_runs, row_index)
        final = metric_distributions(final_runs, row_index)
        imageperf_rank0 = IMAGEPERF_RANK0_US[row_index]
        deltas = {
            metric: 100.0 * (final[metric]["median"] / baseline[metric]["median"] - 1)
            for metric in ("rank0", "mean_rank", "max_rank")
        }
        speedups = {
            metric: baseline[metric]["median"] / final[metric]["median"]
            for metric in ("rank0", "mean_rank", "max_rank")
        }
        results.append(
            {
                "m": m,
                "recv": MIMO.recv_by_m[m],
                "touched_experts": MIMO.touched_by_m[m],
                "baseline_layout": layout_for("baseline", m),
                "final_layout": layout_for("final", m),
                "imageperf_rank0_us": imageperf_rank0,
                "baseline_vs_imageperf_rank0_median_delta_pct": 100.0
                * (baseline["rank0"]["median"] / imageperf_rank0 - 1.0),
                "final_vs_imageperf_rank0_median_delta_pct": 100.0
                * (final["rank0"]["median"] / imageperf_rank0 - 1.0),
                "baseline": baseline,
                "final": final,
                "final_vs_baseline_delta_pct": deltas,
                "baseline_over_final_speedup": speedups,
            }
        )
    return results


def geomean_delta(results: list[dict[str, Any]], metric: str) -> float:
    log_ratios = [
        math.log(row["final"][metric]["median"] / row["baseline"][metric]["median"])
        for row in results
    ]
    return 100.0 * (math.exp(statistics.mean(log_ratios)) - 1.0)


def write_individual_csv(
    output_root: Path,
    baseline_runs: list[list[dict[str, Any]]],
    final_runs: list[list[dict[str, Any]]],
) -> None:
    path = output_root / "individual_baseline_vs_final_30.csv"
    fields = [
        "source", "round", "log", "m", "recv", "experts", "layout",
        "rank0_us", "mean_rank_us", "max_rank_us", "tflops", "hbm_gbs",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for source, runs in (("baseline", baseline_runs), ("final", final_runs)):
            for round_index, rows in enumerate(runs, 1):
                for row in rows:
                    writer.writerow(
                        {
                            "source": source,
                            "round": round_index,
                            "log": f"round_{round_index:02d}_mimo.log",
                            "layout": layout_for(source, row["m"]),
                            **row,
                        }
                    )


def flat_rows(results: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for result in results:
        flat: dict[str, Any] = {
            key: result[key]
            for key in (
                "m", "recv", "touched_experts", "baseline_layout", "final_layout",
                "imageperf_rank0_us",
                "baseline_vs_imageperf_rank0_median_delta_pct",
                "final_vs_imageperf_rank0_median_delta_pct",
            )
        }
        for source in ("baseline", "final"):
            for metric in ("rank0", "mean_rank", "max_rank"):
                for statistic, value in result[source][metric].items():
                    flat[f"{source}_{metric}_{statistic}_us"] = value
        for metric in ("rank0", "mean_rank", "max_rank"):
            flat[f"final_vs_baseline_{metric}_median_delta_pct"] = result[
                "final_vs_baseline_delta_pct"
            ][metric]
            flat[f"baseline_over_final_{metric}_speedup"] = result[
                "baseline_over_final_speedup"
            ][metric]
        rows.append(flat)
    return rows


def markdown_report(results: list[dict[str, Any]], geomean: dict[str, float]) -> str:
    lines = [
        "# CUDA 13.2 MiMo final kernel versus frozen Aichen baseline",
        "",
        (
            "Both sides are recomputed from raw logs. Each side contributes 30 "
            "fresh complete processes; every process contains the same ordered 11 M "
            "values and uses the protected benchmark with `--num-tests 20`."
        ),
        (
            "Primary comparison is rank0 median30, matching the frozen baseline. "
            "Mean-rank and max-rank are retained for deployment diagnostics. "
            "Negative delta means the final kernel is faster. Performance is reported, "
            "not used as a parser pass/fail gate."
        ),
        "",
        "| M | recv | layout baseline → final | ImagePerf rank0 | rank0 baseline | baseline vs image | rank0 final | final vs image | final vs baseline | baseline P10–P90 | final P10–P90 | baseline CI X10–X21 | final CI X10–X21 | max median baseline → final | max delta | mean median baseline → final | mean delta |",
        "|--:|--:|:--|--:|--:|--:|--:|--:|--:|:--|:--|:--|:--|:--|--:|:--|--:|",
    ]
    for row in results:
        b = row["baseline"]
        f = row["final"]
        d = row["final_vs_baseline_delta_pct"]
        lines.append(
            f"| {row['m']} | {row['recv']} | {row['baseline_layout']} → {row['final_layout']} | "
            f"{row['imageperf_rank0_us']:.1f} | {b['rank0']['median']:.2f} | "
            f"{row['baseline_vs_imageperf_rank0_median_delta_pct']:+.2f}% | "
            f"{f['rank0']['median']:.2f} | "
            f"{row['final_vs_imageperf_rank0_median_delta_pct']:+.2f}% | "
            f"{d['rank0']:+.2f}% | "
            f"{b['rank0']['p10']:.2f}–{b['rank0']['p90']:.2f} | "
            f"{f['rank0']['p10']:.2f}–{f['rank0']['p90']:.2f} | "
            f"{b['rank0']['median_ci_x10_low']:.1f}–{b['rank0']['median_ci_x21_high']:.1f} | "
            f"{f['rank0']['median_ci_x10_low']:.1f}–{f['rank0']['median_ci_x21_high']:.1f} | "
            f"{b['max_rank']['median']:.2f} → {f['max_rank']['median']:.2f} | {d['max_rank']:+.2f}% | "
            f"{b['mean_rank']['median']:.2f} → {f['mean_rank']['median']:.2f} | {d['mean_rank']:+.2f}% |"
        )
    lines.extend(
        [
            "",
            "## Equal-weight 11-point geometric mean latency change",
            "",
            f"- rank0: {geomean['rank0']:+.3f}%",
            f"- mean-rank: {geomean['mean_rank']:+.3f}%",
            f"- max-rank: {geomean['max_rank']:+.3f}%",
            "",
            (
                f"Each median interval is the distribution-free {MEDIAN_CI_COVERAGE_PCT:.4f}% "
                "X10–X21 interval from all 30 process values."
            ),
            "",
            (
                "The baseline uses its stock automatic layout (BN256 through M=1024, "
                "BN128 from M=2048). The final release policy intentionally keeps the "
                "MiMo geometry on BN256 grouped-nibble for all eleven points."
            ),
        ]
    )
    return "\n".join(lines) + "\n"


def summarize(
    final_root: Path,
    baseline_root: Path,
    *,
    enforce_pinned_baseline: bool = True,
) -> dict[str, Any]:
    final_root = final_root.resolve()
    baseline_root = baseline_root.resolve()
    if not final_root.is_dir():
        fail(f"final run directory does not exist: {final_root}")
    if not baseline_root.is_dir():
        fail(f"baseline run directory does not exist: {baseline_root}")

    baseline_runs = load_baseline_runs(
        baseline_root, enforce_pinned_identity=enforce_pinned_baseline
    )
    final_runs = load_final_runs(final_root)
    results = make_results(baseline_runs, final_runs)
    geomean = {
        metric: geomean_delta(results, metric)
        for metric in ("rank0", "mean_rank", "max_rank")
    }

    write_individual_csv(final_root, baseline_runs, final_runs)
    flattened = flat_rows(results)
    with (final_root / "comparison_baseline_vs_final_30.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=list(flattened[0]))
        writer.writeheader()
        writer.writerows(flattened)

    payload: dict[str, Any] = {
        "protocol": {
            "structural_status": "PASS",
            "performance_is_exit_gate": False,
            "baseline_source": "ba7ee094 + cuda132-dependent-template-compat",
            "final_source": "75186dde + cuda132-dependent-template-compat",
            "baseline_raw_processes": 30,
            "final_excluded_complete_warmups": 1,
            "final_raw_measured_processes": 30,
            "ordered_m": list(EXPECTED_M),
            "num_tests": NUM_TESTS,
            "primary_metric": "rank0 nvfp4= median30",
            "secondary_metrics": ["mean_rank median30", "max_rank median30"],
            "outlier_policy": "none; no retry, deletion, replacement, or best-of-N",
            "median_ci": f"{MEDIAN_CI_COVERAGE_PCT:.4f}% X10-X21",
        },
        "shape": {
            "hidden": MIMO.hidden,
            "intermediate_hidden": MIMO.intermediate_hidden,
            "num_experts": MIMO.num_experts,
            "num_topk": MIMO.num_topk,
            "num_ranks": 8,
        },
        "equal_weight_geomean_latency_change_pct": geomean,
        "results": results,
    }
    (final_root / "comparison_baseline_vs_final_30.json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    report = markdown_report(results, geomean)
    (final_root / "comparison_baseline_vs_final_30.md").write_text(
        report, encoding="utf-8"
    )
    (final_root / "structural_status.txt").write_text("PASS\n", encoding="utf-8")
    (final_root / "performance_status.txt").write_text(
        "REPORTED_NOT_GATED\n", encoding="utf-8"
    )
    print(report, end="")
    return payload


def expect_failure(function: Callable[[], Any], contains: str) -> None:
    try:
        function()
    except (OSError, ValueError, json.JSONDecodeError) as error:
        if contains not in str(error):
            raise AssertionError(
                f"expected failure containing {contains!r}, got {error!r}"
            ) from error
    else:
        raise AssertionError(f"expected failure containing {contains!r}")


def synthetic_log(
    model: str,
    *,
    round_index: int = 0,
    factor: float = 1.0,
    bad_recv: bool = False,
    compile_marker: bool = False,
) -> str:
    spec = MODEL_SPECS[model]
    lines = [spec.header]
    if compile_marker:
        lines.append("Running NVCC command: synthetic")
    for index, m in enumerate(EXPECTED_M):
        base = 400.0 + 0.8 * m
        jitter = 1.0 + (round_index - 15.5) * 0.0002
        rank0 = base * factor * jitter
        recv = spec.recv_by_m[m] + (1 if bad_recv and index == 0 else 0)
        lines.append(
            f" tokens={m:4d}  recv={recv:5d}  experts={spec.touched_by_m[m]:4d}  "
            f"nvfp4={rank0:7.1f}us mean_rank={rank0 + 0.4:7.1f}us "
            f"max_rank={rank0 + 0.9:7.1f}us ( 10.0TF, 1000GB/s)  (rank0)"
        )
    return "\n".join(lines) + "\n"


def write_synthetic_run(
    root: Path,
    expected: list[tuple[str, str, str, int | None]],
    *,
    mimo_factor: float,
) -> None:
    ledger_rows: list[dict[str, str]] = []
    for label, kind, model, round_index in expected:
        path = root / f"{label}.log"
        path.write_text(
            synthetic_log(
                model,
                round_index=round_index or 0,
                factor=mimo_factor if model == "mimo" and kind == "measured" else 1.0,
            ),
            encoding="utf-8",
        )
        ledger_rows.append(
            {
                "label": label,
                "kind": kind,
                "started_at": "2026-07-16T00:00:00-07:00",
                "ended_at": "2026-07-16T00:00:01-07:00",
                "exit_code": "0",
                "log_sha256": sha256_file(path),
            }
        )
    with (root / "process_ledger.tsv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=LEDGER_HEADER, delimiter="\t")
        writer.writeheader()
        writer.writerows(ledger_rows)


def self_test() -> None:
    assert len(expected_baseline_entries()) == 93
    assert len(expected_final_entries()) == 31
    assert baseline_round_order(1) == ("flash", "pro", "mimo")
    assert baseline_round_order(2) == ("pro", "mimo", "flash")
    assert baseline_round_order(3) == ("mimo", "flash", "pro")

    with tempfile.TemporaryDirectory() as temp:
        path = Path(temp) / "mimo.log"
        path.write_text(synthetic_log("mimo"), encoding="utf-8")
        assert len(parse_benchmark_log(path)) == 11
        path.write_text(synthetic_log("mimo", bad_recv=True), encoding="utf-8")
        expect_failure(lambda: parse_benchmark_log(path), "recv mismatch")
        path.write_text(synthetic_log("mimo", compile_marker=True), encoding="utf-8")
        expect_failure(
            lambda: parse_benchmark_log(path, forbid_compile=True),
            "unexpectedly compiled JIT",
        )

    with tempfile.TemporaryDirectory() as temp:
        root = Path(temp)
        baseline = root / "baseline"
        final = root / "final"
        baseline.mkdir()
        final.mkdir()
        write_synthetic_run(
            baseline, expected_baseline_entries(), mimo_factor=1.0
        )
        # Deliberately slower: performance must not become a structural failure.
        write_synthetic_run(final, expected_final_entries(), mimo_factor=1.05)
        with (root / "stdout.txt").open("w", encoding="utf-8") as sink:
            with contextlib.redirect_stdout(sink):
                payload = summarize(
                    final, baseline, enforce_pinned_baseline=False
                )
        assert payload["protocol"]["structural_status"] == "PASS"
        assert payload["equal_weight_geomean_latency_change_pct"]["rank0"] > 4.0
        assert (final / "structural_status.txt").read_text() == "PASS\n"
        assert (final / "performance_status.txt").read_text() == "REPORTED_NOT_GATED\n"
        assert (final / "individual_baseline_vs_final_30.csv").read_text().count("\n") == 661
        assert (final / "comparison_baseline_vs_final_30.csv").read_text().count("\n") == 12

        target = final / "round_30_mimo.log"
        saved = target.read_text(encoding="utf-8")
        target.write_text(saved + "mutation\n", encoding="utf-8")
        expect_failure(lambda: load_final_runs(final), "log hash mismatch")
        target.write_text(saved, encoding="utf-8")

        extra = final / "round_31_mimo.log"
        extra.write_text(synthetic_log("mimo"), encoding="utf-8")
        expect_failure(lambda: load_final_runs(final), "protocol log set mismatch")
        extra.unlink()

        ledger = final / "process_ledger.tsv"
        saved_ledger = ledger.read_text(encoding="utf-8")
        lines = saved_ledger.splitlines()
        lines[2], lines[3] = lines[3], lines[2]
        ledger.write_text("\n".join(lines) + "\n", encoding="utf-8")
        expect_failure(lambda: load_final_runs(final), "process order/class mismatch")
        ledger.write_text(saved_ledger, encoding="utf-8")
    print("FINAL_MIMO_COMPARISON_PARSER_SELF_TEST_PASS")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Strict final MiMo versus frozen baseline raw-log comparison"
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--run-dir", type=Path)
    group.add_argument("--validate-log", type=Path)
    group.add_argument("--validate-baseline", action="store_true")
    group.add_argument("--self-test", action="store_true")
    parser.add_argument("--baseline-dir", type=Path, default=DEFAULT_BASELINE_DIR)
    parser.add_argument("--allow-compile", action="store_true")
    args = parser.parse_args()

    try:
        if args.self_test:
            self_test()
        elif args.validate_log is not None:
            rows = parse_benchmark_log(
                args.validate_log, forbid_compile=not args.allow_compile
            )
            print(f"LOG_VALIDATION_PASS model=mimo rows={len(rows)} path={args.validate_log}")
        elif args.validate_baseline:
            runs = load_baseline_runs(
                args.baseline_dir, enforce_pinned_identity=True
            )
            print(f"PINNED_BASELINE_RAW_VALIDATION_PASS processes={len(runs)}")
        else:
            summarize(args.run_dir, args.baseline_dir)
    except (OSError, ValueError, json.JSONDecodeError, AssertionError) as error:
        print(f"VALIDATION_ERROR: {error}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
