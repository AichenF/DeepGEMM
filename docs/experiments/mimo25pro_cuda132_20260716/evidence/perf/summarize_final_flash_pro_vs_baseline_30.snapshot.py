#!/usr/bin/env python3
"""Strict Flash/Pro final-kernel comparison against the frozen Aichen baseline.

The baseline and final values are rebuilt from raw logs.  Performance never
controls the exit status; malformed, missing, extra, retried, or mutated
evidence does.  Shared parsing primitives are imported from the sealed MiMo
comparison parser and verified by SHA-256 before import.
"""

from __future__ import annotations

import argparse
import contextlib
import csv
import hashlib
import importlib.util
import json
import math
import statistics
import sys
import tempfile
from pathlib import Path
from typing import Any, Callable


COMMON_SHA256 = "7b8ca133b0bad86a032b50dd75022d8f519f31d1dea756aeba778990bed51ee9"
COMMON_FALLBACK = Path(
    "/home/xuechengw/MegaMoe/artifacts/final_mimo_cuda132_20260716/"
    "summarize_final_mimo_vs_baseline_30.py"
)
MODELS = ("flash", "pro")
IMAGEPERF_RANK0_US = {
    "flash": (226.6, 237.6, 249.6, 263.5, 260.4, 292.0,
              481.2, 813.6, 1440.5, 2684.3, 5215.5),
    "pro": (594.6, 760.0, 809.3, 831.4, 878.4, 886.1,
            1345.9, 2143.3, 3763.6, 6964.8, 13336.8),
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_common() -> Any:
    local = Path(__file__).with_name("common_final_parser.py")
    path = local if local.is_file() else COMMON_FALLBACK
    if not path.is_file():
        raise RuntimeError(f"missing sealed common parser: {path}")
    observed = sha256_file(path)
    if observed != COMMON_SHA256:
        raise RuntimeError(
            f"common parser hash mismatch: expected {COMMON_SHA256}, got {observed}"
        )
    spec = importlib.util.spec_from_file_location("sealed_final_parser_common", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import common parser: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


common = load_common()
EXPECTED_M = common.EXPECTED_M
EXPECTED_ROUNDS = common.EXPECTED_ROUNDS
NUM_TESTS = common.NUM_TESTS
MEDIAN_CI_COVERAGE_PCT = common.MEDIAN_CI_COVERAGE_PCT
DEFAULT_BASELINE_DIR = common.DEFAULT_BASELINE_DIR


def fail(message: str) -> None:
    raise ValueError(message)


def final_round_order(round_index: int) -> tuple[str, str]:
    if not 1 <= round_index <= EXPECTED_ROUNDS:
        fail(f"round outside 1..30: {round_index}")
    return MODELS if round_index % 2 else tuple(reversed(MODELS))


def expected_final_entries() -> list[tuple[str, str, str, int | None]]:
    entries = [
        (f"warmup_{model}", "excluded_warmup", model, None)
        for model in MODELS
    ]
    for round_index in range(1, EXPECTED_ROUNDS + 1):
        for model in final_round_order(round_index):
            entries.append(
                (f"round_{round_index:02d}_{model}", "measured", model, round_index)
            )
    return entries


def load_baseline_runs(
    root: Path, model: str, *, enforce_pinned_identity: bool
) -> list[list[dict[str, Any]]]:
    root = root.resolve()
    if model not in MODELS:
        fail(f"unsupported model: {model}")
    if enforce_pinned_identity:
        common.validate_pinned_baseline(root)
    entries = common.read_and_validate_ledger(root, common.expected_baseline_entries())
    runs: list[list[dict[str, Any]]] = []
    next_round = 1
    for label, kind, entry_model, round_index, log_path, ledger_sha in entries:
        if entry_model != model:
            continue
        rows = common.parse_benchmark_log(
            log_path, model, forbid_compile=(kind == "measured")
        )
        if sha256_file(log_path) != ledger_sha:
            fail(f"{log_path}: log changed while being parsed")
        if kind == "measured":
            if round_index != next_round:
                fail(f"{label}: expected {model} round {next_round}, got {round_index}")
            runs.append(rows)
            next_round += 1
    if len(runs) != EXPECTED_ROUNDS:
        fail(f"baseline {model}: expected 30 processes, got {len(runs)}")
    return runs


def load_final_runs(root: Path) -> dict[str, list[list[dict[str, Any]]]]:
    root = root.resolve()
    entries = common.read_and_validate_ledger(root, expected_final_entries())
    runs: dict[str, list[list[dict[str, Any]]]] = {model: [] for model in MODELS}
    warmups: set[str] = set()
    for label, kind, model, round_index, log_path, ledger_sha in entries:
        rows = common.parse_benchmark_log(
            log_path, model, forbid_compile=(kind == "measured")
        )
        if sha256_file(log_path) != ledger_sha:
            fail(f"{log_path}: log changed while being parsed")
        if kind == "excluded_warmup":
            if model in warmups or round_index is not None:
                fail(f"final: duplicate or misclassified {model} warmup")
            warmups.add(model)
        else:
            if round_index != len(runs[model]) + 1:
                fail(f"{label}: non-contiguous {model} round")
            runs[model].append(rows)
    if warmups != set(MODELS):
        fail(f"final: warmup set mismatch: {sorted(warmups)}")
    for model in MODELS:
        if len(runs[model]) != EXPECTED_ROUNDS:
            fail(f"final {model}: expected 30 processes, got {len(runs[model])}")
    return runs


def layout_for(model: str, source: str, m: int) -> str:
    if m >= 2048:
        return "BN128-split"
    if source == "final" and model == "flash":
        return "BN256-grouped"
    return "BN256-standard"


def metric_distributions(
    runs: list[list[dict[str, Any]]], row_index: int
) -> dict[str, dict[str, float]]:
    return common.metric_distributions(runs, row_index)


def make_results(
    model: str,
    baseline_runs: list[list[dict[str, Any]]],
    final_runs: list[list[dict[str, Any]]],
) -> list[dict[str, Any]]:
    spec = common.MODEL_SPECS[model]
    results: list[dict[str, Any]] = []
    for row_index, m in enumerate(EXPECTED_M):
        baseline = metric_distributions(baseline_runs, row_index)
        final = metric_distributions(final_runs, row_index)
        imageperf = IMAGEPERF_RANK0_US[model][row_index]
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
                "model": model,
                "display_name": spec.display_name,
                "m": m,
                "recv": spec.recv_by_m[m],
                "touched_experts": spec.touched_by_m[m],
                "baseline_layout": layout_for(model, "baseline", m),
                "final_layout": layout_for(model, "final", m),
                "imageperf_rank0_us": imageperf,
                "baseline_vs_imageperf_rank0_median_delta_pct": 100.0
                * (baseline["rank0"]["median"] / imageperf - 1),
                "final_vs_imageperf_rank0_median_delta_pct": 100.0
                * (final["rank0"]["median"] / imageperf - 1),
                "baseline": baseline,
                "final": final,
                "final_vs_baseline_delta_pct": deltas,
                "baseline_over_final_speedup": speedups,
            }
        )
    return results


def geomean_ratio_delta(
    results: list[dict[str, Any]], numerator: Callable[[dict[str, Any]], float],
    denominator: Callable[[dict[str, Any]], float],
) -> float:
    ratios = [numerator(row) / denominator(row) for row in results]
    return 100.0 * (math.exp(statistics.mean(math.log(value) for value in ratios)) - 1)


def geomeans(results: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "final_vs_baseline": {
            metric: geomean_ratio_delta(
                results,
                lambda row, metric=metric: row["final"][metric]["median"],
                lambda row, metric=metric: row["baseline"][metric]["median"],
            )
            for metric in ("rank0", "mean_rank", "max_rank")
        },
        "baseline_rank0_vs_imageperf": geomean_ratio_delta(
            results,
            lambda row: row["baseline"]["rank0"]["median"],
            lambda row: row["imageperf_rank0_us"],
        ),
        "final_rank0_vs_imageperf": geomean_ratio_delta(
            results,
            lambda row: row["final"]["rank0"]["median"],
            lambda row: row["imageperf_rank0_us"],
        ),
    }


def write_individual_csv(
    output_root: Path,
    baseline_by_model: dict[str, list[list[dict[str, Any]]]],
    final_by_model: dict[str, list[list[dict[str, Any]]]],
) -> None:
    fields = [
        "model", "source", "round", "log", "m", "recv", "experts", "layout",
        "rank0_us", "mean_rank_us", "max_rank_us", "tflops", "hbm_gbs",
    ]
    path = output_root / "individual_flash_pro_baseline_vs_final_30.csv"
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for model in MODELS:
            for source, runs in (
                ("baseline", baseline_by_model[model]),
                ("final", final_by_model[model]),
            ):
                for round_index, rows in enumerate(runs, 1):
                    for row in rows:
                        writer.writerow(
                            {
                                "model": model,
                                "source": source,
                                "round": round_index,
                                "log": f"round_{round_index:02d}_{model}.log",
                                "layout": layout_for(model, source, row["m"]),
                                **row,
                            }
                        )


def flat_rows(results_by_model: dict[str, list[dict[str, Any]]]) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    for model in MODELS:
        for result in results_by_model[model]:
            flat: dict[str, Any] = {
                key: result[key]
                for key in (
                    "model", "display_name", "m", "recv", "touched_experts",
                    "baseline_layout", "final_layout", "imageperf_rank0_us",
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
            output.append(flat)
    return output


def markdown_report(
    results_by_model: dict[str, list[dict[str, Any]]],
    geomean_by_model: dict[str, dict[str, Any]],
) -> str:
    lines = [
        "# CUDA 13.2 Flash/Pro final kernel versus frozen Aichen baseline",
        "",
        (
            "ImagePerf, baseline and final use the same eleven M values. Baseline "
            "and final statistics are independently recomputed from 30 complete "
            "process logs per model; each process uses 8 ranks and `--num-tests 20`."
        ),
        (
            "All times are microseconds. Negative delta means final is faster. "
            "Max-rank is each process's slowest rank, followed by median across 30 "
            "processes; it is not the sum of eight GPU times."
        ),
    ]
    for model in MODELS:
        spec = common.MODEL_SPECS[model]
        lines.extend(
            [
                "",
                f"## {spec.display_name} ({spec.hidden}/{spec.intermediate_hidden}/"
                f"{spec.num_experts}/top-k{spec.num_topk})",
                "",
                "| M | recv | layout baseline → final | ImagePerf rank0 | rank0 baseline | baseline vs image | rank0 final | final vs image | final vs baseline | baseline P10–P90 | final P10–P90 | baseline CI X10–X21 | final CI X10–X21 | max median baseline → final | max delta | mean median baseline → final | mean delta |",
                "|--:|--:|:--|--:|--:|--:|--:|--:|--:|:--|:--|:--|:--|:--|--:|:--|--:|",
            ]
        )
        for row in results_by_model[model]:
            baseline = row["baseline"]
            final = row["final"]
            delta = row["final_vs_baseline_delta_pct"]
            lines.append(
                f"| {row['m']} | {row['recv']} | {row['baseline_layout']} → "
                f"{row['final_layout']} | {row['imageperf_rank0_us']:.1f} | "
                f"{baseline['rank0']['median']:.2f} | "
                f"{row['baseline_vs_imageperf_rank0_median_delta_pct']:+.2f}% | "
                f"{final['rank0']['median']:.2f} | "
                f"{row['final_vs_imageperf_rank0_median_delta_pct']:+.2f}% | "
                f"{delta['rank0']:+.2f}% | "
                f"{baseline['rank0']['p10']:.2f}–{baseline['rank0']['p90']:.2f} | "
                f"{final['rank0']['p10']:.2f}–{final['rank0']['p90']:.2f} | "
                f"{baseline['rank0']['median_ci_x10_low']:.1f}–"
                f"{baseline['rank0']['median_ci_x21_high']:.1f} | "
                f"{final['rank0']['median_ci_x10_low']:.1f}–"
                f"{final['rank0']['median_ci_x21_high']:.1f} | "
                f"{baseline['max_rank']['median']:.2f} → "
                f"{final['max_rank']['median']:.2f} | {delta['max_rank']:+.2f}% | "
                f"{baseline['mean_rank']['median']:.2f} → "
                f"{final['mean_rank']['median']:.2f} | {delta['mean_rank']:+.2f}% |"
            )
        gm = geomean_by_model[model]
        lines.extend(
            [
                "",
                "Equal-weight 11-point geometric mean latency change:",
                "",
                f"- final vs baseline rank0: {gm['final_vs_baseline']['rank0']:+.3f}%",
                f"- final vs baseline mean-rank: {gm['final_vs_baseline']['mean_rank']:+.3f}%",
                f"- final vs baseline max-rank: {gm['final_vs_baseline']['max_rank']:+.3f}%",
                f"- baseline rank0 vs ImagePerf: {gm['baseline_rank0_vs_imageperf']:+.3f}%",
                f"- final rank0 vs ImagePerf: {gm['final_rank0_vs_imageperf']:+.3f}%",
            ]
        )
    lines.extend(
        [
            "",
            (
                f"The machine-readable CSV/JSON retain rank0, mean-rank and max-rank "
                f"median/P10/P90/min/max and the {MEDIAN_CI_COVERAGE_PCT:.4f}% "
                "distribution-free X10–X21 median interval."
            ),
            "",
            (
                "Baseline uses stock auto BN256 through M=1024 and BN128 from M=2048. "
                "Final Flash uses BN256 grouped-nibble through M=1024 and BN128 split "
                "afterward; final Pro remains BN256 standard then BN128 split."
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
    if not final_root.is_dir() or not baseline_root.is_dir():
        fail("final or baseline directory does not exist")
    baseline_by_model = {
        model: load_baseline_runs(
            baseline_root, model, enforce_pinned_identity=enforce_pinned_baseline
        )
        for model in MODELS
    }
    final_by_model = load_final_runs(final_root)
    results_by_model = {
        model: make_results(model, baseline_by_model[model], final_by_model[model])
        for model in MODELS
    }
    geomean_by_model = {
        model: geomeans(results_by_model[model]) for model in MODELS
    }

    write_individual_csv(final_root, baseline_by_model, final_by_model)
    flattened = flat_rows(results_by_model)
    csv_path = final_root / "comparison_flash_pro_baseline_vs_final_30.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(flattened[0]))
        writer.writeheader()
        writer.writerows(flattened)

    payload: dict[str, Any] = {
        "protocol": {
            "structural_status": "PASS",
            "performance_is_exit_gate": False,
            "baseline_source": "ba7ee094 + cuda132-dependent-template-compat",
            "final_source": "75186dd + 16-site cuda132-dependent-template-compat",
            "baseline_raw_processes_per_model": 30,
            "final_excluded_warmups": 2,
            "final_raw_measured_processes_per_model": 30,
            "final_measurement_order": "FP/PF alternating",
            "ordered_m": list(EXPECTED_M),
            "num_tests": NUM_TESTS,
            "primary_metric": "rank0 nvfp4= median30",
            "secondary_metrics": ["mean-rank median30", "max-rank median30"],
            "outlier_policy": "none; no retry, deletion, replacement, or best-of-N",
            "median_ci": f"{MEDIAN_CI_COVERAGE_PCT:.4f}% X10-X21",
        },
        "models": {
            model: {
                "shape": {
                    "hidden": common.MODEL_SPECS[model].hidden,
                    "intermediate_hidden": common.MODEL_SPECS[model].intermediate_hidden,
                    "num_experts": common.MODEL_SPECS[model].num_experts,
                    "num_topk": common.MODEL_SPECS[model].num_topk,
                    "num_ranks": 8,
                },
                "equal_weight_geomean_latency_change_pct": geomean_by_model[model],
                "results": results_by_model[model],
            }
            for model in MODELS
        },
    }
    (final_root / "comparison_flash_pro_baseline_vs_final_30.json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    report = markdown_report(results_by_model, geomean_by_model)
    (final_root / "comparison_flash_pro_baseline_vs_final_30.md").write_text(
        report, encoding="utf-8"
    )
    (final_root / "structural_status.txt").write_text("PASS\n", encoding="utf-8")
    (final_root / "performance_status.txt").write_text(
        "REPORTED_NOT_GATED\n", encoding="utf-8"
    )
    print(report, end="")
    return payload


def write_synthetic_final(root: Path, factor: float) -> None:
    ledger_rows: list[dict[str, str]] = []
    for label, kind, model, round_index in expected_final_entries():
        path = root / f"{label}.log"
        path.write_text(
            common.synthetic_log(
                model,
                round_index=round_index or 0,
                factor=factor if kind == "measured" else 1.0,
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
        writer = csv.DictWriter(handle, fieldnames=common.LEDGER_HEADER, delimiter="\t")
        writer.writeheader()
        writer.writerows(ledger_rows)


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


def self_test() -> None:
    assert len(expected_final_entries()) == 62
    assert final_round_order(1) == ("flash", "pro")
    assert final_round_order(2) == ("pro", "flash")
    with tempfile.TemporaryDirectory() as temp:
        root = Path(temp)
        baseline = root / "baseline"
        final = root / "final"
        baseline.mkdir()
        final.mkdir()
        common.write_synthetic_run(
            baseline, common.expected_baseline_entries(), mimo_factor=1.0
        )
        write_synthetic_final(final, 1.05)
        with (root / "stdout.txt").open("w", encoding="utf-8") as sink:
            with contextlib.redirect_stdout(sink):
                payload = summarize(
                    final, baseline, enforce_pinned_baseline=False
                )
        for model in MODELS:
            assert payload["models"][model][
                "equal_weight_geomean_latency_change_pct"
            ]["final_vs_baseline"]["rank0"] > 4.0
        individual = final / "individual_flash_pro_baseline_vs_final_30.csv"
        assert individual.read_text(encoding="utf-8").count("\n") == 1321
        summary_csv = final / "comparison_flash_pro_baseline_vs_final_30.csv"
        assert summary_csv.read_text(encoding="utf-8").count("\n") == 23
        target = final / "round_30_pro.log"
        saved = target.read_text(encoding="utf-8")
        target.write_text(saved + "mutation\n", encoding="utf-8")
        expect_failure(lambda: load_final_runs(final), "log hash mismatch")
    print("FINAL_FLASH_PRO_COMPARISON_PARSER_SELF_TEST_PASS")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Strict final Flash/Pro versus frozen baseline comparison"
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--run-dir", type=Path)
    group.add_argument("--validate-log", type=Path)
    group.add_argument("--validate-baseline", action="store_true")
    group.add_argument("--self-test", action="store_true")
    parser.add_argument("--model", choices=MODELS)
    parser.add_argument("--baseline-dir", type=Path, default=DEFAULT_BASELINE_DIR)
    parser.add_argument("--allow-compile", action="store_true")
    args = parser.parse_args()
    try:
        if args.self_test:
            self_test()
        elif args.validate_log is not None:
            if args.model is None:
                fail("--validate-log requires --model")
            rows = common.parse_benchmark_log(
                args.validate_log,
                args.model,
                forbid_compile=not args.allow_compile,
            )
            print(
                f"LOG_VALIDATION_PASS model={args.model} rows={len(rows)} "
                f"path={args.validate_log}"
            )
        elif args.validate_baseline:
            for model in MODELS:
                runs = load_baseline_runs(
                    args.baseline_dir, model, enforce_pinned_identity=True
                )
                print(f"PINNED_BASELINE_RAW_VALIDATION_PASS model={model} processes={len(runs)}")
        else:
            summarize(args.run_dir, args.baseline_dir)
    except (OSError, RuntimeError, ValueError, json.JSONDecodeError, AssertionError) as error:
        print(f"VALIDATION_ERROR: {error}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
