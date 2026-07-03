#!/usr/bin/env python3
"""Strictly summarize per-rank JSON emitted by h200_fp8_matrix_runner.py."""

import argparse
import csv
import json
import math
import re
from collections import defaultdict
from pathlib import Path


def geometric_mean(values):
    return math.exp(sum(math.log(v) for v in values) / len(values))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--logs", type=Path, required=True)
    parser.add_argument("--csv", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()

    runs = []
    errors = []
    # The driver log tees all child output and therefore contains many runs;
    # consume only one-run leaf logs here.
    for path in sorted(args.logs.glob("baseline_*_M*_S*_L*_*_N*.log")):
        text = path.read_text(errors="replace")
        meta_match = re.search(r"^RUN_META_JSON (.+)$", text, re.MULTILINE)
        if not meta_match:
            continue
        meta = json.loads(meta_match.group(1))
        exits = re.findall(r"^RUN_EXIT=(\d+)$", text, re.MULTILINE)
        stats = [
            json.loads(match)
            for match in re.findall(r"^BENCH_STAT_JSON (.+)$", text, re.MULTILINE)
        ]
        routes = sorted(
            (int(rank), int(tokens), int(recv), int(experts))
            for rank, tokens, recv, experts in re.findall(
                r"MATRIX_RANK=(\d+)\s+tokens=\s*(\d+)\s+recv=\s*(\d+)\s+experts=\s*(\d+)",
                text,
            )
        )
        ranks = sorted(row["rank"] for row in stats)
        if exits != ["0"] or ranks != list(range(8)) or len(routes) != 8:
            errors.append(
                f"{path.name}: exits={exits}, ranks={ranks}, routes={len(routes)}"
            )
            continue
        if any(row["num_samples"] != meta["num_tests"] for row in stats):
            errors.append(f"{path.name}: incomplete sample count")
            continue
        runs.append(
            {
                **meta,
                "rank0_us": next(row["returned_us"] for row in stats if row["rank"] == 0),
                "max_rank_us": max(row["returned_us"] for row in stats),
                "min_rank_us": min(row["returned_us"] for row in stats),
                "routes": routes,
                "log": path.name,
            }
        )

    if errors:
        raise SystemExit("invalid logs:\n" + "\n".join(errors))

    grouped = defaultdict(list)
    for row in runs:
        grouped[(row["shape"], row["m"], row["impl"])].append(row)

    final = []
    for shape in ("flash", "pro"):
        for m in (8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192):
            ours = grouped.get((shape, m, "ours"), [])
            pr = grouped.get((shape, m, "pr323"), [])
            if not ours or not pr:
                continue
            route_signatures = {tuple(row["routes"]) for row in ours + pr}
            if len(route_signatures) != 1:
                errors.append(f"route mismatch: shape={shape}, M={m}")
                continue
            ours_max = geometric_mean([row["max_rank_us"] for row in ours])
            pr_max = geometric_mean([row["max_rank_us"] for row in pr])
            ours_r0 = geometric_mean([row["rank0_us"] for row in ours])
            pr_r0 = geometric_mean([row["rank0_us"] for row in pr])
            final.append(
                {
                    "shape": shape,
                    "m": m,
                    "ours_max_rank_us": ours_max,
                    "pr323_max_rank_us": pr_max,
                    "ours_vs_pr323_pct": (ours_max / pr_max - 1) * 100,
                    "ours_rank0_us": ours_r0,
                    "pr323_rank0_us": pr_r0,
                    "ours_observations": len(ours),
                    "pr323_observations": len(pr),
                }
            )

    if errors:
        raise SystemExit("invalid comparison:\n" + "\n".join(errors))

    args.csv.parent.mkdir(parents=True, exist_ok=True)
    with args.csv.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(final[0]) if final else [])
        if final:
            writer.writeheader()
            writer.writerows(final)

    lines = [
        "# H200 FP8 baseline",
        "",
        "| Shape | M | ours max-rank (us) | PR323 max-rank (us) | ours gap | ours rank0 (us) | PR323 rank0 (us) |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in final:
        lines.append(
            f"| {row['shape']} | {row['m']} | {row['ours_max_rank_us']:.1f} | "
            f"{row['pr323_max_rank_us']:.1f} | {row['ours_vs_pr323_pct']:+.2f}% | "
            f"{row['ours_rank0_us']:.1f} | {row['pr323_rank0_us']:.1f} |"
        )
    lines += ["", f"Complete points: {len(final)}/22; valid runs: {len(runs)}."]
    args.report.write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
