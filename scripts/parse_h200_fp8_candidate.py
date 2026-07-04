#!/usr/bin/env python3
"""Validate and summarize one H200 FP8 candidate campaign."""

import argparse
import csv
import json
import re
import statistics
from collections import defaultdict
from pathlib import Path


MS = (8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192)


def extract_marked_json(text: str, marker: str):
    decoder = json.JSONDecoder()
    rows = []
    start = 0
    while True:
        marker_pos = text.find(marker, start)
        if marker_pos < 0:
            return rows
        payload_pos = marker_pos + len(marker)
        payload = text[payload_pos:].lstrip()
        row, consumed = decoder.raw_decode(payload)
        rows.append(row)
        start = payload_pos + (len(text[payload_pos:]) - len(payload)) + consumed


def parse_run(path: Path, candidate: str):
    text = path.read_text(errors="replace")
    metas = extract_marked_json(text, "RUN_META_JSON ")
    if len(metas) != 1 or metas[0].get("candidate") != candidate:
        return None
    meta = metas[0]
    exits = re.findall(r"^RUN_EXIT=(\d+)$", text, re.MULTILINE)
    stats = extract_marked_json(text, "BENCH_STAT_JSON ")
    routes = sorted(
        (int(rank), int(tokens), int(recv), int(experts))
        for rank, tokens, recv, experts in re.findall(
            r"MATRIX_RANK=(\d+)\s+tokens=\s*(\d+)\s+recv=\s*(\d+)\s+experts=\s*(\d+)",
            text,
        )
    )
    ranks = sorted(row["rank"] for row in stats)
    if exits != ["0"]:
        raise ValueError(f"{path.name}: exits={exits}")
    if ranks != list(range(8)):
        raise ValueError(f"{path.name}: ranks={ranks}")
    if len(routes) != 8:
        raise ValueError(f"{path.name}: route rows={len(routes)}")
    if any(row["num_samples"] != meta["num_tests"] for row in stats):
        raise ValueError(f"{path.name}: incomplete sample count")
    return {
        **meta,
        "rank0_us": next(row["returned_us"] for row in stats if row["rank"] == 0),
        "max_rank_us": max(row["returned_us"] for row in stats),
        "min_rank_us": min(row["returned_us"] for row in stats),
        "routes": routes,
        "log": path.name,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--logs", type=Path, required=True)
    parser.add_argument("--candidate", required=True)
    parser.add_argument("--csv", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--expected-observations", type=int, default=3)
    args = parser.parse_args()

    runs = []
    errors = []
    for path in sorted(args.logs.glob("*.log")):
        try:
            run = parse_run(path, args.candidate)
            if run is not None:
                runs.append(run)
        except (ValueError, json.JSONDecodeError) as exc:
            errors.append(str(exc))
    if errors:
        raise SystemExit("invalid logs:\n" + "\n".join(errors))

    grouped = defaultdict(list)
    for row in runs:
        grouped[(row["shape"], row["m"], row["impl"])].append(row)

    final = []
    for shape in ("flash", "pro"):
        for m in MS:
            ours = grouped.get((shape, m, "ours"), [])
            pr323 = grouped.get((shape, m, "pr323"), [])
            if not ours and not pr323:
                continue
            if len(ours) != args.expected_observations:
                errors.append(
                    f"shape={shape} M={m}: ours observations={len(ours)}"
                )
                continue
            observation_ids = [row["observation"] for row in ours]
            if sorted(observation_ids) != list(
                range(1, args.expected_observations + 1)
            ):
                errors.append(
                    f"shape={shape} M={m}: ours observation ids={observation_ids}"
                )
                continue
            row = {
                "shape": shape,
                "m": m,
                "ours_max_rank_us": statistics.median(
                    item["max_rank_us"] for item in ours
                ),
                "ours_rank0_us": statistics.median(
                    item["rank0_us"] for item in ours
                ),
                "ours_observations_us": ";".join(
                    f"{item['max_rank_us']:.3f}"
                    for item in sorted(ours, key=lambda item: item["observation"])
                ),
                "pr323_max_rank_us": "",
                "pr323_rank0_us": "",
                "pr323_observations_us": "",
                "ours_vs_pr323_pct": "",
            }
            if pr323:
                if len(pr323) != args.expected_observations:
                    errors.append(
                        f"shape={shape} M={m}: PR323 observations={len(pr323)}"
                    )
                    continue
                route_signatures = {tuple(item["routes"]) for item in ours + pr323}
                if len(route_signatures) != 1:
                    errors.append(f"shape={shape} M={m}: route mismatch")
                    continue
                pr_max = statistics.median(item["max_rank_us"] for item in pr323)
                pr_rank0 = statistics.median(item["rank0_us"] for item in pr323)
                row.update(
                    pr323_max_rank_us=pr_max,
                    pr323_rank0_us=pr_rank0,
                    pr323_observations_us=";".join(
                        f"{item['max_rank_us']:.3f}"
                        for item in sorted(
                            pr323, key=lambda item: item["observation"]
                        )
                    ),
                    ours_vs_pr323_pct=(row["ours_max_rank_us"] / pr_max - 1) * 100,
                )
            final.append(row)

    if errors:
        raise SystemExit("invalid campaign:\n" + "\n".join(errors))

    args.csv.parent.mkdir(parents=True, exist_ok=True)
    with args.csv.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(final[0]) if final else [])
        if final:
            writer.writeheader()
            writer.writerows(final)

    lines = [
        f"# {args.candidate}",
        "",
        "| Shape | M | ours observations (us) | ours median (us) | PR323 observations (us) | PR323 median (us) | gap |",
        "|---|---:|---|---:|---|---:|---:|",
    ]
    gated = []
    for row in final:
        pr_median = row["pr323_max_rank_us"]
        gap = row["ours_vs_pr323_pct"]
        lines.append(
            f"| {row['shape']} | {row['m']} | {row['ours_observations_us']} | "
            f"{row['ours_max_rank_us']:.1f} | {row['pr323_observations_us'] or '—'} | "
            f"{f'{pr_median:.1f}' if pr_median != '' else '—'} | "
            f"{f'{gap:+.2f}%' if gap != '' else 'not gated'} |"
        )
        if row["m"] >= 128 and row["m"] != 512 and gap != "":
            gated.append(row)
    failures = [row for row in gated if row["ours_vs_pr323_pct"] >= 0]
    lines += [
        "",
        f"PR323 gate: {'PASS' if gated and not failures else 'FAIL'} "
        f"({len(gated) - len(failures)}/{len(gated)} in-scope points faster).",
        f"Valid leaf runs: {len(runs)}.",
    ]
    args.report.write_text("\n".join(lines) + "\n")
    print("\n".join(lines))
    if failures:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
