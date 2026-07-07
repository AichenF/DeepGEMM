#!/usr/bin/env python3
"""Strict parser for the H200 DeepEP low-latency masked-FP8 matrix."""

import argparse
import csv
import json
import re
import statistics
from pathlib import Path


SHAPES = ("flash", "pro", "mimo_pro")
MS = (8, 16, 32, 64, 128)


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


def parse_leaf(path: Path, candidate: str, observations: int, samples: int):
    text = path.read_text(errors="replace")
    metas = extract_marked_json(text, "RUN_META_JSON ")
    if len(metas) != 1 or metas[0].get("candidate") != candidate:
        return None
    meta = metas[0]
    if meta.get("impl") != "deepep_low_latency_masked_fp8":
        raise ValueError(f"{path.name}: implementation mismatch")
    if int(meta.get("cap", -1)) != int(meta["m"]):
        raise ValueError(f"{path.name}: cap must match M")
    exits = re.findall(r"^RUN_EXIT=(\d+)$", text, re.MULTILINE)
    stats = extract_marked_json(text, "LOW_LATENCY_STAT_JSON ")
    obs = extract_marked_json(text, "LOW_LATENCY_OBSERVATION_JSON ")
    if exits != ["0"]:
        raise ValueError(f"{path.name}: exits={exits}")
    if len(stats) != observations * 8:
        raise ValueError(f"{path.name}: rank stats={len(stats)}")
    if len(obs) != observations:
        raise ValueError(f"{path.name}: observations={len(obs)}")

    for observation in range(1, observations + 1):
        rows = [row for row in stats if row["observation"] == observation]
        if sorted(row["rank"] for row in rows) != list(range(8)):
            raise ValueError(f"{path.name}: observation {observation} ranks invalid")
        aggregate = next(
            (row for row in obs if row["observation"] == observation), None
        )
        if aggregate is None:
            raise ValueError(f"{path.name}: observation {observation} missing")
        for row in rows:
            raw = row.get("samples_us", ())
            if row["num_samples"] != samples or len(raw) != samples:
                raise ValueError(f"{path.name}: rank {row['rank']} samples invalid")
            if abs(statistics.median(raw) - row["returned_us"]) > 1e-6:
                raise ValueError(f"{path.name}: rank {row['rank']} median mismatch")
            if row["cap"] != meta["cap"] or row["max_m"] != meta["cap"] * 8:
                raise ValueError(f"{path.name}: rank {row['rank']} capacity mismatch")
            if row["route_counts"] != row["expected_route_counts"]:
                raise ValueError(f"{path.name}: rank {row['rank']} route mismatch")
            if sum(row["route_counts"]) != row["route_total"]:
                raise ValueError(f"{path.name}: rank {row['rank']} route total mismatch")
            if row["requested_flush_l2_bytes"] != meta["flush_l2_bytes"]:
                raise ValueError(f"{path.name}: rank {row['rank']} flush request mismatch")
            if row["actual_flush_l2_bytes"] < meta["flush_l2_bytes"]:
                raise ValueError(f"{path.name}: rank {row['rank']} incomplete flush")
        parsed_max = max(row["returned_us"] for row in rows)
        if abs(parsed_max - aggregate["max_rank_us"]) > 1e-6:
            raise ValueError(f"{path.name}: observation {observation} max mismatch")
        if aggregate["actual_flush_l2_bytes_min"] < meta["flush_l2_bytes"]:
            raise ValueError(f"{path.name}: aggregate flush mismatch")

    signatures = {json.dumps(row["routes"], sort_keys=True) for row in obs}
    if len(signatures) != 1:
        raise ValueError(f"{path.name}: routes changed across observations")
    return {"meta": meta, "observations": sorted(obs, key=lambda x: x["observation"])}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--logs", type=Path, required=True)
    parser.add_argument("--candidate", required=True)
    parser.add_argument("--csv", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--observations", type=int, default=3)
    parser.add_argument("--samples", type=int, default=20)
    parser.add_argument("--shapes", nargs="+", choices=SHAPES, default=SHAPES)
    parser.add_argument("--ms", nargs="+", type=int, choices=MS, default=MS)
    args = parser.parse_args()

    parsed = {}
    errors = []
    for path in sorted(args.logs.glob("*.log")):
        try:
            leaf = parse_leaf(path, args.candidate, args.observations, args.samples)
            if leaf is None:
                continue
            key = (leaf["meta"]["shape"], int(leaf["meta"]["m"]))
            if key in parsed:
                raise ValueError(f"duplicate leaf for {key}")
            parsed[key] = leaf
        except (ValueError, json.JSONDecodeError) as exc:
            errors.append(str(exc))

    rows = []
    for shape in args.shapes:
        for m in args.ms:
            leaf = parsed.get((shape, m))
            if leaf is None:
                errors.append(f"missing shape={shape} M={m}")
                continue
            values = [row["max_rank_us"] for row in leaf["observations"]]
            rows.append(
                {
                    "shape": shape,
                    "m": m,
                    "observations_us": ";".join(f"{v:.3f}" for v in values),
                    "median_max_rank_us": statistics.median(values),
                    "requested_flush_l2_bytes": leaf["meta"]["flush_l2_bytes"],
                    "actual_flush_l2_bytes_min": min(
                        row["actual_flush_l2_bytes_min"]
                        for row in leaf["observations"]
                    ),
                }
            )
    if errors:
        raise SystemExit("invalid low-latency campaign:\n" + "\n".join(errors))

    args.csv.parent.mkdir(parents=True, exist_ok=True)
    with args.csv.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    lines = [
        f"# {args.candidate}",
        "",
        "| Shape | M | max-rank observations (us) | median (us) | flush min (bytes) |",
        "|---|---:|---|---:|---:|",
    ]
    for row in rows:
        lines.append(
            f"| {row['shape']} | {row['m']} | {row['observations_us']} | "
            f"{row['median_max_rank_us']:.1f} | {row['actual_flush_l2_bytes_min']} |"
        )
    expected = len(args.shapes) * len(args.ms)
    lines += ["", f"Complete points: {len(rows)}/{expected}."]
    args.report.write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
