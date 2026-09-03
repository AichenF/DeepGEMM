#!/usr/bin/env python3
import argparse
import csv
import io
import subprocess
from collections import defaultdict


def as_int(value: str) -> int:
    try:
        return int(value or 0)
    except ValueError:
        return 0


parser = argparse.ArgumentParser()
parser.add_argument("report")
parser.add_argument("--top", type=int, default=12)
args = parser.parse_args()

result = subprocess.run(
    [
        "ncu",
        "--import",
        args.report,
        "--page",
        "source",
        "--print-source",
        "sass",
        "--csv",
    ],
    stdout=subprocess.PIPE,
    stderr=subprocess.DEVNULL,
    text=True,
    check=True,
)

kernel = ""
header = None
rows = defaultdict(list)
for row in csv.reader(io.StringIO(result.stdout)):
    if row and row[0] == "Kernel Name":
        kernel = row[1]
        header = None
    elif row and row[0] == "Address":
        header = row
    elif header is not None and len(row) == len(header) and row[0].startswith("0x"):
        rows[kernel].append(dict(zip(header, row)))

stall_names = [
    "stall_barrier",
    "stall_long_sb",
    "stall_math",
    "stall_wait",
    "stall_not_selected",
    "stall_short_sb",
]

for kernel, kernel_rows in rows.items():
    print(f"KERNEL {kernel}")
    total_not_issued = sum(
        as_int(row["Warp Stall Sampling (Not-issued Samples)"])
        for row in kernel_rows
    )
    totals = {
        name: sum(as_int(row[f"{name} (Not Issued)"]) for row in kernel_rows)
        for name in stall_names
    }
    print(f"NOT_ISSUED_TOTAL {total_not_issued}")
    print(
        "STALL_TOTALS "
        + " ".join(
            f"{name}={count}({100.0 * count / total_not_issued:.2f}%)"
            for name, count in sorted(
                totals.items(), key=lambda item: item[1], reverse=True
            )
            if count
        )
    )

    def print_hotspots(label: str, key: str) -> None:
        selected = sorted(
            kernel_rows, key=lambda row: as_int(row[key]), reverse=True
        )[: args.top]
        print(f"HOTSPOTS {label}")
        for row in selected:
            value = as_int(row[key])
            if not value:
                break
            breakdown = ",".join(
                f"{name.removeprefix('stall_')}="
                f"{as_int(row[f'{name} (Not Issued)'])}"
                for name in stall_names
                if as_int(row[f"{name} (Not Issued)"])
            )
            print(
                f"  {value:5d} {row['Address']} {row['Source'].strip()}"
                f" [{breakdown}]"
            )

    print_hotspots("all_not_issued", "Warp Stall Sampling (Not-issued Samples)")
    for name in stall_names:
        print_hotspots(name, f"{name} (Not Issued)")
