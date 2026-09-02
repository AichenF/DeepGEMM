#!/usr/bin/env python3
import csv
import io
import subprocess
import sys
from collections import defaultdict


def as_int(value: str) -> int:
    try:
        return int(value or 0)
    except ValueError:
        return 0


report = sys.argv[1]
result = subprocess.run(
    [
        "ncu",
        "--import",
        report,
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
rows = []
global_rows = []
stall_totals = defaultdict(lambda: defaultdict(int))
sample_totals = defaultdict(int)
for row in csv.reader(io.StringIO(result.stdout)):
    if row and row[0] == "Kernel Name":
        kernel = row[1]
    elif row and row[0] == "Address":
        header = row
    elif header is not None and len(row) == len(header):
        excessive = as_int(row[header.index("L1 Wavefronts Shared Excessive")])
        if excessive:
            rows.append(
                (
                    excessive,
                    kernel,
                    row[0],
                    row[1],
                    row[header.index("L1 Conflicts Shared N-Way")],
                    row[header.index("L1 Wavefronts Shared")],
                    row[header.index("L1 Wavefronts Shared Ideal")],
                )
            )
        global_excessive = as_int(
            row[header.index("L2 Theoretical Sectors Global Excessive")]
        )
        if global_excessive:
            global_rows.append(
                (
                    global_excessive,
                    kernel,
                    row[0],
                    row[1],
                    row[header.index("L2 Theoretical Sectors Global")],
                    row[header.index("L2 Theoretical Sectors Global Ideal")],
                )
            )
        sample_totals[kernel] += as_int(
            row[header.index("Warp Stall Sampling (Not-issued Samples)")]
        )
        for name in header:
            if name.startswith("stall_") and name.endswith(" (Not Issued)"):
                stall_totals[kernel][name.removesuffix(" (Not Issued)")] += as_int(
                    row[header.index(name)]
                )

print("SHARED_EXCESSIVE")
for row in sorted(rows, reverse=True):
    print(" | ".join(str(value) for value in row))

print("GLOBAL_EXCESSIVE")
for row in sorted(global_rows, reverse=True):
    print(" | ".join(str(value) for value in row))

print("STALL_NOT_ISSUED")
for kernel, stalls in stall_totals.items():
    total = sample_totals[kernel]
    print(kernel)
    for name, count in sorted(stalls.items(), key=lambda item: item[1], reverse=True):
        if count:
            percent = 100.0 * count / total if total else 0.0
            print(f"  {name}: {count} ({percent:.2f}%)")
