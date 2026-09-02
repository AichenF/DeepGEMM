#!/usr/bin/env python3
import csv
import io
import subprocess
import sys


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

for row in sorted(rows, reverse=True):
    print(" | ".join(str(value) for value in row))
