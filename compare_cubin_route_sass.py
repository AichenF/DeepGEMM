#!/usr/bin/env python3
import argparse
import hashlib
import json
import re
import subprocess


FUNCTIONS = {
    "w13_tp4_split2": "_Z10route_gemmILi4096ELi1024ELi2ELb1ELi0ELb0ELb0EEv14CUtensorMap_stS0_PKhS2_PKfS2_S4_PKiS6_S6_S4_PfPK5uint2Piii",
    "w13_tp4_split4": "_Z10route_gemmILi4096ELi1024ELi4ELb1ELi0ELb0ELb0EEv14CUtensorMap_stS0_PKhS2_PKfS2_S4_PKiS6_S6_S4_PfPK5uint2Piii",
    "w13_tp8_split2": "_Z10route_gemmILi4096ELi512ELi2ELb1ELi0ELb0ELb0EEv14CUtensorMap_stS0_PKhS2_PKfS2_S4_PKiS6_S6_S4_PfPK5uint2Piii",
    "w13_tp8_split4": "_Z10route_gemmILi4096ELi512ELi4ELb1ELi0ELb0ELb0EEv14CUtensorMap_stS0_PKhS2_PKfS2_S4_PKiS6_S6_S4_PfPK5uint2Piii",
    "w2_tp4": "_Z10route_gemmILi512ELi4096ELi1ELb0ELi0ELb0ELb0EEv14CUtensorMap_stS0_PKhS2_PKfS2_S4_PKiS6_S6_S4_PfPK5uint2Piii",
    "w2_tp8": "_Z10route_gemmILi256ELi4096ELi1ELb0ELi0ELb0ELb0EEv14CUtensorMap_stS0_PKhS2_PKfS2_S4_PKiS6_S6_S4_PfPK5uint2Piii",
}


def run(*command: str) -> str:
    return subprocess.run(
        command,
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    ).stdout


def normalized_sass(binary: str, function: str) -> str:
    text = run("cuobjdump", "-sass", "-fun", function, binary)
    return "\n".join(
        line for line in text.splitlines() if "identifier =" not in line
    )


def resource(binary: str, function: str) -> dict[str, int]:
    text = run("cuobjdump", "--dump-resource-usage", binary)
    pattern = re.compile(
        rf"Function {re.escape(function)}:\n"
        r"\s+REG:(\d+) STACK:(\d+) SHARED:(\d+) LOCAL:(\d+)"
    )
    match = pattern.search(text)
    if match is None:
        raise RuntimeError(f"resource entry missing for {function}")
    return dict(zip(("registers", "stack", "shared", "local"), map(int, match.groups())))


def summary(text: str) -> dict[str, int | str]:
    mnemonics = (
        "BSSY",
        "BSYNC",
        "BRA",
        "ISETP",
        "LDG",
        "STS",
        "SYNCS.PHASECHK",
        "WARPGROUP.DEPBAR",
    )
    return {
        "sha256": hashlib.sha256(text.encode()).hexdigest(),
        "instructions": len(
            re.findall(r"^\s*/\*[0-9a-f]{4,}\*/", text, re.MULTILINE)
        ),
        **{name.lower().replace(".", "_"): text.count(name) for name in mnemonics},
    }


parser = argparse.ArgumentParser()
parser.add_argument("control")
parser.add_argument("candidate")
args = parser.parse_args()

for label, function in FUNCTIONS.items():
    control = normalized_sass(args.control, function)
    candidate = normalized_sass(args.candidate, function)
    record = {
        "function": label,
        "identical": control == candidate,
        "control": summary(control),
        "candidate": summary(candidate),
        "control_resource": resource(args.control, function),
        "candidate_resource": resource(args.candidate, function),
    }
    print(json.dumps(record, sort_keys=True))
