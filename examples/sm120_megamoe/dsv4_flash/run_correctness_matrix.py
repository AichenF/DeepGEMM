#!/usr/bin/env python3
"""Run the SM120 MegaMoE fail-closed correctness matrix and write a receipt.

Covers the balanced, skewed and empty routing fixtures across ragged, tail,
capacity and masked row counts, every oracle, and the three-epoch slot-reuse
pattern the transport protocol depends on. A case passes only when the process
exits zero, every rank reports ``status=pass`` with bit-exact BF16 output, and
every protocol, owner, counter, signal, acknowledgement, ready-scheduler, stage
and guard counter is zero.

Run this on the SM120 host after ``build.sh``.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import time
from pathlib import Path

IMAGE = "nvcr.io/nvidia/pytorch:26.07-py3"

# GPU index -> the InfiniBand HCA on the same PCIe switch, from `nvidia-smi topo -m`.
GPU_HCA = {0: "mlx5_2", 1: "mlx5_3", 2: "mlx5_0", 3: "mlx5_1",
           4: "mlx5_6", 5: "mlx5_7", 6: "mlx5_4", 7: "mlx5_5"}
GPU_UVERBS = {0: 2, 1: 3, 2: 0, 3: 1, 4: 6, 5: 7, 6: 4, 7: 5}
NUMA_CPUS = {0: "0-55,112-167", 1: "56-111,168-223"}

ZERO_COUNTERS = (
    "protocol_error", "owner_mismatches", "counter_mismatches",
    "signal_mismatches", "ack_signal_mismatches", "ready_audit_mismatches",
    "w1_bf16_mismatches", "requant_fp8_sf_mismatches",
    "w2_bf16_partial_mismatches", "output_mismatches",
    "output_guard_mismatches", "launch_mismatches", "max_abs_error",
)


def case_matrix(world_size: int) -> list[dict]:
    """Routing, capacity and masking fixtures the transport must survive."""
    cases: list[dict] = []
    for rows in (1, 17, 113, 1024, 2048):
        cases.append({"rows": rows, "route": "balanced", "oracle": "distinct_k32", "mask": 0})
    for route in ("skewed", "empty"):
        for rows in (17, 2048):
            cases.append({"rows": rows, "route": route, "oracle": "distinct_k32", "mask": 0})
    for mask in (3, 7):
        cases.append({"rows": 2048, "route": "balanced", "oracle": "distinct_k32", "mask": mask})
    for oracle in ("zero", "analytic"):
        cases.append({"rows": 113, "route": "balanced", "oracle": oracle, "mask": 0})
    for case in cases:
        case["world_size"] = world_size
    return cases


def docker_command(build: Path, gpus: list[int], case: dict) -> list[str]:
    devices: list[str] = ["--device=/dev/infiniband/rdma_cm"]
    for gpu in gpus:
        devices.append(f"--device=/dev/infiniband/uverbs{GPU_UVERBS[gpu]}")
    numas = sorted({0 if gpu < 4 else 1 for gpu in gpus})
    cpuset = ",".join(NUMA_CPUS[numa] for numa in numas)
    memset = ",".join(str(numa) for numa in numas)
    inner = (
        "export LD_LIBRARY_PATH=/opt/hpcx/nccl_spectrum-x_plugin/lib:"
        "/usr/lib/x86_64-linux-gnu:${LD_LIBRARY_PATH:-}\n"
        "exec timeout 900s /lib64/ld-linux-x86-64.so.2 /artifact/correctness\n"
    )
    return [
        "docker", "run", "--rm", "--pull=never",
        "--gpus", f'"device={",".join(str(g) for g in gpus)}"',
        "--network", "host", "--ipc=private", "--shm-size=1g",
        f"--cpuset-cpus={cpuset}", f"--cpuset-mems={memset}",
        "--ulimit", "memlock=-1:-1", "--cap-add=IPC_LOCK", *devices,
        "-e", "CUDA_DEVICE_ORDER=PCI_BUS_ID",
        "-e", f"CUDA_VISIBLE_DEVICES={','.join(str(i) for i in range(len(gpus)))}",
        "-e", f"NTHREADS={case['world_size']}",
        "-e", "NCCL_GIN_TYPE=3",
        "-e", f"NCCL_IB_HCA={','.join(GPU_HCA[g] for g in gpus)}",
        "-e", "NCCL_NET_PLUGIN=spcx", "-e", "NCCL_DEBUG=ERROR",
        "-e", f"CAKE_ACTIVE_ROWS={case['rows']}",
        "-e", f"CAKE_ORACLE={case['oracle']}",
        "-e", f"CAKE_ROUTE_MODE={case['route']}",
        "-e", f"CAKE_MASK_PERIOD={case['mask']}",
        "-v", f"{build}:/artifact:ro", "-w", "/artifact",
        IMAGE, "bash", "-lc", inner,
    ]


def evaluate(stdout: str, case: dict) -> tuple[bool, list[str], list[dict]]:
    ranks = [json.loads(line.split("=", 1)[1])
             for line in stdout.splitlines() if line.startswith("RANK_RESULT_JSON=")]
    finals = [json.loads(line.split("=", 1)[1])
              for line in stdout.splitlines() if line.startswith("RESULT_JSON=")]

    reasons: list[str] = []
    if len(ranks) != case["world_size"]:
        reasons.append(f"expected {case['world_size']} rank records, saw {len(ranks)}")
    if len(finals) != 1:
        reasons.append(f"expected one final record, saw {len(finals)}")
    elif finals[0].get("status") != "pass" or finals[0].get("failures") != 0:
        reasons.append(f"final record {finals[0].get('status')}/"
                       f"{finals[0].get('failures')} failures")

    for record in ranks:
        rank = record.get("rank")
        if record.get("status") != "pass":
            reasons.append(f"rank {rank} status {record.get('status')}")
        if not record.get("exact_bf16_equal"):
            reasons.append(f"rank {rank} not bit-exact")
        for counter in ZERO_COUNTERS:
            if record.get(counter):
                reasons.append(f"rank {rank} {counter}={record[counter]}")
        if record.get("epoch_slots") != [0, 1, 0]:
            reasons.append(f"rank {rank} slot pattern {record.get('epoch_slots')}")
        totals = record.get("epoch_route_totals") or []
        if len(set(totals)) != 1:
            reasons.append(f"rank {rank} route totals drift {totals}")
        for epoch in record.get("stage_mismatches_per_epoch") or []:
            if any(epoch):
                reasons.append(f"rank {rank} stage mismatches {epoch}")
        if record.get("launch_count_per_epoch") != 1:
            reasons.append(f"rank {rank} launched {record.get('launch_count_per_epoch')} "
                           "kernels per epoch")
    return not reasons, reasons, ranks


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--build", required=True, type=Path)
    parser.add_argument("--gpus", default="0,1,2,3")
    parser.add_argument("--world-size", type=int, required=True)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()

    gpus = [int(g) for g in args.gpus.split(",")]
    if args.world_size > len(gpus):
        parser.error("world size exceeds the exposed GPU count")

    binary = args.build / "correctness"
    digest = hashlib.sha256(binary.read_bytes()).hexdigest()

    records = []
    failed = 0
    for case in case_matrix(args.world_size):
        started = time.time()
        completed = subprocess.run(docker_command(args.build, gpus, case),
                                   capture_output=True, text=True)
        ok, reasons, ranks = evaluate(completed.stdout, case)
        ok = ok and completed.returncode == 0
        if completed.returncode != 0:
            reasons.append(f"exit status {completed.returncode}")
        if not ok:
            failed += 1
        records.append({
            **case,
            "passed": ok,
            "reasons": reasons,
            "seconds": round(time.time() - started, 3),
            "stderr_tail": completed.stderr.strip().splitlines()[-3:],
            "rank_records": ranks,
        })
        status = "pass" if ok else "FAIL"
        print(f"[{status}] rows={case['rows']:>4} route={case['route']:<8} "
              f"oracle={case['oracle']:<12} mask={case['mask']} "
              f"({records[-1]['seconds']}s)"
              + ("" if ok else f" -> {reasons[:3]}"), flush=True)

    receipt = {
        "kind": "sm120_megamoe_correctness_matrix",
        "build": str(args.build),
        "correctness_sha256": digest,
        "config_flags": (args.build / "config-flags.txt").read_text().split(),
        "gpus": gpus,
        "world_size": args.world_size,
        "cases": len(records),
        "failures": failed,
        "status": "pass" if failed == 0 else "fail",
        "records": records,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(receipt, indent=1, sort_keys=True))
    print(f"\n{len(records) - failed}/{len(records)} cases passed -> {args.output}")
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
