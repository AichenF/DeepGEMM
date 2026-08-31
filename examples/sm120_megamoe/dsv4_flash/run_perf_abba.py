#!/usr/bin/env python3
"""Paired ABBA performance measurement for the SM120 MegaMoE family.

Timing comes from CUPTI concurrent-kernel activity records collected through
``CUDA_INJECTION64_PATH``, not from CUDA events, so the reported latency is the
device-side envelope of one full MegaMoE iteration. The envelope spans the first
expected kernel start to the last dependent kernel end on a rank, which keeps a
multi-kernel candidate comparable with a single fused kernel: launch gaps and
inter-kernel idle time stay inside the measurement.

Each sample is reduced across ranks with a max before any statistic is taken, so
the reported median and tail describe the critical-path rank, and arms are run
in A/B/B/A order so linear drift cancels in the paired comparison.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import statistics
import subprocess
import tempfile
import time
from pathlib import Path


def host_identity(gpus: list[int]) -> dict:
    """Bind the receipt to the machine and the exact devices it ran on."""

    def run(command: list[str]) -> str:
        return subprocess.run(command, capture_output=True, text=True).stdout.strip()

    uuids = run(["nvidia-smi", "-i", ",".join(str(g) for g in gpus),
                 "--query-gpu=index,uuid,name", "--format=csv,noheader"])
    return {
        "hostname": run(["hostname"]),
        "addresses": run(["hostname", "-I"]).split(),
        "driver": run(["nvidia-smi", "--query-gpu=driver_version",
                       "--format=csv,noheader"]).splitlines()[:1],
        "devices": [line.strip() for line in uuids.splitlines()],
    }


def gpu_telemetry(gpus: list[int]) -> list[dict]:
    """Sample the clock and power state the measurement ran at."""

    query = ("index,clocks.sm,clocks.max.sm,power.draw,temperature.gpu,"
             "pstate,clocks_throttle_reasons.active")
    completed = subprocess.run(
        ["nvidia-smi", "-i", ",".join(str(g) for g in gpus),
         f"--query-gpu={query}", "--format=csv,noheader,nounits"],
        capture_output=True, text=True)
    rows = []
    for line in completed.stdout.strip().splitlines():
        fields = [f.strip() for f in line.split(",")]
        if len(fields) != 7:
            continue
        rows.append({"index": int(fields[0]), "sm_mhz": int(fields[1]),
                     "max_sm_mhz": int(fields[2]), "power_w": float(fields[3]),
                     "temperature_c": int(fields[4]), "pstate": fields[5],
                     "throttle_reasons": fields[6]})
    return rows

IMAGE = "nvcr.io/nvidia/pytorch:26.07-py3"
CUPTI_LIB_DIR = "/usr/local/cuda-13.3/targets/x86_64-linux/lib"

GPU_HCA = {0: "mlx5_2", 1: "mlx5_3", 2: "mlx5_0", 3: "mlx5_1",
           4: "mlx5_6", 5: "mlx5_7", 6: "mlx5_4", 7: "mlx5_5"}
GPU_UVERBS = {0: 2, 1: 3, 2: 0, 3: 1, 4: 6, 5: 7, 6: 4, 7: 5}
NUMA_CPUS = {0: "0-55,112-167", 1: "56-111,168-223"}

PRODUCTION_KERNEL = "kernel_cake_sm120_production_canonical_fused_ready_chunk8"


def docker_command(build: Path, trace_dir: Path, injection: Path, gpus: list[int],
                   args: argparse.Namespace, trace_name: str,
                   overrides: dict[str, str]) -> list[str]:
    devices = ["--device=/dev/infiniband/rdma_cm"]
    for gpu in gpus:
        devices.append(f"--device=/dev/infiniband/uverbs{GPU_UVERBS[gpu]}")
    numas = sorted({0 if gpu < 4 else 1 for gpu in gpus})
    inner = (
        "export LD_LIBRARY_PATH=/opt/hpcx/nccl_spectrum-x_plugin/lib:"
        f"/usr/lib/x86_64-linux-gnu:{CUPTI_LIB_DIR}:${{LD_LIBRARY_PATH:-}}\n"
        "export CUDA_INJECTION64_PATH=/inject/" + injection.name + "\n"
        "exec timeout 1800s /lib64/ld-linux-x86-64.so.2 /artifact/perf\n"
    )
    return [
        "docker", "run", "--rm", "--pull=never",
        "--gpus", f'"device={",".join(str(g) for g in gpus)}"',
        "--network", "host", "--ipc=private", "--shm-size=1g",
        "--cpuset-cpus=" + ",".join(NUMA_CPUS[n] for n in numas),
        "--cpuset-mems=" + ",".join(str(n) for n in numas),
        "--ulimit", "memlock=-1:-1", "--cap-add=IPC_LOCK", *devices,
        "-e", "CUDA_DEVICE_ORDER=PCI_BUS_ID",
        "-e", f"CUDA_VISIBLE_DEVICES={','.join(str(i) for i in range(len(gpus)))}",
        "-e", f"NTHREADS={args.world_size}",
        "-e", "NCCL_GIN_TYPE=3",
        "-e", f"NCCL_IB_HCA={','.join(GPU_HCA[g] for g in gpus)}",
        "-e", "NCCL_NET_PLUGIN=spcx", "-e", "NCCL_DEBUG=ERROR",
        "-e", f"CAKE_ACTIVE_ROWS={args.rows}",
        "-e", f"CAKE_ORACLE={args.oracle}",
        "-e", f"CAKE_ROUTE_MODE={args.route_mode}",
        "-e", "CAKE_MASK_PERIOD=0",
        "-e", f"CAKE_WARMUP={args.warmup}",
        "-e", f"CAKE_REPEAT={args.repeat}",
        "-e", "CAKE_EPOCH_BASE=0",
        "-e", f"CAKE_CUPTI_TRACE=/trace/{trace_name}",
        "-v", f"{build}:/artifact:ro",
        "-v", f"{trace_dir}:/trace",
        "-v", f"{injection.parent}:/inject:ro",
        "-w", "/artifact",
    ] + [item for key, value in sorted(overrides.items())
         for item in ("-e", f"{key}={value}")] + [IMAGE, "bash", "-lc", inner]


def iteration_envelopes(trace: Path, kernels: list[str], world_size: int,
                        skip: int, count: int) -> dict:
    """Group kernel activity into per-rank iteration envelopes.

    ``kernels`` lists the kernel names that make up one iteration. Records are
    ordered per device and consumed ``len(kernels)`` at a time, so the envelope
    of iteration i on a device is the first start to the last end of that group.
    """
    per_device: dict[int, list[dict]] = {}
    for line in trace.read_text().splitlines():
        if not line.strip():
            continue
        record = json.loads(line)
        if not any(name in record["name"] for name in kernels):
            continue
        per_device.setdefault(record["device"], []).append(record)

    if len(per_device) != world_size:
        return {"error": f"traced {len(per_device)} devices, expected {world_size}"}

    per_rank_ms: dict[int, list[float]] = {}
    for device, records in sorted(per_device.items()):
        records.sort(key=lambda r: r["start_ns"])
        groups = [records[i:i + len(kernels)]
                  for i in range(0, len(records), len(kernels))]
        selected = groups[skip:skip + count]
        if len(selected) != count:
            return {"error": f"device {device} produced {len(groups)} iterations, "
                             f"need {skip + count}"}
        per_rank_ms[device] = [
            (max(r["end_ns"] for r in group) - min(r["start_ns"] for r in group)) / 1e6
            for group in selected
        ]

    devices = sorted(per_rank_ms)
    max_rank_ms = [max(per_rank_ms[d][i] for d in devices) for i in range(count)]
    critical_rank = [max(devices, key=lambda d: per_rank_ms[d][i]) for i in range(count)]
    return {
        "per_rank_ms": {str(d): per_rank_ms[d] for d in devices},
        "max_rank_ms": max_rank_ms,
        "critical_rank_per_sample": critical_rank,
    }


def summarize(samples: list[float]) -> dict:
    ordered = sorted(samples)
    def percentile(fraction: float) -> float:
        if len(ordered) == 1:
            return ordered[0]
        position = fraction * (len(ordered) - 1)
        low = int(position)
        high = min(low + 1, len(ordered) - 1)
        return ordered[low] + (ordered[high] - ordered[low]) * (position - low)
    return {
        "samples": len(ordered),
        "min_ms": ordered[0],
        "median_ms": statistics.median(ordered),
        "mean_ms": statistics.fmean(ordered),
        "p95_ms": percentile(0.95),
        "p99_ms": percentile(0.99),
        "max_ms": ordered[-1],
        "stdev_ms": statistics.pstdev(ordered),
    }


def useful_tflops(args: argparse.Namespace, ms: float) -> float:
    """Conventional FMA=2 useful GEMM throughput over all ranks."""
    flop = (6 * args.rows * args.world_size * args.topk
            * args.hidden * args.intermediate)
    return flop / (ms * 1e-3) / 1e12


def run_arm(name: str, build: Path, injection: Path, gpus: list[int],
            args: argparse.Namespace, index: int,
            overrides: dict[str, str]) -> dict:
    with tempfile.TemporaryDirectory(prefix="cake-cupti-") as tmp:
        trace_dir = Path(tmp)
        trace_name = f"{name}-{index}.jsonl"
        before_telemetry = gpu_telemetry(gpus)
        started = time.time()
        completed = subprocess.run(
            docker_command(build, trace_dir, injection, gpus, args, trace_name,
                           overrides),
            capture_output=True, text=True)
        elapsed = time.time() - started
        trace = trace_dir / trace_name

        record: dict = {
            "arm": name,
            "position": index,
            "clocks_before": before_telemetry,
            "clocks_after": gpu_telemetry(gpus),
            "build": str(build),
            "environment_overrides": dict(overrides),
            "exit_status": completed.returncode,
            "wall_seconds": round(elapsed, 3),
        }

        finals = [json.loads(l.split("=", 1)[1]) for l in completed.stdout.splitlines()
                  if l.startswith("READY_CHUNK8_PERF_RESULT_JSON=")]
        ranks = [json.loads(l.split("=", 1)[1]) for l in completed.stdout.splitlines()
                 if l.startswith("RANK_RESULT_JSON=")]
        record["final_record"] = finals[0] if finals else None
        record["correctness_ok"] = bool(
            finals and finals[0].get("status") == "pass"
            and finals[0].get("failures") == 0
            and ranks and all(r.get("exact_bf16_equal") for r in ranks))

        if completed.returncode != 0 or not trace.exists():
            record["error"] = completed.stderr.strip().splitlines()[-5:]
            return record

        envelopes = iteration_envelopes(trace, [PRODUCTION_KERNEL],
                                        args.world_size,
                                        args.audit_epochs + args.warmup,
                                        args.repeat)
        record.update(envelopes)
        if "max_rank_ms" in envelopes:
            stats = summarize(envelopes["max_rank_ms"])
            record["statistics"] = stats
            record["useful_tflops_at_median"] = useful_tflops(args, stats["median_ms"])
        return record


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--arm", action="append", required=True,
                        metavar="NAME=BUILD_DIR",
                        help="measurement arm; pass exactly two for a paired run")
    parser.add_argument("--arm-env", action="append", default=[],
                        metavar="NAME:KEY=VALUE",
                        help="environment override applied only to one arm")
    parser.add_argument("--injection", required=True, type=Path,
                        help="path to libcake_cupti_trace.so")
    parser.add_argument("--gpus", default="0,1,2,3")
    parser.add_argument("--world-size", type=int, required=True)
    parser.add_argument("--rows", type=int, default=2048)
    parser.add_argument("--topk", type=int, default=6)
    parser.add_argument("--hidden", type=int, default=4096)
    parser.add_argument("--intermediate", type=int, default=4096)
    parser.add_argument("--oracle", default="distinct_k32")
    parser.add_argument("--route-mode", default="balanced")
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--repeat", type=int, default=20, choices=(20, 100))
    parser.add_argument("--audit-epochs", type=int, default=3,
                        help="production launches the host runs before warmup")
    parser.add_argument("--priming-arms", type=int, default=1,
                        help="discarded arms run before the paired sequence to "
                             "settle the GPU clocks")
    parser.add_argument("--fault-retries", type=int, default=1,
                        help="retries for an arm that produced no sample")
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()

    arms = {}
    for entry in args.arm:
        name, _, path = entry.partition("=")
        if not path:
            parser.error(f"malformed arm {entry!r}, expected NAME=BUILD_DIR")
        arms[name] = Path(path)
    if len(arms) != 2:
        parser.error("a paired ABBA run needs exactly two distinct arms")

    overrides: dict[str, dict[str, str]] = {name: {} for name in arms}
    for entry in args.arm_env:
        name, _, assignment = entry.partition(":")
        key, _, value = assignment.partition("=")
        if name not in overrides or not key:
            parser.error(f"malformed arm environment override {entry!r}")
        overrides[name][key] = value

    gpus = [int(g) for g in args.gpus.split(",")]
    first, second = list(arms)
    order = [first, second, second, first]

    # These GPUs idle at 180 MHz against a 3090 MHz ceiling and the harness
    # clamps its own warmup to five launches, so the first arm after an idle
    # period measures a clock ramp rather than the kernel. Run and discard one
    # priming arm before the paired sequence.
    priming = []
    for _ in range(args.priming_arms):
        priming.append({
            "arm": first,
            "discarded": True,
            **{k: v for k, v in run_arm(first, arms[first], args.injection, gpus,
                                        args, -1, overrides[first]).items()
               if k in ("exit_status", "clocks_before", "clocks_after")},
        })
    if priming:
        print(f"[priming] {len(priming)} discarded arm(s)", flush=True)

    runs = []
    for index, name in enumerate(order):
        attempts = 0
        while True:
            attempts += 1
            run = run_arm(name, arms[name], args.injection, gpus, args, index,
                          overrides[name])
            # A run that produced no sample carries no measurement, so a
            # communicator or launch fault there is infrastructure, not a
            # result. Retry it once and record the attempt. A run that produced
            # samples is kept as measured, whatever it says.
            if "max_rank_ms" in run or attempts > args.fault_retries:
                break
        run["attempts"] = attempts
        runs.append(run)
        stats = run.get("statistics")
        if stats:
            print(f"[{index}] {name:<12} median {stats['median_ms']:.6f} ms  "
                  f"p95 {stats['p95_ms']:.6f} ms  "
                  f"{run['useful_tflops_at_median']:.2f} TFLOP/s  "
                  f"correct={run['correctness_ok']}", flush=True)
        else:
            print(f"[{index}] {name:<12} FAILED: {run.get('error', run.get('error'))}",
                  flush=True)

    pooled = {}
    for name in arms:
        samples: list[float] = []
        for run in runs:
            if run["arm"] == name and "max_rank_ms" in run:
                samples += run["max_rank_ms"]
        if samples:
            pooled[name] = summarize(samples)
            pooled[name]["useful_tflops_at_median"] = useful_tflops(
                args, pooled[name]["median_ms"])

    comparison = None
    if len(pooled) == 2:
        base, cand = first, second
        comparison = {
            "baseline": base,
            "candidate": cand,
            "median_speedup": pooled[base]["median_ms"] / pooled[cand]["median_ms"],
            "p95_speedup": pooled[base]["p95_ms"] / pooled[cand]["p95_ms"],
            "baseline_endpoint_drift_percent": (
                100.0 * (runs[3]["statistics"]["median_ms"] - runs[0]["statistics"]["median_ms"])
                / runs[0]["statistics"]["median_ms"]
                if runs[0].get("statistics") and runs[3].get("statistics") else None),
        }

    receipt = {
        "kind": "sm120_megamoe_perf_abba",
        "host": host_identity(gpus),
        "timing_source": "cupti_concurrent_kernel_activity",
        "sample_reduction": "max_over_ranks_then_statistic",
        "order": order,
        "workload": {
            "world_size": args.world_size, "gpus": gpus, "rows_per_rank": args.rows,
            "topk": args.topk, "hidden": args.hidden,
            "intermediate": args.intermediate, "oracle": args.oracle,
            "route_mode": args.route_mode, "warmup": args.warmup,
            "repeat": args.repeat,
        },
        "arms": {name: {
            "build": str(path),
            "environment_overrides": overrides[name],
            "perf_sha256": hashlib.sha256((path / "perf").read_bytes()).hexdigest(),
            "config_flags": (path / "config-flags.txt").read_text().split(),
        } for name, path in arms.items()},
        "priming_arms": priming,
        "runs": runs,
        "retried_arms": sum(1 for r in runs if r.get("attempts", 1) > 1),
        "pooled": pooled,
        "comparison": comparison,
        "all_correct": all(r["correctness_ok"] for r in runs),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(receipt, indent=1, sort_keys=True))
    print(f"\nreceipt -> {args.output}")
    return 0 if receipt["all_correct"] and len(pooled) == 2 else 1


if __name__ == "__main__":
    raise SystemExit(main())
