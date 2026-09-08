"""Benchmark the SM120 routed-MoE kernel on DeepSeek V4 Flash shapes."""

# ruff: noqa: E402

from __future__ import annotations

import argparse
import hashlib
import json
import os
import statistics
import sys
import threading
import time
from pathlib import Path
from typing import Any

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import deep_gemm
from deep_gemm.testing import bench_kineto
from routed_moe_sm120_utils import (
    CONTRACT,
    EXPERTS,
    HIDDEN,
    INTERMEDIATE,
    RECIPE_ID,
    TOP_K,
    WORLD_SIZE,
    make_dsv4_w4a8_inputs,
    make_dsv4_w4a8_weights,
)

KERNEL_NAME = "sm120_fp8_fp4_routed_moe_impl"
ROWS = (1024, 2048, 4096, 8192)


def _gpu_uuid(local_rank: int) -> str:
    value = torch.cuda.get_device_properties(local_rank).uuid
    value = value.decode() if isinstance(value, bytes) else str(value)
    return value if value.startswith(("GPU-", "MIG-")) else f"GPU-{value}"


class ClockSampler:
    def __init__(self, device_uuid: str):
        import pynvml

        self._nvml = pynvml
        self._nvml.nvmlInit()
        self._handle = self._nvml.nvmlDeviceGetHandleByUUID(device_uuid)
        self.reference_mhz = self._nvml.nvmlDeviceGetDefaultApplicationsClock(
            self._handle, self._nvml.NVML_CLOCK_SM
        )
        self.samples: list[int] = []
        self.error: Exception | None = None
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)

    def _run(self) -> None:
        while not self._stop.is_set():
            try:
                self.samples.append(
                    self._nvml.nvmlDeviceGetClockInfo(
                        self._handle, self._nvml.NVML_CLOCK_SM
                    )
                )
            except self._nvml.NVMLError as error:
                self.error = error
                return
            self._stop.wait(0.01)

    def start(self) -> None:
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        self._thread.join(timeout=10)
        if self._thread.is_alive():
            raise RuntimeError("GPU clock sampler did not stop")
        self._nvml.nvmlShutdown()
        if self.error is not None:
            raise RuntimeError("GPU clock sampling failed") from self.error
        if not self.samples:
            raise RuntimeError("GPU clock sampling produced no samples")

    def receipt(self) -> dict[str, float | int]:
        ordered = sorted(self.samples)
        p10 = ordered[round(0.1 * (len(ordered) - 1))]
        return {
            "samples": len(ordered),
            "p10_mhz": p10,
            "median_mhz": statistics.median(ordered),
            "reference_mhz": self.reference_mhz,
            "p10_ratio": p10 / self.reference_mhz,
        }


class ProcessWatchdog:
    def __init__(self, rank: int, timeout_seconds: float):
        self.rank = rank
        self.timeout_seconds = timeout_seconds
        self._condition = threading.Condition()
        self._deadline: float | None = None
        self._phase = ""
        self._closed = False
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def arm(self, phase: str) -> None:
        with self._condition:
            self._phase = phase
            self._deadline = time.monotonic() + self.timeout_seconds
            self._condition.notify_all()

    def close(self) -> None:
        with self._condition:
            self._closed = True
            self._condition.notify_all()
        self._thread.join(timeout=10)
        if self._thread.is_alive():
            raise RuntimeError("benchmark watchdog did not stop")

    def _run(self) -> None:
        while True:
            with self._condition:
                if self._closed:
                    return
                if self._deadline is None:
                    self._condition.wait()
                    continue
                remaining = self._deadline - time.monotonic()
                if remaining > 0:
                    self._condition.wait(remaining)
                    continue
                message = json.dumps({
                    "rank": self.rank,
                    "phase": self._phase,
                    "timeout_seconds": self.timeout_seconds,
                })
            os.write(2, f"WATCHDOG_JSON={message}\n".encode())
            os._exit(124)


def _distributed_backend() -> tuple[Any, Any, int, int]:
    if int(os.environ.get("WORLD_SIZE", "1")) != WORLD_SIZE:
        raise RuntimeError(f"benchmark requires torchrun with {WORLD_SIZE} ranks")
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    if not torch.distributed.is_initialized():
        torch.distributed.init_process_group(backend="nccl", init_method="env://")
    group = torch.distributed.group.WORLD
    control_group = torch.distributed.new_group(backend="gloo")
    return group, control_group, torch.distributed.get_rank(group), local_rank


class BenchmarkState:
    def __init__(self, group: Any, rank: int, rows: int):
        self.device = torch.device("cuda", int(os.environ["LOCAL_RANK"]))
        self.inputs = make_dsv4_w4a8_inputs(rank, rows, self.device)
        self.weights = make_dsv4_w4a8_weights(rank, self.device)
        self.session = deep_gemm.SM120RoutedMoESession(group, self.device)
        try:
            self.workspace = deep_gemm.SM120RoutedMoEWorkspace(self.device)
        except Exception:
            self.session.close()
            raise

    def launch(self) -> None:
        deep_gemm.fp8_fp4_routed_moe_sm120(
            self.session,
            self.workspace,
            self.inputs.x,
            self.inputs.x_scales,
            self.inputs.topk_indices,
            self.inputs.topk_weights,
            self.weights.w1_up_gate,
            self.weights.w2_down,
        )

    def close(self) -> None:
        try:
            self.session.close()
        finally:
            if self.session.closed:
                self.workspace.close()


def _benchmark(
    state: BenchmarkState,
    control_group: Any,
    repeats: int,
    num_tests: int,
    watchdog: ProcessWatchdog,
) -> list[float]:
    observations = []
    for repeat in range(repeats):
        watchdog.arm(f"benchmark repeat {repeat}")
        local_seconds = bench_kineto(
            state.launch,
            KERNEL_NAME,
            num_tests=num_tests,
            suppress_kineto_output=True,
            flush_l2=True,
            barrier=lambda: torch.distributed.barrier(group=control_group),
        )
        max_rank = torch.tensor(local_seconds, dtype=torch.float64)
        torch.distributed.all_reduce(
            max_rank, op=torch.distributed.ReduceOp.MAX, group=control_group
        )
        observations.append(float(max_rank.item()))
    return observations


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--m", type=int, choices=ROWS, default=2048)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--num-tests", type=int, default=20)
    parser.add_argument("--min-clock-ratio", type=float, default=0.95)
    parser.add_argument("--rank-timeout-seconds", type=float, default=300.0)
    arguments = parser.parse_args()
    if arguments.repeats <= 0 or arguments.num_tests <= 0:
        parser.error("repeat counts must be positive")
    if not 0 < arguments.min_clock_ratio <= 1:
        parser.error("--min-clock-ratio must be in (0, 1]")
    if arguments.rank_timeout_seconds <= 0:
        parser.error("--rank-timeout-seconds must be positive")
    return arguments


def main() -> int:
    args = _parse_args()
    group, control_group, rank, local_rank = _distributed_backend()
    watchdog = ProcessWatchdog(rank, args.rank_timeout_seconds)
    state = None
    clock = None
    try:
        watchdog.arm("initialization")
        state = BenchmarkState(group, rank, args.m)
        torch.distributed.barrier(group=control_group)
        state.launch()
        torch.cuda.synchronize()
        if int(state.workspace.diagnostics()["protocol_error"].item()) != 0:
            raise RuntimeError(f"rank {rank} reported a protocol error")

        clock = ClockSampler(_gpu_uuid(local_rank))
        clock.start()
        observations = _benchmark(
            state, control_group, args.repeats, args.num_tests, watchdog
        )
        clock.stop()
        local_clock = clock.receipt()
        clocks: list[dict[str, float | int] | None] = [None] * WORLD_SIZE
        torch.distributed.all_gather_object(clocks, local_clock, group=control_group)
        if any(item is None for item in clocks):
            raise RuntimeError("failed to gather clock telemetry")
        rank_clocks = [item for item in clocks if item is not None]
        accepted = min(float(item["p10_ratio"]) for item in rank_clocks) >= args.min_clock_ratio

        if rank == 0:
            median_seconds = statistics.median(observations)
            useful_flops = args.m * TOP_K * 6 * HIDDEN * INTERMEDIATE
            result = {
                "status": "accepted" if accepted else "rejected",
                "model": {
                    "hidden": HIDDEN,
                    "intermediate": INTERMEDIATE,
                    "experts": EXPERTS,
                    "top_k": TOP_K,
                    "w1": CONTRACT.w1,
                    "w2": CONTRACT.w2,
                    "precision": "MXFP8 E4M3 x MXFP4 E2M1 K32, BF16 output",
                },
                "ep": WORLD_SIZE,
                "m_tokens_per_rank": args.m,
                "max_rank_median_us": median_seconds * 1e6,
                "max_rank_samples_us": [value * 1e6 for value in observations],
                "effective_tflops": useful_flops / median_seconds / 1e12,
                "clock_by_rank": rank_clocks,
                "fixture_recipe": RECIPE_ID,
                "kernel_sha256": hashlib.sha256(
                    (Path(deep_gemm.__file__).resolve().parent /
                     "include/deep_gemm/impls/sm120_fp8_fp4_routed_moe.cuh").read_bytes()
                ).hexdigest(),
                "timing": "cold-L2 Kineto/CUPTI kernel activity; max of rank-local means",
            }
            print("RESULT_JSON=" + json.dumps(result, sort_keys=True), flush=True)
        return 0 if accepted else 2
    finally:
        watchdog.arm("cleanup")
        if clock is not None and clock._thread.is_alive():
            clock.stop()
        if state is not None:
            state.close()
        torch.distributed.destroy_process_group(control_group)
        torch.distributed.destroy_process_group(group)
        watchdog.close()


if __name__ == "__main__":
    raise SystemExit(main())
