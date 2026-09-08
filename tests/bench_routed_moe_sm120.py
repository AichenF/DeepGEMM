"""Fail-closed eight-rank benchmark for the SM120 routed-MoE fast path.

Example::

    torchrun --nproc-per-node=8 tests/bench_routed_moe_sm120.py --m 2048

Latency is the maximum rank time for each iteration. Clock sampling runs while
the measured kernels execute; a run with an underclocked rank emits a rejected
receipt and returns a non-zero status.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import statistics
import threading
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any

import torch

import deep_gemm
from deep_gemm.testing.routed_moe_sm120 import (
    ACTIVATION_CLAMP,
    CONTRACT,
    EXPERTS,
    FAST_MATH,
    HIDDEN,
    INTERMEDIATE,
    K_GROUP,
    RECIPE_ID,
    TOP_K,
    WORLD_SIZE,
    epoch_slots,
    make_dsv4_w4a8_inputs,
    make_dsv4_w4a8_weights,
)

SCHEMA = "deepgemm.sm120-routed-moe.benchmark/v3"
DEFAULT_NETWORK_PORT_COUNT = 8
DEFAULT_NETWORK_PORT_GBPS = 400.0
MIXED_QMMA_GFLOPS_PER_SM_AT_1GHZ = 2048.0


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _source_receipt() -> dict[str, str]:
    package = Path(deep_gemm.__file__).resolve().parent
    files = {
        "kernel": package
        / "include/deep_gemm/impls/sm120_fp8_fp4_routed_moe.cuh",
        "fixture": package / "testing/routed_moe_sm120.py",
        "benchmark": Path(__file__).resolve(),
    }
    return {name: _sha256(path) for name, path in files.items()}


def _network_inventory(
    expected_port_count: int,
    expected_port_gbps: float,
    sysfs_root: Path = Path("/sys/class/infiniband"),
) -> dict[str, Any]:
    raw_hcas = os.environ.get("NCCL_IB_HCA", "")
    hcas = [item.split(":", 1)[0] for item in raw_hcas.split(",") if item]
    ports = []
    error = None
    if not hcas or raw_hcas.startswith("^"):
        error = "NCCL_IB_HCA must explicitly list the benchmark HCAs"
    else:
        try:
            for hca in hcas:
                port_root = sysfs_root / hca / "ports"
                for port in sorted(port_root.iterdir(), key=lambda path: int(path.name)):
                    rate_text = (port / "rate").read_text().strip()
                    rate_gbps = float(rate_text.split()[0])
                    state = (port / "state").read_text().strip()
                    physical_state = (port / "phys_state").read_text().strip()
                    ports.append({
                        "hca": hca,
                        "port": int(port.name),
                        "rate_gbps": rate_gbps,
                        "state": state,
                        "physical_state": physical_state,
                        "active": state.startswith("4:") and physical_state.startswith("5:"),
                    })
        except (OSError, ValueError) as exc:
            error = f"{type(exc).__name__}: {exc}"
    accepted = (
        error is None
        and len(ports) == expected_port_count
        and all(
            port["active"] and port["rate_gbps"] == expected_port_gbps
            for port in ports
        )
    )
    return {
        "nccl_ib_hca": raw_hcas,
        "ports": ports,
        "expected_port_count": expected_port_count,
        "expected_port_gbps": expected_port_gbps,
        "accepted": accepted,
        "error": error,
    }


def _percentile(values: list[float], fraction: float) -> float:
    if not values:
        raise ValueError("cannot compute a percentile of an empty sample")
    ordered = sorted(values)
    index = min(len(ordered) - 1, max(0, round((len(ordered) - 1) * fraction)))
    return ordered[index]


def _gpu_uuid(local_rank: int) -> str:
    properties = torch.cuda.get_device_properties(local_rank)
    uuid = getattr(properties, "uuid", None)
    if uuid:
        value = uuid.decode() if isinstance(uuid, bytes) else str(uuid)
        return value if value.startswith(("GPU-", "MIG-")) else f"GPU-{value}"
    visible = os.environ.get("CUDA_VISIBLE_DEVICES", "")
    entries = [item.strip() for item in visible.split(",") if item.strip()]
    if len(entries) > local_rank:
        return entries[local_rank]
    return str(local_rank)


class ClockSampler:
    def __init__(self, device: str, interval_seconds: float = 0.01):
        import pynvml

        self.device = device
        self.interval_seconds = interval_seconds
        self.samples: list[float] = []
        self.max_clock = 0.0
        self.error: str | None = None
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._nvml = pynvml
        self._nvml.nvmlInit()
        self._handle = self._nvml.nvmlDeviceGetHandleByUUID(device)
        self.reference_clock = self._nvml.nvmlDeviceGetDefaultApplicationsClock(
            self._handle, self._nvml.NVML_CLOCK_SM
        )

    def start(self) -> None:
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def _run(self) -> None:
        while not self._stop.is_set():
            try:
                current = self._nvml.nvmlDeviceGetClockInfo(
                    self._handle, self._nvml.NVML_CLOCK_SM
                )
                maximum = self._nvml.nvmlDeviceGetMaxClockInfo(
                    self._handle, self._nvml.NVML_CLOCK_SM
                )
                self.samples.append(current)
                self.max_clock = max(self.max_clock, maximum)
            except self._nvml.NVMLError as exc:
                self.error = f"{type(exc).__name__}: {exc}"
                return
            self._stop.wait(self.interval_seconds)

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=10)
            if self._thread.is_alive():
                raise RuntimeError("GPU clock sampler did not stop")
        self._nvml.nvmlShutdown()
        if self.error is not None:
            raise RuntimeError(f"GPU clock sampling failed: {self.error}")
        if not self.samples or self.max_clock <= 0:
            raise RuntimeError("GPU clock sampling produced no usable samples")

    def receipt(self, minimum_mhz: float) -> dict[str, Any]:
        p10 = _percentile(self.samples, 0.10)
        return {
            "samples": len(self.samples),
            "min_mhz": min(self.samples),
            "p10_mhz": p10,
            "median_mhz": statistics.median(self.samples),
            "max_observed_mhz": max(self.samples),
            "max_supported_mhz": self.max_clock,
            "default_application_mhz": self.reference_clock,
            "minimum_mhz": minimum_mhz,
            "above_absolute_minimum": p10 >= minimum_mhz,
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

    def disarm(self) -> None:
        with self._condition:
            self._deadline = None
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
                    "schema": SCHEMA,
                    "status": "timeout",
                    "rank": self.rank,
                    "phase": self._phase,
                    "timeout_seconds": self.timeout_seconds,
                })
            os.write(2, f"WATCHDOG_JSON={message}\n".encode())
            os._exit(124)


def _distributed_backend() -> tuple[Any, Any, int, int]:
    if int(os.environ.get("WORLD_SIZE", "1")) != WORLD_SIZE:
        raise RuntimeError("benchmark requires torchrun --nproc-per-node=8")
    if not torch.cuda.is_available():
        raise RuntimeError("benchmark requires CUDA")
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    if not torch.distributed.is_initialized():
        torch.distributed.init_process_group(backend="nccl", init_method="env://")
    group = torch.distributed.group.WORLD
    torch.distributed.barrier(group=group, device_ids=[local_rank])
    control_group = torch.distributed.new_group(backend="gloo")
    return group, control_group, torch.distributed.get_rank(group), local_rank


class PublicRoutedMoEState:
    """Benchmark state expressed exclusively through the public SM120 API."""

    def __init__(self, group: Any, rank: int, active_rows: int):
        self.rank = rank
        self.active_rows = active_rows
        self.device = torch.device("cuda", int(os.environ["LOCAL_RANK"]))
        self.layout = dict(deep_gemm._C.get_sm120_routed_moe_layout())
        self.inputs = make_dsv4_w4a8_inputs(rank, active_rows, self.device)
        self.weights = make_dsv4_w4a8_weights(rank, self.device)
        self.session: deep_gemm.SM120RoutedMoESession | None = None
        self.workspace: deep_gemm.SM120RoutedMoEWorkspace | None = None
        self.session = deep_gemm.SM120RoutedMoESession(group, self.device)
        try:
            self.workspace = deep_gemm.SM120RoutedMoEWorkspace(self.device)
        except Exception:
            self.session.close()
            raise

        local_grid = min(
            torch.cuda.get_device_properties(self.device).multi_processor_count,
            110,
        )
        grid = torch.tensor(local_grid, dtype=torch.int32, device=self.device)
        torch.distributed.all_reduce(
            grid,
            op=torch.distributed.ReduceOp.MIN,
            group=group,
        )
        self.grid_ctas = int(grid.item())
        if self.grid_ctas < WORLD_SIZE:
            self.close()
            raise RuntimeError(f"cooperative grid has only {self.grid_ctas} CTAs")

    def diagnostics(self) -> dict[str, torch.Tensor]:
        """Read a stable post-launch snapshot through the public workspace API."""

        if self.workspace is None:
            raise RuntimeError("benchmark state is closed")
        return self.workspace.diagnostics()

    def launch(self) -> None:
        if self.session is None or self.workspace is None:
            raise RuntimeError("benchmark state is closed")
        deep_gemm.fp8_fp4_routed_moe_sm120(
            self.session,
            self.workspace,
            self.inputs.x,
            self.inputs.x_scales,
            self.inputs.topk_indices,
            self.inputs.topk_weights,
            self.weights.w1_up_gate,
            self.weights.w2_down,
            grid_ctas=self.grid_ctas,
        )

    def close(self) -> None:
        try:
            if self.session is not None:
                self.session.close()
        finally:
            if self.workspace is not None:
                self.workspace.close()
            self.workspace = None
            self.session = None


def _time_epochs(
    state: PublicRoutedMoEState,
    control_group: Any,
    iterations: int,
    watchdog: ProcessWatchdog,
) -> list[float]:
    starts = [torch.cuda.Event(enable_timing=True) for _ in range(iterations)]
    ends = [torch.cuda.Event(enable_timing=True) for _ in range(iterations)]
    for index in range(iterations):
        watchdog.arm(f"timed iteration {index}")
        starts[index].record()
        state.launch()
        ends[index].record()
        ends[index].synchronize()
        torch.distributed.barrier(group=control_group)
    local = torch.tensor(
        [start.elapsed_time(end) for start, end in zip(starts, ends, strict=True)],
        dtype=torch.float64,
    )
    ranks = [torch.empty_like(local) for _ in range(WORLD_SIZE)]
    torch.distributed.all_gather(ranks, local, group=control_group)
    return torch.stack(ranks).amax(dim=0).tolist()


def _communication_receipt(
    rank: int,
    diagnostics: dict[str, torch.Tensor],
    layout: dict[str, Any],
) -> dict[str, Any]:
    """Summarize the last completed epoch after the timed region."""

    owner_record_counts = diagnostics["owner_record_counts"].to(
        device="cpu", dtype=torch.int64
    )
    source_route_counts = diagnostics["source_route_counts"].to(
        device="cpu", dtype=torch.int64
    )
    result_ovf_cursor = diagnostics["result_ovf_cursor"].to(
        device="cpu", dtype=torch.int64
    )
    chunks_per_peer = int(layout["codec_max_chunks_per_peer"])
    raw_blocks_by_source = [
        int(
            result_ovf_cursor[
                source * chunks_per_peer:(source + 1) * chunks_per_peer
            ].sum().item()
        )
        for source in range(WORLD_SIZE)
    ]
    remote_raw_blocks = sum(
        blocks
        for source, blocks in enumerate(raw_blocks_by_source)
        if source != rank
    )
    remote_dispatch_records = int(
        owner_record_counts.sum().item() - owner_record_counts[rank].item()
    )
    remote_result_routes = int(
        source_route_counts.sum().item() - source_route_counts[rank].item()
    )
    application_egress_bytes_by_peer = []
    for peer in range(WORLD_SIZE):
        if peer == rank:
            application_egress_bytes_by_peer.append(0)
            continue
        dispatch_records = int(owner_record_counts[peer].item())
        result_routes = int(source_route_counts[peer].item())
        result_bytes = (
            result_routes * int(layout["codec_encoded_row_bytes"])
            + raw_blocks_by_source[peer] * int(layout["codec_raw_tail_bytes"])
            if result_routes
            else int(layout["max_rows"])
        )
        application_egress_bytes_by_peer.append(
            int(layout["dispatch_header_slot_bytes"])
            + dispatch_records * int(layout["dispatch_record_bytes"])
            + 4 * max(result_routes, 1)
            + result_bytes
            + 1
        )
    application_egress_bytes = sum(application_egress_bytes_by_peer)
    return {
        "total_valid_routes": int(diagnostics["total_valid_routes"].item()),
        "total_m_tasks": int(diagnostics["total_m_tasks"].item()),
        "remote_dispatch_records": remote_dispatch_records,
        "remote_result_routes": remote_result_routes,
        "remote_raw_blocks": remote_raw_blocks,
        "result_ovf_cursor_raw_blocks": int(result_ovf_cursor.sum().item()),
        "owner_record_counts_by_destination": owner_record_counts.tolist(),
        "source_route_counts_by_source": source_route_counts.tolist(),
        "result_raw_blocks_by_source": raw_blocks_by_source,
        "application_egress_bytes_by_peer": application_egress_bytes_by_peer,
        "application_egress_bytes": application_egress_bytes,
    }


def _aggregate_communication(
    per_rank: list[dict[str, Any]],
) -> dict[str, Any]:
    counter_names = (
        "total_valid_routes",
        "total_m_tasks",
        "remote_dispatch_records",
        "remote_result_routes",
        "remote_raw_blocks",
        "result_ovf_cursor_raw_blocks",
        "application_egress_bytes",
    )
    return {
        "last_completed_epoch_per_rank": per_rank,
        "global_totals": {
            name: sum(int(receipt[name]) for receipt in per_rank)
            for name in counter_names
        },
        "counter_semantics": {
            "remote_dispatch_records": (
                "sum(owner_record_counts) excluding this rank's owner"
            ),
            "remote_result_routes": (
                "sum(source_route_counts) excluding this rank's source"
            ),
            "result_ovf_cursor_raw_blocks": "sum(result_ovf_cursor)",
            "remote_raw_blocks": "sum(result_ovf_cursor) excluding this rank's source",
            "application_egress_bytes": (
                "accepted normal-path remote put payload: dispatch, result codec, route "
                "map, header, acknowledgement, and empty-result payload bytes; excludes "
                "GIN signal and transport overhead"
            ),
        },
    }


def _sol_receipt(
    per_rank: list[dict[str, Any]],
    layout: dict[str, Any],
    active_rows: int,
    observed_ms: float,
    empirical_compute_roof_tflops: float | None,
    empirical_compute_roof_source: str | None,
    sm_counts: list[int],
    maximum_sm_clocks_mhz: list[float],
    network_port_count: int,
    network_port_gbps: float,
) -> dict[str, Any]:
    flops_per_route = 6 * HIDDEN * INTERMEDIATE
    useful_flops_per_rank = active_rows * TOP_K * flops_per_route
    executed_flops_by_rank = [
        int(receipt["total_m_tasks"])
        * int(layout["task_rows"])
        * flops_per_route
        for receipt in per_rank
    ]
    if len(sm_counts) != len(per_rank) or len(maximum_sm_clocks_mhz) != len(per_rank):
        raise ValueError("compute-roof hardware facts must cover every rank")
    theoretical_compute_roofs = [
        sm_count
        * maximum_clock_mhz
        * MIXED_QMMA_GFLOPS_PER_SM_AT_1GHZ
        / 1.0e6
        for sm_count, maximum_clock_mhz in zip(
            sm_counts,
            maximum_sm_clocks_mhz,
            strict=True,
        )
    ]
    theoretical_compute_floor_ms = max(
        executed_flops / (roof_tflops * 1.0e9)
        for executed_flops, roof_tflops in zip(
            executed_flops_by_rank,
            theoretical_compute_roofs,
            strict=True,
        )
    )
    empirical_compute_floor_ms = (
        max(executed_flops_by_rank)
        / (empirical_compute_roof_tflops * 1.0e9)
        if empirical_compute_roof_tflops is not None
        else None
    )
    total_application_payload_bytes = sum(
        int(receipt["application_egress_bytes"])
        for receipt in per_rank
    )
    aggregate_network_gb_per_second = (
        network_port_count * network_port_gbps / 8.0
    )
    network_floor_ms = (
        total_application_payload_bytes / (aggregate_network_gb_per_second * 1.0e6)
    )
    theoretical_floor_ms = max(theoretical_compute_floor_ms, network_floor_ms)
    empirical_floor_ms = (
        max(empirical_compute_floor_ms, network_floor_ms)
        if empirical_compute_floor_ms is not None
        else None
    )
    return {
        "useful_flops_per_rank": useful_flops_per_rank,
        "executed_flops_by_rank": executed_flops_by_rank,
        "padding_ratio_by_rank": [
            executed / useful_flops_per_rank
            for executed in executed_flops_by_rank
        ],
        "compute": {
            "mixed_qmma_gflops_per_sm_at_1ghz": (
                MIXED_QMMA_GFLOPS_PER_SM_AT_1GHZ
            ),
            "instruction_limit": (
                "E4M3 x E2M1 mixed QMMA is limited by the FP8 operand rate"
            ),
            "sm_count_by_rank": sm_counts,
            "maximum_sm_clock_mhz_by_rank": maximum_sm_clocks_mhz,
            "physical_theoretical_roof_tflops_by_rank": (
                theoretical_compute_roofs
            ),
            "physical_theoretical_floor_ms": theoretical_compute_floor_ms,
            "empirical_roof_tflops_per_gpu": (
                empirical_compute_roof_tflops
            ),
            "empirical_roof_source": empirical_compute_roof_source,
            "empirical_floor_ms": empirical_compute_floor_ms,
        },
        "network": {
            "physical_port_count": network_port_count,
            "physical_port_line_rate_gbps": network_port_gbps,
            "aggregate_egress_line_rate_gb_per_second": (
                aggregate_network_gb_per_second
            ),
            "normal_path_application_payload_bytes": total_application_payload_bytes,
            "application_payload_floor_ms": network_floor_ms,
        },
        "physical_theoretical_floor_ms": theoretical_floor_ms,
        "observed_over_physical_theoretical_floor": (
            observed_ms / theoretical_floor_ms
        ),
        "physical_theoretical_sol_efficiency": (
            theoretical_floor_ms / observed_ms
        ),
        "empirical_attainable_floor_ms": empirical_floor_ms,
        "observed_over_empirical_attainable_floor": (
            observed_ms / empirical_floor_ms
            if empirical_floor_ms is not None
            else None
        ),
        "empirical_attainable_sol_efficiency": (
            empirical_floor_ms / observed_ms
            if empirical_floor_ms is not None
            else None
        ),
        "status": "complete" if empirical_floor_ms is not None else "theoretical_only",
        "scope": (
            "The physical theoretical floor is the max of the rank-local, "
            "padding-aware mixed-QMMA floor at each GPU's maximum supported SM clock "
            "and the aggregate physical-port application-payload floor. The optional "
            "empirical floor substitutes an independently measured, source-identified "
            "sustained mixed-QMMA roof. Both assume perfect compute/communication "
            "overlap and exclude GIN/wire overhead, launch latency, and control traffic."
        ),
    }


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--m", type=int, choices=(1024, 2048, 4096, 8192), default=2048)
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--iterations", type=int, default=100)
    parser.add_argument(
        "--empirical-compute-roof-tflops",
        type=float,
        default=None,
        help="independently measured sustained mixed E4M3 x E2M1 QMMA roof",
    )
    parser.add_argument(
        "--empirical-compute-roof-source",
        default=None,
        help="stable receipt path, hash, or other provenance for the compute roof",
    )
    parser.add_argument(
        "--network-port-count",
        type=int,
        default=DEFAULT_NETWORK_PORT_COUNT,
    )
    parser.add_argument(
        "--network-port-gbps",
        type=float,
        default=DEFAULT_NETWORK_PORT_GBPS,
    )
    parser.add_argument("--min-sm-clock-mhz", type=float, default=0.0)
    parser.add_argument(
        "--rank-timeout-seconds",
        type=float,
        default=300.0,
        help="abort the process when one rank makes no benchmark progress",
    )
    parser.add_argument(
        "--min-application-sm-clock-ratio",
        type=float,
        default=0.95,
        help="reject when a rank's clock p10 is below this fraction of its default application clock",
    )
    arguments = parser.parse_args()
    if arguments.warmup < 3:
        parser.error("--warmup must be at least 3 to exercise slot reuse [0,1,0]")
    if arguments.iterations < 1:
        parser.error("--iterations must be positive")
    if (
        arguments.empirical_compute_roof_tflops is None
    ) != (arguments.empirical_compute_roof_source is None):
        parser.error(
            "--empirical-compute-roof-tflops and "
            "--empirical-compute-roof-source must be supplied together"
        )
    if (
        arguments.empirical_compute_roof_tflops is not None
        and arguments.empirical_compute_roof_tflops <= 0
    ):
        parser.error("--empirical-compute-roof-tflops must be positive")
    if arguments.network_port_count <= 0 or arguments.network_port_gbps <= 0:
        parser.error("network port count and line rate must be positive")
    if not 0.0 < arguments.min_application_sm_clock_ratio <= 1.0:
        parser.error("--min-application-sm-clock-ratio must be in (0, 1]")
    if arguments.min_sm_clock_mhz < 0:
        parser.error("--min-sm-clock-mhz must be nonnegative")
    if arguments.rank_timeout_seconds <= 0:
        parser.error("--rank-timeout-seconds must be positive")
    return arguments


def main() -> int:
    args = _parse_args()
    group, control_group, rank, local_rank = _distributed_backend()
    state: PublicRoutedMoEState | None = None
    clock: ClockSampler | None = None
    watchdog = ProcessWatchdog(rank, args.rank_timeout_seconds)
    try:
        watchdog.arm("network preflight")
        network_inventory = _network_inventory(
            args.network_port_count,
            args.network_port_gbps,
        )
        network_accepted = torch.tensor(
            int(network_inventory["accepted"]),
            dtype=torch.int32,
        )
        torch.distributed.all_reduce(
            network_accepted,
            op=torch.distributed.ReduceOp.MIN,
            group=control_group,
        )
        torch.distributed.barrier(group=control_group)
        watchdog.arm("state initialization")
        state = PublicRoutedMoEState(group, rank, active_rows=args.m)
        torch.distributed.barrier(group=control_group)
        first_measured_epoch = args.warmup
        clock = ClockSampler(_gpu_uuid(local_rank))
        for epoch in range(args.warmup):
            watchdog.arm(f"warmup epoch {epoch}")
            state.launch()
            torch.cuda.synchronize(state.device)
            torch.distributed.barrier(group=control_group)
        warmup_diagnostics = state.diagnostics()
        warmup_protocol_error = int(warmup_diagnostics["protocol_error"].item())
        if warmup_protocol_error != 0:
            raise RuntimeError(
                f"rank {rank} protocol error after warmup: "
                f"{warmup_protocol_error}"
            )
        if epoch_slots([0, 1, 2]) != [0, 1, 0]:
            raise AssertionError("two-slot epoch lifecycle is broken")

        torch.distributed.barrier(group=control_group)
        clock.start()
        max_rank_ms = _time_epochs(
            state, control_group, args.iterations, watchdog
        )
        watchdog.arm("post-timing collection")
        clock.stop()
        post_timing_diagnostics = state.diagnostics()
        communication_receipt = _communication_receipt(
            rank,
            post_timing_diagnostics,
            state.layout,
        )
        communication_by_rank: list[dict[str, Any] | None] = [None] * WORLD_SIZE
        torch.distributed.all_gather_object(
            communication_by_rank,
            communication_receipt,
            group=control_group,
        )
        if any(receipt is None for receipt in communication_by_rank):
            raise RuntimeError("failed to gather communication receipts from all ranks")
        complete_communication_by_rank = [
            receipt for receipt in communication_by_rank if receipt is not None
        ]
        clock_receipt = clock.receipt(args.min_sm_clock_mhz)
        local_clock = torch.tensor(
            [
                clock_receipt["p10_mhz"],
                clock_receipt["default_application_mhz"],
                float(clock_receipt["above_absolute_minimum"]),
                clock_receipt["max_supported_mhz"],
                torch.cuda.get_device_properties(local_rank).multi_processor_count,
            ],
            dtype=torch.float64,
        )
        gathered_clocks = [torch.empty_like(local_clock) for _ in range(WORLD_SIZE)]
        torch.distributed.all_gather(
            gathered_clocks, local_clock, group=control_group
        )
        peer_p10_mhz = [float(value[0].item()) for value in gathered_clocks]
        peer_reference_mhz = [float(value[1].item()) for value in gathered_clocks]
        peer_maximum_mhz = [float(value[3].item()) for value in gathered_clocks]
        peer_sm_counts = [int(value[4].item()) for value in gathered_clocks]
        application_clock_ratios = [
            observed / reference
            for observed, reference in zip(peer_p10_mhz, peer_reference_mhz)
        ]
        all_above_minimum = all(bool(value[2].item()) for value in gathered_clocks)
        all_above_application_ratio = (
            min(application_clock_ratios) >= args.min_application_sm_clock_ratio
        )
        all_clocks_accepted = all_above_minimum and all_above_application_ratio
        clock_receipt.update({
            "peer_p10_mhz": peer_p10_mhz,
            "peer_default_application_mhz": peer_reference_mhz,
            "application_clock_ratios": application_clock_ratios,
            "minimum_application_clock_ratio": args.min_application_sm_clock_ratio,
            "accepted": all_clocks_accepted,
        })
        local_protocol = torch.tensor(
            int(post_timing_diagnostics["protocol_error"].item()),
            dtype=torch.int32,
        )
        torch.distributed.all_reduce(
            local_protocol, op=torch.distributed.ReduceOp.MAX, group=control_group
        )

        rank_record = {
            "schema": SCHEMA,
            "rank": rank,
            "local_rank": local_rank,
            "gpu": torch.cuda.get_device_name(local_rank),
            "gpu_id": _gpu_uuid(local_rank),
            "clock": clock_receipt,
            "communication": communication_receipt,
        }
        print("RANK_CLOCK_JSON=" + json.dumps(rank_record, sort_keys=True), flush=True)
        torch.distributed.barrier(group=control_group)

        if rank == 0:
            median_ms = statistics.median(max_rank_ms)
            status = (
                "accepted"
                if (
                    all_clocks_accepted
                    and int(local_protocol.item()) == 0
                    and int(network_accepted.item()) == 1
                )
                else "rejected"
            )
            result = {
                "schema": SCHEMA,
                "status": status,
                "rejection_reasons": [
                    reason
                    for reason, present in (
                        ("one or more GPU clocks were below the absolute minimum", not all_above_minimum),
                        (
                            "one or more GPU clocks were below the required fraction of their default application clock",
                            not all_above_application_ratio,
                        ),
                        (
                            "IB port inventory did not match the requested physical network roof",
                            int(network_accepted.item()) != 1,
                        ),
                        (f"protocol_error={int(local_protocol.item())}", int(local_protocol.item()) != 0),
                    )
                    if present
                ],
                "scenario": "prefill",
                "ep": WORLD_SIZE,
                "m_tokens_per_rank": args.m,
                "provenance": {
                    "source_sha256": _source_receipt(),
                    "fixture_recipe": RECIPE_ID,
                    "deep_gemm": deep_gemm.__version__,
                    "torch": torch.__version__,
                    "cuda": torch.version.cuda,
                    "session": state.session.properties,
                    "network_inventory": network_inventory,
                    "nccl_environment": {
                        name: os.environ.get(name)
                        for name in (
                            "NCCL_GIN_TYPE",
                            "NCCL_GIN_NCONNECTIONS",
                            "NCCL_IB_HCA",
                            "NCCL_NETDEVS_POLICY",
                            "NCCL_NET_PLUGIN",
                        )
                    },
                },
                "model": {
                    "hidden": HIDDEN,
                    "intermediate": INTERMEDIATE,
                    "experts": EXPERTS,
                    "top_k": TOP_K,
                    "w1": CONTRACT.w1,
                    "w2": CONTRACT.w2,
                },
                "precision": {
                    "shorthand": "W4A8",
                    "activation": "MXFP8 E4M3",
                    "weight": "MXFP4 E2M1",
                    "scale_group_k": K_GROUP,
                    "w1_boundary": "BF16",
                    "w2_boundary": "BF16",
                    "activation_clamp": ACTIVATION_CLAMP,
                    "fast_math": FAST_MATH,
                },
                "launch": {
                    "grid_ctas": state.grid_ctas,
                    "warmup": args.warmup,
                    "iterations": args.iterations,
                    "first_measured_epoch": first_measured_epoch,
                    "first_three_slots": epoch_slots([0, 1, 2]),
                },
                "max_rank_latency_ms": {
                    "min": min(max_rank_ms),
                    "median": statistics.median(max_rank_ms),
                    "mean": statistics.fmean(max_rank_ms),
                    "p90": _percentile(max_rank_ms, 0.90),
                    "max": max(max_rank_ms),
                },
                "max_rank_latency_us_median": median_ms * 1000.0,
                "effective_tflops": (
                    args.m
                    * TOP_K
                    * 6
                    * HIDDEN
                    * INTERMEDIATE
                    / (median_ms * 1.0e9)
                ),
                "effective_tflops_convention": (
                    "tokens_per_rank * top_k * 3 projections * 2 FLOP/MAC * hidden * "
                    "intermediate / max-rank latency"
                ),
                "clock_policy": {
                    "minimum_mhz": args.min_sm_clock_mhz,
                    "minimum_application_clock_ratio": args.min_application_sm_clock_ratio,
                    "observed_application_clock_ratios": application_clock_ratios,
                    "decision_statistic": "per-rank SM-clock p10 divided by default application clock",
                },
                "communication": _aggregate_communication(
                    complete_communication_by_rank
                ),
                "sol": _sol_receipt(
                    complete_communication_by_rank,
                    state.layout,
                    args.m,
                    median_ms,
                    args.empirical_compute_roof_tflops,
                    args.empirical_compute_roof_source,
                    peer_sm_counts,
                    peer_maximum_mhz,
                    args.network_port_count,
                    args.network_port_gbps,
                ),
                "contract": asdict(CONTRACT),
                "timestamp_unix": time.time(),
            }
            print("RESULT_JSON=" + json.dumps(result, sort_keys=True), flush=True)
        return (
            0
            if (
                all_clocks_accepted
                and int(local_protocol.item()) == 0
                and int(network_accepted.item()) == 1
            )
            else 2
        )
    finally:
        watchdog.arm("cleanup")
        if clock is not None and clock._thread is not None and clock._thread.is_alive():
            clock.stop()
        if state is not None:
            state.close()
            torch.distributed.barrier(group=control_group)
        torch.distributed.destroy_process_group(control_group)
        torch.distributed.destroy_process_group(group)
        watchdog.disarm()
        watchdog.close()


if __name__ == "__main__":
    raise SystemExit(main())
