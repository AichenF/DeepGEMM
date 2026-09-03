import argparse
import importlib
import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
RUNNER_DIR = ROOT / "examples" / "sm120_megamoe" / "dsv4_flash"
sys.path.insert(0, str(RUNNER_DIR))

correctness = importlib.import_module("run_correctness_matrix")
performance = importlib.import_module("run_perf_abba")
runner_common = importlib.import_module("runner_common")


def make_build(tmp_path: Path, *flags: str, binary: str = "perf") -> Path:
    build = tmp_path / "build"
    build.mkdir()
    (build / binary).write_bytes(b"artifact")
    (build / "config-flags.txt").write_text(" ".join(flags))
    return build


def rank_record(rank: int) -> dict:
    record = {
        "rank": rank,
        "status": "pass",
        "exact_bf16_equal": True,
        "epoch_slots": [0, 1, 0],
        "epoch_route_totals": [24, 24, 24],
        "stage_mismatches_per_epoch": [[0, 0], [0, 0], [0, 0]],
        "launch_count_per_epoch": 1,
    }
    record.update({counter: 0 for counter in correctness.ZERO_COUNTERS})
    return record


def write_trace(path: Path, records: list[dict]) -> None:
    path.write_text("".join(json.dumps(record) + "\n" for record in records))


def test_parse_gpus_requires_an_exact_unique_assignment() -> None:
    assert runner_common.parse_gpus("0,1,2,3", 4) == [0, 1, 2, 3]
    with pytest.raises(ValueError, match="exactly 4 GPUs"):
        runner_common.parse_gpus("0,1", 4)
    with pytest.raises(ValueError, match="duplicates"):
        runner_common.parse_gpus("0,0", 2)
    with pytest.raises(ValueError, match="no topology mapping"):
        runner_common.parse_gpus("8", 1)


def test_topology_uses_the_local_hca_and_numa_node() -> None:
    placement = runner_common.topology([0, 1, 2, 3])
    assert placement["hcas"] == "mlx5_2,mlx5_3,mlx5_0,mlx5_1"
    assert placement["cpuset"] == "0-55,112-167"
    assert placement["memset"] == "0"
    assert placement["devices"] == [
        "--device=/dev/infiniband/rdma_cm",
        "--device=/dev/infiniband/uverbs2",
        "--device=/dev/infiniband/uverbs3",
        "--device=/dev/infiniband/uverbs0",
        "--device=/dev/infiniband/uverbs1",
    ]


def test_validate_build_binds_the_formal_ep4_shape(tmp_path: Path) -> None:
    build = make_build(tmp_path, "-DCAKE_MOE_LOCAL_EXPERTS=64")
    identity = runner_common.validate_build(
        build,
        binary="perf",
        world_size=4,
        rows=2048,
        hidden=4096,
        intermediate=4096,
        experts=256,
        topk=6,
    )
    assert identity["binary_sha256"] == runner_common.file_sha256(build / "perf")
    assert identity["config"]["CAKE_MOE_LOCAL_EXPERTS"] == 64

    with pytest.raises(ValueError, match="CAKE_MOE_INTERMEDIATE=4096"):
        runner_common.validate_build(
            build,
            binary="perf",
            world_size=4,
            rows=2048,
            hidden=4096,
            intermediate=2048,
            experts=256,
            topk=6,
        )
    with pytest.raises(ValueError, match="gives 256 experts"):
        runner_common.validate_build(
            build,
            binary="perf",
            world_size=4,
            rows=2048,
            hidden=4096,
            intermediate=4096,
            experts=128,
            topk=6,
        )


def test_default_build_matches_ep8(tmp_path: Path) -> None:
    build = make_build(tmp_path)
    identity = runner_common.validate_build(
        build,
        binary="perf",
        world_size=8,
        rows=2048,
        hidden=4096,
        intermediate=4096,
        experts=256,
        topk=6,
    )
    assert identity["config"]["CAKE_MOE_LOCAL_EXPERTS"] == 32


def test_correctness_matrix_and_fail_closed_evaluation() -> None:
    cases = correctness.case_matrix(4)
    assert len(cases) == 13
    assert {case["world_size"] for case in cases} == {4}

    case = cases[0]
    ranks = [rank_record(rank) for rank in range(4)]
    stdout = "".join(f"RANK_RESULT_JSON={json.dumps(record)}\n" for record in ranks)
    stdout += 'RESULT_JSON={"status":"pass","failures":0}\n'
    ok, reasons, parsed = correctness.evaluate(stdout, case)
    assert ok
    assert not reasons
    assert parsed == ranks

    ranks[2]["output_mismatches"] = 1
    bad_stdout = "".join(f"RANK_RESULT_JSON={json.dumps(record)}\n" for record in ranks)
    bad_stdout += 'RESULT_JSON={"status":"pass","failures":0}\n'
    ok, reasons, _ = correctness.evaluate(bad_stdout, case)
    assert not ok
    assert "rank 2 output_mismatches=1" in reasons

    duplicate_ranks = [rank_record(0) for _ in range(4)]
    duplicate_stdout = "".join(
        f"RANK_RESULT_JSON={json.dumps(record)}\n" for record in duplicate_ranks
    )
    duplicate_stdout += 'RESULT_JSON={"status":"pass","failures":0}\n'
    ok, reasons, _ = correctness.evaluate(duplicate_stdout, case)
    assert not ok
    assert any(reason.startswith("expected rank IDs 0..3") for reason in reasons)


def test_docker_gpu_argument_does_not_contain_shell_quotes(tmp_path: Path) -> None:
    case = correctness.case_matrix(4)[0]
    command = correctness.docker_command(tmp_path, [0, 1, 2, 3], case)
    gpu_argument = command[command.index("--gpus") + 1]
    assert gpu_argument == "device=0,1,2,3"
    assert '"' not in gpu_argument


def test_iteration_envelopes_include_multi_kernel_gaps(tmp_path: Path) -> None:
    trace = tmp_path / "trace.jsonl"
    records = []
    for iteration in range(3):
        base = iteration * 10_000_000
        for device, duration in ((0, 4_000_000), (1, 5_000_000)):
            records.extend(
                [
                    {
                        "device": device,
                        "name": "dispatch_kernel",
                        "start_ns": base,
                        "end_ns": base + 1_000_000,
                    },
                    {
                        "device": device,
                        "name": "compute_kernel",
                        "start_ns": base + 2_000_000,
                        "end_ns": base + duration,
                    },
                ]
            )
    write_trace(trace, records)

    result = performance.iteration_envelopes(
        trace, ["dispatch_kernel", "compute_kernel"], 2, skip=1, count=2
    )
    assert result["per_rank_ms"] == {"0": [4.0, 4.0], "1": [5.0, 5.0]}
    assert result["max_rank_ms"] == [5.0, 5.0]
    assert result["critical_rank_per_sample"] == [1, 1]


def test_iteration_envelopes_reject_an_invalid_plan(tmp_path: Path) -> None:
    trace = tmp_path / "trace.jsonl"
    write_trace(
        trace,
        [
            {"device": 0, "name": "compute", "start_ns": 0, "end_ns": 2},
            {"device": 0, "name": "dispatch", "start_ns": 3, "end_ns": 4},
        ],
    )
    result = performance.iteration_envelopes(
        trace, ["dispatch", "compute"], 1, skip=0, count=1
    )
    assert "does not match the declared sequence" in result["error"]

    duplicate = performance.iteration_envelopes(
        trace, ["compute", "compute"], 1, skip=0, count=1
    )
    assert duplicate == {
        "error": "expected kernel sequence must be nonempty and unique"
    }


def test_useful_tflops_counts_both_routed_gemms() -> None:
    args = argparse.Namespace(
        rows=2048,
        world_size=4,
        topk=6,
        hidden=4096,
        intermediate=4096,
    )
    expected_flops = 6 * 2048 * 4 * 6 * 4096 * 4096
    assert performance.useful_tflops(args, 8.0) == expected_flops / 0.008 / 1e12
