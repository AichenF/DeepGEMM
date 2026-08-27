import hashlib
import json
import re
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
QUALIFIED = ROOT / "examples" / "sm120_megamoe" / "qualified_g8"
MANIFEST = json.loads((QUALIFIED / "qualification-manifest.json").read_text())

EXPECTED_CASES = [
    "world1-r1-distinct-balanced-mask0",
    "world2-r1-distinct-balanced-mask0",
    "world2-r1-zero-balanced-mask1",
    "world2-r17-analytic-balanced-mask7",
    "world2-r17-analytic-empty-mask0",
    "world2-r17-analytic-skewed-mask0",
    "world4-r128-distinct-balanced-mask0",
    "world8-r113-distinct-balanced-mask0",
    "world8-r2048-distinct-balanced-mask0",
]


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_qualified_artifacts_match_manifest() -> None:
    assert MANIFEST["schema"] == "deepgemm-sm120-megamoe-qualified-g8-v3"
    for name, expected in MANIFEST["artifacts"].items():
        assert sha256(QUALIFIED / name) == expected

    donor = MANIFEST["direct_math_donor"]
    assert sha256(ROOT / donor["path"]) == donor["sha256"]

    kernel_hash = MANIFEST["artifacts"][
        "cake_sm120_megamoe_production_canonical_fused_ready_chunk8.cu"
    ]
    provenance = MANIFEST["source_provenance"]
    lineage = MANIFEST["reference_lineage"]
    assert provenance["kind"] == "gpu_qualified_flattened_cuda"
    assert provenance["selected_source_sha256"] == kernel_hash
    assert provenance["host_sources_byte_identical"] is True
    assert lineage["selected_w2_scheduler"]["sha256"] == kernel_hash
    assert (
        lineage["selected_three_stage"]["sha256"]
        == provenance["base_stage3_source_sha256"]
    )
    assert lineage["flattened_deepgemm_reference"]["repository_commit"].startswith(
        "a35b6975"
    )


def test_selected_matrix_receipt_is_fail_closed() -> None:
    receipt_path = QUALIFIED / "selected-matrix-receipt.json"
    receipt = json.loads(receipt_path.read_text())
    assert sha256(receipt_path) == MANIFEST["correctness"]["receipt_sha256"]
    assert (
        receipt["schema"]
        == "deepgemm-sm120-megamoe-w2-scheduler-correctness-v1"
    )
    assert (
        receipt["source_sha256"]
        == MANIFEST["artifacts"][
            "cake_sm120_megamoe_production_canonical_fused_ready_chunk8.cu"
        ]
    )
    assert receipt["binary_sha256"] == MANIFEST["build"]["correctness_binary_sha256"]
    assert receipt["status"] == "pass"
    assert receipt["case_count"] == 9
    assert receipt["rank_record_count"] == 31
    for field in (
        "all_exact_bf16",
        "all_exit_status_zero",
        "all_fail_fields_zero",
        "all_stderr_empty",
    ):
        assert receipt[field] is True

    assert list(receipt["cases"]) == EXPECTED_CASES
    assert sum(case["rank_count"] for case in receipt["cases"].values()) == 31
    for case in receipt["cases"].values():
        assert case["exit_status"] == 0
        assert case["rank_count"] == len(case["routes"])
        assert all(route >= 0 for route in case["routes"])
        assert case["stderr_bytes"] == 0


def test_w2_scheduler_performance_receipt_is_fail_closed() -> None:
    receipt_path = QUALIFIED / "w2-scheduler-performance-receipt.json"
    receipt = json.loads(receipt_path.read_text())
    perf = MANIFEST["performance"]
    assert sha256(receipt_path) == perf["receipt_sha256"]
    assert (
        receipt["schema"]
        == "deepgemm-sm120-megamoe-w2-scheduler-performance-v1"
    )
    assert receipt["status"] == "pass"
    assert receipt["decision"] == "retain_w2_scheduler_optimization"

    candidate = receipt["artifacts"]["candidate"]
    assert (
        candidate["source_sha256"]
        == MANIFEST["artifacts"][
            "cake_sm120_megamoe_production_canonical_fused_ready_chunk8.cu"
        ]
    )
    assert (
        candidate["correctness_binary_sha256"]
        == MANIFEST["build"]["correctness_binary_sha256"]
    )
    assert (
        candidate["performance_binary_sha256"]
        == MANIFEST["build"]["performance_binary_sha256"]
    )
    assert (
        receipt["correctness"]["receipt_sha256"]
        == MANIFEST["correctness"]["receipt_sha256"]
    )

    workload = receipt["workload"]
    assert workload["world_size"] == perf["world_size"] == 8
    assert workload["active_rows"] == perf["active_rows"] == 2048
    assert workload["warmup_launches_per_run"] == perf["warmup_launches_per_run"] == 5

    assert workload["repeat_launches_per_run"] == 100
    assert workload["sample_count_per_variant"] == 200
    assert receipt["comparison"]["both_candidate_positions_faster"] is True
    assert receipt["comparison"]["latency_gain_percent"] >= 3.0
    assert receipt["stability"]["pass"] is True
    assert receipt["pooled"]["candidate"]["mean_ms"] == perf["candidate_mean_ms"]
    assert (
        receipt["pooled"]["candidate"]["aggregate_tensor_tflops"]
        == perf["candidate_aggregate_tensor_tflops"]
    )


def test_kernel_and_hosts_have_no_staging_scaffolding() -> None:
    kernel = (
        QUALIFIED / "cake_sm120_megamoe_production_canonical_fused_ready_chunk8.cu"
    ).read_text()
    support_host = (
        QUALIFIED / "deepgemm_fp8_fp4_mega_moe_sm120_production_host.cu"
    ).read_text()
    runner = (QUALIFIED / "run_sm120_canonical_fused_ready_chunk8_perf.py").read_text()

    assert kernel.count("__global__") == 1
    stage_signature = "64, 128, 128, 128, 128, 0, 3, 128, 256, 109,"
    old_stage_signature = "64, 128, 128, 128, 128, 0, 2, 128, 256, 109,"
    assert kernel.count(stage_signature) == 2
    assert old_stage_signature not in kernel
    assert kernel.count("cake_sm120_g8_capture_epoch_baselines") == 2
    assert kernel.count("cake_sm120_ready_mirror_pair_cached(") == 3
    assert "cake_sm120_ready_mirror_pair(" not in kernel
    assert kernel.count("cake_sm120_canonical_ready_publish_w2_chunk(") == 2
    assert re.search(r"\bint\s+main\s*\(", support_host) is None
    assert "run_rank" not in support_host
    assert "importlib" not in runner
    assert "coarse_perf" not in runner

    combined = kernel + support_host + runner
    for token in (
        "ReSharper",
        "correctness_main_unused",
        "maybe_unused",
        "TODO",
        "FIXME",
    ):
        assert token not in combined


def test_qualified_directory_contains_only_reviewed_sources() -> None:
    expected = {
        "README.md",
        "cake_sm120_megamoe_production_canonical_fused_ready_chunk8.cu",
        "deepgemm_fp8_fp4_mega_moe_sm120_production_canonical_fused_ready_chunk8_host.cu",
        "deepgemm_fp8_fp4_mega_moe_sm120_production_canonical_fused_ready_chunk8_perf_host.cu",
        "deepgemm_fp8_fp4_mega_moe_sm120_production_host.cu",
        "qualification-manifest.json",
        "run_sm120_canonical_fused_ready_chunk8_perf.py",
        "selected-matrix-receipt.json",
        "w2-scheduler-performance-receipt.json",
    }
    assert {path.name for path in QUALIFIED.iterdir() if path.is_file()} == expected


def test_formal_qualification_limits_remain_explicit() -> None:
    assert MANIFEST["formal_qualification"] == {
        "functional_qualified": False,
        "resource_qualified": False,
        "performance_qualified": False,
        "shared_expert_in_scope": False,
    }
    perf = MANIFEST["performance"]
    assert perf["single_launch_full_chain"] is True
    assert perf["branch_latency_change_percent"] <= -3.0
    assert all(gain > 0.0 for gain in perf["independent_position_gains_percent"])
    assert perf["stability"]["pass"] is True
