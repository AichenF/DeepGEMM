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
    assert MANIFEST["schema"] == "deepgemm-sm120-megamoe-qualified-g8-v2"
    for name, expected in MANIFEST["artifacts"].items():
        assert sha256(QUALIFIED / name) == expected

    donor = MANIFEST["direct_math_donor"]
    assert sha256(ROOT / donor["path"]) == donor["sha256"]
    lineage = MANIFEST["reference_lineage"]
    assert lineage["flattened_deepgemm_reference"]["repository_commit"].startswith(
        "a35b6975"
    )
    assert lineage["flattened_deepgemm_reference"]["sha256"].startswith("dffd9acd")
    assert lineage["typed_two_stage"]["cake_commit"].startswith("177d3a72")
    assert (
        lineage["typed_two_stage"]["sha256"]
        == MANIFEST["performance"]["typed_two_stage_source_sha256"]
    )
    assert (
        lineage["selected_three_stage"]["cake_commit"]
        == MANIFEST["cake_ir_generation"]["commit"]
    )
    assert (
        lineage["selected_three_stage"]["sha256"]
        == MANIFEST["artifacts"][
            "cake_sm120_megamoe_production_canonical_fused_ready_chunk8.cu"
        ]
    )


def test_selected_matrix_receipt_is_fail_closed() -> None:
    receipt_path = QUALIFIED / "selected-matrix-receipt.json"
    receipt = json.loads(receipt_path.read_text())
    assert sha256(receipt_path) == MANIFEST["correctness"]["receipt_sha256"]
    assert receipt["schema"] == "cake-sm120-megamoe-stage3-correctness-matrix-v1"
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
    assert list(receipt["cases"]) == EXPECTED_CASES
    for field in ("all_exact_bf16", "all_fail_fields_zero"):
        assert receipt[field] is True
    assert sum(case["rank_count"] for case in receipt["cases"].values()) == 31
    for case in receipt["cases"].values():
        assert case["rank_count"] == len(case["routes"])
        assert all(route >= 0 for route in case["routes"])
        assert case["stderr_bytes"] == 0


def test_stage3_performance_receipt_is_fail_closed() -> None:
    receipt_path = QUALIFIED / "stage3-performance-receipt.json"
    receipt = json.loads(receipt_path.read_text())
    assert sha256(receipt_path) == MANIFEST["performance"]["receipt_sha256"]
    assert receipt["schema"] == "cake-sm120-megamoe-stage3-paired-performance-v1"
    assert receipt["status"] == "pass"
    assert receipt["world_size"] == 8
    assert receipt["active_rows"] == 2048
    assert receipt["run_count_per_variant"] == 4
    assert receipt["repeat_per_run"] == 100
    assert receipt["pooled_current"]["sample_count"] == 400
    assert receipt["pooled_stage3"]["sample_count"] == 400
    assert (
        receipt["candidate"]["source_sha256"]
        == MANIFEST["artifacts"][
            "cake_sm120_megamoe_production_canonical_fused_ready_chunk8.cu"
        ]
    )
    assert (
        receipt["candidate"]["binary_sha256"]
        == MANIFEST["build"]["performance_binary_sha256"]
    )
    assert (
        receipt["full_correctness_matrix"]["receipt_sha256"]
        == MANIFEST["correctness"]["receipt_sha256"]
    )
    assert receipt["stage3_vs_current_percent"] < -3.0


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
        "stage3-performance-receipt.json",
    }
    assert {path.name for path in QUALIFIED.iterdir() if path.is_file()} == expected


def test_formal_qualification_limits_remain_explicit() -> None:
    formal = MANIFEST["formal_qualification"]
    assert formal == {
        "functional_qualified": False,
        "resource_qualified": False,
        "performance_qualified": False,
        "shared_expert_in_scope": False,
    }
    assert MANIFEST["performance"]["single_launch_full_chain"] is True
    assert MANIFEST["performance"]["stage3_vs_current_percent"] < -3.0
    assert (
        MANIFEST["performance"]["candidate_mean_ms"]
        < MANIFEST["performance"]["typed_two_stage_mean_ms"]
    )
