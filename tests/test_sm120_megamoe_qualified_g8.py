import hashlib
import json
import re
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
QUALIFIED = ROOT / "examples" / "sm120_megamoe" / "qualified_g8"
MANIFEST = json.loads((QUALIFIED / "qualification-manifest.json").read_text())

EXPECTED_CASES = [
    "world1-r1-distinct-balanced-c110",
    "world2-r1-distinct-balanced-c110",
    "world2-r1-zero-balanced-mask1-c110",
    "world2-r17-analytic-balanced-mask7-c110",
    "world2-r17-analytic-empty-mask0-c110",
    "world2-r17-analytic-skewed-mask0-c110",
    "world4-r128-distinct-balanced-mask0-c110",
    "world8-r113-distinct-balanced-mask0-c110",
    "world8-r2048-distinct-balanced-mask0-c110",
]


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_qualified_artifacts_match_manifest() -> None:
    assert MANIFEST["schema"] == "deepgemm-sm120-megamoe-qualified-g8-v1"
    for name, expected in MANIFEST["artifacts"].items():
        assert sha256(QUALIFIED / name) == expected

    donor = MANIFEST["direct_math_donor"]
    assert sha256(ROOT / donor["path"]) == donor["sha256"]


def test_selected_matrix_receipt_is_fail_closed() -> None:
    receipt_path = QUALIFIED / "selected-matrix-receipt.json"
    receipt = json.loads(receipt_path.read_text())
    assert sha256(receipt_path) == MANIFEST["correctness"]["receipt_sha256"]
    assert receipt["status"] == "pass"
    assert receipt["case_count"] == 9
    assert receipt["rank_record_count"] == 31
    assert receipt["cases"] == EXPECTED_CASES
    assert receipt["epoch_slots"] == [0, 1, 0]
    for field in (
        "all_case_exit_zero",
        "all_rank_status_pass",
        "all_exact_bf16",
        "all_fail_fields_zero",
    ):
        assert receipt[field] is True

    qualification = receipt["qualification"]
    assert qualification["runtime_replay_case_count"] == 7
    assert qualification["runtime_replay_rank_record_count"] == 15
    assert qualification["full_sass_equivalence_case_count"] == 2
    assert qualification["full_sass_equivalence_rank_record_count"] == 16
    assert len(qualification["full_sass_sha256"]) == 64


def test_kernel_and_hosts_have_no_staging_scaffolding() -> None:
    kernel = (
        QUALIFIED / "cake_sm120_megamoe_production_canonical_fused_ready_chunk8.cu"
    ).read_text()
    support_host = (
        QUALIFIED / "deepgemm_fp8_fp4_mega_moe_sm120_production_host.cu"
    ).read_text()
    runner = (QUALIFIED / "run_sm120_canonical_fused_ready_chunk8_perf.py").read_text()

    assert kernel.count("__global__") == 1
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
    equivalence = MANIFEST["performance"]["baseline_equivalence"]
    assert (
        equivalence["clean_full_sass_sha256"]
        == equivalence["reference_full_sass_sha256"]
    )
    assert abs(equivalence["mean_delta_percent"]) < 1.0
    assert equivalence["all_pre_post_correctness_checks_passed"] is True
