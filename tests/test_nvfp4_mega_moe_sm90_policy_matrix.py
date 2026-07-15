#!/usr/bin/env python3
"""Fresh-process correctness matrix for SM90 NVFP4 policy boundaries."""

import argparse
import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
CORRECTNESS_TEST = REPO_ROOT / "tests" / "test_nvfp4_mega_moe_sm90_correctness.py"
sys.path.insert(0, str(REPO_ROOT))

POLICY_CASES = {
    "flash_fused_boundaries": [
        "--batches", "15", "16", "42", "43",
        "--hidden", "4096",
        "--intermediate-hidden", "2048",
        "--num-experts", "32",
        "--num-topk", "6",
        "--num-max-tokens-per-rank", "8192",
        "--nvfp4-block-n", "256",
    ],
    "pro_fused_boundaries": [
        "--batches", "63", "64", "65",
        "--hidden", "7168",
        "--intermediate-hidden", "3072",
        "--num-experts", "48",
        "--num-topk", "6",
        "--num-max-tokens-per-rank", "8192",
        "--nvfp4-block-n", "256",
    ],
    "middle_fused_boundaries": [
        "--batches", "63", "64", "65",
        "--hidden", "5120",
        "--intermediate-hidden", "2560",
        "--num-experts", "48",
        "--num-topk", "6",
        "--num-max-tokens-per-rank", "8192",
        "--nvfp4-block-n", "256",
    ],
    "flash_layout_cutoff": [
        "--batches", "384", "385",
        "--hidden", "1024",
        "--intermediate-hidden", "2048",
        "--num-experts", "8",
        "--num-topk", "4",
        "--num-max-tokens-per-rank", "8192",
    ],
    "pro_layout_cutoff": [
        "--batches", "380", "381",
        "--hidden", "1024",
        "--intermediate-hidden", "3072",
        "--num-experts", "8",
        "--num-topk", "4",
        "--num-max-tokens-per-rank", "8192",
    ],
    "forced_bn128_boundaries": [
        "--batches", "64", "128", "192", "256", "384",
        "--hidden", "1024",
        "--intermediate-hidden", "1024",
        "--num-experts", "8",
        "--num-topk", "4",
        "--num-max-tokens-per-rank", "8192",
        "--nvfp4-block-n", "128",
    ],
}

COMMON_ARGS = [
    "--weight-scales", "0.05",
    "--global-scale-modes", "none", "expert",
]


def check_layout_policy() -> None:
    from deep_gemm.mega import choose_nvfp4_block_n_for_mega_moe_sm90

    cases = [
        # Flash and middle use the measured expected-192 cutoff.
        (1024, 6, 32, 2048, 256),
        (1025, 6, 32, 2048, 128),
        (1536, 6, 48, 2560, 256),
        (1537, 6, 48, 2560, 128),
        # Pro crosses between expected 190 and 192.
        (1520, 6, 48, 3072, 256),
        (1521, 6, 48, 3072, 128),
        # MiMo 2.5 Pro uses the measured all-range fused layout.
        (1, 8, 48, 2048, 256),
        (8192, 8, 48, 2048, 256),
    ]
    for num_tokens, topk, local_experts, intermediate, expected_block_n in cases:
        actual_block_n = choose_nvfp4_block_n_for_mega_moe_sm90(
            num_tokens, topk, local_experts, intermediate
        )
        assert actual_block_n == expected_block_n, (
            num_tokens,
            topk,
            local_experts,
            intermediate,
            actual_block_n,
            expected_block_n,
        )
    print("Layout policy selection: PASS", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run isolated SM90 NVFP4 policy-boundary correctness cases"
    )
    parser.add_argument(
        "--cases",
        nargs="+",
        choices=tuple(POLICY_CASES),
        default=list(POLICY_CASES),
    )
    parser.add_argument(
        "--jit-cache-root",
        type=Path,
        default=Path("/tmp/deep_gemm_nvfp4_policy_matrix"),
    )
    args = parser.parse_args()
    check_layout_policy()

    run_id = datetime.now().strftime("%Y%m%d_%H%M%S") + f"_{os.getpid()}"
    run_root = args.jit_cache_root / run_id
    run_root.mkdir(parents=True, exist_ok=False)

    for case_name in args.cases:
        env = os.environ.copy()
        env["DG_JIT_CACHE_DIR"] = str(run_root / case_name)
        env.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
        command = [
            sys.executable,
            str(CORRECTNESS_TEST),
            *POLICY_CASES[case_name],
            *COMMON_ARGS,
        ]
        print(f"[RUN] {case_name}", flush=True)
        subprocess.run(command, cwd=REPO_ROOT, env=env, check=True)
        print(f"[PASS] {case_name}", flush=True)

    print(f"Policy matrix PASS; fresh JIT caches: {run_root}", flush=True)


if __name__ == "__main__":
    main()
