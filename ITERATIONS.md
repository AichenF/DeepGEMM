# AKO Iterations: SM90 FP8 MegaMoE H200 Retuning

## Objective and fixed references

- Optimization branch: `opt/megamoe-sm90-fp8-h200-retune`.
- Current FP8 baseline: `3552b62545e3602d60bde6ed3542934f6dcf6232`.
- PR323 upstream head: `8ddf7f96cb3300011f69458e88c7651a1e305a8c`.
- PR323 CUDA 13.2 syntax-only fix: `184d74cb052a028ac9d960d65abd35ec231146df`.
- Hardware evidence accepted for new decisions: 8x NVIDIA H200 only.
- Architecture constraint: retain the existing split L1/L2 FP8 kernels; no
  PR323 fused-kernel implementation.
- Success gate: every Flash/Pro M >= 128 point faster than PR323; M < 128 no
  confirmed regression from the current FP8 baseline.

## Prior evidence motivating iteration 1

- The previous three-way H200 matrix showed Pro gaps versus PR323 of 1.26%,
  1.18%, 11.88%, 19.16%, 19.99%, 27.79%, and 21.81% at M 128 through 8192.
- The Pro selector enables direct L2 scatter when expected tokens per expert
  reach 64, which is exactly M=512 for top-k 6 and 48 local experts.
- The direct path emits 32-bit remote stores, while the non-direct path packs
  128-bit stores. This is the first H200-only parameter attribution to test.
- At Pro M >= 2048, ours and PR323 already use the same main tile, stage count,
  thread layout, SM count, and experts per wave. Therefore the first search
  focuses on existing epilogue/scheduler modes rather than only BM/BN.

## Iteration record template

Each measured source or promoted-selector iteration records:

1. hypothesis and exact source/config change;
2. hardware, source commits, command, routes, and cache identity;
3. correctness result;
4. rank-zero and max-rank latency versus current best and PR323;
5. decision: retain, reject, or gather more repetitions;
6. raw artifact paths and commit hash.
