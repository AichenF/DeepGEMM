# SM90 FP8 MegaMoE H200 Retuning Design

## Objective

Retune the existing SM90 FP8 MegaMoE implementation for NVIDIA H200 on the
Flash and Pro model shapes. For every benchmark point with `M >= 128`, the
retuned implementation must be faster than the pinned PR323 FP8 reference.
Points with `M < 128` must not show a confirmed regression from the current
`megamoe_sm90_opt` baseline.

This work is local-only. Commits are allowed and no branch will be pushed.

## Scope

The implementation under test is the existing split two-kernel FP8 path on
the `megamoe_sm90_opt` line:

- Linear 1: dispatch, GEMM, SwiGLU, and FP8 intermediate output;
- Linear 2: GEMM, BF16 contribution scatter, and combine.

Only existing legal configuration dimensions and their H200 selector are in
scope. The work may tune block shapes, pipeline depth, expert waves, warp
allocation, and existing scheduler/epilogue modes. It may add H200-specific
selector buckets and tests, but it must not import, restore, or implement the
PR323 fused single-kernel path.

NVFP4 is explicitly out of scope for this iteration.

## Hardware Isolation

All new performance decisions must be measured on 8x NVIDIA H200. Historical
H20 results may explain old defaults but are not valid evidence for retaining
or selecting a parameter on H200.

The new selector must identify H200 explicitly. It must not apply H200-tuned
parameters to every high-SM-count SM90 device. Existing behavior on H20,
H100, and generic SM90 devices remains unchanged.

## Shapes and Points

| Shape | Hidden | Intermediate | Experts | Top-k |
|---|---:|---:|---:|---:|
| Flash | 4096 | 2048 | 256 | 6 |
| Pro | 7168 | 3072 | 384 | 6 |

The required M points are:

`8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192`.

## References and Timing Boundary

- Current baseline: commit `3552b62545e3602d60bde6ed3542934f6dcf6232`.
- PR323 reference: exact upstream head
  `8ddf7f96cb3300011f69458e88c7651a1e305a8c` plus the syntax-only CUDA
  13.2 compatibility commit `184d74cb052a028ac9d960d65abd35ec231146df`.
- Both implementations use the same input tensors, routing seed, capacity,
  rank topology, and no-L2-flush timing protocol.
- The FP8 split baseline is timed as the sum of its matched L1 and L2 kernel
  events. PR323 is timed as its single full-MoE kernel event.
- The decision metric is max-rank latency. Rank-zero latency is retained as a
  diagnostic metric.

## Search Strategy

The search is parameter-first and does not create a new kernel architecture.

1. Reproduce the pinned baseline and PR323 results on the new H200 allocation.
2. Run targeted single-axis screens around the current selector. Pro begins
   with `direct_l2_scatter=0` at `M >= 512`, because the current selector
   enables the 32-bit direct-store path at exactly that boundary.
3. Search legal combinations of:
   - block M and block N;
   - pipeline stages;
   - experts per wave;
   - dispatch warps and epilogue warpgroups;
   - direct L2 scatter;
   - L2 N-major scheduling;
   - one-warp cleanup;
   - swapAB at legal small-M points.
4. Use a bounded beam search instead of a full Cartesian product. Retain the
   best configurations by max-rank latency for each shape/load bucket, then
   validate neighboring M points before encoding a selector rule.
5. Express production choices using shape identity and
   `expected_tokens_per_expert` buckets. Do not hard-code an isolated raw M
   unless no stable neighboring bucket exists and the point is required by the
   serving contract.

Temporary environment variables may force candidates during experiments.
The final production selector must not require a new environment variable.

## Benchmark Stages

### Fast signal

- one routing seed;
- balanced A/B order where practical;
- median of 10 timed samples;
- fresh JIT cache per configuration identity;
- reject crashes, incomplete rank output, or routing mismatches immediately.

### Candidate confirmation

- seeds 101, 202, and 303;
- two balanced process orders per implementation;
- median of 20 or more timed samples;
- identical routing across implementations;
- retain only repeatable wins.

### Final gate

- seeds 101, 202, and 303;
- two observations per implementation and seed;
- balanced process ordering;
- median of 30 timed samples;
- all eight rank records required for every observation.

For `M >= 128`, the geometric max-rank center must be lower than PR323 at
every point. A nominal win smaller than 1% is treated as unconfirmed and gets
additional repetitions. For `M < 128`, a repeatable regression larger than 1%
from the pinned baseline rejects the selector change.

## Correctness and Safety

- Run the existing SM90 FP8 MegaMoE correctness suite before performance
  tuning and after every promoted selector change.
- Exercise Flash and Pro boundary points, masked routes, multiple top-k values,
  zero-token paths, activation clamp, and both fast-math modes where supported
  by the existing suite.
- Verify all outputs are finite and within the existing FP8 tolerance.
- Confirm the generated configuration and JIT template arguments for every
  required M point.
- Preserve the CUDA 13.2 dependent-template compatibility fix.

## Artifacts and Git Policy

The optimization branch is
`opt/megamoe-sm90-fp8-h200-retune`, based on the CUDA 13.2-compatible
`megamoe_sm90_opt` worktree. The workspace will retain:

- `HINTS.md` with the user constraints;
- `ITERATIONS.md` with every measured iteration;
- reproducible sweep and final-validation scripts;
- raw benchmark logs and summarized CSV/Markdown reports;
- exact source commits, generated configuration records, and environment
  metadata.

Each completed iteration may be committed. No commit or branch will be pushed.
