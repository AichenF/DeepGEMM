# SM90 FP8 H200 Conservative Selector Design

## Goal

Productize the H200 FP8 MegaMoE tuning results that have already passed
performance and correctness checks, while keeping all H20, H100, and generic
SM90 behavior unchanged.

The final selector covers the Flash shape `(H=4096, I=2048, E=256, top-k=6)`
and Pro shape `(H=7168, I=3072, E=384, top-k=6)`.  `M=512` is explicitly out
of scope and must fall through to the existing default path.

## Decisions

- Use the existing global BF16 scaled-accumulation path for previously
  validated Pro candidates.
- Do not add phase-specific BF16 accumulation.
- Do not implement or copy PR323's fused kernel.
- Do not change any H20-tuned selector entry.
- Commit the completed work locally, but do not push it.

## Approaches considered

### 1. Conservative evidence-only selector (selected)

Match H200, the exact model shape, and only the measured M points or one-sided
large-M bucket supported by repeated data.  Leave all other cases on the
existing selector.  This minimizes the blast radius and makes every automatic
choice traceable to an H200 benchmark.

### 2. Interpolated M ranges

Apply a measured candidate to ranges between benchmark points.  This improves
coverage for arbitrary M values but has no direct performance evidence at the
intermediate points and risks hidden regressions.  Reject for this change.

### 3. Rebuild from the pre-experiment baseline

Reapply only winning kernel mechanisms onto `3552b62`.  This could produce the
smallest final diff, but it would reintroduce substantial integration and
correctness risk.  Defer cleanup until the selector is proven.

## Device and workload matching

Identify H200 explicitly from `cudaDeviceProp::name` containing `H200`; do not
reuse the existing generic `HighSm` bucket because H100 and H200 can have the
same SM count.

Automatic overrides require a full workload match:

- Flash: eight ranks, 256 total experts, H4096, I2048, top-k6.
- Pro: eight ranks, 384 total experts, H7168, I3072, top-k6.

An environment override remains available for experiments, but production
behavior must not depend on one.  Explicit experiment overrides take
precedence over automatic defaults.

## Selector policy

### Flash

- `M < 128`: unchanged default path.
- `M = 128`: retained E5M2, non-direct, N-major, EPW4 candidate.
- `M = 256`: unchanged retained default path.
- `M = 512`: unchanged default path; excluded from the acceptance gate.
- `M = 1024, 2048, 4096`: unchanged retained default path.
- `M = 8192`: retained E5M2, non-direct, N-major, stage3, EPW32 candidate.
- All unmeasured M values: unchanged default path.

### Pro

- `M < 128`: unchanged default path.
- `M = 128`: retained L1-BN512/L2-BN256 global-BF16 candidate.
- `M = 256`: retained L1-BK256/L2-BK128, stage2/3, E5M2,
  L1/L2-EPW8/48, 128-SM candidate.
- `M = 512`: unchanged default path; excluded from the acceptance gate.
- `M = 1024, 2048, 4096, 8192`: retained L1-BN512/L2-BN256 global-BF16
  candidate.
- All unmeasured M values: unchanged default path.

The implementation should represent these choices as a small structured H200
policy rather than scattered shape checks.  Phase tile/stage/wave/SM overrides
and global execution features are applied from that policy before JIT source
generation.

## Numerical gate for global BF16

Before enabling a Pro BF16 entry automatically, compare on identical inputs:

1. FP32 accumulation against the golden reference.
2. Global BF16 accumulation against the same golden reference.
3. Global BF16 output directly against FP32 output.

Run the actual Pro shape at M128/1024/2048/4096/8192 across multiple seeds.
Every rank must remain finite and `calc_diff < 0.01`; record the maximum and
worst seed rather than only pass/fail.

## Validation

- Re-run Flash and Pro M8--8192, excluding M512 from the faster-than-PR323
  requirement.
- For every other M >= 128, require repeated max-rank median-20 latency below
  PR323.
- For M8/16/32/64, require no confirmed regression against `3552b62`.
- Run exact-shape correctness for every newly selected calculation path.
- Add selector/config tests showing H200 receives the new policy and H20,
  H100, generic SM90, unmatched shapes, and M512 retain existing behavior.

## Cleanup and commits

Land the work in separable local commits:

1. Approved design and implementation plan.
2. H200 detection and structured selector policy.
3. Selector tests and exact-shape numerical tests.
4. H200 performance confirmation.
5. Removal of rejected default-off experiments that are not required by any
   retained path, followed by the full regression gate.

Do not push any commit or branch.
