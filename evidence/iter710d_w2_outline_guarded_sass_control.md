# Iteration 710d — guarded W2 phase-outline SASS control

## Question

Does the fused-only `AssumeValidMblock=true` template specialization cause
the outlined persistent W2 loop's 1:1 QGMMA/wait schedule?  The previously
audited standalone W2 specialization uses the defensive `false` path and
generates 32 QGMMAs with only 16 dependency waits.

## Configuration

- Compile-only H20 GPU1, selected TP4 bundle.
- `V4_SINGLE_LAUNCH_W2_PHASE_NOINLINE=1`.
- `V4_SINGLE_LAUNCH_ASSUME_VALID_GEMM_TASKS=0`.
- Extension: `v4tp_0e10226e8b3f3485da1e_v178mspec`.
- Exact M128/split-K2 phase callee symbol is the local
  `single_launch_w2_gemm_phase<false>` clone in code section 153.

## Result

- Main resources remain `REG56 STACK48 SHARED2048 LOCAL0`.
- Guarded callee grows slightly from 20,800 to 20,944 bytes.
- Exact callee SASS SHA256:
  `dc31fa292294a79602920f9e473b1c0c0f82c31927f523086d110a2b27ceb5be`.
- It still contains 32 QGMMAs and 32 `WARPGROUP.DEPBAR.LE` instructions.

No business kernel was launched and no performance result is claimed.

## Decision

**Reject the guard specialization as the cause.**  A task body with the same
template booleans as standalone still serializes when inlined into a
grid-stride phase callee.  The remaining actionable code-generation
difference is the outer persistent task loop itself.

Next build a compact per-task W2 call: reuse the existing CTA-shared pointer
record after W13/activation, pass only that pointer and one task index, and
keep W13 inline/compact as selected.  This creates a standalone-shaped W2
callee without the historical thirteen-pointer per-task ABI.  Require 32:16
QGMMA/wait in the callee and no M128 occupancy regression before any launch.

Raw artifacts:

- `bench/results/iter710d_w2_outline_guarded_jit.log`
- `bench/results/iter710d_w2_outline_guarded_resources.log`
- `bench/results/iter710d_w2_outline_guarded_m128_split2.sass`
