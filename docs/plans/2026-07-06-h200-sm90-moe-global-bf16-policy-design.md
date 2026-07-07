# H200 SM90 MegaMoE Global BF16 Policy Design

## Objective

Make the H200 SM90 FP8 MegaMoE selector easier to reason about by separating
schedule selection from numerical-format selection.  Global packed-BF16x2
scaled accumulation should be the default for every validated point, rather
than being embedded in a few exact-M schedule rows.

The validation matrix is:

- Flash: H=4096, I=2048, E=256, top-k=6
- Pro: H=7168, I=3072, E=384, top-k=6
- M={8,16,32,64,128,256,512,1024,2048,4096,8192} for both shapes

Only H200 behavior is in scope.  H20, H100, generic SM90, and the PR323
implementation must remain unchanged.  No branch or commit may be pushed.

## Selected architecture

Use a capability-driven numerical policy independent of the existing schedule
policy.

1. Classify exact H200 workload families independently of M.
2. Keep block sizes, stages, waves, SM counts, and scheduling choices in the
   schedule policy.
3. Resolve the final L1 and L2 configurations.
4. Select global packed-BF16x2 scaled accumulation from a numerical policy.
5. Verify that both resolved phase configurations support the selected
   numerical path.  Do not silently fall back to FP32 for a selected target.
6. Keep the existing environment override for controlled FP32/BF16 A/B tests.

BF16 is default-on for the validated workload families.  A point may remain
FP32 only when repeated same-configuration tests show a BF16 regression above
0.5%, or when it fails the unchanged numerical correctness gate.  Any such
exception belongs in a small, explicit numerical exception table with its
measurement reference; it must not be hidden inside schedule configuration.

The completed attribution identified four such exceptions: Pro M64 stays on
FP32 scaled accumulation after a five-observation H200 confirmation showed a
0.978% BF16 regression; Flash M32, M4096, and M8192 stay on FP32 after the full
production-policy confirmation showed 2.361%, 0.580%, and 2.151% regressions,
respectively.  All other validated Flash/Pro points passed.

After that numerical-policy refactor, the final direct PR323 gate exposed a
separate Flash M1024 schedule regression.  A bounded H200 retune retained the
generic EPW32/stage4 phases and selected only L2 N-major scheduling plus E5M2
combine.  Five paired observations measured 550.9 us versus PR323 at 592.3 us
(-6.99%).  After selector integration, a separate five-observation no-override
gate measured 564.3 us versus 587.5 us (-3.95%), and four route/weight seeds
passed the unchanged production-policy numerical gate.  This is a
schedule-retune row, not a BF16 exception or a phase-specific accumulation
mode.

## Kernel support

The current packed-BF16 loop supports non-swap M64N128/N256 consumers with
BK128.  Full matrix coverage requires two extensions:

- BK256: process every independent K scale domain, keep WGMMA raw dot products
  in FP32, convert each correctly scaled contribution to BF16x2, accumulate it
  with packed FMA, and convert once to FP32 for the unchanged epilogue.
- Low-M swap-AB: preserve the existing swap layout and scale mapping while
  storing the persistent scaled accumulator in BF16x2.  Schedule and tile
  selection must remain identical between FP32 and BF16 attribution runs.

The implementation must not combine raw WGMMA outputs across different scale
domains before applying their activation and weight scales.  Final output
format and combine representation remain controlled by their existing paths;
global BF16 scaled accumulation is an internal GEMM accumulator choice.

## Validation

For the completed implementation:

- run compile/config smoke tests for every matrix point;
- compare FP32 and BF16 with identical inputs, routing, schedule, combine mode,
  and timing boundaries;
- use 8 H200 ranks on a full-NVSwitch node;
- use rank-local median-20 and compare the maximum returned rank;
- alternate FP32/BF16 order for at least three observations;
- treat <=0.5% BF16 regression as acceptable and >0.5% as requiring repeated
  confirmation before adding an exception;
- require finite output and the unchanged calc_diff < 0.01 gate;
- use multiple route/weight seeds for the final numerical verdict.

Every implementation experiment is recorded in ITERATIONS.md.  Do not create
a commit unless the user explicitly requests one, and do not push any branch.
