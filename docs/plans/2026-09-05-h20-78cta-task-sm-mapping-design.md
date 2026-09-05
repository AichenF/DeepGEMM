# H20 78-CTA task-to-SM mapping experiment

## Scope and invariant

This is a reversible TP4 scheduler experiment inside the approved one-launch
MXFP4 MegaMoE design.  The public ABI remains prequantized FP8-E4M3 X plus
FP32 group-128 scales, precomputed top-k metadata, and MXFP4 weights/scales.
No upstream X quantization is added.  W13, SwiGLU and intermediate requant,
W2, ordered k6 reduction, TP all-reduce, and cleanup remain in one kernel.

The production/default 624x128 path and the experimental 78x1024 path remain
unchanged unless an explicit environment flag selects this experiment.

## Evidence motivating the experiment

Iteration 395 held a synthetic 624x128 launch at eight blocks/SM and a
78x1024 launch at one block/SM.  Across three identical trials every H20 SM
received exactly eight/one blocks, but the old block sets were neither
contiguous groups of eight nor simple stride-78 groups.  The existing
78-CTA implementation assigns `cta * 8 + independent_wg`, so it cannot
reproduce the selected path's physical task distribution.

## Considered approaches

1. **Instrument the real kernels, then use an SM-indexed LUT (selected).**
   Record each real CTA's `%smid` after the phase-3 W2 barrier, compare the
   production mapping with Iteration 395, and only then construct the LUT.
   This provides direct evidence and keeps the eventual lookup tiny.
2. **Use the Iteration 395 LUT immediately.**  This is smaller work but relies
   on a synthetic kernel whose instruction/register resource signature is
   not identical to the production kernel.
3. **Search a topology-independent task hash/swizzle.**  This avoids an
   H20-specific table, but introduces a much larger tuning space without
   evidence that a generic hash will restore the selected path's TMA/HBM
   balance.

## Diagnostic instrumentation

Add a compile-time-only `V4_SINGLE_LAUNCH_TRACE_SMID` flag.  When enabled,
thread zero of every CTA records `%smid` into the beginning of `partials`
immediately after all W2 work has passed the existing phase-3 whole-grid
barrier and before communication.  At that point no later stage reads W13
partials, so this does not change output or allocate another buffer/kernel.
The compute-only profiler prints the recorded map once after synchronization.

Run M=8 on GPU0 for the selected 624x128 path and the 78x1024 path.  Inspect
register/shared-memory resource lines and require 8/1 resident CTAs per SM.
The trace is diagnostic only and is removed or compiled out before timing.

## Candidate mapping

If real-kernel traces confirm the synthetic mapping, add a separate
`V4_SINGLE_LAUNCH_78CTA_SMID_MAP` compile-time flag valid only with the
78-CTA/eight-independent-WG specialization.  Each CTA reads `%smid`; each
128-thread WG receives the corresponding old logical CTA ID from a compact
78x8 constant table.  For task wave `r`, the logical task is
`table[smid][wg] + r * 624` in W13, activation requant, and W2.

Branches are uniform within each 128-thread WG.  Out-of-range logical tasks
are skipped by the whole WG, while all WGs still reach the full-CTA phase
edge before the existing grid barrier.  No route, numerical, or collective
ordering changes are allowed.

## Verification and decision rule

1. Compute-only correctness at M=8 and M=128 against the selected path,
   including barrier generations and phase stamps.
2. Same-process TP4 control/candidate CUDA Graph screen at M=8 and M=128,
   with a separate excluded 256 MiB L2 clear before every replay.
3. Reject the mapping if either endpoint is slower by more than noise or if
   M128 replay stability worsens.  Keep it default-off unless it materially
   closes the current 37%/26% endpoint gap.
4. Only a winning endpoint screen proceeds to all-M 10x200 cold-L2 testing;
   the final objective remains at least 1.10x geometric-mean speedup over the
   multi-kernel baseline and exactly one timed business kernel node.
