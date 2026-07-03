# SM90 NVFP4 Next-Quad LUT Pipeline Experiment

## Goal

Reduce shared-LUT latency in the retained Pro shared/shared decoder without
changing standard NVFP4 semantics, the 80-byte deployment row, or the fused
kernel architecture.

## Alternatives

1. Keep the current schedule: load both LUT entries for one K32 quad, decode
   its four packed words, then move to the next quad.
2. Keep two LUT pairs live: while decoding the current quad, load the next
   quad's pair and rotate it into the current pair. This adds only four
   32-bit live values and may overlap shared-memory latency with DP4A/PRMT.
3. Preload all eight LUT entries. Earlier experiments increased register
   pressure and lost performance, so this remains rejected.

Option 2 is selected because it is the smallest untested latency-hiding
window and does not alter the weight layout.

## Screening Method

Add a separate next-quad variant to the isolated row-decoder benchmark. The
control and candidate use identical packed rows, paired DP4A selectors, shared
LUT storage, output swizzle, block count, and alternating run order. Validate
bit-exact output for uniform, model-like, and exhaustive scale distributions.

Inspect the compiled SASS to ensure LUT loads form a rolling current/next
window rather than an all-entry preload. Record register count, stack frame,
local-memory traffic, and median net decoder cycles.

## Integration Gate

Do not modify the Pro kernel unless the isolated candidate is bit-exact, has
no stack or local-memory traffic, and shows a repeatable reduction large enough
to survive the current 168-register launch bound. If it passes, integrate the
schedule only in the independent Pro candidate and run M=8/64 correctness plus
multi-seed ABBA against the retained N24 policy.

No commit or push is allowed during the experiment.
