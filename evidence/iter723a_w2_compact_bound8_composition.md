# Iteration 723a — compose compact per-task W2 call at bound eight

- **Gap in prior matrix:** Iteration 710f tested the compact two-argument W2
  task callee only under the selected M128 56-register/nine-CTA contract and
  found 32 QGMMAs / 32 dependency waits.  Iteration 711b tested the fused
  64-register/eight-CTA contract only with inline W2 and also found 32/32.
  No experiment combined the compact task boundary with the 64-register
  contract.  Standalone bound-eight W2 is known to emit 32/16.
- **Change:** permit existing default-off
  `V4_SINGLE_LAUNCH_W2_COMPACT_TASK_CALL=1` when M128 bound-nine is disabled
  and both launch bound and requested residency are exactly eight CTAs/SM.
  The selected compact-W13 shared argument record, task order, arithmetic,
  barriers, epilogue and embedded collective remain unchanged.  Unbounded
  M128 remains rejected.
- **Static hard gate:** fresh JIT only.  The exact compact W2 callee must
  retain 32 QGMMAs while reducing dependency waits materially toward 16;
  top-level M128 must be at most 64 registers with zero local memory.  Reject
  before CUDA business execution if either gate fails.
- **Runtime gate if admitted:** local full-output correctness and packed
  generation wrap, then same-process TP4 M128 cold-L2 A/B against the matched
  compact-W13 bound-eight inline control.  Only a stable endpoint gain can
  advance to M8/16/32/64 or the formal baseline.
