# Iteration 711f — compose unbounded M128 with whole-W2 phase outline

## Hypothesis

The two individually tested conditions are insufficient on their own:

- a whole-W2 phase callee under the selected 56-register contract remains
  32 QGMMAs / 32 waits;
- a natural 79-register fused entry with inline W2 also remains 32/32.

Combining phase-local code generation with natural register headroom is the
remaining bounded experiment that could reproduce standalone W2's 32/16
schedule without adding a CUDA launch.

## Change

Allow the existing default-off
`V4_SINGLE_LAUNCH_W2_PHASE_NOINLINE=1` diagnostic to compose with
`V4_SINGLE_LAUNCH_M128_UNBOUNDED=1`.  No algorithm, task order, arithmetic,
grid synchronization, public ABI, TP communication, or default behavior is
changed.  The unbounded flag still rejects the legacy per-task call and
persistent W2 state.

Python bytecode compilation passes.  Staged kernel SHA256:
`a43349d9ac78e6bf14f5ba79eb406f1f316206e007a1ac862c1c3d6e283995b6`.

No JIT or CUDA launch is claimed.  Require no local spill, runtime-admissible
residency, and exact callee SASS of 32 QGMMAs / about 16 waits before running
correctness or latency.

