# Iteration 724c — compose W2 with the terminal no-return ABI

## Hypothesis

Iteration 724b proves that CUDA 12.8 lowers `[[noreturn]]` device functions to
a real PTX `.noreturn` / SASS `EXIT` callee without a return continuation.
The prior whole-W2 outline was an ordinary returning function and remained
32 QGMMAs / 32 dependency waits.  A terminal boundary can remove caller-live
preservation and return-state constraints that remained present in every
earlier same-code-object W2 experiment.

## Change

Add default-off
`V4_SINGLE_LAUNCH_W2_TERMINAL_NORETURN_PROBE=1`.  It is restricted to the
selected TP4 M128 compact-W13 schedule with explicit bound-eight, both M128
wave rotations disabled, and all alternative W2 pipelines disabled.  After
the normal route, W13, and activation phases, each CTA calls a duplicated
ordinary whole-W2 grid-stride loop annotated `[[noreturn]]`; the callee ends
with device `exit`.

This first composition deliberately omits the phase-3 grid barrier, ordered
k6 combine, TP collective, and graph-generation cleanup.  Therefore the
candidate is compile/SASS-only and must never be launched.  If admitted, the
follow-up must move all terminal semantics into the callee before any
correctness or timing execution.

## Static hard gate

Fresh JIT only.  Require:

1. exact terminal W2 callee retains all 32 QGMMAs;
2. dependency waits fall materially below 32 toward standalone's 16;
3. PTX/SASS retain a true no-return call and terminal `EXIT`;
4. main-entry and callee resources do not cross a prohibitive occupancy,
   local-memory, or stack cliff.

Reject and restore production source before any CUDA business-kernel launch
if one of these conditions fails.
