# Iteration 717a: compose compact whole-W2 ABI with RDC

## Missing composition

Iteration 662 tested a one-pointer whole-W2 phase call in the ordinary
single translation unit.  It retained REG56/STACK32 but ptxas still emitted
one dependency wait per QGMMA.  Iterations 713–714 separately proved that RDC
recovers standalone W2's 32-QGMMA/16-DEPBAR issue pattern, but only through
the legacy thirteen-pointer-plus-four-scalar phase ABI; that callee forced the
entry to REG195/STACK112.  These two properties have not been composed.

## Controlled change

`bench/iter717_make_unique_rdc_compact_w2.py` starts from the exact uniquely
named Iteration-714b RDC source/build.  It makes only the following generated-
source changes:

- assigns fresh Iteration-717a host/kernel symbols;
- adds an M128-only whole-W2 RDC callee receiving one pointer to the existing
  CTA-shared `SingleLaunchW13PhaseArgs` record;
- republishes the same W2 pointers into that record after the activation
  barrier; and
- derives routes, task count, CTA and grid coordinates inside the callee.

The W2 task order, `route_gemm_task` template, FP32 accumulation, BF16 route
output, barriers, terminal TP collective, and all lower-M call paths remain
unchanged.  Production source is not edited.

## Pre-CUDA gates

Commit this composition before building.  Reject before launch unless:

- RDC compile and device link succeed;
- TP4 M128 split-K2 uses at most 64 registers, at most 48 stack bytes/thread,
  and no fixed local allocation;
- the exact compact W2 callee contains 32 QGMMAs and approximately 16
  dependency barriers.

Only if all static gates pass should the candidate receive TP4 M128 local
bitwise correctness and a short paired cold-L2 runtime screen.
