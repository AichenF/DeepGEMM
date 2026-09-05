# M128 Compact-W13 Split-Major Task Enumeration

Date: 2026-09-06

Status: approved by the user on 2026-09-06 (`继续`)

## Goal

Test whether the selected TP4 M128 one-launch kernel can improve cold-L2 W13
throughput by changing only the physical enumeration of its compact outlined
W13 tasks.  The public FP8/MXFP4 ABI, route preparation, task set, split-K
factor, numerical epilogue, W2, local k6 combine, embedded TP all-reduce, and
one-business-kernel requirement remain unchanged.

The current production path launches 702 128-thread CTAs (nine CTAs per each
of 78 H20 SMs).  For M128, every CTA executes a grid-stride chain of W13 tasks.
The experiment tests whether placing all N128 tiles for one split next to each
other improves activation-slice locality and issue behavior relative to the
selected `(mblock, n_tile, split)` enumeration.

## Alternatives considered

1. **Split-major enumeration in the current compact W13 outline (selected).**
   Reinterpret only the physical ordinal as `(mblock, split, n_tile)` and map
   it back to the existing logical task index before calling the unchanged
   route-GEMM body.  This is the smallest experiment and leaves every later
   phase untouched.
2. **Batch two N128 outputs in one CTA.**  This could reuse activation staging
   more directly, but changes task granularity and synchronization.  Earlier
   N256/dual-WG experiments regressed, so it is not justified before the
   isolated ordering test.
3. **Resume fine-grained W13/W2 interleaving.**  Prior static-DAG, cohort,
   tail-pipeline, and Hopper-role adaptations consistently lost cold weight
   bandwidth or added barrier overhead.  Repeating that family is rejected.

The historical standalone W13 split-major experiment was neutral and
unstable.  This experiment is still distinct because the selected M128 kernel
uses a 702-CTA persistent grid in which each CTA executes several tasks inside
one compact device-call phase.  That historical result sets a deliberately
strict performance gate rather than supporting an expected win.

## Architecture and data flow

Add one default-off compile-time option scoped to the selected M128 compact
W13 phase.  Given `SplitK=2` and eight N128 output tiles per mblock, physical
ordinal `p` is decoded as:

```
mblock = p / (8 * SplitK)
inner  = p % (8 * SplitK)
split  = inner / 8
n_tile = inner % 8
logical_task = (mblock * 8 + n_tile) * SplitK + split
```

The unchanged `route_gemm_task` consumes `logical_task`.  The mapping is a
bijection over exactly the existing task range, so it creates no new task,
omits no task, and does not change per-task K or N coordinates.  Grid size,
grid-stride count, dynamic shared memory, mbarriers, output workspace, and all
whole-grid phase barriers remain unchanged.

The option is valid only with schedule 0, the compact W13 phase ABI, dynamic
route shared memory, M128 bound-9 admission, WOUT128, two weight stages, and
the selected valid-task invariant.  It is incompatible with persistent W13
state and every alternative scheduler/topology.  TP8 and M<=64 continue to
compile and run the production path because the option defaults off and the
candidate specialization is M128-only.

## Correctness and failure handling

Host validation rejects incompatible option combinations before JIT.  The
option participates in the extension cache key, compile definitions, and A/B
benchmark metadata so control and candidate cubins cannot alias.

Reject the candidate immediately if any of the following occurs:

- JIT failure, register allocation above the production 56 registers/thread,
  a larger caller stack, or any fixed local-memory allocation;
- missing/duplicate task symptoms, timeout, nonfinite values, or packed
  barrier generation mismatch;
- any difference from the independent full routed-W2 compute reference;
- less than 1% median latency improvement, or fewer than four winning outer
  batches out of five in the TP4 cold-L2 paired gate.

The option remains default-off unless every gate passes.  Failed and partial
runs are logged in `ITERATIONS.md` and committed before any later probe or
edit.

## Verification sequence

1. Run Python compilation and fresh SM90a JIT; inspect the exact M128 split-K2
   cubin for registers, stack, static shared memory, and local allocation.
2. Run TP-disabled M128 random-route full-output correctness after an excluded
   256 MiB L2 clear, including a packed-generation wrap check.
3. Run same-process TP4 control/candidate CUDA Graphs on GPUs 0,5,6,7 with
   five balanced AB/BA outer batches and at least 30 independently cold-L2
   samples per arm per batch.  Every replay receives a separate excluded
   256 MiB clear and timing uses the maximum rank.
4. Accept only if the candidate is at least 1% faster in the pooled median and
   wins at least four of five paired batch medians.  Otherwise reject without
   expanding to the other M values.
5. If accepted, repeat a longer M128 confirmation, then run all five TP4 M
   values and a TP8 one-launch correctness/run-through before changing a
   production default.

