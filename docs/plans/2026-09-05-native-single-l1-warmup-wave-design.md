# Native TP4 single-L1-warmup-wave design

## Goal and measured problem

The selected native Hopper kernel executes route preparation, MXFP4 W13,
SwiGLU plus FP8 requantization, MXFP4 W2, ordered k6 reduction, TP all-reduce
and replay cleanup in one launch.  Its formal independently cold-L2 TP4
geometric mean is still 28.58% slower than the selected five/six-kernel
control.  NCU shows that M8 is latency/role bound: 56.35% of scheduler cycles
have no eligible warp, while DRAM throughput is only 40.17%.

The kernel uses the Hopper `InterleavedMegaMoEScheduler`, but the generic
deadlock-avoidance formula selects two complete L1-only task waves for this
shape.  At M8, two waves cover all W13 tasks, so the nominally interleaved
pipeline does not begin W2 until the W13 issue tail.  This design specializes
the warmup bound without changing task bodies or numerical boundaries.

## Alternatives

1. **One L1-only warmup wave (selected).**  Override only the TP4 scheduler's
   per-CTA warmup count after it has computed task totals.  This is the
   smallest direct test of W13/W2 overlap and retains the Hopper mailbox.
2. **Ready-first global W2 claims.**  Test readiness before claiming arbitrary
   W2 tasks.  This adds global scans/atomics and repeats the contention and
   polling failures measured in Iterations 162–163 and 413–417.
3. **Reorder all CTA roles into a 352-thread block.**  Put math warpgroups at
   warps 0–7 and support roles afterward.  This repairs the WGMMA alignment
   failure from Iteration 467, but it removes only one padding warp and does
   not itself make W13 and W2 overlap at M8.

The selected experiment is bounded and precedes the larger role rewrite.

## TP4 safety proof

For TP4, `I/rank=512`, `BLOCK_N=256`, and the scheduler has:

- four W13 tasks per BM8 pool block (`2*I/BLOCK_N = 4`);
- sixteen W2 tasks per pool block (`H/BLOCK_N = 16`);
- 156 persistent CTA workers in the selected two-CTA-per-SM launch.

One full L1 claim wave assigns task indices 0 through 155.  Because W13 has
four contiguous N tasks per pool block, those claims completely cover the
first 39 pool blocks.  The following full L2 claim wave contains at most 156
tasks, touching only the first ten pool blocks.  Every W2 task that can be
claimed in that wave therefore depends on a pool block already fully claimed
by W13.  Its activation loader may wait for actual W13 stores, but some CTA
always owns the required producer task, so there is no producer-consumer
claim cycle.

After each CTA's first W2 task, the existing scheduler forces one L1 claim
before another W2 claim.  Progress for later pool blocks is therefore
preserved.  Zero warmup is unsafe because every CTA could claim W2 before any
W13 producer exists; one wave is the minimum safe specialization.

For workloads with fewer than 39 pool blocks, the first wave claims all W13
tasks before any unready W2 dependency can be introduced.  TP8 is not enabled
for the initial specialization because its N-task ratio differs; it retains
the selected path until separately proven.

## Implementation

Add default-off `V4_NATIVE_SINGLE_L1_WARMUP_WAVE` and matching compile macro.
It is valid only for the selected TP4 `I/rank=512`, 156-CTA, interleaved,
BM8/BN256 path.  Immediately after the B-loader producer calls
`fetch_expert_recv_count()`, clamp `num_l1_warmup_waves` to one while
preserving zero.  A-loader and math consumers continue reading the same
two-stage mailbox; TMA stages, WGMMA, epilogues, communication and cleanup are
unchanged.

The flag participates in the extension hash/name and benchmark metadata.
The selected default remains off until all gates pass.

## Verification and decision gates

1. Compile/resource gate: 384 threads, 156 cooperative CTAs, 80 allocated
   registers/thread, 102.4 KiB dynamic shared memory and no static spills.
2. Deterministic local M8 and M128: complete output bitwise-equal to the
   selected native control; mutable post-workspace diagnostics are excluded.
3. TP4 CUDA Graph smoke at M8/M128: final and embedded-communication gates
   pass repeatedly, with exactly one business kernel.
4. Same-process selected-native versus candidate A/B at M8/M128, balanced
   order, at least 20 independently cold-L2 rank-max samples per arm.
5. Retain only if the endpoint geometric mean improves materially and neither
   endpoint regresses beyond noise.  A retained candidate proceeds to all
   five M values under the formal `10x200` cold-L2 protocol and TP8 fallback
   run-through.  The project is complete only at at least `1.10x` over the
   multi-kernel control.

Any deadlock, correctness failure, resource regression, or performance loss
is recorded and committed before repair or rollback.
