# CTA-Local Split-K W13 Fusion Design

Date: 2026-09-05

Status: approved by the user on 2026-09-05 (`ok`)

## Goal

Build one TP4 CUDA business kernel for the complete routed-expert layer:
device route preparation, MXFP4 W13, SwiGLU plus FP8-E4M3 group-128
requantization, MXFP4 W2, ordered weighted k=6 local combine, TP all-reduce,
and replay-state cleanup.  The formal target remains at least 1.10x faster
than the selected five/six-kernel CUDA-Graph baseline over the equal-weight
geometric mean of M=8,16,32,64,128 under independently cold L2.

This experiment keeps the required 78 persistent CTAs on H20 and attacks the
measured compute deficit without interleaving the cold W13 and W2 weight
streams.

## Evidence and rejected alternatives

The selected 624x128 phase kernel is the fastest one-launch implementation so
far, but remains about 5.6%, 7.7%, 12.8%, 13.4%, and 14.6% slower than the
multi-kernel baseline at the five target M values.  The mapped 78x1024 path
proved that eight WGMMA warpgroups can occupy every H20 SM and that physical
task placement matters, but its global W13 partial materialization and
separate requant phase still leave it behind.

Three broader approaches were considered:

1. **CTA-local split-K W13 fusion (selected).**  Co-locate the gate/up split
   tasks for an activation N128 group in one 1024-thread CTA, reduce their
   partials through shared memory, and immediately run SwiGLU/requant.  Keep
   W2 phase-separated.  This preserves eight resident math warpgroups per SM,
   removes the global W13 partial round trip and one whole-grid phase, and
   avoids the cold-weight interference observed in iterations 284, 420, 432,
   and 492.
2. **624x128 adaptive split-K8/N64 tails.**  Lower implementation risk, but
   it only attacks underfilled final waves.  The order-sensitive split-K4
   tail result in iterations 323-324 suggests a noise-sized ceiling and it
   does not satisfy the approved 78-persistent-CTA design.
3. **Add split-K/fixup to the 156x384 Hopper-native role pipeline.**  This is
   architecturally faithful but high risk.  The current native implementation
   is 28.6% slower geometrically, and adding fixup would duplicate machinery
   already available in the faster flat implementation.

## Architecture

Launch exactly 78 CTAs of 1,024 threads.  Each CTA contains eight independent
128-thread WGMMA warpgroups.  Named barrier slots 1 through 8 remain private
to the existing per-WG GEMM pipeline.  Slots 9 and 10 synchronize only the
W13 cohorts described below.

For TP4, the intermediate width is 512 and W13 produces eight N128 tiles:
four gate groups and four matching up groups.

- M=8/16/32 use split-K4.  All eight WGs form one cohort: WGs 0-3 compute the
  four gate splits and WGs 4-7 compute the four matching up splits for one
  activation group.
- M=64/128 use split-K2.  The CTA forms two independent four-WG cohorts.
  Within each cohort, two WGs compute gate split 0/1 and two compute up split
  0/1.  Both cohorts can advance independently.

The cohort barrier uses the non-aligned named CTA-barrier instruction with a
fixed participant count (1,024 for split-K4 or 512 for split-K2).  It is not a
full-CTA `__syncthreads()` in the split-K2 specialization, and the two cohorts
use distinct barrier slots.

## Shared-memory lifetime

The existing selected two-stage WGMMA task reserves 18,432 dynamic shared
bytes per WG.  A completed W13 task needs only 4,096 bytes for its 8x128 FP32
partial.  After all WGMMA/TMA operations for that task have retired, its
weight/activation slab is dead until the next task, so the partial overwrites
the start of the same slab.

The existing `SharedPartial` route-GEMM epilogue is extended to the
eight-independent-WG specialization.  Every WG writes a disjoint 8x128 tile
to its own slab.  No new global workspace, dynamic shared allocation, or DSM
cluster is introduced.

The first cohort barrier publishes all partial stores.  Cohort WG 0 then
reads the gate/up slabs in ascending split order, rounds gate and up to BF16,
computes BF16 SwiGLU, calculates the per-row N128 maximum, publishes the W2
scale with the existing global-scale convention, and writes FP8-E4M3 output.
The second cohort barrier prevents any producer WG from reusing its slab
before the leader finishes reading it.

## Task mapping

A macro task is `(mblock, activation_group)`, not an individual split tile.
For split-K4, one CTA executes one macro at a time.  For split-K2, its two
cohorts execute two adjacent macro tasks at a time.  Macro rounds are statically
striped over 78 CTAs; no global task claim or per-task readiness atomic is
used.

The 78-block launch's measured `%smid` inverse map determines the logical CTA
stripe so repeated runs preserve the validated H20 physical placement.  An
incomplete final split-K2 pair lets the second cohort skip the GEMM uniformly
and still participate in its own cohort barriers.  No WG exits a barrier
epoch early.

After every macro is complete, the existing packed whole-grid barrier
publishes the complete quantized W2 input.  W2 then uses the already-validated
real-SMID eight-WG mapping and remains a pure cold-W2 phase.  The existing
terminal packed barrier and TP4 multicast/P2P collective paths remain
unchanged.

## Numerical contract

The implementation must preserve the public boundaries already used by the
multi-kernel baseline:

- caller-provided FP8-E4M3 X and FP32 group-128 scale;
- MXFP4 dequantization inside both GEMMs;
- fixed-order FP32 split reduction followed by BF16 gate/up rounding;
- BF16 SwiGLU result before FP8-E4M3 group-128 requantization;
- BF16 W2 route output;
- ordered k=6 FP32 weighted local combine using the existing routed scale;
- exactly one final TP all-reduce.

The route IDs and weights remain graph inputs and may change between replays.
Router/top-k generation, input quantization, allocation, weight preprocessing,
JIT, and graph capture remain outside timing.

## Isolation and failure handling

The candidate is guarded by a new default-off compile-time environment flag
and has its own JIT cache-key component.  It is valid only for TP4,
schedule-0, WOUT128, compact interleaved scale, 78x1024, real-SMID-map builds.
All existing selected paths remain byte-for-byte available as controls.

Reject the candidate immediately on any of these conditions:

- CUDA JIT failure;
- more than 64 allocated registers per thread, declared local memory, material
  dynamic spills, or fewer than one 1,024-thread CTA resident per H20 SM;
- timeout, named-barrier/synccheck error, nonfinite output, or replay-generation
  mismatch;
- loss of full-route correctness at M8 or M128;
- less than 3% endpoint improvement in a same-process cold-L2 screen.

Every failed or partial test is recorded in `ITERATIONS.md` and committed
before the next probe or edit.

## Verification sequence

1. Compile Python and CUDA; inspect M8 split-K4 and M128 split-K2 cubin
   resources and occupancy.
2. Run TP-disabled all-route correctness at M8 and M128, including packed
   generation wrap and full output comparison against the independent local
   multi-kernel pipeline.
3. Run TP4 same-process, replay-interleaved CUDA-Graph A/B at M8 and M128;
   every timed replay has its own excluded 256 MiB L2 clear.
4. If both endpoints improve by at least 3%, run a longer endpoint
   confirmation and NCU on the committed binary.
5. Run the formal TP4 M=8,16,32,64,128 comparison with 10 outer batches x 200
   cold samples per arm, rank-max statistics, min/median/max, per-batch
   medians, correctness, and equal-weight geometric mean.
6. Use Nsight Systems to prove one timed business-kernel node per replay after
   excluding the separate L2 clear.  No hidden memset, child launch, reset, or
   copy is allowed.
7. Run the TP8 one-launch correctness/run-through specialization.  TP8 is not
   used to tune the primary TP4 score.

The goal is complete only if the final committed one-kernel path is at least
1.10x faster than the selected multi-kernel baseline by the formal TP4
geometric-mean metric and every required correctness/single-launch/TP8 gate
passes.
