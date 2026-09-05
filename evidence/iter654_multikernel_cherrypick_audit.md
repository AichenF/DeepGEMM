# Iteration 654: multi-kernel-to-one-kernel cherry-pick audit

## Scope

Read-only source, Git-history, prior cold-L2 evidence, and byte-accounting
audit.  No CUDA source behavior changed, no GPU kernel was launched, and no
new latency result is claimed.

## Literal cherry-pick result

- Frozen multi-kernel baseline: `tpmoe_multikernel_baseline` at
  `071abc27ac106803c76ad02bb9fb5586426f6922`.
- Current one-kernel head at the audit start: `4fe5bb2`.
- `git merge-base` is exactly `071abc2`; left/right commit count is `0/513`.
- Therefore there is no multi-only commit to cherry-pick.  A Git cherry-pick
  would either be empty or replay code already in the one-kernel history.

## What is already shared

The standalone W13/W2 launches and the TP4/TP8 one-kernel entries instantiate
the same `route_gemm_task` template.  Consequently the following multi-kernel
work is already present in the one-kernel binary:

- compact interleaved MXFP4 weight/scale layout;
- bulk TMA weight fetch, evict-first policy, activation reuse policy;
- E2M1 decode/LUT path, S2R prefetch, Mode-2 braid, WGMMA issue loop;
- M-dependent W13 split-K policy and W2 split-K1 arithmetic;
- W13 FP32 partial epilogue, internal BF16/SwiGLU/group-128 FP8 requant, and
  W2 routed BF16 epilogue;
- TP4 small-M CARv2-compatible multicast one-shot push and large-M ordinary
  P2P two-shot path; and
- caller-provided FP8 X plus FP32 group-128 scales.  External X quantization
  is absent from both compared paths.

The selected one-kernel path also already tried the natural ways to make a
persistent CTA resemble a fresh standalone CTA: whole-phase outlining,
persistent LUT/mbarrier state, cross-task TMA prefetch, split-major ordering,
fresh-CTA turnover, sharded turnover, ten-CTA launch bounds, and residual-tail
rebalancing.  The last item is now statically rejected by Iteration 653b's
measured optimal SM-level tail distribution.

## What makes the multi-kernel path faster but is not a code fragment

Each standalone launch obtains hardware CTA scheduling and a phase-specific
compiler/resource contract.  Saved M128 cold-L2 NCU reports measure standalone
W13 at 47 registers, 58.22% achieved occupancy, 25.75% no-eligible cycles and
0.74 issued warps/scheduler/cycle.  The complete one-kernel entry is fixed at
56 registers, 56.23%, 36.64%, and 0.63.  Kernel exit also supplies phase
completion and destroys per-CTA state without a resident whole-grid poll.

Those properties cannot be cherry-picked into one fixed entry: CUDA assigns
one block size and one maximum per-thread resource footprint to the complete
kernel, and its resident CTAs must explicitly rendezvous before reusing the
same blocks for the next phase.  The exact M128 stage comparison attributes
`24.432 us` of positive penalty to persistent W13/W2 (`+15.472/+8.960 us`),
while fusing route plus requant saves only `4.448 us`.

## Rejected apparent transfers

- Full-K 128-thread W13 loses 3.07--11.65% locally across M8/M32/M128.
- Paired gate/up full-K W13 loses 26.03--37.82% end to end locally.
- Dual-WG split-K W13 loses 11.44% at M128 despite sharing activation.
- Hopper-style coarse CTA pipeline is about 1.93--2.16x slower at M8/M128.
- Whole-W2 and dual-phase outlines do not recover standalone throughput;
  the dual outline remains 16.07% behind multi at M128.
- Isolated W13 and W2 persistent state lose 1.48% and 0.44% respectively.
- W13/W2 next-task TMA prefetch loses 0.31% geometric and 1.28% geometric
  respectively after adjacent-control normalization.

These paths remain useful evidence, but none is a performance cherry-pick.

## Previously untested narrow idea: one WG owns two adjacent N128 tiles

History/source search found no exact experiment in which one 128-thread WG
retains the selected split-K policy and interleaves two adjacent N128
accumulators solely to reuse one activation K128 tile.  This differs from the
rejected full-K N256 and two-WG N256 implementations.

Its optimistic byte ceiling is nevertheless too low for the objective.  For
TP4 M128 W13/split-K2, one N128 task streams per 16 K128 steps:

- MXFP4 weight plus compact scales: `(8192 + 512) * 16 = 139264 B` (`136 KiB`);
- FP8 activation plus eight FP32 scales: `(1024 + 32) * 16 = 16896 B`
  (`16.5 KiB`).

Two independent tasks therefore stream `312320 B`; perfect activation reuse
would save only `16896 B`, or `5.4098%`.  The same ratio applies to W2 because
both byte classes scale with its four K128 steps.  Applying this impossible
zero-cost ceiling to the measured M128 one-kernel W13/W2 times
`203.056/107.200 us` saves at most `16.784 us`, moving complete latency only
from `0.375088` to about `0.358304 ms` (`4.47%`).  It would still be 7.95%
slower than multi's `0.331904 ms`, and 18.75% above the `0.301731 ms` latency
needed to beat multi by 10%.

The implementation would also need two live N128 accumulator sets (at least
eight extra FP32 accumulator registers/lane), dual activation staging or a
less-overlapped weight schedule.  The current complete M128 entry is already
at 56 registers for nine CTAs/SM; a 64-register outcome would fall to eight
CTAs/SM.  This resource impact is an inference, not a measured result, but it
makes the 5.41% byte ceiling still less attainable.

## Decision

Do not implement the adjacent-N pair as the next experiment: even its
unphysical perfect ceiling cannot meet the requested objective and it risks
discarding the selected ninth CTA.  There is no safe literal cherry-pick from
the frozen multi branch.  A material next step must change the one-kernel
dataflow enough to remove phase/materialization cost while retaining one-WG
cold-weight issue rate; copying existing multi-kernel bodies or launch knobs
cannot provide that result.
