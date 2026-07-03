# SM90 NVFP4 Math-Warpgroup Phase-Skew Design

## Scope

This is a Pro-only, fused-only experiment on top of the retained braided
three-stage winner. It does not change the external ModelOpt NVFP4 format, the
80-byte deployment row, routing, swap-AB policy, or any runtime argument. All
kernel, JIT, API, and Python wiring is isolated from the retained candidate.

## Motivation

The retained kernel has two N128 math warpgroups. Today both warpgroups decode
their independent B halves together and then submit WGMMA together. NCU shows
low issue eligibility and a remaining PRMT/sign short-scoreboard chain while
tensor work is not bandwidth-bound. Same-warpgroup K+1 decode is moved behind
`WARPGROUP.DEPBAR` by ptxas, and assigning a third warp to decode contends with
the math warps.

## Schedule

Add one transaction barrier per three-stage ring slot, initialized with one
arrival. On every K block:

1. Both math warpgroups wait for the normal stage-full barrier.
2. WG0 decodes only its existing rows 0--127.
3. One elected lane in each WG0 warp arrives at the stage phase-skew barrier.
4. WG1 waits for that phase, decodes only its existing rows 128--255, and then
   enters its unchanged WGMMA path.
5. Each warp retains its existing empty-barrier arrival. WG0 may advance to a
   later full stage while WG1 completes the prior stage; a stage cannot be
   reused until all eight math warps have arrived at its empty barrier.

This differs from the rejected cross-warpgroup producer experiment: neither
WG decodes the other WG's B half, no 256-thread handoff is added, and WG0 is
not required to wait for WG1 between stages.

## Invariants

- Three phase-skew barriers add 24 bytes, staying below SM90 shared capacity.
- One WG0 arrival and one WG1 wait occur per stage reuse and parity.
- Full/empty ownership, WGMMA descriptors, accumulators, and epilogues remain
  unchanged.
- Correctness must pass M=8/64 with `global_scale=none/expert` before timing.
- A fresh real-shape cubin must retain zero spill/local traffic.

## Acceptance Gate

First screen Pro M64 seed 101 against the retained three-stage kernel using
process-level A/B/B/A. Advance only if max-rank latency improves by at least
2% without a resource regression. Then test M=8/16/32/64 and routing seeds
101/202/303. Rejected wiring is removed; no experiment is committed or pushed.

## Iteration Result

The first variant signaled only after all four WG0 warps completed the full
row. Correctness and resources passed, but seed-101 A/B/B/A rejected it:
baseline and candidate max-rank geometric centers were approximately
1169.1/1187.6 us, a 1.58% regression. Phase profiling nevertheless reduced
the independently measured WG0 dequant interval from 13141 to 11343 timer
units, showing that lower simultaneous decoder pressure helps. The loss came
from full-row tail imbalance (`math_loop` increased despite slightly lower
`gemm_core`).

The only follow-up is a half-row signal. WG0 releases the barrier after quads
0--1 and continues quads 2--3 while WG1 begins its full row. This preserves
useful overlap while reducing the initial decoder issue pressure. It uses the
same barrier ownership and acceptance gate; no further delay sweep is planned
if the half-row form is not positive.

The half-row follow-up also failed. It retained exact correctness and the same
168 registers, 56-byte stack, and zero local memory, but baseline/candidate
max-rank geometric centers were approximately 1173.2/1198.9 us, a 2.19%
regression. Any imposed phase offset hurts the CTA's two-WG balance more than
lower decoder contention helps. Both phase-skew implementations and all
runtime wiring are removed; this direction is closed.
