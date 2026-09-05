# Iteration 649: multi-kernel transfer recheck and barrier-design correction

This is a Git/source/SASS/history audit.  It launches no CUDA kernel and makes
no new latency claim.

## Literal cherry-pick result

At the current audited one-kernel head `ca0abcc`,
`tpmoe_multikernel_baseline` still points to `071abc2`, which is the exact
merge base and ancestor of `tpmegamoe_single_launch`.  `git rev-list
--left-right --count` reports `0 505`: there are zero multi-only commits.
There is consequently no omitted multi-kernel commit to cherry-pick.

## What is already shared

The standalone W13/W2 wrappers and the one-kernel W13/W2 phases instantiate
the same `route_gemm_task` template.  The interleaved MXFP4 layout, LUT decode,
TMA staging, S2R prefetch, WGMMA sequence, split-K policy, epilogues, and
assume-valid specialization are therefore already common code.  The one-shot
multicast push used at small M and P2P two-shot policy used at large M are also
already embedded from the CustomAllReduceV2 policy.

The exact M128 cubin comparison remains:

| path | text bytes | registers | stack bytes | shared bytes | local bytes |
|---|---:|---:|---:|---:|---:|
| standalone W13 | 23,552 | 47 | 0 | 2,048 | 0 |
| compact one-kernel W13 callee | 23,888 | caller contract | caller contract | caller contract | caller contract |
| standalone W2 | 21,504 | 61 | 0 | 2,048 | 0 |
| complete one-kernel entry | 71,680 | 56 | 32 | 2,048 | 0 |

The compact W13 callee is only 21 SASS instructions larger than standalone;
copying the standalone function body cannot account for a double-digit gap.

## Where the M128 local gap is

The separately cold-L2 phase measurements from Iterations 646 and 647 use
the same H20 GPU and route seed.  They are diagnostic rather than a paired
distributed verdict:

| phase | multi median (us) | one-kernel median (us) | one minus multi (us) |
|---|---:|---:|---:|
| route | 7.232 | 4.416 | -2.816 |
| W13 | 187.584 | 203.056 | +15.472 |
| SwiGLU/requant | 8.096 | 6.464 | -1.632 |
| W2 | 98.240 | 107.200 | +8.960 |

The fused path already wins route plus requant by 4.448 us.  Its persistent
GEMMs lose 24.432 us, with W13 accounting for about 63% of the positive GEMM
penalty.  Thus copying route, requant, combine, or collective snippets is the
wrong target.

## Previously tested attempts to reproduce launch properties

The remaining multi-kernel advantages are launch properties: one task per
fresh CTA, phase-specific register contracts, and kernel-exit publication.
They have not been left completely unexplored:

- Oversubscribed one-task-per-CTA turnover was 25.30% slower at M8 and 27.20%
  slower at M128 (Iteration 271).  A 78-way sharded variant was still
  26.98%/28.24% slower (Iteration 273).
- A ten-resident-CTA monolithic bound was 14.11% slower at M8 and 19.62%
  slower at M128 than the contemporaneous multi control (Iteration 310).
- Whole-W2 phase outlining changed M128 by only 0.12% and produced replay
  relative-L2 excursions of 0.00653--0.00761 (Iteration 378).
- Cross-task W13 TMA prefetch regressed the normalized endpoint geometric
  mean by 0.31% (Iterations 313--314).
- The selected compact-W13 ABI is the one successful adaptation, but its
  long all-M normalized gain is only 0.3575% (Iteration 605).

These results reject fresh-CTA emulation, launch-bound forcing, and ordinary
phase outlining as routes to the requested 10% win.

## Correction to the Iteration-648 next-design suggestion

The Hopper `megamoe_nvfp4_dev_m` implementation initializes a full/empty
mbarrier ring once per CTA and advances stage parity.  Its 384-thread CTA has
64 dispatch threads, two 32-thread TMA loader roles, and two 128-thread math
warpgroups, with a two-stage task mailbox and dynamic register
redistribution.  It does not alternate whole mbarrier banks merely to avoid
reusing a barrier generation.

The Iteration-648 double-mbarrier-bank suggestion is therefore superseded
before implementation.  The existing compact persistent-state experiment
already removed the post-task sync, advanced parity, remained bitwise
correct, and was 1.48% slower in all three cold-L2 batches (Iteration 640).
Adding a second whole barrier bank would add state/addressing without the
producer/consumer role specialization that makes the Hopper ring useful.

## Decision

No production-worthy direct cherry-pick remains.  The only small bounded
source adaptation not already timed is selective register-shadowing of the
compact W13 shared argument record, to imitate standalone constant-parameter
access while preserving the selected 56-register/32-byte-stack entry.  Its
static ceiling is small because only five shared-record loads occur per task;
it must be treated as a default-off micro-experiment, not as the 10% plan.

At the Iteration-612 five-M result, multi and one-kernel geometric means are
0.165906740 and 0.183687688 ms, respectively.  Reaching a true 1.10x win over
multi requires the current one-kernel geometric mean to fall by 17.89%.
That scale of gain requires a new overlapping execution architecture, not a
literal cherry-pick from the multi-kernel branch.
