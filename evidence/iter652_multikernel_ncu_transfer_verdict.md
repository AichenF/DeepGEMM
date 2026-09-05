# Iteration 652: multi-kernel NCU transfer verdict

This is a read-only Git/source/history and saved-NCU-report audit.  It launches
no CUDA kernel, changes no CUDA behavior, and makes no new latency claim.

## Literal Git result

At current one-kernel head `83eb9b6`, `tpmoe_multikernel_baseline` remains at
`071abc2`, the exact merge base and an ancestor of the one-kernel branch.
`git rev-list --left-right --count` now reports `0 510`: there are still zero
multi-only commits to cherry-pick.

The standalone W13/W2 wrappers and the one-kernel phases already instantiate
the same `route_gemm_task`.  Consequently the selected MXFP4 layout/decode,
TMA transfers, S2R path, WGMMA sequence, split-K math, epilogues, valid-task
specialization, and the CARv2-derived small-M multicast-push / large-M P2P
two-shot policy are common code rather than omitted patches.

## Saved-report comparison

Read-only imports used the already collected, application-cold-L2 reports:

- `results/iter650b_production_m128_cold_sourcecounters.ncu-rep`
- `results/iter651b_standalone_w13_m128_cold_sourcecounters.ncu-rep`

The profiler did not launch new kernels during this audit.  NCU replay timing
is diagnostic; the formal separately cold stage medians remain Iterations 646
and 647.

| metric | complete one-kernel entry | standalone W13 |
|---|---:|---:|
| NCU replay duration (us) | 344.83 | 187.23 |
| registers/thread | 56 | 47 |
| theoretical occupancy | 56.25% | 62.50% |
| achieved occupancy | 56.23% | 58.22% |
| no eligible scheduler cycles | 36.64% | 25.75% |
| eligible warps/scheduler | 1.82 | 2.61 |
| issued warps/scheduler/cycle | 0.63 | 0.74 |
| DRAM throughput | 50.52% | 61.49% |
| L2 throughput | 62.33% | 74.72% |
| all barrier PC samples | 9,510 | 3,249 |
| not-issued barrier PC samples | 4,571 | 684 |

The scopes differ, so the table does not treat the complete one-kernel replay
duration as a standalone-W13 latency comparison.  It does show the resource
contract and scheduler issue advantage which kernel boundaries give the
standalone phase.

The exact production SASS/source counters localize the decisive difference:

| production entry location | all samples | not-issued samples | attribution |
|---|---:|---:|---|
| offset `0x2a60` | 3,422 | 2,324 | branch after W13 phase grid barrier; every sample is barrier-classified |
| offset `0x8950` | 1,943 | 943 | exit after terminal W2 grid barrier |
| five compact-W13 shared-record LDS sites combined | 19 | 2 | thirteen uniform pointer fields |
| compact-W13 post-task `BAR.SYNC`, offset `0x11090` | 0 | 0 | no measured hotspot |

The standalone W13 has no analogous software phase barrier; its largest
individual not-issued locations are ordinary WGMMA/control/dataflow sites,
with the largest at 136 samples.  The compact W13 callee's argument loads and
post-task synchronization therefore cannot plausibly explain the measured
15.472-us normal-timing W13 deficit.

## Transfer verdict

The useful multi-kernel properties are launch properties, not source snippets:

1. one task per newly scheduled CTA, with hardware tail load balancing;
2. a 47-register stack-free W13 contract and a separate 61-register
   stack-free W2 contract;
3. kernel-exit global publication/reset instead of resident-grid software
   barriers.

The first property has been emulated too broadly before: all-task
oversubscribed turnover and its 78-way sharded form lost 25--28%.  Static
per-SM remapping also lost 13--18% because its one-time atomics changed the
weight-task mix for every wave.  Fine-grained DAG/readiness schedulers,
coarse Hopper-style CTA mailboxes, W13-to-activation tail overlap, balanced
worker counts, N64 residual subdivision, phase outlining, and persistent
task state have likewise been rejected.  None should be repeated as a
nominal cherry-pick.

This report also rejects Iteration 649's last micro-candidate: caching the
compact W13 argument record.  Five LDS instructions contribute only 19 total
samples, while a direct high-argument ABI previously increased the complete
entry from 56 to 64 registers and lost roughly 1.2% in the M128 W13 phase.

## Narrow untested adaptation

The one multi-kernel scheduling property not found in the prior experiments
is **residual-wave-only work stealing**:

- retain the production compact W13 body, launch geometry, full static waves,
  task order, weights, and all existing barriers;
- after every CTA completes exactly `floor(tasks / ctas)` ordinary W13 tasks,
  have lane 0 perform one relaxed ticket allocation from a single cursor;
- only tickets below `tasks % ctas` execute one final W13 task; broadcast the
  ticket through one CTA-shared word;
- initialize the cursor before the already-required route-to-W13 grid
  publication, and keep the existing W13 phase barrier as the completion
  fence.

For the qualified M128 case this preserves all five complete static waves and
uses 702 ticket operations to distribute only the 474-task sixth wave to the
first CTAs that become free.  It is materially different from Iterations
271/273 (dynamic scheduling of every task in an oversubscribed monolithic
grid) and 517--522 (one-time SM-slot remapping which changed every wave).

Expected value is bounded: it can recover only persistent final-wave
imbalance, not the fixed 56-register monolithic resource contract.  It should
therefore be implemented first as an M128-only, default-off W13 experiment,
with resource and bitwise gates before a paired TP4 cold-L2 screen.  Even a
positive result cannot alone satisfy the required 17.89% geometric-mean
reduction; it is a diagnostic transfer of the only remaining hardware-launch
advantage, not a claimed 10% solution.

## Decision

There is no literal or body-level cherry-pick from the multi-kernel branch.
Do not modify production defaults.  If explicitly approved, implement only
the bounded residual-wave W13 ticket experiment described above; extend it to
W2/M64 only after an independently cold-L2 M128 win.
