# Iteration 648: W13 task-boundary SASS audit

This is a static cubin audit.  No CUDA kernel was launched and no new latency
claim is made.

## Artifacts

- Production one-kernel / standalone shared-source cubin:
  `/tmp/multi_one_audit_644/cuda.sm_90a.cubin`
- Rejected compact-W13 persistent-state cubin:
  `/tmp/w13_persistent_audit_648/cuda.sm_90a.cubin`

The audited functions are TP4 M128 split-K2.  `nvdisasm` function boundaries
were cross-checked against the exact local/global symbol sizes from
`readelf -WsW`.

## Instruction comparison

| Function | instructions | LDS | LDC | LDG.E.CONSTANT | STS | BRA | barrier/sync |
|---|---:|---:|---:|---:|---:|---:|---:|
| standalone W13 kernel | 1,472 | 145 | 18 | 8 | 15 | 42 | 31 |
| production compact W13 phase callee | 1,493 | 151 | 2 | 8 | 15 | 42 | 32 |
| rejected persistent compact W13 callee | 1,502 | 151 | 2 | 8 | 15 | 42 | 31 |

Production compact W13 is only 21 instructions larger than standalone.  Its
grid-stride loop target includes five shared-record load instructions
(`LDS.128`, three `LDS.64`, and another `LDS.128`) and the task tail contains
one `BAR.SYNC.DEFER_BLOCKING` before deciding whether to execute the next
task.  Relative to standalone, the complete callee has six more LDS and
sixteen fewer LDC instructions.

The prior persistent-state experiment removes exactly that one static sync
instruction, but adds nine other instructions while keeping the same LDS,
LDG, store and branch counts.  Its exact paired cold-L2 result from Iteration
640 was nevertheless 1.48% slower, losing all three batch medians.

## Interpretation

The multi-kernel gap is not explained by a large unported instruction body.
It is also not fixed by merely removing the explicit post-task CTA barrier.
The rejected persistent path changes consecutive tasks to reuse the same TMA
mbarrier objects and advances their generations across the grid-stride loop;
that state dependency and alternating metadata addressing can outweigh the
saved CTA sync even though static instruction count barely changes.

The compact shared-record ABI does cause repeated task-loop LDS operations,
but the delta is too small to plausibly provide the requested 10% by itself.
A direct-argument ABI was already rejected before compact W13 because its
larger call frame lost the M128 resource target.

## Next-design constraints

Any new W13 task-boundary design should satisfy all of the following:

1. retain one business launch and the 702x128 resident M128 grid;
2. retain the compact one-pointer phase call and 56-register/32-byte-stack
   M128 entry;
3. avoid the production path's duplicate post-task plus next-task-entry CTA
   rendezvous;
4. avoid immediate reuse of the same persistent TMA mbarrier objects;
5. retain alternating route/activation metadata so a fast warp cannot
   overwrite data still consumed by a slow warp from the preceding task;
6. remain default-off until exact multi-reference output, packed-generation
   wrap, cubin resources, and paired distributed cold-L2 timing pass.

The most direct bounded implementation is to ping-pong two small sets of TMA
mbarrier objects together with the existing two metadata slots, while retaining
the nonpersistent K-loop parity and no cross-task weight prefetch.  This differs
from the rejected persistent state: consecutive tasks do not immediately reuse
the same barrier objects.  It requires design approval before implementation.
