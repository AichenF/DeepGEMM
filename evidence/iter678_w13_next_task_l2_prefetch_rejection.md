# Iteration 678: reject lightweight W13 next-task L2 lookahead

Date: 2026-09-06

## Hypothesis

The Hopper MegaMoE producer keeps future task metadata in a CTA-local mailbox
and starts data movement ahead of the math consumers.  The prior single-kernel
cross-task TMA experiment carried shared stages and mbarrier parity across
tasks and lost.  This narrower adaptation retained the selected M128 compact
W13 task body and issued exactly one
`cp.async.bulk.prefetch.L2.global` for the next persistent task's first compact
weight+scale record while the current task had four K128 iterations left.

The hinted record is 8,704 bytes.  The authoritative next task still
initializes its ordinary barriers and performs its normal TMA copy.  No task,
math, output, whole-grid barrier, requantization, W2, combine, or allreduce
operation changed.  The residual W13 wave used the same selected mapping.

Candidate source SHA-256:
`cc57cb303cd3bb539ffb9369c537f7840f4aac173a137b689dd1cf558855f042`.

## Correctness and resource gates

Single-GPU random-route M128 passed against the independent same-source local
reference:

- cosine 1.0, relative L2 0.0, finite;
- 1,992 padded rows and W13 split-K2;
- packed barrier words `[2048, 2048, 2048, 2048]`.

Both M128 split-K2 and split-K4 entries retained the selected resource shape:

- `REG56 STACK32 SHARED2048 LOCAL0`;
- nine resident 128-thread CTAs per H20 SM.

Both TP4 timing arms also passed all-rank reference and embedded-allreduce
checks with cosine 0.9999956 and relative L2 0.0029673.

## TP4 cold-L2 screen

Physical GPUs 0,5,6,7; M128 random route seed 20260902; CUDA Graph; two
batches of 20 replay-interleaved samples per implementation and four warmups.
Every replay received a separate excluded 256 MiB L2 clear immediately before
the implementation.  Candidate ratios use the same-process, same-source
multi-kernel control.

| mode | multi median (ms) | one median (ms) | one / multi |
|---|---:|---:|---:|
| next-record L2 lookahead ON | 0.304976 | 0.353808 | 1.160118 |
| exact flag-OFF source | 0.304752 | 0.349728 | 1.147582 |

ON/OFF after denominator normalization is 1.010923: the hint makes the
one-kernel path **1.09% slower**.  Both ON batch medians are stable at
0.353872/0.353696 ms; OFF is 0.349728/0.349776 ms.

## Verdict

Reject without a long run and restore production source/benchmark hashes
`7ac22134...` and `e39ab6a4...`.  Even a single future compulsory record adds
enough L2/TMA contention to outweigh task-boundary lookahead.  Together with
the earlier full shared-stage and bulk-L2-prefetch failures, this closes
software prefetch as a transfer of Hopper's producer/mailbox advantage.  A
future structural path must remove work or decouple producer issue, not add
duplicate requests to the existing cold stream.

Raw evidence:

- `bench/results/iter678_w13_next_task_l2_prefetch_m128_jit_correctness_20260906.log`
- `bench/results/iter678_w13_next_task_l2_prefetch_on_a_tp4_m128_cold_20260906.log`
- `bench/results/iter678_w13_next_task_l2_prefetch_off_a_tp4_m128_cold_20260906.log`
