# Iteration 676: reject the 78-CTA 5:3 W13/W2 role pipeline

Date: 2026-09-06

## Transfer hypothesis

The multi-kernel control gives W13 and W2 fresh CTA scheduling and distinct
phase-specific resource contracts.  This experiment tried to transfer that
property into one business kernel without adding a global work queue:

- launch exactly 78 CTAs (one per H20 SM), each with eight 128-thread warp groups;
- statically assign five warp groups to W13 plus internal activation requantization;
- statically assign three warp groups to W2;
- hand off mblocks through a two-slot CTA-shared mailbox;
- preserve the selected MXFP4 decode, WGMMA tile bodies, global partial layout,
  local top-k=6 combine, and embedded two-shot allreduce;
- retain one CUDA-graph business-kernel node per rank.

Candidate source SHA-256:
`4c5f224076dc2cd6d140aae455938ea143ac7d858c61c7e34d93dc52cfaa6137`.

## Gates

Single-GPU M128, random seed 20260902:

- cosine: 1.0;
- relative L2: 0.0;
- all finite;
- packed generation words: `[2048, 0, 0, 2048]`;
- padded rows: 1992;
- W13 split-K: 2.

Resource report for both M128 split-K=2 and split-K=4 instantiations:

- registers: 64/thread;
- stack: 48 bytes;
- static shared memory: 3072 bytes;
- local memory: 0 bytes;
- enforced occupancy: one 1024-thread CTA/SM.

The TP4 graph-level correctness gate also passed for both candidate and control:
cosine 0.9999956, relative L2 0.0029673, finite on every rank, and embedded
allreduce agreement passed.

## Cold-L2 result

Command shape: TP4 on GPUs 0,5,6,7, M=128, random route, two outers, 20
replays per implementation per outer.  Every replay used a separate 256 MiB L2
clear immediately before the implementation; the clear itself was excluded from
CUDA-event timing.

| implementation | min (ms) | median (ms) | max (ms) |
|---|---:|---:|---:|
| same-source multi-kernel control | 0.302176 | 0.305840 | 0.320064 |
| one-kernel 78-CTA 5:3 role pipeline | 0.471680 | 0.476320 | 0.485024 |

Candidate/control = 1.557416, so the candidate is **55.74% slower**.  Control
speedup over candidate is 0.6421x.

Raw logs:

- `bench/results/iter676b_78cta_role_pipe_m128_jit_correctness_20260906.log`
- `bench/results/iter676_78cta_role_pipe_tp4_m128_cold_short_20260906.log`

## Verdict

Reject and restore the selected Iteration-665 source and benchmark hashes.
The experiment shows that a fixed 5:3 in-CTA role split is not an equivalent
replacement for the multi-kernel launch boundary.  It permanently strands one
role cohort as phase demand and tail-wave balance change, while keeping a
64-register monolithic contract.  Any further transfer should preserve dynamic
CTA reassignment or reduce the phase contract without pinning most of an SM to a
single phase.
