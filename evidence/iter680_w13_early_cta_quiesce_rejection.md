# Iteration 680: reject W13 early-CTA 16 us quiescence

Date: 2026-09-06

## Hypothesis

At M128, selected W13 has 3,984 tasks over 702 resident CTAs: five complete
waves and a 474-task residual wave.  The other 228 CTAs enter the phase-1
packed grid barrier without a residual task.  Existing 64 ns fixed and
64-to-512 ns adaptive polling still wake their leader warps repeatedly.
This candidate made only those 228 leaders execute one 16,384 ns
`__nanosleep` before joining the barrier, attempting to remove them from
the scheduler while residual WGMMA/TMA work completed.

Task ownership, W13/W2 arithmetic, weight traffic, barrier memory ordering,
communication, and the multi-kernel control were unchanged.  The opt-in was
`V4_SINGLE_LAUNCH_W13_EARLY_CTA_QUIESCE_NS=16384`.

## Static and correctness gates

- Candidate source SHA-256:
  `64a64831519a275ddffccf4c0a2d5c4f8e6b7b6c68100b6fea0906f2954a96f1`.
- JIT extension:
  `/tmp/torch_ext_v4_tp/v4tp_91df378f0433a858188d_v178mspec`.
- M128 split-K2/4 resources remain exactly
  `REG56 STACK32 SHARED2048 LOCAL0`.
- Single-GPU random-route M128 is bitwise equal to the independent local
  reference (cosine 1, relative L2 0, finite), with 1,992 padded rows.
- Packed generation wrap completes exactly at `[0,0,0,0]`.

## TP4 cold-L2 endpoint gate

Physical GPUs 0,5,6,7; random route seed 20260902; replay-interleaved CUDA
Graphs; two batches of 20 samples after four warmups.  Each replay had a
separate excluded 256 MiB L2 clear immediately before it.  Times are
TP-rank-max.

| implementation | min (ms) | median (ms) | max (ms) | batch medians (ms) |
|---|---:|---:|---:|---|
| multi-kernel control | 0.303200 | 0.305808 | 0.336640 | 0.305824, 0.305648 |
| one kernel, 16 us quiescence | 0.366144 | 0.379664 | 0.419520 | 0.379664, 0.379680 |

Candidate/control is `1.241511`: the candidate is 24.15% slower.  All-rank
correctness and embedded P2P two-shot allreduce checks pass, with cosine
0.99999560 and relative L2 0.00296726.

## Verdict

Reject and restore the exact Iteration-665 production source
(`7ac22134c953d17c8dea9310011818ca483b5b8a96b06324381abd2c8c9c3f45`).
A CTA without a residual logical task is not guaranteed to finish its fifth
task before every residual holder finishes its sixth; delayed leaders become
late arrivals and extend the critical barrier.  Together with Iteration 374's
adaptive-poll rejection, this closes barrier-sleep tuning as a path to the
required gain.

Raw evidence:

- `bench/results/iter680_w13_quiesce16k_m128_jit_correctness_20260906.log`
- `bench/results/iter680_w13_quiesce16k_tp4_m128_cold_short_20260906.log`
