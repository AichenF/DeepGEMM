# Iteration 682: reject low-overhead W13 base rotation 65

Date: 2026-09-06

## Hypothesis and isolation

The measured M128 grid has 702 resident CTAs. W13 has five complete waves and
a 474-task residual. Shift 65 reduces the residual per-SM owner histogram
from `{5: 8, 6: 56, 7: 14}` to `{6: 72, 7: 6}` without increasing its
maximum.

To avoid Iteration 681's per-wave scheduler rewrite, this candidate relabeled
the physical CTA once at W13 phase entry as `(blockIdx.x + 65) mod 702` and
then ran the selected grid-stride loop unchanged. The relative shift of 13
between complete waves, task set, arithmetic, traffic, W2 and communication
remain unchanged. The opt-in was `V4_SINGLE_LAUNCH_W13_BASE_ROTATE=65`.

## Static and correctness gates

- Candidate SHA-256:
  `f2066b6a83d5fd7b9364eb661145603f2839c84c405d4b0630dbf9c4b0d8c9e7`.
- JIT extension: `v4tp_07d20010183ce04a60b6_v178mspec`.
- M128 split-K2/4 remains `REG56 STACK32 SHARED2048 LOCAL0`, preserving nine
  resident CTAs/SM.
- Single-GPU random-route M128 is bitwise equal to the independent local
  reference: cosine 1, relative L2 0, finite, with 1,992 padded rows.
- Packed generation wrap returns exactly `[0,0,0,0]`.

## TP4 cold-L2 endpoint gate

Physical GPUs 0,5,6,7; random route seed 20260902; replay-interleaved CUDA
Graphs; two batches of 20 samples after four warmups. Every replay had a
separate excluded 256 MiB L2 clear. Times are TP-rank-max.

| implementation | min (ms) | median (ms) | max (ms) | batch medians (ms) |
|---|---:|---:|---:|---|
| multi-kernel control | 0.303264 | 0.305040 | 0.337888 | 0.304992, 0.305168 |
| one kernel, W13 base shift 65 | 0.363328 | 0.365248 | 0.388544 | 0.365088, 0.365664 |

Candidate/control is `1.197377`: the candidate is 19.74% slower. All-rank
correctness and embedded P2P two-shot allreduce checks pass, with cosine
0.99999560 and relative L2 0.00296726.

## Verdict

Reject and restore exact Iteration-665 production source
(`7ac22134c953d17c8dea9310011818ca483b5b8a96b06324381abd2c8c9c3f45`).
Although the implementation preserves the hot loop and tightens the residual
count distribution, a base relabel necessarily changes every complete
wave's absolute SM-to-weight assignment. The stable regression shows that
this costs more than the balanced residual can recover.

Raw evidence:

- `bench/results/iter682_w13_base_rotate65_m128_jit_correctness_20260906.log`
- `bench/results/iter682_w13_base_rotate65_m128_resources_20260906.log`
- `bench/results/iter682_w13_base_rotate65_tp4_m128_cold_short_20260906.log`
