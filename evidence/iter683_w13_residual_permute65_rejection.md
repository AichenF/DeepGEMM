# Iteration 683: reject owner-preserving W13 residual task permutation

Date: 2026-09-06

## Hypothesis and isolation

This candidate leaves the production physical owner set unchanged: the same
474 CTAs execute the M128 W13 residual wave and each SM retains the same
5--7 residual task count. Only the logical residual indices are rotated by
65 modulo the dynamic residual count. The five complete waves, task count,
arithmetic, traffic, W2 and communication remain unchanged. The opt-in was
`V4_SINGLE_LAUNCH_W13_RESIDUAL_PERMUTE=65`.

## Static and correctness gates

- Candidate SHA-256:
  `85d505b7c0462b9138f57b81d97849d804870afa94dfda2bb8df5e5327819091`.
- JIT extension: `v4tp_2cc4594d271735395df9_v178mspec`.
- M128 split-K2/4 remains `REG56 STACK32 SHARED2048 LOCAL0`, preserving nine
  resident CTAs/SM.
- Single-GPU random-route M128 is bitwise equal to the independent local
  reference: cosine 1, relative L2 0, finite, with 1,992 padded rows.
- Packed generation wrap returns exactly `[0,0,0,0]`.

## TP4 cold-L2 endpoint gate

Physical GPUs 0,5,6,7; random route seed 20260902; replay-interleaved CUDA
Graphs; two batches of 20 samples after four warmups; separate excluded
256 MiB L2 clear before every replay. Times are TP-rank-max.

| implementation/window | min (ms) | median (ms) | max (ms) | one/multi |
|---|---:|---:|---:|---:|
| exact production multi | 0.302816 | 0.305008 | 0.329184 | - |
| exact production one | 0.347008 | 0.349616 | 0.393728 | 1.146252 |
| permutation-run multi | 0.302208 | 0.305088 | 0.348640 | - |
| one, residual permutation 65 | 0.349984 | 0.352384 | 0.374432 | 1.155024 |

The normalized ratio regresses by 0.765%; direct one-kernel latency regresses
by 0.792%. Candidate batches are 0.352320 and 0.352496 ms. All-rank
correctness and embedded P2P two-shot allreduce checks pass, with cosine
0.99999560 and relative L2 0.00296726.

## Verdict

Reject and restore exact Iteration-665 production source
(`7ac22134c953d17c8dea9310011818ca483b5b8a96b06324381abd2c8c9c3f45`).
Together with Iterations 681 and 682, this closes W13 residual ownership and
logical-task remapping: the selected complete-wave-only rotation remains the
best measured policy.

Raw evidence:

- `bench/results/iter682b_production_tp4_m128_cold_anchor_20260906.log`
- `bench/results/iter683_w13_residual_permute65_m128_jit_correctness_20260906.log`
- `bench/results/iter683_w13_residual_permute65_m128_resources_20260906.log`
- `bench/results/iter683_w13_residual_permute65_tp4_m128_cold_short_20260906.log`
