# Iteration 681: reject whole-loop W13 residual-wave rotation

Date: 2026-09-06

## Hypothesis and placement check

M128 W13 has 3,984 tasks over 702 resident CTAs: five complete waves and a
474-task residual wave. Continuing the selected shift-13 mapping into wave
five gives shift 65. Against the measured production CTA-to-SM placement,
shift 65 changes the residual histogram from `{5: 8, 6: 56, 7: 14}` to
`{6: 72, 7: 6}`. It therefore improves the minimum load without increasing
the optimal maximum of seven residual CTAs per SM.

The first implementation made every physical CTA iterate over every logical
wave, rotated the residual logical rank, and skipped ranks outside the 474
tasks. Arithmetic, logical task set, weight traffic, W2, barriers and
communication were unchanged. The opt-in was
`V4_SINGLE_LAUNCH_W13_RESIDUAL_ROTATE=65`.

## Static and correctness gates

- Candidate SHA-256:
  `6e147f399b2f4cd0196ad06be383f54474d73c929abe0b84f24ff29e19ad4d30`.
- JIT extension: `v4tp_794915f3aec0a3e55110_v178mspec`.
- M128 split-K2/4 remains `REG56 STACK32 SHARED2048 LOCAL0`, retaining nine
  resident CTAs/SM.
- Single-GPU random-route M128 is bitwise equal to the independent local
  reference: cosine 1, relative L2 0, finite, with 1,992 padded rows.
- Packed generation wrap returns exactly `[0,0,0,0]`.

## TP4 cold-L2 endpoint gate

Physical GPUs 0,5,6,7; random route seed 20260902; replay-interleaved CUDA
Graphs; two batches of 20 samples after four warmups. Every timed replay had
its own separate excluded 256 MiB L2 clear. Times are TP-rank-max.

| implementation | min (ms) | median (ms) | max (ms) | batch medians (ms) |
|---|---:|---:|---:|---|
| multi-kernel control | 0.303680 | 0.305328 | 0.361120 | 0.305216, 0.305488 |
| one kernel, whole-loop residual shift 65 | 0.361376 | 0.364112 | 0.400096 | 0.363840, 0.364320 |

Candidate/control is `1.192527`: the candidate is 19.25% slower. All-rank
correctness and embedded P2P two-shot allreduce checks pass, with cosine
0.99999560 and relative L2 0.00296726.

## Verdict

Reject and restore exact Iteration-665 production source
(`7ac22134c953d17c8dea9310011818ca483b5b8a96b06324381abd2c8c9c3f45`).
This implementation perturbed every complete wave by replacing the selected
grid-stride loop with wave-count arithmetic plus a per-wave validity branch.
The endpoint therefore rejects this implementation, not yet the narrow
residual-ownership hypothesis. A follow-up must preserve the complete-wave
hot path and handle only the residual in a separate tail.

Raw evidence:

- `bench/results/iter681_w13_residual_rotate65_m128_jit_correctness_20260906.log`
- `bench/results/iter681_w13_residual_rotate65_m128_resources_20260906.log`
- `bench/results/iter681_w13_residual_rotate65_tp4_m128_cold_short_20260906.log`
