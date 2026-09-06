# Iteration 679: reject M128 bound-10 with fused-W2 S2R disabled

Date: 2026-09-06

## Hypothesis and isolation

The selected M128 one-kernel entry uses 56 registers and nine resident
128-thread CTAs per H20 SM, while standalone W13 has a lower phase-local
register contract.  This experiment retained the selected compact whole-W13
outline and N128 task arithmetic, disabled register-heavy S2R lookahead only
for the fused M128 W2 call, and imposed a ten-CTA launch bound.  Standalone
multi-kernel W2 kept its selected S2R prefetch, so the paired control was not
artificially slowed.

The opt-in control was
`V4_SINGLE_LAUNCH_M128_BOUND10_W2_NO_S2R=1`.  Candidate source SHA-256 was
`f06536e532bbcbe1981844c14279648a1dc562f23c701577876e6f7169464340`;
the JIT extension was
`/tmp/torch_ext_v4_tp/v4tp_f8d95318a4d27788e88d_v178mspec`.

## Resource and correctness gates

- M128 split-K2 and split-K4 entry resources:
  `REG48 STACK64 SHARED2048 LOCAL0`.
- CUDA admitted exactly ten CTAs/SM, for a 780-CTA resident grid.
- Single-GPU random-route M128/split-K2 was bitwise equal to the independent
  multi-kernel local route tensor: cosine 1.0, relative L2 0.0, finite.
- A packed-generation wrap test ended at `[0,0,0,0]`; padded rows were 1,992.

The register target passed, but the stack frame doubled from the selected
32 bytes to 64 bytes.

## TP4 cold-L2 endpoint gate

Physical GPUs 0,5,6,7; random route seed 20260902; one CUDA graph per
implementation; replay-interleaved pairing; two batches of 20 samples after
four warmups.  Every timed replay had its own separate 256 MiB L2 clear
immediately before it, excluded from CUDA events.  Results are TP-rank-max.

| implementation | min (ms) | median (ms) | max (ms) | batch medians (ms) |
|---|---:|---:|---:|---|
| multi-kernel control | 0.303680 | 0.305312 | 0.383520 | 0.305328, 0.305312 |
| one kernel, bound-10/no-S2R | 0.365792 | 0.368512 | 0.381376 | 0.368192, 0.368640 |

The candidate/control median ratio is `1.207001`: the candidate is 20.70%
slower.  Correctness and allreduce checks passed on every rank with identical
candidate/control metrics (cosine 0.99999560, relative L2 0.00296726).

## Verdict

Reject and restore the exact Iteration-665 production source
(`7ac22134c953d17c8dea9310011818ca483b5b8a96b06324381abd2c8c9c3f45`).
Disabling fused-W2 S2R does not remove the 64-byte stack cost of a 48-register
entry, and the tighter register schedule plus lost W2 latency hiding outweigh
the extra resident CTA.  Do not expand this candidate to the other M values
or TP8.

Raw evidence:

- `bench/results/iter679_bound10_w2_nos2r_m128_jit_correctness_20260906.log`
- `bench/results/iter679_bound10_w2_nos2r_tp4_m128_cold_short_20260906.log`
