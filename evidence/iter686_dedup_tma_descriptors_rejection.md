# Iteration 686 — reject duplicate TMA-descriptor removal as noise-sized

## Hypothesis and change

The TP4 single-kernel entry accepted four `__grid_constant__` CUtensorMap
values, but compact-interleaved MXFP4 uses one common map for weight and scale
within each GEMM.  The launcher consequently passed the W13 descriptor twice
and the W2 descriptor twice.  The multi-kernel entry carries only the maps for
its current GEMM.

The candidate removed the two duplicate TP4 kernel parameters and aliased the
task body's existing weight-scale map reference to the corresponding unique
map.  It did not change bytes, addresses, task order, arithmetic, barriers,
CTA geometry, or communication.  TP8 was untouched.

- Candidate SHA-256:
  `c3f780c50beef4122a337f3263b3ff107590397153e43c0d9ec233d3bc7336cb`
- Selected source SHA-256:
  `7ac22134c953d17c8dea9310011818ca483b5b8a96b06324381abd2c8c9c3f45`
- Extension: `v4tp_87ee8546948a5b607d83_v178mspec`

## Gates

- Python syntax and CUDA JIT compilation passed.
- One-GPU M128 random-route full-output validation was bitwise equal to the
  same-source multi reference: cosine 1, relative L2 0, finite output, 1,992
  padded rows, and packed generations `[0,0,0,0]`.
- TP4 validation passed with identical control/candidate output metrics and
  `allreduce_ok=true` on every rank.

Resource usage for both M128 split-K2 and split-K4:

| Build | REG | STACK | SHARED | LOCAL | CONSTANT[0] |
|---|---:|---:|---:|---:|---:|
| Selected | 56 | 32 B | 2,048 B | 0 | 1,361 B |
| Deduplicated maps | 56 | 32 B | 2,048 B | 0 | 1,105 B |

The ABI change removed exactly 256 constant bytes but did not improve the
register, stack, or residency contract.

## Cold-L2 TP4 screen

- Physical GPUs 0/5/6/7, random seed 20260902, M128, W13 split-K2.
- Replay-paired CUDA Graphs, two batches x 20 samples after four warmups.
- A separate 256 MiB Triton cache clear ran before each implementation replay
  and was excluded from CUDA-event timing.

| Path | min (ms) | median (ms) | max (ms) | batch medians (ms) |
|---|---:|---:|---:|---|
| Multi control | 0.304576 | 0.307216 | 0.320352 | 0.307360, 0.307216 |
| Deduplicated single | 0.349120 | 0.351776 | 0.357888 | 0.351360, 0.352000 |

Candidate/control is `1.1450445`.  The adjacent exact-production anchor is
`1.146252`, making the normalized movement about **-0.11%**, well inside
run-to-run noise and far below the 1% retention gate.  Direct latency cannot
be used as a win because the control in this run also drifted upward.

## Decision

Reject and restore selected production byte-for-byte.  The constant-parameter
reduction is real but performance-neutral because ptxas materializes the same
phase-local state and the limiting 56-register/32-byte-stack contract is
unchanged.  Do not carry this ABI cleanup as an optimization claim.

Raw logs:

- `bench/results/iter686_dedup_tma_descriptors_m128_jit_correctness_20260906.log`
- `bench/results/iter686_dedup_tma_descriptors_m128_resources_20260906.log`
- `bench/results/iter686_dedup_tma_descriptors_tp4_m128_cold_short_20260906.log`
