# Iteration 672: reject persistent W13 TMA-issuer rotation

## Question

The standalone W13 baseline starts a new CTA for every route-GEMM task, while
the single-launch compact W13 phase keeps one CTA alive for roughly six tasks.
Test whether rotating the weight-TMA issuer role across the four warps on each
persistent task recovers part of the standalone scheduler/codegen advantage.

The experimental sequence was warp 1, 2, 3, 0, repeating.  Sequence zero kept
warp 1 exactly like the selected distributed-prep implementation.  The switch
was instantiated only by the compact M128 single-launch W13 call; standalone
W13, W2, task order, grid size, barriers, data layout, and communication were
unchanged.

## Gates

- Local M128 full-output reference: cosine `1.0`, relative L2 `0.0`, finite.
- Packed generation wrap: `[0, 0, 0, 0]`, pass.
- TP4 final output and allreduce: pass in every run.
- M128 single-launch cubin resources: `REG 56`, `STACK 32`, `LOCAL 0`,
  `SHARED 2048`, identical to the selected implementation.

Correctness log:
`bench/results/iter672b_w13_tma_issuer_rotate_m128_correctness_20260906.log`.

## Cold-L2 TP4 result

All runs used random routing at M128, CUDA Graph replay, two outer batches,
20 samples per implementation per outer batch, and a separate 256 MiB L2 clear
immediately before every implementation replay.  The clear was outside the
timed events.  Independent process order was OFF / ON / ON / OFF.

| Run | Multi (ms) | Single (ms) | Single / multi |
| --- | ---: | ---: | ---: |
| OFF A | 0.305200 | 0.349904 | 1.146474 |
| ON A | 0.304752 | 0.349184 | 1.145797 |
| ON B | 0.305120 | 0.349392 | 1.145097 |
| OFF B | 0.304848 | 0.349648 | 1.146958 |

- OFF mean ratio: `1.146716470`.
- ON mean ratio: `1.145447105`.
- Normalized ON/OFF ratio: `0.998893044`, only `0.111%` faster.
- Direct single-launch mean: `0.349776 ms` OFF versus `0.349288 ms` ON,
  only `0.140%` faster.

Raw logs:

- `bench/results/iter672_w13_tma_issuer_off_a_tp4_m128_cold_20260906.log`
- `bench/results/iter672_w13_tma_issuer_on_a_tp4_m128_cold_20260906.log`
- `bench/results/iter672_w13_tma_issuer_on_b_tp4_m128_cold_20260906.log`
- `bench/results/iter672_w13_tma_issuer_off_b_tp4_m128_cold_20260906.log`

## Decision

Reject and remove the implementation.  A normalized `0.111%` change is below
the approximately one-percent noise floor and far below the adoption gate.
The standalone advantage therefore is not explained by repeatedly using one
physical warp as the TMA issuer in the fused persistent W13 phase.

The selected source was restored byte-for-byte to SHA-256
`7ac22134c953d17c8dea9310011818ca483b5b8a96b06324381abd2c8c9c3f45`.
