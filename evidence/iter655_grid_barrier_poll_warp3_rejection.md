# Iteration 655: grid-barrier polling-warp rotation rejection

## Hypothesis and isolated change

The ordinary packed whole-grid barrier used `threadIdx.x == 0` for arrival
and generation polling.  The selected route GEMM also concentrates its TMA
and WGMMA issue work in warp 0.  An early CTA polling on the same scheduler
could therefore interfere with a peer CTA still finishing the residual GEMM
wave on that SM.

The experiment adds
`V4_SINGLE_LAUNCH_GRID_BARRIER_POLL_WARP={0,1,2,3}`, default 0, and selects
`threadIdx.x == 32 * warp` as the sole ordinary-barrier leader.  It changes no
task mapping, numerical operation, memory order or communication path.  It is
rejected when cooperative or hierarchical grid barriers are selected.

## Build, resources and correctness

Static Python compilation and `git diff --check` pass.  Exact cubin resource
reports are identical for warp 0 and warp 3:

| shape | registers | stack bytes | shared bytes | local bytes |
|---|---:|---:|---:|---:|
| M8, split-K4 | 63 | 32 | 2,048 | 0 |
| M128, split-K2 | 56 | 32 | 2,048 | 0 |

The selected extensions were:

```text
warp 0: /tmp/torch_ext_v4_tp/v4tp_4aff081c4baf3abd476b_v178mspec
warp 3: /tmp/torch_ext_v4_tp/v4tp_dfe18bdabf90530f7932_v178mspec
```

Local cold-L2 correctness passes for warp-3 M8/M128 and warp-0 M8:

- `cosine=1`, `rel_l2=0`, finite;
- M8 split-K4 has 344 padded rows;
- M128 split-K2 has 1,992 padded rows; and
- packed generation-wrap state is `[0,0,0,0]`.

## TP4 cold-L2 A/B

Protocol: physical H20 GPUs 0/5/6/7, route seed 20260902, M8 and M128, CUDA
Graph, four warmups, two outer batches x 20 replay samples per process, and an
excluded 256 MiB clear before every replay.  All ranks passed correctness and
allreduce.  Independent process order was OFF-A, ON-A, ON-B, OFF-B.

| window | M8 multi (ms) | M8 one (ms) | ratio | M128 multi (ms) | M128 one (ms) | ratio | endpoint GM ratio |
|---|---:|---:|---:|---:|---:|---:|---:|
| OFF-A | 0.071264 | 0.075792 | 1.063538 | 0.307680 | 0.354928 | 1.153562 | 1.107636 |
| ON-A | 0.071344 | 0.075872 | 1.063467 | 0.307472 | 0.354192 | 1.151949 | 1.106824 |
| ON-B | 0.071328 | 0.075568 | 1.059444 | 0.308048 | 0.355024 | 1.152496 | 1.104991 |
| OFF-B | 0.071088 | 0.075376 | 1.060320 | 0.307488 | 0.354992 | 1.154491 | 1.106404 |

Bracket averages:

| metric | warp 0/off | warp 3/on | on/off | apparent gain |
|---|---:|---:|---:|---:|
| normalized M8 ratio | 1.061929 | 1.061455 | 0.999554 | 0.0446% |
| normalized M128 ratio | 1.154026 | 1.152222 | 0.998437 | 0.1563% |
| normalized endpoint GM | 1.107020 | 1.105907 | 0.998995 | 0.1005% |
| direct M8 one-kernel | 0.075584 | 0.075720 | 1.001799 | -0.180% |
| direct M128 one-kernel | 0.354960 | 0.354608 | 0.999008 | 0.099% |

The direct M8 direction disagrees with the normalized result and all effects
are noise-scale.  This cannot explain the persistent-GEMM multi-kernel gap.

## Environment qualification and raw records

Several earlier attempts stopped before timing while reconstructing the
container's compatible SGLang CustomAllReduceV2 Python/JIT environment.  They
are preserved as `iter655a` through `iter655g` failure logs and make no CUDA
latency claim.  The successful A/B uses the older `_symm_tensor` CAR checkout
with a temporary, source-checkout-preserving `ipc.cuh` compatibility overlay.

Raw successful records:

- `bench/results/iter655_pollwarp3_m8_correctness_20260906.log`
- `bench/results/iter655_pollwarp3_m128_correctness_20260906.log`
- `bench/results/iter655_pollwarp0_m8_correctness_20260906.log`
- `bench/results/iter655h_pollwarp_off_a_tp4_m8_m128_cold_20260906.log`
- `bench/results/iter655h_pollwarp_on_a_tp4_m8_m128_cold_20260906.log`
- `bench/results/iter655h_pollwarp_on_b_tp4_m8_m128_cold_20260906.log`
- `bench/results/iter655h_pollwarp_off_b_tp4_m8_m128_cold_20260906.log`

## Decision

Reject the warp-3 variant, leave production at warp 0, and do not spend
all-M or warp-1/2 runs on a roughly 0.1% normalized signal.  Retain only the
default-off diagnostic knob and metadata so the negative result is exactly
reproducible.
