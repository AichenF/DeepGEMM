# Iteration 675: reject full W2 N64 / ten-CTA transfer

## Multi-kernel property under test

The same-source standalone W2 kernel has a lower-resource phase contract than
the fused entry.  This experiment tried to reproduce that property inside the
single kernel at TP4 M128 by splitting every logical W2 N128 task into two
disjoint N64 CTAs.  The resulting entry admits ten CTAs per H20 SM instead of
the selected fused kernel's nine.

This is not a literal commit cherry-pick: `tpmoe_multikernel_baseline` is an
ancestor of this branch and the GEMM body is already shared.  It is a bounded
transfer of the standalone launch's lower-register scheduling property.

Candidate controls:

```text
V4_SINGLE_LAUNCH_W2_FULL_N64=1
V4_SINGLE_LAUNCH_M128_BOUND9=0
V4_SINGLE_LAUNCH_MIN_BLOCKS=10
V4_SINGLE_LAUNCH_CTAS_PER_SM=10
V4_SINGLE_LAUNCH_W13_WAVE_ROTATE=0
V4_SINGLE_LAUNCH_W2_WAVE_ROTATE=0
```

## Resource and correctness gate

- H20 M128 entry: `REG48 STACK64 SHARED2048 LOCAL0`.
- CUDA occupancy admits 10 CTAs/SM, giving a 780-CTA physical grid.
- An initial register-only form had sparse nondeterministic one-route by
  16-column W2 corruption.  The N64 mapping itself was exonerated because the
  unconstrained 64-register build was bitwise exact.
- Explicitly storing the four long-lived FP32 accumulators per lane in the
  otherwise-unused upper half of each physical N128 shared-memory weight stage
  removed the corruption.
- Five independent random-route seeds (`20260902` through `20260906`) all
  produced cosine approximately 1.0, relative L2 exactly 0.0, finite output,
  and packed generation words `[0, 0, 0, 0]` after the requested wrap test.

## TP4 cold-L2 endpoint gate

Protocol: GPUs 0/5/6/7, M128, random seed 20260902, CUDA Graph, four warmup
replays, two batches of 20 replay-interleaved samples per implementation, and
a separate excluded 256 MiB L2 clear immediately before every replay.
Same-source multi-kernel correctness and candidate embedded-allreduce checks
passed on all ranks.

| implementation | median (ms) | min (ms) | max (ms) |
|---|---:|---:|---:|
| multi-kernel control | 0.305104 | 0.303488 | 0.310336 |
| single, full W2 N64 / 10 CTA | 0.420336 | 0.416928 | 0.432896 |

The candidate/control ratio is `1.377681`: the candidate is 37.77% slower.
Its two rank-max batch medians were `0.419952` and `0.420880 ms`, so this is
not a single outlier.

## Decision

Reject and restore the selected Iteration-665 source
(`sha256 7ac22134c953d17c8dea9310011818ca483b5b8a96b06324381abd2c8c9c3f45`).
The extra resident CTA does not compensate for doubling the W2 task count and
activation-side traffic, plus the 64-byte stack footprint needed for strict
correctness.  A useful transfer from the multi-kernel path must preserve N128
arithmetic/traffic while reducing the fused entry's live state; changing the
whole W2 tile granularity is the wrong mechanism.

Raw performance evidence:

- `bench/results/iter675_full_w2_n64_10cta_tp4_m128_cold_short_20260906.log`

