# Iteration 633: 156 CTA × 4 WG TP4 M128 paired cold-L2 endpoint

## Method

The comparison is identical to Iteration 632 except M=128: TP4 GPUs 0,5,6,7, random routes, shared weights/inputs, two captured end-to-end one-kernel graphs, 150 samples per variant, per-sample A/B↔B/A order, excluded 256 MiB clear immediately before every replay, and maximum event time across ranks.

Both graphs include route preparation, W13, activation/requant, W2, local combine, and the embedded P2P two-shot TP allreduce selected for M128.

## Result

| Variant | Min (ms) | Median (ms) | Max (ms) | Batch medians (ms) |
|---|---:|---:|---:|---|
| 624×128 selected control | 0.343296 | 0.347712 | 0.467712 | 0.346272, 0.347040, 0.351504 |
| 156×512/4-WG candidate | 0.382720 | 0.399008 | 0.411648 | 0.398000, 0.398928, 0.402976 |

The outputs are bitwise identical on every rank. `control / candidate = 0.871441×`; equivalently, the candidate adds 0.051296 ms or 14.75% median latency.

## Decision

The candidate loses at both measured endpoints (M8 by 16.07%, M128 by 14.75%), with directionally consistent batches. The intended barrier-participant saving does not produce a crossover, so intermediate M values are not worth spending shared-GPU time on and this topology is rejected for production selection.
