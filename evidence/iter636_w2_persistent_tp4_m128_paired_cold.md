# Iteration 636: W2-only persistent-state TP4 M128 paired cold-L2 result

## Protocol

- Physical GPUs: H20 0, 5, 6 and 7; rank-max timing.
- Shape: random-route DeepSeek-V4-Flash TP4 M128, seed 20260902.
- Arms: selected production one-launch kernel versus the same kernel with
  only `V4_SINGLE_LAUNCH_W2_PERSISTENT_STATE=1`.
- CUDA Graph: one complete MegaMoE kernel per arm, including embedded P2P
  two-shot TP all-reduce.
- Samples: 3 outer batches x 50 samples = 150 per arm after six alternating
  warmups.  Every sample alternates A/B then B/A.
- Cache: a separate 256 MiB clear immediately precedes each graph replay;
  the clear is outside CUDA event timing.

## Correctness

All ranks are bitwise equal between arms:

```text
exact_all_ranks = true
relative L2 max-rank = 0.0
maximum absolute difference = 0.0
finite_all_ranks = true
```

## Latency

| Arm | Min (ms) | Median (ms) | Max (ms) | Batch medians (ms) |
|---|---:|---:|---:|---|
| Production control | 0.344320 | 0.350032 | 0.504480 | 0.346288, 0.349968, 0.366320 |
| W2 persistent state | 0.346400 | 0.351568 | 0.374336 | 0.349072, 0.351184, 0.369680 |

The pooled control/candidate ratio is `0.995631x`: the candidate is 0.44%
slower.  All three paired batches agree on the losing direction.  Because
M128 supplies the greatest number of consecutive W2 tasks per CTA and still
shows no gain, this isolated state reuse is rejected without screening M8.
