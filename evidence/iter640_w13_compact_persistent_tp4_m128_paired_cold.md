# Iteration 640: compact-W13 persistent-state TP4 M128 rejection

## Protocol

- TP4 physical H20 GPUs 0, 5, 6 and 7; maximum latency across ranks.
- Random-route DeepSeek-V4-Flash M128, seed 20260902.
- Selected production one-launch control and same kernel with only
  `V4_SINGLE_LAUNCH_W13_COMPACT_PERSISTENT_STATE=1` enabled.
- Three outer batches x fifty samples = 150 samples per arm after six
  alternating warmups; per-sample A/B then B/A ordering.
- Separate 256 MiB L2 clear immediately before every graph replay, excluded
  from CUDA event timing.
- Each graph has the complete MegaMoE business kernel including embedded P2P
  two-shot TP all-reduce.

## Correctness

```text
exact_all_ranks = true
relative L2 max-rank = 0.0
maximum absolute difference = 0.0
finite_all_ranks = true
```

## Latency

| Arm | Min (ms) | Median (ms) | Max (ms) | Batch medians (ms) |
|---|---:|---:|---:|---|
| Production control | 0.344064 | 0.355696 | 0.539520 | 0.346800, 0.356784, 0.357504 |
| Compact-W13 persistent | 0.348864 | 0.360960 | 0.368640 | 0.351456, 0.361488, 0.363488 |

The control/candidate median ratio is `0.985417x`; the candidate is 1.48%
slower.  All three paired batches agree, with candidate penalties of roughly
4.7--6.0 microseconds.  The one control maximum outlier does not affect this
direction.  The candidate is rejected and remains default-off.
