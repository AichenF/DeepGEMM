# Iteration 644: compact-W13 split-major TP4 M128 rejection

## Protocol

- TP4 physical H20 GPUs 0, 5, 6, and 7; maximum rank latency.
- Random-route M128, seed 20260902.
- Production one-launch control versus only
  `V4_SINGLE_LAUNCH_W13_COMPACT_SPLIT_MAJOR_TASKS=1`.
- Five outer batches x thirty samples = 150 samples per arm after six
  alternating warmups.
- Replay-level A/B then B/A ordering and a separate excluded 256 MiB L2 clear
  immediately before every graph replay.
- Complete one-kernel MegaMoE including embedded P2P two-shot TP all-reduce.

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
| Production control | 0.344640 | 0.359328 | 0.493248 | 0.346560, 0.346656, 0.367264, 0.368000, 0.367424 |
| Split-major candidate | 0.344992 | 0.361888 | 0.375776 | 0.347744, 0.348016, 0.367248, 0.369104, 0.367808 |

Control/candidate is `0.992926x`; the candidate is 0.71% slower and loses
four of five batch medians.  Its single 0.016-microsecond nominal batch win is
noise-sized.  The candidate is rejected and remains default-off.
