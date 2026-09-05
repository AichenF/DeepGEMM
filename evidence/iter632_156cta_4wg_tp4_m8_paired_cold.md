# Iteration 632: 156 CTA × 4 WG TP4 M8 paired cold-L2 screen

## Compared implementations

- Control: selected one-kernel MegaMoE, 624 CTAs × 128 threads, production compact-W13/dynamic-route/M128-bound bundle.
- Candidate: 156 CTAs × 512 threads, four independent 128-thread WGMMA groups per CTA, isolated inline/bound-8 configuration.
- Both graphs include route preparation, W13, activation/requant, W2, local combine, and the M8 embedded multicast-push TP allreduce.

## Method

TP4 used physical GPUs 0,5,6,7. The existing paired graph harness alternated order for every sample and cleared a separate 256 MiB buffer immediately before each timed replay; the clear is excluded from events. Reported values are the maximum elapsed time across ranks. There were 150 samples per arm in three batches.

## Result

| Variant | Min (ms) | Median (ms) | Max (ms) | Batch medians (ms) |
|---|---:|---:|---:|---|
| 624×128 control | 0.074048 | 0.075552 | 0.229952 | 0.075744, 0.075424, 0.075792 |
| 156×512 candidate | 0.085952 | 0.087696 | 0.089888 | 0.087872, 0.087568, 0.087680 |

`control / candidate = 0.861522×`; equivalently the candidate adds 0.012144 ms or 16.07% median latency. Outputs are bitwise identical on every rank. The single control max outlier does not affect the median or batch-direction conclusion.

## Conclusion

Packing four WGs per CTA reduces grid-barrier participants but increases small-M latency substantially, so it is rejected for M8. One M128 endpoint remains useful to rule in or out a size-dependent crossover.
