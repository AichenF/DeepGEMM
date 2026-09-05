# Iteration 646: same-source multi-kernel M128 cold-L2 stage profile

## Protocol

- H20 physical GPU1, one local TP4 rank.
- M128 random routes, seed 20260902, prequantized FP8-E4M3 activation and
  MXFP4 weights.
- Selected same-source multi-kernel pipeline: route alignment, W13,
  SwiGLU/FP8 requantization, W2, and local k6 reduction.
- CUDA Graph, eight warmups and 100 measured replays.
- Every complete local pipeline replay was immediately preceded by a separate
  256 MiB L2 clear.  The clear was excluded from all CUDA events.  L2 was not
  incorrectly cleared between dependent stages.
- Communication was intentionally absent because this measurement isolates
  local compute scheduling; no end-to-end TP latency claim is made.

## Result

| Stage | min us | median us | max us |
|---|---:|---:|---:|
| route alignment | 6.624 | 7.232 | 7.936 |
| W13 | 185.984 | 187.584 | 190.080 |
| activation + FP8 requant | 7.904 | 8.096 | 9.088 |
| W2 | 97.216 | 98.240 | 99.040 |
| local k6 reduce | 6.240 | 6.400 | 7.104 |
| complete local pipeline | 304.768 | 307.760 | 310.048 |

The sum of stage medians is 307.552 us, within 0.208 us of the independently
recorded total median.  This validates the event partition and shows that W13
plus W2 account for 285.824 us, or 92.87% of the local multi-kernel pipeline.

## Decision

This is the multi-kernel side of a paired diagnostic, not a candidate gate.
Collect the production one-kernel M128 phase stamps next using the same GPU,
route seed, prequantized input contract, and whole-pipeline cold-L2 policy.
Do not infer a cross-implementation delta until that matching measurement is
closed.

Raw log: `bench/results/iter646_multi_m128_stage_cold.log`.
