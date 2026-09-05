# Iteration 625 evidence

TP4 cold-L2 process bracket.  Values are one-kernel candidate median divided
by the same-window same-source multi-kernel median.

| M | OFF_A | ON_A | ON_B | OFF_B | normalized ON change |
|---:|---:|---:|---:|---:|---:|
| 64 | 1.133923364 | 1.144810688 | 1.148330269 | 1.132549625 | +1.18% slower |
| 128 | 1.131842298 | 1.156347895 | 1.154553512 | 1.131952411 | +2.08% slower |
| geometric mean | 1.132882353 | 1.150564831 | 1.151437686 | 1.132250979 | +1.63% slower |

Absolute one-kernel candidate medians (ms):

| M | OFF_A | ON_A | ON_B | OFF_B |
|---:|---:|---:|---:|---:|
| 64 | 0.280832 | 0.283968 | 0.283904 | 0.280528 |
| 128 | 0.347376 | 0.354416 | 0.353552 | 0.347120 |

Each cell contains 40 independently cold-L2 samples per implementation.  Both
ON windows reproduce the loss and all correctness/allreduce checks pass, so
the experiment is rejected without a long run.

Raw logs:

```text
bench/results/iter625_w2_n64_off_a_20260905.log
bench/results/iter625_w2_n64_on_a_20260905.log
bench/results/iter625_w2_n64_on_b_20260905.log
bench/results/iter625_w2_n64_off_b_20260905.log
```
