# Iteration 618 evidence

TP4 M64/M128 short process bracket.  Values below are one-kernel candidate
median divided by its same-window same-source multi-kernel control median.
Every replay is independently cold-L2 via an excluded 256 MiB clear.

| M | OFF_A | ON_A | ON_B | OFF_B | normalized ON change |
|---:|---:|---:|---:|---:|---:|
| 64 | 1.132059654 | 1.132305538 | 1.132102289 | 1.131831335 | +0.0228% slower |
| 128 | 1.132050877 | 1.130938116 | 1.131651778 | 1.132740171 | -0.0972% faster |
| geometric mean | 1.132055266 | 1.131621621 | 1.131877011 | 1.132285662 | -0.0372% faster |

Absolute candidate medians (ms):

| M | OFF_A | ON_A | ON_B | OFF_B |
|---:|---:|---:|---:|---:|
| 64 | 0.280624 | 0.280848 | 0.280544 | 0.281328 |
| 128 | 0.347440 | 0.347008 | 0.346720 | 0.348304 |

Each cell contains 40 cold samples per implementation.  All candidate and
control results pass all-rank finite/cosine/relative-L2 and embedded-allreduce
checks.  The M128 normalized improvement is only 0.097%; aggregate improvement
is 0.037%, so the feature remains default-off pending a long confirmation.

Raw logs:

```text
bench/results/iter618_balanced_activation_off_a_20260905.log
bench/results/iter618_balanced_activation_on_a_20260905.log
bench/results/iter618_balanced_activation_on_b_20260905.log
bench/results/iter618_balanced_activation_off_b_20260905.log
```
