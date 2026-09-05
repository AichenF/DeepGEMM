# Iteration 619 evidence

Long TP4 cold-L2 process bracket.  Each cell is one-kernel candidate median
divided by its same-window same-source multi-kernel control median, from 300
samples per implementation/M.  The cache clear is a separate excluded 256 MiB
operation before every graph replay.

| M | OFF_A | ON_A | ON_B | OFF_B | normalized ON change |
|---:|---:|---:|---:|---:|---:|
| 64 | 1.133546022 | 1.134420780 | 1.132463019 | 1.132689723 | +0.0286% slower |
| 128 | 1.128247474 | 1.125366115 | 1.123755040 | 1.126942577 | -0.2691% faster |
| geometric mean | 1.130893645 | 1.129884378 | 1.128100627 | 1.129812496 | -0.1204% faster |

Absolute one-kernel candidate medians (ms), shown to expose process drift:

| M | OFF_A | ON_A | ON_B | OFF_B |
|---:|---:|---:|---:|---:|
| 64 | 0.289952 | 0.290448 | 0.292864 | 0.292832 |
| 128 | 0.371040 | 0.375008 | 0.373680 | 0.372432 |

All candidate and control outputs have identical numerical metrics.  The
activation-only assignment remains default-off: M64 is slightly worse and the
aggregate 0.1204% normalized tendency is well inside measurement/system drift.

Raw logs:

```text
bench/results/iter619_balanced_activation_off_a_long_20260905.log
bench/results/iter619_balanced_activation_on_a_long_20260905.log
bench/results/iter619_balanced_activation_on_b_long_20260905.log
bench/results/iter619_balanced_activation_off_b_long_20260905.log
```
