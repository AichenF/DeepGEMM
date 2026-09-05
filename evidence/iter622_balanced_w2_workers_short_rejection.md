# Iteration 622 evidence

TP4 cold-L2 short bracket; values are one-kernel candidate median divided by
the same-window same-source multi-kernel control median.

| M | OFF_A | ON_A | ON_B | OFF_B | normalized ON change |
|---:|---:|---:|---:|---:|---:|
| 64 | 1.132846843 | 1.157209176 | 1.157201018 | 1.132356737 | +2.17% slower |
| 128 | 1.130119835 | 1.181291747 | 1.179088872 | 1.132480417 | +4.32% slower |
| geometric mean | 1.131482518 | 1.169188457 | 1.168093679 | 1.132418575 | +3.24% slower |

Absolute one-kernel candidate medians (ms):

| M | OFF_A | ON_A | ON_B | OFF_B |
|---:|---:|---:|---:|---:|
| 64 | 0.280656 | 0.287136 | 0.286560 | 0.280752 |
| 128 | 0.346992 | 0.362288 | 0.361952 | 0.346992 |

Every cell contains 40 independently cold-L2 samples per implementation.
All-rank numerical and embedded-allreduce checks pass.  The two ON windows
reproduce the large loss, so the option is rejected without a longer run.

Raw logs:

```text
bench/results/iter622_balanced_w2_off_a_20260905.log
bench/results/iter622_balanced_w2_on_a_20260905.log
bench/results/iter622_balanced_w2_on_b_20260905.log
bench/results/iter622_balanced_w2_off_b_20260905.log
```
