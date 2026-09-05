# Iteration 604 evidence

All samples are TP4 CUDA-graph replays with a separate excluded 256 MiB cold-L2 clear immediately before each implementation replay. Ratios compare the candidate with its same-process same-source multi-kernel control.

| M | OFF_A ratio | ON_A ratio | ON_B ratio | OFF_B ratio | normalized ON change |
|---:|---:|---:|---:|---:|---:|
| 8 | 1.067416 | 1.067446 | 1.067835 | 1.071991 | -0.19% |
| 16 | 1.088855 | 1.091354 | 1.084836 | 1.091430 | -0.19% |
| 32 | 1.131510 | 1.125655 | 1.126957 | 1.128953 | -0.35% |
| 64 | 1.127780 | 1.125377 | 1.124624 | 1.128916 | -0.30% |

Negative change denotes lower candidate/control latency. M8/M16 are noise-scale but do not show a pooled regression; M32/M64 have both ON windows below both OFF windows.

