# Iteration 611 evidence

TP4 H20 GPUs 0,5,6,7; random routes, seed 20260902; CUDA-Graph replay-level
AB/BA; 6 x 50 = 300 samples per implementation and M; excluded 256 MiB clear
before every replay.  Only `V4_SINGLE_LAUNCH_TP4=1` was set for the selected
path; bundle/component overrides were unset.

| M | active/padded rows | split-K | Humming+CARv2 median ms | one-kernel median ms | Humming/custom | candidate latency reduction |
|---:|---:|---:|---:|---:|---:|---:|
| 8 | 43/344 | 4 | 0.088735998 | 0.075824000 | 1.170289x | 14.5510% |
| 16 | 82/656 | 4 | 0.144383997 | 0.124191999 | 1.162587x | 13.9849% |
| 32 | 140/1120 | 4 | 0.224048004 | 0.198272005 | 1.130003x | 11.5047% |
| 64 | 203/1624 | 2 | 0.313823998 | 0.282191992 | 1.112094x | 10.0795% |
| 128 | 248/1992 | 2 | 0.394656003 | 0.359103993 | 1.099002x | 9.0084% |

Geometric means: Humming 0.2042551275 ms, candidate 0.1800467661 ms;
Humming/custom 1.134455963x, candidate latency reduction 11.8520%.

All candidate and Humming outputs reported `finite_all_ranks=true` and
`allreduce_ok=true`.  Candidate batch medians (ms):

```text
M8   0.075520, 0.075856, 0.076032, 0.075840, 0.075632, 0.076000
M16  0.124048, 0.124336, 0.124400, 0.124272, 0.124096, 0.123744
M32  0.198128, 0.198256, 0.197808, 0.198432, 0.198304, 0.198592
M64  0.281136, 0.284544, 0.285920, 0.281920, 0.280784, 0.281248
M128 0.350400, 0.353072, 0.358608, 0.360064, 0.382736, 0.396768
```

Humming M128 batch medians drift in the same direction
(`0.382464` to `0.439264` ms), so the paired ratio remains the defensible
quantity; no cross-process absolute-latency inference is made.
