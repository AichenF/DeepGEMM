# Iteration 615 evidence

Read-only import of:

```text
/home/xutingz/fac/profile_results/iter614_current_default_m128_compute.ncu-rep
```

Selected aggregate metrics:

```text
Duration                              343.42 us
Compute (SM) Throughput                60.07 %
DRAM Throughput                        50.73 %
Memory Throughput                       2.44 TB/s
L2 Hit Rate                              5.57 %
Instructions Executed             113,664,963
Grid / block                         702 / 128
Registers Per Thread                        56
Theoretical / Achieved Occupancy    56.25 / 56.22 %
Local Memory Spilling Requests                  0
Issued Warp Per Scheduler                    0.63
Eligible Warps Per Scheduler                 1.82
No Eligible                                 36.61 %
Warp Cycles Per Issued Instruction          14.19
Stall Barrier                                5.37 cycles
Stall Long Scoreboard                        1.52 cycles
Stall Wait                                   1.22 cycles
```

Top not-issued barrier source samples:

| generated source site | samples | share of 4,436 |
|:---|---:|---:|
| activation static-loop exit / next phase barrier | 2,259 | 50.92% |
| terminal W2 convergence mapped to collective guard | 905 | 20.40% |
| WGMMA wait group | 351 | 7.91% |
| W2 loop/task tail | 165 | 3.72% |

At M128, activation work is 128*6*4=3,072 group tasks.  Static stride over
702 physical CTAs assigns five tasks to 264 CTAs and four to 438 CTAs.  A
phase-local worker count of ceil(3072/5)=615 preserves five rounds while
leaving only three four-task workers.  The proposed experiment changes this
activation assignment only; W13 and W2 retain all 702 workers.
