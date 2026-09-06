# Iteration 705: latest W2-FP16 one-kernel versus multi baseline

The formal run uses physical H20 GPUs 1,5,6,7, random routes, seed 20260902,
CUDA Graphs and 300 samples per implementation/M. Each graph replay follows
its own excluded 256 MiB L2 clear. The multi control is the selected
same-source kernel sequence plus SGLang CustomAllReduceV2; the candidate is
the complete TP4 MegaMoE+embedded-all-reduce single launch with only its W2
WGMMA accumulator changed to packed FP16.

| M | multi median (ms) | one median (ms) | one / multi | one slower |
|---:|---:|---:|---:|---:|
| 8 | 0.070960 | 0.075296 | 1.061105 | 6.11% |
| 16 | 0.114368 | 0.123168 | 1.076945 | 7.69% |
| 32 | 0.179872 | 0.200560 | 1.115015 | 11.50% |
| 64 | 0.260112 | 0.291424 | 1.120379 | 12.04% |
| 128 | 0.330672 | 0.374768 | 1.133353 | 13.34% |

Equal-weight geometric means are 0.165870 ms multi and 0.182625 ms one, so
one-kernel is 10.101% slower (`multi/one=0.908254x`). To become 1.10x faster
than multi from this measured point, one-kernel needs another 17.431%
geometric-mean latency reduction.

Both paths pass the independent reference and all-reduce checks at every M;
candidate minimum cosine is 0.999994105, maximum relative L2 is 0.00343392,
and every rank is finite. This candidate is still an opt-in numerical
tradeoff and not a default selection.
