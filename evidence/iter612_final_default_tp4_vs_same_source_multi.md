# Iteration 612 evidence

TP4 physical GPUs 0,5,6,7; random route seed 20260902; CUDA-Graph
replay-granularity AB/BA; 6 x 50 = 300 samples per implementation/M; an
excluded 256 MiB clear before every implementation replay.

| M | active/padded rows | split-K | multi median ms | one-kernel median ms | candidate/control | one-kernel overhead |
|---:|---:|---:|---:|---:|---:|---:|
| 8 | 43/344 | 4 | 0.070975997 | 0.075871997 | 1.068981x | 6.8981% |
| 16 | 82/656 | 4 | 0.114047997 | 0.124191999 | 1.088945x | 8.8945% |
| 32 | 140/1120 | 4 | 0.178623997 | 0.200511999 | 1.122537x | 12.2537% |
| 64 | 203/1624 | 2 | 0.261920005 | 0.295087993 | 1.126634x | 12.6634% |
| 128 | 248/1992 | 2 | 0.331903994 | 0.375087991 | 1.130110x | 13.0110% |

Geometric means: multi 0.1659067372 ms; one-kernel 0.1836876845 ms;
candidate/control 1.107174354x, control/candidate 0.903200112x.

All correctness records reported identical candidate/control values,
`finite_all_ranks=true`, and `allreduce_ok=true`.  The metadata independently
confirms all six selected default components were active.  M64/M128 and their
controls show common process drift, so only paired ratios are used for the
conclusion.
