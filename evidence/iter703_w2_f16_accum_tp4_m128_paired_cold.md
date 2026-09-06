# Iteration 703: W2 FP16 accumulator full TP4 M128 paired result

On physical H20 GPUs 1,5,6,7, all CUDA peer-access matrix entries are true
and every pair is NV18. The run compares complete FP32-control and FP16-W2
one-kernel MegaMoE+embedded-all-reduce CUDA Graphs in one process. Each of
150 samples per arm follows its own excluded 256 MiB L2 clear; sample order
alternates A/B then B/A.

| variant | min / median / max (ms) |
|---|---:|
| selected FP32 control | 0.346880 / 0.353776 / 0.545664 |
| W2 packed-FP16 candidate | 0.346752 / 0.352432 / 0.364672 |

Median control/candidate speedup is `1.003813x` (candidate latency -0.381%).
All five paired batch medians favor the candidate. Final graph outputs are
bitwise equal on every rank with cosine 1, relative L2 0, max absolute error
0 and all finite values.

`SGLANG_SKIP_P2P_CHECK=1` bypasses only the environment-broken standalone
Python checker from Iteration 702 after peer access was independently
verified; actual peer communication and the embedded collective execute in
both timed graphs. The candidate remains opt-in pending M8/16/32/64 gates.
