# Iteration 704: W2 FP16 accumulator TP4 remaining-shape gates

Complete one-kernel MegaMoE+embedded-all-reduce FP32 control and W2-FP16
candidate CUDA Graphs were compared on physical H20 GPUs 1,5,6,7. Each M
uses 150 samples per arm, separate excluded 256 MiB L2 clears before every
replay and per-sample alternating order.

| M | FP32 median (ms) | FP16 median (ms) | speedup |
|---:|---:|---:|---:|
| 8 | 0.075312 | 0.075168 | 1.001916x |
| 16 | 0.123552 | 0.122736 | 1.006648x |
| 32 | 0.198432 | 0.197536 | 1.004536x |
| 64 | 0.281920 | 0.280384 | 1.005478x |
| 128 (Iter. 703) | 0.353776 | 0.352432 | 1.003813x |

Five-point equal-weight geometric means are 0.179075 ms control and 0.178277
ms candidate, so FP16 improves 0.448%. M64 and M128 final outputs are bitwise
equal for this seed. M8/M16/M32 are finite and pass the predefined tolerance
gate, with cosine 0.9999977–0.9999979 and relative L2 0.00215–0.00220, but
are not bitwise. The candidate remains opt-in pending an explicit precision
policy decision and a fresh all-M comparison to the multi-kernel baseline.
