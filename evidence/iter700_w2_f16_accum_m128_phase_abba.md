# Iteration 700: M128 W2 packed-FP16 cold phase A/B

Five OFF and five ON processes were interleaved
`0/1/1/0/1/0/0/1/1/0` on physical H20 GPU1. Each process warms once, then
runs exactly one phase-stamped candidate after a separate excluded 256 MiB
L2 clear. TP communication is disabled only for phase attribution.

| metric (us) | FP32 OFF min/median/max | FP16 ON min/median/max | median change |
|---|---:|---:|---:|
| W2 | 106.112 / 106.912 / 107.360 | 104.928 / 105.728 / 106.176 | -1.107% |
| route+W13+requant+W2 | 326.208 / 327.968 / 329.152 | 324.416 / 326.496 / 328.192 | -0.449% |

Means are 106.784/105.664 us for W2 and 327.866/326.336 us for the four
phase sum, favoring FP16 by 1.049% and 0.467%, respectively. Every FP16 run
repeats cosine `0.9999993512848424`, relative L2
`0.001139531250097178`, finite output and correct packed-barrier wrap; the
FP32 runs remain bitwise equal to the independent multi reference.

This eager device-stamp result is a local gate, not a distributed benchmark.
The candidate advances to M128 TP4 CUDA-Graph A/B because its W2 distributions
nearly separate and total phase median/mean both improve.
