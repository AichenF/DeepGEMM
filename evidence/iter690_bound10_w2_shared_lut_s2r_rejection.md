# Iteration 690: reject shared-broadcast W2 LUT S2R

The M128 W2 candidate preserves packed-weight lookahead, but has one lane per
four-lane weight quartet synthesize each next decoded LUT and write it into a
128-entry (1 KiB) dynamic-shared stage. All lanes consume the shared LUT in
the next K step. This removes decoded LUT values from the cross-QGMMA register
live range while retaining ahead-of-QGMMA synthesis.

Candidate SHA-256 is
`b9f1db40e7c37e004718999c5088d4b1c308608cb8b9bbfb6a56a9b00a32a217`;
extension is `v4tp_54d4ad56c9bfe74a39f7_v178mspec`. Both M128 split-K2/4
entries are `REG48 STACK64 SHARED2048 LOCAL0 CONSTANT[0]1361`; dynamic shared
is 19,456 bytes and the runtime occupancy guard admits ten CTA/SM.

Single-GPU full output is bitwise equal to the independent reference, finite,
and all packed generations wrap to zero. TP4 correctness exactly matches the
control metrics: cosine-min `0.9999955976741226`, rel-L2-max
`0.0029672639980990933`, finite and embedded-allreduce correct.

Cold-L2 timing used physical H20 GPUs 0/5/6/7, M128 random seed 20260902,
replay-paired CUDA Graphs, four warmups and 2 x 20 samples/implementation.
Every arm received its own excluded 256 MiB cache clear.

| Path | Min (ms) | Median (ms) | Max (ms) |
|---|---:|---:|---:|
| Same-source multi-kernel control | 0.303104 | 0.305424 | 0.317280 |
| Shared-LUT-S2R one kernel | 0.373984 | 0.376688 | 0.395040 |

Candidate/control is `1.2333280869`: 23.33% slower. It is also about 7.74%
slower than the adjacent selected one-kernel anchor (0.349616 ms). The added
warp ordering and shared broadcast are more expensive than duplicate LUT
synthesis despite the real ten-CTA occupancy gain.

Verdict: reject and restore selected source SHA-256
`7ac22134c953d17c8dea9310011818ca483b5b8a96b06324381abd2c8c9c3f45`.
