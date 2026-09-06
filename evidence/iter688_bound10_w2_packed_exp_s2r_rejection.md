# Iteration 688: reject packed-weight plus exponent W2 S2R

The candidate extends Iteration 687's low-register M128 W2 path by carrying
the next two E8M0 exponents across the current QGMMA along with both packed
weight words. It therefore hides all shared loads but intentionally leaves
LUT synthesis in the consuming iteration.

Candidate SHA-256 is
`3c51bc84f02f811a9705276a772ff43bac35d10a85d43376ffeb18211503c122`;
the JIT extension is `v4tp_37f1cb1981a2be64af96_v178mspec`. Both M128
split-K2/4 entries are `REG48 STACK48 SHARED2048 LOCAL0 CONSTANT[0]1361`, so
the desired ten-CTA/SM admission is preserved without fixed local spill.

Single-GPU random-route M128 full output is bitwise equal to the independent
same-source reference, finite, and wraps every packed grid generation word to
zero. TP4 also passes: cosine-min `0.9999955976741226`, rel-L2-max
`0.0029672639980990933`, finite and embedded-allreduce correct on all ranks.

Cold-L2 timing used physical GPUs 0/5/6/7, random seed 20260902, replay-paired
CUDA Graphs, four warmups and 2 x 20 samples per implementation. Every arm
received a separate excluded 256 MiB cache clear.

| Path | Min (ms) | Median (ms) | Max (ms) |
|---|---:|---:|---:|
| Same-source multi-kernel control | 0.303328 | 0.305600 | 0.327616 |
| Packed+exponent S2R one kernel | 0.361824 | 0.363888 | 0.385152 |

Candidate/control is `1.1907329601`, or 19.07% slower. It is statistically
the same as Iteration 687's 0.363680-ms packed-only result (+0.057%). This
isolates the missing benefit to the ahead-of-QGMMA LUT synthesis, not to the
E8M0 shared-memory loads.

Verdict: reject and restore selected source SHA-256
`7ac22134c953d17c8dea9310011818ca483b5b8a96b06324381abd2c8c9c3f45`.
