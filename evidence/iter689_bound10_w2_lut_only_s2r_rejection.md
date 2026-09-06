# Iteration 689: reject LUT-only W2 lookahead

This candidate reverses Iterations 687/688: it keeps decoded LUT values live
across QGMMA but reloads packed MXFP4 weights in the consuming iteration.
Candidate SHA-256 is
`d1cbe44b15d07857d87da9e0b78d1e9ec71a3f2763b5b997d76a48c6d1557acb`;
extension is `v4tp_43592085816b37400d77_v178mspec`.

M128 split-K2/4 resources are `REG48 STACK64 SHARED2048 LOCAL0
CONSTANT[0]1361`, so ten CTA/SM is admitted. However local full output is no
longer exact (cosine `0.9999980904790449`, rel-L2
`0.0019542379331360005`). TP4 remains accepted but worsens to cosine-min
`0.9999926659126305`, rel-L2-max `0.0038299181204722544`, and max absolute
error 17,408 versus 1,024 for control.

Cold-L2 timing used physical H20 GPUs 0/5/6/7, M128 random seed 20260902,
replay-paired CUDA Graphs, four warmups and 2 x 20 samples/implementation.
Every timed arm had an independent excluded 256 MiB cache clear.

| Path | Min (ms) | Median (ms) | Max (ms) |
|---|---:|---:|---:|
| Same-source multi-kernel control | 0.302752 | 0.305792 | 0.333504 |
| LUT-only-lookahead one kernel | 0.365376 | 0.368032 | 0.443584 |

Candidate/control is `1.2035370083`; the candidate is 20.35% slower. It is
also slower than both packed-lookahead low-register candidates, so delayed
packed loads are neither a performance nor an exactness-safe trade.

Verdict: reject and restore selected source SHA-256
`7ac22134c953d17c8dea9310011818ca483b5b8a96b06324381abd2c8c9c3f45`.
