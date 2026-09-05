# Iteration 647: production one-kernel M128 cold-L2 phase profile

## Protocol

- H20 physical GPU1, local TP4-shape compute with communication disabled.
- Production one-kernel defaults plus diagnostic-only device phase stamps.
- M128 random routes, seed 20260902, 1,992 padded rows, split-K2.
- Twelve independent processes/samples.  Every measured one-kernel launch was
  immediately preceded by a separate excluded 256 MiB L2 clear.
- Each sample first constructed the independent same-source multi-kernel local
  reference.  All twelve complete routed W2 tensors were bitwise equal to
  that reference: cosine 1.0, relative L2 0.0, finite output.

## Production one-kernel phase distribution

| Device phase | min us | median us | max us |
|---|---:|---:|---:|
| in-kernel route construction | 3.904 | 4.416 | 4.576 |
| persistent W13 | 202.048 | 203.056 | 205.856 |
| SwiGLU + FP8 requant | 6.304 | 6.464 | 6.624 |
| persistent W2 | 106.432 | 107.200 | 108.448 |

Per-sample sums of the four device phases have min/median/max
`319.488/321.440/323.744 us`.  The sum of independently computed phase
medians is 321.136 us.

## Diagnostic comparison with Iteration 646

The matching multi-kernel medians, measured on the same physical GPU, route
seed, input contract, and whole-pipeline cold-L2 policy, are 7.232 us route,
187.584 us W13, 8.096 us requant, and 98.240 us W2.  This is not a paired
same-process latency verdict, but it localizes a stable code-generation and
scheduling signal:

| Phase | multi median us | one-kernel median us | one minus multi us |
|---|---:|---:|---:|
| route | 7.232 | 4.416 | -2.816 |
| W13 | 187.584 | 203.056 | +15.472 |
| requant | 8.096 | 6.464 | -1.632 |
| W2 | 98.240 | 107.200 | +8.960 |

The one-kernel route and requant phases are already 4.448 us faster in
aggregate.  Persistent W13/W2 lose 24.432 us and dominate the net roughly
20 us local-phase deficit.  W13 contributes about 63% and W2 about 37% of
the positive GEMM penalty.  This directly rejects further route/requant
micro-optimization as the next 10%-target direction.

## Decision

The next design must attack persistent GEMM execution while preserving one
business launch.  The measurement favors a W13-first structural experiment,
because its absolute penalty is 1.73x W2's.  Do not infer total TP speedup
from this local diagnostic; retain paired distributed cold-L2 CUDA Graphs as
the selection gate.

Raw log: `bench/results/iter647_one_m128_phase_cold.log`.
