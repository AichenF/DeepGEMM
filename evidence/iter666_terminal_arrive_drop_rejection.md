# Iteration 666: terminal W2 arrive-drop rejection

## Multi-kernel property under test

The standalone W2 kernel publishes all output by kernel completion.  In the
selected single launch, all 624 CTAs (M8--M64) or 702 CTAs (M128) instead wait
at the terminal phase-3 software grid barrier, although only 78 small-message
or 64 large-message CTAs subsequently execute the collective.

The default-off experiment preserved every CTA's release arrival on the
packed barrier word.  CTAs outside the communication prefix then exited
without polling; only collective CTAs waited for and acquired the final
generation.  This preserves one business kernel, all W2 tasks and the
CARv2-derived collective.  It tests whether standalone kernel-exit semantics
can be approximated by an in-kernel arrive-and-drop tail.

## Correctness and resource gate

- Random-route M128 compute-only output is bitwise equal to the same-source
  multi local result: cosine 1.0, relative L2 0.0 and finite output.
- All four packed generation words advance to
  `[2048, 2048, 2048, 2048]`.
- Every TP4 OFF/ON endpoint passes the same all-rank reference and embedded
  all-reduce checks.
- OFF and ON M128 entries both compile to
  `REG56 STACK32 SHARED2048 LOCAL0`; occupancy is unchanged.

## Local cold-L2 phase ABBA

Protocol: physical GPU1, M128 random seed 20260902, eight independent
processes in `OFF/ON/ON/OFF/OFF/ON/ON/OFF` order, each with a separate
excluded 256 MiB L2 clear.

- OFF W2 phase: `105.664, 106.624, 106.720, 107.872 us`; median
  `106.672 us`, mean `106.720 us`.
- ON W2 phase: `106.592, 105.824, 105.376, 107.200 us`; median
  `106.208 us`, mean `106.248 us`.

Median and mean improve only 0.44%, below the 1% gate.

## TP4 cold-L2 endpoint bracket

Protocol: GPUs 0/5/6/7, random seed 20260902, CUDA Graph, four warmups and two
batches of 20 replay-interleaved samples per implementation/process.  Every
replay has its own excluded 256 MiB L2 clear.  Independent process order is
OFF/ON/ON/OFF.  The table averages each process's one/multi ratio, avoiding
cross-window clock drift.

| M | OFF one/multi | ON one/multi | normalized improvement |
|---:|---:|---:|---:|
| 8 | 1.066540 | 1.064377 | 0.20% |
| 16 | 1.083917 | 1.083078 | 0.08% |
| 32 | 1.120379 | 1.118306 | 0.19% |
| 64 | 1.127018 | 1.125293 | 0.15% |
| 128 | 1.147744 | 1.146042 | 0.15% |

The geometric mean of the five averaged ratios moves only
`1.108722 -> 1.107022`, a 0.15% improvement.  M128 absolute one-kernel
latency averages `0.350016 -> 0.349616 ms`, only 0.11%.  No M reaches the 1%
selection threshold.

## Decision

Reject and remove the experiment.  The phase-3 polling population is not a
material source of the single-kernel gap, and a roughly 0.15% endpoint signal
does not justify 70 lines of synchronization and configuration complexity.
Production source and benchmark return byte-identical to commit `b2c3e9d`.
The result further narrows the useful multi-kernel advantage to fresh-CTA GEMM
scheduling/resource contracts, not terminal barrier occupancy.

Raw evidence:

- `bench/results/iter666_terminal_arrive_drop_m128_gate_20260906.log`
- `bench/results/iter666_terminal_arrive_drop_m128_phase_abba_20260906.log`
- `bench/results/iter666_terminal_drop_{off,on}_{a,b}_tp4_m128_cold_20260906.log`
- `bench/results/iter666_terminal_drop_{off,on}_{a,b}_tp4_lower_m_cold_20260906.log`
