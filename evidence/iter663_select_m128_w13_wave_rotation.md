# Iteration 663: select M128 W13 complete-wave rotation

## Selection

Set `V4_SINGLE_LAUNCH_W13_WAVE_ROTATE` to 13 by default when the TP4 compact
bundle is active.  An explicit environment value of 0 remains a complete
rollback.  The compile-time branch is restricted to TP4 `Tokens==128` inside
the compact W13 callee, so M8/M16/M32/M64 and the separate TP8 kernel do not
execute the mapping.

This promotes the independently bracketed Iteration-659 result: every ON
local W13 sample beat every OFF sample, and four independent TP4 processes
showed a 0.58% normalized / 0.82% direct M128 end-to-end improvement.  The
negative Iterations 660--662 establish that the mechanism must not be copied
to W2 or lower M.

## TP4 all-M formal result

Physical GPUs 0/5/6/7, random routes seed 20260902, CUDA Graph, ten warmups,
six outer batches x fifty replay-interleaved samples, giving 300 samples per
implementation/M.  Every candidate and multi replay had a separate excluded
256 MiB L2 clear immediately before it.  Input X was prequantized FP8-E4M3
plus FP32 group-128 scales and weights were MXFP4.  The one-kernel path
includes route preparation, W13, activation/requant, W2, local top-k6 combine
and embedded all-reduce; multi uses its full same-source kernel sequence plus
SGLang CustomAllReduceV2.

| M | multi median (ms) | one median (ms) | one/multi | one overhead |
|---:|---:|---:|---:|---:|
| 8 | 0.071104 | 0.075696 | 1.064582 | 6.46% |
| 16 | 0.114608 | 0.124144 | 1.083205 | 8.32% |
| 32 | 0.179440 | 0.200752 | 1.118769 | 11.88% |
| 64 | 0.262496 | 0.295024 | 1.123918 | 12.39% |
| 128 | 0.326576 | 0.372384 | 1.140268 | 14.03% |

All five shapes passed all-rank finite/reference/all-reduce gates.  Geometric
mean is `0.165816228 ms` for multi and `0.183358642 ms` for one, so the
selected one-kernel path remains 10.58% slower (`one/multi=1.105794`).  To be
1.10x faster than this multi baseline it still needs a 17.79% reduction from
its current geometric mean.  This formal run measures the selected default;
the causal OFF/ON credit comes from Iteration 659, not a comparison with an
older process regime.

## TP8 runtime gate

All eight H20s, the same five random-route shapes, CUDA Graph, two warmups and
five independently cold-L2 replays per shape.  Every output was finite,
`allreduce_ok=true`, and the run covered split-K4 at M8/M16/M32 and split-K2
at M64/M128.  M8/M16 used embedded multicast push; M32/M64/M128 used embedded
NVLS pull.

TP8 medians for M8/16/32/64/128 were
`0.054720/0.079040/0.122784/0.169248/0.216640 ms`.  This is a liveness and
correctness check, not a new TP8 performance claim.  The TP4-only rotation
does not alter the TP8 implementation.

## Decision

Retain shift 13 as the TP4 compact-bundle default with explicit zero rollback.
The change is a real but small M128 improvement and does not materially close
the objective.  Further work must change persistent GEMM execution rather
than transfer more standalone body snippets.

Raw artifacts:

- `bench/results/iter663_w13_wave_rotate_default_tp4_allm_cold_screen_20260906.log`
- `bench/results/iter663_w13_wave_rotate_default_tp4_allm_cold_formal_20260906.log`
- `bench/results/iter663_w13_wave_rotate_default_tp8_allm_runtime_20260906.log`
