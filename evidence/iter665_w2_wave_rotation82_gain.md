# Iteration 665: select TP4 M128 W2 complete-wave rotation 82

## Question and transfer from the multi-kernel control

There is no multi-only commit to cherry-pick: `tpmoe_multikernel_baseline`
is an exact ancestor of the single-launch branch, and both implementations
already call the same `route_gemm_task` body.  The remaining multi-kernel
advantage is structural: W2 starts with fresh CTAs, whereas the single kernel
reuses persistent CTAs after W13 and activation.

This experiment transfers one bounded scheduling property instead of copying
code.  For TP4 M128 only, complete W2 waves rotate logical task ownership by
82 workers per wave.  The eleven complete waves remain a bijection over the
same tasks; the residual wave, arithmetic, barriers, W13, activation and
embedded all-reduce are unchanged.  `V4_SINGLE_LAUNCH_W2_WAVE_ROTATE=0` is the
explicit rollback.

## Resource and correctness gate

- Shift 0 and shift 82 both compile to `REG56 STACK32 SHARED2048 LOCAL0` for
  the M128 entry, preserving nine resident CTAs per SM.
- Random-route M128 output is bitwise equal to the same-source multi-kernel
  local result: cosine 1.0, relative L2 0.0 and all values finite.
- Packed generation words finish at `[2048, 2048, 2048, 2048]` for both
  controls, so the persistent barrier generation accounting is unchanged.
- TP8 uses a separate kernel and compile-time excludes this mapping.

## Local cold-L2 phase screen

Protocol: physical GPU1, random seed 20260902, eight independent processes in
`0,82,82,0,0,82,82,0` order.  Every sample uses a separate excluded 256 MiB
L2 clear.  W2 phase measurements were:

- shift 0: `107.232, 107.200, 109.024, 107.488 us`; median `107.360 us`, mean
  `107.736 us`;
- shift 82: `106.144, 106.976, 105.632, 106.016 us`; median `106.080 us`, mean
  `106.192 us`.

The W2 median improves 1.19% and the mean improves 1.43%, passing the
predeclared 1% phase gate.  All samples remain bitwise correct.

## TP4 cold-L2 endpoint bracket

Protocol: GPUs 0/5/6/7, M128, random seed 20260902, CUDA Graph, four warmups,
two batches of 20 replay-interleaved samples per implementation, and a
separate excluded 256 MiB L2 clear before every replay.  Order was
OFF/ON/ON/OFF in independent processes.

| Window | Multi (ms) | One (ms) | One / multi |
|---|---:|---:|---:|
| OFF A | 0.304800 | 0.351456 | 1.153071 |
| ON A | 0.304848 | 0.349376 | 1.146066 |
| ON B | 0.304976 | 0.349392 | 1.145638 |
| OFF B | 0.305568 | 0.351696 | 1.150958 |

The average OFF one-kernel latency is `0.351576 ms`; ON is `0.349384 ms`, a
0.62% reduction.  Normalizing each process to its same-run multi control,
the average ratio improves from 1.152015 to 1.145852, or 0.54%.  Both ON
windows beat both OFF windows after normalization.

A longer ON-only run (eight warmups, four batches of 50 samples per
implementation) reports `0.322464/0.368352 ms` multi/one, ratio 1.142304.
Its batch medians drift substantially later in the run, so it supports the
direction but is not used to inflate the effect size.

## Formal TP4 all-M rebaseline

Protocol: GPUs 0/5/6/7; M in 8, 16, 32, 64 and 128; random route seed
20260902; CUDA Graph; ten warmups plus six batches of 50 replay-interleaved
samples per implementation and M; separate excluded 256 MiB L2 clear before
every replay.  All reference and embedded-allreduce gates pass on all ranks.

| M | Multi median (ms) | One median (ms) | One overhead |
|---:|---:|---:|---:|
| 8 | 0.071040 | 0.075504 | 6.28% |
| 16 | 0.114144 | 0.123680 | 8.35% |
| 32 | 0.177152 | 0.198256 | 11.91% |
| 64 | 0.260624 | 0.292720 | 12.32% |
| 128 | 0.332992 | 0.377408 | 13.34% |

The geometric means are `0.165634214 ms` for multi and `0.182873954 ms` for
one, so one remains 10.41% slower.  Relative to Iteration 663, the M128 gap
falls from approximately 14.03% to 13.34%, consistent with the short
OFF/ON bracket; the five-M ratio improves only about 0.15%.  Clock drift is
visible at M64/M128, so paired ratios, not raw cross-run latencies, are the
primary comparison.

To reach the stated objective of 1.10x faster than multi, the current
one-kernel geometric mean would have to fall to `0.150576558 ms`: another
17.66% reduction.

## TP8 runtime gate

All eight GPUs and all five M values pass CUDA-Graph liveness, finite/reference
and all-reduce checks under cold L2.  Five-sample medians are
`0.055296, 0.078144, 0.122112, 0.168576, 0.216320 ms`; this is a runtime gate,
not a performance selection run.

## Decision

Retain shift 82 as the compact-bundle default for TP4 M128 only, with explicit
zero rollback.  This is a reproducible but small scheduling-property
adaptation, not a literal cherry-pick and not a material closure of the
single-kernel gap.  A larger gain still requires a different persistent GEMM
execution/dataflow rather than another copy of the standalone kernel body.

Raw evidence:

- `bench/results/iter665_w2_wave_rotate0_82_m128_phase_abba_20260906.log`
- `bench/results/iter665_w2_wave_rotate{0,82}_m128_gate_20260906.log`
- `bench/results/iter665_w2_wave_rotate_{off,on}_{a,b}_tp4_m128_cold_20260906.log`
- `bench/results/iter665_w2_wave_rotate82_tp4_m128_cold_long_20260906.log`
- `bench/results/iter665_w2_wave_rotate82_default_tp4_allm_cold_formal_20260906.log`
- `bench/results/iter665_w2_wave_rotate82_default_tp8_allm_runtime_20260906.log`
