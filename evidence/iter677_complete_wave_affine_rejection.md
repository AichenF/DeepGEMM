# Iteration 677: reject complete-wave affine ownership maps

Date: 2026-09-06

## Multi-kernel property under test

Standalone W13 and W2 obtain completion-driven fresh-CTA task ownership.  The
selected one-kernel implementation already approximates cross-wave owner
decorrelation with M128-only W13 shift 13 and W2 shift 82.  This experiment
asked whether a second static bijection *inside* each complete 702-task wave
could transfer more of the standalone scheduling benefit without a queue,
atomic, barrier, or math change.  Residual waves remained unchanged.

## Resource pre-gate

A coprime multiply-by-five W13 map was numerically exact but increased both
M128 split-K2 and split-K4 entry stack frames from 32 to 48 bytes while
retaining 56 registers.  Replacing runtime remainder with bounded constant
subtractions did not restore the frame.  That version was rejected before
timing because it was not a pure ownership A/B.

The bounded reverse maps (`701 - worker`) restored the selected resources for
both M128 instantiations:

- registers: 56/thread;
- stack: 32 bytes;
- static shared memory: 2,048 bytes;
- local memory: 0 bytes;
- nine resident 128-thread CTAs/SM.

Candidate source hashes were
`fa1bc11921325abc73f40c8ed2328d7d5b0a89d68ae1abed827728d3eb84f2a3`
for the W13 reverse test and
`95219adae35c9b7e674f0ff02bb1e60e4f96bdbfaefc85be34d26eed677d911e`
for the W2 reverse test.

## Correctness

Single-GPU random-route M128 checks passed bitwise for both reverse maps:
cosine 1.0, relative L2 0.0, finite output, 1,992 padded rows, split-K2, and
packed barrier generations `[2048, 2048, 2048, 2048]`.

Every distributed control/candidate run also passed all-rank reference,
finite, and allreduce checks with cosine 0.9999956 and relative L2 0.0029673.

## TP4 cold-L2 screens

Physical GPUs 0,5,6,7; M128 random route seed 20260902; CUDA Graph; two
batches of 20 replay-interleaved samples per implementation and four warmups.
Every replay had a separate 256 MiB L2 clear immediately before it, excluded
from event timing.  Ratios use each process's same-source multi-kernel control.

| map | multi median (ms) | one median (ms) | one / multi |
|---|---:|---:|---:|
| selected W13 map | 0.305232 | 0.350016 | 1.146721 |
| W13 reverse | 0.304640 | 0.349968 | 1.148792 |
| selected W2 map, exact flag-off source | 0.305056 | 0.349392 | 1.145337 |
| W2 reverse | 0.304448 | 0.348800 | 1.145680 |

After normalization, W13 reverse is 0.181% worse and W2 reverse is 0.030%
worse.  The small raw-latency changes track denominator drift and do not meet
the predeclared approximately 0.5% expansion threshold.

## Verdict

Reject both reverse maps and restore the exact selected source/benchmark
SHA-256 values `7ac22134...` and `e39ab6a4...`.  The fresh-CTA advantage is
completion-driven rather than an arbitrary static permutation.  The selected
cross-wave rotations already capture the useful static part; do not sweep
more complete-wave affine maps.

Raw evidence:

- `bench/results/iter677_w13_wave_affine5_m128_jit_correctness_20260906.log`
- `bench/results/iter677b_w13_wave_affine5_constmod_m128_jit_correctness_20260906.log`
- `bench/results/iter677c_w13_wave_affine5_constmod_rebuild_correctness_20260906.log`
- `bench/results/iter677d_w13_wave_reverse_m128_jit_correctness_20260906.log`
- `bench/results/iter677_w13_wave_reverse_{on,off}_a_tp4_m128_cold_20260906.log`
- `bench/results/iter677e_w2_wave_reverse_m128_jit_correctness_20260906.log`
- `bench/results/iter677_w2_wave_reverse_{on,off}_a_tp4_m128_cold_20260906.log`
