# Iteration 659: M128 complete-wave W13 ownership rotation

## Hypothesis and mapping

Iteration 653b measured all 702 resident M128 CTAs and showed that consecutive
78-block launch waves place one CTA on each of the H20's 78 SMs.  In the fused
persistent loop the same physical CTA/SM owns a fixed logical stripe in every
W13 wave, unlike standalone W13 where fresh blocks are assigned as prior
blocks retire.  That fixed mapping can repeatedly couple the same cold expert
weight offsets to the same SM/L2 path.

The default-off `V4_SINGLE_LAUNCH_W13_WAVE_ROTATE=13` path keeps each of the
five complete 702-task W13 waves globally contiguous and complete, but maps
wave `q` slot `cta` to `(cta + 13*q) mod 702`.  Against the measured CTA-to-SM
map, every complete wave is assigned to a different physical SM.  The 474-task
residual wave remains byte-for-byte on the production mapping, preserving its
already optimal maximum of seven residual CTAs/SM.  Arithmetic, task count,
task sequence within each CTA, split-K, phase barriers, W2 and communication
are unchanged.

## Static and correctness gates

- Fresh SM90a extension:
  `v4tp_8f28fe62bec8ef6d07d3_v178mspec`.
- M128 split-K2/4 resources remain
  `REG56 STACK32 SHARED2048 LOCAL0`, retaining exactly nine CTAs/SM.
- Random-route M128, seed 20260902, is bitwise equal to the independent
  same-source multi-kernel local output (`cosine=1`, `rel_l2=0`, finite), with
  1,992 padded rows.
- Packed barrier generations seeded at `2^22-1` return exactly `[0,0,0,0]`.

## Local cold-L2 phase screen

Physical H20 GPU1, eight separate process samples ordered
OFF/ON/ON/OFF/OFF/ON/ON/OFF.  Every checked launch had its own excluded
256 MiB L2 clear.  Public X was prequantized FP8-E4M3 plus FP32 group-128
scales; weights were MXFP4; TP communication was disabled only for phase
isolation.

W13 microseconds:

- OFF: `216.832, 214.368, 214.016, 215.040`
- ON: `211.040, 210.720, 211.008, 210.976`

Every ON sample beats every OFF sample.  Median W13 improves from
`214.704` to `210.992 us` (1.73%); mean improves from `215.064` to
`210.936 us` (1.92%).

The sum of route, W13, requant and W2 improves from median
`331.952` to `328.816 us` (0.94%), and from mean `332.296` to
`328.736 us` (1.07%).

## TP4 cold-L2 result

Physical GPUs 0/5/6/7, CUDA Graph, random route seed 20260902, four warmups
and two outer batches x 20 replay-interleaved samples per process.  Every
single and multi replay had a separate excluded 256 MiB clear.  Four process
windows provided 80 samples per mode.  All ranks were finite/equal and both
the single candidate and same-source multi control reported the identical
accepted result (`cosine=0.9999955977`, `rel_l2=0.0029672640`, allreduce
true).

| window | rotate | multi (ms) | single (ms) | single/multi |
|---|---:|---:|---:|---:|
| ON-A | 13 | 0.304768 | 0.351600 | 1.153664 |
| OFF-A | 0 | 0.305312 | 0.353920 | 1.159208 |
| OFF-B | 0 | 0.306032 | 0.354752 | 1.159199 |
| ON-B | 13 | 0.305120 | 0.351296 | 1.151337 |

Average normalized single/multi ratio improves from `1.159203` to
`1.152501`, a repeatable 0.58% end-to-end gain.  Direct single-kernel average
improves from `0.354336` to `0.351448 ms`, or 0.82%.

## Decision

Retain as a default-off positive M128 option and extend the same complete-wave
ownership test independently to W2 before selecting a default or running all
M.  The gain is real but much smaller than the remaining objective; selecting
it alone would not materially change the five-shape verdict.

Raw artifacts:

- `bench/results/iter659_w13_wave_rotate13_{jit,resources}_20260906.log`
- `bench/results/iter659_w13_wave_rotate13_m128_correctness_20260906.log`
- `bench/results/iter659_w13_wave_rotate*_m128_phase*_20260906.log`
- `bench/results/iter659_w13_wave_rotate*_tp4_m128_cold_20260906.log`
