# Iteration 661: M128 W13 wave rotation does not transfer to M32/M64

## Hypothesis and implementation

Iteration 659's M128 gain came from rotating physical CTA ownership by 13
slots between complete W13 waves while preserving each wave's exact logical
task set.  This experiment inserted the same transform into the production
inline W13 loop for M32 and M64.  The residual wave, task count, arithmetic,
split-K, barriers, activation, W2 and communication were unchanged.

For random routes at seed 20260902:

- M32: 1,120 padded rows, 140 mblocks, split-K4, 4,480 W13 tasks over
  624 resident CTAs (seven complete waves plus a 112-task residual).
- M64: 1,624 padded rows, 203 mblocks, split-K2, 3,248 W13 tasks over
  624 resident CTAs (five complete waves plus a 128-task residual).

## Static and correctness gates

- The existing rotate-13 extension identity was rebuilt from the changed
  source: `v4tp_8f28fe62bec8ef6d07d3_v178mspec`.
- Resources were unchanged: M32 split-K4 stayed at
  `REG63 STACK32 SHARED2048 LOCAL0`; M64 split-K2 stayed at
  `REG64 STACK32 SHARED2048 LOCAL0`.  Both retain eight CTAs/SM.
- Random-route M32 and M64 were bitwise equal to the independent same-source
  multi-kernel local output (`rel_l2=0`, finite; cosine was 1 within printed
  floating precision).
- Packed barrier generations seeded at `2^22-1` returned `[0,0,0,0]` for
  both shapes.

## Local cold-L2 phase screen

Physical H20 GPU1, eight independent process samples per shape ordered
OFF/ON/ON/OFF/OFF/ON/ON/OFF.  Each launch used a separate excluded 256 MiB
L2 clear.  Public X was prequantized FP8-E4M3 plus FP32 group-128 scales;
weights were MXFP4; communication was disabled only for phase isolation.

M32 W13 microseconds:

- OFF: `120.320, 121.888, 122.432, 120.288`
- ON: `124.064, 124.576, 124.672, 122.944`
- median `121.104 -> 124.320 us` (2.66% slower); mean
  `121.232 -> 124.064 us` (2.34% slower).

M64 W13 microseconds:

- OFF: `171.232, 170.400, 171.424, 171.936`
- ON: `173.632, 172.384, 171.808, 173.312`
- median `171.328 -> 172.848 us` (0.89% slower); mean
  `171.248 -> 172.784 us` (0.90% slower).

## Decision

Reject before TP4 and remove the M32/M64 mapping.  The M128 result is not a
generic benefit from scrambling persistent ownership: it depends on M128's
distinct 702-CTA, nine-CTA/SM residency and residual-wave geometry.  Source
returns byte-identical to Iteration 660 and preserves only the independently
validated M128 specialization behind the default-off flag.

Raw artifacts:

- `bench/results/iter661_w13_wave_rotate_m32_m64_jit_20260906.log`
- `bench/results/iter661_w13_wave_rotate_m32_m64_resources_20260906.log`
- `bench/results/iter661_w13_wave_rotate_m32_m64_correctness_20260906.log`
- `bench/results/iter661_w13_wave_rotate_m32_m64_phase_window{1,2}_20260906.log`
