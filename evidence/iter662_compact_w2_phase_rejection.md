# Iteration 662: compact M128 W2 phase ABI rejected

## Hypothesis and implementation

The multi-kernel transfer audit identified one remaining bounded adaptation:
repeat the selected compact-W13 call technique for the whole W2 phase.  The
prototype reused W13's 104-byte CTA-shared pointer record after activation,
then passed one pointer to an M128-only outlined W2 grid-stride loop.  The
callee derived routes, task count and CTA/grid coordinates from the shape and
device state.  M32 and smaller shapes stayed on their existing inline W2 path.

This differs materially from the rejected Iterations 375--378 outline, whose
device call passed thirteen pointers plus grid/task scalars.  Arithmetic,
task order, TMA descriptors, MXFP4 decode, WGMMA, barriers and collective
were unchanged.

## Static and correctness gates

- Fresh SM90a extension:
  `v4tp_2ecd2d14b81d23c3fae4_v178mspec`.
- M128 split-K2/4 remained `REG56 STACK32 SHARED2048 LOCAL0`, preserving
  nine CTAs/SM.  M8/M16/M32 stayed `REG63 STACK32`, and M64 stayed
  `REG64 STACK32`; no lower-M resource contract changed.
- Random-route M128, seed 20260902, was bitwise equal to the independent
  same-source multi local output (`cosine=1`, `rel_l2=0`, finite), with 1,992
  padded rows.
- Packed generations seeded at `2^22-1` wrapped exactly to `[0,0,0,0]`.
  All eight timed-process outputs were also bitwise equal locally.

## Local cold-L2 phase screen

Physical H20 GPU1, eight independent process samples ordered
OFF/ON/ON/OFF/OFF/ON/ON/OFF.  Every launch used a separate excluded 256 MiB
L2 clear.  X was prequantized FP8-E4M3 plus FP32 group-128 scales; weights
were MXFP4; TP communication was disabled only for phase isolation.

W2 microseconds:

- inline OFF: `108.384, 106.336, 106.752, 107.008`
- compact outline ON: `108.704, 109.696, 108.928, 108.064`

Median regresses from `106.880` to `108.816 us` (1.81% slower); mean regresses
from `107.120` to `108.848 us` (1.61% slower).  The complete four-phase sum
also regresses: median `331.568 -> 334.256 us` (0.81%) and mean
`332.024 -> 333.848 us` (0.55%).

## Decision

Reject before TP4 and remove the complete prototype.  Compressing the ABI
eliminates neither the per-CTA call cost nor the persistent grid-stride W2
penalty.  Because the isolated phase is already robustly slower, distributed
testing cannot rescue it; it also cannot justify revisiting the old long-
replay numerical instability.  Production source returns byte-identical to
Iteration 661.

Raw artifacts:

- `bench/results/iter662_compact_w2_phase_jit_20260906.log`
- `bench/results/iter662_compact_w2_phase_resources_full_20260906.log`
- `bench/results/iter662_compact_w2_phase_m128_correctness_20260906.log`
- `bench/results/iter662_compact_w2_phase_m128_phase_window_20260906.log`
