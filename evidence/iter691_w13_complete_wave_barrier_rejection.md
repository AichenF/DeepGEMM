# Iteration 691: complete-wave W13 barrier rejection

## Question

Can the useful part of the standalone multi-kernel launch schedule be
transferred into the one-launch M128 W13 phase by forcing its five complete
702-task waves to begin in lockstep?

The candidate retained the selected 702 resident CTAs, 128 threads/CTA,
N128 tile, split-K2 arithmetic, complete-wave rotation 13, residual-wave
mapping, activation, W2, and communication. It added a reusable packed grid
rendezvous only between complete W13 waves 0/1, 1/2, 2/3, and 3/4. No
rendezvous was added before the 474-task residual wave.

## Gates

- Candidate source SHA-256:
  `4cfd66777ab01c7b75d88f42491dfe2bab3db4fce74902183eed37c87202b42a`.
- Candidate extension:
  `v4tp_808f72d4f858e2e559ec_v178mspec`.
- Cubin resources stayed `REG56 STACK32 SHARED2048 LOCAL0`, identical to the
  selected M128 one-kernel entry.
- Five independently cold single-GPU runs used physical H20 GPU 0, random
  routes with seed 20260902, prequantized FP8-E4M3 X plus group-128 scale,
  and a separate excluded 256 MiB L2 clear immediately before the measured
  kernel.
- Every run was bitwise equal to the independent same-source multi-kernel
  local output: cosine 1.0, relative L2 0.0, finite, with 1,992 padded rows.
- The packed phase-1 word advanced to 10,240 rather than 2,048, confirming
  that the four additional grid-barrier generations actually executed.

## Result

Candidate W13 min/median/max was
`231.424/235.328/235.488 us`; its four-phase sum was
`347.264/351.040/351.680 us`.

Four adjacent exact-production anchors used the same GPU, routes, phase
profiler, and cold-L2 protocol. Production W13 min/median/max was
`208.768/208.992/210.336 us`; its four-phase sum was
`325.472/325.952/327.488 us`.

The candidate therefore regressed median W13 by 12.60% and the complete
local phase sum by 7.70%, while leaving the resource contract unchanged.
The synchronization cost and forced straggler coupling are larger than any
benefit from wave-order alignment. This fails the local 1% gate decisively,
so no distributed TP4 endpoint run is justified.

## Decision

Reject the complete-wave barrier and restore the exact selected source
SHA-256
`7ac22134c953d17c8dea9310011818ca483b5b8a96b06324381abd2c8c9c3f45`.
The standalone advantage is not reproducible by adding phase boundaries
inside the resident grid. Together with the previous Git/body/SASS audits,
this leaves no literal multi-only code or profitable launch-scheduling
snippet to cherry-pick; a material gain requires a different fused dataflow.

The first two attempts were launcher-only failures (direct bench-script
import path, then missing Humming `PYTHONPATH`) and produced no CUDA result.

Raw artifacts:

- `bench/results/iter691_w13_wave_barrier_m128_jit_correctness.log`
- `bench/results/iter691b_w13_wave_barrier_m128_jit_correctness.log`
- `bench/results/iter691c_w13_wave_barrier_m128_jit_correctness.log`
- `bench/results/iter691d_w13_wave_barrier_m128_phase_repeat.log`
- `bench/results/iter691e_w13_wave_barrier_m128_resources.log`
- `bench/results/iter691f_production_m128_phase_repeat.log`
