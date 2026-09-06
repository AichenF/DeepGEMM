# Iteration 674: reject compact grid-constant kernel ABI

Date: 2026-09-06

## Hypothesis

The standalone multi-kernel W13 phase enters through a substantially smaller
argument surface than the monolithic TP4 kernel.  Bundle all TP4 launch
parameters (four tensor maps, pointers, and collective scalars) into one
`__grid_constant__` POD so ptxas can load phase-local fields on demand.  The
acceptance gate was a reduction of the selected M128 kernel from 56 registers
to at most 48 registers, without local spills, so that a later 10-CTA/SM test
would be meaningful.

## Candidate

- Candidate source SHA256:
  `e2c821b166b5a2ff2847c4fb1692d3f94c5d3153dc9c7521b0c32086cd388505`
- JIT extension:
  `/tmp/torch_ext_v4_tp/v4tp_73c7a23e4968b173281c_v178mspec/v4tp_73c7a23e4968b173281c_v178mspec.so`
- New experiment flag: `V4_SINGLE_LAUNCH_COMPACT_KERNEL_ABI=1`
- TP8 code was unchanged.

The first compile exposed a macro-alias collision with members such as
`w13_phase_args.activation`.  Replacing the macros with typed function-local
aliases compiled successfully.

## Correctness smoke gate

One-GPU, M128, random routing, packed-generation wrap test:

- `accepted=true`
- down output cosine `1.0`, relative L2 `0.0`, finite `true`
- packed generation words after wrap `[0, 0, 0, 0]`
- H20 SM count reported as 78

This smoke test deliberately disabled the TP collective.  No TP4 timing was
run because the resource gate failed.

## Resource gate

`cuobjdump --dump-resource-usage` for
`tp4_megamoe_single_launch_kernel<2, 128>`:

- candidate: `REG:56 STACK:32 SHARED:2048 LOCAL:0 CONSTANT[0]:1408`
- selected production: `REG:56 STACK:32 LOCAL:0`

The SplitK=4 M128 specialization was also `REG:56 STACK:32 LOCAL:0`.
M8/M16/M32 specializations used 63 registers and M64 used 64, so the packed
entry did not improve any target specialization.

## Decision

Rejected before timing.  Packing the parameter ABI changes how parameters are
addressed, but ptxas still materializes the same live state inside the fused
control flow.  It does not transfer the standalone kernel's 47-register phase
boundary into a monolithic kernel.  The exact selected production source and
benchmark SHAs were restored:

- source: `7ac22134c953d17c8dea9310011818ca483b5b8a96b06324381abd2c8c9c3f45`
- benchmark: `e39ab6a4b9727f16688668be39372825ebc781ce7320a08e149f2087aa1bba97`

