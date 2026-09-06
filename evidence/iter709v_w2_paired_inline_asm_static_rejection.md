# Iteration 709v — ptxas splits an early-clobber paired PTX block

## Exact build/result

- Source SHA256:
  `940f76e0df36849e77769a40e18bf57f1be0a1e7e9d1f56de86bf375f47250e2`.
- Extension: `v4tp_0c965391348888f4e713_v178mspec`.
- Exact M128/split-K2 entry: `REG56 STACK32 SHARED2048 LOCAL0`, plus the
  candidate 19,456-byte dynamic shared allocation.
- SASS SHA256:
  `5fdd85099b22d766bf4d8a1eb9e6b2a9c27f2d4fd7d90cb164d794cd3f1e9ad0`.
- Static result: 64 QGMMAs and 64 dependency barriers across the two W2
  runtime paths.

NVCC accepts `+&f` device-asm constraints and the candidate path does obtain
distinct physical accumulator/source bases (`R44/R32` and `R40/R36`).  Ptxas
nevertheless schedules independent decode and shared-load instructions
between the two QGMMAs and inserts a wait after each.  At `0x5430` it issues
the first QGMMA, waits at `0x5440`, loads the spilled value at `0x5450`, then
issues the second at `0x5470`.  A single volatile PTX block is not an opaque
SASS scheduling region.

## Decision

**REJECT before business-kernel launch/timing.**  The next bounded probe adds
a real per-warp synchronization instruction after operand preparation and
before the paired asm.  Unlike a compiler fence, this is a machine scheduling
boundary; retain only if the extra warp barrier forces both operand groups
ready and reduces dependency waits by approximately half.

Raw artifacts:

- `bench/results/iter709v_w2_paired_inline_asm_jit.log`
- `bench/results/iter709v_w2_paired_inline_asm_resources.log`
- `bench/results/iter709v_w2_paired_inline_asm_m128_split2.sass`
