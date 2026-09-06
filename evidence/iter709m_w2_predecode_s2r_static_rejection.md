# Iteration 709m — reject W2 S2R predecode at the static gate

## Exact build

- Host: H20-GPU-06 container `megamoe`; compilation visibility restricted to
  physical H20 GPU1.
- Source: Iteration 709l SHA256
  `55c4cb0d114f21f129f61d23a1e284978a09fdf81585b1c16cf51b3bcb806f7a`.
- Flags: `V4_SINGLE_LAUNCH_TP4=1` and
  `V4_SINGLE_LAUNCH_W2_PREDECODE_S2R=1`.
- Extension: `v4tp_a586be89b63a5cb2a66c_v178mspec`.
- Target: exact `tp4_megamoe_single_launch_kernel<2,128>` symbol.

## Result

JIT compilation and link pass.  The target entry remains `REG56 STACK32
SHARED2048 LOCAL0`, preserving the nine-CTA/SM register tier without fixed
local-memory spill.

The extracted target SASS has SHA256
`0db68b78d4e937d88bfdabda358cb2c5b386877f29fb66e14cfa28ba558d5500`.
Across the two emitted runtime W2 paths it contains 64 QGMMA instructions and
64 `WARPGROUP.DEPBAR.LE` instructions.  Inspection shows every W2 QGMMA still
followed by a dependency barrier.  The 1:1 ratio is unchanged from the fused
path and does not approach standalone W2's measured 2:1 QGMMA-to-barrier
ratio.

## Decision

**REJECT before correctness or timing.**  The candidate saves no compiler
dependency barriers despite retaining the resource envelope, so it fails its
declared structural gate.  No CUDA business kernel was launched and there is
no cold-L2 latency claim.  Keep the diagnostic path opt-in/default-off for
reproducibility; the next experiment must directly change accumulator or
WGMMA issue lifetime instead of only shortening weight-lookahead state.

Raw generated artifacts on the host:

- `bench/results/iter709m_w2_predecode_s2r_jit.log`
- `bench/results/iter709m_w2_predecode_s2r_resources.log`
- `bench/results/iter709m_w2_predecode_s2r_m128_split2.sass`
