# Iteration 709s — force paired W2 source operands live at issue

## Hypothesis

Iteration 709r proved that ptxas delayed the manually spilled second-group
load until after the first QGMMA wait, so the intended source-level pair never
existed in SASS.  CuTe's `warpgroup_fence_operand(uint32_t&)` is an empty
read/write inline-assembly constraint with a memory clobber.  Applying it to
all materialized FP8 source registers at the issue boundary should prevent
that legal load/decode motion without adding a hardware instruction.

## Source change

- Add opt-in `V4_SINGLE_LAUNCH_W2_PAIR_OPERAND_FENCE=1`, requiring the exact
  predecode + paired issue + one-value shared-spill diagnostics.
- Immediately before the adjacent QGMMA issue loop, apply the CuTe compiler
  operand fence to `.x/.y` of both FP8 `uint2` values for both N64 groups.
- QGMMA, commit, hardware WGMMA fence/wait, math, task mapping, reduction and
  communication remain unchanged.
- Default TP4/TP8 paths remain disabled and the flag enters the JIT identity.

## Pre-CUDA gate

- Python bytecode compilation: PASS.
- Staged SHA256:
  `f8890202e7cdedc8c372b0515a1ac6a8dfc202e65f931ac4d4c648ba998cd1ae`.
- No CUDA build, launch, correctness, or timing claim yet.

## Required static gate

Require REG56/no fixed local spill/nine-CTA admission, 64 QGMMAs, and a
dependency-barrier count reduced from 64 toward 32.  Also inspect the first
pair to prove the spilled `LDS` now precedes both QGMMAs.  Reject before any
business-kernel launch otherwise.
