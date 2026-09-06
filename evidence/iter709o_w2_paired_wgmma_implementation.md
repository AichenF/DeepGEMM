# Iteration 709o — explicit paired W2 QGMMA issue

## Hypothesis

Standalone W2 retains distinct register-source and accumulator bases for two
N64 QGMMAs, issues both, then executes one dependency wait.  Fused REG56
aliases those lifetimes and waits after each instruction.  The new candidate
forces both current FP8 operand pairs to remain explicit through a separate
issue loop, so ptxas cannot satisfy the C++ dataflow by immediately recycling
the first pair before the second QGMMA.

## Source change

- Add opt-in `V4_SINGLE_LAUNCH_W2_PAIR_WGMMA_GROUPS=1`, requiring the prior
  predecoded-S2R diagnostic and therefore inheriting its strict TP4 M128,
  bound-9, flat inline FP32 W2 isolation.
- Materialize `current_fp8_0/1[2]` while preserving the selected shared loads,
  LUT synthesis and decoded-next-step lookahead.
- Issue the two N64 QGMMAs in a second adjacent unrolled loop; commit, operand
  fences and wait remain in their original position after both instructions.
- Keep both diagnostics default-off and include the paired flag in the JIT
  identity/compiler definition.  Production TP4 and TP8 remain unchanged.

## Pre-CUDA gate

- Python bytecode compilation: PASS.
- Staged source SHA256:
  `b5a7d47f7497fdbdaabfe3af2fbf208b3633d9200a7e539f461a8a3cef261450`.
- No CUDA compilation, launch, correctness, or timing result exists yet.

## Required static gate

The exact M128/split-K2 entry must stay spill-free at REG56 with nine-CTA/SM
admission, keep 64 QGMMAs across its two emitted W2 paths, and reduce static
`WARPGROUP.DEPBAR.LE` from 64 toward 32.  Reject before timing otherwise.
