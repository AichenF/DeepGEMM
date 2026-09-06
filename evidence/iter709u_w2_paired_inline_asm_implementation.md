# Iteration 709u — two W2 QGMMAs in one early-clobber PTX block

## Hypothesis

Separate CuTe calls let NVCC/ptxas overlap accumulator and register-source
allocations, then repair the hazard with an immediate dependency wait.  One
inline-PTX block can expose both instructions as a single compiler operation.
Marking all eight FP32 accumulators read/write and early-clobber explicitly
forbids overlap with the eight FP8 register inputs.

## Source change

- Add opt-in `V4_SINGLE_LAUNCH_W2_PAIR_INLINE_ASM=1`, requiring the exact
  fenced, one-value-spill, predecoded paired diagnostic chain.
- Emit two `wgmma.mma_async.sync.aligned.m64n8k32.f32.e4m3.e4m3`
  instructions in one volatile PTX block, sharing the activation descriptor
  and scale predicate.
- Preserve group order, accumulator values, ScaleOut/ScaleIn=One, the existing
  surrounding WGMMA fence/commit/wait, FP32 epilogue, task mapping, reduction
  and communication.
- Use `+&f` on every accumulator to prevent any source register from sharing
  an output allocation.  The experiment remains default-off and TP4-M128
  only; TP8 and production are unchanged.

## Pre-CUDA gate

- Python bytecode compilation: PASS.
- Staged SHA256:
  `940f76e0df36849e77769a40e18bf57f1be0a1e7e9d1f56de86bf375f47250e2`.
- NVCC support for the early-clobber device-asm constraint is not assumed;
  a compiler rejection will be recorded as such.
- No CUDA launch, correctness, or timing result yet.

## Required static gate

Require build success, REG56/no fixed local spill/nine-CTA admission, 64
QGMMAs and dependency barriers reduced from 64 toward 32.  Reject before a
business-kernel launch otherwise.
