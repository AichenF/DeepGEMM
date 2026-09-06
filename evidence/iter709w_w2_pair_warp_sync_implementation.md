# Iteration 709w — real warp scheduling boundary before paired W2

## Hypothesis

Ptxas moved the second operand group's shared load and decode through all
compiler-only constraints, including a single volatile PTX block.  A real
`bar.warp.sync` after preparation is a machine-level control/memory boundary:
the load cannot remain after that point, while the following paired PTX block
still exposes distinct early-clobber accumulators and sources.

## Source change

- Add opt-in `V4_SINGLE_LAUNCH_W2_PAIR_WARP_SYNC=1`, requiring the full
  paired-inline-asm diagnostic chain.
- Emit one `bar.warp.sync 0xffffffff` after all four source values for both
  N64 groups are prepared/fenced and immediately before the two-QGMMA block.
- The synchronization is warp-local because every value is thread-private;
  no cross-warp data exchange is required.  It adds one instruction per warp
  per K32 step and intentionally trades this cost for one fewer WGMMA
  dependency wait per pair.
- Default TP4/TP8 paths remain unchanged and the flag enters the JIT identity.

## Pre-CUDA gate

- Python bytecode compilation: PASS.
- Staged SHA256:
  `f3d1f5debbc07089fed0a1ea519a1aa1f8ee8579b2603f862097d72df44c8175`.
- No CUDA build, business-kernel launch, correctness, or timing claim yet.

## Required static gate

Require REG56/no fixed local spill/nine-CTA admission, 64 QGMMAs, about 32
dependency waits, and direct SASS proof that the spilled LDS precedes the
warp barrier and both QGMMAs.  Reject before launch otherwise.
