# Iteration 709q — one-`uint2` manual shared spill for paired W2

## Hypothesis

The paired/predecoded schedule should need approximately the standalone W2
register set minus four registers, placing it close to the fused REG56 cliff.
Ptxas still aliases source and accumulator bases.  This probe explicitly
moves only one two-register next-step FP8 value from the second N64 group to
shared memory, attempting to create enough headroom for two in-flight QGMMAs
without discarding the selected packed-weight/LUT/decode lookahead.

## Source change

- Add opt-in `V4_SINGLE_LAUNCH_W2_PAIR_SPILL_ONE=1`, requiring both the
  predecoded-S2R and paired-QGMMA diagnostic flags.
- Append a 1,024-byte, structure-of-arrays shared stage: two 128-thread
  `uint32_t` planes.  The second group's `next_fp8_1` is stored there after
  decode and loaded by the same thread on the next K32 step.
- Use explicit shared PTX loads/stores with a memory clobber so the compiler
  cannot silently retain the value as an ordinary C++ live range.
- Do not add a CTA barrier: each location is private to one thread, the load
  occurs in the following loop iteration after the existing WGMMA wait, and
  no cross-thread publication is involved.
- The candidate remains strictly default-off and M128/TP4-only through its
  prerequisite validators.  TP8 and production launch sizes are unchanged.

## Pre-CUDA gate

- Python bytecode compilation: PASS.
- Staged source SHA256:
  `8e9456ed51b74f6d89807fa0e39c75621403827c5c2e8b43498ca07997d69c1a`.
- No CUDA compilation, correctness, or latency result yet.

## Required static gate

Require the exact target to retain REG56/no fixed local spill/nine-CTA
admission with total dynamic+static shared memory fitting nine H20 CTAs.  It
must keep 64 QGMMAs and reduce 64 static dependency barriers toward 32 before
any business-kernel launch.
