# Iteration 718a: pair the two FP16 W2 N64 WGMMA groups in one asm block

## Why this composition is still open

The selected monolithic M128 entry remains at REG56/STACK32 but ptxas emits
32 W2 QGMMAs with 32 dependency barriers.  Standalone W2 and natural-register
RDC both emit the preferable 32-QGMMA/16-DEPBAR schedule, but RDC requires a
REG198 callable window even after the argument ABI is compacted.

The earlier paired-inline probes used FP32 WGMMA accumulators.  The already
qualified opt-in FP16 W2 accumulator cuts each N64 group's destination state
from four 32-bit registers to two.  It has not been composed with a single
early-clobber asm block that makes both groups' source and destination ranges
simultaneously visible to ptxas.

## Isolated change

- Add `V4_SINGLE_LAUNCH_W2_F16_PAIR_INLINE_ASM=1`, default off.
- Require the existing `V4_SINGLE_LAUNCH_W2_F16_WGMMA_ACCUM=1` numerical
  tradeoff and the selected TP4 schedule-0, compact-W13, M128-bound9 path.
- For M128 W2 only, retain both groups' four FP8 source words, fence those
  operands, and issue the two existing `m64n8k32.f16.e4m3.e4m3` operations in
  one asm block with early-clobber packed-F16 destinations.
- Keep task ownership, K-loop order, TMA pipeline, commit/wait placement,
  output conversion, k6 combine and terminal all-reduce unchanged.  M8-M64
  do not select the paired issue branch.
- Report both FP16 switches in benchmark metadata and include the new switch
  in the JIT identity and compile definitions.

Candidate source hashes before any JIT:

- `v4_flash_tp_wgmma.py`:
  `9aadb434253d13258ce0b40b022e29b8c98e66493d721ae3580eeb34877e070e`
- `bench/v4_flash_tp_single_vs_multi_graph.py`:
  `75448c6d9361e1317bfa97e74fa1c40119085fdf839dd7bdab650b376a327dd2`

## Predeclared gates

1. Python syntax and a fresh M128 JIT must succeed.
2. The exact M128 split-K2 entry must remain at or below REG64, no fixed local
   allocation, and no material stack growth beyond the selected STACK32.
3. The exact W2 instruction interval must contain 32 FP16 QGMMAs and no more
   than 20 dependency barriers; the target is 16.  Any `STL`/`LDL` spill
   traffic rejects the candidate before CUDA execution.
4. If static gates pass, local M128 output must match the existing FP16
   candidate within its already measured envelope (cosine >= 0.999999 and
   relative L2 <= 0.0012 versus the FP32 multi-kernel reference).
5. Only then measure alternating cold-L2 phase windows and TP4 graph timing.
   Promotion still requires improvement outside noise; the final project
   target remains at least 1.10x over the same cold-L2 multi-kernel baseline.

No CUDA launch, correctness result, timing result, or speedup is claimed by
this composition checkpoint.
