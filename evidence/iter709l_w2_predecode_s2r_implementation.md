# Iteration 709l — predecode fused W2 S2R lookahead

## Hypothesis

The aligned NCU/SASS audit found identical W2 QGMMA work but twice as many
`WARPGROUP.DEPBAR.LE` instructions in the fused entry as in standalone W2.
The fused entry must stay at 56 registers/thread for nine-CTA admission.  The
candidate therefore shortens the live lookahead state without changing GEMM
work: each next packed MXFP4 word and synthesized LUT pair is immediately
decoded to two FP8 `uint2` values before the current QGMMA, instead of carrying
the packed words plus LUTs across the instruction.

## Source change

- Add opt-in `V4_SINGLE_LAUNCH_W2_PREDECODE_S2R=1` and include it in the JIT
  cache key and compiler definitions.
- Restrict it to the selected TP4 M128, bound-9, schedule-0, inline, flat
  WOUT128 FP32 W2 path with compact interleaved scales and two stages.
- Preserve the selected shared-memory loads, scale/LUT synthesis, FP8 decode,
  QGMMA count, FP32 accumulation, task mapping, reduction and communication.
- Keep the feature off by default, so production TP4 and TP8 behavior is
  unchanged until it passes every gate.

## Pre-CUDA gate

- `python3 -m py_compile v4_flash_tp_wgmma.py`: PASS on the staged source.
- Staged source SHA256:
  `55c4cb0d114f21f129f61d23a1e284978a09fdf81585b1c16cf51b3bcb806f7a`.
- No CUDA kernel has been compiled or launched for this candidate yet; no
  correctness or performance claim is made.

## Required next gate

Compile the exact opt-in entry and inspect the cubin before timing.  Continue
only if the entry remains spill-free at 56 registers/thread, admits nine
CTAs/SM, and reduces the fused W2 dependency-barrier count from 32 toward the
standalone value of 16.  Otherwise reject without consuming benchmark noise.
