# Iteration 724a — qualify a terminal no-return device boundary

## Reassessment

Three consecutive phase-local W2 experiments failed: compact ping-pong W13,
vector-atomic W2 combine, and compact W2 call composed with bound-eight.  The
last result closes ordinary return-capable device calls: the W2 callee remains
32 QGMMAs / 32 dependency waits even at the register tier where standalone W2
is 32 / 16.

CUDA's documented `[[noreturn]]` support and PTX `.func ... .noreturn`
directive expose a materially different ABI boundary that the earlier
Iteration-710 discussion did not compile.  A terminal function need not
preserve a continuation in its caller.  This matches the actual single-kernel
phase graph: after W2, only the ordered k6 TP collective and cleanup remain.

## First hard gate

Before changing the business kernel, compile a minimal CUDA 12.8 probe with
matched returning and terminal device functions.  Require all of the
following:

1. NVCC accepts `[[noreturn]]` in device code.
2. Generated PTX marks the terminal function `.noreturn`.
3. Generated SASS has a call into the terminal function and terminates there,
   rather than synthesizing a normal return continuation.

If this compiler-mechanics gate fails, reject without touching the business
kernel.  If it passes, add a default-off M128-only static probe that outlines
the complete W2 phase into a terminal no-return callee and exits after W2.
That second probe is compile/SASS-only: it must recover materially fewer than
32 W2 dependency waits before collective migration or any CUDA business
execution is allowed.

## Sources consulted

- NVIDIA CUDA C++ Programming Guide, Noreturn Annotation:
  <https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#noreturn-annotation>
- NVIDIA PTX ISA, `.func` and `.noreturn`:
  <https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#performance-tuning-directives-noreturn>
- NVIDIA PTX ISA, `setmaxnreg`:
  <https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#miscellaneous-instructions-setmaxnreg>

The same PTX documentation also rules out an SM-wide register donor: the
register pool used by `setmaxnreg` is per CTA, not per SM.
