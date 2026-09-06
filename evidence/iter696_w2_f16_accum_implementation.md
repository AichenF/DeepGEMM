# Iteration 696: isolated W2 packed-FP16 WGMMA accumulator probe

The exact selected production source
`7ac22134c953d17c8dea9310011818ca483b5b8a96b06324381abd2c8c9c3f45`
was extended with a default-off
`V4_SINGLE_LAUNCH_W2_F16_WGMMA_ACCUM` specialization.  Candidate source
SHA-256 is
`973ba670d1b64c3c27c4e97b5f8a138617b36a48a1409f60be488a9b418541d0`.

The specialization changes only the selected flat inline TP4 one-kernel W2
template instance.  Each K32 FP8 WGMMA writes two packed FP16 registers
instead of four FP32 registers; after the four K32 operations for one K128
activation-scale group, both packed registers are promoted into the existing
four FP32 cross-group accumulators.  Standalone/multi W2 retains its original
FP32 WGMMA specialization, allowing same-build arithmetic A/B comparison.

Python compilation succeeds and the source has no whitespace errors.  This
record contains no CUDA result.  Required next gates are fresh SM90a JIT,
linked-cubin resource inspection, complete random-route output comparison,
then excluded-256-MiB-clear cold-L2 timing only if correctness passes.
