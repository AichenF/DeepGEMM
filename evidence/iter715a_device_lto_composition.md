# Iteration 715a: unique device-LTO composition

## Hypothesis

The uniquely attributed RDC experiment recovered standalone W2's 32-QGMMA / 16-DEPBAR schedule, but its retained device call forced the M128 entry to 195 registers and a 112-byte stack per thread and was runtime-neutral/slower than the matched ordinary entry. CUDA device link-time optimization is a distinct, previously untested boundary: compile the same RDC source with `-dlto` at both compile and device-link stages, allowing nvlink to inline or specialize the phase while starting from independently optimized device IR.

The static success gate is deliberately strict: the uniquely named TP4 M128 split-K2 entry must avoid fixed local allocation and materially reduce the RDC entry's 195-register/112-byte-stack footprint, while the W2 interval must retain approximately 32 QGMMAs / 16 dependency barriers. Failure of either condition rejects the candidate before a business-kernel launch.

## Composition

`bench/iter715_make_unique_dlto.py` copies the exact uniquely named Iteration-714 RDC build inputs, renames both host and CUDA entry symbols again to prevent ELF/CUDA interposition in a later same-process A/B, and adds `-dlto` to both the CUDA compile and device-link commands. It does not edit production kernel or benchmark source, operator math, task order, synchronization, communication, or public ABI.

## Qualification

This checkpoint is source/build-recipe composition only. No JIT, CUDA launch, correctness result, or latency result is claimed until after this commit.
