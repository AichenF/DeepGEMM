# Iteration 706: isolated W13 packed-FP16 WGMMA accumulator probe

Default-off flag `V4_SINGLE_LAUNCH_W13_F16_WGMMA_ACCUM` reaches only the
selected TP4 one-kernel W13 template instances. M128 uses the compact
one-device-call-per-CTA phase; M8/16/32/64 use the selected inline phase.
Standalone/multi W13 stays FP32, and W2 FP16 remains a separate default-off
flag.

Within each K128 activation-scale group, each N8 fragment uses two packed
FP16 registers instead of four FP32 registers. The four values are promoted
back to FP32 before the existing activation-scale multiply and cross-group
accumulation. Routing, W13 split-K workspace/reduction, SwiGLU, FP8 requant,
W2 and communication are unchanged.

Candidate source SHA-256 is
`37545fd28d36186241c2e1e678f561e0c3839ba4cd59a05c7dce8eaddcd3fb1e`.
Python syntax validation passes. This record contains no CUDA or performance
claim; fresh M128 JIT, resource inspection and complete-output correctness
are required next.
