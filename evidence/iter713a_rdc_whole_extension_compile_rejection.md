# Iteration 713a: whole-extension RDC compile rejection

Date: 2026-09-06

## Question

Can the already generated Iteration-711g CUDA translation unit be switched to
relocatable device code, without changing its math or call graph, so that an
outlined W2 phase gets a separately compiled register schedule?

## Exact probe

- Input CUDA source: generated Iteration-711g unbounded-M128 plus whole-W2-
  phase-outline source.
- Source SHA256:
  `26d10ebbc86de2ec3440e2949066034c73242d23d277fcd1a1982dc3ccf16fca`.
- Added compile option: `-rdc=true`.
- Intended second step: `nvcc cuda.cuda.o -dlink ...`.
- Temporary Ninja SHA256:
  `7c00bd8b0858b347f152326c987da38b6efee51271c14cb29518068bc0a7e4b2`.
- Raw build log SHA256:
  `9abaa618bce023d7b785b104ab29c6c065bf80e535a715633b0fcba3a3c172b9`.

No production source was modified.  No CUDA business kernel was launched.

## Result

NVCC reaches ptxas but fails the relocatable compilation before `-dlink`.
There are 18 `ptxas error` diagnostics:

- Eight TP4 M8/M16/M32/M64 entry-to-W2 calls fail because the entries have a
  maximum register count of 64 while `single_launch_w2_gemm_phase<true>` needs
  195 registers.
- Four TP8 small-message entry-to-multicast calls fail because the callee needs
  134 registers and the entries have a maximum of 64.
- Six TP8 large-message entry-to-NVLS calls fail because the callee needs 100
  registers and the entries have a maximum of 64.

The build therefore produces neither a linkable RDC object nor inspectable
final SASS, and there is no correctness or latency result.

## Interpretation and decision

This is a hard compile-time incompatibility for a drop-in, all-shape RDC
conversion: externally callable device functions must satisfy caller/callee
register contracts that ptxas could previously resolve within the monolithic
translation unit.

The log does not emit the same W2 error for TP4 M128 because the selected M128
entry uses `__launch_bounds__(128, 1)` and is intentionally unbounded.  Thus the
result does **not** yet prove that M128-only RDC leaves the W2 schedule at
32 QGMMAs / 32 dependency barriers.  The next and final low-cost test is to
compile a generated-source TP4-M128-only specialization, inspect its linked
SASS, and proceed no further unless it recovers approximately 32/16.
