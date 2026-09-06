# Iteration 699: W2 FP16 accumulator resource/correctness gate

The repaired candidate compiled as extension
`v4tp_790c6b763a4cf8954fae_v178mspec`. On physical H20 GPU1 with M128
random routes (seed 20260902), the complete routed W2 output versus the
unaffected same-source multi reference reports:

- cosine: `0.9999993512848424`
- relative L2: `0.001139531250097178`
- finite: `true`
- padded rows: `1992`
- packed barrier generation words after wrap test: `[0,0,0,0]`

This is acceptable under the profiler's cosine >= 0.999 gate, but it is a
real precision change rather than bitwise preservation. The exact
M128/split-K2 one-kernel entry remains `REG56 STACK32 SHARED2048 LOCAL0
CONSTANT[0]1361`; there is no new CTA-residency tier. Its extracted SASS has
SHA-256
`aaa99bee6be30ecab80ed803f9852e571cb010b958c53359c7763d51fa7e3de3`
and contains 32 F16 QGMMA instructions, confirming the intended instruction
variant was emitted.

The diagnostic launch followed a separate excluded 256 MiB L2 clear and did
not execute TP communication. It is a correctness/resource gate, not a
latency result. The candidate advances only to repeated adjacent cold-L2
W2 phase timing.
