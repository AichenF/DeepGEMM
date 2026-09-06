# Iteration 712d — natural-register paired inline W2 is unchanged

## Scope

Compile-only H20 GPU1 audit of the already implemented paired-W2 inline-asm
path under the unbounded TP4 M128 entry.  The build enables predecode, paired
groups, one-value shared spill, compiler operand fences and the two-QGMMA
inline-asm block.  It retains selected compact W13, disables both wave
rotations and launches no business kernel.

Extension: `v4tp_cac6ed3c5ead6db18c24_v178mspec`.

## Result

The M128/split-K2 global entry is `REG80 STACK32 SHARED2048 LOCAL0`.  Exact
symbol-205 SASS contains 32 W2 QGMMAs and 32 dependency barriers.  Its SHA256
is `7b021bb546c86fc4bf21138595eb0f86be9fc75978a85587c76909fedab6a211`,
byte-identical to Iteration 711i without paired inline assembly.

## Decision

**Reject before correctness/timing.**  Under natural register allocation,
ptxas lowers the CUTE-call and two-instruction inline-asm forms to the same
serialized W2 machine code.  Together with the bound9 variants, this closes
the in-translation-unit source/operand-lifetime family.  Do not add more
fences or inline-asm permutations.  Only a separately compiled relocatable
device callee can still test whether standalone's function compilation
context is recoverable while retaining one global business launch.

