# Iteration 710c — repaired existing W2 outline SASS audit

## Scope

Compile and inspect the historical whole-W2 `__noinline__` phase after the
Iteration-710b explicit-scope repair.  This is a static gate for the proposed
terminal non-returning W2+collective device tail; no CUDA business kernel was
launched.

## Exact build

- H20 physical GPU 1, `TORCH_CUDA_ARCH_LIST=9.0a`.
- Selected TP4 bundle plus
  `V4_SINGLE_LAUNCH_W2_PHASE_NOINLINE=1`.
- Extension: `v4tp_b9572d91268c0bf6a995_v178mspec`.
- M128/split-K2 main entry:
  `REG56 STACK48 SHARED2048 LOCAL0`.
- The extracted phase callee is a 20,800-byte local function in the exact
  M128/split-K2 code section.

## SASS result

- Callee SASS SHA256:
  `16fc8775103384986e55dd5e6dbfacba51cbdf7d67a23b006bb3fbac80929bfe`.
- 32 QGMMAs and 32 `WARPGROUP.DEPBAR.LE` instructions.
- Every W2 QGMMA remains serialized by its own dependency wait; the callee
  does not recover standalone W2's 32:16 ratio.

## Decision

The existing high-argument phase outline is repaired for reproducibility, but
**reject it as evidence for a non-returning tail before runtime**.  Removing
only its eventual return cannot fix a 1:1 schedule already baked into the
callee.  The next static control disables the fused-only
`AssumeValidMblock=true` specialization while preserving the outline.  The
audited standalone 2:1 W2 entry uses `AssumeValidMblock=false`; this isolates a
previously confounded template/code-generation difference before adding any
new tail architecture.

Raw artifacts:

- `bench/results/iter710c_existing_w2_phase_outline_jit.log`
- `bench/results/iter710c_existing_w2_phase_outline_resources.log`
- `bench/results/iter710c_existing_w2_phase_outline_m128_split2.sass`
