# Iteration 718b: paired FP16 inline W2 retains the serialized schedule

## Static result

The fresh Iteration-718a JIT succeeded.  Both TP4 M128 entries satisfy the
resource gate:

- split-K2: REG56, STACK32, SHARED2048, LOCAL0;
- split-K4: REG56, STACK32, SHARED2048, LOCAL0.

The exact split-K2 entry contains 32 FP16 W2 QGMMAs.  In the W2 interval
(SASS slice lines 2200-4420), it also contains 32 `WARPGROUP.DEPBAR`, 32
`WARPGROUP.ARRIVE`, and zero `STL`/`LDL`.  Representative pairs are:

```text
WARPGROUP.ARRIVE
QGMMA.64x8x32.F16.E4M3.E4M3 R34, R28, gdesc[UR8], RZ, !UPT, gsb0
WARPGROUP.DEPBAR.LE gsb0, 0x0
WARPGROUP.ARRIVE
QGMMA.64x8x32.F16.E4M3.E4M3 R24, R28, gdesc[UR8], RZ, !UPT, gsb0
WARPGROUP.DEPBAR.LE gsb0, 0x0
```

The first two destinations (`R34` and `R24`) do not overlap each other or
their common source base (`R28`), so the retained serialization is not the
FP32 candidate's simple destination/source register-alias problem.  The
compiler still injects an arrive/wait around every operation in the
divergent monolithic control-flow region.

## Artifacts

- generated CUDA source SHA256:
  `1bcaf88af595073645704b1ddc348b239ec35b2902043c52e543d275332147da`
- extension SHA256:
  `5296713eee4cb9c1f5549859c46bbd32a8069aed467f9bc64c1f6ba3b095a9c8`
- JIT log SHA256:
  `5d6687d098c937f7b911ab917f2a92affef47c83ac9b7d5c1589109ae0a31f7b`
- exact split-K2 SASS slice SHA256:
  `e77582b7c48d87f705755178543307c63d1cec4199f5b93220d95058c9f9938a`
- machine-readable summary:
  `bench/results/iter718a_f16_pair_static_20260906.log`
- raw JIT log and SASS slice:
  `bench/results/iter718a_f16_pair_jit_20260906.log` and
  `bench/results/iter718a_f16_pair_m128_split2.sass`

## Decision

**Reject before CUDA business-kernel launch.**  The candidate fails the
predeclared at-most-20-DEPBAR gate (32 observed, target 16).  Resource usage
is safe, but the intended W2 issue overlap is absent, so no correctness or
timing run can establish the hypothesized benefit.  Default behavior remains
unchanged because both FP16 switches are opt-in.

This closes the reduced-accumulator paired-inline variant of the fine-grain
compiler-scheduling family.  Further work should change coarse phase/dataflow
ownership rather than add another spelling of the same two WGMMA calls.
