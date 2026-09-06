# Iteration 709n — standalone W2 operand-lifetime audit

## Scope

Read-only `cuobjdump` of the standalone W2 specialization already present in
the exact Iteration-709m extension.  No CUDA kernel was launched and no
latency was measured.

## Exact function

`route_gemm<512,4096,1,false,0,false,false>` uses `REG61 STACK0 SHARED2048
LOCAL0`.  Its extracted SASS SHA256 is
`d62b23aa061e2bd3a02a9ce3c35d763a4a8a590777a2aec01257adc70610a9ce`.
It contains exactly 32 QGMMA and 16 `WARPGROUP.DEPBAR.LE` instructions,
independently reproducing the 2:1 ratio observed in the saved NCU report.

## Actionable SASS difference

The standalone first pair at PCs `0xe70` and `0x10c0` uses separate
accumulator/output bases (`R28`, `R24`) and separate register-source bases
(`R48`, `R44`), then waits once at `0x1280`.  In the fused entry, one emitted
path starts at `0xcd80` with `R24` serving both destination and register-source
bases, waits at `0xce50`, issues the other half at `0xceb0`, and waits again at
`0xcec0`.

This is strong compiler evidence that the 56-register fused contract aliases
WGMMA operand and accumulator lifetimes, forcing immediate serialization.  It
is not by itself a proof of all ptxas scheduling decisions.

## Next experiment

Preserve the Iteration-709l predecoded lookahead, but explicitly materialize
both current N64 FP8 operand pairs before issuing two adjacent QGMMAs.  Require
`REG56`, no spill, nine-CTA admission, 64 QGMMAs and approximately 32 static
dependency barriers across the two emitted paths before any correctness or
cold-L2 timing.

Raw artifact: `bench/results/iter709n_standalone_w2_static.sass`.
