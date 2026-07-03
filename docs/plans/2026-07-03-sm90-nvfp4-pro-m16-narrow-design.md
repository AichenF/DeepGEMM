# SM90 NVFP4 Pro M16 Narrow Fused Design

## Scope

Create a separate Pro braided three-stage fused candidate for runtime
`num_tokens <= 16`. Keep the standard ModelOpt NVFP4 input, braided 80-byte
deployment layout, three-stage pipeline, single active dispatch warp, and all
numerical behavior unchanged. Do not modify the retained winner.

## Mechanism

For one expert, exact routing cannot produce more expert tokens than the
request's `num_tokens`. Therefore M<=16 makes swap-AB N24 and N64 unreachable.
The candidate instantiates only:

- L1: N8, otherwise N16.
- L2: N8, otherwise N16.

The host wrapper asserts `num_tokens <= 16`; larger requests continue to use
the retained kernel. This should remove two WGMMA template families and their
epilogue promotion paths from the candidate cubin. It adds no env, argument,
metadata, or request-time prepack.

## Evidence And Gate

The earlier Pro N24 winner improved over an all-bucket version while reducing
cubin size from roughly 196KB to 164KB, so a code-footprint screen is justified
at the current worst-gap M16 point. First require exact-NVFP4 correctness at
M=8/16 for both global-scale modes, unchanged 168-register/no-spill resources,
and a smaller cubin/SASS footprint. Then run seed-101 M16 A/B/B/A. Advance to
three seeds only for at least a 1.5% max-rank gain with no M8 regression.
Rejected wiring is removed; no commit or push is allowed.
