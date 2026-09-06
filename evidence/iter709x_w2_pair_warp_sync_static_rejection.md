# Iteration 709x — real warp sync still does not pair fused W2 QGMMAs

## Exact result

- Source SHA256:
  `f3d1f5debbc07089fed0a1ea519a1aa1f8ee8579b2603f862097d72df44c8175`.
- Extension: `v4tp_89c670dde5203868ac7d_v178mspec`.
- Exact M128/split-K2 entry: `REG56 STACK32 SHARED2048 LOCAL0`, with
  19,456-byte dynamic shared memory.
- Exact SASS SHA256:
  `c7821a29f87835a6278480933fc0bb9882f00bb1780ca57bdafcca304d42f9f1`.
- Counts: 64 QGMMAs, 64 `WARPGROUP.DEPBAR.LE`, and 24 total WARPSYNC
  instructions in the complete entry.

The real warp barrier is emitted, but ptxas continues to schedule and wait
each QGMMA independently.  The required 2:1 ratio does not materialize, while
the barrier adds machine work.

## Decision

**REJECT before CUDA business launch/timing.**  Close the family of same-frame
REG56 predecode, explicit pair, manual spill, compiler fence, paired PTX and
warp-barrier variants.  None changes the fused dependency-wait count.

Next perform a read-only 64-register/eight-CTA fused control build and SASS
audit.  If it pairs, the obstruction is the 56-register capacity cliff; if it
does not, the cause is broader persistent-entry code generation and further
manual spilling is unjustified.

Raw artifacts:

- `bench/results/iter709x_w2_pair_warp_sync_jit.log`
- `bench/results/iter709x_w2_pair_warp_sync_resources.log`
- `bench/results/iter709x_w2_pair_warp_sync_m128_split2.sass`
