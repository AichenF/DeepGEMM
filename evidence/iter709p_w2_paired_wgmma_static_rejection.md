# Iteration 709p — reject explicit paired W2 QGMMA issue

## Exact build

- H20 GPU1 visibility, source SHA256
  `b5a7d47f7497fdbdaabfe3af2fbf208b3633d9200a7e539f461a8a3cef261450`.
- Flags: selected TP4 defaults plus W2 predecoded S2R and explicit paired
  WGMMA issue.
- Extension: `v4tp_edcaefbd74668cbf0c6a_v178mspec`.
- Exact target: `tp4_megamoe_single_launch_kernel<2,128>`.

## Result

JIT/link pass.  The target remains `REG56 STACK32 SHARED2048 LOCAL0`, so it
retains nine-CTA/SM register admission and no fixed local spill.  Extracted
SASS SHA256 is
`96ada2a70463cfe1282009fa83cc201e9e5a4e5a5fd96b86311c6eb427847b7a`.

Despite the separate adjacent C++ issue loop, ptxas emits 64 QGMMAs and 64
`WARPGROUP.DEPBAR.LE` instructions across the two runtime W2 paths.  The
critical ratio stays 1:1.  Examples still alias destination and source bases
(`R24,R24` and `R28,R28`) and wait immediately.

## Decision

**REJECT at the static gate.**  No CUDA business kernel was launched and no
correctness or cold-L2 timing is claimed.  Explicit C++ operand lifetime plus
predecoded lookahead is insufficient to overcome ptxas register aliasing at
the fused 56-register bound.  Keep diagnostics default-off for reproduction
and stop further same-frame prefetch/issue rearrangements.

Raw artifacts:

- `bench/results/iter709p_w2_paired_wgmma_jit.log`
- `bench/results/iter709p_w2_paired_wgmma_resources.log`
- `bench/results/iter709p_w2_paired_wgmma_m128_split2.sass`
