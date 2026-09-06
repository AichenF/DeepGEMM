# Iteration 719b: late WGMMA arrive compiles to byte-identical SASS

## Result

The fresh late-arrive JIT succeeds and retains the same safe M128 resources
as Iteration 718:

- split-K2: REG56, STACK32, SHARED2048, LOCAL0;
- split-K4: REG56, STACK32, SHARED2048, LOCAL0.

However, the exact M128 split-K2 function is not merely equivalent in counts;
its complete 9,237-line SASS slice is byte-for-byte identical to the early-
arrive Iteration-718 function.  Both files have SHA256
`e77582b7c48d87f705755178543307c63d1cec4199f5b93220d95058c9f9938a`,
and `cmp` returns zero.

Consequently the exact W2 interval remains:

- 32 FP16 QGMMA;
- 32 `WARPGROUP.DEPBAR`;
- 32 `WARPGROUP.ARRIVE`;
- zero `STL` and zero `LDL`.

Ptxas canonicalizes the explicit late fence and the prior early fence plus
injected fences to the same machine schedule.  The DeepGEMM source-order rule
is valid, but moving the source-level fence inside this divergent monolithic
entry cannot override ptxas's register-lifetime safety transformation.

## Artifacts

- generated CUDA source SHA256:
  `348cf31f7bbe091feaa4f1e0e0099457b55e217498ad3e202c58b0e112340168`
- extension SHA256:
  `9a85bd54bdd88442cc184af828c0b96de25dd0d5a0ad6c9a997dd0d51435df47`
- JIT log SHA256:
  `4653cf270f98aaec04d7ae2e837058b84786d25b21a736eacd7fbbe3ff1de0e9`
- exact M128 split-K2 SASS SHA256:
  `e77582b7c48d87f705755178543307c63d1cec4199f5b93220d95058c9f9938a`
- summary:
  `bench/results/iter719a_f16_pair_late_arrive_static_20260906.log`
- raw JIT/SASS:
  `bench/results/iter719a_f16_pair_late_arrive_jit_20260906.log` and
  `bench/results/iter719a_f16_pair_late_arrive_m128_split2.sass`

## Decision

**Reject before CUDA business-kernel launch.**  The at-most-20-DEPBAR gate
fails with 32, and byte identity proves there is no hidden static difference
worth timing.  No correctness or performance result is claimed.

This closes the same-frame W2 pairing family: predecode, shared spill,
compiler operand fences, early-clobber multi-instruction asm, warp sync,
reduced FP16 destinations, and correct late hardware-fence placement all
produce the same 1:1 QGMMA/wait schedule.  The next candidate must change
coarse dataflow or phase ownership.
