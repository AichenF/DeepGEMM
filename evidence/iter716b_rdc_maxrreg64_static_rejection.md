# Iteration 716b: reject the 64-register-capped RDC candidate statically

## Result

The unique capped-RDC build completed and produced
`/tmp/iter716a_unique_rdc_r64/v4tp_49f35bd1b995105c53bc_v178mspec.so`.
The TP4 M128 entry meets the numeric register cap but fails the predeclared
stack/spill gate:

| specialization | registers | stack/thread | shared | local |
| --- | ---: | ---: | ---: | ---: |
| split-K2, M128 | 64 | 360 B | 1616 B | 0 B |
| split-K4, M128 | 64 | 360 B | 1616 B | 0 B |

The exact out-of-line W2 callee still has the favorable static schedule:

- 32 `QGMMA` instructions;
- 16 `DEPBAR` instructions;
- 32 `WARPGROUP.ARRIVE` instructions.

However, its first instruction is `IADD3 R1, R1, -0xf8, RZ`, creating a
248-byte callee frame.  Its SASS contains 61 `STL` and 66 `LDL` instructions,
or 127 explicit local-stack accesses.  The spill sequence begins immediately
with stores of R61, R60, R55, R54, and further live registers.

## Decision

**Reject before CUDA launch.**  A 360-byte entry stack is a material cliff
relative to natural RDC's 112 bytes and the ordinary monolith's 32 bytes.  The
64-register cap preserves W2's 32/16 issue grouping only by converting live
state into local-memory traffic; it does not provide the phase-local resource
contract sought by this experiment.  The candidate therefore fails its
committed static gate, so there is intentionally no correctness or timing run.

Production source and benchmark-driver hashes are unchanged:

- `v4_flash_tp_wgmma.py`:
  `30e6c402b4dc220d2a4eb03099d1f1cd0f82e59b0af38747b59906d84c3fdbdf`
- `bench/v4_flash_tp_single_vs_multi_graph.py`:
  `16bb9e09d6d23247ea2291f08992f48c1cd394309829913732c6bff9c55901bb`

Raw static output and artifact hashes are recorded in
`bench/results/iter716a_rdc_maxrreg64_static_20260906.log`.
