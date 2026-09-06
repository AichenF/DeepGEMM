# Iteration 709r — one-value shared spill does not form a QGMMA pair

## Exact build

- Source SHA256:
  `8e9456ed51b74f6d89807fa0e39c75621403827c5c2e8b43498ca07997d69c1a`.
- H20 GPU1 visibility; selected TP4 defaults plus predecode, paired issue and
  one-value shared-spill flags.
- Extension: `v4tp_6b38843471d1945a31a0_v178mspec`.
- Exact M128/split-K2 entry: `REG56 STACK32 SHARED2048 LOCAL0`; dynamic
  shared allocation is 19,456 bytes, so total per CTA is 21,504 bytes and
  nine CTAs require 193,536 bytes.

## Static result

The exact SASS SHA256 is
`ef5ae0036d8bdf026bbdee714fbfdef2742aa6a3d988168f14c4ed2989c4e1e1`.
It still contains 64 QGMMAs and 64 dependency barriers across the two emitted
W2 paths.  Thus it fails the required 2:1 QGMMA-to-wait target.

The SASS explains why the C++ pair did not materialize.  In one path, ptxas
issues the first QGMMA at `0x6670`, immediately waits at `0x6680`, only then
loads the spilled second-group value at `0x6690`, and issues the second QGMMA
at `0x66b0`.  It legally moved the load out of the C++ preparation loop to
minimize live registers, defeating adjacency.

## Decision

**REJECT before CUDA launch/timing.**  Keep the diagnostic default-off.  The
next bounded test will add empty compiler operand fences on every materialized
FP8 source register immediately before the issue loop.  This must force the
shared load/decode to complete before the first QGMMA; retain only if the
machine-code barrier ratio then changes.

Raw artifacts:

- `bench/results/iter709r_w2_pair_spill_one_jit.log`
- `bench/results/iter709r_w2_pair_spill_one_resources.log`
- `bench/results/iter709r_w2_pair_spill_one_m128_split2.sass`
