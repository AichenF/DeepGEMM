# SM90 NVFP4 Lane-Native RS Decode Experiment

## Goal

Test whether a lossless lane-native deployment prepack can make Hopper FP8
register-source WGMMA competitive with the retained Pro N24 shared-source
kernel.  The experiment targets the measured Pro small-M gap to optimized
W8A8 while retaining standard NVFP4 E2M1 values and UE4M3 scales.

## Motivation

The previous RS candidate removed 97% of decoded-FP8 shared stores and kept
the same GMMA count, but its final pair-native layout still took about 1.68 ms
at Pro M64 versus about 1.34 ms for N24.  Each RS lane consumed only one half
of a packed pair, so the decoder issued work for values owned by its neighbor.
That structural duplication outweighed the removed shared-memory traffic.

CUTLASS `ALayout_64x32` assigns each lane sixteen FP8 A values: four values
for each combination of two output rows and two K16 scale groups.  A deployment
layout can therefore give every lane exactly its own sixteen E2M1 values and
avoid decoding a neighbor's half.

## Deployment Layout

For each 64x32 RS fragment:

- Store exactly 8 bytes of E2M1 data per lane.  The sixteen nibbles are grouped
  as four independent four-value selectors in RS register order.
- Store one four-byte UE4M3 scale record per four-lane row group.  The four
  lanes share the record through shared-memory broadcast.
- Preserve every original E2M1 nibble and UE4M3 byte exactly.  The transform
  is a bijective model-load prepack, not a requantization.
- Keep the complete BN256/BK128 deployment tile at or below its existing
  20 KB allocation.  Do not materialize FP8 values or an expanded FP8 table.

The first decoder expands grouped nibbles directly into the four 32-bit
register operands required by RS WGMMA.  A pair-native one-half decoder is
kept in the same harness as the relevant control.

## Screening Gates

1. Exhaustive nibble/scale cases and randomized fragments must produce
   bit-identical FP8 register bytes to the canonical NVFP4 LUT decoder.
2. Inspect SASS for packed/scale loads, `IDP.4A`, `PRMT`, local memory, stack,
   and registers.  No spill is acceptable.
3. The isolated net decoder time must improve by roughly 20% over the
   pair-native RS control.  A smaller gain cannot plausibly recover the prior
   candidate's approximately 25% endpoint deficit and is rejected before
   full-kernel integration.
4. If the gate passes, implement the RS kernel in new candidate files only,
   then run Pro M=8/64 correctness for `global_scale=none/expert`, a seed-101
   screen, and three-seed A/B/B/A against both retained N24 and optimized W8A8.

No production fused/split body, environment variable, runtime argument,
commit, or push is part of this experiment.

## Outcome

The fragment layout and decoder were implemented only in the isolated
experiment harness and passed bit-exact comparison for all 127 valid UE4M3
scale codes times all 16 E2M1 codes, plus randomized model-like inputs.

The best form braided the sign bits of two four-value groups into the unused
selector bits of the other group. Its cubin used 24 registers, no stack, and
no local memory. Relative to the branchless pair-native control, SASS reduced
`IDP.4A` from 16 to zero and `PRMT` from eight to four. Median net fragment
cycles improved from 985 to 811 on the exhaustive input (17.7%) and from 1027
to 851 on model-like scales (17.1%).

This misses the 20% integration gate. More importantly, decoder improvement
is only a fraction of endpoint latency, whereas the prior optimized RS kernel
was about 25% slower than the retained N24 kernel. Full-kernel integration
cannot recover that deficit from this signal, so the RS architecture remains
rejected. Raw cycles, resources, and SASS are under
`/root/fac/scripts/megamoe/nvfp4_rs_lane_native_20260703/`.
