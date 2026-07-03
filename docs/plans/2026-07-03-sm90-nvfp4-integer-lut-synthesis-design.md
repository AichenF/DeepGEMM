# SM90 NVFP4 Integer LUT-Synthesis Experiment

## Goal

Measure whether the Pro decoder can replace each random shared-memory
UE4M3/E2M1 LUT load with a bit-exact integer synthesis of the same eight FP8
bytes. This keeps the retained shared-source WGMMA architecture and targets
the measured excess shared loads without storing FP8-derived deployment data.

## Construction

For every valid nonnegative UE4M3 code `s` in 0..126, all eight positive
E2M1 products can be derived exactly from integer code operations:

- `0.5*s` uses round-to-nearest-even halving below code 16 and an exponent
  decrement above it.
- `1.5*s` is a periodic mantissa correction plus saturating exponent logic.
- Multiplication by two is either subnormal doubling or a saturating exponent
  increment.
- `3*s`, `4*s`, and `6*s` follow from the same operations, with exact
  low-subnormal handling.

The synthesized pair must be byte-identical to
`kE2M1AndUe4m3ToFp8Lut[s]`. No auxiliary metadata, FP8 weight copy, or
per-scale FP8 table is cached. Only the isolated harness is modified first.

## Gates

1. Exhaustively compare all 127 valid scales and all 16 signed E2M1 codes.
2. Compare net cycles against the current paired-DP4A shared-LUT Pro decoder
   under both exhaustive and model-like scale distributions.
3. Inspect ALU/LSU SASS and resources. Integrate into a separate Pro kernel
   only if the decoder improves by at least 15% without spills.
4. Any integrated candidate must then pass real M=8/64 correctness and
   multi-seed A/B/B/A before retention.

No production kernel, runtime control, commit, or push is changed at the
screening stage.

## Outcome

The integer construction was bit-exact for every valid scale and E2M1 code,
but it was decisively slower. Median net row-decode cycles increased from
1549 to 3651 on the exhaustive distribution and from 649 to 3629 on
model-like scales. The narrow model distribution makes the current shared LUT
especially effective through broadcast and low-conflict reuse, while the
synthesis executes the same arithmetic for every thread.

The candidate misses the 15% gate by a wide margin and is not integrated.
Raw logs, resources, and SASS are under
`/root/fac/scripts/megamoe/nvfp4_integer_lut_20260703/`.
