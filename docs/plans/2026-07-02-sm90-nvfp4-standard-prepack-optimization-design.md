# SM90 Standard-NVFP4 Runtime Dequant Optimization

## Objective

Move SM90 NVFP4 MegaMoE latency as close as possible to the optimized W8A8
kernel without retaining any FP8 weight copy and without changing numerical
quantization semantics.

## Compatibility Contract

- Canonical weights remain standard E2M1 packed values with one nonnegative
  UE4M3 scale per 16 values, matching ModelOpt quantization, rounding, and
  saturation.
- Model loading may perform one lossless deployment prepack and cache only the
  transformed NVFP4 representation. No request performs repacking.
- The deployment representation may reorder nibbles/scales and add integer
  selector metadata, but it must not contain materialized FP8 weights or
  precomputed FP8 weight tables.
- FP8 values produced transiently in shared memory are allowed and never
  survive the kernel launch.
- Exact-NVFP4 reference thresholds remain unchanged.

## Sequence

### 1. Prepack-aware dequant micro-optimization

Keep the current 80-byte BK128 row as the zero-growth target: 64 bytes of E2M1,
8 bytes of UE4M3 scales, and 8 currently padded bytes. Evaluate lossless FP4
nibble permutations and a 16-byte scale/metadata tail that reduce selector
packing, dependent scale extraction, LUT loads, address arithmetic, or STS
reordering. Do not store FP8-derived table entries.

Measure each candidate first in an isolated dequant benchmark and inspect SASS.
Only candidates with bit-identical output and a meaningful cycle reduction are
eligible for MegaMoE integration.

### 2. Cross-warpgroup single-kernel pipeline

Retain the two N128 math warpgroups in the BN256 fused CTA. For each K stage,
one warpgroup submits WGMMA for its current N half while the other warpgroup,
which has no pending WGMMA, decodes its N half of the next stage. Then exchange
roles. A prologue decodes stage zero and the final stage omits lookahead.

This keeps 384 CTA threads, preserves the existing accumulator footprint, and
does not make TMA producers wait for dequant. The implementation is accepted
only if SASS places PRMT/STS from one warpgroup inside the other warpgroup's
WGMMA interval without spills.

### 3. Optimized FP8-derived two-kernel fallback

If the single-kernel paths do not close the gap, derive NVFP4 L1 and L2 kernels
from `aichenf/megamoe_sm90_opt`, retaining its phase-specific scheduler,
compact frontend, direct L2 scatter, and cleanup policies. Replace each FP8 B
load with packed standard-NVFP4 TMA plus transient shared-memory dequant.

Test BN128 first because an N128 math warpgroup leaves enough of the 384-thread
register budget for an independent producer. Then test BN256 with math-side L1
dequant and producer-side L2 dequant. The L1 FP8 intermediate is an activation,
not a persistent weight copy.

## Verification

- Dequant unit test: bit-identical FP8 bytes for every E2M1 nibble and every
  valid UE4M3 scale, plus random packed rows.
- MegaMoE correctness: Flash, Pro, and middle-I; both global-scale modes;
  swap-AB boundaries; no relaxed tolerance.
- Resources: threads, registers, stack, local loads/stores, shared memory, and
  dequant/WGMMA SASS ordering.
- Performance: multi-seed process-level ABBA at M=8/16/32/64/128, followed by
  M=256 guardrails. Compare maximum-rank latency against both the current
  NVFP4 baseline and the canonical W8A8 results.
- Retain a candidate only when it improves repeated steady latency and does
  not materially regress another validated range.

No experiment is committed or pushed until explicitly requested.

## Prepack Screening Results

Two lossless 80-byte candidates were screened and rejected before production
integration:

- **Scale replica in padding:** duplicated the eight UE4M3 bytes into the
  existing eight-byte tail and selected a copy with `row & 8`. NCU reduced
  shared-load bank conflicts from 9,360 to 8,736 over 78 decoder CTAs with
  unchanged shared instruction counts. Isolated high-concurrency decoder
  cycles improved by roughly 0--5% depending on selector schedule, but
  process-level M64 A/B/B/A was not decisive: Flash centers improved only
  about 0.5%, while Pro centers regressed about 0.16%. The production change
  was removed.
- **Braided selector bits:** losslessly reordered each eight E2M1 codes so two
  16-bit magnitude selectors could feed LUT PRMT directly while signs occupied
  unused selector bits. Output was bit-identical, but it added one register.
  Isolated net cycles were 1,169 versus 1,135 for the current Flash low-hybrid
  path and 883 versus 779 for the current Pro paired-DP4A path. It was never
  integrated into production.

The experiment harnesses remain under
`docs/experiments/sm90_nvfp4_standard_prepack/`. No runtime control, format
change, or production code from either candidate remains.

## Cross-Warpgroup Result

The two N128 math warpgroups were made to alternate roles for non-swap BN256:
WG0 computed its current N half while WG1 decoded WG0's next-stage half, then
the roles exchanged. Two 256-thread named barriers protected the phase
handoff. Flash M64 passed exact-NVFP4 correctness in both global-scale modes,
and the cubin retained 168 registers, a 56-byte stack, and zero local memory.

The schedule was rejected because it serialized the two warpgroups' WGMMA
submission. Flash M64 steady median increased from roughly 481 us to 539.3 us
(about 12%). Pro M128 was a false-path control because its BLOCK_M=128 layout
does not use split-N; candidate and baseline were 1839.3 and 1831.3 us. The
cross-warpgroup implementation was removed completely.

## FP8-Derived Split Fallback Audit

The original audit was incomplete. The existing NVFP4 split L1/L2 bodies have
the FP8 phase structure, but they do **not** contain the optimized FP8 small-M
`swapAB` path (`BN128/BM64`, two M64N64 math warpgroups, and dynamic
`N_SWAP=8..64`). Historical measurements of the ordinary split kernels could
therefore not reject this specific design.

The missing path was implemented in two isolated experimental bodies,
`sm90_nvfp4_mega_moe_split_swap_l1_body.inl` and
`sm90_nvfp4_mega_moe_split_swap_l2_body.inl`. The old split bodies were not
modified. The port kept the standard 80-byte NVFP4 packed row, applied UE4M3
scales during transient shared-memory dequant, and reused the current NVFP4
global-scale and top-k-weight semantics. No FP8 weight copy or new runtime
control was introduced.

Correctness passed Flash (`I=2048`), middle (`I=2560`), and Pro (`I=3072`) at
M=8/16/32/64 with both `global_scale=none` and `expert` (24 cases, minimum
per-token cosine 0.9987). Three producer layouts were then screened on eight
H20 ranks at H=7168, E=256, top-k=8:

- Four dedicated dequant warps plus two swapAB math warpgroups used a
  512-thread L1 CTA. Against BN256 fused at M=8/16/32/64, I=2048 regressed
  31.8%/31.6%/45.8%/33.4%; I=3072 regressed
  41.7%/75.4%/49.0%/31.5%. L1 was capped at 128 registers.
- A compact `64 dispatch + 64 producer + 256 math` layout restored 168 L1
  registers but made the two producer warps wait for TMA and then dequant
  together. It was substantially slower (Flash 1.63--1.81 ms, Pro
  2.31--2.46 ms).
- A single TMA warp plus one dedicated dequant warp restored role overlap but
  made one warp decode all 128 B rows and reduced the packed-scratch pipeline
  to five stages. It was no better (Flash 1.71--1.95 ms, Pro
  2.37--2.68 ms).

For context, the unmodified optimized W8A8 branch on its canonical routing
presets measured Flash 379.4/521.7/379.0/406.5 us and Pro
847.3/969.7/1028.2/1134.3 us at M=8/16/32/64. These are not used for direct
ratios because the standalone NVFP4 and W8A8 scripts produced different
routing counts; the process-level BN256-versus-BN128 comparisons above use the
same NVFP4 harness and routing.

The split swapAB direction is rejected for the current deployment layouts.
BN128 doubles the N-tile CTA count relative to the already optimized BN256
fused swapAB path, adds an L1/L2 kernel boundary, and cannot reproduce FP8's
compact frontend without assigning scarce producer capacity to dequant. The
experimental production wiring was removed after measurement.

## LUT and Selector-Metadata Follow-up

Replicating or splitting the shared LUT did not improve the decoder. A
two-address deployment encoding reduced random bank conflicts from 120 to 92
per decoder CTA, but model-like scale distributions were flat in a single-CTA
test and did not justify an extra 1 KB shared LUT.

Larger lossless prepack rows were also evaluated. A 144-byte row stored direct
integer selectors plus the original sign words; a 112-byte row stored one
16-bit selector per packed FP4 word. Neither contained FP8 values. Both passed
Flash/Pro exact-NVFP4 correctness and kept the fused cubins at 168 registers,
56 bytes of stack, and zero local memory. Isolated decoder cycles improved,
but real phase profiling showed no reduction in math-dequant time because the
new shared loads replaced the removed selector instructions. The larger TMA
rows regressed M64 ABBA by about 2.8% on Flash and 1.8% on Pro. The production
wiring was removed; these results rule out selector metadata that increases
the 80-byte row size.

## Zero-Growth Padding-Selector Follow-up

The remaining eight padding bytes in the standard 80-byte row were populated
with four 16-bit integer magnitude selectors. This preserved the exact storage
size and all E2M1/UE4M3 values. The isolated 624-CTA decoder improved by about
21% for the Flash low-selector schedule and 31% for Pro DP4A, so the layout was
integrated into a separate candidate kernel and fused body without modifying
the production implementation.

Flash/Pro correctness passed at expected 4/8 in both global-scale modes, and
resources stayed at 168 registers, 56 bytes of stack, and zero local memory.
The real M64 ABBA did not retain the microbenchmark gain: Flash was about 0.2%
slower, while Pro was only about 0.3% faster with inconsistent run direction.
Pro phase profiling showed just a 1% math-dequant reduction, and SASS removed
16 `IDP.4A` instructions rather than changing the decoder critical path.

The candidate was rejected and all runtime wiring was removed. This closes the
integer selector-metadata direction for the current decode schedule even when
metadata fits entirely in existing padding. Future work should target work
that remains on the measured critical path rather than reducing selector
instruction count in isolation.

## Nibble-Group Prepack Follow-up

A zero-growth layout permutation produced the first repeatable endpoint win.
Within every packed E2M1 `uint32_t`, the four high nibbles and four low nibbles
are grouped into separate 16-bit halves. Each half is therefore a direct
four-byte magnitude selector for `PRMT`; a second `PRMT` plus `LOP3` restores
the original sign bits. This is a bijection over the standard E2M1 codes. The
UE4M3 scales and 80-byte fused row are unchanged, and no FP8-derived value is
cached.

The implementation lives in separate candidate kernel and fused-body files.
The production fused and split implementations were not edited. Isolated
624-CTA Flash decode improved by about 10%, and M64 phase profiling reduced
math-dequant by 27.2%. Loader dequant required a two-row interleaved schedule;
decoding a complete row at a time lost overlap and regressed larger M.

Correctness passed both dequant schedules and both global-scale modes with a
minimum per-token cosine of 0.9987. Resources remained 168 registers, 56
bytes of stack, and zero local memory. Multi-seed eight-rank ABBA over the
canonical Flash BN256 M values yielded a conservative equal-point geometric
latency improvement of 1.93%. M256 and M819 were flat within 0.2%; the largest
repeatable improvements were at M8/M16, followed by M32/M64.

The candidate remains experimental and is not selected by the production
wrapper. If promoted, the intended policy is a model-load-time, shape-derived
Flash+BN256 prepack and matching separate kernel dispatch, with no environment
variable or new request-time argument. Raw results are in
`/root/fac/scripts/megamoe/nvfp4_nibble_group_abba_20260703/`.
