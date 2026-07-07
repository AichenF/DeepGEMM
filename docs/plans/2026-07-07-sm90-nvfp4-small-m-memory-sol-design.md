# SM90 NVFP4 small-M memory-SOL optimization design

## Goal and measured baseline

Optimize the dense SM90 NVFP4 grouped GEMM for the official
`128x4096x7168/E8` memory-bound shape while preserving genuine packed-FP4
persistent storage.  Using the accepted H20-3e peaks of 296 TFLOP/s and
4.814304 TB/s, the minimum-byte roofline is 27.851842 us and 70% SOL requires
at most 39.788 us.  The large `8192x4096x7168/E8` shape must remain at or above
80% compute SOL.

Iteration 85 is the current small-M best:

- 96.7675 us
- 28.7822% bound-aware roofline SOL
- N32 transposed WGMMA, K256 cooperative stage
- 72 registers, no local memory or stack, 25,600 bytes dynamic shared memory
- true packed FP4 globally; FP8 exists only transiently in registers/shared

Iterations 86-88 established that N64 padding and replacing the full-DP4A
selector path with PRMT do not improve this baseline.

## Non-negotiable invariants

1. The branch remains based on `main`; `megamoe_nvfp4` is reference-only.
2. Persistent weights contain exactly the original FP4 codes, block scales,
   and global scales.  `prepare()` may only apply a lossless byte permutation.
3. No persistent FP8/BF16 replica, residual term, correction table, duplicated
   weight payload, or sparse path is introduced.
4. The timed kernel must consume packed FP4 and perform any FP8 expansion only
   on chip.
5. The Cupra checker is unchanged.  Focused probes must have zero violations;
   the already accepted K7168 single-E4M3 representation tail remains reported
   honestly rather than hidden.

## Approved implementation sequence

### Phase 1: bank-skew the transient dequant LUT

The current shared LUT stores 128 `uint2` entries at an eight-byte stride.
With 32 four-byte shared-memory banks, an entry starts at bank
`(2 * scale) mod 32`, so random per-row scales can begin on only 16 banks.
Every warp performs repeated random `uint2` lookups in the decode hot loop.

Store each entry as three 32-bit words instead: two payload words plus one pad.
The entry then begins at `(3 * scale) mod 32`; because 3 and 32 are coprime,
all 32 starting banks are reachable.  The LUT grows only from 1,024 to 1,536
bytes of transient shared memory.  Persistent bytes, decoded FP8 values, DP4A
selector arithmetic, WGMMA work, and output mapping remain unchanged.

Implementation details:

- Restore the exact Iteration 85 N32/full-DP4A kernel before editing.
- Initialize `smem_lut[3*i + 0:2]` from the same constant table and leave the
  third word unused.
- Load the hot-loop `uint2` explicitly from words `3*scale` and `3*scale+1`.
- Increase the host-side small-M LUT allocation from 1,024 to 1,536 bytes.
- Retain K256, two pass slots, N32 WGMMA, and the exact-byte fused layout.

### Phase 2: N128 cooperative CTA if Phase 1 is insufficient

Template the small-M kernel for one or two output warpgroups.  The official
path uses two 128-thread warpgroups in one N128 CTA.  Each warpgroup owns an
independent N64 weight tile and N32 WGMMA accumulator, while both share one
activation stage and one LUT initialization.  This halves the grid and
amortizes activation/prologue work without duplicating persistent weights.

Keep a one-warpgroup N64 fallback for shapes whose N is not divisible by 128.
The two-warpgroup path uses approximately 41 KiB dynamic shared memory at K256;
resource inspection decides whether three-CTA residency is spill-free before
performance timing.  If register pressure or the scheduling tail outweighs
the amortization, revert exactly to the best measured Phase 1 revision.

### Later directions

Only after the two approved phases are measured should more invasive options
be considered, such as a separately scheduled producer/consumer pipeline or
cluster distributed shared memory.  Split-K plus global reduction is a last
resort because it adds timed output traffic to an already memory-bound shape.

## Verification and iteration protocol

For every candidate:

1. Build with a fresh JIT cache.
2. Run the known K256 routing/tail probe with offsets `[0,17,17,69,70]` and the
   E1 128-token cross-pass probe; require zero violations and no NaN/Inf.
3. Inspect cubin resources and SASS; reject local-memory spills and record
   register, IDP, PRMT, LDG, STS, BAR, and QGMMA counts.
4. Run the official focused benchmark with 8 input slots, 10 warmups, and 30
   cold-L2 timed iterations.
5. Immediately append `ITERATIONS.md` and commit the iteration.

Once the focused shape reaches at least 70% SOL, run the complete six-shape
suite, repeat the primary large shape, audit exact persistent byte counts and
lossless prepared-layout round trip, verify active source contains no sparse
path, and match source hashes on the H20 host.  Completion is valid only if the
small memory-bound shape is at least 70% and the large compute-bound shape is
still at least 80%.
