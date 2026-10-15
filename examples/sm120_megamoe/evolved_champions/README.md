# SM120 MegaMoE Evolved Champions (P8/E384, P4/E384, P4/E192)

This directory contains the generated CUDA sources of the three qualified
champion kernels produced by the SM120 MegaMoE evolution campaign
(2026-08-28 .. 2026-08-31). Each file is a complete, standalone, single-entry
persistent kernel covering dispatch, task build, W1, fused SwiGLU +
requantization, W2, incremental remote return, weighted combine, and
acknowledgement — the same operator contract as `../qualified_g8/`, with the
same strict tolerances (dense oracle rel-L2 <= 0.01, max-frac <= 0.02,
bit-exact FP8/scale-factor intermediates, exact-zero protocol gates).

The kernels are generated from typed Loom Weave IR seeds (no raw-CUDA
sidecars); the seed and generated-source hashes are pinned in
`qualification-manifest.json`. They are qualification artifacts, not a
DeepGEMM JIT/API integration: host-side setup (NCCL GIN windows/signals,
TMA descriptors, buffer registration) follows the same protocol as the
qualified_g8 host and is documented per-kernel in the manifest.

## Files

- `megamoe_p8_e384_c56.cu` — 8-rank, 384 routed experts (48 local), candidate
  C56 (dynamic in-order tile claim). 110 CTAs x 384 threads, 101376 B SMEM.
- `megamoe_p4_e384_c56_maxrows.cu` — 4-rank, 384 routed experts (96 local),
  C56 + MAX_ROWS-8192 four-shape trunk. 110 CTAs x 640 threads, 100992 B SMEM.
- `megamoe_p4_e192_c56_maxrows.cu` — 4-rank, 192 routed experts (48 local),
  byte-parallel payload to P4/E384. 110 CTAs x 640 threads.
- `megamoe_p8_e256_v4flash_c56.cu` — 8-rank, 256 routed experts (32 local),
  DeepSeek V4 Flash geometry (H4096/I4096/O4096), same C56 schedule and
  precision recipe. 110 CTAs x 384 threads, 101376 B SMEM.
- `qualification-manifest.json` — pinned hashes, measurements, and receipts.

## Qualified performance (R2048 tokens/rank, top-k 6, H7168/I3072/O7168)

All numbers are strict same-session paired CUPTI contract medians with full
correctness gates passing (>= 3 replays, median + per-pair majority rule).
Baseline = the frozen live production kernel of each topology.

| Topology | Champion | Contract latency (ms) | vs frozen live | vs campaign start |
|---|---|---|---|---|
| P8/E384  | C56 | **10.835** (5/5 paired sweep) | **2.40x** (median) | 1.76x (19.035 ms) |
| P4/E384  | C56+maxrows | **10.93** (best set; 11.69 set 1) | ~2.07x/1.84x | ~1.33x |
| P4/E192  | C56+maxrows | **9.99** (best set; 11.41 set 1) | ~1.65x/1.59x | ~1.37x |

Four-shape suite (candidate-only single-entry CUPTI, exact correctness at all
rows, x3 medians; denominators = the campaign-start frozen baselines migrated
buffers-only to 8192 rows, faithful to ~0.5% at R2048 vs the live paired legs):

| Topology | R512 | R2048 | R4096 | R8192 | Suite geomean vs frozen baseline |
|---|---|---|---|---|---|
| P8/E384  | 5.035 (1.825x) | 10.775 (2.423x) | 19.621 (2.701x) | 35.670 (2.760x) | **2.396x** |
| P4/E384  | 6.130 (1.563x) | 11.668 (2.071x) | 21.869 (2.059x) | 42.099 (2.100x) | **1.934x** |
| P4/E192  | 4.165 (1.542x) | 11.277 (1.668x) | 21.774 (1.677x) | 42.436 (1.690x) | **1.643x** |

## Speed-of-light accounting (measured basis)

Machine constants (measured, not datasheet): 110 SMs/GPU, FP8
`mxf8f6f4 m16n8k32` ~585 TFLOPS sustained, GDDR7 ~1344 GB/s, inter-GPU GIN
put egress **46 GB/s per GPU** (NIC fabric, pattern-independent; 8 rails all
line-rate under balanced all-to-all — no free striping bandwidth).

| Component @R2048 (per rank, ms) | P8/E384 | P4/E384 | P4/E192 |
|---|---|---|---|
| GEMM (12288 routes = 1.624 TFLOP) | 2.78 | 2.78 | 2.78 |
| Weight stream (FP4) | 1.18 | 2.36 | 1.18 |
| Dispatch egress | 1.30 | 0.83 | 0.83 |
| Return egress (BF16) | 3.35 | 2.87 | 2.87 |
| Ideal fully-overlapped SOL | ~4.7 | ~3.7 | ~3.7 |
| Champion vs ideal SOL | 43.0% | 34.2-34.5% | 36.5-36.9% |

The ideal row assumes perfect comm/compute overlap and GEMM at sustained
peak. The measured *operational* floor for P8 — front 1.7 + compute window
5.1 (tensor-pipe 62% vs 89% for the isolated DeepGEMM donor GEMM; bounded by
the 101376 B SMEM cap, the absence of a packed-b4 ldmatrix path in PTX, and
fused-epilogue issue pressure) + remote-wait tail 1.3 (last-chunk burst dual
of expert-major completion order) + combine/ACK 0.6 — is **~8.7 ms
in-kernel (~9.1 ms contract)**; C52 runs at ~83% of that floor. Front,
tail-transport, ACK, fabric-bandwidth, and B-repack families were each closed
with measured negative or ISA/driver evidence (see campaign ledger).

## What the champions contain (mechanism summary vs. campaign-start baseline)

1. **Incremental chunked return + vectorized combine** — result chunks are
   published as their producing W2 tiles complete instead of after a global
   return phase; combine uses 128-bit vector loads.
2. **Fused SwiGLU + requantization in the W1 epilogue** — the standalone
   requant phase and one full W1_D round-trip through GMEM are eliminated
   (bit-exact BF16 round-trip preserved for the dense oracle).
3. **Header-carried per-expert counts + per-source gated scatter/W1** —
   receivers start histogram/prefix/task-build from headers and gate W1 tasks
   on per-(expert, source) readiness instead of wait-all.
4. **De-staggered, donor-architecture GEMM core** — DeepGEMM
   `sm120_fp8_fp4_gemm_1d1d` warp-tile geometry (32x64 per warp, TASK_M=128,
   N128 tiles) ported into the persistent kernel; the legacy K-step stagger
   register pipeline was removed after being root-caused as a latent
   correctness race.
5. **Expert-major return indexing (C23)** — owners return route rows in
   per-source expert-major pool order with a translated index map, so result
   chunks close (and put) throughout W2 instead of bursting at its end.
6. **Unified W1/W2 interleaved tile stream (C26)** — one guarded stream with
   W2 trailing W1 by a fixed task distance (Delta=8), removing the inter-phase
   grid syncs; W2 lags the W1 window by only ~0.18 ms.
7. **Service-warp parallel return tally (C31)** — 32-lane parallel chunk
   tally with a single elected GIN put issuer.
8. **Tail chunk split + donor TMA-store epilogue (C43 = C41 + C42b)** — the
   last <=256 routes per source return in 64-route chunks (halving the
   synchronized last-chunk burst), and W1_D/W2_D stores go through a 32 KB
   swizzled SMEM staging + `cp.async.bulk.tensor` epilogue. Individually
   neutral at the R2048 compute/wire equilibrium, jointly +2.2%.
9. **Chunk-ordered dispatch pack (C44 + C47)** — senders pack records in
   min-expert-sorted chunk order with per-(source, chunk) strong signals, so
   receivers scatter and open W1 gates incrementally as chunks arrive
   (first-W1 2.86 -> ~1.7 ms); C47 adds a sorted (owner, record) index so the
   first chunk hits the wire before the pack pass finishes.
10. **Shape-adaptive dispatch chunk floor (C52)** — chunk quantum 160 -> 96
    records for R<=512, recovering small-shape dispatch parallelism.

P4/E384 and P4/E192 carry the same mechanism family through item 8 (their
front economics differ: chunked dispatch pack is P8-specific — at
WORLD_SIZE=4 the dispatch wire is ~0.8 ms and per-source gating already
covers it, measured regression when ported).

## Qualification protocol

Every champion passed, in order: structural/validator/TVM-FFI/codegen gates;
R512 watchdog smoke (5 s); exported dense oracle (exact, original
tolerances); R2048 canary + 20-epoch soak with zero mismatches / protocol
errors and exact tile/requant counters; compute-sanitizer synccheck
(0 errors); then same-session paired CUPTI replays (>= 3, both GPU groups for
P4-class) under the median + per-pair majority promotion rule. PM-sampling
(4 us tensor-pipe/issue/DRAM/L2 time series) gated scheduler-family
candidates before paired replays.

## DeepSeek V4 Flash geometry variant (exploratory)

`megamoe_p8_e256_v4flash_c56.cu` generalizes the P8 champion to the V4 Flash
geometry: hidden 4096, intermediate 4096, output 4096, 256 routed experts
(8 x 32), top-k 6, identical precision recipe and schedule (constants-only
delta: six extents plus the dispatch-record layout offsets).

Steady-state in-kernel span medians (rank-0 phase timestamps, epochs 10-19 of
20, every epoch exact under the bit-level transport mirrors, exact-zero
protocol gates and task-build audits; no frozen-baseline contract exists for
this geometry, so these are exploratory coordinates, not paired-CUPTI
qualification rows):

| Tokens/rank | 512 | 2048 | 4096 | 8192 |
|---|---|---|---|---|
| in-kernel span (ms) | 2.64 | 6.24 | 12.42 | 22.21 |

The return wire is fully streamed under compute at this geometry (exposed
service-to-arrival tail 0.05-0.06 ms at every shape): with I/H = 1 the
FLOP-per-return-byte ratio is 1.33x the V4-Pro geometry, so the champion's
incremental return machinery hides the entire wire. The largest exposed
component is the R8192 dispatch front (~4.4 ms), pinned by the same
transport-level delivery-ordering limitation documented for the V4-Pro rows.
