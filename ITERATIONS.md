# Iteration Log

## Baseline

Branch `w4a8`, commit `3f46416`, H20-3e, CUDA 13.0.1. Cupra uses eight cold-buffer slots, ten warmups, and thirty CUPTI-timed iterations per shape.

| Shape | Runtime (ms) | TFLOPS | H20 FP8 util | Jitter |
|---|---:|---:|---:|---:|
| 512x2048x2048/E8 | 0.8246 | 5.21 | 1.76% | 20.95% |
| 4096x2048x2048/E8 | 4.0141 | 8.56 | 2.89% | 6.29% |
| 8192x4096x2048/E16 | 16.1041 | 8.53 | 2.88% | 2.02% |
| 4096x7168x2048/E32 | 16.7268 | 7.19 | 2.43% | 6.71% |
| 8192x4096x7168/E8 | 54.8834 | 8.76 | 2.96% | 3.07% |
| 128x4096x7168/E8 | 2.9405 | 2.56 | 0.86% | 17.34% |

## Summary

| Iter | Title | Large-M best util | Small-M latency | Status |
|---:|---|---:|---:|---|
| 1 | Remove residual FP8 WGMMA pass | 2.97% | 2.9381 ms | no-change |
| 2 | LUT/PRMT vectorized dequant | — | — | failed |
| 3 | Normalize LUT by eight | — | — | failed |
| 4 | Vectorize exact two-term dequant | 9.01% | 0.9704 ms | improved |
| 5 | Widen output tile to N=128 | 11.96% | 0.6980 ms | improved |
| 6 | Widen output tile to N=256 | 13.45% | 0.8482 ms | mixed |
| 7 | Dispatch N128/N256 by tile count | 13.43% | 0.7010 ms | improved |
| 8 | Merge primary/residual WGMMA batch | 13.44% | 0.7030 ms | no-change |
| 9 | Double-buffer decode under async WGMMA | 17.90% | 0.5516 ms | improved |
| 10 | Dedicated producer/consumer warpgroups | 21.78% | 0.5511 ms | improved |
| 11 | Increase large pipeline to 3 stages | 20.97% | 0.5520 ms | regression |
| 12 | Prepack weights for coalesced N reads | 40.52% | 0.4588 ms | improved |
| 13 | Account for expert tails in tile policy | 40.54% | 0.3880 ms | improved |
| 14 | Disable producer warpgroup for near-full grids | 40.55% | 0.3877 ms | no-change |
| 15 | Add sparse FP8 WGMMA wrapper and metadata probe | — | — | failed |
| 16 | Make SM90 MMA header self-contained | — | — | failed |
| 17 | Execute sparse WGMMA probe with SM90a target | — | — | failed |
| 18 | Classify sparse probe mismatches | — | — | diagnostic |
| 19 | Correct B32/B64 sparse-probe swizzles | — | — | improved |
| 20 | Decode packed NVFP4 into dual sparse WGMMA | — | — | improved |
| 21 | Integrate transposed dual-sparse grouped kernel | — | 15.7854 ms | failed |
| 22 | Replace scalar decomposition with generated LUTs | — | — | failed |
| 23 | Materialize generated LUT initializers | — | — | failed |
| 24 | Use explicit packed sparse decode LUTs | — | 5.6098 ms | improved |
| 25 | Add dual-warpgroup sparse pipeline | — | — | failed |
| 26 | Run dual-warpgroup sparse pipeline | — | 6.0962 ms | regression |
| 27 | Remove dynamic local arrays from sparse producer | — | 3.5266 ms | improved |
| 28 | Move top-2 selection into shared lookup | — | 3.0802 ms | improved |
| 29 | Split each weight row across two producer threads | — | 2.4294 ms | improved |
| 30 | Vectorize activation staging with uint4 | — | 1.0397 ms | improved |
| 31 | Fuse q0/q1/rank selector construction | — | 1.0400 ms | no-change |
| 32 | Use two sparse producer warpgroups | — | 1.0079 ms | mixed |
| 33 | Restore one-producer sparse pipeline | — | 1.0411 ms | improved |
| 34 | Test N192 sparse consumer tile | 21.40% raw | 1.0410 ms | mixed |

## Iterations

### Iter 1 — Remove residual FP8 WGMMA pass

- **Hypothesis:** A single FP8 approximation, as used by the `megamoe_nvfp4` reference, may satisfy Cupra while removing half of the tensor-core work.
- **Changes:** Added a single-term E2M1×E4M3 conversion, removed the residual shared-memory tile and the second four-instruction WGMMA pass, and reduced requested shared memory from 24 KiB to 16 KiB.
- **Bench:**
  - Compiled: True
  - Correct: True (all 6 Cupra shapes)
  - Large M: 54.7286 ms, 8.79 TFLOPS, 2.97% utilization (8192x4096x7168/E8)
  - Small M: 2.9381 ms (128x4096x7168/E8)
  - Change: -0.28% large-M latency; -0.08% small-M latency
- **Analysis:** Accuracy permits a single term, but runtime is unchanged. Scalar half conversion/dequantization plus serialized shared-memory production dominates; the second asynchronous WGMMA batch was not the limiting stage.
- **Next:** Replace per-value half conversion with the reference branch's constant-memory LUT plus PRMT vector decode, then widen the tile/pipeline after measuring the decode gain.

### Iter 2 — LUT/PRMT vectorized dequant

- **Hypothesis:** Replacing 16 scalar half conversions per scale group with two packed 32-bit decodes and PRMT interleaving should remove the dominant loader cost.
- **Changes:** Imported the reference branch's 1 KiB E2M1×E4M3 LUT, staged it in shared memory, decoded eight packed bytes at a time, and emitted one 128-bit shared store per 16 weights.
- **Bench:**
  - Compiled: True
  - Correct: False (0/6; all outputs contain NaN)
  - Runtime: not measured because the correctness gate failed
- **Analysis:** The first apparent 6/6 and unchanged timing came from stale headers under `build/lib...`; after synchronizing package and source header hashes, the actual candidate compiled but produced NaNs. The reference LUT encodes unscaled E2M1×scale values, whereas this task requires the existing divide-by-eight normalization to keep products within E4M3 range. Large block-scale codes saturate the LUT to FP8 448, and the accumulated products overflow.
- **Next:** Generate/use a divide-by-eight LUT (equivalent to mapping each positive E4M3 scale code down by three exponent steps before the reference PRMT decode), preserve the epilogue ×8 compensation, and re-run the full gate.

### Iter 3 — Normalize LUT by eight

- **Hypothesis:** An offline-generated LUT matching `round_fp8(E2M1 × E4M3 / 8)` should reproduce the single-term scalar path without overflow while retaining packed PRMT decode.
- **Changes:** Regenerated all 128 LUT rows with half-precision multiply and exact `/8` normalization; updated the LUT-free fallback consistently.
- **Bench:**
  - Compiled: True
  - Correct: False (0/6 Cupra shapes)
  - Cosine: 0.999774-0.999777
  - Relative error: 2.1017-2.1169%
  - NaN: False
  - Runtime: not measured because the correctness gate failed
- **Analysis:** The normalized single FP8 term is numerically well behaved but does not satisfy Cupra's elementwise `2 + 5% × |reference|` contract. Iter 1's apparent pass was also from the stale build-package header, so a correction mechanism is required; pure nearest-FP8 rounding is insufficient.
- **Next:** Quantify violation count and error tails, then test whether a cheap selective residual correction can pass while keeping the extra WGMMA fraction small enough to preserve the large-M utilization target.

### Iter 4 — Vectorize exact two-term dequant

- **Hypothesis:** A second signed residual LUT can preserve the original two-term accuracy while removing scalar half arithmetic from the critical path.
- **Changes:** Added a 1 KiB signed residual LUT, changed sign application to XOR, staged both LUTs in shared memory, decoded primary and residual in packed 32-bit groups, restored the residual WGMMA pass, and rebuilt the host extension with 26 KiB dynamic shared memory.
- **Bench:**
  - Compiled: True
  - Correct: True (all 6 Cupra shapes)
  - Large M: 18.0361 ms, 26.67 TFLOPS, 9.01% utilization (8192x4096x7168/E8)
  - Small M: 0.9704 ms, 7.75 TFLOPS, 2.62% utilization (128x4096x7168/E8)
  - Speedup: 3.04x large M; 3.03x small M versus baseline
- **Analysis:** Packed LUT/PRMT decode removes the dominant scalar loader cost while retaining full accuracy. The remaining gap is now structural: one 64-column tile, two serialized WGMMA batches, and a `__syncthreads()` around every K=128 slice.
- **Next:** Widen BLOCK_N and reduce per-tile synchronization/descriptor overhead, then introduce producer/consumer staging to overlap packed load+decode with WGMMA.

### Iter 5 — Widen output tile to N=128

- **Hypothesis:** WGMMA N=128 should halve tile count, expert scans, synchronization, and epilogue overhead without changing arithmetic.
- **Changes:** Changed the compiled tile from 64x64x128 to 64x128x128 and rebuilt the host extension with the corresponding 42 KiB shared-memory request.
- **Bench:**
  - Compiled: True
  - Correct: True (all 6 Cupra shapes)
  - Large M: 13.5865 ms, 35.41 TFLOPS, 11.96% utilization (8192x4096x7168/E8)
  - Small M: 0.6980 ms, 10.77 TFLOPS, 3.64% utilization (128x4096x7168/E8)
  - Speedup: 1.33x versus Iter 4 and 4.04x versus baseline at large M; 4.21x versus baseline at small M
- **Analysis:** Wider N materially improves both regimes, confirming per-tile synchronization/scheduling overhead remains large. Small-M jitter is high (32.78%), so later small-path decisions require repeated isolated runs.
- **Next:** Test N=256 resource viability; if register pressure regresses, keep N=128 and spend the shared-memory budget on a multi-stage producer/consumer pipeline.

### Iter 6 — Widen output tile to N=256

- **Hypothesis:** N=256 may further amortize synchronization and scheduling if its 128-register accumulator does not spill excessively.
- **Changes:** Changed WGMMA and host tile policy to N=256; made each thread decode two B rows cooperatively; rebuilt with a 74 KiB shared-memory request.
- **Bench:**
  - Compiled: True
  - Correct: True (all 6 Cupra shapes)
  - Large M: 12.0831 ms, 39.81 TFLOPS, 13.45% utilization (8192x4096x7168/E8)
  - Small M: 0.8482 ms, 8.86 TFLOPS, 2.99% utilization (128x4096x7168/E8)
  - Change versus Iter 5: 1.12x faster large M; 21.5% slower small M
  - Speedup versus baseline: 4.54x large M; 3.47x small M
- **Analysis:** N=256 is viable and wins at large M, but accumulator/decode resource pressure and reduced tile parallelism hurt small M. One tile policy cannot optimize both regimes.
- **Next:** Compile both N=128 and N=256 specializations and dispatch by M/total-tile count; retain N=256 for large M and N=128 for small M, then pipeline the large path.

### Iter 7 — Dispatch N128/N256 by tile count

- **Hypothesis:** Selecting N=256 only when its estimated tile count fills all SMs will retain large-M throughput without sacrificing small-M parallelism.
- **Changes:** Allowed both template widths, made JIT generation width-specific, and selected N=256 when `ceil(M/64) × (N/256) >= num_sms`, otherwise N=128.
- **Bench:**
  - Compiled: True
  - Correct: True (all 6 Cupra shapes)

| Shape | Runtime | TFLOPS | Util | Baseline speedup |
|---|---:|---:|---:|---:|
| 512x2048x2048/E8 | 0.2413 ms | 17.80 | 6.01% | 3.42x |
| 4096x2048x2048/E8 | 0.9558 ms | 35.95 | 12.15% | 4.20x |
| 8192x4096x2048/E16 | 3.7355 ms | 36.79 | 12.43% | 4.31x |
| 4096x7168x2048/E32 | 3.7564 ms | 32.01 | 10.82% | 4.45x |
| 8192x4096x7168/E8 | 12.0966 ms | 39.77 | 13.43% | 4.54x |
| 128x4096x7168/E8 | 0.7010 ms | 10.72 | 3.62% | 4.20x |

- **Analysis:** The dispatch recovers the N=128 small-M result while retaining N=256 large-M throughput. All shapes now improve by 3.42-4.54x, but large-M useful utilization remains only about 13%, so tile policy alone is exhausted.
- **Next:** Replace the serialized per-K `load/decode → __syncthreads → primary WGMMA wait → residual WGMMA wait` sequence with a multistage producer/consumer pipeline.

### Iter 8 — Merge primary/residual WGMMA batch

- **Hypothesis:** Issuing the primary and residual instructions in one WGMMA commit group should remove one wait/fence sequence per K slice.
- **Changes:** Removed the intermediate commit, accumulator fence, wait, and second warpgroup arrive; all eight WGMMA instructions now share one commit/wait.
- **Bench:**
  - Compiled: True
  - Correct: True (all 6 Cupra shapes)
  - Large M: 12.0926 ms, 39.78 TFLOPS, 13.44% utilization
  - Small M: 0.7030 ms, 10.69 TFLOPS, 3.61% utilization
  - Change: within noise versus Iter 7
- **Analysis:** The extra WGMMA group boundary was fully hidden or negligible. The hard serialization is the CTA-wide load/decode barrier around each K slice.
- **Next:** Implement double-buffered load/decode overlap with asynchronous WGMMA, then move to a dedicated producer warpgroup if compiler scheduling cannot overlap same-warpgroup work.

### Iter 9 — Double-buffer decode under async WGMMA

- **Hypothesis:** WGMMA can execute asynchronously while the same warpgroup loads and decodes the next K slice into a disjoint shared-memory stage.
- **Changes:** Added two A/primary/residual stages, preloaded stage 0, issued the current WGMMA batch, decoded the next slice before `warpgroup_wait<0>`, and doubled dynamic shared-memory staging.
- **Bench:**
  - Compiled: True
  - Correct: True (all 6 Cupra shapes)
  - Large M: 9.0798 ms, 52.98 TFLOPS, 17.90% utilization
  - Small M: 0.5516 ms, 13.63 TFLOPS, 4.60% utilization
  - Speedup: 1.33x large M and 1.27x small M versus Iter 7; 6.05x and 5.33x versus baseline
- **Analysis:** Same-warpgroup overlap is real and valuable, but decode and WGMMA still share one instruction stream and only one WGMMA group can make progress per stage. A dedicated producer is now justified.
- **Next:** Split loading/decode and WGMMA into producer/consumer warpgroups with mbarrier-managed stages; retain the simple 128-thread path for very small workloads if the extra warpgroup hurts latency.

### Iter 10 — Dedicated producer/consumer warpgroups

- **Hypothesis:** A 128-thread producer can continuously decode packed weights while a separate 128-thread consumer issues WGMMA and stores prior tiles.
- **Changes:** Added a large-workload 256-thread specialization with two full/empty mbarrier pairs, producer-only A/B load and dual-LUT decode, consumer-only WGMMA/epilogue, and persistent cross-tile ring reuse. Kept Iter 9's 128-thread path for N=128/small workloads.
- **Bench:**
  - Compiled: True
  - Correct: True (all 6 Cupra shapes)
  - Large M: 7.4629 ms, 64.46 TFLOPS, 21.78% utilization
  - Small M: 0.5511 ms, 13.64 TFLOPS, 4.61% utilization
  - Speedup: 1.22x versus Iter 9 and 7.35x versus baseline at large M; small M retained at 5.34x baseline speedup
- **Analysis:** True warp specialization adds another meaningful gain and safely overlaps across tile boundaries. At 64.46 useful TFLOPS, the two-term algorithm is executing about 128.9 physical FP8 TFLOPS before decode overhead, so tensor-core work is becoming material but not yet saturated.
- **Next:** Tune stage depth/register allocation and reduce producer work. In parallel, pursue a single-WGMMA correction representation because two full WGMMA terms cap useful peak utilization below the requested 80%.

### Iter 11 — Increase large pipeline to 3 stages

- **Hypothesis:** A third 73,728-byte stage may prevent short producer stalls from starving the consumer.
- **Changes:** Increased only the warp-specialized path from two to three stages, using about 223 KiB shared memory; kept the small path at two stages.
- **Bench:**
  - Compiled: True
  - Correct: True (all 6 Cupra shapes)
  - Large M: 7.7505 ms, 62.06 TFLOPS, 20.97% utilization
  - Small M: 0.5520 ms, 13.62 TFLOPS, 4.60% utilization
  - Change: 3.9% slower than Iter 10 at large M; small M unchanged
- **Analysis:** Two stages already cover the producer/consumer distance. The third stage increases shared-memory address span and pipeline bookkeeping without hiding additional latency.
- **Next:** Restore two stages. Reduce producer work per useful tensor operation, starting with weight-layout preprocessing so the timed kernel no longer performs canonical-to-WGMMA swizzle and repeated scale-address arithmetic.

### Iter 12 — Prepack weights for coalesced N reads

- **Hypothesis:** Reordering packed weights to `[E,K/16,N,8]` and scales to `[E,K/16,N]` will turn producer warp loads from K-strided transactions into contiguous N transactions.
- **Changes:** Restored the two-stage large pipeline; added an untimed Cupra `prepare()` transform; extended the public API and JIT specialization to accept either canonical or pretransformed layouts; kept the canonical compatibility path.
- **Bench:**
  - Compiled: True
  - Correct: True (all 6 Cupra shapes)
  - Large M: 4.0113 ms, 119.92 useful TFLOPS, 40.51% useful utilization
  - Small M: 0.4576 ms, 16.43 TFLOPS, 5.55% utilization
  - Speedup: 1.86x versus Iter 10 and 13.68x versus baseline at large M; 6.41x versus baseline at small M

| Shape | Runtime | TFLOPS | Useful util | Baseline speedup |
|---|---:|---:|---:|---:|
| 512x2048x2048/E8 | 0.1579 ms | 27.20 | 9.19% | 5.22x |
| 4096x2048x2048/E8 | 0.3683 ms | 93.30 | 31.52% | 10.89x |
| 8192x4096x2048/E16 | 1.3368 ms | 102.81 | 34.73% | 12.05x |
| 4096x7168x2048/E32 | 1.5436 ms | 77.91 | 26.32% | 10.84x |
| 8192x4096x7168/E8 | 4.0113 ms | 119.92 | 40.51% | 13.68x |
| 128x4096x7168/E8 | 0.4576 ms | 16.43 | 5.55% | 6.43x |
- **Analysis:** Coalescing was the dominant remaining producer optimization. Because exact arithmetic uses two full FP8 WGMMA terms, the large result corresponds to about 239.9 physical FP8 TFLOPS, or 81.0% of H20's 296-TFLOPS FP8 peak. Cupra reports 40.52% because it counts the useful W4A8 FLOPs only once.
- **Next:** Preserve this physical ≥80% large-M result, then reduce or eliminate the residual tensor-core term to raise useful utilization toward the user's 80% target. Re-run all six shapes and continue small-path tuning.

### Iter 13 — Account for expert tails in tile policy

- **Hypothesis:** For small total M, `ceil(M/64)` undercounts tiles because each nonempty expert contributes a partial M tile; including expert tails can safely select N=256 when the real grid nearly fills the GPU.
- **Changes:** Estimated M tiles as `max(ceil(M/64), min(M, num_groups))` and allowed N=256 at 75% of the SM count.
- **Bench:**
  - Compiled: True
  - Correct: True (all 6 Cupra shapes)
  - M128: 0.3880 ms, 19.37 TFLOPS, 6.54% utilization, 2.21% jitter
  - M512: 0.1203 ms, 35.72 TFLOPS, 12.07% utilization
  - Large M: 4.0082 ms, 120.01 useful TFLOPS, 40.54% useful / about 81.1% physical FP8 utilization
  - Speedup versus baseline: 7.58x at M128, 6.85x at M512, 13.69x at the largest shape
- **Analysis:** The tail-aware policy improves the smallest shape by 15% and stabilizes its timing while preserving large-M throughput. Small M still has lower utilization because weights and fixed tile overhead dominate, but latency has moved substantially.
- **Next:** Confirm the small-shape results in the next full sweep, then evaluate cheaper residual correction or structured-sparse residual execution without weakening correctness.

### Iter 14 — Disable producer warpgroup for near-full grids

- **Hypothesis:** M128/N256 has nearly one tile per SM, so the 128-thread same-warpgroup double buffer may avoid producer/mbarrier overhead without losing overlap.
- **Changes:** Kept N256 at the 75%-occupancy threshold, but enabled the dedicated producer only when estimated N256 tiles reach the full SM count.
- **Bench:**
  - Compiled: True
  - Correct: True (all 6 Cupra shapes)
  - M128: 0.3877 ms, 19.38 TFLOPS, 6.55% utilization, 1.94% jitter
  - M512: 0.1627 ms, 26.40 TFLOPS, 8.92% utilization
  - Large M: 4.0074 ms, 120.04 useful TFLOPS, 40.55% useful / about 81.1% physical utilization
- **Analysis:** M128 is statistically identical to Iter 13, so the extra producer warpgroup is unnecessary at this grid size. The prior 0.1203-ms M512 sample did not reproduce; 0.1627 ms is consistent with the Iter 12 full sweep's 0.1579 ms.
- **Next:** Keep the lower-resource small path. Finish authoritative regression/sanitizer checks and document the sparse-residual follow-up separately unless a safe direct-B sparse instruction path is found.

### Iter 15 — Add sparse FP8 WGMMA wrapper and metadata probe

- **Hypothesis:** A direct wrapper around Hopper's `m64n128k64` sparse FP8 WGMMA plus a varying-pattern microprobe can validate the compressed-A descriptor, per-thread metadata word, and output-fragment mapping before integration into the grouped kernel.
- **Changes:** Added sparse FP8 WGMMA selectors for N=32/64/128 and a standalone 64x128x64 probe using B32-swizzled compressed A, B64-swizzled dense B, and metadata words derived from CUTLASS `ELayout_64x64`.
- **Bench:**
  - Compiled: False
  - Correct: Not run
  - Runtime: Not measured
  - Error: standalone inclusion of `sm90.cuh` did not provide `cute::index_sequence` or `deep_gemm::math`, which existing production include order had supplied transitively.
- **Analysis:** This is a header self-containment failure before sparse PTX instantiation, not evidence against the descriptor or metadata design. The probe exposed a latent dependency in the existing MMA header.
- **Next:** Include the integer-sequence and math dependencies directly from `sm90.cuh`, rebuild the probe, then use exact output equality to validate the sparse contract.

### Iter 16 — Make SM90 MMA header self-contained

- **Hypothesis:** Adding the two direct dependencies exposed by Iter 15 will allow the standalone sparse probe to instantiate and execute its WGMMA instruction.
- **Changes:** Included CuTe integer-sequence support and DeepGEMM math utilities directly in `sm90.cuh`.
- **Bench:**
  - C++ compilation: True
  - PTX assembly: False
  - Correct: Not run
  - Error: `wgmma.fence`, `wgmma.commit_group`, and `wgmma.wait_group` were assembled for `.target sm_90` and rejected; Hopper WGMMA requires the architecture-specific `sm_90a` target.
- **Analysis:** Header dependencies are fixed and the sparse wrapper reaches code generation. The standalone NVCC command did not preserve the `a` architecture feature despite its shorthand `-arch` argument, so this remains a probe invocation issue rather than a kernel-layout result.
- **Next:** Compile with explicit `-gencode arch=compute_90a,code=sm_90a` (matching DeepGEMM's JIT flags) and execute the exact-equality probe.

### Iter 17 — Execute sparse WGMMA probe with SM90a target

- **Hypothesis:** Explicit SM90a code generation will execute the sparse instruction and validate the hand-derived B32/B64 shared layouts and `ELayout_64x64` metadata mapping.
- **Changes:** No kernel change; corrected the standalone build command to `-gencode arch=compute_90a,code=sm_90a`.
- **Bench:**
  - Compiled: True
  - Executed: True
  - Correct: False
  - Mismatches: 7120 / 8192
  - Maximum absolute error: 62.25
- **Analysis:** The sparse instruction and register metadata path are executable, but at least one of the compressed-A swizzle, descriptor stride, metadata thread mapping, or expected-value convention is incorrect. The 1072 exact matches make a total descriptor failure unlikely and justify structured row/column diagnostics.
- **Next:** Report mismatch counts and representative actual/reference values by row and column, then compare the observed permutation against CUTLASS's sparse shared-memory atom and metadata layout.

### Iter 18 — Classify sparse probe mismatches

- **Hypothesis:** Per-row and per-column mismatch counts will distinguish a metadata-row mapping error from a shared-memory swizzle or output-fragment permutation.
- **Changes:** Added representative mismatch values plus complete row/column mismatch histograms to the standalone probe.
- **Bench:**
  - Compiled and executed: True
  - Correct: False (unchanged 7120 / 8192 mismatches; maximum absolute error 62.25)
  - Row mismatch counts: every row affected, repeating mostly 96/128 with a four-row-family pattern
  - Column mismatch counts: strict period eight (`32,64,64,64,64,64,64,29`)
- **Analysis:** The deterministic period-eight column signature points to shared-memory address swizzling or descriptor interpretation rather than random metadata corruption. Output fragment stores land in the expected coordinate families, but the hand-written B32/B64 XOR address formulas have not yet been proven against the actual CuTe layouts.
- **Next:** Instantiate CUTLASS `Layout_K_SW32_SpAtom` and `Layout_K_SW64_Atom` directly (or print their coordinate maps) and use those mappings for shared stores and descriptor construction.

### Iter 19 — Correct B32/B64 sparse-probe swizzles

- **Hypothesis:** CuTe `Swizzle<B,4,3>` selects absolute address bits starting at bit seven; for physical row widths below 128 bytes, the XOR source is therefore a higher row bit than the hand-written formulas used.
- **Changes:** Corrected B32 compressed-A swizzling from row bit 0 to row bit 2 and B64 dense-B swizzling from row bits [0,1] to row bits [1,2]. Descriptor strides and metadata mapping were unchanged.
- **Bench:**
  - Compiled and executed: True
  - Correct: True
  - Exact matches: 8192 / 8192
  - Mismatches: 0
  - Maximum absolute error: 0
- **Analysis:** The sparse WGMMA contract is now fully validated for varying metadata patterns, compressed values, dense operands, rows, and columns. The verified per-thread metadata mapping and physical shared layouts can be integrated into the true-NVFP4 grouped path without persistent expanded weights.
- **Next:** Build a dual complementary-2:4 probe from packed NVFP4 plus block scales, keep both decoded terms on chip, combine `acc0 + 3*acc1`, and compare against the dense FP4 reference before adding residual correction.

### Iter 20 — Decode packed NVFP4 into dual sparse WGMMA

- **Hypothesis:** Packed E2M1 weights and their original per-16 E4M3 scales can be decoded entirely inside one CTA into two complementary 2:4 operands whose WGMMA results combine as `acc0 + 3*acc1`.
- **Changes:** Added a standalone probe whose only global weight state is 2048 bytes of packed FP4 plus 256 bytes of original scales. It computes `q0`, `q1`, per-group improvement ranking, both compressed operands, and both metadata streams on chip, then cross-checks dual sparse WGMMA against an independent scalar device reference.
- **Bench:**
  - Compiled and executed: True
  - Dual-sparse contract correct: True
  - Exact matches: 8192 / 8192
  - Maximum WGMMA-versus-reconstruction error: 0
  - Maximum uncorrected reconstruction-versus-original FP4 dot-product error: 1.2578125
  - Persistent weight bytes: 2048 packed FP4 + 256 original scale bytes; no expanded/correction payload
- **Analysis:** The complete true-NVFP4 transient decode path is valid. Both sparse terms and metadata are CTA-local, and the coefficient-three epilogue is exact for this test. The remaining numerical gap is the already-modeled rare residual tail, not a sparse-layout error.
- **Next:** Integrate the K64 dual-sparse path into the grouped kernel with a 64-weight-row by 128-token transposed tile, first without residual correction to validate scheduling and measure its raw performance.

### Iter 21 — Integrate transposed dual-sparse grouped kernel

- **Hypothesis:** A first production-shaped 64-weight-row by 128-token transposed kernel will validate grouped scheduling, multi-K accumulation, prepacked addressing, and BF16 transposed stores before decode/pipeline optimization.
- **Changes:** Added the K64 grouped sparse kernel, on-chip `q0/q1` decomposition, two metadata streams, transposed epilogue, and temporary default dispatch through the existing public API. The exact two-dense implementation remains in-tree as fallback.
- **Bench:**
  - Compiled and focused scheduling checks: True
  - Cupra precision: 5 / 6 shapes passed
  - Largest shape: precision failed only (`cosine=0.999983`, relative error 0.5280%); no performance timing per Cupra policy
  - 512x2048x2048/E8: 3.0836 ms, 1.39 TFLOPS, 0.47%
  - 4096x2048x2048/E8: 11.8827 ms, 2.89 TFLOPS, 0.98%
  - 8192x4096x2048/E16: 45.8879 ms, 3.00 TFLOPS, 1.01%
  - 4096x7168x2048/E32: 55.0091 ms, 2.19 TFLOPS, 0.74%
  - M128: 15.7854 ms, 0.48 TFLOPS, 0.16%
- **Analysis:** Layout, grouped tile mapping, K accumulation, and transposed output are functional, and the lone precision failure matches the numerical prototype's rare residual tail. Performance is unusable because every weight executes float decode, two FP8 conversions, two reverse conversions, squared-error ranking, and serialized K-stage barriers. Sparse tensor-core savings are completely hidden by scalar producer work.
- **Next:** Replace scalar decomposition with packed `q0`, `q1`, and improvement-rank LUT/PRMT decoding, preserving the same transient sparse representation; then remeasure before adding the residual tail.

### Iter 22 — Replace scalar decomposition with generated LUTs

- **Hypothesis:** Model-independent 128x8 `q1` and improvement-rank LUTs, staged beside the existing `q0` LUT, can replace all per-weight floating conversion and squared-error work with packed PRMT decoding and byte comparisons.
- **Changes:** Added constexpr E4M3 table generation, a packed rank decoder, shared staging for three 1 KiB universal LUTs, and a 16-weight-at-a-time producer loop using two 32-bit packed loads per scale block.
- **Bench:**
  - Compiled: False
  - Correct/performance: Not run
  - NVCC error: dynamic initialization is not supported for the two `__constant__` variables initialized directly from constexpr generator function calls.
- **Analysis:** The hot-loop rewrite parses, but NVCC does not accept the current namespace-scope constant-memory initialization form. This is a declaration/materialization issue before kernel code generation, not a rejection of the packed decode algorithm.
- **Next:** Materialize each generated table as an `inline constexpr` aggregate first (or use an explicit aggregate initializer), initialize device constant memory from that compile-time object, and validate table output against the scalar decomposition probe.

### Iter 23 — Materialize generated LUT initializers

- **Hypothesis:** Naming the generated aggregates as `inline constexpr` objects before constant-memory initialization will force static initialization accepted by NVCC.
- **Changes:** Split each generator result into a named constexpr initializer and used those objects to initialize the aligned device constant tables.
- **Bench:**
  - Compiled: False
  - Correct/performance: Not run
  - NVCC error: both generator expressions exceeded the compiler's constexpr function-call complexity limit and therefore were not folded to constant values.
- **Analysis:** Static initialization form is no longer the issue; exhaustive compile-time E4M3 nearest-code searches are simply too expensive at JIT compile time. Raising the global compiler limit would slow and couple unrelated kernels.
- **Next:** Replace generator calls with explicit generated `uint2[128]` aggregate data for `q1` and improvement ranks, preserving the same universal 2 KiB constant footprint and packed hot loop.

### Iter 24 — Use explicit packed sparse decode LUTs

- **Hypothesis:** Explicit generated constant tables will avoid JIT constexpr limits and make the packed PRMT producer materially faster than Iter 21's scalar decomposition.
- **Changes:** Replaced compile-time searches with explicit 128-row `q1` and improvement-rank tables, retained per-CTA shared staging, and kept the 16-weight packed decode loop unchanged.
- **Bench:**
  - Compiled: True
  - Focused scheduling/numerics: unchanged and correct
  - Cupra precision: 5 / 6 shapes passed; largest remains the expected uncorrected-tail precision failure (`cosine=0.999983`, relative error 0.5280%)
  - 512x2048x2048/E8: 1.2459 ms, 3.45 TFLOPS, 1.16% (2.47x faster than Iter 21)
  - 4096x2048x2048/E8: 5.1016 ms, 6.74 TFLOPS, 2.28% (2.33x)
  - 8192x4096x2048/E16: 18.4916 ms, 7.43 TFLOPS, 2.51% (2.48x)
  - 4096x7168x2048/E32: 21.5470 ms, 5.58 TFLOPS, 1.89% (2.55x)
  - M128: 5.6098 ms, 1.34 TFLOPS, 0.45% (2.81x)
- **Analysis:** Packed decoding is substantially faster but still far from viable. The simple N128 path keeps two 64-register WGMMA fragments plus decode state in every thread; likely register spills and serialized decode/WGMMA execution now dominate. This must be measured before further arithmetic tuning.
- **Next:** Compile with ptxas verbose resource reporting and inspect registers, stack, and spills; then either reduce token width to N64 or split decode into a dedicated producer warpgroup so consumer threads do not carry producer state.

### Iter 25 — Add dual-warpgroup sparse pipeline

- **Hypothesis:** A 128-thread packed-decode producer, a 128-thread sparse-WGMMA consumer, two mbarrier stages, and Hopper register reallocation will overlap the serialized work identified in Iter 24 while retaining N128 tensor-core efficiency.
- **Changes:** Added a 256-thread sparse specialization with two shared stages, persistent producer/consumer tile traversal, full/empty barriers, producer register deallocation to 80, consumer allocation to 248, and host dispatch/shared-memory sizing for the new kernel.
- **Bench:**
  - Compiled: False
  - Correct/performance: Not run
  - NVCC errors: direct declarations for `warpgroup_reg_alloc/dealloc` and `ptx::sync_aligned` were unavailable because the new header did not include their defining register-reconfiguration and PTX utility headers.
- **Analysis:** This is a missing direct-include failure before code generation; pipeline logic and resources have not yet been exercised.
- **Next:** Include CUTLASS register reconfiguration and DeepGEMM PTX utility definitions explicitly, then rerun focused correctness and resource reporting.

### Iter 26 — Run dual-warpgroup sparse pipeline

- **Hypothesis:** With direct register-reconfiguration and named-barrier includes, the two-stage producer/consumer implementation will overlap packed decode with sparse WGMMA.
- **Changes:** Added the missing direct includes; no pipeline logic change from Iter 25.
- **Bench:**
  - Compiled: True
  - Focused correctness: unchanged and correct
  - Cupra precision: 5 / 6; largest remains the expected uncorrected-tail failure
  - 512x2048x2048/E8: 1.3264 ms, 3.24 TFLOPS, 1.09%
  - 4096x2048x2048/E8: 5.3574 ms, 6.41 TFLOPS, 2.17%
  - 8192x4096x2048/E16: 19.5904 ms, 7.02 TFLOPS, 2.37%
  - 4096x7168x2048/E32: 22.8535 ms, 5.26 TFLOPS, 1.78%
  - M128: 6.0962 ms, 1.23 TFLOPS, 0.42%
  - Change versus Iter 24: 5-9% slower across timed shapes
- **Analysis:** Splitting warpgroups does not help because the producer is much slower than the two sparse WGMMA instructions; the consumer remains starved and barrier/register overhead adds a regression. The packed producer still uses dynamic four-element arrays, variable indexing, ranking loops, complement construction, and byte scatter for every 2:4 group.
- **Next:** Audit compiled resources/instruction shape, then replace per-group dynamic selection with a compact model-independent selection LUT or branchless bit operations that directly emit metadata and compressed-byte selectors.

### Iter 27 — Remove dynamic local arrays from sparse producer

- **Hypothesis:** The producer's 99 `LDL.U8` plus additional `LDL/STL` instructions come from dynamic four-element arrays and variable indexing; scalar sorting and PRMT compression should remove local-stack traffic and feed the consumer faster.
- **Changes:** Added a register-only five-comparator sorting network with deterministic tie-breaking, derived selected/complement masks via bit operations, compressed each sparse pair with PRMT, emitted 16-bit shared stores, and used the same helper in simple and warp-specialized paths.
- **Bench:**
  - Compiled: True
  - Focused correctness: unchanged and correct
  - Cupra precision: 5 / 6; largest remains the expected uncorrected-tail failure
  - 512x2048x2048/E8: 0.8762 ms, 4.90 TFLOPS, 1.66% (1.51x faster than Iter 26)
  - 4096x2048x2048/E8: 3.7846 ms, 9.08 TFLOPS, 3.07% (1.42x)
  - 8192x4096x2048/E16: 13.1510 ms, 10.45 TFLOPS, 3.53% (1.49x)
  - 4096x7168x2048/E32: 14.6512 ms, 8.21 TFLOPS, 2.77% (1.56x)
  - M128: 3.5266 ms, 2.13 TFLOPS, 0.72% (1.73x)
- **Analysis:** Removing dynamic local indexing gives a large, broad gain and validates the SASS diagnosis, but producer throughput remains far below the tensor-core target. Per group still requires rank PRMT, a sorting network, two compression PRMTs, and metadata construction.
- **Next:** Reinspect SASS for residual local traffic and instruction counts; encode the selected pair more directly (for example, a compact rank-order/select table) and vectorize shared stores across adjacent groups.

### Iter 28 — Move top-2 selection into shared lookup

- **Hypothesis:** A model-independent 4096-entry shared table indexed by four 3-bit ranks can replace the per-group sorting network and index extraction; metadata nibbles can then directly drive PRMT compression from the non-interleaved decode registers.
- **Changes:** Cooperatively generate the 4 KiB selection table once per persistent CTA, encode both q0/q1 metadata nibbles per entry, replace hot-loop sorting/`ffs` work with one shared byte lookup, and remove two q0/q1 interleave PRMTs per group by addressing low/high decode registers directly.
- **Bench:**
  - Compiled: True
  - Focused correctness: unchanged and correct
  - Cupra precision: 5 / 6; largest remains the expected uncorrected-tail failure
  - 512x2048x2048/E8: 0.7942 ms, 5.41 TFLOPS, 1.83% (1.10x faster than Iter 27)
  - 4096x2048x2048/E8: 3.5070 ms, 9.80 TFLOPS, 3.31% (1.08x)
  - 8192x4096x2048/E16: 11.9957 ms, 11.46 TFLOPS, 3.87% (1.10x)
  - 4096x7168x2048/E32: 13.2330 ms, 9.09 TFLOPS, 3.07% (1.11x)
  - M128: 3.0802 ms, 2.44 TFLOPS, 0.82% (1.15x)
- **Analysis:** The shared selection lookup consistently improves producer throughput, but decode remains overwhelmingly dominant. Every eight packed weights still traverse three independent magnitude LUT streams (`q0`, `q1`, rank), six PRMTs for those streams, then six PRMTs to compress two groups.
- **Next:** Fuse q0/q1/rank records or precompute a more direct packed decode representation so one magnitude selection supplies all three fields, then vectorize adjacent sparse shared stores and metadata assembly.

### Iter 29 — Split each weight row across two producer threads

- **Hypothesis:** Sparse metadata already has one independent 32-bit word per K32 half, so two producer threads can decode disjoint halves of each weight row without atomics and nearly halve producer critical-path depth.
- **Changes:** Mapped all 128 producer threads to `(weight_row, K32 half)`, reduced each thread from four to two scale blocks, wrote one independent metadata word, and applied the same mapping to simple and warp-specialized paths.
- **Bench:**
  - Compiled: True
  - Focused correctness: unchanged and correct
  - Cupra precision: 5 / 6; largest remains the expected uncorrected-tail failure
  - 512x2048x2048/E8: 0.6890 ms, 6.23 TFLOPS, 2.11% (1.15x faster than Iter 28)
  - 4096x2048x2048/E8: 3.2105 ms, 10.70 TFLOPS, 3.62% (1.09x)
  - 8192x4096x2048/E16: 10.7830 ms, 12.75 TFLOPS, 4.31% (1.11x)
  - 4096x7168x2048/E32: 11.4804 ms, 10.48 TFLOPS, 3.54% (1.15x)
  - M128: 2.4294 ms, 3.09 TFLOPS, 1.05% (1.27x)
- **Analysis:** Full producer occupancy helps every shape, but the gain is below 2x, so activation movement, synchronization, sparse decode primitives, and transposed output stores now contribute materially alongside per-row decode.
- **Next:** Measure producer-ready versus consumer-complete stage timing (or controlled A/B variants) to quantify decode/load, sparse WGMMA, barrier, and epilogue shares before the next structural optimization.

### Iter 30 — Vectorize activation staging with uint4

- **Hypothesis:** The B64 shared layout preserves each aligned 16-byte activation chunk, so replacing scalar byte copies with `uint4` should reduce every producer thread from 64 load/store iterations to four for an N128xK64 stage.
- **Changes:** Replaced scalar activation `LDG.U8/STS.U8` loops in simple and warp-specialized kernels with predicated 128-bit global loads and swizzled 128-bit shared stores.
- **Bench:**
  - Compiled: True
  - Focused correctness: unchanged and correct
  - Cupra precision: 5 / 6; largest remains the expected uncorrected-tail failure
  - 512x2048x2048/E8: 0.2103 ms, 20.43 TFLOPS, 6.90% (3.28x faster than Iter 29)
  - 4096x2048x2048/E8: 0.6809 ms, 50.46 TFLOPS, 17.05% (4.72x)
  - 8192x4096x2048/E16: 2.4856 ms, 55.29 TFLOPS, 18.68% (4.34x)
  - 4096x7168x2048/E32: 3.3681 ms, 35.71 TFLOPS, 12.06% (3.41x)
  - M128: 1.0397 ms, 7.23 TFLOPS, 2.44% (2.34x)
- **Analysis:** Scalar activation movement was a dominant producer bottleneck. With it removed, the dual-sparse path enters a useful throughput regime, though it is still slower than the exact dense fallback and far below the 80% target. Packed weight decode/selection and producer-consumer balance are now the main candidates.
- **Next:** Measure the raw largest-shape latency despite its expected precision failure, inspect updated SASS/runtime balance, and vectorize or restructure packed weight decode before integrating the residual tail.

### Iter 31 — Fuse q0/q1/rank selector construction

- **Hypothesis:** Computing packed magnitude selectors once per 32-bit word and reusing them for q0, q1, and rank PRMT streams will remove duplicated selector construction in the producer.
- **Changes:** Added one fused sparse-word decoder that computes high/low selectors and sign masks once, emits all six packed q0/q1/rank registers, and feeds the existing per-group compressor.
- **Bench:**
  - Compiled: True
  - Correctness/precision: unchanged (focused correct; Cupra 5 / 6 before residual)
  - 512x2048x2048/E8: 0.2102 ms (vs 0.2103)
  - 4096x2048x2048/E8: 0.6778 ms (vs 0.6809)
  - 8192x4096x2048/E16: 2.4863 ms (vs 2.4856)
  - 4096x7168x2048/E32: 3.3711 ms (vs 3.3681)
  - M128: 1.0400 ms (vs 1.0397)
  - Largest raw diagnostic from Iter 30 code: 7.7152 ms, 62.35 TFLOPS, 21.06% useful utilization
- **Analysis:** Results are statistically identical, indicating NVCC already common-subexpression-eliminated selector construction across the three inlined decoders. Arithmetic fusion alone will not close the gap.
- **Next:** Use two producer warpgroups and map four producer threads per weight row (one K16 scale block each), while retaining one N128 consumer warpgroup and two shared stages.

### Iter 32 — Use two sparse producer warpgroups

- **Hypothesis:** Two producer warpgroups can map four threads per weight row (one K16 scale block each), halve decode depth again, and keep one N128 consumer fed.
- **Changes:** Expanded the CTA to 384 threads, changed full-barrier arrivals to two, distributed activation chunks across 256 producer threads, assigned one scale block per producer thread, and wrote independent 16-bit metadata quarters.
- **Bench:**
  - Compiled: True
  - Focused correctness: unchanged and correct
  - Cupra precision: 5 / 6; largest remains the expected uncorrected-tail failure
  - 512x2048x2048/E8: 0.2279 ms (8.4% slower than Iter 31)
  - 4096x2048x2048/E8: 0.8136 ms (20.0% slower)
  - 8192x4096x2048/E16: 2.8094 ms (13.0% slower)
  - 4096x7168x2048/E32: 3.4911 ms (3.6% slower)
  - M128: 1.0079 ms (3.1% faster)
- **Analysis:** Extra decode parallelism helps only the long-K smallest-M case. For larger workloads, a 384-thread CTA, extra producer barrier arrival, and reduced consumer/warp scheduling resources cost more than the shorter decode path. This is not the large-M configuration.
- **Next:** Restore one producer warpgroup for medium/large dispatch; optionally retain a separate dual-producer small-M specialization only if it remains useful after token-width tuning.

### Iter 33 — Restore one-producer sparse pipeline

- **Hypothesis:** Restoring the Iter 31 single-producer configuration will recover medium/large throughput; the 3% M128 gain from a second producer is not worth retaining as duplicate complexity before small-tile tuning.
- **Changes:** Mechanically restored the 256-thread, one-producer/one-consumer specialization, one full-barrier arrival, two K16 scale blocks per producer thread, and the original launch policy.
- **Bench:**
  - Compiled: True
  - Correctness/precision: unchanged (focused correct; Cupra 5 / 6 before residual)
  - 512x2048x2048/E8: 0.2097 ms, 20.48 TFLOPS, 6.92%
  - 4096x2048x2048/E8: 0.6787 ms, 50.63 TFLOPS, 17.10%
  - 8192x4096x2048/E16: 2.4859 ms, 55.29 TFLOPS, 18.68%
  - 4096x7168x2048/E32: 3.3725 ms, 35.66 TFLOPS, 12.05%
  - M128: 1.0411 ms, 7.22 TFLOPS, 2.44%
- **Analysis:** Results reproduce Iter 30/31 and confirm the extra producer should be discarded for the large path. Weight decode is now best amortized by increasing useful token work per decoded weight tile, subject to consumer register limits.
- **Next:** Test a sparse WGMMA token width above 128 (starting at N192) to reduce weight-decode repetitions by up to 1.5x on large expert segments; dispatch narrower tiles for small/tail-heavy workloads.

### Iter 34 — Test N192 sparse consumer tile

- **Hypothesis:** N192 can amortize each packed weight decode across 1.5x more tokens while keeping two 96-register accumulator fragments within the 248-register consumer allocation.
- **Changes:** Added the sparse `m64n192k64` wrapper and selected token width 192 when average tokens per expert reached 192; retained N128 for tail-heavy/small shapes.
- **Bench:**
  - Compiled: True
  - Correctness/precision: unchanged (focused correct; Cupra 5 / 6 before residual)
  - 4096x2048x2048/E8: 0.7152 ms versus 0.6787 ms (5.4% slower)
  - 8192x4096x2048/E16: 2.4889 ms versus 2.4859 ms (no change)
  - Other timed shapes stayed on N128 and were unchanged
  - Largest raw: 7.5929 ms, 63.35 TFLOPS, 21.40% useful versus 62.35 TFLOPS / 21.06% for N128 (1.6% faster)
- **Analysis:** The wider single consumer is register/issue constrained and does not recover the expected decode amortization. Its tiny largest-shape gain does not justify medium-shape regressions as a general policy.
- **Next:** Keep WGMMA N128 per consumer but use two consumer warpgroups over a 256-token CTA tile, sharing one producer decode and activation stage; this amortizes weights without doubling per-thread accumulators.

### Iteration 35 — Two N128 consumer warp-groups per CTA

- Change: generalized the warp-specialized packed-NVFP4 kernel to let one producer warp-group feed two independent N128 consumer warp-groups (384 threads/CTA), with per-consumer activation descriptors and output offsets.
- Focused benchmark: `M=4096, N=2048, K=2048, E=8`.
- Result: failed — the kernel held GPU 0 at 100% utilization for over 90 seconds and produced no benchmark output; the run was terminated. This is a synchronization/barrier deadlock, so no latency or TFLOP/s number is reportable.
- Decision: do not use the two-consumer policy. Restore the known-good one-producer/one-consumer path before making the next isolated scheduling change.

### Iteration 36 — Fit dual-consumer SETMAXNREG budget

- Change: reduced the 384-thread CTA producer warp-group from 64 to 56 registers so that two 224-register consumers plus the producer fit the launch-bounds allocation (64.5K registers).
- Focused benchmark: `M=4096, N=2048, K=2048, E=8`, isolated after removing the stale Iter35 container.
- Result: failed — the isolated kernel again held GPU 0 at 100% utilization with no output for over 30 seconds and was terminated. The register-pool overcommit was real, but it was not the only deadlock in the two-consumer synchronization model.
- Decision: restore the one-consumer runtime policy and keep multi-consumer disabled until its barriers can be validated in a small standalone probe.

### Iteration 37 — Restore validated single-consumer policy

- Change: forced the runtime policy back to one N128 consumer warp-group and rebuilt both the extension and packaged headers inside the Python 3.12 `lmsysorg/sglang:dev` environment actually used by Cupra.
- Focused result (`4096x2048x2048/E8`): **0.6773 ms, 50.73 TFLOP/s, 17.14% useful utilization, PASS**.
- This reproduces the Iter30/33 baseline (about 0.68 ms) and confirms the active benchmark environment is clean.
- Audit note: Iter36's source change was copied by a host Python 3.10 build and therefore did not enter the Python 3.12 package used by that run; its claimed 56-register runtime validation is invalid. Iter35's original dual-consumer deadlock remains valid, but Iter36 must not be used as evidence against the 56-register fix.

### Iteration 38 — Valid Python 3.12 dual-consumer retest

- Change: re-enabled two N128 consumer warp-groups for average expert sizes of at least 256 tokens, using the 56-register producer from Iter36; rebuilt the active Python 3.12 extension and used a fresh JIT cache.
- Focused result (`4096x2048x2048/E8`): **0.6568 ms, 52.32 TFLOP/s, 17.67% useful utilization, PASS**.
- Versus the clean one-consumer Iter37 baseline (0.6773 ms), this is **3.0% lower latency / 3.1% higher throughput**.
- Conclusion: the Iter35 deadlock was the CTA register-budget overcommit. The 56-register producer makes the dual-consumer pipeline valid, although the modest gain confirms producer spilling/decode pressure still dominates.

#### Iteration 38 full-suite validation

| Shape | Time | Useful TFLOP/s | Status |
|---|---:|---:|---|
| `512x2048x2048/E8` | 0.2099 ms | 20.46 | PASS |
| `4096x2048x2048/E8` | 0.6593 ms | 52.12 | PASS |
| `8192x4096x2048/E16` | 2.2766 ms | 60.37 | PASS |
| `4096x7168x2048/E32` | 3.3676 ms | 35.71 | PASS |
| `8192x4096x7168/E8` | — | — | PRECFAIL (cosine 0.999983, relative error 0.5280%) |
| `128x4096x7168/E8` | 1.0402 ms | 7.23 | PASS |

- The 256-token reuse improves the E16 medium-large case from Iter30's 2.4856 ms to **2.2766 ms (8.4%)**. E32 and small-M shapes remain effectively unchanged because their per-expert token counts do not select a full dual-consumer tile.

### Iteration 39 — Coalesce transformed packed-weight reads

- Change: for pretransformed `[scale_k, N, 8-byte]` weights, mapped each producer warp to 32 consecutive weight rows at one K-scale half instead of interleaving lanes across two distant scale planes. Canonical row-major weights keep the old mapping.
- Focused result (`8192x4096x2048/E16`): **2.2563 ms, 60.91 TFLOP/s, 20.58% useful utilization, PASS**.
- Versus Iter38 full-suite result (2.2766 ms), latency improves **0.9%**. The packed/global loads are now coalesced, but the small gain shows instruction-heavy LUT/PRMT/metadata construction remains the main producer cost.

### Iteration 40 — Precompute complete sparse selection plans

- Change: expanded the transient shared selection LUT from 4 KiB of metadata bytes to 16 KiB of 32-bit plans. Each lookup now returns both sparse metadata nibbles and the precomputed Q0/Q1 `PRMT` source selectors, removing per-group position-to-byte index reconstruction. The table is generated once per CTA in shared memory; persistent packed FP4 weights are unchanged.
- Focused result (`8192x4096x2048/E16`): **2.1531 ms, 63.83 TFLOP/s, 21.57% useful utilization, PASS**.
- Versus Iter39 (2.2563 ms), latency improves **4.6%** and throughput improves **4.8%**. This confirms sparse selection/packing integer instructions are a material producer bottleneck.

### Iteration 41 — Fixed complementary 2:4 partition

- Change: replaced dynamic top-two selection with constant Q0 positions `{0,1}` and `3×Q1` positions `{2,3}`. This removes rank/plan LUT initialization and lookup, makes both metadata registers constant, and packs both consecutive groups with one `PRMT` per term.
- Full-suite correctness: **0/6 passed**. Cosine remains about 0.999835–0.999839, but relative error rises to 1.778–1.800%, well beyond the 0.5% limit.
- Conclusion: the fixed partition is useful as a decode-throughput probe but cannot be a production approximation without correction. Dynamic per-group selection is quantitatively necessary; retain Iter40 as the correctness baseline.

#### Iteration 41 raw throughput probe

- Ignoring the known precision failure, `8192x4096x2048/E16` runs in **1.6576 ms / 82.91 useful TFLOP/s**.
- This is 23.0% lower latency and 29.9% higher throughput than Iter40's dynamic optimal selection (2.1531 ms / 63.83 TFLOP/s). The recovered headroom justifies searching for a cheaper selector, but fixed partition alone is not admissible.

### Iteration 42 — Direct mantissa/magnitude selection-plan table

- Change: generated a 32K-entry, 128 KiB shared plan table indexed by E4M3 scale mantissa and four FP4 magnitudes. Normal scales bypass rank decoding and combine two adjacent 2:4 groups into one Q0 and one Q1 `PRMT`; scales below code 64 retain the exact Iter40 fallback.
- Focused result (`8192x4096x2048/E16`): **2.3094 ms, 59.51 TFLOP/s, 20.11% useful utilization, PASS**.
- This is **7.3% slower** than Iter40 (2.1531 ms). The 128 KiB per-CTA table construction and random shared footprint outweigh the hot-loop instruction reduction. Reject this materialized direct-table design; a useful selector must stay compact.

### Iteration 43 — Combine adjacent groups after compact plan lookup

- Change: restored Iter40's compact 16 KiB rank-indexed plan table, then looked up both adjacent 2:4 group plans before packing. A single 32-bit `PRMT` now emits both groups for Q0 and another emits both for Q1; shared writes are two 32-bit stores instead of four 16-bit stores. Selection and metadata remain identical to Iter40.
- Focused result (`8192x4096x2048/E16`): **1.9103 ms, 71.95 TFLOP/s, 24.31% useful utilization, PASS**.
- Versus Iter40 (2.1531 ms / 63.83 TFLOP/s), latency improves **11.3%** and throughput improves **12.7%** while retaining correctness. This is the new sparse baseline.

#### Iteration 43 full-suite validation

| Shape | Time | Useful TFLOP/s | Status |
|---|---:|---:|---|
| `512x2048x2048/E8` | 0.1657 ms | 25.92 | PASS |
| `4096x2048x2048/E8` | 0.5555 ms | 61.85 | PASS |
| `8192x4096x2048/E16` | 1.9047 ms | 72.16 | PASS |
| `4096x7168x2048/E32` | 2.5730 ms | 46.74 | PASS |
| `8192x4096x7168/E8` | — | — | PRECFAIL (cosine 0.999983, relative error 0.5280%) |
| `128x4096x7168/E8` | 0.7855 ms | 9.57 | PASS |

- Relative to Iter38, latency improves 21.1% (M512), 15.7% (M4096/E8), 16.3% (E16), 23.6% (E32), and 24.5% (M128). The same packing optimization benefits both one- and two-consumer paths.

### Iteration 44 — Two N192 consumers for large experts

- Change: enabled the existing sparse WGMMA N192 wrapper in the warp-specialized kernel and dispatch two N192 consumers (384-token tile) when average tokens per expert are at least 384. The CTA remains 384 threads with the validated 224/56 register split.
- Focused result (`8192x4096x2048/E16`): **1.7359 ms, 79.18 TFLOP/s, 26.75% useful utilization, PASS**.
- Versus Iter43 N128×2 (1.9103 ms / 71.95 TFLOP/s), latency improves **9.1%** and throughput improves **10.1%** by amortizing each packed-weight decode over 50% more tokens.

#### Iteration 44 full-suite validation

| Shape | Time | Useful TFLOP/s | Status |
|---|---:|---:|---|
| `512x2048x2048/E8` | 0.1657 ms | 25.93 | PASS |
| `4096x2048x2048/E8` | 0.4989 ms | 68.87 | PASS |
| `8192x4096x2048/E16` | 1.7343 ms | 79.25 | PASS |
| `4096x7168x2048/E32` | 2.5722 ms | 46.75 | PASS |
| `8192x4096x7168/E8` | — | — | PRECFAIL (cosine 0.999983, relative error 0.5280%) |
| `128x4096x7168/E8` | 0.7840 ms | 9.59 | PASS |

- N192×2 improves the two selected K=2048 large-expert cases by 10.2% (E8) and 9.8% (E16) over Iter43. Narrow/tail-heavy cases remain statistically unchanged.

### Iteration 45 — Three N128 consumer warp-groups

- Change: tested a 512-thread CTA with three N128 consumers and one producer for the same 384-token reuse as Iter44, initially budgeting 156 registers per consumer and 36 for the producer.
- Result: compile failed. PTX `setmaxnreg.inc/dec` immediates must be multiples of 8; ptxas rejected both 156 and 36 before any kernel ran.
- Decision: retry only with legal aligned budgets. The safe candidate is 152 consumer / 40 producer (63.5K total); avoid the exact-capacity 160/32 split that could reproduce the Iter35 allocation wait.

### Iteration 46 — Legal triple-consumer register split

- Change: retried the 512-thread N128×3 CTA with legal aligned `setmaxnreg` values: 152 registers per consumer and 40 for the producer (63.5K total).
- Focused result (`8192x4096x2048/E16`): **5.7879 ms, 23.75 TFLOP/s, 8.02% useful utilization, PASS**.
- This is 3.34× slower than Iter44 N192×2. The N128 consumer needs substantially more than 152 registers for two accumulator fragments plus epilogue state; register pressure destroys throughput even though no correctness issue appears. Reject the three-consumer configuration and restore N192×2.

### Iteration 47 — Exact packed-NVFP4 fallback for long K

- Change: restored the Iter44 N192×2 sparse policy for K=2048 and dispatched K=7168 to the existing exact dense primary+residual kernel. Both paths keep only canonical/pretransformed packed FP4 plus original scales in global memory and perform all FP8 expansion transiently on chip.
- Full suite: **6/6 PASS** — the branch now has no precision failure.

| Shape | Time | Useful TFLOP/s | Utilization |
|---|---:|---:|---:|
| `512x2048x2048/E8` | 0.1655 ms | 25.95 | 8.77% |
| `4096x2048x2048/E8` | 0.4963 ms | 69.23 | 23.39% |
| `8192x4096x2048/E16` | 1.7369 ms | 79.13 | 26.73% |
| `4096x7168x2048/E32` | 2.5697 ms | 46.80 | 15.81% |
| `8192x4096x7168/E8` | 4.0114 ms | 119.92 | 40.51% |
| `128x4096x7168/E8` | 0.3880 ms | 19.37 | 6.54% |

- Compared with Iter44's uncorrected sparse K=7168 path, this removes the 0.528% precision failure; versus its earlier raw sparse timing (~6 ms after Iter43 packing, inferred), the exact dense path is also materially faster. The reported 40.51% useful utilization corresponds to ~81.0% of H20 dense-FP8 physical work because the exact decomposition executes two dense terms.

### Iteration 48 — Consumer-side activation prefetch

- Change: moved sparse activation staging out of the weight-decode producer. Each consumer warp-group now loads its own activation slice and, after issuing current-stage WGMMA, prefetches the next stage while tensor-core work is in flight. Full barriers combine one weight arrival with one activation arrival per consumer; the producer is dedicated to packed-FP4 decode.
- Focused result (`8192x4096x2048/E16`): **1.6165 ms, 85.02 TFLOP/s, 28.72% useful utilization, PASS**.
- Versus Iter44/47 (1.7343–1.7369 ms / ~79.2 TFLOP/s), latency improves **6.8%** and throughput improves **7.4%**. The barrier redesign is correct and directly shortens the producer critical path.

#### Iteration 48 full-suite validation

| Shape | Time | Useful TFLOP/s | Utilization | Status |
|---|---:|---:|---:|---|
| `512x2048x2048/E8` | 0.1495 ms | 28.73 | 9.71% | PASS |
| `4096x2048x2048/E8` | 0.5053 ms | 67.99 | 22.97% | PASS |
| `8192x4096x2048/E16` | 1.6163 ms | 85.04 | 28.73% | PASS |
| `4096x7168x2048/E32` | 2.4420 ms | 49.25 | 16.64% | PASS |
| `8192x4096x7168/E8` | 4.0137 ms | 119.85 | 40.49% | PASS |
| `128x4096x7168/E8` | 0.3885 ms | 19.35 | 6.54% | PASS |

- Versus Iter47, M512 improves 10.7%, E16 improves 7.5%, and E32 improves 5.2%; dense K=7168 is unchanged. M4096/E8 is 1.8% slower in this noisy run (11.2% jitter), so its dispatch needs a repeated focused comparison before changing policy.

### Iteration 49 — Use idle register budget in single-consumer producer

- Change: raised the one-consumer sparse producer allocation from 80 to 128 registers. The two-warp-group CTA has ample total register capacity, so the N128 consumer remains at 248 and the dual-N192 policy is untouched.
- Focused result (`512x2048x2048/E8`): **0.1471 ms, 29.20 TFLOP/s, 9.87% useful utilization, PASS**.
- Versus Iter48 (0.1495 ms / 28.73 TFLOP/s), latency improves **1.6%**. The gain is small but consistent with decode benefiting from the otherwise unused register budget.

### Iteration 50 — Copy a precomputed universal selection-plan table

- Change: replaced per-CTA generation of all 4096 sparse selection plans with a model-independent 16 KiB read-only table compiled into the kernel module. CTAs copy it to shared memory using coalesced `uint4` transfers; no model weight, correction index, or expanded payload is stored.
- Focused result (`512x2048x2048/E8`): **0.1458 ms, 29.45 TFLOP/s, 9.95% useful utilization, PASS**.
- Versus Iter49 (0.1471 ms), latency improves **0.9%** and jitter drops from 4.0% to 2.6%. The startup saving is modest but positive on the smallest K=2048 case.

#### Iteration 50 full-suite validation (current best)

| Shape | Time | Useful TFLOP/s | Utilization | Status |
|---|---:|---:|---:|---|
| `512x2048x2048/E8` | 0.1461 ms | 29.39 | 9.93% | PASS |
| `4096x2048x2048/E8` | 0.5012 ms | 68.55 | 23.16% | PASS |
| `8192x4096x2048/E16` | 1.6112 ms | 85.30 | 28.82% | PASS |
| `4096x7168x2048/E32` | 2.4066 ms | 49.97 | 16.88% | PASS |
| `8192x4096x7168/E8` | 4.0156 ms | 119.79 | 40.47% | PASS |
| `128x4096x7168/E8` | 0.3883 ms | 19.36 | 6.54% | PASS |

- All 4096 precomputed entries reproduce the runtime plan generator across the six correctness workloads. Relative to Iter48, E16 improves another 0.3% and E32 1.5%; other differences are within run jitter.

### Iteration 51 — Fuse Q0/Q1 shared decode records

- Change: interleaved the two per-scale 64-bit Q0/Q1 lookup rows into one 128-bit shared record in the warp-specialized kernel, while leaving the 64-bit rank row and all selection math unchanged.
- Focused result (`8192x4096x2048/E16`): **1.6107 ms, 85.33 TFLOP/s, 28.83% useful utilization, PASS**.
- This is statistically identical to Iter50 (1.6112 ms). NVCC/shared-memory issue width already hides or combines the adjacent 64-bit accesses; retain the cleaner fused record but do not expect further gains from LUT load width alone.

### Iteration 52 — Shared tile-prefix binary search

- Change: built a 33-entry per-CTA prefix of expert tile counts and replaced each warp-group's linear expert scan with binary search.
- Focused result (`8192x4096x2048/E16`): **1.6710 ms, 82.25 TFLOP/s, 27.79% useful utilization, PASS**.
- This is **3.7% slower** than Iter51 (1.6107 ms). With only 16 experts and persistent tile order, the short linear scan is cheaper than repeated shared-memory binary-search dependencies. Reject the prefix path and restore the linear mapper.

### Iteration 53 — Restore linear expert mapper

- Change: removed the shared tile-prefix state and restored the compact linear expert scan from Iter51.
- Focused result (`8192x4096x2048/E16`): **1.6136 ms, 85.17 TFLOP/s, 28.78% useful utilization, PASS**.
- This recovers the Iter51 baseline within 0.2% run variance and leaves the branch on the current best mapping policy.

### Iteration 54 — Shift dual-consumer registers to producer

- Change: changed the dual-N192 register split from consumer/producer 224/56 to 216/72, keeping the same 64.5K CTA total.
- Focused result (`8192x4096x2048/E16`): **5.6903 ms, 24.15 TFLOP/s, 8.16% useful utilization, PASS**.
- This is 3.53× slower than Iter53. N192's two 96-register accumulator fragments plus activation-prefetch/descriptors require the full 224-register consumer allocation; the 216-register cap causes severe hidden register pressure. Reject and restore 224/56.

### Iteration 55 — Restore dual-N192 register balance

- Change: restored 224 registers per N192 consumer and 56 for the producer.
- Focused result (`8192x4096x2048/E16`): **1.6125 ms, 85.23 TFLOP/s, 28.79% useful utilization, PASS**.
- This recovers the Iter51/53 best range and is the stable register policy retained in the branch.

### Iteration 56 — Dense single-primary FP8 staging

- Change: removed the dense primary/residual decomposition. The K=7168 fallback now expands each packed NVFP4 weight to one transient primary E4M3 value in shared memory and issues one WGMMA term. Dense shared memory drops from A + 2B plus two LUTs to A + B plus one LUT; persistent packed weights and scales are unchanged.
- Build: Python 3.12 extension rebuilt successfully in `lmsysorg/sglang:dev`; all kernels used a fresh `.dg_single_primary56` JIT cache.
- Full-suite result: **4/6 PASS**.

| Shape | Time | Useful TFLOP/s | Utilization | Status |
|---|---:|---:|---:|---|
| `512x2048x2048/E8` | 0.1477 ms | 29.08 | 9.82% | PASS |
| `4096x2048x2048/E8` | 0.5027 ms | 68.36 | 23.09% | PASS |
| `8192x4096x2048/E16` | 1.6170 ms | 84.99 | 28.71% | PASS |
| `4096x7168x2048/E32` | 2.4342 ms | 49.40 | 16.69% | PASS |
| `8192x4096x7168/E8` | — | — | — | PRECFAIL (cosine 0.999776, relative error 2.1054%) |
| `128x4096x7168/E8` | — | — | — | PRECFAIL (cosine 0.999777, relative error 2.1017%) |

- Analysis: the unchanged K=2048 sparse path stays in its previous performance range, proving the edit is isolated. A single primary E4M3 staging value is not accurate enough for either K=7168 workload under Cupra's unchanged elementwise contract. Cupra suppresses timing for precision failures, so a separate diagnostic run is required to report exact violation counts and raw single-primary latency.
- Exact seed-0 checker diagnostics:
  - `8192x4096x7168/E8`: **1,412,726 / 33,554,432 elements violate**, maximum absolute difference **10.5**.
  - `128x4096x7168/E8`: **22,098 / 524,288 elements violate**, maximum absolute difference **8.5**.
- Raw single-primary timing with Cupra's normal 8-slot, cold-L2, 30-sample profiler method:
  - `8192x4096x7168/E8`: **3.2842 ms, 146.47 useful TFLOP/s, 49.48% useful utilization**, 3.18% jitter. This is 18.2% lower latency and 22.3% higher useful throughput than Iter50's primary+residual 4.0156 ms.
  - `128x4096x7168/E8`: **0.3459 ms, 21.73 useful TFLOP/s, 7.34% useful utilization**, 2.23% jitter. This is 10.9% lower latency than Iter50's 0.3883 ms.
- Storage audit: original and prepared payloads are both exactly **132,120,608 bytes** for K=7168, with identical `uint8` packed-code, E4M3 block-scale, and FP32 global-scale dtype/numel tuples. No persistent expanded or residual weight data was added.
- Next: choose the smallest on-chip-only correction that restores 6/6 correctness; single-primary alone is a useful performance probe, not an admissible final solution.

### Iteration 57 — Single BF16 on-chip staging, first focused launch

- Change: replaced the dense single-E4M3 approximation with one exact BF16 on-chip staging tile and one BF16 WGMMA stream. Global storage and `prepare()` remain packed NVFP4 plus the original scales; no primary/residual split is present.
- Reference audit: `github/megamoe_nvfp4` checks its LUT against `exact_product.to(float8_e4m3fn)`, and its end-to-end gate uses cosine thresholds of 0.9 plus a norm-ratio range of 0.5–2.0. Iteration 56's cosine 0.999776 would pass that gate despite Cupra's elementwise failures. The reference also uses a one-level scale and an 80-byte fused row containing 8 bytes of persistent padding, so its accuracy claim and layout are not directly transferable to this task.
- Focused benchmark attempt: no kernel ran. Cupra rejected the human label `128x4096x7168/E8`; `--shape` requires four plain integers. This is a command-parse failure, so no compile, correctness, latency, or throughput result is reportable.
- Next: rerun as `--shape 128,4096,7168,8` with the same fresh JIT cache policy.

### Iteration 58 — Single BF16 focused runtime validation

- Change: no source change from Iteration 57; reran the intended `128,4096,7168,8` shape with a fresh `.dg_single_bf16_58` JIT cache.
- Result: JIT compilation succeeded, but the first correctness launch ended in `cudaErrorIllegalAddress`; **0/1 passed** and no performance number is reportable.
- Analysis: the 165,920-byte dynamic shared-memory request is within the H20 limit, so the leading suspect is the hand-written BF16 B128 swizzle/descriptor addressing rather than capacity. The FP8 layout advanced in byte-sized K atoms; BF16 needs the repository's typed descriptor helper to account for two-byte elements and 128-byte atom boundaries.
- Next: replace manual descriptor increments with `make_gmma_desc`/`advance_gmma_desc_lo` and validate the BF16 swizzle against the existing SM90 BF16 kernel layout before another benchmark.

### Iteration 59 — BF16 two-atom shared layout

- Change: reorganized each BF16 K=128 stage into two K=64 B128-swizzle atoms, `[K-atom][row][K64]`, and created a fresh BF16 descriptor for every K16 WGMMA instruction. This removes the invalid FP8-style row-contiguous second atom and manual descriptor increments from Iteration 58.
- Focused result (`128x4096x7168/E8`): JIT compilation succeeded, but the correctness launch still ended in `cudaErrorIllegalAddress`; **0/1 passed**, with no latency or throughput.
- Analysis: atom addressing alone was not the fault. The next isolation must separate producer expansion from WGMMA: first validate the generated BF16 A/B shared tiles in a standalone/guarded path, then validate one WGMMA K atom. Repeated full launches do not identify whether the fault is conversion, shared layout, or descriptor legality.
- Next: stop blind benchmark retries and use a minimal CUDA probe or compute-sanitizer-style isolation before changing the production kernel again.

### Iteration 60 — Bound BF16 LUT initialization to 128 rows

- Diagnostic: `compute-sanitizer` isolated the warp-specialized crash to 16-byte shared writes at kernel PC `+0x760`. A 256-thread CTA initialized a 128-row BF16 LUT with every thread; threads 130+ wrote past the LUT and the 32-byte barrier tail. The non-warp-specialized 128-thread specialization had already passed because it could not overflow the table.
- Change: guard LUT construction with `tid < 128`. Retained the corrected two-K64-atom BF16 layout from Iteration 59.
- Sanitizer validation (`1024x2048x7168/E1`, forced warp-specialized N256): **PASS, 0 errors**.
- Focused Cupra result (`128x4096x7168/E8`): **0.4732 ms, 15.88 useful TFLOP/s, 5.37% useful utilization, PASS**, 1.74% jitter.
- Comparison: this exact single-BF16 path is 21.9% slower than Iteration 50's primary+residual FP8 result (0.3883 ms) and 36.8% slower than Iteration 56's invalid single-E4M3 result (0.3459 ms). It satisfies the strict checker without splitting weights, but BF16 conversion/staging is costly for small M.
- Next: validate the large K=7168 shape and then run the full six-shape suite; the K=2048 sparse path should remain unchanged.

### Iteration 61 — Full single-BF16 staging validation

- Change: no further kernel change; rebuilt/reran all six default shapes with a fresh `.dg_single_bf16_61` JIT cache after the sanitizer-clean Iteration 60 fix.
- Full suite: **6/6 PASS** under Cupra's unchanged elementwise checker.

| Shape | Time | Useful TFLOP/s | Utilization | Status |
|---|---:|---:|---:|---|
| `512x2048x2048/E8` | 0.1479 ms | 29.05 | 9.81% | PASS |
| `4096x2048x2048/E8` | 0.5037 ms | 68.22 | 23.05% | PASS |
| `8192x4096x2048/E16` | 1.6146 ms | 85.12 | 28.76% | PASS |
| `4096x7168x2048/E32` | 2.4366 ms | 49.35 | 16.67% | PASS |
| `8192x4096x7168/E8` | 5.3619 ms | 89.71 | 30.31% | PASS |
| `128x4096x7168/E8` | 0.4732 ms | 15.88 | 5.37% | PASS |

- Accuracy: PASS means zero elements violate `abs(out-ref) <= 2 + 0.05*abs(ref)` for both K=7168 shapes. This removes Iteration 56's 1,412,726 and 22,098 violations without a primary/residual split.
- Performance tradeoff: versus Iteration 50's exact primary+residual FP8 path, single BF16 is 33.5% slower on `8192x4096x7168/E8` (5.3619 vs 4.0156 ms) and 21.9% slower on `128x4096x7168/E8` (0.4732 vs 0.3883 ms). Versus invalid single E4M3, it is 63.3% and 36.8% slower, respectively.
- Storage: the public adapter is unchanged from the 132,120,608-byte original/prepared equality audit. Only transient shared-memory staging changed; global weights remain packed FP4 plus the original scales.
- Decision: this is the valid no-split solution. The reference branch's single-E4M3 path remains a faster option only under its much looser cosine/norm gate, not under Cupra's elementwise contract.

### Iteration 62 — Restore the single-E4M3 performance baseline

- User directive: set the known representation error aside and optimize raw useful throughput toward 80% of Cupra's 296 TFLOP/s H20 FP8 peak. The checker is not modified; PRECFAIL status remains documented.
- Change: restored the Iteration 56 dense path with one transient E4M3 B stage and one FP8 WGMMA stream. K<=2048 sparse dispatch and the byte-preserving packed-NVFP4 adapter are unchanged. Unused BF16 helper code remains available but is not dispatched.
- Primary-shape raw timing (`8192x4096x7168/E8`): **3.2850 ms, 146.44 useful TFLOP/s, 49.47% utilization**, 3.48% jitter.
- This reproduces Iteration 56's 3.2842 ms / 146.47 TFLOP/s within 0.03%, establishing a clean baseline for the dual-M-consumer pipeline.
- Target gap: 80% requires at most 2.032 ms / at least 236.8 TFLOP/s, so the remaining required speedup is **1.618x**.
- Next: share each decoded N256 B tile across two M64 consumer warpgroups, move activation staging to the consumers, and retain the one-consumer path for small M.

### Iteration 63 — Dual-M consumers share one dense B decode

- Change: added a 384-thread large-M dense specialization with one 56-register producer and two 224-register N256 consumer warpgroups. The producer decodes one transient E4M3 B tile for an M128 CTA; each consumer stages its own M64 activation tile and reuses that B stage. Full/empty barriers combine three producers-of-stage data and eight consumer-warp releases. K=7168 small-M retains the original one-consumer dispatch.
- Primary-shape raw timing (`8192x4096x7168/E8`): **2.0588 ms, 233.64 useful TFLOP/s, 78.93% utilization**, 6.53% jitter.
- Versus Iteration 62 (3.2850 ms / 146.44 TFLOP/s), latency improves **37.3%** and throughput improves **59.6%**. This confirms packed-weight decode/reload was the dominant non-WGMMA cost.
- Target gap: 80% requires 236.8 TFLOP/s / 2.032 ms. The candidate is only **1.33% short**, about 26.8 us, so a small scheduling/decode reduction is sufficient; a more invasive cluster design is not justified.
- Precision remains the known single-E4M3 approximation and was intentionally not used as this iteration's timing gate. Persistent storage remains byte-identical packed NVFP4.
- Next: reduce dual-consumer fixed overhead or producer decode issue count by at least 1.5%, then repeat the exact same cold-L2 timing.

### Iteration 64 — Use the uniform dense register budget

- SASS evidence: Iteration 63's cubin reports `REG:168`, `LOCAL:0`, `STACK:0`, and the highest referenced general register is R165. Unlike the sparse dual-consumer kernel, each dense N256 consumer owns only one 128-register accumulator array, so its inherited 224-register allocation was unnecessary while the producer was artificially capped at 56.
- Change: removed `setmaxnreg` reconfiguration from the dense dual-M specialization. All three warpgroups now use the uniform ~168-register budget implied by `__launch_bounds__(384, 1)`, giving the decode producer enough registers without constraining either consumer.
- Primary-shape raw timing (`8192x4096x7168/E8`): **2.0281 ms, 237.19 useful TFLOP/s, 80.13% utilization**, 6.35% jitter.
- Versus Iteration 63, latency improves **1.50%** and throughput improves **1.52%**. Versus the Iteration 62 single-consumer baseline, latency improves **38.3%** and throughput improves **62.0%**.
- Result: the primary 80% target is reached. The measured latency is 3.9 us below the 2.032 ms threshold and throughput is 0.39 TFLOP/s above 236.8 TFLOP/s, so final verification needs a repeated focused run plus the complete six-shape suite.
- Next: repeat the primary shape with a fresh cache/run, then measure all six shapes and audit dispatch/storage before declaring the goal complete.
## Iteration 65 — repeat verification of uniform-register dual-M kernel

- Change: no source change; repeated the focused H20 raw benchmark for the primary `8192x4096x7168/E8` shape from commit `46967b4`.
- Protocol: standard Cupra timing (`8` input slots, `10` warmups, `30` timed iterations, cold-L2 rotation).
- Result: `2.028123 ms`, `237.1830` useful TFLOP/s, `80.1294%` of the `296 TFLOP/s` H20 FP8 peak, `6.1127%` jitter.
- Assessment: independently reproduces the 80% target, but with only about `3.9 us` latency margin; continue optimizing for a robust margin before final full-suite validation.
## Iteration 66 — three-stage dual-M build, benchmark blocked by adapter rank

- Change: increased the dense dual-M ring from two to three on-chip stages (`~148 KiB` tile staging) without changing global packed-FP4 storage or arithmetic.
- Result: benchmark did not reach timing; the freshly rebuilt extension rejected the Cupra adapter's four-dimensional prepacked weight view at `layout.hpp:39` (`t.dim() == N`).
- Assessment: infrastructure/interface failure, not a kernel performance result. Fix the adapter to expose the same pretransformed bytes as the API's required rank, then rerun this exact kernel revision.
## Iteration 67 — three-stage dual-M timing

- Change: reran Iteration 66 with the freshly rebuilt extension selected before the source-tree package; no adapter or data-layout change.
- Protocol: standard Cupra timing (`8` input slots, `10` warmups, `30` timed iterations, cold-L2 rotation).
- Result: `2.0297075 ms`, `236.9979` useful TFLOP/s, `80.0668%` of peak, `6.0829%` jitter.
- Assessment: three stages still reaches 80%, but is `1.58 us` slower than the two-stage repeat and reduces the target margin. Revert to two stages; extra buffering does not relieve the steady-state bottleneck.
## Iteration 68 — final two-stage full-suite raw timing

- Change: reverted the no-benefit three-stage experiment to the faster two-stage dual-M pipeline.
- Protocol: all six official Cupra shapes, standard timing (`8` input slots, `10` warmups, `30` timed iterations, cold-L2 rotation), freshly rebuilt extension and fresh JIT cache.

| Shape | Latency (ms) | Useful TFLOP/s | H20 peak utilization | Jitter |
| --- | ---: | ---: | ---: | ---: |
| `512x2048x2048/E8` | 0.147343 | 29.1494 | 9.8478% | 4.8723% |
| `4096x2048x2048/E8` | 0.4967435 | 69.1700 | 23.3682% | 14.6856% |
| `8192x4096x2048/E16` | 1.5827995 | 86.8328 | 29.3354% | 12.3318% |
| `4096x7168x2048/E32` | 2.3789805 | 50.5507 | 17.0779% | 10.2906% |
| `8192x4096x7168/E8` | **2.0207070** | **238.0535** | **80.4235%** | **1.4498%** |
| `128x4096x7168/E8` | 0.3448610 | 21.7948 | 7.3631% | 3.1333% |

- Assessment: the primary large-M shape reaches the 80% target for a third independent run, now with a `~11.3 us` latency margin and low `1.45%` jitter. Keep the two-stage implementation for final validation.
## Final validation — storage and functional audit

- Packed-storage audit (`E8, N4096, K7168`): original and prepared weights/scales are both exactly `132,120,608` bytes. The packed weight payload remains `117,440,512` bytes of `uint8` (`2` FP4 values per byte); `prepare()` only permutes it. No persistent FP8 weight tensor is created.
- Source identity: SHA-256 hashes for the local kernel, host launcher, and Cupra adapter match the files used on the H20 host.
- Functional check: all four `K=2048` shapes pass strict elementwise Cupra checking with zero violations. Both `K=7168` shapes reproduce the already-known single-E4M3 staging error (`cosine ~= 0.999776`, `rel_err ~= 2.105%`) with no NaN/Inf; per user direction this known quantization-reference mismatch is not a performance gate for the current campaign.
- Repository audit: branch `w4a8` descends directly from `origin/main`; `github/megamoe_nvfp4` was used only as a reference.
## Iteration 69 — remove active sparse path, dense-only baseline

- Change: deleted the complementary-2:4 NVFP4 kernel and dequant tables, removed the sparse WGMMA helper/selector and all host dispatch/smem branches, and routed every shape through the existing dense single-E4M3 staging kernels. Persistent packed-FP4 layout is unchanged.
- Protocol: all six official Cupra shapes, standard timing (`8` input slots, `10` warmups, `30` timed iterations, cold-L2 rotation), freshly rebuilt extension and fresh JIT cache.

| Shape | Latency (ms) | Compute SOL | Roofline SOL | Bound |
| --- | ---: | ---: | ---: | --- |
| `512x2048x2048/E8` | 0.131215 | 11.0582% | 11.0582% | compute |
| `4096x2048x2048/E8` | 0.199568 | 58.1657% | 58.1657% | compute |
| `8192x4096x2048/E16` | 0.734481 | 63.2175% | 63.2175% | compute |
| `4096x7168x2048/E32` | 0.9178875 | 44.2626% | 44.2626% | compute |
| `8192x4096x7168/E8` | **2.019571** | **80.4687%** | **80.4687%** | compute |
| `128x4096x7168/E8` | 0.345104 | 7.3579% | **8.0706%** | memory |

- Assessment: sparse removal improves three large K2048 shapes substantially and preserves the large K7168 target. The memory-bound small shape is unchanged and remains `8.68x` slower than the `39.79 us` / 70% gate, confirming that a new M8 warp-MMA path is required.
## Iteration 70 — direct-register M8 warp-MMA path

- Change: added a dense small-M specialization using transposed `mma.sync.aligned.m16n8k32` (`W[N16,K32] @ A^T[K32,M8]`). Four warps cover N64 per CTA; each warp decodes packed NVFP4 directly into E4M3 A-fragment registers and reuses each weight fragment across all token groups for its expert. The `16x64x128/E1` functional probe passed with zero violations.
- Protocol: focused official `128x4096x7168/E8`, standard Cupra timing (`8` input slots, `10` warmups, `30` timed iterations, cold-L2 rotation).
- Result: `0.7669735 ms`, `9.7998` useful TFLOP/s, `3.3107%` compute SOL, `3.6314%` roofline SOL, `20.93%` jitter.
- Assessment: correct but `2.22x` slower than the M64 WGMMA baseline (`0.345104 ms`). The smaller token tile removes padding but legacy warp-MMA plus per-warp decode/activation issue does not approach Hopper WGMMA throughput; inspect SASS/register spills and move to a WGMMA-compatible cross-expert packing strategy or another hardware-supported path.
## Iteration 71 — transposed N8 WGMMA small-M path

- Change: added a Hopper-native low-token kernel using `wgmma.mma_async.m64n8k32`, with WGMMA M64 mapped to output columns and N8 mapped to tokens. A producer warpgroup decodes/stages one packed-FP4 N64 tile and all activation rows; one consumer warpgroup reuses that weight tile across every token group through a two-stage barrier pipeline. The small functional probe again passed with zero violations.
- Protocol: focused official `128x4096x7168/E8`, standard Cupra timing (`8` input slots, `10` warmups, `30` timed iterations, cold-L2 rotation).
- Result: `0.269569 ms`, `27.8823` useful TFLOP/s, `9.4197%` compute SOL, `10.3320%` roofline SOL, `30.27%` jitter.
- Assessment: restores native WGMMA throughput and improves the dense M64 baseline by `1.28x` (`345.104 -> 269.569 us`) and the warp-MMA prototype by `2.85x`, but remains `6.78x` above the `39.79 us` target. The per-token-group commit/wait sequence, oversized M128 activation staging, and N64 CTA granularity are the next concrete overheads to remove.
## Iteration 72 — four-PRMT packed-NVFP4 decode

- Change: replaced the small-M producer's 12-PRMT decode of each 16-weight group with four direct LUT PRMTs. Each pair of packed bytes supplies the four magnitude selector nibbles directly; integer multiply/mask operations expand the four FP4 sign bits into the E4M3 byte sign mask. The functional probe remained bit-equivalent at the checked output tolerance.
- Protocol: focused official `128x4096x7168/E8`, standard Cupra timing (`8` input slots, `10` warmups, `30` timed iterations, cold-L2 rotation).
- Result: `0.265535 ms`, `28.3058` useful TFLOP/s, `9.5628%` compute SOL, `10.4890%` roofline SOL, `32.80%` jitter.
- Assessment: only a `1.5%` latency win (`269.569 -> 265.535 us`). PRMT issue is not the sole limiter; increasing resident producer concurrency and removing per-token-group WGMMA commit/wait are required.
## Iteration 73 — 32-token accumulator window and batched WGMMA

- Change: reduced the transposed small-M kernel's live accumulator/activation window from `16` M8 groups (`128` tokens) to `4` groups (`32` tokens), processing larger experts in consecutive passes. WGMMA operations for two token groups are now issued in one commit group before `wait<0>`, replacing the previous commit/wait after every group. Pipeline stage and phase state remains continuous across passes; resetting it caused a functional-probe barrier deadlock and was corrected before timing.
- Correctness: a deliberately cross-pass `64x64x128/E1` probe completed with zero elementwise violations, no NaN/Inf, cosine `0.9997383`, and maximum absolute difference `1.1796875`.
- Protocol: focused official `128x4096x7168/E8`, standard Cupra timing (`8` input slots, `10` warmups, `30` timed iterations, cold-L2 rotation).
- Result: `0.252830 ms`, `29.7282` useful TFLOP/s, `10.0433%` compute SOL, `11.0160%` roofline SOL, `29.53%` jitter.
- Assessment: latency improves `4.79%` versus Iteration 72 (`265.535 -> 252.830 us`), validating the smaller live window and WGMMA batching, but remains `6.35x` above the `39.788 us` / `70%` roofline target. The next step is to inspect the generated resource footprint and test a still smaller live window or a wider N tile that raises decode-producer concurrency without multiplying persistent weight storage.
## Iteration 74 — exact-byte fused packed/scale tile layout

- Change: for `M <= 128`, replaced the disjoint pretransformed packed-FP4 and scale tensors with one exact-byte, vector-major uint8 payload. Each K128 tile stores four coalesced `[N,16-byte]` packed vectors followed by two `[N,4-byte]` scale-word vectors. The producer now obtains each K64 half with two aligned 128-bit loads plus one 32-bit scale load. The public call signature is unchanged and `M > 128` retains the previous prepared layout and kernels.
- Storage audit: on the official shape, canonical and prepared totals are both exactly `132,120,608` bytes. The fused uint8 payload is `132,120,576` bytes (`117,440,512` packed FP4 bytes plus `14,680,064` original scale bytes), the scale placeholder has zero elements, and the existing global scales remain 32 bytes. A round-trip audit reconstructed both canonical tensors byte-for-byte.
- Correctness: the `64x64x128/E1` cross-pass probe and a `[0,17,17,69,70]` routing probe covering an empty expert, a 52-token multi-pass expert, and partial M8 tails both passed with zero elementwise violations and no NaN/Inf.
- Resources: launch bounds retain three CTAs per SM; the generated cubin reports `REG:80`, `LOCAL:0`, and `STACK:0`.
- Protocol: focused official `128x4096x7168/E8`, standard Cupra timing (`8` input slots, `10` warmups, `30` timed iterations, cold-L2 rotation).
- Result: `0.238432 ms`, `31.5234` useful TFLOP/s, `10.6498%` compute SOL, `11.6813%` roofline SOL, `28.81%` jitter.
- Assessment: latency improves `5.69%` versus Iteration 73 (`252.830 -> 238.432 us`), confirming that packed/scale loader transactions and address generation were material. The candidate remains `5.99x` above the `39.788 us` target, so the next iteration must increase parallel decode capacity or eliminate more per-group integer work rather than only retune the load layout.
## Iteration 75 — two producer warpgroups per CTA

- Change: expanded the small-M CTA from 256 to 384 threads: one consumer warpgroup plus two producer warpgroups. The two producers split the 64 output rows and four packed K vectors, each warp decoding only two scale groups; 256 producer threads also partition activation staging without duplication. The full barrier waits for both producers. At `REG:80`, two CTAs reside per SM, providing four producer warpgroups versus Iteration 74's three.
- Correctness: the empty-expert, multi-pass, and partial-tail routing probe passed with zero elementwise violations and no NaN/Inf. The cubin reports `LOCAL:0` and `STACK:0`.
- Protocol: focused official `128x4096x7168/E8`, standard Cupra timing (`8` input slots, `10` warmups, `30` timed iterations, cold-L2 rotation).
- Result: `0.258206 ms`, `29.1093` useful TFLOP/s, `9.8342%` compute SOL, `10.7867%` roofline SOL, `29.95%` jitter.
- Assessment: despite 33% more resident producer warpgroups, latency regresses `8.29%` versus Iteration 74 (`238.432 -> 258.206 us`). Reducing resident consumer warpgroups from three to two and adding a second producer arrival/scheduling domain costs more than the extra decode parallelism saves. Do not retain this topology; the next direction is a 128-thread cooperative decode/compute CTA that increases the number of independent CTAs without adding producer barriers.
## Iteration 76 — single-warpgroup cooperative decode and WGMMA

- Change: added a 128-thread small-M kernel in which one warpgroup cooperatively decodes/stages each K128 tile and immediately consumes it with N8 WGMMA. A CTA-wide synchronization replaces the two-stage producer/consumer full/empty pipeline. Shared memory drops from about 25 KiB to about 13 KiB and the dedicated producer warpgroup is removed; independent CTAs provide latency hiding. The Iteration 74 exact-byte fused FP4/scale payload is unchanged.
- Correctness: the empty-expert, multi-pass, and partial-tail routing probe passed with zero elementwise violations and no NaN/Inf.
- Resources: the cubin reports `REG:72`, `LOCAL:0`, and `STACK:0`. With 128 threads and 13,312 dynamic shared-memory bytes, registers permit seven resident CTAs per SM (28 warps), versus Iteration 74's three 256-thread CTAs (24 warps).
- Protocol: focused official `128x4096x7168/E8`, standard Cupra timing (`8` input slots, `10` warmups, `30` timed iterations, cold-L2 rotation).
- Result: `0.150479 ms`, `49.9484` useful TFLOP/s, `16.8745%` compute SOL, `18.5088%` roofline SOL, `49.90%` jitter.
- Assessment: latency improves `36.89%` versus the valid Iteration 74 topology (`238.432 -> 150.479 us`) and `41.72%` versus Iteration 75. Removing cross-warpgroup barriers is the first structural gain large enough to materially change the gap, but the kernel remains `3.78x` above the `39.788 us` target. High jitter and repeated CTA-wide synchronization now point to K-stage granularity and load/decode scheduling as the next targets.
## Iteration 77 — 64-token cooperative accumulation window

- Change: expanded the cooperative kernel's live window from four to eight M8 groups (`32 -> 64` tokens). Every official benchmark expert (maximum 42 tokens across the standard slots) therefore scans and decodes its packed weights only once instead of the largest experts requiring two passes.
- Correctness: a `128x64x128/E1` probe forced two 64-token passes and passed with zero elementwise violations and no NaN/Inf.
- Resources: the larger accumulator set raises the cubin from 72 to 96 registers per thread, with `LOCAL:0` and `STACK:0`. Residency falls from seven to five CTAs per SM, so the 512-CTA grid no longer fits in one resident wave.
- Protocol: focused official `128x4096x7168/E8`, standard Cupra timing (`8` input slots, `10` warmups, `30` timed iterations, cold-L2 rotation).
- Result: `0.194720 ms`, `38.6000` useful TFLOP/s, `13.0405%` compute SOL, `14.3035%` roofline SOL, `22.07%` jitter.
- Assessment: eliminating second weight scans does not compensate for the occupancy loss; latency regresses `29.40%` versus Iteration 76 (`150.479 -> 194.720 us`). Eight groups are unnecessarily large because six M8 groups already cover the observed 42-token maximum. Test a six-group window under a six-CTA launch bound before rejecting single-pass accumulation entirely.
## Iteration 78 — six-group single-scan window

- Change: reduced the live window from eight to six M8 groups (`48` tokens), which still covers the standard slots' maximum 42-token expert in one packed-weight scan. A six-CTA launch bound constrains register allocation while retaining the cooperative topology.
- Correctness: a `128x64x128/E1` three-pass probe passed with zero elementwise violations and no NaN/Inf.
- Resources: the cubin reports `REG:80`, `LOCAL:0`, and `STACK:0`, permitting six CTAs per SM. The first resident wave holds 468 of 512 CTAs, leaving only a 44-CTA scheduling tail.
- Protocol: focused official `128x4096x7168/E8`, standard Cupra timing (`8` input slots, `10` warmups, `30` timed iterations, cold-L2 rotation).
- Result: `0.152961 ms`, `49.1380` useful TFLOP/s, `16.6007%` compute SOL, `18.2085%` roofline SOL, `52.36%` jitter.
- Assessment: this recovers almost all of Iteration 77's occupancy regression but remains `1.65%` slower than the four-group Iteration 76 (`150.479 us`). The residual scheduling tail and larger WGMMA accumulator loop outweigh avoiding the second scan for a small subset of experts. Restore four groups as the best window and optimize its per-K128 decode/synchronization path.
## Iteration 79 — K256 cooperative stage with two K128 swizzle slabs

- Change: restored the best four-group window and made the cooperative K tile a JIT parameter. Shapes divisible by 256 now stage two independently swizzled K128 weight/activation slabs before one CTA synchronization, then accumulate both slabs. K128 remains the fallback for API-compatible shapes. This halves CTA-wide synchronization frequency on every official shape without changing the exact-byte global payload.
- Bring-up: the first K256 layout incorrectly modeled row-stride-256 as one WGMMA swizzle surface and failed the functional probe. It was corrected before timing by storing two independent K128 slabs and constructing one descriptor per slab.
- Correctness: both K128 and K256 probes covering an empty expert, multi-pass expert, and partial M8 tails pass with zero elementwise violations and no NaN/Inf.
- Resources: both cubins report `REG:72`, `LOCAL:0`, and `STACK:0`. The K256 path uses 25,600 dynamic shared-memory bytes; registers still permit seven CTAs per SM, enough for the 512-CTA grid to reside in one wave.
- Protocol: focused official `128x4096x7168/E8`, standard Cupra timing (`8` input slots, `10` warmups, `30` timed iterations, cold-L2 rotation).
- Result: `0.1358725 ms`, `55.3180` useful TFLOP/s, `18.6885%` compute SOL, `20.4985%` roofline SOL, `30.53%` jitter.
- Assessment: latency improves `9.71%` versus Iteration 76 (`150.479 -> 135.8725 us`), proving CTA synchronization and stage setup were material. The remaining gap is `3.41x` to the `39.788 us` gate; continue increasing K-stage granularity while preserving seven-CTA residency, then attack the packed decode instruction stream.
## Iteration 80 — same-warpgroup double-buffer overlap

- Change: switched back to K128 and added two shared-memory stages. After committing the last WGMMA batch for the current stage, the same warpgroup decodes and stages the next K128 slice into the alternate buffer before waiting; one CTA barrier then establishes both WGMMA completion and next-stage visibility. The intent was to overlap CUDA-core decode with asynchronous tensor-core work without a dedicated producer warpgroup.
- Correctness: the prior K256 zero-violation seed ran 20 times with stable output and reproduced Iteration 79's precision metrics exactly. A K2048 sample's 35 violations were also deterministic and reproduced the known single-E4M3 reference tail rather than a pipeline race.
- Resources: the cubin reports `REG:80`, `LOCAL:0`, and `STACK:0`; double buffering uses about 25 KiB shared memory and permits six CTAs per SM, leaving a 44-CTA scheduling tail.
- Protocol: focused official `128x4096x7168/E8`, standard Cupra timing (`8` input slots, `10` warmups, `30` timed iterations, cold-L2 rotation).
- Result: `0.1855195 ms`, `40.5143` useful TFLOP/s, `13.6873%` compute SOL, `15.0129%` roofline SOL, `52.59%` jitter.
- Assessment: latency regresses `36.54%` versus Iteration 79 (`135.8725 -> 185.5195 us`). Heavy decode instructions issued by the same warpgroup do not provide useful overlap with its WGMMA stream, while the extra stage reduces residency. Restore the K256 single-stage kernel; future overlap must use a genuinely independent execution context or remove decode instructions outright.
## Iteration 81 — nibble-plane DP4A selector decode

- Change: restored the Iteration 79 K256 single-stage topology and changed only the small-M exact-byte prepared representation and decoder. Every eight FP4 codes still occupy four bytes, but K0:K3 are stored in the high nibbles and K4:K7 in the low nibbles. The reference branch's full-DP4A selector path can therefore emit two contiguous FP8 words directly, replacing general integer sign/selector expansion without an extra interleave PRMT. Persistent payload size and 4-bit E2M1 information are unchanged.
- Correctness/storage: the prior zero-violation K256 seed reproduces Iteration 79's precision metrics exactly; prepared packed-plus-scale bytes still equal the canonical payload byte-for-byte in size. No global FP8 tensor is created.
- SASS/resources: `LOP3` drops `172 -> 108`, `SHF` drops `67 -> 35`, and `IMAD` drops `192 -> 176`; `64 IDP` instructions are added and `PRMT` remains 32. The cubin remains `REG:72`, `LOCAL:0`, and `STACK:0`, preserving seven-CTA residency.
- Protocol: focused official `128x4096x7168/E8`, standard Cupra timing (`8` input slots, `10` warmups, `30` timed iterations, cold-L2 rotation).
- Result: `0.121439 ms`, `61.8927` useful TFLOP/s, `20.9097%` compute SOL, `22.9348%` roofline SOL, `36.21%` jitter.
- Assessment: latency improves `10.62%` versus Iteration 79 (`135.8725 -> 121.439 us`), validating the reference branch's DP4A policy for this low-token, large-K path. The remaining gap is `3.05x` to the `39.788 us` target. Continue reducing selector/sign work and combine it with stage-level overhead reductions that keep seven CTAs resident.
## Iteration 82 — split token passes into independent CTAs

- Change: expanded the small-M grid with four pass slots per `(expert,N64)` tile. Each active CTA handles exactly one four-group/32-token pass; inactive slots return before LUT or weight access. This allows a 33–42-token expert's two passes to execute independently instead of serially in one CTA. Persistent weights and per-active-pass weight traffic are unchanged.
- Correctness: an E1 128-token probe exercised all four concurrent passes, and the empty-expert/partial-tail probe also passed with zero elementwise violations and no NaN/Inf.
- Protocol: focused official `128x4096x7168/E8`, standard Cupra timing (`8` input slots, `10` warmups, `30` timed iterations, cold-L2 rotation).
- Result: `0.126191 ms`, `59.5620` useful TFLOP/s, `20.1223%` compute SOL, `22.0712%` roofline SOL, `19.96%` jitter.
- Assessment: jitter falls substantially, but median latency regresses `3.91%` versus Iteration 81 (`121.439 -> 126.191 us`). Scheduling 2,048 total blocks, most of which immediately return, plus the active-CTA tail costs more than parallelizing the few second passes saves. Restore the compact 512-CTA grid and continue optimizing the per-CTA decode body.
## Iteration 83 — two-way alternating pass split

- Change: reduced the pass grid from four to two slots per `(expert,N64)` tile (`2,048 -> 1,024` total blocks). The standard workload's first and second passes remain independent; for general experts requiring more passes, the two CTAs process alternating pass indices serially.
- Correctness: the 128-token E1 probe exercised all four passes across the two alternating CTAs and passed with zero elementwise violations and no NaN/Inf.
- Protocol: focused official `128x4096x7168/E8`, standard Cupra timing (`8` input slots, `10` warmups, `30` timed iterations, cold-L2 rotation).
- Result: `0.121408 ms`, `61.9085` useful TFLOP/s, `20.9150%` compute SOL, `22.9407%` roofline SOL, `11.23%` jitter.
- Assessment: median latency is effectively tied with Iteration 81 but improves by `0.026%` (`121.439 -> 121.408 us`), while jitter drops from `36.21%` to `11.23%`. Keep the two-way split as the more stable best revision, but it does not materially close the `3.05x` target gap; further work must reduce decode/WGMMA work per active CTA.
## Iteration 84 — widen the cooperative token tile to N16

- Change: widened the cooperative transposed WGMMA token tile from N8 to N16 while keeping the live window at 32 tokens (`4 x M8 -> 2 x M16`). This halves the static WGMMA instruction and descriptor count per K tile without changing useful Tensor Core work, packed-FP4 global storage, or the two-way pass split.
- Correctness: the `[0,17,17,69,70]` empty-expert/partial-tail probe and the `128`-token E1 cross-pass probe both pass with zero elementwise violations and no NaN/Inf. The checked precision metrics match the retained N8 path.
- Resources/SASS: the cubin remains `REG:72`, `LOCAL:0`, and `STACK:0`, preserving seven-CTA register residency. Static `QGMMA` count falls `32 -> 16`; decode remains `64 IDP`, while extra N16 epilogue/addressing raises `IMAD/PRMT/SHF` modestly.
- Protocol: focused official `128x4096x7168/E8`, standard Cupra timing (`8` input slots, `10` warmups, `30` timed iterations, cold-L2 rotation).
- Result: `0.1088325 ms`, `69.0620` useful TFLOP/s, `23.3318%` compute SOL, `25.5915%` roofline SOL, `29.31%` jitter.
- Assessment: latency improves `10.36%` versus Iteration 83 (`121.408 -> 108.8325 us`), so WGMMA issue/descriptor overhead is material even though the physical Tensor Core work is unchanged. Keep N16 as the new best; it remains `2.74x` above the `39.788 us` / `70%` gate. The next width experiment should test N32, where static QGMMA issue halves again but token padding and accumulator mapping costs increase.
## Iteration 85 — widen the cooperative token tile to N32

- Change: widened the same 32-token accumulation window from `2 x N16` to one N32 WGMMA group. The pass payload, activation staging bytes, packed-FP4 traffic, accumulator-register count, and two-way pass scheduler are unchanged; only the WGMMA shape and epilogue fragment traversal change.
- Correctness: both the empty-expert/partial-tail routing probe and the 128-token E1 cross-pass probe pass with zero elementwise violations and reproduce the retained precision metrics.
- Resources/SASS: the cubin remains `REG:72`, `LOCAL:0`, and `STACK:0`. Static `QGMMA` count falls `16 -> 8`; `IDP/LDG/STS` remain `64/24/16`, and `IMAD` falls `217 -> 174` relative to N16 after the compiler simplifies the single-group path.
- Protocol: focused official `128x4096x7168/E8`, standard Cupra timing (`8` input slots, `10` warmups, `30` timed iterations, cold-L2 rotation).
- Result: `0.0967675 ms`, `77.6727` useful TFLOP/s, `26.2408%` compute SOL, `28.7822%` roofline SOL, `44.02%` jitter.
- Assessment: latency improves `11.09%` versus Iteration 84 (`108.8325 -> 96.7675 us`) and `20.30%` versus Iteration 83. Keep N32 as the new best despite the noisy timing; it remains `2.43x` above the `39.788 us` / `70%` gate. The monotonic N8 -> N16 -> N32 trend warrants testing N64, but that doubles the staged/physical token rows per active pass and must retain sufficient CTA residency.
## Iteration 86 — test the N64 WGMMA width boundary

- Change: widened the one-group token tile from N32 to N64 and increased the host dynamic-shared-memory allocation accordingly. This lets every standard expert fit in one weight scan, but doubles activation staging and physical Tensor Core token rows while leaving the static WGMMA instruction count unchanged.
- Correctness: the routing/tail probe and 128-token E1 probe both pass with zero elementwise violations and unchanged precision metrics.
- Resources/SASS: registers rise `72 -> 80`; the K256 stage grows `25,600 -> 33,792` bytes. Static `QGMMA` remains `8`, while `LDG` rises `24 -> 32`; six CTAs can reside per SM instead of seven, leaving a scheduling tail for the 512 active standard tiles.
- Protocol: focused official `128x4096x7168/E8`, standard Cupra timing (`8` input slots, `10` warmups, `30` timed iterations, cold-L2 rotation).
- Result: `0.1292805 ms`, `58.1386` useful TFLOP/s, `19.6414%` compute SOL, `21.5437%` roofline SOL, `70.53%` jitter.
- Assessment: latency regresses `33.60%` versus Iteration 85 (`96.7675 -> 129.2805 us`). N64 crosses the useful width boundary: padding/physical WGMMA work, larger activation staging, and lower residency dominate. Restore the exact Iteration 85 N32 code before pursuing a different axis.
## Iteration 87 — replace DP4A selector packing with PRMT

- Change: restored the exact Iteration 85 N32 kernel and switched both nibble-plane magnitude selectors from the two-DP4A packer to the shift/add/PRMT packer. The prepared bytes, FP8 values, WGMMA work, and launch topology are unchanged.
- Correctness: both focused probes pass with zero elementwise violations and bit-equivalent reported precision metrics.
- Resources/SASS: the cubin remains `REG:72`, `LOCAL:0`, and `STACK:0`. Selector packing changes from `64 IDP + 41 PRMT + 174 IMAD` to `0 IDP + 70 PRMT + 142 IMAD`; WGMMA and memory instructions are unchanged.
- Protocol: focused official `128x4096x7168/E8`, standard Cupra timing (`8` input slots, `10` warmups, `30` timed iterations, cold-L2 rotation).
- Result: `0.1044000 ms`, `71.9942` useful TFLOP/s, `24.3224%` compute SOL, `26.6780%` roofline SOL, `36.36%` jitter.
- Assessment: latency regresses `7.89%` versus Iteration 85 (`96.7675 -> 104.4000 us`). Hopper executes the full-DP4A selector path more efficiently than replacing those instructions with extra PRMT traffic. Restore Iteration 85's `<true,true>` decoder; further decode work must reduce both instruction classes or change how decoded weights are shared.
## Iteration 88 — balance DP4A and PRMT selector packing

- Change: restored the Iteration 85 N32 kernel, then used DP4A for the high-nibble selector and shift/add/PRMT for the low-nibble selector. This is the exact midpoint between Iteration 85's all-DP4A and Iteration 87's all-PRMT instruction mixes.
- Correctness: both focused probes pass with zero elementwise violations and unchanged precision metrics.
- Resources/SASS: the cubin remains `REG:72`, `LOCAL:0`, and `STACK:0`; the hot body contains `32 IDP`, `54 PRMT`, and `158 IMAD`, between the two endpoint variants. WGMMA and memory instruction counts are unchanged.
- Protocol: focused official `128x4096x7168/E8`, standard Cupra timing (`8` input slots, `10` warmups, `30` timed iterations, cold-L2 rotation).
- Result: `0.0986240 ms`, `76.2106` useful TFLOP/s, `25.7468%` compute SOL, `28.2404%` roofline SOL, `44.99%` jitter.
- Assessment: the mixed path is `5.53%` faster than all-PRMT but still `1.92%` slower than Iteration 85's all-DP4A result (`96.7675 us`). Iterations 86–88 provide three consecutive candidates without a >=3% improvement over the best, so pause this local tuning axis and re-assess the pipeline topology before Iteration 89.
## Iteration 89 — bank-skew the transient dequant LUT

- Change: restored the exact Iteration 85 N32/full-DP4A kernel and padded each shared dequant-LUT entry from two to three 32-bit words. Random scale entries can therefore start on all 32 shared-memory banks instead of only the 16 even banks. The LUT grows only `1,024 -> 1,536` transient bytes; persistent packed-FP4 storage and decoded values are unchanged.
- Correctness: the `[0,17,17,69,70]` routing/tail probe and the 128-token E1 cross-pass probe both pass with zero elementwise violations and unchanged precision metrics.
- Resources/SASS: the cubin remains `REG:72`, `LOCAL:0`, and `STACK:0`, preserving seven-CTA residency. Misaligned 12-byte entries require scalar loads, changing `8 LDS.64 -> 16 LDS`; initialization adds two STS and address generation adds seven IMAD. IDP, PRMT, WGMMA, and global-load counts are unchanged.
- Protocol: focused official `128x4096x7168/E8`, standard Cupra timing (`8` input slots, `10` warmups, `30` timed iterations, cold-L2 rotation).
- Result: `0.0968630 ms`, `77.5961` useful TFLOP/s, `26.2149%` compute SOL, `28.7539%` roofline SOL, `41.11%` jitter.
- Assessment: the result is effectively tied with but `0.10%` slower than Iteration 85 (`96.7675 -> 96.8630 us`). Reduced bank concentration only compensates for the extra scalar LDS/address work; it does not provide a net gain. Do not retain the padded LUT. Restore the exact Iteration 85 implementation and proceed to the approved N128 two-warpgroup CTA topology.
## Iteration 90 — two output warpgroups in an N128 CTA

- Change: restored Iteration 85, then templated the cooperative kernel for one or two output warpgroups. The official path uses a 256-thread N128 CTA: each warpgroup independently decodes and computes one N64 output half while sharing the N32 activation stage and LUT initialization. N64 shapes retain the original 128-thread fallback. Grid size is halved; packed-FP4 storage and per-output weight work are unchanged.
- Correctness: the N128 routing/tail probe, N128 128-token E1 cross-pass probe, and N64 fallback probe all pass with zero elementwise violations and no NaN/Inf.
- Resources/SASS: both specializations report `REG:72`, `LOCAL:0`, and `STACK:0`. The N128 path uses about 41 KiB dynamic shared memory and permits three CTAs/SM (24 resident warps); its per-thread IDP/PRMT/QGMMA/global-load counts are effectively unchanged from N64.
- Protocol: focused official `128x4096x7168/E8`, standard Cupra timing (`8` input slots, `10` warmups, `30` timed iterations, cold-L2 rotation).
- Result: `0.1558235 ms`, `48.2353` useful TFLOP/s, `16.2957%` compute SOL, `17.8740%` roofline SOL, `71.62%` jitter.
- Assessment: latency regresses `61.03%` versus Iteration 85 (`96.7675 -> 155.8235 us`). Halving activation/prologue work is negligible relative to packed-weight decode, while combining two independent WGMMA/decode streams in one CTA reduces resident CTA scheduling domains and leaves a 22-CTA tail. Reject this topology and restore Iteration 85; the next direction must reduce weight-decode work or synchronize less, not merely amortize activation staging.
## Iteration 91 — adapt WGMMA width to each expert pass

- Profiling basis: an IKET-instrumented, single-active-CTA `32x64x7168/E1` run measured a 53.248 us warp lifetime. Across 28 K256 stages, per-warp range sums were about 17.0-19.2 us for activation staging, 17.0-17.2 us for packed-weight load/decode, 9.47-9.54 us for WGMMA issue/wait, and 0.96-1.15 us for the two CTA barriers. The IKET injector crashed during final cleanup, but tracker/config generation succeeded and all four active warps produced complete decoded range data.
- Change: restored Iteration 85 and selected N8/N16/N24/N32 WGMMA at runtime from each pass's actual token count rounded to eight. Activation staging now writes only those rounded rows rather than an unconditional 32; accumulator storage remains the N32 maximum and the packed-weight decode/pass scheduler are unchanged.
- Correctness: a custom routing probe exercised all four WGMMA widths plus a 33-token second pass, and a 128-token E1 probe exercised four full passes. Both pass with zero elementwise violations and no NaN/Inf.
- Resources/SASS: the cubin remains `REG:72`, `LOCAL:0`, and `STACK:0`. Four static WGMMA branches raise code size (`32` static QGMMA and `21` BRA), but only one width executes per stage; global/decode instructions and seven-CTA residency are unchanged.
- Protocol: focused official `128x4096x7168/E8`, standard Cupra timing (`8` input slots, `10` warmups, `30` timed iterations, cold-L2 rotation).
- Result: `0.1008640 ms`, `74.5181` useful TFLOP/s, `25.1750%` compute SOL, `27.6133%` roofline SOL, `27.09%` jitter.
- Assessment: despite reducing padded activation and Tensor Core work, latency regresses `4.23%` versus Iteration 85 (`96.7675 -> 100.8640 us`). Repeated uniform width selection and four duplicated WGMMA code paths outweigh the saved work under the official expert distribution. Do not retain this monolithic adaptive branch; any future width specialization must move dispatch outside the K loop or use separately compact kernels.

## Iteration 92 — direct-register FP8 RS-WGMMA weights

- Change: restored the exact Iteration 85 N32 baseline and replaced the exact-byte path's shared-memory FP8 weight tile with `wgmma.m64n32k32` RS operands. Each thread decodes the packed NVFP4 values needed by the documented four-register A fragment directly into `uint32_t` registers; activation remains in 128-byte-swizzled shared memory. Two K32 fragments are issued per commit/wait group. Persistent storage remains the same packed FP4 codes plus original scales, and transient weight shared memory falls from `16 KiB` to zero for K256. The non-exact canonical fallback was also corrected to use its contiguous-K x4 decoder instead of the nibble-plane DP4A ordering.
- Correctness: exact-byte K128 and K256 probes with offsets `[0,17,17,69,70]` both pass with zero violations. K256 reproduces the retained metrics exactly (cosine `0.9997497201`, relative error `0.02211719`, maximum absolute difference `1.6875`). The canonical K128 public-API fallback also passes with zero violations, cosine `0.99976814`, and maximum absolute difference `1.0`.
- Resources/SASS: the official K256 specialization reports `REG:78`, `LOCAL:0`, `STACK:0`, and only `9,216` dynamic shared-memory bytes. Static instructions include `64 IDP`, `32 PRMT`, `51 LDG`, `32 LDS`, `7 STS`, `8 QGMMA`, and `8 WARPGROUP`; removing weight staging cuts STS but holding two A fragments lowers residency from seven to six CTAs per SM and adds four commit/wait groups per K256 stage.
- Protocol: focused official `128x4096x7168/E8`, standard Cupra timing (`8` input slots, `10` warmups, `30` timed iterations, cold-L2 rotation).
- Result: `0.1563055 ms`, `48.0866` useful TFLOP/s, `16.2455%` compute SOL, `17.8188%` roofline SOL, `44.25%` jitter.
- Assessment: latency regresses `61.53%` versus Iteration 85 (`96.7675 -> 156.3055 us`). The RS form removes FP8 weight SMEM traffic, but duplicated per-fragment packed/scale/LUT loads, four asynchronous completion groups per K256 tile, and the six-CTA residency limit dominate. Do not retain this implementation. A viable RS follow-up would need a fragment-major packed layout plus a lower-overhead register transpose or enough independent A-register buffering to amortize commit/wait without losing the resident wave; otherwise restore Iteration 85 and pursue a different structural axis.

## Iteration 93 — single-warpgroup sequential N128 CTA

- Change: restored Iteration 85 and halved the exact-byte small-M grid by letting one 128-thread warpgroup compute two adjacent N64 output tiles. Both decoded FP8 weight tiles are staged in separate transient shared-memory regions, while the N32 activation K256 tile is staged only once and reused by both output tiles. The warpgroup issues one eight-WGMMA commit group per N64 half and waits after both; unlike Iteration 90, no second warpgroup or producer/consumer scheduling domain is introduced. Persistent packed-FP4 storage is unchanged. The canonical fallback retains the corrected contiguous-K x4 decoder.
- Correctness: exact-byte K128 and K256 routing probes pass with zero violations and reproduce Iteration 85's precision metrics; the canonical K128 public-API probe also passes with zero violations.
- Resources/SASS: K256 reports `REG:80`, `LOCAL:0`, and `STACK:0`, with `41,984` dynamic shared-memory bytes. The 256 base CTAs still fit in one resident wave. Static work becomes `128 IDP`, `64 PRMT`, `31 LDG`, `16 LDS`, `23 STS`, `16 QGMMA`, and `2 WARPGROUP`; activation staging and CTA count halve, while packed-weight decode and useful Tensor Core work remain doubled per CTA.
- Protocol: focused official `128x4096x7168/E8`, standard Cupra timing (`8` input slots, `10` warmups, `30` timed iterations, cold-L2 rotation).
- Result: `0.1325125 ms`, `56.7206` useful TFLOP/s, `19.1624%` compute SOL, `21.0183%` roofline SOL, `7.93%` jitter.
- Assessment: N128 single-warpgroup execution is substantially more stable and `15.22%` faster than the failed RS candidate, but it still regresses `36.94%` versus Iteration 85 (`96.7675 -> 132.5125 us`). Halving activation staging and scheduling domains does not repay the doubled decode/WGMMA critical path. Reject N128 aggregation and restore exact Iteration 85; reducing activation duplication must not serialize two complete weight streams in one CTA.

## Iteration 94 — overlap FP8 activation TMA with packed-weight decode

- Change: restored the exact Iteration 85 N32 kernel and replaced its per-thread activation LDG/STS loop with a host-created FP8 TensorMap and one transaction barrier. One elected thread issues one K128 TMA slice (or two slices for K256) before the full warpgroup decodes packed NVFP4 weights, so activation transfer and CUDA-core decode can overlap. The reused activation tile is cleared once per token pass to make the final global-M OOB tail deterministic; packed FP4 weights, exact-byte prepared payload, WGMMA work, and the two-way pass scheduler are unchanged. The canonical non-exact fallback also restores its contiguous-nibble x4 decoder.
- Correctness: exact-byte K128 and K256 probes with offsets `[0,17,17,69,70]` both pass with zero violations. K256 reproduces the retained Iteration 85 metrics exactly (cosine `0.9997497201`, relative error `0.02211719`, maximum absolute difference `1.6875`); K128 reports cosine `0.9997668266` and maximum absolute difference `1.125`. The official K7168 timing repeatedly reuses all 28 barrier phases without a hang or asynchronous fault.
- Protocol: focused official `128x4096x7168/E8`, standard Cupra timing (`8` input slots, `10` warmups, `30` timed iterations, cold-L2 rotation) on an otherwise idle H20-3e.
- Result: `0.0858245 ms`, `87.5763` useful TFLOP/s, `29.5866%` compute SOL, `32.4521%` roofline SOL, `43.83%` jitter.
- Assessment: latency improves `11.31%` versus Iteration 85 (`96.7675 -> 85.8245 us`), validating direct TMA/decode overlap as the first post-Iter85 structural gain. The kernel remains `2.16x` above the `39.788 us` / `70%` gate, and activation staging has not disappeared completely because each token pass still pays a one-time 8 KiB SMEM clear plus one CTA barrier. Retain this as the new best and inspect generated resources/SASS before choosing between removing the clear for non-final experts, adding a second activation stage, or attacking the remaining weight-decode path.

## Iteration 95 — remove the post-WGMMA CTA barrier

- Change: retained Iteration 94's TMA/decode overlap and removed the CTA-wide barrier after `warpgroup_wait<0>()` in each K256 stage. The kernel contains exactly one warpgroup, and the wait guarantees that its prior asynchronous WGMMA reads have completed before the same threads overwrite the shared weight and activation tiles in the next stage; the pre-WGMMA synchronization remains unchanged.
- Correctness/stability: exact-byte K128 and K256 probes remain at zero violations with unchanged precision metrics. A K2048 multi-stage probe ran the same prepared input 21 times and every output was bit-identical (`0` mismatched elements on all 20 repeats); its 36 checker violations are the previously accepted single-E4M3 representation error, not a synchronization race.
- Protocol: focused official `128x4096x7168/E8`, standard Cupra timing (`8` input slots, `10` warmups, `30` timed iterations, cold-L2 rotation) on an idle H20-3e.
- Result: `0.0851195 ms`, `88.3017` useful TFLOP/s, `29.8316%` compute SOL, `32.7209%` roofline SOL, `41.49%` jitter.
- Assessment: removing 28 redundant CTA barriers improves latency by `0.82%` versus Iteration 94 (`85.8245 -> 85.1195 us`). The gain is real but small because the prior WGMMA wait and the decode/TMA synchronization dominate stage ordering. Retain the simpler barrier protocol as the new best; the remaining `2.14x` gap to `39.788 us` requires reducing active-pass work or weight decode rather than further post-WGMMA synchronization tuning.

## Iteration 96 — clear only the global-M activation tail

- Change: retained Iteration 95 and replaced the unconditional 8 KiB activation-SMEM clear at the start of every active token pass with a uniform tail branch. TMA is allowed to read unused rows from the following expert because those accumulator columns are never stored; only rows beyond global `shape_m`, which TMA skips, are explicitly zeroed in the swizzled K128 slabs. Most active CTAs therefore skip both the clear and its barrier.
- Correctness: the `[0,17,17,69,70]` probe specifically exercises a one-token final expert with 31 global-OOB rows. Both K128 and K256 pass with zero violations and unchanged precision metrics, confirming the selective swizzled-tail addressing.
- Protocol: focused official `128x4096x7168/E8`, standard Cupra timing (`8` input slots, `10` warmups, `30` timed iterations, cold-L2 rotation) on an idle H20-3e.
- Result: `0.0856315 ms`, `87.7737` useful TFLOP/s, `29.6533%` compute SOL, `32.5252%` roofline SOL, `47.60%` jitter.
- Assessment: despite skipping the clear for most active CTAs, latency regresses `0.60%` versus Iteration 95 (`85.1195 -> 85.6315 us`). The extra uniform bounds arithmetic, division/remainder lowering, and tail branch offset the small reduction in one-time pass setup under this noisy cold-L2 workload. Do not retain the selective path; restore Iteration 95's simple unconditional clear before changing a higher-impact axis.

## Iteration 97 — force eight-CTA register residency

- Change: restored Iteration 95's unconditional activation clear and raised the cooperative kernel launch bound from six to eight resident CTAs per SM. Iteration 94 had reduced K256 register use to 66, so the compiler was asked to fit the same code into the 64-register occupancy boundary; dynamic SMEM (`25,608` bytes/CTA) already permits eight CTAs on H20.
- Correctness/resources: K128 and K256 exact-byte probes remain at zero violations. K256 compiles at `REG:63`, `LOCAL:0`, `STACK:0` (down from Iteration 95's 66 registers), and K128 compiles at `REG:56`, so eight-CTA residency is physically achievable without spills.
- Protocol: focused official `128x4096x7168/E8`, standard Cupra timing (`8` input slots, `10` warmups, `30` timed iterations, cold-L2 rotation) on an idle H20-3e.
- Result: `0.0943685 ms`, `79.6473` useful TFLOP/s, `26.9079%` compute SOL, `29.5139%` roofline SOL, `4.27%` jitter.
- Assessment: latency regresses `10.87%` versus Iteration 95 (`85.1195 -> 94.3685 us`) despite eliminating the theoretical residency tail and dramatically reducing jitter. Constraining the hot decode/TMA/WGMMA schedule to 63 registers lengthens each CTA more than the eighth resident CTA helps. Reject the eight-CTA launch bound and restore Iteration 95's six-minimum/66-register code; occupancy above seven is not a profitable axis.

## Iteration 98 — prefetch the next packed-weight tile into L2

- Stall reassessment: Iterations 95–97 produced three consecutive candidates without a >=3% improvement over Iteration 94. NCU remains unavailable by prior policy, so the reassessment combined Iteration 91's IKET breakdown, Iteration 94/95 SASS, the iteration history, official Hopper TMA guidance, and CUTLASS's SM90 TMA/WGMMA pipelining patterns. The resulting new axis is to overlap packed-weight movement with current-stage WGMMA rather than continue occupancy or barrier micro-tuning.
- Change: restored Iteration 95 and added a software L2 prefetch for the next exact-byte K256 weight tile after current decode and before the activation wait/WGMMA sequence. For each K128 subtile, one lane per aligned 128-byte line prefetches the two packed vectors and one scale line owned by its warp. Persistent packed-FP4 bytes, actual decode loads, SMEM use, and numerical work are unchanged.
- Correctness/resources: K128 and K256 probes remain at zero violations. The K256 cubin contains six `CCTL.E.PF2` instructions, confirming that PTX prefetches were retained; it reports `REG:70`, `LOCAL:0`, `STACK:0`, so seven-CTA residency remains possible but register use rises from 66.
- Protocol: focused official `128x4096x7168/E8`, standard Cupra timing (`8` input slots, `10` warmups, `30` timed iterations, cold-L2 rotation) on an idle H20-3e.
- Result: `0.0962075 ms`, `78.1248` useful TFLOP/s, `26.3935%` compute SOL, `28.9498%` roofline SOL, `26.47%` jitter.
- Assessment: explicit prefetch regresses `13.03%` versus Iteration 95 (`85.1195 -> 96.2075 us`). The extra address-generation/register lifetime and cache-control issue stream cost more than any latency hidden behind eight WGMMA instructions; independent CTAs already hide the coalesced direct-LDG latency effectively. Reject software prefetch. A viable weight-transfer redesign must remove the direct LDG instruction stream via TMA staging, not duplicate it with cache hints.

## Iteration 99 — TMA-stage packed FP4 weights and scales

- Change: restored Iteration 95 and added two no-swizzle TensorMaps over the existing exact-byte payload, interpreted as 16-byte chunks without changing a single persistent byte. For every K256 stage, TMA copies four `[N64,16B]` packed vectors and two `[N64,4B]` scale vectors per K128 subtile into one 9,216-byte transient SMEM buffer. Threads decode from LDS rather than direct weight LDG. After decode and the existing CTA synchronization, the same compact buffer is refilled with the next K256 tile while current WGMMA executes. Activation retains its independent TMA/barrier. The shared runtime dtype mapping gains the missing `torch.uint8 -> CU_TENSOR_MAP_DATA_TYPE_UINT8` case required to encode the packed TensorMaps.
- Correctness/stability: exact-byte K128 and K256 routing probes both pass with zero violations and unchanged precision metrics. A K2048 pipeline probe is bit-identical across 20 repeats; its 36 reference violations remain the accepted single-E4M3 representation error.
- Resources/SASS: K256 compiles at `REG:52`, `LOCAL:0`, `STACK:0`, confirming removal of long-lived global weight addresses/data, with 26 static `UTMALDG` instructions across the activation and two inlined packed-weight issue sites. Dynamic SMEM rises from `25,608` to `34,832` bytes, reducing residency from seven to six CTAs/SM.
- Protocol: focused official `128x4096x7168/E8`, standard Cupra timing (`8` input slots, `10` warmups, `30` timed iterations, cold-L2 rotation) on an idle H20-3e.
- Result: `0.1244790 ms`, `60.3812` useful TFLOP/s, `20.3991%` compute SOL, `22.3747%` roofline SOL, `57.25%` jitter.
- Assessment: latency regresses `46.24%` versus Iteration 95 (`85.1195 -> 124.4790 us`). The twelve small weight TMA requests per K256 stage, extra transaction-barrier phase, LDS traffic, and six-CTA scheduling tail dominate; four WGMMA K32 operations per subtile do not provide enough time to hide the next packed transfer. Reject packed-weight TMA staging. The direct coalesced LDG+decode path is substantially better, so the next high-impact direction must reduce decode work or the number of active weight scans rather than add another memory hierarchy level.

## Iteration 100 — bit-woven FP4 layout with direct PRMT selectors

- Change: restored Iteration 95 and replaced only the lossless small-M prepared bit permutation plus exact-byte decoder. Each 32-bit word still stores eight genuine E2M1 codes, but K0:K3 magnitudes occupy the low PRMT selector nibbles, K4:K7 magnitudes occupy the high selector nibbles, K0:K3 signs already sit in FP8 byte-sign positions, and K4:K7 signs reach those positions with one shift. Two existing LUT PRMTs therefore emit the two contiguous FP8 words directly, eliminating all DP4A selector packing without adding PRMTs or persistent bytes.
- Correctness/storage: an explicit inverse permutation reconstructs canonical packed weights and scales byte-for-byte. Canonical and prepared totals both equal `73,744` bytes on the K256 audit shape (and preserve the official `132,120,608`-byte formula). K128 and K256 routing probes pass with zero violations and unchanged precision metrics.
- Resources/SASS: K256 changes from Iteration 95's `REG:66`, `64 IDP`, and `130 IMAD` to `REG:62`, `0 IDP`, and `98 IMAD`; `PRMT:36`, `LDG:17`, `UTMALDG:2`, and `QGMMA:8` are unchanged. `LOCAL/STACK` remain zero. The lower register count also makes eight-CTA residency possible under the unchanged 25,608-byte SMEM allocation.
- Protocol: focused official `128x4096x7168/E8`, standard Cupra timing (`8` input slots, `10` warmups, `30` timed iterations, cold-L2 rotation) on an idle H20-3e.
- Result: `0.0906235 ms`, `82.9387` useful TFLOP/s, `28.0198%` compute SOL, `30.7336%` roofline SOL, `9.56%` jitter.
- Assessment: despite removing the entire DP4A selector stream, latency regresses `6.47%` versus Iteration 95 (`85.1195 -> 90.6235 us`). The result is nevertheless `3.97%` faster than Iteration 97's explicitly register-constrained eight-CTA version, suggesting the decoder itself helped but crossing from seven to eight resident CTAs introduced resource contention/scheduling overhead. Keep the bit-woven design as a candidate and isolate residency next by padding dynamic SMEM just above the eight-CTA boundary without changing registers or instructions.

## Iteration 101 — retain seven CTAs for the bit-woven decoder

- Change: kept Iteration 100's bit-woven packed-FP4 representation and decoder byte-for-byte, but added 4,096 unused dynamic-SMEM bytes only to the K256 small-M launch. Total requested dynamic SMEM rises from `25,608` to `29,704` bytes, crossing the eight-CTA capacity boundary while still fitting seven CTAs. Kernel SASS, `REG:62`, global traffic, useful work, and persistent storage are unchanged.
- Correctness: numerical and storage paths are identical to Iteration 100, whose inverse-permutation audit and K128/K256 zero-violation probes passed. The only change is launch resource reservation.
- Protocol: focused official `128x4096x7168/E8`, standard Cupra timing (`8` input slots, `10` warmups, `30` timed iterations, cold-L2 rotation) on an idle H20-3e.
- Result: `0.0814070 ms`, `92.3286` useful TFLOP/s, `31.1921%` compute SOL, `34.2131%` roofline SOL, `50.72%` jitter.
- Assessment: restoring seven-CTA residency improves latency `10.17%` versus the otherwise identical Iteration 100 (`90.6235 -> 81.4070 us`) and `4.36%` versus the prior overall best Iteration 95 (`85.1195 -> 81.4070 us`). This isolates the bit-woven decoder as a real win: zero IDP plus seven, rather than eight, resident CTAs is the new best. Retain both the lossless layout and the residency pad; the kernel remains `2.05x` above the `39.788 us` / `70%` gate, so continue reducing decode/global-load work without crossing the occupancy boundary.

## Iteration 102 — serialize token passes in a compact grid

- Change: retained Iteration 101's bit-woven decoder and seven-CTA resource profile, but changed the pass split from two to one in both host grid sizing and kernel scheduling. The grid falls from 1,024 to 512 blocks; each `(expert,N64)` CTA processes every 32-token pass serially. Only three of the eight official input slots contain an expert above 32 tokens, so this tests whether eliminating inactive second-pass blocks beats parallelizing those rare extra scans.
- Correctness: the standard `[0,17,17,69,70]` probe includes a 52-token expert and therefore exercises two serial K-loop passes in one CTA. K128 and K256 both pass with zero violations and unchanged precision metrics.
- Protocol: focused official `128x4096x7168/E8`, standard Cupra timing (`8` input slots, `10` warmups, `30` timed iterations, cold-L2 rotation) on an idle H20-3e.
- Result: `0.0817740 ms`, `91.9142` useful TFLOP/s, `31.0521%` compute SOL, `34.0595%` roofline SOL, `45.58%` jitter.
- Assessment: the compact grid is effectively tied but `0.45%` slower than Iteration 101 (`81.4070 -> 81.7740 us`). Early-return second-pass blocks are cheap enough that removing them does not repay serializing the few long experts; the earlier Iteration 81/83 tie remains true after TMA and bit-woven decode. Restore the two-way pass split and keep Iteration 101 as best.

## Iteration 103 — round physical token work to N16 chunks

- Change: restored Iteration 101's two-way pass split, bit-woven FP4 decoder, activation TMA overlap, and seven-CTA resource reservation. Replaced each fixed N32 WGMMA with one or two N16 operations selected from the pass's actual token count. This rounds physical Tensor Core rows to 16 rather than always 32 without adding persistent bytes, changing the packed-FP4 payload, or duplicating the four WGMMA-width code paths from Iteration 91.
- Correctness: the standard K128 and K256 routing/tail probes both pass with zero violations and reproduce Iteration 101's metrics exactly. K256 reports cosine `0.9997497201`, relative error `0.02211719`, and maximum absolute difference `1.6875`.
- Resources/SASS: K256 remains `REG:62`, `LOCAL:0`, and `STACK:0`, so the 29,704-byte launch still permits seven CTAs per SM. Static QGMMA rises `8 -> 16` because both possible N16 chunks are present; the candidate contains `17 LDG`, `8 LDS`, `16 STS`, `40 PRMT`, `100 IMAD`, `46 SHF`, `2 UTMALDG`, and `4 BAR` instructions.
- Protocol: focused official `128x4096x7168/E8`, standard Cupra timing (`8` input slots, `10` warmups, `30` timed iterations, cold-L2 rotation) on an idle H20-3e.
- Result: `0.0885110 ms`, `84.9182` useful TFLOP/s, `28.6886%` compute SOL, `31.4671%` roofline SOL, `33.01%` jitter.
- Assessment: despite reducing physical token rows substantially, latency regresses `8.73%` versus Iteration 101 (`81.4070 -> 88.5110 us`). The extra N16 WGMMA issue/descriptors and uniform loop/control overhead outweigh the reduced Tensor Core work, consistent with the earlier monotonic N8/N16/N32 issue-width results. Reject chunked N16 and keep Iteration 101 as best. Further adaptive work must retain one WGMMA instruction per K32, with width dispatch moved out of the hot K loop or otherwise made compact.

## Iteration 104 — choose one N16 or N32 WGMMA per pass

- Change: retained Iteration 103's 16-token physical-row rounding, but replaced the one-or-two N16 loop with a uniform two-way choice: passes with at most 16 active tokens issue one N16 WGMMA per K32, while wider passes issue one N32 WGMMA. This restores one QGMMA instruction per K32 for long passes while keeping the bit-woven decoder, activation TMA, packed-FP4 payload, two-way pass split, and seven-CTA topology unchanged.
- Correctness: the standard K128 and K256 routing/tail probes both pass with zero violations and exactly reproduce the retained numerical metrics.
- Resources/SASS: K256 reports `REG:64`, `LOCAL:0`, and `STACK:0`, preserving seven-CTA residency. Both uniform paths produce `16` static QGMMA instructions and `21` branches; the cubin also contains `17 LDG`, `8 LDS`, `16 STS`, `41 PRMT`, `100 IMAD`, `48 SHF`, `2 UTMALDG`, and `4 BAR` instructions.
- Protocol: focused official `128x4096x7168/E8`, standard Cupra timing (`8` input slots, `10` warmups, `30` timed iterations, cold-L2 rotation) on an idle H20-3e.
- Result: `0.0864320 ms`, `86.9608` useful TFLOP/s, `29.3786%` compute SOL, `32.2240%` roofline SOL, `34.17%` jitter.
- Assessment: selecting one width recovers `2.35%` versus Iteration 103 (`88.5110 -> 86.4320 us`), proving that duplicated N16 issue was costly, but remains `6.17%` slower than Iteration 101 (`81.4070 us`). Even the reduced two-path hot-loop dispatch/code footprint costs more than the physical Tensor Core work it removes. Reject the inline N16/N32 selector and restore Iteration 101; any viable width specialization must remove width dispatch from the K loop entirely.

## Iteration 105 — dispatch N16/N32 outside the complete K loop

- Change: retained Iteration 104's one-instruction N16/N32 policy but lifted the uniform width choice outside the complete K loop. A templated pass body emits separate compact N16 and N32 loops, so each CTA branches once per token pass rather than once in every K256 stage. The packed bit-woven FP4 bytes, activation TMA, decode arithmetic, pass split, and seven-CTA resource reservation are unchanged.
- Correctness: K128 and K256 routing/tail probes pass with zero violations and reproduce the retained precision metrics exactly.
- Resources/SASS: K256 remains `REG:64`, `LOCAL:0`, and `STACK:0`, retaining seven-CTA residency. Both specialized bodies make the static cubin larger (`16 QGMMA`, `23 LDG`, `16 LDS`, `24 STS`, `73 PRMT`, `162 IMAD`, `78 SHF`, `4 UTMALDG`, `6 BAR`, `30 BRA`), but each CTA executes only its selected body.
- Protocol: focused official `128x4096x7168/E8`, standard Cupra timing (`8` input slots, `10` warmups, `30` timed iterations, cold-L2 rotation) on an idle H20-3e.
- Result: `0.0807525 ms`, `93.0769` useful TFLOP/s, `31.4449%` compute SOL, `34.4904%` roofline SOL, `41.84%` jitter.
- Assessment: moving dispatch out of the K loop improves `6.57%` versus Iteration 104 (`86.4320 -> 80.7525 us`) and `0.80%` versus the prior best Iteration 101 (`81.4070 us`). This validates outer width specialization, although the gain is still small relative to the `39.788 us` target and the doubled static code footprint is not free. Keep Iteration 105 as the new best and next reduce the inactive-path/code footprint or add a compact N8 tail only if it preserves one QGMMA per K32.

## Iteration 106 — overlap the second K128 decode with WGMMA

- Change: retained Iteration 105's outer N16/N32 specialization and K256 shared-memory footprint, but split each K256 stage into two asynchronous WGMMA groups. After decoding and issuing the first K128 half, the warpgroup commits it, decodes the second K128 half into its disjoint shared-memory slab while the first group is pending, then issues the second half and waits for both groups. Persistent packed bytes, global traffic, activation TMA requests, and seven-CTA occupancy are unchanged.
- Correctness: K128 and K256 routing/tail probes both pass with zero violations and reproduce the retained numerical metrics, validating the pending-WGMMA/shared-slab lifetime ordering.
- Resources/SASS: K256 remains `REG:64`, `LOCAL:0`, and `STACK:0`. The two selected code bodies contain `16 QGMMA`, `23 LDG`, `16 LDS`, `24 STS`, `74 PRMT`, `165 IMAD`, `78 SHF`, `4 UTMALDG`, `10 BAR`, and `6 WARPGROUP` instructions; the extra group and midpoint CTA synchronization do not change occupancy.
- Protocol: focused official `128x4096x7168/E8`, standard Cupra timing (`8` input slots, `10` warmups, `30` timed iterations, cold-L2 rotation) on an idle H20-3e.
- Result: `0.0814400 ms`, `92.2912` useful TFLOP/s, `31.1794%` compute SOL, `34.1992%` roofline SOL, `45.11%` jitter.
- Assessment: the attempted same-warpgroup overlap is `0.85%` slower than Iteration 105 (`80.7525 -> 81.4400 us`). Four first-half WGMMA instructions do not hide enough decode to repay a second commit/fence and midpoint CTA barrier; this matches Iteration 80's broader negative result without its occupancy loss. Reject the split K128 grouping and restore Iteration 105 as best.

## Iteration 107 — specialize all N8/N16/N24/N32 widths outside K

- Change: restored Iteration 105, then extended its outer pass dispatch from N16/N32 to N8/N16/N24/N32. Every active pass now rounds physical Tensor Core rows to the nearest eight while still issuing exactly one width-matched WGMMA per K32. The width choice remains outside the complete K loop; packed bit-woven FP4 bytes, activation TMA, decode work, two-way pass split, and dynamic shared memory are unchanged.
- Correctness: K128 and K256 routing/tail probes pass with zero violations and reproduce the retained numerical metrics exactly.
- Resources/SASS: the four specialized bodies compile at `REG:66`, `LOCAL:0`, and `STACK:0`, preserving seven-CTA residency. Cubin size rises to `39,368` bytes, with `32 QGMMA`, `35 LDG`, `32 LDS`, `40 STS`, `145 PRMT`, `285 IMAD`, `146 SHF`, `8 UTMALDG`, `10 BAR`, and `54 BRA` statically; each CTA executes only the body selected for its expert pass.
- Protocol: focused official `128x4096x7168/E8`, standard Cupra timing (`8` input slots, `10` warmups, `30` timed iterations, cold-L2 rotation) on an idle H20-3e.
- Result: `0.0777920 ms`, `96.6191` useful TFLOP/s, `32.6416%` compute SOL, `35.8030%` roofline SOL, `40.35%` jitter.
- Assessment: exact outer width specialization improves `3.67%` versus Iteration 105 (`80.7525 -> 77.7920 us`) and `4.44%` versus Iteration 101 (`81.4070 us`). The additional physical-work reduction outweighs the larger static code body because each expert's contiguous N tiles reuse one selected path. Keep Iteration 107 as the new best. It remains `1.96x` above the `39.788 us` gate, so the next structural target is the packed-weight decode/load stream rather than further token-width tuning.

## Iteration 108 — reuse the residency pad for an N48 pass

- Change: expanded the token pass from 32 to 48 and added outer N40/N48 WGMMA specializations. The existing 4,096-byte K256 residency pad was converted into real activation storage, so requested dynamic shared memory remains exactly `29,704` bytes. Experts with 33–48 tokens now scan/decode their packed weights once instead of running a second pass; persistent FP4/scale bytes and per-scan weight traffic are unchanged.
- Correctness: the standard probe's 52-token expert exercises an N48 first pass plus an N8 second pass. K128 and K256 both pass with zero violations and retain the same numerical metrics.
- Resources/SASS: K256 reaches `REG:72`, `LOCAL:0`, and `STACK:0`, exactly retaining seven-CTA residency. The six specialized bodies expand the cubin to `55,752` bytes and contain `48 QGMMA`, `51 LDG`, `48 LDS`, `84 STS`, `217 PRMT`, `458 IMAD`, `214 SHF`, `12 UTMALDG`, `14 BAR`, and `94 BRA` statically.
- Protocol: focused official `128x4096x7168/E8`, standard Cupra timing (`8` input slots, `10` warmups, `30` timed iterations, cold-L2 rotation) on an idle H20-3e.
- Result: `0.0787675 ms`, `95.4225` useful TFLOP/s, `32.2373%` compute SOL, `35.3596%` roofline SOL, `12.24%` jitter.
- Assessment: eliminating the rare long-expert rescan lowers jitter substantially but regresses median latency `1.25%` versus Iteration 107 (`77.7920 -> 78.7675 us`). Only three of eight official slots contain an expert above 32 tokens, while every slot pays the larger instruction footprint and higher register pressure. Reject N48 and restore Iteration 107's N32 pass plus residency pad.

## Iteration 109 — force six-CTA residency

- Change: restored Iteration 107 exactly, then increased only the unused K256 dynamic-shared-memory reservation from `4,096` to `8,192` bytes. Requested K256 shared memory rises from `29,704` to `33,800` bytes, forcing six rather than seven resident CTAs without changing kernel instructions, registers, persistent bytes, numerical work, or memory accesses.
- Correctness/resources: the executable and all numerical paths are byte-identical to Iteration 107; only the launch resource reservation changes. K256 remains `REG:66`, `LOCAL:0`, and `STACK:0`.
- Protocol: focused official `128x4096x7168/E8`, standard Cupra timing (`8` input slots, `10` warmups, `30` timed iterations, cold-L2 rotation) on an idle H20-3e.
- Result: `0.1018235 ms`, `73.8159` useful TFLOP/s, `24.9378%` compute SOL, `27.3531%` roofline SOL, `55.95%` jitter.
- Assessment: six-CTA residency regresses `30.89%` versus Iteration 107 (`77.7920 -> 101.8235 us`). Unlike the beneficial eight-to-seven transition, seven-to-six leaves a 44-CTA second scheduling wave and removes too much latency-hiding capacity. Reject the larger pad and retain seven CTAs as the clear occupancy optimum.

## Iteration 110 — lane-paired direct-register RS WGMMA

- Change: restored Iteration 107 and redesigned the exact-byte path around RS WGMMA weight operands. For each register fragment, paired lanes load different output rows instead of redundantly loading the same packed word, reconstruct both row fragments with one XOR-lane shuffle, and broadcast each scale word across its four-lane fragment group. Bit-woven FP4 decoding and activation TMA are retained; decoded weights stay in registers and the 16 KiB FP8 weight shared-memory stores are removed. The RS selector/helper and integer operand fence overload are reintroduced only for this path.
- Correctness: exact K128 and K256 routing/tail probes both pass with zero violations and reproduce Iteration 107's numerical metrics exactly, validating the optimized fragment layout and lane exchange.
- Resources/SASS: K256 reaches `REG:72`, `LOCAL:0`, and `STACK:0`, preserving seven CTAs. Static weight staging drops to `8 STS`, but the four width-specialized RS bodies contain `32 QGMMA`, `91 LDG`, `64 LDS`, `139 PRMT`, `80 SHFL`, `303 IMAD`, `32 WARPGROUP`, `8 UTMALDG`, and `10 BAR`; each K256 tile requires four two-fragment commit/wait groups.
- Protocol: focused official `128x4096x7168/E8`, standard Cupra timing (`8` input slots, `10` warmups, `30` timed iterations, cold-L2 rotation) on an idle H20-3e.
- Result: `0.1008175 ms`, `74.5525` useful TFLOP/s, `25.1866%` compute SOL, `27.6260%` roofline SOL, `42.67%` jitter.
- Assessment: optimized RS is `29.60%` slower than Iteration 107 (`77.7920 -> 100.8175 us`). Removing weight SMEM traffic and duplicate global loads is insufficient: four RS completion groups per K256, register-fragment shuffles, and narrower asynchronous batches dominate. Reject this pair-at-a-time RS path. Any further RS attempt must pipeline pending groups with two fragment buffers rather than waiting after every pair.

## Iteration 111 — keep one RS group pending during decode

- Change: retained Iteration 110's lane-paired RS layout and two-fragment register footprint, but issued the first K32 fragment as its own committed group before decoding the second fragment. The first RS group therefore remains pending while CUDA cores build the second fragment; after committing the second group, one `wait<0>` retires both. Global bytes, bit-woven arithmetic, activation TMA, registers, and dynamic shared memory are unchanged.
- Correctness: exact K128 and K256 probes both pass with zero violations and reproduce the retained numerical metrics.
- Resources/SASS: K256 remains `REG:72`, `LOCAL:0`, and `STACK:0`, preserving seven CTAs. Static instruction counts are nearly identical to Iteration 110 except WARPGROUP operations rise `32 -> 48` and IMAD rises `303 -> 306`; the path still has `32 QGMMA`, `91 LDG`, `64 LDS`, `139 PRMT`, `80 SHFL`, and `8 STS`.
- Protocol: focused official `128x4096x7168/E8`, standard Cupra timing (`8` input slots, `10` warmups, `30` timed iterations, cold-L2 rotation) on an idle H20-3e.
- Result: `0.0985605 ms`, `76.2597` useful TFLOP/s, `25.7634%` compute SOL, `28.2586%` roofline SOL, `39.35%` jitter.
- Assessment: pending-group overlap improves `2.24%` versus Iteration 110 (`100.8175 -> 98.5605 us`) but remains `26.70%` slower than Iteration 107 (`77.7920 us`). The improvement confirms some decode/WGMMA overlap, yet the RS fragment shuffle/load stream and eight single-instruction commit groups per K256 remain too costly. Stop the RS axis and restore the shared-weight SS path.

## Iteration 112 — two warpgroup cooperative decode

- Change: restored Iteration 107's shared-weight SS path, expanded the CTA from 128 to 256 threads, and assigned each warpgroup one K128 half of every K256 weight tile. Both warpgroups therefore decode cooperatively with half the per-thread packed-weight work; only the first warpgroup issues WGMMA and writes output. A post-WGMMA CTA barrier prevents the second warpgroup from overwriting weight SMEM before completion. Persistent bytes, activation TMA, exact width specialization, and K256 shared-memory size are unchanged.
- Correctness: K128 and K256 routing/tail probes pass with zero violations and retain the same numerical metrics.
- Resources/SASS: `__launch_bounds__(256,4)` compiles at `REG:56`, `LOCAL:0`, and `STACK:0`, allowing four CTAs/SM and eight resident decode warpgroups. Static counts include `32 QGMMA`, `23 LDG`, `16 LDS`, `24 STS`, `81 PRMT`, `242 IMAD`, `106 SHF`, `8 UTMALDG`, `14 BAR`, and `16 WARPGROUP`.
- Protocol: focused official `128x4096x7168/E8`, standard Cupra timing (`8` input slots, `10` warmups, `30` timed iterations, cold-L2 rotation) on an idle H20-3e.
- Result: `0.1008960 ms`, `74.4945` useful TFLOP/s, `25.1670%` compute SOL, `27.6045%` roofline SOL, `11.78%` jitter.
- Assessment: doubling per-CTA decode capacity regresses `29.70%` versus Iteration 107 (`77.7920 -> 100.8960 us`). Four 256-thread CTAs require about 1.64 active waves for the 512 base tiles, and the added post-WGMMA barrier/idle second warpgroup outweigh the half-length decode section. Reject the two-warpgroup cooperative topology and retain the single-wave seven-CTA design.

## Iteration 113 — packed-weight load-only lower bound

- Diagnostic change: restored Iteration 107's 128-thread launch, exact pass mapping, K256 layout, grid, and 29,704-byte dynamic-SMEM reservation, then replaced the functional body with the production coalesced packed-FP4/scale loads plus one checksum store per thread. Dequantization, activation TMA, shared stores, WGMMA, and numerical output are intentionally omitted. This is a non-functional lower-bound measurement, not a candidate solution.
- Validation/resources: the probe intentionally fails numerics. K256 compiles at `REG:38`, `LOCAL:0`, `STACK:0`, and no static shared memory, with `8 LDG`, `41 IMAD`, `12 LOP3`, `10 SHF`, no QGMMA, and no STS. Launch-time dynamic SMEM still holds residency at seven CTAs, matching Iteration 107's scheduling condition.
- Protocol: focused `128x4096x7168/E8`, the same `8` input slots, `10` warmups, and `30` cold-L2 timed iterations used by the official benchmark; reported SOL is diagnostic only because output is invalid.
- Result: `0.0496160 ms`, corresponding to `56.1348%` of the minimum-byte roofline, with `28.52%` jitter.
- Assessment: the production packed-weight address stream alone already exceeds the `39.788 us` target by `24.70%`, before any decode or GEMM work. Iteration 107's remaining `28.176 us` above this lower bound is decode/Tensor Core/activation/epilogue cost, but optimizing only that work cannot reach 70% SOL. The next required axis is a more efficient lossless packed-byte transfer layout or pass schedule that raises cold-L2 bandwidth; restore the functional Iteration 107 kernel before implementing it.

## Iteration 114 — N64-tiled single-request TMA lower bound

- Diagnostic change: losslessly reordered each exact K128 payload from `[72,N]` into contiguous `[N/64,72,64]` tiles while preserving every FP4 and scale byte. Added API recognition for the alternate four-dimensional view and the missing uint8 TensorMap dtype. The diagnostic kernel issues one no-swizzle 4,608-byte TMA request per `(expert,K128,N64)` tile, reads the staged bytes into a checksum, and omits dequantization/GEMM. The existing TensorMap argument is temporarily repurposed for weights. This is intentionally non-functional.
- Validation/resources: the diagnostic executes successfully and fails numerics as expected. K256 compiles at `REG:33`, `LOCAL:0`, `STACK:0`, with one looped UTMALDG, `35 LDS`, no QGMMA, and no STS. Prepared payload size is unchanged; only its byte permutation and tensor view differ.
- Protocol: focused `128x4096x7168/E8`, the same `8` input slots, `10` warmups, and `30` cold-L2 timed iterations; reported SOL is a non-functional transfer lower bound.
- Result: `0.0542080 ms`, corresponding to `51.3796%` minimum-byte roofline SOL, with `68.75%` jitter.
- Assessment: one large TMA request per K128 is `9.26%` slower than Iteration 113's direct-LDG lower bound (`49.6160 -> 54.2080 us`). Repeated transaction-barrier waits plus shared-memory checksum reads outweigh fewer global issue instructions, even after eliminating Iteration 99's twelve small requests per K256. Reject packed-weight TMA; restore Iteration 107's direct coalesced loads and original exact-byte view.

## Iteration 115 — direct-LDG L2 256-byte promotion hint

- Diagnostic change: restored Iteration 113's original lossless layout and load-only body, then replaced the two 128-bit packed loads and one 32-bit scale load with inline PTX carrying the `L2::256B` prefetch-size hint. A preliminary attempt to combine each row into one 256-bit load was rejected at compilation because Hopper only supports that load width from SM100 onward; it was not timed or retained. This diagnostic remains intentionally non-functional.
- Validation/resources: SASS confirms `LDG.E.LTC256B.128` for packed vectors and `LDG.E.LTC256B` for scales. The K256 diagnostic uses `REG:40`, `LOCAL:0`, `STACK:0`, no QGMMA, and no shared-memory operations; output fails numerics as expected.
- Protocol: focused `128x4096x7168/E8`, the same `8` slots, `10` warmups, and `30` cold-L2 timed iterations; SOL is diagnostic only.
- Result: `0.0501915 ms`, corresponding to `55.4912%` minimum-byte roofline SOL, with `24.24%` jitter.
- Assessment: L2 256-byte promotion is `1.16%` slower than Iteration 113's default-cache direct loads (`49.6160 -> 50.1915 us`). The original warp-coalesced streams already use cache sectors efficiently; promotion adds no useful bandwidth. Reject the hints and retain ordinary compiler-generated LDG for the functional path.

## Iteration 116 — streaming/evict-first packed loads

- Diagnostic change: kept Iteration 115's exact original layout and load-only address stream but changed the inline loads from L2 256-byte promotion to PTX `ld.global.cs`, marking packed vectors and scales as streaming/evict-first. No addresses, bytes, grid, pass duplication, or launch resources change. The body remains intentionally non-functional.
- Validation/resources: SASS confirms `LDG.E.EF.128` for packed vectors and `LDG.E.EF` for scales. Numerical output fails as expected; resource shape remains spill-free with no QGMMA or shared-memory work.
- Protocol: focused `128x4096x7168/E8`, the same `8` slots, `10` warmups, and `30` cold-L2 timed iterations; SOL is diagnostic only.
- Result: `0.0462080 ms`, corresponding to `60.2749%` minimum-byte roofline SOL, with `34.03%` jitter.
- Assessment: streaming loads improve `6.87%` over Iteration 113's default-cache lower bound (`49.6160 -> 46.2080 us`) and `7.94%` over the L2-promotion variant. The one-use 132 MB payload benefits from evict-first policy, although the load-only stream still exceeds the `39.788 us` target. Retain the cache operator as a functional candidate and measure it inside Iteration 107's full decoder/WGMMA path before deciding.

## Iteration 117 — use streaming loads in the functional kernel

- Change: restored Iteration 107's complete functional kernel and replaced only the exact-byte path's two packed uint4 loads and scale-word load with inline PTX `ld.global.cs`. The bit-woven layout, outer N8/N16/N24/N32 specialization, activation TMA, shared FP8 staging, WGMMA groups, grid, and launch resources are otherwise byte-for-byte unchanged.
- Correctness: K128 and K256 routing/tail probes both pass with zero violations and reproduce the retained numerical metrics exactly.
- Resources/SASS: K256 remains `REG:66`, `LOCAL:0`, and `STACK:0`, preserving seven CTAs. SASS confirms `LDG.E.EF.128` and `LDG.E.EF` throughout all four exact width bodies; no instruction or SMEM topology changes occur outside the load cache operator.
- Protocol: focused official `128x4096x7168/E8`, standard Cupra timing (`8` input slots, `10` warmups, `30` timed iterations, cold-L2 rotation) on an idle H20-3e.
- Result: `0.0777120 ms`, `96.7186` useful TFLOP/s, `32.6752%` compute SOL, `35.8398%` roofline SOL, `41.77%` jitter.
- Assessment: the functional kernel improves only `0.10%` versus Iteration 107 (`77.7920 -> 77.7120 us`), despite the `6.87%` load-only gain. Decode, shared stores, synchronization, and WGMMA overlap most of the memory-policy benefit. Keep streaming loads as the narrowly measured new best, but treat the difference as near noise and continue structural work; the gap to `39.788 us` remains `1.95x`.

## Iteration 118 — schedule pass-zero tiles contiguously

- Change: retained Iteration 117's full functional kernel and changed only the two-pass block mapping. Instead of alternating pass 0 and pass 1 for every N64 tile, blocks `0:512` now cover all pass-zero tiles contiguously and blocks `512:1024` cover pass one. This lets the complete base workload occupy the first resident wave without interleaved early-return CTAs; packed bytes, cache operators, decode, WGMMA, grid size, and launch resources are unchanged.
- Correctness: K128 and K256 probes, including the 52-token second-pass case, pass with zero violations and unchanged numerical metrics.
- Resources: kernel instructions, `REG:66`, `LOCAL:0`, `STACK:0`, 29,704-byte dynamic SMEM, and seven-CTA residency are unchanged apart from block-index arithmetic.
- Protocol: focused official `128x4096x7168/E8`, standard Cupra timing (`8` input slots, `10` warmups, `30` timed iterations, cold-L2 rotation) on an idle H20-3e.
- Result: `0.0747530 ms`, `100.5470` useful TFLOP/s, `33.9686%` compute SOL, `37.2585%` roofline SOL, `54.95%` jitter.
- Assessment: pass-major scheduling improves `3.81%` versus Iteration 117 (`77.7120 -> 74.7530 us`) and `3.91%` versus Iteration 107. Keeping the first wave free of inactive pass-one blocks matters more than delaying the rare long-expert tail. Retain pass-major mapping as the new best; the kernel remains `1.88x` above the `39.788 us` target.

## Iteration 119 — remove the second pass grid

- Change: retained Iteration 118's streaming loads and exact width specialization but changed the pass split from two to one. The grid shrinks from 1,024 to 512 blocks; every `(expert,N64)` CTA serially processes all 32-token passes. This removes the pass-one tail entirely at the cost of making the three official slots with a >32-token expert rescan weights inside a first-wave CTA.
- Correctness: K128 and K256 probes, including the 52-token serial second pass, pass with zero violations and unchanged metrics.
- Resources: kernel instructions and per-CTA resources remain `REG:66`, `LOCAL:0`, `STACK:0`, 29,704-byte dynamic SMEM, and seven-CTA residency; only grid size and pass stride change.
- Protocol: focused official `128x4096x7168/E8`, standard Cupra timing (`8` input slots, `10` warmups, `30` timed iterations, cold-L2 rotation) on an idle H20-3e.
- Result: `0.0760640 ms`, `98.8141` useful TFLOP/s, `33.3831%` compute SOL, `36.6163%` roofline SOL, `51.11%` jitter.
- Assessment: removing the tail regresses `1.75%` versus Iteration 118 (`74.7530 -> 76.0640 us`). A rare long expert on the critical first wave is more expensive than launching the pass-major tail after all base tiles finish. Reject the single-pass grid and retain two pass-major slots.

## Iteration 120 — compact the long-expert tail grid

- Hypothesis: the second pass needs work only for experts with more than 32 tokens. Since the small-M path has at most 128 tokens, at most three experts can be long; ranking those experts can replace the full 512-CTA tail grid.
- Change: kept the 512 pass-zero CTAs contiguous, then launched three ranked long-expert slots per N tile (192 tail CTAs, 704 total). One lane per warp scans the eight offsets and broadcasts the selected expert. True packed FP4 storage and on-chip-only decode are unchanged.
- Correctness: K128 and K256 probes both pass with zero violations and unchanged numerical metrics.
- Protocol: focused official `128x4096x7168/E8`, standard Cupra timing (`8` input slots, `10` warmups, `30` timed iterations, cold-L2 rotation) on an idle H20-3e.
- Result: `0.0744960 ms`, `100.8939` useful TFLOP/s, `34.0858%` compute SOL, `37.3870%` roofline SOL, `54.74%` jitter.
- Assessment: the compact tail is `0.34%` faster than Iteration 118 (`74.7530 -> 74.4960 us`), confirming that inactive tail CTAs have a measurable but very small cost. Retain it as the new best; this scheduling gain is far too small to close the `39.788 us` target.

## Iteration 121 — issue both K128 packed halves before decode

- Profiling basis: Iteration 120 NCU reports nearly perfect 31.86/32-byte global-load sectors but long-scoreboard as the largest scheduler stall (`3.28` stalled warps per issued instruction). The K256 kernel had eight registers of headroom before losing seven-CTA residency.
- Change: moved only the second K128 half's two existing `uint4` packed loads ahead of first-half decode. Scale loads remain demand-driven, no byte is reread or prefetched twice, and packed FP4 storage is unchanged.
- Correctness/resources: K128 and K256 probes pass with zero violations. K256 reports `REG:72`, `LOCAL:0`, `STACK:0`, preserving seven CTAs/SM; K128 remains at 55 registers.
- Protocol: focused official `128x4096x7168/E8`, standard Cupra timing (`8` input slots, `10` warmups, `30` timed iterations, cold-L2 rotation) on an idle H20-3e.
- Result: `0.0759680 ms`, `98.9389` useful TFLOP/s, `33.4253%` compute SOL, `36.6626%` roofline SOL, `49.86%` jitter.
- Assessment: dual issuing regresses `1.98%` versus Iteration 120 (`74.4960 -> 75.9680 us`). The extra eight-register live range and altered instruction schedule cost more than the added within-warp memory-level parallelism, even without reducing occupancy. Reject and restore Iteration 120.

## Iteration 122 — make each packed N64 tile contiguous

- Hypothesis: the exact payload was grouped by vector plane across all N rows, forcing every CTA to carry full-`N` strides through the K loop. Grouping the same bytes into contiguous `[expert,K128,N64,4608B]` tiles should simplify address generation, improve CTA locality, and create a one-request TMA-compatible layout without expanding storage.
- Change: permuted the model-load-time bit-woven payload into N64-tile-major order. Each tile still contains exactly 4,096 packed-FP4 bytes and 512 original FP8 scale bytes; the kernel retains direct streaming LDG and on-chip-only FP8 decode.
- Correctness/resources: K128 and K256 probes pass with zero violations and unchanged numerical metrics. K256 reports `REG:64`, `LOCAL:0`, `STACK:0`; K128 reports 56 registers. The prepared tensor keeps the same `[E,K/128,72,N]` shape and byte count.
- Protocol: focused official `128x4096x7168/E8`, standard Cupra timing (`8` input slots, `10` warmups, `30` timed iterations, cold-L2 rotation) on an idle H20-3e.
- Result: `0.0731995 ms`, `102.6809` useful TFLOP/s, `34.6895%` compute SOL, `38.0492%` roofline SOL, `60.23%` jitter.
- Assessment: tile-major locality improves `1.74%` versus Iteration 120 (`74.4960 -> 73.1995 us`) and becomes the new best. Retain the lossless layout; its contiguous 4,608-byte unit now enables a single-request raw-weight TMA experiment while preserving seven-CTA capacity.

## Iteration 123 — single-request raw-weight bulk TMA

- Hypothesis: Iteration 122's contiguous 4,608-byte N64/K128 tile can replace 128 threads' direct packed LDG stream with one bulk TMA request. Reusing the residency pad as a raw buffer keeps seven CTAs, and issuing the next K256 first half before current WGMMA should hide half the transfers.
- Change: added one 4,608-byte raw shared buffer and a second transaction barrier. K256 exact tiles arrive through one bulk TMA per K128; threads decode from LDS. A midpoint CTA barrier protects the single buffer before its second-half refill. K128 retains direct streaming LDG.
- Correctness/resources: K128 and K256 probes pass with zero violations. K256 reports `REG:56`, `LOCAL:0`, `STACK:0`, and 30,224 bytes dynamic SMEM, preserving seven CTAs. SASS contains the expected raw-weight UTMALDG/LDS path and no direct packed LDG.
- Protocol: focused official `128x4096x7168/E8`, standard Cupra timing (`8` input slots, `10` warmups, `30` timed iterations, cold-L2 rotation) on an idle H20-3e.
- Result: `0.0764795 ms`, `98.2772` useful TFLOP/s, `33.2018%` compute SOL, `36.4174%` roofline SOL, `66.45%` jitter.
- Assessment: bulk TMA regresses `4.48%` versus Iteration 122 (`73.1995 -> 76.4795 us`). Removing long-scoreboard LDG is outweighed by raw LDS, transaction-barrier waits, and the added midpoint CTA barrier, matching the earlier load-only TMA diagnostic. Reject and restore Iteration 122's direct streaming loads.

## Iteration 124 — bypass L1 allocation for packed weights

- Profiling basis: Iteration 122 still has only 1.71% L1 hit rate while one-use packed weights dominate DRAM traffic. The previous `ld.global.cs` maps to evict-first but still allocates in L1.
- Change: retained Iteration 122's tile-major layout, addresses, bytes, decode, synchronization, and grid, but changed exact packed/scales loads to `ld.global.L1::no_allocate`. No data is prefetched or duplicated.
- Correctness/resources: K128 and K256 probes pass with zero violations. K256 remains `REG:64`, `LOCAL:0`, `STACK:0`; SASS changes the hot loads from `LDG.E.EF.128/LDG.E.EF` to `LDG.E.NA.128/LDG.E.NA`.
- Protocol: focused official `128x4096x7168/E8`, standard Cupra timing (`8` input slots, `10` warmups, `30` timed iterations, cold-L2 rotation) on an idle H20-3e.
- Result: `0.0722880 ms`, `103.9757` useful TFLOP/s, `35.1269%` compute SOL, `38.5290%` roofline SOL, `58.44%` jitter.
- Assessment: no-allocate improves `1.25%` versus Iteration 122 (`73.1995 -> 72.2880 us`) and is the new best. Retain it; the result confirms that avoiding L1 fill is better than merely marking the one-use stream evict-first, though the target still requires a much larger structural gain.

## Iteration 125 — use the non-coherent read-only path

- Hypothesis: Hopper's non-coherent global-read path can offer more cache bandwidth at higher latency. Seven resident CTAs may hide that latency, while retaining Iteration 124's no-allocate policy prevents one-use weight fills.
- Change: changed only the exact packed/scales PTX loads from coherent `ld.global.L1::no_allocate` to `ld.global.nc.L1::no_allocate`; addresses, bytes, layout, decode, and scheduling are identical.
- Correctness/resources: K128 and K256 probes pass with zero violations. K256 remains spill-free at 64 registers. SASS changes only to `LDG.E.NA.128.CONSTANT` and `LDG.E.NA.CONSTANT`; cubin size is unchanged.
- Protocol: focused official `128x4096x7168/E8`, standard Cupra timing (`8` input slots, `10` warmups, `30` timed iterations, cold-L2 rotation) on an idle H20-3e.
- Result: `0.0721600 ms`, `104.1601` useful TFLOP/s, `35.1892%` compute SOL, `38.5973%` roofline SOL, `59.03%` jitter.
- Assessment: the read-only path is `0.18%` faster than Iteration 124 (`72.2880 -> 72.1600 us`). Retain it as the measured best, but treat the margin as near noise; cache-operator tuning is now exhausted and cannot close the remaining `1.81x` target gap.

## Iteration 126 — pair two independent N64 warpgroup tiles per CTA

- Hypothesis: two independent warpgroups in one 256-thread CTA can process adjacent N64 output tiles while sharing the activation TMA, LUT initialization, offsets, and CTA scheduling. Four resident CTAs would still expose eight weight-decode warpgroups per SM, versus seven in Iteration 125, while halving activation/L2 traffic and fixed per-CTA work.
- Change: expanded the small-M CTA from N64 to N128 without changing either WGMMA's N64 weight tile or the lossless prepared payload. Each warpgroup loads, decodes, stages, computes, and stores one adjacent N64 tile; both consume one shared N32 activation tile. The grid shrinks from 704 to 352 CTAs, and K256 dynamic shared memory grows from 29,704 to 41,992 bytes.
- Correctness/resources: K128 and K256 routing/tail probes pass with zero violations and unchanged numerical metrics. K256 reports `REG:63`, `LOCAL:0`, `STACK:0`; K128 reports 56 registers. Four K256 CTAs provide eight resident warpgroups per SM.
- Protocol: focused official `128x4096x7168/E8`, standard Cupra timing (`8` input slots, `10` warmups, `30` timed iterations, cold-L2 rotation) on an idle H20-3e.
- Result: `0.0872325 ms`, `86.1628` useful TFLOP/s, `29.1090%` compute SOL, `31.9283%` roofline SOL, `12.35%` jitter.
- Assessment: pairing tiles regresses `20.89%` versus Iteration 125 (`72.1600 -> 87.2325 us`). Nominal warpgroup occupancy does not replace independent CTA-level latency hiding: the two warpgroups share the same activation barrier and CTA synchronization, while the larger shared allocation limits the scheduler to four independent CTAs. Reject the N128 CTA and restore Iteration 125's N64/128-thread topology.

## Iteration 127 — overlap packed-FP4 cp.async with K128 WGMMA

- Profiling basis: Iteration 122/125 has nearly perfect global-load sector utilization (`31.9/32` bytes), but only `0.98` eligible warps per scheduler and long-scoreboard as the largest stall. The retained K256 launch also reserves an otherwise unused 4,096-byte residency pad.
- Change: restored Iteration 125's N64/128-thread topology and repurposed the existing pad as a 4 KiB raw packed-FP4 buffer. Each thread uses two `cp.async.cg` transfers for its own K128 packed vectors. Two existing 8 KiB decoded-weight slots alternate: K128-1 transfer/decode overlaps K128-0 WGMMA, and the next K256's K128-0 transfer overlaps the current K128-1 WGMMA. Scales remain direct no-allocate loads. Persistent bytes and 29,704-byte K256 shared allocation are unchanged.
- Correctness/resources: K128 and K256 probes pass with zero violations and unchanged numerical metrics. K256 reports `REG:71`, `LOCAL:0`, `STACK:0`, retaining seven CTAs/SM; SASS contains the expected `LDGSTS` path. K128 remains on the retained direct-load implementation at 56 registers.
- Protocol: focused official `128x4096x7168/E8`, standard Cupra timing (`8` input slots, `10` warmups, `30` timed iterations, cold-L2 rotation) on an idle H20-3e.
- Result: `0.0788815 ms`, `95.2846` useful TFLOP/s, `32.1907%` compute SOL, `35.3085%` roofline SOL, `65.51%` jitter.
- Assessment: the pipeline regresses `9.31%` versus Iteration 125 (`72.1600 -> 78.8815 us`). Hiding part of the packed load latency does not repay the added raw LDS stream, an extra CTA synchronization per K128, a second WGMMA commit group per K256, and the seven-register live-state increase. Reject cp.async staging and restore Iteration 125's direct non-coherent loads.

## Iteration 128 — multicast activation across a two-CTA cluster

- Profiling basis: each N64 CTA rereads the same N32 activation tile for all 64 output-N tiles of an expert. A two-CTA Hopper cluster can halve this duplicated TMA/L2 stream without combining weight-decode scheduling domains or changing per-CTA registers/shared memory.
- Change: restored Iteration 125's direct-load path, launched adjacent N64 blocks as two-CTA clusters, initialized barriers with a cluster prologue, and changed only activation TMA to `UTMALDG.2D.MULTICAST`. Packed-FP4 bytes, weight loads/decode, WGMMA, output stores, and the 29,704-byte K256 allocation are unchanged.
- Focused validation/resources: single K128 and K256 routing/tail probes both pass with zero violations and unchanged metrics. K256 remains `REG:64`, `LOCAL:0`, `STACK:0`; SASS confirms multicast TMA.
- Official result: the standard 8-slot cold-L2 run did not complete. GPU 1 remained at `100%` utilization with about `46.2 GiB` allocated for over 40 seconds and produced no timing output; the run was terminated. This is a failed iteration, not a performance result.
- Assessment: the two CTAs can drift between K stages because they independently decode different weights while sharing a single multicast activation buffer. A faster cluster leader can issue the next multicast before its peer has completed WGMMA on the current buffer, causing the observed multi-run deadlock even though a short probe happens to pass. Reject cluster multicast: adding a cluster synchronization every K256 would restore lifetime safety but would put an expensive inter-SM barrier on all 28 stages and is not a viable latency path.

## Iteration 129 — JIT-specialize small-M dimensions

- Hypothesis: the JIT already builds one kernel per problem shape, but the small-M device function still treats N, K, and expert count as runtime values. Promoting them to template constants should fold hot-loop divisions, strides, bounds, and 64-bit address arithmetic without changing any data or launch resources.
- Change: restored Iteration 125's non-clustered direct-load execution and instantiated the cooperative kernel with compile-time `N`, `K`, and `E`. The ABI arguments remain for compatibility but are unused inside this specialization. Packed-FP4 storage, preparation, cache operators, decode, WGMMA, grid, and dynamic shared memory are unchanged.
- Correctness/resources: K128 and K256 probes pass with zero violations and unchanged metrics. On the probe specialization, K256 falls from `REG:64` to `REG:46`, static IMAD falls `295 -> 226`, and cubin size falls `39,368 -> 38,344` bytes; `LOCAL/STACK` remain zero. The 29,704-byte allocation still deliberately holds residency at seven CTAs.
- Protocol: focused official `128x4096x7168/E8`, standard Cupra timing (`8` input slots, `10` warmups, `30` timed iterations, cold-L2 rotation) on an idle H20-3e.
- Result: `0.0724320 ms`, `103.7690` useful TFLOP/s, `35.0571%` compute SOL, `38.4524%` roofline SOL, `56.17%` jitter.
- Assessment: despite the large compile-time simplification, latency is `0.38%` slower than Iteration 125 (`72.1600 -> 72.4320 us`) and statistically tied under the cold-L2 variance. Address/control instructions are not on the dominant critical path once the N64-tiled layout is in place. Reject the extra shape-template surface and retain Iteration 125 as the measured best.
