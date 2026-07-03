# SM90 NVFP4 Pro Dispatch-Assist Dequant Candidate

## Goal

Use the otherwise idle second dispatch warp in the retained three-stage
braided fused kernel to reduce math-warp FP4-to-FP8 work without involving the
two TMA loader warps. The production fused/split kernels, standard NVFP4
format, 80-byte braided row, and request-time ABI remain unchanged.

## Alternatives

1. Split the current stage between one helper warp and both math warpgroups.
   This leaves the helper decoding two rows per lane and puts it directly on
   the current-stage critical path, so it is rejected analytically.
2. Have the idle dispatch warp decode two complete 32-row tails for the next
   stage while the math warpgroups decode and execute WGMMA on the current
   stage. This is selected because three packed stages are already prefetched
   and the loader warps remain free.
3. Use the second dispatch warp for direct global token copies. This changes
   the NVLink transfer path and release ordering, so it is deferred until the
   lower-risk dequant helper is measured.

## Pipeline

The first K block of every scheduled L1/L2 task is a prologue: both math
warpgroups decode all 256 rows. For each later K block, the dispatch helper
decodes rows 96--127 and 224--255 one stage ahead. Each math warpgroup skips
its corresponding final 32-row warp and decodes the other 96 rows.

One transaction barrier per stage publishes helper completion. The helper
arrives without stores for every prologue slot so barrier parity remains
aligned when that ring slot is reused. For non-prologue stages it waits on the
normal full barrier, writes both disjoint 32-row tails, executes a warp sync,
and arrives once. Both math warpgroups wait on that same dequant barrier before
WGMMA. Existing empty-barrier ownership remains math-only.

## Isolation

The existing three-stage template receives one default-false compile-time
boolean. A separately named JIT runtime, C++ API, and Python adapter instantiate
it as true. The accepted three-stage API still omits the new argument and
therefore generates its byte-identical default-false kernel.

## Gates

1. Fresh-JIT exact-NVFP4 correctness at M=8/64 with
   `global_scale=none/expert`, followed by M=16/32/128 if it passes.
2. Both helper and math paths must remain spill-free; the kernel must stay at
   or below 168 registers and 232448 bytes dynamic shared memory.
3. Phase profiling must show lower `math_dequant` without increasing
   `math_dequant_wait` enough to erase it.
4. Seed-101 M=8/16/32/64 screen against the retained three-stage winner.
   Only a broad win advances to three-seed A/B/B/A; no exact-M gate is allowed.

No experiment is committed or pushed.

## Outcome

Rejected after correctness and endpoint screening. M=8/64 passed both global
scale modes with 0.9988 minimum cosine, 168 registers, a 56-byte stack, and no
spills. At M64 the helper regressed max-rank latency by 19.1%. Its barrier wait
was negligible, but the additional decoder warp raised `gemm_core` by 23.5%
through shared-memory and instruction-issue contention. All candidate wiring
and the compile-time branch were removed.
