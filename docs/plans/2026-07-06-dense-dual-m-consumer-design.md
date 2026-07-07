# SM90 NVFP4 Dense Dual-M Consumer Design

## Objective

Raise the raw useful throughput of `8192x4096x7168/E8` from the single-E4M3 baseline of 146.47 TFLOP/s (3.2842 ms, 49.48% of Cupra's 296 TFLOP/s peak) to at least 236.8 TFLOP/s (2.032 ms, 80%). The known single-E4M3 precision failure is reported but is not an optimization gate for this campaign. Cupra's checker itself remains unchanged.

Persistent weights remain genuine packed NVFP4. `prepare()` may only reorder the existing packed codes and block/global scales without increasing their dtype, element count, or total bytes.

## Recommended architecture

The current dense warp-specialized CTA computes one `M64 x N256` tile. Its producer reloads and decodes the same expert/N weight tile for every M64 tile. The new large-M specialization computes `M128 x N256` with three warpgroups:

- one producer warpgroup loads packed FP4 plus block scales and builds one transient E4M3 B stage;
- two consumer warpgroups each load a distinct M64 activation slice, issue WGMMA against the shared B stage, and store their own output rows;
- the producer and both consumers retain the existing two-stage ring, with each stage released only after both consumers finish;
- K=7168 small-M keeps the existing one-consumer specialization to preserve tile-level parallelism.

The change halves packed-weight load/decode work per useful output while leaving total useful WGMMA work and activation traffic unchanged.

## Resource and synchronization design

Each stage contains two M64 A tiles and one N256 B tile: `(2*64 + 256) * 128 = 48 KiB`; two stages plus the 1 KiB LUT and barriers fit comfortably below H20's 232,448-byte opt-in shared-memory limit.

The CTA has 384 threads. Each N256 consumer receives the register budget needed for its 128 FP32 accumulator values; the producer receives the remainder. The validated sparse dual-consumer 224/56 split is the starting point because `2*224 + 56` registers per lane-equivalent consumes 64.5K registers per CTA. Consumer-side activation loading keeps the producer critical path limited to B decode.

Full-stage readiness combines one producer arrival and one arrival from each consumer. Empty-stage release combines all eight consumer warps. Named warpgroup barriers use distinct IDs so concurrently loading warpgroups cannot satisfy one another's local synchronization.

## Alternatives

1. Further LUT/PRMT and packed-load instruction reduction is lower risk but is unlikely to provide the required 1.62x alone. Apply it after B reuse if the producer remains limiting.
2. CTA-cluster/DSM weight sharing avoids the single-CTA register budget but adds cluster scheduling and DSM latency. Reserve it for a measured dual-consumer resource wall.

## Validation

1. Restore and remeasure the single-E4M3 baseline with a fresh JIT cache.
2. Validate a small forced dual-consumer shape under compute-sanitizer.
3. Measure the primary shape with Cupra's normal cold-L2 profiler even if correctness reports PRECFAIL; log useful FLOPs only.
4. After each benchmark, append `ITERATIONS.md` and commit before any further probe.
5. When the primary shape reaches 80%, run all six shapes and audit the persistent prepared payload bytes.
