# SM120 MegaMoE — M64xN32 two-stage residency schedule

A native `sm_120a` CUDA export of an optimized single-entry MegaMoE kernel for
the DeepSeek V4 Flash routed-expert geometry. One persistent cooperative kernel
covers the whole distributed path — NCCL GIN dispatch, destination task
construction, W1, fused SwiGLU and FP8 requantization, W2, BF16 return,
weighted combine and acknowledgement — over a two-slot `[0, 1, 0]` result
lifecycle. Both GEMM phases run one runtime-scheduled M64xN32xK128 tile family
with four contiguous M16xN32 math-warp owners, four loader-warp roles and one
elected TMA issuer.

The kernel is a CAKE code-generator export. It has no runtime dependency on
CAKE or Python; the translation unit here only needs `nccl_device.h` and the
CUDA BF16/FP8 headers.

## Geometry as exported

| Property | Value |
| --- | --- |
| hidden / intermediate / output | 4096 / 2048 / 4096 |
| experts | 256 global = 8 ranks x 32 local, top-k 6 |
| precision | MXFP8 E4M3 activations, MXFP4 E2M1 weights, UE8M0 K32 scales, W4A8 GEMM, BF16 output |
| rank capacity | 8 physical ranks, 2 result/dispatch ring slots |
| token capacity | 8192 rows per rank |
| task granularity | 128 rows per task |
| grid | one service/coordinator CTA plus worker CTAs, 384 threads each |
| shared memory | 101376 B per CTA |
| shared-expert path | none |

The intermediate size is 2048, not 4096: the released
`DeepSeek-V4-Flash-0731` `config.json` reports `moe_intermediate_size = 2048`,
and the shipped safetensors expert shapes agree. Latencies below are therefore
not comparable to a 4096-intermediate configuration.

The worker count is resolved at launch rather than hard-coded. Every rank picks
the same resident grid from the minimum cooperative occupancy and the minimum
exact tile work across ranks, so no worker is left unowned in either GEMM
phase. On a 110-SM part this resolves to 110 CTAs.

## Kernel resources

`nvcc -cubin --generate-code=arch=compute_120a,code=sm_120a -std=c++17 -O3
--resource-usage`, CUDA 13.0:

| Metric | Value |
| --- | --- |
| registers | 166 |
| spill stores / spill loads | 0 / 0 |
| stack frame | 8 B |
| barriers | 16 |
| shared memory | 101376 B |
| cubin size | 366328 B |

## Measured performance

Measurement environment: 8 GPUs of compute capability 12.0 (`sm_120a`, 110 SMs,
73 GB each), no NVLink — inter-GPU transport is NCCL GIN over RDMA NICs
(`NCCL_GIN_TYPE=3`, GDAKI, NCCL 2.30.7 pinned); driver 580.95.05, CUDA 13.0.
All eight ranks participate in every measurement.

Timing protocol: CUPTI concurrent-kernel activity envelope of the single
full-chain entry, cold L2, one epoch per sample, reduced with a **max across
ranks** before any statistic. Each arm runs for 1000 ms; the reported value is
the slowest-rank median. Arms alternate legs (A/B/B/A) inside one process so
both arms see the same clocks, the same fabric and the same allocation.

### End-to-end epoch latency

Two independent sessions, 3 pairs per shape per session, balanced top-6 routing:

| active rows / rank | session 1 (ms) | session 2 (ms) |
| --- | --- | --- |
| 512 | 0.872 | 0.873 |
| 2048 | 2.732 | 2.726 |
| 4096 | 5.221 | 5.221 |
| 8192 | 10.362 | 10.390 |

### Against the starting point of the optimization campaign

Same host, same protocol, cold-L2 CUPTI medians of 3 runs per shape:

| active rows / rank | starting kernel (ms) | this kernel (ms) | speedup |
| --- | --- | --- | --- |
| 512 | 2.086 | 0.873 | 2.39x |
| 2048 | 4.414 | 2.729 | 1.62x |
| 4096 | 8.104 | 5.215 | 1.55x |
| 8192 | 15.671 | 10.366 | 1.51x |

Chaining the thirteen paired A/B promotions that produced this kernel gives
0.42 / 0.60 / 0.63 / 0.66 of the starting latency, which agrees with the
absolute table above.

### Largest single step: the lossless BF16 return codec

Paired A/B of the return-wire codec against the immediately preceding kernel,
two sessions x 3 pairs per shape, 24/24 pairs same-signed:

| active rows / rank | before (ms) | after (ms) | ratio s1 | ratio s2 |
| --- | --- | --- | --- | --- |
| 512 | 0.903 | 0.869 | 0.962 | 0.963 |
| 2048 | 3.177 | 2.756 | 0.867 | 0.870 |
| 4096 | 6.166 | 5.275 | 0.855 | 0.854 |
| 8192 | 12.234 | 10.471 | 0.856 | 0.856 |

### Wire budget

The return path is bandwidth-bound at every shape above 512 rows, so the codec
is the dominant lever. A returned BF16 row is encoded as 64 blocks of 64
values; each block carries a 2-byte header (block maximum exponent) plus 64
12-bit codes, and a block whose internal exponent span exceeds 15 falls back to
a raw 128-byte copy in a bounded per-row overflow area.

| Quantity | Raw BF16 | Codec v2 |
| --- | --- | --- |
| returned row | 8192 B | 6272 B (-23.4%) |
| result slot pitch | 8196 B | 8320 B |
| per-rank wire egress at R512 / R2048 / R4096 / R8192 | 32.3 / 129.3 / 258.6 / 517.1 MB | 26.9 / 107.5 / 215 / 430 MB |

Encoding runs in registers in the W2 epilogue and decoding runs inside the
incremental combine, so neither adds a serial pass. Against a 46 GB/s per-rank
egress floor computed on the **post-codec** wire bytes, the kernel attains
68 / 87 / 90 / 91 % at R512 / R2048 / R4096 / R8192. R512 is compute-window
bound rather than wire bound (its drain tail is about 60 us).

The overflow area is capped at 16 blocks per row (25% of the row, against a
measured 0.29% raw-block rate on real weights — an 86x margin). This is a
capacity bound, not a tolerance: output stays bit-exact, and a row that cannot
be represented exactly raises a loud `protocol_error` and fails the epoch
instead of silently producing a wrong result. The bound also keeps the result
window at approximately the raw size (13.1 GB rather than 22.9 GB for an
unbounded worst-case capacity).

## Correctness

Output is bit-exact BF16 against the reference; zero mismatches and zero
protocol errors across the whole ladder:

- a short watchdogged smoke run,
- four synthetic shapes (512 / 2048 / 4096 / 8192 rows) x 20 epochs,
- real MXFP4 weights at 512 and 2048 rows, real FP8 weights at 2048 rows,
  taken from layer 20 of the released DeepSeek V4 Flash checkpoint together
  with its real routing.

Every promotion in this kernel's history had to pass the same ladder before it
was allowed to be benchmarked, and the codec's raw-overflow path is covered by
the real-weight cases.

## What the schedule changed

Grouped by mechanism, in the order they were adopted:

1. **Warp-role split in the service CTA.** Pull drain and scatter move to
   loader warps; math and TMA warps enter the tile stream directly after task
   build. Dispatch becomes a push: full-chunk strong puts are issued from the
   pack loop and a per-source watcher warp releases chunk counters.
2. **Parallel front end.** Task build runs on one warp with one lane per local
   expert and warp-scan prefixes; the per-slot route prologue is parallel; tail
   audit and service loops are runtime-bounded; header and dispatch-chunk puts
   are issued by one CTA0 warp per peer in parallel.
3. **Loader-warp L2 prefetch** of the A and B tensors two K blocks ahead in
   both GEMM phases.
4. **Barrier-free epoch start.** Cumulative expected signal bases replace the
   per-epoch snapshot; the GIN world barrier runs only on the first launch.
   Host-side completion is observed by a device synchronize instead of a
   sleep-poll wait, with a watchdog retained as the deadlock guard.
5. **Combine memory-level parallelism.** Unconditional slot loads with guarded
   FP32 accumulation, two 256-element steps per iteration, and a dynamic
   token-claim so combine work follows arrival rather than a fixed partition.
6. **Return-path issue.** Per-peer in-order return puts with private issuer
   cursors and source ownership by warp; one issuer warp per peer; a 32-route
   return quantum below 768 routes per pair; per-chunk combine gating driven by
   the owner's own strong-put progress, so a chunk is combined as soon as that
   chunk lands. Acknowledgement puts are likewise issued one warp per peer.
7. **Unified-stream interleave.** The W1 to W2 interleave distance drops from 8
   to 4, which halves the post-compute return drain.
8. **Lossless BF16 return codec** as described above.

Mechanisms that were tried and rejected on measured evidence are not in this
file: uniform enlargement of the return quantum, W2 admission ordered by W1
completion, dispatch-side activation entropy coding, GIN dual-connection
variants, and m-tile skipping for padded rows.

## Provenance

| Item | Value |
| --- | --- |
| `cake_sm120_megamoe_m64n32_stage2_residency.cu` sha256 | `0711d40363ca0c8cd36adfe2fa2bcd17fa44799a524974e550d7f8fcd96cd260` |
| size | 198787 B, 3615 lines |
| entry symbol | `kernel_deepgemm_sm120_megamoe_m64n32_stage2_residency_dispatch` |
| launch bounds | 384 threads |

The `.cu` is the code generator's output byte for byte, so its hash can be
reproduced from the generator and compared against the resource-usage receipt
above. No hand edits were applied after generation.

## Not included here

This directory carries the device translation unit and this report only. It
does not yet ship a host driver, a `build.sh` or a runner, so it does not build
through the harness in `dsv4_flash/`. Wiring it to that host requires an
adapter for the differences this schedule introduces: the encoded result-slot
layout and pitch, the per-(source, chunk) dispatch and return signal
allocation, and the launch-time grid resolution described above. The numbers in
this report were produced by the CAKE harness the kernel was developed in, not
by `dsv4_flash/run_perf_abba.py`; they use the same max-across-ranks CUPTI
activity-envelope definition, and reproducing them under the host in
`dsv4_flash/` is the natural follow-up.
