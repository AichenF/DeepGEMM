# ITERATIONS — TP MXFP4 MegaMoE

Metric: `RUNTIME` = TP forward latency (per-rank compute + all-reduce), max across ranks, lower is better.
Correctness: cosine vs torch mxfp4-dequant golden (gate >= 0.99 in loop).

Config: M=8, E=128, topk=8, tp=8, H=6144, I=2048 (Is=256). E reduced 384->128 for GPU-contention memory headroom (etgong on GPUs 5-7); touched experts <= M*topk = 64.

## Summary
| iter | direction | RUNTIME (ms) | cosine | notes |
|------|-----------|--------------|--------|-------|
| baseline | vectorized-torch partial FFN (rank0 compute) | 20.52 | 0.99999 | dequant ALL 128 experts fp32 + fp32 matmul |
| 1 | dequant touched-only + bf16 tensor-core matmul | 9.54 | 0.99999 | 2.15x; only <=64 touched experts, bf16 GEMM |
| 2 | CUDA MXFP4->bf16 dequant kernel (load_inline) | 2.04 | 0.99999 | 4.7x; dequant was 94% (8.9ms), now fused 1-pass GPU |
| 3 | vectorized dequant (uint32 read / uint4 write, 8-nibble chunk/thread) | 1.52 | 0.99999 | 1.34x; dequant 1.42->~0.9ms (was uncoalesced 1B/thread) |
| 4 | FULLY FUSED kernel: dequant+FC1+SwiGLU+FC2 in-register, 1 block/pair | 1.12 | 1.00000 | 1.36x; no bf16 materialization; reads only packed weights; fp32 accum -> cosine 1.0. THE target kernel shape. |
| 5 | warp-per-output reduction (coalesced K) + 512 threads | 1.09 | 1.00000 | 1.02x only -> occupancy floor: just 64 blocks (M*topk pairs) on 78 SMs. Next axis = more blocks (split output tiles), not warp/thread tuning. |

## FINAL verdict (best = iter-5, HEAD)
Real config E=384 (kernel cost independent of E: only <=M*topk touched experts computed):
- M=8:  RUNTIME 1.09 ms  cosine 1.00000  (53.2x vs torch mxfp4-dequant ref 58ms)  FULL_FORWARD 2.86ms
- M=16: RUNTIME 2.13 ms  cosine 1.00000  (27.8x)
- M=32: RUNTIME 4.25 ms  cosine 1.00000  (14.4x)
RUNTIME ~linear in M (1 block per token-expert pair). Overall baseline->iter5: 20.5->1.09ms = 18.8x on the compute; correctness perfect (fp32 accum).
Next axis (untried): 2-kernel split (FC1-once + FC2 tiled) to raise block count above the 64-block floor for small M; payoff capped by GPU-0 contention (etgong), so measurement is noisy. Also: fold all-reduce into symm-buffer NVLink (currently dist.all_reduce) per the original spec.

## Container single-GPU compute bench (in mlb_refactor_bench_20260809; the per-rank kernel we optimize)
Prior single-fused kernel (iter5) was FLAT ~1.07ms for M=1..8 (raw kernel 1.06ms, python wrapper 0.03ms) = latency/occupancy starved (M=1 -> only 8 blocks on 78 SMs).
| iter | M=1 | M=8 | notes |
|------|-----|-----|-------|
| iter5 (1 block/pair) | 1.07 ms | 1.09 ms | flat = occupancy floor |
| 6 (2-kernel tiled nA=8/nB=16) | 0.154 ms | 0.95 ms | 7x at M=1; grid P*nA(FC1)+P*nB(FC2); cosine 1.0 |
| 7 (ako4x-swept nA=16/nB=48) | 0.140 ms | 0.93 ms | 8-GPU parallel (nA,nB,threads) sweep in container; M2=0.27 M4=0.48; runtime now linear in M (occupancy fixed), scalar-compute-bound |

ako4x sweep (M=1, parallel across free GPUs 0-4 in container, no recompile): (nA=16,nB=48)=0.139 (nA=32,nB=96)=0.140 (nA=16,nB=96)=0.141 ... (nA=8,nB=16)=0.150. Cluster ~0.14ms floor for M=1 with scalar 2-kernel. Next lever = tensor cores (MXFP4->fp8 WGMMA), a larger rewrite.

## GOAL 2: tp=4, M=8..128, SOL>=70% (in megamoe container: nvcr.io/nvidia/pytorch:25.02-py3, torch2.7/cu12.8, ncu OK)
tp=4 => Is=512 (kernel now dispatches Is in {256,512}). Scalar 2-kernel baseline (nA=16,nB=48), cosine 1.0:
  M=8 1.83ms | M=16 3.54 | M=32 6.97 | M=64 13.9 | M=128 27.66  (linear in M, ~27us/pair)
ncu SpeedOfLight (tp4 M32): fc1 Compute-SM 98.6% / Mem 16.5% / DRAM 3.4% (4.99ms); fc2 Compute-SM 97.7% / Mem 20% (2.53ms).
=> SCALAR-PIPE SATURATED (98%) but memory idle (16%). Arithmetic intensity ~11 FLOP/byte << H20 ridge ~666 => problem is MEMORY-BOUND; scalar dequant+FMA is the wall. To hit 70% MEMORY SOL must move compute to TENSOR CORES (MXFP4->fp8 WGMMA) so kernel becomes memory-bound (weight reads). Memory floor ~0.3-0.6ms for M=32 (touched experts x 4.5MB / 3TB/s) => ~15-25x headroom. Next: fp8 WGMMA FC1/FC2 with in-register MXFP4->fp8 dequant (fold E8M0), per-expert grouping (BLOCK_M pad; per-expert M tiny but memory-bound so tensor-core m-underutil is hidden).

## iter 9: branchless ARITHMETIC dequant (no __constant__ LUT) — the constant LUT kFP4V[nib&7] had DIVERGENT indices -> constant-cache serialized up to 8-way. Replaced with bit-construct float. cosine 1.0.
tp=4 (arith): M8 0.764 | M16 1.443 | M32 2.810 | M64 5.561 | M128 11.04 ms  (~2.4x vs iter8 scalar-LUT). Materialize+matmul variant was 4x SLOWER (30ms M32) -> fused is right.

## iter 9 SOL after arith: ncu tp4 M32 fc1 Compute-SM 88.2%% / Mem 42%% (was 98.6/16.5). Compute SOL >70%. float-xs-in-smem only +2%% (not adopted). Config sweep tp4 M32: nA16/nB48 optimal (2.81ms). Next lever for memory-SOL/abs-speed = group-by-expert (M128 ~2.7 tok/expert -> ~2.7x less redundant dequant).

## iter 10: shared-mem LUT dequant (replace ~6 arith ops with 1 LDS+sign; smem broadcast avoids constant-cache serialize). tp4: M8 0.50 | M16 0.94 | M32 1.82 | M64 3.59 | M128 7.12 ms (~1.5x vs arith). USER CORRECT: kernel is ALU-bound (ncu ALU pipe 89.6%%, DRAM only 431GB/s=~13%% peak) not mem-bound. SOL vs THEORETICAL memory time: tp4 M32 theoretical ~0.36ms (1.2GB/3.3TBs per-pair) -> current 1.82ms = SOL ~20%%. Target 70%% = ~0.5ms. Levers: cheaper dequant (ALU), vectorize weight loads (uint4), group-by-expert (kill per-pair redundant reads).

## iter 10 reprofile + roofline honesty: smem-LUT reprofile (tp4 M32): SpeedOfLight Memory 96.5%% but that is SHARED-mem LDS (the LUT); DRAM only 14%% (431 GB/s). Still pipe-bound not DRAM-bound. Theoretical mem ~0.28-0.36ms vs 1.82ms => SOL ~15-20%%. FUNDAMENTAL: MXFP4 dequant ~3 op/nibble min, reads 0.5 byte/nibble => ~6 op/byte ~ H20 ridge (~5) => pure 70%% memory-SOL is HARD for W4A16 dequant-GEMV (theoretical ceiling ~50-60%% SOL). uint4+scale-hoist tried -> REGRESSED (4.8ms, register spill from 32-nibble unroll+arrays), discarded. Next lever: group-by-expert (M128 ~2.7x, approach unique-read floor).

## fold experiment (DEAD END for scalar): fp8-fold dequant (SIMD ~0.5op/nibble, validated 1518GB/s materializing fp8) -> but scalar GEMV needs FLOAT, so fp8->float cvt per element costs more than saved: 2.47ms (SLOWER than smem-LUT 1.82) + had a correctness bug. CONCLUSION: fp8-fold ONLY pays with fp8 WGMMA (consumes fp8 directly, no cvt). On Hopper scalar path, bf16 smem-LUT (1.82ms) is near-optimal; it is compute/LSU-bound, ~20%% mem-SOL. To reach 70%% MEMORY SOL REQUIRES the full fp8-fold + fp8-WGMMA (grouped-by-expert) kernel = EP dev_m technique = large build. Best kept = iter10 smem-LUT.

## Log

### baseline — vectorized torch
- What: per-rank sharded FFN via mxfp4-dequant + einsum grouped matmul + SwiGLU + `dist.all_reduce`. MXFP4 dequant + compute in torch (to be CUDA-ized).
- Result: RUNTIME=48.93 ms, cosine=0.99999, REF=20.25 ms, SPEEDUP=0.41x. finite=True.
- Read: torch einsum gathers per-pair weights `gate_w[fe]` -> [M*topk, Is, H] materialization; re-dequants l1+l2 every call. Both are the obvious first targets. Real perf will come from a CUDA fused MXFP4 kernel; intermediate iters can cut torch waste first.

## 2026-09-02 V4-Flash TP baseline reset

The optimization target was reset to the real pure-TP serving path: H=4096,
I=2048, E=256, top-k=6, TP4 primary (TP8 runnable), with precomputed real
`topk_ids`/`topk_weights`, no EP dispatch/combine, and one SGLang
`CustomAllReduceV2` after the local expert reduction.  Formal timing is the
maximum rank latency of a full CUDA-Graph replay.

### Baseline harness smoke — Humming OCP MXFP4 + CustomAllReduceV2

- What: added `bench/v4_flash_tp_humming_graph.py`; graph contains SGLang route
  alignment, BF16-to-FP8 group-128 quant, indexed Humming MXFP4 W13, SwiGLU,
  second quant, indexed W2, local k6 weighted sum, and custom all-reduce v2.
- Shape: TP4 W13 `[256,1024,4096]`, W2 `[256,4096,512]`; H20 reports 78 SM.
  Humming transformed bytes/rank are 570,425,344 (W13) and 285,212,672 (W2).
- First diagnostic run: M8 0.1830 ms, but generic random E8M0 bit patterns made
  FP32 norms overflow (`cos=0`, `rel_l2=NaN`); rejected as a correctness result.
- Fix: keep packed FP4 codes random, initialize E8M0 scales to 1, and compute
  untimed diagnostics in FP64.  This changes data values only, not the kernel
  path or timed graph.
- Valid smoke (M8, balanced routes, only 2 replays): 0.156736 ms; custom-AR graph
  output vs NCCL sum has min-rank cosine 0.9999957033, max-rank rel-L2
  0.0029315142, and finite output on all ranks.  Marked smoke only; formal
  5-point/7-outer-run baseline still pending.

### Baseline formal attempt 1 — rejected pull-path reference

- TP4 balanced-route max-rank medians (7 outer samples x 100 graph replays):
  M8 0.092633, M16 0.156537, M32 0.294000, M64 0.389452,
  M128 0.418304 ms.
- Push-path checks (M8/16/32) passed with cosine >=0.9999955 and rel-L2
  <=0.00299.  Pull-path M64/128 showed cosine 1.0 and rel-L2 exactly 0.75.
- Root cause: `TWO_SHOT_PULL` may overwrite/reuse the custom-AR input.  The
  harness cloned that already-reduced buffer and then reduced it once more via
  NCCL, making the reference exactly 4x the graph result.  This is a reference
  bug, not evidence of a kernel error.
- Decision: do not accept this attempt as the formal baseline.  Correctness now
  recomputes the local pipeline into independent untimed buffers before NCCL;
  rerun all five points.

### Accepted TP4 Humming baseline — balanced routes, full CUDA Graph

- Protocol: 9 outer samples, 200 graph replays/sample, 50 warm-up replays;
  reported latency is the maximum rank average for every outer sample.
- Max-rank latency min / median / max (ms):
  - M8: 0.092545 / 0.092644 / 0.093283
  - M16: 0.156270 / 0.156437 / 0.156623
  - M32: 0.289523 / 0.294363 / 0.351842
  - M64: 0.397270 / 0.412274 / 0.435562
  - M128: 0.411079 / 0.430198 / 0.459386
- Geometric mean of the five medians: 0.237563 ms.
- Correctness: every point finite; min-rank cosine >=0.9999886295 and
  max-rank rel-L2 <=0.0047687842 against NCCL sum of an independently
  recomputed local output.  Both one-shot-push (M<=32) and graph
  two-shot-pull (M>=64) paths pass.
- Noise note: M32's last three samples and the larger-M samples show transient
  slowdown.  Keep min/median/max and require repeatability when judging an
  optimization; do not compare a candidate's minimum with this median.
- Evidence log:
  `bench/results/tp4_humming_graph_balanced_formal_v2_20260902.log`.

### Accepted TP4 Humming route-skew control

- Route: every token selects the same six distinct experts (legal maximal
  skew); router remains precomputed and untimed.
- Max-rank latency min / median / max (ms), same 9x200 protocol:
  - M8: 0.031925 / 0.031967 / 0.032896
  - M16: 0.039892 / 0.039931 / 0.039978
  - M32: 0.059635 / 0.059686 / 0.060581
  - M64: 0.094630 / 0.094663 / 0.094918
  - M128: 0.163625 / 0.164054 / 0.164349
- All five NCCL checks pass; min cosine 0.9999352, max rel-L2 0.0113814.
- Finding: route distribution is first-order.  At M32, changing only active
  experts from 192 to 6 reduces the same graph from 0.294363 to 0.059686 ms
  (4.93x).  Raw `G` cannot stand in for a real routed-MoE benchmark.
- Policy: balanced routes remain the primary optimization score, while this
  skew case is a mandatory counterexample check.
- Evidence log: `bench/results/tp4_humming_graph_skew_formal_20260902.log`.

### Route-aware RS-WGMMA bring-up — correctness first

- Starting point: preserved the braided MXFP4-to-FP8 register dequant and
  swap-AB `m64n8k32` RS-WGMMA core from `step_e_lutg.py`/`step_e_fc2.py`.
- Changed: raw `G` and shared `X[8,K]` were replaced by SGLang
  `sorted_ids`/`expert_ids`/`num_tokens_padded`; each tile gathers its real
  token rows, applies per-token FP8 group-128 scales per K tile, writes W13
  split-K partials, reduces into BF16 SwiGLU, and fuses W2 route weighting into
  FP32 local-token scatter.  Shapes specialize both TP4 Is=512 and TP8 Is=256.
- TP4 M8 balanced (48 active experts, all 48 routes, padded to 384 rows):
  W13 cosine 0.999999998 / rel-L2 0.000076187; activation cosine
  0.999999759; W2+local-reduce cosine 0.999999993 / rel-L2 0.000126339.
- TP4 M8 maximal skew (6 active experts, 8 real tokens/expert): W13 cosine
  0.999999997, activation 0.999999649, W2 0.999999992.
- TP8-template M8 balanced (Is=256): W13 cosine 0.999999997, activation
  0.999999745, W2 0.999999710 / rel-L2 0.000762151.
- All checks cover every output block and every split-K partial against a
  torch dequant/matmul reference; all outputs finite.  No performance claim
  yet.

### Route-aware WGMMA full-graph smoke (split-K=4)

- Added the end-to-end graph path with the same timed route alignment,
  BF16-to-FP8 group-128 quantization, and SGLang `CustomAllReduceV2` as the
  Humming baseline.  W13 split partial reduction + SwiGLU and W2 weighted local
  scatter are custom kernels inherited from the reviewed WGMMA implementation.
- TP4 M8 balanced, cold smoke protocol (1 outer sample x only 2 replays):
  0.225584 ms, custom-AR vs independent local+NCCL cosine 0.9999956064,
  rel-L2 0.0029644006, finite.
- The accepted Humming M8 median is 0.092644 ms, so this preliminary point is
  2.44x slower.  Do not treat that ratio as final: Humming's analogous 2-replay
  smoke was 0.156736 ms (1.69x above its long-replay result).  Next run uses the
  exact baseline 9x200 protocol for all five M values.

### Route-aware WGMMA iteration 1 — formal TP4 graph baseline

- Protocol matches Humming exactly: balanced precomputed routes, 9 outer
  samples x 200 graph replays, max rank, route align + both group-128 FP8
  quantizations + local pipeline + default CustomAllReduceV2 all timed.
- Max-rank latency min / median / max (ms):
  - M8: 0.136533 / 0.136652 / 0.137908
  - M16: 0.237250 / 0.237389 / 0.237955
  - M32: 0.444652 / 0.464574 / 0.473932
  - M64: 0.599627 / 0.611745 / 0.625516
  - M128: 0.616757 / 0.629871 / 0.641963
- Humming median comparison (custom / Humming): M8 1.475x, M16 1.517x,
  M32 1.578x, M64 1.484x, M128 1.464x.  Five-point median geometric mean is
  0.357101 vs 0.237563 ms = 1.503x slower.  The target is not met.
- Correctness: all custom-AR checks pass; min cosine 0.99999555, max rel-L2
  0.00298319, all finite.
- Read: the real route-aware integration invalidates any claim based on the old
  raw-G bandwidth alone.  Profile stage costs before changing the kernel;
  likely candidates are W13 split-K partial traffic, W2 throughput, and excess
  fixed-grid CTAs from graph-safe route alignment.
- Evidence log:
  `bench/results/tp4_wgmma_graph_balanced_s4_formal_v1_20260902.log`.

### Benchmark protocol correction — explicit cold L2 is mandatory

- User directive (2026-09-02): every subsequent performance benchmark must
  use cold L2.  The earlier continuous-replay results above are retained only
  as warm/steady-state diagnostics and must not be used as the optimization
  score or final comparison.
- Measured H20 L2 size is 62,914,560 bytes (60 MiB).  Both the Humming and
  custom graph harnesses now allocate Triton's standard 256 MiB benchmark
  cache buffer and clear it on the benchmark stream before every individually
  timed graph replay.  The start event is recorded after the clear and the end
  event after the replay, so cache eviction is ordered but excluded from the
  reported latency.
- TP4 M8 balanced smoke, 1 outer x 10 cold replays:
  - Humming min / median / max: 0.095904 / 0.096752 / 0.248672 ms.
  - Custom split-K=4 min / median / max: 0.141408 / 0.142560 / 0.296864 ms.
  - Median custom / Humming: 1.473x (custom is 47.3% slower).
- Both graph outputs pass CustomAllReduceV2 versus an independent local
  recompute plus NCCL sum: minimum cosine is 0.999995606 and all values are
  finite.  Ten samples only validate the cold-L2 mechanism; they are not the
  formal performance result.
- Evidence logs:
  `bench/results/tp4_humming_graph_coldl2_smoke_m8.log` and
  `bench/results/tp4_wgmma_graph_coldl2_smoke_m8_s4.log`.

### Accepted TP4 cold-L2 baseline — Humming versus route-aware WGMMA

- Protocol: balanced precomputed routes, full CUDA Graph, 9 outer batches x
  200 individually timed replays per M, and a 256 MiB same-stream clear before
  every replay.  Latency is the maximum rank event time; medians below pool all
  1,800 cold samples per point.
- Humming min / median / max (ms):
  - M8: 0.095040 / 0.096864 / 0.296320
  - M16: 0.158400 / 0.160064 / 1.614496
  - M32: 0.290560 / 0.293984 / 0.349056
  - M64: 0.386656 / 0.406176 / 0.458016
  - M128: 0.392128 / 0.412752 / 0.493440
- Custom split-K=4 min / median / max (ms):
  - M8: 0.140352 / 0.142080 / 0.286400
  - M16: 0.241152 / 0.243136 / 0.289536
  - M32: 0.448192 / 0.450896 / 0.839104
  - M64: 0.596192 / 0.607264 / 0.635904
  - M128: 0.612864 / 0.627456 / 0.685664
- Median custom / Humming ratios: 1.467x, 1.519x, 1.534x, 1.495x,
  and 1.520x for M8 through M128.  Five-point geometric means are 0.358661
  versus 0.238033 ms, so custom is 1.507x slower.  The target is not met.
- Correctness: all ten graph outputs are finite and pass custom all-reduce
  against independent local recompute plus NCCL.  Minimum cosine is
  0.999987363 for Humming and 0.999995477 for custom; maximum rel-L2 is
  0.00502738 and 0.00300767 respectively.
- Rare global maxima show scheduler/system tails (notably Humming M16), while
  the nine per-batch medians are stable.  Optimization decisions use pooled
  medians plus the batch-median distribution, never minima.
- Evidence logs:
  `bench/results/tp4_humming_graph_coldl2_formal_20260902.log` and
  `bench/results/tp4_wgmma_graph_coldl2_s4_formal_20260902.log`.

### TP4 cold-L2 W13 split-K screen

- Screen protocol: balanced full graph, 3 outer batches x 100 individually
  cold replays per M, with the same 256 MiB clear and max-rank timing as the
  accepted baseline.  Correctness passes for every configuration and point.
- Median latency (ms), M8 / M16 / M32 / M64 / M128:
  - split-K=1: 0.143616 / 0.250672 / 0.454784 / 0.595040 / 0.610048;
    geometric mean 0.358761 ms.
  - split-K=2: 0.144224 / 0.245664 / 0.447424 / 0.588320 / 0.603328;
    geometric mean 0.354857 ms.
  - split-K=4 accepted formal reference: 0.142080 / 0.243136 / 0.450896 /
    0.607264 / 0.627456; geometric mean 0.358661 ms.
  - split-K=8: 0.146368 / 0.252864 / 0.467472 / 0.618048 / 0.638432;
    geometric mean 0.368846 ms.
- Finding: split-K=4 remains best at M8/M16, while split-K=2 wins at
  M32/M64/M128 by about 0.8%/3.1%/3.8% versus the split-K=4 formal medians.
  Split-K=8 is a regression everywhere.  A global split-K=2 default improves
  the five-point geometric mean only 1.1%, so split selection cannot close the
  roughly 50% Humming gap; profile stage costs before altering the core.
- Evidence logs:
  `bench/results/tp4_wgmma_graph_coldl2_s{1,2,8}_screen_20260902.log`.

### Cold-L2 M32 stage profile — core scheduling is the bottleneck

- Added profiler-only entry points which expose exactly one 256 MiB cache
  clear followed by one full graph or per-rank local pipeline.  Initialization,
  JIT, and warm-up are outside the CUDA profiler range.  Nsight Compute also
  uses `--cache-control all` for every metric replay.
- Nsight Systems full TP4 graph node medians (microseconds):
  - Custom split-K=2: W13 282.65, W2 142.00.
  - Humming: W13 180.83, W2 92.50.
  - Custom/Humming is 1.563x for W13 and 1.535x for W2.  Route alignment,
    both quantizers, activation/reduction, and casts are each only a few
    microseconds, so the roughly 50% end-to-end gap is in both GEMM cores.
- Nsight Compute detailed replay confirms the mechanism:
  - Custom W13/W2: 6,144 / 12,288 CTAs, 6.56 / 13.13 waves per SM,
    34 registers/thread, achieved occupancy 71.6% / 72.3%, and DRAM
    throughput 29.6% / 29.3%.
  - Humming W13/W2: 234 / 312 persistent CTAs, 0.75 / 1.00 waves per SM,
    124 / 101 registers/thread, achieved occupancy 16.6% / 20.8%, and DRAM
    throughput 45.7% / 44.8%.
  - NCU durations (which include metric-replay perturbation) are custom
    306.34 / 154.24 us versus Humming 195.74 / 100.93 us, preserving the
    same ratios as the low-overhead Systems trace.
- Diagnosis: custom launches one 64-output-channel x 8-route CTA per tile and
  repeats CTA setup, route/LUT loads, and activation-tile loads thousands of
  times.  Humming uses 128 output channels and persistent CTAs that loop over
  logical tiles.  High custom occupancy does not compensate for the fragmented
  schedule and excess L1/shared traffic.
- Next isolated change: increase the custom output-channel tile from 64 to 128
  so two WGMMA output groups reuse one activation tile and halve CTA count.
  Persistent scheduling remains the follow-on if the larger tile is not enough.
- Evidence reports:
  `bench/results/tp4_{wgmma_m32_s2,humming_m32}_coldl2.{nsys-rep,sqlite}`
  and `bench/results/tp4_{wgmma_m32_s2,humming_m32}_coldl2_ncu.ncu-rep`.

### WGMMA iteration 2 — 128-channel output tile regresses

- Change: added an isolated `V4_WOUT=128` variant.  One CTA loads a 128x128
  packed-weight stage and reuses the same 8x128 activation tile for two
  64x8 WGMMA output groups, halving the logical CTA grid.  The default remains
  64 so the accepted path is unchanged.
- Correctness passes before timing for TP4 balanced, TP4 maximal skew, and the
  TP8 Is=256 shape.  Final W2 cosine is 0.999999999 and rel-L2 is about
  5.7e-5 in all three tests; W13 split-K=2 full-output cosine rounds to 1.0.
- Cold-L2 screen protocol: TP4 balanced full graph, 3x100 samples per M,
  256 MiB clear before every replay.  `V4_W13_SPLIT_K=2 V4_WOUT=128`
  medians for M8/M16/M32/M64/M128 are 0.154176 / 0.259296 / 0.464032 /
  0.601872 / 0.616192 ms; geometric mean is 0.369416 ms.
- Versus the identical split-K=2 64-channel screen, the variant regresses by
  6.9% / 5.5% / 3.7% / 2.3% / 2.1%, or 4.1% geometric mean.  Reject it.
- Interpretation: each 64-row WGMMA group currently has its own commit/wait;
  serializing two groups lengthens each CTA enough to outweigh reduced setup
  and activation loads.  A future wider-tile attempt would need asynchronous
  grouping or deeper pipelining, not this serialized construction.
- Evidence logs:
  `bench/results/v4_flash_tp_wgmma_wo128_s2_correctness*_20260902.log` and
  `bench/results/tp4_wgmma_graph_coldl2_s2_wo128_screen_20260902.log`.

### WGMMA iteration 3 — grouped async issue recovers some, still rejected

- Change: for the 128-channel variant, issue both independent 64x8 WGMMA
  operations into one commit group and wait once, following the already
  validated pattern in `step_e_rs2m.py`.  The 64-channel default compiles the
  same single-operation path as before.
- Correctness again passes TP4 balanced/skew and TP8-shape full-block tests;
  final W2 cosine is 0.999999999 with rel-L2 about 5.7e-5.
- Cold-L2 3x100 full-graph medians for M8/M16/M32/M64/M128 are
  0.151936 / 0.254800 / 0.453472 / 0.588352 / 0.602016 ms; geometric mean
  0.362019 ms.
- This recovers roughly 2% geometric mean versus the serialized 128-channel
  attempt, but remains 5.3% / 3.7% / 1.4% / 0.0% slower and only 0.2% faster
  at M128 than the 64-channel split-K=2 screen.  Five-point geometric mean is
  still 2.0% worse.  Reject the wider tile; keep `V4_WOUT=64` as default.
- Evidence logs:
  `bench/results/v4_flash_tp_wgmma_wo128_grouped_s2_correctness*_20260902.log`
  and
  `bench/results/tp4_wgmma_graph_coldl2_s2_wo128_grouped_screen_20260902.log`.

### Iteration 3 default-path regression audit

- Because the generic grouped code also compiles the default 64-channel path,
  it was re-screened under the same 3x100 cold-L2 protocol rather than assumed
  unchanged.  Medians are 0.148672 / 0.254720 / 0.464960 / 0.611488 /
  0.627200 ms; geometric mean 0.368045 ms.
- This is 3.7% slower in geometric mean than the pre-refactor split-K=2
  64-channel screen (0.354857 ms), so the generic refactor itself regresses the
  accepted default despite identical math and tile size.
- Root cause in source: the original keeps each WGMMA accumulator live across
  all four k32 operations in a K128 activation-scale group and applies the
  scale once.  The generic version creates a temporary accumulator per k32 and
  performs four scalar scaled accumulations.  Restore the original K128
  accumulator lifetime before evaluating grouped issue again.
- Evidence log:
  `bench/results/tp4_wgmma_graph_coldl2_s2_wo64_v3_screen_20260902.log`.

### WGMMA iteration 4 — K128-lived accumulators make 128-channel tile win

- Change: keep each output group's WGMMA accumulators live across all four
  k32 instructions in a K128 activation-scale group, issue the two independent
  64-row groups in one async commit, then apply activation scale once.  This
  restores the original 64-channel instruction structure and removes the
  scalar work introduced by iteration 3.
- Correctness passes TP4 balanced, TP4 maximal skew, and TP8 Is=256 full-block
  tests.  The worst listed final W2 result is cosine 0.999999992 and rel-L2
  0.000129309; all outputs are finite.
- Cold-L2 3x100 medians (M8/M16/M32/M64/M128):
  - 64-channel control: 0.144368 / 0.245344 / 0.447456 / 0.589088 /
    0.604144 ms; geometric mean 0.355029 ms.  This matches the pre-refactor
    0.354857 ms result within 0.05%.
  - 128-channel grouped issue: 0.144640 / 0.240624 / 0.428464 / 0.555296 /
    0.569792 ms; geometric mean 0.342576 ms.
- The 128-channel tile is 1.9% / 4.2% / 5.7% / 5.7% faster for M16 through
  M128, costs only 0.2% at M8, and improves five-point geometric mean by 3.5%.
  It is the first accepted core optimization, subject to the formal 9x200
  confirmation.  The default remains 64 until that confirmation completes.
- Evidence logs:
  `bench/results/v4_flash_tp_wgmma_wo{64,128}_accum_s2_correctness*_20260902.log`
  and
  `bench/results/tp4_wgmma_graph_coldl2_s2_wo{64,128}_accum_screen_20260902.log`.

### Iteration 4 formal cold-L2 confirmation

- Formal protocol: TP4 balanced full graph, 9x200 individually cold replays
  per M, max rank, `V4_W13_SPLIT_K=2 V4_WOUT=128`.
- Min / median / max latency (ms):
  - M8: 0.142112 / 0.144096 / 0.285920
  - M16: 0.237600 / 0.239712 / 0.634560
  - M32: 0.424352 / 0.428160 / 0.514944
  - M64: 0.558368 / 0.573552 / 1.396896
  - M128: 0.570272 / 0.589632 / 0.650912
- Geometric mean is 0.346593 ms, 3.4% faster than the accepted split-K=4,
  64-channel formal baseline (0.358661 ms).  Pointwise it is 1.4% slower at
  M8, then 1.4% / 5.0% / 5.6% / 6.0% faster from M16 through M128.
- Against the formal cold Humming medians, custom/Humming is 1.488x / 1.498x /
  1.456x / 1.412x / 1.429x.  Geometric-mean ratio improves from 1.507x to
  1.456x, but the target is still missed by a wide margin.
- Correctness passes all five points; minimum cosine 0.99999555, maximum
  rel-L2 0.00298332, all finite.  Large global maxima are tails; the nine
  batch medians remain the stability diagnostic.
- Evidence log:
  `bench/results/tp4_wgmma_graph_coldl2_s2_wo128_formal_20260902.log`.

### WGMMA iteration 5 — naive persistent grid is a large regression

- Change: following DeepGEMM MegaMoE's device-side schedule at a structural
  level, cap the physical grid and let each CTA traverse valid route/output
  tasks with a grid-stride loop.  LUT is loaded once per physical CTA and
  inactive `expert_ids` capacity does not enter the logical task count.
- TP4 balanced full-block correctness passes with 312 workers before timing.
- Cold-L2 3x100 five-point geometric means by physical worker count:
  - 0 (full grid control): 0.350811 ms.
  - 78: 1.337024 ms.
  - 156: 0.861095 ms.
  - 234: 0.648026 ms.
  - 312: 0.526386 ms.
- Point medians for 312 workers are 0.211456 / 0.351792 / 0.665984 /
  0.891248 / 0.915280 ms for M8 through M128, all much slower than the
  iteration-4 winner.  Smaller persistent grids regress even more severely.
- Diagnosis: this one-warpgroup CTA has no dedicated producer and no
  cross-task producer/consumer overlap.  It relies on many independent CTAs
  to hide TMA, dequantization, and WGMMA waits; serial tile traversal removes
  that latency hiding.  DeepGEMM MegaMoE's persistent schedule works together
  with producer/consumer warp specialization and a mailbox pipeline, so the
  scheduler cannot be transplanted alone.  Reject and restore iteration 4.
- The full-grid control is also slower than iteration 4 because every
  preallocated idle CTA now loads the LUT before discovering that its task is
  outside `num_tokens_padded`; this generic loop must not remain on the winner.
- Evidence logs:
  `bench/results/tp4_wgmma_graph_coldl2_s2_wo128_p{0,78,156,234,312}_screen_20260902.log`.

### Iteration 5 rollback — restore the non-persistent winner

- Removed the rejected grid-stride persistent loop and restored one logical
  route/output/split tile per CTA.  Apart from the JIT extension suffix (`v6`),
  the kernel and benchmark paths match the iteration-4 winner.
- Full-route TP4-shape correctness at M32 balanced passes: W13 cosine
  0.999999997, activation cosine 0.999999691, W2 cosine 0.999999992,
  W2 rel-L2 0.000127067, and all outputs finite.
- Required cold-L2 regression screen: TP4 balanced full CUDA Graph,
  3x100 individually cold replays per M, max rank,
  `V4_W13_SPLIT_K=2 V4_WOUT=128`.
- Min / median / max latency (ms):
  - M8: 0.142784 / 0.144544 / 0.288256
  - M16: 0.238208 / 0.240000 / 0.318848
  - M32: 0.425408 / 0.427744 / 0.464416
  - M64: 0.552416 / 0.555296 / 0.648256
  - M128: 0.566016 / 0.569888 / 0.593312
- Five-point geometric mean is 0.342249 ms.  This reproduces the earlier
  iteration-4 3x100 screen (0.342576 ms) within 0.10%, confirming the rollback
  recovered the winner.  Continue from this code, not the persistent variant.
- Evidence log:
  `bench/results/tp4_wgmma_graph_coldl2_wout128_s2_restore_20260902.log`.

### WGMMA iteration 6 — dual K32 accumulator chains regress

- Change: within each K128 tile, dequantize adjacent K32 slices into two
  independent accumulator chains and issue both chains in one WGMMA async
  group.  This halves commit/wait pairs from four to two per K128 and follows
  the previously explored `step_e_rs2c.py` scheduling idea while preserving
  the required per-K128 activation-scale promotion.
- Full-route TP4-shape correctness at M32 balanced passes: W13 cosine
  0.999999999, activation cosine 0.999999809, W2 cosine 0.999999997,
  W2 rel-L2 0.000086157, and all outputs finite.
- Required cold-L2 screen: TP4 balanced full CUDA Graph, 3x100 individually
  cold replays per M, max rank, `V4_W13_SPLIT_K=2 V4_WOUT=128`.
- Medians for M8/M16/M32/M64/M128 are 0.167600 / 0.258432 / 0.453344 /
  0.592800 / 0.606208 ms; geometric mean is 0.371292 ms.
- This is 8.5% slower in geometric mean than the restored winner (0.342249
  ms), and is slower at every M.  The likely cause is the longer live range
  and extra register footprint for two accumulator chains plus predecoded
  operands; fewer WGMMA commit/wait pairs do not offset that cost.  Reject and
  restore iteration 5 rollback (`v6`).
- Evidence log:
  `bench/results/tp4_wgmma_graph_coldl2_wout128_s2_dualchain_screen_20260902.log`.

### Iteration 6 rollback — restore single K128-lived chain

- Restored the exact iteration-5 rollback kernel (`v6`) after rejecting dual
  chains.  A fresh required cold-L2 3x100 TP4 screen gives medians 0.144352 /
  0.240032 / 0.427664 / 0.555872 / 0.569312 ms for M8 through M128 and a
  five-point geometric mean of 0.342156 ms.
- This is within 0.03% of the pre-experiment 0.342249 ms screen, confirming
  that the accepted implementation is recovered.
- Evidence log:
  `bench/results/tp4_wgmma_graph_coldl2_wout128_s2_postdual_restore_20260902.log`.

### WGMMA iteration 7 — 256-channel output tile regresses

- Change: extend the generic grouped WGMMA tile from 128 to 256 output
  channels, halving route/activation setup and CTA count again.  The dynamic
  shared-memory footprint remains about 34 KiB and both TP4/TP8 shapes divide
  evenly.
- Full-route TP4-shape correctness at M32 balanced passes: W13 cosine
  0.999999997, activation cosine 0.999999691, W2 cosine 0.999999992,
  W2 rel-L2 0.000127067, and all outputs finite.
- Required cold-L2 3x100 TP4 screen medians for M8/M16/M32/M64/M128 are
  0.168336 / 0.296672 / 0.508864 / 0.642160 / 0.657344 ms; geometric mean
  is 0.403737 ms.  This is 18.0% slower than the 128-channel winner's latest
  0.342156 ms, with no winning M point.
- Cubin resource usage explains the reversal: the TP4 route kernels rise from
  45 registers/thread at WOUT=128 to 69 at WOUT=256 (no local-memory spill).
  For 128-thread CTAs this cuts the register-limited resident CTA/warp budget
  sharply, outweighing reduced CTA setup.  Reject WOUT=256 and restore 128.
- Evidence log:
  `bench/results/tp4_wgmma_graph_coldl2_wout256_s2_screen_20260902.log`.

### Iteration 7 selection — make the measured winner the default

- Changed the no-environment defaults from the historical
  `W13_SPLIT_K=4, WOUT=64` to the accepted `W13_SPLIT_K=2, WOUT=128`.
  Experimental alternatives remain explicit environment overrides.
- The unset-environment full-route M32 check passes with W2 cosine
  0.999999992 and rel-L2 0.000127067.
- A fresh required cold-L2 3x100 TP4 screen with both tuning variables unset
  gives medians 0.144448 / 0.240128 / 0.427520 / 0.556000 / 0.569520 ms for
  M8 through M128 and geometric mean 0.342246 ms.  This exactly reproduces
  the selected winner within measurement noise.
- Evidence log:
  `bench/results/tp4_wgmma_graph_coldl2_default_s2_wout128_screen_20260902.log`.

### W2 iteration 8 — route output plus SGLang reduction wins narrowly

- Change: replace the W2 kernel's FP32 weighted atomic scatter and following
  BF16 cast with direct per-route BF16 stores, then invoke the exact SGLang
  `moe_fused_mul_sum` used by the Humming baseline.  This preserves precomputed
  routes and the 1.5 routed scaling factor; it does not add dispatch/combine.
- Full-route TP4-shape M32 balanced correctness against the dequantized torch
  reference passes: W2 cosine 0.999997241, rel-L2 0.002349063, all finite.
  The modest error increase is expected from matching Humming's BF16 route
  output before top-k reduction.
- Required cold-L2 3x100 TP4 screen medians for M8/M16/M32/M64/M128 are
  0.143744 / 0.238848 / 0.425952 / 0.552144 / 0.563376 ms; geometric mean
  is 0.340083 ms.
- Every point improves versus the immediately preceding default-control run,
  and the five-point geometric mean improves by 0.63%.  Atomic scatter was a
  secondary cost, not the core GEMM gap.  Keep provisionally and require a
  9x200 confirmation plus skew and TP8 numerical checks.
- Evidence log:
  `bench/results/tp4_wgmma_graph_coldl2_w2_route_output_screen_20260902.log`.

### Iteration 8 formal cold-L2 confirmation

- Additional full-route checks pass for maximal-skew TP4 at M128 and the TP8
  local shape (`intermediate_per_rank=256`) at M32.  Worst reported W2 cosine
  is 0.999997240 and rel-L2 is 0.002349543; all outputs are finite.
- Formal protocol: TP4 balanced full graph, 9x200 individually cold replays
  per M, max rank, default split-K/output tile with
  `V4_W2_ROUTE_OUTPUT=1`.
- Min / median / max latency (ms):
  - M8: 0.141344 / 0.143712 / 0.291584
  - M16: 0.236512 / 0.238848 / 0.358208
  - M32: 0.422688 / 0.426400 / 0.492320
  - M64: 0.553120 / 0.569984 / 0.627200
  - M128: 0.563200 / 0.583104 / 0.787424
- Geometric mean is 0.344674 ms.  Against the prior 9x200 atomic formal run,
  route output is 0.27% / 0.36% / 0.41% / 0.63% / 1.12% faster and improves
  geometric mean by 0.56%.  Accept it despite the small margin because all
  five points agree and the larger sample confirms the effect.
- Against formal cold Humming, custom/Humming latency ratios remain 1.484x /
  1.492x / 1.450x / 1.403x / 1.413x; geometric-mean ratio is 1.448x.  This
  optimization does not alter the conclusion that the GEMM core dominates
  the remaining deficit.
- Evidence log:
  `bench/results/tp4_wgmma_graph_coldl2_w2_route_output_formal_20260902.log`.

### Iteration 8 selection — route output is now the default

- Set `V4_W2_ROUTE_OUTPUT`'s unset default to the formally accepted route
  output path.  With all tuning environment variables unset, a fresh required
  cold-L2 3x100 TP4 screen gives 0.143552 / 0.238368 / 0.426208 / 0.551536 /
  0.564592 ms and geometric mean 0.339968 ms.  Correctness passes all points.
- Evidence log:
  `bench/results/tp4_wgmma_graph_coldl2_accepted_defaults_screen_20260902.log`.

### WGMMA iteration 9 — forced 12-CTA launch bound regresses

- Change: add a configurable minimum-block launch bound and test
  `__launch_bounds__(128, 12)` on the accepted WOUT=128 route-output kernel.
  The intent was to relieve the register-limited occupancy found in cubin
  inspection.
- The compiler does honor it: all TP4 route-GEMM specializations fall from
  45 to 40 registers/thread, with `LOCAL:0`.  Full-route M32 correctness is
  unchanged (W2 cosine 0.999997241, rel-L2 0.002349063).
- Required cold-L2 3x100 TP4 medians for M8/M16/M32/M64/M128 are 0.148064 /
  0.248224 / 0.441760 / 0.571088 / 0.584384 ms; geometric mean is 0.352189
  ms.  This is 3.6% slower than the immediately preceding accepted-default
  screen (0.339968 ms), with every point regressing.
- Conclusion: the additional nominal residency does not compensate for the
  compiler's reduced scheduling freedom/ILP.  Reject `min_blocks=12` and keep
  unconstrained launch bounds.
- Evidence log:
  `bench/results/tp4_wgmma_graph_coldl2_mb12_screen_20260902.log`.

### Iteration 9 rollback — unconstrained launch bounds remain default

- Recompiled the same source with the unset/default `min_blocks=0` path and
  reran the required cold-L2 3x100 TP4 screen.  Medians are 0.143824 /
  0.239376 / 0.426336 / 0.552048 / 0.564608 ms; geometric mean is 0.340469
  ms, within 0.15% of the pre-experiment accepted-default screen.  All points
  pass correctness.  Keep the launch-bounds override diagnostic-only.
- Evidence log:
  `bench/results/tp4_wgmma_graph_coldl2_mb0_restore_screen_20260902.log`.

### Accepted WOUT=128 cold profile

- Reprofiled the current default at TP4 M32 after route-output acceptance.
  The profiler range is explicitly `256 MiB L2 clear -> one local pipeline`,
  and NCU also uses cache-control `all` on every replay.
- W13 / W2 route-GEMM metrics: 3072 / 6144 CTAs, 3.94 / 7.88 waves per
  SM, 45 registers/thread (48 allocated), 62.5% theoretical occupancy,
  58.45% / 59.73% achieved occupancy, 31.57% / 32.31% DRAM throughput, and
  289.632 / 140.320 us NCU duration.
- Compared with the earlier WOUT=64 profile, grouping two output tiles halves
  the grid and raises DRAM utilization by only about 2-3 points while reducing
  occupancy.  L1TEX throughput remains high (83.0% / 72.7%), so the next
  experiment moves the strided E8M0 scale loads/stores from all 128 threads to
  an asynchronous TMA transaction per stage.
- Evidence report:
  `bench/results/tp4_wgmma_m32_wout128_route_coldl2_ncu.ncu-rep`.

### WGMMA iteration 10 — TMA E8M0 scale stages are a large win

- Change: encode the strided E8M0 weight-scale tensor as a second TMA map and
  attach its transfer to the packed-weight stage's existing mbarrier.  The
  inner contiguous TMA box must be 16 bytes, so each row fetches the aligned
  group of four K128 scale quads and consumes the quad selected by
  `global_kt % 4`.  This replaces hundreds of per-CTA scalar global byte loads
  and shared stores with one asynchronous TMA transaction per K128 stage.
- Two implementation hazards were caught before timing: a 4-byte box is
  rejected by `cuTensorMapEncodeTiled`, and an unaligned 4-byte coordinate
  causes an illegal instruction.  The accepted descriptor uses a 16-byte box
  and aligned coordinates.  TP8 W2 (`K/32=8` byte row stride) retains a
  compile-time scalar-load fallback because its global stride cannot satisfy
  TMA's 16-byte requirement.
- Full-route TP4 M32 balanced correctness passes unchanged: W13 cosine
  0.999999997, activation cosine 0.999999691, W2 cosine 0.999997241,
  W2 rel-L2 0.002349063, all finite.
- Required cold-L2 3x100 TP4 medians for M8/M16/M32/M64/M128 are 0.124688 /
  0.193952 / 0.343264 / 0.461008 / 0.471136 ms; geometric mean is 0.282618
  ms.  Versus the immediately preceding accepted control, latency falls by
  17.0% geometrically (1.205x speedup), with every M winning.
- Against formal cold Humming at this screening stage, custom/Humming ratios
  are about 1.287x / 1.212x / 1.168x / 1.135x / 1.141x.  Require formal
  9x200 confirmation and TP8/skew correctness before final acceptance.
- Evidence log:
  `bench/results/tp4_wgmma_graph_coldl2_tma_scales_screen_20260902.log`.

### Iteration 10 formal confirmation and routing checks

- Full-route maximal-skew TP4 M128 and balanced TP8-local-shape M32 checks
  both pass.  The TP8 W2 scalar-scale fallback is exercised; worst W2 cosine
  is 0.999997240, worst rel-L2 is 0.002349543, and all outputs are finite.
- Formal protocol: TP4 balanced full graph, 9x200 individually cold replays
  per M, max rank, all accepted settings at their unset defaults.
- Min / median / max latency (ms):
  - M8: 0.121440 / 0.124288 / 0.264416
  - M16: 0.190848 / 0.193152 / 0.244320
  - M32: 0.334624 / 0.372864 / 0.471584
  - M64: 0.456192 / 0.494304 / 0.542528
  - M128: 0.472800 / 0.512704 / 0.545408
- Geometric mean is 0.295902 ms, a 14.15% latency reduction (1.165x
  speedup) against the pre-TMA-scale route-output formal result (0.344674
  ms).  Accept the optimization.
- Machine contention/clock drift is visible in the batch medians: M32 rises
  from about 0.345 to 0.425 ms and M64/M128 also rise later in the run.  The
  old Humming formal run was collected in a different time window, so its
  nominal custom/Humming ratios (1.283x / 1.207x / 1.268x / 1.217x / 1.242x)
  are diagnostic only.  Re-run Humming cold in the same current window before
  reporting a new apples-to-apples gap.
- Evidence log:
  `bench/results/tp4_wgmma_graph_coldl2_tma_scales_formal_20260902.log`.

### Contemporary cold-L2 baseline and repeat

- Re-ran Humming and then custom in the same GPU window with the identical
  9x200 per-point cold protocol.  Humming medians are 0.096960 / 0.160192 /
  0.293440 / 0.402896 / 0.412240 ms; geometric mean 0.237585 ms.
- The immediate custom repeat gives 0.124512 / 0.193312 / 0.373488 /
  0.497968 / 0.510480 ms; geometric mean 0.296337 ms.  Its geometric mean is
  within 0.15% of the preceding custom formal run, so the result is
  reproducible despite visible within-run batch drift at M32 and above.
- On the contemporary pair, custom/Humming latency ratios are 1.284x / 1.207x
  / 1.273x / 1.236x / 1.238x for M8/M16/M32/M64/M128; geometric-mean ratio
  1.247x.  The TMA scale optimization closes most of the former ~45% gap but
  does not yet beat Humming.
- Evidence logs:
  `bench/results/tp4_humming_graph_coldl2_formal_contemporary_20260902.log`
  and
  `bench/results/tp4_wgmma_graph_coldl2_tma_scales_formal_repeat_20260902.log`.

### TMA-scale cold profile

- Repeated the same cold NCU profile after iteration 10.  W13 falls from
  289.632 to 217.70 us and DRAM throughput rises from 31.57% to 41.91%; W2
  falls from 140.320 to 119.36 us and DRAM rises from 32.31% to 37.95%.
  This independently confirms the graph-level speedup comes from both GEMM
  cores, not a timing artifact in the epilogue or all-reduce.
- TMA scale staging increases per-block dynamic shared memory to 21.50 KB.
  Both kernels are now shared-memory limited to nine resident CTAs (56.25%
  theoretical occupancy); achieved occupancy is 52.56% / 53.84%.  W13 has
  4.38 waves and an NCU-estimated partial-wave tail of up to 20%, motivating
  a fresh split-K screen under the new pipeline.
- Evidence report:
  `bench/results/tp4_wgmma_m32_tma_scales_coldl2_ncu.ncu-rep`.

### TMA-scale split-K refresh

- Re-screened W13 split-K=4 after the scale pipeline changed the core balance.
  Cold 3x100 medians for M8/M16/M32/M64/M128 are 0.116176 / 0.187232 /
  0.363952 / 0.461488 / 0.480784 ms; geometric mean 0.281145 ms.
- Compared with the split-K=2 TMA-scale screen, split-K=4 wins clearly only
  at M8/M16, is roughly tied at M64, and loses at M32/M128.  A dedicated
  9x200 cold confirmation for M8/M16 is stable at 0.116320 / 0.187136 ms.
  Against split-K=2's formal low-M values this is 6.41% / 3.11% lower
  latency; against contemporary Humming it leaves 1.200x / 1.168x ratios.
- Accept split-K=4 for M<=16 and keep split-K=2 for M>=32.  The current
  compile-time setting must be replaced by a dual-specialization runtime
  dispatch before this can be the actual default path.
- Evidence logs:
  `bench/results/tp4_wgmma_graph_coldl2_tma_scales_s4_screen_20260902.log`
  and
  `bench/results/tp4_wgmma_graph_coldl2_tma_scales_s4_lowm_formal_20260902.log`.

### WGMMA iteration 11 — runtime W13 split-K dispatch

- Change: compile both W13 split-K=2 and split-K=4 specializations into the
  same extension and select one before CUDA Graph capture.  The default
  routed-row policy uses split-K=4 for at most 96 routed rows (M<=16 for
  top-k=6) and split-K=2 otherwise; `V4_W13_SPLIT_K=2|4` remains available
  for controlled experiments.  Partial and reduction buffers reserve four
  planes so both graph-specialized paths share one stable allocation.
- This selector intentionally uses the real routed-row count, not
  `expert_ids.numel()`: the latter is padded routing-buffer capacity and was
  proven not to represent active experts or route imbalance.  Consequently
  the accepted default is M/routed-row based, not falsely distribution-aware;
  skewed-route split-K=2/4 measurements remain a separate required check.
- Balanced TP4 correctness passes at both dispatch branches: M8 selects 4,
  M32 selects 2, and worst final cosine is 0.99999562 with rel-L2 0.00298.
  Maximal-skew TP4 M128 selects 2 and passes, and TP8-local-shape M32 selects
  2 and exercises the K=256 scalar-scale fallback; worst W2 cosine across
  those checks is 0.999997240 and all outputs are finite.
- Required cold-L2 3x100 TP4 min / median / max latency (ms):
  - M8, split 4: 0.114816 / 0.116512 / 0.255424
  - M16, split 4: 0.185600 / 0.187504 / 0.220032
  - M32, split 2: 0.334432 / 0.347344 / 0.425376
  - M64, split 2: 0.448224 / 0.457440 / 0.495936
  - M128, split 2: 0.456960 / 0.472736 / 0.532448
- Geometric-mean median is 0.277344 ms, 1.87% lower than the fixed split-K=2
  TMA-scale screening result (0.282618 ms).  Accept the dual specialization
  and routed-row dispatch; run a formal 9x200 confirmation after the skew
  policy check.
- Evidence log:
  `bench/results/tp4_wgmma_graph_coldl2_tma_scales_dynamic_split_screen_20260902.log`.

### WGMMA iteration 12 — active-expert-aware split dispatch

- The maximal-skew route (all tokens select the same six experts) disproves
  the assumption that routed rows alone determine the best W13 split.  With
  M128 fixed, cold 3x100 split-K=2 and split-K=4 medians are 0.202848 and
  0.199360 ms in the first paired screen, a 1.72% split-K=4 advantage.
- Expanded forced-policy screens show split-K=4 versus split-K=2 medians for
  skew M8/M16/M32/M64 of 0.045728/0.055904/0.085536/0.123232 ms versus
  0.054032/0.059760/0.085472/0.131504 ms.  Split-K=4 wins clearly at M8,
  M16, and M64, while M32 is tied.  A later M128 replay is 0.203872 ms, only
  0.50% slower than the earlier split-K=2 result, so the M128 direction is
  within short-screen drift rather than established.
- Change: when route IDs are fixed benchmark inputs, count their unique
  experts once before graph capture and select split-K=4 if routed rows <=96
  or active experts <=96; otherwise select split-K=2.  The one-time route
  inspection and host decision are explicitly outside the timed graph.  The
  kernel wrapper also accepts an explicit split so W13 and its reduction
  cannot choose inconsistent specializations.
- The resulting automatic skew cold 3x100 medians for M8/M16/M32/M64/M128
  are 0.045760 / 0.056000 / 0.085568 / 0.122960 / 0.203872 ms, all selecting
  split-K=4 and all passing end-to-end correctness (worst cosine 0.99999553,
  worst rel-L2 0.00299).  Versus the forced split-K=2 screens, geometric-mean
  latency falls from 0.094059 to 0.088720 ms (5.68%, 1.060x speedup).
- This is a static-route CUDA-Graph policy, not a claim that production can
  obtain active-expert count for free when route IDs change every replay.
  The cost/location of dynamic route statistics must be included before
  integrating this dispatch into such a runtime.
- Evidence logs:
  `bench/results/tp4_wgmma_graph_coldl2_tma_scales_skew_m128_s2_screen_20260902.log`,
  `bench/results/tp4_wgmma_graph_coldl2_tma_scales_skew_m128_s4_screen_20260902.log`,
  `bench/results/tp4_wgmma_graph_coldl2_tma_scales_skew_s2_screen_20260902.log`,
  `bench/results/tp4_wgmma_graph_coldl2_tma_scales_skew_s4_screen_20260902.log`,
  and
  `bench/results/tp4_wgmma_graph_coldl2_tma_scales_active_dispatch_skew_screen_20260902.log`.

### Benchmark correction — random-score routes and explicit padding

- DeepGEMM's existing MegaMoE benchmark constructs routes by drawing random
  expert scores and applying top-k.  Our former `balanced` sequence instead
  maximizes the number of touched experts and therefore maximizes 8-row
  per-expert padding; it is a useful spread stress test, but not a defensible
  proxy for ordinary routing.
- Both the Humming baseline and custom graph benchmark now support and default
  to `random`: a private CPU generator with the same seed on every TP rank
  generates random scores, then top-k IDs are copied to the GPU before graph
  capture.  Router/top-k work remains excluded exactly as requested.  The
  logs now expose both `padded_rows` and `padding_ratio`; `balanced` and
  maximal `skew` remain explicit stress modes.
- Seed 20260902 random routes touch 43/82/140/203/248 experts and align to
  344/656/1120/1624/1992 rows for M8/M16/M32/M64/M128, versus 48/96/192/
  384/768 actual routed rows.  Custom and Humming report identical metadata.
- Contemporary cold-L2 3x100 Humming medians are 0.090240 / 0.143296 /
  0.225408 / 0.313792 / 0.381536 ms (geomean 0.203496 ms).  Custom medians
  are 0.106464 / 0.167040 / 0.264160 / 0.368688 / 0.461600 ms (geomean
  0.240194 ms), with every correctness check passing.  Custom/Humming ratios
  are 1.180x / 1.166x / 1.172x / 1.175x / 1.210x; geomean 1.180x.
- These are screening results, not the final claim.  They show that route
  distribution materially changes the measured gap and require a paired
  9x200 cold confirmation.
- Evidence logs:
  `bench/results/tp4_humming_graph_coldl2_random_screen_20260902.log` and
  `bench/results/tp4_wgmma_graph_coldl2_random_screen_20260902.log`.

### Random-route formal cold-L2 baseline

- Formal protocol is 9 batches x 200 individually cold graph replays per M;
  every replay is preceded by the 256 MiB flush outside its CUDA events, and
  latency is the max rank for each replay.
- Humming min / median / max latency (ms):
  - M8: 0.088448 / 0.090016 / 0.247488
  - M16: 0.141088 / 0.142976 / 0.847264
  - M32: 0.222848 / 0.225088 / 0.275456
  - M64: 0.312288 / 0.329184 / 0.402944
  - M128: 0.380608 / 0.411104 / 0.467296
- Custom min / median / max latency (ms):
  - M8: 0.104576 / 0.106496 / 0.251904
  - M16: 0.164896 / 0.166880 / 0.193536
  - M32: 0.261568 / 0.281264 / 0.334240
  - M64: 0.368832 / 0.396288 / 0.430656
  - M128: 0.472224 / 0.495952 / 0.559200
- Humming/custom geometric means are 0.208288/0.250300 ms.  Custom/Humming
  latency ratios are 1.183x/1.167x/1.250x/1.204x/1.206x, geomean 1.202x;
  the current implementation therefore remains 20.2% slower geometrically.
  All end-to-end correctness checks pass.
- M32+ batch medians visibly drift within each sequential process run, so a
  future interleaved harness is desirable.  This formal pair is retained as
  the conservative headline until such a harness is verified; the shorter
  contemporary screen gave an 18.0% gap.
- Evidence logs:
  `bench/results/tp4_humming_graph_coldl2_random_formal_20260902.log` and
  `bench/results/tp4_wgmma_graph_coldl2_random_formal_20260902.log`.

### WGMMA iteration 13 — 128-row windowed LUT regresses

- Change: use DeepGEMM's bit-exact 128-row E8M0 LUT window and clamp scale
  codes through `e8m0_lut_index`, instead of copying all 256 rows into shared
  memory.  This saves 1 KiB static shared memory per CTA and was intended to
  raise the shared-memory residency limit from nine to ten CTAs per SM.
- Full-block correctness passes for both TP4 (I/rank=512) and the TP8 local
  shape (I/rank=256): worst W2 cosine is 0.999997241/0.999997267 and all
  outputs are finite.
- Contemporary random-route cold 3x100 medians for the 256-row control are
  0.106304 / 0.167104 / 0.264192 / 0.367776 / 0.459008 ms (geomean
  0.239757 ms).  The immediately repeated 128-row variant gives 0.108832 /
  0.171520 / 0.270048 / 0.380064 / 0.473488 ms (geomean 0.246340 ms).
- The windowed variant is slower at every M by 2.2-3.3%, and regresses
  geometric-mean latency by 2.75%.  Any occupancy opportunity is outweighed
  by the hot-loop clamp/index instructions.  Reject it and restore the
  branch-free 256-row lookup as default; retain `V4_LUT_ROWS=128` only as an
  auditable experiment switch.
- Evidence logs:
  `bench/results/v4_flash_tp_wgmma_lut128_correctness_tp4_20260902.log`,
  `bench/results/v4_flash_tp_wgmma_lut128_correctness_tp8shape_20260902.log`,
  `bench/results/tp4_wgmma_graph_coldl2_random_lut128_screen_20260902.log`,
  `bench/results/tp4_wgmma_graph_coldl2_random_lut256_control_screen_20260902.log`,
  and
  `bench/results/tp4_wgmma_graph_coldl2_random_lut128_repeat_screen_20260902.log`.

### Iteration 13 rollback — restore 256-row LUT default

- Restored the branch-free 256-row shared LUT as the unset default while
  retaining `V4_LUT_ROWS=128` for reproducibility.  The extension remains
  keyed by LUT size, so changing the experiment switch cannot reuse the wrong
  binary.
- Unset-default random-route cold 3x100 medians are 0.106208 / 0.167168 /
  0.264288 / 0.368080 / 0.461104 ms (geomean 0.240008 ms), matching the
  explicit 256-row control's 0.239757 ms geomean within 0.11%.  All graph
  correctness checks pass.
- Evidence log:
  `bench/results/tp4_wgmma_graph_coldl2_random_lut256_default_restore_20260902.log`.

### WGMMA iteration 14 — reuse one scale quartet across four K128 tiles

- Observation: the 16 E8M0 bytes fetched per output row cover four adjacent
  K128 tiles, but iteration 10 redundantly issued the same scale TMA for every
  tile.  Change the double-buffer schedule so the first quartet arrives with
  tile 0 and each following quartet is prefetched alongside the preceding
  quartet's final weight tile.  Four compute tiles then reuse that shared
  scale block.  Weight TMA, WGMMA math, and shared allocation are unchanged;
  `V4_SCALE_QUAD_REUSE=1` preserves an exact control specialization.
- Full-block correctness passes W13 split-K=2 and split-K=4 at TP4, plus the
  TP8 local shape whose W2 K=256 path still uses scalar scale loads.  Worst W2
  cosine is 0.999997241 and all outputs are finite.
- Contemporary random-route cold 3x100 control medians (reuse=1) are
  0.106528 / 0.167456 / 0.263872 / 0.368800 / 0.460144 ms; geometric mean
  0.240153 ms.  Reuse=4 gives 0.103712 / 0.162656 / 0.258208 / 0.354560 /
  0.440512 ms; geometric mean 0.232564 ms.
- Reuse=4 wins every M by 2.15-4.27% and reduces geometric-mean latency by
  3.16% (1.033x speedup).  Accept it as the default, subject to formal 9x200
  confirmation.
- Evidence logs:
  `bench/results/v4_flash_tp_wgmma_scale_reuse4_correctness_tp4_20260902.log`,
  `bench/results/v4_flash_tp_wgmma_scale_reuse4_correctness_tp4_split4_20260902.log`,
  `bench/results/v4_flash_tp_wgmma_scale_reuse4_correctness_tp8shape_20260902.log`,
  `bench/results/tp4_wgmma_graph_coldl2_random_scale_reuse1_control_20260902.log`,
  and
  `bench/results/tp4_wgmma_graph_coldl2_random_scale_reuse4_screen_20260902.log`.

### Iteration 14 formal confirmation and contemporary Humming gap

- Repeated both implementations in the same machine window with the formal
  9x200 individually cold protocol.  Humming medians for M8/M16/M32/M64/M128
  are 0.090208 / 0.142880 / 0.225312 / 0.332864 / 0.409184 ms; geometric
  mean 0.208659 ms.
- Scale-reuse custom medians are 0.103808 / 0.162400 / 0.257408 / 0.376240 /
  0.476144 ms; geometric mean 0.238852 ms.  Relative to the earlier pre-reuse
  custom formal geomean (0.250300 ms), this is a 4.57% reduction, consistent
  in direction with the controlled screen.  Every correctness check passes.
- Custom/Humming latency ratios are 1.151x / 1.137x / 1.142x / 1.130x /
  1.164x; geometric-mean ratio 1.145x.  The current implementation remains
  13.0-16.4% slower pointwise and 14.47% slower geometrically; it has not met
  the beat-Humming target.
- Evidence logs:
  `bench/results/tp4_humming_graph_coldl2_random_formal_scale_reuse_window_20260902.log`
  and
  `bench/results/tp4_wgmma_graph_coldl2_random_scale_reuse4_formal_20260902.log`.

### WGMMA iteration 15 — one scale buffer regresses

- Change: retain quartet reuse but replace the two 2 KiB scale buffers with
  one.  A dedicated mbarrier loads the next quartet only after the current
  quartet's fourth tile has consumed the buffer.  This saves 2 KiB/CTA and
  should permit ten rather than nine resident CTAs per SM, at the cost of one
  exposed scale-TMA boundary every four K128 tiles.  TP8 W2's scalar-scale
  fallback deliberately retains two physical stage buffers.
- Full-block correctness passes TP4 split-K=2, TP4 split-K=4, and TP8 local
  shapes; worst W2 cosine is 0.999997241 and all results are finite.
- Contemporary random-route cold 3x100 double-buffer control medians are
  0.103840 / 0.162656 / 0.258000 / 0.356224 / 0.442080 ms (geomean
  0.232968 ms).  The one-buffer variant gives 0.104256 / 0.164320 / 0.260880 /
  0.356720 / 0.447568 ms (geomean 0.234792 ms).
- The one-buffer path loses at every M by 0.14-1.24% and regresses geometric
  mean by 0.78%.  The exposed quartet-boundary wait outweighs the occupancy
  benefit.  Reject and restore two scale buffers as default; retain the
  single-buffer specialization for audit only.
- Evidence logs:
  `bench/results/v4_flash_tp_wgmma_scale_buffer1_correctness_tp4_20260902.log`,
  `bench/results/v4_flash_tp_wgmma_scale_buffer1_correctness_tp4_split4_20260902.log`,
  `bench/results/v4_flash_tp_wgmma_scale_buffer1_correctness_tp8shape_20260902.log`,
  `bench/results/tp4_wgmma_graph_coldl2_random_scale_buffer1_screen_20260902.log`,
  and
  `bench/results/tp4_wgmma_graph_coldl2_random_scale_buffer2_control_20260902.log`.

### Iteration 15 rollback — restore two scale buffers

- Restored two prefetched scale buffers as the unset default.  A cold 3x100
  regression check at M8/M32/M128 reports 0.103520 / 0.254784 / 0.441936 ms,
  all with `scale_buffers=2`, expected split selection, and passing graph
  correctness.  The first identical run also completed remotely before its
  SSH client timed out; the clean repeat is the accepted evidence.
- Evidence log:
  `bench/results/tp4_wgmma_graph_coldl2_random_scale_buffer2_default_restore_repeat_20260903.log`.

### Post-iteration-15 profile — random-route cold-L2 core gap

- Repaired the local profiler to use the benchmark's current route API and
  made `random` (DeepGEMM MegaMoE-style random scores followed by top-k) its
  default.  The route tensor is generated before capture and is identical for
  both implementations.  The profiler also records the LUT, scale-reuse, and
  scale-buffer specialization in every log.
- Profiled M32, TP4 local W13+W2 pipelines on GPU 1.  Each NCU launch has a
  stream-ordered 256 MiB cache flush immediately before the profiled pipeline;
  the flush itself is outside the target NVTX range.  These NCU durations are
  diagnostic metric-replay measurements, not benchmark headline latency.
- Custom W13 (`K=4096,N=1024,split-K=2`) takes 167.04 us versus Humming's
  146.82 us, a 13.77% core gap.  Custom W2 (`K=512,N=4096`) takes 88.03 us
  versus Humming's 74.59 us, an 18.02% core gap.
- Custom W13/W2 DRAM-throughput utilization is 41.16%/37.76%, L2 utilization
  56.70%/50.76%, and compute utilization 70.56%/70.60%.  Humming reports
  44.58%/44.42%, 54.85%/52.68%, and 62.01%/59.19%, respectively.  Humming is
  faster despite lower compute utilization, consistent with its persistent
  producer/consumer schedule doing less per-tile control and movement work.
- Custom W13/W2 achieve 51.20%/53.54% occupancy with 45 registers and 24.57 KiB
  total shared memory per CTA.  Scheduler issue is healthy but not saturated:
  eligible cycles are 73.76%/72.18%, with 8.18/8.53 active and 1.96/2.01
  eligible warps per scheduler.
- NCU flags 45% excessive shared-memory wavefronts for both layers
  (6,881,280 of 15,182,720 W13 wavefronts and 3,440,640 of 7,678,720 W2
  wavefronts).  Source attribution includes WGMMA shared-operand activity, so
  this aggregate alone does not prove that the explicit packed-weight loads
  are bank-conflicted.  The next experiment must isolate those source lines
  before changing the TMA/shared layout.
- Evidence:
  `bench/results/tp4_wgmma_m32_scale_reuse4_random_coldl2_ncu.ncu-rep`,
  `bench/results/tp4_wgmma_m32_scale_reuse4_random_coldl2_ncu.log`,
  `bench/results/tp4_humming_m32_random_coldl2_ncu.ncu-rep`,
  `bench/results/tp4_humming_m32_random_coldl2_ncu.log`,
  `bench/results/tp4_wgmma_m32_scale_reuse4_random_stalls_coldl2_ncu.ncu-rep`,
  and
  `bench/results/tp4_wgmma_m32_scale_reuse4_random_stalls_coldl2_ncu.log`.

### WGMMA iteration 16a — fixed-origin 64-byte weight swizzle is invalid

- Source-level NCU attribution resolves the aggregate shared-memory warning:
  every excessive wavefront belongs to the sixteen unrolled `LDS` packed
  weight loads in W13 (and the corresponding loads in W2).  Each reports four
  wavefronts for one ideal wavefront; no WGMMA descriptor access contributes
  to this counter.  The current packed-weight access is therefore a real
  four-way bank conflict.
- First attempt: enable `CU_TENSOR_MAP_SWIZZLE_64B` and read logical 16-byte
  chunk `x` at physical chunk `x ^ (row & 3)`, initially assuming that the
  dynamic shared-memory base is aligned to the swizzle-pattern origin.
- Reject before timing: TP4 full-block correctness fails badly (W13 cosine
  0.246405566, W2 cosine 0.251794636).  CUDA's TMA mapping also includes
  `(smem_pointer / 128) % 4`; an aligned dynamic declaration does not justify
  compiling that runtime origin term away.  Retain this failed state in git
  before testing the corrected address mapping.
- Evidence log:
  `bench/results/v4_flash_tp_wgmma_weight_swizzle64_correctness_tp4_20260903.log`.

### WGMMA iteration 16b — pointer offset alone does not repair the mapping

- Added CUDA's documented runtime origin term,
  `(weight_smem_address / 128) % 4`, to the 64-byte TMA swizzle index.
- Reject before timing: TP4 correctness is byte-for-byte unchanged from 16a
  (W13 cosine 0.246405566, W2 cosine 0.251794636).  The observed dynamic
  shared-memory origin is already on the zero phase, so the missing term is
  the swizzle row coordinate itself.
- A 64-byte logical tensor row occupies half of a 128-byte shared-memory bank
  line.  The TMA pattern's `y` coordinate is consequently `logical_row / 2`,
  while the low/high 64-byte half is preserved.  The next specialization uses
  `chunk ^ (((logical_row >> 1) + origin) & 3)`.
- Evidence log:
  `bench/results/v4_flash_tp_wgmma_weight_swizzle64_offset_correctness_tp4_20260903.log`.

### WGMMA iteration 16c — correct 128-byte-line coordinate removes the conflict

- Corrected the 64-byte TMA swizzle address to
  `chunk ^ (((logical_row >> 1) + origin) & 3)`.  Two 64-byte logical weight
  rows share one 128-byte shared-memory bank line, while the low/high half of
  that line remains unchanged by the two-bit XOR.
- Full-block correctness passes TP4 split-K=4, TP4 forced split-K=2, and the
  TP8 local shape (including its K=256 scalar-scale fallback).  Worst W2
  cosine is 0.999997256/0.999997278 and every output is finite.
- Contemporary TP4 random-route cold 3x100 control medians without swizzle
  are 0.103456 / 0.162528 / 0.257952 / 0.357488 / 0.437616 ms; geometric mean
  0.232442 ms.  The immediately following 64-byte-swizzle medians are
  0.102736 / 0.160016 / 0.252224 / 0.353456 / 0.437920 ms; geometric mean
  0.229869 ms.
- Swizzling improves M8/M16/M32/M64 by 0.70%/1.55%/2.22%/1.13%, is 0.07%
  slower at M128 (noise-sized), and improves geometric mean by 1.11%.  Keep
  it experimental (`V4_WEIGHT_SWIZZLE=64`) until a repeat screen, source NCU,
  and formal 9x200 run confirm the small gain.
- Evidence logs:
  `bench/results/v4_flash_tp_wgmma_weight_swizzle64_line_correctness_tp4_20260903.log`,
  `bench/results/v4_flash_tp_wgmma_weight_swizzle64_line_correctness_tp4_split2_20260903.log`,
  `bench/results/v4_flash_tp_wgmma_weight_swizzle64_line_correctness_tp8shape_20260903.log`,
  `bench/results/tp4_wgmma_graph_coldl2_random_weight_swizzle0_control_20260903.log`,
  and
  `bench/results/tp4_wgmma_graph_coldl2_random_weight_swizzle64_screen_20260903.log`.

### Iteration 16c repeat, NCU proof, and formal confirmation

- The reverse-order 3x100 repeat confirms the result.  Swizzled medians are
  0.102528 / 0.159296 / 0.251936 / 0.354144 / 0.437216 ms (geomean
  0.229532 ms), followed by unswizzled control medians 0.103552 / 0.162464 /
  0.257840 / 0.355104 / 0.439632 ms (geomean 0.232350 ms).  Swizzle wins all
  five points by 0.27-2.29% and improves geometric mean by 1.21%.
- SourceCounters NCU proves the intended mechanism.  W13 shared wavefronts
  fall from 15,182,720 (8,301,440 ideal plus 6,881,280 excessive) to exactly
  8,301,440 ideal, with zero excessive wavefronts.  W2 falls from 7,678,720
  (4,238,080 ideal plus 3,440,640 excessive) to exactly 4,238,080 ideal, also
  with zero excessive wavefronts.
- Formal TP4 random-route 9x200 unswizzled medians are 0.103360 / 0.162368 /
  0.258304 / 0.379264 / 0.475216 ms (geomean 0.239091 ms).  The immediately
  following swizzled medians are 0.102464 / 0.159136 / 0.254928 / 0.375200 /
  0.472880 ms (geomean 0.236349 ms).  It wins every M by
  0.49-1.99%, reduces geometric-mean latency by 1.15%, and passes every graph
  correctness check.  Accept 64-byte swizzle as the next default.
- Evidence:
  `bench/results/tp4_wgmma_graph_coldl2_random_weight_swizzle64_repeat_20260903.log`,
  `bench/results/tp4_wgmma_graph_coldl2_random_weight_swizzle0_control_repeat_20260903.log`,
  `bench/results/tp4_wgmma_m32_weight_swizzle64_random_source_coldl2_ncu.ncu-rep`,
  `bench/results/tp4_wgmma_m32_weight_swizzle64_random_source_coldl2_ncu.log`,
  `bench/results/tp4_wgmma_graph_coldl2_random_weight_swizzle0_formal_20260903.log`,
  and
  `bench/results/tp4_wgmma_graph_coldl2_random_weight_swizzle64_formal_20260903.log`.

### Iteration 16 selection — make 64-byte weight swizzle the default

- Changed the unset `V4_WEIGHT_SWIZZLE` default from 0 to 64; setting it to 0
  preserves the exact unswizzled control.  The extension key already includes
  this specialization, so the two layouts cannot share a stale JIT binary.
- Unset-default TP4 random-route cold 3x100 medians are 0.102448 / 0.159424 /
  0.252656 / 0.356240 / 0.439504 ms (geomean 0.230175 ms), all reporting
  `weight_swizzle_bytes=64` and passing graph correctness.  This agrees with
  both explicit-swizzle screens; the M64/M128 batch drift remains visible and
  is why formal per-point medians, not this short run, are retained above.
- Evidence log:
  `bench/results/tp4_wgmma_graph_coldl2_random_weight_swizzle64_default_restore_20260903.log`.

### Post-iteration-16 paired Humming gap

- Re-ran Humming and the accepted default back-to-back with the formal TP4
  random-route 9x200 cold-L2 protocol.  Both CUDA graphs contain local W13,
  activation, local W2, route reduction, and SGLang `CustomAllReduceV2`; route
  construction/alignment remains precomputed outside capture and timing.
- Humming medians for M8/M16/M32/M64/M128 are 0.090048 / 0.142880 /
  0.225120 / 0.332640 / 0.408576 ms (geomean 0.208459 ms).  Custom medians are
  0.102496 / 0.159264 / 0.253376 / 0.376032 / 0.474192 ms (geomean
  0.236349 ms).
- Custom/Humming latency ratios are 1.138x / 1.115x / 1.126x / 1.130x /
  1.161x; geometric-mean ratio 1.134x.  The accepted implementation therefore
  remains 11.5-16.1% slower pointwise and 13.38% slower geometrically.  The
  swizzle improvement is real but does not change the overall verdict: it has
  not beaten Humming.
- Evidence logs:
  `bench/results/tp4_humming_graph_coldl2_random_formal_post_weight_swizzle_window_20260903.log`
  and
  `bench/results/tp4_wgmma_graph_coldl2_random_weight_swizzle64_formal_post_humming_window_20260903.log`.

### WGMMA iteration 17 — share one swizzled weight address across row pairs

- Post-swizzle basic NCU shows W13/W2 at 160.58/87.20 us, but the direct
  swizzled address expressions raise register use from the old 45/45 to
  62/59 and reduce theoretical occupancy from 56.25% to 50%.  The conflict
  fix therefore exposed an address-generation cost.
- Observation: for the RS fragment mapping, `row1=row0+8` and each additional
  WGMMA group adds 64 rows.  Both shifts leave `(logical_row >> 1) & 3`
  unchanged.  Compute one physical address for `row0`, then use compile-time
  byte offsets for `row1` and every N64 group instead of materializing a
  separate lane-dependent address for each packed load.
- Full-block correctness passes TP4 split-K=4, TP4 forced split-K=2, and the
  TP8 local shape.  Resource inspection drops W13 to 46 registers and TP4 W2
  to 48 registers while retaining the 64-byte TMA swizzle.
- Contemporary TP4 random-route cold 3x100 medians with the common address
  are 0.100160 / 0.155856 / 0.244304 / 0.339920 / 0.425600 ms (geomean
  0.223021 ms).  The immediately following same-source direct-address control
  gives 0.102304 / 0.159296 / 0.251776 / 0.353408 / 0.434320 ms (geomean
  0.229002 ms).  The rewrite wins all five points by 2.0-3.8% and improves
  geometric mean by 2.61%.  Keep it experimental behind
  `V4_WEIGHT_COMMON_ADDRESS=1` pending reverse-order and formal confirmation.
- Evidence:
  `bench/results/tp4_wgmma_m32_weight_swizzle64_random_basic_coldl2_ncu.ncu-rep`,
  `bench/results/tp4_wgmma_m32_weight_swizzle64_random_basic_coldl2_ncu.log`,
  `bench/results/v4_flash_tp_wgmma_weight_common_address_correctness_tp4_20260903.log`,
  `bench/results/v4_flash_tp_wgmma_weight_common_address_correctness_tp4_split2_20260903.log`,
  `bench/results/v4_flash_tp_wgmma_weight_common_address_correctness_tp8shape_20260903.log`,
  `bench/results/tp4_wgmma_graph_coldl2_random_weight_common_address_screen_20260903.log`,
  and
  `bench/results/tp4_wgmma_graph_coldl2_random_weight_common_address0_control_20260903.log`.

### Iteration 17 repeat, profile, and formal confirmation

- Reverse-order 3x100 screens retain a 1.65% geometric-mean win despite a
  visibly drifting M64 batch.  The common-address version wins M8/M16/M32/M128
  by 2.3-2.8%; its M64 whole-run median is 1.9% worse because the three batch
  medians rise from 0.336 to 0.361 ms.  No individual batch was selected or
  discarded.
- Basic NCU confirms that the resource reduction becomes core time: versus
  the direct swizzled-address profile, W13 drops from 160.58 to 155.10 us
  (-3.41%) and W2 from 87.20 to 83.39 us (-4.37%).  W13/W2 register counts are
  46/48, theoretical occupancy returns to 56.25%, and achieved occupancy is
  50.95%/53.27%.
- Formal TP4 random-route 9x200 direct-address control medians are 0.102336 /
  0.159136 / 0.255840 / 0.375136 / 0.478656 ms (geomean 0.237026 ms).  The
  immediately following common-address medians are 0.100224 / 0.155616 /
  0.250688 / 0.366160 / 0.462768 ms (geomean 0.231334 ms).  It wins every M
  by 2.0-3.3%, reduces geometric-mean latency by 2.40%, and passes all graph
  correctness checks.  Accept the common-address path as the next default.
- Evidence:
  `bench/results/tp4_wgmma_graph_coldl2_random_weight_common_address0_control_repeat_20260903.log`,
  `bench/results/tp4_wgmma_graph_coldl2_random_weight_common_address_repeat_20260903.log`,
  `bench/results/tp4_wgmma_graph_coldl2_random_weight_common_address0_formal_20260903.log`,
  `bench/results/tp4_wgmma_graph_coldl2_random_weight_common_address_formal_20260903.log`,
  `bench/results/tp4_wgmma_m32_weight_common_address_random_basic_coldl2_ncu.ncu-rep`,
  and
  `bench/results/tp4_wgmma_m32_weight_common_address_random_basic_coldl2_ncu.log`.

### Iteration 17 selection — common weight address is now default

- Changed the unset `V4_WEIGHT_COMMON_ADDRESS` default from 0 to 1; explicit
  0 retains the separate-address control.  The JIT extension name includes
  the switch and cannot alias the two binaries.
- Unset-default TP4 random-route cold 3x100 medians are 0.100192 / 0.155824 /
  0.244160 / 0.338448 / 0.427904 ms (geomean 0.223047 ms), all reporting both
  64-byte weight swizzle and common addressing and passing graph correctness.
- Evidence log:
  `bench/results/tp4_wgmma_graph_coldl2_random_weight_common_address_default_20260903.log`.

### Post-iteration-17 paired Humming gap

- Re-ran Humming and the accepted common-address default back-to-back with
  the formal TP4 random-route 9x200 cold-L2 protocol.  Every individually
  timed CUDA Graph replay is preceded by a stream-ordered 256 MiB cache clear
  excluded from the CUDA-event interval; all 1,800 samples per M are retained.
- Humming medians for M8/M16/M32/M64/M128 are 0.090016 / 0.142848 /
  0.225120 / 0.325520 / 0.407456 ms (geomean 0.207421 ms).  Custom medians are
  0.100288 / 0.155648 / 0.246960 / 0.368560 / 0.433088 ms (geomean
  0.227940 ms).
- Custom/Humming latency ratios are 1.114x / 1.090x / 1.097x / 1.132x /
  1.063x; geometric-mean ratio 1.099x.  The accepted implementation remains
  6.3-13.2% slower pointwise and 9.89% slower geometrically.  Common addressing
  closes roughly 3.5 percentage points of the previous paired geometric gap,
  but the implementation still does not beat Humming.
- Evidence logs:
  `bench/results/tp4_humming_graph_coldl2_random_formal_post_common_address_window_20260903.log`
  and
  `bench/results/tp4_wgmma_graph_coldl2_random_weight_common_address_formal_post_humming_window_20260903.log`.

### WGMMA iteration 18 — dequant selector instruction sweep (rejected)

- Added compile-time switches for the two selector-generation halves of
  `dequant_mxfp4_to_fp8_pair_with_lut`: DP4A or shift/add/PRMT independently
  for high and low nibbles.  The extension key and benchmark metadata include
  both switches; the unset default remains the pre-existing DP4A/DP4A path.
- All three alternative combinations pass full-path correctness for TP4 W13
  split-K=4, TP4 forced split-K=2, and the TP8 local shape.  Their W13,
  activation, and W2 errors are identical to the accepted default.
- First TP4 random-route cold 3x100 sweep geometric means are 0.224909 ms for
  DP4A/DP4A, 0.228719 ms for DP4A/PRMT, 0.227620 ms for PRMT/DP4A, and
  0.236553 ms for PRMT/PRMT.  The existing path wins all five M points against
  every alternative.
- Reverse-order comparison of the nearest candidate gives 0.227508 ms for
  PRMT/DP4A and 0.225157 ms for the immediately following DP4A/DP4A control:
  the candidate remains 1.04% slower geometrically.  It is slower at four of
  five points; the 0.05% M64 reversal is below observed batch drift.  Reject
  all selector alternatives and retain DP4A/DP4A.
- Every timed replay in every sweep is preceded by the standard 256 MiB L2
  clear outside the CUDA-event interval.
- Evidence:
  `bench/results/v4_flash_tp_wgmma_dequant10_correctness_20260903.log`,
  `bench/results/v4_flash_tp_wgmma_dequant01_00_correctness_20260903.log`,
  `bench/results/tp4_wgmma_graph_coldl2_random_dequant11_screen_20260903.log`,
  `bench/results/tp4_wgmma_graph_coldl2_random_dequant10_screen_20260903.log`,
  `bench/results/tp4_wgmma_graph_coldl2_random_dequant01_screen_20260903.log`,
  `bench/results/tp4_wgmma_graph_coldl2_random_dequant00_screen_20260903.log`,
  `bench/results/tp4_wgmma_graph_coldl2_random_dequant01_reverse_20260903.log`,
  and
  `bench/results/tp4_wgmma_graph_coldl2_random_dequant11_control_reverse_20260903.log`.

### Post-iteration-18 detailed cold profile

- Profiled one M32 random-route local pipeline after a 256 MiB L2 clear and
  collected SpeedOfLight, memory, scheduler, warp-state, source-counter, and
  instruction sections for both accepted route GEMMs.
- W13/W2 durations are 155.04/83.81 us.  Compute throughput is 76.82%/74.36%,
  DRAM throughput is 44.37%/39.68%, and schedulers have no eligible warp for
  20.77%/22.53% of active cycles.
- Of not-issued warp samples, W13 is 33.39% long-scoreboard, 25.04% barrier,
  and 19.08% wait; W2 is 47.91% long-scoreboard, 15.70% barrier, and 14.07%
  wait.  This motivates an isolated deeper-TMA-pipeline experiment while also
  checking its shared-memory occupancy cost.
- Source counters find no residual shared-memory conflicts.  The only notable
  uncoalesced global load is the predicated eight-route FP32 activation-scale
  gather (`__ldg`), with NCU estimating only a 2-3% kernel-level upper bound;
  this cannot by itself close the remaining Humming gap.
- Evidence:
  `bench/results/tp4_wgmma_m32_common_address_random_detailed_coldl2_ncu.ncu-rep`
  and
  `bench/results/tp4_wgmma_m32_common_address_random_detailed_coldl2_ncu.log`.

### WGMMA iteration 19 — deeper packed-weight TMA pipeline (rejected)

- Added a compile-time `V4_WEIGHT_STAGES` specialization for two, three, or
  four packed-weight TMA buffers.  Barrier arrays, prefetch distance, dynamic
  shared-memory sizing, extension keys, profiler metadata, and benchmark
  metadata all follow the selected depth.  The default remains two stages.
- Three- and four-stage variants pass TP4 split-K=4, TP4 forced split-K=2,
  and TP8-local-shape full-path correctness with the same errors as the
  accepted two-stage path.
- TP4 random-route cold 3x100 medians for M8/M16/M32/M64/M128 are:
  - 2-stage control: 0.100256 / 0.156064 / 0.244416 / 0.351376 /
    0.426976 ms; geomean 0.224773 ms.
  - 3 stages: 0.108064 / 0.171664 / 0.268224 / 0.373376 / 0.465392 ms;
    geomean 0.243986 ms.
  - 4 stages: 0.111488 / 0.178144 / 0.279872 / 0.392736 / 0.487024 ms;
    geomean 0.254286 ms.
- Three and four stages lose at every M and regress geometric mean by 8.55%
  and 13.13%.  The additional shared-memory residency cost dominates any
  reduction in TMA wait time.  Reject both and retain two stages; the margins
  are too large and uniform to justify a reverse-order expansion.
- Every timed replay is preceded by the standard 256 MiB L2 clear outside the
  CUDA-event interval.
- Evidence:
  `bench/results/v4_flash_tp_wgmma_weight_stages3_4_correctness_20260903.log`,
  `bench/results/tp4_wgmma_graph_coldl2_random_weight_stages2_control_20260903.log`,
  `bench/results/tp4_wgmma_graph_coldl2_random_weight_stages3_screen_20260903.log`,
  and
  `bench/results/tp4_wgmma_graph_coldl2_random_weight_stages4_screen_20260903.log`.

### WGMMA iteration 20 — register-synthesized E2M1/E8M0 LUT probe (rejected)

- Added an explicit experimental specialization that synthesizes the two
  packed positive E4M3 lookup words from the E8M0 exponent in registers and
  removes the per-CTA 2 KiB shared LUT.  This affine expression is bit-exact
  for the benchmark/test exponent range 125..128; arbitrary production E8M0
  would require Humming-style offline scale normalization, so the default
  remains the full 256-row LUT regardless of timing.
- The probe passes TP4 W13 split-K=4, TP4 forced split-K=2, and TP8 local-shape
  full-path correctness with errors identical to the accepted LUT path.
- TP4 random-route cold 3x100 medians for the shared-LUT control are 0.100896 /
  0.156816 / 0.245360 / 0.347568 / 0.433248 ms (geomean 0.225615 ms).
  Register synthesis gives 0.099968 / 0.156416 / 0.247488 / 0.349008 /
  0.432016 ms (geomean 0.225531 ms).
- The 0.04% geometric difference is noise-sized and pointwise mixed: the
  probe wins M8/M16/M128 but loses M32/M64.  Extra integer synthesis offsets
  the LUT initialization/shared-residency saving.  Reject it without a larger
  reverse-order run and retain the general 256-row LUT default.
- Every timed replay is preceded by the standard 256 MiB L2 clear outside the
  CUDA-event interval.
- Evidence:
  `bench/results/v4_flash_tp_wgmma_dequant_synth_lut_probe_correctness_20260903.log`,
  `bench/results/tp4_wgmma_graph_coldl2_random_dequant_synth_lut0_control_20260903.log`,
  and
  `bench/results/tp4_wgmma_graph_coldl2_random_dequant_synth_lut_probe_screen_20260903.log`.

### WGMMA iteration 21 — offline Mode2 braid removes selector DP4A (candidate)

- Adapted DeepGEMM MegaMoE's Mode2 sign/magnitude braid to this TP kernel's
  unchanged row-major packed-weight tensor.  A new extension kernel rewrites
  every 32-bit packed word in place once at weight initialization; this is
  outside graph capture and inference timing and does not depend on Humming.
- The equivalent runtime decoder selects the two four-value magnitude groups
  directly with two PRMT instructions.  It removes the accepted decoder's
  four selector-building DP4A instructions per pair while preserving the same
  LUT, TMA tensor shape, bytes read, 64-byte swizzle, and WGMMA operand order.
- Independent torch-reference correctness uses an untouched copy of the
  ordinary braided weights.  TP4 split-K=4, TP4 forced split-K=2, and TP8
  local-shape tests all pass with exactly the prior W13/activation/W2 errors.
- TP4 random-route cold 3x100 ordinary-layout control medians are 0.100672 /
  0.156704 / 0.245376 / 0.348624 / 0.427552 ms (geomean 0.225026 ms).
  Mode2 medians are 0.093280 / 0.142464 / 0.221920 / 0.310720 /
  0.389152 ms (geomean 0.204379 ms).
- Mode2 wins every M by 7.3-10.9%, reducing geometric-mean latency by 9.18%
  (1.101x speedup).  Keep it as a candidate pending formal 9x200 confirmation,
  an immediate paired Humming run, and core/resource profiling; do not change
  the unset default yet.
- Every timed replay is preceded by the standard 256 MiB L2 clear outside the
  CUDA-event interval.
- Evidence:
  `bench/results/v4_flash_tp_wgmma_mode2_braid_correctness_20260903.log`,
  `bench/results/tp4_wgmma_graph_coldl2_random_mode2_braid0_control_20260903.log`,
  and
  `bench/results/tp4_wgmma_graph_coldl2_random_mode2_braid_screen_20260903.log`.

### WGMMA iteration 21 formal confirmation and paired Humming comparison

- Formal TP4 random-route 9x200 cold-L2 Mode2 medians for
  M8/M16/M32/M64/M128 are 0.093600 / 0.141952 / 0.224912 / 0.338576 /
  0.426752 ms (geomean 0.212350 ms).  All 9,000 max-rank graph samples use a
  stream-ordered 256 MiB clear outside the event interval and all correctness
  checks pass.  M32 and the two larger cases show visible batch drift, so the
  isolated run is not used alone to claim a win.
- In the immediately paired Humming-then-Mode2 window, Humming medians are
  0.089888 / 0.142656 / 0.225152 / 0.330480 / 0.411296 ms (geomean
  0.208331 ms), while Mode2 gives 0.093152 / 0.141600 / 0.224384 / 0.327568 /
  0.420784 ms (geomean 0.209953 ms).  Mode2 is 0.78% slower geometrically:
  it wins M16/M32/M64, but loses M8/M128.
- Reversing launch order gives Mode2 0.093280 / 0.141792 / 0.223872 /
  0.338928 / 0.424288 ms (geomean 0.211759 ms), followed by Humming at
  0.089792 / 0.142688 / 0.225216 / 0.332512 / 0.412992 ms (geomean
  0.208735 ms).  Mode2 is 1.45% slower geometrically.  M16/M32 still win,
  M8 still loses by 3.9%, and M64/M128 move with the system drift.
- A detailed one-replay cold NCU profile measures Mode2 W13/W2 at
  138.144/74.112 us, down 10.9%/11.6% from the pre-Mode2
  155.040/83.808 us.  Compute throughput is 65.39% for both kernels, DRAM is
  49.70%/44.85%, and achieved occupancy is 50.73%/53.26%.  Against the most
  recent comparable Humming route-GEMM profile (146.82/74.59 us), the custom
  cores are already about 5.9%/0.6% faster; the residual graph gap is therefore
  in split reduction/activation quantization, epilogue, communication, and
  launch overhead rather than either route-GEMM core.
- An attempted Nsight Systems capture deadlocked in the profiler control layer
  (both host processes slept in futex, GPU utilization was zero, and no report
  was emitted), so it was terminated without using its empty log as evidence.
- The formal confirmation preserves the candidate decision: Mode2 is a large,
  correctness-clean improvement over the ordinary layout, but the honest
  end-to-end conclusion is near parity, not yet a Humming win.
- Evidence:
  `bench/results/tp4_wgmma_graph_coldl2_random_mode2_braid_formal_20260903.log`,
  `bench/results/tp4_humming_graph_coldl2_random_formal_post_mode2_window_20260903.log`,
  `bench/results/tp4_wgmma_graph_coldl2_random_mode2_braid_formal_post_humming_window_20260903.log`,
  `bench/results/tp4_wgmma_graph_coldl2_random_mode2_braid_formal_reverse_window_20260903.log`,
  `bench/results/tp4_humming_graph_coldl2_random_formal_post_mode2_reverse_window_20260903.log`,
  `bench/results/tp4_wgmma_m32_mode2_braid_random_detailed_coldl2_ncu.ncu-rep`,
  and
  `bench/results/tp4_wgmma_m32_mode2_braid_random_detailed_coldl2_ncu.log`.

### WGMMA iteration 21 selection — Mode2 braid is now default

- Changed the unset `V4_MODE2_BRAID` default from 0 to 1.  Explicit
  `V4_MODE2_BRAID=0` remains the ordinary-layout control, and both the offline
  weight conversion and JIT extension key continue to follow the switch.
- Unset-default full-path correctness passes TP4 split-K=4, TP4 forced
  split-K=2, and TP8 local shapes.  W13 cosine is at least 0.999999997,
  activation cosine at least 0.999999691, and W2 cosine at least 0.999997241.
- Unset-default TP4 random-route cold 3x100 medians for
  M8/M16/M32/M64/M128 are 0.093120 / 0.142336 / 0.221552 / 0.313312 /
  0.390416 ms (geomean 0.204676 ms).  Metadata reports `mode2_braid=true`,
  all graph/all-reduce checks pass, and every sample uses the standard 256 MiB
  pre-replay L2 clear outside the timing events.
- Evidence:
  `bench/results/v4_flash_tp_wgmma_mode2_default_correctness_20260903.log`
  and
  `bench/results/tp4_wgmma_graph_coldl2_random_mode2_default_20260903.log`.

### WGMMA iteration 22 — retune Mode2 W13 split threshold (accepted)

- Re-swept the existing split-K=2 and split-K=4 implementations after making
  Mode2 the default.  Forced split-2 medians for M8/M16/M32/M64/M128 are
  0.095424 / 0.145632 / 0.221792 / 0.308976 / 0.387744 ms (geomean
  0.205810 ms); forced split-4 gives 0.093024 / 0.141824 / 0.218128 /
  0.312688 / 0.391456 ms (geomean 0.203878 ms).  Thus split-4 remains best at
  M8/M16, split-2 remains best at M64/M128, and M32 has moved to split-4.
- A reverse-order M32 5x200 cold pair confirms split-4 at 0.216896 ms versus
  the immediately following split-2 control at 0.218784 ms, a 0.86% latency
  reduction across 1,000 samples per variant.  Both pass graph correctness.
- Moved the route-count branch from `routed_rows <= 96` to
  `routed_rows <= 192`; the active-expert fallback remains 96 for larger route
  counts.  The new unset-default screen selects 4/4/4/2/2 and gives
  0.093264 / 0.142496 / 0.218816 / 0.307152 / 0.389216 ms (geomean
  0.203342 ms), with every graph/all-reduce check passing.  Independent M32
  full-path correctness gives W13/activation/W2 cosine
  0.999999997/0.999999691/0.999997241.
- Every timing sample uses the standard 256 MiB stream-ordered cold-L2 clear
  outside the event interval.
- Evidence:
  `bench/results/tp4_wgmma_graph_coldl2_random_mode2_forced_s2_screen_20260903.log`,
  `bench/results/tp4_wgmma_graph_coldl2_random_mode2_forced_s4_screen_20260903.log`,
  `bench/results/tp4_wgmma_graph_coldl2_random_mode2_m32_s4_confirm_20260903.log`,
  `bench/results/tp4_wgmma_graph_coldl2_random_mode2_m32_s2_reverse_control_20260903.log`,
  `bench/results/v4_flash_tp_wgmma_mode2_split_threshold192_correctness_20260903.log`,
  and
  `bench/results/tp4_wgmma_graph_coldl2_random_mode2_split_threshold192_default_20260903.log`.

### WGMMA iteration 23 — fuse split reduction, SwiGLU, and FP8 quantization (candidate)

- Added an opt-in `V4_FUSED_ACT_QUANT=1` path that replaces the separate
  split-K reduction/SwiGLU CUDA kernel plus Humming group-128 quantization
  kernel with one 128-thread CUDA block per route/group.  It preserves both
  BF16 roundings used by the public path, computes the same max/448 FP32
  scale, emits E4M3 with saturating round-to-nearest conversion, and skips
  materializing the intermediate BF16 activation in the benchmark.
- The original two-kernel path remains under `V4_FUSED_ACT_QUANT=0` and is
  still the unset default for this candidate commit.  The extension exposes
  both paths from the same binary, so A/B runs cannot accidentally compare
  different route-GEMM builds.
- Full-path correctness passes TP4 split-K=4, TP4 split-K=2, and TP8 local
  shapes with errors identical to the original path: W13 cosine is at least
  0.999999997, activation cosine at least 0.999999666, and W2 cosine at least
  0.999997243.
- In the control-then-candidate TP4 random-route cold 3x100 pair, control
  medians are 0.093024 / 0.141984 / 0.218464 / 0.315744 / 0.390304 ms
  (geomean 0.204263 ms), and fused medians are 0.092320 / 0.140992 /
  0.217696 / 0.307072 / 0.386144 ms (geomean 0.201960 ms).  Fusion wins all
  five points and improves geometric mean by 1.13%.
- The reverse candidate-then-control pair gives fused 0.092224 / 0.140864 /
  0.216768 / 0.309216 / 0.388432 ms (geomean 0.202229 ms), versus control
  0.093024 / 0.142016 / 0.218704 / 0.313296 / 0.389264 ms (geomean
  0.203891 ms).  Fusion again wins every M and improves geometric mean by
  0.82%; keep it as an accepted candidate pending the unset-default check.
- Every sample uses the standard 256 MiB stream-ordered cold-L2 clear outside
  the event interval.
- Evidence:
  `bench/results/v4_flash_tp_wgmma_fused_act_quant_correctness_20260903.log`,
  `bench/results/tp4_wgmma_graph_coldl2_random_fused_act_quant0_control_20260903.log`,
  `bench/results/tp4_wgmma_graph_coldl2_random_fused_act_quant1_screen_20260903.log`,
  `bench/results/tp4_wgmma_graph_coldl2_random_fused_act_quant1_reverse_20260903.log`,
  and
  `bench/results/tp4_wgmma_graph_coldl2_random_fused_act_quant0_reverse_control_20260903.log`.

### WGMMA iteration 23 selection — fused activation quantization is now default

- Changed the unset `V4_FUSED_ACT_QUANT` default from 0 to 1; explicit 0 keeps
  the fully independent legacy reduction plus Humming quantizer control.
- Unset-default correctness passes both TP4 and TP8 local shapes with the same
  W13/activation/W2 errors as the candidate tests.
- Unset-default TP4 random-route cold 3x100 medians for
  M8/M16/M32/M64/M128 are 0.092304 / 0.141312 / 0.217728 / 0.306624 /
  0.388368 ms (geomean 0.202224 ms).  Metadata reports the fused path, all
  graph/all-reduce checks pass, and every sample uses a 256 MiB pre-replay L2
  clear outside the event interval.
- Evidence:
  `bench/results/v4_flash_tp_wgmma_fused_act_quant_default_correctness_20260903.log`
  and
  `bench/results/tp4_wgmma_graph_coldl2_random_fused_act_quant_default_20260903.log`.

### Post-iteration-23 formal comparison and interleaved-pair audit

- Two conventional whole-run 9x200 cold pairs disagree because the H20 clocks
  drift during the longer M32/M64/M128 batches.  Humming-then-custom gives
  geomeans 0.208331/0.211112 ms (custom 1.34% slower); reversing the order
  gives custom/Humming 0.206800/0.208643 ms (custom 0.88% faster).  The batch
  sequences, including M32 moving from roughly 0.217 to 0.258 ms, are retained
  rather than hiding the instability with minimum latency.
- Added a paired CUDA-Graph harness that captures both implementations in one
  process, uses the same static routes, token input, and
  `CustomAllReduceV2` instance, gives each replay its own 256 MiB L2 clear,
  and alternates AB/BA order at replay granularity.  Both graph outputs pass
  their independent local-recompute plus NCCL correctness checks.
- Its formal 9x200-per-implementation result is Humming 0.090368 / 0.146016 /
  0.237824 / 0.351104 / 0.426816 ms (geomean 0.216008 ms), versus custom
  0.092800 / 0.143808 / 0.226640 / 0.329200 / 0.405232 ms (geomean
  0.209491 ms).  Custom loses M8 by 2.69%, wins M16/M32/M64/M128 by
  1.54%/4.93%/6.65%/5.33%, and wins geometric mean by 3.11%.
- This replay-granularity result is a useful drift-controlled stress test but
  is not declared the sole source of truth: alternating two disjoint 0.86 GB
  weight sets can perturb TLB residency beyond an isolated serving replay.
  A batch-granularity paired run is required for the final headline.
- Cold M8 Nsight Systems timelines isolate the residual small-M loss.  Custom
  versus Humming kernel durations are W13 48.895/48.831 us, activation path
  2.048/(1.248+1.216) us, W2 27.040/22.144 us, and epilogue
  1.632/2.624 us.  Thus W13 is tied and fusion is working; the approximately
  4.9 us W2 deficit is the only material M8 core difference.
- Evidence:
  `bench/results/tp4_humming_graph_coldl2_random_formal_post_fused_act_quant_window_20260903.log`,
  `bench/results/tp4_wgmma_graph_coldl2_random_fused_act_quant_formal_post_humming_window_20260903.log`,
  `bench/results/tp4_wgmma_graph_coldl2_random_fused_act_quant_formal_reverse_window_20260903.log`,
  `bench/results/tp4_humming_graph_coldl2_random_formal_post_fused_act_quant_reverse_window_20260903.log`,
  `bench/results/tp4_paired_graph_coldl2_random_fused_act_quant_formal_20260903.log`,
  `bench/results/tp4_wgmma_m8_fused_act_quant_random_coldl2_nsys.nsys-rep`,
  and
  `bench/results/tp4_humming_m8_random_refresh_coldl2_nsys.nsys-rep`.

### WGMMA iteration 24 — naive persistent W2 grid (rejected)

- Added an opt-in runtime W2 grid cap and a grid-stride tile loop so each CTA
  can reuse its 2 KiB MXFP4 LUT across tasks.  W13 is unchanged.  The unset
  `V4_W2_PERSISTENT_BLOCKS_PER_SM=0` path retains one CTA per logical tile.
- The 4-blocks/SM candidate passes TP4 split-K=4, TP4 split-K=2, and TP8 local
  full-path correctness with errors identical to the default.
- TP4 random-route cold 3x100 geomeans for 0/2/4/6/8 blocks per SM are
  0.204139 / 0.273953 / 0.221692 / 0.209585 / 0.207606 ms.  Every persistent
  setting loses to the 0 control, by 34.20% / 8.60% / 2.67% / 1.70%; each is
  slower at all five M values.
- Merely copying Humming's 312-CTA grid is insufficient.  The custom loop
  serializes too many independently pipelined K512 tiles and reinitializes TMA
  barriers per task; Humming's persistent CTA has a different internal
  scheduler/pipeline.  Reject all caps and retain 0.
- Every timing sample uses the standard 256 MiB cold-L2 clear outside events.
- Evidence:
  `bench/results/v4_flash_tp_wgmma_w2_persistent4_correctness_20260903.log`
  and
  `bench/results/tp4_wgmma_graph_coldl2_random_w2_persistent{0,2,4,6,8}_screen_20260903.log`.

### Iteration 24 rollback — restore one-task W2 CTAs

- Removed the rejected grid-stride loop and runtime grid-cap plumbing instead
  of merely leaving the cap at zero.  This restores the pre-experiment
  one-logical-tile-per-CTA kernel and avoids a generic loop branch on the
  accepted path; the only unrelated retained change is fused activation
  quantization from iteration 23.
- TP4 local full-path correctness passes with W13/activation/W2 cosine
  0.999999998/0.999999759/0.999997256.
- The required TP4 random-route cold 3x100 rollback screen gives
  0.092384 / 0.141088 / 0.217472 / 0.315632 / 0.387536 ms for
  M8/M16/M32/M64/M128 (geomean 0.203234 ms).  This recovers the persistent
  experiment's 0-control geomean within 0.45% and is 2.15% faster than the
  best persistent setting (8 blocks/SM).  All graph/all-reduce checks pass.
- Evidence:
  `bench/results/v4_flash_tp_wgmma_w2_persistent_rollback_correctness_20260903.log`
  and
  `bench/results/tp4_wgmma_graph_coldl2_random_w2_persistent_rollback_20260903.log`.

### WGMMA iteration 25 — direct global W2 LUT (rejected)

- Added an opt-in `V4_W2_GLOBAL_LUT=1` specialization that skips the per-CTA
  shared LUT copy only for W2 and loads the same complete 256-row `uint2` LUT
  through the read-only global path during dequantization.  W13 is unchanged,
  arbitrary E8M0 codes remain supported, and the unset default stays 0.
- Full-path correctness passes TP4 split-K=4, TP4 split-K=2, and TP8 local
  shapes with errors identical to the shared-LUT path.
- In the TP4 random-route cold 3x100 control-then-candidate pair, shared-LUT
  medians are 0.092288 / 0.140832 / 0.217312 / 0.306256 / 0.385136 ms
  (geomean 0.201616 ms), while direct-global gives 0.093888 / 0.143776 /
  0.221024 / 0.312704 / 0.395376 ms (geomean 0.205767 ms).
- The candidate loses all five M values and regresses geometric mean by 2.06%.
  Saving LUT initialization cannot offset the global load dependency in every
  dequantization step; retain shared LUT as default without a reverse run.
- Every timing sample uses the standard 256 MiB cold-L2 clear outside events.
- Evidence:
  `bench/results/v4_flash_tp_wgmma_w2_global_lut_correctness_20260903.log`,
  `bench/results/tp4_wgmma_graph_coldl2_random_w2_global_lut0_control_20260903.log`,
  and
  `bench/results/tp4_wgmma_graph_coldl2_random_w2_global_lut1_screen_20260903.log`.

### Batch-granularity paired cold-L2 source of truth

- Extended `bench/v4_flash_tp_paired_graph.py` with
  `--pair-granularity {batch,replay}`.  Batch mode alternates complete timed
  outer batches in AB/BA order while retaining a separate 256 MiB
  stream-ordered L2 clear before every graph replay.  This avoids the extra
  TLB churn caused by switching between two disjoint approximately 0.86 GB
  weight sets on every replay, while still balancing clock drift between the
  two implementations.
- The formal random-route run uses ten outer batches of 200 cold replays per
  implementation and M (2,000 samples each), with five Humming-first and five
  custom-first batches.  All local and all-reduce graph correctness checks
  pass.
- Humming medians for M8/M16/M32/M64/M128 are 0.089920 / 0.145472 /
  0.232800 / 0.334576 / 0.410272 ms (geomean 0.210978 ms).  Custom medians are
  0.092416 / 0.143488 / 0.229232 / 0.337440 / 0.418880 ms (geomean
  0.212141 ms).
- Custom is 2.78% slower at M8, 1.38% faster at M16, 1.56% faster at M32,
  0.86% slower at M64, and 2.10% slower at M128.  Its five-shape geometric
  mean is 0.55% slower than Humming.  Therefore the current honest headline
  is statistical near parity with a slight Humming advantage, not a custom
  win; the earlier replay-granularity 3.11% result remains a TLB-stress
  sensitivity result rather than the primary comparison.
- Evidence:
  `bench/results/tp4_paired_graph_batch_coldl2_random_smoke_20260903.log`
  and
  `bench/results/tp4_paired_graph_batch_coldl2_random_fused_act_quant_formal_20260903.log`.

### Route-distribution sensitivity audit

- Ran the accepted implementation and Humming in the same batch-paired TP4
  CUDA-Graph harness for deterministic balanced and maximal-skew routes.  The
  screen uses four AB/BA-balanced outer batches of 100 replays, so each
  implementation receives 400 samples per M; every replay has its own 256 MiB
  L2 clear outside the measured interval.
- Balanced routes activate 48/96/192/256/256 experts at
  M8/M16/M32/M64/M128.  Humming medians are 0.096768 / 0.163712 / 0.301296 /
  0.401408 / 0.420640 ms and custom medians are 0.099232 / 0.159808 /
  0.287232 / 0.392368 / 0.425472 ms.  Custom loses M8/M128 by 2.55%/1.15%,
  wins M16/M32/M64 by 2.44%/4.90%/2.30%, and is 1.17% faster in five-shape
  geometric mean.
- Maximal skew activates only six experts at every M.  Humming medians are
  0.039328 / 0.049216 / 0.067968 / 0.101600 / 0.168928 ms and custom medians
  are 0.042496 / 0.053344 / 0.072128 / 0.101536 / 0.168672 ms.  Custom loses
  M8/M16/M32 by 8.06%/8.39%/6.12% and is within 0.2% at M64/M128; its
  geometric mean is 4.40% slower.
- The result disproves any route-independent speedup claim.  The custom
  schedule is competitive when many experts expose enough independent tiles,
  but its fixed overhead/parallelism is inferior to Humming for a few active
  experts and small routed-row counts.  All graph and all-reduce checks pass;
  Humming's skew-route cosine is lower (minimum about 0.99987) but remains
  finite and within the harness's reference tolerance.
- Evidence:
  `bench/results/tp4_paired_graph_batch_coldl2_balanced_fused_act_quant_screen_20260903.log`
  and
  `bench/results/tp4_paired_graph_batch_coldl2_skew_fused_act_quant_screen_20260903.log`.

### WGMMA iteration 26 — global 64-channel tile for skew routes (rejected)

- Tested the existing `V4_WOUT=64` specialization on maximal-skew TP4 routes
  to determine whether doubling the number of independent CTAs fixes the
  six-active-expert small-M deficit.  Both W13 and W2 use the smaller tile in
  this diagnostic; the accepted default remains 128.
- In a batch-paired cold-L2 4x100 screen, custom M8 improves from the preceding
  0.042496 ms to 0.041632 ms (about 2.0%), but M16/M32/M64/M128 become
  0.055328 / 0.073824 / 0.114976 / 0.193664 ms versus the 128-channel
  control's 0.053344 / 0.072128 / 0.101536 / 0.168672 ms.  Against the paired
  Humming samples, the 64-channel custom path is 5.77% / 12.13% / 8.41% /
  13.17% / 14.88% slower and loses five-shape geometric mean by 10.82%.
- The small M8 gain confirms a parallelism corner, but a global tile-size
  change is decisively wrong because duplicated CTA setup and activation
  traffic dominate as routed rows rise.  Reject the global variant; any
  follow-up must isolate W13 from W2 and select only the narrow corner.
- All graph and all-reduce correctness checks pass, and all 400 samples per
  implementation and M use a 256 MiB pre-replay L2 clear outside timing.
- Evidence:
  `bench/results/tp4_paired_graph_batch_coldl2_skew_wout64_screen_20260903.log`.

### Skew-M8 layer diagnosis

- Cold Nsight Systems replays show that the 128-channel custom path spends
  16.448 us in W13 and 9.728 us in W2, versus Humming's 15.168 us and
  6.336 us.  The custom fused activation/quant kernel takes 1.792 us versus
  Humming's 1.376 us SwiGLU plus 2.592 us across its two quant kernels, so the
  fused middle stage saves about 2.18 us; W2 is the dominant remaining gap.
- The 64-channel diagnostic changes custom W13/W2 to 15.776/9.088 us, showing
  that both layers contribute roughly half of its small M8-only gain.  Forcing
  W13 split-K=2 instead makes W13 23.232 us, decisively validating split-K=4
  for this low-parallelism shape.
- A focused 400-sample test of the previously rejected direct-global W2 LUT
  gives 0.042464 ms versus the shared-LUT audit's 0.042496 ms, indistinguishable
  at this noise level and still 8.33% behind its paired Humming sample.  There
  is no skew-only reversal to justify that path.
- Evidence:
  `bench/results/tp4_{wgmma_m8_skew_wout128,humming_m8_skew,wgmma_m8_skew_wout64,wgmma_m8_skew_split2}_coldl2_nsys.{log,nsys-rep}`
  and
  `bench/results/tp4_paired_graph_batch_coldl2_skew_m8_w2_global_lut_screen_20260903.log`.

### Random-M8 64-channel layer audit

- A cold local trace closes the possibility that only W2 wants the narrower
  output tile on the primary random-route workload.  With 64 channels, W13
  and W2 take 52.192/30.464 us, versus 48.895/27.040 us for the accepted
  128-channel trace.  Both layers regress by about 3.3 us, so a W2-only
  64-channel specialization would add complexity without a primary-workload
  benefit.
- Evidence:
  `bench/results/tp4_wgmma_m8_random_wout64_coldl2_nsys.{log,nsys-rep}`.

### TP8 distributed CUDA-Graph run-through

- With all eight H20s idle, ran the accepted kernel at the real TP8 shard
  (`I/rank=256`, W13 `[256,512,4096]`, W2 `[256,4096,256]`) through the full
  eight-rank graph including SGLang `CustomAllReduceV2`, rather than relying
  only on a single-GPU TP8-shape unit test.
- The deliberately small M8 random-route smoke uses two outer batches of ten
  individually cold replays.  Median max-rank latency is 0.072816 ms; minimum
  rank cosine is 0.999991970, relative L2 is 0.0040074, all values are finite,
  and the custom all-reduce output matches an independent local recompute plus
  NCCL reference.
- This 20-sample result proves TP8 graph/run-time compatibility only and is
  not promoted to a performance headline.  Each replay uses the standard
  256 MiB L2 clear outside timing.  Evidence:
  `bench/results/tp8_wgmma_graph_coldl2_random_m8_smoke_20260903.log`.

### WGMMA iteration 27 — two concurrent W2 warp-groups per CTA (rejected)

- Added an opt-in `V4_W2_DUAL_TASK_CTA=1` specialization.  A 256-thread CTA
  runs two independent 128-thread WGMMA groups concurrently, with separate
  TMA stages/barriers/activation storage and one shared E8M0 LUT.  The grid is
  halved without serially walking tasks, so the design remains independent of
  the runtime expert distribution and directly tests whether block/LUT setup
  can be amortized safely.
- The first TP8-shape check correctly failed with non-finite W2 output: the
  K=256 scalar scale-loader retained `blockDim.x` as its stride and each
  warp-group therefore skipped half its scale entries.  Changing that loop to
  the warp-group width of 128 and forcing a fresh extension fixed the bug.
  Final TP4 skew, TP4 balanced, and TP8-shape checks all pass; worst final W2
  cosine is 0.999997235 and all outputs are finite.
- The batch-paired TP4 random-route cold 4x100 screen gives custom medians
  0.097536 / 0.152448 / 0.232448 / 0.323680 / 0.446416 ms, versus Humming
  0.089856 / 0.145872 / 0.225312 / 0.316608 / 0.421648 ms.  The candidate
  loses every M by 2.23-8.55% and loses geometric mean by 4.84%.
- It also fails its intended M8 maximal-skew corner: 0.042768 ms versus
  Humming 0.039456 ms (8.39% slower), and is slightly slower than the
  one-warp-group custom control's 0.042496 ms.  Doubling CTA width couples the
  groups at block-wide barriers and halves CTA-level scheduling flexibility;
  LUT amortization cannot recover those costs.  Reject the candidate and keep
  `V4_W2_DUAL_TASK_CTA=0` pending an exact rollback of generic-kernel changes.
- Every performance sample uses a separate 256 MiB pre-replay L2 clear outside
  the event interval.  Evidence:
  `bench/results/v4_flash_tp_wgmma_w2_dual_task_skew_m8_correctness_20260903.log`,
  `bench/results/v4_flash_tp_wgmma_w2_dual_task_tp8shape_correctness_20260903.log`,
  `bench/results/v4_flash_tp_wgmma_w2_dual_task_tp8shape_correctness_fixed_20260903.log`,
  `bench/results/v4_flash_tp_wgmma_w2_dual_task_tp4_{skew,balanced}_correctness_fixed_20260903.log`,
  `bench/results/tp4_paired_graph_batch_coldl2_random_w2_dual_task_screen_20260903.log`,
  and
  `bench/results/tp4_paired_graph_batch_coldl2_skew_m8_w2_dual_task_screen_20260903.log`.

### Iteration 27 rollback — restore the single-warp-group W2 winner

- Removed the rejected dual-task specialization and all generic-kernel
  indexing changes, rather than only disabling its environment switch.  A
  direct diff against commit `4c1f891` confirms that the five source/harness
  files are identical to the pre-experiment winner except for the JIT suffix
  (`v27`), which forces a clean rebuild.
- The rebuilt TP4 balanced full-path check passes with W13/activation/W2
  cosine 0.999999997/0.999999691/0.999997241 and finite output.
- A batch-paired random-route cold 4x100 rollback screen gives custom medians
  0.092736 / 0.143488 / 0.217632 / 0.310976 / 0.446352 ms, versus Humming
  0.090528 / 0.145760 / 0.226432 / 0.326160 / 0.416592 ms.  The batch-level
  drift is again large at M128, so this short run is not a new headline; its
  five-shape geomeans are 0.209333/0.209749 ms (custom 0.20% faster), which
  confirms recovery from the dual-task candidate's 4.84% loss.
- Every one of the 400 samples per implementation and M has its own 256 MiB
  pre-replay L2 clear outside timing.  Evidence:
  `bench/results/v4_flash_tp_wgmma_w2_dual_task_rollback_correctness_20260903.log`
  and
  `bench/results/tp4_paired_graph_batch_coldl2_random_w2_dual_task_rollback_screen_20260903.log`.

### WGMMA iteration 28 — W2-only three-stage TMA pipeline (rejected)

- Added an isolated `V4_W2_WEIGHT_STAGES` specialization so W13 can retain
  its accepted two-stage pipeline while only the short-K W2 tests a third
  packed-weight buffer.  This separates the W2 question from iteration 19's
  global three-stage regression and directly targets W2's 47.9% long-
  scoreboard share in the detailed NCU sample.
- W2 stage count defaults to `V4_WEIGHT_STAGES` for backward-compatible
  diagnostics and is two when all tuning variables are unset.  Explicit
  three-stage TP4 and TP8-shape full-path checks pass with errors identical to
  the two-stage path; worst W2 cosine is 0.999997249 and outputs are finite.
- The batch-paired TP4 random-route cold 4x100 screen gives three-stage custom
  medians 0.097472 / 0.152224 / 0.231552 / 0.323264 / 0.455424 ms.  Against
  the immediately preceding two-stage rollback screen's custom medians, these
  are slower by about 5.11% / 6.09% / 6.40% / 3.95% / 2.03%; all five points
  lose despite run-to-run drift at M128.
- Its paired Humming medians are 0.090496 / 0.145728 / 0.225680 / 0.317632 /
  0.422752 ms, so three-stage custom loses every M and five-shape geometric
  mean by 4.82%.  The extra 8 KiB packed-weight buffer reduces resident CTA
  capacity more than its additional prefetch distance hides scoreboards.
  Reject the candidate and retain W2 stages=2 pending exact generic-path
  rollback.
- Every one of the 400 samples per implementation and M has a separate
  256 MiB L2 clear outside timing.  Evidence:
  `bench/results/v4_flash_tp_wgmma_w2_stages3_tp4_correctness_20260903.log`,
  `bench/results/v4_flash_tp_wgmma_w2_stages3_tp8shape_correctness_20260903.log`,
  and
  `bench/results/tp4_paired_graph_batch_coldl2_random_w2_stages3_screen_20260903.log`.

### Iteration 28 rollback — restore two TMA weight stages

- Removed the W2-only stage plumbing and restored the generic route GEMM to
  the exact pre-iteration-28 source; a direct diff against commit `e22169a`
  shows only the JIT suffix changed from `v27` to `v29` to force a clean
  rebuild.
- Rebuilt TP4 correctness passes with W13/activation/W2 cosine
  0.999999997/0.999999691/0.999997241 and finite output.
- The batch-paired random-route cold 4x100 rollback screen gives custom
  0.092656 / 0.143552 / 0.217920 / 0.311616 / 0.398720 ms versus Humming
  0.090336 / 0.145824 / 0.226384 / 0.327856 / 0.403600 ms.  Custom loses M8
  by 2.57% and wins M16/M32/M64/M128 by 1.56%/3.88%/5.21%/1.22%; five-shape
  geometric mean is 1.81% faster.  This short run confirms full recovery but
  does not replace the 2,000-sample formal headline because large-M batch
  drift remains visible.
- Every one of the 400 samples per implementation and M has a separate
  256 MiB L2 clear outside timing.  Evidence:
  `bench/results/v4_flash_tp_wgmma_w2_stages3_rollback_correctness_20260903.log`
  and
  `bench/results/tp4_paired_graph_batch_coldl2_random_w2_stages3_rollback_screen_20260903.log`.

### WGMMA iteration 29a — initial warp-specialized persistent W2 (incorrect)

- Added an opt-in `V4_W2_WS_PERSIST=1` W2 specialization.  A fixed grid of
  256-thread CTAs uses one 128-thread producer warpgroup and one 128-thread
  RS-WGMMA consumer warpgroup.  Both roles derive the same strided task stream
  from device `num_tokens_padded`; two full/empty stages are initialized once
  and reused across K tiles and output tasks.  The design does not specialize
  launch geometry to the observed route distribution.
- The first TP4 M8 balanced full-path gate compiled and returned, with W13 and
  fused activation cosines 0.999999998 and 0.999999759, but W2 was identically
  zero: cosine 0, relative L2 1, finite output.  Because correctness failed,
  the skew test and cold-L2 performance screen were intentionally skipped.
- Preserve this state before repair.  Leading suspects are the producer/
  consumer role-local epilogue state and mbarrier phase publication, rather
  than route alignment or the inherited WGMMA math (the accepted path in the
  same binary remains unchanged).
- Evidence:
  `bench/results/v4_flash_tp_wgmma_w2_ws_persistent_initial_correctness_20260903.log`.

### WGMMA iteration 29b — producer-side async-proxy fence (still incorrect)

- Moved `fence.proxy.async.shared::cta` semantics to every producer thread
  after its ordinary activation store and before the producer-only publication
  barrier.  This tests whether the all-zero W2 result came from publishing the
  gathered activation through the generic shared proxy while WGMMA consumed it
  through the async proxy.
- A forced-rebuild TP4 M8 balanced full-path check is unchanged: W13 cosine
  0.999999998, fused activation cosine 0.999999759, but W2 cosine 0 and
  relative L2 1 with finite output.  Therefore reader/writer proxy placement
  alone is not the cause.  Correctness again prevents any performance timing.
- Next diagnosis will write a controlled consumer-epilogue sentinel before
  changing scheduling or math, separating missing epilogue execution from
  zero WGMMA operands.
- Evidence:
  `bench/results/v4_flash_tp_wgmma_w2_ws_writer_proxy_fence_correctness_20260903.log`.

### WGMMA iteration 29c — consumer epilogue sentinel diagnosis

- Added a compile-time diagnostic sentinel and output statistics to the
  correctness harness.  In sentinel mode every valid W2 route/channel store
  writes BF16 123 instead of the accumulator, without changing scheduling or
  route predicates.
- The reduced output has absmax 185 (`123 * 1.5`, BF16-rounded) and all
  32,768 M8 output elements are nonzero.  Thus the consumer warpgroup reaches
  the final-K epilogue, covers all routes/channels, and returns normally.  The
  original all-zero result is upstream in the accumulator inputs/WGMMA path,
  not task termination or route-output coverage.
- Sentinel mode is diagnostic only and is never eligible for timing.  The
  next minimal check will expose ordinary shared activation and packed-weight
  values from the consumer before altering the pipeline topology.
- Evidence:
  `bench/results/v4_flash_tp_wgmma_w2_ws_epilogue_sentinel_correctness_20260903.log`.

### WGMMA iteration 29d — consumer input-visibility diagnosis

- Added a diagnostic epilogue that encodes four consumer-visible inputs as
  bits: ordinary shared activation byte (1), TMA packed-weight byte (2),
  ordinary shared activation scale (4), and TMA E8M0 scale (8).
- TP4 M8 balanced reports a post-reduction absmax of 21, corresponding to a
  maximum per-route code of 14.  The consumer sees packed weights and both
  scales (`2|4|8`) but not the probed activation byte (`1`).  This matches the
  non-debug accumulator being identically zero and localizes the fault to
  publication of the 128 producer threads' ordinary activation stores.
- The existing full barrier has arrival count two: lane 0 accounts for TMA and
  later publishes after a producer-only named barrier.  The next repair will
  instead make all 128 writers arrive on the full mbarrier themselves, plus
  one separate TMA arrival, so every writer directly participates in the
  release/acquire relation.
- Evidence:
  `bench/results/v4_flash_tp_wgmma_w2_ws_input_visibility_correctness_20260903.log`.

### WGMMA iteration 29e — per-writer full/empty mbarrier arrivals (incorrect)

- Changed each full stage from two arrivals to 129: one TMA transaction
  arrival plus one direct arrival from every producer thread after its
  activation store and async-proxy fence.  Likewise, each empty stage now
  requires all 128 consumers to arrive after their final shared read.
- The forced-rebuild TP4 M8 balanced check remains exactly zero in W2
  (`absmax=0`, `nonzero=0`, cosine 0, relative L2 1); W13 and fused activation
  retain their correct cosines.  Therefore transitive lane-0 publication was
  not the root cause, and this more expensive barrier topology is rejected.
- No cold-L2 timing ran because correctness failed.  The next diagnostic will
  inspect the producer's global qactivation source separately from its shared
  destination before making another synchronization change.
- Evidence:
  `bench/results/v4_flash_tp_wgmma_w2_ws_per_writer_mbarrier_correctness_20260903.log`.

### WGMMA iteration 29f — global qactivation source audit

- Added correctness-only qactivation statistics to distinguish a zero source
  from a producer/shared-memory fault.  This does not alter the kernel or any
  timed benchmark path.
- TP4 M8 balanced has 20,716 nonzero qactivation bytes; the final K128 tile
  has 5,175 nonzero bytes and 42/48 route-first bytes are nonzero.  The W2
  result nevertheless remains exactly zero.  Combined with iteration 29d's
  missing shared activation bit, this proves the global activation source is
  valid but does not survive the producer-to-consumer shared path.
- SASS resource metadata for the TP4 specialization is 64 registers/thread
  and 4 KiB static shared plus 22 KiB dynamic shared.  Four 256-thread CTAs per
  H20 SM are resource-feasible, so grid=312 remains a later tuning point once
  correctness is restored.
- Evidence:
  `bench/results/v4_flash_tp_wgmma_w2_ws_global_qactivation_probe_20260903.log`.

### WGMMA iteration 29g — consumer-warpgroup activation gather (incorrect)

- Reworked the pipeline so the producer warpgroup only issues packed-weight
  and E8M0 TMA, while the 128-thread WGMMA consumer gathers qactivation and
  activation scales itself.  Full barriers now have one TMA arrival; empty
  barriers have one arrival from each of the four consumer warps after a
  consumer-only synchronization.  This removes all cross-warpgroup ordinary
  shared-store publication from the activation path.
- TP4 M8 balanced is still identically zero in W2 (`absmax=0`, no nonzero
  outputs) while qactivation is demonstrably nonzero.  The fault therefore is
  not producer-to-consumer visibility.  This design does not advance to a
  cold-L2 timing gate.
- The remaining high-probability issue is reuse of the inherited RS-WGMMA
  register/layout sequence from physical warpgroup 1 rather than warpgroup 0.
  A minimal single-task CTA-synchronized probe is required before investing
  further in persistence mechanics.
- Evidence:
  `bench/results/v4_flash_tp_wgmma_w2_ws_consumer_gather_correctness_20260903.log`.

### WGMMA iteration 29h — physical warpgroup role swap (incorrect)

- Swapped physical roles only: warpgroup 0 now runs the consumer-local gather,
  RS-WGMMA, and epilogue; warpgroup 1 issues TMA.  The task sequence, shared
  layout, barriers, register row mapping, and grid remain unchanged.
- The forced rebuild remains exactly zero in W2 while W13, activation, and the
  global qactivation probe are unchanged.  Therefore the inherited RS-WGMMA is
  not failing merely because it ran in physical warpgroup 1.
- No timing ran.  Next, preserve the just-loaded qactivation register through
  the final K tile and compare it with the shared reload in the diagnostic
  epilogue; this isolates address/layout from global load and route logic.
- Evidence:
  `bench/results/v4_flash_tp_wgmma_w2_ws_role_swap_correctness_20260903.log`.

### WGMMA iteration 29i — controlled-input and single-task probes

- Added a correctness-only `V4_TEST_QACT_ONE=1` harness switch which fills
  qactivation and its scales with one after the normal W13/middle path.  Even
  this controlled nonzero input leaves balanced M8 W2 exactly zero, ruling out
  the activation distribution and FP8 quantizer as causes.
- Ran normal maximal-skew M8 without changing the kernel.  It has 192 runtime
  W2 tasks under grid=234, so each active CTA processes exactly one output
  task.  W2 is still wrong, but now partially nonzero and numerically explosive
  (cosine 0.0151, rel-L2 1.063, absmax 77,824, 10,179 nonzero outputs).
- Cross-task task/scale-slot reuse is therefore not required to trigger the
  fault.  TP4 still reuses each of its two stages within one task (K tiles
  0/2 and 1/3); the next discriminator is the TP8 K=256 specialization, whose
  two K tiles consume each stage exactly once.
- Evidence:
  `bench/results/v4_flash_tp_wgmma_w2_ws_qact_one_and_single_task_probes_20260903.log`.

### WGMMA iteration 29j — TP8 no-stage-reuse discriminator

- Ran the TP8-shape K=256 specialization at maximal-skew M8.  There are two
  K128 tiles, so each of the two pipeline stages is consumed exactly once;
  grid=192 also gives one output task per active CTA.
- W2 remains badly wrong (cosine 0.00861, rel-L2 1.054, absmax 49,920), with
  only 4,096 final output elements nonzero.  Thus neither cross-task nor
  within-task stage reuse is required for the failure.
- This reduces the problem to the single-use tile path: role register
  reconfiguration, full-barrier/TMA handoff, and duplicated WGMMA codegen.
  Test these independently before restoring persistence.
- Evidence:
  `bench/results/v4_flash_tp_wgmma_w2_ws_tp8_no_stage_reuse_probe_20260903.log`.

### WGMMA iteration 29k — remove dynamic register reconfiguration (incorrect)

- Removed both `warpgroup_reg_dealloc<48>` and
  `warpgroup_reg_alloc<112>` from the candidate, leaving all scheduling,
  synchronization, and math unchanged.
- The forced-rebuild TP8 maximal-skew result is numerically identical to the
  prior binary: cosine 0.008614961, rel-L2 1.054372772, absmax 49,920, and
  4,096 nonzero output elements.  Dynamic register reconfiguration is not the
  correctness cause.
- No performance timing ran.  Further work must compare the duplicated
  single-tile math/data indexing directly against the accepted generic kernel
  rather than continue speculative barrier changes.
- Evidence:
  `bench/results/v4_flash_tp_wgmma_w2_ws_no_reg_reconfig_correctness_20260903.log`.

### WGMMA iteration 29l — repair route/scale metadata indexing (correct)

- Found a deterministic indexing error in the duplicated persistent consumer:
  the first eight threads populated `route_ids[0..7]` and activation scales
  from `token_slot = consumer_tid / 16`, which is zero for all eight metadata
  writers.  Thus every metadata entry referred to route 0 even though the
  activation tile itself was correctly gathered with sixteen threads per row.
- Changed only metadata loading to index `sorted_ids` by `consumer_tid` and to
  load the matching activation scale.  TP8 K=256 maximal skew now passes W2
  with cosine 0.999997249, as do TP4 K=512 balanced (0.999997256) and maximal
  skew (0.999997235).  W13 and fused activation remain correct in all gates.
- The prior zero/sparse outputs and diagnostic symptoms are fully explained by
  output-route aliasing and scale mismatch; speculative barrier and WGMMA-role
  explanations are rejected.  The candidate may now enter the cold-L2 screen.
- Evidence:
  `bench/results/v4_flash_tp_wgmma_w2_ws_metadata_index_fix_correctness_20260903.log`.

### WGMMA iteration 29m — persistent W2 grid 234 cold-L2 screen (rejected)

- Screened the now-correct candidate in the paired TP4 CUDA-Graph harness with
  random real route metadata, 4 x 100 samples per implementation and M, and a
  separate 256 MiB cache flush before every replay outside the event interval.
- The candidate loses at every M: custom/Humming is 1.07984, 1.09124, 1.09083,
  1.11210, and 1.08439 for M=8,16,32,64,128.  Geometric-mean latency is
  0.226701 ms versus Humming's 0.207673 ms, so custom is 9.16% slower.
- Correctness remains valid at every point.  This configuration is rejected;
  before removing the prototype, screen the runtime-only 156- and 312-CTA
  choices to distinguish an occupancy/grid error from intrinsic pipeline cost.
- Evidence:
  `bench/results/tp4_paired_graph_coldl2_w2_ws_p234_screen_20260903.log`.

### WGMMA iteration 29n — persistent W2 grid 156 cold-L2 screen (rejected)

- Reduced the runtime persistent grid from three to two CTAs per H20 SM and
  repeated the same 4 x 100 paired, per-replay cold-L2 TP4 screen.
- Performance degrades monotonically with M: custom/Humming is 1.15717,
  1.19820, 1.22328, 1.24727, and 1.27552.  Geometric-mean custom latency is
  0.249109 ms versus 0.204254 ms for Humming, a 21.96% loss.
- All correctness checks pass.  Grid 156 is rejected and shows that limiting
  the 256-thread candidate to two resident CTA waves is clearly insufficient.
- Evidence:
  `bench/results/tp4_paired_graph_coldl2_w2_ws_p156_screen_20260903.log`.

### WGMMA iteration 29o — persistent W2 grid 312 cold-L2 screen (rejected)

- Increased the runtime grid to four CTAs per H20 SM, the resource-feasible
  maximum measured for this 256-thread kernel, and repeated the 4 x 100 paired
  cold-L2 TP4 screen.
- This is substantially better than grids 156 and 234 but still loses at every
  M: custom/Humming is 1.04343, 1.03507, 1.02906, 1.03749, and 1.01779.
  Geometric means are 0.215961 ms custom and 0.209156 ms Humming, a 3.25% loss.
- The accepted one-warpgroup path had previously reached roughly parity under
  the stricter 2,000-sample protocol.  Consequently the extra warpgroup and
  full/empty synchronization cost more than TMA overlap can recover.  Reject
  the entire warp-specialized persistent branch and restore the exact accepted
  source before the next optimization direction.
- Evidence:
  `bench/results/tp4_paired_graph_coldl2_w2_ws_p312_screen_20260903.log`.

### WGMMA iteration 29p — exact accepted-winner restore

- Restored `v4_flash_tp_wgmma.py` and its correctness test byte-for-byte from
  accepted commit `7cc55d5` after rejecting the warp-specialized branch.  The
  SHA-256 digests match the git objects exactly; experimental commits and logs
  remain in history.
- TP4 balanced/skew and TP8-shape skew correctness gates all pass, with W2
  cosine between 0.999997235 and 0.999997256.  A fresh 4 x 100 paired cold-L2
  random-route regression gives geometric means 0.210477 ms custom and
  0.209839 ms Humming, only 0.304% slower.  This restores the prior near-parity
  state; M128 has high batch variability and is not used to supersede the
  existing 2,000-sample formal estimate.
- Evidence:
  `bench/results/tp4_paired_graph_coldl2_exact_winner_restore_20260903.log`.

### Post-iteration-29 profile — Humming W2 scheduler target

- Collected a current 21-pass detailed NCU report for Humming's TP4 M32
  random-route local path after the standard explicit 256 MiB cache clear.
  The selected MXFP4 indexed W2 uses 128 threads, grid 312, five stages, four
  CTAs/SM, 101 registers/thread, and 53.248 KiB shared memory per CTA.
- Against the accepted custom detailed report, Humming W2 executes 21.36M
  instructions versus custom's 24.80M and only 159k branch instructions versus
  749k.  It reaches 68.58 us versus the historical custom 74.11 us despite
  much lower achieved occupancy (21.1% versus 53.3%).  Warp cycles per issued
  instruction are 5.63 versus 12.65.
- This rejects occupancy as the primary goal.  The next candidate will retain
  one 128-thread WGMMA warpgroup, amortize scheduler/LUT/mbarrier setup across
  tasks, carry barrier phase continuously, and expose all four K128 weight
  loads with a five-stage buffer.  W13 remains unchanged.
- Evidence:
  `bench/results/tp4_humming_m32_random_detailed_coldl2_current_ncu.{log,ncu-rep}`.

### WGMMA iteration 30a — single-warpgroup persistent W2 correctness

- Reworked the earlier runtime-grid experiment around one 128-thread WGMMA
  warpgroup.  Unlike iteration 24, LUT and mbarriers are initialized only once
  per CTA; barrier parity follows the continuous task/K-tile sequence.  The
  fixed grid reads the actual task bound from device `num_tokens_padded` and
  therefore remains valid when captured-graph route metadata changes.
- Double-buffered route metadata allows the next task's eight writers to run
  before the single CTA rendezvous without clobbering the preceding task.
  Packed-weight stages remain at the accepted depth of two for this first
  isolated scheduler test.
- TP8 K=256 maximal skew and TP4 K=512 balanced/maximal-skew full-path checks
  all reproduce the accepted errors exactly.  W2 cosine ranges from
  0.999997235 to 0.999997256 and all outputs are finite.  The candidate is
  eligible for cold-L2 grid screening.
- Evidence:
  `bench/results/v4_flash_tp_wgmma_w2_single_wg_persistent_correctness_20260903.log`.

### WGMMA iteration 30b — single-WG persistent grid 8/SM screen

- Screened the correct barrier-reuse scheduler at 624 CTAs (eight per H20 SM)
  with 4 x 100 paired, individually cold TP4 samples at every target M.
- It narrowly wins M32/M64 by 0.34%/0.45% but loses M8/M16/M128 by
  3.44%/1.43%/3.85%.  Geometric means are 0.212772 ms custom and 0.209485 ms
  Humming, so the candidate is 1.57% slower overall.
- This does not establish an improvement.  Before rejection, run the refactored
  uncapped p0 control and the p12 boundary: p0 measures loop/source overhead,
  while p12 reduces serialized tasks per CTA at the cost of a second CTA wave.
- Evidence:
  `bench/results/tp4_paired_graph_coldl2_single_wg_persistent_p8_screen_20260903.log`.

### WGMMA iteration 30c — refactored non-persistent p0 control

- Disabled the runtime grid cap in the identical candidate binary and repeated
  the 4 x 100 paired cold-L2 screen.  Geometric means are 0.209953 ms custom
  and 0.210511 ms Humming, a noise-sized 0.265% custom lead consistent with
  the exact-winner near-parity result.
- Relative to the immediately preceding p8 run, p0 is 1.34% faster
  geometrically and has lower custom medians at M8/M16/M32/M64.  The source
  refactor itself did not create the p8 loss; serial task execution does.
- One resource-bound p12 check remains useful because it trades a second CTA
  wave for fewer tasks per persistent CTA.  If it cannot beat p0, reject this
  two-stage persistent formulation before adding deeper stages.
- Evidence:
  `bench/results/tp4_paired_graph_coldl2_single_wg_persistent_p0_control_20260903.log`.

### WGMMA iteration 30d — single-WG persistent grid 12/SM screen

- Raised the fixed grid to 936 CTAs, beyond the one-wave resource limit, and
  repeated the 4 x 100 paired cold-L2 TP4 screen.
- It wins only M32/M64 by 0.92%/1.06% and loses M8/M16/M128 by
  2.83%/0.57%/4.97%.  Geometric means are 0.211969 ms custom and 0.209347 ms
  Humming, so p12 is 1.25% slower and also loses to the adjacent p0 control.
- Reject the two-stage persistent formulation at both p8 and p12.  The next
  isolated test adopts the remaining scheduler distinction directly supported
  by Humming's NCU/config evidence: five W2-only stages at grid 312, allowing
  all four K128 weight transfers to be outstanding before computation.
- Evidence:
  `bench/results/tp4_paired_graph_coldl2_single_wg_persistent_p12_screen_20260903.log`.

### WGMMA iteration 30e — five-stage W2 initial TP8 launch failure

- Added a W2-only compile-time stage count and configured TP4 for five stages
  at grid 312, matching the structural feature identified in Humming's NCU
  report.  TP4 balanced and maximal-skew correctness both pass with exactly
  the accepted errors.
- Applying five stages unchanged to TP8 K=256 requests 52,224 bytes of dynamic
  shared memory because scalar weight scales are stage-local, and the launch
  fails with `cudaErrorInvalidValue` at the default 48 KiB limit.  No timing is
  eligible while TP8 does not run.
- Preserve the failure, then specialize TP8 to two stages: its two K128 tiles
  are already fully exposed at that depth, so additional buffers have no
  pipeline value.
- Evidence:
  `bench/results/v4_flash_tp_wgmma_w2_persistent_stage5_initial_correctness_20260903.log`.

### WGMMA iteration 30f — specialize TP8 back to two stages

- Specialized only K=256 W2 to two stages.  TP8 has exactly two K128 tiles,
  so this exposes every transfer while reducing dynamic shared memory below
  the launch limit; K=512 TP4 retains the five-stage candidate.
- Forced-rebuild TP8 maximal-skew and TP4 balanced checks both pass with the
  exact accepted numerical errors (W2 cosine 0.999997249/0.999997256).  Added
  explicit benchmark metadata for TP4 and TP8 W2 stage counts.
- The five-stage TP4 candidate is now eligible for cold-L2 performance timing.
- Evidence:
  `bench/results/v4_flash_tp_wgmma_w2_persistent_stage5_tp8_fix_correctness_20260903.log`.

### WGMMA iteration 30g — five-stage persistent W2 screen (rejected)

- Screened the TP4 five-stage, 312-CTA candidate with 4 x 100 paired,
  individually cold samples.  It loses at every M by 2.77-8.83%.
  Geometric means are 0.222104 ms custom and 0.209687 ms Humming, a 5.92% loss.
- Simply issuing all four K128 weight transfers before computation sacrifices
  too much residency and does not reproduce Humming's tightly interleaved
  s2r/dequant pipeline.  Reject five stages.
- One resource-exact intermediate remains: four stages consume about 40 KiB
  total shared memory and permit five resident CTAs/SM.  Add grid factor five
  and screen stage4/p5 before closing the single-WG persistent direction.
- Evidence:
  `bench/results/tp4_paired_graph_coldl2_single_wg_persistent_stage5_p4_screen_20260903.log`.

### WGMMA iteration 30h — four-stage W2 at five CTAs/SM correctness

- Added grid factor five and compiled a four-stage TP4 W2 specialization.  Its
  roughly 40 KiB total shared footprint permits five resident CTAs per H20 SM,
  so grid 390 uses the full resource-feasible first wave.  TP8 remains at two
  stages because it has only two K128 tiles.
- TP4 balanced/maximal-skew and TP8-shape maximal-skew all pass with numerical
  errors identical to the accepted winner.  W2 cosine is at least 0.999997235.
- The candidate is eligible for a paired cold-L2 screen.
- Evidence:
  `bench/results/v4_flash_tp_wgmma_w2_persistent_stage4_p5_correctness_20260903.log`.

### WGMMA iteration 30i — four-stage persistent W2 screen (rejected)

- Screened stage4/grid390 using 4 x 100 paired, per-replay cold-L2 TP4 samples.
  It loses at all target M values by 2.29-8.46%; geometric means are
  0.220499 ms custom and 0.209563 ms Humming, a 5.22% loss.
- Combined with the p0/p8/p12 two-stage and stage5/p4 results, simple
  single-warpgroup task serialization is closed: neither amortized setup nor
  deeper weight staging compensates for reduced independent-CTA concurrency.
  Restore the exact accepted winner before pursuing instruction/data-path
  changes that keep one logical task per CTA.
- Evidence:
  `bench/results/tp4_paired_graph_coldl2_single_wg_persistent_stage4_p5_screen_20260903.log`.

### WGMMA iteration 30j — exact accepted-winner restore after persistent closure

- Restored all four runtime/benchmark files byte-for-byte from accepted commit
  `7cc55d5`; all SHA-256 hashes match that commit.
- TP4 balanced/maximal-skew and TP8-shape maximal-skew correctness pass with
  the accepted numerical errors; W2 cosine is at least 0.999997235.
- A fresh 4 x 100 paired, per-replay cold-L2 TP4 regression gives geometric
  means 0.209169 ms custom and 0.210933 ms Humming: custom is 0.843% faster in
  this screen.  This is consistent with near parity and does not supersede the
  prior 2,000-sample formal comparison.
- The next optimization axis must retain one logical task per CTA and reduce
  its W2 instruction/scoreboard cost; no persistent scheduler code remains in
  the runtime files.
- Evidence:
  `bench/results/tp4_paired_graph_coldl2_exact_winner_restore_v2_20260903.log`.

### WGMMA iteration 31a — pack W2 E8M0 scale quads into registers

- Source comparison with Humming identified a narrower S2R difference: the
  accepted W2 consumes sixteen scalar `LDS.U8` scale loads per thread/K128,
  while Humming loads each four-scale group as a register word before its
  unrolled fused dequant loop.
- Added opt-in `V4_W2_SCALE_REGS=1`.  Only W2 loads two packed scale words per
  64-channel group after the TMA barrier, then extracts the four bytes in
  registers.  W13, grid scheduling, weight bytes, math, epilogue, route
  metadata, and graph/all-reduce path are unchanged.
- TP4 balanced/maximal-skew and TP8-shape maximal-skew correctness reproduce
  the accepted errors exactly; W2 cosine is at least 0.999997235.
- The TP4 W2 binary uses 47 registers/thread (accepted 48), no local memory,
  and retains eight-CTA/SM resource eligibility.  Its static unrolled SASS has
  zero `LDS.U8` occurrences, so the intended dependency removal survived
  compilation.  Proceed to paired cold-L2 timing.
- Evidence:
  `bench/results/v4_flash_tp_wgmma_w2_scale_regs_correctness_20260903.log`.

### WGMMA iteration 31b — packed scale registers cold screen (rejected)

- The 4 x 100 paired cold-L2 candidate screen gives geometric means
  0.209728 ms custom and 0.210071 ms Humming, only a noise-sized 0.163% lead.
  An immediate identical-source `V4_W2_SCALE_REGS=0` reverse control gives
  0.209993 ms custom and 0.210464 ms Humming, likewise near parity.
- Candidate/control custom median ratios for M8/M16/M32/M64/M128 are
  1.00638 / 1.00868 / 1.00852 / 1.00694 / 0.96394.  Thus the apparent 0.126%
  candidate geometric edge comes entirely from the known highly variable
  M128 point; candidate consistently loses the more stable M8-M64 points by
  0.64-0.87%.
- Eliminating scalar scale LDS dependencies in SASS is not sufficient: byte
  extraction/register live ranges cost slightly more than the broadcast
  shared loads.  Reject this candidate and restore the exact accepted source.
- Every sample uses the required separate 256 MiB cold-L2 clear outside the
  event interval; all graph correctness and all-reduce checks pass.
- Evidence:
  `bench/results/tp4_paired_graph_coldl2_w2_scale_regs_screen_20260903.log`.

### WGMMA iteration 32a — prefetch next W2 K32 S2R operands across QGMMA

- Added opt-in `V4_W2_S2R_PREFETCH=1` on the restored one-task-per-CTA path.
  Each W2 group retains the next K32 packed-weight words and shared-LUT result
  in registers: they are loaded before the current QGMMA and dequantized after
  its dependency wait.  This mirrors Humming's `load_stage_iter(next)` before
  `mma.run(current)` without changing CTA scheduling or accumulating two MMA
  chains (the rejected iteration-6 experiment).
- TP4 balanced/maximal-skew and TP8-shape maximal-skew correctness reproduce
  accepted errors exactly; W2 cosine is at least 0.999997235.
- TP4 W2 uses 56 registers/thread, no local memory, and unchanged shared
  memory, remaining below the 64-register eight-CTA/SM boundary.  SASS audit
  confirms next-iteration LDS operations precede current QGMMA and their data
  remains live across `WARPGROUP.DEPBAR`.
- Proceed to the paired per-replay cold-L2 screen.
- Evidence:
  `bench/results/v4_flash_tp_wgmma_w2_s2r_prefetch_correctness_20260903.log`.

### WGMMA iteration 32b — S2R prefetch paired confirmation (accepted)

- Ran candidate/control/candidate, each with 4 x 100 paired, individually
  cold-L2 TP4 graph samples at all five random-route M values.  Relative to
  the intervening identical-source control, both candidate runs win every M:
  0.29-2.84% and 0.38-1.36%, respectively.  Their geometric-mean speedups are
  1.093% and 0.751%; the geometric average is 0.922%.
- Against Humming the two candidate windows range from a 0.47% to 1.92%
  geometric lead, so the defensible full-path status remains near parity, not
  the 20% target.  Set `V4_W2_S2R_PREFETCH=1` as the new default because its
  self-control direction is consistent across all ten pointwise comparisons.
- A matching cold NCU control/candidate pair isolates W2 at M32: duration
  falls 69.47 -> 67.36 us (3.04%) while instructions rise 0.86%, occupancy is
  unchanged, eligible cycles rise 66.70% -> 68.66%, and warp cycles per issued
  instruction fall 12.80 -> 12.44.  This validates latency hiding rather than
  instruction deletion as the mechanism.
- Evidence:
  `bench/results/tp4_paired_graph_coldl2_w2_s2r_prefetch_screen_20260903.log`
  and `bench/results/tp4_wgmma_m32_w2_s2r_prefetch_ncu_20260903.log`.

### WGMMA iteration 32c — S2R prefetch route sensitivity

- Balanced routes give geometric means 0.236129 ms custom versus 0.240270 ms
  Humming, a 1.75% custom lead.  Maximal skew gives 0.077064 versus 0.074530
  ms, a 3.40% custom loss.  Every point uses 400 per-replay cold-L2 samples.
- The new path narrows the previously measured roughly 4.40% skew deficit but
  does not remove route-distribution sensitivity.  It is accepted as an
  incremental default, not as evidence of a universal Humming win.
- Evidence:
  `bench/results/tp4_paired_graph_coldl2_w2_s2r_prefetch_routes_20260903.log`.

### WGMMA iteration 32d — new default TP8 distributed run-through

- With all eight H20s idle, ran the new default at the true TP8 shard
  (`I/rank=256`) through an eight-rank CUDA Graph including SGLang
  `CustomAllReduceV2`.  M8 random routes pass independent local-recompute plus
  NCCL-reference validation: minimum-rank cosine 0.999991970, relative L2
  0.004007414, all values finite, and `allreduce_ok=true`.
- Twenty individually cold samples give median max-rank latency 0.072976 ms,
  essentially matching the previous default's 0.072816 ms smoke.  This proves
  TP8 graph/runtime compatibility only; the primary optimization target and
  performance decisions remain TP4.
- Evidence:
  `bench/results/tp8_wgmma_graph_coldl2_s2r_prefetch_m8_smoke_20260903.log`.

### WGMMA iteration 33a — extend cross-QGMMA S2R prefetch to W13

- Added independent opt-in `V4_W13_S2R_PREFETCH=1`, reusing the validated
  W2 next-K32 packed-weight/LUT prefetch while keeping the new W2 default on.
- TP4 auto split-K=4, TP4 forced split-K=2, and TP8-shape split-K=4 all pass
  with accepted numerical errors.  W13 cosine is at least 0.999999997 and W2
  cosine at least 0.999997249.
- All TP4/TP8 W13 specializations use 56 registers/thread with no local memory
  and unchanged shared memory.  The register increase does not reduce
  resource-feasible residency because shared memory already limits the kernel
  to nine CTAs/SM.
- Proceed to the paired random-route cold-L2 screen before changing the W13
  default.
- Evidence:
  `bench/results/v4_flash_tp_wgmma_w13_s2r_prefetch_correctness_20260903.log`.

### WGMMA iteration 33b — W13 S2R prefetch confirmation (accepted)

- Ran candidate/control/candidate with 400 paired, per-replay cold-L2 samples
  per M.  The second candidate versus the intervening control changes custom
  medians by -0.94%/-0.40%/+0.04%/-0.84%/-0.38% for M8..M128, a 0.509%
  geometric improvement.  The first candidate's anomalously low M128 result
  is explicitly excluded from the causal claim.
- Independent matching NCU captures show W13 falling 127.97 -> 125.82 us
  (1.68%) while instruction count rises 1.44%, occupancy stays unchanged,
  eligible cycles rise 70.19% -> 72.11%, and warp cycles per issued
  instruction fall 12.20 -> 11.89.  This reproduces the W2 latency-hiding
  mechanism at smaller end-to-end magnitude.
- Set `V4_W13_S2R_PREFETCH=1` as the new default.  Against Humming, candidate
  B is 1.23% faster geometrically but still loses M8 and M128; this remains an
  incremental winner, not a 20% result.
- Evidence:
  `bench/results/tp4_paired_graph_coldl2_w13_s2r_prefetch_screen_20260903.log`
  and `bench/results/tp4_wgmma_m32_w13_s2r_prefetch_ncu_20260903.log`.

### WGMMA iteration 34a — asynchronous indexed-activation G2S

- Added opt-in `V4_ACTIVATION_CP_ASYNC=1`.  Each 8x128 FP8 activation tile is
  now moved by 64 lanes issuing 16-byte `cp.async.cg` transactions, including
  hardware zero fill for padded routes, instead of 128 synchronous uint2
  LDG+STS pairs.  The async group is committed before the packed-weight TMA
  wait and consumed before the CTA/WGMMA barrier, matching Humming's indexed
  legacy-G2S structure.
- TP4 split-K=4, TP4 split-K=2, and TP8-shape maximal-skew correctness exactly
  reproduce accepted errors.  The skew case explicitly validates zero fill.
- Registers, local memory, and shared memory are unchanged from the accepted
  S2R-prefetch winner; SASS contains LDGSTS in both W13 and W2.  Proceed to a
  paired per-replay cold-L2 screen.
- Evidence:
  `bench/results/v4_flash_tp_wgmma_activation_cp_async_correctness_20260903.log`.

### WGMMA iteration 34b — async activation paired screen (rejected)

- Ran candidate/control/candidate with 400 paired, individually cold-L2 TP4
  samples per M on random routes.  Candidate A versus control is 0.166% faster
  geometrically; candidate B is 0.165% slower.  The pointwise directions also
  disagree: A changes M8..M128 by +0.03%/-0.09%/-0.06%/-0.74%/+0.02%, while
  B changes them by -0.24%/-0.07%/+0.23%/-0.46%/+1.38%.
- Reject the candidate.  Its positive and negative geometric deltas are
  symmetric at noise scale, and it does not consistently improve the five
  required M values.  Restore the exact W13/W2 S2R-prefetch winner rather than
  retaining an unproven async path in the runtime source.
- Evidence:
  `bench/results/tp4_paired_graph_coldl2_activation_cp_async_screen_20260903.log`.

### WGMMA iteration 35a — M-major FP8 activation scales

- Added opt-in `V4_M_MAJOR_ACTIVATION_SCALE=1`.  Both input quantizers now
  emit group-major scales, `[K/128, rows]`, and the route GEMMs gather the
  eight scale values of one K tile from consecutive rows.  The fused
  SwiGLU/quant kernel writes its W2 scales directly in the same layout.
- This targets the only notable uncoalesced global read left by the detailed
  source-counter profile.  It changes neither quantized FP8 values nor scale
  math; only the scale tensor layout and addresses differ.
- Full-block TP4 split-K=4, TP4 forced split-K=2, and TP8-shape maximal-skew
  checks reproduce the accepted numerical errors.  W13 cosine is at least
  0.999999997 and W2 cosine at least 0.999997235.  Candidate W13/W2 retain 56
  registers/thread, no local memory, and unchanged shared-memory footprints.
- Proceed to a candidate/control/candidate random-route TP4 cold-L2 screen.
- Evidence:
  `bench/results/v4_flash_tp_wgmma_m_major_scale_correctness_20260903.log`.

### WGMMA iteration 35b — M-major scale screen (rejected)

- Candidate then same-source control, each with 400 paired per-replay cold-L2
  TP4 samples per M, gives candidate/control custom-median ratios
  1.0156/1.0105/1.0152/1.0077/1.0201 for M8..M128.  The candidate loses every
  point and regresses geometric mean by 1.381%.
- Humming is 0.51% faster in the candidate window, so system drift cannot
  explain the custom regression.  Reject without a redundant third run; the
  five consistent margins are all larger than the noise band seen in the
  immediately preceding async-G2S experiment.
- An attempted Systems stage breakdown repeated the known profiler-control
  deadlock and was terminated by exact PID after no progress; it produced no
  usable report and contributes no timing evidence.
- Restore the exact S2R winner.  Coalescing eight tiny cached scale loads does
  not repay the changed quantizer/store layout and runtime indexing.
- Evidence:
  `bench/results/tp4_paired_graph_coldl2_m_major_scale_screen_20260903.log`.

### WGMMA iteration 36a — split-major W13 task ordering

- Added opt-in `V4_SPLIT_MAJOR_TASK_ORDER=1`.  The accepted physical grid and
  one-task-per-CTA execution are unchanged.  Within each expert, W13 block IDs
  now enumerate every N tile of one split before advancing to the next split,
  instead of enumerating every split of one N tile first.  Adjacent CTAs can
  therefore reuse the same 8xK-slice activation data in cache.  W2 has
  `SplitK=1`, so its logical order is unchanged.
- TP4 split-K=4, TP4 forced split-K=2, and TP8-shape checks exactly reproduce
  accepted numerical errors.  TP4 W13 and W2 retain 56 registers/thread, no
  local memory, and unchanged shared-memory footprints.
- Proceed to the paired random-route cold-L2 screen; a win would be a cache
  scheduling effect, not less math or fewer weight bytes.
- Evidence:
  `bench/results/v4_flash_tp_wgmma_split_major_order_correctness_20260903.log`.

### WGMMA iteration 36b — split-major task-order screen (rejected)

- Candidate/control/candidate, each with 400 paired per-replay cold-L2 TP4
  samples per M, gives candidate/control geometric ratios 0.994964 and
  1.000845.  The first apparent gain is dominated by an M128 window that is
  2.62% faster; the repeat instead loses M128 by 1.04% and loses four of five
  M values overall.
- Reject and restore the exact S2R winner.  Reordering CTAs for activation
  cache locality does not produce a stable benefit; the activation footprint
  is evidently already cached well enough relative to the cold weight stream.
- Evidence:
  `bench/results/tp4_paired_graph_coldl2_split_major_order_screen_20260903.log`.

### Post-iteration-36 cold CUDA-Graph stage budget

- Added `bench/profile_v4_flash_tp_stages.py`, which places external CUDA
  timing-event nodes around each local stage inside one captured graph.  This
  avoids both eager Python launch gaps and the repeatable Nsight Systems
  profiler-control deadlock.  Each of 50 samples per implementation/M has a
  separate 256 MiB pre-replay L2 clear outside the graph and timing events.
- Custom local totals for M8/16/32/64/128 are
  106.240/154.592/224.496/304.800/365.984 us, versus Humming
  108.736/162.096/241.632/330.752/392.912 us.  Custom is 2.30-7.85% lower in
  latency, but this intentionally excludes the common all-reduce and is not a
  replacement for the paired end-to-end score.
- Custom W13/W2 sums are 80.512/127.728/197.216/275.600/333.696 us.  At
  M32/M64/M128 those sums alone exceed 0.8x Humming's entire local pipeline by
  3.91/11.00/19.37 us.  Consequently, deleting only route/quant/activation/
  reduction overhead cannot meet the 20% goal; another core GEMM speedup is
  mathematically required.
- The remaining total reduction needed is 18.12%/16.12%/13.89%/13.19%/14.11%
  at M8..M128.  Prioritize earlier weight-TMA refill and other core overlap,
  while retaining the already faster fused middle path.
- Evidence:
  `bench/results/tp4_local_graph_stage_budget_coldl2_20260903.log`.

### WGMMA iteration 37a — early TMA stage refill

- Added opt-in `V4_EARLY_STAGE_REFILL=1`.  After the final K32 WGMMA wait
  retires, the consumed packed-weight stage is dead.  The candidate issues its
  next weight TMA (and, at quartet boundaries, the next scale TMA) before the
  FP32 activation-scale accumulation rather than after it.  This preserves two
  buffers and one logical task per CTA while extending useful TMA overlap.
- TP4 split-K=4, TP4 forced split-K=2, and TP8-shape checks reproduce accepted
  errors exactly.  TP4 W13 rises from 56 to 59 registers/thread, while W2
  falls to 55; neither spills and shared-memory footprints are unchanged.
- Use the new graph-internal stage profiler for a core screen before paying
  for a three-window distributed comparison.  Only a repeatable W13/W2
  reduction is eligible for the full paired benchmark.
- Evidence:
  `bench/results/v4_flash_tp_wgmma_early_stage_refill_correctness_20260903.log`.

### WGMMA iteration 37b — early-refill core screen (rejected)

- Graph-internal event timing with 100 individually cold-L2 local TP4 samples
  per point compares candidate and same-source control at M8/M32/M128.
  Candidate/control total ratios are 0.99893/1.00806/1.00433.  At M32 and
  M128, W13 regresses 0.58%/0.52% and W2 regresses 0.97%/0.74%.
- Reject before consuming four GPUs for a full paired screen.  M8's 0.11%
  total improvement is noise-sized and contains a 0.76% W2 loss.  Starting
  the TMA a few FP32 instructions earlier does not offset W13's 56-to-59
  register live-range increase and schedule perturbation.
- Restore the exact accepted S2R winner.
- Evidence:
  `bench/results/tp4_local_graph_early_stage_refill_screen_20260903.log`.

### WGMMA iteration 38a — double-buffered indexed activation prefetch

- Added opt-in `V4_ACTIVATION_DOUBLE_BUFFER=1`.  The candidate uses two
  8x128 FP8 activation buffers.  Sixty-four lanes issue 16-byte `cp.async`
  copies for tile `k+1` immediately after tile `k` becomes visible, then the
  four K32 dequant/WGMMA steps execute while that copy is outstanding.  The
  next iteration waits for the copy and synchronizes the CTA before forming
  its WGMMA descriptor.
- Padded route rows retain hardware zero fill.  Weight/scale staging and the
  accepted cross-QGMMA S2R prefetch are unchanged; dynamic shared memory
  grows by only 1 KiB.  This experiment tests actual compute overlap rather
  than another launch/metadata micro-optimization.
- Validate TP4 split-K=4, forced split-K=2, and TP8 shape before measuring.
  If correct, use graph-internal event nodes with an excluded 256 MiB cold-L2
  clear before every sample, and require both W13/W2 core evidence before a
  distributed end-to-end screen.

### WGMMA iteration 38b — activation double-buffer screen (rejected)

- TP4 split-K=4, forced split-K=2 maximal skew, and TP8-shape maximal-skew
  checks reproduce the accepted numerical errors.  Candidate TP4 W13 uses 50
  registers/thread and W2 uses 56, with no local-memory spill.
- Graph-internal timing with 100 individually cold-L2 samples per point gives
  candidate/control total ratios 1.01004/1.02793/1.01627 at M8/M32/M128.
  Every measured W13 and W2 point loses; at M32 both regress about 3%.
- Reject before distributed timing and restore the exact S2R winner.  The
  current synchronous activation move already overlaps the packed-weight TMA
  wait; an extra async group and CTA synchronization perturb the compute
  schedule more than they hide activation latency.
- Evidence:
  `bench/results/v4_flash_tp_wgmma_activation_double_buffer_correctness_20260903.log`
  and
  `bench/results/tp4_local_graph_activation_double_buffer_screen_20260903.log`.

### WGMMA iteration 39a — synthesized scale LUT plus S2R prefetch

- Added support for combining `V4_DEQUANT_SYNTH_LUT=1` with both accepted
  W13/W2 cross-QGMMA S2R prefetch paths.  The first K32 step synthesizes its
  two packed E4M3 magnitude words from E8M0 in registers; every later step
  computes the following words before the current WGMMA, shortening the
  dependency chain that made the isolated iteration-20 synth probe neutral.
- This removes the per-CTA 2 KiB LUT initialization and all inner shared-LUT
  reads, but is bit-exact only for the benchmark exponent range 125..128.
  Treat it as a core diagnostic: it cannot become a production default unless
  the real checkpoint transform proves or creates the required scale range.
- Gate on TP4 split-K=4, forced split-K=2, and TP8-shape correctness, then
  compare candidate/control W13 and W2 using graph-internal events with a
  separate excluded 256 MiB cold-L2 clear before every sample.

### WGMMA iteration 39b — synth plus S2R wins, but remains diagnostic

- TP4 split-K=4, forced split-K=2 skew, and TP8-shape skew reproduce accepted
  errors for codes 125..128.  TP4 W13/W2 fall from 56 to 54 registers/thread,
  static shared memory falls from 4 to 2 KiB, and neither kernel spills.
- Two graph-internal candidate runs around a same-source control improve W13
  by 1.86-3.00%, W2 by 1.52-2.79%, and local total by 1.01-2.40% at every
  measured M8/M32/M128 point.
- TP4 candidate/control/candidate distributed screens use 400 individually
  cold-L2 graph samples per implementation and M.  Candidate A beats control
  custom latency at every M by 1.25-2.80% (1.94% geomean); paired-Humming
  normalization gives a 1.38-2.91% pointwise gain (2.55% geomean).  Candidate
  B confirms M8-M64; its system-shifted M128 result is not used causally.
- Keep this as positive mechanism evidence, not a production selection.  The
  affine generator exactly matches the full LUT only for E8M0 codes 122..133.
  Next add a global-LUT fallback outside that range, validate fallback codes,
  and remeasure before changing the default.
- Evidence:
  `bench/results/v4_flash_tp_wgmma_synth_s2r_correctness_20260903.log` and
  `bench/results/tp4_synth_s2r_coldl2_screen_20260903.log`.

### WGMMA iteration 40a — full-domain synthesized-LUT fast path

- Extended the positive iteration-39 mechanism to arbitrary E8M0 codes.
  Codes 122..133 use the bit-exact affine register generator; all lower and
  upper codes load the corresponding entry directly from the immutable global
  256-row LUT.  The unsigned range test covers both sides without clamping.
- The specialization still omits the per-CTA 2 KiB shared LUT and its setup.
  It therefore remains fast for checkpoint groups in the common affine range,
  while fallback groups trade latency rather than correctness.  No offline
  normalization assumption is needed.
- Extended the untracked correctness harness with an explicit scale range so
  the normal fast path, lower fallback, and upper fallback can be tested
  independently at TP4 split-K=4/2 and TP8 shape before cold-L2 timing.

### WGMMA iteration 40b — per-word global fallback is correct but rejected

- Correctness covers affine codes 125..128, low fallback codes 118..121,
  upper fallback code 134 with finite test operands, and TP8 mixed 118..124.
  All finite cases pass.  Random large-magnitude code-134 weights reproduce
  the canonical LUT's E4M3 NaNs and are excluded as numerically ill-posed.
- Despite all benchmark scales taking the affine branch, graph-internal cold
  timing regresses local total by 11.03%/15.93%/18.24% at M8/M32/M128;
  W13 alone loses as much as 21.37%.  The runtime branch destroys the inner
  schedule, not just fallback performance.
- Reject before distributed timing and restore the exact general-scale S2R
  winner.  Humming source inspection identifies the viable production path:
  normalize each expert's E8M0 range to offsets 1..12 once at model load,
  adjust only clamped FP4 groups, and apply one expert scale in the epilogue.
- Evidence:
  `bench/results/v4_flash_tp_wgmma_synth_global_fallback_correctness_20260903.log`
  and
  `bench/results/tp4_local_graph_synth_global_fallback_screen_20260903.log`.

### WGMMA iteration 41a — offline per-expert E8M0 normalization

- Added opt-in `V4_NORMALIZED_WEIGHT_SCALE=1`.  At model-load time, each
  expert's raw E8M0 groups are mapped to offsets 1..12.  If its original range
  exceeds eleven exponents, only groups below the retained window have their
  packed E2M1 values rescaled and requantized.  A single FP32 expert factor
  restores the removed base exponent plus Humming's six-bit FP4-to-E4M3
  offset.  This is the same numerical normalization principle as Humming, but
  the timed route GEMMs remain our implementation.
- The runtime inner loop is branch-free again and uses the successful affine
  generator plus accepted cross-QGMMA S2R prefetch.  Instead of multiplying
  every output accumulator, eight metadata lanes fold the expert factor into
  each K128 activation scale once, minimizing the added runtime work.
- Weight normalization, any packed-weight adjustment, Mode2 braiding, tensor
  allocation, and scale-range reductions all occur before CUDA Graph capture
  and are excluded for both implementations.  The graph/paired/stage harnesses
  now carry separate W13/W2 expert-scale tensors for TP4 and TP8.
- First gate normal-range equivalence, a wider-than-eleven range that exercises
  packed FP4 adjustment, TP4 split-K=4/2, and TP8 shape.  Only then measure the
  branch-free normalized core against the exact shared-LUT winner.

### WGMMA iteration 41b — offline normalization selected

- Correctness passes TP4 split-K=4, forced split-K=2 skew and TP8 shape for
  normal raw codes 125..128.  A TP4 case with raw codes 118..130 also passes,
  explicitly exercising the packed-E2M1 adjustment needed when an expert's
  exponent span exceeds eleven.  W13/W2 use 54 registers/thread and 2 KiB
  static shared memory, versus 56 registers and 4 KiB for the shared-LUT
  control, with no spill.
- Two graph-internal candidate windows around a same-source control improve
  both W13 and W2 at M8/M32/M128.  Local total improves 1.67-3.94% in the
  conservative first window and 1.98-3.86% in the confirmation window.
- TP4 candidate/control/candidate distributed screens each use 400 separately
  cold-L2 CUDA-Graph samples per implementation and M, including the same
  SGLang CustomAllReduceV2 in both graphs.  Candidate A/control custom latency
  improves at every M by 2.14-4.18% (3.72% geomean); paired-Humming
  normalization gives 3.92%.  Candidate B confirms every point with a 4.55%
  direct geomean improvement.
- Select normalized scales as the new default.  Normalization, optional packed
  weight adjustment, braiding and allocations remain model-load work outside
  the captured/timed graph.  Runtime kernels are branch-free and retain the
  accepted S2R schedule.  Use Candidate A versus its adjacent control as the
  conservative causal result; do not headline Candidate B's favorable system
  window.
- Evidence:
  `bench/results/v4_flash_tp_wgmma_normalized_scale_correctness_20260903.log`
  and
  `bench/results/tp4_normalized_scale_coldl2_screen_20260903.log`.

### Post-selection TP8 end-to-end smoke

- The normalized-scale default runs the full TP8 CUDA Graph at M8 across all
  eight H20s, including random k6 route metadata, local route reduction and
  SGLang CustomAllReduceV2.  Twenty individually cold-L2 samples give median
  0.069408 ms; cosine is 0.999991970, relative L2 is 0.004007414, every rank
  is finite, and the all-reduce check passes.
- This is a bounded runnability/correctness gate because GPUs 5-7 are shared;
  TP4 remains the optimization and formal-comparison target.
- Evidence:
  `bench/results/tp8_wgmma_normalized_scale_default_smoke_20260903.log`.

### Post-selection TP4 formal cold-L2 score

- Formal paired run uses TP4 GPUs 1-4, random real k6 route metadata,
  alternating AB/BA complete batches, 10 outer batches x 200 replays, and
  2,000 samples per implementation/M.  Both implementations include the same
  SGLang CustomAllReduceV2 in their captured graphs.  A separate 256 MiB L2
  clear precedes every individual graph replay and is excluded from timing.
- Humming/custom medians (ms) at M8/16/32/64/128 are
  0.090048/0.087840, 0.145760/0.136128, 0.232064/0.225824,
  0.332224/0.317712, and 0.411488/0.406976.  Custom wins every point by
  1.11-7.08%; geometric means are 0.210815/0.203518 ms, or 3.59% speedup
  (custom/Humming=0.965386).
- All finite, cosine and independent-NCCL all-reduce checks pass.  This is the
  new accepted headline result, but remains far short of the 20% goal.  The
  noisy M32 batch spread also reinforces using the full 2,000-sample result,
  not the favorable 4x100 screening window, for claims.
- Evidence:
  `bench/results/tp4_paired_normalized_scale_default_coldl2_formal_20260903.log`.

### Post-normalization local stage budget

- Graph-internal event nodes over 100 separately cold-L2 samples show custom
  local totals of 102.528/148.320/216.544/292.704/353.088 us at
  M8/16/32/64/128, versus Humming 109.936/162.816/240.480/328.464/390.032
  us.  The normalized custom path is 6.74-10.89% faster locally.
- Custom W13+W2 account for 75.1% of M8 and 90.6% of M128 local latency.
  Non-GEMM launch stages are too small to reach the 20% objective alone, so
  further work remains focused on both route-GEMM cores.
- Evidence:
  `bench/results/tp4_normalized_scale_default_stage_budget_coldl2_20260903.log`.

### WGMMA iteration 42a — contiguous offline TMA-tile layout

- Added opt-in `V4_TILED_WEIGHT_LAYOUT=1`.  Model-load preprocessing permutes
  packed MXFP4 bytes from logical `[expert,N,K/2]` rows into contiguous
  `[expert,N128,K128,N-in-tile,packed-K-in-tile]` storage.  Each 8 KiB weight
  TMA transfer then reads one contiguous global tile instead of gathering 64
  bytes from each of 128 rows whose stride is K/2.
- E8M0 scales are likewise tiled by the existing 16-byte/four-K128 quartet,
  with no padding or byte-count change for both TP4 kernels and TP8 W13.  TP8
  W2 keeps its original eight-byte scale rows and scalar scale fallback while
  still using tiled packed weights.
- This is a physical-layout-only transform after numerical normalization and
  Mode2 braiding.  It is outside graph capture/timing and leaves MXFP4 data,
  scale semantics, route metadata, and output math unchanged.  Gate TP4
  split-K=4/2 and TP8 correctness before graph-internal cold-L2 core timing.

### WGMMA iteration 42b — contiguous TMA tiles selected

- Correctness passes TP4 split-K=4, forced split-K=2 skew, and TP8 shape with
  exactly the accepted errors.  Thus both tiled weight/scale TMA coordinates
  and TP8 W2's tiled-weight/scalar-scale combination are covered.
- Graph-internal candidate/control/candidate measurements use 100 separately
  cold-L2 samples at M8/M32/M128.  Candidate A/control local-total ratios are
  0.945/0.980/0.986; both W13 and W2 improve at every point.  Candidate B
  repeats all three directions.
- TP4 distributed candidate/control/candidate screens use 400 individually
  cold-L2 CUDA-Graph samples per implementation/M and the same
  CustomAllReduceV2 in both graphs.  Conservative Candidate A/control custom
  ratios are 0.938478/0.963120/0.974130/0.974059/0.973095, a 0.964478
  geomean or 3.68% speedup.  Normalizing by each run's paired Humming window
  gives 0.964132 geomean or 3.72%.  Every M improves; Candidate B confirms.
- Select tiled layout as the new default and require a fresh 10x200 formal
  paired run before updating the headline Humming gap.  Physical permutation
  remains one-time model-load work excluded from graph replay timing.
- Evidence:
  `bench/results/v4_flash_tp_wgmma_tiled_layout_correctness_20260903.log`,
  `bench/results/tp4_tiled_weight_layout_local_coldl2_screen_20260903.log`,
  and `bench/results/tp4_tiled_layout_{candidate_a,control,candidate_b}_coldl2_screen_20260903.log`.

### Post-iteration-42 TP4 formal cold-L2 score

- Formal paired protocol is unchanged: random real k6 routes, TP4 GPUs 1-4,
  same CustomAllReduceV2, alternating AB/BA complete batches, 10x200 replays,
  and a separate excluded 256 MiB L2 clear before every implementation replay.
- Humming/custom medians (ms) at M8/16/32/64/128 are
  0.090016/0.082400, 0.145664/0.130944, 0.232576/0.218464,
  0.337200/0.311648, and 0.408784/0.396704.  Custom wins every point by
  3.05-11.24%; geometric means are 0.211214/0.196293 ms, so
  custom/Humming=0.929354 and Humming/custom=1.076017 (7.60% speedup).
- Versus the immediately preceding normalized-scale formal custom geomean
  0.203518 ms, tiled storage improves 3.68%, matching its conservative
  screen estimate.  All finite, cosine and independent-NCCL all-reduce checks
  pass.  The accepted result is still well short of the 20% objective.
- Evidence:
  `bench/results/tp4_paired_tiled_layout_default_coldl2_formal_20260903.log`.

### WGMMA iteration 43a — share normalized scale LUTs within lane quads

- Added opt-in `V4_QUAD_LUT_SHUFFLE=1`.  Four lanes map to the same output row
  and E8M0 group for each K32 step, but the winner redundantly performs four
  shared-byte loads and four affine LUT syntheses.  The candidate lets the
  first lane load/synthesize one `uint2`, then broadcasts its two words to the
  other three lanes with warp shuffles.
- Packed FP4 loads, PRMT/sign reconstruction, WGMMA issue, tiled global TMA
  traffic and output math are unchanged.  Gate all TP4 split modes and TP8
  before graph-internal candidate/control/candidate cold-L2 timing; the extra
  shuffles may cost more than the eliminated shared/integer work.
