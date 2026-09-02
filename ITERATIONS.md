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
