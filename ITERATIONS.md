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
