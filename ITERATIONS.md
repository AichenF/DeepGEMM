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

### WGMMA iteration 43b — lane-quad LUT sharing rejected

- TP4 split-K=4, forced split-K=2 skew and TP8 shape reproduce the selected
  numerical errors.  However, two warp shuffles per synthesized `uint2` extend
  live ranges: TP4 W13/W2 rise from 54 to 64 registers/thread without spill.
- Graph-internal 100-sample cold-L2 medians for candidate/control/candidate
  local totals (us) are M8 108.800/96.688/108.640, M32
  246.912/210.464/247.232, and M128 409.424/346.352/409.392.  Both GEMMs
  regress by double digits at every point, so no distributed screen is
  justified.
- Reject and restore the exact iteration-42 tiled-layout winner.  The raw
  stage log's `custom_tiled_weight_layout=false` is a metadata-only bug: the
  field read an unset environment variable with the old default after tiled
  layout had become kernel-default true.  All candidate/control runs used
  tiled weights; the logger default is corrected for subsequent profiles.
- Evidence:
  `bench/results/v4_flash_tp_wgmma_quad_lut_shuffle_correctness_20260903.log`
  and
  `bench/results/tp4_quad_lut_shuffle_local_coldl2_screen_20260903.log`.

### WGMMA iteration 44a — retune WOUT after tiled storage

- Revisit `V4_WOUT=64` against the selected 128-channel output tile.  Earlier
  row-major weights favored WOUT128 by amortizing CTA setup and activation
  loads, but contiguous offline TMA tiles materially change each CTA's global
  transaction shape.  WOUT64 also lowers accumulator/register pressure and
  exposes twice as many independent CTAs, which may matter at small M.
- This is an existing compiled specialization, not a new math path.  Validate
  tiled-layout coordinates at TP4 split-K=4/2 and TP8, then run graph-internal
  WOUT64/128/64 cold-L2 stage screens at M8/M32/M128.  Only a repeatable core
  win merits an M-dependent dispatch or distributed benchmark.

### WGMMA iteration 44b — WOUT64 remains rejected

- TP4 split-K=4 and forced split-K=2 reproduce the WOUT128 errors.  TP8 runs
  but W2 cosine drops from 0.999997249 to 0.999928707 (relative L2 rises from
  0.00235 to 0.01194), another reason not to use this specialization there.
  WOUT64 lowers TP4 route-GEMM registers from 54 to 45 without spill.
- Graph-internal WOUT64/128/64 100-sample cold-L2 local medians (us) are
  M8 106.864/98.464/106.720, M32 250.800/211.216/249.632, and M128
  414.848/348.288/413.744.  Both W13 and W2 regress at every point; the
  additional CTAs and duplicated setup dominate the lower register footprint.
- Reject without distributed timing.  Keep the exact WOUT128 tiled-layout
  winner and no M-dependent tile dispatch.
- Evidence:
  `bench/results/v4_flash_tp_wgmma_wout64_tiled_correctness_20260903.log` and
  `bench/results/tp4_wout64_tiled_local_coldl2_screen_20260903.log`.

### WGMMA iteration 45a — retune W13 split-K after tiled storage

- Re-sweep legal W13 split-K 2 and 4 under the selected contiguous TMA layout.
  The current auto boundary (split4 through M32, split2 at M64/M128) was
  selected before normalization and tiled storage; those changes can alter
  the balance between parallel waves, TMA latency and partial-reduction cost.
- No kernel source or math changes.  Use identical seeded random routes and
  graph-internal events with a separate excluded 256 MiB L2 clear for every
  sample.  Measure all M values and repeat boundary points before changing
  the pre-capture dispatch policy.

### WGMMA iteration 45b — retain current W13 split policy

- Graph-internal split2/split4/split2 runs use 100 separately cold-L2 samples
  per M/config.  Split4 has the lower W13 median at M8/16/32 in the central
  comparison (45.408/74.576/119.232 us versus split2
  46.416/76.384/120.512 us).  Split2 is lower at M128
  (205.472 then 204.064 us versus split4 207.168 us).
- M64 is noise-sized: the first split2 window is 167.472 us, split4 is
  168.128 us, and the second split2 window is 168.160 us.  There is no robust
  evidence to move the boundary.
- Retain auto split4 through routed_rows=192 (M32 here), then split2 when
  active experts exceed 96.  No distributed timing or source change is
  warranted.
- Evidence:
  `bench/results/tp4_w13_split_post_tiled_local_coldl2_screen_20260903.log`.

### WGMMA iteration 46a — pre-swizzled one-dimensional bulk weight copies

- Added opt-in `V4_BULK_WEIGHT_COPY=1`, requiring tiled storage.  Model-load
  preprocessing additionally materializes the exact 64-byte TMA shared-memory
  swizzle inside each contiguous packed-weight tile.  Runtime can therefore
  replace each tensor-map 2D weight transfer with one linear 8 KiB
  `cp.async.bulk` transaction while preserving the LDS addresses used by the
  dequantizer.
- The contiguous 2 KiB scale quartet uses the same linear bulk instruction.
  Two weight stages, scale reuse, barriers, S2R/dequant, WGMMA, data bytes and
  output math are unchanged.  The experiment isolates tensor-map descriptor
  and coordinate overhead; all pre-swizzling occurs before graph capture.
- Gate TP4 split-K=4/2 and TP8 correctness, including TP8 W2's scalar scale
  fallback, before candidate/control/candidate graph-internal cold-L2 timing.

### WGMMA iteration 46b — accept pre-swizzled bulk copies

- Correctness passes for TP4 split-K=4, forced TP4 split-K=2 and TP8.  The
  compiled kernels retain the selected register footprint (54 registers for
  TP4 W13/W2 and 48 for TP8 W2, with no spills).
- Graph-internal candidate/control/candidate cold-L2 local totals (us) are
  M8 96.736/97.792/96.592, M32 208.912/211.680/208.576, and
  M128 339.904/348.272/339.696.  The repeatable gain is concentrated in W13
  (2.8–3.2%); W2 is neutral at small M and improves 1.7% at M128.
- The TP4 distributed screen uses random real k6 routes, the same SGLang
  CustomAllReduceV2, alternating Humming/custom batches, four batches of 100
  graph replays, and a separate excluded 256 MiB L2 clear before every replay.
  Candidate A/control/candidate B custom geometric means are
  0.186910/0.192937/0.187155 ms.  Their direct reductions are 3.12% and
  3.00%; normalizing each run by its paired Humming result still gives 1.79%
  and 2.24%.  Every M point is positive in both candidate windows.
- Accept `V4_BULK_WEIGHT_COPY=1` as the default.  This moves the screened
  Humming/custom ratio from 1.10696 in the same-source control to 1.12708 and
  1.13235 in the two candidate windows.  Run a default-path correctness gate,
  TP8 graph smoke, and the formal TP4 10x200 cold-L2 benchmark before updating
  the headline score.
- Evidence:
  `bench/results/v4_flash_tp_wgmma_bulk_copy_correctness_20260903.log`,
  `bench/results/tp4_bulk_weight_copy_local_coldl2_screen_20260903.log`, and
  `bench/results/tp4_bulk_copy_{candidate_a,control,candidate_b}_coldl2_screen_20260903.log`.

### Post-iteration-46 TP8 end-to-end smoke

- The default bulk-copy path runs the full TP8 CUDA Graph at M8 on all eight
  H20s.  It includes random real k6 route metadata, local route reduction and
  SGLang CustomAllReduceV2, with a separate excluded 256 MiB L2 flush before
  each of 20 timed replays.
- Median max-rank latency is 0.066320 ms.  Independent local recomputation plus
  NCCL-reference validation gives minimum-rank cosine 0.999991970, relative L2
  0.004007414, finite output on every rank and `allreduce_ok=true`.
- This is a bounded TP8 runnability gate, not a formal score; compared with the
  previous 0.069408 ms TP8 smoke it is 4.45% lower, but the sample is too small
  for selection.  TP4 remains the primary optimization target.
- Evidence:
  `bench/results/tp8_wgmma_bulk_copy_default_smoke_20260903.log`.

### Post-iteration-46 TP4 formal cold-L2 score

- Formal paired protocol remains random real k6 routes on TP4 GPUs 1–4, both
  full CUDA Graphs including the same SGLang CustomAllReduceV2, alternating
  complete Humming/custom batches, 10 outer batches x 200 replays, and a
  separate excluded 256 MiB L2 clear before every implementation replay.
- Humming/custom medians (ms) at M8/16/32/64/128 are
  0.090048/0.080800, 0.145664/0.128768, 0.232272/0.212304,
  0.342592/0.298672, and 0.408608/0.385072.  Custom wins every point; all
  finite, cosine and independent NCCL all-reduce validations pass.
- Geometric means are 0.211827/0.190978 ms, so custom/Humming=0.901574 and
  Humming/custom=1.109171: 10.92% speedup (or 9.84% latency reduction).
  Relative to the previous tiled-layout formal score of 1.076017, the accepted
  bulk-copy default improves the Humming/custom ratio by 3.08%.  It remains
  well short of the 20% objective.
- Evidence:
  `bench/results/tp4_paired_bulk_copy_default_coldl2_formal_20260903.log`.

### WGMMA iteration 47a — fuse TP route alignment with input quantization

- The selected local stage profile still spends roughly 8 us in SGLang's
  E=256 route alignment and 6–8 us in the independent H=4096 group-128 FP8
  input quantizer.  The two operations have no data dependency but currently
  serialize as separate graph kernels.
- Add an opt-in TP-specialized preparation kernel for fixed E=256, top-k=6,
  block-M=8 and H=4096.  CTA 0 builds the expert histogram, padded offsets,
  expert-block IDs and route permutation in shared memory; the remaining CTA
  work performs the same BF16-to-E4M3 group-128 quantization.  Both halves run
  in one launch, with preallocated graph-stable output buffers.
- Keep `V4_FUSED_ROUTE_QUANT=0` by default.  First require route-contract
  checks against `moe_align_block_size`, exact/near-exact quantizer comparison,
  full TP4 split-K=4/2 and TP8 numerical checks, then graph-internal cold-L2
  candidate/control/candidate timing.  The Humming baseline remains unchanged.

### WGMMA iteration 47b — serial-prefix fused preparation rejected

- The first implementation is numerically exact where required: random and
  balanced route contracts match semantically through M128, its FP8 bytes and
  FP32 scales are bitwise identical to Humming `quant_input`, and complete TP4
  split-K=4/2 plus TP8-shape output checks retain the selected errors.
- Its CTA-0 alignment prefix is serial over 256 experts.  Graph-internal
  cold-L2 fused-stage medians at M8/M32/M128 are 14.368/15.040/17.920 us,
  versus 13.536/13.728/16.160 us for the control's separate align plus quant
  medians.  Local total medians regress from 95.648/208.240/338.816 us to
  96.480/208.640/340.896 us even though the candidate graph contains one less
  profiling event node.
- Reject this implementation as a selectable default and preserve the raw
  result.  Retain the opt-in framework for one bounded repair: use a
  256-thread block-wide exclusive scan and quantize two group-128 slices per
  CTA, removing both the serial prefix and half of the quant CTAs.
- Evidence:
  `bench/results/v4_flash_tp_wgmma_fused_route_quant_{first_,}correctness_20260903.log`
  and `bench/results/tp4_fused_route_quant_stage_coldl2_screen_20260903.log`.

### WGMMA iteration 47c — accept parallel-scan fused preparation

- Replace CTA 0's serial expert loop with a 256-thread CUB exclusive scan and
  let each CTA quantize two independent group-128 slices.  Route semantics
  pass through the maximum M128 balanced case; FP8 bytes and scales remain
  bitwise identical to Humming `quant_input`.  Full TP4 and TP8-shape output
  checks retain the selected W2 cosine/relative-L2 values.
- Graph-internal candidate/control/candidate local total medians (us) are
  M8 88.656/96.064/88.528, M32 200.384/207.648/200.544, and
  M128 331.520/339.168/331.776.  The fused preparation itself is
  6.432/6.768/8.832 us versus 13.664/13.856/16.160 us for the two control
  stages in the central window.
- Distributed TP4 candidate A/control/candidate B custom geometric means are
  0.182974/0.186461/0.184167 ms.  Direct reductions are 1.87% and 1.23%; after
  normalizing each window by its paired Humming graph they remain 2.59% and
  1.73%.  Candidate Humming/custom ratios are 1.15905 and 1.14897.
- Accept `V4_FUSED_ROUTE_QUANT=1` as the default and require a TP8 full-graph
  smoke plus the formal TP4 10x200 cold-L2 score.  This optimization is valid
  only for the fixed V4 Flash TP contract (E=256, top-k=6, block-M=8,
  H=4096); unsupported shapes must not call it.
- Evidence:
  `bench/results/v4_flash_tp_wgmma_parallel_fused_route_quant_{first_,}correctness_20260903.log`,
  `bench/results/tp4_parallel_fused_route_quant_stage_coldl2_screen_20260903.log`,
  and
  `bench/results/tp4_parallel_fused_route_quant_{candidate_a,control,candidate_b}_coldl2_screen_20260903.log`.

### Post-iteration-47 TP8 smoke and TP4 formal score

- The selected default runs the true eight-rank M8 CUDA Graph in 0.062608 ms
  median over 20 individually cold-L2 samples.  Minimum-rank cosine is
  0.999991970, relative L2 is 0.004007414, all outputs are finite and the
  independent NCCL all-reduce check passes.  This remains a runnability gate.
- Formal TP4 10x200 Humming/custom medians (ms) at M8/16/32/64/128 are
  0.090144/0.078144, 0.145824/0.125152, 0.233584/0.209856,
  0.336496/0.301216, and 0.407968/0.385184.  Humming/custom speedups are
  1.15356/1.16518/1.11307/1.11713/1.05915; every point remains a win.
- Geometric means are 0.211331/0.188521 ms, giving custom/Humming=0.892067
  and Humming/custom=1.120993: 12.10% speedup (10.79% latency reduction).
  Versus the pre-fusion formal ratio 1.109171, the ratio improves 1.07%.
  Correctness and the same excluded 256 MiB per-replay cold-L2 policy pass.
  The result remains short of the 20% goal, with the large-M W13/W2 kernels
  dominating the remaining gap.
- Evidence:
  `bench/results/tp8_wgmma_fused_route_quant_default_smoke_20260903.log` and
  `bench/results/tp4_paired_fused_route_quant_default_coldl2_formal_20260903.log`.

### WGMMA iteration 48a — fixed-shape vectorized local k6 reduction

- The selected custom path still uses SGLang's generic Triton
  `moe_fused_mul_sum`; its cold-L2 stage costs about 5.7–7.5 us and launches
  roughly 4M 256-thread CTAs for this BF16/top-k=6/H=4096 contract.
- Add an opt-in custom reduction with one 256-thread CTA per token.  Each lane
  traverses the fixed hidden row using `bfloat162` vector loads, accumulates
  the six route values in FP32 with the same FP32 weights and routed factor,
  and emits BF16.  W2 output layout, MXFP4 math and CustomAllReduceV2 are
  unchanged.
- Keep `V4_CUSTOM_ROUTE_REDUCE=0` by default.  Require full-path numerical
  checks and graph-internal candidate/control/candidate cold-L2 timing before
  any distributed comparison; reject if reduced CTA count under-fills H20 or
  changes output beyond the existing tolerance.

### WGMMA iteration 48b — one-CTA-per-token reduction rejected

- The fixed kernel is bitwise identical to SGLang `moe_fused_mul_sum` on the
  full M8 route output, and end-to-end W2 cosine remains 0.999997272.  This
  rules out numerical differences as an explanation for timing.
- It under-fills the H20 and makes every lane execute a long serial hidden
  loop.  At M8/M32 the candidate/control local-reduction medians are
  7.840/5.744 and 9.600/6.592 us; corresponding local total medians regress
  from 89.536/201.360 to 91.344/203.344 us (2.02%/0.99%).  The candidate's
  unpaired M128 reduction is also 10.240 us, already worse than recent
  7.4-us controls.
- Stop the remaining local repeats, do not spend a distributed run, and
  remove the candidate path to restore the exact iteration-47 winner.  Fewer
  CTAs are not useful here; any future reduction work must retain enough
  independent hidden tiles to occupy 78 SMs.
- Evidence:
  `bench/results/v4_flash_tp_wgmma_fixed_k6_reduce_{first_,}correctness_20260903.log`
  and `bench/results/tp4_fixed_k6_reduce_stage_coldl2_screen_20260903.log`.

### WGMMA iteration 49a — pair activation-quant groups per CTA

- The fused W13 reduction/SwiGLU/group-128 FP8 quantizer still launches one
  128-thread CTA per route/group.  Test an opt-in 256-thread specialization
  that handles two independent groups per CTA, halving CTA count while keeping
  each 128-lane reduction subgroup and every arithmetic operation unchanged.
- Keep `V4_ACT_QUANT_PAIR=0` by default and retain both implementations in one
  JIT module for same-binary A/B timing.  Gate TP4/TP8 output numerics, then
  compare activation-stage and total cold-L2 medians.  Do not distribute-test
  a noise-sized or mixed-sign local result.

### WGMMA iteration 49b — paired activation quant rejected

- The candidate passes the full M8 numerical gate: preparation remains
  bitwise exact, W13/activation/W2 cosines are
  0.999999982/0.999999289/0.999997272, and all outputs are finite.
- In the graph-internal candidate/control/candidate screen, activation-quant
  medians are 5.984/6.112/5.984 us at M8, 6.656/6.976/6.656 us at M32, and
  8.960/8.896/8.960 us at M128.  Halving the CTA count helps the isolated
  stage only for the small shapes and slightly hurts M128; local-pipeline
  medians differ by at most about 0.35% and are dominated by noise elsewhere.
- The true four-rank CUDA-Graph candidate A/control/candidate B custom
  geometric means are 0.184652/0.183510/0.184148 ms over 400 independently
  cold-L2 samples per M.  The candidate is therefore 0.62% and 0.35% slower
  than the central control.  Candidate A regresses every M, while candidate B
  still regresses M64/M128 by 0.21%/1.76%.
- Reject the specialization and restore the exact iteration-47 winner.  The
  result also confirms that reducing launch/CTA overhead in this already
  fused stage cannot materially close the remaining gap; future work should
  target W13/W2 data movement or cross-stage fusion.
- Evidence:
  `bench/results/v4_flash_tp_wgmma_act_quant_pair_correctness_20260903.log`,
  `bench/results/tp4_act_quant_pair_stage_coldl2_screen_v2_20260903.log`, and
  `bench/results/tp4_act_quant_pair_{candidate_a,control,candidate_b}_coldl2_screen_20260903.log`.

### WGMMA iteration 50a — interleave bulk weight and scale records

- The selected tiled path issues a separate linear bulk copy for every 8 KiB
  packed-weight tile and every 2 KiB scale quartet.  Test an opt-in model-load
  layout that places each required scale quartet immediately after the weight
  tile which prefetches it, so one 10 KiB `cp.async.bulk` transaction and one
  mbarrier completion replace the two independent transactions.
- Preserve two weight stages and two scale buffers in the same 20 KiB total
  shared footprint.  With legal W13 split boundaries aligned to K-tile 8,
  records 0 and 3 of every eight-tile block carry the even and odd scale
  quartets; TP4 W2 carries its sole quartet with record 0.  TP8 W2 K=256 keeps
  the existing scalar-scale fallback unchanged.
- Keep `V4_INTERLEAVED_BULK_COPY=0` by default and require the existing TP4
  split-K=4/2 and TP8 numerical gates.  Then compare W13/W2 and total local
  cold-L2 medians candidate/control/candidate before any distributed run.

### WGMMA iteration 50b — interleaved-copy core screen passes

- TP4 split-K=4, forced split-K=2, and TP8-shape tests reproduce the selected
  W13/activation/W2 numerical errors.  TP8 W2 continues to use the unchanged
  scalar-scale path; only K>=512 specializations consume interleaved records.
- Graph-internal candidate A/control/candidate B local totals (us) are
  M8 88.272/88.608/88.336, M32 199.360/200.336/199.104, and
  M128 330.512/331.248/330.256.  Both candidate windows improve every total
  by 0.22-0.61% despite the small effect size.
- The mechanism is concentrated in the core at useful occupancy: both M32
  W13 and W2 improve in each candidate window (0.34-0.49% and 0.78-0.98%),
  while M128 W2 improves 1.07-1.08%.  M8 and M128 W13 contain mixed/noise-size
  stage effects, so require a distributed A/control/A result before selection.
- Evidence:
  `bench/results/v4_flash_tp_wgmma_interleaved_bulk_correctness_20260903.log`,
  `bench/results/v4_flash_tp_wgmma_interleaved_bulk_additional_correctness_20260903.log`,
  and
  `bench/results/tp4_interleaved_bulk_stage_coldl2_screen_20260903.log`.

### WGMMA iteration 50c — select interleaved bulk layout

- Four-rank candidate A/control/candidate B custom geometric means are
  0.181630/0.184928/0.183210 ms across M8/16/32/64/128, 400 separately
  cold-L2 graph samples per implementation/M, and the same captured SGLang
  `CustomAllReduceV2`.  Direct custom latency reductions are 1.78% and 0.93%.
- Normalizing every window by its paired Humming graph retains gains of 1.07%
  and 0.50%; candidate Humming/custom ratios are 1.15438 and 1.14785 versus
  control 1.14214.  Both independent windows therefore confirm the smaller
  core-screen effect despite large-system noise at M128.
- Select `V4_INTERLEAVED_BULK_COPY=1` as the default.  Treat the second
  window's 0.50% normalized gain as the conservative causal result, and gate
  the new headline on a TP8 full-graph smoke plus TP4 10x200 formal run.
- Evidence:
  `bench/results/tp4_interleaved_bulk_{candidate_a,control,candidate_b}_coldl2_screen_20260903.log`.

### Post-iteration-50 TP8 smoke and TP4 formal score

- The selected default runs the true eight-rank M8 CUDA Graph in 0.062800 ms
  median over 20 individually cold-L2 samples.  Minimum-rank cosine is
  0.999991240, relative L2 is 0.004185668, all outputs are finite, and the
  independent NCCL all-reduce check passes.  TP8 W2 retains its original
  scalar-scale storage and runtime path.
- Formal TP4 10x200 Humming/custom medians (ms) at M8/16/32/64/128 are
  0.090080/0.077664, 0.145760/0.124096, 0.232752/0.204256,
  0.340896/0.297328, and 0.409120/0.380640.  Humming/custom speedups are
  1.15987/1.17457/1.13951/1.14653/1.07482; every point remains a win.
- Geometric means are 0.211800/0.186029 ms, giving custom/Humming=0.878322
  and Humming/custom=1.138534: 13.85% speedup (12.17% latency reduction).
  Versus the iteration-47 formal score, custom latency falls 1.32%, Humming
  shifts only +0.22%, and the speedup ratio improves 1.56%.
- Correctness and the same excluded 256 MiB per-replay cold-L2 policy pass.
  The selected result remains 6.15 percentage points short of the 20% target;
  continue with core W13/W2 optimization rather than launch micro-tuning.
- Evidence:
  `bench/results/tp8_wgmma_interleaved_bulk_default_smoke_20260903.log` and
  `bench/results/tp4_paired_interleaved_bulk_default_coldl2_formal_20260903.log`.

### WGMMA iteration 51a — L2-prefetch future interleaved records

- Two shared stages avoid the severe residency loss of the rejected deeper
  shared-memory pipelines, but each cold weight record still waits on HBM.
  Test an opt-in Hopper `cp.async.bulk.prefetch.L2.global` hint for the record
  two K128 iterations ahead while retaining exactly two shared stages.
- The prefetch executes inside the timed GEMM, so it does not violate the cold
  policy or hide bytes outside measurement.  Every record is still fetched
  once from HBM; the later shared-memory bulk copy may consume it from L2 at
  the cost of additional L2 traffic.  Weight/scale storage, math, barriers,
  shared memory, grid, and epilogue remain unchanged.
- Keep `V4_BULK_L2_PREFETCH=0` by default.  First gate TP4 split-K=4/2 and TP8,
  then require candidate/control/candidate W13 and W2 cold-L2 evidence; reject
  immediately if the hint merely doubles cache traffic without hiding waits.

### WGMMA iteration 51b — software L2 prefetch rejected

- TP4 split-K=4, forced split-K=2, and TP8-shape numerical gates pass after
  explicitly excluding TP8 W2's non-interleaved K=256 specialization from the
  hint.  The repair is preserved separately from the initial experiment.
- Candidate A/control/candidate B local total medians (us) are
  M8 93.376/87.984/93.168, M32 204.848/199.712/204.800, and
  M128 339.744/329.744/339.584.  Both candidate windows regress by roughly
  5.9%/2.6%/3.0%; W13 alone regresses about 4-8%, while W2 also never wins.
- Reject without distributed timing and remove the hint path.  The prefetch
  duplicates L2/TMA traffic and contention rather than hiding the compulsory
  cold HBM access; deeper logical prefetch without shared capacity is not a
  substitute for the hardware bulk-copy pipeline here.
- Evidence:
  `bench/results/v4_flash_tp_wgmma_bulk_l2_prefetch2_correctness_20260903.log`,
  `bench/results/v4_flash_tp_wgmma_bulk_l2_prefetch2_repair_correctness_20260903.log`,
  and
  `bench/results/tp4_bulk_l2_prefetch2_stage_coldl2_screen_20260903.log`.

### WGMMA iteration 52a — direct BF16 atomic W2 accumulation

- Revisit the accepted route-output epilogue with a new opt-in path, not the
  previously rejected FP32 atomic path.  Emit each W2 route result as BF16,
  apply its FP32 top-k weight and routed factor, then use native Hopper BF16
  global atomics directly into the zeroed `[M,H]` local output.
- This removes the `[M,6,H]` BF16 route buffer and SGLang local reduction from
  the custom graph, but preserves the baseline unchanged.  It intentionally
  trades six contended BF16 atomic contributions per output for less global
  traffic and one fewer kernel.  A captured BF16 output memset remains timed.
- Keep `V4_W2_BF16_ATOMIC=0` by default.  Require TP4/TP8 numerical gates and
  compare the combined zero+W2 atomic stage against control W2+local-reduce
  totals under per-replay cold L2; stop if BF16 summation error or contention
  outweighs the eliminated route tensor.

### WGMMA iteration 52b — direct BF16 atomic W2 rejected

- TP4 M8/M128 balanced and M128 fully skewed routes plus TP8-shape M8
  balanced/M128 skewed numerical gates all pass.  W2 cosine remains
  0.9999929-0.9999939 with relative L2 0.00349-0.00378 and finite output, so
  native BF16 atomics are numerically usable for this benchmark.
- Cold-L2 candidate A/control/candidate B local total medians average to
  91.936/88.080 us at M8, 214.728/198.688 us at M32, and
  378.152/330.768 us at M128.  The BF16-atomic path therefore regresses by
  4.38%/8.07%/14.33%, consistently in both candidate windows.
- The combined zero+W2-atomic medians are 35.176/84.904/162.432 us versus
  control W2+SGLang-reduce totals 31.248/69.328/114.592 us.  Contention cost
  grows from 12.57% to 41.75% as M increases and overwhelms the removed route
  tensor and reduction launch.  Reject without distributed timing, retain
  `V4_W2_BF16_ATOMIC=0`, and restore the iteration-50 runtime exactly.
- Evidence:
  `bench/results/iter52_bf16_atomic_correctness_20260903.log` and
  `bench/results/iter52_bf16_atomic_stage_aba_coldl2_20260903.log`.

### WGMMA iteration 53a — leader-only TMA mbarrier polling

- The selected route GEMM currently has all 128 lanes execute the same
  `mbarrier.try_wait.parity` polling loop for each packed weight stage, then
  immediately performs a CTA-wide barrier.  During cold-HBM waits this can
  consume scheduler issue slots with redundant polling from four warps.
- Add opt-in `V4_LEADER_MBAR_WAIT=1`: only lane 0 performs the acquire wait;
  every lane then rendezvous at the unchanged CTA barrier before reading the
  TMA-filled shared stage.  Grid, two-stage layout, copied bytes, dequant,
  WGMMA, registers holding math state, and output semantics remain unchanged.
- Keep the default off.  First require TP4 split-K=4/2 and TP8 correctness,
  because this experiment relies on the leader wait followed by `bar.sync`
  publishing the async-proxy writes to the CTA.  If correct, use cold-L2
  candidate/control/candidate stage timing at M8/M32/M128; only a repeatable
  W13/W2 reduction proceeds to distributed timing.

### WGMMA iteration 53b — isolate leader wait to long-K W13

- TP4 split-K=4/2 and TP8-shape numerical gates reproduce the selected
  errors exactly, confirming that a lane-0 acquire wait followed by
  `bar.sync` safely publishes the TMA-written shared stage on H20.
- With leader waits enabled for both route GEMMs, candidate A/control/
  candidate B total medians average to 88.672/88.448 us at M8,
  200.200/199.360 us at M32, and 331.224/329.904 us at M128: small but
  consistent 0.25%/0.42%/0.40% regressions.
- The layer split is equally consistent.  W13 improves by
  0.69%/0.95%/0.87%, while W2 regresses by 2.92%/3.02%/2.99%.  Long-K W13
  benefits from removing redundant polling; short-K W2 instead pays for the
  leader warp plus CTA-barrier wakeup latency.
- Preserve this evidence and make one bounded repair: apply
  `V4_LEADER_MBAR_WAIT=1` only when `IsW13`, leaving W2's exact selected wait
  path unchanged.  Re-run numerical gates and A/control/A cold-L2 stage
  timing; the expected W13-only savings are roughly 0.3/1.1/1.7 us.
- Evidence:
  `bench/results/iter53_leader_mbar_wait_correctness_20260903.log` and
  `bench/results/iter53_leader_mbar_wait_stage_aba_coldl2_20260903.log`.

### WGMMA iteration 53c — W13-only leader wait passes core screen

- The repaired specialization again passes TP4 split-K=4/2 and TP8-shape
  correctness with errors identical to the selected kernel.  W2 now compiles
  and executes the exact all-warp mbarrier wait path regardless of the flag.
- W13-only candidate A/control/candidate B local total medians average to
  87.704/88.256 us at M8, 198.016/199.136 us at M32, and
  328.064/329.680 us at M128.  The candidate improves the full local path by
  0.63%/0.56%/0.49%, consistently in both surrounding windows.
- W13 medians improve by 0.90%/0.89%/0.86%; W2 medians stay within 0.16% of
  control.  Proceed to TP4 candidate/control/candidate full-graph timing with
  the same SGLang `CustomAllReduceV2` and per-replay excluded cold-L2 clear.
  The effect is small enough that paired-Humming normalization is required.
- Evidence:
  `bench/results/iter53_w13_leader_wait_correctness_20260903.log` and
  `bench/results/iter53_w13_leader_wait_stage_aba_coldl2_20260903.log`.

### WGMMA iteration 53d — select W13-only leader wait

- TP4 candidate A/control/candidate B full-graph geometric means are
  0.182771/0.183345/0.183502 ms over 400 separately cold-L2 samples per M.
  Candidate A improves 0.31%; candidate B is 0.09% slower overall because
  its M128 window enters a system-level high-latency mode.  Both candidates
  improve M8-M32, and M64 is within 0.15% in all three windows.
- Because four-batch M128 medians vary by tens of microseconds in both
  implementations, run a dedicated replay-interleaved 10x200 audit instead
  of selecting a favorable short window.  Candidate/control M128 medians are
  0.342352/0.343840 ms, a 0.43% candidate reduction.  Their paired
  Humming/custom ratios are 1.16937/1.16054, a 0.76% normalized improvement.
- This confirms the 0.49% local M128 gain and the mechanism is isolated to
  W13; W2 and communication are unchanged.  Select
  `V4_LEADER_MBAR_WAIT=1` as the default, applying only to W13.  Require a
  TP8 full-graph smoke and TP4 10x200 five-point formal run before updating
  the headline; expect only a sub-percent improvement.
- Evidence:
  `bench/results/tp4_iter53_w13_leader_{candidate_a,control,candidate_b}_coldl2_screen_20260903.log`
  and
  `bench/results/tp4_iter53_w13_leader_m128_{candidate,control}_coldl2_audit_20260903.log`.

### Post-iteration-53 TP8 smoke and TP4 formal score

- The selected default runs the true TP8 M8 CUDA Graph in 0.062288 ms median
  over 20 individually cold-L2 samples.  Minimum-rank cosine is 0.999991970,
  relative L2 is 0.004007414, every output is finite, and the independent
  NCCL all-reduce check passes.
- Formal TP4 10x200 Humming/custom medians (ms) at M8/16/32/64/128 are
  0.090144/0.077184, 0.145824/0.123456, 0.235952/0.204800,
  0.343424/0.302096, and 0.408320/0.380144.  Humming/custom speedups are
  1.16791/1.18118/1.15211/1.13680/1.07412; every point remains a win.
- Geometric means are 0.212659/0.186248 ms, giving custom/Humming=0.875807
  and Humming/custom=1.141804: 14.18% speedup (12.42% latency reduction).
  The headline ratio rises 0.29% relative to iteration 50's 1.138534.
- Do not attribute the whole headline shift to the candidate: custom geomean
  is 0.12% higher than the previous formal window while Humming is 0.41%
  higher.  The controlled core/distributed audits support only the expected
  0.3-0.5% causal gain.  At this window's baseline, reaching 1.20x requires
  custom geomean about 0.177216 ms, another 9.03 us or 4.85% reduction.
- Evidence:
  `bench/results/tp8_wgmma_w13_leader_default_smoke_20260903.log` and
  `bench/results/tp4_paired_w13_leader_default_coldl2_formal_20260903.log`.

### WGMMA iteration 54a — one-multiply normalized LUT synthesis

- Offline normalization guarantees every timed-kernel E8M0 offset is in
  1..12.  The selected affine LUT generator nevertheless computes
  `e*0x08080800` and `e*0x08080808` independently for every output row and
  K32 step.
- For this bounded domain, every byte of `e*0x08080808` is exactly `8e` with
  no cross-byte carry.  Clearing its low byte is therefore bit-identical to
  `e*0x08080800`.  Test a generator with one multiply, one mask, and the same
  two adds, removing one dependent IMAD per synthesized `uint2` without
  changing packed weights, scales, WGMMA operands, grid, or memory traffic.
- Keep an opt-in `V4_SINGLE_IMAD_LUT=0` first.  Require exact LUT-word
  equivalence for all normalized codes plus TP4 split-K=4/2 and TP8 numerical
  gates.  Then use candidate/control/candidate cold-L2 W13/W2 stage timing;
  reject if compiler code generation, register live ranges, or integer-pipe
  scheduling erase the source-level instruction reduction.

### WGMMA iteration 54b — one-multiply LUT rejected

- The two formulas are word-exact for all normalized offsets 1..12, and TP4
  split-K=4/2 plus TP8-shape numerical gates reproduce the selected errors.
  This isolates performance from correctness and data-layout changes.
- Candidate A/control/candidate B local total medians average to
  91.632/87.712 us at M8, 209.976/197.936 us at M32, and
  349.216/328.112 us at M128.  The source-level instruction reduction instead
  regresses the pipeline by 4.47%/6.08%/6.43%; both W13 and W2 lose in every
  candidate window.
- Resource inspection does not show an occupancy explanation: TP4 W13 and W2
  use 53 registers/thread versus 54 for control, with no local spill.  The
  shared affine value serializes the two result words and adds a mask on their
  common dependency; the control's independent constant IMADs expose more
  integer instruction-level parallelism to nvcc/Hopper.
- Reject without distributed timing, remove `V4_SINGLE_IMAD_LUT`, and restore
  the exact iteration-53 source.  Fewer source multiplies are not equivalent
  to a shorter dequant critical path.
- Evidence:
  `bench/results/iter54_single_imad_lut_correctness_20260903.log`,
  `bench/results/iter54_single_imad_lut_stage_aba_coldl2_20260903.log`, and
  `bench/results/iter54_single_imad_lut_resource_control_candidate_20260903.log`.

### WGMMA iteration 55a — FP16 W13 split-K workspace

- W13 currently writes every split accumulator to an FP32
  `[split,routes,2I/TP]` workspace, then the fused activation kernel rereads
  all splits in FP32 before the required BF16 gate/up boundary.  This traffic
  is internal and carries more mantissa than the public pipeline exposes.
- Test opt-in `V4_W13_FP16_PARTIAL=1`: convert each completed split partial to
  FP16 on store, load/convert it back to FP32 for split reduction, then retain
  the exact selected BF16/SwiGLU/BF16/FP8 sequence.  The workspace write/read
  bytes are halved; W13 weights, MXFP4 math, output dtype, routes, grid, and
  split policy remain unchanged.
- FP16 range is the primary risk, and per-split rounding differs from
  Humming's single post-sum BF16 boundary.  Keep FP32 default; gate balanced
  and maximally skewed TP4 split-K=4/2 plus TP8 under the wide scale test,
  require finite outputs and no material cosine/relative-L2 regression, then
  use candidate/control/candidate cold-L2 stage timing.  Reject on either
  numerical risk or a non-repeatable sub-microsecond result.

### WGMMA iteration 55b — FP16 W13 split-K workspace rejected

- Balanced TP4 M8/M128, maximally skewed TP4 M128, TP8-shape M128 skew, and
  the elevated-scale M8 gate all remain finite.  Final W2 cosine/relative-L2
  are 0.99999722--0.99999728 and 0.00233--0.00236, effectively unchanged.
  However, the new rounding is visible upstream: balanced M8 W13 and
  activation relative-L2 rise to 0.0002823 and 0.0016985 versus roughly
  0.000193 and 0.001192 for the selected FP32 workspace.
- Candidate A/control/candidate B cold-L2 local-total medians average to
  87.840/87.968 us at M8, 197.944/197.936 us at M32, and
  328.104/327.952 us at M128.  This is a 0.15% M8 reduction, effectively
  zero at M32, and a 0.05% M128 regression.  The half-size workspace makes
  activation/reduction 0.096--0.160 us faster, but FP16 conversion in W13
  offsets it; W13 itself improves only 0.35% at M8 and regresses at M32/M128.
- Reject without distributed timing.  The result is below the predeclared
  repeatability threshold, supplies no end-to-end gain at the important
  M32/M128 points, and introduces narrower intermediate range plus extra
  rounding.  Restore the exact iteration-53 FP32 workspace implementation.
- Evidence:
  `bench/results/iter55_fp16_w13_partial_correctness_20260903.log` and
  `bench/results/iter55_fp16_w13_partial_stage_aba_coldl2_20260903.log`.

### WGMMA iteration 56a — clustered W13 reduction/activation plan

- The remaining actionable W13 boundary is structural: split-K CTAs write an
  FP32 global workspace, then a separate 128-thread kernel rereads all gate/up
  splits, applies the BF16/SwiGLU/BF16 contract, and emits group-128 FP8.
  Local timing assigns 6--9 us to that second launch, while iteration 55 shows
  that merely narrowing the workspace cannot remove it.
- Prototype an opt-in Hopper thread-block-cluster W13 path at `WOUT=128`.
  One cluster contains both the gate and up tile and every K split: cluster
  size is `2*SplitK` (8 for the small-M split-4 policy, 4 for split-2).  Each
  CTA computes the unchanged RS-WGMMA tile, places its FP32 8x128 accumulator
  in its now-dead dynamic shared-memory stages, and synchronizes the cluster.
  DSM consumers then sum the original FP32 splits, preserve the exact
  BF16/SwiGLU/BF16 sequence, reduce the group-128 scale, and emit the W2 FP8
  activation directly.  Assign one routed row per CTA for cluster-8 and two
  per CTA for cluster-4 so no single leader serializes all eight rows.
- A second cluster barrier must keep every remote shared allocation alive
  until DSM reads finish.  Reorder only the W13 grid into adjacent
  `(gate/up, split)` groups; weights, route metadata, active-expert policy,
  MXFP4 dequantization, WGMMA math, W2, reduction, all-reduce, and cold-L2
  protocol remain unchanged.  Launch through `cudaLaunchKernelEx` so CUDA
  Graph records the cluster dimension.
- Gate TP4 balanced/skew split-4/split-2, TP8-shape split-2, and elevated
  scales before timing.  Then compare fused `W13+activation` and complete
  local/full graphs against the exact iteration-53 control.  Reject if DSM
  synchronization or cluster residency consumes the removed launch/traffic;
  do not select from a single favorable cold-L2 window.

### WGMMA iteration 56b — clustered W13 reduction/activation rejected

- The extended launch and DSM protocol are functionally sound.  TP4 M8
  split-4, TP4 M128 balanced split-2 and skew split-4, TP8-shape M128 skew
  split-2, and the elevated-scale M8 split-4 test all complete without a
  hang or non-finite value.  Activation cosine is
  0.99999927--0.99999976 and final W2 cosine is
  0.99999722--0.99999726 with relative-L2 0.00234--0.00236.
- Performance rejects both cluster sizes by a wide margin.  At M32/split-4,
  candidate/control cold-L2 local medians are 242.144/198.224 us (+22.2%);
  fused W13+activation is 165.200 us versus 115.136+6.592=121.728 us
  (+35.7%).  At M128/split-2, totals are 388.160/328.256 us (+18.2%) and
  W13+activation is 264.320 versus 196.224+8.768=204.992 us (+28.9%).
- The 4-CTA result rules out a small-M-only cluster-size problem.  Requiring
  all split and gate/up CTAs to be co-resident, then holding their shared
  allocations through two cluster barriers, costs 59--60 us at the complete
  local-pipeline level--an order of magnitude more than the removed 6--9 us
  activation launch.  Reject before distributed timing and restore the exact
  iteration-53 non-cluster source.
- Evidence:
  `bench/results/iter56_cluster_w13_compile_correctness_20260903.log` and
  `bench/results/iter56_cluster_w13_stage_initial_20260903.log`.

### WGMMA iteration 57a — last-arriving W13 CTA activation plan

- Retain the selected independent W13 CTA scheduling and FP32 global partial
  layout.  Add one completion counter per `(padded M8 block, gate/up N128
  pair)`.  After its coalesced partial stores, every contributing CTA makes
  its own stores globally visible and increments the counter; only the last
  of `2*SplitK` arrivals rereads the original FP32 splits and emits the exact
  BF16/SwiGLU/BF16/group-128-FP8 activation for its up-to-eight routed rows.
- This avoids cluster co-residency and DSM lifetime barriers.  It removes the
  separate activation kernel launch but deliberately does not claim the
  workspace-traffic saving that iteration 56 attempted.  A captured counter
  reset is part of the candidate local/full graph and therefore part of every
  cold-L2 timing sample; first prototype it as an explicit zero and fuse that
  reset into route preparation only if the complete candidate wins.
- The uniform last-arrival CTA processes one valid routed row at a time with
  all 128 threads; padded rows skip the reduction.  Require balanced/skew
  split-4/split-2, TP8-shape, elevated-scale, and repeated graph correctness
  to catch ordering/reset races.  Screen candidate/control/candidate on
  fused `W13+activation` and total local latency.  Reject if global fences,
  atomics, reduced activation parallelism, or the reset cost exceed the
  removed 6--9 us launch.

### WGMMA iteration 57b — last-arriving W13 CTA activation rejected

- The last-arrival memory protocol passes TP4 balanced/skew split-4/split-2,
  TP8-shape split-2, and elevated scales.  Raw W13 cosine is at least
  0.99999998 in the covered tests, activation cosine is at least 0.99999927,
  and final W2 cosine is at least 0.99999721 with finite output.
- With an explicit captured reset, candidate/control cold-L2 local medians
  are 91.328/87.728 us at M8, 137.680/131.936 us at M16, and
  206.240/198.432 us at M32.  M8 alone initially appeared salvageable from
  the combined-stage intervals, so a second version fused counter clearing
  into route/input quantization and instantiated separate tail/non-tail W13
  kernels; only M8 (48 routes) selected the tail path.
- Equal-event-node candidate/control/candidate M8 stage totals for that
  second version are 90.048/87.744/89.968 us: candidate average 90.008 us is
  2.58% slower.  The tail makes the W13 launch about 4.41 us longer, while
  deleting the standalone activation work recovers only about 2.94 us; W2
  also starts roughly 0.82 us later in these windows.
- The decisive TP4 production graph screen includes fused route preparation,
  W13, activation, W2, local k6 reduction, and the same CustomAllReduceV2,
  with a separate 256 MiB L2 clear before every replay.  Over 5x100 samples,
  candidate/control M8 medians are 0.079456/0.077248 ms: the candidate is
  2.86% slower.  Candidate graph correctness still passes with cosine
  0.999995565, relative-L2 0.00297848, finite output, and all-reduce OK.
- Reject and restore iteration 53.  Avoiding one launch cannot repay
  per-producer-thread device fences, completion atomics, and the loss of the
  route-parallel activation grid, even when counter reset is free inside an
  existing kernel.
- Evidence:
  `bench/results/iter57_tail_w13_compile_correctness_20260903.log`,
  `bench/results/iter57_tail_w13_stage_initial_20260903.log`,
  `bench/results/iter57b_fused_reset_m8_stage_aba_20260903.log`, and
  `bench/results/iter57b_tp4_m8_tail_{candidate,control}_coldl2_screen_20260903.log`.

### WGMMA iteration 58a — post-stall W2/profile reassessment

- Three structurally different W13 fusion/narrowing attempts (iterations
  55--57) failed, so pause source changes and re-profile the exact selected
  iteration-53 implementation under the mandatory excluded 256 MiB cold-L2
  protocol.  A fresh M32 Nsight Compute capture of the selected W2 launch
  reports 61.34 us under profiler replay, 2.60 TB/s DRAM throughput, 54.05%
  DRAM-throughput utilization, 71.87% compute-throughput utilization, 6.74%
  L2 hit rate, 54 registers/thread, no spills, 56.25% theoretical and 53.26%
  achieved occupancy.  The grid is 6144 CTAs of 128 threads (8.75 waves/SM).
  This supersedes the older iteration-32 W2 profile: the selected W2 is
  neither at a simple HBM ceiling nor clearly occupancy-starved.
- A same-window CUDA-Graph stage audit shows Humming/custom local medians of
  108.672/88.352 us at M8, 240.288/198.624 us at M32, and
  389.920/328.480 us at M128.  Thus the complete local custom pipeline is
  already 18.7%--21.0% faster than Humming.  W2 itself is
  26.336/25.920 us, 70.816/63.184 us, and 120.896/106.992 us respectively;
  it is no longer the old implementation's isolated loss.  The formal full
  TP ratio is lower because the identical CustomAllReduceV2 latency is a
  shared serial term, especially at M128.
- The only cheap local boundary not tested in the selected implementation is
  the fixed-shape k=6 weighted route reduction.  SGLang's BF16 Triton path
  uses BLOCK_M=2 and BLOCK_K=512, giving only 32 CTAs at M8.  Iteration 48's
  rejected CUDA replacement used one CTA per token and serialized all H=4096;
  that does not test a hidden-tiled multi-CTA mapping.  Next, screen fixed
  CUDA variants with 8 or 16 hidden tiles per token and 128/256 threads,
  preserving FP32 route weights, the exact 1.5 multiplier, FP32 ordered k=6
  accumulation, BF16 output, graph topology, and cold-L2 protocol.  Any
  sub-microsecond result must survive candidate/control/candidate timing and
  a full distributed graph before selection.
- Evidence:
  `bench/results/iter58_current_w2_m32_coldl2_detailed_ncu.{log,ncu-rep}`
  and
  `bench/results/iter58_current_humming_custom_stage_coldl2_20260903.log`.

### WGMMA iteration 58b — tiled-k6 launch batch failed before execution

- Added an opt-in fixed H=4096/k=6 CUDA route reducer with four compiled
  mappings: 128 threads x 1/2 BF16 pairs and 256 threads x 1/2 BF16 pairs,
  yielding 16/8/8/4 CTAs per token.  Each output pair has one writer and
  preserves route order, FP32 `weight*1.5`, FP32 FMA accumulation, and BF16
  output.  The graph and stage harness select it only when
  `V4_TILED_K6_REDUCE_MODE` is nonzero; mode 0 remains the exact SGLang
  control.
- The first combined correctness/timing batch did not execute a GPU kernel.
  Its remote loop variables were embedded in a locally double-quoted ssh
  command, so the local zsh expanded `$mode` and `$m` to empty strings before
  transmission.  Imports failed on an empty reducer mode and argparse failed
  on an empty M.  This is a benchmark-launcher quoting failure, not candidate
  correctness or performance evidence; retain the candidate unselected and
  rerun the same suite with escaped remote variables.
- Evidence:
  `bench/results/iter58_tiled_k6_initial_correctness_stage_coldl2_20260903.log`.

### WGMMA iteration 58c — tiled-k6 mappings pass and show a small signal

- Re-ran the batch with a literal remote stdin script.  All four mappings
  complete the full TP4-shape M8 numerical test and their complete BF16
  `[8,4096]` output is bitwise equal to SGLang `moe_fused_mul_sum`
  (`max_abs=0`).  The full custom result remains finite with cosine
  0.999997256 and relative-L2 0.002342691 versus the MXFP4 reference.
- Cold-L2 local-reduce medians (us) for control/mode1/mode2/mode3/mode4 are
  5.536/5.312/5.184/5.120/5.184 at M8,
  6.624/5.728/5.792/5.728/5.872 at M32, and
  7.360/7.584/7.168/7.584/7.168 at M128.  The hidden-tiled mapping therefore
  recovers 0.19--0.90 us where the old one-CTA/token implementation lost
  2--3 us.  Modes 2 (128 threads x two pairs) and 4 (256 threads x two
  pairs) are the only finalists that improve the isolated stage at all three
  M values.
- One-window complete-local medians are too confounded by W13/W2 drift to
  select from: control is 87.984/199.136/328.144 us, mode 2 is
  87.504/197.888/328.800 us, and mode 4 is
  87.664/197.904/327.808 us at M8/M32/M128.  Next use an interleaved
  candidate/control sequence for modes 2 and 4, then require a distributed
  full-graph result because the maximum possible local gain is under 1 us.
- Evidence:
  `bench/results/iter58c_tiled_k6_initial_correctness_stage_coldl2_20260903.log`.

### WGMMA iteration 58d — interleaved tiled-k6 finalist audit

- Interleaved mode2/control/mode4/control/mode2/mode4 with 200 separately
  cold-L2 graph samples per window confirms the isolated reducer advantage,
  but not a broad complete-pipeline win.  At M8, control local-reduce windows
  average 5.616 us, mode 2 averages 5.280 us, and mode 4 averages 5.200 us.
  Their complete-local averages are 87.992/88.304/87.864 us, so only mode 4
  has a tiny 0.15% total signal.
- At M32, control/mode2/mode4 reducer averages are
  6.432/5.952/5.784 us, but complete-local averages are
  197.896/198.656/198.264 us: both candidates lose 0.19%--0.38% once normal
  W13/W2 window variation is included.  At M128 the same figures are
  7.424/7.328/7.296 us and 327.800/328.128/328.440 us, again a
  0.10%--0.20% total regression.
- Do not enable either mode generally.  Give mode 4 one bounded TP4 M8
  full-distributed candidate/control/candidate screen because its two local
  windows straddle both controls and average 0.128 us faster.  Select it only
  if the production graph, including the same CustomAllReduceV2, repeats a
  reduction beyond noise; otherwise reject the tiled reducer wholesale.
- Evidence:
  `bench/results/iter58d_tiled_k6_modes2_4_interleaved_stage_coldl2_20260903.log`.

### WGMMA iteration 58e — mode-4 TP4 M8 full graph wins narrowly

- Production TP4 candidate/control/candidate graph medians over 5x100
  separately cold-L2 samples are 76.640/77.024/76.672 us.  The two mode-4
  candidate windows average 76.656 us, a repeatable 0.368 us or 0.478%
  reduction from control.  All three runs use a 64 KiB SGLang
  `ONE_SHOT_PUSH` CustomAllReduceV2 after the reducer.
- Candidate correctness is unchanged: minimum-rank cosine is 0.999995565,
  relative-L2 is 0.002978479, all values are finite, and the independent
  NCCL sum check passes.  This validates the M8 local-stage signal in the
  actual distributed graph rather than merely event-boundary timing.
- Retain mode 4 as a provisional M8-specific winner, not a global default:
  M32/M128 complete-local screens did not win.  Screen the two missing score
  points M16/M64 before encoding an automatic policy.  Even if M16 also
  wins, this optimization contributes far below the roughly 9 us geometric-
  mean reduction still needed for the 1.20x objective, so resume a larger
  structural hotspot afterward.
- Evidence:
  `bench/results/iter58e_tiled_k6_mode4_tp4_m8_fullgraph_aba_coldl2_20260903.log`.

### WGMMA iteration 58f — bound tiled-k6 selection to M <= 16

- TP4 M16 production candidate/control/candidate medians are
  121.184/121.728/121.152 us over 5x100 cold-L2 samples per window.  Mode 4
  averages 121.168 us, a repeatable 0.560 us or 0.46% reduction.  Minimum-
  rank cosine remains 0.999995547, relative-L2 0.002984397, all values are
  finite, and both candidate windows pass the NCCL all-reduce check.
- TP4 M64 candidate/control/candidate medians are
  272.768/272.384/273.600 us.  The two candidate windows average
  273.184 us, 0.29% slower than control.  This larger point also exhibits a
  system-level two-mode distribution across its five batch medians, but the
  candidate fails even under an average of its surrounding direction.
- Select fixed mode 4 only for M <= 16; retain SGLang `moe_fused_mul_sum` at
  M >= 32.  This policy is backed by full distributed graphs at every
  boundary-adjacent point: M8 and M16 repeatably win, while M32/M64/M128 do
  not show a complete-pipeline win.  Encode the policy without changing the
  compile-time kernel or graph node count, then run TP4/TP8 numerical gates
  and a five-point formal paired audit.
- Evidence:
  `bench/results/iter58f_tiled_k6_mode4_tp4_m16_m64_fullgraph_aba_coldl2_20260903.log`.

### WGMMA iteration 58g — select small-M tiled-k6 auto policy

- The selected default is now `auto`: M8/M16 use fixed tiled CUDA mode 4;
  M32/M64/M128 use the unchanged SGLang reducer.  Explicit 0--4 overrides
  remain available for controlled regressions.  TP4 balanced M8, skew M16,
  balanced M32, and TP8-shape M8 all pass.  The two selected small-M outputs
  are bitwise equal to SGLang (`max_abs=0`), while final local cosine is
  0.99999723--0.99999728 and every output is finite.
- Formal TP4 paired 10x200 Humming/custom medians (ms) are
  0.090240/0.076928 at M8, 0.146048/0.122944 at M16,
  0.232272/0.206368 at M32, 0.336192/0.298816 at M64, and
  0.407856/0.379152 at M128.  Point speedups are
  1.17304/1.18792/1.12552/1.12508/1.07571.  Every point passes independent
  Humming and custom correctness plus the all-reduce check.
- Humming/custom geometric means are 0.211153/0.185751 ms, speedup 1.136755
  (13.68%).  Relative to the previous selected formal window, custom itself
  falls from 0.186248 to 0.185751 ms (0.27%), consistent with the bounded
  M8/M16 wins.  Humming simultaneously falls 0.71%, so the cross-window
  headline ratio is lower; do not misattribute that baseline drift to this
  candidate.  At this paired baseline, 1.20x requires custom <=0.175961 ms,
  another 9.79 us or 5.27% reduction.
- The true TP8 M8 graph also passes with a 0.061840 ms median over 20 cold-L2
  samples, minimum-rank cosine 0.999991970, relative-L2 0.004007414, finite
  output, and all-reduce OK.  Select iteration 58's auto policy, then return
  to a structural hotspot capable of multi-microsecond savings; further
  local-reducer tuning cannot close the objective.
- Evidence:
  `bench/results/iter58g_tiled_k6_auto_correctness_tp4_formal_tp8_smoke_coldl2_20260903.log`.

### WGMMA iteration 59a — communication-boundary reassessment

- Audit the exact SGLang `CustomAllReduceV2` used by both graphs.  It owns a
  PyTorch symmetric-memory slab containing two phases of per-source push
  buffers, a pull buffer, and pull semaphores.  TP4 graph messages are
  64/128/256/512/1024 KiB at M8/16/32/64/128.  The SM90 table selects
  1shot-push through 384 KiB, hence M8--M32, and graph-mode 2shot-pull for
  M64/M128.  The former pushes each local BF16 vector into all four peers'
  slots, polls four local slots, reduces ranks in FP32, restores the positive-
  zero empty sentinel, and advances one phase counter per one of 78 CTAs.
- Re-profile selected M8/M128 graphs with Nsight Systems graph-node tracing.
  `cudaProfilerStart()` itself is not rank-synchronized: early ranks wait in
  the collective for ranks still entering the profiler, producing false
  1--188 ms AR durations.  Do not use those waiting ranks as latency data.
  The last-arriving rank reports an approximate intrinsic lower-bound of
  4.032 us for M8 1shot-push and 10.623 us for M128 2shot-pull.  Adjacent
  route reduction is 1.63--1.70 us at M8 and 3.42--3.71 us at M128.  These
  traces are diagnostic only; formal CUDA-event results remain authoritative.
- Upstream source review confirms that modern PyTorch/SGLang symmetric-memory
  paths are graph-capturable and that production serving increasingly fuses
  neighboring epilogues into all-reduce rather than treating communication
  as untouchable.  The local checked-out V2 code is the source of truth for
  its sentinel/counter protocol; do not substitute the original MegaMoE EP
  barrier or a generic NCCL call.
- Next prototype an opt-in TP4 fused k6+1shot-push kernel for M8--M32.  Reuse
  the *same* SGLang symmetric push slots and 78-element phase counter so it
  can interleave safely with the unmodified Humming baseline.  Each 16-byte
  vector must compute the six BF16 route rows in ordered FP32, convert to
  BF16, replace positive zero with negative zero, system-scope push to all
  peers, poll/reduce four rank slots, write the final BF16 output, clear the
  slots, and increment exactly the same per-block counter.  Baseline remains
  stock SGLang; TP8 and nonselected sizes initially fall back.  Gate repeated
  graph replay for deadlock/phase correctness before any timing claim.
- Evidence:
  `bench/results/iter59b_current_tp4_m8_m128_cold_graph_node_nsys.log`,
  `bench/results/iter59b_current_tp4_m8_m128_cold_graph_kernel_stats.log`,
  and `bench/results/iter59b_current_tp4_m{8,128}_cold_graph_rank*.nsys-rep`.

### WGMMA iteration 60 — fused k6 plus TP4 one-shot push compiles and replays

- Added an opt-in TP4 kernel that directly forms each 16-byte output vector
  from the six BF16 route rows in ordered FP32, replaces positive-zero
  sentinels, pushes the vector into all four ranks' existing SGLang symmetric
  workspaces, polls and reduces the four sources, restores empty sentinels,
  and advances the communicator's existing 78 phase counters.  The stock
  Humming baseline and default custom path remain unchanged; TP8 and M>32
  continue to fall back to `CustomAllReduceV2`.
- The first TP4 M8 run JIT-compiles successfully, completes graph capture and
  20 consecutive cold-L2 replays without a phase deadlock.  Numerical gates
  pass with minimum-rank cosine 0.999995565, relative-L2 0.002978479, finite
  output, and an independent NCCL all-reduce check.  This proves the reused
  symmetric-memory layout/counter protocol is functional across replay.
- The single 20-sample window has a 0.076784 ms median (0.075648 ms minimum;
  one 0.234816 ms outlier).  It is only an initial signal, not a speedup
  verdict: the selected iteration-58 formal M8 custom median was 0.076928 ms
  and was measured in a different window.  Next run interleaved
  fused/control/fused full graphs at M8, then test M16/M32 only if the paired
  result survives drift.
- Evidence:
  `bench/results/iter60_fused_k6_push_tp4_m8_compile_correctness_smoke_coldl2_20260903.log`.

### WGMMA iteration 61 — same-process fused versus stock CARv2 audit

- Added a diagnostic harness that captures fused and stock custom graphs in
  one TP4 process with identical X, routing, weights, and one shared SGLang
  communicator.  It alternates complete fused/control and control/fused
  batches, performs a separate 256 MiB L2 clear immediately before every
  replay, excludes the clear from CUDA events, and reduces every sample to
  the slowest rank.  This removes cross-process clock and communicator drift
  from the selection decision.
- Over six batches and 1200 samples per implementation per point, fused versus
  control medians are 0.076416/0.076864 ms at M8, 0.122592/0.123648 ms at
  M16, and 0.193600/0.194528 ms at M32.  The corresponding reductions are
  0.448 us (0.583%), 1.056 us (0.854%), and 0.928 us (0.477%); three-point
  geometric-mean speedup is 1.006422x.  Every fused batch-median sequence is
  below or substantially overlaps its control neighborhood, so this is a
  small but credible win rather than the multi-microsecond gain initially
  hypothesized.
- Both paths pass the independent NCCL sum gate at all three shapes and their
  complete graph outputs are bitwise identical (`max_abs=0`).  Select the
  fused path provisionally for TP4 M8/M16/M32, but do not claim the 20%
  objective: with M64/M128 unchanged, this can improve the five-point score
  by only roughly half a percent.  Profile the two tail implementations next
  to locate why eliminating a graph node saves less than one microsecond at
  M8/M32.
- Evidence:
  `bench/results/iter61_fused_k6_push_tp4_m8_m16_m32_ab_coldl2_20260903.log`.

### WGMMA iteration 62 — fused-tail graph-node profile

- Captured one explicitly cold graph replay for the selected fused path at
  TP4 M8 and M32.  As in iteration 59, the CUDA profiler start call is not
  synchronized after each rank enters it, so early ranks spin in the
  collective for milliseconds; those durations are synchronization artifacts
  and must not be interpreted as kernel cost.  The last-arriving ranks give
  usable tail measurements of 4.416 us for the 128-thread M8 fused kernel and
  6.048 us for the 256-thread M32 fused kernel.
- The comparable stock M8 trace had approximately 1.63--1.70 us of tiled k6
  reduction followed by a 4.032 us one-shot-push kernel.  Fusing therefore
  removes a graph node and intermediate BF16 write/read, but moves the six-row
  calculation into the communication CTA and still spends about 4.4 us in
  its push/poll protocol.  The profile predicts at most roughly 1.2 us of M8
  tail reduction; iteration 61 observes 0.448 us at the full-graph max-rank
  boundary.  The result rules out launch fusion alone as a route to the
  remaining approximately 5% geometric-mean reduction.
- Keep the bitwise-correct fused tail, but change structural direction.  A
  plausible larger target is producer-side W2 epilogue progress: publish
  completed token/hidden tiles before the entire W2 grid drains, while a
  following communication kernel only polls/reduces.  This can overlap NVLink
  stores with W2 tail work; it must avoid in-producer polling and global
  co-residency barriers to prevent the failures seen in iterations 56--57.
- Evidence:
  `bench/results/iter62_fused_k6_push_tp4_m8_m32_cold_graph_node_nsys.log`
  and `bench/results/iter62_fused_tp4_m{8,32}_cold_graph_rank*.nsys-rep`.

### WGMMA iteration 63 — multicast push prototype compiles and replays

- The existing read-only Hopper capability probe confirms this H20 NVLink
  clique exposes a nonzero symmetric-memory multicast VA.  Four ranks issue
  `multimem.red.add.f32` with values 1--4 and every local replica reads the
  exact sum 10 (`maxerr=0`).  This makes SGLang K3's multicast-push technique
  applicable on the actual benchmark machine rather than merely supported by
  the architecture in principle.
- Added an opt-in TP4 multicast-push specialization of iteration 60's fused
  k6 tail.  It retains the exact SGLang two-phase push workspace and 78
  counters, but replaces four peer system stores per 16-byte vector with one
  `multimem.st` that fans out to every rank.  It also adopts K3's safe
  4-byte-empty-marker test: an all-positive-zero BF16 pair is changed to
  `{-0,+0}`, while mixed pairs need no remapping.  The original unicast
  candidate, stock baseline, TP8 fallback, and M128 fallback remain intact.
- TP4 M8 JIT-compiles, captures, and completes 20 consecutive cold-L2 graph
  replays without a phase deadlock.  Correctness is unchanged: minimum-rank
  cosine 0.999995565, relative-L2 0.002978479, finite output, and independent
  NCCL all-reduce check pass.  The initial median is 0.076144 ms, versus
  0.076416 ms for unicast in iteration 61, but this cross-window 0.272 us
  difference is not selection evidence.  Next use the same-process AB/BA
  harness for M8--M64.
- Evidence:
  `bench/results/iter63_tp4_multimem_capability_probe_20260903.log` and
  `bench/results/iter63_multicast_push_tp4_m8_compile_correctness_smoke_coldl2_20260903.log`.

### WGMMA iteration 64 — multicast push same-process selection audit

- Compared multicast-push against stock `CustomAllReduceV2` in one TP4
  process with identical inputs/weights and communicator, six balanced AB/BA
  batches, 1200 max-rank cold-L2 samples per implementation and point.  Both
  paths pass the independent NCCL sum check at M8/M16/M32/M64, and every
  complete graph output is bitwise identical (`max_abs=0`).
- Multicast/control medians are 0.075904/0.076672 ms at M8,
  0.122048/0.123360 ms at M16, 0.192832/0.194736 ms at M32, and
  0.287904/0.289248 ms at M64.  The point speedups are 1.01012x, 1.01075x,
  1.00987x, and 1.00467x; four-point geometric-mean speedup is 1.00885x.
  Replacing four unicast stores with one multicast store approximately
  doubles iteration 61's small-tail gain at M8/M32.
- M8--M32 are repeatable enough to select provisionally: all six M8/M16
  candidate batch medians beat control, and M32's pooled result saves 1.904
  us despite system drift.  M64 is not selected yet because both paths show
  a broad 0.27--0.41 ms two-mode distribution and candidate/control batch
  order is mixed; its 1.344 us pooled advantage needs a focused repeat.
  Even the four-point result remains far below the 20% end-to-end objective,
  so retain multicast push as a component and continue toward multicast pull
  or producer/communication overlap, especially for M128.
- Evidence:
  `bench/results/iter64_multicast_push_tp4_m8_m16_m32_m64_ab_coldl2_20260903.log`.

### WGMMA iteration 65 — production low-SM NVLS pull integration

- Added an opt-in path that writes the existing k6 local reduction directly
  into `CustomAllReduceV2`'s symmetric pull workspace, then calls SGLang's
  production K3 `all_reduce_pull_res`.  That kernel uses a low-SM pipelined
  `multimem.ld_reduce` reduce-scatter plus multicast broadcast in place and
  reuses the exact CARv2 pull-semaphore reservation protocol.  No new
  synchronization protocol or private communication allocation was added;
  baseline capture still calls stock CARv2, and TP8 falls back unchanged.
- The strict TP4 M128 first-run gate succeeds: K3 JIT compilation, CUDA Graph
  capture, and 20 consecutive 1 MiB cold-L2 replays complete without hang.
  Minimum-rank cosine is 0.999995609, relative-L2 is 0.002963508, output is
  finite, and the independent NCCL sum check passes.  This validates both
  in-place symmetric output lifetime and interoperability with the existing
  communicator semaphores.
- The initial M128 median is 0.323728 ms (0.322656 ms minimum), much lower
  than iteration 58's 0.379152 ms formal stock-custom point.  This is a
  cross-window observation and is not yet a speedup claim; the unusually
  large delta exceeds the previously measured standalone AR tail and must be
  audited in the same process against stock CARv2 at all five M values.
- Evidence:
  `bench/results/iter65_multicast_pull_tp4_m128_compile_correctness_smoke_coldl2_20260903.log`.

### WGMMA iteration 66 — low-SM NVLS pull five-point audit

- Same-process stock/NVLS-pull AB/BA over 1200 cold-L2 samples per path and
  point disproves iteration 65's apparent 55 us cross-window gain.  Pull versus
  stock medians are 0.078080/0.076736 ms at M8, 0.124896/0.123392 ms at M16,
  0.194368/0.194720 ms at M32, 0.289968/0.287040 ms at M64, and
  0.369872/0.377648 ms at M128.  Point speedups are 0.98279x, 0.98796x,
  1.00181x, 0.98990x, and 1.02102x; the five-point geometric mean is a
  0.34% regression.
- Every candidate/control graph remains bitwise identical and passes the
  independent NCCL sum check.  Therefore the result is a performance
  rejection, not a numerical or semaphore failure.  The tuned K3 low-SM
  setup cost is inappropriate for 64--128 KiB, multicast push remains the
  selected small-message family, and the current default pull geometry does
  not win at M64.
- Keep NVLS pull opt-in only.  M128 is the sole possible selection, saving
  7.776 us in the pooled median, but its six paired batch medians are mixed
  and the platform again enters two latency modes.  Before selecting it,
  sweep K3's exposed `(num_blocks, unroll)` on focused M128 same-process
  windows and require candidate/control/candidate consistency.  The
  iteration-65 single-window 0.323728 ms result is explicitly rejected as a
  formal estimate.
- Evidence:
  `bench/results/iter66_multicast_pull_tp4_allm_ab_coldl2_20260903.log`.

### Iteration 67 — M128 NVLS-pull launch-geometry sweep (cold L2)

- Change: made the production K3 NVLS-pull reducer geometry tunable with `V4_MC_PULL_BLOCKS` and `V4_MC_PULL_UNROLL`, and threaded both controls through the graph and same-process A/B harness.
- Protocol: TP4 on GPUs 1–4, M=128, random real route metadata, identical communicator/weights/inputs, paired fused/control then control/fused batches, 4 outer batches × 100 graph replays per implementation. A separate 256 MiB L2 clear ran immediately before every replay and outside CUDA-event timing.
- Correctness: every configuration was finite and passed the all-reduce/reference checks (`cosine_min_rank≈0.99999561`, `rel_l2_max_rank≈0.002964`). Fused versus stock output differed by at most 128 BF16 units because the reduction order differs; both independently matched the FP32 reference.
- Median speedup (`stock CARv2 / NVLS pull`) by `(blocks, unroll)`: `(1,4)=0.9909`, `(1,8)=1.0068`, `(1,16)=1.0171`; `(2,4)=1.0089`, `(2,8)=0.9799`, `(2,16)=0.9946`; `(4,4)=0.9913`, `(4,8)=1.0066`, `(4,16)=1.0374`; `(8,4)=1.0213`, `(8,8)=1.0182`, `(8,16)=0.9589`; `(16,4)=1.0335`, `(16,8)=1.0057`, `(16,16)=1.0075`.
- Result: the best one-window point was `(4,16)` at 0.344784 ms versus 0.357664 ms (1.03736×), closely followed by `(16,4)` at 1.03346×. The non-monotonic/bimodal batches show this is only a tuning lead, not yet a stable accepted policy; it needs a longer focused A/B/A confirmation.
- Evidence: `bench/results/iter67_multicast_pull_tp4_m128_blocks_unroll_sweep_coldl2_20260903.log`.

### Iteration 68 — chunked W2 / multicast-push overlap prototype (cold L2)

- Change: added an opt-in TP4 M128 CUDA-Graph pipeline. W2 is split into four disjoint 1024-channel launches on the graph's main stream; after each chunk event, a second stream runs an eight-CTA fused k6 reduction plus multicast one-shot push into the existing SGLang CARv2 symmetric workspace. The main stream joins the communication stream before graph completion. The stock path and baseline are unchanged.
- Safety/design: each communication chunk starts only after its entire W2 chunk completes, so this prototype needs no producer completion atomics, device fences, sentinels, dispatch, or combine. It reuses CARv2's phase counters and push slots and writes each reduced chunk into its final `[M,4096]` offset.
- Protocol: same-process TP4 M128 pipeline/control, identical communicator/weights/input/routes, balanced two-batch AB/BA order, 20 replays per implementation per batch. Every replay gets a separate 256 MiB L2 clear immediately beforehand and outside CUDA-event timing.
- Correctness: multi-stream graph capture and repeated replay complete without hang. Candidate and control independently pass (`cosine_min_rank=0.999995609`, `rel_l2_max_rank=0.002963504`, finite, all-reduce OK), and their BF16 outputs are bitwise identical (`max_abs=0`).
- Performance: pipeline median is 0.411248 ms versus 0.326768 ms control, `control/pipeline=0.79458x`; both candidate batch medians (0.410832/0.411824 ms) lose decisively to both control batches (0.326704/0.327056 ms).
- Result: reject four-chunk/eight-CTA overlap as configured. Splitting the route-GEMM destroys more scheduling/weight-pipeline efficiency and/or creates more compute/communication contention than it hides. Preserve the opt-in prototype for stage profiling or a bounded two-chunk/low-CTA test; do not select it.
- Evidence: `bench/results/iter68_pipeline_w2_mc_push_tp4_m128_smoke_coldl2_20260903.log`.

### Iteration 69 — bounded chunk/CTA geometry audit rejects W2 overlap (cold L2)

- Experiment: without changing the iteration-68 source, sweep TP4 M128 pipeline geometry over `chunks={2,4}` and multicast-push `blocks={16,32,78}`. Every configuration uses a same-process candidate/control pair, identical data and communicator, two balanced AB/BA batches × 30 separately cold-L2 replays per implementation.
- Correctness: every candidate/control pair passes the independent reference and all-reduce gates (`cosine_min_rank=0.999995609`, `rel_l2_max_rank=0.002963504`, finite) and remains bitwise identical (`max_abs=0`).
- Results (`control/pipeline`): 2 chunks gives 0.87716x at 16 blocks (0.372912 vs 0.327104 ms), 0.94623x at 32 blocks (0.347568 vs 0.328880 ms), and 0.96121x at 78 blocks (0.339504 vs 0.326336 ms). Four chunks gives 0.90581x at 16 blocks, 0.92906x at 32 blocks, and 0.91911x at 78 blocks.
- Finding: raising communication parallelism recovers most of the eight-CTA serialization, but the best geometry still loses 13.168 us or 4.04% at M128, and both of its candidate batch medians lose. Four chunks are worse than two once launch/scheduling and concurrent resource contention are included.
- Decision: reject chunk-level W2/AR overlap as a selectable path. A producer-progress scheme would need finer readiness without fragmenting the W2 grid; the current chunk launch topology cannot close the 20% objective and should not receive broader M or formal testing.
- Evidence: `bench/results/iter69_pipeline_w2_mc_push_tp4_m128_geometry_sweep_coldl2_20260903.log`.

### Iteration 70 — fused rank×route NVLS reduction is correct but too expensive (cold L2)

- Change: added an opt-in TP4 M128 path that leaves W2 as one full route-GEMM launch but places its `[M*6,4096]` BF16 output in CARv2's multicast-bound symmetric pull slab. A 16-CTA kernel then issues six `multimem.ld_reduce...bf16x2` operations per output vector, applies the shared route weights in FP32, and writes final `[M,4096]`. It reuses CARv2's 128-byte pull semaphores and exact 2×world reservation/entry/exit windows.
- Rationale: by linearity, summing each route across four TP ranks before the fixed k6 weighted sum replaces separate local k6 materialization plus output all-reduce without splitting or fencing the W2 producer grid.
- Protocol: same-process TP4 M128 candidate/control, shared inputs/weights/routes/communicator, two balanced AB/BA batches × 20 separately cold-L2 graph replays per implementation.
- Correctness: graph capture and repeated semaphore reuse do not hang. The changed reduction order remains within the MXFP4 gate (`cosine_min_rank=0.999992262`, `rel_l2_max_rank=0.003933958`, finite, all-reduce OK); control is 0.999995609/0.002963504. Candidate/control BF16 max absolute difference is 1024 in the benchmark's output units.
- Performance: candidate median is 0.391136 ms versus 0.324320 ms control, `control/candidate=0.82917x`; candidate batches 0.390896/0.391296 ms both lose to 0.323984/0.324512 ms control.
- Result: reject. Although the kernel eliminates a local pass and one output collective, six rank-reducing NVLS loads per output vector expand cross-rank data movement enough to add 66.816 us. Linearity is algebraically useful but the communication placement is wrong for k=6.
- Evidence: `bench/results/iter70_rank_route_nvls_pull_tp4_m128_smoke_coldl2_20260903.log`.
## Iteration 71 — fused local k6 + one-shot NVLS pull

- Change: added the opt-in `FUSED_K6_NVLS_PULL_AR` path. Each CTA reduces its disjoint local k6 output into the SGLang symmetric output slab, synchronizes through the existing CustomAllReduceV2 semaphore window, then performs one TP4 `multimem.ld_reduce` pull and writes the final output. Added graph-harness and same-process A/B support plus `K6_NVLS_PULL_BLOCKS` tuning (tested at 16).
- Protocol: TP4 on GPUs 1–4; random routes; identical inputs/weights and communicator; CUDA Graph; 256 MiB L2 clear immediately before every replay and excluded from timing; 2 paired-order outer batches × 20 cold samples per implementation; max-rank latency. Log: `bench/results/iter71_fused_k6_nvls_pull_tp4_m8_m128_smoke_coldl2_20260903.log`.
- Correctness: both sizes passed all-rank reference checks. M=8 fused/control were bitwise identical (`max_abs=0`); M=128 had expected reduction-order variation (`max_abs=32`) with fused cosine `0.9999956431` and rel-L2 `0.0029519584`.
- M=8: control median `0.076656 ms`, fused median `0.079952 ms`, speedup `0.9588x` (4.30% slower).
- M=128: control median `0.327056 ms`, fused median `0.360576 ms`, speedup `0.9070x` (10.25% slower).
- Two-point geometric mean: control `0.158338 ms`, fused `0.169790 ms`, speedup `0.9325x`.
- Decision: reject. Removing the intermediate local-k6 launch does not repay the extra synchronization/launch geometry inside the fused NVLS kernel; no parameter sweep is justified after both endpoint sizes regress materially. Keep the path opt-in for evidence only; the selected default remains stock CARv2 with the iteration-58g reducer policy.
### Iteration 72 — long-window M128 NVLS-pull confirmation rejects the short-window lead

- Re-tested the two best iteration-67 production K3 NVLS-pull geometries at TP4 M128 in the same process as stock CARv2. Each geometry used 10 balanced AB/BA outer batches × 200 samples per implementation (2,000 cold samples each), identical inputs/weights/routes/communicator, max-rank timing, and a separate excluded 256 MiB L2 clear immediately before every graph replay.
- `(blocks=4, unroll=16)`: stock median `0.363376 ms`, NVLS pull `0.363472 ms`, stock/candidate `0.999736x`. Candidate batch medians span `0.343312–0.398304 ms`; control spans `0.342352–0.469728 ms`, confirming strong platform bimodality rather than a stable candidate advantage.
- `(blocks=16, unroll=4)`: stock median `0.360528 ms`, NVLS pull `0.360400 ms`, stock/candidate `1.000355x`, only `0.128 us` or `0.036%` and far below noise. Both configurations retain finite output, independent all-reduce/reference success, cosine `0.999995609`, and rel-L2 about `0.00296351`; reduction-order max difference versus stock is 128 output units.
- Decision: reject the iteration-67 short-window 3.74% lead. Neither finalist supplies a repeatable M128 improvement across 2,000 samples, so stock CARv2 remains the selected large-message path. Close launch-geometry tuning and return to the local compute/data path.
- Evidence: `bench/results/iter72_m128_nvls_pull_long_ab_coldl2_20260903.log`.
### Iteration 73 — long-window multicast one-shot push confirmation

- Re-audited the already implemented TP4 multicast fused-k6/one-shot-push path against stock CARv2 at M=8/16/32. The same process, communicator, inputs, weights and routes were used for 10 balanced AB/BA batches × 200 samples per implementation/M. Every replay had an immediately preceding excluded 256 MiB L2 clear; reported latency is the maximum rank.
- Candidate/control medians are `0.076192/0.076704 ms` at M8, `0.122400/0.123664 ms` at M16, and `0.194336/0.196032 ms` at M32. Stock/candidate speedups are `1.006720x`, `1.010327x`, and `1.008727x`; the three-point geometric mean is `1.008590x`.
- Correctness is exact relative to the stock graph at all sizes (`fused_vs_control_max_abs=0`), with finite outputs, independent NCCL-reference/all-reduce success, minimum cosine `>=0.99999556`, and rel-L2 `<=0.00297861`.
- Decision: the long window confirms the iteration-64 signal. Accept multicast fused k6 + one-shot push for TP4 M8/M16/M32; retain stock CARv2 for M64/M128 and all TP8 shapes. This is a real but bounded ~0.86% three-point tail gain and cannot alone satisfy the 1.20x full-score target. Encoding the dispatch policy awaits the pending numerical-boundary design approval so changes can be batched coherently.
- Evidence: `bench/results/iter73_multicast_push_tp4_m8_m16_m32_long_ab_coldl2_20260903.log`.
### Iteration 74 — route-distribution stage budget for the interleaved design

- Without changing source, profiled current custom and exact Humming local graphs at TP4 M=8/32/128 for balanced and maximal-skew routes. Each result uses 200 separately cold-L2 graph replays, with a 256 MiB clear before the complete local pipeline and excluded from event timing. Stage events add instrumentation overhead, so only same-harness comparisons and stage proportions are used.
- Balanced custom/Humming total medians (us): M8 `93.056/115.488` (`1.2411x`), M32 `253.792/307.968` (`1.2135x`), M128 `334.704/397.984` (`1.1891x`). Custom W13/W2 medians are `47.744/27.536`, `151.488/82.432`, and `200.240/109.632 us`; together they consume 80.9%, 92.2%, and 92.6% of custom local time.
- Max-skew custom/Humming total medians (us): M8 `43.872/58.336` (`1.3297x`), M32 `68.160/82.464` (`1.2099x`), M128 `152.736/179.488` (`1.1751x`). Custom W13/W2 medians are `15.808/10.640`, `30.784/18.864`, and `82.144/47.008 us`. Reusing one expert's weights makes fixed preparation/epilogue work dominant; W2 is slightly slower than Humming at skew M8/M32 even though the complete path wins.
- Finding: latency is strongly governed by active-expert cold weight traffic, not routed-row count alone. M32 balanced (192 distinct experts) is slower than M128 balanced (256 experts but denser rows/expert) only modestly despite 4x routes, while skew collapses the cores. A real next-stage kernel must keep dynamic route support, schedule contiguous expert waves, use small block-M for sparse experts and larger block-M for dense experts, and interleave W2 only after all W13 N fragments for that expert block are ready. This supports adapting DeepGEMM's device-side interleaved MegaMoE scheduler rather than static-route grid specialization.
- Evidence: `bench/results/iter74_route_distribution_stage_budget_coldl2_20260903.log`.
### Iteration 75 — first TP interleaved scheduler probe exposes a publication bug

- Added a scheduler-only 78-CTA TP probe plus a CUDA-Graph route-mutation test. Persistent CTA leaders dynamically claim W13 tasks, publish an M block after its fourth W13 tile, and preferentially claim sixteen W2 tiles from the ready queue. Trace tensors record owners and global issue order; no GEMM math or performance timing is involved.
- The extension compiles and the first TP4 M8 balanced case terminates without a deadlock. Device and CPU agree on 48 active padded M blocks, 192 W13 tasks and 768 W2 tasks. All 768 W2 tasks are claimed and the trace observes genuine W13/W2 overlap.
- The correctness gate fails with five device-side violations before the remaining eager and graph-mutation cases run. The W13 cursor overshoots to 382 because idle CTAs continue `atomicAdd` after the 192-task bound; this is wasteful but not itself a missing task. The violation counter can currently combine invalid expert/mblock, premature W2 readiness and excess W13 arrival faults, so the exact publication failure is not yet isolated.
- Decision: reject this first scheduler protocol. Preserve the failed binary/log, then split violation counters and replace the publish-tail-before-payload queue with an explicit per-slot state or acquire/release PTX protocol. Also bound W13 claims with CAS so the diagnostic state is deterministic before integrating any WGMMA.
- Evidence: `bench/results/iter75_interleaved_scheduler_probe_20260903.log`.
### Iteration 76 — isolate stale ready-queue payload loads

- Replaced unbounded W13 `atomicAdd` claims with a bounded CAS loop, made W2 skip unpublished queue slots instead of spinning after claim, and split the violation counter into invalid-W13, invalid-W2-mblock, premature-readiness and duplicate-W13-arrival categories.
- The rebuilt M8 balanced probe again terminates and observes W13/W2 overlap. W13 is now exact and deterministic: claim cursor 192 for 192 tasks, zero invalid W13 blocks and zero duplicate arrivals. All 768 W2 tasks are claimed and there are zero premature-readiness observations.
- The gate still fails, now precisely with 38 invalid W2 mblock payloads and no other violation type. The consumer waits on an atomic valid flag but reads the subsequently produced `ready_queue` payload through `__ldg`, whose read-only/cache semantics are inappropriate for a same-kernel producer-consumer location and can retain the reset `-1` value.
- Decision: reject iteration 76 but keep its isolation evidence. Replace `__ldg(ready_queue)` with an acquire/volatile global load paired with a release publication, then rerun the full eager and graph-route-mutation matrix before any GEMM integration.
- Evidence: `bench/results/iter76_interleaved_scheduler_publication_fix_20260903.log`.
### Iteration 77 — acquire/release scheduler protocol passes dynamic-route graph gate

- Replaced the dynamic ready-queue `__ldg` with GPU-scope `ld.acquire` helpers and published each slot with `st.release` on its valid word after writing the payload. The bounded CAS task claims and split violation counters from iteration 76 remain.
- All nine eager combinations pass at TP4 for M=8/32/128 × balanced/skew/random. Active device M-block counts match the CPU oracle exactly: M8 `48/6/45`, M32 `192/24/146`, and M128 `256/96/247`. Corresponding W13 and W2 task cursors equal their exact bounds, every owner is in `[0,78)`, each W2 `(mblock,n-tile)` appears once, and all four violation categories stay zero.
- Every case observes a W2 issue before the final W13 task, proving the scheduler is actually interleaving rather than merely producing a correct sequential trace.
- A single captured M32 CUDA Graph then replays after in-place route mutations balanced→skew→random→balanced. Device bounds change `192→24→146→192` without host specialization; all task/readiness invariants remain valid and no stale state survives replay.
- Decision: accept the scheduler publication/claim protocol as the control-flow foundation. It is correctness-only and makes no latency claim. Next integrate one real full-K W13 paired gate/up task and validate its BF16/SwiGLU/FP8 intermediate before enabling W2.
- Evidence: `bench/results/iter77_interleaved_scheduler_acquire_release_20260903.log`.

## Iteration 78 — full-K W13 task feasibility (2026-09-03)

- Hypothesis: the native MegaMoE-style full-K W13 task (`split_k=1`) may be a better granularity for an eventual paired gate/up task than the current split-K policy.
- Change: instantiated legal TP4 (`K=4096,N=1024`) and TP8 (`K=4096,N=512`) split-K=1 route GEMMs and matching SwiGLU / fused SwiGLU+dynamic-FP8 reductions. Extension suffix: `v78split1`.
- Correctness: PASS for TP4 M8 balanced and skew, and TP8-shape M8 balanced. W13 cosine was 0.999999997–0.999999998, activation cosine 0.999999652–0.999999746, and W2 cosine 0.999997235–0.999997278; all outputs finite.
- Method: TP4 GPU1, random routes, CUDA Graph, 200 samples per point, separate 256 MiB L2 clear immediately before every replay and excluded from event timing. Paired order was split1 -> auto -> split1.
- Median cold-L2 results (first/second split1 averaged against auto):
  - M8: W13 53.848 us vs 43.888 us (split1 22.70% slower); local total 97.664 us vs 87.472 us (11.65% slower).
  - M32: W13 125.704 us vs 115.088 us (9.22% slower); local total 208.248 us vs 198.368 us (4.98% slower).
  - M128: W13 205.320 us vs 195.904 us (4.81% slower); local total 337.456 us vs 327.392 us (3.07% slower).
- Decision: REJECT full-K as the standalone W13 execution granularity. It is correct but loses consistently, most severely at small M. Preserve split4 for M8/M32 and split2 for M128; the paired/interleaved implementation must retain parallel K work (or use the native 384-thread producer/math pipeline) rather than serializing all 32 K tiles in one 128-thread CTA.
- Evidence: `bench/results/iter78_fullk_w13_feasibility_correctness_stage_coldl2_20260903.log`.

## Iteration 79a — first paired-warpgroups W13 execution fault (2026-09-03)

- Implemented the first real-compute version of the approved MegaMoE task: model-load gate/up row interleaving at granularity 8, one N256/full-K W13 task per output group, a 128-thread producer, two 128-thread RS-WGMMA consumer warpgroups, two weight/activation stages, and an in-CTA BF16/SwiGLU/BF16/group128-FP8 epilogue. The graph path can bypass the old split workspace and activation kernel; a debug-only raw W13 output remains available to the correctness test.
- The JIT extension compiled successfully and fused route/input preparation remained exact (`routes_ok=True`, bitwise FP8 input quantization, zero scale error).
- The first TP4 M8 balanced execution faults asynchronously with `CUDA error: an illegal memory access` before the W2 allocation. No performance timing ran and the candidate is not selectable.
- The likely fault classes are bounded and device-local: producer/consumer mbarrier phase protocol, the two-record shared-memory addresses, or the paired epilogue indexing. Diagnose with synchronous launch plus compute-sanitizer or a staged debug output before making any performance claim.
- Evidence: `bench/results/iter79_paired_w13_first_compile_correctness_20260903.log`.

## Iteration 79b — repair paired-W13 shared-scale addressing (2026-09-03)

- Root cause of iteration 79a: the consumer formed a 32-bit shared-memory address for each scale stage and then cast that integer to a generic C++ pointer. The generated generic load treated the shared offset as a global address, causing the illegal access. Weight data was unaffected because it already used explicit `ld.shared` PTX.
- Repair: retain 32-bit addresses only for descriptors/PTX instructions and use a pointer derived from the actual `weight_smem` generic pointer for scale-byte loads. No math, task geometry, or synchronization changed.
- Synchronous TP4 M8 balanced correctness now passes. Preparation remains exact; W13 cosine/relative-L2 are `0.999999998/0.000076187`, activation `0.999999746/0.000712793`, W2 `0.999997256/0.002342693`, and output is finite.
- Decision: accept the address repair. The new task is now eligible for TP4 skew, TP8-shape, graph replay, and cold-L2 performance screening; no speed claim yet.
- Evidence: `bench/results/iter79b_paired_w13_shared_scale_fix_correctness_20260903.log`.

## Iteration 80 — paired two-warpgroups W13 core screen (2026-09-03)

- Correctness coverage passes for TP4 M8 maximal skew and TP8-shape M8 balanced in addition to iteration 79b's TP4 balanced case. W13 cosine is at least `0.999999997`, activation cosine at least `0.999999652`, W2 cosine at least `0.999997235`, and all outputs are finite.
- Method: TP4 GPU1, random routes, CUDA Graph, 200 samples per point, separate excluded 256 MiB L2 clear before every replay. Candidate/control/candidate order used the same selected W2 and route/local reducers.
- Median cold-L2 candidate averages versus control:
  - M8: fused W13+activation `82.936 us` vs `43.856+6.016=49.872 us` (66.30% slower); local total `121.016 vs 87.808 us` (37.82% slower).
  - M32: fused W13+activation `173.672 us` vs `115.008+6.560=121.568 us` (42.86% slower); local total `251.480 vs 197.840 us` (27.11% slower).
  - M128: fused W13+activation `289.360 us` vs `196.064+8.784=204.848 us` (41.26% slower); local total `413.744 vs 328.288 us` (26.03% slower).
- Decision: REJECT this first 384-thread paired compute body. Correct gate/up pairing and direct epilogue do not compensate for its full-K task serialization and producer/consumer synchronization. Do not connect W2 to this body or claim scheduler overlap. Preserve the validated dynamic scheduler separately; inspect compiled resource use and isolate whether the producer pipeline itself is defective before considering a split-K or persistent repair.
- Evidence: `bench/results/iter80_paired_w13_correctness_stage_aba_coldl2_20260903.log`.

## Iteration 81 — NCU diagnosis of the rejected paired body (2026-09-03)

- Collected matching cold-cache NCU profiles of one TP4 M8 balanced paired launch and the selected split4 W13 launch. Profiling replay durations are diagnostic, not benchmark headline numbers.
- Paired/control durations are `77.18/44.10 us`; DRAM throughput is `29.66%/52.14%` and SM throughput `38.26%/63.78%`. The paired kernel is neither HBM- nor tensor-pipe-saturated.
- The paired CTA uses 59 registers/thread (64 allocated) and 47.10 KiB shared memory. Both registers and shared memory cap it at two 384-thread CTAs/SM. The control uses 54 registers/thread (56 allocated), 23.55 KiB shared memory and permits nine 128-thread CTAs/SM. Paired barrier-stall pressure is also higher (`4.78` versus `3.59` instructions per issue-active cycle).
- Finding: the first body dedicates one third of resident warps to producers while only four math warpgroups/SM remain; its two-stage empty/full barrier topology then leaves both compute and DRAM underfilled. This is an implementation/occupancy failure, not evidence against gate/up interleaving itself.
- Next bounded repair: use a 256-thread CTA containing only two aligned math warpgroups. Each warpgoup independently performs the selected two-stage load/dequant/WGMMA loop for one N128 half, then the CTA shares only the group128 epilogue. This removes four producer warps and empty-stage barriers while retaining paired semantics. Reject if it cannot approach the selected `W13 + activation` latency.
- Evidence: `bench/results/iter81_{paired,control}_w13_m8_ncu.{log,ncu-rep}` and `bench/results/iter81_paired_vs_control_w13_m8_ncu_summary.log`.


### Iteration 82a — invalid launcher path (not a kernel result)

- Intent: compile and validate the 256-thread dual-math-warpgroup W13 repair on TP4-shape M8.
- Source change staged locally: replace the 384-thread producer/consumer CTA with two independent 128-thread math warpgroups, each owning two weight and activation stages.
- Launcher failed before Python or CUDA started because container `dpskv4_h20_weekly_gap_20260727` does not expose `/home/xutingz/fac/DeepGEMM_tp`; its repository mount is under `/lustre/raplab/client/xutingz/fac/DeepGEMM_tp`.
- Evidence: `results/v4_flash_tp_wgmma_pair2wg_tp4_m8_invalid_path_20260903.log`.
- Classification: infrastructure invocation error; no correctness or performance conclusion. Retry only with the resolved container path.

### Iteration 82b — 256-thread dual-WG TP4 M8 correctness

- Candidate: the two 128-thread math warpgroups independently issue their two-stage W13 weight/scale bulk copies and duplicate the indexed M8 activation tile into private stages; the producer warpgroup and empty-stage barriers are absent from the compiled path.
- Shape: TP4 local intermediate 512, M=8, balanced 48 active experts / 384 padded routes.
- Result: route preparation exact; W13 cosine 0.999999998, relative L2 0.000076187; fused SwiGLU/requant cosine 0.999999746, relative L2 0.000712793; W2 cosine 0.999997256, relative L2 0.002342693, finite.
- Evidence: `results/v4_flash_tp_wgmma_pair2wg_tp4_m8_correctness_20260903.log`.
- Decision: correctness gate passes. Next iteration must measure cold-L2 candidate/control/candidate before considering any default integration.

### Iteration 82c — 256-thread dual-WG cold-L2 rejection

- Method: TP4 local shape, M=8 balanced routing, CUDA Graph stage profiler, 200 timed samples per process, candidate/control/candidate ordering. A separate 256 MiB buffer is cleared immediately before every replay and outside the CUDA-event interval; H20 reports 60 MiB L2.
- Candidate fused W13+activation/requant median: 67.680 / 67.760 us, average 67.720 us. Control split W13 plus fused activation/requant median: 47.680 + 6.016 = 53.696 us. Candidate is 26.12% slower at the target stage.
- Candidate total local-pipeline median: 108.016 / 108.144 us, average 108.080 us. Control total median: 93.632 us. Candidate is 15.43% slower end to end before all-reduce.
- The repair materially improves the first 384-thread prototype (which was roughly 38% slower in total at M8), confirming that producer warps/barriers were real overhead, but full-K N256 still loses the concurrency supplied by split-K N128 tasks.
- Evidence: `results/v4_flash_tp_wgmma_pair2wg_tp4_m8_coldl2_abc_20260903.log`.
- Decision: reject the full-K paired W13 family and keep it opt-in only. Do not spend another iteration on this same fusion topology; pivot back to the accepted split-K path or W2/collective overlap.

### Iteration 83a — W2-progress multicast prototype deadlocks

- Added an opt-in producer-progress path without fragmenting the W2 grid. Each completed W2 N128 tile publishes route completion; N1024 chunks enter a device ready queue. Eight low-footprint CTAs on a second stream reduce local k6 chunks and multicast them without remote waits, followed by a separate 78-CTA rank-reduction/cleanup kernel.
- The extension compiles and the four-rank harness initializes on H20 GPUs 1–4, but the first M8 random-route warmup does not return after more than two minutes and emits no correctness record. The targeted process was interrupted; no latency sample exists.
- Evidence: `results/iter83_w2_progress_tp4_m8_compile_correctness_smoke_coldl2_20260903.log`.
- Classification: synchronization failure. Keep the path opt-in. Before another end-to-end attempt, run a no-communication publication probe that exposes tile counts, chunk counts, queue tail/valid entries, and worker claim completion; do not tune performance while any count is missing.

### Iteration 83b — W2 progress publication is complete

- Added a single-rank diagnostic that runs the exact progress-enabled W2 specialization, copies its state after synchronization, and compares the route output bitwise against the selected W2 kernel.
- TP4-shape M8 random routing passes: all 256 `(token,N128)` counters equal 6, all 32 `(token,N1024)` counters equal 8, queue tail and valid sum both equal 32, and the queue is an exact permutation of all 32 tasks. The progress W2 output is finite and bitwise equal to control.
- Evidence: `results/iter83b_w2_progress_publication_probe_20260903.log`.
- Finding: iteration 83a is not caused by missing W2 publications. Isolate local worker completion without communication next, then inspect multicast phase/slot layout only if every worker claim terminates.

### Iteration 83c0 — worker probe launcher rejected `--m`

- Added worker-exit accounting and a TP4 diagnostic for all four multicast source-slot ranges. The diagnostic command did not execute: this torchrun parser treated the script's `--m` as an ambiguous launcher abbreviation for `--master-*`.
- Evidence: `results/iter83c0_w2_progress_worker_launcher_error_20260903.log`.
- Classification: launcher error, not a kernel result. Retry with an explicit `--` separator before the training script.

### Iteration 83c1 — worker probe phase-read host error

- The corrected torchrun invocation initializes all four ranks and builds the new extension, but the diagnostic exits before launching the worker: PyTorch CUDA does not implement `bitwise_and` for SGLang's uint32 push counter.
- Evidence: `results/iter83c1_w2_progress_worker_phase_read_error_20260903.log`.
- Classification: host diagnostic error, not a synchronization result. Read the scalar with `.item()` before applying integer parity and rerun unchanged device code.

### Iteration 83c2 — worker-only probe remains silent

- Fixed the uint32 phase read and reran the worker-only TP4 probe with the finish kernel disabled. The process produced no stage/result line for more than two minutes and was interrupted.
- Evidence: `results/iter83c2_w2_progress_worker_multicast_probe_20260903.log`.
- Classification: probable device-side stall, but this version prints only after model setup and final synchronization, so it cannot yet distinguish slow preprocessing from worker non-termination. Add rank-synchronized stage markers immediately before worker launch, after W2 launch, and after the worker event; also run a zero-worker/control setup to bound initialization time.

### Iteration 83c3 — sequential worker and multicast coverage pass

- Added explicit stage markers and ran progress W2 to completion before starting eight queue consumers, with the finish kernel still disabled.
- All four ranks pass. Each rank reports tile counts 6, chunk counts 8, queue tail 32, worker claim 40 (=32 tasks+8 exit claims), worker-done 8, and a complete task permutation. In every rank-local symmetric workspace, all 16,384 uint32 words are nonzero for each of the four source slots, proving the multicast address, source/phase offsets, zero sentinel, and worker termination are correct in sequential execution.
- Evidence: `results/iter83c3_w2_progress_worker_sequential_probe_20260903.log`.
- Finding: the original deadlock boundary is concurrency, not queue completeness or multicast coverage. Run the same marked probe with worker/W2 concurrency; if it stalls after launches are submitted, replace spinning resident consumers with programmatic launch/dependency or a nonresident progress mechanism rather than touching the verified publication math.

### Iteration 83c4 — one concurrent polling CTA still stalls

- Reused the marked worker probe in concurrent mode with only one 128-thread worker CTA per GPU. Setup completes and both launches are submitted, but the worker event never completes; the process was interrupted after a bounded observation window.
- Evidence: `results/iter83c4_w2_progress_worker1_concurrent_probe_20260903.log`.
- Finding: this is not occupancy pressure from eight workers. A resident tight acquire-poll loop issued before W2 creates a forward-progress dependency the scheduler does not resolve here. Do not sweep worker counts. Add nanosleep/backoff plus a finite device timeout/error word so the consumer can yield/exit and expose whether W2 publication begins only after the polling kernel retires.

### Iteration 83c5 — producer-first concurrent worker passes

- Changed only host submission order: record a pre-W2 event, submit the finite full-grid W2 producer on the main stream, then submit the polling worker on the auxiliary stream with a dependency on that earlier event. The worker remains eligible to overlap W2 but cannot prevent producer admission.
- With one worker CTA, all four ranks complete. Counts and queue are exact, worker claim/done are 33/1, and every rank-local workspace contains all 16,384 nonzero words for all four source slots.
- Evidence: `results/iter83c5_w2_first_worker1_concurrent_probe_20260903.log`.
- Decision: accept producer-first ordering as the synchronization repair. Verify eight workers under the same probe before re-enabling the finish kernel and repeated CUDA Graph replay.

## Iteration 83c6 — producer-first progress path at full worker concurrency

- **Change under test:** no source change; raised the TP4 concurrent progress consumer launch from one block to eight blocks.
- **Command:** `torchrun --standalone --nproc-per-node=4 -- bench/test_v4_flash_tp_w2_progress_worker.py --m 8 --workers 8 --launch-mode concurrent --route-pattern random` on GPUs 1–4.
- **Result:** PASS on all four ranks. Every rank reported tile counters `6`, chunk counters `8`, queue tail `32`, a full task permutation, `worker_claim=40`, and `worker_done=8`.
- **Communication evidence:** every local symmetric workspace contained all `16384/16384` nonzero words for each of the four source-rank slots.
- **Decision:** accept the producer-first concurrent protocol at the intended eight-worker occupancy. This is a correctness/progress probe only, not a performance result.
- **Next:** exercise the finish kernel and repeated CUDA Graph replay end to end, then cold-L2 A/B against the accepted path.

## Iteration 83d — end-to-end progress overlap cold-L2 graph screen

- **Change under test:** eight-block producer-progress W2 + local-k6/multicast consumers + finish kernel versus the accepted stock-CARv2 control, in the same process and communicator.
- **Method:** TP4 on GPUs 1–4, random routes, `M=8`, two order-balanced outer batches, five timed graph replays per implementation per batch. A separate 256 MiB L2 clear ran immediately before every replay and outside CUDA events.
- **Correctness:** PASS on all ranks. Candidate and control had identical reported accuracy (`cosine_min_rank=0.999995565`, `rel_l2_max_rank=0.00297848`) and `fused_vs_control_max_abs=0.0`.
- **Cold latency:** accepted control median `0.079488 ms` (min `0.078528`, max `0.088160`); progress candidate median `0.083920 ms` (min `0.083488`, max `0.105952`).
- **Result:** control/candidate `0.947188x`; the progress path is `5.58%` slower than control at M=8.
- **Decision:** reject this eight-worker progress implementation as a default. The concurrency protocol is valid, but W2 publication/fence/counter cost plus worker/finish overhead exceeds the overlap benefit.
- **Next:** isolate producer-only W2 tax from worker/finish cost before deciding whether a cheaper publication granularity is viable.

## Iteration 83e — M8 progress-worker concurrency cold-L2 sweep

- **Change under test:** no source change; swept producer-progress consumer blocks `1,2,4,16` at TP4 M8. Iteration 83d supplies the directly comparable 8-block point.
- **Method:** each point used a fresh four-rank process, random routes, same-process order-balanced control/candidate CUDA Graph A/B, two outer batches and five timed replays per implementation per batch. Every replay had a separate 256 MiB L2 clear outside the CUDA event.
- **Correctness:** every point passed all-rank numerical checks and was bitwise identical to its control (`fused_vs_control_max_abs=0`).
- **Cold medians:** workers 1: control/candidate `79.648/124.704 us` (`0.63870x`); 2: `79.968/100.896 us` (`0.79258x`); 4: `79.504/89.296 us` (`0.89034x`); 16: `79.312/81.664 us` (`0.97120x`). The prior eight-worker point was `79.488/83.920 us` (`0.94719x`).
- **Interpretation:** candidate latency improves monotonically through 16 workers. Queue-drain/finish tail dominates more than W2 resource interference in this range; 16 workers narrows the deficit to `2.352 us` (`2.97%` slower).
- **Decision:** retain only as an opt-in experiment. Continue to 24/32 workers to test whether the curve crosses control.

## Iteration 83f0 — invalid 24-worker screen invocation

- **Attempt:** requested progress-worker counts `24,32` in the existing cold-L2 A/B harness.
- **Failure:** argparse rejected `24`; the supported set is `{1,2,4,8,16,32}`. `set -e` stopped the loop before the 32-worker case, so no GPU performance result was produced.
- **Decision:** this is a benchmark-invocation error, not a kernel result. Keep the harness restriction and rerun the valid 32-worker endpoint.

## Iteration 83f1 — invalid odd outer-count invocation

- **Attempt:** valid 32-worker endpoint with `outer=3`, `replays=7`.
- **Failure:** the order-balanced A/B harness requires a positive even outer count and rejected the invocation before GPU execution.
- **Decision:** no kernel or performance result; rerun unchanged with `outer=4`.

## Iteration 83f2 — 32-worker progress endpoint cold-L2 screen

- **Change under test:** no source change; measured the maximum supported 32 progress-worker blocks at TP4 M8.
- **Method:** same-process, same-communicator, order-balanced CUDA Graph A/B with four outer batches and seven timed replays each; separate 256 MiB cold-L2 clear immediately before every replay and outside timing.
- **Correctness:** PASS on all ranks and bitwise equal to control (`fused_vs_control_max_abs=0`).
- **Cold latency:** control median `0.079104 ms` (min `0.077856`, max `0.088768`); candidate median `0.081152 ms` (min `0.080192`, max `0.086656`).
- **Result:** control/candidate `0.974763x`; candidate remains `2.59%` slower by `2.048 us`. This is only a small improvement over the 16-worker point and does not cross control.
- **Decision:** reject the current progress protocol as a default. Stop worker-count sweeping and isolate the producer-only publication tax.

## Iteration 83g — isolate the producer-only W2 publication tax

- **Benchmark change:** extended the single-rank progress probe with order-balanced CUDA Graph A/B timing. Control and candidate share the same tensors and both capture an identical progress-state reset, so only `run_w2` versus `run_w2_progress` differs.
- **Method:** TP4-local M8 random routes on GPU1, four alternating outer batches × 200 samples per implementation. A separate 256 MiB L2 clear ran immediately before every graph replay and outside CUDA events.
- **Correctness:** publication counts and queue permutation pass; progress and control W2 route outputs are bitwise equal both eagerly and through the captured graphs.
- **Cold W2 latency:** control median `29.792 us` (min `28.992`, max `32.000`); progress median `33.120 us` (min `31.872`, max `35.008`). Control/progress is `0.899517x`.
- **Finding:** per-tile thread fences/counters/queue publication add `3.328 us`, or `11.17%` of the ordinary W2 stage. The full 32-worker experiment regressed by only `2.048 us`, so overlap hides about `1.28 us` but cannot repay producer publication.
- **Decision:** stop consumer-count tuning. Any viable overlap successor must reduce publication frequency or synchronization scope before another end-to-end screen.

## Iteration 83h0 — first TMA-store progress epilogue does not compile

- **Change:** staged each progress W2 `[8,128]` BF16 tile in dead weight shared memory, then attempted elected-lane 1D TMA stores followed by `tma_store_wait<0>` before ready publication. The ordinary W2 path is unchanged.
- **Rationale:** this follows DeepGEMM MegaMoE's shared-to-TMA-store-to-notify epilogue ordering and is intended to replace 128 per-thread device fences with an explicit asynchronous-copy completion boundary.
- **Failure:** NVCC stopped before GPU execution because the unqualified helper name `tma_store_1d` is undefined at the call site. No correctness or performance result exists.
- **Decision:** retain this failed compile as evidence, resolve the helper's actual namespace from the read-only DeepGEMM header, and make only that qualification repair before retrying.

## Iteration 83h1 — TMA-store progress epilogue compiles and passes

- **Repair:** qualified the DeepGEMM helper as `ptx::tma_store_1d`. The progress-only W2 path stages `[8,128]` BF16 in shared memory, issues one 256-byte TMA store per valid route, waits for bulk-group completion, then publishes counters/queue entries from the elected lane.
- **Correctness:** TP4-local M8 random publication counts, queue permutation, eager output, and captured-graph output all pass; progress W2 is bitwise equal to ordinary W2.
- **Method:** four order-balanced outer batches × 200 graph replays per implementation, identical captured state reset on both sides, separately cold L2 before every replay.
- **Cold W2 latency:** control median `29.872 us` (min `29.056`, max `32.096`); TMA-progress median `32.800 us` (min `31.776`, max `34.816`), control/progress `0.910732x`.
- **Finding:** TMA lowers the publication tax from iteration 83g's `3.328 us` to `2.928 us`, recovering only about `0.40 us`. Per-thread fences were a cost, but per-CTA completion and multi-level publication remain dominant.
- **Decision:** keep the TMA ordering for one end-to-end TP4 screen; it is still opt-in and not a selected path.

## Iteration 83h2 — end-to-end TP4 TMA-progress screen

- **Change under test:** iteration 83h1's shared/TMA W2 epilogue with 32 concurrent progress workers versus the accepted stock-CARv2 graph.
- **Method:** TP4 M8 random routes, same process/data/communicator, four order-balanced outer batches × 20 graph replays per implementation; separate excluded 256 MiB cold-L2 clear before every replay.
- **Correctness:** both paths pass all-rank reference and all-reduce checks and are bitwise identical (`fused_vs_control_max_abs=0`).
- **Cold latency:** control median `0.079376 ms` (min `0.077888`, max `0.102880`); TMA-progress median `0.081152 ms` (min `0.079744`, max `0.126304`). Control/candidate is `0.978115x`.
- **Finding:** TMA preserves about `0.272 us` of the isolated improvement end to end, narrowing the old 32-worker deficit from `2.048 us` to `1.776 us`, but remains `2.24%` slower.
- **Decision:** still reject as a selected path. Replace the multi-level atomic ready queue with per-route tile release markers and static chunk ownership.

## Iteration 83i — direct route/tile marker producer gate

- **Change:** replaced the progress path's tile counts, chunk counts, ready queue, valid flags, and queue-tail atomics with one release marker per unique `(route,N128)` producer. Static token/N1024 workers wait in parallel on their 6x8 markers. Progress state is now `M*6*32+2` int32 values.
- **Correctness:** TP4-local M8 random produces exactly 1,536 markers, all equal to one; eager and captured progress W2 outputs are bitwise equal to ordinary W2.
- **Method:** four order-balanced outer batches × 200 graph replays per implementation, identical captured state reset on both sides, separately cold L2 before every replay.
- **Cold W2 latency:** control median `29.728 us` (min `28.960`, max `31.840`); direct-marker progress median `32.416 us` (min `31.424`, max `34.816`), control/candidate `0.917078x`.
- **Finding:** removing the multi-level queue recovers another `0.384 us` versus iteration 83h1 and about `0.64 us` versus the original progress producer, but `2.688 us` remains. Shared staging and per-CTA TMA completion dominate more than the removed atomics.
- **Decision:** protocol remains opt-in. Validate the static worker/multicast state on TP4 before an end-to-end timing screen.

## Iteration 83j — direct-marker TP4 concurrent worker gate

- **Change under test:** no source change; ran the static 32-worker direct-marker consumer concurrently with producer-first W2 on four ranks.
- **Result:** PASS on every rank. Each reports all 1,536 route/N128 markers equal to one, `task_done=32`, and `worker_done=32`.
- **Communication evidence:** every rank-local symmetric workspace contains `16384/16384` nonzero words for each of the four source slots; no task or multicast region is missing.
- **Decision:** accept the direct-marker protocol's TP4 concurrency correctness. Proceed to a complete finish-kernel/CUDA-Graph cold-L2 A/B; no performance claim from this probe.

## Iteration 83k — direct-marker end-to-end TP4 screen

- **Change under test:** direct release markers plus 32 static progress workers and the finish kernel versus accepted stock CARv2.
- **Method:** TP4 M8 random routes, same process/data/communicator, four order-balanced outer batches × 20 graph replays per implementation; separate excluded 256 MiB cold-L2 clear before each replay.
- **Correctness:** both paths pass every all-rank reference/all-reduce check and are bitwise identical (`fused_vs_control_max_abs=0`).
- **Cold latency:** control median `0.079200 ms` (min `0.077728`, max `0.105120`); candidate median `0.080560 ms` (min `0.078688`, max `0.110144`). Control/candidate is `0.983118x`.
- **Finding:** direct markers recover another `0.416 us` end to end versus TMA plus the dynamic queue and reduce the original 32-worker deficit from `2.048 us` to `1.360 us`. All four candidate batch medians remain above control, so the `1.72%` regression is stable.
- **Decision:** reject as default. Before testing a cheaper CTA-leader fence publication, establish that its cross-thread/device-scope happens-before chain is valid under the CUDA memory model.

## Iteration 83l — direct global stores plus release markers

- **Change:** removed progress-only shared staging, TMA stores, and explicit device fences. All W2 lanes use the ordinary direct BF16 global-store epilogue, then `__syncthreads()` establishes strong-happens-before into each route lane's `st.release.gpu` marker; workers retain matching `ld.acquire.gpu` loads.
- **Memory-model basis:** CUDA documents that block synchronization strongly-happens-before participating threads resume, while scoped release/acquire atomics publish prior memory actions at device scope. The chain is therefore defined rather than an empirical visibility shortcut.
- **Correctness:** TP4-local M8 random emits all 1,536 markers exactly once, and eager/captured progress W2 outputs are bitwise equal to ordinary W2.
- **Method:** four order-balanced outer batches × 200 graph replays per implementation, identical state-reset graph node, separate excluded 256 MiB cold-L2 clear before every replay.
- **Cold W2 latency:** control median `29.536 us` (min `28.704`, max `31.360`); release-marker progress median `30.880 us` (min `29.952`, max `33.024`), control/candidate `0.956477x`.
- **Finding:** producer tax falls to `1.344 us`, half iteration 83i's `2.688 us` and `1.984 us` below the original queue implementation. This reaches the range previously hidden by communication overlap.
- **Decision:** advance to the TP4 concurrent worker gate and an end-to-end cold-L2 A/B; remain opt-in until both pass.

## Iteration 83m — release-marker TP4 concurrent worker gate

- **Change under test:** no source change; ran the direct-global-store/release-marker producer with all 32 static consumers concurrently on TP4.
- **Result:** PASS on all ranks. Every rank reports 1,536 markers all equal to one, `task_done=32`, and `worker_done=32`.
- **Communication evidence:** all four source regions in every local symmetric workspace contain `16384/16384` nonzero words; the acquire readers observe complete W2 route tiles before multicast.
- **Decision:** concurrency correctness passes. Proceed to complete finish-kernel and repeated cold-L2 CUDA Graph timing.

## Iteration 83n — release-marker overlap reaches near parity

- **Change under test:** direct global W2 stores, release markers, 32 static N1024 workers, and finish kernel versus accepted stock CARv2.
- **Method:** TP4 M8 random routes, same process/data/communicator, four balanced AB/BA batches × 50 graph replays per implementation; separate excluded 256 MiB cold-L2 clear before every replay.
- **Correctness:** all-rank reference and all-reduce checks pass; candidate and control graph outputs are bitwise identical.
- **Cold latency:** control median `0.079040 ms` (min `0.077664`, max `0.095744`); candidate median `0.079424 ms` (min `0.078144`, max `0.088800`). Control/candidate is `0.995165x`.
- **Finding:** the end-to-end deficit is now only `0.384 us` or `0.49%`, down from iteration 83f2's `2.048 us`. All four candidate batch medians still lose narrowly, so this is near parity rather than a selected win.
- **Decision:** keep opt-in. Sweep N2048/N1024/N512 worker chunk granularity; finer N512 readiness needs to hide only another 0.4 us to cross control.

### Iteration 83o — parameterized W2 progress chunks; TP4 chunks=8/64-worker protocol validation (2026-09-03)

- Change: parameterized the release-marker progress worker at 2/4/8 N-chunks. The dispatch maps chunks 2/4/8 to 256/128/64 threads per task, respectively, and permits up to 64 persistent worker blocks.
- Motivation: the accepted 4-chunk/32-worker overlap prototype remained 0.384 us slower than control at M=8. Eight N=512 chunks expose 64 tasks and allow work to start after four N=128 release markers, testing whether finer readiness granularity can hide the remaining launch/progress cost.
- Validation command: TP4, GPUs 1-4, M=8, 64 workers, 8 chunks, concurrent producer/worker launch, random routing.
- Result: PASS on all four ranks. Every rank observed marker_sum=1536 with marker_min=marker_max=1, task_done=64, worker_done=64, and all four symmetric source slots contained the expected 16384 nonzero words.
- Correctness scope: validates the device-scope release/acquire publication protocol and chunk/task completion under concurrent TP4 execution. End-to-end numerical and cold-L2 performance selection follows separately.
- Artifact: results/iter83o_chunks8_worker64_tp4_m8_compile_concurrent_probe_20260903.log

### Iteration 83p — cold-L2 M=8 progress chunk/worker sweep (2026-09-03)

- Method: same-process CUDA-Graph A/B against the stock W2 + local k6 reduction + SGLang CARv2 control, TP4 on GPUs 1-4. A separate 256 MiB cache clear ran immediately before every replay and outside CUDA events. Each arm received 4 x 50 = 200 cold samples; AB/BA batch order was balanced.
- Correctness: all three candidates passed the distributed reference checks and were bitwise identical to their paired control graph (fused_vs_control_max_abs=0).
- 2 chunks / 16 workers: control 79.072 us, candidate 79.632 us, control/candidate 0.992968 (candidate +0.560 us).
- 4 chunks / 32 workers: control 79.264 us, candidate 79.616 us, control/candidate 0.995579 (candidate +0.352 us).
- 8 chunks / 64 workers: control 79.264 us, candidate 79.456 us, control/candidate 0.997584 (candidate +0.192 us).
- Decision: reject all three as defaults because none beats control. Retain 8 chunks / 64 workers as the best progress-overlap research point; finer readiness reduced the deficit versus the previously confirmed 4-chunk prototype, but the remaining fixed worker/finish overhead still exceeds the hidden communication time.
- Artifacts:
  - results/iter83p_progress_chunks2_workers16_tp4_m8_cold_ab_20260903.log
  - results/iter83p_progress_chunks4_workers32_tp4_m8_cold_ab_20260903.log
  - results/iter83p_progress_chunks8_workers64_tp4_m8_cold_ab_20260903.log

### Iteration 83q0 — inline-finish compile/run reached host-probe dtype failure (2026-09-03)

- Change: added an opt-in inline finish to the W2 release-marker worker. After every local worker has multicast its chunks, the same resident grid polls all four symmetric source slots, performs the rank sum, clears the slots, and advances all 78 CARv2 phase counters. The legacy separate finish kernel remains available.
- Intended benefit: remove one CUDA-Graph kernel node and reuse the already resident 8-chunk/64-worker grid.
- TP4 M=8 concurrent probe: the new extension compiled and all GPU work reached worker_complete without deadlock.
- Failure: the host-side probe then called min/max on the uint32 CARv2 counter tensor; this PyTorch build reports NotImplementedError: min_all not implemented for UInt32 on every rank. Therefore no counter/slot correctness verdict is claimed from this run.
- Decision: implementation remains unaccepted. Repair only the probe by viewing counters as int32, then repeat the identical gate before benchmarking.
- Artifact: results/iter83q_inline_finish_chunks8_workers64_tp4_m8_compile_probe_20260903.log

### Iteration 83q1 — inline-finish TP4 protocol gate passes (2026-09-03)

- Probe repair: view the host copy of the uint32 CARv2 phase counters as int32 before min/max; no CUDA implementation change from 83q0.
- Configuration: TP4 GPUs 1-4, M=8, random routes, 8 chunks, 64 workers, producer and inline-finish worker launched concurrently.
- Result: PASS on all four ranks without deadlock. Every rank reported marker_sum=1536 with marker_min=marker_max=1, task_done=64, worker_done=64, counter_min=counter_max=1, all four source slots fully cleared (zero nonzero words), and finite output.
- Scope: establishes local publication, local resident-grid gate, remote polling/reduction completion, slot cleanup, and CARv2 phase advancement. Numerical equivalence is intentionally deferred to the same-process end-to-end graph A/B gate.
- Artifact: results/iter83q1_inline_finish_chunks8_workers64_tp4_m8_probe_20260903.log

### Iteration 83q2 — inline finish wins first cold-L2 M=8 end-to-end gate (2026-09-03)

- Method: same-process CUDA-Graph A/B against stock W2 + local k6 reduction + SGLang CARv2, TP4 GPUs 1-4, random routes, 8 chunks / 64 workers. Separate 256 MiB cache clear immediately preceded every replay outside timing. Four balanced AB/BA batches of 50 supplied 200 cold samples per arm.
- Correctness: candidate and control both passed the distributed reference; candidate versus control graph output was bitwise identical (max_abs=0).
- Control median: 78.928 us. Inline-finish candidate median: 78.208 us.
- Result: control/candidate = 1.009206; candidate wins by 0.720 us (0.912% of control latency).
- Decision: first successful end-to-end result for the release-marker overlap branch. Keep opt-in pending a longer confirmation window and M=16/M=32 generalization; do not yet alter the accepted default dispatch.
- Artifact: results/iter83q2_inline_finish_chunks8_workers64_tp4_m8_cold_ab_20260903.log

### Iteration 83q3 — long cold-L2 inline-finish confirmation and shape boundary (2026-09-03)

- Method: same-process TP4 CUDA-Graph A/B, random routes, 8 chunks / 64 workers / inline finish, 6 x 100 = 600 cold samples per arm and M. Cache clear and correctness policy match 83q2.
- M=8: control 79.040 us, candidate 78.720 us, control/candidate 1.004065; candidate wins by 0.320 us. Candidate/control output max_abs=0.
- M=16: control 123.616 us, candidate 124.896 us, control/candidate 0.989752; candidate loses by 1.280 us. Candidate/control output max_abs=0.
- M=32: control 193.696 us, candidate 199.456 us, control/candidate 0.971121; candidate loses by 5.760 us. Candidate/control output max_abs=0.
- Three-shape geometric mean: control 123.694 us, candidate 125.168 us, control/candidate 0.988220.
- Decision: reject inline finish for M>=16 and as a general progress dispatch. M=8 remains a narrowly positive opt-in point, confirmed over 600 cold samples, but its sub-percent margin warrants one independent repeat before default selection.
- Artifact: results/iter83q3_inline_finish_chunks8_workers64_tp4_m8_m16_m32_cold_long_ab_20260903.log

## Iteration 84 — current selected M=128 W13/W2 detailed NCU baseline (2026-09-03)

- Profiled the exact selected TP4 custom local path at M=128/random routes after the required 256 MiB L2 clear. Nsight Compute cache control was enabled; W13 and W2 route_gemm launches were captured separately with the same detailed section set.
- W13 (K4096,N1024,split2): 195.90 us, 2.89 TB/s, 60.09% DRAM-throughput utilization, 75.22% compute utilization, 77.26% cycles with an eligible warp, 54 registers/thread, no spills, 7.32 waves/SM. L2 hit rate is 5.12%.
- W2 (K512,N4096,split1): 107.01 us, 2.66 TB/s, 55.19% DRAM-throughput utilization, 74.54% compute utilization, 77.18% cycles with an eligible warp, 54 registers/thread, no spills, 14.63 waves/SM. L2 hit rate is 9.38%.
- NCU flags uncoalesced global access: W13 has 132,864 excessive sectors (11% of 1,224,096; estimated 10.56% bound), while W2 has 189,824 excessive sectors (19% of 973,632; estimated 18.83% bound).
- Finding: neither core is at the H20 HBM ceiling, and simply deepening shared stages was already rejected. The highest-value bounded follow-up is to resolve NCU source attribution for the excessive sectors and distinguish useful TMA scale overfetch from avoidable activation/metadata access before attempting interleaving.
- Artifacts: bench/results/iter84_current_w13_m128_coldl2_detailed_ncu.{log,ncu-rep}, bench/results/iter84_current_w2_m128_coldl2_detailed_ncu.{log,ncu-rep}, and the corresponding iter84_current_*_m128_ncu_details.log files.

## Iteration 85 — Sorted-position W2 activation and scale layout correctness gate

- **Hypothesis:** NCU attributed 66,432 excessive W2 sectors to the eight-lane route-scale gather. Writing the post-SwiGLU FP8 activation in aligned expert-mblock order and storing scales as `[mblock, K128, slot]` should turn each eight-scale load into one contiguous 32-byte access without changing route semantics.
- **Change:** Added opt-in `V4_W2_SORTED_ACT=1`. Fused route alignment emits `route_to_sorted`; fused SwiGLU/quant writes FP8 rows by sorted position and scales by mblock/K-tile/slot; W2 reads those positions while retaining original route IDs for output and k6 reduction. Default remains disabled.
- **Verification:** TP4 balanced M8, TP4 skew M8, and TP8-shape balanced M8 all passed the full all-route reference. W2 cosine was 0.999997256, 0.999997235, and 0.999997278 respectively; all outputs finite. Fused route/quant remained bit-exact to its reference.
- **Decision:** Correctness and TP8-runnable gate passed. Performance is not yet measured, so this remains an opt-in candidate and is not selected.
- **Artifact:** `results/iter85_w2_sorted_act_correctness_20260903.log`.

## Iteration 85b — TP4 M128 cold-L2 A/B/A/B screen of sorted W2 layout

- **Method:** Four independent CUDA-Graph runs in control/candidate/control/candidate order, random routes, TP4 on GPUs 1–4. Each run used 3×100 samples; every replay had a separate 256 MiB L2 clear excluded from CUDA events; rank-max latency reported.
- **Control medians:** 0.345328 and 0.348320 ms; paired geometric mean 0.346821 ms.
- **Candidate medians:** 0.341680 and 0.356272 ms; paired geometric mean 0.348900 ms.
- **Result:** control/candidate = 0.994041x; the candidate is 0.599% slower overall. Individual pair direction changed, so the apparent scale-sector saving did not produce stable end-to-end latency improvement.
- **Decision:** Reject `V4_W2_SORTED_ACT=1`; keep it opt-in and keep the formal default disabled. Next isolate scale-only coalescing or attack the larger scalar W2 output-store sector waste.
- **Artifact:** `results/iter85b_w2_sorted_act_tp4_m128_cold_abba_20260903.log`.

## Iteration 86 — Shared-memory coalesced W2 epilogue correctness gate

- **Hypothesis:** NCU attributed 123,392 excessive W2 sectors to the four scalar BF16 route-output assignments. Staging each 8×128 CTA tile through 2 KiB shared memory lets one half warp per route emit aligned 16-byte stores while preserving the route-major tensor required by SGLang's k6 reducer.
- **Change:** Added opt-in `V4_W2_COALESCED_STORE=1`; the WGMMA accumulator owners write the 8×128 BF16 tile to shared memory, synchronize once, then 128 threads issue contiguous `uint4` global stores. Default remains disabled.
- **Verification:** TP4 balanced M8, TP4 skew M8, and TP8-shape balanced M8 all passed the full all-route reference. W2 cosine was 0.999997256, 0.999997235, and 0.999997278 respectively; all outputs finite.
- **Decision:** Correctness and TP8-runnable gate passed. Candidate remains opt-in until cold-L2 TP4 A/B.
- **Artifact:** `results/iter86_w2_coalesced_store_correctness_20260903.log`.

## Iteration 86b — TP4 M128 cold-L2 A/B/A/B screen of coalesced W2 epilogue

- **Method:** Four independent CUDA-Graph runs in control/candidate/control/candidate order, random routes, TP4 on GPUs 1–4. Each run used 3×100 samples; every replay had a separate 256 MiB L2 clear excluded from CUDA events; rank-max latency reported.
- **Control medians:** 0.354288 and 0.346304 ms; paired geometric mean 0.350273 ms.
- **Candidate medians:** 0.343968 and 0.350880 ms; paired geometric mean 0.347407 ms.
- **Result:** control/candidate = 1.008251x (0.825% preliminary gain). The first pair favored the candidate by 3.00% but the second disfavored it by 1.30%, so run-to-run drift is larger than the net result.
- **Decision:** Do not select yet. Retain opt-in and run a longer single-process-window comparison before testing other M values.
- **Artifact:** `results/iter86b_w2_coalesced_store_tp4_m128_cold_abba_20260903.log`.

## Iteration 86c — In-process paired cold-L2 graph comparison harness

- **Problem:** Independent torchrun invocations showed run-to-run drift larger than the sub-percent W2 epilogue signal.
- **Change:** Added `bench/compare_v4_flash_tp_w2_store.py`. It loads control and candidate compile-time variants into one TP process set, captures two CUDA Graphs over shared weights/routes, alternates A/B then B/A per sample, performs an independent 256 MiB L2 clear immediately before every replay, and rank-max reduces both event arrays.
- **Verification:** Remote Python bytecode compilation passed. GPU protocol and numerical equality are intentionally gated by the first paired run.
- **Decision:** Benchmark infrastructure only; no performance selection.

## Iteration 86d0 — Paired harness invocation failure

- **Attempt:** Launch the TP4 M128 600-sample paired cold-L2 comparison.
- **Failure:** `torchrun` consumed the benchmark's `--m` as an ambiguous launcher option because the training-script argument separator was omitted. No GPU benchmark samples were collected.
- **Resolution:** Rerun with `torchrun ... -- bench/compare_v4_flash_tp_w2_store.py ...`.
- **Artifact:** `results/iter86d_w2_store_paired_tp4_m128_cold_600_20260903.log`.

## Iteration 86d1 — In-process paired TP4 M128 cold-L2 W2-store result

- **Method:** Control and coalesced-store candidate were loaded together, captured as two graphs over identical random routes/weights, and replayed in alternating A/B then B/A order. Six batches × 100 samples per variant; each individual replay had its own 256 MiB L2 clear excluded from events; four-rank maximum latency.
- **Correctness:** Outputs were bitwise equal on all four ranks after alternating the graphs.
- **Result:** control median 0.356272 ms; candidate median 0.358176 ms; control/candidate 0.994684x. Every batch median was 1.36–2.37 microseconds slower for the candidate.
- **Decision:** Reject `V4_W2_COALESCED_STORE=1` and retain scalar direct stores as default. The extra shared-memory traffic plus CTA barrier costs more than the global-sector reduction.
- **Artifact:** `results/iter86d1_w2_store_paired_tp4_m128_cold_600_20260903.log`.

## Iteration 87 — Warp-private coalesced W2 epilogue correctness gate

- **Hypothesis:** Iteration 86's sector reduction was outweighed by a CTA-wide barrier. Each warp already owns disjoint 16-column spans in both N64 groups, so a warp-private eight-route × 32-column shared slice can reorder stores using only `syncwarp`.
- **Change:** Reworked the opt-in coalesced epilogue into four independent 512-byte warp buffers. Each group of four lanes writes one route's two N16 spans with aligned `uint4` stores; the CTA-wide barrier was removed.
- **Verification:** TP4 balanced M8, TP4 skew M8, and TP8-shape balanced M8 all passed the full all-route reference with W2 cosine 0.999997256, 0.999997235, and 0.999997278; all outputs finite.
- **Decision:** Correctness and TP8-runnable gate passed; performance remains opt-in pending paired cold-L2 measurement.
- **Artifact:** `results/iter87_w2_warp_store_correctness_20260903.log`.

## Iteration 87b — Paired TP4 M128 cold-L2 warp-private W2 epilogue

- **Method:** Same-process dual graph, identical random routes/weights, alternating A/B then B/A, 6×100 samples per variant, separate untimed 256 MiB L2 clear before each replay, four-rank maximum latency.
- **Correctness:** Control and candidate outputs were bitwise equal on all ranks.
- **Result:** control median 0.360528 ms; warp-private candidate 0.363232 ms; control/candidate 0.992556x. Candidate was slower in every batch by 2.16–3.62 microseconds.
- **Decision:** Reject the warp-private shared epilogue too. Removing the CTA barrier did not recover the shared store/load and address-remap cost; direct scalar stores remain selected.
- **Artifact:** `results/iter87b_w2_warp_store_paired_tp4_m128_cold_600_20260903.log`.

## Iteration 88a — Isolated mblock-major W2 scale-layout implementation

- **Hypothesis:** Iteration 85 coupled scale coalescing to a padded sorted FP8 activation layout and regressed. Keeping activation route-major while storing only the small scale tensor as `[mblock,K128,slot]` isolates the 66,432 excessive scale-gather sectors identified by NCU.
- **Change:** Added opt-in `V4_W2_MBLOCK_SCALE=1`, reusing the dynamic `route_to_sorted` inverse map but changing only scale output/read addressing. Extended the paired harness to select a compile-time flag through `V4_COMPARE_FLAG`.
- **Verification:** Local Python syntax compilation passed for the kernel wrapper, graph benchmark, numerical test, and paired harness. CUDA correctness/performance are pending.
- **Decision:** Implementation checkpoint only; default remains disabled.

## Iteration 88b — Mblock-major W2 scale correctness gate

- **Verification:** With `V4_W2_MBLOCK_SCALE=1`, TP4 balanced M8, TP4 skew M8, and TP8-shape balanced M8 all passed the full all-route reference. W2 cosine was 0.999997256, 0.999997235, and 0.999997278; all outputs finite. Input fused quantization remained bit-exact.
- **Coverage:** FP8 activation stayed route-major; only activation-scale production/consumption used the inverse route-to-sorted map. Both heavy padding and six-expert skew were exercised.
- **Decision:** Correctness and TP8-runnable gate passed. Candidate remains opt-in pending paired cold-L2 performance.
- **Artifact:** `results/iter88b_w2_mblock_scale_correctness_20260903.log`.

## Iteration 88c — Paired TP4 M128 cold-L2 mblock-major W2 scales

- **Method:** Same-process control/candidate CUDA Graphs over identical random routes and weights; alternating order; 6×100 samples per variant; independent untimed 256 MiB L2 clear before each replay; four-rank maximum latency.
- **Correctness:** Outputs were bitwise equal on all ranks.
- **Result:** control median 0.358016 ms; scale-layout candidate 0.359520 ms; control/candidate 0.995817x. Candidate was slower in every batch by 0.91–2.59 microseconds.
- **Decision:** Reject `V4_W2_MBLOCK_SCALE=1`. The inverse-map/scale-transpose overhead exceeds the benefit of coalescing the standalone scale gather; keep route-major scales selected.
- **Artifact:** `results/iter88c_w2_mblock_scale_paired_tp4_m128_cold_600_20260903.log`.

## Iteration 89 — Exact route-capacity upper-bound diagnostic

- **Method:** Same-process paired TP4 M128 graphs with identical kernels/data. Control retained the dynamic-safe 321-mblock capacity; candidate resized metadata to the known static route distribution's exact 249 mblocks. Alternating order, 6×100 samples, independent untimed 256 MiB L2 clear, rank-max latency.
- **Correctness:** Outputs were bitwise equal on all ranks.
- **Result:** dynamic-capacity control median 0.359296 ms; exact-capacity candidate 0.358640 ms; control/candidate 1.001829x (+0.183%). Batch differences were sub-microsecond and one batch reversed.
- **Decision:** This is an upper-bound diagnostic, not a selectable dynamic-route result. Empty over-capacity CTAs are not a material source of the remaining ~10 microseconds; do not prioritize a large persistent-scheduler refactor merely to eliminate them.
- **Artifact:** `results/iter89_exact_route_capacity_paired_tp4_m128_cold_600_20260903.log`.

## Iteration 90 — Fold W2 expert-global scale into activation scales

- **Hypothesis:** Normalized MXFP4 W2 repeatedly loads one expert-global scale per N128 CTA and multiplies it into eight activation scales for every K128 tile. W2 linearity permits doing the same FP32 product once per route/group in the existing fused SwiGLU/FP8 quantizer.
- **Change:** Added opt-in `V4_W2_FOLD_GLOBAL_SCALE=1`. The quantizer stores `group_scale * w2_global_scale[expert]` while retaining the unfused group scale for FP8 conversion; W2 compiles out its expert-global load/multiply. W13 is unchanged because SwiGLU prevents this reassociation.
- **Verification:** TP4 balanced/skew M8 and TP8-shape balanced M8 all passed the full all-route reference. W2 cosine was 0.999997256, 0.999997235, and 0.999997278; outputs finite.
- **Decision:** Correctness and TP8-runnable gate passed; remain opt-in pending paired cold-L2 timing.
- **Artifact:** `results/iter90_w2_fold_global_scale_correctness_20260903.log`.

## Iteration 90b — Paired TP4 M128 cold-L2 folded W2 expert scale

- **Method:** Same-process control/candidate graphs over identical random routes/weights, alternating order, 6×100 samples per variant, separate untimed 256 MiB L2 clear before each replay, four-rank maximum latency.
- **Correctness:** Outputs were bitwise equal on all ranks.
- **Result:** control median 0.358560 ms; folded-scale candidate 0.358544 ms; control/candidate 1.000045x, only 0.016 microseconds. Batch direction was mixed.
- **Decision:** Treat as neutral and reject for selection. Removing repeated W2 expert-scale loads/multiplies merely moves equivalent work into the quantizer and does not shorten the graph critical path.
- **Artifact:** `results/iter90b_w2_fold_global_scale_paired_tp4_m128_cold_600_20260903.log`.

## Iteration 91 — Distributed W13 preparation correctness gate

- **Hypothesis:** Selected M128 W13 spends 32.9% of average issue distance stalled at CTA barriers. Warp 0 currently performs TMA issue, all eight activation-scale gathers, and leader-only mbarrier polling while sibling warps arrive early.
- **Change:** Added opt-in `V4_W13_DISTRIBUTED_PREP=1`: warp 1 lane 0 issues W13 bulk TMA, warp 0 retains the mbarrier wait, and two activation scales are loaded by each warp. W2 and all math/data bytes are unchanged.
- **Verification:** TP4 balanced/skew M8 and TP8-shape balanced M8 passed the full all-route reference with the same W13/activation/W2 cosines as control; all outputs finite.
- **Decision:** Correctness and TP8-runnable gate passed; candidate remains opt-in pending paired cold-L2 timing.
- **Artifact:** `results/iter91_w13_distributed_prep_correctness_20260903.log`.

## Iteration 91b — Paired TP4 M128 cold-L2 distributed W13 preparation

- **Method:** Same-process control/candidate graphs over identical random routes/weights; alternating A/B then B/A; 6×100 samples per variant; separate untimed 256 MiB L2 clear before every replay; four-rank maximum latency.
- **Correctness:** Graph outputs were bitwise equal on all ranks.
- **Result:** control median 0.347104 ms; distributed-prep candidate 0.343808 ms; control/candidate 1.009587x (+0.959%, 3.296 microseconds). Candidate beat control in all six batch medians by 3.09–5.06 microseconds.
- **Decision:** First credible new core gain after the NCU audit. Retain opt-in and test all five M values before selection; also profile W13 to confirm barrier-stall reduction rather than relying only on the mechanism hypothesis.
- **Artifact:** `results/iter91b_w13_distributed_prep_paired_tp4_m128_cold_600_20260903.log`.

## Iteration 91c — Five-shape paired selection audit for distributed W13 prep

- **Method:** Added M8/M16/M32/M64 same-process control/candidate audits to the prior M128 result. Each point used identical random routes/weights, alternating replay order, 6×100 samples per variant, separate untimed 256 MiB L2 clear per graph replay, and four-rank maximum latency.
- **Control/candidate medians (ms):** M8 0.078560/0.077552, M16 0.124256/0.122816, M32 0.197440/0.195120, M64 0.284656/0.281440, M128 0.347104/0.343808.
- **Speedups:** 1.012998x, 1.011725x, 1.011890x, 1.011427x, and 1.009587x; five-shape geometric mean 1.011525x. Candidate batch medians beat control throughout all four new points, as they did at M128.
- **Correctness:** Candidate and control graph outputs were bitwise equal on every rank and shape.
- **Decision:** Accept the mechanism provisionally across all TP4 shapes. Before changing the default/headline, collect a focused W13 NCU confirmation and run TP8 graph plus exact Humming/custom full formal.
- **Artifact:** `results/iter91c_w13_distributed_prep_paired_tp4_m8_m16_m32_m64_cold_600_20260903.log` plus iteration 91b's M128 log.

## Iteration 92 — Distributed W13 preparation NCU capture gate

- Captured the accepted `V4_W13_DISTRIBUTED_PREP=1` TP4 M128/random-route W13 kernel with Nsight Compute `detailed`, selecting the first `route_gemm` launch only.
- The profiled replay preserves the benchmark cold-L2 policy: a separate 256 MiB cache clear immediately precedes the local pipeline and is outside the measured/profiled workload. NCU completed 20 passes and produced a valid report.
- This iteration records collection only; metric extraction and the control/candidate mechanism comparison are intentionally separated into the next audit before changing the default.
- Artifacts: `bench/results/iter92_w13_distributed_prep_m128_coldl2_detailed_ncu.{log,ncu-rep}`.

## Iteration 92a — Distributed W13 NCU first-pass interpretation

- Against the iteration-84 selected-control report, distributed preparation reduced NCU's uncoalesced-global-access warning from 132,864 excessive sectors (11% of 1,224,096) to 80,384 (7% of the same 1,224,096): 52,480 fewer sectors, or 39.50%. This is direct mechanism evidence that the redistributed activation-scale gathers are more coalesced.
- Launch geometry is unchanged at 5,136 CTAs, 128 threads, and 7.32 waves/SM. Registers rise from 54 to 55 per thread; theoretical/achieved occupancy remain effectively unchanged (56.25%/52.97% in the candidate), with no local-memory spills.
- The separate NCU runs report 195.90 us control versus 213.57 us candidate, but clocks were uncontrolled and multi-pass profiling is not the acceptance timing source. The same-process 600-sample cold-L2 paired audit remains the valid latency evidence.
- The candidate `detailed` report omitted Scheduler Statistics and Warp State Statistics (20 passes versus 21 previously), so barrier-stall confirmation is incomplete. Run a focused `SchedulerStats,WarpStateStats,InstructionStats` capture before default selection.
- Artifact: `bench/results/iter92_w13_distributed_prep_m128_ncu_details.log`.

## Iteration 92b — Focused W13 scheduler/warp-state NCU gate

- Re-profiled the same TP4 M128/random-route candidate under the same cold-L2 replay, explicitly requesting `SchedulerStats`, `WarpStateStats`, `InstructionStats`, and `SourceCounters`; NCU completed 13 passes.
- Candidate metrics: No Eligible 21.16%; Eligible Warps/Scheduler 2.65; Warp Cycles/Issued Instruction 10.74; Executed Instructions 83,493,190. The iteration-84 control was 22.74%, 2.38, 10.96 cycles, and 80,218,472 instructions respectively. The requested candidate report did not emit a per-state Stall Barrier row, so the old 3.6-cycle barrier value has no direct candidate counterpart.
- Use the normalized scheduler/warp-state comparison—not cross-run NCU duration—together with the already-completed same-process 600-sample paired latency audit to decide default selection.
- Artifacts: `bench/results/iter92b_w13_distributed_prep_m128_coldl2_sched_ncu.{log,ncu-rep}` and `bench/results/iter92b_w13_distributed_prep_m128_sched_ncu_details.log`.

## Iteration 93 — Select distributed W13 preparation as the default

- **Decision:** Changed the default of `V4_W13_DISTRIBUTED_PREP` from disabled to enabled. This selects the candidate that won every batch across M={8,16,32,64,128} in the same-process 600-sample TP4 cold-L2 paired audit (five-shape geometric speedup 1.01152x). Setting the environment variable to `0` remains an explicit rollback/control.
- **Mechanism gate:** candidate versus iteration-84 control reduced No Eligible from 22.74% to 21.16%, raised Eligible Warps/Scheduler from 2.38 to 2.65, shortened Warp Cycles/Issued Instruction from 10.96 to 10.74, and cut excessive global sectors from 132,864 to 80,384. Registers rose by one and executed instructions by about 4.1%, without occupancy or spill regression.
- **Correctness:** default-enabled TP4 balanced, TP4 max-skew, and TP8-local-shape tests all passed. Route alignment and input quantization were exact; W13 cosine was at least 0.999999997 and W2 cosine at least 0.999997235; all outputs were finite.
- **Cold-L2 status:** this iteration is a correctness/default-selection gate; its timing evidence is the already-recorded iteration-91 paired benchmark, where every replay had an excluded 256 MiB clear.
- Artifact: `results/iter93_w13_distributed_prep_default_correctness_20260903.log`.

## Iteration 94 — Formal default TP4 Humming/custom cold-L2 audit

- **Protocol:** exact MXFP4 Humming and default custom pipelines, both CUDA-Graph captured and both ending in the same SGLang `CustomAllReduceV2`; TP4 GPUs 1–4, random precomputed top-k6 routes, M={8,16,32,64,128}, 10 AB/BA-alternated outer batches x 200 replays = 2,000 cold samples per implementation per M. Every replay has a separate 256 MiB L2 clear excluded from CUDA events; latency is the maximum rank.
- **Correctness:** both implementations passed all-reduce checks at every M. Custom minimum-rank cosine was at least 0.999995575, maximum relative L2 was 0.002974846, and every rank/output was finite.
- **Median latency and speedup (Humming/custom ms, Humming÷custom):** M8 0.090112/0.077920, 1.15647x; M16 0.145792/0.124352, 1.17241x; M32 0.232400/0.210400, 1.10456x; M64 0.344352/0.299440, 1.14999x; M128 0.392048/0.371008, 1.05671x.
- **Five-shape geometric mean:** Humming 0.210386984 ms versus custom 0.186641453 ms = 1.127225x (12.72% faster). A 1.20x result at this Humming level requires custom <=0.175322486 ms, leaving 0.011318967 ms (11.32 us), or 6.06% of current custom latency.
- **Noise audit:** M32–M128 batch medians are visibly dispersed, including custom M128 0.344096–0.466880 ms and Humming M128 0.382144–0.477280 ms. Preserve this run as formal evidence but repeat the identical protocol before attributing the difference from the previous 13.68% headline to code.
- Artifact: `bench/results/iter94_tp4_humming_custom_w13dist_default_coldl2_formal_20260903.log`.

## Iteration 95 — Random-route multicast fused-k6/one-shot-push acceptance audit

- Re-audited the TP4 multicast fused-k6/one-shot-push candidate against stock SGLang CARv2 with the now-default distributed-W13 preparation. Each M used identical weights, inputs, random routes and communicator in one process, 10 order-balanced batches x 200 replays, maximum-rank latency, and a separate excluded 256 MiB L2 clear immediately before every graph replay.
- Candidate/control medians: M8 0.076544/0.077344 ms (control/candidate 1.010451x), M16 0.123712/0.124896 ms (1.009571x), M32 0.196288/0.200224 ms (1.020052x). Three-shape geometric means are 0.122952941/0.124593990 ms, a stable 1.013347x gain.
- Correctness matches stock exactly at every shape (`fused_vs_control_max_abs=0`), with all-reduce/reference checks passing, minimum cosine 0.999995564, maximum relative L2 0.002978603, and finite outputs on every rank.
- Decision: accept as the TP4 M8/M16/M32 default communication tail. Keep M64/M128 and every TP8 shape on stock `CustomAllReduceV2`; do not enable the already-supported M64 multicast specialization.
- Artifact: `bench/results/iter95_multicast_push_current_random_tp4_coldl2_2000_20260903.log`.

## Iteration 96 — Encode accepted TP4 small-M multicast dispatch

- **Change:** Enabled `V4_FUSED_K6_MC_PUSH_AR` by default and narrowed its graph dispatch from M<=64 to the accepted M<=32 boundary. The guard still requires `comm.world_size == 4`; therefore M64/M128 and all TP8 executions fall through to stock SGLang `CustomAllReduceV2`. Environment value `0` remains the explicit rollback.
- Added the distributed-W13 and multicast-tail selections to the exact Humming/custom paired benchmark metadata, and repaired the adjacent metadata indentation.
- **Dispatch/correctness smoke:** TP4 random-route M8 selected `multicast_push`, while M64 selected `stock`/CARv2. Both passed independent NCCL-sum checks; cosine was 0.999995565 and 0.999995583 respectively, relative L2 below 0.002979, and every output finite.
- The 40-sample smoke is not used as performance evidence. Selection relies on iteration 95's 2,000-sample/M cold-L2 audit; this smoke verifies only default wiring and the M32 boundary.
- Artifact: `bench/results/iter96_multicast_default_dispatch_tp4_smoke_20260903.log`.

## Iteration 97 — Selected-default exact Humming formal audit A

- Ran the exact paired protocol after enabling both selected defaults: `w13_distributed_prep=true` and TP4 M<=32 `fused_k6_mc_push_ar=true`. The log explicitly records both flags. Random routes, identical Humming/custom inputs, CUDA Graphs, shared SGLang CARv2 object, 10x200 samples/M, max rank, and excluded 256 MiB clear before every replay are unchanged.
- Median Humming/custom latency and speedup: M8 0.090272/0.077216 ms = 1.169084x; M16 0.146016/0.123104 = 1.186119x; M32 0.230912/0.208544 = 1.107258x; M64 0.337952/0.298816 = 1.130970x; M128 0.409216/0.383664 = 1.066600x.
- Five-shape geometric means: Humming 0.211271796 ms, custom 0.186769695 ms, Humming/custom 1.131189x (13.12% faster). At this baseline, 1.20x requires custom <=0.176059830 ms: another 0.010709865 ms (10.71 us), or 5.73% of current latency.
- Correctness passed for both implementations at every M; custom minimum cosine 0.999995575, maximum relative L2 0.002974846, finite on all ranks.
- This is formal replicate A. M32 and larger retain appreciable batch dispersion, so run an independent identical replicate B before reporting a stabilized post-selection headline.
- Artifact: `bench/results/iter97_tp4_humming_custom_selected_coldl2_formal_a_20260903.log`.

## Iteration 97b — Selected-default exact Humming formal audit B

- Independent identical replicate B reports median Humming/custom latency and speedup: M8 0.090016/0.076944 ms = 1.169890x; M16 0.145984/0.122944 = 1.187402x; M32 0.232864/0.211072 = 1.103244x; M64 0.342176/0.302928 = 1.129562x; M128 0.410784/0.381136 = 1.077789x.
- Replicate-B five-shape geometric means are Humming 0.212186659 ms versus custom 0.187303726 ms = 1.132848x (13.28% faster). All correctness gates pass with the same bounds as replicate A.
- Stabilized A/B geometric aggregation: Humming 0.211728734 ms, custom 0.187036520 ms, speedup 1.132018x (13.20% faster). The two independently measured speedups differ by only 0.001659x.
- At the aggregate Humming level, 1.20x requires custom <=0.176440612 ms. The residual is 0.010595908 ms (5.67% of current custom latency).
- Decision: use the two-run aggregate as the current selected-default headline. The 20% target remains unproven; focus further work on the M32–M128 local W13/W2 path rather than the already-selected small-M communication tail.
- Artifact: `bench/results/iter97b_tp4_humming_custom_selected_coldl2_formal_b_20260903.log`.

## Iteration 98 — Current large-M cold-L2 local-stage budget

- Profiled current selected custom and exact MXFP4 Humming local pipelines at TP4 M32/M64/M128, random routes, CUDA Graph, 200 samples each. A separate 256 MiB clear immediately precedes every pipeline replay and is excluded from stage events. All values below are medians; event instrumentation means use only same-harness comparisons.
- M32 custom/Humming total 200.832/240.032 us (Humming/custom 1.19519x). Custom W13/W2 are 117.024/63.840 us versus Humming 137.184/70.784 us.
- M64 custom/Humming total 275.808/326.720 us (1.18459x). Custom W13/W2 are 164.928/89.232 us versus Humming 192.992/100.224 us.
- M128 custom/Humming total 334.704/390.992 us (1.16817x). Custom W13/W2 are 201.216/108.560 us versus Humming 233.040/121.376 us.
- W13+W2 consume 90.06%, 92.15%, and 92.55% of custom local time at M32/M64/M128. Route/input quantization, activation quantization and local k6 reduction jointly leave only about 20–25 us, so the residual 1.20x end-to-end gap cannot be closed by tail-only work. Prioritize core GEMM issue/dataflow at M32–M128.
- Artifact: `bench/results/iter98_current_stage_budget_m32_m64_m128_coldl2_20260903.log`.

## Iteration 99 — W13 merged-WGMMA-group first correctness launch blocked

- Added an opt-in `V4_W13_MERGED_WGMMA_GROUP=1` candidate. Within each K128 activation-scale group it keeps the four K32 RS-WGMMA operations in one commit group and performs one final wait, while retaining the per-step operand fence/arrival. Default remains disabled.
- The three requested correctness processes executed route preparation and the kernel path but the harness then raised `AttributeError` while printing the new flag. Investigation found an accidentally uploaded stale `bench/v4_flash_tp_wgmma.py` shadowing the repository-root module because Python places the script directory first on `sys.path`.
- This is a test-launch/import failure, not eligible correctness or performance evidence. Preserve the candidate and failure log; remove only the newly introduced stale shadow module, then rerun all three gates with an unambiguous root-module import.
- Artifact: `results/iter99_w13_merged_wgmma_group_correctness_20260903.log`.

## Iteration 99b — W13 merged-WGMMA-group correctness gate

- Re-ran via `python -m bench.test_v4_flash_tp_wgmma` after removing the accidental shadow module, guaranteeing import of the repository-root candidate.
- TP4 balanced auto-split4, TP4 max-skew forced split2, and TP8-local-shape auto-split4 all pass. Route alignment/input quantization remain exact; W13 cosine is at least 0.999999997, W2 cosine at least 0.999997235, and all outputs are finite.
- Numerical values reproduce the selected control to printed precision, including the split2 skew path. This confirms that four same-accumulator K32 WGMMA operations may share one commit group under the retained operand-fence protocol.
- Decision: correctness gate passed; keep opt-in and proceed to same-process cold-L2 paired timing before any default change.
- Artifact: `results/iter99b_w13_merged_wgmma_group_correctness_20260903.log`.

## Iteration 99c — W13 merged-WGMMA-group paired screen

- Compared current default (`V4_W13_MERGED_WGMMA_GROUP=0`) with the candidate in one TP4 process at M128/random routes, identical weights/routes/communicator, six x 100 samples, per-replay A/B then B/A alternation, maximum-rank latency, and an excluded 256 MiB clear before every replay.
- Control/candidate medians are 0.357791990/0.357743993 ms, control/candidate 1.000134x: only 0.048 us or 0.013%. Batch medians are effectively paired ties and direction is mixed. Candidate output is bitwise identical on all ranks.
- Decision: reject as noise-sized and keep the default disabled. Do not extend the same change to W2: the expected wait-overlap benefit is absent even at the M128 W13-dominated point, indicating the dependency/issue chain already hides or serializes this wait structure.
- Artifact: `bench/results/iter99c_w13_merged_wgmma_group_tp4_m128_paired_cold600_20260903.log`.

## Iteration 100 — Normalized 13-entry shared-LUT correctness gate

- Added opt-in `V4_NORMALIZED_SHARED_LUT=1`, valid only with normalized weight scales. It initializes a 13-entry (104-byte) per-CTA shared LUT for normalized exponent offsets 0..12 and replaces the two independent affine LUT-synthesis IMAD chains per weight word with shared `uint2` loads. Default remains disabled.
- TP4 balanced auto-split4, TP4 max-skew forced split2, and TP8-local-shape auto-split4 all pass. Route/input quantization is exact; W13 cosine is at least 0.999999997, W2 cosine at least 0.999997235, and all outputs are finite.
- Numerical values match the current default to printed precision. Proceed to resource inspection and same-process cold-L2 timing; reject if shared-load latency or bank pressure outweighs reduced ALU work.
- Artifact: `results/iter100_normalized_shared_lut_correctness_20260903.log`.

### Iteration 100b — reject normalized shared LUT on paired cold-L2 timing

- Compared `V4_NORMALIZED_SHARED_LUT=0/1` in one TP4 process at M=128 with random routing, exact graph-output comparison, ABBA ordering, 6 outer batches × 100 replays per variant.
- Cache policy remained the required cold-L2 protocol: a separate 256 MiB clear immediately before every graph replay, outside CUDA-event timing.
- Control median: **0.351600 ms**; candidate median: **0.370528 ms**; control/candidate: **0.948916x**. Equivalently, the candidate is about **5.38% slower**.
- All six candidate batch medians were slower than their paired controls; outputs were bitwise identical on all ranks. This is a decisive regression rather than timing noise.
- Conclusion: reject the shared normalized LUT and keep `V4_NORMALIZED_SHARED_LUT=0` by default. Replacing independent integer synthesis with shared-memory lookups adds more latency/traffic than it removes from the ALU path.

## Iteration 101 — W2 distributed preparation correctness gate

- Hypothesis: W2 still concentrates all eight noncontiguous activation-scale gathers in warp 0. The selected W13 distributed-preparation path reduced NCU's excessive global sectors by 39.5%, while the prior W2 NCU audit specifically attributed 66,432 excessive sectors to the analogous route-scale gather. Distribute two scale loads to each warp and move the TMA issuer to warp 1 without changing math or bytes.
- Added opt-in `V4_W2_DISTRIBUTED_PREP=1`, default off. The generic route GEMM now chooses distributed preparation independently for W13 and W2; accepted W13 behavior is unchanged.
- Correctness passes TP4 balanced/auto split4, TP4 maximal-skew/forced split2, and the TP8 local intermediate=256 shape. Route alignment and FP8 input quantization are exact; W13 cosine is at least 0.999999997, W2 cosine is at least 0.999997240, and all outputs are finite.
- Next gate is an in-process paired TP4 M128 cold-L2 graph comparison. Select only if the sector/coalescing mechanism produces a repeatable end-to-end win.
- Artifact: `results/iter101_w2_distributed_prep_correctness_20260903.log`.

## Iteration 101b — W2 distributed-preparation M128 paired screen

- TP4 M128 random routing, identical weights/input/communicator, one-process control/candidate graphs, ABBA per-sample ordering, and 6×100 samples per variant. A separate 256 MiB clear immediately preceded every replay and was excluded from events.
- Control median: **0.357984 ms**; candidate median: **0.357424 ms**; control/candidate: **1.001567x** (0.560 us, 0.157%). Complete graph outputs are bitwise identical on all ranks.
- All six candidate batch medians beat their paired controls by 0.192–1.024 us. This is a small but directionally consistent signal, not enough to select from one shape alone.
- Keep the flag opt-in while screening M8/M16/M32/M64 under the same paired cold-L2 protocol. A broad win would justify a five-shape long-window acceptance audit; a shape-specific reversal would bound dispatch instead.
- Artifact: `results/iter101b_w2_distributed_prep_tp4_m128_paired_cold600_20260903.log`.

## Iteration 101c — W2 distributed-preparation four-shape paired screen

- Extended the same-process TP4 random-route cold-L2 comparison to M8/M16/M32/M64, with 4×100 samples per variant and per-sample ABBA ordering. Every replay used the excluded 256 MiB cache clear.
- Control/candidate medians (ms) and speedups were: M8 `0.076864/0.076416 = 1.005863x`; M16 `0.121664/0.121088 = 1.004757x`; M32 `0.191616/0.190688 = 1.004867x`; M64 `0.280128/0.279136 = 1.003554x`. Outputs were bitwise identical on all ranks at every shape.
- Combined with iteration 101b M128 (`1.001567x`), the five-shape geometric-mean improvement is approximately **1.00412x**. All 22 paired batch medians across the five shapes favor the candidate.
- The effect is small but broad and unusually consistent, matching the expected benefit from distributing W2's route-scale gathers. Keep opt-in until a longer five-shape exact-Humming/custom audit confirms the selected-score impact; then consider making W2 distributed preparation the default.
- Artifact: `results/iter101c_w2_distributed_prep_tp4_m8_m16_m32_m64_paired_cold400_20260903.log`.

## Iteration 101d — exact-Humming five-shape formal audit with W2 distributed preparation

- Ran the exact MXFP4 Humming/custom TP4 CUDA graphs with `V4_W2_DISTRIBUTED_PREP=1`, random routes, M={8,16,32,64,128}, 10 balanced batch-order windows × 200 samples per implementation/M. Every graph replay was preceded by a separate excluded 256 MiB L2 clear; both graphs used the same SGLang `CustomAllReduceV2` instance.
- Humming/custom medians (ms) and speedups: M8 `0.090048/0.076480 = 1.177406x`; M16 `0.145824/0.122144 = 1.193870x`; M32 `0.232320/0.209808 = 1.107298x`; M64 `0.339008/0.301328 = 1.125046x`; M128 `0.407936/0.383168 = 1.064640x`.
- Geometric means are Humming **0.211367559 ms** and custom **0.186609677 ms**, for **1.132672x (13.27%)**. Both implementations pass their independent reference and all-reduce checks at every shape.
- Against this exact window, a 1.20x result requires custom <= `0.176139633 ms`; the remaining geometric-mean reduction is `0.010470044 ms`, or **5.61%** of current custom latency.
- Cross-window headline drift masks most of the candidate's approximately 0.41% same-process control win, so selection rests on iterations 101b/c's 22/22 paired-batch direction, while this run establishes end-to-end correctness and the new absolute score. Make W2 distributed preparation the default, re-gate the no-environment TP4/TP8 paths, then resume structural GEMM work.
- Artifact: `results/iter101d_w2_distributed_prep_exact_humming_tp4_formal_cold2000_20260903.log`.

## Iteration 102 — select W2 distributed preparation as the default

- Changed the no-environment default of `V4_W2_DISTRIBUTED_PREP` from 0 to 1 after the five-shape paired screen favored it in all 22 batch comparisons and the exact-Humming formal graph passed.
- Re-ran the default path without the flag: TP4 balanced auto-split4, TP4 maximal-skew forced split2, and TP8-local intermediate=256 all pass. Route/input preparation is exact; W13 cosine is at least 0.999999997, W2 cosine is at least 0.999997240, and every output is finite.
- TP8 remains source- and shape-valid while the available non-shared GPUs constrain distributed performance work to TP4. Resume optimization from this selected default; the remaining 1.20x gap is about 10.47 us geometric mean (5.61% of custom).
- Artifact: `results/iter102_w2_distributed_prep_default_correctness_20260903.log`.

## Iteration 103 — CTA-scope bulk-TMA correctness gate

- Hypothesis: selected kernels copy only into their own CTA shared memory but encode linear bulk loads with the remote-capable `shared::cluster` destination. PTX defines `shared::cta` as the strictly non-remote global-to-shared form; test whether narrower scope lowers address/synchronization overhead without changing the transfer.
- Added opt-in `V4_TMA_CTA_SCOPE=1`, default off, and routed selected linear bulk weight/scale copies through one helper. Tensor-map fallback and the already-rejected paired-W13 prototype remain unchanged. Packed layout, bytes, barriers, math, grid, and output are identical.
- Correctness passes TP4 balanced auto-split4, TP4 maximal-skew forced split2, and TP8-local intermediate=256. Route/quant preparation is exact; W13 cosine is at least 0.999999997, W2 cosine is at least 0.999997240, and all outputs are finite.
- Next compare the two scope forms in one process under per-replay cold L2 and inspect cubin/SASS. Keep disabled unless timing shows a repeatable benefit.
- Artifact: `results/iter103_tma_cta_scope_correctness_20260903.log`.

## Iteration 103b — CTA-scope bulk-TMA paired rejection

- Same-process TP4 M128 random-route comparison, 6×100 samples per scope form, exact graph-output check, per-sample ABBA ordering, and an excluded 256 MiB L2 clear immediately before each replay.
- Cluster-scope control median: **0.355024 ms**; CTA-scope candidate: **0.355456 ms**; control/candidate: **0.998785x**. The candidate is 0.432 us (0.122%) slower.
- Batch direction is mixed (three candidate wins and three losses) and all differences are sub-microsecond. Narrowing the PTX shared-memory scope supplies no measurable end-to-end benefit on H20.
- Reject `V4_TMA_CTA_SCOPE=1` and keep the default off. A cubin/SASS comparison may explain whether ptxas maps both forms to the same instruction, but no broader timing sweep is justified.
- Artifact: `results/iter103b_tma_cta_scope_tp4_m128_paired_cold600_20260903.log`.

## Iteration 103c — CTA/cluster scope lowers to identical H20 SASS

- Disassembled the exact TP4 W2 `route_gemm<K=512,N=4096,SplitK=1>` specialization from the control and candidate JIT objects with `cuobjdump`.
- After removing only the source-path identifier, the complete function dumps have an empty byte-level diff. Both PTX scope spellings lower to the same `UBLKCP.S.G` instructions and identical surrounding SASS on sm_90a.
- This explains iteration 103b's mixed noise-scale timing and closes the scope experiment conclusively. Keep `V4_TMA_CTA_SCOPE=0`; no further shape sweep is warranted.
- Artifact: `results/iter103c_tma_scope_sass_diff_20260903.log`.

## Iteration 104 — direct activation `evict_last` does not assemble on sm90a

- Added opt-in `V4_ACTIVATION_EVICT_LAST=1` to mark the repeatedly reused FP8 activation rows, activation scales, and expert-global scales as L2 eviction-last while leaving cold packed-weight TMA unchanged.
- The JIT reaches ptxas but fails before any kernel runs: sm90a reports that an `ld` carrying the direct `.L2::evict_last` modifier requires `.v8.b32` or `.v4.b64`. The route kernel's natural per-thread accesses are 8-byte activation vectors and 4-byte scales, so the direct priority qualifier is illegal at this width.
- This is a compile failure, not a correctness or performance result. Preserve the failure log, then repair with the general `.L2::cache_hint` operand backed by `createpolicy.fractional.L2::evict_last`, which PTX permits independently of the direct wide-load modifier.
- Artifact: `results/iter104_activation_evict_last_correctness_20260903.log`.

## Iteration 104b — legal activation L2 cache-policy correctness gate

- Repaired iteration 104 by creating one per-thread `createpolicy.fractional.L2::evict_last` value and using the general `ld.global.L2::cache_hint` operand for 8-byte FP8 activation loads and 4-byte activation/expert scales. This form is legal at the natural access widths.
- The candidate compiles and passes TP4 balanced auto-split4, TP4 maximal-skew forced split2, and TP8-local intermediate=256. Route/input preparation is exact; W13 cosine is at least 0.999999997, W2 cosine is at least 0.999997240, and all outputs are finite.
- The 64-bit policy may increase register pressure, so the next gate must inspect cubin resources as well as same-process paired cold-L2 M128 timing. Reject if occupancy cost outweighs reuse.
- Artifact: `results/iter104b_activation_cache_policy_correctness_20260903.log`.

## Iteration 104c — activation cache-policy paired rejection

- Cubin resource inspection initially looked favorable: TP4 W2 fell from 55 to 54 registers/thread and W13 split2 from 55 to 48, with no local spill. The inline PTX shortened compiler-managed load/address live ranges despite retaining a 64-bit policy.
- Same-process TP4 M128 random-route timing used 8×100 samples per variant, per-sample ABBA ordering, exact graph-output equality, and a separate excluded 256 MiB L2 clear before every replay.
- Control median: **0.353680 ms**; cache-policy candidate: **0.358192 ms**; control/candidate: **0.987403x**. The candidate is 4.512 us (1.276%) slower, and all eight candidate batch medians lose.
- Conclusion: the `createpolicy`/cache-hint execution and/or forced eviction priority costs more than any activation reuse benefit. Reject `V4_ACTIVATION_EVICT_LAST=1`, retain default off, and do not extend it to the other M values.
- Artifact: `results/iter104c_activation_cache_policy_tp4_m128_paired_cold800_20260903.log`.

## Iteration 105 — streaming-weight evict-first correctness gate

- Added opt-in `V4_WEIGHT_EVICT_FIRST=1`, default off. Each selected linear bulk-TMA issue creates a short-lived fractional L2 evict-first policy and passes it through `.L2::cache_hint`; activation/metadata loads, packed layout, transfer bytes, barriers, math, and grid are unchanged.
- The policy matches the actual cold workload: each active expert's packed weight record is streamed once, while much smaller activation and route data are reused by many output tiles/splits.
- Correctness passes TP4 balanced auto-split4, TP4 maximal-skew forced split2, and TP8-local intermediate=256. Route/input preparation is exact; W13 cosine is at least 0.999999997, W2 cosine is at least 0.999997240, and all outputs are finite.
- Next run a long in-process paired TP4 M128 cold-L2 gate. Keep disabled unless the cache hint beats the extra policy instruction repeatably.
- Artifact: `results/iter105_weight_evict_first_correctness_20260903.log`.

## Iteration 105b — streaming-weight evict-first M128 paired win

- Same-process TP4 M128 random-route comparison, identical data/communicator, exact graph-output equality, per-sample ABBA ordering, and 8×100 samples per variant. Every replay used a separate excluded 256 MiB L2 clear.
- Control median: **0.359440 ms**; evict-first candidate: **0.356224 ms**; control/candidate: **1.009028x**. The candidate saves 3.216 us (0.894%).
- All eight candidate batch medians beat their paired controls, by approximately 2.75–5.28 us. The consistent multi-microsecond separation is materially stronger than the rejected scope/cache-policy noise.
- Keep opt-in while screening M8/M16/M32/M64. If every shape remains non-regressive, proceed to a long five-shape exact-Humming audit and select it as the default.
- Artifact: `results/iter105b_weight_evict_first_tp4_m128_paired_cold800_20260903.log`.

## Iteration 105c — streaming-weight evict-first wins all TP4 shapes

- Extended the same-process random-route cold-L2 comparison to M8/M16/M32/M64 with 4×100 samples per variant, exact output checks, per-sample ABBA ordering, and a separate excluded 256 MiB clear before every replay.
- Control/candidate medians (ms) and speedups: M8 `0.076000/0.075136 = 1.011499x`; M16 `0.120992/0.119424 = 1.013130x`; M32 `0.192160/0.190320 = 1.009668x`; M64 `0.281312/0.279200 = 1.007564x`.
- Combined with iteration 105b M128 (`1.009028x`), the five-shape geometric-mean improvement is approximately **1.01018x**. All 28 paired batch medians across the five shapes favor evict-first, and all graph outputs are bitwise identical.
- This is a broad ~1% custom-pipeline improvement with a mechanism matching cold streamed weights. Proceed to the exact MXFP4 Humming/custom 10×200 five-shape formal audit before changing the default.
- Artifact: `results/iter105c_weight_evict_first_tp4_m8_m16_m32_m64_paired_cold400_20260903.log`.

## Iteration 105d — exact-Humming formal audit accepts streaming-weight evict-first

- Ran exact MXFP4 Humming/custom TP4 CUDA graphs with `V4_WEIGHT_EVICT_FIRST=1`, random routes, M={8,16,32,64,128}, 10 balanced batch-order windows × 200 samples per implementation/M. A separate excluded 256 MiB L2 clear preceded every graph replay; both paths shared one SGLang `CustomAllReduceV2` communicator.
- Humming/custom medians (ms) and speedups: M8 `0.090080/0.075072 = 1.199915x`; M16 `0.145728/0.120480 = 1.209562x`; M32 `0.231632/0.207296 = 1.117397x`; M64 `0.334704/0.300832 = 1.112594x`; M128 `0.408144/0.379744 = 1.074787x`.
- Geometric means are Humming **0.210711799 ms** and custom **0.184569426 ms**, for **1.141640x (14.16%)**. Every point passes independent reference and all-reduce correctness.
- Relative to iteration 101d, custom geomean improves from 0.186609677 to 0.184569426 ms (1.105%); simultaneous Humming drift is -0.31%, so headline speedup rises by 0.90%. Same-process iterations 105b/c isolate the candidate itself at approximately 1.018%.
- At this window, 1.20x requires custom <= `0.175593166 ms`; the residual is `0.008976260 ms`, or **4.86%** of current custom latency. Accept streaming-weight evict-first and make it the default, then re-gate no-environment TP4/TP8 paths.
- Artifact: `results/iter105d_weight_evict_first_exact_humming_tp4_formal_cold2000_20260903.log`.

## Iteration 106 — select streaming-weight evict-first as default

- Changed the no-environment default of `V4_WEIGHT_EVICT_FIRST` from 0 to 1 after all 28 paired batches favored it and the exact-Humming formal graph reached 1.141640x.
- Re-ran the selected default without the flag: TP4 balanced auto-split4, TP4 maximal-skew forced split2, and TP8-local intermediate=256 all pass. Route/input preparation is exact; W13 cosine is at least 0.999999997, W2 cosine is at least 0.999997240, and all outputs are finite.
- Current formal TP4 cold-L2 score is Humming/custom `0.210711799/0.184569426 ms = 1.141640x`; the remaining 1.20x gap is approximately 8.98 us geometric mean, or 4.86% of custom.
- Artifact: `results/iter106_weight_evict_first_default_correctness_20260903.log`.

## Iteration 107 — hoisted streaming-weight policy correctness gate

- SASS for the selected evict-first path showed roughly six uniform policy-construction instructions immediately before each bulk TMA. Added opt-in `V4_WEIGHT_POLICY_HOIST=1`, default off, to construct the same eviction policy once per CTA and reuse it for every weight/scale transfer.
- Cache semantics, transfer bytes, layout, synchronization, math, and grid are unchanged. The current per-issue construction remains the control.
- Correctness passes TP4 balanced auto-split4, TP4 maximal-skew forced split2, and TP8-local intermediate=256. Route/input preparation is exact; W13 cosine is at least 0.999999997, W2 cosine is at least 0.999997240, and all outputs are finite.
- Next inspect SASS/resource usage to verify the intended hoist and ensure no occupancy loss, then run a long paired TP4 M128 cold-L2 gate.
- Artifact: `results/iter107_weight_policy_hoist_correctness_20260903.log`.

## Iteration 107b — hoisted weight policy is a paired regression

- Cubin inspection verified the intended static simplification: TP4 W2 SASS shrank by eight instruction lines and registers fell 55→54; W13 split2 registers fell 55→48, with no spill.
- Same-process TP4 M128 random-route timing used 8×100 samples per variant, exact graph-output equality, per-sample ABBA ordering, and a separate excluded 256 MiB L2 clear before every replay.
- Per-issue control median: **0.354976 ms**; hoisted-policy candidate: **0.359552 ms**; control/candidate: **0.987273x**. The candidate is 4.576 us (1.289%) slower, and all eight candidate batch medians lose.
- Static instruction/register reductions did not shorten the dynamic critical path; retaining the policy across the kernel likely adds dependency/uniform-transfer pressure to each TMA issue. Reject `V4_WEIGHT_POLICY_HOIST=1` and retain per-issue policy creation as the selected default.
- Artifact: `results/iter107b_weight_policy_hoist_tp4_m128_paired_cold800_20260903.log`.

## Iteration 108 — read back H20 evict-first policy encoding

- Added a standalone sm90a CUDA probe that executes `createpolicy.fractional.L2::evict_first.b64 ...,1.0` and copies the opaque 64-bit result to the host.
- On the benchmark H20/CUDA toolchain the exact value is **`0x12f0000000000000`**.
- This enables a bounded candidate that materializes the same constant immediately beside each TMA use, preserving the selected short dependency lifetime while replacing the multi-instruction policy synthesis. The value is architecture/toolchain-specific, so the candidate must remain guarded for sm90a and be proven bit-identical by SASS/runtime gates before selection.
- Artifact: `bench/probe_l2_policy.cu` and `results/iter108_l2_policy_encoding_probe_20260903.log`.

## Iteration 109 — short-lived constant weight-policy correctness gate

- Added opt-in `V4_WEIGHT_POLICY_CONSTANT=1`, mutually exclusive with the rejected hoisted-policy path.  At each selected bulk-TMA issue site it materializes the probed H20/sm90a value `0x12f0000000000000` immediately before `cp.async.bulk ... .L2::cache_hint`; transfer bytes, packed layout, barriers, math, scheduling, and communication are unchanged.
- The value is deliberately guarded by an opt-in flag and is only a target-specific experiment for this fixed H20/sm90a toolchain.  The selected per-issue `createpolicy` path remains the default/control.
- Correctness passes TP4 balanced auto-split4, TP4 maximal-skew forced split2, and TP8-local intermediate=256.  Route/input preparation is exact; W13 cosine is at least `0.999999997`, W2 cosine is at least `0.999997240`, and all outputs are finite.
- Next inspect cubin/SASS to confirm the intended materialization, then run a long same-process TP4 M128 cold-L2 paired gate before considering other shapes.
- Artifact: `results/iter109_weight_policy_constant_correctness_20260903.log`.

## Iteration 109b — M128 paired launch rejected by torchrun parser

- The intended 8x100 TP4 M128 cold-L2 paired command exited before worker or GPU launch because this environment's `torchrun` parser treated the script argument `--m 128` as an ambiguous launcher option (`--max-restarts`, `--monitor-interval`, `--module`, and `--master-*`).
- No timing samples were produced and this says nothing about candidate performance.  Re-run with the unambiguous `--m=128` script-argument form; keep every benchmark and cache-control setting unchanged.
- Artifact: `results/iter109b_weight_policy_constant_tp4_m128_paired_cold800_20260903.log`.

## Iteration 109c — constant weight policy narrowly wins M128

- Cubin inspection confirms the intended lowering.  The per-issue control's roughly six uniform policy-synthesis instructions become two adjacent `UMOV` instructions (`UR20=0`, `UR21=0x12f00000`) before `UBLKCP.S.G`; total cubin SASS output falls from 43,034 to 42,874 lines.  Relevant W13/W2 registers, stack, local memory, and shared memory are unchanged.
- Same-process TP4 M128 random-route timing used 8x100 samples per variant, exact graph-output equality, per-sample ABBA ordering, and a separate excluded 256 MiB L2 clear before every replay.
- Per-issue-create control median: **0.357456 ms**; short-lived-constant candidate: **0.357248 ms**; control/candidate: **1.000582x**.  The candidate saves 0.208 us, and seven of eight candidate batch medians beat their paired controls.
- The direction is encouraging but far below one percent.  Keep opt-in and screen M8/M16/M32/M64 under the same paired cold-L2 protocol before selecting or rejecting it.
- Artifact: `results/iter109c_weight_policy_constant_tp4_m128_paired_cold800_20260903.log`.

## Iteration 109d — constant weight policy rejected across TP4 shapes

- Extended the same-process random-route cold-L2 comparison to M8/M16/M32/M64 with 4x100 samples per variant, exact output checks, per-sample ABBA ordering, and a separate excluded 256 MiB clear before every replay.
- Control/candidate medians (ms) and speedups: M8 `0.075072/0.075424 = 0.995333x`; M16 `0.119264/0.119488 = 0.998125x`; M32 `0.192256/0.192192 = 1.000333x`; M64 `0.279616/0.279664 = 0.999828x`.
- Including iteration 109c M128 (`1.000582x`), the five-shape geometric mean is approximately **0.99884x**, a 0.116% regression.  Only 12 of the 24 paired batch medians across all five shapes favor the candidate; M8 and M16 lose every batch.
- Decision: reject `V4_WEIGHT_POLICY_CONSTANT=1` and retain the selected short-lived per-issue `createpolicy` path.  Static instruction reduction alone does not produce a repeatable end-to-end win, and the opaque target-specific encoding is not justified by these results.
- Artifact: `results/iter109d_weight_policy_constant_tp4_m8_m16_m32_m64_paired_cold400_20260903.log`.

## Iteration 110 — selected-default TP4 local stage budget

- Re-profiled the current selected default without experimental flags at M={8,16,32,64,128}, random routing, 200 CUDA-Graph samples per point.  Every replay has a separate excluded 256 MiB clear; the H20 reports 60 MiB L2.  External stage-event overhead means these values are diagnostic and are not headline end-to-end timings.
- Total/W13/W2 median latencies (us): M8 `86.080/43.456/25.088`; M16 `131.520/72.640/40.576`; M32 `197.504/115.776/62.848`; M64 `271.456/162.640/87.872`; M128 `327.808/197.632/106.880`.
- W13+W2 consume 79.6%, 86.1%, 90.4%, 92.3%, and 92.9% of local time respectively.  At M32/M64/M128, W13 alone is 58.6-60.3% and W2 is 31.8-32.6%; route quantization, fused activation/requantization, and local k6 reduction together are only 7.1-9.6%.
- Conclusion: the remaining 1.20x gap cannot be closed by epilogue or route-preparation micro-optimizations at medium/large M.  Prioritize a structural W13 improvement, preserving the selected W2 and communication paths.
- Artifact: `results/iter110_selected_custom_tp4_stage_budget_cold1000_20260903.log`.

## Iteration 111 — dual-WG split-K W13 first build is not importable

- Added opt-in `V4_W13_DUAL_WG_SPLIT=1`: one 256-thread CTA contains two independent N128 WGMMA warpgroups with separate packed-weight/barrier state while sharing one 8xK128 activation and activation-scale stage.  The existing split-K factor, per-issue weight cache policy, output workspace, activation epilogue, W2, and communication remain unchanged; TP4 and TP8 output widths both divide into N256 task pairs.
- The first JIT attempt reached extension import but failed before any GPU launch: the shared object did not expose the `PyInit_*` symbol matching the new, very long extension name.  Consequently no correctness or performance conclusion is available from this attempt.
- Preserve the failed source/log, inspect the exported symbol, then repair only the extension-key naming before repeating the same three numerical gates.
- Artifact: `results/iter111_w13_dual_wg_split_correctness_20260903.log`.

### Iteration 111a — deterministic import failure and symbol audit

- A cache-hit standalone import reproduces the same missing-`PyInit_*` error, ruling out an incomplete first-build race.
- `nm -D` shows that the `.so` does contain a globally exported `PyInit_v4_flash_tp_wgmma_..._v111dwg` symbol with the configured flag suffix.  The failure is therefore in resolving the oversized module short name rather than CUDA compilation or kernel launch.
- Repair by shortening only the Python extension name; keep every compile-time field in the CUDA flags and retain the `dwg` discriminator plus iteration suffix so control/candidate caches remain distinct.

### Iteration 111b — compact extension key runs and exposes activation reuse race

- Replaced the oversized readable extension name with a 20-hex SHA1 of the complete configuration plus a readable iteration suffix.  Compile flags and runtime metadata still retain every individual switch, and control/candidate modules remain distinct.
- The compact module imports and runs.  TP4 balanced split4 and TP8-local balanced split4 exactly reproduce the selected W13/activation/W2 errors.  TP4 maximal-skew forced split2 passes the old broad threshold but does **not** reproduce the control: W13 cosine drops from about `0.999999997` to `0.999999330`, and rel-L2 rises from about `7.76e-5` to `1.158e-3`.
- This is consistent with WG0 overwriting the single shared activation tile for the next K128 iteration before WG1 has completed its prior WGMMA read.  Do not time or accept this version.  Add a one-lane consumer-complete mbarrier plus a WG0-only named-barrier gate before buffer reuse, avoiding a second full-CTA barrier.
- Artifact: `results/iter111b_w13_dual_wg_split_correctness_20260903.log`.

### Iteration 111c — asymmetric activation-buffer reuse handshake is correct

- Added one shared mbarrier for the dual-WG activation buffer.  After its final WGMMA wait, WG1 leader publishes buffer-empty; before the next K128 overwrite, WG0 leader waits and releases its own 128 lanes through a WG0-only named barrier.  WG1 can proceed independently until the existing pre-math full-CTA barrier, avoiding a second full-CTA synchronization per tile.
- TP4 balanced auto-split4, TP4 maximal-skew forced split2, and TP8-local intermediate=256 now exactly reproduce the selected error metrics.  In particular the split2 W13 cosine/rel-L2 recover from `0.999999330/1.158e-3` to `0.999999997/7.76e-5`; route/input quantization is exact, W2 cosine is at least `0.999997240`, and all outputs are finite.
- Correctness and TP8 runnability gates pass.  Keep opt-in, inspect cubin occupancy/resources, then run a same-process TP4 M128 cold-L2 paired performance gate.
- Artifact: `results/iter111c_w13_dual_wg_split_handshake_correctness_20260903.log`.

### Iteration 111d — dual-WG split-K W13 is a large M128 regression

- Cubin resources show 56 registers/thread and no local spill for dual-WG split2/split4.  With about 41 KiB dynamic shared memory, the 256-thread CTA is register-limited to four resident CTAs/SM, or 32 warps.  The selected 55-register 128-thread kernel reaches nine CTAs/SM, or 36 warps.
- Same-process TP4 M128 random-route timing used 8x100 samples per variant, exact graph-output equality, per-sample ABBA ordering, and a separate excluded 256 MiB L2 clear before every replay.
- Selected control median: **0.344176 ms**; dual-WG candidate: **0.383552 ms**; control/candidate: **0.897339x**.  The candidate is 39.376 us (11.44%) slower, and all eight candidate batch medians lose.
- Decision: reject `V4_W13_DUAL_WG_SPLIT=1` and retain default off.  Halving repeated activation staging and CTA setup does not compensate for lower resident-warp concurrency plus the cross-WG reuse handshake.  The loss is decisive enough not to spend more samples on smaller M.
- Artifact: `results/iter111d_w13_dual_wg_split_tp4_m128_paired_cold800_20260903.log`.

## Iteration 112 — isolate W2 evict-first correctness gate

- Added opt-in `V4_W2_NO_WEIGHT_EVICT_FIRST=1`.  W13 retains the selected per-issue evict-first policy; only W2 falls back to ordinary bulk TMA.  Weight bytes/layout, barriers, dequantization, math, output, and communication are unchanged.  The experiment tests whether policy construction costs more than its cache benefit on W2's four K128 tiles.
- Correctness passes TP4 balanced auto-split4, TP4 maximal-skew forced split2, and TP8-local intermediate=256 with error metrics identical to the selected control.  Route/input quantization is exact, W13 cosine is at least `0.999999997`, W2 cosine is at least `0.999997240`, and all outputs are finite.
- Keep opt-in and run a long same-process TP4 M128 cold-L2 paired gate before screening other shapes.
- Artifact: `results/iter112_w2_no_evict_first_correctness_20260903.log`.

### Iteration 112b — W2 no-evict candidate is a paired regression

- Same-process TP4 M128 random-route timing used 8x100 samples per variant, exact graph-output equality, per-sample ABBA ordering, and a separate excluded 256 MiB L2 clear before every replay.
- Selected W2-evict-first control median: **0.357984 ms**; W2-no-evict candidate: **0.359136 ms**; control/candidate: **0.996792x**.  Removing the hint costs 1.152 us (0.322%), and all eight candidate batch medians lose.
- Decision: reject `V4_W2_NO_WEIGHT_EVICT_FIRST=1` and retain W2 evict-first.  Even across only four K128 tiles, the protection of reused activation/metadata cache lines is worth more than per-TMA policy creation.  The all-batch loss is sufficient to stop before other M values.
- Artifact: `results/iter112b_w2_no_evict_first_tp4_m128_paired_cold800_20260903.log`.

## Iteration 113 — exact-Humming TP4 local stage comparison

- Profiled exact MXFP4 Humming at M={8,16,32,64,128}, random routing, 200 CUDA-Graph samples per point, with the same stage harness as iteration 110.  Every replay has a separate excluded 256 MiB clear.  External event nodes add instrumentation overhead, so these results locate work but do not replace end-to-end paired timings.
- Humming total/W13/W2 medians (us): M8 `108.272/52.336/26.304`; M16 `162.176/86.336/44.576`; M32 `240.256/137.728/70.848`; M64 `326.784/192.832/100.256`; M128 `389.760/231.872/120.912`.
- Against iteration 110's selected custom path, local-total Humming/custom ratios are `1.2578x`, `1.2331x`, `1.2165x`, `1.2038x`, and `1.1890x`.  Custom W13 wins every point by 17.3-20.4%; custom W2 wins every point by 4.8-14.1%.
- Finding: neither MXFP4 Humming GEMM is currently ahead.  The exact end-to-end score is much lower because the common communication/graph tail dilutes local-kernel gains; at M128, however, the local ratio itself is still just below 1.20x.  Future work must both reduce the TP communication tail and retain a structural W13 path, rather than optimizing against a false premise that Humming's individual GEMMs are faster.
- Artifact: `results/iter113_exact_humming_tp4_stage_budget_cold1000_20260903.log`.

## Iteration 114 — long M64 multicast-push boundary remains unstable

- Added a runtime `V4_FUSED_K6_MC_PUSH_MAX_M` dispatch bound, defaulting to the selected 32 and accepting explicit 64 only for controlled reproduction.  The already-validated M64 fused k6 + multicast one-shot kernel, symmetric workspace, and CARv2 counter protocol are unchanged.
- Same-process TP4 M64 random-route timing used 10 balanced AB/BA batches x 200 samples per path, identical data/communicator, exact graph-output equality, independent reference/all-reduce checks, and a separate excluded 256 MiB L2 clear before every replay.
- Stock median: **0.284864 ms**; multicast-fused median: **0.283328 ms**; stock/fused: **1.005421x**.  Despite the pooled 1.536 us lead, only five of ten candidate batch medians beat their paired controls, and both paths show a broad bimodal distribution.
- Decision: the earlier M64 lead still does not survive a direction-consistent long audit.  Keep the production bound at M<=32 and stock CARv2 at M64; retain the explicit bound only as a reproducibility knob.  Correctness is bit-exact versus stock (`max_abs=0`).
- Artifact: `results/iter114_m64_multicast_push_long_ab_cold2000_20260903.log`.

## Iteration 115 — H20 CARv2 graph-policy screen

- Added `bench/v4_flash_tp_car_policy_ab.py`, a same-process full-pipeline A/B harness that leaves the exact baseline dispatch untouched and captures candidate CARv2 algorithms/block counts into separate CUDA graphs using the same communicator, weights, inputs, and routes.  Every measured replay receives a separate excluded 256 MiB L2 clear, timings are reduced to the TP4 max rank, and candidates are checked independently against the distributed reference.
- Swept M64/M128 across graph `1shot_pull` and `2shot_pull` with 24/32/40/48/56/64 pull blocks (stock control is graph `2shot_pull`, 64 blocks), using four balanced candidate/control batches x 100 cold samples per implementation.  All candidates pass the all-reduce gate and are bit-exact versus the stock graph output (`max_abs=0`).
- Most `1shot_pull` points lose or are directionally mixed.  Its apparent M64/48-block `1.01796x` and M64/64-block `1.00854x` wins have only two and one winning batches respectively; M128/64 loses by 2.88%.  Do not advance the one-shot direction.
- For M64, `2shot_pull/32` is the only candidate that wins all four paired batches: control/candidate medians `0.294608/0.288992 ms = 1.01943x`.  The numerically larger `2shot_pull/24` pooled result (`1.06161x`) wins only two of four batches because that window crosses a strong absolute-latency mode shift, so it is not credible yet.
- For M128, `2shot_pull/40` is the best directionally supported point: control/candidate medians `0.378928/0.372416 ms = 1.01749x`, with three of four batches winning.  Other block counts are neutral/mixed.
- Conclusion: H200's fixed 64-block geometry is plausibly suboptimal for the 78-SM H20, but this broad screen is noisy.  Advance only M64/32 and M128/40 to independent long 10x200 AB/BA audits before changing runtime policy; retain stock 64 blocks meanwhile.
- Artifact: `results/iter115_car_policy_screen_tp4_m64_m128_cold400_20260903.log`.

### Iteration 115b — reduced CARv2 pull-block signal does not survive long audit

- Re-tested the two screen-selected 2-shot candidates and their cross-points at M64/M128 with ten balanced candidate/control batches x 200 cold-L2 samples per implementation.  The control remains stock graph `2shot_pull/64`; every graph output is bit-exact versus control and every independent distributed-reference check passes.
- M64/32 reverses from the short screen's `1.01943x` to control/candidate `0.281952/0.283472 ms = 0.99464x`, with five of ten batch wins.  M64/40 is worse at `0.299600/0.310256 ms = 0.96565x`, despite six nominal batch wins; the mismatch between pooled medians and paired directions reflects the same large mode shifts seen in iteration 114.
- M128/32 is neutral/slightly slower at `0.376384/0.376656 ms = 0.99928x` with five of ten wins.  M128/40 is only `0.378160/0.377648 ms = 1.00136x`, also five of ten wins and far below a selectable effect.
- Conclusion: the H200-derived 64-block geometry is not disproven on H20 by end-to-end cold-L2 evidence.  Reject reduced pull-block dispatch and retain stock 64 blocks for M64/M128.  The screen's apparent 1.7-1.9% gains were window noise, not a durable optimization.
- Artifact: `results/iter115b_car_pull_blocks_long_tp4_m64_m128_cold2000_20260903.log`.

## Iteration 116 — selected W13 NCU launch hit a stale profiler driver

- Attempted a detailed, cache-controlled NCU capture of the first current TP4 M128 `route_gemm` launch, with profiling enabled only around one explicitly cold local pipeline.
- The application exited before any target kernel launch because the untracked remote root copy of `profile_v4_flash_tp_local.py` still expected the old four-tensor `make_weights` return, while the current benchmark returns six tensors including normalized global scales (`ValueError: too many values to unpack`).  This is profiler-driver skew, not a kernel correctness or performance failure.
- No NCU metrics or timing result were produced.  Replace only the stale profiler helper with the already-current local staging copy and repeat the identical capture; do not alter the selected kernel.
- Artifact: `results/iter116_selected_tp4_m128_w13_cold_ncu.log`.

### Iteration 116b — current W13 NCU report captured; metadata print is stale

- Replaced the remote profiler helper with the current six-tensor weight API and repeated the same detailed/cache-controlled NCU run.  NCU successfully profiled the selected M128 W13 `route_gemm` for all 20 replay passes and wrote a valid `.ncu-rep`.
- The Python process then exited only while printing metadata because the helper still referenced the subsequently removed `W13_FP16_PARTIAL` module flag.  This post-profile `AttributeError` does not invalidate the already-written report, but the shell's `&&` correctly prevented automatic report import.
- Preserve the report and failure log.  Next import the existing report read-only for bottleneck analysis, then remove the stale metadata field before future captures; no selected-kernel source changed in this iteration.
- Artifacts: `profile_v4_flash_tp_local.py`, `results/iter116b_selected_tp4_m128_w13_cold_ncu.log`, and `results/iter116b_selected_tp4_m128_w13_cold_ncu.ncu-rep`.

### Iteration 116c — selected M128 W13 is issue/occupancy bound, not at HBM roof

- Parsed the valid iteration 116b report for the current TP4 M128 split2 W13 specialization: grid 5,136 CTAs, 128 threads, 55 registers/thread, 21.50 KiB dynamic plus 1.02 KiB static shared memory, and 7.32 waves/SM.
- NCU duration is `213.47 us` under kernel replay.  SM/issue-slot throughput is `77.53%`, while DRAM throughput is only `54.34%`; L2 throughput is `67.33%`, effective memory throughput is `2.61 TB/s`, and cold-streaming L2 hit rate is `5.59%`.  The kernel is therefore not sitting at the H20 HBM roof.
- Both registers and shared memory limit residency to nine 128-thread CTAs/SM (36 warps, 56.25% theoretical occupancy); achieved occupancy is 52.95% / 33.89 warps.  Branch efficiency is 70.88%, consistent with route-bound predicates and max-grid tail work.
- Direction: seek a single-warpgoup W13 change that reduces per-K128 issue/dequant/synchronization work or resource footprint without the dual-WG candidate's occupancy loss.  Weight bandwidth/cache-policy and CAR geometry are already separately gated; do not infer a bandwidth-only solution from this profile.
- Artifact: `results/iter116c_selected_tp4_m128_w13_ncu_details_20260903.log`.

## Iteration 117 — compact per-K128 interleaved scale layout correctness

- Hypothesis: append each pre-swizzled 8192-byte MXFP4 K128 weight tile with exactly its 512-byte (N128 x 4) E8M0 scales.  This preserves one bulk TMA and identical total bytes per tile quartet while reducing two-stage dynamic scale storage from 4096 to 1024 bytes.
- Implementation: added opt-in `V4_COMPACT_INTERLEAVED_SCALE=1`, compact model-load packing, per-stage scale addressing, K=256 support, benchmark metadata, and layout-aware paired comparison weights.  Kept launch bounds unchanged to isolate layout/dataflow.
- Verification on H20 GPU1:
  - TP4 shape I/rank=512, M8 balanced, auto split4: route/quant exact; W13 cosine 0.999999998, activation cosine 0.999999759, W2 cosine 0.999997256.
  - TP4 shape I/rank=512, M8 skew, forced split2: route/quant exact; W13 cosine 0.999999997, activation cosine 0.999999649, W2 cosine 0.999997235.
  - TP8 shape I/rank=256, M8 balanced, auto split4: route/quant exact; W13 cosine 0.999999997, activation cosine 0.999999745, W2 cosine 0.999997278.
- Result: all three paths passed and stayed finite.  Candidate is correctness-qualified for cold-L2 paired performance testing; it is not selected yet.
- Evidence: `results/iter117_compact_interleaved_scale_correctness_20260903.log`.

## Iteration 117b — compact-layout M128 TP4 screen launch failure

- Intended test: same-process control/candidate TP4 M128 random-route screen, 6 outer x 100 replays, cold L2 before every graph replay.
- Failure: this torchrun build parsed the training-script argument `--m=128` as an ambiguous torchrun option (`--max-restarts`, `--monitor-interval`, `--module`, or `--master-*`) and exited before workers, CUDA Graph capture, or GPU timing.
- Result: no performance sample was produced and no kernel conclusion is drawn.  Retry must insert torchrun's `--` separator before the benchmark script arguments.
- Evidence: `results/iter117b_compact_interleaved_scale_m128_screen_20260903.log`.

## Iteration 117c — compact interleaved scale M128 TP4 screen

- Method: exact same-process control/candidate CUDA Graphs, TP4 random routes, rank-max latency, 6 outer x 100 samples per variant, per-sample AB/BA alternation, and a separate 256MiB L2 clear immediately before every replay outside CUDA timing.
- Correctness: graph outputs were bit-exact on all four ranks.
- Control: min/median/max 0.324000/0.354512/0.763392 ms.
- Compact candidate: min/median/max 0.317696/0.346992/0.400576 ms.
- Paired result: control/candidate = 1.021672x (2.12% latency reduction); candidate won all 6 outer batch medians despite common temporal drift.
- Result: promising screen.  Keep candidate provisional and test smaller M plus a formal 10 x 200 run before selection.
- Evidence: `results/iter117c_compact_interleaved_scale_m128_screen_20260903.log`.

## Iteration 117d — compact interleaved scale M32 TP4 screen

- Method: exact same-process control/candidate CUDA Graphs, TP4 random routes, rank-max latency, 6 outer x 100 samples per variant, per-sample AB/BA alternation, separate 256MiB cold-L2 clear outside timing.
- Correctness: graph outputs were bit-exact on all four ranks.
- Control: min/median/max 0.186464/0.192832/0.337664 ms.
- Compact candidate: min/median/max 0.183424/0.189472/0.195808 ms.
- Paired result: control/candidate = 1.017733x (1.74% latency reduction); candidate won all 6 outer batch medians.
- Result: benefit extends to the split4 M32 path.  Proceed to formal all-M 10 x 200 validation before selecting the default.
- Evidence: `results/iter117d_compact_interleaved_scale_m32_screen_20260903.log`.

## Iteration 117e — compact interleaved scale formal all-M TP4 validation

- Method: exact same-process control/candidate CUDA Graphs, TP4 random routes, rank-max latency, 10 outer x 200 samples per variant at every M, per-sample AB/BA alternation, and separate 256MiB cold-L2 clear outside timing.
- Correctness: candidate graph output was bit-exact to control on all four ranks at every M.
- Median latency control -> compact candidate:
  - M8: 0.074848 -> 0.073568 ms, 1.017399x.
  - M16: 0.119488 -> 0.117632 ms, 1.015778x.
  - M32: 0.195488 -> 0.192128 ms, 1.017488x.
  - M64: 0.283344 -> 0.277600 ms, 1.020692x.
  - M128: 0.365872 -> 0.358544 ms, 1.020438x.
- Geometric-mean latency: 0.178506307 -> 0.175288491 ms; control/candidate 1.018357x (1.80% reduction).
- Stability: compact won all 50 paired outer batch medians (10/10 at each M), including the intervals with shared temporal drift.
- Result: accept compact per-K128 scales as the new selected layout; switch its default on, then rerun default-path correctness and exact Humming comparison.
- Evidence: `results/iter117e_compact_interleaved_scale_allm_formal_20260903.log`.

## Iteration 117f — select compact interleaved scales by default

- Change: default `V4_COMPACT_INTERLEAVED_SCALE` from 0 to 1 after the formal all-M win; expose `compact_scale` in the correctness-test record.
- Default TP4 I/rank=512, M8 balanced: compact_scale=True, routes/quant exact; W13 cosine 0.999999998, activation cosine 0.999999759, W2 cosine 0.999997256.
- Default TP8-shape I/rank=256, M8 balanced: compact_scale=True, routes/quant exact; W13 cosine 0.999999997, activation cosine 0.999999745, W2 cosine 0.999997278.
- Result: selected default passes both TP4 and TP8-shape paths without an environment override.
- Evidence: `results/iter117f_select_compact_scale_default_correctness_20260903.log`.

## Iteration 117g — selected compact default versus exact MXFP4 Humming

- Benchmark contract: DeepSeek-V4-Flash TP4 (H=4096, I/rank=512, E=256, precomputed top-k6 routes), no router/dispatch/combine; exact Humming MXFP4 indexed W13+W2 and custom full local MoE path each feed the same SGLang `CustomAllReduceV2`.  Both are CUDA Graphs.  Ten balanced batch-order AB/BA windows x 200 samples/M use rank-max timing and a separate excluded 256MiB L2 clear before every replay.
- Environment confirms H20 with 78 SM, compact_interleaved_scale=true, fused small-M multicast=true, and streaming weight evict-first=true.
- Humming/custom medians and Humming/custom speedup:
  - M8: 0.090176/0.073568 ms = 1.225750x.
  - M16: 0.146048/0.118848 ms = 1.228864x.
  - M32: 0.232816/0.204336 ms = 1.139378x.
  - M64: 0.331872/0.299424 ms = 1.108368x.
  - M128: 0.410048/0.378512 ms = 1.083316x.
- Five-shape geometric means: Humming 0.210902116 ms, custom 0.182506525 ms, Humming/custom 1.155587x (15.56% speedup ratio over custom).
- Correctness: both implementations passed independent local/reference and all-reduce checks at every M; custom minimum cosine 0.999995575, maximum relative L2 0.002974846, finite on every rank.
- Remaining 1.20x target: custom must reach <=0.175751763 ms; residual 0.006754762 ms (3.70% of current custom geomean).
- Result: compact layout is accepted and improves the exact-Humming headline, but the 1.20x five-shape target remains unproven.  Continue on the M32-M128 W13 issue/occupancy bottleneck.
- Evidence: `results/iter117g_compact_default_exact_humming_tp4_formal_cold2000_20260903.log`.

## Iteration 118 — W13-only 10-CTA launch-bound correctness

- Hypothesis: compact scale storage removes the shared-memory barrier to 10 resident 128-thread W13 CTAs; an opt-in W13-only minimum-block launch bound may reduce the current 55 registers/thread to at most 51 and raise occupancy.  W2 receives only a min-block value of one in this probe and must be checked at cubin level for unchanged allocation.
- Implementation: added opt-in `V4_W13_LAUNCH_BOUND_10=1`, guarded it against global launch bounds and 256-thread W13 variants, and exposed it in comparison/correctness/profile metadata.  Default remains off.
- TP4 I/rank=512 M8 balanced split4 and maximal-skew forced split2 both pass; W13 cosine >=0.999999997 and W2 cosine >=0.999997235.
- TP8-shape I/rank=256 M8 balanced also passes; W13 cosine 0.999999997 and W2 cosine 0.999997278.
- Result: correctness-qualified only.  Inspect cubin resources before any cold-L2 performance screen.
- Evidence: `results/iter118_w13_launch_bound10_correctness_20260903.log`.

## Iteration 118a — neutralize W2 side effect of W13 launch-bound probe

- Cubin audit of iteration 118 found the W13 objective was met (55 -> 48 registers/thread), but the non-W13 false arm `min_blocks=1` let ptxas inflate W2 from 55 to 62 registers/thread, dropping its register-limited residency from nine to eight CTAs.  That mixed candidate was not timed.
- Changed the non-W13 arm to `min_blocks=9`, matching W2's natural occupancy ceiling, while retaining ten for W13.
- Resource result: TP4/TP8 W13 split2/split4 are 48 registers/thread with no local allocation; K=512 and K=256 W2 are 53 registers/thread with no local allocation.  Thus W13 can reach ten resident CTAs and W2 retains nine-register-limited-CTA capacity.
- Correctness again passes TP4 split4, TP4 split2 maximal skew, and TP8-shape; W13 cosine >=0.999999997 and W2 cosine >=0.999997235.
- Result: corrected candidate is qualified for paired cold-L2 timing; default remains off.
- Evidence: `results/iter118a_w13_launch_bound10_w2_neutral_correctness_resources_20260903.log`.

## Iteration 118b — W13 launch-bound M128 paired screen blocked by bitwise gate

- Intended method: TP4 M128 random routes, same-process control/candidate CUDA Graphs, 6 x 100 rank-max samples, AB/BA replay alternation, separate excluded 256MiB cold-L2 clear per replay.
- The run stopped before timing because control and candidate graph outputs were not bitwise equal on all ranks.
- This does not contradict iteration 118a's independent reference correctness: changing W13 residency changes the inter-CTA order of split2 atomic additions, so floating-point last bits need not be bit-identical even when both satisfy the numerical reference bounds.
- Result: no latency sample and no performance conclusion.  Extend the paired harness to report and gate finite/cosine/relative-L2 for codegen variants that can reorder split-K accumulation, while retaining exact equality reporting.
- Evidence: `results/iter118b_w13_launch_bound10_m128_screen_cold600_20260903.log`.

## Iteration 118c — W13 10-CTA launch bound is a large M128 regression

- Extended the paired harness for explicitly identified split-K reordering flags: exact equality is still reported, while acceptance requires all-rank finite output, cosine >=0.99999, and relative L2 <=0.005.  All other flags still require bitwise equality.
- TP4 M128 random-route method: same-process control/candidate CUDA Graphs, 6 x 100 rank-max samples, per-replay AB/BA alternation, separate excluded 256MiB cold-L2 clear before every replay.
- Numerical comparison passes: cosine 0.999999225, relative L2 0.001221262, finite on all ranks; bitwise equality is false as expected from changed split2 atomic arrival order.
- Control min/median/max: 0.316704/0.343600/0.703712 ms.  Candidate: 0.337376/0.367152/0.415776 ms.
- Result: control/candidate = 0.935852x; the launch-bound candidate is 6.85% slower and loses all 6 batch medians.  Reject it immediately; higher nominal occupancy does not repay the 55 -> 48 register constraint.  Default remains off and no smaller-M sweep is justified.
- Evidence: `results/iter118c_w13_launch_bound10_m128_screen_cold600_20260903.log`.

## Iteration 119 — selected compact W13 M128 NCU capture

- Captured the first selected TP4 M128 random-route W13 `route_gemm<4096,1024,split2>` launch with Nsight Compute `detailed` over 20 passes.
- The profile helper explicitly records compact_interleaved_scale=true, W13 launch-bound probe=false, auto split policy, H20 60MiB L2, and a separate 256MiB cold-L2 clear immediately before the local pipeline.
- Collection completed successfully and produced a valid report.  Metric extraction and comparison with iteration 116c are deferred to the next audit so the raw artifact remains independently reproducible.
- Evidence: `results/iter119_compact_selected_tp4_m128_w13_cold_ncu.{log,ncu-rep}`.

## Iteration 119a — compact selected W13 NCU interpretation

- Against iteration 116c's pre-compact selected W13 profile, compact layout reduces dynamic shared memory from 21.50 to 18.43 kB/block and the separately collected NCU duration from 213.47 to 206.30 us (1.0348x).  Registers remain 55/thread with no spills.
- DRAM throughput rises 54.34% -> 56.22% (2.61 -> 2.71 TB/s), L2 throughput 67.33% -> 69.47%, while SM issue utilization changes 77.53% -> 76.07%.  The kernel remains ALU/issue-heavy rather than at the HBM roof.
- Achieved occupancy remains 52.92% (33.87 warps/SM): both register and shared-memory limits still report nine CTAs/SM.  The runtime selected only 200.70 kB shared configuration for the compact kernel even though H20 exposes 233,472 bytes/SM and 232,448 bytes/block opt-in.
- Branch efficiency remains 70.88%; uncoalesced-access warning remains 80,384 excessive sectors (7% of 1,224,096), so compact's gain comes from simpler scale staging/addressing rather than fewer sectors.
- Finding: iteration 118c's 48-register launch bound could not create a tenth CTA under the automatic 200.70 kB carveout.  A bounded next test should combine that register cap with maximum shared-memory carveout; only this combination can test actual 10-CTA residency.
- Evidence: `results/iter119a_compact_selected_tp4_m128_w13_ncu_details_20260903.log` and iteration 119 report.

## Iteration 120 — W13 10-CTA plus maximum shared carveout correctness

- Added opt-in `V4_W13_LB10_MAX_SMEM=1`: it enables the previously validated 10-CTA W13 register bound and sets `cudaFuncAttributePreferredSharedMemoryCarveout` to `cudaSharedmemCarveoutMaxShared` for W13 instantiations only.  Default remains off.
- Rationale: H20 reports 233,472 shared bytes/SM and 232,448 bytes/block opt-in, while selected compact W13 uses 18,432 dynamic bytes plus static/driver overhead.  Maximum carveout can accommodate ten CTAs; automatic 200.70 kB configuration could not.
- TP4 I/rank=512 M8 balanced split4 and maximal-skew forced split2 pass; W13 cosine >=0.999999997 and W2 cosine >=0.999997235.
- TP8-shape I/rank=256 M8 balanced passes; W13 cosine 0.999999997 and W2 cosine 0.999997278.
- Result: combination is correctness-qualified.  Profile actual shared configuration/occupancy before timing.
- Evidence: `results/iter120_w13_lb10_max_smem_correctness_20260903.log`.

## Iteration 120a — W13 10-CTA maximum-carveout occupancy capture

- Captured the TP4 M128 random-route split2 W13 candidate with Nsight Compute LaunchStats+Occupancy in one pass.
- The helper confirms compact scales, launch_bound_10=true, max_smem_carveout=true, and the standard excluded 256MiB cold-L2 clear.
- Collection completed and produced a valid report.  Parse the raw report next to determine configured shared memory and actual theoretical/achieved residency before performance timing.
- Evidence: `results/iter120a_w13_lb10_max_smem_m128_ncu.{log,ncu-rep}`.

## Iteration 120b — maximum carveout creates the intended tenth W13 CTA

- NCU confirms the compound candidate uses 48 registers/thread, 233.47 kB shared configuration, 18.43 kB dynamic + 1.02 kB static + 1.02 kB driver shared/block.
- Residency limits are now registers=10 CTAs and shared memory=11 CTAs, versus selected control's 9/9.  Theoretical occupancy rises 56.25% -> 62.50%; achieved occupancy rises 52.92% -> 58.37% (33.87 -> 37.36 active warps/SM).
- Grid remains 5,136 CTAs x 128 threads.  This verifies the intended mechanism that iteration 118c lacked.
- Result: proceed to the same-process M128 cold-L2 paired latency gate.  Occupancy is mechanism evidence only, not a performance win.
- Evidence: `results/iter120b_w13_lb10_max_smem_m128_ncu_details_20260903.log`.

## Iteration 120c — real 10-CTA W13 occupancy still regresses M128

- Method: TP4 M128 random routes, same-process selected-control/compound-candidate CUDA Graphs, 6 x 100 rank-max samples, per-replay AB/BA alternation, separate excluded 256MiB cold-L2 clear.
- Numerical comparison passes: cosine 0.999999583, relative L2 0.001013555, finite on all ranks; bitwise equality is false because split2 atomic arrival order changes.
- Control min/median/max: 0.316736/0.342576/0.542688 ms.  Candidate: 0.337120/0.365120/0.398720 ms.
- Result: control/candidate = 0.938256x; despite NCU-confirmed 10-CTA residency, the candidate is 6.58% slower and loses all 6 batch medians.  The 48-register constraint itself dominates any occupancy benefit.  Reject the compound flag and keep both launch-bound/carveout defaults off.
- Evidence: `results/iter120c_w13_lb10_max_smem_m128_screen_cold600_20260903.log`.

## Iteration 121 — selected compact W13 focused stall capture

- Captured TP4 M128 random-route selected W13 with Nsight Compute SchedulerStats, WarpStateStats, and InstructionStats over ten passes.
- The helper confirms compact selected defaults, launch-bound/carveout probes off, auto split2, and the standard excluded 256MiB cold-L2 clear.
- Collection completed successfully and produced a valid report.  Parse and rank the issue stalls next before choosing another kernel change.
- Evidence: `results/iter121_compact_selected_m128_w13_sched_ncu.{log,ncu-rep}`.

## Iteration 121a — selected compact W13 scheduler interpretation

- Scheduler is eligible on 78.07% of cycles and has no eligible warp on 21.93%; it issues 0.78 warp/scheduler/cycle from 8.48 active and 2.75 eligible warps/scheduler.
- Warp cycles per issued instruction are 10.86.  Average active/not-predicated threads are 30.33/30.12, so ordinary lane underfill is small despite the low branch-efficiency counter.
- The kernel executes 79,693,436 instructions (79,738,732 issued) in this launch.
- This NCU section set did not include the optional PC-sampling counter, so it cannot attribute the 21.93% no-eligible interval to barrier, scoreboard, wait, or dispatch stalls.  Collect explicit `smsp__warp_issue_stalled_*` metrics before changing code.
- Evidence: `results/iter121a_compact_selected_m128_w13_sched_details_20260903.log`.

## Iteration 121b — explicit selected-W13 stall-metric capture

- Collected all nineteen `smsp__warp_issue_stalled_*_per_warp_active` metrics for selected compact TP4 M128 split2 W13 over seven replay passes.
- The profile uses the same random-route cold-L2 local replay and leaves both rejected launch-bound/carveout probes off.
- Collection completed successfully with a valid report.  Parse and rank stall ratios next.
- Evidence: `results/iter121b_compact_selected_m128_w13_stall_metrics_ncu.{log,ncu-rep}`.

## Iteration 121c — selected compact W13 explicit stall ranking

- Active-warp stall distribution: not-selected 23.37%, barrier 20.15%, math-pipe throttle 11.83%, selected 9.23%, fixed-latency wait 9.20%, GMMA 9.15%, long scoreboard 6.92%, short scoreboard 3.92%, dispatch stall 3.61%, branch resolving 1.95%, no-instruction 0.57%; all other classes <=0.10%.
- The largest actionable class is CTA barrier waiting, not global-memory throttle (LG/TEX throttle are zero and long scoreboard is only 6.92%).  Math-pipe and GMMA pressure are secondary and consistent with NCU's ALU-heavy diagnosis.
- The report also confirms selected resources: 55 registers/thread allocated as 56, 20.48 kB total shared/block, nine CTAs and 36 theoretical warps/SM.
- Direction: inspect the per-K128 barrier topology and test a semantics-preserving reduction of CTA-wide barriers before changing dequant math.  Preserve the existing leader-only mbarrier polling and split-K arithmetic.
- Evidence: `results/iter121c_compact_selected_m128_w13_stall_metrics_20260903.log`.

## Iteration 122 — unsynchronized early W13 stage refill is numerically invalid

- Candidate moved the same-stage TMA refill from the end of each K128 iteration to after the final K32 WGMMA commit, intending to overlap it with GMMA wait and accumulation.  W2 and split-K arithmetic were unchanged; default remained off.
- TP4 split4 balanced degrades to W13 cosine 0.999993116 / rel-L2 0.003710623 and activation cosine 0.999982946 / rel-L2 0.005841401.  TP8-shape split4 similarly degrades to W13 cosine 0.999992579 and activation rel-L2 0.005338274.
- TP4 forced split2 happened to retain prior accuracy, but this does not validate the race.  The test's broad legacy pass threshold prints OK, yet the candidate clearly fails the selected-control numerical standard.
- Cause: WGMMA commit is not a cross-warp completion barrier for prior shared loads; the TMA issuer can overwrite the released stage before another warp completes its final load.
- Result: reject the unsynchronized candidate without performance timing.  A repair must use an explicit cross-warp release handshake before the refill.
- Evidence: `results/iter122_w13_early_stage_refill_correctness_20260903.log`.

## Iteration 122a — split-phase release barrier does not repair early refill

- Added a named-barrier release handshake before early refill: three non-issuer warps use non-blocking `bar.arrive`, while the TMA issuer warp uses `bar.sync` before overwriting the stage.  Also strengthened the standalone gate to W13 rel-L2 <=0.001, activation rel-L2 <=0.002, W2 cosine >=0.99999 / rel-L2 <=0.005, and finite output.
- The strengthened gate correctly fails TP4 split4: W13 cosine/rel-L2 0.999988722/0.004749736 and activation cosine/rel-L2 0.999977113/0.006765970.  TP8-shape split4 also fails with W13 rel-L2 0.004476430 and activation rel-L2 0.005196902.
- TP4 forced split2 still passes, but split4 corruption proves the stage-release protocol remains unsafe; the hybrid named-barrier arrival/sync does not establish the required lifecycle for this reuse point.
- Result: reject without performance timing.  Keep the stronger correctness gate.  If early refill is tested once more, require a full four-warp barrier after WGMMA commit; otherwise retain end-of-iteration refill.
- Evidence: `results/iter122a_w13_early_stage_refill_release_barrier_correctness_20260903.log`.

## Iteration 122b — full-barrier pre-wait refill remains invalid

- Replaced the split-phase release handshake with a full four-warp `bar.sync` after the final K32 WGMMA commit and before same-stage TMA refill.
- The strengthened gate still fails TP4 split4 (W13 cosine/rel-L2 0.999992477/0.003879049; activation rel-L2 0.005216170) and TP8-shape split4 (W13 cosine/rel-L2 0.999988827/0.004727490; activation rel-L2 0.004419442).  Forced split2 again happens to pass.
- A full cross-warp memory barrier is therefore insufficient while WGMMA is still outstanding.  The likely lifetime is the asynchronous WGMMA register-source group: issuing refill/address work before `warpgroup_wait<0>` permits compiler/register reuse before hardware completion.
- Result: reject without timing.  Any final early-refill probe must occur only after `warpgroup_wait<0>` and may overlap only the scalar accumulation/loop tail.
- Evidence: `results/iter122b_w13_early_refill_full_barrier_correctness_20260903.log`.

## Iteration 122c — issuer-local post-WGMMA-wait refill is still unsafe

- Moved early refill after the final `warpgroup_wait<0>` and before tile accumulation, removing the pre-wait barrier.  This was intended to respect asynchronous WGMMA source lifetime.
- The strengthened gate still fails TP4 split4 (W13 cosine/rel-L2 0.999996422/0.002675167; activation rel-L2 0.003549752) and TP8-shape split4 (W13 cosine/rel-L2 0.999995812/0.002894152; activation rel-L2 0.004505135).  Forced split2 again passes.
- The reduced but persistent corruption means the issuer warp can return from its wait and overwrite shared stage data before another warpgroup member has completed the corresponding lifetime.
- Result: reject without timing.  The only remaining safe placement is a full four-warp barrier after `warpgroup_wait<0>`; test that once, then close early refill regardless of performance if it fails.
- Evidence: `results/iter122c_w13_post_wait_stage_refill_correctness_20260903.log`.

## Iteration 122d — post-wait full barrier still cannot validate early refill

- Added a full four-warp named barrier after the final `warpgroup_wait<0>` and before refill, the strongest candidate stage-release placement short of retaining the original end-of-iteration order.
- The strengthened gate still fails TP4 split4 (W13 cosine/rel-L2 0.999988170/0.004864239; activation rel-L2 0.005961795) and TP8-shape split4 (W13 cosine/rel-L2 0.999995290/0.003069672; activation rel-L2 0.002781517).  Forced split2 again passes.
- Since even post-wait full synchronization does not preserve split4, moving the refill call across the tile-accumulation scope is not semantically safe in this implementation (whether due to async proxy/barrier lifecycle or compiler operand lifetime).
- Result: permanently reject the early-stage-refill direction without performance timing.  Retain the selected end-of-iteration refill and remove/disable this numerically unsafe opt-in path.
- Evidence: `results/iter122d_w13_post_wait_full_barrier_refill_correctness_20260903.log`.

## Iteration 123 — remove the rejected early-refill path

- Deleted `V4_W13_EARLY_STAGE_REFILL` end to end: Python validation/configuration, CUDA compile-time branch, benchmark/profile metadata, and compare-harness support.  The selected kernel now has only the original end-of-iteration stage refill, so the known-corrupt split4 experiment cannot be enabled accidentally.
- Strengthened default correctness passes TP4 balanced auto split4: W13 cosine/rel-L2 0.999999998/0.000076187, activation 0.999999759/0.000694956, W2 0.999997256/0.002342691 finite.
- TP4 skew forced split2 passes: W13 rel-L2 0.000077291, activation 0.000838720, W2 0.002351904 finite.  TP8-shape balanced auto split4 also passes: W13 rel-L2 0.000076998, activation 0.000714756, W2 0.002333323 finite.
- Result: retain the compact selected default and close early refill permanently.  The next performance experiment must preserve this full-route numerical gate.
- Evidence: `results/iter123_remove_unsafe_early_refill_default_correctness_20260903.log`.

## Iteration 124 — compact selected W2 exposes avoidable local barrier-address traffic

- Profiled the selected TP4 M128 random-route W2 `route_gemm<512,4096,split1>` after the standard excluded 256 MiB cold-L2 clear, selecting the second route-GEMM launch explicitly.  Full NCU replay completed in 38 passes.
- W2 duration is 114.88 us under replay, with compute/DRAM/L2 throughput 77.47%/50.70%/64.19%, 2.44 TB/s DRAM, 55 registers/thread, 18.43 KiB dynamic shared memory, nine resident CTAs, and 54.27% achieved occupancy.  Scheduler eligible/no-eligible fractions are 78.60%/21.40%.
- Per-issued-instruction warp-state costs are led by not-selected 2.74 cycles, barrier 1.90, math-pipe throttle 1.37, long scoreboard 1.14, fixed wait 1.13, selected 1.00, GMMA 0.55, short scoreboard 0.47, and dispatch 0.39.  W2 remains issue/ALU/barrier bound rather than HBM-roof bound.
- Source/SASS correlation identifies explicit local memory caused by `uint32_t barrier_addr[kStages]`: 31,872 `STL.64`-classified stores and 127,488 `LDL` loads, with zero register spilling.  For 7,968 active M-block/N-tile CTAs, these counts are exactly four warp stores per CTA and four K128 loads per warp.  This is address-array traffic, not an unavoidable accumulator spill.
- Direction: replace the dynamically indexed local array with a shared-address base plus `stage * 8`, behind a reversible compile-time flag; require full TP4/TP8 numerical gates and same-process cold-L2 A/B before selection.
- Evidence: `results/iter124_compact_selected_tp4_m128_w2_cold_ncu.{log,ncu-rep}` and `results/iter124a_compact_selected_tp4_m128_w2_ncu_details_20260903.log`.

## Iteration 125 — direct weight-barrier addressing passes TP4/TP8 correctness

- Added opt-in `V4_DIRECT_BARRIER_ADDR=1`.  Instead of materializing the two `full_barriers` shared addresses in a dynamically indexed local array, the candidate keeps the first shared address and computes `base + stage * 8`; all mbarrier init, TMA arrival, copy completion, and consumer-wait sites use the same helper.  Default remains off for an unbiased paired comparison.
- TP4 balanced auto split4 exactly matches the selected numerical metrics: W13 cosine/rel-L2 0.999999998/0.000076187, activation 0.999999759/0.000694956, W2 0.999997256/0.002342691 finite.
- TP4 skew forced split2 passes with W13/activation/W2 rel-L2 0.000077291/0.000838720/0.002351904.  TP8-shape balanced auto split4 passes with 0.000076998/0.000714756/0.002333323.  Route alignment and input quantization remain exact.
- Result: the address substitution is semantics-preserving across split4, split2, TP4, and TP8 shapes.  Proceed to same-process TP4 M128 cold-L2 A/B, then verify SASS local traffic only if latency is favorable.
- Evidence: `results/iter125_direct_barrier_addr_correctness_20260903.log`.

### Iteration 125b — direct barrier addressing is a consistent M128 regression

- Same-process TP4 M128 random-route comparison used identical inputs, weights, routes, and one CARv2 communicator, with eight batches x 100 samples per arm.  Replay order alternated A/B then B/A and every replay received a separate excluded 256 MiB cold-L2 clear.
- Control min/median/max is 0.315904/0.352528/0.475040 ms; direct-address candidate is 0.319616/0.356800/0.397440 ms.  Control/candidate is 0.988027x, so the candidate is 4.272 us or 1.21% slower.
- The candidate loses all eight paired batch medians by 3.44--4.58 us.  Complete graph outputs are bitwise equal on every TP rank (cosine 1.0, rel-L2/max-abs 0), excluding arithmetic or communicator drift.
- Result: reject `V4_DIRECT_BARRIER_ADDR=1` and retain the local two-address array default.  The local accesses are L1-resident and apparently better scheduled than repeated uniform base/stage arithmetic; do not infer latency savings from the sector diagnostic alone.
- Evidence: `results/iter125b_direct_barrier_addr_tp4_m128_paired_cold800_20260903.log`.

## Iteration 126 — two-way outer K128 unroll passes TP4/TP8 correctness

- Added opt-in `V4_ROUTE_K_UNROLL2=1`, applying `#pragma unroll 2` to the existing route-GEMM outer K128 loop while leaving the selected non-unrolled path as control.  With two weight stages, each unrolled body has a fixed stage parity and can simplify barrier/shared-address indexing without changing the inner K32 WGMMA order or task geometry.
- TP4 balanced auto split4 reproduces selected metrics exactly: W13 cosine/rel-L2 0.999999998/0.000076187, activation 0.999999759/0.000694956, W2 0.999997256/0.002342691 finite.
- TP4 skew forced split2 and TP8-shape balanced auto split4 also reproduce the selected W13/activation/W2 errors exactly; all route-alignment and input-quantization checks pass.
- Result: two-way unrolling is numerically safe for every required topology/split gate.  Keep default off pending same-process TP4 M128 cold-L2 timing and resource/SASS inspection if favorable.
- Evidence: `results/iter126_route_k_unroll2_correctness_20260903.log`.

### Iteration 126b — two-way K128 unroll wins TP4 M128 consistently

- Same-process TP4 M128 random-route A/B used eight batches x 100 samples per arm, identical tensors and CARv2 communicator, per-sample alternating order, rank-max timing, and a separate excluded 256 MiB cold-L2 clear immediately before every replay.
- Control min/median/max is 0.316224/0.354144/0.583776 ms; unroll2 candidate is 0.310656/0.347488/0.385312 ms.  Control/candidate is 1.019155x, a 6.656 us or 1.88% latency reduction.
- Candidate wins all eight paired batch medians by 5.696--7.184 us.  Complete graph outputs are bitwise equal on all ranks (cosine 1.0, rel-L2/max-abs 0).
- Result: this is a directionally strong candidate rather than a drift-sized screen.  Keep opt-in and test M8/M16/M32/M64 under the same cold-L2 paired protocol before selecting; inspect resources/SASS to confirm whether stage-index constant folding removed local traffic.
- Evidence: `results/iter126b_route_k_unroll2_tp4_m128_paired_cold800_20260903.log`.

### Iteration 126c — unroll2 wins M16/M32/M64; M8 is blocked by an over-strict bitwise gate

- Screened M8/M16/M32/M64 sequentially with same-process TP4 control/candidate graphs, five batches x 100 cold-L2 rank-max samples per arm, per-replay alternating order, and the standard separate excluded 256 MiB clear.
- M16 control/candidate medians are 0.117184/0.116000 ms = 1.010207x; M32 0.189568/0.186448 = 1.016734x; M64 0.275936/0.271280 = 1.017163x.  Candidate wins all five paired batch medians at each of these three shapes, and outputs are bitwise identical on all ranks.
- M8 stopped before timing because the generic comparison harness required bitwise identity for this newly added flag.  Unrolling permits compiler reassociation/scheduling across adjacent K128 accumulation bodies: candidate/control remains finite with cosine 0.999999762 and rel-L2 0.000721341, well inside the strengthened production thresholds, but is not bitwise equal (max-abs 1536 at the large synthetic output scale).
- This is a harness-classification failure, not evidence of reference incorrectness: iteration 126 independently passed TP4 split4, split2, and TP8 full-route references.  Add `V4_ROUTE_K_UNROLL2` to the existing tolerance-qualified reordering set and rerun M8; selection still requires an independent exact-Humming/reference audit.
- Evidence: `results/iter126c_route_k_unroll2_tp4_m8_m16_m32_m64_paired_cold500_20260903.log`.

### Iteration 126d — tolerance-qualified harness rerun makes M8 neutral

- Renamed the compare harness's narrowly scoped set to `TOLERANCE_QUALIFIED_FLAGS` and added the already reference-validated unroll2 flag.  Non-bitwise candidates still must be finite with cosine >=0.99999 and rel-L2 <=0.005; all other flags remain bitwise-gated.
- Repeated TP4 M8 random-route timing for eight batches x 100 samples per arm with identical graphs/data/CARv2, rank-max events, alternating replay order, and an excluded 256 MiB cold-L2 clear before every replay.
- Control min/median/max is 0.072064/0.073536/0.227104 ms; unroll2 is 0.072352/0.073504/0.075712 ms, for control/candidate 1.000435x.  The 0.032 us pooled difference is neutral at this shape; batch directions are mixed.
- This repeat happened to be bitwise identical on all ranks, so iteration 126c's small non-bitwise difference is not a deterministic corruption signature.  Combined screens now show neutral M8 and consistent 1.02--1.92% wins for M16--M128.  Advance to resource/SASS confirmation and a long five-shape exact-Humming audit before selection.
- Evidence: `results/iter126d_route_k_unroll2_tp4_m8_paired_cold800_20260903.log`.

### Iteration 126e — exact-Humming five-shape formal audit reaches 1.1620x

- Ran exact MXFP4 Humming versus unroll2 custom TP4 graphs at M={8,16,32,64,128}, random routes, ten balanced batch-order windows x 200 samples per implementation/M.  Each replay had a separate excluded 256 MiB L2 clear; both paths used the same SGLang `CustomAllReduceV2` instance.
- Humming/custom medians (ms) and speedups: M8 0.089920/0.073664 = 1.220678x; M16 0.146048/0.117568 = 1.242243x; M32 0.233120/0.204016 = 1.142655x; M64 0.336448/0.296256 = 1.135666x; M128 0.409520/0.380352 = 1.076687x.
- Geometric means are Humming 0.211361020 ms and custom 0.181891459 ms, for 1.162017x (16.20%).  Relative to iteration 117g's selected formal window (1.155587x), the headline improves by about 0.56 percentage points; cross-window drift means causal selection still rests on iterations 126b--d's direct control/candidate results.
- Both implementations pass independent full-pipeline references and all-reduce checks at every M.  Custom minimum cosine is 0.999995575 and maximum rel-L2 is 0.002974846, all finite.
- At this Humming geometric mean, 1.20x requires custom <=0.176134183 ms; the remaining reduction is 0.005757276 ms, or 3.17% of the candidate.  Retain opt-in until a long same-process five-shape control/candidate audit confirms the broad screen before changing the default.
- Evidence: `results/iter126e_route_k_unroll2_exact_humming_tp4_formal_cold2000_20260903.log`.

### Iteration 126f — long five-shape self-control confirms unroll2 broadly

- Ran the same-process TP4 control/unroll2 comparison at all five required M values for ten batches x 200 samples per arm.  Every replay used rank-max CUDA events, AB/BA alternation, identical tensors/CARv2, and a separate excluded 256 MiB cold-L2 clear.
- Control/candidate medians (ms) and speedups: M8 0.073536/0.073472 = 1.000871x; M16 0.117376/0.116160 = 1.010468x; M32 0.192320/0.188960 = 1.017782x; M64 0.280448/0.275648 = 1.017413x; M128 0.363072/0.356128 = 1.019499x.
- Geometric means are control 0.176031251 ms and candidate 0.173740818 ms, for 1.013183x (1.32%).  Candidate wins every one of the 40 paired batch medians at M16--M128; M8 is neutral with ties/mixed sub-0.1-us differences.
- Outputs are bitwise identical at M8/M16/M32/M128.  M64 differs only by cosine 0.999999881 / rel-L2 0.000017986 and passes the tolerance-qualified gate; all values are finite.
- Result: select `V4_ROUTE_K_UNROLL2=1` as the new default.  It is reference-correct, TP8-runnable, neutral at M8, and provides direction-consistent 1.0--1.95% full-pipeline gains at all larger shapes.
- Evidence: `results/iter126f_route_k_unroll2_tp4_allm_paired_cold2000_20260903.log`.

## Iteration 127 — select two-way K128 unroll by default

- Changed the no-environment default of `V4_ROUTE_K_UNROLL2` from 0 to 1 after the long five-shape self-control established a 1.013183x geometric-mean gain with no required-shape regression.  The flag remains available as `0` for reproducible rollback/control measurements.
- Fresh selected-extension TP4 balanced auto split4 passes with W13/activation/W2 rel-L2 0.000076187/0.000694956/0.002342691; TP4 skew forced split2 passes with 0.000077291/0.000838720/0.002351904; TP8-shape balanced auto split4 passes with 0.000076998/0.000714756/0.002333323.  All outputs are finite, and route/quant checks are exact.
- Result: the production default now uses compact per-K128 scales plus two-way outer K unrolling for both W13 and W2.  Continue optimization from this selected baseline; all future benchmarks remain per-replay cold L2.
- Evidence: `results/iter127_select_route_k_unroll2_default_correctness_20260903.log`.

## Iteration 128 — unroll2 does not rescue ten-CTA W13 occupancy

- Re-tested the existing `V4_W13_LB10_MAX_SMEM=1` compound on the newly selected unroll2 path.  This is a materially different resource point from iteration 120: unroll2 already gives W13 48 registers/thread and zero stack without launch bounds, so the candidate primarily exposes the tenth resident CTA through maximum shared-memory carveout.
- TP4 balanced auto split4, TP4 skew forced split2, and TP8-shape balanced auto split4 all pass the strengthened full-route reference with errors identical to selected.  Cubin resource inspection confirms both arms keep W13 at 48 registers and W2 at 56, with zero stack/local allocation.
- Same-process TP4 M128, eight x 100 rank-max cold-L2 samples per arm: selected control min/median/max 0.311232/0.344288/0.528000 ms; ten-CTA candidate 0.321664/0.358800/0.426272 ms.  Control/candidate is 0.959554x, so the candidate is 14.512 us or 4.22% slower and loses all eight batch medians.
- Outputs are bitwise identical on all ranks.  Result: reject ten-CTA carveout again.  The regression survives removal of the old register-compression confound, showing the additional residency itself causes harmful issue/shared/TMA contention; retain automatic shared configuration and nine CTAs.
- Evidence: `results/iter128_unroll2_lb10_max_smem_correctness_20260903.log` and `results/iter128b_unroll2_lb10_max_smem_tp4_m128_paired_cold800_20260903.log`.

## Iteration 129 — four-way K128 unroll passes TP4/TP8 correctness

- Added opt-in `V4_ROUTE_K_UNROLL4=1` on top of the selected two-way default.  The candidate changes only the outer route-GEMM K128 loop pragma from unroll2 to unroll4; task geometry, TMA stages, barriers, K32 WGMMA order, epilogues, and CARv2 are unchanged.
- TP4 balanced auto split4 exactly reproduces selected W13/activation/W2 cosine and rel-L2 metrics.  TP4 skew forced split2 and TP8-shape balanced auto split4 likewise reproduce the selected errors exactly; route alignment and input quantization are exact and all outputs finite.
- Result: four-way unrolling is numerically safe across the required TP4/TP8 and split4/split2 gates.  Keep default at two-way pending cubin-resource inspection and same-process M128 cold-L2 timing; reject early if code/register growth removes the prior gain.
- Evidence: `results/iter129_route_k_unroll4_correctness_20260903.log`.

## Iteration 129b — M128 timing launch rejected before kernel execution

- The first M128 paired-timing invocation used malformed concatenated CLI options (`--m128`, `--outer8`, `--replays100`, and `--warmup-replays20`). `argparse` rejected them on all four ranks before graph construction or kernel execution.
- This run contains no performance evidence and does not change the four-way-unroll decision. The corrected invocation will use `--m 128 --outer 8 --replays 100 --warmup-replays 20`.
- Evidence: `results/iter129b_route_k_unroll4_tp4_m128_paired_cold800_20260903.log`

## Iteration 129c — torchrun consumed the corrected script option

- The spaced script arguments were syntactically valid for the benchmark, but this container's `torchrun` parser consumed the trailing `--m` as an abbreviated launcher option and rejected it as ambiguous before workers or kernels started.
- This run again contains no performance evidence. The next invocation will place an explicit `--` between launcher options and the training script so all benchmark options reach the script unchanged.
- Evidence: `results/iter129c_route_k_unroll4_tp4_m128_paired_cold800_20260903.log`

## Iteration 129d — four-way K128 unroll wins the M128 gate

- Cubin inspection shows the candidate remains spill-free but raises route-GEMM resources: W2 moves from 56 to 61 registers/thread and W13 from 48 to 52; stack/local allocation stays zero.
- Same-process TP4 M128, eight x 100 rank-max cold-L2 samples per arm with per-replay alternating A/B then B/A: selected two-way control min/median/max 0.310752/0.351776/0.464736 ms; four-way candidate 0.300768/0.337200/0.367904 ms.
- Control/candidate is 1.043227x: four-way unrolling removes 14.576 us (4.14% of control) and wins all eight batch medians despite the register increase. Outputs are bitwise identical on all ranks.
- Result: four-way unrolling passes the M128 performance gate. Continue with M16/M32/M64 and a separate M8 neutrality check before changing the default.
- Evidence: `results/iter129d_route_k_unroll4_tp4_m128_paired_cold800_20260903.log`

## Iteration 129e — four-way K128 unroll wins every required M

- Same-process TP4 random-route screening used five x 100 rank-max cold-L2 samples per arm, alternating A/B then B/A before every replay. Selected-two-way versus four-way median latency and control/candidate speedup: M8 0.073712/0.071456 ms (1.031572x), M16 0.116128/0.112496 ms (1.032286x), M32 0.188880/0.181648 ms (1.039813x), and M64 0.276432/0.264960 ms (1.043297x).
- Candidate min/median/max latency is 0.069984/0.071456/0.073184 ms at M8, 0.111392/0.112496/0.114496 ms at M16, 0.174048/0.181648/0.186272 ms at M32, and 0.245568/0.264960/0.270304 ms at M64. The control min/median/max values are respectively 0.072384/0.073712/0.220480, 0.114528/0.116128/0.230176, 0.180352/0.188880/0.334112, and 0.254336/0.276432/0.372960 ms.
- Four-way unrolling wins every one of the 20 batch medians and is bitwise identical on all ranks. Combined with the M128 gate, its five-shape geometric-mean self-control speedup is about 1.0380x with no required-shape regression.
- Result: promote four-way unrolling to the formal exact-Humming-plus-CARv2 benchmark gate. Do not change the production default until that 10 x 200 comparison and fresh no-environment TP4/TP8 correctness both pass.
- Evidence: `results/iter129e_route_k_unroll4_tp4_m8_m64_paired_cold2000_20260903.log`

## Iteration 129f — exact Humming plus CARv2 geometric mean reaches 1.2027x

- Formal TP4 random-route comparison uses exact Humming MXFP4 W13/W2 and the same SGLang `CustomAllReduceV2` instance for both paths, CUDA Graph replay, ten outer batches x 200 samples per implementation and M, rank-max timing, and a separate 256 MiB Triton clear immediately before every timed replay (clear excluded from events).
- Humming/custom median latency and Humming-over-custom speedup: M8 0.090368/0.071328 ms (1.266936x), M16 0.146208/0.113856 ms (1.284148x), M32 0.233904/0.201536 ms (1.160607x), M64 0.340064/0.285200 ms (1.192370x), and M128 0.409504/0.366336 ms (1.117837x).
- Custom min/median/max is 0.069824/0.071328/0.283744, 0.112224/0.113856/0.397088, 0.176480/0.201536/0.279552, 0.247296/0.285200/0.414048, and 0.308064/0.366336/1.196768 ms for M8 through M128. Humming min/median/max is 0.088512/0.090368/0.353120, 0.144256/0.146208/0.200000, 0.224800/0.233904/0.312352, 0.313792/0.340064/0.467200, and 0.379744/0.409504/0.724768 ms.
- Five-shape geometric-mean latency is 0.212211275 ms for Humming and 0.176441013 ms for custom, giving 1.202732x. This improves custom geometric mean by about 3.09% versus the selected unroll2 formal result (0.181891459 ms).
- Both paths pass the full route reference and all-reduce checks for every M. Custom minimum cosine is 0.999995575 and maximum rel-L2 is 0.002974846; all outputs are finite.
- Result: the opt-in four-way path clears the 1.2x aggregate goal in this run, but only by 0.23 percentage points. Select it provisionally, then require fresh no-environment TP4/TP8 correctness and an independent full formal repeat before treating the threshold as robust.
- Evidence: `results/iter129f_route_k_unroll4_exact_humming_tp4_allm_cold2000_20260903.log`

## Iteration 130 — provisional four-way default; first correctness launch lacked Humming path

- Changed the no-environment default of `V4_ROUTE_K_UNROLL4` from 0 to 1 and bumped the extension suffix to `v130sel`, while preserving `V4_ROUTE_K_UNROLL4=0` as the selected two-way control/rollback.
- The first fresh-default correctness invocation supplied only the repository on `PYTHONPATH`; `bench/test_v4_flash_tp_wgmma.py` exited at `from humming import ops` with `ModuleNotFoundError` before extension compilation or GPU execution.
- This run contains no numerical evidence. Repeat the same TP4 balanced auto-split4, TP4 skew forced-split2, and TP8-shape balanced auto-split4 gates with the exact Humming and SGLang source paths restored.
- Evidence: `results/iter130_select_route_k_unroll4_default_correctness_20260903.log`

## Iteration 130b — selected four-way default passes fresh TP4/TP8 correctness

- Re-ran with both route-unroll environment variables explicitly absent, loading the fresh `v130sel` extension. The reported configuration confirms `route_k_unroll2=True` and `route_k_unroll4=True`, so the four-way branch is selected by default.
- TP4 balanced auto split4 passes with W13/activation/W2 rel-L2 0.000076187/0.000694956/0.002342691; TP4 skew forced split2 passes with 0.000077291/0.000838720/0.002351904; TP8-shape balanced auto split4 passes with 0.000076998/0.000714756/0.002333323.
- Route preparation and activation quantization checks are exact in all three cases, all outputs are finite, and the errors exactly match the pre-selection four-way candidate gates.
- Result: correctness no longer blocks the four-way production default. Require one independent 10 x 200 exact-Humming-plus-CARv2 cold-L2 repeat to quantify threshold robustness.
- Evidence: `results/iter130b_select_route_k_unroll4_default_correctness_20260903.log`

## Iteration 130c — independent selected-default repeat confirms more than 1.2x

- Repeated the full exact-Humming-plus-CARv2 TP4 random-route benchmark with both route-unroll environment variables absent, fresh graph objects, ten x 200 rank-max cold-L2 samples per path and M, and identical batch-level AB/BA pairing.
- Humming/custom medians and speedups are: M8 0.090112/0.071200 ms (1.265618x), M16 0.146080/0.113760 ms (1.284107x), M32 0.233504/0.197392 ms (1.182946x), M64 0.340096/0.282784 ms (1.202671x), and M128 0.409552/0.363456 ms (1.126827x).
- Custom min/median/max is 0.069856/0.071200/0.213984, 0.112224/0.113760/0.154944, 0.176160/0.197392/0.265792, 0.246368/0.282784/0.350272, and 0.308192/0.363456/0.430912 ms for M8 through M128. Humming min/median/max is 0.088384/0.090112/0.431520, 0.143872/0.146080/0.193376, 0.224256/0.233504/0.294016, 0.314048/0.340096/0.423616, and 0.381952/0.409552/0.486528 ms.
- Five-shape geometric-mean latency is 0.211990140 ms for Humming and 0.175041553 ms for custom, giving 1.211085x. The two independent formal runs therefore both clear 1.2x (1.202732x and 1.211085x), while custom geometric mean differs by only 0.80% between them.
- Every M again passes the full route reference and same-CARv2 correctness checks; custom minimum cosine is 0.999995575, maximum rel-L2 is 0.002974846, and all outputs are finite.
- Result: select four-way outer K128 unrolling as the new optimization baseline. The aggregate target is now independently reproduced; continue from this baseline with fresh profiling rather than treating the first threshold crossing as sufficient.
- Evidence: `results/iter130c_selected_unroll4_exact_humming_tp4_allm_repeat_cold2000_20260903.log`

## Iteration 131 — selected four-way route GEMMs captured for fresh NCU analysis

- Captured both selected-default TP4 M128 random-route kernels in one Nsight Compute full-set report after the standard excluded 256 MiB cold-L2 clear: W13 `route_gemm<4096,1024,split2>` followed by W2 `route_gemm<512,4096,split1>`.
- NCU completed 38 replay passes for each kernel and the runtime metadata confirms both two-way and four-way flags are true, so this is the newly selected four-way path rather than the prior iteration-124 unroll2 profile.
- Result: profile collection succeeded. Preserve the raw report and filtered metric export; use the next analysis step to identify whether W13 or W2 now offers the larger issue/barrier/memory opportunity before changing source.
- Evidence: `results/iter131_selected_unroll4_tp4_m128_route_gemms_cold_ncu.{log,ncu-rep}` and `results/iter131a_selected_unroll4_tp4_m128_route_gemms_ncu_details_20260903.log`.

## Iteration 131b — four-way profile shifts the remaining opportunity to W13

- Exported the selected profile in human-readable and compact raw-metric forms and compared W2 directly with the iteration-124 unroll2 report. W2 replay duration falls from 114.88 to 105.728 us (-7.97%) and issued instructions per scheduler from 142,626.66 to 125,363.69 (-12.10%). DRAM read bandwidth rises from 2.409 to 2.618 TB/s.
- The gain survives a resource tradeoff: W2 registers rise from 55 in the iteration-124 cubin to 61, achieved occupancy falls from 54.27% to 48.26%, issue-active falls from 78.60% to 75.00%, and eligible warps/scheduler fall from 2.93 to 2.37. The dominant per-issued costs are now not-selected 2.144, barrier 2.037, long-scoreboard 1.137, wait 1.068, selected 1.000, math-pipe 1.256, GMMA 0.479, and short-scoreboard 0.491 cycles.
- Selected W13 is now the larger route kernel at 196.032 us, 52 registers, 52.86% achieved occupancy, 2.822 TB/s DRAM reads, 72.65% compute and 59.01% DRAM throughput. Its main per-issued costs are barrier 2.504, not-selected 2.196, wait 1.264, math-pipe 1.231, GMMA 1.036, selected 1.000, and long-scoreboard 0.799 cycles.
- Interpretation: four-way unrolling fully covers W2's four K128 iterations, so further global K unrolling should leave W2 code generation essentially unchanged while targeting W13's 32 iterations. Test an opt-in eight-way pragma with strict cubin-resource and M128 paired gates; reject if W13 code/register growth erases the instruction saving.
- Evidence: `results/iter131b_selected_unroll4_tp4_m128_route_gemms_ncu_human_20260903.log`, `results/iter131c_selected_unroll4_tp4_m128_route_gemms_ncu_compact_metrics_20260903.csv`, and `results/iter131d_prior_unroll2_tp4_m128_w2_ncu_compact_metrics_20260903.csv`.

## Iteration 132 — eight-way K128 unroll passes TP4/TP8 correctness

- Added opt-in `V4_ROUTE_K_UNROLL8=1` above the selected four-way branch and propagated it through extension hashing, compile flags, correctness/graph metadata, local profiling, and the same-process comparison harness. The production default remains four-way.
- The hypothesis is deliberately asymmetric: W2 has only four K128 iterations and is already fully expanded by the selected pragma, whereas W13 has 32 iterations and can still shed loop/stage/index work in groups of eight.
- TP4 balanced auto split4 passes with W13/activation/W2 rel-L2 0.000076187/0.000694956/0.002342691; TP4 skew forced split2 passes with 0.000077291/0.000838720/0.002351904; TP8-shape balanced auto split4 passes with 0.000076998/0.000714756/0.002333323.
- Route preparation and activation quantization are exact, all outputs are finite, and every numerical metric exactly matches the selected four-way path.
- Result: eight-way unrolling is numerically safe. Inspect cubin registers/stack and then require same-process M128 cold-L2 A/B before any wider timing sweep.
- Evidence: `results/iter132_route_k_unroll8_correctness_20260903.log`.

## Iteration 132b — eight-way unroll gives a small stable M128 gain

- Cubin inspection shows W2 remains unchanged at 61 registers and zero stack/local allocation. W13 split2, used by M128, rises modestly from 52 to 54 registers; W13 split4 rises more sharply from 52 to 60, so the small-M gate remains essential.
- Same-process TP4 M128 random-route timing used eight x 100 rank-max cold-L2 samples per arm and per-replay alternating A/B then B/A. Selected four-way control min/median/max is 0.300384/0.347424/0.474688 ms; eight-way candidate is 0.300352/0.344816/0.383840 ms.
- Control/candidate is 1.007563x, a 2.608 us median reduction, and the candidate wins all eight batch medians. Outputs are bitwise identical on all ranks.
- Result: eight-way unrolling passes the M128 gate, but the gain is too small to offset an unmeasured split4 regression. Screen M8/M16/M32 and M64 before any formal benchmark or default change.
- Evidence: `results/iter132b_route_k_unroll8_tp4_m128_paired_cold800_20260903.log`.

## Iteration 132c — global eight-way unroll loses on split4 and is rejected

- Same-process TP4 random-route screening used five x 100 rank-max cold-L2 samples per arm, with per-replay A/B then B/A alternation. Four-way-control/eight-way-candidate medians and control/candidate speedups: M8 0.071488/0.071968 ms (0.993330x), M16 0.112480/0.113072 ms (0.994764x), M32 0.181856/0.183120 ms (0.993097x), and M64 0.264480/0.262736 ms (1.006638x).
- Candidate min/median/max is 0.070624/0.071968/0.074080, 0.111840/0.113072/0.115264, 0.174944/0.183120/0.192640, and 0.243520/0.262736/0.272704 ms for M8 through M64. Control min/median/max is 0.070240/0.071488/0.207232, 0.111200/0.112480/0.251648, 0.173824/0.181856/0.522272, and 0.244096/0.264480/0.454432 ms.
- Outputs are bitwise identical on all ranks. Together with M128's 1.007563x gain, the five-shape geometric mean is only about 0.9991x: M8/M16/M32 all regress while the split2 M64/M128 points improve.
- Result: reject global eight-way unrolling and keep the production default at four-way. Refine the candidate to use pragma-eight only for `SplitK <= 2` while retaining pragma-four for split4; this follows the measured split-policy boundary and should preserve small-M code generation.
- Evidence: `results/iter132c_route_k_unroll8_tp4_m8_m64_paired_cold2000_20260903.log`.

## Iteration 133 — split-aware eight-way K unroll passes correctness

- Added opt-in `V4_ROUTE_K_UNROLL8_SPLIT2=1` without changing the rejected global-eight flag. The pragma factor is the template constant expression `SplitK <= 2 ? 8 : 4`: split1/split2 kernels receive eight-way unrolling, while split4 kernels retain the selected four-way code path.
- Propagated the new flag through extension hashing/compile flags, graph and correctness metadata, local profiling, and the same-process comparison harness. The production default remains pure four-way.
- TP4 balanced auto split4, TP4 skew forced split2, and TP8-shape balanced auto split4 all exactly reproduce the selected W13/activation/W2 numerical metrics. Route preparation and quantization are exact and all outputs are finite.
- Result: split-aware unrolling is numerically safe and NVCC accepts the template-dependent pragma. Verify that cubin resources match selected for split4 and global-eight for split2, then run M128 and all-M cold-L2 paired gates.
- Evidence: `results/iter133_splitaware_route_k_unroll8_correctness_20260903.log`.

## Iteration 133b — split-aware eight-way still perturbs W2 and is rejected

- Cubin resources validate the intended W13 split specialization: split4 is restored to the selected 52 registers, while split2 remains at the global-eight 54 registers; W2 remains at 61 registers. All kernels are spill-free.
- Same-process TP4 all-M timing used five x 100 rank-max cold-L2 samples per arm with per-replay AB/BA. Four-way-control/candidate medians and speedups: M8 0.071104/0.071264 ms (0.997755x), M16 0.112000/0.112256 ms (0.997720x), M32 0.182304/0.182240 ms (1.000351x), M64 0.265328/0.263712 ms (1.006128x), M128 0.338864/0.336240 ms (1.007804x).
- Candidate min/median/max is 0.070016/0.071264/0.073120, 0.110976/0.112256/0.114272, 0.174144/0.182240/0.186368, 0.243488/0.263712/0.276352, and 0.300256/0.336240/0.387104 ms. Control is 0.069888/0.071104/0.488000, 0.110816/0.112000/0.238272, 0.174336/0.182304/0.337504, 0.243968/0.265328/0.450976, and 0.299968/0.338864/0.495936 ms.
- All outputs are bitwise identical. Five-shape geometric-mean speedup is only about 1.00194x, and M8/M16 consistently regress by about 0.23%. Because W2 has `SplitK=1`, the split-aware expression also applies pragma-eight to W2 even though its loop has only four iterations; unchanged register count does not imply identical scheduling/code layout.
- Result: reject this general split-aware candidate. Refine once more to pragma-eight only when `K == 4096 && SplitK <= 2`, keeping both W2 and W13 split4 on the exact selected pragma-four path.
- Evidence: `results/iter133b_splitaware_route_k_unroll8_tp4_allm_paired_cold2500_20260903.log`.

## Iteration 134 — W13-only split1/split2 eight-way unroll passes correctness

- Added opt-in `V4_W13_K_UNROLL8_SPLIT2=1` with pragma factor `K == 4096 && SplitK <= 2 ? 8 : 4`. This excludes W2 (`K=512`) as well as W13 split4 from the eight-way path while preserving the previous global and general-split flags for reproducibility.
- Propagated the flag through extension configuration, compile flags, all benchmark/correctness metadata, comparison validation, and local profiling. The production default remains four-way.
- TP4 balanced auto split4, TP4 skew forced split2, and TP8-shape balanced auto split4 exactly reproduce selected W13/activation/W2 errors; route preparation and input quantization are exact and all outputs are finite.
- Result: the W13-only specialization is numerically safe. Verify cubin identity for W2/split4 and then use an all-M cold-L2 paired screen to test whether it retains M64/M128 gains without M8/M16 regressions.
- Evidence: `results/iter134_w13_split2_route_k_unroll8_correctness_20260903.log`.

## Iteration 134b — W13-only candidate gains at split2; small-M difference needs disambiguation

- Cubin resources match the intended specialization: W2 stays at 61 registers, W13 split4 stays at 52, W13 split2 moves to 54, and all remain stack/local-spill free.
- Same-process TP4 all-M screening used five x 100 rank-max cold-L2 samples per arm with per-replay AB/BA. Four-way-control/candidate medians and speedups: M8 0.071232/0.071360 ms (0.998206x), M16 0.112128/0.112416 ms (0.997438x), M32 0.181872/0.181760 ms (1.000616x), M64 0.265584/0.263824 ms (1.006671x), and M128 0.335664/0.333792 ms (1.005608x).
- Candidate min/median/max is 0.070112/0.071360/0.074048, 0.111168/0.112416/0.114624, 0.173888/0.181760/0.188032, 0.243264/0.263824/0.271168, and 0.299168/0.333792/1.034176 ms. Control is 0.069760/0.071232/0.399008, 0.110880/0.112128/0.264576, 0.173568/0.181872/0.358496, 0.243904/0.265584/0.392128, and 0.300256/0.335664/0.497568 ms.
- All outputs are bitwise identical. The five-shape geometric-mean speedup is about 1.00170x; M64/M128 improve, M32 is neutral, while M8/M16 differ by only -0.18%/-0.26% even though their executed W13 split4 and W2 paths should compile with factor four.
- Result: neither select nor reject yet. Compare per-function SASS between control and candidate, then run a longer 10 x 200 self-control. Treat the tiny small-M shift as a real regression only if machine code or the long run corroborates it.
- Evidence: `results/iter134b_w13_split2_route_k_unroll8_tp4_allm_paired_cold2500_20260903.log`.

## Iteration 134c — long self-control and byte-identical small-M kernels qualify formal testing

- Repeated all five shapes with ten x 200 rank-max cold-L2 samples per arm and per-replay AB/BA. Four-way-control/candidate medians and speedups: M8 0.070960/0.071072 ms (0.998424x), M16 0.112000/0.112224 ms (0.998004x), M32 0.185312/0.185504 ms (0.998965x), M64 0.270208/0.268576 ms (1.006077x), and M128 0.351872/0.349360 ms (1.007190x).
- Candidate min/median/max is 0.069536/0.071072/0.073312, 0.110784/0.112224/0.115104, 0.173728/0.185504/0.285312, 0.243360/0.268576/0.405632, and 0.299744/0.349360/0.498912 ms. Control is 0.069472/0.070960/0.213984, 0.110496/0.112000/0.273472, 0.173760/0.185312/0.325440, 0.243584/0.270208/0.408768, and 0.300608/0.351872/0.503840 ms.
- The five-shape geometric-mean speedup is about 1.00172x and all outputs are bitwise identical. M64/M128 gains reproduce; the apparent 0.10-0.20% M8/M16/M32 losses also reproduce.
- Extracted every cubin text section from control and candidate. All sections are byte-identical except W13 K=4096 split1/split2 for the TP4 N=1024 and TP8 N=512 instantiations. In particular, W2 has identical SHA-256 `4ae7459b...b9f94` and W13 split4 has identical `722d3ef5...d8490`. Since M8/M16/M32 use split4, their entire executed custom pipeline has identical extension kernel text; the tiny paired delta is a two-module placement/measurement artifact, not changed instructions.
- Result: qualify the W13-only specialization for a formal exact-Humming-plus-CARv2 run. Selection still requires aggregate improvement and no material per-shape loss in that production comparison.
- Evidence: `results/iter134c_w13_split2_route_k_unroll8_tp4_allm_paired_cold10000_20260903.log` and `results/iter134d_w13_split2_route_k_unroll8_cubin_text_diff_20260903.log`.

## Iteration 134e — W13-only candidate misses the first formal aggregate gate

- Ran exact Humming MXFP4 plus the same CARv2 instance against the W13-only candidate for ten x 200 rank-max cold-L2 samples per implementation and M, using the standard batch-level AB/BA order and random routes.
- Humming/custom medians and speedups: M8 0.090144/0.071200 ms (1.266067x), M16 0.145888/0.113568 ms (1.284587x), M32 0.231872/0.200384 ms (1.157138x), M64 0.335920/0.285392 ms (1.177048x), and M128 0.405568/0.373952 ms (1.084546x).
- Candidate min/median/max is 0.069952/0.071200/0.201280, 0.112256/0.113568/0.157920, 0.176320/0.200384/0.240128, 0.246560/0.285392/0.360032, and 0.310176/0.373952/0.410240 ms. Humming is 0.088288/0.090144/0.151072, 0.144288/0.145888/0.190368, 0.223488/0.231872/0.292992, 0.313536/0.335920/0.393472, and 0.378752/0.405568/2.651200 ms.
- Five-shape geometric-mean latency is 0.210717622 ms for Humming and 0.176836266 ms for custom, giving only 1.191597x. Correctness and all-reduce checks pass for every M with the same maximum custom rel-L2 0.002974846 and all outputs finite.
- This conflicts with the long same-process 1.00172x candidate gain because both sides drifted relative to the selected-default repeat (Humming improves from 0.211990 to 0.210718 ms while custom worsens from 0.175042 to 0.176836 ms), a combined swing far larger than the candidate's expected 0.17% aggregate effect.
- Result: the formal gate is not passed; do not select this candidate. Check contemporaneous GPU interference and permit one controlled repeat, but do not cherry-pick the faster of noisy runs. The production default remains four-way and its two independently passing results remain the headline.
- Evidence: `results/iter134e_w13_split2_unroll8_exact_humming_tp4_allm_cold2000_20260903.log`.

## Iteration 134f — second formal miss rejects W13-only eight-way selection

- Repeated the exact Humming MXFP4 plus same-CARv2 benchmark after confirming GPUs 1-4 were idle: ten x 200 rank-max cold-L2 samples per implementation and M, standard batch AB/BA pairing, same fixed random routes.
- Humming/custom medians and speedups: M8 0.090048/0.070976 ms (1.268711x), M16 0.145952/0.113536 ms (1.285513x), M32 0.232720/0.200256 ms (1.162112x), M64 0.332016/0.286448 ms (1.159079x), and M128 0.409824/0.370192 ms (1.107058x).
- Candidate min/median/max is 0.069760/0.070976/0.122016, 0.112384/0.113536/0.143616, 0.176096/0.200256/0.255456, 0.245920/0.286448/0.367008, and 0.310912/0.370192/0.493728 ms. Humming is 0.088256/0.090048/0.159872, 0.144192/0.145952/0.192544, 0.224032/0.232720/0.286560, 0.313344/0.332016/0.421024, and 0.379264/0.409824/0.526432 ms.
- Geometric-mean latency is 0.210792353 ms for Humming and 0.176465859 ms for custom, giving 1.194522x. Correctness and all-reduce checks again pass at every M.
- Result: reject W13-only eight-way unrolling for production selection. Its small 1.00172x self-control gain is real for split2 but is too weak to survive whole-baseline variance, and two consecutive formal runs miss the 1.2x gate (1.191597x and 1.194522x). Retain the four-way default and its independently reproduced 1.202732x/1.211085x headline results.
- Evidence: `results/iter134f_w13_split2_unroll8_exact_humming_tp4_allm_repeat_cold2000_20260903.log`.

## Iteration 135 — W13 split1/split2 full unroll passes correctness

- Added opt-in `V4_W13_K_UNROLL16_SPLIT2=1` with factor `K == 4096 && SplitK <= 2 ? 16 : 4`. For the common W13 split2 case this fully expands all 16 local K128 iterations; W2 and W13 split4 remain on factor four. The default remains the selected four-way path.
- Propagated the candidate through extension hashing/flags, correctness and graph metadata, comparison validation, and profiling.
- TP4 balanced auto split4, TP4 skew forced split2, and TP8-shape balanced auto split4 exactly reproduce selected numerical metrics. Route and input-quantization checks are exact and all outputs are finite.
- Result: full unrolling is numerically safe. Inspect W13 split2 register count and resident-block limit before timing; if it crosses the nine-to-eight-CTA threshold, use one M128 paired gate and reject early on a loss.
- Evidence: `results/iter135_w13_split2_route_k_unroll16_correctness_20260903.log`.

## Iteration 135b — W13 split2 full unroll collapses occupancy and is rejected

- Cubin inspection finds that full factor-16 unrolling raises W13 split2 from 52 selected registers/thread (54 for factor eight) to 72, while W2 remains 61 and W13 split4 remains 52. Stack/local allocation stays zero, but 72 registers lowers the register-limited resident-CTA ceiling from nine to roughly seven.
- Same-process TP4 M128 random-route timing used eight x 100 rank-max cold-L2 samples per arm with per-replay AB/BA. Four-way control min/median/max is 0.301472/0.336544/0.457728 ms; factor-16 candidate is 0.317536/0.359024/0.402784 ms.
- Control/candidate is only 0.937386x: the candidate is 22.480 us or 6.68% slower and loses all eight batch medians. Outputs remain bitwise identical on all ranks.
- Result: reject W13 factor-16 unrolling immediately; do not spend GPU time on other M values or a formal Humming run. The residency cliff dominates any instruction saving, confirming factor eight is the practical ceiling and factor four remains the robust production default.
- Evidence: `results/iter135b_w13_split2_route_k_unroll16_tp4_m128_paired_cold800_20260903.log`.

## Optimization evidence refresh — Hopper barrier/control-path audit

- Refreshed `instructions.txt` with narrow, verbatim extracts from the Hopper WGMMA execution, warp-specialization, and PTX `mbarrier` references before changing kernel source.
- The barrier protocol constraints rule out speculative early refill or relaxed publication: a full stage requires both arrival and transaction counts to reach zero, parity must follow slot reuse, and the consumer must finish all shared-memory reads before release.
- The scheduling references identify L2-local tile grouping/raster order as the remaining independent lever, while the current profile and prior iterations already reject deeper staging, wider global unrolling, and unsafe barrier changes.
- Result: no production source or benchmark result changed. Use source-correlated SASS/profile evidence to decide whether the next candidate targets necessary barrier wait, address/control instructions, or route-tile locality.

## Iteration 136 — source-correlated NCU hotspot audit

- Added `analyze_ncu_hotspots.py` and applied it to the selected four-way TP4 M128 cold-L2 NCU report. The parser reports per-kernel not-issued stall totals and the hottest SASS PCs without collecting a new profile.
- W13 has 4,046 not-issued samples: barrier 1,272 (31.44%), wait 674 (16.66%), math 588 (14.53%), long-scoreboard 566 (13.99%), and short-scoreboard 342 (8.45%). W2 has 2,040: barrier 542 (26.57%), long-scoreboard 464 (22.75%), math 344 (16.86%), wait 302 (14.80%), and short-scoreboard 144 (7.06%).
- The dominant W13 barrier samples land on four repeated `BSSY` reconvergence sites (162, 87, 79, and 62 samples), while the hottest explicit `WARPGROUP.DEPBAR.LE` sites have only 11 samples. Therefore the aggregate `stall_barrier` label does not justify weakening `mbarrier` publication or adding stages; most sampled pressure is compiler control/reconvergence around expanded K bodies.
- The hottest W13 long-scoreboard sites are shared-memory metadata/activation publication stores and their dependent scale multiply, not the `mbarrier.try_wait` loop. Any next control-path experiment must preserve 52-register occupancy and target repeated divergent setup in the unrolled body rather than synchronization correctness.
- Result: prefer a narrow source/SASS mapping of the repeated `BSSY` regions, then either hoist/eliminate their invariant branch or abandon this direction in favor of route-tile locality. Do not repeat direct-barrier-address or deeper-stage experiments.
- Evidence: `results/iter136_selected_unroll4_tp4_m128_ncu_sass_hotspots_20260904.log`.

## Iteration 137 — predicated padded-activation loads compile and pass first correctness gate

- Added opt-in `V4_PREDICATED_PADDED_ACTIVATION=1`. Padded activation and activation-scale reads now use inline-PTX predicated global loads with zero-initialized destinations; scale publication uses a predicated shared store. Clamped row/slot addresses remain in bounds even when the load/store predicate is false. The production default remains unchanged.
- This preserves the old no-transaction zero-fill semantics while removing the two nested divergent C++ branches from each expanded K128 iteration. The flag is included in JIT identity, graph metadata, and the same-process comparison harness.
- TP4 balanced auto-split4 exactly reproduces the selected W13/activation/W2 rel-L2 values 0.000076187/0.000694956/0.002342691. TP4 skew forced-split2 likewise reproduces 0.000077291/0.000838720/0.002351904. Both K6 reductions are bitwise equal to SGLang and all outputs are finite.
- TP8-shape (Is=256) runs through and passes the existing cosine gate: W13/activation/W2 cosine is 0.999999997/0.999999745/0.999930755 and rel-L2 is 0.000076998/0.000714756/0.011769490. The W2 error is higher than the previously recorded selected-path value, so this candidate is not yet correctness-qualified despite passing the coarse harness threshold.
- Result: compile and basic TP4/TP8 execution pass. Before timing, require a same-input control/candidate output comparison (including TP8) and inspect cubin resources/SASS to confirm that the intended branches disappeared without register or semantic drift.
- Evidence: `results/iter137_predicated_padded_activation_tp4_tp8shape_correctness_20260904.log`.

## Iteration 137b — control confirms a TP8-shape correctness regression

- Re-ran the same M8 balanced Is=256 test at `V4_PREDICATED_PADDED_ACTIVATION=0`, on the same HEAD, seed, and GPU. The control reproduces the previously selected W2 cosine/rel-L2 0.999997278/0.002333323; W13 and activation are identical to the candidate run.
- The candidate's W2 cosine/rel-L2 0.999930755/0.011769490 is therefore a real TP8-shape regression, not an outdated reference value or run-to-run noise. The coarse `cos > 0.99` test was insufficient to qualify this transformation.
- Result: block all performance timing. Diagnose the predicated activation/scale load semantics with direct control/candidate tensors or narrow the experiment to W13 only; do not select the current all-route-GEMM flag.
- Evidence: `results/iter137b_control_tp8shape_correctness_20260904.log`.

## Iteration 137c — coherent predicated loads reduce but do not eliminate TP8 drift

- PTX documentation confirms `ld.global.nc` uses the non-coherent read-only cache, which is inappropriate for W2 consuming qactivation freshly produced by the preceding graph kernel. Replaced only the default predicated-load forms with ordinary coherent `ld.global`; cache-hinted variants were already ordinary global loads.
- The M8 balanced Is=256 candidate improves from W2 cosine/rel-L2 0.999930755/0.011769490 to 0.999994742/0.003243021, confirming that the non-coherent load was a real bug. It still does not reproduce the control's 0.999997278/0.002333323.
- W13 and activation remain at their exact prior metrics and K6 remains bitwise equal to SGLang, so the remaining drift is isolated to applying the branchless transformation in W2 or to its changed W2 code generation.
- Result: keep performance timing blocked. Restrict the experiment to `IsW13` so W2 compiles onto the selected path byte-for-byte, then require TP4/TP8 control equivalence before timing.
- Evidence: `results/iter137c_coherent_predicated_tp8shape_correctness_20260904.log`.
