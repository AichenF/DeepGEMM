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
