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

## Log

### baseline — vectorized torch
- What: per-rank sharded FFN via mxfp4-dequant + einsum grouped matmul + SwiGLU + `dist.all_reduce`. MXFP4 dequant + compute in torch (to be CUDA-ized).
- Result: RUNTIME=48.93 ms, cosine=0.99999, REF=20.25 ms, SPEEDUP=0.41x. finite=True.
- Read: torch einsum gathers per-pair weights `gate_w[fe]` -> [M*topk, Is, H] materialization; re-dequants l1+l2 every call. Both are the obvious first targets. Real perf will come from a CUDA fused MXFP4 kernel; intermediate iters can cut torch waste first.
