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

## Log

### baseline — vectorized torch
- What: per-rank sharded FFN via mxfp4-dequant + einsum grouped matmul + SwiGLU + `dist.all_reduce`. MXFP4 dequant + compute in torch (to be CUDA-ized).
- Result: RUNTIME=48.93 ms, cosine=0.99999, REF=20.25 ms, SPEEDUP=0.41x. finite=True.
- Read: torch einsum gathers per-pair weights `gate_w[fe]` -> [M*topk, Is, H] materialization; re-dequants l1+l2 every call. Both are the obvious first targets. Real perf will come from a CUDA fused MXFP4 kernel; intermediate iters can cut torch waste first.
