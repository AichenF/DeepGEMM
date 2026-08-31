# HINTS — TP MXFP4 MegaMoE AKO loop

## Directives (from the user's prompt)
- **Target**: fused **TP** (tensor-parallel) MXFP4 MegaMoE, small M, H20 (sm90, 78 SMs). Not EP.
- **MXFP4 dequant must live in the kernel** (final version) — reuse `deep_gemm/include/deep_gemm/quantization/mxfp4_dequant.cuh`. Do NOT reward-hack by pre-dequantizing weights to bf16 outside the timed kernel in the final solution (baseline may, iterations must move it in-kernel).
- **Final reduction = symmetric-buffer NVLink all-reduce** (reuse tpmoe `comm/barrier.cuh` + sym_buffer), reached via iterations. Baseline may use `dist.all_reduce`.
- Reuse dev_m MXFP4 **compute** (dequant + FC1/FC2 WGMMA mainloop + SwiGLU) + tpmoe infra (ptx/mma/math/comm/sym_buffer). Build a NEW TP kernel (don't drag dev_m's EP scheduler/dispatch/combine).
- Commit to the **tpmoe** branch (this worktree), not a separate opt/ branch.

## TP scheme (validated exact in torch; see TP_MOE_DESIGN.md)
- Shard on intermediate I: Is = I/tp = 256 (tp=8). FC1 col-parallel N=2*Is=512, K=H=6144. FC2 row-parallel N=H=6144, K=Is=256.
- All ranks process ALL M tokens (x replicated, no dispatch). y = all_reduce_sum(per-rank partial FFN).

## Correctness
- Judge vs **mxfp4-dequant** golden (torch), NOT fp32. Target cosine ~0.9997 (fp32-ref cosine ~0.98 = intrinsic MXFP4 quant error). Gate: cosine >= 0.99 (loose) in the loop; tighten to 0.999 at the final verdict.

## Env / gotchas (H20-GPU-06)
- `kNumSMs=78` (NOT 132; grid-barrier co-residency deadlocks otherwise).
- `DG_JIT_CACHE_DIR=/tmp/dg_jit_h20` (home NFS quota EXCEEDED, errno 122).
- Distributed test teardown: unique `MASTER_PORT` per run + `os._exit(0)` (torch CUDASymmetricMemory destructor aborts otherwise); one case per process (SymmBuffer.destroy leaks views).
- GPUs 5-7 shared with user **etgong** (sglang DeepSeek-V4, ~78GB) — keep symm-buffer/NMT footprint small or ranks 5-7 OOM. Kernel barriers gate on slowest rank, so contention inflates latency; note it in ITERATIONS.md.

## Iteration signal
- Primary: `RUNTIME` (TP forward latency, lower better). `SPEEDUP` is vs a rough torch full-FFN ref (not apples-to-apples across sharding — track RUNTIME trend).
