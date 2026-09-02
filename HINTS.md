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

## 2026-09-02 DeepSeek-V4-Flash target (supersedes stale shape/baseline text above)
- Model shape is read from the local `DeepSeek-V4-Flash/config.json`: `H=4096`, routed `I=2048`, `E=256`, `topk=6`, SwiGLU.
- Optimize **TP4** first: `Is=512`, FC1/W13 `[E, 1024, 4096]`, FC2/W2 `[E, 4096, 512]` per rank. The same implementation must run correctly at **TP8**: `Is=256`, FC1 `[E, 512, 4096]`, FC2 `[E, 4096, 256]`.
- Required benchmark token counts are exactly `M in {8, 16, 32, 64, 128}`. Do not infer M or active experts from `G`; `G` is not a routing model.
- Inputs retain precomputed `topk_idx [M,6]` and `topk_weights [M,6]`, matching the DeepGEMM MegaMoE contract. Router/top-k generation and shared-expert compute are outside this benchmark.
- Pure TP has no EP token dispatch or EP return/combine communication. Every rank consumes the same X and routing metadata, owns every expert's I-shard, computes a local `[M,H]` weighted route sum, then performs exactly one TP all-reduce.
- The source implementation is the already-reviewed WGMMA work in this `DeepGEMM_tp` worktree (`step_e_lutg.py`, `step_e_fc2.py`, existing TP and symmetric-memory code). Extend it; do not restart the compute kernel from scratch and do not modify the dirty `/home/xutingz/fac/DeepGEMM` checkout.
- Apples-to-apples baseline is the SGLang Humming **MXFP4 indexed** path: route alignment, dynamic FP8-E4M3 group-128 input quantization, Humming MXFP4 W13, SwiGLU, dynamic FP8 group-128 requantization, Humming MXFP4 W2, local k=6 weighted sum, then SGLang `CustomAllReduceV2`.
- Formal performance verdict uses CUDA Graph replay only. Eager timings are diagnostic. Weight preprocessing, workspace allocation, JIT, router/top-k generation and graph capture are untimed.
- Primary score is TP4 max-rank latency for the full routed-expert pipeline plus `CustomAllReduceV2`, reported independently for all five M values and as their equal-weight geometric mean. TP8 is a required correctness/run-through target, not the primary tuning score.
- For every formal point, report at least five outer runs as min/median/max. Keep stage timings and a pre-quantized-X core timing only as diagnostics; neither may replace the full BF16-entry result.
- **Cold-L2 is mandatory for every performance benchmark from 2026-09-02 onward.** H20 reports a 60 MiB L2. Before each individually timed graph replay, use Triton's standard 256 MiB cache-clear buffer on the same CUDA stream; record the start event after the clear and the end event immediately after the replay so cache eviction is enforced but its cost is excluded. Apply the identical protocol to Humming and custom kernels. Earlier continuous-replay results are warm/steady-state diagnostics only and must never be used for a final speedup claim.
