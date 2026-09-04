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

## 2026-09-03 approved interleaved-TP design (from the user's `ok` reply)
- Pursue a TP-local persistent/interleaved MegaMoE path adapted from the read-only DeepGEMM SM90 MegaMoE scheduler/body; do not implement a static-route shortcut.
- Preserve the public numerical boundaries: W13 emits BF16, SwiGLU emits BF16, the W2 input is dynamically FP8-E4M3 group-128 quantized, W2 emits BF16 route rows, and `topk_weight * 1.5` is applied only in the local k=6 reduction. Do not move route weights into W2 activation scales merely for speed.
- Route IDs and weights are graph inputs and may change between replays. Scheduler bounds/readiness must be produced on device without host inspection of the captured route distribution.
- Reuse local replicated X/routes; remove the original EP dispatch, remote expert ownership, return/combine, and EP barriers. The only inter-rank operation is the final TP reduction.
- TP4 is the optimized specialization. TP8 must remain correct and runnable and may initially use the selected separate-kernel fallback.
- For TP4 M=8/16/32, the bitwise-equivalent multicast fused-k6/one-shot-push path is an accepted component after iteration 73. M=64/128 retain stock SGLang CARv2 unless stronger cold-L2 evidence replaces it.

## 2026-09-04 large-M collective audit (from the user's follow-up)
- Re-validate TP4 M=64/128 one-shot versus two-shot and NVLS multicast versus ordinary P2P with clean, same-process controls; do not infer the answer from earlier mixed W2-overlap experiments.
- Separate AR-only transport timing from the full current TP-MoE CUDA Graph.  AR-only inputs must be random, nonzero and restored before every replay; restore first, then clear 256 MiB L2, and exclude both restore and cache clear from CUDA-event timing.
- Include ordinary-P2P 1-shot push, ordinary-P2P 1-shot pull, ordinary-P2P 2-shot pull, NVLS multicast 1-shot push, and direct-symmetric-memory NVLS 2-shot pull whenever supported.  Keep identical communicator, tensor shape, input values and graph protocol.
- Report TP4 max-rank min/median/max plus per-batch medians from a balanced-order 10x200 cold-L2 window.  Treat pooled sub-percent wins as unselected if paired batches are directionally mixed.
- Do not change the production M<=32 multicast dispatch boundary unless the clean long-window evidence is repeatable and materially faster.

## 2026-09-04 single-launch TP MegaMoE objective (supersedes the earlier preparation/tail split)
- The deliverable is one CUDA kernel launch **per TP rank** for the complete routed-expert layer.  A CUDA Graph containing several kernels does not satisfy this requirement.
- The single kernel accepts BF16 `X`, precomputed `topk_idx/topk_weights`, MXFP4 weights/scales and graph-stable symmetric-memory pointers, and performs device route preparation, BF16-to-FP8 group-128 quantization, W13, SwiGLU plus FP8 requantization, W2, ordered weighted k=6 reduction, TP all-reduce and replay-state cleanup before returning.
- Router logits/top-k selection, weight preprocessing, allocation, JIT and graph capture remain outside the kernel/timed interval.  There is no EP dispatch/combine.
- Use branch `megamoe_nvfp4_dev_m` only as a read-only reference for its one-launch persistent scheduler, W13/W2 dependency pipeline and in-kernel communication structure.  Do not copy its EP ownership semantics or NVFP4 numerical path, and do not modify or checkout the dirty `/lustre/raplab/client/xutingz/fac/DeepGEMM` worktree.
- TP4 uses exactly 78 persistent CTAs and is the performance target.  TP8 must have a correct one-launch specialization/run-through; it is not the primary performance score.
- The comparison baseline is the currently selected custom TP implementation, which launches five kernels at M=8/16/32 and six at M=64/128.  Success requires at least `1.10x` equal-weight geometric-mean speedup over M=8/16/32/64/128 in a same-process, TP-rank-max, CUDA-Graph, 10x200 independently cold-L2 comparison.  Report every M and do not hide regressions behind the aggregate.
- Nsight verification must show exactly one timed kernel node per replay after excluding the separate 256 MiB cold-L2 clear.  No hidden timed memset, copy, reset or child-kernel launch is allowed.
