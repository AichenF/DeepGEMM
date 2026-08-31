# TP MXFP4 MegaMoE — design & conversion spec (sm90/H20, tpmoe branch)

## Goal
Tensor-parallel (Megatron-style) MXFP4 MegaMoE for small M on H20 (78 SMs), by
converting dev_m's fused MXFP4 EP megakernel to TP. Reuse dev_m's MXFP4 compute
(dequant + FC1/FC2 WGMMA mainloop + SwiGLU) + tpmoe infra (ptx/mma/math/comm/
sym_buffer). Final reduction = symmetric-buffer NVLink all-reduce.

## Parallelism (VALIDATED exact in torch; sum-of-shards == full FFN, maxdiff 1.7e-6)
- tp = 8 ranks. Weights sharded on intermediate I. Is = I/tp = 2048/8 = 256.
- W1=[gate;up] [E,2I,H] -> rank r: gate[r*Is:(r+1)*Is] ++ up[I+r*Is:...] = [E,2*Is=512,H]. FC1 col-parallel.
- W2 [E,H,I] -> rank r: W2[:,:,r*Is:(r+1)*Is] = [E,H,Is=256]. FC2 row-parallel.
- All ranks process ALL M tokens (x replicated, NO dispatch). Per token m, per selected expert e (weight w):
    gate_r = W1g[e]@x[m]; up_r = W1u[e]@x[m]; act_r = silu(gate_r)*up_r; y_r[m] += w*(W2r[e]@act_r)
  y_r is partial over the I-shard. Final: y = all_reduce_sum_r(y_r).

## Per-rank kernel shapes (vs EP dev_m)
| | TP (this) | EP dev_m |
|--|--|--|
| FC1 N (out) | 2*Is = 512 | 2I = 4096 |
| FC1 K (contract) | H = 6144 | 6144 |
| FC2 N (out) | H = 6144 | 6144 |
| FC2 K (contract) | Is = 256 | I = 2048 |
| FC1 fused W K-storage | 48 BK128 tiles * 80B = 3840 | 3840 |
| FC2 fused W K-storage | 2 BK128 tiles * 80B = 160 | 16*80=1280 |
- Wrapper constant changes: kIntermediateHidden -> Is=256; L1_SHAPE_N -> 512; L2_SHAPE_K -> 256; kNumSMs -> 78 (H20). FC2 K-loop only 2 blocks.

## Weight Python (step A DONE, validated shapes + lossless fuse)
Reuse EP transform verbatim on SHARDED weights: quantize_to_mxfp4(shard, gs=32) ->
mxfp4_scale_to_tile_major -> mxfp4_fuse_packed_with_scale_tile_major -> _braid_mxfp4_mode2_signs.
Shard func: gate=W1[:, :I][:, r*Is:(r+1)*Is]; up=W1[:, I:][:, r*Is:(r+1)*Is]; W1s=cat(gate,up); W2s=W2[:,:,r*Is:(r+1)*Is].
Fused layout is lossless (dequant raw == dequant fused, maxdiff 0). 80-byte BK128 rows: 64 FP4 + 8 E8M0(dup x2) + 8 pad.

## EP -> TP kernel conversion (the 3 changes; compute core UNCHANGED)
1. INPUT (replace dispatch): EP pulls tokens cross-rank into local pools by expert. TP: all M tokens are LOCAL;
   build per-expert token groups by a LOCAL permutation of the M*topk (token,expert) pairs (histogram+argsort).
   First cut: compute grouping host-side (torch argsort of topk_idx) and pass sorted token ids + expert offsets;
   feed x (replicated, fp8 per-token per-128) into the grouped GEMM. No symm-buffer token movement.
2. COMPUTE (KEEP): FC1(sharded N=512) -> SwiGLU(Is=256, requant fp8 per-128: 256/128=2 blocks) -> FC2(sharded K=256).
   Reuse dev_m MXFP4 dequant + WGMMA mainloop + SwiGLU epilogue. No global scales (MXFP4).
3. OUTPUT (replace combine with all-reduce): EP scatters each token's expert output to source rank + gathers.
   TP: scatter-add locally y_partial[token] += w * y_e[token] (weighted, over the token's topk experts);
   then symm-buffer NVLink all-reduce sum of y_partial[M,H] across tp ranks -> y[M,H].

## Correctness judging
kernel y vs mxfp4-DEQUANT reference (not fp32) -> target cosine ~0.9997 (like EP). fp32-ref cosine ~0.98 = intrinsic MXFP4 quant error.

## Env / gotchas (H20-GPU-06, worktree /home/xutingz/fac/DeepGEMM_tp on tpmoe)
kNumSMs=78 (not 132; grid-barrier co-residency). DG_JIT_CACHE_DIR=/tmp/dg_jit_h20 (home NFS quota full).
Test: unique MASTER_PORT/run, os._exit teardown, one case/process, small NMT vs GPU5-7 contention (etgong).
Validated scripts: /tmp/tp_moe_ref_check.py (sharding exact), /tmp/tp_moe_dist_test.py (8-rank pipeline), /tmp/tp_shape_check.py (transform shapes).
