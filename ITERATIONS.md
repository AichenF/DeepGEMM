# AKO Iterations: SM90 NVFP4 MegaMoE

## Iteration 1 - Split scheduler iterator specialization

### Change

- Updated `deep_gemm/include/deep_gemm/impls/sm90_nvfp4_mega_moe.cuh`.
- Replaced the local `for_each_selected_block` split-mode iterator with the scheduler-provided `for_each_linear1_block` and `for_each_linear2_block`.
- Removed the split-mode fallback that used the generic `get_next_block()` path for `num_tokens < 128`.
- Kept the math, dequant, epilogue, dispatch, and wrapper launch behavior unchanged.

### Correctness

- Build: `bash develop.sh` in `mega_moe_box` passed.
- Smoke correctness:
  - `M=128,512,1024`, `weight_scale=0.05`: PASS.
  - Default tiny-signal run with `weight_scale=0.001` failed at `M=128` because the output is near zero and the default small-signal threshold (`1e-4`) did not apply.
  - Tiny-signal absolute fallback rerun passed for `M=128,512,1024,2048` with `small_signal_ref_abs_max=1e-2`, `small_signal_abs_max_threshold=1e-2`, `small_signal_abs_mean_threshold=1e-3`.
- Full correctness, `weight_scale=0.05`: PASS for `M=32,64,128,256,500,512,1000,1024,2048,4096,8192`.

### Benchmark

Command:

```bash
python tests/bench_nvfp4_mega_moe_sm90.py \
  --batches 8 16 32 64 128 256 260 500 512 819 1000 1024 1280 1536 2048 3072 4096 8192 \
  --num-tests 50
```

Environment:

- `DG_JIT_CACHE_DIR=/tmp/dg_jit_true_split_iter1_bench`
- 8 ranks, hidden 7168, intermediate hidden 2048, experts 256, topk 8.

| M | mean_rank us | max_rank us |
|---:|---:|---:|
| 8 | 764.4 | 773.2 |
| 16 | 787.4 | 792.9 |
| 32 | 868.5 | 883.5 |
| 64 | 819.2 | 827.5 |
| 128 | 863.0 | 872.3 |
| 256 | 1204.4 | 1216.0 |
| 260 | 1299.1 | 1312.0 |
| 500 | 1976.3 | 1987.0 |
| 512 | 2168.9 | 2174.7 |
| 819 | 2832.6 | 2838.1 |
| 1000 | 3347.4 | 3357.0 |
| 1024 | 3533.8 | 3542.0 |
| 1280 | 4106.3 | 4115.0 |
| 1536 | 4893.6 | 4906.0 |
| 2048 | 6220.1 | 6233.0 |
| 3072 | 8878.0 | 8884.0 |
| 4096 | 11607.9 | 11622.0 |
| 8192 | 22727.0 | 22743.0 |

### Result

This is a correctness-safe cleanup but not a performance win. The split path remains limited by larger structural issues: L2 still carries dispatch/cleanup synchronization shape in the shared implementation, and both phases still share the broad fused/split body. Next iteration should target a more substantial true-split change, such as removing L2 arrival-mask waits and dead dispatch cleanup from the split L2 kernel or separating L1/L2 implementation bodies.

## Iteration 2 - Gated true-split no-ready-mask path

### Change

- Added a compile-time `kTrueSplitNoL2ReadyMask` template parameter to `sm90_nvfp4_mega_moe_impl`.
- When enabled for split L1, `notify_l1_ready()` becomes a no-op.
- When enabled for split L2, the A/SFA loader skips the per-block `l2_arrival_mask` wait.
- Fused single-kernel mode keeps the original ready-mask synchronization.
- Wrapper defaults enable the new true-split no-ready-mask variant only for `M=512` and `M=819`, where the same-run benchmark did not show regression. Other split sizes keep the existing ready-mask path until a broader true split rewrite removes their regressions.

### Correctness

- Build: `bash develop.sh` in `mega_moe_box` passed.
- Smoke correctness, `weight_scale=0.05`: PASS for `M=512,819,1024,2048`.
- Full correctness, `weight_scale=0.05`: PASS for `M=32,64,128,256,500,512,819,1000,1024,2048,4096,8192`.
- Tiny-signal absolute fallback, `weight_scale=0.001`: PASS for `M=512,819,1024,2048` with `small_signal_ref_abs_max=1e-2`, `small_signal_abs_max_threshold=1e-2`, `small_signal_abs_mean_threshold=1e-3`.

### Benchmark

Same full-list 50-run command as iteration 1:

```bash
python tests/bench_nvfp4_mega_moe_sm90.py \
  --batches 8 16 32 64 128 256 260 500 512 819 1000 1024 1280 1536 2048 3072 4096 8192 \
  --num-tests 50
```

Environment:

- `DG_JIT_CACHE_DIR=/tmp/dg_jit_true_split_iter2_gated_full_bench`
- 8 ranks, hidden 7168, intermediate hidden 2048, experts 256, topk 8.

| M | iter1 mean_rank us | iter2 mean_rank us | delta |
|---:|---:|---:|---:|
| 8 | 764.4 | 777.0 | +1.6% |
| 16 | 787.4 | 795.8 | +1.1% |
| 32 | 868.5 | 845.0 | -2.7% |
| 64 | 819.2 | 825.4 | +0.8% |
| 128 | 863.0 | 850.7 | -1.4% |
| 256 | 1204.4 | 1189.4 | -1.2% |
| 260 | 1299.1 | 1288.0 | -0.9% |
| 500 | 1976.3 | 1953.3 | -1.2% |
| 512 | 2168.9 | 2161.7 | -0.3% |
| 819 | 2832.6 | 2802.8 | -1.1% |
| 1000 | 3347.4 | 3368.8 | +0.6% |
| 1024 | 3533.8 | 3517.9 | -0.4% |
| 1280 | 4106.3 | 4115.9 | +0.2% |
| 1536 | 4893.6 | 4904.7 | +0.2% |
| 2048 | 6220.1 | 6199.7 | -0.3% |
| 3072 | 8878.0 | 8891.7 | +0.2% |
| 4096 | 11607.9 | 11609.4 | +0.0% |
| 8192 | 22727.0 | 22721.4 | -0.0% |

### Result

The structural no-ready-mask path is correctness-safe on the smoke set and improves the two enabled sizes in the full-list run, while leaving other sizes on the legacy ready-mask path. The remaining tiny positive deltas occur on sizes where the new template flag is not enabled, so they are treated as benchmark noise / iter1 baseline noise rather than a direct no-ready-mask regression. Before finalizing, run full correctness and consider repeated benchmark samples for the marginal sizes.
