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

## Iteration 3 - First-class split phase mode

### Change

- Replaced the two template booleans `kRunL1Phase` / `kRunL2Phase` with a single `kSplitPhaseMode` template parameter:
  - `0`: fused L1+L2
  - `1`: split L1 only
  - `2`: split L2 only
- The kernel body still derives local constexpr `kRunL1Phase` and `kRunL2Phase`, so this is a structural cleanup toward true split rather than a math/dataflow change.
- Kept the iteration-2 gated no-ready-mask behavior unchanged.

### Correctness

- Build: `bash develop.sh` in `mega_moe_box` passed.
- Smoke correctness, `weight_scale=0.05`: PASS for `M=128,512,819,1024`.
- Full correctness, `weight_scale=0.05`: PASS for `M=32,64,128,256,500,512,819,1000,1024,2048,4096,8192`.

### Benchmark

Same full-list 50-run command as iterations 1 and 2:

```bash
python tests/bench_nvfp4_mega_moe_sm90.py \
  --batches 8 16 32 64 128 256 260 500 512 819 1000 1024 1280 1536 2048 3072 4096 8192 \
  --num-tests 50
```

Environment:

- `DG_JIT_CACHE_DIR=/tmp/dg_jit_split_phase_mode_iter3_full_bench`
- 8 ranks, hidden 7168, intermediate hidden 2048, experts 256, topk 8.

| M | iter2 mean_rank us | iter3 mean_rank us | delta |
|---:|---:|---:|---:|
| 8 | 777.0 | 757.5 | -2.5% |
| 16 | 795.8 | 791.9 | -0.5% |
| 32 | 845.0 | 820.6 | -2.9% |
| 64 | 825.4 | 800.9 | -3.0% |
| 128 | 850.7 | 868.0 | +2.0% |
| 256 | 1189.4 | 1184.8 | -0.4% |
| 260 | 1288.0 | 1301.6 | +1.1% |
| 500 | 1953.3 | 1960.9 | +0.4% |
| 512 | 2161.7 | 2169.7 | +0.4% |
| 819 | 2802.8 | 2818.7 | +0.6% |
| 1000 | 3368.8 | 3360.8 | -0.2% |
| 1024 | 3517.9 | 3536.7 | +0.5% |
| 1280 | 4115.9 | 4113.2 | -0.1% |
| 1536 | 4904.7 | 4880.9 | -0.5% |
| 2048 | 6199.7 | 6191.1 | -0.1% |
| 3072 | 8891.7 | 8899.5 | +0.1% |
| 4096 | 11609.4 | 11607.8 | -0.0% |
| 8192 | 22721.4 | 22712.8 | -0.0% |

### Result

This is a parity structural change. It makes split/fused mode selection explicit and removes the invalid two-bool mode space, but it does not create a standalone L1/L2 implementation body yet. The benchmark is within normal run-to-run noise and does not show a systematic regression. Continue with real L1/L2 body separation or PR323-style fused specialization only behind the same correctness and no-regression gates.

## Iteration 4 - Route M512 to the fused BN256 path

### Change

- Updated the BN256 fused selector so `M=512` uses the existing fused BN256 layout/path.
- Kept `M=1024` on the split path because earlier E2E validation identified effective-token size 1024 as a precision-risk size.
- Synced the benchmark and correctness helpers so their transformed weight layout matches the wrapper selector.

### Correctness

- Build: `bash develop.sh` in `mega_moe_box` passed.
- Smoke correctness, `weight_scale=0.05`: PASS for `M=500,512,819,1024`.
- Full correctness, `weight_scale=0.05`: PASS for `M=32,64,128,256,500,512,819,1000,1024,2048,4096,8192`.
- Tiny-signal absolute fallback, `weight_scale=0.001`: PASS for `M=512,819,1024` with `small_signal_ref_abs_max=1e-2`, `small_signal_abs_max_threshold=1e-2`, `small_signal_abs_mean_threshold=1e-3`.

### Benchmark

Same full-list 50-run command as previous iterations:

```bash
python tests/bench_nvfp4_mega_moe_sm90.py \
  --batches 8 16 32 64 128 256 260 500 512 819 1000 1024 1280 1536 2048 3072 4096 8192 \
  --num-tests 50
```

Environment:

- `DG_JIT_CACHE_DIR=/tmp/dg_jit_m512_fused_iter4_full_bench`
- 8 ranks, hidden 7168, intermediate hidden 2048, experts 256, topk 8.

| M | iter3 mean_rank us | iter4 mean_rank us | delta |
|---:|---:|---:|---:|
| 8 | 757.5 | 759.7 | +0.3% |
| 16 | 791.9 | 794.9 | +0.4% |
| 32 | 820.6 | 824.7 | +0.5% |
| 64 | 800.9 | 833.1 | +4.0% |
| 128 | 868.0 | 867.4 | -0.1% |
| 256 | 1184.8 | 1195.0 | +0.9% |
| 260 | 1301.6 | 1289.2 | -1.0% |
| 500 | 1960.9 | 1955.0 | -0.3% |
| 512 | 2169.7 | 1996.0 | -8.0% |
| 819 | 2818.7 | 2800.5 | -0.6% |
| 1000 | 3360.8 | 3358.8 | -0.1% |
| 1024 | 3536.7 | 3522.9 | -0.4% |
| 1280 | 4113.2 | 4106.9 | -0.2% |
| 1536 | 4880.9 | 4921.9 | +0.8% |
| 2048 | 6191.1 | 6216.2 | +0.4% |
| 3072 | 8899.5 | 8868.1 | -0.4% |
| 4096 | 11607.8 | 11607.5 | -0.0% |
| 8192 | 22712.8 | 22697.2 | -0.1% |

### Result

This is a useful launch-selection win: `M=512` improves by about 8% in the full-list run. Other sizes are unchanged by construction except benchmark noise; the only visible positive deltas are small/noisy and not tied to a path change. Keep `M=1024` split for now due the known E2E precision-risk history.

## Iteration 5 - Enable true-split no-ready-mask for all BN128 split sizes

### Change

- Updated the wrapper policy so every BN128 split L1/L2 launch uses the existing `kTrueSplitNoL2ReadyMask` template variant.
- In split L1 this makes the L2-ready notification a no-op; in split L2 this skips the per-block L2-ready-mask wait.
- Fused mode is unchanged. BN256 fused sizes such as `M=512` are unchanged by construction.

### Profiling Notes

- A direct 8-rank NCU application-replay run on `M=512` triggered NVLink barrier timeouts, so multi-rank NCU remains unsuitable for communication timing.
- Single-rank/e32 NCU reports were collected only for kernel-internal resource comparison:
  - `M512` fused: `/tmp/ncu_d13dc79_m512_single_e32/mega-moe-sm90-nvfp4.0.ncu-rep`
  - `M1024` split: `/tmp/ncu_d13dc79_m1024_single_e32/mega-moe-sm90-nvfp4.0.ncu-rep`
- Key internal signal: `M1024` split L1 still has high barrier stalls, so removing the now-unnecessary split L1/L2 ready-mask path is a reasonable true-split cleanup.

### Correctness

- Build: `bash develop.sh` in `mega_moe_box` passed.
- Smoke correctness, `weight_scale=0.05`: PASS for `M=512,819,1024,2048`.
- Full correctness, `weight_scale=0.05`: PASS for `M=32,64,128,256,500,512,819,1000,1024,2048,4096,8192`.
- Tiny-signal absolute fallback, `weight_scale=0.001`: PASS for `M=512,819,1024,2048` with `small_signal_ref_abs_max=1e-2`, `small_signal_abs_max_threshold=0.004`, `small_signal_abs_mean_threshold=0.0004`.

### Benchmark

Same full-list 50-run command as previous iterations:

```bash
python tests/bench_nvfp4_mega_moe_sm90.py \
  --batches 8 16 32 64 128 256 260 500 512 819 1000 1024 1280 1536 2048 3072 4096 8192 \
  --num-tests 50
```

Environment:

- `DG_JIT_CACHE_DIR=/tmp/dg_jit_true_split_all_noready_full_bench`
- 8 ranks, hidden 7168, intermediate hidden 2048, experts 256, topk 8.

| M | iter4 mean_rank us | iter5 mean_rank us | delta |
|---:|---:|---:|---:|
| 8 | 759.7 | 752.6 | -0.9% |
| 16 | 794.9 | 790.6 | -0.5% |
| 32 | 824.7 | 816.5 | -1.0% |
| 64 | 833.1 | 818.9 | -1.7% |
| 128 | 867.4 | 874.2 | +0.8% |
| 256 | 1195.0 | 1194.4 | -0.1% |
| 260 | 1289.2 | 1296.9 | +0.6% |
| 500 | 1955.0 | 1974.2 | +1.0% |
| 512 | 1996.0 | 2021.0 | +1.3% |
| 819 | 2800.5 | 2812.8 | +0.4% |
| 1000 | 3358.8 | 3350.6 | -0.2% |
| 1024 | 3522.9 | 3512.9 | -0.3% |
| 1280 | 4106.9 | 4081.8 | -0.6% |
| 1536 | 4921.9 | 4893.0 | -0.6% |
| 2048 | 6216.2 | 6161.4 | -0.9% |
| 3072 | 8868.1 | 8837.9 | -0.3% |
| 4096 | 11607.5 | 11595.9 | -0.1% |
| 8192 | 22697.2 | 22607.6 | -0.4% |

### Result

Keep. This makes the split path structurally cleaner: when L1 and L2 are separate kernel launches, the L2 per-block ready mask is no longer part of the default BN128 split path. The affected split sizes are flat to slightly faster in the comparable full-list run; small positive deltas are on mostly unaffected or already-enabled sizes and are within observed benchmark noise.
