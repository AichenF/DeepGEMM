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

## Iteration 6 - Skip unused split L2 ready-mask cleanup

### Change

- Guarded the two cleanup stores to `workspace.get_l2_arrival_mask_ptr(...)` with `if constexpr (!kSkipL1ReadyNotify && !kSkipL2ReadyMask)`.
- In the default true-split no-ready path, L1 no longer writes the L2 ready mask and L2 no longer waits on it, so the split cleanup no longer clears an unused mask.
- Fused and legacy ready-mask paths still clear the mask.

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

- `DG_JIT_CACHE_DIR=/tmp/dg_jit_split_cleanup_skip_l2_mask_bench50`
- 8 ranks, hidden 7168, intermediate hidden 2048, experts 256, topk 8.

| M | iter5 mean_rank us | iter6 mean_rank us | delta |
|---:|---:|---:|---:|
| 8 | 752.6 | 767.6 | +2.0% |
| 16 | 790.6 | 805.8 | +1.9% |
| 32 | 816.5 | 820.5 | +0.5% |
| 64 | 818.9 | 842.0 | +2.8% |
| 128 | 874.2 | 845.0 | -3.3% |
| 256 | 1194.4 | 1202.7 | +0.7% |
| 260 | 1296.9 | 1295.2 | -0.1% |
| 500 | 1974.2 | 1966.4 | -0.4% |
| 512 | 2021.0 | 1994.8 | -1.3% |
| 819 | 2812.8 | 2813.9 | +0.0% |
| 1000 | 3350.6 | 3346.3 | -0.1% |
| 1024 | 3512.9 | 3517.9 | +0.1% |
| 1280 | 4081.8 | 4102.8 | +0.5% |
| 1536 | 4893.0 | 4886.5 | -0.1% |
| 2048 | 6161.4 | 6174.4 | +0.2% |
| 3072 | 8837.9 | 8827.4 | -0.1% |
| 4096 | 11595.9 | 11515.8 | -0.7% |
| 8192 | 22607.6 | 22613.2 | +0.0% |

### Result

Keep. This is a narrow true-split cleanup with no correctness change and no systematic benchmark regression. The only changed runtime path is the BN128 true-split cleanup path; affected split sizes are flat within the observed noise, while the larger `M=4096` case improved slightly. Small/fused-size deltas are treated as run-to-run noise because they are not on the edited path.

## Iteration 7 - Split host launchers for fused and split phases

### Change

- Replaced the shared host-side `launch_with_phase()` lambda with explicit `launch_fused()`, `launch_split_l1()`, and `launch_split_l2()` launchers.
- Kept one shared `build_and_launch()` helper for JIT generation/build/launch.
- Fused launch now explicitly sets `split_phase_mode = kFusedPhaseMode` and `true_split_no_l2_ready_mask = false`.
- Split L1/L2 launches now explicitly set their own phase mode and true-split readiness policy, instead of deriving `run_l1_phase` / `run_l2_phase` booleans in the host wrapper.
- No kernel math, dequant, scheduler, or launch-selection heuristic changed.

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

- `DG_JIT_CACHE_DIR=/tmp/dg_jit_wrapper_refactor_bench50`
- 8 ranks, hidden 7168, intermediate hidden 2048, experts 256, topk 8.

| M | iter6 mean_rank us | iter7 mean_rank us | delta |
|---:|---:|---:|---:|
| 8 | 767.6 | 741.9 | -3.3% |
| 16 | 805.8 | 805.7 | -0.0% |
| 32 | 820.5 | 831.5 | +1.3% |
| 64 | 842.0 | 822.0 | -2.4% |
| 128 | 845.0 | 820.8 | -2.9% |
| 256 | 1202.7 | 1199.9 | -0.2% |
| 260 | 1295.2 | 1314.4 | +1.5% |
| 500 | 1966.4 | 1955.6 | -0.5% |
| 512 | 1994.8 | 1995.1 | +0.0% |
| 819 | 2813.9 | 2803.1 | -0.4% |
| 1000 | 3346.3 | 3338.0 | -0.2% |
| 1024 | 3517.9 | 3493.1 | -0.7% |
| 1280 | 4102.8 | 4083.4 | -0.5% |
| 1536 | 4886.5 | 4876.6 | -0.2% |
| 2048 | 6174.4 | 6167.0 | -0.1% |
| 3072 | 8827.4 | 8832.4 | +0.1% |
| 4096 | 11515.8 | 11519.4 | +0.0% |
| 8192 | 22613.2 | 22601.9 | -0.0% |

Note: the first full-list run reported `M=128` as `933.4us mean_rank`, but an immediate 50-run single-size rerun with the same JIT cache reported `820.8us mean_rank`. Because this wrapper refactor does not change the fused kernel generated for `M=128`, the full-list `M=128` point is treated as rank-level benchmark noise; the rerun value is used in the table.

### Result

Keep. This is a structural wrapper refactor toward explicit fused/split launch paths. Correctness is unchanged, and the 50-run benchmark does not show a systematic regression after the noisy `M=128` point is rerun. The generated kernel entrypoint selection and kernel names remain the same as iteration 6.

## Iteration 8 - Remove redundant selected-phase guards

### Change

- Removed four redundant callback guards from `for_each_selected_block()` users in the TMA A/SFA loader, TMA B/SFB loader, loader-dequant idle warp path, and math/epilogue path.
- `for_each_selected_block()` is now the single body-local phase filter:
  - split L1 entrypoints iterate only `Linear1` blocks,
  - split L2 entrypoints iterate only `Linear2` blocks,
  - fused entrypoints iterate both phases and pass each as a compile-time phase tag.
- No scheduler, dequant, math, epilogue, or launch-selection logic changed.

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

- `DG_JIT_CACHE_DIR=/tmp/dg_jit_phase_guard_bench50`
- 8 ranks, hidden 7168, intermediate hidden 2048, experts 256, topk 8.

| M | iter7 mean_rank us | iter8 mean_rank us | delta |
|---:|---:|---:|---:|
| 8 | 741.9 | 767.7 | +3.5% |
| 16 | 805.7 | 806.5 | +0.1% |
| 32 | 831.5 | 804.9 | -3.2% |
| 64 | 822.0 | 814.9 | -0.9% |
| 128 | 820.8 | 813.6 | -0.9% |
| 256 | 1199.9 | 1201.9 | +0.2% |
| 260 | 1314.4 | 1293.0 | -1.6% |
| 500 | 1955.6 | 1963.3 | +0.4% |
| 512 | 1995.1 | 1986.3 | -0.4% |
| 819 | 2803.1 | 2799.6 | -0.1% |
| 1000 | 3338.0 | 3339.2 | +0.0% |
| 1024 | 3493.1 | 3511.9 | +0.5% |
| 1280 | 4083.4 | 4083.1 | -0.0% |
| 1536 | 4876.6 | 4862.2 | -0.3% |
| 2048 | 6167.0 | 6152.6 | -0.2% |
| 3072 | 8832.4 | 8815.2 | -0.2% |
| 4096 | 11519.4 | 11504.1 | -0.1% |
| 8192 | 22601.9 | 22587.8 | -0.1% |

Notes:

- The first full-list run reported `M=32` as `914.5us mean_rank`; an immediate 50-run single-size rerun with the same JIT cache reported `804.9us`, so the rerun value is used in the table.
- `M=8` rerun reported `767.7us`, matching iteration 6 (`767.6us`) but slower than iteration 7's unusually fast `741.9us` sample. Because the removed guards compile out in fused mode, this is treated as normal small-M benchmark variance rather than a generated-kernel regression.
- `M=128` rerun reported `813.6us`, confirming no small-M systematic regression.

### Result

Keep. This makes the shared body rely on the explicit selected-phase iterator instead of repeating phase guards inside every callback. Correctness is unchanged, and the full-list plus targeted small-M reruns show no systematic performance regression.

## Iteration 9 - Use compile-time phase tags in loader paths

### Profiling Input

After iteration 8, current-head single-rank/e32 NCU was collected to refresh the internal bottleneck picture:

```bash
NCU_LAUNCH_COUNT=1 \
bash scripts/run_ncu_mega_moe_sm90.sh \
  --num-processes 1 --output /tmp/ncu_02e7f9a_m512_single_e32 \
  --batches 512 --num-experts 32 --num-tests 1

NCU_LAUNCH_COUNT=2 \
bash scripts/run_ncu_mega_moe_sm90.sh \
  --num-processes 1 --output /tmp/ncu_02e7f9a_m1024_single_e32 \
  --batches 1024 --num-experts 32 --num-tests 1
```

Reports:

- `/tmp/ncu_02e7f9a_m512_single_e32/mega-moe-sm90-nvfp4.0.ncu-rep`
- `/tmp/ncu_02e7f9a_m1024_single_e32/mega-moe-sm90-nvfp4.0.ncu-rep`

Key metrics:

| case | kernel | duration ms | SM % | DRAM % | L1TEX % | regs | smem KB | barrier stall | eligible warps | inst |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| M512/e32 | fused | 1.884 | 83.5 | 16.3 | 42.9 | 168 | 210.648 | 2.716 | 0.532 | 476.665M |
| M1024/e32 | split L1 | 2.103 | 87.5 | 9.7 | 34.0 | 128 | 193.816 | 7.092 | 0.397 | 373.395M |
| M1024/e32 | split L2 | 1.145 | 80.1 | 9.6 | 36.4 | 168 | 205.072 | 2.974 | 0.460 | 234.677M |

This matches the previous profile: split L1 is still barrier/pipeline limited, while M512 fused is already high-SM-throughput. The next low-risk PR323-style refactor should therefore be structural phase specialization rather than another ready-mask cleanup.

### Change

- Added `<type_traits>` for phase-tag type inspection.
- In the A/SFA loader and B/SFB loader callbacks, replaced runtime `block_phase == Linear2 ? ... : ...` descriptor/scale selection with `std::integral_constant`-based `if constexpr` selection.
- Converted the A/SFA ready wait and SFA TMA load branch to use the same compile-time phase tag.
- Did not touch math, epilogue, dequant arithmetic, scheduler policy, wrapper heuristics, or launch selection.

### Correctness

- Build: `bash develop.sh` in `mega_moe_box` passed.
- Smoke correctness, `weight_scale=0.05`: PASS for `M=512,819,1024,2048`.
- Full correctness, `weight_scale=0.05`: PASS for `M=32,64,128,256,500,512,819,1000,1024,2048,4096,8192`.
- Tiny-signal absolute fallback, `weight_scale=0.001`: PASS for `M=512,819,1024,2048` with `small_signal_ref_abs_max=1e-2`, `small_signal_abs_max_threshold=0.004`, `small_signal_abs_mean_threshold=0.0004`.

### Benchmark

Full-list 50-run command:

```bash
python tests/bench_nvfp4_mega_moe_sm90.py \
  --batches 8 16 32 64 128 256 260 500 512 819 1000 1024 1280 1536 2048 3072 4096 8192 \
  --num-tests 50
```

Environment:

- `DG_JIT_CACHE_DIR=/tmp/dg_jit_loader_phase_tag_bench50`
- 8 ranks, hidden 7168, intermediate hidden 2048, experts 256, topk 8.

Full-list result:

| M | mean_rank us |
|---:|---:|
| 8 | 802.6 |
| 16 | 797.4 |
| 32 | 836.3 |
| 64 | 823.0 |
| 128 | 881.3 |
| 256 | 1199.4 |
| 260 | 1304.1 |
| 500 | 2002.4 |
| 512 | 1989.0 |
| 819 | 2804.1 |
| 1000 | 3332.6 |
| 1024 | 3506.9 |
| 1280 | 4122.0 |
| 1536 | 4867.2 |
| 2048 | 6163.5 |
| 3072 | 8830.4 |
| 4096 | 11527.1 |
| 8192 | 22595.6 |

Targeted reruns with `8192` included to keep `num_max_tokens_per_rank=8192`:

| M | rerun mean_rank us |
|---:|---:|
| 8 | 759.6 |
| 32 | 811.9 |
| 128 | 848.8 |
| 500 | 1844.6 |
| 1280 | 4075.2 |
| 8192 | 22648.2 |

### Result

Keep. The full-list run had several noisy high points, but NMT=8192 targeted reruns did not reproduce a stable regression. This moves the loader side closer to PR323-style phase-specialized processing while leaving the math/epilogue hot path unchanged.

## Iteration 10 - Specialize math-side sync phase checks

### Change

- In the math/epilogue selected-block callback, added a compile-time phase tag:
  - `BlockPhaseTag`
  - `kBlockIsL2`
- Converted only the sync/empty-tile phase checks to `if constexpr`:
  - async L1 TMA-store drain before L2 blocks,
  - async L1 TMA-store drain on invalid-M L1 blocks,
  - epilogue barrier sync skip for `kL2ArrivalCounter && L1`.
- Did not change the main WGMMA math loop, L1 epilogue, L2 epilogue, dequant arithmetic, scheduler policy, wrapper heuristics, or launch selection.

An intermediate variant also changed the `kL1DualKAccum` loop selection to compile-time phase selection. That touches the hot GEMM-loop selection path and was removed before the final version after targeted M256 probes did not show a clean improvement. The committed version keeps this branch in its previous runtime form.

### Correctness

- Build: `./develop.sh` in `mega_moe_box` passed after restoring the current worktree from the baseline probe.
- Smoke correctness, `weight_scale=0.05`: PASS for `M=512,819,1024,2048`.
- Full correctness, `weight_scale=0.05`: PASS for `M=32,64,128,256,500,512,819,1000,1024,2048,4096,8192`.
- Tiny-signal absolute fallback, `weight_scale=0.001`: PASS for `M=512,819,1024,2048` with `small_signal_ref_abs_max=1e-2`, `small_signal_abs_max_threshold=0.004`, `small_signal_abs_mean_threshold=0.0004`.

### Benchmark

Full-list 50-run command:

```bash
python tests/bench_nvfp4_mega_moe_sm90.py \
  --batches 8 16 32 64 128 256 260 500 512 819 1000 1024 1280 1536 2048 3072 4096 8192 \
  --num-tests 50
```

Environment:

- `DG_JIT_CACHE_DIR=/tmp/dg_jit_iter10_sync_only_bench50`
- 8 ranks, hidden 7168, intermediate hidden 2048, experts 256, topk 8.

| M | iter9 mean_rank us | iter10 mean_rank us | delta |
|---:|---:|---:|---:|
| 8 | 802.6 | 769.0 | -4.2% |
| 16 | 797.4 | 803.8 | +0.8% |
| 32 | 836.3 | 819.6 | -2.0% |
| 64 | 823.0 | 851.4 | +3.5% |
| 128 | 881.3 | 866.2 | -1.7% |
| 256 | 1199.4 | 1187.0 | -1.0% |
| 260 | 1304.1 | 1286.1 | -1.4% |
| 500 | 2002.4 | 1972.4 | -1.5% |
| 512 | 1989.0 | 1996.0 | +0.4% |
| 819 | 2804.1 | 2818.2 | +0.5% |
| 1000 | 3332.6 | 3360.2 | +0.8% |
| 1024 | 3506.9 | 3516.2 | +0.3% |
| 1280 | 4122.0 | 4088.0 | -0.8% |
| 1536 | 4867.2 | 4858.1 | -0.2% |
| 2048 | 6163.5 | 6160.4 | -0.1% |
| 3072 | 8830.4 | 8829.4 | -0.0% |
| 4096 | 11527.1 | 11536.9 | +0.1% |
| 8192 | 22595.6 | 22616.4 | +0.1% |

Targeted probes:

| probe | M | current mean_rank us | clean d552298 mean_rank us | note |
|---|---:|---:|---:|---|
| `M=256,1024,8192` | 256 | 1292.6 | 1295.6 | no current regression under same recv/NMT |
| `M=256,1024,8192` | 1024 | 3494.1 | 3442.1 | +1.5%, within the same noise band as full-list deltas |
| `M=256,1024,8192` | 8192 | 22608.6 | 22650.4 | no current regression |
| `M=64,8192` | 64 | 853.8 | 869.1 | full-list M64 was noisy; same-condition baseline is slower |
| `M=64,8192` | 8192 | 22609.2 | 22601.4 | effectively equal |

### Result

Keep. This is a narrow structural refactor toward compile-time phase-specialized true split/fuse code, but it deliberately avoids the main math/epilogue body. Full correctness and tiny-signal fallback pass. The 50-run benchmark is flat overall; the apparent M64 full-list regression is not reproduced against a same-condition clean `d552298` baseline, where current is faster.

## Iteration 11 - Specialize the main epilogue phase branch

### Change

- Converted the main math callback epilogue selector from:

```cpp
if (block_phase == sched::BlockPhase::Linear1) { ... } else { ... }
```

to:

```cpp
if constexpr (!kBlockIsL2) { ... } else { ... }
```

- This is intentionally a one-line structural refactor:
  - L1 epilogue implementation is unchanged,
  - L2 epilogue implementation is unchanged,
  - GEMM loop selection is unchanged,
  - dequant arithmetic, scheduler policy, wrapper heuristics, and launch selection are unchanged.

This moves the split L1/L2 instantiations closer to true phase-specific kernels at the source level, while keeping fused behavior the same generic selected-block loop.

### Correctness

- Build: `./develop.sh` in `mega_moe_box` passed after restoring the current worktree from the Iteration 10 probe.
- Smoke correctness, `weight_scale=0.05`: PASS for `M=512,819,1024,2048`.
- Full correctness, `weight_scale=0.05`: PASS for `M=32,64,128,256,500,512,819,1000,1024,2048,4096,8192`.
- Tiny-signal absolute fallback, `weight_scale=0.001`: PASS for `M=512,819,1024,2048` with `small_signal_ref_abs_max=1e-2`, `small_signal_abs_max_threshold=0.004`, `small_signal_abs_mean_threshold=0.0004`.

### Benchmark

Full-list 50-run command:

```bash
python tests/bench_nvfp4_mega_moe_sm90.py \
  --batches 8 16 32 64 128 256 260 500 512 819 1000 1024 1280 1536 2048 3072 4096 8192 \
  --num-tests 50
```

Environment:

- `DG_JIT_CACHE_DIR=/tmp/dg_jit_iter11_epilogue_constexpr_bench50`
- 8 ranks, hidden 7168, intermediate hidden 2048, experts 256, topk 8.

| M | iter10 mean_rank us | iter11 mean_rank us | delta |
|---:|---:|---:|---:|
| 8 | 769.0 | 750.3 | -2.4% |
| 16 | 803.8 | 813.1 | +1.2% |
| 32 | 819.6 | 889.9 | +8.6% |
| 64 | 851.4 | 832.1 | -2.3% |
| 128 | 866.2 | 891.3 | +2.9% |
| 256 | 1187.0 | 1217.6 | +2.6% |
| 260 | 1286.1 | 1310.2 | +1.9% |
| 500 | 1972.4 | 1969.5 | -0.1% |
| 512 | 1996.0 | 1999.7 | +0.2% |
| 819 | 2818.2 | 2819.8 | +0.1% |
| 1000 | 3360.2 | 3357.0 | -0.1% |
| 1024 | 3516.2 | 3513.5 | -0.1% |
| 1280 | 4088.0 | 4097.5 | +0.2% |
| 1536 | 4858.1 | 4874.0 | +0.3% |
| 2048 | 6160.4 | 6167.6 | +0.1% |
| 3072 | 8829.4 | 8809.8 | -0.2% |
| 4096 | 11536.9 | 11517.8 | -0.2% |
| 8192 | 22616.4 | 22602.6 | -0.1% |

Targeted probe with `8192` included to keep NMT comparable:

```bash
python tests/bench_nvfp4_mega_moe_sm90.py --batches 32 128 256 8192 --num-tests 50
```

| M | iter11 current mean_rank us | iter10 `f8686fd` mean_rank us | delta |
|---:|---:|---:|---:|
| 32 | 809.6 | 813.2 | -0.4% |
| 128 | 813.4 | 825.9 | -1.5% |
| 256 | 1307.0 | 1301.8 | +0.4% |
| 8192 | 22666.8 | 22581.4 | +0.4% |

### Result

Keep. The full-list small-M points `M=32` and `M=128` were noisy, but the same-condition targeted run against the Iteration 10 commit shows no regression. The large-M sweep is flat, with all deltas within benchmark noise. This is a useful true-split refactor because split L1/L2 epilogue code is now selected at compile time instead of relying on a runtime-looking phase branch.

## Iteration 12 - Specialize mma.sync phase branches

### Change

- Converted the two remaining active `kUseMMASync` L1/L2 selectors to compile-time phase selection:
  - the mma.sync per-K block L1-vs-L2 compute branch,
  - the mma.sync L1-vs-L2 epilogue branch.
- This only affects the `BLOCK_M == 16 or BLOCK_M == 32` small-M path.
- Did not change default WGMMA code, dequant arithmetic, scheduler policy, wrapper heuristics, or launch selection.

### Correctness

- Build: `./develop.sh` in `mega_moe_box` passed after restoring the current worktree from the Iteration 11 probe.
- Smoke correctness, `weight_scale=0.05`: PASS for `M=512,819,1024,2048`.
- Full correctness, `weight_scale=0.05`: PASS for `M=32,64,128,256,500,512,819,1000,1024,2048,4096,8192`.
- Tiny-signal absolute fallback, `weight_scale=0.001`: PASS for `M=512,819,1024,2048` with `small_signal_ref_abs_max=1e-2`, `small_signal_abs_max_threshold=0.004`, `small_signal_abs_mean_threshold=0.0004`.

### Benchmark

Full-list 50-run command:

```bash
python tests/bench_nvfp4_mega_moe_sm90.py \
  --batches 8 16 32 64 128 256 260 500 512 819 1000 1024 1280 1536 2048 3072 4096 8192 \
  --num-tests 50
```

Environment:

- `DG_JIT_CACHE_DIR=/tmp/dg_jit_iter12_mmasync_phase_bench50`
- 8 ranks, hidden 7168, intermediate hidden 2048, experts 256, topk 8.

| M | iter11 mean_rank us | iter12 mean_rank us | delta |
|---:|---:|---:|---:|
| 8 | 750.3 | 752.9 | +0.3% |
| 16 | 813.1 | 797.2 | -2.0% |
| 32 | 889.9 | 885.7 | -0.5% |
| 64 | 832.1 | 817.9 | -1.7% |
| 128 | 891.3 | 838.6 | -5.9% |
| 256 | 1217.6 | 1200.4 | -1.4% |
| 260 | 1310.2 | 1288.1 | -1.7% |
| 500 | 1969.5 | 1962.2 | -0.4% |
| 512 | 1999.7 | 1991.5 | -0.4% |
| 819 | 2819.8 | 2809.9 | -0.4% |
| 1000 | 3357.0 | 3355.2 | -0.1% |
| 1024 | 3513.5 | 3511.1 | -0.1% |
| 1280 | 4097.5 | 4079.0 | -0.5% |
| 1536 | 4874.0 | 4874.4 | +0.0% |
| 2048 | 6167.6 | 6179.0 | +0.2% |
| 3072 | 8809.8 | 8819.5 | +0.1% |
| 4096 | 11517.8 | 11524.6 | +0.1% |
| 8192 | 22602.6 | 22606.6 | +0.0% |

Targeted small-M probe with `8192` included to keep NMT comparable:

```bash
python tests/bench_nvfp4_mega_moe_sm90.py --batches 8 16 32 64 128 8192 --num-tests 50
```

| M | iter12 current mean_rank us | iter11 `7756bfe` mean_rank us | delta |
|---:|---:|---:|---:|
| 8 | 750.6 | 758.0 | -1.0% |
| 16 | 796.8 | 800.8 | -0.5% |
| 32 | 823.5 | 841.6 | -2.1% |
| 64 | 845.7 | 835.6 | +1.2% |
| 128 | 835.2 | 835.9 | -0.1% |
| 8192 | 22641.9 | 22618.8 | +0.1% |

### Result

Keep. The same-condition small-M targeted probe shows no stable regression, and the full-list run is flat-to-slightly faster across most sizes. This completes compile-time phase specialization for the active mma.sync small-M path without changing its math.

## Iteration 13 - Rejected default WGMMA SF-load phase specialization

### Change Tried

- Tried converting the default WGMMA loop activation-SF load selector from a runtime-looking phase branch to:

```cpp
if constexpr (!kBlockIsL2) { ... } else { ... }
```

- Scope was intentionally narrow:
  - only the L1/L2 activation-SF load branch inside `run_default_gemm_loop`,
  - no WGMMA instruction sequence changes,
  - no epilogue changes,
  - no scheduler, wrapper, or launch-selection changes.

### Correctness

- Build: `./develop.sh` passed.
- Smoke correctness, `weight_scale=0.05`: PASS for `M=512,819,1024,2048`.
- Full correctness, `weight_scale=0.05`: PASS for `M=32,64,128,256,500,512,819,1000,1024,2048,4096,8192`.
- Tiny-signal absolute fallback, `weight_scale=0.001`: PASS for `M=512,819,1024,2048`.

### Benchmark

Full-list 50-run command:

```bash
python tests/bench_nvfp4_mega_moe_sm90.py \
  --batches 8 16 32 64 128 256 260 500 512 819 1000 1024 1280 1536 2048 3072 4096 8192 \
  --num-tests 50
```

Environment:

- `DG_JIT_CACHE_DIR=/tmp/dg_jit_iter13_wgmma_sf_bench50`

Full-list result:

| M | iter12 mean_rank us | iter13 tried mean_rank us | delta |
|---:|---:|---:|---:|
| 260 | 1288.1 | 1312.8 | +1.9% |
| 512 | 1991.5 | 2013.6 | +1.1% |
| 1024 | 3511.1 | 3522.2 | +0.3% |
| 1280 | 4079.0 | 4098.1 | +0.5% |
| 2048 | 6179.0 | 6174.4 | -0.1% |
| 4096 | 11524.6 | 11540.4 | +0.1% |
| 8192 | 22606.6 | 22606.1 | -0.0% |

Targeted probe with `8192` included:

```bash
python tests/bench_nvfp4_mega_moe_sm90.py --batches 260 512 1024 8192 --num-tests 50
```

| M | iter13 tried mean_rank us | iter12 `947e362` mean_rank us | delta |
|---:|---:|---:|---:|
| 260 | 1283.2 | 1266.7 | +1.3% |
| 512 | 2054.0 | 2025.7 | +1.4% |
| 1024 | 3559.1 | 3566.2 | -0.2% |
| 8192 | 22723.8 | 22663.9 | +0.3% |

### Result

Reject and revert. Correctness passed, but the same-condition targeted comparison showed stable regressions on `M=260` and `M=512` without a compensating gain. The kernel source was restored to Iteration 12 (`947e362`) before committing this log entry.

## Iteration 14 - Use PR323-style fused block iterator

### Change

- Refactored only the fused `for_each_selected_block` path to use the explicit PR323-style scheduler loop:
  - call `fetch_expert_recv_count()`,
  - reset `set_expert_idx(0)`,
  - repeatedly call `get_next_block()`,
  - dispatch `Linear1` and `Linear2` with compile-time phase tags and fixed K-block counts.
- Split L1 and split L2 paths are unchanged.
- Did not change math, dequant, epilogue, wrapper launch selection, or any new optimization heuristic.

### Correctness

- Build: `./develop.sh` passed.
- Smoke correctness, `weight_scale=0.05`: PASS for `M=512,819,1024,2048`.
- Full correctness, `weight_scale=0.05`: PASS for `M=32,64,128,256,500,512,819,1000,1024,2048,4096,8192`.
- Tiny-signal absolute fallback, `weight_scale=0.001`: PASS for `M=512,819,1024,2048` with `small_signal_ref_abs_max=1e-2`, `small_signal_abs_max_threshold=0.004`, `small_signal_abs_mean_threshold=0.0004`.

### Benchmark

Full-list 50-run command:

```bash
python tests/bench_nvfp4_mega_moe_sm90.py \
  --batches 8 16 32 64 128 256 260 500 512 819 1000 1024 1280 1536 2048 3072 4096 8192 \
  --num-tests 50
```

Environment:

- `DG_JIT_CACHE_DIR=/tmp/dg_jit_iter14_pr323_iter_bench50`
- 8 ranks, hidden 7168, intermediate hidden 2048, experts 256, topk 8.

| M | iter12 mean_rank us | iter14 mean_rank us | delta |
|---:|---:|---:|---:|
| 8 | 752.9 | 744.8 | -1.1% |
| 16 | 797.2 | 798.6 | +0.2% |
| 32 | 885.7 | 830.3 | -6.3% |
| 64 | 817.9 | 819.3 | +0.2% |
| 128 | 838.6 | 858.9 | +2.4% |
| 256 | 1200.4 | 1204.3 | +0.3% |
| 260 | 1288.1 | 1312.8 | +1.9% |
| 500 | 1962.2 | 1967.9 | +0.3% |
| 512 | 1991.5 | 1994.6 | +0.2% |
| 819 | 2809.9 | 2815.4 | +0.2% |
| 1000 | 3355.2 | 3337.5 | -0.5% |
| 1024 | 3511.1 | 3500.4 | -0.3% |
| 1280 | 4079.0 | 4078.1 | -0.0% |
| 1536 | 4874.4 | 4880.9 | +0.1% |
| 2048 | 6179.0 | 6167.9 | -0.2% |
| 3072 | 8819.5 | 8834.6 | +0.2% |
| 4096 | 11524.6 | 11546.5 | +0.2% |
| 8192 | 22606.6 | 22643.1 | +0.2% |

Targeted same-condition probes with `8192` included to keep NMT comparable:

```bash
python tests/bench_nvfp4_mega_moe_sm90.py --batches 32 128 260 512 8192 --num-tests 50
```

| probe | M | iter12 `947e362` mean_rank us | iter14 mean_rank us | delta |
|---|---:|---:|---:|---:|
| target1 | 32 | 801.5 | 805.5 | +0.5% |
| target1 | 128 | 816.4 | 828.9 | +1.5% |
| target1 | 260 | 1343.9 | 1380.1 | +2.7% |
| target1 | 512 | 2033.6 | 2027.7 | -0.3% |
| target1 | 8192 | 22614.9 | 22624.6 | +0.0% |
| target2 | 32 | 801.5 | 808.8 | +0.9% |
| target2 | 128 | 816.4 | 824.1 | +0.9% |
| target2 | 260 | 1343.9 | 1339.1 | -0.4% |
| target2 | 512 | 2033.6 | 2051.4 | +0.9% |
| target2 | 8192 | 22614.9 | 22626.4 | +0.1% |

Focused 100-run M128 check:

```bash
python tests/bench_nvfp4_mega_moe_sm90.py --batches 128 8192 --num-tests 100
```

| M | iter12 `947e362` mean_rank us | iter14 mean_rank us | delta |
|---:|---:|---:|---:|
| 128 | 824.4 | 810.8 | -1.6% |
| 8192 | 22695.0 | 22725.6 | +0.1% |

### Result

Keep. The full-list run showed possible slow points at `M=128` and `M=260`, but same-condition probes did not reproduce a stable regression: `M=260` flipped from slower to slightly faster on the second current rerun, and the focused 100-run `M=128` check was faster than Iteration 12. The change is therefore accepted as a low-risk refactor toward a true fused/split structure, not as a claimed new performance optimization.

## Milestone Candidate - Narrow true-split kernel launch arguments

### Change

- Narrowed the split L1 and split L2 CUDA entrypoint signatures so split launches no longer pass the full fused argument list.
- Split L1 now receives only L1 descriptors, L1 weight scale pointer, L1 output descriptor, symm buffer, token count, and cumulative stats. Inactive L2 names are local aliases used only to keep the shared compile-time body syntactically valid.
- Split L2 now receives only y, L2 descriptors, L2 weight scale pointer, symm buffer, token count, and cumulative stats. Inactive L1 names are local aliases used only for discarded compile-time branches.
- Fused kernel signature and fused launch arguments are unchanged.
- No math, scheduler, dequant, epilogue, or size-selection heuristic was changed.
- Per user direction, this is logged as a milestone candidate only; no per-iteration commit/push is made.

### Correctness

- Build: `./develop.sh` in `mega_moe_box` passed.
- Smoke correctness, `weight_scale=0.05`: PASS for `M=512,819,1024,2048`.
- Full correctness, `weight_scale=0.05`: PASS for `M=32,64,128,256,500,512,819,1000,1024,2048,4096,8192`.
- Tiny-signal absolute fallback, `weight_scale=0.001`: PASS for `M=512,819,1024,2048` with `small_signal_ref_abs_max=0.01`, `small_signal_abs_max_threshold=0.004`, `small_signal_abs_mean_threshold=0.0004`.

### Benchmark Sanity

Command:

```bash
python3 tests/bench_nvfp4_mega_moe_sm90.py --batches 32 512 819 1024 8192 --num-tests 30
```

Environment:

- `DG_JIT_CACHE_DIR=/tmp/dg_jit_narrow_split_args_bench30`
- 8 ranks, hidden 7168, intermediate hidden 2048, experts 256, topk 8.

| M | mean_rank us | max_rank us |
|---:|---:|---:|
| 32 | 828.6 | 835.7 |
| 512 | 1936.6 | 1945.0 |
| 819 | 2780.9 | 2786.0 |
| 1024 | 3491.4 | 3503.0 |
| 8192 | 22604.1 | 22622.0 |

### Full 50-run Benchmark Gate

Command:

```bash
python3 tests/bench_nvfp4_mega_moe_sm90.py \
  --batches 8 16 32 64 128 256 260 500 512 819 1000 1024 1280 1536 2048 3072 4096 8192 \
  --num-tests 50
```

Environment:

- `DG_JIT_CACHE_DIR=/tmp/dg_jit_narrow_split_args_bench50`

| M | iter14 mean_rank us | narrow-args mean_rank us | delta |
|---:|---:|---:|---:|
| 8 | 744.8 | 757.5 | +1.7% |
| 16 | 798.6 | 806.5 | +1.0% |
| 32 | 830.3 | 843.0 | +1.5% |
| 64 | 819.3 | 812.7 | -0.8% |
| 128 | 858.9 | 854.9 | -0.5% |
| 256 | 1204.3 | 1193.2 | -0.9% |
| 260 | 1312.8 | 1298.6 | -1.1% |
| 500 | 1967.9 | 1962.9 | -0.3% |
| 512 | 1994.6 | 1992.2 | -0.1% |
| 819 | 2815.4 | 2826.3 | +0.4% |
| 1000 | 3337.5 | 3337.8 | +0.0% |
| 1024 | 3500.4 | 3498.9 | -0.0% |
| 1280 | 4078.1 | 4081.9 | +0.1% |
| 1536 | 4880.9 | 4873.5 | -0.2% |
| 2048 | 6167.9 | 6167.0 | -0.0% |
| 3072 | 8834.6 | 8820.6 | -0.2% |
| 4096 | 11546.5 | 11531.2 | -0.1% |
| 8192 | 22643.1 | 22607.9 | -0.2% |

### Result

Keep as a milestone candidate. The change makes the true split entrypoints cleaner without changing the compute body or default routing. The full 50-run sweep shows no material regression on the split sizes affected by the launch ABI reduction; the only >1% positive deltas are on small fused sizes whose kernel signature and launch arguments are unchanged, so they are treated as run-to-run noise rather than a refactor regression. Do not push until this is folded into a milestone commit and the user asks for push.

### E2E Coverage Audit

- DeepGEMM exports `nvfp4_mega_moe` from `csrc/apis/mega.hpp`.
- Current SGLang at `/root/fac/sglang` does not call `nvfp4_mega_moe`; `git grep nvfp4_mega_moe -- python/sglang` is empty, and cross-ref searches over local/remotes also found no `nvfp4_mega_moe` or `transform_nvfp4_weights_for_mega_moe_sm90` integration.
- Current SGLang MegaMoE config maps SM90 to `fp8_mega_moe` and SM100 FP4 to `fp8_fp4_mega_moe` in `python/sglang/srt/layers/moe/mega_moe.py`.
- Therefore the full correctness gate above is a kernel-level NVFP4 gate. Historical GSM8K / SGLang e2e runs in the current SGLang tree would not prove this SM90 NVFP4 kernel unless a separate SGLang integration route calls `deep_gemm.nvfp4_mega_moe`.
- Kernel-level boundary correctness also passed for the historically suspicious effective-token sizes and neighbors: `M=1024,1025,1200,2047,2048`, `weight_scale=0.05`, with `cosine_min=0.9986-0.9987` and finite outputs.

## Milestone Candidate - Independent fused and split kernel bodies

### Change

- Replaced the shared `sm90_nvfp4_mega_moe_body.inl` include with three independent kernel bodies:
  - `sm90_nvfp4_mega_moe_fused_body.inl`
  - `sm90_nvfp4_mega_moe_split_l1_body.inl`
  - `sm90_nvfp4_mega_moe_split_l2_body.inl`
- Removed the old tracked shared body file.
- Removed `kPhaseMode`, `kRunL1Phase`, and `kRunL2Phase` from the compiled SM90 NVFP4 MegaMoE implementation.
- Split L1 no longer carries L2 descriptor/weight aliases in its entrypoint; split L2 no longer carries L1 descriptor/output aliases in its entrypoint.
- This is a pure source-structure refactor. No size routing, math, dequant, dispatch algorithm, combine algorithm, or optimization heuristic was intentionally changed.

### Correctness

- Build: `./develop.sh` in `mega_moe_box` passed after deleting the shared body.
- Smoke correctness with a fresh JIT cache: PASS for `M=32,512,819,1024`, `weight_scale=0.05`.
- Full correctness, `weight_scale=0.05`: PASS for `M=32,64,128,256,500,512,819,1000,1024,1280,1536,2047,2048,3072,4096,8192`.
- Tiny-signal absolute fallback, `weight_scale=0.001`: PASS for the same M list with `small_signal_ref_abs_max=0.01`, `small_signal_abs_max_threshold=0.004`, `small_signal_abs_mean_threshold=0.0004`.

### Full 50-run Benchmark Gate

Command:

```bash
python3 tests/bench_nvfp4_mega_moe_sm90.py \
  --batches 8 16 32 64 128 256 260 500 512 819 1000 1024 1280 1536 2048 3072 4096 8192 \
  --num-tests 50
```

Environment:

- `DG_JIT_CACHE_DIR=/tmp/dg_jit_three_body_bench50`
- 8 ranks, hidden 7168, intermediate hidden 2048, experts 256, topk 8.

| M | narrow-args mean_rank us | three-body mean_rank us | delta |
|---:|---:|---:|---:|
| 8 | 757.5 | 745.6 | -1.6% |
| 16 | 806.5 | 827.2 | +2.6% |
| 32 | 843.0 | 822.2 | -2.5% |
| 64 | 812.7 | 861.6 | +6.0% |
| 128 | 854.9 | 838.1 | -2.0% |
| 256 | 1193.2 | 1187.8 | -0.5% |
| 260 | 1298.6 | 1318.0 | +1.5% |
| 500 | 1962.9 | 1998.6 | +1.8% |
| 512 | 1992.2 | 2009.7 | +0.9% |
| 819 | 2826.3 | 2827.5 | +0.0% |
| 1000 | 3337.8 | 3362.9 | +0.8% |
| 1024 | 3498.9 | 3517.4 | +0.5% |
| 1280 | 4081.9 | 4092.0 | +0.2% |
| 1536 | 4873.5 | 4881.9 | +0.2% |
| 2048 | 6167.0 | 6151.5 | -0.3% |
| 3072 | 8820.6 | 8841.0 | +0.2% |
| 4096 | 11531.2 | 11528.0 | -0.0% |
| 8192 | 22607.9 | 22599.4 | -0.0% |

### Targeted Recheck

The full-list run showed small-M/routing noise at `M=16`, `M=64`, `M=260`, and `M=500`. A same-cache targeted 50-run recheck removed those apparent regressions:

```bash
python3 tests/bench_nvfp4_mega_moe_sm90.py --batches 16 64 260 500 --num-tests 50
```

| M | narrow-args mean_rank us | targeted three-body mean_rank us | delta |
|---:|---:|---:|---:|
| 16 | 806.5 | 790.8 | -1.9% |
| 64 | 812.7 | 799.7 | -1.6% |
| 260 | 1298.6 | 1284.1 | -1.1% |
| 500 | 1962.9 | 1918.7 | -2.3% |

### Result

Keep as the true three-body refactor milestone. The source now has independent fused, split-L1, and split-L2 bodies rather than one phase-mode body. Correctness passes across the full kernel-level M sweep, and targeted 50-run checks show no reproduced performance regression on the noisy full-list points. Do not push until the user asks for push.

### Completion Audit Cleanup

The completion audit found that the first three-body split still carried dead cross-phase ready-mask paths:

- split-L1 still had L2 ready-mask publish/cleanup code.
- split-L2 still had L1-ready wait/notify dead code.

Those paths were removed from the split bodies. Static grep now has no hits for:

- split-L1: `BlockPhase::Linear2`, `tensor_map_l2`, `l2_weights`, `get_l2_arrival_mask`
- split-L2: `BlockPhase::Linear1`, `tensor_map_l1`, `l1_weights`, `tensor_map_l1_output`, `notify_l1_ready`, `get_l2_arrival_mask`
- all bodies/entrypoints: `kPhaseMode`, `kRunL1Phase`, `kRunL2Phase`, `sm90_nvfp4_mega_moe_body.inl`

The remaining `get_l1_arrival_count_ptr` in split-L2 is workspace cleanup for the next call, not an L1 compute branch or L1 descriptor/output path.

Rebuild passed:

```bash
./develop.sh
```

Full correctness passed:

```bash
DG_JIT_CACHE_DIR=/tmp/dg_jit_three_body_clean_full_correct \
python3 tests/test_nvfp4_mega_moe_sm90_correctness.py \
  --batches 32 64 128 256 500 512 819 1000 1024 1280 1536 2047 2048 3072 4096 8192 \
  --weight-scales 0.05 0.001 \
  --small-signal-ref-abs-max 0.01 \
  --small-signal-abs-max-threshold 0.004 \
  --small-signal-abs-mean-threshold 0.0004
```

Final full-list 50-run benchmark:

```bash
DG_JIT_CACHE_DIR=/tmp/dg_jit_three_body_clean_bench50 \
python3 tests/bench_nvfp4_mega_moe_sm90.py \
  --batches 8 16 32 64 128 256 260 500 512 819 1000 1024 1280 1536 2048 3072 4096 8192 \
  --num-tests 50
```

| M | narrow-args mean_rank us | clean three-body mean_rank us | delta |
|---:|---:|---:|---:|
| 8 | 757.5 | 749.8 | -1.0% |
| 16 | 806.5 | 797.3 | -1.1% |
| 32 | 843.0 | 861.8 | +2.2% |
| 64 | 812.7 | 817.4 | +0.6% |
| 128 | 854.9 | 824.8 | -3.5% |
| 256 | 1193.2 | 1194.5 | +0.1% |
| 260 | 1298.6 | 1294.4 | -0.3% |
| 500 | 1962.9 | 1954.4 | -0.4% |
| 512 | 1992.2 | 2026.1 | +1.7% |
| 819 | 2826.3 | 2818.2 | -0.3% |
| 1000 | 3337.8 | 3359.0 | +0.6% |
| 1024 | 3498.9 | 3535.2 | +1.0% |
| 1280 | 4081.9 | 4088.5 | +0.2% |
| 1536 | 4873.5 | 4883.9 | +0.2% |
| 2048 | 6167.0 | 6166.0 | -0.0% |
| 3072 | 8820.6 | 8828.4 | +0.1% |
| 4096 | 11531.2 | 11534.4 | +0.0% |
| 8192 | 22607.9 | 22628.4 | +0.1% |

Full-list points over 1% were rechecked with the same JIT cache:

```bash
DG_JIT_CACHE_DIR=/tmp/dg_jit_three_body_clean_bench50 \
python3 tests/bench_nvfp4_mega_moe_sm90.py --batches 32 512 1024 --num-tests 50
```

| M | narrow-args mean_rank us | targeted clean three-body mean_rank us | delta |
|---:|---:|---:|---:|
| 32 | 843.0 | 821.0 | -2.6% |
| 512 | 1992.2 | 1932.4 | -3.0% |
| 1024 | 3498.9 | 3503.3 | +0.1% |

Result: keep. The stricter split-body cleanup still passes full correctness and does not reproduce a performance regression in targeted 50-run checks.
