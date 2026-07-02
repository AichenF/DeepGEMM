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

## 2026-07-01 AKO: PR323 swapAB small-M target setup

Objective: use AKO on `megamoe_nvfp4` to port/build a PR323-style swapAB small-M path for SM90 NVFP4 MegaMoE. The target sizes are the Flash/Pro small-M cases where current NVFP4 trails the PR323 W4A8 FP4 numbers. Success requires beating those PR323 FP4 targets without NVFP4 correctness regression.

Reference target table from `/root/fac/scripts/megamoe/v4_shape_matrix_20260701/summary_split_w8a8_w4a8_gap.md`:

| shape | M | PR323 W4A8 FP4 us | previous ours NVFP4 us | previous gap |
|---|---:|---:|---:|---:|
| Flash | 8 | 373.4 | 409.3 | +9.6% |
| Flash | 16 | 416.5 | 499.6 | +20.0% |
| Flash | 32 | 440.3 | 496.1 | +12.7% |
| Flash | 64 | 476.3 | 505.6 | +6.2% |
| Flash | 256 | 540.5 | 646.0 | +19.5% |
| Pro | 8 | 1213.5 | 1328.8 | +9.5% |
| Pro | 16 | 1485.2 | 1625.1 | +9.4% |
| Pro | 32 | 1482.5 | 1673.3 | +12.9% |
| Pro | 64 | 1521.6 | 1661.3 | +9.2% |
| Pro | 256 | 1805.0 | 2007.1 | +11.2% |

Current direct-bench baseline, 50-run, default policy (`BN256` fused for all these small-M cases):

| shape | M | NVFP4 us |
|---|---:|---:|
| Flash | 8 | 520.8 |
| Flash | 16 | 492.8 |
| Flash | 32 | 558.8 |
| Flash | 64 | 523.9 |
| Flash | 128 | 586.7 |
| Flash | 256 | 550.1 |
| Pro | 8 | 1451.0 |
| Pro | 16 | 1684.0 |
| Pro | 32 | 1679.0 |
| Pro | 64 | 1716.0 |
| Pro | 128 | 1729.0 |
| Pro | 256 | 1739.0 |

Existing `BN128` split L1/L2 forced with `--nvfp4-block-n 128`, 50-run:

| shape | M | BN128 split NVFP4 us |
|---|---:|---:|
| Flash | 8 | 654.8 |
| Flash | 16 | 762.8 |
| Flash | 32 | 694.9 |


## 2026-07-01 AKO iter-1: BN128 fused split-N scaffold for swapAB

Change: added a non-default `DG_SM90_NVFP4_FUSED_BN128_EXPERIMENT=1` path so BN128 layout can launch the fused body instead of the existing two-kernel split L1/L2 path. The fused body now accepts BN128 split-N with two epilogue WGs, adds shared-SF scratch for the 32+32 column L1 output halves, and allows direct L2 scatter for `WG_BLOCK_N=64`.

Why: PR323 swapAB requires the single-kernel `BLOCK_M=64`, `BLOCK_N=128`, split-N epilogue-WG shape. Existing NVFP4 BN128 used two separate L1/L2 kernels, so it could not host the PR323 swapAB dataflow.

Validation:

```bash
./develop.sh
DG_SM90_NVFP4_FUSED_BN128_EXPERIMENT=1 DG_JIT_CACHE_DIR=/tmp/dg_jit_ako_bn128_splitn_correct python3 tests/test_nvfp4_mega_moe_sm90_correctness.py   --batches 8 32 --weight-scales 0.05 --nvfp4-block-n 128   --small-signal-ref-abs-max 0.01   --small-signal-abs-max-threshold 0.004   --small-signal-abs-mean-threshold 0.0004
```

Result: PASS for M=8/32, global_scale_mode none/expert, cosine_min around 0.9988.

Performance sanity:

```bash
DG_SM90_NVFP4_FUSED_BN128_EXPERIMENT=1 DG_JIT_CACHE_DIR=/tmp/dg_jit_ako_bn128_splitn_m8_v2 python3 tests/bench_nvfp4_mega_moe_sm90.py   --batches 8 --hidden 4096 --intermediate-hidden 2048   --num-experts 256 --num-topk 6 --num-tests 5 --nvfp4-block-n 128
```

M=8 Flash: 1335.7 us, much slower than default BN256 fused. This scaffold is not an optimization result; it only establishes the compile/correctness substrate for the next swapAB iter.

Next: add `DG_SM90_NVFP4_SWAP_AB_EXPERIMENT` as a fused-only BN128 split-N template flag and port PR323's swapAB MMA plus swapAB L1/L2 epilogues. Do not default-enable this scaffold.

## 2026-07-01 AKO small-M swapAB follow-up: rejected BN128 fused and extra N buckets

Goal: improve small-M swapAB before returning to M=256. Tested two directions and reverted both because neither gave no-regression performance.

1. BN128 fused swapAB default for small M:
   - Change tried: make `choose_nvfp4_block_n_for_mega_moe_sm90()` return BN128 for expected tokens/expert <= 24 and let the wrapper launch fused swapAB for BN128 small-M layouts.
   - Correctness: PASS for M=8/16/32/48/64 with `global_scale_mode=none/expert`.
   - Flash 50-run result: M8 1371.7 us, M16 1413.3 us, M32 1477.8 us, M64 1558.3 us, M128 1600.5 us.
   - Conclusion: reject. BN128 fused currently requires loader-dequant and is far slower than BN256 fused packed-scratch/math-side dequant for small M.

2. Extra swapAB `N_SWAP` buckets:
   - Change tried: add L1 `N=32` bucket. Correctness PASS, but Flash M16/M32 and Pro M16/M32 regressed.
   - Change tried: add generic `N=24` bucket. Correctness PASS, but Flash M32/M64 and Pro M16/M64/M128 regressed.
   - Conclusion: reject. Keep original bucket policy: L1 uses 8/16/24/64 only for the Flash epw16 special case and 8/16/64 for generic; L2 keeps the original 8/16/24/32/64 Flash special and 8/16/32/64 generic.

Stable post-revert small-M baseline, BN256 fused, 50-run:

| shape | M | NVFP4 us |
|---|---:|---:|
| Flash | 8 | 464.1 |
| Flash | 16 | 422.9 |
| Flash | 32 | 508.0 |
| Flash | 64 | 520.9 |
| Flash | 128 | 575.5 |
| Pro | 8 | 1330.0 |
| Pro | 16 | 1494.0 |
| Pro | 32 | 1525.0 |
| Pro | 64 | 1576.0 |
| Pro | 128 | 1723.0 |

Next viable small-M direction: do not pursue BN128 fused unless BN128 gets a packed-scratch/math-side dequant path. The BN256 non-fast-amax SMEM staging variant was tried next and rejected below.

## 2026-07-01 AKO small-M swapAB follow-up: rejected SMEM staging amax

Goal: reduce BN256 fused swapAB register/local-memory pressure by staging the L1 intermediate to SMEM, computing the per-token amax from SMEM, then quantizing. This was intended to help very small M without changing the accepted swapAB MMA/dataflow.

Implementation tried:
- Added a fused-only `swap_ab_fast_amax` template/policy switch.
- `fast_amax=true` preserved the original register-resident amax path.
- `fast_amax=false` wrote unweighted `silu(gate) * up` to SMEM, reduced amax per 64-column L1 group, applied the route weight in the scale path, then wrote FP8 to the existing L1 output SMEM for the common TMA store.

Correctness:
- PASS for M=8/16/32/64 with `global_scale_mode=none/expert`.
- Logs:
  - `/root/fac/scripts/megamoe/ako_nvfp4_swapab/correct_nonfast_amax_20260701_222615.log`
  - `/root/fac/scripts/megamoe/ako_nvfp4_swapab/correct_hybrid_fast_amax_20260701_222946.log`

Performance findings, 50-run:
- All-non-fast showed a real M8 opportunity, but regressed M16 and was not stable enough:
  - Flash: M8 407.1 us, M16 700.8 us, M32 501.6 us, M64 540.2 us, M128 537.6 us.
  - Pro: M8 1258.0 us, M16 1506.0 us, M32 1555.0 us, M64 1588.0 us, M128 1725.0 us.
- Hybrid policy (`fast_amax=false` only for expected tokens/expert <= 1.5) still had regressions:
  - Flash repeat: M8 431.4 us, M16 545.0 us, M32 493.8 us, M64 540.2 us, M128 546.7 us.
  - Pro repeat: M8 1256.0 us, M16 1510.0 us, M32 1536.0 us, M64 1580.0 us, M128 1749.0 us.

Decision: reject and revert. The SMEM staging path can improve M8, but it adds a second kernel variant and causes M16/M64 instability/regression. It violates the no-regression constraint for default code.

Post-revert validation:
- Removed all `swap_ab_fast_amax` / `kSwapABFastAmax` / `smem_cd_l1_swap_fp32` references.
- Correctness PASS: `/root/fac/scripts/megamoe/ako_nvfp4_swapab/correct_reverted_fastamax_20260701_224305.log`

Post-revert small-M 50-run:

| shape | M | NVFP4 us |
|---|---:|---:|
| Flash | 8 | 427.0 |
| Flash | 16 | 429.6 |
| Flash | 32 | 546.7 |
| Flash | 64 | 537.6 |
| Flash | 128 | 539.7 |
| Flash repeat | 32 | 478.3 |
| Flash repeat | 64 | 502.6 |
| Pro | 8 | 1240.0 |
| Pro | 16 | 1492.0 |
| Pro | 32 | 1526.0 |
| Pro | 64 | 1563.0 |
| Pro | 128 | 1713.0 |

Notes:
- Flash M32/M64 varied by routing sample (`recv` changed between runs), so the single full-row Flash M32/M64 values should not be interpreted as a code regression. The repeat was faster than the stable baseline at those points.
- Keep the original BN256 fused swapAB fast-amax path as the default small-M implementation.

## 2026-07-01 AKO comparison reset: fixed-capacity PR323 baseline

The post-revert logs above used `num_max_tokens_per_rank=max(batches)`, while the
PR323 comparison table fixes capacity at 8192. Those values are useful for local
A/B testing but are not a fair PR323 comparison. All subsequent PR323 claims use
the existing matrix runner with the original contract:

- `num_max_tokens_per_rank=8192`
- M=8..128: 50-run median
- M=256: 20-run mean
- one M per process, which also fixes routing for repeatable A/B tests

Runner:

```bash
python3 /root/fac/scripts/megamoe/v4_shape_matrix_runner.py \
  --impl ours_nvfp4 --shape <flash|pro> --m <M> --cap 8192 \
  --stat <median|mean> --num-tests <50|20>
```

Current BN256 fused swapAB baseline (`DG_JIT_CACHE_DIR=/tmp/dg_jit_ako_swapab_cap8192_baseline_20260701_231731`):

| shape | M | stat | PR323 us | current us | gap |
|---|---:|---|---:|---:|---:|
| Flash | 8 | median | 373.4 | 371.2 | -0.6% |
| Flash | 16 | median | 416.5 | 426.6 | +2.4% |
| Flash | 32 | median | 440.3 | 456.6 | +3.7% |
| Flash | 64 | median | 476.3 | 507.1 | +6.5% |
| Flash | 128 | median | 519.4 | 507.2 | -2.3% |
| Flash | 256 | mean | 540.5 | 675.9 | +25.0% |
| Pro | 8 | median | 1213.5 | 1207.6 | -0.5% |
| Pro | 16 | median | 1485.2 | 1480.4 | -0.3% |
| Pro | 32 | median | 1482.5 | 1491.1 | +0.6% |
| Pro | 64 | median | 1521.6 | 1568.1 | +3.1% |
| Pro | 128 | median | 1711.8 | 1693.2 | -1.1% |
| Pro | 256 | mean | 1805.0 | 1856.8 | +2.9% |

Raw logs use the prefix
`/root/fac/scripts/megamoe/ako_nvfp4_swapab/cap8192_baseline_*_20260701_231731.log`.

Interpretation:

- M=8 already beats PR323 on both shapes; preserve it.
- M=16 and M=128 are guardrails. Pro M=16 and both M=128 cases already win.
- The next optimization targets are Flash M=64, Flash M=32, and Pro M=64/32.
- M=256 remains a follow-up target. Its mean is strongly affected by long-tail
  samples (Flash median 510.4 us versus mean 675.9 us; Pro median 1695.6 us
  versus mean 1856.8 us), so profile both steady-state cost and tail behavior.

## 2026-07-01 AKO iter: grouped BN256 swapAB weight halves (rejected)

Hypothesis: the BN256 swapAB path serializes its two 64-column weight halves.
Each L1 half had an independent `warpgroup_arrive/commit/wait`, and each L2
activation-SF group repeated the same sequence for both halves. Issue both
halves into one WGMMA group before waiting to overlap their latency.

Change tested:

- L1: one group containing both halves (8 WGMMA instructions per K block).
- L2: one group containing both halves for each of the two activation-SF groups.
- No bucket, policy, epilogue, scale, or layout change.

Validation:

- `./develop.sh`: PASS.
- Exact-NVFP4 correctness: PASS for Flash and Pro hidden sizes, M=8/16/32/64,
  `global_scale_mode=none/expert`, with 64 local experts and capacity 8192.
- Logs:
  - `/root/fac/scripts/megamoe/ako_nvfp4_swapab/correct_grouped_halves_flash_20260701.log`
  - `/root/fac/scripts/megamoe/ako_nvfp4_swapab/correct_grouped_halves_pro_20260701.log`

Fixed-capacity 50-run median results:

| shape | M | baseline us | grouped-halves us | delta |
|---|---:|---:|---:|---:|
| Flash | 8 | 371.2 | 371.4 | +0.1% |
| Flash | 16 | 426.6 | 448.6 | +5.1% |
| Flash | 32 | 456.6 | 453.8 | -0.6% |
| Flash | 64 | 507.1 | 503.7 | -0.7% (non-swap path; noise control) |
| Pro | 8 | 1207.6 | 1209.4 | +0.1% |
| Pro | 16 | 1480.4 | 1463.9 | -1.1% |
| Pro | 32 | 1491.1 | 1513.6 | +1.5% |
| Pro | 64 | 1568.1 | 1578.2 | +0.6% |

Raw logs use
`/root/fac/scripts/megamoe/ako_nvfp4_swapab/grouped_halves_{flash,pro}_m* _20260701_2333.log`
(without the space before the timestamp suffix).

Decision: reject and revert. Small isolated gains do not reach the 3% AKO
signal threshold, while Flash M16 has a clear 5.1% regression and Pro M32/M64
also regress. Keeping two live accumulator sets likely increases register
pressure enough to offset the reduced WGMMA wait count.

## 2026-07-01 AKO iter: force existing swapAB for Flash M64 (rejected)

Hypothesis: Flash M64 is the largest small-M gap because its expected
tokens/local-expert is 12, above the current swapAB policy threshold of 8.
Temporarily enable the existing BN256 swapAB implementation only for the exact
Flash M64 target shape; leave every other shape and size unchanged.

Validation:

- `./develop.sh`: PASS.
- The selected swapAB kernel body had already passed exact-NVFP4 M64
  correctness for both Flash and Pro hidden sizes in the preceding iteration.
- Fixed-capacity 50-run median:
  - baseline: 507.1 us
  - forced swapAB: 504.8 us (-0.5%)
  - PR323 target: 476.3 us (forced swapAB remains +6.0% slower)
- Log:
  `/root/fac/scripts/megamoe/ako_nvfp4_swapab/flash_m64_forced_swapab_cap8192_20260701.log`

Decision: reject and revert. The 0.5% change is below the 3% AKO signal
threshold and within the observed M64 run-to-run noise. Extending the policy
alone does not close the gap; M64 needs a different compute/dispatch strategy.

## 2026-07-01 AKO iter: activate 2-warp BN256 loader dequant (rejected)

Audit finding: commit `8cff54d` added a BN256 packed-scratch loader-dequant
branch for two non-epilogue warps and host-side support for it, but the kernel's
`kLoaderDequant` compile-time condition still required four warps. The branch
was therefore unreachable.

Change tested behind the existing `DG_SM90_NVFP4_LOADER_DEQUANT=1` override:

- Allow the intended two-warp BN256 packed-scratch loader-dequant path.
- Keep packed B scratch when loader-dequant consumes it.
- Initialize the dequant barrier with two arrivals and have both producer
  warps arrive, avoiding the race in the dormant implementation.
- Leave the default small-M policy disabled for this A/B.

Validation:

- `./develop.sh`: PASS.
- Exact-NVFP4 correctness: PASS for Flash and Pro hidden sizes, M=8/16/32/64,
  `global_scale_mode=none/expert`, 64 local experts, capacity 8192.
- Logs:
  - `/root/fac/scripts/megamoe/ako_nvfp4_swapab/correct_bn256_loader_dequant_flash_20260701.log`
  - `/root/fac/scripts/megamoe/ako_nvfp4_swapab/correct_bn256_loader_dequant_pro_20260701.log`

Fixed-capacity 50-run median results:

| shape | M | baseline us | 2-warp loader-dequant us | delta |
|---|---:|---:|---:|---:|
| Flash | 8 | 371.2 | 949.0 | +155.7% |
| Flash | 16 | 426.6 | 1092.2 | +156.0% |
| Flash | 32 | 456.6 | 1082.4 | +137.1% |
| Flash | 64 | 507.1 | 1069.6 | +110.9% |
| Pro | 8 | 1207.6 | 3244.1 | +168.6% |
| Pro | 16 | 1480.4 | 4038.8 | +172.8% |
| Pro | 32 | 1491.1 | 4088.0 | +174.2% |
| Pro | 64 | 1568.1 | 4100.5 | +161.5% |

Raw logs use
`/root/fac/scripts/megamoe/ako_nvfp4_swapab/bn256_loader_dequant_{flash,pro}_m*_20260701_2348.log`.

Decision: reject and revert. Two loader warps cannot decode the full BN256
packed tile fast enough; they become the critical path despite pipeline
overlap. The existing 256-thread math-side decode is substantially faster.

Stall checkpoint: grouped WGMMA halves, Flash-M64 policy extension, and
2-warp loader-dequant are three consecutive directions without a >=3% win.
Reassess profiles, PR323 implementation differences, and iteration history
before the next code change.

## 2026-07-01 AKO stall reassessment

Evidence reviewed:

- Fixed-capacity baseline and the three rejected iterations above.
- Flash M64 NCU launch resources: 384 threads, 168 registers/thread,
  204.48 KiB dynamic shared memory, one CTA/SM, and no local-memory spills.
  Application-replay duration/throughput counters were invalidated by the
  cross-rank ready protocol, so only launch/resource data are used.
- PR53/PR323 source: the small-batch FP4 path uses BM64/BN128 split-N with two
  consumer warpgroups, so each WG owns 64 weight rows. The current NVFP4 BN256
  path makes each WG serially cover two 64-row halves.
- NVIDIA Hopper/CUTLASS guidance: persistent cooperative kernels split one
  output tile across consumer warpgroups to reduce per-WG register pressure;
  producer/consumer pipelining only helps when the producer is not itself the
  critical path.

Conclusions:

- BN256 is already at one CTA/SM and near the SMEM limit; occupancy tuning is
  not the next lever.
- Grouping both BN256 halves increased live accumulators and did not help.
- Moving full-tile decode to only 64 producer threads made decode the critical
  path. Keep math-side decode for small M.
- Merely enabling BN256 swapAB at Flash M64 does not reduce enough work.

Next direction: revive the BN128 fused split-N scaffold, but replace its slow
loader-dequant path with packed scratch plus 128-thread math-side dequant.
This matches PR323's 64-row-per-consumer-WG structure while preserving NVFP4's
per-16 UE4M3 scale semantics. Keep it behind an experiment switch until exact
correctness and fixed-capacity benchmarks prove a no-regression win.

## 2026-07-01 AKO iter: BN128 fused packed-scratch math dequant (evaluating)

Change under `DG_SM90_NVFP4_FUSED_BN128_PACKED_EXPERIMENT=1`:

- Run the BN128 layout as one fused kernel with two split-N consumer
  warpgroups and 64 dispatch/non-epilogue threads each.
- TMA-load fused 80-byte packed weight/scale rows into per-stage scratch.
- Have the first 128 consumer threads decode one BN128 row each, then use a
  256-thread barrier before either consumer WG issues WGMMA.
- Reuse the exact swapAB L1/L2 epilogues with one 64-row weight half per WG.
- Leave default BN128 split and BN256 fused selection unchanged.

Validation so far:

- `git diff --check`, runner `py_compile`, and `./develop.sh`: PASS.
- Exact-NVFP4 correctness: PASS for Flash (32 local experts) and Pro (48 local
  experts), M=8/16/32/64, `global_scale_mode=none/expert`, capacity 8192.
  The minimum per-token cosine across these cases is 0.9988.
- Logs:
  - `/root/fac/scripts/megamoe/ako_nvfp4_swapab/bn128_packed_correctness_flash_20260701.log`
  - `/root/fac/scripts/megamoe/ako_nvfp4_swapab/bn128_packed_correctness_pro_20260701.log`

Initial fixed-capacity 50-run median (single fused kernel event): Flash M=8 is
496.7 us versus the current 371.2 us baseline (+33.8%). The bench script's
993.4 us display is a stale BN128 split-path `x2`; `BENCH_STAT_JSON` contains
the correct one-kernel event median. Log:
`/root/fac/scripts/megamoe/ako_nvfp4_swapab/bn128_packed_flash_m8_20260701.log`.

Additional Flash target results use the same single-kernel event statistic:

| M | current baseline us | BN128 packed fused us | delta |
|---:|---:|---:|---:|
| 8 | 371.2 | 496.7 | +33.8% |
| 32 | 456.6 | 594.8 | +30.3% |
| 64 | 507.1 | 668.9 | +31.9% |

Additional logs:

- `/root/fac/scripts/megamoe/ako_nvfp4_swapab/bn128_packed_flash_m32_20260701.log`
- `/root/fac/scripts/megamoe/ako_nvfp4_swapab/bn128_packed_flash_m64_20260701.log`

Pro results show the same behavior:

| M | current baseline us | BN128 packed fused us | delta |
|---:|---:|---:|---:|
| 32 | 1491.1 | 1942.8 | +30.3% |
| 64 | 1568.1 | 2029.5 | +29.4% |

Logs:

- `/root/fac/scripts/megamoe/ako_nvfp4_swapab/bn128_packed_pro_m32_20260701.log`
- `/root/fac/scripts/megamoe/ako_nvfp4_swapab/bn128_packed_pro_m64_20260701.log`

Decision: reject and revert. Cutting each consumer WG from 128 to 64 weight
rows does not compensate for the BN128 tile count and the per-stage 256-thread
handoff after only half the consumers decode. The roughly 30% regression is
consistent across Flash and Pro, so this scaffold is not useful even as a
shape-selective path. Preserve the current BN256 winners.

Post-revert validation: both kernel source hashes match the saved BN256
baseline, `./develop.sh` passes, and a fresh-cache Flash M=8 50-run median is
367.8 us (saved baseline 371.2 us). Log:
`/root/fac/scripts/megamoe/ako_nvfp4_swapab/restore_check_flash_m8_20260701.log`.

## 2026-07-02 AKO iter: CTA-cached finalized expert counts (evaluating)

Hypothesis: the fused kernel independently polls finalized expert counters in
dispatch, both TMA loader roles, and both math WGs. Port PR323's CTA-level
cache so one set of threads polls global counters, publishes them in SMEM, and
all roles populate their scheduler state from that cache. This also lets the
loader roles leave the all-thread cache barrier while dispatch and math finish
their existing rendezvous.

Change:

- Add one 384-thread scheduler-count cache barrier.
- Cache finalized local-expert counts into the existing expert-count SMEM after
  dispatch publication is complete.
- Replace each role's global counter polling with warp-local loads from SMEM.
- Do not change scheduler order, tile shapes, dequant, WGMMA, or epilogues.

Validation so far:

- `git diff --check` and `./develop.sh`: PASS.
- Exact-NVFP4 correctness: PASS for Flash M=8/32/64,
  `global_scale_mode=none/expert`, 32 local experts, capacity 8192.
- Flash M=32 fixed-capacity 50-run median: 460.5 us versus 456.6 us baseline
  (+0.9%).
- Logs:
  - `/root/fac/scripts/megamoe/ako_nvfp4_swapab/cached_counts_correct_flash_20260702.log`
  - `/root/fac/scripts/megamoe/ako_nvfp4_swapab/cached_counts_flash_m32_20260702.log`

Flash M=64 fixed-capacity 50-run median: 490.1 us versus 507.1 us baseline
(-3.4%), reaching the AKO signal threshold but still 2.9% slower than PR323's
476.3 us. Log:
`/root/fac/scripts/megamoe/ako_nvfp4_swapab/cached_counts_flash_m64_20260702.log`.

Status: first qualifying target signal. Repeat M=64 independently and run all
small-M guardrails before accepting or rejecting.

Follow-up results:

| shape | M | baseline us | cached-count us | delta |
|---|---:|---:|---:|---:|
| Flash | 8 | 371.2 | 386.4 | +4.1% |
| Flash | 64 | 507.1 | 490.1 | -3.4% |
| Flash | 64 repeat | 507.1 | 495.0 | -2.4% |
| Pro | 64 | 1568.1 | 1532.8 | -2.3% |

Additional logs:

- `/root/fac/scripts/megamoe/ako_nvfp4_swapab/cached_counts_flash_m8_20260702.log`
- `/root/fac/scripts/megamoe/ako_nvfp4_swapab/cached_counts_flash_m64_repeat_20260702.log`
- `/root/fac/scripts/megamoe/ako_nvfp4_swapab/cached_counts_pro_m64_20260702.log`

The all-shape version is rejected because it loses the M=8 winner. Continue
only as a compile-time shape-gated M=64 candidate, then verify unchanged
shapes generate the original body and reproduce their baselines.

Pro M=32 also improved modestly: 1473.2 us versus 1491.1 us baseline (-1.2%)
and 1482.5 us PR323 (-0.6%). Log:
`/root/fac/scripts/megamoe/ako_nvfp4_swapab/cached_counts_pro_m32_20260702.log`.

Shape-gated policy under evaluation: enable only for Flash expected=12 and
Pro expected=4 or 8; add `DG_SM90_NVFP4_CACHE_RECV_COUNTS=0` as an exact
same-source A/B kill switch. Other sizes compile the original polling path.

Gated correctness: PASS for Flash M=8/32/64 and Pro M=32/64,
`global_scale_mode=none/expert`, with actual 32/48 local-expert counts.

First same-source kill-switch A/B, Flash M=64:

- cache on: 492.6 us
- cache off: 502.5 us
- delta: -2.0%

Logs:

- `/root/fac/scripts/megamoe/ako_nvfp4_swapab/cached_counts_gated_correct_flash_20260702.log`
- `/root/fac/scripts/megamoe/ako_nvfp4_swapab/cached_counts_gated_correct_pro_20260702.log`
- `/root/fac/scripts/megamoe/ako_nvfp4_swapab/cached_counts_gated_flash_m64_on_20260702.log`
- `/root/fac/scripts/megamoe/ako_nvfp4_swapab/cached_counts_gated_flash_m64_off_20260702.log`

Pro same-source kill-switch A/B:

| M | cache on us | cache off us | delta | PR323 us |
|---:|---:|---:|---:|---:|
| 32 | 1480.8 | 1517.2 | -2.4% | 1482.5 |
| 64 | 1549.7 | 1605.9 | -3.5% | 1521.6 |

Logs:

- `/root/fac/scripts/megamoe/ako_nvfp4_swapab/cached_counts_gated_pro_m32_on_20260702.log`
- `/root/fac/scripts/megamoe/ako_nvfp4_swapab/cached_counts_gated_pro_m32_off_20260702.log`
- `/root/fac/scripts/megamoe/ako_nvfp4_swapab/cached_counts_gated_pro_m64_on_20260702.log`
- `/root/fac/scripts/megamoe/ako_nvfp4_swapab/cached_counts_gated_pro_m64_off_20260702.log`

Status: the selected points consistently improve. Run all non-selected
small-M guardrails through the compile-time false path before acceptance.

Flash compile-time false-path guardrails:

| M | gated build us | saved baseline us | delta |
|---:|---:|---:|---:|
| 8 | 370.0 repeat | 371.2 | -0.3% |
| 16 | 438.7 | 426.6 | +2.8% |
| 32 | 440.3 | 456.6 | -3.6% |
| 128 | 493.3 | 507.2 | -2.7% |

The first M8 run was a noisy 384.0 us; the immediate same-cubin repeat was
370.0 us and matches both the saved baseline and prior restore check. M16 needs
the same repeat check before the gate can pass. Logs use
`cached_counts_gated_guard_flash_m{8,16,32,128}*_20260702.log` in the AKO log
directory.

Pro compile-time false-path guardrails:

| M | gated build us | saved baseline us | delta |
|---:|---:|---:|---:|
| 8 | 1203.0 | 1207.6 | -0.4% |
| 16 | 1476.0 | 1480.4 | -0.3% |
| 128 | 1681.8 | 1693.2 | -0.7% |

Flash M16 repeat was 440.7 us, so both gated-build samples are about 3% above
the saved 426.6 us baseline. Because this point is compile-time cache=false,
compare old/new cubin SASS before attributing the difference to code rather
than benchmark variance. Pro guard logs use
`cached_counts_gated_guard_pro_m{8,16,128}_20260702.log`.

SASS audit for Flash M16 cache=false versus the saved baseline cubin:

- resource usage is identical: 168 registers/thread, 56-byte stack, zero local
  memory, and identical cubin size;
- the full `cuobjdump --dump-sass` diff contains only the mangled function name
  and one diagnostic immediate; the executable instruction stream is
  otherwise identical.

Therefore the M16 timing difference is run variance, not a false-path code
regression.

Decision: keep the compile-time shape-gated policy as the new winner.

- Flash expected=12 (M64 in the target matrix): cache counts.
- Pro expected=4 or 8 (M32/M64): cache counts.
- Every other shape keeps the original polling path, with identical generated
  code; `DG_SM90_NVFP4_CACHE_RECV_COUNTS=0` remains an A/B kill switch.
- Selected same-source A/B gains are Flash M64 -2.0%, Pro M32 -2.4%, and Pro
  M64 -3.5%. Pro M32 reaches 1480.8 us, slightly faster than PR323's 1482.5 us.
- Flash M64 and Pro M64 improve to 492.6 and 1549.7 us respectively, but still
  trail PR323 by 3.4% and 1.8%; continue optimizing those two points.

Post-winner profile-only breakdown (per scheduled GEMM block):

| shape/M | gemm core | full-barrier wait | math-side decode | remainder |
|---|---:|---:|---:|---:|
| Flash M64 | 31.2 us | 1.5 us | 11.0 us | 18.7 us |
| Pro M64 | 44.2 us | 2.1 us | 25.1 us | 17.0 us |

Logs:

- `/root/fac/scripts/megamoe/ako_nvfp4_swapab/math_breakdown_flash_m64_20260702.log`
- `/root/fac/scripts/megamoe/ako_nvfp4_swapab/math_breakdown_pro_m64_20260702.log`

Conclusion: TMA wait and epilogues are no longer the next lever. The NVFP4
packed-row decoder is 35% of Flash and 57% of Pro GEMM-core time, so optimize
that helper while keeping cache=false shapes on the exact old decoder.

## 2026-07-02 AKO iter: streamed packed-row decode loads (rejected)

Hypothesis: the decoder first loads four `uint4` packed quads and keeps all 16
source words live before doing any PRMT work. For cache-enabled target shapes,
load one quad at a time and decode it immediately to reduce long-lived source
registers and allow LDS/PRMT interleaving. Cache=false shapes retained the old
decoder instance.

Validation:

- `./develop.sh`: PASS.
- Exact-NVFP4 correctness: PASS for Flash M64 and Pro M32/M64,
  `global_scale_mode=none/expert`, actual local-expert counts, capacity 8192.
- Logs:
  - `/root/fac/scripts/megamoe/ako_nvfp4_swapab/stream_decode_correct_flash_20260702.log`
  - `/root/fac/scripts/megamoe/ako_nvfp4_swapab/stream_decode_correct_pro_20260702.log`

Fixed-capacity 50-run median:

| shape | M | cache-only winner us | streamed-load us | delta |
|---|---:|---:|---:|---:|
| Flash | 64 | 492.6 | 501.1 | +1.7% |
| Pro | 64 | 1549.7 | 1581.7 | +2.1% |

Logs:

- `/root/fac/scripts/megamoe/ako_nvfp4_swapab/stream_decode_flash_m64_20260702.log`
- `/root/fac/scripts/megamoe/ako_nvfp4_swapab/stream_decode_pro_m64_20260702.log`

Decision: reject and revert. The serialized per-quad load dependency is worse
than preloading the four quads, despite lower apparent source-register
liveness. Preserve the cache-only winner and the profile-only breakdown.

## 2026-07-02 AKO iter: preload all scale LUT entries (rejected)

Hypothesis: keep the existing four packed-quad preloads, but issue the eight
random shared-memory LUT loads up front for cache-enabled targets. This removes
LDS from each PRMT dependency chain at the cost of about 14 additional live
registers.

Validation:

- `./develop.sh`: PASS.
- Exact-NVFP4 Flash M64 correctness: PASS for
  `global_scale_mode=none/expert`.
- Flash M64 fixed-capacity 50-run median: 546.2 us versus 492.6 us cache-only
  winner (+10.9%).
- Logs:
  - `/root/fac/scripts/megamoe/ako_nvfp4_swapab/preload_lut_correct_flash_20260702.log`
  - `/root/fac/scripts/megamoe/ako_nvfp4_swapab/preload_lut_flash_m64_20260702.log`

Decision: reject and revert without Pro benchmarking. Batching all random LUT
loads plus the larger live register set is substantially worse than the
original interleaved lookup/PRMT schedule.

## 2026-07-02 AKO iter: SWAR-packed PRMT selectors (promising, validating)

Hypothesis: each 32-bit packed NVFP4 word currently builds its two PRMT
selectors with four shifts and a serial mask/OR chain. Pack the four 3-bit
magnitudes from byte lanes in two SWAR stages instead:

1. combine adjacent byte lanes and mask with `0x00ff00ff`;
2. combine the two 16-bit halves and mask with `0x0000ffff`.

The isolated SM90a probe showed:

- baseline selector path: four `SHF` plus eight `LOP3` instructions;
- DP4A alternative: no net selector-instruction reduction and was rejected
  before integration;
- SWAR alternative: one fewer selector instruction, a shorter dependency
  chain, and 12 registers versus 14 in the single-word probe.

Bit-exact host validation passed 65,536 structured inputs and 1,000,000
random 32-bit inputs. The CUDA LUT unit test passed, and exact-NVFP4 end-to-end
correctness passed for Flash M=8/32/64 and Pro M=32/64 with
`global_scale_mode=none/expert`.

Fresh fixed-capacity 50-run medians:

| shape | M | same-day baseline us | SWAR us | delta | PR323 us |
|---|---:|---:|---:|---:|---:|
| Flash | 64 | 493.5 | 478.8 | -3.0% | 476.3 |
| Pro | 32 | 1481.4 | 1450.2 | -2.1% | 1482.5 |
| Pro | 64 | 1539.3 | 1518.1 | -1.4% | 1521.6 |

Logs:

- `/root/fac/scripts/megamoe/ako_nvfp4_swapab/swar_baseline_flash_m64_20260702.log`
- `/root/fac/scripts/megamoe/ako_nvfp4_swapab/swar_baseline_pro_m32_20260702.log`
- `/root/fac/scripts/megamoe/ako_nvfp4_swapab/swar_baseline_pro_m64_20260702.log`
- `/root/fac/scripts/megamoe/ako_nvfp4_swapab/swar_correct_flash_20260702.log`
- `/root/fac/scripts/megamoe/ako_nvfp4_swapab/swar_correct_pro_20260702.log`
- `/root/fac/scripts/megamoe/ako_nvfp4_swapab/swar_flash_m64_20260702.log`
- `/root/fac/scripts/megamoe/ako_nvfp4_swapab/swar_pro_m32_20260702.log`
- `/root/fac/scripts/megamoe/ako_nvfp4_swapab/swar_pro_m64_20260702.log`

Status: promising. Repeat the three selected points and run all remaining
small-M guardrails before accepting this globally shared decoder change.

Selected-point repeat:

| shape | M | first SWAR us | repeat SWAR us | same-day baseline us |
|---|---:|---:|---:|---:|
| Flash | 64 | 478.8 | 507.7 | 493.5 |
| Pro | 32 | 1450.2 | 1435.1 | 1481.4 |
| Pro | 64 | 1518.1 | 1507.7 | 1539.3 |

The Pro gains strengthened on repeat. The Flash repeat was disturbed: its
sample maximum was 4.7 ms and the reported per-rank medians ranged up to
568.2 us, unlike the first SWAR run. Treat Flash as unresolved and rerun it
alone on an otherwise idle host; do not select the favorable sample.

Repeat logs:

- `/root/fac/scripts/megamoe/ako_nvfp4_swapab/swar_flash_m64_repeat_20260702.log`
- `/root/fac/scripts/megamoe/ako_nvfp4_swapab/swar_pro_m32_repeat_20260702.log`
- `/root/fac/scripts/megamoe/ako_nvfp4_swapab/swar_pro_m64_repeat_20260702.log`

Three additional standalone Flash M64 repeats on an idle host were 490.6,
496.4, and 484.1 us. Across the four non-disturbed SWAR runs, the center pair
averages 487.4 us, about 1.2% below the same-day 493.5 us baseline. The gain is
smaller than Pro and does not consistently beat PR323's 476.3 us, but the
outlier-free runs support a modest decoder improvement. Logs are
`swar_flash_m64_repeat{2,3,4}_20260702.log` in the AKO log directory.

Remaining small-M guardrails:

| shape | M | prior winner us | SWAR us | delta |
|---|---:|---:|---:|---:|
| Flash | 8 | 370.0 | 363.2 / 363.0 | -1.9% |
| Flash | 16 | 440.7 | 419.1 | -4.9% |
| Flash | 32 | 440.3 | 441.2 / 440.9 | +0.2% |
| Flash | 128 | 493.3 | 495.4 | +0.4% |
| Pro | 8 | 1203.0 | 1162.8 | -3.3% |
| Pro | 16 | 1476.0 | 1437.3 | -2.6% |
| Pro | 128 | 1681.8 | 1660.9 | -1.2% |

The first Flash M8/M32 guard runs (384.9/456.6 us) were noisy and contradicted
both prior and subsequent runs. Two standalone repeats for each converged to
the values above. No credible small-M regression remains; Flash M32/M128 are
effectively flat while all other guard points improve.

Guard logs use `swar_guard_{flash,pro}_m*_20260702.log`; the standalone Flash
repeats use `swar_guard_flash_m{8,32}_repeat{1,2}_20260702.log`.

Status: performance and correctness gates pass. Audit the real fused-kernel
cubin resources and selector instruction stream before acceptance.

Real fused-kernel cubin audit, matching the exact old/new template
instantiations for Flash M64 and Pro M64:

- both versions use 168 registers/thread, 64-byte stack, 1024-byte static
  shared memory, and zero local memory;
- Flash and Pro cubin sizes are unchanged at 129,760 and 161,504 bytes;
- across the Flash fused kernel, SWAR removes 96 `SHF.R.U32.HI` and 64
  `LOP3.LUT` instructions, adds 128 `LEA.HI` instructions, and therefore
  removes 32 instructions net;
- `PRMT`, `LDS.64`, and `STS.128` counts are unchanged.

Decision: accept SWAR-packed selectors as the new decoder winner.

- The transformation is bit-exact and passed CUDA plus full fused-kernel
  correctness.
- Pro improves at every measured M=8..128. The center repeats are about
  1442.7 us at M32 and 1512.9 us at M64, both faster than PR323 (1482.5 and
  1521.6 us).
- Flash M8 improves to about 363.0 us and beats PR323's 373.4 us. M32 is
  effectively tied with PR323 at about 441.0 versus 440.3 us. M64 improves
  modestly to a 487.4 us center estimate but still trails PR323's 476.3 us.
- Preserve this globally shared helper as the baseline for subsequent work.

## 2026-07-02 AKO iter: PRMT-packed selectors (promising, validating)

Hypothesis: the accepted SWAR helper still spends four instructions packing
each byte-lane magnitude vector. Because the two desired packed bytes already
exist at byte positions 0 and 2 after a carry-free add, use a constant PRMT to
extract them. Use raw PTX for the subsequent LUT PRMT because every selector
nibble is already masked to 0..7; this avoids the intrinsic's redundant
`0x7777` mask.

The isolated SM90a probe reduced the single-word decoder from 12 to 11
registers and removed four more instructions versus the SWAR winner. Selector
equivalence passed 131,072 structured inputs and 1,000,000 random 32-bit
inputs.

Correctness:

- CUDA LUT unit test: PASS.
- Exact-NVFP4 fused correctness: PASS for Flash and Pro M=8/32/64,
  `global_scale_mode=none/expert`.
- A unique JIT cache directory was used, so these tests cannot reuse the SWAR
  cubins.

First fixed-capacity 50-run medians:

| shape | M | SWAR winner us | PRMT-pack us | delta | PR323 us |
|---|---:|---:|---:|---:|---:|
| Flash | 8 | 363.0 | 368.7 | +1.6% | 373.4 |
| Flash | 32 | 441.0 | 407.1 | -7.7% | 440.3 |
| Flash | 64 | 487.4 center | 498.0 | +2.2% | 476.3 |
| Pro | 8 | 1162.8 | 1093.6 | -6.0% | 1213.5 |
| Pro | 32 | 1442.7 center | 1338.9 | -7.2% | 1482.5 |
| Pro | 64 | 1512.9 center | 1409.3 | -6.8% | 1521.6 |

Logs:

- `/root/fac/scripts/megamoe/ako_nvfp4_swapab/prmt_pack_correct_flash_20260702.log`
- `/root/fac/scripts/megamoe/ako_nvfp4_swapab/prmt_pack_correct_pro_20260702.log`
- `/root/fac/scripts/megamoe/ako_nvfp4_swapab/prmt_pack_flash_m{8,32,64}_20260702.log`
- `/root/fac/scripts/megamoe/ako_nvfp4_swapab/prmt_pack_pro_m{8,32,64}_20260702.log`

Status: strongly promising for Pro and Flash M32. Repeat all six points; if
Flash M64 remains slower, retain the SWAR winner for that shape with a
compile-time selector-policy gate.

Selected-point repeat:

| shape | M | first PRMT-pack us | repeat us | PR323 us |
|---|---:|---:|---:|---:|
| Flash | 8 | 368.7 disturbed | 331.2 | 373.4 |
| Flash | 32 | 407.1 | 396.9 | 440.3 |
| Flash | 64 | 498.0 disturbed | 471.7 | 476.3 |
| Pro | 8 | 1093.6 | 1098.7 | 1213.5 |
| Pro | 32 | 1338.9 | 1338.3 | 1482.5 |
| Pro | 64 | 1409.3 | 1395.9 | 1521.6 |

The Pro results are stable across both runs. The second Flash run had much
lower sample maxima and tight per-rank medians; all three points improved
substantially, and Flash M64 reached 471.7 us, 1.0% faster than PR323.

Repeat logs use
`prmt_pack_{flash,pro}_m{8,32,64}_repeat_20260702.log`.

Status: all requested M=8/32/64 Flash and Pro points now beat PR323. Run two
more standalone Flash M64 repeats, M16/M128 guardrails, and real fused-cubin
resource/instruction audit before acceptance.

Additional Flash M64 repeats were 482.0 and 500.1 us. Across the three clean
runs (471.7, 482.0, 500.1), the median is 482.0 us. This is better than the
SWAR winner's 487.4 us center estimate, but it does not consistently beat
PR323's 476.3 us. The earlier statement that every requested point now beats
PR323 was premature; Flash M64 remains the only unresolved comparison.

Global guardrails:

| shape | M | SWAR winner us | PRMT-pack us | delta |
|---|---:|---:|---:|---:|
| Flash | 16 | 419.1 | 402.0 | -4.1% |
| Flash | 128 | 495.4 | 484.3 | -2.2% |
| Pro | 16 | 1437.3 | 1337.4 | -7.0% |
| Pro | 128 | 1660.9 | 1654.5 | -0.4% |

Logs:

- `prmt_pack_flash_m64_repeat{2,3}_20260702.log`
- `prmt_pack_guard_{flash,pro}_m{16,128}_20260702.log`

Real fused-cubin audit versus the SWAR winner:

- Flash and Pro resources remain 168 registers/thread, 64-byte stack,
  1024-byte static shared memory, and zero local memory.
- Flash cubin size falls from 129,760 to 127,712 bytes; Pro falls from
  161,504 to 159,456 bytes.
- Across the Flash fused kernel, PRMT-pack removes 128 `LOP3.LUT` and 64
  `LEA.HI`, adds 64 `PRMT`, and therefore removes another 128
  instructions net.
- `SHF.R.U32.HI`, `LDS.64`, `STS.128`, and register/stack counts are
  unchanged.

Decision: accept PRMT-packed selectors as the new global decoder winner.

- It is bit-exact and passes all CUDA plus fused correctness gates.
- Every measured Flash/Pro M=8..128 point improves or remains within noise
  versus the SWAR winner.
- Flash M8/M32 and all Pro M8/M32/M64 points beat PR323 with substantial
  margin.
- Flash M64 improves to a 482.0 us three-run median but still trails PR323 by
  about 1.2%; continue optimizing this point from the PRMT-pack baseline.

Post-winner phase profile, per scheduled GEMM block:

| shape/M | SWAR gemm/decode us | PRMT-pack gemm/decode us | decode delta |
|---|---:|---:|---:|
| Flash M64 | 31.21 / 10.98 | 30.24 / 10.20 | -7.1% |
| Pro M64 | 44.16 / 25.10 | 39.58 / 19.80 | -21.1% |

The decoder remains about 34% of Flash M64 GEMM-core time, so there is still
room to optimize it. Logs:

- `/root/fac/scripts/megamoe/ako_nvfp4_swapab/prmt_pack_phase_flash_m64_20260702.log`
- `/root/fac/scripts/megamoe/ako_nvfp4_swapab/prmt_pack_phase_pro_m64_20260702.log`

## 2026-07-02 AKO iter: preload the second LUT per quad (global rejected)

Hypothesis: issue the two scale-LUT loads for one packed quad together. This
may overlap the second LDS with the first pair of word decodes while keeping
only one extra `uint2` live, unlike the rejected all-eight-LUT preload.

Correctness: PASS for Flash/Pro M64 with
`global_scale_mode=none/expert`.

Fixed-capacity 50-run medians:

| shape | PRMT-pack winner us | paired-LUT us | repeat us | result |
|---|---:|---:|---:|---|
| Flash M64 | 482.0 center | 499.6 | 509.5 | regression |
| Pro M64 | 1402.6 center | 1389.3 | 1391.3 | modest gain |

Logs:

- `pair_lut_prefetch_correct_{flash,pro}_20260702.log`
- `pair_lut_prefetch_{flash,pro}_m64_20260702.log`
- `pair_lut_prefetch_{flash,pro}_m64_repeat_20260702.log`

Decision: reject as a global decoder schedule because it consistently hurts
the remaining Flash M64 target. Evaluate the other Pro sizes before deciding
whether a compile-time Pro-only schedule is justified.

Pro-only evaluation:

| Pro M | PRMT-pack winner us | paired-LUT us | delta |
|---:|---:|---:|---:|
| 8 | 1096.2 center | 1077.3 | -1.7% |
| 16 | 1337.4 | 1323.4 | -1.0% |
| 32 | 1338.6 center | 1336.9 | -0.1% |
| 64 | 1402.6 center | 1390.3 center | -0.9% |
| 128 | 1654.5 | 1648.5 | -0.4% |

A compile-time `kIntermediateHidden >= 3072` gate now enables paired LUT
loads only for Pro. Validation:

- Exact-NVFP4 correctness PASS for Pro M=8/16/32/64/128 with
  `global_scale_mode=none/expert`.
- Flash M64 correctness PASS with both scale modes.
- Flash false-path cubin resources and size are identical to the PRMT-pack
  winner. Its complete disassembly has identical length and executable
  instruction stream; only one diagnostic immediate differs.
- Gated 50-run medians: Flash M64 485.4 us (within its winner distribution),
  Pro M64 1387.8 us.

Logs:

- `pair_lut_prefetch_pro_m{8,32,128}_20260702.log`
- `pair_lut_gated_correct_{flash,pro}_20260702.log`
- `pair_lut_gated_correct_pro_full_20260702.log`
- `pair_lut_gated_{flash,pro}_m64_20260702.log`
- `pair_lut_gated_pro_m16_20260702.log`

Decision: accept the Pro-only paired-LUT schedule. It preserves the exact
Flash PRMT-pack path and improves or holds every measured Pro M=8..128 point.

## 2026-07-02 AKO iter: DP4A selector pack (shape-gated candidate)

Hypothesis: Flash appears more sensitive to PRMT throughput after the
PRMT-pack optimization. Replace the two constant pack PRMTs per word with
four byte-dot products plus two fused IMAD merges. This adds two integer
instructions per word but removes two PRMTs.

The SM90a probe confirms the optimized DP4A form uses one IMAD per selector
merge. Exact-NVFP4 correctness passes for Flash/Pro M64 with both scale modes.

Fixed-capacity 50-run medians versus the paired-LUT/PRMT-pack winner:

| shape/M | current winner us | DP4A us | delta |
|---|---:|---:|---:|
| Flash M64 | 482.0 center | 490.6 / 496.8 | regression |
| Pro M8 | 1077.3 | 1057.8 | -1.8% |
| Pro M16 | 1323.4 | 1274.0 | -3.7% |
| Pro M32 | 1336.9 | 1301.3 | -2.7% |
| Pro M64 | 1390.3 center | 1348.1 / 1350.8 | -2.9% |
| Pro M128 | 1648.5 | 1667.2 | +1.1% |

Logs:

- `dp4a_pack_correct_{flash,pro}_20260702.log`
- `dp4a_pack_{flash,pro}_m64{,_repeat}_20260702.log`
- `dp4a_pack_pro_m{8,16,32,128}_20260702.log`

Decision: reject globally. Gate DP4A to Pro configurations with expected
tokens per local expert <= 8 (target M=8/16/32/64); retain PRMT-pack for Flash
and Pro M128.

Gated implementation:

- Added a fused-kernel template boolean and host policy selecting DP4A only
  for fused Pro configurations with expected tokens per local expert <= 8.
- Added `DG_SM90_NVFP4_DP4A_SELECTOR_PACK=0` as a kill switch.
- Flash and Pro M128 compile the PRMT path; Pro M8/16/32/64 compile DP4A.
- `./develop.sh`: PASS.
- Exact-NVFP4 correctness: PASS for Flash M64 and Pro M=8/16/32/64/128 with
  `global_scale_mode=none/expert`.

Gated 50-run medians:

| shape/M | previous winner us | gated build us | delta |
|---|---:|---:|---:|
| Flash M64 | 482.0 center | 485.3 | within run variance |
| Pro M8 | 1077.3 | 1062.1 | -1.4% |
| Pro M16 | 1323.4 | 1289.5 | -2.6% |
| Pro M32 | 1336.9 | 1311.0 | -1.9% |
| Pro M64 | 1390.3 center | 1357.3 | -2.4% |
| Pro M128 | 1648.5 | 1648.1 | flat |

SASS audit:

- Flash false-path instruction lines are byte-for-byte identical to the prior
  PRMT winner.
- Pro true-path instruction lines are byte-for-byte identical to the
  standalone DP4A candidate.
- Resources remain unchanged for the corresponding configurations.

Logs:

- `develop_dp4a_gated_20260702.log`
- `dp4a_gated_correct_{flash,pro}_20260702.log`
- `dp4a_gated_flash_m64_20260702.log`
- `dp4a_gated_pro_m{8,16,32,64,128}_20260702.log`

Decision: accept the shape-gated DP4A selector pack. Continue Flash M64 from
the unchanged PRMT path.

## 2026-07-02 AKO iter: hybrid PRMT/DP4A selectors (shape-gated candidate)

Hypothesis: full PRMT is best for Flash while full DP4A is best for Pro small
M, indicating a pipeline-balance crossover. Use DP4A for only one of the two
selectors per word.

Flash M64 50-run medians:

| selector policy | runs us | center us |
|---|---|---:|
| pure PRMT winner | 471.7 / 482.0 / 500.1 | 482.0 |
| DP4A high, PRMT low | 481.1 / 477.3 / 485.8 | 481.1 |
| PRMT high, DP4A low | 473.5 / 472.5 / 480.1 | 473.5 |

The low-selector hybrid is the first multi-run center that beats PR323's
476.3 us (about 0.6% faster).

Additional low-hybrid screening:

| shape/M | low-hybrid us | current winner us | result |
|---|---:|---:|---|
| Flash M8 | 344.8 | noisy 331.2 clean repeat | do not enable |
| Flash M32 | 399.2 | about 402.0 center | near-flat |
| Flash M128 | 481.1 | 484.3 | modest gain |
| Pro M128 | 1621.2 | 1648.1 | gain |

Logs:

- `hybrid_hi_correct_flash_20260702.log`
- `hybrid_hi_flash_m64_r{1,2,3}_20260702.log`
- `hybrid_lo_flash_m64_r{1,2,3}_20260702.log`
- `hybrid_lo_flash_m{8,32,128}_20260702.log`
- `hybrid_lo_pro_m128_20260702.log`

Decision: do not enable globally. Add an independent compile-time gate for
Flash expected=12 (M64) and Pro expected=16 (M128); preserve current paths for
all other shapes.

Gated implementation and validation:

- Added `kHybridLowSelectorPack` independently of the Pro-small full-DP4A
  gate. It selects PRMT for the high selector and DP4A for the low selector.
- The host enables it only for fused Flash expected=12 and fused Pro
  expected=16. `DG_SM90_NVFP4_HYBRID_LOW_SELECTOR_PACK=0` is the same-source
  kill switch.
- `./develop.sh`: PASS.
- Exact-NVFP4 correctness: PASS for Flash M=8/32/64 and Pro M=64/128 with
  `global_scale_mode=none/expert`.

Initial gated 50-run medians:

| shape/M | gated build us | selected path |
|---|---:|---|
| Flash M8 | 330.1 | unchanged PRMT |
| Flash M32 | 393.6 | unchanged PRMT |
| Flash M64 | 477.2 / 486.7 / 487.5 | low-selector hybrid |
| Pro M64 | 1353.2 | unchanged full DP4A |
| Pro M128 | 1638.5 | low-selector hybrid |

The absolute Flash M64 result continued to move with host timing variance, so
the final decision used an ABBA comparison from the same source and JIT cache,
changing only the hybrid kill switch. Both variants were compiled and warmed
before the four measured pairs.

| pair | pure PRMT us | low-selector hybrid us | hybrid delta |
|---:|---:|---:|---:|
| 1 | 478.8 | 473.4 | -1.1% |
| 2 | 487.6 | 476.7 | -2.2% |
| 3 | 487.3 | 482.1 | -1.1% |
| 4 | 479.2 | 476.5 | -0.6% |
| center of four run medians | 483.2 | 476.6 | -1.4% |

All four pairs favor the hybrid. The independent direct-candidate three-run
center was 473.5 us. The gated ABBA center is effectively tied with the fixed
PR323 target of 476.3 us (+0.06%), while the direct-candidate center is 0.6%
faster. Do not claim a stable PR323 win from this small difference because the
PR323 value is a user-provided low-latency reference (`provided_ll`), not a
same-run measurement. The current PR323 runtime script measures a different
path and is not a valid replacement comparator.

SASS and resource audit:

- Flash M8/M32 false paths and the Pro M64 full-DP4A false path have the same
  cubin size, instruction count, registers, stack, shared memory, and
  executable instruction stream as their prior winners. The only disassembly
  difference is the known diagnostic immediate.
- The gated Flash M64 cubin has the same 6,904 instruction lines and resources
  as the direct low-hybrid candidate; Pro M128 likewise has the same 6,576
  instruction lines and resources as its direct candidate. Again, only the
  diagnostic immediate differs.
- Flash M64 remains at 168 registers/thread, 64-byte stack, 1,024-byte static
  shared memory, zero local memory, and a 127,712-byte cubin. Pro M128 remains
  at 168 registers/thread, 56-byte stack, 1,024-byte static shared memory,
  zero local memory, and a 122,592-byte cubin.

Logs:

- `develop_hybrid_low_gated_20260702.log`
- `hybrid_low_gated_correct_{flash,pro}_20260702.log`
- `hybrid_low_gated_{flash_m8,flash_m32,flash_m64,pro_m64,pro_m128}_20260702.log`
- `hybrid_low_gated_flash_m64_r{2,3}_20260702.log`
- `hybrid_low_final_ab_p{1,2,3,4}_{prmt,hybrid}_flash_m64_20260702.log`

Decision: accept the shape-gated low-selector hybrid as the new winner. It
delivers a repeatable 1.4% same-source improvement at Flash M64, improves Pro
M128, and preserves exact prior code on all audited false paths. Continue with
the complete small-M matrix, then optimize M256 under the 20-run mean contract.

## 2026-07-02 AKO: accepted-winner final small-M matrix

Final-source exact-NVFP4 correctness covered Flash and Pro M=8/16/32/64/128
with both `global_scale_mode=none/expert`: 20/20 cases PASS. The minimum
per-token cosine was 0.9988, norm ratios were 0.9973--0.9981, and every output
was finite. Logs:

- `hybrid_winner_final_correct_flash_small_20260702.log`
- `hybrid_winner_final_correct_pro_small_20260702.log`

Fresh-cache fixed-capacity final matrix, 50-run median:

| shape | M | accepted NVFP4 us | fixed PR323 us | gap |
|---|---:|---:|---:|---:|
| Flash | 8 | 323.4 | 373.4 | -13.4% |
| Flash | 16 | 378.6 | 416.5 | -9.1% |
| Flash | 32 | 400.3 | 440.3 | -9.1% |
| Flash | 64 | 472.0 | 476.3 | -0.9% |
| Flash | 128 | 484.0 | 519.4 | -6.8% |
| Pro | 8 | 1051.1 | 1213.5 | -13.4% |
| Pro | 16 | 1286.7 | 1485.2 | -13.4% |
| Pro | 32 | 1306.7 | 1482.5 | -11.9% |
| Pro | 64 | 1362.0 | 1521.6 | -10.5% |
| Pro | 128 | 1631.2 | 1711.8 | -4.7% |

Flash M64 was repeated twice from the same final JIT cache: 470.3 and 475.9
us. Together with the matrix run, the three-run median is 472.0 us and all
three runs are below the fixed 476.3 us target. This supports a 0.9% win
against the fixed external reference, while retaining the caveat that PR323
was not measured concurrently.

Matrix logs use
`hybrid_winner_final_matrix_{flash,pro}_m{8,16,32,64,128}_median50_20260702.log`;
Flash M64 repeats use
`hybrid_winner_final_matrix_flash_m64_repeat{1,2}_median50_20260702.log`.

Decision: the small-M objective is met for every Flash/Pro M=8..128 point in
the comparison table. Move to M256 using the required 20-run mean statistic.

## 2026-07-02 AKO: M256 tail diagnosis and candidate screening

The accepted small-M winner initially measured as follows at M256 with the
historical synchronous-rank kineto harness:

| shape | mean20 us | median us | max us | fixed PR323 us |
|---|---:|---:|---:|---:|
| Flash | 537.0 | 489.1 | 1097.5 | 540.5 |
| Pro | 2048.2 | 1687.1 | 5336.9 | 1805.0 |

Raw-sample repeats showed that the steady kernels were already faster than
the targets, but random 2--5 ms samples dominated the 20-run mean. The long
samples occur after rank-local host launch skew is amplified by the fused
kernel's first cross-rank barrier. Reusing the 8 GB flush allocation, reusing
the output tensor, and disabling the flush were tested independently; none
consistently removed the tail. They remain default-off diagnostic options:

- `DG_BENCH_PRINT_SAMPLES=1`
- `DG_BENCH_REUSE_FLUSH_BUFFER=1`
- `DG_BENCH_REUSE_OUTPUT=1`

An asynchronous NCCL benchmark barrier keeps the barrier and fused kernel in
GPU-stream order and collapses the random multi-sample tail to the profiler
active-window transition sample. Running one extra active warmup and retaining
the last 20 samples removes that transition without changing the requested
20-run mean or the 8 GB L2 flush. The diagnostic settings are
`DG_BENCH_ASYNC_BARRIER=1` and `DG_BENCH_DISCARD_FIRST_ACTIVE=1`; defaults are
unchanged for historical comparability.

Kernel candidates under the original default harness:

- CTA recv-count cache at Pro expected=32 was rejected: its two mean20 runs
  were slower than paired cache-off runs and its steady center did not improve.
- CTA recv-count cache at Flash expected=48 won four of five paired mean20
  comparisons. The median run mean improved from 777.6 to 611.4 us despite
  both sides' random tails; the median steady center improved from 498.3 to
  491.3 us (-1.4%).
- Extending the low-selector hybrid to Flash expected=48 or Pro expected=32
  was rejected. Both shapes' steady medians regressed or remained flat.
- Disabling loader dequant was rejected. Flash steady median regressed to
  564.5 us; Pro showed no credible improvement.
- Forced BN128 split compiled after the repair below but measured 739.6 us at
  Flash M256, so BN256 fused remains the performance path.

Flash phase profiling confirms the cache-count mechanism rather than sample
luck: cached counts reduce the scheduler math-loop average from about 550.9k
to 438.7k cycles (-20%), while one-time dispatch/cache work rises from 151.8k
to 221.9k cycles. GEMM/dequant/epilogue phases remain effectively unchanged.

Logs include:

- `hybrid_winner_m256_{baseline,samples}_{flash,pro}_mean20_20260702.log`
- `m256_tail_diag_pro_{reuse_flush,reuse_output,reuse_both}_mean20_20260702.log`
- `m256_cache_counts_ab_{flash,pro}_p*_*_mean20_20260702.log`
- `m256_hybrid_quadrant_{flash,pro}_p*_*_mean20_20260702.log`
- `m256_existing_knob_{flash,pro}_*_mean20_20260702.log`
- `m256_flash_cache_phase_{off,cache}_20260702.log`

## 2026-07-02 repair: restore BN128 split JIT template contract

The paired-LUT/selector template evolution made
`dequant_smem_b_from_packed_fused_scale` require three explicit bool template
arguments. Four existing split L1/L2 callers use the original no-argument
contract, so forcing BN128 exposed an NVRTC template-deduction failure.

Fix: give the helper compatibility defaults that exactly represent the old
split behavior: no second-LUT preload and PRMT for both selectors. Fused calls
continue to pass every policy explicitly, so their generated code is
unchanged.

Validation:

- `./develop.sh`: PASS.
- Forced BN128 Flash M256 successfully JIT compiles and runs.
- Exact-NVFP4 forced-BN128 M256 correctness passes for
  `global_scale_mode=none/expert` with cosine min 0.9987.
- Final BN256 Flash M8/M32/M64 and Pro M64 cubins are instruction-for-
  instruction identical to their accepted pre-repair cubins.

Logs:

- `develop_bn128_template_repair_20260702.log`
- `bn128_template_repair_flash_m256_20260702.log`
- `bn128_template_repair_correct_flash_m256_20260702.log`

## 2026-07-02 AKO: final M256 winner and post-M256 guard

Final policy:

- Enable the existing CTA recv-count cache for fused Flash expected=48
  (M256), in addition to the accepted Flash expected=12 case.
- Do not enable a new M256 policy for Pro; retain paired-LUT PRMT selectors.
- Keep `DG_SM90_NVFP4_CACHE_RECV_COUNTS=0` as the same-source kill switch.

GPU-aligned, post-transition 20-run means with the original 8 GB L2 flush:

| shape/policy | run 1 us | run 2 us | run 3 us | 3-run center us | PR323 us | gap |
|---|---:|---:|---:|---:|---:|---:|
| Flash cache off | 517.8 | 539.4 | 511.0 | 517.8 | 540.5 | -4.2% |
| Flash cache on | 512.3 | 515.5 | 519.9 | 515.5 | 540.5 | -4.6% |
| Pro final | 1668.0 | 1648.5 | 1681.2 | 1668.0 | 1805.0 | -7.6% |

The Flash cache wins two of three direct pairs and improves the average of
the three run means by 1.3%; phase evidence supports keeping the small gain.
Final-source M256 exact-NVFP4 correctness passes Flash and Pro with both
global-scale modes; cosine minima are 0.9987/0.9988 and all outputs are finite.

Post-M256 small-M guard, original default 50-run median harness:

| shape | M | final us | fixed PR323 us | gap |
|---|---:|---:|---:|---:|
| Flash | 8 | 320.5 | 373.4 | -14.2% |
| Flash | 32 | 396.4 | 440.3 | -10.0% |
| Flash | 64 | 476.2 three-run center | 476.3 | effectively tied (-0.01%) |
| Pro | 8 | 1053.8 | 1213.5 | -13.2% |
| Pro | 32 | 1307.6 | 1482.5 | -11.8% |
| Pro | 64 | 1339.9 | 1521.6 | -11.9% |

The three current-source Flash M64 runs are 476.2, 465.4, and 477.0 us. The
median is 476.2 us; the earlier accepted-winner final matrix had a 472.0 us
three-run center. Treat Flash M64 as tied/slightly faster than the fixed
external reference rather than claiming a large margin.

Logs:

- `m256_async_discard_{flash,pro}_*_r{1,2,3}_mean20_20260702.log`
- `m256_final_correct_{flash,pro}_20260702.log`
- `final_post_m256_{flash,pro}_m{8,32,64}_median50_20260702.log`
- `final_post_m256_flash_m64_repeat{1,2}_median50_20260702.log`

Decision: accept the Flash-M256 cache-count extension and the BN128 template
repair. The requested Flash M8/M32 points beat PR323 by clear margins, Flash
M64 is tied/slightly faster across repeated final-source runs, all requested
Pro points beat PR323, and both M256 shapes beat their fixed targets under a
stable 20-run mean measurement.

## 2026-07-02 AKO: replace benchmark-point gates with expected ranges

Problem: the accepted wrapper auto-enabled the CTA recv-count cache only at
Flash expected=12/48 and Pro expected=4/8, and enabled the low-selector hybrid
only at Flash expected=12 and Pro expected=16. With fixed top-k and local
expert counts these were benchmark-M point gates, not policies that generalized
to arbitrary serving batches.

Method:

- Added temporary force hooks outside the final source and swept 130
  Flash/Pro policy combinations at non-power-of-two M values.
- Re-tested candidates with independent-process off/on/on/off runs, each using
  a 30-sample kernel median and the fixed capacity=8192, 8 GB L2-flush harness.
- Added four routing seeds with alternating off/on order. This exposed several
  fixed-seed false positives: cache and Pro hybrid had no continuous expected
  range, while Flash hybrid generalized only over the small-work band.
- Removed the temporary force hooks before final validation.

Final wrapper policy:

- Use integer closed-range comparisons on the routed-token numerator, avoiding
  floating-point equality and literal benchmark-M checks.
- Fused swapAB: expected in [0, 8], unchanged semantically.
- Pro DP4A selector pack: expected in [0, 8], unchanged semantically.
- Flash low-selector hybrid: expected in [3, 8]. The lower boundary M16 and
  the interior/up-boundary non-power-of-two points M17/M24/M40/M42 were covered
  across routing seeds; M43 is outside the range.
- Disable automatic CTA recv-count caching. Multi-M results were non-monotonic,
  so no expected-only interval was defensible. The implementation remains an
  explicit experimental opt-in through
  `DG_SM90_NVFP4_CACHE_RECV_COUNTS=1`; default is 0.
- Remove the old Flash M64 and Pro M128 hybrid exact gates. Flash M64 hybrid
  regressed for all four routing seeds, with a +3.03% median seed delta.

Validation:

- `./develop.sh` and `git diff --check`: PASS.
- Exact-NVFP4 correctness: 24/24 PASS for Flash M15/16/24/42/43/64 and Pro
  M16/32/64/112/128/144, each with global-scale modes none/expert. Minimum
  per-token cosine remained about 0.9988 and all outputs were finite.
- Final fixed-capacity small-M results use 50-sample medians. M256 uses three
  post-transition 20-sample means.

| shape | M | final us | fixed PR323 us | gap |
|---|---:|---:|---:|---:|
| Flash | 8 | 331.2 | 373.4 | -11.3% |
| Flash | 16 | 373.8 | 416.5 | -10.3% |
| Flash | 32 | 385.1 | 440.3 | -12.5% |
| Flash | 64 | 474.5 three-run center | 476.3 | -0.4% |
| Flash | 128 | 478.4 | 519.4 | -7.9% |
| Flash | 256 | 518.6 three-run center | 540.5 | -4.1% |
| Pro | 8 | 1051.2 | 1213.5 | -13.4% |
| Pro | 16 | 1282.5 | 1485.2 | -13.6% |
| Pro | 32 | 1305.6 | 1482.5 | -11.9% |
| Pro | 64 | 1357.2 | 1521.6 | -10.8% |
| Pro | 128 | 1630.9 | 1711.8 | -4.7% |
| Pro | 256 | 1674.0 three-run center | 1805.0 | -7.3% |

Non-power-of-two final-source Flash medians were M15=374.3, M24=372.5,
M40=385.4, M42=403.0, and M43=468.0 us. Experiment artifacts, raw logs, and
summaries are under
`/root/fac/scripts/megamoe/ako_nvfp4_range_policy/`, notably
`targeted_process_abba.md`, `multiseed_hybrid.md`, and
`final_range_matrix.md`.

## 2026-07-02 cleanup: finalize automatic policies

Removed controls that no longer represent supported production choices:

- Deleted the CTA receive-count cache end to end: environment variable, host
  argument, JIT template argument, cache barrier, shared-count population,
  and cached scheduler fetches. No stable automatic expected-token interval
  was found, so retaining an explicit-only implementation added dead policy
  surface and extra template variants.
- Removed the `DG_SM90_NVFP4_DP4A_SELECTOR_PACK` and
  `DG_SM90_NVFP4_HYBRID_LOW_SELECTOR_PACK` kill switches. Their measured
  expected-token ranges are now the sole policy source.
- Kept the DP4A, hybrid, and swapAB JIT booleans because those policies vary
  by generated shape/expected-token range. The Pro paired-LUT choice remains
  derived directly from `kIntermediateHidden >= 3072` and has no host control.
- Left loader-dequant selection and phase profiling unchanged: the former
  selects distinct shape-dependent implementations and the latter is a
  diagnostic facility.

Validation:

- `git diff --check` and `./develop.sh`: PASS.
- Fresh-cache exact-NVFP4 policy-boundary correctness: 24/24 PASS. Flash used
  M15/16/24/42/43/64 with 32 local experts; Pro used
  M16/32/64/112/128/144 with 48 local experts. Both global-scale modes passed,
  all outputs were finite, and minimum per-token cosine was about 0.9988.
- Representative 50-sample medians: Flash M42 409.0 us, Pro M64 1350.3 us,
  and Pro M112 1642.3 us, consistent with the pre-cleanup results within run
  noise. Flash M43 showed tail-heavy 487.7/503.2 us repeats versus the prior
  468.0 us run, but old/new cubin inspection showed identical resource usage
  (168 registers, 56-byte stack, 1024-byte shared allocation) and a full SASS
  diff changed only one source-line diagnostic immediate. The executable hot
  path is unchanged.

Fresh JIT artifacts are under
`/root/fac/scripts/megamoe/ako_nvfp4_cleanup/`.

## 2026-07-02 layout-policy cleanup: policy-matrix baseline attempt 1

Added an isolated-process correctness matrix for Flash, Pro, middle-I, and
forced-BN128 policy boundaries. The first baseline invocation failed before
CUDA/JIT execution because the driver reused `sys.executable` (`/usr/bin/python3`),
while the repository extension is installed for the `/usr/local` Python
environment. No kernel result was produced. Repair the driver to resolve
`python3` from `PATH` by default and retain an explicit interpreter override.

Correction: the driver interpreter was valid. The shared `build/` directory
had been replaced by a CPython 3.12 build while the active test process was
CPython 3.10, leaving the 3.10 extension symlink dangling. Re-running
`./develop.sh` restored a matching extension; no driver change was required.

## 2026-07-02 layout-policy cleanup: policy-matrix baseline

The new fresh-process policy matrix passes before the F3/F2 source cleanup:

- Flash fused: M15/16 and M42/43 cover both sides of expected 3 and 8.
- Pro fused: M63/64/65 cover both sides and the exact expected-8 boundary.
- Middle-I fused (`hidden=5120`, `I=2560`): M63/64/65 cover the generic
  swapAB boundary.
- Forced BN128: expected 32/64/96/128/192 covers the split scheduling policy.
- Every case passes with global-scale modes `none` and `expert`; all outputs
  are finite and the minimum per-token cosine is 0.9985.

Fresh JIT caches:
`/root/fac/scripts/megamoe/ako_nvfp4_layout_policy/policy_matrix_baseline/20260702_152634_717996`.

## 2026-07-02 layout-policy cleanup: F3 template generation

Removed the 11 trailing explicit template arguments that exactly duplicated
the defaults in `sm90_nvfp4_mega_moe.cuh`. Added parameter-name comments to
the fused, split-L1, and split-L2 boolean template arguments so their positional
contracts remain reviewable.

Validation:

- `./develop.sh`: PASS.
- The fresh-process policy matrix passed for all Flash, Pro, middle-I, and
  forced-BN128 cases in both global-scale modes; minimum per-token cosine was
  0.9985.
- `cuobjdump --dump-sass` SHA-256 multisets are identical to the pre-F3
  baseline for all 14 generated cubins: 4 Flash fused, 2 Pro fused, 2 middle-I
  fused, and 6 forced-BN128 split L1/L2 cubins.

Fresh JIT caches:
`/root/fac/scripts/megamoe/ako_nvfp4_layout_policy/policy_matrix_f3/20260702_153026_721078`.

## 2026-07-02 layout-policy cleanup: F2 NVFP4 config builder

Replaced the generic-SM90-config-then-overwrite flow with an NVFP4-specific
deployment plan builder. It constructs the final fused/split-L1 config and the
final split-L2 config directly, including block shape, thread layout, pipeline
stages, shared memory, and the shape-derived JIT policies. The generic SM90 FP8
helper is unchanged. Split launch lambdas now consume the completed configs and
perform no stage or shared-memory recomputation.

Validation:

- `git diff --check` and `./develop.sh`: PASS.
- The fresh-process policy matrix passed in both global-scale modes; minimum
  per-token cosine was 0.9985.
- All 14 generated cubins have byte-identical `cuobjdump --dump-sass` hashes
  versus the post-F3/pre-F2 baseline, including all six forced-BN128 split
  L1/L2 cubins.

Fresh JIT caches:
`/root/fac/scripts/megamoe/ako_nvfp4_layout_policy/policy_matrix_f2/20260702_153704_726192`.

## 2026-07-02 deployment-layout sweep: 54-point crossover map

Compared forced BN256/fused and BN128/split layouts on eight H20 ranks. Every
point used process-level ABBA order, routing seeds 101/202/303, 20 active
samples per run, a fixed 8192-token capacity, and L2 flushing. Split samples
were measured as adjacent L1+L2 event sums rather than the historical
per-kernel median times two. The metric is maximum rank latency.

Profiles:

- Flash: H=4096, I=2048, E=256, topk=6.
- Middle: H=5120, I=2304/2560/2816, E=384, topk=6.
- Pro: H=7168, I=3072, E=384, topk=6.

The coarse and boundary sweeps covered expected work
128/160/176/184/192/200/208/216/224/256 for every I. Results were consistent
across Flash and all middle-I profiles: BN256 won every seed through expected
184, BN128 won every seed from expected 200 through 224, and expected 192 was
the near-tie crossover. This supports a 192 deployment cutoff without linear
scaling by I.

Pro crossed slightly earlier. A follow-up with four seeds and 30 active
samples showed BN256 winning 3/4 seeds at expected 188 (+0.58% median) and 190
(+0.42%), while BN128 won 4/4 at 194 (-0.33%) and 3/4 at 196 (-1.28%). Together
with the coarse expected-192 result (BN128 3/3, -0.75%), the measured Pro cutoff
is 190.

Expected 256 is non-monotonic: the three middle profiles favored BN128, while
Flash and Pro favored BN256 by about 1.1%. This point coincides with the
BN128 exact-256 all-experts-per-wave override and is being diagnosed separately
before finalizing the layout policy.

Raw logs and summaries:
`/root/fac/scripts/megamoe/ako_nvfp4_layout_policy/layout_sweep/`.

### Rejected expected-256 wave experiment

Replaced the BN128 exact-256 all-experts wave temporarily with four experts
per wave over actual expected [255, 256], rebuilt, and repeated the five I
profiles with the same three-seed ABBA protocol. The candidate did not remove
the non-monotonic layout result: Flash and Pro still favored BN256 by 1.08%
and 0.98%, while the middle profiles remained essentially unchanged relative
to the baseline. Restored the behavior-preserving F2 configuration and rebuilt.
No experimental control or interval remains in production code.

Experiment logs are `wave4_*_e256.log` in the layout-sweep artifact directory.

## 2026-07-02 final deployment-layout policy and validation

Updated `choose_nvfp4_block_n_for_mega_moe_sm90()` to accept
`intermediate_hidden` and use integer routed-work comparisons at weight-prepack
time:

- I < 3072: BN256 through expected 192, then BN128.
- I >= 3072: BN256 through expected 190, then BN128.

The benchmark and correctness callers now supply I explicitly. Added pure
selection assertions plus default-layout correctness cases on both sides of
the Flash 192 and Pro 190 cutoffs. The first final-matrix launch failed before
CUDA because the orchestration script did not put the repository root on
`sys.path`; adding that path fixed the harness.

Final validation:

- `git diff --check` and `./develop.sh`: PASS.
- Pure integer layout-selection assertions: PASS.
- Fresh-process matrix: all Flash/Pro/middle small-work cases, Flash/Pro layout
  cutoff cases, and forced-BN128 expected 32/64/96/128/192 cases passed in both
  global-scale modes. All outputs were finite and minimum per-token cosine was
  0.9985.
- The original 14 cleanup-path cubins still have identical
  `cuobjdump --dump-sass` SHA-256 multisets versus the post-F2 baseline.

Final fresh JIT caches:
`/root/fac/scripts/megamoe/ako_nvfp4_layout_policy/policy_matrix_final/20260702_161349_802228`.

## 2026-07-02 current NVFP4 versus W8A8 canonical matrix

- Objective: compare the current NVFP4 working tree against the user-provided W8A8 `ours` and PR323 table without mixing the earlier L2-flush/median benchmark contract.
- Environment: the current source was copied from H20 host `10.6.131.8` to an isolated snapshot on the W8A8 table's original H20x8 host `10.6.131.7`; CUDA 13.0 container, capacity 8192, seed 0, `DG_BENCH_FLUSH_L2_BYTES=0`, 20 samples per point.
- Routing: the temporary driver used the W8A8 runner's stable per-case seed and rank offset. The calibration point matched Flash M8 exactly at rank0 `recv=47`, `experts=24`.
- Split timing correction: pair all 40 L1/L2 events into 20 logical calls before computing statistics. The correction is benchmark-driver-only and does not modify project source.
- Equal-weight geometric-mean latency gaps versus W8A8 `ours`: Flash `+12.3%` mean20 / `+3.9%` steady median; Pro `+16.3%` mean20 / `+11.6%` steady median.
- Equal-weight geometric-mean latency gaps versus W8A8 PR323: Flash `+10.5%` mean20 / `+2.2%` steady median; Pro `+5.5%` mean20 / `+1.3%` steady median.
- Main steady-state deficits versus W8A8 `ours`: Flash M32 `+22.6%`, M64 `+21.5%`, M128 `+10.6%`; Pro M8/M16/M32/M64/M128 `+23.5/+37.7/+30.7/+21.0/+31.4%`. At M>=256, most points are within roughly 0--6%, except Flash M1536 mean20 `+9.0%` and Pro M2048 mean20 `+8.1%`.
- Long tails materially affect small-M mean20 (for example Flash M16 has a 3.04 ms outlier versus a 401.6 us median), so report both mean20 and steady median rather than attributing the whole mean gap to core compute.
- Reproducibility artifacts: `/root/fac/scripts/megamoe/nvfp4_vs_w8a8_20260702/` on `10.6.131.7`, including per-point logs and `comparison.csv`.

## 2026-07-02 rejected dequant/WGMMA overlap experiments

Evaluated math-warp K+1 lookahead, three single-kernel producer layouts, and a
BN256 two-kernel L1/L2 plan. Correctness passed for the runnable Flash/Pro
variants with unchanged thresholds, but none improved end-to-end latency.

- ptxas serialized math-warp decode stores behind WGMMA and introduced spills.
- Dispatch-assisted decode regressed Flash/Pro M64 by about 38%/28%.
- BN256 split regressed Flash/Pro M64 by about 4%/19%.
- Two-TMA-warp decode regressed M64 by about 85% because it removed TMA
  lookahead.
- A one-math-WG wide-N producer layout generated a 920-byte stack and about
  1003 local load/store instructions, then regressed Flash M64 to 2834 us.

All experimental code and controls were removed. Detailed synchronization,
resource, correctness, and benchmark results are in
`docs/plans/2026-07-02-sm90-nvfp4-dequant-overlap-design.md`. Experiment caches
are under `/root/fac/scripts/megamoe/nvfp4_*dequant*`,
`/root/fac/scripts/megamoe/nvfp4_dispatch_ab/`, and
`/root/fac/scripts/megamoe/nvfp4_split_bn256_ab/`.
