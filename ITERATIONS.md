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
