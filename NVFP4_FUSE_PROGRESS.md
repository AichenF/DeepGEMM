# NVFP4 Fused MegaMoE Progress

Date: 2026-06-10
Machine: 10.6.131.8
Container: `mega_moe_box`
Workspace: `/root/fac/megamoe/deepgemm_fp4_fuse`
Branch: `megamoe_nvfp4_fuse`
Base: PR323 current HEAD `23f46aa Add SM90 MegaMoE decode support`
Reference NVFP4 repo: `/root/fac/megamoe/DeepGEMM_megamoe_nvfp4`

## Goal

Port the existing SM90 NVFP4 MegaMoE support onto the latest PR323 fused MegaMoE codebase, with dispatch, L1, L2, and combine running through the single-kernel fused path by default.

## Implemented

- Added NVFP4 dequant primitive: `deep_gemm/include/deep_gemm/quantization/nvfp4_dequant.cuh`
- Added SM90 NVFP4 MegaMoE kernel and host launcher:
  - `deep_gemm/include/deep_gemm/impls/sm90_nvfp4_mega_moe.cuh`
  - `csrc/jit_kernels/impls/sm90_nvfp4_mega_moe.hpp`
- Added Python quantization and validation scripts:
  - `deep_gemm/quantization_nvfp4.py`
  - `tests/test_nvfp4_mega_moe_sm90_correctness.py`
  - `tests/bench_nvfp4_mega_moe_sm90.py`
- Wired NVFP4 into PR323 SM90 API:
  - `csrc/apis/sm90_mega.hpp`
  - `deep_gemm/mega/__init__.py`
  - `deep_gemm/__init__.py`
- Set NVFP4 default to the fused single-kernel path: `DG_SM90_MOE_SPLIT_L1_L2=0` in the NVFP4 launcher.
- Pinned NVFP4 launcher to the currently supported dequant shape: `block_m=64`, `block_n=128`, `block_k=128`, `num_epilogue_threads=128`.

## Fixes Needed During Port

1. PR323 JIT include path did not include CUTLASS third-party headers.
   - Symptom: NVCC could not find `cute/arch/mma_sm89.hpp`.
   - Fix: add `../third-party/cutlass/include` to NVCC and NVRTC JIT include paths in `csrc/jit/compiler.hpp`.
2. PR323 TMA descriptor helper did not accept `torch::kUInt8`.
   - Symptom: `Unsupported dtype` while creating packed NVFP4 weight TMA descriptors.
   - Fix: map `torch::kUInt8` to `CU_TENSOR_MAP_DATA_TYPE_UINT8` in `csrc/jit_kernels/impls/runtime_utils.hpp`.
3. PR323 `math.cuh` used `__fmul2_rn` directly on SM90.
   - Symptom: NVCC JIT error: `identifier __fmul2_rn is undefined`.
   - Fix: add `math::mul2`, using `__fmul2_rn` on SM100+ and scalar fallback otherwise.
4. Python NVFP4 transform depended on `_interleave_l1_weights` from the reference branch.
   - Fix: add `_interleave_l1_weights` wrapper in `deep_gemm/mega/__init__.py`.

## Build

Command:

```bash
cd /root/fac/megamoe/deepgemm_fp4_fuse && touch csrc/python_api.cpp && python setup.py build_ext --inplace -j 16
```

Result: PASS.

Note: fresh clone required `git submodule update --init --recursive` before build because CUTLASS/fmt submodules were empty.

## Correctness

Primary semantic gate is `fp8-bridge`, matching the current NVFP4 bridge kernel design: packed NVFP4 weights are expanded to FP8 for WGMMA.

Smoke result:

- Dequant unit test: PASS
- CUDA LUT dequant unit test: PASS
- M=32, weight_scale=0.05: cosine_min=0.9996, cosine_mean=0.9997, norm_ratio=0.9997

Expanded bridge command:

```bash
MASTER_ADDR=127.0.0.1 MASTER_PORT=29591 \
DG_JIT_CACHE_DIR=/tmp/dg_jit_nvfp4_fuse_full_<ts> \
DG_SM90_MOE_SPLIT_L1_L2=0 \
python3 -u tests/test_nvfp4_mega_moe_sm90_correctness.py \
  --batches 32 128 256 \
  --weight-scales 0.001 0.01 0.05 0.3 \
  --reference-mode fp8-bridge \
  --cosine-mean-threshold 0.99 \
  --cosine-min-threshold 0.99 \
  --norm-ratio-min 0.99 \
  --norm-ratio-max 1.01
```

Expanded bridge result: PASS for all 12 cases.

Worst observed bridge case:

- M=256, weight_scale=0.3: cosine_min=0.9991, cosine_mean=0.9997, norm_ratio=0.9988

Exact-NVFP4 cross-check:

- M=32/256, weight_scale=0.05 and 0.3: cosine_mean about 0.9966-0.9982, norm_ratio about 0.9967-0.9995.
- M=32, weight_scale=0.001 fails exact semantic check: cosine_mean=0.4235, norm_ratio=0.5381.
- Interpretation: this is expected for the current FP8-bridge semantics at tiny output magnitudes; exact-BF16/NVFP4 reference includes quantization effects the bridge path intentionally does not preserve. The default correctness gate remains `fp8-bridge`.

## Benchmark

1-rank smoke:

- tokens=32, recv=128, experts=7, nvfp4=362.4 us

8-rank command:

```bash
MASTER_ADDR=127.0.0.1 MASTER_PORT=29595 \
DG_JIT_CACHE_DIR=/tmp/dg_jit_nvfp4_fuse_bench8_<ts> \
DG_SM90_MOE_SPLIT_L1_L2=0 \
python3 -u tests/bench_nvfp4_mega_moe_sm90.py \
  --num-processes 8 --batches 32 64 128 256 512 --num-tests 20
```

8-rank result on rank0:

| tokens | recv | experts | NVFP4 time | TFLOPS | HBM GB/s |
|---:|---:|---:|---:|---:|---:|
| 32 | 279 | 32 | 1322.0 us | 18.6 | 605 |
| 64 | 524 | 32 | 1293.0 us | 35.7 | 623 |
| 128 | 1025 | 32 | 1256.0 us | 71.9 | 652 |
| 256 | 1993 | 32 | 1813.0 us | 96.8 | 465 |
| 512 | 4087 | 32 | 3455.0 us | 104.2 | 260 |

## Current Assessment

- The port is buildable and runs the fused single-kernel NVFP4 path by default.
- The current implementation is correct under FP8-bridge semantics.
- The exact-NVFP4 low-scale gap is a semantic/quality limitation of the bridge design, not a dispatch/combine fusion bug.
- Performance is currently slower than the earlier verified standalone NVFP4 result for M=32, so the port is not yet a performance win.

## Next Work

- Add final NVFP4 config print after the PR323 heuristic override, so `DG_PRINT_CONFIGS=1` reports the actual instantiated NVFP4 shape.
- Compare against PR323 FP8 fused benchmark under the same run protocol.
- Profile fused NVFP4 M=32/M=128 to separate fixed dispatch/combine overhead from NVFP4 dequant overhead.
- Consider reintroducing PR323 decode-specific scheduling improvements into the NVFP4 path after BN128 correctness is stable.

## 2026-06-10 Wide Sweep Update

### PR323 Baseline Calibration

Remote PR323 head was checked with:

```bash
git ls-remote upstream pull/323/head
```

Result:

```text
23f46aa68c892a349bb7ce331a325e36acceb57e refs/pull/323/head
```

So the comparable W8A8 baseline below is from PR323 head `23f46aa Add SM90 MegaMoE decode support`, not from the older `megamoe_sm90` branch.

Comparable shape:

```text
hidden=7168
intermediate_hidden=2048
num_experts=256
num_topk=8
num_processes=8
batches=32,64,128,256,512,1024,2048,4096,8192
```

### Latest Correctness

Default fused BN128 path was rechecked after the BN256 opt-in experiment:

```bash
MASTER_ADDR=127.0.0.1 MASTER_PORT=29607 \
DG_JIT_CACHE_DIR=/tmp/dg_jit_nvfp4_bn128_check_$$ \
DG_SM90_MOE_SPLIT_L1_L2=0 \
python3 -u tests/test_nvfp4_mega_moe_sm90_correctness.py \
  --batches 32 256 \
  --weight-scales 0.001 0.05 0.3 \
  --reference-mode fp8-bridge \
  --cosine-mean-threshold 0.99 \
  --cosine-min-threshold 0.99 \
  --norm-ratio-min 0.99 \
  --norm-ratio-max 1.01
```

Result: PASS for all 6 cases.

Opt-in BN256 path was also checked:

```bash
MASTER_ADDR=127.0.0.1 MASTER_PORT=29608 \
DG_JIT_CACHE_DIR=/tmp/dg_jit_nvfp4_bn256_check_$$ \
DG_SM90_MOE_SPLIT_L1_L2=0 \
DG_SM90_NVFP4_BLOCK_N=256 \
python3 -u tests/test_nvfp4_mega_moe_sm90_correctness.py \
  --batches 32 256 \
  --weight-scales 0.001 0.05 0.3 \
  --reference-mode fp8-bridge \
  --cosine-mean-threshold 0.99 \
  --cosine-min-threshold 0.99 \
  --norm-ratio-min 0.99 \
  --norm-ratio-max 1.01
```

Result: PASS for all 6 cases, but performance is worse than default, so BN256 is not recommended as default.

### Latest Wide Benchmark

W8A8 PR323 baseline command:

```bash
MASTER_ADDR=127.0.0.1 MASTER_PORT=29602 \
DG_JIT_CACHE_DIR=/tmp/dg_jit_w8a8_pr323_same_<ts> \
python3 -u tests/test_mega_moe_hopper.py \
  --fused-only-sweep \
  --num-processes 8 \
  --hidden 7168 \
  --intermediate-hidden 2048 \
  --num-experts 256 \
  --num-topk 8 \
  --batches 32 64 128 256 512 1024 2048 4096 8192 \
  --num-bench-tests 20
```

NVFP4 default fused command:

```bash
MASTER_ADDR=127.0.0.1 MASTER_PORT=29615 \
DG_JIT_CACHE_DIR=/tmp/dg_jit_nvfp4_final_wide_$$ \
DG_SM90_MOE_SPLIT_L1_L2=0 \
python3 -u tests/bench_nvfp4_mega_moe_sm90.py \
  --num-processes 8 \
  --hidden 7168 \
  --intermediate-hidden 2048 \
  --num-experts 256 \
  --num-topk 8 \
  --batches 32 64 128 256 512 1024 2048 4096 8192 \
  --num-tests 20
```

| tokens | W8A8 PR323 | NVFP4 fused | NVFP4 / W8A8 | Regression |
|---:|---:|---:|---:|---:|
| 32 | 810.6 us | 1306.0 us | 1.61x | +61.1% |
| 64 | 992.7 us | 1346.0 us | 1.36x | +35.6% |
| 128 | 848.1 us | 1294.0 us | 1.53x | +52.6% |
| 256 | 1427.0 us | 1863.0 us | 1.31x | +30.6% |
| 512 | 2238.0 us | 3172.0 us | 1.42x | +41.7% |
| 1024 | 3727.0 us | 5535.0 us | 1.49x | +48.5% |
| 2048 | 6232.0 us | 10245.0 us | 1.64x | +64.4% |
| 4096 | 11337.0 us | 19471.0 us | 1.72x | +71.7% |
| 8192 | 21921.0 us | 38304.0 us | 1.75x | +74.7% |

Earlier initial NVFP4 fused wide result, before the BN256 opt-in patch, was similar:

| tokens | Initial NVFP4 fused |
|---:|---:|
| 32 | 1384.0 us |
| 64 | 1276.0 us |
| 128 | 1303.0 us |
| 256 | 1826.0 us |
| 512 | 3223.0 us |
| 1024 | 5520.0 us |
| 2048 | 10293.0 us |
| 4096 | 19522.0 us |
| 8192 | 38278.0 us |

Interpretation: the BN256 opt-in changes did not materially improve the default path; default BN128 performance remains in the same range.

### Phase Profile

Command used `DG_SM90_MOE_PHASE_PROFILE=1` for M=32,128,1024,8192 with `--num-tests 5`. Phase profiling adds overhead, so use these numbers for attribution only.

| tokens | NVFP4 profiled time | dispatch_total avg cycles | math_loop avg cycles | combine_barrier avg cycles | combine_reduce avg cycles |
|---:|---:|---:|---:|---:|---:|
| 32 | 1243 us | 36,855 | 1,171,597 | 65,595 | 1,867 |
| 128 | 1499 us | 283,360 | 1,407,127 | 58,964 | 7,448 |
| 1024 | 5459 us | 235,075 | 5,180,060 | 212,955 | 65,782 |
| 8192 | 38557 us | 1,728,360 | 37,323,621 | 618,405 | 574,109 |

Main conclusion: `math_loop` dominates at every size, including M=32. The fused NVFP4 slowdown is therefore not primarily fixed combine/scatter overhead. The main structural cost is the NVFP4-to-FP8 bridge inside the math pipeline: load packed FP4 B, load UE4M3 scale, LUT/arithmetic dequant, write expanded FP8 B to shared memory, synchronize, then WGMMA consumes FP8 B from shared memory.

### Split Branch Cross-Check

Reference split NVFP4 branch `/root/fac/megamoe/DeepGEMM_megamoe_nvfp4` was rebuilt and benchmarked with the same shape.

| tokens | Split NVFP4 branch |
|---:|---:|
| 32 | 6142.0 us |
| 64 | 1265.0 us |
| 128 | 1267.5 us |
| 256 | 1877.0 us |
| 512 | 3153.0 us |
| 1024 | 5610.0 us |
| 2048 | 10174.0 us |
| 4096 | 19452.0 us |
| 8192 | 38359.0 us |

Except for an M=32 long-tail anomaly, split branch performance is essentially the same as fused for M>=64. This supports the phase-profile conclusion: dispatch/combine fusion is not the current dominant bottleneck.

### Existing Dequant Knob Sweep

All runs used M=32,128,1024, `--num-tests 20`.

| Variant | M32 | M128 | M1024 | Result |
|---|---:|---:|---:|---|
| loader default | 1376 us | 1254 us | 5769 us | best among tested existing knobs |
| `DG_SM90_NVFP4_LOADER_DEQUANT=0` | 1460 us | 1478 us | 6314 us | slower |
| `DG_SM90_NVFP4_LOADER_DEQUANT=0 DG_SM90_NVFP4_DIRECT_SCALE_GMEM=1` | 1735 us | 1790 us | 7077 us | slower |
| `DG_SM90_NVFP4_LOADER_DEQUANT=0 DG_SM90_NVFP4_PACKED_B_SCRATCH=1` | 1465 us | 1750 us | 6177 us | slower |

Effective so far:

- Keeping the loader-side dequant path enabled.
- Keeping default BN128 for NVFP4.
- Using FP8-bridge correctness as the primary gate for this implementation.

Ineffective or reverted/not recommended:

- Math-warp dequant instead of loader dequant.
- Direct scale load from global memory in the math path.
- Packed-B scratch path.
- BN256 opt-in: correctness passes but benchmark regresses, so it should not become default.
- BN128 + 256 epilogue-thread split-N: correctness passes but benchmark regresses, so the experiment was reverted.
- `DG_SM90_NVFP4_DISPATCH_THREADS=64` with 128 non-epilogue threads: invalid warpgroup alignment; now guarded by host/static asserts.
- `DG_SM90_NVFP4_DISPATCH_THREADS=64 DG_SM90_NVFP4_NON_EPILOGUE_THREADS=64`: correctness passes but disables loader-dequant and benchmarks slower.

### Current Bottleneck Assessment

Small M, especially M=32/64/128:

- W8A8 PR323 already has a very optimized fixed-overhead fused path.
- NVFP4 saves B bandwidth, but it also pays decode work and an expanded FP8 shared-memory bridge before WGMMA.
- The phase profile shows the main-loop cost dominates even at M=32, so the extra dequant/shared-memory bridge cannot be hidden by dispatch/combine fixed overhead.

Medium and large M, M=256 through 8192:

- If the design could reduce real DRAM traffic without paying the FP8 bridge cost, NVFP4 should theoretically become more attractive.
- In this implementation, expanded FP8 shared-memory staging plus scale handling keeps the math loop slower than W8A8, so bandwidth savings do not translate into end-to-end speedup.
- The gap grows again at large M, which indicates the bridge/dequant pipeline is throughput-limiting rather than a pure small-M launch/fixed-cost issue.

### Next Optimization Direction

The BN128 + 2 epilogue-WG split-N path was implemented as an opt-in experiment, passed correctness, benchmarked slower, and was reverted. The remaining high-value path is therefore not another split-N/thread-count knob.

Current best next directions:

1. Reduce the NVFP4-to-FP8 bridge cost inside the loader path: fewer shared-memory stores/loads, cheaper scale handling, or less synchronization before WGMMA consumes B.
2. Build a separate small-M kernel that dequants in registers/fragments and avoids the expanded FP8 shared-memory bridge, likely with an `mma.sync` style path. On SM90 WGMMA cannot consume NVFP4 directly, so the current bridge has a hard overhead floor.
3. Use phase profiling around the loader-dequant and WGMMA wait points to determine whether the bridge bottleneck is instruction count, shared-memory bandwidth, or producer/consumer barrier latency.

### Attempt: BN128 + 2 Epilogue WG Split-N

Hypothesis: PR323 W8A8 uses `block_m=64`, `block_n=128`, `num_epilogue_threads=256` for small M. Current NVFP4 default uses `block_m=64`, `block_n=128`, `num_epilogue_threads=128`. Porting the small-M 2-epilogue-WG schedule might improve occupancy/split-N behavior.

Implementation attempted as an opt-in only:

```bash
DG_SM90_NVFP4_EPILOGUE_THREADS=256
```

Changes made for the experiment:

- Allowed BN128 with 256 epilogue threads in the host launcher.
- Added BN128 split-N shape constants matching PR323 W8A8's `M64N64` per-WG WGMMA shape.
- Added shared-SF logic for the case where each N-split warpgroup owns only 32 post-SwiGLU columns and both WGs must share one per-64-column L2 activation scale.
- Relaxed loader-dequant so the idle non-epilogue warps could still dequant B for BN128.

Correctness:

```bash
DG_SM90_MOE_SPLIT_L1_L2=0 \
DG_SM90_NVFP4_EPILOGUE_THREADS=256 \
python3 -u tests/test_nvfp4_mega_moe_sm90_correctness.py \
  --batches 32 128 256 \
  --weight-scales 0.001 0.05 0.3 \
  --reference-mode fp8-bridge \
  --cosine-mean-threshold 0.99 \
  --cosine-min-threshold 0.99 \
  --norm-ratio-min 0.99 \
  --norm-ratio-max 1.01
```

Result: PASS for all 9 cases.

Benchmark:

| tokens | Default BN128/128 epi recheck | BN128/256 epi opt-in | Result |
|---:|---:|---:|---|
| 32 | 1243.0 us | 1375.0 us | slower |
| 64 | 1255.0 us | 1342.0 us | slower |
| 128 | 1257.0 us | 1345.0 us | slower |
| 256 | 1847.0 us | 1979.0 us | slower |
| 512 | 3151.0 us | 3347.0 us | slower |
| 1024 | 5460.0 us | 5749.0 us | slower |

Conclusion: no performance benefit. The opt-in code was reverted after the benchmark. Default correctness was rerun after revert and PASSed for M=32/256 with weight scales 0.001/0.05/0.3.

### Attempt: Dispatch 64 Threads

Hypothesis: W8A8 PR323 can use fewer dispatch/non-epilogue threads for small M, so reducing NVFP4 dispatch threads from 128 to 64 might reduce fixed overhead while keeping non-epilogue threads at 128 for loader-dequant.

Command shape:

```bash
DG_SM90_MOE_SPLIT_L1_L2=0 \
DG_SM90_NVFP4_DISPATCH_THREADS=64 \
python3 -u tests/test_nvfp4_mega_moe_sm90_correctness.py \
  --batches 32 256 \
  --weight-scales 0.001 0.05 0.3 \
  --reference-mode fp8-bridge
```

Result: invalid. It hit a CUDA illegal instruction before correctness completed.

Root cause: `dispatch_threads=64` and `non_epilogue_threads=128` make the epilogue role start after 192 threads, which is not aligned to a 4-warp warpgroup boundary. The epilogue role uses warpgroup-level register reconfiguration, so the role boundary must be 128-thread aligned.

Fix retained:

- Added a kernel static assert: `(kNumDispatchThreads + kNumNonEpilogueThreads) % 128 == 0`.
- Added the matching host assert before JIT launch.

Default correctness after adding the guard:

```bash
DG_SM90_MOE_SPLIT_L1_L2=0 \
python3 -u tests/test_nvfp4_mega_moe_sm90_correctness.py \
  --batches 32 \
  --weight-scales 0.05 \
  --reference-mode fp8-bridge \
  --cosine-mean-threshold 0.99 \
  --cosine-min-threshold 0.99 \
  --norm-ratio-min 0.99 \
  --norm-ratio-max 1.01
```

Result: PASS, M=32 weight_scale=0.05, cosine_min=0.9996, cosine_mean=0.9997, norm_ratio=0.9997.

### Attempt: 64 Dispatch / 64 Non-Epilogue Threads

Hypothesis: use a warpgroup-aligned lightweight thread topology similar to W8A8 small-M, while accepting that 64 non-epilogue threads disables the NVFP4 loader-dequant path.

Correctness command:

```bash
DG_SM90_MOE_SPLIT_L1_L2=0 \
DG_SM90_NVFP4_DISPATCH_THREADS=64 \
DG_SM90_NVFP4_NON_EPILOGUE_THREADS=64 \
python3 -u tests/test_nvfp4_mega_moe_sm90_correctness.py \
  --batches 32 256 \
  --weight-scales 0.001 0.05 0.3 \
  --reference-mode fp8-bridge \
  --cosine-mean-threshold 0.99 \
  --cosine-min-threshold 0.99 \
  --norm-ratio-min 0.99 \
  --norm-ratio-max 1.01
```

Result: PASS for all 6 cases.

Benchmark:

| tokens | Default recheck | 64/64 threads | Result |
|---:|---:|---:|---|
| 32 | 1243.0 us | 1499.0 us | slower |
| 64 | 1255.0 us | 1682.0 us | slower |
| 128 | 1257.0 us | 1508.0 us | slower |
| 256 | 1847.0 us | 2099.0 us | slower |
| 512 | 3151.0 us | 3604.0 us | slower |
| 1024 | 5460.0 us | 6357.0 us | slower |

Conclusion: not useful. The reduced-thread topology cannot compensate for losing loader-side NVFP4 dequant.

### Phase Profile: Dequant Attribution

Added phase-profile-only metrics:

- `loader_dequant`: time spent in the loader-side dequant call, including its wait for the packed-B/SFB full barrier.
- `math_dequant_wait`: time math WG spends waiting for loader-dequant completion.

These metrics are compiled only when `DG_SM90_MOE_PHASE_PROFILE=1`; default benchmark path is unchanged.

Command:

```bash
DG_SM90_MOE_SPLIT_L1_L2=0 \
DG_SM90_MOE_PHASE_PROFILE=1 \
python3 -u tests/bench_nvfp4_mega_moe_sm90.py \
  --num-processes 8 \
  --hidden 7168 \
  --intermediate-hidden 2048 \
  --num-experts 256 \
  --num-topk 8 \
  --batches 32 128 1024 \
  --num-tests 5
```

| tokens | profiled NVFP4 | math_loop avg | gemm_core avg | loader_dequant avg | math_dequant_wait avg |
|---:|---:|---:|---:|---:|---:|
| 32 | 1482 us | 1,356,051 cyc | 34,957 cyc | 1,050 cyc | 420 cyc |
| 128 | 1468 us | 1,347,676 cyc | 35,110 cyc | 1,061 cyc | 410 cyc |
| 1024 | 6184 us | 5,843,169 cyc | 33,879 cyc | 1,032 cyc | 416 cyc |

Interpretation:

- Math-side waiting for dequant completion is small, around 410-420 cycles on average.
- Loader-dequant itself is nontrivial work, around 1,030-1,060 cycles per recorded dequant point, but it is mostly overlapped with the pipeline.
- The current bottleneck is therefore not a large producer/consumer barrier wait. It is the structural FP4-to-FP8 bridge work: extra dequant instructions, FP8 shared-memory writes, later WGMMA shared-memory reads, and the additional non-epilogue role pressure needed to sustain that bridge.
- This explains why thread-count/split-N knobs did not help: the W4/NVFP4 bandwidth saving is not converting to speedup while the bridge path still materializes full FP8 B in shared memory.

### Current Default Benchmark After Latest Changes

After reverting the BN128+256-epilogue experiment and adding the warpgroup-alignment guard plus phase-profile-only dequant metrics, the default path was benchmarked again with phase profiling disabled.

Command:

```bash
DG_SM90_MOE_SPLIT_L1_L2=0 \
python3 -u tests/bench_nvfp4_mega_moe_sm90.py \
  --num-processes 8 \
  --hidden 7168 \
  --intermediate-hidden 2048 \
  --num-experts 256 \
  --num-topk 8 \
  --batches 32 64 128 256 512 1024 \
  --num-tests 20
```

| tokens | W8A8 PR323 | Current NVFP4 default | NVFP4 / W8A8 | Regression |
|---:|---:|---:|---:|---:|
| 32 | 810.6 us | 1267.0 us | 1.56x | +56.3% |
| 64 | 992.7 us | 1346.0 us | 1.36x | +35.6% |
| 128 | 848.1 us | 1270.0 us | 1.50x | +49.7% |
| 256 | 1427.0 us | 1833.0 us | 1.28x | +28.5% |
| 512 | 2238.0 us | 3250.0 us | 1.45x | +45.2% |
| 1024 | 3727.0 us | 5506.0 us | 1.48x | +47.7% |

This is the latest default result for the current source tree. The earlier 32-8192 wide table remains useful for large-M trend, but this table is the clean latest post-guard benchmark for 32-1024.

### Attempt: Stage And Schedule Knobs

Added an opt-in stage override:

```bash
DG_SM90_NVFP4_NUM_STAGES=<2..max>
```

Default behavior is unchanged: if the env is absent, the host still uses the max stage count allowed by SMEM capacity. The knob is retained for profiling/tuning only.

Correctness:

- `DG_SM90_NVFP4_NUM_STAGES=2,3,4,5,6,7` all PASSed M=32/256 with `weight_scale=0.05` under `fp8-bridge` reference.

Screening benchmark, 8 GPUs, `--num-tests 10`:

| stages | M32 | M128 | M256 | M1024 | Result |
|---:|---:|---:|---:|---:|---|
| 2 | 1865 us | 1893 us | 2917 us | 8073 us | much slower |
| 3 | 1570 us | 1503 us | 2313 us | 6532 us | slower |
| 4 | 1282 us | 1297 us | 2032 us | 5774 us | no clear win |
| 5 | 1367 us | 1274 us | 2002 us | 5579 us | no clear win |
| 6 | 1307 us | 1291 us | 1983 us | 5552 us | no clear win |
| 7 | 1251 us | 1270 us | 1975 us | 5646 us | best small-M, current default/max |

Conclusion: keep max-stage default. Lower stages reduce pipeline depth too much and do not close the W8A8 gap.

Schedule/accum knobs tested:

| Variant | Correctness | M32 | M64 | M128 | M256 | M512 | M1024 | Result |
|---|---|---:|---:|---:|---:|---:|---:|---|
| `DG_SM90_MOE_L2_NMAJOR=1` | PASS | 12759 us | 3463 us | 3481 us | 4044 us | 7776 us | 12550 us | much slower |
| `DG_SM90_MOE_L1_NMAJOR=1` | PASS | 3461 us | 3474 us | 3431 us | 4039 us | 7751 us | 12546 us | much slower |
| `DG_SM90_MOE_K2_DIRECT_ACCUM=1` | PASS | 3459 us | 3480 us | 11436 us | 4046 us | 7803 us | 12563 us | much slower |
| `DG_SM90_MOE_L2_DUAL_ACCUM=0` | PASS | 11929 us | 3465 us | 3468 us | 4030 us | 7775 us | 12521 us | much slower |

`DG_SM90_MOE_ASYNC_L1_STORE=1`:

- Result: failed correctness with CUDA illegal memory access.
- Action retained: host launcher now rejects this env for NVFP4 with a clear assert instead of reaching a CUDA memory fault.
- Default correctness after adding the guard: PASS for M=32, `weight_scale=0.05`.

Conclusion: the PR323 schedule/accum knobs do not transfer to this NVFP4 bridge path. The bridge path remains dominated by materializing FP8 B in shared memory and the pipeline shape around that work.

### Attempt: Constant LUT Dequant

Hypothesis: avoid copying the 1KB NVFP4 E2M1+UE4M3 LUT into shared memory for every CTA and read the `__constant__` LUT directly. This also frees the LUT shared-memory allocation.

Temporary implementation:

- Set `SMEM_NVFP4_LUT_SIZE=0`.
- Point dequant helpers at `deep_gemm::nvfp4::kE2M1AndUe4m3ToFp8Lut` directly.
- Removed the per-CTA `thread_idx < 64` LUT copy.
- Updated host SMEM estimate accordingly.

Correctness:

- PASS for M=32/256, `weight_scale=0.001/0.05/0.3`, `fp8-bridge` reference.

Benchmark, 8 GPUs, `--num-tests 20`:

| tokens | shared LUT default | constant LUT | Result |
|---:|---:|---:|---|
| 32 | 1267 us | 3706 us | much slower |
| 64 | 1346 us | 3705 us | much slower |
| 128 | 1270 us | 3731 us | much slower |
| 256 | 1833 us | 6838 us | much slower |
| 512 | 3250 us | 8374 us | much slower |
| 1024 | 5506 us | 16260 us | much slower |

Conclusion: constant LUT is not viable. The shared-memory LUT is required for throughput. The experiment was reverted, and default correctness after revert PASSed for M=32, `weight_scale=0.05`.

### Attempt: BM128 + BN256 Tile

Hypothesis: W8A8 PR323 uses larger tiles for larger M, so NVFP4 might recover throughput for M=256+ with `block_m=128`, `block_n=256`.

Implementation attempted:

- Added temporary `DG_SM90_NVFP4_BLOCK_M=128` host knob.
- Required `DG_SM90_NVFP4_BLOCK_N=256`.
- Fixed the kernel `kSplitNWarpgroups` predicate to require `BLOCK_M == 64`; otherwise BM128+BN256 was incorrectly treated as split-N.

Results:

1. `BM128+BN256` with default `dispatch=128, non_epilogue=128, epilogue=256` failed JIT/ptxas:

```text
Insufficient registers (128) ... Try to compile with register target of 154 or higher.
```

2. `BM128+BN256` with `dispatch=64, non_epilogue=64, epilogue=256` compiled and passed correctness for M=32/256/1024, `weight_scale=0.05`.

Benchmark, 8 GPUs, `--num-tests 20`:

| tokens | Current default | BM128+BN256 64/64 | Result |
|---:|---:|---:|---|
| 32 | 1267 us | 3990 us | much slower |
| 64 | 1346 us | 4011 us | much slower |
| 128 | 1270 us | 4005 us | much slower |
| 256 | 1833 us | 4132 us | much slower |
| 512 | 3250 us | 6370 us | slower |
| 1024 | 5506 us | 10131 us | slower |

Conclusion: BM128+BN256 is not useful in the current bridge design. The temporary `BLOCK_M` knob was reverted. The `BLOCK_M == 64` split-N predicate fix was retained because it is a correctness/maintainability fix and does not change existing BM64 behavior. Default correctness after revert PASSed for M=32, `weight_scale=0.05`.

### Attempt: BM32 mma.sync Path

Hypothesis: a small-M `BM32/BN128` mma.sync path may reduce M-padding and avoid part of the WGMMA bridge overhead.

Implementation attempted as an opt-in only:

```bash
DG_SM90_NVFP4_MMA_SYNC=1
```

Temporary changes:

- Host set `block_m=32`, `block_n=128` for the opt-in path.
- Required loader-side dequant.
- Added a `dequant_barriers[stage].wait()` in the mma.sync K loop so the path consumes dequantized FP8 B, rather than packed FP4.
- Forced activation TMA swizzle to 0 for the mma.sync path, because the mma.sync code reads row-major shared memory.

Result:

- The path compiled after removing the old `block_m >= 64` host assert.
- Runtime still failed correctness with CUDA illegal memory access, even with `CUDA_LAUNCH_BLOCKING=1`.
- Likely remaining issue is a deeper mismatch in the dormant mma.sync path's SMEM/TMA/layout assumptions, not just the missing dequant wait.

Action:

- Reverted the opt-in host exposure and the unverified mma.sync changes.
- Restored the default `block_m=64` WGMMA bridge path.
- Default correctness after revert PASSed for M=32, `weight_scale=0.05`.

Conclusion: the existing mma.sync branch is not close enough to production-ready for a quick port. A real small-M register-dequant/mma.sync kernel would need a dedicated implementation and test plan, not a simple host knob.


### Attempt: Enable and Fix Split L1/L2 Runtime Path

Hypothesis: the fused NVFP4 bridge keeps dispatch, L1, L2, and combine in one launch, but the FP4->FP8 bridge adds per-stage fixed work and SMEM pressure. Splitting L1 and L2 into two launches may reduce per-kernel resource pressure enough to beat the fused runtime despite the extra launch.

Implementation:

- Fixed the NVFP4 split path compile failure. The old code called removed scheduler methods `for_each_linear1_block` / `for_each_linear2_block`. Replaced `for_each_selected_block` with a `fetch_expert_recv_count()` + `get_next_block()` iterator that filters `Linear1` / `Linear2` based on `kRunL1Phase` and `kRunL2Phase`.
- Changed NVFP4 host default to `DG_SM90_MOE_SPLIT_L1_L2=1` while keeping `DG_SM90_MOE_SPLIT_L1_L2=0` as the fused fallback.
- Synchronized the NVFP4 benchmark script default split flag with the host default.

Correctness:

```bash
python3 -u tests/test_nvfp4_mega_moe_sm90_correctness.py \
  --batches 32 256 --weight-scales 0.001 0.05 0.3
```

Result: PASS. Dequant unit test PASS, CUDA LUT dequant unit test PASS, M=32/256 PASS for all three weight scales. Worst retained cosine in this run was `0.9991` at M=256, weight_scale=0.3; norm ratios stayed around `0.9988..1.0016`.

Benchmark, 8 GPUs, explicit split path, `--num-tests 30`:

```bash
DG_SM90_MOE_SPLIT_L1_L2=1 python3 -u tests/bench_nvfp4_mega_moe_sm90.py \
  --num-processes 8 --num-tests 30 \
  --batches 32 64 128 256 512 1024 2048 4096 8192
```

| tokens | fused NVFP4 30-run | split NVFP4 30-run | speedup vs fused | PR323 W8A8 | split gap vs W8A8 |
|---:|---:|---:|---:|---:|---:|
| 32 | 1305.0 us | 1208.2 us | 1.080x | 810.6 us | 1.490x slower |
| 64 | 1327.0 us | 1272.9 us | 1.042x | 992.7 us | 1.282x slower |
| 128 | 1293.0 us | 1236.9 us | 1.045x | 848.1 us | 1.458x slower |
| 256 | 1939.0 us | 1766.3 us | 1.098x | 1427.0 us | 1.238x slower |
| 512 | 3357.0 us | 3083.0 us | 1.089x | 2238.0 us | 1.378x slower |
| 1024 | 5508.0 us | 5216.0 us | 1.056x | 3727.0 us | 1.399x slower |
| 2048 | 10177.0 us | 9719.0 us | 1.047x | 6232.0 us | 1.560x slower |
| 4096 | 19492.0 us | 18463.0 us | 1.056x | 11337.0 us | 1.628x slower |
| 8192 | 38288.0 us | 36302.0 us | 1.055x | 21921.0 us | 1.656x slower |

50-run small-M cross-check:

| tokens | fused 50-run | split 50-run | speedup |
|---:|---:|---:|---:|
| 32 | 1241.0 us | 1207.9 us | 1.027x |
| 64 | 1285.0 us | 1225.6 us | 1.048x |
| 128 | 1279.0 us | 1256.8 us | 1.018x |
| 256 | 1836.0 us | 1752.3 us | 1.048x |

Conclusion: split L1/L2 is a real retained improvement across all measured sizes, but it is still far from the PR323 W8A8 baseline. The improvement is launch-structure/resource-pressure related, not a dequant arithmetic speedup.

### Attempt: L2 No-Dispatch Pipeline on Split Path

Hypothesis: in split mode, the L2-only launch may not need the dispatch pipeline and could save fixed overhead.

Correctness:

```bash
DG_SM90_MOE_SPLIT_L1_L2=1 DG_SM90_MOE_L2_NO_DISPATCH_PIPELINE=1 \
python3 -u tests/test_nvfp4_mega_moe_sm90_correctness.py --batches 32 256 --weight-scales 0.05
```

Result: PASS.

Benchmark, 8 GPUs, `--num-tests 30`:

| tokens | split default | split + L2 no-dispatch | result |
|---:|---:|---:|---|
| 32 | 1284.1 us | 1198.5 us | faster in this 30-run |
| 64 | 1229.9 us | 1365.3 us | slower |
| 128 | 1201.7 us | 1231.0 us | slower |
| 256 | 1821.4 us | 1745.0 us | faster in this 30-run |
| 512 | 2978.0 us | 3105.0 us | slower |
| 1024 | 5178.0 us | 5319.0 us | slower |
| 2048 | 9670.0 us | 9709.0 us | about flat/slower |
| 4096 | 18469.0 us | 18451.0 us | about flat |
| 8192 | 36373.0 us | 36267.0 us | about flat |

50-run small-M cross-check did not confirm the 30-run M32/M256 gains:

| tokens | split 50-run | split + L2 no-dispatch 50-run | result |
|---:|---:|---:|---|
| 32 | 1207.9 us | 1272.5 us | slower |
| 64 | 1225.6 us | 1215.0 us | slight/noisy faster |
| 128 | 1256.8 us | 1258.4 us | flat |
| 256 | 1752.3 us | 1815.9 us | slower |

Conclusion: do not enable L2 no-dispatch by default. It is noisy and not robust.

### Attempt: NVFP4 Dequant Placement / Load Variants

All variants were tested with `DG_SM90_MOE_SPLIT_L1_L2=1`. Correctness gate was M=32, weight_scale=0.05 before benchmark.

| variant | correctness | 10-run M32 | 10-run M128 | 10-run M512 | conclusion |
|---|---|---:|---:|---:|---|
| loader dequant off / math-side dequant | PASS | 1512.6 us | 1483.4 us | 3512.0 us | slower |
| math-side + direct scale gmem | PASS | 1699.5 us | 1768.4 us | 4086.0 us | slower |
| math-side + packed B scratch | PASS | 1433.7 us | 1455.4 us | 3773.0 us | slower |
| math-side + strided gmem load | PASS | 3511.0 us | 3515.0 us | 7725.0 us | much slower |
| math-side + split dequant barrier | compile fail | n/a | n/a | n/a | old helper `ptx::mbarrier_wait` missing |

Conclusion: current loader-side dequant remains the best measured NVFP4 bridge path. This matches the phase profile: math-side dequant wait was only ~400 cycles, so moving dequant back into math threads mostly increases contention rather than hiding useful work.

### Attempt: Exact Split-Kernel Timing in Benchmark Script

Hypothesis: replace substring matching + `*2` with exact timing of `sm90_nvfp4_mega_moe_l1` and `sm90_nvfp4_mega_moe_l2`.

Result: reverted. The profiler CUDA kernel names are generated from the templated device symbol, not the JIT build name, so exact `_l1` / `_l2` string matching returned 0 us. The retained benchmark numbers continue to use the existing substring aggregation plus `*2`; explicit `DG_SM90_MOE_SPLIT_L1_L2=1` is used for retained split measurements.


### Attempt: Re-test BN256 Under Split L1/L2

Hypothesis: BN256 was previously slower in the old fused/default structure, but split L1/L2 changes resource pressure and might make larger N tiles useful again.

Correctness:

```bash
DG_SM90_MOE_SPLIT_L1_L2=1 DG_SM90_NVFP4_BLOCK_N=256 DG_SM90_NVFP4_EPILOGUE_THREADS=256 \
python3 -u tests/test_nvfp4_mega_moe_sm90_correctness.py --batches 32 256 --weight-scales 0.05
```

Result: PASS.

Benchmark, 8 GPUs, `--num-tests 30`:

| tokens | BN128 split retained | BN256 split | result |
|---:|---:|---:|---|
| 32 | 1208.2 us | 1330.8 us | slower |
| 64 | 1272.9 us | 1378.7 us | slower |
| 128 | 1236.9 us | 1421.2 us | slower |
| 256 | 1766.3 us | 1951.8 us | slower |
| 512 | 3083.0 us | 3399.0 us | slower |
| 1024 | 5216.0 us | 5828.0 us | slower |
| 2048 | 9719.0 us | 10764.0 us | slower |
| 4096 | 18463.0 us | 20535.0 us | slower |
| 8192 | 36302.0 us | 40391.0 us | slower |

Conclusion: keep BN128. BN256 remains a bad fit for the current Hopper FP4->FP8 bridge.

### Attempt: Split-Path num_stages Sweep and Dynamic Default

Hypothesis: after enabling split L1/L2, max stages may not be the best default for every M. Screened stages 2..7 with M=32/128/512, `--num-tests 10`; all screened stages passed M=32 correctness.

Screening result:

| stages | M32 | M128 | M512 | conclusion |
|---:|---:|---:|---:|---|
| 2 | 2179.0 us | 1859.1 us | 4590.0 us | too slow |
| 3 | 1532.0 us | 1564.6 us | 3720.0 us | too slow |
| 4 | 1245.2 us | 1255.5 us | 3340.0 us | okay but not best |
| 5 | 1182.8 us | 1208.5 us | 3027.0 us | promising |
| 6 | 1238.5 us | 1279.8 us | 2991.0 us | promising for M512 |
| 7 | 1465.8 us | 1224.2 us | 3026.0 us | noisy M32, otherwise okay |

30-run confirmation:

| tokens | retained max-stage split | stage5 | stage6 | dynamic default after patch |
|---:|---:|---:|---:|---:|
| 32 | 1208.2 us | 1243.7 us | 1234.3 us | 1366.7 us in all-size run; 1208.2 us in isolated 50-run |
| 64 | 1272.9 us | 1238.4 us | 1236.6 us | 1239.2 us |
| 128 | 1236.9 us | 1214.0 us | 1210.7 us | 1238.9 us |
| 256 | 1766.3 us | 1836.6 us | 1807.7 us | 1802.0 us |
| 512 | 3083.0 us | 3086.0 us | 2997.0 us | 3013.0 us |
| 1024 | 5216.0 us | 5266.0 us | 5221.0 us | 5199.0 us |
| 2048 | 9719.0 us | 9786.0 us | 9687.0 us | 9704.0 us |
| 4096 | 18463.0 us | 18663.0 us | 18447.0 us | 18448.0 us |
| 8192 | 36302.0 us | 36644.0 us | 36315.0 us | 36271.0 us |

Implementation retained:

- Host default now uses max stages for M=32.
- For BN128 and M>32, host caps default stages at 6 if max stages is larger.
- `DG_SM90_NVFP4_NUM_STAGES` still overrides the default.

Correctness after dynamic default:

```bash
python3 -u tests/test_nvfp4_mega_moe_sm90_correctness.py \
  --batches 32 256 --weight-scales 0.001 0.05 0.3
```

Result: PASS.

Conclusion: dynamic stage default is a small, noisy improvement for M64+ and neutral after isolated M32 recheck. It is not enough to close the W8A8 gap.


### Attempt: BN256 Loader-Side Dequant

Hypothesis: BN256 is slow partly because it disables loader-side dequant and falls back to math-side dequant. Since all 128 non-epilogue threads call `dequant_loaded_b_stage`, add a BN256 specialization where each non-epilogue thread dequants two B rows.

Implementation attempted:

- Temporarily allowed `nvfp4_loader_dequant` for `block_n=256, epilogue_threads=256`.
- Added a `kNumEpilogueThreads == 256` branch in `dequant_loaded_b_stage` using `dequant_smem_b_inplace_two_rows<128, 8>`.

Correctness:

```bash
DG_SM90_MOE_SPLIT_L1_L2=1 DG_SM90_NVFP4_BLOCK_N=256 DG_SM90_NVFP4_EPILOGUE_THREADS=256 \
python3 -u tests/test_nvfp4_mega_moe_sm90_correctness.py --batches 32 256 --weight-scales 0.05
```

Result: PASS.

Benchmark, 8 GPUs, `--num-tests 30`:

| tokens | BN256 math-side dequant | BN256 loader-dequant | result |
|---:|---:|---:|---|
| 32 | 1330.8 us | 6379.0 us | much slower |
| 64 | 1378.7 us | 1549.6 us | slower |
| 128 | 1421.2 us | 1427.7 us | flat/slower |
| 256 | 1951.8 us | 2011.3 us | slower |
| 512 | 3399.0 us | 3374.0 us | flat |
| 1024 | 5828.0 us | 5815.0 us | flat |
| 2048 | 10764.0 us | 10722.0 us | flat |
| 4096 | 20535.0 us | 20544.0 us | flat |
| 8192 | 40391.0 us | 40294.0 us | flat |

Conclusion: failed. BN256 loader-dequant was reverted. It likely serializes A/B loader warps on the full barrier and destroys small-M overlap.


### Attempt: Split Path 64/64 Dispatch/Non-Epilogue Topology

Hypothesis: the smaller dispatch/non-epilogue topology was slower in fused mode, but split L1/L2 might reduce the pressure enough to make it useful.

Correctness:

```bash
DG_SM90_MOE_SPLIT_L1_L2=1 \
DG_SM90_NVFP4_DISPATCH_THREADS=64 DG_SM90_NVFP4_NON_EPILOGUE_THREADS=64 \
python3 -u tests/test_nvfp4_mega_moe_sm90_correctness.py --batches 32 --weight-scales 0.05
```

Result: PASS.

10-run screening:

| tokens | default split | split 64/64 | result |
|---:|---:|---:|---|
| 32 | ~1208 us | 1511.8 us | slower |
| 128 | ~1237 us | 1543.6 us | slower |
| 512 | ~3083 us | 3659.0 us | slower |

Conclusion: still invalid as a performance direction. Keep default 128 dispatch + 128 non-epilogue + 128 epilogue for BN128.


### Attempt: Combine Chunk Count

Hypothesis: NVFP4 forces `hidden=7168` combine into 7 chunks, while the PR323 W8A8 non-split-MN path uses its default chunk heuristic. Fewer chunks may reduce combine TMA/load/store fixed overhead.

Temporary change:

- Replaced `kNumChunks = (kHidden == 7168) ? 7 : kNumDefaultChunks` with `kNumChunks = kNumDefaultChunks`.

Correctness:

```bash
python3 -u tests/test_nvfp4_mega_moe_sm90_correctness.py --batches 32 --weight-scales 0.05
```

Result: PASS.

Benchmark, 8 GPUs:

| tokens | retained 7 chunks | 2/default chunks | result |
|---:|---:|---:|---|
| 32 | 1208.2 us | 1178.6 us in 50-run; 1292.0 us in all-size 30-run | noisy/mixed |
| 64 | 1239.2 us | 1220.1 us in 50-run; 1212.0 us in all-size 30-run | faster |
| 128 | 1238.9 us | 1279.1 us in 50-run; 1279.5 us in all-size 30-run | slower |
| 256 | 1802.0 us | 1783.9 us in 50-run; 1746.8 us in all-size 30-run | faster/noisy |
| 512 | 3013.0 us | 3001.0 us | slight faster |
| 1024 | 5199.0 us | 5163.0 us | slight faster |
| 2048 | 9704.0 us | 9621.0 us | slight faster |
| 4096 | 18448.0 us | 18352.0 us | slight faster |
| 8192 | 36271.0 us | 36060.0 us | slight faster |

Follow-up attempted:

- Added a temporary size-gated template/env knob to use 2 chunks for M<=64 or M>=512, and 7 chunks for M128/256.
- Correctness PASSed for M=32/128/256.
- 30-run benchmark was not robust: M64 and M1024 regressed, M128 improved, M256 regressed.

Conclusion: not retained. The global and gated chunk strategies are both noisy and can regress default cases. Reverted to the original 7-chunk combine policy. Post-revert correctness PASSed for M=32, weight_scale=0.05.


### Attempt: L2 Arrival Counter for NVFP4 BN128

Hypothesis: PR323 W8A8 has an L2 arrival-counter mode for some split paths. NVFP4 BN128 uses a bitmask update and a CTA-wide epilogue sync before publishing each L1 tile. A counter could remove that sync and reduce fixed overhead.

Implementation attempted as opt-in:

```bash
DG_SM90_NVFP4_L2_ARRIVAL_COUNTER=1
```

Temporary changes:

- Added a template flag and host env knob.
- L2 wait path used a 32-bit counter expected value instead of the 64-bit bitmask.
- L1 publish path used `red_add_rel` instead of `red_or_rel_gpu`, avoiding the pre-publish full epilogue sync in the counter branch.

Correctness:

```bash
DG_SM90_NVFP4_L2_ARRIVAL_COUNTER=1 \
python3 -u tests/test_nvfp4_mega_moe_sm90_correctness.py --batches 32 --weight-scales 0.05
```

Result: PASS.

Benchmark:

| tokens | retained default | arrival-counter | result |
|---:|---:|---:|---|
| 32 | 1208.2 us | 1191.8 us 30-run; 1191.0 us 50-run | slight faster |
| 64 | 1239.2 us | 1241.4 us 30-run; 4762.0 us 50-run outlier | unsafe/noisy |
| 128 | 1238.9 us | 1281.9 us 30-run; 1314.6 us 50-run | slower |
| 256 | 1802.0 us | 1732.4 us 30-run; 1767.1 us 50-run | faster/noisy |
| 512 | 3013.0 us | 3030.0 us | flat/slower |
| 1024 | 5199.0 us | 5200.0 us | flat |
| 2048 | 9704.0 us | 9696.0 us | flat |
| 4096 | 18448.0 us | 18487.0 us | flat/slower |
| 8192 | 36271.0 us | 36257.0 us | flat |

Conclusion: not retained. M32/M256 have small positive signal, but M128 regresses and M64 produced a severe 50-run outlier. The opt-in code was reverted. Post-revert correctness PASSed for M=32, weight_scale=0.05.


### Retained State After Reverted Experiments

After reverting the combine-chunk and L2-arrival-counter experiments, the retained code is:

- Split L1/L2 enabled by default for NVFP4.
- Dynamic stage default retained: M=32 uses max stages; BN128 and M>32 caps default at 6 when max is larger.
- Loader-side dequant remains the default for BN128.
- BN256, L2 no-dispatch, arrival-counter, combine-chunk changes remain non-default/reverted.

Correctness sanity after reverts:

```bash
python3 -u tests/test_nvfp4_mega_moe_sm90_correctness.py --batches 32 --weight-scales 0.05
```

Result: PASS.

Retained benchmark after reverts, 8 GPUs, `--num-tests 30`:

```bash
DG_SM90_MOE_SPLIT_L1_L2=1 python3 -u tests/bench_nvfp4_mega_moe_sm90.py \
  --num-processes 8 --num-tests 30 \
  --batches 32 64 128 256 512 1024 2048 4096 8192
```

| tokens | retained NVFP4 | PR323 W8A8 | gap vs W8A8 | speedup vs fused NVFP4 pre-split |
|---:|---:|---:|---:|---:|
| 32 | 1223.0 us | 810.6 us | 1.509x slower | 1.067x |
| 64 | 1219.0 us | 992.7 us | 1.228x slower | 1.089x |
| 128 | 1227.2 us | 848.1 us | 1.447x slower | 1.054x |
| 256 | 1862.9 us | 1427.0 us | 1.305x slower | 1.041x |
| 512 | 2985.0 us | 2238.0 us | 1.334x slower | 1.125x |
| 1024 | 5233.0 us | 3727.0 us | 1.404x slower | 1.053x |
| 2048 | 9680.0 us | 6232.0 us | 1.553x slower | 1.051x |
| 4096 | 18501.0 us | 11337.0 us | 1.632x slower | 1.054x |
| 8192 | 36299.0 us | 21921.0 us | 1.656x slower | 1.055x |

Conclusion: retained changes are correctness-clean and improve the original fused NVFP4 bridge, but the kernel is still far from W8A8. Remaining gap is structural: Hopper still materializes FP4 weights into FP8 shared memory before WGMMA, so global bandwidth savings are not translating to tensor-core throughput.


### Attempt: Cluster Size 2

Hypothesis: the NVFP4 kernel already has `kClusterSize` plumbing. Cluster size 2 could let paired CTAs share/multicast some TMA traffic and help medium/large M.

Temporary change:

- Added an opt-in host env `DG_SM90_NVFP4_CLUSTER_SIZE=2`.

Correctness:

```bash
DG_SM90_NVFP4_CLUSTER_SIZE=2 \
python3 -u tests/test_nvfp4_mega_moe_sm90_correctness.py --batches 32 256 --weight-scales 0.05
```

Result: PASS.

10-run screening:

| tokens | retained cluster1 | cluster2 | result |
|---:|---:|---:|---|
| 32 | ~1223 us | 1238.6 us | slower/noisy |
| 128 | ~1227 us | 1221.7 us | flat/slight faster |
| 512 | ~2985 us | 3038.0 us | slower |
| 1024 | ~5233 us | 5254.0 us | slower |

Conclusion: not retained. Reverted the cluster env and kept `cluster_size=1`.


### Attempt: Re-open BM32 mma.sync Path

Important process note: host-side changes in `csrc/jit_kernels/impls/sm90_nvfp4_mega_moe.hpp` require rebuilding `deep_gemm._C`. Earlier mma.sync checks before rebuild did not actually instantiate BM32; JIT cache inspection showed BM64 kernels.

Hypothesis: the previous BM32 mma.sync illegal memory access was caused by missing host-side shared-memory accounting. The kernel reserves `SMEM_CD_ACCUM_SIZE = BLOCK_M * BLOCK_N * sizeof(float)` for mma.sync FP32 accumulator staging, but the host smem-size calculation did not include it.

Temporary changes:

- Added opt-in `DG_SM90_NVFP4_MMA_SYNC=1` setting `block_m=32`, `block_n=128`, activation TMA swizzle 0.
- Disabled direct L2 scatter for mma.sync, matching kernel-side `!kUseMMASync` gating.
- Added host smem accounting for `SMEM_CD_ACCUM_SIZE`.
- Added `dequant_barriers[stage].wait()` in the mma.sync GEMM loop when loader-side dequant is active.
- Rebuilt extension with `touch csrc/python_api.cpp && python setup.py build_ext --inplace -j 16`.

Result 1:

- M32 correctness no longer crashed with illegal memory access.
- Output became non-finite (`nan`), so the old illegal memory issue was at least partly fixed.

Follow-up:

- Found a likely CUTE tiling bug: `MMASyncTiled` used `_4` N atoms, covering only 32 columns for a 16x8 atom, while the epilogue reads 128 columns from `smem_accum_f32`.
- Temporarily changed N tiling to `BLOCK_N / 8`.

Result 2:

- Output became finite but numerically exploded (`abs_max ~9.6e35`, cosine_mean ~0.001), so full-N tiling alone is not a correct port.
- Likely remaining issue is CUTE mma.sync A/B layout/atom tiling semantics or accumulator-to-row/column mapping, not just smem size.

Action:

- Reverted the mma.sync host exposure, smem opt-in changes, dequant wait, and CUTE tile change.
- Rebuilt extension again after revert.
- Retained default correctness after rebuild PASSed for M=32, weight_scale=0.05.

Conclusion: BM32 mma.sync is now better understood but not retained. The next mma.sync attempt should be a dedicated port with a standalone GEMM/dequant unit test for the CUTE tile layout before wiring it back into MegaMoE.


### Retained Benchmark After Host Rebuild

After the mma.sync experiment, the extension was rebuilt again with the retained code. This matters because `csrc/jit_kernels/impls/sm90_nvfp4_mega_moe.hpp` changes only take effect after rebuilding `deep_gemm._C`.

Correctness after rebuild:

```bash
python3 -u tests/test_nvfp4_mega_moe_sm90_correctness.py --batches 32 --weight-scales 0.05
```

Result: PASS.

Benchmark after rebuild, 8 GPUs, `--num-tests 30`:

| tokens | retained NVFP4 after rebuild | PR323 W8A8 | gap vs W8A8 |
|---:|---:|---:|---:|
| 32 | 1232.6 us | 810.6 us | 1.521x slower |
| 64 | 1370.2 us | 992.7 us | 1.380x slower |
| 128 | 1227.6 us | 848.1 us | 1.447x slower |
| 256 | 1804.1 us | 1427.0 us | 1.264x slower |
| 512 | 3072.0 us | 2238.0 us | 1.373x slower |
| 1024 | 5204.0 us | 3727.0 us | 1.396x slower |
| 2048 | 9687.0 us | 6232.0 us | 1.554x slower |
| 4096 | 18449.0 us | 11337.0 us | 1.627x slower |
| 8192 | 36277.0 us | 21921.0 us | 1.655x slower |

M64 is likely affected by the known long-tail behavior in this run; other points are close to the previous retained numbers.


### Attempt: All Non-Epilogue Warps Participate in Loader Dequant

Hypothesis: current loader-dequant uses only the two idle non-epilogue warps, with each thread dequantizing two B rows. Letting the A/B TMA loader warps also participate would use all four non-epilogue warps and reduce per-stage dequant latency, which might help small M.

Temporary change:

- In `dequant_loaded_b_stage`, changed the `kNumMMANonEpilogueWarps == 4 && kNumEpilogueThreads == 128` path from two idle warps using `dequant_smem_b_inplace_two_rows<64>` to all 128 non-epilogue threads using `dequant_smem_b_inplace<kNumNonEpilogueThreads>`.
- This was a JIT header-only experiment and did not require rebuilding `deep_gemm._C`.

Correctness:

```bash
DG_SM90_MOE_SPLIT_L1_L2=1 \
python3 -u tests/test_nvfp4_mega_moe_sm90_correctness.py \
  --batches 32 --weight-scales 0.05 --reference-mode fp8-bridge \
  --cosine-mean-threshold 0.99 --cosine-min-threshold 0.99 \
  --norm-ratio-min 0.99 --norm-ratio-max 1.01
```

Result: PASS.

10-run screening:

| tokens | all-warp loader dequant | retained after rebuild | result |
|---:|---:|---:|---|
| 32 | 1504.9 us | 1232.6 us | slower |
| 64 | 1493.8 us | 1370.2 us | slower |
| 128 | 1530.7 us | 1227.6 us | slower |
| 256 | 2084.5 us | 1804.1 us | slower |
| 512 | 3619.0 us | 3072.0 us | slower |
| 1024 | 6097.0 us | 5204.0 us | slower |

Conclusion: not retained. The result suggests the current design needs the A/B loader warps to keep issuing future TMA stages; using them for dequant reduces dequant instruction parallelism but destroys pipeline overlap.


### Recheck: Fused Single-Kernel Path vs Split Default

Reason: current retained default is split L1/L2. If the split path's two kernel launches and separate scheduling were the main source of large-M overhead, size-gating back to the fused single-kernel path could help.

Correctness for fused path:

```bash
DG_SM90_MOE_SPLIT_L1_L2=0 \
python3 -u tests/test_nvfp4_mega_moe_sm90_correctness.py \
  --batches 32 256 --weight-scales 0.05 --reference-mode fp8-bridge \
  --cosine-mean-threshold 0.99 --cosine-min-threshold 0.99 \
  --norm-ratio-min 0.99 --norm-ratio-max 1.01
```

Result: PASS.

10-run fused benchmark:

| tokens | fused single-kernel | retained split after rebuild | result |
|---:|---:|---:|---|
| 32 | 1267.0 us | 1232.6 us | slower/slightly noisy |
| 128 | 1312.0 us | 1227.6 us | slower |
| 256 | 2376.0 us | 1804.1 us | slower |
| 512 | 3115.0 us | 3072.0 us | flat/slower |
| 1024 | 5455.0 us | 5204.0 us | slower |
| 2048 | 10225.0 us | 9687.0 us | slower |
| 4096 | 19729.0 us | 18449.0 us | slower |
| 8192 | 38227.0 us | 36277.0 us | slower |

Conclusion: no size-gated fused default for now. The split path remains the retained default despite the extra launch, because it is faster or equal across the checked sizes.


### Attempt: Loader-Dequant Direct Global Scale Loads

Hypothesis: loader-dequant currently TMA-loads the 1KB UE4M3 scale tile into shared memory for every B stage. Letting the dequant warps read scale bytes directly from global memory could remove SFB smem, reduce full-barrier transaction bytes, and increase max stages.

Temporary changes:

- Allowed `DG_SM90_NVFP4_DIRECT_SCALE_GMEM=1` to coexist with loader-dequant in the host launcher.
- Changed kernel `kDirectScaleGmem` gating so loader-dequant could see it.
- Passed the current `scale_tile` pointer to `dequant_loaded_b_stage`; idle dequant warps computed the same expert/N/K scale tile as the B loader.
- Rebuilt `deep_gemm._C` after host launcher changes.

Correctness command:

```bash
DG_SM90_MOE_SPLIT_L1_L2=1 \
DG_SM90_NVFP4_DIRECT_SCALE_GMEM=1 \
python3 -u tests/test_nvfp4_mega_moe_sm90_correctness.py \
  --batches 32 256 --weight-scales 0.05 --reference-mode fp8-bridge \
  --cosine-mean-threshold 0.99 --cosine-min-threshold 0.99 \
  --norm-ratio-min 0.99 --norm-ratio-max 1.01
```

Result: FAIL before benchmarking.

- First case M=32 produced non-finite output (`abs_max=nan`, `finite=False`).
- The failed path was reverted immediately.
- Extension was rebuilt after revert.
- Retained default M=32 correctness was rerun and PASSed.

Conclusion: not retained. Either the loader-side direct global scale pointer path still has an addressing/synchronization bug, or direct global scale loads are incompatible with the current loader-dequant barrier structure. Do not use this path without a dequant-level kernel test that validates direct scale addressing independently.


### Attempt: Split SFA TMA for NVFP4

Hypothesis: `DG_SM90_MOE_SPLIT_SFA_TMA=1` uses an extra producer warp for SFA TMA. It disables loader-dequant in the current NVFP4 launcher, so this tests whether better producer overlap can compensate for moving FP4->FP8 dequant back onto the math path.

Correctness:

```bash
DG_SM90_MOE_SPLIT_L1_L2=1 \
DG_SM90_MOE_SPLIT_SFA_TMA=1 \
python3 -u tests/test_nvfp4_mega_moe_sm90_correctness.py \
  --batches 32 --weight-scales 0.05 --reference-mode fp8-bridge \
  --cosine-mean-threshold 0.99 --cosine-min-threshold 0.99 \
  --norm-ratio-min 0.99 --norm-ratio-max 1.01
```

Result: PASS.

10-run screening:

| tokens | split-SFA | retained after rebuild | result |
|---:|---:|---:|---|
| 32 | 1677.9 us | 1232.6 us | slower |
| 128 | 1560.8 us | 1227.6 us | slower |
| 512 | 3556.0 us | 3072.0 us | slower |
| 1024 | 6294.0 us | 5204.0 us | slower |

Conclusion: not retained. Loader-dequant is necessary; moving dequant to math-side costs more than split-SFA TMA can recover.


### Attempt: Disable L1 Dual-K Accumulation

Hypothesis: unlike the earlier W4A8 BN256 split-N shape, the current NVFP4 BN128 shape can activate `DG_SM90_MOE_L1_DUAL_K` by default. Disabling it might reduce small-M overhead or register pressure.

Correctness:

```bash
DG_SM90_MOE_SPLIT_L1_L2=1 \
DG_SM90_MOE_L1_DUAL_K=0 \
python3 -u tests/test_nvfp4_mega_moe_sm90_correctness.py \
  --batches 32 --weight-scales 0.05 --reference-mode fp8-bridge \
  --cosine-mean-threshold 0.99 --cosine-min-threshold 0.99 \
  --norm-ratio-min 0.99 --norm-ratio-max 1.01
```

Result: PASS.

30-run A/B, same time window:

| tokens | L1 dual-K off | default | result |
|---:|---:|---:|---|
| 64 | 1264.4 us | 1245.2 us | slower |
| 128 | 1284.7 us | 1242.3 us | slower |
| 256 | 1927.4 us | 1908.8 us | slower |
| 512 | 3118.0 us | 3002.0 us | slower |
| 1024 | 5313.0 us | 5297.0 us | slower/flat |

Conclusion: not retained. The default L1 dual-K path is still better for this BN128 NVFP4 topology.


### Attempt: Disable L2 Dual Accumulation

Hypothesis: `DG_SM90_MOE_L2_DUAL_ACCUM=1` may add register pressure in the NVFP4 bridge. Disabling it could help larger M if L2 combine/scatter or epilogue pressure dominates.

Correctness:

```bash
DG_SM90_MOE_SPLIT_L1_L2=1 \
DG_SM90_MOE_L2_DUAL_ACCUM=0 \
python3 -u tests/test_nvfp4_mega_moe_sm90_correctness.py \
  --batches 32 --weight-scales 0.05 --reference-mode fp8-bridge \
  --cosine-mean-threshold 0.99 --cosine-min-threshold 0.99 \
  --norm-ratio-min 0.99 --norm-ratio-max 1.01
```

Result: PASS.

30-run large-M check:

| tokens | L2 dual accum off | default 30-run subset | result |
|---:|---:|---:|---|
| 256 | 1903.6 us | 1908.8 us | flat/noisy |
| 512 | 3135.0 us | 3002.0 us | slower |
| 1024 | 5208.0 us | 5297.0 us | slight faster/noisy |

10-run small-M screening was also mixed: M32 and M128 were slower, M64 was near flat.

Conclusion: not retained. The only positive signal is a small M1024 improvement, but it is not enough to justify size-gating because M512 regresses and the target focus remains small/medium M.


### Attempt: L2 N-Major Schedule

Hypothesis: `DG_SM90_MOE_L2_NMAJOR=1` changes L2 block traversal order and could improve L2 weight locality or reduce tail effects for medium M.

Correctness:

```bash
DG_SM90_MOE_SPLIT_L1_L2=1 \
DG_SM90_MOE_L2_NMAJOR=1 \
python3 -u tests/test_nvfp4_mega_moe_sm90_correctness.py \
  --batches 32 --weight-scales 0.05 --reference-mode fp8-bridge \
  --cosine-mean-threshold 0.99 --cosine-min-threshold 0.99 \
  --norm-ratio-min 0.99 --norm-ratio-max 1.01
```

Result: PASS.

10-run screening showed apparent M256/M512 gains but a large M32 long-tail and M1024 regression, so M256/M512 were rechecked with 30 runs.

30-run medium-M check:

| tokens | L2 N-major | default 30-run subset | result |
|---:|---:|---:|---|
| 256 | 1906.9 us | 1908.8 us | flat/noisy |
| 512 | 3060.0 us | 3002.0 us | slower |

Conclusion: not retained. The 10-run gain did not survive 30-run validation.


### Attempt: L1 N-Major Schedule

Hypothesis: `DG_SM90_MOE_L1_NMAJOR=1` changes L1 block traversal order and could reduce expert/N tail effects for small and medium M.

Correctness:

```bash
DG_SM90_MOE_SPLIT_L1_L2=1 \
DG_SM90_MOE_L1_NMAJOR=1 \
python3 -u tests/test_nvfp4_mega_moe_sm90_correctness.py \
  --batches 32 --weight-scales 0.05 --reference-mode fp8-bridge \
  --cosine-mean-threshold 0.99 --cosine-min-threshold 0.99 \
  --norm-ratio-min 0.99 --norm-ratio-max 1.01
```

Result: PASS.

10-run screening showed an apparent M256 improvement, so M256/M512/M1024 were rechecked with 30 runs.

30-run check:

| tokens | L1 N-major | default 30-run subset | result |
|---:|---:|---:|---|
| 256 | 1900.4 us | 1908.8 us | flat/noisy |
| 512 | 3097.0 us | 3002.0 us | slower |
| 1024 | 5244.0 us | 5297.0 us | slight faster/noisy |

Conclusion: not retained. The positive signals are too small and inconsistent, while M512 regresses.


### Structural Finding: W8A8 Topology Is Not a Simple NVFP4 Knob

W8A8 PR323 M32 config print shows:

```text
block_m=64, block_n=128, block_k=128, num_stages=7,
num_dispatch_threads=64, num_non_epilogue_threads=64, num_epilogue_threads=256
```

The retained NVFP4 path intentionally uses `128/128/128` threads so two non-epilogue idle warps can perform loader-dequant. Simply opening `64/64/256` for NVFP4 is not safe yet:

- NVFP4 `kSplitNWarpgroups` currently only supports `BLOCK_N=256, epilogue_warpgroups=2`, where `WG_BLOCK_N=128`.
- W8A8 supports generic split-N, including `BLOCK_N=128, epilogue_warpgroups=2`, where `WG_BLOCK_N=64` and `WG_L1_OUT_BLOCK_N=32`.
- W8A8 has `kSplitNSharesSF`: both N-split warpgroups reduce a shared per-64-column L1-output scale through smem scratch, then one WG issues a combined TMA store.
- NVFP4 epilogue still assumes `WG_L1_OUT_BLOCK_N % 64 == 0`, so BN128+2WG would fail or compute wrong SF unless this logic is ported.
- NVFP4 math-side dequant also assumes the number of dequanting epilogue threads matches `LOAD_BLOCK_N`; supporting epilogue=256 needs a masked dequant helper where only the first 128 threads read/write rows but all 256 participate in the safety barrier.

Conclusion: matching W8A8 topology requires a real split-N epilogue port plus masked dequant, not just relaxing host asserts. This is the next structural path if we continue chasing W8A8-like small-M latency.


## 2026-06-11: W8A8-like BN128 64/64/256 split-N epilogue experiment

Hypothesis: PR323 W8A8 small-M wins partly from a 64 dispatch / 64 non-epilogue / 256 epilogue topology. Porting NVFP4 BN128 to a split-N epilogue might reduce fixed overhead enough to close the M32 gap.

Changes tried:
- Added masked math-side SMEM-B dequant for `kNumEpilogueThreads > LOAD_BLOCK_N` so BN128 could run with 256 epilogue threads.
- Generalized NVFP4 split-N detection from BN256-only to per-WG N64/N128.
- Added a shared-SF L1 epilogue path for BN128 split-N, including cross-WG amax scratch reduction and one TMA store of the full post-SwiGLU tile.
- Relaxed host assertions and TMA descriptor calculation to allow `DG_SM90_NVFP4_DISPATCH_THREADS=64 DG_SM90_NVFP4_NON_EPILOGUE_THREADS=64 DG_SM90_NVFP4_EPILOGUE_THREADS=256`.

Correctness:
- `M=32`, `weight_scale=0.05`, `fp8-bridge` reference: PASS.
- Dequant unit test: PASS.
- CUDA LUT dequant unit test: PASS.

10-run benchmark for the opt-in 64/64/256 topology:

```text
M32    1463.8 us
M64    1486.5 us
M128   1548.2 us
M256   2145.2 us
M512   3663.0 us
M1024  6555.0 us
```

Decision: rejected and reverted. It is slower than the retained default at every measured size; M32 regresses from roughly 1230 us to 1464 us. The extra 256-epilogue shared-SF/scratch/barrier work costs more than the PR323-style topology saves for NVFP4. After revert and rebuild, retained default M32 correctness passed again (`cosine_min=0.9996`, `cosine_mean=0.9997`, `norm_ratio=0.9997`).


## 2026-06-11: BN128 `num_stages=7` default experiment

Hypothesis: PR323 W8A8 small/medium M reports `num_stages=7`; NVFP4 BN128 defaults to 6 stages for `num_tokens > 32`. Increasing the default to 7 might improve pipeline overlap.

Correctness:
- Explicit `DG_SM90_NVFP4_NUM_STAGES=7`, M32, weight_scale=0.05: PASS.
- After temporarily making stage7 the BN128 default, expanded correctness over M32/M256/M1024 and weight_scale 0.001/0.05/1.0: PASS, including dequant unit and CUDA LUT unit tests.

Performance observations:
- Initial 30-run A/B suggested small gains for M32-M256 and flat large M.
- Final all-size 30-run showed M256 regressing to ~1921 us.
- Follow-up 50-run same-shape A/B showed stage7 was worse than stage6:

```text
stages=6: M64 1195.8 us, M128 1217.3 us, M256 1919.5 us
stages=7: M64 1212.5 us, M128 1309.4 us, M256 1950.6 us
```

Decision: rejected and reverted. The early gain was noise/shape-sensitive; retained default remains the previous rule: BN128 uses 6 stages for `num_tokens > 32`, otherwise max stages. Rebuilt after revert and M32 correctness passed again (`cosine_min=0.9996`, `cosine_mean=0.9997`, `norm_ratio=0.9997`).


## 2026-06-11: Retained combine chunk count change, 7 chunks -> default 2 chunks

Hypothesis: PR323 W8A8 only uses the 7-way combine split for split-MN topology. NVFP4 was using 7 chunks unconditionally for hidden=7168, even in the default BN128/single-M-WG path. This increases combine TMA/load/store loop count and fixed combine overhead. Match the PR323 default by using `kNumDefaultChunks` for NVFP4 combine.

Code change retained:

```cpp
constexpr uint32_t kNumChunks = kNumDefaultChunks;
```

Correctness:
- Dequant unit test: PASS.
- CUDA LUT dequant unit test: PASS.
- M32/M256 with weight_scale 0.001/0.05/1.0, `fp8-bridge` reference, cosine/norm thresholds 0.99/0.99/0.99-1.01: PASS.

Benchmark, 30-run all-size sweep after the change:

```text
M32    1213.8 us
M64    1202.0 us
M128   1217.0 us
M256   1721.9 us
M512   3047.0 us
M1024  5224.0 us
M2048  9623.0 us
M4096  18311.0 us
M8192  36031.0 us
```

50-run same-shape validation versus the previous combine=7 retained path:

```text
previous combine=7: M64 1195.8 us, M128 1217.3 us, M256 1919.5 us
new combine=2:      M64 1195.0 us, M128 1213.3 us, M256 1882.4 us, M512 2973.0 us
```

Phase profile evidence:
- M32 combine_reduce avg cycles: 1868 -> 945.
- M256 combine_reduce avg cycles: 14879 -> 7943.
- M1024 combine_reduce avg cycles: 65800 -> 35977.

Decision: retained. This is a real PR323-aligned fixed-overhead reduction. It does not close the W8A8 gap by itself because the dominant time is still GEMM/math/dispatch, but it reduces combine fixed cost and improves medium/large cases without correctness risk.


## 2026-06-11: L2 direct scatter off experiment

Hypothesis: PR323 W8A8 uses an smem-staged L2 epilogue scatter. NVFP4 default uses direct L2 scatter from registers to the combine buffer. Disabling direct scatter could reduce address/metadata work or match W8A8 behavior for small M.

Temporary change:
- Host `direct_l2_scatter = false`, forcing the smem-staged L2 scatter path.

Correctness:
- M32, weight_scale=0.05: PASS, including dequant unit and CUDA LUT unit tests.

Benchmark:

10-run first signal:
```text
M32 1194.6 us, M64 1195.4 us, M128 1222.6 us, M256 1731.3 us, M512 3097.0 us, M1024 5186.0 us
```

30-run validation:
```text
M32 1220.3 us, M64 1302.6 us, M128 1226.7 us, M256 1809.5 us, M512 3004.0 us, M1024 5235.0 us
```

Decision: rejected and reverted. The 10-run small-M signal did not survive 30-run; direct L2 scatter remains the retained default. Rebuilt after revert and M32 correctness passed.


## 2026-06-11: Combine chunk count 1 experiment

Hypothesis: after 7 -> 2 chunks improved combine overhead, using 1 chunk might reduce loop/TMA count further if register pressure remains acceptable.

Temporary change:
```cpp
constexpr uint32_t kNumChunks = 1;
```

Correctness:
- M32, weight_scale=0.05: PASS, including dequant unit and CUDA LUT unit tests.

10-run benchmark:
```text
M32 1240.8 us, M64 1239.8 us, M128 1228.7 us, M256 1757.4 us, M512 2981.0 us, M1024 5177.0 us
```

Decision: rejected and reverted. One chunk hurts the target small-M cases, likely from increased per-lane combine accumulator/register pressure. Retained path is combine=2 (`kNumDefaultChunks`). Reverted and M32 correctness passed.


## 2026-06-11: Constant-memory LUT lookup experiment

Hypothesis: avoid shared LUT loads during NVFP4 dequant by reading the constant LUT directly through `load_e2m1_ue4m3_lut`. To isolate lookup cost, the experiment kept the shared LUT allocation/preload unchanged and only changed the lookup in dequant helpers.

Correctness:
- M32, weight_scale=0.05: PASS, including dequant unit and CUDA LUT unit tests.

10-run benchmark:
```text
M32 1508.6 us, M64 1455.2 us, M128 1799.2 us, M256 2091.3 us, M512 3525.0 us, M1024 6126.0 us
```

Decision: rejected and reverted. Constant-memory LUT lookup is much slower, likely due non-uniform scale indices causing constant-cache serialization. Retained path remains shared LUT lookup. Reverted and M32 correctness passed.


## 2026-06-11: Single-phase `std::integral_constant` scheduler experiment

Hypothesis: W8A8 PR323 has a split-phase hot path that calls L1/L2 callbacks with compile-time phase constants. NVFP4 split kernels still passed `block_phase` as a runtime enum through `for_each_selected_block`. Passing `std::integral_constant<BlockPhase, Linear1/Linear2>` in single-phase kernels could let the compiler fold phase branches and pointer selections.

Temporary change:
- `for_each_selected_block` passed compile-time phase constants for `kRunL1Phase && !kRunL2Phase` and `!kRunL1Phase && kRunL2Phase`.
- Callback phase parameters were changed from `const sched::BlockPhase&` to `const auto&`.

Correctness:
- M32, weight_scale=0.05: PASS, including dequant unit and CUDA LUT unit tests.

Benchmark:
```text
10-run: M32 1223.3 us, M64 1213.2 us, M128 1258.2 us, M256 1724.0 us, M512 2993.0 us, M1024 5161.0 us
30-run: M32 1184.0 us, M64 1213.0 us, M128 1241.2 us, M256 1765.3 us, M512 2984.0 us, M1024 5149.0 us, M2048 9642.0 us, M4096 18346.0 us, M8192 35999.0 us
50-run: M32 1219.2 us, M64 1188.0 us, M128 1273.9 us, M256 1741.0 us
```

Decision: rejected and reverted. It has useful signals for M64/M256 but regresses M128 and does not consistently improve M32. A size-gated template variant would be too narrow for the current goal. Reverted and M32 correctness passed.


## 2026-06-11: Epilogue register budget 208 -> 256 experiment

Hypothesis: default NVFP4 has spare register budget with 128 dispatch / 128 non-epilogue / 128 epilogue threads. Raising math/epilogue `setmaxnreg` from 208 to 256 might reduce hidden local-memory pressure.

Correctness:
- M32, weight_scale=0.05: PASS, including dequant unit and CUDA LUT unit tests.

10-run benchmark:
```text
M32 1746.7 us, M64 1366.2 us, M128 1222.2 us, M256 1729.1 us, M512 2982.0 us, M1024 5163.0 us
```

Decision: rejected and reverted. More epilogue registers severely hurt small M, likely from occupancy/scheduling pressure rather than spill relief. Retained epilogue register budget remains 208. Reverted and M32 correctness passed.


## 2026-06-11: Current retained benchmark and W8A8 PR323 gap after combine=2

Retained code state for this measurement:
- NVFP4 compact fused path, split L1/L2 default.
- BN128 default, loader-dequant on, direct L2 scatter on.
- Retained combine chunk change: `kNumChunks = kNumDefaultChunks` (2 chunks for hidden=7168 default topology).
- Reverted failed experiments: BN128 64/64/256 split-N, stage7 default, staged L2 scatter, combine=1, constant LUT lookup, constant-phase scheduler, epilogue reg 256.

Correctness after final reverts:
- M32, weight_scale=0.05, `fp8-bridge` reference: PASS.
- Dequant unit test: PASS.
- CUDA LUT dequant unit test: PASS.

W8A8 PR323 same-shape 30-run recheck using `tests/test_mega_moe_hopper.py --fused-only-sweep --hidden 7168 --intermediate-hidden 2048 --num-experts 256 --num-topk 8`:

```text
M32    923.9 us
M64    869.9 us
M128   837.1 us
M256   1460.0 us
M512   2320.0 us
M1024  3518.0 us
M2048  6274.0 us
M4096  11402.0 us
M8192  21904.0 us
```

NVFP4 retained final 30-run using `tests/bench_nvfp4_mega_moe_sm90.py --num-tests 30`:

```text
M32    1206.5 us
M64    1233.9 us
M128   1200.4 us
M256   1773.3 us
M512   3132.0 us
M1024  5159.0 us
M2048  9618.0 us
M4096  18311.0 us
M8192  36019.0 us
```

Gap vs same-shape W8A8 PR323:

| tokens | W8A8 PR323 | NVFP4 retained | NVFP4/W8A8 | regression |
|---:|---:|---:|---:|---:|
| 32 | 923.9 us | 1206.5 us | 1.306x | +30.6% |
| 64 | 869.9 us | 1233.9 us | 1.418x | +41.8% |
| 128 | 837.1 us | 1200.4 us | 1.434x | +43.4% |
| 256 | 1460.0 us | 1773.3 us | 1.215x | +21.5% |
| 512 | 2320.0 us | 3132.0 us | 1.350x | +35.0% |
| 1024 | 3518.0 us | 5159.0 us | 1.466x | +46.6% |
| 2048 | 6274.0 us | 9618.0 us | 1.533x | +53.3% |
| 4096 | 11402.0 us | 18311.0 us | 1.606x | +60.6% |
| 8192 | 21904.0 us | 36019.0 us | 1.644x | +64.4% |

Interpretation:
- The retained combine=2 change reduced combine fixed overhead, but does not solve the primary gap.
- Phase profile still shows math_loop/gemm scheduling dominates; loader-dequant wait is small, but the bridge must still materialize FP8 B in shared memory before WGMMA.
- Large-M being slower despite lower weight bandwidth means the current online FP4->FP8 bridge overhead and/or WGMMA pipeline shape outweighs bandwidth savings for this shape. A real path to W8A8 parity likely needs either a dedicated small-M register/mma.sync kernel or a weight-load-time FP8 shadow cache for latency-sensitive sizes.


## 2026-06-11: FP8 shadow upper-bound experiment for NVFP4 weights

Purpose: isolate how much of the NVFP4 compact fused gap comes from online FP4->FP8 bridge work in the hot kernel. This experiment does not replace the retained compact NVFP4 kernel. It materializes NVFP4 weights to an FP8 shadow cache once at weight-load time using `materialize_nvfp4_fp8_shadow_for_mega_moe_sm90`, then calls the PR323 `fp8_mega_moe` kernel with all-one FP8 block scales.

Temporary scripts only:
- `/tmp/test_nvfp4_shadow_correctness.py`
- `/tmp/bench_nvfp4_shadow.py`

Correctness:
- `fp8-bridge` reference.
- M32/M256.
- weight_scale 0.001, 0.05, 1.0.
- thresholds: cosine_mean >= 0.99, cosine_min >= 0.99, norm_ratio in [0.99, 1.01].
- Result: PASS for all cases.

Correctness highlights:
```text
M32  weight_scale=0.001 cosine_min=0.9996 cosine_mean=0.9997 norm_ratio=0.9993 PASS
M256 weight_scale=0.001 cosine_min=0.9996 cosine_mean=0.9996 norm_ratio=0.9970 PASS
M32  weight_scale=0.05  cosine_min=0.9996 cosine_mean=0.9997 norm_ratio=0.9997 PASS
M256 weight_scale=0.05  cosine_min=0.9996 cosine_mean=0.9996 norm_ratio=0.9970 PASS
M32  weight_scale=1     cosine_min=0.9992 cosine_mean=0.9998 norm_ratio=1.0059 PASS
M256 weight_scale=1     cosine_min=0.9991 cosine_mean=0.9998 norm_ratio=0.9996 PASS
```

First 30-run shadow benchmark had obvious long-tail outliers at M128 and M512:
```text
M32 798.5 us, M64 812.9 us, M128 7425.0 us, M256 1420.0 us, M512 7502.0 us,
M1024 5396.0 us, M2048 11219.0 us, M4096 11386.0 us, M8192 21956.0 us
```

Repeated with 50 runs using the same JIT cache:
```text
M32    840.8 us
M64    866.8 us
M128   845.8 us
M256   1427.0 us
M512   2220.0 us
M1024  3525.0 us
M2048  6212.0 us
M4096  11400.0 us
M8192  21954.0 us
```

Comparison against same-shape PR323 W8A8 baseline:

| tokens | W8A8 PR323 30-run | NVFP4 FP8-shadow 50-run | shadow/W8A8 | delta |
|---:|---:|---:|---:|---:|
| 32 | 923.9 us | 840.8 us | 0.910x | -9.0% |
| 64 | 869.9 us | 866.8 us | 0.996x | -0.4% |
| 128 | 837.1 us | 845.8 us | 1.010x | +1.0% |
| 256 | 1460.0 us | 1427.0 us | 0.977x | -2.3% |
| 512 | 2320.0 us | 2220.0 us | 0.957x | -4.3% |
| 1024 | 3518.0 us | 3525.0 us | 1.002x | +0.2% |
| 2048 | 6274.0 us | 6212.0 us | 0.990x | -1.0% |
| 4096 | 11402.0 us | 11400.0 us | 1.000x | -0.0% |
| 8192 | 21904.0 us | 21954.0 us | 1.002x | +0.2% |

Interpretation:
- FP8 shadow reaches PR323 W8A8 parity across the full sweep and is faster at M32/M256/M512.
- This strongly suggests the retained compact NVFP4 fused gap is dominated by online FP4 unpack + UE4M3 scale application + shared-memory FP8 materialization/barrier/pipeline costs, not by the PR323 dispatch/combine skeleton.
- It also explains why compact NVFP4 does not win at large M despite lower compressed weight bandwidth: Hopper WGMMA still consumes FP8 from shared memory, so the kernel pays an extra bridge before every MMA tile.

Decision:
- Keep compact NVFP4 retained code unchanged for now.
- Treat FP8 shadow as the practical parity route for latency-sensitive deployments if extra FP8 shadow memory is acceptable.
- Next code direction should be either an official opt-in shadow benchmark/API path, or a new compact kernel design that avoids per-tile bridge overhead; small tweaks to the existing loader-dequant path are unlikely to close the 30-60% compact gap.


## 2026-06-11: Official opt-in FP8-shadow mode in NVFP4 tests/benchmarks

Change:
- Added `--weight-mode {compact,fp8-shadow}` to `tests/test_nvfp4_mega_moe_sm90_correctness.py`.
- Added `--weight-mode {compact,fp8-shadow}` to `tests/bench_nvfp4_mega_moe_sm90.py`.
- Default remains `compact`, which preserves the retained NVFP4 fused kernel path and split L1/L2 timing.
- `fp8-shadow` materializes NVFP4 weights to an FP8 shadow cache once via `materialize_nvfp4_fp8_shadow_for_mega_moe_sm90`, then calls PR323 `fp8_mega_moe` with Kineto filtering on `sm90_fp8_mega_moe_impl`.
- Correctness script rejects `--weight-mode fp8-shadow --reference-mode exact-nvfp4`; shadow mode must be validated against the `fp8-bridge` reference because it intentionally bakes NVFP4 weights into FP8.

Default compact validation after adding the flag:
```text
M32 weight_scale=0.05 fp8-bridge: PASS
cosine_min=0.9996 cosine_mean=0.9997 norm_ratio=0.9997
```

Official fp8-shadow expanded correctness:
```text
M32  weight_scale=0.001 cosine_min=0.9996 cosine_mean=0.9997 norm_ratio=0.9993 PASS
M256 weight_scale=0.001 cosine_min=0.9996 cosine_mean=0.9996 norm_ratio=0.9970 PASS
M32  weight_scale=0.05  cosine_min=0.9996 cosine_mean=0.9997 norm_ratio=0.9997 PASS
M256 weight_scale=0.05  cosine_min=0.9996 cosine_mean=0.9996 norm_ratio=0.9970 PASS
M32  weight_scale=1     cosine_min=0.9992 cosine_mean=0.9998 norm_ratio=1.0059 PASS
M256 weight_scale=1     cosine_min=0.9991 cosine_mean=0.9998 norm_ratio=0.9996 PASS
```

Official fp8-shadow 50-run benchmark command:
```bash
DG_JIT_CACHE_DIR=/tmp/dg_jit_shadow_official_bench_0611 \
python3 -u tests/bench_nvfp4_mega_moe_sm90.py \
  --weight-mode fp8-shadow \
  --num-processes 8 \
  --num-tests 50 \
  --batches 32 64 128 256 512 1024 2048 4096 8192
```

Official fp8-shadow 50-run result:
```text
M32    832.7 us
M64    856.6 us
M128   847.7 us
M256   1426.0 us
M512   2238.0 us
M1024  3528.0 us
M2048  6206.0 us
M4096  11380.0 us
M8192  21918.0 us
```

Gap vs same-shape PR323 W8A8 baseline:

| tokens | W8A8 PR323 | fp8-shadow | shadow/W8A8 | delta |
|---:|---:|---:|---:|---:|
| 32 | 923.9 us | 832.7 us | 0.901x | -9.9% |
| 64 | 869.9 us | 856.6 us | 0.985x | -1.5% |
| 128 | 837.1 us | 847.7 us | 1.013x | +1.3% |
| 256 | 1460.0 us | 1426.0 us | 0.977x | -2.3% |
| 512 | 2320.0 us | 2238.0 us | 0.965x | -3.5% |
| 1024 | 3518.0 us | 3528.0 us | 1.003x | +0.3% |
| 2048 | 6274.0 us | 6206.0 us | 0.989x | -1.1% |
| 4096 | 11402.0 us | 11380.0 us | 0.998x | -0.2% |
| 8192 | 21904.0 us | 21918.0 us | 1.001x | +0.1% |

Compact default benchmark sanity after the script change:
```text
M32 compact=1260.4 us, M64 compact=1221.4 us, num_tests=3
```

Decision:
- This is the first retained opt-in path that reaches W8A8 parity while preserving NVFP4 as the storage/loading format.
- It is not a compact online-NVFP4 kernel win; it spends extra memory on an FP8 shadow cache and intentionally removes FP4 decoding from the hot path.
- Recommend keeping `compact` as the correctness/space-efficient kernel path and exposing `fp8-shadow` as the latency path for H20/Hopper when memory allows.


## 2026-06-11: W8A8 50-run recheck and small-M volatility

Reason: the official fp8-shadow result above used 50 runs, while the earlier PR323 W8A8 comparison used 30 runs. Because small-M runs have visible long-tail effects, reran W8A8 with 50 runs.

PR323 W8A8 same-shape 50-run:
```text
M32    809.5 us
M64    842.5 us
M128   885.5 us
M256   1448.0 us
M512   2223.0 us
M1024  3551.0 us
M2048  6210.0 us
M4096  11366.0 us
M8192  21935.0 us
```

Official fp8-shadow 50-run versus W8A8 50-run:

| tokens | W8A8 PR323 50-run | fp8-shadow 50-run | shadow/W8A8 | delta |
|---:|---:|---:|---:|---:|
| 32 | 809.5 us | 832.7 us | 1.029x | +2.9% |
| 64 | 842.5 us | 856.6 us | 1.017x | +1.7% |
| 128 | 885.5 us | 847.7 us | 0.957x | -4.3% |
| 256 | 1448.0 us | 1426.0 us | 0.985x | -1.5% |
| 512 | 2223.0 us | 2238.0 us | 1.007x | +0.7% |
| 1024 | 3551.0 us | 3528.0 us | 0.994x | -0.6% |
| 2048 | 6210.0 us | 6206.0 us | 0.999x | -0.1% |
| 4096 | 11366.0 us | 11380.0 us | 1.001x | +0.1% |
| 8192 | 21935.0 us | 21918.0 us | 0.999x | -0.1% |

Small-M immediate rerun, 50 runs each:
```text
fp8-shadow: M32 891.0 us, M64 852.8 us, M128 861.2 us, M256 1439.0 us
W8A8 PR323: M32 838.2 us, M64 929.1 us, M128 989.4 us, M256 1451.0 us
```

Interpretation update:
- fp8-shadow is best described as W8A8-parity, not a stable M32 win.
- M32 ranking is especially volatile: depending on the 50-run sample, shadow can look close or slower by ~3-6%.
- M64/M128/M256 often match or beat W8A8 in the same time window, but this should be reported with the benchmark's average-time caveat.
- `bench_kineto` returns average kernel time from the profiler table, not median/p95, so long-tail effects directly move the printed number.


## 2026-06-11: 8-rank compact-vs-fp8-shadow consistency check

Purpose: add distributed correctness evidence for the fp8-shadow route. The single-rank correctness script checks against an `fp8-bridge` oracle, but does not cover 8-rank dispatch/combine. This temporary test runs both paths on the same 8-rank inputs and compares outputs per rank:
- compact NVFP4 fused kernel: `transform_nvfp4_weights_for_mega_moe_sm90` + `nvfp4_mega_moe`
- fp8-shadow path: `materialize_nvfp4_fp8_shadow_for_mega_moe_sm90` + `fp8_mega_moe`

Temporary script:
- `/tmp/test_nvfp4_shadow_distributed_consistency.py`

First run:
- M32 passed exactly.
- M256 had valid cosine/norm but one rank reported non-finite in the aggregate flag, so the run was not counted as PASS.

Debug reruns:
```text
M=256 distributed compact-vs-shadow:
cos_min_global=0.999797
cos_mean_min_rank=0.999873
norm_ratio_range=[0.997438,0.997562]
max_abs_global=1.800000e+01
mean_abs_max_rank=2.275815e+00
finite_all=True
```

Final consecutive M32+M256 rerun:
```text
M=32 distributed compact-vs-shadow:
cos_min_global=1.000000
cos_mean_min_rank=1.000000
norm_ratio_range=[1.000000,1.000000]
max_abs_global=0.000000e+00
mean_abs_max_rank=0.000000e+00
finite_all=True

M=256 distributed compact-vs-shadow:
cos_min_global=0.999797
cos_mean_min_rank=0.999873
norm_ratio_range=[0.997438,0.997562]
max_abs_global=1.800000e+01
mean_abs_max_rank=2.275815e+00
finite_all=True
```

Decision:
- Count the final rerun as PASS for 8-rank distributed consistency between compact and fp8-shadow at M32/M256, weight_scale=0.05.
- This is not an FP32 distributed oracle, but it validates that fp8-shadow follows the same distributed dispatch/combine behavior as the compact NVFP4 path.


## 2026-06-11: Opt-in FP8 unit-weight-scale specialization for NVFP4 shadow

Hypothesis: the fp8-shadow path materializes NVFP4 scale into FP8 weights and returns all-one FP8 block scales. The PR323 W8A8 kernel still loads B block scales and multiplies `scale_a * scale_b * accum`. Add an opt-in specialization to skip B scale loads/multiplies when weight scales are known to be 1.

Change:
- Added template bool `kUnitWeightScale` to `sm90_fp8_mega_moe_impl`, default false.
- Added host env `DG_SM90_FP8_UNIT_WEIGHT_SCALE=1` to instantiate the unit-scale variant.
- Default W8A8 path is unchanged because the env defaults to 0.
- The specialization is intended only for fp8-shadow weights produced by `materialize_nvfp4_fp8_shadow_for_mega_moe_sm90`.

Build:
```text
python setup.py build_ext --inplace -j 16: PASS
```

Correctness with `DG_SM90_FP8_UNIT_WEIGHT_SCALE=1`, `--weight-mode fp8-shadow`:
```text
M32  weight_scale=0.001 cosine_min=0.9996 cosine_mean=0.9997 norm_ratio=0.9993 PASS
M256 weight_scale=0.001 cosine_min=0.9996 cosine_mean=0.9996 norm_ratio=0.9970 PASS
M32  weight_scale=0.05  cosine_min=0.9996 cosine_mean=0.9997 norm_ratio=0.9997 PASS
M256 weight_scale=0.05  cosine_min=0.9996 cosine_mean=0.9996 norm_ratio=0.9970 PASS
M32  weight_scale=1     cosine_min=0.9992 cosine_mean=0.9998 norm_ratio=1.0059 PASS
M256 weight_scale=1     cosine_min=0.9991 cosine_mean=0.9998 norm_ratio=0.9996 PASS
```

8-rank distributed compact-vs-unit-shadow consistency, `DG_SM90_FP8_UNIT_WEIGHT_SCALE=1`:
```text
M=32:
cos_min_global=1.000000
cos_mean_min_rank=1.000000
norm_ratio_range=[1.000000,1.000000]
finite_all=True

M=256:
cos_min_global=0.999797
cos_mean_min_rank=0.999873
norm_ratio_range=[0.997438,0.997562]
finite_all=True
```

50-run benchmark, `DG_SM90_FP8_UNIT_WEIGHT_SCALE=1 --weight-mode fp8-shadow`:
```text
M32    789.7 us
M64    819.7 us
M128   850.3 us
M256   1456.0 us
M512   2212.0 us
M1024  3524.0 us
M2048  6235.0 us
M4096  11353.0 us
M8192  21887.0 us
```

Gap vs PR323 W8A8 50-run:

| tokens | W8A8 PR323 50-run | unit-shadow 50-run | unit-shadow/W8A8 | delta |
|---:|---:|---:|---:|---:|
| 32 | 809.5 us | 789.7 us | 0.976x | -2.4% |
| 64 | 842.5 us | 819.7 us | 0.973x | -2.7% |
| 128 | 885.5 us | 850.3 us | 0.960x | -4.0% |
| 256 | 1448.0 us | 1456.0 us | 1.006x | +0.6% |
| 512 | 2223.0 us | 2212.0 us | 0.995x | -0.5% |
| 1024 | 3551.0 us | 3524.0 us | 0.992x | -0.8% |
| 2048 | 6210.0 us | 6235.0 us | 1.004x | +0.4% |
| 4096 | 11366.0 us | 11353.0 us | 0.999x | -0.1% |
| 8192 | 21935.0 us | 21887.0 us | 0.998x | -0.2% |

Small-M rerun, 50 runs:
```text
M32  772.9 us
M64  811.7 us
M128 858.7 us
M256 1447.0 us
```

Decision:
- Retain as opt-in for fp8-shadow. This is the first path that is consistently at or slightly faster than W8A8 on M32/M64/M128 while keeping correctness.
- Do not enable globally for W8A8 because normal W8A8 block scales are not guaranteed to be 1.
- For deployment, use this only when the weights were produced by the NVFP4 fp8-shadow materializer.


## 2026-06-11: Make fp8-shadow auto-enable unit weight scale in tests/benchmarks

Follow-up usability change after the unit-weight-scale optimization:
- `tests/test_nvfp4_mega_moe_sm90_correctness.py --weight-mode fp8-shadow` now sets `DG_SM90_FP8_UNIT_WEIGHT_SCALE=1` automatically.
- `tests/bench_nvfp4_mega_moe_sm90.py --weight-mode fp8-shadow` now sets `DG_SM90_FP8_UNIT_WEIGHT_SCALE=1` automatically.
- Added `--no-unit-weight-scale` to both scripts for debug / A-B comparison.
- `compact` mode is unchanged.

Correctness without manually setting the env:
```text
M32  weight_scale=0.05 unit_weight_scale=True cosine_min=0.9996 cosine_mean=0.9997 norm_ratio=0.9997 PASS
M256 weight_scale=0.05 unit_weight_scale=True cosine_min=0.9996 cosine_mean=0.9996 norm_ratio=0.9970 PASS
```

Small-M 50-run benchmark without manually setting the env:
```text
M32  804.0 us
M64  800.5 us
M128 829.6 us
M256 1437.0 us
```

Decision:
- Treat `--weight-mode fp8-shadow` as the recommended benchmark/correctness command for the latency path; it now selects the faster unit-scale kernel automatically.
- Keep `--no-unit-weight-scale` only for comparison against the previous non-specialized shadow path.


## 2026-06-11: Default-path safety checks after unit-scale specialization

Purpose: ensure the FP8 unit-weight-scale template bool is truly opt-in and does not contaminate default PR323 W8A8 or compact NVFP4 paths.

W8A8 PR323 default small-M 50-run after the code change, without `DG_SM90_FP8_UNIT_WEIGHT_SCALE`:
```text
M32  828.5 us
M64  833.5 us
M128 888.8 us
M256 1436.0 us
```

This is within the previously observed W8A8 small-M variation range.

Compact NVFP4 sanity, `--weight-mode compact`:
```text
unit_weight_scale=False
M32 compact=1189.7 us
M64 compact=1190.8 us
```

Decision:
- Unit-weight-scale remains isolated to `fp8-shadow` unless explicitly enabled by env or by the benchmark/correctness scripts in shadow mode.
- Default W8A8 and compact NVFP4 paths remain usable.

## 2026-06-11: Compact phase profile and all-loader-thread dequant experiment

Question: is compact NVFP4 primarily blocked on the two idle loader warps doing FP4->FP8 dequant too slowly?

Short phase-profile run, compact path, `DG_SM90_MOE_PHASE_PROFILE=1`, 5 runs:

```text
M32 compact=1619.5 us
  math_loop avg=647570 cycles, gemm_core avg=33974 cycles
  loader_dequant avg=1018 cycles, math_dequant_wait avg=448 cycles
M256 compact=2106.8 us
  math_loop avg=882846 cycles, gemm_core avg=33211 cycles
  loader_dequant avg=1012 cycles, math_dequant_wait avg=430 cycles
M1024 compact=6351.0 us
  math_loop avg=2875325 cycles, gemm_core avg=32782 cycles
  loader_dequant avg=1004 cycles, math_dequant_wait avg=423 cycles
```

Interpretation:
- Per-stage dequant is about 1k cycles and math waits about 0.4k cycles for the dequant barrier.
- The per-stage cost is not huge by itself, but it is paid for every CTA/K tile. The count scales with M and with L1/L2 tile count.
- This supports the current conclusion that true compact NVFP4 is losing to W8A8 mostly because Hopper must repeatedly materialize FP4 weights into FP8 shared memory before WGMMA.

Experiment: added an opt-in `DG_SM90_NVFP4_LOADER_DEQUANT_ALL_THREADS=1` branch locally. It made all 4 non-epilogue warps participate in loader-side dequant, changing the dequant work from 2 idle warps x 2 rows/thread to 4 warps x 1 row/thread.

Correctness, compact, M32/M256, weight_scale 0.001/0.05/1.0:

```text
PASS all cases
worst cosine_min=0.9991
norm_ratio in [0.9993, 1.0059]
```

30-run benchmark with all-loader-thread dequant:

| tokens | previous compact 30-run | all-loader-thread | result |
|---:|---:|---:|---:|
| 32 | 1206.5 us | 1504.2 us | slower |
| 64 | 1233.9 us | 1513.7 us | slower |
| 128 | 1200.4 us | 1472.5 us | slower |
| 256 | 1773.3 us | 2118.9 us | slower |
| 512 | 3132.0 us | 3506.0 us | slower |
| 1024 | 5159.0 us | 5986.0 us | slower |
| 2048 | 9618.0 us | 11053.0 us | slower |
| 4096 | 18311.0 us | 20947.0 us | slower |
| 8192 | 36019.0 us | 40920.0 us | slower |

Decision:
- Revert the all-loader-thread code path.
- Do not gate this by M. The loss is broad, which means preserving TMA producer overlap is more valuable than shortening per-stage dequant with more threads.
- Next useful direction should target reducing repeated dequant/materialization count or changing the small-M compute path, not simply adding more dequant threads.

## 2026-06-11: Fused launch, L2 no-dispatch, dual-accum, stages, and N-major checks

### Fused single-launch path

The compact fused `DG_SM90_MOE_SPLIT_L1_L2=0` path initially failed JIT compilation:

```text
cannot determine which instance of function template "deep_gemm::sm90_nvfp4_mega_moe_impl" is intended
```

Fix: add an unused template tail parameter `kInstantiationTag` and pass `(run_l1_phase ? 1 : 0) | (run_l2_phase ? 2 : 0)` from the host launcher. This disambiguates the `true,true` fused instantiation without changing runtime behavior.

Correctness after the fix:

```text
compact fused M32/M256, weight_scale=0.001/0.05/1.0: PASS
worst cosine_min=0.9991
norm_ratio in [0.9993, 1.0059]
```

30-run fused single-launch benchmark:

| tokens | retained split compact | fused single-launch | result |
|---:|---:|---:|---:|
| 32 | 1206.5 us | 1272.0 us | slower |
| 64 | 1233.9 us | 1250.0 us | slower |
| 128 | 1200.4 us | 1286.0 us | slower |
| 256 | 1773.3 us | 1868.0 us | slower |
| 512 | 3132.0 us | 3141.0 us | flat/slower |
| 1024 | 5159.0 us | 5432.0 us | slower |
| 2048 | 9618.0 us | 10160.0 us | slower |
| 4096 | 18311.0 us | 19334.0 us | slower |
| 8192 | 36019.0 us | 38040.0 us | slower |

Decision:
- Keep the JIT disambiguation fix because it restores correctness coverage for the fused path.
- Do not use fused single-launch as default; split L1/L2 remains faster.

### Split L2 without dispatch pipeline

Correctness with `DG_SM90_MOE_L2_NO_DISPATCH_PIPELINE=1`: PASS for M32/M256 and weight_scale 0.001/0.05/1.0.

30-run result:

| tokens | retained split compact | L2 no-dispatch | result |
|---:|---:|---:|---:|
| 32 | 1206.5 us | 1185.3 us | small faster in this run |
| 64 | 1233.9 us | 1251.5 us | slower |
| 128 | 1200.4 us | 1239.2 us | slower |
| 256 | 1773.3 us | 1750.8 us | small faster in this run |
| 512 | 3132.0 us | 3119.0 us | flat |
| 1024 | 5159.0 us | 5167.0 us | flat/slower |
| 2048 | 9618.0 us | 9622.0 us | flat |
| 4096 | 18311.0 us | 18304.0 us | flat |
| 8192 | 36019.0 us | 36037.0 us | flat |

50-run small-M paired check:

| tokens | default split 50-run | L2 no-dispatch 50-run | result |
|---:|---:|---:|---:|
| 32 | 1189.9 us | 1203.7 us | slower |
| 64 | 1215.6 us | 1222.5 us | slower |
| 128 | 1243.4 us | 1237.1 us | noise-level faster |
| 256 | 1748.7 us | 1829.6 us | slower |

Decision: do not enable by default.

### L1 dual-K off

Correctness with `DG_SM90_MOE_L1_DUAL_K=0`: PASS for M32/M256 and weight_scale 0.001/0.05/1.0.

30-run result:

```text
M32  1226.8 us
M64  1235.7 us
M128 1220.5 us
M256 1896.1 us
M512 3077.0 us
M1024 5266.0 us
M2048 9847.0 us
M4096 18750.0 us
M8192 36937.0 us
```

Decision: keep L1 dual-K enabled. Disabling it does not help small M and hurts M256+.

### L2 dual-accum off

Correctness with `DG_SM90_MOE_L2_DUAL_ACCUM=0`: PASS for M32/M256 and weight_scale 0.001/0.05/1.0.

30-run result:

```text
M32  1255.7 us
M64  1248.3 us
M128 1206.9 us
M256 1834.8 us
M512 3148.0 us
M1024 5209.0 us
M2048 9689.0 us
M4096 18422.0 us
M8192 36112.0 us
```

Decision: keep L2 dual-accum enabled.

### Pipeline stages

`DG_SM90_NVFP4_NUM_STAGES=5` correctness: PASS for M32/M256 and weight_scale 0.001/0.05/1.0.

30-run stage=5:

```text
M32  1218.0 us
M64  1218.0 us
M128 1218.4 us
M256 1767.5 us
M512 3034.0 us
M1024 5451.0 us
M2048 9741.0 us
M4096 18570.0 us
M8192 36426.0 us
```

`DG_SM90_NVFP4_NUM_STAGES=4` correctness: PASS for M32/M256 and weight_scale 0.001/0.05/1.0.

30-run stage=4:

```text
M32  1273.1 us
M64  1249.7 us
M128 1270.7 us
M256 1875.6 us
M512 3087.0 us
M1024 5440.0 us
M2048 9915.0 us
M4096 18993.0 us
M8192 37134.0 us
```

`DG_SM90_NVFP4_NUM_STAGES=6` M32 correctness: PASS for weight_scale 0.001/0.05/1.0.

50-run stage=6 small-M:

```text
M32  1198.4 us
M64  1187.5 us
M128 1264.4 us
M256 1756.7 us
```

Decision:
- Do not change default stage heuristic. stage=5/6 has isolated wins but not stable across M; stage=4 is worse.

### N-major scheduler flags

Correctness with `DG_SM90_MOE_L1_NMAJOR=1 DG_SM90_MOE_L2_NMAJOR=1`: PASS.
Correctness with `DG_SM90_MOE_L2_NMAJOR=1 DG_SM90_MOE_L1_NMAJOR=0`: PASS.

However, code inspection shows `kL2NMajorScheduleRequested` and `kL1NMajorScheduleRequested` are template parameters in `sm90_nvfp4_mega_moe.cuh` but are not actually used by the NVFP4 scheduler path. The observed benchmark differences are therefore not a valid functional optimization signal.

30-run L2-only N-major numbers were:

```text
M32  1187.3 us
M64  1270.6 us
M128 1191.1 us
M256 1737.3 us
M512 2990.0 us
M1024 5141.0 us
M2048 9674.0 us
M4096 18342.0 us
M8192 36077.0 us
```

50-run paired check, default vs L2-only N-major:

| tokens | default split 50-run | L2-only N-major 50-run |
|---:|---:|---:|
| 32 | 1242.2 us | 1196.8 us |
| 64 | 1190.4 us | 1200.4 us |
| 128 | 1251.1 us | 1220.7 us |
| 256 | 1856.7 us | 1745.9 us |
| 512 | 3013.0 us | 2990.0 us |
| 1024 | 5181.0 us | 5187.0 us |

Decision:
- Do not default-enable N-major based on these numbers because the flags are currently dead in NVFP4 code.
- If N-major scheduling is desired later, it needs an actual scheduler implementation first.

## 2026-06-11: Final default compact validation after fused-JIT fix

Retained code change from this round:
- Add `kInstantiationTag` as a tail template parameter to `sm90_nvfp4_mega_moe_impl`.
- Host passes phase tag 1/2/3 for L1-only/L2-only/fused instantiations.
- Purpose: fix NVCC template address ambiguity for the compact fused `run_l1_phase=true, run_l2_phase=true` path.
- No performance experiment from this round is enabled by default.

Final default compact correctness, split L1/L2, M32/M256, weight_scale 0.001/0.05/1.0:

```text
NVFP4 dequant unit test: PASS
NVFP4 CUDA dequant LUT unit test: PASS
M32/M256 all weight scales: PASS
worst cosine_min=0.9991
norm_ratio in [0.9993, 1.0059]
```

Final default compact 30-run benchmark after the JIT fix:

| tokens | compact NVFP4 default |
|---:|---:|
| 32 | 1237.7 us |
| 64 | 1218.7 us |
| 128 | 1203.3 us |
| 256 | 1802.2 us |
| 512 | 3112.0 us |
| 1024 | 5186.0 us |
| 2048 | 9655.0 us |
| 4096 | 18388.0 us |
| 8192 | 36004.0 us |

Comparison to retained compact 30-run before this round:

| tokens | retained compact | final default | delta |
|---:|---:|---:|---:|
| 32 | 1206.5 us | 1237.7 us | +2.6% |
| 64 | 1233.9 us | 1218.7 us | -1.2% |
| 128 | 1200.4 us | 1203.3 us | +0.2% |
| 256 | 1773.3 us | 1802.2 us | +1.6% |
| 512 | 3132.0 us | 3112.0 us | -0.6% |
| 1024 | 5159.0 us | 5186.0 us | +0.5% |
| 2048 | 9618.0 us | 9655.0 us | +0.4% |
| 4096 | 18311.0 us | 18388.0 us | +0.4% |
| 8192 | 36019.0 us | 36004.0 us | 0.0% |

Comparison to PR323 W8A8 30-run baseline:

| tokens | W8A8 PR323 | compact NVFP4 final | compact/W8A8 |
|---:|---:|---:|---:|
| 32 | 923.9 us | 1237.7 us | 1.340x |
| 64 | 869.9 us | 1218.7 us | 1.401x |
| 128 | 837.1 us | 1203.3 us | 1.437x |
| 256 | 1460.0 us | 1802.2 us | 1.234x |
| 512 | 2320.0 us | 3112.0 us | 1.341x |
| 1024 | 3518.0 us | 5186.0 us | 1.474x |
| 2048 | 6274.0 us | 9655.0 us | 1.539x |
| 4096 | 11402.0 us | 18388.0 us | 1.613x |
| 8192 | 21904.0 us | 36004.0 us | 1.644x |

Decision:
- Keep the JIT disambiguation fix.
- Do not claim a compact NVFP4 speedup from this round.
- The compact path remains correct but still significantly slower than PR323 W8A8 because FP4->FP8 shared-memory materialization is repeated per CTA/K tile.

## 2026-06-11: Port reference NVFP4 heuristics from DeepGEMM_megamoe_nvfp4

Reference repo:

```text
/root/fac/megamoe/DeepGEMM_megamoe_nvfp4
HEAD 2d087b2 [nvfp4] Tune small-M heuristics
```

Imported ideas:
- Enable BM128 + 256 epilogue threads by default for M256/M512/M1024/M2048/M4096/M8192 when the compact layout is BN128.
- Allow loader-side dequant with 256 epilogue threads by removing the old BN128+128-thread static assumption.
- Adjust register budget for BM128 loader-dequant: non-epilogue 64 regs, epilogue 192 regs.
- Enable direct L2 scatter for BN128 even with 256 epilogue threads.
- Use L2 no-dispatch pipeline by default for M256/M512/M1024/M2048.
- Keep M64/M128 split by default in this repo. The reference repo used fused for M64/M128, but it was slower here.
- Update the benchmark split-kernel accounting so it matches the host default when `DG_SM90_MOE_SPLIT_L1_L2` is unset.

Rejected from the reference defaults for this repo:
- M64/M128 fused default. 30-run default fused was ~1304/1314 us; forced split 50-run was ~1212/1217 us.
- BM32/mma.sync. Probe triggered illegal address and the reference repo still explicitly keeps BM32 disabled.

Correctness after port, default compact path:

```text
M32/M256/M512, weight_scale=0.001/0.05/1.0: PASS
worst cosine_min=0.9990
norm_ratio in [0.9992, 1.0059]
```

Final default compact 50-run benchmark:

| tokens | compact NVFP4 |
|---:|---:|
| 32 | 1223.5 us |
| 64 | 1203.8 us |
| 128 | 1203.7 us |
| 256 | 1551.6 us |
| 512 | 2403.1 us |
| 1024 | 3829.0 us |
| 2048 | 6704.0 us |
| 4096 | 13505.0 us |
| 8192 | 26130.0 us |

Comparison to previous default compact 30-run after the fused-JIT fix:

| tokens | previous default | final default | delta |
|---:|---:|---:|---:|
| 32 | 1237.7 us | 1223.5 us | 1.1% faster |
| 64 | 1218.7 us | 1203.8 us | 1.2% faster |
| 128 | 1203.3 us | 1203.7 us | 0.0% slower |
| 256 | 1802.2 us | 1551.6 us | 13.9% faster |
| 512 | 3112.0 us | 2403.1 us | 22.8% faster |
| 1024 | 5186.0 us | 3829.0 us | 26.2% faster |
| 2048 | 9655.0 us | 6704.0 us | 30.6% faster |
| 4096 | 18388.0 us | 13505.0 us | 26.6% faster |
| 8192 | 36004.0 us | 26130.0 us | 27.4% faster |

Comparison to PR323 W8A8 30-run baseline:

| tokens | W8A8 PR323 | compact NVFP4 final | compact/W8A8 |
|---:|---:|---:|---:|
| 32 | 923.9 us | 1223.5 us | 1.324x |
| 64 | 869.9 us | 1203.8 us | 1.384x |
| 128 | 837.1 us | 1203.7 us | 1.438x |
| 256 | 1460.0 us | 1551.6 us | 1.063x |
| 512 | 2320.0 us | 2403.1 us | 1.036x |
| 1024 | 3518.0 us | 3829.0 us | 1.088x |
| 2048 | 6274.0 us | 6704.0 us | 1.069x |
| 4096 | 11402.0 us | 13505.0 us | 1.184x |
| 8192 | 21904.0 us | 26130.0 us | 1.193x |

Decision:
- Keep the reference-derived BM128/default heuristic path.
- It does not make compact NVFP4 faster than W8A8, but it materially closes the M256-M2048 gap.
- Small M remains dominated by fixed dispatch/combine/dequant materialization overhead and still trails W8A8 by ~30-49%.

## 2026-06-11: Extend L2 no-dispatch default to M4096/M8192

Reference clue:
- `/root/fac/megamoe/DeepGEMM_megamoe_nvfp4` had an uncommitted experiment enabling `DG_SM90_MOE_L2_NO_DISPATCH_PIPELINE` by default for M4096.
- The current fuse repo already uses the default combine chunk strategy, so only the L2 no-dispatch default needed validation.

Change:
- Extend `l2_no_dispatch_pipeline_default` from M256/M512/M1024/M2048 to M256/M512/M1024/M2048/M4096/M8192.

Correctness:
- M32/M256/M512 with weight_scale=0.001/0.05/1.0: PASS, worst cosine_min=0.9990.
- M4096 with weight_scale=0.001/0.05/1.0: PASS, worst cosine_min=0.9987.
- M8192 with weight_scale=0.05: PASS, cosine_min=0.9995.

Benchmark evidence:

| tokens | previous default | new retained | delta |
|---:|---:|---:|---:|
| 4096 | 13505.0 us | 12212.0 us | 9.6% faster |
| 8192 | 26130.0 us | 23657.0 us | 9.5% faster |

Retained compact NVFP4 50-run after this change:

| tokens | compact NVFP4 | compact/W8A8 PR323 |
|---:|---:|---:|
| 32 | 1229.3 us | 1.331x |
| 64 | 1199.7 us | 1.379x |
| 128 | 1199.6 us | 1.433x |
| 256 | 1561.8 us | 1.070x |
| 512 | 2401.0 us | 1.035x |
| 1024 | 3803.0 us | 1.081x |
| 2048 | 6719.0 us | 1.071x |
| 4096 | 12212.0 us | 1.071x |
| 8192 | 23657.0 us | 1.080x |

Decision:
- Keep this as default. It does not touch the numerical dequant path and materially improves the large-M cases.
- The remaining gap is now concentrated in M32/M64/M128 fixed overhead and a residual 3.5-8.1% gap for M256-M8192.

## 2026-06-11: Small-M follow-up probes after L2 no-dispatch extension

Tried and rejected:
- Force L2 no-dispatch for M32/M64/M128. M32 was noisy and same-window default won: 1200.2 us default vs 1208.9 us forced. M64/M128 also did not show a stable win.
- Force L2 dual accum for M64. First run looked better, but repeat was effectively tied: 1194.4 us disabled vs 1196.0 us enabled.
- Force L1 dual-K for M32. Result was within noise: 1221.0 us disabled vs 1217.4 us enabled.
- BN256 compact scale layout. Latency-only M64/M128 looked attractive (873.5/919.5 us), but correctness failed immediately at M64 weight_scale=0.001: cosine_mean=0.0127 and norm_ratio=136.08. The same sweep later hit illegal memory access at M256. Host-side BN256 opt-in is therefore disabled until the split-N NVFP4 dequant path is fixed.

Decision:
- Keep only the M4096/M8192 L2 no-dispatch default from this follow-up.
- Do not use BN256 numbers as valid performance; they are from an incorrect path.

## 2026-06-11: BN256 split-N dequant repair attempt, rejected

Hypothesis:
- BN256 latency-only probes showed M64/M128 could approach W8A8, but correctness failed. The first bug was that loader-dequant only expanded 128 B rows, leaving the second split-N half undequantized.

Attempt:
- Prototype BN256 loader-dequant with all 128 non-epilogue threads, each expanding two B rows, covering 256 rows.
- Fix split-N L1 output TMA descriptor width from 64 to 128 so the single split-N TMA store writes both output halves.

Results:
- BN256 M64/M128 correctness recovered after both fixes: weight_scale=0.001/0.05/1.0 PASS.
- Correct BN256 latency lost the apparent win: M32=1285.6 us, M64=1380.4 us, M128=1289.1 us, all slower than BN128 retained defaults.
- BN256 M256 still hit illegal memory access during correctness.

Decision:
- Reverted BN256 code changes and kept host BN256 disabled.
- Do not use earlier fast BN256 numbers; they came from an incorrect path.
- Small-M gap remains unsolved by BN256 split-N in the current implementation.

## 2026-06-11: Small-M phase profile and launch-shape probes

Phase profile, M32/M64, with DG_SM90_MOE_PHASE_PROFILE=1:
- M32 profile run latency: 1422.9 us. Large components were math_loop avg 670718 cycles, dispatch_total avg 56275 cycles, gemm_core avg 34018 cycles. loader_dequant avg was only 1041 cycles and math_dequant_wait avg 462 cycles.
- M64 profile run latency: 1496.5 us. math_loop avg 640380 cycles, dispatch_total avg 209422 cycles, gemm_core avg 33380 cycles. loader_dequant avg was 1007 cycles and math_dequant_wait avg 423 cycles.
- Interpretation: small-M gap is not dominated by raw NVFP4 dequant instructions; fixed dispatch/scheduler/synchronization and the two-phase math loop dominate.

Tried and rejected:
- Fused single-launch small-M (DG_SM90_MOE_SPLIT_L1_L2=0): M32=1288.0 us, M64=1243.0 us, M128=1261.0 us. Still slower than retained split defaults.
- Dispatch64 + non-epilogue64 fallback (DG_SM90_NVFP4_DISPATCH_THREADS=64, DG_SM90_NVFP4_NON_EPILOGUE_THREADS=64): M32=1396.9 us, M64=1465.8 us, M128=1517.1 us. This disables the loader-dequant shape and is worse.

Decision:
- Keep split L1/L2 and 128 dispatch + 128 non-epilogue threads as defaults.
- Next viable small-M work likely needs structural reduction of per-phase scheduling/dispatch/combine overhead, not a dequant micro-optimization.

## 2026-06-11: Small-M SM-count sweep, rejected

Hypothesis:
- Small M might suffer from too many SMs/CTAs and tail synchronization, so limiting active SMs could reduce fixed overhead.

Results, 30-run compact NVFP4:

| DG_SM90_MOE_SET_NUM_SMS | M32 | M64 | M128 | decision |
|---:|---:|---:|---:|---|
| default | ~1200-1230 us | ~1200 us | ~1200 us | retained |
| 72 | 1288.5 us | 1304.4 us | 1409.4 us | slower |
| 64 | 1409.8 us | 1404.4 us | 1379.4 us | slower |
| 48 | 1902.2 us | 1874.9 us | 1856.2 us | slower |
| 32 | 2761.2 us | 2770.1 us | 2784.0 us | much slower |

Notes:
- 80 and 96 exceeded the H20 visible SM count and failed the runtime assertion.

Decision:
- Do not limit SM count for small M. Full-SM scheduling remains best among tested values.

## 2026-06-11: Dequant fallback and stage-count sweeps, rejected

Dequant fallback probes, 50-run small M:

| config | M32 | M64 | M128 | decision |
|---|---:|---:|---:|---|
| loader_dequant=0 | 1398.8 us | 1488.0 us | 1492.7 us | slower |
| loader_dequant=0 + direct_scale_gmem=1 | 1695.6 us | 1754.3 us | 1812.8 us | much slower |
| loader_dequant=0 + packed_b_scratch=1 | 1390.6 us | 1459.3 us | 1472.4 us | slower |

Stage-count sweep, 30-run small M:

| DG_SM90_NVFP4_NUM_STAGES | M32 | M64 | M128 | decision |
|---:|---:|---:|---:|---|
| 2 | compile fail | compile fail | compile fail | hidden combine buffer too large |
| 3 | 1227.9 us | 1561.3 us | 1622.1 us | slower |
| 4 | 1233.9 us | 1257.1 us | 1283.1 us | slower |
| 5 | 1229.0 us | 1307.6 us | 1215.7 us | mixed/no stable win |
| 6 | 1214.6 us | 1243.9 us | 1262.7 us | retained/default-like |
| 7 | 1238.1 us | 1229.2 us | 1225.6 us | no stable win |

Decision:
- Keep loader-dequant enabled.
- Keep default stage heuristic; do not override stages for small M.

## 2026-06-11: Constant-LUT dequant prototype, rejected

Hypothesis:
- Avoid shared-memory LUT reads by loading NVFP4 E2M1+UE4M3 decode entries from the existing constant LUT (`load_e2m1_ue4m3_lut`).

Attempt:
- Temporarily replaced all `lut_smem[scale]` loads in the NVFP4 dequant helpers with `deep_gemm::nvfp4::load_e2m1_ue4m3_lut(scale)`.
- Kept the rest of the ABI/smem layout unchanged for a quick signal test.

Result, 50-run small M:
- M32=1461.8 us, M64=1437.7 us, M128=1446.7 us.

Decision:
- Reverted. Constant LUT is slower than the staged shared-memory LUT in this access pattern.

## 2026-06-11: Mid/large split and dual-accum probes, rejected

Tried and rejected:
- Fused single-launch for M256-M8192 (`DG_SM90_MOE_SPLIT_L1_L2=0`) was much slower: M256=3486 us, M512=5714 us, M1024=8858 us, M2048=15672 us, M4096=29309 us, M8192=56459 us.
- L2 dual accum off (`DG_SM90_MOE_L2_DUAL_ACCUM=0`) showed an apparent isolated win in one sweep, but same-window A/B favored the default: M256 1517.7 us with dual=1 vs 1521.8 us with dual=0; M2048 6556 us with dual=1 vs 6572 us with dual=0.
- L1 dual-K off was mixed/noisy: M256 1534.2 us with dual-K=1 vs 1554.4 us with dual-K=0; M2048 6517 us with dual-K=1 vs 6483 us with dual-K=0.

Decision:
- Keep split L1/L2, L2 dual accum default on except the existing M64 exception, and L1 dual-K default on for M>32.

## 2026-06-11: Loader-dequant packed-B scratch prototype, rejected

Hypothesis:
- Avoid the in-place FP4-to-FP8 overwrite barrier by TMA-loading packed B into a separate scratch buffer, then having loader-dequant read scratch and write FP8 smem_b.

Attempt:
- Temporarily allowed `DG_SM90_NVFP4_PACKED_B_SCRATCH=1` together with loader-dequant.
- Loader-dequant read rows from `smem_packed_b` using `dequant_smem_b_from_packed` and wrote expanded FP8 rows into `smem_b`.

Result:
- Correctness hung on the first M32 case, indicating the prototype violated the existing full/empty/dequant barrier lifecycle.
- The exact test process was terminated by PID; no broad `pkill` was used.

Decision:
- Reverted. This direction would need a more careful pipeline/barrier redesign before it can be tested safely.
