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


## 2026-06-11: Reference NVFP4 repo comparison and split-timing correction

Reference repo checked:
- /root/fac/megamoe/DeepGEMM_megamoe_nvfp4 at 2d087b2 [nvfp4] Tune small-M heuristics with local uncommitted NVFP4 changes.
- Reusable deltas found: M4096 disables L1 dual-K and L2 dual-accum by default; L2 no-dispatch for M4096; combine chunking uses the default chunk heuristic.
- The fuse repo already had L2 no-dispatch through M8192 and default combine chunking. The only kernel default copied in this iteration was the M4096 dual-off policy.

Measurement correction:
- bench_kineto cannot distinguish sm90_nvfp4_mega_moe_l1 and sm90_nvfp4_mega_moe_l2 or l2_nodisp by JIT build suffix because the profiler table exposes the generated CUDA function name.
- An attempted explicit tuple match returned 0.0 us for all sizes and was reverted.
- The retained benchmark code now keeps substring matching with with_multiple_kernels=True and documents that the returned per-kernel average is multiplied by two to estimate one split L1+L2 MoE call.
- This means older M512+ compact numbers around 2.4/3.8/6.7/12.2/23.7 ms were half-time records, not end-to-end split latency.

M4096 dual-off A/B:
- M4096-only, same input shape, 30-run: default 24423 us; DG_SM90_MOE_L1_DUAL_K=0 DG_SM90_MOE_L2_DUAL_ACCUM=0 24287 us.
- M8192 20-run: default 50350 us; dual-off 50256 us.
- Decision: keep only the M4096 default change from the reference repo. The gain is small, about 0.6%, but consistent and shape-specific.

Correctness after the change:
- Command: tests/test_nvfp4_mega_moe_sm90_correctness.py --batches 32 256 4096 --weight-scales 0.001 0.05 1.0 --reference-mode fp8-bridge --weight-mode compact --cosine-mean-threshold 0.99 --cosine-min-threshold 0.99 --norm-ratio-min 0.99 --norm-ratio-max 1.01
- Result: PASS. Worst observed cosine_min=0.9987, finite=True for all cases.

Latest compact 30-run benchmark with split L1+L2 total timing:

| tokens | compact NVFP4 | W8A8 PR323 | compact/W8A8 |
|---:|---:|---:|---:|
| 32 | 1191.2 us | 923.9 us | 1.289x |
| 64 | 1179.1 us | 869.9 us | 1.355x |
| 128 | 1183.2 us | 837.1 us | 1.414x |
| 256 | 1504.0 us | 1460.0 us | 1.030x |
| 512 | 4727.0 us | 2320.0 us | 2.037x |
| 1024 | 5950.0 us | 3518.0 us | 1.691x |
| 2048 | 13772.0 us | 6274.0 us | 2.195x |
| 4096 | 24330.0 us | 11402.0 us | 2.134x |
| 8192 | 50534.0 us | 21904.0 us | 2.307x |

Interpretation:
- M32-M128 are still fixed-overhead dominated and 29-41% slower than PR323 W8A8.
- M256 remains the only near-parity compact case in the current true split timing, about 3% slower.
- M512+ are not close to W8A8 once split L1+L2 is summed; earlier near-parity records were undercounted by roughly a factor of two.
- Next useful work should focus on avoiding two full split launches or making the compact NVFP4 L2 phase reuse a materialized representation more like the FP8-shadow upper bound.


## 2026-06-11: Rank-aware benchmark and rejected mid/large knobs

Why this was needed:
- The compact benchmark historically printed only rank0 latency.
- Kineto table checks showed split NVFP4 has two real sm90_nvfp4_mega_moe_impl launches per MoE call; M256 rank0 was about 972 us + 526 us = 1498 us, matching the benchmark output.
- The same table showed severe rank skew. Example M512: rank0 was about 3.28 ms + 1.30 ms, while other ranks were around 3.5 ms + 3.2 ms.
- W8A8 has similar rank tail in a one-iteration table check, so comparisons must keep rank0 vs max_rank conventions separate.

Benchmark script change:
- tests/bench_nvfp4_mega_moe_sm90.py now all-reduces per-rank measured latency and prints mean_rank and max_rank alongside the existing rank0 field.
- deep_gemm/testing/bench.py has an opt-in DG_SHOW_KINETO_TABLE=1 profiler-table print for future timing audits.

Latest compact 30-run full sweep:

| tokens | rank0 | mean_rank | max_rank |
|---:|---:|---:|---:|
| 32 | 1190.8 us | 3151.6 us | 3441.9 us |
| 64 | 1179.0 us | 3143.4 us | 3432.2 us |
| 128 | 1182.6 us | 3137.1 us | 3435.2 us |
| 256 | 1504.3 us | 3465.1 us | 3760.4 us |
| 512 | 4734.0 us | 6695.8 us | 6990.0 us |
| 1024 | 5956.0 us | 7919.9 us | 8216.0 us |
| 2048 | 13787.0 us | 15751.3 us | 16049.0 us |
| 4096 | 24356.0 us | 26316.1 us | 26612.0 us |
| 8192 | 50588.0 us | 52540.6 us | 52848.0 us |

Rejected knobs under rank-aware output:
- BM64 fallback (DG_SM90_MOE_BLOCK_M=64): M512 rank0/max 5271/7510 us, M1024 10088/12337 us, M2048 19433/21687 us. Worse than BM128 default.
- DG_SM90_MOE_L2_NMAJOR=0: M512 similar but M1024 worse; no default change.
- DG_SM90_MOE_L1_NMAJOR=1: no win; M512 worse and large sizes tied/slower.
- DG_SM90_MOE_K2_DIRECT_ACCUM=1: no meaningful win across M256-M4096.
- DG_SM90_MOE_L2_NO_DISPATCH_PIPELINE=0: consistently worse; keep no-dispatch default for M256+.
- DG_SM90_NVFP4_NUM_STAGES=5: similar/no win for M512/M1024. DG_SM90_NVFP4_NUM_STAGES=7 is invalid for this shape and trips host max-stage assert.

Interpretation:
- The current compact NVFP4 kernel is close to the current in-repo W8A8 rank0 timing for M512/M1024, but the historical PR323 W8A8 table in README is not directly comparable without revalidating its branch and rank convention.
- The largest remaining engineering target is rank-tail and the two heavy split launches, not raw FP4 dequant instruction latency.


## 2026-06-11: W8A8 baseline revalidation notes

Baseline ambiguity found:
- Current fuse repo W8A8 via tests/test_mega_moe_hopper.py gives M256=1398 us, M512=4566 us, M1024=5833 us on rank0 for a 30-run sweep.
- The existing README table lists PR323 W8A8 M512=2320 us and M1024=3518 us, which does not match the current in-repo W8A8 path.
- /root/fac/megamoe/DeepGEMM on branch megamoe_sm90 uses a different split benchmark script and also has rank-tail / prefix-sum timing ambiguity.

PR323 worktree attempt:
- Created /root/fac/megamoe/deepgemm_805_w8a8 at commit 805f7e8 Restore SM90 block64 default heuristic.
- Build failed before benchmark because third-party CUTLASS in that worktree lacks cute/arch/mma_sm100_desc.hpp.
- Also tried /root/fac/megamoe/deepgemm_pr323_w8a8 from upstream/pr-323, but that remote was 23f46aa rather than 805f7e8 and hit the same missing CUTLASS header.

Decision:
- Do not use the historical 2320/3518/21904 us W8A8 numbers as the only comparison until the exact PR323 build environment is restored.
- For current-code comparisons, compact NVFP4 rank0 is close to current in-repo W8A8 at M512/M1024, but max-rank latency remains high for both.


## 2026-06-11: Current W8A8 rank-aware comparison and M2048 follow-up

W8A8 same-repo rank-aware benchmark:

| tokens | W8A8 rank0 | W8A8 mean | W8A8 max |
|---:|---:|---:|---:|
| 32 | 787.5 us | 2762.4 us | 3052.0 us |
| 64 | 805.2 us | 2786.9 us | 3074.0 us |
| 128 | 814.1 us | 2768.1 us | 3087.0 us |
| 256 | 1399.0 us | 3381.2 us | 3671.0 us |
| 512 | 4565.0 us | 6528.0 us | 6815.0 us |
| 1024 | 5666.0 us | 7645.9 us | 7941.0 us |
| 2048 | 10850.0 us | 12835.0 us | 13125.0 us |
| 4096 | 23474.0 us | 25450.6 us | 25743.0 us |
| 8192 | 46233.0 us | 48227.6 us | 48532.0 us |

NVFP4 / current W8A8 ratio:

| tokens | rank0 ratio | max-rank ratio |
|---:|---:|---:|
| 32 | 1.512x | 1.128x |
| 64 | 1.464x | 1.116x |
| 128 | 1.453x | 1.113x |
| 256 | 1.075x | 1.024x |
| 512 | 1.037x | 1.026x |
| 1024 | 1.051x | 1.035x |
| 2048 | 1.271x | 1.223x |
| 4096 | 1.038x | 1.034x |
| 8192 | 1.094x | 1.089x |

M2048 follow-up:
- fp8-shadow upper bound: rank0=10831 us, max_rank=13090 us, essentially matching W8A8. The M2048 gap is therefore compact online dequant/materialization overhead, not general EP rank tail.
- L1 dual-K off: rank0/max 13729/15981 us, no meaningful improvement.
- L2 dual-accum off: rank0/max 13772/16027 us, no improvement.
- Both off: rank0/max 13910/16156 us, worse.
- NUM_STAGES=4: rank0/max 14315/16571 us, worse.
- LOADER_DEQUANT=0: invalid for this M2048 path, triggered CUDA illegal memory access; no default change.
- After that invalid env run, default compact quick correctness was rerun for M32 and M2048 at weight_scale=0.05. Result: PASS, worst cosine_min=0.9995, finite=True.
- Kineto table for M2048 compact split shows two heavy kernels: rank0 about 7.27 ms + 6.49 ms; slower ranks about 9.7 ms + 6.2 ms. Both phases contribute to the gap.
- SPLIT_L1_L2=0 fused single-launch M2048 is much worse: rank0=32041 us, max_rank=34285 us. Simple launch fusion is not the fix.

Decision:
- Keep current defaults. The next M2048-specific optimization must reduce online compact dequant/materialization cost rather than retune scheduling flags.


## 2026-06-11: Reference split NVFP4 comparison - fused small-M default

Reference repo checked: /root/fac/megamoe/DeepGEMM_megamoe_nvfp4.

Relevant difference:
- Reference split NVFP4 defaults to fused single-kernel mode for M64/M128 and BM64 M4096.
- Current fuse repo only defaults fused for M4096 with BM64; M32/M64/M128 remain split L1/L2.

Validation before benchmarking:
- Forced DG_SM90_MOE_SPLIT_L1_L2=0 for M32/M64/M128.
- Correctness PASS for weight_scale 0.001, 0.05, 1.0 with fp8-bridge compact reference.
- Worst cosine_min was 0.9992 and all norm ratios stayed inside 0.99..1.01.

30-run benchmark with DG_SM90_MOE_SPLIT_L1_L2=0:

| tokens | rank0 | mean-rank | max-rank |
|---:|---:|---:|---:|
| 32 | 1235.0 us | 3184.5 us | 3484.0 us |
| 64 | 1223.0 us | 3174.6 us | 3473.0 us |
| 128 | 1220.0 us | 3186.3 us | 3475.0 us |

Comparison against current split default from the latest full 30-run:
- M32 split default was rank0/max 1190.8/3441.9 us, so fused is slower.
- M64 split default was rank0/max 1179.0/3432.2 us, so fused is slower.
- M128 split default was rank0/max 1182.6/3435.2 us, so fused is slower.

Decision:
- Do not copy the reference repo M64/M128 fused default into this fuse repo.
- The true compact path is already doing in-kernel tile dequant, not whole-layer host-side dequant; the next useful reference direction is lower-level dequant/materialization or per-size pipeline changes, not single-launch fusion.


## 2026-06-11: Reference split NVFP4 comparison - split scheduler helpers

Reference repo checked: /root/fac/megamoe/DeepGEMM_megamoe_nvfp4.

Attempt:
- Ported scheduler.for_each_linear1_block and scheduler.for_each_linear2_block from the reference split NVFP4 repo.
- Switched sm90_nvfp4_mega_moe.cuh split-only launches to use the phase-specific helpers instead of the generic get_next_block loop with phase filtering.
- Rationale: reduce split L1/L2 scheduler overhead, especially for small M fixed-cost dominated cases.

Validation:
- Rebuild PASS.
- Correctness PASS for M32, M128, M256, M2048, M4096 and weight_scale 0.001, 0.05, 1.0.
- Worst observed cosine_min was 0.9985; finite=True and norm ratios stayed within 0.99..1.01.

30-run benchmark after the change:

| tokens | rank0 | mean-rank | max-rank |
|---:|---:|---:|---:|
| 32 | 1212.3 us | 3171.4 us | 3458.0 us |
| 64 | 1226.0 us | 3168.4 us | 3473.1 us |
| 128 | 1225.4 us | 3180.4 us | 3473.1 us |
| 256 | 1521.4 us | 3481.4 us | 3769.8 us |
| 512 | 4746.0 us | 6684.5 us | 6992.0 us |
| 1024 | 5992.0 us | 7957.0 us | 8250.0 us |
| 2048 | 13665.0 us | 15748.9 us | 16084.0 us |
| 4096 | 24227.0 us | 26158.4 us | 26463.0 us |
| 8192 | 50789.0 us | 52750.2 us | 53058.0 us |

Outcome:
- M2048/M4096 rank0 improved slightly, but M32/M64/M128/M256 regressed and max-rank did not improve reliably.
- Because default path must not regress small M, the code change was reverted.
- This suggests the generic scheduler loop is not the dominant small-M cost in the fuse repo; the gap is still more likely in online compact dequant/materialization plus split-phase fixed overhead.


## 2026-06-11: Phase profile refresh and async L1 store gate attempt

Phase profile refresh, default compact path, DG_SM90_MOE_PHASE_PROFILE=1, 10 runs:

| tokens | profiled rank0 | mean-rank | max-rank | key observations |
|---:|---:|---:|---:|---|
| 32 | 1427.4 us | 3343.8 us | 3665.1 us | math_loop avg 661653 cycles, loader_dequant avg 1044 cycles, math_dequant_wait avg 464 cycles |
| 2048 | 14631.0 us | 16594.6 us | 16882.0 us | math_loop avg 7211679 cycles, loader_dequant avg 2600 cycles, dispatch_pull avg 404059 cycles |

Interpretation:
- Small M remains fixed-overhead dominated; raw loader-dequant and dequant wait are not individually large.
- M2048 pays the compact materialization cost many times; fp8-shadow matching W8A8 at M2048 still indicates the gap is true compact online dequant/materialization rather than generic EP tail.

Reference split NVFP4 comparison also showed host support for DG_SM90_MOE_ASYNC_L1_STORE was not hard-disabled there, while this fuse repo disables it.

Attempt:
- Temporarily enabled async L1 store as an opt-in env path in the NVFP4 host launcher.
- Added host smem accounting for the double-buffered async L1 output area so the kernel shared-memory layout matched the async branch.

Result:
- Build PASS.
- Correctness with DG_SM90_MOE_ASYNC_L1_STORE=1 hung on the first M32 case before producing kernel output.
- The exact residual bash/python PIDs from this test were killed; no broad python kill was used.

Decision:
- Reverted the async L1 store host changes.
- Do not use async L1 store for NVFP4 until the L1 ready-mask / TMA store drain lifecycle is debugged inside the kernel.


## 2026-06-11: Direct benchmark of reference split NVFP4 repo

Reference repo measured read-only:
- /root/fac/megamoe/DeepGEMM_megamoe_nvfp4
- Branch HEAD: 2d087b2 [nvfp4] Tune small-M heuristics, with local uncommitted NVFP4 changes present.
- Its benchmark script is older and does not support --weight-mode or rank-aware mean/max output, so this is rank0-only.

Command shape:
- python3 -u tests/bench_nvfp4_mega_moe_sm90.py --num-processes 8 --batches 32 64 128 2048 --num-tests 30

Results:

| tokens | reference rank0 |
|---:|---:|
| 32 | 1227.5 us |
| 64 | 1212.0 us |
| 128 | 1219.0 us |
| 2048 | 20857.0 us |

Comparison:
- Current fuse repo latest retained full run was M32=1190.8 us, M64=1179.0 us, M128=1182.6 us, M2048=13787.0 us on rank0.
- Therefore the reference split NVFP4 repo is not a faster compact implementation for these measured cases.
- Directly copying the remaining reference implementation details is unlikely to close the compact/W8A8 gap.


## 2026-06-11: Correct W8A8 comparison after vector-load dequant probe

Correction:
- The earlier W8A8 rank-aware table with M8192=46233 us is not a trustworthy PR323 W8A8 baseline for performance comparison.
- Stable PR323 W8A8 M8192 measurements in this log are around 21904 us, with nearby reruns around 21887-21935 us.
- Therefore comparisons against 46233 us are invalid and overly optimistic for NVFP4.

Vector-load dequant probe:
- Temporarily changed the default loader-dequant two-row helper to read packed FP4 rows as eight uint4 loads instead of 32 scalar uint32 loads.
- Correctness PASSed for M32 and M2048 at weight_scale=0.05.
- Full 30-run result was:

| tokens | vector-load NVFP4 |
|---:|---:|
| 32 | 1258.3 us |
| 64 | 1341.9 us |
| 128 | 1281.1 us |
| 256 | 1650.0 us |
| 512 | 2560.3 us |
| 1024 | 3877.0 us |
| 2048 | 6873.0 us |
| 4096 | 12517.0 us |
| 8192 | 24306.0 us |

Correct comparison against PR323 W8A8:

| tokens | PR323 W8A8 | vector-load NVFP4 | ratio |
|---:|---:|---:|---:|
| 32 | 923.9 us | 1258.3 us | 1.362x |
| 64 | 869.9 us | 1341.9 us | 1.543x |
| 128 | 837.1 us | 1281.1 us | 1.530x |
| 256 | 1460.0 us | 1650.0 us | 1.130x |
| 512 | 2320.0 us | 2560.3 us | 1.104x |
| 1024 | 3518.0 us | 3877.0 us | 1.102x |
| 2048 | 6274.0 us | 6873.0 us | 1.095x |
| 4096 | 11402.0 us | 12517.0 us | 1.098x |
| 8192 | 21904.0 us | 24306.0 us | 1.110x |

Comparison against retained compact default after L2 no-dispatch extension:
- Retained M8192 was 23657 us, so vector-load dequant regressed M8192 by about 2.7%.
- Retained M512/M1024/M2048 were 2401/3803/6719 us, so vector-load also regressed those cases slightly.
- Small M regressed more severely, especially M64.

Decision:
- Rejected and reverted vector-load dequant.
- Keep the scalar two-row loader-dequant helper.
- Use the PR323 W8A8 table around M8192=21904 us for gap reporting unless a fresh same-window W8A8 rerun replaces it.


## 2026-06-12: Block-level pipeline follow-up - async L1 store and wave size probe

Context:
- The latest phase analysis supports the hypothesis that FC1/FC2 dequant is already overlapped with MMA: `math_dequant_wait` is small relative to the total math loop.
- The missing piece is not another simple dequant/MMA overlap knob, but a stable way to overlap L1 epilogue/TMA-store or L2 scatter with later tile load/dequant at block/wave granularity.

### Async L1 TMA store lifecycle fix, opt-in only

Change:
- Re-enabled `DG_SM90_MOE_ASYNC_L1_STORE=1` as an opt-in NVFP4 path only for the safe small-M shape `BM64/BN128/128 epilogue threads`.
- Added host shared-memory accounting for the double-buffered async L1 output area.
- Fixed the previous hang by draining all pending async L1 TMA stores before the split L1-only kernel returns, so L2 phase can observe the published arrival mask.
- L2-only phase explicitly disables the async L1 template flag.

Validation:
- Rebuild PASS.
- Correctness PASS with `DG_SM90_MOE_ASYNC_L1_STORE=1`, batches 32/64/128, weight scales 0.001/0.05/1.0, fp8-bridge compact reference, cosine/norm thresholds 0.99.
- JIT cache inspection confirmed L1 kernels instantiated with async=true and L2 kernels with async=false.

Same-window 30-run A/B, small M:

| tokens | default off | async L1 opt-in | delta |
|---:|---:|---:|---:|
| 32 | 1222.3 us | 1203.5 us | +1.5% |
| 64 | 1203.7 us | 1202.9 us | ~flat |
| 128 | 1211.1 us | 1242.4 us | -2.6% |

Decision:
- Do not enable as default. It is not a stable broad win and M128 regresses.
- Keep only as an opt-in/debug path for now. A possible future M32-only default would need 50-run confirmation because the win is small and small-M tails are noisy.

### Phase profile refresh

Default compact path with `DG_SM90_MOE_PHASE_PROFILE=1`, 5 runs:

| tokens | profiled rank0 | dispatch_total avg cycles | math_loop avg cycles | l1_epi avg | l2_epi avg | loader_dequant avg | math_dequant_wait avg |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 32 | 1458.7 us | 61151 | 675714 | 2188 | 526 | 1046 | 464 |
| 256 | 1684.4 us | 93673 | 782822 | 4384 | 1207 | 1275 | 169 |
| 1024 | 4188.0 us | 244973 | 1966729 | 4008 | 1651 | 1265 | 174 |
| 8192 | 26240.0 us | 1726469 | 12767500 | 4039 | 1817 | 1258 | 175 |

Interpretation:
- Loader-side dequant is visible but `math_dequant_wait` is low, so the math warpgroup is usually not blocked waiting for dequant.
- The remaining compact gap is mostly total block lifecycle/resource pressure: online FP4 unpack + FP8 materialization + barriers + L1/L2 staging, not an isolated dequant-wait stall.

### Smaller experts-per-wave probe, reverted

Hypothesis:
- Reducing `num_experts_per_wave` might make fused/split scheduling switch from L1 to L2 earlier, approximating a finer-grained block-level pipeline.

Temporary change:
- Added `DG_SM90_NVFP4_EXPERTS_PER_WAVE` host override for screening, then reverted after measurement.

Validation:
- `DG_SM90_NVFP4_EXPERTS_PER_WAVE=4` correctness PASS for M32/M256 and weight scales 0.001/0.05/1.0.

30-run benchmark with EPW=4:

| tokens | EPW=4 compact |
|---:|---:|
| 32 | 1476.3 us |
| 64 | 1325.0 us |
| 128 | 1354.5 us |
| 256 | 1702.8 us |
| 512 | 2671.0 us |
| 1024 | 3963.0 us |
| 2048 | 6779.0 us |
| 4096 | 12233.0 us |
| 8192 | 23719.0 us |

Decision:
- Reverted. Smaller wave size starts L2 earlier but increases waiting/scheduling overhead and strongly regresses small M. It is not the right block-level pipeline mechanism.

Current conclusion:
- The true compact NVFP4 path remains slower than PR323 W8A8. Correct PR323 comparison still uses W8A8 M8192 around 21904 us, not the invalid 46233 us table.
- A high-benefit block-level pipeline likely needs a real wavefront scheduler (for example L1 chunk N, then L2 of the previous ready chunk while issuing L1 for the next chunk) rather than simply shrinking expert waves or async-draining L1 stores.


### Async L1 store 50-run follow-up

Because small-M results have visible tail/noise, the apparent 30-run M32 win was rechecked with 50-run A/B:

| tokens | default off 50-run | async L1 opt-in 50-run | result |
|---:|---:|---:|---|
| 32 | 1211.6 us | 1219.0 us | async slower |

Final decision for this attempt:
- Reverted the async L1 store host/kernel changes.
- Do not keep even as an opt-in performance path in this branch; it is correctness-fixable but not a stable latency win.
- The result reinforces that simple async-drain of L1 stores is not sufficient; the needed optimization is a real wavefront/block-level scheduler, not just delaying `tma_store_wait`.


### Final default validation for this round

After reverting the async and EPW experiments, rebuilt the extension and reran default compact correctness:
- PASS for M32/M256, weight scales 0.001/0.05/1.0.
- Dequant unit test PASS and CUDA dequant LUT unit test PASS.
- Worst observed cosine_min was 0.9991; norm ratios stayed within 0.99..1.01.

Latest default compact 30-run benchmark:

| tokens | NVFP4 compact | mean-rank | max-rank | PR323 W8A8 | ratio vs W8A8 |
|---:|---:|---:|---:|---:|---:|
| 32 | 1226.7 us | 1237.6 us | 1266.6 us | 923.9 us | 1.328x |
| 64 | 1205.3 us | 1207.8 us | 1214.7 us | 869.9 us | 1.386x |
| 128 | 1209.0 us | 1203.5 us | 1210.3 us | 837.1 us | 1.444x |
| 256 | 1530.6 us | 1540.3 us | 1552.2 us | 1460.0 us | 1.048x |
| 512 | 2417.0 us | 2436.3 us | 2446.0 us | 2320.0 us | 1.042x |
| 1024 | 3965.0 us | 3952.7 us | 4003.0 us | 3518.0 us | 1.127x |
| 2048 | 6672.0 us | 6665.1 us | 6672.0 us | 6274.0 us | 1.063x |
| 4096 | 12170.0 us | 12162.9 us | 12197.0 us | 11402.0 us | 1.067x |
| 8192 | 23692.0 us | 23689.9 us | 23715.0 us | 21904.0 us | 1.082x |

Round outcome:
- No new performance code change retained from this round.
- The only retained source difference remains the earlier M4096 default policy (`L1_DUAL_K=0`, `L2_DUAL_ACCUM=0`) plus benchmark/log/test harness changes already present before this round.
- The next meaningful implementation direction is a real wavefront scheduler or a lower-cost compact dequant/materialization redesign; simple async store drain, smaller expert waves, stage retuning, no-dispatch small-M, packed scratch, and vector-load dequant have all failed to produce a robust win.


## 2026-06-12: Wavefront fused scheduler prototype, rejected

Hypothesis:
- Current scheduler processes a wave as all L1 blocks followed by all L2 blocks. A fused single-kernel wavefront order might overlap L1 epilogue/staging of later pool blocks with L2 load/dequant of earlier ready pool blocks.

Prototype:
- Added a temporary `get_next_block_wavefront()` scheduler path.
- The order used a small L1 lag/prologue, then issued L1 for the current pool block and L2 for an older pool block within the same fused kernel.
- Temporarily mapped `DG_SM90_MOE_WAVEFRONT_SCHEDULE=1` onto the otherwise unused `l1_nmajor_schedule` template bit for screening.
- Tested with `DG_SM90_MOE_SPLIT_L1_L2=0 DG_SM90_MOE_WAVEFRONT_SCHEDULE=1`.

Correctness:
- PASS for M32/M256 and weight scales 0.001/0.05/1.0 against fp8-bridge compact reference.
- Worst observed cosine_min was 0.9991 and norm ratios stayed within 0.99..1.01.

30-run benchmark:

| tokens | wavefront fused compact | latest split default | result |
|---:|---:|---:|---|
| 32 | 1377.0 us | 1226.7 us | slower |
| 64 | 1398.0 us | 1205.3 us | slower |
| 128 | 1383.0 us | 1209.0 us | slower |
| 256 | 3572.0 us | 1530.6 us | much slower |
| 512 | 5676.0 us | 2417.0 us | much slower |
| 1024 | 9180.0 us | 3965.0 us | much slower |
| 2048 | 16753.0 us | 6672.0 us | much slower |
| 4096 | 20200.0 us | 12170.0 us | slower |
| 8192 | 71738.0 us | 23692.0 us | much slower |

Decision:
- Reverted the prototype.
- The result suggests naive deterministic wavefront ordering makes many CTAs enter L2 too early and wait on arrival masks, while also reintroducing fused-kernel resource pressure. It does not create the stable high-benefit block-level pipeline we need.
- A future wavefront design would need an explicit ready-aware work queue or a more balanced producer/consumer partition, not just reordered static block indices.


## 2026-06-12: Fused producer/consumer SM partition prototype, rejected

Hypothesis:
- Instead of statically reordering individual blocks, split SMs inside the fused kernel into L1 producers and L2 consumers. L1 SMs keep producing L1 output tiles while L2 SMs consume ready arrival masks, providing explicit block-level phase overlap.

Prototype:
- Added a temporary scheduler `block_stride` so different SM partitions could cover disjoint L1 and L2 block ranges without dropping blocks.
- In fused mode, `DG_SM90_MOE_PC_PIPELINE=1` used about half the SMs for Linear1 and half for Linear2.
- Tested with `DG_SM90_MOE_SPLIT_L1_L2=0 DG_SM90_MOE_PC_PIPELINE=1`.

Correctness:
- PASS for M32/M256 and weight scales 0.001/0.05/1.0 against fp8-bridge compact reference.
- Worst observed cosine_min was 0.9991 and norm ratios stayed within 0.99..1.01.

30-run benchmark:

| tokens | PC fused compact | latest split default | result |
|---:|---:|---:|---|
| 32 | 1660.0 us | 1226.7 us | slower |
| 64 | 1710.0 us | 1205.3 us | slower |
| 128 | 1721.0 us | 1209.0 us | slower |
| 256 | 3689.0 us | 1530.6 us | much slower |
| 512 | 5770.0 us | 2417.0 us | much slower |
| 1024 | 9097.0 us | 3965.0 us | much slower |
| 2048 | 16096.0 us | 6672.0 us | much slower |
| 4096 | 21432.0 us | 12170.0 us | slower |
| 8192 | 57184.0 us | 23692.0 us | much slower |

Decision:
- Reverted the prototype.
- Correctness proves the producer/consumer partition is semantically possible, but performance is worse than split L1/L2. Halving the active SMs per layer plus fused-kernel resource pressure costs more than the overlap can recover.
- This also explains why the current split-L1/L2 default remains best: giving each phase the full GPU is still faster than coarse concurrent partitioning.


## 2026-06-12: Literature reassessment after block-level pipeline failures

Checked current W4A8 references after repeated no-win iterations:
- LiquidGEMM / LiquidQuant reports that high-performance W4A8 needs two things together: a dequant-friendly quantization method with very low arithmetic cost, and an implicit fine-grained pipeline that overlaps weight loading, dequantization, and MMA without extra software synchronization or redundant memory traffic. Paper: https://arxiv.org/abs/2509.01229
- QServe/QoQ similarly frames W4A8 performance around minimizing CUDA-core dequant overhead and using weight reordering/register-level parallelism. Paper: https://arxiv.org/abs/2405.04532

Relevance to this NVFP4 bridge:
- The current true compact NVFP4 path must decode E2M1 nibbles with UE4M3 per-16 scales, materialize FP8 B into shared memory, then feed Hopper WGMMA. That creates extra shared writes, barriers, and later WGMMA shared reads.
- Profile evidence already shows `math_dequant_wait` is small, so scheduling overlap alone is not enough. The cost is the whole bridge/materialization lifecycle.
- Static wavefront and producer/consumer fused scheduling both made correctness pass but performance worse, confirming that simply adding overlap without reducing materialization cost does not beat split L1/L2.

Implication:
- To beat PR323 W8A8 with true NVFP4 on SM90, the next real implementation would need a new compact compute path, such as a fully correct small-M mma.sync/register-fed design or a different dequant-friendly weight representation prepared at load time. More env/shape tuning is unlikely to close the gap.


## 2026-06-12: mma.sync path reassessment

Purpose:
- After block-level pipeline attempts failed, reassess the remaining small-M path: a dedicated mma.sync/register-style kernel that avoids WGMMA's M64 padding and may reduce small-M fixed overhead.

Findings from current code and CUTLASS reference:
- Current dormant NVFP4 mma.sync branch uses `SM89_16x8x32_F32E4M3E4M3F32_TN` and writes FP32 accumulators through `smem_accum_f32` before running the MegaMoE epilogue.
- Earlier attempts already fixed host smem accounting and dequant wait enough to avoid illegal memory access, but results were either NaN or numerically exploded.
- CUTLASS's own cooperative GEMM unit test for the same atom uses:
  - `TiledMMA<MMA_Atom<SM89_16x8x32_F32E4M3E4M3F32_TN>, Layout<Shape<_2,_2,_1>>>`
  - logical B layout as `(N, K)` with the helper's col-major/cooperative-copy path
  - C fragment stored via `thr_mma.partition_C(...)`, not by assuming a simple row-major full `BLOCK_M x BLOCK_N` tile is covered by a manually chosen atom count.
- The current MegaMoE branch instead builds row-major `sA`, row-major `sB`, and a row-major `sC` over `BLOCK_M x 128`, then tries to hand-map `tCrFinal` into `smem_accum_f32`. This likely mismatches the atom's A/B/C layout semantics.

Implication:
- Re-opening `DG_SM90_NVFP4_MMA_SYNC=1` inside MegaMoE is still not the right next edit.
- The next real mma.sync attempt should first create a standalone M32/N128/K128 FP8 GEMM/dequant unit test using the CUTLASS cooperative_gemm pattern, validate C fragment coordinates and B layout, then port only that verified layout back into MegaMoE.
- Until that standalone test exists, direct MegaMoE mma.sync edits are likely to repeat the previous NaN/exploding-output failure.


## 2026-06-12: Direct L2 scatter sync-skip probe, rejected

Hypothesis:
- In the direct L2 scatter path (`BN128`), the per-block `ptx::sync_aligned(kNumEpilogueThreads, kEpilogueFullBarrierIdx)` after register-to-NVLink scatter might serialize scatter tail work before the next block can enter load/dequant/MMA.
- If direct scatter does not reuse `smem_cd_l2`, skipping that barrier could expose more block-level overlap between L2 epilogue/scatter and the next block's pipeline.

Temporary change:
- Added opt-in `DG_SM90_NVFP4_SKIP_L2_SCATTER_SYNC=1` using an extra bit in `kInstantiationTag`.
- In `kDirectL2Scatter`, skipped the post-scatter epilogue full barrier when the opt-in bit was set.

Validation result:
- Rebuild PASS.
- Correctness with the opt-in passed the first M32 / weight_scale=0.001 case.
- The same run then hung at M256 before producing the case output. This indicates the post-scatter sync is still part of the required warpgroup/block lifecycle, even though direct scatter does not reuse the L2 SMEM output buffer.
- The exact residual correctness PIDs were inspected and only those PIDs were killed; no broad `pkill python` was used.

Decision:
- Reverted the code change completely; no `.cuh` diff remains from this probe.
- Do not skip the direct L2 scatter sync. A safe block-level pipeline needs a different synchronization design, not just removing this barrier.


## 2026-06-12: Standalone SM89 mma.sync layout probe and MegaMoE retry

Purpose:
- Revisit the BM32 `mma.sync` path with a standalone unit probe before touching MegaMoE, because earlier direct MegaMoE attempts produced NaN/exploding output.

Standalone probe:
- Created `/tmp/sm89_mma_probe.cu` outside the repo.
- Shape: M32/N128/K128, `SM89_16x8x32_F32E4M3E4M3F32_TN`, one 128-thread CTA.
- Used CUTLASS/CUTE `cooperative_gemm`, `thr_mma.partition_C`, and CPU reference using the same FP8-rounded inputs.

Results:
- `TiledMMA<MMA_Atom<SM89_16x8x32_F32E4M3E4M3F32_TN>, Layout<Shape<_2,_2,_1>>>`: PASS, `bad=0`, `max_abs=0`, `max_rel=0`.
- Same tiled layout with row-major A and row-major C mailbox, matching MegaMoE's intended SMEM layout: PASS, `bad=0`, `max_abs=0`, `max_rel=0`.
- Current MegaMoE-style tiled layout `Layout<Shape<_2,_4,_1>>`: FAIL, 2034 bad elements, many columns zero from n=16 onward, `max_abs=0.429688`, `max_rel=1`.

Useful finding:
- The previous dormant MegaMoE mma.sync layout was definitely wrong for M32/N128/K128. The validated tiled layout is `_2,_2,_1`, not `_2,_4,_1`.

MegaMoE retry:
- Temporarily added opt-in `DG_SM90_NVFP4_MMA_SYNC=1` to select BM32/BN128/128-epi-thread shape.
- Fixed host shared-memory accounting to include the BM32 FP32 accumulator mailbox and to disable direct L2 scatter for `kUseMMASync`, matching the kernel's compile-time gate.
- Switched the MegaMoE mma.sync branch to the validated `_2,_2,_1` tiled layout.
- First retry hung at M32 before output. Then added a missing `dequant_barriers[stage_idx]->wait(phase)` in the loader-dequant case.
- Second retry no longer hung but failed correctness with non-finite output (`Kernel: abs_max=nan`, cosine/norm `nan`).

Decision:
- Reverted all MegaMoE source changes from this retry; no `.cuh` diff remains.
- Keep the standalone probe result as guidance: `_2,_2,_1` is required, but the full MegaMoE mma.sync port still needs additional lifecycle/numeric debugging before it can become a performance candidate.
- Do not enable `DG_SM90_NVFP4_MMA_SYNC` in default or opt-in code until M32 correctness passes.

Final validation after reverting failed probes:
- Rebuilt the extension after reverting the direct-scatter sync-skip and MegaMoE mma.sync source changes.
- Default compact correctness smoke PASS for M32/M256, weight_scale=0.05, fp8-bridge reference.
- Dequant unit test PASS and CUDA LUT dequant unit test PASS.
- No new performance code was retained from this round, so the latest valid benchmark remains the previous default 30-run table.


## 2026-06-12: no-swizzle BM32 mma.sync probe, correctness pass but performance rejected

Hypothesis:
- The earlier MegaMoE mma.sync retry failed because the WGMMA path dequantizes B into an XOR-swizzled FP8 shared-memory layout, while the SM89_16x8x32 mma.sync branch reads B as a plain row-major tensor.
- If the opt-in BM32 mma.sync path dequantizes B into a no-swizzle FP8 SMEM layout and also uses an unswizzled activation TMA descriptor, the path should become numerically correct.

Temporary change:
- Added dequant_smem_b_inplace_two_rows_no_swizzle and used it only when kUseMMASync.
- Kept default WGMMA compact NVFP4 path unchanged.
- Added opt-in DG_SM90_NVFP4_MMA_SYNC=1 to select BM32/BN128/128 epilogue threads, disable direct L2 scatter, reserve FP32 accumulator mailbox SMEM, and set config.swizzle_acts_mode=0 for activation TMA.
- Switched the mma.sync tiled layout to the standalone-validated Layout<Shape<_2,_2,_1>> and waited on dequant_barriers before consuming loader-dequantized B.

Correctness:
- Initial run without host activation swizzle gating failed with cosine_mean=0.1652 because the kernel read A as unswizzled while TMA still produced a 128B-swizzled SMEM tile.
- After setting config.swizzle_acts_mode=0 for the opt-in path, M32 weight_scale=0.05 PASS: cosine_min=0.9996, cosine_mean=0.9997, norm_ratio=0.9998.
- Extended correctness PASS for M32/M64/M128 and weight_scale 0.001/0.05/1.0; worst observed cosine_min=0.9992, norm ratios stayed within 0.99..1.01.

Benchmark, 8 ranks, 50 tests, compact mode, DG_SM90_NVFP4_MMA_SYNC=1:

| tokens | opt-in BM32 mma.sync | latest default compact NVFP4 | PR323 W8A8 | conclusion |
| ---: | ---: | ---: | ---: | --- |
| 32 | 2780.2 us | 1226.7 us | 923.9 us | 2.27x slower than default |
| 64 | 2816.2 us | 1205.3 us | 869.9 us | 2.34x slower than default |
| 128 | 4509.0 us | 1209.0 us | 837.1 us | 3.73x slower than default |

Decision:
- Correctness is now fixed for the opt-in BM32 mma.sync path, but performance is far below both default compact NVFP4 and PR323 W8A8.
- Do not use this path as default and do not continue optimizing it before there is a separate explanation for the massive fixed overhead.
- Return to the default WGMMA compact NVFP4 path and focus on block-level scheduling: epilogue/scatter overlap with the next tile load/dequant.

## 2026-06-12: block-level pipeline follow-up, retained gated defaults

Goal:
- Continue optimizing the true compact NVFP4 MegaMoE kernel based on the analysis that dequant/MMA overlap already exists in FC1/FC2, while epilogue/scatter and next-tile load/dequant need better block-level scheduling.
- Target: close the gap to PR323 W8A8, preferably beat it for at least one meaningful M range.

Implemented and retained:
- Ported the FP8-style L2 arrival-counter idea into NVFP4 as a template flag.
  - L1 epilogue can publish one ready event per active M warpgroup with red_add_rel on the L2 arrival word.
  - L2 A/SFA loaders can wait on an expected counter instead of the old bitmask.
  - This removes the CTA-wide L1 epilogue full sync on gated sizes.
- Added host default gating for DG_SM90_NVFP4_L2_ARRIVAL_COUNTER:
  - default ON for M=32,512,2048,8192
  - default OFF for M=64,128,256,1024,4096 after measurements showed regressions or instability there.
- Changed L1 dual-K default:
  - keep ON for M=64,128,256
  - turn OFF for M>=512
  - reason: dual-K was negative for M512/M1024/M2048 but still useful or neutral for smaller sizes.
- Changed L2 N-major default to ON for M128 as well.

Correctness:
- Rebuilt successfully.
- Final default correctness PASS for M=128,512,1024,2048,4096,8192, weight_scale=0.05, fp8-bridge reference.
- Earlier in the same round, default correctness also PASS for M=32,64,128,256,512,1024,2048,4096,8192.
- Dequant unit test PASS and CUDA LUT dequant unit test PASS in all correctness runs.

Final benchmark notes:
- 50-run benchmarks still show sweep-to-sweep long-tail effects, especially when many sizes are run in one process.
- The most reliable final numbers below are from single-size or small grouped 50-run measurements after final code cleanup.

| tokens | final NVFP4 | previous NVFP4 default | PR323 W8A8 | vs prev NVFP4 | vs W8A8 |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 32 | 1194.4 us | 1226.7 us | 923.9 us | +2.7% | 1.293x slower |
| 64 | 1199.7 us | 1205.3 us | 869.9 us | +0.5% | 1.379x slower |
| 128 | 1208.2 us | 1209.0 us | 837.1 us | +0.1% | 1.443x slower |
| 256 | 1510.1 us | 1530.6 us | 1460.0 us | +1.4% | 1.034x slower |
| 512 | 2273.4 us | 2417.0 us | 2320.0 us | +6.3% | 0.980x, faster than W8A8 |
| 1024 | 3778.0 us | 3965.0 us | 3518.0 us | +5.0% | 1.074x slower |
| 2048 | 6608.0 us | 6672.0 us | 6274.0 us | +1.0% | 1.053x slower |
| 4096 | 12257.0 us | 12170.0 us | 11402.0 us | -0.7% in latest run | 1.075x slower |
| 8192 | 23295.0 us | 23692.0 us | 21904.0 us | +1.7% | 1.064x slower |

Useful attempts:
- L2 arrival counter: useful for M32/M512/M2048/M8192; not stable or negative for M64/M128/M256/M1024/M4096, so it is gated.
- L1 dual-K OFF: useful for M512/M1024/M2048 and retained by default for M>=512.
- M128 L2 N-major ON: single-size 50-run improved to 1204.7-1208.2 us and is now default.

Rejected attempts:
- BM32 mma.sync/no-swizzle: correctness was fixed, but performance was far worse than WGMMA default (M32 2780us, M64 2816us, M128 4509us), so all source changes from that retry were removed from the retained code.
- Fused L1/L2 single kernel: correctness passed but was slower at both small and large M. M32/64/128 were 1280/1243/1306us; M256/512/1024 were 3477/5731/8947us.
- EPI256 for BN128: JIT static assert, because the current WGMMA path requires one M64 tile per warpgroup and BN128+2WG would make WG_BLOCK_M=32.
- L2_NO_DISPATCH_PIPELINE=0: correctness passed but was slower (M1024 4231us, M2048 7324us), so the existing no-dispatch L2 pipeline remains default for 256+.
- L2_NMAJOR=0 for M32/M64: M64 regressed; not retained.

Conclusion:
- This round produced a real structural improvement in the 512-2048 region, with M512 now slightly faster than the PR323 W8A8 baseline in the retained default path.
- Small M remains far behind W8A8 because fixed dispatch/combine/scheduler cost dominates and W4/NVFP4 dequant cannot compensate at M32/64/128.
- Remaining gap for M1024+ is now mostly outside the dequant wait itself; further work should focus on reducing L2 scatter/combine fixed cost and improving scheduling stability rather than rewriting dequant again.

## 2026-06-12: PR323 combine chunking and direct-scatter metadata probes

Purpose:
- Continue the block-level pipeline work after confirming dequant/MMA overlap already exists in FC1/FC2.
- Test whether PR323-style combine chunking or lower direct-scatter metadata overhead can reduce the remaining L2 epilogue/scatter/combine fixed cost.

Attempt 1: configurable combine chunk count, rejected as default:
- Added a temporary compile-time DG_SM90_NVFP4_COMBINE_CHUNKS override to test the PR323 idea of splitting hidden=7168 combine into more chunks.
- Correctness PASS for chunks=7 and chunks=4 on M512/M1024/M2048, weight_scale=0.05.
- Correctness PASS for chunks=7, chunks=4, and chunks=14 on M4096/M8192, weight_scale=0.05.

30-run signal benchmark:

| chunks | M512 | M1024 | M2048 | M4096 | result |
|---:|---:|---:|---:|---:|---|
| 7 | 2314.2 us | 3786.0 us | 6475.0 us | 12358.0 us | M2048 looked better, others neutral/slower |
| 4 | 2352.1 us | 3804.0 us | 6547.0 us | 12225.0 us | M4096 looked slightly better, M512/M1024 slower |

50-run follow-up:

| chunks | M2048 | M4096 | M8192 | result |
|---:|---:|---:|---:|---|
| 7 | 6633.0 us | 12325.0 us | 23341.0 us | no stable win versus retained default |
| 4 | 6613.0 us | 12202.0 us | 23266.0 us | tiny M4096/M8192 signal, not enough for default |
| 14 | - | 12246.0 us | 23606.0 us | worse |

Same-code A/B check after a narrow default gate was briefly tried:

| setting | M4096 | M8192 | note |
|---|---:|---:|---|
| default gate to chunks=4 | 12116.0 us | 23358.0 us | same recv distribution as chunks=0 run |
| explicit chunks=0 | 12110.0 us | 23392.0 us | M4096 was slightly faster with chunks=0 |

Decision:
- Removed the combine chunk override from the retained source.
- More chunks reduce per-lane reduce registers but add more TMA load/store chunks. On this kernel the tradeoff is noisy and not a stable default win.
- Do not default-enable PR323-style 7-way combine chunking for the current true compact NVFP4 path.

Attempt 2: direct L2 scatter metadata broadcast, rejected:
- Hypothesis: in the direct L2 scatter path, every 4 lanes for the same output row redundantly load the same TokenSrcMetadata; only col_idx==0 could load and then shuffle rank/token/topk to the row group.
- Implemented the broadcast temporarily in scatter_direct_row.
- Rebuild PASS.
- Correctness hung at M32 before producing case output.
- Only the residual PIDs from this correctness command were killed: bash PID 2075080 and python PID 2075104. No broad pkill python was used.
- Reverted the metadata broadcast source change.

Final state after reverting rejected probes:
- Rebuild PASS.
- Default correctness smoke PASS for M32/M512/M4096, weight_scale=0.05, fp8-bridge compact reference.
- Dequant unit test PASS and CUDA LUT dequant unit test PASS.
- A later 8-rank benchmark after the revert was not used as a performance conclusion because GPU0 had an unrelated external /usr/bin/python3 process at 100% utilization and about 9.6 GB memory. That run produced doubled latencies and is considered contaminated.
- Latest valid retained benchmark remains the previous single-size/small-group 50-run table:
  - M32 1194.4 us
  - M64 1199.7 us
  - M128 1208.2 us
  - M256 1510.1 us
  - M512 2273.4 us
  - M1024 3778.0 us
  - M2048 6608.0 us
  - M4096 12257.0 us
  - M8192 23295.0 us

Conclusion:
- This continuation did not find a new stable default improvement beyond the retained L2 arrival-counter and heuristic changes.
- PR323 combine chunking is not directly transferable as a default for NVFP4.
- Direct-scatter metadata broadcast is unsafe in the current warp/lane lifecycle and was reverted.
- Further benchmark work should wait until GPU0 is free, or use an isolated 8-GPU window, because one busy rank is enough to invalidate max-rank MegaMoE timings.


## 2026-06-12: Row-mask direct-scatter metadata broadcast retained; pipeline probes

Purpose:
- Continue from the analysis that FC1/FC2 dequant/MMA overlap is already present.
- Focus on the remaining fixed overhead around L1 epilogue ready notification and L2 direct scatter.
- Keep true compact NVFP4 input path; do not use FP8 shadow weights.

Important correction to the previous metadata-broadcast attempt:
- The earlier direct-scatter metadata broadcast hung because it used a full-warp `__shfl_sync(0xffffffff)` while partial rows can skip the shuffle, so not all lanes in the mask participated.
- Reimplemented it with a 4-lane row-group mask: leader is `lane_idx & ~3u`, mask is `0xfu << leader`.
- Added a compile-time and host env gate `DG_SM90_NVFP4_DIRECT_SCATTER_METADATA_BCAST`.
- Default gate: ON only when direct L2 scatter is active and `num_tokens >= 512`.
- M256 and below keep the old per-lane metadata path by default.

Retained source changes in this continuation:
- `kDirectScatterMetadataBroadcast`: one lane per row-group loads `TokenSrcMetadata`, then broadcasts rank/token/topk to the 4 lanes that write that row.
- `kL2ArrivalCounter`: kept from the previous retained state. L1 publishes ready events with a counter on selected sizes, avoiding an epilogue-wide sync in the gated path.
- Host defaults currently retained:
  - `DG_SM90_NVFP4_L2_ARRIVAL_COUNTER`: default ON for M32/M512/M2048/M8192.
  - `DG_SM90_NVFP4_DIRECT_SCATTER_METADATA_BCAST`: default ON for M>=512 when direct L2 scatter is active.
  - `DG_SM90_MOE_L1_DUAL_K`: default OFF for M>=512.
  - `DG_SM90_MOE_L2_NMAJOR`: default ON.

Correctness:
- Rebuild PASS after final retained source.
- Final correctness command used independent JIT cache, no global cache deletion:
  - `DG_JIT_CACHE_DIR=/tmp/dg_jit_nvfp4_metadata_only_final_correct_0612`
  - `tests/test_nvfp4_mega_moe_sm90_correctness.py --batches 32 512 1024 2048 4096 --weight-scales 0.05 --reference-mode fp8-bridge --weight-mode compact --cosine-mean-threshold 0.99 --cosine-min-threshold 0.99 --norm-ratio-min 0.99 --norm-ratio-max 1.01`
- PASS for M32/M512/M1024/M2048/M4096.
- Dequant unit test PASS and CUDA LUT dequant unit test PASS.
- Earlier wide correctness in the same continuation also PASS for M32/M64/M128/M256/M512/M1024/M2048/M4096.

Final current-source 50-run benchmark:
- Command shape: `python3 tests/bench_nvfp4_mega_moe_sm90.py --num-processes 8 --batches <M> --num-tests 50 --weight-mode compact`.
- Main final sweep log: `/tmp/nvfp4_metadata_only_final_m*_50_0612.log`.
- Rerun logs for noisy M32/M64/M1024: `/tmp/nvfp4_metadata_only_final_rerun_m*_50_0612.log`.
- Small-M numbers remain noisy; M64 recovered on rerun, M32 did not.

| tokens | final/latest NVFP4 | PR323 W8A8 | NVFP4/W8A8 | note |
| ---: | ---: | ---: | ---: | --- |
| 32 | 1282.7 us | 923.9 us | 1.388x slower | latest rerun; small-M long tail still severe |
| 64 | 1191.5 us | 869.9 us | 1.370x slower | rerun recovered from 1459.3us long tail |
| 128 | 1205.9 us | 837.1 us | 1.441x slower | final sweep |
| 256 | 1524.4 us | 1460.0 us | 1.044x slower | final sweep |
| 512 | 2257.8 us | 2320.0 us | 0.973x, 2.7% faster | retained path beats W8A8 here |
| 1024 | 3751.0 us | 3518.0 us | 1.066x slower | rerun |
| 2048 | 6597.0 us | 6274.0 us | 1.051x slower | final sweep |
| 4096 | 12073.0 us | 11402.0 us | 1.059x slower | final sweep |
| 8192 | 23143.0 us | 21904.0 us | 1.057x slower | final sweep |

Best current-source 50-run signals observed in this continuation:
- M512: 2246.1-2257.8 us depending on run, consistently faster than PR323 W8A8 2320.0 us.
- M1024: best observed 3720.0 us, latest retained after reverting pointer-broadcast 3751.0 us.
- M2048/M4096/M8192 still land about 5-6% slower than W8A8 in latest retained runs.

Pipeline and scatter probes tried:

1. Async L1 TMA store opt-in, rejected and reverted:
- Temporarily allowed `DG_SM90_MOE_ASYNC_L1_STORE=1` through the host launcher.
- Correctness PASS for M512/M1024.
- 50-run benchmark with async ON:
  - M512 2250.5 us, M1024 3737.0 us, M2048 6631.0 us, M4096 12021.0 us, M8192 23220.0 us.
- Not a stable win: M1024 was only marginally better; M2048/M8192 regressed.
- Reverted host opt-in; final source keeps the original host assert and default false.

2. Fully fused L1+L2 single kernel, rejected:
- Tested `DG_SM90_MOE_SPLIT_L1_L2=0` on true compact NVFP4.
- Correctness PASS for M512/M1024.
- 50-run benchmark was much slower:
  - M32 1259.0 us, M64 1274.0 us, M128 1267.0 us, M256 3589.0 us, M512 3160.0 us, M1024 5348.0 us, M2048 9543.0 us, M4096 18057.0 us, M8192 34196.0 us.
- Conclusion: saving the split launch is not enough; fused path increases resource/scheduling pressure too much.

3. Direct-scatter pointer broadcast, rejected and reverted:
- Hypothesis: after metadata broadcast, only `col_idx==0` should compute `sym_buffer.map()` and broadcast the mapped pointer to the row group.
- Correctness PASS for M512/M1024.
- 50-run benchmark:
  - M512 2246.1 us, M1024 3720.0 us, M2048 6608.0 us, M4096 12036.0 us, M8192 23154.0 us.
- M512/M1024 had tiny signal, but M2048+ was neutral or worse and the extra template branch was not worth retaining.
- Reverted pointer-broadcast gate and code; final source keeps metadata-only broadcast.

4. Phase profile comparison:
- Default phase profile M512/M1024 showed `math_dequant_wait` average around 130-138 cycles, while `math_loop` max was much longer.
- Async L1 store did not materially change phase timings; it mostly moved or exposed waiting rather than reducing end-to-end latency.
- This supports the earlier conclusion: dequant/MMA overlap is not the main remaining bottleneck.

Current conclusion:
- The only new retained improvement in this continuation is safe row-mask metadata broadcast for direct L2 scatter, plus the already retained L2 arrival-counter heuristic.
- M512 is now consistently the one size where true compact NVFP4 beats PR323 W8A8.
- M1024-8192 remain roughly 5-6% slower than W8A8; small M is still dominated by fixed dispatch/combine/scheduler overhead and is far behind W8A8.
- Next credible direction is not more dequant tuning. It should be a real scheduling/layout change that reduces L2 scatter/combine and max-rank tail, or a new pipeline that lets L2 consume L1 output at finer granularity without the current fused-kernel resource penalty.


Addendum: direct-scatter final sync skip probe, rejected and reverted:
- Tried an opt-in `DG_SM90_NVFP4_SKIP_DIRECT_SCATTER_SYNC=1` gate to skip the epilogue-wide barrier after direct L2 scatter.
- Motivation: this was the most direct test of overlapping L2 scatter with the next block load/dequant lifecycle.
- Rebuild PASS, dequant unit test PASS, CUDA LUT unit test PASS.
- Correctness hung at M512 before producing the case result and timed out after 20 minutes.
- Cleaned only the residual processes from this exact command: container bash PID 2284721, container python PID 2284836, and the mapped host GPU PID 362462. No broad pkill was used.
- Reverted the skip-sync template/env/code completely.
- Rebuilt the reverted source and ran M512 correctness smoke PASS with `DG_JIT_CACHE_DIR=/tmp/dg_jit_nvfp4_after_skip_revert_correct_0612`.
- Conclusion: the direct-scatter epilogue-wide barrier is required for the current warpgroup/block lifecycle. A real overlap design must replace it with a correct per-warpgroup or per-buffer handoff, not simply remove it.


Addendum: direct-scatter per-warpgroup sync probe, rejected and reverted:
- Tried replacing the direct-scatter epilogue-wide barrier with a per-warpgroup barrier under an opt-in `DG_SM90_NVFP4_DIRECT_SCATTER_WG_SYNC=1` gate.
- Motivation: preserve intra-WG lifecycle safety while allowing different epilogue WGs to overlap scatter with later block work.
- Rebuild PASS, dequant unit test PASS, CUDA LUT unit test PASS.
- Correctness hung at M512 before producing the case result and timed out after 15 minutes.
- Cleaned only the residual processes from this exact command: container bash PID 2285731, container python PID 2285846, and the mapped host GPU PID 376218. No broad pkill was used.
- Reverted the per-WG sync template/env/code completely.
- Rebuilt the reverted source and ran M512 correctness smoke PASS with `DG_JIT_CACHE_DIR=/tmp/dg_jit_nvfp4_after_wg_sync_revert_correct_0612`.
- Conclusion: both removing the direct-scatter full barrier and weakening it to per-WG sync are unsafe in the current implementation. The next viable design needs an explicit producer/consumer handoff for the relevant shared scheduler or pipeline state rather than weakening this barrier in place.


## 2026-06-12: Retune retained config gates after metadata broadcast

Purpose:
- Recheck existing retained knobs after row-mask metadata broadcast changed the L2 direct-scatter shape.
- No new kernel algorithm was added in this step; only default heuristics were retuned using existing env gates.

50-run A/B results:

| tokens | setting | time | decision |
| ---: | --- | ---: | --- |
| 1024 | default before retune | 3748.0 us | baseline |
| 1024 | L2 arrival counter ON | 3708.0 us | useful |
| 1024 | L2 dual accum OFF | 3727.0 us | less useful than arrival ON |
| 1024 | metadata broadcast OFF | 3762.0 us | worse |
| 2048 | default | 6560.0 us | keep existing arrival ON, dual ON |
| 2048 | L2 arrival counter OFF | 6600.0 us | worse |
| 2048 | L2 dual accum OFF | 6572.0 us | slightly worse |
| 2048 | metadata broadcast OFF | 6581.0 us | worse |
| 4096 | default before retune | 12081.0 us | baseline |
| 4096 | L2 arrival counter ON | 11989.0 us | useful |
| 4096 | L2 dual accum ON | 12044.0 us | better than old default but not as good as arrival ON with dual OFF |
| 4096 | metadata broadcast OFF | 12122.0 us | worse |
| 8192 | default before retune | 23224.0 us | baseline |
| 8192 | L2 arrival counter OFF | 23312.0 us | worse |
| 8192 | L2 dual accum OFF | 23122.0 us | useful |
| 8192 | metadata broadcast OFF | 23244.0 us | worse |

Combo checks:
- M1024 arrival ON + L2 dual OFF: 3722.0 us, worse than arrival ON alone.
- M4096 arrival ON + L2 dual ON: 11998.0 us, about the same as arrival ON with dual OFF, but not better.
- M8192 L2 dual OFF + metadata OFF: 23237.0 us, worse than keeping metadata broadcast ON.

Retained default heuristic changes:
- `DG_SM90_NVFP4_L2_ARRIVAL_COUNTER` default now includes M1024 and M4096.
- `DG_SM90_MOE_L2_DUAL_ACCUM` default is now OFF for M8192 as well as M64/M4096.

Correctness after heuristic change:
- Rebuild PASS.
- `DG_JIT_CACHE_DIR=/tmp/dg_jit_nvfp4_config_defaults_correct_0612`
- PASS for M1024/M4096/M8192 with fp8-bridge compact reference and strict cosine/norm thresholds.
- Dequant unit test PASS and CUDA LUT dequant unit test PASS.

Updated-default 50-run benchmark:

| tokens | updated default NVFP4 | PR323 W8A8 | NVFP4/W8A8 |
| ---: | ---: | ---: | ---: |
| 512 | 2271.2 us | 2320.0 us | 0.979x, 2.1% faster |
| 1024 | 3691.0 us | 3518.0 us | 1.049x slower |
| 2048 | 6618.0 us | 6274.0 us | 1.055x slower |
| 4096 | 11995.0 us | 11402.0 us | 1.052x slower |
| 8192 | 23126.0 us | 21904.0 us | 1.056x slower |

Conclusion:
- Retuning existing gates gives a small but real improvement for M1024 and M4096, and a small M8192 improvement.
- It still does not close the remaining W8A8 gap except at M512.
- Metadata broadcast remains useful for M1024+ and should stay default ON for direct-scatter sizes.


Addendum: BM64 and loader-dequant-off probes, rejected:
- Tested whether PR323 W8A8 block64 default heuristic transfers to compact NVFP4 by disabling `DG_SM90_NVFP4_BM128_HEURISTIC` and forcing split L1/L2.
- 50-run results were much worse:
  - M1024 5256.0 us
  - M2048 9799.0 us
  - M4096 18708.0 us
  - M8192 36650.0 us
- Decision: keep BM128 heuristic for M>=256. PR323 block64 is not transferable to this NVFP4 bridge.

- Tested `DG_SM90_NVFP4_LOADER_DEQUANT=0` on M1024/M4096 to revalidate that loader-side dequant remains required.
- M1024 failed with CUDA illegal memory access during benchmark warmup.
- No residual GPU process remained after the failure.
- Decision: do not use math-side fallback dequant for this path; keep loader dequant default ON.


## 2026-06-12: Block-level pipeline follow-up and M512 dual-accum retune

Purpose:
- Follow up on the analysis that FC1/FC2 dequant-MMA overlap is already present, while the remaining gap may come from epilogue/scatter and next-tile load/dequant handoff.
- Focus on true compact NVFP4, not fp8-shadow.

Phase profile evidence:
- Re-ran phase profiling for M512/M1024/M2048/M4096/M8192 with current defaults.
- `math_dequant_wait` stayed very low, around 114-138 cycles average.
- `l1_epilogue` and `l2_epilogue` were also small per block, around 2.7-3.0K cycles and 1.3-1.7K cycles average respectively.
- `math_loop` and dispatch pull dominate cumulative time. This indicates the current loader warps already overlap next-tile load/dequant with math epilogue reasonably well; simply weakening epilogue barriers is not safe or sufficient.

Rejected experiments:

1. BM128 with `DG_SM90_NVFP4_EPILOGUE_THREADS=128`:
- Temporarily allowed the env override to stay inside the BM128 heuristic.
- JIT failed at M512 with static assertion: `Each warpgroup must run exactly one WGMMA per K-block`.
- Reason: 1 epilogue WG does not match the current BM128 WGMMA tiling.
- Reverted the host heuristic change; BM128 remains fixed at 256 epilogue threads.

2. Direct-scatter K2 direct accumulate:
- Prototype allowed `DG_SM90_MOE_K2_DIRECT_ACCUM=1` with direct L2 scatter.
- L2 epilogue atomically accumulated BF16 pairs into symmetric combine slot0, then skipped topk combine reduce and only copied slot0 to `y`.
- Correctness PASS for M512/M1024, weight_scale=0.05, strict cosine/norm thresholds.
- 30-run benchmark with opt-in was much slower:
  - M512 3047.0 us
  - M1024 5222.0 us
  - M2048 9181.0 us
  - M4096 14323.0 us
  - M8192 27631.0 us
- Decision: rejected and reverted. BF16 atomic overhead is much larger than the saved combine-reduce work.

3. Four-warp loader dequant:
- Prototype changed loader dequant from 2 idle non-epilogue warps doing two rows each to all 4 non-epilogue warps doing one row each.
- Correctness PASS for M512/M1024, weight_scale=0.05.
- 30-run benchmark was slower than current default:
  - M512 2445.4 us
  - M1024 3940.0 us
  - M2048 6831.0 us
  - M4096 12638.0 us
  - M8192 24100.0 us
- Decision: rejected and reverted. The extra dequant parallelism delays the TMA loader warps from issuing the next stage, so the block-level pipeline gets worse despite lower per-row work.

4. M256 metadata/arrival gates:
- `DG_SM90_NVFP4_DIRECT_SCATTER_METADATA_BCAST=1` at M256: 1517.6 us vs 1511.1 us with it OFF.
- `DG_SM90_NVFP4_L2_ARRIVAL_COUNTER=1` at M256: 1571.4 us vs 1531.6 us with it OFF.
- Decision: keep both defaults OFF for M256.

Retained change:
- Disable `DG_SM90_MOE_L2_DUAL_ACCUM` by default for M512.
- Previous default kept L2 dual accum ON for M512; same-input 50-run A/B showed OFF is faster:
  - M512 baseline with L1 dual-K OFF, L2 dual-accum ON: 2285.2 us
  - M512 L1 dual-K ON, L2 dual-accum ON: 2269.3 us
  - M512 L1 dual-K OFF, L2 dual-accum OFF: 2236.5 us
  - M512 L1 dual-K ON, L2 dual-accum OFF: 2262.7 us
- Retained default: L1 dual-K OFF, L2 dual-accum OFF for M512.

Correctness after retained change:
- Rebuild PASS.
- M512 correctness PASS with weight_scale 0.001/0.05/1.0, fp8-bridge compact reference, cosine/norm thresholds 0.99/0.99/0.99-1.01.
- Final smoke correctness PASS for M512/M1024/M2048/M4096/M8192 at weight_scale=0.05.
- Dequant unit test PASS and CUDA LUT dequant unit test PASS.

Final 50-run benchmark, same `512 1024 2048 4096 8192` sweep shape:

| tokens | compact NVFP4 | PR323 W8A8 reference | NVFP4/W8A8 |
|---:|---:|---:|---:|
| 512 | 2223.2 us | 2320.0 us | 0.958x, 4.2% faster |
| 1024 | 3780.0 us | 3518.0 us | 1.074x slower |
| 2048 | 6482.0 us | 6274.0 us | 1.033x slower |
| 4096 | 12080.0 us | 11402.0 us | 1.059x slower |
| 8192 | 23123.0 us | 21904.0 us | 1.056x slower |

Single-M M512 confirmation:
- M512-only 50-run after default change: 2231.7 us, confirming the M512 improvement on the same input distribution.

All-size 50-run note:
- Running `32 64 128 256 512 1024 2048 4096 8192` in one command changes the generated random routing/recv distribution for later sizes.
- That sweep produced M512=2392.3 us with recv=4087, while the comparable `512..8192` sweep produced M512=2223.2 us with recv=4063.
- Use same batch-list/input distribution when doing A/B comparisons.

Conclusion:
- This round retained one small default improvement: M512 L2 dual-accum OFF.
- M512 now beats the PR323 W8A8 reference by about 4% in the comparable 512..8192 sweep.
- M1024-8192 remain slower than PR323 W8A8 by about 3-7%.
- The rejected experiments reinforce that the current pipeline already depends on keeping TMA loader warps ahead; using those warps for more dequant or adding BF16 atomics hurts.

M1024 single-M confirmation after the retained M512 gate:
- M1024-only 50-run: 3730.0 us (recv=8015). This sits between the prior retained 3691.0 us and the same-shape sweep 3780.0 us, so the observed M1024 movement is treated as input/tail variance rather than a regression from the M512 default change.


Addendum: L2 direct-scatter metadata prefetch, rejected and reverted:
- Prototype added an opt-in `DG_SM90_NVFP4_DIRECT_SCATTER_METADATA_PREFETCH=1` gate.
- Design: preload/broadcast `TokenSrcMetadata` for the two rows during the final L2 WGMMA batch, between `warpgroup_commit_batch()` and `warpgroup_wait<0>()`, so the epilogue direct scatter could avoid metadata global loads on the scatter path.
- Covered both L2 dual-accum and L2 non-dual paths. The first insertion accidentally touched L1; that was fixed before measurement.
- Correctness PASS for M512/M1024, weight_scale=0.05, fp8-bridge compact reference.
- 50-run benchmark with opt-in:
  - M512-only: 2250.3 us vs default 2231.7 us, slower.
  - M1024-only: 5061.0 us vs default 3730.0 us, much slower.
- Decision: rejected and fully reverted. The added live metadata registers and instructions near WGMMA wait hurt scheduling/register pressure more than the epilogue metadata load saved.
- Rebuilt after revert and M512 correctness smoke PASSed with `DG_JIT_CACHE_DIR=/tmp/dg_jit_nvfp4_after_metaprefetch_revert_correct_0612`.

## 2026-06-12 Iteration - Packed direct-scatter metadata experiment (reverted)

Hypothesis: pack TokenSrcMetadata from three uint32_t fields into one uint32_t to reduce direct L2 scatter metadata load and row-group shuffle overhead.

Change tested:
- Temporarily changed layout::TokenSrcMetadata to a packed 32-bit representation.
- Updated SM90 NVFP4, SM90 FP8, and SM100 FP8/FP4 metadata writes/reads accordingly.

Validation:
- Rebuild PASS.
- NVFP4 strict correctness PASS for M512/M1024, weight_scale=0.05, compact weights, FP8-bridge reference, cosine/norm thresholds 0.99.

Benchmark signal, 50-run compact mode:
- M512 single-size: 2251.3 us vs retained same-shape baseline 2231.7 us, -0.9% regression.
- M1024 single-size: 3713.0 us vs retained same-shape baseline 3730.0 us, +0.5% improvement.
- M512..8192 list: 2223.4 / 3734.0 / 6474.0 / 12127.0 / 23108.0 us. Compared with retained same-list baseline 2223.2 / 3780.0 / 6482.0 / 12080.0 / 23123.0 us, only M1024 improved materially and M4096 regressed slightly.

Decision: reverted. The signal is too small and mixed, and the change touches shared MegaMoE metadata layout beyond the true NVFP4 kernel. Not suitable as a retained default optimization.

## 2026-06-12 Iteration - L2 N-major scheduler experiment (reverted)

Hypothesis: make the existing DG_SM90_MOE_L2_NMAJOR knob functional for NVFP4 by scheduling L2 blocks in N-major order within each expert. The goal was to improve B/scale locality and make next-tile load/dequant more stable while the previous tile epilogue/scatter runs.

Change tested:
- Added opt-in L1/L2 N-major scheduling parameters to MegaMoEScheduler.
- Wired SM90 NVFP4 to pass kL1NMajorScheduleRequested/kL2NMajorScheduleRequested.
- Kept the opt-in default off during measurement to avoid changing default behavior before A/B validation.

Validation:
- Rebuild PASS.
- NVFP4 strict correctness PASS for M512/M1024 with DG_SM90_MOE_L2_NMAJOR=1, compact weights, FP8-bridge reference, cosine/norm thresholds 0.99.

Benchmark signal, 50-run compact mode with DG_SM90_MOE_L2_NMAJOR=1:
- M512-only: 2242.7 us vs retained same-shape baseline 2231.7 us, -0.5% regression.
- M1024-only: 3727.0 us vs retained same-shape baseline 3730.0 us, roughly flat.
- M512..8192 list: 2327.8 / 3719.0 / 6657.0 / 12156.0 / 23185.0 us. Compared with retained same-list baseline 2223.2 / 3780.0 / 6482.0 / 12080.0 / 23123.0 us, M1024 improved but M512, M2048, M4096, and M8192 regressed.

Decision: reverted. N-major scheduling is not a reliable large-M win for this kernel; it likely hurts L1-output/L2-activation locality or CTA load balance more than it improves B reuse.

## 2026-06-12 Iteration - Fused L1/L2 pipeline check (rejected)

Hypothesis: disabling split L1/L2 launches with DG_SM90_MOE_SPLIT_L1_L2=0 might expose cross-phase overlap between L1 epilogue/store and L2 load/dequant, reducing fixed pipeline cost.

Validation:
- NVFP4 strict correctness PASS for M512/M1024/M2048, weight_scale=0.05, compact weights, FP8-bridge reference, cosine/norm thresholds 0.99.

Benchmark signal, 50-run compact mode with DG_SM90_MOE_SPLIT_L1_L2=0:
- M512..8192 list: 3201.0 / 5367.0 / 9434.0 / 18073.0 / 34998.0 us.
- This is far slower than retained split default 2223.2 / 3780.0 / 6482.0 / 12080.0 / 23123.0 us.

Decision: rejected. For the current BM128 NVFP4 shape, split L1/L2 remains required; fused mode loses too much occupancy/resource balance to pay for any cross-phase overlap.

## 2026-06-12 Iteration - Stage and L2 dual-accum sweeps (not retained)

Stage sweep:
- M512 correctness PASS for DG_SM90_NVFP4_NUM_STAGES=2/3/4/5. Stage 6 exceeds max_num_stages.
- 30-run M512/M1024 signal:
  - stages=2: 2562.1 / 4228.0 us, clearly worse.
  - stages=3: 2240.1 / 3793.0 us.
  - stages=4: 2241.1 / 3716.0 us, best M1024 signal but M512 not better.
  - stages=5: 2260.5 / 3751.0 us.
- stages=4 smoke correctness PASS for M512/M1024/M2048/M4096/M8192.
- stages=4 final 50-run M512..8192: 2241.8 / 3744.0 / 6482.0 / 12090.0 / 23138.0 us vs retained 2223.2 / 3780.0 / 6482.0 / 12080.0 / 23123.0 us.
- stages=4 M1024-only 50-run: 3724.0 us vs retained M1024-only 3730.0 us.
Decision: not retained. The M1024 gain is small and M512/large-M do not improve.

L2 dual-accum OFF sweep:
- Correctness PASS for M1024/M2048 with DG_SM90_MOE_L2_DUAL_ACCUM=0.
- 50-run M512..8192 with OFF: 2254.7 / 3781.0 / 6458.0 / 12075.0 / 23164.0 us. Only M2048 showed a small signal.
- M2048-only default vs OFF: 6572.0 vs 6560.0 us, about 0.2% improvement.
Decision: not retained for M1024/M2048; improvement is too small/noisy. Keep existing default gates.

## 2026-06-12 Final default check after reverted pipeline probes

Default correctness smoke:
- Rebuild after reverted packed-metadata changes PASSed.
- Default M512/M1024 correctness PASS with weight_scale=0.05, compact weights, FP8-bridge reference, cosine/norm thresholds 0.99.

Default 50-run benchmark, compact mode, batches 512 1024 2048 4096 8192:

| tokens | latest default NVFP4 | PR323 W8A8 reference | NVFP4/W8A8 |
|---:|---:|---:|---:|
| 512 | 2256.7 us | 2320.0 us | 0.973x, 2.7% faster |
| 1024 | 3816.0 us | 3518.0 us | 1.085x slower |
| 2048 | 6426.0 us | 6274.0 us | 1.024x slower |
| 4096 | 12117.0 us | 11402.0 us | 1.063x slower |
| 8192 | 23140.0 us | 21904.0 us | 1.056x slower |

Interpretation:
- The current retained code still beats PR323 W8A8 at M512 in this run.
- M1024+ remain slower than W8A8; the largest remaining gap in this run is M1024.
- The latest failed probes were fully reverted; retained code path is unchanged except for the already documented defaults from earlier retained iterations.


## 2026-06-12 Iteration - Direct L2 scatter store-width probes (reverted)

Context:
- Follow-up to the block-level pipeline analysis: FC1/FC2 loader dequant already overlaps with MMA, and current loader warps can also start loading/dequanting later tiles once math releases the stage empty barriers before epilogue.
- A remaining plausible fixed cost was L2 direct scatter, where the default metadata-broadcast path still issues many per-lane 32-bit stores from each 4-lane row group.

Phase profile refresh, current default, 5-run compact mode:

| tokens | compact | dispatch_total avg cycles | dispatch_pull avg | math_loop avg | combine_barrier avg | combine_reduce avg | gemm_core avg | l1_epi avg | l2_epi avg | loader_dequant avg | math_dequant_wait avg |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 512 | 2498.0 us | 323637 | 105311 | 1206851 | 134850 | 13154 | 38795 | 2694 | 1366 | 1206 | 125 |
| 1024 | 4068.0 us | 255793 | 206134 | 1981364 | 26574 | 27936 | 39102 | 2794 | 1493 | 1209 | 133 |
| 2048 | 7080.0 us | 783234 | 405423 | 3602164 | 53012 | 60532 | 38962 | 2891 | 1599 | 1209 | 129 |
| 4096 | 13178.0 us | 895526 | 799750 | 6389484 | 177091 | 125732 | 38545 | 2937 | 1661 | 1210 | 115 |

Interpretation:
- `math_dequant_wait` is still tiny compared with `math_loop`, confirming dequant/MMA overlap is not the immediate limiter.
- `l2_epilogue` is small per block, but it is the only part of L2 scatter directly under this kernel's control without changing global scheduling.
- `dispatch_pull` and combine synchronization are significant fixed/cross-rank costs, but they are mostly shared with W8A8 and not specific to NVFP4 dequant.

Experiment 1: 4-lane row-group vector store
- Change: in the direct-scatter metadata-broadcast branch, each 4-lane row group shuffled BF16 pairs to `col_idx==0`, then the leader issued 16-byte `uint4` stores. This also reduced `sym_buffer.map()` to leader lanes.
- Correctness: PASS for M512/M1024/M2048, compact weights, `weight_scale=0.05`, FP8-bridge reference, strict cosine/norm thresholds.
- 50-run benchmark:
  - M512 5692.0 us
  - M1024 9754.0 us
  - M2048 17312.0 us
  - M4096 32580.0 us
  - M8192 63026.0 us
- Decision: reverted. Store instruction count dropped, but scatter parallelism collapsed; the row-group shuffles and leader-only writes made performance much worse.

Experiment 2: 2-lane pair `uint2` store
- Change: a more conservative variant where lanes 0/1 and 2/3 of each row group produced 8-byte `uint2` stores, preserving half the store-lane parallelism.
- Correctness: PASS for M512/M1024, compact weights, `weight_scale=0.05`, FP8-bridge reference, strict cosine/norm thresholds.
- 30-run benchmark:
  - M512 3774.0 us
  - M1024 6415.0 us
  - M2048 11469.0 us
  - M4096 21179.0 us
  - M8192 41018.0 us
- Decision: reverted. Even halving the number of active store lanes was too costly.

Conclusion:
- The default direct L2 scatter should keep the current scalar per-lane stores. It is lane-parallelism limited rather than store-instruction-count limited.
- The remaining NVFP4 vs PR323 W8A8 gap is unlikely to close through L2 scatter store-width tuning.
- A credible next direction must either reduce online FP4->FP8 materialization cost without reducing loader overlap, or change the global scheduling/dispatch/combine lifecycle. Simple scatter vectorization is now ruled out.


## 2026-06-12 Iteration - Async L1 TMA store for BM128 two-WG path (reverted)

Hypothesis:
- The existing async L1 TMA store code path only instantiated when `kNumEpilogueWarpgroups == 1`, so the main BM128/BN128/256-thread path for M512+ could not use it.
- Allowing async L1 store for the two-WG BM128 path might explicitly overlap L1 epilogue/TMA-store with the next tile's load/dequant, matching the block-level pipeline direction.

Temporary change:
- Removed the one-epilogue-WG restriction from `kAsyncL1TMAStore`.
- Re-enabled `DG_SM90_MOE_ASYNC_L1_STORE=1` in the NVFP4 host launcher as an opt-in path.
- Added host shared-memory accounting for the double-buffered async L1 output area.
- Forced the L2-only no-dispatch phase to instantiate with async L1 store disabled.
- Rebuilt `deep_gemm._C` before testing.

Validation result:
- Timeout-guarded correctness command:
  `timeout 240s env DG_SM90_MOE_ASYNC_L1_STORE=1 ... --batches 512 1024 ...`
- The command reached the first M512 case header, then timed out at 240s before producing kernel output.
- `ps` after timeout showed no residual `test_nvfp4_mega_moe_sm90_correctness.py` process.

Decision:
- Reverted the async 2WG host/kernel changes and rebuilt `deep_gemm._C` after revert.
- Default M512 strict correctness after revert PASSed (`cosine_min=0.9996`, `norm_ratio=0.9995`).
- Conclusion: the current async L1 store lifecycle is not safe for BM128/two-WG. The existing async branch cannot simply be generalized; it needs a redesigned per-WG drain/arrival protocol before it can be a viable block-level pipeline mechanism.


## 2026-06-12 Iteration - L2 no-dispatch cleanup on loader warp (reverted)

Hypothesis:
- In the retained split path, the L2 no-dispatch kernel has no dispatch warps.
- Workspace cleanup is currently performed by epilogue threads after combine reduce through `finish_no_dispatch_k2_cleanup()`.
- The non-epilogue loader/dequant warps are idle after all L2 tiles are loaded/dequantized. Moving workspace cleanup to the A-loader warp could overlap cleanup with epilogue combine and reduce fixed tail cost.

Temporary change:
- Added a compile-time `kNoDispatchCleanupOnLoader` path for `!kRunL1Phase && kRunL2Phase && kNumDispatchWarps == 0`.
- After the A-loader warp completed `for_each_selected_block`, it cleaned expert send/recv counts and L1/L2 arrival state using 32 lanes, then ran the after-clean NVLink barrier.
- The epilogue no-dispatch cleanup path was skipped under that condition.

Validation result:
- Timeout-guarded correctness command with pipefail:
  `timeout 240s env DG_JIT_CACHE_DIR=/tmp/dg_jit_nvfp4_loadercleanup_correct_0612 ... --batches 512 1024 ...`
- The command reached the first M512 case header, then timed out at 240s before kernel output.
- `ps` after timeout showed no residual correctness process.

Decision:
- Reverted the loader-cleanup overlap code.
- Default M512 strict correctness after revert PASSed (`cosine_min=0.9996`, `norm_ratio=0.9995`).
- Conclusion: cleanup cannot be moved ahead of epilogue combine with the current NVLink barrier/grid-sync lifecycle. The after-clean barrier likely conflicts with the concurrent combine barrier or rank progress ordering. This direction needs a more explicit multi-phase protocol, not a simple warp handoff.


## 2026-06-12 Retained NVFP4 benchmark refresh and W8A8 rerun blocker

Purpose:
- After reverting several failed pipeline/scatter experiments, refresh the retained true compact NVFP4 benchmark in the current source tree.
- Try to rerun the PR323/805 W8A8 baseline in the same window before choosing the next tuning target.

Current retained NVFP4, 50-run compact mode:

| tokens | recv | NVFP4 compact |
|---:|---:|---:|
| 512 | 4063 | 2232.7 us |
| 1024 | 8268 | 3743.0 us |
| 2048 | 16504 | 6474.0 us |
| 4096 | 32619 | 12039.0 us |
| 8192 | 65159 | 23146.0 us |

Comparison against the retained PR323 W8A8 reference table:

| tokens | NVFP4 refresh | PR323 W8A8 reference | NVFP4/W8A8 |
|---:|---:|---:|---:|
| 512 | 2232.7 us | 2320.0 us | 0.962x, 3.8% faster |
| 1024 | 3743.0 us | 3518.0 us | 1.064x slower |
| 2048 | 6474.0 us | 6274.0 us | 1.032x slower |
| 4096 | 12039.0 us | 11402.0 us | 1.056x slower |
| 8192 | 23146.0 us | 21904.0 us | 1.057x slower |

W8A8 same-window rerun attempt:
- `/root/fac/megamoe/deepgemm_pr323_w8a8` could not import `deep_gemm._C`.
- Building it failed with missing `cute/arch/mma_sm100_desc.hpp`.
- `/root/fac/megamoe/deepgemm_805_w8a8` is at `805f7e8 Restore SM90 block64 default heuristic`, but it also lacked `deep_gemm._C` and failed to build with the same missing CUTLASS header.
- Therefore no fresh same-window W8A8 number was produced in this pass; keep using the prior verified PR323 W8A8 reference table unless a buildable W8A8 checkout is restored.

Implication:
- Current true compact NVFP4 now clearly beats W8A8 at M512, but still trails at M1024/M2048/M4096/M8192.
- The largest refreshed relative gap is M1024, followed by M4096/M8192. Any next retained optimization should be size-gated and validated against these refreshed numbers.


## 2026-06-12 Feasibility note - fused packed-B plus scale layout

Motivation:
- The current true compact NVFP4 path TMA-loads packed B (64 bytes per row for BK128) and separately TMA-loads UE4M3 scales (8 bytes per row) into `smem_sfb`.
- Loader-dequant then reads both and materializes FP8 B in shared memory for WGMMA.
- `math_dequant_wait` is already low, but the path still pays an extra SFB shared-memory allocation, a separate scale TMA, and an extra barrier transaction per stage.

Candidate structural layout:
- At weight-load transform time, store each BK128 row as `64B packed FP4 + 8B UE4M3 scale`, i.e. 72 bytes per row.
- TMA-load one fused B tile of `BN * 72` bytes into the existing `smem_b` staging area.
- Loader-dequant reads packed FP4 from `row * 72 + 0` and scale from `row * 72 + 64`, then writes the normal FP8 output to `row * 128` after all rows have first loaded their FP4/scale registers.
- This preserves true compact NVFP4 input semantics; it does not use the FP8 shadow path.

Expected upside:
- Removes the separate SFB TMA load and `smem_sfb` stage allocation for the loader-dequant path.
- Keeps scale bytes co-located with packed FP4 rows, which may reduce barrier bookkeeping and improve locality.
- Does not reduce store-lane parallelism in direct L2 scatter and does not weaken existing epilogue barriers.

Required code changes:
- Python transform: add a fused layout helper in `deep_gemm/mega/__init__.py` or `deep_gemm/quantization_nvfp4.py` that packs `(E, N, K/2)` and tile-major scales into a 3D `(E, N, K/128 * 72)` tensor.
- API/host checks: `csrc/apis/sm90_mega.hpp` must derive logical K from `storage_k / 72 * 128` when the fused layout is enabled, while still using `l*_weights_sf` for block_n/layout validation or replacing that validation with a fused-layout marker.
- Host launcher: `csrc/jit_kernels/impls/sm90_nvfp4_mega_moe.hpp` must build B tensor maps with `BLOCK_K_STORAGE=72` instead of `BLOCK_K/2=64`, and shared-memory byte accounting must drop `SMEM_SFB_SIZE_PER_STAGE` for this path.
- Kernel: `sm90_nvfp4_mega_moe.cuh` needs a template/runtime flag so loader-dequant uses row stride 72 and scale offset 64. The TMA transaction byte count must be `BN * 72`, not `SMEM_B_SIZE_PER_STAGE / 2`.
- Tests: correctness must cover dequant-level row stride, then M512/M1024 strict E2E correctness, then 50-run benchmark.

Risks:
- TMA box width 72 must be accepted by the existing descriptor helper and hardware constraints.
- In-place dequant is only safe if every dequant thread loads both packed FP4 and scale into registers before any thread writes FP8 to `row * 128`; the current two-row helper already has this barrier shape, but a new stride-aware helper is required.
- This is an ABI/layout change, so it should be opt-in until correctness and benchmark prove it.

Decision for this pass:
- Not implemented as a quick patch because it touches Python transform, C++ API shape inference, host tensor maps, kernel dequant, and tests.
- This remains the most concrete untried structural direction for reducing online FP4->FP8 materialization overhead without reverting to FP8 shadow weights.


## 2026-06-12 Iteration - Fused packed-B plus scale default (retained)

Context:
- The phase profile before this iteration showed FC1/FC2 dequant/MMA overlap was already effective: `math_dequant_wait` was tiny compared with `math_loop`.
- The remaining NVFP4-specific fixed cost was the B-side staging path: packed B TMA plus a separate SFB 1D TMA and per-stage SFB shared-memory allocation.

Retained change:
- Added a fused NVFP4 B layout where each BK128 row is stored as `64B packed FP4 + 8B UE4M3 scale + 8B padding`.
- The physical weight tensor remains compact NVFP4 input, shaped as `(E, N, K/128*80)`; it is not an FP8 shadow cache.
- B TMA now loads one 80B row tile into the normal `smem_b` staging area. Loader-dequant reads packed FP4 and scale from the same row, then writes the usual FP8 WGMMA-ready row layout into `smem_b`.
- The separate SFB TMA and per-stage `smem_sfb` allocation are removed for the default fused layout.
- `DG_SM90_NVFP4_FUSED_B_SCALE=0` remains an opt-out fallback to the previous separate packed-B + SFB layout.

Validation:
- 72B row layout was attempted first and failed at tensor-map creation with `CUDA_ERROR_INVALID_VALUE`; TMA requires a legal inner tile width, so this was changed to 80B rows.
- 80B fused layout unit check PASS: fused tensor dequant is bit-exact with the original packed+tile-major-scale reference.
- Default strict correctness PASS after making fused layout default:
  - M512/M1024, weight_scale=0.05, cosine_min=0.9996, norm_ratio around 0.9994-0.9995.
- Broad correctness PASS:
  - dequant unit test PASS.
  - CUDA LUT dequant unit test PASS.
  - M32/M256/M512 with weight_scale=0.001/0.05/1.0 PASS.
  - Worst observed cosine_min=0.9990; norm_ratio range stayed within [0.9992, 1.0059].

Phase-profile signal, fused80 default, 5-run compact mode:

| tokens | compact | math_loop avg cycles | loader_dequant avg | math_dequant_wait avg |
|---:|---:|---:|---:|---:|
| 512 | 2328.8 us | 1054117 | 1096 | 67 |
| 1024 | 3887.0 us | 1860576 | 1099 | 63 |
| 2048 | 6625.0 us | 3215234 | 1101 | 60 |
| 4096 | 12199.0 us | 5894984 | 1096 | 55 |

This improves the prior retained profile where `loader_dequant` was about 1206 cycles and `math_dequant_wait` about 115-133 cycles.

Final default 50-run benchmark, compact true-NVFP4 input:

| tokens | previous retained NVFP4 | final default NVFP4 | delta vs retained | PR323 W8A8 ref | final NVFP4/W8A8 |
|---:|---:|---:|---:|---:|---:|
| 512 | 2232.7 us | 2214.1 us | 0.8% faster | 2320.0 us | 0.954x, 4.6% faster |
| 1024 | 3743.0 us | 3658.0 us | 2.3% faster | 3518.0 us | 1.040x slower |
| 2048 | 6474.0 us | 6329.0 us | 2.2% faster | 6274.0 us | 1.009x slower |
| 4096 | 12039.0 us | 11906.0 us | 1.1% faster | 11402.0 us | 1.044x slower |
| 8192 | 23146.0 us | 22805.0 us | 1.5% faster | 21904.0 us | 1.041x slower |

Best single fused80 run observed before the final full-sweep rerun:
- M512/M1024/M2048/M4096/M8192 = 2200.7 / 3651.0 / 6315.0 / 11885.0 / 22777.0 us.
- This run showed the same directional gain but final reporting keeps the later full-sweep default run above.

Tried and rejected in this pass:
- Loader global-scale dequant (`DG_SM90_NVFP4_LOADER_SCALE_GMEM=1` prototype): removed. Recomputing scale pointers in idle dequant warps failed correctness (`cosine_min=0.8609`); a shared-pointer handoff variant produced NaNs. This path was fully removed before final validation.
- M4096 `L2_DUAL_ACCUM=1` default: 30/targeted 50-run had a positive signal, but the full final sweep regressed M4096 to 12040 us, so the default was reverted.
- `DG_SM90_NVFP4_DIRECT_SCATTER_METADATA_BCAST=0`: improved rank0 M1024 in one sweep but worsened mean/max or larger sizes; not retained.
- `DG_SM90_MOE_L2_NMAJOR=0`: no stable win in 50-run recheck; not retained.
- `DG_SM90_MOE_L1_DUAL_K=1`: clear regression for M1024+; not retained.

Decision:
- Keep fused packed-B+scale 80B row layout as the default true compact NVFP4 path.
- The target of beating W8A8 is achieved at M512 and nearly reached at M2048, but M1024/M4096/M8192 still trail PR323 W8A8 by about 4%.
- The next structural direction should not be more dequant/MMA overlap; the fused layout already reduced that cost. Remaining gap is likely in global scheduling / dispatch-combine lifecycle / WGMMA issue efficiency rather than SFB staging alone.


## 2026-06-12 Follow-up - Async L1 store rejected, 20-run sweep with M64/M128/M256

Context:
- The next structural hypothesis was to overlap L1 epilogue/TMA store with following-tile load/dequant for the BM128/two-epilogue-WG path.
- A naive extension of `DG_SM90_MOE_ASYNC_L1_STORE` to two epilogue warpgroups, including a `tma_store_wait<0>` reuse guard, did not pass validation.

Rejected attempt:
- `DG_SM90_MOE_ASYNC_L1_STORE=1` with M512/M1024 strict correctness timed out after the dequant unit tests and before completing M512 E2E correctness.
- This opt-in path was reverted/disabled for NVFP4: host now asserts if `DG_SM90_MOE_ASYNC_L1_STORE` is requested, and the kernel-side async condition is back to the original single-epilogue-WG shape.
- No benchmark result from the async attempt is retained.

Post-revert validation:
- Rebuild PASS: `touch csrc/python_api.cpp && python setup.py build_ext --inplace -j 16`.
- Strict compact correctness PASS after async disable:
  - M512/M1024, weight_scale=0.05, fp8-bridge reference, compact weight mode.
  - M512 cosine_min=0.9996, norm_ratio=0.9995.
  - M1024 cosine_min=0.9996, norm_ratio=0.9994.

Post-revert 20-run benchmark with small sizes included:

Command:
```bash
MASTER_ADDR=127.0.0.1 MASTER_PORT=29663 \
DG_JIT_CACHE_DIR=/tmp/dg_jit_nvfp4_post_async_revert_bench20_0612 \
python3 -u tests/bench_nvfp4_mega_moe_sm90.py \
  --num-processes 8 \
  --batches 64 128 256 512 1024 2048 4096 8192 \
  --num-tests 20 \
  --weight-mode compact \
  2>&1 | tee /tmp/nvfp4_post_async_revert_bench20.log
```

| tokens | recv | compact NVFP4 20-run | mean_rank | max_rank | PR323 W8A8 ref | compact/W8A8 |
|---:|---:|---:|---:|---:|---:|---:|
| 64 | 517 | 1167.1 us | 1152.5 us | 1167.1 us | 869.9 us | 1.342x, 34.2% slower |
| 128 | 1013 | 1340.3 us | 1307.1 us | 1340.3 us | 837.1 us | 1.601x, 60.1% slower |
| 256 | 1990 | 1492.7 us | 1501.8 us | 1509.6 us | 1460.0 us | 1.022x, 2.2% slower |
| 512 | 4131 | 2340.2 us | 2337.6 us | 2345.7 us | 2320.0 us | 1.009x, 0.9% slower |
| 1024 | 8052 | 3657.0 us | 3697.9 us | 3714.0 us | 3518.0 us | 1.040x, 4.0% slower |
| 2048 | 16403 | 6428.0 us | 6432.1 us | 6450.0 us | 6274.0 us | 1.025x, 2.5% slower |
| 4096 | 32681 | 11920.0 us | 11907.3 us | 11922.0 us | 11402.0 us | 1.045x, 4.5% slower |
| 8192 | 65556 | 22722.0 us | 22738.4 us | 22762.0 us | 21904.0 us | 1.037x, 3.7% slower |

Interpretation:
- The 20-run sweep with M64/M128/M256 shows the fixed-overhead regime clearly: M64/M128 remain far behind W8A8, while M256 and above are close but still mostly slower in this run.
- This 20-run M512 did not reproduce the earlier retained 50-run best of 2214.1 us; keep the 50-run table as the best verified retained result and this table as the latest current-code 20-run including small sizes.
- The async L1 store direction is not retained. A correct version would need a redesigned per-WG/per-stage store completion protocol rather than a simple CTA-global TMA wait change.


## 2026-06-12 Iteration - Enable L2 no-dispatch + arrival counter for M128 (retained)

Context:
- The post-async-revert 20-run sweep added M64/M128/M256 and showed M128 was still in a fixed-overhead regime: 1340.3 us versus the PR323 W8A8 reference of 837.1 us.
- Phase profile for M64/M128 showed `math_dequant_wait` was not the primary bottleneck. The heavier fixed costs were dispatch/combine lifecycle and a math loop floor that barely changed from M64 to M128.

Env sweep:
- `DG_SM90_MOE_L2_NO_DISPATCH_PIPELINE=1` on M64/M128:
  - M64: 1222.4 us, worse than the default 1127-1167 us range.
  - M128: 1136.7 us, a strong improvement over the post-revert default 1340.3 us.
- `DG_SM90_MOE_L2_NO_DISPATCH_PIPELINE=1 DG_SM90_NVFP4_L2_ARRIVAL_COUNTER=1`:
  - M64: 1275.1 us, worse.
  - M128: 1132.8 us, slightly better than no-dispatch alone.
- Adding `DG_SM90_NVFP4_DIRECT_SCATTER_METADATA_BCAST=1` produced lower rank0 M128 (1102.4 us) but worse mean/max ranks (mean 1161.1 us, max 1173.1 us), so it was not retained.

Retained change:
- Added `num_tokens == 128` to the default L2 no-dispatch pipeline set.
- Added `num_tokens == 128` to the default L2 arrival-counter set.
- M64 is intentionally not included because both no-dispatch variants regressed it.

Correctness:
- Strict M64/M128/M256 correctness PASS after changing the default gate:
  - dequant unit test PASS.
  - CUDA LUT dequant unit test PASS.
  - M64/M128/M256, weight_scale=0.05, cosine_min=0.9996.
- Broad M128 correctness PASS:
  - weight_scale=0.001: cosine_min=0.9996, norm_ratio=0.9994.
  - weight_scale=0.05: cosine_min=0.9996, norm_ratio=0.9997.
  - weight_scale=1.0: cosine_min=0.9993, norm_ratio=0.9979.

Default 20-run benchmark after retaining the M128 gate:

| tokens | recv | compact NVFP4 20-run | mean_rank | max_rank | PR323 W8A8 ref | compact/W8A8 |
|---:|---:|---:|---:|---:|---:|---:|
| 64 | 517 | 1127.3 us | 1136.7 us | 1149.0 us | 869.9 us | 1.296x, 29.6% slower |
| 128 | 1013 | 1121.3 us | 1116.4 us | 1123.0 us | 837.1 us | 1.340x, 34.0% slower |
| 256 | 1990 | 1511.8 us | 1506.3 us | 1513.2 us | 1460.0 us | 1.035x, 3.5% slower |
| 512 | 4131 | 2355.5 us | 2347.1 us | 2358.5 us | 2320.0 us | 1.015x, 1.5% slower |
| 1024 | 8052 | 3673.0 us | 3664.8 us | 3673.0 us | 3518.0 us | 1.044x, 4.4% slower |
| 2048 | 16403 | 6390.0 us | 6385.4 us | 6395.0 us | 6274.0 us | 1.019x, 1.9% slower |
| 4096 | 32681 | 11928.0 us | 11938.0 us | 11964.0 us | 11402.0 us | 1.046x, 4.6% slower |
| 8192 | 65556 | 22705.0 us | 22700.1 us | 22709.0 us | 21904.0 us | 1.037x, 3.7% slower |

Effect versus post-async-revert 20-run:
- M128 improved from 1340.3 us to 1121.3 us, about 16.3% faster.
- M64 improved from 1167.1 us to 1127.3 us in this run, but the retained code does not deliberately change M64; treat this as run-to-run variation unless reproduced.
- M256/M512 changed within about 1-2%, so the new default is scoped enough to keep larger sizes stable.

Decision:
- Keep the M128 L2 no-dispatch + arrival counter default.
- This is a real small-M win, but it does not close the W8A8 gap: M64/M128 are still about 30-34% slower than PR323 W8A8, and M256+ remain close but mostly behind in the 20-run sweep.


## 2026-06-12 Profile Analysis - Tile-to-tile overlap status

Current status:
- FC1/FC2 B-side NVFP4 dequant is already overlapped with WGMMA well enough that it is no longer the main exposed stall.
- The current default path still does not have a stable explicit block-level pipeline that overlaps L1 epilogue/TMA store or L2 scatter with the next tile's load/dequant.

Evidence from existing phase profile (`/tmp/nvfp4_post_async_revert_phase_profile.log`):

| tokens | compact | math_loop avg | gemm_core avg | loader_dequant avg | math_dequant_wait avg | l1_epilogue avg | l2_epilogue avg | combine_barrier avg |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 64 | 1408.1 us | 563464 | 29154 | 877 | 279 | 2355 | 727 | 11679 |
| 128 | 1299.2 us | 567079 | 29299 | 883 | 282 | 2335 | 809 | 10685 |
| 256 | 1668.8 us | 752896 | 38656 | 1222 | 73 | 3769 | 1151 | 14653 |
| 512 | 2359.2 us | 1049623 | 35787 | 1095 | 68 | 2633 | 1578 | 209454 |
| 1024 | 3776.0 us | 1718859 | 35995 | 1102 | 64 | 2772 | 1728 | 290576 |

Interpretation:
- `math_dequant_wait` is tiny for M256+ (64-73 cycles average) compared with `gemm_core` (~36k cycles) and `math_loop` (0.75M-1.72M cycles). This means loader-dequant is mostly hidden behind WGMMA/math work.
- M64/M128 still show only ~280 cycles `math_dequant_wait`, which is also small relative to the ~563k cycle math loop floor. Their gap to W8A8 is not primarily dequant wait.
- M512/M1024 show large `combine_barrier` averages (209k/291k cycles). This points to phase/wave synchronization and combine lifecycle as a larger exposed fixed cost than dequant.
- L1/L2 epilogue averages are small per event, but they are serialized at the block boundary; the current profile does not expose a direct overlap percentage between epilogue/scatter and the next tile load/dequant.

Code-path evidence:
- `DG_SM90_MOE_ASYNC_L1_STORE` is currently disabled for NVFP4 by host assertion after the two-epilogue-WG experiment timed out.
- In the non-async L1 epilogue path, the kernel issues TMA store, then executes `ptx::tma_store_wait<0>()`, and only then calls `notify_l1_ready(...)`.
- The default split L1/L2 path launches L1 and L2 as separate kernels, so L2 does not overlap with L1 at the kernel/phase level.
- Therefore, any store/load overlap today is incidental hardware overlap across independent CTAs/waves, not a deliberate per-block tile pipeline.

Recent attempt:
- Extending async L1 TMA store to two epilogue warpgroups was tested with a `tma_store_wait<0>` reuse guard, but M512/M1024 correctness timed out before completing E2E validation.
- That path was reverted/disabled. It should not be counted as a working overlap pipeline.

Conclusion:
- Tile-level dequant/MMA overlap: good enough; not the active bottleneck.
- Epilogue/scatter versus next tile load/dequant overlap: not solved. The implementation still waits at the L1 store boundary and uses split L1/L2 phase boundaries, so the remaining optimization should target a correct async store/readiness protocol or a different scheduler shape that can overlap L1 output publication and L2 consumption without deadlock.

Next instrumentation gap:
- Existing phase profile is useful but insufficient to quantify overlap directly. A better profile needs separate markers for:
  - L1 epilogue compute before TMA store issue.
  - L1 TMA store issue-to-wait lifetime.
  - L1 ready notification latency.
  - L2 wait-for-ready time.
  - First TMA load/dequant of the next L2 tile after readiness.
- This would make the actual missed overlap visible instead of inferring it from barriers and code structure.


## 2026-06-12 Instrumentation - Direct tile-overlap phase metrics

Change:
- Added opt-in phase-profile-only metrics to expose the missing tile-overlap pieces more directly:
  - `l1_tma_wait`: time spent waiting for L1 TMA store completion before publishing L1 output readiness.
  - `l1_ready_notify`: time spent publishing L1-ready state to the L2 arrival mask/counter.
  - `l2_ready_wait`: time spent waiting for L1 output readiness before L2 starts loading/processing a tile.
  - `l2_scatter`: L2 direct or SMEM-mediated scatter section inside the L2 epilogue.
- Increased the benchmark phase-profile scratch allocation from 64 to 96 `int32` slots and updated the C++ API shape check accordingly.

Files touched for instrumentation:
- `deep_gemm/include/deep_gemm/impls/sm90_nvfp4_mega_moe.cuh`
- `tests/bench_nvfp4_mega_moe_sm90.py`
- `csrc/apis/sm90_mega.hpp`

Validation status:
- `python setup.py build_ext --inplace -j 16` PASS after the instrumentation patch.
- Runtime correctness/profile collection is pending because another 8-rank benchmark is currently occupying the GPUs in `/root/fac/megamoe/DeepGEMM_megamoe_nvfp4`:
  - `DG_SM90_NVFP4_DISPATCH_THREADS=64 python3 tests/bench_nvfp4_mega_moe_sm90.py --num-processes 8 --batches 4096 --num-tests 30`
  - Log `/tmp/nvfp4_fused_dispatch64_M4096.log` has not progressed past the banner.
- This external process was not killed, per the no-pkill/no-interference rule.

Next validation command once GPUs are free:
```bash
MASTER_ADDR=127.0.0.1 MASTER_PORT=29701 \
DG_SM90_MOE_PHASE_PROFILE=1 \
DG_JIT_CACHE_DIR=/tmp/dg_jit_nvfp4_overlap_profile_0612 \
python3 -u tests/bench_nvfp4_mega_moe_sm90.py \
  --num-processes 8 \
  --batches 64 128 256 512 1024 2048 \
  --num-tests 5 \
  --weight-mode compact \
  2>&1 | tee /tmp/nvfp4_overlap_phase_profile.log
```

Expected use of new metrics:
- If `l1_tma_wait` is large while `l2_ready_wait` is also large, L2 is directly exposed to L1 store publication latency.
- If `l1_tma_wait` is large but `l2_ready_wait` is small, the scheduler/waves are hiding L1 publication latency well enough.
- If `l2_scatter` is large and `combine_barrier` remains large, the bottleneck is likely L2 scatter/combine lifecycle rather than L1 store readiness.


## 2026-06-12 Profile Results - Direct tile-overlap metrics

Command:
```bash
MASTER_ADDR=127.0.0.1 MASTER_PORT=29701 \
DG_SM90_MOE_PHASE_PROFILE=1 \
DG_JIT_CACHE_DIR=/tmp/dg_jit_nvfp4_overlap_profile_0612 \
python3 -u tests/bench_nvfp4_mega_moe_sm90.py \
  --num-processes 8 \
  --batches 64 128 256 512 1024 2048 \
  --num-tests 5 \
  --weight-mode compact \
  2>&1 | tee /tmp/nvfp4_overlap_phase_profile.log
```

Results with the new overlap metrics:

| tokens | compact | math_loop | combine_barrier | loader_dequant | math_dequant_wait | l1_tma_wait | l1_ready_notify | l2_ready_wait | l2_scatter |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 64 | 1271.5 us | 564334 | 9561 | 882 | 276 | 72 | 438 | 235 | 743 |
| 128 | 1258.3 us | 568509 | 9386 | 902 | 308 | 74 | 421 | 233 | 829 |
| 256 | 1679.8 us | 752570 | 10043 | 1222 | 74 | 70 | 525 | 255 | 1203 |
| 512 | 3152.0 us | 1264692 | 211946 | 1098 | 65 | 72 | 513 | 233 | 1610 |
| 1024 | 3905.0 us | 1749112 | 299594 | 1101 | 63 | 72 | 510 | 252 | 1699 |
| 2048 | 6726.0 us | 3162050 | 253524 | 1101 | 59 | 73 | 548 | 236 | 1802 |

Interpretation:
- `l1_tma_wait` is only about 70-74 cycles across all profiled sizes. The synchronous `tma_store_wait<0>` after L1 output TMA store is not a large exposed stall.
- `l2_ready_wait` is only about 233-255 cycles. L2 is not spending significant time waiting for L1 output readiness.
- `l1_ready_notify` is about 420-550 cycles, larger than the store wait but still small compared with math loop and combine waits.
- `l2_scatter` accounts for most of `l2_epilogue`, but is still only about 0.7-1.8k cycles per event. It is worth optimizing locally, but not enough by itself to explain the W8A8 gap.
- `math_dequant_wait` remains small, confirming again that B-side NVFP4 dequant/MMA overlap is not the active bottleneck.
- M512/M1024/M2048 still show much larger exposed fixed costs in `combine_barrier` and `dispatch_pull/dispatch_total` than in L1 store readiness or L2 ready wait.

Conclusion:
- The previous hypothesis that the missing gain mainly comes from L1 epilogue/TMA store not overlapping next-tile load/dequant is not supported by the new metrics.
- Async L1 TMA store is unlikely to be the next high-value direction: it previously timed out, and the measured synchronous wait is only ~70 cycles.
- The next structural focus should shift toward L2 scatter/combine lifecycle and scheduler/wave behavior, especially the large `combine_barrier` and `dispatch_pull` costs at M512+.


## 2026-06-12 Experiment - Reduce SM count for combine/tail (rejected)

Hypothesis:
- The new overlap profile showed `combine_barrier` and `dispatch_pull/dispatch_total` are much larger exposed costs than L1 TMA wait or L2 ready wait.
- Try reducing participating SMs to see whether less wave/tail pressure lowers global barrier latency.

Command shape:
```bash
python3 -u tests/bench_nvfp4_mega_moe_sm90.py \
  --num-processes 8 \
  --batches 512 1024 2048 \
  --num-tests 20 \
  --weight-mode compact
```

Results:

| config | M512 | M1024 | M2048 |
|---|---:|---:|---:|
| default SM count | 2242.5 us | 3671.0 us | 6308.0 us |
| `DG_SM90_MOE_SET_NUM_SMS=72` | 2354.9 us | 3894.0 us | 6791.0 us |
| `DG_SM90_MOE_SET_NUM_SMS=64` | 2748.5 us | 4355.0 us | 7576.0 us |

Decision:
- Rejected. Reducing SM count consistently regresses M512/M1024/M2048.
- The barrier/tail issue is not solved by simply reducing the number of participating SMs; full SM count remains the best default.


## 2026-06-12 Validation - Post-instrumentation correctness and default benchmark

Correctness command:
```bash
MASTER_ADDR=127.0.0.1 MASTER_PORT=29705 \
DG_JIT_CACHE_DIR=/tmp/dg_jit_nvfp4_post_instrument_correct_0612 \
python3 -u tests/test_nvfp4_mega_moe_sm90_correctness.py \
  --batches 128 512 \
  --weight-scales 0.05 \
  --reference-mode fp8-bridge \
  --weight-mode compact \
  --cosine-mean-threshold 0.99 \
  --cosine-min-threshold 0.99 \
  --norm-ratio-min 0.99 \
  --norm-ratio-max 1.01
```

Result:
- dequant unit test PASS.
- CUDA LUT dequant unit test PASS.
- M128 PASS: cosine_min=0.9996, norm_ratio=0.9997.
- M512 PASS: cosine_min=0.9996, norm_ratio=0.9995.

Default non-profile 20-run benchmark after adding profile-only instrumentation:

| tokens | compact NVFP4 | mean_rank | max_rank | PR323 W8A8 ref | compact/W8A8 |
|---:|---:|---:|---:|---:|---:|
| 64 | 1142.5 us | 1130.7 us | 1142.5 us | 869.9 us | 1.313x, 31.3% slower |
| 128 | 1239.2 us | 1221.4 us | 1241.8 us | 837.1 us | 1.480x, 48.0% slower |
| 256 | 1527.7 us | 1539.6 us | 1556.9 us | 1460.0 us | 1.046x, 4.6% slower |
| 512 | 2340.4 us | 2328.0 us | 2342.9 us | 2320.0 us | 1.009x, 0.9% slower |
| 1024 | 3671.0 us | 3660.9 us | 3673.0 us | 3518.0 us | 1.043x, 4.3% slower |
| 2048 | 6562.0 us | 6532.2 us | 6562.0 us | 6274.0 us | 1.046x, 4.6% slower |
| 4096 | 11878.0 us | 11929.8 us | 11956.0 us | 11402.0 us | 1.042x, 4.2% slower |
| 8192 | 22718.0 us | 22712.0 us | 22726.0 us | 21904.0 us | 1.037x, 3.7% slower |

Interpretation:
- Profile-only instrumentation does not affect the default non-profile path in an obvious way; results stay in the existing run-to-run band.
- M128 remains improved versus the old pre-gate 1340.3 us result, though this run did not reproduce the best 1121.3 us M128 result.
- M512 remains near parity with W8A8; M1024/M4096/M8192 remain about 3.7-4.3% slower in this run.



## 2026-06-12 Profile - Full-size tile overlap check

Purpose:
- Re-check whether the true compact NVFP4 path is losing time because L1 epilogue/scatter is not overlapped with the next tile's load/dequant.
- Extend the earlier phase profile from M64-M2048 to M4096/M8192.

Command:
```bash
MASTER_ADDR=127.0.0.1 MASTER_PORT=29731 \
DG_SM90_MOE_PHASE_PROFILE=1 \
DG_JIT_CACHE_DIR=/tmp/dg_jit_nvfp4_overlap_profile_full_0612 \
python3 -u tests/bench_nvfp4_mega_moe_sm90.py \
  --num-processes 8 \
  --batches 256 512 1024 2048 4096 8192 \
  --num-tests 3 \
  --weight-mode compact \
  2>&1 | tee /tmp/nvfp4_overlap_phase_profile_full_0612.log
```

Results:

| tokens | compact | math_loop avg cyc | dispatch_pull avg cyc | combine_barrier avg cyc | combine_reduce avg cyc | math_dequant_wait avg cyc | l1_tma_wait avg cyc | l2_ready_wait avg cyc | l2_scatter avg cyc |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 256 | 1664.0 us | 756342 | 54016 | 25282 | 4787 | 74 | 70 | 256 | 1175 |
| 512 | 2461.0 us | 1055416 | 102918 | 393528 | 12579 | 64 | 72 | 232 | 1579 |
| 1024 | 3813.0 us | 2161959 | 203394 | 112328 | 27466 | 62 | 71 | 253 | 1700 |
| 2048 | 6638.0 us | 3240604 | 401488 | 106723 | 60480 | 59 | 72 | 236 | 1776 |
| 4096 | 12411.0 us | 5974021 | 810892 | 255649 | 126696 | 54 | 73 | 227 | 1996 |
| 8192 | 23697.0 us | 11425480 | 1614828 | 455804 | 259861 | 53 | 73 | 225 | 2039 |

Interpretation:
- Tile-to-tile overlap is not the dominant exposed loss in the current profile.
- `math_dequant_wait` is only 53-74 cycles at M256-M8192, so B-side NVFP4 dequant is being hidden well by the math loop.
- `l1_tma_wait` stays around 70-73 cycles for all sizes; synchronous L1 output store completion is not a meaningful wall-time limiter.
- `l2_ready_wait` stays around 225-256 cycles; L2 is not materially blocked waiting for L1 publication.
- `l2_scatter` grows only from about 1.2k to 2.0k cycles per event; worth local cleanup but too small to explain the 3-5% W8A8 gap by itself.
- The exposed large terms are `dispatch_pull`, `combine_barrier`, `combine_reduce` at large M, and the `math_loop` floor. M512 is especially noisy: this run shows a large `combine_barrier` spike.

Decision:
- Do not spend more effort on async L1 TMA store as the primary direction; measured wait is tiny and previous 2-WG async attempts timed out.
- Next structural directions should target global dispatch/combine lifecycle, chunking/reduction behavior, or per-size kernel/config specialization rather than tile load/dequant versus epilogue overlap.


## 2026-06-12 Experiment - Force hidden=7168 combine reduce to 7 chunks (rejected)

Hypothesis:
- FP8/PR323 uses a split-MN-specific combine chunking path where hidden=7168 can be split into 7 chunks.
- NVFP4 default combine currently uses `kNumDefaultChunks` only, which is 2 chunks for hidden=7168.
- Forcing 7 chunks reduces each lane's BF16 reduce accumulator from about 56 `float2` values to 16 `float2` values, but increases chunk-loop and TMA load/store count.

Patch tested:
- Temporarily changed NVFP4 combine `kNumChunks` from `kNumDefaultChunks` to 7 when `kHidden % 7 == 0`.
- Rebuilt successfully.

Correctness:
```bash
MASTER_ADDR=127.0.0.1 MASTER_PORT=29732 \
DG_JIT_CACHE_DIR=/tmp/dg_jit_nvfp4_combine7_correct_0612 \
python3 -u tests/test_nvfp4_mega_moe_sm90_correctness.py \
  --batches 256 512 2048 \
  --weight-scales 0.05 \
  --reference-mode fp8-bridge \
  --weight-mode compact \
  --cosine-mean-threshold 0.99 \
  --cosine-min-threshold 0.99 \
  --norm-ratio-min 0.99 \
  --norm-ratio-max 1.01
```
- PASS.
- M256 cosine_min=0.9996 norm_ratio=0.9996.
- M512 cosine_min=0.9996 norm_ratio=0.9995.
- M2048 cosine_min=0.9995 norm_ratio=0.9995.

20-run benchmark:

| tokens | combine7 compact | previous retained compact | delta |
|---:|---:|---:|---:|
| 64 | 1410.8 us | 1142.5 us | 23.5% slower |
| 128 | 1166.9 us | 1239.2 us | 5.8% faster |
| 256 | 1538.8 us | 1527.7 us | 0.7% slower |
| 512 | 2381.7 us | 2340.4 us | 1.8% slower |
| 1024 | 3708.0 us | 3671.0 us | 1.0% slower |
| 2048 | 6429.0 us | 6562.0 us | 2.0% faster |
| 4096 | 11936.0 us | 11878.0 us | 0.5% slower |
| 8192 | 22774.0 us | 22718.0 us | 0.2% slower |

Phase profile comparison:
- `combine_reduce` did not improve; it got worse.
- M64 `combine_reduce`: about 1.8k cycles retained path -> 3.7k cycles combine7.
- M128 `combine_reduce`: about 3.6k cycles retained path -> 7.4k cycles combine7.
- M512 `combine_reduce`: about 13k cycles retained path -> 18k cycles combine7.
- M2048 `combine_reduce`: about 60k cycles retained path -> 81k cycles combine7.

Decision:
- Rejected and reverted.
- The 20-run M128/M2048 improvement is not supported by phase-profile causality; it is likely run-to-run noise or interaction with unrelated barrier timing.
- Keep `kNumChunks = kNumDefaultChunks` for NVFP4 default.
- Do not add per-M combine7 gating unless a future longer run proves stable end-to-end benefit and a profile shows the source of the win.


## 2026-06-12 Sweep - Existing per-M knobs after overlap profile

Purpose:
- The overlap profile showed tile-to-tile ready waits are tiny and the visible costs are dispatch/combine/math-loop.
- Sweep existing runtime knobs before adding new code, to see whether any per-M default should be changed.

Baseline used for comparison:
- PR323 W8A8 refs: M512 2320 us, M1024 3518 us, M2048 6274 us, M4096 11402 us, M8192 21904 us.
- Retained NVFP4 default remains split L1/L2 + compact fused B+scale + BM128 for M256+.

20-run sweep results:

| config | M512 | M1024 | M2048 | M4096 | M8192 | decision |
|---|---:|---:|---:|---:|---:|---|
| default | 2192.2 | 3774.0 | 6333.0 | 11927.0 | 22787.0 | baseline sweep |
| `DG_SM90_MOE_L2_NO_DISPATCH_PIPELINE=0` | 2411.0 | 4022.0 | 7088.0 | 13155.0 | 24660.0 | reject, all slower |
| `DG_SM90_NVFP4_DIRECT_SCATTER_METADATA_BCAST=0` | 2218.7 | 3789.0 | 6316.0 | 11932.0 | 22885.0 | reject, only tiny M2048 signal |
| `DG_SM90_NVFP4_L2_ARRIVAL_COUNTER=0` | 2275.9 | 3654.0 | 6344.0 | 12045.0 | 22988.0 | reject; M1024 signal not stable |
| `DG_SM90_MOE_L2_DUAL_ACCUM=1` | 2216.7 | 4114.0 | 6294.0 | 11981.0 | 22870.0 | reject, M1024 bad |
| `DG_SM90_MOE_L2_DUAL_ACCUM=0` | 2212.4 | 3626.0 | 6373.0 | 11895.0 | 22773.0 | reject after M1024 30-run |
| `DG_SM90_MOE_L2_NMAJOR=0` | 2386.4 | 3894.0 | 6348.0 | 11882.0 | 22784.0 | reject, broad regression |
| `DG_SM90_NVFP4_BM128_HEURISTIC=0` | 2742.0 | 4771.0 | 8796.0 | 18143.0 | 33175.0 | reject, BM128 is required |
| `DG_SM90_NVFP4_NUM_STAGES=2` | 2584.2 | 4158.0 | 7082.0 | 13274.0 | 25550.0 | reject, 3-stage required |
| `DG_SM90_MOE_SPLIT_L1_L2=0` | 3318.0 | 5275.0 | 9268.0 | 18028.0 | 34616.0 | reject, split L1/L2 required |
| `DG_SM90_MOE_L1_NMAJOR=1` | 2241.3 | 3683.0 | 6313.0 | 11860.0 | 22779.0 | reject for default; tiny/noisy M2048-M4096 signal |

30-run stability checks:

| config | M1024 | M2048 | M4096 | decision |
|---|---:|---:|---:|---|
| default | 3633.0 | 6463.0 | 11770.0 | retained default |
| `DG_SM90_MOE_L2_DUAL_ACCUM=0` | 3737.0 | - | - | M1024 signal reversed, reject |
| `DG_SM90_NVFP4_DIRECT_SCATTER_METADATA_BCAST=0` | - | 6469.0 | - | no M2048 gain, reject |
| `DG_SM90_MOE_L1_NMAJOR=1` | - | 6444.0 | - | only ~0.3% faster than same-run default; below noise, reject for default |

Invalid config:
- `DG_SM90_NVFP4_DISPATCH_THREADS=64` alone hits host assert because dispatch + non-epilogue threads must be a multiple of 128. It was not pursued because reducing non-epilogue threads would disable the retained loader-dequant path.

Current best retained/default observations versus PR323 W8A8:

| tokens | best retained NVFP4 | PR323 W8A8 | gap |
|---:|---:|---:|---:|
| 64 | 1142.5 us | 869.9 us | 31.3% slower |
| 128 | 1239.2 us | 837.1 us | 48.0% slower |
| 256 | 1527.7 us | 1460.0 us | 4.6% slower |
| 512 | 2192.2 us | 2320.0 us | 5.5% faster |
| 1024 | 3633.0 us | 3518.0 us | 3.3% slower |
| 2048 | 6333.0 us | 6274.0 us | 0.9% slower, but 30-run same-code sample was 6463 us |
| 4096 | 11770.0 us | 11402.0 us | 3.2% slower |
| 8192 | 22718.0 us | 21904.0 us | 3.7% slower |

Decision:
- No existing runtime knob is stable enough to change the default.
- The strongest retained result is still M512, where NVFP4 is already faster than W8A8.
- M2048 is close but noisy; the knob sweeps did not find a reliable per-M default improvement.
- Next useful step is W8A8-vs-NVFP4 phase-profile comparison, not more blind knob sweeping.


## 2026-06-12 Rebaseline - Same-script current FP8 shadow/W8A8 comparison

Reason:
- Earlier tables used the PR323 retained W8A8 reference numbers.
- To avoid comparing against a stale/noisy external reference, run the same benchmark script in the same repo/container with `--weight-mode fp8-shadow`.
- This path materializes NVFP4 weights through the FP8 shadow bridge and calls `deep_gemm.fp8_mega_moe`, so it is the closest same-script W8A8/fp8-kernel latency reference available in this repo.

Commands:
```bash
python3 -u tests/bench_nvfp4_mega_moe_sm90.py \
  --num-processes 8 \
  --batches 64 128 256 \
  --num-tests 20 \
  --weight-mode fp8-shadow

python3 -u tests/bench_nvfp4_mega_moe_sm90.py \
  --num-processes 8 \
  --batches 512 1024 2048 4096 8192 \
  --num-tests 20 \
  --weight-mode fp8-shadow
```

Current FP8 shadow results:

| tokens | fp8-shadow/W8A8 current |
|---:|---:|
| 64 | 823.3 us |
| 128 | 811.8 us |
| 256 | 1435.0 us |
| 512 | 2126.0 us |
| 1024 | 3539.0 us |
| 2048 | 6019.0 us |
| 4096 | 11468.0 us |
| 8192 | 21825.0 us |

Same-script comparison using retained NVFP4 observations from the matching/default logs:

| tokens | retained NVFP4 | current fp8-shadow/W8A8 | gap |
|---:|---:|---:|---:|
| 64 | 1142.5 us | 823.3 us | 38.8% slower |
| 128 | 1239.2 us | 811.8 us | 52.6% slower |
| 256 | 1527.7 us | 1435.0 us | 6.5% slower |
| 512 | 2192.2 us | 2126.0 us | 3.1% slower |
| 1024 | 3774.0 us | 3539.0 us | 6.6% slower |
| 2048 | 6333.0 us | 6019.0 us | 5.2% slower |
| 4096 | 11927.0 us | 11468.0 us | 4.0% slower |
| 8192 | 22787.0 us | 21825.0 us | 4.4% slower |

Interpretation:
- Against the older PR323 reference, NVFP4 looked faster at M512 and close at M2048.
- Against the same-script current FP8 shadow reference, retained NVFP4 is still slower at every measured size.
- The remaining target is therefore stricter than the old PR323 table suggested: large-M NVFP4 needs another ~3-6% and small-M needs a different structural path.
- FP8 shadow does not have the NVFP4 phase-profile instrumentation, so this is a timing baseline, not a phase-level breakdown.


## 2026-06-12 True fused NVFP4 - Compile-time phase dispatch

Context:
- User clarified that the real target should be the true fused NVFP4 kernel, while retaining the current split L1/L2 two-kernel path.
- Before this step, `DG_SM90_MOE_SPLIT_L1_L2=0` true fused path was correct but much slower than the split path.

Change tested:
- In the fused path (`kRunL1Phase && kRunL2Phase`), dispatch scheduler blocks to the role lambdas with compile-time phase constants:
  - `std::integral_constant<sched::BlockPhase, Linear1>` for L1 blocks.
  - `std::integral_constant<sched::BlockPhase, Linear2>` for L2 blocks.
- Goal: allow compiler folding of L1/L2 branches inside fused loader/math/epilogue code.
- The split L1/L2 path remains retained and is still available as the two-kernel fallback/default path.

Correctness command:
```bash
DG_SM90_MOE_SPLIT_L1_L2=0 \
DG_JIT_CACHE_DIR=/tmp/dg_jit_nvfp4_true_fused_ctphase_correct_0612 \
python3 -u tests/test_nvfp4_mega_moe_sm90_correctness.py \
  --batches 128 512 2048 \
  --weight-scales 0.05 \
  --reference-mode fp8-bridge \
  --weight-mode compact \
  --cosine-mean-threshold 0.99 \
  --cosine-min-threshold 0.99 \
  --norm-ratio-min 0.99 \
  --norm-ratio-max 1.01
```

Correctness result:
- PASS.
- M128 cosine_min=0.9996 norm_ratio=0.9997.
- M512 cosine_min=0.9996 norm_ratio=0.9995.
- M2048 cosine_min=0.9995 norm_ratio=0.9995.

Current true fused 20-run benchmark after compile-time phase dispatch:

| tokens | true fused current | previous true fused | true fused improvement |
|---:|---:|---:|---:|
| 64 | 1128.0 us | - | - |
| 128 | 1330.0 us | - | - |
| 256 | 1837.0 us | - | - |
| 512 | 2665.0 us | 3318.0 us | 19.7% faster |
| 1024 | 4297.0 us | 5275.0 us | 18.5% faster |
| 2048 | 7507.0 us | 9268.0 us | 19.0% faster |
| 4096 | 13629.0 us | 18028.0 us | 24.4% faster |
| 8192 | 26167.0 us | 34616.0 us | 24.4% faster |

Comparison against current retained split L1/L2 two-kernel path:

| tokens | true fused current | split two-kernel current | true fused vs split |
|---:|---:|---:|---:|
| 64 | 1128.0 us | 1170.9 us | 3.7% faster |
| 128 | 1330.0 us | 1217.6 us | 9.2% slower |
| 256 | 1837.0 us | 1502.3 us | 22.3% slower |
| 512 | 2665.0 us | 2319.3 us | 14.9% slower |
| 1024 | 4297.0 us | 3675.0 us | 16.9% slower |
| 2048 | 7507.0 us | 6504.0 us | 15.4% slower |
| 4096 | 13629.0 us | 11841.0 us | 15.1% slower |
| 8192 | 26167.0 us | 22712.0 us | 15.2% slower |

Interpretation:
- Compile-time phase dispatch is a real true-fused-path improvement; it removes roughly one-fifth of large-M true fused runtime.
- It is still not competitive with the split two-kernel path for M128+.
- The true fused path is now only better at M64 in this 20-run sample, likely because it avoids the second launch while M is too small for split-phase work to dominate.
- For M512+, the remaining fused gap is still about 15%, so the real fused path needs another structural improvement before it can replace split L1/L2.

## 2026-06-12 Iteration: true fused auto phase propagation

Change:
- Kept split L1/L2 path intact.
- In sm90_nvfp4_mega_moe.cuh, changed the five for_each_selected_block lambda phase parameters from const sched::BlockPhase& to const auto& so the true fused scheduler compile-time Linear1/Linear2 phase can propagate into loader/math lambdas.

Correctness:
- Command: DG_SM90_MOE_SPLIT_L1_L2=0 DG_JIT_CACHE_DIR=/tmp/dg_jit_nvfp4_true_fused_autophase_correct_0612 python3 -u tests/test_nvfp4_mega_moe_sm90_correctness.py --batches 128 512 2048 --weight-scales 0.05 --reference-mode fp8-bridge --weight-mode compact --cosine-mean-threshold 0.99 --cosine-min-threshold 0.99 --norm-ratio-min 0.99 --norm-ratio-max 1.01
- Result: PASS. Dequant unit PASS, CUDA LUT dequant unit PASS. M128 cosine_min=0.9996 norm_ratio=0.9997; M512 cosine_min=0.9996 norm_ratio=0.9995; M2048 cosine_min=0.9995 norm_ratio=0.9995.

Benchmark, true fused compact NVFP4, 8 ranks, 20 runs:

| M | time us |
|---:|---:|
| 64 | 1112.0 |
| 128 | 1134.0 |
| 256 | 1799.0 |
| 512 | 2613.0 |
| 1024 | 4127.0 |
| 2048 | 7499.0 |
| 4096 | 13609.0 |
| 8192 | 26174.0 |

Delta vs previous true fused default run: previous was 1440, 1260, 1898, 2625, 4292, 7446, 13659, 26170 us for M64..8192.
- M64: 22.8% faster.
- M128: 10.0% faster.
- M256: 5.2% faster.
- M512/M4096/M8192: roughly flat.
- M1024: 3.8% faster.
- M2048: 0.7% slower.

Resource check:
- BM64 true fused variants: REG=168 STACK=24.
- BM128 true fused variants: REG=128 STACK=216 or 248, still not reduced versus prior, so there are still runtime phase branches/local states in the hot path.

Conclusion:
- Effective for small M, keep for now.
- Not enough to close the W8A8 gap. Next target is converting the remaining hot math block_phase branches to compile-time phase traits where the caller passes a compile-time phase.

## 2026-06-12 Iteration: retained true fused defaults after knob sweep

Retained changes:
- Kept the previous true fused auto phase propagation: for_each_selected_block lambda phase parameters use const auto& so compile-time Linear1/Linear2 phase can reach the hot lambdas.
- Added a host heuristic for true fused M256 only: default L1 dual-K is disabled when DG_SM90_MOE_SPLIT_L1_L2=0 and num_tokens=256. Split L1/L2 path is unchanged.

Correctness:
- Command: DG_SM90_MOE_SPLIT_L1_L2=0 DG_JIT_CACHE_DIR=/tmp/dg_jit_nvfp4_true_fused_retained_correct_0612 python3 -u tests/test_nvfp4_mega_moe_sm90_correctness.py --batches 128 256 512 2048 --weight-scales 0.05 --reference-mode fp8-bridge --weight-mode compact --cosine-mean-threshold 0.99 --cosine-min-threshold 0.99 --norm-ratio-min 0.99 --norm-ratio-max 1.01
- Result: PASS. Dequant unit PASS, CUDA LUT dequant unit PASS. M128 cosine_min=0.9996 norm_ratio=0.9997; M256 cosine_min=0.9996 norm_ratio=0.9996; M512 cosine_min=0.9996 norm_ratio=0.9995; M2048 cosine_min=0.9995 norm_ratio=0.9995.

Latest benchmark, true fused compact NVFP4, 8 ranks, 20 runs:

| M | NVFP4 true fused us | W8A8/fp8-shadow ref us | NVFP4 / W8A8 | gap |
|---:|---:|---:|---:|---:|
| 64 | 1114.0 | 823.3 | 1.353x | 35.3% slower |
| 128 | 1162.0 | 811.8 | 1.432x | 43.1% slower |
| 256 | 1676.0 | 1435.0 | 1.168x | 16.8% slower |
| 512 | 2670.0 | 2126.0 | 1.256x | 25.6% slower |
| 1024 | 4171.0 | 3539.0 | 1.179x | 17.9% slower |
| 2048 | 7384.0 | 6019.0 | 1.227x | 22.7% slower |
| 4096 | 13646.0 | 11468.0 | 1.190x | 19.0% slower |
| 8192 | 26284.0 | 21825.0 | 1.204x | 20.4% slower |

Delta vs previous true fused default before this iteration: previous was 1440, 1260, 1898, 2625, 4292, 7446, 13659, 26170 us for M64..8192.
- M64: 22.6% faster.
- M128: 7.8% faster.
- M256: 11.7% faster.
- M512: 1.7% slower.
- M1024: 2.8% faster.
- M2048: 0.8% faster.
- M4096: flat.
- M8192: flat/slightly slower.

Resource check after retained changes:
- BM64 true fused variants: REG=168 STACK=24.
- BM128 true fused variants: REG=128 STACK=208/216. The M256 L1-dual-off variant is the 208B stack case; other BM128 variants still carry 216B stack.

Effective attempts:
- Auto phase propagation into loader/math lambdas: effective for small M, especially M64/M128/M256.
- True fused M256 default L1_DUAL_K=0: effective. M256 improved from 1799 us to about 1676-1684 us in 20-run measurements.

Ineffective or reverted attempts:
- L1_DUAL_K=0 globally: rejected. Helped M256, but slowed M64/M128/M512/M1024.
- BM64 for M256+ by disabling BM128 heuristic: rejected. M256 was 1734 us and M512/M1024 regressed.
- L2_DUAL_ACCUM=1 in true fused: rejected. M512+ regressed to very large runtimes around 10 ms to 107 ms.
- L2_ARRIVAL_COUNTER=0 globally: rejected. It was acceptable for some mid sizes but worse for M4096/M8192; current per-M default is better.
- DIRECT_SCATTER_METADATA_BCAST=0: rejected after rerun. Initial M512/M1024 looked promising, but same-shape rerun showed default was better.
- L2_NMAJOR=0: rejected after rerun. M2048-only rerun showed default 7388 us versus 7418 us with L2_NMAJOR=0.
- if constexpr phase branch rewrite in math/epilogue: reverted. It preserved correctness but regressed M64/M128 badly to 1881/2123 us, despite tiny gains at M2048+.

Current conclusion:
- The retained true fused path is better than the start of this iteration at small M, but still does not reach the W8A8 target.
- The remaining true fused gap is mainly BM128 hot-path overhead: gemm_core, loader_dequant, and epilogue remain higher than the split path/W8A8 reference. Resource usage still shows BM128 stack around 208-216B, so L1/L2 state separation is incomplete.
- Recommended next structural direction: avoid a monolithic lambda carrying both L1 and L2 local states. A safer version of the reverted if constexpr attempt would explicitly split run_l1_block and run_l2_block helper functions/lambdas rather than trying to rewrite branch predicates in-place.

## 2026-06-12 Sanity: retained split L1/L2 path

Purpose:
- The user asked to keep the existing two-kernel split L1/L2 path while returning to true fused work. After auto phase propagation and the true-fused-only M256 heuristic, reran split path correctness and benchmark to make sure the fallback/current best-overall path is still valid.

Correctness, default split path:
- Command: DG_JIT_CACHE_DIR=/tmp/dg_jit_nvfp4_split_retained_correct_0612 python3 -u tests/test_nvfp4_mega_moe_sm90_correctness.py --batches 128 256 512 2048 --weight-scales 0.05 --reference-mode fp8-bridge --weight-mode compact --cosine-mean-threshold 0.99 --cosine-min-threshold 0.99 --norm-ratio-min 0.99 --norm-ratio-max 1.01
- Result: PASS. Dequant unit PASS, CUDA LUT dequant unit PASS. M128/M256/M512/M2048 all cosine_min around 0.9995-0.9996 and norm_ratio around 0.9995-0.9997.

Benchmark, default split path, compact NVFP4, 8 ranks, 20 runs:

| M | NVFP4 split us | W8A8/fp8-shadow ref us | NVFP4 / W8A8 | gap |
|---:|---:|---:|---:|---:|
| 64 | 1362.2 | 823.3 | 1.655x | 65.5% slower |
| 128 | 1287.9 | 811.8 | 1.586x | 58.6% slower |
| 256 | 1512.3 | 1435.0 | 1.054x | 5.4% slower |
| 512 | 2336.1 | 2126.0 | 1.099x | 9.9% slower |
| 1024 | 3637.0 | 3539.0 | 1.028x | 2.8% slower |
| 2048 | 6383.0 | 6019.0 | 1.060x | 6.0% slower |
| 4096 | 11831.0 | 11468.0 | 1.032x | 3.2% slower |
| 8192 | 22620.0 | 21825.0 | 1.036x | 3.6% slower |

Interpretation:
- Current best overall path is mixed: true fused is better for M64/M128, split is better for M256 and larger.
- Split path is now close to the W8A8/fp8-shadow reference for M256+ but not faster. True fused still needs structural work before it can replace split for larger M.

## 2026-06-12 Reverted attempt: mixed default split/fused by M

Attempt:
- Changed host default so M64/M128 would use true fused while M256+ kept split, with DG_SM90_MOE_SPLIT_L1_L2 still available as override.

Validation:
- Default mixed correctness PASS for M64/M128/M256/M512/M2048.

Benchmark, default mixed path, 8 ranks, 20 runs:

| M | time us |
|---:|---:|
| 64 | 2210.0 |
| 128 | 2314.0 |
| 256 | 1521.0 |
| 512 | 2314.5 |
| 1024 | 3602.0 |
| 2048 | 6443.0 |
| 4096 | 11804.0 |
| 8192 | 22584.0 |

Result:
- Reverted. M64/M128 became roughly 2x slower than explicit true fused, likely because the Python benchmark/wrapper still organizes default execution around the split multi-kernel path. Host-only default switching is therefore unsafe without changing the upper-level launch/timing path consistently.
- After revert, default split_l1_l2_default is restored. Explicit DG_SM90_MOE_SPLIT_L1_L2=0 remains the way to benchmark true fused.

## 2026-06-12 Reverted attempt: opt-in async L1 TMA store for NVFP4

Motivation:
- This targets the block-level pipeline hypothesis: overlap L1 epilogue/TMA store and L2 readiness notification with later load/dequant work instead of waiting inside the L1 epilogue.
- Existing kernel code already has kAsyncL1TMAStore and double-buffered smem_cd_l1 support, but the NVFP4 host launcher asserted DG_SM90_MOE_ASYNC_L1_STORE unsupported.

Attempt:
- Temporarily removed the host assert and passed DG_SM90_MOE_ASYNC_L1_STORE into args.async_l1_tma_store as an opt-in only. Default stayed false.

Validation result:
- Failed correctness before benchmark. Command used DG_SM90_MOE_SPLIT_L1_L2=0 DG_SM90_MOE_ASYNC_L1_STORE=1 with compact true fused M64/M128.
- M64 hit CUDA illegal memory access.
- DG_PRINT_CONFIGS showed the tested single-rank shape selecting block_m=64, block_n=128, stages=7, dispatch=64, non_epi=64, epilogue=256. That shape does not satisfy the intended single-epilogue-WG async condition and is not a safe async-L1 test configuration.
- Explicitly setting DG_SM90_NVFP4_DISPATCH_THREADS=128, DG_SM90_NVFP4_NON_EPILOGUE_THREADS=128, and DG_SM90_NVFP4_EPILOGUE_THREADS=128 still printed 64/64/256 in this correctness path and still illegal-accessed.

Result:
- Reverted. The host assert and args.async_l1_tma_store=false are restored.
- Do not retry this as a host-only opt-in. A real async L1 pipeline needs launcher/config cleanup first so the kernel is instantiated with a supported one-epilogue-WG shape, plus a correctness gate before benchmarking.

## 2026-06-12 Retained: coordinated mixed default and benchmark timing fix

Motivation:
- The previous host-only mixed default attempt appeared to regress M64/M128 to about 2x, but investigation showed the benchmark script still inferred split_l1_l2=True and multiplied the timing by two.
- This was a measurement-path bug, not necessarily a kernel regression.

Change:
- Host launcher default now selects true fused for M64/M128 and split L1/L2 for M256+ unless DG_SM90_MOE_SPLIT_L1_L2 explicitly overrides it.
- tests/bench_nvfp4_mega_moe_sm90.py now uses the same default split/fused inference, so M64/M128 true fused timings are not doubled.
- Split path remains available and remains default for M256+.

Correctness:
- Command: DG_JIT_CACHE_DIR=/tmp/dg_jit_nvfp4_mixed_default_correct2_0612 python3 -u tests/test_nvfp4_mega_moe_sm90_correctness.py --batches 64 128 256 512 2048 --weight-scales 0.05 --reference-mode fp8-bridge --weight-mode compact --cosine-mean-threshold 0.99 --cosine-min-threshold 0.99 --norm-ratio-min 0.99 --norm-ratio-max 1.01
- Result: PASS. Dequant unit PASS, CUDA LUT dequant unit PASS. M64/M128/M256/M512/M2048 all cosine_min around 0.9995-0.9996, norm_ratio around 0.9995-0.9998.

Benchmark, corrected mixed default, compact NVFP4, 8 ranks, 20 runs:

| M | NVFP4 mixed default us | W8A8/fp8-shadow ref us | NVFP4 / W8A8 | gap |
|---:|---:|---:|---:|---:|
| 64 | 1130.0 | 823.3 | 1.373x | 37.3% slower |
| 128 | 1200.0 | 811.8 | 1.478x | 47.8% slower |
| 256 | 1477.2 | 1435.0 | 1.029x | 2.9% slower |
| 512 | 2367.5 | 2126.0 | 1.114x | 11.4% slower |
| 1024 | 3724.0 | 3539.0 | 1.052x | 5.2% slower |
| 2048 | 6457.0 | 6019.0 | 1.073x | 7.3% slower |
| 4096 | 11835.0 | 11468.0 | 1.032x | 3.2% slower |
| 8192 | 22631.0 | 21825.0 | 1.037x | 3.7% slower |

Result:
- Keep. This does not make NVFP4 faster than W8A8, but it makes the default path use the current best measured mode per size: true fused for M64/M128, split for M256+.
- Remaining target gap is now smallest at M256, about 2.9%, but M64/M128 and M512/M2048 are still materially slower than W8A8.

## 2026-06-12 Split-path M256/M512/M1024 W8A8 comparison and knob sweep

Reason:
- Corrected mixed default made M256 close to W8A8. Because benchmark variance depends strongly on the sampled recv distribution, reran compact and fp8-shadow on the same batch list before claiming speedup/regression.

30-run same-list comparison, batches 256/512/1024:

| M | recv | NVFP4 compact us | W8A8/fp8-shadow us | NVFP4 / W8A8 | result |
|---:|---:|---:|---:|---:|---|
| 256 | 1964 | 1489.3 | 1486.0 | 1.002x | tie, 0.2% slower |
| 512 | 4063 | 2372.0 | 2465.0 | 0.962x | 3.8% faster |
| 1024 | 8207 | 3666.0 | 3551.0 | 1.032x | 3.2% slower |

M1024 focused sweep, 20-run, recv=8015:

| Config | M1024 compact us | Notes |
|---|---:|---|
| default | 3612.0 | best compact in this sweep |
| DG_SM90_NVFP4_L2_ARRIVAL_COUNTER=0 | 3678.0 | worse |
| DG_SM90_NVFP4_DIRECT_SCATTER_METADATA_BCAST=0 | 3844.0 | worse |
| DG_SM90_MOE_L2_NMAJOR=0 | 3673.0 | worse |
| fp8-shadow same recv | 3480.0 | W8A8 still 3.8% faster than compact default |

Earlier 20-run M256/M512/M1024 split knob sweep, recv=1964/4063/8207:
- DG_SM90_MOE_L1_DUAL_K=0: 1548.2, 2436.0, 3596.0 us. Not retained; M256/M512 worse, M1024 tiny/noisy improvement only.
- DG_SM90_MOE_L2_DUAL_ACCUM=0: 1685.2, 2376.0, 3808.0 us. M512 looked better, but M512 default already uses L2_DUAL_ACCUM=0; no default change needed.
- DG_SM90_MOE_L2_NO_DISPATCH_PIPELINE=0: 1619.9, 2521.0, 3882.0 us. Worse.

Conclusion:
- Current NVFP4 mixed default can beat W8A8 at M512 on the same 30-run batch list.
- M256 is essentially tied.
- M1024 remains about 3-4% slower; tested existing split-path knobs did not close it.

## 2026-06-12 M1024 pipeline/profile and combine chunking attempts

M1024 split-path config/resource:
- Actual JIT split kernels are L1 and L2_nodisp.
- L1: block_m=128, block_n=128, stages=5, dispatch=128, non_epi=128, epi=256, REG=128, STACK=0.
- L2_nodisp: block_m=128, block_n=128, stages=6, dispatch=0, non_epi=128, epi=256, REG=168, STACK=32.

M1024 phase profile, split compact, recv=8015:
- dispatch_total avg=315395 cycles
- dispatch_pull avg=199822 cycles
- math_loop avg=1696314 cycles
- combine_barrier avg=305690 cycles
- combine_reduce avg=27964 cycles
- gemm_core avg=35675 cycles/block
- l1_epilogue avg=2920 cycles/block
- l2_epilogue avg=1871 cycles/block
- loader_dequant avg=1074 cycles/block
- math_dequant_wait avg=57 cycles/block
- l2_scatter avg=1775 cycles/block

Interpretation:
- Dequant/MMA overlap is already working: math_dequant_wait is tiny.
- M1024 residual gap is mostly fixed overhead plus L2 scatter/combine/dispatch, not dequant wait.

M1024 stages sweep, 20-run, recv=8015:
- default: 3612.0 us
- DG_SM90_NVFP4_NUM_STAGES=4: 3647.0 us, worse
- DG_SM90_NVFP4_NUM_STAGES=5: 3637.0 us, worse/noisy
- DG_SM90_NVFP4_NUM_STAGES=6: invalid, requested_num_stages exceeds max_num_stages
Result: no stages default change.

Attempt: port FP8-style hidden=7168 combine 7-chunking into NVFP4 combine.
- Correctness PASS for M256/M512/M1024.
- 30-run compact, recv=1964/4063/8207:
  - M256: 1512.5 us, worse than retained 1489.3 us
  - M512: 2554.0 us, worse than retained 2372.0 us
  - M1024: 3632.0 us, better than retained 3666.0 us but still slower than W8A8 3551.0 us
- Result: reverted. 7 chunks are not a safe default; they help M1024 slightly but regress M256/M512 materially.

Conclusion:
- Existing FP8 combine chunking is not directly portable as a global NVFP4 default.
- A future M1024-only chunk policy could be explored, but the observed gain is under 1% and does not close the W8A8 gap.

## 2026-06-12 Reverted attempt: M1024 experts-per-wave override

Attempt:
- Added temporary host env DG_SM90_NVFP4_EXPERTS_PER_WAVE to override config.num_experts_per_wave for NVFP4 experiments.
- Goal was to see whether M1024 fixed overhead/tail behavior improves by changing wave granularity.

M1024 compact, 20-run, recv=8015:
- default: 3612.0 us
- experts_per_wave=8: 3643.0 us
- experts_per_wave=16: 3680.0 us
- experts_per_wave=32: 3653.0 us

Result:
- Reverted. Default heuristic was best; no need to keep an unused env knob.

## 2026-06-12 - BN256 revert and true-fused mid-M probe

- Reverted the temporary `DG_SM90_NVFP4_BLOCK_N=256` host assert relaxation after BN256 correctness failed with illegal memory access. Verified code is back to BN128-only for compact NVFP4 bridge.
- Rebuilt successfully after the revert.
- Correctness PASS after revert for compact mixed default: M=64/128/256/512/1024/2048, weight_scale=0.05, fp8-bridge reference, cosine_min >= 0.9995, norm_ratio within [0.9994, 0.9998].
- Correctness PASS for forced true-fused (`DG_SM90_MOE_SPLIT_L1_L2=0`) on M=256/512/1024/2048.
- 20-run benchmark, same generated token lists:

| M | compact default | compact true-fused | fp8-shadow/W8A8 | default gap vs W8A8 | true-fused gap vs W8A8 |
|---:|---:|---:|---:|---:|---:|
| 256 | 1517.8 us | 1654.0 us | 1432.0 us | +5.99% | +15.50% |
| 512 | 2434.0 us | 2963.0 us | 2315.0 us | +5.14% | +27.99% |
| 1024 | 3631.0 us | 4188.0 us | 3547.0 us | +2.37% | +18.07% |
| 2048 | 6355.0 us | 7299.0 us | 6180.0 us | +2.83% | +18.11% |

Conclusion: forcing true-fused for mid/large M is not retained. It saves a launch but loses far more from resource pressure and weaker split-phase specialization. Continue optimizing split compact path toward W8A8.

## 2026-06-12 - Hybrid follow-up: small-M and scatter overlap probes

Context:
- Continue with hybrid policy: M64/M128 true-fused, M256+ split L1/L2.
- Current machine state showed strong long-tail/load effects. M512 compact and fp8-shadow both moved to ~4.57 ms in one run, so those absolute numbers should not be compared against earlier ~2.3 ms runs. Relative same-run comparisons remain useful.

Correctness / safety:
- Restored the failed BN256 probe to BN128-only earlier.
- Tried skipping the full epilogue barrier at the end of direct L2 scatter when `kL2ArrivalCounter` is enabled, aiming to allow scatter vs next tile load/dequant overlap. Correctness hung at M512 after passing M64/M128/M256. Killed only the exact hung test PIDs, then reverted the change. Not retained.
- Rebuilt after reverting the failed sync experiment. Default correctness PASS for M64/M128/M256/M512.

Small-M probes:

| Attempt | Correctness | Result |
|---|---|---|
| `DG_SM90_NVFP4_DISPATCH_THREADS=64` | host assert | invalid because dispatch+non-epilogue threads not 128-aligned |
| `DG_SM90_NVFP4_DISPATCH_THREADS=64 DG_SM90_NVFP4_NON_EPILOGUE_THREADS=64 DG_SM90_NVFP4_FUSED_B_SCALE=0` | PASS M64/M128, illegal address M256 | M64/M128 slower: 1417/1416 us vs default 1110/1130 us, not retained |
| `DG_SM90_NVFP4_DIRECT_SCATTER_METADATA_BCAST=1` | PASS M64/M128 | M64 1114 us, M128 1114 us; M128 small gain but max_rank nearly unchanged, not retained |
| `DG_SM90_NVFP4_L2_ARRIVAL_COUNTER=1` | PASS M64 | M64 1109 us vs default 1110 us, noise-level, not retained |
| counter+broadcast | PASS M64/M128 | M64 1117 us, M128 1116 us, worse, not retained |
| `DG_SM90_MOE_L1_DUAL_K=0` | PASS M64/M128 | first run improved to 1102/1111 us, but same-state forced-on run was 1111/1114 us and new default rerun was 1113/1119 us; too unstable, reverted default |

Same-run small-M W8A8/fp8-shadow reference before the load shift:
- M64: 775.7 us
- M128: 784.8 us

Conclusion:
- No small-M knob from this round is strong enough to keep by default.
- The direct L2 scatter barrier is necessary for current NVFP4 implementation; removing it deadlocks/hangs at M512.
- Continue with current hybrid defaults and focus on M1024+ split-path combine/scatter or a properly gated structural change.

## 2026-06-12 - Commit handoff state

Current retained default:
- Hybrid policy remains the best runnable default observed so far:
  - M64/M128: true-fused single kernel
  - M256+: split L1/L2 kernels
- No unstable small-M knobs are enabled by default.
- BN256 compact NVFP4 bridge remains gated off; verified default shape is BN128.

Latest validation before commit:
- Rebuild PASS after reverting failed small-M/default experiments.
- Default correctness PASS for M64/M128/M256/M512 after the failed scatter-sync experiment was reverted.
- Added opt-in `DG_SM90_NVFP4_COMBINE_7CHUNK=1` for future M1024-only combine exploration; default is OFF.
- Opt-in 7chunk correctness PASS for M256/M512/M1024 with fp8-bridge reference, weight_scale=0.05, cosine_min >= 0.9996, norm_ratio around 0.9994-0.9996.
- 7chunk is not enabled as default because earlier global 7chunk benchmarking regressed M256/M512 even though it slightly helped M1024.

Best reliable benchmark references retained in this log:
- Same-list 30-run: M256 compact 1489.3 us vs W8A8 1486.0 us; M512 compact 2372.0 us vs W8A8 2465.0 us; M1024 compact 3666.0 us vs W8A8 3551.0 us.
- Same-list 20-run later: M256 compact 1517.8 us vs W8A8 1432.0 us; M512 compact 2434.0 us vs W8A8 2315.0 us; M1024 compact 3631.0 us vs W8A8 3547.0 us; M2048 compact 6355.0 us vs W8A8 6180.0 us.
- The later M512 ~4.57 ms runs affected both compact and fp8-shadow and are treated as machine/load or benchmark-tail artifacts, not code regression evidence.
