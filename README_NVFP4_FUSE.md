# NVFP4 SM90 MegaMoE Notes

## Latest Current-Code 20-run Sweep Including Small M

This sweep was run after disabling the failed two-epilogue-WG async L1 store experiment, then retaining the M128 L2 no-dispatch + arrival-counter default. It uses true compact NVFP4 input, not FP8 shadow weights.

```bash
MASTER_ADDR=127.0.0.1 MASTER_PORT=29663 \
DG_JIT_CACHE_DIR=/tmp/dg_jit_nvfp4_post_async_revert_bench20_0612 \
python3 -u tests/bench_nvfp4_mega_moe_sm90.py \
  --num-processes 8 \
  --batches 64 128 256 512 1024 2048 4096 8192 \
  --num-tests 20 \
  --weight-mode compact
```

| tokens | compact NVFP4 20-run | PR323 W8A8 reference | compact/W8A8 | delta |
|---:|---:|---:|---:|---:|
| 64 | 1127.3 us | 869.9 us | 1.296x | 29.6% slower |
| 128 | 1121.3 us | 837.1 us | 1.340x | 34.0% slower |
| 256 | 1511.8 us | 1460.0 us | 1.035x | 3.5% slower |
| 512 | 2355.5 us | 2320.0 us | 1.015x | 1.5% slower |
| 1024 | 3673.0 us | 3518.0 us | 1.044x | 4.4% slower |
| 2048 | 6390.0 us | 6274.0 us | 1.019x | 1.9% slower |
| 4096 | 11928.0 us | 11402.0 us | 1.046x | 4.6% slower |
| 8192 | 22705.0 us | 21904.0 us | 1.037x | 3.7% slower |

Correctness immediately before this benchmark:
- PASS M64/M128/M256, weight_scale=0.05, plus broad M128 weight_scale=0.001/0.05/1.0, fp8-bridge reference, compact weight mode.
- Dequant unit test PASS and CUDA LUT dequant unit test PASS.

Note: the earlier 50-run retained fused80 table still contains the best verified M512 result, where compact NVFP4 reached 2214.1 us and beat the PR323 W8A8 reference by 4.6%. This 20-run sweep is the latest current-code sweep including M64/M128/M256.


This branch adds an SM90/Hopper MegaMoE path whose kernel input is packed NVFP4 weights.

Primary path for a real NVFP4-input kernel measurement:

```text
NVFP4 packed weights + UE4M3 scales
-> sm90_nvfp4_mega_moe compact fused kernel
-> online FP4->FP8 shared-memory bridge before WGMMA
```

Use `--weight-mode compact` for this path. This is the path to use when validating or benchmarking the NVFP4 kernel itself.

There is also an optional `--weight-mode fp8-shadow` path. It starts from NVFP4 weights, materializes an FP8 shadow copy at weight-load time, then calls the PR323 SM90 FP8 fused kernel with a unit-weight-scale specialization. This path is useful as a latency upper-bound and deployment option when extra FP8 shadow memory is acceptable, but it is not the compact online-NVFP4 kernel.

## Environment

Verified environment:

```text
Machine: 10.6.131.8
GPU: 8x H20-3e, SM90/Hopper class
Repo: /root/fac/megamoe/deepgemm_fp4_fuse
Branch: megamoe_nvfp4_fuse
Container: mega_moe_box
Container image: nvcr.io/nvidia/pytorch:25.02-py3
```

Connect to the machine:

```bash
sshpass -p 'BenchMark123!' ssh \
  -o StrictHostKeyChecking=no \
  -o PreferredAuthentications=password \
  -o PubkeyAuthentication=no \
  root@10.6.131.8
```

All build, correctness, and benchmark commands should run inside the container, not directly on the host:

```bash
docker exec -it mega_moe_box bash
cd /root/fac/megamoe/deepgemm_fp4_fuse
git checkout megamoe_nvfp4_fuse
```

Non-interactive one-liner pattern:

```bash
docker exec mega_moe_box bash -lc 'cd /root/fac/megamoe/deepgemm_fp4_fuse && <cmd>'
```

Do not use `pkill python` on this machine because other jobs may share the GPUs.

## Files

- `deep_gemm/include/deep_gemm/impls/sm90_nvfp4_mega_moe.cuh`: compact NVFP4 fused kernel.
- `csrc/jit_kernels/impls/sm90_nvfp4_mega_moe.hpp`: NVFP4 host launcher/config entry.
- `deep_gemm/include/deep_gemm/quantization/nvfp4_dequant.cuh`: NVFP4 E2M1 + UE4M3 dequant helpers.
- `deep_gemm/quantization_nvfp4.py`: Python NVFP4 quantize/dequant/prepack utilities.
- `deep_gemm/mega/__init__.py`: NVFP4 transform and optional FP8-shadow materializer.
- `deep_gemm/include/deep_gemm/impls/sm90_fp8_mega_moe.cuh`: PR323 FP8 kernel plus optional unit-weight-scale specialization for shadow weights.
- `tests/test_nvfp4_mega_moe_sm90_correctness.py`: single-rank NVFP4 correctness gate.
- `tests/bench_nvfp4_mega_moe_sm90.py`: 8-GPU NVFP4 benchmark.
- `NVFP4_FUSE_PROGRESS.md`: full iteration log, including failed attempts and measured results.

## Build

Run inside `mega_moe_box`:

```bash
cd /root/fac/megamoe/deepgemm_fp4_fuse
touch csrc/python_api.cpp
python setup.py build_ext --inplace -j 16
```

## Quick Test Flow: NVFP4 Kernel Input

From inside `mega_moe_box`, use this flow to test the actual compact NVFP4-input kernel.

```bash
cd /root/fac/megamoe/deepgemm_fp4_fuse

# 1. Rebuild
touch csrc/python_api.cpp
python setup.py build_ext --inplace -j 16

# 2. Correctness for the compact NVFP4-input kernel
MASTER_ADDR=127.0.0.1 MASTER_PORT=29590 \
DG_JIT_CACHE_DIR=/tmp/dg_jit_nvfp4_compact_correct \
python3 -u tests/test_nvfp4_mega_moe_sm90_correctness.py \
  --batches 32 256 \
  --weight-scales 0.001 0.05 1.0 \
  --reference-mode fp8-bridge \
  --weight-mode compact \
  --cosine-mean-threshold 0.99 \
  --cosine-min-threshold 0.99 \
  --norm-ratio-min 0.99 \
  --norm-ratio-max 1.01

# 3. Benchmark the compact NVFP4-input kernel
DG_JIT_CACHE_DIR=/tmp/dg_jit_nvfp4_compact_bench \
DG_SM90_MOE_SPLIT_L1_L2=1 \
python3 -u tests/bench_nvfp4_mega_moe_sm90.py \
  --weight-mode compact \
  --num-processes 8 \
  --num-tests 30 \
  --batches 32 64 128 256 512 1024 2048 4096 8192

# 4. PR323 W8A8 same-shape baseline
DG_JIT_CACHE_DIR=/tmp/dg_jit_w8a8_pr323_bench \
python3 -u tests/test_mega_moe_hopper.py \
  --fused-only-sweep \
  --num-processes 8 \
  --num-bench-tests 30 \
  --hidden 7168 \
  --intermediate-hidden 2048 \
  --num-experts 256 \
  --num-topk 8 \
  --batches 32 64 128 256 512 1024 2048 4096 8192
```

The compact benchmark should report `weight_mode=compact unit_weight_scale=False`.

## Optional FP8-Shadow Upper-Bound Flow

This is not the compact online-NVFP4 kernel. It is included to show what happens when NVFP4 weights are converted to an FP8 shadow cache outside the hot path.

```bash
cd /root/fac/megamoe/deepgemm_fp4_fuse

# Correctness for the shadow latency path
MASTER_ADDR=127.0.0.1 MASTER_PORT=29591 \
DG_JIT_CACHE_DIR=/tmp/dg_jit_nvfp4_shadow_correct \
python3 -u tests/test_nvfp4_mega_moe_sm90_correctness.py \
  --batches 32 256 \
  --weight-scales 0.001 0.05 1.0 \
  --reference-mode fp8-bridge \
  --weight-mode fp8-shadow \
  --cosine-mean-threshold 0.99 \
  --cosine-min-threshold 0.99 \
  --norm-ratio-min 0.99 \
  --norm-ratio-max 1.01

# Benchmark the optional shadow latency path
DG_JIT_CACHE_DIR=/tmp/dg_jit_nvfp4_shadow_bench \
python3 -u tests/bench_nvfp4_mega_moe_sm90.py \
  --weight-mode fp8-shadow \
  --num-processes 8 \
  --num-tests 50 \
  --batches 32 64 128 256 512 1024 2048 4096 8192
```

The shadow benchmark should report `weight_mode=fp8-shadow unit_weight_scale=True`.



### Latest Default Compact Timing: Fused Packed-B + Scale

Default compact mode now uses fused NVFP4 B storage: each BK128 row is `64B packed FP4 + 8B UE4M3 scale + 8B padding` and is loaded by one B TMA. This is still true compact NVFP4 input, not the FP8-shadow path. Set `DG_SM90_NVFP4_FUSED_B_SCALE=0` to fall back to the older separate packed-B plus SFB layout.

Final default 50-run result:

| tokens | compact NVFP4 | PR323 W8A8 ref | compact/W8A8 |
|---:|---:|---:|---:|
| 512 | 2214.1 us | 2320.0 us | 0.954x, 4.6% faster |
| 1024 | 3658.0 us | 3518.0 us | 1.040x slower |
| 2048 | 6329.0 us | 6274.0 us | 1.009x slower |
| 4096 | 11906.0 us | 11402.0 us | 1.044x slower |
| 8192 | 22805.0 us | 21904.0 us | 1.041x slower |

Correctness status for this default:

```text
dequant unit test: PASS
CUDA LUT dequant unit test: PASS
M32/M256/M512, weight_scale=0.001/0.05/1.0: PASS
worst cosine_min=0.9990
norm_ratio in [0.9992, 1.0059]
```

## Correctness Status

Compact NVFP4-input kernel:

```text
M32/M256/M512, weight_scale=0.001/0.05/1.0: PASS
worst cosine_min=0.9990
norm_ratio in [0.9992, 1.0059]
```

Run the compact quick-test command above before relying on a new build. M256+ uses the BM128 heuristic by default.

Optional FP8-shadow path:

```text
M32/M256, weight_scale=0.001/0.05/1.0: PASS
cosine_min >= 0.9991
norm_ratio in [0.9970, 1.0059]
```

8-rank compact-vs-shadow consistency was checked during development:

```text
M32:  cos_min_global=1.000000, finite_all=True
M256: cos_min_global=0.999797, norm_ratio_range=[0.997438,0.997562], finite_all=True
```

## Performance vs PR323 W8A8


### Current Accurate Compact Timing: Split L1+L2 Sum

The compact NVFP4 path currently launches split L1 and L2 kernels by default. The profiler exposes generated CUDA function names rather than the JIT build names, so the benchmark estimates one end-to-end MoE call by matching the shared kernel substring with with_multiple_kernels=True and multiplying the returned per-kernel average by two. Earlier compact tables that were near W8A8 for M512+ should be treated as half-time records, not end-to-end latency.

Latest compact 30-run after applying the reference repo M4096 dual-off default:

| tokens | W8A8 PR323 | compact NVFP4 | compact/W8A8 | delta |
|---:|---:|---:|---:|---:|
| 32 | 923.9 us | 1191.2 us | 1.289x | 28.9% slower |
| 64 | 869.9 us | 1179.1 us | 1.355x | 35.5% slower |
| 128 | 837.1 us | 1183.2 us | 1.414x | 41.4% slower |
| 256 | 1460.0 us | 1504.0 us | 1.030x | 3.0% slower |
| 512 | 2320.0 us | 4727.0 us | 2.037x | 103.7% slower |
| 1024 | 3518.0 us | 5950.0 us | 1.691x | 69.1% slower |
| 2048 | 6274.0 us | 13772.0 us | 2.195x | 119.5% slower |
| 4096 | 11402.0 us | 24330.0 us | 2.134x | 113.4% slower |
| 8192 | 21904.0 us | 50534.0 us | 2.307x | 130.7% slower |

Correctness for this build: M32/M256/M4096 with weight_scale=0.001/0.05/1.0 PASS, worst cosine_min=0.9987.


### Rank-Aware Compact Timing

The benchmark now also reports mean_rank and max_rank. The original compact field remains rank0 latency for continuity with prior logs, but distributed tail latency is closer to max_rank.

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

A one-iteration profiler table check showed the same kind of rank tail in W8A8 as well, so W8A8 comparisons must use the same rank0 or max-rank convention.


### Current In-Repo W8A8 Comparison

Current W8A8 from tests/test_mega_moe_hopper.py was re-run with the same rank-aware reporting. This is the comparable in-repo baseline; the historical PR323 table below remains superseded until its exact build environment is restored.

| tokens | W8A8 rank0 | NVFP4 rank0 | rank0 ratio | W8A8 max | NVFP4 max | max ratio |
|---:|---:|---:|---:|---:|---:|---:|
| 32 | 787.5 us | 1190.8 us | 1.512x | 3052.0 us | 3441.9 us | 1.128x |
| 64 | 805.2 us | 1179.0 us | 1.464x | 3074.0 us | 3432.2 us | 1.116x |
| 128 | 814.1 us | 1182.6 us | 1.453x | 3087.0 us | 3435.2 us | 1.113x |
| 256 | 1399.0 us | 1504.3 us | 1.075x | 3671.0 us | 3760.4 us | 1.024x |
| 512 | 4565.0 us | 4734.0 us | 1.037x | 6815.0 us | 6990.0 us | 1.026x |
| 1024 | 5666.0 us | 5956.0 us | 1.051x | 7941.0 us | 8216.0 us | 1.035x |
| 2048 | 10850.0 us | 13787.0 us | 1.271x | 13125.0 us | 16049.0 us | 1.223x |
| 4096 | 23474.0 us | 24356.0 us | 1.038x | 25743.0 us | 26612.0 us | 1.034x |
| 8192 | 46233.0 us | 50588.0 us | 1.094x | 48532.0 us | 52848.0 us | 1.089x |

Interpretation: with the current in-repo W8A8 baseline, compact NVFP4 is within about 2-4% max-rank at M256/M512/M1024/M4096, about 9% at M8192, 11-13% at M32-M128, and still about 22% slower at M2048.


### Latest Compact NVFP4 vs PR323 W8A8 Snapshot (2026-06-12)

This is the latest true compact NVFP4 result after the M512 default retune (`DG_SM90_MOE_L2_DUAL_ACCUM=0` for M512). Use the same batch list when comparing A/B results, because the benchmark generates different routing/recv distributions for different batch-list orders.

Command shape:

```bash
python3 tests/bench_nvfp4_mega_moe_sm90.py \
  --num-processes 8 \
  --batches 512 1024 2048 4096 8192 \
  --num-tests 50 \
  --weight-mode compact
```

| tokens | compact NVFP4 | PR323 W8A8 reference | compact/W8A8 | delta |
|---:|---:|---:|---:|---:|
| 512 | 2223.2 us | 2320.0 us | 0.958x | 4.2% faster |
| 1024 | 3780.0 us | 3518.0 us | 1.074x | 7.4% slower |
| 2048 | 6482.0 us | 6274.0 us | 1.033x | 3.3% slower |
| 4096 | 12080.0 us | 11402.0 us | 1.059x | 5.9% slower |
| 8192 | 23123.0 us | 21904.0 us | 1.056x | 5.6% slower |

Latest correctness smoke:
- PASS for M512/M1024/M2048/M4096/M8192 at weight_scale=0.05 with fp8-bridge compact reference and strict cosine/norm thresholds.
- M512 additionally PASSed weight_scale=0.001/0.05/1.0.

Interpretation: true compact NVFP4 now beats the PR323 W8A8 reference at M512 in the comparable sweep, while M1024-M8192 remain about 3-7% slower. The old table below is kept only as history.


### Superseded Compact NVFP4 Half-Time Table

Latest retained compact 50-run compared to same-shape PR323 W8A8 30-run:

| tokens | W8A8 PR323 | compact NVFP4 | compact/W8A8 | delta |
|---:|---:|---:|---:|---:|
| 32 | 923.9 us | 1229.3 us | 1.331x | 33.1% slower |
| 64 | 869.9 us | 1199.7 us | 1.379x | 37.9% slower |
| 128 | 837.1 us | 1199.6 us | 1.433x | 43.3% slower |
| 256 | 1460.0 us | 1561.8 us | 1.070x | 7.0% slower |
| 512 | 2320.0 us | 2401.0 us | 1.035x | 3.5% slower |
| 1024 | 3518.0 us | 3803.0 us | 1.081x | 8.1% slower |
| 2048 | 6274.0 us | 6719.0 us | 1.071x | 7.1% slower |
| 4096 | 11402.0 us | 12212.0 us | 1.071x | 7.1% slower |
| 8192 | 21904.0 us | 23657.0 us | 1.080x | 8.0% slower |

Interpretation: the compact online-NVFP4 kernel is still slower than W8A8 on all measured sizes, but the BM128/default-heuristic path plus L2 no-dispatch for M4096/M8192 closes the M256-M8192 gap to roughly 3.5-8.1%.

### Optional FP8-shadow latency path

Primary full-sweep 50-run comparison:

| tokens | W8A8 PR323 | NVFP4 fp8-shadow + unit-scale | delta |
|---:|---:|---:|---:|
| 32 | 809.5 us | 789.7 us | 2.4% faster |
| 64 | 842.5 us | 819.7 us | 2.7% faster |
| 128 | 885.5 us | 850.3 us | 4.0% faster |
| 256 | 1448.0 us | 1456.0 us | 0.6% slower |
| 512 | 2223.0 us | 2212.0 us | 0.5% faster |
| 1024 | 3551.0 us | 3524.0 us | 0.8% faster |
| 2048 | 6210.0 us | 6235.0 us | 0.4% slower |
| 4096 | 11366.0 us | 11353.0 us | 0.1% faster |
| 8192 | 21935.0 us | 21887.0 us | 0.2% faster |

Interpretation: shadow mode is W8A8 parity overall and often faster on M32/M64/M128, but it should not be reported as compact NVFP4 kernel performance.