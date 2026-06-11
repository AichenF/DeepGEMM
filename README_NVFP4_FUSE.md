# NVFP4 SM90 MegaMoE Notes

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

### Compact NVFP4-input kernel

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
