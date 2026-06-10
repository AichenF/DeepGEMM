# SM90 MegaMoE NVFP4 Notes

This branch adds a true packed NVFP4 runtime-dequant path for the SM90 MegaMoE kernel. Run validation and benchmarks inside the `mega_moe_box` container, not directly on the host.

Before running GPU tests, check that no other compute process is active:

```bash
nvidia-smi --query-compute-apps=gpu_uuid,pid,process_name,used_gpu_memory --format=csv,noheader,nounits
```

## Correctness

Use this test for the main NVFP4 correctness gate:

```bash
docker exec -e TORCH_CUDA_ARCH_LIST=9.0a mega_moe_box bash -lc \
  "cd /root/fac/megamoe/DeepGEMM_megamoe_nvfp4 && \
   python3 tests/test_nvfp4_mega_moe_sm90_correctness.py"
```

`tests/test_nvfp4_mega_moe_sm90_correctness.py` covers:

- Python NVFP4 dequant and layout-transform checks.
- CUDA byte-level LUT/dequant checks for `128 x 16` UE4M3 scale-code / E2M1 nibble combinations.
- End-to-end SM90 NVFP4 MegaMoE correctness for default `M=32`, `M=128`, `M=256`, `M=512`, `M=1024`, `M=2048`, and `M=4096` cases.
- Finite-output, cosine similarity, norm-ratio, max-abs-diff, and mean-abs-diff reporting.

For a broader guarded run, use:

```bash
docker exec mega_moe_box bash -lc \
  "cd /root/fac/megamoe/DeepGEMM_megamoe_nvfp4 && \
   scripts/run_nvfp4_gates_when_idle.sh"
```

That script performs GPU-idle preflight, a small correctness sweep over several weight scales, default correctness, NVFP4 benchmark, W8A8 benchmark, and latency-ratio parsing.

## Benchmarks

NVFP4 true packed runtime-dequant benchmark:

```bash
docker exec mega_moe_box bash -lc \
  "cd /root/fac/megamoe/DeepGEMM_megamoe_nvfp4 && \
   python3 tests/bench_nvfp4_mega_moe_sm90.py --num-processes 8 --batches 32 --num-tests 5"
```

W8A8 / FP8 baseline benchmark:

```bash
docker exec mega_moe_box bash -lc \
  "cd /root/fac/megamoe/DeepGEMM_megamoe_nvfp4 && \
   python3 tests/bench_mega_moe_sm90.py --num-processes 8 --batches 32 --num-tests 5"
```

## Latest Result

Latest container correctness gates on `mega_moe_box` passed with:

- `NVFP4 dequant unit test: PASS`
- `NVFP4 CUDA dequant LUT unit test: PASS`
- Default path, `M=32`: `cosine_min=0.9996`, `cosine_mean=0.9997`, `norm_ratio=0.9997`, `max_abs_diff=1.0846e+01`, `mean_abs_diff=1.1152e+00`
- Default path, `M=128`: `cosine_min=0.9996`, `cosine_mean=0.9996`, `norm_ratio=0.9997`, `max_abs_diff=1.2675e+01`, `mean_abs_diff=1.1399e+00`
- Default path, `M=256`: `cosine_min=0.9996`, `cosine_mean=0.9996`, `norm_ratio=0.9996`, `max_abs_diff=1.1957e+01`, `mean_abs_diff=1.0920e+00`
- Default path, `M=512`: `cosine_min=0.9996`, `cosine_mean=0.9996`, `norm_ratio=0.9995`, `max_abs_diff=1.4097e+01`, `mean_abs_diff=1.1205e+00`
- Default path, `M=1024`: `cosine_min=0.9996`, `cosine_mean=0.9996`, `norm_ratio=0.9994`, `max_abs_diff=1.4297e+01`, `mean_abs_diff=1.1184e+00`
- Default path, `M=2048`: `cosine_min=0.9995`, `cosine_mean=0.9996`, `norm_ratio=0.9995`, `max_abs_diff=1.6218e+01`, `mean_abs_diff=1.1196e+00`
- Default path, `M=4096`: `cosine_min=0.9995`, `cosine_mean=0.9996`, `norm_ratio=0.9995`, `max_abs_diff=1.6195e+01`, `mean_abs_diff=1.1162e+00`

Current default NVFP4 uses the BM64 shape for all token counts. It uses fused single-kernel execution for `M=128` and `M=4096`; for split L1/L2 cases it disables the L2 dispatch pipeline at `M=2048`. The BM128/2-epilogue-WG shape remains available only as an explicit experiment through `DG_SM90_NVFP4_BM128_HEURISTIC=1` or direct shape overrides, because it reproduced non-finite outputs in large correctness tests.

Latest same-container benchmark, `hidden=7168`, `intermediate_hidden=2048`, `num_experts=256`, `topk=8`, `num_processes=8`, `num_tests=20`:

| tokens | NVFP4 us | W8A8 us | NVFP4/W8A8 |
|---:|---:|---:|---:|
| 32 | 1269.1 | 900.9 | 1.41x |
| 64 | 1314.2 | 946.9 | 1.39x |
| 128 | 1291.0 | 930.2 | 1.39x |
| 256 | 1828.7 | 1299.5 | 1.41x |
| 512 | 3191.0 | 2412.6 | 1.32x |
| 1024 | 5522.0 | 3833.0 | 1.44x |
| 2048 | 10163.0 | 7077.0 | 1.44x |
| 4096 | 19354.0 | 13568.0 | 1.43x |
| 8192 | 38287.0 | 26611.0 | 1.44x |

Current default NVFP4 is correctness-safe but not yet close to W8A8; the remaining gap is roughly `1.32x-1.44x` in this same-container sweep.

## Removed AKO Logs

`HINTS.md` and `ITERATIONS.md` were AKO working files: `HINTS.md` stored optimization constraints, and `ITERATIONS.md` stored detailed per-iteration experiment logs. They are useful during kernel search, but they are not required for the final branch review, so this branch keeps the shorter summary above instead.
