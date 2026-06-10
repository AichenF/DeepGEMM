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
- End-to-end SM90 NVFP4 MegaMoE correctness for default `M=32`, `M=64`, `M=128`, `M=256`, `M=512`, `M=1024`, `M=2048`, `M=4096`, and `M=8192` cases.
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
- Default path, `M=64`: `cosine_min=0.9996`, `cosine_mean=0.9996`, `norm_ratio=0.9998`, `max_abs_diff=1.0569e+01`, `mean_abs_diff=1.0643e+00`
- Default path, `M=128`: `cosine_min=0.9996`, `cosine_mean=0.9996`, `norm_ratio=0.9997`, `max_abs_diff=1.2675e+01`, `mean_abs_diff=1.1399e+00`
- Default path, `M=256`: `cosine_min=0.9996`, `cosine_mean=0.9996`, `norm_ratio=0.9996`, `max_abs_diff=1.1957e+01`, `mean_abs_diff=1.0920e+00`
- Default path, `M=512`: `cosine_min=0.9996`, `cosine_mean=0.9996`, `norm_ratio=0.9995`, `max_abs_diff=1.4097e+01`, `mean_abs_diff=1.1205e+00`
- Default path, `M=1024`: `cosine_min=0.9996`, `cosine_mean=0.9996`, `norm_ratio=0.9994`, `max_abs_diff=1.4297e+01`, `mean_abs_diff=1.1184e+00`
- Default path, `M=2048`: `cosine_min=0.9995`, `cosine_mean=0.9996`, `norm_ratio=0.9995`, `max_abs_diff=1.6218e+01`, `mean_abs_diff=1.1196e+00`
- Default path, `M=4096`: `cosine_min=0.9995`, `cosine_mean=0.9996`, `norm_ratio=0.9995`, `max_abs_diff=1.6195e+01`, `mean_abs_diff=1.1162e+00`
- Default path, `M=8192`: `cosine_min=0.9995`, `cosine_mean=0.9996`, `norm_ratio=0.9995`, `max_abs_diff=1.7863e+01`, `mean_abs_diff=1.1133e+00`

Current default NVFP4 uses BM64 for small token counts, fused single-kernel execution for `M=128`, and the BM128/2-epilogue-WG split L1/L2 shape for `M=256`, `M=512`, `M=1024`, `M=2048`, `M=4096`, and `M=8192`. The BM128 path uses a wider loader-dequant non-epilogue register budget (`64` regs) with a reduced epilogue budget (`192` regs), which fixed the previous non-finite output. Set `DG_SM90_NVFP4_BM128_HEURISTIC=0` to force the BM64 fallback; with that fallback, BM64 `M=4096` still uses fused mode.

Additional BM128 coverage passed for `M=256/512/1024/2048/4096/8192` with `weight_scale=1.0`, `0.05`, and `0.001`; the lowest observed cosine was `0.9985` at `M=2048`, `weight_scale=1.0`, with finite output and norm ratio `0.9991`. For the newly enabled large-M defaults, `M=4096/8192` both passed all three weight scales with lowest cosine `0.9987`.

Latest same-container benchmark, `hidden=7168`, `intermediate_hidden=2048`, `num_experts=256`, `topk=8`, `num_processes=8`, `num_tests=20`:

| tokens | NVFP4 us | W8A8 us | NVFP4/W8A8 |
|---:|---:|---:|---:|
| 32 | 1294.7 | 1252.9 | 1.03x |
| 64 | 1289.4 | 898.7 | 1.43x |
| 128 | 1266.0 | 939.5 | 1.35x |
| 256 | 1640.3 | 1275.2 | 1.29x |
| 512 | 2646.0 | 2297.0 | 1.15x |
| 1024 | 4077.0 | 3852.0 | 1.06x |
| 2048 | 6892.0 | 7213.0 | 0.96x |
| 4096 | 12984.0 | 13617.0 | 0.95x |
| 8192 | 25150.0 | 26647.0 | 0.94x |

Current default NVFP4 is now close to W8A8 for `M=1024`, and faster for `M=2048/4096/8192`. `M=512` is within about `15%`. The remaining clear gaps are `M=64`, `M=128`, and `M=256`.

## Removed AKO Logs

`HINTS.md` and `ITERATIONS.md` were AKO working files: `HINTS.md` stored optimization constraints, and `ITERATIONS.md` stored detailed per-iteration experiment logs. They are useful during kernel search, but they are not required for the final branch review, so this branch keeps the shorter summary above instead.
