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
- End-to-end SM90 NVFP4 MegaMoE correctness for default `M=32` and `M=256` cases.
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
- Default path, `M=256`: `cosine_min=0.9996`, `cosine_mean=0.9996`, `norm_ratio=0.9996`, `max_abs_diff=1.1957e+01`, `mean_abs_diff=1.0920e+00`
- Large-M heuristic path, `M=2048`: `cosine_min=0.9995`, `cosine_mean=0.9996`, `norm_ratio=0.9995`, `max_abs_diff=1.6218e+01`, `mean_abs_diff=1.1196e+00`

Current default NVFP4 uses the BM64 path for `num_tokens < 2048` and a BM128/2-epilogue-WG path for `num_tokens >= 2048`. The BM128 path keeps true packed NVFP4 runtime dequant, but amortizes each B-tile dequant over a larger M tile.

Latest same-container benchmark, `hidden=7168`, `intermediate_hidden=2048`, `num_experts=256`, `topk=8`, `num_processes=8`, `num_tests=20`:

| tokens | NVFP4 us | W8A8 us | NVFP4/W8A8 |
|---:|---:|---:|---:|
| 32 | 1407.0 | 869.5 | 1.62x |
| 64 | 1261.9 | 1023.4 | 1.23x |
| 128 | 1272.7 | 918.1 | 1.39x |
| 256 | 1831.9 | 1308.5 | 1.40x |
| 512 | 3365.0 | 2458.4 | 1.37x |
| 1024 | 5475.0 | 3792.0 | 1.44x |
| 2048 | 10028.0 | 7154.0 | 1.40x |
| 4096 | 18281.0 | 13552.0 | 1.35x |
| 8192 | 35155.0 | 26660.0 | 1.32x |

Compared with the prior NVFP4 BM64 baseline in this branch, the large-M heuristic improves the main large batches approximately from `10205 -> 10028 us` at M=2048, `19459 -> 18281 us` at M=4096, and `38335 -> 35155 us` at M=8192 in the latest same-run comparison.

## Removed AKO Logs

`HINTS.md` and `ITERATIONS.md` were AKO working files: `HINTS.md` stored optimization constraints, and `ITERATIONS.md` stored detailed per-iteration experiment logs. They are useful during kernel search, but they are not required for the final branch review, so this branch keeps the shorter summary above instead.
