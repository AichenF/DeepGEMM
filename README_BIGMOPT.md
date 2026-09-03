# SM90 NVFP4 MegaMoE big M opt — splitT1 + L2 scatter + 2 stream

Base: `megamoe_nvfp4_dev` @ `c277f44`.

One weight copy, one entry point. Each forward picks the kernel family from the
routed load this rank actually sees:

```
rho = M * topk / local_experts

rho <= 192  ->  BN256 fused
rho  > 192  ->  BN128 split L1 + L2
```

Decided in the C++ entry from this call's `y.size(0)`, so decode, small prefill
and large prefill can take different families in one serving process. Model name
and hidden size no longer participate. Both families read the same packed-B
bytes; `block_n` selects only the retained scale-metadata view.

`kernel_family="auto"|"fused"|"split"` — production must use `"auto"`, the other
two are diagnostics.

## What changed



### Large M (split) — the bigMopt work

Always enabled together, emitted as compile-time defines into the JIT key so no
environment variable can select an unvalidated variant.


| Name           | Change                                                                                                                                                                                                                                                                                                                                    | Where                                                                                                |
| -------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ---------------------------------------------------------------------------------------------------- |
| **splitT1**    | Emit `DG_NVLINK_BARRIER_TRAP_ONLY_TIMEOUT` for the split JIT source, replacing the 7-arg device `printf` in the NVLink barrier's timeout branch with a bare trap. That branch cost a vararg buffer and register pressure inside the hot cross-rank spin loop. **dev already did this for fused; split did not.** No compute logic changes | `impls/sm90_nvfp4_mega_moe.hpp:129`; branch at `comm/barrier.cuh:63-79` (unchanged)                  |
| **2 stream**   | L1 dispatch-dequant half-stream mode4 + physical-warp remap. **New code** — dev has no half-stream machinery at all                                                                                                                                                                                                                       | `impls/..._split_l1_body.inl`, 983 -> 1679 lines; defines at `impls/sm90_nvfp4_mega_moe.hpp:120-127` |
| **L2 scatter** | L2 (down) stages one warp-private 8-row half tile in shared memory so each destination row leaves as a single 256-byte burst, instead of eight scattered 16-byte requests each billed as a 32-byte sector. **New code**                                                                                                                   | `impls/..._split_l2_body.inl`, 659 -> 751 lines                                                      |


Risk is not uniform: splitT1 touches no compute, while 2 stream and L2 scatter
add ~790 lines of new device code in dequant timing and shared-memory layout —
the paths where an error is silent.

### Small M (fused)

The fused device body is **byte-identical to dev**, and the dequant functions it
calls (`dequant_braided_quad*`, `dequant_braided_selector_word`,
`dequant_smem_b_from_packed_braided_lut_window`) are **unchanged** — the +295
lines in `mode2_dequant.cuh` are split-only functions that fused never calls.

Two host-side lines differ from dev:


| Location                         | Change                                                                          | Effect                                                                                                                              |
| -------------------------------- | ------------------------------------------------------------------------------- | ----------------------------------------------------------------------------------------------------------------------------------- |
| `heuristics/..._small_m.hpp:130` | `use_mode2_lop3_decoder` `false -> true`, **only in the** `rho <= 6` **bucket** | different decode path at very small M; both decoders are lossless, so output should be bit-exact against dev — **not yet verified** |
| `impls/..._small_m.hpp:287`      | fused launch passes `launch_pdl=false` (dev's `LaunchArgs` defaults to `true`)  | launch behaviour only, no numerical effect                                                                                          |


The other five tuning buckets are unchanged.

### Weight side

`choose_nvfp4_block_n_for_mega_moe_sm90` drops the `intermediate_hidden >= 3072`
special case (cutoff 190) and always uses 192. This moves Pro's fused/split
crossover from M=1520 to M=1536; Flash and MiMo are unaffected.

## Performance

Microseconds, 8-GPU rank-MAX median.
`dev` = `AichenF/DeepGEMM` @ `c277f44` · `W8A8` = `deepseek-ai/DeepGEMM` PR #383
@ `bc4f33a`.

bigMopt changes only the split path, so the two ranges are different claims.

### Small M — fused (`rho <= 192`)

bigMopt runs dev's fused body byte-for-byte here, so this is a precision comparison,
not a bigMopt speedup claim. Negative = NVFP4 faster.


| Model | M    | NVFP4 fused (dev = bigMopt) | W8A8   | W8A8 vs NVFP4 |
| ----- | ---- | ---------------------- | ------ | ------------- |
| Flash | 8    | 205.2                  | 235.9  | **-15.0%**    |
| Flash | 16   | 215.5                  | 262.2  | **-21.7%**    |
| Flash | 32   | 237.8                  | 256.7  | **-7.9%**     |
| Flash | 64   | 245.2                  | 267.9  | **-9.3%**     |
| Flash | 128  | 244.2                  | 276.7  | **-13.3%**    |
| Flash | 256  | 289.7                  | 290.2  | -0.2%         |
| Flash | 512  | 453.1                  | 414.7  | +8.5%         |
| Flash | 1024 | 747.4                  | 580.2  | +22.4%        |
| Pro   | 8    | 576.8                  | 699.0  | **-21.2%**    |
| Pro   | 16   | 710.9                  | 798.0  | **-12.3%**    |
| Pro   | 32   | 724.5                  | 842.3  | **-16.3%**    |
| Pro   | 64   | 753.2                  | 856.7  | **-13.7%**    |
| Pro   | 128  | 839.5                  | 866.6  | **-3.2%**     |
| Pro   | 256  | 822.6                  | 877.8  | **-6.7%**     |
| Pro   | 512  | 1234.5                 | 1000.5 | +19.0%        |
| Pro   | 1024 | 2006.0                 | 1367.8 | +31.8%        |


Geometric mean: **M<=256 (12 pts) NVFP4 wins by 11.55%**; **M>=512 (4 pts) W8A8
wins by 20.84%**. The crossover sits between M=256 and M=512 — NVFP4 fused is
the faster kernel for decode and small prefill. The 16-point average (W8A8
+2.38%) is a crossover artifact; do not quote it alone.

Caveat: PR #383's FP8 path is always split L1+L2 while NVFP4 here is fused, so
these rows compare two kernel structures, not one algorithm at two precisions.

### Large M — split (`rho > 192`)


| Model | M    | dev    | bigMopt     | W8A8   | bigMopt vs dev   | bigMopt vs W8A8 |
| ----- | ---- | ------ | ------ | ------ | ----------- | ---------- |
| Flash | 2048 | 1234.7 | 1103.2 | 957.6  | **+10.65%** | -15.20%    |
| Flash | 4096 | 2173.6 | 1882.8 | 1698.2 | **+13.38%** | -10.87%    |
| Flash | 8192 | 4170.4 | 3602.9 | 3239.6 | **+13.61%** | -11.21%    |
| Pro   | 2048 | 2990.6 | 2765.3 | 2392.3 | **+7.54%**  | -15.59%    |
| Pro   | 4096 | 5304.1 | 4859.4 | 4140.5 | **+8.38%**  | -17.36%    |
| Pro   | 8192 | 9844.4 | 9021.8 | 7841.7 | **+8.36%**  | -15.05%    |


`bigMopt vs dev` positive = bigMopt faster. `bigMopt vs W8A8` negative = bigMopt slower.

Geometric mean vs dev: Flash **-12.56%**, Pro **-8.09%**, **all 6 points
-10.35% latency (**`1.1155x`**)**. Versus W8A8, bigMopt is still slower at every point,
geometric mean +14.19%.

bigMopt is a clear large-M improvement over dev. It does not close the gap to W8A8.

### Protocol

Small-M table: `bench_kineto`, 8 GB L2 flush, `num_tests=20` internal mean is
one observation, median of 50 observations for M<=128 and 20 for M>=256, same
rule for every arm.

Large-M table: the protocol behind the `bigMopt` column was not supplied and no raw
logs are archived here, so it is not reproducible from this repository. Its
`dev` and `W8A8` columns agree with the small-M campaign to within 1.2%.

## Validation status

Done: selector boundary unit test (M=768 fused / M=769 split under R1 shape);  
packed-B from BN128 and BN256 metadata byte-identical; generated split L1 JIT  
source carries both mode4 and remap macros with a separate L2 cubin; M=192/193  
forced cross-validation both directions, token cosine min 0.9980-0.9985; 8-GPU  
R1-shape smoke at M=128 fused and M=1024 split, all ranks complete; 238-call  
multi-round reuse all finite; formal component measurement over 8 M points with  
40x8 outputs bit-exact against their pre-timing references.

## Usage

```python
deep_gemm.nvfp4_mega_moe(..., kernel_family="auto", family_threshold=192)
```

Weights are prepared once at load time with
`transform_nvfp4_weights_for_mega_moe_sm90` and cached; both families share the
result.