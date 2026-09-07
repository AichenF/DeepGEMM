# SM120 MegaMoE — M64xN32 two-stage residency schedule (two variants)

Native `sm_120a` CUDA exports of an optimized single-entry MegaMoE kernel for
the DeepSeek V4 Flash expert geometry. One persistent cooperative kernel covers
the whole distributed path — NCCL GIN dispatch, destination task construction,
W1, fused SwiGLU and FP8 requantization, W2, BF16 return (lossless block
codec), weighted combine and acknowledgement — over a two-slot `[0, 1, 0]`
result lifecycle. Both GEMM phases run one runtime-scheduled M64xN32xK128 tile
family with four contiguous M16xN32 math-warp owners, four loader-warp roles
and one elected TMA issuer.

This directory ships **two variants of the same kernel**, which differ only in
how the model's single shared expert is handled:

| file | variant | shared expert | output |
| --- | --- | --- | --- |
| `cake_sm120_megamoe_m64n32_stage2_residency_fp8shared.cu` | `fp8shared` | computed in-kernel on the checkpoint's **FP8 e4m3 weights** (exact) | full MoE output: routed top-6 sum + shared expert, BF16 |
| `cake_sm120_megamoe_m64n32_stage2_residency_noshared.cu` | `noshared` | **not computed in-kernel**; the caller adds it | routed top-6 sum only, BF16 |

Both are CAKE code-generator exports with no runtime dependency on CAKE or
Python; each translation unit only needs `nccl_device.h` and the CUDA BF16/FP8
headers. Both replace the single kernel previously in this directory (sha
`0711d403…`, "codec v2"), whose shared expert was computed on a lossy MXFP4
requantization of the checkpoint weights — see "Why two variants" below.

## Geometry as exported (both variants)

| Property | Value |
| --- | --- |
| hidden / intermediate / output | 4096 / 2048 / 4096 |
| routed experts | 256 global = 8 ranks x 32 local, top-k 6, routing weights normalized and scaled by the caller |
| routed-expert precision | MXFP8 E4M3 activations, MXFP4 E2M1 weights, UE8M0 K32 scales, W4A8 GEMM, FP32 accumulation |
| shared expert (`fp8shared`) | FP8 E4M3 weights as stored in the checkpoint (`ffn.shared_experts.{w1,w3,w2}`, 128x128 E8M0 block scales, replicated exactly into UE8M0 K32 scales), MXFP8 activations, W8A8 GEMM |
| shared expert (`noshared`) | none in-kernel |
| activation | SwiGLU with `swiglu_limit = 10` (up clamped to ±10, gate clamped at 10), routing weight applied before the W2 requantization |
| rank capacity | 8 physical ranks, 2 result/dispatch ring slots |
| token capacity | 8192 rows per rank |
| task granularity | 128 rows per task |
| grid | one service/coordinator CTA plus worker CTAs, 384 threads each; the worker count is resolved at launch (110 CTAs on a 110-SM part) |
| shared memory | 101376 B per CTA |

The intermediate size is 2048, as reported by the released
`DeepSeek-V4-Flash-0731` `config.json` (`moe_intermediate_size = 2048`) and
confirmed by the safetensors expert shapes. All 43 decoder layers and the 3
MTP layers share this geometry; the first three layers only differ in how the
caller produces the top-k indices (hash routing), which is outside the kernel.

## Why two variants

The released checkpoint stores the routed experts as MXFP4 (E2M1 + UE8M0 K32
scales, consumed here byte for byte) but the **shared expert as FP8 e4m3 with
128x128 block scales**. The previous kernel in this directory requantized the
shared expert to MXFP4 (relative L2 error 0.12 per weight matrix), which put
the MoE output about **10 % rel-L2 away from the released model** while the
kernel itself was exact against its own oracle. Upstream DeepGEMM's sm100
MegaMoE keeps the shared expert FP8xFP8. The two variants remove that error in
the two ways a deployment can choose from:

- `fp8shared` runs the shared expert in-kernel on the exact FP8 weights (two
  extra plain-u8 TMA descriptors; shared tiles use `kind::mxf8f6f4`
  e4m3 x e4m3 instead of e4m3 x e2m1 through a per-tile warp-uniform branch
  into a duplicated K loop; the SMEM footprint per B tile is unchanged, the TMA
  transaction grows from 8 KB to 16 KB).
- `noshared` removes the shared expert from the kernel entirely (no pool tail,
  no prepended shared tasks, no `shared_out`, no combine add) for stacks that
  compute the shared expert elsewhere (e.g. fused with other dense GEMMs or
  overlapped on another stream); it also raises the R512 dispatch chunk floor
  from 96 to 64 records, which is the measured optimum once the shared tasks no
  longer fill the dispatch front.

## Fidelity against the released weights

Measured on layer 20 of `DeepSeek-V4-Flash-0731` (real routed and shared
weights, real gate routing, 8 x 2048 synthetic Gaussian tokens) with exact
FP32 references:

| output compared with … | previous kernel (MXFP4 shared) | `fp8shared` | `noshared` + external FP8 shared expert |
| --- | --- | --- | --- |
| the kernel's own quantization recipe, computed exactly in FP32 | 6.2e-5 | 1.2e-4 (99.9 % of BF16 values bit-exact) | 5.8e-5 (routed part; 99.98 % bit-exact) |
| the reference `inference/model.py` recipe (per-128 activation scales) | 0.1015 | 0.0161 | 0.0142 |
| an unquantized reference (FP32 activations, exact weights) | 0.1086 | 0.0447 | 0.0447 |

Relative L2 throughout. The residual 0.045 against the unquantized reference is
the FP8 activation-quantization noise of this layer: the reference recipe of
`inference/model.py` sits at the same 0.0447, so the kernels are at the
precision floor of any FP8-activation implementation of this model. The 1.4–1.6
% between the two recipes is two independent rounding paths (per-32 vs
per-128 activation scales, exp2 sigmoid, no BF16 rounding before requant), not
a fidelity gap. Weights are consumed exactly in both variants: all 256 routed
experts are byte-equal to the safetensors and the FP8 shared tensors decode to
the checkpoint exactly. For reference, requantizing the shared expert to MXFP4
— what the previous kernel did — costs 0.122 / 0.123 / 0.127 relative L2 on
`w1` / `w3` / `w2`.

The `noshared` fidelity column was dumped from a build whose routed path is
identical to the shipped `noshared` kernel but which carries a different
generator seed; the routed numerics it reports therefore apply unchanged, and
the shipped kernel's own correctness ladder is the one reported below.

## Kernel resources

`nvcc -cubin --generate-code=arch=compute_120a,code=sm_120a -std=c++17 -O3
--resource-usage`, CUDA 13.0:

| Metric | `fp8shared` | `noshared` | previous kernel |
| --- | --- | --- | --- |
| registers | 164 | 166 | 166 |
| spill stores / spill loads | 0 / 0 | 0 / 0 | 0 / 0 |
| stack frame | 8 B | 8 B | 8 B |
| barriers | 16 | 16 | 16 |
| shared memory | 101376 B | 101376 B | 101376 B |
| cubin size | 383336 B | 354288 B | 366328 B |

## Measured performance

Measurement environment: 8 GPUs of compute capability 12.0 (`sm_120a`, 110 SMs,
73 GB each), no NVLink — inter-GPU transport is NCCL GIN over RDMA NICs
(`NCCL_GIN_TYPE=3`, GDAKI, NCCL 2.30.7 pinned); driver 580.95.05, CUDA 13.0.
All eight ranks participate in every measurement.

Timing protocol: CUPTI concurrent-kernel activity envelope of the single
full-chain entry, cold L2, one epoch per sample, reduced with a **max across
ranks** before any statistic. Each arm runs for 1000 ms; the reported value is
the slowest-rank median. Arms alternate legs (A/B/B/A) inside one process so
both arms see the same clocks, the same fabric and the same allocation. A = the
previous kernel in this directory (sha `0711d403…`), B = the variant. Two
independent sessions, 3 pairs per shape per session, balanced top-6 routing.
The A/A noise floor of this protocol is ±1.5 %.

Each ratio cell below is the session median followed by its three individual
pairs; the absolute A and B columns are the session-2 medians.

### `fp8shared` vs the previous kernel

| active rows / rank | A (ms) | B (ms) | B/A session 1 | B/A session 2 |
| --- | --- | --- | --- | --- |
| 512 | 0.870 | 0.877 | **1.010** (1.006 / 1.010 / 1.010) | **1.007** (1.006 / 1.007 / 1.011) |
| 2048 | 2.731 | 2.733 | **1.000** (0.999 / 1.000 / 1.001) | **1.000** (1.000 / 1.000 / 1.001) |
| 4096 | 5.212 | 5.211 | **1.002** (1.000 / 1.002 / 1.006) | **0.999** (0.998 / 0.999 / 1.001) |
| 8192 | 10.349 | 10.361 | **0.997** (0.994 / 0.997 / 1.002) | **1.002** (0.998 / 1.002 / 1.002) |

Restoring the shared expert's FP8 precision costs nothing measurable. The
+0.7–1.0 % at R512 (inside the noise floor, but same-signed in 6/6 pairs) is
the doubled shared-expert weight stream (24 MB instead of 12 MB per epoch, 16 KB
instead of 8 KB per shared B tile) on the one shape whose critical path is the
compute window; in-kernel phase stamps show the shared W1 phase growing from
78 to 79 us (synthetic) / 80 to 85 us (real weights) and the tile stream ending
4 us later. At 2048 rows and above it is hidden under the wire-bound window.

### `noshared` vs the previous kernel

| active rows / rank | A (ms) | B (ms) | B/A session 1 | B/A session 2 |
| --- | --- | --- | --- | --- |
| 512 | 0.869 | 0.831 | **0.958** (0.957 / 0.958 / 0.961) | **0.958** (0.955 / 0.958 / 0.965) |
| 2048 | 2.734 | 2.711 | **0.992** (0.992 / 0.992 / 0.993) | **0.992** (0.989 / 0.992 / 0.993) |
| 4096 | 5.209 | 5.171 | **0.994** (0.994 / 0.994 / 0.996) | **0.993** (0.991 / 0.993 / 0.998) |
| 8192 | 10.356 | 10.287 | **0.991** (0.975 / 0.991 / 0.991) | **0.994** (0.993 / 0.994 / 0.994) |

R512 is compute-window bound, so removing the shared tasks (about 11 % of the
tiles) plus the smaller first dispatch chunk saves 4 %; the wire-bound shapes
gain a consistent 0.6–0.9 % from the shorter W2 epilogue and the combine no
longer reading 8 KB per token. **These numbers do not include the shared expert
itself.** Computed as a standalone FP8 dense GEMM pair on the same GPU it costs
roughly 0.05 / 0.2–0.3 / 0.5 / 1 ms at 512 / 2048 / 4096 / 8192 rows, so
`noshared` is a net win only where the shared expert can be fused with other
work or overlapped; otherwise `fp8shared` is the precision-correct choice at
parity with the previous kernel.

The dispatch-chunk-floor sweep behind the `noshared` R512 figure was
single-peaked: 96 → 0.970, **64 → 0.958**, 48 → 0.968, 32 → 0.974 (B/A at R512,
same protocol); the other shapes were insensitive.

### Against the starting point of the optimization campaign

Same host, same protocol, cold-L2 CUPTI medians (starting kernel: 3 solo runs;
variants: session-2 B legs above):

| active rows / rank | starting kernel (ms) | `fp8shared` (ms) | `noshared` (ms) |
| --- | --- | --- | --- |
| 512 | 2.086 | 0.877 (2.38x) | 0.831 (2.51x) |
| 2048 | 4.414 | 2.733 (1.62x) | 2.711 (1.63x) |
| 4096 | 8.104 | 5.211 (1.56x) | 5.171 (1.57x) |
| 8192 | 15.671 | 10.361 (1.51x) | 10.287 (1.52x) |

### Wire budget (unchanged from the previous kernel)

The return path is bandwidth-bound at every shape above 512 rows. A returned
BF16 row is encoded as 64 blocks of 64 values; each block carries a 2-byte
header (block maximum exponent) plus 64 12-bit codes, and a block whose
internal exponent span exceeds 15 falls back to a raw 128-byte copy in a
bounded per-row overflow area (16 blocks per row; measured raw rate on real
weights 0.29 %). A row that cannot be represented exactly raises a loud
`protocol_error` and fails the epoch instead of silently producing a wrong
result.

| Quantity | Raw BF16 | Codec v2 |
| --- | --- | --- |
| returned row | 8192 B | 6272 B (-23.4 %) |
| result slot pitch | 8196 B | 8320 B |
| per-rank wire egress at R512 / R2048 / R4096 / R8192 | 32.3 / 129.3 / 258.6 / 517.1 MB | 26.9 / 107.5 / 215 / 430 MB |

Against a 46 GB/s per-rank egress floor computed on the post-codec wire bytes,
both variants attain about 68 / 87 / 90 / 91 % at R512 / R2048 / R4096 / R8192
(the `noshared` R512 figure is 4 % better than that of `fp8shared`, whose R512
is compute-window bound).

## Correctness

Both variants produce bit-exact BF16 against their validators (which were
updated with the semantics: the real-weight oracle's shared expert is the exact
FP8 checkpoint weights for `fp8shared` and absent for `noshared`); zero
mismatches and zero protocol errors across the whole ladder:

- a short watchdogged smoke run (5 s watchdog, since both variants change
  control flow in the tile loop / task table),
- four synthetic shapes (512 / 2048 / 4096 / 8192 rows) x 20 epochs,
- real weights at 512 and 2048 rows (MXFP4 routed experts + FP8 shared expert,
  both from the released `-0731` checkpoint) and the FP8 checkpoint's layer 20
  at 2048 rows, each with the real gate routing.

Per-stage real-weight oracle (exact FP32 dequant): routed W1 / W2 rel-L2 3.9e-5
/ 3.1e-5 (both variants), `fp8shared` shared W1 / W2 vs the exact FP8 weights
3.5e-5 / 5.1e-5, requantization codes and scales bit-exact.

## What the schedule changed

Everything the previous kernel in this directory carried (warp-role split,
parallel front end, loader-warp L2 prefetch, barrier-free epoch start, combine
memory-level parallelism, per-peer in-order return issue with per-chunk combine
gating, W1→W2 interleave distance 4, lossless BF16 return codec v2) is present
unchanged in both variants. On top of it:

- `fp8shared`: two extra plain-u8 2-D TMA descriptors (`W1_B8`, `W2_B8`; box
  128 K x 128 N, 128 B swizzle) carry the shared expert's FP8 weights. The
  loader branches on the task's local-expert index and issues the 16 KB
  transaction for shared tiles; the math warps branch once per tile into a
  duplicated K loop whose B fragments are loaded with `ldmatrix.x2` (E4M3
  metadata) so the MMA is `kind::mxf8f6f4` e4m3 x e4m3 with the same UE8M0
  K32 scales. The shared expert's scales live in the extra segment (index 32)
  of the existing SFB tensors as before; its FP4 weight segment is gone.
- `noshared`: `SHARED_MAX_TASKS = 0` (pool tail, prepended shared tasks, pool
  fill, `shared_out` stores in the W2 epilogue and the combine add are
  removed; three independent shared-task counts in the task audit, the reset
  bound and the launch-time grid estimate are zeroed), `DISPATCH_CHUNK_MIN_RECORDS_R512 = 64`.

## Provenance

| Item | `fp8shared` | `noshared` |
| --- | --- | --- |
| file | `cake_sm120_megamoe_m64n32_stage2_residency_fp8shared.cu` | `cake_sm120_megamoe_m64n32_stage2_residency_noshared.cu` |
| sha256 | `7fd8801736b89db96f90d1ea215fa3bf4712dcc43a81c64ae8185cac98751ae5` | `f1c0c32c560fd3a11c2019f0d484cf79d2b961c6296d35edce09c1f147c90768` |
| size | 212722 B, 3764 lines | 194075 B, 3532 lines |
| generator seed sha256 (first 8) | `fecf4b6b` | `b4cfd42d` |
| entry symbol | `kernel_deepgemm_sm120_megamoe_m64n32_stage2_residency_dispatch` | same |
| launch bounds | 384 threads | 384 threads |

Each `.cu` is the code generator's output byte for byte; no hand edits were
applied after generation, so each hash above is reproducible from the generator
and pins the exact artifact every number in this report was measured on.

## Not included here — host integration notes

This directory carries the device translation units and this report only; it
does not ship a host driver, `build.sh` or runner, so it does not build through
the harness in `dsv4_flash/`. Relative to the previous kernel's launch ABI (same
argument order otherwise):

- `fp8shared` adds two `LoomTensorMap const*` arguments, `W1_B8` right after
  `W1_B` and `W2_B8` right after `W2_B`. `W1_B8` is a 2-D u8 tensor map over
  the shared expert's W1 weights in the kernel's physical row order (4096 rows =
  gate|up 8-interleaved: physical row `p` of each 16-row group holds gate row
  `g*8+l` for lanes 0–7 and up row `g*8+l` for lanes 8–15; 4096 K bytes per
  row), `W2_B8` over its W2 weights (4096 rows x 2048 K bytes); box
  `(128, 128)`, 128 B swizzle, no OOB fill. Their UE8M0 K32 scales are written
  into segment 32 of `W1_SFB` / `W2_SFB` as packed `u32` words (the 128x128
  block's E8M0 exponent byte replicated to the four K32 groups it covers).
- `noshared` drops the `__nv_bfloat16* shared_out` argument (between
  `final_output` and `pool_fp8_u32`); `final_output` is the routed top-6 sum.
  The weight/scale tensors need only the 32 local-expert segments. The caller
  computes the shared expert (FP8 weights + FP8 activations, per-32 or per-128
  UE8M0 scales — both sit at the same precision floor, see above) and adds it
  to `final_output`.

The other adapter items from the previous README still apply: the encoded
result-slot layout and pitch, the per-(source, chunk) dispatch and return
signal allocation, and the launch-time grid resolution. The numbers in this
report were produced by the CAKE harness the kernels were developed in, not by
`dsv4_flash/run_perf_abba.py`; they use the same max-across-ranks CUPTI
activity-envelope definition, and reproducing them under the host in
`dsv4_flash/` is the natural follow-up.
