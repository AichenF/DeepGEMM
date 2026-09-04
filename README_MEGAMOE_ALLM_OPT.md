# SM90 NVFP4 MegaMoE all-M optimization

This implementation combines the accepted small-M portfolio with the merged
`bigMopt` large-M split kernel behind the existing `nvfp4_mega_moe` API.

## Provenance

- Integration base: `AichenF/DeepGEMM:megamoe_nvfp4_dev@dbd995f`
- The base is the upstream merge commit for PR #5 and contains
  `bigMopt@53d943e`.
- Small-M dynamic base: `AichenF/DeepGEMM:megamoe_nvfp4_dev_m@8b59b194`.
- Small-M RS source: archived `dynamic scheduler+RS mode5` snapshot.
- Static-RS source: accepted Kernel Factory candidate 424 (`KF424`).
- `D40` denotes the accepted ten-point Flash/Pro `M=8..128` portfolio assembled
  from those small-M implementations.

## Runtime policy

The existing family decision remains the outer selector:

```text
rho = M * topk / local_experts

rho <= family_threshold (default 192) -> dev-m dynamic fused side
rho >  family_threshold               -> BN128 bigM split side
```

At the default threshold this puts Flash through `M=1024`, Pro through
`M=1536`, and MiMo through `M=1152` on the fused side. The next integer M for
each model selects the split side.

The exact pointwise optimization table is applied only when all of the
following are true:

- SM90 and `132` SMs (validated on NVIDIA H200; the selector keys on SM count);
- eight ranks;
- exact Flash (`H=4096`, `I=2048`, `E=256`, `topk=6`) or Pro (`H=7168`,
  `I=3072`, `E=384`, `topk=6`) geometry;
- `M` is exactly one of `8`, `16`, `32`, `64`, or `128`;
- the outer family selector chose the fused side.

The accepted exact-key table is:

In the implementation labels, SS means shared-memory A and B operands, while
RS means a register A operand and shared-memory B operand for FP8 WGMMA.

| Model | M   | Selected implementation                        |
| ----- | --- | ---------------------------------------------- |
| Flash | 8   | dev-m dynamic + RS mode5                       |
| Flash | 16  | KF424 static + RS mode5                        |
| Flash | 32  | KF424 static + RS mode5                        |
| Flash | 64  | KF424 static + RS mode5                        |
| Flash | 128 | dev-m dynamic + SS                             |
| Pro   | 8   | dev-m dynamic + RS mode5                       |
| Pro   | 16  | dev-m dynamic + RS mode5                       |
| Pro   | 32  | dev-m dynamic + SS                             |
| Pro   | 64  | dev-m dynamic + RS mode5                       |
| Pro   | 128 | KF424 static + RS mode5, compact-SMEM stage 4  |

The default implementation for the fused/small-M side is the dev-m dynamic
scheduler. Exact Flash/Pro keys in the table may replace it with dynamic+RS or
the KF424 static-RS winner. Other integer M values do not inherit a neighbouring
optimized bucket; they remain dev-m dynamic + SS. The large-M side alone uses
bigM split mode4/remap.

The dev-m host heuristic currently supports the eight-rank Flash and Pro
geometries above, plus MiMo (`H=6144`, `I=2048`, `E=384`, `topk=8`), on an
SM90 device with 132 SMs. An unsupported fused hardware/model geometry fails
closed instead of silently switching to the old bigM static small-M kernel.

Production should use `kernel_family="auto"`. `kernel_family="fused"` forces
the outer fused family and still applies the exact-key selector above; the
dev-m heuristic may use an internal BN128 tile at higher M. The
`kernel_family="split"` value forces the BN128 bigM path. These two values are
diagnostic overrides of the outer threshold, not separate production policies.

## Small-M changes relative to dev-m

The comparison baseline in this section is Aichen dev-m dynamic, not the older
static dev branch.

### pointwise selector

No single kernel won all ten Flash/Pro points. The new host selector preserves
dev-m dynamic where it wins and chooses an RS specialization only on accepted
exact keys. It does not convert the measurements into unvalidated continuous
M ranges.

Implementation:

- `csrc/jit_kernels/heuristics/sm90_nvfp4_mega_moe_allm.hpp`
- `csrc/apis/mega.hpp`

### dev-m dynamic + RS mode5

The dev-m weight-loader remains the sole dynamic task producer. It claims L1
and L2 tasks through global counters and publishes each task to the CTA through
a two-stage shared-memory mailbox. The RS port is confined to the decoder/MMA
consumer path and does not change mailbox ownership or barrier transitions.

Mode5 adds:

- register-source FP8 WGMMA for the swap-AB path;
- four live K32 A-register fragments per WGMMA group;
- vector preload of the eight row-local UE4M3 scale bytes;
- the exact shared-memory LUT lookup used by the validated decoder;
- a rendezvous before packed-B/A/SFA pipeline storage is recycled.

SM90/Hopper has no native FP4 tensor-core MMA. Packed NVFP4 is decoded to E4M3
and the tensor-core operation is FP8 WGMMA.

Implementation:

- `csrc/jit_kernels/impls/sm90_nvfp4_mega_moe_h200_fused.hpp`
- `deep_gemm/include/deep_gemm/impls/sm90_nvfp4_mega_moe_h200_fused.cuh`
- `deep_gemm/include/deep_gemm/impls/sm90_nvfp4_mega_moe_h200_fused_body.inl`
- `deep_gemm/include/deep_gemm/scheduler/mega_moe.cuh`

### KF424 static-RS

The four selected 424 points retain the static small-M scheduler but use
route-exact block-M and experts-per-wave configurations. Pro M128 additionally
uses BM24, four stages and compact shared memory: the unused expanded-B stage
is retired for the RS path and the allocation is bounded by the larger of the
GEMM pipeline footprint and the later combine scratch lifetime.

The rejected direct-LUT mode3 is not enabled. Accepted RS points keep mode5's
batch-four fragments, vector-scale preload and exact shared LUT.

Implementation:

- `csrc/jit_kernels/heuristics/sm90_nvfp4_mega_moe_small_m.hpp`
- `csrc/jit_kernels/impls/sm90_nvfp4_mega_moe_small_m.hpp`
- `deep_gemm/include/deep_gemm/impls/sm90_nvfp4_mega_moe_small_m.cuh`
- `deep_gemm/include/deep_gemm/impls/sm90_nvfp4_mega_moe_small_m_fused_body.inl`

### Common 128-byte workspace ABI

All current version comparisons used the dev-m 128-byte workspace prefix. This
implementation therefore reserves the same layout for every arm:

- bytes 0–27: existing grid/NVLink state;
- bytes 28–31: dev-m L1 task counter;
- bytes 32–35: dev-m L2 task counter;
- bytes 36–127: padding before expert counters.

Static small-M and bigM split kernels do not use the task counters, but they
consume the same shifted buffer layout. This avoids the invalid early D40
comparison that mixed 32-byte and 128-byte workspace ABIs.

### All-M small-kernel JIT register policy

The original all-M optimization binaries used ptxas' default register-usage
level (`5`). The general DeepGEMM JIT appends
`--register-usage-level=10`; applying that policy to the long-lived Pro M64 RS
fragments changed the measured gain into a `-0.44%` regression. An isolated
controlled H200 run showed that restoring level 5 recovers `+0.735%`, matching
the AOT and archived result.

`Compiler::build` therefore accepts optional per-kernel flags. Only the All-M
dev-m/dynamic-RS and KF424 calls append:

```text
--ptxas-options=--register-usage-level=5
--use_fast_math  # only when fast_math=True
```

The effective flags are part of the JIT cache signature. The bigM split kernels
retain the upstream global JIT flags unchanged.

## Small-M performance versus dev-m dynamic

These numbers are the archived D40 measurements. They are physical 8×H200,
profile-off, same-process A/B/B/A, CUDA-event rank-MAX results. Every adopted
non-identity point won all nine retained blocks and passed the independent
correctness oracle.

| Model | M   | all-m-opt arm              | dev-m p50  | selected p50 | Paired speedup |
| ----- | --- | -------------------------- | ---------- | ------------ | -------------- |
| Flash | 8   | dynamic+RS                 | 186.750 us | 184.216 us   | +1.368%        |
| Flash | 16  | KF424 static-RS            | 201.939 us | 197.674 us   | +2.153%        |
| Flash | 32  | KF424 static-RS            | 202.712 us | 202.270 us   | +0.201%        |
| Flash | 64  | KF424 static-RS            | 225.322 us | 216.358 us   | +4.120%        |
| Flash | 128 | dev-m dynamic              | 223.278 us | 223.278 us   | identity       |
| Pro   | 8   | dynamic+RS                 | 620.517 us | 616.152 us   | +0.706%        |
| Pro   | 16  | dynamic+RS                 | 678.959 us | 675.257 us   | +0.552%        |
| Pro   | 32  | dev-m dynamic              | 683.060 us | 683.060 us   | identity       |
| Pro   | 64  | dynamic+RS                 | 711.036 us | 705.450 us   | +0.809%        |
| Pro   | 128 | KF424 static-RS compact    | 754.536 us | 742.838 us   | +2.329%        |

The archived ten-point, equally weighted virtual-portfolio geometric-mean
speedup is **1.216%** versus dev-m dynamic.

### Combined public-API revalidation (2026-09-03)

The clean rebuilt wheel was retested through the real
`_C.nvfp4_mega_moe` entry, host selector and fresh JIT on 8×H200. This was a
profile-off, same-process, 9-block A/B/B/A run with 32 calls per observation
and rank-MAX timing.

| Model | M   | Combined p50 | Paired speedup vs dev-m | Winning blocks |
| ----- | --- | ------------ | ----------------------- | -------------- |
| Flash | 8   | 184.961 us   | +1.160%                 | 9/9            |
| Flash | 16  | 197.223 us   | +2.360%                 | 9/9            |
| Flash | 32  | 201.659 us   | +0.465%                 | 9/9            |
| Flash | 64  | 216.123 us   | +4.561%                 | 9/9            |
| Flash | 128 | 223.548 us   | -0.015%                 | identity arm   |
| Pro   | 8   | 616.214 us   | +1.234%                 | 9/9            |
| Pro   | 16  | 675.363 us   | +0.581%                 | 9/9            |
| Pro   | 32  | 683.510 us   | -0.011%                 | identity arm   |
| Pro   | 64  | 704.011 us   | +0.611%                 | 9/9            |
| Pro   | 128 | 738.297 us   | +1.736%                 | 9/9            |

The combined wheel's ten-point geometric-mean speedup is **1.260%**, with a
95% bootstrap interval for winner/baseline of `[0.97903, 0.99439]`. All eight
non-identity optimized points won 9/9 blocks. The two dev-m identity points are
within 0.02% of baseline and do not represent an intended optimization.

This final rerun used the selector in which every non-split fallback is dev-m
dynamic. It confirms that removing the old bigM static small-M fallback did not
change the accepted D40 exact-key performance.

### All-M runtime health

The final clean wheel also passed a 48/48 public-API health sweep across Flash,
Pro, and MiMo. The sweep covered `M=1`, non-anchor small-M values, all D40
anchors, both sides of each default family threshold, and large-M points through
`M=8192`. Every point passed auto-versus-forced-family comparison, repeated
invocation, workspace reuse, finiteness, and hang-free completion on eight
physical H200 GPUs. The ten D40 anchors additionally passed two independently
generated input states against the correctness oracle.

## Large-M bigM path

The large-M algorithm and split kernel path are preserved from the upstream
`bigMopt` merge; only the common 128-byte workspace ABI integration applies:

- splitT1 trap-only NVLink timeout path;
- L1 dispatch-dequant half-stream mode4;
- calibrated physical-warp remap;
- L2 warp-private contiguous scatter staging.

Archived 8×H200 rank-MAX results versus the static Aichen dev baseline are:

| Model | M    | dev       | bigM      | Speedup |
| ----- | ---- | --------- | --------- | ------- |
| Flash | 2048 | 1234.7 us | 1103.2 us | +10.65% |
| Flash | 4096 | 2173.6 us | 1882.8 us | +13.38% |
| Flash | 8192 | 4170.4 us | 3602.9 us | +13.61% |
| Pro   | 2048 | 2990.6 us | 2765.3 us | +7.54%  |
| Pro   | 4096 | 5304.1 us | 4859.4 us | +8.38%  |
| Pro   | 8192 | 9844.4 us | 9021.8 us | +8.36%  |

The six-point geometric-mean latency reduction is **10.35%**. Archived
correctness acceptance covered all six points, eight ranks and
`1,291,845,632` BF16 outputs with zero bit mismatches.
