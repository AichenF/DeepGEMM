# SM90 NVFP4 MegaMoE all-M optimization

This branch provides one native DeepGEMM implementation for 8-GPU H20 and
H200. Production uses `kernel_family="auto"`; `fused` and `split` remain
diagnostic overrides.

## M and the two-level selector

`M` is the number of source tokens on each rank before top-k expansion. With
`R` ranks, the global routed-slot count is `R * M * topk`. The old dev rule
`M * topk / local_experts <= 192` measured average routed slots per expert; it
did not use a different definition of M.

Direct H20/H200 measurements select the family from raw M:

```text
M <= 256  -> BN256 fused
M >= 257  -> BN128 split L1 + L2
```

At M256, fused wins the five-router median on both targets. From M257 through
M1024, split wins all 165 paired workloads on each target
(`3 models * 11 M values * 5 routers`). The abrupt crossover is caused by the
fused physical plan changing from BM64/BN256 to its BM128/BN128/stage6
fallback at M257, not by one extra token's arithmetic cost.

All SM90 policy is in
`csrc/jit_kernels/heuristics/sm90_nvfp4_mega_moe.hpp`.

## Fused-family buckets

RS means register-source A plus shared-memory B; SS means both WGMMA operands
come from shared memory. `dev-m` is dynamic scheduler + SS. Static-SS and
static-RS remain legal candidates, but only measured winning ranges are used
by production.

| Target | Model | M range | Production strategy |
|---|---|---:|---|
| H20 | Flash | 1–128 | dev-m dynamic + RS |
| H20 | Flash | 129–256 | dev-m dynamic + SS |
| H20 | Pro | 1–191 | dev-m dynamic + RS |
| H20 | Pro | 192–256 | dev-m dynamic + SS |
| H20 | MiMo | 1–144 | dev-m dynamic + RS |
| H20 | MiMo | 145–256 | dev-m dynamic + SS |
| H200 | Flash | 1–35 | dynamic + RS |
| H200 | Flash | 36–64 | static + SS |
| H200 | Flash | 65–256 | dev-m dynamic + SS |
| H200 | Pro | 1–128 | dynamic + RS |
| H200 | Pro | 129–256 | dev-m dynamic + SS |
| H200 | MiMo | 1–16 | dynamic + RS |
| H200 | MiMo | 17–32 | dev-m dynamic + SS |
| H200 | MiMo | 33–96 | dynamic + RS |
| H200 | MiMo | 97–256 | dev-m dynamic + SS |
| H20/H200 | all three | 257+ | split family |

The RS/static specializations use register-usage level 5. Dynamic-SS fallback
uses the original dev-m JIT flags; it does not inherit the specialization's
register policy.

## Baselines and measurement

- Small-M M1–1024: dev-m `8b59b194` plus the validated Flash/Pro adapter.
  M257–1024 remains a dev-m comparison even though production selects split.
- Large-M M2048/4096/8192: current dev `dbd995f`, forced to the split family.
  Its split device files are byte-identical to this branch, so this comparison
  is identity.
- W8A8: DeepGEMM PR383 at `bc4f33a`.

Selector qualification used capacity 8448, profile-off CUDA events and the
maximum latency across all eight ranks. Each router/arm retained 50 individual
kernel calls. Pairwise decisions used 25 same-process A/B/B/A blocks, giving
50 calls per arm, before taking the router median. The five router set contains
hash, contiguous, two random instances and a rank-balanced-hot stress case.

The H200 M256 stress router prefers split, while the other four routers and the
five-router median prefer fused. This is the only boundary exception; the
fixed production selector follows the median. Some H200 single-call
distributions were bimodal on `viking-prod-214`, but every paired M257–1024
router still selected split.

## H20 performance

Hardware: 8x H20-3e (`viking-prod-583`). Latencies are the median of two
50-call processes per format in W4/W8/W8/W4 order with balanced routing.
`vs baseline` is the five-router paired median for optimized fused points and
identity for dev-m/split fallback.

| Model | M | Family | W4A8 us | Baseline | vs baseline | W8A8 us | W4 vs W8 |
|---|---:|---|---:|---|---:|---:|---:|
| Flash | 8 | fused | 263.0 | dev-m | +12.65% | 284.0 | +7.99% |
| Flash | 64 | fused | 295.1 | dev-m | +20.18% | 328.6 | +11.35% |
| Flash | 128 | fused | 320.7 | dev-m | +3.96% | 389.9 | +21.59% |
| Flash | 256 | fused | 447.9 | dev-m | identity | 468.4 | +4.60% |
| Flash | 257 | split | 566.7 | dev-m | +71.32% | 466.4 | -17.70% |
| Flash | 512 | split | 902.9 | dev-m | +12.82% | 880.3 | -2.50% |
| Flash | 1024 | split | 1395.4 | dev-m | +23.25% | 1264.9 | -9.35% |
| Flash | 2048 | split | 2521.4 | dev split | identity | 2465.1 | -2.23% |
| Flash | 4096 | split | 4937.9 | dev split | identity | 4821.4 | -2.36% |
| Flash | 8192 | split | 9833.5 | dev split | identity | 9583.3 | -2.55% |
| Pro | 8 | fused | 965.3 | dev-m | +15.49% | 1080.2 | +11.90% |
| Pro | 64 | fused | 999.1 | dev-m | +21.10% | 1132.3 | +13.34% |
| Pro | 128 | fused | 1096.1 | dev-m | +54.39% | 1254.1 | +14.42% |
| Pro | 256 | fused | 1593.3 | dev-m | identity | 1624.6 | +1.97% |
| Pro | 257 | split | 1903.7 | dev-m | +93.58% | 1620.8 | -14.86% |
| Pro | 512 | split | 1921.2 | dev-m | +40.46% | 1692.3 | -11.92% |
| Pro | 1024 | split | 3170.2 | dev-m | +28.67% | 3155.6 | -0.46% |
| Pro | 2048 | split | 6258.7 | dev split | identity | 6199.1 | -0.95% |
| Pro | 4096 | split | 12406.5 | dev split | identity | 12225.3 | -1.46% |
| Pro | 8192 | split | 24686.4 | dev split | identity | 24322.6 | -1.47% |
| MiMo | 8 | fused | 571.6 | dev-m | +11.14% | 609.5 | +6.63% |
| MiMo | 64 | fused | 638.5 | dev-m | +22.93% | 710.3 | +11.24% |
| MiMo | 128 | fused | 687.9 | dev-m | +21.80% | 937.1 | +36.23% |
| MiMo | 256 | fused | 951.0 | dev-m | identity | 958.4 | +0.78% |
| MiMo | 257 | split | 1153.4 | dev-m | +86.22% | 958.8 | -16.87% |
| MiMo | 512 | split | 1862.4 | dev-m | +16.60% | 1844.0 | -0.99% |
| MiMo | 1024 | split | 3003.9 | dev-m | +42.41% | 2765.1 | -7.95% |
| MiMo | 2048 | split | 5487.9 | dev split | identity | 5464.2 | -0.43% |
| MiMo | 4096 | split | 10197.3 | dev split | identity | 9951.7 | -2.41% |
| MiMo | 8192 | split | 19857.1 | dev split | identity | 19785.7 | -0.36% |

Across the displayed H20 grid, W4A8 is +9.30% over PR383 for M<=256.
For M257–1024, PR383 latency is 9.40% lower; over the completed
M2048/4096/8192 large-M matrix it is 1.58% lower.

The rank-balanced-hot router can reverse the H20 high-RS buckets even though
the five-router median and the other four routers strongly favor RS. For
example, Pro M128 has a +54.39% router median but a -41.03% hot-router result.
The table reports the registered median policy rather than hiding this stress
case.

## H200 performance

The selector was requalified on 8x H200 (`viking-prod-214`) with 50-call
paired measurements. The complete M2048/4096/8192 cross-format matrix was run
on 8x H200 `viking-prod-332`, with two 50-call processes per format. The
smaller rows retain the stable H200 table and the new M256 fused result. No
failed or partially written process is used.

| Model | M | Family | W4A8 us | Baseline | vs baseline | W8A8 us | W4 vs W8 |
|---|---:|---|---:|---|---:|---:|---:|
| Flash | 8 | fused | 197.2 | dev-m | +2.67% | 227.4 | +15.33% |
| Flash | 64 | fused | 215.7 | dev-m | +4.54% | 249.5 | +15.68% |
| Flash | 128 | fused | 228.9 | dev-m | identity | 247.3 | +8.04% |
| Flash | 256 | fused | 251.9 | dev-m | identity | 253.8 | +0.74% |
| Flash | 512 | split | 367.9 | dev-m | +10.50% | 418.9 | +13.86% |
| Flash | 1024 | split | 621.5 | dev-m | +20.12% | 574.1 | -7.62% |
| Flash | 2048 | split | 993.6 | dev split | identity | 952.4 | -4.15% |
| Flash | 4096 | split | 1924.4 | dev split | identity | 1764.2 | -8.32% |
| Flash | 8192 | split | 3753.4 | dev split | identity | 3457.8 | -7.87% |
| Pro | 8 | fused | 672.8 | dev-m | +1.34% | 803.8 | +19.48% |
| Pro | 64 | fused | 703.6 | dev-m | +1.21% | 835.3 | +18.72% |
| Pro | 128 | fused | 773.8 | dev-m | +3.86% | 854.1 | +10.38% |
| Pro | 256 | fused | 868.2 | dev-m | identity | 853.0 | -1.75% |
| Pro | 512 | split | 1050.6 | dev-m | +43.82% | 897.1 | -14.61% |
| Pro | 1024 | split | 1387.8 | dev-m | +26.11% | 1322.5 | -4.70% |
| Pro | 2048 | split | 2512.1 | dev split | identity | 2408.5 | -4.13% |
| Pro | 4096 | split | 4832.8 | dev split | identity | 4351.6 | -9.96% |
| Pro | 8192 | split | 9495.9 | dev split | identity | 8435.4 | -11.17% |
| MiMo | 8 | fused | 398.5 | dev-m | identity | 479.9 | +20.42% |
| MiMo | 64 | fused | 456.8 | dev-m | +1.95% | 521.6 | +14.20% |
| MiMo | 128 | fused | 482.6 | dev-m | identity | 512.3 | +6.15% |
| MiMo | 256 | fused | 541.4 | dev-m | identity | 524.4 | -3.14% |
| MiMo | 512 | split | 781.0 | dev-m | +15.92% | 892.1 | +14.22% |
| MiMo | 1024 | split | 1363.5 | dev-m | +37.26% | 1253.1 | -8.10% |
| MiMo | 2048 | split | 2131.9 | dev split | identity | 2069.1 | -2.95% |
| MiMo | 4096 | split | 4026.9 | dev split | identity | 3673.2 | -8.78% |
| MiMo | 8192 | split | 7536.6 | dev split | identity | 7223.4 | -4.16% |

At the M257 boundary, the H200 five-router paired median versus dev-m is
+24.97% for Flash, +42.22% for Pro and +39.20% for MiMo.

Across the completed H200 M2048/4096/8192 matrix, PR383 latency is 6.87%
lower. The W4A8 values are identity versus current dev forced split.

The prior H200 random-router small-M qualification remains +2.39% over dev-m
on the 198 workloads where production differs from fallback; all 315
model/M/router workloads passed correctness and timing hygiene.

## Validation

- Both targets: M256 paired boundary correctness and timing passed for all
  15 model/router workloads.
- H20: M257–1024 split won 165/165 paired workloads; all timing hygiene gates
  passed.
- H200: M257–1024 split won 165/165 paired workloads; correctness passed.
  Per-call bimodality is recorded, and no unpaired reversal is used for policy.
- H20/H200 historical full-integer fused health: 573/573 M1–191 points passed
  on each target.
- Current branch rebuilt successfully after the selector change. The Python
  and C++ selectors agree at M256/257 for Flash, Pro and MiMo.

No instrumented or profiled timing is used in any performance claim.
