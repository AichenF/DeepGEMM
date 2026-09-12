# SM90 NVFP4 MegaMoE all-M optimization

This branch provides one native DeepGEMM implementation for H20 and H200.
Production uses `kernel_family="auto"`; `fused` and `split` are diagnostic
overrides only.

## Runtime policy

```text
M < 192  -> BN256 fused small-M portfolio
M >= 192 -> BN128 bigM split

fused:
  78 SMs  -> H20 buckets
  132 SMs -> H200 buckets
  then select Flash / Pro / MiMo and the continuous M range
```

All SM90 NVFP4 policy is in
`csrc/jit_kernels/heuristics/sm90_nvfp4_mega_moe.hpp`.

Baselines:

- small M: Aichen dev-m dynamic scheduler (`8b59b194` plus the validated
  Flash/Pro adapter);
- large M: current Aichen dev (`dbd995f`), which already contains the same
  `bigMopt@53d943e`, so the comparison is identity;
- W8A8: DeepGEMM PR383 at `bc4f33a`.

## Small-M buckets

RS means register-source A plus shared-memory B for FP8 WGMMA after the NVFP4
weights have been decoded. SS means both MMA operands are in shared memory.

### H20-3e, 78 SMs

| Model | M range | Strategy |
|---|---:|---|
| Flash | 1–128 | dev-m dynamic + RS |
| Flash | 129–191 | dev-m dynamic + SS |
| Pro | 1–191 | dev-m dynamic + RS |
| MiMo | 1–144 | dev-m dynamic + RS |
| MiMo | 145–191 | dev-m dynamic + SS |

### H200, 132 SMs

| Model | M range | Strategy |
|---|---:|---|
| Flash | 1–35 | dynamic + RS |
| Flash | 36–64 | static + SS |
| Flash | 65–191 | dev-m dynamic + SS |
| Pro | 1–128 | dynamic + RS |
| Pro | 129–191 | dev-m dynamic + SS |
| MiMo | 1–16 | dynamic + RS |
| MiMo | 17–32 | dev-m dynamic + SS |
| MiMo | 33–96 | dynamic + RS |
| MiMo | 97–191 | dev-m dynamic + SS |

The optimized RS/static arms use the validated register-usage level 5. A
dynamic-SS fallback uses the original dev-m JIT flags, so fallback does not
inherit the specialization's register policy.

## How the tables were measured

- W4A8 versus W8A8: balanced routing, clamp 10, profile-off CUDA events,
  rank-MAX, two W4 and two W8 processes in W4/W8/W8/W4 order.
- H200 fresh table: 400 ms target bursts; every retained burst exceeded
  100 ms; three retained observations per process.
- Small-M bucket selection: same-process A/B/B/A over multiple random routing
  instances. This is a different contract from the balanced W8A8 comparison.
- Positive percentages mean W4A8 is faster. `identity` means the selected code
  is the same current-dev bigM path or a dev-m fallback.

## H20 results

The latency columns are the balanced W4/W8 comparison. The `vs baseline`
column is the paired dev-m result for M<192: M8–64 uses the median of hash,
contiguous and two random production routing profiles; M128/M191 uses the
nine-random-router production qualification. No H20 allocation is currently
active to rerun all three binaries in one new process sequence.

### Flash

| M | Family | W4A8 | Aichen baseline | W8A8 | W4 vs baseline | W4 vs W8 |
|---:|---|---:|---|---:|---:|---:|
| 8 | fused | 262.513 us | dev-m | 279.895 us | +12.73% | +6.621% |
| 16 | fused | 264.404 us | dev-m | 292.600 us | +13.64% | +10.664% |
| 32 | fused | 281.242 us | dev-m | 285.287 us | +10.87% | +1.438% |
| 64 | fused | 318.219 us | dev-m | 323.855 us | +20.41% | +1.771% |
| 128 | fused | 319.045 us | dev-m | 419.756 us | +2.25% | +31.566% |
| 191 | fused | 433.837 us | dev-m | 456.926 us | identity | +5.322% |
| 192 | split | 562.045 us | current dev | 457.747 us | identity | -18.557% |
| 256 | split | 564.907 us | current dev | 463.455 us | identity | -17.959% |
| 512 | split | 901.202 us | current dev | 876.452 us | identity | -2.746% |
| 1024 | split | 1395.465 us | current dev | 1262.150 us | identity | -9.553% |
| 2048 | split | 2518.191 us | current dev | 2461.707 us | identity | -2.243% |
| 4096 | split | 4948.623 us | current dev | 4823.763 us | identity | -2.523% |
| 8192 | split | 9839.537 us | current dev | 9586.367 us | identity | -2.573% |

### Pro

| M | Family | W4A8 | Aichen baseline | W8A8 | W4 vs baseline | W4 vs W8 |
|---:|---|---:|---|---:|---:|---:|
| 8 | fused | 966.393 us | dev-m | 1092.396 us | +16.25% | +13.039% |
| 16 | fused | 980.224 us | dev-m | 1226.528 us | +15.41% | +25.127% |
| 32 | fused | 993.045 us | dev-m | 1139.693 us | +12.74% | +14.768% |
| 64 | fused | 1112.202 us | dev-m | 1143.592 us | +18.30% | +2.822% |
| 128 | fused | 1100.064 us | dev-m | 1430.974 us | +47.89% | +30.081% |
| 191 | fused | 1266.034 us | dev-m | 1588.387 us | +2.08% | +25.462% |
| 192 | split | 1901.680 us | current dev | 1632.150 us | identity | -14.173% |
| 256 | split | 1904.822 us | current dev | 1654.887 us | identity | -13.121% |
| 512 | split | 1920.703 us | current dev | 1697.176 us | identity | -11.638% |
| 1024 | split | 3172.809 us | current dev | 3161.267 us | identity | -0.364% |
| 2048 | split | 6261.037 us | current dev | 6194.158 us | identity | -1.068% |
| 4096 | split | 12406.715 us | current dev | 12216.341 us | identity | -1.534% |
| 8192 | split | 24676.404 us | current dev | 24326.343 us | identity | -1.419% |

### MiMo 2.5 Pro

| M | Family | W4A8 | Aichen baseline | W8A8 | W4 vs baseline | W4 vs W8 |
|---:|---|---:|---|---:|---:|---:|
| 8 | fused | 567.538 us | dev-m | 610.421 us | +11.57% | +7.556% |
| 16 | fused | 574.591 us | dev-m | 663.890 us | +11.36% | +15.541% |
| 32 | fused | 588.246 us | dev-m | 626.202 us | +12.64% | +6.452% |
| 64 | fused | 665.165 us | dev-m | 782.130 us | +22.63% | +17.584% |
| 128 | fused | 692.975 us | dev-m | 934.186 us | +20.83% | +34.808% |
| 191 | fused | 928.741 us | dev-m | 951.286 us | identity | +2.428% |
| 192 | split | 1149.503 us | current dev | 968.364 us | identity | -15.758% |
| 256 | split | 1152.810 us | current dev | 1005.921 us | identity | -12.742% |
| 512 | split | 1863.512 us | current dev | 1841.303 us | identity | -1.192% |
| 1024 | split | 3001.004 us | current dev | 2763.704 us | identity | -7.907% |
| 2048 | split | 5487.336 us | current dev | 5461.660 us | identity | -0.468% |
| 4096 | split | 10190.833 us | current dev | 9949.552 us | identity | -2.368% |
| 8192 | split | 19850.376 us | current dev | 19793.335 us | identity | -0.287% |

H20 summary:

- random-router selected-policy qualification: +26.28% on the 99 optimized
  workloads and +18.66% across all 135 workloads including fallback;
- six small-M W4/W8 points per model: W4A8 +13.57% over PR383;
- seven large-M points per model: PR383 W8A8 is +7.28% faster overall;
- all 573 integer M1–191 correctness/health points passed.

## H200 results

The following is a fresh three-way matrix on 8x H200. Dev-m was run twice for
every fused point. For M>=192, the Aichen baseline is current dev and its
latency is the W4 latency because both branches contain the same bigM source.

### Flash

| M | Family | Aichen baseline | W4A8 | W8A8 | W4 vs baseline | W4 vs W8 |
|---:|---|---:|---:|---:|---:|---:|
| 8 | fused | 195.907 us | 197.179 us | 227.414 us | -0.645% | +15.334% |
| 16 | fused | 198.139 us | 198.547 us | 230.959 us | -0.206% | +16.324% |
| 32 | fused | 202.434 us | 203.105 us | 236.385 us | -0.330% | +16.386% |
| 64 | fused | 221.005 us | 215.698 us | 249.508 us | +2.461% | +15.675% |
| 128 | fused | 228.276 us | 228.928 us | 247.323 us | -0.284% | +8.036% |
| 191 | fused | 234.550 us | 235.156 us | 246.590 us | -0.258% | +4.862% |
| 192 | split | 284.201 us | 284.201 us | 246.623 us | identity | -13.222% |
| 256 | split | 291.117 us | 291.117 us | 253.771 us | identity | -12.829% |
| 512 | split | 367.919 us | 367.919 us | 418.893 us | identity | +13.855% |
| 1024 | split | 621.454 us | 621.454 us | 574.081 us | identity | -7.623% |
| 2048 | split | 1028.647 us | 1028.647 us | 991.437 us | identity | -3.617% |
| 4096 | split | 2005.476 us | 2005.476 us | 1825.380 us | identity | -8.980% |
| 8192 | split | 3939.894 us | 3939.894 us | 3599.542 us | identity | -8.639% |

### Pro

| M | Family | Aichen baseline | W4A8 | W8A8 | W4 vs baseline | W4 vs W8 |
|---:|---|---:|---:|---:|---:|---:|
| 8 | fused | 671.519 us | 672.769 us | 803.796 us | -0.186% | +19.476% |
| 16 | fused | 674.476 us | 676.103 us | 806.737 us | -0.241% | +19.322% |
| 32 | fused | 682.609 us | 693.771 us | 818.797 us | -1.609% | +18.021% |
| 64 | fused | 702.129 us | 703.611 us | 835.347 us | -0.211% | +18.723% |
| 128 | fused | 855.084 us | 773.783 us | 854.090 us | +10.507% | +10.378% |
| 191 | fused | 850.619 us | 856.117 us | 865.940 us | -0.642% | +1.147% |
| 192 | split | 983.296 us | 983.296 us | 865.698 us | identity | -11.960% |
| 256 | split | 994.342 us | 994.342 us | 852.986 us | identity | -14.216% |
| 512 | split | 1050.592 us | 1050.592 us | 897.118 us | identity | -14.608% |
| 1024 | split | 1387.751 us | 1387.751 us | 1322.493 us | identity | -4.702% |
| 2048 | split | 2616.582 us | 2616.582 us | 2466.142 us | identity | -5.750% |
| 4096 | split | 5096.120 us | 5096.120 us | 4545.044 us | identity | -10.814% |
| 8192 | split | 9973.137 us | 9973.137 us | 8792.473 us | identity | -11.838% |

### MiMo 2.5 Pro

| M | Family | Aichen baseline | W4A8 | W8A8 | W4 vs baseline | W4 vs W8 |
|---:|---|---:|---:|---:|---:|---:|
| 8 | fused | 396.403 us | 398.512 us | 479.902 us | -0.529% | +20.423% |
| 16 | fused | 397.686 us | 399.631 us | 488.124 us | -0.487% | +22.144% |
| 32 | fused | 411.262 us | 413.441 us | 496.425 us | -0.527% | +20.071% |
| 64 | fused | 462.048 us | 456.762 us | 521.607 us | +1.157% | +14.197% |
| 128 | fused | 483.974 us | 482.649 us | 512.322 us | +0.275% | +6.148% |
| 191 | fused | 500.403 us | 501.815 us | 525.780 us | -0.281% | +4.776% |
| 192 | split | 589.133 us | 589.133 us | 522.798 us | identity | -11.260% |
| 256 | split | 598.074 us | 598.074 us | 524.416 us | identity | -12.316% |
| 512 | split | 781.048 us | 781.048 us | 892.134 us | identity | +14.223% |
| 1024 | split | 1363.490 us | 1363.490 us | 1253.070 us | identity | -8.098% |
| 2048 | split | 2232.972 us | 2232.972 us | 2163.399 us | identity | -3.116% |
| 4096 | split | 4206.323 us | 4206.323 us | 3796.652 us | identity | -9.739% |
| 8192 | split | 7933.118 us | 7933.118 us | 7437.712 us | identity | -6.245% |

H200 summary:

- balanced small-M 18-point matrix: W4A8 +0.41% versus dev-m and +13.79%
  versus PR383;
- random-router bucket qualification: +2.39% on 198 optimized workloads;
  all 315 workloads passed correctness and timing hygiene;
- large-M 21-point matrix: current dev is identity; PR383 W8A8 is +8.11%
  faster overall;
- all 39 W4/W8 points: W4A8 +1.78% overall;
- all 573 integer M1–191 correctness/health points passed.

## Historical bigM origin

Before `bigMopt` was merged into Aichen dev, the accepted physical-H200 result
was:

| Model | M | Old dev | bigMopt | bigM faster |
|---|---:|---:|---:|---:|
| Flash | 2048 | 1234.7 us | 1103.2 us | +10.65% |
| Flash | 4096 | 2173.6 us | 1882.8 us | +13.38% |
| Flash | 8192 | 4170.4 us | 3602.9 us | +13.61% |
| Pro | 2048 | 2990.6 us | 2765.3 us | +7.54% |
| Pro | 4096 | 5304.1 us | 4859.4 us | +8.38% |
| Pro | 8192 | 9844.4 us | 9021.8 us | +8.36% |

The six-point geometric-mean improvement was +10.35%. Independent correctness
covered 1,291,845,632 BF16 outputs with zero bit mismatches. This is historical
provenance only; against current dev, large M is identity.

## Validation

- H20: 573/573 integer M1–191 points passed.
- H200: 573/573 integer M1–191 points passed.
- H200 random-router performance: 315/315 workloads passed correctness and
  timing hygiene.
- H200 fresh W4/W8 all-M matrix: all 12 processes and 39 points passed.
- The clean `all-m-opt` worktree was independently rebuilt and its boundary
  validation passed.

No instrumented kernel timing is used in the performance tables.
