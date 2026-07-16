# CUDA 13.2 MiMo final kernel versus frozen Aichen baseline

Both sides are recomputed from raw logs. Each side contributes 30 fresh complete processes; every process contains the same ordered 11 M values and uses the protected benchmark with `--num-tests 20`.
Primary comparison is rank0 median30, matching the frozen baseline. Mean-rank and max-rank are retained for deployment diagnostics. Negative delta means the final kernel is faster. Performance is reported, not used as a parser pass/fail gate.

| M | recv | layout baseline → final | ImagePerf rank0 | rank0 baseline | baseline vs image | rank0 final | final vs image | final vs baseline | baseline P10–P90 | final P10–P90 | baseline CI X10–X21 | final CI X10–X21 | max median baseline → final | max delta | mean median baseline → final | mean delta |
|--:|--:|:--|--:|--:|--:|--:|--:|--:|:--|:--|:--|:--|:--|--:|:--|--:|
| 8 | 64 | BN256-standard → BN256-grouped | 491.4 | 503.55 | +2.47% | 451.10 | -8.20% | -10.42% | 489.03–539.20 | 431.77–485.53 | 495.4–511.3 | 437.9–461.4 | 512.95 → 460.60 | -10.21% | 505.75 → 453.45 | -10.34% |
| 16 | 118 | BN256-standard → BN256-grouped | 536.4 | 576.05 | +7.39% | 511.85 | -4.58% | -11.14% | 557.67–649.07 | 493.36–578.76 | 565.3–582.0 | 499.6–521.1 | 584.10 → 519.05 | -11.14% | 576.60 → 511.15 | -11.35% |
| 32 | 241 | BN256-standard → BN256-grouped | 554.7 | 578.55 | +4.30% | 523.20 | -5.68% | -9.57% | 558.75–593.95 | 504.83–575.94 | 569.8–584.1 | 515.8–536.3 | 585.95 → 531.85 | -9.23% | 579.60 → 524.40 | -9.52% |
| 64 | 546 | BN256-standard → BN256-grouped | 568.8 | 550.75 | -3.17% | 541.95 | -4.72% | -1.60% | 529.07–619.63 | 525.96–573.63 | 542.1–557.3 | 537.8–560.3 | 556.50 → 557.60 | +0.20% | 549.60 → 549.35 | -0.05% |
| 128 | 984 | BN256-standard → BN256-grouped | 524.4 | 570.50 | +8.79% | 566.15 | +7.96% | -0.76% | 540.55–613.86 | 538.18–606.92 | 558.8–583.2 | 556.9–578.1 | 577.40 → 570.60 | -1.18% | 567.95 → 560.50 | -1.31% |
| 256 | 2037 | BN256-standard → BN256-grouped | 540.2 | 563.55 | +4.32% | 565.25 | +4.64% | +0.30% | 545.55–594.07 | 551.44–592.71 | 554.1–573.2 | 560.6–572.9 | 574.05 → 573.30 | -0.13% | 566.00 → 566.80 | +0.14% |
| 512 | 4097 | BN256-standard → BN256-grouped | 1002.2 | 1035.00 | +3.27% | 1027.00 | +2.47% | -0.77% | 1022.60–1056.30 | 1013.00–1046.20 | 1029.0–1044.0 | 1019.0–1032.0 | 1050.00 → 1032.50 | -1.67% | 1041.65 → 1025.60 | -1.54% |
| 1024 | 8140 | BN256-standard → BN256-grouped | 1500.1 | 1547.00 | +3.13% | 1547.50 | +3.16% | +0.03% | 1533.80–1570.30 | 1521.90–1571.20 | 1542.0–1554.0 | 1539.0–1556.0 | 1553.00 → 1553.00 | +0.00% | 1546.85 → 1544.20 | -0.17% |
| 2048 | 16409 | BN128-split → BN256-grouped | 2852.3 | 2862.00 | +0.34% | 2895.00 | +1.50% | +1.15% | 2850.90–2879.10 | 2879.70–2926.90 | 2854.0–2870.0 | 2883.0–2907.0 | 2873.00 → 2899.00 | +0.90% | 2866.35 → 2892.90 | +0.93% |
| 4096 | 32686 | BN128-split → BN256-grouped | 5337.9 | 5450.50 | +2.11% | 5453.00 | +2.16% | +0.05% | 5427.60–5483.30 | 5437.00–5468.20 | 5437.0–5463.0 | 5447.0–5458.0 | 5462.50 → 5458.00 | -0.08% | 5452.70 → 5449.70 | -0.06% |
| 8192 | 65538 | BN128-split → BN256-grouped | 10398.5 | 10195.50 | -1.95% | 10558.50 | +1.54% | +3.56% | 10174.90–10212.50 | 10535.70–10615.10 | 10183.0–10202.0 | 10548.0–10575.0 | 10200.50 → 10571.50 | +3.64% | 10192.85 → 10560.00 | +3.60% |

## Equal-weight 11-point geometric mean latency change

- rank0: -2.779%
- mean-rank: -2.824%
- max-rank: -2.749%

Each median interval is the distribution-free 95.7226% X10–X21 interval from all 30 process values.

The baseline uses its stock automatic layout (BN256 through M=1024, BN128 from M=2048). The final release policy intentionally keeps the MiMo geometry on BN256 grouped-nibble for all eleven points.
