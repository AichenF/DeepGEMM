# Iteration 621 evidence

```text
extension=v4tp_05e8202bb68fd087b718_v178mspec
bundle=true
balanced_w2_workers=true
balanced_activation_workers=false
balanced_workers=false
m128_bound9=true
```

Selected resource records:

```text
tp4_megamoe_single_launch_kernel<2,128> REG:56 STACK:32 SHARED:2048 LOCAL:0
tp4_megamoe_single_launch_kernel<4,128> REG:56 STACK:32 SHARED:2048 LOCAL:0
tp4_megamoe_single_launch_kernel<2,64>  REG:64 STACK:32 SHARED:2048 LOCAL:0
tp4_megamoe_single_launch_kernel<4,64>  REG:64 STACK:32 SHARED:2048 LOCAL:0
```

M128 compute-only result after the separate 256 MiB cache clear:

```json
{"accepted": true, "down_check": {"cosine": 1.0, "finite": true,
 "rel_l2": 0.0}, "m": 128,
 "packed_generation_words_after": [0, 0, 0, 0],
 "packed_generation_wrap_ok": true, "padded_rows": 1992,
 "sm_count": 78, "tp_collective_executed": false, "w13_split_k": 2}
```

The raw JIT/cuobjdump/correctness transcript is
`bench/results/iter621_balanced_w2_workers_m128_gate_20260905.log`.
