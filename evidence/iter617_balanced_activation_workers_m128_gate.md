# Iteration 617 evidence

Resolved configuration:

```json
{"assume_valid_tasks": true, "balanced_activation_workers": true,
 "balanced_workers": false, "bundle": true,
 "extension": "v4tp_d1173cd481ac0e15d647_v178mspec",
 "m128_bound9": true, "release_arrival": true}
```

Selected cubin resources:

```text
tp4_megamoe_single_launch_kernel<2,128> REG:56 STACK:32 SHARED:2048 LOCAL:0
tp4_megamoe_single_launch_kernel<4,128> REG:56 STACK:32 SHARED:2048 LOCAL:0
tp4_megamoe_single_launch_kernel<2,64>  REG:64 STACK:32 SHARED:2048 LOCAL:0
tp4_megamoe_single_launch_kernel<4,64>  REG:64 STACK:32 SHARED:2048 LOCAL:0
```

Compute-only M128 result after the separate 256 MiB cold-L2 clear:

```json
{"accepted": true, "down_check": {"cosine": 1.0, "finite": true,
 "rel_l2": 0.0}, "m": 128,
 "packed_generation_words_after": [0, 0, 0, 0],
 "packed_generation_wrap_ok": true,
 "packed_generation_wrap_requested": true, "padded_rows": 1992,
 "sm_count": 78, "tp_collective_executed": false, "w13_split_k": 2}
```

The host-side `tee` could not create
`bench/results/iter617_balanced_activation_workers_m128_gate_20260905.log`
because that directory had become root-owned.  Consequently the outer pipeline
exit code is 1 even though the inner commands above completed successfully.
This markdown evidence preserves the returned output; timing is deferred until
the result-directory ownership is repaired.
