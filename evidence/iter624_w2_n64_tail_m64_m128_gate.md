# Iteration 624 evidence

```text
extension=v4tp_3cf360f93d0ea77f0518_v178mspec
bundle=true
w2_n64_tail=true
balanced_w2_workers=false
m128_bound9=true
```

Resources:

```text
tp4_megamoe_single_launch_kernel<2,64>  REG:64 STACK:32 SHARED:2048 LOCAL:0
tp4_megamoe_single_launch_kernel<4,64>  REG:64 STACK:32 SHARED:2048 LOCAL:0
tp4_megamoe_single_launch_kernel<2,128> REG:56 STACK:32 SHARED:2048 LOCAL:0
tp4_megamoe_single_launch_kernel<4,128> REG:56 STACK:32 SHARED:2048 LOCAL:0
```

Cold compute-only exact checks:

```text
M64:  accepted=true cosine=1 rel_l2=0 finite=true padded_rows=1624
      packed_generation_words_after=[0,0,0,0]
M128: accepted=true cosine=1 rel_l2=0 finite=true padded_rows=1992
      packed_generation_words_after=[0,0,0,0]
```

The collective was deliberately disabled only for this local state-machine
gate.  Raw transcript:
`bench/results/iter624_w2_n64_tail_m64_m128_gate_20260905.log`.
