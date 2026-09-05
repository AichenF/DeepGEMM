# Iteration 631: 156 CTA × 4 WG M128 compute gate

## Conditions

- Device: one H20, physical GPU 1; 78 SM.
- Input: prequantized FP8-E4M3 X plus FP32 group-128 scale.
- Route pattern/seed: random / 20260902.
- Cache policy: cold 256 MiB clear outside the profiled kernel.
- Kernel topology: 156 CTAs × 512 threads, four independent 128-thread groups per CTA.
- TP collective: disabled for this gate.

## Result

```json
{"accepted":true,"m":128,"sm_count":78,"w13_split_k":2,"padded_rows":1992,"down_check":{"cosine":1.0,"rel_l2":0.0,"finite":true}}
```

The large-M path launches at the required two resident CTAs/SM and matches the multi-kernel output bit for bit. Along with Iteration 630, this covers both route-size extremes before any performance decision.
