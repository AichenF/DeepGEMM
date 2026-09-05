# Iteration 630: 156 CTA × 4 WG M8 compute gate

## Conditions

- Device: one H20, physical GPU 1; 78 SM.
- Input: prequantized FP8-E4M3 X plus FP32 group-128 scale.
- Route pattern/seed: random / 20260902.
- Cache policy: cold 256 MiB clear outside the profiled kernel.
- Kernel topology: 156 CTAs × 512 threads, four independent 128-thread groups per CTA.
- TP collective: disabled to isolate the compute-body launch and output.

## Result

```json
{"accepted":true,"m":8,"sm_count":78,"w13_split_k":4,"padded_rows":344,"down_check":{"cosine":1.0,"rel_l2":0.0,"finite":true}}
```

The launch path's exact two-resident-CTA occupancy check did not reject the kernel, and the complete output matches the multi-kernel reference bit for bit. This validates the packed group's route, W13, activation/requant, and W2 sequencing for M8, but not TP communication or end-to-end performance.
