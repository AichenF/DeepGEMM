# Iteration 709i: custom-only cold profile smoke

The repaired fixture completed on physical H20 GPU1 and emitted:

```text
LOCAL_PROFILE_REPLAY {"impl":"custom","m":128,"tp":4,
"route_pattern":"random","l2_cache_bytes":62914560,
"l2_flush_bytes":268435456,"l2_clear_impl":"tensor_zero",
"l2_policy":"cold; 256MiB clear immediately before pipeline",
"output_tile_channels":128,"weight_stages":2,
"w13_split_policy":"auto","w13_s2r_prefetch":true,
"w2_s2r_prefetch":true,"w2_route_output":true}
```

This is a liveness/cache-policy gate only; it reports no timing or tensor
correctness.  It qualifies the same fixture for the standalone-W2 NCU
collection.
