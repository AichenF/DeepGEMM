# Iteration 610 evidence

Default-only contract:

```text
V4_SINGLE_LAUNCH_TP4=1
V4_SINGLE_LAUNCH_COMPACT_W13_BUNDLE unset
V4_SINGLE_LAUNCH_W13_PHASE_NOINLINE unset
V4_SINGLE_LAUNCH_W13_PHASE_COMPACT_ABI unset
V4_SINGLE_LAUNCH_ROUTE_DYNAMIC_SMEM unset
V4_SINGLE_LAUNCH_M128_BOUND9 unset
V4_SINGLE_LAUNCH_RELEASE_GRID_ARRIVAL unset
V4_SINGLE_LAUNCH_ASSUME_VALID_GEMM_TASKS unset
```

Both ranks-zero metadata records resolved:

```text
single_launch_compact_w13_bundle=true
single_launch_release_grid_arrival=true
single_launch_assume_valid_gemm_tasks=true
single_launch_m128_bound9=true
l2_policy=cold; 256MiB Triton clear before every replay, clear excluded
```

| TP | M | active/padded rows | split-K | median ms | min/max ms | cosine min-rank | rel-L2 max-rank | embedded AR |
|---:|---:|---:|---:|---:|---:|---:|---:|:---|
| 4 | 8 | 43/344 | 4 | 0.075808 | 0.075200/0.337280 | 0.9999957922 | 0.0029011667 | multicast push |
| 4 | 16 | 82/656 | 4 | 0.125216 | 0.122880/0.137248 | 0.9999955955 | 0.0029680026 | multicast push |
| 4 | 32 | 140/1120 | 4 | 0.198976 | 0.198240/0.223328 | 0.9999956213 | 0.0029592747 | multicast push |
| 4 | 64 | 203/1624 | 2 | 0.281984 | 0.280128/0.298592 | 0.9999956225 | 0.0029589234 | P2P two-shot |
| 4 | 128 | 248/1992 | 2 | 0.346560 | 0.345632/0.353760 | 0.9999955977 | 0.0029672640 | P2P two-shot |
| 8 | 8 | 43/344 | 4 | 0.054496 | 0.053728/0.242816 | 0.9999921603 | 0.0039597519 | multicast push |
| 8 | 16 | 82/656 | 4 | 0.078880 | 0.077600/0.098656 | 0.9999919594 | 0.0040103194 | multicast push |
| 8 | 32 | 140/1120 | 4 | 0.121888 | 0.120960/0.141600 | 0.9999919617 | 0.0040095810 | NVLS pull |
| 8 | 64 | 203/1624 | 2 | 0.169664 | 0.168928/0.193984 | 0.9999920369 | 0.0039908141 | NVLS pull |
| 8 | 128 | 248/1992 | 2 | 0.219072 | 0.217056/0.232704 | 0.9999919587 | 0.0040103522 | NVLS pull |

All rows reported `finite_all_ranks=true` and `allreduce_ok=true`.  TP4 and TP8
geometric-mean medians were 0.1791577916 and 0.1142598909 ms respectively.
The isolated high maxima at M8 are retained as evidence of system noise; they
are why this short gate is not used for the final performance claim.
