# Iteration 608 evidence

Command contract: production graph benchmark with only
`V4_SINGLE_LAUNCH_TP4=1`; bundle/component environment overrides unset.  Each
reported replay is preceded by an excluded 256 MiB L2 clear.  Warmup=5,
outer=2, replays/outer=20.

| TP | M | active/padded rows | split-K | median ms | min ms | max ms | cosine min-rank | rel-L2 max-rank | embedded AR |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|:---|
| 4 | 8 | 43/344 | 4 | 0.075872 | not retained in summary | not retained in summary | 0.9999957922 | 0.0029011667 | multicast push |
| 4 | 16 | 82/656 | 4 | 0.125824 | not retained in summary | not retained in summary | 0.9999955955 | 0.0029680026 | multicast push |
| 4 | 32 | 140/1120 | 4 | 0.202976 | not retained in summary | not retained in summary | 0.9999956213 | 0.0029592747 | multicast push |
| 4 | 64 | 203/1624 | 2 | 0.286592 | not retained in summary | not retained in summary | 0.9999956225 | 0.0029589234 | P2P two-shot |
| 4 | 128 | 248/1992 | 2 | 0.355456 | not retained in summary | not retained in summary | 0.9999955977 | 0.0029672640 | P2P two-shot |
| 8 | 8 | 43/344 | 4 | 0.055104 | 0.053760 | 0.199616 | 0.9999921603 | 0.0039597519 | multicast push |
| 8 | 16 | 82/656 | 4 | 0.078720 | 0.077824 | 0.099008 | 0.9999919594 | 0.0040103194 | multicast push |
| 8 | 32 | 140/1120 | 4 | 0.123328 | 0.121824 | 0.131520 | 0.9999919617 | 0.0040095810 | NVLS pull |
| 8 | 64 | 203/1624 | 2 | 0.171312 | 0.169792 | 0.233536 | 0.9999920369 | 0.0039908141 | NVLS pull |
| 8 | 128 | 248/1992 | 2 | 0.219504 | 0.218112 | 0.240832 | 0.9999919587 | 0.0040103522 | NVLS pull |

All shapes reported `finite_all_ranks=true` and `allreduce_ok=true`.  TP4
geometric-mean median was 0.1815799608 ms; TP8 was 0.1150037560 ms.

Default metadata also exposed the remaining wiring issue:

```
single_launch_compact_w13_bundle=true
single_launch_m128_bound9=true
single_launch_assume_valid_gemm_tasks=false
single_launch_release_grid_arrival=false
```
