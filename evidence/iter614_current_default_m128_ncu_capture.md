# Iteration 614 evidence

NCU connected to `/usr/bin/python3.12`, selected exactly the production
`tp4_megamoe_single_launch_kernel<2,128>`, completed 21 kernel-replay passes,
and saved:

```text
/home/xutingz/fac/profile_results/iter614_current_default_m128_compute.ncu-rep
```

Application correctness record:

```json
{"accepted":true,"down_check":{"cosine":1.0,"finite":true,"rel_l2":0.0},"l2_policy":"cold 256MiB clear outside profiled kernel","m":128,"packed_generation_words_after":[2048,2048,2048,2048],"padded_rows":1992,"tp_collective_executed":false,"w13_split_k":2}
```

Counter values are deliberately deferred until the saved report is imported;
the capture command did not print them to stdout.
