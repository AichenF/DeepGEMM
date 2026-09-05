# Iteration 620 evidence

New default-off experiment:

```text
V4_SINGLE_LAUNCH_BALANCED_W2_WORKERS=1
```

Only the W2 loop's logical worker count changes for M>=64:

```text
w2_rounds  = ceil(w2_tasks / physical_ctas)
w2_workers = ceil(w2_tasks / w2_rounds)
```

For the fixed Iteration-615/619 random-route shapes:

| shape | W2 tasks | physical CTAs | rounds | W2 workers |
|---:|---:|---:|---:|---:|
| M64 | 6,496 | 624 | 11 | 591 |
| M128 | 7,968 | 702 | 12 | 664 |

W13, activation, physical launch dimensions, grid-barrier participant count,
and embedded collective geometry do not change.

Static check:

```text
python3 -m py_compile \
  v4_flash_tp_wgmma.py \
  bench/v4_flash_tp_single_vs_multi_graph.py \
  bench/v4_flash_tp_paired_graph.py \
  bench/v4_flash_tp_wgmma_graph.py
exit_code=0
stdout/stderr=<empty>
```

No CUDA launch or latency result is claimed here.
