# Iteration 623 evidence

Default-off experiment:

```text
V4_SINGLE_LAUNCH_W2_N64_TAIL=1
```

Residual-wave geometry for the fixed random-route cases:

| shape | full N128 tasks | residual N128 | residual N64 | physical CTAs |
|---:|---:|---:|---:|---:|
| M64 | 6,240 | 256 | 512 | 624 |
| M128 | 7,722 | 246 | 492 | 702 |

Only the residual wave changes.  The implementation dynamically retains N128
when `2 * residual_tasks > physical_ctas`, so arbitrary route distributions do
not accidentally introduce a second tail wave.

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

No CUDA build, correctness, or latency result is claimed in this iteration.
