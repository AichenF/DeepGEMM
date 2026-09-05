# Iteration 616 evidence

Change under test:

```text
V4_SINGLE_LAUNCH_BALANCED_ACTIVATION_WORKERS=1
```

The new path changes only schedule-0 activation assignment for `Tokens >= 64`:

```text
rounds  = ceil(activation_groups / ctas)
workers = ceil(activation_groups / rounds)
```

W13 and W2 worker counts are unchanged.  At TP4 M128 this maps 3,072 activation
tasks from 702 workers (264 five-task and 438 four-task workers) onto 615
workers (612 five-task and 3 four-task workers).

Local static check:

```text
python3 -m py_compile \
  v4_flash_tp_wgmma.py \
  bench/v4_flash_tp_single_vs_multi_graph.py \
  bench/v4_flash_tp_paired_graph.py \
  bench/v4_flash_tp_wgmma_graph.py
exit_code=0
stdout/stderr=<empty>
```

No GPU benchmark or correctness claim is made in this iteration.
