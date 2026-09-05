# Iteration 626: 156 CTA x four independent WGs static gate

Date: 2026-09-05

The new default-off `V4_SINGLE_LAUNCH_156CTA_4WG` specialization uses two
512-thread CTAs per H20 SM and four private 128-thread route-GEMM task groups
per CTA.  It preserves 624 logical GEMM workers while reducing physical
whole-grid barrier participants to 156.

Isolation and observability checks:

- The 156x4 and historical 78x8 modes are mutually exclusive.
- The new path rejects compact-W13/bound-9, alternate scheduler, overlap,
  producer-combine, task-prefetch, and task-granularity experiments.
- Dynamic shared memory is four route-task slabs; launch bounds request two
  512-thread CTAs per SM and the host requires the occupancy query to admit
  exactly two.
- The JIT cache key, NVCC macro list, same-source benchmark metadata, full
  graph metadata, and compute trace sizing all include the specialization.
- Production TP4 defaults and the separate TP8 entry are unchanged.

Static command:

```text
python3 -m py_compile v4_flash_tp_wgmma.py \
  bench/profile_v4_flash_tp_single_compute.py \
  bench/v4_flash_tp_single_vs_multi_graph.py \
  bench/v4_flash_tp_wgmma_graph.py
```

Result: exit code 0, no diagnostics.

Qualification: static source gate only.  No CUDA object was compiled or run,
and no correctness or performance conclusion is made in this iteration.
