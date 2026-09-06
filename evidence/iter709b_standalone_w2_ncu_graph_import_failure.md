# Iteration 709b: standalone W2 graph-import failure

The retry initialized the pinned SGLang overlay without calling the runner's
Humming setup.  It still exited before CUDA because
`v4_flash_tp_wgmma_graph.py` imports `humming.ops` for the benchmark's untimed
weight canonicalization.  Importing Humming's dtype table raised:

```text
AttributeError: module 'torch' has no attribute 'float8_e8m0fnu'
==ERROR== The application returned an error code (1).
```

No JIT kernel launch, L2 clear, correctness result, timing sample, or NCU
metric was produced.  The next bootstrap will provide an import-only dtype
alias and then install the same runner weight-preprocess replacement before
running the unchanged custom profiler.
