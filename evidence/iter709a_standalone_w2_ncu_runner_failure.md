# Iteration 709a: standalone W2 NCU runner failure

The intended TP4 M128 standalone-W2 SourceCounters collection exited before
any CUDA kernel was profiled.  The custom-only invocation entered
`v4_bench_env_runner.py`, whose unconditional Humming setup imported
`humming/dtypes.py`; megamoe's Torch does not expose
`torch.float8_e8m0fnu`, producing:

```text
AttributeError: module 'torch' has no attribute 'float8_e8m0fnu'
==ERROR== The application returned an error code (1).
```

No JIT kernel launch, L2 clear, correctness result, timing sample, or NCU
metric was produced.  The retry will call only the runner's pinned-SGLang
environment setup before executing the unchanged custom local profiler.
