# Iteration 709e: custom profile weight-stub failure

The SM90a-only TP4 M128 local-profile smoke reached custom case construction
but exited before any CUDA kernel launch.  Weight scale normalization called
the untimed benchmark utility absent from the first custom-only stub:

```text
AttributeError: module 'humming.ops' has no attribute
'process_mxfp4_w4a8_weight'
```

No L2 clear, correctness result, timing sample, or NCU metric was produced.
The repair will reuse the exact Torch negative-zero-nibble canonicalizer
already present in `/home/xutingz/fac/v4_bench_env_runner.py`.
