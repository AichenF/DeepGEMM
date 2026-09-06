# Iteration 709c: standalone W2 quant-fixture failure

Pinned SGLang and a minimal Humming import stub allowed the custom graph
helper to import, but its case constructor then called
`humming_ops.quant_input`.  The stub intentionally did not emulate Humming's
numerical input path, and Python raised:

```text
AttributeError: module 'humming.ops' has no attribute 'quant_input'
==ERROR== The application returned an error code (1).
```

This occurred before any JIT kernel launch, cache clear, correctness result,
timing sample, or NCU metric.  Further work will use the current formal
same-source harness's already validated pure-Torch FP8 input construction
instead of adding another compatibility shim.
