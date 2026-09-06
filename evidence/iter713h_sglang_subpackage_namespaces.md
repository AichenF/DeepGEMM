# Iteration 713h: selective SGLang subpackage namespaces

Date: 2026-09-06

After bypassing only `sglang/__init__.py`, the unchanged benchmark imports
`sglang.srt.layers.moe.fused_moe_triton`.  Python first executes
`sglang.srt.layers.moe.__init__`, which imports the full MoE runner and model
configuration stack and fails on an unrelated Transformers API mismatch.

The prebuilt-extension runner now accepts a comma-separated
`V4_SGLANG_NAMESPACE_SUBPACKAGES`.  For each exact `sglang.*` package name it
sets only the package search path to the corresponding directory under the
chosen checkout.  It does not stub leaf modules or functions.

For this benchmark the intended value is:

```text
sglang.srt.distributed,sglang.srt.layers.moe
```

This skips broad package initializers while retaining the checkout's exact
`parallel_state.py`, `fused_moe_triton.py`, JIT helpers and
`custom_all_reduce_v2.py`.  The option is off by default.  Python compilation
passes; no CUDA business kernel launched for this import-only change.
