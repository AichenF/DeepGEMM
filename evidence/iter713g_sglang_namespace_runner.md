# Iteration 713g: SGLang namespace-only runner

Date: 2026-09-06

The baseline-compatible SGLang checkout's top-level `sglang/__init__.py`
imports frontend/network APIs, including optional packages not installed in
the benchmark container.  The paired test only needs SRT distributed state,
JIT helpers, and `CustomAllReduceV2`.

`bench/iter713_prebuilt_extension_runner.py` now accepts
`V4_SGLANG_NAMESPACE_ROOT`.  When set, it creates a Python namespace package
whose `__path__` is `<root>/sglang`, without executing the top-level
`__init__.py`.  Imports such as
`sglang.srt.distributed.device_communicators.custom_all_reduce_v2` still load
the exact files from that source tree; no CAR implementation is stubbed or
replaced.

The option is off by default.  Python bytecode compilation is required before
use.  This is import isolation only and changes neither benchmark logic nor
CUDA code.  No CUDA business kernel has launched for this change.
