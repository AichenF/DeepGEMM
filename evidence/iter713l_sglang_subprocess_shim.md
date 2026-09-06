# Iteration 713l: subprocess-visible SGLang checkout shim

Date: 2026-09-06

`bench/iter713_sglang_shim/sglang/__init__.py` provides a filesystem package
entry for helper processes that do not execute the prebuilt-extension runner.
It reads `V4_SGLANG_NAMESPACE_ROOT`, verifies the selected checkout's
`sglang` directory, and assigns that directory as its sole `__path__`.

The shim contains no SGLang API implementation.  In particular it does not
replace `parallel_state`, P2P checks, JIT code, symmetric-memory handling, or
`CustomAllReduceV2`; those leaf modules continue to load from the exact old
CAR checkout.  Placing the shim directory first in `PYTHONPATH` makes this
selection visible to CAR's fresh Python subprocess as well as torchrun ranks.

Python bytecode compilation passes.  No CUDA business kernel launched for
this import-environment change.
