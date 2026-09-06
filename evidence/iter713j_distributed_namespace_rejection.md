# Iteration 713j: distributed namespace rejection

Date: 2026-09-06

The first four-rank TP4 smoke selected
`V4_SGLANG_NAMESPACE_SUBPACKAGES=sglang.srt.distributed`.  This was too broad:
the checkout's `sglang.srt.distributed.__init__` normally re-exports
`GroupCoordinator`, and `dp_attention.py` imports that public name while the
world group is constructed.

All four ranks exit with `ImportError: cannot import name 'GroupCoordinator'`
before communicator construction, CUDA Graph capture, correctness, or timing.
No CUDA business kernel launched and no performance claim is made.

Decision: remove only the distributed namespace selection.  Retain the
top-level SGLang isolation and exact lightweight MoE-align re-export, then let
the original distributed package initializer run unchanged.
