# Iteration 713k: P2P subprocess namespace failure

Date: 2026-09-06

After allowing the selected checkout's distributed package initializer to run,
all ranks construct their world groups.  CARv2 then invokes its normal
`gpu_p2p_access_check`, which launches
`custom_all_reduce_utils.py` in a fresh Python subprocess on rank 0.

That subprocess does not inherit the runner's in-memory `sys.modules` package
namespace.  It resolves a different SGLang checkout supplied by the miniforge
environment's `.pth` files, executes that checkout's top-level frontend API,
and fails because optional `aiohttp` is absent.  Peer ranks subsequently lose
their Gloo connection.

The failure precedes CAR communicator creation, CUDA Graph capture,
correctness, timing, and every MegaMoE business-kernel launch.

Decision: provide a small filesystem `sglang/__init__.py` shim before all
other `PYTHONPATH` entries.  Its only job is to set the package search path to
`V4_SGLANG_NAMESPACE_ROOT/sglang`, making the same source selection effective
inside helper subprocesses.
