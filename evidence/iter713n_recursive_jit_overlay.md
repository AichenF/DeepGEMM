# Iteration 713n: recursive JIT overlay selection

Date: 2026-09-06

The runner and subprocess-visible shim now accept
`V4_SGLANG_NAMESPACE_OVERLAY_ROOTS`.  Each comma-separated root contributes a
validated `<root>/sglang` entry before the original checkout directory in the
top-level package `__path__`.

The selected value is `/home/xutingz/fac/.tpmoe_tmp`.  That tree contains the
full copied `sglang/jit_kernel` package used by earlier accepted baselines,
including the two-line TVM-FFI 0.1.11 IPC tuple compatibility change.  It does
not contain `sglang.srt`, so distributed Python modules and CARv2 policy still
fall through to the exact old checkout.

This path ordering is applied identically in torchrun workers and the CAR P2P
helper subprocess.  No module implementation is synthesized, production
checkout remains unmodified, and the environment variables are default-off.
Both Python files compile; no CUDA business kernel launched for this change.
