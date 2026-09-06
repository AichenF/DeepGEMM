# Iteration 713p: pin MoE kernel ops to the selected checkout

Date: 2026-09-06

When `V4_SGLANG_LIGHT_MOE_ALIGN=1`, the prebuilt-extension runner now installs
`sglang.kernels` and `sglang.kernels.ops` as namespace packages rooted only at
the selected original checkout before importing the exact alignment leaf.
Consequently, `sglang.kernels.ops.moe` executes the original registry-backed
initializer and exports its real `moe_align_block_size` function.

The top-level `sglang` namespace remains overlay-first, so
`sglang.jit_kernel` and the accepted TVM-FFI IPC compatibility header still
come from `/home/xutingz/fac/.tpmoe_tmp`.  This narrows path ownership instead
of altering alignment, CAR, route, GEMM, or communication code.  The change is
active only under the already opt-in lightweight-alignment runner mode.

Pre-CUDA validation: the runner passes `py_compile`.
