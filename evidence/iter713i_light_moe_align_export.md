# Iteration 713i: exact lightweight MoE-align re-export

Date: 2026-09-06

The paired graph imports `moe_align_block_size` from
`sglang.srt.layers.moe.fused_moe_triton`.  In the selected older CAR checkout,
that package initializer first imports the full `FusedMoE`/MoeRunner/model
stack, although the benchmark never uses it.

With `V4_SGLANG_LIGHT_MOE_ALIGN=1`, the isolated runner:

1. treats the intervening MoE/triton-utils directories as namespace packages;
2. imports the checkout's exact
   `moe_runner/triton_utils/moe_align_block_size.py`; and
3. exposes that same function object under the public package name expected by
   the unchanged benchmark.

No implementation is copied, stubbed, or substituted.  Route alignment still
executes SGLang's original function and underlying kernel.  The option is off
by default; Python compilation passes.  No CUDA business kernel launched for
this import-only change.
