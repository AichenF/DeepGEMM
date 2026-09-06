# Iteration 713o: overlay kernel namespace failure

Date: 2026-09-06

The first TP4 M128 cold-L2 smoke using the miniforge RDC library and the
accepted JIT overlay exits in Python import setup on all four ranks.  No
distributed communicator, CUDA Graph, correctness check, timing loop, or CUDA
business kernel is reached.

The overlay root contains namespace-only compatibility stubs outside
`sglang/jit_kernel`, including
`sglang/kernels/ops/moe/__init__.py`.  Top-level overlay-first resolution
therefore selects that stub when the exact old-checkout
`moe_align_block_size.py` imports:

```text
from sglang.kernels.ops.moe import moe_align_block_size
```

The stub exports no function, so every rank raises:

```text
ImportError: cannot import name 'moe_align_block_size' from
'sglang.kernels.ops.moe'
```

Decision: preserve overlay-first selection for `sglang.jit_kernel`, but force
the lightweight MoE-align dependency's `sglang.kernels` and
`sglang.kernels.ops` namespace roots to the original checkout before importing
the real `sglang.kernels.ops.moe` initializer.  This is an import-routing fix
only; no benchmark, route, kernel, or communication implementation changes.

Raw log:
`bench/results/iter713o_rdc_tp4_m128_cold_smoke_20260906.log`.
