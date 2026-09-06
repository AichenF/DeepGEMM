# Iteration 712a — function-specific W2 register-contract implementation

## Hypothesis

The frozen standalone W2 entry uses 64 registers and emits two adjacent N64
QGMMAs per dependency wait, while the one-launch W2 phase serializes those
halves under every tested caller/lifetime arrangement.  CUDA exposes a
function-specific `__maxnreg__` attribute; applying a 64-register contract to
the outlined W2 callee may preserve standalone-like scheduling without a
second business launch.

## Change

Add default-off `V4_SINGLE_LAUNCH_W2_FUNC_MAXNREG64`.  It requires TP4,
schedule 0, the unbounded M128 entry and the whole-W2 phase outline.  The
outlined device function receives `__maxnreg__(64)` only in that build.  When
the experiment is enabled, lower-token specializations retain their inline
W2 path; TP8 and the multi-kernel entries are unchanged.  The flag is part of
the JIT identity, compiler definitions and paired benchmark metadata.

No arithmetic, route/task mapping, intermediate boundary, synchronization,
communication, public ABI or default behavior changes.

## Pre-CUDA gate

Both Python sources pass bytecode compilation.  Source SHA256 values are:

- `v4_flash_tp_wgmma.py`:
  `11d5135412b3d3d5f185b11d693774e0dcd0eef7449206a23b2767f0cfeea5d5`
- `bench/v4_flash_tp_single_vs_multi_graph.py`:
  `d4b4f29f6e482c714b2e52ecbc1bb1aeea7ec9c3d27134a6d551685e3be17690`

No JIT or CUDA launch is claimed.  Compile next and reject before a business
launch unless the exact M128 W2 callee has no fixed local spill and about 16
dependency waits for 32 QGMMAs.

