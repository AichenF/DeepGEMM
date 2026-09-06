# Iteration 725a: W2 B32 shared-decoded SS-WGMMA composition

## Hypothesis

The fused M128 entry emits one dependency wait per W2 WGMMA (32 QGMMA / 32
DEPBAR), while the standalone W2 entry emits two independent N64 operations
per wait (32 / 16).  The exact fused SASS aliases decoded RS source registers
with accumulator destinations.  Materializing each pair of decoded N64xK32
FP8 tiles in 4 KiB of B32-swizzled shared memory removes those register source
operands and should let ptxas retain the two-WGMMA batch.

## Isolated candidate

- Opt-in only: `V4_SINGLE_LAUNCH_W2_SHARED_DECODED=1`.
- TP4, schedule 0, inline bound-8, WOUT128, two packed stages only.
- W13, activation, k6 reduction, and embedded TP collective are unchanged.
- Each K32 step decodes both N64 halves into two 2 KiB B32 tiles, performs an
  async-proxy fence plus CTA convergence, issues two SS-WGMMAs, then reuses the
  scratch only after `warpgroup_wait<0>`.
- Dynamic shared memory grows from 18 KiB to 22 KiB per CTA; the requested
  eight CTAs/SM should remain feasible on the 78-SM H20, subject to the actual
  occupancy query.

## Hard gate

Compile a fresh M128 split-2 specialization and inspect the exact selected
entry before any business launch.  Continue only if:

1. registers <= 64, local spill bytes == 0, and actual occupancy remains 8;
2. exact W2 interval contains 32 QGMMA with <= 20 DEPBAR waits;
3. no unexpected call/return boundary appears in the W2 interval.

If the static gate passes, run correctness first, then same-process ABBA
cold-L2 CUDA Graph A/B at TP4 M128 and M8.  Reject on no reproducible latency
gain.
