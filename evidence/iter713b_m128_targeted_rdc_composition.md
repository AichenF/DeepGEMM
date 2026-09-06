# Iteration 713b: M128-targeted RDC composition

Date: 2026-09-06

## Hypothesis

Iteration 713a proved that a drop-in all-shape RDC build is incompatible with
the existing lower-M and TP8 64-register caller contracts.  It did not report
that conflict for TP4 M128, whose selected entry is deliberately unbounded.
The remaining narrow question is whether device linking changes that M128
entry's outlined-W2 schedule from fused 32-QGMMA/32-wait behavior toward the
standalone 32-QGMMA/16-wait behavior.

## Temporary generated-source changes

Starting from the exact Iteration-711g generated CUDA source used in 713a:

```diff
-        ? 9 : K_SINGLE_LAUNCH_MIN_BLOCKS;
+        ? 9 : 1;
@@
-__global__ __launch_bounds__(128, 8)
+__global__ __launch_bounds__(128, 1)
 void tp8_megamoe_single_launch_kernel(
```

These changes only relax unrelated entry register contracts so ptxas can
produce an RDC object.  They are not a candidate implementation and must not
be copied into production source.  No math, task ownership, synchronization,
communication or call ABI changes.

Patched generated-source SHA256:
`85dafc4cd81d0fa52599430df9da301c48d5d17f11b888da47d554521f695a9b`.

## Acceptance gate

1. Compile with `-rdc=true` and complete `nvcc -dlink`.
2. Identify the exact linked TP4 M128 split-K2 entry and outlined W2 callee.
3. Require 32 W2 QGMMAs with about 16 dependency barriers and no prohibitive
   fixed local spill.
4. If the exact W2 schedule remains 32/32, reject without a GPU launch.

## Static result

The relocatable compile and `nvcc -dlink` both succeed after the two temporary
launch-bound relaxations.

- Build log SHA256:
  `0e73bcc28f960fd310a1ea9987041229206da5b35ffa3d709f8bb360791cee13`.
- Device-link resource log SHA256:
  `380fdb80f899bd6cf05595f134a183dfbe86c9b57d153954d61ea66b76b0fbc2`.
- Linked full-SASS SHA256:
  `750c9c2a952809a6775af68c6511534460b767c1c26de62c4ff752ddd5a347b7`.
- Exact outlined-W2 SASS SHA256:
  `adf4eb6fd60c7954ca7c2bec966d6dc005c160d0f78d66fbc27804daca7f418e`.
- Exact W2 instruction counts: 32 `QGMMA`, 16 `WARPGROUP.DEPBAR`, and
  32 `WARPGROUP.ARRIVE`.
- TP4 M128 split-K2 entry resources:
  `REG195 STACK112 SHARED1616 LOCAL0`.

The 32/16 static acceptance gate therefore passes.  This establishes that
RDC, unlike same-translation-unit outlining, can preserve paired W2 issue.
It does not establish a performance win: the entry's register allocation
increases from the prior roughly 80 registers to 195, and device calls add a
112-byte per-thread stack frame.  Those costs can collapse occupancy and add
local-stack traffic.

## Decision

Link the temporary objects into a loadable extension, then run only TP4 M128
compute-only correctness and phase timing.  Reject before a production
multi-translation-unit refactor unless the runtime result beats the current
same-source fused phase despite the register and stack costs.

No CUDA business kernel was launched for this static result.
