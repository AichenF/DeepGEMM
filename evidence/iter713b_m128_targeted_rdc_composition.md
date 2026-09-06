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

No CUDA compilation or launch is claimed by this composition record.
