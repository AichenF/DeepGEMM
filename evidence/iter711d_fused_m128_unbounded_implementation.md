# Iteration 711d — occupancy-derived unbounded M128 implementation

## Hypothesis

Standalone W2 retains 32-QGMMA/16-wait issue at the 64-register/eight-CTA
tier, while the fused entry still serializes at its 64-register ceiling.  Give
only the fused M128 specialization an effectively unconstrained ptxas
register contract, then derive the complete resident grid from CUDA's runtime
occupancy query.  This may expose the transient register headroom ptxas needs
to schedule both N64 QGMMAs before one dependency wait.

## Change

- Add default-off `V4_SINGLE_LAUNCH_M128_UNBOUNDED` to Python validation, JIT
  identity and compiler definitions.
- For TP4 `Tokens==128` only, select `__launch_bounds__(128,1)` instead of the
  production min-blocks 8/9 contract.  M=8/16/32/64 and every TP8
  specialization are unchanged.
- Reuse the existing host occupancy query and clamp the requested eight CTAs
  per SM to the actual active-block count.  Therefore a later test, if
  admitted, still launches an entirely co-resident software-grid-barrier
  population.
- Require the selected schedule-0 compact-W13/dynamic-smem path, bound9 off,
  both 702-grid wave rotations off, and ordinary inline W2.
- Report the experiment explicitly in the single-vs-multi benchmark metadata.

## Pre-CUDA gate

Python bytecode compilation passes for kernel and benchmark.  Staged SHA256:

```text
v4_flash_tp_wgmma.py: 28b8ac7a215bad395e0f61eeca451d69be5628acf85ae7532362edadd9793d13
bench/v4_flash_tp_single_vs_multi_graph.py: 16bb9e09d6d23247ea2291f08992f48c1cd394309829913732c6bff9c55901bb
```

No JIT or CUDA launch is claimed yet.  The candidate advances only if the
exact M128 fused W2 interval reaches 32 QGMMAs / about 16 waits with no local
spill and runtime occupancy of at least one full 78-CTA communication wave.

