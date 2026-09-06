# Iteration 710b — make the W2 outline branch structurally explicit

## Change

Replace the historical

```cpp
} else for (...) {
```

form after `kSingleLaunchW2PhaseNoInline` with an explicit
`else { for (...) { ... } }` scope.  The loop body, bounds, task order,
WGMMA instantiation, synchronization and every production-default condition
are unchanged.

This is required because the current large TP4 template fails NVCC
instantiation when the preceding `if constexpr` is true, despite lexically
balanced braces.  The explicit scope removes that parser ambiguity and is
also the correct control-flow foundation for a later terminal W2 device tail.

## Pre-CUDA gate

- `python3 -m py_compile` passes.
- Staged source SHA256:
  `716ce0b3745597895511391c6eb9bb09a01a993b234e2828bc9475a5b051954a`.
- The new branch remains default-off; selected production semantics do not
  change.

No JIT, CUDA execution, correctness result or latency is claimed here.  Next
rebuild the exact Iteration-710a opt-in and require an emitted cubin before
inspecting its device-callee schedule.
