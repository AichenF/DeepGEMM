# Iteration 710a — current-source W2 phase-outline JIT failure

## Intent

Before implementing a terminal non-returning W2+collective device tail, audit
the already retained whole-W2 `__noinline__` phase callee.  If that callee
already recovers standalone W2's paired-QGMMA schedule, a non-returning tail
can target only its call/return and cross-phase continuation overhead.

## Configuration

- H20 physical GPU 1, compile only.
- `V4_SINGLE_LAUNCH_TP4=1`
- `V4_SINGLE_LAUNCH_W2_PHASE_NOINLINE=1`
- All Iteration 709 diagnostic pairing flags default off.
- Intended extension: `v4tp_b9572d91268c0bf6a995_v178mspec`.

## Result

**JIT failure before CUDA execution.**  NVCC reaches every TP4 token
specialization but reports `expected a "}"` at generated CUDA line 7421 while
instantiating the single-launch entry.  The generated function has balanced
lexical braces.  The failure appears only when
`kSingleLaunchW2PhaseNoInline` is compile-time true, at the historical
`else if constexpr (...) { phase_call; } else for (...) { ... }` chain.
This is consistent with an NVCC parser/instantiation regression in the
unbraced `else for` form after the now much larger diagnostic template.

No extension/cubin was emitted, no business kernel launched, no cache clear or
timing occurred, and this iteration provides no performance conclusion.

## Next action

Make the final branch structurally explicit as `else { for (...) { ... } }`,
without changing either branch's runtime work.  Re-run Python syntax, default
JIT and the opt-in JIT.  Only after the opt-in cubin exists may its device
callee QGMMA/wait schedule be inspected.

Raw log: `bench/results/iter710a_existing_w2_phase_outline_jit.log`.
