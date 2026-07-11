# SM90 MegaMoE Environment Cleanup

## Goal

Remove the experimental SM90 MegaMoE environment-variable surface and the
kernel variants that exist only behind it. Preserve the current production
selector output and BF16 combine behavior.

## Production Selection

The selector receives only runtime facts: hardware SM count, ranks, experts,
tokens, top-k, shape, and SF pool capacity. Hardware class is derived from the
SM count. Tile sizes, stages, experts per wave, direct scatter, N-major order,
cleanup mode, and swap-AB are selected by the generic heuristic or the measured
range policies; none can be overridden through `DG_SM90_MOE_*` variables.

The existing default behavior becomes unconditional:

- BN256 split-N and legal swap-AB candidates remain available.
- Generic direct scatter, N-major scheduling, and one-warp cleanup remain off.
- Specialized range policies retain their current values.
- Experts per wave and pipeline stages remain derived or range-selected.

## Removed Variants

FP8 combine is removed end to end. The symmetric combine buffer always stores
BF16, shared-memory sizing always uses BF16 elements, and the combine reduction
has no E5M2 conversion branch or JIT template flag.

Phase profiling is removed with its kernel clocks, counters, template flag, and
oversized stats-buffer contract. EPLB, skew, and masked workload hints are also
removed; their former false/default behavior is folded into the range table.

The repository-wide `DG_JIT_DEBUG` and `DG_PRINT_CONFIGS` diagnostics remain.
They are not SM90 MegaMoE tuning controls.

## Verification

- Require zero `DG_SM90_MOE_*` references in production code and tests.
- Build and run the host selector golden test in Release mode.
- Rebuild the Python extension and JIT kernels.
- Run representative distributed correctness on H20 and H200.
- Compare the selected configs and benchmark results against the current
  default-env baseline; the cleanup must not introduce a stable regression.
