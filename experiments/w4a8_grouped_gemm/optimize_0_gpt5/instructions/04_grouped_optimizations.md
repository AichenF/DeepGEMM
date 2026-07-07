## 4. Multi-Stage Software Pipelining

Overlap memory loads with computation using multiple pipeline stages.

## 9. Packed TileInfo Lookup Table

Replace O(ng) linear scan + integer division for tile-to-group mapping with O(1) pre-computed lookup. This is the single most impactful optimization (~4-5% improvement).

## 12. Persistent vs Simple Kernel Split

Use two kernel variants based on workload size for optimal performance:

- **Persistent kernel**: For large workloads (total_tiles > NUM_SMS). All SMs stay active, tiles distributed via lookup table. Uses full `KernelParams` (~3.5KB).
- **Simple kernel**: For small workloads (total_tiles ≤ NUM_SMS). One tile per block, no persistence loop. Uses compact `SimpleKernelParams` (~984B).

The simple kernel benefits from:
- 72% smaller argument copy (984B vs 3.5KB)
- Single-group TMA patching (only patch the group this block needs)
- No persistence loop overhead

### Common Anti-Patterns (Empirically Confirmed Regressions)

| Anti-Pattern | Regression | Why |
|-------------|-----------|-----|
| `cudaLaunchKernel` replacing `cuLaunchKernelEx` | -15 to -30% | Loses programmatic serialization |
| Device global + `cudaMemcpyAsync` per call | -10 to -60% | H2D memcpy overhead exceeds benefit |
| pybind11 / torch extension for launch | -8 to -540% | Python/C++ bridging overhead |
| BLOCK_N=64 (doubling tile count) | -10 to -21% | Per-tile overhead increase |
| Reducing NUM_STAGES below 6 | -2 to -7% | Less TMA/MMA overlap |
| Dual-path epilogue (full-tile + partial) | -2 to -4% | Register pressure from code duplication |

