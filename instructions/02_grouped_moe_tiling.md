## 1. L2 Cache Locality via Tile Grouping

Group adjacent tiles to maximize L2 cache reuse. Tiles that share input data should be scheduled close together.

```cpp
// L2 group size: number of tiles processed before moving to next region
// Larger values = better L2 reuse, but may reduce parallelism
constexpr int L2_GROUP_SIZE = 8;

// When scheduling tiles, group them in 2D blocks
// E.g., process an 8x1 or 4x2 group of tiles before moving on
struct TileGroupScheduler {
    int group_size_m;
    int group_size_n;

    // Convert linear tile index to grouped (m, n) coordinates
    __device__ int2 get_tile_coord(int linear_idx, int tiles_m, int tiles_n) {
        // Number of complete groups
        int tiles_per_group = group_size_m * group_size_n;
        int groups_per_row = (tiles_n + group_size_n - 1) / group_size_n;

        int group_idx = linear_idx / tiles_per_group;
        int within_group = linear_idx % tiles_per_group;

        int group_m = group_idx / groups_per_row;
        int group_n = group_idx % groups_per_row;

        int local_m = within_group / group_size_n;
        int local_n = within_group % group_size_n;

        return make_int2(
            group_m * group_size_m + local_m,
            group_n * group_size_n + local_n
        );
    }
};
```

## 10. L2 Cache Policy Tuning

Use cache eviction hints to control L2 residency based on data size and reuse patterns.

```cpp
// A data: small footprint, high reuse across N-tiles → always keep in L2
uint64_t cA = EVICT_LAST;

// B data: size-dependent policy
// Large B thrashes L2, hurting A reuse; evict large B early
uint64_t total_B_bytes = sum_of_all_groups_B_size;
uint64_t global_cb = (total_B_bytes > 72LL*1024*1024) ? EVICT_FIRST : EVICT_LAST;

// Per-group override for very large individual groups
uint64_t B_bytes_this_group = N * K * sizeof(element);
uint64_t cB = (B_bytes_this_group > 8*1024*1024) ? EVICT_FIRST : global_cb;

// High-K groups always evict B first (many K-iterations = many B loads)
if (K > 4096) cB = EVICT_FIRST;

// Scale factor (SF) data: always small → keep in L2
uint64_t cSF = EVICT_LAST;
```

**Applied via inline PTX load hints**:
```cpp
// EVICT_LAST: hint to keep in L2 as long as possible
asm volatile("ld.global.L2::evict_last.b32 %0, [%1];" : "=r"(val) : "l"(ptr));
// EVICT_FIRST: hint to evict from L2 quickly
asm volatile("ld.global.L2::evict_first.b32 %0, [%1];" : "=r"(val) : "l"(ptr));
```

## 12. Persistent vs Simple Kernel Split

Use two kernel variants based on workload size for optimal performance:

- **Persistent kernel**: For large workloads (total_tiles > NUM_SMS). All SMs stay active, tiles distributed via lookup table. Uses full `KernelParams` (~3.5KB).
- **Simple kernel**: For small workloads (total_tiles ≤ NUM_SMS). One tile per block, no persistence loop. Uses compact `SimpleKernelParams` (~984B).

```cpp
if (total_tiles <= NUM_SMS) {
    // Simple kernel: 1 tile per block, compact params
    SimpleKernelParams skp;
    // ... pack skp ...
    launch_simple_kernel<<<total_tiles, THREADS, SMEM>>>(skp);
} else {
    // Persistent kernel: all SMs active, loop over tiles
    KernelParams kp;
    // ... pack kp with full tile table ...
    launch_persistent_kernel<<<NUM_SMS, THREADS, SMEM>>>(kp);
}
```

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
