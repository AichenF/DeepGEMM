# Group GEMM Advanced Optimization Strategies

These optimization techniques are derived from high-performance implementations like SonicMoE.

## 1. L2 Cache Locality via Tile Grouping

Group adjacent tiles to maximize L2 cache reuse. Tiles that share input data should be scheduled close together.

## 2. Raster Order Optimization

Choose tile traversal order based on problem shape for optimal cache behavior.

## 7. Variable-Length M Dimension (Varlen Scheduling)

Handle problems where different batches have different sequence lengths:

```cpp
struct VarlenScheduler {
    int* cu_seqlens;
    int num_batches;
    int tile_m;

    __device__ void find_batch_for_tile(
        int tile_m_idx,
        int* out_batch_idx,
        int* out_m_start,
        int* out_m_end
    ) {
        int m_global = tile_m_idx * tile_m;

        int lo = 0, hi = num_batches;
        while (lo < hi) {
            int mid = (lo + hi) / 2;
            if (cu_seqlens[mid + 1] <= m_global) {
                lo = mid + 1;
            } else {
                hi = mid;
            }
        }

        *out_batch_idx = lo;
        *out_m_start = cu_seqlens[lo];
        *out_m_end = cu_seqlens[lo + 1];
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

// High-K groups always evict B first (many K-iterations = many B loads)
if (K > 4096) cB = EVICT_FIRST;
```

**Applied via inline PTX load hints**:
```cpp
asm volatile("ld.global.L2::evict_last.b32 %0, [%1];" : "=r"(val) : "l"(ptr));
asm volatile("ld.global.L2::evict_first.b32 %0, [%1];" : "=r"(val) : "l"(ptr));
```

## 12. Persistent vs Simple Kernel Split

Use two kernel variants based on workload size for optimal performance:

- **Persistent kernel**: For large workloads (total_tiles > NUM_SMS). All SMs stay active, tiles distributed via lookup table.
- **Simple kernel**: For small workloads (total_tiles ≤ NUM_SMS). One tile per block, no persistence loop.

The simple kernel benefits from:
- Smaller argument copy
- No persistence loop overhead

### Memory Bandwidth
- Loading pointer arrays and strides adds global memory traffic
- Amortize by processing multiple tiles per problem fetch
- Prefetch problem info to shared memory when possible
