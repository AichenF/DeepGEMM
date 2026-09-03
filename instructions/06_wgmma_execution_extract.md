##### Accumulation Loop
```cpp
// Compute over K dimension
warpgroup_arrive();
for (int k = 0; k < BK/WGMMA_K; ++k) {
    wgmma64<1, 1, 1, 0, 0>(d, &sA[k*WGMMA_K], &sB[k*WGMMA_K]);
}
warpgroup_commit_batch();
warpgroup_wait<0>();
```

**For BK=64, WGMMA_K=16**: Need 4 WGMMA operations per block.

**Important**: The indexing pattern depends on the memory layout of sA and sB:

**Example 1 - sA row-major [BM, BK], sB column-major [BK, BN]**:
```cpp
// For sA [64, 64] row-major: advance by WGMMA_K columns
// For sB [64, 64] column-major (K contiguous): advance by WGMMA_K elements
warpgroup_arrive();
wgmma64<1, 1, 1, 0, 0>(d, &sA[0*WGMMA_K], &sB[0*WGMMA_K]);
wgmma64<1, 1, 1, 0, 0>(d, &sA[1*WGMMA_K], &sB[1*WGMMA_K]);
wgmma64<1, 1, 1, 0, 0>(d, &sA[2*WGMMA_K], &sB[2*WGMMA_K]);
wgmma64<1, 1, 1, 0, 0>(d, &sA[3*WGMMA_K], &sB[3*WGMMA_K]);
warpgroup_commit_batch();
warpgroup_wait<0>();
```

**Explanation for sB column-major indexing**:
- sB is stored as `[BK, BN] = [64, 64]` in column-major format
- The K dimension is contiguous (stride-1)
- To advance by WGMMA_K=16 elements in the K dimension, simply offset by 16 elements

- Leading dimension is BK=64 (the major dimension stride)
