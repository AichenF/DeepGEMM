##### WGMMA Synchronization Primitives
```cpp
__device__ void warpgroup_arrive() {
    asm volatile("wgmma.fence.sync.aligned;\n" ::: "memory");
}

__device__ void warpgroup_commit_batch() {
    asm volatile("wgmma.commit_group.sync.aligned;\n" ::: "memory");
}

template <int N>
__device__ void warpgroup_wait() {
    static_assert(N >= 0 && N <= 7, "N must be in range [0, 7]");
    asm volatile("wgmma.wait_group.sync.aligned %0;\n" ::"n"(N) : "memory");
}
```

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

##### WGMMA Register Distribution
Each thread in the warp group holds a portion of the 64x64 result:
- **d[4][8]**: 32 float registers per thread
- Distribution pattern: 2x2 element blocks per thread across the tile

##### Store Loop Structure

###### Example 1 - Row-Major Output C [M, N]
```cpp
int tid = threadIdx.x;
int lane = tid % 32;
int warp = tid / 32;
uint32_t row = warp*16 + lane / 4;  // Each thread covers certain rows

bf16 *block_C = C + num_block_m*BM*N + num_block_n*BN;

for (int m_it = 0; m_it < BM/WGMMA_M; ++m_it) {
    for (int n_it = 0; n_it < BN/WGMMA_N; ++n_it) {
        for (int w = 0; w < WGMMA_N/16; ++w) {
            int col = 16*w + 2*(tid % 4);

            // Row-major indexing: row * N + col
            #define IDX(i, j) ((i + m_it*WGMMA_M)*N + (j + n_it*WGMMA_N))

            block_C[IDX(row, col)]       = d[w][0];
            block_C[IDX(row, col+1)]     = d[w][1];
            block_C[IDX(row+8, col)]     = d[w][2];
            block_C[IDX(row+8, col+1)]   = d[w][3];
            block_C[IDX(row, col+8)]     = d[w][4];
            block_C[IDX(row, col+9)]     = d[w][5];
            block_C[IDX(row+8, col+8)]   = d[w][6];
            block_C[IDX(row+8, col+9)]   = d[w][7];

            #undef IDX
        }
    }
}
```
