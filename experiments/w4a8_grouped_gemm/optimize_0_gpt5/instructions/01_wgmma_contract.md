##### WGMMA Operation
- **Instruction**: `wgmma.mma_async.sync.aligned.m64n64k16.f32.bf16.bf16`
- **Tile Size**: Computes a 64x64 output tile using 64x16 (A) and 16x64 (B) input tiles
- **Data Flow**: D[64x64] = A[64x16] x B[16x64] + D[64x64]
- **K-dimension**: Must iterate K/16 times to accumulate full result

Instruction wgmma.mma_async issues a MxNxK matrix multiply and accumulate operation, D = A*B+D, where the A matrix is MxK, the B matrix is KxN, and the D matrix is MxN.

The operation of the form D = A*B is issued when the input predicate argument scale-d is false.

wgmma.fence instruction must be used to fence the register accesses of wgmma.mma_async instruction from their prior accesses. Otherwise, the behavior is undefined.

wgmma.commit_group and wgmma.wait_group operations must be used to wait for the completion of the asynchronous matrix multiply and accumulate operations before the results are accessed.

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

