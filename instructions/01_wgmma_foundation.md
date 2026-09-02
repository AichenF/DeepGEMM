##### WGMMA Operation
- **Instruction**: `wgmma.mma_async.sync.aligned.m64n64k16.f32.bf16.bf16`
- **Tile Size**: Computes a 64x64 output tile using 64x16 (A) and 16x64 (B) input tiles
- **Data Flow**: D[64x64] = A[64x16] x B[16x64] + D[64x64]
- **K-dimension**: Must iterate K/16 times to accumulate full result

1. **Maximize TMA Utilization**: Use 128B swizzling for optimal memory bandwidth
2. **Pipeline TMA and Compute**: Use double buffering with multiple barriers
3. **Minimize Register Pressure**: Reuse accumulator registers when possible
4. **Align Shared Memory**: Use `alignas(128)` for TMA requirements
5. **Warp Group Size**: Use 128 threads (4 warps) for optimal WGMMA performance
