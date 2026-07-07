**Example 2: Column-major B `'mk,nk->mn'`**
```python
A = torch.randn(M, K)  # Shape: [M, K], row-major, K contiguous
B = torch.randn(N, K)  # Shape: [N, K], row-major, K contiguous
C = torch.einsum('mk,nk->mn', A, B)

# Memory interpretation:
# - A[i, j] is at: A_ptr[i * K + j]
# - B[i, j] is at: B_ptr[i * K + j]
# - For WGMMA: A is row-major [M, K] does not need transpose TransA=0, B is effectively column-major [K, N], needs TransB=0
# - This is more efficient because both A and B have K contiguous!
```

##### Key Implications for WGMMA

###### Understanding "Which Dimension is Contiguous"
The memory layout determines how you iterate through the K dimension:

1. **Row-major A [M, K]**: K is contiguous
   - To advance by 16 elements in K: offset by `+16` elements

2. **Row-major B [K, N]**: N is contiguous
   - To advance by 16 rows in K: offset by `+16 * N` elements

3. **Column-major B [K, N]**: K is contiguous
   - To advance by 16 elements in K: offset by `+16` elements

###### Example: 128B Swizzling with fp8 Elements, TMA loads Smem block with BK = WGMMA_K*4 = 128 (B is always K-major since wgmma does not support transposing for fp8, BK has to be 128x to align with Swizzle layout atom fast dimension)

For K-major wA [WGMMA_MxWGMMA_K] = [64x32] matrix, sA [BMxBK] = [64x128] matrix:
Swizzle layout atom has a shape of 8x(8*(128/8)) = 8x128, in K-dimension, there is only 1 atom, so LBO = 1; SBO = 8*128*1*1(byte) = 1024.

For K-major wB [WGMMA_KxWGMMA_N] = [32x64] matrix, sB [BKxBN] = [128x64] matrix:
Swizzle layout atom has a shape of (8*(128/8))x8 = 128x8, in K-dimension, there is only 1 atom, so LBO = 1; SBO = 128x8*1*1(byte) = 1024.

##### Descriptor Encoding Function
```cpp
__device__ static inline uint64_t matrix_descriptor_encode(uint64_t x) {
    return (((x) & 0x3FFFF) >> 0x4);
}
```
**Purpose**: Encodes values by masking to 18 bits and shifting right by 4 (divides by 16).
