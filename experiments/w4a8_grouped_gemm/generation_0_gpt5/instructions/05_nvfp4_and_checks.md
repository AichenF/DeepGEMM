## Memory Layout Details

### FP4 Data Packing in Shared Memory

Two 4-bit elements are packed into each byte:
```
Byte layout: [elem1][elem0]
             high 4  low 4
```

For a (128, 64) tile:
- 128 rows × 64 columns = 8192 elements
- 8192 elements / 2 = 4096 bytes
- With 128-byte swizzling applied

### Scale Factor Format

Scale factors use UE4M3 format (8-bit):
- 4-bit unsigned exponent
- 3-bit mantissa
- No sign bit

For a (128, 64) MMA with `.block16`:
- scale_A: (128, 4) elements → K dimension divided into 4 chunks (64/16=4)
- scale_B: (4, 64) elements → K dimension divided into 4 chunks
- Reshaped to (32, 4, 4) for tensor memory → 512 bytes each

##### Descriptor Leading Dimension Mismatch
**Problem**: Using wrong leading dimension for matrix layout.
**Solution**:
- For row-major [M, N]: leading_dim = N x sizeof(element)
- Always verify the actual memory stride

##### Shared Memory Indexing for N-major sB
**Problem**: Using `&sB[WGMMA_K]` when N is contiguous.
**Solution**: Use `&sB[WGMMA_K * WIDTH]` to skip full WGMMA_K * WIDTH block, not just K elements. WIDTH is the maximum number of elements that fit in a 128-byte swizzle block for a given data type.

##### WGMMA TransB Parameter Mismatch
**Problem**: Using wrong TransB flag causes illegal memory access or incorrect results. For example, using `wgmma256<1,1,1,0,0>` (TransB=0) when B is row-major [K, N].
**Solution**: Keep TranA=0 with common K-major case, and match TransB to B's memory layout:
- **TransA=0**: For row-major A [M, K] (K-major, where K is the contiguous dimension)
- **TransB=1**: For row-major B [K, N] (N-major, where N is the contiguous dimension)
- **TransB=0**: For column-major B [K, N] (K-major, where K is the contiguous dimension)
The TransB flag tells WGMMA how to interpret the swizzled memory access pattern defined by the shared memory descriptor.

- [ ] Descriptor leading dimensions match actual matrix strides
- [ ] Descriptor matrix size parameters are correct for tile dimensions
- [ ] TMA tensor map shape is [width, height] = [minor_dim, major_dim]
- [ ] TMA coordinates are (col_offset, row_offset)
- [ ] Shared memory indexing advances correctly in K dimension
- [ ] Both descriptor and TMA use 128B swizzle
- [ ] All threads participate in barrier synchronization
- [ ] Output store pattern matches WGMMA register distribution
- [ ] TransA/TransB flags match actual matrix layouts in shared memory
