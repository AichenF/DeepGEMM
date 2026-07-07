# Hopper TMA activation-overlap excerpts

Source: `cudac-tensor-core-expert/references/wgmma_03_tma_tensor_map_creation.md`

**Important Note on BK and Swizzling**:
Take K-major matrix loading for example (including Row-Major Matrix A [M, K] and Column-Major Matrix B [K, N]), when using 128B swizzling mode with bf16 elements, BK must be set to 64 or a multiple thereof.
This ensures proper alignment with the swizzle layout atom structure (8x64 for K-major layout with 128B swizzling), which is critical for TMA to function correctly.
Using BK=16 will cause memory access errors because it doesn't align with the swizzle atom boundaries required by the 128B swizzling pattern.

Similarly:
When using 128B swizzling mode with fp16 elements, BK must be set to 64 or a multiple thereof.
When using 128B swizzling mode with fp8 elements, BK must be set to 128 or a multiple thereof.
When using 128B swizzling mode with tf32 elements, BK must be set to 32 or a multiple thereof.

**Important Note on OOB Fill when `global_width < box_width`**:
When `box_width` exceeds `global_width` (e.g., `box_width=64` but `global_width=32` for a small head dimension), TMA fills out-of-bounds elements according to the `oob_fill` parameter:

- `CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE`: TMA **skips writing** OOB positions — smem retains prior contents (often zeros on fresh allocation, but **not guaranteed**). Appears to work on B200 due to zero-initialized dynamic smem, but is **unstable** — relies on implementation-specific behavior that may break across GPU generations or with smem reuse.
- `CU_TENSOR_MAP_FLOAT_OOB_FILL_NAN_REQUEST_ZERO_FMA`: TMA **actively writes NaN** into OOB positions. Classic FMA treats NaN as zero, but **Gen5 `tcgen05.mma` does NOT honor this semantic** — NaN propagates through MMA and corrupts results. **Proven broken on B200 for tcgen05.mma.**

Source: `cudac-tensor-core-expert/references/wgmma_04_tma_loading_pattern.md`

##### Barrier Initialization

```cpp
#include <cuda/barrier>

using barrier = cuda::barrier<cuda::thread_scope_block>;
namespace cde = cuda::device::experimental;

###pragma nv_diag_suppress static_var_with_dynamic_init
__shared__ barrier barA;
__shared__ barrier barB;

if (threadIdx.x == 0) {
    init(&barA, blockDim.x);
    init(&barB, blockDim.x);
    cde::fence_proxy_async_shared_cta();
}
__syncthreads();
```

**Key Point**: Only thread 0 initiates TMA; all threads must arrive at barrier.

WIDTH is the maximum number of elements that fit in a 128-byte swizzle block for a given data type. The swizzle width depends on the element type:
When using 128B swizzling mode, TMA requires input in contiguous blocks aligned to the swizzle width.

- **bf16/fp16**: WIDTH = 64 elements (128 bytes / 2 bytes)
- **fp8**: WIDTH = 128 elements (128 bytes / 1 byte)
- **tf32**: WIDTH = 32 elements (128 bytes / 4 bytes)

Key Insight: When the major dimension (e.g., BLOCK_K) is larger than the swizzling atom leading dimension (WIDTH), you
must issue **multiple 2D TMA loads** to fetch the complete tile. Each TMA load fetches a [WIDTH, :] slice.

Source: `cudac-tensor-core-expert/references/optimization_01_mbarrier.md`

#### 2. Concept: phases, parity, and tx-count

An `mbarrier` phase completes when:

- arrival count reaches 0, and
- tx-count reaches 0

We reuse each slot across many phases. The **parity bit** tells you which phase you’re waiting for.

#### 7. Critical tips / common failure modes

- [ ] `full[]` and `empty[]` are `.b64` aligned to 8 bytes in shared memory.
- [ ] Barriers are initialized exactly once per block before use.
- [ ] Parity bit `p` flips consistently on slot reuse (commonly on `qidx` wrap).
- [ ] `full[]` expected arrivals is **1** (producer only); consumers do not call `arrive()` on it.
- [ ] `empty[]` expected arrivals equals `num_consumers` and **exactly one thread per consumer warpgroup** calls `arrive()` per slot.
- [ ] Producer calls `expect_bytes(full[q], bytes_total)` **before** issuing all copies linked to `full[q]`.
- [ ] The sum of copy bytes linked to `full[q]` equals `bytes_total`.
- [ ] Do not mix CUDA barrier objects with `mbarrier` in the same queue protocol.
- [ ] Do not reinitialize `mbarrier` objects per tile or per K-iteration; reuse phases via parity.
- [ ] If results “sometimes pass, sometimes fail”, suspect: wrong byte count, wrong parity flip, or tensor-map addressability (tma pointer not in device memory).
- [ ] Store mapping uses warpgroup-local indexing (`tid = threadIdx.x % 128`) when writing WGMMA fragments.
