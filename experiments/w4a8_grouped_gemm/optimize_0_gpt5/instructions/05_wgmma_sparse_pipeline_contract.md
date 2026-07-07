###### FP8 floating point type

wgmma.mma_async.sync.aligned.shape.dtype.atype.btype  d, a-desc, b-desc, scale-d, imm-scale-a, imm-scale-b;

wgmma.mma_async.sync.aligned.shape.dtype.atype.btype  d, a, b-desc, scale-d, imm-scale-a, imm-scale-b;

.shape   = {.m64n8k32, .m64n16k32, .m64n24k32, .m64n32k32,
            .m64n40k32, .m64n48k32, .m64n56k32, .m64n64k32,
            .m64n72k32, .m64n80k32, .m64n88k32, .m64n96k32,
            .m64n104k32, .m64n112k32, .m64n120k32, .m64n128k32,
            .m64n136k32, .m64n144k32, .m64n152k32, .m64n160k32,
            .m64n168k32, .m64n176k32, .m64n184k32, .m64n192k32,
            .m64n200k32, .m64n208k32, .m64n216k32, .m64n224k32,
            .m64n232k32, .m64n240k32, .m64n248k32, .m64n256k32};
.atype  = {.e4m3, .e5m2};
.btype  = {.e4m3, .e5m2};
.dtype  = {.f16, .f32};

##### WGMMA Register Distribution
Each thread in the warp group holds a portion of the 64x64 result:
- **d[4][8]**: 32 float registers per thread
- Distribution pattern: 2x2 element blocks per thread across the tile

##### 2.1 Thread layout: warpgroups

On Hopper, one WGMMA warpgroup is **128 threads (4 warps)**. Partition the CTA into warpgroups:

```cpp
constexpr int NUM_THREADS = 128 * (1 + num_consumers);
const int wg_idx = threadIdx.x / 128;   // 0 = producer, >=1 = consumer
const int tid    = threadIdx.x % 128;   // 0..127 within warpgroup
```

Rule of thumb: start with `num_consumers = 1` (one producer + one consumer).

#### 5. Critical tip: store mapping must use warpgroup-local `tid`

WGMMA register layouts and the standard store mapping are defined **within a single warpgroup**. Always compute store indices from warpgroup-local `tid`:

```cpp
const int tid  = threadIdx.x % 128;
const int lane = tid % 32;
const int warp = tid / 32; // 0..3
```

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

## 12. Persistent vs Simple Kernel Split

Use two kernel variants based on workload size for optimal performance:

- **Persistent kernel**: For large workloads (total_tiles > NUM_SMS). All SMs stay active, tiles distributed via lookup table. Uses full `KernelParams` (~3.5KB).
- **Simple kernel**: For small workloads (total_tiles ≤ NUM_SMS). One tile per block, no persistence loop. Uses compact `SimpleKernelParams` (~984B).

