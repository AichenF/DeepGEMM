// Why the SM90 MXFP4 MegaMoE stores its weights as contiguous, bulk-copied
// tiles -- measured, so the choice can be rechecked on a new part rather than
// taken on faith.
//
// A pipeline stage wants BLOCK_N weight rows of one k-block. Addressing those
// rows through a 2D TMA, as the row-major layout required, makes the stage
// BLOCK_N strided requests; grouping the tile makes it one bulk copy. This
// runs each scheme on its own, 132 (or 78) resident CTAs issuing loads and
// nothing else, and reports the throughput each sustains.
//
// Needs only nvcc and an sm_90a device -- no torch, no cluster:
//   nvcc -O3 -arch=sm_90a -o bench_sm90_weight_load \
//        tests/bench_sm90_weight_load.cu -lcuda -L/usr/local/cuda/lib64/stubs
//   ./bench_sm90_weight_load
//
// Measured H200 / H20-3e: the strided 80 B rows the kernel used to issue reach
// 2.77 / 2.80 TB/s, the shipped contiguous tiles 4.53 / 4.54, and the device
// itself streams about 4.2 TB/s. Contiguity, not the byte count, was the whole
// deficit. -DKSTAGES=N varies the pipeline depth; it makes no difference above
// four, which is why the depth in the selector is not tuned for bandwidth.
#include <cuda.h>
#include <cuda_runtime.h>
#include <cstdio>
#include <cstdint>
#include <vector>
#include <type_traits>

#define CHK(x) do { auto e = (x); if (e != cudaSuccess) { \
    printf("cuda error %s @%d\n", cudaGetErrorString(e), __LINE__); exit(1);} } while(0)
#define DCHK(x) do { CUresult e = (x); if (e != CUDA_SUCCESS) { const char* s; \
    cuGetErrorString(e, &s); printf("cu error %s @%d\n", s, __LINE__); exit(1);} } while(0)

constexpr int kBlockN   = 256;
#ifndef KSTAGES
#define KSTAGES 6
#endif
constexpr int kStages   = KSTAGES;
constexpr int kRowA     = 80;     // bytes per k-block row, layout A
constexpr int kRowB     = 64;     // bytes per k-block row, layout B
constexpr int kSfBytes  = 4 * kBlockN;   // 1024 B of E8M0 per tile

__device__ __forceinline__ void mbar_init(uint64_t* b, uint32_t c) {
    asm volatile("mbarrier.init.shared::cta.b64 [%0], %1;" :: "l"(__cvta_generic_to_shared(b)), "r"(c));
}
__device__ __forceinline__ void mbar_expect(uint64_t* b, uint32_t bytes) {
    asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %1;"
                 :: "l"(__cvta_generic_to_shared(b)), "r"(bytes));
}
__device__ __forceinline__ void mbar_arrive(uint64_t* b) {
    asm volatile("mbarrier.arrive.shared::cta.b64 _, [%0];" :: "l"(__cvta_generic_to_shared(b)));
}
__device__ __forceinline__ void mbar_wait(uint64_t* b, uint32_t phase) {
    asm volatile("{\n.reg .pred P;\nLAB_WAIT:\n"
                 "mbarrier.try_wait.parity.shared::cta.b64 P, [%0], %1;\n"
                 "@P bra DONE;\nbra LAB_WAIT;\nDONE:\n}\n"
                 :: "l"(__cvta_generic_to_shared(b)), "r"(phase));
}
__device__ __forceinline__ void tma2d(const CUtensorMap* m, uint64_t* bar, void* dst,
                                      int32_t c0, int32_t c1) {
    asm volatile("cp.async.bulk.tensor.2d.shared::cluster.global.tile.mbarrier::complete_tx::bytes"
                 " [%0], [%1, {%3, %4}], [%2];"
                 :: "l"(__cvta_generic_to_shared(dst)), "l"(m),
                    "l"(__cvta_generic_to_shared(bar)), "r"(c0), "r"(c1) : "memory");
}
__device__ __forceinline__ void bulk1d(void* dst, const void* src, uint32_t bytes, uint64_t* bar) {
    asm volatile("cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes"
                 " [%0], [%1], %2, [%3];"
                 :: "l"(__cvta_generic_to_shared(dst)), "l"(src), "r"(bytes),
                    "l"(__cvta_generic_to_shared(bar)) : "memory");
}

// mode 0 = A (one 2D TMA, 80 B rows), 1 = B (2D TMA 64 B rows + 2D TMA scale row),
// mode 2 = C (one 1D bulk copy of the packed tile)
template <int kMode, int kRowBytes>
__global__ __launch_bounds__(128) void stream_kernel(
        const __grid_constant__ CUtensorMap map,
        const __grid_constant__ CUtensorMap sf_map,
        const uint8_t* __restrict__ raw, int num_tiles, int k_blocks,
        uint64_t tile_stride, uint32_t* __restrict__ sink) {
    extern __shared__ __align__(1024) uint8_t smem[];
    constexpr uint32_t kTileBytes = kRowBytes * kBlockN;
    constexpr uint32_t kStageBytes =
        (kMode == 0 or kMode == 3 or kMode == 4) ? kTileBytes : kTileBytes + kSfBytes;
    uint64_t* bars = reinterpret_cast<uint64_t*>(smem + kStages * kStageBytes);
    const uint32_t tid = threadIdx.x;
    if (tid == 0) {
        for (int i = 0; i < kStages * 2; ++i) mbar_init(bars + i, 1);
    }
    asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
    __syncthreads();

    uint32_t acc = 0;
    const int total = num_tiles * k_blocks;
    int stage = 0; uint32_t ph = 0;
    // prologue
    for (int s = 0; s < kStages && s < total; ++s) {
        if (tid == 0) {
            uint8_t* dst = smem + s * kStageBytes;
            const int tile = blockIdx.x * num_tiles + s / k_blocks, kb = s % k_blocks;
            if constexpr (kMode == 0) {
                mbar_expect(bars + s, kStageBytes);
                tma2d(&map, bars + s, dst, kb * kRowBytes, tile * kBlockN);
            } else if constexpr (kMode == 1) {
                mbar_expect(bars + s, kStageBytes);
                tma2d(&map, bars + s, dst, 0, (tile * k_blocks + kb) * kBlockN);
                tma2d(&sf_map, bars + s, dst + kTileBytes, 0, (tile * k_blocks + kb) * (kSfBytes / 256));
            } else if constexpr (kMode == 4) {
                mbar_expect(bars + s, kStageBytes);
                const uint8_t* src = raw + (uint64_t)(tile * k_blocks + kb) * (kStageBytes / 2);
                bulk1d(dst, src, kStageBytes / 2, bars + s);
                bulk1d(dst + kStageBytes / 2, src + tile_stride, kStageBytes / 2, bars + s);
            } else {
                mbar_expect(bars + s, kStageBytes);
                bulk1d(dst, raw + (uint64_t)(tile * k_blocks + kb) * tile_stride, kStageBytes, bars + s);
            }
        }
    }
    for (int i = 0; i < total; ++i) {
        mbar_wait(bars + stage, ph);
        // One word per thread is enough to keep the stage dependency; a full
        // sweep of the tile makes the consumer, not the TMA, the bottleneck.
        const uint32_t* p = reinterpret_cast<const uint32_t*>(smem + stage * kStageBytes);
        acc ^= p[tid];
        __syncthreads();
        const int nxt = i + kStages;
        if (tid == 0 && nxt < total) {
            uint8_t* dst = smem + stage * kStageBytes;
            const int tile = blockIdx.x * num_tiles + nxt / k_blocks, kb = nxt % k_blocks;
            if constexpr (kMode == 0) {
                mbar_expect(bars + stage, kStageBytes);
                tma2d(&map, bars + stage, dst, kb * kRowBytes, tile * kBlockN);
            } else if constexpr (kMode == 1) {
                mbar_expect(bars + stage, kStageBytes);
                tma2d(&map, bars + stage, dst, 0, (tile * k_blocks + kb) * kBlockN);
                tma2d(&sf_map, bars + stage, dst + kTileBytes, 0, (tile * k_blocks + kb) * (kSfBytes / 256));
            } else if constexpr (kMode == 4) {
                mbar_expect(bars + stage, kStageBytes);
                const uint8_t* src = raw + (uint64_t)(tile * k_blocks + kb) * (kStageBytes / 2);
                bulk1d(dst, src, kStageBytes / 2, bars + stage);
                bulk1d(dst + kStageBytes / 2, src + tile_stride, kStageBytes / 2, bars + stage);
            } else {
                mbar_expect(bars + stage, kStageBytes);
                bulk1d(dst, raw + (uint64_t)(tile * k_blocks + kb) * tile_stride, kStageBytes, bars + stage);
            }
        } else if (tid == 0) {
            mbar_arrive(bars + stage);
        }
        if (++stage == kStages) { stage = 0; ph ^= 1; }
    }
    if (acc == 0xdeadbeefu) sink[0] = acc;
}

static CUtensorMap make_map(void* addr, uint64_t rows, uint64_t row_bytes,
                            uint64_t stride_bytes, uint32_t box_bytes, uint32_t box_rows) {
    CUtensorMap m{};
    uint64_t dims[2]   = {row_bytes, rows};
    uint64_t strides[1] = {stride_bytes};
    uint32_t box[2]    = {box_bytes, box_rows};
    uint32_t elem[2]   = {1, 1};
    DCHK(cuTensorMapEncodeTiled(&m, CU_TENSOR_MAP_DATA_TYPE_UINT8, 2, addr, dims, strides,
                                box, elem, CU_TENSOR_MAP_INTERLEAVE_NONE,
                                CU_TENSOR_MAP_SWIZZLE_NONE, CU_TENSOR_MAP_L2_PROMOTION_NONE,
                                CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE));
    return m;
}

int main() {
    CHK(cudaFree(nullptr));
    int sms = 0; CHK(cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, 0));
    cudaDeviceProp prop{}; CHK(cudaGetDeviceProperties(&prop, 0));
    printf("device %s, %d SMs\n", prop.name, sms);

    // One expert-sized working set per SM: 48 k-blocks x 256 rows, x num_tiles tiles.
    const int k_blocks = 48;
    const int tiles_per_sm = 12;
    const int num_tiles = sms * tiles_per_sm;

    uint32_t* sink; CHK(cudaMalloc(&sink, 4));

    struct Res { const char* name; double gbps; double us; size_t bytes; };
    std::vector<Res> res;

    // ---- A: 80 B rows, k-major within a row-major (tile_row, k) plane
    {
        const uint64_t rows = (uint64_t)num_tiles * kBlockN;
        const uint64_t row_bytes = (uint64_t)k_blocks * kRowA;
        uint8_t* buf; CHK(cudaMalloc(&buf, rows * row_bytes)); CHK(cudaMemset(buf, 1, rows * row_bytes));
        auto map = make_map(buf, rows, row_bytes, row_bytes, kRowA, kBlockN);
        CUtensorMap dummy = map;
        size_t smem = (size_t)kStages * kRowA * kBlockN + kStages * 2 * 8;
        CHK(cudaFuncSetAttribute(stream_kernel<0, kRowA>, cudaFuncAttributeMaxDynamicSharedMemorySize, smem));
        cudaEvent_t e0, e1; CHK(cudaEventCreate(&e0)); CHK(cudaEventCreate(&e1));
        for (int r = 0; r < 3; ++r) stream_kernel<0, kRowA><<<sms, 128, smem>>>(map, dummy, buf, tiles_per_sm, k_blocks, 0, sink);
        CHK(cudaDeviceSynchronize());
        CHK(cudaEventRecord(e0));
        for (int r = 0; r < 10; ++r) stream_kernel<0, kRowA><<<sms, 128, smem>>>(map, dummy, buf, tiles_per_sm, k_blocks, 0, sink);
        CHK(cudaEventRecord(e1)); CHK(cudaDeviceSynchronize());
        float ms; CHK(cudaEventElapsedTime(&ms, e0, e1));
        size_t bytes = (size_t)sms * tiles_per_sm * k_blocks * kBlockN * kRowA;
        res.push_back({"A 2D TMA, 80B rows (today)", bytes / (ms / 10 * 1e-3) / 1e12, ms / 10 * 1e3, bytes});
        CHK(cudaFree(buf));
    }
    // ---- B: 64 B rows contiguous per (tile,k) + a separate 1 KB scale row
    {
        const uint64_t rows = (uint64_t)num_tiles * k_blocks * kBlockN;
        uint8_t* buf; CHK(cudaMalloc(&buf, rows * kRowB)); CHK(cudaMemset(buf, 1, rows * kRowB));
        const uint64_t sf_rows = (uint64_t)num_tiles * k_blocks;
        uint8_t* sf; CHK(cudaMalloc(&sf, sf_rows * kSfBytes)); CHK(cudaMemset(sf, 1, sf_rows * kSfBytes));
        auto map    = make_map(buf, rows, kRowB, kRowB, kRowB, kBlockN);
        auto sf_map = make_map(sf, sf_rows * (kSfBytes / 256), 256, 256, 256, kSfBytes / 256);
        size_t smem = (size_t)kStages * (kRowB * kBlockN + kSfBytes) + kStages * 2 * 8;
        CHK(cudaFuncSetAttribute(stream_kernel<1, kRowB>, cudaFuncAttributeMaxDynamicSharedMemorySize, smem));
        cudaEvent_t e0, e1; CHK(cudaEventCreate(&e0)); CHK(cudaEventCreate(&e1));
        for (int r = 0; r < 3; ++r) stream_kernel<1, kRowB><<<sms, 128, smem>>>(map, sf_map, buf, tiles_per_sm, k_blocks, 0, sink);
        CHK(cudaDeviceSynchronize());
        CHK(cudaEventRecord(e0));
        for (int r = 0; r < 10; ++r) stream_kernel<1, kRowB><<<sms, 128, smem>>>(map, sf_map, buf, tiles_per_sm, k_blocks, 0, sink);
        CHK(cudaEventRecord(e1)); CHK(cudaDeviceSynchronize());
        float ms; CHK(cudaEventElapsedTime(&ms, e0, e1));
        size_t bytes = (size_t)sms * tiles_per_sm * k_blocks * (kBlockN * kRowB + kSfBytes);
        res.push_back({"B 2D TMA, 64B rows + SF row", bytes / (ms / 10 * 1e-3) / 1e12, ms / 10 * 1e3, bytes});
        CHK(cudaFree(buf)); CHK(cudaFree(sf));
    }
    // ---- C: one 1D bulk copy of the whole 17408 B tile
    {
        const uint64_t tile_stride = (uint64_t)kRowB * kBlockN + kSfBytes;
        const uint64_t n = (uint64_t)num_tiles * k_blocks * tile_stride;
        uint8_t* buf; CHK(cudaMalloc(&buf, n)); CHK(cudaMemset(buf, 1, n));
        CUtensorMap dummy = make_map(buf, 1024, 256, 256, 16, 1);
        size_t smem = (size_t)kStages * (kRowB * kBlockN + kSfBytes) + kStages * 2 * 8;
        CHK(cudaFuncSetAttribute(stream_kernel<2, kRowB>, cudaFuncAttributeMaxDynamicSharedMemorySize, smem));
        cudaEvent_t e0, e1; CHK(cudaEventCreate(&e0)); CHK(cudaEventCreate(&e1));
        for (int r = 0; r < 3; ++r) stream_kernel<2, kRowB><<<sms, 128, smem>>>(dummy, dummy, buf, tiles_per_sm, k_blocks, tile_stride, sink);
        CHK(cudaDeviceSynchronize());
        CHK(cudaEventRecord(e0));
        for (int r = 0; r < 10; ++r) stream_kernel<2, kRowB><<<sms, 128, smem>>>(dummy, dummy, buf, tiles_per_sm, k_blocks, tile_stride, sink);
        CHK(cudaEventRecord(e1)); CHK(cudaDeviceSynchronize());
        float ms; CHK(cudaEventElapsedTime(&ms, e0, e1));
        size_t bytes = (size_t)sms * tiles_per_sm * k_blocks * tile_stride;
        res.push_back({"C 1D bulk copy, 17408B tile", bytes / (ms / 10 * 1e-3) / 1e12, ms / 10 * 1e3, bytes});
        CHK(cudaFree(buf));
    }

    // ---- D/E: one 1D bulk copy of a contiguous tile, no separate scale plane.
    //          E keeps today's 80 B row image byte-for-byte, so only the
    //          addressing changes; D is the 64 B compact row.
    auto run_mode3 = [&](auto row_tag, const char* name) {
        constexpr int kRow = decltype(row_tag)::value;
        const uint64_t tile_stride = (uint64_t)kRow * kBlockN;
        const uint64_t n = (uint64_t)num_tiles * k_blocks * tile_stride;
        uint8_t* buf; CHK(cudaMalloc(&buf, n)); CHK(cudaMemset(buf, 1, n));
        CUtensorMap dummy = make_map(buf, 1024, 256, 256, 16, 1);
        size_t smem = (size_t)kStages * kRow * kBlockN + kStages * 2 * 8;
        CHK(cudaFuncSetAttribute(stream_kernel<3, kRow>, cudaFuncAttributeMaxDynamicSharedMemorySize, smem));
        cudaEvent_t e0, e1; CHK(cudaEventCreate(&e0)); CHK(cudaEventCreate(&e1));
        for (int r = 0; r < 3; ++r) stream_kernel<3, kRow><<<sms, 128, smem>>>(dummy, dummy, buf, tiles_per_sm, k_blocks, tile_stride, sink);
        CHK(cudaDeviceSynchronize());
        CHK(cudaEventRecord(e0));
        for (int r = 0; r < 10; ++r) stream_kernel<3, kRow><<<sms, 128, smem>>>(dummy, dummy, buf, tiles_per_sm, k_blocks, tile_stride, sink);
        CHK(cudaEventRecord(e1)); CHK(cudaDeviceSynchronize());
        float ms; CHK(cudaEventElapsedTime(&ms, e0, e1));
        size_t bytes = (size_t)sms * tiles_per_sm * k_blocks * tile_stride;
        res.push_back({name, bytes / (ms / 10 * 1e-3) / 1e12, ms / 10 * 1e3, bytes});
        CHK(cudaFree(buf));
    };
    run_mode3(std::integral_constant<int, kRowA>{}, "E 1D bulk, contiguous 80B rows");
    run_mode3(std::integral_constant<int, kRowB>{}, "D 1D bulk, 64B rows, no SF");

    // The shipped layout: a stage is two 8704 B tiles `k_blocks` apart, which
    // is what the kernel's (expert, n_tile, k_block) ordering produces.
    auto run_split = [&](uint64_t split_tiles, const char* name) {
        constexpr int kRow = 68;
        const uint64_t stage = (uint64_t)kRow * kBlockN;      // 17408
        const uint64_t split = split_tiles ? split_tiles * (stage / 2) : stage / 2;
        const uint64_t n = (uint64_t)num_tiles * k_blocks * stage + split + stage;
        uint8_t* buf; CHK(cudaMalloc(&buf, n)); CHK(cudaMemset(buf, 1, n));
        CUtensorMap dummy = make_map(buf, 1024, 256, 256, 16, 1);
        size_t smem = (size_t)kStages * stage + kStages * 2 * 8;
        CHK(cudaFuncSetAttribute(stream_kernel<4, kRow>, cudaFuncAttributeMaxDynamicSharedMemorySize, smem));
        cudaEvent_t e0, e1; CHK(cudaEventCreate(&e0)); CHK(cudaEventCreate(&e1));
        for (int r = 0; r < 3; ++r) stream_kernel<4, kRow><<<sms, 128, smem>>>(dummy, dummy, buf, tiles_per_sm, k_blocks, split, sink);
        CHK(cudaDeviceSynchronize());
        CHK(cudaEventRecord(e0));
        for (int r = 0; r < 10; ++r) stream_kernel<4, kRow><<<sms, 128, smem>>>(dummy, dummy, buf, tiles_per_sm, k_blocks, split, sink);
        CHK(cudaEventRecord(e1)); CHK(cudaDeviceSynchronize());
        float ms; CHK(cudaEventElapsedTime(&ms, e0, e1));
        size_t bytes = (size_t)sms * tiles_per_sm * k_blocks * stage;
        res.push_back({name, bytes / (ms / 10 * 1e-3) / 1e12, ms / 10 * 1e3, bytes});
        CHK(cudaFree(buf));
    };
    run_split(48, "G 2x8704B k_blocks apart (today)");

    printf("\n%-32s %10s %10s %12s\n", "variant", "TB/s", "us/iter", "MB/iter");
    for (auto& r : res)
        printf("%-32s %10.3f %10.1f %12.1f\n", r.name, r.gbps, r.us, r.bytes / 1e6);
    return 0;
}
