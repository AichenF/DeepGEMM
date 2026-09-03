// MXFP4 vs NVFP4 weight-dequant microbenchmark for SM90.
//
// Measures the fused weight->FP8 bridge that the SM90 MegaMoE kernels run in
// their mainloop prologue: stream packed FP4 weight bytes plus their block
// scales from global memory, LUT-dequant to FP8, and consume the result.
//
// Both sides run the path the kernel actually ships: NVFP4 through its 128-row
// constant LUT, MXFP4 through the register-folded scale (no table, no load).
// Two effects are in play and they are not separable here -- MXFP4's group_size
// 32 halves the scale bytes per 128-element K tile (4 vs NVFP4's 8), and its
// E8M0 scale folds into the FP8 exponent instead of being looked up. Scale
// traffic dominates; folding is worth well under a point on its own.
//
// This is a bridge microbenchmark, not a GEMM or an end-to-end serving result.
// It deliberately does not claim a MegaMoE speedup.
//
// Build: nvcc -O3 -std=c++20 -gencode=arch=compute_90a,code=sm_90a \
//          -I deep_gemm/include tests/bench_fp4_dequant_sm90.cu -o /tmp/bench
#include <cstdio>
#include <cstdint>
#include <vector>
#include <cuda_runtime.h>

#include <deep_gemm/quantization/mxfp4_fused_scale.cuh>
#include <deep_gemm/quantization/nvfp4_dequant.cuh>

#define CHECK(expr) do { \
    cudaError_t err_ = (expr); \
    if (err_ != cudaSuccess) { \
        printf("CUDA error %s at %s:%d\n", cudaGetErrorString(err_), __FILE__, __LINE__); \
        return 1; \
    } \
} while (0)

// One BK128 tile row: 64 packed FP4 bytes (128 elements) plus its scales.
constexpr int kPackedBytesPerTile = 64;
constexpr int kNvfp4ScalesPerTile = 128 / 16;  // 8
constexpr int kMxfp4ScalesPerTile = 128 / 32;  // 4

template <bool kIsMxfp4>
__global__ __launch_bounds__(256) void dequant_stream(
        const uint32_t* __restrict__ packed,
        const uint8_t* __restrict__ scales,
        uint32_t* __restrict__ sink,
        const int num_tiles) {
    constexpr int kScalesPerTile = kIsMxfp4 ? kMxfp4ScalesPerTile : kNvfp4ScalesPerTile;
    // 64 packed bytes = 16 uint32 words per tile.
    constexpr int kWordsPerTile = kPackedBytesPerTile / 4;

    uint32_t acc_hi = 0, acc_lo = 0;
    deep_gemm::mxfp4::ScaledLut mx_lut{};
    for (int tile = blockIdx.x * blockDim.x + threadIdx.x;
         tile < num_tiles;
         tile += gridDim.x * blockDim.x) {
        const uint32_t* tile_words = packed + static_cast<size_t>(tile) * kWordsPerTile;
        const uint8_t* tile_scales = scales + static_cast<size_t>(tile) * kScalesPerTile;
#pragma unroll
        for (int w = 0; w < kWordsPerTile; ++w) {
            // Each scale governs 16 (NVFP4) or 32 (MXFP4) elements; a uint32
            // word holds 8 elements, so the scale index advances every 2 or 4
            // words respectively.
            const int scale_idx = kIsMxfp4 ? (w / 4) : (w / 2);
            const uint8_t sf = tile_scales[scale_idx];
            uint2 out;
            if constexpr (kIsMxfp4) {
                // Built once per 32-element group, as dequant_group32 does --
                // not once per word, which would quadruple the fold work and
                // understate the format.
                if (w % 4 == 0) mx_lut = deep_gemm::mxfp4::make_scaled_lut(sf);
                out = deep_gemm::mxfp4::dequant_word(tile_words[w], mx_lut);
            } else {
                out = deep_gemm::nvfp4::dequant_nvfp4_to_fp8_pair(tile_words[w], sf);
            }
            acc_hi ^= out.x;
            acc_lo ^= out.y;
        }
    }
    // Keep the work live without adding a store per element.
    if (acc_hi == 0xdeadbeefu && acc_lo == 0xfeedfaceu)
        sink[0] = acc_hi + acc_lo;
}

template <bool kIsMxfp4>
static float time_kernel(const uint32_t* packed, const uint8_t* scales,
                         uint32_t* sink, int num_tiles, int iters) {
    constexpr int kBlock = 256;
    const int grid = 1024;
    // Warm up (JIT-free, but still pays first-launch and cache effects).
    for (int i = 0; i < 20; ++i)
        dequant_stream<kIsMxfp4><<<grid, kBlock>>>(packed, scales, sink, num_tiles);
    cudaDeviceSynchronize();

    cudaEvent_t start, stop;
    cudaEventCreate(&start); cudaEventCreate(&stop);
    cudaEventRecord(start);
    for (int i = 0; i < iters; ++i)
        dequant_stream<kIsMxfp4><<<grid, kBlock>>>(packed, scales, sink, num_tiles);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float ms = 0.f;
    cudaEventElapsedTime(&ms, start, stop);
    cudaEventDestroy(start); cudaEventDestroy(stop);
    return ms / iters;
}

int main(int argc, char** argv) {
    // Default sizes the DeepSeek-V4-Flash routed experts actually produce:
    // hidden 6144, intermediate 2048 -> L1 weight is (2*2048) x 6144 per expert.
    int num_tiles = (argc > 1) ? atoi(argv[1]) : (2 * 2048) * (6144 / 128);
    const int iters = (argc > 2) ? atoi(argv[2]) : 200;

    cudaDeviceProp prop{};
    CHECK(cudaGetDeviceProperties(&prop, 0));
    printf("device: %s (sm_%d%d), tiles=%d, iters=%d\n\n",
           prop.name, prop.major, prop.minor, num_tiles, iters);

    const size_t packed_words = static_cast<size_t>(num_tiles) * (kPackedBytesPerTile / 4);
    const size_t nv_scale_bytes = static_cast<size_t>(num_tiles) * kNvfp4ScalesPerTile;
    const size_t mx_scale_bytes = static_cast<size_t>(num_tiles) * kMxfp4ScalesPerTile;

    std::vector<uint32_t> h_packed(packed_words);
    for (size_t i = 0; i < packed_words; ++i)
        h_packed[i] = static_cast<uint32_t>(i * 2654435761u);
    // Scale codes must VARY. A constant code lets NVFP4's table lookup hit the
    // same constant-cache row every time, which hides exactly the cost this
    // benchmark exists to measure. Codes span 120..134: E8M0 2^-7..2^7 (inside
    // MXFP4's register fast path) and mid-range UE4M3, so neither side is
    // measuring saturation or an underflow flush.
    std::vector<uint8_t> h_scales(nv_scale_bytes);
    for (size_t i = 0; i < nv_scale_bytes; ++i)
        h_scales[i] = static_cast<uint8_t>(120 + (i * 2654435761u >> 13) % 15);

    uint32_t *d_packed = nullptr, *d_sink = nullptr;
    uint8_t *d_nv_scales = nullptr, *d_mx_scales = nullptr;
    CHECK(cudaMalloc(&d_packed, packed_words * sizeof(uint32_t)));
    CHECK(cudaMalloc(&d_nv_scales, nv_scale_bytes));
    CHECK(cudaMalloc(&d_mx_scales, mx_scale_bytes));
    CHECK(cudaMalloc(&d_sink, sizeof(uint32_t)));
    CHECK(cudaMemcpy(d_packed, h_packed.data(), packed_words * sizeof(uint32_t), cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(d_nv_scales, h_scales.data(), nv_scale_bytes, cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(d_mx_scales, h_scales.data(), mx_scale_bytes, cudaMemcpyHostToDevice));

    const float nv_ms = time_kernel<false>(d_packed, d_nv_scales, d_sink, num_tiles, iters);
    const float mx_ms = time_kernel<true>(d_packed, d_mx_scales, d_sink, num_tiles, iters);

    const double packed_bytes = static_cast<double>(num_tiles) * kPackedBytesPerTile;
    const double nv_bytes = packed_bytes + static_cast<double>(nv_scale_bytes);
    const double mx_bytes = packed_bytes + static_cast<double>(mx_scale_bytes);

    printf("%-8s %10s %12s %12s %12s\n", "format", "ms", "bytes read", "GB/s", "Gelem/s");
    printf("%-8s %10.4f %12.0f %12.1f %12.1f\n", "NVFP4", nv_ms, nv_bytes,
           nv_bytes / (nv_ms * 1e-3) / 1e9, num_tiles * 128.0 / (nv_ms * 1e-3) / 1e9);
    printf("%-8s %10.4f %12.0f %12.1f %12.1f\n", "MXFP4", mx_ms, mx_bytes,
           mx_bytes / (mx_ms * 1e-3) / 1e9, num_tiles * 128.0 / (mx_ms * 1e-3) / 1e9);
    printf("\nMXFP4 vs NVFP4: %+.2f%% time, %.2f%% of the scale bytes\n",
           (mx_ms - nv_ms) / nv_ms * 100.0,
           static_cast<double>(mx_scale_bytes) / nv_scale_bytes * 100.0);

    cudaFree(d_packed); cudaFree(d_nv_scales); cudaFree(d_mx_scales); cudaFree(d_sink);
    return 0;
}
