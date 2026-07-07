#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <vector>

#include <cuda_runtime.h>

#include <deep_gemm/mma/sm90.cuh>
#include <deep_gemm/ptx/wgmma.cuh>

namespace {

constexpr int kM = 64;
constexpr int kN = 128;
constexpr int kK = 64;

__host__ __device__ constexpr uint8_t fp8_value(int selector) {
    // Exact E4M3 encodings: 0.5, 1, 2, 4.
    constexpr uint8_t values[] = {0x30, 0x38, 0x40, 0x48};
    return values[selector & 3];
}

__host__ __device__ constexpr void selected_pair(
        int row, int group, int& first, int& second) {
    constexpr int pairs[6][2] = {
        {0, 1}, {0, 2}, {0, 3}, {1, 2}, {1, 3}, {2, 3},
    };
    const int pattern = (row * 5 + group * 3 + group / 2) % 6;
    first = pairs[pattern][0];
    second = pairs[pattern][1];
}

__host__ __device__ constexpr uint32_t metadata_nibble(
        int first, int second) {
    return static_cast<uint32_t>(first | (second << 2));
}

__global__ __launch_bounds__(128, 1) void sparse_wgmma_probe(float* output) {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900)
    using WGMMA = typename deep_gemm::mma::sm90::SparseFP8MMASelector<kN>::type;

    __shared__ __align__(1024) uint8_t sparse_a[kM * (kK / 2)];
    __shared__ __align__(1024) uint8_t dense_b[kN * kK];

    const int tid = threadIdx.x;

    // A is sparse and uses a physical K=32 B32 swizzle. Each logical group
    // contributes exactly two compressed FP8 bytes in metadata order.
    for (int item = tid; item < kM * (kK / 4); item += blockDim.x) {
        const int row = item / (kK / 4);
        const int group = item % (kK / 4);
        int first, second;
        selected_pair(row, group, first, second);
        const int physical_k = 2 * group;
        // Swizzle<1,4,3> XORs address bit 7 into bit 4. With a 32-byte
        // physical row, bit 7 is row bit 2 (not row bit 0).
        const int swizzle = ((row >> 2) & 1) << 4;
        sparse_a[row * (kK / 2) + (physical_k ^ swizzle)] =
            fp8_value(row + group + first);
        sparse_a[row * (kK / 2) + ((physical_k + 1) ^ swizzle)] =
            fp8_value(row + group + second + 1);
    }

    // B is dense and uses a physical K=64 B64 swizzle. Vary both K and N so
    // incorrect metadata or output-fragment mappings cannot accidentally pass.
    for (int item = tid; item < kN * kK; item += blockDim.x) {
        const int row = item / kK;
        const int k = item % kK;
        // Swizzle<2,4,3> XORs address bits [7,8] into [4,5]. With a
        // 64-byte physical row, those are row bits [1,2].
        const int physical_k = k ^ (((row >> 1) & 3) << 4);
        dense_b[row * kK + physical_k] = fp8_value(row * 3 + k * 5 + k / 7);
    }
    __syncthreads();

    // CUTLASS ELayout_64x64 maps one 32-bit metadata register to one matrix
    // row and one K32 half. Build that word directly instead of materializing
    // metadata outside the CTA.
    const int metadata_row =
        (tid & 1) * 8 + ((tid >> 2) & 7) + ((tid >> 5) & 3) * 16;
    const int metadata_k_begin = ((tid >> 1) & 1) * 32;
    uint32_t metadata = 0;
#pragma unroll
    for (int local_group = 0; local_group < 8; ++local_group) {
        const int group = metadata_k_begin / 4 + local_group;
        int first, second;
        selected_pair(metadata_row, group, first, second);
        metadata |= metadata_nibble(first, second) << (4 * local_group);
    }

    auto a_desc = deep_gemm::mma::sm90::make_smem_desc(
        sparse_a, static_cast<int>(cute::SM90::GMMA::LayoutType::B32), 0, 256);
    auto b_desc = deep_gemm::mma::sm90::make_smem_desc(
        dense_b, static_cast<int>(cute::SM90::GMMA::LayoutType::B64), 0, 512);

    float accum[WGMMA::kNumAccum] = {0.0f};
#pragma unroll
    for (int i = 0; i < WGMMA::kNumAccum; ++i)
        deep_gemm::ptx::warpgroup_fence_operand(accum[i]);
    deep_gemm::ptx::warpgroup_arrive();
    WGMMA::wgmma(a_desc, b_desc, accum, metadata, false);
    deep_gemm::ptx::warpgroup_commit_batch();
    deep_gemm::ptx::warpgroup_wait<0>();
#pragma unroll
    for (int i = 0; i < WGMMA::kNumAccum; ++i)
        deep_gemm::ptx::warpgroup_fence_operand(accum[i]);

    const int lane = tid & 31;
    const int warp = tid >> 5;
    const int row0 = warp * 16 + lane / 4;
    const int row1 = row0 + 8;
#pragma unroll
    for (int group = 0; group < kN / 16; ++group) {
        const int col0 = group * 16 + 2 * (tid & 3);
        const int col1 = col0 + 8;
        const int base = group * 8;
        output[row0 * kN + col0] = accum[base + 0];
        output[row0 * kN + col0 + 1] = accum[base + 1];
        output[row1 * kN + col0] = accum[base + 2];
        output[row1 * kN + col0 + 1] = accum[base + 3];
        output[row0 * kN + col1] = accum[base + 4];
        output[row0 * kN + col1 + 1] = accum[base + 5];
        output[row1 * kN + col1] = accum[base + 6];
        output[row1 * kN + col1 + 1] = accum[base + 7];
    }
#endif
}

float fp8_as_float(int selector) {
    constexpr float values[] = {0.5f, 1.0f, 2.0f, 4.0f};
    return values[selector & 3];
}

void check_cuda(cudaError_t status, const char* operation) {
    if (status != cudaSuccess) {
        std::fprintf(stderr, "%s: %s\n", operation, cudaGetErrorString(status));
        std::exit(2);
    }
}

}  // namespace

int main() {
    float* output_device = nullptr;
    check_cuda(cudaMalloc(&output_device, kM * kN * sizeof(float)), "cudaMalloc");
    sparse_wgmma_probe<<<1, 128>>>(output_device);
    check_cuda(cudaGetLastError(), "kernel launch");
    check_cuda(cudaDeviceSynchronize(), "kernel execution");

    std::vector<float> output(kM * kN);
    check_cuda(cudaMemcpy(output.data(), output_device,
                          output.size() * sizeof(float), cudaMemcpyDeviceToHost),
               "cudaMemcpy");
    check_cuda(cudaFree(output_device), "cudaFree");

    float max_abs_error = 0.0f;
    int mismatches = 0;
    int row_mismatches[kM] = {};
    int col_mismatches[kN] = {};
    int samples_printed = 0;
    for (int row = 0; row < kM; ++row) {
        for (int col = 0; col < kN; ++col) {
            float expected = 0.0f;
            for (int group = 0; group < kK / 4; ++group) {
                int first, second;
                selected_pair(row, group, first, second);
                const int k0 = group * 4 + first;
                const int k1 = group * 4 + second;
                const float a0 = fp8_as_float(row + group + first);
                const float a1 = fp8_as_float(row + group + second + 1);
                const float b0 = fp8_as_float(col * 3 + k0 * 5 + k0 / 7);
                const float b1 = fp8_as_float(col * 3 + k1 * 5 + k1 / 7);
                expected += a0 * b0 + a1 * b1;
            }
            const float error = std::fabs(output[row * kN + col] - expected);
            max_abs_error = error > max_abs_error ? error : max_abs_error;
            if (error != 0.0f) {
                ++mismatches;
                ++row_mismatches[row];
                ++col_mismatches[col];
                if (samples_printed < 24) {
                    std::printf("mismatch row=%d col=%d actual=%.9g expected=%.9g error=%.9g\n",
                                row, col, output[row * kN + col], expected, error);
                    ++samples_printed;
                }
            }
        }
    }

    std::printf("sparse_wgmma_probe mismatches=%d max_abs_error=%.9g\n",
                mismatches, max_abs_error);
    std::printf("row mismatches:");
    for (int row = 0; row < kM; ++row)
        std::printf(" %d:%d", row, row_mismatches[row]);
    std::printf("\ncol mismatches:");
    for (int col = 0; col < kN; ++col)
        std::printf(" %d:%d", col, col_mismatches[col]);
    std::printf("\n");
    return mismatches == 0 ? 0 : 1;
}
