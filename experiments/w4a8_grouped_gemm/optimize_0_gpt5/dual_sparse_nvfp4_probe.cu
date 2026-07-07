#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <vector>

#include <cuda_fp16.h>
#include <cuda_fp8.h>
#include <cuda_runtime.h>

#include <deep_gemm/mma/sm90.cuh>
#include <deep_gemm/ptx/wgmma.cuh>
#include <deep_gemm/quantization/nvfp4_dequant.cuh>

namespace {

constexpr int kM = 64;
constexpr int kN = 128;
constexpr int kK = 64;
constexpr int kPackedK = kK / 2;
constexpr int kScaleK = kK / 16;

struct GroupDecomposition {
    uint8_t q0[4];
    uint8_t q1[4];
    float original[4];
    bool use_q1[4];
};

__device__ __forceinline__ float decode_fp8(uint8_t value) {
    const __half decoded = __nv_cvt_fp8_to_halfraw(value, __NV_E4M3);
    return __half2float(decoded);
}

__device__ __forceinline__ float decode_nvfp4(
        uint8_t code, uint8_t scale_code) {
    const uint32_t magnitude = code & 7u;
    const __half magnitude_h = __ushort_as_half(
        deep_gemm::nvfp4::e2m1_mag_to_fp16_bits(magnitude));
    const __half scale_h = __nv_cvt_fp8_to_halfraw(
        scale_code & 0x7fu, __NV_E4M3);
    float value = __half2float(magnitude_h) * __half2float(scale_h) * 0.125f;
    return (code & 8u) != 0 ? -value : value;
}

__device__ __forceinline__ uint8_t encode_fp8(float value) {
    return static_cast<uint8_t>(__nv_cvt_float_to_fp8(
        value, __NV_SATFINITE, __NV_E4M3));
}

__device__ __forceinline__ GroupDecomposition decompose_group(
        const uint8_t codes[4], uint8_t scale_code) {
    GroupDecomposition result{};
    float improvement[4];
#pragma unroll
    for (int i = 0; i < 4; ++i) {
        const float original = decode_nvfp4(codes[i], scale_code);
        result.original[i] = original;
        result.q0[i] = encode_fp8(original);
        result.q1[i] = encode_fp8(original / 3.0f);
        const float q0_value = decode_fp8(result.q0[i]);
        const float q1_value = 3.0f * decode_fp8(result.q1[i]);
        const float q0_error = original - q0_value;
        const float q1_error = original - q1_value;
        improvement[i] = q0_error * q0_error - q1_error * q1_error;
    }

    int first = 0;
#pragma unroll
    for (int i = 1; i < 4; ++i)
        if (improvement[i] > improvement[first])
            first = i;
    int second = first == 0 ? 1 : 0;
#pragma unroll
    for (int i = 0; i < 4; ++i)
        if (i != first && improvement[i] > improvement[second])
            second = i;
    result.use_q1[first] = true;
    result.use_q1[second] = true;
    return result;
}

__device__ __forceinline__ uint32_t metadata_nibble(int first, int second) {
    return static_cast<uint32_t>(first | (second << 2));
}

__global__ __launch_bounds__(128, 1) void dual_sparse_nvfp4(
        const uint8_t* packed_weight,
        const uint8_t* block_scale,
        const uint8_t* activation,
        float* output) {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900)
    using WGMMA = typename deep_gemm::mma::sm90::SparseFP8MMASelector<kN>::type;

    __shared__ __align__(1024) uint8_t sparse_q0[kM * (kK / 2)];
    __shared__ __align__(1024) uint8_t sparse_q1[kM * (kK / 2)];
    __shared__ __align__(1024) uint8_t dense_b[kN * kK];
    __shared__ uint32_t metadata_q0[kM * 2];
    __shared__ uint32_t metadata_q1[kM * 2];

    const int tid = threadIdx.x;
    for (int item = tid; item < kN * kK; item += blockDim.x) {
        const int row = item / kK;
        const int k = item % kK;
        const int physical_k = k ^ (((row >> 1) & 3) << 4);
        dense_b[row * kK + physical_k] = activation[item];
    }

    if (tid < kM) {
        uint32_t row_metadata_q0[2] = {0u, 0u};
        uint32_t row_metadata_q1[2] = {0u, 0u};
#pragma unroll
        for (int group = 0; group < kK / 4; ++group) {
            const int k_begin = group * 4;
            const uint8_t packed0 = packed_weight[tid * kPackedK + k_begin / 2];
            const uint8_t packed1 = packed_weight[tid * kPackedK + k_begin / 2 + 1];
            const uint8_t codes[4] = {
                static_cast<uint8_t>(packed0 & 0x0f),
                static_cast<uint8_t>(packed0 >> 4),
                static_cast<uint8_t>(packed1 & 0x0f),
                static_cast<uint8_t>(packed1 >> 4),
            };
            const auto decomposition = decompose_group(
                codes, block_scale[tid * kScaleK + k_begin / 16]);

            int q0_indices[2];
            int q1_indices[2];
            int q0_count = 0;
            int q1_count = 0;
#pragma unroll
            for (int i = 0; i < 4; ++i) {
                if (decomposition.use_q1[i])
                    q1_indices[q1_count++] = i;
                else
                    q0_indices[q0_count++] = i;
            }

            const int physical_k = group * 2;
            const int swizzle = ((tid >> 2) & 1) << 4;
            sparse_q0[tid * (kK / 2) + (physical_k ^ swizzle)] =
                decomposition.q0[q0_indices[0]];
            sparse_q0[tid * (kK / 2) + ((physical_k + 1) ^ swizzle)] =
                decomposition.q0[q0_indices[1]];
            sparse_q1[tid * (kK / 2) + (physical_k ^ swizzle)] =
                decomposition.q1[q1_indices[0]];
            sparse_q1[tid * (kK / 2) + ((physical_k + 1) ^ swizzle)] =
                decomposition.q1[q1_indices[1]];

            const int half = group / 8;
            const int shift = 4 * (group & 7);
            row_metadata_q0[half] |=
                metadata_nibble(q0_indices[0], q0_indices[1]) << shift;
            row_metadata_q1[half] |=
                metadata_nibble(q1_indices[0], q1_indices[1]) << shift;
        }
        metadata_q0[tid * 2] = row_metadata_q0[0];
        metadata_q0[tid * 2 + 1] = row_metadata_q0[1];
        metadata_q1[tid * 2] = row_metadata_q1[0];
        metadata_q1[tid * 2 + 1] = row_metadata_q1[1];
    }
    __syncthreads();

    const int metadata_row =
        (tid & 1) * 8 + ((tid >> 2) & 7) + ((tid >> 5) & 3) * 16;
    const int metadata_half = (tid >> 1) & 1;
    const uint32_t e0 = metadata_q0[metadata_row * 2 + metadata_half];
    const uint32_t e1 = metadata_q1[metadata_row * 2 + metadata_half];

    auto q0_desc = deep_gemm::mma::sm90::make_smem_desc(
        sparse_q0, static_cast<int>(cute::SM90::GMMA::LayoutType::B32), 0, 256);
    auto q1_desc = deep_gemm::mma::sm90::make_smem_desc(
        sparse_q1, static_cast<int>(cute::SM90::GMMA::LayoutType::B32), 0, 256);
    auto b_desc = deep_gemm::mma::sm90::make_smem_desc(
        dense_b, static_cast<int>(cute::SM90::GMMA::LayoutType::B64), 0, 512);

    float accum_q0[WGMMA::kNumAccum] = {0.0f};
    float accum_q1[WGMMA::kNumAccum] = {0.0f};
#pragma unroll
    for (int i = 0; i < WGMMA::kNumAccum; ++i) {
        deep_gemm::ptx::warpgroup_fence_operand(accum_q0[i]);
        deep_gemm::ptx::warpgroup_fence_operand(accum_q1[i]);
    }
    deep_gemm::ptx::warpgroup_arrive();
    WGMMA::wgmma(q0_desc, b_desc, accum_q0, e0, false);
    WGMMA::wgmma(q1_desc, b_desc, accum_q1, e1, false);
    deep_gemm::ptx::warpgroup_commit_batch();
    deep_gemm::ptx::warpgroup_wait<0>();
#pragma unroll
    for (int i = 0; i < WGMMA::kNumAccum; ++i) {
        deep_gemm::ptx::warpgroup_fence_operand(accum_q0[i]);
        deep_gemm::ptx::warpgroup_fence_operand(accum_q1[i]);
    }

    const int lane = tid & 31;
    const int warp = tid >> 5;
    const int row0 = warp * 16 + lane / 4;
    const int row1 = row0 + 8;
#pragma unroll
    for (int group = 0; group < kN / 16; ++group) {
        const int col0 = group * 16 + 2 * (tid & 3);
        const int col1 = col0 + 8;
        const int base = group * 8;
#define STORE_COMBINED(ROW, COL, INDEX) \
        output[(ROW) * kN + (COL)] = accum_q0[(INDEX)] + 3.0f * accum_q1[(INDEX)]
        STORE_COMBINED(row0, col0, base + 0);
        STORE_COMBINED(row0, col0 + 1, base + 1);
        STORE_COMBINED(row1, col0, base + 2);
        STORE_COMBINED(row1, col0 + 1, base + 3);
        STORE_COMBINED(row0, col1, base + 4);
        STORE_COMBINED(row0, col1 + 1, base + 5);
        STORE_COMBINED(row1, col1, base + 6);
        STORE_COMBINED(row1, col1 + 1, base + 7);
#undef STORE_COMBINED
    }
#endif
}

__global__ void scalar_reference(
        const uint8_t* packed_weight,
        const uint8_t* block_scale,
        const uint8_t* activation,
        float* reconstructed,
        float* original) {
    const int item = blockIdx.x * blockDim.x + threadIdx.x;
    if (item >= kM * kN)
        return;
    const int row = item / kN;
    const int col = item % kN;
    float reconstructed_accum = 0.0f;
    float original_accum = 0.0f;
#pragma unroll
    for (int group = 0; group < kK / 4; ++group) {
        const int k_begin = group * 4;
        const uint8_t packed0 = packed_weight[row * kPackedK + k_begin / 2];
        const uint8_t packed1 = packed_weight[row * kPackedK + k_begin / 2 + 1];
        const uint8_t codes[4] = {
            static_cast<uint8_t>(packed0 & 0x0f),
            static_cast<uint8_t>(packed0 >> 4),
            static_cast<uint8_t>(packed1 & 0x0f),
            static_cast<uint8_t>(packed1 >> 4),
        };
        const auto decomposition = decompose_group(
            codes, block_scale[row * kScaleK + k_begin / 16]);
#pragma unroll
        for (int i = 0; i < 4; ++i) {
            const float b = decode_fp8(activation[col * kK + k_begin + i]);
            const float approximation = decomposition.use_q1[i]
                ? 3.0f * decode_fp8(decomposition.q1[i])
                : decode_fp8(decomposition.q0[i]);
            reconstructed_accum += approximation * b;
            original_accum += decomposition.original[i] * b;
        }
    }
    reconstructed[item] = reconstructed_accum;
    original[item] = original_accum;
}

void check_cuda(cudaError_t status, const char* operation) {
    if (status != cudaSuccess) {
        std::fprintf(stderr, "%s: %s\n", operation, cudaGetErrorString(status));
        std::exit(2);
    }
}

uint8_t activation_fp8(int selector) {
    constexpr uint8_t values[] = {0x30, 0x38, 0x40, 0x48};
    uint8_t value = values[selector & 3];
    if ((selector >> 2) & 1)
        value |= 0x80;
    return value;
}

}  // namespace

int main() {
    std::vector<uint8_t> packed(kM * kPackedK);
    std::vector<uint8_t> scales(kM * kScaleK);
    std::vector<uint8_t> activation(kN * kK);
    for (int row = 0; row < kM; ++row) {
        for (int k = 0; k < kK; k += 2) {
            const uint8_t code0 = static_cast<uint8_t>((row * 13 + k * 7 + k / 4) & 15);
            const uint8_t code1 = static_cast<uint8_t>((row * 3 + (k + 1) * 11 + k / 8) & 15);
            packed[row * kPackedK + k / 2] = code0 | (code1 << 4);
        }
        for (int block = 0; block < kScaleK; ++block)
            scales[row * kScaleK + block] =
                static_cast<uint8_t>(0x38 + 8 * ((row + block * 3) % 6));
    }
    for (int row = 0; row < kN; ++row)
        for (int k = 0; k < kK; ++k)
            activation[row * kK + k] = activation_fp8(row * 5 + k * 3 + k / 7);

    uint8_t* packed_device = nullptr;
    uint8_t* scales_device = nullptr;
    uint8_t* activation_device = nullptr;
    float* sparse_output_device = nullptr;
    float* reconstructed_device = nullptr;
    float* original_device = nullptr;
    check_cuda(cudaMalloc(&packed_device, packed.size()), "cudaMalloc packed");
    check_cuda(cudaMalloc(&scales_device, scales.size()), "cudaMalloc scales");
    check_cuda(cudaMalloc(&activation_device, activation.size()), "cudaMalloc activation");
    check_cuda(cudaMalloc(&sparse_output_device, kM * kN * sizeof(float)), "cudaMalloc sparse output");
    check_cuda(cudaMalloc(&reconstructed_device, kM * kN * sizeof(float)), "cudaMalloc reconstructed");
    check_cuda(cudaMalloc(&original_device, kM * kN * sizeof(float)), "cudaMalloc original");
    check_cuda(cudaMemcpy(packed_device, packed.data(), packed.size(), cudaMemcpyHostToDevice),
               "cudaMemcpy packed");
    check_cuda(cudaMemcpy(scales_device, scales.data(), scales.size(), cudaMemcpyHostToDevice),
               "cudaMemcpy scales");
    check_cuda(cudaMemcpy(activation_device, activation.data(), activation.size(), cudaMemcpyHostToDevice),
               "cudaMemcpy activation");

    dual_sparse_nvfp4<<<1, 128>>>(
        packed_device, scales_device, activation_device, sparse_output_device);
    scalar_reference<<<(kM * kN + 255) / 256, 256>>>(
        packed_device, scales_device, activation_device,
        reconstructed_device, original_device);
    check_cuda(cudaGetLastError(), "kernel launch");
    check_cuda(cudaDeviceSynchronize(), "kernel execution");

    std::vector<float> sparse_output(kM * kN);
    std::vector<float> reconstructed(kM * kN);
    std::vector<float> original(kM * kN);
    check_cuda(cudaMemcpy(sparse_output.data(), sparse_output_device,
                          sparse_output.size() * sizeof(float), cudaMemcpyDeviceToHost),
               "cudaMemcpy sparse output");
    check_cuda(cudaMemcpy(reconstructed.data(), reconstructed_device,
                          reconstructed.size() * sizeof(float), cudaMemcpyDeviceToHost),
               "cudaMemcpy reconstructed");
    check_cuda(cudaMemcpy(original.data(), original_device,
                          original.size() * sizeof(float), cudaMemcpyDeviceToHost),
               "cudaMemcpy original");

    int mismatches = 0;
    float max_contract_error = 0.0f;
    float max_approx_error = 0.0f;
    for (int i = 0; i < kM * kN; ++i) {
        const float contract_error = std::fabs(sparse_output[i] - reconstructed[i]);
        const float approx_error = std::fabs(sparse_output[i] - original[i]);
        mismatches += contract_error != 0.0f;
        max_contract_error = std::fmax(max_contract_error, contract_error);
        max_approx_error = std::fmax(max_approx_error, approx_error);
    }
    std::printf(
        "dual_sparse_nvfp4_probe mismatches=%d max_contract_error=%.9g "
        "max_approx_error=%.9g packed_weight_bytes=%zu scale_bytes=%zu\n",
        mismatches, max_contract_error, max_approx_error,
        packed.size(), scales.size());

    cudaFree(packed_device);
    cudaFree(scales_device);
    cudaFree(activation_device);
    cudaFree(sparse_output_device);
    cudaFree(reconstructed_device);
    cudaFree(original_device);
    return mismatches == 0 ? 0 : 1;
}
