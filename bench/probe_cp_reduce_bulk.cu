#include <cuda_runtime.h>

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <vector>

#define CUDA_CHECK(expr)                                                     \
    do {                                                                     \
        const cudaError_t error = (expr);                                    \
        if (error != cudaSuccess) {                                          \
            std::fprintf(stderr, "%s:%d: %s\n", __FILE__, __LINE__,        \
                         cudaGetErrorString(error));                         \
            std::exit(1);                                                    \
        }                                                                    \
    } while (0)

__device__ __forceinline__ void bulk_reduce_add_f32(
        float* dst, const float* src, uint32_t bytes) {
    const uint32_t src_smem =
        static_cast<uint32_t>(__cvta_generic_to_shared(src));
    asm volatile(
        "cp.reduce.async.bulk.global.shared::cta.bulk_group.add.f32 "
        "[%0], [%1], %2;"
        : : "l"(dst), "r"(src_smem), "r"(bytes) : "memory");
}

__global__ void probe_kernel(
        float* dst, int operations, int destination_groups) {
    constexpr int kValues = 128;
    constexpr int kPitch = 132;
    constexpr int kMaxOperations = 8;
    __shared__ __align__(16) float staging[kMaxOperations * kPitch];

    for (int index = threadIdx.x;
         index < operations * kValues; index += blockDim.x) {
        const int operation = index / kValues;
        const int column = index - operation * kValues;
        staging[operation * kPitch + column] = 1.0f;
    }
    __syncthreads();

    if (threadIdx.x == 0) {
        // cudaMemset plus a host synchronization established the generic
        // zero initialization.  Import it into the async proxy before the
        // shared-to-global reduction, and publish the CTA's generic shared
        // stores into that same proxy.
        asm volatile("fence.proxy.async.global;" ::: "memory");
        asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
        const int destination_group = blockIdx.x % destination_groups;
        #pragma unroll
        for (int operation = 0; operation < kMaxOperations; ++operation) {
            if (operation < operations) {
                bulk_reduce_add_f32(
                    dst + (destination_group * operations + operation)
                        * kValues,
                    staging + operation * kPitch,
                    kValues * sizeof(float));
            }
        }
        asm volatile("cp.async.bulk.commit_group;" ::: "memory");
        asm volatile("cp.async.bulk.wait_group 0;" ::: "memory");
    }
}

static bool run_case(int blocks, int operations, int destination_groups) {
    const size_t values = static_cast<size_t>(destination_groups)
        * operations * 128;
    float* dst = nullptr;
    CUDA_CHECK(cudaMalloc(&dst, values * sizeof(float)));
    CUDA_CHECK(cudaMemset(dst, 0, values * sizeof(float)));
    CUDA_CHECK(cudaDeviceSynchronize());

    std::printf("PROBE_START blocks=%d operations=%d destinations=%d\n",
                blocks, operations, destination_groups);
    std::fflush(stdout);
    probe_kernel<<<blocks, 128>>>(dst, operations, destination_groups);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    std::vector<float> host(values);
    CUDA_CHECK(cudaMemcpy(host.data(), dst, values * sizeof(float),
                          cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaFree(dst));
    const float expected = static_cast<float>(blocks / destination_groups);
    float max_error = 0.0f;
    for (const float value : host)
        max_error = std::fmax(max_error, std::fabs(value - expected));
    std::printf(
        "PROBE_RESULT blocks=%d operations=%d destinations=%d "
        "expected=%.1f max_error=%g pass=%s\n",
        blocks, operations, destination_groups, expected, max_error,
        max_error == 0.0f ? "true" : "false");
    std::fflush(stdout);
    return max_error == 0.0f;
}

int main() {
    int device = 0;
    cudaDeviceProp properties{};
    CUDA_CHECK(cudaGetDevice(&device));
    CUDA_CHECK(cudaGetDeviceProperties(&properties, device));
    std::printf("PROBE_ENV gpu=%s sm=%d.%d sms=%d\n",
                properties.name, properties.major, properties.minor,
                properties.multiProcessorCount);
    std::fflush(stdout);

    bool ok = true;
    ok &= run_case(1, 1, 1);
    ok &= run_case(1, 8, 1);
    ok &= run_case(624, 8, 624);
    ok &= run_case(624, 8, 104);
    return ok ? 0 : 2;
}
