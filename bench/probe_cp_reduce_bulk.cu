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

// Match the MegaMoE integration more closely than probe_kernel: the bulk
// destination row is loaded from ordinary global memory after all but lane 0
// have branched away.  This forces ptxas to transfer the dynamically loaded
// address from a regular register into the uniform-register operands used by
// UBLKRED, exactly like a route-derived token destination in the W2 epilogue.
__global__ void indexed_probe_kernel(
        float* dst, const int* destination_rows, int operations) {
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
        asm volatile("fence.proxy.async.global;" ::: "memory");
        asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
        #pragma unroll
        for (int operation = 0; operation < kMaxOperations; ++operation) {
            if (operation < operations) {
                const int destination_row =
                    destination_rows[blockIdx.x * operations + operation];
                bulk_reduce_add_f32(
                    dst + static_cast<size_t>(destination_row) * kValues,
                    staging + operation * kPitch,
                    kValues * sizeof(float));
            }
        }
        asm volatile("cp.async.bulk.commit_group;" ::: "memory");
        asm volatile("cp.async.bulk.wait_group 0;" ::: "memory");
    }
}

__device__ __forceinline__ void one_use_grid_barrier(
        unsigned int* state, int blocks) {
    __syncthreads();
    if (threadIdx.x == 0) {
        __threadfence();
        const unsigned int ticket = atomicAdd(state, 1u);
        if (ticket + 1u == static_cast<unsigned int>(blocks)) {
            atomicExch(state, 0u);
            __threadfence();
            atomicExch(state + 1, 1u);
        } else {
            while (atomicAdd(state + 1, 0u) == 0u)
                __nanosleep(64);
        }
    }
    __syncthreads();
}

// Also reproduce the integrated lifetime of the reduction destination.  The
// same persistent grid first clears it with generic stores, publishes those
// stores through an ordinary software whole-grid barrier, and only then
// issues the async-proxy reductions.
__global__ void in_kernel_zero_probe_kernel(
        float* dst, const int* destination_rows, unsigned int* barrier_state,
        int destination_count, int operations) {
    constexpr int kValues = 128;
    constexpr int kPitch = 132;
    constexpr int kMaxOperations = 8;
    __shared__ __align__(16) float staging[kMaxOperations * kPitch];

    const int global_thread = blockIdx.x * blockDim.x + threadIdx.x;
    const int global_threads = gridDim.x * blockDim.x;
    for (int index = global_thread;
         index < destination_count * kValues; index += global_threads)
        dst[index] = 0.0f;
    // Proxy fences are thread-scoped, so match the strongest integrated
    // diagnostic and publish each writer lane before the grid rendezvous.
    asm volatile("fence.proxy.async.global;" ::: "memory");

    for (int index = threadIdx.x;
         index < operations * kValues; index += blockDim.x) {
        const int operation = index / kValues;
        const int column = index - operation * kValues;
        staging[operation * kPitch + column] = 1.0f;
    }
    one_use_grid_barrier(barrier_state, gridDim.x);

    if (threadIdx.x == 0) {
        asm volatile("fence.proxy.async.global;" ::: "memory");
        asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
        #pragma unroll
        for (int operation = 0; operation < kMaxOperations; ++operation) {
            if (operation < operations) {
                const int destination_row =
                    destination_rows[blockIdx.x * operations + operation];
                bulk_reduce_add_f32(
                    dst + static_cast<size_t>(destination_row) * kValues,
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

static bool run_indexed_case(
        int blocks, int operations, int destination_rows, bool crossing_order,
        bool in_kernel_zero = false) {
    constexpr int kValues = 128;
    std::vector<int> host_rows(static_cast<size_t>(blocks) * operations);
    std::vector<int> expected_counts(destination_rows, 0);
    for (int block = 0; block < blocks; ++block) {
        const int group = crossing_order
            ? block % destination_rows
            : block % (destination_rows / operations);
        for (int operation = 0; operation < operations; ++operation) {
            const int ordered_operation =
                crossing_order && (block & 1)
                ? operations - 1 - operation : operation;
            const int destination_row = crossing_order
                ? (group + ordered_operation) % destination_rows
                : group * operations + ordered_operation;
            host_rows[static_cast<size_t>(block) * operations + operation] =
                destination_row;
            ++expected_counts[destination_row];
        }
    }

    float* dst = nullptr;
    int* device_rows = nullptr;
    unsigned int* barrier_state = nullptr;
    CUDA_CHECK(cudaMalloc(&dst,
                          static_cast<size_t>(destination_rows) * kValues
                              * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&device_rows, host_rows.size() * sizeof(int)));
    if (!in_kernel_zero) {
        CUDA_CHECK(cudaMemset(
            dst, 0,
            static_cast<size_t>(destination_rows) * kValues * sizeof(float)));
    } else {
        CUDA_CHECK(cudaMalloc(&barrier_state, 2 * sizeof(unsigned int)));
        CUDA_CHECK(cudaMemset(barrier_state, 0, 2 * sizeof(unsigned int)));
    }
    CUDA_CHECK(cudaMemcpy(device_rows, host_rows.data(),
                          host_rows.size() * sizeof(int),
                          cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaDeviceSynchronize());

    std::printf(
        "INDEXED_PROBE_START blocks=%d operations=%d destinations=%d "
        "crossing=%s in_kernel_zero=%s\n",
        blocks, operations, destination_rows,
        crossing_order ? "true" : "false",
        in_kernel_zero ? "true" : "false");
    std::fflush(stdout);
    if (in_kernel_zero) {
        in_kernel_zero_probe_kernel<<<blocks, 128>>>(
            dst, device_rows, barrier_state, destination_rows, operations);
    } else {
        indexed_probe_kernel<<<blocks, 128>>>(dst, device_rows, operations);
    }
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    std::vector<float> host(
        static_cast<size_t>(destination_rows) * kValues);
    CUDA_CHECK(cudaMemcpy(host.data(), dst, host.size() * sizeof(float),
                          cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaFree(device_rows));
    if (barrier_state != nullptr)
        CUDA_CHECK(cudaFree(barrier_state));
    CUDA_CHECK(cudaFree(dst));
    float max_error = 0.0f;
    for (int row = 0; row < destination_rows; ++row) {
        const float expected = static_cast<float>(expected_counts[row]);
        for (int column = 0; column < kValues; ++column) {
            max_error = std::fmax(
                max_error,
                std::fabs(host[static_cast<size_t>(row) * kValues + column]
                          - expected));
        }
    }
    std::printf(
        "INDEXED_PROBE_RESULT blocks=%d operations=%d destinations=%d "
        "crossing=%s in_kernel_zero=%s max_error=%g pass=%s\n",
        blocks, operations, destination_rows,
        crossing_order ? "true" : "false",
        in_kernel_zero ? "true" : "false", max_error,
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
    ok &= run_indexed_case(624, 1, 624, false);
    ok &= run_indexed_case(624, 1, 104, false);
    ok &= run_indexed_case(624, 8, 832, false);
    ok &= run_indexed_case(624, 8, 104, true);
    ok &= run_indexed_case(624, 8, 104, true, true);
    return ok ? 0 : 2;
}
