#include <cuda_runtime.h>

#include <cstdint>
#include <cstdio>

__global__ void probe_policy(std::uint64_t* output) {
    std::uint64_t policy;
    asm volatile(
        "createpolicy.fractional.L2::evict_first.b64 %0,1.0;"
        : "=l"(policy));
    if (threadIdx.x == 0 && blockIdx.x == 0)
        output[0] = policy;
}

int main() {
    std::uint64_t* device_output = nullptr;
    std::uint64_t host_output = 0;
    if (cudaMalloc(&device_output, sizeof(host_output)) != cudaSuccess)
        return 1;
    probe_policy<<<1, 1>>>(device_output);
    if (cudaGetLastError() != cudaSuccess)
        return 2;
    if (cudaMemcpy(
            &host_output, device_output, sizeof(host_output),
            cudaMemcpyDeviceToHost) != cudaSuccess)
        return 3;
    std::printf("evict_first_fraction_1_policy=0x%016llx\n",
                static_cast<unsigned long long>(host_output));
    cudaFree(device_output);
    return 0;
}
