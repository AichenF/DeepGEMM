// CUDA 12.8 frontend/codegen probe for a terminal device-function boundary.
// This file does not benchmark or execute a business kernel.

__device__ __noinline__ void returning_tail(int* output, int value) {
    output[threadIdx.x] = value + static_cast<int>(threadIdx.x);
}

[[noreturn]] __device__ __noinline__ void terminal_tail(
        int* output, int value) {
    output[threadIdx.x] = value + static_cast<int>(threadIdx.x);
    asm volatile("exit;" : : : "memory");
    __builtin_unreachable();
}

extern "C" __global__ __launch_bounds__(128, 8)
void returning_entry(int* output, int value) {
    returning_tail(output, value);
}

extern "C" __global__ __launch_bounds__(128, 8)
void terminal_entry(int* output, int value) {
    terminal_tail(output, value);
}
