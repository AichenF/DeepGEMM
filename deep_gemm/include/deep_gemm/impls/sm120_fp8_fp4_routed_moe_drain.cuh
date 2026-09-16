#pragma once

#include <nccl_device.h>

#ifndef DG_SM120_ROUTED_MOE_DRAIN_KERNEL
#error "Define the exported drain kernel symbol before including this file"
#endif

#if DG_SM120_ROUTED_MOE_WORLD_SIZE != 4 && DG_SM120_ROUTED_MOE_WORLD_SIZE != 8
#error "SM120 routed MoE drain supports EP4 and EP8"
#endif

extern "C" __global__ void
DG_SM120_ROUTED_MOE_DRAIN_KERNEL(
    ncclDevComm const* __restrict__ gin_dev_comm,
    unsigned long long* __restrict__ ack_signal_base_scratch) {
    constexpr int kWorldSize = DG_SM120_ROUTED_MOE_WORLD_SIZE;
    if (threadIdx.x != 0 or ack_signal_base_scratch[kWorldSize] == 0)
        return;

    ncclGin gin{*gin_dev_comm, 0};
    #pragma unroll 1
    for (int source = 0; source < kWorldSize; ++source) {
        gin.waitSignal(
            ncclCoopThread(),
            static_cast<ncclGinSignal_t>(2 * kWorldSize + source),
            ack_signal_base_scratch[source] + 1,
            64,
            cuda::memory_order_acquire);
        ack_signal_base_scratch[source] += 1;
    }
    ack_signal_base_scratch[kWorldSize] = 0;
}
