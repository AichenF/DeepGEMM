/*************************************************************************
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
 * SPDX-License-Identifier: Apache-2.0
 *************************************************************************/

// Native SM120a CUDA export of the P8 M64xN32 two-stage residency schedule.
// A host adapter maps the qualified MegaMoE resources to this native launch;
// the kernel has no runtime dependency on CAKE, Loom, or Python.

#include <nccl_device.h>

struct __align__(128) LoomTensorMap { uint64_t opaque[16]; };
template <int N>
struct __align__(128) LoomTensorMapPack { LoomTensorMap maps[N]; };

#include <cuda_bf16.h>
#include <deep_gemm/impls/sm120_fp8_fp4_gemm_1d1d.cuh>

__device__ __forceinline__ int make_warp_uniform(int x) {
    int result;
    asm volatile("shfl.sync.idx.b32 %0, %1, 0, 0x1F, 0xFFFFFFFF;"
                 : "=r"(result) : "r"(x));
    return result;
}

#define LOOM_INF CUDART_INF_F
#define NUM_W1_PIPE_STAGES 2
#define NUM_W2_PIPE_STAGES 2
#define SMEM_CLAIMED_RECORDS_OFF 1024
#define SMEM_CLAIMED_RECORDS_STAGE_BYTES 256
#define SMEM_CLAIMED_RECORDS_STRIDE 256
#define SMEM_WARP_DST_ROW_OFF 26368
#define SMEM_WARP_DST_ROW_STAGE_BYTES 32
#define SMEM_WARP_DST_ROW_STRIDE 32
#define SMEM_W1_SMEM_A_OFF 1024
#define SMEM_W1_SMEM_A_STAGE_BYTES 8192
#define SMEM_W1_SMEM_A_STRIDE 8192
#define SMEM_W1_SMEM_B_OFF 17408
#define SMEM_W1_SMEM_B_STAGE_BYTES 4096
#define SMEM_W1_SMEM_B_STRIDE 4096
#define SMEM_W1_SMEM_SFA_OFF 25600
#define SMEM_W1_SMEM_SFA_STAGE_BYTES 256
#define SMEM_W1_SMEM_SFA_STRIDE 256
#define SMEM_W1_SMEM_SFB_OFF 26112
#define SMEM_W1_SMEM_SFB_STAGE_BYTES 128
#define SMEM_W1_SMEM_SFB_STRIDE 128
#define SMEM_TOTAL 26624
#define THREADS 256

#include <math_constants.h>
#include <cooperative_groups.h>
#include <nccl_device.h>

__device__ __forceinline__ uint32_t elect_sync() {
    uint32_t pred = 0;
    asm volatile(
        "{\n\t"
        ".reg .pred %%px;\n\t"
        "elect.sync _|%%px, %1;\n\t"
        "@%%px mov.s32 %0, 1;\n\t"
        "}\n"
        : "+r"(pred)
        : "r"(0xFFFFFFFF));
    return pred;
}


__device__ __forceinline__ void mbarrier_init(int mbar_addr, int count) {
    asm volatile("mbarrier.init.shared::cta.b64 [%0], %1;"
        :: "r"(mbar_addr), "r"(count));
}


__device__ __forceinline__ uint32_t mbarrier_try_wait(int mbar_addr, int phase) {
    uint32_t token;
    asm volatile(
        "{\n\t"
        ".reg .pred P1;\n\t"
        "mbarrier.try_wait.parity.acquire.cta.shared::cta.b64"
        " P1, [%1], %2;\n\t"
        "selp.u32 %0, 1, 0, P1;\n\t"
        "}\n"
        : "=r"(token)
        : "r"(mbar_addr), "r"(phase) : "memory");
    return token;
}

__device__ __forceinline__ uint32_t mbarrier_try_wait_cluster(int mbar_addr, int phase) {
    uint32_t token;
    asm volatile(
        "{\n\t"
        ".reg .pred P1;\n\t"
        "mbarrier.try_wait.parity.acquire.cluster.shared::cta.b64"
        " P1, [%1], %2;\n\t"
        "selp.u32 %0, 1, 0, P1;\n\t"
        "}\n"
        : "=r"(token)
        : "r"(mbar_addr), "r"(phase) : "memory");
    return token;
}

__device__ __forceinline__ void mbarrier_wait(int mbar_addr, int phase) {
    uint32_t ticks = 0x989680;
    asm volatile(
        "{\n\t"
        ".reg .pred P1;\n\t"
        "LAB_WAIT:\n\t"
        "mbarrier.try_wait.parity.acquire.cta.shared::cta.b64"
        " P1, [%0], %1, %2;\n\t"
        "@P1 bra.uni DONE;\n\t"
        "bra.uni LAB_WAIT;\n\t"
        "DONE:\n\t"
        "}\n"
        :: "r"(mbar_addr), "r"(phase), "r"(ticks) : "memory");
}

__device__ __forceinline__ void mbarrier_wait_cluster(int mbar_addr, int phase) {
    uint32_t ticks = 0x989680;
    asm volatile(
        "{\n\t"
        ".reg .pred P1;\n\t"
        "LAB_WAIT_CLUSTER:\n\t"
        "mbarrier.try_wait.parity.acquire.cluster.shared::cta.b64"
        " P1, [%0], %1, %2;\n\t"
        "@P1 bra.uni DONE_CLUSTER;\n\t"
        "bra.uni LAB_WAIT_CLUSTER;\n\t"
        "DONE_CLUSTER:\n\t"
        "}\n"
        :: "r"(mbar_addr), "r"(phase), "r"(ticks) : "memory");
}

__device__ __forceinline__ void mbarrier_wait_token(int mbar_addr, int phase, uint32_t token) {
    if (token == 0) {
        mbarrier_wait(mbar_addr, phase);
    }
}

__device__ __forceinline__ void mbarrier_wait_token_cluster(int mbar_addr, int phase, uint32_t token) {
    if (token == 0) {
        mbarrier_wait_cluster(mbar_addr, phase);
    }
}


__device__ __forceinline__ void mbarrier_arrive(int mbar_addr) {
    asm volatile(
        "mbarrier.arrive.release.cta.shared::cta.b64 _, [%0];"
        :: "r"(mbar_addr) : "memory");
}


__device__ __forceinline__ void mbarrier_arrive_expect_tx(int mbar_addr, uint32_t bytes) {
    asm volatile(
        "mbarrier.arrive.expect_tx.release.cta.shared::cta.b64 _, [%0], %1;"
        :: "r"(mbar_addr), "r"(bytes) : "memory");
}


__device__ __forceinline__ float approx_exp2(float x) {
    float y;
    asm("ex2.approx.ftz.f32 %0, %1;" : "=f"(y) : "f"(x));
    return y;
}


__device__ __forceinline__ float max_noftz(float a, float b) {
    float c;
    asm("max.f32 %0, %1, %2;" : "=f"(c) : "f"(a), "f"(b));
    return c;
}


__device__ __forceinline__ void fence_async_shared() {
    asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
}


__device__ __forceinline__ uint64_t desc_encode(uint64_t x) {
    return (x & 0x3FFFFULL) >> 4ULL;
}


__device__ __forceinline__ uint64_t make_smem_desc(int addr) {
    const int SBO = 1024;
    return desc_encode(addr)
         | (desc_encode(SBO) << 32ULL)
         | (1ULL << 46ULL)
         | (2ULL << 61ULL);
}


__device__ __forceinline__ void tma_2d_gmem2smem(
    int dst, const void *tmap_ptr, int x, int y, int mbar_addr) {
    asm volatile(
        "cp.async.bulk.tensor.2d.shared::cta.global"
        ".mbarrier::complete_tx::bytes"
        " [%0], [%1, {%2, %3}], [%4];"
        :: "r"(dst), "l"(tmap_ptr), "r"(x), "r"(y),
           "r"(mbar_addr) : "memory");
}

struct alignas(16) CakeSm120CanonicalFusedReadyParams {
    int* topk_idx_i32;
    float* topk_weights;
    int* x_fp8_i32;
    int* x_sf_i32;
    unsigned int* owner_record_counts;
    unsigned int* owner_route_counts;
    int* route_result_index;
    unsigned int* protocol_error;
    unsigned long long* dispatch_signal_base_scratch;
    unsigned long long* result_signal_base_scratch;
    int rank;
    int world_size;
    int active_rows;
    unsigned int epoch;
    ncclDevComm const* gin_dev_comm;
    int* dispatch_header_out;
    ncclWindow_t dispatch_header_out_window;
    uint8_t* dispatch_payload_out;
    ncclWindow_t dispatch_payload_out_window;
    int* dispatch_header_inbox;
    ncclWindow_t dispatch_header_inbox_window;
    uint8_t* dispatch_payload_inbox;
    ncclWindow_t dispatch_payload_inbox_window;
    unsigned int* pool_fp8_u32;
    unsigned int* pool_sf_u32;
    float* routing_weight_pool;
    int* meta_source_rank;
    int* meta_token;
    int* meta_slot;
    int* meta_result_index;
    int* expert_counts;
    int* source_record_counts;
    int* source_route_counts;
    int* source_active_rows;
    int* expert_row_offsets;
    int* expert_scatter_offsets;
    int* task_expert;
    int* task_source_rank;
    int* task_owner_rank;
    int* task_local_expert;
    int* task_pool_row;
    int* task_m_local;
    int* task_valid_m;
    int* grouped_layout;
    int* total_valid_routes;
    int* total_padded_rows;
    int* total_m_tasks;
    unsigned int* histogram_done;
    unsigned int* prefix_done;
    __nv_fp8_e4m3* w1_weight;
    cutlass::bfloat16_t* w1_bf16;
    uint8_t* intermediate_fp8;
    uint8_t* intermediate_sfa_u8;
    unsigned int* requant_groups_done;
    __nv_fp8_e4m3* w2_weight;
    cutlass::bfloat16_t* w2_bf16;
    __nv_bfloat16* final_output;
    __nv_bfloat16* result_out;
    ncclWindow_t result_out_window;
    __nv_bfloat16* result_inbox;
    ncclWindow_t result_inbox_window;
    cute::TmaDescriptor* tensor_map_buffer;
    cute::TmaDescriptor const* w1_tensor_map_a;
    cute::TmaDescriptor const* w1_tensor_map_b;
    cute::TmaDescriptor const* w1_tensor_map_sfa;
    cute::TmaDescriptor const* w1_tensor_map_sfb;
    cute::TmaDescriptor const* w1_tensor_map_d;
    cute::TmaDescriptor const* w2_tensor_map_a;
    cute::TmaDescriptor const* w2_tensor_map_b;
    cute::TmaDescriptor const* w2_tensor_map_sfa;
    cute::TmaDescriptor const* w2_tensor_map_sfb;
    cute::TmaDescriptor const* w2_tensor_map_d;

    unsigned int* w1_warp_done;
    unsigned int* w1_task_ready;
    unsigned int* w1_next_tile;
    unsigned int* w1_tiles_completed;
    unsigned int* epilogue_claimed;
    unsigned int* epilogue_completed;
    unsigned int* w2_task_ready;
    unsigned int* w2_task_claimed;
    unsigned int* w2_tile_warp_done;
    unsigned int* w2_tiles_completed;
    unsigned int* source_w2_done;
    unsigned int* combine_ready;
    unsigned int* combine_ctas_done;
    unsigned int* epoch_done;
    unsigned int* ready_audit_counts;
    int* worker_task;
    int* worker_n;
    unsigned long long* combine_ack_signal_base_scratch;
    uint8_t* ack_out;
    ncclWindow_t ack_out_window;
    uint8_t* ack_inbox;
    ncclWindow_t ack_inbox_window;
};

static_assert(sizeof(LoomTensorMap) == 128);

extern "C" {

__global__ __launch_bounds__(256) void
kernel_deepgemm_sm120_megamoe_m64n32_stage2_residency_dispatch(LoomTensorMap const* W1_A, LoomTensorMap const* W1_B, LoomTensorMap const* W1_SFA, LoomTensorMap const* W1_SFB, __nv_bfloat16* __restrict__ W1_D, LoomTensorMap const* W2_A, LoomTensorMap const* W2_B, LoomTensorMap const* W2_SFA, LoomTensorMap const* W2_SFB, __nv_bfloat16* __restrict__ W2_D, uint8_t* __restrict__ intermediate_fp8, uint8_t* __restrict__ intermediate_sfa_u8, int* __restrict__ requant_groups_done, int* __restrict__ epilogue_tasks_completed, int* __restrict__ w2_warp_done, int* __restrict__ w2_tiles_completed, int* __restrict__ topk_idx_i32, float* __restrict__ topk_weights, int* __restrict__ x_fp8_i32, int* __restrict__ x_sf_i32, int* __restrict__ owner_record_counts, int* __restrict__ owner_route_counts, int* __restrict__ route_result_index, int* __restrict__ protocol_error, unsigned long long* __restrict__ signal_base_scratch, unsigned long long* __restrict__ result_signal_base_scratch, unsigned long long* __restrict__ ack_signal_base_scratch, __nv_bfloat16* __restrict__ final_output, unsigned int* __restrict__ pool_fp8_u32, unsigned int* __restrict__ pool_sf_u32, float* __restrict__ routing_weight_pool, int* __restrict__ meta_source_rank, int* __restrict__ meta_token, int* __restrict__ meta_slot, int* __restrict__ meta_result_index, int* __restrict__ expert_counts, int* __restrict__ source_record_counts, int* __restrict__ source_route_counts, int* __restrict__ source_active_rows, int* __restrict__ expert_row_offsets, int* __restrict__ expert_scatter_offsets, int* __restrict__ task_expert, int* __restrict__ task_source_rank, int* __restrict__ task_owner_rank, int* __restrict__ task_local_expert, int* __restrict__ task_pool_row, int* __restrict__ task_m_local, int* __restrict__ task_valid_m, int* __restrict__ total_valid_routes, int* __restrict__ total_padded_rows, int* __restrict__ total_m_tasks, int* __restrict__ histogram_done, int* __restrict__ prefix_done, int* __restrict__ w1_warp_done, int* __restrict__ w1_tiles_completed, int rank, int world_size, int active_rows, unsigned int epoch, ncclDevComm const* __restrict__ gin_dev_comm, uint8_t* __restrict__ dispatch_header_out, ncclWindow_t dispatch_header_out_window, uint8_t* __restrict__ dispatch_payload_out, ncclWindow_t dispatch_payload_out_window, uint8_t* __restrict__ dispatch_header_inbox, ncclWindow_t dispatch_header_inbox_window, uint8_t* __restrict__ dispatch_payload_inbox, ncclWindow_t dispatch_payload_inbox_window, uint8_t* __restrict__ result_out, ncclWindow_t result_out_window, uint8_t* __restrict__ result_inbox, ncclWindow_t result_inbox_window, uint8_t* __restrict__ ack_out, ncclWindow_t ack_out_window, uint8_t* __restrict__ ack_inbox, ncclWindow_t ack_inbox_window)
{
    const int tid = threadIdx.x;
    const int warp = make_warp_uniform(tid / 32);
    const int lane = tid % 32;

    extern __shared__ __align__(1024) char smem_raw[];
    int smem;
    smem = (int)(unsigned long long)__cvta_generic_to_shared(smem_raw);

    const int bid = blockIdx.x;
    const int num_bids = gridDim.x;
    if (tid == 0) {
        asm volatile("fence.proxy.tensormap::generic.acquire.sys [%0], 128;" :: "l"((uint64_t)(W1_A)) : "memory");
        asm volatile("fence.proxy.tensormap::generic.acquire.sys [%0], 128;" :: "l"((uint64_t)(W1_B)) : "memory");
        asm volatile("fence.proxy.tensormap::generic.acquire.sys [%0], 128;" :: "l"((uint64_t)(W1_SFA)) : "memory");
        asm volatile("fence.proxy.tensormap::generic.acquire.sys [%0], 128;" :: "l"((uint64_t)(W1_SFB)) : "memory");
        asm volatile("fence.proxy.tensormap::generic.acquire.sys [%0], 128;" :: "l"((uint64_t)(W2_A)) : "memory");
        asm volatile("fence.proxy.tensormap::generic.acquire.sys [%0], 128;" :: "l"((uint64_t)(W2_B)) : "memory");
        asm volatile("fence.proxy.tensormap::generic.acquire.sys [%0], 128;" :: "l"((uint64_t)(W2_SFA)) : "memory");
        asm volatile("fence.proxy.tensormap::generic.acquire.sys [%0], 128;" :: "l"((uint64_t)(W2_SFB)) : "memory");
    }
    __syncthreads();


    // Kernel setup ops
    int* claimed_records = reinterpret_cast<int*>(smem_raw + 1024);
    const int claimed_records_addr = smem + 1024;
    int* warp_dst_row = reinterpret_cast<int*>(smem_raw + 26368);
    const int warp_dst_row_addr = smem + 26368;
    uint8_t* w1_smem_a = reinterpret_cast<uint8_t*>(smem_raw + 1024);
    const int w1_smem_a_addr = smem + 1024;
    uint8_t* w1_smem_b = reinterpret_cast<uint8_t*>(smem_raw + 17408);
    const int w1_smem_b_addr = smem + 17408;
    unsigned int* w1_smem_sfa = reinterpret_cast<unsigned int*>(smem_raw + 25600);
    const int w1_smem_sfa_addr = smem + 25600;
    unsigned int* w1_smem_sfb = reinterpret_cast<unsigned int*>(smem_raw + 26112);
    const int w1_smem_sfb_addr = smem + 26112;

    // Mbarrier init (4 groups, 8 barriers)
    // Mbarriers at smem_raw[0..64)

    if (warp == 0) {
        uint32_t leader = elect_sync();
        if (leader) {
            // --- pipeline 'w1_pipe' ---
            // w1_full: 2 barriers, init_count=1
            mbarrier_init(smem + 0, 1);
            mbarrier_init(smem + 8, 1);
            // w1_empty: 2 barriers, init_count=4
            mbarrier_init(smem + 16, 4);
            mbarrier_init(smem + 24, 4);
            // --- pipeline 'w2_pipe' ---
            // w2_full: 2 barriers, init_count=1
            mbarrier_init(smem + 32, 1);
            mbarrier_init(smem + 40, 1);
            // w2_empty: 2 barriers, init_count=4
            mbarrier_init(smem + 48, 4);
            mbarrier_init(smem + 56, 4);
            asm volatile("fence.mbarrier_init.release.cluster;");
        }
    }

    __syncthreads();

    const int mbar_base = smem;
    #define w1_full_addr (mbar_base + 0)
    #define w1_empty_addr (mbar_base + 16)
    #define w2_full_addr (mbar_base + 32)
    #define w2_empty_addr (mbar_base + 48)

    // === Task calls (dependency order) ===
    int reset_tid = bid * 256 + tid;
    int reset_threads = num_bids * 256;
    #pragma unroll 1
    for (int reset_peer = reset_tid; reset_peer < 8; reset_peer += reset_threads) {
        owner_record_counts[reset_peer] = 0;
        owner_route_counts[reset_peer] = 0;
        source_record_counts[reset_peer] = 0;
        source_route_counts[reset_peer] = 0;
        source_active_rows[reset_peer] = 0;
    }
    #pragma unroll 1
    for (int reset_route = reset_tid; reset_route < active_rows * 6; reset_route += reset_threads) {
        route_result_index[reset_route] = -1;
    }
    #pragma unroll 1
    for (int reset_pool_word = reset_tid; reset_pool_word < 181579776; reset_pool_word += reset_threads) {
        pool_fp8_u32[reset_pool_word] = 0;
    }
    #pragma unroll 1
    for (int reset_sf_word = reset_tid; reset_sf_word < 5674368; reset_sf_word += reset_threads) {
        pool_sf_u32[reset_sf_word] = 0;
    }
    #pragma unroll 1
    for (int reset_row = reset_tid; reset_row < 101328; reset_row += reset_threads) {
        routing_weight_pool[reset_row] = 0.0f;
        meta_source_rank[reset_row] = -1;
        meta_token[reset_row] = -1;
        meta_slot[reset_row] = -1;
        meta_result_index[reset_row] = -1;
    }
    #pragma unroll 1
    for (int reset_expert = reset_tid; reset_expert < 48; reset_expert += reset_threads) {
        expert_counts[reset_expert] = 0;
        expert_row_offsets[reset_expert] = 0;
        expert_scatter_offsets[reset_expert] = 0;
    }
    #pragma unroll 1
    for (int reset_task = reset_tid; reset_task < 1584; reset_task += reset_threads) {
        task_expert[reset_task] = -1;
        task_source_rank[reset_task] = -1;
        task_owner_rank[reset_task] = -1;
        task_local_expert[reset_task] = -1;
        task_pool_row[reset_task] = -1;
        task_m_local[reset_task] = -1;
        task_valid_m[reset_task] = -1;
    }
    #pragma unroll 1
    for (int reset_w1_tile = reset_tid; reset_w1_tile < 304128; reset_w1_tile += reset_threads) {
        w1_warp_done[reset_w1_tile] = 0;
    }
    #pragma unroll 1
    for (int reset_w2_tile = reset_tid; reset_w2_tile < 354816; reset_w2_tile += reset_threads) {
        w2_warp_done[reset_w2_tile] = 0;
    }
    if (reset_tid == 0) {
        protocol_error[0] = 0;
        total_valid_routes[0] = 0;
        total_padded_rows[0] = 0;
        total_m_tasks[0] = 0;
        histogram_done[0] = 0;
        prefix_done[0] = 0;
        w1_tiles_completed[0] = 0;
        requant_groups_done[0] = 0;
        epilogue_tasks_completed[0] = 0;
        w2_tiles_completed[0] = 0;
    }
    cooperative_groups::this_grid().sync();
    int slot = (int)(epoch & 1);
    int launch_valid = (int)(rank >= 0 && rank < world_size && world_size >= 1 && world_size <= 8 && active_rows >= 1 && active_rows <= 2048 && num_bids >= 2);
    if (launch_valid == 0) {
        if (warp == 0) {
            if (elect_sync()) {
                atomicMax(&protocol_error[0], 1);
            }
        }
    }
    if (bid == 0) {
        #pragma unroll 1
        for (int source = 0; source < 8; source++) {
            if (warp == 0) {
                if (elect_sync()) {
                    // gin_read_signal: acquire, 64-bit signal snapshot
                    uint64_t _gin_signal_0;
                    {
                        ncclGin __gin{*(gin_dev_comm), (int)(0)};
                        _gin_signal_0 = __gin.readSignal((ncclGinSignal_t)(source), 64, cuda::memory_order_acquire);
                    }
                    signal_base_scratch[source] = _gin_signal_0;
                    if (source < world_size) {
                        // gin_read_signal: acquire, 64-bit signal snapshot
                        uint64_t _gin_signal_1;
                        {
                            ncclGin __gin{*(gin_dev_comm), (int)(0)};
                            _gin_signal_1 = __gin.readSignal((ncclGinSignal_t)(8 + source), 64, cuda::memory_order_acquire);
                        }
                        result_signal_base_scratch[source] = _gin_signal_1;
                        // gin_read_signal: acquire, 64-bit signal snapshot
                        uint64_t _gin_signal_2;
                        {
                            ncclGin __gin{*(gin_dev_comm), (int)(0)};
                            _gin_signal_2 = __gin.readSignal((ncclGinSignal_t)(16 + source), 64, cuda::memory_order_acquire);
                        }
                        ack_signal_base_scratch[source] = _gin_signal_2;
                    }
                }
            }
        }
        __syncthreads();
        // gin_world_barrier: CTA/world rendezvous, no put/get drain
        {
            ncclGin __gin{*(gin_dev_comm), (int)(0)};
            ncclGinBarrierSession<ncclCoopCta> __bar{ncclCoopCta(), __gin, ncclTeamTagWorld(), (uint32_t)(0)};
            __bar.sync(ncclCoopCta(), cuda::memory_order_acquire, ncclGinFenceLevel::None);
        }
    }
    cooperative_groups::this_grid().sync();
    if (warp < 8) {
        int global_warp = bid * 8 + warp;
        int warps_per_grid = num_bids * 8;
        #pragma unroll 1
        for (int token = global_warp; token < active_rows; token += warps_per_grid) {
            #pragma unroll 1
            for (int owner = 0; owner < world_size; owner++) {
                if (lane == 0) {
                    int route_count = 0;
                    #pragma unroll
                    for (int route_slot = 0; route_slot < 6; route_slot++) {
                        int pair = token * 6 + route_slot;
                        int expert = topk_idx_i32[pair * 2];
                        int expert_hi = topk_idx_i32[pair * 2 + 1];
                        int masked = (int)(expert == -1 && expert_hi == -1);
                        int valid = (int)(expert >= 0 && expert < world_size * 48 && expert_hi == 0);
                        if (valid == 0 && masked == 0) {
                            atomicMax(&protocol_error[0], 1);
                        }
                        if (valid != 0 && expert / 48 == owner) {
                            route_count = route_count + 1;
                        }
                    }
                    int claim = -1;
                    int route_base = -1;
                    if (route_count > 0) {
                        int _atomic_old_0 = atomicAdd(&owner_record_counts[owner], 1);
                        claim = _atomic_old_0;
                        if (claim >= 2048) {
                            atomicMax(&protocol_error[0], 1);
                            claim = -1;
                        }
                        int _atomic_old_1 = atomicAdd(&owner_route_counts[owner], route_count);
                        route_base = _atomic_old_1;
                        if (route_base + route_count > 12288) {
                            atomicMax(&protocol_error[0], 1);
                            claim = -1;
                        }
                    }
                    claimed_records[warp * 8 + owner] = claim;
                    if (claim >= 0) {
                        unsigned long long record_byte = (unsigned long long)(owner * 2 + slot) * 15466496 + (unsigned long long)claim * 7552;
                        unsigned long long record_word = record_byte / 4;
                        *(reinterpret_cast<int*>(reinterpret_cast<int*>(dispatch_payload_out) + record_word) + (0)) = token;
                        *(reinterpret_cast<int*>(reinterpret_cast<int*>(dispatch_payload_out) + (record_word + 1)) + (0)) = route_count;
                        int write_route = 0;
                        #pragma unroll
                        for (int route_slot_2 = 0; route_slot_2 < 6; route_slot_2++) {
                            int pair_2 = token * 6 + route_slot_2;
                            int expert_2 = topk_idx_i32[pair_2 * 2];
                            int expert_hi_2 = topk_idx_i32[pair_2 * 2 + 1];
                            int valid_2 = (int)(expert_2 >= 0 && expert_2 < world_size * 48 && expert_hi_2 == 0);
                            if (valid_2 != 0 && expert_2 / 48 == owner) {
                                *(reinterpret_cast<int*>(reinterpret_cast<int*>(dispatch_payload_out) + (record_word + 2 + write_route)) + (0)) = expert_2 - owner * 48;
                                *(reinterpret_cast<int*>(reinterpret_cast<int*>(dispatch_payload_out) + (record_word + 8 + write_route)) + (0)) = route_slot_2;
                                *(reinterpret_cast<float*>(reinterpret_cast<float*>(dispatch_payload_out) + (record_word + 14 + write_route)) + (0)) = topk_weights[pair_2];
                                route_result_index[pair_2] = route_base + write_route;
                                write_route = write_route + 1;
                            }
                        }
                        *(reinterpret_cast<int*>(reinterpret_cast<int*>(dispatch_payload_out) + (record_word + 20)) + (0)) = rank;
                        *(reinterpret_cast<int*>(reinterpret_cast<int*>(dispatch_payload_out) + (record_word + 21)) + (0)) = 1347571524;
                        *(reinterpret_cast<int*>(reinterpret_cast<int*>(dispatch_payload_out) + (record_word + 22)) + (0)) = route_base;
                    }
                }
                __syncwarp();
                int record_idx = claimed_records[warp * 8 + owner];
                if (record_idx >= 0) {
                    unsigned long long record_byte_2 = (unsigned long long)(owner * 2 + slot) * 15466496 + (unsigned long long)record_idx * 7552;
                    unsigned long long record_word_2 = record_byte_2 / 4;
                    unsigned long long src_activation = (unsigned long long)token * 1792;
                    unsigned long long dst_activation = record_word_2 + 32;
                    #pragma unroll 1
                    for (int word = lane; word < 1792; word += 32) {
                        *(reinterpret_cast<int*>(reinterpret_cast<int*>(dispatch_payload_out) + (dst_activation + (unsigned long long)word)) + (0)) = x_fp8_i32[src_activation + (unsigned long long)word];
                    }
                    unsigned long long src_sf = (unsigned long long)token * 56;
                    unsigned long long dst_sf = record_word_2 + 1824;
                    #pragma unroll 1
                    for (int sf_word = lane; sf_word < 56; sf_word += 32) {
                        *(reinterpret_cast<int*>(reinterpret_cast<int*>(dispatch_payload_out) + (dst_sf + (unsigned long long)sf_word)) + (0)) = x_sf_i32[src_sf + (unsigned long long)sf_word];
                    }
                }
                __syncwarp();
            }
        }
    }
    __threadfence_system();
    cooperative_groups::this_grid().sync();
    if (bid == 0) {
        #pragma unroll 1
        for (int peer = 0; peer < world_size; peer++) {
            int local_header_word = (peer * 2 + slot) * 8;
            int local_header_byte = local_header_word * 4;
            int local_payload_byte = (peer * 2 + slot) * 15466496;
            int remote_header_byte = (rank * 2 + slot) * 32;
            int remote_payload_byte = (rank * 2 + slot) * 15466496;
            if (warp == 0) {
                if (elect_sync()) {
                    int count = owner_record_counts[peer];
                    int peer_route_count = owner_route_counts[peer];
                    int _max_0 = ((count) > (0) ? (count) : (0));
                    int _min_0 = ((_max_0) < (2048) ? (_max_0) : (2048));
                    int safe_count = _min_0;
                    int _max_1 = ((safe_count) > (1) ? (safe_count) : (1));
                    int send_count = _max_1;
                    int payload_bytes = send_count * 7552;
                    int header_values[8];
                    header_values[0] = 1347571524;
                    header_values[1] = 1;
                    header_values[2] = (int)epoch;
                    header_values[3] = rank;
                    header_values[4] = peer;
                    header_values[5] = safe_count;
                    header_values[6] = peer_route_count;
                    header_values[7] = active_rows;
                    {
                        int4 _iv4 = make_int4(header_values[0 + 0], header_values[0 + 1], header_values[0 + 2], header_values[0 + 3]);
                        *reinterpret_cast<int4*>(reinterpret_cast<int*>(dispatch_header_out) + local_header_word + 0) = _iv4;
                    }
                    {
                        int4 _iv4 = make_int4(header_values[4 + 0], header_values[4 + 1], header_values[4 + 2], header_values[4 + 3]);
                        *reinterpret_cast<int4*>(reinterpret_cast<int*>(dispatch_header_out) + (local_header_word + 4) + 0) = _iv4;
                    }
                    __threadfence_system();
                    // gin_put_signal_add: weak remote completion on context 0
                    {
                        ncclGin __gin{*(gin_dev_comm), (int)(0)};
                        __gin.put(ncclTeamWorld(*(gin_dev_comm)), (int)(peer), dispatch_header_inbox_window, (size_t)(remote_header_byte), dispatch_header_out_window, (size_t)(local_header_byte), (size_t)(32),
                            ncclGin_WeakSignalAdd{(ncclGinSignal_t)(rank), (uint64_t)(1)}, ncclGin_None{}, ncclCoopThread());
                    }
                    // gin_put_signal_add: strong remote completion on context 0
                    {
                        ncclGin __gin{*(gin_dev_comm), (int)(0)};
                        __gin.put(ncclTeamWorld(*(gin_dev_comm)), (int)(peer), dispatch_payload_inbox_window, (size_t)(remote_payload_byte), dispatch_payload_out_window, (size_t)(local_payload_byte), (size_t)(payload_bytes),
                            ncclGin_StrongSignalAdd{(ncclGinSignal_t)(rank), (uint64_t)(1)}, ncclGin_None{}, ncclCoopThread());
                    }
                }
            }
        }
        #pragma unroll 1
        for (int source_2 = 0; source_2 < world_size; source_2++) {
            if (warp == 0) {
                if (elect_sync()) {
                    // gin_wait_signal: acquire, rolling 64-bit comparison
                    {
                        ncclGin __gin{*(gin_dev_comm), (int)(0)};
                        __gin.waitSignal(ncclCoopThread(), (ncclGinSignal_t)(source_2), (uint64_t)(signal_base_scratch[source_2] + 2), 64, cuda::memory_order_acquire);
                    }
                }
            }
        }
    }
    int grid_tid = bid * 256 + tid;
    int grid_threads = num_bids * 256;
    #pragma unroll 1
    for (int source_3 = grid_tid; source_3 < world_size; source_3 += grid_threads) {
        int header_word = (source_3 * 2 + slot) * 8;
        int count_1 = reinterpret_cast<const int*>(reinterpret_cast<int*>(dispatch_header_inbox) + (header_word + 5))[0];
        int routes = reinterpret_cast<const int*>(reinterpret_cast<int*>(dispatch_header_inbox) + (header_word + 6))[0];
        int rows = reinterpret_cast<const int*>(reinterpret_cast<int*>(dispatch_header_inbox) + (header_word + 7))[0];
        int header_ok = (int)(reinterpret_cast<const int*>(reinterpret_cast<int*>(dispatch_header_inbox) + header_word)[0] == 1347571524 && reinterpret_cast<const int*>(reinterpret_cast<int*>(dispatch_header_inbox) + (header_word + 1))[0] == 1 && reinterpret_cast<const int*>(reinterpret_cast<int*>(dispatch_header_inbox) + (header_word + 2))[0] == (int)epoch && reinterpret_cast<const int*>(reinterpret_cast<int*>(dispatch_header_inbox) + (header_word + 3))[0] == source_3 && reinterpret_cast<const int*>(reinterpret_cast<int*>(dispatch_header_inbox) + (header_word + 4))[0] == rank && count_1 >= 0 && count_1 <= 2048 && routes >= 0 && routes <= 12288 && rows >= 1 && rows <= 2048);
        if (header_ok != 0) {
            source_record_counts[source_3] = count_1;
            source_route_counts[source_3] = routes;
            source_active_rows[source_3] = rows;
        } else {
            source_record_counts[source_3] = 0;
            source_route_counts[source_3] = 0;
            source_active_rows[source_3] = 0;
            atomicMax(&protocol_error[0], 1);
        }
    }
    cooperative_groups::this_grid().sync();
    #pragma unroll 1
    for (int candidate = grid_tid; candidate < 98304; candidate += grid_threads) {
        int source_4 = candidate / 12288;
        int candidate_rem = candidate - source_4 * 12288;
        int record = candidate_rem / 6;
        int record_route = candidate_rem - record * 6;
        if (source_4 < world_size && record < source_record_counts[source_4]) {
            unsigned long long candidate_record_word = (unsigned long long)(source_4 * 2 + slot) * 3866624 + (unsigned long long)record * 1888;
            int candidate_route_count = reinterpret_cast<const int*>(reinterpret_cast<int*>(dispatch_payload_inbox) + (candidate_record_word + 1))[0];
            int candidate_route_base = reinterpret_cast<const int*>(reinterpret_cast<int*>(dispatch_payload_inbox) + (candidate_record_word + 22))[0];
            int candidate_record_ok = (int)(candidate_route_count >= 1 && candidate_route_count <= 6 && candidate_route_base >= 0 && candidate_route_base + candidate_route_count <= source_route_counts[source_4] && candidate_route_base + candidate_route_count <= 12288 && reinterpret_cast<const int*>(reinterpret_cast<int*>(dispatch_payload_inbox) + (candidate_record_word + 20))[0] == source_4 && reinterpret_cast<const int*>(reinterpret_cast<int*>(dispatch_payload_inbox) + (candidate_record_word + 21))[0] == 1347571524);
            if (candidate_record_ok == 0) {
                atomicMax(&protocol_error[0], 1);
            } else if (record_route < candidate_route_count) {
                int candidate_local_expert = reinterpret_cast<const int*>(reinterpret_cast<int*>(dispatch_payload_inbox) + (candidate_record_word + 2 + (unsigned long long)record_route))[0];
                int candidate_token = reinterpret_cast<const int*>(reinterpret_cast<int*>(dispatch_payload_inbox) + candidate_record_word)[0];
                int candidate_topk_slot = reinterpret_cast<const int*>(reinterpret_cast<int*>(dispatch_payload_inbox) + (candidate_record_word + 8 + (unsigned long long)record_route))[0];
                float candidate_route_weight = reinterpret_cast<const float*>(reinterpret_cast<float*>(dispatch_payload_inbox) + (candidate_record_word + 14 + (unsigned long long)record_route))[0];
                unsigned int candidate_weight_bits = 0;
                candidate_weight_bits = reinterpret_cast<unsigned int*>(&candidate_route_weight)[0];
                int candidate_finite = (int)((candidate_weight_bits & 2139095040) != 2139095040);
                int candidate_route_ok = (int)(candidate_local_expert >= 0 && candidate_local_expert < 48 && candidate_token >= 0 && candidate_token < source_active_rows[source_4] && candidate_topk_slot >= 0 && candidate_topk_slot < 6 && candidate_finite != 0);
                if (candidate_route_ok != 0) {
                    atomicAdd(&expert_counts[candidate_local_expert], 1);
                } else {
                    atomicMax(&protocol_error[0], 1);
                }
            }
        }
    }
    if (tid == 0) {
        atomicAdd(&histogram_done[0], 1);
    }
    cooperative_groups::this_grid().sync();
    if (bid == 0 && tid == 0) {
        if (histogram_done[0] != num_bids) {
            atomicMax(&protocol_error[0], 1);
        }
        int running = 0;
        int task_idx = 0;
        int valid_routes = 0;
        #pragma unroll 1
        for (int expert_3 = 0; expert_3 < 48; expert_3++) {
            int expert_count = expert_counts[expert_3];
            int padded = (expert_count + 64 - 1) / 64 * 64;
            expert_row_offsets[expert_3] = running;
            valid_routes = valid_routes + expert_count;
            #pragma unroll 1
            for (int m_local = 0; m_local < padded; m_local += 64) {
                if (task_idx < 1584) {
                    task_expert[task_idx] = rank * 48 + expert_3;
                    task_source_rank[task_idx] = 0;
                    task_owner_rank[task_idx] = rank;
                    task_local_expert[task_idx] = expert_3;
                    task_pool_row[task_idx] = running + m_local;
                    task_m_local[task_idx] = m_local;
                    int _min_1 = ((64) < (expert_count - m_local) ? (64) : (expert_count - m_local));
                    task_valid_m[task_idx] = _min_1;
                } else {
                    atomicMax(&protocol_error[0], 1);
                }
                task_idx = task_idx + 1;
            }
            running = running + padded;
        }
        if (valid_routes > 98304 || running > 101328 || task_idx > 1584) {
            atomicMax(&protocol_error[0], 1);
        }
        total_valid_routes[0] = valid_routes;
        total_padded_rows[0] = running;
        total_m_tasks[0] = task_idx;
        __threadfence();
        prefix_done[0] = 1;
    }
    cooperative_groups::this_grid().sync();
    if (warp < 8) {
        int task_global_warp = bid * 8 + warp;
        int task_grid_warps = num_bids * 8;
        #pragma unroll 1
        for (int candidate_2 = task_global_warp; candidate_2 < 98304; candidate_2 += task_grid_warps) {
            int source_5 = candidate_2 / 12288;
            int candidate_rem_2 = candidate_2 - source_5 * 12288;
            int record_2 = candidate_rem_2 / 6;
            int record_route_2 = candidate_rem_2 - record_2 * 6;
            if (source_5 < world_size && record_2 < source_record_counts[source_5]) {
                unsigned long long scatter_record_word = (unsigned long long)(source_5 * 2 + slot) * 3866624 + (unsigned long long)record_2 * 1888;
                int scatter_route_count = reinterpret_cast<const int*>(reinterpret_cast<int*>(dispatch_payload_inbox) + (scatter_record_word + 1))[0];
                int scatter_route_base = reinterpret_cast<const int*>(reinterpret_cast<int*>(dispatch_payload_inbox) + (scatter_record_word + 22))[0];
                int scatter_record_ok = (int)(scatter_route_count >= 1 && scatter_route_count <= 6 && scatter_route_base >= 0 && scatter_route_base + scatter_route_count <= source_route_counts[source_5] && scatter_route_base + scatter_route_count <= 12288 && reinterpret_cast<const int*>(reinterpret_cast<int*>(dispatch_payload_inbox) + (scatter_record_word + 20))[0] == source_5 && reinterpret_cast<const int*>(reinterpret_cast<int*>(dispatch_payload_inbox) + (scatter_record_word + 21))[0] == 1347571524);
                if (scatter_record_ok == 0) {
                    if (lane == 0) {
                        atomicMax(&protocol_error[0], 1);
                    }
                } else if (record_route_2 < scatter_route_count) {
                    int scatter_local_expert = reinterpret_cast<const int*>(reinterpret_cast<int*>(dispatch_payload_inbox) + (scatter_record_word + 2 + (unsigned long long)record_route_2))[0];
                    int scatter_token = reinterpret_cast<const int*>(reinterpret_cast<int*>(dispatch_payload_inbox) + scatter_record_word)[0];
                    int scatter_topk_slot = reinterpret_cast<const int*>(reinterpret_cast<int*>(dispatch_payload_inbox) + (scatter_record_word + 8 + (unsigned long long)record_route_2))[0];
                    float scatter_route_weight = reinterpret_cast<const float*>(reinterpret_cast<float*>(dispatch_payload_inbox) + (scatter_record_word + 14 + (unsigned long long)record_route_2))[0];
                    int scatter_result_index = scatter_route_base + record_route_2;
                    unsigned int scatter_weight_bits = 0;
                    scatter_weight_bits = reinterpret_cast<unsigned int*>(&scatter_route_weight)[0];
                    int scatter_finite = (int)((scatter_weight_bits & 2139095040) != 2139095040);
                    int scatter_route_ok = (int)(scatter_local_expert >= 0 && scatter_local_expert < 48 && scatter_token >= 0 && scatter_token < source_active_rows[source_5] && scatter_topk_slot >= 0 && scatter_topk_slot < 6 && scatter_finite != 0);
                    if (scatter_route_ok != 0) {
                        if (lane == 0) {
                            int _atomic_old_2 = atomicAdd(&expert_scatter_offsets[scatter_local_expert], 1);
                            int scatter_claim = _atomic_old_2;
                            int dst_row = expert_row_offsets[scatter_local_expert] + scatter_claim;
                            if (scatter_claim < 0 || scatter_claim >= expert_counts[scatter_local_expert] || dst_row < 0 || dst_row >= 101328) {
                                atomicMax(&protocol_error[0], 1);
                                dst_row = -1;
                            } else {
                                meta_source_rank[dst_row] = source_5;
                                meta_token[dst_row] = scatter_token;
                                meta_slot[dst_row] = scatter_topk_slot;
                                meta_result_index[dst_row] = scatter_result_index;
                                routing_weight_pool[dst_row] = scatter_route_weight;
                            }
                            warp_dst_row[warp] = dst_row;
                        }
                        __syncwarp();
                        int scatter_dst_row = warp_dst_row[warp];
                        if (scatter_dst_row >= 0) {
                            unsigned long long src_activation_word = scatter_record_word + 32;
                            unsigned long long dst_activation_word = (unsigned long long)scatter_dst_row * 1792;
                            #pragma unroll 1
                            for (int activation_word_2 = lane; activation_word_2 < 1792; activation_word_2 += 32) {
                                pool_fp8_u32[dst_activation_word + (unsigned long long)activation_word_2] = (unsigned int)reinterpret_cast<const int*>(reinterpret_cast<int*>(dispatch_payload_inbox) + (src_activation_word + (unsigned long long)activation_word_2))[0];
                            }
                            unsigned long long src_scale_word = scatter_record_word + 1824;
                            #pragma unroll 1
                            for (int scale_word_2 = lane; scale_word_2 < 56; scale_word_2 += 32) {
                                pool_sf_u32[(unsigned long long)scale_word_2 * 101328 + (unsigned long long)scatter_dst_row] = (unsigned int)reinterpret_cast<const int*>(reinterpret_cast<int*>(dispatch_payload_inbox) + (src_scale_word + (unsigned long long)scale_word_2))[0];
                            }
                        }
                        __syncwarp();
                    } else if (lane == 0) {
                        atomicMax(&protocol_error[0], 1);
                    }
                }
            }
        }
    }
    cooperative_groups::this_grid().sync();
    if (bid == 0 && tid == 0) {
        int scatter_sum = 0;
        int expected_tasks = 0;
        #pragma unroll 1
        for (int expert_4 = 0; expert_4 < 48; expert_4++) {
            int expert_count_2 = expert_counts[expert_4];
            scatter_sum = scatter_sum + expert_scatter_offsets[expert_4];
            expected_tasks = expected_tasks + (expert_count_2 + 64 - 1) / 64;
            if (expert_scatter_offsets[expert_4] != expert_counts[expert_4]) {
                atomicMax(&protocol_error[0], 1);
            }
        }
        if (scatter_sum != total_valid_routes[0] || expected_tasks != total_m_tasks[0] || prefix_done[0] != 1) {
            atomicMax(&protocol_error[0], 1);
        }
    }
    cooperative_groups::this_grid().sync();
    int worker_ctas = num_bids - 1;
    if (warp == 4) {
        if (elect_sync()) {
            asm volatile("prefetch.tensormap [%0];" :: "l"((uint64_t)(W1_A)) : "memory");
            asm volatile("prefetch.tensormap [%0];" :: "l"((uint64_t)(W1_B)) : "memory");
            asm volatile("prefetch.tensormap [%0];" :: "l"((uint64_t)(W1_SFA)) : "memory");
            asm volatile("prefetch.tensormap [%0];" :: "l"((uint64_t)(W1_SFB)) : "memory");
        }
    }
    unsigned int _phase_w1_empty = 1;
    if (bid > 0 && warp >= 4) {
        unsigned int w1_load_stage = 0;
        int w1_tile_count = total_m_tasks[0] * 192;
        #pragma unroll 1
        for (int w1_tile = bid - 1; w1_tile < w1_tile_count; w1_tile += worker_ctas) {
            int w1_task = w1_tile / 192;
            int w1_n_block = w1_tile - w1_task * 192;
            int w1_pool_row = task_pool_row[w1_task];
            int w1_local_expert = task_local_expert[w1_task];
            #pragma unroll 1
            for (int w1_k_block = 0; w1_k_block < 56; w1_k_block++) {
                mbarrier_wait(w1_empty_addr + (w1_load_stage) * 8, _phase_w1_empty);
                if (warp == 4) {
                    if (elect_sync()) {
                        tma_2d_gmem2smem(w1_smem_sfa_addr + w1_load_stage * 256, W1_SFA, w1_pool_row, w1_k_block, w1_full_addr + (w1_load_stage) * 8);
                        tma_2d_gmem2smem(w1_smem_sfb_addr + w1_load_stage * 128, W1_SFB, w1_n_block * 32, w1_local_expert * 56 + w1_k_block, w1_full_addr + (w1_load_stage) * 8);
                        tma_2d_gmem2smem(w1_smem_a_addr + w1_load_stage * 8192, W1_A, w1_k_block * 128, w1_pool_row, w1_full_addr + (w1_load_stage) * 8);
                        tma_2d_gmem2smem(w1_smem_b_addr + w1_load_stage * 4096, W1_B, w1_k_block * 128, w1_local_expert * 6144 + w1_n_block * 32, w1_full_addr + (w1_load_stage) * 8);
                        mbarrier_arrive_expect_tx(w1_full_addr + (w1_load_stage) * 8, 10624);
                    }
                }
                w1_load_stage += 1;
                if (w1_load_stage == 2) { w1_load_stage = 0; _phase_w1_empty ^= 1; }
            }
        }
    }
    unsigned int _phase_w1_full = 0;
    if (bid > 0 && warp < 4) {
        unsigned int w1_math_stage = 0;
        int w1_warp_m = warp;
        int w1_group_id = lane / 4;
        int w1_thread_id = lane % 4;
        float w1_accum[16];
        unsigned int w1_a_frag[4];
        unsigned int w1_b_frag[2];
        unsigned int w1_sfa_word[1];
        unsigned int w1_sfb_word[1];
        int w1_tile_count_2 = total_m_tasks[0] * 192;
        #pragma unroll 1
        for (int w1_tile_2 = bid - 1; w1_tile_2 < w1_tile_count_2; w1_tile_2 += worker_ctas) {
            w1_accum[0] = 0.0f;
            w1_accum[1] = 0.0f;
            w1_accum[2] = 0.0f;
            w1_accum[3] = 0.0f;
            w1_accum[4] = 0.0f;
            w1_accum[5] = 0.0f;
            w1_accum[6] = 0.0f;
            w1_accum[7] = 0.0f;
            w1_accum[8] = 0.0f;
            w1_accum[9] = 0.0f;
            w1_accum[10] = 0.0f;
            w1_accum[11] = 0.0f;
            w1_accum[12] = 0.0f;
            w1_accum[13] = 0.0f;
            w1_accum[14] = 0.0f;
            w1_accum[15] = 0.0f;
            int w1_task_2 = w1_tile_2 / 192;
            int w1_n_block_2 = w1_tile_2 - w1_task_2 * 192;
            int w1_pool_row_2 = task_pool_row[w1_task_2];
            #pragma unroll 1
            for (int w1_k_block_2 = 0; w1_k_block_2 < 56; w1_k_block_2++) {
                mbarrier_wait(w1_full_addr + (w1_math_stage) * 8, _phase_w1_full);
                asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                #pragma unroll
                for (int w1_k_step = 0; w1_k_step < 4; w1_k_step++) {
                    int w1_a_row = (lane & 7) + (lane >> 3 & 1) * 8 + w1_warp_m * 16;
                    int w1_a_col = (lane >> 4) * 16 + w1_k_step * 32;
                    int w1_a_addr = w1_smem_a_addr + w1_math_stage * 8192 + (unsigned int)(w1_a_row * 128) + (unsigned int)(w1_a_col ^ (w1_a_row & 7) << 4);
                    asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                        : "=r"(w1_a_frag[0]), "=r"(w1_a_frag[1]), "=r"(w1_a_frag[2]), "=r"(w1_a_frag[3])
                        : "r"(w1_a_addr)
                        : "memory");
                    int w1_sfa_row = w1_warp_m * 16 + w1_group_id + (w1_thread_id & 1) * 8;
                    asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&w1_sfa_word[0])) : "r"(w1_smem_sfa_addr + w1_math_stage * 256 + (unsigned int)(w1_sfa_row * 4)));
                    w1_sfa_word[0] = w1_sfa_word[0] >> (unsigned int)(w1_k_step * 8) & 255;
                    #pragma unroll
                    for (int w1_n_tile = 0; w1_n_tile < 4; w1_n_tile++) {
                        int w1_global_n_tile = w1_n_tile;
                        int w1_b_row = (lane & 7) + w1_global_n_tile * 8;
                        int w1_b_col = (lane >> 3 & 1) * 16 + w1_k_step * 32;
                        int w1_b_addr = w1_smem_b_addr + w1_math_stage * 4096 + (unsigned int)(w1_b_row * 128) + (unsigned int)(w1_b_col ^ (w1_b_row & 7) << 4);
                        asm volatile("ldmatrix.sync.aligned.shared::cta.m8n16.x2.b8x16.b4x16_p64 {%0, %1}, [%2];\n"
                            : "=r"(w1_b_frag[0]), "=r"(w1_b_frag[1])
                            : "r"(w1_b_addr)
                            : "memory");
                        int w1_sfb_row = w1_global_n_tile * 8 + w1_group_id;
                        asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&w1_sfb_word[0])) : "r"(w1_smem_sfb_addr + w1_math_stage * 128 + (unsigned int)(w1_sfb_row * 4)));
                        w1_sfb_word[0] = w1_sfb_word[0] >> (unsigned int)(w1_k_step * 8) & 255;
                        asm volatile("mma.sync.aligned.kind::mxf8f6f4.block_scale.scale_vec::1X.m16n8k32.row.col.f32.e4m3.e2m1.f32.ue8m0 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3}, {%10}, {%11, %12}, {%13}, {%14, %15};\n"
                            : "+f"((w1_accum + w1_n_tile * 4)[0]), "+f"((w1_accum + w1_n_tile * 4)[1]), "+f"((w1_accum + w1_n_tile * 4)[2]), "+f"((w1_accum + w1_n_tile * 4)[3])
                            : "r"(w1_a_frag[0]), "r"(w1_a_frag[1]), "r"(w1_a_frag[2]), "r"(w1_a_frag[3]), "r"(((uint32_t)(w1_b_frag[0]) << 2)), "r"(((uint32_t)(w1_b_frag[1]) << 2)), "r"(w1_sfa_word[0]), "h"(((uint16_t)0)), "h"(((uint16_t)0)), "r"(w1_sfb_word[0]), "h"(((uint16_t)0)), "h"(((uint16_t)0)));
                    }
                }
                if (elect_sync()) {
                    mbarrier_arrive(w1_empty_addr + (w1_math_stage) * 8);
                }
                w1_math_stage += 1;
                if (w1_math_stage == 2) { w1_math_stage = 0; _phase_w1_empty ^= 1; _phase_w1_full ^= 1; }
            }
            #pragma unroll
            for (int w1_n_tile_2 = 0; w1_n_tile_2 < 4; w1_n_tile_2++) {
                int w1_output_n_tile = w1_n_tile_2;
                int w1_output_col = w1_n_block_2 * 32 + w1_output_n_tile * 8 + w1_thread_id * 2;
                int w1_output_row_0 = w1_pool_row_2 + w1_warp_m * 16 + w1_group_id;
                int w1_output_row_1 = w1_output_row_0 + 8;
                int w1_acc_base = w1_n_tile_2 * 4;
                W1_D[w1_output_row_0 * 6144 + w1_output_col] = w1_accum[w1_acc_base];
                W1_D[w1_output_row_0 * 6144 + w1_output_col + 1] = w1_accum[w1_acc_base + 1];
                W1_D[w1_output_row_1 * 6144 + w1_output_col] = w1_accum[w1_acc_base + 2];
                W1_D[w1_output_row_1 * 6144 + w1_output_col + 1] = w1_accum[w1_acc_base + 3];
            }
            __threadfence();
            __syncwarp();
            if (elect_sync()) {
                int _atomic_old_3 = atomicAdd(&w1_warp_done[w1_tile_2], 1);
                int w1_previous = _atomic_old_3;
                if (w1_previous == 3) {
                    atomicAdd(&w1_tiles_completed[0], 1);
                } else if (w1_previous >= 4) {
                    atomicMax(&protocol_error[0], 1);
                }
            }
        }
    }
    cooperative_groups::this_grid().sync();
    if (bid == 0 && tid == 0) {
        if (w1_tiles_completed[0] != total_m_tasks[0] * 192) {
            atomicMax(&protocol_error[0], 1);
        }
    }
    cooperative_groups::this_grid().sync();
    if (bid > 0) {
        #pragma unroll 1
        for (int epilogue_task = bid - 1; epilogue_task < total_m_tasks[0]; epilogue_task += worker_ctas) {
            if (tid < 256) {
                int epilogue_warp = tid / 32;
                #pragma unroll 1
                for (int local_rg = epilogue_warp; local_rg < 6144; local_rg += 8) {
                    int epilogue_row = epilogue_task * 64 + local_rg / 96;
                    int epilogue_group = local_rg % 96;
                    int logical_n = epilogue_group * 32 + lane;
                    int physical_gate = logical_n / 8 * 16 + (logical_n & 7);
                    int physical_up = physical_gate + 8;
                    float gate = (float)W1_D[epilogue_row * 6144 + physical_gate];
                    float up = (float)W1_D[epilogue_row * 6144 + physical_up];
                    float _min_2 = fminf(gate, 10.0f);
                    gate = _min_2;
                    float _max_2 = max_noftz(up, -10.0f);
                    float _min_3 = fminf(_max_2, 10.0f);
                    up = _min_3;
                    float _exp2_0 = approx_exp2((-gate) * 1.4426950408889634f);
                    float sigmoid = 1.0f / (1.0f + _exp2_0);
                    float routed = gate * sigmoid * up * routing_weight_pool[epilogue_row];
                    float _max_3 = max_noftz(routed, -routed);
                    float requant_amax = _max_3;
                    float _shfl_xor_0 = __shfl_xor_sync(0xFFFFFFFF, requant_amax, 16);
                    float _max_4 = max_noftz(requant_amax, _shfl_xor_0);
                    requant_amax = _max_4;
                    float _shfl_xor_1 = __shfl_xor_sync(0xFFFFFFFF, requant_amax, 8);
                    float _max_5 = max_noftz(requant_amax, _shfl_xor_1);
                    requant_amax = _max_5;
                    float _shfl_xor_2 = __shfl_xor_sync(0xFFFFFFFF, requant_amax, 4);
                    float _max_6 = max_noftz(requant_amax, _shfl_xor_2);
                    requant_amax = _max_6;
                    float _shfl_xor_3 = __shfl_xor_sync(0xFFFFFFFF, requant_amax, 2);
                    float _max_7 = max_noftz(requant_amax, _shfl_xor_3);
                    requant_amax = _max_7;
                    float _shfl_xor_4 = __shfl_xor_sync(0xFFFFFFFF, requant_amax, 1);
                    float _max_8 = max_noftz(requant_amax, _shfl_xor_4);
                    requant_amax = _max_8;
                    float requant_sf = requant_amax * 0.002232142857142857f;
                    unsigned int requant_sf_bits = 0;
                    requant_sf_bits = reinterpret_cast<unsigned int*>(&requant_sf)[0];
                    unsigned int requant_sf_exp = (requant_sf_bits >> 23 & 255) + ((requant_sf_bits & 8388607) + 8388607 >> 23);
                    unsigned int _min_4 = ((requant_sf_exp) < (254) ? (requant_sf_exp) : (254));
                    requant_sf_exp = _min_4;
                    unsigned int requant_sf_inv_bits = 254 - requant_sf_exp << 23;
                    float requant_sf_inv = 0.0f;
                    requant_sf_inv = reinterpret_cast<float*>(&requant_sf_inv_bits)[0];
                    if (lane == 0) {
                        int requant_scale_index = ((epilogue_group >> 2) * 101328 + epilogue_row) * 4 + (epilogue_group & 3);
                        *(reinterpret_cast<unsigned char*>(intermediate_sfa_u8 + requant_scale_index) + (0)) = (unsigned char)(requant_sf_exp);
                        atomicAdd(&requant_groups_done[0], 1);
                    }
                    {
                        unsigned short _fp8_pair;
                        asm("cvt.rn.satfinite.e4m3x2.f32 %0, 0f00000000, %1;" : "=h"(_fp8_pair) : "f"(routed * requant_sf_inv));
                        *(reinterpret_cast<unsigned char*>(intermediate_fp8 + (epilogue_row * 3072 + logical_n)) + (0)) = (unsigned char)(_fp8_pair & 0xFF);
                    }
                }
            }
            __threadfence();
            __syncthreads();
            if (tid == 0) {
                atomicAdd(&epilogue_tasks_completed[0], 1);
            }
            __syncthreads();
        }
    }
    cooperative_groups::this_grid().sync();
    if (bid == 0 && tid == 0) {
        int expected_requant_groups = total_m_tasks[0] * 64 * 96;
        if (requant_groups_done[0] != expected_requant_groups || epilogue_tasks_completed[0] != total_m_tasks[0]) {
            atomicMax(&protocol_error[0], 1);
        }
    }
    cooperative_groups::this_grid().sync();
    if (warp == 4) {
        if (elect_sync()) {
            asm volatile("prefetch.tensormap [%0];" :: "l"((uint64_t)(W2_A)) : "memory");
            asm volatile("prefetch.tensormap [%0];" :: "l"((uint64_t)(W2_B)) : "memory");
            asm volatile("prefetch.tensormap [%0];" :: "l"((uint64_t)(W2_SFA)) : "memory");
            asm volatile("prefetch.tensormap [%0];" :: "l"((uint64_t)(W2_SFB)) : "memory");
        }
    }
    unsigned int _phase_w2_empty = 1;
    if (bid > 0 && warp >= 4) {
        unsigned int w2_load_stage = 0;
        int w2_tile_count = total_m_tasks[0] * 224;
        #pragma unroll 1
        for (int w2_tile = bid - 1; w2_tile < w2_tile_count; w2_tile += worker_ctas) {
            int w2_task = w2_tile / 224;
            int w2_n_block = w2_tile - w2_task * 224;
            int w2_pool_row = task_pool_row[w2_task];
            int w2_local_expert = task_local_expert[w2_task];
            #pragma unroll 1
            for (int w2_k_block = 0; w2_k_block < 24; w2_k_block++) {
                mbarrier_wait(w2_empty_addr + (w2_load_stage) * 8, _phase_w2_empty);
                if (warp == 4) {
                    if (elect_sync()) {
                        tma_2d_gmem2smem(w1_smem_sfa_addr + w2_load_stage * 256, W2_SFA, w2_pool_row, w2_k_block, w2_full_addr + (w2_load_stage) * 8);
                        tma_2d_gmem2smem(w1_smem_sfb_addr + w2_load_stage * 128, W2_SFB, w2_n_block * 32, w2_local_expert * 24 + w2_k_block, w2_full_addr + (w2_load_stage) * 8);
                        tma_2d_gmem2smem(w1_smem_a_addr + w2_load_stage * 8192, W2_A, w2_k_block * 128, w2_pool_row, w2_full_addr + (w2_load_stage) * 8);
                        tma_2d_gmem2smem(w1_smem_b_addr + w2_load_stage * 4096, W2_B, w2_k_block * 128, w2_local_expert * 7168 + w2_n_block * 32, w2_full_addr + (w2_load_stage) * 8);
                        mbarrier_arrive_expect_tx(w2_full_addr + (w2_load_stage) * 8, 10624);
                    }
                }
                w2_load_stage += 1;
                if (w2_load_stage == 2) { w2_load_stage = 0; _phase_w2_empty ^= 1; }
            }
        }
    }
    unsigned int _phase_w2_full = 0;
    if (bid > 0 && warp < 4) {
        unsigned int w2_math_stage = 0;
        int w2_warp_m = warp;
        int w2_group_id = lane / 4;
        int w2_thread_id = lane % 4;
        float w2_accum[16];
        unsigned int w2_a_frag[4];
        unsigned int w2_b_frag[2];
        unsigned int w2_sfa_word[1];
        unsigned int w2_sfb_word[1];
        int w2_tile_count_2 = total_m_tasks[0] * 224;
        #pragma unroll 1
        for (int w2_tile_2 = bid - 1; w2_tile_2 < w2_tile_count_2; w2_tile_2 += worker_ctas) {
            w2_accum[0] = 0.0f;
            w2_accum[1] = 0.0f;
            w2_accum[2] = 0.0f;
            w2_accum[3] = 0.0f;
            w2_accum[4] = 0.0f;
            w2_accum[5] = 0.0f;
            w2_accum[6] = 0.0f;
            w2_accum[7] = 0.0f;
            w2_accum[8] = 0.0f;
            w2_accum[9] = 0.0f;
            w2_accum[10] = 0.0f;
            w2_accum[11] = 0.0f;
            w2_accum[12] = 0.0f;
            w2_accum[13] = 0.0f;
            w2_accum[14] = 0.0f;
            w2_accum[15] = 0.0f;
            int w2_task_2 = w2_tile_2 / 224;
            int w2_n_block_2 = w2_tile_2 - w2_task_2 * 224;
            int w2_pool_row_2 = task_pool_row[w2_task_2];
            #pragma unroll 1
            for (int w2_k_block_2 = 0; w2_k_block_2 < 24; w2_k_block_2++) {
                mbarrier_wait(w2_full_addr + (w2_math_stage) * 8, _phase_w2_full);
                asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                #pragma unroll
                for (int w2_k_step = 0; w2_k_step < 4; w2_k_step++) {
                    int w2_a_row = (lane & 7) + (lane >> 3 & 1) * 8 + w2_warp_m * 16;
                    int w2_a_col = (lane >> 4) * 16 + w2_k_step * 32;
                    int w2_a_addr = w1_smem_a_addr + w2_math_stage * 8192 + (unsigned int)(w2_a_row * 128) + (unsigned int)(w2_a_col ^ (w2_a_row & 7) << 4);
                    asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                        : "=r"(w2_a_frag[0]), "=r"(w2_a_frag[1]), "=r"(w2_a_frag[2]), "=r"(w2_a_frag[3])
                        : "r"(w2_a_addr)
                        : "memory");
                    int w2_sfa_row = w2_warp_m * 16 + w2_group_id + (w2_thread_id & 1) * 8;
                    asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&w2_sfa_word[0])) : "r"(w1_smem_sfa_addr + w2_math_stage * 256 + (unsigned int)(w2_sfa_row * 4)));
                    w2_sfa_word[0] = w2_sfa_word[0] >> (unsigned int)(w2_k_step * 8) & 255;
                    #pragma unroll
                    for (int w2_n_tile = 0; w2_n_tile < 4; w2_n_tile++) {
                        int w2_global_n_tile = w2_n_tile;
                        int w2_b_row = (lane & 7) + w2_global_n_tile * 8;
                        int w2_b_col = (lane >> 3 & 1) * 16 + w2_k_step * 32;
                        int w2_b_addr = w1_smem_b_addr + w2_math_stage * 4096 + (unsigned int)(w2_b_row * 128) + (unsigned int)(w2_b_col ^ (w2_b_row & 7) << 4);
                        asm volatile("ldmatrix.sync.aligned.shared::cta.m8n16.x2.b8x16.b4x16_p64 {%0, %1}, [%2];\n"
                            : "=r"(w2_b_frag[0]), "=r"(w2_b_frag[1])
                            : "r"(w2_b_addr)
                            : "memory");
                        int w2_sfb_row = w2_global_n_tile * 8 + w2_group_id;
                        asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&w2_sfb_word[0])) : "r"(w1_smem_sfb_addr + w2_math_stage * 128 + (unsigned int)(w2_sfb_row * 4)));
                        w2_sfb_word[0] = w2_sfb_word[0] >> (unsigned int)(w2_k_step * 8) & 255;
                        asm volatile("mma.sync.aligned.kind::mxf8f6f4.block_scale.scale_vec::1X.m16n8k32.row.col.f32.e4m3.e2m1.f32.ue8m0 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3}, {%10}, {%11, %12}, {%13}, {%14, %15};\n"
                            : "+f"((w2_accum + w2_n_tile * 4)[0]), "+f"((w2_accum + w2_n_tile * 4)[1]), "+f"((w2_accum + w2_n_tile * 4)[2]), "+f"((w2_accum + w2_n_tile * 4)[3])
                            : "r"(w2_a_frag[0]), "r"(w2_a_frag[1]), "r"(w2_a_frag[2]), "r"(w2_a_frag[3]), "r"(((uint32_t)(w2_b_frag[0]) << 2)), "r"(((uint32_t)(w2_b_frag[1]) << 2)), "r"(w2_sfa_word[0]), "h"(((uint16_t)0)), "h"(((uint16_t)0)), "r"(w2_sfb_word[0]), "h"(((uint16_t)0)), "h"(((uint16_t)0)));
                    }
                }
                if (elect_sync()) {
                    mbarrier_arrive(w2_empty_addr + (w2_math_stage) * 8);
                }
                w2_math_stage += 1;
                if (w2_math_stage == 2) { w2_math_stage = 0; _phase_w2_empty ^= 1; _phase_w2_full ^= 1; }
            }
            #pragma unroll
            for (int w2_n_tile_2 = 0; w2_n_tile_2 < 4; w2_n_tile_2++) {
                int w2_output_n_tile = w2_n_tile_2;
                int w2_output_col = w2_n_block_2 * 32 + w2_output_n_tile * 8 + w2_thread_id * 2;
                int w2_output_row_0 = w2_pool_row_2 + w2_warp_m * 16 + w2_group_id;
                int w2_output_row_1 = w2_output_row_0 + 8;
                int w2_acc_base = w2_n_tile_2 * 4;
                W2_D[w2_output_row_0 * 7168 + w2_output_col] = w2_accum[w2_acc_base];
                W2_D[w2_output_row_0 * 7168 + w2_output_col + 1] = w2_accum[w2_acc_base + 1];
                W2_D[w2_output_row_1 * 7168 + w2_output_col] = w2_accum[w2_acc_base + 2];
                W2_D[w2_output_row_1 * 7168 + w2_output_col + 1] = w2_accum[w2_acc_base + 3];
            }
            __threadfence();
            __syncwarp();
            if (elect_sync()) {
                int _atomic_old_4 = atomicAdd(&w2_warp_done[w2_tile_2], 1);
                int w2_previous = _atomic_old_4;
                if (w2_previous == 3) {
                    atomicAdd(&w2_tiles_completed[0], 1);
                } else if (w2_previous >= 4) {
                    atomicMax(&protocol_error[0], 1);
                }
            }
        }
    }
    cooperative_groups::this_grid().sync();
    if (bid == 0 && tid == 0) {
        if (w2_tiles_completed[0] != total_m_tasks[0] * 224) {
            atomicMax(&protocol_error[0], 1);
        }
    }
    int result_element_count = total_padded_rows[0] * 7168;
    #pragma unroll 1
    for (int result_element = bid * 256 + tid; result_element < result_element_count; result_element += num_bids * 256) {
        int result_row = result_element / 7168;
        int result_column = result_element - result_row * 7168;
        int result_source = meta_source_rank[result_row];
        if (result_source != -1) {
            if (result_source < 0 || result_source >= world_size) {
                atomicMax(&protocol_error[0], 1);
            } else {
                int result_index = meta_result_index[result_row];
                int result_count = source_route_counts[result_source];
                if (result_index < 0 || result_index >= result_count) {
                    atomicMax(&protocol_error[0], 1);
                } else {
                    unsigned long long result_out_index = (unsigned long long)(result_source * 2 + slot) * 88080384 + (unsigned long long)result_index * 7168 + (unsigned long long)result_column;
                    *(reinterpret_cast<__nv_bfloat16*>(reinterpret_cast<__nv_bfloat16*>(result_out) + result_out_index) + (0)) = __float2bfloat16_rn((float)W2_D[result_element]);
                }
            }
        }
    }
    __threadfence_system();
    cooperative_groups::this_grid().sync();
    if (bid == 0) {
        #pragma unroll 1
        for (int result_source_2 = 0; result_source_2 < world_size; result_source_2++) {
            if (warp == 0) {
                if (elect_sync()) {
                    int result_routes = source_route_counts[result_source_2];
                    int _max_9 = ((result_routes) > (0) ? (result_routes) : (0));
                    int _min_5 = ((_max_9) < (12288) ? (_max_9) : (12288));
                    int safe_result_routes = _min_5;
                    int _max_10 = ((safe_result_routes) > (1) ? (safe_result_routes) : (1));
                    int result_send_routes = _max_10;
                    int result_bytes = result_send_routes * 7168 * 2;
                    unsigned long long local_result_byte = (unsigned long long)(result_source_2 * 2 + slot) * 176160768;
                    unsigned long long remote_result_byte = (unsigned long long)(rank * 2 + slot) * 176160768;
                    // gin_put_signal_add: strong remote completion on context 0
                    {
                        ncclGin __gin{*(gin_dev_comm), (int)(0)};
                        __gin.put(ncclTeamWorld(*(gin_dev_comm)), (int)(result_source_2), result_inbox_window, (size_t)(remote_result_byte), result_out_window, (size_t)(local_result_byte), (size_t)(result_bytes),
                            ncclGin_StrongSignalAdd{(ncclGinSignal_t)(8 + rank), (uint64_t)(1)}, ncclGin_None{}, ncclCoopThread());
                    }
                }
            }
        }
        #pragma unroll 1
        for (int result_owner = 0; result_owner < world_size; result_owner++) {
            if (warp == 0) {
                if (elect_sync()) {
                    // gin_wait_signal: acquire, rolling 64-bit comparison
                    {
                        ncclGin __gin{*(gin_dev_comm), (int)(0)};
                        __gin.waitSignal(ncclCoopThread(), (ncclGinSignal_t)(8 + result_owner), (uint64_t)(result_signal_base_scratch[result_owner] + 1), 64, cuda::memory_order_acquire);
                    }
                }
            }
        }
    }
    cooperative_groups::this_grid().sync();
    int final_element_count = active_rows * 7168;
    #pragma unroll 1
    for (int final_element = bid * 256 + tid; final_element < final_element_count; final_element += num_bids * 256) {
        int final_token = final_element / 7168;
        int final_column = final_element - final_token * 7168;
        float combined = 0.0f;
        #pragma unroll
        for (int final_slot = 0; final_slot < 6; final_slot++) {
            int final_pair = final_token * 6 + final_slot;
            int final_expert = topk_idx_i32[final_pair * 2];
            int final_expert_hi = topk_idx_i32[final_pair * 2 + 1];
            int final_masked = (int)(final_expert == -1 && final_expert_hi == -1);
            int final_valid = (int)(final_expert >= 0 && final_expert < world_size * 48 && final_expert_hi == 0);
            if (final_valid == 0 && final_masked == 0) {
                atomicMax(&protocol_error[0], 1);
            }
            if (final_valid != 0) {
                int final_owner = final_expert / 48;
                int final_result_index = route_result_index[final_pair];
                int final_owner_count = owner_route_counts[final_owner];
                if (final_result_index < 0 || final_result_index >= final_owner_count) {
                    atomicMax(&protocol_error[0], 1);
                } else {
                    unsigned long long final_inbox_index = (unsigned long long)(final_owner * 2 + slot) * 88080384 + (unsigned long long)final_result_index * 7168 + (unsigned long long)final_column;
                    combined = combined + (float)reinterpret_cast<const __nv_bfloat16*>(reinterpret_cast<__nv_bfloat16*>(result_inbox) + final_inbox_index)[0];
                }
            }
        }
        final_output[final_element] = combined;
    }
    __threadfence_system();
    cooperative_groups::this_grid().sync();
    if (bid == 0) {
        #pragma unroll 1
        for (int ack_owner = 0; ack_owner < world_size; ack_owner++) {
            if (warp == 0) {
                if (elect_sync()) {
                    // gin_put_signal_add: strong remote completion on context 0
                    {
                        ncclGin __gin{*(gin_dev_comm), (int)(0)};
                        __gin.put(ncclTeamWorld(*(gin_dev_comm)), (int)(ack_owner), ack_inbox_window, (size_t)(rank), ack_out_window, (size_t)(ack_owner), (size_t)(1),
                            ncclGin_StrongSignalAdd{(ncclGinSignal_t)(16 + rank), (uint64_t)(1)}, ncclGin_None{}, ncclCoopThread());
                    }
                }
            }
        }
        #pragma unroll 1
        for (int ack_source = 0; ack_source < world_size; ack_source++) {
            if (warp == 0) {
                if (elect_sync()) {
                    // gin_wait_signal: acquire, rolling 64-bit comparison
                    {
                        ncclGin __gin{*(gin_dev_comm), (int)(0)};
                        __gin.waitSignal(ncclCoopThread(), (ncclGinSignal_t)(16 + ack_source), (uint64_t)(ack_signal_base_scratch[ack_source] + 1), 64, cuda::memory_order_acquire);
                    }
                }
            }
        }
    }
    cooperative_groups::this_grid().sync();

    // Cleanup
    __syncthreads();
}

} // extern "C"
