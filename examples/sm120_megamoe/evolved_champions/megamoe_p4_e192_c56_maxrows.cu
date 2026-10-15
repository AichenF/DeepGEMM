#include <nccl_device.h>

struct __align__(128) LoomTensorMap { uint64_t opaque[16]; };
template <int N>
struct __align__(128) LoomTensorMapPack { LoomTensorMap maps[N]; };

#include <cuda_bf16.h>

__device__ __forceinline__ int make_warp_uniform(int x) {
    int result;
    asm volatile("shfl.sync.idx.b32 %0, %1, 0, 0x1F, 0xFFFFFFFF;"
                 : "=r"(result) : "r"(x));
    return result;
}

#define LOOM_INF CUDART_INF_F
#define NUM_W1_PIPE_STAGES 2
#define NUM_W2_PIPE_STAGES 2
#define SMEM_CLAIMED_RECORDS_OFF 0
#define SMEM_CLAIMED_RECORDS_STAGE_BYTES 144
#define SMEM_CLAIMED_RECORDS_STRIDE 144
#define SMEM_WARP_DST_ROW_OFF 84480
#define SMEM_WARP_DST_ROW_STAGE_BYTES 36
#define SMEM_WARP_DST_ROW_STRIDE 36
#define SMEM_D_STAGE_OFF 84480
#define SMEM_D_STAGE_STAGE_BYTES 16384
#define SMEM_D_STAGE_STRIDE 16384
#define SMEM_SERVICE_READY_CHUNKS_OFF 0
#define SMEM_SERVICE_READY_CHUNKS_STAGE_BYTES 256
#define SMEM_SERVICE_READY_CHUNKS_STRIDE 256
#define SMEM_W1_SMEM_A_OFF 0
#define SMEM_W1_SMEM_A_STAGE_BYTES 8192
#define SMEM_W1_SMEM_A_STRIDE 8192
#define SMEM_W1_SMEM_B_OFF 16384
#define SMEM_W1_SMEM_B_STAGE_BYTES 32768
#define SMEM_W1_SMEM_B_STRIDE 32768
#define SMEM_W1_SMEM_SFA_OFF 81920
#define SMEM_W1_SMEM_SFA_STAGE_BYTES 256
#define SMEM_W1_SMEM_SFA_STRIDE 256
#define SMEM_W1_SMEM_SFB_OFF 82432
#define SMEM_W1_SMEM_SFB_STAGE_BYTES 1024
#define SMEM_W1_SMEM_SFB_STRIDE 1024
#define SMEM_TOTAL 100992
#define THREADS 640

#include <math_constants.h>
#include <cooperative_groups.h>

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
        :: "r"(mbar_addr), "r"(count) : "memory");
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

// CTA-local pipelines have short, resident producer/consumer edges.  Omitting
// suspendTimeHint keeps a miss on the lightweight TRYWAIT retry path; the
// explicit loop still makes this helper blocking until acquire succeeds.
__device__ __forceinline__ void mbarrier_wait(int mbar_addr, int phase) {
    asm volatile(
        "{\n\t"
        ".reg .pred P1;\n\t"
        "LAB_WAIT:\n\t"
        "mbarrier.try_wait.parity.acquire.cta.shared::cta.b64"
        " P1, [%0], %1;\n\t"
        "@P1 bra.uni DONE;\n\t"
        "bra.uni LAB_WAIT;\n\t"
        "DONE:\n\t"
        "}\n"
        :: "r"(mbar_addr), "r"(phase) : "memory");
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


__device__ __forceinline__ void tma_store_2d(
    const void *tmap, int x, int y, unsigned smem_addr) {
    asm volatile(
        "cp.async.bulk.tensor.2d.global.shared::cta.tile.bulk_group"
        " [%0, {%1, %2}], [%3];"
        :: "l"(tmap), "r"(x), "r"(y), "r"(smem_addr) : "memory");
}

extern "C" {

__global__ __launch_bounds__(640) void
kernel_deepgemm_sm120_megamoe_dispatch(LoomTensorMap const* W1_A, LoomTensorMap const* W1_B, LoomTensorMap const* W1_SFA, LoomTensorMap const* W1_SFB, LoomTensorMap const* W1_D, LoomTensorMap const* W2_A, LoomTensorMap const* W2_B, LoomTensorMap const* W2_SFA, LoomTensorMap const* W2_SFB, LoomTensorMap const* W2_D, uint8_t* __restrict__ intermediate_fp8, uint8_t* __restrict__ intermediate_sfa_u8, int* __restrict__ requant_groups_done, int* __restrict__ w2_warp_done, int* __restrict__ w2_tiles_completed, int* __restrict__ topk_idx_i32, float* __restrict__ topk_weights, int* __restrict__ x_fp8_i32, int* __restrict__ x_sf_i32, int* __restrict__ owner_record_counts, int* __restrict__ owner_route_counts, int* __restrict__ route_result_index, int* __restrict__ protocol_error, unsigned int* __restrict__ w2_task_counter, unsigned int* __restrict__ scatter_source_counter, int* __restrict__ c56_claim_cursor, int* __restrict__ c56_tile_mailbox, int* __restrict__ result_chunk_total, int* __restrict__ result_chunk_tally, unsigned long long* __restrict__ signal_base_scratch, unsigned long long* __restrict__ result_signal_base_scratch, unsigned long long* __restrict__ ack_signal_base_scratch, __nv_bfloat16* __restrict__ final_output, unsigned int* __restrict__ pool_fp8_u32, unsigned int* __restrict__ pool_sf_u32, float* __restrict__ routing_weight_pool, int* __restrict__ meta_source_rank, int* __restrict__ meta_token, int* __restrict__ meta_slot, int* __restrict__ meta_result_index, int* __restrict__ expert_counts, int* __restrict__ owner_expert_route_counts, int* __restrict__ source_route_sum, int* __restrict__ source_expert_counts, int* __restrict__ expert_source_base, int* __restrict__ expert_source_offsets, int* __restrict__ source_expert_prefix, int* __restrict__ task_max_source, int* __restrict__ source_record_counts, int* __restrict__ source_route_counts, int* __restrict__ source_active_rows, int* __restrict__ expert_row_offsets, int* __restrict__ expert_scatter_offsets, int* __restrict__ task_expert, int* __restrict__ task_source_rank, int* __restrict__ task_owner_rank, int* __restrict__ task_local_expert, int* __restrict__ task_pool_row, int* __restrict__ task_m_local, int* __restrict__ task_valid_m, int* __restrict__ total_valid_routes, int* __restrict__ total_padded_rows, int* __restrict__ total_m_tasks, int* __restrict__ histogram_done, int* __restrict__ prefix_done, int* __restrict__ w1_warp_done, int* __restrict__ w1_tiles_completed, int rank, int world_size, int active_rows, unsigned int epoch, ncclDevComm const* __restrict__ gin_dev_comm, uint8_t* __restrict__ dispatch_header_out, ncclWindow_t dispatch_header_out_window, uint8_t* __restrict__ dispatch_payload_out, ncclWindow_t dispatch_payload_out_window, uint8_t* __restrict__ dispatch_header_inbox, ncclWindow_t dispatch_header_inbox_window, uint8_t* __restrict__ dispatch_payload_inbox, ncclWindow_t dispatch_payload_inbox_window, uint8_t* __restrict__ result_out, ncclWindow_t result_out_window, uint8_t* __restrict__ result_inbox, ncclWindow_t result_inbox_window, uint8_t* __restrict__ ack_out, ncclWindow_t ack_out_window, uint8_t* __restrict__ ack_inbox, ncclWindow_t ack_inbox_window)
{
    const int tid = threadIdx.x;
    const int warp = make_warp_uniform(tid / 32);
    const int lane = tid % 32;

    extern __shared__ __align__(1024) char smem_raw[];
    int smem;
    smem = (int)(unsigned long long)__cvta_generic_to_shared(smem_raw);
    const int mbar_base = smem + 100864;
    #define w1_full_addr (mbar_base + 0)
    #define w1_empty_addr (mbar_base + 16)
    #define w2_full_addr (mbar_base + 32)
    #define w2_empty_addr (mbar_base + 48)

    const int bid = blockIdx.x;
    const int num_bids = gridDim.x;
    if (tid == 0) {
        asm volatile("fence.proxy.tensormap::generic.acquire.sys [%0], 128;" :: "l"((uint64_t)(W1_A)) : "memory");
        asm volatile("fence.proxy.tensormap::generic.acquire.sys [%0], 128;" :: "l"((uint64_t)(W1_B)) : "memory");
        asm volatile("fence.proxy.tensormap::generic.acquire.sys [%0], 128;" :: "l"((uint64_t)(W1_SFA)) : "memory");
        asm volatile("fence.proxy.tensormap::generic.acquire.sys [%0], 128;" :: "l"((uint64_t)(W1_SFB)) : "memory");
        asm volatile("fence.proxy.tensormap::generic.acquire.sys [%0], 128;" :: "l"((uint64_t)(W1_D)) : "memory");
        asm volatile("fence.proxy.tensormap::generic.acquire.sys [%0], 128;" :: "l"((uint64_t)(W2_A)) : "memory");
        asm volatile("fence.proxy.tensormap::generic.acquire.sys [%0], 128;" :: "l"((uint64_t)(W2_B)) : "memory");
        asm volatile("fence.proxy.tensormap::generic.acquire.sys [%0], 128;" :: "l"((uint64_t)(W2_SFA)) : "memory");
        asm volatile("fence.proxy.tensormap::generic.acquire.sys [%0], 128;" :: "l"((uint64_t)(W2_SFB)) : "memory");
        asm volatile("fence.proxy.tensormap::generic.acquire.sys [%0], 128;" :: "l"((uint64_t)(W2_D)) : "memory");
    }
    __syncthreads();


    // Kernel setup ops
    int* claimed_records = reinterpret_cast<int*>(smem_raw + 0);
    const int claimed_records_addr = smem + 0;
    int* warp_dst_row = reinterpret_cast<int*>(smem_raw + 84480);
    const int warp_dst_row_addr = smem + 84480;
    __nv_bfloat16* d_stage = reinterpret_cast<__nv_bfloat16*>(smem_raw + 84480);
    const int d_stage_addr = smem + 84480;
    int* service_ready_chunks = reinterpret_cast<int*>(smem_raw + 0);
    const int service_ready_chunks_addr = smem + 0;
    uint8_t* w1_smem_a = reinterpret_cast<uint8_t*>(smem_raw + 0);
    const int w1_smem_a_addr = smem + 0;
    uint8_t* w1_smem_b = reinterpret_cast<uint8_t*>(smem_raw + 16384);
    const int w1_smem_b_addr = smem + 16384;
    unsigned int* w1_smem_sfa = reinterpret_cast<unsigned int*>(smem_raw + 81920);
    const int w1_smem_sfa_addr = smem + 81920;
    unsigned int* w1_smem_sfb = reinterpret_cast<unsigned int*>(smem_raw + 82432);
    const int w1_smem_sfb_addr = smem + 82432;

    // Mbarrier init (4 groups, 8 barriers)
    // Mbarriers at smem_raw[100864..100928)

    if (warp == 0) {
        uint32_t leader = elect_sync();
        if (leader) {
            // --- pipeline 'w1_pipe' ---
            // w1_full: 2 barriers, init_count=1
            mbarrier_init(smem + 100864, 1);
            mbarrier_init(smem + 100872, 1);
            // w1_empty: 2 barriers, init_count=16
            mbarrier_init(smem + 100880, 16);
            mbarrier_init(smem + 100888, 16);
            // --- pipeline 'w2_pipe' ---
            // w2_full: 2 barriers, init_count=1
            mbarrier_init(smem + 100896, 1);
            mbarrier_init(smem + 100904, 1);
            // w2_empty: 2 barriers, init_count=16
            mbarrier_init(smem + 100912, 16);
            mbarrier_init(smem + 100920, 16);
            asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
        }
    }

    __syncthreads();

    // === Task calls (dependency order) ===
    int reset_tid = bid * 640 + tid;
    int reset_threads = num_bids * 640;
    int _max_0 = ((world_size * active_rows * 6 + 3024) > (0) ? (world_size * active_rows * 6 + 3024) : (0));
    int _min_0 = ((_max_0) < (199632) ? (_max_0) : (199632));
    int reset_rows_bound = _min_0;
    int _max_1 = (((world_size * active_rows * 6 + 64 - 1) / 64 + 48) > (0) ? ((world_size * active_rows * 6 + 64 - 1) / 64 + 48) : (0));
    int _min_1 = ((_max_1) < (3120) ? (_max_1) : (3120));
    int reset_tasks_bound = _min_1;
    #pragma unroll 1
    for (int reset_peer = reset_tid; reset_peer < 4; reset_peer += reset_threads) {
        owner_record_counts[reset_peer] = 0;
        owner_route_counts[reset_peer] = 0;
        source_record_counts[reset_peer] = 0;
        source_route_counts[reset_peer] = 0;
        source_active_rows[reset_peer] = 0;
        source_route_sum[reset_peer] = 0;
    }
    #pragma unroll 1
    for (int reset_owner_expert = reset_tid; reset_owner_expert < 192; reset_owner_expert += reset_threads) {
        owner_expert_route_counts[reset_owner_expert] = 0;
        source_expert_counts[reset_owner_expert] = 0;
        expert_source_base[reset_owner_expert] = 0;
        expert_source_offsets[reset_owner_expert] = 0;
        source_expert_prefix[reset_owner_expert] = 0;
    }
    #pragma unroll 1
    for (int reset_scatter_source = reset_tid; reset_scatter_source < 4; reset_scatter_source += reset_threads) {
        {
            unsigned int* _gcr_p = reinterpret_cast<unsigned int*>(scatter_source_counter) + (reset_scatter_source);
            asm volatile("st.release.gpu.global.u32 [%0], %1;" : : "l"(_gcr_p), "r"(0u) : "memory");
        }
    }
    #pragma unroll 1
    for (int reset_route = reset_tid; reset_route < active_rows * 6; reset_route += reset_threads) {
        route_result_index[reset_route] = -1;
    }
    #pragma unroll 1
    for (int reset_pool_word = reset_tid; reset_pool_word < reset_rows_bound * 1792; reset_pool_word += reset_threads) {
        pool_fp8_u32[reset_pool_word] = 0;
    }
    #pragma unroll 1
    for (int reset_sf_pair = reset_tid; reset_sf_pair < 56 * reset_rows_bound; reset_sf_pair += reset_threads) {
        int reset_sf_word = reset_sf_pair / reset_rows_bound;
        int reset_sf_row = reset_sf_pair - reset_sf_word * reset_rows_bound;
        pool_sf_u32[reset_sf_word * 199632 + reset_sf_row] = 0;
    }
    #pragma unroll 1
    for (int reset_row = reset_tid; reset_row < reset_rows_bound; reset_row += reset_threads) {
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
    for (int reset_task = reset_tid; reset_task < reset_tasks_bound; reset_task += reset_threads) {
        task_expert[reset_task] = -1;
        task_source_rank[reset_task] = -1;
        task_owner_rank[reset_task] = -1;
        task_local_expert[reset_task] = -1;
        task_pool_row[reset_task] = -1;
        task_m_local[reset_task] = -1;
        task_valid_m[reset_task] = -1;
        task_max_source[reset_task] = -1;
    }
    #pragma unroll 1
    for (int reset_w1_tile = reset_tid; reset_w1_tile < reset_tasks_bound * 24; reset_w1_tile += reset_threads) {
        w1_warp_done[reset_w1_tile] = 0;
    }
    #pragma unroll 1
    for (int reset_w2_tile = reset_tid; reset_w2_tile < reset_tasks_bound * 28; reset_w2_tile += reset_threads) {
        w2_warp_done[reset_w2_tile] = 0;
    }
    #pragma unroll 1
    for (int reset_w2_task = reset_tid; reset_w2_task < reset_tasks_bound; reset_w2_task += reset_threads) {
        {
            unsigned int* _gcr_p = reinterpret_cast<unsigned int*>(w2_task_counter) + (reset_w2_task);
            asm volatile("st.release.gpu.global.u32 [%0], %1;" : : "l"(_gcr_p), "r"(0u) : "memory");
        }
    }
    #pragma unroll 1
    for (int reset_chunk = reset_tid; reset_chunk < 780; reset_chunk += reset_threads) {
        result_chunk_total[reset_chunk] = 0;
        result_chunk_tally[reset_chunk] = 0;
    }
    #pragma unroll 1
    for (int reset_c56_slot = reset_tid; reset_c56_slot < 16384; reset_c56_slot += reset_threads) {
        c56_tile_mailbox[reset_c56_slot] = 0;
    }
    if (reset_tid == 0) {
        c56_claim_cursor[0] = 0;
        c56_claim_cursor[1] = 0;
        protocol_error[0] = 0;
        total_valid_routes[0] = 0;
        total_padded_rows[0] = 0;
        total_m_tasks[0] = 0;
        histogram_done[0] = 0;
        prefix_done[0] = 0;
        w1_tiles_completed[0] = 0;
        requant_groups_done[0] = 0;
        w2_tiles_completed[0] = 0;
    }
    cooperative_groups::this_grid().sync();
    int slot = (int)(epoch & 1);
    int launch_valid = (int)(rank >= 0 && rank < world_size && world_size >= 1 && world_size <= 4 && active_rows >= 1 && active_rows <= 8192);
    if (launch_valid == 0) {
        if (warp == 0) {
            if (elect_sync()) {
                atomicMax(&protocol_error[0], 1);
            }
        }
    }
    if (bid == 0) {
        #pragma unroll 1
        for (int source = 0; source < 4; source++) {
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
        // gin_world_barrier: CTA/world rendezvous, no put drain
        {
            ncclGin __gin{*(gin_dev_comm), (int)(0)};
            ncclGinBarrierSession<ncclCoopCta> __bar{ncclCoopCta(), __gin, ncclTeamTagWorld(), (uint32_t)(0)};
            __bar.sync(ncclCoopCta(), cuda::memory_order_acquire, ncclGinFenceLevel::None);
        }
    }
    cooperative_groups::this_grid().sync();
    if (warp < 9) {
        int global_warp = bid * 9 + warp;
        int warps_per_grid = num_bids * 9;
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
                        if (claim >= 8192) {
                            atomicMax(&protocol_error[0], 1);
                            claim = -1;
                        }
                        int _atomic_old_1 = atomicAdd(&owner_route_counts[owner], route_count);
                        route_base = _atomic_old_1;
                        if (route_base + route_count > 49152) {
                            atomicMax(&protocol_error[0], 1);
                            claim = -1;
                        }
                    }
                    claimed_records[warp * 4 + owner] = claim;
                    if (claim >= 0) {
                        unsigned long long record_byte = (unsigned long long)(owner * 2 + slot) * 61865984 + (unsigned long long)claim * 7552;
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
                                atomicAdd(&owner_expert_route_counts[owner * 48 + (expert_2 - owner * 48)], 1);
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
                int record_idx = claimed_records[warp * 4 + owner];
                if (record_idx >= 0) {
                    unsigned long long record_byte_2 = (unsigned long long)(owner * 2 + slot) * 61865984 + (unsigned long long)record_idx * 7552;
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
            int local_header_word = (peer * 2 + slot) * 56;
            int local_header_byte = local_header_word * 4;
            int remote_header_byte = (rank * 2 + slot) * 224;
            if (warp == 0) {
                if (elect_sync()) {
                    int count = owner_record_counts[peer];
                    int peer_route_count = owner_route_counts[peer];
                    int _max_2 = ((count) > (0) ? (count) : (0));
                    int _min_2 = ((_max_2) < (8192) ? (_max_2) : (8192));
                    int safe_count = _min_2;
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
                    #pragma unroll 1
                    for (int header_expert = 0; header_expert < 48; header_expert++) {
                        *(reinterpret_cast<int*>(reinterpret_cast<int*>(dispatch_header_out) + (local_header_word + 8 + header_expert)) + (0)) = owner_expert_route_counts[peer * 48 + header_expert];
                    }
                    __threadfence_system();
                    // gin_put_signal_add: strong remote completion on context 0
                    {
                        ncclGin __gin{*(gin_dev_comm), (int)(0)};
                        __gin.put(ncclTeamWorld(*(gin_dev_comm)), (int)(peer), dispatch_header_inbox_window, (size_t)(remote_header_byte), dispatch_header_out_window, (size_t)(local_header_byte), (size_t)(224),
                            ncclGin_StrongSignalAdd{(ncclGinSignal_t)(rank), (uint64_t)(1)}, ncclGin_None{}, ncclCoopThread());
                    }
                }
            }
        }
        #pragma unroll 1
        for (int peer_2_i = 0; peer_2_i < world_size; peer_2_i++) {
            int peer_2 = peer_2_i + rank + 1;
            if (peer_2 >= world_size) {
                peer_2 = peer_2 - world_size;
            }
            unsigned long long local_payload_byte = (unsigned long long)(peer_2 * 2 + slot) * 61865984;
            unsigned long long remote_payload_byte = (unsigned long long)(rank * 2 + slot) * 61865984;
            if (warp == 0) {
                if (elect_sync()) {
                    int payload_count = owner_record_counts[peer_2];
                    int _max_3 = ((payload_count) > (0) ? (payload_count) : (0));
                    int _min_3 = ((_max_3) < (8192) ? (_max_3) : (8192));
                    int payload_safe_count = _min_3;
                    int _max_4 = ((payload_safe_count) > (1) ? (payload_safe_count) : (1));
                    int payload_send_count = _max_4;
                    int payload_bytes = payload_send_count * 7552;
                    // gin_put_signal_add: strong remote completion on context 0
                    {
                        ncclGin __gin{*(gin_dev_comm), (int)(0)};
                        __gin.put(ncclTeamWorld(*(gin_dev_comm)), (int)(peer_2), dispatch_payload_inbox_window, (size_t)(remote_payload_byte), dispatch_payload_out_window, (size_t)(local_payload_byte), (size_t)(payload_bytes),
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
                        __gin.waitSignal(ncclCoopThread(), (ncclGinSignal_t)(source_2), (uint64_t)(signal_base_scratch[source_2] + 1), 64, cuda::memory_order_acquire);
                    }
                }
            }
        }
    }
    int grid_tid = bid * 640 + tid;
    int grid_threads = num_bids * 640;
    #pragma unroll 1
    for (int source_3 = grid_tid; source_3 < world_size; source_3 += grid_threads) {
        int header_word = (source_3 * 2 + slot) * 56;
        int count_1 = reinterpret_cast<const int*>(reinterpret_cast<int*>(dispatch_header_inbox) + (header_word + 5))[0];
        int routes = reinterpret_cast<const int*>(reinterpret_cast<int*>(dispatch_header_inbox) + (header_word + 6))[0];
        int rows = reinterpret_cast<const int*>(reinterpret_cast<int*>(dispatch_header_inbox) + (header_word + 7))[0];
        int header_ok = (int)(reinterpret_cast<const int*>(reinterpret_cast<int*>(dispatch_header_inbox) + header_word)[0] == 1347571524 && reinterpret_cast<const int*>(reinterpret_cast<int*>(dispatch_header_inbox) + (header_word + 1))[0] == 1 && reinterpret_cast<const int*>(reinterpret_cast<int*>(dispatch_header_inbox) + (header_word + 2))[0] == (int)epoch && reinterpret_cast<const int*>(reinterpret_cast<int*>(dispatch_header_inbox) + (header_word + 3))[0] == source_3 && reinterpret_cast<const int*>(reinterpret_cast<int*>(dispatch_header_inbox) + (header_word + 4))[0] == rank && count_1 >= 0 && count_1 <= 8192 && routes >= 0 && routes <= 49152 && routes <= rows * 6 && rows >= 1 && rows <= 8192 && rows == active_rows);
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
    for (int hist_pair = grid_tid; hist_pair < 192; hist_pair += grid_threads) {
        int hist_source = hist_pair / 48;
        int hist_expert = hist_pair - hist_source * 48;
        if (hist_source < world_size) {
            int hist_header_word = (hist_source * 2 + slot) * 56;
            int hist_count = reinterpret_cast<const int*>(reinterpret_cast<int*>(dispatch_header_inbox) + (hist_header_word + 8 + hist_expert))[0];
            int hist_ok = (int)(hist_count >= 0 && hist_count <= source_route_counts[hist_source]);
            if (hist_ok == 0) {
                atomicMax(&protocol_error[0], 1);
            } else {
                source_expert_counts[hist_pair] = hist_count;
            }
            if (hist_ok != 0 && hist_count > 0) {
                atomicAdd(&expert_counts[hist_expert], hist_count);
                atomicAdd(&source_route_sum[hist_source], hist_count);
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
        #pragma unroll 1
        for (int audit_source = 0; audit_source < 4; audit_source++) {
            if (audit_source < world_size) {
                if (source_route_sum[audit_source] != source_route_counts[audit_source]) {
                    atomicMax(&protocol_error[0], 1);
                }
            }
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
            int es_run = 0;
            #pragma unroll 1
            for (int es_source = 0; es_source < 4; es_source++) {
                expert_source_base[expert_3 * 4 + es_source] = es_run;
                if (expert_3 == 0) {
                    source_expert_prefix[es_source] = 0;
                } else {
                    source_expert_prefix[expert_3 * 4 + es_source] = source_expert_prefix[(expert_3 - 1) * 4 + es_source] + source_expert_counts[es_source * 48 + (expert_3 - 1)];
                }
                if (es_source < world_size) {
                    es_run = es_run + source_expert_counts[es_source * 48 + expert_3];
                }
            }
            if (es_run != expert_count) {
                atomicMax(&protocol_error[0], 1);
            }
            #pragma unroll 1
            for (int m_local = 0; m_local < padded; m_local += 64) {
                if (task_idx < 3120) {
                    int _min_4 = ((m_local + 64) < (expert_count) ? (m_local + 64) : (expert_count));
                    int task_last_row = _min_4;
                    int task_ms = -1;
                    #pragma unroll 1
                    for (int ms_source = 0; ms_source < 4; ms_source++) {
                        if (ms_source < world_size) {
                            int ms_base = expert_source_base[expert_3 * 4 + ms_source];
                            int ms_count = source_expert_counts[ms_source * 48 + expert_3];
                            if (ms_count > 0 && ms_base < task_last_row && m_local < ms_base + ms_count) {
                                task_ms = ms_source;
                            }
                        }
                    }
                    task_max_source[task_idx] = task_ms;
                    task_expert[task_idx] = rank * 48 + expert_3;
                    task_source_rank[task_idx] = 0;
                    task_owner_rank[task_idx] = rank;
                    task_local_expert[task_idx] = expert_3;
                    task_pool_row[task_idx] = running + m_local;
                    task_m_local[task_idx] = m_local;
                    int _min_5 = ((64) < (expert_count - m_local) ? (64) : (expert_count - m_local));
                    task_valid_m[task_idx] = _min_5;
                } else {
                    atomicMax(&protocol_error[0], 1);
                }
                task_idx = task_idx + 1;
            }
            running = running + padded;
        }
        if (valid_routes > 196608 || running > 199632 || task_idx > 3120) {
            atomicMax(&protocol_error[0], 1);
        }
        total_valid_routes[0] = valid_routes;
        total_padded_rows[0] = running;
        total_m_tasks[0] = task_idx;
        __threadfence();
        prefix_done[0] = 1;
    }
    cooperative_groups::this_grid().sync();
    int scatter_source = bid % 4;
    int scatter_group_rank = bid / 4;
    int scatter_group_ctas = (num_bids - scatter_source + 4 - 1) / 4;
    if (scatter_source < world_size) {
        if (warp == 0) {
            if (elect_sync()) {
                // gin_wait_signal: acquire, rolling 64-bit comparison
                {
                    ncclGin __gin{*(gin_dev_comm), (int)(0)};
                    __gin.waitSignal(ncclCoopThread(), (ncclGinSignal_t)(scatter_source), (uint64_t)(signal_base_scratch[scatter_source] + 2), 64, cuda::memory_order_acquire);
                }
            }
        }
        __syncthreads();
        if (warp < 9) {
            int task_global_warp = scatter_group_rank * 9 + warp;
            int task_grid_warps = scatter_group_ctas * 9;
            int _max_5 = ((source_record_counts[scatter_source]) > (0) ? (source_record_counts[scatter_source]) : (0));
            int _min_6 = ((_max_5) < (8192) ? (_max_5) : (8192));
            int scatter_cand_count = _min_6 * 6;
            if (scatter_source >= world_size) {
                scatter_cand_count = 0;
            }
            #pragma unroll 1
            for (int candidate_2 = scatter_source * 49152 + task_global_warp; candidate_2 < scatter_source * 49152 + scatter_cand_count; candidate_2 += task_grid_warps) {
                int source_5 = scatter_source;
                int candidate_rem_2 = candidate_2 - source_5 * 49152;
                int record_2 = candidate_rem_2 / 6;
                int record_route_2 = candidate_rem_2 - record_2 * 6;
                if (source_5 < world_size && record_2 < source_record_counts[source_5]) {
                    unsigned long long scatter_record_word = (unsigned long long)(source_5 * 2 + slot) * 15466496 + (unsigned long long)record_2 * 1888;
                    int scatter_route_count = reinterpret_cast<const int*>(reinterpret_cast<int*>(dispatch_payload_inbox) + (scatter_record_word + 1))[0];
                    int scatter_route_base = reinterpret_cast<const int*>(reinterpret_cast<int*>(dispatch_payload_inbox) + (scatter_record_word + 22))[0];
                    int scatter_record_ok = (int)(scatter_route_count >= 1 && scatter_route_count <= 6 && scatter_route_base >= 0 && scatter_route_base + scatter_route_count <= source_route_counts[source_5] && scatter_route_base + scatter_route_count <= 49152 && reinterpret_cast<const int*>(reinterpret_cast<int*>(dispatch_payload_inbox) + (scatter_record_word + 20))[0] == source_5 && reinterpret_cast<const int*>(reinterpret_cast<int*>(dispatch_payload_inbox) + (scatter_record_word + 21))[0] == 1347571524);
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
                                int scatter_es = scatter_local_expert * 4 + source_5;
                                int _atomic_old_2 = atomicAdd(&expert_source_offsets[scatter_es], 1);
                                int scatter_claim = _atomic_old_2;
                                atomicAdd(&expert_scatter_offsets[scatter_local_expert], 1);
                                int dst_row = expert_row_offsets[scatter_local_expert] + expert_source_base[scatter_es] + scatter_claim;
                                if (scatter_claim < 0 || scatter_claim >= source_expert_counts[source_5 * 48 + scatter_local_expert] || dst_row < 0 || dst_row >= 199632) {
                                    atomicMax(&protocol_error[0], 1);
                                    dst_row = -1;
                                } else {
                                    meta_source_rank[dst_row] = source_5;
                                    meta_token[dst_row] = scatter_token;
                                    meta_slot[dst_row] = scatter_topk_slot;
                                    int scatter_new_index = source_expert_prefix[scatter_es] + scatter_claim;
                                    meta_result_index[dst_row] = scatter_new_index;
                                    *(reinterpret_cast<int*>(reinterpret_cast<int*>(result_out) + ((unsigned long long)(source_5 * 2 + slot) * 176209920 + 176160768 + (unsigned long long)scatter_result_index)) + (0)) = scatter_new_index;
                                    routing_weight_pool[dst_row] = scatter_route_weight;
                                    int _max_6 = ((source_route_counts[source_5]) > (0) ? (source_route_counts[source_5]) : (0));
                                    int _min_7 = ((_max_6) < (49152) ? (_max_6) : (49152));
                                    int c41_sc_r = _min_7;
                                    int _max_7 = (((c41_sc_r + 256 - 1) / 256 - 1) > (0) ? ((c41_sc_r + 256 - 1) / 256 - 1) : (0));
                                    int c41_sc_full = _max_7;
                                    int c41_sc_ts = c41_sc_full * 256;
                                    int c41_scatter_chunk = 0;
                                    if (scatter_new_index < c41_sc_ts) {
                                        c41_scatter_chunk = scatter_new_index / 256;
                                    } else {
                                        c41_scatter_chunk = c41_sc_full + (scatter_new_index - c41_sc_ts) / 64;
                                    }
                                    atomicAdd(&result_chunk_total[source_5 * 195 + c41_scatter_chunk], 1);
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
                                    pool_sf_u32[(unsigned long long)scale_word_2 * 199632 + (unsigned long long)scatter_dst_row] = (unsigned int)reinterpret_cast<const int*>(reinterpret_cast<int*>(dispatch_payload_inbox) + (src_scale_word + (unsigned long long)scale_word_2))[0];
                                }
                            }
                            __syncwarp();
                        } else if (lane == 0) {
                            atomicMax(&protocol_error[0], 1);
                        }
                    }
                }
                if (source_5 < world_size) {
                    __threadfence();
                    if (elect_sync()) {
                        {
                            unsigned int* _gc_p = reinterpret_cast<unsigned int*>(scatter_source_counter) + (source_5);
                            unsigned int _gc_old;
                            asm volatile("atom.release.gpu.global.add.u32 %0, [%1], 1;" : "=r"(_gc_old) : "l"(_gc_p) : "memory");
                        }
                    }
                }
            }
        }
    }
    __syncthreads();
    if (bid == 0 && tid == 0) {
        int expected_tasks = 0;
        #pragma unroll 1
        for (int expert_4 = 0; expert_4 < 48; expert_4++) {
            int expert_count_2 = expert_counts[expert_4];
            expected_tasks = expected_tasks + (expert_count_2 + 64 - 1) / 64;
        }
        if (expected_tasks != total_m_tasks[0] || prefix_done[0] != 1) {
            atomicMax(&protocol_error[0], 1);
        }
    }
    if (warp == 16) {
        if (elect_sync()) {
            asm volatile("prefetch.tensormap [%0];" :: "l"((uint64_t)(W1_A)) : "memory");
            asm volatile("prefetch.tensormap [%0];" :: "l"((uint64_t)(W1_B)) : "memory");
            asm volatile("prefetch.tensormap [%0];" :: "l"((uint64_t)(W1_SFA)) : "memory");
            asm volatile("prefetch.tensormap [%0];" :: "l"((uint64_t)(W1_SFB)) : "memory");
        }
    }
    unsigned int _phase_w1_empty = 1;
    if (warp == 16) {
        unsigned int w1_load_stage = 0;
        int w1_tile_count = total_m_tasks[0] * 24;
        int c56_w1_seq = 0;
        int c56_w1_mb = bid * 32;
        #pragma unroll 1
        for (int c56_w1_iter = 0; c56_w1_iter < 74880; c56_w1_iter++) {
            int w1_tile = 0;
            if (elect_sync()) {
                int _atomic_old_3 = atomicAdd(&c56_claim_cursor[0], 1);
                w1_tile = _atomic_old_3;
            }
            int _shfl_0 = __shfl_sync(0xFFFFFFFF, w1_tile, 0);
            w1_tile = _shfl_0;
            if (w1_tile >= w1_tile_count) {
                if (elect_sync()) {
                    atomicMax(&c56_tile_mailbox[c56_w1_mb + (c56_w1_seq & 3)], 2147483647);
                }
                break;
            }
            if (elect_sync()) {
                atomicMax(&c56_tile_mailbox[c56_w1_mb + (c56_w1_seq & 3)], w1_tile + 1);
            }
            c56_w1_seq += 1;
            int w1_task = w1_tile / 24;
            int w1_n_block = w1_tile - w1_task * 24;
            int w1_pool_row = task_pool_row[w1_task];
            int w1_local_expert = task_local_expert[w1_task];
            int w1_gate_ms = task_max_source[w1_task];
            if (warp == 16) {
                if (elect_sync()) {
                    #pragma unroll 1
                    for (int w1_gate_s = 0; w1_gate_s < 4; w1_gate_s++) {
                        if (w1_gate_ms >= w1_gate_s) {
                            int _max_8 = ((source_record_counts[w1_gate_s]) > (0) ? (source_record_counts[w1_gate_s]) : (0));
                            int _min_8 = ((_max_8) < (8192) ? (_max_8) : (8192));
                            {
                                unsigned int* _gca_p = reinterpret_cast<unsigned int*>(scatter_source_counter) + (w1_gate_s);
                                while (true) {
                                    unsigned int _gca_v;
                                    asm volatile("ld.acquire.gpu.global.u32 %0, [%1];" : "=r"(_gca_v) : "l"(_gca_p));
                                    if (_gca_v >= (unsigned int)(_min_8 * 6)) break;
                                }
                            }
                        }
                    }
                }
            }
            #pragma unroll 1
            for (int w1_k_block = 0; w1_k_block < 56; w1_k_block++) {
                mbarrier_wait(w1_empty_addr + (w1_load_stage) * 8, _phase_w1_empty);
                if (warp == 16) {
                    if (elect_sync()) {
                        tma_2d_gmem2smem(w1_smem_sfa_addr + w1_load_stage * 256, W1_SFA, w1_pool_row, w1_k_block, w1_full_addr + (w1_load_stage) * 8);
                        tma_2d_gmem2smem(w1_smem_sfb_addr + w1_load_stage * 1024, W1_SFB, w1_n_block * 256, w1_local_expert * 56 + w1_k_block, w1_full_addr + (w1_load_stage) * 8);
                        tma_2d_gmem2smem(w1_smem_a_addr + w1_load_stage * 8192, W1_A, w1_k_block * 128, w1_pool_row, w1_full_addr + (w1_load_stage) * 8);
                        tma_2d_gmem2smem(w1_smem_b_addr + w1_load_stage * 32768, W1_B, w1_k_block * 128, w1_local_expert * 6144 + w1_n_block * 256, w1_full_addr + (w1_load_stage) * 8);
                        mbarrier_arrive_expect_tx(w1_full_addr + (w1_load_stage) * 8, 25856);
                    }
                }
                w1_load_stage += 1;
                if (w1_load_stage == 2) { w1_load_stage = 0; _phase_w1_empty ^= 1; }
            }
        }
    }
    unsigned int _phase_w1_full = 0;
    if (warp < 16) {
        unsigned int w1_math_stage = 0;
        int w1_warp_m = warp / 4;
        int w1_warp_n = warp % 4;
        int w1_group_id = lane / 4;
        int w1_thread_id = lane % 4;
        float w1_accum[32];
        unsigned int w1_a_frag_0[4];
        unsigned int w1_b_frag_0[2];
        unsigned int w1_sfa_word_0[1];
        unsigned int w1_sfb_word_0[1];
        float w1_routed_a[8];
        float w1_routed_b[8];
        int w1_tile_count_2 = total_m_tasks[0] * 24;
        int c56m_w1_last = 0;
        int c56m_w1_seq = 0;
        int c56m_w1_mb = bid * 32;
        #pragma unroll 1
        for (int c56m_w1_iter = 0; c56m_w1_iter < 74880; c56m_w1_iter++) {
            int c56m_w1_val = 0;
            if (elect_sync()) {
                #pragma unroll 1
                for (int c56m_w1_spin = 0; c56m_w1_spin < 1073741824; c56m_w1_spin++) {
                    int _atomic_old_4 = atomicAdd(&c56_tile_mailbox[c56m_w1_mb + (c56m_w1_seq & 3)], 0);
                    int c56m_w1_probe = _atomic_old_4;
                    if (c56m_w1_probe > c56m_w1_last) {
                        c56m_w1_val = c56m_w1_probe;
                        break;
                    }
                }
            }
            int _shfl_1 = __shfl_sync(0xFFFFFFFF, c56m_w1_val, 0);
            c56m_w1_val = _shfl_1;
            if (c56m_w1_val == 2147483647) {
                break;
            }
            if (c56m_w1_val <= c56m_w1_last) {
                if (elect_sync()) {
                    atomicMax(&protocol_error[0], 1);
                }
                break;
            }
            c56m_w1_last = c56m_w1_val;
            c56m_w1_seq += 1;
            int w1_tile_2 = c56m_w1_val - 1;
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
            w1_accum[16] = 0.0f;
            w1_accum[17] = 0.0f;
            w1_accum[18] = 0.0f;
            w1_accum[19] = 0.0f;
            w1_accum[20] = 0.0f;
            w1_accum[21] = 0.0f;
            w1_accum[22] = 0.0f;
            w1_accum[23] = 0.0f;
            w1_accum[24] = 0.0f;
            w1_accum[25] = 0.0f;
            w1_accum[26] = 0.0f;
            w1_accum[27] = 0.0f;
            w1_accum[28] = 0.0f;
            w1_accum[29] = 0.0f;
            w1_accum[30] = 0.0f;
            w1_accum[31] = 0.0f;
            int w1_task_2 = w1_tile_2 / 24;
            int w1_n_block_2 = w1_tile_2 - w1_task_2 * 24;
            int w1_pool_row_2 = task_pool_row[w1_task_2];
            int w1_gate_ms_2 = task_max_source[w1_task_2];
            if (elect_sync()) {
                #pragma unroll 1
                for (int w1_gate_s_2 = 0; w1_gate_s_2 < 4; w1_gate_s_2++) {
                    if (w1_gate_ms_2 >= w1_gate_s_2) {
                        int _max_9 = ((source_record_counts[w1_gate_s_2]) > (0) ? (source_record_counts[w1_gate_s_2]) : (0));
                        int _min_9 = ((_max_9) < (8192) ? (_max_9) : (8192));
                        {
                            unsigned int* _gca_p = reinterpret_cast<unsigned int*>(scatter_source_counter) + (w1_gate_s_2);
                            while (true) {
                                unsigned int _gca_v;
                                asm volatile("ld.acquire.gpu.global.u32 %0, [%1];" : "=r"(_gca_v) : "l"(_gca_p));
                                if (_gca_v >= (unsigned int)(_min_9 * 6)) break;
                            }
                        }
                    }
                }
            }
            __syncwarp();
            #pragma unroll 1
            for (int w1_k_block_2 = 0; w1_k_block_2 < 56; w1_k_block_2++) {
                mbarrier_wait(w1_full_addr + (w1_math_stage) * 8, _phase_w1_full);
                asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                #pragma unroll
                for (int w1_k_step = 0; w1_k_step < 4; w1_k_step++) {
                    asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                        : "=r"(w1_a_frag_0[0]), "=r"(w1_a_frag_0[1]), "=r"(w1_a_frag_0[2]), "=r"(w1_a_frag_0[3])
                        : "r"(w1_smem_a_addr + w1_math_stage * 8192 + (unsigned int)(((lane & 7) + (lane >> 3 & 1) * 8 + w1_warp_m * 16) * 128) + (unsigned int)((lane >> 4) * 16 + w1_k_step * 32 ^ ((lane & 7) + (lane >> 3 & 1) * 8 + w1_warp_m * 16 & 7) << 4))
                        : "memory");
                    asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&w1_sfa_word_0[0])) : "r"(w1_smem_sfa_addr + w1_math_stage * 256 + (unsigned int)((w1_warp_m * 16 + w1_group_id + (w1_thread_id & 1) * 8) * 4)));
                    w1_sfa_word_0[0] = w1_sfa_word_0[0] >> (unsigned int)(w1_k_step * 8) & 255;
                    #pragma unroll
                    for (int w1_n_tile = 0; w1_n_tile < 8; w1_n_tile++) {
                        asm volatile("ldmatrix.sync.aligned.shared::cta.m8n16.x2.b8x16.b4x16_p64 {%0, %1}, [%2];\n"
                            : "=r"(w1_b_frag_0[0]), "=r"(w1_b_frag_0[1])
                            : "r"(w1_smem_b_addr + w1_math_stage * 32768 + (unsigned int)(((lane & 7) + (w1_warp_n * 8 + w1_n_tile) * 8) * 128) + (unsigned int)((lane >> 3 & 1) * 16 + w1_k_step * 32 ^ ((lane & 7) + (w1_warp_n * 8 + w1_n_tile) * 8 & 7) << 4))
                            : "memory");
                        asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&w1_sfb_word_0[0])) : "r"(w1_smem_sfb_addr + w1_math_stage * 1024 + (unsigned int)(((w1_warp_n * 8 + w1_n_tile) * 8 + w1_group_id) * 4)));
                        w1_sfb_word_0[0] = w1_sfb_word_0[0] >> (unsigned int)(w1_k_step * 8) & 255;
                        asm volatile("mma.sync.aligned.kind::mxf8f6f4.block_scale.scale_vec::1X.m16n8k32.row.col.f32.e4m3.e2m1.f32.ue8m0 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3}, {%10}, {%11, %12}, {%13}, {%14, %15};\n"
                            : "+f"((w1_accum + w1_n_tile * 4)[0]), "+f"((w1_accum + w1_n_tile * 4)[1]), "+f"((w1_accum + w1_n_tile * 4)[2]), "+f"((w1_accum + w1_n_tile * 4)[3])
                            : "r"(w1_a_frag_0[0]), "r"(w1_a_frag_0[1]), "r"(w1_a_frag_0[2]), "r"(w1_a_frag_0[3]), "r"(((uint32_t)(w1_b_frag_0[0]) << 2)), "r"(((uint32_t)(w1_b_frag_0[1]) << 2)), "r"(w1_sfa_word_0[0]), "h"(((uint16_t)0)), "h"(((uint16_t)0)), "r"(w1_sfb_word_0[0]), "h"(((uint16_t)0)), "h"(((uint16_t)0)));
                    }
                }
                __syncwarp();
                if (elect_sync()) {
                    mbarrier_arrive(w1_empty_addr + (w1_math_stage) * 8);
                }
                w1_math_stage += 1;
                if (w1_math_stage == 2) { w1_math_stage = 0; _phase_w1_empty ^= 1; _phase_w1_full ^= 1; }
            }
            int w1_stage_row_0 = w1_warp_m * 16 + w1_group_id;
            int w1_stage_row_1 = w1_stage_row_0 + 8;
            #pragma unroll
            for (int w1_d_pass = 0; w1_d_pass < 2; w1_d_pass++) {
                if (warp == 0) {
                    if (elect_sync()) {
                        asm volatile("cp.async.bulk.wait_group.read 0;");
                    }
                }
                asm volatile("barrier.sync 15, 512;" ::: "memory");
                if (w1_warp_n / 2 == w1_d_pass) {
                    #pragma unroll
                    for (int w1_n_tile_2 = 0; w1_n_tile_2 < 8; w1_n_tile_2++) {
                        int w1_local_col = w1_warp_n % 2 * 64 + w1_n_tile_2 * 8 + w1_thread_id * 2;
                        int w1_acc_base = w1_n_tile_2 * 4;
                        int w1_sub_t = w1_local_col / 64;
                        int w1_col_in = w1_local_col - w1_sub_t * 64;
                        int w1_addr0 = w1_sub_t * 8192 + w1_stage_row_0 * 128 + w1_col_in * 2;
                        int w1_addr1 = w1_sub_t * 8192 + w1_stage_row_1 * 128 + w1_col_in * 2;
                        d_stage[w1_addr0 / 2] = w1_accum[w1_acc_base];
                        d_stage[w1_addr0 / 2 + 1] = w1_accum[w1_acc_base + 1];
                        d_stage[w1_addr1 / 2] = w1_accum[w1_acc_base + 2];
                        d_stage[w1_addr1 / 2 + 1] = w1_accum[w1_acc_base + 3];
                    }
                }
                asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                asm volatile("barrier.sync 15, 512;" ::: "memory");
                if (warp == 0) {
                    if (elect_sync()) {
                        tma_store_2d(W1_D, w1_n_block_2 * 256 + w1_d_pass * 128, w1_pool_row_2, d_stage_addr);
                        tma_store_2d(W1_D, w1_n_block_2 * 256 + w1_d_pass * 128 + 64, w1_pool_row_2, d_stage_addr + 8192);
                        asm volatile("cp.async.bulk.commit_group;");
                    }
                }
            }
            int w1_fused_row_0 = w1_pool_row_2 + w1_warp_m * 16 + w1_group_id;
            int w1_fused_row_1 = w1_fused_row_0 + 8;
            float w1_rw_0 = routing_weight_pool[w1_fused_row_0];
            float w1_rw_1 = routing_weight_pool[w1_fused_row_1];
            unsigned long long w1_int_base_0 = (unsigned long long)w1_fused_row_0 * 3072;
            unsigned long long w1_int_base_1 = (unsigned long long)w1_fused_row_1 * 3072;
            int w1_req_group = w1_n_block_2 * 4 + w1_warp_n;
            float w1_amax_0 = 0.0f;
            float w1_amax_1 = 0.0f;
            #pragma unroll
            for (int w1_rq = 0; w1_rq < 4; w1_rq++) {
                int w1_gate_base = w1_rq * 2 * 4;
                int w1_up_base = w1_gate_base + 4;
                #pragma unroll
                for (int w1_rc = 0; w1_rc < 2; w1_rc++) {
                    float w1_gate_0 = (float)(__nv_bfloat16)w1_accum[w1_gate_base + w1_rc];
                    float w1_up_0 = (float)(__nv_bfloat16)w1_accum[w1_up_base + w1_rc];
                    float _min_10 = fminf(w1_gate_0, 10.0f);
                    w1_gate_0 = _min_10;
                    float _max_10 = max_noftz(w1_up_0, -10.0f);
                    float _min_11 = fminf(_max_10, 10.0f);
                    w1_up_0 = _min_11;
                    float _exp2_0 = approx_exp2((-w1_gate_0) * 1.4426950408889634f);
                    float w1_sig_0 = 1.0f / (1.0f + _exp2_0);
                    float w1_routed_val_0 = w1_gate_0 * w1_sig_0 * w1_up_0 * w1_rw_0;
                    w1_routed_a[w1_rq * 2 + w1_rc] = w1_routed_val_0;
                    float _max_11 = max_noftz(w1_routed_val_0, -w1_routed_val_0);
                    float _max_12 = max_noftz(w1_amax_0, _max_11);
                    w1_amax_0 = _max_12;
                    float w1_gate_1 = (float)(__nv_bfloat16)w1_accum[w1_gate_base + 2 + w1_rc];
                    float w1_up_1 = (float)(__nv_bfloat16)w1_accum[w1_up_base + 2 + w1_rc];
                    float _min_12 = fminf(w1_gate_1, 10.0f);
                    w1_gate_1 = _min_12;
                    float _max_13 = max_noftz(w1_up_1, -10.0f);
                    float _min_13 = fminf(_max_13, 10.0f);
                    w1_up_1 = _min_13;
                    float _exp2_1 = approx_exp2((-w1_gate_1) * 1.4426950408889634f);
                    float w1_sig_1 = 1.0f / (1.0f + _exp2_1);
                    float w1_routed_val_1 = w1_gate_1 * w1_sig_1 * w1_up_1 * w1_rw_1;
                    w1_routed_b[w1_rq * 2 + w1_rc] = w1_routed_val_1;
                    float _max_14 = max_noftz(w1_routed_val_1, -w1_routed_val_1);
                    float _max_15 = max_noftz(w1_amax_1, _max_14);
                    w1_amax_1 = _max_15;
                }
            }
            float _shfl_xor_0 = __shfl_xor_sync(0xFFFFFFFF, w1_amax_0, 2);
            float _max_16 = max_noftz(w1_amax_0, _shfl_xor_0);
            w1_amax_0 = _max_16;
            float _shfl_xor_1 = __shfl_xor_sync(0xFFFFFFFF, w1_amax_0, 1);
            float _max_17 = max_noftz(w1_amax_0, _shfl_xor_1);
            w1_amax_0 = _max_17;
            float _shfl_xor_2 = __shfl_xor_sync(0xFFFFFFFF, w1_amax_1, 2);
            float _max_18 = max_noftz(w1_amax_1, _shfl_xor_2);
            w1_amax_1 = _max_18;
            float _shfl_xor_3 = __shfl_xor_sync(0xFFFFFFFF, w1_amax_1, 1);
            float _max_19 = max_noftz(w1_amax_1, _shfl_xor_3);
            w1_amax_1 = _max_19;
            float w1_sf_0 = w1_amax_0 * 0.002232142857142857f;
            unsigned int w1_sf_0_bits = 0;
            w1_sf_0_bits = reinterpret_cast<unsigned int*>(&w1_sf_0)[0];
            unsigned int w1_sf_0_exp = (w1_sf_0_bits >> 23 & 255) + ((w1_sf_0_bits & 8388607) + 8388607 >> 23);
            unsigned int _min_14 = ((w1_sf_0_exp) < (254) ? (w1_sf_0_exp) : (254));
            w1_sf_0_exp = _min_14;
            unsigned int w1_sf_0_inv_bits = 254 - w1_sf_0_exp << 23;
            float w1_sf_0_inv = 0.0f;
            w1_sf_0_inv = reinterpret_cast<float*>(&w1_sf_0_inv_bits)[0];
            float w1_sf_1 = w1_amax_1 * 0.002232142857142857f;
            unsigned int w1_sf_1_bits = 0;
            w1_sf_1_bits = reinterpret_cast<unsigned int*>(&w1_sf_1)[0];
            unsigned int w1_sf_1_exp = (w1_sf_1_bits >> 23 & 255) + ((w1_sf_1_bits & 8388607) + 8388607 >> 23);
            unsigned int _min_15 = ((w1_sf_1_exp) < (254) ? (w1_sf_1_exp) : (254));
            w1_sf_1_exp = _min_15;
            unsigned int w1_sf_1_inv_bits = 254 - w1_sf_1_exp << 23;
            float w1_sf_1_inv = 0.0f;
            w1_sf_1_inv = reinterpret_cast<float*>(&w1_sf_1_inv_bits)[0];
            if (w1_thread_id == 0) {
                int w1_sf_index_0 = ((w1_req_group >> 2) * 199632 + w1_fused_row_0) * 4 + (w1_req_group & 3);
                *(reinterpret_cast<unsigned char*>(intermediate_sfa_u8 + w1_sf_index_0) + (0)) = (unsigned char)(w1_sf_0_exp);
                int w1_sf_index_1 = ((w1_req_group >> 2) * 199632 + w1_fused_row_1) * 4 + (w1_req_group & 3);
                *(reinterpret_cast<unsigned char*>(intermediate_sfa_u8 + w1_sf_index_1) + (0)) = (unsigned char)(w1_sf_1_exp);
            }
            #pragma unroll
            for (int w1_sq = 0; w1_sq < 4; w1_sq++) {
                #pragma unroll
                for (int w1_sc = 0; w1_sc < 2; w1_sc++) {
                    int w1_log_n = w1_req_group * 32 + w1_sq * 8 + w1_thread_id * 2 + w1_sc;
                    {
                        unsigned short _fp8_pair;
                        asm("cvt.rn.satfinite.e4m3x2.f32 %0, 0f00000000, %1;" : "=h"(_fp8_pair) : "f"(w1_routed_a[w1_sq * 2 + w1_sc] * w1_sf_0_inv));
                        *(reinterpret_cast<unsigned char*>(intermediate_fp8 + (w1_int_base_0 + (unsigned long long)w1_log_n)) + (0)) = (unsigned char)(_fp8_pair & 0xFF);
                    }
                    {
                        unsigned short _fp8_pair;
                        asm("cvt.rn.satfinite.e4m3x2.f32 %0, 0f00000000, %1;" : "=h"(_fp8_pair) : "f"(w1_routed_b[w1_sq * 2 + w1_sc] * w1_sf_1_inv));
                        *(reinterpret_cast<unsigned char*>(intermediate_fp8 + (w1_int_base_1 + (unsigned long long)w1_log_n)) + (0)) = (unsigned char)(_fp8_pair & 0xFF);
                    }
                }
            }
            if (elect_sync()) {
                atomicAdd(&requant_groups_done[0], 16);
            }
            __threadfence();
            __syncwarp();
            if (elect_sync()) {
                int _atomic_old_5 = atomicAdd(&w1_warp_done[w1_tile_2], 1);
                int w1_previous = _atomic_old_5;
                if (w1_previous == 15) {
                    atomicAdd(&w1_tiles_completed[0], 1);
                } else if (w1_previous >= 16) {
                    atomicMax(&protocol_error[0], 1);
                }
            }
        }
    }
    cooperative_groups::this_grid().sync();
    if (bid == 0 && tid == 0) {
        if (w1_tiles_completed[0] != total_m_tasks[0] * 24) {
            atomicMax(&protocol_error[0], 1);
        }
        int scatter_sum = 0;
        #pragma unroll 1
        for (int audit_es = 0; audit_es < 192; audit_es++) {
            int audit_e = audit_es / 4;
            int audit_s = audit_es - audit_e * 4;
            if (audit_s < world_size) {
                scatter_sum = scatter_sum + expert_source_offsets[audit_es];
                if (expert_source_offsets[audit_es] != source_expert_counts[audit_s * 48 + audit_e]) {
                    atomicMax(&protocol_error[0], 1);
                }
            }
        }
        if (scatter_sum != total_valid_routes[0]) {
            atomicMax(&protocol_error[0], 1);
        }
    }
    cooperative_groups::this_grid().sync();
    cooperative_groups::this_grid().sync();
    if (bid == 0 && tid == 0) {
        int expected_requant_groups = total_m_tasks[0] * 64 * 96;
        if (requant_groups_done[0] != expected_requant_groups) {
            atomicMax(&protocol_error[0], 1);
        }
    }
    cooperative_groups::this_grid().sync();
    if (warp == 16) {
        if (elect_sync()) {
            asm volatile("prefetch.tensormap [%0];" :: "l"((uint64_t)(W2_A)) : "memory");
            asm volatile("prefetch.tensormap [%0];" :: "l"((uint64_t)(W2_B)) : "memory");
            asm volatile("prefetch.tensormap [%0];" :: "l"((uint64_t)(W2_SFA)) : "memory");
            asm volatile("prefetch.tensormap [%0];" :: "l"((uint64_t)(W2_SFB)) : "memory");
        }
    }
    unsigned int _phase_w2_empty = 1;
    if (bid > 0 && warp == 16) {
        unsigned int w2_load_stage = 0;
        int w2_tile_count = total_m_tasks[0] * 28;
        int c56_w2_seq = 0;
        int c56_w2_mb = 8192 + bid * 32;
        #pragma unroll 1
        for (int c56_w2_iter = 0; c56_w2_iter < 87360; c56_w2_iter++) {
            int w2_tile = 0;
            if (elect_sync()) {
                int _atomic_old_6 = atomicAdd(&c56_claim_cursor[1], 1);
                w2_tile = _atomic_old_6;
            }
            int _shfl_2 = __shfl_sync(0xFFFFFFFF, w2_tile, 0);
            w2_tile = _shfl_2;
            if (w2_tile >= w2_tile_count) {
                if (elect_sync()) {
                    atomicMax(&c56_tile_mailbox[c56_w2_mb + (c56_w2_seq & 3)], 2147483647);
                }
                break;
            }
            if (elect_sync()) {
                atomicMax(&c56_tile_mailbox[c56_w2_mb + (c56_w2_seq & 3)], w2_tile + 1);
            }
            c56_w2_seq += 1;
            int w2_task = w2_tile / 28;
            int w2_n_block = w2_tile - w2_task * 28;
            int w2_pool_row = task_pool_row[w2_task];
            int w2_local_expert = task_local_expert[w2_task];
            #pragma unroll 1
            for (int w2_k_block = 0; w2_k_block < 24; w2_k_block++) {
                mbarrier_wait(w2_empty_addr + (w2_load_stage) * 8, _phase_w2_empty);
                if (warp == 16) {
                    if (elect_sync()) {
                        tma_2d_gmem2smem(w1_smem_sfa_addr + w2_load_stage * 256, W2_SFA, w2_pool_row, w2_k_block, w2_full_addr + (w2_load_stage) * 8);
                        tma_2d_gmem2smem(w1_smem_sfb_addr + w2_load_stage * 1024, W2_SFB, w2_n_block * 256, w2_local_expert * 24 + w2_k_block, w2_full_addr + (w2_load_stage) * 8);
                        tma_2d_gmem2smem(w1_smem_a_addr + w2_load_stage * 8192, W2_A, w2_k_block * 128, w2_pool_row, w2_full_addr + (w2_load_stage) * 8);
                        tma_2d_gmem2smem(w1_smem_b_addr + w2_load_stage * 32768, W2_B, w2_k_block * 128, w2_local_expert * 7168 + w2_n_block * 256, w2_full_addr + (w2_load_stage) * 8);
                        mbarrier_arrive_expect_tx(w2_full_addr + (w2_load_stage) * 8, 25856);
                    }
                }
                w2_load_stage += 1;
                if (w2_load_stage == 2) { w2_load_stage = 0; _phase_w2_empty ^= 1; }
            }
        }
    }
    unsigned int _phase_w2_full = 0;
    if (bid > 0 && warp < 16) {
        unsigned int w2_math_stage = 0;
        int w2_warp_m = warp / 4;
        int w2_warp_n = warp % 4;
        int w2_group_id = lane / 4;
        int w2_thread_id = lane % 4;
        float w2_accum[32];
        unsigned int w2_a_frag_0[4];
        unsigned int w2_b_frag_0[2];
        unsigned int w2_sfa_word_0[1];
        unsigned int w2_sfb_word_0[1];
        int w2_tile_count_2 = total_m_tasks[0] * 28;
        int c56m_w2_last = 0;
        int c56m_w2_seq = 0;
        int c56m_w2_mb = 8192 + bid * 32;
        #pragma unroll 1
        for (int c56m_w2_iter = 0; c56m_w2_iter < 87360; c56m_w2_iter++) {
            int c56m_w2_val = 0;
            if (elect_sync()) {
                #pragma unroll 1
                for (int c56m_w2_spin = 0; c56m_w2_spin < 1073741824; c56m_w2_spin++) {
                    int _atomic_old_7 = atomicAdd(&c56_tile_mailbox[c56m_w2_mb + (c56m_w2_seq & 3)], 0);
                    int c56m_w2_probe = _atomic_old_7;
                    if (c56m_w2_probe > c56m_w2_last) {
                        c56m_w2_val = c56m_w2_probe;
                        break;
                    }
                }
            }
            int _shfl_3 = __shfl_sync(0xFFFFFFFF, c56m_w2_val, 0);
            c56m_w2_val = _shfl_3;
            if (c56m_w2_val == 2147483647) {
                break;
            }
            if (c56m_w2_val <= c56m_w2_last) {
                if (elect_sync()) {
                    atomicMax(&protocol_error[0], 1);
                }
                break;
            }
            c56m_w2_last = c56m_w2_val;
            c56m_w2_seq += 1;
            int w2_tile_2 = c56m_w2_val - 1;
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
            w2_accum[16] = 0.0f;
            w2_accum[17] = 0.0f;
            w2_accum[18] = 0.0f;
            w2_accum[19] = 0.0f;
            w2_accum[20] = 0.0f;
            w2_accum[21] = 0.0f;
            w2_accum[22] = 0.0f;
            w2_accum[23] = 0.0f;
            w2_accum[24] = 0.0f;
            w2_accum[25] = 0.0f;
            w2_accum[26] = 0.0f;
            w2_accum[27] = 0.0f;
            w2_accum[28] = 0.0f;
            w2_accum[29] = 0.0f;
            w2_accum[30] = 0.0f;
            w2_accum[31] = 0.0f;
            int w2_task_2 = w2_tile_2 / 28;
            int w2_n_block_2 = w2_tile_2 - w2_task_2 * 28;
            int w2_pool_row_2 = task_pool_row[w2_task_2];
            #pragma unroll 1
            for (int w2_k_block_2 = 0; w2_k_block_2 < 24; w2_k_block_2++) {
                mbarrier_wait(w2_full_addr + (w2_math_stage) * 8, _phase_w2_full);
                asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                #pragma unroll
                for (int w2_k_step = 0; w2_k_step < 4; w2_k_step++) {
                    asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                        : "=r"(w2_a_frag_0[0]), "=r"(w2_a_frag_0[1]), "=r"(w2_a_frag_0[2]), "=r"(w2_a_frag_0[3])
                        : "r"(w1_smem_a_addr + w2_math_stage * 8192 + (unsigned int)(((lane & 7) + (lane >> 3 & 1) * 8 + w2_warp_m * 16) * 128) + (unsigned int)((lane >> 4) * 16 + w2_k_step * 32 ^ ((lane & 7) + (lane >> 3 & 1) * 8 + w2_warp_m * 16 & 7) << 4))
                        : "memory");
                    asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&w2_sfa_word_0[0])) : "r"(w1_smem_sfa_addr + w2_math_stage * 256 + (unsigned int)((w2_warp_m * 16 + w2_group_id + (w2_thread_id & 1) * 8) * 4)));
                    w2_sfa_word_0[0] = w2_sfa_word_0[0] >> (unsigned int)(w2_k_step * 8) & 255;
                    #pragma unroll
                    for (int w2_n_tile = 0; w2_n_tile < 8; w2_n_tile++) {
                        asm volatile("ldmatrix.sync.aligned.shared::cta.m8n16.x2.b8x16.b4x16_p64 {%0, %1}, [%2];\n"
                            : "=r"(w2_b_frag_0[0]), "=r"(w2_b_frag_0[1])
                            : "r"(w1_smem_b_addr + w2_math_stage * 32768 + (unsigned int)(((lane & 7) + (w2_warp_n * 8 + w2_n_tile) * 8) * 128) + (unsigned int)((lane >> 3 & 1) * 16 + w2_k_step * 32 ^ ((lane & 7) + (w2_warp_n * 8 + w2_n_tile) * 8 & 7) << 4))
                            : "memory");
                        asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&w2_sfb_word_0[0])) : "r"(w1_smem_sfb_addr + w2_math_stage * 1024 + (unsigned int)(((w2_warp_n * 8 + w2_n_tile) * 8 + w2_group_id) * 4)));
                        w2_sfb_word_0[0] = w2_sfb_word_0[0] >> (unsigned int)(w2_k_step * 8) & 255;
                        asm volatile("mma.sync.aligned.kind::mxf8f6f4.block_scale.scale_vec::1X.m16n8k32.row.col.f32.e4m3.e2m1.f32.ue8m0 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3}, {%10}, {%11, %12}, {%13}, {%14, %15};\n"
                            : "+f"((w2_accum + w2_n_tile * 4)[0]), "+f"((w2_accum + w2_n_tile * 4)[1]), "+f"((w2_accum + w2_n_tile * 4)[2]), "+f"((w2_accum + w2_n_tile * 4)[3])
                            : "r"(w2_a_frag_0[0]), "r"(w2_a_frag_0[1]), "r"(w2_a_frag_0[2]), "r"(w2_a_frag_0[3]), "r"(((uint32_t)(w2_b_frag_0[0]) << 2)), "r"(((uint32_t)(w2_b_frag_0[1]) << 2)), "r"(w2_sfa_word_0[0]), "h"(((uint16_t)0)), "h"(((uint16_t)0)), "r"(w2_sfb_word_0[0]), "h"(((uint16_t)0)), "h"(((uint16_t)0)));
                    }
                }
                __syncwarp();
                if (elect_sync()) {
                    mbarrier_arrive(w2_empty_addr + (w2_math_stage) * 8);
                }
                w2_math_stage += 1;
                if (w2_math_stage == 2) { w2_math_stage = 0; _phase_w2_empty ^= 1; _phase_w2_full ^= 1; }
            }
            int w2_ret_row_0 = w2_pool_row_2 + w2_warp_m * 16 + w2_group_id;
            int w2_ret_row_1 = w2_ret_row_0 + 8;
            int w2_ret_src_0 = meta_source_rank[w2_ret_row_0];
            int w2_ret_src_1 = meta_source_rank[w2_ret_row_1];
            int w2_ret_valid_0 = (int)(w2_ret_src_0 >= 0 && w2_ret_src_0 < world_size);
            int w2_ret_valid_1 = (int)(w2_ret_src_1 >= 0 && w2_ret_src_1 < world_size);
            unsigned long long w2_ret_base_0 = 0;
            unsigned long long w2_ret_base_1 = 0;
            if (w2_ret_valid_0 != 0) {
                w2_ret_base_0 = (unsigned long long)(w2_ret_src_0 * 2 + slot) * 352419840 + (unsigned long long)meta_result_index[w2_ret_row_0] * 7168;
            }
            if (w2_ret_valid_1 != 0) {
                w2_ret_base_1 = (unsigned long long)(w2_ret_src_1 * 2 + slot) * 352419840 + (unsigned long long)meta_result_index[w2_ret_row_1] * 7168;
            }
            #pragma unroll
            for (int w2_n_tile_2 = 0; w2_n_tile_2 < 8; w2_n_tile_2++) {
                int w2_output_n_tile = w2_warp_n * 8 + w2_n_tile_2;
                int w2_output_col = w2_n_block_2 * 256 + w2_output_n_tile * 8 + w2_thread_id * 2;
                int w2_acc_base = w2_n_tile_2 * 4;
                if (w2_ret_valid_0 != 0) {
                    {
                        __nv_bfloat162 _pk = __floats2bfloat162_rn(w2_accum[w2_acc_base + 0], w2_accum[w2_acc_base + 1]);
                        *reinterpret_cast<__nv_bfloat162*>(&((__nv_bfloat16*)(reinterpret_cast<__nv_bfloat16*>(result_out) + (w2_ret_base_0 + (unsigned long long)w2_output_col)))[0]) = _pk;
                    }
                }
                if (w2_ret_valid_1 != 0) {
                    {
                        __nv_bfloat162 _pk = __floats2bfloat162_rn(w2_accum[w2_acc_base + 2 + 0], w2_accum[w2_acc_base + 2 + 1]);
                        *reinterpret_cast<__nv_bfloat162*>(&((__nv_bfloat16*)(reinterpret_cast<__nv_bfloat16*>(result_out) + (w2_ret_base_1 + (unsigned long long)w2_output_col)))[0]) = _pk;
                    }
                }
            }
            int w2_stage_row_0 = w2_warp_m * 16 + w2_group_id;
            int w2_stage_row_1 = w2_stage_row_0 + 8;
            #pragma unroll
            for (int w2_d_pass = 0; w2_d_pass < 2; w2_d_pass++) {
                if (warp == 0) {
                    if (elect_sync()) {
                        asm volatile("cp.async.bulk.wait_group.read 0;");
                    }
                }
                asm volatile("barrier.sync 15, 512;" ::: "memory");
                if (w2_warp_n / 2 == w2_d_pass) {
                    #pragma unroll
                    for (int w2_n_tile_3 = 0; w2_n_tile_3 < 8; w2_n_tile_3++) {
                        int w2_local_col = w2_warp_n % 2 * 64 + w2_n_tile_3 * 8 + w2_thread_id * 2;
                        int w2_acc_base_2 = w2_n_tile_3 * 4;
                        int w2_sub_t = w2_local_col / 64;
                        int w2_col_in = w2_local_col - w2_sub_t * 64;
                        int w2_addr0 = w2_sub_t * 8192 + w2_stage_row_0 * 128 + w2_col_in * 2;
                        int w2_addr1 = w2_sub_t * 8192 + w2_stage_row_1 * 128 + w2_col_in * 2;
                        d_stage[w2_addr0 / 2] = w2_accum[w2_acc_base_2];
                        d_stage[w2_addr0 / 2 + 1] = w2_accum[w2_acc_base_2 + 1];
                        d_stage[w2_addr1 / 2] = w2_accum[w2_acc_base_2 + 2];
                        d_stage[w2_addr1 / 2 + 1] = w2_accum[w2_acc_base_2 + 3];
                    }
                }
                asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                asm volatile("barrier.sync 15, 512;" ::: "memory");
                if (warp == 0) {
                    if (elect_sync()) {
                        tma_store_2d(W2_D, w2_n_block_2 * 256 + w2_d_pass * 128, w2_pool_row_2, d_stage_addr);
                        tma_store_2d(W2_D, w2_n_block_2 * 256 + w2_d_pass * 128 + 64, w2_pool_row_2, d_stage_addr + 8192);
                        asm volatile("cp.async.bulk.commit_group;");
                    }
                }
            }
            __threadfence();
            __syncwarp();
            if (elect_sync()) {
                int _atomic_old_8 = atomicAdd(&w2_warp_done[w2_tile_2], 1);
                int w2_previous = _atomic_old_8;
                if (w2_previous == 15) {
                    atomicAdd(&w2_tiles_completed[0], 1);
                    {
                        unsigned int* _gc_p = reinterpret_cast<unsigned int*>(w2_task_counter) + (w2_task_2);
                        unsigned int _gc_old;
                        asm volatile("atom.release.gpu.global.add.u32 %0, [%1], 1;" : "=r"(_gc_old) : "l"(_gc_p) : "memory");
                    }
                } else if (w2_previous >= 16) {
                    atomicMax(&protocol_error[0], 1);
                }
            }
        }
    }
    if (bid == 0) {
        if (warp == 0) {
            if (elect_sync()) {
                #pragma unroll 1
                for (int map_source = 0; map_source < 4; map_source++) {
                    if (map_source < world_size) {
                        int _max_20 = ((source_record_counts[map_source]) > (0) ? (source_record_counts[map_source]) : (0));
                        int _min_16 = ((_max_20) < (8192) ? (_max_20) : (8192));
                        {
                            unsigned int* _gca_p = reinterpret_cast<unsigned int*>(scatter_source_counter) + (map_source);
                            while (true) {
                                unsigned int _gca_v;
                                asm volatile("ld.acquire.gpu.global.u32 %0, [%1];" : "=r"(_gca_v) : "l"(_gca_p));
                                if (_gca_v >= (unsigned int)(_min_16 * 6)) break;
                            }
                        }
                        int _max_21 = ((source_route_counts[map_source]) > (0) ? (source_route_counts[map_source]) : (0));
                        int _min_17 = ((_max_21) < (49152) ? (_max_21) : (49152));
                        int map_routes = _min_17;
                        int _max_22 = ((map_routes) > (1) ? (map_routes) : (1));
                        int map_count = _max_22;
                        unsigned long long map_local_byte = (unsigned long long)(map_source * 2 + slot) * 704839680 + 704643072;
                        unsigned long long map_remote_byte = (unsigned long long)(rank * 2 + slot) * 704839680 + 704643072;
                        __threadfence_system();
                        // gin_put_signal_add: strong remote completion on context 0
                        {
                            ncclGin __gin{*(gin_dev_comm), (int)(0)};
                            __gin.put(ncclTeamWorld(*(gin_dev_comm)), (int)(map_source), result_inbox_window, (size_t)(map_remote_byte), result_out_window, (size_t)(map_local_byte), (size_t)(map_count * 4),
                                ncclGin_StrongSignalAdd{(ncclGinSignal_t)(8 + rank), (uint64_t)(1)}, ncclGin_None{}, ncclCoopThread());
                        }
                    }
                }
            }
        }
        if (warp == 0) {
            __syncwarp();
            #pragma unroll 1
            for (int service_task = 0; service_task < 3120; service_task++) {
                if (service_task < total_m_tasks[0]) {
                    if (elect_sync()) {
                        {
                            unsigned int* _gca_p = reinterpret_cast<unsigned int*>(w2_task_counter) + (service_task);
                            while (true) {
                                unsigned int _gca_v;
                                asm volatile("ld.acquire.gpu.global.u32 %0, [%1];" : "=r"(_gca_v) : "l"(_gca_p));
                                if (_gca_v >= (unsigned int)(28)) break;
                            }
                        }
                    }
                    __syncwarp();
                    int service_row_base = task_pool_row[service_task];
                    #pragma unroll 1
                    for (int service_row_offset = lane; service_row_offset < 64; service_row_offset += 32) {
                        service_ready_chunks[service_row_offset] = -1;
                    }
                    __syncwarp();
                    #pragma unroll 1
                    for (int service_row_offset_2 = lane; service_row_offset_2 < 64; service_row_offset_2 += 32) {
                        int service_row = service_row_base + service_row_offset_2;
                        int service_source = meta_source_rank[service_row];
                        if (service_source >= 0 && service_source < world_size) {
                            int _max_23 = ((source_route_counts[service_source]) > (0) ? (source_route_counts[service_source]) : (0));
                            int _min_18 = ((_max_23) < (49152) ? (_max_23) : (49152));
                            int c41_tr = _min_18;
                            int _max_24 = (((c41_tr + 256 - 1) / 256 - 1) > (0) ? ((c41_tr + 256 - 1) / 256 - 1) : (0));
                            int c41_tfull = _max_24;
                            int c41_tts = c41_tfull * 256;
                            int c41_tidx = meta_result_index[service_row];
                            int service_chunk_local = 0;
                            if (c41_tidx < c41_tts) {
                                service_chunk_local = c41_tidx / 256;
                            } else {
                                service_chunk_local = c41_tfull + (c41_tidx - c41_tts) / 64;
                            }
                            int service_chunk = service_source * 195 + service_chunk_local;
                            int _atomic_old_9 = atomicAdd(&result_chunk_tally[service_chunk], 1);
                            int service_previous = _atomic_old_9;
                            int service_total = result_chunk_total[service_chunk];
                            if (service_previous + 1 == service_total) {
                                service_ready_chunks[service_row_offset_2] = service_chunk;
                            } else if (service_previous >= service_total) {
                                atomicMax(&protocol_error[0], 1);
                            }
                        }
                    }
                    __syncwarp();
                    if (elect_sync()) {
                        #pragma unroll 1
                        for (int service_ready_row = 0; service_ready_row < 64; service_ready_row++) {
                            int service_chunk_2 = service_ready_chunks[service_ready_row];
                            if (service_chunk_2 >= 0) {
                                int service_source_2 = service_chunk_2 / 195;
                                int service_chunk_local_2 = service_chunk_2 - service_source_2 * 195;
                                int _max_25 = ((source_route_counts[service_source_2]) > (0) ? (source_route_counts[service_source_2]) : (0));
                                int _min_19 = ((_max_25) < (49152) ? (_max_25) : (49152));
                                int service_routes = _min_19;
                                int _max_26 = (((service_routes + 256 - 1) / 256 - 1) > (0) ? ((service_routes + 256 - 1) / 256 - 1) : (0));
                                int c41_pfull = _max_26;
                                int c41_pts = c41_pfull * 256;
                                int c41_start = 0;
                                int c41_cap = 256;
                                if (service_chunk_local_2 < c41_pfull) {
                                    c41_start = service_chunk_local_2 * 256;
                                } else {
                                    c41_start = c41_pts + (service_chunk_local_2 - c41_pfull) * 64;
                                    c41_cap = 64;
                                }
                                int _min_20 = ((service_routes - c41_start) < (c41_cap) ? (service_routes - c41_start) : (c41_cap));
                                int service_chunk_rows = _min_20;
                                unsigned long long service_local_byte = (unsigned long long)(service_source_2 * 2 + slot) * 704839680 + (unsigned long long)c41_start * 14336;
                                unsigned long long service_remote_byte = (unsigned long long)(rank * 2 + slot) * 704839680 + (unsigned long long)c41_start * 14336;
                                __threadfence_system();
                                // gin_put_signal_add: strong remote completion on context 0
                                {
                                    ncclGin __gin{*(gin_dev_comm), (int)(0)};
                                    __gin.put(ncclTeamWorld(*(gin_dev_comm)), (int)(service_source_2), result_inbox_window, (size_t)(service_remote_byte), result_out_window, (size_t)(service_local_byte), (size_t)(service_chunk_rows * 7168 * 2),
                                        ncclGin_StrongSignalAdd{(ncclGinSignal_t)(8 + rank), (uint64_t)(1)}, ncclGin_None{}, ncclCoopThread());
                                }
                            }
                        }
                    }
                    __syncwarp();
                }
            }
            if (elect_sync()) {
                #pragma unroll 1
                for (int tail_source = 0; tail_source < 4; tail_source++) {
                    if (tail_source < world_size) {
                        int _max_27 = ((source_route_counts[tail_source]) > (0) ? (source_route_counts[tail_source]) : (0));
                        int _min_21 = ((_max_27) < (49152) ? (_max_27) : (49152));
                        int tail_routes = _min_21;
                        if (tail_routes == 0) {
                            unsigned long long tail_local_byte = (unsigned long long)(tail_source * 2 + slot) * 704839680;
                            unsigned long long tail_remote_byte = (unsigned long long)(rank * 2 + slot) * 704839680;
                            __threadfence_system();
                            // gin_put_signal_add: strong remote completion on context 0
                            {
                                ncclGin __gin{*(gin_dev_comm), (int)(0)};
                                __gin.put(ncclTeamWorld(*(gin_dev_comm)), (int)(tail_source), result_inbox_window, (size_t)(tail_remote_byte), result_out_window, (size_t)(tail_local_byte), (size_t)(14336),
                                    ncclGin_StrongSignalAdd{(ncclGinSignal_t)(8 + rank), (uint64_t)(1)}, ncclGin_None{}, ncclCoopThread());
                            }
                        }
                        int _max_28 = (((tail_routes + 256 - 1) / 256 - 1) > (0) ? ((tail_routes + 256 - 1) / 256 - 1) : (0));
                        int c41_afull = _max_28;
                        int tail_chunk_count = c41_afull + (tail_routes - c41_afull * 256 + 64 - 1) / 64;
                        #pragma unroll 1
                        for (int verify_chunk = 0; verify_chunk < 195; verify_chunk++) {
                            if (tail_chunk_count > verify_chunk) {
                                int verify_index = tail_source * 195 + verify_chunk;
                                if (result_chunk_tally[verify_index] != result_chunk_total[verify_index]) {
                                    atomicMax(&protocol_error[0], 1);
                                }
                            }
                        }
                    }
                }
            }
        }
    }
    cooperative_groups::this_grid().sync();
    if (bid == 0 && tid == 0) {
        if (w2_tiles_completed[0] != total_m_tasks[0] * 28) {
            atomicMax(&protocol_error[0], 1);
        }
    }
    if (bid == 0) {
        #pragma unroll 1
        for (int result_owner = 0; result_owner < world_size; result_owner++) {
            if (warp == 0) {
                if (elect_sync()) {
                    int owner_routes = owner_route_counts[result_owner];
                    int _max_29 = ((owner_routes) > (0) ? (owner_routes) : (0));
                    int _min_22 = ((_max_29) < (49152) ? (_max_29) : (49152));
                    int safe_owner_routes = _min_22;
                    int _max_30 = (((safe_owner_routes + 256 - 1) / 256 - 1) > (0) ? ((safe_owner_routes + 256 - 1) / 256 - 1) : (0));
                    int c41_wfull = _max_30;
                    int _max_31 = ((c41_wfull + (safe_owner_routes - c41_wfull * 256 + 64 - 1) / 64) > (1) ? (c41_wfull + (safe_owner_routes - c41_wfull * 256 + 64 - 1) / 64) : (1));
                    int owner_chunk_count = _max_31 + 1;
                    // gin_wait_signal: acquire, rolling 64-bit comparison
                    {
                        ncclGin __gin{*(gin_dev_comm), (int)(0)};
                        __gin.waitSignal(ncclCoopThread(), (ncclGinSignal_t)(8 + result_owner), (uint64_t)(result_signal_base_scratch[result_owner] + (unsigned long long)owner_chunk_count), 64, cuda::memory_order_acquire);
                    }
                }
            }
        }
    }
    cooperative_groups::this_grid().sync();
    int combine_warp = bid * 20 + warp;
    int combine_warps = num_bids * 20;
    #pragma unroll 1
    for (int combine_token = combine_warp; combine_token < active_rows; combine_token += combine_warps) {
        unsigned long long combine_bases[6];
        int combine_valids[6];
        #pragma unroll
        for (int meta_slot_1 = 0; meta_slot_1 < 6; meta_slot_1++) {
            int meta_pair = combine_token * 6 + meta_slot_1;
            int meta_expert = topk_idx_i32[meta_pair * 2];
            int meta_expert_hi = topk_idx_i32[meta_pair * 2 + 1];
            int meta_masked = (int)(meta_expert == -1 && meta_expert_hi == -1);
            int meta_valid = (int)(meta_expert >= 0 && meta_expert < world_size * 48 && meta_expert_hi == 0);
            if (meta_valid == 0 && meta_masked == 0) {
                if (lane == 0) {
                    atomicMax(&protocol_error[0], 1);
                }
            }
            combine_valids[meta_slot_1] = 0;
            combine_bases[meta_slot_1] = 0;
            if (meta_valid != 0) {
                int meta_owner = meta_expert / 48;
                int meta_result_slot = route_result_index[meta_pair];
                int meta_owner_count = owner_route_counts[meta_owner];
                if (meta_result_slot < 0 || meta_result_slot >= meta_owner_count) {
                    if (lane == 0) {
                        atomicMax(&protocol_error[0], 1);
                    }
                } else {
                    int meta_new_slot = reinterpret_cast<int*>(result_inbox)[(unsigned long long)(meta_owner * 2 + slot) * 176209920 + 176160768 + (unsigned long long)meta_result_slot];
                    if (meta_new_slot < 0 || meta_new_slot >= meta_owner_count) {
                        if (lane == 0) {
                            atomicMax(&protocol_error[0], 1);
                        }
                    } else {
                        combine_valids[meta_slot_1] = 1;
                        combine_bases[meta_slot_1] = (unsigned long long)(meta_owner * 2 + slot) * 352419840 + (unsigned long long)meta_new_slot * 7168;
                    }
                }
            }
        }
        unsigned long long combine_out_base = (unsigned long long)combine_token * 7168;
        #pragma unroll 1
        for (int combine_base = lane * 8; combine_base < 7168; combine_base += 256) {
            float combine_acc[8];
            combine_acc[0] = 0.0f;
            combine_acc[1] = 0.0f;
            combine_acc[2] = 0.0f;
            combine_acc[3] = 0.0f;
            combine_acc[4] = 0.0f;
            combine_acc[5] = 0.0f;
            combine_acc[6] = 0.0f;
            combine_acc[7] = 0.0f;
            #pragma unroll
            for (int combine_slot = 0; combine_slot < 6; combine_slot++) {
                if (combine_valids[combine_slot] != 0) {
                    float _vec_load_0[8];
                    {
                        const uint4* _vptr_0 = reinterpret_cast<const uint4*>(reinterpret_cast<__nv_bfloat16*>(result_inbox) + (combine_bases[combine_slot] + (unsigned long long)combine_base) + 0);
                        uint4 _vld_0[1];
                        #pragma unroll
                        for (int _blk = 0; _blk < 1; _blk++) {
                            _vld_0[_blk] = _vptr_0[_blk];
                            uint32_t* _vpairs_0 = reinterpret_cast<uint32_t*>(&_vld_0[_blk]);
                            #pragma unroll
                            for (int _pair = 0; _pair < 4; _pair++) {
                                asm volatile(
                                    "{\n\t"
                                    "shl.b32 %0, %2, 16;\n\t"
                                    "and.b32 %1, %2, 0xffff0000;\n\t"
                                    "}\n"
                                    : "=f"((&_vec_load_0[0 + _blk * 8 + _pair * 2])[0]), "=f"((&_vec_load_0[0 + _blk * 8 + _pair * 2])[1])
                                    : "r"(_vpairs_0[_pair]));
                            }
                        }
                    }
                    #pragma unroll
                    for (int combine_j = 0; combine_j < 8; combine_j++) {
                        combine_acc[combine_j] = combine_acc[combine_j] + _vec_load_0[combine_j];
                    }
                }
            }
            {
                __nv_bfloat162 _pk[4];
                _pk[0] = __floats2bfloat162_rn(combine_acc[0 + 0], combine_acc[0 + 1]);
                _pk[1] = __floats2bfloat162_rn(combine_acc[0 + 2], combine_acc[0 + 3]);
                _pk[2] = __floats2bfloat162_rn(combine_acc[0 + 4], combine_acc[0 + 5]);
                _pk[3] = __floats2bfloat162_rn(combine_acc[0 + 6], combine_acc[0 + 7]);
                *reinterpret_cast<uint4*>(&((__nv_bfloat16*)(final_output + (combine_out_base + (unsigned long long)combine_base)))[0]) = *reinterpret_cast<uint4*>(&_pk[0]);
            }
        }
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

