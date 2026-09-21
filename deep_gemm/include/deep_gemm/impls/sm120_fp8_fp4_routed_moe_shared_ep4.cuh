#define DG_SM120_SHARED_TB_SCR_STAGE_BYTES 18576
#include <deep_gemm/impls/sm120_fp8_fp4_routed_moe_shared_common.cuh>

#define kernel_deepgemm_sm120_megamoe_m64n32_stage2_residency_dispatch \
    kernel_deepgemm_sm120_megamoe_m64n32_stage2_residency_fp8shared_ep4

extern "C" {

__global__ __launch_bounds__(384) void
kernel_deepgemm_sm120_megamoe_m64n32_stage2_residency_dispatch(LoomTensorMap const* W1_A, LoomTensorMap const* W1_A64, LoomTensorMap const* W1_B, LoomTensorMap const* W1_B8, LoomTensorMap const* W1_SFA, LoomTensorMap const* W1_SFA64, LoomTensorMap const* W1_SFB, LoomTensorMap const* W1_D, LoomTensorMap const* W2_A, LoomTensorMap const* W2_A64, LoomTensorMap const* W2_B, LoomTensorMap const* W2_B8, LoomTensorMap const* W2_SFA, LoomTensorMap const* W2_SFA64, LoomTensorMap const* W2_SFB, LoomTensorMap const* W2_D, uint8_t* __restrict__ intermediate_fp8, uint8_t* __restrict__ intermediate_sfa_u8, int* __restrict__ requant_groups_done, int* __restrict__ w2_warp_done, int* __restrict__ w2_tiles_completed, int* __restrict__ topk_idx_i32, float* __restrict__ topk_weights, int* __restrict__ x_fp8_i32, int* __restrict__ x_sf_i32, int* __restrict__ owner_record_counts, int* __restrict__ owner_route_counts, int* __restrict__ owner_minexp_record_base, int* __restrict__ owner_minexp_record_cursor, int* __restrict__ sorted_record_token, int* __restrict__ sorted_record_route_base, int* __restrict__ route_result_index, int* __restrict__ protocol_error, unsigned long long* __restrict__ phase_timestamps, unsigned long long* __restrict__ peer_phase_timestamps, unsigned int* __restrict__ w2_task_counter, unsigned int* __restrict__ w1_task_counter, unsigned int* __restrict__ dispatch_chunk_scatter_counter, unsigned int* __restrict__ pull_chunk_arrived, unsigned int* __restrict__ c24_front_sync, unsigned int* __restrict__ result_owner_ready, unsigned int* __restrict__ result_owner_progress, unsigned long long* __restrict__ pull_request_scratch, int* __restrict__ dispatch_chunk_targets, int* __restrict__ c56_claim_cursor, int* __restrict__ combine_claim_cursor, int* __restrict__ c56_tile_mailbox, int* __restrict__ task_gate_packed, int* __restrict__ result_chunk_total, int* __restrict__ result_chunk_tally, int* __restrict__ result_ovf_cursor, unsigned long long* __restrict__ signal_base_scratch, unsigned long long* __restrict__ dispatch_chunk_signal_base_scratch, unsigned long long* __restrict__ result_signal_base_scratch, unsigned long long* __restrict__ ack_signal_base_scratch, __nv_bfloat16* __restrict__ final_output, __nv_bfloat16* __restrict__ shared_out, unsigned int* __restrict__ pool_fp8_u32, unsigned int* __restrict__ pool_sf_u32, float* __restrict__ routing_weight_pool, int* __restrict__ meta_source_rank, int* __restrict__ meta_token, int* __restrict__ meta_slot, int* __restrict__ meta_result_index, int* __restrict__ expert_counts, int* __restrict__ owner_expert_route_counts, int* __restrict__ source_route_sum, int* __restrict__ source_expert_counts, int* __restrict__ expert_source_base, int* __restrict__ expert_source_offsets, int* __restrict__ source_expert_prefix, int* __restrict__ task_max_source, int* __restrict__ source_record_counts, int* __restrict__ source_route_counts, int* __restrict__ source_active_rows, int* __restrict__ expert_row_offsets, int* __restrict__ expert_task_base, int* __restrict__ expert_block_task, int* __restrict__ task_source_slot_base, int* __restrict__ tb_pub, int* __restrict__ expert_scatter_offsets, int* __restrict__ task_expert, int* __restrict__ task_source_rank, int* __restrict__ task_owner_rank, int* __restrict__ task_local_expert, int* __restrict__ task_pool_row, int* __restrict__ task_m_local, int* __restrict__ task_valid_m, int* __restrict__ task_rows_landed, int* __restrict__ total_valid_routes, int* __restrict__ total_padded_rows, int* __restrict__ total_m_tasks, int* __restrict__ histogram_done, int* __restrict__ prefix_done, int* __restrict__ w1_warp_done, int* __restrict__ w1_tiles_completed, int rank, int world_size, int active_rows, unsigned int epoch, ncclDevComm const* __restrict__ gin_dev_comm, uint8_t* __restrict__ dispatch_header_out, ncclWindow_t dispatch_header_out_window, uint8_t* __restrict__ dispatch_payload_out, ncclWindow_t dispatch_payload_out_window, uint8_t* __restrict__ dispatch_header_inbox, ncclWindow_t dispatch_header_inbox_window, uint8_t* __restrict__ dispatch_payload_inbox, ncclWindow_t dispatch_payload_inbox_window, uint8_t* __restrict__ result_out, ncclWindow_t result_out_window, uint8_t* __restrict__ result_inbox, ncclWindow_t result_inbox_window, uint8_t* __restrict__ ack_out, ncclWindow_t ack_out_window, uint8_t* __restrict__ ack_inbox, ncclWindow_t ack_inbox_window)
{
    const int tid = threadIdx.x;
    const int warp = make_warp_uniform(tid / 32);
    const int lane = tid % 32;

    extern __shared__ __align__(1024) char smem_raw[];
    int smem;
    smem = (int)(unsigned long long)__cvta_generic_to_shared(smem_raw);

    const int mbar_base = smem;
    #define w1_full_addr (mbar_base + 0)
    #define w1_empty_addr (mbar_base + 16)
    #define w2_full_addr (mbar_base + 32)
    #define w2_empty_addr (mbar_base + 48)
    #define epi_full_addr (mbar_base + 64)
    #define epi_free_addr (mbar_base + 72)

    const int bid = blockIdx.x;
    const int num_bids = gridDim.x;
    if (tid == 0) {
        asm volatile("fence.proxy.tensormap::generic.acquire.sys [%0], 128;" :: "l"((uint64_t)(W1_A)) : "memory");
        asm volatile("fence.proxy.tensormap::generic.acquire.sys [%0], 128;" :: "l"((uint64_t)(W1_A64)) : "memory");
        asm volatile("fence.proxy.tensormap::generic.acquire.sys [%0], 128;" :: "l"((uint64_t)(W1_B)) : "memory");
        asm volatile("fence.proxy.tensormap::generic.acquire.sys [%0], 128;" :: "l"((uint64_t)(W1_B8)) : "memory");
        asm volatile("fence.proxy.tensormap::generic.acquire.sys [%0], 128;" :: "l"((uint64_t)(W1_SFA)) : "memory");
        asm volatile("fence.proxy.tensormap::generic.acquire.sys [%0], 128;" :: "l"((uint64_t)(W1_SFA64)) : "memory");
        asm volatile("fence.proxy.tensormap::generic.acquire.sys [%0], 128;" :: "l"((uint64_t)(W1_SFB)) : "memory");
        asm volatile("fence.proxy.tensormap::generic.acquire.sys [%0], 128;" :: "l"((uint64_t)(W1_D)) : "memory");
        asm volatile("fence.proxy.tensormap::generic.acquire.sys [%0], 128;" :: "l"((uint64_t)(W2_A)) : "memory");
        asm volatile("fence.proxy.tensormap::generic.acquire.sys [%0], 128;" :: "l"((uint64_t)(W2_A64)) : "memory");
        asm volatile("fence.proxy.tensormap::generic.acquire.sys [%0], 128;" :: "l"((uint64_t)(W2_B)) : "memory");
        asm volatile("fence.proxy.tensormap::generic.acquire.sys [%0], 128;" :: "l"((uint64_t)(W2_B8)) : "memory");
        asm volatile("fence.proxy.tensormap::generic.acquire.sys [%0], 128;" :: "l"((uint64_t)(W2_SFA)) : "memory");
        asm volatile("fence.proxy.tensormap::generic.acquire.sys [%0], 128;" :: "l"((uint64_t)(W2_SFA64)) : "memory");
        asm volatile("fence.proxy.tensormap::generic.acquire.sys [%0], 128;" :: "l"((uint64_t)(W2_SFB)) : "memory");
        asm volatile("fence.proxy.tensormap::generic.acquire.sys [%0], 128;" :: "l"((uint64_t)(W2_D)) : "memory");
    }
    __syncthreads();


    // Kernel setup ops
    int* tb_scr = reinterpret_cast<int*>(smem_raw + 13312);
    const int tb_scr_addr = smem + 13312;
    __nv_bfloat16* d_stage = reinterpret_cast<__nv_bfloat16*>(smem_raw + 68608);
    const int d_stage_addr = smem + 68608;
    int* service_ready_chunks = reinterpret_cast<int*>(smem_raw + 1024);
    const int service_ready_chunks_addr = smem + 1024;
    int* svc_queue = reinterpret_cast<int*>(smem_raw + 5120);
    const int svc_queue_addr = smem + 5120;
    int* svc_qctl = reinterpret_cast<int*>(smem_raw + 9216);
    const int svc_qctl_addr = smem + 9216;
    uint8_t* w1_smem_a = reinterpret_cast<uint8_t*>(smem_raw + 1024);
    const int w1_smem_a_addr = smem + 1024;
    uint8_t* w1_smem_b = reinterpret_cast<uint8_t*>(smem_raw + 33792);
    const int w1_smem_b_addr = smem + 33792;
    unsigned int* w1_smem_sfa = reinterpret_cast<unsigned int*>(smem_raw + 66560);
    const int w1_smem_sfa_addr = smem + 66560;
    unsigned int* w1_smem_sfb = reinterpret_cast<unsigned int*>(smem_raw + 67584);
    const int w1_smem_sfb_addr = smem + 67584;

    // Mbarrier init (6 groups, 10 barriers)
    // Mbarriers at smem_raw[0..80)

    if (warp == 0) {
        uint32_t leader = elect_sync();
        if (leader) {
            // --- pipeline 'w1_pipe' ---
            // w1_full: 2 barriers, init_count=1
            mbarrier_init(smem + 0, 1);
            mbarrier_init(smem + 8, 1);
            // w1_empty: 2 barriers, init_count=8
            mbarrier_init(smem + 16, 8);
            mbarrier_init(smem + 24, 8);
            // --- pipeline 'w2_pipe' ---
            // w2_full: 2 barriers, init_count=1
            mbarrier_init(smem + 32, 1);
            mbarrier_init(smem + 40, 1);
            // w2_empty: 2 barriers, init_count=8
            mbarrier_init(smem + 48, 8);
            mbarrier_init(smem + 56, 8);
            // Each math / epilogue thread directly releases its d_stage accesses.
            // epi_full: 1 barriers, init_count=256
            mbarrier_init(smem + 64, 256);
            // epi_free: 1 barriers, init_count=96
            mbarrier_init(smem + 72, 96);
            asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
        }
    }

    __syncthreads();

    // === Task calls (dependency order) ===
    // CTA0 owns the queue; the reset grid barrier publishes these initial values.
    if (bid == 0) {
        #pragma unroll 1
        for (int reset_queue_slot = tid; reset_queue_slot < 1024; reset_queue_slot += 384) {
            svc_queue[reset_queue_slot] = 0;
        }
    }
    int reset_tid = bid * 384 + tid;
    int reset_threads = num_bids * 384;
    int _max_0 = ((world_size * active_rows * 6 + 8128) > (0) ? (world_size * active_rows * 6 + 8128) : (0));
    int _min_0 = ((_max_0) < (204736) ? (_max_0) : (204736));
    int reset_rows_bound = _min_0;
    int _max_1 = (((world_size * active_rows * 6 + 128 - 1) / 128 + 64 + (active_rows + 128 - 1) / 128) > (0) ? ((world_size * active_rows * 6 + 128 - 1) / 128 + 64 + (active_rows + 128 - 1) / 128) : (0));
    int _min_1 = ((_max_1) < (1664) ? (_max_1) : (1664));
    int reset_tasks_bound = _min_1;
    #pragma unroll 1
    for (int reset_peer = reset_tid; reset_peer < 4; reset_peer += reset_threads) {
        owner_record_counts[reset_peer] = 0;
        owner_route_counts[reset_peer] = 0;
        source_record_counts[reset_peer] = 0;
        source_route_counts[reset_peer] = 0;
        source_active_rows[reset_peer] = 0;
        source_route_sum[reset_peer] = 0;
        peer_phase_timestamps[reset_peer] = 0;
    }
    #pragma unroll 1
    for (int reset_owner_expert = reset_tid; reset_owner_expert < 256; reset_owner_expert += reset_threads) {
        owner_expert_route_counts[reset_owner_expert] = 0;
        source_expert_counts[reset_owner_expert] = 0;
        expert_source_base[reset_owner_expert] = 0;
        expert_source_offsets[reset_owner_expert] = 0;
        source_expert_prefix[reset_owner_expert] = 0;
        owner_minexp_record_base[reset_owner_expert] = 0;
        owner_minexp_record_cursor[reset_owner_expert] = 0;
    }
    #pragma unroll 1
    for (int reset_dispatch_chunk = reset_tid; reset_dispatch_chunk < 32; reset_dispatch_chunk += reset_threads) {
        {
            unsigned int* _gcr_p = reinterpret_cast<unsigned int*>(dispatch_chunk_scatter_counter) + (reset_dispatch_chunk);
            asm volatile("st.release.gpu.global.u32 [%0], %1;" : : "l"(_gcr_p), "r"(0u) : "memory");
        }
        dispatch_chunk_targets[reset_dispatch_chunk] = 0;
        {
            unsigned int* _gcr_p = reinterpret_cast<unsigned int*>(pull_chunk_arrived) + (reset_dispatch_chunk);
            asm volatile("st.release.gpu.global.u32 [%0], %1;" : : "l"(_gcr_p), "r"(0u) : "memory");
        }
        pull_request_scratch[reset_dispatch_chunk * 2] = 0;
        pull_request_scratch[reset_dispatch_chunk * 2 + 1] = 0;
        if (reset_dispatch_chunk < 4) {
            {
                unsigned int* _gcr_p = reinterpret_cast<unsigned int*>(result_owner_ready) + (reset_dispatch_chunk);
                asm volatile("st.release.gpu.global.u32 [%0], %1;" : : "l"(_gcr_p), "r"(0u) : "memory");
            }
            {
                unsigned int* _gcr_p = reinterpret_cast<unsigned int*>(result_owner_progress) + (reset_dispatch_chunk);
                asm volatile("st.release.gpu.global.u32 [%0], %1;" : : "l"(_gcr_p), "r"(0u) : "memory");
            }
        }
    }
    #pragma unroll 1
    for (int reset_route = reset_tid; reset_route < active_rows * 6; reset_route += reset_threads) {
        route_result_index[reset_route] = -1;
    }
    #pragma unroll 1
    for (int reset_row = reset_tid; reset_row < reset_rows_bound; reset_row += reset_threads) {
        routing_weight_pool[reset_row] = 0.0f;
        meta_source_rank[reset_row] = -1;
        meta_token[reset_row] = -1;
        meta_slot[reset_row] = -1;
        meta_result_index[reset_row] = -1;
    }
    int shared_fill_tasks = (active_rows + 128 - 1) / 128;
    int shared_fill_rows = shared_fill_tasks * 128;
    #pragma unroll 1
    for (int shared_word = reset_tid; shared_word < shared_fill_rows * 1024; shared_word += reset_threads) {
        int shared_word_value = 0;
        if (shared_word < active_rows * 1024) {
            shared_word_value = x_fp8_i32[shared_word];
        }
        pool_fp8_u32[209649664 + shared_word] = (unsigned int)shared_word_value;
    }
    #pragma unroll 1
    for (int shared_sf_pair = reset_tid; shared_sf_pair < 32 * shared_fill_rows; shared_sf_pair += reset_threads) {
        int shared_sf_word = shared_sf_pair / shared_fill_rows;
        int shared_sf_row = shared_sf_pair - shared_sf_word * shared_fill_rows;
        int shared_sf_value = 0;
        if (shared_sf_row < active_rows) {
            shared_sf_value = x_sf_i32[shared_sf_row * 32 + shared_sf_word];
        }
        pool_sf_u32[shared_sf_word * 212928 + 204736 + shared_sf_row] = (unsigned int)shared_sf_value;
    }
    #pragma unroll 1
    for (int shared_reset_row = reset_tid; shared_reset_row < shared_fill_rows; shared_reset_row += reset_threads) {
        float shared_row_weight = 0.0f;
        if (shared_reset_row < active_rows) {
            shared_row_weight = 1.0f;
        }
        routing_weight_pool[204736 + shared_reset_row] = shared_row_weight;
        meta_source_rank[204736 + shared_reset_row] = -1;
        meta_token[204736 + shared_reset_row] = -1;
        meta_slot[204736 + shared_reset_row] = -1;
        meta_result_index[204736 + shared_reset_row] = -1;
    }
    #pragma unroll 1
    for (int reset_expert = reset_tid; reset_expert < 64; reset_expert += reset_threads) {
        expert_counts[reset_expert] = 0;
        expert_row_offsets[reset_expert] = 0;
        expert_task_base[reset_expert] = 0;
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
        task_rows_landed[reset_task] = 0;
        task_max_source[reset_task] = -1;
        task_gate_packed[reset_task] = 0;
    }
    #pragma unroll 1
    for (int reset_w1_tile = reset_tid; reset_w1_tile < reset_tasks_bound * 32; reset_w1_tile += reset_threads) {
        w1_warp_done[reset_w1_tile] = 0;
    }
    #pragma unroll 1
    for (int reset_w2_tile = reset_tid; reset_w2_tile < reset_tasks_bound * 32; reset_w2_tile += reset_threads) {
        w2_warp_done[reset_w2_tile] = 0;
    }
    #pragma unroll 1
    for (int reset_w2_task = reset_tid; reset_w2_task < reset_tasks_bound; reset_w2_task += reset_threads) {
        {
            unsigned int* _gcr_p = reinterpret_cast<unsigned int*>(w2_task_counter) + (reset_w2_task);
            asm volatile("st.release.gpu.global.u32 [%0], %1;" : : "l"(_gcr_p), "r"(0u) : "memory");
        }
        {
            unsigned int* _gcr_p = reinterpret_cast<unsigned int*>(w1_task_counter) + (reset_w2_task);
            asm volatile("st.release.gpu.global.u32 [%0], %1;" : : "l"(_gcr_p), "r"(0u) : "memory");
        }
    }
    #pragma unroll 1
    for (int reset_chunk = reset_tid; reset_chunk < 3076; reset_chunk += reset_threads) {
        result_chunk_total[reset_chunk] = 0;
        result_chunk_tally[reset_chunk] = 0;
        result_ovf_cursor[reset_chunk] = 0;
    }
    #pragma unroll 1
    for (int reset_c56_slot = reset_tid; reset_c56_slot < 8192; reset_c56_slot += reset_threads) {
        c56_tile_mailbox[reset_c56_slot] = 0;
    }
    if (reset_tid == 0) {
        phase_timestamps[17] = 9223372036854775807;
        phase_timestamps[18] = 9223372036854775807;
        phase_timestamps[20] = 9223372036854775807;
        phase_timestamps[19] = 0;
        phase_timestamps[21] = 9223372036854775807;
        phase_timestamps[22] = 0;
        phase_timestamps[23] = 9223372036854775807;
        phase_timestamps[24] = 0;
        phase_timestamps[15] = 0;
        svc_qctl[0] = 0;
        svc_qctl[1] = 0;
        svc_qctl[2] = 0;
        svc_qctl[3] = 0;
        #pragma unroll 1
        for (int h29_reset_ctl = 4; h29_reset_ctl < 16; h29_reset_ctl++) {
            svc_qctl[h29_reset_ctl] = 0;
        }
        c56_claim_cursor[0] = 0;
        combine_claim_cursor[0] = 0;
        protocol_error[0] = 0;
        total_valid_routes[0] = 0;
        total_padded_rows[0] = 0;
        total_m_tasks[0] = (active_rows + 128 - 1) / 128;
        histogram_done[0] = 0;
        prefix_done[0] = 0;
        #pragma unroll
        for (int c24_r = 0; c24_r < 4; c24_r++) {
            {
                unsigned int* _gcr_p = reinterpret_cast<unsigned int*>(c24_front_sync) + (c24_r);
                asm volatile("st.release.gpu.global.u32 [%0], %1;" : : "l"(_gcr_p), "r"(0u) : "memory");
            }
        }
        w1_tiles_completed[0] = 0;
        requant_groups_done[0] = 0;
        w2_tiles_completed[0] = 0;
    }
    cooperative_groups::this_grid().sync();
    if (bid == 0 && tid == 0) {
        unsigned long long gtimer_0;
        asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(gtimer_0) :: "memory");
        phase_timestamps[0] = gtimer_0;
    }
    int slot = (int)(epoch & 1);
    int dispatch_chunk_min_records = 160;
    if (active_rows <= 512) {
        dispatch_chunk_min_records = 96;
    }
    int launch_valid = (int)(rank >= 0 && rank < world_size && world_size == 4 && active_rows >= 1 && active_rows <= 8192 && num_bids >= 2 && num_bids >= world_size);
    if (launch_valid == 0) {
        if (warp == 0) {
            if (elect_sync()) {
                atomicMax(&protocol_error[0], 1);
            }
        }
    }
    if (bid == 0) {
        #pragma unroll 1
        for (int credit_source = 0; credit_source < 4; credit_source++) {
            if (warp == 0) {
                if (elect_sync()) {
                    if (ack_signal_base_scratch[4] != 0 && credit_source < world_size) {
                        // gin_wait_signal: acquire, rolling 64-bit comparison
                        {
                            ncclGin __gin{*(gin_dev_comm), (int)(0)};
                            __gin.waitSignal(ncclCoopThread(), (ncclGinSignal_t)(8 + credit_source), (uint64_t)(ack_signal_base_scratch[credit_source] + 1), 64, cuda::memory_order_acquire);
                        }
                    }
                }
            }
        }
        if (warp == 0) {
            if (elect_sync()) {
                if (ack_signal_base_scratch[4] != 0) {
                    #pragma unroll 1
                    for (int ack_adv = 0; ack_adv < 4; ack_adv++) {
                        if (ack_adv < world_size) {
                            ack_signal_base_scratch[ack_adv] = ack_signal_base_scratch[ack_adv] + 1;
                        }
                    }
                }
            }
        }
        int h8_first_launch = (int)(ack_signal_base_scratch[4] == 0);
        __syncthreads();
        if (h8_first_launch != 0) {
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
                                _gin_signal_1 = __gin.readSignal((ncclGinSignal_t)(4 + source), 64, cuda::memory_order_acquire);
                            }
                            result_signal_base_scratch[source] = _gin_signal_1;
                            // gin_read_signal: acquire, 64-bit signal snapshot
                            uint64_t _gin_signal_2;
                            {
                                ncclGin __gin{*(gin_dev_comm), (int)(1)};
                                _gin_signal_2 = __gin.readSignal((ncclGinSignal_t)(44 + source), 64, cuda::memory_order_acquire);
                            }
                            result_signal_base_scratch[4 + source] = _gin_signal_2;
                            // gin_read_signal: acquire, 64-bit signal snapshot
                            uint64_t _gin_signal_3;
                            {
                                ncclGin __gin{*(gin_dev_comm), (int)(0)};
                                _gin_signal_3 = __gin.readSignal((ncclGinSignal_t)(8 + source), 64, cuda::memory_order_acquire);
                            }
                            ack_signal_base_scratch[source] = _gin_signal_3;
                        }
                    }
                }
            }
            #pragma unroll 1
            for (int chunk_signal = 0; chunk_signal < 32; chunk_signal++) {
                if (warp == 0) {
                    if (elect_sync()) {
                        if (chunk_signal / 8 < world_size) {
                            // gin_read_signal: acquire, 64-bit signal snapshot
                            uint64_t _gin_signal_4;
                            {
                                ncclGin __gin{*(gin_dev_comm), (int)(0)};
                                _gin_signal_4 = __gin.readSignal((ncclGinSignal_t)(12 + chunk_signal), 64, cuda::memory_order_acquire);
                            }
                            dispatch_chunk_signal_base_scratch[chunk_signal] = _gin_signal_4;
                        }
                    }
                }
            }
            if (warp == 0) {
                if (elect_sync()) {
                    ack_signal_base_scratch[4] = 1;
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
    }
    cooperative_groups::this_grid().sync();
    if (warp < 8) {
        int global_warp = bid * 8 + warp;
        int warps_per_grid = num_bids * 8;
        #pragma unroll 1
        for (int token = global_warp; token < active_rows; token += warps_per_grid) {
            int owner = lane;
            if (owner < world_size) {
                if (lane < 4) {
                    int route_count = 0;
                    int min_expert = 64;
                    #pragma unroll
                    for (int route_slot = 0; route_slot < 6; route_slot++) {
                        int pair = token * 6 + route_slot;
                        int expert = topk_idx_i32[pair * 2];
                        int expert_hi = topk_idx_i32[pair * 2 + 1];
                        int masked = (int)(expert == -1 && expert_hi == -1);
                        int valid = (int)(expert >= 0 && expert < world_size * 64 && expert_hi == 0);
                        if (valid == 0 && masked == 0) {
                            atomicMax(&protocol_error[0], 1);
                        }
                        if (valid != 0 && expert / 64 == owner) {
                            route_count = route_count + 1;
                            int _min_2 = ((min_expert) < (expert - owner * 64) ? (min_expert) : (expert - owner * 64));
                            min_expert = _min_2;
                            atomicAdd(&owner_expert_route_counts[owner * 64 + (expert - owner * 64)], 1);
                        }
                    }
                    if (route_count > 0) {
                        atomicAdd(&owner_minexp_record_base[owner * 64 + min_expert], 1);
                    }
                }
            }
        }
    }
    cooperative_groups::this_grid().sync();
    if (bid == 0 && tid < 4) {
        int prefix_owner = tid;
        int minexp_running = 0;
        #pragma unroll 1
        for (int prefix_expert = 0; prefix_expert < 64; prefix_expert++) {
            int minexp_count = owner_minexp_record_base[prefix_owner * 64 + prefix_expert];
            owner_minexp_record_base[prefix_owner * 64 + prefix_expert] = minexp_running;
            minexp_running = minexp_running + minexp_count;
        }
        if (minexp_running > 8192) {
            atomicMax(&protocol_error[0], 1);
        }
        owner_record_counts[prefix_owner] = minexp_running;
    }
    __threadfence();
    cooperative_groups::this_grid().sync();
    if (warp < 8) {
        int index_global_warp = bid * 8 + warp;
        int index_warps_per_grid = num_bids * 8;
        #pragma unroll 1
        for (int index_token = index_global_warp; index_token < active_rows; index_token += index_warps_per_grid) {
            int index_owner = lane;
            if (index_owner < world_size) {
                if (lane < 4) {
                    int index_route_count = 0;
                    int index_min_expert = 64;
                    #pragma unroll
                    for (int index_route_slot = 0; index_route_slot < 6; index_route_slot++) {
                        int index_pair = index_token * 6 + index_route_slot;
                        int index_expert = topk_idx_i32[index_pair * 2];
                        int index_expert_hi = topk_idx_i32[index_pair * 2 + 1];
                        int index_valid = (int)(index_expert >= 0 && index_expert < world_size * 64 && index_expert_hi == 0);
                        if (index_valid != 0 && index_expert / 64 == index_owner) {
                            index_route_count = index_route_count + 1;
                            int _min_3 = ((index_min_expert) < (index_expert - index_owner * 64) ? (index_min_expert) : (index_expert - index_owner * 64));
                            index_min_expert = _min_3;
                        }
                    }
                    int claim = -1;
                    int route_base = -1;
                    if (index_route_count > 0) {
                        int sort_slot = index_owner * 64 + index_min_expert;
                        int _atomic_old_0 = atomicAdd(&owner_minexp_record_cursor[sort_slot], 1);
                        claim = owner_minexp_record_base[sort_slot] + _atomic_old_0;
                        if (claim < 0 || claim >= owner_record_counts[index_owner]) {
                            atomicMax(&protocol_error[0], 1);
                            claim = -1;
                        }
                        int _atomic_old_1 = atomicAdd(&owner_route_counts[index_owner], index_route_count);
                        route_base = _atomic_old_1;
                        if (route_base + index_route_count > 49152) {
                            atomicMax(&protocol_error[0], 1);
                            claim = -1;
                        }
                    }
                    if (claim >= 0) {
                        int sorted_index = index_owner * 8192 + claim;
                        sorted_record_token[sorted_index] = index_token;
                        sorted_record_route_base[sorted_index] = route_base;
                        int index_write_route = 0;
                        #pragma unroll
                        for (int index_route_slot_2 = 0; index_route_slot_2 < 6; index_route_slot_2++) {
                            int index_pair_2 = index_token * 6 + index_route_slot_2;
                            int index_expert_2 = topk_idx_i32[index_pair_2 * 2];
                            int index_expert_hi_2 = topk_idx_i32[index_pair_2 * 2 + 1];
                            int index_valid_2 = (int)(index_expert_2 >= 0 && index_expert_2 < world_size * 64 && index_expert_hi_2 == 0);
                            if (index_valid_2 != 0 && index_expert_2 / 64 == index_owner) {
                                route_result_index[index_pair_2] = route_base + index_write_route;
                                index_write_route = index_write_route + 1;
                            }
                        }
                    }
                }
            }
        }
    }
    __threadfence();
    cooperative_groups::this_grid().sync();
    if (bid == 0 && tid == 0) {
        unsigned long long gtimer_0_1;
        asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(gtimer_0_1) :: "memory");
        phase_timestamps[1] = gtimer_0_1;
    }
    if (bid == 0 && warp == 0) {
        int c24_shared_tasks = (active_rows + 128 - 1) / 128;
        #pragma unroll 1
        for (int c24_st = lane; c24_st < c24_shared_tasks; c24_st += 32) {
            if (c24_st < 1664) {
                task_max_source[c24_st] = -1;
                task_expert[c24_st] = world_size * 64;
                task_source_rank[c24_st] = 0;
                task_owner_rank[c24_st] = rank;
                task_local_expert[c24_st] = 64;
                task_pool_row[c24_st] = 204736 + c24_st * 128;
                task_m_local[c24_st] = c24_st * 128;
                int _min_4 = ((128) < (active_rows - c24_st * 128) ? (128) : (active_rows - c24_st * 128));
                task_valid_m[c24_st] = _min_4;
                task_gate_packed[c24_st] = 0;
            }
        }
        __threadfence();
    }
    if (bid == 0) {
        int peer_i = warp;
        if (peer_i < world_size) {
            int peer = peer_i + rank + 1;
            if (peer >= world_size) {
                peer = peer - world_size;
            }
            int local_header_word = (peer * 2 + slot) * 136;
            int local_header_byte = local_header_word * 4;
            int remote_header_byte = (rank * 2 + slot) * 544;
            if (elect_sync()) {
                int count = owner_record_counts[peer];
                int peer_route_count = owner_route_counts[peer];
                int _max_2 = ((count) > (0) ? (count) : (0));
                int _min_5 = ((_max_2) < (8192) ? (_max_2) : (8192));
                int safe_count = _min_5;
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
                for (int header_expert = 0; header_expert < 64; header_expert++) {
                    *(reinterpret_cast<int*>(reinterpret_cast<int*>(dispatch_header_out) + (local_header_word + 8 + header_expert)) + (0)) = owner_expert_route_counts[peer * 64 + header_expert];
                }
                #pragma unroll 1
                for (int header_prefix_expert = 0; header_prefix_expert < 64; header_prefix_expert++) {
                    int header_prefix_records = safe_count;
                    if (header_prefix_expert < 63) {
                        header_prefix_records = owner_minexp_record_base[peer * 64 + header_prefix_expert + 1];
                    }
                    *(reinterpret_cast<int*>(reinterpret_cast<int*>(dispatch_header_out) + (local_header_word + 8 + 64 + header_prefix_expert)) + (0)) = header_prefix_records;
                }
                __threadfence_system();
                // gin_put_signal_add: strong remote completion on context 0
                {
                    ncclGin __gin{*(gin_dev_comm), (int)(0)};
                    __gin.put(ncclTeamWorld(*(gin_dev_comm)), (int)(peer), dispatch_header_inbox_window, (size_t)(remote_header_byte), dispatch_header_out_window, (size_t)(local_header_byte), (size_t)(544),
                        ncclGin_StrongSignalAdd{(ncclGinSignal_t)(rank), (uint64_t)(1)}, ncclGin_None{}, ncclCoopThread());
                }
            }
        }
    }
    cooperative_groups::this_grid().sync();
    int chunk_pack_owner = bid % world_size;
    int chunk_pack_owner_cta = bid / world_size;
    int chunk_pack_owner_ctas = (num_bids - chunk_pack_owner + world_size - 1) / world_size;
    int pull_source = world_size;
    if (bid < world_size && bid != rank && warp == 9) {
        pull_source = bid;
    }
    int pull_engaged = (int)(pull_source < world_size);
    int pull_header_seen = 0;
    int pull_count = 0;
    int pull_q = 1;
    int pull_chunks = 0;
    int pull_issued = 0;
    #pragma unroll 1
    for (int chunk_pack = 0; chunk_pack < 8; chunk_pack++) {
        if (warp < 8) {
            if (chunk_pack_owner < world_size) {
                int pack_owner = chunk_pack_owner;
                int _max_3 = ((owner_record_counts[pack_owner]) > (0) ? (owner_record_counts[pack_owner]) : (0));
                int _min_6 = ((_max_3) < (8192) ? (_max_3) : (8192));
                int pack_owner_count = _min_6;
                int _max_4 = (((pack_owner_count + 8 - 1) / 8) > (dispatch_chunk_min_records) ? ((pack_owner_count + 8 - 1) / 8) : (dispatch_chunk_min_records));
                int pack_chunk_q = _max_4;
                int pack_chunk_lo = chunk_pack * pack_chunk_q;
                int _min_7 = ((pack_chunk_lo + pack_chunk_q) < (pack_owner_count) ? (pack_chunk_lo + pack_chunk_q) : (pack_owner_count));
                int pack_chunk_hi = _min_7;
                #pragma unroll 1
                for (int pack_record = pack_chunk_lo + chunk_pack_owner_cta * 8 + warp; pack_record < pack_chunk_hi; pack_record += chunk_pack_owner_ctas * 8) {
                    int sorted_index_2 = pack_owner * 8192 + pack_record;
                    int pack_token = sorted_record_token[sorted_index_2];
                    int pack_route_base = sorted_record_route_base[sorted_index_2];
                    unsigned long long record_byte = (unsigned long long)(pack_owner * 2 + slot) * 35651584 + (unsigned long long)pack_record * 4352;
                    unsigned long long record_word = record_byte / 4;
                    int pk_expert = -1;
                    int pk_expert_hi = -1;
                    float pk_weight = 0.0f;
                    if (lane < 6) {
                        int pk_pair = pack_token * 6 + lane;
                        pk_expert = topk_idx_i32[pk_pair * 2];
                        pk_expert_hi = topk_idx_i32[pk_pair * 2 + 1];
                        pk_weight = topk_weights[pk_pair];
                    }
                    int pk_mine = (int)(lane < 6 && pk_expert >= 0 && pk_expert < world_size * 64 && pk_expert_hi == 0 && pk_expert / 64 == pack_owner);
                    int pk_pos = 0;
                    int pk_count = 0;
                    int _shfl_0 = __shfl_sync(0xFFFFFFFF, pk_mine, 0);
                    int pk_flag = _shfl_0;
                    if (lane > 0) {
                        pk_pos = pk_pos + pk_flag;
                    }
                    pk_count = pk_count + pk_flag;
                    int _shfl_1 = __shfl_sync(0xFFFFFFFF, pk_mine, 1);
                    int pk_flag_0 = _shfl_1;
                    if (lane > 1) {
                        pk_pos = pk_pos + pk_flag_0;
                    }
                    pk_count = pk_count + pk_flag_0;
                    int _shfl_2 = __shfl_sync(0xFFFFFFFF, pk_mine, 2);
                    int pk_flag_1 = _shfl_2;
                    if (lane > 2) {
                        pk_pos = pk_pos + pk_flag_1;
                    }
                    pk_count = pk_count + pk_flag_1;
                    int _shfl_3 = __shfl_sync(0xFFFFFFFF, pk_mine, 3);
                    int pk_flag_2 = _shfl_3;
                    if (lane > 3) {
                        pk_pos = pk_pos + pk_flag_2;
                    }
                    pk_count = pk_count + pk_flag_2;
                    int _shfl_4 = __shfl_sync(0xFFFFFFFF, pk_mine, 4);
                    int pk_flag_3 = _shfl_4;
                    if (lane > 4) {
                        pk_pos = pk_pos + pk_flag_3;
                    }
                    pk_count = pk_count + pk_flag_3;
                    int _shfl_5 = __shfl_sync(0xFFFFFFFF, pk_mine, 5);
                    int pk_flag_4 = _shfl_5;
                    if (lane > 5) {
                        pk_pos = pk_pos + pk_flag_4;
                    }
                    pk_count = pk_count + pk_flag_4;
                    if (pk_mine != 0) {
                        *(reinterpret_cast<int*>(reinterpret_cast<int*>(dispatch_payload_out) + (record_word + 2 + pk_pos)) + (0)) = pk_expert - pack_owner * 64;
                        *(reinterpret_cast<int*>(reinterpret_cast<int*>(dispatch_payload_out) + (record_word + 8 + pk_pos)) + (0)) = lane;
                        *(reinterpret_cast<float*>(reinterpret_cast<float*>(dispatch_payload_out) + (record_word + 14 + pk_pos)) + (0)) = pk_weight;
                    }
                    if (lane == 0) {
                        if (pack_token < 0 || pack_token >= active_rows) {
                            atomicMax(&protocol_error[0], 1);
                        }
                        if (pk_count <= 0 || pack_route_base < 0 || pack_route_base + pk_count > 49152) {
                            atomicMax(&protocol_error[0], 1);
                        }
                        *(reinterpret_cast<int*>(reinterpret_cast<int*>(dispatch_payload_out) + record_word) + (0)) = pack_token;
                        *(reinterpret_cast<int*>(reinterpret_cast<int*>(dispatch_payload_out) + (record_word + 1)) + (0)) = pk_count;
                        *(reinterpret_cast<int*>(reinterpret_cast<int*>(dispatch_payload_out) + (record_word + 20)) + (0)) = rank;
                        *(reinterpret_cast<int*>(reinterpret_cast<int*>(dispatch_payload_out) + (record_word + 21)) + (0)) = 1347571524;
                        *(reinterpret_cast<int*>(reinterpret_cast<int*>(dispatch_payload_out) + (record_word + 22)) + (0)) = pack_route_base;
                    }
                    unsigned long long src_activation = (unsigned long long)pack_token * 1024;
                    unsigned long long dst_activation = record_word + 32;
                    unsigned long long pk_lane_word = (unsigned long long)(lane * 4);
                    int _vec_load_0[4];
                    {
                        int4 _iv4 = *reinterpret_cast<const int4*>(x_fp8_i32 + (src_activation + pk_lane_word) + 0);
                        _vec_load_0[0 + 0] = _iv4.x;
                        _vec_load_0[0 + 1] = _iv4.y;
                        _vec_load_0[0 + 2] = _iv4.z;
                        _vec_load_0[0 + 3] = _iv4.w;
                    }
                    int _vec_load_1[4];
                    {
                        int4 _iv4 = *reinterpret_cast<const int4*>(x_fp8_i32 + (src_activation + pk_lane_word + 128) + 0);
                        _vec_load_1[0 + 0] = _iv4.x;
                        _vec_load_1[0 + 1] = _iv4.y;
                        _vec_load_1[0 + 2] = _iv4.z;
                        _vec_load_1[0 + 3] = _iv4.w;
                    }
                    int _vec_load_2[4];
                    {
                        int4 _iv4 = *reinterpret_cast<const int4*>(x_fp8_i32 + (src_activation + pk_lane_word + 256) + 0);
                        _vec_load_2[0 + 0] = _iv4.x;
                        _vec_load_2[0 + 1] = _iv4.y;
                        _vec_load_2[0 + 2] = _iv4.z;
                        _vec_load_2[0 + 3] = _iv4.w;
                    }
                    int _vec_load_3[4];
                    {
                        int4 _iv4 = *reinterpret_cast<const int4*>(x_fp8_i32 + (src_activation + pk_lane_word + 384) + 0);
                        _vec_load_3[0 + 0] = _iv4.x;
                        _vec_load_3[0 + 1] = _iv4.y;
                        _vec_load_3[0 + 2] = _iv4.z;
                        _vec_load_3[0 + 3] = _iv4.w;
                    }
                    int _vec_load_4[4];
                    {
                        int4 _iv4 = *reinterpret_cast<const int4*>(x_fp8_i32 + (src_activation + pk_lane_word + 512) + 0);
                        _vec_load_4[0 + 0] = _iv4.x;
                        _vec_load_4[0 + 1] = _iv4.y;
                        _vec_load_4[0 + 2] = _iv4.z;
                        _vec_load_4[0 + 3] = _iv4.w;
                    }
                    int _vec_load_5[4];
                    {
                        int4 _iv4 = *reinterpret_cast<const int4*>(x_fp8_i32 + (src_activation + pk_lane_word + 640) + 0);
                        _vec_load_5[0 + 0] = _iv4.x;
                        _vec_load_5[0 + 1] = _iv4.y;
                        _vec_load_5[0 + 2] = _iv4.z;
                        _vec_load_5[0 + 3] = _iv4.w;
                    }
                    int _vec_load_6[4];
                    {
                        int4 _iv4 = *reinterpret_cast<const int4*>(x_fp8_i32 + (src_activation + pk_lane_word + 768) + 0);
                        _vec_load_6[0 + 0] = _iv4.x;
                        _vec_load_6[0 + 1] = _iv4.y;
                        _vec_load_6[0 + 2] = _iv4.z;
                        _vec_load_6[0 + 3] = _iv4.w;
                    }
                    int _vec_load_7[4];
                    {
                        int4 _iv4 = *reinterpret_cast<const int4*>(x_fp8_i32 + (src_activation + pk_lane_word + 896) + 0);
                        _vec_load_7[0 + 0] = _iv4.x;
                        _vec_load_7[0 + 1] = _iv4.y;
                        _vec_load_7[0 + 2] = _iv4.z;
                        _vec_load_7[0 + 3] = _iv4.w;
                    }
                    {
                        int4 _iv4 = make_int4(_vec_load_0[0 + 0], _vec_load_0[0 + 1], _vec_load_0[0 + 2], _vec_load_0[0 + 3]);
                        *reinterpret_cast<int4*>(reinterpret_cast<int*>(dispatch_payload_out) + (dst_activation + pk_lane_word) + 0) = _iv4;
                    }
                    {
                        int4 _iv4 = make_int4(_vec_load_1[0 + 0], _vec_load_1[0 + 1], _vec_load_1[0 + 2], _vec_load_1[0 + 3]);
                        *reinterpret_cast<int4*>(reinterpret_cast<int*>(dispatch_payload_out) + (dst_activation + pk_lane_word + 128) + 0) = _iv4;
                    }
                    {
                        int4 _iv4 = make_int4(_vec_load_2[0 + 0], _vec_load_2[0 + 1], _vec_load_2[0 + 2], _vec_load_2[0 + 3]);
                        *reinterpret_cast<int4*>(reinterpret_cast<int*>(dispatch_payload_out) + (dst_activation + pk_lane_word + 256) + 0) = _iv4;
                    }
                    {
                        int4 _iv4 = make_int4(_vec_load_3[0 + 0], _vec_load_3[0 + 1], _vec_load_3[0 + 2], _vec_load_3[0 + 3]);
                        *reinterpret_cast<int4*>(reinterpret_cast<int*>(dispatch_payload_out) + (dst_activation + pk_lane_word + 384) + 0) = _iv4;
                    }
                    {
                        int4 _iv4 = make_int4(_vec_load_4[0 + 0], _vec_load_4[0 + 1], _vec_load_4[0 + 2], _vec_load_4[0 + 3]);
                        *reinterpret_cast<int4*>(reinterpret_cast<int*>(dispatch_payload_out) + (dst_activation + pk_lane_word + 512) + 0) = _iv4;
                    }
                    {
                        int4 _iv4 = make_int4(_vec_load_5[0 + 0], _vec_load_5[0 + 1], _vec_load_5[0 + 2], _vec_load_5[0 + 3]);
                        *reinterpret_cast<int4*>(reinterpret_cast<int*>(dispatch_payload_out) + (dst_activation + pk_lane_word + 640) + 0) = _iv4;
                    }
                    {
                        int4 _iv4 = make_int4(_vec_load_6[0 + 0], _vec_load_6[0 + 1], _vec_load_6[0 + 2], _vec_load_6[0 + 3]);
                        *reinterpret_cast<int4*>(reinterpret_cast<int*>(dispatch_payload_out) + (dst_activation + pk_lane_word + 768) + 0) = _iv4;
                    }
                    {
                        int4 _iv4 = make_int4(_vec_load_7[0 + 0], _vec_load_7[0 + 1], _vec_load_7[0 + 2], _vec_load_7[0 + 3]);
                        *reinterpret_cast<int4*>(reinterpret_cast<int*>(dispatch_payload_out) + (dst_activation + pk_lane_word + 896) + 0) = _iv4;
                    }
                    unsigned long long src_sf = (unsigned long long)pack_token * 32;
                    unsigned long long dst_sf = record_word + 1056;
                    #pragma unroll 1
                    for (int sf_word = lane; sf_word < 32; sf_word += 32) {
                        *(reinterpret_cast<int*>(reinterpret_cast<int*>(dispatch_payload_out) + (dst_sf + (unsigned long long)sf_word)) + (0)) = x_sf_i32[src_sf + (unsigned long long)sf_word];
                    }
                    __syncwarp();
                }
            }
        }
        __threadfence_system();
        cooperative_groups::this_grid().sync();
        if (bid == 0) {
            int peer_2_i = warp;
            if (peer_2_i < world_size) {
                int peer_2 = peer_2_i + rank + 1;
                if (peer_2 >= world_size) {
                    peer_2 = peer_2 - world_size;
                }
                if (elect_sync()) {
                    int payload_count = owner_record_counts[peer_2];
                    int _max_5 = ((payload_count) > (0) ? (payload_count) : (0));
                    int _min_8 = ((_max_5) < (8192) ? (_max_5) : (8192));
                    int payload_safe_count = _min_8;
                    int _max_6 = (((payload_safe_count + 8 - 1) / 8) > (dispatch_chunk_min_records) ? ((payload_safe_count + 8 - 1) / 8) : (dispatch_chunk_min_records));
                    int chunk_q = _max_6;
                    int chunk_lo = chunk_pack * chunk_q;
                    if (chunk_lo < payload_safe_count) {
                        int _min_9 = ((chunk_q) < (payload_safe_count - chunk_lo) ? (chunk_q) : (payload_safe_count - chunk_lo));
                        int chunk_records = _min_9;
                        unsigned long long local_payload_byte = (unsigned long long)(peer_2 * 2 + slot) * 35651584 + (unsigned long long)chunk_lo * 4352;
                        unsigned long long remote_payload_byte = (unsigned long long)(rank * 2 + slot) * 35651584 + (unsigned long long)chunk_lo * 4352;
                        int put_bytes = 4;
                        if (peer_2 != rank) {
                            put_bytes = chunk_records * 4352;
                        }
                        // gin_put_signal_add: strong remote completion on context 0
                        {
                            ncclGin __gin{*(gin_dev_comm), (int)(0)};
                            __gin.put(ncclTeamWorld(*(gin_dev_comm)), (int)(peer_2), dispatch_payload_inbox_window, (size_t)(remote_payload_byte), dispatch_payload_out_window, (size_t)(local_payload_byte), (size_t)(put_bytes),
                                ncclGin_StrongSignalAdd{(ncclGinSignal_t)(12 + rank * 8 + chunk_pack), (uint64_t)(1)}, ncclGin_None{}, ncclCoopThread());
                        }
                    }
                }
            }
        }
    }
    if (bid == 0) {
        if (warp == 0) {
            if (elect_sync()) {
                unsigned long long gtimer_0_2;
                asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(gtimer_0_2) :: "memory");
                phase_timestamps[2] = gtimer_0_2;
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
        if (warp == 0) {
            if (elect_sync()) {
                unsigned long long gtimer_0_3;
                asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(gtimer_0_3) :: "memory");
                phase_timestamps[3] = gtimer_0_3;
            }
        }
    }
    int c24_front = (int)(bid == 0 || warp >= 9);
    int c24_threads = 96;
    int c24_tid = 384 + (bid - 1) * 96 + (warp - 9) * 32 + lane;
    int c24_leader = (int)(warp == 9 && lane == 0);
    if (bid == 0) {
        c24_threads = 384;
        c24_tid = tid;
        c24_leader = (int)(tid == 0);
    }
    int c24_all = 384 + (num_bids - 1) * 96;
    if (c24_front != 0) {
        int grid_tid = c24_tid;
        int grid_threads = c24_all;
        #pragma unroll 1
        for (int source_3 = grid_tid; source_3 < world_size; source_3 += grid_threads) {
            int header_word = (source_3 * 2 + slot) * 136;
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
        __syncwarp();
        asm volatile("barrier.sync 13, %0;" :: "r"(c24_threads) : "memory");
        if (c24_leader != 0) {
            {
                unsigned int* _gc_p = reinterpret_cast<unsigned int*>(c24_front_sync) + (0);
                unsigned int _gc_old;
                asm volatile("atom.release.gpu.global.add.u32 %0, [%1], 1;" : "=r"(_gc_old) : "l"(_gc_p) : "memory");
            }
            {
                unsigned int* _gca_p = reinterpret_cast<unsigned int*>(c24_front_sync) + (0);
                while (true) {
                    unsigned int _gca_v;
                    asm volatile("ld.acquire.gpu.global.u32 %0, [%1];" : "=r"(_gca_v) : "l"(_gca_p));
                    if (_gca_v >= (unsigned int)(num_bids)) break;
                }
            }
        }
        __syncwarp();
        asm volatile("barrier.sync 13, %0;" :: "r"(c24_threads) : "memory");
        #pragma unroll 1
        for (int hist_pair = grid_tid; hist_pair < 256; hist_pair += grid_threads) {
            int hist_source = hist_pair / 64;
            int hist_expert = hist_pair - hist_source * 64;
            if (hist_source < world_size) {
                int hist_header_word = (hist_source * 2 + slot) * 136;
                int hist_count = reinterpret_cast<const int*>(reinterpret_cast<int*>(dispatch_header_inbox) + (hist_header_word + 8 + hist_expert))[0];
                int hist_ok = (int)(hist_count >= 0 && hist_count <= source_route_counts[hist_source]);
                if (hist_ok == 0) {
                    atomicMax(&protocol_error[0], 1);
                } else {
                    source_expert_counts[hist_pair] = hist_count;
                }
                int hist_prefix = reinterpret_cast<const int*>(reinterpret_cast<int*>(dispatch_header_inbox) + (hist_header_word + 8 + 64 + hist_expert))[0];
                int hist_prefix_prev = 0;
                if (hist_expert > 0) {
                    hist_prefix_prev = reinterpret_cast<const int*>(reinterpret_cast<int*>(dispatch_header_inbox) + (hist_header_word + 8 + 64 + hist_expert - 1))[0];
                }
                int hist_prefix_ok = (int)(hist_prefix >= hist_prefix_prev && hist_prefix >= 0 && hist_prefix <= source_record_counts[hist_source] && (hist_expert < 63 || hist_prefix == source_record_counts[hist_source]));
                if (hist_prefix_ok == 0) {
                    atomicMax(&protocol_error[0], 1);
                }
                if (hist_ok != 0 && hist_count > 0) {
                    atomicAdd(&expert_counts[hist_expert], hist_count);
                    atomicAdd(&source_route_sum[hist_source], hist_count);
                }
            }
        }
        if (c24_leader != 0) {
            atomicAdd(&histogram_done[0], 1);
        }
        __syncwarp();
        asm volatile("barrier.sync 13, %0;" :: "r"(c24_threads) : "memory");
        if (c24_leader != 0) {
            {
                unsigned int* _gc_p = reinterpret_cast<unsigned int*>(c24_front_sync) + (1);
                unsigned int _gc_old;
                asm volatile("atom.release.gpu.global.add.u32 %0, [%1], 1;" : "=r"(_gc_old) : "l"(_gc_p) : "memory");
            }
            {
                unsigned int* _gca_p = reinterpret_cast<unsigned int*>(c24_front_sync) + (1);
                while (true) {
                    unsigned int _gca_v;
                    asm volatile("ld.acquire.gpu.global.u32 %0, [%1];" : "=r"(_gca_v) : "l"(_gca_p));
                    if (_gca_v >= (unsigned int)(num_bids)) break;
                }
            }
        }
        __syncwarp();
        asm volatile("barrier.sync 13, %0;" :: "r"(c24_threads) : "memory");
        if (bid == 0 && tid == 0) {
            unsigned long long gtimer_0_4;
            asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(gtimer_0_4) :: "memory");
            phase_timestamps[4] = gtimer_0_4;
            unsigned long long gtimer_1;
            asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(gtimer_1) :: "memory");
            phase_timestamps[5] = gtimer_1;
        }
        if (bid == 0 && warp == 0) {
            int tb_lane = lane;
            if (tb_lane == 0) {
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
                #pragma unroll 1
                for (int target_source = 0; target_source < 4; target_source++) {
                    int target_count = 0;
                    if (target_source < world_size) {
                        target_count = source_record_counts[target_source];
                    }
                    int _max_7 = (((target_count + 8 - 1) / 8) > (dispatch_chunk_min_records) ? ((target_count + 8 - 1) / 8) : (dispatch_chunk_min_records));
                    int target_q = _max_7;
                    #pragma unroll 1
                    for (int target_chunk = 0; target_chunk < 8; target_chunk++) {
                        int target_lo = target_chunk * target_q;
                        int _max_8 = ((target_count - target_lo) > (0) ? (target_count - target_lo) : (0));
                        int _min_10 = ((_max_8) < (target_q) ? (_max_8) : (target_q));
                        int target_records = _min_10;
                        dispatch_chunk_targets[target_source * 8 + target_chunk] = target_records;
                    }
                }
            }
            int build_shared_tasks = (active_rows + 128 - 1) / 128;
            #pragma unroll 1
            for (int shared_task = tb_lane; shared_task < build_shared_tasks; shared_task += 32) {
                if (shared_task < 1664) {
                    task_max_source[shared_task] = -1;
                    task_expert[shared_task] = world_size * 64;
                    task_source_rank[shared_task] = 0;
                    task_owner_rank[shared_task] = rank;
                    task_local_expert[shared_task] = 64;
                    task_pool_row[shared_task] = 204736 + shared_task * 128;
                    task_m_local[shared_task] = shared_task * 128;
                    int _min_11 = ((128) < (active_rows - shared_task * 128) ? (128) : (active_rows - shared_task * 128));
                    task_valid_m[shared_task] = _min_11;
                    task_gate_packed[shared_task] = 0;
                } else {
                    atomicMax(&protocol_error[0], 1);
                }
            }
            int tb_carry_padded = 0;
            int tb_carry_tasks = 0;
            int tb_carry_count = 0;
            int tb_tot[9];
            #pragma unroll
            for (int tb_k0 = 0; tb_k0 < 9; tb_k0++) {
                tb_tot[tb_k0] = 0;
            }
            #pragma unroll 1
            for (int tb_z0 = tb_lane; tb_z0 < 36; tb_z0 += 32) {
                tb_scr[4608 + tb_z0] = 0;
            }
            __syncwarp();
            #pragma unroll
            for (int tb_pass = 0; tb_pass < 2; tb_pass++) {
                int tb_e = tb_pass * 32 + tb_lane;
                int tb_count = expert_counts[tb_e];
                int tb_padded = (tb_count + 128 - 1) / 128 * 128;
                int tb_tasks = tb_padded / 128;
                int tb_padded_incl = tb_padded;
                int tb_tasks_incl = tb_tasks;
                int tb_count_incl = tb_count;
                int _shfl_up_0 = __shfl_up_sync(0xFFFFFFFF, tb_padded_incl, 1, 32);
                int tb_up_p = _shfl_up_0;
                int _shfl_up_1 = __shfl_up_sync(0xFFFFFFFF, tb_tasks_incl, 1, 32);
                int tb_up_t = _shfl_up_1;
                int _shfl_up_2 = __shfl_up_sync(0xFFFFFFFF, tb_count_incl, 1, 32);
                int tb_up_c = _shfl_up_2;
                if (tb_lane >= 1) {
                    tb_padded_incl = tb_padded_incl + tb_up_p;
                    tb_tasks_incl = tb_tasks_incl + tb_up_t;
                    tb_count_incl = tb_count_incl + tb_up_c;
                }
                int _shfl_up_3 = __shfl_up_sync(0xFFFFFFFF, tb_padded_incl, 2, 32);
                int tb_up_p_0 = _shfl_up_3;
                int _shfl_up_4 = __shfl_up_sync(0xFFFFFFFF, tb_tasks_incl, 2, 32);
                int tb_up_t_1 = _shfl_up_4;
                int _shfl_up_5 = __shfl_up_sync(0xFFFFFFFF, tb_count_incl, 2, 32);
                int tb_up_c_2 = _shfl_up_5;
                if (tb_lane >= 2) {
                    tb_padded_incl = tb_padded_incl + tb_up_p_0;
                    tb_tasks_incl = tb_tasks_incl + tb_up_t_1;
                    tb_count_incl = tb_count_incl + tb_up_c_2;
                }
                int _shfl_up_6 = __shfl_up_sync(0xFFFFFFFF, tb_padded_incl, 4, 32);
                int tb_up_p_3 = _shfl_up_6;
                int _shfl_up_7 = __shfl_up_sync(0xFFFFFFFF, tb_tasks_incl, 4, 32);
                int tb_up_t_4 = _shfl_up_7;
                int _shfl_up_8 = __shfl_up_sync(0xFFFFFFFF, tb_count_incl, 4, 32);
                int tb_up_c_5 = _shfl_up_8;
                if (tb_lane >= 4) {
                    tb_padded_incl = tb_padded_incl + tb_up_p_3;
                    tb_tasks_incl = tb_tasks_incl + tb_up_t_4;
                    tb_count_incl = tb_count_incl + tb_up_c_5;
                }
                int _shfl_up_9 = __shfl_up_sync(0xFFFFFFFF, tb_padded_incl, 8, 32);
                int tb_up_p_6 = _shfl_up_9;
                int _shfl_up_10 = __shfl_up_sync(0xFFFFFFFF, tb_tasks_incl, 8, 32);
                int tb_up_t_7 = _shfl_up_10;
                int _shfl_up_11 = __shfl_up_sync(0xFFFFFFFF, tb_count_incl, 8, 32);
                int tb_up_c_8 = _shfl_up_11;
                if (tb_lane >= 8) {
                    tb_padded_incl = tb_padded_incl + tb_up_p_6;
                    tb_tasks_incl = tb_tasks_incl + tb_up_t_7;
                    tb_count_incl = tb_count_incl + tb_up_c_8;
                }
                int _shfl_up_12 = __shfl_up_sync(0xFFFFFFFF, tb_padded_incl, 16, 32);
                int tb_up_p_9 = _shfl_up_12;
                int _shfl_up_13 = __shfl_up_sync(0xFFFFFFFF, tb_tasks_incl, 16, 32);
                int tb_up_t_10 = _shfl_up_13;
                int _shfl_up_14 = __shfl_up_sync(0xFFFFFFFF, tb_count_incl, 16, 32);
                int tb_up_c_11 = _shfl_up_14;
                if (tb_lane >= 16) {
                    tb_padded_incl = tb_padded_incl + tb_up_p_9;
                    tb_tasks_incl = tb_tasks_incl + tb_up_t_10;
                    tb_count_incl = tb_count_incl + tb_up_c_11;
                }
                int tb_row_base = tb_carry_padded + tb_padded_incl - tb_padded;
                int tb_task_base = build_shared_tasks + tb_carry_tasks + tb_tasks_incl - tb_tasks;
                int _shfl_6 = __shfl_sync(0xFFFFFFFF, tb_padded_incl, 31);
                tb_carry_padded = tb_carry_padded + _shfl_6;
                int _shfl_7 = __shfl_sync(0xFFFFFFFF, tb_tasks_incl, 31);
                tb_carry_tasks = tb_carry_tasks + _shfl_7;
                int _shfl_8 = __shfl_sync(0xFFFFFFFF, tb_count_incl, 31);
                tb_carry_count = tb_carry_count + _shfl_8;
                expert_row_offsets[tb_e] = tb_row_base;
                expert_task_base[tb_e] = tb_task_base;
                if (active_rows >= 1024) {
                    int tb_secs[4];
                    int tb_need[4];
                    int tb_base[4];
                    #pragma unroll
                    for (int tb_s = 0; tb_s < 4; tb_s++) {
                        tb_secs[tb_s] = 0;
                        tb_need[tb_s] = 0;
                        if (tb_s < world_size) {
                            tb_secs[tb_s] = source_expert_counts[tb_s * 64 + tb_e];
                            int _max_9 = ((source_record_counts[tb_s]) > (0) ? (source_record_counts[tb_s]) : (0));
                            int _min_12 = ((_max_9) < (8192) ? (_max_9) : (8192));
                            int tb_rc = _min_12;
                            int _max_10 = (((tb_rc + 8 - 1) / 8) > (dispatch_chunk_min_records) ? ((tb_rc + 8 - 1) / 8) : (dispatch_chunk_min_records));
                            int tb_q = _max_10;
                            int tb_pf = reinterpret_cast<const int*>(reinterpret_cast<int*>(dispatch_header_inbox) + ((tb_s * 2 + slot) * 136 + 8 + 64 + tb_e))[0];
                            int _max_11 = ((tb_pf) > (0) ? (tb_pf) : (0));
                            int _min_13 = ((_max_11) < (tb_rc) ? (_max_11) : (tb_rc));
                            tb_pf = _min_13;
                            tb_need[tb_s] = (tb_pf + tb_q - 1) / tb_q;
                        }
                    }
                    int es_run = 0;
                    #pragma unroll
                    for (int tb_s_1 = 0; tb_s_1 < 4; tb_s_1++) {
                        int tb_b = 0;
                        #pragma unroll
                        for (int tb_s2 = 0; tb_s2 < 4; tb_s2++) {
                            if (tb_need[tb_s2] < tb_need[tb_s_1] || tb_need[tb_s2] == tb_need[tb_s_1] && tb_s2 < tb_s_1) {
                                tb_b = tb_b + tb_secs[tb_s2];
                            }
                        }
                        tb_base[tb_s_1] = tb_b;
                        expert_source_base[tb_e * 4 + tb_s_1] = tb_b;
                        es_run = es_run + tb_secs[tb_s_1];
                    }
                    if (es_run != tb_count) {
                        atomicMax(&protocol_error[0], 1);
                    }
                    #pragma unroll 1
                    for (int tb_z = 0; tb_z < 36; tb_z++) {
                        tb_scr[tb_e * 36 + tb_z] = 0;
                    }
                    int tb_cnt[9];
                    #pragma unroll
                    for (int tb_k = 0; tb_k < 9; tb_k++) {
                        tb_cnt[tb_k] = 0;
                    }
                    #pragma unroll 1
                    for (int tb_m = 0; tb_m < tb_padded; tb_m += 128) {
                        int tb_key = 0;
                        #pragma unroll
                        for (int tb_s_2 = 0; tb_s_2 < 4; tb_s_2++) {
                            if (tb_secs[tb_s_2] > 0 && tb_base[tb_s_2] < tb_m + 128 && tb_m < tb_base[tb_s_2] + tb_secs[tb_s_2]) {
                                int _max_12 = ((tb_key) > (tb_need[tb_s_2]) ? (tb_key) : (tb_need[tb_s_2]));
                                tb_key = _max_12;
                            }
                        }
                        #pragma unroll
                        for (int tb_s_3 = 0; tb_s_3 < 4; tb_s_3++) {
                            int _max_13 = ((tb_base[tb_s_3]) > (tb_m) ? (tb_base[tb_s_3]) : (tb_m));
                            int tb_lo = _max_13;
                            int _min_14 = ((tb_base[tb_s_3] + tb_secs[tb_s_3]) < (tb_m + 128) ? (tb_base[tb_s_3] + tb_secs[tb_s_3]) : (tb_m + 128));
                            int tb_hi = _min_14;
                            if (tb_hi > tb_lo) {
                                tb_scr[tb_e * 36 + tb_key * 4 + tb_s_3] = tb_scr[tb_e * 36 + tb_key * 4 + tb_s_3] + (tb_hi - tb_lo);
                            }
                        }
                        #pragma unroll
                        for (int tb_k_1 = 0; tb_k_1 < 9; tb_k_1++) {
                            tb_cnt[tb_k_1] = tb_cnt[tb_k_1] + (int)(tb_key == tb_k_1);
                        }
                    }
                    int tb_pre[9];
                    int tb_below[9];
                    #pragma unroll
                    for (int tb_k_2 = 0; tb_k_2 < 9; tb_k_2++) {
                        int tb_inc = tb_cnt[tb_k_2];
                        int _shfl_up_15 = __shfl_up_sync(0xFFFFFFFF, tb_inc, 1, 32);
                        int tb_up3 = _shfl_up_15;
                        if (tb_lane >= 1) {
                            tb_inc = tb_inc + tb_up3;
                        }
                        int _shfl_up_16 = __shfl_up_sync(0xFFFFFFFF, tb_inc, 2, 32);
                        int tb_up3_0 = _shfl_up_16;
                        if (tb_lane >= 2) {
                            tb_inc = tb_inc + tb_up3_0;
                        }
                        int _shfl_up_17 = __shfl_up_sync(0xFFFFFFFF, tb_inc, 4, 32);
                        int tb_up3_1 = _shfl_up_17;
                        if (tb_lane >= 4) {
                            tb_inc = tb_inc + tb_up3_1;
                        }
                        int _shfl_up_18 = __shfl_up_sync(0xFFFFFFFF, tb_inc, 8, 32);
                        int tb_up3_2 = _shfl_up_18;
                        if (tb_lane >= 8) {
                            tb_inc = tb_inc + tb_up3_2;
                        }
                        int _shfl_up_19 = __shfl_up_sync(0xFFFFFFFF, tb_inc, 16, 32);
                        int tb_up3_3 = _shfl_up_19;
                        if (tb_lane >= 16) {
                            tb_inc = tb_inc + tb_up3_3;
                        }
                        tb_pre[tb_k_2] = tb_tot[tb_k_2] + tb_inc - tb_cnt[tb_k_2];
                        int _shfl_9 = __shfl_sync(0xFFFFFFFF, tb_inc, 31);
                        tb_tot[tb_k_2] = tb_tot[tb_k_2] + _shfl_9;
                    }
                    int tb_acc2 = 0;
                    #pragma unroll
                    for (int tb_k_3 = 0; tb_k_3 < 9; tb_k_3++) {
                        tb_below[tb_k_3] = tb_acc2;
                        tb_acc2 = tb_acc2 + tb_cnt[tb_k_3];
                    }
                    #pragma unroll
                    for (int tb_k_4 = 0; tb_k_4 < 9; tb_k_4++) {
                        #pragma unroll
                        for (int tb_s_4 = 0; tb_s_4 < 4; tb_s_4++) {
                            __syncwarp();
                            int tb_vc = tb_scr[4608 + tb_k_4 * 4 + tb_s_4];
                            int tb_v = tb_scr[tb_e * 36 + tb_k_4 * 4 + tb_s_4];
                            int tb_vi = tb_v;
                            int _shfl_up_20 = __shfl_up_sync(0xFFFFFFFF, tb_vi, 1, 32);
                            int tb_up4 = _shfl_up_20;
                            if (tb_lane >= 1) {
                                tb_vi = tb_vi + tb_up4;
                            }
                            int _shfl_up_21 = __shfl_up_sync(0xFFFFFFFF, tb_vi, 2, 32);
                            int tb_up4_0 = _shfl_up_21;
                            if (tb_lane >= 2) {
                                tb_vi = tb_vi + tb_up4_0;
                            }
                            int _shfl_up_22 = __shfl_up_sync(0xFFFFFFFF, tb_vi, 4, 32);
                            int tb_up4_1 = _shfl_up_22;
                            if (tb_lane >= 4) {
                                tb_vi = tb_vi + tb_up4_1;
                            }
                            int _shfl_up_23 = __shfl_up_sync(0xFFFFFFFF, tb_vi, 8, 32);
                            int tb_up4_2 = _shfl_up_23;
                            if (tb_lane >= 8) {
                                tb_vi = tb_vi + tb_up4_2;
                            }
                            int _shfl_up_24 = __shfl_up_sync(0xFFFFFFFF, tb_vi, 16, 32);
                            int tb_up4_3 = _shfl_up_24;
                            if (tb_lane >= 16) {
                                tb_vi = tb_vi + tb_up4_3;
                            }
                            tb_scr[2304 + tb_e * 36 + tb_k_4 * 4 + tb_s_4] = tb_vc + tb_vi - tb_v;
                            int _shfl_10 = __shfl_sync(0xFFFFFFFF, tb_vi, 31);
                            int tb_vt = _shfl_10;
                            __syncwarp();
                            if (tb_lane == 0) {
                                tb_scr[4608 + tb_k_4 * 4 + tb_s_4] = tb_vc + tb_vt;
                            }
                        }
                    }
                    #pragma unroll
                    for (int tb_s_5 = 0; tb_s_5 < 4; tb_s_5++) {
                        tb_pub[tb_e * 71 + tb_s_5] = tb_need[tb_s_5];
                        tb_pub[tb_e * 71 + 4 + tb_s_5] = tb_base[tb_s_5];
                    }
                    #pragma unroll
                    for (int tb_k_5 = 0; tb_k_5 < 9; tb_k_5++) {
                        tb_pub[tb_e * 71 + 17 + tb_k_5] = tb_pre[tb_k_5];
                        tb_pub[tb_e * 71 + 26 + tb_k_5] = tb_below[tb_k_5];
                    }
                    #pragma unroll 1
                    for (int tb_z2 = 0; tb_z2 < 36; tb_z2++) {
                        tb_pub[tb_e * 71 + 35 + tb_z2] = tb_scr[2304 + tb_e * 36 + tb_z2];
                    }
                } else {
                    int so_run = 0;
                    #pragma unroll 1
                    for (int es_source = 0; es_source < 4; es_source++) {
                        int tb_sec = source_expert_counts[es_source * 64 + tb_e];
                        int tb_sec_incl = tb_sec;
                        int _shfl_up_25 = __shfl_up_sync(0xFFFFFFFF, tb_sec_incl, 1, 32);
                        int tb_up_s = _shfl_up_25;
                        if (tb_lane >= 1) {
                            tb_sec_incl = tb_sec_incl + tb_up_s;
                        }
                        int _shfl_up_26 = __shfl_up_sync(0xFFFFFFFF, tb_sec_incl, 2, 32);
                        int tb_up_s_0 = _shfl_up_26;
                        if (tb_lane >= 2) {
                            tb_sec_incl = tb_sec_incl + tb_up_s_0;
                        }
                        int _shfl_up_27 = __shfl_up_sync(0xFFFFFFFF, tb_sec_incl, 4, 32);
                        int tb_up_s_1 = _shfl_up_27;
                        if (tb_lane >= 4) {
                            tb_sec_incl = tb_sec_incl + tb_up_s_1;
                        }
                        int _shfl_up_28 = __shfl_up_sync(0xFFFFFFFF, tb_sec_incl, 8, 32);
                        int tb_up_s_2 = _shfl_up_28;
                        if (tb_lane >= 8) {
                            tb_sec_incl = tb_sec_incl + tb_up_s_2;
                        }
                        int _shfl_up_29 = __shfl_up_sync(0xFFFFFFFF, tb_sec_incl, 16, 32);
                        int tb_up_s_3 = _shfl_up_29;
                        if (tb_lane >= 16) {
                            tb_sec_incl = tb_sec_incl + tb_up_s_3;
                        }
                        int tb_so_c = tb_scr[4608 + es_source];
                        source_expert_prefix[tb_e * 4 + es_source] = tb_so_c + tb_sec_incl - tb_sec;
                        int _shfl_11 = __shfl_sync(0xFFFFFFFF, tb_sec_incl, 31);
                        int tb_so_t = _shfl_11;
                        __syncwarp();
                        if (tb_lane == 0) {
                            tb_scr[4608 + es_source] = tb_so_c + tb_so_t;
                        }
                        __syncwarp();
                        expert_source_base[tb_e * 4 + es_source] = so_run;
                        if (es_source < world_size) {
                            so_run = so_run + tb_sec;
                        }
                    }
                    if (so_run != tb_count) {
                        atomicMax(&protocol_error[0], 1);
                    }
                    #pragma unroll 1
                    for (int m_local = 0; m_local < tb_padded; m_local += 128) {
                        int tb_task_idx = tb_task_base + m_local / 128;
                        if (tb_task_idx < 1664) {
                            int _min_15 = ((m_local + 128) < (tb_count) ? (m_local + 128) : (tb_count));
                            int task_last_row = _min_15;
                            int task_ms = -1;
                            #pragma unroll 1
                            for (int ms_source = 0; ms_source < 4; ms_source++) {
                                if (ms_source < world_size) {
                                    int ms_base = expert_source_base[tb_e * 4 + ms_source];
                                    int ms_count = source_expert_counts[ms_source * 64 + tb_e];
                                    if (ms_count > 0 && ms_base < task_last_row && m_local < ms_base + ms_count) {
                                        task_ms = ms_source;
                                    }
                                }
                            }
                            task_max_source[tb_task_idx] = task_ms;
                            task_expert[tb_task_idx] = rank * 64 + tb_e;
                            task_source_rank[tb_task_idx] = 0;
                            task_owner_rank[tb_task_idx] = rank;
                            task_local_expert[tb_task_idx] = tb_e;
                            task_pool_row[tb_task_idx] = tb_row_base + m_local;
                            task_m_local[tb_task_idx] = m_local;
                            int _min_16 = ((128) < (tb_count - m_local) ? (128) : (tb_count - m_local));
                            task_valid_m[tb_task_idx] = _min_16;
                            int task_gate = 0;
                            #pragma unroll 1
                            for (int gate_source = 0; gate_source < 4; gate_source++) {
                                if (gate_source < world_size && task_ms >= gate_source) {
                                    int gate_count = source_record_counts[gate_source];
                                    int _max_14 = (((gate_count + 8 - 1) / 8) > (dispatch_chunk_min_records) ? ((gate_count + 8 - 1) / 8) : (dispatch_chunk_min_records));
                                    int gate_q = _max_14;
                                    int gate_prefix = reinterpret_cast<const int*>(reinterpret_cast<int*>(dispatch_header_inbox) + ((gate_source * 2 + slot) * 136 + 8 + 64 + tb_e))[0];
                                    int _max_15 = ((gate_prefix) > (0) ? (gate_prefix) : (0));
                                    int _min_17 = ((_max_15) < (gate_count) ? (_max_15) : (gate_count));
                                    gate_prefix = _min_17;
                                    int gate_chunks = (gate_prefix + gate_q - 1) / gate_q;
                                    task_gate = task_gate | gate_chunks << gate_source * 4;
                                }
                            }
                            task_gate_packed[tb_task_idx] = task_gate;
                            expert_block_task[tb_e * 1664 + m_local / 128] = tb_task_idx;
                            #pragma unroll
                            for (int sb_source = 0; sb_source < 4; sb_source++) {
                                int sb_base = expert_source_base[tb_e * 4 + sb_source];
                                int sb_count = 0;
                                if (sb_source < world_size) {
                                    sb_count = source_expert_counts[sb_source * 64 + tb_e];
                                }
                                int _max_16 = ((m_local - sb_base) > (0) ? (m_local - sb_base) : (0));
                                int _min_18 = ((_max_16) < (sb_count) ? (_max_16) : (sb_count));
                                task_source_slot_base[tb_task_idx * 4 + sb_source] = source_expert_prefix[tb_e * 4 + sb_source] + _min_18;
                            }
                        } else {
                            atomicMax(&protocol_error[0], 1);
                        }
                    }
                }
            }
            int tb_total_padded = tb_carry_padded;
            int tb_total_tasks = build_shared_tasks + tb_carry_tasks;
            int tb_valid_routes = tb_carry_count;
            if (active_rows >= 1024) {
                int tb_kbase[9];
                int tb_acc = 0;
                #pragma unroll
                for (int tb_k_6 = 0; tb_k_6 < 9; tb_k_6++) {
                    tb_kbase[tb_k_6] = tb_acc;
                    tb_acc = tb_acc + tb_tot[tb_k_6];
                }
                __syncwarp();
                if (tb_lane == 0) {
                    #pragma unroll
                    for (int tb_s_6 = 0; tb_s_6 < 4; tb_s_6++) {
                        int tb_ka = 0;
                        #pragma unroll
                        for (int tb_k_7 = 0; tb_k_7 < 9; tb_k_7++) {
                            int tb_kt = tb_scr[4608 + tb_k_7 * 4 + tb_s_6];
                            tb_scr[4608 + tb_k_7 * 4 + tb_s_6] = tb_ka;
                            tb_ka = tb_ka + tb_kt;
                        }
                    }
                }
                __syncwarp();
                #pragma unroll
                for (int tb_pass2 = 0; tb_pass2 < 2; tb_pass2++) {
                    int tb_e2 = tb_pass2 * 32 + tb_lane;
                    #pragma unroll
                    for (int tb_k_8 = 0; tb_k_8 < 9; tb_k_8++) {
                        tb_pub[tb_e2 * 71 + 8 + tb_k_8] = tb_kbase[tb_k_8];
                    }
                }
                if (tb_lane == 0) {
                    #pragma unroll 1
                    for (int tb_z3 = 0; tb_z3 < 36; tb_z3++) {
                        tb_pub[4544 + tb_z3] = tb_scr[4608 + tb_z3];
                    }
                }
            }
            __syncwarp();
            if (tb_lane == 0) {
                if (tb_valid_routes > 196608 || tb_total_padded > 204736 || tb_total_tasks > 1664) {
                    atomicMax(&protocol_error[0], 1);
                }
                total_valid_routes[0] = tb_valid_routes;
                total_padded_rows[0] = tb_total_padded;
                total_m_tasks[0] = tb_total_tasks;
                __threadfence();
                prefix_done[0] = 1;
            }
        }
        __syncwarp();
        asm volatile("barrier.sync 13, %0;" :: "r"(c24_threads) : "memory");
        if (c24_leader != 0) {
            {
                unsigned int* _gc_p = reinterpret_cast<unsigned int*>(c24_front_sync) + (2);
                unsigned int _gc_old;
                asm volatile("atom.release.gpu.global.add.u32 %0, [%1], 1;" : "=r"(_gc_old) : "l"(_gc_p) : "memory");
            }
            {
                unsigned int* _gca_p = reinterpret_cast<unsigned int*>(c24_front_sync) + (2);
                while (true) {
                    unsigned int _gca_v;
                    asm volatile("ld.acquire.gpu.global.u32 %0, [%1];" : "=r"(_gca_v) : "l"(_gca_p));
                    if (_gca_v >= (unsigned int)(num_bids)) break;
                }
            }
        }
        __syncwarp();
        asm volatile("barrier.sync 13, %0;" :: "r"(c24_threads) : "memory");
        if (bid == 0 && tid == 0) {
            unsigned long long gtimer_0_5;
            asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(gtimer_0_5) :: "memory");
            phase_timestamps[6] = gtimer_0_5;
        }
        if (active_rows >= 1024) {
            int p2_warp = c24_tid / 32;
            int p2_warps = c24_all / 32;
            int p2_shared = (active_rows + 128 - 1) / 128;
            int p2_total = total_m_tasks[0] - p2_shared;
            int p2_lane = lane;
            #pragma unroll 1
            for (int p2_f = p2_warp; p2_f < p2_total; p2_f += p2_warps) {
                int p2_hit = 0;
                #pragma unroll
                for (int p2_pass = 0; p2_pass < 2; p2_pass++) {
                    int p2_el = p2_pass * 32 + p2_lane;
                    int p2_cnt_l = expert_counts[p2_el];
                    int p2_blk_l = (p2_cnt_l + 128 - 1) / 128;
                    int p2_bp_l = expert_task_base[p2_el] - p2_shared;
                    int _max_17 = ((p2_hit) > ((int)(p2_bp_l <= p2_f && p2_f < p2_bp_l + p2_blk_l) * (p2_el + 1)) ? (p2_hit) : ((int)(p2_bp_l <= p2_f && p2_f < p2_bp_l + p2_blk_l) * (p2_el + 1)));
                    p2_hit = _max_17;
                }
                int _shfl_xor_0 = __shfl_xor_sync(0xFFFFFFFF, p2_hit, 16);
                int _max_18 = ((p2_hit) > (_shfl_xor_0) ? (p2_hit) : (_shfl_xor_0));
                p2_hit = _max_18;
                int _shfl_xor_1 = __shfl_xor_sync(0xFFFFFFFF, p2_hit, 8);
                int _max_19 = ((p2_hit) > (_shfl_xor_1) ? (p2_hit) : (_shfl_xor_1));
                p2_hit = _max_19;
                int _shfl_xor_2 = __shfl_xor_sync(0xFFFFFFFF, p2_hit, 4);
                int _max_20 = ((p2_hit) > (_shfl_xor_2) ? (p2_hit) : (_shfl_xor_2));
                p2_hit = _max_20;
                int _shfl_xor_3 = __shfl_xor_sync(0xFFFFFFFF, p2_hit, 2);
                int _max_21 = ((p2_hit) > (_shfl_xor_3) ? (p2_hit) : (_shfl_xor_3));
                p2_hit = _max_21;
                int _shfl_xor_4 = __shfl_xor_sync(0xFFFFFFFF, p2_hit, 1);
                int _max_22 = ((p2_hit) > (_shfl_xor_4) ? (p2_hit) : (_shfl_xor_4));
                p2_hit = _max_22;
                if (p2_hit > 0) {
                    int p2_e = p2_hit - 1;
                    int p2_j = p2_f - (expert_task_base[p2_e] - p2_shared);
                    int p2_m = p2_j * 128;
                    int p2_count = expert_counts[p2_e];
                    int p2_row_base = expert_row_offsets[p2_e];
                    int p2_secs = 0;
                    int p2_need = 0;
                    int p2_base = 0;
                    if (p2_lane < 4) {
                        p2_secs = source_expert_counts[p2_lane * 64 + p2_e];
                        p2_need = tb_pub[p2_e * 71 + p2_lane];
                        p2_base = tb_pub[p2_e * 71 + 4 + p2_lane];
                    }
                    int p2_isect = (int)(p2_secs > 0 && p2_base < p2_m + 128 && p2_m < p2_base + p2_secs);
                    int p2_key = p2_isect * p2_need;
                    int p2_ms = p2_isect * (p2_lane + 1) - 1;
                    int _shfl_xor_5 = __shfl_xor_sync(0xFFFFFFFF, p2_key, 16);
                    int _max_23 = ((p2_key) > (_shfl_xor_5) ? (p2_key) : (_shfl_xor_5));
                    p2_key = _max_23;
                    int _shfl_xor_6 = __shfl_xor_sync(0xFFFFFFFF, p2_key, 8);
                    int _max_24 = ((p2_key) > (_shfl_xor_6) ? (p2_key) : (_shfl_xor_6));
                    p2_key = _max_24;
                    int _shfl_xor_7 = __shfl_xor_sync(0xFFFFFFFF, p2_key, 4);
                    int _max_25 = ((p2_key) > (_shfl_xor_7) ? (p2_key) : (_shfl_xor_7));
                    p2_key = _max_25;
                    int _shfl_xor_8 = __shfl_xor_sync(0xFFFFFFFF, p2_key, 2);
                    int _max_26 = ((p2_key) > (_shfl_xor_8) ? (p2_key) : (_shfl_xor_8));
                    p2_key = _max_26;
                    int _shfl_xor_9 = __shfl_xor_sync(0xFFFFFFFF, p2_key, 1);
                    int _max_27 = ((p2_key) > (_shfl_xor_9) ? (p2_key) : (_shfl_xor_9));
                    p2_key = _max_27;
                    int _shfl_xor_10 = __shfl_xor_sync(0xFFFFFFFF, p2_ms, 16);
                    int _max_28 = ((p2_ms) > (_shfl_xor_10) ? (p2_ms) : (_shfl_xor_10));
                    p2_ms = _max_28;
                    int _shfl_xor_11 = __shfl_xor_sync(0xFFFFFFFF, p2_ms, 8);
                    int _max_29 = ((p2_ms) > (_shfl_xor_11) ? (p2_ms) : (_shfl_xor_11));
                    p2_ms = _max_29;
                    int _shfl_xor_12 = __shfl_xor_sync(0xFFFFFFFF, p2_ms, 4);
                    int _max_30 = ((p2_ms) > (_shfl_xor_12) ? (p2_ms) : (_shfl_xor_12));
                    p2_ms = _max_30;
                    int _shfl_xor_13 = __shfl_xor_sync(0xFFFFFFFF, p2_ms, 2);
                    int _max_31 = ((p2_ms) > (_shfl_xor_13) ? (p2_ms) : (_shfl_xor_13));
                    p2_ms = _max_31;
                    int _shfl_xor_14 = __shfl_xor_sync(0xFFFFFFFF, p2_ms, 1);
                    int _max_32 = ((p2_ms) > (_shfl_xor_14) ? (p2_ms) : (_shfl_xor_14));
                    p2_ms = _max_32;
                    int p2_lo = 0;
                    int p2_hi = p2_j;
                    #pragma unroll 1
                    for (int p2_it = 0; p2_it < 10; p2_it++) {
                        if (p2_lo < p2_hi) {
                            int p2_mid = (p2_lo + p2_hi) / 2;
                            int p2_mm = p2_mid * 128;
                            int p2_ki = (int)(p2_secs > 0 && p2_base < p2_mm + 128 && p2_mm < p2_base + p2_secs) * p2_need;
                            int _shfl_xor_15 = __shfl_xor_sync(0xFFFFFFFF, p2_ki, 16);
                            int _max_33 = ((p2_ki) > (_shfl_xor_15) ? (p2_ki) : (_shfl_xor_15));
                            p2_ki = _max_33;
                            int _shfl_xor_16 = __shfl_xor_sync(0xFFFFFFFF, p2_ki, 8);
                            int _max_34 = ((p2_ki) > (_shfl_xor_16) ? (p2_ki) : (_shfl_xor_16));
                            p2_ki = _max_34;
                            int _shfl_xor_17 = __shfl_xor_sync(0xFFFFFFFF, p2_ki, 4);
                            int _max_35 = ((p2_ki) > (_shfl_xor_17) ? (p2_ki) : (_shfl_xor_17));
                            p2_ki = _max_35;
                            int _shfl_xor_18 = __shfl_xor_sync(0xFFFFFFFF, p2_ki, 2);
                            int _max_36 = ((p2_ki) > (_shfl_xor_18) ? (p2_ki) : (_shfl_xor_18));
                            p2_ki = _max_36;
                            int _shfl_xor_19 = __shfl_xor_sync(0xFFFFFFFF, p2_ki, 1);
                            int _max_37 = ((p2_ki) > (_shfl_xor_19) ? (p2_ki) : (_shfl_xor_19));
                            p2_ki = _max_37;
                            if (p2_ki >= p2_key) {
                                p2_hi = p2_mid;
                            } else {
                                p2_lo = p2_mid + 1;
                            }
                        }
                    }
                    int p2_j0 = p2_lo;
                    int p2_kbase = tb_pub[p2_e * 71 + 8 + p2_key];
                    int p2_pre = tb_pub[p2_e * 71 + 17 + p2_key];
                    int p2_below = tb_pub[p2_e * 71 + 26 + p2_key];
                    int p2_idx = p2_shared + p2_kbase + p2_pre - p2_below + p2_j;
                    if (p2_idx < 1664 && p2_j < 1664) {
                        if (p2_lane == 0) {
                            expert_block_task[p2_e * 1664 + p2_j] = p2_idx;
                            task_max_source[p2_idx] = p2_ms;
                            task_expert[p2_idx] = rank * 64 + p2_e;
                            task_source_rank[p2_idx] = 0;
                            task_owner_rank[p2_idx] = rank;
                            task_local_expert[p2_idx] = p2_e;
                            task_pool_row[p2_idx] = p2_row_base + p2_m;
                            task_m_local[p2_idx] = p2_m;
                            int _min_19 = ((128) < (p2_count - p2_m) ? (128) : (p2_count - p2_m));
                            task_valid_m[p2_idx] = _min_19;
                            task_gate_packed[p2_idx] = 0;
                        }
                        if (p2_lane < 4) {
                            int _min_20 = ((p2_base + p2_secs) < (p2_m) ? (p2_base + p2_secs) : (p2_m));
                            int _max_38 = ((p2_base) > (p2_j0 * 128) ? (p2_base) : (p2_j0 * 128));
                            int _max_39 = ((_min_20 - _max_38) > (0) ? (_min_20 - _max_38) : (0));
                            int p2_run = _max_39;
                            task_source_slot_base[p2_idx * 4 + p2_lane] = tb_pub[4544 + p2_key * 4 + p2_lane] + tb_pub[p2_e * 71 + 35 + p2_key * 4 + p2_lane] + p2_run;
                        }
                    } else if (p2_lane == 0) {
                        atomicMax(&protocol_error[0], 1);
                    }
                }
                __syncwarp();
            }
        }
        int pad_tid = c24_tid;
        int pad_threads = c24_all;
        #pragma unroll 1
        for (int pad_expert = 0; pad_expert < 64; pad_expert++) {
            int pad_count = expert_counts[pad_expert];
            int pad_padded = (pad_count + 128 - 1) / 128 * 128;
            int pad_first = expert_row_offsets[pad_expert] + pad_count;
            int pad_rows = pad_padded - pad_count;
            if (pad_rows > 0) {
                int pad_word_base = pad_first * 1024;
                #pragma unroll 1
                for (int pad_word = pad_tid; pad_word < pad_rows * 1024; pad_word += pad_threads) {
                    pool_fp8_u32[pad_word_base + pad_word] = 0;
                }
                #pragma unroll 1
                for (int pad_sf = pad_tid; pad_sf < 32 * pad_rows; pad_sf += pad_threads) {
                    int pad_sf_word = pad_sf / pad_rows;
                    int pad_sf_row = pad_first + (pad_sf - pad_sf_word * pad_rows);
                    pool_sf_u32[pad_sf_word * 212928 + pad_sf_row] = 0;
                }
            }
        }
        __threadfence();
        __syncwarp();
        asm volatile("barrier.sync 13, %0;" :: "r"(c24_threads) : "memory");
        if (c24_leader != 0) {
            {
                unsigned int* _gc_p = reinterpret_cast<unsigned int*>(c24_front_sync) + (3);
                unsigned int _gc_old;
                asm volatile("atom.release.gpu.global.add.u32 %0, [%1], 1;" : "=r"(_gc_old) : "l"(_gc_p) : "memory");
            }
            {
                unsigned int* _gca_p = reinterpret_cast<unsigned int*>(c24_front_sync) + (3);
                while (true) {
                    unsigned int _gca_v;
                    asm volatile("ld.acquire.gpu.global.u32 %0, [%1];" : "=r"(_gca_v) : "l"(_gca_p));
                    if (_gca_v >= (unsigned int)(num_bids)) break;
                }
            }
        }
        __syncwarp();
        asm volatile("barrier.sync 13, %0;" :: "r"(c24_threads) : "memory");
    }
    int scatter_source = bid % 4;
    int scatter_group_rank = bid / 4;
    int scatter_group_ctas = (num_bids - scatter_source + 4 - 1) / 4;
    int* scatter_src_i32 = reinterpret_cast<int*>(dispatch_payload_inbox);
    float* scatter_src_f32 = reinterpret_cast<float*>(dispatch_payload_inbox);
    if (scatter_source == rank) {
        scatter_src_i32 = reinterpret_cast<int*>(dispatch_payload_out);
        scatter_src_f32 = reinterpret_cast<float*>(dispatch_payload_out);
    }
    if (pull_engaged != 0) {
        if (elect_sync()) {
            int watch_count = source_record_counts[pull_source];
            int _max_40 = (((watch_count + 8 - 1) / 8) > (dispatch_chunk_min_records) ? ((watch_count + 8 - 1) / 8) : (dispatch_chunk_min_records));
            int watch_q = _max_40;
            int watch_chunks = (watch_count + watch_q - 1) / watch_q;
            #pragma unroll 1
            for (int watch_c = 0; watch_c < 8; watch_c++) {
                if (watch_chunks > watch_c) {
                    // gin_wait_signal: acquire, rolling 64-bit comparison
                    {
                        ncclGin __gin{*(gin_dev_comm), (int)(0)};
                        __gin.waitSignal(ncclCoopThread(), (ncclGinSignal_t)(12 + pull_source * 8 + watch_c), (uint64_t)(dispatch_chunk_signal_base_scratch[pull_source * 8 + watch_c] + 1), 64, cuda::memory_order_acquire);
                    }
                    __threadfence();
                    {
                        unsigned int* _gc_p = reinterpret_cast<unsigned int*>(pull_chunk_arrived) + (pull_source * 8 + watch_c);
                        unsigned int _gc_old;
                        asm volatile("atom.release.gpu.global.add.u32 %0, [%1], 1;" : "=r"(_gc_old) : "l"(_gc_p) : "memory");
                    }
                }
            }
            if (watch_chunks > 0) {
                unsigned long long gtimer_0_6;
                asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(gtimer_0_6) :: "memory");
                peer_phase_timestamps[pull_source] = gtimer_0_6;
            }
        }
        __syncwarp();
    }
    if (scatter_source < world_size) {
        int scatter_has_puller = (int)(scatter_source != rank);
        int scatter_stride_index = scatter_group_rank * 3 + (warp - 9) - scatter_has_puller;
        if (warp >= 9 && scatter_stride_index >= 0) {
            int task_global_warp = scatter_stride_index;
            int task_grid_warps = scatter_group_ctas * 3 - scatter_has_puller;
            int scatter_count = source_record_counts[scatter_source];
            int _max_41 = (((scatter_count + 8 - 1) / 8) > (dispatch_chunk_min_records) ? ((scatter_count + 8 - 1) / 8) : (dispatch_chunk_min_records));
            int scatter_q = _max_41;
            int scatter_ready_chunks = 0;
            #pragma unroll 1
            for (int record_2 = task_global_warp; record_2 < scatter_count; record_2 += task_grid_warps) {
                int source_5 = scatter_source;
                int scatter_chunk = record_2 / scatter_q;
                int sc_tpk0 = 0;
                int sc_tpk1 = 0;
                int sc_tpk2 = 0;
                int sc_tvalid = 0;
                if (scatter_chunk >= scatter_ready_chunks) {
                    if (scatter_source == rank) {
                        if (elect_sync()) {
                            // gin_wait_signal: acquire, rolling 64-bit comparison
                            {
                                ncclGin __gin{*(gin_dev_comm), (int)(0)};
                                __gin.waitSignal(ncclCoopThread(), (ncclGinSignal_t)(12 + scatter_source * 8 + scatter_chunk), (uint64_t)(dispatch_chunk_signal_base_scratch[scatter_source * 8 + scatter_chunk] + 1), 64, cuda::memory_order_acquire);
                            }
                        }
                    } else {
                        if (elect_sync()) {
                            {
                                unsigned int* _gca_p = reinterpret_cast<unsigned int*>(pull_chunk_arrived) + (scatter_source * 8 + scatter_chunk);
                                while (true) {
                                    unsigned int _gca_v;
                                    asm volatile("ld.acquire.gpu.global.u32 %0, [%1];" : "=r"(_gca_v) : "l"(_gca_p));
                                    if (_gca_v >= (unsigned int)(1)) break;
                                }
                            }
                        }
                    }
                    __syncwarp();
                    scatter_ready_chunks = scatter_chunk + 1;
                }
                if (source_5 < world_size) {
                    unsigned long long scatter_record_word = (unsigned long long)(source_5 * 2 + slot) * 8912896 + (unsigned long long)record_2 * 1088;
                    int scatter_route_count = reinterpret_cast<const int*>(scatter_src_i32 + (scatter_record_word + 1))[0];
                    int scatter_route_base = reinterpret_cast<const int*>(scatter_src_i32 + (scatter_record_word + 22))[0];
                    int scatter_token = reinterpret_cast<const int*>(scatter_src_i32 + scatter_record_word)[0];
                    int scatter_record_ok = (int)(scatter_route_count >= 1 && scatter_route_count <= 6 && scatter_route_base >= 0 && scatter_route_base + scatter_route_count <= source_route_counts[source_5] && scatter_route_base + scatter_route_count <= 49152 && reinterpret_cast<const int*>(scatter_src_i32 + (scatter_record_word + 20))[0] == source_5 && reinterpret_cast<const int*>(scatter_src_i32 + (scatter_record_word + 21))[0] == 1347571524);
                    if (scatter_record_ok == 0) {
                        if (lane == 0) {
                            atomicMax(&protocol_error[0], 1);
                        }
                    } else {
                        #pragma unroll 1
                        for (int record_route_2 = 0; record_route_2 < scatter_route_count; record_route_2++) {
                            int scatter_local_expert = reinterpret_cast<const int*>(scatter_src_i32 + (scatter_record_word + 2 + (unsigned long long)record_route_2))[0];
                            int scatter_topk_slot = reinterpret_cast<const int*>(scatter_src_i32 + (scatter_record_word + 8 + (unsigned long long)record_route_2))[0];
                            float scatter_route_weight = reinterpret_cast<const float*>(scatter_src_f32 + (scatter_record_word + 14 + (unsigned long long)record_route_2))[0];
                            int scatter_result_index = scatter_route_base + record_route_2;
                            unsigned int scatter_weight_bits = 0;
                            scatter_weight_bits = reinterpret_cast<unsigned int*>(&scatter_route_weight)[0];
                            int scatter_finite = (int)((scatter_weight_bits & 2139095040) != 2139095040);
                            int scatter_route_ok = (int)(scatter_local_expert >= 0 && scatter_local_expert < 64 && scatter_token >= 0 && scatter_token < source_active_rows[source_5] && scatter_topk_slot >= 0 && scatter_topk_slot < 6 && scatter_finite != 0);
                            if (scatter_route_ok != 0) {
                                int scatter_dst_row = -1;
                                if (lane == 0) {
                                    int scatter_es = scatter_local_expert * 4 + source_5;
                                    int sc_row_offset = expert_row_offsets[scatter_local_expert];
                                    int sc_source_base = expert_source_base[scatter_es];
                                    int sc_source_count = source_expert_counts[source_5 * 64 + scatter_local_expert];
                                    int _max_42 = ((source_route_counts[source_5]) > (0) ? (source_route_counts[source_5]) : (0));
                                    int _min_21 = ((_max_42) < (49152) ? (_max_42) : (49152));
                                    int c41_sc_r = _min_21;
                                    int _atomic_old_2 = atomicAdd(&expert_source_offsets[scatter_es], 1);
                                    int scatter_claim = _atomic_old_2;
                                    atomicAdd(&expert_scatter_offsets[scatter_local_expert], 1);
                                    int sc_rrow = sc_source_base + scatter_claim;
                                    int _max_43 = ((sc_rrow) > (0) ? (sc_rrow) : (0));
                                    int _min_22 = ((_max_43 / 128) < (1663) ? (_max_43 / 128) : (1663));
                                    int sc_blk = _min_22;
                                    int sc_task = expert_block_task[scatter_local_expert * 1664 + sc_blk];
                                    int dst_row = sc_row_offset + sc_source_base + scatter_claim;
                                    if (scatter_claim < 0 || scatter_claim >= sc_source_count || dst_row < 0 || dst_row >= 204736) {
                                        atomicMax(&protocol_error[0], 1);
                                        dst_row = -1;
                                    } else {
                                        meta_source_rank[dst_row] = source_5;
                                        meta_token[dst_row] = scatter_token;
                                        meta_slot[dst_row] = scatter_topk_slot;
                                        int _max_44 = ((sc_blk * 128) > (sc_source_base) ? (sc_blk * 128) : (sc_source_base));
                                        int scatter_new_index = task_source_slot_base[sc_task * 4 + source_5] + (sc_rrow - _max_44);
                                        meta_result_index[dst_row] = scatter_new_index;
                                        *(reinterpret_cast<int*>(reinterpret_cast<int*>(result_out) + ((unsigned long long)(source_5 * 2 + slot) * 102285312 + 102236160 + (unsigned long long)scatter_result_index)) + (0)) = scatter_new_index;
                                        routing_weight_pool[dst_row] = scatter_route_weight;
                                        int c41_sc_q = 128;
                                        if (c41_sc_r >= 6144) {
                                            c41_sc_q = 64;
                                        }
                                        if (c41_sc_r < 768) {
                                            c41_sc_q = 32;
                                        }
                                        int _max_45 = (((c41_sc_r + c41_sc_q - 1) / c41_sc_q - 1) > (0) ? ((c41_sc_r + c41_sc_q - 1) / c41_sc_q - 1) : (0));
                                        int c41_sc_full = _max_45;
                                        int c41_sc_ts = c41_sc_full * c41_sc_q;
                                        int c41_scatter_chunk = 0;
                                        if (scatter_new_index < c41_sc_ts) {
                                            c41_scatter_chunk = scatter_new_index / c41_sc_q;
                                        } else {
                                            c41_scatter_chunk = c41_sc_full + (scatter_new_index - c41_sc_ts) / 64;
                                        }
                                        atomicAdd(&result_chunk_total[source_5 * 769 + c41_scatter_chunk], 1);
                                    }
                                    scatter_dst_row = dst_row;
                                    if (dst_row >= 0) {
                                        sc_tvalid = sc_tvalid | 1 << record_route_2;
                                        if (record_route_2 < 2) {
                                            sc_tpk0 = sc_tpk0 | sc_task << 12 * record_route_2;
                                        } else if (record_route_2 < 4) {
                                            sc_tpk1 = sc_tpk1 | sc_task << 12 * (record_route_2 - 2);
                                        } else {
                                            sc_tpk2 = sc_tpk2 | sc_task << 12 * (record_route_2 - 4);
                                        }
                                    }
                                }
                                __syncwarp();
                                int _shfl_12 = __shfl_sync(0xFFFFFFFF, scatter_dst_row, 0);
                                scatter_dst_row = _shfl_12;
                                if (scatter_dst_row >= 0) {
                                    unsigned long long src_activation_word = scatter_record_word + 32;
                                    unsigned long long dst_activation_word = (unsigned long long)scatter_dst_row * 1024;
                                    unsigned long long sc_lane_word = (unsigned long long)(lane * 4);
                                    int _vec_load_8[4];
                                    {
                                        int4 _iv4 = *reinterpret_cast<const int4*>(scatter_src_i32 + (src_activation_word + sc_lane_word) + 0);
                                        _vec_load_8[0 + 0] = _iv4.x;
                                        _vec_load_8[0 + 1] = _iv4.y;
                                        _vec_load_8[0 + 2] = _iv4.z;
                                        _vec_load_8[0 + 3] = _iv4.w;
                                    }
                                    int _vec_load_9[4];
                                    {
                                        int4 _iv4 = *reinterpret_cast<const int4*>(scatter_src_i32 + (src_activation_word + sc_lane_word + 128) + 0);
                                        _vec_load_9[0 + 0] = _iv4.x;
                                        _vec_load_9[0 + 1] = _iv4.y;
                                        _vec_load_9[0 + 2] = _iv4.z;
                                        _vec_load_9[0 + 3] = _iv4.w;
                                    }
                                    int _vec_load_10[4];
                                    {
                                        int4 _iv4 = *reinterpret_cast<const int4*>(scatter_src_i32 + (src_activation_word + sc_lane_word + 256) + 0);
                                        _vec_load_10[0 + 0] = _iv4.x;
                                        _vec_load_10[0 + 1] = _iv4.y;
                                        _vec_load_10[0 + 2] = _iv4.z;
                                        _vec_load_10[0 + 3] = _iv4.w;
                                    }
                                    int _vec_load_11[4];
                                    {
                                        int4 _iv4 = *reinterpret_cast<const int4*>(scatter_src_i32 + (src_activation_word + sc_lane_word + 384) + 0);
                                        _vec_load_11[0 + 0] = _iv4.x;
                                        _vec_load_11[0 + 1] = _iv4.y;
                                        _vec_load_11[0 + 2] = _iv4.z;
                                        _vec_load_11[0 + 3] = _iv4.w;
                                    }
                                    int _vec_load_12[4];
                                    {
                                        int4 _iv4 = *reinterpret_cast<const int4*>(scatter_src_i32 + (src_activation_word + sc_lane_word + 512) + 0);
                                        _vec_load_12[0 + 0] = _iv4.x;
                                        _vec_load_12[0 + 1] = _iv4.y;
                                        _vec_load_12[0 + 2] = _iv4.z;
                                        _vec_load_12[0 + 3] = _iv4.w;
                                    }
                                    int _vec_load_13[4];
                                    {
                                        int4 _iv4 = *reinterpret_cast<const int4*>(scatter_src_i32 + (src_activation_word + sc_lane_word + 640) + 0);
                                        _vec_load_13[0 + 0] = _iv4.x;
                                        _vec_load_13[0 + 1] = _iv4.y;
                                        _vec_load_13[0 + 2] = _iv4.z;
                                        _vec_load_13[0 + 3] = _iv4.w;
                                    }
                                    int _vec_load_14[4];
                                    {
                                        int4 _iv4 = *reinterpret_cast<const int4*>(scatter_src_i32 + (src_activation_word + sc_lane_word + 768) + 0);
                                        _vec_load_14[0 + 0] = _iv4.x;
                                        _vec_load_14[0 + 1] = _iv4.y;
                                        _vec_load_14[0 + 2] = _iv4.z;
                                        _vec_load_14[0 + 3] = _iv4.w;
                                    }
                                    int _vec_load_15[4];
                                    {
                                        int4 _iv4 = *reinterpret_cast<const int4*>(scatter_src_i32 + (src_activation_word + sc_lane_word + 896) + 0);
                                        _vec_load_15[0 + 0] = _iv4.x;
                                        _vec_load_15[0 + 1] = _iv4.y;
                                        _vec_load_15[0 + 2] = _iv4.z;
                                        _vec_load_15[0 + 3] = _iv4.w;
                                    }
                                    {
                                        int4 _iv4 = make_int4(_vec_load_8[0 + 0], _vec_load_8[0 + 1], _vec_load_8[0 + 2], _vec_load_8[0 + 3]);
                                        *reinterpret_cast<int4*>(pool_fp8_u32 + (dst_activation_word + sc_lane_word) + 0) = _iv4;
                                    }
                                    {
                                        int4 _iv4 = make_int4(_vec_load_9[0 + 0], _vec_load_9[0 + 1], _vec_load_9[0 + 2], _vec_load_9[0 + 3]);
                                        *reinterpret_cast<int4*>(pool_fp8_u32 + (dst_activation_word + sc_lane_word + 128) + 0) = _iv4;
                                    }
                                    {
                                        int4 _iv4 = make_int4(_vec_load_10[0 + 0], _vec_load_10[0 + 1], _vec_load_10[0 + 2], _vec_load_10[0 + 3]);
                                        *reinterpret_cast<int4*>(pool_fp8_u32 + (dst_activation_word + sc_lane_word + 256) + 0) = _iv4;
                                    }
                                    {
                                        int4 _iv4 = make_int4(_vec_load_11[0 + 0], _vec_load_11[0 + 1], _vec_load_11[0 + 2], _vec_load_11[0 + 3]);
                                        *reinterpret_cast<int4*>(pool_fp8_u32 + (dst_activation_word + sc_lane_word + 384) + 0) = _iv4;
                                    }
                                    {
                                        int4 _iv4 = make_int4(_vec_load_12[0 + 0], _vec_load_12[0 + 1], _vec_load_12[0 + 2], _vec_load_12[0 + 3]);
                                        *reinterpret_cast<int4*>(pool_fp8_u32 + (dst_activation_word + sc_lane_word + 512) + 0) = _iv4;
                                    }
                                    {
                                        int4 _iv4 = make_int4(_vec_load_13[0 + 0], _vec_load_13[0 + 1], _vec_load_13[0 + 2], _vec_load_13[0 + 3]);
                                        *reinterpret_cast<int4*>(pool_fp8_u32 + (dst_activation_word + sc_lane_word + 640) + 0) = _iv4;
                                    }
                                    {
                                        int4 _iv4 = make_int4(_vec_load_14[0 + 0], _vec_load_14[0 + 1], _vec_load_14[0 + 2], _vec_load_14[0 + 3]);
                                        *reinterpret_cast<int4*>(pool_fp8_u32 + (dst_activation_word + sc_lane_word + 768) + 0) = _iv4;
                                    }
                                    {
                                        int4 _iv4 = make_int4(_vec_load_15[0 + 0], _vec_load_15[0 + 1], _vec_load_15[0 + 2], _vec_load_15[0 + 3]);
                                        *reinterpret_cast<int4*>(pool_fp8_u32 + (dst_activation_word + sc_lane_word + 896) + 0) = _iv4;
                                    }
                                    unsigned long long src_scale_word = scatter_record_word + 1056;
                                    #pragma unroll 1
                                    for (int scale_word_2 = lane; scale_word_2 < 32; scale_word_2 += 32) {
                                        pool_sf_u32[(unsigned long long)scale_word_2 * 212928 + (unsigned long long)scatter_dst_row] = (unsigned int)reinterpret_cast<const int*>(scatter_src_i32 + (src_scale_word + (unsigned long long)scale_word_2))[0];
                                    }
                                }
                                __syncwarp();
                            } else if (lane == 0) {
                                atomicMax(&protocol_error[0], 1);
                            }
                        }
                    }
                }
                __threadfence();
                if (elect_sync()) {
                    {
                        unsigned int* _gc_p = reinterpret_cast<unsigned int*>(dispatch_chunk_scatter_counter) + (source_5 * 8 + scatter_chunk);
                        unsigned int _gc_old;
                        asm volatile("atom.release.gpu.global.add.u32 %0, [%1], 1;" : "=r"(_gc_old) : "l"(_gc_p) : "memory");
                    }
                    #pragma unroll
                    for (int sc_r = 0; sc_r < 6; sc_r++) {
                        if ((sc_tvalid >> sc_r & 1) != 0) {
                            int sc_word = sc_tpk0 * (int)(sc_r < 2) + sc_tpk1 * (int)(sc_r >= 2 && sc_r < 4) + sc_tpk2 * (int)(sc_r >= 4);
                            int sc_rt = sc_word >> 12 * (sc_r & 1) & 4095;
                            atomicAdd(&task_rows_landed[sc_rt], 1);
                        }
                    }
                }
            }
        }
        if (warp == 9 && scatter_group_rank == 0 && scatter_source == rank) {
            if (elect_sync()) {
                int stamp_count = source_record_counts[scatter_source];
                if (stamp_count > 0) {
                    int _max_46 = (((stamp_count + 8 - 1) / 8) > (dispatch_chunk_min_records) ? ((stamp_count + 8 - 1) / 8) : (dispatch_chunk_min_records));
                    int stamp_q = _max_46;
                    int stamp_last_chunk = (stamp_count + stamp_q - 1) / stamp_q - 1;
                    // gin_wait_signal: acquire, rolling 64-bit comparison
                    {
                        ncclGin __gin{*(gin_dev_comm), (int)(0)};
                        __gin.waitSignal(ncclCoopThread(), (ncclGinSignal_t)(12 + scatter_source * 8 + stamp_last_chunk), (uint64_t)(dispatch_chunk_signal_base_scratch[scatter_source * 8 + stamp_last_chunk] + 1), 64, cuda::memory_order_acquire);
                    }
                    unsigned long long gtimer_0_7;
                    asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(gtimer_0_7) :: "memory");
                    peer_phase_timestamps[scatter_source] = gtimer_0_7;
                }
            }
        }
    }
    if (warp >= 9) {
        unsigned long long gtimer_0_8;
        asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(gtimer_0_8) :: "memory");
        unsigned long long c28_rejoin_ts = gtimer_0_8;
        if (elect_sync()) {
            atomicMin(&phase_timestamps[20], c28_rejoin_ts);
            atomicMax(&phase_timestamps[19], c28_rejoin_ts);
        }
    }
    if (bid == 0 && tid == 0) {
        unsigned long long gtimer_0_9;
        asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(gtimer_0_9) :: "memory");
        phase_timestamps[7] = gtimer_0_9;
    }
    if (bid == 0 && tid == 0) {
        int expected_tasks = (active_rows + 128 - 1) / 128;
        #pragma unroll 1
        for (int expert_4 = 0; expert_4 < 64; expert_4++) {
            int expert_count_2 = expert_counts[expert_4];
            expected_tasks = expected_tasks + (expert_count_2 + 128 - 1) / 128;
        }
        if (expected_tasks != total_m_tasks[0] || prefix_done[0] != 1) {
            atomicMax(&protocol_error[0], 1);
        }
    }
    if (bid == 0 && tid == 0) {
        unsigned long long gtimer_0_10;
        asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(gtimer_0_10) :: "memory");
        phase_timestamps[8] = gtimer_0_10;
    }
    if (warp == 8) {
        if (elect_sync()) {
            asm volatile("prefetch.tensormap [%0];" :: "l"((uint64_t)(W1_A)) : "memory");
            asm volatile("prefetch.tensormap [%0];" :: "l"((uint64_t)(W1_B)) : "memory");
            asm volatile("prefetch.tensormap [%0];" :: "l"((uint64_t)(W1_B8)) : "memory");
            asm volatile("prefetch.tensormap [%0];" :: "l"((uint64_t)(W1_SFA)) : "memory");
            asm volatile("prefetch.tensormap [%0];" :: "l"((uint64_t)(W1_SFB)) : "memory");
        }
    }
    if (warp == 8) {
        if (elect_sync()) {
            asm volatile("prefetch.tensormap [%0];" :: "l"((uint64_t)(W2_A)) : "memory");
            asm volatile("prefetch.tensormap [%0];" :: "l"((uint64_t)(W2_B)) : "memory");
            asm volatile("prefetch.tensormap [%0];" :: "l"((uint64_t)(W2_B8)) : "memory");
            asm volatile("prefetch.tensormap [%0];" :: "l"((uint64_t)(W2_SFA)) : "memory");
            asm volatile("prefetch.tensormap [%0];" :: "l"((uint64_t)(W2_SFB)) : "memory");
        }
    }
    unsigned int _phase_w1_empty = 1;
    if (bid > 0 && warp == 8) {
        unsigned int w1_load_stage = 0;
        int w1_gate_done_packed = 0;
        int c56_total = total_m_tasks[0];
        int c56_bound = c56_total * 64;
        int _min_23 = ((4) < (c56_total) ? (4) : (c56_total));
        int c56_a_rounds = _min_23;
        int c56_a_len = c56_a_rounds * 32;
        int _max_47 = ((c56_total - 4) > (0) ? (c56_total - 4) : (0));
        int c56_b_len = _max_47 * 64;
        int _max_48 = ((c56_total) > (4) ? (c56_total) : (4));
        int c56_c_base = _max_48;
        int c56_final = 0;
        int c56_pre_bound = c56_a_len + c56_b_len;
        int c56_seq = 0;
        int c56_mb_base = (bid - 1) * 32;
        #pragma unroll 1
        for (int c56_iter = 0; c56_iter < 106752; c56_iter++) {
            int c56_q = 0;
            if (elect_sync()) {
                int _atomic_old_3 = atomicAdd(&c56_claim_cursor[0], 1);
                c56_q = _atomic_old_3;
            }
            int _shfl_13 = __shfl_sync(0xFFFFFFFF, c56_q, 0);
            c56_q = _shfl_13;
            if (c56_final == 0 && c56_q >= c56_pre_bound) {
                if (elect_sync()) {
                    {
                        unsigned int* _gca_p = reinterpret_cast<unsigned int*>(c24_front_sync) + (3);
                        while (true) {
                            unsigned int _gca_v;
                            asm volatile("ld.acquire.gpu.global.u32 %0, [%1];" : "=r"(_gca_v) : "l"(_gca_p));
                            if (_gca_v >= (unsigned int)(num_bids)) break;
                        }
                    }
                }
                __syncwarp();
                c56_total = 0;
                if (elect_sync()) {
                    int _atomic_old_4 = atomicAdd(&total_m_tasks[0], 0);
                    c56_total = _atomic_old_4;
                }
                int _shfl_14 = __shfl_sync(0xFFFFFFFF, c56_total, 0);
                c56_total = _shfl_14;
                c56_bound = c56_total * 64;
                int _min_24 = ((4) < (c56_total) ? (4) : (c56_total));
                c56_a_rounds = _min_24;
                c56_a_len = c56_a_rounds * 32;
                int _max_49 = ((c56_total - 4) > (0) ? (c56_total - 4) : (0));
                c56_b_len = _max_49 * 64;
                int _max_50 = ((c56_total) > (4) ? (c56_total) : (4));
                c56_c_base = _max_50;
                c56_final = 1;
            }
            if (c56_q >= c56_bound) {
                if (elect_sync()) {
                    int _atomic_old_5 = atomicAdd(&c56_tile_mailbox[c56_mb_base + 9], 2);
                    int c56_epi_old = _atomic_old_5;
                    int c56_sentinel = 1073741823;
                    if ((c56_epi_old & 1) != 0) {
                        c56_sentinel = 2147483647;
                    }
                    atomicMax(&c56_tile_mailbox[c56_mb_base + (c56_seq & 3)], c56_sentinel);
                }
                break;
            }
            int u_tile = 0;
            if (c56_q < c56_a_len) {
                int c56_ar = c56_q / 32;
                u_tile = c56_ar * 64 + (c56_q - c56_ar * 32);
            } else if (c56_q < c56_a_len + c56_b_len) {
                int c56_qb = c56_q - c56_a_len;
                int c56_br = c56_qb / 64;
                u_tile = (4 + c56_br) * 64 + (c56_qb - c56_br * 64);
            } else {
                int c56_qc = c56_q - c56_a_len - c56_b_len;
                int c56_cr = c56_qc / 32;
                u_tile = (c56_c_base + c56_cr) * 64 + 32 + (c56_qc - c56_cr * 32);
            }
            if (elect_sync()) {
                int _atomic_old_6 = atomicAdd(&c56_tile_mailbox[c56_mb_base + 9], 0);
                int c56_epi_flag = _atomic_old_6;
                int c56_pub = u_tile + 1;
                if ((c56_epi_flag & 1) != 0) {
                    c56_pub = c56_pub + 1073741824;
                }
                atomicMax(&c56_tile_mailbox[c56_mb_base + (c56_seq & 3)], c56_pub);
            }
            c56_seq += 1;
            int u_round = u_tile / 64;
            int u_pos = u_tile - u_round * 64;
            if (u_pos < 32 && u_round < c56_total) {
                int w1_tile = u_round * 32 + u_pos;
                int w1_task = w1_tile / 32;
                int w1_n_block = w1_tile - w1_task * 32;
                int w1_pool_row = 0;
                if (elect_sync()) {
                    int _atomic_old_7 = atomicAdd(&task_pool_row[w1_task], 0);
                    w1_pool_row = _atomic_old_7;
                }
                int _shfl_15 = __shfl_sync(0xFFFFFFFF, w1_pool_row, 0);
                w1_pool_row = _shfl_15;
                int w1_local_expert = 0;
                if (elect_sync()) {
                    int _atomic_old_8 = atomicAdd(&task_local_expert[w1_task], 0);
                    w1_local_expert = _atomic_old_8;
                }
                int _shfl_16 = __shfl_sync(0xFFFFFFFF, w1_local_expert, 0);
                w1_local_expert = _shfl_16;
                if (warp == 8) {
                    if (elect_sync()) {
                        if (w1_pool_row < 204736) {
                            int w1_need_rows = task_valid_m[w1_task];
                            #pragma unroll 1
                            for (int w1_gate_spin = 0; w1_gate_spin < 1073741824; w1_gate_spin++) {
                                int _atomic_old_9 = atomicAdd(&task_rows_landed[w1_task], 0);
                                int w1_have_rows = _atomic_old_9;
                                if (w1_have_rows >= w1_need_rows) {
                                    break;
                                }
                            }
                            __threadfence();
                        }
                    }
                }
                int w1_short_a = 0;
                if (warp == 8) {
                    if (elect_sync()) {
                        int _atomic_old_10 = atomicAdd(&task_valid_m[w1_task], 0);
                        w1_short_a = (int)(_atomic_old_10 <= 64);
                    }
                }
                int _shfl_17 = __shfl_sync(0xFFFFFFFF, w1_short_a, 0);
                w1_short_a = _shfl_17;
                #pragma unroll 1
                for (int w1_k_block = 0; w1_k_block < 32; w1_k_block++) {
                    mbarrier_wait(w1_empty_addr + (w1_load_stage) * 8, _phase_w1_empty);
                    if (warp == 8) {
                        if (elect_sync()) {
                            if (w1_short_a != 0) {
                                tma_2d_gmem2smem(w1_smem_sfa_addr + w1_load_stage * 512, W1_SFA64, w1_pool_row, w1_k_block, w1_full_addr + (w1_load_stage) * 8);
                            } else {
                                tma_2d_gmem2smem(w1_smem_sfa_addr + w1_load_stage * 512, W1_SFA, w1_pool_row, w1_k_block, w1_full_addr + (w1_load_stage) * 8);
                            }
                            tma_2d_gmem2smem(w1_smem_sfb_addr + w1_load_stage * 512, W1_SFB, w1_n_block * 128, w1_local_expert * 32 + w1_k_block, w1_full_addr + (w1_load_stage) * 8);
                            if (w1_short_a != 0) {
                                tma_2d_gmem2smem(w1_smem_a_addr + w1_load_stage * 16384, W1_A64, w1_k_block * 128, w1_pool_row, w1_full_addr + (w1_load_stage) * 8);
                            } else {
                                tma_2d_gmem2smem(w1_smem_a_addr + w1_load_stage * 16384, W1_A, w1_k_block * 128, w1_pool_row, w1_full_addr + (w1_load_stage) * 8);
                            }
                            if (w1_local_expert == 64) {
                                tma_2d_gmem2smem(w1_smem_b_addr + w1_load_stage * 16384, W1_B8, w1_k_block * 128, w1_n_block * 128, w1_full_addr + (w1_load_stage) * 8);
                                mbarrier_arrive_expect_tx(w1_full_addr + (w1_load_stage) * 8, 33792 - w1_short_a * 8448);
                                if (w1_k_block + 2 < 32) {
                                    asm volatile("cp.async.bulk.prefetch.tensor.2d.L2.global.tile [%0, {%1, %2}];" :: "l"((uint64_t)(W1_B8)), "r"((int)((w1_k_block + 2) * 128)), "r"((int)(w1_n_block * 128)) : "memory");
                                    if (w1_short_a != 0) {
                                        asm volatile("cp.async.bulk.prefetch.tensor.2d.L2.global.tile [%0, {%1, %2}];" :: "l"((uint64_t)(W1_A64)), "r"((int)((w1_k_block + 2) * 128)), "r"((int)(w1_pool_row)) : "memory");
                                    } else {
                                        asm volatile("cp.async.bulk.prefetch.tensor.2d.L2.global.tile [%0, {%1, %2}];" :: "l"((uint64_t)(W1_A)), "r"((int)((w1_k_block + 2) * 128)), "r"((int)(w1_pool_row)) : "memory");
                                    }
                                }
                            } else {
                                tma_2d_gmem2smem(w1_smem_b_addr + w1_load_stage * 16384, W1_B, w1_k_block * 128, w1_local_expert * 4096 + w1_n_block * 128, w1_full_addr + (w1_load_stage) * 8);
                                mbarrier_arrive_expect_tx(w1_full_addr + (w1_load_stage) * 8, 25600 - w1_short_a * 8448);
                                if (w1_k_block + 2 < 32) {
                                    asm volatile("cp.async.bulk.prefetch.tensor.2d.L2.global.tile [%0, {%1, %2}];" :: "l"((uint64_t)(W1_B)), "r"((int)((w1_k_block + 2) * 128)), "r"((int)(w1_local_expert * 4096 + w1_n_block * 128)) : "memory");
                                    if (w1_short_a != 0) {
                                        asm volatile("cp.async.bulk.prefetch.tensor.2d.L2.global.tile [%0, {%1, %2}];" :: "l"((uint64_t)(W1_A64)), "r"((int)((w1_k_block + 2) * 128)), "r"((int)(w1_pool_row)) : "memory");
                                    } else {
                                        asm volatile("cp.async.bulk.prefetch.tensor.2d.L2.global.tile [%0, {%1, %2}];" :: "l"((uint64_t)(W1_A)), "r"((int)((w1_k_block + 2) * 128)), "r"((int)(w1_pool_row)) : "memory");
                                    }
                                }
                            }
                        }
                    }
                    w1_load_stage += 1;
                    if (w1_load_stage == 2) { w1_load_stage = 0; _phase_w1_empty ^= 1; }
                }
            }
            if (u_pos >= 32 && u_round >= 4 && u_round < c56_total + 4) {
                int w2_tile = (u_round - 4) * 32 + (u_pos - 32);
                int w2_task = w2_tile / 32;
                int w2_n_block = w2_tile - w2_task * 32;
                int w2_pool_row = 0;
                if (elect_sync()) {
                    int _atomic_old_11 = atomicAdd(&task_pool_row[w2_task], 0);
                    w2_pool_row = _atomic_old_11;
                }
                int _shfl_18 = __shfl_sync(0xFFFFFFFF, w2_pool_row, 0);
                w2_pool_row = _shfl_18;
                int w2_local_expert = 0;
                if (elect_sync()) {
                    int _atomic_old_12 = atomicAdd(&task_local_expert[w2_task], 0);
                    w2_local_expert = _atomic_old_12;
                }
                int _shfl_19 = __shfl_sync(0xFFFFFFFF, w2_local_expert, 0);
                w2_local_expert = _shfl_19;
                if (warp == 8) {
                    if (elect_sync()) {
                        {
                            unsigned int* _gca_p = reinterpret_cast<unsigned int*>(w1_task_counter) + (w2_task);
                            while (true) {
                                unsigned int _gca_v;
                                asm volatile("ld.acquire.gpu.global.u32 %0, [%1];" : "=r"(_gca_v) : "l"(_gca_p));
                                if (_gca_v >= (unsigned int)(32)) break;
                            }
                        }
                    }
                }
                int w2_short_a = 0;
                if (warp == 8) {
                    if (elect_sync()) {
                        int _atomic_old_13 = atomicAdd(&task_valid_m[w2_task], 0);
                        w2_short_a = (int)(_atomic_old_13 <= 64);
                    }
                }
                int _shfl_20 = __shfl_sync(0xFFFFFFFF, w2_short_a, 0);
                w2_short_a = _shfl_20;
                #pragma unroll 1
                for (int w2_k_block = 0; w2_k_block < 16; w2_k_block++) {
                    mbarrier_wait(w1_empty_addr + (w1_load_stage) * 8, _phase_w1_empty);
                    if (warp == 8) {
                        if (elect_sync()) {
                            if (w2_short_a != 0) {
                                tma_2d_gmem2smem(w1_smem_sfa_addr + w1_load_stage * 512, W2_SFA64, w2_pool_row, w2_k_block, w1_full_addr + (w1_load_stage) * 8);
                            } else {
                                tma_2d_gmem2smem(w1_smem_sfa_addr + w1_load_stage * 512, W2_SFA, w2_pool_row, w2_k_block, w1_full_addr + (w1_load_stage) * 8);
                            }
                            tma_2d_gmem2smem(w1_smem_sfb_addr + w1_load_stage * 512, W2_SFB, w2_n_block * 128, w2_local_expert * 16 + w2_k_block, w1_full_addr + (w1_load_stage) * 8);
                            if (w2_short_a != 0) {
                                tma_2d_gmem2smem(w1_smem_a_addr + w1_load_stage * 16384, W2_A64, w2_k_block * 128, w2_pool_row, w1_full_addr + (w1_load_stage) * 8);
                            } else {
                                tma_2d_gmem2smem(w1_smem_a_addr + w1_load_stage * 16384, W2_A, w2_k_block * 128, w2_pool_row, w1_full_addr + (w1_load_stage) * 8);
                            }
                            if (w2_local_expert == 64) {
                                tma_2d_gmem2smem(w1_smem_b_addr + w1_load_stage * 16384, W2_B8, w2_k_block * 128, w2_n_block * 128, w1_full_addr + (w1_load_stage) * 8);
                                mbarrier_arrive_expect_tx(w1_full_addr + (w1_load_stage) * 8, 33792 - w2_short_a * 8448);
                                if (w2_k_block + 2 < 16) {
                                    asm volatile("cp.async.bulk.prefetch.tensor.2d.L2.global.tile [%0, {%1, %2}];" :: "l"((uint64_t)(W2_B8)), "r"((int)((w2_k_block + 2) * 128)), "r"((int)(w2_n_block * 128)) : "memory");
                                    if (w2_short_a != 0) {
                                        asm volatile("cp.async.bulk.prefetch.tensor.2d.L2.global.tile [%0, {%1, %2}];" :: "l"((uint64_t)(W2_A64)), "r"((int)((w2_k_block + 2) * 128)), "r"((int)(w2_pool_row)) : "memory");
                                    } else {
                                        asm volatile("cp.async.bulk.prefetch.tensor.2d.L2.global.tile [%0, {%1, %2}];" :: "l"((uint64_t)(W2_A)), "r"((int)((w2_k_block + 2) * 128)), "r"((int)(w2_pool_row)) : "memory");
                                    }
                                }
                            } else {
                                tma_2d_gmem2smem(w1_smem_b_addr + w1_load_stage * 16384, W2_B, w2_k_block * 128, w2_local_expert * 4096 + w2_n_block * 128, w1_full_addr + (w1_load_stage) * 8);
                                mbarrier_arrive_expect_tx(w1_full_addr + (w1_load_stage) * 8, 25600 - w2_short_a * 8448);
                                if (w2_k_block + 2 < 16) {
                                    asm volatile("cp.async.bulk.prefetch.tensor.2d.L2.global.tile [%0, {%1, %2}];" :: "l"((uint64_t)(W2_B)), "r"((int)((w2_k_block + 2) * 128)), "r"((int)(w2_local_expert * 4096 + w2_n_block * 128)) : "memory");
                                    if (w2_short_a != 0) {
                                        asm volatile("cp.async.bulk.prefetch.tensor.2d.L2.global.tile [%0, {%1, %2}];" :: "l"((uint64_t)(W2_A64)), "r"((int)((w2_k_block + 2) * 128)), "r"((int)(w2_pool_row)) : "memory");
                                    } else {
                                        asm volatile("cp.async.bulk.prefetch.tensor.2d.L2.global.tile [%0, {%1, %2}];" :: "l"((uint64_t)(W2_A)), "r"((int)((w2_k_block + 2) * 128)), "r"((int)(w2_pool_row)) : "memory");
                                    }
                                }
                            }
                        }
                    }
                    w1_load_stage += 1;
                    if (w1_load_stage == 2) { w1_load_stage = 0; _phase_w1_empty ^= 1; }
                }
            }
        }
    }
    unsigned int _phase_epi_free_0 = 1;
    unsigned int _phase_w1_full = 0;
    if (bid > 0 && warp < 8) {
        unsigned int w1_math_stage = 0;
        int w1_warp_m = warp / 2;
        int w1_warp_n = warp % 2;
        int w1_group_id = lane / 4;
        int w1_thread_id = lane % 4;
        float w1_accum[64];
        unsigned int w1_a_frag[8];
        unsigned int w1_b_frag[16];
        unsigned int w1_b8_frag[16];
        unsigned int w1_sfa_word[1];
        unsigned int w1_sfb_word[1];
        unsigned int w1_sfa_arr[2];
        unsigned int w1_sfb_arr[8];
        uint8_t w1_sfa_byte[1];
        uint8_t w1_sfb_byte[1];
        float w1_routed_a[8];
        float w1_routed_b[8];
        int w1_math_gate_done_packed = 0;
        int c56m_last = 0;
        int c56m_seq = 0;
        int c56m_mb_base = (bid - 1) * 32;
        #pragma unroll 1
        for (int c56m_iter = 0; c56m_iter < 106752; c56m_iter++) {
            int c56m_val = 0;
            if (elect_sync()) {
                #pragma unroll 1
                for (int c56m_spin = 0; c56m_spin < 1073741824; c56m_spin++) {
                    int _atomic_old_14 = atomicAdd(&c56_tile_mailbox[c56m_mb_base + (c56m_seq & 3)], 0);
                    int c56m_probe = _atomic_old_14;
                    if (c56m_probe > c56m_last) {
                        c56m_val = c56m_probe;
                        break;
                    }
                }
            }
            int _shfl_21 = __shfl_sync(0xFFFFFFFF, c56m_val, 0);
            c56m_val = _shfl_21;
            if (c56m_val == 2147483647) {
                mbarrier_wait(epi_free_addr, _phase_epi_free_0);
                _phase_epi_free_0 ^= 1;
                if (warp == 0) {
                    if (elect_sync()) {
                        atomicMax(&c56_tile_mailbox[c56m_mb_base + 8], 2147483647);
                    }
                }
                mbarrier_arrive(epi_full_addr);
                break;
            }
            if (c56m_val == 1073741823) {
                break;
            }
            if (c56m_val <= c56m_last) {
                if (elect_sync()) {
                    atomicMax(&protocol_error[0], 1);
                }
                break;
            }
            c56m_last = c56m_val;
            c56m_seq += 1;
            int c56m_mode = c56m_val >> 30 & 1;
            int u_tile_2 = (c56m_val & 1073741823) - 1;
            int u_round_2 = u_tile_2 / 64;
            int u_pos_2 = u_tile_2 - u_round_2 * 64;
            int c56m_total = 0;
            if (elect_sync()) {
                int _atomic_old_15 = atomicAdd(&total_m_tasks[0], 0);
                c56m_total = _atomic_old_15;
            }
            int _shfl_22 = __shfl_sync(0xFFFFFFFF, c56m_total, 0);
            c56m_total = _shfl_22;
            if (u_pos_2 < 32 && u_round_2 < c56m_total) {
                int w1_tile_2 = u_round_2 * 32 + u_pos_2;
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
                w1_accum[32] = 0.0f;
                w1_accum[33] = 0.0f;
                w1_accum[34] = 0.0f;
                w1_accum[35] = 0.0f;
                w1_accum[36] = 0.0f;
                w1_accum[37] = 0.0f;
                w1_accum[38] = 0.0f;
                w1_accum[39] = 0.0f;
                w1_accum[40] = 0.0f;
                w1_accum[41] = 0.0f;
                w1_accum[42] = 0.0f;
                w1_accum[43] = 0.0f;
                w1_accum[44] = 0.0f;
                w1_accum[45] = 0.0f;
                w1_accum[46] = 0.0f;
                w1_accum[47] = 0.0f;
                w1_accum[48] = 0.0f;
                w1_accum[49] = 0.0f;
                w1_accum[50] = 0.0f;
                w1_accum[51] = 0.0f;
                w1_accum[52] = 0.0f;
                w1_accum[53] = 0.0f;
                w1_accum[54] = 0.0f;
                w1_accum[55] = 0.0f;
                w1_accum[56] = 0.0f;
                w1_accum[57] = 0.0f;
                w1_accum[58] = 0.0f;
                w1_accum[59] = 0.0f;
                w1_accum[60] = 0.0f;
                w1_accum[61] = 0.0f;
                w1_accum[62] = 0.0f;
                w1_accum[63] = 0.0f;
                int w1_task_2 = w1_tile_2 / 32;
                int w1_n_block_2 = w1_tile_2 - w1_task_2 * 32;
                int w1_pool_row_2 = 0;
                if (elect_sync()) {
                    int _atomic_old_16 = atomicAdd(&task_pool_row[w1_task_2], 0);
                    w1_pool_row_2 = _atomic_old_16;
                }
                int _shfl_23 = __shfl_sync(0xFFFFFFFF, w1_pool_row_2, 0);
                w1_pool_row_2 = _shfl_23;
                int w1_valid_m_2 = 0;
                if (elect_sync()) {
                    int _atomic_old_17 = atomicAdd(&task_valid_m[w1_task_2], 0);
                    w1_valid_m_2 = _atomic_old_17;
                }
                int _shfl_24 = __shfl_sync(0xFFFFFFFF, w1_valid_m_2, 0);
                w1_valid_m_2 = _shfl_24;
                int w1_slab_active = (int)(w1_valid_m_2 > w1_warp_m * 32);
                int w1_n_inactive = 2 * (4 - (w1_valid_m_2 + 31) / 32);
                int w1_arrive_count = 1;
                if (warp == 0) {
                    w1_arrive_count = 1 + w1_n_inactive;
                }
                if (elect_sync()) {
                    if (w1_pool_row_2 < 204736) {
                        int w1_need_rows_2 = w1_valid_m_2;
                        #pragma unroll 1
                        for (int w1_gate_spin_2 = 0; w1_gate_spin_2 < 1073741824; w1_gate_spin_2++) {
                            int _atomic_old_18 = atomicAdd(&task_rows_landed[w1_task_2], 0);
                            int w1_have_rows_2 = _atomic_old_18;
                            if (w1_have_rows_2 >= w1_need_rows_2) {
                                break;
                            }
                        }
                        __threadfence();
                    }
                }
                __syncwarp();
                unsigned long long gtimer_0_11;
                asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(gtimer_0_11) :: "memory");
                unsigned long long c28_w1_ts = gtimer_0_11;
                if (elect_sync()) {
                    if (w1_pool_row_2 >= 204736) {
                        atomicMin(&phase_timestamps[21], c28_w1_ts);
                    } else {
                        atomicMin(&phase_timestamps[17], c28_w1_ts);
                    }
                }
                int w1_shared_tile_2 = (int)(w1_pool_row_2 >= 204736);
                if (w1_shared_tile_2 != 0) {
                    if (w1_slab_active != 0) {
                        #pragma unroll 1
                        for (int w1_k_block_2_s8 = 0; w1_k_block_2_s8 < 32; w1_k_block_2_s8++) {
                            mbarrier_wait(w1_full_addr + (w1_math_stage) * 8, _phase_w1_full);
                            asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                            #pragma unroll
                            for (int w1_sfl_mt_s8 = 0; w1_sfl_mt_s8 < 2; w1_sfl_mt_s8++) {
                                int w1_sfl_row_s8 = w1_warp_m * 32 + w1_sfl_mt_s8 * 16 + w1_group_id + (w1_thread_id & 1) * 8;
                                asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&w1_sfa_word[0])) : "r"(w1_smem_sfa_addr + w1_math_stage * 512 + (unsigned int)(w1_sfl_row_s8 * 4)));
                                w1_sfa_arr[w1_sfl_mt_s8] = w1_sfa_word[0];
                            }
                            #pragma unroll
                            for (int w1_sfl_nt_s8 = 0; w1_sfl_nt_s8 < 8; w1_sfl_nt_s8++) {
                                int w1_sfl_gnt_s8 = w1_warp_n * 8 + w1_sfl_nt_s8;
                                int w1_sfl_brow_s8 = w1_sfl_gnt_s8 * 8 + w1_group_id;
                                asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&w1_sfb_word[0])) : "r"(w1_smem_sfb_addr + w1_math_stage * 512 + (unsigned int)(w1_sfl_brow_s8 * 4)));
                                w1_sfb_arr[w1_sfl_nt_s8] = w1_sfb_word[0];
                            }
                            #pragma unroll
                            for (int w1_k_step_s8 = 0; w1_k_step_s8 < 4; w1_k_step_s8++) {
                                #pragma unroll
                                for (int w1_mt_s8 = 0; w1_mt_s8 < 2; w1_mt_s8++) {
                                    int w1_a_row_s8 = (lane & 7) + (lane >> 3 & 1) * 8 + w1_warp_m * 32 + w1_mt_s8 * 16;
                                    int w1_a_col_s8 = (lane >> 4) * 16 + w1_k_step_s8 * 32;
                                    int w1_a_addr_s8 = w1_smem_a_addr + w1_math_stage * 16384 + (unsigned int)(w1_a_row_s8 * 128) + (unsigned int)(w1_a_col_s8 ^ (w1_a_row_s8 & 7) << 4);
                                    asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                                        : "=r"(w1_a_frag[w1_mt_s8 * 4]), "=r"(w1_a_frag[w1_mt_s8 * 4 + 1]), "=r"(w1_a_frag[w1_mt_s8 * 4 + 2]), "=r"(w1_a_frag[w1_mt_s8 * 4 + 3])
                                        : "r"(w1_a_addr_s8)
                                        : "memory");
                                }
                                #pragma unroll
                                for (int w1_n_tile_s8 = 0; w1_n_tile_s8 < 8; w1_n_tile_s8++) {
                                    int w1_global_n_tile_s8 = w1_warp_n * 8 + w1_n_tile_s8;
                                    int w1_b_row_s8 = (lane & 7) + w1_global_n_tile_s8 * 8;
                                    int w1_b_col_s8 = (lane >> 3 & 1) * 16 + w1_k_step_s8 * 32;
                                    int w1_b_addr_s8 = w1_smem_b_addr + w1_math_stage * 16384 + (unsigned int)(w1_b_row_s8 * 128) + (unsigned int)(w1_b_col_s8 ^ (w1_b_row_s8 & 7) << 4);
                                    asm volatile("ldmatrix.sync.aligned.m8n8.x2.shared.b16 {%0, %1}, [%2];\n"
                                        : "=r"(w1_b8_frag[w1_n_tile_s8 * 2]), "=r"(w1_b8_frag[w1_n_tile_s8 * 2 + 1])
                                        : "r"(w1_b_addr_s8)
                                        : "memory");
                                }
                                #pragma unroll
                                for (int w1_mt_s8_1 = 0; w1_mt_s8_1 < 2; w1_mt_s8_1++) {
                                    w1_sfa_byte[0] = w1_sfa_arr[w1_mt_s8_1] >> (unsigned int)(w1_k_step_s8 * 8) & 255;
                                    #pragma unroll
                                    for (int w1_n_tile_s8_1 = 0; w1_n_tile_s8_1 < 8; w1_n_tile_s8_1++) {
                                        w1_sfb_byte[0] = w1_sfb_arr[w1_n_tile_s8_1] >> (unsigned int)(w1_k_step_s8 * 8) & 255;
                                        asm volatile("mma.sync.aligned.kind::mxf8f6f4.block_scale.scale_vec::1X.m16n8k32.row.col.f32.e4m3.e4m3.f32.ue8m0 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3}, {%10}, {%11, %12}, {%13}, {%14, %15};\n"
                                            : "+f"((w1_accum + (w1_mt_s8_1 * 8 + w1_n_tile_s8_1) * 4)[0]), "+f"((w1_accum + (w1_mt_s8_1 * 8 + w1_n_tile_s8_1) * 4)[1]), "+f"((w1_accum + (w1_mt_s8_1 * 8 + w1_n_tile_s8_1) * 4)[2]), "+f"((w1_accum + (w1_mt_s8_1 * 8 + w1_n_tile_s8_1) * 4)[3])
                                            : "r"((w1_a_frag + w1_mt_s8_1 * 4)[0]), "r"((w1_a_frag + w1_mt_s8_1 * 4)[1]), "r"((w1_a_frag + w1_mt_s8_1 * 4)[2]), "r"((w1_a_frag + w1_mt_s8_1 * 4)[3]), "r"((w1_b8_frag + w1_n_tile_s8_1 * 2)[0]), "r"((w1_b8_frag + w1_n_tile_s8_1 * 2)[1]), "r"((uint32_t)(w1_sfa_byte[0])), "h"((uint16_t)0), "h"((uint16_t)0), "r"((uint32_t)(w1_sfb_byte[0])), "h"((uint16_t)0), "h"((uint16_t)0));
                                    }
                                }
                            }
                            __syncwarp();
                            asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                            if (elect_sync()) {
                                asm volatile(
                                    "mbarrier.arrive.shared::cta.b64 _, [%0], %1;"
                                    :: "r"(w1_empty_addr + (w1_math_stage) * 8), "r"((uint32_t)(w1_arrive_count)) : "memory");
                            }
                            w1_math_stage += 1;
                            if (w1_math_stage == 2) { w1_math_stage = 0; _phase_w1_empty ^= 1; _phase_w1_full ^= 1; }
                        }
                    }
                } else if (w1_slab_active != 0) {
                    #pragma unroll 1
                    for (int w1_k_block_2 = 0; w1_k_block_2 < 32; w1_k_block_2++) {
                        mbarrier_wait(w1_full_addr + (w1_math_stage) * 8, _phase_w1_full);
                        asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                        #pragma unroll
                        for (int w1_sfl_mt = 0; w1_sfl_mt < 2; w1_sfl_mt++) {
                            int w1_sfl_row = w1_warp_m * 32 + w1_sfl_mt * 16 + w1_group_id + (w1_thread_id & 1) * 8;
                            asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&w1_sfa_word[0])) : "r"(w1_smem_sfa_addr + w1_math_stage * 512 + (unsigned int)(w1_sfl_row * 4)));
                            w1_sfa_arr[w1_sfl_mt] = w1_sfa_word[0];
                        }
                        #pragma unroll
                        for (int w1_sfl_nt = 0; w1_sfl_nt < 8; w1_sfl_nt++) {
                            int w1_sfl_gnt = w1_warp_n * 8 + w1_sfl_nt;
                            int w1_sfl_brow = w1_sfl_gnt * 8 + w1_group_id;
                            asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&w1_sfb_word[0])) : "r"(w1_smem_sfb_addr + w1_math_stage * 512 + (unsigned int)(w1_sfl_brow * 4)));
                            w1_sfb_arr[w1_sfl_nt] = w1_sfb_word[0];
                        }
                        #pragma unroll
                        for (int w1_k_step = 0; w1_k_step < 4; w1_k_step++) {
                            #pragma unroll
                            for (int w1_mt = 0; w1_mt < 2; w1_mt++) {
                                int w1_a_row = (lane & 7) + (lane >> 3 & 1) * 8 + w1_warp_m * 32 + w1_mt * 16;
                                int w1_a_col = (lane >> 4) * 16 + w1_k_step * 32;
                                int w1_a_addr = w1_smem_a_addr + w1_math_stage * 16384 + (unsigned int)(w1_a_row * 128) + (unsigned int)(w1_a_col ^ (w1_a_row & 7) << 4);
                                asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                                    : "=r"(w1_a_frag[w1_mt * 4]), "=r"(w1_a_frag[w1_mt * 4 + 1]), "=r"(w1_a_frag[w1_mt * 4 + 2]), "=r"(w1_a_frag[w1_mt * 4 + 3])
                                    : "r"(w1_a_addr)
                                    : "memory");
                            }
                            #pragma unroll
                            for (int w1_n_tile = 0; w1_n_tile < 8; w1_n_tile++) {
                                int w1_global_n_tile = w1_warp_n * 8 + w1_n_tile;
                                int w1_b_row = (lane & 7) + w1_global_n_tile * 8;
                                int w1_b_col = (lane >> 3 & 1) * 16 + w1_k_step * 32;
                                int w1_b_addr = w1_smem_b_addr + w1_math_stage * 16384 + (unsigned int)(w1_b_row * 128) + (unsigned int)(w1_b_col ^ (w1_b_row & 7) << 4);
                                asm volatile("ldmatrix.sync.aligned.shared::cta.m8n16.x2.b8x16.b4x16_p64 {%0, %1}, [%2];\n"
                                    : "=r"(w1_b_frag[w1_n_tile * 2]), "=r"(w1_b_frag[w1_n_tile * 2 + 1])
                                    : "r"(w1_b_addr)
                                    : "memory");
                            }
                            #pragma unroll
                            for (int w1_mt_1 = 0; w1_mt_1 < 2; w1_mt_1++) {
                                w1_sfa_byte[0] = w1_sfa_arr[w1_mt_1] >> (unsigned int)(w1_k_step * 8) & 255;
                                #pragma unroll
                                for (int w1_n_tile_1 = 0; w1_n_tile_1 < 8; w1_n_tile_1++) {
                                    w1_sfb_byte[0] = w1_sfb_arr[w1_n_tile_1] >> (unsigned int)(w1_k_step * 8) & 255;
                                    asm volatile("mma.sync.aligned.kind::mxf8f6f4.block_scale.scale_vec::1X.m16n8k32.row.col.f32.e4m3.e2m1.f32.ue8m0 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3}, {%10}, {%11, %12}, {%13}, {%14, %15};\n"
                                        : "+f"((w1_accum + (w1_mt_1 * 8 + w1_n_tile_1) * 4)[0]), "+f"((w1_accum + (w1_mt_1 * 8 + w1_n_tile_1) * 4)[1]), "+f"((w1_accum + (w1_mt_1 * 8 + w1_n_tile_1) * 4)[2]), "+f"((w1_accum + (w1_mt_1 * 8 + w1_n_tile_1) * 4)[3])
                                        : "r"((w1_a_frag + w1_mt_1 * 4)[0]), "r"((w1_a_frag + w1_mt_1 * 4)[1]), "r"((w1_a_frag + w1_mt_1 * 4)[2]), "r"((w1_a_frag + w1_mt_1 * 4)[3]), "r"(((uint32_t)((w1_b_frag + w1_n_tile_1 * 2)[0]) << 2)), "r"(((uint32_t)((w1_b_frag + w1_n_tile_1 * 2)[1]) << 2)), "r"((uint32_t)(w1_sfa_byte[0])), "h"((uint16_t)0), "h"((uint16_t)0), "r"((uint32_t)(w1_sfb_byte[0])), "h"((uint16_t)0), "h"((uint16_t)0));
                                }
                            }
                        }
                        __syncwarp();
                        asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                        if (elect_sync()) {
                            asm volatile(
                                "mbarrier.arrive.shared::cta.b64 _, [%0], %1;"
                                :: "r"(w1_empty_addr + (w1_math_stage) * 8), "r"((uint32_t)(w1_arrive_count)) : "memory");
                        }
                        w1_math_stage += 1;
                        if (w1_math_stage == 2) { w1_math_stage = 0; _phase_w1_empty ^= 1; _phase_w1_full ^= 1; }
                    }
                }
                if (warp == 0) {
                    if (elect_sync()) {
                        asm volatile("cp.async.bulk.wait_group.read 0;");
                    }
                }
                if (c56m_mode != 0) {
                    mbarrier_wait(epi_free_addr, _phase_epi_free_0);
                    _phase_epi_free_0 ^= 1;
                }
                asm volatile("barrier.sync 15, 256;" ::: "memory");
                #pragma unroll
                for (int w1_mt_2 = 0; w1_mt_2 < 2; w1_mt_2++) {
                    int w1_stage_row_0 = w1_warp_m * 32 + w1_mt_2 * 16 + w1_group_id;
                    int w1_stage_row_1 = w1_stage_row_0 + 8;
                    #pragma unroll
                    for (int w1_n_tile_2 = 0; w1_n_tile_2 < 8; w1_n_tile_2++) {
                        int w1_output_n_tile = w1_warp_n * 8 + w1_n_tile_2;
                        int w1_local_col = w1_output_n_tile * 8 + w1_thread_id * 2;
                        int w1_acc_base = (w1_mt_2 * 8 + w1_n_tile_2) * 4;
                        int w1_sub_t = w1_local_col / 64;
                        int w1_col_in = w1_local_col - w1_sub_t * 64;
                        int w1_addr0 = w1_sub_t * 16384 + w1_stage_row_0 * 128 + (w1_col_in * 2 ^ (w1_stage_row_0 & 7) * 16);
                        int w1_addr1 = w1_sub_t * 16384 + w1_stage_row_1 * 128 + (w1_col_in * 2 ^ (w1_stage_row_1 & 7) * 16);
                        d_stage[w1_addr0 / 2] = w1_accum[w1_acc_base];
                        d_stage[w1_addr0 / 2 + 1] = w1_accum[w1_acc_base + 1];
                        d_stage[w1_addr1 / 2] = w1_accum[w1_acc_base + 2];
                        d_stage[w1_addr1 / 2 + 1] = w1_accum[w1_acc_base + 3];
                    }
                }
                asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                asm volatile("barrier.sync 15, 256;" ::: "memory");
                if (warp == 0) {
                    if (elect_sync()) {
                        tma_store_2d(W1_D, w1_n_block_2 * 128, w1_pool_row_2, d_stage_addr);
                        tma_store_2d(W1_D, w1_n_block_2 * 128 + 64, w1_pool_row_2, d_stage_addr + 16384);
                        asm volatile("cp.async.bulk.commit_group;");
                    }
                }
                if (c56m_mode != 0) {
                    if (warp == 0) {
                        if (elect_sync()) {
                            atomicMax(&c56_tile_mailbox[c56m_mb_base + 8], u_tile_2 + 1);
                        }
                    }
                    mbarrier_arrive(epi_full_addr);
                } else {
                    int w1_req_group = w1_n_block_2 * 2 + w1_warp_n;
                    #pragma unroll
                    for (int w1_fmt = 0; w1_fmt < 2; w1_fmt++) {
                        int w1_fused_row_0 = w1_pool_row_2 + w1_warp_m * 32 + w1_fmt * 16 + w1_group_id;
                        int w1_fused_row_1 = w1_fused_row_0 + 8;
                        float w1_rw_0 = routing_weight_pool[w1_fused_row_0];
                        float w1_rw_1 = routing_weight_pool[w1_fused_row_1];
                        unsigned long long w1_int_base_0 = (unsigned long long)w1_fused_row_0 * 2048;
                        unsigned long long w1_int_base_1 = (unsigned long long)w1_fused_row_1 * 2048;
                        float w1_amax_0 = 0.0f;
                        float w1_amax_1 = 0.0f;
                        #pragma unroll
                        for (int w1_rq = 0; w1_rq < 4; w1_rq++) {
                            int w1_gate_base = (w1_fmt * 8 + w1_rq * 2) * 4;
                            int w1_up_base = w1_gate_base + 4;
                            #pragma unroll
                            for (int w1_rc = 0; w1_rc < 2; w1_rc++) {
                                float w1_gate_0 = (float)(__nv_bfloat16)w1_accum[w1_gate_base + w1_rc];
                                float w1_up_0 = (float)(__nv_bfloat16)w1_accum[w1_up_base + w1_rc];
                                float _min_25 = fminf(w1_gate_0, 10.0f);
                                w1_gate_0 = _min_25;
                                float _max_51 = max_noftz(w1_up_0, -10.0f);
                                float _min_26 = fminf(_max_51, 10.0f);
                                w1_up_0 = _min_26;
                                float _exp2_0 = approx_exp2((-w1_gate_0) * 1.4426950408889634f);
                                float w1_sig_0 = 1.0f / (1.0f + _exp2_0);
                                float w1_routed_val_0 = w1_gate_0 * w1_sig_0 * w1_up_0 * w1_rw_0;
                                w1_routed_a[w1_rq * 2 + w1_rc] = w1_routed_val_0;
                                float _max_52 = max_noftz(w1_routed_val_0, -w1_routed_val_0);
                                float _max_53 = max_noftz(w1_amax_0, _max_52);
                                w1_amax_0 = _max_53;
                                float w1_gate_1 = (float)(__nv_bfloat16)w1_accum[w1_gate_base + 2 + w1_rc];
                                float w1_up_1 = (float)(__nv_bfloat16)w1_accum[w1_up_base + 2 + w1_rc];
                                float _min_27 = fminf(w1_gate_1, 10.0f);
                                w1_gate_1 = _min_27;
                                float _max_54 = max_noftz(w1_up_1, -10.0f);
                                float _min_28 = fminf(_max_54, 10.0f);
                                w1_up_1 = _min_28;
                                float _exp2_1 = approx_exp2((-w1_gate_1) * 1.4426950408889634f);
                                float w1_sig_1 = 1.0f / (1.0f + _exp2_1);
                                float w1_routed_val_1 = w1_gate_1 * w1_sig_1 * w1_up_1 * w1_rw_1;
                                w1_routed_b[w1_rq * 2 + w1_rc] = w1_routed_val_1;
                                float _max_55 = max_noftz(w1_routed_val_1, -w1_routed_val_1);
                                float _max_56 = max_noftz(w1_amax_1, _max_55);
                                w1_amax_1 = _max_56;
                            }
                        }
                        float _shfl_xor_20 = __shfl_xor_sync(0xFFFFFFFF, w1_amax_0, 2);
                        float _max_57 = max_noftz(w1_amax_0, _shfl_xor_20);
                        w1_amax_0 = _max_57;
                        float _shfl_xor_21 = __shfl_xor_sync(0xFFFFFFFF, w1_amax_0, 1);
                        float _max_58 = max_noftz(w1_amax_0, _shfl_xor_21);
                        w1_amax_0 = _max_58;
                        float _shfl_xor_22 = __shfl_xor_sync(0xFFFFFFFF, w1_amax_1, 2);
                        float _max_59 = max_noftz(w1_amax_1, _shfl_xor_22);
                        w1_amax_1 = _max_59;
                        float _shfl_xor_23 = __shfl_xor_sync(0xFFFFFFFF, w1_amax_1, 1);
                        float _max_60 = max_noftz(w1_amax_1, _shfl_xor_23);
                        w1_amax_1 = _max_60;
                        float w1_sf_0 = w1_amax_0 * 0.002232142857142857f;
                        unsigned int w1_sf_0_bits = 0;
                        w1_sf_0_bits = reinterpret_cast<unsigned int*>(&w1_sf_0)[0];
                        unsigned int w1_sf_0_exp = (w1_sf_0_bits >> 23 & 255) + ((w1_sf_0_bits & 8388607) + 8388607 >> 23);
                        unsigned int _min_29 = ((w1_sf_0_exp) < (254) ? (w1_sf_0_exp) : (254));
                        w1_sf_0_exp = _min_29;
                        unsigned int w1_sf_0_inv_bits = 254 - w1_sf_0_exp << 23;
                        float w1_sf_0_inv = 0.0f;
                        w1_sf_0_inv = reinterpret_cast<float*>(&w1_sf_0_inv_bits)[0];
                        float w1_sf_1 = w1_amax_1 * 0.002232142857142857f;
                        unsigned int w1_sf_1_bits = 0;
                        w1_sf_1_bits = reinterpret_cast<unsigned int*>(&w1_sf_1)[0];
                        unsigned int w1_sf_1_exp = (w1_sf_1_bits >> 23 & 255) + ((w1_sf_1_bits & 8388607) + 8388607 >> 23);
                        unsigned int _min_30 = ((w1_sf_1_exp) < (254) ? (w1_sf_1_exp) : (254));
                        w1_sf_1_exp = _min_30;
                        unsigned int w1_sf_1_inv_bits = 254 - w1_sf_1_exp << 23;
                        float w1_sf_1_inv = 0.0f;
                        w1_sf_1_inv = reinterpret_cast<float*>(&w1_sf_1_inv_bits)[0];
                        if (w1_thread_id == 0) {
                            int w1_sf_index_0 = ((w1_req_group >> 2) * 212928 + w1_fused_row_0) * 4 + (w1_req_group & 3);
                            *(reinterpret_cast<unsigned char*>(intermediate_sfa_u8 + w1_sf_index_0) + (0)) = (unsigned char)(w1_sf_0_exp);
                            int w1_sf_index_1 = ((w1_req_group >> 2) * 212928 + w1_fused_row_1) * 4 + (w1_req_group & 3);
                            *(reinterpret_cast<unsigned char*>(intermediate_sfa_u8 + w1_sf_index_1) + (0)) = (unsigned char)(w1_sf_1_exp);
                        }
                        #pragma unroll
                        for (int w1_sq = 0; w1_sq < 4; w1_sq++) {
                            int w1_log_n = w1_req_group * 32 + w1_sq * 8 + w1_thread_id * 2;
                            {
                                unsigned short _fp8_pair;
                                asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_fp8_pair) : "f"(w1_routed_a[w1_sq * 2 + 1] * w1_sf_0_inv), "f"(w1_routed_a[w1_sq * 2 + 0] * w1_sf_0_inv));
                                *reinterpret_cast<unsigned short*>(reinterpret_cast<unsigned char*>(intermediate_fp8 + (w1_int_base_0 + (unsigned long long)w1_log_n)) + (0)) = _fp8_pair;
                            }
                            {
                                unsigned short _fp8_pair;
                                asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_fp8_pair) : "f"(w1_routed_b[w1_sq * 2 + 1] * w1_sf_1_inv), "f"(w1_routed_b[w1_sq * 2 + 0] * w1_sf_1_inv));
                                *reinterpret_cast<unsigned short*>(reinterpret_cast<unsigned char*>(intermediate_fp8 + (w1_int_base_1 + (unsigned long long)w1_log_n)) + (0)) = _fp8_pair;
                            }
                        }
                    }
                    if (elect_sync()) {
                        atomicAdd(&requant_groups_done[0], 32);
                    }
                    __threadfence();
                    __syncwarp();
                    unsigned long long gtimer_1_1;
                    asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(gtimer_1_1) :: "memory");
                    unsigned long long c28_w1_end_ts = gtimer_1_1;
                    if (elect_sync()) {
                        if (w1_pool_row_2 >= 204736) {
                            atomicMax(&phase_timestamps[22], c28_w1_end_ts);
                        }
                    }
                    if (elect_sync()) {
                        int _atomic_old_19 = atomicAdd(&w1_warp_done[w1_tile_2], 1);
                        int w1_previous = _atomic_old_19;
                        if (w1_previous == 7) {
                            atomicAdd(&w1_tiles_completed[0], 1);
                            {
                                unsigned int* _gc_p = reinterpret_cast<unsigned int*>(w1_task_counter) + (w1_task_2);
                                unsigned int _gc_old;
                                asm volatile("atom.release.gpu.global.add.u32 %0, [%1], 1;" : "=r"(_gc_old) : "l"(_gc_p) : "memory");
                            }
                        } else if (w1_previous >= 8) {
                            atomicMax(&protocol_error[0], 1);
                        }
                    }
                }
            }
            if (u_pos_2 >= 32 && u_round_2 >= 4 && u_round_2 < c56m_total + 4) {
                int w2_tile_2 = (u_round_2 - 4) * 32 + (u_pos_2 - 32);
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
                w1_accum[32] = 0.0f;
                w1_accum[33] = 0.0f;
                w1_accum[34] = 0.0f;
                w1_accum[35] = 0.0f;
                w1_accum[36] = 0.0f;
                w1_accum[37] = 0.0f;
                w1_accum[38] = 0.0f;
                w1_accum[39] = 0.0f;
                w1_accum[40] = 0.0f;
                w1_accum[41] = 0.0f;
                w1_accum[42] = 0.0f;
                w1_accum[43] = 0.0f;
                w1_accum[44] = 0.0f;
                w1_accum[45] = 0.0f;
                w1_accum[46] = 0.0f;
                w1_accum[47] = 0.0f;
                w1_accum[48] = 0.0f;
                w1_accum[49] = 0.0f;
                w1_accum[50] = 0.0f;
                w1_accum[51] = 0.0f;
                w1_accum[52] = 0.0f;
                w1_accum[53] = 0.0f;
                w1_accum[54] = 0.0f;
                w1_accum[55] = 0.0f;
                w1_accum[56] = 0.0f;
                w1_accum[57] = 0.0f;
                w1_accum[58] = 0.0f;
                w1_accum[59] = 0.0f;
                w1_accum[60] = 0.0f;
                w1_accum[61] = 0.0f;
                w1_accum[62] = 0.0f;
                w1_accum[63] = 0.0f;
                int w2_task_2 = w2_tile_2 / 32;
                int w2_n_block_2 = w2_tile_2 - w2_task_2 * 32;
                int w2_pool_row_2 = 0;
                if (elect_sync()) {
                    int _atomic_old_20 = atomicAdd(&task_pool_row[w2_task_2], 0);
                    w2_pool_row_2 = _atomic_old_20;
                }
                int _shfl_25 = __shfl_sync(0xFFFFFFFF, w2_pool_row_2, 0);
                w2_pool_row_2 = _shfl_25;
                int w2_valid_m_2 = 0;
                if (elect_sync()) {
                    int _atomic_old_21 = atomicAdd(&task_valid_m[w2_task_2], 0);
                    w2_valid_m_2 = _atomic_old_21;
                }
                int _shfl_26 = __shfl_sync(0xFFFFFFFF, w2_valid_m_2, 0);
                w2_valid_m_2 = _shfl_26;
                int w2_slab_active = (int)(w2_valid_m_2 > w1_warp_m * 32);
                int w2_n_inactive = 2 * (4 - (w2_valid_m_2 + 31) / 32);
                int w2_arrive_count = 1;
                if (warp == 0) {
                    w2_arrive_count = 1 + w2_n_inactive;
                }
                if (elect_sync()) {
                    {
                        unsigned int* _gca_p = reinterpret_cast<unsigned int*>(w1_task_counter) + (w2_task_2);
                        while (true) {
                            unsigned int _gca_v;
                            asm volatile("ld.acquire.gpu.global.u32 %0, [%1];" : "=r"(_gca_v) : "l"(_gca_p));
                            if (_gca_v >= (unsigned int)(32)) break;
                        }
                    }
                }
                __syncwarp();
                unsigned long long gtimer_0_12;
                asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(gtimer_0_12) :: "memory");
                unsigned long long c28_w2_ts = gtimer_0_12;
                if (elect_sync()) {
                    if (w2_pool_row_2 >= 204736) {
                        atomicMin(&phase_timestamps[23], c28_w2_ts);
                    } else {
                        atomicMin(&phase_timestamps[18], c28_w2_ts);
                    }
                }
                int w2_shared_tile_2 = (int)(w2_pool_row_2 >= 204736);
                if (w2_shared_tile_2 != 0) {
                    if (w2_slab_active != 0) {
                        #pragma unroll 1
                        for (int w2_k_block_2_s8 = 0; w2_k_block_2_s8 < 16; w2_k_block_2_s8++) {
                            mbarrier_wait(w1_full_addr + (w1_math_stage) * 8, _phase_w1_full);
                            asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                            #pragma unroll
                            for (int w2_sfl_mt_s8 = 0; w2_sfl_mt_s8 < 2; w2_sfl_mt_s8++) {
                                int w2_sfl_row_s8 = w1_warp_m * 32 + w2_sfl_mt_s8 * 16 + w1_group_id + (w1_thread_id & 1) * 8;
                                asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&w1_sfa_word[0])) : "r"(w1_smem_sfa_addr + w1_math_stage * 512 + (unsigned int)(w2_sfl_row_s8 * 4)));
                                w1_sfa_arr[w2_sfl_mt_s8] = w1_sfa_word[0];
                            }
                            #pragma unroll
                            for (int w2_sfl_nt_s8 = 0; w2_sfl_nt_s8 < 8; w2_sfl_nt_s8++) {
                                int w2_sfl_gnt_s8 = w1_warp_n * 8 + w2_sfl_nt_s8;
                                int w2_sfl_brow_s8 = w2_sfl_gnt_s8 * 8 + w1_group_id;
                                asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&w1_sfb_word[0])) : "r"(w1_smem_sfb_addr + w1_math_stage * 512 + (unsigned int)(w2_sfl_brow_s8 * 4)));
                                w1_sfb_arr[w2_sfl_nt_s8] = w1_sfb_word[0];
                            }
                            #pragma unroll
                            for (int w2_k_step_s8 = 0; w2_k_step_s8 < 4; w2_k_step_s8++) {
                                #pragma unroll
                                for (int w2_mt_s8 = 0; w2_mt_s8 < 2; w2_mt_s8++) {
                                    int w2_a_row_s8 = (lane & 7) + (lane >> 3 & 1) * 8 + w1_warp_m * 32 + w2_mt_s8 * 16;
                                    int w2_a_col_s8 = (lane >> 4) * 16 + w2_k_step_s8 * 32;
                                    int w2_a_addr_s8 = w1_smem_a_addr + w1_math_stage * 16384 + (unsigned int)(w2_a_row_s8 * 128) + (unsigned int)(w2_a_col_s8 ^ (w2_a_row_s8 & 7) << 4);
                                    asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                                        : "=r"(w1_a_frag[w2_mt_s8 * 4]), "=r"(w1_a_frag[w2_mt_s8 * 4 + 1]), "=r"(w1_a_frag[w2_mt_s8 * 4 + 2]), "=r"(w1_a_frag[w2_mt_s8 * 4 + 3])
                                        : "r"(w2_a_addr_s8)
                                        : "memory");
                                }
                                #pragma unroll
                                for (int w2_n_tile_s8 = 0; w2_n_tile_s8 < 8; w2_n_tile_s8++) {
                                    int w2_global_n_tile_s8 = w1_warp_n * 8 + w2_n_tile_s8;
                                    int w2_b_row_s8 = (lane & 7) + w2_global_n_tile_s8 * 8;
                                    int w2_b_col_s8 = (lane >> 3 & 1) * 16 + w2_k_step_s8 * 32;
                                    int w2_b_addr_s8 = w1_smem_b_addr + w1_math_stage * 16384 + (unsigned int)(w2_b_row_s8 * 128) + (unsigned int)(w2_b_col_s8 ^ (w2_b_row_s8 & 7) << 4);
                                    asm volatile("ldmatrix.sync.aligned.m8n8.x2.shared.b16 {%0, %1}, [%2];\n"
                                        : "=r"(w1_b8_frag[w2_n_tile_s8 * 2]), "=r"(w1_b8_frag[w2_n_tile_s8 * 2 + 1])
                                        : "r"(w2_b_addr_s8)
                                        : "memory");
                                }
                                #pragma unroll
                                for (int w2_mt_s8_1 = 0; w2_mt_s8_1 < 2; w2_mt_s8_1++) {
                                    w1_sfa_byte[0] = w1_sfa_arr[w2_mt_s8_1] >> (unsigned int)(w2_k_step_s8 * 8) & 255;
                                    #pragma unroll
                                    for (int w2_n_tile_s8_1 = 0; w2_n_tile_s8_1 < 8; w2_n_tile_s8_1++) {
                                        w1_sfb_byte[0] = w1_sfb_arr[w2_n_tile_s8_1] >> (unsigned int)(w2_k_step_s8 * 8) & 255;
                                        asm volatile("mma.sync.aligned.kind::mxf8f6f4.block_scale.scale_vec::1X.m16n8k32.row.col.f32.e4m3.e4m3.f32.ue8m0 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3}, {%10}, {%11, %12}, {%13}, {%14, %15};\n"
                                            : "+f"((w1_accum + (w2_mt_s8_1 * 8 + w2_n_tile_s8_1) * 4)[0]), "+f"((w1_accum + (w2_mt_s8_1 * 8 + w2_n_tile_s8_1) * 4)[1]), "+f"((w1_accum + (w2_mt_s8_1 * 8 + w2_n_tile_s8_1) * 4)[2]), "+f"((w1_accum + (w2_mt_s8_1 * 8 + w2_n_tile_s8_1) * 4)[3])
                                            : "r"((w1_a_frag + w2_mt_s8_1 * 4)[0]), "r"((w1_a_frag + w2_mt_s8_1 * 4)[1]), "r"((w1_a_frag + w2_mt_s8_1 * 4)[2]), "r"((w1_a_frag + w2_mt_s8_1 * 4)[3]), "r"((w1_b8_frag + w2_n_tile_s8_1 * 2)[0]), "r"((w1_b8_frag + w2_n_tile_s8_1 * 2)[1]), "r"((uint32_t)(w1_sfa_byte[0])), "h"((uint16_t)0), "h"((uint16_t)0), "r"((uint32_t)(w1_sfb_byte[0])), "h"((uint16_t)0), "h"((uint16_t)0));
                                    }
                                }
                            }
                            __syncwarp();
                            asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                            if (elect_sync()) {
                                asm volatile(
                                    "mbarrier.arrive.shared::cta.b64 _, [%0], %1;"
                                    :: "r"(w1_empty_addr + (w1_math_stage) * 8), "r"((uint32_t)(w2_arrive_count)) : "memory");
                            }
                            w1_math_stage += 1;
                            if (w1_math_stage == 2) { w1_math_stage = 0; _phase_w1_empty ^= 1; _phase_w1_full ^= 1; }
                        }
                    }
                } else if (w2_slab_active != 0) {
                    #pragma unroll 1
                    for (int w2_k_block_2 = 0; w2_k_block_2 < 16; w2_k_block_2++) {
                        mbarrier_wait(w1_full_addr + (w1_math_stage) * 8, _phase_w1_full);
                        asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                        #pragma unroll
                        for (int w2_sfl_mt = 0; w2_sfl_mt < 2; w2_sfl_mt++) {
                            int w2_sfl_row = w1_warp_m * 32 + w2_sfl_mt * 16 + w1_group_id + (w1_thread_id & 1) * 8;
                            asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&w1_sfa_word[0])) : "r"(w1_smem_sfa_addr + w1_math_stage * 512 + (unsigned int)(w2_sfl_row * 4)));
                            w1_sfa_arr[w2_sfl_mt] = w1_sfa_word[0];
                        }
                        #pragma unroll
                        for (int w2_sfl_nt = 0; w2_sfl_nt < 8; w2_sfl_nt++) {
                            int w2_sfl_gnt = w1_warp_n * 8 + w2_sfl_nt;
                            int w2_sfl_brow = w2_sfl_gnt * 8 + w1_group_id;
                            asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&w1_sfb_word[0])) : "r"(w1_smem_sfb_addr + w1_math_stage * 512 + (unsigned int)(w2_sfl_brow * 4)));
                            w1_sfb_arr[w2_sfl_nt] = w1_sfb_word[0];
                        }
                        #pragma unroll
                        for (int w2_k_step = 0; w2_k_step < 4; w2_k_step++) {
                            #pragma unroll
                            for (int w2_mt = 0; w2_mt < 2; w2_mt++) {
                                int w2_a_row = (lane & 7) + (lane >> 3 & 1) * 8 + w1_warp_m * 32 + w2_mt * 16;
                                int w2_a_col = (lane >> 4) * 16 + w2_k_step * 32;
                                int w2_a_addr = w1_smem_a_addr + w1_math_stage * 16384 + (unsigned int)(w2_a_row * 128) + (unsigned int)(w2_a_col ^ (w2_a_row & 7) << 4);
                                asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                                    : "=r"(w1_a_frag[w2_mt * 4]), "=r"(w1_a_frag[w2_mt * 4 + 1]), "=r"(w1_a_frag[w2_mt * 4 + 2]), "=r"(w1_a_frag[w2_mt * 4 + 3])
                                    : "r"(w2_a_addr)
                                    : "memory");
                            }
                            #pragma unroll
                            for (int w2_n_tile = 0; w2_n_tile < 8; w2_n_tile++) {
                                int w2_global_n_tile = w1_warp_n * 8 + w2_n_tile;
                                int w2_b_row = (lane & 7) + w2_global_n_tile * 8;
                                int w2_b_col = (lane >> 3 & 1) * 16 + w2_k_step * 32;
                                int w2_b_addr = w1_smem_b_addr + w1_math_stage * 16384 + (unsigned int)(w2_b_row * 128) + (unsigned int)(w2_b_col ^ (w2_b_row & 7) << 4);
                                asm volatile("ldmatrix.sync.aligned.shared::cta.m8n16.x2.b8x16.b4x16_p64 {%0, %1}, [%2];\n"
                                    : "=r"(w1_b_frag[w2_n_tile * 2]), "=r"(w1_b_frag[w2_n_tile * 2 + 1])
                                    : "r"(w2_b_addr)
                                    : "memory");
                            }
                            #pragma unroll
                            for (int w2_mt_1 = 0; w2_mt_1 < 2; w2_mt_1++) {
                                w1_sfa_byte[0] = w1_sfa_arr[w2_mt_1] >> (unsigned int)(w2_k_step * 8) & 255;
                                #pragma unroll
                                for (int w2_n_tile_1 = 0; w2_n_tile_1 < 8; w2_n_tile_1++) {
                                    w1_sfb_byte[0] = w1_sfb_arr[w2_n_tile_1] >> (unsigned int)(w2_k_step * 8) & 255;
                                    asm volatile("mma.sync.aligned.kind::mxf8f6f4.block_scale.scale_vec::1X.m16n8k32.row.col.f32.e4m3.e2m1.f32.ue8m0 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3}, {%10}, {%11, %12}, {%13}, {%14, %15};\n"
                                        : "+f"((w1_accum + (w2_mt_1 * 8 + w2_n_tile_1) * 4)[0]), "+f"((w1_accum + (w2_mt_1 * 8 + w2_n_tile_1) * 4)[1]), "+f"((w1_accum + (w2_mt_1 * 8 + w2_n_tile_1) * 4)[2]), "+f"((w1_accum + (w2_mt_1 * 8 + w2_n_tile_1) * 4)[3])
                                        : "r"((w1_a_frag + w2_mt_1 * 4)[0]), "r"((w1_a_frag + w2_mt_1 * 4)[1]), "r"((w1_a_frag + w2_mt_1 * 4)[2]), "r"((w1_a_frag + w2_mt_1 * 4)[3]), "r"(((uint32_t)((w1_b_frag + w2_n_tile_1 * 2)[0]) << 2)), "r"(((uint32_t)((w1_b_frag + w2_n_tile_1 * 2)[1]) << 2)), "r"((uint32_t)(w1_sfa_byte[0])), "h"((uint16_t)0), "h"((uint16_t)0), "r"((uint32_t)(w1_sfb_byte[0])), "h"((uint16_t)0), "h"((uint16_t)0));
                                }
                            }
                        }
                        __syncwarp();
                        asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                        if (elect_sync()) {
                            asm volatile(
                                "mbarrier.arrive.shared::cta.b64 _, [%0], %1;"
                                :: "r"(w1_empty_addr + (w1_math_stage) * 8), "r"((uint32_t)(w2_arrive_count)) : "memory");
                        }
                        w1_math_stage += 1;
                        if (w1_math_stage == 2) { w1_math_stage = 0; _phase_w1_empty ^= 1; _phase_w1_full ^= 1; }
                    }
                }
                if (c56m_mode != 0) {
                    if (warp == 0) {
                        if (elect_sync()) {
                            asm volatile("cp.async.bulk.wait_group.read 0;");
                        }
                    }
                    mbarrier_wait(epi_free_addr, _phase_epi_free_0);
                    _phase_epi_free_0 ^= 1;
                    asm volatile("barrier.sync 15, 256;" ::: "memory");
                    #pragma unroll
                    for (int w2c_mt = 0; w2c_mt < 2; w2c_mt++) {
                        int w2c_stage_row_0 = w1_warp_m * 32 + w2c_mt * 16 + w1_group_id;
                        int w2c_stage_row_1 = w2c_stage_row_0 + 8;
                        #pragma unroll
                        for (int w2c_nt = 0; w2c_nt < 8; w2c_nt++) {
                            int w2c_output_n_tile = w1_warp_n * 8 + w2c_nt;
                            int w2c_acc_base = (w2c_mt * 8 + w2c_nt) * 4;
                            int w2c_local_col = w2c_output_n_tile * 8 + w1_thread_id * 2;
                            int w2c_sub_t = w2c_local_col / 64;
                            int w2c_col_in = w2c_local_col - w2c_sub_t * 64;
                            int w2c_addr0 = w2c_sub_t * 16384 + w2c_stage_row_0 * 128 + (w2c_col_in * 2 ^ (w2c_stage_row_0 & 7) * 16);
                            int w2c_addr1 = w2c_sub_t * 16384 + w2c_stage_row_1 * 128 + (w2c_col_in * 2 ^ (w2c_stage_row_1 & 7) * 16);
                            d_stage[w2c_addr0 / 2] = w1_accum[w2c_acc_base];
                            d_stage[w2c_addr0 / 2 + 1] = w1_accum[w2c_acc_base + 1];
                            d_stage[w2c_addr1 / 2] = w1_accum[w2c_acc_base + 2];
                            d_stage[w2c_addr1 / 2 + 1] = w1_accum[w2c_acc_base + 3];
                        }
                    }
                    asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                    asm volatile("barrier.sync 15, 256;" ::: "memory");
                    if (warp == 0) {
                        if (elect_sync()) {
                            tma_store_2d(W2_D, w2_n_block_2 * 128, w2_pool_row_2, d_stage_addr);
                            tma_store_2d(W2_D, w2_n_block_2 * 128 + 64, w2_pool_row_2, d_stage_addr + 16384);
                            asm volatile("cp.async.bulk.commit_group;");
                            atomicMax(&c56_tile_mailbox[c56m_mb_base + 8], u_tile_2 + 1);
                        }
                    }
                    mbarrier_arrive(epi_full_addr);
                } else {
                    if (warp == 0) {
                        if (elect_sync()) {
                            asm volatile("cp.async.bulk.wait_group.read 0;");
                        }
                    }
                    asm volatile("barrier.sync 15, 256;" ::: "memory");
                    #pragma unroll
                    for (int w2_mt_2 = 0; w2_mt_2 < 2; w2_mt_2++) {
                        int w2_ret_row_0 = w2_pool_row_2 + w1_warp_m * 32 + w2_mt_2 * 16 + w1_group_id;
                        int w2_ret_row_1 = w2_ret_row_0 + 8;
                        int w2_ret_src_0 = meta_source_rank[w2_ret_row_0];
                        int w2_ret_src_1 = meta_source_rank[w2_ret_row_1];
                        int w2_ret_valid_0 = (int)(w2_ret_src_0 >= 0 && w2_ret_src_0 < world_size);
                        int w2_ret_valid_1 = (int)(w2_ret_src_1 >= 0 && w2_ret_src_1 < world_size);
                        int w2_shr_valid = (int)(w2_pool_row_2 >= 204736);
                        unsigned long long w2_shr_base_0 = 0;
                        unsigned long long w2_shr_base_1 = 0;
                        if (w2_shr_valid != 0) {
                            w2_shr_base_0 = (unsigned long long)(w2_ret_row_0 - 204736) * 4096;
                            w2_shr_base_1 = (unsigned long long)(w2_ret_row_1 - 204736) * 4096;
                        }
                        int w2_stage_row_0 = w1_warp_m * 32 + w2_mt_2 * 16 + w1_group_id;
                        int w2_stage_row_1 = w2_stage_row_0 + 8;
                        #pragma unroll
                        for (int w2_n_tile_2 = 0; w2_n_tile_2 < 8; w2_n_tile_2++) {
                            int w2_output_n_tile = w1_warp_n * 8 + w2_n_tile_2;
                            int w2_output_col = w2_n_block_2 * 128 + w2_output_n_tile * 8 + w1_thread_id * 2;
                            int w2_acc_base = (w2_mt_2 * 8 + w2_n_tile_2) * 4;
                            int w2_local_col = w2_output_n_tile * 8 + w1_thread_id * 2;
                            int w2_sub_t = w2_local_col / 64;
                            int w2_col_in = w2_local_col - w2_sub_t * 64;
                            int w2_addr0 = w2_sub_t * 16384 + w2_stage_row_0 * 128 + (w2_col_in * 2 ^ (w2_stage_row_0 & 7) * 16);
                            int w2_addr1 = w2_sub_t * 16384 + w2_stage_row_1 * 128 + (w2_col_in * 2 ^ (w2_stage_row_1 & 7) * 16);
                            d_stage[w2_addr0 / 2] = w1_accum[w2_acc_base];
                            d_stage[w2_addr0 / 2 + 1] = w1_accum[w2_acc_base + 1];
                            d_stage[w2_addr1 / 2] = w1_accum[w2_acc_base + 2];
                            d_stage[w2_addr1 / 2 + 1] = w1_accum[w2_acc_base + 3];
                            if (w2_shr_valid != 0) {
                                {
                                    __nv_bfloat162 _pk = __floats2bfloat162_rn(w1_accum[w2_acc_base + 0], w1_accum[w2_acc_base + 1]);
                                    *reinterpret_cast<__nv_bfloat162*>(&((__nv_bfloat16*)(shared_out + (w2_shr_base_0 + (unsigned long long)w2_output_col)))[0]) = _pk;
                                }
                                {
                                    __nv_bfloat162 _pk = __floats2bfloat162_rn(w1_accum[w2_acc_base + 2 + 0], w1_accum[w2_acc_base + 2 + 1]);
                                    *reinterpret_cast<__nv_bfloat162*>(&((__nv_bfloat16*)(shared_out + (w2_shr_base_1 + (unsigned long long)w2_output_col)))[0]) = _pk;
                                }
                            }
                        }
                        int ce_block = w2_n_block_2 * 2 + w1_warp_n;
                        #pragma unroll
                        for (int ce_r = 0; ce_r < 2; ce_r++) {
                            unsigned int ce_bits[16];
                            unsigned int ce_emax = 0;
                            unsigned int ce_emin = 255;
                            #pragma unroll
                            for (int ce_nt = 0; ce_nt < 8; ce_nt++) {
                                #pragma unroll
                                for (int ce_pair = 0; ce_pair < 2; ce_pair++) {
                                    float ce_val = (float)(__nv_bfloat16)w1_accum[(w2_mt_2 * 8 + ce_nt) * 4 + ce_r * 2 + ce_pair];
                                    unsigned int ce_vbits = 0;
                                    ce_vbits = reinterpret_cast<unsigned int*>(&ce_val)[0];
                                    ce_bits[ce_nt * 2 + ce_pair] = ce_vbits >> 16;
                                    unsigned int ce_e = ce_vbits >> 23 & 255;
                                    unsigned int _max_61 = ((ce_emax) > (ce_e) ? (ce_emax) : (ce_e));
                                    ce_emax = _max_61;
                                    unsigned int _min_31 = ((ce_emin) < (ce_e) ? (ce_emin) : (ce_e));
                                    ce_emin = _min_31;
                                }
                            }
                            unsigned int _shfl_xor_24 = __shfl_xor_sync(0xFFFFFFFF, ce_emax, 1);
                            unsigned int _max_62 = ((ce_emax) > (_shfl_xor_24) ? (ce_emax) : (_shfl_xor_24));
                            ce_emax = _max_62;
                            unsigned int _shfl_xor_25 = __shfl_xor_sync(0xFFFFFFFF, ce_emax, 2);
                            unsigned int _max_63 = ((ce_emax) > (_shfl_xor_25) ? (ce_emax) : (_shfl_xor_25));
                            ce_emax = _max_63;
                            unsigned int _shfl_xor_26 = __shfl_xor_sync(0xFFFFFFFF, ce_emin, 1);
                            unsigned int _min_32 = ((ce_emin) < (_shfl_xor_26) ? (ce_emin) : (_shfl_xor_26));
                            ce_emin = _min_32;
                            unsigned int _shfl_xor_27 = __shfl_xor_sync(0xFFFFFFFF, ce_emin, 2);
                            unsigned int _min_33 = ((ce_emin) < (_shfl_xor_27) ? (ce_emin) : (_shfl_xor_27));
                            ce_emin = _min_33;
                            int ce_raw = (int)(ce_emax - ce_emin > 15);
                            int ce_valid = w2_ret_valid_0 * (1 - ce_r) + w2_ret_valid_1 * ce_r;
                            int ce_src = w2_ret_src_0 * (1 - ce_r) + w2_ret_src_1 * ce_r;
                            int ce_rowi = w2_ret_row_0 + ce_r * 8;
                            int ce_k = 0;
                            int ce_idx = 0;
                            int ce_start = 0;
                            int ce_rows = 0;
                            int ce_chunk = 0;
                            if (ce_valid != 0) {
                                ce_idx = meta_result_index[ce_rowi];
                                int _max_64 = ((source_route_counts[ce_src]) > (0) ? (source_route_counts[ce_src]) : (0));
                                int _min_34 = ((_max_64) < (49152) ? (_max_64) : (49152));
                                int ce_rc = _min_34;
                                int ce_q = 128;
                                if (ce_rc >= 6144) {
                                    ce_q = 64;
                                }
                                if (ce_rc < 768) {
                                    ce_q = 32;
                                }
                                int _max_65 = (((ce_rc + ce_q - 1) / ce_q - 1) > (0) ? ((ce_rc + ce_q - 1) / ce_q - 1) : (0));
                                int ce_full = _max_65;
                                int ce_ts = ce_full * ce_q;
                                if (ce_idx < ce_ts) {
                                    ce_chunk = ce_idx / ce_q;
                                    ce_start = ce_chunk * ce_q;
                                    ce_rows = ce_q;
                                } else {
                                    ce_chunk = ce_full + (ce_idx - ce_ts) / 64;
                                    ce_start = ce_ts + (ce_chunk - ce_full) * 64;
                                    int _min_35 = ((64) < (ce_rc - ce_start) ? (64) : (ce_rc - ce_start));
                                    ce_rows = _min_35;
                                }
                                if (ce_raw != 0 && w1_thread_id == 0) {
                                    int _atomic_old_22 = atomicAdd(&result_ovf_cursor[ce_src * 769 + ce_chunk], 1);
                                    ce_k = _atomic_old_22;
                                }
                            }
                            int _shfl_27 = __shfl_sync(0xFFFFFFFF, ce_k, w1_group_id * 4);
                            ce_k = _shfl_27;
                            if (ce_valid != 0 && ce_raw != 0 && ce_k >= ce_rows * 16) {
                                if (w1_thread_id == 0) {
                                    atomicMax(&protocol_error[0], 1);
                                }
                                ce_k = ce_k & 2047;
                            }
                            if (ce_valid != 0) {
                                unsigned long long ce_chunk_word = (unsigned long long)(ce_src * 2 + slot) * 102285312 + (unsigned long long)ce_start * 2080;
                                unsigned long long ce_row_word = ce_chunk_word + (unsigned long long)(ce_idx - ce_start) * 1568;
                                if (ce_raw != 0) {
                                    float ce_raw_vals[16];
                                    #pragma unroll
                                    for (int ce_nt2 = 0; ce_nt2 < 8; ce_nt2++) {
                                        #pragma unroll
                                        for (int ce_pair2 = 0; ce_pair2 < 2; ce_pair2++) {
                                            ce_raw_vals[ce_nt2 * 2 + ce_pair2] = w1_accum[(w2_mt_2 * 8 + ce_nt2) * 4 + ce_r * 2 + ce_pair2];
                                        }
                                    }
                                    unsigned long long ce_ovf_elem = (ce_chunk_word + (unsigned long long)ce_rows * 1568) * 2 + (unsigned long long)ce_k * 64 + (unsigned long long)w1_thread_id * 16;
                                    {
                                        __nv_bfloat162 _pk[8];
                                        _pk[0] = __floats2bfloat162_rn(ce_raw_vals[0 + 0], ce_raw_vals[0 + 1]);
                                        _pk[1] = __floats2bfloat162_rn(ce_raw_vals[0 + 2], ce_raw_vals[0 + 3]);
                                        _pk[2] = __floats2bfloat162_rn(ce_raw_vals[0 + 4], ce_raw_vals[0 + 5]);
                                        _pk[3] = __floats2bfloat162_rn(ce_raw_vals[0 + 6], ce_raw_vals[0 + 7]);
                                        _pk[4] = __floats2bfloat162_rn(ce_raw_vals[0 + 8], ce_raw_vals[0 + 9]);
                                        _pk[5] = __floats2bfloat162_rn(ce_raw_vals[0 + 10], ce_raw_vals[0 + 11]);
                                        _pk[6] = __floats2bfloat162_rn(ce_raw_vals[0 + 12], ce_raw_vals[0 + 13]);
                                        _pk[7] = __floats2bfloat162_rn(ce_raw_vals[0 + 14], ce_raw_vals[0 + 15]);
                                        *reinterpret_cast<uint4*>(&((__nv_bfloat16*)(reinterpret_cast<__nv_bfloat16*>(result_out) + ce_ovf_elem))[0]) = *reinterpret_cast<uint4*>(&_pk[0]);
                                        *reinterpret_cast<uint4*>(&((__nv_bfloat16*)(reinterpret_cast<__nv_bfloat16*>(result_out) + ce_ovf_elem))[8]) = *reinterpret_cast<uint4*>(&_pk[4]);
                                    }
                                    if (w1_thread_id == 0) {
                                        unsigned long long ce_hdr_byte_r = ce_row_word * 4 + (unsigned long long)ce_block * 2;
                                        unsigned int ce_hdr_lo_r = (unsigned int)ce_k & 255;
                                        unsigned int ce_hdr_hi_r = 128 | (unsigned int)ce_k >> 8 & 127;
                                        *(reinterpret_cast<unsigned char*>(reinterpret_cast<uint8_t*>(result_out) + ce_hdr_byte_r) + (0)) = (unsigned char)(ce_hdr_lo_r);
                                        *(reinterpret_cast<unsigned char*>(reinterpret_cast<uint8_t*>(result_out) + (ce_hdr_byte_r + 1)) + (0)) = (unsigned char)(ce_hdr_hi_r);
                                    }
                                } else {
                                    unsigned int ce_c[16];
                                    #pragma unroll
                                    for (int ce_kk = 0; ce_kk < 16; ce_kk++) {
                                        ce_c[ce_kk] = ce_bits[ce_kk] >> 15 << 11 | ce_emax - (ce_bits[ce_kk] >> 7 & 255) << 7 | ce_bits[ce_kk] & 127;
                                    }
                                    int ce_w[6];
                                    unsigned int ce_t0 = ce_c[0] | ce_c[1] << 12 | (ce_c[2] & 255) << 24;
                                    unsigned int ce_t1 = ce_c[2] >> 8 | ce_c[3] << 4 | ce_c[4] << 16 | (ce_c[5] & 15) << 28;
                                    unsigned int ce_t2 = ce_c[5] >> 4 | ce_c[6] << 8 | ce_c[7] << 20;
                                    unsigned int ce_t3 = ce_c[8] | ce_c[9] << 12 | (ce_c[10] & 255) << 24;
                                    unsigned int ce_t4 = ce_c[10] >> 8 | ce_c[11] << 4 | ce_c[12] << 16 | (ce_c[13] & 15) << 28;
                                    unsigned int ce_t5 = ce_c[13] >> 4 | ce_c[14] << 8 | ce_c[15] << 20;
                                    ce_w[0] = (int)ce_t0;
                                    ce_w[1] = (int)ce_t1;
                                    ce_w[2] = (int)ce_t2;
                                    ce_w[3] = (int)ce_t3;
                                    ce_w[4] = (int)ce_t4;
                                    ce_w[5] = (int)ce_t5;
                                    unsigned long long ce_body_word = ce_row_word + 32 + (unsigned long long)ce_block * 24;
                                    {
                                        int4 _iv4 = make_int4(ce_w[0 + 0], ce_w[0 + 1], ce_w[0 + 2], ce_w[0 + 3]);
                                        *reinterpret_cast<int4*>(reinterpret_cast<int*>(result_out) + (ce_body_word + (unsigned long long)w1_thread_id * 4) + 0) = _iv4;
                                    }
                                    {
                                        int2 _iv2 = make_int2(ce_w[4 + 0], ce_w[4 + 1]);
                                        *reinterpret_cast<int2*>(reinterpret_cast<int*>(result_out) + (ce_body_word + 16 + (unsigned long long)w1_thread_id * 2) + 0) = _iv2;
                                    }
                                    if (w1_thread_id == 0) {
                                        unsigned long long ce_hdr_byte_e = ce_row_word * 4 + (unsigned long long)ce_block * 2;
                                        unsigned int ce_hdr_lo_e = ce_emax;
                                        unsigned int ce_hdr_hi_e = 0;
                                        *(reinterpret_cast<unsigned char*>(reinterpret_cast<uint8_t*>(result_out) + ce_hdr_byte_e) + (0)) = (unsigned char)(ce_hdr_lo_e);
                                        *(reinterpret_cast<unsigned char*>(reinterpret_cast<uint8_t*>(result_out) + (ce_hdr_byte_e + 1)) + (0)) = (unsigned char)(ce_hdr_hi_e);
                                    }
                                }
                            }
                        }
                    }
                    asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                    asm volatile("barrier.sync 15, 256;" ::: "memory");
                    if (warp == 0) {
                        if (elect_sync()) {
                            tma_store_2d(W2_D, w2_n_block_2 * 128, w2_pool_row_2, d_stage_addr);
                            tma_store_2d(W2_D, w2_n_block_2 * 128 + 64, w2_pool_row_2, d_stage_addr + 16384);
                            asm volatile("cp.async.bulk.commit_group;");
                        }
                    }
                    __threadfence();
                    __syncwarp();
                    unsigned long long gtimer_1_2;
                    asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(gtimer_1_2) :: "memory");
                    unsigned long long c28_w2_end_ts = gtimer_1_2;
                    if (elect_sync()) {
                        if (w2_pool_row_2 >= 204736) {
                            atomicMax(&phase_timestamps[24], c28_w2_end_ts);
                        }
                    }
                    if (elect_sync()) {
                        int _atomic_old_23 = atomicAdd(&w2_warp_done[w2_tile_2], 1);
                        int w2_previous = _atomic_old_23;
                        if (w2_previous == 7) {
                            atomicAdd(&w2_tiles_completed[0], 1);
                            {
                                unsigned int* _gc_p = reinterpret_cast<unsigned int*>(w2_task_counter) + (w2_task_2);
                                unsigned int _gc_old;
                                asm volatile("atom.release.gpu.global.add.u32 %0, [%1], 1;" : "=r"(_gc_old) : "l"(_gc_p) : "memory");
                            }
                        } else if (w2_previous >= 8) {
                            atomicMax(&protocol_error[0], 1);
                        }
                    }
                }
            }
        }
    }
    unsigned int _phase_epi_full_0 = 0;
    if (bid > 0 && warp >= 9) {
        int epi_mb_base = (bid - 1) * 32;
        asm volatile("barrier.sync 14, 96;" ::: "memory");
        if (warp == 9) {
            if (elect_sync()) {
                int _atomic_old_24 = atomicAdd(&c56_tile_mailbox[epi_mb_base + 9], 1);
                int epi_old = _atomic_old_24;
                atomicMax(&c56_tile_mailbox[epi_mb_base + 10], 1 + (epi_old & 2));
            }
        }
        asm volatile("barrier.sync 14, 96;" ::: "memory");
        int epi_join = 0;
        if (elect_sync()) {
            int _atomic_old_25 = atomicAdd(&c56_tile_mailbox[epi_mb_base + 10], 0);
            epi_join = _atomic_old_25;
        }
        int _shfl_28 = __shfl_sync(0xFFFFFFFF, epi_join, 0);
        epi_join = _shfl_28;
        if ((epi_join & 2) == 0) {
            int epi_vw_base = warp - 9;
            int epi_gid = lane / 4;
            int epi_tid = lane % 4;
            unsigned int epi_word[1];
            unsigned int epi_lo_bits = 0;
            unsigned int epi_hi_bits = 0;
            float epi_a[16];
            float epi_b[16];
            #pragma unroll 1
            for (int epi_iter = 0; epi_iter < 106753; epi_iter++) {
                mbarrier_wait(epi_full_addr, _phase_epi_full_0);
                _phase_epi_full_0 ^= 1;
                int epi_val = 0;
                if (elect_sync()) {
                    int _atomic_old_26 = atomicAdd(&c56_tile_mailbox[epi_mb_base + 8], 0);
                    epi_val = _atomic_old_26;
                }
                int _shfl_29 = __shfl_sync(0xFFFFFFFF, epi_val, 0);
                epi_val = _shfl_29;
                if (epi_val == 2147483647) {
                    break;
                }
                int epi_u = epi_val - 1;
                int epi_round = epi_u / 64;
                int epi_pos = epi_u - epi_round * 64;
                if (epi_pos < 32) {
                    int e1_tile = epi_round * 32 + epi_pos;
                    int e1_task = epi_round;
                    int e1_nb = epi_pos;
                    int e1_pool_row = 0;
                    if (elect_sync()) {
                        int _atomic_old_27 = atomicAdd(&task_pool_row[e1_task], 0);
                        e1_pool_row = _atomic_old_27;
                    }
                    int _shfl_30 = __shfl_sync(0xFFFFFFFF, e1_pool_row, 0);
                    e1_pool_row = _shfl_30;
                    #pragma unroll
                    for (int e1_k = 0; e1_k < 3; e1_k++) {
                        int e1_vw = epi_vw_base + e1_k * 3;
                        if (e1_vw < 8) {
                            int e1_vm = e1_vw / 2;
                            int e1_vn = e1_vw % 2;
                            int e1_req_group = e1_nb * 2 + e1_vn;
                            #pragma unroll
                            for (int e1_fmt = 0; e1_fmt < 2; e1_fmt++) {
                                int e1_srow_0 = e1_vm * 32 + e1_fmt * 16 + epi_gid;
                                int e1_srow_1 = e1_srow_0 + 8;
                                int e1_row_0 = e1_pool_row + e1_srow_0;
                                int e1_row_1 = e1_pool_row + e1_srow_1;
                                float e1_rw_0 = routing_weight_pool[e1_row_0];
                                float e1_rw_1 = routing_weight_pool[e1_row_1];
                                unsigned long long e1_int_base_0 = (unsigned long long)e1_row_0 * 2048;
                                unsigned long long e1_int_base_1 = (unsigned long long)e1_row_1 * 2048;
                                float e1_amax_0 = 0.0f;
                                float e1_amax_1 = 0.0f;
                                #pragma unroll
                                for (int e1_rq = 0; e1_rq < 4; e1_rq++) {
                                    int e1_gcol = e1_rq * 2 * 8 + epi_tid * 2;
                                    int e1_ucol = (e1_rq * 2 + 1) * 8 + epi_tid * 2;
                                    int e1_g_addr_0 = e1_vn * 16384 + e1_srow_0 * 128 + (e1_gcol * 2 ^ (e1_srow_0 & 7) * 16);
                                    int e1_u_addr_0 = e1_vn * 16384 + e1_srow_0 * 128 + (e1_ucol * 2 ^ (e1_srow_0 & 7) * 16);
                                    int e1_g_addr_1 = e1_vn * 16384 + e1_srow_1 * 128 + (e1_gcol * 2 ^ (e1_srow_1 & 7) * 16);
                                    int e1_u_addr_1 = e1_vn * 16384 + e1_srow_1 * 128 + (e1_ucol * 2 ^ (e1_srow_1 & 7) * 16);
                                    asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&epi_word[0])) : "r"(d_stage_addr + (unsigned int)e1_g_addr_0));
                                    unsigned int e1_gw_0 = epi_word[0];
                                    asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&epi_word[0])) : "r"(d_stage_addr + (unsigned int)e1_u_addr_0));
                                    unsigned int e1_uw_0 = epi_word[0];
                                    asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&epi_word[0])) : "r"(d_stage_addr + (unsigned int)e1_g_addr_1));
                                    unsigned int e1_gw_1 = epi_word[0];
                                    asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&epi_word[0])) : "r"(d_stage_addr + (unsigned int)e1_u_addr_1));
                                    unsigned int e1_uw_1 = epi_word[0];
                                    #pragma unroll
                                    for (int e1_rc = 0; e1_rc < 2; e1_rc++) {
                                        epi_lo_bits = e1_gw_0 << (unsigned int)(16 * (1 - e1_rc)) & 4294901760;
                                        float e1_gate_0 = 0.0f;
                                        e1_gate_0 = reinterpret_cast<float*>(&epi_lo_bits)[0];
                                        epi_hi_bits = e1_uw_0 << (unsigned int)(16 * (1 - e1_rc)) & 4294901760;
                                        float e1_up_0 = 0.0f;
                                        e1_up_0 = reinterpret_cast<float*>(&epi_hi_bits)[0];
                                        float _min_36 = fminf(e1_gate_0, 10.0f);
                                        e1_gate_0 = _min_36;
                                        float _max_66 = max_noftz(e1_up_0, -10.0f);
                                        float _min_37 = fminf(_max_66, 10.0f);
                                        e1_up_0 = _min_37;
                                        float _exp2_2 = approx_exp2((-e1_gate_0) * 1.4426950408889634f);
                                        float e1_sig_0 = 1.0f / (1.0f + _exp2_2);
                                        float e1_val_0 = e1_gate_0 * e1_sig_0 * e1_up_0 * e1_rw_0;
                                        epi_a[e1_rq * 2 + e1_rc] = e1_val_0;
                                        float _max_67 = max_noftz(e1_val_0, -e1_val_0);
                                        float _max_68 = max_noftz(e1_amax_0, _max_67);
                                        e1_amax_0 = _max_68;
                                        epi_lo_bits = e1_gw_1 << (unsigned int)(16 * (1 - e1_rc)) & 4294901760;
                                        float e1_gate_1 = 0.0f;
                                        e1_gate_1 = reinterpret_cast<float*>(&epi_lo_bits)[0];
                                        epi_hi_bits = e1_uw_1 << (unsigned int)(16 * (1 - e1_rc)) & 4294901760;
                                        float e1_up_1 = 0.0f;
                                        e1_up_1 = reinterpret_cast<float*>(&epi_hi_bits)[0];
                                        float _min_38 = fminf(e1_gate_1, 10.0f);
                                        e1_gate_1 = _min_38;
                                        float _max_69 = max_noftz(e1_up_1, -10.0f);
                                        float _min_39 = fminf(_max_69, 10.0f);
                                        e1_up_1 = _min_39;
                                        float _exp2_3 = approx_exp2((-e1_gate_1) * 1.4426950408889634f);
                                        float e1_sig_1 = 1.0f / (1.0f + _exp2_3);
                                        float e1_val_1 = e1_gate_1 * e1_sig_1 * e1_up_1 * e1_rw_1;
                                        epi_b[e1_rq * 2 + e1_rc] = e1_val_1;
                                        float _max_70 = max_noftz(e1_val_1, -e1_val_1);
                                        float _max_71 = max_noftz(e1_amax_1, _max_70);
                                        e1_amax_1 = _max_71;
                                    }
                                }
                                float _shfl_xor_28 = __shfl_xor_sync(0xFFFFFFFF, e1_amax_0, 2);
                                float _max_72 = max_noftz(e1_amax_0, _shfl_xor_28);
                                e1_amax_0 = _max_72;
                                float _shfl_xor_29 = __shfl_xor_sync(0xFFFFFFFF, e1_amax_0, 1);
                                float _max_73 = max_noftz(e1_amax_0, _shfl_xor_29);
                                e1_amax_0 = _max_73;
                                float _shfl_xor_30 = __shfl_xor_sync(0xFFFFFFFF, e1_amax_1, 2);
                                float _max_74 = max_noftz(e1_amax_1, _shfl_xor_30);
                                e1_amax_1 = _max_74;
                                float _shfl_xor_31 = __shfl_xor_sync(0xFFFFFFFF, e1_amax_1, 1);
                                float _max_75 = max_noftz(e1_amax_1, _shfl_xor_31);
                                e1_amax_1 = _max_75;
                                float e1_sf_0 = e1_amax_0 * 0.002232142857142857f;
                                unsigned int e1_sf_0_bits = 0;
                                e1_sf_0_bits = reinterpret_cast<unsigned int*>(&e1_sf_0)[0];
                                unsigned int e1_sf_0_exp = (e1_sf_0_bits >> 23 & 255) + ((e1_sf_0_bits & 8388607) + 8388607 >> 23);
                                unsigned int _min_40 = ((e1_sf_0_exp) < (254) ? (e1_sf_0_exp) : (254));
                                e1_sf_0_exp = _min_40;
                                unsigned int e1_sf_0_inv_bits = 254 - e1_sf_0_exp << 23;
                                float e1_sf_0_inv = 0.0f;
                                e1_sf_0_inv = reinterpret_cast<float*>(&e1_sf_0_inv_bits)[0];
                                float e1_sf_1 = e1_amax_1 * 0.002232142857142857f;
                                unsigned int e1_sf_1_bits = 0;
                                e1_sf_1_bits = reinterpret_cast<unsigned int*>(&e1_sf_1)[0];
                                unsigned int e1_sf_1_exp = (e1_sf_1_bits >> 23 & 255) + ((e1_sf_1_bits & 8388607) + 8388607 >> 23);
                                unsigned int _min_41 = ((e1_sf_1_exp) < (254) ? (e1_sf_1_exp) : (254));
                                e1_sf_1_exp = _min_41;
                                unsigned int e1_sf_1_inv_bits = 254 - e1_sf_1_exp << 23;
                                float e1_sf_1_inv = 0.0f;
                                e1_sf_1_inv = reinterpret_cast<float*>(&e1_sf_1_inv_bits)[0];
                                if (epi_tid == 0) {
                                    int e1_sf_index_0 = ((e1_req_group >> 2) * 212928 + e1_row_0) * 4 + (e1_req_group & 3);
                                    *(reinterpret_cast<unsigned char*>(intermediate_sfa_u8 + e1_sf_index_0) + (0)) = (unsigned char)(e1_sf_0_exp);
                                    int e1_sf_index_1 = ((e1_req_group >> 2) * 212928 + e1_row_1) * 4 + (e1_req_group & 3);
                                    *(reinterpret_cast<unsigned char*>(intermediate_sfa_u8 + e1_sf_index_1) + (0)) = (unsigned char)(e1_sf_1_exp);
                                }
                                #pragma unroll
                                for (int e1_sq = 0; e1_sq < 4; e1_sq++) {
                                    int e1_log_n = e1_req_group * 32 + e1_sq * 8 + epi_tid * 2;
                                    {
                                        unsigned short _fp8_pair;
                                        asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_fp8_pair) : "f"(epi_a[e1_sq * 2 + 1] * e1_sf_0_inv), "f"(epi_a[e1_sq * 2 + 0] * e1_sf_0_inv));
                                        *reinterpret_cast<unsigned short*>(reinterpret_cast<unsigned char*>(intermediate_fp8 + (e1_int_base_0 + (unsigned long long)e1_log_n)) + (0)) = _fp8_pair;
                                    }
                                    {
                                        unsigned short _fp8_pair;
                                        asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_fp8_pair) : "f"(epi_b[e1_sq * 2 + 1] * e1_sf_1_inv), "f"(epi_b[e1_sq * 2 + 0] * e1_sf_1_inv));
                                        *reinterpret_cast<unsigned short*>(reinterpret_cast<unsigned char*>(intermediate_fp8 + (e1_int_base_1 + (unsigned long long)e1_log_n)) + (0)) = _fp8_pair;
                                    }
                                }
                            }
                        }
                    }
                    __threadfence();
                    __syncwarp();
                    unsigned long long gtimer_0_13;
                    asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(gtimer_0_13) :: "memory");
                    unsigned long long e1_end_ts = gtimer_0_13;
                    int e1_completion_count = (7 - epi_vw_base) / 3 + 1;
                    if (elect_sync()) {
                        atomicAdd(&requant_groups_done[0], e1_completion_count * 16 * 2);
                        if (e1_pool_row >= 204736) {
                            atomicMax(&phase_timestamps[22], e1_end_ts);
                        }
                        int _atomic_old_28 = atomicAdd(&w1_warp_done[e1_tile], e1_completion_count);
                        int e1_previous = _atomic_old_28;
                        if (e1_previous + e1_completion_count == 8) {
                            atomicAdd(&w1_tiles_completed[0], 1);
                            {
                                unsigned int* _gc_p = reinterpret_cast<unsigned int*>(w1_task_counter) + (e1_task);
                                unsigned int _gc_old;
                                asm volatile("atom.release.gpu.global.add.u32 %0, [%1], 1;" : "=r"(_gc_old) : "l"(_gc_p) : "memory");
                            }
                        } else if (e1_previous + e1_completion_count > 8) {
                            atomicMax(&protocol_error[0], 1);
                        }
                    }
                } else {
                    int e2_tile = (epi_round - 4) * 32 + (epi_pos - 32);
                    int e2_task = epi_round - 4;
                    int e2_nb = epi_pos - 32;
                    int e2_pool_row = 0;
                    if (elect_sync()) {
                        int _atomic_old_29 = atomicAdd(&task_pool_row[e2_task], 0);
                        e2_pool_row = _atomic_old_29;
                    }
                    int _shfl_31 = __shfl_sync(0xFFFFFFFF, e2_pool_row, 0);
                    e2_pool_row = _shfl_31;
                    int e2_shr_valid = (int)(e2_pool_row >= 204736);
                    #pragma unroll
                    for (int e2_k = 0; e2_k < 3; e2_k++) {
                        int e2_vw = epi_vw_base + e2_k * 3;
                        if (e2_vw < 8) {
                            int e2_vm = e2_vw / 2;
                            int e2_vn = e2_vw % 2;
                            int e2_block = e2_nb * 2 + e2_vn;
                            #pragma unroll
                            for (int e2_mt = 0; e2_mt < 2; e2_mt++) {
                                int e2_srow_0 = e2_vm * 32 + e2_mt * 16 + epi_gid;
                                int e2_srow_1 = e2_srow_0 + 8;
                                int e2_row_0 = e2_pool_row + e2_srow_0;
                                int e2_row_1 = e2_pool_row + e2_srow_1;
                                int e2_src_0 = meta_source_rank[e2_row_0];
                                int e2_src_1 = meta_source_rank[e2_row_1];
                                int e2_valid_0 = (int)(e2_src_0 >= 0 && e2_src_0 < world_size);
                                int e2_valid_1 = (int)(e2_src_1 >= 0 && e2_src_1 < world_size);
                                unsigned long long e2_shr_base_0 = 0;
                                unsigned long long e2_shr_base_1 = 0;
                                if (e2_shr_valid != 0) {
                                    e2_shr_base_0 = (unsigned long long)(e2_row_0 - 204736) * 4096;
                                    e2_shr_base_1 = (unsigned long long)(e2_row_1 - 204736) * 4096;
                                }
                                #pragma unroll
                                for (int e2_nt = 0; e2_nt < 8; e2_nt++) {
                                    int e2_col = e2_nt * 8 + epi_tid * 2;
                                    int e2_addr_0 = e2_vn * 16384 + e2_srow_0 * 128 + (e2_col * 2 ^ (e2_srow_0 & 7) * 16);
                                    int e2_addr_1 = e2_vn * 16384 + e2_srow_1 * 128 + (e2_col * 2 ^ (e2_srow_1 & 7) * 16);
                                    asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&epi_word[0])) : "r"(d_stage_addr + (unsigned int)e2_addr_0));
                                    epi_lo_bits = epi_word[0] << 16;
                                    epi_hi_bits = epi_word[0] & 4294901760;
                                    float e2_v00 = 0.0f;
                                    e2_v00 = reinterpret_cast<float*>(&epi_lo_bits)[0];
                                    float e2_v01 = 0.0f;
                                    e2_v01 = reinterpret_cast<float*>(&epi_hi_bits)[0];
                                    epi_a[e2_nt * 2] = e2_v00;
                                    epi_a[e2_nt * 2 + 1] = e2_v01;
                                    asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&epi_word[0])) : "r"(d_stage_addr + (unsigned int)e2_addr_1));
                                    epi_lo_bits = epi_word[0] << 16;
                                    epi_hi_bits = epi_word[0] & 4294901760;
                                    float e2_v10 = 0.0f;
                                    e2_v10 = reinterpret_cast<float*>(&epi_lo_bits)[0];
                                    float e2_v11 = 0.0f;
                                    e2_v11 = reinterpret_cast<float*>(&epi_hi_bits)[0];
                                    epi_b[e2_nt * 2] = e2_v10;
                                    epi_b[e2_nt * 2 + 1] = e2_v11;
                                    if (e2_shr_valid != 0) {
                                        int e2_output_col = e2_nb * 128 + (e2_vn * 8 + e2_nt) * 8 + epi_tid * 2;
                                        {
                                            __nv_bfloat162 _pk = __floats2bfloat162_rn(epi_a[e2_nt * 2 + 0], epi_a[e2_nt * 2 + 1]);
                                            *reinterpret_cast<__nv_bfloat162*>(&((__nv_bfloat16*)(shared_out + (e2_shr_base_0 + (unsigned long long)e2_output_col)))[0]) = _pk;
                                        }
                                        {
                                            __nv_bfloat162 _pk = __floats2bfloat162_rn(epi_b[e2_nt * 2 + 0], epi_b[e2_nt * 2 + 1]);
                                            *reinterpret_cast<__nv_bfloat162*>(&((__nv_bfloat16*)(shared_out + (e2_shr_base_1 + (unsigned long long)e2_output_col)))[0]) = _pk;
                                        }
                                    }
                                }
                                #pragma unroll
                                for (int ce_r_1 = 0; ce_r_1 < 2; ce_r_1++) {
                                    unsigned int ce_bits_1[16];
                                    unsigned int ce_emax_1 = 0;
                                    unsigned int ce_emin_1 = 255;
                                    #pragma unroll
                                    for (int ce_kk0 = 0; ce_kk0 < 16; ce_kk0++) {
                                        float ce_vf = epi_a[ce_kk0];
                                        if (ce_r_1 == 1) {
                                            ce_vf = epi_b[ce_kk0];
                                        }
                                        unsigned int ce_vbits_1 = 0;
                                        ce_vbits_1 = reinterpret_cast<unsigned int*>(&ce_vf)[0];
                                        ce_bits_1[ce_kk0] = ce_vbits_1 >> 16;
                                        unsigned int ce_e_1 = ce_vbits_1 >> 23 & 255;
                                        unsigned int _max_76 = ((ce_emax_1) > (ce_e_1) ? (ce_emax_1) : (ce_e_1));
                                        ce_emax_1 = _max_76;
                                        unsigned int _min_42 = ((ce_emin_1) < (ce_e_1) ? (ce_emin_1) : (ce_e_1));
                                        ce_emin_1 = _min_42;
                                    }
                                    unsigned int _shfl_xor_32 = __shfl_xor_sync(0xFFFFFFFF, ce_emax_1, 1);
                                    unsigned int _max_77 = ((ce_emax_1) > (_shfl_xor_32) ? (ce_emax_1) : (_shfl_xor_32));
                                    ce_emax_1 = _max_77;
                                    unsigned int _shfl_xor_33 = __shfl_xor_sync(0xFFFFFFFF, ce_emax_1, 2);
                                    unsigned int _max_78 = ((ce_emax_1) > (_shfl_xor_33) ? (ce_emax_1) : (_shfl_xor_33));
                                    ce_emax_1 = _max_78;
                                    unsigned int _shfl_xor_34 = __shfl_xor_sync(0xFFFFFFFF, ce_emin_1, 1);
                                    unsigned int _min_43 = ((ce_emin_1) < (_shfl_xor_34) ? (ce_emin_1) : (_shfl_xor_34));
                                    ce_emin_1 = _min_43;
                                    unsigned int _shfl_xor_35 = __shfl_xor_sync(0xFFFFFFFF, ce_emin_1, 2);
                                    unsigned int _min_44 = ((ce_emin_1) < (_shfl_xor_35) ? (ce_emin_1) : (_shfl_xor_35));
                                    ce_emin_1 = _min_44;
                                    int ce_raw_1 = (int)(ce_emax_1 - ce_emin_1 > 15);
                                    int ce_valid_1 = e2_valid_0 * (1 - ce_r_1) + e2_valid_1 * ce_r_1;
                                    int ce_src_1 = e2_src_0 * (1 - ce_r_1) + e2_src_1 * ce_r_1;
                                    int ce_rowi_1 = e2_row_0 + ce_r_1 * 8;
                                    int ce_k_1 = 0;
                                    int ce_idx_1 = 0;
                                    int ce_start_1 = 0;
                                    int ce_rows_1 = 0;
                                    int ce_chunk_1 = 0;
                                    if (ce_valid_1 != 0) {
                                        ce_idx_1 = meta_result_index[ce_rowi_1];
                                        int _max_79 = ((source_route_counts[ce_src_1]) > (0) ? (source_route_counts[ce_src_1]) : (0));
                                        int _min_45 = ((_max_79) < (49152) ? (_max_79) : (49152));
                                        int ce_rc_1 = _min_45;
                                        int ce_q_1 = 128;
                                        if (ce_rc_1 >= 6144) {
                                            ce_q_1 = 64;
                                        }
                                        if (ce_rc_1 < 768) {
                                            ce_q_1 = 32;
                                        }
                                        int _max_80 = (((ce_rc_1 + ce_q_1 - 1) / ce_q_1 - 1) > (0) ? ((ce_rc_1 + ce_q_1 - 1) / ce_q_1 - 1) : (0));
                                        int ce_full_1 = _max_80;
                                        int ce_ts_1 = ce_full_1 * ce_q_1;
                                        if (ce_idx_1 < ce_ts_1) {
                                            ce_chunk_1 = ce_idx_1 / ce_q_1;
                                            ce_start_1 = ce_chunk_1 * ce_q_1;
                                            ce_rows_1 = ce_q_1;
                                        } else {
                                            ce_chunk_1 = ce_full_1 + (ce_idx_1 - ce_ts_1) / 64;
                                            ce_start_1 = ce_ts_1 + (ce_chunk_1 - ce_full_1) * 64;
                                            int _min_46 = ((64) < (ce_rc_1 - ce_start_1) ? (64) : (ce_rc_1 - ce_start_1));
                                            ce_rows_1 = _min_46;
                                        }
                                        if (ce_raw_1 != 0 && epi_tid == 0) {
                                            int _atomic_old_30 = atomicAdd(&result_ovf_cursor[ce_src_1 * 769 + ce_chunk_1], 1);
                                            ce_k_1 = _atomic_old_30;
                                        }
                                    }
                                    int _shfl_32 = __shfl_sync(0xFFFFFFFF, ce_k_1, epi_gid * 4);
                                    ce_k_1 = _shfl_32;
                                    if (ce_valid_1 != 0 && ce_raw_1 != 0 && ce_k_1 >= ce_rows_1 * 16) {
                                        if (epi_tid == 0) {
                                            atomicMax(&protocol_error[0], 1);
                                        }
                                        ce_k_1 = ce_k_1 & 2047;
                                    }
                                    if (ce_valid_1 != 0) {
                                        unsigned long long ce_chunk_word_1 = (unsigned long long)(ce_src_1 * 2 + slot) * 102285312 + (unsigned long long)ce_start_1 * 2080;
                                        unsigned long long ce_row_word_1 = ce_chunk_word_1 + (unsigned long long)(ce_idx_1 - ce_start_1) * 1568;
                                        if (ce_raw_1 != 0) {
                                            float ce_raw_vals_1[16];
                                            #pragma unroll
                                            for (int ce_kk1 = 0; ce_kk1 < 16; ce_kk1++) {
                                                ce_raw_vals_1[ce_kk1] = epi_a[ce_kk1];
                                                if (ce_r_1 == 1) {
                                                    ce_raw_vals_1[ce_kk1] = epi_b[ce_kk1];
                                                }
                                            }
                                            unsigned long long ce_ovf_elem_1 = (ce_chunk_word_1 + (unsigned long long)ce_rows_1 * 1568) * 2 + (unsigned long long)ce_k_1 * 64 + (unsigned long long)epi_tid * 16;
                                            {
                                                __nv_bfloat162 _pk[8];
                                                _pk[0] = __floats2bfloat162_rn(ce_raw_vals_1[0 + 0], ce_raw_vals_1[0 + 1]);
                                                _pk[1] = __floats2bfloat162_rn(ce_raw_vals_1[0 + 2], ce_raw_vals_1[0 + 3]);
                                                _pk[2] = __floats2bfloat162_rn(ce_raw_vals_1[0 + 4], ce_raw_vals_1[0 + 5]);
                                                _pk[3] = __floats2bfloat162_rn(ce_raw_vals_1[0 + 6], ce_raw_vals_1[0 + 7]);
                                                _pk[4] = __floats2bfloat162_rn(ce_raw_vals_1[0 + 8], ce_raw_vals_1[0 + 9]);
                                                _pk[5] = __floats2bfloat162_rn(ce_raw_vals_1[0 + 10], ce_raw_vals_1[0 + 11]);
                                                _pk[6] = __floats2bfloat162_rn(ce_raw_vals_1[0 + 12], ce_raw_vals_1[0 + 13]);
                                                _pk[7] = __floats2bfloat162_rn(ce_raw_vals_1[0 + 14], ce_raw_vals_1[0 + 15]);
                                                *reinterpret_cast<uint4*>(&((__nv_bfloat16*)(reinterpret_cast<__nv_bfloat16*>(result_out) + ce_ovf_elem_1))[0]) = *reinterpret_cast<uint4*>(&_pk[0]);
                                                *reinterpret_cast<uint4*>(&((__nv_bfloat16*)(reinterpret_cast<__nv_bfloat16*>(result_out) + ce_ovf_elem_1))[8]) = *reinterpret_cast<uint4*>(&_pk[4]);
                                            }
                                            if (epi_tid == 0) {
                                                unsigned long long ce_hdr_byte_r_1 = ce_row_word_1 * 4 + (unsigned long long)e2_block * 2;
                                                unsigned int ce_hdr_lo_r_1 = (unsigned int)ce_k_1 & 255;
                                                unsigned int ce_hdr_hi_r_1 = 128 | (unsigned int)ce_k_1 >> 8 & 127;
                                                *(reinterpret_cast<unsigned char*>(reinterpret_cast<uint8_t*>(result_out) + ce_hdr_byte_r_1) + (0)) = (unsigned char)(ce_hdr_lo_r_1);
                                                *(reinterpret_cast<unsigned char*>(reinterpret_cast<uint8_t*>(result_out) + (ce_hdr_byte_r_1 + 1)) + (0)) = (unsigned char)(ce_hdr_hi_r_1);
                                            }
                                        } else {
                                            unsigned int ce_c_1[16];
                                            #pragma unroll
                                            for (int ce_kk_1 = 0; ce_kk_1 < 16; ce_kk_1++) {
                                                ce_c_1[ce_kk_1] = ce_bits_1[ce_kk_1] >> 15 << 11 | ce_emax_1 - (ce_bits_1[ce_kk_1] >> 7 & 255) << 7 | ce_bits_1[ce_kk_1] & 127;
                                            }
                                            int ce_w_1[6];
                                            unsigned int ce_t0_1 = ce_c_1[0] | ce_c_1[1] << 12 | (ce_c_1[2] & 255) << 24;
                                            unsigned int ce_t1_1 = ce_c_1[2] >> 8 | ce_c_1[3] << 4 | ce_c_1[4] << 16 | (ce_c_1[5] & 15) << 28;
                                            unsigned int ce_t2_1 = ce_c_1[5] >> 4 | ce_c_1[6] << 8 | ce_c_1[7] << 20;
                                            unsigned int ce_t3_1 = ce_c_1[8] | ce_c_1[9] << 12 | (ce_c_1[10] & 255) << 24;
                                            unsigned int ce_t4_1 = ce_c_1[10] >> 8 | ce_c_1[11] << 4 | ce_c_1[12] << 16 | (ce_c_1[13] & 15) << 28;
                                            unsigned int ce_t5_1 = ce_c_1[13] >> 4 | ce_c_1[14] << 8 | ce_c_1[15] << 20;
                                            ce_w_1[0] = (int)ce_t0_1;
                                            ce_w_1[1] = (int)ce_t1_1;
                                            ce_w_1[2] = (int)ce_t2_1;
                                            ce_w_1[3] = (int)ce_t3_1;
                                            ce_w_1[4] = (int)ce_t4_1;
                                            ce_w_1[5] = (int)ce_t5_1;
                                            unsigned long long ce_body_word_1 = ce_row_word_1 + 32 + (unsigned long long)e2_block * 24;
                                            {
                                                int4 _iv4 = make_int4(ce_w_1[0 + 0], ce_w_1[0 + 1], ce_w_1[0 + 2], ce_w_1[0 + 3]);
                                                *reinterpret_cast<int4*>(reinterpret_cast<int*>(result_out) + (ce_body_word_1 + (unsigned long long)epi_tid * 4) + 0) = _iv4;
                                            }
                                            {
                                                int2 _iv2 = make_int2(ce_w_1[4 + 0], ce_w_1[4 + 1]);
                                                *reinterpret_cast<int2*>(reinterpret_cast<int*>(result_out) + (ce_body_word_1 + 16 + (unsigned long long)epi_tid * 2) + 0) = _iv2;
                                            }
                                            if (epi_tid == 0) {
                                                unsigned long long ce_hdr_byte_e_1 = ce_row_word_1 * 4 + (unsigned long long)e2_block * 2;
                                                unsigned int ce_hdr_lo_e_1 = ce_emax_1;
                                                unsigned int ce_hdr_hi_e_1 = 0;
                                                *(reinterpret_cast<unsigned char*>(reinterpret_cast<uint8_t*>(result_out) + ce_hdr_byte_e_1) + (0)) = (unsigned char)(ce_hdr_lo_e_1);
                                                *(reinterpret_cast<unsigned char*>(reinterpret_cast<uint8_t*>(result_out) + (ce_hdr_byte_e_1 + 1)) + (0)) = (unsigned char)(ce_hdr_hi_e_1);
                                            }
                                        }
                                    }
                                }
                            }
                        }
                    }
                    __threadfence();
                    __syncwarp();
                    unsigned long long gtimer_0_14;
                    asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(gtimer_0_14) :: "memory");
                    unsigned long long e2_end_ts = gtimer_0_14;
                    int e2_completion_count = (7 - epi_vw_base) / 3 + 1;
                    if (elect_sync()) {
                        if (e2_pool_row >= 204736) {
                            atomicMax(&phase_timestamps[24], e2_end_ts);
                        }
                        int _atomic_old_31 = atomicAdd(&w2_warp_done[e2_tile], e2_completion_count);
                        int e2_previous = _atomic_old_31;
                        if (e2_previous + e2_completion_count == 8) {
                            atomicAdd(&w2_tiles_completed[0], 1);
                            {
                                unsigned int* _gc_p = reinterpret_cast<unsigned int*>(w2_task_counter) + (e2_task);
                                unsigned int _gc_old;
                                asm volatile("atom.release.gpu.global.add.u32 %0, [%1], 1;" : "=r"(_gc_old) : "l"(_gc_p) : "memory");
                            }
                        } else if (e2_previous + e2_completion_count > 8) {
                            atomicMax(&protocol_error[0], 1);
                        }
                    }
                }
                __syncwarp();
                mbarrier_arrive(epi_free_addr);
            }
        }
    }
    if (bid == 0 && tid == 0) {
        unsigned long long gtimer_0_15;
        asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(gtimer_0_15) :: "memory");
        phase_timestamps[9] = gtimer_0_15;
    }
    if (bid == 0 && tid == 0) {
        unsigned long long gtimer_0_16;
        asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(gtimer_0_16) :: "memory");
        phase_timestamps[10] = gtimer_0_16;
    }
    if (bid == 1 && tid == 0) {
        unsigned long long gtimer_0_17;
        asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(gtimer_0_17) :: "memory");
        phase_timestamps[31] = gtimer_0_17;
    }
    if (bid == 2 && tid == 32) {
        unsigned long long gtimer_0_18;
        asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(gtimer_0_18) :: "memory");
        phase_timestamps[32] = gtimer_0_18;
    }
    if (bid == 0) {
        if (warp >= 1 && warp < world_size) {
            int map_source = rank + warp;
            if (map_source >= world_size) {
                map_source = map_source - world_size;
            }
            if (elect_sync()) {
                int map_records = source_record_counts[map_source];
                int _max_81 = (((map_records + 8 - 1) / 8) > (dispatch_chunk_min_records) ? ((map_records + 8 - 1) / 8) : (dispatch_chunk_min_records));
                int map_q = _max_81;
                int map_chunks = (map_records + map_q - 1) / map_q;
                #pragma unroll 1
                for (int map_chunk = 0; map_chunk < map_chunks; map_chunk++) {
                    {
                        unsigned int* _gca_p = reinterpret_cast<unsigned int*>(dispatch_chunk_scatter_counter) + (map_source * 8 + map_chunk);
                        while (true) {
                            unsigned int _gca_v;
                            asm volatile("ld.acquire.gpu.global.u32 %0, [%1];" : "=r"(_gca_v) : "l"(_gca_p));
                            if (_gca_v >= (unsigned int)(dispatch_chunk_targets[map_source * 8 + map_chunk])) break;
                        }
                    }
                }
                int _max_82 = ((source_route_counts[map_source]) > (0) ? (source_route_counts[map_source]) : (0));
                int _min_47 = ((_max_82) < (49152) ? (_max_82) : (49152));
                int map_routes = _min_47;
                int _max_83 = ((map_routes) > (1) ? (map_routes) : (1));
                int map_count = _max_83;
                unsigned long long map_local_byte = (unsigned long long)(map_source * 2 + slot) * 409141248 + 408944640;
                unsigned long long map_remote_byte = (unsigned long long)(rank * 2 + slot) * 409141248 + 408944640;
                __threadfence_system();
                // gin_put_signal_add: strong remote completion on context 0
                {
                    ncclGin __gin{*(gin_dev_comm), (int)(0)};
                    __gin.put(ncclTeamWorld(*(gin_dev_comm)), (int)(map_source), result_inbox_window, (size_t)(map_remote_byte), result_out_window, (size_t)(map_local_byte), (size_t)(map_count * 4),
                        ncclGin_StrongSignalAdd{(ncclGinSignal_t)(4 + rank), (uint64_t)(1)}, ncclGin_None{}, ncclCoopThread());
                }
                int h29_idx = 0;
                #pragma unroll 1
                for (int h29_iter = 0; h29_iter < 3084; h29_iter++) {
                    int h29_ready = 0;
                    #pragma unroll 1
                    for (int h29_spin = 0; h29_spin < 1073741824; h29_spin++) {
                        int _atomic_old_32 = atomicAdd(&svc_qctl[2], 0);
                        int h29_done = _atomic_old_32;
                        __threadfence_block();
                        int _atomic_old_33 = atomicAdd(&svc_qctl[1], 0);
                        int h29_tail_seen = _atomic_old_33;
                        if (h29_tail_seen > h29_idx) {
                            h29_ready = 1;
                            break;
                        }
                        if (h29_done != 0) {
                            break;
                        }
                    }
                    if (h29_ready == 0) {
                        break;
                    }
                    __threadfence_block();
                    int h29_chunk = atomicAdd(&svc_queue[h29_idx & 1023], 0);
                    h29_idx += 1;
                    // Finish reading before publishing that this slot is reusable.
                    __threadfence_block();
                    atomicMax(&svc_qctl[3 + warp], h29_idx);
                    int h29_source = h29_chunk / 769;
                    if (h29_source == map_source) {
                        int h29_chunk_local = h29_chunk - h29_source * 769;
                        int _max_84 = ((source_route_counts[h29_source]) > (0) ? (source_route_counts[h29_source]) : (0));
                        int _min_48 = ((_max_84) < (49152) ? (_max_84) : (49152));
                        int h29_routes = _min_48;
                        int h29_pq = 128;
                        if (h29_routes >= 6144) {
                            h29_pq = 64;
                        }
                        if (h29_routes < 768) {
                            h29_pq = 32;
                        }
                        int _max_85 = (((h29_routes + h29_pq - 1) / h29_pq - 1) > (0) ? ((h29_routes + h29_pq - 1) / h29_pq - 1) : (0));
                        int h29_pfull = _max_85;
                        int h29_pts = h29_pfull * h29_pq;
                        int h29_start = 0;
                        int h29_cap = h29_pq;
                        if (h29_chunk_local < h29_pfull) {
                            h29_start = h29_chunk_local * h29_pq;
                        } else {
                            h29_start = h29_pts + (h29_chunk_local - h29_pfull) * 64;
                            h29_cap = 64;
                        }
                        int _min_49 = ((h29_routes - h29_start) < (h29_cap) ? (h29_routes - h29_start) : (h29_cap));
                        int h29_chunk_rows = _min_49;
                        int _atomic_old_34 = atomicAdd(&result_ovf_cursor[h29_source * 769 + h29_chunk_local], 0);
                        int h29_ovf = _atomic_old_34;
                        unsigned long long h29_local_byte = (unsigned long long)(h29_source * 2 + slot) * 409141248 + (unsigned long long)h29_start * 8320;
                        unsigned long long h29_remote_byte = (unsigned long long)(rank * 2 + slot) * 409141248 + (unsigned long long)h29_start * 8320;
                        int _min_50 = ((h29_ovf) < (h29_chunk_rows * 16) ? (h29_ovf) : (h29_chunk_rows * 16));
                        int h29_put_bytes = h29_chunk_rows * 6272 + _min_50 * 128;
                        int _max_86 = ((owner_route_counts[h29_source]) > (0) ? (owner_route_counts[h29_source]) : (0));
                        int _min_51 = ((_max_86) < (49152) ? (_max_86) : (49152));
                        int c13a2_dual = (int)(h29_routes * 8 >= active_rows * 18 || h29_routes * 8 >= active_rows * 15 && _min_51 * 8 >= active_rows * 18);
                        __threadfence_system();
                        if (c13a2_dual != 0 && (h29_chunk_local & 1) == 1) {
                            // gin_put_signal_add: strong remote completion on context 1
                            {
                                ncclGin __gin{*(gin_dev_comm), (int)(1)};
                                __gin.put(ncclTeamWorld(*(gin_dev_comm)), (int)(h29_source), result_inbox_window, (size_t)(h29_remote_byte), result_out_window, (size_t)(h29_local_byte), (size_t)(h29_put_bytes),
                                    ncclGin_StrongSignalAdd{(ncclGinSignal_t)(44 + rank), (uint64_t)(1)}, ncclGin_None{}, ncclCoopThread());
                            }
                        } else {
                            // gin_put_signal_add: strong remote completion on context 0
                            {
                                ncclGin __gin{*(gin_dev_comm), (int)(0)};
                                __gin.put(ncclTeamWorld(*(gin_dev_comm)), (int)(h29_source), result_inbox_window, (size_t)(h29_remote_byte), result_out_window, (size_t)(h29_local_byte), (size_t)(h29_put_bytes),
                                    ncclGin_StrongSignalAdd{(ncclGinSignal_t)(4 + rank), (uint64_t)(1)}, ncclGin_None{}, ncclCoopThread());
                            }
                        }
                    }
                }
                unsigned long long gtimer_0_19;
                asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(gtimer_0_19) :: "memory");
                atomicMax(&phase_timestamps[15], gtimer_0_19);
            }
            __syncwarp();
        }
        if (warp == 0) {
            __syncwarp();
            int _max_87 = ((total_m_tasks[0]) > (0) ? (total_m_tasks[0]) : (0));
            int _min_52 = ((_max_87) < (1664) ? (_max_87) : (1664));
            int service_task_total = _min_52;
            int h20_tail = 0;
            #pragma unroll 1
            for (int service_task = 0; service_task < service_task_total; service_task++) {
                if (service_task < total_m_tasks[0]) {
                    if (elect_sync()) {
                        {
                            unsigned int* _gca_p = reinterpret_cast<unsigned int*>(w2_task_counter) + (service_task);
                            while (true) {
                                unsigned int _gca_v;
                                asm volatile("ld.acquire.gpu.global.u32 %0, [%1];" : "=r"(_gca_v) : "l"(_gca_p));
                                if (_gca_v >= (unsigned int)(32)) break;
                            }
                        }
                    }
                    __syncwarp();
                    int service_row_base = task_pool_row[service_task];
                    #pragma unroll 1
                    for (int service_row_offset = lane; service_row_offset < 128; service_row_offset += 32) {
                        service_ready_chunks[service_row_offset] = -1;
                    }
                    __syncwarp();
                    #pragma unroll 1
                    for (int service_row_offset_2 = lane; service_row_offset_2 < 128; service_row_offset_2 += 32) {
                        int service_row = service_row_base + service_row_offset_2;
                        int service_source = meta_source_rank[service_row];
                        if (service_source >= 0 && service_source < world_size) {
                            int _max_88 = ((source_route_counts[service_source]) > (0) ? (source_route_counts[service_source]) : (0));
                            int _min_53 = ((_max_88) < (49152) ? (_max_88) : (49152));
                            int c41_tr = _min_53;
                            int c41_tq = 128;
                            if (c41_tr >= 6144) {
                                c41_tq = 64;
                            }
                            if (c41_tr < 768) {
                                c41_tq = 32;
                            }
                            int _max_89 = (((c41_tr + c41_tq - 1) / c41_tq - 1) > (0) ? ((c41_tr + c41_tq - 1) / c41_tq - 1) : (0));
                            int c41_tfull = _max_89;
                            int c41_tts = c41_tfull * c41_tq;
                            int c41_tidx = meta_result_index[service_row];
                            int service_chunk_local = 0;
                            if (c41_tidx < c41_tts) {
                                service_chunk_local = c41_tidx / c41_tq;
                            } else {
                                service_chunk_local = c41_tfull + (c41_tidx - c41_tts) / 64;
                            }
                            int service_chunk = service_source * 769 + service_chunk_local;
                            int _atomic_old_35 = atomicAdd(&result_chunk_tally[service_chunk], 1);
                            int service_previous = _atomic_old_35;
                            int service_total = c41_tq;
                            if (service_chunk_local >= c41_tfull) {
                                int _min_54 = ((64) < (c41_tr - (c41_tts + (service_chunk_local - c41_tfull) * 64)) ? (64) : (c41_tr - (c41_tts + (service_chunk_local - c41_tfull) * 64)));
                                service_total = _min_54;
                            }
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
                        for (int service_ready_row = 0; service_ready_row < 128; service_ready_row++) {
                            int service_chunk_2 = service_ready_chunks[service_ready_row];
                            if (service_chunk_2 >= 0) {
                                #pragma unroll 1
                                for (int h20_full_spin = 0; h20_full_spin < 1073741824; h20_full_spin++) {
                                    int h20_consumed = 1073741824;
                                    #pragma unroll 1
                                    for (int h29_cw = 1; h29_cw < 4; h29_cw++) {
                                        if (h29_cw < world_size) {
                                            int _atomic_old_36 = atomicAdd(&svc_qctl[3 + h29_cw], 0);
                                            int h29_cur = _atomic_old_36;
                                            int _min_55 = ((h20_consumed) < (h29_cur) ? (h20_consumed) : (h29_cur));
                                            h20_consumed = _min_55;
                                        }
                                    }
                                    if (h20_tail - h20_consumed < 1024) {
                                        break;
                                    }
                                }
                                __threadfence_block();
                                // Single producer: replace atomically before publishing the tail.
                                int h20_old_chunk = atomicAdd(&svc_queue[h20_tail & 1023], 0);
                                atomicAdd(&svc_queue[h20_tail & 1023], service_chunk_2 - h20_old_chunk);
                                h20_tail += 1;
                                __threadfence_block();
                                atomicAdd(&svc_qctl[1], 1);
                            }
                        }
                    }
                    __syncwarp();
                }
            }
            if (elect_sync()) {
                __threadfence_block();
                atomicAdd(&svc_qctl[2], 1);
                unsigned long long gtimer_0_20;
                asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(gtimer_0_20) :: "memory");
                atomicMax(&phase_timestamps[15], gtimer_0_20);
                #pragma unroll 1
                for (int tail_source_i = 0; tail_source_i < 4; tail_source_i++) {
                    if (tail_source_i < world_size) {
                        int tail_source = tail_source_i + rank + 1;
                        if (tail_source >= world_size) {
                            tail_source = tail_source - world_size;
                        }
                        int _max_90 = ((source_route_counts[tail_source]) > (0) ? (source_route_counts[tail_source]) : (0));
                        int _min_56 = ((_max_90) < (49152) ? (_max_90) : (49152));
                        int tail_routes = _min_56;
                        if (tail_routes == 0 && tail_source != rank) {
                            unsigned long long tail_local_byte = (unsigned long long)(tail_source * 2 + slot) * 409141248;
                            unsigned long long tail_remote_byte = (unsigned long long)(rank * 2 + slot) * 409141248;
                            __threadfence_system();
                            // gin_put_signal_add: strong remote completion on context 0
                            {
                                ncclGin __gin{*(gin_dev_comm), (int)(0)};
                                __gin.put(ncclTeamWorld(*(gin_dev_comm)), (int)(tail_source), result_inbox_window, (size_t)(tail_remote_byte), result_out_window, (size_t)(tail_local_byte), (size_t)(8192),
                                    ncclGin_StrongSignalAdd{(ncclGinSignal_t)(4 + rank), (uint64_t)(1)}, ncclGin_None{}, ncclCoopThread());
                            }
                        }
                        int c41_aq = 128;
                        if (tail_routes >= 6144) {
                            c41_aq = 64;
                        }
                        if (tail_routes < 768) {
                            c41_aq = 32;
                        }
                        int _max_91 = (((tail_routes + c41_aq - 1) / c41_aq - 1) > (0) ? ((tail_routes + c41_aq - 1) / c41_aq - 1) : (0));
                        int c41_afull = _max_91;
                        int tail_chunk_count = c41_afull + (tail_routes - c41_afull * c41_aq + 64 - 1) / 64;
                        #pragma unroll 1
                        for (int verify_chunk = 0; verify_chunk < tail_chunk_count; verify_chunk++) {
                            if (verify_chunk < 769) {
                                int verify_index = tail_source * 769 + verify_chunk;
                                int verify_tally = result_chunk_tally[verify_index];
                                int verify_start = verify_chunk * c41_aq;
                                int verify_total = c41_aq;
                                if (c41_afull <= verify_chunk) {
                                    verify_start = c41_afull * c41_aq + (verify_chunk - c41_afull) * 64;
                                    int _min_57 = ((64) < (tail_routes - verify_start) ? (64) : (tail_routes - verify_start));
                                    verify_total = _min_57;
                                }
                                if (verify_tally != result_chunk_total[verify_index] || verify_tally != verify_total) {
                                    atomicMax(&protocol_error[0], 1);
                                }
                                if (verify_tally < verify_total && tail_source != rank) {
                                    unsigned long long verify_local_byte = (unsigned long long)(tail_source * 2 + slot) * 409141248 + (unsigned long long)verify_start * 8320;
                                    unsigned long long verify_remote_byte = (unsigned long long)(rank * 2 + slot) * 409141248 + (unsigned long long)verify_start * 8320;
                                    int _max_92 = ((owner_route_counts[tail_source]) > (0) ? (owner_route_counts[tail_source]) : (0));
                                    int _min_58 = ((_max_92) < (49152) ? (_max_92) : (49152));
                                    int c13a2_dual_v = (int)(tail_routes * 8 >= active_rows * 18 || tail_routes * 8 >= active_rows * 15 && _min_58 * 8 >= active_rows * 18);
                                    __threadfence_system();
                                    if (c13a2_dual_v != 0 && (verify_chunk & 1) == 1) {
                                        // gin_put_signal_add: strong remote completion on context 1
                                        {
                                            ncclGin __gin{*(gin_dev_comm), (int)(1)};
                                            __gin.put(ncclTeamWorld(*(gin_dev_comm)), (int)(tail_source), result_inbox_window, (size_t)(verify_remote_byte), result_out_window, (size_t)(verify_local_byte), (size_t)(verify_total * 6272),
                                                ncclGin_StrongSignalAdd{(ncclGinSignal_t)(44 + rank), (uint64_t)(1)}, ncclGin_None{}, ncclCoopThread());
                                        }
                                    } else {
                                        // gin_put_signal_add: strong remote completion on context 0
                                        {
                                            ncclGin __gin{*(gin_dev_comm), (int)(0)};
                                            __gin.put(ncclTeamWorld(*(gin_dev_comm)), (int)(tail_source), result_inbox_window, (size_t)(verify_remote_byte), result_out_window, (size_t)(verify_local_byte), (size_t)(verify_total * 6272),
                                                ncclGin_StrongSignalAdd{(ncclGinSignal_t)(4 + rank), (uint64_t)(1)}, ncclGin_None{}, ncclCoopThread());
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
                unsigned long long gtimer_1_3;
                asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(gtimer_1_3) :: "memory");
                phase_timestamps[16] = gtimer_1_3;
            }
        }
    }
    cooperative_groups::this_grid().sync();
    if (bid == 0 && tid == 0) {
        unsigned long long gtimer_0_21;
        asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(gtimer_0_21) :: "memory");
        phase_timestamps[11] = gtimer_0_21;
    }
    if (bid == 0 && tid == 0) {
        if (w1_tiles_completed[0] != total_m_tasks[0] * 32) {
            atomicMax(&protocol_error[0], 1);
        }
        int expected_requant_groups = total_m_tasks[0] * 128 * 64;
        if (requant_groups_done[0] != expected_requant_groups) {
            atomicMax(&protocol_error[0], 1);
        }
        int scatter_sum = 0;
        #pragma unroll 1
        for (int audit_es = 0; audit_es < 256; audit_es++) {
            int audit_e = audit_es / 4;
            int audit_s = audit_es - audit_e * 4;
            if (audit_s < world_size) {
                scatter_sum = scatter_sum + expert_source_offsets[audit_es];
                if (expert_source_offsets[audit_es] != source_expert_counts[audit_s * 64 + audit_e]) {
                    atomicMax(&protocol_error[0], 1);
                }
            }
        }
        if (scatter_sum != total_valid_routes[0]) {
            atomicMax(&protocol_error[0], 1);
        }
        if (w2_tiles_completed[0] != total_m_tasks[0] * 32) {
            atomicMax(&protocol_error[0], 1);
        }
    }
    if (bid == 0 && tid == 0) {
        unsigned long long gtimer_0_22;
        asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(gtimer_0_22) :: "memory");
        phase_timestamps[12] = gtimer_0_22;
    }
    if (bid == 0 && warp >= 1 && warp < world_size) {
        int wait_owner = rank + warp;
        if (wait_owner >= world_size) {
            wait_owner = wait_owner - world_size;
        }
        if (elect_sync()) {
            int owner_routes = owner_route_counts[wait_owner];
            int _max_93 = ((owner_routes) > (0) ? (owner_routes) : (0));
            int _min_59 = ((_max_93) < (49152) ? (_max_93) : (49152));
            int safe_owner_routes = _min_59;
            int c41_wq = 128;
            if (safe_owner_routes >= 6144) {
                c41_wq = 64;
            }
            if (safe_owner_routes < 768) {
                c41_wq = 32;
            }
            int _max_94 = (((safe_owner_routes + c41_wq - 1) / c41_wq - 1) > (0) ? ((safe_owner_routes + c41_wq - 1) / c41_wq - 1) : (0));
            int c41_wfull = _max_94;
            int _max_95 = ((c41_wfull + (safe_owner_routes - c41_wfull * c41_wq + 64 - 1) / 64) > (1) ? (c41_wfull + (safe_owner_routes - c41_wfull * c41_wq + 64 - 1) / 64) : (1));
            int owner_chunk_count = _max_95 + 1;
            #pragma unroll 1
            for (int h29_k = 1; h29_k < 771; h29_k++) {
                if (owner_chunk_count < h29_k) {
                    break;
                }
                int c13a_c = h29_k - 2;
                int _max_96 = ((source_route_counts[wait_owner]) > (0) ? (source_route_counts[wait_owner]) : (0));
                int _min_60 = ((_max_96) < (49152) ? (_max_96) : (49152));
                int c13a2_dual_r = (int)(safe_owner_routes * 8 >= active_rows * 18 || safe_owner_routes * 8 >= active_rows * 15 && _min_60 * 8 >= active_rows * 18);
                if (c13a2_dual_r != 0 && c13a_c >= 0 && (c13a_c & 1) == 1) {
                    // gin_wait_signal: acquire, rolling 64-bit comparison
                    {
                        ncclGin __gin{*(gin_dev_comm), (int)(1)};
                        __gin.waitSignal(ncclCoopThread(), (ncclGinSignal_t)(44 + wait_owner), (uint64_t)(result_signal_base_scratch[4 + wait_owner] + (unsigned long long)((c13a_c + 1) / 2)), 64, cuda::memory_order_acquire);
                    }
                } else {
                    int c13a_ord0 = 1;
                    if (c13a_c >= 0) {
                        if (c13a2_dual_r != 0) {
                            c13a_ord0 = c13a_c / 2 + 2;
                        } else {
                            c13a_ord0 = c13a_c + 2;
                        }
                    }
                    // gin_wait_signal: acquire, rolling 64-bit comparison
                    {
                        ncclGin __gin{*(gin_dev_comm), (int)(0)};
                        __gin.waitSignal(ncclCoopThread(), (ncclGinSignal_t)(4 + wait_owner), (uint64_t)(result_signal_base_scratch[wait_owner] + (unsigned long long)c13a_ord0), 64, cuda::memory_order_acquire);
                    }
                }
                {
                    unsigned int* _gc_p = reinterpret_cast<unsigned int*>(result_owner_progress) + (wait_owner);
                    unsigned int _gc_old;
                    asm volatile("atom.release.gpu.global.add.u32 %0, [%1], 1;" : "=r"(_gc_old) : "l"(_gc_p) : "memory");
                }
            }
            {
                unsigned int* _gc_p = reinterpret_cast<unsigned int*>(result_owner_ready) + (wait_owner);
                unsigned int _gc_old;
                asm volatile("atom.release.gpu.global.add.u32 %0, [%1], 1;" : "=r"(_gc_old) : "l"(_gc_p) : "memory");
            }
        }
        __syncwarp();
    }
    #pragma unroll 1
    for (int h37_iter = 0; h37_iter < 8193; h37_iter++) {
        int combine_token = 0;
        if (elect_sync()) {
            int _atomic_old_37 = atomicAdd(&combine_claim_cursor[0], 1);
            combine_token = _atomic_old_37;
        }
        int _shfl_33 = __shfl_sync(0xFFFFFFFF, combine_token, 0);
        combine_token = _shfl_33;
        if (combine_token >= active_rows) {
            break;
        }
        unsigned long long combine_bases[6];
        unsigned long long combine_ovfs[6];
        int combine_valids[6];
        int combine_selfs[6];
        #pragma unroll
        for (int meta_slot_1 = 0; meta_slot_1 < 6; meta_slot_1++) {
            int meta_pair = combine_token * 6 + meta_slot_1;
            int meta_expert = topk_idx_i32[meta_pair * 2];
            int meta_expert_hi = topk_idx_i32[meta_pair * 2 + 1];
            int meta_masked = (int)(meta_expert == -1 && meta_expert_hi == -1);
            int meta_valid = (int)(meta_expert >= 0 && meta_expert < world_size * 64 && meta_expert_hi == 0);
            if (meta_valid == 0 && meta_masked == 0) {
                if (lane == 0) {
                    atomicMax(&protocol_error[0], 1);
                }
            }
            combine_valids[meta_slot_1] = 0;
            combine_bases[meta_slot_1] = 0;
            combine_ovfs[meta_slot_1] = 0;
            combine_selfs[meta_slot_1] = 0;
            if (meta_valid != 0) {
                int meta_owner = meta_expert / 64;
                if (meta_owner != rank) {
                    if (elect_sync()) {
                        {
                            unsigned int* _gca_p = reinterpret_cast<unsigned int*>(result_owner_progress) + (meta_owner);
                            while (true) {
                                unsigned int _gca_v;
                                asm volatile("ld.acquire.gpu.global.u32 %0, [%1];" : "=r"(_gca_v) : "l"(_gca_p));
                                if (_gca_v >= (unsigned int)(1)) break;
                            }
                        }
                    }
                }
                __syncwarp();
                int meta_result_slot = route_result_index[meta_pair];
                int meta_owner_count = owner_route_counts[meta_owner];
                if (meta_result_slot < 0 || meta_result_slot >= meta_owner_count) {
                    if (lane == 0) {
                        atomicMax(&protocol_error[0], 1);
                    }
                } else {
                    int* meta_map_i32 = reinterpret_cast<int*>(result_inbox);
                    if (meta_owner == rank) {
                        meta_map_i32 = reinterpret_cast<int*>(result_out);
                    }
                    int meta_new_slot = meta_map_i32[(unsigned long long)(meta_owner * 2 + slot) * 102285312 + 102236160 + (unsigned long long)meta_result_slot];
                    if (meta_new_slot < 0 || meta_new_slot >= meta_owner_count) {
                        if (lane == 0) {
                            atomicMax(&protocol_error[0], 1);
                        }
                    } else {
                        int cd_rq = 128;
                        if (meta_owner_count >= 6144) {
                            cd_rq = 64;
                        }
                        if (meta_owner_count < 768) {
                            cd_rq = 32;
                        }
                        int _max_97 = (((meta_owner_count + cd_rq - 1) / cd_rq - 1) > (0) ? ((meta_owner_count + cd_rq - 1) / cd_rq - 1) : (0));
                        int cd_rfull = _max_97;
                        int cd_rts = cd_rfull * cd_rq;
                        int cd_chunk = 0;
                        int cd_start = 0;
                        int cd_rows = 0;
                        if (meta_new_slot < cd_rts) {
                            cd_chunk = meta_new_slot / cd_rq;
                            cd_start = cd_chunk * cd_rq;
                            cd_rows = cd_rq;
                        } else {
                            cd_chunk = cd_rfull + (meta_new_slot - cd_rts) / 64;
                            cd_start = cd_rts + (cd_chunk - cd_rfull) * 64;
                            int _min_61 = ((64) < (meta_owner_count - cd_start) ? (64) : (meta_owner_count - cd_start));
                            cd_rows = _min_61;
                        }
                        if (meta_owner != rank) {
                            if (elect_sync()) {
                                {
                                    unsigned int* _gca_p = reinterpret_cast<unsigned int*>(result_owner_progress) + (meta_owner);
                                    while (true) {
                                        unsigned int _gca_v;
                                        asm volatile("ld.acquire.gpu.global.u32 %0, [%1];" : "=r"(_gca_v) : "l"(_gca_p));
                                        if (_gca_v >= (unsigned int)(cd_chunk + 2)) break;
                                    }
                                }
                            }
                            __syncwarp();
                        }
                        combine_valids[meta_slot_1] = 1;
                        combine_selfs[meta_slot_1] = (int)(meta_owner == rank);
                        unsigned long long cd_chunk_word = (unsigned long long)(meta_owner * 2 + slot) * 102285312 + (unsigned long long)cd_start * 2080;
                        combine_bases[meta_slot_1] = cd_chunk_word + (unsigned long long)(meta_new_slot - cd_start) * 1568;
                        combine_ovfs[meta_slot_1] = cd_chunk_word + (unsigned long long)cd_rows * 1568;
                    }
                }
            }
        }
        unsigned long long combine_out_base = (unsigned long long)combine_token * 4096;
        int cd_q = lane & 3;
        #pragma unroll 1
        for (int cd_blk = lane >> 2; cd_blk < 64; cd_blk += 8) {
            float cd_acc[16];
            cd_acc[0] = 0.0f;
            cd_acc[1] = 0.0f;
            cd_acc[2] = 0.0f;
            cd_acc[3] = 0.0f;
            cd_acc[4] = 0.0f;
            cd_acc[5] = 0.0f;
            cd_acc[6] = 0.0f;
            cd_acc[7] = 0.0f;
            cd_acc[8] = 0.0f;
            cd_acc[9] = 0.0f;
            cd_acc[10] = 0.0f;
            cd_acc[11] = 0.0f;
            cd_acc[12] = 0.0f;
            cd_acc[13] = 0.0f;
            cd_acc[14] = 0.0f;
            cd_acc[15] = 0.0f;
            #pragma unroll
            for (int cd_slot = 0; cd_slot < 6; cd_slot++) {
                int* cd_src_i32 = reinterpret_cast<int*>(result_inbox);
                __nv_bfloat16* cd_src_bf16 = reinterpret_cast<__nv_bfloat16*>(result_inbox);
                if (combine_selfs[cd_slot] != 0) {
                    cd_src_i32 = reinterpret_cast<int*>(result_out);
                    cd_src_bf16 = reinterpret_cast<__nv_bfloat16*>(result_out);
                }
                int _vec_load_16[1];
                {
                    _vec_load_16[0] = *reinterpret_cast<const int*>(cd_src_i32 + (combine_bases[cd_slot] + (unsigned long long)(cd_blk >> 1)) + 0);
                }
                unsigned long long cd_body_word = combine_bases[cd_slot] + 32 + (unsigned long long)cd_blk * 24;
                int _vec_load_17[4];
                {
                    int4 _iv4 = *reinterpret_cast<const int4*>(cd_src_i32 + (cd_body_word + (unsigned long long)cd_q * 4) + 0);
                    _vec_load_17[0 + 0] = _iv4.x;
                    _vec_load_17[0 + 1] = _iv4.y;
                    _vec_load_17[0 + 2] = _iv4.z;
                    _vec_load_17[0 + 3] = _iv4.w;
                }
                int _vec_load_18[1];
                {
                    _vec_load_18[0] = *reinterpret_cast<const int*>(cd_src_i32 + (cd_body_word + 16 + (unsigned long long)cd_q * 2) + 0);
                }
                int _vec_load_19[1];
                {
                    _vec_load_19[0] = *reinterpret_cast<const int*>(cd_src_i32 + (cd_body_word + 17 + (unsigned long long)cd_q * 2) + 0);
                }
                int cd_h = _vec_load_16[0];
                unsigned int cd_h16 = (unsigned int)cd_h >> (unsigned int)((cd_blk & 1) * 16) & 65535;
                unsigned int cd_emax = cd_h16 & 255;
                int cd_u0i = _vec_load_17[0];
                int cd_u1i = _vec_load_17[1];
                int cd_u2i = _vec_load_17[2];
                int cd_u3i = _vec_load_17[3];
                int cd_u4i = _vec_load_18[0];
                int cd_u5i = _vec_load_19[0];
                unsigned int cd_u0 = (unsigned int)cd_u0i;
                unsigned int cd_u1 = (unsigned int)cd_u1i;
                unsigned int cd_u2 = (unsigned int)cd_u2i;
                unsigned int cd_u3 = (unsigned int)cd_u3i;
                unsigned int cd_u4 = (unsigned int)cd_u4i;
                unsigned int cd_u5 = (unsigned int)cd_u5i;
                unsigned int cd_c[16];
                cd_c[0] = cd_u0 & 4095;
                cd_c[1] = cd_u0 >> 12 & 4095;
                cd_c[2] = cd_u0 >> 24 & 255 | (cd_u1 & 15) << 8;
                cd_c[3] = cd_u1 >> 4 & 4095;
                cd_c[4] = cd_u1 >> 16 & 4095;
                cd_c[5] = cd_u1 >> 28 & 15 | (cd_u2 & 255) << 4;
                cd_c[6] = cd_u2 >> 8 & 4095;
                cd_c[7] = cd_u2 >> 20 & 4095;
                cd_c[8] = cd_u3 & 4095;
                cd_c[9] = cd_u3 >> 12 & 4095;
                cd_c[10] = cd_u3 >> 24 & 255 | (cd_u4 & 15) << 8;
                cd_c[11] = cd_u4 >> 4 & 4095;
                cd_c[12] = cd_u4 >> 16 & 4095;
                cd_c[13] = cd_u4 >> 28 & 15 | (cd_u5 & 255) << 4;
                cd_c[14] = cd_u5 >> 8 & 4095;
                cd_c[15] = cd_u5 >> 20 & 4095;
                float cd_vals[16];
                #pragma unroll
                for (int cd_kk = 0; cd_kk < 16; cd_kk++) {
                    unsigned int cd_vb = (cd_c[cd_kk] >> 11 << 15 | cd_emax - (cd_c[cd_kk] >> 7 & 15) << 7 | cd_c[cd_kk] & 127) << 16;
                    float cd_vf = 0.0f;
                    cd_vf = reinterpret_cast<float*>(&cd_vb)[0];
                    cd_vals[cd_kk] = cd_vf;
                }
                if ((cd_h16 & 32768) != 0 && combine_valids[cd_slot] != 0) {
                    unsigned int cd_raw_idx = cd_h16 & 32767;
                    float _vec_load_20[16];
                    {
                        const uint4* _vptr_0 = reinterpret_cast<const uint4*>(cd_src_bf16 + (combine_ovfs[cd_slot] * 2 + (unsigned long long)cd_raw_idx * 64 + (unsigned long long)cd_q * 16) + 0);
                        uint4 _vld_0[2];
                        #pragma unroll
                        for (int _blk = 0; _blk < 2; _blk++) {
                            _vld_0[_blk] = _vptr_0[_blk];
                            uint32_t* _vpairs_0 = reinterpret_cast<uint32_t*>(&_vld_0[_blk]);
                            #pragma unroll
                            for (int _pair = 0; _pair < 4; _pair++) {
                                asm volatile(
                                    "{\n\t"
                                    "shl.b32 %0, %2, 16;\n\t"
                                    "and.b32 %1, %2, 0xffff0000;\n\t"
                                    "}\n"
                                    : "=f"((&_vec_load_20[0 + _blk * 8 + _pair * 2])[0]), "=f"((&_vec_load_20[0 + _blk * 8 + _pair * 2])[1])
                                    : "r"(_vpairs_0[_pair]));
                            }
                        }
                    }
                    #pragma unroll
                    for (int cd_rk = 0; cd_rk < 16; cd_rk++) {
                        cd_vals[cd_rk] = _vec_load_20[cd_rk];
                    }
                }
                if (combine_valids[cd_slot] != 0) {
                    #pragma unroll
                    for (int cd_ak = 0; cd_ak < 16; cd_ak++) {
                        cd_acc[cd_ak] = cd_acc[cd_ak] + cd_vals[cd_ak];
                    }
                }
            }
            unsigned long long cd_col0 = combine_out_base + (unsigned long long)cd_blk * 64 + (unsigned long long)cd_q * 2;
            #pragma unroll
            for (int cd_nt = 0; cd_nt < 8; cd_nt++) {
                float _vec_load_21[2];
                {
                    uint32_t _bf16x2_bits_1;
                    _bf16x2_bits_1 = *reinterpret_cast<const uint32_t*>(shared_out + (cd_col0 + (unsigned long long)(cd_nt * 8)) + 0);
                    asm volatile(
                        "{\n\t"
                        "shl.b32 %0, %2, 16;\n\t"
                        "and.b32 %1, %2, 0xffff0000;\n\t"
                        "}\n"
                        : "=f"((&_vec_load_21[0])[0]), "=f"((&_vec_load_21[0])[1])
                        : "r"(_bf16x2_bits_1));
                }
                cd_acc[cd_nt * 2] = cd_acc[cd_nt * 2] + _vec_load_21[0];
                cd_acc[cd_nt * 2 + 1] = cd_acc[cd_nt * 2 + 1] + _vec_load_21[1];
            }
            #pragma unroll
            for (int cd_nt2 = 0; cd_nt2 < 8; cd_nt2++) {
                {
                    __nv_bfloat162 _pk = __floats2bfloat162_rn(cd_acc[cd_nt2 * 2 + 0], cd_acc[cd_nt2 * 2 + 1]);
                    *reinterpret_cast<__nv_bfloat162*>(&((__nv_bfloat16*)(final_output + (cd_col0 + (unsigned long long)(cd_nt2 * 8))))[0]) = _pk;
                }
            }
        }
    }
    __threadfence_system();
    cooperative_groups::this_grid().sync();
    if (bid == 0 && tid == 0) {
        unsigned long long gtimer_0_23;
        asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(gtimer_0_23) :: "memory");
        phase_timestamps[13] = gtimer_0_23;
    }
    if (bid == 0) {
        int ack_owner_i = warp;
        if (ack_owner_i < world_size) {
            int ack_owner = ack_owner_i + rank + 1;
            if (ack_owner >= world_size) {
                ack_owner = ack_owner - world_size;
            }
            if (elect_sync()) {
                // gin_put_signal_add: strong remote completion on context 0
                {
                    ncclGin __gin{*(gin_dev_comm), (int)(0)};
                    __gin.put(ncclTeamWorld(*(gin_dev_comm)), (int)(ack_owner), ack_inbox_window, (size_t)(rank), ack_out_window, (size_t)(ack_owner), (size_t)(1),
                        ncclGin_StrongSignalAdd{(ncclGinSignal_t)(8 + rank), (uint64_t)(1)}, ncclGin_None{}, ncclCoopThread());
                }
            }
        }
        if (warp == 0) {
            if (elect_sync()) {
                #pragma unroll 1
                for (int h8_s = 0; h8_s < 4; h8_s++) {
                    if (h8_s < world_size) {
                        signal_base_scratch[h8_s] = signal_base_scratch[h8_s] + 1;
                        int _max_98 = ((source_record_counts[h8_s]) > (0) ? (source_record_counts[h8_s]) : (0));
                        int _min_62 = ((_max_98) < (8192) ? (_max_98) : (8192));
                        int h8_count = _min_62;
                        int _max_99 = (((h8_count + 8 - 1) / 8) > (dispatch_chunk_min_records) ? ((h8_count + 8 - 1) / 8) : (dispatch_chunk_min_records));
                        int h8_q = _max_99;
                        int h8_chunks = (h8_count + h8_q - 1) / h8_q;
                        #pragma unroll 1
                        for (int h8_c = 0; h8_c < 8; h8_c++) {
                            if (h8_chunks > h8_c) {
                                dispatch_chunk_signal_base_scratch[h8_s * 8 + h8_c] = dispatch_chunk_signal_base_scratch[h8_s * 8 + h8_c] + 1;
                            }
                        }
                        if (h8_s != rank) {
                            int _max_100 = ((owner_route_counts[h8_s]) > (0) ? (owner_route_counts[h8_s]) : (0));
                            int _min_63 = ((_max_100) < (49152) ? (_max_100) : (49152));
                            int h8_routes = _min_63;
                            int h8_rq = 128;
                            if (h8_routes >= 6144) {
                                h8_rq = 64;
                            }
                            if (h8_routes < 768) {
                                h8_rq = 32;
                            }
                            int _max_101 = (((h8_routes + h8_rq - 1) / h8_rq - 1) > (0) ? ((h8_routes + h8_rq - 1) / h8_rq - 1) : (0));
                            int h8_full = _max_101;
                            int _max_102 = ((h8_full + (h8_routes - h8_full * h8_rq + 64 - 1) / 64) > (1) ? (h8_full + (h8_routes - h8_full * h8_rq + 64 - 1) / 64) : (1));
                            int h8_owner_chunks = _max_102 + 1;
                            int h8_nchunks = h8_owner_chunks - 1;
                            int _max_103 = ((source_route_counts[h8_s]) > (0) ? (source_route_counts[h8_s]) : (0));
                            int _min_64 = ((_max_103) < (49152) ? (_max_103) : (49152));
                            if (h8_routes * 8 >= active_rows * 18 || h8_routes * 8 >= active_rows * 15 && _min_64 * 8 >= active_rows * 18) {
                                result_signal_base_scratch[h8_s] = result_signal_base_scratch[h8_s] + (unsigned long long)(1 + (h8_nchunks + 1) / 2);
                                result_signal_base_scratch[4 + h8_s] = result_signal_base_scratch[4 + h8_s] + (unsigned long long)(h8_nchunks / 2);
                            } else {
                                result_signal_base_scratch[h8_s] = result_signal_base_scratch[h8_s] + (unsigned long long)h8_owner_chunks;
                            }
                        }
                    }
                }
            }
        }
    }
    cooperative_groups::this_grid().sync();
    if (bid == 0 && tid == 0) {
        unsigned long long gtimer_0_24;
        asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(gtimer_0_24) :: "memory");
        phase_timestamps[14] = gtimer_0_24;
    }

    // Cleanup
    __syncthreads();
}

} // extern "C"
#undef DG_SM120_SHARED_TB_SCR_STAGE_BYTES
