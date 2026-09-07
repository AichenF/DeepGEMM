// TP-specific SM90 MXFP4 MegaMoE tile-warp-specialized kernel body.
//
// The persistent microtask is (expert-local routed BM8 block, intermediate
// K128 slice).  It executes W13 -> SwiGLU/FP8 -> all sixteen W2 N256 tiles
// before releasing the CTA-local FP8 activation.  Only FP32 W2 K-slice
// partials cross CTA boundaries through global memory.
#if (defined(__CUDA_ARCH__) and (__CUDA_ARCH__ >= 900) and (__CUDA_ARCH__ < 1000)) or defined(__CLION_IDE__)
    using Barrier = cutlass::arch::ClusterTransactionBarrier;

    // =====================================================================
    // Template checks
    // =====================================================================
    DG_STATIC_ASSERT(K_NATIVE_TP_TILE_WS,
                     "tile-WS body requires its isolated JIT selector");
    DG_STATIC_ASSERT(K_NATIVE_REGISTER_DEQUANT &&
                     K_NATIVE_RS_K128_BATCH &&
                     !K_NATIVE_TWO_CTA_PER_SM &&
                     K_NATIVE_NORMALIZED_WEIGHT_SCALE &&
                     K_NATIVE_DUAL_ACTIVE_DISPATCH &&
                     K_NATIVE_TP_LOCAL_ROUTE_BUILD &&
                     K_NATIVE_TP_LOCAL_BARRIER_FASTPATH,
                     "tile-WS body requires the qualified TP-local bundle");
    DG_STATIC_ASSERT(!K_NATIVE_H20_EXACT_OUTER &&
                     !K_NATIVE_SPLIT_WEIGHT_SCALE_TMA &&
                     !K_NATIVE_TILE_WEIGHT_SCALE_TMA,
                     "tile-WS v1 supports only fused normalized weight rows");
    DG_STATIC_ASSERT(BLOCK_M == 8 || BLOCK_M == 16 ||
                     BLOCK_M == 24 || BLOCK_M == 64 || BLOCK_M == 128,
                     "H200 fused kernel requires BM8/BM16/BM24/BM64/BM128");
    DG_STATIC_ASSERT((BLOCK_M == 8 && kNumStages == 4) ||
                     ((BLOCK_M == 16 || BLOCK_M == 24) && kNumStages == 3) ||
                     (BLOCK_M == 64 && kNumStages == 3) ||
                     (BLOCK_M == 128 && kNumStages == 6),
                     "Unexpected H200 pipeline depth");
    DG_STATIC_ASSERT((BLOCK_M == 128) == (BLOCK_N == 128),
                     "BM128 is paired with the BN128 split-M topology");
    DG_STATIC_ASSERT(!kSwapABRequested || BLOCK_M <= 24,
                     "swap-AB is only selected through the M64 bucket");

    // =====================================================================
    // Thread / warp identification
    // =====================================================================
    const uint32_t sm_idx     = blockIdx.x;
    const uint32_t thread_idx = threadIdx.x;
    const uint32_t warp_idx   = cutlass::canonical_warp_idx_sync();
    const uint32_t lane_idx   = ptx::get_lane_idx();

    if constexpr (K_NATIVE_PHASE_STAMPS) {
        if (cumulative_local_expert_recv_stats != nullptr &&
                sm_idx == 0 && thread_idx == 0) {
            native_write_phase_stamp(cumulative_local_expert_recv_stats, 0);
        }
    }

    if (warp_idx == 0 and cute::elect_one_sync()) {
        cute::prefetch_tma_descriptor(&tensor_map_l1_acts);
        cute::prefetch_tma_descriptor(&tensor_map_l1_acts_sf);
        cute::prefetch_tma_descriptor(&tensor_map_l1_weights);
        if constexpr (K_NATIVE_SPLIT_WEIGHT_SCALE_TMA)
            cute::prefetch_tma_descriptor(&tensor_map_l1_weight_scales);
        cute::prefetch_tma_descriptor(&tensor_map_l1_output);
        cute::prefetch_tma_descriptor(&tensor_map_l2_acts);
        cute::prefetch_tma_descriptor(&tensor_map_l2_acts_sf);
        cute::prefetch_tma_descriptor(&tensor_map_l2_weights);
        if constexpr (K_NATIVE_SPLIT_WEIGHT_SCALE_TMA)
            cute::prefetch_tma_descriptor(&tensor_map_l2_weight_scales);
    }

    // =====================================================================
    // Workspaces and symmetric buffer slicing. The framework reserves
    // per-64 SF capacity; this per-128 path uses its first half
    // as a dense per-128 layout so no framework allocation change is needed.
    // =====================================================================
    const auto workspace = layout::Workspace(
        sym_buffer.get_base_ptr(), kNumRanks, kNumExperts, kNumMaxTokensPerRank, kNumTopk);

    constexpr auto fp8_token_layout              = layout::Data(kHidden);
    constexpr auto bf16_token_layout             = layout::Data(kHidden * sizeof(nv_bfloat16));
    constexpr auto fp8_intermediate_token_layout = layout::Data(kIntermediateHidden);
    // Per-128 K float SF: 4 bytes per per-128 group => `kHidden / 32` bytes/token (same as SM100 packing)
    constexpr auto fp8_sf_layout                 = layout::Data(kHidden / 32, false);
    // Physical per-64 capacity: logical per-128 scales occupy the first half.
    constexpr auto fp8_intermediate_sf_layout    = layout::Data(kIntermediateHidden / 16);
    constexpr auto input_topk_idx_layout         = layout::Data(kNumTopk * sizeof(int64_t), false);
    constexpr auto input_topk_weights_layout     = layout::Data(kNumTopk * sizeof(float), false);
    constexpr auto l1_topk_weights_layout        = layout::Data(sizeof(float), false);

    // Registered input area
    const auto input_token_buffer        = layout::Buffer(fp8_token_layout, 1, kNumMaxTokensPerRank, workspace.get_end_ptr());
    const auto input_sf_buffer           = layout::Buffer(fp8_sf_layout, 1, kNumMaxTokensPerRank, input_token_buffer.get_end_ptr());
    const auto input_topk_idx_buffer     = layout::Buffer(input_topk_idx_layout, 1, kNumMaxTokensPerRank, input_sf_buffer.get_end_ptr());
    const auto input_topk_weights_buffer = layout::Buffer(input_topk_weights_layout, 1, kNumMaxTokensPerRank, input_topk_idx_buffer.get_end_ptr());

    // L1 input area
    const auto l1_token_buffer        = layout::Buffer(fp8_token_layout, 1, kNumMaxPoolTokens, input_topk_weights_buffer.get_end_ptr());
    const auto l1_sf_buffer           = layout::Buffer(fp8_sf_layout, 1, kNumPaddedSFPoolTokens, l1_token_buffer.get_end_ptr());
    const auto l1_topk_weights_buffer = layout::Buffer(l1_topk_weights_layout, 1, kNumMaxPoolTokens, l1_sf_buffer.get_end_ptr());

    // L2 input area
    const auto l2_token_buffer = layout::Buffer(fp8_intermediate_token_layout, 1, kNumMaxPoolTokens, l1_topk_weights_buffer.get_end_ptr());
    const auto l2_sf_buffer    = layout::Buffer(fp8_intermediate_sf_layout, 1, kNumPaddedSFPoolTokens, l2_token_buffer.get_end_ptr());

    // Combine input area
    const auto combine_token_buffer = layout::Buffer(bf16_token_layout, kNumTopk, kNumMaxTokensPerRank, l2_sf_buffer.get_end_ptr());

    // =====================================================================
    // GEMM data types and shape constants
    // =====================================================================
    using a_dtype_t = cutlass::float_e4m3_t;
    using b_dtype_t = cutlass::float_e4m3_t;
    using task_info_t = sched::TaskInfo;
    using interleaved_scheduler_t = sched::InterleavedMegaMoEScheduler<
        BLOCK_M, BLOCK_N, BLOCK_K,
        L1_SHAPE_N, L1_SHAPE_K,
        L2_SHAPE_N, L2_SHAPE_K,
        kNumExpertsPerRank, kNumSMs, kNumRanks>;
    constexpr uint32_t kNumRoutedL1BlockNs = L1_SHAPE_N / BLOCK_N;
    constexpr uint32_t kNumIntermediateSlices =
        kIntermediateHidden / BLOCK_K;
    constexpr uint32_t kNumW2OutputTiles = L2_SHAPE_N / BLOCK_N;
    DG_STATIC_ASSERT(kNumRoutedL1BlockNs == kNumIntermediateSlices,
                     "one W13 N256 tile must produce one W2 K128 slice");
    DG_STATIC_ASSERT(kNumW2OutputTiles == 16,
                     "V4 Flash hidden4096 requires sixteen W2 N256 tiles");
    constexpr bool kSplitMDecodedWeightReuse =
        BLOCK_M == 128 && BLOCK_N == 128 && kNumEpilogueWarpgroups == 2;
    constexpr uint32_t WG_BLOCK_M =
        kSplitMDecodedWeightReuse ? BLOCK_M / 2 : BLOCK_M;
    constexpr uint32_t WG_BLOCK_N =
        kSplitMDecodedWeightReuse ? BLOCK_N : BLOCK_N / 2;
    constexpr uint32_t L1_OUT_BLOCK_N = BLOCK_N / 2;       // post-SwiGLU tile N
    constexpr uint32_t WG_L1_OUT_BLOCK_N = WG_BLOCK_N / 2; // post-SwiGLU per-WG N
    constexpr uint32_t kSwapABTokenChunks = BLOCK_M / 8;
    constexpr uint32_t kSwapABWeightHalves = WG_BLOCK_N / 64;
    constexpr uint32_t kSwapABHalfAccumPerThread = 64 * 64 / 128;
    DG_STATIC_ASSERT(!kSwapABRequested || WG_L1_OUT_BLOCK_N == 64,
                     "swapAB expects BN256 split-N with 64 L1 output columns per WG");
    // Both dispatch warps participate in CTA-wide barriers. Selected plans may
    // use one warp for routing and token pulls, leaving the other warp's send
    // buffer available for an additional GEMM stage.
    constexpr uint32_t kNumActiveDispatchWarps =
        kSingleActiveDispatchWarp ? 1u : kNumDispatchWarps;
    constexpr uint32_t kNumActiveDispatchThreads = kNumActiveDispatchWarps * 32;
    constexpr bool kQuadDequantIlp =
        BLOCK_M == 8 && kNumStages == 4;
    using L1WGMMA = typename mma::sm90::FP8MMASelector<WG_BLOCK_N>::type;
    static_assert(L1WGMMA::M == 64 and L1WGMMA::N == WG_BLOCK_N and L1WGMMA::K == 32,
                  "Unexpected WGMMA shape");
    // A and B are CTA-local in the fixed cluster-size-one plan.
    constexpr uint32_t LOAD_BLOCK_M    = BLOCK_M;
    constexpr uint32_t LOAD_BLOCK_N    = BLOCK_N;
    constexpr uint32_t kSwizzleAMode   = BLOCK_K * sizeof(a_dtype_t);   // 128
    constexpr uint32_t kL2ActsSFGranK =
        kSplitMDecodedWeightReuse ? 64u : 128u;
    DG_STATIC_ASSERT(kSplitMDecodedWeightReuse ||
                     WG_L1_OUT_BLOCK_N < kL2ActsSFGranK,
                     "split-N warpgroups must share one L2 activation scale");

    // =====================================================================
    // Shared memory layout
    // =====================================================================
    constexpr uint32_t kSharedMemoryAlignment = 1024;
    extern __shared__ __align__(kSharedMemoryAlignment) uint8_t smem_buffer[];

    constexpr uint32_t SMEM_EXPERT_COUNT_SIZE =
        math::constexpr_align<uint32_t>(kNumExperts * sizeof(uint32_t), kSharedMemoryAlignment);
    constexpr uint32_t SMEM_SEND_BUFFER_SIZE =
        math::constexpr_align(fp8_token_layout.get_num_bytes() * kNumActiveDispatchWarps, kSharedMemoryAlignment);
    // E8M0 fold LUT: a 128-row window over codes [72,199] (see mxfp4_dequant.cuh).
    // Entries outside the window are constant, so clamping keeps this at 1KB.
    constexpr uint32_t SMEM_MXFP4_LUT_SIZE =
        math::constexpr_align<uint32_t>(
            deep_gemm::mxfp4::kE8M0LutCount * sizeof(uint2), kSharedMemoryAlignment);
    constexpr uint32_t SMEM_A_SIZE_PER_STAGE = LOAD_BLOCK_M * BLOCK_K * sizeof(a_dtype_t);
    constexpr uint32_t SMEM_B_SIZE_PER_STAGE = LOAD_BLOCK_N * BLOCK_K * sizeof(b_dtype_t);
    // BM128 split-M alternates two decoded-B slots. This lets one WG begin
    // decoding K+1 after its K WGMMA completes without overwriting the slot
    // that the paired WG may still be consuming.
    constexpr bool kRegisterDequant = K_NATIVE_REGISTER_DEQUANT;
    DG_STATIC_ASSERT(!kRegisterDequant ||
                     (BLOCK_M == 8 && BLOCK_N == 256 &&
                      kSwapABRequested && kUseMode2RowDecoder &&
                      !kSplitMDecodedWeightReuse),
                     "register dequant currently supports BM8/BN256 swap-AB");
    DG_STATIC_ASSERT(!((K_NATIVE_SPLIT_WEIGHT_SCALE_TMA ||
                        K_NATIVE_TILE_WEIGHT_SCALE_TMA) &&
                       K_NATIVE_RS_SCALE_WORD_CACHE),
                     "compact scale TMA is incompatible with scale-word cache");
    DG_STATIC_ASSERT(!(K_NATIVE_SPLIT_WEIGHT_SCALE_TMA &&
                       K_NATIVE_TILE_WEIGHT_SCALE_TMA),
                     "split and tile TMA modes are exclusive");
    constexpr uint32_t kNumDecodedBStages = kRegisterDequant ? 0u :
        (kSplitMDecodedWeightReuse ? 2u : kNumStages);
    constexpr bool kCompactWeightScaleTma =
        K_NATIVE_SPLIT_WEIGHT_SCALE_TMA ||
        K_NATIVE_TILE_WEIGHT_SCALE_TMA;
    constexpr uint32_t B_LOAD_BYTES_PER_ROW =
        kCompactWeightScaleTma ? 64u : 80u;
    constexpr uint32_t SMEM_PACKED_B_SIZE_PER_STAGE =
        LOAD_BLOCK_N * B_LOAD_BYTES_PER_ROW * sizeof(b_dtype_t);
    constexpr uint32_t SMEM_B_SCALE_SIZE_PER_STAGE =
        kCompactWeightScaleTma ? LOAD_BLOCK_N * 4u : 0u;
    constexpr uint32_t SMEM_PACKED_B_STAGE_SIZE =
        SMEM_PACKED_B_SIZE_PER_STAGE + SMEM_B_SCALE_SIZE_PER_STAGE;
    // L1 and L2 each consume one per-128 activation scale per row and K tile.
    constexpr uint32_t kL2SFAHalfStride =
        math::constexpr_align<uint32_t>(BLOCK_M * sizeof(float), 128u) / sizeof(float);
    constexpr uint32_t kNumL2SFAGroups =
        kSplitMDecodedWeightReuse ? 2u : 1u;
    constexpr uint32_t SMEM_SFA_SIZE_PER_STAGE =
        kNumL2SFAGroups * kL2SFAHalfStride * sizeof(float);
    // CD output: max of L1 FP8 (BLOCK_M * (BLOCK_N/2) * 1 byte * num_wg) and
    // L2 BF16 (BLOCK_M * BLOCK_N * 2 bytes * num_wg).
    constexpr uint32_t SMEM_CD_L1_SIZE =
        kNumEpilogueWarpgroups * WG_BLOCK_M * WG_L1_OUT_BLOCK_N * sizeof(cutlass::float_e4m3_t);
    constexpr uint32_t SMEM_CD_L2_SIZE = kSwapABRequested ?
        BLOCK_M * BLOCK_N * sizeof(nv_bfloat16) : 0u;
    constexpr uint32_t SMEM_CD_OUTPUT_BASE_SIZE =
        SMEM_CD_L1_SIZE > SMEM_CD_L2_SIZE ? SMEM_CD_L1_SIZE : SMEM_CD_L2_SIZE;
    constexpr uint32_t SMEM_CD_L1_SHARED_SF_SLOTS =
        kNumEpilogueWarpgroups * BLOCK_M;
    constexpr uint32_t SMEM_CD_L1_SWAP_AMAX_SLOTS = kSwapABRequested ?
        BLOCK_M * kNumEpilogueWarps : 0u;
    constexpr uint32_t SMEM_CD_L1_EXTRA_FLOAT_SLOTS =
        SMEM_CD_L1_SHARED_SF_SLOTS > SMEM_CD_L1_SWAP_AMAX_SLOTS ?
        SMEM_CD_L1_SHARED_SF_SLOTS : SMEM_CD_L1_SWAP_AMAX_SLOTS;
    constexpr uint32_t SMEM_CD_L1_SHARED_SF_SIZE =
        SMEM_CD_L1_EXTRA_FLOAT_SLOTS * sizeof(float);
    constexpr uint32_t SMEM_CD_OUTPUT_UNALIGNED_SIZE =
        SMEM_CD_OUTPUT_BASE_SIZE + SMEM_CD_L1_SHARED_SF_SIZE;
    constexpr uint32_t SMEM_CD_SIZE = math::constexpr_align(
        SMEM_CD_OUTPUT_UNALIGNED_SIZE, kSharedMemoryAlignment);

    constexpr uint32_t SMEM_GEMM_STORAGE_SIZE =
        SMEM_EXPERT_COUNT_SIZE + SMEM_SEND_BUFFER_SIZE + SMEM_MXFP4_LUT_SIZE + SMEM_CD_SIZE +
        kNumStages * (SMEM_A_SIZE_PER_STAGE + SMEM_PACKED_B_STAGE_SIZE) +
        kNumDecodedBStages * SMEM_B_SIZE_PER_STAGE;
    // The post-GEMM top-k combine reuses the prefix as three buffers over two
    // hidden chunks.  Decoded-B used to make that prefix large implicitly;
    // keep the alias contract explicit when register dequant removes it.
    constexpr uint32_t SMEM_COMBINE_ALIAS_SIZE =
        3u * kNumEpilogueWarps * kHidden * sizeof(nv_bfloat16) / 2u;
    constexpr uint32_t SMEM_BEFORE_BARRIER_SIZE =
        SMEM_GEMM_STORAGE_SIZE > SMEM_COMBINE_ALIAS_SIZE
        ? SMEM_GEMM_STORAGE_SIZE : SMEM_COMBINE_ALIAS_SIZE;

    // SMEM pointers
    auto smem_expert_count = reinterpret_cast<uint32_t*>(smem_buffer);
    const auto smem_send_buffers = layout::Buffer(
        fp8_token_layout, kNumActiveDispatchWarps, 1,
        math::advance_ptr(smem_buffer, SMEM_EXPERT_COUNT_SIZE));
    auto smem_mxfp4_lut = reinterpret_cast<uint2*>(math::advance_ptr<uint8_t>(
        smem_buffer, SMEM_EXPERT_COUNT_SIZE + SMEM_SEND_BUFFER_SIZE));

    auto smem_gemm_base = math::advance_ptr(
        smem_buffer, SMEM_EXPERT_COUNT_SIZE + SMEM_SEND_BUFFER_SIZE + SMEM_MXFP4_LUT_SIZE);

    auto smem_cd_base = smem_gemm_base;
    // CD output is shared by L1 (FP8) and L2 (BF16); reinterpret-cast as needed.
    auto smem_cd_l1 = reinterpret_cast<cutlass::float_e4m3_t*>(smem_cd_base);
    auto smem_cd_l1_shared_sf =
        math::advance_ptr<float>(smem_cd_base, SMEM_CD_OUTPUT_BASE_SIZE);
    auto smem_cd_l2 = reinterpret_cast<nv_bfloat16*>(smem_cd_base);

    auto smem_a = utils::PatternVisitor([=](const uint32_t& i) {
        return math::advance_ptr<a_dtype_t>(smem_gemm_base, SMEM_CD_SIZE + i * SMEM_A_SIZE_PER_STAGE);
    });
    auto smem_b = utils::PatternVisitor([=](const uint32_t& i) {
        const uint32_t decoded_stage =
            kSplitMDecodedWeightReuse ? (i & 1u) : i;
        return math::advance_ptr<b_dtype_t>(
            smem_gemm_base,
            SMEM_CD_SIZE + kNumStages * SMEM_A_SIZE_PER_STAGE +
            decoded_stage * SMEM_B_SIZE_PER_STAGE);
    });
    auto smem_packed_b = utils::PatternVisitor([=](const uint32_t& i) {
        return math::advance_ptr<b_dtype_t>(
            smem_gemm_base, SMEM_CD_SIZE + kNumStages * SMEM_A_SIZE_PER_STAGE +
            kNumDecodedBStages * SMEM_B_SIZE_PER_STAGE +
            i * SMEM_PACKED_B_STAGE_SIZE);
    });
    auto smem_b_scale = utils::PatternVisitor([=](const uint32_t& i) {
        return math::advance_ptr<uint8_t>(
            smem_packed_b[i], SMEM_PACKED_B_SIZE_PER_STAGE);
    });
    auto sf_start_ptr = math::advance_ptr<uint8_t>(
        smem_buffer, SMEM_BEFORE_BARRIER_SIZE);
    auto smem_sfa = utils::PatternVisitor([=](const uint32_t& i) {
        return reinterpret_cast<float*>(sf_start_ptr + i * SMEM_SFA_SIZE_PER_STAGE);
    });
    // Barriers live after SF.
    auto barrier_start_ptr = reinterpret_cast<Barrier*>(
        sf_start_ptr + kNumStages * SMEM_SFA_SIZE_PER_STAGE);
    auto dispatch_barriers = utils::PatternVisitor([=](const uint32_t& i) { return barrier_start_ptr + i; });
    auto full_barriers     = utils::PatternVisitor([=](const uint32_t& i) { return barrier_start_ptr + kNumDispatchWarps + i; });
    auto empty_barriers    = utils::PatternVisitor([=](const uint32_t& i) { return barrier_start_ptr + kNumDispatchWarps + kNumStages + i; });
    auto combine_barriers  = utils::PatternVisitor([=](const uint32_t& i) { return barrier_start_ptr + kNumDispatchWarps + kNumStages * 2 + i; });
    constexpr uint32_t kNumBaseBarriers =
        kNumDispatchWarps + kNumStages * 2 + kNumEpilogueWarps * 2;
    auto task_info_full_barriers = barrier_start_ptr + kNumBaseBarriers;
    auto task_info_empty_barriers = task_info_full_barriers +
        interleaved_scheduler_t::kNumScheduleStages;
    auto task_infos = reinterpret_cast<task_info_t*>(
        task_info_empty_barriers +
        interleaved_scheduler_t::kNumScheduleStages);
    constexpr uint32_t kInterleavedSchedulerSMEMBytes =
        2 * interleaved_scheduler_t::kNumScheduleStages * sizeof(Barrier) +
        interleaved_scheduler_t::kNumScheduleStages * sizeof(task_info_t);
    DG_STATIC_ASSERT(
        kInterleavedSchedulerSMEMBytes ==
            layout::kSM90InterleavedSchedulerSMEMBytes,
        "Host and device scheduler shared-memory layouts disagree");
    constexpr uint32_t kInterleavedSMEMEnd =
        SMEM_BEFORE_BARRIER_SIZE + kNumStages * SMEM_SFA_SIZE_PER_STAGE +
        kNumBaseBarriers * sizeof(Barrier) +
        kInterleavedSchedulerSMEMBytes;
    DG_STATIC_ASSERT(!kUseInterleavedScheduler || kInterleavedSMEMEnd <= 232448,
                     "Interleaved scheduler exceeds the SM90 shared-memory capacity");
    DG_STATIC_ASSERT(!kRegisterDequant || kInterleavedSMEMEnd <= 102400,
                     "register-dequant shared-memory budget exceeded");

    // =====================================================================
    // Initialization
    // =====================================================================
    // Build the 128-row E8M0 -> FP8 fold LUT window arithmetically (row i holds
    // the entry for E8M0 code kE8M0LutBase + i); one row per thread.
    if constexpr (!K_NATIVE_NORMALIZED_WEIGHT_SCALE) {
        if (thread_idx < deep_gemm::mxfp4::kE8M0LutCount) {
            smem_mxfp4_lut[thread_idx] =
                deep_gemm::mxfp4::load_e2m1_e8m0_lut(
                    thread_idx + deep_gemm::mxfp4::kE8M0LutBase);
        }
    }

    if (warp_idx == 0) {
        // Clean expert-count shared memory
        #pragma unroll
        for (uint32_t i = lane_idx; i < kNumExperts; i += 32)
            ptx::st_shared(smem_expert_count + i, 0u);
    } else if (warp_idx == 1) {
        // Init dispatch m-barriers
        #pragma unroll
        for (uint32_t i = lane_idx; i < kNumDispatchWarps; i += 32)
            dispatch_barriers[i]->init(1);
        cutlass::arch::fence_barrier_init();
    } else if (warp_idx == 2) {
        // Init GEMM full/empty barriers and combine barriers
        if (cute::elect_one_sync()) {
            #pragma unroll
            for (uint32_t i = 0; i < kNumStages; ++ i) {
                // Producer arrivals: A(+SFA) + B(TMA+SFB). SFB is copied with
                // cp.async.bulk and counted as B-loader transaction bytes, so
                // it does not need a separate producer arrival.
                full_barriers[i]->init(2);
                empty_barriers[i]->init(kNumEpilogueWarps);
            }
            #pragma unroll
            for (uint32_t i = 0; i < kNumEpilogueWarps * 2; ++ i)
                combine_barriers[i]->init(1);
            if constexpr (kUseInterleavedScheduler) {
                #pragma unroll
                for (uint32_t i = 0;
                     i < interleaved_scheduler_t::kNumScheduleStages;
                     ++ i) {
                    task_info_full_barriers[i].init(1);
                    task_info_empty_barriers[i].init(kNumEpilogueWarps);
                }
            }
        }
        cutlass::arch::fence_barrier_init();
    }
    __syncthreads();

    // =====================================================================
    // Scheduler (cluster=1)
    // =====================================================================
    auto scheduler = sched::MegaMoEScheduler<
        BLOCK_M, BLOCK_N, BLOCK_K,
        L1_SHAPE_N, L1_SHAPE_K,
        L2_SHAPE_N, L2_SHAPE_K,
        kNumExpertsPerRank, kNumExpertsPerWave,
        kNumSMs, kNumRanks, 1>(workspace);
    auto interleaved_scheduler = interleaved_scheduler_t(
        workspace,
        task_info_full_barriers,
        task_info_empty_barriers,
        task_infos);

    // Pipeline state shared by TMA loaders and math warpgroups
    uint32_t stage_idx = 0, phase = 0;
    auto advance_pipeline = [&](uint32_t& k_block_idx) {
        ++ k_block_idx;
        stage_idx = stage_idx == kNumStages - 1 ? 0 : stage_idx + 1;
        phase ^= stage_idx == 0;
    };
    // Intra-SM barrier indices (mirroring SM100)
    constexpr uint32_t kDispatchBarrierIdx              = 0;
    constexpr uint32_t kDispatchWithEpilogueBarrierIdx  = 1;
    constexpr uint32_t kEpilogueFullBarrierIdx          = 2;
    constexpr uint32_t kEpilogueWGBarrierStartIdx       = 3;
    constexpr uint32_t kSplitMDecodeBarrierIdx          = 8;

    // Cross-rank NVLink barrier tags
    constexpr uint32_t kBeforeDispatchPullBarrierTag    = 1;
    constexpr uint32_t kBeforeCombineReduceBarrierTag   = 2;
    constexpr uint32_t kAfterWorkspaceCleanBarrierTag   = 3;

    // Register reconfiguration counts (chosen to fit in 64512 reg budget).
    constexpr uint32_t kNumDispatchRegisters    = 48;
    constexpr uint32_t kNumNonEpilogueRegisters =
        kUseInterleavedScheduler ? 64 : 40;
    // Register-dequant reduces shared memory enough for two resident CTAs.
    // 88 registers per math lane also stays below the cubin's 80-reg/thread
    // initial CTA allocation after producer warps deallocate; requesting 96
    // would need 1,024 registers from outside the CTA and can deadlock.
    // The default reference retains 208.
    constexpr uint32_t kNumEpilogueRegisters =
        K_NATIVE_TWO_CTA_PER_SM ? 88 : 208;
    DG_STATIC_ASSERT(kNumDispatchRegisters * kNumDispatchThreads +
                     kNumNonEpilogueRegisters * kNumNonEpilogueThreads +
                     kNumEpilogueRegisters * kNumEpilogueThreads <= 64512,
                     "Too many registers");

    constexpr uint32_t kDispatchGridSyncIndex = 0;
    constexpr uint32_t kEpilogueGridSyncIndex = 1;

    const auto for_each_static_selected_block = [&](auto&& func) {
        scheduler.fetch_expert_recv_count();
        scheduler.set_expert_idx(0);
        while (true) {
            CUTE_TIE_DECL(scheduler.get_next_block(),
                          block_phase, local_expert_idx, m_block_idx, n_block_idx);
            if (block_phase == sched::BlockPhase::None)
                break;
            if (block_phase == sched::BlockPhase::Linear1) {
                func(std::integral_constant<sched::BlockPhase, sched::BlockPhase::Linear1>{},
                     local_expert_idx, L1_SHAPE_K / BLOCK_K, m_block_idx, n_block_idx,
                     scheduler.get_current_pool_block_offset() + m_block_idx,
                     scheduler.template get_valid_m<false>());
            } else {
                func(std::integral_constant<sched::BlockPhase, sched::BlockPhase::Linear2>{},
                     local_expert_idx, L2_SHAPE_K / BLOCK_K, m_block_idx, n_block_idx,
                     scheduler.get_current_pool_block_offset() + m_block_idx,
                     scheduler.template get_valid_m<false>());
            }
        }
    };

    const auto invoke_interleaved_task = [&](const task_info_t& task_info,
                                              auto&& func) {
        // Tile-WS publishes only fused route/slice tasks.  Reuse TaskInfo's
        // Linear1 shape to preserve the proven expert/pool-block mapping;
        // n_block_idx is the local intermediate K128 slice.
        func(task_info.local_expert_idx,
             task_info.m_block_idx, task_info.n_block_idx,
             task_info.pool_block_idx, task_info.valid_m);
    };

    const auto for_each_published_block = [&](auto&& func) {
        task_info_t task_info;
        while (interleaved_scheduler.get_published_task(task_info))
            invoke_interleaved_task(task_info, func);
    };

    const auto produce_interleaved_blocks = [&](auto&& func) {
        interleaved_scheduler.fetch_expert_recv_count();
        while (true) {
            interleaved_scheduler.wait_task_slot_empty();
            const uint32_t task_idx = interleaved_scheduler_t::get_next_task_idx(
                workspace.get_l1_task_count_ptr());
            const auto task_info =
                task_idx < interleaved_scheduler.num_total_m_blocks *
                               kNumIntermediateSlices
                ? interleaved_scheduler.create_task(
                      sched::BlockPhase::Linear1, task_idx,
                      kNumIntermediateSlices, L1_SHAPE_N, L1_SHAPE_K)
                : task_info_t();
            interleaved_scheduler.publish_task(task_info);
            if (!task_info.is_valid())
                break;
            invoke_interleaved_task(task_info, func);
        }
    };

    const auto cleanup_workspace = [&]() {
        DG_STATIC_ASSERT(kNumSMs > 1, "Invalid SM count");
        if (sm_idx == 0) {
            #pragma unroll
            for (uint32_t i = lane_idx; i < kNumExperts; i += 32)
                *workspace.get_expert_send_count_ptr(i) = 0;
            if constexpr (kUseInterleavedScheduler) {
                if (lane_idx == 0) {
                    *workspace.get_l1_task_count_ptr() = 0;
                    *workspace.get_l2_task_count_ptr() = 0;
                }
            }
        } else {
            // TP-local uses 256 experts on one rank, so a CTA revisits this
            // loop several times.  Keep cleanup on one warp: the original
            // cross-warp barrier scheme assumes the small EP expert count and
            // can alias barrier generations once a CTA owns 3-4 experts.
            for (uint32_t i = sm_idx - 1; i < kNumExpertsPerRank; i += kNumSMs - 1) {
                const auto num_recv_tokens = static_cast<uint32_t>(
                    *workspace.get_expert_recv_count_sum_ptr(i));
                const auto num_recv_m_blocks = math::ceil_div(num_recv_tokens, BLOCK_M);
                const auto cleanup_pool_block_offset = scheduler.get_pool_block_offset(i);

                if (lane_idx == 0)
                    *workspace.get_expert_recv_count_sum_ptr(i) = 0;

                for (uint32_t j = lane_idx; j < kNumRanks; j += 32)
                    *workspace.get_expert_recv_count_ptr(j, i) = 0;

                for (uint32_t j = lane_idx; j < num_recv_m_blocks; j += 32) {
                    *workspace.get_l1_arrival_count_ptr(cleanup_pool_block_offset + j) = 0;
                    *workspace.get_l2_arrival_mask_ptr(cleanup_pool_block_offset + j) = 0;
                }
                __syncwarp();
            }
        }
    };

    // =====================================================================
    // ROLE 1: DISPATCH WARPS
    //   Mirrors SM100 dispatch with two changes:
    //     * SF is per-128 channel float (no UTCCP transpose). We store the
    //       remote per-token SF directly into the local L1 SF buffer in
    //       MN-major layout: `local_sf[k_chunk * num_padded_sf_pool_tokens + token_idx]`.
    //     * The "token_idx_in_expert" → SF token index is now the simple
    //       per-block linear mapping (no 4×32 transpose).
    // =====================================================================
    if (warp_idx < kNumDispatchWarps) {
        cutlass::arch::warpgroup_reg_dealloc<kNumDispatchRegisters>();

        DG_STATIC_ASSERT(kNumTopk <= 32, "Invalid number of topk");
        constexpr uint32_t kNumActivateLanes = kNumTokensPerWarp * kNumTopk;
        const auto read_topk_idx = [&](const auto& process) {
            if (warp_idx < kNumActiveDispatchWarps) {
                #pragma unroll
                for (uint32_t i = (sm_idx * kNumActiveDispatchWarps + warp_idx) * kNumTokensPerWarp;
                     i < num_tokens;
                     i += kNumSMs * kNumActiveDispatchWarps * kNumTokensPerWarp) {
                    int expert_idx = -1;
                    if (i + (lane_idx / kNumTopk) < num_tokens and lane_idx < kNumActivateLanes) {
                        expert_idx = static_cast<int>(
                            __ldg(input_topk_idx_buffer.get_base_ptr<int64_t>() + i * kNumTopk + lane_idx));
                        if (expert_idx >= 0)
                            process(i * kNumTopk + lane_idx, expert_idx);
                    }
                    __syncwarp();
                }
            }
        };

        if constexpr (K_NATIVE_TP_LOCAL_ROUTE_BUILD) {
            // Pure TP has one logical routing rank.  Claim the final local
            // expert slot directly for each real route.  This replaces the
            // generic EP protocol's one 64-bit global atomic for every
            // (persistent CTA, expert) pair with one 32-bit atomic per route.
            read_topk_idx([&](const uint32_t& token_topk_idx,
                              const int& expert_idx) {
                auto* slot_count = reinterpret_cast<uint32_t*>(
                    workspace.get_expert_send_count_ptr(expert_idx));
                const uint32_t dst_slot_idx = atomicAdd(slot_count, 1u);
                *workspace.get_src_token_topk_idx_ptr(
                    expert_idx, 0, dst_slot_idx) = token_topk_idx;
            });
        } else {
            // Count tokens per expert
            read_topk_idx([&](const uint32_t& token_topk_idx,
                              const int& expert_idx) {
                atomicAdd_block(smem_expert_count + expert_idx, 1);
            });
            ptx::sync_aligned(kNumDispatchThreads, kDispatchBarrierIdx);

            // Stake out per-expert SM offsets via global atomic
            #pragma unroll
            for (uint32_t i = thread_idx; i < kNumExperts;
                 i += kNumDispatchThreads) {
                const uint64_t send_value =
                    (1ull << 32) |
                    static_cast<uint64_t>(smem_expert_count[i]);
                smem_expert_count[i] = static_cast<uint32_t>(ptx::atomic_add(
                    workspace.get_expert_send_count_ptr(i), send_value));
            }
            ptx::sync_aligned(kNumDispatchThreads, kDispatchBarrierIdx);

            // Write source token-topk indices to remote ranks
            read_topk_idx([&](const uint32_t& token_topk_idx,
                              const int& expert_idx) {
                const auto dst_rank_idx = expert_idx / kNumExpertsPerRank;
                const auto dst_slot_idx =
                    atomicAdd_block(smem_expert_count + expert_idx, 1);
                const auto dst_ptr = workspace.get_src_token_topk_idx_ptr(
                    expert_idx % kNumExpertsPerRank, sym_buffer.rank_idx,
                    dst_slot_idx);
                *sym_buffer.map(dst_ptr, dst_rank_idx) = token_topk_idx;
            });
        }

        comm::grid_sync<kNumSMs, kDispatchGridSyncIndex>(
            workspace, sm_idx, thread_idx,
            [=]() { ptx::sync_aligned(kNumDispatchThreads, kDispatchBarrierIdx); }
        );

        if (sm_idx == 0 and thread_idx < kNumActiveDispatchThreads) {
            #pragma unroll
            for (uint32_t i = thread_idx; i < kNumExperts; i += kNumActiveDispatchThreads) {
                const auto dst_rank_idx = i / kNumExpertsPerRank;
                const auto dst_local_expert_idx = i % kNumExpertsPerRank;
                if constexpr (K_NATIVE_TP_LOCAL_ROUTE_BUILD) {
                    const uint32_t expert_count =
                        *reinterpret_cast<const uint32_t*>(
                            workspace.get_expert_send_count_ptr(i));
                    *workspace.get_expert_recv_count_ptr(
                        0, dst_local_expert_idx) = expert_count;
                    *workspace.get_expert_recv_count_sum_ptr(
                        dst_local_expert_idx) =
                        (static_cast<uint64_t>(kNumSMs) << 32) |
                        expert_count;
                } else {
                    const auto expert_status =
                        *workspace.get_expert_send_count_ptr(i);
                    *sym_buffer.map(
                        workspace.get_expert_recv_count_ptr(
                            sym_buffer.rank_idx, dst_local_expert_idx),
                        dst_rank_idx) = expert_status & 0xffffffff;
                    ptx::atomic_add_sys(
                        sym_buffer.map(
                            workspace.get_expert_recv_count_sum_ptr(
                                dst_local_expert_idx),
                            dst_rank_idx),
                        expert_status);
                }
            }
        }
        ptx::sync_aligned(kNumDispatchThreads, kDispatchBarrierIdx);

        if constexpr (K_NATIVE_TP_LOCAL_BARRIER_FASTPATH) {
            // TP-local routing has one logical rank.  The only required
            // operation here is publication of SM0's finalized recv counts
            // to every persistent CTA; a local grid barrier is sufficient.
            comm::grid_sync<kNumSMs, kDispatchGridSyncIndex>(
                workspace, sm_idx, thread_idx,
                [=]() {
                    ptx::sync_aligned(
                        kNumDispatchThreads, kDispatchBarrierIdx);
                });
        } else {
            comm::nvlink_barrier<kNumRanks, kNumSMs, kNumDispatchThreads,
                                 kDispatchGridSyncIndex,
                                 kBeforeDispatchPullBarrierTag>(
                workspace, sym_buffer, sm_idx, thread_idx,
                [=]() {
                    ptx::sync_aligned(
                        kNumDispatchThreads, kDispatchBarrierIdx);
                },
                false, true);
        }

        if constexpr (K_NATIVE_PHASE_STAMPS) {
            if (cumulative_local_expert_recv_stats != nullptr &&
                    sm_idx == 0 && thread_idx == 0) {
                native_write_phase_stamp(
                    cumulative_local_expert_recv_stats, 1);
            }
        }

        // Sync with epilogue warps before pulling tokens
        ptx::sync_unaligned(kNumDispatchThreads + kNumEpilogueThreads, kDispatchWithEpilogueBarrierIdx);

        // Token / SF pull loop
        if (warp_idx < kNumActiveDispatchWarps) {
            uint32_t pull_mbarrier_phase = 0;
            const auto pull_buffer = smem_send_buffers.get_rank_buffer(warp_idx).get_data_buffer(0);
            const auto pull_mbarrier = dispatch_barriers[warp_idx];

            scheduler.fetch_expert_recv_count();

            constexpr uint32_t kNumRanksPerLane = math::constexpr_ceil_div(kNumRanks, 32u);
            int      current_expert_idx = -1;
            uint32_t stored_rank_count[kNumRanksPerLane] = {};
            uint32_t expert_start_idx = 0, expert_end_idx = 0;
            uint32_t expert_pool_block_offset = 0;

            constexpr uint32_t kNumGlobalWarps = kNumSMs * kNumActiveDispatchWarps;
            for (uint32_t token_idx = sm_idx * kNumActiveDispatchWarps + warp_idx; ; token_idx += kNumGlobalWarps) {
                int old_expert_idx = current_expert_idx;
                while (token_idx >= expert_end_idx) {
                    if (++ current_expert_idx >= kNumExpertsPerRank)
                        break;
                    expert_pool_block_offset += math::ceil_div(expert_end_idx - expert_start_idx, BLOCK_M);
                    expert_start_idx = expert_end_idx;
                    expert_end_idx += scheduler.get_num_tokens(current_expert_idx);
                }
                if (current_expert_idx >= kNumExpertsPerRank)
                    break;

                if constexpr (!K_NATIVE_TP_LOCAL_DISPATCH_FASTPATH) {
                  if (old_expert_idx != current_expert_idx) {
                    old_expert_idx = current_expert_idx;
                    #pragma unroll
                    for (uint32_t i = 0; i < kNumRanksPerLane; ++ i) {
                        const uint32_t j = i * 32 + lane_idx;
                        stored_rank_count[i] = j < kNumRanks ?
                            static_cast<uint32_t>(*workspace.get_expert_recv_count_ptr(j, current_expert_idx)) : 0;
                    }
                  }
                }

                // Pure TP has one local routing rank.  Keep the generic EP
                // round-robin as the control but bypass its two warp-wide
                // reductions per route in the TP-local fast path.
                uint32_t current_rank_in_expert_idx = 0;
                uint32_t remaining[kNumRanksPerLane];
                uint32_t token_idx_in_expert = token_idx - expert_start_idx;
                uint32_t token_idx_in_rank = token_idx_in_expert;
                if constexpr (!K_NATIVE_TP_LOCAL_DISPATCH_FASTPATH) {
                    #pragma unroll
                    for (uint32_t i = 0; i < kNumRanksPerLane; ++ i) {
                        remaining[i] = stored_rank_count[i];
                    }
                    uint32_t offset = 0;
                    uint32_t slot_idx = token_idx_in_expert;
                    while (true) {
                        uint32_t num_actives_in_lane = 0;
                        uint32_t min_in_lane = 0xffffffff;
                        #pragma unroll
                        for (uint32_t i = 0; i < kNumRanksPerLane; ++ i) {
                            num_actives_in_lane += remaining[i] > 0;
                            if (remaining[i] > 0)
                                min_in_lane = cute::min(min_in_lane, remaining[i]);
                        }
                        const uint32_t num_active_ranks =
                            __reduce_add_sync(0xffffffff, num_actives_in_lane);
                        const uint32_t length =
                            __reduce_min_sync(0xffffffff, min_in_lane);

                        const uint32_t num_round_tokens =
                            length * num_active_ranks;
                        if (slot_idx < num_round_tokens) {
                            const uint32_t slot_idx_in_round =
                                slot_idx % num_active_ranks;
                            uint32_t num_seen_ranks = 0;
                            current_rank_in_expert_idx = 0;
                            #pragma unroll
                            for (uint32_t i = 0;
                                 i < kNumRanksPerLane; ++ i) {
                                const uint32_t mask = __ballot_sync(
                                    0xffffffff, remaining[i] > 0);
                                const uint32_t num_active_lanes = __popc(mask);
                                if (slot_idx_in_round >= num_seen_ranks and
                                    slot_idx_in_round <
                                        num_seen_ranks + num_active_lanes) {
                                    current_rank_in_expert_idx =
                                        i * 32 + __fns(
                                            mask, 0,
                                            slot_idx_in_round -
                                                num_seen_ranks + 1);
                                }
                                num_seen_ranks += num_active_lanes;
                            }
                            token_idx_in_rank =
                                offset + (slot_idx / num_active_ranks);
                            break;
                        }
                        slot_idx -= num_round_tokens;
                        offset += length;
                        #pragma unroll
                        for (uint32_t i = 0;
                             i < kNumRanksPerLane; ++ i) {
                            remaining[i] -= cute::min(remaining[i], length);
                        }
                    }
                }

                const uint32_t src_token_topk_idx = *workspace.get_src_token_topk_idx_ptr(
                    current_expert_idx, current_rank_in_expert_idx, token_idx_in_rank);
                const uint32_t src_token_idx = src_token_topk_idx / kNumTopk;
                const uint32_t src_topk_idx  = src_token_topk_idx % kNumTopk;

                const uint32_t pool_token_idx =
                    expert_pool_block_offset * BLOCK_M + token_idx_in_expert;

                if constexpr (K_NATIVE_TP_LOCAL_DIRECT_COPY) {
                    constexpr uint32_t kTokenVecs = kHidden / sizeof(uint4);
                    const auto src = input_token_buffer
                        .get_data_buffer(src_token_idx)
                        .get_base_ptr<const uint4>();
                    auto dst = l1_token_buffer
                        .get_data_buffer(pool_token_idx)
                        .get_base_ptr<uint4>();
                    #pragma unroll
                    for (uint32_t j = lane_idx; j < kTokenVecs; j += 32)
                        dst[j] = src[j];
                    __syncwarp();
                } else {
                    // Generic EP path: TMA pull token data into SMEM.
                    if (cute::elect_one_sync()) {
                        ptx::tma_load_1d(
                            pull_buffer.get_base_ptr(),
                            sym_buffer.map(
                                input_token_buffer
                                    .get_data_buffer(src_token_idx)
                                    .get_base_ptr(),
                                current_rank_in_expert_idx),
                            pull_mbarrier, kHidden);
                    }
                    __syncwarp();
                }

                // Copy SF: per-128 K floats, written linearly (no UTCCP transpose).
                constexpr uint32_t kNumSFFloats = kHidden / 128;
                DG_STATIC_ASSERT(kNumSFFloats > 0 and kHidden % 128 == 0, "Invalid SF");
                const auto remote_sf_ptr = sym_buffer.map(
                    input_sf_buffer.get_data_buffer(src_token_idx).get_base_ptr<float>(),
                    current_rank_in_expert_idx);
                const auto local_sf_ptr  = l1_sf_buffer.get_base_ptr<float>();
                float route_weight_global_scale = 1.0f;
                if constexpr (K_NATIVE_FOLD_W13_GLOBAL_SCALE) {
                    if (lane_idx == 0) {
                        route_weight_global_scale =
                            __ldg(w13_global_scale + current_expert_idx);
                    }
                    route_weight_global_scale = __shfl_sync(
                        0xffffffffu, route_weight_global_scale, 0);
                }
                #pragma unroll
                for (uint32_t i = 0; i < math::constexpr_ceil_div(kNumSFFloats, 32u); ++ i) {
                    const uint32_t j = i * 32 + lane_idx;
                    if (j < kNumSFFloats)
                        local_sf_ptr[j * kNumPaddedSFPoolTokens + pool_token_idx] =
                            remote_sf_ptr[j] * route_weight_global_scale;
                }
                __syncwarp();

                if (cute::elect_one_sync()) {
                    const auto weight =
                        K_NATIVE_TP_LOCAL_DISPATCH_FASTPATH
                        ? input_topk_weights_buffer.get_base_ptr<float>()[
                              src_token_topk_idx]
                        : *sym_buffer.map(
                              input_topk_weights_buffer.get_base_ptr<float>()
                                  + src_token_topk_idx,
                              current_rank_in_expert_idx);
                    *l1_topk_weights_buffer.get_data_buffer(pool_token_idx).get_base_ptr<float>() = weight;

                    if constexpr (!K_NATIVE_TP_LOCAL_DIRECT_COPY) {
                        ptx::mbarrier_arrive_and_set_tx(
                            pull_mbarrier, kHidden);
                        ptx::mbarrier_wait_and_flip_phase(
                            pull_mbarrier, pull_mbarrier_phase);

                        ptx::tma_store_1d(
                            l1_token_buffer
                                .get_data_buffer(pool_token_idx)
                                .get_base_ptr(),
                            pull_buffer.get_base_ptr(),
                            pull_buffer.get_num_bytes());
                    }

                    *workspace.get_token_src_metadata_ptr(pool_token_idx) =
                        {current_rank_in_expert_idx, src_token_idx, src_topk_idx};

                    if constexpr (!K_NATIVE_TP_LOCAL_DIRECT_COPY) {
                        cute::tma_store_arrive();
                        ptx::tma_store_wait<0>();
                    }
                    ptx::red_add_rel(
                        workspace.get_l1_arrival_count_ptr(
                            expert_pool_block_offset + token_idx_in_expert / BLOCK_M),
                        1u);
                }
                __syncwarp();
            }
        }

        // Cleanup workspace, overlapping with combine
        ptx::sync_unaligned(kNumDispatchThreads + kNumEpilogueThreads, kDispatchWithEpilogueBarrierIdx);

        // TP routing is local (`kNumRanks == 1`), so cleanup needs a grid
        // rendezvous but no self-directed NVLink signal.  Keep both cleanup
        // and its grid-sync scope on warp 0; warp 1 has no writes after the
        // dispatch/epilogue rendezvous above.
        if (warp_idx == 0) {
            cleanup_workspace();
            // The TP wrapper has a stronger all-thread/all-CTA drain after
            // the body, once the final combine TMA store is complete.  The
            // original Hopper EP body needs an after-cleanup NVLink barrier;
            // TP-local does not.  Allow the redundant local rendezvous to be
            // removed while keeping the conservative default available.
            if constexpr (!K_NATIVE_SKIP_CLEANUP_GRID_SYNC) {
                comm::grid_sync<kNumSMs, kDispatchGridSyncIndex>(
                    workspace, sm_idx, thread_idx, [=]() { __syncwarp(); });
            }
        }
    } else if (warp_idx == kNumDispatchWarps) {
        // =====================================================================
        // ROLE 2: GEMM TMA LOAD warps (load A+SFA, B+SFB)
        //   The two warps inside `kNumNonEpilogueThreads` load A + SFA and
        //   B + SFB, respectively.
        // =====================================================================
        cutlass::arch::warpgroup_reg_dealloc<kNumNonEpilogueRegisters>();

        const auto load_a_phase = [&](const auto& block_phase,
                                     const uint32_t& local_expert_idx,
                                     const uint32_t& num_k_blocks,
                                     const uint32_t& m_block_idx,
                                     const uint32_t& n_block_idx,
                                     const uint32_t& pool_block_idx,
                                     const uint32_t& valid_m,
                                     const uint32_t& k_block_start) {
            using BlockPhaseTag = std::remove_cv_t<std::remove_reference_t<decltype(block_phase)>>;
            constexpr bool kBlockIsL2 = BlockPhaseTag::value == sched::BlockPhase::Linear2;
            const auto tensor_map_a_ptr = kBlockIsL2 ?
                &tensor_map_l2_acts : &tensor_map_l1_acts;
            const auto tensor_map_sfa_ptr = kBlockIsL2 ?
                &tensor_map_l2_acts_sf : &tensor_map_l1_acts_sf;

            const bool has_valid_m = valid_m > 0;

            // W13 waits for route publication.  W2 consumes the FP8 tile
            // produced later by these same math warpgroups, so its A-loader
            // contribution is deliberately a zero-byte mbarrier arrival.
            if constexpr (!kBlockIsL2) {
                if (has_valid_m) {
                    const auto ptr =
                        workspace.get_l1_arrival_count_ptr(pool_block_idx);
                    while (ptx::ld_acq(ptr) != valid_m) {}
                }
            }
            for (uint32_t k_block_idx = 0; k_block_idx < num_k_blocks; advance_pipeline(k_block_idx)) {
                empty_barriers[stage_idx]->wait(phase ^ 1);

                if (cute::elect_one_sync()) {
                    if constexpr (kBlockIsL2) {
                        full_barriers[stage_idx]->arrive();
                    } else if (has_valid_m) {
                        const uint32_t m_idx = pool_block_idx * BLOCK_M;
                        const uint32_t logical_k_block_idx =
                            k_block_start + k_block_idx;
                        const uint32_t k_idx =
                            logical_k_block_idx * BLOCK_K;

                        // TMA load A
                        tma::copy<BLOCK_K, LOAD_BLOCK_M, kSwizzleAMode, a_dtype_t>(
                            tensor_map_a_ptr, full_barriers[stage_idx], smem_a[stage_idx],
                            k_idx, m_idx, 1);

                        if constexpr (!kBlockIsL2 || !kSplitMDecodedWeightReuse) {
                            tma::copy<BLOCK_M, 1, 0, float>(
                                tensor_map_sfa_ptr, full_barriers[stage_idx], smem_sfa[stage_idx],
                                m_idx, logical_k_block_idx, 1);
                            full_barriers[stage_idx]->arrive_and_expect_tx(
                                SMEM_A_SIZE_PER_STAGE + BLOCK_M * sizeof(float));
                        } else {
                            // BN128 L1 produces per-64 activation scales. L2
                            // consumes both scale groups for each BK128 tile.
                            tma::copy<BLOCK_M, 1, 0, float>(
                                tensor_map_sfa_ptr, full_barriers[stage_idx], smem_sfa[stage_idx],
                                m_idx, logical_k_block_idx * 2, 1);
                            tma::copy<BLOCK_M, 1, 0, float>(
                                tensor_map_sfa_ptr, full_barriers[stage_idx],
                                smem_sfa[stage_idx] + kL2SFAHalfStride,
                                m_idx, logical_k_block_idx * 2 + 1, 1);
                            full_barriers[stage_idx]->arrive_and_expect_tx(
                                SMEM_A_SIZE_PER_STAGE + 2 * BLOCK_M * sizeof(float));
                        }
                    } else {
                        full_barriers[stage_idx]->arrive();
                    }
                }
                __syncwarp();
            }
        };
        const auto load_a_task = [&](const uint32_t& local_expert_idx,
                                     const uint32_t& m_block_idx,
                                     const uint32_t& slice_idx,
                                     const uint32_t& pool_block_idx,
                                     const uint32_t& valid_m) {
            load_a_phase(
                std::integral_constant<
                    sched::BlockPhase, sched::BlockPhase::Linear1>{},
                local_expert_idx, L1_SHAPE_K / BLOCK_K,
                m_block_idx, slice_idx, pool_block_idx, valid_m, 0u);
            #pragma unroll
            for (uint32_t n_block_idx = 0;
                 n_block_idx < kNumW2OutputTiles; ++ n_block_idx) {
                load_a_phase(
                    std::integral_constant<
                        sched::BlockPhase, sched::BlockPhase::Linear2>{},
                    local_expert_idx, 1u, m_block_idx, n_block_idx,
                    pool_block_idx, valid_m, slice_idx);
            }
        };
        for_each_published_block(load_a_task);

    } else if (warp_idx == kNumDispatchWarps + 1) {
        cutlass::arch::warpgroup_reg_dealloc<kNumNonEpilogueRegisters>();

        const auto load_b_phase = [&](const auto& block_phase,
                                     const uint32_t& local_expert_idx,
                                     const uint32_t& num_k_blocks,
                                     const uint32_t& m_block_idx,
                                     const uint32_t& n_block_idx,
                                     const uint32_t& pool_block_idx,
                                     const uint32_t& valid_m,
                                     const uint32_t& k_block_start) {
            using BlockPhaseTag = std::remove_cv_t<std::remove_reference_t<decltype(block_phase)>>;
            constexpr bool kBlockIsL2 = BlockPhaseTag::value == sched::BlockPhase::Linear2;
            const auto tensor_map_b_ptr = kBlockIsL2 ?
                &tensor_map_l2_weights : &tensor_map_l1_weights;
            const auto tensor_map_b_scale_ptr = kBlockIsL2 ?
                &tensor_map_l2_weight_scales : &tensor_map_l1_weight_scales;
            constexpr uint32_t shape_n = kBlockIsL2 ? L2_SHAPE_N : L1_SHAPE_N;
            constexpr uint32_t shape_k = kBlockIsL2 ? L2_SHAPE_K : L1_SHAPE_K;
            constexpr uint32_t scale_n_tiles = shape_n / LOAD_BLOCK_N;

            for (uint32_t k_block_idx = 0; k_block_idx < num_k_blocks; advance_pipeline(k_block_idx)) {
                empty_barriers[stage_idx]->wait(phase ^ 1);

                const uint32_t n_idx = local_expert_idx * shape_n + n_block_idx * BLOCK_N;
                const uint32_t logical_k_block_idx =
                    k_block_start + k_block_idx;
                if (cute::elect_one_sync()) {
                    if constexpr (K_NATIVE_TILE_WEIGHT_SCALE_TMA) {
                        const uint32_t tile_idx =
                            (local_expert_idx * scale_n_tiles + n_block_idx)
                            * (shape_k / BLOCK_K) + logical_k_block_idx;
                        // One contiguous 128x136 box contains a 16 KiB
                        // row-major packed tile followed by its 1 KiB scale
                        // plane, avoiding the split candidate's second TMA.
                        tma::copy<128, 136, 0, uint8_t>(
                            tensor_map_b_ptr, full_barriers[stage_idx],
                            reinterpret_cast<uint8_t*>(
                                smem_packed_b[stage_idx]),
                            0, tile_idx * 136u, 1);
                    } else {
                        const uint32_t k_idx =
                            logical_k_block_idx * B_LOAD_BYTES_PER_ROW;
                        tma::copy<
                            B_LOAD_BYTES_PER_ROW, LOAD_BLOCK_N, 0, b_dtype_t>(
                            tensor_map_b_ptr, full_barriers[stage_idx],
                            smem_packed_b[stage_idx],
                            k_idx, n_idx, 1);
                    }
                    if constexpr (K_NATIVE_SPLIT_WEIGHT_SCALE_TMA) {
                        const uint32_t scale_outer_idx =
                            ((local_expert_idx * scale_n_tiles + n_block_idx)
                                 * (shape_k / BLOCK_K) + logical_k_block_idx)
                            * (LOAD_BLOCK_N / 4u);
                        tma::copy<16, LOAD_BLOCK_N / 4u, 0, uint8_t>(
                            tensor_map_b_scale_ptr, full_barriers[stage_idx],
                            smem_b_scale[stage_idx],
                            0, scale_outer_idx, 1);
                    }
                    full_barriers[stage_idx]->arrive_and_expect_tx(
                        SMEM_PACKED_B_STAGE_SIZE);
                }
                __syncwarp();
            }
        };
        const auto load_b_task = [&](const uint32_t& local_expert_idx,
                                     const uint32_t& m_block_idx,
                                     const uint32_t& slice_idx,
                                     const uint32_t& pool_block_idx,
                                     const uint32_t& valid_m) {
            load_b_phase(
                std::integral_constant<
                    sched::BlockPhase, sched::BlockPhase::Linear1>{},
                local_expert_idx, L1_SHAPE_K / BLOCK_K,
                m_block_idx, slice_idx, pool_block_idx, valid_m, 0u);
            #pragma unroll
            for (uint32_t n_block_idx = 0;
                 n_block_idx < kNumW2OutputTiles; ++ n_block_idx) {
                load_b_phase(
                    std::integral_constant<
                        sched::BlockPhase, sched::BlockPhase::Linear2>{},
                    local_expert_idx, 1u, m_block_idx, n_block_idx,
                    pool_block_idx, valid_m, slice_idx);
            }
        };
        produce_interleaved_blocks(load_b_task);

    } else {
        // =====================================================================
        // ROLE 3: MATH WARPGROUPS (WGMMA + epilogue + combine)
        // =====================================================================
        cutlass::arch::warpgroup_reg_alloc<kNumEpilogueRegisters>();

        const uint32_t epilogue_warp_idx  = warp_idx - (kNumDispatchWarps + kNumMMANonEpilogueWarps);
        const uint32_t epilogue_wg_idx    = epilogue_warp_idx / 4;
        const uint32_t epilogue_thread_idx = epilogue_warp_idx * 32 + lane_idx;
        const uint32_t warp_idx_in_wg     = epilogue_warp_idx % 4;

        const auto arrive_empty_barrier = [&](const uint32_t& s) {
            if (lane_idx == 0)
                empty_barriers[s]->arrive();
        };

        const auto notify_l1_ready = [&](const uint32_t& ready_pool_block_idx,
                                         const uint32_t& ready_n_block_idx) {
            if (epilogue_warp_idx == 0 and cute::elect_one_sync()) {
                ptx::red_or_rel_gpu(
                    workspace.get_l2_arrival_mask_ptr(ready_pool_block_idx),
                    1ull << ready_n_block_idx);
            }
            __syncwarp();
        };

        // WGMMA-output register layout helpers
        const uint32_t row_idx = lane_idx / 4;
        const uint32_t col_idx = lane_idx % 4;
        const uint32_t r_0 = warp_idx_in_wg * 16 + row_idx;
        const uint32_t r_1 = r_0 + 8;

        DG_STATIC_ASSERT(kSwapABRequested ||
                         (WG_BLOCK_M == L1WGMMA::M and WG_BLOCK_N == L1WGMMA::N),
                         "Split-N WGs must each run one M64N128 WGMMA per K-block");

        // Sync with dispatch
        ptx::sync_unaligned(kNumDispatchThreads + kNumEpilogueThreads, kDispatchWithEpilogueBarrierIdx);

        const auto run_math_phase = [&](const auto& block_phase,
                                     const uint32_t& local_expert_idx,
                                     const uint32_t& num_k_blocks,
                                     const uint32_t& m_block_idx,
                                     const uint32_t& n_block_idx,
                                     const uint32_t& pool_block_idx,
                                     const uint32_t& valid_m,
                                     const uint32_t& k_block_start,
                                     const bool& release_task_info) {
            const uint32_t m_idx = pool_block_idx * BLOCK_M;
            const uint32_t wg_n_idx =
                kSplitMDecodedWeightReuse ? 0u : epilogue_wg_idx * WG_BLOCK_N;
            const uint32_t wg_l1_out_n_idx =
                kSplitMDecodedWeightReuse ? 0u : epilogue_wg_idx * WG_L1_OUT_BLOCK_N;
            const uint32_t n_idx = n_block_idx * BLOCK_N + wg_n_idx;
            const uint32_t row_block_offset =
                kSplitMDecodedWeightReuse ? epilogue_wg_idx * WG_BLOCK_M : 0u;
            const uint32_t row_offset_r0 = row_block_offset + r_0;
            const uint32_t row_offset_r1 = row_block_offset + r_1;
            using BlockPhaseTag = std::remove_cv_t<std::remove_reference_t<decltype(block_phase)>>;
            constexpr bool kBlockIsL2 = BlockPhaseTag::value == sched::BlockPhase::Linear2;
            float task_weight_global_scale = 1.0f;
            if constexpr (K_NATIVE_NORMALIZED_WEIGHT_SCALE &&
                          ((kBlockIsL2 && !K_NATIVE_FOLD_W2_GLOBAL_SCALE) ||
                           (!kBlockIsL2 && !K_NATIVE_FOLD_W13_GLOBAL_SCALE))) {
                if (lane_idx == 0) {
                    const float* global_scale =
                        kBlockIsL2 ? w2_global_scale : w13_global_scale;
                    task_weight_global_scale =
                        __ldg(global_scale + local_expert_idx);
                }
                task_weight_global_scale = __shfl_sync(
                    0xffffffffu, task_weight_global_scale, 0);
            }
            float output_weight_global_scale = 1.0f;
            if constexpr (K_NATIVE_FOLD_W2_GLOBAL_SCALE && !kBlockIsL2) {
                if (epilogue_warp_idx == 0 && lane_idx == 0) {
                    output_weight_global_scale =
                        __ldg(w2_global_scale + local_expert_idx);
                }
                output_weight_global_scale = __shfl_sync(
                    0xffffffffu, output_weight_global_scale, 0);
            }
            const auto cast_l2_scaled_bf16_pair = [&](float x, float y) -> uint32_t {
                return math::cast_into_bf16_and_pack(x, y);
            };

            // ---------------- GEMM ----------------
            using WGMMA = L1WGMMA;
            constexpr uint32_t kAccumPerThread = WGMMA::kNumAccum;  // 64 for M=64,N=128
            float final_accum[kAccumPerThread] = {};
            const auto decode_b_stage = [&](const uint32_t& decoded_stage) {
                if constexpr (kSplitMDecodedWeightReuse) {
                    // Both M64 consumers share the same decoded N128 tile.
                    // The two physical decoded slots remove the overwrite
                    // hazard; the trailing pair barrier publishes K+1 before
                    // either consumer issues its WGMMA.
                    DG_STATIC_ASSERT(kUseMode2RowDecoder,
                                     "BM128 split-M uses the cooperative Mode2 decoder");
                    deep_gemm::mxfp4::
                        dequant_smem_b_from_packed_mode2_nibble_split_m(
                            reinterpret_cast<uint8_t*>(smem_b[decoded_stage]),
                            reinterpret_cast<const uint8_t*>(smem_packed_b[decoded_stage]),
                            epilogue_thread_idx, smem_mxfp4_lut);
                    cutlass::arch::fence_view_async_shared();
                    asm volatile("bar.sync %0, %1;" : :
                                 "n"(kSplitMDecodeBarrierIdx), "n"(256) : "memory");
                } else {
                    if constexpr (kUseMode2RowDecoder) {
                        deep_gemm::mxfp4::dequant_smem_b_from_packed_mode2_nibble<
                            kQuadDequantIlp>(
                            reinterpret_cast<uint8_t*>(smem_b[decoded_stage]),
                            reinterpret_cast<const uint8_t*>(smem_packed_b[decoded_stage]),
                            epilogue_thread_idx, smem_mxfp4_lut);
                    } else {
                        deep_gemm::mxfp4::dequant_smem_b_from_packed_braided_lut_window<
                            kQuadDequantIlp>(
                            reinterpret_cast<uint8_t*>(smem_b[decoded_stage]),
                            reinterpret_cast<const uint8_t*>(smem_packed_b[decoded_stage]),
                            epilogue_thread_idx, smem_mxfp4_lut);
                    }
                    cutlass::arch::fence_view_async_shared();
                    ptx::sync_aligned(
                        128, kEpilogueWGBarrierStartIdx + epilogue_wg_idx);
                }
            };
            for (uint32_t k_block_idx = 0;
                 k_block_idx < num_k_blocks;
                 advance_pipeline(k_block_idx)) {
                full_barriers[stage_idx]->wait(phase);
                if (release_task_info && k_block_idx == 0)
                    interleaved_scheduler.release_task_info(lane_idx);
                if constexpr (!kRegisterDequant)
                    decode_b_stage(stage_idx);

                // Read SF (must precede warpgroup_arrive)
                const float scale_a_0_lo = kBlockIsL2 ?
                    ptx::ld_shared(
                        smem_cd_l1_shared_sf + row_offset_r0 *
                            kNumEpilogueWarps + 1u)
                        * task_weight_global_scale :
                    ptx::ld_shared(smem_sfa[stage_idx] + row_offset_r0)
                        * task_weight_global_scale;
                const float scale_a_1_lo = kBlockIsL2 ?
                    ptx::ld_shared(
                        smem_cd_l1_shared_sf + row_offset_r1 *
                            kNumEpilogueWarps + 1u)
                        * task_weight_global_scale :
                    ptx::ld_shared(smem_sfa[stage_idx] + row_offset_r1)
                        * task_weight_global_scale;
                float scale_a_0_hi = 0.0f;
                float scale_a_1_hi = 0.0f;
                if constexpr (kBlockIsL2 && kSplitMDecodedWeightReuse) {
                    scale_a_0_hi = ptx::ld_shared(
                        smem_sfa[stage_idx] + kL2SFAHalfStride + row_offset_r0)
                        * task_weight_global_scale;
                    scale_a_1_hi = ptx::ld_shared(
                        smem_sfa[stage_idx] + kL2SFAHalfStride + row_offset_r1)
                        * task_weight_global_scale;
                }

                // MXFP4 E8M0 weight scales are folded into the FP8 operand
                // either by the legacy shared-memory expansion or directly
                // in registers.  The accumulator therefore only needs the
                // activation scale after each K128 block.

                const auto run_register_dequant_swap_ab = [&]() {
                    DG_STATIC_ASSERT(BLOCK_M == 8 && BLOCK_N == 256,
                                     "RS dequant path is specialized for BM8/BN256");
                    using RSWGMMA = cute::SM90::GMMA::
                        MMA_64x8x32_F32E4M3E4M3_RS_TN<>;
                    constexpr uint32_t kRSAccum = 4;
                    float swap_accum[kSwapABWeightHalves][kRSAccum] = {};
                    const uint32_t packed_k_offset = col_idx * sizeof(uint32_t);
                    uint2 cached_scale_words0[kSwapABWeightHalves];
                    uint2 cached_scale_words1[kSwapABWeightHalves];

                    if constexpr (K_NATIVE_RS_SCALE_WORD_CACHE) {
                        // Each fused 80-byte row carries four E8M0 codes,
                        // duplicated as [e0,e0,e1,e1,e2,e2,e3,e3].  Load the
                        // full K128 record once instead of issuing two byte
                        // loads for every K32 step.
                        #pragma unroll
                        for (uint32_t half = 0;
                             half < kSwapABWeightHalves; ++ half) {
                            const uint32_t packed_row0 =
                                wg_n_idx + half * 64u + r_0;
                            const uint32_t packed_row1 =
                                wg_n_idx + half * 64u + r_1;
                            const uint8_t* row_ptr0 =
                                reinterpret_cast<const uint8_t*>(
                                    smem_packed_b[stage_idx])
                                + packed_row0 * B_LOAD_BYTES_PER_ROW;
                            const uint8_t* row_ptr1 =
                                reinterpret_cast<const uint8_t*>(
                                    smem_packed_b[stage_idx])
                                + packed_row1 * B_LOAD_BYTES_PER_ROW;
                            cached_scale_words0[half] =
                                *reinterpret_cast<const uint2*>(row_ptr0 + 64u);
                            cached_scale_words1[half] =
                                *reinterpret_cast<const uint2*>(row_ptr1 + 64u);
                        }
                    }

                    // The legacy decoder's named barrier also converged all
                    // four consumer warps after their independent mbarrier
                    // waits.  RS keeps operands private, but WGMMA remains a
                    // warpgroup collective and needs the same convergence.
                    ptx::sync_aligned(
                        128, kEpilogueWGBarrierStartIdx + epilogue_wg_idx);

                    if constexpr (K_NATIVE_RS_K128_BATCH) {
                        #pragma unroll
                        for (uint32_t half = 0;
                             half < kSwapABWeightHalves; ++ half) {
                            #pragma unroll
                            for (uint32_t i = 0; i < kRSAccum; ++ i)
                                ptx::warpgroup_fence_operand(
                                    swap_accum[half][i]);
                        }
                        ptx::warpgroup_arrive();
                    }

                    #pragma unroll
                    for (uint32_t k = 0; k < BLOCK_K / 32; ++ k) {
                        if constexpr (!K_NATIVE_RS_K128_BATCH) {
                            #pragma unroll
                            for (uint32_t half = 0;
                                 half < kSwapABWeightHalves; ++ half) {
                                #pragma unroll
                                for (uint32_t i = 0; i < kRSAccum; ++ i)
                                    ptx::warpgroup_fence_operand(
                                        swap_accum[half][i]);
                            }
                            ptx::warpgroup_arrive();
                        }

                        const auto activation_desc =
                            mma::sm90::make_smem_desc(
                                (kBlockIsL2 ? smem_cd_l1 : smem_a[stage_idx])
                                    + k * 32,
                                1);
                        if constexpr (K_NATIVE_RS_HALF_PREFETCH) {
                            uint32_t packed0[kSwapABWeightHalves];
                            uint32_t packed1[kSwapABWeightHalves];
                            uint32_t exponent0[kSwapABWeightHalves];
                            uint32_t exponent1[kSwapABWeightHalves];
                            uint2 lut0[kSwapABWeightHalves];
                            uint2 lut1[kSwapABWeightHalves];

                            // Make the two independent half-tile shared loads
                            // visible before either LUT/dequant dependency
                            // chain.  Keep the live range inside one K32 so
                            // the two-CTA 88-register budget remains viable.
                            #pragma unroll
                            for (uint32_t half = 0;
                                 half < kSwapABWeightHalves; ++ half) {
                                const uint32_t packed_row0 =
                                    wg_n_idx + half * 64u + r_0;
                                const uint32_t packed_row1 =
                                    wg_n_idx + half * 64u + r_1;
                                const uint8_t* row_ptr0 =
                                    reinterpret_cast<const uint8_t*>(
                                        smem_packed_b[stage_idx])
                                    + packed_row0 * B_LOAD_BYTES_PER_ROW;
                                const uint8_t* row_ptr1 =
                                    reinterpret_cast<const uint8_t*>(
                                        smem_packed_b[stage_idx])
                                    + packed_row1 * B_LOAD_BYTES_PER_ROW;
                                if constexpr (kCompactWeightScaleTma) {
                                    exponent0[half] = smem_b_scale[stage_idx][
                                        (packed_row0 >> 2u) * 16u
                                        + (packed_row0 & 3u) * 4u + k];
                                    exponent1[half] = smem_b_scale[stage_idx][
                                        (packed_row1 >> 2u) * 16u
                                        + (packed_row1 & 3u) * 4u + k];
                                } else {
                                    exponent0[half] = row_ptr0[64u + k * 2u];
                                    exponent1[half] = row_ptr1[64u + k * 2u];
                                }
                                packed0[half] =
                                    *reinterpret_cast<const uint32_t*>(
                                        row_ptr0 + k * 16u + packed_k_offset);
                                packed1[half] =
                                    *reinterpret_cast<const uint32_t*>(
                                        row_ptr1 + k * 16u + packed_k_offset);
                            }
                            #pragma unroll
                            for (uint32_t half = 0;
                                 half < kSwapABWeightHalves; ++ half) {
                                lut0[half] = native_load_mxfp4_lut(
                                    exponent0[half], smem_mxfp4_lut);
                                lut1[half] = native_load_mxfp4_lut(
                                    exponent1[half], smem_mxfp4_lut);
                            }
                            #pragma unroll
                            for (uint32_t half = 0;
                                 half < kSwapABWeightHalves; ++ half) {
                                const uint2 fp8_0 = deep_gemm::mxfp4::
                                    dequant_mode2_nibble_word(
                                        packed0[half], lut0[half]);
                                const uint2 fp8_1 = deep_gemm::mxfp4::
                                    dequant_mode2_nibble_word(
                                        packed1[half], lut1[half]);
                                RSWGMMA::fma(
                                    fp8_0.y, fp8_1.y, fp8_0.x, fp8_1.x,
                                    activation_desc,
                                    swap_accum[half][0],
                                    swap_accum[half][1],
                                    swap_accum[half][2],
                                    swap_accum[half][3],
                                    cute::SM90::GMMA::ScaleOut::One);
                            }
                        } else {
                            #pragma unroll
                            for (uint32_t half = 0;
                                 half < kSwapABWeightHalves; ++ half) {
                                const uint32_t packed_row0 =
                                    wg_n_idx + half * 64u + r_0;
                                const uint32_t packed_row1 =
                                    wg_n_idx + half * 64u + r_1;
                                const uint8_t* row_ptr0 =
                                    reinterpret_cast<const uint8_t*>(
                                        smem_packed_b[stage_idx])
                                    + packed_row0 * B_LOAD_BYTES_PER_ROW;
                                const uint8_t* row_ptr1 =
                                    reinterpret_cast<const uint8_t*>(
                                        smem_packed_b[stage_idx])
                                    + packed_row1 * B_LOAD_BYTES_PER_ROW;
                                const uint32_t packed0 =
                                    *reinterpret_cast<const uint32_t*>(
                                        row_ptr0 + k * 16u + packed_k_offset);
                                const uint32_t packed1 =
                                    *reinterpret_cast<const uint32_t*>(
                                        row_ptr1 + k * 16u + packed_k_offset);
                                uint32_t exponent0;
                                uint32_t exponent1;
                                if constexpr (K_NATIVE_RS_SCALE_WORD_CACHE) {
                                    const uint32_t scale_word0 = k < 2
                                        ? cached_scale_words0[half].x
                                        : cached_scale_words0[half].y;
                                    const uint32_t scale_word1 = k < 2
                                        ? cached_scale_words1[half].x
                                        : cached_scale_words1[half].y;
                                    exponent0 =
                                        (scale_word0 >> ((k & 1u) * 16u))
                                        & 0xffu;
                                    exponent1 =
                                        (scale_word1 >> ((k & 1u) * 16u))
                                        & 0xffu;
                                } else if constexpr (kCompactWeightScaleTma) {
                                    exponent0 = smem_b_scale[stage_idx][
                                        (packed_row0 >> 2u) * 16u
                                        + (packed_row0 & 3u) * 4u + k];
                                    exponent1 = smem_b_scale[stage_idx][
                                        (packed_row1 >> 2u) * 16u
                                        + (packed_row1 & 3u) * 4u + k];
                                } else {
                                    exponent0 = row_ptr0[64u + k * 2u];
                                    exponent1 = row_ptr1[64u + k * 2u];
                                }
                                const uint2 lut0 = native_load_mxfp4_lut(
                                    exponent0, smem_mxfp4_lut);
                                const uint2 lut1 = native_load_mxfp4_lut(
                                    exponent1, smem_mxfp4_lut);
                                const uint2 fp8_0 = deep_gemm::mxfp4::
                                    dequant_mode2_nibble_word(packed0, lut0);
                                const uint2 fp8_1 = deep_gemm::mxfp4::
                                    dequant_mode2_nibble_word(packed1, lut1);
                                RSWGMMA::fma(
                                    fp8_0.y, fp8_1.y, fp8_0.x, fp8_1.x,
                                    activation_desc,
                                    swap_accum[half][0],
                                    swap_accum[half][1],
                                    swap_accum[half][2],
                                    swap_accum[half][3],
                                    cute::SM90::GMMA::ScaleOut::One);
                            }
                        }
                        // Commit the first K64 half while the consumer lanes
                        // prepare the second half's packed MXFP4 operands.
                        // Both groups keep the same accumulator registers and
                        // remain unread until the final wait below.
                        if constexpr (K_NATIVE_RS_K128_BATCH &&
                                      K_NATIVE_RS_K64_COMMIT_GROUPS) {
                            if ((k & 1u) == 1u)
                                ptx::warpgroup_commit_batch();
                        }
                        if constexpr (!K_NATIVE_RS_K128_BATCH) {
                            ptx::warpgroup_commit_batch();
                            #pragma unroll
                            for (uint32_t half = 0;
                                 half < kSwapABWeightHalves; ++ half) {
                                #pragma unroll
                                for (uint32_t i = 0; i < kRSAccum; ++ i)
                                    ptx::warpgroup_fence_operand(
                                        swap_accum[half][i]);
                            }
                            ptx::warpgroup_wait<0>();
                        }
                    }

                    if constexpr (K_NATIVE_RS_K128_BATCH) {
                        if constexpr (!K_NATIVE_RS_K64_COMMIT_GROUPS)
                            ptx::warpgroup_commit_batch();
                        #pragma unroll
                        for (uint32_t half = 0;
                             half < kSwapABWeightHalves; ++ half) {
                            #pragma unroll
                            for (uint32_t i = 0; i < kRSAccum; ++ i)
                                ptx::warpgroup_fence_operand(
                                    swap_accum[half][i]);
                        }
                        ptx::warpgroup_wait<0>();
                    }

                    #pragma unroll
                    for (uint32_t half = 0;
                         half < kSwapABWeightHalves; ++ half) {
                        const uint32_t accum_offset =
                            half * kSwapABHalfAccumPerThread;
                        const uint32_t token_0 = col_idx * 2;
                        const uint32_t token_1 = token_0 + 1;
                        if (token_0 < valid_m) {
                            const float scale_0 = kBlockIsL2 ?
                                ptx::ld_shared(
                                    smem_cd_l1_shared_sf + token_0 *
                                        kNumEpilogueWarps + 1u)
                                    * task_weight_global_scale :
                                ptx::ld_shared(smem_sfa[stage_idx] + token_0)
                                    * task_weight_global_scale;
                            final_accum[accum_offset + 0] +=
                                scale_0 * swap_accum[half][0];
                            final_accum[accum_offset + 2] +=
                                scale_0 * swap_accum[half][2];
                        }
                        if (token_1 < valid_m) {
                            const float scale_1 = kBlockIsL2 ?
                                ptx::ld_shared(
                                    smem_cd_l1_shared_sf + token_1 *
                                        kNumEpilogueWarps + 1u)
                                    * task_weight_global_scale :
                                ptx::ld_shared(smem_sfa[stage_idx] + token_1)
                                    * task_weight_global_scale;
                            final_accum[accum_offset + 1] +=
                                scale_1 * swap_accum[half][1];
                            final_accum[accum_offset + 3] +=
                                scale_1 * swap_accum[half][3];
                        }
                    }
                    arrive_empty_barrier(stage_idx);
                };

                if constexpr (!kBlockIsL2) {
                    if constexpr (kSwapABRequested) {
                        if constexpr (kRegisterDequant) {
                            run_register_dequant_swap_ab();
                        } else {
                        auto run_swap_ab_l1 = [&]<uint32_t N_SWAP>() {
                            using SwapWGMMA = typename mma::sm90::FP8MMASelector<N_SWAP>::type;
                            constexpr uint32_t kSwapAccum = SwapWGMMA::kNumAccum;
                            float swap_accum[kSwapAccum];

                            #pragma unroll
                            for (uint32_t half = 0; half < kSwapABWeightHalves; ++ half) {
                                #pragma unroll
                                for (uint32_t i = 0; i < kSwapAccum; ++ i)
                                    ptx::warpgroup_fence_operand(swap_accum[i]);
                                ptx::warpgroup_arrive();
                                #pragma unroll
                                for (uint32_t k = 0; k < BLOCK_K / SwapWGMMA::K; ++ k) {
                                    auto desc_a = mma::sm90::make_smem_desc(
                                        smem_b[stage_idx] + (wg_n_idx + half * 64u) * BLOCK_K + k * SwapWGMMA::K, 1);
                                    auto desc_b = mma::sm90::make_smem_desc(
                                        smem_a[stage_idx] + k * SwapWGMMA::K, 1);
                                    SwapWGMMA::wgmma(desc_a, desc_b, swap_accum, k);
                                }
                                ptx::warpgroup_commit_batch();
                                #pragma unroll
                                for (uint32_t i = 0; i < kSwapAccum; ++ i)
                                    ptx::warpgroup_fence_operand(swap_accum[i]);
                                ptx::warpgroup_wait<0>();

                                #pragma unroll
                                for (uint32_t i = 0; i < kSwapAccum / 4; ++ i) {
                                    const uint32_t accum_offset = half * kSwapABHalfAccumPerThread + i * 4;
                                    const uint32_t token_0 = i * 8 + col_idx * 2;
                                    const uint32_t token_1 = token_0 + 1;
                                    if (token_0 < valid_m) {
                                        const float scale_0 = ptx::ld_shared(smem_sfa[stage_idx] + token_0);
                                        final_accum[accum_offset + 0] += scale_0 * swap_accum[i * 4 + 0];
                                        final_accum[accum_offset + 2] += scale_0 * swap_accum[i * 4 + 2];
                                    }
                                    if (token_1 < valid_m) {
                                        const float scale_1 = ptx::ld_shared(smem_sfa[stage_idx] + token_1);
                                        final_accum[accum_offset + 1] += scale_1 * swap_accum[i * 4 + 1];
                                        final_accum[accum_offset + 3] += scale_1 * swap_accum[i * 4 + 3];
                                    }
                                }
                            }

                            arrive_empty_barrier(stage_idx);
                        };

                        if constexpr (BLOCK_M == 8) {
                            run_swap_ab_l1.template operator()<8>();
                        } else if constexpr (BLOCK_M == 16) {
                            const uint32_t n_swap = ((valid_m + 7u) / 8u) * 8u;
                            if (n_swap <= 8) {
                                run_swap_ab_l1.template operator()<8>();
                            } else {
                                run_swap_ab_l1.template operator()<16>();
                            }
                        } else if constexpr (BLOCK_M == 24) {
                            const uint32_t n_swap = ((valid_m + 7u) / 8u) * 8u;
                            if (n_swap <= 8) {
                                run_swap_ab_l1.template operator()<8>();
                            } else if (n_swap <= 16) {
                                run_swap_ab_l1.template operator()<16>();
                            } else {
                                run_swap_ab_l1.template operator()<24>();
                            }
                        }
                        }
                    } else {
                        float accum[kAccumPerThread];
                        // Single per-128 K-block WGMMA group
                        #pragma unroll
                        for (uint32_t i = 0; i < kAccumPerThread; ++ i)
                            ptx::warpgroup_fence_operand(accum[i]);
                        ptx::warpgroup_arrive();
                        #pragma unroll
                        for (uint32_t k = 0; k < BLOCK_K / WGMMA::K; ++ k) {
                            auto desc_a = mma::sm90::make_smem_desc(
                                smem_a[stage_idx] + row_block_offset * BLOCK_K +
                                k * WGMMA::K, 1);
                            auto desc_b = mma::sm90::make_smem_desc(
                                smem_b[stage_idx] + wg_n_idx * BLOCK_K + k * WGMMA::K, 1);
                            WGMMA::wgmma(desc_a, desc_b, accum, k);
                        }
                        ptx::warpgroup_commit_batch();
                        #pragma unroll
                        for (uint32_t i = 0; i < kAccumPerThread; ++ i)
                            ptx::warpgroup_fence_operand(accum[i]);
                        ptx::warpgroup_wait<0>();

                        arrive_empty_barrier(stage_idx);

                        // L1: gate/up alternate at gran=8 along N; each `i` block
                        // of 8 cols belongs entirely to one of {gate, up}, so .x
                        // and .y share the same scalar.
                        #pragma unroll
                        for (uint32_t i = 0; i < kAccumPerThread / 4; ++ i) {
                            final_accum[i*4+0] += scale_a_0_lo * accum[i*4+0];
                            final_accum[i*4+1] += scale_a_0_lo * accum[i*4+1];
                            final_accum[i*4+2] += scale_a_1_lo * accum[i*4+2];
                            final_accum[i*4+3] += scale_a_1_lo * accum[i*4+3];
                        }
                    }
                } else {
                    if constexpr (kSwapABRequested) {
                        DG_STATIC_ASSERT(kL2ActsSFGranK == 128,
                                         "L2 swap-AB requires per-128 activation scales");
                        if constexpr (kRegisterDequant) {
                            run_register_dequant_swap_ab();
                        } else {
                        auto run_swap_ab_l2 = [&]<uint32_t N_SWAP>() {
                            using SwapWGMMA = typename mma::sm90::FP8MMASelector<N_SWAP>::type;
                            constexpr uint32_t kSwapAccum = SwapWGMMA::kNumAccum;
                            float swap_accum[kSwapAccum];

                            #pragma unroll
                            for (uint32_t half = 0; half < kSwapABWeightHalves; ++ half) {
                                #pragma unroll
                                for (uint32_t i = 0; i < kSwapAccum; ++ i)
                                    ptx::warpgroup_fence_operand(swap_accum[i]);
                                ptx::warpgroup_arrive();
                                #pragma unroll
                                for (uint32_t k = 0; k < BLOCK_K / SwapWGMMA::K; ++ k) {
                                    auto desc_a = mma::sm90::make_smem_desc(
                                        smem_b[stage_idx] + (wg_n_idx + half * 64u) * BLOCK_K + k * SwapWGMMA::K, 1);
                                    auto desc_b = mma::sm90::make_smem_desc(
                                        smem_a[stage_idx] + k * SwapWGMMA::K, 1);
                                    SwapWGMMA::wgmma(desc_a, desc_b, swap_accum, k);
                                }
                                ptx::warpgroup_commit_batch();
                                #pragma unroll
                                for (uint32_t i = 0; i < kSwapAccum; ++ i)
                                    ptx::warpgroup_fence_operand(swap_accum[i]);
                                ptx::warpgroup_wait<0>();
                                #pragma unroll
                                for (uint32_t i = 0; i < kSwapAccum / 4; ++ i) {
                                    const uint32_t accum_offset =
                                        half * kSwapABHalfAccumPerThread + i * 4;
                                    const uint32_t token_0 = i * 8 + col_idx * 2;
                                    const uint32_t token_1 = token_0 + 1;
                                    if (token_0 < valid_m) {
                                        const float scale_0 = ptx::ld_shared(
                                            smem_sfa[stage_idx] + token_0);
                                        final_accum[accum_offset + 0] +=
                                            scale_0 * swap_accum[i * 4 + 0];
                                        final_accum[accum_offset + 2] +=
                                            scale_0 * swap_accum[i * 4 + 2];
                                    }
                                    if (token_1 < valid_m) {
                                        const float scale_1 = ptx::ld_shared(
                                            smem_sfa[stage_idx] + token_1);
                                        final_accum[accum_offset + 1] +=
                                            scale_1 * swap_accum[i * 4 + 1];
                                        final_accum[accum_offset + 3] +=
                                            scale_1 * swap_accum[i * 4 + 3];
                                    }
                                }
                            }

                            arrive_empty_barrier(stage_idx);
                        };

                        if constexpr (BLOCK_M == 8) {
                            run_swap_ab_l2.template operator()<8>();
                        } else if constexpr (BLOCK_M == 16) {
                            const uint32_t n_swap = ((valid_m + 7u) / 8u) * 8u;
                            if (n_swap <= 8) {
                                run_swap_ab_l2.template operator()<8>();
                            } else {
                                run_swap_ab_l2.template operator()<16>();
                            }
                        } else if constexpr (BLOCK_M == 24) {
                            const uint32_t n_swap = ((valid_m + 7u) / 8u) * 8u;
                            if (n_swap <= 8) {
                                run_swap_ab_l2.template operator()<8>();
                            } else if (n_swap <= 16) {
                                run_swap_ab_l2.template operator()<16>();
                            } else {
                                run_swap_ab_l2.template operator()<24>();
                            }
                        }
                        }
                    } else {
                        float accum[kAccumPerThread];
                        const auto promote_l2_accum = [&](const float& scale_r0,
                                                          const float& scale_r1) {
                            #pragma unroll
                            for (uint32_t i = 0; i < kAccumPerThread / 4; ++ i) {
                                final_accum[i*4+0] += scale_r0 * accum[i*4+0];
                                final_accum[i*4+1] += scale_r0 * accum[i*4+1];
                                final_accum[i*4+2] += scale_r1 * accum[i*4+2];
                                final_accum[i*4+3] += scale_r1 * accum[i*4+3];
                            }
                        };
                        if constexpr (kSplitMDecodedWeightReuse) {
                            #pragma unroll
                            for (uint32_t sf_group = 0; sf_group < 2; ++ sf_group) {
                                #pragma unroll
                                for (uint32_t i = 0; i < kAccumPerThread; ++ i)
                                    ptx::warpgroup_fence_operand(accum[i]);
                                ptx::warpgroup_arrive();
                                #pragma unroll
                                for (uint32_t k = 0;
                                     k < (BLOCK_K / 2) / WGMMA::K; ++ k) {
                                    const uint32_t k_off =
                                        sf_group * (BLOCK_K / 2) + k * WGMMA::K;
                                    auto desc_a = mma::sm90::make_smem_desc(
                                        smem_a[stage_idx] + row_block_offset * BLOCK_K + k_off, 1);
                                    auto desc_b = mma::sm90::make_smem_desc(
                                        smem_b[stage_idx] + wg_n_idx * BLOCK_K + k_off, 1);
                                    WGMMA::wgmma(desc_a, desc_b, accum, k);
                                }
                                ptx::warpgroup_commit_batch();
                                #pragma unroll
                                for (uint32_t i = 0; i < kAccumPerThread; ++ i)
                                    ptx::warpgroup_fence_operand(accum[i]);
                                ptx::warpgroup_wait<0>();
                                if (sf_group == 0)
                                    promote_l2_accum(scale_a_0_lo, scale_a_1_lo);
                                else
                                    promote_l2_accum(scale_a_0_hi, scale_a_1_hi);
                            }
                            arrive_empty_barrier(stage_idx);
                        } else {
                            // One per-128 scale permits a single four-instruction
                            // WGMMA group and one accumulator promotion per K tile.
                            #pragma unroll
                            for (uint32_t i = 0; i < kAccumPerThread; ++ i)
                                ptx::warpgroup_fence_operand(accum[i]);
                            ptx::warpgroup_arrive();
                            #pragma unroll
                            for (uint32_t k = 0; k < BLOCK_K / WGMMA::K; ++ k) {
                                auto desc_a = mma::sm90::make_smem_desc(
                                    smem_a[stage_idx] + row_block_offset * BLOCK_K +
                                    k * WGMMA::K, 1);
                                auto desc_b = mma::sm90::make_smem_desc(
                                    smem_b[stage_idx] + wg_n_idx * BLOCK_K + k * WGMMA::K, 1);
                                WGMMA::wgmma(desc_a, desc_b, accum, k);
                            }
                            ptx::warpgroup_commit_batch();
                            #pragma unroll
                            for (uint32_t i = 0; i < kAccumPerThread; ++ i)
                                ptx::warpgroup_fence_operand(accum[i]);
                            ptx::warpgroup_wait<0>();
                            arrive_empty_barrier(stage_idx);
                            promote_l2_accum(scale_a_0_lo, scale_a_1_lo);
                        }
                    }
                }
            }

            // Skip epilogue when block is past valid M (the GEMM loop already
            // released its pipeline stages). Drain any prior L1 async store.
            if (valid_m == 0) {
                ptx::sync_aligned(kNumEpilogueThreads, kEpilogueFullBarrierIdx);
                return;
            }

            if constexpr (!kBlockIsL2) {
                if constexpr (kSwapABRequested) {
                    auto silu = [](float x) -> float {
                        const float e = kFastMath ? __expf(-x) : expf(-x);
                        const float sig = kFastMath ? math::fast_rcp(1.0f + e) : 1.0f / (1.0f + e);
                        return x * sig;
                    };
                    auto clamp_gate = [](float& x) {
                        if constexpr (kActivationClamp != cute::numeric_limits<float>::infinity())
                            x = cute::min(x, kActivationClamp);
                    };
                    auto clamp_up = [](float& x) {
                        if constexpr (kActivationClamp != cute::numeric_limits<float>::infinity())
                            x = cute::min(cute::max(x, -kActivationClamp), kActivationClamp);
                    };

                    constexpr uint32_t reduce_warp_start = 0;
                    constexpr uint32_t reduce_warp_count = kNumEpilogueWarps;
                    const uint32_t scale_token_thread = epilogue_thread_idx;
                    constexpr uint32_t scale_token_stride = kNumEpilogueThreads;
                    float swap_v0[kSwapABWeightHalves][kSwapABTokenChunks] = {};
                    float swap_v1[kSwapABWeightHalves][kSwapABTokenChunks] = {};

                    auto store_l1_swap_chunk = [&](const uint32_t& i) {
                        const uint32_t token_0 = i * 8 + col_idx * 2;
                        const uint32_t token_1 = token_0 + 1;

                        float v0_amax = 0.0f;
                        float v1_amax = 0.0f;
                        #pragma unroll
                        for (uint32_t half = 0; half < kSwapABWeightHalves; ++ half) {
                            const uint32_t accum_offset = half * kSwapABHalfAccumPerThread + i * 4;
                            float v0 = 0.0f;
                            if (token_0 < valid_m) {
                                float g0 = final_accum[accum_offset + 0];
                                float u0 = final_accum[accum_offset + 2];
                                clamp_gate(g0);
                                clamp_up(u0);
                                v0 = silu(g0) * u0;
                                swap_v0[half][i] = v0;
                                v0_amax = cute::max(v0_amax, cute::abs(v0));
                            }

                            float v1 = 0.0f;
                            if (token_1 < valid_m) {
                                float g1 = final_accum[accum_offset + 1];
                                float u1 = final_accum[accum_offset + 3];
                                clamp_gate(g1);
                                clamp_up(u1);
                                v1 = silu(g1) * u1;
                                swap_v1[half][i] = v1;
                                v1_amax = cute::max(v1_amax, cute::abs(v1));
                            }
                        }

                        const float amax0 = math::warp_reduce<4, true>(
                            v0_amax, math::ReduceMax<float>());
                        const float amax1 = math::warp_reduce<4, true>(
                            v1_amax, math::ReduceMax<float>());
                        if (row_idx == 0) {
                            if (token_0 < valid_m)
                                smem_cd_l1_shared_sf[token_0 * kNumEpilogueWarps + epilogue_warp_idx] = amax0;
                            if (token_1 < valid_m)
                                smem_cd_l1_shared_sf[token_1 * kNumEpilogueWarps + epilogue_warp_idx] = amax1;
                        }
                    };

                    const uint32_t num_swap_token_chunks = (valid_m + 7u) / 8u;
                    store_l1_swap_chunk(0);
                    if (valid_m > 8) {
                        #pragma unroll
                        for (uint32_t i = 1; i < kSwapABTokenChunks; ++ i) {
                            if (i < num_swap_token_chunks)
                                store_l1_swap_chunk(i);
                        }
                    }

                    ptx::sync_aligned(kNumEpilogueThreads, kEpilogueFullBarrierIdx);

                    for (uint32_t token = scale_token_thread;
                         token < valid_m;
                         token += scale_token_stride) {
                        float amax = 0.0f;
                        #pragma unroll
                        for (uint32_t w = 0; w < reduce_warp_count; ++ w)
                            amax = cute::max(
                                amax, smem_cd_l1_shared_sf[token * kNumEpilogueWarps + reduce_warp_start + w]);
                        float2 amax_pair = {amax, amax};
                        float2 sf_pair, sf_inv_pair;
                        math::get_e4m3_sf_and_sf_inv(amax_pair, sf_pair, sf_inv_pair);

                        // Keep both quantization factors CTA-local. Slot zero
                        // is consumed by the FP8 stores below; slot one is the
                        // dequant scale consumed by every W2 N tile.
                        smem_cd_l1_shared_sf[
                            token * kNumEpilogueWarps + reduce_warp_start] =
                            sf_inv_pair.x;
                        smem_cd_l1_shared_sf[
                            token * kNumEpilogueWarps + 1u] =
                            sf_pair.x * output_weight_global_scale;
                    }

                    ptx::sync_aligned(kNumEpilogueThreads, kEpilogueFullBarrierIdx);

                    #pragma unroll
                    for (uint32_t i = 0; i < kSwapABTokenChunks; ++ i) {
                        const uint32_t token_0 = i * 8 + col_idx * 2;
                        const uint32_t token_1 = token_0 + 1;
                        #pragma unroll
                        for (uint32_t half = 0; half < kSwapABWeightHalves; ++ half) {
                            const uint32_t out_col_base =
                                wg_l1_out_n_idx + half * 32u + warp_idx_in_wg * 8 + row_idx;
                            if (token_0 < valid_m) {
                                const float sf_inv =
                                    smem_cd_l1_shared_sf[token_0 * kNumEpilogueWarps + reduce_warp_start];
                                const __nv_fp8_e4m3 q(swap_v0[half][i] * sf_inv);
                                const uint32_t physical_col = out_col_base ^
                                    ((token_0 & 7u) * 16u);
                                reinterpret_cast<uint8_t*>(smem_cd_l1)[
                                    token_0 * L1_OUT_BLOCK_N + physical_col] =
                                    *reinterpret_cast<const uint8_t*>(&q);
                            }
                            if (token_1 < valid_m) {
                                const float sf_inv =
                                    smem_cd_l1_shared_sf[token_1 * kNumEpilogueWarps + reduce_warp_start];
                                const __nv_fp8_e4m3 q(swap_v1[half][i] * sf_inv);
                                const uint32_t physical_col = out_col_base ^
                                    ((token_1 & 7u) * 16u);
                                reinterpret_cast<uint8_t*>(smem_cd_l1)[
                                    token_1 * L1_OUT_BLOCK_N + physical_col] =
                                    *reinterpret_cast<const uint8_t*>(&q);
                            }
                        }
                    }
                    ptx::sync_aligned(kNumEpilogueThreads, kEpilogueFullBarrierIdx);
                } else {
                    // ---------------- L1 EPILOGUE: SwiGLU + FP8 quantize + TMA store ----------------
                    const bool valid_r0 = row_offset_r0 < valid_m;
                    const bool valid_r1 = row_offset_r1 < valid_m;
                    // Layout in `final_accum`:
                    //   16 chunks of 8 N-cols, each chunk = 4 floats per thread = (r0c0, r0c1, r1c0, r1c1).
                    //   Gate chunks: even (0, 2, ..., 14). Up chunks: odd (1, 3, ..., 15).
                    //   Pair `p` ∈ [0, 8): gate chunk = 2p, up chunk = 2p+1.
                    //
                    // For each pair we produce 4 post-SwiGLU floats per thread, mapped to
                    // output cols (p*8 + col_idx*2 + {0,1}) for both r0 and r1.

                    constexpr uint32_t kNumPairs = kAccumPerThread / 8;
                    constexpr uint32_t kNumSFGroups = 1;
                    float swiglu_r0[kNumPairs][2];
                    float swiglu_r1[kNumPairs][2];

                    // Per-row amax, one scale for each 64-col L1 output group.
                    float amax_r0[kNumSFGroups] = {};
                    float amax_r1[kNumSFGroups] = {};

                    // Compute SwiGLU + per-group amax.
                    #pragma unroll
                    for (uint32_t p = 0; p < kNumPairs; ++ p) {
                        const uint32_t gate = 2 * p, up = 2 * p + 1;
                        const uint32_t sf_group = p / 8;

                        auto clamp_gate = [](float& x) {
                            if constexpr (kActivationClamp != cute::numeric_limits<float>::infinity())
                                x = cute::min(x, kActivationClamp);
                        };
                        auto clamp_up = [](float& x) {
                            if constexpr (kActivationClamp != cute::numeric_limits<float>::infinity())
                                x = cute::min(cute::max(x, -kActivationClamp), kActivationClamp);
                        };
                        float g_r0_c0 = final_accum[gate*4 + 0]; clamp_gate(g_r0_c0);
                        float g_r0_c1 = final_accum[gate*4 + 1]; clamp_gate(g_r0_c1);
                        float g_r1_c0 = final_accum[gate*4 + 2]; clamp_gate(g_r1_c0);
                        float g_r1_c1 = final_accum[gate*4 + 3]; clamp_gate(g_r1_c1);
                        float u_r0_c0 = final_accum[up*4   + 0]; clamp_up(u_r0_c0);
                        float u_r0_c1 = final_accum[up*4   + 1]; clamp_up(u_r0_c1);
                        float u_r1_c0 = final_accum[up*4   + 2]; clamp_up(u_r1_c0);
                        float u_r1_c1 = final_accum[up*4   + 3]; clamp_up(u_r1_c1);

                        auto silu = [](float x) -> float {
                            const float e = kFastMath ? __expf(-x) : expf(-x);
                            const float sig = kFastMath ? math::fast_rcp(1.0f + e) : 1.0f / (1.0f + e);
                            return x * sig;
                        };

                        if (valid_r0) {
                            swiglu_r0[p][0] = silu(g_r0_c0) * u_r0_c0;
                            swiglu_r0[p][1] = silu(g_r0_c1) * u_r0_c1;
                            amax_r0[sf_group] = cute::max(
                                amax_r0[sf_group],
                                cute::max(cute::abs(swiglu_r0[p][0]), cute::abs(swiglu_r0[p][1])));
                        } else {
                            swiglu_r0[p][0] = 0.0f;
                            swiglu_r0[p][1] = 0.0f;
                        }
                        if (valid_r1) {
                            swiglu_r1[p][0] = silu(g_r1_c0) * u_r1_c0;
                            swiglu_r1[p][1] = silu(g_r1_c1) * u_r1_c1;
                            amax_r1[sf_group] = cute::max(
                                amax_r1[sf_group],
                                cute::max(cute::abs(swiglu_r1[p][0]), cute::abs(swiglu_r1[p][1])));
                        } else {
                            swiglu_r1[p][0] = 0.0f;
                            swiglu_r1[p][1] = 0.0f;
                        }
                    }


                    const float weight_r0 = valid_r0 ? *l1_topk_weights_buffer
                        .get_data_buffer(m_idx + row_offset_r0)
                        .template get_base_ptr<float>() : 0.0f;
                    const float weight_r1 = valid_r1 ? *l1_topk_weights_buffer
                        .get_data_buffer(m_idx + row_offset_r1)
                        .template get_base_ptr<float>() : 0.0f;
                    #pragma unroll
                    for (uint32_t p = 0; p < kNumPairs; ++ p) {
                        swiglu_r0[p][0] *= weight_r0;
                        swiglu_r0[p][1] *= weight_r0;
                        swiglu_r1[p][0] *= weight_r1;
                        swiglu_r1[p][1] *= weight_r1;
                    }
                    #pragma unroll
                    for (uint32_t g = 0; g < kNumSFGroups; ++ g) {
                        amax_r0[g] *= cute::abs(weight_r0);
                        amax_r1[g] *= cute::abs(weight_r1);
                    }
                    #pragma unroll
                    for (uint32_t g = 0; g < kNumSFGroups; ++ g) {
                        amax_r0[g] = math::warp_reduce<4, false>(amax_r0[g], math::ReduceMax<float>());
                        amax_r1[g] = math::warp_reduce<4, false>(amax_r1[g], math::ReduceMax<float>());
                    }

                    if (col_idx == 0) {
                        smem_cd_l1_shared_sf[epilogue_wg_idx * BLOCK_M + row_offset_r0] = amax_r0[0];
                        smem_cd_l1_shared_sf[epilogue_wg_idx * BLOCK_M + row_offset_r1] = amax_r1[0];
                    }
                    ptx::sync_aligned(kNumEpilogueThreads, kEpilogueFullBarrierIdx);
                    if constexpr (kSplitMDecodedWeightReuse) {
                        amax_r0[0] = smem_cd_l1_shared_sf[
                            epilogue_wg_idx * BLOCK_M + row_offset_r0];
                        amax_r1[0] = smem_cd_l1_shared_sf[
                            epilogue_wg_idx * BLOCK_M + row_offset_r1];
                    } else {
                        amax_r0[0] = cute::max(
                            smem_cd_l1_shared_sf[row_offset_r0],
                            smem_cd_l1_shared_sf[BLOCK_M + row_offset_r0]);
                        amax_r1[0] = cute::max(
                            smem_cd_l1_shared_sf[row_offset_r1],
                            smem_cd_l1_shared_sf[BLOCK_M + row_offset_r1]);
                    }

                    float sf_r0[kNumSFGroups], sf_inv_r0[kNumSFGroups];
                    float sf_r1[kNumSFGroups], sf_inv_r1[kNumSFGroups];
                    #pragma unroll
                    for (uint32_t g = 0; g < kNumSFGroups; ++ g) {
                        float2 amax_pair = {amax_r0[g], amax_r1[g]};
                        float2 sf_pair, sf_inv_pair;
                        math::get_e4m3_sf_and_sf_inv(amax_pair, sf_pair, sf_inv_pair);
                        sf_r0[g] = sf_pair.x; sf_inv_r0[g] = sf_inv_pair.x;
                        sf_r1[g] = sf_pair.y; sf_inv_r1[g] = sf_inv_pair.y;
                    }

                    // Quantize and write to smem_cd_l1 (row-major, no swizzle).
                    #pragma unroll
                    for (uint32_t p = 0; p < kNumPairs; ++ p) {
                        const uint32_t sf_group = p / 8;
                        const float v00 = swiglu_r0[p][0] * sf_inv_r0[sf_group];
                        const float v01 = swiglu_r0[p][1] * sf_inv_r0[sf_group];
                        const float v10 = swiglu_r1[p][0] * sf_inv_r1[sf_group];
                        const float v11 = swiglu_r1[p][1] * sf_inv_r1[sf_group];

                        const __nv_fp8x2_e4m3 r0_pair(make_float2(v00, v01));
                        const __nv_fp8x2_e4m3 r1_pair(make_float2(v10, v11));

                        const uint32_t col = p * 8 + col_idx * 2;
                        auto* p0 = reinterpret_cast<uint16_t*>(
                            smem_cd_l1 + row_offset_r0 * L1_OUT_BLOCK_N +
                            wg_l1_out_n_idx + col);
                        auto* p1 = reinterpret_cast<uint16_t*>(
                            smem_cd_l1 + row_offset_r1 * L1_OUT_BLOCK_N +
                            wg_l1_out_n_idx + col);
                        if (valid_r0)
                            *p0 = r0_pair.__x;
                        if (valid_r1)
                            *p1 = r1_pair.__x;
                    }

                    // Write one physical L2-activation scale per 128 output columns.
                    if (col_idx == 0) {
                        auto sf_base_ptr = l2_sf_buffer.get_base_ptr<float>();
                        const uint32_t token_r0 = m_idx + row_offset_r0;
                        const uint32_t token_r1 = m_idx + row_offset_r1;
                        const uint32_t base_k_sf_idx =
                            (n_block_idx * L1_OUT_BLOCK_N + wg_l1_out_n_idx) / kL2ActsSFGranK;
                        #pragma unroll
                        for (uint32_t g = 0; g < kNumSFGroups; ++ g) {
                            const uint32_t sf_k_idx = base_k_sf_idx + g;
                            if ((kSplitMDecodedWeightReuse || epilogue_wg_idx == 0) && valid_r0)
                                sf_base_ptr[sf_k_idx * kNumPaddedSFPoolTokens + token_r0] =
                                    sf_r0[g] * output_weight_global_scale;
                            if ((kSplitMDecodedWeightReuse || epilogue_wg_idx == 0) && valid_r1)
                                sf_base_ptr[sf_k_idx * kNumPaddedSFPoolTokens + token_r1] =
                                    sf_r1[g] * output_weight_global_scale;
                        }
                    }

                    // Issue TMA store of the entire tile. Padding rows beyond
                    // `valid_m` are written with stale/garbage FP8 to the L1-output
                    // pool buffer, but they are never consumed downstream: the L2
                    // GEMM tile loads them, but its NVLink-scatter epilogue is
                    // gated by `m_idx_in_block >= valid_m`, and stale SF in the
                    // padding rows can produce NaN accumulators that simply stay
                    // in registers (only valid rows are converted to BF16 and
                    // STSM'd into smem). Using TMA for partial tiles is a large
                    // win for low-batch / decode where every tile is partial.
                    ptx::sync_aligned(kNumEpilogueThreads, kEpilogueFullBarrierIdx);
                    if (epilogue_warp_idx == 0 and cute::elect_one_sync()) {
                        const uint32_t out_n_idx = n_block_idx * L1_OUT_BLOCK_N;
                        cute::tma_store_fence();
                        cute::SM90_TMA_STORE_2D::copy(
                            &tensor_map_l1_output,
                            smem_cd_l1,
                            out_n_idx,
                            m_idx);
                        cute::tma_store_arrive();
                    }
                    __syncwarp();
                    ptx::tma_store_wait<0>();
                    ptx::sync_aligned(kNumEpilogueThreads, kEpilogueFullBarrierIdx);
                    notify_l1_ready(pool_block_idx, n_block_idx);
                }
            } else {
                // ---------------- W2 EPILOGUE: FP32 K128 slice partial ----------------
                // Each routed row maps back to a unique (token, top-k slot).
                // The slice dimension is disjoint across fused microtasks, and
                // the two math WGs own disjoint hidden columns, so no atomics
                // are required here.
                DG_STATIC_ASSERT(kSwapABRequested && BLOCK_M == 8,
                                 "tile-WS partial store requires BM8 swap-AB");
                const uint32_t slice_idx = k_block_start;
                auto store_partial = [&](const uint32_t& token,
                                         const uint32_t& col,
                                         const float& value) {
                    if (token < valid_m) {
                        const auto src_metadata =
                            *workspace.get_token_src_metadata_ptr(
                                pool_block_idx * BLOCK_M + token);
                        const uint64_t partial_idx =
                            (((static_cast<uint64_t>(slice_idx) * kNumTopk +
                               src_metadata.topk_idx) *
                                  kNumMaxTokensPerRank +
                              src_metadata.token_idx) *
                                 kHidden) +
                            n_idx + col;
                        w2_partials[partial_idx] = value;
                    }
                };

                const uint32_t token_0 = col_idx * 2;
                const uint32_t token_1 = token_0 + 1;
                #pragma unroll
                for (uint32_t half = 0;
                     half < kSwapABWeightHalves; ++ half) {
                    const uint32_t accum_offset =
                        half * kSwapABHalfAccumPerThread;
                    const uint32_t col_offset = half * 64u;
                    store_partial(token_0, col_offset + r_0,
                                  final_accum[accum_offset + 0]);
                    store_partial(token_0, col_offset + r_1,
                                  final_accum[accum_offset + 2]);
                    store_partial(token_1, col_offset + r_0,
                                  final_accum[accum_offset + 1]);
                    store_partial(token_1, col_offset + r_1,
                                  final_accum[accum_offset + 3]);
                }
            }

            if constexpr (K_NATIVE_PHASE_STAMPS) {
                if (cumulative_local_expert_recv_stats != nullptr &&
                        epilogue_thread_idx == 0) {
                    const uint32_t phase_offset = kBlockIsL2 ? kNumSMs : 0;
                    native_write_phase_stamp(
                        cumulative_local_expert_recv_stats,
                        4 + phase_offset + sm_idx);
                }
            }
        };
        const auto run_math_task = [&](const uint32_t& local_expert_idx,
                                       const uint32_t& m_block_idx,
                                       const uint32_t& slice_idx,
                                       const uint32_t& pool_block_idx,
                                       const uint32_t& valid_m) {
            run_math_phase(
                std::integral_constant<
                    sched::BlockPhase, sched::BlockPhase::Linear1>{},
                local_expert_idx, L1_SHAPE_K / BLOCK_K,
                m_block_idx, slice_idx, pool_block_idx, valid_m,
                0u, true);
            #pragma unroll
            for (uint32_t n_block_idx = 0;
                 n_block_idx < kNumW2OutputTiles; ++ n_block_idx) {
                run_math_phase(
                    std::integral_constant<
                        sched::BlockPhase, sched::BlockPhase::Linear2>{},
                    local_expert_idx, 1u, m_block_idx, n_block_idx,
                    pool_block_idx, valid_m, slice_idx, false);
            }
        };
        for_each_published_block(run_math_task);

        // ---------------- COMBINE ----------------
        // NVLink barrier first: signals remote ranks that this rank's GEMM
        // outputs (NVLink scatter targets) are fully written.
        if constexpr (K_NATIVE_TP_LOCAL_BARRIER_FASTPATH) {
            // W2 writes are local in pure TP.  One grid publication is enough
            // before the ordered local k6 reduction; the inherited EP barrier
            // otherwise performs two grid rendezvous around a rank-1 signal.
            comm::grid_sync<kNumSMs, kEpilogueGridSyncIndex>(
                workspace, sm_idx, epilogue_thread_idx,
                [&]() {
                    ptx::sync_aligned(
                        kNumEpilogueThreads, kEpilogueFullBarrierIdx);
                });
        } else {
            comm::nvlink_barrier<kNumRanks, kNumSMs, kNumEpilogueThreads,
                                 kEpilogueGridSyncIndex,
                                 kBeforeCombineReduceBarrierTag>(
                workspace, sym_buffer, sm_idx, epilogue_thread_idx,
                [&]() {
                    ptx::sync_aligned(
                        kNumEpilogueThreads, kEpilogueFullBarrierIdx);
                });
        }

        if constexpr (K_NATIVE_PHASE_STAMPS) {
            if (cumulative_local_expert_recv_stats != nullptr &&
                    sm_idx == 0 && epilogue_thread_idx == 0) {
                native_write_phase_stamp(
                    cumulative_local_expert_recv_stats, 2);
            }
        }

        // Sync with dispatch (paired with dispatch's pre-cleanup sync) so that
        // dispatch may now safely clean workspace state.
        ptx::sync_unaligned(kNumDispatchThreads + kNumEpilogueThreads, kDispatchWithEpilogueBarrierIdx);

        // Ordered local reduction of the disjoint K128 slice partials.  Match
        // the public numerical boundary: first accumulate the full W2 K for
        // one route in FP32, round that route result to BF16, then apply
        // topk_weight * 1.5 and advance to the next slot in fixed k6 order.
        // Invalid top-k slots are skipped.
        auto* local_output = reinterpret_cast<__nv_bfloat16*>(y);
        const uint32_t global_math_thread =
            sm_idx * kNumEpilogueThreads + epilogue_thread_idx;
        const uint32_t num_math_threads =
            kNumSMs * kNumEpilogueThreads;
        const uint32_t num_output_elements = num_tokens * kHidden;
        for (uint32_t output_idx = global_math_thread;
             output_idx < num_output_elements;
             output_idx += num_math_threads) {
            const uint32_t token_idx = output_idx / kHidden;
            const uint32_t hidden_idx = output_idx % kHidden;
            float reduced = 0.0f;
            #pragma unroll
            for (uint32_t slot_idx = 0;
                 slot_idx < kNumTopk; ++ slot_idx) {
                const uint32_t route_idx = token_idx * kNumTopk + slot_idx;
                const int64_t expert_idx = __ldg(
                    input_topk_idx_buffer.get_base_ptr<int64_t>() + route_idx);
                if (expert_idx >= 0) {
                    float route_value = 0.0f;
                    #pragma unroll
                    for (uint32_t slice_idx = 0;
                         slice_idx < kNumIntermediateSlices; ++ slice_idx) {
                        const uint64_t partial_idx =
                            (((static_cast<uint64_t>(slice_idx) * kNumTopk +
                               slot_idx) * kNumMaxTokensPerRank + token_idx) *
                                 kHidden) +
                            hidden_idx;
                        route_value += w2_partials[partial_idx];
                    }
                    const float route_weight = __ldg(
                        input_topk_weights_buffer.get_base_ptr<float>() +
                        route_idx) * 1.5f;
                    reduced += __bfloat162float(
                                   __float2bfloat16_rn(route_value)) *
                               route_weight;
                }
            }
            local_output[output_idx] = __float2bfloat16_rn(reduced);
        }
    }
#else
    if (blockIdx.x == 0 and threadIdx.x == 0)
        DG_DEVICE_ASSERT(false and "This kernel only supports sm_90");
#endif
