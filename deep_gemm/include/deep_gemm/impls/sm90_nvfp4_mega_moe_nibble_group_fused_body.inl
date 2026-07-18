// Independent SM90 NVFP4 MegaMoE fused kernel body.
#if (defined(__CUDA_ARCH__) and (__CUDA_ARCH__ >= 900) and (__CUDA_ARCH__ < 1000)) or defined(__CLION_IDE__)
    using Barrier = cutlass::arch::ClusterTransactionBarrier;

    // =====================================================================
    // Template checks
    // =====================================================================
    DG_STATIC_ASSERT(kNumDispatchThreads == 64 or kNumDispatchThreads % 128 == 0, "Invalid number of dispatch threads");
    DG_STATIC_ASSERT(kNumNonEpilogueThreads == 64 or kNumNonEpilogueThreads == 128, "Invalid number of GEMM TMA warps (2 or 4 warps expected)");
    DG_STATIC_ASSERT(kNumEpilogueThreads % 128 == 0, "Invalid number of math/epilogue threads");
    DG_STATIC_ASSERT((kNumDispatchThreads + kNumNonEpilogueThreads) % 128 == 0,
                     "Epilogue warpgroups must start at a warpgroup-aligned thread offset");
    DG_STATIC_ASSERT(kNumExperts % kNumRanks == 0, "Invalid number of experts or ranks");
    DG_STATIC_ASSERT(kClusterSize == 1 or kClusterSize == 2, "Invalid cluster size");
    DG_STATIC_ASSERT(kNumSMs % kClusterSize == 0, "SM count must be divisible by cluster size");
    DG_STATIC_ASSERT(BLOCK_M % 64 == 0, "BLOCK_M must be a multiple of WGMMA::M (64)");
    DG_STATIC_ASSERT(BLOCK_N == 64 or BLOCK_N == 128 or BLOCK_N == 256, "BLOCK_N must be 64/128/256 for this SM90 path");
    DG_STATIC_ASSERT(BLOCK_N == 128 or BLOCK_N == 256,
                     "NVFP4 smem dequant supports BN128 and opt-in BN256 scale tile layouts");
    DG_STATIC_ASSERT(BLOCK_K == 128, "BLOCK_K is fixed to 128 (per-128 SF)");
    DG_STATIC_ASSERT(kIntermediateHidden <= 2048,
                     "nibble-group candidate is restricted to Flash layouts");
    DG_STATIC_ASSERT(BLOCK_M == 64 && BLOCK_N == 256,
                     "nibble-group candidate is restricted to small-M BN256");
    DG_STATIC_ASSERT((!kLoaderDequantRequested) or kNumNonEpilogueThreads == 128 or
                     (kNumNonEpilogueThreads == 64 and BLOCK_N == 256),
                     "NVFP4 loader dequant expects four non-epilogue warps or the BN256 packed-scratch path");

    // =====================================================================
    // Thread / warp identification
    // =====================================================================
    const uint32_t sm_idx     = blockIdx.x;
    const uint32_t thread_idx = threadIdx.x;
    const uint32_t warp_idx   = cutlass::canonical_warp_idx_sync();
    const uint32_t lane_idx   = ptx::get_lane_idx();

    if (warp_idx == 0 and cute::elect_one_sync()) {
        cute::prefetch_tma_descriptor(&tensor_map_l1_acts);
        cute::prefetch_tma_descriptor(&tensor_map_l1_acts_sf);
        cute::prefetch_tma_descriptor(&tensor_map_l1_weights);
        cute::prefetch_tma_descriptor(&tensor_map_l1_output);
        cute::prefetch_tma_descriptor(&tensor_map_l2_acts);
        cute::prefetch_tma_descriptor(&tensor_map_l2_acts_sf);
        cute::prefetch_tma_descriptor(&tensor_map_l2_weights);
    }

    // =====================================================================
    // Workspaces and symmetric buffer slicing (mirror SM100 layout, except SF
    // for L2 activations uses per-64 K granularity)
    // =====================================================================
    const auto workspace = layout::Workspace(
        sym_buffer.get_base_ptr(), kNumRanks, kNumExperts, kNumMaxTokensPerRank, kNumTopk);

    constexpr auto fp8_token_layout              = layout::Data(kHidden);
    constexpr auto bf16_token_layout             = layout::Data(kHidden * sizeof(nv_bfloat16));
    constexpr auto fp8_intermediate_token_layout = layout::Data(kIntermediateHidden);
    // Per-128 K float SF: 4 bytes per per-128 group => `kHidden / 32` bytes/token (same as SM100 packing)
    constexpr auto fp8_sf_layout                 = layout::Data(kHidden / 32);
    // Per-64 K float SF (SM90 only): 4 bytes per per-64 group => `kIntermediateHidden / 16` bytes/token
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
    constexpr bool kSplitNWarpgroups =
        BLOCK_M == 64 && BLOCK_N == 256 && kNumEpilogueWarpgroups == 2;
    constexpr bool kSerialNWarpgroups = false;
    constexpr bool kWideNWarpgroups =
        BLOCK_N == 256 && kNumEpilogueWarpgroups == 1;
    constexpr uint32_t WG_BLOCK_M = kSplitNWarpgroups ? BLOCK_M : BLOCK_M / kNumEpilogueWarpgroups;
    constexpr uint32_t WG_BLOCK_N = (kSplitNWarpgroups || kSerialNWarpgroups) ? BLOCK_N / 2 : BLOCK_N;
    constexpr uint32_t L1_OUT_BLOCK_N = BLOCK_N / 2;       // post-SwiGLU tile N
    constexpr uint32_t WG_L1_OUT_BLOCK_N = WG_BLOCK_N / 2; // post-SwiGLU per-WG N
    constexpr bool kL2DualAccum = false;
    constexpr bool kL2ArrivalCounter = false;
    constexpr bool kSkipL2ReadyMask = false;
    constexpr bool kSkipL1ReadyNotify = false;
    constexpr bool kSwapABEligible = kSwapABRequested && kSplitNWarpgroups &&
        BLOCK_M == 64 && BLOCK_N == 256 && WG_BLOCK_N == 128;
    constexpr bool kSwapABL1Active = kSwapABEligible;
    constexpr bool kSwapABL2Active = kSwapABEligible;
    constexpr bool kSwapABFlashFineDispatch =
        kSwapABEligible && kIntermediateHidden <= 2048;
    constexpr uint32_t kSwapABTokenChunks = BLOCK_M / 8;
    constexpr uint32_t kSwapABWeightHalves = WG_BLOCK_N / 64;
    constexpr uint32_t kSwapABHalfAccumPerThread = 64 * 64 / 128;
    DG_STATIC_ASSERT(!kSwapABEligible || WG_L1_OUT_BLOCK_N == 64,
                     "swapAB expects BN256 split-N with 64 L1 output columns per WG");
    // Keep the 4-warp dispatch allocation for warpgroup/register alignment,
    // but only use two warps for BN128 split L1 dispatch. The extra NVFP4
    // dispatch warps mostly add fixed small/mid-M overhead while loader-dequant
    // still needs its aligned 4-warp non-epilogue group.
    constexpr uint32_t kNumActiveDispatchWarps = kNumDispatchWarps;
    constexpr uint32_t kNumActiveDispatchThreads = kNumActiveDispatchWarps * 32;
    constexpr bool kLoaderDequant = kLoaderDequantRequested && kNumMMANonEpilogueWarps == 4;
    constexpr bool kPackedBScratch = BLOCK_N == 256 && (!kLoaderDequant);
    // The host selects EPW8 only for the H200 MiMo M512 plan.
    constexpr bool kH200MimoM512QuadILP =
        kNumSMs >= 132 && kNumRanks == 8 && kNumTopk == 8 &&
        kNumExperts == 384 && kNumExpertsPerWave == 8 &&
        kHidden == 6144 && kIntermediateHidden == 2048 &&
        BLOCK_M == 64 && BLOCK_N == 256;
    constexpr bool kH200MimoM1024QuadILP =
        kNumSMs >= 132 && kNumRanks == 8 && kNumTopk == 8 &&
        kNumExperts == 384 && kNumExpertsPerWave == 48 &&
        kHidden == 6144 && kIntermediateHidden == 2048 &&
        BLOCK_M == 64 && BLOCK_N == 256 && kQuadDequantILP;
    constexpr bool kH200MimoQuadILP =
        kH200MimoM512QuadILP || kH200MimoM1024QuadILP;
    DG_STATIC_ASSERT(kLoaderDequant || kPackedBScratch,
                     "Fused NVFP4 B+scale layout requires loader dequant or packed scratch");
    using L1WGMMA   = typename mma::sm90::FP8MMASelector<WG_BLOCK_N>::type;
    using L2WGMMA   = typename mma::sm90::FP8MMASelector<WG_BLOCK_N>::type;
    static_assert(L1WGMMA::M == 64 and L1WGMMA::N == WG_BLOCK_N and L1WGMMA::K == 32,
                  "Unexpected WGMMA shape");
    DG_STATIC_ASSERT((!kSplitNWarpgroups) or (BLOCK_M == 64 and (WG_BLOCK_N == 64 or WG_BLOCK_N == 128)),
                     "Split-N path expects M64N64 or M64N128 WGMMA consumers");

    // A is always CTA-local.  When kClusterSize=2 the scheduler pairs adjacent
    // M blocks with identical expert/N/K coordinates so the B TMA can multicast.
    constexpr uint32_t LOAD_BLOCK_M    = BLOCK_M;
    constexpr uint32_t LOAD_BLOCK_N    = BLOCK_N;
    constexpr uint32_t kSwizzleAMode   = BLOCK_K * sizeof(a_dtype_t);   // 128
    constexpr uint32_t kSwizzleBMode   = BLOCK_K * sizeof(b_dtype_t);   // 128
    constexpr uint32_t kSwizzleCDMode  = 128;
    constexpr uint32_t kGranK          = 128;          // L1 acts SF, weights SF
    constexpr uint32_t kL2ActsSFGranK  = 64;           // L2 acts SF (per-64 K, SM90 only)
    constexpr bool kSplitNSharesSF = kSplitNWarpgroups && WG_L1_OUT_BLOCK_N < kL2ActsSFGranK;

    // =====================================================================
    // Shared memory layout
    // =====================================================================
    constexpr uint32_t kSharedMemoryAlignment = 1024;
    extern __shared__ __align__(kSharedMemoryAlignment) uint8_t smem_buffer[];

    constexpr uint32_t SMEM_EXPERT_COUNT_SIZE =
        math::constexpr_align<uint32_t>(kNumExperts * sizeof(uint32_t), kSharedMemoryAlignment);
    constexpr uint32_t SMEM_SEND_BUFFER_SIZE =
        math::constexpr_align(fp8_token_layout.get_num_bytes() * kNumActiveDispatchWarps, kSharedMemoryAlignment);
    constexpr uint32_t SMEM_NVFP4_LUT_SIZE =
        math::constexpr_align<uint32_t>(128u * sizeof(uint2), kSharedMemoryAlignment);
    constexpr uint32_t SMEM_A_SIZE_PER_STAGE = LOAD_BLOCK_M * BLOCK_K * sizeof(a_dtype_t);
    constexpr uint32_t SMEM_B_SIZE_PER_STAGE = LOAD_BLOCK_N * BLOCK_K * sizeof(b_dtype_t);
    constexpr uint32_t B_LOAD_BYTES_PER_ROW = 80u;
    constexpr uint32_t SMEM_B_LOAD_SIZE_PER_STAGE = LOAD_BLOCK_N * B_LOAD_BYTES_PER_ROW;
    constexpr uint32_t SMEM_PACKED_B_SIZE_PER_STAGE = kPackedBScratch ?
        LOAD_BLOCK_N * B_LOAD_BYTES_PER_ROW * sizeof(b_dtype_t) : 0u;
    // SFA per-stage must be sized for the larger of L1 (BLOCK_M floats) and L2
    // (two per-64-K halves). Each TMA destination must be 128B aligned, so
    // the second L2 half cannot start immediately after 16 floats in M16 decode.
    constexpr uint32_t kL2SFAHalfStride =
        math::constexpr_align<uint32_t>(BLOCK_M * sizeof(float), 128u) / sizeof(float);
    constexpr uint32_t SMEM_SFA_SIZE_PER_STAGE = 2 * kL2SFAHalfStride * sizeof(float);
    // CD output: max of L1 FP8 (BLOCK_M * (BLOCK_N/2) * 1 byte * num_wg) and
    // L2 BF16 (BLOCK_M * BLOCK_N * 2 bytes * num_wg).
    constexpr uint32_t SMEM_CD_ACCUM_SIZE = 0u;
    constexpr uint32_t SMEM_CD_L1_SIZE =
        kNumEpilogueWarpgroups * WG_BLOCK_M * WG_L1_OUT_BLOCK_N * sizeof(cutlass::float_e4m3_t);
    constexpr uint32_t SMEM_CD_L2_SIZE = kSwapABL2Active ?
        BLOCK_M * BLOCK_N * sizeof(nv_bfloat16) : 0u;
    constexpr uint32_t SMEM_CD_OUTPUT_BASE_SIZE =
        SMEM_CD_L1_SIZE > SMEM_CD_L2_SIZE ? SMEM_CD_L1_SIZE : SMEM_CD_L2_SIZE;
    constexpr uint32_t SMEM_CD_L1_SHARED_SF_SLOTS = kSplitNSharesSF ?
        kNumEpilogueWarpgroups * BLOCK_M : 0u;
    constexpr uint32_t SMEM_CD_L1_SWAP_AMAX_SLOTS = kSwapABL1Active ?
        BLOCK_M * kNumEpilogueWarps : 0u;
    constexpr uint32_t SMEM_CD_L1_EXTRA_FLOAT_SLOTS =
        SMEM_CD_L1_SHARED_SF_SLOTS > SMEM_CD_L1_SWAP_AMAX_SLOTS ?
        SMEM_CD_L1_SHARED_SF_SLOTS : SMEM_CD_L1_SWAP_AMAX_SLOTS;
    constexpr uint32_t SMEM_CD_L1_SHARED_SF_SIZE =
        SMEM_CD_L1_EXTRA_FLOAT_SLOTS * sizeof(float);
    constexpr uint32_t SMEM_CD_OUTPUT_UNALIGNED_SIZE =
        SMEM_CD_OUTPUT_BASE_SIZE + SMEM_CD_L1_SHARED_SF_SIZE;
    constexpr uint32_t SMEM_CD_OUTPUT_SIZE = math::constexpr_align(
        SMEM_CD_OUTPUT_UNALIGNED_SIZE, kSharedMemoryAlignment);
    constexpr uint32_t SMEM_CD_SIZE = SMEM_CD_ACCUM_SIZE + SMEM_CD_OUTPUT_SIZE;

    constexpr uint32_t SMEM_BEFORE_BARRIER_SIZE =
        SMEM_EXPERT_COUNT_SIZE + SMEM_SEND_BUFFER_SIZE + SMEM_NVFP4_LUT_SIZE + SMEM_CD_SIZE +
        kNumStages * (SMEM_A_SIZE_PER_STAGE + SMEM_B_SIZE_PER_STAGE + SMEM_PACKED_B_SIZE_PER_STAGE);

    // SMEM pointers
    auto smem_expert_count = reinterpret_cast<uint32_t*>(smem_buffer);
    const auto smem_send_buffers = layout::Buffer(
        fp8_token_layout, kNumActiveDispatchWarps, 1,
        math::advance_ptr(smem_buffer, SMEM_EXPERT_COUNT_SIZE));
    auto smem_nvfp4_lut = reinterpret_cast<uint2*>(math::advance_ptr<uint8_t>(
        smem_buffer, SMEM_EXPERT_COUNT_SIZE + SMEM_SEND_BUFFER_SIZE));

    auto smem_gemm_base = math::advance_ptr(
        smem_buffer, SMEM_EXPERT_COUNT_SIZE + SMEM_SEND_BUFFER_SIZE + SMEM_NVFP4_LUT_SIZE);

    auto smem_cd_base = smem_gemm_base;
    // CD output is shared by L1 (FP8) and L2 (BF16); reinterpret-cast as needed.
    auto smem_cd_l1 = reinterpret_cast<cutlass::float_e4m3_t*>(smem_cd_base);
    auto smem_cd_l1_shared_sf = reinterpret_cast<float*>(smem_cd_base + SMEM_CD_OUTPUT_BASE_SIZE);
    auto smem_cd_l2 = reinterpret_cast<nv_bfloat16*>(smem_cd_base);

    auto smem_a = utils::PatternVisitor([=](const uint32_t& i) {
        return math::advance_ptr<a_dtype_t>(smem_gemm_base, SMEM_CD_SIZE + i * SMEM_A_SIZE_PER_STAGE);
    });
    auto smem_b = utils::PatternVisitor([=](const uint32_t& i) {
        return math::advance_ptr<b_dtype_t>(smem_gemm_base, SMEM_CD_SIZE + kNumStages * SMEM_A_SIZE_PER_STAGE + i * SMEM_B_SIZE_PER_STAGE);
    });
    auto smem_packed_b = utils::PatternVisitor([=](const uint32_t& i) {
        return math::advance_ptr<b_dtype_t>(
            smem_gemm_base, SMEM_CD_SIZE + kNumStages * (SMEM_A_SIZE_PER_STAGE + SMEM_B_SIZE_PER_STAGE) +
            i * SMEM_PACKED_B_SIZE_PER_STAGE);
    });
    auto sf_start_ptr = math::advance_ptr<uint8_t>(smem_gemm_base,
        SMEM_CD_SIZE + kNumStages * (SMEM_A_SIZE_PER_STAGE + SMEM_B_SIZE_PER_STAGE + SMEM_PACKED_B_SIZE_PER_STAGE));
    auto smem_sfa = utils::PatternVisitor([=](const uint32_t& i) {
        return reinterpret_cast<float*>(sf_start_ptr + i * SMEM_SFA_SIZE_PER_STAGE);
    });
    constexpr uint32_t kNumDequantBarriers = kLoaderDequant ? kNumStages : 0u;

    // Barriers live after SF.
    auto barrier_start_ptr = reinterpret_cast<Barrier*>(
        sf_start_ptr + kNumStages * SMEM_SFA_SIZE_PER_STAGE);
    auto dispatch_barriers = utils::PatternVisitor([=](const uint32_t& i) { return barrier_start_ptr + i; });
    auto full_barriers     = utils::PatternVisitor([=](const uint32_t& i) { return barrier_start_ptr + kNumDispatchWarps + i; });
    auto empty_barriers    = utils::PatternVisitor([=](const uint32_t& i) { return barrier_start_ptr + kNumDispatchWarps + kNumStages + i; });
    auto dequant_barriers  = utils::PatternVisitor([=](const uint32_t& i) { return barrier_start_ptr + kNumDispatchWarps + kNumStages * 2 + i; });
    auto combine_barriers  = utils::PatternVisitor([=](const uint32_t& i) { return barrier_start_ptr + kNumDispatchWarps + kNumStages * 2 + kNumDequantBarriers + i; });

    // =====================================================================
    // Initialization
    // =====================================================================
    if (thread_idx < 64) {
        reinterpret_cast<uint4*>(smem_nvfp4_lut)[thread_idx] =
            reinterpret_cast<const uint4*>(deep_gemm::nvfp4::kE2M1AndUe4m3ToFp8Lut)[thread_idx];
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
                // With cluster multicast the leader CTA's TMA warp waits on peer
                // empty barriers too, so every math warp releases both CTAs.
                empty_barriers[i]->init(kClusterSize * kNumEpilogueWarps);
            }
            if constexpr (kLoaderDequant) {
                #pragma unroll
                for (uint32_t i = 0; i < kNumStages; ++ i) dequant_barriers[i]->init(1);
            }
            #pragma unroll
            for (uint32_t i = 0; i < kNumEpilogueWarps * 2; ++ i)
                combine_barriers[i]->init(1);
        }
        cutlass::arch::fence_barrier_init();
    }
    if constexpr (kClusterSize > 1) {
        cute::cluster_sync();
    } else {
        __syncthreads();
    }

    // =====================================================================
    // Scheduler (cluster=1)
    // =====================================================================
    auto scheduler = sched::MegaMoEScheduler<
        BLOCK_M, BLOCK_N, BLOCK_K,
        L1_SHAPE_N, L1_SHAPE_K,
        L2_SHAPE_N, L2_SHAPE_K,
        kNumExpertsPerRank, kNumExpertsPerWave,
        kNumSMs, kNumRanks, kClusterSize>(workspace);

    // Pipeline state shared by TMA loaders and math warpgroups
    uint32_t stage_idx = 0, phase = 0;
    auto advance_pipeline = [&](uint32_t& k_block_idx) {
        ++ k_block_idx;
        stage_idx = stage_idx == kNumStages - 1 ? 0 : stage_idx + 1;
        phase ^= stage_idx == 0;
    };
    const auto dequant_loaded_b_stage = [&](const uint32_t& s, const uint32_t& p,
                                            const uint32_t& non_epilogue_thread_idx) {
        if constexpr (kLoaderDequant) {
            if constexpr (kNumMMANonEpilogueWarps == 2 && kPackedBScratch && LOAD_BLOCK_N == 256) {
                full_barriers[s]->wait(p);
                #pragma unroll
                for (uint32_t row = non_epilogue_thread_idx; row < LOAD_BLOCK_N; row += 64u)
                    deep_gemm::nvfp4::dequant_smem_b_from_packed_grouped_nibble<
                        kH200MimoQuadILP>(
                        reinterpret_cast<uint8_t*>(smem_b[s]),
                        reinterpret_cast<const uint8_t*>(smem_packed_b[s]),
                        row, smem_nvfp4_lut);
                cutlass::arch::fence_view_async_shared();
                ptx::sync_aligned(64, 8);
                if (non_epilogue_thread_idx == 0)
                    dequant_barriers[s]->arrive();
            } else if constexpr (kNumMMANonEpilogueWarps == 4) {
                if constexpr (LOAD_BLOCK_N == 256) {
                    full_barriers[s]->wait(p);
                    const uint32_t dequant_tid = non_epilogue_thread_idx;
                    deep_gemm::nvfp4::dequant_smem_b_inplace_two_rows_grouped_nibble<
                        128u, 8u, 80u>(
                        reinterpret_cast<uint8_t*>(smem_b[s]), dequant_tid, smem_nvfp4_lut);
                    cutlass::arch::fence_view_async_shared();
                    ptx::sync_aligned(128, 8);
                    if (dequant_tid == 0)
                        dequant_barriers[s]->arrive();
                } else if (non_epilogue_thread_idx >= 64u) {
                    full_barriers[s]->wait(p);
                    const uint32_t dequant_tid = non_epilogue_thread_idx - 64u;
                    deep_gemm::nvfp4::dequant_smem_b_inplace_two_rows_grouped_nibble<
                        64u, 8u, 80u>(
                        reinterpret_cast<uint8_t*>(smem_b[s]), dequant_tid, smem_nvfp4_lut);
                    cutlass::arch::fence_view_async_shared();
                    ptx::sync_aligned(64, 8);
                    if (dequant_tid == 0)
                        dequant_barriers[s]->arrive();
                }
            }
        } else {
            (void)s; (void)p; (void)non_epilogue_thread_idx;
        }
    };

    // Intra-SM barrier indices (mirroring SM100)
    constexpr uint32_t kDispatchBarrierIdx              = 0;
    constexpr uint32_t kDispatchWithEpilogueBarrierIdx  = 1;
    constexpr uint32_t kEpilogueFullBarrierIdx          = 2;
    constexpr uint32_t kEpilogueWGBarrierStartIdx       = 3;

    // Cross-rank NVLink barrier tags
    constexpr uint32_t kBeforeDispatchPullBarrierTag    = 1;
    constexpr uint32_t kBeforeCombineReduceBarrierTag   = 2;
    constexpr uint32_t kAfterWorkspaceCleanBarrierTag   = 3;

    // Register reconfiguration counts (chosen to fit in 64512 reg budget).
    // For the 256-epilogue-thread loader-dequant case (block_m=128, 2 math WGs),
    // give non-epilogue warps more room for two-row NVFP4 dequant while staying
    // inside the 64K register file: 128*48 + 128*64 + 256*192 = 63488.
    constexpr uint32_t kNumDispatchRegisters    = 48;
    constexpr uint32_t kNumNonEpilogueRegisters =
        (kLoaderDequant && kNumEpilogueThreads == 128) ? 80 :
        (kLoaderDequant && kNumEpilogueThreads == 256 && LOAD_BLOCK_N == 256) ? 80 :
        (kLoaderDequant && kNumEpilogueThreads == 256) ? 64 : 40;
    constexpr uint32_t kNumEpilogueRegisters    =
        (kLoaderDequant && kNumEpilogueThreads == 256 && LOAD_BLOCK_N == 256) ? 184 :
        (kLoaderDequant && kNumEpilogueThreads == 256) ? 192 :
        (kSerialNWarpgroups or kWideNWarpgroups) ? 256 :
        208;
    DG_STATIC_ASSERT(kNumDispatchRegisters * kNumDispatchThreads +
                     kNumNonEpilogueRegisters * kNumNonEpilogueThreads +
                     kNumEpilogueRegisters * kNumEpilogueThreads <= 64512,
                     "Too many registers");

    constexpr uint32_t kDispatchGridSyncIndex = 0;
    constexpr uint32_t kEpilogueGridSyncIndex = 1;

    constexpr uint32_t kProfileDispatchTotal = 0;
    constexpr uint32_t kProfileDispatchPull = 1;
    constexpr uint32_t kProfileMathLoop = 2;
    constexpr uint32_t kProfileCombineBarrier = 3;
    constexpr uint32_t kProfileCombineReduce = 4;
    constexpr uint32_t kProfileGemmCore = 5;
    constexpr uint32_t kProfileL1Epilogue = 6;
    constexpr uint32_t kProfileL2Epilogue = 7;
    constexpr uint32_t kProfileLoaderDequant = 8;
    constexpr uint32_t kProfileMathDequantWait = 9;
    constexpr uint32_t kProfileL1TMAWait = 10;
    constexpr uint32_t kProfileL1ReadyNotify = 11;
    constexpr uint32_t kProfileL2ReadyWait = 12;
    constexpr uint32_t kProfileL2Scatter = 13;
    constexpr uint32_t kProfileMathFullWait = 14;
    constexpr uint32_t kProfileMathDequant = 15;
    constexpr uint32_t kNumPhaseProfileMetrics = 16;
    const auto phase_profile_clock = [&]() -> unsigned long long {
        if constexpr (kPhaseProfileRequested) {
            unsigned long long t;
            asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t));
            return t;
        } else {
            return 0ull;
        }
    };
    const auto phase_profile_record = [&](const uint32_t& metric, const unsigned long long& cycles) {
        if constexpr (kPhaseProfileRequested) {
            if (cumulative_local_expert_recv_stats != nullptr and cycles > 0) {
                auto profile = reinterpret_cast<unsigned long long*>(
                    cumulative_local_expert_recv_stats + kNumExpertsPerRank);
                atomicAdd(profile + metric, cycles);
                atomicMax(profile + kNumPhaseProfileMetrics + metric, cycles);
                atomicAdd(profile + 2 * kNumPhaseProfileMetrics + metric, 1ull);
            }
        }
    };

    const auto for_each_selected_block = [&](auto&& func) {
        scheduler.fetch_expert_recv_count();
        scheduler.set_expert_idx(0);
        while (true) {
            CUTE_TIE_DECL(scheduler.get_next_block(),
                          block_phase, local_expert_idx, m_block_idx, n_block_idx);
            if (block_phase == sched::BlockPhase::None)
                break;
            if (block_phase == sched::BlockPhase::Linear1) {
                func(std::integral_constant<sched::BlockPhase, sched::BlockPhase::Linear1>{},
                     local_expert_idx, L1_SHAPE_K / BLOCK_K, m_block_idx, n_block_idx);
            } else {
                func(std::integral_constant<sched::BlockPhase, sched::BlockPhase::Linear2>{},
                     local_expert_idx, L2_SHAPE_K / BLOCK_K, m_block_idx, n_block_idx);
            }
        }
    };

    const auto cleanup_workspace = [&]() {
        if constexpr (kNumDispatchWarps == 0) {
            return;
        } else {
        DG_STATIC_ASSERT(kNumSMs > 1, "Invalid SM count");
        if (sm_idx == 0) {
            #pragma unroll
            for (uint32_t i = thread_idx; i < kNumExperts; i += kNumDispatchThreads)
                *workspace.get_expert_send_count_ptr(i) = 0;
        } else {
            for (uint32_t i = sm_idx - 1; i < kNumExpertsPerRank; i += kNumSMs - 1) {
                const auto num_recv_tokens = static_cast<uint32_t>(
                    *workspace.get_expert_recv_count_sum_ptr(i));
                const auto num_recv_m_blocks = math::ceil_div(num_recv_tokens, BLOCK_M);
                const auto cleanup_pool_block_offset = scheduler.get_pool_block_offset(i);

                ptx::sync_aligned(kNumDispatchThreads, kDispatchBarrierIdx);

                DG_STATIC_ASSERT(kNumDispatchWarps >= 2, "Not enough dispatch warps");
                if (warp_idx == 0) {
                    *workspace.get_expert_recv_count_sum_ptr(i) = 0;
                } else if (warp_idx == 1) {
                    if (cute::elect_one_sync() and cumulative_local_expert_recv_stats != nullptr)
                        ptx::red_add(cumulative_local_expert_recv_stats + i, static_cast<int>(num_recv_tokens));
                    __syncwarp();
                }

                for (uint32_t j = thread_idx; j < kNumRanks; j += kNumDispatchThreads)
                    *workspace.get_expert_recv_count_ptr(j, i) = 0;
                __syncwarp();

                for (uint32_t j = thread_idx; j < num_recv_m_blocks; j += kNumDispatchThreads) {
                    *workspace.get_l1_arrival_count_ptr(cleanup_pool_block_offset + j) = 0;
                    if constexpr (!kSkipL1ReadyNotify && !kSkipL2ReadyMask)
                        *workspace.get_l2_arrival_mask_ptr(cleanup_pool_block_offset + j) = 0;
                }
                __syncwarp();
            }
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
        if constexpr (kNumDispatchWarps == 0) {
            return;
        } else {
        cutlass::arch::warpgroup_reg_dealloc<kNumDispatchRegisters>();

        const unsigned long long dispatch_total_start = phase_profile_clock();

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

        // Count tokens per expert
        read_topk_idx([&](const uint32_t& token_topk_idx, const int& expert_idx) {
            atomicAdd_block(smem_expert_count + expert_idx, 1);
        });
        ptx::sync_aligned(kNumDispatchThreads, kDispatchBarrierIdx);

        // Stake out per-expert SM offsets via global atomic
        #pragma unroll
        for (uint32_t i = thread_idx; i < kNumExperts; i += kNumDispatchThreads) {
            const uint64_t send_value = (1ull << 32) | static_cast<uint64_t>(smem_expert_count[i]);
            smem_expert_count[i] = static_cast<uint32_t>(
                ptx::atomic_add(workspace.get_expert_send_count_ptr(i), send_value));
        }
        ptx::sync_aligned(kNumDispatchThreads, kDispatchBarrierIdx);

        // Write source token-topk indices to remote ranks
        read_topk_idx([&](const uint32_t& token_topk_idx, const int& expert_idx) {
            const auto dst_rank_idx = expert_idx / kNumExpertsPerRank;
            const auto dst_slot_idx = atomicAdd_block(smem_expert_count + expert_idx, 1);
            const auto dst_ptr = workspace.get_src_token_topk_idx_ptr(
                expert_idx % kNumExpertsPerRank, sym_buffer.rank_idx, dst_slot_idx);
            *sym_buffer.map(dst_ptr, dst_rank_idx) = token_topk_idx;
        });

        comm::grid_sync<kNumSMs, kDispatchGridSyncIndex>(
            workspace, sm_idx, thread_idx,
            [=]() { ptx::sync_aligned(kNumDispatchThreads, kDispatchBarrierIdx); }
        );

        if (sm_idx == 0 and thread_idx < kNumActiveDispatchThreads) {
            #pragma unroll
            for (uint32_t i = thread_idx; i < kNumExperts; i += kNumActiveDispatchThreads) {
                const auto dst_rank_idx = i / kNumExpertsPerRank;
                const auto dst_local_expert_idx = i % kNumExpertsPerRank;
                const auto expert_status = *workspace.get_expert_send_count_ptr(i);
                *sym_buffer.map(
                    workspace.get_expert_recv_count_ptr(sym_buffer.rank_idx, dst_local_expert_idx),
                    dst_rank_idx) = expert_status & 0xffffffff;
                ptx::atomic_add_sys(
                    sym_buffer.map(workspace.get_expert_recv_count_sum_ptr(dst_local_expert_idx), dst_rank_idx),
                    expert_status);
            }
        }
        ptx::sync_aligned(kNumDispatchThreads, kDispatchBarrierIdx);

        comm::nvlink_barrier<kNumRanks, kNumSMs, kNumDispatchThreads,
                             kDispatchGridSyncIndex, kBeforeDispatchPullBarrierTag>(
            workspace, sym_buffer, sm_idx, thread_idx,
            [=]() { ptx::sync_aligned(kNumDispatchThreads, kDispatchBarrierIdx); },
            false, true);

        // Sync with epilogue warps before pulling tokens
        ptx::sync_unaligned(kNumDispatchThreads + kNumEpilogueThreads, kDispatchWithEpilogueBarrierIdx);
        const unsigned long long dispatch_pull_start = phase_profile_clock();

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

                if (old_expert_idx != current_expert_idx) {
                    old_expert_idx = current_expert_idx;
                    #pragma unroll
                    for (uint32_t i = 0; i < kNumRanksPerLane; ++ i) {
                        const uint32_t j = i * 32 + lane_idx;
                        stored_rank_count[i] = j < kNumRanks ?
                            static_cast<uint32_t>(*workspace.get_expert_recv_count_ptr(j, current_expert_idx)) : 0;
                    }
                }

                // Round-robin rank selection (identical to SM100)
                uint32_t current_rank_in_expert_idx;
                uint32_t remaining[kNumRanksPerLane];
                #pragma unroll
                for (uint32_t i = 0; i < kNumRanksPerLane; ++ i)
                    remaining[i] = stored_rank_count[i];
                uint32_t offset = 0;
                uint32_t token_idx_in_expert = token_idx - expert_start_idx;
                uint32_t slot_idx = token_idx_in_expert;
                uint32_t token_idx_in_rank;
                while (true) {
                    uint32_t num_actives_in_lane = 0;
                    uint32_t min_in_lane = 0xffffffff;
                    #pragma unroll
                    for (uint32_t i = 0; i < kNumRanksPerLane; ++ i) {
                        num_actives_in_lane += remaining[i] > 0;
                        if (remaining[i] > 0)
                            min_in_lane = cute::min(min_in_lane, remaining[i]);
                    }
                    const uint32_t num_active_ranks = __reduce_add_sync(0xffffffff, num_actives_in_lane);
                    const uint32_t length = __reduce_min_sync(0xffffffff, min_in_lane);

                    const uint32_t num_round_tokens = length * num_active_ranks;
                    if (slot_idx < num_round_tokens) {
                        const uint32_t slot_idx_in_round = slot_idx % num_active_ranks;
                        uint32_t num_seen_ranks = 0;
                        current_rank_in_expert_idx = 0;
                        #pragma unroll
                        for (uint32_t i = 0; i < kNumRanksPerLane; ++ i) {
                            const uint32_t mask = __ballot_sync(0xffffffff, remaining[i] > 0);
                            const uint32_t num_active_lanes = __popc(mask);
                            if (slot_idx_in_round >= num_seen_ranks and slot_idx_in_round < num_seen_ranks + num_active_lanes)
                                current_rank_in_expert_idx = i * 32 + __fns(mask, 0, slot_idx_in_round - num_seen_ranks + 1);
                            num_seen_ranks += num_active_lanes;
                        }
                        token_idx_in_rank = offset + (slot_idx / num_active_ranks);
                        break;
                    }
                    slot_idx -= num_round_tokens;
                    offset += length;
                    #pragma unroll
                    for (uint32_t i = 0; i < kNumRanksPerLane; ++ i)
                        remaining[i] -= cute::min(remaining[i], length);
                }

                const uint32_t src_token_topk_idx = *workspace.get_src_token_topk_idx_ptr(
                    current_expert_idx, current_rank_in_expert_idx, token_idx_in_rank);
                const uint32_t src_token_idx = src_token_topk_idx / kNumTopk;
                const uint32_t src_topk_idx  = src_token_topk_idx % kNumTopk;

                // TMA pull token data into SMEM
                if (cute::elect_one_sync()) {
                    ptx::tma_load_1d(
                        pull_buffer.get_base_ptr(),
                        sym_buffer.map(input_token_buffer.get_data_buffer(src_token_idx).get_base_ptr(),
                                       current_rank_in_expert_idx),
                        pull_mbarrier, kHidden);
                }
                __syncwarp();

                // Copy SF: per-128 K floats, written linearly (no UTCCP transpose).
                constexpr uint32_t kNumSFFloats = kHidden / 128;
                DG_STATIC_ASSERT(kNumSFFloats > 0 and kHidden % 128 == 0, "Invalid SF");
                const auto remote_sf_ptr = sym_buffer.map(
                    input_sf_buffer.get_data_buffer(src_token_idx).get_base_ptr<float>(),
                    current_rank_in_expert_idx);
                const auto local_sf_ptr  = l1_sf_buffer.get_base_ptr<float>();
                const uint32_t sf_pool_token_idx = expert_pool_block_offset * BLOCK_M + token_idx_in_expert;
                #pragma unroll
                for (uint32_t i = 0; i < math::constexpr_ceil_div(kNumSFFloats, 32u); ++ i) {
                    const uint32_t j = i * 32 + lane_idx;
                    if (j < kNumSFFloats)
                        local_sf_ptr[j * kNumPaddedSFPoolTokens + sf_pool_token_idx] = remote_sf_ptr[j];
                }
                __syncwarp();

                const uint32_t pool_token_idx = expert_pool_block_offset * BLOCK_M + token_idx_in_expert;
                if (cute::elect_one_sync()) {
                    const auto weight = *sym_buffer.map(
                        input_topk_weights_buffer.get_base_ptr<float>() + src_token_topk_idx,
                        current_rank_in_expert_idx);
                    *l1_topk_weights_buffer.get_data_buffer(pool_token_idx).get_base_ptr<float>() = weight;

                    ptx::mbarrier_arrive_and_set_tx(pull_mbarrier, kHidden);
                    ptx::mbarrier_wait_and_flip_phase(pull_mbarrier, pull_mbarrier_phase);

                    ptx::tma_store_1d(
                        l1_token_buffer.get_data_buffer(pool_token_idx).get_base_ptr(),
                        pull_buffer.get_base_ptr(), pull_buffer.get_num_bytes());

                    *workspace.get_token_src_metadata_ptr(pool_token_idx) =
                        {current_rank_in_expert_idx, src_token_idx, src_topk_idx};

                    cute::tma_store_arrive();
                    ptx::tma_store_wait<0>();
                    ptx::red_add_rel(
                        workspace.get_l1_arrival_count_ptr(expert_pool_block_offset + token_idx_in_expert / BLOCK_M), 1);
                }
                __syncwarp();
            }



            // Cleanup workspace, overlapping with combine
            const unsigned long long dispatch_pull_end = phase_profile_clock();
            if (lane_idx == 0) {
                phase_profile_record(kProfileDispatchPull, dispatch_pull_end - dispatch_pull_start);
                phase_profile_record(kProfileDispatchTotal, dispatch_pull_end - dispatch_total_start);
            }
        }
        ptx::sync_unaligned(kNumDispatchThreads + kNumEpilogueThreads, kDispatchWithEpilogueBarrierIdx);

            cleanup_workspace();
            comm::nvlink_barrier<kNumRanks, kNumSMs, kNumDispatchThreads,
                                 kDispatchGridSyncIndex, kAfterWorkspaceCleanBarrierTag>(
                workspace, sym_buffer, sm_idx, thread_idx,
                [=]() { ptx::sync_aligned(kNumDispatchThreads, kDispatchBarrierIdx); },
                true, false);


    // =====================================================================
    // ROLE 2: GEMM TMA LOAD warps (load A+SFA, B+SFB)
    //   Warps inside `kNumNonEpilogueThreads` (= 4 warps): warp 0 loads
    //   A + SFA, warp 1 loads B + SFB, warps 2..3 idle.
    // =====================================================================
        }
    } else if (warp_idx == kNumDispatchWarps) {
        cutlass::arch::warpgroup_reg_dealloc<kNumNonEpilogueRegisters>();

        for_each_selected_block([&](const auto& block_phase,
                                     const uint32_t& local_expert_idx,
                                     const uint32_t& num_k_blocks,
                                     const uint32_t& m_block_idx, const uint32_t& n_block_idx) {
            using BlockPhaseTag = std::remove_cv_t<std::remove_reference_t<decltype(block_phase)>>;
            constexpr bool kBlockIsL2 = BlockPhaseTag::value == sched::BlockPhase::Linear2;
            const auto tensor_map_a_ptr = [&]() {
                if constexpr (kBlockIsL2)
                    return &tensor_map_l2_acts;
                else
                    return &tensor_map_l1_acts;
            }();
            const auto tensor_map_sfa_ptr = [&]() {
                if constexpr (kBlockIsL2)
                    return &tensor_map_l2_acts_sf;
                else
                    return &tensor_map_l1_acts_sf;
            }();

            const uint32_t pool_block_idx = scheduler.get_current_pool_block_offset() + m_block_idx;
            const uint32_t valid_m = scheduler.template get_valid_m<false>();
            const bool has_valid_m = valid_m > 0;

            // Wait for the pool to be ready. Cluster peers can be dummy CTAs for
            // the tail M unit when an expert has an odd number of M blocks.
            const unsigned long long ready_wait_start = phase_profile_clock();
            if (has_valid_m) {
                if constexpr (!kBlockIsL2) {
                    const auto ptr = workspace.get_l1_arrival_count_ptr(pool_block_idx);
                    const auto expected = valid_m;
                    while (ptx::ld_acq(ptr) != expected);
                } else if constexpr (!kSkipL2ReadyMask) {
                    // Each L1 N block publishes one ready event per active M warpgroup.
                    constexpr uint32_t kNumL1BlockNs = L1_SHAPE_N / BLOCK_N;
                    if constexpr (kL2ArrivalCounter) {
                        const auto ptr = reinterpret_cast<const uint32_t*>(
                            workspace.get_l2_arrival_mask_ptr(pool_block_idx));
                        const uint32_t active_m_wgs = math::ceil_div(valid_m, WG_BLOCK_M);
                        const uint32_t expected = kNumL1BlockNs * active_m_wgs;
                        while (ptx::ld_acq(ptr) != expected);
                    } else {
                        const auto ptr = workspace.get_l2_arrival_mask_ptr(pool_block_idx);
                        const uint64_t expected = (kNumL1BlockNs >= 64)
                            ? ~0ull : ((1ull << kNumL1BlockNs) - 1ull);
                        while (ptx::ld_acq_gpu(ptr) != expected);
                    }
                }
            }
            const unsigned long long ready_wait_end = phase_profile_clock();
            if constexpr (kBlockIsL2) {
                if (has_valid_m and lane_idx == 0)
                    phase_profile_record(kProfileL2ReadyWait, ready_wait_end - ready_wait_start);
            }
            for (uint32_t k_block_idx = 0; k_block_idx < num_k_blocks; advance_pipeline(k_block_idx)) {
                empty_barriers[stage_idx]->wait(phase ^ 1);

                if (cute::elect_one_sync()) {
                    if (has_valid_m) {
                    const uint32_t m_idx = pool_block_idx * BLOCK_M;
                    const uint32_t k_idx = k_block_idx * BLOCK_K;

                    // TMA load A
                    tma::copy<BLOCK_K, LOAD_BLOCK_M, kSwizzleAMode, a_dtype_t>(
                        tensor_map_a_ptr, full_barriers[stage_idx], smem_a[stage_idx],
                        k_idx, m_idx, 1);

                    // TMA load SFA
                    if constexpr (!kBlockIsL2) {
                        // L1 SFA per-128: load (BLOCK_M, 1) at K=k_block_idx
                        tma::copy<BLOCK_M, 1, 0, float>(
                            tensor_map_sfa_ptr, full_barriers[stage_idx], smem_sfa[stage_idx],
                            m_idx, k_block_idx, 1);
                        full_barriers[stage_idx]->arrive_and_expect_tx(
                            SMEM_A_SIZE_PER_STAGE + BLOCK_M * sizeof(float));
                    } else {
                        // L2 SFA per-64: descriptor box is (block_mn, 1) (see make_tma_sf_desc),
                        // so we must issue two single-group TMAs and place them at smem offsets
                        // 0 and BLOCK_M to match math's load offsets (`+ 0 * BLOCK_M` / `+ 1 * BLOCK_M`).
                        tma::copy<BLOCK_M, 1, 0, float>(
                            tensor_map_sfa_ptr, full_barriers[stage_idx], smem_sfa[stage_idx],
                            m_idx, k_block_idx * 2, 1);
                        tma::copy<BLOCK_M, 1, 0, float>(
                            tensor_map_sfa_ptr, full_barriers[stage_idx],
                            smem_sfa[stage_idx] + kL2SFAHalfStride,
                            m_idx, k_block_idx * 2 + 1, 1);
                        full_barriers[stage_idx]->arrive_and_expect_tx(
                            SMEM_A_SIZE_PER_STAGE + 2 * BLOCK_M * sizeof(float));
                    }
                    } else {
                        full_barriers[stage_idx]->arrive();
                    }
                }
                __syncwarp();
                dequant_loaded_b_stage(stage_idx, phase, lane_idx);
            }
        });

    } else if (warp_idx == kNumDispatchWarps + 1) {
        cutlass::arch::warpgroup_reg_dealloc<kNumNonEpilogueRegisters>();

        for_each_selected_block([&](const auto& block_phase,
                                     const uint32_t& local_expert_idx,
                                     const uint32_t& num_k_blocks,
                                     const uint32_t& m_block_idx, const uint32_t& n_block_idx) {
            using BlockPhaseTag = std::remove_cv_t<std::remove_reference_t<decltype(block_phase)>>;
            constexpr bool kBlockIsL2 = BlockPhaseTag::value == sched::BlockPhase::Linear2;
            const auto tensor_map_b_ptr = [&]() {
                if constexpr (kBlockIsL2)
                    return &tensor_map_l2_weights;
                else
                    return &tensor_map_l1_weights;
            }();
            constexpr uint32_t shape_n = kBlockIsL2 ? L2_SHAPE_N : L1_SHAPE_N;

            for (uint32_t k_block_idx = 0; k_block_idx < num_k_blocks; advance_pipeline(k_block_idx)) {
                empty_barriers[stage_idx]->wait(phase ^ 1);

                const uint32_t n_idx = local_expert_idx * shape_n + n_block_idx * BLOCK_N;
                // NVFP4 fused B+scale layout stores 64B packed FP4 + 16B
                // UE4M3 scale per BK128 row.
                const uint32_t k_idx = k_block_idx * B_LOAD_BYTES_PER_ROW;
                if (cute::elect_one_sync()) {
                    tma::copy<B_LOAD_BYTES_PER_ROW, LOAD_BLOCK_N, 0, b_dtype_t>(
                        tensor_map_b_ptr, full_barriers[stage_idx],
                        kPackedBScratch ? smem_packed_b[stage_idx] : smem_b[stage_idx],
                        k_idx, n_idx, kClusterSize);
                    full_barriers[stage_idx]->arrive_and_expect_tx(SMEM_B_LOAD_SIZE_PER_STAGE);
                }
                __syncwarp();
                dequant_loaded_b_stage(stage_idx, phase, 32u + lane_idx);
            }
        });

    } else if (warp_idx < kNumDispatchWarps + kNumMMANonEpilogueWarps) {
        // Idle non-epilogue warps (kNumDispatchWarps+2, +3). They must still
        // participate in the warpgroup-collective `setmaxnreg.dec.sync.aligned`
        // so that the math warpgroup's `warpgroup_reg_alloc` can succeed.
        cutlass::arch::warpgroup_reg_dealloc<kNumNonEpilogueRegisters>();

        if constexpr (kLoaderDequant) {
            const uint32_t non_epilogue_warp_idx = warp_idx - kNumDispatchWarps;
            const uint32_t non_epilogue_thread_idx = non_epilogue_warp_idx * 32 + lane_idx;
            for_each_selected_block([&](const auto& block_phase,
                                         const uint32_t&, const uint32_t& num_k_blocks,
                                         const uint32_t&, const uint32_t&) {
                for (uint32_t k_block_idx = 0; k_block_idx < num_k_blocks; advance_pipeline(k_block_idx)) {
                    const unsigned long long loader_dequant_start = phase_profile_clock();
                    dequant_loaded_b_stage(stage_idx, phase, non_epilogue_thread_idx);
                    const unsigned long long loader_dequant_end = phase_profile_clock();
                    if (non_epilogue_thread_idx == 64u)
                        phase_profile_record(kProfileLoaderDequant, loader_dequant_end - loader_dequant_start);
                    __syncwarp();
                }
            });
        }
    } else if (warp_idx >= kNumDispatchWarps + kNumMMANonEpilogueWarps) {
    // =====================================================================
    // ROLE 3: MATH WARPGROUPS (WGMMA + epilogue + combine)
    // =====================================================================
        cutlass::arch::warpgroup_reg_alloc<kNumEpilogueRegisters>();

        const uint32_t epilogue_warp_idx  = warp_idx - (kNumDispatchWarps + kNumMMANonEpilogueWarps);
        const uint32_t epilogue_wg_idx    = epilogue_warp_idx / 4;
        const uint32_t epilogue_thread_idx = epilogue_warp_idx * 32 + lane_idx;
        const uint32_t warp_idx_in_wg     = epilogue_warp_idx % 4;

        const auto arrive_empty_barrier = [&](const uint32_t& s) {
            if constexpr (kClusterSize == 1) {
                if (lane_idx == 0)
                    empty_barriers[s]->arrive();
            } else {
                if (lane_idx < kClusterSize)
                    empty_barriers[s]->arrive(lane_idx);
            }
        };

        const auto notify_l1_ready = [&](const uint32_t& ready_pool_block_idx,
                                         const uint32_t& ready_n_block_idx) {
            const unsigned long long notify_start = phase_profile_clock();
            if constexpr (kSkipL1ReadyNotify) {
                (void)ready_pool_block_idx;
                (void)ready_n_block_idx;
            } else if constexpr (kL2ArrivalCounter) {
                if (warp_idx_in_wg == 0 and cute::elect_one_sync()) {
                    ptx::red_add_rel(
                        reinterpret_cast<uint32_t*>(workspace.get_l2_arrival_mask_ptr(ready_pool_block_idx)),
                        1u);
                }
            } else {
                if (epilogue_warp_idx == 0 and cute::elect_one_sync()) {
                    ptx::red_or_rel_gpu(
                        workspace.get_l2_arrival_mask_ptr(ready_pool_block_idx),
                        1ull << ready_n_block_idx);
                }
            }
            __syncwarp();
            const unsigned long long notify_end = phase_profile_clock();
            if (epilogue_warp_idx == 0 and lane_idx == 0)
                phase_profile_record(kProfileL1ReadyNotify, notify_end - notify_start);
        };

        const auto cleanup_workspace_from_epilogue = [&]() {
            DG_STATIC_ASSERT(kNumSMs > 1, "Invalid SM count");
            if (sm_idx == 0) {
                #pragma unroll
                for (uint32_t i = epilogue_thread_idx; i < kNumExperts; i += kNumEpilogueThreads)
                    *workspace.get_expert_send_count_ptr(i) = 0;
            } else {
                for (uint32_t i = sm_idx - 1; i < kNumExpertsPerRank; i += kNumSMs - 1) {
                    const auto num_recv_tokens = static_cast<uint32_t>(
                        *workspace.get_expert_recv_count_sum_ptr(i));
                    const auto num_recv_m_blocks = math::ceil_div(num_recv_tokens, BLOCK_M);
                    const auto cleanup_pool_block_offset = scheduler.get_pool_block_offset(i);

                    ptx::sync_aligned(kNumEpilogueThreads, kEpilogueFullBarrierIdx);
                    if (epilogue_thread_idx == 0) {
                        *workspace.get_expert_recv_count_sum_ptr(i) = 0;
                        if (cumulative_local_expert_recv_stats != nullptr)
                            ptx::red_add(cumulative_local_expert_recv_stats + i, static_cast<int>(num_recv_tokens));
                    }

                    for (uint32_t j = epilogue_thread_idx; j < kNumRanks; j += kNumEpilogueThreads)
                        *workspace.get_expert_recv_count_ptr(j, i) = 0;

                    for (uint32_t j = epilogue_thread_idx; j < num_recv_m_blocks; j += kNumEpilogueThreads) {
                        *workspace.get_l1_arrival_count_ptr(cleanup_pool_block_offset + j) = 0;
                        if constexpr (!kSkipL1ReadyNotify && !kSkipL2ReadyMask)
                            *workspace.get_l2_arrival_mask_ptr(cleanup_pool_block_offset + j) = 0;
                    }
                    ptx::sync_aligned(kNumEpilogueThreads, kEpilogueFullBarrierIdx);
                }
            }
        };

        const auto finish_no_dispatch_cleanup = [&]() {

        };

        // WGMMA-output register layout helpers
        const uint32_t row_idx = lane_idx / 4;
        const uint32_t col_idx = lane_idx % 4;
        const uint32_t r_0 = warp_idx_in_wg * 16 + row_idx;
        const uint32_t r_1 = r_0 + 8;

        DG_STATIC_ASSERT(kSplitNWarpgroups || (BLOCK_M % kNumEpilogueWarpgroups == 0), "Invalid block M");
        if constexpr (kSplitNWarpgroups) {
            DG_STATIC_ASSERT(WG_BLOCK_M == L1WGMMA::M and WG_BLOCK_N == L1WGMMA::N,
                             "Split-N WGs must each run one M64N128 WGMMA per K-block");
        } else if constexpr (kSerialNWarpgroups) {
            DG_STATIC_ASSERT(WG_BLOCK_M == L1WGMMA::M and WG_BLOCK_N == L1WGMMA::N,
                             "Serial-N path runs two M64N128 WGMMAs per K-block");
        } else {
            DG_STATIC_ASSERT(WG_BLOCK_M == L1WGMMA::M, "Each warpgroup must run exactly one WGMMA per K-block");
        }

        // Sync with dispatch
        ptx::sync_unaligned(kNumDispatchThreads + kNumEpilogueThreads, kDispatchWithEpilogueBarrierIdx);

        const unsigned long long math_loop_start = phase_profile_clock();

        for_each_selected_block([&](const auto& block_phase,
                                     const uint32_t& local_expert_idx,
                                     const uint32_t& num_k_blocks,
                                     const uint32_t& m_block_idx, const uint32_t& n_block_idx) {
            const uint32_t valid_m = scheduler.template get_valid_m<false>();
            const uint32_t pool_block_idx = scheduler.get_current_pool_block_offset() + m_block_idx;
            const uint32_t m_idx = pool_block_idx * BLOCK_M;
            const uint32_t wg_n_idx = kSplitNWarpgroups ? epilogue_wg_idx * WG_BLOCK_N : 0;
            const uint32_t wg_l1_out_n_idx = kSplitNWarpgroups ? epilogue_wg_idx * WG_L1_OUT_BLOCK_N : 0;
            const uint32_t n_idx = n_block_idx * BLOCK_N + wg_n_idx;
            const uint32_t row_block_offset = kSplitNWarpgroups ? 0 : epilogue_wg_idx * WG_BLOCK_M;
            const uint32_t row_offset_r0 = row_block_offset + r_0;
            const uint32_t row_offset_r1 = row_block_offset + r_1;
            const bool valid_r0 = row_offset_r0 < valid_m;
            const bool valid_r1 = row_offset_r1 < valid_m;
            using BlockPhaseTag = std::remove_cv_t<std::remove_reference_t<decltype(block_phase)>>;
            constexpr bool kBlockIsL2 = BlockPhaseTag::value == sched::BlockPhase::Linear2;
            const float l2_global_scale = l2_global_scales == nullptr ? 1.0f : __ldg(l2_global_scales + local_expert_idx);
            const auto cast_l2_scaled_bf16_pair = [&](float x, float y) -> uint32_t {
                x *= l2_global_scale;
                y *= l2_global_scale;
                return math::cast_into_bf16_and_pack(x, y);
            };


            if constexpr (kLoaderDequant && (!kSplitNWarpgroups)) {
                if (row_block_offset >= valid_m) {
                    for (uint32_t k_block_idx = 0; k_block_idx < num_k_blocks; advance_pipeline(k_block_idx)) {
                        dequant_barriers[stage_idx]->wait(phase);
                        arrive_empty_barrier(stage_idx);
                        __syncwarp();
                    }
                    if constexpr (!kL2ArrivalCounter || kBlockIsL2)
                        ptx::sync_aligned(kNumEpilogueThreads, kEpilogueFullBarrierIdx);
                    return;
                }
            }


            if constexpr (kSerialNWarpgroups) {
                using WGMMA = L1WGMMA;
                constexpr uint32_t kAccumPerThread = WGMMA::kNumAccum;
                constexpr uint32_t kNumSerialN = 2;
                float final_accum[kNumSerialN][kAccumPerThread] = {};
                float accum[kAccumPerThread];

                for (uint32_t k_block_idx = 0; k_block_idx < num_k_blocks; advance_pipeline(k_block_idx)) {
                    full_barriers[stage_idx]->wait(phase);

                    float scale_a_0_lo, scale_a_1_lo;
                    float scale_a_0_hi, scale_a_1_hi;
                    if (block_phase == sched::BlockPhase::Linear1) {
                        scale_a_0_lo = ptx::ld_shared(smem_sfa[stage_idx] + row_offset_r0);
                        scale_a_1_lo = ptx::ld_shared(smem_sfa[stage_idx] + row_offset_r1);
                    } else {
                        scale_a_0_lo = ptx::ld_shared(smem_sfa[stage_idx] + row_offset_r0);
                        scale_a_1_lo = ptx::ld_shared(smem_sfa[stage_idx] + row_offset_r1);
                        scale_a_0_hi = ptx::ld_shared(smem_sfa[stage_idx] + kL2SFAHalfStride + row_offset_r0);
                        scale_a_1_hi = ptx::ld_shared(smem_sfa[stage_idx] + kL2SFAHalfStride + row_offset_r1);
                    }

                    constexpr uint32_t kL1SFKBlocks   = kHidden / 128;
                    constexpr uint32_t kL2SFKBlocks   = kIntermediateHidden / 128;
                    constexpr uint32_t kL1SFGateBlks  = kIntermediateHidden / 128;
                    constexpr uint32_t kL1SFPerExpert = (kIntermediateHidden * 2 / 128) * kL1SFKBlocks;
                    constexpr uint32_t kL2SFPerExpert = (kHidden / 128) * kL2SFKBlocks;

                    #pragma unroll
                    for (uint32_t serial_n_idx = 0; serial_n_idx < kNumSerialN; ++serial_n_idx) {
                        const uint32_t serial_wg_n_idx = serial_n_idx * WG_BLOCK_N;
                        float gate_sf = 0.0f, up_sf = 0.0f, l2_sf = 0.0f;
                        if (block_phase == sched::BlockPhase::Linear1) {
                            gate_sf = 1.0f;
                            up_sf = 1.0f;

                            #pragma unroll
                            for (uint32_t i = 0; i < kAccumPerThread; ++ i) ptx::warpgroup_fence_operand(accum[i]);
                            ptx::warpgroup_arrive();
                            #pragma unroll
                            for (uint32_t k = 0; k < BLOCK_K / WGMMA::K; ++ k) {
                                auto desc_a = mma::sm90::make_smem_desc(
                                    smem_a[stage_idx] + row_block_offset * BLOCK_K + k * WGMMA::K, 1);
                                auto desc_b = mma::sm90::make_smem_desc(
                                    smem_b[stage_idx] + serial_wg_n_idx * BLOCK_K + k * WGMMA::K, 1);
                                WGMMA::wgmma(desc_a, desc_b, accum, k);
                            }
                            ptx::warpgroup_commit_batch();
                            #pragma unroll
                            for (uint32_t i = 0; i < kAccumPerThread; ++ i) ptx::warpgroup_fence_operand(accum[i]);
                            ptx::warpgroup_wait<0>();

                            #pragma unroll
                            for (uint32_t i = 0; i < kAccumPerThread / 4; ++ i) {
                                const float sb = (i & 1u) ? up_sf : gate_sf;
                                final_accum[serial_n_idx][i*4+0] += scale_a_0_lo * sb * accum[i*4+0];
                                final_accum[serial_n_idx][i*4+1] += scale_a_0_lo * sb * accum[i*4+1];
                                final_accum[serial_n_idx][i*4+2] += scale_a_1_lo * sb * accum[i*4+2];
                                final_accum[serial_n_idx][i*4+3] += scale_a_1_lo * sb * accum[i*4+3];
                            }
                        } else {
                            l2_sf = 1.0f;

                            #pragma unroll
                            for (uint32_t i = 0; i < kAccumPerThread; ++ i) ptx::warpgroup_fence_operand(accum[i]);
                            ptx::warpgroup_arrive();
                            #pragma unroll
                            for (uint32_t k = 0; k < (BLOCK_K / 2) / WGMMA::K; ++ k) {
                                auto desc_a = mma::sm90::make_smem_desc(
                                    smem_a[stage_idx] + row_block_offset * BLOCK_K + k * WGMMA::K, 1);
                                auto desc_b = mma::sm90::make_smem_desc(
                                    smem_b[stage_idx] + serial_wg_n_idx * BLOCK_K + k * WGMMA::K, 1);
                                WGMMA::wgmma(desc_a, desc_b, accum, k);
                            }
                            ptx::warpgroup_commit_batch();
                            #pragma unroll
                            for (uint32_t i = 0; i < kAccumPerThread; ++ i) ptx::warpgroup_fence_operand(accum[i]);
                            ptx::warpgroup_wait<0>();

                            #pragma unroll
                            for (uint32_t i = 0; i < kAccumPerThread / 4; ++ i) {
                                final_accum[serial_n_idx][i*4+0] += scale_a_0_lo * l2_sf * accum[i*4+0];
                                final_accum[serial_n_idx][i*4+1] += scale_a_0_lo * l2_sf * accum[i*4+1];
                                final_accum[serial_n_idx][i*4+2] += scale_a_1_lo * l2_sf * accum[i*4+2];
                                final_accum[serial_n_idx][i*4+3] += scale_a_1_lo * l2_sf * accum[i*4+3];
                            }

                            #pragma unroll
                            for (uint32_t i = 0; i < kAccumPerThread; ++ i) ptx::warpgroup_fence_operand(accum[i]);
                            ptx::warpgroup_arrive();
                            #pragma unroll
                            for (uint32_t k = 0; k < (BLOCK_K / 2) / WGMMA::K; ++ k) {
                                const uint32_t k_off = (BLOCK_K / 2) + k * WGMMA::K;
                                auto desc_a = mma::sm90::make_smem_desc(
                                    smem_a[stage_idx] + row_block_offset * BLOCK_K + k_off, 1);
                                auto desc_b = mma::sm90::make_smem_desc(
                                    smem_b[stage_idx] + serial_wg_n_idx * BLOCK_K + k_off, 1);
                                WGMMA::wgmma(desc_a, desc_b, accum, k);
                            }
                            ptx::warpgroup_commit_batch();
                            #pragma unroll
                            for (uint32_t i = 0; i < kAccumPerThread; ++ i) ptx::warpgroup_fence_operand(accum[i]);
                            ptx::warpgroup_wait<0>();

                            #pragma unroll
                            for (uint32_t i = 0; i < kAccumPerThread / 4; ++ i) {
                                final_accum[serial_n_idx][i*4+0] += scale_a_0_hi * l2_sf * accum[i*4+0];
                                final_accum[serial_n_idx][i*4+1] += scale_a_0_hi * l2_sf * accum[i*4+1];
                                final_accum[serial_n_idx][i*4+2] += scale_a_1_hi * l2_sf * accum[i*4+2];
                                final_accum[serial_n_idx][i*4+3] += scale_a_1_hi * l2_sf * accum[i*4+3];
                            }
                        }
                    }

                    arrive_empty_barrier(stage_idx);
                    __syncwarp();
                }

                if (row_block_offset >= valid_m) {
                    ptx::sync_aligned(kNumEpilogueThreads, kEpilogueFullBarrierIdx);
                    return;
                }

                if (block_phase == sched::BlockPhase::Linear1) {
                    const float l1_global_scale = l1_global_scales == nullptr ? 1.0f : __ldg(l1_global_scales + local_expert_idx);
                constexpr uint32_t kNumPairs = kAccumPerThread / 8;
                    #pragma unroll
                    for (uint32_t serial_n_idx = 0; serial_n_idx < kNumSerialN; ++serial_n_idx) {
                        const uint32_t serial_l1_out_n_idx = serial_n_idx * WG_L1_OUT_BLOCK_N;
                        float swiglu_r0[kNumPairs][2];
                        float swiglu_r1[kNumPairs][2];
                        float amax_r0 = 0.0f, amax_r1 = 0.0f;

                        #pragma unroll
                        for (uint32_t p = 0; p < kNumPairs; ++ p) {
                            const uint32_t gate = 2 * p, up = 2 * p + 1;
                            auto clamp_gate = [](float& x) {
                                if constexpr (kActivationClamp != cute::numeric_limits<float>::infinity())
                                    x = cute::min(x, kActivationClamp);
                            };
                            auto clamp_up = [](float& x) {
                                if constexpr (kActivationClamp != cute::numeric_limits<float>::infinity())
                                    x = cute::min(cute::max(x, -kActivationClamp), kActivationClamp);
                            };
                            float g_r0_c0 = final_accum[serial_n_idx][gate*4 + 0] * l1_global_scale; clamp_gate(g_r0_c0);
                            float g_r0_c1 = final_accum[serial_n_idx][gate*4 + 1] * l1_global_scale; clamp_gate(g_r0_c1);
                            float g_r1_c0 = final_accum[serial_n_idx][gate*4 + 2] * l1_global_scale; clamp_gate(g_r1_c0);
                            float g_r1_c1 = final_accum[serial_n_idx][gate*4 + 3] * l1_global_scale; clamp_gate(g_r1_c1);
                            float u_r0_c0 = final_accum[serial_n_idx][up*4   + 0] * l1_global_scale; clamp_up(u_r0_c0);
                            float u_r0_c1 = final_accum[serial_n_idx][up*4   + 1] * l1_global_scale; clamp_up(u_r0_c1);
                            float u_r1_c0 = final_accum[serial_n_idx][up*4   + 2] * l1_global_scale; clamp_up(u_r1_c0);
                            float u_r1_c1 = final_accum[serial_n_idx][up*4   + 3] * l1_global_scale; clamp_up(u_r1_c1);
                            auto silu = [](float x) -> float {
                                const float e = kFastMath ? __expf(-x) : expf(-x);
                                const float sig = kFastMath ? math::fast_rcp(1.0f + e) : 1.0f / (1.0f + e);
                                return x * sig;
                            };
                            if (valid_r0) {
                                swiglu_r0[p][0] = silu(g_r0_c0) * u_r0_c0;
                                swiglu_r0[p][1] = silu(g_r0_c1) * u_r0_c1;
                                amax_r0 = cute::max(amax_r0, cute::max(cute::abs(swiglu_r0[p][0]), cute::abs(swiglu_r0[p][1])));
                            } else {
                                swiglu_r0[p][0] = 0.0f;
                                swiglu_r0[p][1] = 0.0f;
                            }
                            if (valid_r1) {
                                swiglu_r1[p][0] = silu(g_r1_c0) * u_r1_c0;
                                swiglu_r1[p][1] = silu(g_r1_c1) * u_r1_c1;
                                amax_r1 = cute::max(amax_r1, cute::max(cute::abs(swiglu_r1[p][0]), cute::abs(swiglu_r1[p][1])));
                            } else {
                                swiglu_r1[p][0] = 0.0f;
                                swiglu_r1[p][1] = 0.0f;
                            }
                        }

                        float weight_r0 = valid_r0 ? *l1_topk_weights_buffer
                            .get_data_buffer(m_idx + row_offset_r0)
                            .template get_base_ptr<float>() : 0.0f;
                        float weight_r1 = valid_r1 ? *l1_topk_weights_buffer
                            .get_data_buffer(m_idx + row_offset_r1)
                            .template get_base_ptr<float>() : 0.0f;
                        #pragma unroll
                        for (uint32_t p = 0; p < kNumPairs; ++ p) {
                            swiglu_r0[p][0] *= weight_r0;
                            swiglu_r0[p][1] *= weight_r0;
                            swiglu_r1[p][0] *= weight_r1;
                            swiglu_r1[p][1] *= weight_r1;
                        }
                        amax_r0 *= cute::abs(weight_r0);
                        amax_r1 *= cute::abs(weight_r1);
                        amax_r0 = math::warp_reduce<4, false>(amax_r0, math::ReduceMax<float>());
                        amax_r1 = math::warp_reduce<4, false>(amax_r1, math::ReduceMax<float>());

                        float sf_r0, sf_inv_r0, sf_r1, sf_inv_r1;
                        {
                            float2 amax_pair = {amax_r0, amax_r1};
                            float2 sf_pair, sf_inv_pair;
                            math::get_e4m3_sf_and_sf_inv(amax_pair, sf_pair, sf_inv_pair);
                            sf_r0 = sf_pair.x; sf_inv_r0 = sf_inv_pair.x;
                            sf_r1 = sf_pair.y; sf_inv_r1 = sf_inv_pair.y;
                        }

                        #pragma unroll
                        for (uint32_t p = 0; p < kNumPairs; ++ p) {
                            const float v00 = swiglu_r0[p][0] * sf_inv_r0;
                            const float v01 = swiglu_r0[p][1] * sf_inv_r0;
                            const float v10 = swiglu_r1[p][0] * sf_inv_r1;
                            const float v11 = swiglu_r1[p][1] * sf_inv_r1;
                            const __nv_fp8x2_e4m3 r0_pair(make_float2(v00, v01));
                            const __nv_fp8x2_e4m3 r1_pair(make_float2(v10, v11));
                            const uint32_t col = p * 8 + col_idx * 2;
                            auto* p0 = reinterpret_cast<uint16_t*>(
                                smem_cd_l1 + r_0 * L1_OUT_BLOCK_N + serial_l1_out_n_idx + col);
                            auto* p1 = reinterpret_cast<uint16_t*>(
                                smem_cd_l1 + r_1 * L1_OUT_BLOCK_N + serial_l1_out_n_idx + col);
                            if (valid_r0)
                                *p0 = r0_pair.__x;
                            if (valid_r1)
                                *p1 = r1_pair.__x;
                        }

                        if (col_idx == 0) {
                            auto sf_base_ptr = l2_sf_buffer.get_base_ptr<float>();
                            const uint32_t token_r0 = pool_block_idx * BLOCK_M + row_offset_r0;
                            const uint32_t token_r1 = pool_block_idx * BLOCK_M + row_offset_r1;
                            const uint32_t k_sf_idx = (n_block_idx * L1_OUT_BLOCK_N + serial_l1_out_n_idx) / 64u;
                            if (valid_r0)
                                sf_base_ptr[k_sf_idx * kNumPaddedSFPoolTokens + token_r0] = sf_r0;
                            if (valid_r1)
                                sf_base_ptr[k_sf_idx * kNumPaddedSFPoolTokens + token_r1] = sf_r1;
                        }
                    }

                    ptx::sync_aligned(128, kEpilogueWGBarrierStartIdx + epilogue_wg_idx);
                    if (warp_idx_in_wg == 0 and cute::elect_one_sync()) {
                        const uint32_t out_n_idx = n_block_idx * L1_OUT_BLOCK_N;
                        cute::tma_store_fence();
                        cute::SM90_TMA_STORE_2D::copy(
                            &tensor_map_l1_output,
                            smem_cd_l1,
                            out_n_idx,
                            m_idx + row_block_offset);
                        cute::tma_store_arrive();
                    }
                    __syncwarp();
                    const unsigned long long l1_tma_wait_start = phase_profile_clock();
                    ptx::tma_store_wait<0>();
                    const unsigned long long l1_tma_wait_end = phase_profile_clock();
                    if (epilogue_warp_idx == 0 and lane_idx == 0)
                        phase_profile_record(kProfileL1TMAWait, l1_tma_wait_end - l1_tma_wait_start);
                    ptx::sync_aligned(kNumEpilogueThreads, kEpilogueFullBarrierIdx);
                    notify_l1_ready(pool_block_idx, n_block_idx);
                } else {
                    constexpr uint32_t kNumRowsPerWarp = WG_BLOCK_M / 8;
                    #pragma unroll
                    for (uint32_t serial_n_idx = 0; serial_n_idx < kNumSerialN; ++serial_n_idx) {
                        const uint32_t serial_n_idx_base = n_block_idx * BLOCK_N + serial_n_idx * WG_BLOCK_N;

                        #pragma unroll
                        for (uint32_t i = 0; i < kAccumPerThread / 8; ++ i) {
                            const uint32_t chunk_lo = 2 * i, chunk_hi = 2 * i + 1;
                            auto write_pair = [&](uint32_t row, uint32_t col, uint32_t packed) {
                                auto smem_ptr = smem_cd_l2 + row * WG_BLOCK_N + col;
                                *reinterpret_cast<uint32_t*>(smem_ptr) = packed;
                            };
                            if (valid_r0) {
                                const uint32_t r0_lo = cast_l2_scaled_bf16_pair(
                                    final_accum[serial_n_idx][chunk_lo*4 + 0], final_accum[serial_n_idx][chunk_lo*4 + 1]);
                                const uint32_t r0_hi = cast_l2_scaled_bf16_pair(
                                    final_accum[serial_n_idx][chunk_hi*4 + 0], final_accum[serial_n_idx][chunk_hi*4 + 1]);
                                write_pair(r_0, chunk_lo * 8 + col_idx * 2, r0_lo);
                                write_pair(r_0, chunk_hi * 8 + col_idx * 2, r0_hi);
                            }
                            if (valid_r1) {
                                const uint32_t r1_lo = cast_l2_scaled_bf16_pair(
                                    final_accum[serial_n_idx][chunk_lo*4 + 2], final_accum[serial_n_idx][chunk_lo*4 + 3]);
                                const uint32_t r1_hi = cast_l2_scaled_bf16_pair(
                                    final_accum[serial_n_idx][chunk_hi*4 + 2], final_accum[serial_n_idx][chunk_hi*4 + 3]);
                                write_pair(r_1, chunk_lo * 8 + col_idx * 2, r1_lo);
                                write_pair(r_1, chunk_hi * 8 + col_idx * 2, r1_hi);
                            }
                        }
                        ptx::sync_aligned(128, kEpilogueWGBarrierStartIdx + epilogue_wg_idx);

                        const uint32_t row_in_warp_block = lane_idx / 16;
                        const uint32_t lane_in_row = lane_idx % 16;
                        constexpr uint32_t cols_per_lane = WG_BLOCK_N / 16;
                        #pragma unroll
                        for (uint32_t j = 0; j < kNumRowsPerWarp; ++ j) {
                            const uint32_t row_in_wg = warp_idx_in_wg * 16 + j * 2 + row_in_warp_block;
                            const uint32_t m_idx_in_block = row_block_offset + row_in_wg;
                            if (m_idx_in_block >= valid_m) break;

                            const auto src_metadata = *workspace.get_token_src_metadata_ptr(m_idx + m_idx_in_block);
                            const uint32_t dst_rank_idx = src_metadata.rank_idx;
                            const uint32_t dst_token_idx = src_metadata.token_idx;
                            const uint32_t dst_topk_idx = src_metadata.topk_idx;
                            auto smem_ptr = smem_cd_l2 + row_in_wg * WG_BLOCK_N + lane_in_row * cols_per_lane;
                            const auto dst_token = combine_token_buffer.get_rank_buffer(dst_topk_idx)
                                                   .get_data_buffer(dst_token_idx);
                            const auto packed = *reinterpret_cast<uint4*>(smem_ptr);
                            auto dst_ptr = math::advance_ptr<uint4>(
                                dst_token.get_base_ptr(),
                                serial_n_idx_base * sizeof(nv_bfloat16) + lane_in_row * sizeof(uint4));
                            *sym_buffer.map(dst_ptr, dst_rank_idx) = packed;
                        }
                        ptx::sync_aligned(kNumEpilogueThreads, kEpilogueFullBarrierIdx);
                    }
                }
                return;
            }

            // ---------------- GEMM ----------------
            using WGMMA = L1WGMMA;
            constexpr uint32_t kAccumPerThread = WGMMA::kNumAccum;  // 64 for M=64,N=128
            float final_accum[kAccumPerThread] = {};
            float accum[kAccumPerThread];
            unsigned long long block_math_full_wait = 0;
            unsigned long long block_math_dequant = 0;

            const unsigned long long block_gemm_start = phase_profile_clock();
            const auto run_default_gemm_loop = [&]() {
for (uint32_t k_block_idx = 0; k_block_idx < num_k_blocks; advance_pipeline(k_block_idx)) {
                if constexpr (kLoaderDequant) {
                    const unsigned long long dequant_wait_start = phase_profile_clock();
                    dequant_barriers[stage_idx]->wait(phase);
                    const unsigned long long dequant_wait_end = phase_profile_clock();
                    if (epilogue_warp_idx == 0 && lane_idx == 0)
                        phase_profile_record(kProfileMathDequantWait, dequant_wait_end - dequant_wait_start);
                } else {
                    const unsigned long long full_wait_start = phase_profile_clock();
                    full_barriers[stage_idx]->wait(phase);
                    const unsigned long long full_wait_end = phase_profile_clock();
                    if constexpr (kPhaseProfileRequested)
                        block_math_full_wait += full_wait_end - full_wait_start;

                    // NVFP4: expand packed FP4 (first 8KB) -> FP8 (full 16KB) in smem_b.
                    // Each math-WG thread handles one row (64 -> 128 bytes).
                    {
                        const unsigned long long dequant_start = phase_profile_clock();
                        DG_STATIC_ASSERT(kPackedBScratch,
                                         "Math-side NVFP4 fused-layout dequant requires packed scratch");
                        const uint32_t _tid_in_wg = epilogue_thread_idx;
                        deep_gemm::nvfp4::dequant_smem_b_from_packed_grouped_nibble<
                            kH200MimoQuadILP>(
                            reinterpret_cast<uint8_t*>(smem_b[stage_idx]),
                            reinterpret_cast<const uint8_t*>(smem_packed_b[stage_idx]),
                            _tid_in_wg, smem_nvfp4_lut);
                        const unsigned long long dequant_end = phase_profile_clock();
                        if constexpr (kPhaseProfileRequested)
                            block_math_dequant += dequant_end - dequant_start;
                    }
                    cutlass::arch::fence_view_async_shared();
                    ptx::sync_aligned(128, kEpilogueWGBarrierStartIdx + epilogue_wg_idx);
                }

                // Read SF (must precede warpgroup_arrive)
                float scale_a_0_lo, scale_a_1_lo;
                float scale_a_0_hi, scale_a_1_hi;  // Only used in L2 (per-64 K)
                if (block_phase == sched::BlockPhase::Linear1) {
                    scale_a_0_lo = ptx::ld_shared(smem_sfa[stage_idx] + row_offset_r0);
                    scale_a_1_lo = ptx::ld_shared(smem_sfa[stage_idx] + row_offset_r1);
                } else {
                    // L2: SFA layout is (K=2, M=BLOCK_M) MN-major; first half SF at offset 0, second at BLOCK_M
                    scale_a_0_lo = ptx::ld_shared(smem_sfa[stage_idx] + row_offset_r0);
                    scale_a_1_lo = ptx::ld_shared(smem_sfa[stage_idx] + row_offset_r1);
                    scale_a_0_hi = ptx::ld_shared(smem_sfa[stage_idx] + kL2SFAHalfStride + row_offset_r0);
                    scale_a_1_hi = ptx::ld_shared(smem_sfa[stage_idx] + kL2SFAHalfStride + row_offset_r1);
                }

                // NVFP4 UE4M3 weight scales are applied during FP4 -> FP8 smem
                // expansion, so the WGMMA accumulator only needs activation SF.

                if (block_phase == sched::BlockPhase::Linear1) {
                    if constexpr (kSwapABL1Active) {
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

                        const uint32_t n_swap = ((valid_m + 7u) / 8u) * 8u;
                        if constexpr (kSwapABFlashFineDispatch) {
                            if (n_swap <= 8) {
                                run_swap_ab_l1.template operator()<8>();
                            } else if (n_swap <= 16) {
                                run_swap_ab_l1.template operator()<16>();
                            } else if (n_swap <= 24) {
                                run_swap_ab_l1.template operator()<24>();
                            } else {
                                run_swap_ab_l1.template operator()<64>();
                            }
                        } else {
                            if (n_swap <= 8) {
                                run_swap_ab_l1.template operator()<8>();
                            } else if (n_swap <= 16) {
                                run_swap_ab_l1.template operator()<16>();
                            } else {
                                run_swap_ab_l1.template operator()<64>();
                            }
                        }
                    } else {
                    // Single per-128 K-block WGMMA group
                    #pragma unroll
                    for (uint32_t i = 0; i < kAccumPerThread; ++ i) ptx::warpgroup_fence_operand(accum[i]);
                    ptx::warpgroup_arrive();
                    #pragma unroll
                    for (uint32_t k = 0; k < BLOCK_K / WGMMA::K; ++ k) {
                        auto desc_a = mma::sm90::make_smem_desc(
                            smem_a[stage_idx] + row_block_offset * BLOCK_K + k * WGMMA::K, 1);
                        auto desc_b = mma::sm90::make_smem_desc(
                            smem_b[stage_idx] + wg_n_idx * BLOCK_K + k * WGMMA::K, 1);  // NVFP4: no swizzle on B
                        WGMMA::wgmma(desc_a, desc_b, accum, k);
                    }
                    ptx::warpgroup_commit_batch();
                    #pragma unroll
                    for (uint32_t i = 0; i < kAccumPerThread; ++ i) ptx::warpgroup_fence_operand(accum[i]);
                    ptx::warpgroup_wait<0>();

                    arrive_empty_barrier(stage_idx);

                    // L1: gate/up alternate at gran=8 along N; each `i` block of 8
                    // cols belongs entirely to one of {gate, up}, so .x and .y
                    // share the same scalar.
                    #pragma unroll
                    for (uint32_t i = 0; i < kAccumPerThread / 4; ++ i) {
                        final_accum[i*4+0] += scale_a_0_lo * accum[i*4+0];
                        final_accum[i*4+1] += scale_a_0_lo * accum[i*4+1];
                        final_accum[i*4+2] += scale_a_1_lo * accum[i*4+2];
                        final_accum[i*4+3] += scale_a_1_lo * accum[i*4+3];
                    }
                    }
                } else {
                    if constexpr (kSwapABL2Active) {
                        DG_STATIC_ASSERT(kL2ActsSFGranK == 64,
                                         "L2 swapAB assumes per-64 activation scales");
                        auto run_swap_ab_l2 = [&]<uint32_t N_SWAP>() {
                            using SwapWGMMA = typename mma::sm90::FP8MMASelector<N_SWAP>::type;
                            constexpr uint32_t kSwapAccum = SwapWGMMA::kNumAccum;
                            float swap_accum[kSwapAccum];

                            auto promote_swap_accum = [&](const uint32_t& half, const uint32_t& sf_group) {
                                #pragma unroll
                                for (uint32_t i = 0; i < kSwapAccum / 4; ++ i) {
                                    const uint32_t accum_offset = half * kSwapABHalfAccumPerThread + i * 4;
                                    const uint32_t token_0 = i * 8 + col_idx * 2;
                                    const uint32_t token_1 = token_0 + 1;
                                    if (token_0 < valid_m) {
                                        const float scale_0 = ptx::ld_shared(
                                            smem_sfa[stage_idx] + sf_group * kL2SFAHalfStride + token_0);
                                        final_accum[accum_offset + 0] += scale_0 * swap_accum[i * 4 + 0];
                                        final_accum[accum_offset + 2] += scale_0 * swap_accum[i * 4 + 2];
                                    }
                                    if (token_1 < valid_m) {
                                        const float scale_1 = ptx::ld_shared(
                                            smem_sfa[stage_idx] + sf_group * kL2SFAHalfStride + token_1);
                                        final_accum[accum_offset + 1] += scale_1 * swap_accum[i * 4 + 1];
                                        final_accum[accum_offset + 3] += scale_1 * swap_accum[i * 4 + 3];
                                    }
                                }
                            };

                            #pragma unroll
                            for (uint32_t half = 0; half < kSwapABWeightHalves; ++ half) {
                            #pragma unroll
                            for (uint32_t i = 0; i < kSwapAccum; ++ i)
                                ptx::warpgroup_fence_operand(swap_accum[i]);
                            ptx::warpgroup_arrive();
                            #pragma unroll
                            for (uint32_t k = 0; k < (BLOCK_K / 2) / SwapWGMMA::K; ++ k) {
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
                            promote_swap_accum(half, 0);

                            #pragma unroll
                            for (uint32_t i = 0; i < kSwapAccum; ++ i)
                                ptx::warpgroup_fence_operand(swap_accum[i]);
                            ptx::warpgroup_arrive();
                            #pragma unroll
                            for (uint32_t k = 0; k < (BLOCK_K / 2) / SwapWGMMA::K; ++ k) {
                                const uint32_t k_off = (BLOCK_K / 2) + k * SwapWGMMA::K;
                                auto desc_a = mma::sm90::make_smem_desc(
                                    smem_b[stage_idx] + (wg_n_idx + half * 64u) * BLOCK_K + k_off, 1);
                                auto desc_b = mma::sm90::make_smem_desc(
                                    smem_a[stage_idx] + k_off, 1);
                                SwapWGMMA::wgmma(desc_a, desc_b, swap_accum, k);
                            }
                            ptx::warpgroup_commit_batch();
                            #pragma unroll
                            for (uint32_t i = 0; i < kSwapAccum; ++ i)
                                ptx::warpgroup_fence_operand(swap_accum[i]);
                            ptx::warpgroup_wait<0>();
                            promote_swap_accum(half, 1);
                            }

                            arrive_empty_barrier(stage_idx);
                        };

                        const uint32_t n_swap = ((valid_m + 7u) / 8u) * 8u;
                        if constexpr (kSwapABFlashFineDispatch) {
                            if (n_swap <= 8) {
                                run_swap_ab_l2.template operator()<8>();
                            } else if (n_swap <= 16) {
                                run_swap_ab_l2.template operator()<16>();
                            } else if (n_swap <= 24) {
                                run_swap_ab_l2.template operator()<24>();
                            } else if (n_swap <= 32) {
                                run_swap_ab_l2.template operator()<32>();
                            } else {
                                run_swap_ab_l2.template operator()<64>();
                            }
                        } else {
                            if (n_swap <= 8) {
                                run_swap_ab_l2.template operator()<8>();
                            } else if (n_swap <= 16) {
                                run_swap_ab_l2.template operator()<16>();
                            } else if (n_swap <= 32) {
                                run_swap_ab_l2.template operator()<32>();
                            } else {
                                run_swap_ab_l2.template operator()<64>();
                            }
                        }
                    } else if constexpr (kL2DualAccum) {
                        float accum_hi[kAccumPerThread];

                        const auto desc_a_lo0 = mma::sm90::make_smem_desc(
                            smem_a[stage_idx] + row_block_offset * BLOCK_K, 1);
                        const auto desc_b_lo0 = mma::sm90::make_smem_desc(
                            smem_b[stage_idx] + wg_n_idx * BLOCK_K, 1);
                        const auto desc_a_lo1 = mma::sm90::make_smem_desc(
                            smem_a[stage_idx] + row_block_offset * BLOCK_K + WGMMA::K, 1);
                        const auto desc_b_lo1 = mma::sm90::make_smem_desc(
                            smem_b[stage_idx] + wg_n_idx * BLOCK_K + WGMMA::K, 1);
                        const auto desc_a_hi0 = mma::sm90::make_smem_desc(
                            smem_a[stage_idx] + row_block_offset * BLOCK_K + BLOCK_K / 2, 1);
                        const auto desc_b_hi0 = mma::sm90::make_smem_desc(
                            smem_b[stage_idx] + wg_n_idx * BLOCK_K + BLOCK_K / 2, 1);
                        const auto desc_a_hi1 = mma::sm90::make_smem_desc(
                            smem_a[stage_idx] + row_block_offset * BLOCK_K + BLOCK_K / 2 + WGMMA::K, 1);
                        const auto desc_b_hi1 = mma::sm90::make_smem_desc(
                            smem_b[stage_idx] + wg_n_idx * BLOCK_K + BLOCK_K / 2 + WGMMA::K, 1);

                        #pragma unroll
                        for (uint32_t i = 0; i < kAccumPerThread; ++ i) {
                            ptx::warpgroup_fence_operand(accum[i]);
                            ptx::warpgroup_fence_operand(accum_hi[i]);
                        }
                        ptx::warpgroup_arrive();
                        WGMMA::wgmma(desc_a_lo0, desc_b_lo0, accum, false);
                        WGMMA::wgmma(desc_a_lo1, desc_b_lo1, accum, true);
                        WGMMA::wgmma(desc_a_hi0, desc_b_hi0, accum_hi, false);
                        WGMMA::wgmma(desc_a_hi1, desc_b_hi1, accum_hi, true);
                        ptx::warpgroup_commit_batch();
                        #pragma unroll
                        for (uint32_t i = 0; i < kAccumPerThread; ++ i) {
                            ptx::warpgroup_fence_operand(accum[i]);
                            ptx::warpgroup_fence_operand(accum_hi[i]);
                        }
                        ptx::warpgroup_wait<0>();

                        arrive_empty_barrier(stage_idx);

                        #pragma unroll
                        for (uint32_t i = 0; i < kAccumPerThread / 4; ++ i) {
                            final_accum[i*4+0] += scale_a_0_lo * accum[i*4+0];
                            final_accum[i*4+1] += scale_a_0_lo * accum[i*4+1];
                            final_accum[i*4+2] += scale_a_1_lo * accum[i*4+2];
                            final_accum[i*4+3] += scale_a_1_lo * accum[i*4+3];
                            final_accum[i*4+0] += scale_a_0_hi * accum_hi[i*4+0];
                            final_accum[i*4+1] += scale_a_0_hi * accum_hi[i*4+1];
                            final_accum[i*4+2] += scale_a_1_hi * accum_hi[i*4+2];
                            final_accum[i*4+3] += scale_a_1_hi * accum_hi[i*4+3];
                        }
                    } else {
                    // L2: split BLOCK_K=128 into two halves (per-64 SFA), each 2 WGMMAs.
                    // First half: K=0..63, SFA = scale_a_*_lo
                    #pragma unroll
                    for (uint32_t i = 0; i < kAccumPerThread; ++ i) ptx::warpgroup_fence_operand(accum[i]);
                    ptx::warpgroup_arrive();
                    #pragma unroll
                    for (uint32_t k = 0; k < (BLOCK_K / 2) / WGMMA::K; ++ k) {
                        auto desc_a = mma::sm90::make_smem_desc(
                            smem_a[stage_idx] + row_block_offset * BLOCK_K + k * WGMMA::K, 1);
                        auto desc_b = mma::sm90::make_smem_desc(
                            smem_b[stage_idx] + wg_n_idx * BLOCK_K + k * WGMMA::K, 1);
                        WGMMA::wgmma(desc_a, desc_b, accum, k);
                    }
                    ptx::warpgroup_commit_batch();
                    #pragma unroll
                    for (uint32_t i = 0; i < kAccumPerThread; ++ i) ptx::warpgroup_fence_operand(accum[i]);
                    ptx::warpgroup_wait<0>();

                    // L2 weight SF is per 128 output columns; M64N256 spans two SF groups.
                    #pragma unroll
                    for (uint32_t i = 0; i < kAccumPerThread / 4; ++ i) {
                        final_accum[i*4+0] += scale_a_0_lo * accum[i*4+0];
                        final_accum[i*4+1] += scale_a_0_lo * accum[i*4+1];
                        final_accum[i*4+2] += scale_a_1_lo * accum[i*4+2];
                        final_accum[i*4+3] += scale_a_1_lo * accum[i*4+3];
                    }

                    // Second half: K=64..127, SFA = scale_a_*_hi
                    #pragma unroll
                    for (uint32_t i = 0; i < kAccumPerThread; ++ i) ptx::warpgroup_fence_operand(accum[i]);
                    ptx::warpgroup_arrive();
                    #pragma unroll
                    for (uint32_t k = 0; k < (BLOCK_K / 2) / WGMMA::K; ++ k) {
                        const uint32_t k_off = (BLOCK_K / 2) + k * WGMMA::K;
                        auto desc_a = mma::sm90::make_smem_desc(
                            smem_a[stage_idx] + row_block_offset * BLOCK_K + k_off, 1);
                        auto desc_b = mma::sm90::make_smem_desc(
                            smem_b[stage_idx] + wg_n_idx * BLOCK_K + k_off, 1);
                        WGMMA::wgmma(desc_a, desc_b, accum, k);
                    }
                    ptx::warpgroup_commit_batch();
                    #pragma unroll
                    for (uint32_t i = 0; i < kAccumPerThread; ++ i) ptx::warpgroup_fence_operand(accum[i]);
                    ptx::warpgroup_wait<0>();

                    arrive_empty_barrier(stage_idx);

                    // L2 second half: same SFA half, still choose weight SF by N chunk.
                    #pragma unroll
                    for (uint32_t i = 0; i < kAccumPerThread / 4; ++ i) {
                        final_accum[i*4+0] += scale_a_0_hi * accum[i*4+0];
                        final_accum[i*4+1] += scale_a_0_hi * accum[i*4+1];
                        final_accum[i*4+2] += scale_a_1_hi * accum[i*4+2];
                        final_accum[i*4+3] += scale_a_1_hi * accum[i*4+3];
                    }
                    }
                }
            }
            };

            run_default_gemm_loop();

            const unsigned long long block_gemm_end = phase_profile_clock();
            if (epilogue_warp_idx == 0 and lane_idx == 0) {
                phase_profile_record(kProfileGemmCore, block_gemm_end - block_gemm_start);
                phase_profile_record(kProfileMathFullWait, block_math_full_wait);
                phase_profile_record(kProfileMathDequant, block_math_dequant);
            }

            // Skip epilogue when block is past valid M (still must release via empty).
            // A dummy cluster peer may still carry an async L1 store from the
            // previous valid block, so drain it before leaving the L1 wave.
            if (row_block_offset >= valid_m) {
                if constexpr (!kL2ArrivalCounter || kBlockIsL2)
                    ptx::sync_aligned(kNumEpilogueThreads, kEpilogueFullBarrierIdx);
                return;
            }

            const unsigned long long block_epilogue_start = phase_profile_clock();
            if constexpr (!kBlockIsL2) {
                const float l1_global_scale = l1_global_scales == nullptr ? 1.0f : __ldg(l1_global_scales + local_expert_idx);
                if constexpr (kSwapABL1Active) {
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

                    constexpr uint32_t kWarpsPerEpilogueWG =
                        kNumEpilogueWarps / kNumEpilogueWarpgroups;
                    const uint32_t reduce_warp_start = kSplitNSharesSF ?
                        0u : epilogue_wg_idx * kWarpsPerEpilogueWG;
                    const uint32_t reduce_warp_count = kSplitNSharesSF ?
                        kNumEpilogueWarps : kWarpsPerEpilogueWG;
                    const uint32_t scale_token_thread = kSplitNSharesSF ?
                        epilogue_thread_idx : epilogue_thread_idx % 128u;
                    const uint32_t scale_token_stride = kSplitNSharesSF ?
                        kNumEpilogueThreads : 128u;
                    const uint32_t sf_base_k_idx =
                        (n_block_idx * L1_OUT_BLOCK_N + (kSplitNSharesSF ? 0u : wg_l1_out_n_idx)) / 64u;
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
                                float g0 = final_accum[accum_offset + 0] * l1_global_scale;
                                float u0 = final_accum[accum_offset + 2] * l1_global_scale;
                                clamp_gate(g0);
                                clamp_up(u0);
                                const float weight_0 = *l1_topk_weights_buffer
                                    .get_data_buffer(m_idx + token_0)
                                    .template get_base_ptr<float>();
                                v0 = silu(g0) * u0 * weight_0;
                                swap_v0[half][i] = v0;
                                v0_amax = cute::max(v0_amax, cute::abs(v0));
                            }

                            float v1 = 0.0f;
                            if (token_1 < valid_m) {
                                float g1 = final_accum[accum_offset + 1] * l1_global_scale;
                                float u1 = final_accum[accum_offset + 3] * l1_global_scale;
                                clamp_gate(g1);
                                clamp_up(u1);
                                const float weight_1 = *l1_topk_weights_buffer
                                    .get_data_buffer(m_idx + token_1)
                                    .template get_base_ptr<float>();
                                v1 = silu(g1) * u1 * weight_1;
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

                        auto sf_base_ptr = l2_sf_buffer.get_base_ptr<float>();
                        const uint32_t token_idx = pool_block_idx * BLOCK_M + token;
                        sf_base_ptr[sf_base_k_idx * kNumPaddedSFPoolTokens + token_idx] = sf_pair.x;
                        smem_cd_l1_shared_sf[token * kNumEpilogueWarps + reduce_warp_start] = sf_inv_pair.x;
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
                                reinterpret_cast<uint8_t*>(smem_cd_l1)[token_0 * L1_OUT_BLOCK_N + out_col_base] =
                                    *reinterpret_cast<const uint8_t*>(&q);
                            }
                            if (token_1 < valid_m) {
                                const float sf_inv =
                                    smem_cd_l1_shared_sf[token_1 * kNumEpilogueWarps + reduce_warp_start];
                                const __nv_fp8_e4m3 q(swap_v1[half][i] * sf_inv);
                                reinterpret_cast<uint8_t*>(smem_cd_l1)[token_1 * L1_OUT_BLOCK_N + out_col_base] =
                                    *reinterpret_cast<const uint8_t*>(&q);
                            }
                        }
                    }
                    ptx::sync_aligned(kNumEpilogueThreads, kEpilogueFullBarrierIdx);
                    if (epilogue_wg_idx == 0 and warp_idx_in_wg == 0 and cute::elect_one_sync()) {
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
                    const unsigned long long l1_tma_wait_start = phase_profile_clock();
                    ptx::tma_store_wait<0>();
                    const unsigned long long l1_tma_wait_end = phase_profile_clock();
                    if (epilogue_warp_idx == 0 and lane_idx == 0)
                        phase_profile_record(kProfileL1TMAWait, l1_tma_wait_end - l1_tma_wait_start);
                    ptx::sync_aligned(kNumEpilogueThreads, kEpilogueFullBarrierIdx);
                    notify_l1_ready(pool_block_idx, n_block_idx);
                } else {
                // ---------------- L1 EPILOGUE: SwiGLU + FP8 quantize + TMA store ----------------
                // Layout in `final_accum`:
                //   16 chunks of 8 N-cols, each chunk = 4 floats per thread = (r0c0, r0c1, r1c0, r1c1).
                //   Gate chunks: even (0, 2, ..., 14). Up chunks: odd (1, 3, ..., 15).
                //   Pair `p` ∈ [0, 8): gate chunk = 2p, up chunk = 2p+1.
                //
                // For each pair we produce 4 post-SwiGLU floats per thread, mapped to
                // output cols (p*8 + col_idx*2 + {0,1}) for both r0 and r1.

                constexpr uint32_t kNumPairs = kAccumPerThread / 8;
                constexpr uint32_t kNumSFGroups = kSplitNSharesSF ? 1 : WG_L1_OUT_BLOCK_N / 64;
                DG_STATIC_ASSERT(kSplitNSharesSF || WG_L1_OUT_BLOCK_N % 64 == 0,
                                 "L1 output SF is per 64 columns");
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
                    float g_r0_c0 = final_accum[gate*4 + 0] * l1_global_scale; clamp_gate(g_r0_c0);
                    float g_r0_c1 = final_accum[gate*4 + 1] * l1_global_scale; clamp_gate(g_r0_c1);
                    float g_r1_c0 = final_accum[gate*4 + 2] * l1_global_scale; clamp_gate(g_r1_c0);
                    float g_r1_c1 = final_accum[gate*4 + 3] * l1_global_scale; clamp_gate(g_r1_c1);
                    float u_r0_c0 = final_accum[up*4   + 0] * l1_global_scale; clamp_up(u_r0_c0);
                    float u_r0_c1 = final_accum[up*4   + 1] * l1_global_scale; clamp_up(u_r0_c1);
                    float u_r1_c0 = final_accum[up*4   + 2] * l1_global_scale; clamp_up(u_r1_c0);
                    float u_r1_c1 = final_accum[up*4   + 3] * l1_global_scale; clamp_up(u_r1_c1);

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

                if constexpr (kSplitNSharesSF) {
                    DG_STATIC_ASSERT(kNumSFGroups == 1, "split-N shared-SF path has one 64-column output SF group");
                    if (col_idx == 0) {
                        smem_cd_l1_shared_sf[epilogue_wg_idx * BLOCK_M + row_offset_r0] = amax_r0[0];
                        smem_cd_l1_shared_sf[epilogue_wg_idx * BLOCK_M + row_offset_r1] = amax_r1[0];
                    }
                    ptx::sync_aligned(kNumEpilogueThreads, kEpilogueFullBarrierIdx);
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
                constexpr uint32_t l1_store_stage = 0u;
                auto* smem_cd_l1_wg = smem_cd_l1
                    + l1_store_stage * kNumEpilogueWarpgroups * WG_BLOCK_M * L1_OUT_BLOCK_N
                    + (kSplitNWarpgroups ? 0 : epilogue_wg_idx * WG_BLOCK_M * L1_OUT_BLOCK_N);
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
                        smem_cd_l1_wg + r_0 * L1_OUT_BLOCK_N + wg_l1_out_n_idx + col);
                    auto* p1 = reinterpret_cast<uint16_t*>(
                        smem_cd_l1_wg + r_1 * L1_OUT_BLOCK_N + wg_l1_out_n_idx + col);
                    if (valid_r0)
                        *p0 = r0_pair.__x;
                    if (valid_r1)
                        *p1 = r1_pair.__x;
                }

                // Write L2-activation SF as float, one value per 64 output columns.
                if (col_idx == 0) {
                    auto sf_base_ptr = l2_sf_buffer.get_base_ptr<float>();
                    const uint32_t token_r0 = pool_block_idx * BLOCK_M + row_offset_r0;
                    const uint32_t token_r1 = pool_block_idx * BLOCK_M + row_offset_r1;
                    const uint32_t base_k_sf_idx = (n_block_idx * L1_OUT_BLOCK_N + wg_l1_out_n_idx) / 64u;
                    #pragma unroll
                    for (uint32_t g = 0; g < kNumSFGroups; ++ g) {
                        if ((!kSplitNSharesSF || epilogue_wg_idx == 0) && valid_r0)
                            sf_base_ptr[(base_k_sf_idx + g) * kNumPaddedSFPoolTokens + token_r0] = sf_r0[g];
                        if ((!kSplitNSharesSF || epilogue_wg_idx == 0) && valid_r1)
                            sf_base_ptr[(base_k_sf_idx + g) * kNumPaddedSFPoolTokens + token_r1] = sf_r1[g];
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
                if constexpr (kSplitNWarpgroups) {
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
                    const unsigned long long l1_tma_wait_start = phase_profile_clock();
                    ptx::tma_store_wait<0>();
                    const unsigned long long l1_tma_wait_end = phase_profile_clock();
                    if (epilogue_warp_idx == 0 and lane_idx == 0)
                        phase_profile_record(kProfileL1TMAWait, l1_tma_wait_end - l1_tma_wait_start);
                    ptx::sync_aligned(kNumEpilogueThreads, kEpilogueFullBarrierIdx);
                    notify_l1_ready(pool_block_idx, n_block_idx);
                } else {
                    ptx::sync_aligned(128, kEpilogueWGBarrierStartIdx + epilogue_wg_idx);
                    if (warp_idx_in_wg == 0 and cute::elect_one_sync()) {
                        const uint32_t out_n_idx = n_block_idx * L1_OUT_BLOCK_N;
                        cute::tma_store_fence();
                        cute::SM90_TMA_STORE_2D::copy(
                            &tensor_map_l1_output,
                            smem_cd_l1_wg,
                            out_n_idx,
                            m_idx + row_block_offset);
                        cute::tma_store_arrive();
                    }
                    __syncwarp();

                        const unsigned long long l1_tma_wait_start = phase_profile_clock();
                        ptx::tma_store_wait<0>();
                        const unsigned long long l1_tma_wait_end = phase_profile_clock();
                        if (epilogue_warp_idx == 0 and lane_idx == 0)
                            phase_profile_record(kProfileL1TMAWait, l1_tma_wait_end - l1_tma_wait_start);
                        if constexpr (!kL2ArrivalCounter)
                            ptx::sync_aligned(kNumEpilogueThreads, kEpilogueFullBarrierIdx);
                        notify_l1_ready(pool_block_idx, n_block_idx);
                                    }
                }
                const unsigned long long block_epilogue_end = phase_profile_clock();
                if (epilogue_warp_idx == 0 and lane_idx == 0)
                    phase_profile_record(kProfileL1Epilogue, block_epilogue_end - block_epilogue_start);
            } else {
                // ---------------- L2 EPILOGUE: BF16 cast + NVLink scatter ----------------
                constexpr uint32_t kNumRowsPerWarp = WG_BLOCK_M / 8;

                if constexpr (kSwapABL2Active) {
                    auto store_swap_bf16 = [&](const uint32_t& token, const uint32_t& col, const float& value) {
                        if (token < valid_m)
                            smem_cd_l2[token * BLOCK_N + wg_n_idx + col] =
                                __float2bfloat16_rn(value * l2_global_scale);
                    };

                    const unsigned long long l2_scatter_start = phase_profile_clock();
                    const uint32_t num_swap_token_chunks = (valid_m + 7u) / 8u;
                    auto store_l2_swap_chunk = [&](const uint32_t& i) {
                        const uint32_t token_0 = i * 8 + col_idx * 2;
                        const uint32_t token_1 = token_0 + 1;
                        #pragma unroll
                        for (uint32_t half = 0; half < kSwapABWeightHalves; ++ half) {
                            const uint32_t accum_offset = half * kSwapABHalfAccumPerThread + i * 4;
                            const uint32_t col_offset = half * 64u;
                            store_swap_bf16(token_0, col_offset + r_0, final_accum[accum_offset + 0]);
                            store_swap_bf16(token_0, col_offset + r_1, final_accum[accum_offset + 2]);
                            store_swap_bf16(token_1, col_offset + r_0, final_accum[accum_offset + 1]);
                            store_swap_bf16(token_1, col_offset + r_1, final_accum[accum_offset + 3]);
                        }
                    };

                    store_l2_swap_chunk(0);
                    if (valid_m > 8) {
                        #pragma unroll
                        for (uint32_t i = 1; i < kSwapABTokenChunks; ++ i) {
                            if (i < num_swap_token_chunks)
                                store_l2_swap_chunk(i);
                        }
                    }

                    ptx::sync_aligned(128, kEpilogueWGBarrierStartIdx + epilogue_wg_idx);

                    const uint32_t row_in_warp_block = lane_idx / 16;
                    const uint32_t lane_in_row = lane_idx % 16;
                    constexpr uint32_t kColsPerScatterLane = WG_BLOCK_N / 16;
                    DG_STATIC_ASSERT(WG_BLOCK_N % 16 == 0,
                                     "SwapAB L2 scatter expects an even lane partition");
                    DG_STATIC_ASSERT(kColsPerScatterLane == 4 || kColsPerScatterLane == 8,
                                     "SwapAB L2 scatter supports WG_BLOCK_N=64 or 128");

                    #pragma unroll
                    for (uint32_t j = 0; j < kNumRowsPerWarp; ++ j) {
                        const uint32_t token = warp_idx_in_wg * 16 + j * 2 + row_in_warp_block;
                        if (token >= valid_m) break;

                        const auto src_metadata = *workspace.get_token_src_metadata_ptr(m_idx + token);
                        const uint32_t dst_rank_idx = src_metadata.rank_idx;
                        const uint32_t dst_token_idx = src_metadata.token_idx;
                        const uint32_t dst_topk_idx = src_metadata.topk_idx;
                        const auto dst_token = combine_token_buffer.get_rank_buffer(dst_topk_idx)
                                               .get_data_buffer(dst_token_idx);
                        auto smem_ptr = smem_cd_l2
                            + token * BLOCK_N
                            + wg_n_idx
                            + lane_in_row * kColsPerScatterLane;
                        if constexpr (kColsPerScatterLane == 8) {
                            const auto packed = *reinterpret_cast<uint4*>(smem_ptr);
                            auto dst_ptr = math::advance_ptr<uint4>(
                                dst_token.get_base_ptr(),
                                n_idx * sizeof(nv_bfloat16) + lane_in_row * sizeof(uint4));
                            *sym_buffer.map(dst_ptr, dst_rank_idx) = packed;
                        } else {
                            const auto packed = *reinterpret_cast<uint2*>(smem_ptr);
                            auto dst_ptr = math::advance_ptr<uint2>(
                                dst_token.get_base_ptr(),
                                n_idx * sizeof(nv_bfloat16) + lane_in_row * sizeof(uint2));
                            *sym_buffer.map(dst_ptr, dst_rank_idx) = packed;
                        }
                    }
                    ptx::sync_aligned(kNumEpilogueThreads, kEpilogueFullBarrierIdx);
                    const unsigned long long l2_scatter_end = phase_profile_clock();
                    if (epilogue_warp_idx == 0 and lane_idx == 0)
                        phase_profile_record(kProfileL2Scatter, l2_scatter_end - l2_scatter_start);
                } else {
                DG_STATIC_ASSERT(WG_BLOCK_N == 64 || WG_BLOCK_N == 128, "Direct L2 scatter requires N64/N128");
                const unsigned long long l2_scatter_start = phase_profile_clock();

                    auto scatter_direct_row = [&](const uint32_t& row_offset, const bool& valid_row, const uint32_t& row_accum_offset) {
                            if (valid_row) {
                                const auto src_metadata = *workspace.get_token_src_metadata_ptr(m_idx + row_offset);
                                const uint32_t dst_rank_idx = src_metadata.rank_idx;
                                const uint32_t dst_token_idx = src_metadata.token_idx;
                                const uint32_t dst_topk_idx = src_metadata.topk_idx;
                                const auto dst_token = combine_token_buffer.get_rank_buffer(dst_topk_idx)
                                                       .get_data_buffer(dst_token_idx);
                                auto dst_base = math::advance_ptr<uint8_t>(
                                    dst_token.get_base_ptr(), n_idx * sizeof(nv_bfloat16));
                                auto mapped_dst_base = sym_buffer.map(dst_base, dst_rank_idx);

                                #pragma unroll
                                for (uint32_t i = 0; i < kAccumPerThread / 8; ++ i) {
                                    const uint32_t chunk_lo = 2 * i, chunk_hi = 2 * i + 1;
                                    const uint32_t col_lo = chunk_lo * 8 + col_idx * 2;
                                    const uint32_t col_hi = chunk_hi * 8 + col_idx * 2;
                                    const uint32_t packed_lo = cast_l2_scaled_bf16_pair(
                                        final_accum[chunk_lo * 4 + row_accum_offset + 0],
                                        final_accum[chunk_lo * 4 + row_accum_offset + 1]);
                                    const uint32_t packed_hi = cast_l2_scaled_bf16_pair(
                                        final_accum[chunk_hi * 4 + row_accum_offset + 0],
                                        final_accum[chunk_hi * 4 + row_accum_offset + 1]);
                                    *reinterpret_cast<uint32_t*>(mapped_dst_base + col_lo * sizeof(nv_bfloat16)) = packed_lo;
                                    *reinterpret_cast<uint32_t*>(mapped_dst_base + col_hi * sizeof(nv_bfloat16)) = packed_hi;
                                }
                            }
                    };

                    scatter_direct_row(row_offset_r0, valid_r0, 0);
                    scatter_direct_row(row_offset_r1, valid_r1, 2);
                    ptx::sync_aligned(kNumEpilogueThreads, kEpilogueFullBarrierIdx);
                    const unsigned long long l2_scatter_end = phase_profile_clock();
                    if (epilogue_warp_idx == 0 and lane_idx == 0)
                        phase_profile_record(kProfileL2Scatter, l2_scatter_end - l2_scatter_start);
                }
                const unsigned long long block_epilogue_end = phase_profile_clock();
                if (epilogue_warp_idx == 0 and lane_idx == 0)
                    phase_profile_record(kProfileL2Epilogue, block_epilogue_end - block_epilogue_start);
            }
        });
        const unsigned long long math_loop_end = phase_profile_clock();
        if (epilogue_warp_idx == 0 and lane_idx == 0)
            phase_profile_record(kProfileMathLoop, math_loop_end - math_loop_start);



        // ---------------- COMBINE ----------------
        // NVLink barrier first: signals remote ranks that this rank's GEMM
        // outputs (NVLink scatter targets) are fully written.
        const unsigned long long combine_barrier_start = phase_profile_clock();
        comm::nvlink_barrier<kNumRanks, kNumSMs, kNumEpilogueThreads,
                             kEpilogueGridSyncIndex, kBeforeCombineReduceBarrierTag>(
            workspace, sym_buffer, sm_idx, epilogue_thread_idx,
            [&]() { ptx::sync_aligned(kNumEpilogueThreads, kEpilogueFullBarrierIdx); }
        );
        const unsigned long long combine_barrier_end = phase_profile_clock();
        if (epilogue_warp_idx == 0 and lane_idx == 0)
            phase_profile_record(kProfileCombineBarrier, combine_barrier_end - combine_barrier_start);

        // Sync with dispatch (paired with dispatch's pre-cleanup sync) so that
        // dispatch may now safely clean workspace state.
        ptx::sync_unaligned(kNumDispatchThreads + kNumEpilogueThreads, kDispatchWithEpilogueBarrierIdx);
        const unsigned long long combine_reduce_start = phase_profile_clock();

        constexpr uint32_t kNumHiddenBytes = kHidden * sizeof(nv_bfloat16);
        constexpr uint32_t kNumElemsPerUint4 = sizeof(uint4) / sizeof(nv_bfloat162);

        constexpr uint32_t kNumChunkSlots = 3;
        constexpr uint32_t kNumMaxRegistersForBuffer = 128;
        constexpr uint32_t kNumDefaultChunks =
            (kNumChunkSlots * kNumEpilogueWarps * kNumHiddenBytes <= SMEM_BEFORE_BARRIER_SIZE
             and kHidden <= 32 * kNumMaxRegistersForBuffer) ? 1 : 2;
        constexpr uint32_t kNumChunks = kNumDefaultChunks;
        constexpr uint32_t kNumChunkBytes = kNumHiddenBytes / kNumChunks;
        constexpr uint32_t kNumChunkUint4 = kNumChunkBytes / sizeof(uint4);
        constexpr uint32_t kNumUint4PerLane = kNumChunkUint4 / 32;
        DG_STATIC_ASSERT(kHidden % kNumChunks == 0, "Hidden must be divisible by number of chunks");
        DG_STATIC_ASSERT(kNumChunkSlots * kNumEpilogueWarps * kNumHiddenBytes / kNumChunks <= SMEM_BEFORE_BARRIER_SIZE, "Hidden is too large");
        DG_STATIC_ASSERT(kNumChunkBytes % 16 == 0, "Combine chunk must be TMA-aligned (16 bytes)");
        DG_STATIC_ASSERT(kNumChunkBytes % sizeof(uint4) == 0, "Combine chunk must be divisible by 16 bytes");
        DG_STATIC_ASSERT(kNumChunkUint4 % 32 == 0, "Combine chunk must be a multiple of 32 16-byte elements");
        DG_STATIC_ASSERT(kNumTopk <= 32, "Top-k must fit in a single warp");

        DG_DEVICE_ASSERT(kNumChunkSlots * kNumEpilogueWarps * kNumChunkBytes <= static_cast<uint32_t>(
            reinterpret_cast<uint8_t*>(barrier_start_ptr) - smem_buffer));

        const auto combine_load_buffer = utils::PatternVisitor([&](const uint32_t& i) {
            return math::advance_ptr<uint4>(smem_buffer, (epilogue_warp_idx + i * kNumEpilogueWarps) * kNumChunkBytes);
        });
        const auto combine_store_buffer = math::advance_ptr<uint4>(
            smem_buffer, (epilogue_warp_idx + kNumEpilogueWarps * 2) * kNumChunkBytes);

        auto combine_load_barriers = utils::PatternVisitor([&](const uint32_t& i) {
            return combine_barriers[i + epilogue_warp_idx * 2];
        });

        uint32_t combine_phase = 0;
        uint32_t load_stage_idx = 0;
        for (uint32_t token_idx = sm_idx * kNumEpilogueWarps + epilogue_warp_idx;
             token_idx < num_tokens;
             token_idx += kNumSMs * kNumEpilogueWarps) {
            const int stored_topk_slot_idx = lane_idx < kNumTopk ?
                static_cast<int>(__ldg(input_topk_idx_buffer.get_base_ptr<int64_t>() + token_idx * kNumTopk + lane_idx)) : -1;
            const uint32_t total_mask = __ballot_sync(0xffffffff, stored_topk_slot_idx >= 0);

            for (uint32_t chunk = 0; chunk < kNumChunks; ++ chunk) {
                const uint32_t chunk_byte_offset = chunk * kNumChunkBytes;

                uint32_t mask = total_mask;
                const auto move_mask_and_load = [&](const uint32_t& i) {
                    if (mask) {
                        const uint32_t slot_idx = __ffs(mask) - 1;
                        mask ^= 1 << slot_idx;
                        if (cute::elect_one_sync()) {
                            const auto src_ptr = math::advance_ptr<uint8_t>(
                                combine_token_buffer.get_rank_buffer(slot_idx)
                                                    .get_data_buffer(token_idx).get_base_ptr(),
                                chunk_byte_offset);
                            ptx::tma_load_1d(combine_load_buffer[i], src_ptr, combine_load_barriers[i], kNumChunkBytes);
                            ptx::mbarrier_arrive_and_set_tx(combine_load_barriers[i], kNumChunkBytes);
                        }
                        __syncwarp();
                        return true;
                    }
                    return false;
                };

                bool do_reduce = move_mask_and_load(load_stage_idx);

                float2 reduced[kNumUint4PerLane * kNumElemsPerUint4] = {};
                while (do_reduce) {
                    do_reduce = move_mask_and_load(load_stage_idx ^ 1);
                    combine_load_barriers[load_stage_idx]->wait(combine_phase);
                    #pragma unroll
                    for (uint32_t j = 0; j < kNumUint4PerLane; ++ j) {
                        const auto uint4_values = combine_load_buffer[load_stage_idx][j * 32 + lane_idx];
                        const auto bf16_values = reinterpret_cast<const nv_bfloat162*>(&uint4_values);
                        #pragma unroll
                        for (uint32_t l = 0; l < kNumElemsPerUint4; ++ l)
                            ptx::accumulate(reduced[j * kNumElemsPerUint4 + l], bf16_values[l]);
                    }
                    combine_phase ^= load_stage_idx;
                    load_stage_idx ^= 1;
                }

                #pragma unroll
                for (uint32_t j = 0; j < kNumUint4PerLane; ++ j) {
                    uint4 casted;
                    auto casted_bf16 = reinterpret_cast<nv_bfloat162*>(&casted);
                    #pragma unroll
                    for (uint32_t l = 0; l < kNumElemsPerUint4; ++ l)
                        casted_bf16[l] = __float22bfloat162_rn(reduced[j * kNumElemsPerUint4 + l]);

                    if (j == 0) {
                        ptx::tma_store_wait<0>();
                        __syncwarp();
                    }
                    ptx::st_shared(combine_store_buffer + j * 32 + lane_idx,
                                   casted.x, casted.y, casted.z, casted.w);
                }
                __syncwarp();

                if (cute::elect_one_sync()) {
                    cute::tma_store_fence();
                    ptx::tma_store_1d(
                        math::advance_ptr(y, static_cast<uint64_t>(token_idx) * kNumHiddenBytes + chunk_byte_offset),
                        combine_store_buffer, kNumChunkBytes);
                    cute::tma_store_arrive();
                }
                __syncwarp();
            }
        }
        const unsigned long long combine_reduce_end = phase_profile_clock();
        if (epilogue_warp_idx == 0 and lane_idx == 0)
            phase_profile_record(kProfileCombineReduce, combine_reduce_end - combine_reduce_start);
        finish_no_dispatch_cleanup();

    }
#else
    if (blockIdx.x == 0 and threadIdx.x == 0)
        DG_DEVICE_ASSERT(false and "This kernel only supports sm_90");
#endif
