// Independent SM90 NVFP4 MegaMoE l1 kernel body.
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
    DG_STATIC_ASSERT(kNumNonEpilogueThreads == 128,
                     "BN128 split L1 requires the four-warp loader-dequant group");
    DG_STATIC_ASSERT((!kDispatchDequantRequested) or
                     (kNumDispatchThreads == 128 and
                      kNumNonEpilogueThreads == 128 and kNumEpilogueThreads == 256 and
                      BLOCK_M == 128 and BLOCK_N == 128),
                     "Dispatch-assisted dequant requires the split BM128/BN128 layout");

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
    }

    // =====================================================================
    // Workspaces and symmetric buffer slicing. The L2 activation SF allocation
    // intentionally retains its old per-64 physical capacity in stage 1, while
    // the active logical layout densely addresses only per-128 groups.
    // =====================================================================
    const auto workspace = layout::Workspace(
        sym_buffer.get_base_ptr(), kNumRanks, kNumExperts, kNumMaxTokensPerRank, kNumTopk);

    constexpr auto fp8_token_layout              = layout::Data(kHidden);
    constexpr auto fp8_intermediate_token_layout = layout::Data(kIntermediateHidden);
    // Per-128 K float SF: 4 bytes per per-128 group => `kHidden / 32` bytes/token (same as SM100 packing)
    constexpr auto fp8_sf_layout                 = layout::Data(kHidden / 32);
    // Retained physical capacity: 4 bytes per old per-64 group. Logical
    // per-128 scales occupy the dense first half of this allocation.
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
    constexpr bool kL2ArrivalCounter = kL2ArrivalCounterRequested && (!kSplitNWarpgroups) && BLOCK_N == 128;
    // Use two active dispatch warps. The other two dispatch warps form the
    // even-stage dequant team when dispatch-assisted dequant is on.
    constexpr uint32_t kNumActiveDispatchWarps =
        (BLOCK_N == 128 && kNumDispatchWarps == 4) ? 2 : kNumDispatchWarps;
    constexpr uint32_t kNumActiveDispatchThreads = kNumActiveDispatchWarps * 32;
    constexpr bool kLoaderDequant = true;
    constexpr bool kDispatchDequant = kDispatchDequantRequested;
    constexpr bool kPackedBScratch = BLOCK_N == 256 && (!kLoaderDequant);
    DG_STATIC_ASSERT(kLoaderDequant || kPackedBScratch,
                     "Fused NVFP4 B+scale layout requires loader dequant or packed scratch");
    using L1WGMMA   = typename mma::sm90::FP8MMASelector<WG_BLOCK_N>::type;
    static_assert(L1WGMMA::M == 64 and L1WGMMA::N == WG_BLOCK_N and L1WGMMA::K == 32,
                  "Unexpected WGMMA shape");
    DG_STATIC_ASSERT((!kSplitNWarpgroups) or (BLOCK_M == 64 and (WG_BLOCK_N == 64 or WG_BLOCK_N == 128)),
                     "Split-N path expects M64N64 or M64N128 WGMMA consumers");

    // A is always CTA-local.  When kClusterSize=2 the scheduler pairs adjacent
    // M blocks with identical expert/N/K coordinates so the B TMA can multicast.
    constexpr uint32_t LOAD_BLOCK_M    = BLOCK_M;
    constexpr uint32_t LOAD_BLOCK_N    = BLOCK_N;
    constexpr uint32_t kSwizzleAMode   = BLOCK_K * sizeof(a_dtype_t);   // 128
    constexpr uint32_t kLocalL1ActsSFGranK = 64;       // each CTA's local half
    constexpr uint32_t kL2ActsSFGranK  = 128;          // final L1 output / L2 input
    constexpr bool kClusterPairsL1SF =
        kClusterSize == 2 && BLOCK_M == 128 && BLOCK_N == 128 &&
        kNumEpilogueWarpgroups == 2 && WG_L1_OUT_BLOCK_N == kLocalL1ActsSFGranK;
    DG_STATIC_ASSERT(kClusterPairsL1SF,
                     "direct per-128 split L1 requires paired BM128/BN128 CTAs");
    DG_STATIC_ASSERT(L1_SHAPE_N / BLOCK_N <= 64,
                     "paired-N readiness must fit one 64-bit pool-block mask");
    DG_STATIC_ASSERT((L1_SHAPE_N / BLOCK_N) % kClusterSize == 0,
                     "every L1 N block must have a cluster peer");
    DG_STATIC_ASSERT(kIntermediateHidden / kL2ActsSFGranK <= kIntermediateHidden / 64,
                     "logical per-128 SF groups must fit retained physical capacity");

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
    // Keep a two-slot SFA stage allocation so the six-stage ring retains its
    // proven shared-memory/barrier placement. L1 reads only the first per-128
    // slot.
    constexpr uint32_t kL2SFAHalfStride =
        math::constexpr_align<uint32_t>(BLOCK_M * sizeof(float), 128u) / sizeof(float);
    constexpr uint32_t SMEM_SFA_SIZE_PER_STAGE = 2 * kL2SFAHalfStride * sizeof(float);
    // CD output: max of L1 FP8 (BLOCK_M * (BLOCK_N/2) * 1 byte * num_wg) and
    // L2 BF16 (BLOCK_M * BLOCK_N * 2 bytes * num_wg).
    constexpr uint32_t SMEM_CD_ACCUM_SIZE = 0u;
    constexpr uint32_t SMEM_CD_L1_SIZE =
        kNumEpilogueWarpgroups * WG_BLOCK_M * WG_L1_OUT_BLOCK_N * sizeof(cutlass::float_e4m3_t);
    constexpr uint32_t SMEM_CD_L2_SIZE = 0u;
    constexpr uint32_t SMEM_CD_OUTPUT_BASE_SIZE =
        SMEM_CD_L1_SIZE > SMEM_CD_L2_SIZE ? SMEM_CD_L1_SIZE : SMEM_CD_L2_SIZE;
    constexpr uint32_t SMEM_CD_OUTPUT_UNALIGNED_SIZE = SMEM_CD_OUTPUT_BASE_SIZE;
    constexpr uint32_t SMEM_CD_OUTPUT_SIZE = math::constexpr_align(
        SMEM_CD_OUTPUT_UNALIGNED_SIZE, kSharedMemoryAlignment);
    constexpr uint32_t SMEM_CD_SIZE = SMEM_CD_ACCUM_SIZE + SMEM_CD_OUTPUT_SIZE;

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
    auto smem_cd_l1 = reinterpret_cast<cutlass::float_e4m3_t*>(smem_cd_base);

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
                empty_barriers[i]->init(kClusterSize * kNumEpilogueWarpgroups);
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
        if (cute::elect_one_sync()) {
            // Arm the first DSM amax exchange before cluster peers leave the
            // initialization rendezvous. Every tile completes one float
            // transaction for each of its BLOCK_M rows.
            combine_barriers[0]->arrive_and_expect_tx(BLOCK_M * sizeof(float));
        }
    }
    if constexpr (kClusterSize > 1) {
        cute::cluster_sync();
    } else {
        __syncthreads();
    }

    // =====================================================================
    // Cluster-aware L1 scheduler
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
    // Separate named barriers let the two dequant teams publish K stages independently.
    constexpr uint32_t kDequantBarrierIdx = 8;
    constexpr uint32_t kAlternateDequantBarrierIdx = 9;
    const auto dequant_loaded_b_stage = [&](const uint32_t& s, const uint32_t& p,
                                            const uint32_t& k_block_idx,
                                            const uint32_t& non_epilogue_thread_idx) {
        if constexpr (kLoaderDequant) {
            if constexpr (kNumMMANonEpilogueWarps == 2 && kPackedBScratch && LOAD_BLOCK_N == 256) {
                full_barriers[s]->wait(p);
                #pragma unroll
                for (uint32_t row = non_epilogue_thread_idx; row < LOAD_BLOCK_N; row += 64u)
                    deep_gemm::nvfp4::dequant_smem_b_from_packed_mode2_nibble(
                        reinterpret_cast<uint8_t*>(smem_b[s]),
                        reinterpret_cast<const uint8_t*>(smem_packed_b[s]),
                        row, smem_nvfp4_lut);
                if (non_epilogue_thread_idx == 0)
                    dequant_barriers[s]->arrive();
            } else if constexpr (kNumMMANonEpilogueWarps == 4) {
                if constexpr (LOAD_BLOCK_N == 256) {
                    full_barriers[s]->wait(p);
                    const uint32_t dequant_tid = non_epilogue_thread_idx;
                    deep_gemm::nvfp4::dequant_smem_b_inplace_two_rows_mode2_nibble<
                        128u, kDequantBarrierIdx>(
                        reinterpret_cast<uint8_t*>(smem_b[s]), dequant_tid, smem_nvfp4_lut);
                    if (dequant_tid == 0)
                        dequant_barriers[s]->arrive();
                } else if constexpr (kDispatchDequant) {
                    if (non_epilogue_thread_idx >= 64u && (k_block_idx & 1u)) {
                        full_barriers[s]->wait(p);
                        const uint32_t dequant_tid = non_epilogue_thread_idx - 64u;
                        deep_gemm::nvfp4::dequant_smem_b_inplace_two_rows_mode2_nibble<
                            64u, kAlternateDequantBarrierIdx, true>(
                            reinterpret_cast<uint8_t*>(smem_b[s]), dequant_tid, smem_nvfp4_lut);
                        if (dequant_tid == 0)
                            dequant_barriers[s]->arrive();
                    }
                } else if (non_epilogue_thread_idx >= 64u) {
                    full_barriers[s]->wait(p);
                    const uint32_t dequant_tid = non_epilogue_thread_idx - 64u;
                    deep_gemm::nvfp4::dequant_smem_b_inplace_two_rows_mode2_nibble<
                        64u, kDequantBarrierIdx>(
                        reinterpret_cast<uint8_t*>(smem_b[s]), dequant_tid, smem_nvfp4_lut);
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

    // Register reconfiguration counts (chosen to fit in 64512 reg budget).
    // Dispatch-assisted dequant keeps the same 63488-register CTA budget:
    // 128*80 + 128*80 + 256*168.
    constexpr uint32_t kNumDispatchRegisters = kDispatchDequant ? 80 : 48;
    constexpr uint32_t kNumNonEpilogueRegisters =
        (kLoaderDequant && kNumEpilogueThreads == 128) ? 80 :
        (kLoaderDequant && kNumEpilogueThreads == 256 && LOAD_BLOCK_N == 256) ? 80 :
        (kDispatchDequant) ? 80 :
        (kLoaderDequant && kNumEpilogueThreads == 256) ? 64 : 40;
    constexpr uint32_t kNumEpilogueRegisters    =
        (kLoaderDequant && kNumEpilogueThreads == 256 && LOAD_BLOCK_N == 256) ? 184 :
        (kDispatchDequant) ? 168 :
        (kLoaderDequant && kNumEpilogueThreads == 256) ? 192 :
        (kSerialNWarpgroups or kWideNWarpgroups) ? 256 :
        208;
    DG_STATIC_ASSERT(kNumDispatchRegisters * kNumDispatchThreads +
                     kNumNonEpilogueRegisters * kNumNonEpilogueThreads +
                     kNumEpilogueRegisters * kNumEpilogueThreads <= 64512,
                     "Too many registers");

    constexpr uint32_t kDispatchGridSyncIndex = 0;

    constexpr uint32_t kProfileDispatchTotal = 0;
    constexpr uint32_t kProfileDispatchPull = 1;
    constexpr uint32_t kProfileMathLoop = 2;
    constexpr uint32_t kProfileGemmCore = 5;
    constexpr uint32_t kProfileL1Epilogue = 6;
    constexpr uint32_t kProfileLoaderDequant = 8;
    constexpr uint32_t kProfileMathDequantWait = 9;
    constexpr uint32_t kProfileL1TMAWait = 10;
    constexpr uint32_t kNumPhaseProfileMetrics = 14;
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
        scheduler.for_each_linear1_block([&](const uint32_t& local_expert_idx,
                                             const uint32_t& num_k_blocks,
                                             const uint32_t& m_block_idx, const uint32_t& n_block_idx) {
            func(std::integral_constant<sched::BlockPhase, sched::BlockPhase::Linear1>{},
                 local_expert_idx, num_k_blocks, m_block_idx, n_block_idx);
        });
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
        } else if constexpr (kDispatchDequant) {
            const uint32_t dequant_tid =
                (warp_idx - kNumActiveDispatchWarps) * 32 + lane_idx;
            for_each_selected_block([&](const auto&, const uint32_t&,
                                         const uint32_t& num_k_blocks,
                                         const uint32_t&, const uint32_t&) {
                for (uint32_t k_block_idx = 0; k_block_idx < num_k_blocks;
                     advance_pipeline(k_block_idx)) {
                    if ((k_block_idx & 1u) == 0) {
                        full_barriers[stage_idx]->wait(phase);
                        deep_gemm::nvfp4::dequant_smem_b_inplace_two_rows_mode2_nibble<
                            64u, kDequantBarrierIdx, true>(
                            reinterpret_cast<uint8_t*>(smem_b[stage_idx]), dequant_tid,
                            smem_nvfp4_lut);
                        if (dequant_tid == 0)
                            dequant_barriers[stage_idx]->arrive();
                    }
                }
            });
        }
        ptx::sync_unaligned(kNumDispatchThreads + kNumEpilogueThreads, kDispatchWithEpilogueBarrierIdx);
        

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
            const auto tensor_map_a_ptr = [&]() {
                return &tensor_map_l1_acts;
            }();
            const auto tensor_map_sfa_ptr = [&]() {
                return &tensor_map_l1_acts_sf;
            }();

            const uint32_t pool_block_idx = scheduler.get_current_pool_block_offset() + m_block_idx;
            const uint32_t valid_m = scheduler.template get_valid_m<false>();
            const bool has_valid_m = valid_m > 0;

            // Wait for the pool to be ready. Cluster peers can be dummy CTAs for
            // the tail M unit when an expert has an odd number of M blocks.
            const unsigned long long ready_wait_start = phase_profile_clock();
            if (has_valid_m) {
                
                    const auto ptr = workspace.get_l1_arrival_count_ptr(pool_block_idx);
                    const auto expected = valid_m;
                    while (ptx::ld_acq(ptr) != expected);
                
            }
            const unsigned long long ready_wait_end = phase_profile_clock();
            
            for (uint32_t k_block_idx = 0; k_block_idx < num_k_blocks; advance_pipeline(k_block_idx)) {
                empty_barriers[stage_idx]->wait(phase ^ 1);

                if (cute::elect_one_sync()) {
                    if (has_valid_m) {
                    const uint32_t m_idx = pool_block_idx * BLOCK_M;
                    const uint32_t k_idx = k_block_idx * BLOCK_K;

                    // Paired N CTAs share the same M/K coordinates, so the
                    // cluster leader multicasts A and SFA to both CTAs.
                    tma::copy<BLOCK_K, LOAD_BLOCK_M, kSwizzleAMode, a_dtype_t>(
                        tensor_map_a_ptr, full_barriers[stage_idx], smem_a[stage_idx],
                        k_idx, m_idx, kClusterSize);

                    // TMA load SFA
                    
                        // L1 SFA per-128: load (BLOCK_M, 1) at K=k_block_idx
                        tma::copy<BLOCK_M, 1, 0, float>(
                            tensor_map_sfa_ptr, full_barriers[stage_idx], smem_sfa[stage_idx],
                            m_idx, k_block_idx, kClusterSize);
                        full_barriers[stage_idx]->arrive_and_expect_tx(
                            SMEM_A_SIZE_PER_STAGE + BLOCK_M * sizeof(float));
                    
                    } else {
                        full_barriers[stage_idx]->arrive();
                    }
                }
                __syncwarp();
                dequant_loaded_b_stage(stage_idx, phase, k_block_idx, lane_idx);
            }
        });

    } else if (warp_idx == kNumDispatchWarps + 1) {
        cutlass::arch::warpgroup_reg_dealloc<kNumNonEpilogueRegisters>();

        for_each_selected_block([&](const auto& block_phase,
                                     const uint32_t& local_expert_idx,
                                     const uint32_t& num_k_blocks,
                                     const uint32_t& m_block_idx, const uint32_t& n_block_idx) {
            const auto tensor_map_b_ptr = [&]() {
                return &tensor_map_l1_weights;
            }();
            constexpr uint32_t shape_n = L1_SHAPE_N;

            for (uint32_t k_block_idx = 0; k_block_idx < num_k_blocks; advance_pipeline(k_block_idx)) {
                empty_barriers[stage_idx]->wait(phase ^ 1);

                const uint32_t n_idx = local_expert_idx * shape_n + n_block_idx * BLOCK_N;
                // NVFP4 fused B+scale layout stores 64B packed FP4 + 16B
                // UE4M3 scale per BK128 row.
                const uint32_t k_idx = k_block_idx * B_LOAD_BYTES_PER_ROW;
                if (cute::elect_one_sync()) {
                    // The peer owns the adjacent N block, therefore B must be
                    // loaded independently rather than multicast.
                    tma::copy<B_LOAD_BYTES_PER_ROW, LOAD_BLOCK_N, 0, b_dtype_t>(
                        tensor_map_b_ptr, full_barriers[stage_idx],
                        kPackedBScratch ? smem_packed_b[stage_idx] : smem_b[stage_idx],
                        k_idx, n_idx, 1);
                    full_barriers[stage_idx]->arrive_and_expect_tx(SMEM_B_LOAD_SIZE_PER_STAGE);
                }
                __syncwarp();
                dequant_loaded_b_stage(stage_idx, phase, k_block_idx, 32u + lane_idx);
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
                    dequant_loaded_b_stage(stage_idx, phase, k_block_idx, non_epilogue_thread_idx);
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
            if (warp_idx_in_wg != 0)
                return;
            if constexpr (kClusterSize == 1) {
                if (lane_idx == 0)
                    empty_barriers[s]->arrive();
            } else {
                if (lane_idx < kClusterSize)
                    empty_barriers[s]->arrive(lane_idx);
            }
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
        uint32_t pair_scale_phase = 0;

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
            const bool inactive_math_wg = row_block_offset >= valid_m;


            if constexpr (kLoaderDequant && (!kSplitNWarpgroups)) {
                if (inactive_math_wg) {
                    for (uint32_t k_block_idx = 0; k_block_idx < num_k_blocks; advance_pipeline(k_block_idx)) {
                        dequant_barriers[stage_idx]->wait(phase);
                        arrive_empty_barrier(stage_idx);
                        __syncwarp();
                    }
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
                    
                        scale_a_0_lo = ptx::ld_shared(smem_sfa[stage_idx] + row_offset_r0);
                        scale_a_1_lo = ptx::ld_shared(smem_sfa[stage_idx] + row_offset_r1);
                    

                    #pragma unroll
                    for (uint32_t serial_n_idx = 0; serial_n_idx < kNumSerialN; ++serial_n_idx) {
                        const uint32_t serial_wg_n_idx = serial_n_idx * WG_BLOCK_N;
                        float gate_sf = 0.0f, up_sf = 0.0f;
                        
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
                        
                    }

                    arrive_empty_barrier(stage_idx);
                    __syncwarp();
                }

                if (row_block_offset >= valid_m) {
                    ptx::sync_aligned(kNumEpilogueThreads, kEpilogueFullBarrierIdx);
                    return;
                }

                
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
                            float g_r0_c0 = final_accum[serial_n_idx][gate*4 + 0]; clamp_gate(g_r0_c0);
                            float g_r0_c1 = final_accum[serial_n_idx][gate*4 + 1]; clamp_gate(g_r0_c1);
                            float g_r1_c0 = final_accum[serial_n_idx][gate*4 + 2]; clamp_gate(g_r1_c0);
                            float g_r1_c1 = final_accum[serial_n_idx][gate*4 + 3]; clamp_gate(g_r1_c1);
                            float u_r0_c0 = final_accum[serial_n_idx][up*4   + 0]; clamp_up(u_r0_c0);
                            float u_r0_c1 = final_accum[serial_n_idx][up*4   + 1]; clamp_up(u_r0_c1);
                            float u_r1_c0 = final_accum[serial_n_idx][up*4   + 2]; clamp_up(u_r1_c0);
                            float u_r1_c1 = final_accum[serial_n_idx][up*4   + 3]; clamp_up(u_r1_c1);
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
                
                return;
            }

            // ---------------- GEMM ----------------
            using WGMMA = L1WGMMA;
            constexpr uint32_t kAccumPerThread = WGMMA::kNumAccum;  // 64 for M=64,N=128
            float final_accum[kAccumPerThread] = {};
            float accum[kAccumPerThread];

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
                    full_barriers[stage_idx]->wait(phase);

                    // NVFP4: expand packed FP4 (first 8KB) -> FP8 (full 16KB) in smem_b.
                    // Each math-WG thread handles one row (64 -> 128 bytes).
                    {
                        DG_STATIC_ASSERT(kPackedBScratch,
                                         "Math-side NVFP4 fused-layout dequant requires packed scratch");
                        const uint32_t _tid_in_wg = epilogue_thread_idx;
                        deep_gemm::nvfp4::dequant_smem_b_from_packed_mode2_nibble(
                            reinterpret_cast<uint8_t*>(smem_b[stage_idx]),
                            reinterpret_cast<const uint8_t*>(smem_packed_b[stage_idx]),
                            _tid_in_wg, smem_nvfp4_lut);
                    }
                }

                // Read SF (must precede warpgroup_arrive)
                const float scale_a_0 =
                    ptx::ld_shared(smem_sfa[stage_idx] + row_offset_r0);
                const float scale_a_1 =
                    ptx::ld_shared(smem_sfa[stage_idx] + row_offset_r1);

                // NVFP4 UE4M3 weight scales are applied during FP4 -> FP8 smem
                // expansion, so the WGMMA accumulator only needs activation SF.

                
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
                        final_accum[i*4+0] += scale_a_0 * accum[i*4+0];
                        final_accum[i*4+1] += scale_a_0 * accum[i*4+1];
                        final_accum[i*4+2] += scale_a_1 * accum[i*4+2];
                        final_accum[i*4+3] += scale_a_1 * accum[i*4+3];
                    }
                
            }
            };

            if (!inactive_math_wg)
                run_default_gemm_loop();

            const unsigned long long block_gemm_end = phase_profile_clock();
            if (epilogue_warp_idx == 0 and lane_idx == 0)
                phase_profile_record(kProfileGemmCore, block_gemm_end - block_gemm_start);

            // Even a fully invalid tail warpgroup must participate in the
            // CTA-wide scale rendezvous. It carries zero accumulators and only
            // writes padding rows, which L2 never scatters.

            const unsigned long long block_epilogue_start = phase_profile_clock();
            const float l1_global_scale = l1_global_scales == nullptr ? 1.0f : __ldg(l1_global_scales + local_expert_idx);
            
                // ---------------- L1 EPILOGUE: SwiGLU + FP8 quantize + TMA store ----------------
                // Layout in `final_accum`:
                //   16 chunks of 8 N-cols, each chunk = 4 floats per thread = (r0c0, r0c1, r1c0, r1c1).
                //   Gate chunks: even (0, 2, ..., 14). Up chunks: odd (1, 3, ..., 15).
                //   Pair `p` ∈ [0, 8): gate chunk = 2p, up chunk = 2p+1.
                //
                // For each pair we produce 4 post-SwiGLU floats per thread, mapped to
                // output cols (p*8 + col_idx*2 + {0,1}) for both r0 and r1.

                constexpr uint32_t kNumPairs = kAccumPerThread / 8;
                constexpr uint32_t kNumSFGroups =
                    WG_L1_OUT_BLOCK_N / kLocalL1ActsSFGranK;
                DG_STATIC_ASSERT(kClusterPairsL1SF && kNumSFGroups == 1,
                                 "each paired CTA must own one local 64-column SF group");
                float swiglu_r0[kNumPairs][2];
                float swiglu_r1[kNumPairs][2];

                // Each CTA computes one local 64-column amax for all 128 rows.
                // Its adjacent-N cluster peer owns the other half.
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

                // Direct per-128 rendezvous across adjacent-N cluster CTAs.
                // Pack the two output rows owned by each four-lane group into
                // one contiguous float2 DSM slot. This keeps the same 512-byte
                // transaction/barrier protocol while halving DSM store and
                // shared-load instructions versus row-strided scalar slots.
                auto* l2_sf_base = l2_sf_buffer.get_base_ptr<float>();
                const uint32_t token_r0 = pool_block_idx * BLOCK_M + row_offset_r0;
                const uint32_t token_r1 = pool_block_idx * BLOCK_M + row_offset_r1;
                const uint32_t dense_sf_group_idx = n_block_idx / kClusterSize;
                const uint32_t peer_cta_rank = cute::block_rank_in_cluster() ^ 1u;
                auto* smem_cd_l1_wg = smem_cd_l1
                    + (kSplitNWarpgroups ? 0 : epilogue_wg_idx * WG_BLOCK_M * L1_OUT_BLOCK_N);
                const uint32_t row_group_idx = epilogue_thread_idx / 4;
                auto* peer_amax_pair_slot =
                    reinterpret_cast<float2*>(smem_cd_l1) + row_group_idx;

                if (col_idx == 0) {
                    ptx::st_shared_cluster_async(
                        peer_amax_pair_slot, make_float2(amax_r0[0], amax_r1[0]),
                        combine_barriers[0], peer_cta_rank);
                }
                if (epilogue_thread_idx == 0) {
                    combine_barriers[0]->wait(pair_scale_phase);
                    pair_scale_phase ^= 1u;
                    // Pre-arm the next phase. The current quantize/store tail
                    // and the following GEMM separate this from peer stores.
                    combine_barriers[0]->arrive_and_expect_tx(BLOCK_M * sizeof(float));
                }
                ptx::sync_aligned(kNumEpilogueThreads, kEpilogueFullBarrierIdx);

                float sf_r0 = 0.0f, sf_inv_r0 = 0.0f;
                float sf_r1 = 0.0f, sf_inv_r1 = 0.0f;
                if (col_idx == 0) {
                    const float2 peer_amax_pair = ptx::ld_shared(peer_amax_pair_slot);
                    const float2 common_amax_pair = {
                        cute::max(amax_r0[0], peer_amax_pair.x),
                        cute::max(amax_r1[0], peer_amax_pair.y)
                    };
                    float2 sf_pair, sf_inv_pair;
                    math::get_e4m3_sf_and_sf_inv(
                        common_amax_pair, sf_pair, sf_inv_pair);
                    sf_r0 = sf_pair.x;
                    sf_inv_r0 = sf_inv_pair.x;
                    sf_r1 = sf_pair.y;
                    sf_inv_r1 = sf_inv_pair.y;
                }
                const uint32_t row_group_leader = lane_idx & ~3u;
                sf_inv_r0 = __shfl_sync(0xffffffff, sf_inv_r0, row_group_leader);
                sf_inv_r1 = __shfl_sync(0xffffffff, sf_inv_r1, row_group_leader);

                // Both CTAs derive the same scale; only the even-N owner
                // publishes the dense logical per-128 SF entry.
                if ((n_block_idx & 1u) == 0 && col_idx == 0) {
                    if (valid_r0)
                        l2_sf_base[dense_sf_group_idx * kNumPaddedSFPoolTokens + token_r0] = sf_r0;
                    if (valid_r1)
                        l2_sf_base[dense_sf_group_idx * kNumPaddedSFPoolTokens + token_r1] = sf_r1;
                }

                // Quantize and write to smem_cd_l1 (row-major, no swizzle).
                constexpr uint32_t l1_store_stage = 0u;
                smem_cd_l1_wg +=
                    l1_store_stage * kNumEpilogueWarpgroups * WG_BLOCK_M * L1_OUT_BLOCK_N;
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
                        smem_cd_l1_wg + r_0 * L1_OUT_BLOCK_N + wg_l1_out_n_idx + col);
                    auto* p1 = reinterpret_cast<uint16_t*>(
                        smem_cd_l1_wg + r_1 * L1_OUT_BLOCK_N + wg_l1_out_n_idx + col);
                    if (valid_r0)
                        *p0 = r0_pair.__x;
                    if (valid_r1)
                        *p1 = r1_pair.__x;
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
                                    }
                const unsigned long long block_epilogue_end = phase_profile_clock();
                if (epilogue_warp_idx == 0 and lane_idx == 0)
                    phase_profile_record(kProfileL1Epilogue, block_epilogue_end - block_epilogue_start);
            
        });
        const unsigned long long math_loop_end = phase_profile_clock();
        if (epilogue_warp_idx == 0 and lane_idx == 0)
            phase_profile_record(kProfileMathLoop, math_loop_end - math_loop_start);

        
            ptx::sync_unaligned(kNumDispatchThreads + kNumEpilogueThreads, kDispatchWithEpilogueBarrierIdx);
            return;
        
    }
#else
    if (blockIdx.x == 0 and threadIdx.x == 0)
        DG_DEVICE_ASSERT(false and "This kernel only supports sm_90");
#endif
