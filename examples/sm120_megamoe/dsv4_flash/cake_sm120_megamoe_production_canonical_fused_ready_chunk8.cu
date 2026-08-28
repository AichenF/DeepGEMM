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

__device__ __forceinline__ float max_noftz(float a, float b) {
    float c;
    asm("max.f32 %0, %1, %2;" : "=f"(c) : "f"(a), "f"(b));
    return c;
}

__device__ __forceinline__ float approx_rcp(float x) {
    float y;
    asm("rcp.approx.ftz.f32 %0, %1;" : "=f"(y) : "f"(x));
    return y;
}

#include <cuda/atomic>
#include <deep_gemm/impls/sm120_fp8_fp4_gemm_1d1d.cuh>
#include "megamoe_shape_config.cuh"
#include "megamoe_phase_trace.cuh"

static constexpr uint32_t kCakeSm120InvalidResultRowBase = ~0u;

__device__ __forceinline__ void cake_sm120_ready_mirror_pair_cached(
    __nv_bfloat16 const* pair, unsigned int local_row, unsigned int column,
    uint32_t const* result_row_base,
    __nv_bfloat16* result_out)
{
    const uint32_t base = result_row_base[local_row];
    if (base == kCakeSm120InvalidResultRowBase) return;
    const uint32_t out = base + column;
    result_out[out] = pair[0];
    result_out[out + 1ull] = pair[1];
}

__device__ __forceinline__ void cake_sm120_ready_wait_one(
    unsigned int* address)
{
    cuda::atomic_ref<unsigned int, cuda::thread_scope_device> flag(*address);
    while (flag.load(cuda::memory_order_acquire) != 1u) { }
}

namespace deep_gemm::sched {
template <GemmType kGemmType,
          uint32_t BLOCK_M, uint32_t BLOCK_N,
          uint32_t kNumGroups,
          uint32_t kNumMulticast, bool kIsMulticastOnA,
          uint32_t kNumSMs,
          bool kEnsureZeroPadding = true,
          uint32_t kKAlignment = 128u,     // psum k-group start alignment
          uint32_t kSFKSpan = 512u,        // K covered by one k-grouped SF row
          uint32_t kNum1DBlocksPerGroup = get_num_1d_blocks_per_group<kGemmType, BLOCK_M, BLOCK_N, kNumSMs, kIsMulticastOnA>(),
          uint32_t kSplitKFactor = 1>
struct CakeChunk8Scheduler {
    int current_iter = -1;
    uint32_t cake_assigned_m_block;
    uint32_t cake_assigned_n_block;

    // Block configs
    uint32_t num_blocks;
    uint32_t num_mn_blocks;
    uint32_t num_m_blocks;
    uint32_t num_n_blocks;

    // Split-K state
    uint32_t split_k_idx;
    uint32_t k_partition_start;
    uint32_t k_partition_end;

    // For SM90 multicast checks
    uint32_t num_blocks_in_group;
    bool is_peer_cta_alive = true;

    // For grouped GEMM
    int* grouped_layout;
    uint32_t current_group_idx = 0;
    // Only used for masked layout
    uint32_t current_m_cumsum = 0;
    // Only used for contiguous psum layout
    uint32_t last_psum_m = 0, current_psum_m, current_m_block_cumsum = 0;
    // Only used for k-grouped layout
    uint32_t current_shape_k, current_num_valid_groups = 0, current_k_cumsum = 0, current_sf_k_cumsum = 0;
    // NOTES: only used by the non-psum path; the psum path never reads them.
    uint32_t next_group_idx, next_shape_k;
    // Only used for `KGroupedContiguousWithPsumLayout`
    uint32_t current_k_start = 0, current_k_end = 0;

    // Only used for k-grouped gemm
    CUTLASS_DEVICE void get_next_k_group(uint32_t &group_idx, uint32_t &shape_k) const {
        for (; group_idx < kNumGroups; ++ group_idx) {
            shape_k = grouped_layout[group_idx];
            if (shape_k > 0)
                break;
        }
    }

    CUTLASS_DEVICE void get_next_psum_k_group(uint32_t &group_idx, uint32_t &shape_k,
                                               uint32_t &k_start, uint32_t &k_end) const {
        // NOTES: `grouped_layout[i]` is the psum end offset (K elements); each group starts at `align(prev_end, kKAlignment)`. Skip empty groups.
        for (; group_idx < kNumGroups; ++ group_idx) {
            const auto next_k_end = static_cast<uint32_t>(grouped_layout[group_idx]);
            k_start = math::align(k_end, kKAlignment);
            shape_k = next_k_end - k_start;
            k_end = next_k_end;
            if (shape_k > 0)
                break;
        }
    }

    CUTLASS_DEVICE explicit CakeChunk8Scheduler(const uint32_t& shape_m, const uint32_t& shape_n,
                                       const uint32_t& shape_k, int* grouped_layout,
                                       uint32_t assigned_m_block,
                                       uint32_t assigned_n_block) {
        cake_assigned_m_block = assigned_m_block;
        cake_assigned_n_block = assigned_n_block;
        num_m_blocks = math::ceil_div(shape_m, BLOCK_M);
        num_n_blocks = math::ceil_div(shape_n, BLOCK_N);
        num_mn_blocks = num_m_blocks * num_n_blocks;
        current_shape_k = shape_k;
        if constexpr (kGemmType == GemmType::Normal or kGemmType == GemmType::Batched) {
            num_blocks = num_mn_blocks * kSplitKFactor;
        } else if constexpr (kGemmType == GemmType::MGroupedContiguous) {
            num_blocks = num_m_blocks * num_n_blocks;
            this->grouped_layout = grouped_layout;
        } else if constexpr (kGemmType == GemmType::MGroupedMasked) {
            this->grouped_layout = grouped_layout;
        } else if constexpr (kGemmType == GemmType::MGroupedContiguousWithPsumLayout) {
            this->grouped_layout = grouped_layout;
            current_psum_m = grouped_layout[0];
            num_m_blocks = math::ceil_div(current_psum_m, BLOCK_M);
        } else if constexpr (is_k_grouped_contiguous(kGemmType)) {
            num_blocks = num_m_blocks * num_n_blocks;
            this->grouped_layout = grouped_layout;
            if constexpr (kGemmType == GemmType::KGroupedContiguousWithPsumLayout) {
                get_next_psum_k_group(current_group_idx, current_shape_k, current_k_start, current_k_end);
            } else {
                get_next_k_group(current_group_idx, current_shape_k);
                next_group_idx = current_group_idx + 1;
                get_next_k_group(next_group_idx, next_shape_k);
            }
        }
    }

    CUTLASS_DEVICE void get_swizzled_block_idx(const uint32_t& block_idx, uint32_t& m_block_idx, uint32_t& n_block_idx) {
        DG_STATIC_ASSERT(kNum1DBlocksPerGroup % kNumMulticast == 0, "Invalid group size");

        // Swizzle for better L2 usages
        const auto primary_num_blocks = kIsMulticastOnA ? num_n_blocks : num_m_blocks;
        const auto secondary_num_blocks = kIsMulticastOnA ? num_m_blocks : num_n_blocks;
        const auto num_blocks_per_group = secondary_num_blocks * kNum1DBlocksPerGroup;
        const auto group_idx = block_idx / num_blocks_per_group;
        auto first_block_idx = group_idx * kNum1DBlocksPerGroup;
        auto in_group_idx = block_idx % num_blocks_per_group;
        num_blocks_in_group = min(kNum1DBlocksPerGroup, primary_num_blocks - first_block_idx);

        // Fix unaligned TMA multicast
        // NOTES: for SM90 only, as SM90 can dynamically disable TMA multicast
        // while SM100 uses 2-CTA, which can not be dynamically disabled
#if __CUDA_ARCH__ < 1000
        if (kNumMulticast > 1 and num_blocks_in_group % 2 != 0) {
            if (in_group_idx < (num_blocks_in_group ^ 1) * secondary_num_blocks) {
                num_blocks_in_group = num_blocks_in_group ^ 1;
            } else {
                in_group_idx = in_group_idx - (num_blocks_in_group ^ 1) * secondary_num_blocks;
                first_block_idx += num_blocks_in_group ^ 1;
                num_blocks_in_group = 1;
            }
        }
#endif

        // Convert to final M/N block indices
        // `kIsMulticastOnA == true` leads to groups on N
        if constexpr (kIsMulticastOnA) {
            m_block_idx = in_group_idx / num_blocks_in_group;
            n_block_idx = first_block_idx + in_group_idx % num_blocks_in_group;
        } else {
            m_block_idx = first_block_idx + in_group_idx % num_blocks_in_group;
            n_block_idx = in_group_idx / num_blocks_in_group;
        }
    }

    template <bool kWithGroupOffset, IndexType kIndexType = IndexType::MN>
    CUTLASS_DEVICE uint32_t get_global_idx(const uint32_t shape_dim, const uint32_t block_size,
                                             const uint32_t& block_idx, const uint32_t& m_block_idx = 0) {
        if constexpr (kGemmType == GemmType::Normal) {
            return block_idx * block_size;
        } else if constexpr (kGemmType == GemmType::MGroupedContiguous) {
            const auto offset = kWithGroupOffset ? cute::max(0, grouped_layout[m_block_idx * BLOCK_M]) : 0;
            return offset * shape_dim + block_idx * block_size;
        } else if constexpr (kGemmType == GemmType::MGroupedMasked or kGemmType == GemmType::MGroupedContiguousWithPsumLayout) {
            const auto offset = kWithGroupOffset ? current_group_idx : 0;
            return offset * shape_dim + block_idx * block_size;
        } else if constexpr (is_k_grouped_contiguous(kGemmType)) {
            auto offset = 0;
            if constexpr (kWithGroupOffset) {
                if constexpr (kIndexType == IndexType::MN) {
                    offset = current_group_idx * shape_dim;
                } else if constexpr (kIndexType == IndexType::K) {
                    if constexpr (kGemmType == GemmType::KGroupedContiguousWithPsumLayout)
                        offset = current_k_start;
                    else
                        offset = current_k_cumsum;
                } else if constexpr (kIndexType == IndexType::SF_K) {
                    offset = current_sf_k_cumsum;
                }
            }
            return offset + block_idx * block_size;
        } else if constexpr (kGemmType == GemmType::Batched) {
            // Ignore kWithGroupOffset, and apply offset for IndexType::SF_K
            const auto offset = kIndexType == IndexType::SF_K ? current_group_idx : 0;
            return offset * shape_dim + block_idx * block_size;
        }
    }

    // For swap A/B and psum layout only
    CUTLASS_DEVICE uint32_t get_aligned_effective_m_in_block(const uint32_t& m_block_idx) const {
        constexpr uint32_t UMMA_STEP_N = 16;
        DG_STATIC_ASSERT(BLOCK_M % UMMA_STEP_N == 0, "Invalid alignment");
        if constexpr (kGemmType == GemmType::MGroupedContiguousWithPsumLayout and not kEnsureZeroPadding)
            return math::align(m_block_idx == last_psum_m / BLOCK_M + num_m_blocks - 1 ? current_psum_m - m_block_idx * BLOCK_M : BLOCK_M, UMMA_STEP_N);
        return BLOCK_M;
    }

    CUTLASS_DEVICE bool get_next_block(
        uint32_t& m_block_idx, uint32_t& n_block_idx) {
        // The global queue assigns one task-major eight-tile chunk.  The
        // canonical donor keeps its three-stage TMA/mbarrier pipeline alive
        // while walking those consecutive physical N128 tiles.
        const int chunk_offset = ++current_iter;
        const uint32_t next_n = cake_assigned_n_block +
            static_cast<uint32_t>(chunk_offset);
        if (cake_assigned_m_block >= num_m_blocks || chunk_offset < 0 ||
            chunk_offset >= 8 || next_n >= num_n_blocks)
            return false;
        m_block_idx = cake_assigned_m_block;
        n_block_idx = next_n;
        split_k_idx = 0;
        return true;
    }

    // For SM90 only
    CUTLASS_DEVICE bool is_tma_multicast_valid(const uint32_t& m_block_idx) const {
        if (num_blocks_in_group == 1)
            return false;
        if constexpr (kGemmType == GemmType::Normal or kGemmType == GemmType::MGroupedMasked or
                      is_k_grouped_contiguous(kGemmType) or kGemmType == GemmType::Batched or
                      kGemmType == GemmType::MGroupedContiguousWithPsumLayout) {
            return true;
        } else {
            DG_STATIC_ASSERT(kGemmType == GemmType::MGroupedContiguous, "Invalid Gemm type");
            if constexpr (kIsMulticastOnA) {
                return true;
            } else {
                const auto group_idx = grouped_layout[m_block_idx * BLOCK_M];
                const auto peer_group_idx = grouped_layout[(m_block_idx ^ 1) * BLOCK_M];
                return group_idx == peer_group_idx;
            }
        }
    }

    // For SM90 only
    CUTLASS_DEVICE bool is_computation_valid(const uint32_t& m_block_idx, const uint32_t& m_offset) const {
        if constexpr (kGemmType == GemmType::Normal or kGemmType == GemmType::Batched) {
            return true;
        } else if constexpr (kGemmType == GemmType::MGroupedContiguous) {
            return grouped_layout[m_offset + m_block_idx * BLOCK_M] >= 0;
        } else if constexpr (kGemmType == GemmType::MGroupedMasked) {
            return m_offset + m_block_idx * BLOCK_M < grouped_layout[current_group_idx];
        } else if constexpr (kGemmType == GemmType::MGroupedContiguousWithPsumLayout) {
            return m_offset + m_block_idx * BLOCK_M < current_psum_m;
        } else {
            DG_TRAP_ONLY_DEVICE_ASSERT(false);
        }
    }
};
} // namespace deep_gemm::sched

namespace deep_gemm {
template <uint32_t SHAPE_M, uint32_t SHAPE_N, uint32_t SHAPE_K,
          uint32_t kGranKA, uint32_t kGranKB,
          uint32_t kNumGroups,
          uint32_t BLOCK_M, uint32_t BLOCK_N, uint32_t BLOCK_K,
          uint32_t kSwizzleAMode, uint32_t kSwizzleBMode,
          uint32_t kSwizzleCDMode,
          uint32_t kNumStages,
          uint32_t kNumTMAThreads, uint32_t kNumMathThreads,
          uint32_t kNumSMs,
          GemmType kGemmType, bool kWithAccumulation,
          typename cd_dtype_t,
          typename epilogue_type_t = epilogue::transform::EpilogueIdentity,
          bool kIsFP4 = false,
          bool kBIsFP4 = false,
          bool kAIsFP4 = false,
          bool kBKMajor = true,
          bool kKGroupedConstantStride = false,
          uint32_t kEpiSubM = BLOCK_M,
          uint32_t kSplitKFactor = 1>
__device__ __forceinline__ void
cake_sm120_canonical_ready_chunk8_w1_gemm(cd_dtype_t* gmem_d, const cd_dtype_t* gmem_c,
                             __nv_fp8_e4m3* gmem_a_ptr, __nv_fp8_e4m3* gmem_b_ptr,
                             int* grouped_layout,
                             cute::TmaDescriptor* tensor_map_buffer,
                             float* gmem_workspace,
                             uint32_t shape_m, uint32_t shape_n, uint32_t shape_k,
                             uint32_t stride_cd_m, uint32_t stride_cd_n, uint32_t stride_cd_batch,
                             const cute::TmaDescriptor& tensor_map_a_base,
                             const cute::TmaDescriptor& tensor_map_b_base,
                             const cute::TmaDescriptor& tensor_map_sfa,
                             const cute::TmaDescriptor& tensor_map_sfb,
                             const cute::TmaDescriptor& tensor_map_cd,
                             uint32_t cake_assigned_m_block,
                             uint32_t cake_assigned_n_block,
                             unsigned int* cake_w1_warp_done,
                             unsigned int* cake_w1_task_ready,
                             unsigned int* cake_w1_tiles_completed) {
#if (defined(__CUDA_ARCH__) and (__CUDA_ARCH__ >= 1200)) or defined(__CLION_IDE__)
    namespace sm120_mma = mma::sm120;
    using Barrier = cutlass::arch::ClusterTransactionBarrier;

    static constexpr uint32_t MMA_M = 16;
    static constexpr uint32_t MMA_N = 8;
    static constexpr uint32_t MMA_K = kIsFP4 ? sm120_mma::FP4_MMA_K : sm120_mma::FP8_MMA_K;
    static constexpr uint32_t MMA_ACCUM = 4;

    DG_STATIC_ASSERT(cute::is_same_v<cd_dtype_t, float> or cute::is_same_v<cd_dtype_t, cutlass::bfloat16_t>,
                     "Only float or bfloat16 output supported");
    DG_STATIC_ASSERT(!(kIsFP4 && kBIsFP4), "Use kIsFP4 for symmetric FP4x4, not kBIsFP4");
    // kAIsFP4 = mixed FP4_A x FP8_B (swapAB of the FP8xFP4 mixed path): A fp4-unpacked at k32, B fp8.
    DG_STATIC_ASSERT(!(kAIsFP4 && (kIsFP4 || kBIsFP4)), "kAIsFP4 (fp4_A x fp8_B) is exclusive");
    DG_STATIC_ASSERT(!kBIsFP4 || kBKMajor, "Mixed FP8xFP4 requires K-major B");
    DG_STATIC_ASSERT(kNumTMAThreads > 0, "SM120a always uses warp-specialized pipeline");
    DG_STATIC_ASSERT(kNumMathThreads % 32 == 0, "Invalid math threads");
    DG_STATIC_ASSERT(BLOCK_M % MMA_M == 0 and BLOCK_N % MMA_N == 0 and BLOCK_K % MMA_K == 0, "Invalid block dims");

    static constexpr uint32_t kNumSFAStagesPerLoad = (4 * kGranKA) / BLOCK_K;
    static constexpr uint32_t kNumSFBStagesPerLoad = (4 * kGranKB) / BLOCK_K;

    static constexpr uint32_t kNumMathWarps = kNumMathThreads / 32;
    static constexpr uint32_t kNTiles = BLOCK_N / MMA_N;
    static constexpr uint32_t kKSteps = BLOCK_K / MMA_K;

    // Cooperative warp layout: warps split across M and N dimensions
    static constexpr uint32_t kNWarps = 2;
    static constexpr uint32_t kMWarps = kNumMathWarps / kNWarps;
    static constexpr uint32_t kMTilesPerWarp = BLOCK_M / kMWarps / MMA_M;
    static constexpr uint32_t kNTilesPerWarp = kNTiles / kNWarps;
    static constexpr uint32_t kAccumPerWarp = kMTilesPerWarp * kNTilesPerWarp * MMA_ACCUM;

    DG_STATIC_ASSERT(BLOCK_M == kMWarps * kMTilesPerWarp * MMA_M, "M tiles must divide evenly");
    DG_STATIC_ASSERT(kNTiles % kNWarps == 0, "N tiles must divide evenly among N warps");
    DG_STATIC_ASSERT(not kBKMajor or kNTilesPerWarp >= 1, "Need at least 1 N-tile per warp");

    static constexpr uint32_t kTMARegisters = 40;
    static constexpr uint32_t kMMARegisters = 232;

    // SMEM D buffer for TMA store epilogue (sub-tile: kEpiSubM rows at a time)
    static constexpr uint32_t kSafeSwizzleCDMode = kSwizzleCDMode > 0 ? kSwizzleCDMode : 1;
    static constexpr bool kUseTMAStoreEpilogue = kSwizzleCDMode > 0
        and BLOCK_N * sizeof(cd_dtype_t) >= kSwizzleCDMode
        and (BLOCK_N * sizeof(cd_dtype_t)) % kSafeSwizzleCDMode == 0;
    static constexpr uint32_t kNumEpiMSubs = kUseTMAStoreEpilogue ? (BLOCK_M / kEpiSubM) : 0;
    static constexpr uint32_t SMEM_D = kUseTMAStoreEpilogue
        ? static_cast<uint32_t>((BLOCK_N * sizeof(cd_dtype_t) / kSwizzleCDMode) * kSwizzleCDMode * kEpiSubM)
        : 0u;
    static constexpr uint32_t kSwizzleCDShift = kSwizzleCDMode > 0 ? (7 - __builtin_ctz(kSwizzleCDMode)) : 0;
    static constexpr uint32_t kSwizzleCDMask = kSwizzleCDMode > 0 ? (kSwizzleCDMode / 16 - 1) : 0;
    static constexpr uint32_t kTMAStoreInnerDim = kSwizzleCDMode / sizeof(cd_dtype_t);
    static constexpr uint32_t kNumTMAStores = kUseTMAStoreEpilogue
        ? BLOCK_N * sizeof(cd_dtype_t) / kSwizzleCDMode : 0;

    // FP4 uses packed SMEM (4-bit per element = 0.5 bytes), FP8 uses 1 byte per element.
    static constexpr uint32_t kSMEMKBytes = kIsFP4 ? (BLOCK_K / 2) : BLOCK_K;
    static constexpr uint32_t SMEM_A  = BLOCK_M * kSMEMKBytes;
    static constexpr uint32_t SMEM_B  = kBKMajor ? (BLOCK_N * kSMEMKBytes) : (BLOCK_K * BLOCK_N);
    static constexpr uint32_t SMEM_SFA = math::constexpr_align(static_cast<uint32_t>(BLOCK_M * sizeof(int32_t)), 128u);
    static constexpr uint32_t SMEM_SFB = math::constexpr_align(static_cast<uint32_t>(BLOCK_N * sizeof(int32_t)), 128u);
    static constexpr uint32_t TMA_SFA_BYTES = BLOCK_M * sizeof(int32_t);
    static constexpr uint32_t TMA_SFB_BYTES = BLOCK_N * sizeof(int32_t);
    // TMA mbarrier reports GMEM bytes. For .b4x16_p64 (kBIsFP4): GMEM = SMEM/2 (packed).
    // For packed FP4 (kIsFP4): SMEM already uses packed size, so SMEM_B = GMEM bytes.
    static constexpr uint32_t TMA_B_BYTES = kBIsFP4 ? (SMEM_B / 2) : SMEM_B;
    // kAIsFP4: A is fp4 packed in GMEM (.b4x16 expands to unpacked SMEM), so GMEM = SMEM_A/2.
    static constexpr uint32_t TMA_A_BYTES = kAIsFP4 ? (SMEM_A / 2) : SMEM_A;
    static constexpr uint32_t SMEM_TMA_BYTES = TMA_A_BYTES + TMA_B_BYTES + TMA_SFA_BYTES + TMA_SFB_BYTES;
    // ldmatrix K stride in bytes: FP4 packed = MMA_K/2, FP8 = MMA_K. Both = 32 bytes.
    static constexpr uint32_t kLdmK = kIsFP4 ? (MMA_K / 2) : MMA_K;
    // tma::copy swizzle for split computation: FP4 packed with B64 has 64 byte rows = full BLOCK_K,
    // so one TMA copy covers the entire tile. Use 0 to get single-copy path.
    static constexpr uint32_t kTMACopySwizzleA = kIsFP4 ? 0u : kSwizzleAMode;
    static constexpr uint32_t kTMACopySwizzleB = kIsFP4 ? 0u : kSwizzleBMode;

    shape_m = SHAPE_M != 0 ? SHAPE_M : shape_m;
    shape_n = SHAPE_N != 0 ? SHAPE_N : shape_n;
    shape_k = SHAPE_K != 0 ? SHAPE_K : shape_k;

    const uint32_t warp_idx = __shfl_sync(0xffffffff, threadIdx.x / 32, 0);
    const uint32_t lane_idx = threadIdx.x % 32;

    // SMEM layout: pipeline data first (1024-aligned for B128 swizzle),
    // tensor map descriptors at the end (K-grouped only)
    extern __shared__ __align__(1024) uint8_t smem_buffer[];

    auto smem_d_base = reinterpret_cast<cd_dtype_t*>(smem_buffer);

    constexpr uint32_t PIPE_BASE = SMEM_D;
    auto smem_a = utils::PatternVisitor([&](const uint32_t& s) {
        return reinterpret_cast<char*>(smem_buffer + PIPE_BASE + s * SMEM_A);
    });
    auto smem_b = utils::PatternVisitor([&](const uint32_t& s) {
        return reinterpret_cast<char*>(smem_buffer + PIPE_BASE + kNumStages * SMEM_A + s * SMEM_B);
    });
    constexpr uint32_t SF_BASE = PIPE_BASE + kNumStages * (SMEM_A + SMEM_B);
    auto smem_sfa = utils::PatternVisitor([&](const uint32_t& s) {
        return reinterpret_cast<char*>(smem_buffer + SF_BASE + s * SMEM_SFA);
    });
    auto smem_sfb = utils::PatternVisitor([&](const uint32_t& s) {
        return reinterpret_cast<char*>(smem_buffer + SF_BASE + kNumStages * SMEM_SFA + s * SMEM_SFB);
    });
    constexpr uint32_t BAR_BASE = SF_BASE + kNumStages * (SMEM_SFA + SMEM_SFB);
    auto full_barriers = utils::PatternVisitor([&](const uint32_t& s) {
        return reinterpret_cast<Barrier*>(smem_buffer + BAR_BASE + s * sizeof(Barrier));
    });
    auto empty_barriers = utils::PatternVisitor([&](const uint32_t& s) {
        return reinterpret_cast<Barrier*>(smem_buffer + BAR_BASE + (kNumStages + s) * sizeof(Barrier));
    });

    // Tensor map descriptors at the end of SMEM (K-grouped only)
    constexpr uint32_t TM_BASE = BAR_BASE + 2 * kNumStages * sizeof(Barrier);
    auto smem_tm_a = reinterpret_cast<cute::TmaDescriptor*>(smem_buffer + TM_BASE);
    auto smem_tm_b = smem_tm_a + 1;
    auto gmem_tm_a = tensor_map_buffer + blockIdx.x * 2;
    auto gmem_tm_b = gmem_tm_a + 1;

    // Prefetch TMA descriptors
    if (warp_idx == 0 and cute::elect_one_sync()) {
        cute::prefetch_tma_descriptor(&tensor_map_a_base);
        cute::prefetch_tma_descriptor(&tensor_map_b_base);
        cute::prefetch_tma_descriptor(&tensor_map_sfa);
        cute::prefetch_tma_descriptor(&tensor_map_sfb);
        cute::prefetch_tma_descriptor(&tensor_map_cd);
    }
    __syncwarp();

    // Barrier init (done by warp 1 before producer/consumer split)
    if (warp_idx == 1 and cute::elect_one_sync()) {
        if constexpr (kGemmType == GemmType::KGroupedContiguous) {
            *smem_tm_a = tensor_map_a_base;
            *smem_tm_b = tensor_map_b_base;
        }
        #pragma unroll
        for (uint32_t i = 0; i < kNumStages; ++i) {
            full_barriers[i]->init(1);
            empty_barriers[i]->init(kNumMathWarps);
        }
        cutlass::arch::fence_barrier_init();
    }
    __syncthreads();

    // PDL belongs to the standalone launch boundary.  The ordered
    // wrapper performs a cooperative grid rendezvous before this phase.

    // Persistent scheduler
    uint32_t m_block_idx, n_block_idx;
    static constexpr uint32_t kSFKAlignment = (kGranKA > kGranKB ? kGranKA : kGranKB) * 4;
    auto scheduler = sched::CakeChunk8Scheduler<kGemmType, BLOCK_M, BLOCK_N, kNumGroups, 1, false, kNumSMs,
        false, 128u, kSFKAlignment, sched::get_num_1d_blocks_per_group<kGemmType, BLOCK_M, BLOCK_N, kNumSMs, false>(), kSplitKFactor>(
        shape_m, shape_n, shape_k, grouped_layout, cake_assigned_m_block, cake_assigned_n_block);
    const auto get_pipeline = [=](const uint32_t& iter_idx) -> cute::tuple<uint32_t, uint32_t> {
        return {iter_idx % kNumStages, (iter_idx / kNumStages) & 1};
    };

    // PRODUCER WARP GROUP (TMA warps, 40 regs)
    if (warp_idx >= kNumMathWarps) {
        cutlass::arch::warpgroup_reg_dealloc<kTMARegisters>();

        const bool is_tma_leader = (warp_idx == kNumMathWarps and lane_idx == 0);
        uint32_t tma_iter_idx = 0;

        if (is_tma_leader) {
            uint32_t last_group_idx = kNumGroups;
            while (scheduler.get_next_block(m_block_idx, n_block_idx)) {
                // Skip empty/padding tiles in the contiguous grouped layout: m_indices
                // is -1 for blocks with no routed tokens. The worst-case M_sum reserves a
                // block per local expert, but at decode only a few are routed; processing
                // the rest wastes a full-width GEMM tile (the dominant EP-decode cost).
                // Producer and consumer apply the identical check, so no barrier ops are
                // issued for skipped blocks and the pipeline stays in sync.
                if constexpr (kGemmType == GemmType::MGroupedContiguous) {
                    if (__ldg(grouped_layout + m_block_idx * BLOCK_M) < 0)
                        continue;
                }
                if constexpr (kGemmType == GemmType::KGroupedContiguous) {
                    if (last_group_idx != scheduler.current_group_idx) {
                        last_group_idx = scheduler.current_group_idx;

                        const auto a_base = reinterpret_cast<const char*>(gmem_a_ptr);
                        const auto b_base = reinterpret_cast<const char*>(gmem_b_ptr);

                        if constexpr (kKGroupedConstantStride) {
                            const uint64_t a_k_byte_offset = kIsFP4
                                ? (static_cast<uint64_t>(scheduler.current_k_cumsum) / 2)
                                : (static_cast<uint64_t>(scheduler.current_k_cumsum));
                            const uint64_t b_k_byte_offset = (kIsFP4 || kBIsFP4)
                                ? (static_cast<uint64_t>(scheduler.current_k_cumsum) / 2)
                                : (static_cast<uint64_t>(scheduler.current_k_cumsum));
                            ptx::tensor_map_replace_global_addr_in_smem(smem_tm_a, a_base + a_k_byte_offset);
                            ptx::tensor_map_replace_global_addr_in_smem(smem_tm_b, b_base + b_k_byte_offset);
                            ptx::tensor_map_replace_global_dim_in_smem(smem_tm_a, scheduler.current_shape_k);
                            ptx::tensor_map_replace_global_dim_in_smem(smem_tm_b, scheduler.current_shape_k);
                        } else {
                            const uint64_t a_offset = kIsFP4
                                ? (static_cast<uint64_t>(scheduler.current_k_cumsum) * shape_m / 2)
                                : (static_cast<uint64_t>(scheduler.current_k_cumsum) * shape_m);
                            const uint64_t b_offset = (kIsFP4 || kBIsFP4)
                                ? (static_cast<uint64_t>(scheduler.current_k_cumsum) * shape_n / 2)
                                : (static_cast<uint64_t>(scheduler.current_k_cumsum) * shape_n);
                            ptx::tensor_map_replace_global_addr_in_smem(smem_tm_a, a_base + a_offset);
                            ptx::tensor_map_replace_global_addr_in_smem(smem_tm_b, b_base + b_offset);
                            const uint64_t a_new_stride = kIsFP4
                                ? static_cast<uint64_t>(scheduler.current_shape_k / 2)
                                : static_cast<uint64_t>(scheduler.current_shape_k);
                            const uint64_t b_new_stride = (kIsFP4 || kBIsFP4)
                                ? static_cast<uint64_t>(scheduler.current_shape_k / 2)
                                : static_cast<uint64_t>(scheduler.current_shape_k);
                            ptx::tensor_map_replace_global_inner_dim_stride_in_smem(
                                smem_tm_a, scheduler.current_shape_k, a_new_stride);
                            ptx::tensor_map_replace_global_inner_dim_stride_in_smem(
                                smem_tm_b, scheduler.current_shape_k, b_new_stride);
                        }

                        *gmem_tm_a = *smem_tm_a;
                        *gmem_tm_b = *smem_tm_b;
                        ptx::tensor_map_release_gpu();
                        ptx::tensor_map_acquire_gpu(gmem_tm_a);
                        ptx::tensor_map_acquire_gpu(gmem_tm_b);
                    }
                }

                const uint32_t current_shape_k = (kGemmType == GemmType::KGroupedContiguous ? scheduler.current_shape_k : shape_k);
                const uint32_t num_k_blocks = math::ceil_div(current_shape_k, BLOCK_K);
                uint32_t kb_start = 0, kb_end = num_k_blocks;
                if constexpr (kSplitKFactor > 1) {
                    const uint32_t k_per_split = num_k_blocks / kSplitKFactor;
                    kb_start = scheduler.split_k_idx * k_per_split;
                    kb_end = (scheduler.split_k_idx == kSplitKFactor - 1) ? num_k_blocks : kb_start + k_per_split;
                }
                constexpr bool kAGroupOffset = (kGemmType == GemmType::MGroupedMasked);
                const uint32_t m_idx = scheduler.template get_global_idx<kAGroupOffset>(shape_m, BLOCK_M, m_block_idx);
                constexpr bool kBGroupOffset = not (kGemmType == GemmType::Normal or kGemmType == GemmType::KGroupedContiguous);
                const uint32_t n_idx = scheduler.template get_global_idx<kBGroupOffset>(shape_n, BLOCK_N, n_block_idx, m_block_idx);
                const auto tma_a_desc = (kGemmType == GemmType::KGroupedContiguous ? gmem_tm_a : &tensor_map_a_base);
                const auto tma_b_desc = (kGemmType == GemmType::KGroupedContiguous ? gmem_tm_b : &tensor_map_b_base);

                constexpr bool kIsBatchedMM = (kGemmType == GemmType::Batched);
                const uint32_t batch_idx = kIsBatchedMM ? scheduler.current_group_idx : 0;

                for (uint32_t kb = kb_start; kb < kb_end; ++kb) {
                    CUTE_TIE_DECL(get_pipeline(tma_iter_idx++), s, p);
                    empty_barriers[s]->wait(p ^ 1);

                    const uint32_t k_idx = kb * BLOCK_K;
                    uint32_t sfa_k, sfb_k;
                    if constexpr (kGemmType == GemmType::KGroupedContiguous) {
                        sfa_k = scheduler.current_sf_k_cumsum + kb / kNumSFAStagesPerLoad;
                        sfb_k = scheduler.current_sf_k_cumsum + kb / kNumSFBStagesPerLoad;
                    } else {
                        const uint32_t shape_sfa_k = math::ceil_div(shape_k, BLOCK_K * kNumSFAStagesPerLoad);
                        const uint32_t shape_sfb_k = math::ceil_div(shape_k, BLOCK_K * kNumSFBStagesPerLoad);
                        constexpr bool kSFAGroupOffset = not is_m_grouped_contiguous(kGemmType);
                        sfa_k = scheduler.template get_global_idx<kSFAGroupOffset, sched::IndexType::SF_K>(
                            shape_sfa_k, 1, kb / kNumSFAStagesPerLoad, m_block_idx);
                        constexpr bool kSFBGroupOffset = not (kGemmType == GemmType::Normal);
                        sfb_k = scheduler.template get_global_idx<kSFBGroupOffset, sched::IndexType::SF_K>(
                            shape_sfb_k, 1, kb / kNumSFBStagesPerLoad, m_block_idx);
                    }
                    tma::copy<BLOCK_M, BLOCK_K, 0>(&tensor_map_sfa, full_barriers[s], smem_sfa[s], m_block_idx * BLOCK_M, sfa_k, 1);
                    tma::copy<BLOCK_N, BLOCK_K, 0>(&tensor_map_sfb, full_barriers[s], smem_sfb[s], n_block_idx * BLOCK_N, sfb_k, 1);
                    tma::copy<BLOCK_K, BLOCK_M, kTMACopySwizzleA, char, kIsBatchedMM>(tma_a_desc, full_barriers[s], smem_a[s], k_idx, m_idx, 1, batch_idx);
                    if constexpr (kBKMajor) {
                        tma::copy<BLOCK_K, BLOCK_N, kTMACopySwizzleB, char, kIsBatchedMM>(tma_b_desc, full_barriers[s], smem_b[s], k_idx, n_idx, 1, batch_idx);
                    } else {
                        tma::copy<BLOCK_N, BLOCK_K, kSwizzleBMode, char, kIsBatchedMM>(
                            tma_b_desc, full_barriers[s], smem_b[s],
                            n_idx, k_idx, 1, batch_idx);
                    }
                    full_barriers[s]->arrive_and_expect_tx(SMEM_TMA_BYTES);
                }
            }
        }
    }
    // CONSUMER WARP GROUPS (math warps, 232 regs)
    else {
        cutlass::arch::warpgroup_reg_alloc<kMMARegisters>();

        const uint32_t math_warp_idx = warp_idx;
        const uint32_t group_id = lane_idx / 4;
        const uint32_t thread_id = lane_idx % 4;
        const uint32_t warp_m = math_warp_idx / kNWarps;
        const uint32_t warp_n = math_warp_idx % kNWarps;
        const uint32_t m_tile_base = warp_m * kMTilesPerWarp;
        const uint32_t n_tile_base = warp_n * kNTilesPerWarp;

        float accum[kAccumPerWarp];
        uint32_t iter_idx = 0;

        while (scheduler.get_next_block(m_block_idx, n_block_idx)) {
            // Skip empty/padding tiles (m_indices == -1); see the matching check in the
            // producer loop. Both warp groups skip identically, so barriers stay in sync.
            if constexpr (kGemmType == GemmType::MGroupedContiguous) {
                if (__ldg(grouped_layout + m_block_idx * BLOCK_M) < 0)
                    continue;
            }
            const uint32_t current_shape_k = (kGemmType == GemmType::KGroupedContiguous ? scheduler.current_shape_k : shape_k);
            const uint32_t num_k_blocks_total = math::ceil_div(current_shape_k, BLOCK_K);
            uint32_t num_k_blocks_start = 0, num_k_blocks = num_k_blocks_total;
            if constexpr (kSplitKFactor > 1) {
                const uint32_t k_per_split = num_k_blocks_total / kSplitKFactor;
                num_k_blocks_start = scheduler.split_k_idx * k_per_split;
                num_k_blocks = ((scheduler.split_k_idx == kSplitKFactor - 1) ? num_k_blocks_total : num_k_blocks_start + k_per_split) - num_k_blocks_start;
            }

            #pragma unroll
            for (uint32_t i = 0; i < kAccumPerWarp; ++i) accum[i] = 0.f;

            // kAIsFP4 uses the regular (non-perNTileX4) path to keep the fp4-A load localized.
            static constexpr bool kUsePerNTileX4 = kBKMajor and not kBIsFP4 and not kAIsFP4 and (kKSteps >= 2);
            using sf_t = cute::conditional_t<kIsFP4, uint16_t, uint8_t>;

            // SF-major loop: when gran_k >= BLOCK_K, one packed int32 SF covers
            // kNumSFAStagesPerLoad K-blocks. Load SF into registers once per SF tile,
            // extract with compile-time byte index via cute::for_each.
            static constexpr bool kUseSFMajorLoop = (kGranKA >= BLOCK_K) and (kGranKB >= BLOCK_K);
            static_assert(!kUseSFMajorLoop || kNumSFAStagesPerLoad == kNumSFBStagesPerLoad,
                "SF-major loop requires matching A/B SF tile sizes");
            static constexpr uint32_t kSFTileKBlocks = kUseSFMajorLoop ? kNumSFAStagesPerLoad : 1;

            if constexpr (kUseSFMajorLoop) {
            // SF-MAJOR PATH: gran_k >= BLOCK_K
            // Load SF packed int32 into registers once per kSFTileKBlocks K-blocks,
            // extract bytes with compile-time index via cute::for_each.
            // SwizzleContext hoisted outside K-block loop (loop-invariant).
            uint32_t sf_packed_a[kMTilesPerWarp];
            uint32_t sf_packed_b[kNTilesPerWarp];
            const uint32_t num_full_sf_tiles = num_k_blocks / kSFTileKBlocks;
            const uint32_t kb_tail_start = num_full_sf_tiles * kSFTileKBlocks;

            sm120::SwizzleContext<kSwizzleAMode> a_ctx[kMTilesPerWarp];
            #pragma unroll
            for (uint32_t mt = 0; mt < kMTilesPerWarp; ++mt) {
                int a_row = (lane_idx & 7) + ((lane_idx >> 3) & 1) * 8 + (m_tile_base + mt) * 16;
                a_ctx[mt].init(a_row, kSMEMKBytes);
            }
            sm120::SwizzleContext<kSwizzleBMode> b_ctx[kNTilesPerWarp];
            #pragma unroll
            for (uint32_t nt = 0; nt < kNTilesPerWarp; ++nt) {
                int b_row = (lane_idx & 7) + (n_tile_base + nt) * 8;
                b_ctx[nt].init(b_row, kSMEMKBytes);
            }

            // Main body: compile-time unrolled K-blocks within each SF tile
            for (uint32_t sf_tile = 0; sf_tile < num_full_sf_tiles; ++sf_tile) {
            cute::for_each(cute::make_int_sequence<kSFTileKBlocks>{}, [&](auto kb_inner_ic) {
                constexpr uint32_t kb_inner = kb_inner_ic;
                CUTE_TIE_DECL(get_pipeline(iter_idx++), stage, phase);
                full_barriers[stage]->wait(phase);

                const uint32_t kb = sf_tile * kSFTileKBlocks + kb_inner;

                if constexpr (kUsePerNTileX4) {
                    uint32_t b_nt[kNTilesPerWarp][4];
                    uint32_t a_frag[2][kMTilesPerWarp][4];
                    sf_t sfb_hoisted[kNTilesPerWarp];
                    sf_t sfa_hoisted[kMTilesPerWarp];

                    if (kb_inner == 0) {
                        #pragma unroll
                        for (uint32_t nt = 0; nt < kNTilesPerWarp; ++nt)
                            sf_packed_b[nt] = sm120::load_sf(smem_sfb[stage], (n_tile_base + nt) * MMA_N + group_id);
                        #pragma unroll
                        for (uint32_t mt = 0; mt < kMTilesPerWarp; ++mt)
                            sf_packed_a[mt] = sm120::load_sf(smem_sfa[stage],
                                (m_tile_base + mt) * MMA_M + group_id + (thread_id & 1) * 8);
                    }

                    // Compile-time byte index: maps kb_inner to the correct byte within packed SF.
                    // For split-K: k_per_split must be aligned to kSFTileKBlocks so
                    // each partition starts at an SF tile boundary.
                    constexpr uint32_t sf_byte_a = (kb_inner * BLOCK_K / kGranKA) % 4;
                    constexpr uint32_t sf_byte_b = (kb_inner * BLOCK_K / kGranKB) % 4;

                    #pragma unroll
                    for (uint32_t nt = 0; nt < kNTilesPerWarp; ++nt) {
                        if constexpr (kIsFP4) {
                            uint8_t b = sm120_mma::extract_sf_byte(sf_packed_b[nt], sf_byte_b);
                            sfb_hoisted[nt] = static_cast<uint16_t>(b) | (static_cast<uint16_t>(b) << 8);
                        } else {
                            sfb_hoisted[nt] = sm120_mma::extract_sf_byte(sf_packed_b[nt], sf_byte_b);
                        }
                    }
                    #pragma unroll
                    for (uint32_t mt = 0; mt < kMTilesPerWarp; ++mt) {
                        if constexpr (kIsFP4) {
                            uint8_t b = sm120_mma::extract_sf_byte(sf_packed_a[mt], sf_byte_a);
                            sfa_hoisted[mt] = static_cast<uint16_t>(b) | (static_cast<uint16_t>(b) << 8);
                        } else {
                            sfa_hoisted[mt] = sm120_mma::extract_sf_byte(sf_packed_a[mt], sf_byte_a);
                        }
                    }

                    static constexpr uint32_t kKStepPairs = kKSteps / 2;
                    #pragma unroll
                    for (uint32_t kp = 0; kp < kKStepPairs; ++kp) {
                        const uint32_t ks_base = kp * 2;

                        #pragma unroll
                        for (uint32_t mt = 0; mt < kMTilesPerWarp; ++mt)
                            sm120::load_a_fragment(a_frag[0][mt], smem_a[stage], a_ctx[mt], lane_idx, ks_base, kLdmK);

                        #pragma unroll
                        for (uint32_t nt = 0; nt < kNTilesPerWarp; ++nt)
                            sm120::load_b_per_ntile_x4(b_nt[nt], smem_b[stage], b_ctx[nt], lane_idx, kp, kLdmK * 2);

                        #pragma unroll
                        for (uint32_t mt = 0; mt < kMTilesPerWarp; ++mt)
                            sm120::load_a_fragment(a_frag[1][mt], smem_a[stage], a_ctx[mt], lane_idx, ks_base + 1, kLdmK);

                        #pragma unroll
                        for (uint32_t mt = 0; mt < kMTilesPerWarp; ++mt) {
                            #pragma unroll
                            for (uint32_t nt = 0; nt < kNTilesPerWarp; ++nt) {
                                float (&d)[4] = *reinterpret_cast<float(*)[4]>(&accum[(mt * kNTilesPerWarp + nt) * MMA_ACCUM]);
                                if constexpr (kIsFP4)
                                    sm120_mma::fp4_mma_block_scaled(d, a_frag[0][mt], b_nt[nt][0], b_nt[nt][1], sfa_hoisted[mt], sfb_hoisted[nt]);
                                else
                                    sm120_mma::fp8_mma_block_scaled(d, a_frag[0][mt], b_nt[nt][0], b_nt[nt][1], sfa_hoisted[mt], sfb_hoisted[nt]);
                            }
                        }

                        #pragma unroll
                        for (uint32_t mt = 0; mt < kMTilesPerWarp; ++mt) {
                            #pragma unroll
                            for (uint32_t nt = 0; nt < kNTilesPerWarp; ++nt) {
                                float (&d)[4] = *reinterpret_cast<float(*)[4]>(&accum[(mt * kNTilesPerWarp + nt) * MMA_ACCUM]);
                                if constexpr (kIsFP4)
                                    sm120_mma::fp4_mma_block_scaled(d, a_frag[1][mt], b_nt[nt][2], b_nt[nt][3], sfa_hoisted[mt], sfb_hoisted[nt]);
                                else
                                    sm120_mma::fp8_mma_block_scaled(d, a_frag[1][mt], b_nt[nt][2], b_nt[nt][3], sfa_hoisted[mt], sfb_hoisted[nt]);
                            }
                        }
                    }
                } else {
                    // Fallback path for non-SF-major (MN-major B, mixed FP8×FP4) — unchanged
                    const uint32_t sf_byte_a_base = ((sf_tile * kSFTileKBlocks + kb_inner) * BLOCK_K / kGranKA) % 4;
                    const uint32_t sf_byte_b_base = ((sf_tile * kSFTileKBlocks + kb_inner) * BLOCK_K / kGranKB) % 4;
                    sm120::SwizzleContext<kSwizzleBMode> b_ctx[kBKMajor ? kNTilesPerWarp : 1];
                    if constexpr (kBKMajor) {
                        #pragma unroll
                        for (uint32_t nt = 0; nt < kNTilesPerWarp; ++nt) {
                            int b_row = (lane_idx & 7) + (n_tile_base + nt) * 8;
                            b_ctx[nt].init(b_row, kSMEMKBytes);
                        }
                    }
                    uint32_t a_frag[2][kMTilesPerWarp][4];
                    uint32_t b_tile[2][kNTilesPerWarp][2];
                    sf_t sfa_bytes[2][kMTilesPerWarp];
                    sf_t sfb_bytes[2][kNTilesPerWarp];
                    sf_t sfa_hoisted[kMTilesPerWarp];
                    sf_t sfb_hoisted[kNTilesPerWarp];

                    if constexpr (kGranKB >= BLOCK_K) {
                        #pragma unroll
                        for (uint32_t nt = 0; nt < kNTilesPerWarp; ++nt) {
                            auto packed = sm120::load_sf(smem_sfb[stage], (n_tile_base + nt) * MMA_N + group_id);
                            if constexpr (kIsFP4) {
                                uint8_t b = sm120_mma::extract_sf_byte(packed, sf_byte_b_base);
                                sfb_hoisted[nt] = static_cast<uint16_t>(b) | (static_cast<uint16_t>(b) << 8);
                            } else {
                                sfb_hoisted[nt] = sm120_mma::extract_sf_byte(packed, sf_byte_b_base);
                            }
                        }
                    }
                    if constexpr (kGranKA >= BLOCK_K) {
                        #pragma unroll
                        for (uint32_t mt = 0; mt < kMTilesPerWarp; ++mt) {
                            auto packed = sm120::load_sf(smem_sfa[stage],
                                (m_tile_base + mt) * MMA_M + group_id + (thread_id & 1) * 8);
                            if constexpr (kIsFP4) {
                                uint8_t b = sm120_mma::extract_sf_byte(packed, sf_byte_a_base);
                                sfa_hoisted[mt] = static_cast<uint16_t>(b) | (static_cast<uint16_t>(b) << 8);
                            } else {
                                sfa_hoisted[mt] = sm120_mma::extract_sf_byte(packed, sf_byte_a_base);
                            }
                        }
                    }

                    auto load_kstep = [&](int buf, uint32_t ks) {
                        if constexpr (kBKMajor) {
                            if constexpr (kBIsFP4) {
                                #pragma unroll
                                for (uint32_t nt = 0; nt < kNTilesPerWarp; ++nt) {
                                    sm120::load_b_fragment_b4x16_p64(b_tile[buf][nt], smem_b[stage], b_ctx[nt], lane_idx, ks, kLdmK);
                                    b_tile[buf][nt][0] <<= 2;
                                    b_tile[buf][nt][1] <<= 2;
                                }
                            } else {
                                #pragma unroll
                                for (uint32_t nt = 0; nt < kNTilesPerWarp; ++nt)
                                    sm120::load_b_fragment_x2(b_tile[buf][nt], smem_b[stage], b_ctx[nt], lane_idx, ks, kLdmK);
                            }
                        } else {
                            static constexpr uint32_t kBSwizzleB = kSwizzleBMode > 0 ? (__builtin_ctz(kSwizzleBMode) - 4) : 0;
                            static constexpr uint32_t kBSwizzleMask = kSwizzleBMode > 0 ? ((1u << kBSwizzleB) - 1) : 0;
                            static constexpr uint32_t kBSwizzleRowShift = kSwizzleBMode > 0 ? (7 - __builtin_ctz(BLOCK_N)) : 0;
                            #pragma unroll
                            for (uint32_t nt = 0; nt < kNTilesPerWarp; ++nt) {
                                const uint32_t n_col = (n_tile_base + nt) * MMA_N + group_id;
                                uint8_t v[8];
                                #pragma unroll
                                for (uint32_t i = 0; i < 4; ++i) {
                                    const uint32_t k = ks * MMA_K + thread_id * 4 + i;
                                    const uint32_t xor_bits = kSwizzleBMode > 0
                                        ? (((k >> kBSwizzleRowShift) & kBSwizzleMask) << 4) : 0;
                                    v[i] = static_cast<uint8_t>(smem_b[stage][k * BLOCK_N + (n_col ^ xor_bits)]);
                                }
                                #pragma unroll
                                for (uint32_t i = 0; i < 4; ++i) {
                                    const uint32_t k = ks * MMA_K + 16 + thread_id * 4 + i;
                                    const uint32_t xor_bits = kSwizzleBMode > 0
                                        ? (((k >> kBSwizzleRowShift) & kBSwizzleMask) << 4) : 0;
                                    v[4+i] = static_cast<uint8_t>(smem_b[stage][k * BLOCK_N + (n_col ^ xor_bits)]);
                                }
                                b_tile[buf][nt][0] = v[0] | (uint32_t(v[1]) << 8) | (uint32_t(v[2]) << 16) | (uint32_t(v[3]) << 24);
                                b_tile[buf][nt][1] = v[4] | (uint32_t(v[5]) << 8) | (uint32_t(v[6]) << 16) | (uint32_t(v[7]) << 24);
                            }
                        }
                        #pragma unroll
                        for (uint32_t mt = 0; mt < kMTilesPerWarp; ++mt)
                            if constexpr (kAIsFP4) {
                                sm120::load_a_fragment_b4x16(a_frag[buf][mt], smem_a[stage], a_ctx[mt], lane_idx, ks, kLdmK);
                                a_frag[buf][mt][0] <<= 2; a_frag[buf][mt][1] <<= 2;
                                a_frag[buf][mt][2] <<= 2; a_frag[buf][mt][3] <<= 2;
                            } else {
                                sm120::load_a_fragment(a_frag[buf][mt], smem_a[stage], a_ctx[mt], lane_idx, ks, kLdmK);
                            }

                        if constexpr (kGranKA < BLOCK_K or kGranKB < BLOCK_K) {
                            const uint32_t sf_step = (kb * kKSteps + ks);
                            if constexpr (kIsFP4) {
                                const uint32_t sf_byte_a = (sf_step * MMA_K / kGranKA) % 4;
                                const uint32_t sf_byte_b = (sf_step * MMA_K / kGranKB) % 4;
                                #pragma unroll
                                for (uint32_t nt = 0; nt < kNTilesPerWarp; ++nt) {
                                    auto packed = sm120::load_sf(smem_sfb[stage], (n_tile_base + nt) * MMA_N + group_id);
                                    if constexpr (kGranKB <= 32)
                                        sfb_bytes[buf][nt] = sm120_mma::extract_sf_pair(packed, sf_byte_b);
                                    else {
                                        uint8_t b = sm120_mma::extract_sf_byte(packed, sf_byte_b);
                                        sfb_bytes[buf][nt] = static_cast<uint16_t>(b) | (static_cast<uint16_t>(b) << 8);
                                    }
                                }
                                #pragma unroll
                                for (uint32_t mt = 0; mt < kMTilesPerWarp; ++mt) {
                                    auto packed = sm120::load_sf(smem_sfa[stage],
                                        (m_tile_base + mt) * MMA_M + group_id + (thread_id & 1) * 8);
                                    if constexpr (kGranKA <= 32)
                                        sfa_bytes[buf][mt] = sm120_mma::extract_sf_pair(packed, sf_byte_a);
                                    else {
                                        uint8_t b = sm120_mma::extract_sf_byte(packed, sf_byte_a);
                                        sfa_bytes[buf][mt] = static_cast<uint16_t>(b) | (static_cast<uint16_t>(b) << 8);
                                    }
                                }
                            } else {
                                const uint32_t sf_byte_a = (sf_step * MMA_K / kGranKA) % 4;
                                const uint32_t sf_byte_b = (sf_step * MMA_K / kGranKB) % 4;
                                #pragma unroll
                                for (uint32_t nt = 0; nt < kNTilesPerWarp; ++nt)
                                    sfb_bytes[buf][nt] = sm120_mma::extract_sf_byte(
                                        sm120::load_sf(smem_sfb[stage], (n_tile_base + nt) * MMA_N + group_id), sf_byte_b);
                                #pragma unroll
                                for (uint32_t mt = 0; mt < kMTilesPerWarp; ++mt)
                                    sfa_bytes[buf][mt] = sm120_mma::extract_sf_byte(
                                        sm120::load_sf(smem_sfa[stage],
                                            (m_tile_base + mt) * MMA_M + group_id + (thread_id & 1) * 8), sf_byte_a);
                            }
                        }
                    };

                    auto compute_kstep = [&](int buf) {
                        #pragma unroll
                        for (uint32_t mt = 0; mt < kMTilesPerWarp; ++mt) {
                            const sf_t sfa = (kGranKA >= BLOCK_K) ? sfa_hoisted[mt] : sfa_bytes[buf][mt];
                            #pragma unroll
                            for (uint32_t nt = 0; nt < kNTilesPerWarp; ++nt) {
                                float (&d)[4] = *reinterpret_cast<float(*)[4]>(&accum[(mt * kNTilesPerWarp + nt) * MMA_ACCUM]);
                                const sf_t sfb = (kGranKB >= BLOCK_K) ? sfb_hoisted[nt] : sfb_bytes[buf][nt];
                                if constexpr (kAIsFP4)
                                    sm120_mma::fp4_fp8_mixed_mma_block_scaled(d, a_frag[buf][mt], b_tile[buf][nt], sfa, sfb);
                                else if constexpr (kBIsFP4)
                                    sm120_mma::fp8_fp4_mixed_mma_block_scaled(d, a_frag[buf][mt], b_tile[buf][nt], sfa, sfb);
                                else if constexpr (kIsFP4)
                                    sm120_mma::fp4_mma_block_scaled(d, a_frag[buf][mt], b_tile[buf][nt], sfa, sfb);
                                else
                                    sm120_mma::fp8_mma_block_scaled(d, a_frag[buf][mt], b_tile[buf][nt], sfa, sfb);
                            }
                        }
                    };

                    load_kstep(0, 0);
                    #pragma unroll
                    for (uint32_t ks = 0; ks < kKSteps; ++ks) {
                        int cur = ks & 1;
                        int nxt = (ks + 1) & 1;
                        if (ks < kKSteps - 1)
                            load_kstep(nxt, ks + 1);
                        compute_kstep(cur);
                    }
                }

                // Release stage
                if (lane_idx == 0)
                    empty_barriers[stage]->arrive();
            }); // kb_inner (cute::for_each)
            } // sf_tile (SF-major main body)

            // SF-major tail: remaining K-blocks (0 to kSFTileKBlocks-1).
            // Since kUseSFMajorLoop implies kGranK >= BLOCK_K, SF hoisting is always valid.
            for (uint32_t kb = kb_tail_start; kb < num_k_blocks; ++kb) {
                CUTE_TIE_DECL(get_pipeline(iter_idx++), stage, phase);
                full_barriers[stage]->wait(phase);

                const uint32_t sf_byte_a_base = (kb * BLOCK_K / kGranKA) % 4;
                const uint32_t sf_byte_b_base = (kb * BLOCK_K / kGranKB) % 4;

                if constexpr (kUsePerNTileX4) {
                    uint32_t b_nt[kNTilesPerWarp][4];
                    uint32_t a_frag[2][kMTilesPerWarp][4];
                    sf_t sfb_hoisted[kNTilesPerWarp];
                    sf_t sfa_hoisted[kMTilesPerWarp];
                    sf_t sfb_step[kNTilesPerWarp];
                    sf_t sfa_step[2][kMTilesPerWarp];

                    if constexpr (kGranKB >= BLOCK_K) {
                        #pragma unroll
                        for (uint32_t nt = 0; nt < kNTilesPerWarp; ++nt) {
                            auto packed = sm120::load_sf(smem_sfb[stage], (n_tile_base + nt) * MMA_N + group_id);
                            if constexpr (kIsFP4) {
                                uint8_t b = sm120_mma::extract_sf_byte(packed, sf_byte_b_base);
                                sfb_hoisted[nt] = static_cast<uint16_t>(b) | (static_cast<uint16_t>(b) << 8);
                            } else {
                                sfb_hoisted[nt] = sm120_mma::extract_sf_byte(packed, sf_byte_b_base);
                            }
                        }
                    }
                    if constexpr (kGranKA >= BLOCK_K) {
                        #pragma unroll
                        for (uint32_t mt = 0; mt < kMTilesPerWarp; ++mt) {
                            auto packed = sm120::load_sf(smem_sfa[stage],
                                (m_tile_base + mt) * MMA_M + group_id + (thread_id & 1) * 8);
                            if constexpr (kIsFP4) {
                                uint8_t b = sm120_mma::extract_sf_byte(packed, sf_byte_a_base);
                                sfa_hoisted[mt] = static_cast<uint16_t>(b) | (static_cast<uint16_t>(b) << 8);
                            } else {
                                sfa_hoisted[mt] = sm120_mma::extract_sf_byte(packed, sf_byte_a_base);
                            }
                        }
                    }

                    static constexpr uint32_t kKStepPairs_tail = kKSteps / 2;
                    #pragma unroll
                    for (uint32_t kp = 0; kp < kKStepPairs_tail; ++kp) {
                        const uint32_t ks_base = kp * 2;
                        #pragma unroll
                        for (uint32_t mt = 0; mt < kMTilesPerWarp; ++mt)
                            sm120::load_a_fragment(a_frag[0][mt], smem_a[stage], a_ctx[mt], lane_idx, ks_base, kLdmK);
                        #pragma unroll
                        for (uint32_t nt = 0; nt < kNTilesPerWarp; ++nt)
                            sm120::load_b_per_ntile_x4(b_nt[nt], smem_b[stage], b_ctx[nt], lane_idx, kp, kLdmK * 2);

                        auto load_sf_for_step_tail = [&](uint32_t ks, int sf_buf) {
                            if constexpr (kGranKA < BLOCK_K or kGranKB < BLOCK_K) {
                                const uint32_t sf_step = kb * kKSteps + ks;
                                if constexpr (kIsFP4) {
                                    const uint32_t sf_byte_b = (sf_step * MMA_K / kGranKB) % 4;
                                    const uint32_t sf_byte_a = (sf_step * MMA_K / kGranKA) % 4;
                                    if constexpr (kGranKB < BLOCK_K) {
                                        #pragma unroll
                                        for (uint32_t nt = 0; nt < kNTilesPerWarp; ++nt) {
                                            auto packed = sm120::load_sf(smem_sfb[stage], (n_tile_base + nt) * MMA_N + group_id);
                                            if constexpr (kGranKB <= 32)
                                                sfb_step[nt] = sm120_mma::extract_sf_pair(packed, sf_byte_b);
                                            else {
                                                uint8_t b = sm120_mma::extract_sf_byte(packed, sf_byte_b);
                                                sfb_step[nt] = static_cast<uint16_t>(b) | (static_cast<uint16_t>(b) << 8);
                                            }
                                        }
                                    }
                                    if constexpr (kGranKA < BLOCK_K) {
                                        #pragma unroll
                                        for (uint32_t mt = 0; mt < kMTilesPerWarp; ++mt) {
                                            auto packed = sm120::load_sf(smem_sfa[stage],
                                                (m_tile_base + mt) * MMA_M + group_id + (thread_id & 1) * 8);
                                            if constexpr (kGranKA <= 32)
                                                sfa_step[sf_buf][mt] = sm120_mma::extract_sf_pair(packed, sf_byte_a);
                                            else {
                                                uint8_t b = sm120_mma::extract_sf_byte(packed, sf_byte_a);
                                                sfa_step[sf_buf][mt] = static_cast<uint16_t>(b) | (static_cast<uint16_t>(b) << 8);
                                            }
                                        }
                                    }
                                } else {
                                    const uint32_t sf_byte_b = (sf_step * MMA_K / kGranKB) % 4;
                                    const uint32_t sf_byte_a = (sf_step * MMA_K / kGranKA) % 4;
                                    if constexpr (kGranKB < BLOCK_K) {
                                        #pragma unroll
                                        for (uint32_t nt = 0; nt < kNTilesPerWarp; ++nt)
                                            sfb_step[nt] = sm120_mma::extract_sf_byte(
                                                sm120::load_sf(smem_sfb[stage], (n_tile_base + nt) * MMA_N + group_id), sf_byte_b);
                                    }
                                    if constexpr (kGranKA < BLOCK_K) {
                                        #pragma unroll
                                        for (uint32_t mt = 0; mt < kMTilesPerWarp; ++mt)
                                            sfa_step[sf_buf][mt] = sm120_mma::extract_sf_byte(
                                                sm120::load_sf(smem_sfa[stage],
                                                    (m_tile_base + mt) * MMA_M + group_id + (thread_id & 1) * 8), sf_byte_a);
                                    }
                                }
                            }
                        };

                        load_sf_for_step_tail(ks_base, 0);

                        #pragma unroll
                        for (uint32_t mt = 0; mt < kMTilesPerWarp; ++mt)
                            sm120::load_a_fragment(a_frag[1][mt], smem_a[stage], a_ctx[mt], lane_idx, ks_base + 1, kLdmK);

                        #pragma unroll
                        for (uint32_t mt = 0; mt < kMTilesPerWarp; ++mt) {
                            const sf_t sfa0 = (kGranKA >= BLOCK_K) ? sfa_hoisted[mt] : sfa_step[0][mt];
                            #pragma unroll
                            for (uint32_t nt = 0; nt < kNTilesPerWarp; ++nt) {
                                float (&d)[4] = *reinterpret_cast<float(*)[4]>(&accum[(mt * kNTilesPerWarp + nt) * MMA_ACCUM]);
                                const sf_t sfb = (kGranKB >= BLOCK_K) ? sfb_hoisted[nt] : sfb_step[nt];
                                if constexpr (kIsFP4)
                                    sm120_mma::fp4_mma_block_scaled(d, a_frag[0][mt], b_nt[nt][0], b_nt[nt][1], sfa0, sfb);
                                else
                                    sm120_mma::fp8_mma_block_scaled(d, a_frag[0][mt], b_nt[nt][0], b_nt[nt][1], sfa0, sfb);
                            }
                        }

                        load_sf_for_step_tail(ks_base + 1, 1);

                        #pragma unroll
                        for (uint32_t mt = 0; mt < kMTilesPerWarp; ++mt) {
                            const sf_t sfa1 = (kGranKA >= BLOCK_K) ? sfa_hoisted[mt] : sfa_step[1][mt];
                            #pragma unroll
                            for (uint32_t nt = 0; nt < kNTilesPerWarp; ++nt) {
                                float (&d)[4] = *reinterpret_cast<float(*)[4]>(&accum[(mt * kNTilesPerWarp + nt) * MMA_ACCUM]);
                                const sf_t sfb = (kGranKB >= BLOCK_K) ? sfb_hoisted[nt] : sfb_step[nt];
                                if constexpr (kIsFP4)
                                    sm120_mma::fp4_mma_block_scaled(d, a_frag[1][mt], b_nt[nt][2], b_nt[nt][3], sfa1, sfb);
                                else
                                    sm120_mma::fp8_mma_block_scaled(d, a_frag[1][mt], b_nt[nt][2], b_nt[nt][3], sfa1, sfb);
                            }
                        }
                    }
                }

                if (lane_idx == 0)
                    empty_barriers[stage]->arrive();
            } // SF-major tail kb loop

            } else { // !kUseSFMajorLoop
            // ORIGINAL PATH: gran_k < BLOCK_K (per-K-step SF loading)
            // Flat K-block loop with runtime sf_byte, no SF caching.
            for (uint32_t kb = 0; kb < num_k_blocks; ++kb) {
                CUTE_TIE_DECL(get_pipeline(iter_idx++), stage, phase);

                full_barriers[stage]->wait(phase);

                sm120::SwizzleContext<kSwizzleAMode> a_ctx[kMTilesPerWarp];
                #pragma unroll
                for (uint32_t mt = 0; mt < kMTilesPerWarp; ++mt) {
                    int a_row = (lane_idx & 7) + ((lane_idx >> 3) & 1) * 8 + (m_tile_base + mt) * 16;
                    a_ctx[mt].init(a_row, kSMEMKBytes);
                }

                const uint32_t sf_byte_a_base = (kb * BLOCK_K / kGranKA) % 4;
                const uint32_t sf_byte_b_base = (kb * BLOCK_K / kGranKB) % 4;

                if constexpr (kUsePerNTileX4) {
                    sm120::SwizzleContext<kSwizzleBMode> b_ctx[kNTilesPerWarp];
                    #pragma unroll
                    for (uint32_t nt = 0; nt < kNTilesPerWarp; ++nt) {
                        int b_row = (lane_idx & 7) + (n_tile_base + nt) * 8;
                        b_ctx[nt].init(b_row, kSMEMKBytes);
                    }

                    uint32_t b_nt[kNTilesPerWarp][4];
                    uint32_t a_frag[2][kMTilesPerWarp][4];
                    sf_t sfb_hoisted[kNTilesPerWarp];
                    sf_t sfa_hoisted[kMTilesPerWarp];
                    sf_t sfb_step[kNTilesPerWarp];
                    sf_t sfa_step[2][kMTilesPerWarp];

                    if constexpr (kGranKB >= BLOCK_K) {
                        #pragma unroll
                        for (uint32_t nt = 0; nt < kNTilesPerWarp; ++nt) {
                            auto packed = sm120::load_sf(smem_sfb[stage], (n_tile_base + nt) * MMA_N + group_id);
                            if constexpr (kIsFP4) {
                                uint8_t b = sm120_mma::extract_sf_byte(packed, sf_byte_b_base);
                                sfb_hoisted[nt] = static_cast<uint16_t>(b) | (static_cast<uint16_t>(b) << 8);
                            } else {
                                sfb_hoisted[nt] = sm120_mma::extract_sf_byte(packed, sf_byte_b_base);
                            }
                        }
                    }
                    if constexpr (kGranKA >= BLOCK_K) {
                        #pragma unroll
                        for (uint32_t mt = 0; mt < kMTilesPerWarp; ++mt) {
                            auto packed = sm120::load_sf(smem_sfa[stage],
                                (m_tile_base + mt) * MMA_M + group_id + (thread_id & 1) * 8);
                            if constexpr (kIsFP4) {
                                uint8_t b = sm120_mma::extract_sf_byte(packed, sf_byte_a_base);
                                sfa_hoisted[mt] = static_cast<uint16_t>(b) | (static_cast<uint16_t>(b) << 8);
                            } else {
                                sfa_hoisted[mt] = sm120_mma::extract_sf_byte(packed, sf_byte_a_base);
                            }
                        }
                    }

                    static constexpr uint32_t kKStepPairs = kKSteps / 2;
                    #pragma unroll
                    for (uint32_t kp = 0; kp < kKStepPairs; ++kp) {
                        const uint32_t ks_base = kp * 2;

                        #pragma unroll
                        for (uint32_t mt = 0; mt < kMTilesPerWarp; ++mt)
                            sm120::load_a_fragment(a_frag[0][mt], smem_a[stage], a_ctx[mt], lane_idx, ks_base, kLdmK);

                        #pragma unroll
                        for (uint32_t nt = 0; nt < kNTilesPerWarp; ++nt)
                            sm120::load_b_per_ntile_x4(b_nt[nt], smem_b[stage], b_ctx[nt], lane_idx, kp, kLdmK * 2);

                        auto load_sf_for_step = [&](uint32_t ks, int sf_buf) {
                            if constexpr (kGranKA < BLOCK_K or kGranKB < BLOCK_K) {
                                const uint32_t sf_step = kb * kKSteps + ks;
                                if constexpr (kIsFP4) {
                                    const uint32_t sf_byte_b = (sf_step * MMA_K / kGranKB) % 4;
                                    const uint32_t sf_byte_a = (sf_step * MMA_K / kGranKA) % 4;
                                    if constexpr (kGranKB < BLOCK_K) {
                                        #pragma unroll
                                        for (uint32_t nt = 0; nt < kNTilesPerWarp; ++nt) {
                                            auto packed = sm120::load_sf(smem_sfb[stage], (n_tile_base + nt) * MMA_N + group_id);
                                            if constexpr (kGranKB <= 32)
                                                sfb_step[nt] = sm120_mma::extract_sf_pair(packed, sf_byte_b);
                                            else {
                                                uint8_t b = sm120_mma::extract_sf_byte(packed, sf_byte_b);
                                                sfb_step[nt] = static_cast<uint16_t>(b) | (static_cast<uint16_t>(b) << 8);
                                            }
                                        }
                                    }
                                    if constexpr (kGranKA < BLOCK_K) {
                                        #pragma unroll
                                        for (uint32_t mt = 0; mt < kMTilesPerWarp; ++mt) {
                                            auto packed = sm120::load_sf(smem_sfa[stage],
                                                (m_tile_base + mt) * MMA_M + group_id + (thread_id & 1) * 8);
                                            if constexpr (kGranKA <= 32)
                                                sfa_step[sf_buf][mt] = sm120_mma::extract_sf_pair(packed, sf_byte_a);
                                            else {
                                                uint8_t b = sm120_mma::extract_sf_byte(packed, sf_byte_a);
                                                sfa_step[sf_buf][mt] = static_cast<uint16_t>(b) | (static_cast<uint16_t>(b) << 8);
                                            }
                                        }
                                    }
                                } else {
                                    const uint32_t sf_byte_b = (sf_step * MMA_K / kGranKB) % 4;
                                    const uint32_t sf_byte_a = (sf_step * MMA_K / kGranKA) % 4;
                                    if constexpr (kGranKB < BLOCK_K) {
                                        #pragma unroll
                                        for (uint32_t nt = 0; nt < kNTilesPerWarp; ++nt)
                                            sfb_step[nt] = sm120_mma::extract_sf_byte(
                                                sm120::load_sf(smem_sfb[stage], (n_tile_base + nt) * MMA_N + group_id), sf_byte_b);
                                    }
                                    if constexpr (kGranKA < BLOCK_K) {
                                        #pragma unroll
                                        for (uint32_t mt = 0; mt < kMTilesPerWarp; ++mt)
                                            sfa_step[sf_buf][mt] = sm120_mma::extract_sf_byte(
                                                sm120::load_sf(smem_sfa[stage],
                                                    (m_tile_base + mt) * MMA_M + group_id + (thread_id & 1) * 8), sf_byte_a);
                                    }
                                }
                            }
                        };

                        load_sf_for_step(ks_base, 0);

                        #pragma unroll
                        for (uint32_t mt = 0; mt < kMTilesPerWarp; ++mt)
                            sm120::load_a_fragment(a_frag[1][mt], smem_a[stage], a_ctx[mt], lane_idx, ks_base + 1, kLdmK);

                        #pragma unroll
                        for (uint32_t mt = 0; mt < kMTilesPerWarp; ++mt) {
                            const sf_t sfa0 = (kGranKA >= BLOCK_K) ? sfa_hoisted[mt] : sfa_step[0][mt];
                            #pragma unroll
                            for (uint32_t nt = 0; nt < kNTilesPerWarp; ++nt) {
                                float (&d)[4] = *reinterpret_cast<float(*)[4]>(&accum[(mt * kNTilesPerWarp + nt) * MMA_ACCUM]);
                                const sf_t sfb = (kGranKB >= BLOCK_K) ? sfb_hoisted[nt] : sfb_step[nt];
                                if constexpr (kIsFP4)
                                    sm120_mma::fp4_mma_block_scaled(d, a_frag[0][mt], b_nt[nt][0], b_nt[nt][1], sfa0, sfb);
                                else
                                    sm120_mma::fp8_mma_block_scaled(d, a_frag[0][mt], b_nt[nt][0], b_nt[nt][1], sfa0, sfb);
                            }
                        }

                        load_sf_for_step(ks_base + 1, 1);

                        #pragma unroll
                        for (uint32_t mt = 0; mt < kMTilesPerWarp; ++mt) {
                            const sf_t sfa1 = (kGranKA >= BLOCK_K) ? sfa_hoisted[mt] : sfa_step[1][mt];
                            #pragma unroll
                            for (uint32_t nt = 0; nt < kNTilesPerWarp; ++nt) {
                                float (&d)[4] = *reinterpret_cast<float(*)[4]>(&accum[(mt * kNTilesPerWarp + nt) * MMA_ACCUM]);
                                const sf_t sfb = (kGranKB >= BLOCK_K) ? sfb_hoisted[nt] : sfb_step[nt];
                                if constexpr (kIsFP4)
                                    sm120_mma::fp4_mma_block_scaled(d, a_frag[1][mt], b_nt[nt][2], b_nt[nt][3], sfa1, sfb);
                                else
                                    sm120_mma::fp8_mma_block_scaled(d, a_frag[1][mt], b_nt[nt][2], b_nt[nt][3], sfa1, sfb);
                            }
                        }
                    }
                } else {
                    // Fallback: original K-step double-buffer (MN-major B, mixed FP8×FP4)
                    sm120::SwizzleContext<kSwizzleBMode> b_ctx[kBKMajor ? kNTilesPerWarp : 1];
                    if constexpr (kBKMajor) {
                        #pragma unroll
                        for (uint32_t nt = 0; nt < kNTilesPerWarp; ++nt) {
                            int b_row = (lane_idx & 7) + (n_tile_base + nt) * 8;
                            b_ctx[nt].init(b_row, kSMEMKBytes);
                        }
                    }
                    uint32_t a_frag[2][kMTilesPerWarp][4];
                    uint32_t b_tile[2][kNTilesPerWarp][2];
                    sf_t sfa_bytes[2][kMTilesPerWarp];
                    sf_t sfb_bytes[2][kNTilesPerWarp];
                    sf_t sfa_hoisted[kMTilesPerWarp];
                    sf_t sfb_hoisted[kNTilesPerWarp];

                    if constexpr (kGranKB >= BLOCK_K) {
                        #pragma unroll
                        for (uint32_t nt = 0; nt < kNTilesPerWarp; ++nt) {
                            auto packed = sm120::load_sf(smem_sfb[stage], (n_tile_base + nt) * MMA_N + group_id);
                            if constexpr (kIsFP4) {
                                uint8_t b = sm120_mma::extract_sf_byte(packed, sf_byte_b_base);
                                sfb_hoisted[nt] = static_cast<uint16_t>(b) | (static_cast<uint16_t>(b) << 8);
                            } else {
                                sfb_hoisted[nt] = sm120_mma::extract_sf_byte(packed, sf_byte_b_base);
                            }
                        }
                    }
                    if constexpr (kGranKA >= BLOCK_K) {
                        #pragma unroll
                        for (uint32_t mt = 0; mt < kMTilesPerWarp; ++mt) {
                            auto packed = sm120::load_sf(smem_sfa[stage],
                                (m_tile_base + mt) * MMA_M + group_id + (thread_id & 1) * 8);
                            if constexpr (kIsFP4) {
                                uint8_t b = sm120_mma::extract_sf_byte(packed, sf_byte_a_base);
                                sfa_hoisted[mt] = static_cast<uint16_t>(b) | (static_cast<uint16_t>(b) << 8);
                            } else {
                                sfa_hoisted[mt] = sm120_mma::extract_sf_byte(packed, sf_byte_a_base);
                            }
                        }
                    }

                    auto load_kstep = [&](int buf, uint32_t ks) {
                        if constexpr (kBKMajor) {
                            if constexpr (kBIsFP4) {
                                #pragma unroll
                                for (uint32_t nt = 0; nt < kNTilesPerWarp; ++nt) {
                                    sm120::load_b_fragment_b4x16_p64(b_tile[buf][nt], smem_b[stage], b_ctx[nt], lane_idx, ks, kLdmK);
                                    b_tile[buf][nt][0] <<= 2;
                                    b_tile[buf][nt][1] <<= 2;
                                }
                            } else {
                                #pragma unroll
                                for (uint32_t nt = 0; nt < kNTilesPerWarp; ++nt)
                                    sm120::load_b_fragment_x2(b_tile[buf][nt], smem_b[stage], b_ctx[nt], lane_idx, ks, kLdmK);
                            }
                        } else {
                            static constexpr uint32_t kBSwizzleB = kSwizzleBMode > 0 ? (__builtin_ctz(kSwizzleBMode) - 4) : 0;
                            static constexpr uint32_t kBSwizzleMask = kSwizzleBMode > 0 ? ((1u << kBSwizzleB) - 1) : 0;
                            static constexpr uint32_t kBSwizzleRowShift = kSwizzleBMode > 0 ? (7 - __builtin_ctz(BLOCK_N)) : 0;
                            #pragma unroll
                            for (uint32_t nt = 0; nt < kNTilesPerWarp; ++nt) {
                                const uint32_t n_col = (n_tile_base + nt) * MMA_N + group_id;
                                uint8_t v[8];
                                #pragma unroll
                                for (uint32_t i = 0; i < 4; ++i) {
                                    const uint32_t k = ks * MMA_K + thread_id * 4 + i;
                                    const uint32_t xor_bits = kSwizzleBMode > 0
                                        ? (((k >> kBSwizzleRowShift) & kBSwizzleMask) << 4) : 0;
                                    v[i] = static_cast<uint8_t>(smem_b[stage][k * BLOCK_N + (n_col ^ xor_bits)]);
                                }
                                #pragma unroll
                                for (uint32_t i = 0; i < 4; ++i) {
                                    const uint32_t k = ks * MMA_K + 16 + thread_id * 4 + i;
                                    const uint32_t xor_bits = kSwizzleBMode > 0
                                        ? (((k >> kBSwizzleRowShift) & kBSwizzleMask) << 4) : 0;
                                    v[4+i] = static_cast<uint8_t>(smem_b[stage][k * BLOCK_N + (n_col ^ xor_bits)]);
                                }
                                b_tile[buf][nt][0] = v[0] | (uint32_t(v[1]) << 8) | (uint32_t(v[2]) << 16) | (uint32_t(v[3]) << 24);
                                b_tile[buf][nt][1] = v[4] | (uint32_t(v[5]) << 8) | (uint32_t(v[6]) << 16) | (uint32_t(v[7]) << 24);
                            }
                        }
                        #pragma unroll
                        for (uint32_t mt = 0; mt < kMTilesPerWarp; ++mt)
                            if constexpr (kAIsFP4) {
                                sm120::load_a_fragment_b4x16(a_frag[buf][mt], smem_a[stage], a_ctx[mt], lane_idx, ks, kLdmK);
                                a_frag[buf][mt][0] <<= 2; a_frag[buf][mt][1] <<= 2;
                                a_frag[buf][mt][2] <<= 2; a_frag[buf][mt][3] <<= 2;
                            } else {
                                sm120::load_a_fragment(a_frag[buf][mt], smem_a[stage], a_ctx[mt], lane_idx, ks, kLdmK);
                            }

                        if constexpr (kGranKA < BLOCK_K or kGranKB < BLOCK_K) {
                            const uint32_t sf_step = (kb * kKSteps + ks);
                            if constexpr (kIsFP4) {
                                const uint32_t sf_byte_a = (sf_step * MMA_K / kGranKA) % 4;
                                const uint32_t sf_byte_b = (sf_step * MMA_K / kGranKB) % 4;
                                #pragma unroll
                                for (uint32_t nt = 0; nt < kNTilesPerWarp; ++nt) {
                                    auto packed = sm120::load_sf(smem_sfb[stage], (n_tile_base + nt) * MMA_N + group_id);
                                    if constexpr (kGranKB <= 32)
                                        sfb_bytes[buf][nt] = sm120_mma::extract_sf_pair(packed, sf_byte_b);
                                    else {
                                        uint8_t b = sm120_mma::extract_sf_byte(packed, sf_byte_b);
                                        sfb_bytes[buf][nt] = static_cast<uint16_t>(b) | (static_cast<uint16_t>(b) << 8);
                                    }
                                }
                                #pragma unroll
                                for (uint32_t mt = 0; mt < kMTilesPerWarp; ++mt) {
                                    auto packed = sm120::load_sf(smem_sfa[stage],
                                        (m_tile_base + mt) * MMA_M + group_id + (thread_id & 1) * 8);
                                    if constexpr (kGranKA <= 32)
                                        sfa_bytes[buf][mt] = sm120_mma::extract_sf_pair(packed, sf_byte_a);
                                    else {
                                        uint8_t b = sm120_mma::extract_sf_byte(packed, sf_byte_a);
                                        sfa_bytes[buf][mt] = static_cast<uint16_t>(b) | (static_cast<uint16_t>(b) << 8);
                                    }
                                }
                            } else {
                                const uint32_t sf_byte_a = (sf_step * MMA_K / kGranKA) % 4;
                                const uint32_t sf_byte_b = (sf_step * MMA_K / kGranKB) % 4;
                                #pragma unroll
                                for (uint32_t nt = 0; nt < kNTilesPerWarp; ++nt)
                                    sfb_bytes[buf][nt] = sm120_mma::extract_sf_byte(
                                        sm120::load_sf(smem_sfb[stage], (n_tile_base + nt) * MMA_N + group_id), sf_byte_b);
                                #pragma unroll
                                for (uint32_t mt = 0; mt < kMTilesPerWarp; ++mt)
                                    sfa_bytes[buf][mt] = sm120_mma::extract_sf_byte(
                                        sm120::load_sf(smem_sfa[stage],
                                            (m_tile_base + mt) * MMA_M + group_id + (thread_id & 1) * 8), sf_byte_a);
                            }
                        }
                    };

                    auto compute_kstep = [&](int buf) {
                        #pragma unroll
                        for (uint32_t mt = 0; mt < kMTilesPerWarp; ++mt) {
                            const sf_t sfa = (kGranKA >= BLOCK_K) ? sfa_hoisted[mt] : sfa_bytes[buf][mt];
                            #pragma unroll
                            for (uint32_t nt = 0; nt < kNTilesPerWarp; ++nt) {
                                float (&d)[4] = *reinterpret_cast<float(*)[4]>(&accum[(mt * kNTilesPerWarp + nt) * MMA_ACCUM]);
                                const sf_t sfb = (kGranKB >= BLOCK_K) ? sfb_hoisted[nt] : sfb_bytes[buf][nt];
                                if constexpr (kAIsFP4)
                                    sm120_mma::fp4_fp8_mixed_mma_block_scaled(d, a_frag[buf][mt], b_tile[buf][nt], sfa, sfb);
                                else if constexpr (kBIsFP4)
                                    sm120_mma::fp8_fp4_mixed_mma_block_scaled(d, a_frag[buf][mt], b_tile[buf][nt], sfa, sfb);
                                else if constexpr (kIsFP4)
                                    sm120_mma::fp4_mma_block_scaled(d, a_frag[buf][mt], b_tile[buf][nt], sfa, sfb);
                                else
                                    sm120_mma::fp8_mma_block_scaled(d, a_frag[buf][mt], b_tile[buf][nt], sfa, sfb);
                            }
                        }
                    };

                    load_kstep(0, 0);
                    #pragma unroll
                    for (uint32_t ks = 0; ks < kKSteps; ++ks) {
                        int cur = ks & 1;
                        int nxt = (ks + 1) & 1;
                        if (ks < kKSteps - 1)
                            load_kstep(nxt, ks + 1);
                        compute_kstep(cur);
                    }
                }

                if (lane_idx == 0)
                    empty_barriers[stage]->arrive();
            }
            } // else (!kUseSFMajorLoop) — original path

            // Epilogue
            if constexpr (kSplitKFactor > 1) {
                // Split-K: write FP32 partials to workspace
                const uint32_t m_base_sk = m_block_idx * BLOCK_M;
                const uint32_t n_base_sk = n_block_idx * BLOCK_N;
                float* ws = gmem_workspace + static_cast<int64_t>(scheduler.split_k_idx) * shape_m * shape_n;

                #pragma unroll
                for (uint32_t mt = 0; mt < kMTilesPerWarp; ++mt) {
                    #pragma unroll
                    for (uint32_t nt = 0; nt < kNTilesPerWarp; ++nt) {
                        const uint32_t ai = (mt * kNTilesPerWarp + nt) * MMA_ACCUM;
                        const uint32_t col = n_base_sk + (n_tile_base + nt) * MMA_N + thread_id * 2;
                        const uint32_t row0 = m_base_sk + (m_tile_base + mt) * MMA_M + group_id;
                        const uint32_t row1 = row0 + 8;

                        if (row0 < shape_m) {
                            auto idx = static_cast<int64_t>(row0) * shape_n + col;
                            if (col < shape_n)     ws[idx]     = accum[ai + 0];
                            if (col + 1 < shape_n) ws[idx + 1] = accum[ai + 1];
                        }
                        if (row1 < shape_m) {
                            auto idx = static_cast<int64_t>(row1) * shape_n + col;
                            if (col < shape_n)     ws[idx]     = accum[ai + 2];
                            if (col + 1 < shape_n) ws[idx + 1] = accum[ai + 3];
                        }
                    }
                }
            } else {
            // Normal epilogue (non-split-K)
            constexpr bool kEpilogueGroupOffset = not is_m_grouped_contiguous(kGemmType);
            const uint32_t m_base = scheduler.template get_global_idx<kEpilogueGroupOffset>(shape_m, BLOCK_M, m_block_idx);
            const uint32_t n_base = n_block_idx * BLOCK_N;
            const uint32_t total_shape_m = (kGemmType == GemmType::KGroupedContiguous or kGemmType == GemmType::MGroupedMasked)
                ? shape_m * kNumGroups : shape_m;

            auto read_cd = [&](const cd_dtype_t& x) -> float {
                if constexpr (cute::is_same_v<cd_dtype_t, float>) return x;
                else return static_cast<float>(x);
            };

            constexpr bool kIsBatchedEpilogue = (kGemmType == GemmType::Batched);
            const int64_t cd_m_stride = static_cast<int64_t>(stride_cd_m);
            const int64_t cd_batch_offset = kIsBatchedEpilogue
                ? static_cast<int64_t>(scheduler.current_group_idx) * stride_cd_batch : 0;

            if constexpr (kUseTMAStoreEpilogue) {
                #pragma unroll
                for (uint32_t ms = 0; ms < kNumEpiMSubs; ++ms) {
                    const uint32_t epi_m_start = ms * kEpiSubM;

                    if (math_warp_idx == 0 and lane_idx == 0)
                        cute::tma_store_wait<0>();
                    cutlass::arch::NamedBarrier::sync(kNumMathThreads, 0);

                    #pragma unroll
                    for (uint32_t mt = 0; mt < kMTilesPerWarp; ++mt) {
                        const uint32_t local_row0 = (m_tile_base + mt) * MMA_M + group_id;
                        const uint32_t local_row1 = local_row0 + 8;
                        if (local_row0 >= epi_m_start and local_row0 < epi_m_start + kEpiSubM) {
                            const uint32_t sub_row0 = local_row0 - epi_m_start;
                            const uint32_t sub_row1 = sub_row0 + 8;
                            #pragma unroll
                            for (uint32_t nt = 0; nt < kNTilesPerWarp; ++nt) {
                                const uint32_t ai = (mt * kNTilesPerWarp + nt) * MMA_ACCUM;
                                const uint32_t local_col = (n_tile_base + nt) * MMA_N + thread_id * 2;
                                float v0 = accum[ai + 0], v1 = accum[ai + 1];
                                float v2 = accum[ai + 2], v3 = accum[ai + 3];

                                // Batched accumulation is handled by SM90_TMA_REDUCE_ADD_3D
                                // (adds SMEM to the existing global C). Reading gmem_c here
                                // too would double-count, so only the non-batched path (plain
                                // SM90_TMA_STORE_2D) reads and accumulates in registers.
                                // NOTE: relies on the invariant below that batched+accumulation
                                // ALWAYS uses SM90_TMA_REDUCE_ADD_3D. If that dispatch ever
                                // becomes a plain STORE, this skip would drop the accumulation.
                                if constexpr (kWithAccumulation and not kIsBatchedEpilogue) {
                                    const uint32_t gr0 = m_base + local_row0, gr1 = m_base + local_row1;
                                    const uint32_t gc = epilogue_type_t::template apply_index_n<MMA_N>(
                                        n_base + (n_tile_base + nt) * MMA_N) + thread_id * 2;
                                    if (gr0 < total_shape_m and gc + 1 < shape_n) {
                                        const auto ci = cd_batch_offset + static_cast<int64_t>(gr0) * cd_m_stride + gc;
                                        v0 += read_cd(gmem_c[ci]); v1 += read_cd(gmem_c[ci + 1]);
                                    }
                                    if (gr1 < total_shape_m and gc + 1 < shape_n) {
                                        const auto ci = cd_batch_offset + static_cast<int64_t>(gr1) * cd_m_stride + gc;
                                        v2 += read_cd(gmem_c[ci]); v3 += read_cd(gmem_c[ci + 1]);
                                    }
                                }

                                const uint32_t sub_tile = local_col / kTMAStoreInnerDim;
                                const uint32_t col_in_sub = local_col % kTMAStoreInnerDim;
                                const uint32_t col_byte_in_sub = col_in_sub * sizeof(cd_dtype_t);
                                const uint32_t sw0 = col_byte_in_sub ^ (((sub_row0 >> kSwizzleCDShift) & kSwizzleCDMask) << 4);
                                const uint32_t sw1 = col_byte_in_sub ^ (((sub_row1 >> kSwizzleCDShift) & kSwizzleCDMask) << 4);
                                cd_dtype_t p0[2] = {cd_dtype_t(v0), cd_dtype_t(v1)};
                                cd_dtype_t p1[2] = {cd_dtype_t(v2), cd_dtype_t(v3)};
                                auto* smem_d_bytes = reinterpret_cast<char*>(smem_d_base);
                                const uint32_t sub_base = sub_tile * kSwizzleCDMode * kEpiSubM;
                                using pair_store_t = cute::conditional_t<sizeof(cd_dtype_t) <= 2, uint32_t, uint64_t>;
                                *reinterpret_cast<pair_store_t*>(smem_d_bytes + sub_base + sub_row0 * kSwizzleCDMode + sw0) =
                                    *reinterpret_cast<const pair_store_t*>(p0);
                                *reinterpret_cast<pair_store_t*>(smem_d_bytes + sub_base + sub_row1 * kSwizzleCDMode + sw1) =
                                    *reinterpret_cast<const pair_store_t*>(p1);
                            }
                        }
                    }

                    cute::tma_store_fence();
                    cutlass::arch::NamedBarrier::sync(kNumMathThreads, 0);

                    if (math_warp_idx == 0 and lane_idx == 0) {
                        const uint32_t batch_store_idx = kIsBatchedEpilogue ? scheduler.current_group_idx : 0;
                        #pragma unroll
                        for (uint32_t ts = 0; ts < kNumTMAStores; ++ts) {
                            auto* smem_src = reinterpret_cast<char*>(smem_d_base) + ts * kSwizzleCDMode * kEpiSubM;
                            const uint32_t n_store = epilogue_type_t::template apply_index_n<kTMAStoreInnerDim>(
                                n_base + ts * kTMAStoreInnerDim);
                            if constexpr (kIsBatchedEpilogue) {
                                if constexpr (kWithAccumulation)
                                    cute::SM90_TMA_REDUCE_ADD_3D::copy(
                                        &tensor_map_cd, smem_src,
                                        n_store, m_base + epi_m_start, batch_store_idx);
                                else
                                    cute::SM90_TMA_STORE_3D::copy(
                                        &tensor_map_cd, smem_src,
                                        n_store, m_base + epi_m_start, batch_store_idx);
                            } else {
                                cute::SM90_TMA_STORE_2D::copy(
                                    &tensor_map_cd, smem_src,
                                    n_store, m_base + epi_m_start);
                            }
                        }
                        cute::tma_store_arrive();
                    }
                } // ms loop
            } else {
                auto store_pair = [&](cd_dtype_t* ptr, float a, float b) {
                    if constexpr (cute::is_same_v<cd_dtype_t, float>) {
                        *reinterpret_cast<float2*>(ptr) = make_float2(a, b);
                    } else {
                        ptr[0] = cd_dtype_t(a);
                        ptr[1] = cd_dtype_t(b);
                    }
                };

                const bool can_pair = (stride_cd_n == 0);
                const int64_t cd_n_stride = can_pair ? 1 : static_cast<int64_t>(stride_cd_n);

                #pragma unroll
                for (uint32_t mt = 0; mt < kMTilesPerWarp; ++mt) {
                    #pragma unroll
                    for (uint32_t nt = 0; nt < kNTilesPerWarp; ++nt) {
                        const uint32_t ai = (mt * kNTilesPerWarp + nt) * MMA_ACCUM;
                        const uint32_t nt_global = n_tile_base + nt;
                        const uint32_t col = epilogue_type_t::template apply_index_n<MMA_N>(n_base + nt_global * MMA_N) + thread_id * 2;
                        const uint32_t row0 = m_base + (m_tile_base + mt) * MMA_M + group_id;
                        const uint32_t row1 = row0 + 8;

                        if (can_pair) {
                            if (row0 < total_shape_m and col + 1 < shape_n) {
                                auto idx = cd_batch_offset + static_cast<int64_t>(row0) * cd_m_stride + col;
                                float v0 = accum[ai + 0], v1 = accum[ai + 1];
                                if constexpr (kWithAccumulation) { v0 += read_cd(gmem_c[idx]); v1 += read_cd(gmem_c[idx + 1]); }
                                store_pair(&gmem_d[idx], v0, v1);
                            }
                            if (row1 < total_shape_m and col + 1 < shape_n) {
                                auto idx = cd_batch_offset + static_cast<int64_t>(row1) * cd_m_stride + col;
                                float v2 = accum[ai + 2], v3 = accum[ai + 3];
                                if constexpr (kWithAccumulation) { v2 += read_cd(gmem_c[idx]); v3 += read_cd(gmem_c[idx + 1]); }
                                store_pair(&gmem_d[idx], v2, v3);
                            }
                        } else {
                            // Strided store: per-element N bounds check (handles shape_n=1)
                            if (row0 < total_shape_m) {
                                auto base = cd_batch_offset + static_cast<int64_t>(row0) * cd_m_stride;
                                if (col < shape_n)
                                    gmem_d[base + static_cast<int64_t>(col) * cd_n_stride] = cd_dtype_t(accum[ai + 0]);
                                if (col + 1 < shape_n)
                                    gmem_d[base + static_cast<int64_t>(col + 1) * cd_n_stride] = cd_dtype_t(accum[ai + 1]);
                            }
                            if (row1 < total_shape_m) {
                                auto base = cd_batch_offset + static_cast<int64_t>(row1) * cd_m_stride;
                                if (col < shape_n)
                                    gmem_d[base + static_cast<int64_t>(col) * cd_n_stride] = cd_dtype_t(accum[ai + 2]);
                                if (col + 1 < shape_n)
                                    gmem_d[base + static_cast<int64_t>(col + 1) * cd_n_stride] = cd_dtype_t(accum[ai + 3]);
                            }
                        }
                    }
                }
            }
            } // end else (non-split-K epilogue)

            // Every math warp releases its complete M64x16 output stripe.
            // The acq_rel RMW chain makes the final (64*8) publisher observe
            // every preceding BF16 store before publishing task readiness.
            __threadfence();
            __syncwarp();
            if (lane_idx == 0) {
                cuda::atomic_ref<unsigned int, cuda::thread_scope_device> done(
                    cake_w1_warp_done[m_block_idx]);
                const unsigned int previous = done.fetch_add(
                    1u, cuda::memory_order_acq_rel);
                if ((previous & 7u) == 7u)
                    atomicAdd(cake_w1_tiles_completed, 1u);
            }
        } // persistent loop

        // Final TMA store drain
        if constexpr (kUseTMAStoreEpilogue and kSplitKFactor == 1) {
            if (math_warp_idx == 0 and lane_idx == 0)
                cute::tma_store_wait<0>();
        }
    }

    // setmaxnreg's immediate is the absolute target register count.
    // WG0/WG1 return from 232 to 168; WG2 returns from 40 to 168.  Every
    // warp in each warpgroup executes the same instruction, then the whole CTA
    // reconverges before the wrapper's next cooperative phase.
    __syncthreads();
    if (warp_idx < kNumMathWarps) {
        cutlass::arch::warpgroup_reg_dealloc<168>();
    } else {
        cutlass::arch::warpgroup_reg_alloc<168>();
    }
    __syncthreads();

    // Signal completion for PDL (allows dependent reduce kernel to start)
    if constexpr (kSplitKFactor > 1) {
        cudaTriggerProgrammaticLaunchCompletion();
    }

#else
    if (blockIdx.x == 0 and threadIdx.x == 0)
        DG_DEVICE_ASSERT(false and "This kernel only supports sm_120a");
#endif
}

} // namespace deep_gemm

namespace deep_gemm {
template <uint32_t SHAPE_M, uint32_t SHAPE_N, uint32_t SHAPE_K,
          uint32_t kGranKA, uint32_t kGranKB,
          uint32_t kNumGroups,
          uint32_t BLOCK_M, uint32_t BLOCK_N, uint32_t BLOCK_K,
          uint32_t kSwizzleAMode, uint32_t kSwizzleBMode,
          uint32_t kSwizzleCDMode,
          uint32_t kNumStages,
          uint32_t kNumTMAThreads, uint32_t kNumMathThreads,
          uint32_t kNumSMs,
          GemmType kGemmType, bool kWithAccumulation,
          typename cd_dtype_t,
          typename epilogue_type_t = epilogue::transform::EpilogueIdentity,
          bool kIsFP4 = false,
          bool kBIsFP4 = false,
          bool kAIsFP4 = false,
          bool kBKMajor = true,
          bool kKGroupedConstantStride = false,
          uint32_t kEpiSubM = BLOCK_M,
          uint32_t kSplitKFactor = 1>
__device__ __forceinline__ void
cake_sm120_canonical_ready_chunk8_w2_gemm(cd_dtype_t* gmem_d, const cd_dtype_t* gmem_c,
                             __nv_fp8_e4m3* gmem_a_ptr, __nv_fp8_e4m3* gmem_b_ptr,
                             int* grouped_layout,
                             cute::TmaDescriptor* tensor_map_buffer,
                             float* gmem_workspace,
                             uint32_t shape_m, uint32_t shape_n, uint32_t shape_k,
                             uint32_t stride_cd_m, uint32_t stride_cd_n, uint32_t stride_cd_batch,
                             const cute::TmaDescriptor& tensor_map_a_base,
                             const cute::TmaDescriptor& tensor_map_b_base,
                             const cute::TmaDescriptor& tensor_map_sfa,
                             const cute::TmaDescriptor& tensor_map_sfb,
                             const cute::TmaDescriptor& tensor_map_cd,
                             uint32_t cake_assigned_m_block,
                             uint32_t cake_assigned_n_block,
                             int const* cake_meta_source_rank,
                             int const* cake_meta_result_index,
                             int const* cake_source_route_counts,
                             int cake_world_size, int cake_slot,
                             __nv_bfloat16* cake_result_out,
                             unsigned int* cake_protocol_error) {
#if (defined(__CUDA_ARCH__) and (__CUDA_ARCH__ >= 1200)) or defined(__CLION_IDE__)
    DG_STATIC_ASSERT(kSwizzleCDMode == 0,
                     "W2 chunk publication requires the direct-store epilogue");
    DG_STATIC_ASSERT(BLOCK_M == cake_moe::kTaskM && SHAPE_N == cake_moe::kW2ShapeN,
                     "W2 route-row-base cache requires the configured M/N shape");
    namespace sm120_mma = mma::sm120;
    using Barrier = cutlass::arch::ClusterTransactionBarrier;

    static constexpr uint32_t MMA_M = 16;
    static constexpr uint32_t MMA_N = 8;
    static constexpr uint32_t MMA_K = kIsFP4 ? sm120_mma::FP4_MMA_K : sm120_mma::FP8_MMA_K;
    static constexpr uint32_t MMA_ACCUM = 4;

    DG_STATIC_ASSERT(cute::is_same_v<cd_dtype_t, float> or cute::is_same_v<cd_dtype_t, cutlass::bfloat16_t>,
                     "Only float or bfloat16 output supported");
    DG_STATIC_ASSERT(!(kIsFP4 && kBIsFP4), "Use kIsFP4 for symmetric FP4x4, not kBIsFP4");
    // kAIsFP4 = mixed FP4_A x FP8_B (swapAB of the FP8xFP4 mixed path): A fp4-unpacked at k32, B fp8.
    DG_STATIC_ASSERT(!(kAIsFP4 && (kIsFP4 || kBIsFP4)), "kAIsFP4 (fp4_A x fp8_B) is exclusive");
    DG_STATIC_ASSERT(!kBIsFP4 || kBKMajor, "Mixed FP8xFP4 requires K-major B");
    DG_STATIC_ASSERT(kNumTMAThreads > 0, "SM120a always uses warp-specialized pipeline");
    DG_STATIC_ASSERT(kNumMathThreads % 32 == 0, "Invalid math threads");
    DG_STATIC_ASSERT(BLOCK_M % MMA_M == 0 and BLOCK_N % MMA_N == 0 and BLOCK_K % MMA_K == 0, "Invalid block dims");

    static constexpr uint32_t kNumSFAStagesPerLoad = (4 * kGranKA) / BLOCK_K;
    static constexpr uint32_t kNumSFBStagesPerLoad = (4 * kGranKB) / BLOCK_K;

    static constexpr uint32_t kNumMathWarps = kNumMathThreads / 32;
    static constexpr uint32_t kNTiles = BLOCK_N / MMA_N;
    static constexpr uint32_t kKSteps = BLOCK_K / MMA_K;

    // Cooperative warp layout: warps split across M and N dimensions
    static constexpr uint32_t kNWarps = 2;
    static constexpr uint32_t kMWarps = kNumMathWarps / kNWarps;
    static constexpr uint32_t kMTilesPerWarp = BLOCK_M / kMWarps / MMA_M;
    static constexpr uint32_t kNTilesPerWarp = kNTiles / kNWarps;
    static constexpr uint32_t kAccumPerWarp = kMTilesPerWarp * kNTilesPerWarp * MMA_ACCUM;

    DG_STATIC_ASSERT(BLOCK_M == kMWarps * kMTilesPerWarp * MMA_M, "M tiles must divide evenly");
    DG_STATIC_ASSERT(kNTiles % kNWarps == 0, "N tiles must divide evenly among N warps");
    DG_STATIC_ASSERT(not kBKMajor or kNTilesPerWarp >= 1, "Need at least 1 N-tile per warp");

    static constexpr uint32_t kTMARegisters = 40;
    static constexpr uint32_t kMMARegisters = 232;

    // SMEM D buffer for TMA store epilogue (sub-tile: kEpiSubM rows at a time)
    static constexpr uint32_t kSafeSwizzleCDMode = kSwizzleCDMode > 0 ? kSwizzleCDMode : 1;
    static constexpr bool kUseTMAStoreEpilogue = kSwizzleCDMode > 0
        and BLOCK_N * sizeof(cd_dtype_t) >= kSwizzleCDMode
        and (BLOCK_N * sizeof(cd_dtype_t)) % kSafeSwizzleCDMode == 0;
    static constexpr uint32_t kNumEpiMSubs = kUseTMAStoreEpilogue ? (BLOCK_M / kEpiSubM) : 0;
    static constexpr uint32_t SMEM_D = kUseTMAStoreEpilogue
        ? static_cast<uint32_t>((BLOCK_N * sizeof(cd_dtype_t) / kSwizzleCDMode) * kSwizzleCDMode * kEpiSubM)
        : 0u;
    static constexpr uint32_t kSwizzleCDShift = kSwizzleCDMode > 0 ? (7 - __builtin_ctz(kSwizzleCDMode)) : 0;
    static constexpr uint32_t kSwizzleCDMask = kSwizzleCDMode > 0 ? (kSwizzleCDMode / 16 - 1) : 0;
    static constexpr uint32_t kTMAStoreInnerDim = kSwizzleCDMode / sizeof(cd_dtype_t);
    static constexpr uint32_t kNumTMAStores = kUseTMAStoreEpilogue
        ? BLOCK_N * sizeof(cd_dtype_t) / kSwizzleCDMode : 0;

    // FP4 uses packed SMEM (4-bit per element = 0.5 bytes), FP8 uses 1 byte per element.
    static constexpr uint32_t kSMEMKBytes = kIsFP4 ? (BLOCK_K / 2) : BLOCK_K;
    static constexpr uint32_t SMEM_A  = BLOCK_M * kSMEMKBytes;
    static constexpr uint32_t SMEM_B  = kBKMajor ? (BLOCK_N * kSMEMKBytes) : (BLOCK_K * BLOCK_N);
    static constexpr uint32_t SMEM_SFA = math::constexpr_align(static_cast<uint32_t>(BLOCK_M * sizeof(int32_t)), 128u);
    static constexpr uint32_t SMEM_SFB = math::constexpr_align(static_cast<uint32_t>(BLOCK_N * sizeof(int32_t)), 128u);
    static constexpr uint32_t TMA_SFA_BYTES = BLOCK_M * sizeof(int32_t);
    static constexpr uint32_t TMA_SFB_BYTES = BLOCK_N * sizeof(int32_t);
    // TMA mbarrier reports GMEM bytes. For .b4x16_p64 (kBIsFP4): GMEM = SMEM/2 (packed).
    // For packed FP4 (kIsFP4): SMEM already uses packed size, so SMEM_B = GMEM bytes.
    static constexpr uint32_t TMA_B_BYTES = kBIsFP4 ? (SMEM_B / 2) : SMEM_B;
    // kAIsFP4: A is fp4 packed in GMEM (.b4x16 expands to unpacked SMEM), so GMEM = SMEM_A/2.
    static constexpr uint32_t TMA_A_BYTES = kAIsFP4 ? (SMEM_A / 2) : SMEM_A;
    static constexpr uint32_t SMEM_TMA_BYTES = TMA_A_BYTES + TMA_B_BYTES + TMA_SFA_BYTES + TMA_SFB_BYTES;
    // ldmatrix K stride in bytes: FP4 packed = MMA_K/2, FP8 = MMA_K. Both = 32 bytes.
    static constexpr uint32_t kLdmK = kIsFP4 ? (MMA_K / 2) : MMA_K;
    // tma::copy swizzle for split computation: FP4 packed with B64 has 64 byte rows = full BLOCK_K,
    // so one TMA copy covers the entire tile. Use 0 to get single-copy path.
    static constexpr uint32_t kTMACopySwizzleA = kIsFP4 ? 0u : kSwizzleAMode;
    static constexpr uint32_t kTMACopySwizzleB = kIsFP4 ? 0u : kSwizzleBMode;

    shape_m = SHAPE_M != 0 ? SHAPE_M : shape_m;
    shape_n = SHAPE_N != 0 ? SHAPE_N : shape_n;
    shape_k = SHAPE_K != 0 ? SHAPE_K : shape_k;

    const uint32_t warp_idx = __shfl_sync(0xffffffff, threadIdx.x / 32, 0);
    const uint32_t lane_idx = threadIdx.x % 32;

    // SMEM layout: pipeline data first (1024-aligned for B128 swizzle),
    // tensor map descriptors at the end (K-grouped only)
    extern __shared__ __align__(1024) uint8_t smem_buffer[];

    auto smem_d_base = reinterpret_cast<cd_dtype_t*>(smem_buffer);

    constexpr uint32_t PIPE_BASE = SMEM_D;
    auto smem_a = utils::PatternVisitor([&](const uint32_t& s) {
        return reinterpret_cast<char*>(smem_buffer + PIPE_BASE + s * SMEM_A);
    });
    auto smem_b = utils::PatternVisitor([&](const uint32_t& s) {
        return reinterpret_cast<char*>(smem_buffer + PIPE_BASE + kNumStages * SMEM_A + s * SMEM_B);
    });
    constexpr uint32_t SF_BASE = PIPE_BASE + kNumStages * (SMEM_A + SMEM_B);
    auto smem_sfa = utils::PatternVisitor([&](const uint32_t& s) {
        return reinterpret_cast<char*>(smem_buffer + SF_BASE + s * SMEM_SFA);
    });
    auto smem_sfb = utils::PatternVisitor([&](const uint32_t& s) {
        return reinterpret_cast<char*>(smem_buffer + SF_BASE + kNumStages * SMEM_SFA + s * SMEM_SFB);
    });
    constexpr uint32_t BAR_BASE = SF_BASE + kNumStages * (SMEM_SFA + SMEM_SFB);
    auto full_barriers = utils::PatternVisitor([&](const uint32_t& s) {
        return reinterpret_cast<Barrier*>(smem_buffer + BAR_BASE + s * sizeof(Barrier));
    });
    auto empty_barriers = utils::PatternVisitor([&](const uint32_t& s) {
        return reinterpret_cast<Barrier*>(smem_buffer + BAR_BASE + (kNumStages + s) * sizeof(Barrier));
    });

    // Tensor map descriptors at the end of SMEM (K-grouped only)
    constexpr uint32_t TM_BASE = BAR_BASE + 2 * kNumStages * sizeof(Barrier);
    auto smem_tm_a = reinterpret_cast<cute::TmaDescriptor*>(smem_buffer + TM_BASE);
    auto smem_tm_b = smem_tm_a + 1;
    auto gmem_tm_a = tensor_map_buffer + blockIdx.x * 2;
    auto gmem_tm_b = gmem_tm_a + 1;

    // The selected three-stage pipeline ends before byte 76,080.  Reserve
    // both possible K-grouped descriptors, align the phase-local tail, and
    // cache one prevalidated result-row base for each row of this M64 task.
    // ready-pre's offset-92,160 scratch is dead before workers enter W2.
    constexpr uint32_t RESULT_ROW_BASE_CACHE = math::constexpr_align(
        TM_BASE + 2u * static_cast<uint32_t>(sizeof(cute::TmaDescriptor)), 128u);
    static_assert(RESULT_ROW_BASE_CACHE == 76416u);
    static_assert(RESULT_ROW_BASE_CACHE + BLOCK_M * sizeof(uint32_t) <= 94208u);
    constexpr unsigned long long MAX_RESULT_ELEMENT =
        (unsigned long long)cake_moe::kMaxResultElementIndex;
    static_assert(MAX_RESULT_ELEMENT < (unsigned long long)kCakeSm120InvalidResultRowBase);
    auto result_row_base = reinterpret_cast<uint32_t*>(
        smem_buffer + RESULT_ROW_BASE_CACHE);

    if ((uint32_t)threadIdx.x < BLOCK_M) {
        const uint32_t row = cake_assigned_m_block * BLOCK_M +
            (uint32_t)threadIdx.x;
        const int source = cake_meta_source_rank[row];
        uint32_t base = kCakeSm120InvalidResultRowBase;
        if (source != -1) {
            if (source < 0 || source >= cake_world_size || source >= cake_moe::kPhysicalRanks) {
                atomicMax(cake_protocol_error, 1u);
            } else {
                const int result_index = cake_meta_result_index[row];
                const int count = cake_source_route_counts[source];
                if (result_index < 0 || result_index >= count ||
                    count < 0 || count > cake_moe::kMaxRoutesPerPeer) {
                    atomicMax(cake_protocol_error, 1u);
                } else {
                    base = (uint32_t)(source * 2 + cake_slot) * (uint32_t)cake_moe::kResultElementsPerPeer +
                           (uint32_t)result_index * cake_moe::kOutput;
                }
            }
        }
        result_row_base[threadIdx.x] = base;
    }

    // Prefetch TMA descriptors
    if (warp_idx == 0 and cute::elect_one_sync()) {
        cute::prefetch_tma_descriptor(&tensor_map_a_base);
        cute::prefetch_tma_descriptor(&tensor_map_b_base);
        cute::prefetch_tma_descriptor(&tensor_map_sfa);
        cute::prefetch_tma_descriptor(&tensor_map_sfb);
        cute::prefetch_tma_descriptor(&tensor_map_cd);
    }
    __syncwarp();

    // Barrier init (done by warp 1 before producer/consumer split)
    if (warp_idx == 1 and cute::elect_one_sync()) {
        if constexpr (kGemmType == GemmType::KGroupedContiguous) {
            *smem_tm_a = tensor_map_a_base;
            *smem_tm_b = tensor_map_b_base;
        }
        #pragma unroll
        for (uint32_t i = 0; i < kNumStages; ++i) {
            full_barriers[i]->init(1);
            empty_barriers[i]->init(kNumMathWarps);
        }
        cutlass::arch::fence_barrier_init();
    }
    __syncthreads();

    // PDL belongs to the standalone launch boundary.  The ordered
    // wrapper performs a cooperative grid rendezvous before this phase.

    // Persistent scheduler
    uint32_t m_block_idx, n_block_idx;
    static constexpr uint32_t kSFKAlignment = (kGranKA > kGranKB ? kGranKA : kGranKB) * 4;
    auto scheduler = sched::CakeChunk8Scheduler<kGemmType, BLOCK_M, BLOCK_N, kNumGroups, 1, false, kNumSMs,
        false, 128u, kSFKAlignment, sched::get_num_1d_blocks_per_group<kGemmType, BLOCK_M, BLOCK_N, kNumSMs, false>(), kSplitKFactor>(
        shape_m, shape_n, shape_k, grouped_layout, cake_assigned_m_block, cake_assigned_n_block);
    const auto get_pipeline = [=](const uint32_t& iter_idx) -> cute::tuple<uint32_t, uint32_t> {
        return {iter_idx % kNumStages, (iter_idx / kNumStages) & 1};
    };

    // PRODUCER WARP GROUP (TMA warps, 40 regs)
    if (warp_idx >= kNumMathWarps) {
        cutlass::arch::warpgroup_reg_dealloc<kTMARegisters>();

        const bool is_tma_leader = (warp_idx == kNumMathWarps and lane_idx == 0);
        uint32_t tma_iter_idx = 0;

        if (is_tma_leader) {
            uint32_t last_group_idx = kNumGroups;
            while (scheduler.get_next_block(m_block_idx, n_block_idx)) {
                // Skip empty/padding tiles in the contiguous grouped layout: m_indices
                // is -1 for blocks with no routed tokens. The worst-case M_sum reserves a
                // block per local expert, but at decode only a few are routed; processing
                // the rest wastes a full-width GEMM tile (the dominant EP-decode cost).
                // Producer and consumer apply the identical check, so no barrier ops are
                // issued for skipped blocks and the pipeline stays in sync.
                if constexpr (kGemmType == GemmType::MGroupedContiguous) {
                    if (__ldg(grouped_layout + m_block_idx * BLOCK_M) < 0)
                        continue;
                }
                if constexpr (kGemmType == GemmType::KGroupedContiguous) {
                    if (last_group_idx != scheduler.current_group_idx) {
                        last_group_idx = scheduler.current_group_idx;

                        const auto a_base = reinterpret_cast<const char*>(gmem_a_ptr);
                        const auto b_base = reinterpret_cast<const char*>(gmem_b_ptr);

                        if constexpr (kKGroupedConstantStride) {
                            const uint64_t a_k_byte_offset = kIsFP4
                                ? (static_cast<uint64_t>(scheduler.current_k_cumsum) / 2)
                                : (static_cast<uint64_t>(scheduler.current_k_cumsum));
                            const uint64_t b_k_byte_offset = (kIsFP4 || kBIsFP4)
                                ? (static_cast<uint64_t>(scheduler.current_k_cumsum) / 2)
                                : (static_cast<uint64_t>(scheduler.current_k_cumsum));
                            ptx::tensor_map_replace_global_addr_in_smem(smem_tm_a, a_base + a_k_byte_offset);
                            ptx::tensor_map_replace_global_addr_in_smem(smem_tm_b, b_base + b_k_byte_offset);
                            ptx::tensor_map_replace_global_dim_in_smem(smem_tm_a, scheduler.current_shape_k);
                            ptx::tensor_map_replace_global_dim_in_smem(smem_tm_b, scheduler.current_shape_k);
                        } else {
                            const uint64_t a_offset = kIsFP4
                                ? (static_cast<uint64_t>(scheduler.current_k_cumsum) * shape_m / 2)
                                : (static_cast<uint64_t>(scheduler.current_k_cumsum) * shape_m);
                            const uint64_t b_offset = (kIsFP4 || kBIsFP4)
                                ? (static_cast<uint64_t>(scheduler.current_k_cumsum) * shape_n / 2)
                                : (static_cast<uint64_t>(scheduler.current_k_cumsum) * shape_n);
                            ptx::tensor_map_replace_global_addr_in_smem(smem_tm_a, a_base + a_offset);
                            ptx::tensor_map_replace_global_addr_in_smem(smem_tm_b, b_base + b_offset);
                            const uint64_t a_new_stride = kIsFP4
                                ? static_cast<uint64_t>(scheduler.current_shape_k / 2)
                                : static_cast<uint64_t>(scheduler.current_shape_k);
                            const uint64_t b_new_stride = (kIsFP4 || kBIsFP4)
                                ? static_cast<uint64_t>(scheduler.current_shape_k / 2)
                                : static_cast<uint64_t>(scheduler.current_shape_k);
                            ptx::tensor_map_replace_global_inner_dim_stride_in_smem(
                                smem_tm_a, scheduler.current_shape_k, a_new_stride);
                            ptx::tensor_map_replace_global_inner_dim_stride_in_smem(
                                smem_tm_b, scheduler.current_shape_k, b_new_stride);
                        }

                        *gmem_tm_a = *smem_tm_a;
                        *gmem_tm_b = *smem_tm_b;
                        ptx::tensor_map_release_gpu();
                        ptx::tensor_map_acquire_gpu(gmem_tm_a);
                        ptx::tensor_map_acquire_gpu(gmem_tm_b);
                    }
                }

                const uint32_t current_shape_k = (kGemmType == GemmType::KGroupedContiguous ? scheduler.current_shape_k : shape_k);
                const uint32_t num_k_blocks = math::ceil_div(current_shape_k, BLOCK_K);
                uint32_t kb_start = 0, kb_end = num_k_blocks;
                if constexpr (kSplitKFactor > 1) {
                    const uint32_t k_per_split = num_k_blocks / kSplitKFactor;
                    kb_start = scheduler.split_k_idx * k_per_split;
                    kb_end = (scheduler.split_k_idx == kSplitKFactor - 1) ? num_k_blocks : kb_start + k_per_split;
                }
                constexpr bool kAGroupOffset = (kGemmType == GemmType::MGroupedMasked);
                const uint32_t m_idx = scheduler.template get_global_idx<kAGroupOffset>(shape_m, BLOCK_M, m_block_idx);
                constexpr bool kBGroupOffset = not (kGemmType == GemmType::Normal or kGemmType == GemmType::KGroupedContiguous);
                const uint32_t n_idx = scheduler.template get_global_idx<kBGroupOffset>(shape_n, BLOCK_N, n_block_idx, m_block_idx);
                const auto tma_a_desc = (kGemmType == GemmType::KGroupedContiguous ? gmem_tm_a : &tensor_map_a_base);
                const auto tma_b_desc = (kGemmType == GemmType::KGroupedContiguous ? gmem_tm_b : &tensor_map_b_base);

                constexpr bool kIsBatchedMM = (kGemmType == GemmType::Batched);
                const uint32_t batch_idx = kIsBatchedMM ? scheduler.current_group_idx : 0;

                for (uint32_t kb = kb_start; kb < kb_end; ++kb) {
                    CUTE_TIE_DECL(get_pipeline(tma_iter_idx++), s, p);
                    empty_barriers[s]->wait(p ^ 1);

                    const uint32_t k_idx = kb * BLOCK_K;
                    uint32_t sfa_k, sfb_k;
                    if constexpr (kGemmType == GemmType::KGroupedContiguous) {
                        sfa_k = scheduler.current_sf_k_cumsum + kb / kNumSFAStagesPerLoad;
                        sfb_k = scheduler.current_sf_k_cumsum + kb / kNumSFBStagesPerLoad;
                    } else {
                        const uint32_t shape_sfa_k = math::ceil_div(shape_k, BLOCK_K * kNumSFAStagesPerLoad);
                        const uint32_t shape_sfb_k = math::ceil_div(shape_k, BLOCK_K * kNumSFBStagesPerLoad);
                        constexpr bool kSFAGroupOffset = not is_m_grouped_contiguous(kGemmType);
                        sfa_k = scheduler.template get_global_idx<kSFAGroupOffset, sched::IndexType::SF_K>(
                            shape_sfa_k, 1, kb / kNumSFAStagesPerLoad, m_block_idx);
                        constexpr bool kSFBGroupOffset = not (kGemmType == GemmType::Normal);
                        sfb_k = scheduler.template get_global_idx<kSFBGroupOffset, sched::IndexType::SF_K>(
                            shape_sfb_k, 1, kb / kNumSFBStagesPerLoad, m_block_idx);
                    }
                    tma::copy<BLOCK_M, BLOCK_K, 0>(&tensor_map_sfa, full_barriers[s], smem_sfa[s], m_block_idx * BLOCK_M, sfa_k, 1);
                    tma::copy<BLOCK_N, BLOCK_K, 0>(&tensor_map_sfb, full_barriers[s], smem_sfb[s], n_block_idx * BLOCK_N, sfb_k, 1);
                    tma::copy<BLOCK_K, BLOCK_M, kTMACopySwizzleA, char, kIsBatchedMM>(tma_a_desc, full_barriers[s], smem_a[s], k_idx, m_idx, 1, batch_idx);
                    if constexpr (kBKMajor) {
                        tma::copy<BLOCK_K, BLOCK_N, kTMACopySwizzleB, char, kIsBatchedMM>(tma_b_desc, full_barriers[s], smem_b[s], k_idx, n_idx, 1, batch_idx);
                    } else {
                        tma::copy<BLOCK_N, BLOCK_K, kSwizzleBMode, char, kIsBatchedMM>(
                            tma_b_desc, full_barriers[s], smem_b[s],
                            n_idx, k_idx, 1, batch_idx);
                    }
                    full_barriers[s]->arrive_and_expect_tx(SMEM_TMA_BYTES);
                }
            }
        }
    }
    // CONSUMER WARP GROUPS (math warps, 232 regs)
    else {
        cutlass::arch::warpgroup_reg_alloc<kMMARegisters>();

        const uint32_t math_warp_idx = warp_idx;
        const uint32_t group_id = lane_idx / 4;
        const uint32_t thread_id = lane_idx % 4;
        const uint32_t warp_m = math_warp_idx / kNWarps;
        const uint32_t warp_n = math_warp_idx % kNWarps;
        const uint32_t m_tile_base = warp_m * kMTilesPerWarp;
        const uint32_t n_tile_base = warp_n * kNTilesPerWarp;

        float accum[kAccumPerWarp];
        uint32_t iter_idx = 0;

        while (scheduler.get_next_block(m_block_idx, n_block_idx)) {
            // Skip empty/padding tiles (m_indices == -1); see the matching check in the
            // producer loop. Both warp groups skip identically, so barriers stay in sync.
            if constexpr (kGemmType == GemmType::MGroupedContiguous) {
                if (__ldg(grouped_layout + m_block_idx * BLOCK_M) < 0)
                    continue;
            }
            const uint32_t current_shape_k = (kGemmType == GemmType::KGroupedContiguous ? scheduler.current_shape_k : shape_k);
            const uint32_t num_k_blocks_total = math::ceil_div(current_shape_k, BLOCK_K);
            uint32_t num_k_blocks_start = 0, num_k_blocks = num_k_blocks_total;
            if constexpr (kSplitKFactor > 1) {
                const uint32_t k_per_split = num_k_blocks_total / kSplitKFactor;
                num_k_blocks_start = scheduler.split_k_idx * k_per_split;
                num_k_blocks = ((scheduler.split_k_idx == kSplitKFactor - 1) ? num_k_blocks_total : num_k_blocks_start + k_per_split) - num_k_blocks_start;
            }

            #pragma unroll
            for (uint32_t i = 0; i < kAccumPerWarp; ++i) accum[i] = 0.f;

            // kAIsFP4 uses the regular (non-perNTileX4) path to keep the fp4-A load localized.
            static constexpr bool kUsePerNTileX4 = kBKMajor and not kBIsFP4 and not kAIsFP4 and (kKSteps >= 2);
            using sf_t = cute::conditional_t<kIsFP4, uint16_t, uint8_t>;

            // SF-major loop: when gran_k >= BLOCK_K, one packed int32 SF covers
            // kNumSFAStagesPerLoad K-blocks. Load SF into registers once per SF tile,
            // extract with compile-time byte index via cute::for_each.
            static constexpr bool kUseSFMajorLoop = (kGranKA >= BLOCK_K) and (kGranKB >= BLOCK_K);
            static_assert(!kUseSFMajorLoop || kNumSFAStagesPerLoad == kNumSFBStagesPerLoad,
                "SF-major loop requires matching A/B SF tile sizes");
            static constexpr uint32_t kSFTileKBlocks = kUseSFMajorLoop ? kNumSFAStagesPerLoad : 1;

            if constexpr (kUseSFMajorLoop) {
            // SF-MAJOR PATH: gran_k >= BLOCK_K
            // Load SF packed int32 into registers once per kSFTileKBlocks K-blocks,
            // extract bytes with compile-time index via cute::for_each.
            // SwizzleContext hoisted outside K-block loop (loop-invariant).
            uint32_t sf_packed_a[kMTilesPerWarp];
            uint32_t sf_packed_b[kNTilesPerWarp];
            const uint32_t num_full_sf_tiles = num_k_blocks / kSFTileKBlocks;
            const uint32_t kb_tail_start = num_full_sf_tiles * kSFTileKBlocks;

            sm120::SwizzleContext<kSwizzleAMode> a_ctx[kMTilesPerWarp];
            #pragma unroll
            for (uint32_t mt = 0; mt < kMTilesPerWarp; ++mt) {
                int a_row = (lane_idx & 7) + ((lane_idx >> 3) & 1) * 8 + (m_tile_base + mt) * 16;
                a_ctx[mt].init(a_row, kSMEMKBytes);
            }
            sm120::SwizzleContext<kSwizzleBMode> b_ctx[kNTilesPerWarp];
            #pragma unroll
            for (uint32_t nt = 0; nt < kNTilesPerWarp; ++nt) {
                int b_row = (lane_idx & 7) + (n_tile_base + nt) * 8;
                b_ctx[nt].init(b_row, kSMEMKBytes);
            }

            // Main body: compile-time unrolled K-blocks within each SF tile
            for (uint32_t sf_tile = 0; sf_tile < num_full_sf_tiles; ++sf_tile) {
            cute::for_each(cute::make_int_sequence<kSFTileKBlocks>{}, [&](auto kb_inner_ic) {
                constexpr uint32_t kb_inner = kb_inner_ic;
                CUTE_TIE_DECL(get_pipeline(iter_idx++), stage, phase);
                full_barriers[stage]->wait(phase);

                const uint32_t kb = sf_tile * kSFTileKBlocks + kb_inner;

                if constexpr (kUsePerNTileX4) {
                    uint32_t b_nt[kNTilesPerWarp][4];
                    uint32_t a_frag[2][kMTilesPerWarp][4];
                    sf_t sfb_hoisted[kNTilesPerWarp];
                    sf_t sfa_hoisted[kMTilesPerWarp];

                    if (kb_inner == 0) {
                        #pragma unroll
                        for (uint32_t nt = 0; nt < kNTilesPerWarp; ++nt)
                            sf_packed_b[nt] = sm120::load_sf(smem_sfb[stage], (n_tile_base + nt) * MMA_N + group_id);
                        #pragma unroll
                        for (uint32_t mt = 0; mt < kMTilesPerWarp; ++mt)
                            sf_packed_a[mt] = sm120::load_sf(smem_sfa[stage],
                                (m_tile_base + mt) * MMA_M + group_id + (thread_id & 1) * 8);
                    }

                    // Compile-time byte index: maps kb_inner to the correct byte within packed SF.
                    // For split-K: k_per_split must be aligned to kSFTileKBlocks so
                    // each partition starts at an SF tile boundary.
                    constexpr uint32_t sf_byte_a = (kb_inner * BLOCK_K / kGranKA) % 4;
                    constexpr uint32_t sf_byte_b = (kb_inner * BLOCK_K / kGranKB) % 4;

                    #pragma unroll
                    for (uint32_t nt = 0; nt < kNTilesPerWarp; ++nt) {
                        if constexpr (kIsFP4) {
                            uint8_t b = sm120_mma::extract_sf_byte(sf_packed_b[nt], sf_byte_b);
                            sfb_hoisted[nt] = static_cast<uint16_t>(b) | (static_cast<uint16_t>(b) << 8);
                        } else {
                            sfb_hoisted[nt] = sm120_mma::extract_sf_byte(sf_packed_b[nt], sf_byte_b);
                        }
                    }
                    #pragma unroll
                    for (uint32_t mt = 0; mt < kMTilesPerWarp; ++mt) {
                        if constexpr (kIsFP4) {
                            uint8_t b = sm120_mma::extract_sf_byte(sf_packed_a[mt], sf_byte_a);
                            sfa_hoisted[mt] = static_cast<uint16_t>(b) | (static_cast<uint16_t>(b) << 8);
                        } else {
                            sfa_hoisted[mt] = sm120_mma::extract_sf_byte(sf_packed_a[mt], sf_byte_a);
                        }
                    }

                    static constexpr uint32_t kKStepPairs = kKSteps / 2;
                    #pragma unroll
                    for (uint32_t kp = 0; kp < kKStepPairs; ++kp) {
                        const uint32_t ks_base = kp * 2;

                        #pragma unroll
                        for (uint32_t mt = 0; mt < kMTilesPerWarp; ++mt)
                            sm120::load_a_fragment(a_frag[0][mt], smem_a[stage], a_ctx[mt], lane_idx, ks_base, kLdmK);

                        #pragma unroll
                        for (uint32_t nt = 0; nt < kNTilesPerWarp; ++nt)
                            sm120::load_b_per_ntile_x4(b_nt[nt], smem_b[stage], b_ctx[nt], lane_idx, kp, kLdmK * 2);

                        #pragma unroll
                        for (uint32_t mt = 0; mt < kMTilesPerWarp; ++mt)
                            sm120::load_a_fragment(a_frag[1][mt], smem_a[stage], a_ctx[mt], lane_idx, ks_base + 1, kLdmK);

                        #pragma unroll
                        for (uint32_t mt = 0; mt < kMTilesPerWarp; ++mt) {
                            #pragma unroll
                            for (uint32_t nt = 0; nt < kNTilesPerWarp; ++nt) {
                                float (&d)[4] = *reinterpret_cast<float(*)[4]>(&accum[(mt * kNTilesPerWarp + nt) * MMA_ACCUM]);
                                if constexpr (kIsFP4)
                                    sm120_mma::fp4_mma_block_scaled(d, a_frag[0][mt], b_nt[nt][0], b_nt[nt][1], sfa_hoisted[mt], sfb_hoisted[nt]);
                                else
                                    sm120_mma::fp8_mma_block_scaled(d, a_frag[0][mt], b_nt[nt][0], b_nt[nt][1], sfa_hoisted[mt], sfb_hoisted[nt]);
                            }
                        }

                        #pragma unroll
                        for (uint32_t mt = 0; mt < kMTilesPerWarp; ++mt) {
                            #pragma unroll
                            for (uint32_t nt = 0; nt < kNTilesPerWarp; ++nt) {
                                float (&d)[4] = *reinterpret_cast<float(*)[4]>(&accum[(mt * kNTilesPerWarp + nt) * MMA_ACCUM]);
                                if constexpr (kIsFP4)
                                    sm120_mma::fp4_mma_block_scaled(d, a_frag[1][mt], b_nt[nt][2], b_nt[nt][3], sfa_hoisted[mt], sfb_hoisted[nt]);
                                else
                                    sm120_mma::fp8_mma_block_scaled(d, a_frag[1][mt], b_nt[nt][2], b_nt[nt][3], sfa_hoisted[mt], sfb_hoisted[nt]);
                            }
                        }
                    }
                } else {
                    // Fallback path for non-SF-major (MN-major B, mixed FP8×FP4) — unchanged
                    const uint32_t sf_byte_a_base = ((sf_tile * kSFTileKBlocks + kb_inner) * BLOCK_K / kGranKA) % 4;
                    const uint32_t sf_byte_b_base = ((sf_tile * kSFTileKBlocks + kb_inner) * BLOCK_K / kGranKB) % 4;
                    sm120::SwizzleContext<kSwizzleBMode> b_ctx[kBKMajor ? kNTilesPerWarp : 1];
                    if constexpr (kBKMajor) {
                        #pragma unroll
                        for (uint32_t nt = 0; nt < kNTilesPerWarp; ++nt) {
                            int b_row = (lane_idx & 7) + (n_tile_base + nt) * 8;
                            b_ctx[nt].init(b_row, kSMEMKBytes);
                        }
                    }
                    uint32_t a_frag[2][kMTilesPerWarp][4];
                    uint32_t b_tile[2][kNTilesPerWarp][2];
                    sf_t sfa_bytes[2][kMTilesPerWarp];
                    sf_t sfb_bytes[2][kNTilesPerWarp];
                    sf_t sfa_hoisted[kMTilesPerWarp];
                    sf_t sfb_hoisted[kNTilesPerWarp];

                    if constexpr (kGranKB >= BLOCK_K) {
                        #pragma unroll
                        for (uint32_t nt = 0; nt < kNTilesPerWarp; ++nt) {
                            auto packed = sm120::load_sf(smem_sfb[stage], (n_tile_base + nt) * MMA_N + group_id);
                            if constexpr (kIsFP4) {
                                uint8_t b = sm120_mma::extract_sf_byte(packed, sf_byte_b_base);
                                sfb_hoisted[nt] = static_cast<uint16_t>(b) | (static_cast<uint16_t>(b) << 8);
                            } else {
                                sfb_hoisted[nt] = sm120_mma::extract_sf_byte(packed, sf_byte_b_base);
                            }
                        }
                    }
                    if constexpr (kGranKA >= BLOCK_K) {
                        #pragma unroll
                        for (uint32_t mt = 0; mt < kMTilesPerWarp; ++mt) {
                            auto packed = sm120::load_sf(smem_sfa[stage],
                                (m_tile_base + mt) * MMA_M + group_id + (thread_id & 1) * 8);
                            if constexpr (kIsFP4) {
                                uint8_t b = sm120_mma::extract_sf_byte(packed, sf_byte_a_base);
                                sfa_hoisted[mt] = static_cast<uint16_t>(b) | (static_cast<uint16_t>(b) << 8);
                            } else {
                                sfa_hoisted[mt] = sm120_mma::extract_sf_byte(packed, sf_byte_a_base);
                            }
                        }
                    }

                    auto load_kstep = [&](int buf, uint32_t ks) {
                        if constexpr (kBKMajor) {
                            if constexpr (kBIsFP4) {
                                #pragma unroll
                                for (uint32_t nt = 0; nt < kNTilesPerWarp; ++nt) {
                                    sm120::load_b_fragment_b4x16_p64(b_tile[buf][nt], smem_b[stage], b_ctx[nt], lane_idx, ks, kLdmK);
                                    b_tile[buf][nt][0] <<= 2;
                                    b_tile[buf][nt][1] <<= 2;
                                }
                            } else {
                                #pragma unroll
                                for (uint32_t nt = 0; nt < kNTilesPerWarp; ++nt)
                                    sm120::load_b_fragment_x2(b_tile[buf][nt], smem_b[stage], b_ctx[nt], lane_idx, ks, kLdmK);
                            }
                        } else {
                            static constexpr uint32_t kBSwizzleB = kSwizzleBMode > 0 ? (__builtin_ctz(kSwizzleBMode) - 4) : 0;
                            static constexpr uint32_t kBSwizzleMask = kSwizzleBMode > 0 ? ((1u << kBSwizzleB) - 1) : 0;
                            static constexpr uint32_t kBSwizzleRowShift = kSwizzleBMode > 0 ? (7 - __builtin_ctz(BLOCK_N)) : 0;
                            #pragma unroll
                            for (uint32_t nt = 0; nt < kNTilesPerWarp; ++nt) {
                                const uint32_t n_col = (n_tile_base + nt) * MMA_N + group_id;
                                uint8_t v[8];
                                #pragma unroll
                                for (uint32_t i = 0; i < 4; ++i) {
                                    const uint32_t k = ks * MMA_K + thread_id * 4 + i;
                                    const uint32_t xor_bits = kSwizzleBMode > 0
                                        ? (((k >> kBSwizzleRowShift) & kBSwizzleMask) << 4) : 0;
                                    v[i] = static_cast<uint8_t>(smem_b[stage][k * BLOCK_N + (n_col ^ xor_bits)]);
                                }
                                #pragma unroll
                                for (uint32_t i = 0; i < 4; ++i) {
                                    const uint32_t k = ks * MMA_K + 16 + thread_id * 4 + i;
                                    const uint32_t xor_bits = kSwizzleBMode > 0
                                        ? (((k >> kBSwizzleRowShift) & kBSwizzleMask) << 4) : 0;
                                    v[4+i] = static_cast<uint8_t>(smem_b[stage][k * BLOCK_N + (n_col ^ xor_bits)]);
                                }
                                b_tile[buf][nt][0] = v[0] | (uint32_t(v[1]) << 8) | (uint32_t(v[2]) << 16) | (uint32_t(v[3]) << 24);
                                b_tile[buf][nt][1] = v[4] | (uint32_t(v[5]) << 8) | (uint32_t(v[6]) << 16) | (uint32_t(v[7]) << 24);
                            }
                        }
                        #pragma unroll
                        for (uint32_t mt = 0; mt < kMTilesPerWarp; ++mt)
                            if constexpr (kAIsFP4) {
                                sm120::load_a_fragment_b4x16(a_frag[buf][mt], smem_a[stage], a_ctx[mt], lane_idx, ks, kLdmK);
                                a_frag[buf][mt][0] <<= 2; a_frag[buf][mt][1] <<= 2;
                                a_frag[buf][mt][2] <<= 2; a_frag[buf][mt][3] <<= 2;
                            } else {
                                sm120::load_a_fragment(a_frag[buf][mt], smem_a[stage], a_ctx[mt], lane_idx, ks, kLdmK);
                            }

                        if constexpr (kGranKA < BLOCK_K or kGranKB < BLOCK_K) {
                            const uint32_t sf_step = (kb * kKSteps + ks);
                            if constexpr (kIsFP4) {
                                const uint32_t sf_byte_a = (sf_step * MMA_K / kGranKA) % 4;
                                const uint32_t sf_byte_b = (sf_step * MMA_K / kGranKB) % 4;
                                #pragma unroll
                                for (uint32_t nt = 0; nt < kNTilesPerWarp; ++nt) {
                                    auto packed = sm120::load_sf(smem_sfb[stage], (n_tile_base + nt) * MMA_N + group_id);
                                    if constexpr (kGranKB <= 32)
                                        sfb_bytes[buf][nt] = sm120_mma::extract_sf_pair(packed, sf_byte_b);
                                    else {
                                        uint8_t b = sm120_mma::extract_sf_byte(packed, sf_byte_b);
                                        sfb_bytes[buf][nt] = static_cast<uint16_t>(b) | (static_cast<uint16_t>(b) << 8);
                                    }
                                }
                                #pragma unroll
                                for (uint32_t mt = 0; mt < kMTilesPerWarp; ++mt) {
                                    auto packed = sm120::load_sf(smem_sfa[stage],
                                        (m_tile_base + mt) * MMA_M + group_id + (thread_id & 1) * 8);
                                    if constexpr (kGranKA <= 32)
                                        sfa_bytes[buf][mt] = sm120_mma::extract_sf_pair(packed, sf_byte_a);
                                    else {
                                        uint8_t b = sm120_mma::extract_sf_byte(packed, sf_byte_a);
                                        sfa_bytes[buf][mt] = static_cast<uint16_t>(b) | (static_cast<uint16_t>(b) << 8);
                                    }
                                }
                            } else {
                                const uint32_t sf_byte_a = (sf_step * MMA_K / kGranKA) % 4;
                                const uint32_t sf_byte_b = (sf_step * MMA_K / kGranKB) % 4;
                                #pragma unroll
                                for (uint32_t nt = 0; nt < kNTilesPerWarp; ++nt)
                                    sfb_bytes[buf][nt] = sm120_mma::extract_sf_byte(
                                        sm120::load_sf(smem_sfb[stage], (n_tile_base + nt) * MMA_N + group_id), sf_byte_b);
                                #pragma unroll
                                for (uint32_t mt = 0; mt < kMTilesPerWarp; ++mt)
                                    sfa_bytes[buf][mt] = sm120_mma::extract_sf_byte(
                                        sm120::load_sf(smem_sfa[stage],
                                            (m_tile_base + mt) * MMA_M + group_id + (thread_id & 1) * 8), sf_byte_a);
                            }
                        }
                    };

                    auto compute_kstep = [&](int buf) {
                        #pragma unroll
                        for (uint32_t mt = 0; mt < kMTilesPerWarp; ++mt) {
                            const sf_t sfa = (kGranKA >= BLOCK_K) ? sfa_hoisted[mt] : sfa_bytes[buf][mt];
                            #pragma unroll
                            for (uint32_t nt = 0; nt < kNTilesPerWarp; ++nt) {
                                float (&d)[4] = *reinterpret_cast<float(*)[4]>(&accum[(mt * kNTilesPerWarp + nt) * MMA_ACCUM]);
                                const sf_t sfb = (kGranKB >= BLOCK_K) ? sfb_hoisted[nt] : sfb_bytes[buf][nt];
                                if constexpr (kAIsFP4)
                                    sm120_mma::fp4_fp8_mixed_mma_block_scaled(d, a_frag[buf][mt], b_tile[buf][nt], sfa, sfb);
                                else if constexpr (kBIsFP4)
                                    sm120_mma::fp8_fp4_mixed_mma_block_scaled(d, a_frag[buf][mt], b_tile[buf][nt], sfa, sfb);
                                else if constexpr (kIsFP4)
                                    sm120_mma::fp4_mma_block_scaled(d, a_frag[buf][mt], b_tile[buf][nt], sfa, sfb);
                                else
                                    sm120_mma::fp8_mma_block_scaled(d, a_frag[buf][mt], b_tile[buf][nt], sfa, sfb);
                            }
                        }
                    };

                    load_kstep(0, 0);
                    #pragma unroll
                    for (uint32_t ks = 0; ks < kKSteps; ++ks) {
                        int cur = ks & 1;
                        int nxt = (ks + 1) & 1;
                        if (ks < kKSteps - 1)
                            load_kstep(nxt, ks + 1);
                        compute_kstep(cur);
                    }
                }

                // Release stage
                if (lane_idx == 0)
                    empty_barriers[stage]->arrive();
            }); // kb_inner (cute::for_each)
            } // sf_tile (SF-major main body)

            // SF-major tail: remaining K-blocks (0 to kSFTileKBlocks-1).
            // Since kUseSFMajorLoop implies kGranK >= BLOCK_K, SF hoisting is always valid.
            for (uint32_t kb = kb_tail_start; kb < num_k_blocks; ++kb) {
                CUTE_TIE_DECL(get_pipeline(iter_idx++), stage, phase);
                full_barriers[stage]->wait(phase);

                const uint32_t sf_byte_a_base = (kb * BLOCK_K / kGranKA) % 4;
                const uint32_t sf_byte_b_base = (kb * BLOCK_K / kGranKB) % 4;

                if constexpr (kUsePerNTileX4) {
                    uint32_t b_nt[kNTilesPerWarp][4];
                    uint32_t a_frag[2][kMTilesPerWarp][4];
                    sf_t sfb_hoisted[kNTilesPerWarp];
                    sf_t sfa_hoisted[kMTilesPerWarp];
                    sf_t sfb_step[kNTilesPerWarp];
                    sf_t sfa_step[2][kMTilesPerWarp];

                    if constexpr (kGranKB >= BLOCK_K) {
                        #pragma unroll
                        for (uint32_t nt = 0; nt < kNTilesPerWarp; ++nt) {
                            auto packed = sm120::load_sf(smem_sfb[stage], (n_tile_base + nt) * MMA_N + group_id);
                            if constexpr (kIsFP4) {
                                uint8_t b = sm120_mma::extract_sf_byte(packed, sf_byte_b_base);
                                sfb_hoisted[nt] = static_cast<uint16_t>(b) | (static_cast<uint16_t>(b) << 8);
                            } else {
                                sfb_hoisted[nt] = sm120_mma::extract_sf_byte(packed, sf_byte_b_base);
                            }
                        }
                    }
                    if constexpr (kGranKA >= BLOCK_K) {
                        #pragma unroll
                        for (uint32_t mt = 0; mt < kMTilesPerWarp; ++mt) {
                            auto packed = sm120::load_sf(smem_sfa[stage],
                                (m_tile_base + mt) * MMA_M + group_id + (thread_id & 1) * 8);
                            if constexpr (kIsFP4) {
                                uint8_t b = sm120_mma::extract_sf_byte(packed, sf_byte_a_base);
                                sfa_hoisted[mt] = static_cast<uint16_t>(b) | (static_cast<uint16_t>(b) << 8);
                            } else {
                                sfa_hoisted[mt] = sm120_mma::extract_sf_byte(packed, sf_byte_a_base);
                            }
                        }
                    }

                    static constexpr uint32_t kKStepPairs_tail = kKSteps / 2;
                    #pragma unroll
                    for (uint32_t kp = 0; kp < kKStepPairs_tail; ++kp) {
                        const uint32_t ks_base = kp * 2;
                        #pragma unroll
                        for (uint32_t mt = 0; mt < kMTilesPerWarp; ++mt)
                            sm120::load_a_fragment(a_frag[0][mt], smem_a[stage], a_ctx[mt], lane_idx, ks_base, kLdmK);
                        #pragma unroll
                        for (uint32_t nt = 0; nt < kNTilesPerWarp; ++nt)
                            sm120::load_b_per_ntile_x4(b_nt[nt], smem_b[stage], b_ctx[nt], lane_idx, kp, kLdmK * 2);

                        auto load_sf_for_step_tail = [&](uint32_t ks, int sf_buf) {
                            if constexpr (kGranKA < BLOCK_K or kGranKB < BLOCK_K) {
                                const uint32_t sf_step = kb * kKSteps + ks;
                                if constexpr (kIsFP4) {
                                    const uint32_t sf_byte_b = (sf_step * MMA_K / kGranKB) % 4;
                                    const uint32_t sf_byte_a = (sf_step * MMA_K / kGranKA) % 4;
                                    if constexpr (kGranKB < BLOCK_K) {
                                        #pragma unroll
                                        for (uint32_t nt = 0; nt < kNTilesPerWarp; ++nt) {
                                            auto packed = sm120::load_sf(smem_sfb[stage], (n_tile_base + nt) * MMA_N + group_id);
                                            if constexpr (kGranKB <= 32)
                                                sfb_step[nt] = sm120_mma::extract_sf_pair(packed, sf_byte_b);
                                            else {
                                                uint8_t b = sm120_mma::extract_sf_byte(packed, sf_byte_b);
                                                sfb_step[nt] = static_cast<uint16_t>(b) | (static_cast<uint16_t>(b) << 8);
                                            }
                                        }
                                    }
                                    if constexpr (kGranKA < BLOCK_K) {
                                        #pragma unroll
                                        for (uint32_t mt = 0; mt < kMTilesPerWarp; ++mt) {
                                            auto packed = sm120::load_sf(smem_sfa[stage],
                                                (m_tile_base + mt) * MMA_M + group_id + (thread_id & 1) * 8);
                                            if constexpr (kGranKA <= 32)
                                                sfa_step[sf_buf][mt] = sm120_mma::extract_sf_pair(packed, sf_byte_a);
                                            else {
                                                uint8_t b = sm120_mma::extract_sf_byte(packed, sf_byte_a);
                                                sfa_step[sf_buf][mt] = static_cast<uint16_t>(b) | (static_cast<uint16_t>(b) << 8);
                                            }
                                        }
                                    }
                                } else {
                                    const uint32_t sf_byte_b = (sf_step * MMA_K / kGranKB) % 4;
                                    const uint32_t sf_byte_a = (sf_step * MMA_K / kGranKA) % 4;
                                    if constexpr (kGranKB < BLOCK_K) {
                                        #pragma unroll
                                        for (uint32_t nt = 0; nt < kNTilesPerWarp; ++nt)
                                            sfb_step[nt] = sm120_mma::extract_sf_byte(
                                                sm120::load_sf(smem_sfb[stage], (n_tile_base + nt) * MMA_N + group_id), sf_byte_b);
                                    }
                                    if constexpr (kGranKA < BLOCK_K) {
                                        #pragma unroll
                                        for (uint32_t mt = 0; mt < kMTilesPerWarp; ++mt)
                                            sfa_step[sf_buf][mt] = sm120_mma::extract_sf_byte(
                                                sm120::load_sf(smem_sfa[stage],
                                                    (m_tile_base + mt) * MMA_M + group_id + (thread_id & 1) * 8), sf_byte_a);
                                    }
                                }
                            }
                        };

                        load_sf_for_step_tail(ks_base, 0);

                        #pragma unroll
                        for (uint32_t mt = 0; mt < kMTilesPerWarp; ++mt)
                            sm120::load_a_fragment(a_frag[1][mt], smem_a[stage], a_ctx[mt], lane_idx, ks_base + 1, kLdmK);

                        #pragma unroll
                        for (uint32_t mt = 0; mt < kMTilesPerWarp; ++mt) {
                            const sf_t sfa0 = (kGranKA >= BLOCK_K) ? sfa_hoisted[mt] : sfa_step[0][mt];
                            #pragma unroll
                            for (uint32_t nt = 0; nt < kNTilesPerWarp; ++nt) {
                                float (&d)[4] = *reinterpret_cast<float(*)[4]>(&accum[(mt * kNTilesPerWarp + nt) * MMA_ACCUM]);
                                const sf_t sfb = (kGranKB >= BLOCK_K) ? sfb_hoisted[nt] : sfb_step[nt];
                                if constexpr (kIsFP4)
                                    sm120_mma::fp4_mma_block_scaled(d, a_frag[0][mt], b_nt[nt][0], b_nt[nt][1], sfa0, sfb);
                                else
                                    sm120_mma::fp8_mma_block_scaled(d, a_frag[0][mt], b_nt[nt][0], b_nt[nt][1], sfa0, sfb);
                            }
                        }

                        load_sf_for_step_tail(ks_base + 1, 1);

                        #pragma unroll
                        for (uint32_t mt = 0; mt < kMTilesPerWarp; ++mt) {
                            const sf_t sfa1 = (kGranKA >= BLOCK_K) ? sfa_hoisted[mt] : sfa_step[1][mt];
                            #pragma unroll
                            for (uint32_t nt = 0; nt < kNTilesPerWarp; ++nt) {
                                float (&d)[4] = *reinterpret_cast<float(*)[4]>(&accum[(mt * kNTilesPerWarp + nt) * MMA_ACCUM]);
                                const sf_t sfb = (kGranKB >= BLOCK_K) ? sfb_hoisted[nt] : sfb_step[nt];
                                if constexpr (kIsFP4)
                                    sm120_mma::fp4_mma_block_scaled(d, a_frag[1][mt], b_nt[nt][2], b_nt[nt][3], sfa1, sfb);
                                else
                                    sm120_mma::fp8_mma_block_scaled(d, a_frag[1][mt], b_nt[nt][2], b_nt[nt][3], sfa1, sfb);
                            }
                        }
                    }
                }

                if (lane_idx == 0)
                    empty_barriers[stage]->arrive();
            } // SF-major tail kb loop

            } else { // !kUseSFMajorLoop
            // ORIGINAL PATH: gran_k < BLOCK_K (per-K-step SF loading)
            // Flat K-block loop with runtime sf_byte, no SF caching.
            for (uint32_t kb = 0; kb < num_k_blocks; ++kb) {
                CUTE_TIE_DECL(get_pipeline(iter_idx++), stage, phase);

                full_barriers[stage]->wait(phase);

                sm120::SwizzleContext<kSwizzleAMode> a_ctx[kMTilesPerWarp];
                #pragma unroll
                for (uint32_t mt = 0; mt < kMTilesPerWarp; ++mt) {
                    int a_row = (lane_idx & 7) + ((lane_idx >> 3) & 1) * 8 + (m_tile_base + mt) * 16;
                    a_ctx[mt].init(a_row, kSMEMKBytes);
                }

                const uint32_t sf_byte_a_base = (kb * BLOCK_K / kGranKA) % 4;
                const uint32_t sf_byte_b_base = (kb * BLOCK_K / kGranKB) % 4;

                if constexpr (kUsePerNTileX4) {
                    sm120::SwizzleContext<kSwizzleBMode> b_ctx[kNTilesPerWarp];
                    #pragma unroll
                    for (uint32_t nt = 0; nt < kNTilesPerWarp; ++nt) {
                        int b_row = (lane_idx & 7) + (n_tile_base + nt) * 8;
                        b_ctx[nt].init(b_row, kSMEMKBytes);
                    }

                    uint32_t b_nt[kNTilesPerWarp][4];
                    uint32_t a_frag[2][kMTilesPerWarp][4];
                    sf_t sfb_hoisted[kNTilesPerWarp];
                    sf_t sfa_hoisted[kMTilesPerWarp];
                    sf_t sfb_step[kNTilesPerWarp];
                    sf_t sfa_step[2][kMTilesPerWarp];

                    if constexpr (kGranKB >= BLOCK_K) {
                        #pragma unroll
                        for (uint32_t nt = 0; nt < kNTilesPerWarp; ++nt) {
                            auto packed = sm120::load_sf(smem_sfb[stage], (n_tile_base + nt) * MMA_N + group_id);
                            if constexpr (kIsFP4) {
                                uint8_t b = sm120_mma::extract_sf_byte(packed, sf_byte_b_base);
                                sfb_hoisted[nt] = static_cast<uint16_t>(b) | (static_cast<uint16_t>(b) << 8);
                            } else {
                                sfb_hoisted[nt] = sm120_mma::extract_sf_byte(packed, sf_byte_b_base);
                            }
                        }
                    }
                    if constexpr (kGranKA >= BLOCK_K) {
                        #pragma unroll
                        for (uint32_t mt = 0; mt < kMTilesPerWarp; ++mt) {
                            auto packed = sm120::load_sf(smem_sfa[stage],
                                (m_tile_base + mt) * MMA_M + group_id + (thread_id & 1) * 8);
                            if constexpr (kIsFP4) {
                                uint8_t b = sm120_mma::extract_sf_byte(packed, sf_byte_a_base);
                                sfa_hoisted[mt] = static_cast<uint16_t>(b) | (static_cast<uint16_t>(b) << 8);
                            } else {
                                sfa_hoisted[mt] = sm120_mma::extract_sf_byte(packed, sf_byte_a_base);
                            }
                        }
                    }

                    static constexpr uint32_t kKStepPairs = kKSteps / 2;
                    #pragma unroll
                    for (uint32_t kp = 0; kp < kKStepPairs; ++kp) {
                        const uint32_t ks_base = kp * 2;

                        #pragma unroll
                        for (uint32_t mt = 0; mt < kMTilesPerWarp; ++mt)
                            sm120::load_a_fragment(a_frag[0][mt], smem_a[stage], a_ctx[mt], lane_idx, ks_base, kLdmK);

                        #pragma unroll
                        for (uint32_t nt = 0; nt < kNTilesPerWarp; ++nt)
                            sm120::load_b_per_ntile_x4(b_nt[nt], smem_b[stage], b_ctx[nt], lane_idx, kp, kLdmK * 2);

                        auto load_sf_for_step = [&](uint32_t ks, int sf_buf) {
                            if constexpr (kGranKA < BLOCK_K or kGranKB < BLOCK_K) {
                                const uint32_t sf_step = kb * kKSteps + ks;
                                if constexpr (kIsFP4) {
                                    const uint32_t sf_byte_b = (sf_step * MMA_K / kGranKB) % 4;
                                    const uint32_t sf_byte_a = (sf_step * MMA_K / kGranKA) % 4;
                                    if constexpr (kGranKB < BLOCK_K) {
                                        #pragma unroll
                                        for (uint32_t nt = 0; nt < kNTilesPerWarp; ++nt) {
                                            auto packed = sm120::load_sf(smem_sfb[stage], (n_tile_base + nt) * MMA_N + group_id);
                                            if constexpr (kGranKB <= 32)
                                                sfb_step[nt] = sm120_mma::extract_sf_pair(packed, sf_byte_b);
                                            else {
                                                uint8_t b = sm120_mma::extract_sf_byte(packed, sf_byte_b);
                                                sfb_step[nt] = static_cast<uint16_t>(b) | (static_cast<uint16_t>(b) << 8);
                                            }
                                        }
                                    }
                                    if constexpr (kGranKA < BLOCK_K) {
                                        #pragma unroll
                                        for (uint32_t mt = 0; mt < kMTilesPerWarp; ++mt) {
                                            auto packed = sm120::load_sf(smem_sfa[stage],
                                                (m_tile_base + mt) * MMA_M + group_id + (thread_id & 1) * 8);
                                            if constexpr (kGranKA <= 32)
                                                sfa_step[sf_buf][mt] = sm120_mma::extract_sf_pair(packed, sf_byte_a);
                                            else {
                                                uint8_t b = sm120_mma::extract_sf_byte(packed, sf_byte_a);
                                                sfa_step[sf_buf][mt] = static_cast<uint16_t>(b) | (static_cast<uint16_t>(b) << 8);
                                            }
                                        }
                                    }
                                } else {
                                    const uint32_t sf_byte_b = (sf_step * MMA_K / kGranKB) % 4;
                                    const uint32_t sf_byte_a = (sf_step * MMA_K / kGranKA) % 4;
                                    if constexpr (kGranKB < BLOCK_K) {
                                        #pragma unroll
                                        for (uint32_t nt = 0; nt < kNTilesPerWarp; ++nt)
                                            sfb_step[nt] = sm120_mma::extract_sf_byte(
                                                sm120::load_sf(smem_sfb[stage], (n_tile_base + nt) * MMA_N + group_id), sf_byte_b);
                                    }
                                    if constexpr (kGranKA < BLOCK_K) {
                                        #pragma unroll
                                        for (uint32_t mt = 0; mt < kMTilesPerWarp; ++mt)
                                            sfa_step[sf_buf][mt] = sm120_mma::extract_sf_byte(
                                                sm120::load_sf(smem_sfa[stage],
                                                    (m_tile_base + mt) * MMA_M + group_id + (thread_id & 1) * 8), sf_byte_a);
                                    }
                                }
                            }
                        };

                        load_sf_for_step(ks_base, 0);

                        #pragma unroll
                        for (uint32_t mt = 0; mt < kMTilesPerWarp; ++mt)
                            sm120::load_a_fragment(a_frag[1][mt], smem_a[stage], a_ctx[mt], lane_idx, ks_base + 1, kLdmK);

                        #pragma unroll
                        for (uint32_t mt = 0; mt < kMTilesPerWarp; ++mt) {
                            const sf_t sfa0 = (kGranKA >= BLOCK_K) ? sfa_hoisted[mt] : sfa_step[0][mt];
                            #pragma unroll
                            for (uint32_t nt = 0; nt < kNTilesPerWarp; ++nt) {
                                float (&d)[4] = *reinterpret_cast<float(*)[4]>(&accum[(mt * kNTilesPerWarp + nt) * MMA_ACCUM]);
                                const sf_t sfb = (kGranKB >= BLOCK_K) ? sfb_hoisted[nt] : sfb_step[nt];
                                if constexpr (kIsFP4)
                                    sm120_mma::fp4_mma_block_scaled(d, a_frag[0][mt], b_nt[nt][0], b_nt[nt][1], sfa0, sfb);
                                else
                                    sm120_mma::fp8_mma_block_scaled(d, a_frag[0][mt], b_nt[nt][0], b_nt[nt][1], sfa0, sfb);
                            }
                        }

                        load_sf_for_step(ks_base + 1, 1);

                        #pragma unroll
                        for (uint32_t mt = 0; mt < kMTilesPerWarp; ++mt) {
                            const sf_t sfa1 = (kGranKA >= BLOCK_K) ? sfa_hoisted[mt] : sfa_step[1][mt];
                            #pragma unroll
                            for (uint32_t nt = 0; nt < kNTilesPerWarp; ++nt) {
                                float (&d)[4] = *reinterpret_cast<float(*)[4]>(&accum[(mt * kNTilesPerWarp + nt) * MMA_ACCUM]);
                                const sf_t sfb = (kGranKB >= BLOCK_K) ? sfb_hoisted[nt] : sfb_step[nt];
                                if constexpr (kIsFP4)
                                    sm120_mma::fp4_mma_block_scaled(d, a_frag[1][mt], b_nt[nt][2], b_nt[nt][3], sfa1, sfb);
                                else
                                    sm120_mma::fp8_mma_block_scaled(d, a_frag[1][mt], b_nt[nt][2], b_nt[nt][3], sfa1, sfb);
                            }
                        }
                    }
                } else {
                    // Fallback: original K-step double-buffer (MN-major B, mixed FP8×FP4)
                    sm120::SwizzleContext<kSwizzleBMode> b_ctx[kBKMajor ? kNTilesPerWarp : 1];
                    if constexpr (kBKMajor) {
                        #pragma unroll
                        for (uint32_t nt = 0; nt < kNTilesPerWarp; ++nt) {
                            int b_row = (lane_idx & 7) + (n_tile_base + nt) * 8;
                            b_ctx[nt].init(b_row, kSMEMKBytes);
                        }
                    }
                    uint32_t a_frag[2][kMTilesPerWarp][4];
                    uint32_t b_tile[2][kNTilesPerWarp][2];
                    sf_t sfa_bytes[2][kMTilesPerWarp];
                    sf_t sfb_bytes[2][kNTilesPerWarp];
                    sf_t sfa_hoisted[kMTilesPerWarp];
                    sf_t sfb_hoisted[kNTilesPerWarp];

                    if constexpr (kGranKB >= BLOCK_K) {
                        #pragma unroll
                        for (uint32_t nt = 0; nt < kNTilesPerWarp; ++nt) {
                            auto packed = sm120::load_sf(smem_sfb[stage], (n_tile_base + nt) * MMA_N + group_id);
                            if constexpr (kIsFP4) {
                                uint8_t b = sm120_mma::extract_sf_byte(packed, sf_byte_b_base);
                                sfb_hoisted[nt] = static_cast<uint16_t>(b) | (static_cast<uint16_t>(b) << 8);
                            } else {
                                sfb_hoisted[nt] = sm120_mma::extract_sf_byte(packed, sf_byte_b_base);
                            }
                        }
                    }
                    if constexpr (kGranKA >= BLOCK_K) {
                        #pragma unroll
                        for (uint32_t mt = 0; mt < kMTilesPerWarp; ++mt) {
                            auto packed = sm120::load_sf(smem_sfa[stage],
                                (m_tile_base + mt) * MMA_M + group_id + (thread_id & 1) * 8);
                            if constexpr (kIsFP4) {
                                uint8_t b = sm120_mma::extract_sf_byte(packed, sf_byte_a_base);
                                sfa_hoisted[mt] = static_cast<uint16_t>(b) | (static_cast<uint16_t>(b) << 8);
                            } else {
                                sfa_hoisted[mt] = sm120_mma::extract_sf_byte(packed, sf_byte_a_base);
                            }
                        }
                    }

                    auto load_kstep = [&](int buf, uint32_t ks) {
                        if constexpr (kBKMajor) {
                            if constexpr (kBIsFP4) {
                                #pragma unroll
                                for (uint32_t nt = 0; nt < kNTilesPerWarp; ++nt) {
                                    sm120::load_b_fragment_b4x16_p64(b_tile[buf][nt], smem_b[stage], b_ctx[nt], lane_idx, ks, kLdmK);
                                    b_tile[buf][nt][0] <<= 2;
                                    b_tile[buf][nt][1] <<= 2;
                                }
                            } else {
                                #pragma unroll
                                for (uint32_t nt = 0; nt < kNTilesPerWarp; ++nt)
                                    sm120::load_b_fragment_x2(b_tile[buf][nt], smem_b[stage], b_ctx[nt], lane_idx, ks, kLdmK);
                            }
                        } else {
                            static constexpr uint32_t kBSwizzleB = kSwizzleBMode > 0 ? (__builtin_ctz(kSwizzleBMode) - 4) : 0;
                            static constexpr uint32_t kBSwizzleMask = kSwizzleBMode > 0 ? ((1u << kBSwizzleB) - 1) : 0;
                            static constexpr uint32_t kBSwizzleRowShift = kSwizzleBMode > 0 ? (7 - __builtin_ctz(BLOCK_N)) : 0;
                            #pragma unroll
                            for (uint32_t nt = 0; nt < kNTilesPerWarp; ++nt) {
                                const uint32_t n_col = (n_tile_base + nt) * MMA_N + group_id;
                                uint8_t v[8];
                                #pragma unroll
                                for (uint32_t i = 0; i < 4; ++i) {
                                    const uint32_t k = ks * MMA_K + thread_id * 4 + i;
                                    const uint32_t xor_bits = kSwizzleBMode > 0
                                        ? (((k >> kBSwizzleRowShift) & kBSwizzleMask) << 4) : 0;
                                    v[i] = static_cast<uint8_t>(smem_b[stage][k * BLOCK_N + (n_col ^ xor_bits)]);
                                }
                                #pragma unroll
                                for (uint32_t i = 0; i < 4; ++i) {
                                    const uint32_t k = ks * MMA_K + 16 + thread_id * 4 + i;
                                    const uint32_t xor_bits = kSwizzleBMode > 0
                                        ? (((k >> kBSwizzleRowShift) & kBSwizzleMask) << 4) : 0;
                                    v[4+i] = static_cast<uint8_t>(smem_b[stage][k * BLOCK_N + (n_col ^ xor_bits)]);
                                }
                                b_tile[buf][nt][0] = v[0] | (uint32_t(v[1]) << 8) | (uint32_t(v[2]) << 16) | (uint32_t(v[3]) << 24);
                                b_tile[buf][nt][1] = v[4] | (uint32_t(v[5]) << 8) | (uint32_t(v[6]) << 16) | (uint32_t(v[7]) << 24);
                            }
                        }
                        #pragma unroll
                        for (uint32_t mt = 0; mt < kMTilesPerWarp; ++mt)
                            if constexpr (kAIsFP4) {
                                sm120::load_a_fragment_b4x16(a_frag[buf][mt], smem_a[stage], a_ctx[mt], lane_idx, ks, kLdmK);
                                a_frag[buf][mt][0] <<= 2; a_frag[buf][mt][1] <<= 2;
                                a_frag[buf][mt][2] <<= 2; a_frag[buf][mt][3] <<= 2;
                            } else {
                                sm120::load_a_fragment(a_frag[buf][mt], smem_a[stage], a_ctx[mt], lane_idx, ks, kLdmK);
                            }

                        if constexpr (kGranKA < BLOCK_K or kGranKB < BLOCK_K) {
                            const uint32_t sf_step = (kb * kKSteps + ks);
                            if constexpr (kIsFP4) {
                                const uint32_t sf_byte_a = (sf_step * MMA_K / kGranKA) % 4;
                                const uint32_t sf_byte_b = (sf_step * MMA_K / kGranKB) % 4;
                                #pragma unroll
                                for (uint32_t nt = 0; nt < kNTilesPerWarp; ++nt) {
                                    auto packed = sm120::load_sf(smem_sfb[stage], (n_tile_base + nt) * MMA_N + group_id);
                                    if constexpr (kGranKB <= 32)
                                        sfb_bytes[buf][nt] = sm120_mma::extract_sf_pair(packed, sf_byte_b);
                                    else {
                                        uint8_t b = sm120_mma::extract_sf_byte(packed, sf_byte_b);
                                        sfb_bytes[buf][nt] = static_cast<uint16_t>(b) | (static_cast<uint16_t>(b) << 8);
                                    }
                                }
                                #pragma unroll
                                for (uint32_t mt = 0; mt < kMTilesPerWarp; ++mt) {
                                    auto packed = sm120::load_sf(smem_sfa[stage],
                                        (m_tile_base + mt) * MMA_M + group_id + (thread_id & 1) * 8);
                                    if constexpr (kGranKA <= 32)
                                        sfa_bytes[buf][mt] = sm120_mma::extract_sf_pair(packed, sf_byte_a);
                                    else {
                                        uint8_t b = sm120_mma::extract_sf_byte(packed, sf_byte_a);
                                        sfa_bytes[buf][mt] = static_cast<uint16_t>(b) | (static_cast<uint16_t>(b) << 8);
                                    }
                                }
                            } else {
                                const uint32_t sf_byte_a = (sf_step * MMA_K / kGranKA) % 4;
                                const uint32_t sf_byte_b = (sf_step * MMA_K / kGranKB) % 4;
                                #pragma unroll
                                for (uint32_t nt = 0; nt < kNTilesPerWarp; ++nt)
                                    sfb_bytes[buf][nt] = sm120_mma::extract_sf_byte(
                                        sm120::load_sf(smem_sfb[stage], (n_tile_base + nt) * MMA_N + group_id), sf_byte_b);
                                #pragma unroll
                                for (uint32_t mt = 0; mt < kMTilesPerWarp; ++mt)
                                    sfa_bytes[buf][mt] = sm120_mma::extract_sf_byte(
                                        sm120::load_sf(smem_sfa[stage],
                                            (m_tile_base + mt) * MMA_M + group_id + (thread_id & 1) * 8), sf_byte_a);
                            }
                        }
                    };

                    auto compute_kstep = [&](int buf) {
                        #pragma unroll
                        for (uint32_t mt = 0; mt < kMTilesPerWarp; ++mt) {
                            const sf_t sfa = (kGranKA >= BLOCK_K) ? sfa_hoisted[mt] : sfa_bytes[buf][mt];
                            #pragma unroll
                            for (uint32_t nt = 0; nt < kNTilesPerWarp; ++nt) {
                                float (&d)[4] = *reinterpret_cast<float(*)[4]>(&accum[(mt * kNTilesPerWarp + nt) * MMA_ACCUM]);
                                const sf_t sfb = (kGranKB >= BLOCK_K) ? sfb_hoisted[nt] : sfb_bytes[buf][nt];
                                if constexpr (kAIsFP4)
                                    sm120_mma::fp4_fp8_mixed_mma_block_scaled(d, a_frag[buf][mt], b_tile[buf][nt], sfa, sfb);
                                else if constexpr (kBIsFP4)
                                    sm120_mma::fp8_fp4_mixed_mma_block_scaled(d, a_frag[buf][mt], b_tile[buf][nt], sfa, sfb);
                                else if constexpr (kIsFP4)
                                    sm120_mma::fp4_mma_block_scaled(d, a_frag[buf][mt], b_tile[buf][nt], sfa, sfb);
                                else
                                    sm120_mma::fp8_mma_block_scaled(d, a_frag[buf][mt], b_tile[buf][nt], sfa, sfb);
                            }
                        }
                    };

                    load_kstep(0, 0);
                    #pragma unroll
                    for (uint32_t ks = 0; ks < kKSteps; ++ks) {
                        int cur = ks & 1;
                        int nxt = (ks + 1) & 1;
                        if (ks < kKSteps - 1)
                            load_kstep(nxt, ks + 1);
                        compute_kstep(cur);
                    }
                }

                if (lane_idx == 0)
                    empty_barriers[stage]->arrive();
            }
            } // else (!kUseSFMajorLoop) — original path

            // Epilogue
            if constexpr (kSplitKFactor > 1) {
                // Split-K: write FP32 partials to workspace
                const uint32_t m_base_sk = m_block_idx * BLOCK_M;
                const uint32_t n_base_sk = n_block_idx * BLOCK_N;
                float* ws = gmem_workspace + static_cast<int64_t>(scheduler.split_k_idx) * shape_m * shape_n;

                #pragma unroll
                for (uint32_t mt = 0; mt < kMTilesPerWarp; ++mt) {
                    #pragma unroll
                    for (uint32_t nt = 0; nt < kNTilesPerWarp; ++nt) {
                        const uint32_t ai = (mt * kNTilesPerWarp + nt) * MMA_ACCUM;
                        const uint32_t col = n_base_sk + (n_tile_base + nt) * MMA_N + thread_id * 2;
                        const uint32_t row0 = m_base_sk + (m_tile_base + mt) * MMA_M + group_id;
                        const uint32_t row1 = row0 + 8;

                        if (row0 < shape_m) {
                            auto idx = static_cast<int64_t>(row0) * shape_n + col;
                            if (col < shape_n)     ws[idx]     = accum[ai + 0];
                            if (col + 1 < shape_n) ws[idx + 1] = accum[ai + 1];
                        }
                        if (row1 < shape_m) {
                            auto idx = static_cast<int64_t>(row1) * shape_n + col;
                            if (col < shape_n)     ws[idx]     = accum[ai + 2];
                            if (col + 1 < shape_n) ws[idx + 1] = accum[ai + 3];
                        }
                    }
                }
            } else {
            // Normal epilogue (non-split-K)
            constexpr bool kEpilogueGroupOffset = not is_m_grouped_contiguous(kGemmType);
            const uint32_t m_base = scheduler.template get_global_idx<kEpilogueGroupOffset>(shape_m, BLOCK_M, m_block_idx);
            const uint32_t n_base = n_block_idx * BLOCK_N;
            const uint32_t total_shape_m = (kGemmType == GemmType::KGroupedContiguous or kGemmType == GemmType::MGroupedMasked)
                ? shape_m * kNumGroups : shape_m;

            auto read_cd = [&](const cd_dtype_t& x) -> float {
                if constexpr (cute::is_same_v<cd_dtype_t, float>) return x;
                else return static_cast<float>(x);
            };

            constexpr bool kIsBatchedEpilogue = (kGemmType == GemmType::Batched);
            const int64_t cd_m_stride = static_cast<int64_t>(stride_cd_m);
            const int64_t cd_batch_offset = kIsBatchedEpilogue
                ? static_cast<int64_t>(scheduler.current_group_idx) * stride_cd_batch : 0;

            if constexpr (kUseTMAStoreEpilogue) {
                #pragma unroll
                for (uint32_t ms = 0; ms < kNumEpiMSubs; ++ms) {
                    const uint32_t epi_m_start = ms * kEpiSubM;

                    if (math_warp_idx == 0 and lane_idx == 0)
                        cute::tma_store_wait<0>();
                    cutlass::arch::NamedBarrier::sync(kNumMathThreads, 0);

                    #pragma unroll
                    for (uint32_t mt = 0; mt < kMTilesPerWarp; ++mt) {
                        const uint32_t local_row0 = (m_tile_base + mt) * MMA_M + group_id;
                        const uint32_t local_row1 = local_row0 + 8;
                        if (local_row0 >= epi_m_start and local_row0 < epi_m_start + kEpiSubM) {
                            const uint32_t sub_row0 = local_row0 - epi_m_start;
                            const uint32_t sub_row1 = sub_row0 + 8;
                            #pragma unroll
                            for (uint32_t nt = 0; nt < kNTilesPerWarp; ++nt) {
                                const uint32_t ai = (mt * kNTilesPerWarp + nt) * MMA_ACCUM;
                                const uint32_t local_col = (n_tile_base + nt) * MMA_N + thread_id * 2;
                                float v0 = accum[ai + 0], v1 = accum[ai + 1];
                                float v2 = accum[ai + 2], v3 = accum[ai + 3];

                                // Batched accumulation is handled by SM90_TMA_REDUCE_ADD_3D
                                // (adds SMEM to the existing global C). Reading gmem_c here
                                // too would double-count, so only the non-batched path (plain
                                // SM90_TMA_STORE_2D) reads and accumulates in registers.
                                // NOTE: relies on the invariant below that batched+accumulation
                                // ALWAYS uses SM90_TMA_REDUCE_ADD_3D. If that dispatch ever
                                // becomes a plain STORE, this skip would drop the accumulation.
                                if constexpr (kWithAccumulation and not kIsBatchedEpilogue) {
                                    const uint32_t gr0 = m_base + local_row0, gr1 = m_base + local_row1;
                                    const uint32_t gc = epilogue_type_t::template apply_index_n<MMA_N>(
                                        n_base + (n_tile_base + nt) * MMA_N) + thread_id * 2;
                                    if (gr0 < total_shape_m and gc + 1 < shape_n) {
                                        const auto ci = cd_batch_offset + static_cast<int64_t>(gr0) * cd_m_stride + gc;
                                        v0 += read_cd(gmem_c[ci]); v1 += read_cd(gmem_c[ci + 1]);
                                    }
                                    if (gr1 < total_shape_m and gc + 1 < shape_n) {
                                        const auto ci = cd_batch_offset + static_cast<int64_t>(gr1) * cd_m_stride + gc;
                                        v2 += read_cd(gmem_c[ci]); v3 += read_cd(gmem_c[ci + 1]);
                                    }
                                }

                                const uint32_t sub_tile = local_col / kTMAStoreInnerDim;
                                const uint32_t col_in_sub = local_col % kTMAStoreInnerDim;
                                const uint32_t col_byte_in_sub = col_in_sub * sizeof(cd_dtype_t);
                                const uint32_t sw0 = col_byte_in_sub ^ (((sub_row0 >> kSwizzleCDShift) & kSwizzleCDMask) << 4);
                                const uint32_t sw1 = col_byte_in_sub ^ (((sub_row1 >> kSwizzleCDShift) & kSwizzleCDMask) << 4);
                                cd_dtype_t p0[2] = {cd_dtype_t(v0), cd_dtype_t(v1)};
                                cd_dtype_t p1[2] = {cd_dtype_t(v2), cd_dtype_t(v3)};
                                auto* smem_d_bytes = reinterpret_cast<char*>(smem_d_base);
                                const uint32_t sub_base = sub_tile * kSwizzleCDMode * kEpiSubM;
                                using pair_store_t = cute::conditional_t<sizeof(cd_dtype_t) <= 2, uint32_t, uint64_t>;
                                *reinterpret_cast<pair_store_t*>(smem_d_bytes + sub_base + sub_row0 * kSwizzleCDMode + sw0) =
                                    *reinterpret_cast<const pair_store_t*>(p0);
                                *reinterpret_cast<pair_store_t*>(smem_d_bytes + sub_base + sub_row1 * kSwizzleCDMode + sw1) =
                                    *reinterpret_cast<const pair_store_t*>(p1);
                            }
                        }
                    }

                    cute::tma_store_fence();
                    cutlass::arch::NamedBarrier::sync(kNumMathThreads, 0);

                    if (math_warp_idx == 0 and lane_idx == 0) {
                        const uint32_t batch_store_idx = kIsBatchedEpilogue ? scheduler.current_group_idx : 0;
                        #pragma unroll
                        for (uint32_t ts = 0; ts < kNumTMAStores; ++ts) {
                            auto* smem_src = reinterpret_cast<char*>(smem_d_base) + ts * kSwizzleCDMode * kEpiSubM;
                            const uint32_t n_store = epilogue_type_t::template apply_index_n<kTMAStoreInnerDim>(
                                n_base + ts * kTMAStoreInnerDim);
                            if constexpr (kIsBatchedEpilogue) {
                                if constexpr (kWithAccumulation)
                                    cute::SM90_TMA_REDUCE_ADD_3D::copy(
                                        &tensor_map_cd, smem_src,
                                        n_store, m_base + epi_m_start, batch_store_idx);
                                else
                                    cute::SM90_TMA_STORE_3D::copy(
                                        &tensor_map_cd, smem_src,
                                        n_store, m_base + epi_m_start, batch_store_idx);
                            } else {
                                cute::SM90_TMA_STORE_2D::copy(
                                    &tensor_map_cd, smem_src,
                                    n_store, m_base + epi_m_start);
                            }
                        }
                        cute::tma_store_arrive();
                    }
                } // ms loop
            } else {
                auto store_pair = [&](cd_dtype_t* ptr, float a, float b) {
                    if constexpr (cute::is_same_v<cd_dtype_t, float>) {
                        *reinterpret_cast<float2*>(ptr) = make_float2(a, b);
                    } else {
                        ptr[0] = cd_dtype_t(a);
                        ptr[1] = cd_dtype_t(b);
                    }
                };

                const bool can_pair = (stride_cd_n == 0);
                const int64_t cd_n_stride = can_pair ? 1 : static_cast<int64_t>(stride_cd_n);

                #pragma unroll
                for (uint32_t mt = 0; mt < kMTilesPerWarp; ++mt) {
                    #pragma unroll
                    for (uint32_t nt = 0; nt < kNTilesPerWarp; ++nt) {
                        const uint32_t ai = (mt * kNTilesPerWarp + nt) * MMA_ACCUM;
                        const uint32_t nt_global = n_tile_base + nt;
                        const uint32_t col = epilogue_type_t::template apply_index_n<MMA_N>(n_base + nt_global * MMA_N) + thread_id * 2;
                        const uint32_t row0 = m_base + (m_tile_base + mt) * MMA_M + group_id;
                        const uint32_t row1 = row0 + 8;

                        if (can_pair) {
                            if (row0 < total_shape_m and col + 1 < shape_n) {
                                auto idx = cd_batch_offset + static_cast<int64_t>(row0) * cd_m_stride + col;
                                float v0 = accum[ai + 0], v1 = accum[ai + 1];
                                if constexpr (kWithAccumulation) { v0 += read_cd(gmem_c[idx]); v1 += read_cd(gmem_c[idx + 1]); }
                                store_pair(&gmem_d[idx], v0, v1);
                                cake_sm120_ready_mirror_pair_cached(
                                    reinterpret_cast<__nv_bfloat16 const*>(&gmem_d[idx]),
                                    row0 - m_base, col, result_row_base,
                                    cake_result_out);
                            }
                            if (row1 < total_shape_m and col + 1 < shape_n) {
                                auto idx = cd_batch_offset + static_cast<int64_t>(row1) * cd_m_stride + col;
                                float v2 = accum[ai + 2], v3 = accum[ai + 3];
                                if constexpr (kWithAccumulation) { v2 += read_cd(gmem_c[idx]); v3 += read_cd(gmem_c[idx + 1]); }
                                store_pair(&gmem_d[idx], v2, v3);
                                cake_sm120_ready_mirror_pair_cached(
                                    reinterpret_cast<__nv_bfloat16 const*>(&gmem_d[idx]),
                                    row1 - m_base, col, result_row_base,
                                    cake_result_out);
                            }
                        } else {
                            // Strided store: per-element N bounds check (handles shape_n=1)
                            if (row0 < total_shape_m) {
                                auto base = cd_batch_offset + static_cast<int64_t>(row0) * cd_m_stride;
                                if (col < shape_n)
                                    gmem_d[base + static_cast<int64_t>(col) * cd_n_stride] = cd_dtype_t(accum[ai + 0]);
                                if (col + 1 < shape_n)
                                    gmem_d[base + static_cast<int64_t>(col + 1) * cd_n_stride] = cd_dtype_t(accum[ai + 1]);
                            }
                            if (row1 < total_shape_m) {
                                auto base = cd_batch_offset + static_cast<int64_t>(row1) * cd_m_stride;
                                if (col < shape_n)
                                    gmem_d[base + static_cast<int64_t>(col) * cd_n_stride] = cd_dtype_t(accum[ai + 2]);
                                if (col + 1 < shape_n)
                                    gmem_d[base + static_cast<int64_t>(col + 1) * cd_n_stride] = cd_dtype_t(accum[ai + 3]);
                            }
                        }
                    }
                }
            }
            } // end else (non-split-K epilogue)

        } // persistent loop

        // The selected W2 specialization is direct-store only.  Every math
        // thread has now completed all eight consecutive N128 tiles in this
        // claim.  Its system fence precedes the caller's existing full-CTA
        // barrier, which in turn precedes the thread-0 chunk publication.
        __threadfence_system();

        // Final TMA store drain
        if constexpr (kUseTMAStoreEpilogue and kSplitKFactor == 1) {
            if (math_warp_idx == 0 and lane_idx == 0)
                cute::tma_store_wait<0>();
        }
    }

    // setmaxnreg's immediate is the absolute target register count.
    // WG0/WG1 return from 232 to 168; WG2 returns from 40 to 168.  Every
    // warp in each warpgroup executes the same instruction, then the whole CTA
    // reconverges before the wrapper's next cooperative phase.
    __syncthreads();
    if (warp_idx < kNumMathWarps) {
        cutlass::arch::warpgroup_reg_dealloc<168>();
    } else {
        cutlass::arch::warpgroup_reg_alloc<168>();
    }
    __syncthreads();

    // Signal completion for PDL (allows dependent reduce kernel to start)
    if constexpr (kSplitKFactor > 1) {
        cudaTriggerProgrammaticLaunchCompletion();
    }

#else
    if (blockIdx.x == 0 and threadIdx.x == 0)
        DG_DEVICE_ASSERT(false and "This kernel only supports sm_120a");
#endif
}

} // namespace deep_gemm
#define LOOM_INF CUDART_INF_F
#define NUM_MAIN_STAGES 1
#define SMEM_CLAIMED_RECORDS_OFF 0
#define SMEM_CLAIMED_RECORDS_STAGE_BYTES 288
#define SMEM_CLAIMED_RECORDS_STRIDE 288
#define SMEM_TOTAL 288
#define THREADS 384

extern "C" {

__device__ __forceinline__ void
cake_sm120_streaming_phase_dispatch(int* __restrict__ topk_idx_i32, float* __restrict__ topk_weights, int* __restrict__ x_fp8_i32, int* __restrict__ x_sf_i32, unsigned int* __restrict__ owner_record_counts, unsigned int* __restrict__ owner_route_counts, int* __restrict__ route_result_index, unsigned int* __restrict__ protocol_error, unsigned long long* __restrict__ signal_base_scratch, int rank, int world_size, int active_rows, unsigned int epoch, ncclDevComm const* __restrict__ gin_dev_comm, int* __restrict__ dispatch_header_out, ncclWindow_t dispatch_header_out_window, uint8_t* __restrict__ dispatch_payload_out, ncclWindow_t dispatch_payload_out_window, int* __restrict__ dispatch_header_inbox, ncclWindow_t dispatch_header_inbox_window, uint8_t* __restrict__ dispatch_payload_inbox, ncclWindow_t dispatch_payload_inbox_window)
{
    const int tid = threadIdx.x;
    const int warp = make_warp_uniform(tid / 32);
    const int lane = tid % 32;

    extern __shared__ __align__(1024) char smem_raw[];
    const int bid = blockIdx.x;
    const int num_bids = gridDim.x;

    int* claimed_records = reinterpret_cast<int*>(smem_raw + 0);
    int slot = epoch & 1;
    int launch_valid = rank >= 0 && rank < world_size && world_size >= 1 && world_size <= cake_moe::kPhysicalRanks && active_rows >= 1 && active_rows <= cake_moe::kMaxRows;
    if (launch_valid == 0) {
        if (warp == 0) {
            if (elect_sync()) {
                atomicMax(&protocol_error[0], 1);
            }
        }
    }
    if (bid == 0) {
        #pragma unroll 1
        for (int source = 0; source < cake_moe::kPhysicalRanks; source++) {
            if (warp == 0) {
                if (elect_sync()) {
                    // gin_read_signal: acquire, 64-bit signal snapshot
                    uint64_t _gin_signal_0;
                    {
                        ncclGin __gin{*(gin_dev_comm), (int)(0)};
                        _gin_signal_0 = __gin.readSignal((ncclGinSignal_t)(source), 64, cuda::memory_order_acquire);
                    }
                    signal_base_scratch[source] = _gin_signal_0;
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
    // Only the original nine route warps participate.  Warps 9..11
    // remain collective participants but never index claimed_records.
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
                for (int route_slot = 0; route_slot < cake_moe::kTopK; route_slot++) {
                    int pair = token * cake_moe::kTopK + route_slot;
                    int expert = topk_idx_i32[pair * 2];
                    int expert_hi = topk_idx_i32[pair * 2 + 1];
                    int masked = expert == -1 && expert_hi == -1;
                    int valid = expert >= 0 && expert < world_size * cake_moe::kLocalExperts && expert_hi == 0;
                    if (valid == 0 && masked == 0) {
                        atomicMax(&protocol_error[0], 1);
                    }
                    if ((valid & (int)(expert / cake_moe::kLocalExperts == owner)) != 0) {
                        route_count += 1;
                    }
                }
                int claim = -1;
                int route_base = -1;
                if (route_count > 0) {
                    unsigned int _atomic_old_0 = atomicAdd(&owner_record_counts[owner], 1);
                    unsigned int claim_u32 = _atomic_old_0;
                    claim = (int)claim_u32;
                    if (claim >= cake_moe::kMaxRows) {
                        atomicMax(&protocol_error[0], 1);
                        claim = -1;
                    }
                    unsigned int _atomic_old_1 = atomicAdd(&owner_route_counts[owner], (unsigned int)route_count);
                    unsigned int route_base_u32 = _atomic_old_1;
                    route_base = (int)route_base_u32;
                    if (route_base + route_count > cake_moe::kMaxRoutesPerPeer) {
                        atomicMax(&protocol_error[0], 1);
                        claim = -1;
                    }
                }
                claimed_records[warp * 8 + owner] = claim;
                if (claim >= 0) {
                    unsigned long long record_byte = (unsigned long long)(owner * 2 + slot) * cake_moe::kDispatchPeerSlotBytes + (unsigned long long)claim * cake_moe::kRecordBytes;
                    unsigned long long record_word = record_byte / 4;
                    *(reinterpret_cast<int*>(reinterpret_cast<int*>(dispatch_payload_out) + record_word) + (0)) = token;
                    *(reinterpret_cast<int*>(reinterpret_cast<int*>(dispatch_payload_out) + (record_word + 1)) + (0)) = route_count;
                    int write_route = 0;
                    #pragma unroll
                    for (int route_slot_1 = 0; route_slot_1 < cake_moe::kTopK; route_slot_1++) {
                        int pair_1 = token * cake_moe::kTopK + route_slot_1;
                        int expert_1 = topk_idx_i32[pair_1 * 2];
                        int expert_hi_1 = topk_idx_i32[pair_1 * 2 + 1];
                        int valid_1 = expert_1 >= 0 && expert_1 < world_size * cake_moe::kLocalExperts && expert_hi_1 == 0;
                        if ((valid_1 & (int)(expert_1 / cake_moe::kLocalExperts == owner)) != 0) {
                            *(reinterpret_cast<int*>(reinterpret_cast<int*>(dispatch_payload_out) + (record_word + 2 + write_route)) + (0)) = expert_1 - owner * cake_moe::kLocalExperts;
                            *(reinterpret_cast<int*>(reinterpret_cast<int*>(dispatch_payload_out) + (record_word + 8 + write_route)) + (0)) = route_slot_1;
                            *(reinterpret_cast<float*>(reinterpret_cast<float*>(dispatch_payload_out) + (record_word + 14 + write_route)) + (0)) = topk_weights[pair_1];
                            route_result_index[pair_1] = route_base + write_route;
                            write_route += 1;
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
                unsigned long long record_byte_1 = (unsigned long long)(owner * 2 + slot) * cake_moe::kDispatchPeerSlotBytes + (unsigned long long)record_idx * cake_moe::kRecordBytes;
                unsigned long long record_word_1 = record_byte_1 / 4;
                unsigned long long src_activation = (unsigned long long)token * cake_moe::kActivationWordsPerRow;
                unsigned long long dst_activation =
                    record_word_1 + cake_moe::kRecordHeaderWords;
                #pragma unroll 1
                for (int word = lane; word < cake_moe::kActivationWordsPerRow;
                     word += 32) {
                    *(reinterpret_cast<int*>(reinterpret_cast<int*>(dispatch_payload_out) + (dst_activation + (unsigned long long)word)) + (0)) = x_fp8_i32[src_activation + (unsigned long long)word];
                }
                unsigned long long src_sf = (unsigned long long)token * cake_moe::kActivationScaleWordsPerRow;
                unsigned long long dst_sf = record_word_1 + cake_moe::kRecordScaleWordOffset;
                #pragma unroll 1
                for (int sf_word = lane;
                     sf_word < cake_moe::kActivationScaleWordsPerRow;
                     sf_word += 32) {
                    *(reinterpret_cast<int*>(reinterpret_cast<int*>(dispatch_payload_out) + (dst_sf + (unsigned long long)sf_word)) + (0)) = x_sf_i32[src_sf + (unsigned long long)sf_word];
                }
            }
            __syncwarp();
        }
    }
    } // warp < 9
    __threadfence_system();
    cooperative_groups::this_grid().sync();
    if (bid == 0) {
        #pragma unroll 1
        for (int peer = 0; peer < world_size; peer++) {
            int local_header_word = (peer * 2 + slot) * 8;
            int local_header_byte = local_header_word * 4;
            int local_payload_byte =
                (peer * 2 + slot) * (int)cake_moe::kDispatchPeerSlotBytes;
            int remote_header_byte = (rank * 2 + slot) * cake_moe::kHeaderBytes;
            int remote_payload_byte =
                (rank * 2 + slot) * (int)cake_moe::kDispatchPeerSlotBytes;
            if (warp == 0) {
                if (elect_sync()) {
                    unsigned int count_u32 = owner_record_counts[peer];
                    int count = (int)count_u32;
                    unsigned int route_count_u32 = owner_route_counts[peer];
                    int peer_route_count = (int)route_count_u32;
                    int _max_0 = ((count) > (0) ? (count) : (0));
                    int _min_0 = ((_max_0) < (cake_moe::kMaxRows) ? (_max_0) : (cake_moe::kMaxRows));
                    int safe_count = _min_0;
                    int _max_1 = ((safe_count) > (1) ? (safe_count) : (1));
                    int send_count = _max_1;
                    int payload_bytes = send_count * cake_moe::kRecordBytes;
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
        for (int source_1 = 0; source_1 < world_size; source_1++) {
            if (warp == 0) {
                if (elect_sync()) {
                    // gin_wait_signal: acquire, rolling 64-bit comparison
                    {
                        ncclGin __gin{*(gin_dev_comm), (int)(0)};
                        __gin.waitSignal(ncclCoopThread(), (ncclGinSignal_t)(source_1), (uint64_t)(signal_base_scratch[source_1] + 2), 64, cuda::memory_order_acquire);
                    }
                }
            }
        }
    }
}

} // extern "C"

#undef LOOM_INF
#undef NUM_MAIN_STAGES
#undef SMEM_CLAIMED_RECORDS_OFF
#undef SMEM_CLAIMED_RECORDS_STAGE_BYTES
#undef SMEM_CLAIMED_RECORDS_STRIDE
#undef SMEM_TOTAL
#undef THREADS


struct CakeStreamingCandidate {
    int valid;
    int error;
    unsigned long long record_word;
    int local_expert;
    int token;
    int topk_slot;
    float route_weight;
    int result_index;
};

__device__ __forceinline__ CakeStreamingCandidate
cake_sm120_streaming_candidate(
    int const* __restrict__ payload_i32,
    float const* __restrict__ payload_f32,
    int const* __restrict__ source_record_counts,
    int const* __restrict__ source_route_counts,
    int const* __restrict__ source_active_rows,
    int source, int record, int record_route, int slot, int world_size)
{
    CakeStreamingCandidate c{};
    c.result_index = -1;
    if (source < 0 || source >= world_size || record < 0 ||
        record >= source_record_counts[source]) return c;
    c.record_word =
        (unsigned long long)(source * 2 + slot) * (cake_moe::kDispatchPeerSlotBytes / 4ull) +
        (unsigned long long)record * (cake_moe::kRecordBytes / 4ull);
    int route_count = payload_i32[c.record_word + 1];
    int route_base = payload_i32[c.record_word + 22];
    bool record_ok = route_count >= 1 && route_count <= 6 &&
        route_base >= 0 && route_base + route_count <= source_route_counts[source] &&
        route_base + route_count <= cake_moe::kMaxRoutesPerPeer &&
        payload_i32[c.record_word + 20] == source &&
        payload_i32[c.record_word + 21] == 0x50524f44;
    if (!record_ok) { c.error = 1; return c; }
    if (record_route >= route_count) return c;
    c.local_expert = payload_i32[c.record_word + 2 + record_route];
    c.token = payload_i32[c.record_word + 0];
    c.topk_slot = payload_i32[c.record_word + 8 + record_route];
    c.route_weight = payload_f32[c.record_word + 14 + record_route];
    c.result_index = route_base + record_route;
    unsigned int weight_bits = __float_as_uint(c.route_weight);
    bool finite = (weight_bits & 0x7f800000u) != 0x7f800000u;
    bool route_ok = c.local_expert >= 0 && c.local_expert < cake_moe::kLocalExperts &&
        c.token >= 0 && c.token < source_active_rows[source] &&
        c.topk_slot >= 0 && c.topk_slot < 6 && finite;
    if (!route_ok) { c.error = 1; return c; }
    c.valid = 1;
    return c;
}

extern "C" {

__device__ __forceinline__ void
cake_sm120_streaming_phase_task_build(
    int* __restrict__ dispatch_header_inbox,
    int* __restrict__ dispatch_payload_i32,
    float* __restrict__ dispatch_payload_f32,
    unsigned int* __restrict__ pool_fp8_u32,
    unsigned int* __restrict__ pool_sf_u32,
    float* __restrict__ routing_weight_pool,
    int* __restrict__ meta_source_rank,
    int* __restrict__ meta_token,
    int* __restrict__ meta_slot,
    int* __restrict__ meta_result_index,
    int* __restrict__ expert_counts,
    int* __restrict__ source_record_counts,
    int* __restrict__ source_route_counts,
    int* __restrict__ source_active_rows,
    int* __restrict__ expert_row_offsets,
    int* __restrict__ expert_scatter_offsets,
    int* __restrict__ task_expert,
    int* __restrict__ task_source_rank,
    int* __restrict__ task_owner_rank,
    int* __restrict__ task_local_expert,
    int* __restrict__ task_pool_row,
    int* __restrict__ task_m_local,
    int* __restrict__ task_valid_m,
    int* __restrict__ total_valid_routes,
    int* __restrict__ total_padded_rows,
    int* __restrict__ total_m_tasks,
    unsigned int* __restrict__ protocol_error,
    unsigned int* __restrict__ histogram_done,
    unsigned int* __restrict__ prefix_done,
    int rank, int world_size, unsigned int epoch)
{
    cooperative_groups::grid_group grid = cooperative_groups::this_grid();
    int tid = (int)threadIdx.x;
    int bid = (int)blockIdx.x;
    int grid_tid = bid * 384 + tid;
    int grid_threads = (int)gridDim.x * 384;
    int warp = tid / 32;
    int lane = tid & 31;
    int slot = (int)(epoch & 1u);
    extern __shared__ __align__(1024) char smem_raw[];
    int* warp_dst_row = reinterpret_cast<int*>(smem_raw + 92160);

    for (int source = grid_tid; source < world_size; source += grid_threads) {
        int h = (source * 2 + slot) * 8;
        int count = dispatch_header_inbox[h + 5];
        int routes = dispatch_header_inbox[h + 6];
        int rows = dispatch_header_inbox[h + 7];
        bool ok = dispatch_header_inbox[h + 0] == 0x50524f44 &&
            dispatch_header_inbox[h + 1] == 1 &&
            dispatch_header_inbox[h + 2] == (int)epoch &&
            dispatch_header_inbox[h + 3] == source &&
            dispatch_header_inbox[h + 4] == rank &&
            count >= 0 && count <= cake_moe::kMaxRows && routes >= 0 && routes <= cake_moe::kMaxRoutesPerPeer &&
            rows >= 1 && rows <= cake_moe::kMaxRows;
        if (ok) {
            source_record_counts[source] = count;
            source_route_counts[source] = routes;
            source_active_rows[source] = rows;
        } else {
            source_record_counts[source] = 0;
            source_route_counts[source] = 0;
            source_active_rows[source] = 0;
            atomicMax(protocol_error, 1u);
        }
    }
    grid.sync();

    for (int candidate = grid_tid; candidate < cake_moe::kMaxRoutesAllPeers;
         candidate += grid_threads) {
        int source = candidate / (cake_moe::kMaxRows * cake_moe::kTopK);
        int rem = candidate - source * (cake_moe::kMaxRows * cake_moe::kTopK);
        int record = rem / cake_moe::kTopK;
        int record_route = rem - record * cake_moe::kTopK;
        CakeStreamingCandidate c = cake_sm120_streaming_candidate(
            dispatch_payload_i32, dispatch_payload_f32, source_record_counts,
            source_route_counts, source_active_rows, source, record,
            record_route, slot, world_size);
        if (c.error) atomicMax(protocol_error, 1u);
        if (c.valid) atomicAdd(&expert_counts[c.local_expert], 1);
    }
    if (tid == 0) atomicAdd(histogram_done, 1u);
    grid.sync();

    if (bid == 0 && tid == 0) {
        if (*histogram_done != (unsigned int)gridDim.x) atomicMax(protocol_error, 1u);
        int running = 0;
        int task_idx = 0;
        int routes = 0;
        for (int expert = 0; expert < cake_moe::kLocalExperts; ++expert) {
            int count = expert_counts[expert];
            int padded = ((count + cake_moe::kTaskM - 1) / cake_moe::kTaskM) *
                         cake_moe::kTaskM;
            expert_row_offsets[expert] = running;
            routes += count;
            for (int m = 0; m < padded; m += cake_moe::kTaskM) {
                if (task_idx >= cake_moe::kMaxTasks) { atomicMax(protocol_error, 1u); break; }
                task_expert[task_idx] = rank * cake_moe::kLocalExperts + expert;
                task_source_rank[task_idx] = 0;
                task_owner_rank[task_idx] = rank;
                task_local_expert[task_idx] = expert;
                task_pool_row[task_idx] = running + m;
                task_m_local[task_idx] = m;
                task_valid_m[task_idx] = min(cake_moe::kTaskM, count - m);
                ++task_idx;
            }
            running += padded;
        }
        if (routes > cake_moe::kMaxRoutesAllPeers || running > cake_moe::kMaxPaddedRows || task_idx > cake_moe::kMaxTasks)
            atomicMax(protocol_error, 1u);
        total_valid_routes[0] = routes;
        total_padded_rows[0] = running;
        total_m_tasks[0] = task_idx;
        __threadfence();
        *prefix_done = 1u;
    }
    grid.sync();

    // Preserve the qualified 9-warp scatter order and SMEM extent.
    if (warp < 9) {
    int global_warp = bid * 9 + warp;
    int grid_warps = (int)gridDim.x * 9;
    for (int candidate = global_warp; candidate < cake_moe::kMaxRoutesAllPeers;
         candidate += grid_warps) {
        int source = candidate / (cake_moe::kMaxRows * cake_moe::kTopK);
        int rem = candidate - source * (cake_moe::kMaxRows * cake_moe::kTopK);
        int record = rem / cake_moe::kTopK;
        int record_route = rem - record * cake_moe::kTopK;
        CakeStreamingCandidate c = cake_sm120_streaming_candidate(
            dispatch_payload_i32, dispatch_payload_f32, source_record_counts,
            source_route_counts, source_active_rows, source, record,
            record_route, slot, world_size);
        if (c.error && lane == 0) atomicMax(protocol_error, 1u);
        if (c.valid) {
            if (lane == 0) {
                int claim = atomicAdd(&expert_scatter_offsets[c.local_expert], 1);
                int dst = expert_row_offsets[c.local_expert] + claim;
                if (claim < 0 || claim >= expert_counts[c.local_expert] ||
                    dst < 0 || dst >= cake_moe::kMaxPaddedRows) {
                    atomicMax(protocol_error, 1u);
                    dst = -1;
                } else {
                    meta_source_rank[dst] = source;
                    meta_token[dst] = c.token;
                    meta_slot[dst] = c.topk_slot;
                    meta_result_index[dst] = c.result_index;
                    routing_weight_pool[dst] = c.route_weight;
                }
                warp_dst_row[warp] = dst;
            }
            __syncwarp();
            int dst = warp_dst_row[warp];
            if (dst >= 0) {
                unsigned long long src_a =
                    c.record_word + cake_moe::kRecordHeaderWords;
                unsigned long long dst_a = (unsigned long long)dst * cake_moe::kActivationWordsPerRow;
                for (int word = lane; word < cake_moe::kActivationWordsPerRow;
                     word += 32)
                    pool_fp8_u32[dst_a + word] =
                        (unsigned int)dispatch_payload_i32[src_a + word];
                unsigned long long src_sf =
                    c.record_word + cake_moe::kRecordScaleWordOffset;
                // Canonical donor SFA is MN-major: [packed_k32_word][pool_row].
                // Every u32 retains four distinct per-K32 UE8M0 bytes.
                for (int word = lane;
                     word < cake_moe::kActivationScaleWordsPerRow;
                     word += 32)
                    pool_sf_u32[(unsigned long long)word * cake_moe::kMaxPaddedRows +
                                (unsigned long long)dst] =
                        (unsigned int)dispatch_payload_i32[src_sf + word];
            }
            __syncwarp();
        }
    }
    } // warp < 9
    grid.sync();
    if (bid == 0 && tid == 0) {
        int scatter_sum = 0;
        int expected_tasks = 0;
        for (int expert = 0; expert < cake_moe::kLocalExperts; ++expert) {
            scatter_sum += expert_scatter_offsets[expert];
            expected_tasks +=
                (expert_counts[expert] + cake_moe::kTaskM - 1) / cake_moe::kTaskM;
            if (expert_scatter_offsets[expert] != expert_counts[expert])
                atomicMax(protocol_error, 1u);
        }
        if (scatter_sum != total_valid_routes[0] ||
            expected_tasks != total_m_tasks[0] || *prefix_done != 1u)
            atomicMax(protocol_error, 1u);
    }
}

} // extern "C"
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
#if CAKE_MOE_PHASE_TRACE
    unsigned long long* phase_ns;
    unsigned int* phase_count;
#endif
};

__device__ CakeSm120CanonicalFusedReadyParams const* volatile
    cake_sm120_canonical_ready_global_params;

__device__ __forceinline__ CakeSm120CanonicalFusedReadyParams const*
cake_sm120_canonical_ready_reload_params()
{
    return cake_sm120_canonical_ready_global_params;
}

// Publish one completed eight-tile W2 claim without changing the selected
// coarse result service.  w2_tile_warp_done is no longer consumed per tile;
// its first total_m_tasks entries are reset every epoch and reused as task
// counters.  The four acq_rel +8 RMWs form one release sequence.  Its final
// transition publishes one W2 tile group for each valid semantic route so
// the selected
// service's per-source W2 tile target remains byte-exact.
__device__ __forceinline__ void cake_sm120_canonical_ready_publish_w2_chunk(
    CakeSm120CanonicalFusedReadyParams const* __restrict__ p,
    int task, int chunk_n)
{
    const int tasks = p->total_m_tasks[0];
    if (task < 0 || task >= tasks || chunk_n < 0 || chunk_n > 24 ||
        (chunk_n & 7) != 0) {
        atomicMax(p->protocol_error, 1u);
        return;
    }

    atomicAdd(p->w2_tiles_completed, 8u);
    cuda::atomic_ref<unsigned int, cuda::thread_scope_device> task_done(
        p->w2_tile_warp_done[task]);
    const unsigned int task_previous = task_done.fetch_add(
        8u, cuda::memory_order_acq_rel);
    if (task_previous + 8u == (unsigned)cake_moe::kW2TilesPerTask) {
        #pragma unroll 1
        for (unsigned int local_row = 0; local_row < (unsigned)cake_moe::kTaskM;
             ++local_row) {
            const unsigned int row =
                (unsigned int)task * (unsigned)cake_moe::kTaskM + local_row;
            const int source = p->meta_source_rank[row];
            const int result_index = p->meta_result_index[row];
            if (source >= 0 && source < p->world_size && result_index >= 0 &&
                result_index < p->source_route_counts[source]) {
                cuda::atomic_ref<unsigned int, cuda::thread_scope_device>
                    source_done(p->source_w2_done[source]);
                source_done.fetch_add((unsigned)cake_moe::kW2TilesPerTask,
                                      cuda::memory_order_acq_rel);
            } else if (source != -1) {
                atomicMax(p->protocol_error, 1u);
            }
        }
    } else if (task_previous >= (unsigned)cake_moe::kW2TilesPerTask ||
               (task_previous & 7u) != 0u) {
        atomicMax(p->protocol_error, 1u);
    }
}

static_assert(sizeof(cute::TmaDescriptor) == 128);
__device__ __noinline__ void cake_sm120_canonical_ready_pre(
    CakeSm120CanonicalFusedReadyParams const* __restrict__ p) {

    cooperative_groups::grid_group grid = cooperative_groups::this_grid();
    const int tid = (int)threadIdx.x;
    const int global_tid = (int)blockIdx.x * 384 + tid;
    const int global_threads = (int)gridDim.x * 384;

    // The donor scheduler advances by 110 residues.  Any other grid silently
    // omits work, so the entry fails closed before touching production state.
    if ((int)gridDim.x != cake_moe::kCombineCtas || (int)blockDim.x != 384) {
        if (global_tid == 0) atomicMax(p->protocol_error, 1u);
        return;
    }

    // Complete per-p->epoch state reset.  The registered result window is a
    // one-time host-zeroed precondition; every semantic route is overwritten.
    for (int i = global_tid; i < cake_moe::kPhysicalRanks; i += global_threads) {
        p->owner_record_counts[i] = 0;
        p->owner_route_counts[i] = 0;
        p->source_record_counts[i] = 0;
        p->source_route_counts[i] = 0;
        p->source_active_rows[i] = 0;
    }
    for (int i = global_tid; i < cake_moe::kMaxRoutesPerPeer;
         i += global_threads)
        p->route_result_index[i] = -1;
    for (int i = global_tid; i < cake_moe::kLocalExperts;
         i += global_threads) {
        p->expert_counts[i] = 0;
        p->expert_scatter_offsets[i] = 0;
        p->expert_row_offsets[i] = 0;
    }
    for (unsigned long long i = (unsigned long long)global_tid;
         i < cake_moe::kPoolActivationWords; i += (unsigned long long)global_threads)
        p->pool_fp8_u32[i] = 0;
    for (unsigned long long i = (unsigned long long)global_tid;
         i < cake_moe::kPoolScaleWords; i += (unsigned long long)global_threads)
        p->pool_sf_u32[i] = 0;
    for (int i = global_tid; i < cake_moe::kMaxPaddedRows;
         i += global_threads) {
        p->routing_weight_pool[i] = 0.0f;
        p->meta_source_rank[i] = -1;
        p->meta_token[i] = -1;
        p->meta_slot[i] = -1;
        p->meta_result_index[i] = -1;
        p->grouped_layout[i] = -1;
    }
    if (global_tid == 0) {
        p->total_valid_routes[0] = 0;
        p->total_padded_rows[0] = 0;
        p->total_m_tasks[0] = 0;
        p->histogram_done[0] = 0;
        p->prefix_done[0] = 0;
        p->requant_groups_done[0] = 0;
    }
    CAKE_PHASE_GRID_SYNC(grid, p);

    CAKE_PHASE_BEGIN(cake_dispatch_stamp);
    cake_sm120_streaming_phase_dispatch(
        p->topk_idx_i32, p->topk_weights, p->x_fp8_i32, p->x_sf_i32,
        p->owner_record_counts, p->owner_route_counts, p->route_result_index,
        p->protocol_error, p->dispatch_signal_base_scratch, p->rank, p->world_size,
        p->active_rows, p->epoch, p->gin_dev_comm, p->dispatch_header_out,
        p->dispatch_header_out_window, p->dispatch_payload_out,
        p->dispatch_payload_out_window, p->dispatch_header_inbox,
        p->dispatch_header_inbox_window, p->dispatch_payload_inbox,
        p->dispatch_payload_inbox_window);
    CAKE_PHASE_END(cake_dispatch_stamp, cake_moe::trace::kDispatch, p);
    CAKE_PHASE_GRID_SYNC(grid, p);

    CAKE_PHASE_BEGIN(cake_task_build_stamp);
    cake_sm120_streaming_phase_task_build(
        p->dispatch_header_inbox, reinterpret_cast<int*>(p->dispatch_payload_inbox),
        reinterpret_cast<float*>(p->dispatch_payload_inbox), p->pool_fp8_u32,
        p->pool_sf_u32, p->routing_weight_pool, p->meta_source_rank, p->meta_token,
        p->meta_slot, p->meta_result_index, p->expert_counts, p->source_record_counts,
        p->source_route_counts, p->source_active_rows, p->expert_row_offsets,
        p->expert_scatter_offsets, p->task_expert, p->task_source_rank,
        p->task_owner_rank, p->task_local_expert, p->task_pool_row, p->task_m_local,
        p->task_valid_m, p->total_valid_routes, p->total_padded_rows, p->total_m_tasks,
        p->protocol_error, p->histogram_done, p->prefix_done, p->rank, p->world_size, p->epoch);
    CAKE_PHASE_END(cake_task_build_stamp, cake_moe::trace::kTaskBuild, p);
    CAKE_PHASE_GRID_SYNC(grid, p);

    CAKE_PHASE_BEGIN(cake_layout_stamp);
    for (int expert = 0; expert < cake_moe::kLocalExperts; ++expert) {
        const int begin = p->expert_row_offsets[expert];
        const int padded =
            ((p->expert_counts[expert] + cake_moe::kTaskM - 1) / cake_moe::kTaskM) *
            cake_moe::kTaskM;
        for (int row = begin + global_tid; row < begin + padded;
             row += global_threads) {
            if (row >= 0 && row < cake_moe::kMaxPaddedRows)
                p->grouped_layout[row] = expert;
            else
                atomicMax(p->protocol_error, 1u);
        }
    }
    CAKE_PHASE_END(cake_layout_stamp, cake_moe::trace::kGroupedLayout, p);
    CAKE_PHASE_GRID_SYNC(grid, p);

    // Fail closed before a donor can consume a malformed device task shape.
    if ((int)blockIdx.x == 0 && (int)threadIdx.x == 0) {
        const int routes = p->total_valid_routes[0];
        const int padded = p->total_padded_rows[0];
        const int tasks = p->total_m_tasks[0];
        const bool valid =
            routes >= 0 && routes <= cake_moe::kMaxRoutesAllPeers &&
            padded >= 0 && padded <= cake_moe::kMaxPaddedRows &&
            tasks >= 0 && tasks <= cake_moe::kMaxTasks &&
                        padded == tasks * cake_moe::kTaskM;
        if (!valid) {
            atomicMax(p->protocol_error, 1u);
            p->total_padded_rows[0] = 0;
            p->total_m_tasks[0] = 0;
        }
    }
    CAKE_PHASE_GRID_SYNC(cooperative_groups::this_grid(), p);
}
__device__ __noinline__ void cake_sm120_canonical_ready_epilogue_task(
    cutlass::bfloat16_t const* __restrict__ w1_bf16,
    float const* __restrict__ routing_weight_pool,
    int const* __restrict__ meta_source_rank,
    uint8_t* __restrict__ intermediate_fp8,
    uint8_t* __restrict__ intermediate_sfa_u8,
    unsigned int* __restrict__ requant_groups_done,
    unsigned int* __restrict__ protocol_error,
    int task)
{
    const int tid = (int)threadIdx.x;
    if (tid < 256) {
        const int warp = tid / 32;
        const int lane = tid & 31;
        #pragma unroll 1
        for (int local_rg = warp; local_rg < cake_moe::kTaskM * 128;
             local_rg += 8) {
            const int row = task * cake_moe::kTaskM + local_rg / 128;
            const int group = local_rg % 128;
            const int logical_n = group * cake_moe::kScaleGranularityK + lane;
            const int physical_gate = (logical_n / 8) * 16 + (logical_n & 7);
            const int physical_up = physical_gate + 8;
            float gate = __bfloat162float(*reinterpret_cast<__nv_bfloat16 const*>(
                w1_bf16 + (unsigned long long)row * cake_moe::kW1PhysicalN +
                    physical_gate));
            float up = __bfloat162float(*reinterpret_cast<__nv_bfloat16 const*>(
                w1_bf16 + (unsigned long long)row * cake_moe::kW1PhysicalN +
                    physical_up));
            gate = fminf(gate, 10.0f);
            up = fminf(max_noftz(up, -10.0f), 10.0f);
            const float neg_gate_exp = __expf(-gate);
            const float sigmoid = approx_rcp(1.0f + neg_gate_exp);
            const float silu = gate * sigmoid;
            const float swiglu = silu * up;
            const float routed = swiglu * routing_weight_pool[row];
            float amax = max_noftz(routed, -routed);
            #pragma unroll
            for (int delta = 16; delta > 0; delta >>= 1)
                amax = max_noftz(amax, __shfl_xor_sync(0xffffffffu, amax, delta));
            const float sf = amax * (1.0f / 448.0f);
            const unsigned int sf_bits = __float_as_uint(sf);
            unsigned int sf_exp = ((sf_bits >> 23) & 255u) +
                (((sf_bits & 0x7fffffu) + 0x7fffffu) >> 23);
            sf_exp = min(sf_exp, 254u);
            const float sf_inv = __uint_as_float((254u - sf_exp) << 23);
            if (lane == 0) {
                const unsigned long long byte_index =
                    ((unsigned long long)(group >> 2) * cake_moe::kMaxPaddedRows +
                     (unsigned long long)row) * 4ull + (unsigned long long)(group & 3);
                intermediate_sfa_u8[byte_index] = (uint8_t)sf_exp;
                atomicAdd(requant_groups_done, 1u);
            }
            const float routed_scaled = routed * sf_inv;
            unsigned short fp8_pair;
            asm("cvt.rn.satfinite.e4m3x2.f32 %0, 0f00000000, %1;"
                : "=h"(fp8_pair) : "f"(routed_scaled));
            intermediate_fp8[(unsigned long long)row * cake_moe::kIntermediate +
                             logical_n] =
                (uint8_t)(fp8_pair & 0xffu);
        }
        __threadfence();
    }
}

__device__ __noinline__ void cake_sm120_canonical_ready_reset(
    CakeSm120CanonicalFusedReadyParams const* __restrict__ p)
{
    const int global_tid = (int)blockIdx.x * 384 + (int)threadIdx.x;
    const int global_threads = (int)gridDim.x * 384;
    for (int i = global_tid; i < cake_moe::kMaxTasks;
         i += global_threads) {
        p->w1_warp_done[i] = 0u;
        p->w1_task_ready[i] = 0u;
        p->epilogue_claimed[i] = 0u;
        p->w2_task_ready[i] = 0u;
        p->w2_task_claimed[i] = 0u;
    }
    for (int i = global_tid; i < cake_moe::kMaxTasks;
         i += global_threads)
        p->w2_tile_warp_done[i] = 0u;
    for (int i = global_tid; i < cake_moe::kPhysicalRanks; i += global_threads)
        p->source_w2_done[i] = 0u;
    for (int i = global_tid; i < 16; i += global_threads)
        p->ready_audit_counts[i] = 0u;
    if (global_tid == 0) {
        p->w1_next_tile[0] = 0u;
        p->w1_tiles_completed[0] = 0u;
        p->epilogue_completed[0] = 0u;
        p->w2_tiles_completed[0] = 0u;
        p->combine_ready[0] = 0u;
        p->combine_ctas_done[0] = 0u;
        p->epoch_done[0] = 0u;
    }
}

// Zero-argument donor boundaries keep queue cursors and Params out of the
// producer warpgroup's 40-register live range.  Each donor entry advances
// exactly eight consecutive physical N128 tiles for one M64 task.
__device__ __noinline__ void cake_sm120_canonical_ready_chunk8_execute_w1()
{
    CakeSm120CanonicalFusedReadyParams const* p =
        cake_sm120_canonical_ready_reload_params();
    const int worker = (int)blockIdx.x - 1;
    const int task = p->worker_task[worker];
    const int n = p->worker_n[worker];
    deep_gemm::cake_sm120_canonical_ready_chunk8_w1_gemm<
        0, cake_moe::kW1PhysicalN, cake_moe::kW1ShapeK,
        cake_moe::kScaleGranularityK, cake_moe::kScaleGranularityK,
        cake_moe::kLocalExperts,
        cake_moe::kTaskM, cake_moe::kBlockN, cake_moe::kBlockK, 128, 128, 0,
        cake_moe::kNumStages, cake_moe::kNumTmaThreads,
        cake_moe::kNumMathThreads, cake_moe::kWorkerCtas,
        deep_gemm::GemmType::MGroupedContiguous, false,
        cutlass::bfloat16_t,
        deep_gemm::epilogue::transform::EpilogueIdentity,
        false, true, false, true, false, cake_moe::kTaskM, 1>(
            p->w1_bf16, nullptr,
            reinterpret_cast<__nv_fp8_e4m3*>(p->pool_fp8_u32), p->w1_weight,
            p->grouped_layout, p->tensor_map_buffer, nullptr,
            (uint32_t)p->total_padded_rows[0],
            (uint32_t)cake_moe::kW1PhysicalN, (uint32_t)cake_moe::kW1ShapeK,
            (uint32_t)cake_moe::kW1PhysicalN, 0u, 0u, *p->w1_tensor_map_a, *p->w1_tensor_map_b,
            *p->w1_tensor_map_sfa, *p->w1_tensor_map_sfb,
            *p->w1_tensor_map_d, (uint32_t)task, (uint32_t)n,
            p->w1_warp_done, p->w1_task_ready, p->w1_tiles_completed);
    __syncthreads();
    if ((int)threadIdx.x == 0) {
        // The donor's per-warp releases and this CTA rendezvous order every
        // store in this chunk before the release RMW.  Value eight is the only
        // legal epilogue-ready state.
        __threadfence();
        cuda::atomic_ref<unsigned int, cuda::thread_scope_device> chunks(
            p->w1_task_ready[task]);
        const unsigned int previous = chunks.fetch_add(
            1u, cuda::memory_order_acq_rel);
        if (previous >= 8u) atomicMax(p->protocol_error, 1u);
    }
    __syncthreads();
}

__device__ __noinline__ void cake_sm120_canonical_ready_chunk8_execute_w2()
{
    CakeSm120CanonicalFusedReadyParams const* p =
        cake_sm120_canonical_ready_reload_params();
    const int worker = (int)blockIdx.x - 1;
    const int task = p->worker_task[worker];
    const int n = p->worker_n[worker];
    deep_gemm::cake_sm120_canonical_ready_chunk8_w2_gemm<
        0, cake_moe::kW2ShapeN, cake_moe::kW2ShapeK,
        cake_moe::kScaleGranularityK, cake_moe::kScaleGranularityK,
        cake_moe::kLocalExperts,
        cake_moe::kTaskM, cake_moe::kBlockN, cake_moe::kBlockK, 128, 128, 0,
        cake_moe::kNumStages, cake_moe::kNumTmaThreads,
        cake_moe::kNumMathThreads, cake_moe::kWorkerCtas,
        deep_gemm::GemmType::MGroupedContiguous, false,
        cutlass::bfloat16_t,
        deep_gemm::epilogue::transform::EpilogueIdentity,
        false, true, false, true, false, cake_moe::kTaskM, 1>(
            p->w2_bf16, nullptr,
            reinterpret_cast<__nv_fp8_e4m3*>(p->intermediate_fp8),
            p->w2_weight, p->grouped_layout, p->tensor_map_buffer,
            nullptr, (uint32_t)p->total_padded_rows[0],
            (uint32_t)cake_moe::kW2ShapeN, (uint32_t)cake_moe::kW2ShapeK,
            (uint32_t)cake_moe::kW2ShapeN, 0u, 0u, *p->w2_tensor_map_a, *p->w2_tensor_map_b,
            *p->w2_tensor_map_sfa, *p->w2_tensor_map_sfb,
            *p->w2_tensor_map_d, (uint32_t)task, (uint32_t)n,
            p->meta_source_rank, p->meta_result_index, p->source_route_counts,
            p->world_size, (int)(p->epoch & 1u), p->result_out,
            p->protocol_error);
}

__device__ __forceinline__ bool cake_sm120_ready_chunk8_claim_w1(
    unsigned int* next_tile, unsigned int target_tiles,
    unsigned int* protocol_error, int* selected_task, int* selected_n)
{
    while (true) {
        const unsigned int cursor = atomicAdd(next_tile, 0u);
        if (cursor > target_tiles || (cursor % 8u) != 0u) {
            atomicMax(protocol_error, 1u);
            return false;
        }
        if (cursor == target_tiles) return false;
        if (atomicCAS(next_tile, cursor, cursor + 8u) == cursor) {
            *selected_task = (int)(cursor / (unsigned)cake_moe::kW1TilesPerTask);
            *selected_n = (int)(cursor % (unsigned)cake_moe::kW1TilesPerTask);
            return true;
        }
    }
}

__device__ __forceinline__ bool cake_sm120_ready_chunk8_claim_w2(
    unsigned int* ready, unsigned int* claimed, int tasks, int start,
    unsigned int* protocol_error, int* selected_task, int* selected_n)
{
    #pragma unroll 1
    for (int offset = 0; offset < tasks; ++offset) {
        const int task = (start + offset) % tasks;
        cuda::atomic_ref<unsigned int, cuda::thread_scope_device> flag(ready[task]);
        if (flag.load(cuda::memory_order_acquire) != 1u) continue;
        while (true) {
            const unsigned int cursor = atomicAdd(claimed + task, 0u);
            if (cursor > (unsigned)cake_moe::kW2TilesPerTask ||
                (cursor % 8u) != 0u) {
                atomicMax(protocol_error, 1u);
                return false;
            }
            if (cursor == (unsigned)cake_moe::kW2TilesPerTask) break;
            if (atomicCAS(claimed + task, cursor, cursor + 8u) == cursor) {
                *selected_task = task;
                *selected_n = (int)cursor;
                return true;
            }
        }
    }
    return false;
}

__device__ __forceinline__ bool cake_sm120_ready_chunk8_claim_epilogue(
    unsigned int* chunk_count, unsigned int* claimed, int tasks, int start,
    int* selected_task)
{
    #pragma unroll 1
    for (int offset = 0; offset < tasks; ++offset) {
        const int task = (start + offset) % tasks;
        cuda::atomic_ref<unsigned int, cuda::thread_scope_device> count(
            chunk_count[task]);
        if (count.load(cuda::memory_order_acquire) == 8u &&
            atomicCAS(claimed + task, 0u, 1u) == 0u) {
            *selected_task = task;
            return true;
        }
    }
    return false;
}

__device__ __noinline__ void cake_sm120_canonical_ready_worker()
{
    // Work kinds: 0 retry, 1 W1 eight-tile chunk, 2 official epilogue task,
    // 3 W2 eight-tile chunk, -1 complete/fail-close.
    __shared__ int work_kind;
    __shared__ int work_task;
    __shared__ int work_n;
    __shared__ int force_w1_opportunity;
    const int worker_id = (int)blockIdx.x - 1;
#if CAKE_MOE_PHASE_TRACE
    CakeSm120CanonicalFusedReadyParams const* trace_p =
        cake_sm120_canonical_ready_reload_params();
#endif
    int scan_start = worker_id;
    if ((int)threadIdx.x == 0) force_w1_opportunity = 0;
    __syncthreads();
    while (true) {
        CAKE_PHASE_BEGIN(cake_claim_stamp);
        if ((int)threadIdx.x == 0) {
            CakeSm120CanonicalFusedReadyParams const* p =
                cake_sm120_canonical_ready_reload_params();
            work_kind = 0;
            work_task = -1;
            work_n = -1;
            const int tasks = p->total_m_tasks[0];
            const unsigned int w1_target =
            (unsigned int)tasks * (unsigned)cake_moe::kW1TilesPerTask;
            const unsigned int w2_target =
            (unsigned int)tasks * (unsigned)cake_moe::kW2TilesPerTask;
            const bool all_w1_complete =
                atomicAdd(p->w1_tiles_completed, 0u) >= w1_target;
            const unsigned int warmup_target =
                (unsigned int)(tasks < 27 ? tasks : 27);
            const bool warmup_complete =
                atomicAdd(p->epilogue_completed, 0u) >= warmup_target;
            const bool may_consume_w2 =
                all_w1_complete || (warmup_complete && worker_id < 27);
            if (atomicAdd(p->protocol_error, 0u) != 0u) {
                work_kind = -1;
            } else if (all_w1_complete &&
                       atomicAdd(p->epilogue_completed, 0u) >= (unsigned int)tasks &&
                       atomicAdd(p->w2_tiles_completed, 0u) >= w2_target) {
                work_kind = -1;
            } else {
                // A worker that consumed an early W2 chunk gets exactly one
                // W1 claim opportunity before another W2 claim.
                if (force_w1_opportunity != 0 && !all_w1_complete) {
                    force_w1_opportunity = 0;
                    if (cake_sm120_ready_chunk8_claim_w1(
                            p->w1_next_tile, w1_target, p->protocol_error,
                            &work_task, &work_n))
                        work_kind = 1;
                }
                if (work_kind == 0 && may_consume_w2 &&
                    cake_sm120_ready_chunk8_claim_w2(
                        p->w2_task_ready, p->w2_task_claimed, tasks,
                        scan_start, p->protocol_error, &work_task, &work_n))
                    work_kind = 3;
                if (work_kind == 0 && cake_sm120_ready_chunk8_claim_epilogue(
                        p->w1_task_ready, p->epilogue_claimed,
                        tasks, scan_start, &work_task))
                    work_kind = 2;
                if (work_kind == 0 && !all_w1_complete &&
                    cake_sm120_ready_chunk8_claim_w1(
                        p->w1_next_tile, w1_target, p->protocol_error,
                        &work_task, &work_n))
                    work_kind = 1;
            }
            scan_start = tasks > 0 ? (scan_start + 1) % tasks : 0;
            if (work_kind > 0) {
                p->worker_task[worker_id] = work_task;
                p->worker_n[worker_id] = work_n;
                __threadfence();
            }
        }
        __syncthreads();
        CAKE_PHASE_END(cake_claim_stamp, cake_moe::trace::kWorkerScan, trace_p);
        if (work_kind < 0) break;
        if (work_kind == 0) {
            CAKE_PHASE_BEGIN(cake_idle_stamp);
            __syncthreads();
            CAKE_PHASE_END(cake_idle_stamp, cake_moe::trace::kWorkerIdle, trace_p);
            continue;
        }
        if (work_kind == 1) {
            CAKE_PHASE_BEGIN(cake_w1_stamp);
            cake_sm120_canonical_ready_chunk8_execute_w1();
            CAKE_PHASE_END(cake_w1_stamp, cake_moe::trace::kW1Chunk, trace_p);
        } else if (work_kind == 2) {
            CakeSm120CanonicalFusedReadyParams const* p =
                cake_sm120_canonical_ready_reload_params();
            CAKE_PHASE_BEGIN(cake_epilogue_stamp);
            cake_sm120_canonical_ready_epilogue_task(
                p->w1_bf16, p->routing_weight_pool, p->meta_source_rank,
                p->intermediate_fp8, p->intermediate_sfa_u8,
                p->requant_groups_done, p->protocol_error, work_task);
            CAKE_PHASE_END(cake_epilogue_stamp, cake_moe::trace::kEpilogue, trace_p);
            __syncthreads();
            if ((int)threadIdx.x == 0) {
                cuda::atomic_ref<unsigned int, cuda::thread_scope_device> ready(
                    p->w2_task_ready[work_task]);
                ready.store(1u, cuda::memory_order_release);
                atomicAdd(p->epilogue_completed, 1u);
            }
            __syncthreads();
        } else {
            CAKE_PHASE_BEGIN(cake_w2_stamp);
            cake_sm120_canonical_ready_chunk8_execute_w2();
            CAKE_PHASE_END(cake_w2_stamp, cake_moe::trace::kW2Chunk, trace_p);
            __syncthreads();
            if ((int)threadIdx.x == 0) {
                CakeSm120CanonicalFusedReadyParams const* p =
                    cake_sm120_canonical_ready_reload_params();
                cake_sm120_canonical_ready_publish_w2_chunk(
                    p, work_task, work_n);
                const bool all_w1_now = atomicAdd(
                    p->w1_tiles_completed, 0u) >=
                    (unsigned int)p->total_m_tasks[0] *
                        (unsigned)cake_moe::kW1TilesPerTask;
                if (!all_w1_now) force_w1_opportunity = 1;
            }
            __syncthreads();
        }
        __syncthreads();
    }
}

__device__ __forceinline__ void cake_sm120_g8_capture_epoch_baselines(unsigned long long* __restrict__ result_signal_base_scratch, unsigned long long* __restrict__ combine_ack_signal_base_scratch, int world_size, ncclDevComm const* __restrict__ gin_dev_comm)
{
    if (elect_sync()) {
        #pragma unroll
        for (int peer = 0; peer < cake_moe::kPhysicalRanks; peer++) {
            if (peer < world_size) {
                // gin_read_signal: acquire, 64-bit signal snapshot
                uint64_t _gin_signal_0;
                {
                    ncclGin __gin{*(gin_dev_comm), (int)(0)};
                    _gin_signal_0 = __gin.readSignal((ncclGinSignal_t)(peer + cake_moe::kPhysicalRanks), 64, cuda::memory_order_acquire);
                }
                // gin_read_signal: acquire, 64-bit signal snapshot
                uint64_t _gin_signal_1;
                {
                    ncclGin __gin{*(gin_dev_comm), (int)(0)};
                    _gin_signal_1 = __gin.readSignal((ncclGinSignal_t)(peer + 2 * cake_moe::kPhysicalRanks), 64, cuda::memory_order_acquire);
                }
                *(reinterpret_cast<unsigned long long*>(result_signal_base_scratch) + (peer)) = _gin_signal_0;
                *(reinterpret_cast<unsigned long long*>(combine_ack_signal_base_scratch) + (peer)) = _gin_signal_1;
            }
        }
    }
}

__device__ __noinline__ void cake_sm120_canonical_ready_service_precombine(int* __restrict__ source_route_counts, unsigned int* __restrict__ source_w2_done, unsigned int* __restrict__ protocol_error, unsigned long long* __restrict__ result_signal_base_scratch, unsigned int* __restrict__ combine_ready, int world_size, int rank, unsigned int epoch, ncclDevComm const* __restrict__ gin_dev_comm, __nv_bfloat16* __restrict__ result_out, ncclWindow_t result_out_window, __nv_bfloat16* __restrict__ result_inbox, ncclWindow_t result_inbox_window)
{
    if (elect_sync()) {
        unsigned int sent_mask = 0u;
        while (sent_mask != ((1u << (world_size)) - 1u)) {
            #pragma unroll
            for (int source = 0; source < cake_moe::kPhysicalRanks; ++source) {
                if (source >= (world_size) || (sent_mask & (1u << source))) continue;
                int routes = source_route_counts[source];
                if (routes < 0 || routes > cake_moe::kMaxRoutesPerPeer) {
                    atomicMax(protocol_error, 1u);
                    routes = 0;
                }
                const unsigned int target =
                    (unsigned int)routes * (unsigned)cake_moe::kW2TilesPerTask;
                cuda::atomic_ref<unsigned int, cuda::thread_scope_device> done(source_w2_done[source]);
                if (done.load(cuda::memory_order_acquire) < target && atomicAdd(protocol_error, 0u) == 0u) continue;
                const int send_routes = routes > 1 ? routes : 1;
                const size_t bytes =
                    (size_t)send_routes * cake_moe::kOutput * 2u;
                const size_t local_byte = (size_t)(source * 2 + ((epoch) & 1u)) * 100663296ull;
                const size_t remote_byte = (size_t)((rank) * 2 + ((epoch) & 1u)) * 100663296ull;
                __threadfence_system();
                {
                    ncclGin __gin{*(gin_dev_comm), (int)(0)};
                    __gin.put(ncclTeamWorld(*(gin_dev_comm)), source, result_inbox_window, remote_byte, result_out_window, local_byte, bytes,
                        ncclGin_StrongSignalAdd{(ncclGinSignal_t)((rank) + cake_moe::kPhysicalRanks), 1ull}, ncclGin_None{}, ncclCoopThread());
                }
                sent_mask |= 1u << source;
            }
        }
        #pragma unroll
        for (int owner = 0; owner < cake_moe::kPhysicalRanks; ++owner) {
            if (owner < (world_size)) {
                ncclGin __gin{*(gin_dev_comm), (int)(0)};
                __gin.waitSignal(ncclCoopThread(), (ncclGinSignal_t)(owner + cake_moe::kPhysicalRanks), (uint64_t)(result_signal_base_scratch[owner] + 1ull), 64, cuda::memory_order_acquire);
            }
        }
        cuda::atomic_ref<unsigned int, cuda::thread_scope_device> combine(*(combine_ready));
        combine.store(1u, cuda::memory_order_release);
    }
}

__device__ __noinline__ void cake_sm120_canonical_ready_combine(
    CakeSm120CanonicalFusedReadyParams const* __restrict__ p)
{
    CAKE_PHASE_BEGIN(cake_combine_wait_stamp);
    cake_sm120_ready_wait_one(p->combine_ready);
    CAKE_PHASE_END(cake_combine_wait_stamp, cake_moe::trace::kCombineWait, p);
    const int global_tid = (int)blockIdx.x * 384 + (int)threadIdx.x;
    const int global_threads = (int)gridDim.x * 384;
    const int slot = (int)(p->epoch & 1u);
    // One thread owns kCombineVector adjacent output columns of one token. The
    // route metadata is identical for every column of a token, so resolving it
    // once per vector instead of once per column removes
    // kCombineVector * kTopK * 4 redundant scalar loads per vector, and the
    // BF16 gather and store become 16-byte accesses. The per-column
    // accumulation order and the single BF16 rounding are unchanged, so the
    // output stays bit-exact.
    constexpr int kCombineVector = 8;
    static_assert(cake_moe::kOutput % kCombineVector == 0,
                  "combine vector width must divide the output extent");
    const int vectors_per_token = cake_moe::kOutput / kCombineVector;
    const long long total_vectors =
        (long long)p->active_rows * vectors_per_token;
    for (long long v = global_tid; v < total_vectors; v += global_threads) {
        const int token = (int)(v / vectors_per_token);
        const int column = (int)(v - (long long)token * vectors_per_token) *
                           kCombineVector;
        unsigned long long base[cake_moe::kTopK];
        bool present[cake_moe::kTopK];
        #pragma unroll
        for (int topk_slot = 0; topk_slot < cake_moe::kTopK; ++topk_slot) {
            const int pair = token * cake_moe::kTopK + topk_slot;
            const int expert = p->topk_idx_i32[pair * 2];
            const int expert_hi = p->topk_idx_i32[pair * 2 + 1];
            const bool masked = expert == -1 && expert_hi == -1;
            const bool valid = expert >= 0 && expert < p->world_size * cake_moe::kLocalExperts && expert_hi == 0;
            if (!valid && !masked) atomicMax(p->protocol_error, 1u);
            present[topk_slot] = false;
            base[topk_slot] = 0ull;
            if (valid) {
                const int owner = expert / cake_moe::kLocalExperts;
                const int result_index = p->route_result_index[pair];
                const int count = (int)p->owner_route_counts[owner];
                if (result_index < 0 || result_index >= count || count < 0 || count > cake_moe::kMaxRoutesPerPeer) {
                    atomicMax(p->protocol_error, 1u);
                } else {
                    present[topk_slot] = true;
                    base[topk_slot] =
                        (unsigned long long)(owner * 2 + slot) * cake_moe::kResultElementsPerPeer +
                        (unsigned long long)result_index * cake_moe::kOutput;
                }
            }
        }
        float combined[kCombineVector];
        #pragma unroll
        for (int lane_column = 0; lane_column < kCombineVector; ++lane_column)
            combined[lane_column] = 0.0f;
        #pragma unroll
        for (int topk_slot = 0; topk_slot < cake_moe::kTopK; ++topk_slot) {
            if (!present[topk_slot]) continue;
            const __nv_bfloat16* row =
                p->result_inbox + base[topk_slot] + column;
            uint4 packed = *reinterpret_cast<const uint4*>(row);
            const __nv_bfloat16* values =
                reinterpret_cast<const __nv_bfloat16*>(&packed);
            #pragma unroll
            for (int lane_column = 0; lane_column < kCombineVector; ++lane_column)
                combined[lane_column] += __bfloat162float(values[lane_column]);
        }
        __nv_bfloat16 out[kCombineVector];
        #pragma unroll
        for (int lane_column = 0; lane_column < kCombineVector; ++lane_column)
            out[lane_column] = __float2bfloat16_rn(combined[lane_column]);
        *reinterpret_cast<uint4*>(p->final_output +
                                  (long long)token * cake_moe::kOutput + column) =
            *reinterpret_cast<const uint4*>(out);
    }
    __threadfence_system();
    __syncthreads();
    if ((int)threadIdx.x == 0) {
        cuda::atomic_ref<unsigned int, cuda::thread_scope_device> done(*p->combine_ctas_done);
        done.fetch_add(1u, cuda::memory_order_release);
    }
}

__device__ __noinline__ void cake_sm120_canonical_ready_service_postcombine(unsigned int* __restrict__ combine_ctas_done, int* __restrict__ total_m_tasks, int* __restrict__ total_valid_routes, unsigned int* __restrict__ w1_tiles_completed, unsigned int* __restrict__ epilogue_completed, unsigned int* __restrict__ w2_tiles_completed, unsigned int* __restrict__ source_w2_done, unsigned int* __restrict__ w1_warp_done, unsigned int* __restrict__ w1_task_ready, unsigned int* __restrict__ epilogue_claimed, unsigned int* __restrict__ w2_task_ready, unsigned int* __restrict__ w2_task_claimed, unsigned int* __restrict__ w1_next_tile, unsigned int* __restrict__ protocol_error, unsigned int* __restrict__ ready_audit_counts, unsigned long long* __restrict__ combine_ack_signal_base_scratch, unsigned int* __restrict__ epoch_done, int world_size, int rank, ncclDevComm const* __restrict__ gin_dev_comm, uint8_t* __restrict__ ack_out, ncclWindow_t ack_out_window, uint8_t* __restrict__ ack_inbox, ncclWindow_t ack_inbox_window)
{
    if (elect_sync()) {
        cuda::atomic_ref<unsigned int, cuda::thread_scope_device> done(*(combine_ctas_done));
        while (done.load(cuda::memory_order_acquire) != (unsigned)cake_moe::kCombineCtas) { }
        const int tasks = total_m_tasks[0];
        const unsigned int w1_target =
            (unsigned int)tasks * (unsigned)cake_moe::kW1TilesPerTask;
        const unsigned int w2_target =
            (unsigned int)tasks * (unsigned)cake_moe::kW2TilesPerTask;
        const unsigned int route_tile_target =
            (unsigned int)total_valid_routes[0] * (unsigned)cake_moe::kW2TilesPerTask;
        unsigned int source_tile_sum = 0u;
        bool audit_ok = tasks >= 0 && tasks <= cake_moe::kMaxTasks && atomicAdd(w1_tiles_completed, 0u) == w1_target && atomicAdd(epilogue_completed, 0u) == (unsigned int)tasks && atomicAdd(w2_tiles_completed, 0u) == w2_target;
        for (int source = 0; source < (world_size); ++source)
            source_tile_sum += atomicAdd(source_w2_done + source, 0u);
        audit_ok = audit_ok && source_tile_sum == route_tile_target;
        for (int task = 0; task < tasks; ++task) {
            audit_ok = audit_ok && atomicAdd(w1_warp_done + task, 0u) ==
                (unsigned)cake_moe::kW1TilesPerTask * 8u &&
                atomicAdd(w1_task_ready + task, 0u) == 8u &&
                atomicAdd(epilogue_claimed + task, 0u) == 1u &&
                atomicAdd(w2_task_ready + task, 0u) == 1u &&
                atomicAdd(w2_task_claimed + task, 0u) ==
                    (unsigned)cake_moe::kW2TilesPerTask;
        }
        if (!audit_ok) atomicMax(protocol_error, 1u);
        ready_audit_counts[0] = (unsigned int)tasks;
        ready_audit_counts[1] = w1_target;
        ready_audit_counts[2] = atomicAdd(w1_tiles_completed, 0u);
        ready_audit_counts[3] = atomicAdd(epilogue_completed, 0u);
        ready_audit_counts[4] = w2_target;
        ready_audit_counts[5] = atomicAdd(w2_tiles_completed, 0u);
        ready_audit_counts[6] = route_tile_target;
        ready_audit_counts[7] = source_tile_sum;
        ready_audit_counts[8] = atomicAdd(w1_next_tile, 0u);
        ready_audit_counts[9] = (unsigned)cake_moe::kWorkerCtas;
        ready_audit_counts[10] = atomicAdd(combine_ctas_done, 0u);
        #pragma unroll
        for (int owner = 0; owner < cake_moe::kPhysicalRanks; ++owner) {
            if (owner >= (world_size)) continue;
            __threadfence_system();
            {
                ncclGin __gin{*(gin_dev_comm), (int)(0)};
                __gin.put(ncclTeamWorld(*(gin_dev_comm)), owner, ack_inbox_window, (size_t)(rank), ack_out_window, (size_t)owner, (size_t)1,
                    ncclGin_StrongSignalAdd{(ncclGinSignal_t)((rank) + 2 * cake_moe::kPhysicalRanks), 1ull}, ncclGin_None{}, ncclCoopThread());
            }
        }
        #pragma unroll
        for (int source = 0; source < cake_moe::kPhysicalRanks; ++source) {
            if (source < (world_size)) {
                ncclGin __gin{*(gin_dev_comm), (int)(0)};
                __gin.waitSignal(ncclCoopThread(), (ncclGinSignal_t)(source + 2 * cake_moe::kPhysicalRanks), (uint64_t)(combine_ack_signal_base_scratch[source] + 1ull), 64, cuda::memory_order_acquire);
            }
        }
        cuda::atomic_ref<unsigned int, cuda::thread_scope_device> epoch_complete(*(epoch_done));
        epoch_complete.store(1u, cuda::memory_order_release);
    }
}

extern "C" {

__global__ __launch_bounds__(384, 1) void
kernel_cake_sm120_production_canonical_fused_ready_chunk8(
    CakeSm120CanonicalFusedReadyParams const* __restrict__ p)
{
    if ((int)gridDim.x != cake_moe::kCombineCtas || (int)blockDim.x != 384) {
        if ((int)blockIdx.x == 0 && (int)threadIdx.x == 0)
            atomicMax(p->protocol_error, 1u);
        return;
    }

    CAKE_PHASE_BEGIN(cake_reset_stamp);
    cake_sm120_canonical_ready_reset(p);
    CAKE_PHASE_END(cake_reset_stamp, cake_moe::trace::kReset, p);
    CAKE_PHASE_GRID_SYNC(cooperative_groups::this_grid(), p);
    // Capture before ready_pre's existing dispatch world barrier.  No rank
    // can pass that barrier and issue a result PUT until every rank has
    // reached it after completing this capture.
    CAKE_PHASE_BEGIN(cake_baseline_stamp);
    if ((int)blockIdx.x == 0 && (int)threadIdx.x < 32) {
        cake_sm120_g8_capture_epoch_baselines(
            p->result_signal_base_scratch,
            p->combine_ack_signal_base_scratch,
            p->world_size,
            p->gin_dev_comm);
    }
    CAKE_PHASE_END(cake_baseline_stamp, cake_moe::trace::kEpochBaseline, p);
    CAKE_PHASE_GRID_SYNC(cooperative_groups::this_grid(), p);
    cake_sm120_canonical_ready_pre(p);
    CAKE_PHASE_GRID_SYNC(cooperative_groups::this_grid(), p);

    if ((int)blockIdx.x == 0 && (int)threadIdx.x == 0) {
        cake_sm120_canonical_ready_global_params = p;
        __threadfence();
    }
    CAKE_PHASE_GRID_SYNC(cooperative_groups::this_grid(), p);

    const int tasks = p->total_m_tasks[0];
    const bool has_compute = tasks > 0 && tasks <= cake_moe::kMaxTasks &&
        p->total_padded_rows[0] == tasks * cake_moe::kTaskM;
    const int physical = (int)blockIdx.x;

    CAKE_PHASE_BEGIN(cake_service_stamp);
    if (physical == 0) {
        if ((int)threadIdx.x < 32) {
            cake_sm120_canonical_ready_service_precombine(
                p->source_route_counts,
                p->source_w2_done,
                p->protocol_error,
                p->result_signal_base_scratch,
                p->combine_ready,
                p->world_size,
                p->rank,
                p->epoch,
                p->gin_dev_comm,
                p->result_out,
                p->result_out_window,
                p->result_inbox,
                p->result_inbox_window);
        }
    } else if (has_compute) {
        cake_sm120_canonical_ready_worker();
    }
    CAKE_PHASE_END(cake_service_stamp, cake_moe::trace::kServicePrecombine, p);

    // No W1/epilogue/W2/result grid rendezvous: all roles converge through
    // device-scope release/acquire counters and CTA0 preserves GIN progress.
    CAKE_PHASE_BEGIN(cake_combine_stamp);
    cake_sm120_canonical_ready_combine(p);
    CAKE_PHASE_END(cake_combine_stamp, cake_moe::trace::kCombine, p);
    CAKE_PHASE_BEGIN(cake_ack_stamp);
    if (physical == 0 && (int)threadIdx.x < 32) {
        cake_sm120_canonical_ready_service_postcombine(
            p->combine_ctas_done,
            p->total_m_tasks,
            p->total_valid_routes,
            p->w1_tiles_completed,
            p->epilogue_completed,
            p->w2_tiles_completed,
            p->source_w2_done,
            p->w1_warp_done,
            p->w1_task_ready,
            p->epilogue_claimed,
            p->w2_task_ready,
            p->w2_task_claimed,
            p->w1_next_tile,
            p->protocol_error,
            p->ready_audit_counts,
            p->combine_ack_signal_base_scratch,
            p->epoch_done,
            p->world_size,
            p->rank,
            p->gin_dev_comm,
            p->ack_out,
            p->ack_out_window,
            p->ack_inbox,
            p->ack_inbox_window);
    }
    CAKE_PHASE_END(cake_ack_stamp, cake_moe::trace::kServicePostcombine, p);
    CAKE_PHASE_BEGIN(cake_epoch_wait_stamp);
    cake_sm120_ready_wait_one(p->epoch_done);
    CAKE_PHASE_END(cake_epoch_wait_stamp, cake_moe::trace::kGridSync, p);
}

extern "C" __host__ void* cake_sm120_canonical_ready_chunk8_kernel_ptr() {
    return reinterpret_cast<void*>(kernel_cake_sm120_production_canonical_fused_ready_chunk8);
}

} // extern "C"
