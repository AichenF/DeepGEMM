#pragma once

#include <deep_gemm/impls/sm90_nvfp4_mega_moe_nibble_group_dequant.cuh>

namespace deep_gemm {

template <
    uint32_t kNumMaxTokensPerRank,
    uint32_t kHidden, uint32_t kIntermediateHidden,
    uint32_t kNumExperts, uint32_t kNumTopk,
    uint32_t kNumExpertsPerWave,
    uint32_t BLOCK_M, uint32_t BLOCK_N, uint32_t BLOCK_K,
    uint32_t kNumMaxPoolTokens,
    uint32_t kNumPaddedSFPoolTokens,
    uint32_t kNumStages,
    uint32_t kNumDispatchThreads, uint32_t kNumNonEpilogueThreads,
    uint32_t kNumEpilogueThreads,
    uint32_t kClusterSize,
    uint32_t kNumSMs, uint32_t kNumRanks,
    float kActivationClamp,
    bool kFastMath,
    bool kPhaseProfileRequested = false,
    bool kLoaderDequantRequested = false,
    bool kSwapABRequested = false,
    bool kDp4aSelectorPack = false,
    bool kHybridLowSelectorPack = false,
    uint32_t kQuadDequantMask = 0,
    uint32_t L1_SHAPE_N = kIntermediateHidden * 2,
    uint32_t L1_SHAPE_K = kHidden,
    uint32_t L2_SHAPE_N = kHidden,
    uint32_t L2_SHAPE_K = kIntermediateHidden,
    uint32_t kNumDispatchWarps = kNumDispatchThreads / 32,
    uint32_t kNumMMANonEpilogueWarps = kNumNonEpilogueThreads / 32,
    uint32_t kNumEpilogueWarps = kNumEpilogueThreads / 32,
    uint32_t kNumEpilogueWarpgroups = kNumEpilogueWarps / 4,
    uint32_t kNumThreads = kNumDispatchThreads + kNumNonEpilogueThreads + kNumEpilogueThreads,
    uint32_t kNumTokensPerWarp = 32 / kNumTopk,
    uint32_t kNumExpertsPerRank = kNumExperts / kNumRanks
>
CUTLASS_GLOBAL __launch_bounds__(kNumThreads, 1) void
sm90_nvfp4_mega_moe_nibble_group_fused_impl(
        void* y,
        int* cumulative_local_expert_recv_stats,
        const uint32_t num_tokens,
        const __grid_constant__ layout::SymBuffer<kNumRanks> sym_buffer,
        const __grid_constant__ cute::TmaDescriptor tensor_map_l1_acts,
        const __grid_constant__ cute::TmaDescriptor tensor_map_l1_acts_sf,
        const __grid_constant__ cute::TmaDescriptor tensor_map_l1_weights,
        const __grid_constant__ cute::TmaDescriptor tensor_map_l1_output,
        const __grid_constant__ cute::TmaDescriptor tensor_map_l2_acts,
        const __grid_constant__ cute::TmaDescriptor tensor_map_l2_acts_sf,
        const __grid_constant__ cute::TmaDescriptor tensor_map_l2_weights,
        const float* __restrict__ l1_global_scales,
        const float* __restrict__ l2_global_scales) {
#include <deep_gemm/impls/sm90_nvfp4_mega_moe_nibble_group_fused_body.inl>
}

// Mode-2 is a separate kernel symbol so the legacy grouped-nibble template,
// body, and generated H20 cubin stay byte-for-byte independent of this path.
template <
    uint32_t kNumMaxTokensPerRank,
    uint32_t kHidden, uint32_t kIntermediateHidden,
    uint32_t kNumExperts, uint32_t kNumTopk,
    uint32_t kNumExpertsPerWave,
    uint32_t BLOCK_M, uint32_t BLOCK_N, uint32_t BLOCK_K,
    uint32_t kNumMaxPoolTokens,
    uint32_t kNumPaddedSFPoolTokens,
    uint32_t kNumStages,
    uint32_t kNumDispatchThreads, uint32_t kNumNonEpilogueThreads,
    uint32_t kNumEpilogueThreads,
    uint32_t kClusterSize,
    uint32_t kNumSMs, uint32_t kNumRanks,
    float kActivationClamp,
    bool kFastMath,
    bool kPhaseProfileRequested = false,
    bool kLoaderDequantRequested = false,
    bool kSwapABRequested = false,
    bool kDp4aSelectorPack = false,
    bool kHybridLowSelectorPack = false,
    uint32_t kQuadDequantMask = 0,
    uint32_t L1_SHAPE_N = kIntermediateHidden * 2,
    uint32_t L1_SHAPE_K = kHidden,
    uint32_t L2_SHAPE_N = kHidden,
    uint32_t L2_SHAPE_K = kIntermediateHidden,
    uint32_t kNumDispatchWarps = kNumDispatchThreads / 32,
    uint32_t kNumMMANonEpilogueWarps = kNumNonEpilogueThreads / 32,
    uint32_t kNumEpilogueWarps = kNumEpilogueThreads / 32,
    uint32_t kNumEpilogueWarpgroups = kNumEpilogueWarps / 4,
    uint32_t kNumThreads = kNumDispatchThreads + kNumNonEpilogueThreads + kNumEpilogueThreads,
    uint32_t kNumTokensPerWarp = 32 / kNumTopk,
    uint32_t kNumExpertsPerRank = kNumExperts / kNumRanks
>
CUTLASS_GLOBAL __launch_bounds__(kNumThreads, 1) void
sm90_nvfp4_mega_moe_mode2_nibble_group_fused_impl(
        void* y,
        int* cumulative_local_expert_recv_stats,
        const uint32_t num_tokens,
        const __grid_constant__ layout::SymBuffer<kNumRanks> sym_buffer,
        const __grid_constant__ cute::TmaDescriptor tensor_map_l1_acts,
        const __grid_constant__ cute::TmaDescriptor tensor_map_l1_acts_sf,
        const __grid_constant__ cute::TmaDescriptor tensor_map_l1_weights,
        const __grid_constant__ cute::TmaDescriptor tensor_map_l1_output,
        const __grid_constant__ cute::TmaDescriptor tensor_map_l2_acts,
        const __grid_constant__ cute::TmaDescriptor tensor_map_l2_acts_sf,
        const __grid_constant__ cute::TmaDescriptor tensor_map_l2_weights,
        const float* __restrict__ l1_global_scales,
        const float* __restrict__ l2_global_scales) {
#define dequant_smem_b_from_packed_grouped_nibble dequant_smem_b_from_packed_mode2_nibble
#define dequant_smem_b_half_row_grouped_nibble dequant_smem_b_half_row_mode2_nibble
#define dequant_smem_b_inplace_two_rows_grouped_nibble dequant_smem_b_inplace_two_rows_mode2_nibble
#include <deep_gemm/impls/sm90_nvfp4_mega_moe_nibble_group_fused_body.inl>
#undef dequant_smem_b_inplace_two_rows_grouped_nibble
#undef dequant_smem_b_half_row_grouped_nibble
#undef dequant_smem_b_from_packed_grouped_nibble
}

}  // namespace deep_gemm
