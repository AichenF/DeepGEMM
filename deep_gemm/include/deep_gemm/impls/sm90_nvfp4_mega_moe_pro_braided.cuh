#pragma once

#include <deep_gemm/impls/sm90_nvfp4_mega_moe.cuh>

namespace deep_gemm {
namespace nvfp4 {

__device__ __forceinline__ uint2 dequant_braided_selector_word(
        const uint32_t braided, const uint2& lut) {
    const uint32_t sel0 = braided & 0x00007777u;
    const uint32_t sel1 = (braided >> 16) & 0x00007777u;
    uint32_t out0 = byte_perm_unchecked(lut.x, lut.y, sel0);
    uint32_t out1 = byte_perm_unchecked(lut.x, lut.y, sel1);
    out0 |= braided & 0x80808080u;
    out1 |= (braided << 4) & 0x80808080u;
    return make_uint2(out0, out1);
}

__device__ __forceinline__ void dequant_braided_quad(
        uint8_t* __restrict__ fp8_dst,
        const uint4& q,
        const uint2& lut0,
        const uint2& lut1,
        const int scale_i0,
        const uint32_t row_swizzle) {
    const uint2 q0 = dequant_braided_selector_word(q.x, lut0);
    const uint2 q1 = dequant_braided_selector_word(q.y, lut0);
    *reinterpret_cast<uint4*>(fp8_dst + ((scale_i0 * 16) ^ row_swizzle)) =
        make_uint4(q0.x, q0.y, q1.x, q1.y);

    const uint2 q2 = dequant_braided_selector_word(q.z, lut1);
    const uint2 q3 = dequant_braided_selector_word(q.w, lut1);
    *reinterpret_cast<uint4*>(fp8_dst + (((scale_i0 + 1) * 16) ^ row_swizzle)) =
        make_uint4(q2.x, q2.y, q3.x, q3.y);
}

template <int kQuad>
__device__ __forceinline__ void dequant_braided_quad_lut_window(
        uint8_t* __restrict__ fp8_dst,
        const uint4 (&fp4_quads)[4],
        const uint32_t scale_word_lo,
        const uint32_t scale_word_hi,
        const uint2* __restrict__ lut_smem,
        const uint2 lut0,
        const uint2 lut1,
        const uint32_t row_swizzle) {
    uint2 next_lut0;
    uint2 next_lut1;
    if constexpr (kQuad + 1 < 4) {
        constexpr int kNextScaleI0 = (kQuad + 1) * 2;
        constexpr int kNextScaleI1 = kNextScaleI0 + 1;
        const uint32_t next_scale_word = kQuad + 1 < 2 ? scale_word_lo : scale_word_hi;
        const uint32_t next_scale0 =
            (next_scale_word >> ((kNextScaleI0 & 3) * 8)) & 0x7fu;
        const uint32_t next_scale1 =
            (next_scale_word >> ((kNextScaleI1 & 3) * 8)) & 0x7fu;
        next_lut0 = lut_smem[next_scale0];
        next_lut1 = lut_smem[next_scale1];
    }

    dequant_braided_quad(
        fp8_dst, fp4_quads[kQuad], lut0, lut1, kQuad * 2, row_swizzle);

    if constexpr (kQuad + 1 < 4) {
        dequant_braided_quad_lut_window<kQuad + 1>(
            fp8_dst, fp4_quads, scale_word_lo, scale_word_hi, lut_smem,
            next_lut0, next_lut1, row_swizzle);
    }
}

__device__ __forceinline__ void dequant_smem_b_from_packed_braided_lut_window(
        uint8_t* __restrict__ smem_b,
        const uint8_t* __restrict__ packed_b,
        const uint32_t row,
        const uint2* __restrict__ lut_smem) {
    const uint8_t* __restrict__ row_ptr = packed_b + row * 80;
    const uint4* __restrict__ fp4_src = reinterpret_cast<const uint4*>(row_ptr);
    uint4 fp4_quads[4];
#pragma unroll
    for (int i = 0; i < 4; ++i)
        fp4_quads[i] = fp4_src[i];

    const uint2 scale_words = *reinterpret_cast<const uint2*>(row_ptr + 64);
    const uint2 lut0 = lut_smem[scale_words.x & 0x7fu];
    const uint2 lut1 = lut_smem[(scale_words.x >> 8) & 0x7fu];
    dequant_braided_quad_lut_window<0>(
        smem_b + row * 128, fp4_quads, scale_words.x, scale_words.y,
        lut_smem, lut0, lut1, (row & 7u) << 4);
}

}  // namespace nvfp4

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
    bool kSingleActiveDispatchWarp = false,
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
sm90_nvfp4_mega_moe_pro_braided_fused_impl(
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
#include <deep_gemm/impls/sm90_nvfp4_mega_moe_pro_braided_fused_body.inl>
}

}  // namespace deep_gemm
