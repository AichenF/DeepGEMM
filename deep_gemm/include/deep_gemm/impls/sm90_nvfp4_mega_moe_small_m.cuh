#pragma once

#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wunknown-attributes"

#include <cstdint>
#include <type_traits>
#include <cutlass/arch/barrier.h>
#include <cutlass/arch/reg_reconfig.h>

#include <cute/arch/cluster_sm90.hpp>
#include <cute/arch/copy_sm90_tma.hpp>
#include <cute/arch/mma_sm89.hpp>
#include <cute/atom/mma_atom.hpp>
#include <cute/algorithm/cooperative_gemm.hpp>

#include <deep_gemm/common/math.cuh>
#include <deep_gemm/common/tma_copy.cuh>
#include <deep_gemm/common/utils.cuh>
#include <deep_gemm/comm/barrier.cuh>
#include <deep_gemm/layout/sym_buffer.cuh>
#include <deep_gemm/layout/mega_moe.cuh>
#include <deep_gemm/mma/sm90.cuh>
#include <deep_gemm/scheduler/mega_moe.cuh>
#include <deep_gemm/ptx/ld_st.cuh>
#include <deep_gemm/ptx/tma.cuh>
#include <deep_gemm/ptx/utils.cuh>
#include <deep_gemm/ptx/wgmma.cuh>
#include <deep_gemm/impls/sm90_nvfp4_mega_moe_mode2_dequant.cuh>

#define __CLION_IDE__

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

__device__ __forceinline__ void dequant_braided_quad_ilp(
        uint8_t* __restrict__ fp8_dst,
        const uint4& q,
        const uint2& lut0,
        const uint2& lut1,
        const int scale_i0,
        const uint32_t row_swizzle) {
    // Expose all four independent PRMT chains together so ptxas can overlap
    // their selector/sign work before either 128-bit shared-memory store.
    const uint32_t q0_sel0 = q.x & 0x00007777u;
    const uint32_t q0_sel1 = (q.x >> 16) & 0x00007777u;
    const uint32_t q1_sel0 = q.y & 0x00007777u;
    const uint32_t q1_sel1 = (q.y >> 16) & 0x00007777u;
    const uint32_t q2_sel0 = q.z & 0x00007777u;
    const uint32_t q2_sel1 = (q.z >> 16) & 0x00007777u;
    const uint32_t q3_sel0 = q.w & 0x00007777u;
    const uint32_t q3_sel1 = (q.w >> 16) & 0x00007777u;

    uint32_t q0_out0 = byte_perm_unchecked(lut0.x, lut0.y, q0_sel0);
    uint32_t q0_out1 = byte_perm_unchecked(lut0.x, lut0.y, q0_sel1);
    uint32_t q1_out0 = byte_perm_unchecked(lut0.x, lut0.y, q1_sel0);
    uint32_t q1_out1 = byte_perm_unchecked(lut0.x, lut0.y, q1_sel1);
    uint32_t q2_out0 = byte_perm_unchecked(lut1.x, lut1.y, q2_sel0);
    uint32_t q2_out1 = byte_perm_unchecked(lut1.x, lut1.y, q2_sel1);
    uint32_t q3_out0 = byte_perm_unchecked(lut1.x, lut1.y, q3_sel0);
    uint32_t q3_out1 = byte_perm_unchecked(lut1.x, lut1.y, q3_sel1);

    q0_out0 |= q.x & 0x80808080u;
    q0_out1 |= (q.x << 4) & 0x80808080u;
    q1_out0 |= q.y & 0x80808080u;
    q1_out1 |= (q.y << 4) & 0x80808080u;
    q2_out0 |= q.z & 0x80808080u;
    q2_out1 |= (q.z << 4) & 0x80808080u;
    q3_out0 |= q.w & 0x80808080u;
    q3_out1 |= (q.w << 4) & 0x80808080u;

    *reinterpret_cast<uint4*>(fp8_dst + ((scale_i0 * 16) ^ row_swizzle)) =
        make_uint4(q0_out0, q0_out1, q1_out0, q1_out1);
    *reinterpret_cast<uint4*>(fp8_dst + (((scale_i0 + 1) * 16) ^ row_swizzle)) =
        make_uint4(q2_out0, q2_out1, q3_out0, q3_out1);
}

template <
    uint32_t kNumMaxTokensPerRank,
    uint32_t kHidden,
    uint32_t kIntermediateHidden,
    uint32_t kNumExperts,
    uint32_t kNumTopk,
    uint32_t kNumExpertsPerWave,
    uint32_t BLOCK_M,
    uint32_t kNumMaxPoolTokens,
    uint32_t kNumPaddedSFPoolTokens,
    uint32_t kNumStages,
    uint32_t kNumSMs,
    uint32_t kNumRanks,
    float kActivationClamp,
    bool kFastMath,
    bool kSwapABRequested,
    bool kSingleActiveDispatchWarp,
    bool kUseMode2RowDecoder
>
CUTLASS_GLOBAL __launch_bounds__(384, 1) void
sm90_nvfp4_mega_moe_small_m_fused_impl(
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
    constexpr uint32_t BLOCK_N = 256;
    constexpr uint32_t BLOCK_K = 128;
    constexpr uint32_t kNumDispatchThreads = 64;
    constexpr uint32_t kNumNonEpilogueThreads = 64;
    constexpr uint32_t kNumEpilogueThreads = 256;
    constexpr uint32_t kClusterSize = 1;
    constexpr uint32_t L1_SHAPE_N = kIntermediateHidden * 2;
    constexpr uint32_t L1_SHAPE_K = kHidden;
    constexpr uint32_t L2_SHAPE_N = kHidden;
    constexpr uint32_t L2_SHAPE_K = kIntermediateHidden;
    constexpr uint32_t kNumDispatchWarps = kNumDispatchThreads / 32;
    constexpr uint32_t kNumMMANonEpilogueWarps =
        kNumNonEpilogueThreads / 32;
    constexpr uint32_t kNumEpilogueWarps = kNumEpilogueThreads / 32;
    constexpr uint32_t kNumEpilogueWarpgroups =
        kNumEpilogueWarps / 4;
    constexpr uint32_t kNumTokensPerWarp = 32 / kNumTopk;
    constexpr uint32_t kNumExpertsPerRank = kNumExperts / kNumRanks;
#include <deep_gemm/impls/sm90_nvfp4_mega_moe_small_m_fused_body.inl>
}

}  // namespace deep_gemm
