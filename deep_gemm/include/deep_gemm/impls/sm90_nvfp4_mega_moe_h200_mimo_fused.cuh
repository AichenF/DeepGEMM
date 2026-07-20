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
#include <deep_gemm/quantization/nvfp4_dequant.cuh>

namespace deep_gemm {
namespace nvfp4 {

__device__ __forceinline__ uint2 dequant_mode2_nibble_word(
        const uint32_t packed, const uint2& lut) {
    const uint32_t magnitude_selectors = packed & 0x77777777u;
    uint32_t out_hi =
        byte_perm_unchecked(lut.x, lut.y, magnitude_selectors);
    uint32_t out_lo =
        byte_perm_unchecked(lut.x, lut.y, magnitude_selectors >> 16);
    asm("lop3.b32 %0, %0, %1, 0x80808080, 0xf8;"
        : "+r"(out_hi) : "r"(packed));
    const uint32_t shifted = packed << 4;
    asm("lop3.b32 %0, %0, %1, 0x80808080, 0xf8;"
        : "+r"(out_lo) : "r"(shifted));
    return make_uint2(out_hi, out_lo);
}

template <bool kQuadILP = false>
__device__ __forceinline__ void dequant_mode2_nibble_row_regs(
        uint8_t* __restrict__ fp8_dst,
        const uint4 (&fp4_quads)[4],
        const uint2& scale_words,
        const uint32_t row_swizzle,
        const uint2* __restrict__ lut_smem) {
#pragma unroll
    for (int quad_i = 0; quad_i < 4; ++quad_i) {
        const uint4 q = fp4_quads[quad_i];
        const uint32_t scale_word =
            quad_i < 2 ? scale_words.x : scale_words.y;
        const int scale_i0 = quad_i * 2;
        const int scale_i1 = scale_i0 + 1;
        const uint32_t scale0 =
            (scale_word >> ((scale_i0 & 3) * 8)) & 0x7fu;
        const uint32_t scale1 =
            (scale_word >> ((scale_i1 & 3) * 8)) & 0x7fu;
        const uint2 lut0 = lut_smem[scale0];
        const uint2 lut1 = lut_smem[scale1];

        const uint2 q0 = dequant_mode2_nibble_word(q.x, lut0);
        const uint2 q1 = dequant_mode2_nibble_word(q.y, lut0);
        if constexpr (!kQuadILP) {
            *reinterpret_cast<uint4*>(
                fp8_dst + ((scale_i0 * 16) ^ row_swizzle)) =
                make_uint4(q0.x, q0.y, q1.x, q1.y);
        }

        const uint2 q2 = dequant_mode2_nibble_word(q.z, lut1);
        const uint2 q3 = dequant_mode2_nibble_word(q.w, lut1);
        if constexpr (kQuadILP) {
            *reinterpret_cast<uint4*>(
                fp8_dst + ((scale_i0 * 16) ^ row_swizzle)) =
                make_uint4(q0.x, q0.y, q1.x, q1.y);
        }
        *reinterpret_cast<uint4*>(
            fp8_dst + ((scale_i1 * 16) ^ row_swizzle)) =
            make_uint4(q2.x, q2.y, q3.x, q3.y);
    }
}

template <bool kQuadILP = false>
__device__ __forceinline__ void dequant_smem_b_from_packed_mode2_nibble(
        uint8_t* __restrict__ smem_b,
        const uint8_t* __restrict__ packed_b,
        const uint32_t row,
        const uint2* __restrict__ lut_smem) {
    const uint8_t* __restrict__ row_ptr = packed_b + row * 80;
    const uint4* __restrict__ fp4_src =
        reinterpret_cast<const uint4*>(row_ptr);
    uint4 fp4_quads[4];
#pragma unroll
    for (int i = 0; i < 4; ++i)
        fp4_quads[i] = fp4_src[i];
    const uint2 scale_words =
        *reinterpret_cast<const uint2*>(row_ptr + 64);
    dequant_mode2_nibble_row_regs<kQuadILP>(
        smem_b + row * 128, fp4_quads, scale_words,
        (row & 7u) << 4, lut_smem);
}

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

template <int kQuad, bool kQuadIlp>
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

    if constexpr (kQuadIlp) {
        dequant_braided_quad_ilp(
            fp8_dst, fp4_quads[kQuad], lut0, lut1, kQuad * 2, row_swizzle);
    } else {
        dequant_braided_quad(
            fp8_dst, fp4_quads[kQuad], lut0, lut1, kQuad * 2, row_swizzle);
    }

    if constexpr (kQuad + 1 < 4) {
        dequant_braided_quad_lut_window<kQuad + 1, kQuadIlp>(
            fp8_dst, fp4_quads, scale_word_lo, scale_word_hi, lut_smem,
            next_lut0, next_lut1, row_swizzle);
    }
}

template <bool kQuadIlp = false>
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
    dequant_braided_quad_lut_window<0, kQuadIlp>(
        smem_b + row * 128, fp4_quads, scale_words.x, scale_words.y,
        lut_smem, lut0, lut1, (row & 7u) << 4);
}

}  // namespace nvfp4

template <
    uint32_t kNumMaxTokensPerRank,
    uint32_t kNumExpertsPerWave,
    uint32_t BLOCK_M,
    uint32_t kNumMaxPoolTokens,
    uint32_t kNumPaddedSFPoolTokens,
    uint32_t kNumStages,
    float kActivationClamp,
    bool kFastMath,
    bool kSwapABRequested,
    bool kSingleActiveDispatchWarp,
    bool kUseMode2RowDecoder
>
CUTLASS_GLOBAL __launch_bounds__(384, 1) void
sm90_nvfp4_mega_moe_h200_mimo_fused_impl(
        void* y,
        int* cumulative_local_expert_recv_stats,
        const uint32_t num_tokens,
        const __grid_constant__ layout::SymBuffer<8> sym_buffer,
        const __grid_constant__ cute::TmaDescriptor tensor_map_l1_acts,
        const __grid_constant__ cute::TmaDescriptor tensor_map_l1_acts_sf,
        const __grid_constant__ cute::TmaDescriptor tensor_map_l1_weights,
        const __grid_constant__ cute::TmaDescriptor tensor_map_l1_output,
        const __grid_constant__ cute::TmaDescriptor tensor_map_l2_acts,
        const __grid_constant__ cute::TmaDescriptor tensor_map_l2_acts_sf,
        const __grid_constant__ cute::TmaDescriptor tensor_map_l2_weights,
        const float* __restrict__ l1_global_scales,
        const float* __restrict__ l2_global_scales) {
    constexpr uint32_t kHidden = 6144;
    constexpr uint32_t kIntermediateHidden = 2048;
    constexpr uint32_t kNumExperts = 384;
    constexpr uint32_t kNumTopk = 8;
    constexpr uint32_t BLOCK_N = 256;
    constexpr uint32_t BLOCK_K = 128;
    constexpr uint32_t kNumDispatchThreads = 64;
    constexpr uint32_t kNumNonEpilogueThreads = 64;
    constexpr uint32_t kNumEpilogueThreads = 256;
    constexpr uint32_t kNumSMs = 132;
    constexpr uint32_t kNumRanks = 8;
    constexpr uint32_t L1_SHAPE_N = kIntermediateHidden * 2;
    constexpr uint32_t L1_SHAPE_K = kHidden;
    constexpr uint32_t L2_SHAPE_N = kHidden;
    constexpr uint32_t L2_SHAPE_K = kIntermediateHidden;
    constexpr uint32_t kNumDispatchWarps = kNumDispatchThreads / 32;
    constexpr uint32_t kNumMMANonEpilogueWarps = kNumNonEpilogueThreads / 32;
    constexpr uint32_t kNumEpilogueWarps = kNumEpilogueThreads / 32;
    constexpr uint32_t kNumEpilogueWarpgroups = kNumEpilogueWarps / 4;
    constexpr uint32_t kNumTokensPerWarp = 32 / kNumTopk;
    constexpr uint32_t kNumExpertsPerRank = kNumExperts / kNumRanks;
#include <deep_gemm/impls/sm90_nvfp4_mega_moe_h200_mimo_fused_body.inl>
}

}  // namespace deep_gemm

#pragma clang diagnostic pop
