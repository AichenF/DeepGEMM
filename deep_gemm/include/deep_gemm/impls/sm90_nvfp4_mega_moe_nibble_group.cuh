#pragma once

#include <deep_gemm/impls/sm90_nvfp4_mega_moe.cuh>

namespace deep_gemm {
namespace nvfp4 {

__device__ __forceinline__ uint32_t apply_grouped_nibble_signs(
        const uint32_t magnitudes, const uint32_t grouped_nibbles) {
    uint32_t sign_fill;
    const uint32_t shifted = grouped_nibbles << 4;
    asm("prmt.b32 %0, %1, %2, 0x9d8c;"
        : "=r"(sign_fill) : "r"(grouped_nibbles), "r"(shifted));
    uint32_t result;
    asm("lop3.b32 %0, %1, %2, 0x80808080, 0xf8;"
        : "=r"(result) : "r"(magnitudes), "r"(sign_fill));
    return result;
}

__device__ __forceinline__ uint2 dequant_grouped_nibble_word(
        const uint32_t grouped, const uint2& lut) {
    const uint32_t magnitude_selectors = grouped & 0x77777777u;
    uint32_t out_hi = byte_perm_unchecked(lut.x, lut.y, magnitude_selectors);
    uint32_t out_lo = byte_perm_unchecked(lut.x, lut.y, magnitude_selectors >> 16);
    out_hi = apply_grouped_nibble_signs(out_hi, grouped);
    out_lo = apply_grouped_nibble_signs(out_lo, grouped >> 16);
    return make_uint2(out_hi, out_lo);
}

template <bool kQuadILP = false>
__device__ __forceinline__ void dequant_grouped_nibble_row_regs(
        uint8_t* __restrict__ fp8_dst,
        const uint4 (&fp4_quads)[4],
        const uint2& scale_words,
        const uint32_t row_swizzle,
        const uint2* __restrict__ lut_smem) {
    #pragma unroll
    for (int quad_i = 0; quad_i < 4; ++quad_i) {
        const uint4 q = fp4_quads[quad_i];
        const uint32_t scale_word = quad_i < 2 ? scale_words.x : scale_words.y;
        const int scale_i0 = quad_i * 2;
        const int scale_i1 = scale_i0 + 1;
        const uint32_t scale0 = (scale_word >> ((scale_i0 & 3) * 8)) & 0x7fu;
        const uint32_t scale1 = (scale_word >> ((scale_i1 & 3) * 8)) & 0x7fu;
        const uint2 lut0 = lut_smem[scale0];
        const uint2 lut1 = lut_smem[scale1];

        const uint2 q0 = dequant_grouped_nibble_word(q.x, lut0);
        const uint2 q1 = dequant_grouped_nibble_word(q.y, lut0);
        if constexpr (!kQuadILP) {
            *reinterpret_cast<uint4*>(fp8_dst + ((scale_i0 * 16) ^ row_swizzle)) =
                make_uint4(q0.x, q0.y, q1.x, q1.y);
        }

        const uint2 q2 = dequant_grouped_nibble_word(q.z, lut1);
        const uint2 q3 = dequant_grouped_nibble_word(q.w, lut1);
        if constexpr (kQuadILP) {
            // Delay the first vector store so all four decode chains can overlap.
            *reinterpret_cast<uint4*>(fp8_dst + ((scale_i0 * 16) ^ row_swizzle)) =
                make_uint4(q0.x, q0.y, q1.x, q1.y);
        }
        *reinterpret_cast<uint4*>(fp8_dst + ((scale_i1 * 16) ^ row_swizzle)) =
            make_uint4(q2.x, q2.y, q3.x, q3.y);
    }
}

template <bool kQuadILP = false>
__device__ __forceinline__ void dequant_smem_b_from_packed_grouped_nibble(
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
    dequant_grouped_nibble_row_regs<kQuadILP>(
        smem_b + row * 128, fp4_quads, scale_words,
        (row & 7u) << 4, lut_smem);
}

template <uint32_t kNumDequantThreads, uint32_t kBarIdx,
          uint32_t kFusedRowBytes = 80>
__device__ __forceinline__ void dequant_smem_b_inplace_two_rows_grouped_nibble(
        uint8_t* __restrict__ smem_b,
        const uint32_t tid,
        const uint2* __restrict__ lut_smem) {
    const uint32_t row0 = tid;
    const uint32_t row1 = tid + kNumDequantThreads;
    const uint8_t* __restrict__ row_ptr0 = smem_b + row0 * kFusedRowBytes;
    const uint8_t* __restrict__ row_ptr1 = smem_b + row1 * kFusedRowBytes;
    const uint4* __restrict__ fp4_src0 = reinterpret_cast<const uint4*>(row_ptr0);
    const uint4* __restrict__ fp4_src1 = reinterpret_cast<const uint4*>(row_ptr1);
    uint4 fp4_quads0[4];
    uint4 fp4_quads1[4];
    #pragma unroll
    for (int i = 0; i < 4; ++i) {
        fp4_quads0[i] = fp4_src0[i];
        fp4_quads1[i] = fp4_src1[i];
    }
    const uint2 scale_words0 = *reinterpret_cast<const uint2*>(row_ptr0 + 64);
    const uint2 scale_words1 = *reinterpret_cast<const uint2*>(row_ptr1 + 64);

    asm volatile("bar.sync %0, %1;" : : "n"(kBarIdx), "n"(kNumDequantThreads));

    const uint32_t row_swizzle = (tid & 7u) << 4;
    uint8_t* __restrict__ fp8_dst0 = smem_b + row0 * 128;
    uint8_t* __restrict__ fp8_dst1 = smem_b + row1 * 128;
    #pragma unroll
    for (int quad_i = 0; quad_i < 4; ++quad_i) {
        const uint4 q0 = fp4_quads0[quad_i];
        const uint4 q1 = fp4_quads1[quad_i];
        const uint32_t scale_word0 = quad_i < 2 ? scale_words0.x : scale_words0.y;
        const uint32_t scale_word1 = quad_i < 2 ? scale_words1.x : scale_words1.y;
        const int scale_i0 = quad_i * 2;
        const int scale_i1 = scale_i0 + 1;

        const uint32_t scale00 = (scale_word0 >> ((scale_i0 & 3) * 8)) & 0x7fu;
        const uint32_t scale10 = (scale_word1 >> ((scale_i0 & 3) * 8)) & 0x7fu;
        const uint2 lut00 = lut_smem[scale00];
        const uint2 lut10 = lut_smem[scale10];
        const uint2 q0x = dequant_grouped_nibble_word(q0.x, lut00);
        const uint2 q0y = dequant_grouped_nibble_word(q0.y, lut00);
        const uint2 q1x = dequant_grouped_nibble_word(q1.x, lut10);
        const uint2 q1y = dequant_grouped_nibble_word(q1.y, lut10);
        *reinterpret_cast<uint4*>(fp8_dst0 + ((scale_i0 * 16) ^ row_swizzle)) =
            make_uint4(q0x.x, q0x.y, q0y.x, q0y.y);
        *reinterpret_cast<uint4*>(fp8_dst1 + ((scale_i0 * 16) ^ row_swizzle)) =
            make_uint4(q1x.x, q1x.y, q1y.x, q1y.y);

        const uint32_t scale01 = (scale_word0 >> ((scale_i1 & 3) * 8)) & 0x7fu;
        const uint32_t scale11 = (scale_word1 >> ((scale_i1 & 3) * 8)) & 0x7fu;
        const uint2 lut01 = lut_smem[scale01];
        const uint2 lut11 = lut_smem[scale11];
        const uint2 q0z = dequant_grouped_nibble_word(q0.z, lut01);
        const uint2 q0w = dequant_grouped_nibble_word(q0.w, lut01);
        const uint2 q1z = dequant_grouped_nibble_word(q1.z, lut11);
        const uint2 q1w = dequant_grouped_nibble_word(q1.w, lut11);
        *reinterpret_cast<uint4*>(fp8_dst0 + ((scale_i1 * 16) ^ row_swizzle)) =
            make_uint4(q0z.x, q0z.y, q0w.x, q0w.y);
        *reinterpret_cast<uint4*>(fp8_dst1 + ((scale_i1 * 16) ^ row_swizzle)) =
            make_uint4(q1z.x, q1z.y, q1w.x, q1w.y);
    }
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

}  // namespace deep_gemm
