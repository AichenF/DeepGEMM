#pragma once

#include <deep_gemm/impls/sm90_nvfp4_mega_moe.cuh>

namespace deep_gemm {
namespace nvfp4 {

__device__ __forceinline__ uint2 dequant_two_seed_word(
        const uint32_t packed_e2m1, const uint32_t base_codes,
        const uint2* __restrict__ lut_smem) {
    if (__builtin_expect((base_codes & 0x00800080u) != 0u, 0)) {
        const uint2 lut0 = lut_smem[base_codes & 0x7fu];
        const uint2 lut1 = lut_smem[(base_codes >> 16) & 0x7fu];
        uint32_t out0 = byte_perm_unchecked(
            lut0.x, lut0.y, packed_e2m1 & 0x00007777u);
        uint32_t out1 = byte_perm_unchecked(
            lut1.x, lut1.y, (packed_e2m1 >> 16) & 0x00007777u);
        out0 |= (packed_e2m1 << 4) & 0x80808080u;
        out1 |= packed_e2m1 & 0x80808080u;
        return make_uint2(out0, out1);
    }

    const uint32_t buffer10 =
        __byte_perm(base_codes, 0u, 0x1004) + 0x00080000u;
    const uint32_t buffer20 =
        __byte_perm(base_codes, base_codes, 0x1010) + 0x10180810u;
    const uint32_t buffer11 =
        __byte_perm(base_codes, 0u, 0x3224) + 0x00080000u;
    const uint32_t buffer21 =
        __byte_perm(base_codes, base_codes, 0x3232) + 0x10180810u;
    const uint32_t magnitudes0 =
        __byte_perm(buffer10, buffer20, packed_e2m1);
    const uint32_t magnitudes1 =
        __byte_perm(buffer11, buffer21, packed_e2m1 >> 16);
    uint32_t out0;
    uint32_t out1;
    asm("lop3.b32 %0, %1, %2, 0x80808080, 0xf8;"
        : "=r"(out0) : "r"(magnitudes0), "r"(packed_e2m1 << 4));
    asm("lop3.b32 %0, %1, %2, 0x80808080, 0xf8;"
        : "=r"(out1) : "r"(magnitudes1), "r"(packed_e2m1));
    return make_uint2(out0, out1);
}

__device__ __forceinline__ void dequant_two_seed_row_regs(
        uint8_t* __restrict__ fp8_dst,
        const uint4 (&fp4_quads)[4],
        const uint32_t (&base_codes)[4],
        const uint32_t row_swizzle,
        const uint2* __restrict__ lut_smem) {
    #pragma unroll
    for (uint32_t k32 = 0; k32 < 4; ++k32) {
        const uint4 q = fp4_quads[k32];
        const uint2 out0 = dequant_two_seed_word(q.x, base_codes[k32], lut_smem);
        const uint2 out1 = dequant_two_seed_word(q.y, base_codes[k32], lut_smem);
        const uint2 out2 = dequant_two_seed_word(q.z, base_codes[k32], lut_smem);
        const uint2 out3 = dequant_two_seed_word(q.w, base_codes[k32], lut_smem);
        *reinterpret_cast<uint4*>(fp8_dst + ((k32 * 32u) ^ row_swizzle)) =
            make_uint4(out0.x, out1.x, out2.x, out3.x);
        *reinterpret_cast<uint4*>(fp8_dst + ((k32 * 32u + 16u) ^ row_swizzle)) =
            make_uint4(out0.y, out1.y, out2.y, out3.y);
    }
}

__device__ __forceinline__ void dequant_smem_b_from_packed_grouped_nibble(
        uint8_t* __restrict__ smem_b,
        const uint8_t* __restrict__ packed_b,
        const uint32_t row,
        const uint2* __restrict__ lut_smem) {
    const uint8_t* __restrict__ row_ptr = packed_b + row * 80;
    const uint4* __restrict__ fp4_src = reinterpret_cast<const uint4*>(row_ptr);
    uint4 fp4_quads[4];
    uint32_t base_codes[4];
    #pragma unroll
    for (int i = 0; i < 4; ++i) {
        fp4_quads[i] = fp4_src[i];
        base_codes[i] = *reinterpret_cast<const uint32_t*>(row_ptr + 64 + i * 4);
    }
    dequant_two_seed_row_regs(
        smem_b + row * 128, fp4_quads, base_codes,
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
    uint32_t base_codes0[4];
    uint32_t base_codes1[4];
    #pragma unroll
    for (int i = 0; i < 4; ++i) {
        fp4_quads0[i] = fp4_src0[i];
        fp4_quads1[i] = fp4_src1[i];
        base_codes0[i] = *reinterpret_cast<const uint32_t*>(row_ptr0 + 64 + i * 4);
        base_codes1[i] = *reinterpret_cast<const uint32_t*>(row_ptr1 + 64 + i * 4);
    }

    asm volatile("bar.sync %0, %1;" : : "n"(kBarIdx), "n"(kNumDequantThreads));

    const uint32_t row_swizzle = (tid & 7u) << 4;
    uint8_t* __restrict__ fp8_dst0 = smem_b + row0 * 128;
    uint8_t* __restrict__ fp8_dst1 = smem_b + row1 * 128;
    dequant_two_seed_row_regs(
        fp8_dst0, fp4_quads0, base_codes0, row_swizzle, lut_smem);
    dequant_two_seed_row_regs(
        fp8_dst1, fp4_quads1, base_codes1, row_swizzle, lut_smem);
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
