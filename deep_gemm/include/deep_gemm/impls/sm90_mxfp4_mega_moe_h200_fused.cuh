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
#include <deep_gemm/quantization/mxfp4_fused_scale.cuh>

namespace deep_gemm {
namespace mxfp4 {

// MXFP4 decoders for the SM90 fused MegaMoE mainloop.
//
// Two differences from the NVFP4 decoders they replace, both consequences of
// the scale format:
//
//  * Group width. NVFP4 groups 16 elements, so a 32-element quad spans two
//    scales and its four packed words split across lut0/lut1. MXFP4 groups 32,
//    so one quad is exactly one scale group and all four words share a table.
//    A BK128 row therefore carries 4 E8M0 bytes (one uint32) instead of 8
//    UE4M3 bytes (a uint2).
//  * Table provenance. A UE4M3 scale has a mantissa, so NVFP4 must look the
//    scaled magnitudes up. E8M0 is a pure power of two, so the table is built
//    in registers by adding the exponent -- see mxfp4_fused_scale.cuh. No
//    shared memory, no table load.

template <bool kQuadILP = false>
__device__ __forceinline__ void dequant_mode2_nibble_row_regs(
        uint8_t* __restrict__ fp8_dst,
        const uint4 (&fp4_quads)[4],
        const uint32_t scale_word,
        const uint32_t row_swizzle) {
#pragma unroll
    for (int quad_i = 0; quad_i < 4; ++quad_i) {
        const uint4 q = fp4_quads[quad_i];
        const ScaledLut lut =
            make_scaled_lut((scale_word >> (quad_i * 8)) & 0xffu);

        const uint2 w0 = dequant_word(q.x, lut);
        const uint2 w1 = dequant_word(q.y, lut);
        if constexpr (!kQuadILP) {
            *reinterpret_cast<uint4*>(
                fp8_dst + (((quad_i * 2) * 16) ^ row_swizzle)) =
                make_uint4(w0.x, w0.y, w1.x, w1.y);
        }
        const uint2 w2 = dequant_word(q.z, lut);
        const uint2 w3 = dequant_word(q.w, lut);
        if constexpr (kQuadILP) {
            *reinterpret_cast<uint4*>(
                fp8_dst + (((quad_i * 2) * 16) ^ row_swizzle)) =
                make_uint4(w0.x, w0.y, w1.x, w1.y);
        }
        *reinterpret_cast<uint4*>(
            fp8_dst + (((quad_i * 2 + 1) * 16) ^ row_swizzle)) =
            make_uint4(w2.x, w2.y, w3.x, w3.y);
    }
}

template <bool kQuadILP = false>
__device__ __forceinline__ void dequant_smem_b_from_packed_mode2_nibble(
        uint8_t* __restrict__ smem_b,
        const uint8_t* __restrict__ packed_b,
        const uint32_t row) {
    const uint8_t* __restrict__ row_ptr = packed_b + row * 80;
    const uint4* __restrict__ fp4_src = reinterpret_cast<const uint4*>(row_ptr);
    uint4 fp4_quads[4];
#pragma unroll
    for (int i = 0; i < 4; ++i)
        fp4_quads[i] = fp4_src[i];
    const uint32_t scale_word =
        *reinterpret_cast<const uint32_t*>(row_ptr + 64);
    dequant_mode2_nibble_row_regs<kQuadILP>(
        smem_b + row * 128, fp4_quads, scale_word, (row & 7u) << 4);
}

// Threads 0-127 and 128-255 each decode one K64 half of the same N128 tile,
// letting the two M64 warpgroups reuse the decoded weights. A K64 half is two
// quads, i.e. two E8M0 bytes.
__device__ __forceinline__ void dequant_smem_b_from_packed_mode2_nibble_split_m(
        uint8_t* __restrict__ smem_b,
        const uint8_t* __restrict__ packed_b,
        const uint32_t thread_idx) {
    const uint32_t row = thread_idx & 127u;
    const uint32_t k_half_idx = thread_idx >> 7;
    const uint8_t* __restrict__ row_ptr = packed_b + row * 80u;
    const uint4* __restrict__ fp4_src =
        reinterpret_cast<const uint4*>(row_ptr + k_half_idx * 32u);
    const uint32_t scale_pair = *reinterpret_cast<const uint16_t*>(
        row_ptr + 64u + k_half_idx * sizeof(uint16_t));
    uint8_t* __restrict__ fp8_dst = smem_b + row * 128u;
    const uint32_t row_swizzle = (row & 7u) << 4;

#pragma unroll
    for (uint32_t quad_i = 0; quad_i < 2; ++quad_i) {
        const uint4 q = fp4_src[quad_i];
        const ScaledLut lut =
            make_scaled_lut((scale_pair >> (quad_i * 8u)) & 0xffu);
        const uint2 w0 = dequant_word(q.x, lut);
        const uint2 w1 = dequant_word(q.y, lut);
        const uint2 w2 = dequant_word(q.z, lut);
        const uint2 w3 = dequant_word(q.w, lut);
        const uint32_t off0 = k_half_idx * 64u + (quad_i * 2u) * 16u;
        const uint32_t off1 = k_half_idx * 64u + (quad_i * 2u + 1u) * 16u;
        *reinterpret_cast<uint4*>(fp8_dst + (off0 ^ row_swizzle)) =
            make_uint4(w0.x, w0.y, w1.x, w1.y);
        *reinterpret_cast<uint4*>(fp8_dst + (off1 ^ row_swizzle)) =
            make_uint4(w2.x, w2.y, w3.x, w3.y);
    }
}

__device__ __forceinline__ void dequant_braided_quad(
        uint8_t* __restrict__ fp8_dst,
        const uint4& q,
        const ScaledLut& lut,
        const int quad_i,
        const uint32_t row_swizzle) {
    const uint2 w0 = dequant_word(q.x, lut);
    const uint2 w1 = dequant_word(q.y, lut);
    *reinterpret_cast<uint4*>(fp8_dst + (((quad_i * 2) * 16) ^ row_swizzle)) =
        make_uint4(w0.x, w0.y, w1.x, w1.y);
    const uint2 w2 = dequant_word(q.z, lut);
    const uint2 w3 = dequant_word(q.w, lut);
    *reinterpret_cast<uint4*>(fp8_dst + (((quad_i * 2 + 1) * 16) ^ row_swizzle)) =
        make_uint4(w2.x, w2.y, w3.x, w3.y);
}

template <bool kQuadIlp = false>
__device__ __forceinline__ void dequant_smem_b_from_packed_braided_lut_window(
        uint8_t* __restrict__ smem_b,
        const uint8_t* __restrict__ packed_b,
        const uint32_t row) {
    // The NVFP4 variant software-pipelines its table loads a quad ahead to hide
    // shared-memory latency. The fused table has no load to hide, so the plain
    // loop is the pipelined version.
    dequant_smem_b_from_packed_mode2_nibble<kQuadIlp>(smem_b, packed_b, row);
}

}  // namespace mxfp4

template <
    uint32_t kNumMaxTokensPerRank,
    uint32_t kNumExpertsPerWave,
    uint32_t BLOCK_M,
    uint32_t BLOCK_N,
    uint32_t kNumMaxPoolTokens,
    uint32_t kNumPaddedSFPoolTokens,
    uint32_t kNumStages,
    float kActivationClamp,
    bool kFastMath,
    bool kSwapABRequested,
    bool kSingleActiveDispatchWarp,
    bool kUseMode2RowDecoder,
    bool kUseInterleavedScheduler
>
CUTLASS_GLOBAL __launch_bounds__(384, 1) void
sm90_mxfp4_mega_moe_h200_fused_impl(
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
#include <deep_gemm/impls/sm90_mxfp4_mega_moe_h200_fused_body.inl>
}

}  // namespace deep_gemm

#pragma clang diagnostic pop
