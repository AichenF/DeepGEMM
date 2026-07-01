#pragma once

#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wunknown-attributes"

#include <cstdint>
#include <type_traits>
#include <cutlass/arch/barrier.h>
#include <cutlass/arch/reg_reconfig.h>

#include <cute/arch/cluster_sm90.hpp>
#include <deep_gemm/quantization/nvfp4_dequant.cuh>

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
#define __CLION_IDE__


// NVFP4: in-place expand packed FP4 (first 8KB of smem_b) to FP8 (full 16KB).
// Each math-WG thread handles 64 bytes of packed FP4 (one row), reads them
// into registers, applies eight per-16-K UE4M3 scale bytes from tile-major
// scale storage, then overwrites the row with FP8 values for WGMMA.
namespace deep_gemm {
namespace nvfp4 {
template <uint32_t kNumMathThreads, uint32_t kBarIdx>
__device__ __forceinline__ void dequant_smem_b_inplace(uint8_t* __restrict__ smem_b, uint32_t tid_in_wg,
                                                       const uint8_t* __restrict__ scale_row_base,
                                                       const uint2* __restrict__ lut_smem) {
    const uint32_t* __restrict__ fp4_src = reinterpret_cast<const uint32_t*>(smem_b + tid_in_wg * 64);
    uint32_t fp4_regs[16];
    #pragma unroll
    for (int i = 0; i < 16; ++i) fp4_regs[i] = fp4_src[i];

    const uint2 scale_words = *reinterpret_cast<const uint2*>(scale_row_base);
    const uint32_t scale_word_lo = scale_words.x;
    const uint32_t scale_word_hi = scale_words.y;

    asm volatile("bar.sync %0, %1;" : : "n"(kBarIdx), "n"(kNumMathThreads));

    uint8_t* __restrict__ fp8_dst_base = smem_b + tid_in_wg * 128;
    const uint32_t row_swizzle = (tid_in_wg & 7u) << 4;
    #pragma unroll
    for (int scale_i = 0; scale_i < 8; ++scale_i) {
        const uint32_t scale_word = scale_i < 4 ? scale_word_lo : scale_word_hi;
        const uint32_t scale_ue4m3 = (scale_word >> ((scale_i & 3) * 8)) & 0xffu;
        const int i0 = scale_i * 2;
        const int i1 = i0 + 1;
        const uint2 lut = lut_smem[scale_ue4m3 & 0x7fu];
        const uint2 s0 = deep_gemm::nvfp4::dequant_nvfp4_to_fp8_pair_with_lut(fp4_regs[i0], lut);
        const uint2 s1 = deep_gemm::nvfp4::dequant_nvfp4_to_fp8_pair_with_lut(fp4_regs[i1], lut);
        const uint32_t logical = i0 * 8;
        const uint32_t physical = logical ^ row_swizzle;
        *reinterpret_cast<uint4*>(fp8_dst_base + physical) =
            make_uint4(s0.x, s0.y, s1.x, s1.y);
    }
}

template <uint32_t kNumMathThreads>
__device__ __forceinline__ void dequant_smem_b_from_packed(uint8_t* __restrict__ smem_b,
                                                           const uint8_t* __restrict__ packed_b,
                                                           uint32_t tid_in_wg,
                                                           const uint8_t* __restrict__ scale_row_base,
                                                           const uint2* __restrict__ lut_smem) {
    (void)kNumMathThreads;
    const uint32_t* __restrict__ fp4_src = reinterpret_cast<const uint32_t*>(packed_b + tid_in_wg * 64);
    uint32_t fp4_regs[16];
    #pragma unroll
    for (int i = 0; i < 16; ++i) fp4_regs[i] = fp4_src[i];

    const uint2 scale_words = *reinterpret_cast<const uint2*>(scale_row_base);
    const uint32_t scale_word_lo = scale_words.x;
    const uint32_t scale_word_hi = scale_words.y;

    uint8_t* __restrict__ fp8_dst_base = smem_b + tid_in_wg * 128;
    const uint32_t row_swizzle = (tid_in_wg & 7u) << 4;
    #pragma unroll
    for (int scale_i = 0; scale_i < 8; ++scale_i) {
        const uint32_t scale_word = scale_i < 4 ? scale_word_lo : scale_word_hi;
        const uint32_t scale_ue4m3 = (scale_word >> ((scale_i & 3) * 8)) & 0xffu;
        const int i0 = scale_i * 2;
        const int i1 = i0 + 1;
        const uint2 lut = lut_smem[scale_ue4m3 & 0x7fu];
        const uint2 s0 = deep_gemm::nvfp4::dequant_nvfp4_to_fp8_pair_with_lut(fp4_regs[i0], lut);
        const uint2 s1 = deep_gemm::nvfp4::dequant_nvfp4_to_fp8_pair_with_lut(fp4_regs[i1], lut);
        const uint32_t logical = i0 * 8;
        const uint32_t physical = logical ^ row_swizzle;
        *reinterpret_cast<uint4*>(fp8_dst_base + physical) =
            make_uint4(s0.x, s0.y, s1.x, s1.y);
    }
}

__device__ __forceinline__ void dequant_smem_b_from_packed_fused_scale(uint8_t* __restrict__ smem_b,
                                                                      const uint8_t* __restrict__ packed_b,
                                                                      uint32_t tid_in_wg,
                                                                      const uint2* __restrict__ lut_smem) {
    const uint8_t* __restrict__ row_ptr = packed_b + tid_in_wg * 80;
    const uint4* __restrict__ fp4_src = reinterpret_cast<const uint4*>(row_ptr);
    uint4 fp4_quads[4];
    #pragma unroll
    for (int i = 0; i < 4; ++i) fp4_quads[i] = fp4_src[i];

    const uint2 scale_words = *reinterpret_cast<const uint2*>(row_ptr + 64);
    const uint32_t scale_word_lo = scale_words.x;
    const uint32_t scale_word_hi = scale_words.y;

    uint8_t* __restrict__ fp8_dst_base = smem_b + tid_in_wg * 128;
    const uint32_t row_swizzle = (tid_in_wg & 7u) << 4;
    #pragma unroll
    for (int quad_i = 0; quad_i < 4; ++quad_i) {
        const uint4 fp4_quad = fp4_quads[quad_i];
        const uint32_t scale_word = quad_i < 2 ? scale_word_lo : scale_word_hi;

        const int scale_i0 = quad_i * 2;
        const uint32_t scale_ue4m3_0 = (scale_word >> ((scale_i0 & 3) * 8)) & 0xffu;
        const uint2 lut0 = lut_smem[scale_ue4m3_0 & 0x7fu];
        const uint2 q0_s0 = deep_gemm::nvfp4::dequant_nvfp4_to_fp8_pair_with_lut(fp4_quad.x, lut0);
        const uint2 q0_s1 = deep_gemm::nvfp4::dequant_nvfp4_to_fp8_pair_with_lut(fp4_quad.y, lut0);
        const uint32_t physical0 = (scale_i0 * 16) ^ row_swizzle;
        *reinterpret_cast<uint4*>(fp8_dst_base + physical0) =
            make_uint4(q0_s0.x, q0_s0.y, q0_s1.x, q0_s1.y);

        const int scale_i1 = scale_i0 + 1;
        const uint32_t scale_ue4m3_1 = (scale_word >> ((scale_i1 & 3) * 8)) & 0xffu;
        const uint2 lut1 = lut_smem[scale_ue4m3_1 & 0x7fu];
        const uint2 q1_s0 = deep_gemm::nvfp4::dequant_nvfp4_to_fp8_pair_with_lut(fp4_quad.z, lut1);
        const uint2 q1_s1 = deep_gemm::nvfp4::dequant_nvfp4_to_fp8_pair_with_lut(fp4_quad.w, lut1);
        const uint32_t physical1 = (scale_i1 * 16) ^ row_swizzle;
        *reinterpret_cast<uint4*>(fp8_dst_base + physical1) =
            make_uint4(q1_s0.x, q1_s0.y, q1_s1.x, q1_s1.y);
    }
}

template <uint32_t kNumMathThreads>
__device__ __forceinline__ void dequant_smem_b_inplace_split_barrier(
        uint8_t* __restrict__ smem_b, uint32_t tid_in_wg,
        const uint8_t* __restrict__ scale_row_base,
        const uint2* __restrict__ lut_smem,
        cutlass::arch::ClusterTransactionBarrier* read_barrier,
        const uint32_t barrier_phase) {
    const uint32_t* __restrict__ fp4_src = reinterpret_cast<const uint32_t*>(smem_b + tid_in_wg * 64);
    uint32_t fp4_regs[16];
    #pragma unroll
    for (int i = 0; i < 16; ++i) fp4_regs[i] = fp4_src[i];

    const uint2 scale_words = *reinterpret_cast<const uint2*>(scale_row_base);
    const uint32_t scale_word_lo = scale_words.x;
    const uint32_t scale_word_hi = scale_words.y;

    deep_gemm::ptx::mbarrier_arrive(read_barrier);
    if (tid_in_wg < kNumMathThreads / 2u)
        deep_gemm::ptx::mbarrier_wait(read_barrier, barrier_phase);

    uint8_t* __restrict__ fp8_dst_base = smem_b + tid_in_wg * 128;
    const uint32_t row_swizzle = (tid_in_wg & 7u) << 4;
    #pragma unroll
    for (int scale_i = 0; scale_i < 8; ++scale_i) {
        const uint32_t scale_word = scale_i < 4 ? scale_word_lo : scale_word_hi;
        const uint32_t scale_ue4m3 = (scale_word >> ((scale_i & 3) * 8)) & 0xffu;
        const int i0 = scale_i * 2;
        const int i1 = i0 + 1;
        const uint2 lut = lut_smem[scale_ue4m3 & 0x7fu];
        const uint2 s0 = deep_gemm::nvfp4::dequant_nvfp4_to_fp8_pair_with_lut(fp4_regs[i0], lut);
        const uint2 s1 = deep_gemm::nvfp4::dequant_nvfp4_to_fp8_pair_with_lut(fp4_regs[i1], lut);
        const uint32_t logical = i0 * 8;
        const uint32_t physical = logical ^ row_swizzle;
        *reinterpret_cast<uint4*>(fp8_dst_base + physical) =
            make_uint4(s0.x, s0.y, s1.x, s1.y);
    }
}

template <uint32_t kNumMathThreads>
__device__ __forceinline__ void dequant_smem_b_inplace_row_strided(uint8_t* __restrict__ smem_b,
                                                                   uint32_t tid_in_wg,
                                                                   const uint8_t* __restrict__ scale_row_base,
                                                                   const uint2* __restrict__ lut_smem) {
    (void)kNumMathThreads;
    const uint32_t* __restrict__ fp4_src = reinterpret_cast<const uint32_t*>(smem_b + tid_in_wg * 128);
    uint32_t fp4_regs[16];
    #pragma unroll
    for (int i = 0; i < 16; ++i) fp4_regs[i] = fp4_src[i];

    const uint2 scale_words = *reinterpret_cast<const uint2*>(scale_row_base);
    const uint32_t scale_word_lo = scale_words.x;
    const uint32_t scale_word_hi = scale_words.y;

    uint8_t* __restrict__ fp8_dst_base = smem_b + tid_in_wg * 128;
    const uint32_t row_swizzle = (tid_in_wg & 7u) << 4;
    #pragma unroll
    for (int scale_i = 0; scale_i < 8; ++scale_i) {
        const uint32_t scale_word = scale_i < 4 ? scale_word_lo : scale_word_hi;
        const uint32_t scale_ue4m3 = (scale_word >> ((scale_i & 3) * 8)) & 0xffu;
        const int i0 = scale_i * 2;
        const int i1 = i0 + 1;
        const uint2 lut = lut_smem[scale_ue4m3 & 0x7fu];
        const uint2 s0 = deep_gemm::nvfp4::dequant_nvfp4_to_fp8_pair_with_lut(fp4_regs[i0], lut);
        const uint2 s1 = deep_gemm::nvfp4::dequant_nvfp4_to_fp8_pair_with_lut(fp4_regs[i1], lut);
        const uint32_t logical = i0 * 8;
        const uint32_t physical = logical ^ row_swizzle;
        *reinterpret_cast<uint4*>(fp8_dst_base + physical) =
            make_uint4(s0.x, s0.y, s1.x, s1.y);
    }
}

template <uint32_t kNumDequantThreads, uint32_t kBarIdx>
__device__ __forceinline__ void dequant_smem_b_inplace_two_rows(
        uint8_t* __restrict__ smem_b, uint32_t tid,
        const uint8_t* __restrict__ scale_row_base0,
        const uint8_t* __restrict__ scale_row_base1,
        const uint2* __restrict__ lut_smem) {
    const uint32_t row0 = tid;
    const uint32_t row1 = tid + kNumDequantThreads;

    const uint32_t* __restrict__ fp4_src0 = reinterpret_cast<const uint32_t*>(smem_b + row0 * 64);
    const uint32_t* __restrict__ fp4_src1 = reinterpret_cast<const uint32_t*>(smem_b + row1 * 64);
    uint32_t fp4_regs0[16];
    uint32_t fp4_regs1[16];
    #pragma unroll
    for (int i = 0; i < 16; ++i) {
        fp4_regs0[i] = fp4_src0[i];
        fp4_regs1[i] = fp4_src1[i];
    }

    const uint2 scale_words0 = *reinterpret_cast<const uint2*>(scale_row_base0);
    const uint2 scale_words1 = *reinterpret_cast<const uint2*>(scale_row_base1);
    const uint32_t scale_word0_lo = scale_words0.x;
    const uint32_t scale_word0_hi = scale_words0.y;
    const uint32_t scale_word1_lo = scale_words1.x;
    const uint32_t scale_word1_hi = scale_words1.y;

    asm volatile("bar.sync %0, %1;" : : "n"(kBarIdx), "n"(kNumDequantThreads));

    uint8_t* __restrict__ fp8_dst_base0 = smem_b + row0 * 128;
    uint8_t* __restrict__ fp8_dst_base1 = smem_b + row1 * 128;
    const uint32_t row_swizzle = (tid & 7u) << 4;
    #pragma unroll
    for (int scale_i = 0; scale_i < 8; ++scale_i) {
        const uint32_t scale_word0 = scale_i < 4 ? scale_word0_lo : scale_word0_hi;
        const uint32_t scale_word1 = scale_i < 4 ? scale_word1_lo : scale_word1_hi;
        const uint32_t scale_ue4m3_0 = (scale_word0 >> ((scale_i & 3) * 8)) & 0xffu;
        const uint32_t scale_ue4m3_1 = (scale_word1 >> ((scale_i & 3) * 8)) & 0xffu;
        const int i0 = scale_i * 2;
        const int i1 = i0 + 1;
        const uint2 lut0 = lut_smem[scale_ue4m3_0 & 0x7fu];
        const uint2 lut1 = lut_smem[scale_ue4m3_1 & 0x7fu];
        const uint2 r0_s0 = deep_gemm::nvfp4::dequant_nvfp4_to_fp8_pair_with_lut(fp4_regs0[i0], lut0);
        const uint2 r0_s1 = deep_gemm::nvfp4::dequant_nvfp4_to_fp8_pair_with_lut(fp4_regs0[i1], lut0);
        const uint2 r1_s0 = deep_gemm::nvfp4::dequant_nvfp4_to_fp8_pair_with_lut(fp4_regs1[i0], lut1);
        const uint2 r1_s1 = deep_gemm::nvfp4::dequant_nvfp4_to_fp8_pair_with_lut(fp4_regs1[i1], lut1);
        const uint32_t logical = i0 * 8;
        const uint32_t physical = logical ^ row_swizzle;
        *reinterpret_cast<uint4*>(fp8_dst_base0 + physical) =
            make_uint4(r0_s0.x, r0_s0.y, r0_s1.x, r0_s1.y);
        *reinterpret_cast<uint4*>(fp8_dst_base1 + physical) =
            make_uint4(r1_s0.x, r1_s0.y, r1_s1.x, r1_s1.y);
    }
}

template <uint32_t kNumDequantThreads, uint32_t kBarIdx, uint32_t kFusedRowBytes = 80>
__device__ __forceinline__ void dequant_smem_b_inplace_two_rows_fused_scale(
        uint8_t* __restrict__ smem_b, uint32_t tid,
        const uint2* __restrict__ lut_smem) {
    const uint32_t row0 = tid;
    const uint32_t row1 = tid + kNumDequantThreads;

    const uint8_t* __restrict__ row_ptr0 = smem_b + row0 * kFusedRowBytes;
    const uint8_t* __restrict__ row_ptr1 = smem_b + row1 * kFusedRowBytes;
    const uint32_t* __restrict__ fp4_src0 = reinterpret_cast<const uint32_t*>(row_ptr0);
    const uint32_t* __restrict__ fp4_src1 = reinterpret_cast<const uint32_t*>(row_ptr1);
    uint32_t fp4_regs0[16];
    uint32_t fp4_regs1[16];
    #pragma unroll
    for (int i = 0; i < 16; ++i) {
        fp4_regs0[i] = fp4_src0[i];
        fp4_regs1[i] = fp4_src1[i];
    }

    const uint2 scale_words0 = *reinterpret_cast<const uint2*>(row_ptr0 + 64);
    const uint2 scale_words1 = *reinterpret_cast<const uint2*>(row_ptr1 + 64);
    const uint32_t scale_word0_lo = scale_words0.x;
    const uint32_t scale_word0_hi = scale_words0.y;
    const uint32_t scale_word1_lo = scale_words1.x;
    const uint32_t scale_word1_hi = scale_words1.y;

    asm volatile("bar.sync %0, %1;" : : "n"(kBarIdx), "n"(kNumDequantThreads));

    uint8_t* __restrict__ fp8_dst_base0 = smem_b + row0 * 128;
    uint8_t* __restrict__ fp8_dst_base1 = smem_b + row1 * 128;
    const uint32_t row_swizzle = (tid & 7u) << 4;
    #pragma unroll
    for (int scale_i = 0; scale_i < 8; ++scale_i) {
        const uint32_t scale_word0 = scale_i < 4 ? scale_word0_lo : scale_word0_hi;
        const uint32_t scale_word1 = scale_i < 4 ? scale_word1_lo : scale_word1_hi;
        const uint32_t scale_ue4m3_0 = (scale_word0 >> ((scale_i & 3) * 8)) & 0xffu;
        const uint32_t scale_ue4m3_1 = (scale_word1 >> ((scale_i & 3) * 8)) & 0xffu;
        const int i0 = scale_i * 2;
        const int i1 = i0 + 1;
        const uint2 lut0 = lut_smem[scale_ue4m3_0 & 0x7fu];
        const uint2 lut1 = lut_smem[scale_ue4m3_1 & 0x7fu];
        const uint2 r0_s0 = deep_gemm::nvfp4::dequant_nvfp4_to_fp8_pair_with_lut(fp4_regs0[i0], lut0);
        const uint2 r0_s1 = deep_gemm::nvfp4::dequant_nvfp4_to_fp8_pair_with_lut(fp4_regs0[i1], lut0);
        const uint2 r1_s0 = deep_gemm::nvfp4::dequant_nvfp4_to_fp8_pair_with_lut(fp4_regs1[i0], lut1);
        const uint2 r1_s1 = deep_gemm::nvfp4::dequant_nvfp4_to_fp8_pair_with_lut(fp4_regs1[i1], lut1);
        const uint32_t logical = i0 * 8;
        const uint32_t physical = logical ^ row_swizzle;
        *reinterpret_cast<uint4*>(fp8_dst_base0 + physical) =
            make_uint4(r0_s0.x, r0_s0.y, r0_s1.x, r0_s1.y);
        *reinterpret_cast<uint4*>(fp8_dst_base1 + physical) =
            make_uint4(r1_s0.x, r1_s0.y, r1_s1.x, r1_s1.y);
    }
}

}  // namespace nvfp4
}  // namespace deep_gemm

namespace deep_gemm {

// ============================================================================
// SM90 (Hopper) FP8 MegaMoE — full implementation
// ----------------------------------------------------------------------------
// Pipeline (cluster=1, no TMA multicast):
//   * Dispatch warps: pull tokens (FP8) and SF (per-128 channel float) from
//     remote ranks via NVLink into the local L1 pool.
//   * GEMM TMA-load warps (1 for A+SFA, 1 for B+SFB) feed the pipeline stages.
//   * Math warpgroups (1 or 2, totalling kNumEpilogueThreads) consume each
//     stage with WGMMA, accumulate into registers, then run the epilogue:
//       - L1 (Linear1): SwiGLU with gate/up granularity-8 interleaved layout,
//         per-row amax over the 64 post-SwiGLU columns of this block, FP8 e4m3
//         quantize, STSM into SMEM, TMA store to local L1 output buffer.
//         The per-row SF is written as a *float* into the L2-acts SF buffer at
//         per-64 K granularity (one SF per L1 N block), so each block is fully
//         self-contained and no cross-CTA amax synchronisation is needed.
//       - L2 (Linear2): BF16 cast of the GEMM output, STSM into SMEM, then
//         NVLink scatter to remote combine buffers.
//   * After all GEMM blocks, the math warps run the COMBINE step (top-k
//     reduction in BF16) — ported verbatim from the SM100 kernel.
// ============================================================================

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
sm90_nvfp4_mega_moe_fused_impl(void* y,
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
#include <deep_gemm/impls/sm90_nvfp4_mega_moe_fused_body.inl>
}


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
    bool kL2ArrivalCounterRequested = false,
    bool kLoaderDequantRequested = false,
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
sm90_nvfp4_mega_moe_split_l1_impl(int* cumulative_local_expert_recv_stats,
                       const uint32_t num_tokens,
                       const __grid_constant__ layout::SymBuffer<kNumRanks> sym_buffer,
                       const __grid_constant__ cute::TmaDescriptor tensor_map_l1_acts,
                       const __grid_constant__ cute::TmaDescriptor tensor_map_l1_acts_sf,
                       const __grid_constant__ cute::TmaDescriptor tensor_map_l1_weights,
                       const __grid_constant__ cute::TmaDescriptor tensor_map_l1_output,
                       const float* __restrict__ l1_global_scales) {
#include <deep_gemm/impls/sm90_nvfp4_mega_moe_split_l1_body.inl>
}


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
    bool kL2DualAccumRequested = false,
    bool kPhaseProfileRequested = false,
    bool kLoaderDequantRequested = false,
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
sm90_nvfp4_mega_moe_split_l2_impl(void* y,
                       int* cumulative_local_expert_recv_stats,
                       const uint32_t num_tokens,
                       const __grid_constant__ layout::SymBuffer<kNumRanks> sym_buffer,
                       const __grid_constant__ cute::TmaDescriptor tensor_map_l2_acts,
                       const __grid_constant__ cute::TmaDescriptor tensor_map_l2_acts_sf,
                       const __grid_constant__ cute::TmaDescriptor tensor_map_l2_weights,
                       const float* __restrict__ l2_global_scales) {
#include <deep_gemm/impls/sm90_nvfp4_mega_moe_split_l2_body.inl>
}


} // namespace deep_gemm

#pragma clang diagnostic pop
