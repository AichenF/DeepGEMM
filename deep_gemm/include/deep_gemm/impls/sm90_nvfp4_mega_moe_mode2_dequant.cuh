#pragma once

#include <deep_gemm/quantization/nvfp4_dequant.cuh>

namespace deep_gemm::nvfp4 {

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

template <int kQuad, bool kQuadILP>
__device__ __forceinline__ void dequant_mode2_nibble_row_lut_window(
        uint8_t* __restrict__ fp8_dst,
        const uint4 (&fp4_quads)[4],
        const uint2& scale_words,
        const uint32_t row_swizzle,
        const uint2* __restrict__ lut_smem,
        const uint2 lut0,
        const uint2 lut1) {
    uint2 next_lut0;
    uint2 next_lut1;
    if constexpr (kQuad + 1 < 4) {
        constexpr int kNextScaleI0 = (kQuad + 1) * 2;
        constexpr int kNextScaleI1 = kNextScaleI0 + 1;
        const uint32_t next_scale_word =
            kQuad + 1 < 2 ? scale_words.x : scale_words.y;
        const uint32_t next_scale0 =
            (next_scale_word >> ((kNextScaleI0 & 3) * 8)) & 0x7fu;
        const uint32_t next_scale1 =
            (next_scale_word >> ((kNextScaleI1 & 3) * 8)) & 0x7fu;
        next_lut0 = lut_smem[next_scale0];
        next_lut1 = lut_smem[next_scale1];
    }

    const uint4 q = fp4_quads[kQuad];
    constexpr int kScaleI0 = kQuad * 2;
    constexpr int kScaleI1 = kScaleI0 + 1;
    const uint2 q0 = dequant_mode2_nibble_word(q.x, lut0);
    const uint2 q1 = dequant_mode2_nibble_word(q.y, lut0);
    if constexpr (!kQuadILP) {
        *reinterpret_cast<uint4*>(
            fp8_dst + ((kScaleI0 * 16) ^ row_swizzle)) =
            make_uint4(q0.x, q0.y, q1.x, q1.y);
    }

    const uint2 q2 = dequant_mode2_nibble_word(q.z, lut1);
    const uint2 q3 = dequant_mode2_nibble_word(q.w, lut1);
    if constexpr (kQuadILP) {
        *reinterpret_cast<uint4*>(
            fp8_dst + ((kScaleI0 * 16) ^ row_swizzle)) =
            make_uint4(q0.x, q0.y, q1.x, q1.y);
    }
    *reinterpret_cast<uint4*>(
        fp8_dst + ((kScaleI1 * 16) ^ row_swizzle)) =
        make_uint4(q2.x, q2.y, q3.x, q3.y);

    if constexpr (kQuad + 1 < 4) {
        dequant_mode2_nibble_row_lut_window<kQuad + 1, kQuadILP>(
            fp8_dst, fp4_quads, scale_words, row_swizzle,
            lut_smem, next_lut0, next_lut1);
    }
}

template <bool kQuadILP = false>
__device__ __forceinline__ void dequant_smem_b_from_packed_mode2_nibble(
        uint8_t* __restrict__ smem_b,
        const uint8_t* __restrict__ packed_b,
        const uint32_t row,
        const uint2* __restrict__ lut_smem) {
    const uint8_t* __restrict__ row_ptr = packed_b + row * 80;
    const uint2 scale_words =
        *reinterpret_cast<const uint2*>(row_ptr + 64);
    const uint2 lut0 = lut_smem[scale_words.x & 0x7fu];
    const uint2 lut1 = lut_smem[(scale_words.x >> 8) & 0x7fu];
    const uint4* __restrict__ fp4_src =
        reinterpret_cast<const uint4*>(row_ptr);
    uint4 fp4_quads[4];
#pragma unroll
    for (int i = 0; i < 4; ++i)
        fp4_quads[i] = fp4_src[i];
    dequant_mode2_nibble_row_lut_window<0, kQuadILP>(
        smem_b + row * 128, fp4_quads, scale_words,
        (row & 7u) << 4, lut_smem, lut0, lut1);
}

template <int kQuad>
__device__ __forceinline__ void dequant_mode2_nibble_two_rows_lut_window(
        uint8_t* __restrict__ fp8_dst0,
        uint8_t* __restrict__ fp8_dst1,
        const uint4 (&fp4_quads0)[4],
        const uint4 (&fp4_quads1)[4],
        const uint2& scale_words0,
        const uint2& scale_words1,
        const uint32_t row_swizzle,
        const uint2* __restrict__ lut_smem,
        const uint2 lut00,
        const uint2 lut10,
        const uint2 lut01,
        const uint2 lut11) {
    uint2 next_lut00;
    uint2 next_lut10;
    uint2 next_lut01;
    uint2 next_lut11;
    if constexpr (kQuad + 1 < 4) {
        constexpr int kNextScaleI0 = (kQuad + 1) * 2;
        constexpr int kNextScaleI1 = kNextScaleI0 + 1;
        const uint32_t next_scale_word0 =
            kQuad + 1 < 2 ? scale_words0.x : scale_words0.y;
        const uint32_t next_scale_word1 =
            kQuad + 1 < 2 ? scale_words1.x : scale_words1.y;
        const uint32_t next_scale00 =
            (next_scale_word0 >> ((kNextScaleI0 & 3) * 8)) & 0x7fu;
        const uint32_t next_scale10 =
            (next_scale_word1 >> ((kNextScaleI0 & 3) * 8)) & 0x7fu;
        const uint32_t next_scale01 =
            (next_scale_word0 >> ((kNextScaleI1 & 3) * 8)) & 0x7fu;
        const uint32_t next_scale11 =
            (next_scale_word1 >> ((kNextScaleI1 & 3) * 8)) & 0x7fu;
        next_lut00 = lut_smem[next_scale00];
        next_lut10 = lut_smem[next_scale10];
        next_lut01 = lut_smem[next_scale01];
        next_lut11 = lut_smem[next_scale11];
    }

    const uint4 q0 = fp4_quads0[kQuad];
    const uint4 q1 = fp4_quads1[kQuad];
    constexpr int kScaleI0 = kQuad * 2;
    constexpr int kScaleI1 = kScaleI0 + 1;
    const uint2 q0x = dequant_mode2_nibble_word(q0.x, lut00);
    const uint2 q0y = dequant_mode2_nibble_word(q0.y, lut00);
    const uint2 q1x = dequant_mode2_nibble_word(q1.x, lut10);
    const uint2 q1y = dequant_mode2_nibble_word(q1.y, lut10);
    *reinterpret_cast<uint4*>(
        fp8_dst0 + ((kScaleI0 * 16) ^ row_swizzle)) =
        make_uint4(q0x.x, q0x.y, q0y.x, q0y.y);
    *reinterpret_cast<uint4*>(
        fp8_dst1 + ((kScaleI0 * 16) ^ row_swizzle)) =
        make_uint4(q1x.x, q1x.y, q1y.x, q1y.y);

    const uint2 q0z = dequant_mode2_nibble_word(q0.z, lut01);
    const uint2 q0w = dequant_mode2_nibble_word(q0.w, lut01);
    const uint2 q1z = dequant_mode2_nibble_word(q1.z, lut11);
    const uint2 q1w = dequant_mode2_nibble_word(q1.w, lut11);
    *reinterpret_cast<uint4*>(
        fp8_dst0 + ((kScaleI1 * 16) ^ row_swizzle)) =
        make_uint4(q0z.x, q0z.y, q0w.x, q0w.y);
    *reinterpret_cast<uint4*>(
        fp8_dst1 + ((kScaleI1 * 16) ^ row_swizzle)) =
        make_uint4(q1z.x, q1z.y, q1w.x, q1w.y);

    if constexpr (kQuad + 1 < 4) {
        dequant_mode2_nibble_two_rows_lut_window<kQuad + 1>(
            fp8_dst0, fp8_dst1, fp4_quads0, fp4_quads1,
            scale_words0, scale_words1, row_swizzle, lut_smem,
            next_lut00, next_lut10, next_lut01, next_lut11);
    }
}

template <uint32_t kNumDequantThreads, uint32_t kBarIdx,
          bool kSyncAfter = false, uint32_t kFusedRowBytes = 80>
__device__ __forceinline__ void dequant_smem_b_inplace_two_rows_mode2_nibble(
        uint8_t* __restrict__ smem_b,
        const uint32_t tid,
        const uint2* __restrict__ lut_smem) {
    const uint32_t row0 = tid;
    const uint32_t row1 = tid + kNumDequantThreads;
    const uint8_t* __restrict__ row_ptr0 =
        smem_b + row0 * kFusedRowBytes;
    const uint8_t* __restrict__ row_ptr1 =
        smem_b + row1 * kFusedRowBytes;
    const uint2 scale_words0 =
        *reinterpret_cast<const uint2*>(row_ptr0 + 64);
    const uint2 scale_words1 =
        *reinterpret_cast<const uint2*>(row_ptr1 + 64);
    const uint32_t scale00 = scale_words0.x & 0x7fu;
    const uint32_t scale10 = scale_words1.x & 0x7fu;
    const uint32_t scale01 = (scale_words0.x >> 8) & 0x7fu;
    const uint32_t scale11 = (scale_words1.x >> 8) & 0x7fu;
    const uint2 lut00 = lut_smem[scale00];
    const uint2 lut10 = lut_smem[scale10];
    const uint2 lut01 = lut_smem[scale01];
    const uint2 lut11 = lut_smem[scale11];
    const uint4* __restrict__ fp4_src0 =
        reinterpret_cast<const uint4*>(row_ptr0);
    const uint4* __restrict__ fp4_src1 =
        reinterpret_cast<const uint4*>(row_ptr1);
    uint4 fp4_quads0[4];
    uint4 fp4_quads1[4];
#pragma unroll
    for (int i = 0; i < 4; ++i) {
        fp4_quads0[i] = fp4_src0[i];
        fp4_quads1[i] = fp4_src1[i];
    }

    asm volatile("bar.sync %0, %1;"
                 : : "n"(kBarIdx), "n"(kNumDequantThreads));

    uint8_t* __restrict__ fp8_dst0 = smem_b + row0 * 128;
    uint8_t* __restrict__ fp8_dst1 = smem_b + row1 * 128;
    const uint32_t row_swizzle = (tid & 7u) << 4;
    dequant_mode2_nibble_two_rows_lut_window<0>(
        fp8_dst0, fp8_dst1, fp4_quads0, fp4_quads1,
        scale_words0, scale_words1, row_swizzle, lut_smem,
        lut00, lut10, lut01, lut11);
    if constexpr (kSyncAfter) {
        // Every writer must publish its generic-proxy stores before the
        // writer-set rendezvous allows one thread to signal the mbarrier.
        cutlass::arch::fence_view_async_shared();
        asm volatile("bar.sync %0, %1;"
                     : : "n"(kBarIdx), "n"(kNumDequantThreads));
    }
}

}  // namespace deep_gemm::nvfp4
