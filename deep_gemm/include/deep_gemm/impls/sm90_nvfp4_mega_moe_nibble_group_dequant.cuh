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

}  // namespace deep_gemm::nvfp4
