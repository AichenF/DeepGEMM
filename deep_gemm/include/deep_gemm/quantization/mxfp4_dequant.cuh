// SPDX-License-Identifier: MIT
//
// MXFP4 (E2M1) → FP8 (E4M3) dequant helper for SM90 W4A8 fused MegaMoE.
// Ported from vLLM Marlin's `dequant<__nv_fp8x4_e4m3, kFE2M1f, true>` and
// adapted with the integer E8M0-scale fast-path inspired by Humming's
// `fused_dequant_single_for_mxfp4<Float8E4M3>`.

#pragma once

#include <cuda_fp8.h>
#include <cstdint>

namespace deep_gemm {
namespace w4a8 {

#define DG_W4A8_INLINE __device__ __forceinline__

// Convert one packed-FP4 dword (8 nibbles = 8 FP4 values) into 8 FP8 E4M3
// bytes (Marlin layout: q<<4 step produces frag_b[0], q produces frag_b[1]).
// No scale is applied here; per-32 E8M0 group scale must be applied on the
// WGMMA accumulator afterwards.
DG_W4A8_INLINE void dequant_mxfp4_to_fp8(int q, __nv_fp8x4_e4m3* frag_b) {
    constexpr int FP4_EXPONENT = 2;
    constexpr int FP8_EXPONENT = 4;
    constexpr int RIGHT_SHIFT = FP8_EXPONENT - FP4_EXPONENT;  // = 2
    constexpr int MASK = 0x70707070;

    int Out1 = (q & 0x80808080) | ((q & MASK) >> RIGHT_SHIFT);
    q <<= 4;
    int Out2 = (q & 0x80808080) | ((q & MASK) >> RIGHT_SHIFT);

    frag_b[1] = *reinterpret_cast<const __nv_fp8x4_e4m3*>(&Out1);
    frag_b[0] = *reinterpret_cast<const __nv_fp8x4_e4m3*>(&Out2);
}

// Fused dequant + E8M0 scale apply: produce FP8 values already scaled by
// the per-32 E8M0 exponent (Humming-style integer trick). `exp_offset` is
// the E8M0 byte for this 32-element K group, biased by 127 - 2 so that the
// final FP8 value matches `value_fp4 * 2^(e - 127 - 2)` after the FP4→FP8
// bit shift.
DG_W4A8_INLINE void dequant_mxfp4_to_fp8_scaled(int q, std::uint8_t e8m0,
                                                 __nv_fp8x4_e4m3* frag_b) {
    constexpr int MASK = 0x70707070;
    // FP4 nibble has 3 magnitude bits in [bit 0..2]; after >>2 they land in
    // FP8 exponent bits [bit 0..2]. We add (e8m0 - 2) as an exponent offset
    // in FP8 to recover the true value. The "-2" compensates for the FP4 →
    // FP8 shift; the -127 of E8M0 cancels with the +127 of FP8 bias since
    // both are stored biased identically.
    // The offset is broadcast to all 4 bytes of a 32-bit lane.
    std::uint32_t offset_packed =
        (static_cast<std::uint32_t>(e8m0) - 2u) & 0xFFu;
    offset_packed |= offset_packed << 8;
    offset_packed |= offset_packed << 16;

    int q_hi = q;
    int q_lo = q << 4;
    int Out1 = (q_hi & 0x80808080) | ((q_hi & MASK) >> 2);
    int Out2 = (q_lo & 0x80808080) | ((q_lo & MASK) >> 2);

    // Integer add into FP8 exponent bits (mask away sign + mantissa to keep
    // exponent-add semantics; this fast path skips clamping for subnormals).
    Out1 = (Out1 & 0x80808080) | (((Out1 & 0x7F7F7F7F) + offset_packed) & 0x7F7F7F7F);
    Out2 = (Out2 & 0x80808080) | (((Out2 & 0x7F7F7F7F) + offset_packed) & 0x7F7F7F7F);

    frag_b[1] = *reinterpret_cast<const __nv_fp8x4_e4m3*>(&Out1);
    frag_b[0] = *reinterpret_cast<const __nv_fp8x4_e4m3*>(&Out2);
}

// Convert an E8M0 byte to a float multiplicative scale: 2 ** (e - 127).
DG_W4A8_INLINE float e8m0_to_float(std::uint8_t e8m0) {
    std::uint32_t bits = static_cast<std::uint32_t>(e8m0) << 23;
    return *reinterpret_cast<const float*>(&bits);
}

#undef DG_W4A8_INLINE

} // namespace w4a8
} // namespace deep_gemm
