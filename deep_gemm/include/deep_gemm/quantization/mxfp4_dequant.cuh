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

// Apply a precomputed per-byte E8M0 exponent offset to four unsigned FP8 magnitudes.
// scale = 2^(e8m0 - 127), and FP4->FP8 conversion already implies a 2^-6 factor,
// so the byte exponent offset is (e8m0 - 121) << 3.

template <bool kScaleAdd>
DG_W4A8_INLINE std::uint32_t apply_e8m0_shift_to_fp8_mag_precomputed(
    std::uint32_t mag, std::uint32_t off_packed) {
    if constexpr (kScaleAdd) {
        mag = __vaddus4(mag, off_packed);
        mag = __vminu4(mag, 0x7e7e7e7e);
    } else {
        mag = __vsubus4(mag, off_packed);
    }
    return mag & 0x7f7f7f7f;
}

template <bool kScaleAdd>
DG_W4A8_INLINE void make_scaled_e2m1_lookup_tables(
    std::uint32_t off_packed, std::uint32_t& scaled_lo, std::uint32_t& scaled_hi) {
    // Unscaled E2M1 magnitude base table for selectors 0..7:
    //   {0, 0, 8, 12, 16, 20, 24, 28}
    // Entry 0 is true zero and must stay zero after scale-add. Entry 1 shares
    // the same unscaled base but should receive the E8M0 shift.
    constexpr std::uint32_t kBaseLo = 0x0c080000u;
    constexpr std::uint32_t kBaseHi = 0x1c181410u;
    scaled_lo = apply_e8m0_shift_to_fp8_mag_precomputed<kScaleAdd>(kBaseLo, off_packed);
    scaled_hi = apply_e8m0_shift_to_fp8_mag_precomputed<kScaleAdd>(kBaseHi, off_packed);
    scaled_lo &= 0xffffff00u;
}

DG_W4A8_INLINE void dequant_mxfp4_to_fp8_with_scaled_lookup(
    int q, std::uint32_t scaled_lo, std::uint32_t scaled_hi, __nv_fp8x4_e4m3* frag_b) {
    const std::uint32_t uq = static_cast<std::uint32_t>(q);
    const std::uint32_t sel_hi = ((uq >> 4) & 0x00000007u) |
                                 ((uq >> 8) & 0x00000070u) |
                                 ((uq >> 12) & 0x00000700u) |
                                 ((uq >> 16) & 0x00007000u);
    const std::uint32_t sel_lo = (uq & 0x00000007u) |
                                 ((uq >> 4) & 0x00000070u) |
                                 ((uq >> 8) & 0x00000700u) |
                                 ((uq >> 12) & 0x00007000u);
    std::uint32_t out_hi = __byte_perm(scaled_lo, scaled_hi, sel_hi);
    std::uint32_t out_lo = __byte_perm(scaled_lo, scaled_hi, sel_lo);
    out_hi |= uq & 0x80808080;
    out_lo |= (uq << 4) & 0x80808080;
    frag_b[1] = *reinterpret_cast<const __nv_fp8x4_e4m3*>(&out_hi);
    frag_b[0] = *reinterpret_cast<const __nv_fp8x4_e4m3*>(&out_lo);
}

#undef DG_W4A8_INLINE

} // namespace w4a8
} // namespace deep_gemm
