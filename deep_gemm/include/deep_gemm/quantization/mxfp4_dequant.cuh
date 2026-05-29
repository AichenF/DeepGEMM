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

// Fused dequant + E8M0 scale apply via per-byte integer FP8 exponent shift.
// scale = 2^(e8m0 - 127), and FP4->FP8 conversion already implies a 2^-6 factor,
// so we need to add (e8m0 - 121) to each FP8 byte's exponent field (bits 6:3 → +delta*8).
//   delta >= 0: per-byte unsigned saturating ADD of (delta*8). Within [0,12]
//               byte add stays in 7-bit range. delta>12 → saturate to 0x7E.
//   delta <  0: per-byte unsigned saturating SUB via __vsubus4. Bytes that
//               underflow to subnormal range (result_exp == 0) are zeroed to
//               match FP-multiply FP8 round-to-zero behavior.
// e8m0 == 0 → zero output.
DG_W4A8_INLINE std::uint32_t apply_e8m0_shift_to_fp8_mag(std::uint32_t mag,
                                                                std::uint8_t e8m0) {
    if (e8m0 >= 121u) {
        const std::uint32_t off = (static_cast<std::uint32_t>(e8m0) - 121u) << 3;
        const std::uint32_t off_packed = off | (off << 8) | (off << 16) | (off << 24);
        mag = __vaddus4(mag, off_packed);
        mag = __vminu4(mag, 0x7e7e7e7e);
    } else {
        const std::uint32_t absdelta = 121u - static_cast<std::uint32_t>(e8m0);
        const std::uint32_t off = (absdelta >= 16u) ? 0x78u : (absdelta << 3);
        const std::uint32_t off_packed = off | (off << 8) | (off << 16) | (off << 24);
        mag = __vsubus4(mag, off_packed);
    }
    return mag & 0x7f7f7f7f;
}

DG_W4A8_INLINE std::uint32_t correct_e2m1_base_and_apply_scale(std::uint32_t out,
                                                               std::uint32_t mag,
                                                               std::uint8_t e8m0) {
    const std::uint32_t sign = out & 0x80808080;
    std::uint32_t val = out & 0x7f7f7f7f;

    // The fast bit-shift path gives mag * 0.75 for mag=1 and turns mag=0
    // into a non-zero value after exponent add. Correct those two E2M1 cases
    // while keeping the vectorized byte path for mag>=2.
    const std::uint32_t zero_mask = __vcmpeq4(mag, 0x00000000);
    const std::uint32_t one_mask = __vcmpeq4(mag, 0x01010101);
    val = __vsubus4(val, one_mask & 0x04040404);
    val = apply_e8m0_shift_to_fp8_mag(val, e8m0);
    val &= ~zero_mask;
    return sign | val;
}

DG_W4A8_INLINE void dequant_mxfp4_to_fp8_with_int_scale(int q, std::uint8_t e8m0,
                                                        __nv_fp8x4_e4m3* frag_b) {
    constexpr std::uint32_t MASK = 0x70707070;
    const std::uint32_t uq = static_cast<std::uint32_t>(q);
    const std::uint32_t q_hi = uq;
    const std::uint32_t q_lo = uq << 4;

    std::uint32_t out_hi = (q_hi & 0x80808080) | ((q_hi & MASK) >> 2);
    std::uint32_t out_lo = (q_lo & 0x80808080) | ((q_lo & MASK) >> 2);
    const std::uint32_t mag_hi = (uq >> 4) & 0x07070707;
    const std::uint32_t mag_lo = uq & 0x07070707;

    out_hi = correct_e2m1_base_and_apply_scale(out_hi, mag_hi, e8m0);
    out_lo = correct_e2m1_base_and_apply_scale(out_lo, mag_lo, e8m0);

    frag_b[1] = *reinterpret_cast<const __nv_fp8x4_e4m3*>(&out_hi);
    frag_b[0] = *reinterpret_cast<const __nv_fp8x4_e4m3*>(&out_lo);
}

#undef DG_W4A8_INLINE

} // namespace w4a8
} // namespace deep_gemm
