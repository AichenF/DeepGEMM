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
DG_W4A8_INLINE void dequant_mxfp4_to_fp8_with_int_scale(int q, std::uint8_t e8m0,
                                                        __nv_fp8x4_e4m3* frag_b) {
    constexpr int MASK = 0x70707070;
    int q_hi = q;
    int q_lo = q << 4;
    int Out1 = (q_hi & 0x80808080) | ((q_hi & MASK) >> 2);
    int Out2 = (q_lo & 0x80808080) | ((q_lo & MASK) >> 2);

    if (e8m0 == 0u) {
        Out1 = Out1 & 0x80808080;
        Out2 = Out2 & 0x80808080;
    } else if (e8m0 >= 121u && e8m0 <= 133u) {
        std::uint32_t delta = static_cast<std::uint32_t>(e8m0) - 121u;
        std::uint32_t off = delta << 3;
        std::uint32_t off_packed = off | (off << 8) | (off << 16) | (off << 24);
        Out1 = (Out1 & 0x80808080) |
               (((Out1 & 0x7F7F7F7F) + off_packed) & 0x7F7F7F7F);
        Out2 = (Out2 & 0x80808080) |
               (((Out2 & 0x7F7F7F7F) + off_packed) & 0x7F7F7F7F);
    } else if (e8m0 < 121u) {
        // delta < 0: per-byte unsigned saturating subtract via __vsubus4.
        // Underflow yields exp=0 (subnormal); minor precision loss accepted.
        std::uint32_t absdelta = 121u - static_cast<std::uint32_t>(e8m0);
        std::uint32_t off = (absdelta >= 16u) ? 0x78u : (absdelta << 3);
        std::uint32_t off_packed = off | (off << 8) | (off << 16) | (off << 24);
        std::uint32_t mag1 = static_cast<std::uint32_t>(Out1) & 0x7F7F7F7F;
        std::uint32_t mag2 = static_cast<std::uint32_t>(Out2) & 0x7F7F7F7F;
        std::uint32_t sub1 = __vsubus4(mag1, off_packed);
        std::uint32_t sub2 = __vsubus4(mag2, off_packed);
        Out1 = (Out1 & 0x80808080) | static_cast<int>(sub1);
        Out2 = (Out2 & 0x80808080) | static_cast<int>(sub2);
    } else {
        Out1 = (Out1 & 0x80808080) | 0x7E7E7E7E;
        Out2 = (Out2 & 0x80808080) | 0x7E7E7E7E;
    }

    frag_b[1] = *reinterpret_cast<const __nv_fp8x4_e4m3*>(&Out1);
    frag_b[0] = *reinterpret_cast<const __nv_fp8x4_e4m3*>(&Out2);
}

#undef DG_W4A8_INLINE

} // namespace w4a8
} // namespace deep_gemm
