// SPDX-License-Identifier: MIT
//
// MXFP4 (E2M1 + E8M0 scale) -> FP8 (E4M3) dequant helper for SM90
// fused MegaMoE. The packed FP4 layout matches the existing Marlin-style
// byte packing used by the predecessor FP4 path, with MXFP4 E8M0
// power-of-two scale bytes applied per 32 K elements.
//
// The E8M0 micro-scale is a pure power of two (value = 2^(code - 127)), so
// folding it into the FP8 E4M3 magnitude is an exact exponent shift for the
// eight fixed E2M1 magnitudes {0, .5, 1, 1.5, 2, 3, 4, 6}: within FP8 range the
// product needs no rounding, and out-of-range values saturate (SATFINITE) or
// flush to zero exactly like `torch.float8_e4m3fn`. We therefore synthesise the
// 8-byte per-scale LUT entry arithmetically instead of shipping a 256-row table.

#pragma once

#include <cuda_fp8.h>
#include <cstdint>

namespace deep_gemm {
namespace mxfp4 {

#define DG_MXFP4_INLINE __device__ __forceinline__

// The eight positive E2M1 magnitudes indexed by the 3-bit magnitude code.
static __device__ __constant__ const float kE2M1Magnitudes[8] = {
    0.0f, 0.5f, 1.0f, 1.5f, 2.0f, 3.0f, 4.0f, 6.0f};

// The smem fold LUT stores a 128-row window over E8M0 codes [kE8M0LutBase,
// kE8M0LutBase + kE8M0LutCount - 1] = [72, 199]. Every FP4 magnitude underflows
// FP8 to zero for codes <= ~114 and saturates to 0x7f for codes >= ~137, so the
// LUT entry is *constant* below and above the window. Clamping any E8M0 code
// into the window is therefore bit-exact for all 256 codes, and keeps the LUT
// at 1KB (128 * uint2) instead of the full 2KB / 256-row table.
static constexpr std::uint32_t kE8M0LutBase  = 72u;
static constexpr std::uint32_t kE8M0LutCount = 128u;

// Map an 8-bit E8M0 code to its row in the windowed smem LUT.
DG_MXFP4_INLINE std::uint32_t e8m0_lut_index(std::uint32_t code) {
    constexpr std::uint32_t kHi = kE8M0LutBase + kE8M0LutCount - 1u;  // 199
    const std::uint32_t lo = code < kE8M0LutBase ? kE8M0LutBase : code;
    return (lo > kHi ? kHi : lo) - kE8M0LutBase;
}

// Branchless: 3-bit E2M1 magnitude code -> FP16 bit pattern.
// mag3 in 0..7 maps to FP4 values {0, 0.5, 1, 1.5, 2, 3, 4, 6}.
DG_MXFP4_INLINE uint16_t e2m1_mag_to_fp16_bits(std::uint32_t mag3) {
    std::uint32_t exp_raw  = mag3 >> 1u;
    std::uint32_t mant_raw = mag3 & 1u;
    std::uint32_t is_norm  = (exp_raw != 0u) ? 1u : 0u;
    // Normal:  fp16 = ((exp_raw+14)<<10) | (mant_raw<<9)
    // Subnorm (mag3=1, value=0.5): fp16 = 14<<10 = 0x3800 (is_norm=0 zeroes mant)
    // Zero (mag3=0): masked to 0
    std::uint32_t fp16 = ((exp_raw + 14u) << 10u) | ((mant_raw & is_norm) << 9u);
    return static_cast<uint16_t>(fp16 * static_cast<std::uint32_t>(mag3 != 0u));
}

// Build the 8-byte (uint2) FP8 magnitude LUT entry for a single E8M0 scale code.
// Byte layout matches the byte_perm-based decoders: magnitudes 0..3 pack into
// .x and 4..7 into .y. The E8M0 scale is 2^(code - 127); code 0xff (E8M0 NaN)
// is never emitted by the quantizer and simply saturates here.
DG_MXFP4_INLINE uint2 load_e2m1_e8m0_lut(std::uint32_t scale_e8m0) {
    const float scale = exp2f(static_cast<float>(
        static_cast<int>(scale_e8m0 & 0xffu) - 127));
    uint8_t fp8_mag[8];
    // NOSAT (not SATFINITE) matches torch.float8_e4m3fn: overflow rounds to the
    // 0x7f slot rather than clamping to the 0x7e (448) max finite value.
#pragma unroll
    for (int m = 0; m < 8; ++m) {
        fp8_mag[m] = static_cast<uint8_t>(__nv_cvt_float_to_fp8(
            kE2M1Magnitudes[m] * scale, __NV_NOSAT, __NV_E4M3));
    }
    uint2 lut;
    lut.x = static_cast<std::uint32_t>(fp8_mag[0]) |
            (static_cast<std::uint32_t>(fp8_mag[1]) << 8u) |
            (static_cast<std::uint32_t>(fp8_mag[2]) << 16u) |
            (static_cast<std::uint32_t>(fp8_mag[3]) << 24u);
    lut.y = static_cast<std::uint32_t>(fp8_mag[4]) |
            (static_cast<std::uint32_t>(fp8_mag[5]) << 8u) |
            (static_cast<std::uint32_t>(fp8_mag[6]) << 16u) |
            (static_cast<std::uint32_t>(fp8_mag[7]) << 24u);
    return lut;
}

template <bool kUseDp4a>
DG_MXFP4_INLINE std::uint32_t pack_mxfp4_magnitude_selector(std::uint32_t byte_magnitudes) {
    if constexpr (kUseDp4a) {
        const std::uint32_t lo = __dp4a(byte_magnitudes, 0x00001001u, 0u);
        const std::uint32_t hi = __dp4a(byte_magnitudes, 0x10010000u, 0u);
        return lo + (hi << 8);
    } else {
        const std::uint32_t packed_pairs = byte_magnitudes + (byte_magnitudes >> 4);
        return __byte_perm(packed_pairs, 0u, 0x4420u);
    }
}

DG_MXFP4_INLINE std::uint32_t byte_perm_unchecked(std::uint32_t a, std::uint32_t b,
                                                  std::uint32_t selector) {
    // Callers provide 0..7 selector nibbles; raw PTX avoids a redundant 0x7777 mask.
    std::uint32_t out;
    asm("prmt.b32 %0, %1, %2, %3;" : "=r"(out) : "r"(a), "r"(b), "r"(selector));
    return out;
}

template <bool kUseDp4aHi = false, bool kUseDp4aLo = kUseDp4aHi>
DG_MXFP4_INLINE uint2 dequant_mxfp4_to_fp8_pair_with_lut(std::uint32_t uq, const uint2& lut) {
    const std::uint32_t sel_hi =
        pack_mxfp4_magnitude_selector<kUseDp4aHi>((uq >> 4) & 0x07070707u);
    const std::uint32_t sel_lo =
        pack_mxfp4_magnitude_selector<kUseDp4aLo>(uq & 0x07070707u);

    std::uint32_t out_hi = byte_perm_unchecked(lut.x, lut.y, sel_hi);
    std::uint32_t out_lo = byte_perm_unchecked(lut.x, lut.y, sel_lo);
    out_hi |= uq & 0x80808080u;
    out_lo |= (uq << 4) & 0x80808080u;
    return make_uint2(out_hi, out_lo);
}

DG_MXFP4_INLINE uint2 dequant_mxfp4_to_fp8_pair(std::uint32_t q, std::uint32_t scale_e8m0) {
    return dequant_mxfp4_to_fp8_pair_with_lut(q, load_e2m1_e8m0_lut(scale_e8m0));
}

#undef DG_MXFP4_INLINE

} // namespace mxfp4
} // namespace deep_gemm
