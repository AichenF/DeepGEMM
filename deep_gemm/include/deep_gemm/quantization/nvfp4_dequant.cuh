// SPDX-License-Identifier: MIT
//
// NVFP4 (E2M1 + UE4M3 scale) -> FP8 (E4M3) dequant helper for SM90
// fused MegaMoE. The packed FP4 layout matches the existing Marlin-style
// byte packing used by the predecessor FP4 path, with NVFP4 UE4M3
// scale bytes applied per 16 K elements.

#pragma once

#include <cuda_fp8.h>
#include <cstdint>

namespace deep_gemm {
namespace nvfp4 {

#define DG_NVFP4_INLINE __device__ __forceinline__

// Per-scale lookup table for positive E2M1 magnitudes 0..7 converted to FP8 after an exact /8 normalization.
// Each row packs magnitudes 0..3 in .x and 4..7 in .y so __byte_perm can map
// four FP4 nibbles to four FP8 bytes with one instruction. Scale code 0x7f is
// treated as the normalized 448 / 8 = 56 fallback; the quantizer emits only
// 0x00..0x7e.
static __device__ __constant__ __align__(16) const uint2 kE2M1AndUe4m3ToFp8Lut[128] = {
    {0x00000000u, 0x00000000u},
    {0x00000000u, 0x01000000u},
    {0x00000000u, 0x02010100u},
    {0x01000000u, 0x02020101u},
    {0x01000000u, 0x03020201u},
    {0x01010000u, 0x04020201u},
    {0x01010000u, 0x04030202u},
    {0x01010000u, 0x05040302u},
    {0x02010000u, 0x06040302u},
    {0x02010100u, 0x07040302u},
    {0x02010100u, 0x08050402u},
    {0x02010100u, 0x08060403u},
    {0x02020100u, 0x09060403u},
    {0x02020100u, 0x0a060503u},
    {0x03020100u, 0x0a070504u},
    {0x03020100u, 0x0b080604u},
    {0x03020100u, 0x0c080604u},
    {0x03020100u, 0x0e090704u},
    {0x04020100u, 0x0f0a0805u},
    {0x04030100u, 0x100b0806u},
    {0x04030200u, 0x110c0906u},
    {0x05030200u, 0x120d0a06u},
    {0x05040200u, 0x120e0a07u},
    {0x06040200u, 0x130f0b08u},
    {0x06040200u, 0x14100c08u},
    {0x07040200u, 0x16110e09u},
    {0x08050200u, 0x17120f0au},
    {0x08060300u, 0x1813100bu},
    {0x09060300u, 0x1914110cu},
    {0x0a060300u, 0x1a15120du},
    {0x0a070400u, 0x1a16120eu},
    {0x0b080400u, 0x1b17130fu},
    {0x0c080400u, 0x1c181410u},
    {0x0e090400u, 0x1e191611u},
    {0x0f0a0500u, 0x1f1a1712u},
    {0x100b0600u, 0x201b1813u},
    {0x110c0600u, 0x211c1914u},
    {0x120d0600u, 0x221d1a15u},
    {0x120e0700u, 0x221e1a16u},
    {0x130f0800u, 0x231f1b17u},
    {0x14100800u, 0x24201c18u},
    {0x16110900u, 0x26211e19u},
    {0x17120a00u, 0x27221f1au},
    {0x18130b00u, 0x2823201bu},
    {0x19140c00u, 0x2924211cu},
    {0x1a150d00u, 0x2a25221du},
    {0x1a160e00u, 0x2a26221eu},
    {0x1b170f00u, 0x2b27231fu},
    {0x1c181000u, 0x2c282420u},
    {0x1e191100u, 0x2e292621u},
    {0x1f1a1200u, 0x2f2a2722u},
    {0x201b1300u, 0x302b2823u},
    {0x211c1400u, 0x312c2924u},
    {0x221d1500u, 0x322d2a25u},
    {0x221e1600u, 0x322e2a26u},
    {0x231f1700u, 0x332f2b27u},
    {0x24201800u, 0x34302c28u},
    {0x26211900u, 0x36312e29u},
    {0x27221a00u, 0x37322f2au},
    {0x28231b00u, 0x3833302bu},
    {0x29241c00u, 0x3934312cu},
    {0x2a251d00u, 0x3a35322du},
    {0x2a261e00u, 0x3a36322eu},
    {0x2b271f00u, 0x3b37332fu},
    {0x2c282000u, 0x3c383430u},
    {0x2e292100u, 0x3e393631u},
    {0x2f2a2200u, 0x3f3a3732u},
    {0x302b2300u, 0x403b3833u},
    {0x312c2400u, 0x413c3934u},
    {0x322d2500u, 0x423d3a35u},
    {0x322e2600u, 0x423e3a36u},
    {0x332f2700u, 0x433f3b37u},
    {0x34302800u, 0x44403c38u},
    {0x36312900u, 0x46413e39u},
    {0x37322a00u, 0x47423f3au},
    {0x38332b00u, 0x4843403bu},
    {0x39342c00u, 0x4944413cu},
    {0x3a352d00u, 0x4a45423du},
    {0x3a362e00u, 0x4a46423eu},
    {0x3b372f00u, 0x4b47433fu},
    {0x3c383000u, 0x4c484440u},
    {0x3e393100u, 0x4e494641u},
    {0x3f3a3200u, 0x4f4a4742u},
    {0x403b3300u, 0x504b4843u},
    {0x413c3400u, 0x514c4944u},
    {0x423d3500u, 0x524d4a45u},
    {0x423e3600u, 0x524e4a46u},
    {0x433f3700u, 0x534f4b47u},
    {0x44403800u, 0x54504c48u},
    {0x46413900u, 0x56514e49u},
    {0x47423a00u, 0x57524f4au},
    {0x48433b00u, 0x5853504bu},
    {0x49443c00u, 0x5954514cu},
    {0x4a453d00u, 0x5a55524du},
    {0x4a463e00u, 0x5a56524eu},
    {0x4b473f00u, 0x5b57534fu},
    {0x4c484000u, 0x5c585450u},
    {0x4e494100u, 0x5e595651u},
    {0x4f4a4200u, 0x5f5a5752u},
    {0x504b4300u, 0x605b5853u},
    {0x514c4400u, 0x615c5954u},
    {0x524d4500u, 0x625d5a55u},
    {0x524e4600u, 0x625e5a56u},
    {0x534f4700u, 0x635f5b57u},
    {0x54504800u, 0x64605c58u},
    {0x56514900u, 0x66615e59u},
    {0x57524a00u, 0x67625f5au},
    {0x58534b00u, 0x6863605bu},
    {0x59544c00u, 0x6964615cu},
    {0x5a554d00u, 0x6a65625du},
    {0x5a564e00u, 0x6a66625eu},
    {0x5b574f00u, 0x6b67635fu},
    {0x5c585000u, 0x6c686460u},
    {0x5e595100u, 0x6e696661u},
    {0x5f5a5200u, 0x6f6a6762u},
    {0x605b5300u, 0x706b6863u},
    {0x615c5400u, 0x716c6964u},
    {0x625d5500u, 0x726d6a65u},
    {0x625e5600u, 0x726e6a66u},
    {0x635f5700u, 0x736f6b67u},
    {0x64605800u, 0x74706c68u},
    {0x66615900u, 0x76716e69u},
    {0x67625a00u, 0x77726f6au},
    {0x68635b00u, 0x7873706bu},
    {0x69645c00u, 0x7974716cu},
    {0x6a655d00u, 0x7a75726du},
    {0x6a665e00u, 0x7a76726eu},
    {0x6a665e00u, 0x7a76726eu},
};

// Residual term for the same normalized product after nearest-E4M3 rounding.
// Positive inputs may have a negative residual, so the decoder XORs the
// source sign bit with the sign already stored in this table.
static __device__ __constant__ __align__(16) const uint2 kE2M1AndUe4m3ToFp8ResidualLut[128] = {
    {0x00000000u, 0x00000000u},
    {0x00000000u, 0x80000000u},
    {0x00000000u, 0x80008000u},
    {0x80000000u, 0x00800080u},
    {0x80000000u, 0x00008000u},
    {0x80800000u, 0x80008000u},
    {0x00800000u, 0x00000080u},
    {0x00800000u, 0x00808080u},
    {0x80000000u, 0x00000000u},
    {0x80008000u, 0x80000000u},
    {0x80008000u, 0x80008000u},
    {0x00008000u, 0x00800080u},
    {0x00808000u, 0x00000000u},
    {0x00808000u, 0x80008000u},
    {0x80808000u, 0x00000080u},
    {0x80808000u, 0x00808080u},
    {0x00000000u, 0x00000000u},
    {0x00000000u, 0x80008000u},
    {0x80000000u, 0x00008000u},
    {0x00800000u, 0x00000080u},
    {0x00008000u, 0x00000000u},
    {0x80008000u, 0x80008000u},
    {0x00808000u, 0x01000000u},
    {0x80808000u, 0x00000080u},
    {0x00000000u, 0x00000000u},
    {0x80000000u, 0x81008000u},
    {0x80000000u, 0x00000000u},
    {0x00808000u, 0x01000000u},
    {0x00000000u, 0x00000000u},
    {0x80000000u, 0x81008000u},
    {0x00008000u, 0x02000100u},
    {0x00808000u, 0x01000000u},
    {0x00000000u, 0x00000000u},
    {0x80000000u, 0x82008100u},
    {0x00000000u, 0x00000000u},
    {0x00008000u, 0x02000100u},
    {0x00000000u, 0x00000000u},
    {0x80000000u, 0x82008100u},
    {0x01000000u, 0x04000200u},
    {0x00008000u, 0x02000100u},
    {0x00000000u, 0x00000000u},
    {0x81000000u, 0x84008200u},
    {0x00000000u, 0x00000000u},
    {0x01000000u, 0x04000200u},
    {0x00000000u, 0x00000000u},
    {0x81000000u, 0x84008200u},
    {0x02000000u, 0x08000400u},
    {0x01000000u, 0x04000200u},
    {0x00000000u, 0x00000000u},
    {0x82000000u, 0x88008400u},
    {0x00000000u, 0x00000000u},
    {0x02000000u, 0x08000400u},
    {0x00000000u, 0x00000000u},
    {0x82000000u, 0x88008400u},
    {0x04000000u, 0x10000800u},
    {0x02000000u, 0x08000400u},
    {0x00000000u, 0x00000000u},
    {0x84000000u, 0x90008800u},
    {0x00000000u, 0x00000000u},
    {0x04000000u, 0x10000800u},
    {0x00000000u, 0x00000000u},
    {0x84000000u, 0x90008800u},
    {0x08000000u, 0x18001000u},
    {0x04000000u, 0x10000800u},
    {0x00000000u, 0x00000000u},
    {0x88000000u, 0x98009000u},
    {0x00000000u, 0x00000000u},
    {0x08000000u, 0x18001000u},
    {0x00000000u, 0x00000000u},
    {0x88000000u, 0x98009000u},
    {0x10000000u, 0x20001800u},
    {0x08000000u, 0x18001000u},
    {0x00000000u, 0x00000000u},
    {0x90000000u, 0xa0009800u},
    {0x00000000u, 0x00000000u},
    {0x10000000u, 0x20001800u},
    {0x00000000u, 0x00000000u},
    {0x90000000u, 0xa0009800u},
    {0x18000000u, 0x28002000u},
    {0x10000000u, 0x20001800u},
    {0x00000000u, 0x00000000u},
    {0x98000000u, 0xa800a000u},
    {0x00000000u, 0x00000000u},
    {0x18000000u, 0x28002000u},
    {0x00000000u, 0x00000000u},
    {0x98000000u, 0xa800a000u},
    {0x20000000u, 0x30002800u},
    {0x18000000u, 0x28002000u},
    {0x00000000u, 0x00000000u},
    {0xa0000000u, 0xb000a800u},
    {0x00000000u, 0x00000000u},
    {0x20000000u, 0x30002800u},
    {0x00000000u, 0x00000000u},
    {0xa0000000u, 0xb000a800u},
    {0x28000000u, 0x38003000u},
    {0x20000000u, 0x30002800u},
    {0x00000000u, 0x00000000u},
    {0xa8000000u, 0xb800b000u},
    {0x00000000u, 0x00000000u},
    {0x28000000u, 0x38003000u},
    {0x00000000u, 0x00000000u},
    {0xa8000000u, 0xb800b000u},
    {0x30000000u, 0x40003800u},
    {0x28000000u, 0x38003000u},
    {0x00000000u, 0x00000000u},
    {0xb0000000u, 0xc000b800u},
    {0x00000000u, 0x00000000u},
    {0x30000000u, 0x40003800u},
    {0x00000000u, 0x00000000u},
    {0xb0000000u, 0xc000b800u},
    {0x38000000u, 0x48004000u},
    {0x30000000u, 0x40003800u},
    {0x00000000u, 0x00000000u},
    {0xb8000000u, 0xc800c000u},
    {0x00000000u, 0x00000000u},
    {0x38000000u, 0x48004000u},
    {0x00000000u, 0x00000000u},
    {0xb8000000u, 0xc800c000u},
    {0x40000000u, 0x50004800u},
    {0x38000000u, 0x48004000u},
    {0x00000000u, 0x00000000u},
    {0xc0000000u, 0xd000c800u},
    {0x00000000u, 0x00000000u},
    {0x40000000u, 0x50004800u},
    {0x00000000u, 0x00000000u},
    {0xc0000000u, 0xd000c800u},
    {0x48000000u, 0x58005000u},
    {0x48000000u, 0x58005000u},
};
DG_NVFP4_INLINE uint2 load_e2m1_ue4m3_lut(std::uint32_t scale_ue4m3) {
    return kE2M1AndUe4m3ToFp8Lut[scale_ue4m3 & 0x7fu];
}

template <bool kUseDp4a>
DG_NVFP4_INLINE std::uint32_t pack_nvfp4_magnitude_selector(std::uint32_t byte_magnitudes) {
    if constexpr (kUseDp4a) {
        const std::uint32_t lo = __dp4a(byte_magnitudes, 0x00001001u, 0u);
        const std::uint32_t hi = __dp4a(byte_magnitudes, 0x10010000u, 0u);
        return lo + (hi << 8);
    } else {
        const std::uint32_t packed_pairs = byte_magnitudes + (byte_magnitudes >> 4);
        return __byte_perm(packed_pairs, 0u, 0x4420u);
    }
}

DG_NVFP4_INLINE std::uint32_t byte_perm_unchecked(std::uint32_t a, std::uint32_t b,
                                                  std::uint32_t selector) {
    // Callers provide 0..7 selector nibbles; raw PTX avoids a redundant 0x7777 mask.
    std::uint32_t out;
    asm("prmt.b32 %0, %1, %2, %3;" : "=r"(out) : "r"(a), "r"(b), "r"(selector));
    return out;
}

template <bool kUseDp4aHi = false, bool kUseDp4aLo = kUseDp4aHi>
DG_NVFP4_INLINE uint2 dequant_nvfp4_to_fp8_pair_with_lut(std::uint32_t uq, const uint2& lut) {
    const std::uint32_t sel_hi =
        pack_nvfp4_magnitude_selector<kUseDp4aHi>((uq >> 4) & 0x07070707u);
    const std::uint32_t sel_lo =
        pack_nvfp4_magnitude_selector<kUseDp4aLo>(uq & 0x07070707u);

    std::uint32_t out_hi = byte_perm_unchecked(lut.x, lut.y, sel_hi);
    std::uint32_t out_lo = byte_perm_unchecked(lut.x, lut.y, sel_lo);
    out_hi ^= uq & 0x80808080u;
    out_lo ^= (uq << 4) & 0x80808080u;
    return make_uint2(out_hi, out_lo);
}

DG_NVFP4_INLINE uint2 dequant_nvfp4_to_fp8_pair(std::uint32_t q, std::uint32_t scale_ue4m3) {
    return dequant_nvfp4_to_fp8_pair_with_lut(q, load_e2m1_ue4m3_lut(scale_ue4m3));
}


// Branchless: 3-bit E2M1 magnitude code -> FP16 bit pattern.
// mag3 in 0..7 maps to FP4 values {0, 0.5, 1, 1.5, 2, 3, 4, 6}.
DG_NVFP4_INLINE uint16_t e2m1_mag_to_fp16_bits(std::uint32_t mag3) {
    std::uint32_t exp_raw  = mag3 >> 1u;
    std::uint32_t mant_raw = mag3 & 1u;
    std::uint32_t is_norm  = (exp_raw != 0u) ? 1u : 0u;
    // Normal:  fp16 = ((exp_raw+14)<<10) | (mant_raw<<9)
    // Subnorm (mag3=1, value=0.5): fp16 = 14<<10 = 0x3800 (is_norm=0 zeroes mant)
    // Zero (mag3=0): masked to 0
    std::uint32_t fp16 = ((exp_raw + 14u) << 10u) | ((mant_raw & is_norm) << 9u);
    return static_cast<uint16_t>(fp16 * static_cast<std::uint32_t>(mag3 != 0u));
}

// Every finite E4M3 value and every E2M1*E4M3 product used here is exactly
// representable in BF16. Convert through FP16 bits so activation expansion
// does not require scalar FP32 arithmetic in the producer hot loop.
DG_NVFP4_INLINE uint16_t fp16_bits_to_bf16_bits(std::uint32_t fp16) {
    const std::uint32_t sign = fp16 & 0x8000u;
    const std::uint32_t magnitude = fp16 & 0x7fffu;
    if (magnitude == 0u)
        return static_cast<uint16_t>(sign);
    const std::uint32_t exponent = (fp16 >> 10u) & 0x1fu;
    const std::uint32_t mantissa = fp16 & 0x3ffu;
    return static_cast<uint16_t>(
        sign | ((exponent + 112u) << 7u) | (mantissa >> 3u));
}

DG_NVFP4_INLINE std::uint32_t convert_e4m3x2_to_bf16x2(
        std::uint32_t packed_fp8) {
    const __half2_raw converted = __nv_cvt_fp8x2_to_halfraw2(
        static_cast<__nv_fp8x2_storage_t>(packed_fp8), __NV_E4M3);
    const std::uint32_t lo = fp16_bits_to_bf16_bits(converted.x);
    const std::uint32_t hi = fp16_bits_to_bf16_bits(converted.y);
    return lo | (hi << 16u);
}

DG_NVFP4_INLINE uint4 convert_e4m3x8_to_bf16x8(const uint2& packed_fp8) {
    return make_uint4(
        convert_e4m3x2_to_bf16x2(packed_fp8.x),
        convert_e4m3x2_to_bf16x2(packed_fp8.x >> 16u),
        convert_e4m3x2_to_bf16x2(packed_fp8.y),
        convert_e4m3x2_to_bf16x2(packed_fp8.y >> 16u));
}

// A row stores the low and high bytes of eight positive BF16 products in
// separate PRMT-friendly banks: {lo[0:4], lo[4:8], hi[0:4], hi[4:8]}.
DG_NVFP4_INLINE uint4 make_e2m1_ue4m3_to_bf16_lut(
        std::uint32_t scale_ue4m3) {
    uint4 lut = make_uint4(0u, 0u, 0u, 0u);
    const __half scale_h = __nv_cvt_fp8_to_halfraw(
        scale_ue4m3 & 0x7fu, __NV_E4M3);
    #pragma unroll
    for (std::uint32_t magnitude = 0; magnitude < 8; ++magnitude) {
        const __half magnitude_h = __ushort_as_half(
            e2m1_mag_to_fp16_bits(magnitude));
        const std::uint16_t bf16 = fp16_bits_to_bf16_bits(
            __half_as_ushort(__hmul(magnitude_h, scale_h)));
        const std::uint32_t shift = (magnitude & 3u) * 8u;
        if (magnitude < 4u) {
            lut.x |= static_cast<std::uint32_t>(bf16 & 0xffu) << shift;
            lut.z |= static_cast<std::uint32_t>(bf16 >> 8u) << shift;
        } else {
            lut.y |= static_cast<std::uint32_t>(bf16 & 0xffu) << shift;
            lut.w |= static_cast<std::uint32_t>(bf16 >> 8u) << shift;
        }
    }
    return lut;
}

// Expand four packed FP4 bytes (eight values) into eight canonical-order BF16
// values. Sign bits are applied to the high byte after magnitude lookup.
DG_NVFP4_INLINE uint4 dequant_nvfp4_to_bf16x8_with_lut(
        std::uint32_t uq, const uint4& lut) {
    const std::uint32_t even_selector =
        pack_nvfp4_magnitude_selector<false>(uq & 0x07070707u);
    const std::uint32_t odd_selector =
        pack_nvfp4_magnitude_selector<false>((uq >> 4u) & 0x07070707u);

    const std::uint32_t even_lo =
        byte_perm_unchecked(lut.x, lut.y, even_selector);
    const std::uint32_t odd_lo =
        byte_perm_unchecked(lut.x, lut.y, odd_selector);
    const std::uint32_t even_hi =
        byte_perm_unchecked(lut.z, lut.w, even_selector) ^
        ((uq << 4u) & 0x80808080u);
    const std::uint32_t odd_hi =
        byte_perm_unchecked(lut.z, lut.w, odd_selector) ^
        (uq & 0x80808080u);

    const std::uint32_t even_01 = __byte_perm(even_lo, even_hi, 0x5140);
    const std::uint32_t even_23 = __byte_perm(even_lo, even_hi, 0x7362);
    const std::uint32_t odd_01 = __byte_perm(odd_lo, odd_hi, 0x5140);
    const std::uint32_t odd_23 = __byte_perm(odd_lo, odd_hi, 0x7362);
    return make_uint4(
        __byte_perm(even_01, odd_01, 0x5410),
        __byte_perm(even_01, odd_01, 0x7632),
        __byte_perm(even_23, odd_23, 0x5410),
        __byte_perm(even_23, odd_23, 0x7632));
}

// LUT-free dequant: identical output to dequant_nvfp4_to_fp8_pair_with_lut
// but synthesises the 8-byte lut entry arithmetically (no smem LUT load).
// Cost: 1 fp8->half + 8 hmul + 8 half->fp8 per call vs 1 smem load + 2 byte_perm.
DG_NVFP4_INLINE uint2 dequant_nvfp4_to_fp8_pair_lut_free(std::uint32_t uq,
                                                          std::uint32_t scale_ue4m3) {
    __half scale_h = __nv_cvt_fp8_to_halfraw(scale_ue4m3 & 0x7fu, __NV_E4M3);
    uint8_t fp8_mag[8];
#pragma unroll
    for (int m = 0; m < 8; ++m) {
        __half mag_h = __ushort_as_half(
            e2m1_mag_to_fp16_bits(static_cast<std::uint32_t>(m)));
        __half prod  = __hmul(__hmul(scale_h, mag_h), __float2half_rn(0.125f));
        fp8_mag[m]   = static_cast<uint8_t>(__nv_cvt_halfraw_to_fp8(
            __half_raw{__half_as_ushort(prod)}, __NV_SATFINITE, __NV_E4M3));
    }
    uint2 lut_c;
    lut_c.x = (std::uint32_t)fp8_mag[0] | ((std::uint32_t)fp8_mag[1] << 8u) |
              ((std::uint32_t)fp8_mag[2] << 16u) | ((std::uint32_t)fp8_mag[3] << 24u);
    lut_c.y = (std::uint32_t)fp8_mag[4] | ((std::uint32_t)fp8_mag[5] << 8u) |
              ((std::uint32_t)fp8_mag[6] << 16u) | ((std::uint32_t)fp8_mag[7] << 24u);
    return dequant_nvfp4_to_fp8_pair_with_lut(uq, lut_c);
}

#undef DG_NVFP4_INLINE

} // namespace nvfp4
} // namespace deep_gemm
