// SPDX-License-Identifier: MIT
//
// MXFP4 (E2M1 + E8M0) -> FP8 (E4M3) dequant with the block scale folded into
// the FP8 exponent, for SM90 fused MegaMoE.
//
// Why this differs from the NVFP4 bridge rather than just re-tabulating it:
// NVFP4's UE4M3 scale carries a 3-bit mantissa, so `magnitude * scale` is a
// real multiply and the only cheap way to evaluate it is a 128-row table
// indexed by the scale byte -- a dependent shared-memory load on the critical
// path, plus 1 KB of SMEM. MXFP4's E8M0 scale is a pure power of two, so the
// same product is an exponent adjustment: multiplying an E4M3 value by 2^k
// means adding k to its 4-bit exponent field, i.e. adding (k << 3) to the byte.
//
// So the table collapses to 8 constant bytes (the E2M1 magnitudes encoded at
// scale 2^0) held in two immediates, and applying a scale group becomes two
// SIMD byte adds in registers. No SMEM, no table, no dependent load.
//
// Exactness: verified against an independent E2M1 x 2^k -> E4M3 reference over
// all 256 E8M0 codes x 16 nibbles (4096/4096 bit-exact), including the 1690
// saturating cases and the underflow-to-zero cases.

#pragma once

#include <cstdint>

namespace deep_gemm {
namespace mxfp4 {

#define DG_MXFP4_INLINE __device__ __forceinline__

// __byte_perm without the compiler's range check on the selector; the selectors
// below are masked to 0x7 lanes by construction.
DG_MXFP4_INLINE std::uint32_t byte_perm_unchecked(std::uint32_t a, std::uint32_t b,
                                                  std::uint32_t selector) {
    std::uint32_t out;
    asm("prmt.b32 %0, %1, %2, %3;" : "=r"(out) : "r"(a), "r"(b), "r"(selector));
    return out;
}

// E2M1 magnitudes {0, .5, 1, 1.5, 2, 3, 4, 6} encoded as E4M3 at scale 2^0.
// Byte i of (LUT_X, LUT_Y) holds magnitude index i and i+4 respectively, the
// layout __byte_perm consumes.
static constexpr uint32_t kBaseLutX = 0x3c383000u;
static constexpr uint32_t kBaseLutY = 0x4c484440u;

// Largest finite E4M3 (448.0). Saturating adds clamp to 0xff, so results are
// pulled back to this.
static constexpr uint32_t kE4M3MaxBytes = 0x7e7e7e7eu;
// Exponent field 0 encodes zero/subnormal; E4M3 subnormals are not produced by
// this path, so any byte below 0x08 must flush to zero.
static constexpr uint32_t kMinNormalBytes = 0x08080808u;

// The 8-byte magnitude table for one E8M0 scale group, built in registers.
struct ScaledLut {
    std::uint32_t x;
    std::uint32_t y;
};

// Byte 0 of the base table is magnitude 0. Filling it with a nonzero exponent
// keeps the fast path's subtract from borrowing across the lane boundary; the
// lane is masked off after the add.
static constexpr uint32_t kBaseLutXFilled = 0x3c383030u;

DG_MXFP4_INLINE ScaledLut make_scaled_lut_general(int k) {
    const std::uint32_t magnitude = static_cast<std::uint32_t>(k >= 0 ? k : -k) * 8u;
    const std::uint32_t delta = (magnitude > 255u ? 255u : magnitude) * 0x01010101u;
    ScaledLut lut;
    if (k >= 0) {
        lut.x = __vminu4(__vaddus4(kBaseLutX, delta), kE4M3MaxBytes) & 0xffffff00u;
        lut.y = __vminu4(__vaddus4(kBaseLutY, delta), kE4M3MaxBytes);
    } else {
        const std::uint32_t x = __vsubus4(kBaseLutX, delta);
        const std::uint32_t y = __vsubus4(kBaseLutY, delta);
        lut.x = x & __vcmpgeu4(x, kMinNormalBytes);
        lut.y = y & __vcmpgeu4(y, kMinNormalBytes);
    }
    return lut;
}

// Fold scale 2^(code - 127) into the base table.
//
// k >= 0: saturating byte add, then clamp to 448. Byte 0 of x is magnitude 0
//         and must stay zero, so it is masked off -- adding to it would
//         manufacture a nonzero value.
// k <  0: saturating byte subtract, which already floors at zero, then flush
//         any byte whose exponent field reached 0. That case is reachable with
//         real weights: a group whose max is below 6 * 2^-5 gets k < -5, and
//         its smallest magnitude underflows E4M3's normal range.
//
// |k| * 8 is clamped to 255 before broadcasting; beyond that every byte
// saturates anyway, and an unclamped broadcast would corrupt neighbouring
// bytes.
DG_MXFP4_INLINE ScaledLut make_scaled_lut(std::uint32_t scale_ue8m0) {
    const int k = static_cast<int>(scale_ue8m0 & 0xffu) - 127;

    // Fast path. The seven nonzero base bytes hold exponent fields 6..9, so for
    // k in [-5, 6] every byte stays inside [0x08, 0x7c]: no byte overflows into
    // its neighbour and a plain 32-bit add is exact. kBaseLutXFilled carries
    // 0x30 in byte 0 purely so the subtract cannot borrow out of the magnitude-0
    // lane; that lane is masked back to zero afterwards.
    // Three instructions, no table, no memory traffic.
    if (__builtin_expect(k >= -5 && k <= 6, 1)) {
        const std::uint32_t delta = static_cast<std::uint32_t>(k * 8) * 0x01010101u;
        ScaledLut lut;
        lut.x = (kBaseLutXFilled + delta) & 0xffffff00u;
        lut.y = kBaseLutY + delta;
        return lut;
    }

    // Slow path: the group's scale pushes part of the table out of E4M3's
    // normal range, so bytes need individual saturation or flushing. Reachable
    // but rare -- a weight group whose maximum is below 6 * 2^-5, or above
    // 6 * 2^6, lands here.
    return make_scaled_lut_general(k);
}

// Decode eight packed FP4 nibbles (one uint32) into eight FP8 bytes, using a
// table already scaled by make_scaled_lut. Identical shape to the NVFP4
// decoder; only the table's provenance differs.
DG_MXFP4_INLINE uint2 dequant_word(std::uint32_t packed, const ScaledLut& lut) {
    const std::uint32_t selectors = packed & 0x77777777u;
    std::uint32_t out_hi = byte_perm_unchecked(lut.x, lut.y, selectors);
    std::uint32_t out_lo = byte_perm_unchecked(lut.x, lut.y, selectors >> 16);
    // Sign bits ride along untouched: OR them back in place.
    asm("lop3.b32 %0, %0, %1, 0x80808080, 0xf8;" : "+r"(out_hi) : "r"(packed));
    const std::uint32_t shifted = packed << 4;
    asm("lop3.b32 %0, %0, %1, 0x80808080, 0xf8;" : "+r"(out_lo) : "r"(shifted));
    return make_uint2(out_hi, out_lo);
}

// One 32-element MXFP4 group: four packed words sharing a single E8M0 byte.
// NVFP4 needs two tables here because its groups are 16 wide.
DG_MXFP4_INLINE void dequant_group32(
        std::uint8_t* __restrict__ fp8_dst,
        const uint4& packed,
        std::uint32_t scale_ue8m0) {
    const ScaledLut lut = make_scaled_lut(scale_ue8m0);
    const uint2 w0 = dequant_word(packed.x, lut);
    const uint2 w1 = dequant_word(packed.y, lut);
    const uint2 w2 = dequant_word(packed.z, lut);
    const uint2 w3 = dequant_word(packed.w, lut);
    reinterpret_cast<uint4*>(fp8_dst)[0] = make_uint4(w0.x, w0.y, w1.x, w1.y);
    reinterpret_cast<uint4*>(fp8_dst)[1] = make_uint4(w2.x, w2.y, w3.x, w3.y);
}

#undef DG_MXFP4_INLINE

}  // namespace mxfp4
}  // namespace deep_gemm
