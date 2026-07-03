"""Isolated SM90 NVFP4 decoder benchmark using the existing 8-byte padding.

Four 16-bit magnitude selectors are stored in bytes 72..79 of each unchanged
80-byte row.  No production code or runtime layout size is changed.
"""

import argparse
import os
import statistics
import sys

import torch
from torch.utils.cpp_extension import load_inline


REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)


CUDA_SRC = r"""
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <c10/cuda/CUDAException.h>
#include <deep_gemm/quantization/nvfp4_dequant.cuh>

namespace {

constexpr int kRows = 128;
constexpr int kRowBytes = 80;
constexpr int kFp8RowBytes = 128;

template <bool kUseDp4aHi, bool kUseDp4aLo>
__device__ __forceinline__ uint2 decode_current_word(uint32_t q, const uint2& lut) {
    return deep_gemm::nvfp4::dequant_nvfp4_to_fp8_pair_with_lut<
        kUseDp4aHi, kUseDp4aLo>(q, lut);
}

template <bool kStoreHi, bool kUseDp4aHi, bool kUseDp4aLo>
__device__ __forceinline__ uint2 decode_stored_word(
        uint32_t q, uint32_t stored_selector, const uint2& lut) {
    uint32_t sel_hi;
    uint32_t sel_lo;
    if constexpr (kStoreHi) {
        sel_hi = stored_selector;
        sel_lo = deep_gemm::nvfp4::pack_nvfp4_magnitude_selector<kUseDp4aLo>(
            q & 0x07070707u);
    } else {
        sel_hi = deep_gemm::nvfp4::pack_nvfp4_magnitude_selector<kUseDp4aHi>(
            (q >> 4) & 0x07070707u);
        sel_lo = stored_selector;
    }
    uint32_t out_hi = deep_gemm::nvfp4::byte_perm_unchecked(lut.x, lut.y, sel_hi);
    uint32_t out_lo = deep_gemm::nvfp4::byte_perm_unchecked(lut.x, lut.y, sel_lo);
    out_hi |= q & 0x80808080u;
    out_lo |= (q << 4) & 0x80808080u;
    return make_uint2(out_hi, out_lo);
}

__device__ __forceinline__ uint32_t apply_grouped_nibble_signs(
        uint32_t magnitudes, uint32_t grouped_nibbles) {
    uint32_t sign_fill;
    const uint32_t shifted = grouped_nibbles << 4;
    asm("prmt.b32 %0, %1, %2, 0x9d8c;"
        : "=r"(sign_fill) : "r"(grouped_nibbles), "r"(shifted));
    uint32_t result;
    asm("lop3.b32 %0, %1, %2, 0x80808080, 0xf8;"
        : "=r"(result) : "r"(magnitudes), "r"(sign_fill));
    return result;
}

__device__ __forceinline__ uint2 decode_grouped_nibble_word(
        uint32_t grouped, const uint2& lut) {
    const uint32_t magnitude_selectors = grouped & 0x77777777u;
    uint32_t out_hi = deep_gemm::nvfp4::byte_perm_unchecked(
        lut.x, lut.y, magnitude_selectors);
    uint32_t out_lo = deep_gemm::nvfp4::byte_perm_unchecked(
        lut.x, lut.y, magnitude_selectors >> 16);
    out_hi = apply_grouped_nibble_signs(out_hi, grouped);
    out_lo = apply_grouped_nibble_signs(out_lo, grouped >> 16);
    return make_uint2(out_hi, out_lo);
}

template <bool kPaired, bool kUseDp4aHi, bool kUseDp4aLo>
__device__ __forceinline__ void decode_current_row(
        uint8_t* __restrict__ fp8,
        const uint8_t* __restrict__ packed,
        uint32_t row,
        const uint2* __restrict__ lut_smem) {
    const uint8_t* __restrict__ row_ptr = packed + row * kRowBytes;
    const uint4* __restrict__ fp4_src = reinterpret_cast<const uint4*>(row_ptr);
    uint4 fp4_quads[4];
#pragma unroll
    for (int i = 0; i < 4; ++i)
        fp4_quads[i] = fp4_src[i];
    const uint2 scale_words = *reinterpret_cast<const uint2*>(row_ptr + 64);
    const uint32_t scale_word_lo = scale_words.x;
    const uint32_t scale_word_hi = scale_words.y;
    uint8_t* __restrict__ fp8_dst = fp8 + row * kFp8RowBytes;
    const uint32_t row_swizzle = (row & 7u) << 4;

#pragma unroll
    for (int quad_i = 0; quad_i < 4; ++quad_i) {
        const uint4 q = fp4_quads[quad_i];
        const uint32_t scale_word = quad_i < 2 ? scale_word_lo : scale_word_hi;
        const int scale_i0 = quad_i * 2;
        const int scale_i1 = scale_i0 + 1;
        const uint32_t scale0 = (scale_word >> ((scale_i0 & 3) * 8)) & 0x7fu;

        if constexpr (kPaired) {
            const uint32_t scale1 = (scale_word >> ((scale_i1 & 3) * 8)) & 0x7fu;
            const uint2 lut0 = lut_smem[scale0];
            const uint2 lut1 = lut_smem[scale1];
            const uint2 q0 = decode_current_word<kUseDp4aHi, kUseDp4aLo>(q.x, lut0);
            const uint2 q1 = decode_current_word<kUseDp4aHi, kUseDp4aLo>(q.y, lut0);
            *reinterpret_cast<uint4*>(fp8_dst + ((scale_i0 * 16) ^ row_swizzle)) =
                make_uint4(q0.x, q0.y, q1.x, q1.y);
            const uint2 q2 = decode_current_word<kUseDp4aHi, kUseDp4aLo>(q.z, lut1);
            const uint2 q3 = decode_current_word<kUseDp4aHi, kUseDp4aLo>(q.w, lut1);
            *reinterpret_cast<uint4*>(fp8_dst + ((scale_i1 * 16) ^ row_swizzle)) =
                make_uint4(q2.x, q2.y, q3.x, q3.y);
        } else {
            const uint2 lut0 = lut_smem[scale0];
            const uint2 q0 = decode_current_word<kUseDp4aHi, kUseDp4aLo>(q.x, lut0);
            const uint2 q1 = decode_current_word<kUseDp4aHi, kUseDp4aLo>(q.y, lut0);
            *reinterpret_cast<uint4*>(fp8_dst + ((scale_i0 * 16) ^ row_swizzle)) =
                make_uint4(q0.x, q0.y, q1.x, q1.y);
            const uint32_t scale1 = (scale_word >> ((scale_i1 & 3) * 8)) & 0x7fu;
            const uint2 lut1 = lut_smem[scale1];
            const uint2 q2 = decode_current_word<kUseDp4aHi, kUseDp4aLo>(q.z, lut1);
            const uint2 q3 = decode_current_word<kUseDp4aHi, kUseDp4aLo>(q.w, lut1);
            *reinterpret_cast<uint4*>(fp8_dst + ((scale_i1 * 16) ^ row_swizzle)) =
                make_uint4(q2.x, q2.y, q3.x, q3.y);
        }
    }
}

template <bool kStoreHi, int kWordInQuad, bool kPaired,
          bool kUseDp4aHi, bool kUseDp4aLo>
__device__ __forceinline__ void decode_padding_row(
        uint8_t* __restrict__ fp8,
        const uint8_t* __restrict__ packed,
        uint32_t row,
        const uint2* __restrict__ lut_smem) {
    const uint8_t* __restrict__ row_ptr = packed + row * kRowBytes;
    const uint4* __restrict__ fp4_src = reinterpret_cast<const uint4*>(row_ptr);
    uint4 fp4_quads[4];
#pragma unroll
    for (int i = 0; i < 4; ++i)
        fp4_quads[i] = fp4_src[i];
    const uint4 tail = *reinterpret_cast<const uint4*>(row_ptr + 64);
    const uint32_t scale_word_lo = tail.x;
    const uint32_t scale_word_hi = tail.y;
    uint8_t* __restrict__ fp8_dst = fp8 + row * kFp8RowBytes;
    const uint32_t row_swizzle = (row & 7u) << 4;

#pragma unroll
    for (int quad_i = 0; quad_i < 4; ++quad_i) {
        const uint4 q = fp4_quads[quad_i];
        const uint32_t selector_pair = quad_i < 2 ? tail.z : tail.w;
        const uint32_t stored_selector =
            (selector_pair >> ((quad_i & 1) * 16));
        const uint32_t scale_word = quad_i < 2 ? scale_word_lo : scale_word_hi;
        const int scale_i0 = quad_i * 2;
        const int scale_i1 = scale_i0 + 1;
        const uint32_t scale0 = (scale_word >> ((scale_i0 & 3) * 8)) & 0x7fu;

        if constexpr (kPaired) {
            const uint32_t scale1 = (scale_word >> ((scale_i1 & 3) * 8)) & 0x7fu;
            const uint2 lut0 = lut_smem[scale0];
            const uint2 lut1 = lut_smem[scale1];
            const uint2 q0 = kWordInQuad == 0
                ? decode_stored_word<kStoreHi, kUseDp4aHi, kUseDp4aLo>(q.x, stored_selector, lut0)
                : decode_current_word<kUseDp4aHi, kUseDp4aLo>(q.x, lut0);
            const uint2 q1 = kWordInQuad == 1
                ? decode_stored_word<kStoreHi, kUseDp4aHi, kUseDp4aLo>(q.y, stored_selector, lut0)
                : decode_current_word<kUseDp4aHi, kUseDp4aLo>(q.y, lut0);
            *reinterpret_cast<uint4*>(fp8_dst + ((scale_i0 * 16) ^ row_swizzle)) =
                make_uint4(q0.x, q0.y, q1.x, q1.y);
            const uint2 q2 = kWordInQuad == 2
                ? decode_stored_word<kStoreHi, kUseDp4aHi, kUseDp4aLo>(q.z, stored_selector, lut1)
                : decode_current_word<kUseDp4aHi, kUseDp4aLo>(q.z, lut1);
            const uint2 q3 = kWordInQuad == 3
                ? decode_stored_word<kStoreHi, kUseDp4aHi, kUseDp4aLo>(q.w, stored_selector, lut1)
                : decode_current_word<kUseDp4aHi, kUseDp4aLo>(q.w, lut1);
            *reinterpret_cast<uint4*>(fp8_dst + ((scale_i1 * 16) ^ row_swizzle)) =
                make_uint4(q2.x, q2.y, q3.x, q3.y);
        } else {
            const uint2 lut0 = lut_smem[scale0];
            const uint2 q0 = kWordInQuad == 0
                ? decode_stored_word<kStoreHi, kUseDp4aHi, kUseDp4aLo>(q.x, stored_selector, lut0)
                : decode_current_word<kUseDp4aHi, kUseDp4aLo>(q.x, lut0);
            const uint2 q1 = kWordInQuad == 1
                ? decode_stored_word<kStoreHi, kUseDp4aHi, kUseDp4aLo>(q.y, stored_selector, lut0)
                : decode_current_word<kUseDp4aHi, kUseDp4aLo>(q.y, lut0);
            *reinterpret_cast<uint4*>(fp8_dst + ((scale_i0 * 16) ^ row_swizzle)) =
                make_uint4(q0.x, q0.y, q1.x, q1.y);
            const uint32_t scale1 = (scale_word >> ((scale_i1 & 3) * 8)) & 0x7fu;
            const uint2 lut1 = lut_smem[scale1];
            const uint2 q2 = kWordInQuad == 2
                ? decode_stored_word<kStoreHi, kUseDp4aHi, kUseDp4aLo>(q.z, stored_selector, lut1)
                : decode_current_word<kUseDp4aHi, kUseDp4aLo>(q.z, lut1);
            const uint2 q3 = kWordInQuad == 3
                ? decode_stored_word<kStoreHi, kUseDp4aHi, kUseDp4aLo>(q.w, stored_selector, lut1)
                : decode_current_word<kUseDp4aHi, kUseDp4aLo>(q.w, lut1);
            *reinterpret_cast<uint4*>(fp8_dst + ((scale_i1 * 16) ^ row_swizzle)) =
                make_uint4(q2.x, q2.y, q3.x, q3.y);
        }
    }
}

template <bool kPaired>
__device__ __forceinline__ void decode_grouped_nibble_row(
        uint8_t* __restrict__ fp8,
        const uint8_t* __restrict__ packed,
        uint32_t row,
        const uint2* __restrict__ lut_smem) {
    const uint8_t* __restrict__ row_ptr = packed + row * kRowBytes;
    const uint4* __restrict__ fp4_src = reinterpret_cast<const uint4*>(row_ptr);
    uint4 fp4_quads[4];
#pragma unroll
    for (int i = 0; i < 4; ++i)
        fp4_quads[i] = fp4_src[i];
    const uint2 scale_words = *reinterpret_cast<const uint2*>(row_ptr + 64);
    uint8_t* __restrict__ fp8_dst = fp8 + row * kFp8RowBytes;
    const uint32_t row_swizzle = (row & 7u) << 4;

#pragma unroll
    for (int quad_i = 0; quad_i < 4; ++quad_i) {
        const uint4 q = fp4_quads[quad_i];
        const uint32_t scale_word = quad_i < 2 ? scale_words.x : scale_words.y;
        const int scale_i0 = quad_i * 2;
        const int scale_i1 = scale_i0 + 1;
        const uint32_t scale0 = (scale_word >> ((scale_i0 & 3) * 8)) & 0x7fu;
        if constexpr (kPaired) {
            const uint32_t scale1 = (scale_word >> ((scale_i1 & 3) * 8)) & 0x7fu;
            const uint2 lut0 = lut_smem[scale0];
            const uint2 lut1 = lut_smem[scale1];
            const uint2 q0 = decode_grouped_nibble_word(q.x, lut0);
            const uint2 q1 = decode_grouped_nibble_word(q.y, lut0);
            *reinterpret_cast<uint4*>(fp8_dst + ((scale_i0 * 16) ^ row_swizzle)) =
                make_uint4(q0.x, q0.y, q1.x, q1.y);
            const uint2 q2 = decode_grouped_nibble_word(q.z, lut1);
            const uint2 q3 = decode_grouped_nibble_word(q.w, lut1);
            *reinterpret_cast<uint4*>(fp8_dst + ((scale_i1 * 16) ^ row_swizzle)) =
                make_uint4(q2.x, q2.y, q3.x, q3.y);
        } else {
            const uint2 lut0 = lut_smem[scale0];
            const uint2 q0 = decode_grouped_nibble_word(q.x, lut0);
            const uint2 q1 = decode_grouped_nibble_word(q.y, lut0);
            *reinterpret_cast<uint4*>(fp8_dst + ((scale_i0 * 16) ^ row_swizzle)) =
                make_uint4(q0.x, q0.y, q1.x, q1.y);
            const uint32_t scale1 = (scale_word >> ((scale_i1 & 3) * 8)) & 0x7fu;
            const uint2 lut1 = lut_smem[scale1];
            const uint2 q2 = decode_grouped_nibble_word(q.z, lut1);
            const uint2 q3 = decode_grouped_nibble_word(q.w, lut1);
            *reinterpret_cast<uint4*>(fp8_dst + ((scale_i1 * 16) ^ row_swizzle)) =
                make_uint4(q2.x, q2.y, q3.x, q3.y);
        }
    }
}

template <bool kDecode, bool kPadding, bool kStoreHi, int kWordInQuad,
          bool kPaired, bool kUseDp4aHi, bool kUseDp4aLo>
__global__ __launch_bounds__(128) void bench_kernel(
        const uint8_t* __restrict__ input,
        int64_t* __restrict__ cycles,
        int32_t* __restrict__ witnesses) {
    extern __shared__ __align__(16) uint8_t smem[];
    uint8_t* packed = smem;
    uint8_t* fp8 = packed + kRows * kRowBytes;
    auto* lut = reinterpret_cast<uint2*>(fp8 + kRows * kFp8RowBytes);
    const uint32_t tid = threadIdx.x;
    for (int i = tid; i < kRows * kRowBytes; i += blockDim.x)
        packed[i] = input[i];
    for (int i = tid; i < kRows * kFp8RowBytes; i += blockDim.x)
        fp8[i] = 0;
    lut[tid] = deep_gemm::nvfp4::kE2M1AndUe4m3ToFp8Lut[tid];
    __syncthreads();

    uint64_t start = 0;
    if (tid == 0)
        start = clock64();
    __syncthreads();
    if constexpr (kDecode) {
        if constexpr (kPadding) {
            decode_padding_row<kStoreHi, kWordInQuad, kPaired, kUseDp4aHi, kUseDp4aLo>(
                fp8, packed, tid, lut);
        } else {
            decode_current_row<kPaired, kUseDp4aHi, kUseDp4aLo>(fp8, packed, tid, lut);
        }
    }
    __syncthreads();
    if (tid == 0)
        cycles[blockIdx.x] = static_cast<int64_t>(clock64() - start);
    const uint32_t* row_words = reinterpret_cast<const uint32_t*>(fp8 + tid * kFp8RowBytes);
    witnesses[blockIdx.x * kRows + tid] = static_cast<int32_t>(row_words[tid & 31u]);
}

template <bool kDecode, bool kPadding, bool kStoreHi, int kWordInQuad,
          bool kPaired, bool kUseDp4aHi, bool kUseDp4aLo>
void launch(const torch::Tensor& input, torch::Tensor& cycles, torch::Tensor& witnesses) {
    constexpr int kSharedBytes =
        kRows * (kRowBytes + kFp8RowBytes) + 128 * static_cast<int>(sizeof(uint2));
    bench_kernel<kDecode, kPadding, kStoreHi, kWordInQuad,
                 kPaired, kUseDp4aHi, kUseDp4aLo>
        <<<static_cast<int>(cycles.numel()), 128, kSharedBytes>>>(
            input.data_ptr<uint8_t>(), cycles.data_ptr<int64_t>(),
            witnesses.data_ptr<int32_t>());
}

template <bool kPaired>
__global__ __launch_bounds__(128) void bench_grouped_kernel(
        const uint8_t* __restrict__ input,
        int64_t* __restrict__ cycles,
        int32_t* __restrict__ witnesses) {
    extern __shared__ __align__(16) uint8_t smem[];
    uint8_t* packed = smem;
    uint8_t* fp8 = packed + kRows * kRowBytes;
    auto* lut = reinterpret_cast<uint2*>(fp8 + kRows * kFp8RowBytes);
    const uint32_t tid = threadIdx.x;
    for (int i = tid; i < kRows * kRowBytes; i += blockDim.x)
        packed[i] = input[i];
    for (int i = tid; i < kRows * kFp8RowBytes; i += blockDim.x)
        fp8[i] = 0;
    lut[tid] = deep_gemm::nvfp4::kE2M1AndUe4m3ToFp8Lut[tid];
    __syncthreads();

    uint64_t start = 0;
    if (tid == 0)
        start = clock64();
    __syncthreads();
    decode_grouped_nibble_row<kPaired>(fp8, packed, tid, lut);
    __syncthreads();
    if (tid == 0)
        cycles[blockIdx.x] = static_cast<int64_t>(clock64() - start);
    const uint32_t* row_words = reinterpret_cast<const uint32_t*>(fp8 + tid * kFp8RowBytes);
    witnesses[blockIdx.x * kRows + tid] = static_cast<int32_t>(row_words[tid & 31u]);
}

template <bool kPaired>
void launch_grouped(const torch::Tensor& input, torch::Tensor& cycles, torch::Tensor& witnesses) {
    constexpr int kSharedBytes =
        kRows * (kRowBytes + kFp8RowBytes) + 128 * static_cast<int>(sizeof(uint2));
    bench_grouped_kernel<kPaired>
        <<<static_cast<int>(cycles.numel()), 128, kSharedBytes>>>(
            input.data_ptr<uint8_t>(), cycles.data_ptr<int64_t>(),
            witnesses.data_ptr<int32_t>());
}

}  // namespace

std::vector<torch::Tensor> run_dequant_padding_bench(
        torch::Tensor input, int64_t variant, int64_t blocks) {
    TORCH_CHECK(input.is_cuda() && input.scalar_type() == torch::kUInt8);
    auto cycles = torch::empty({blocks}, input.options().dtype(torch::kInt64));
    auto witnesses = torch::empty({blocks, kRows}, input.options().dtype(torch::kInt32));
    switch (variant) {
        case 0: launch<false, false, false, 0, false, false, true >(input, cycles, witnesses); break;
        case 1: launch<true,  false, false, 0, false, false, true >(input, cycles, witnesses); break;
        case 2: launch<true,  false, false, 0, true,  true,  true >(input, cycles, witnesses); break;
#define LAUNCH_PADDING_CASE(ID, STORE_HI, WORD, PAIRED, DP4A_HI, DP4A_LO) \
        case ID: launch<true, true, STORE_HI, WORD, PAIRED, DP4A_HI, DP4A_LO>(input, cycles, witnesses); break
        LAUNCH_PADDING_CASE(3,  false, 0, false, false, true );
        LAUNCH_PADDING_CASE(4,  false, 1, false, false, true );
        LAUNCH_PADDING_CASE(5,  false, 2, false, false, true );
        LAUNCH_PADDING_CASE(6,  false, 3, false, false, true );
        LAUNCH_PADDING_CASE(7,  false, 0, true,  false, true );
        LAUNCH_PADDING_CASE(8,  false, 1, true,  false, true );
        LAUNCH_PADDING_CASE(9,  false, 2, true,  false, true );
        LAUNCH_PADDING_CASE(10, false, 3, true,  false, true );
        LAUNCH_PADDING_CASE(11, false, 0, true,  true,  true );
        LAUNCH_PADDING_CASE(12, false, 1, true,  true,  true );
        LAUNCH_PADDING_CASE(13, false, 2, true,  true,  true );
        LAUNCH_PADDING_CASE(14, false, 3, true,  true,  true );
        LAUNCH_PADDING_CASE(15, true,  0, false, false, true );
        LAUNCH_PADDING_CASE(16, true,  1, false, false, true );
        LAUNCH_PADDING_CASE(17, true,  2, false, false, true );
        LAUNCH_PADDING_CASE(18, true,  3, false, false, true );
        LAUNCH_PADDING_CASE(19, true,  0, true,  true,  true );
        LAUNCH_PADDING_CASE(20, true,  1, true,  true,  true );
        LAUNCH_PADDING_CASE(21, true,  2, true,  true,  true );
        LAUNCH_PADDING_CASE(22, true,  3, true,  true,  true );
        case 23: launch<true, false, false, 0, false, false, false>(input, cycles, witnesses); break;
        case 24: launch<true, false, false, 0, true,  false, false>(input, cycles, witnesses); break;
        LAUNCH_PADDING_CASE(25, false, 0, true,  false, false);
        LAUNCH_PADDING_CASE(26, false, 1, true,  false, false);
        LAUNCH_PADDING_CASE(27, false, 2, true,  false, false);
        LAUNCH_PADDING_CASE(28, false, 3, true,  false, false);
        case 29: launch_grouped<false>(input, cycles, witnesses); break;
        case 30: launch_grouped<true >(input, cycles, witnesses); break;
        case 31: launch_grouped<true >(input, cycles, witnesses); break;
#undef LAUNCH_PADDING_CASE
        default: TORCH_CHECK(false, "unknown variant");
    }
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return {cycles, witnesses};
}
"""


VARIANTS = {
    0: "empty",
    1: "flash/current-hybrid",
    2: "pro/current-paired-dp4a",
}
for variant, prefix in ((3, "flash/low/seq"), (7, "flash/low/paired"),
                        (11, "pro/low/paired"), (15, "flash/high/seq"),
                        (19, "pro/high/paired"), (25, "generic/low/paired")):
    for word in range(4):
        VARIANTS[variant + word] = f"{prefix}/word{word}"
VARIANTS[23] = "generic/current-sequential"
VARIANTS[24] = "generic/current-paired"
VARIANTS[29] = "flash/grouped-nibble/sequential"
VARIANTS[30] = "flash/grouped-nibble/paired"
VARIANTS[31] = "pro/grouped-nibble/paired"


def load_extension():
    cpp_src = "std::vector<torch::Tensor> run_dequant_padding_bench(torch::Tensor, int64_t, int64_t);"
    return load_inline(
        name="deepgemm_sm90_nvfp4_dequant_padding_selector",
        cpp_sources=cpp_src,
        cuda_sources=CUDA_SRC,
        functions=["run_dequant_padding_bench"],
        extra_include_paths=[os.path.join(REPO_ROOT, "deep_gemm", "include")],
        extra_cuda_cflags=["-O3", "-lineinfo", "--expt-relaxed-constexpr"],
        verbose=False,
    )


def make_rows(scale_pattern: str) -> torch.Tensor:
    torch.manual_seed(1234)
    rows = torch.randint(0, 256, (128, 80), dtype=torch.uint8, device="cuda")
    if scale_pattern == "random":
        scales = torch.randint(0, 127, (128, 8), dtype=torch.uint8, device="cuda")
    elif scale_pattern == "model":
        from deep_gemm.quantization_nvfp4 import fp32_to_ue4m3_ceil

        weights = torch.randn((128, 8, 16), dtype=torch.float32, device="cuda") * 0.05
        scales = fp32_to_ue4m3_ceil(weights.abs().amax(dim=-1) / 6.0)
    else:
        raise ValueError(scale_pattern)
    rows[:, 64:72] = scales
    rows[:, 72:80] = scales
    return rows.contiguous()


def add_padding_selectors(rows: torch.Tensor, store_hi: bool, word_in_quad: int) -> torch.Tensor:
    result = rows.clone()
    q_bytes = rows[:, :64].view(128, 16, 4)
    magnitudes = ((q_bytes >> 4) & 0x7) if store_hi else (q_bytes & 0x7)
    selectors = (
        magnitudes[..., 0].to(torch.int32)
        | (magnitudes[..., 1].to(torch.int32) << 4)
        | (magnitudes[..., 2].to(torch.int32) << 8)
        | (magnitudes[..., 3].to(torch.int32) << 12)
    )
    indices = torch.tensor(
        [word_in_quad + 4 * quad for quad in range(4)],
        dtype=torch.long,
        device=rows.device,
    )
    selected = selectors.index_select(1, indices)
    pairs = selected.view(128, 2, 2)
    packed = (pairs[..., 0] | (pairs[..., 1] << 16)).to(torch.int32)
    result[:, 72:80] = packed.contiguous().view(torch.uint8).view(128, 8)
    return result.contiguous().view(-1)


def group_nibbles_by_half(rows: torch.Tensor) -> torch.Tensor:
    result = rows.clone()
    q_bytes = rows[:, :64].view(128, 16, 4).to(torch.int32)

    def pack_nibbles(nibbles: torch.Tensor) -> torch.Tensor:
        return (
            nibbles[..., 0]
            | (nibbles[..., 1] << 4)
            | (nibbles[..., 2] << 8)
            | (nibbles[..., 3] << 12)
        )

    high = pack_nibbles((q_bytes >> 4) & 0xf)
    low = pack_nibbles(q_bytes & 0xf)
    grouped = (high | (low << 16)).to(torch.int32)
    result[:, :64] = grouped.contiguous().view(torch.uint8).view(128, 64)
    return result.contiguous().view(-1)


def variant_input(variant: int, current: torch.Tensor, inputs: dict, grouped: torch.Tensor) -> torch.Tensor:
    if variant < 3:
        return current.view(-1)
    if variant < 7:
        key = (False, variant - 3)
    elif variant < 11:
        key = (False, variant - 7)
    elif variant < 15:
        key = (False, variant - 11)
    elif variant < 19:
        key = (True, variant - 15)
    elif variant < 23:
        key = (True, variant - 19)
    elif variant < 25:
        return current.view(-1)
    elif variant >= 29:
        return grouped
    else:
        key = (False, variant - 25)
    return inputs[key]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--blocks", type=int, default=624)
    parser.add_argument("--rounds", type=int, default=11)
    parser.add_argument("--scale-pattern", choices=("random", "model"), default="model")
    parser.add_argument("--variants", type=int, nargs="+", default=list(VARIANTS))
    args = parser.parse_args()
    assert torch.cuda.get_device_capability()[0] == 9
    if any(variant not in VARIANTS for variant in args.variants):
        raise ValueError(f"variants must be in {list(VARIANTS)}")

    current = make_rows(args.scale_pattern)
    inputs = {
        (store_hi, word): add_padding_selectors(current, store_hi, word)
        for store_hi in (False, True)
        for word in range(4)
    }
    grouped = group_nibbles_by_half(current)
    ext = load_extension()
    reference = None
    samples = {variant: [] for variant in args.variants}
    for round_idx in range(args.rounds):
        order = args.variants if round_idx % 2 == 0 else list(reversed(args.variants))
        for variant in order:
            cycles, witnesses = ext.run_dequant_padding_bench(
                variant_input(variant, current, inputs, grouped), variant, args.blocks
            )
            torch.cuda.synchronize()
            if variant != 0:
                candidate = witnesses[0].cpu()
                if reference is None:
                    reference = candidate
                else:
                    torch.testing.assert_close(candidate, reference, rtol=0, atol=0)
            samples[variant].append(float(cycles.float().median().item()))

    empty = statistics.median(samples[0]) if 0 in samples else 0.0
    print(
        f"scale_pattern={args.scale_pattern} blocks={args.blocks} rounds={args.rounds} "
        f"empty_median={empty:.1f} cycles"
    )
    for variant in args.variants:
        center = statistics.median(samples[variant])
        print(
            f"{variant:2d} {VARIANTS[variant]:30s} raw={center:8.1f} "
            f"net={center - empty:8.1f} rounds={[round(value, 1) for value in samples[variant]]}"
        )


if __name__ == "__main__":
    main()
