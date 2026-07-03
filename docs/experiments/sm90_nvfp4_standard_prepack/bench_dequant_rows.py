"""Isolated SM90 NVFP4 packed-row decoder benchmark.

This is an experiment harness, not a production API. It compares the current
80-byte row layout with a lossless layout that duplicates the eight UE4M3
scale bytes into the existing eight-byte padding. The decoder selects the
copy with ``row & 8`` to spread warp-wide LDS.64 accesses across all banks.
"""

import argparse
import os
import statistics

import torch
from torch.utils.cpp_extension import load_inline


REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))


CUDA_SRC = r"""
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <c10/cuda/CUDAException.h>
#include <deep_gemm/quantization/nvfp4_dequant.cuh>

namespace {

constexpr int kRows = 128;
constexpr int kPackedRowBytes = 80;
constexpr int kFp8RowBytes = 128;

template <bool kBraidedSelectors, bool kUseDp4aHi, bool kUseDp4aLo>
__device__ __forceinline__ uint2 decode_word(uint32_t q, const uint2& lut) {
    if constexpr (kBraidedSelectors) {
        const uint32_t sel_hi = q & 0x00007777u;
        const uint32_t sel_lo = (q >> 16) & 0x00007777u;
        uint32_t out_hi = deep_gemm::nvfp4::byte_perm_unchecked(lut.x, lut.y, sel_hi);
        uint32_t out_lo = deep_gemm::nvfp4::byte_perm_unchecked(lut.x, lut.y, sel_lo);
        out_hi |= q & 0x80808080u;
        out_lo |= (q << 4) & 0x80808080u;
        return make_uint2(out_hi, out_lo);
    } else {
        return deep_gemm::nvfp4::dequant_nvfp4_to_fp8_pair_with_lut<
            kUseDp4aHi, kUseDp4aLo>(q, lut);
    }
}

__device__ __forceinline__ uint32_t double_positive_e4m3_code(uint32_t code) {
    const uint32_t doubled = code < 8u ? code * 2u : code + 8u;
    return min(doubled, 127u);
}

__device__ __forceinline__ uint2 synthesize_nvfp4_lut(uint32_t scale) {
    scale &= 0x7fu;
    const uint32_t half = scale < 16u
        ? (scale >> 1) + static_cast<uint32_t>((scale & 3u) == 3u)
        : scale - 8u;
    const uint32_t mantissa = scale & 7u;
    const uint32_t one_and_half = scale < 8u
        ? ((3u * scale) >> 1) + static_cast<uint32_t>((scale & 3u) == 1u)
        : min(scale + 4u + static_cast<uint32_t>(mantissa >= 1u && mantissa <= 5u), 127u);
    const uint32_t twice = double_positive_e4m3_code(scale);
    const uint32_t three = scale < 8u
        ? (scale <= 5u ? 3u * scale : scale + 11u)
        : double_positive_e4m3_code(one_and_half);
    const uint32_t four = double_positive_e4m3_code(twice);
    const uint32_t six = double_positive_e4m3_code(three);
    return make_uint2(
        (half << 8) | (scale << 16) | (one_and_half << 24),
        twice | (three << 8) | (four << 16) | (six << 24));
}

template <bool kSynthesizeLut>
__device__ __forceinline__ uint2 get_lut(
        uint32_t scale, const uint2* __restrict__ lut_smem) {
    if constexpr (kSynthesizeLut)
        return synthesize_nvfp4_lut(scale);
    return lut_smem[scale & 0x7fu];
}

__device__ __forceinline__ uint2 load_lut_ordered(
        const uint2* __restrict__ lut_smem, uint32_t scale) {
    uint2 lut;
    asm volatile(
        "ld.shared.v2.u32 {%0, %1}, [%2];"
        : "=r"(lut.x), "=r"(lut.y)
        : "l"(__cvta_generic_to_shared(lut_smem + (scale & 0x7fu)))
        : "memory");
    return lut;
}

__device__ __forceinline__ void store_quad_half_ordered(
        uint8_t* __restrict__ dst, const uint2& x, const uint2& y) {
    asm volatile(
        "st.shared.v4.u32 [%0], {%1, %2, %3, %4};"
        :: "l"(__cvta_generic_to_shared(dst)),
           "r"(x.x), "r"(x.y), "r"(y.x), "r"(y.y)
        : "memory");
}

template <bool kBraidedSelectors, bool kUseDp4aHi, bool kUseDp4aLo>
__device__ __forceinline__ void decode_and_store_quad(
        uint8_t* __restrict__ fp8_dst,
        const uint4& q,
        const uint2& lut0,
        const uint2& lut1,
        int scale_i0,
        uint32_t row_swizzle) {
    const uint2 q0 = decode_word<kBraidedSelectors, kUseDp4aHi, kUseDp4aLo>(q.x, lut0);
    const uint2 q1 = decode_word<kBraidedSelectors, kUseDp4aHi, kUseDp4aLo>(q.y, lut0);
    *reinterpret_cast<uint4*>(fp8_dst + ((scale_i0 * 16) ^ row_swizzle)) =
        make_uint4(q0.x, q0.y, q1.x, q1.y);

    const uint2 q2 = decode_word<kBraidedSelectors, kUseDp4aHi, kUseDp4aLo>(q.z, lut1);
    const uint2 q3 = decode_word<kBraidedSelectors, kUseDp4aHi, kUseDp4aLo>(q.w, lut1);
    *reinterpret_cast<uint4*>(fp8_dst + (((scale_i0 + 1) * 16) ^ row_swizzle)) =
        make_uint4(q2.x, q2.y, q3.x, q3.y);
}

template <int kQuad, bool kBraidedSelectors, bool kUseDp4aHi, bool kUseDp4aLo>
__device__ __forceinline__ void decode_row_next_lut_pair(
        uint8_t* __restrict__ fp8_dst,
        const uint4 (&fp4_quads)[4],
        uint32_t scale_word_lo,
        uint32_t scale_word_hi,
        const uint2* __restrict__ lut_smem,
        uint2 lut0,
        uint2 lut1,
        uint32_t row_swizzle) {
    uint2 next_lut0;
    uint2 next_lut1;
    if constexpr (kQuad + 1 < 4) {
        constexpr int kNextScaleI0 = (kQuad + 1) * 2;
        constexpr int kNextScaleI1 = kNextScaleI0 + 1;
        const uint32_t next_scale_word = kQuad + 1 < 2 ? scale_word_lo : scale_word_hi;
        const uint32_t next_scale0 =
            (next_scale_word >> ((kNextScaleI0 & 3) * 8)) & 0xffu;
        const uint32_t next_scale1 =
            (next_scale_word >> ((kNextScaleI1 & 3) * 8)) & 0xffu;
        next_lut0 = lut_smem[next_scale0 & 0x7fu];
        next_lut1 = lut_smem[next_scale1 & 0x7fu];
    }

    decode_and_store_quad<kBraidedSelectors, kUseDp4aHi, kUseDp4aLo>(
        fp8_dst, fp4_quads[kQuad], lut0, lut1, kQuad * 2, row_swizzle);

    if constexpr (kQuad + 1 < 4) {
        decode_row_next_lut_pair<kQuad + 1, kBraidedSelectors, kUseDp4aHi, kUseDp4aLo>(
            fp8_dst, fp4_quads, scale_word_lo, scale_word_hi, lut_smem,
            next_lut0, next_lut1, row_swizzle);
    }
}

template <bool kBraidedSelectors, bool kUseScaleReplica, bool kPreloadSecondLut,
          bool kUseDp4aHi, bool kUseDp4aLo, bool kSynthesizeLut,
          bool kNextQuadLutPipeline, bool kHalfQuadNextLutPipeline,
          bool kPreloadAllLuts>
__device__ __forceinline__ void decode_row(
        uint8_t* __restrict__ fp8,
        const uint8_t* __restrict__ packed,
        uint32_t row,
        const uint2* __restrict__ lut_smem) {
    const uint8_t* __restrict__ row_ptr = packed + row * kPackedRowBytes;
    const uint4* __restrict__ fp4_src = reinterpret_cast<const uint4*>(row_ptr);
    uint4 fp4_quads[4];
#pragma unroll
    for (int i = 0; i < 4; ++i)
        fp4_quads[i] = fp4_src[i];

    const uint32_t replica_offset = kUseScaleReplica ? (row & 8u) : 0u;
    const uint2 scale_words =
        *reinterpret_cast<const uint2*>(row_ptr + 64u + replica_offset);
    const uint32_t scale_word_lo = scale_words.x;
    const uint32_t scale_word_hi = scale_words.y;

    uint8_t* __restrict__ fp8_dst = fp8 + row * kFp8RowBytes;
    const uint32_t row_swizzle = (row & 7u) << 4;
    if constexpr (kPreloadAllLuts) {
        static_assert(kBraidedSelectors && kPreloadSecondLut && !kSynthesizeLut);
        uint2 luts[8];
#pragma unroll
        for (int scale_i = 0; scale_i < 8; ++scale_i) {
            const uint32_t scale_word = scale_i < 4 ? scale_word_lo : scale_word_hi;
            const uint32_t scale = (scale_word >> ((scale_i & 3) * 8)) & 0x7fu;
            luts[scale_i] = lut_smem[scale];
        }
#pragma unroll
        for (int quad_i = 0; quad_i < 4; ++quad_i) {
            decode_and_store_quad<kBraidedSelectors, kUseDp4aHi, kUseDp4aLo>(
                fp8_dst, fp4_quads[quad_i], luts[quad_i * 2],
                luts[quad_i * 2 + 1], quad_i * 2, row_swizzle);
        }
        return;
    }

    if constexpr (kHalfQuadNextLutPipeline) {
        static_assert(kPreloadSecondLut && !kSynthesizeLut);
        uint2 lut0 = load_lut_ordered(lut_smem, scale_word_lo);
        uint2 lut1 = load_lut_ordered(lut_smem, scale_word_lo >> 8);
#pragma unroll
        for (int quad_i = 0; quad_i < 4; ++quad_i) {
            const uint4 q = fp4_quads[quad_i];
            const int scale_i0 = quad_i * 2;
            const uint2 q0 =
                decode_word<kBraidedSelectors, kUseDp4aHi, kUseDp4aLo>(q.x, lut0);
            const uint2 q1 =
                decode_word<kBraidedSelectors, kUseDp4aHi, kUseDp4aLo>(q.y, lut0);
            store_quad_half_ordered(
                fp8_dst + ((scale_i0 * 16) ^ row_swizzle), q0, q1);

            uint2 next_lut0 = lut0;
            uint2 next_lut1 = lut1;
            if (quad_i + 1 < 4) {
                const int next_scale_i0 = (quad_i + 1) * 2;
                const uint32_t next_scale_word =
                    quad_i + 1 < 2 ? scale_word_lo : scale_word_hi;
                next_lut0 = load_lut_ordered(
                    lut_smem,
                    next_scale_word >> ((next_scale_i0 & 3) * 8));
                next_lut1 = load_lut_ordered(
                    lut_smem,
                    next_scale_word >> (((next_scale_i0 + 1) & 3) * 8));
            }

            const uint2 q2 =
                decode_word<kBraidedSelectors, kUseDp4aHi, kUseDp4aLo>(q.z, lut1);
            const uint2 q3 =
                decode_word<kBraidedSelectors, kUseDp4aHi, kUseDp4aLo>(q.w, lut1);
            store_quad_half_ordered(
                fp8_dst + (((scale_i0 + 1) * 16) ^ row_swizzle), q2, q3);
            lut0 = next_lut0;
            lut1 = next_lut1;
        }
        return;
    }

    if constexpr (kNextQuadLutPipeline) {
        static_assert(kPreloadSecondLut && !kSynthesizeLut);
        const uint2 lut0 = lut_smem[scale_word_lo & 0x7fu];
        const uint2 lut1 = lut_smem[(scale_word_lo >> 8) & 0x7fu];
        decode_row_next_lut_pair<0, kBraidedSelectors, kUseDp4aHi, kUseDp4aLo>(
            fp8_dst, fp4_quads, scale_word_lo, scale_word_hi, lut_smem,
            lut0, lut1, row_swizzle);
        return;
    }

#pragma unroll
    for (int quad_i = 0; quad_i < 4; ++quad_i) {
        const uint4 q = fp4_quads[quad_i];
        const uint32_t scale_word = quad_i < 2 ? scale_word_lo : scale_word_hi;
        const int scale_i0 = quad_i * 2;
        const int scale_i1 = scale_i0 + 1;
        const uint32_t scale0 = (scale_word >> ((scale_i0 & 3) * 8)) & 0xffu;

        if constexpr (kPreloadSecondLut) {
            const uint32_t scale1 = (scale_word >> ((scale_i1 & 3) * 8)) & 0xffu;
            const uint2 lut0 = get_lut<kSynthesizeLut>(scale0, lut_smem);
            const uint2 lut1 = get_lut<kSynthesizeLut>(scale1, lut_smem);
            const uint2 q0 = decode_word<kBraidedSelectors, kUseDp4aHi, kUseDp4aLo>(q.x, lut0);
            const uint2 q1 = decode_word<kBraidedSelectors, kUseDp4aHi, kUseDp4aLo>(q.y, lut0);
            *reinterpret_cast<uint4*>(fp8_dst + ((scale_i0 * 16) ^ row_swizzle)) =
                make_uint4(q0.x, q0.y, q1.x, q1.y);

            const uint2 q2 = decode_word<kBraidedSelectors, kUseDp4aHi, kUseDp4aLo>(q.z, lut1);
            const uint2 q3 = decode_word<kBraidedSelectors, kUseDp4aHi, kUseDp4aLo>(q.w, lut1);
            *reinterpret_cast<uint4*>(fp8_dst + ((scale_i1 * 16) ^ row_swizzle)) =
                make_uint4(q2.x, q2.y, q3.x, q3.y);
        } else {
            const uint2 lut0 = get_lut<kSynthesizeLut>(scale0, lut_smem);
            const uint2 q0 = decode_word<kBraidedSelectors, kUseDp4aHi, kUseDp4aLo>(q.x, lut0);
            const uint2 q1 = decode_word<kBraidedSelectors, kUseDp4aHi, kUseDp4aLo>(q.y, lut0);
            *reinterpret_cast<uint4*>(fp8_dst + ((scale_i0 * 16) ^ row_swizzle)) =
                make_uint4(q0.x, q0.y, q1.x, q1.y);

            const uint32_t scale1 = (scale_word >> ((scale_i1 & 3) * 8)) & 0xffu;
            const uint2 lut1 = get_lut<kSynthesizeLut>(scale1, lut_smem);
            const uint2 q2 = decode_word<kBraidedSelectors, kUseDp4aHi, kUseDp4aLo>(q.z, lut1);
            const uint2 q3 = decode_word<kBraidedSelectors, kUseDp4aHi, kUseDp4aLo>(q.w, lut1);
            *reinterpret_cast<uint4*>(fp8_dst + ((scale_i1 * 16) ^ row_swizzle)) =
                make_uint4(q2.x, q2.y, q3.x, q3.y);
        }
    }
}

template <bool kDecode, bool kBraidedSelectors, bool kUseScaleReplica, bool kPreloadSecondLut,
          bool kUseDp4aHi, bool kUseDp4aLo, bool kSynthesizeLut,
          bool kNextQuadLutPipeline, bool kHalfQuadNextLutPipeline,
          bool kPreloadAllLuts>
__global__ __launch_bounds__(128) void bench_kernel(
        const uint8_t* __restrict__ input,
        int64_t* __restrict__ cycles,
        int32_t* __restrict__ witnesses) {
    __shared__ __align__(16) uint8_t packed[kRows * kPackedRowBytes];
    __shared__ __align__(16) uint8_t fp8[kRows * kFp8RowBytes];
    __shared__ __align__(16) uint2 lut[128];

    const uint32_t tid = threadIdx.x;
    for (int i = tid; i < kRows * kPackedRowBytes; i += blockDim.x)
        packed[i] = input[i];
    lut[tid] = deep_gemm::nvfp4::kE2M1AndUe4m3ToFp8Lut[tid];
    for (int i = tid; i < kRows * kFp8RowBytes; i += blockDim.x)
        fp8[i] = 0;
    __syncthreads();

    uint64_t start = 0;
    if (tid == 0)
        start = clock64();
    __syncthreads();
    if constexpr (kDecode)
        decode_row<kBraidedSelectors, kUseScaleReplica, kPreloadSecondLut,
                   kUseDp4aHi, kUseDp4aLo, kSynthesizeLut,
                   kNextQuadLutPipeline, kHalfQuadNextLutPipeline,
                   kPreloadAllLuts>(
            fp8, packed, tid, lut);
    __syncthreads();
    if (tid == 0)
        cycles[blockIdx.x] = static_cast<int64_t>(clock64() - start);

    if (blockIdx.x == 0) {
        const uint32_t* row_words =
            reinterpret_cast<const uint32_t*>(fp8 + tid * kFp8RowBytes);
#pragma unroll
        for (int i = 0; i < kFp8RowBytes / static_cast<int>(sizeof(uint32_t)); ++i)
            witnesses[tid * (kFp8RowBytes / sizeof(uint32_t)) + i] =
                static_cast<int32_t>(row_words[i]);
    }
}

template <bool kDecode, bool kBraidedSelectors, bool kUseScaleReplica, bool kPreloadSecondLut,
          bool kUseDp4aHi, bool kUseDp4aLo, bool kSynthesizeLut = false,
          bool kNextQuadLutPipeline = false,
          bool kHalfQuadNextLutPipeline = false,
          bool kPreloadAllLuts = false>
void launch(const torch::Tensor& input, torch::Tensor& cycles, torch::Tensor& witnesses) {
    const int blocks = static_cast<int>(cycles.numel());
    bench_kernel<kDecode, kBraidedSelectors, kUseScaleReplica, kPreloadSecondLut,
                 kUseDp4aHi, kUseDp4aLo, kSynthesizeLut,
                 kNextQuadLutPipeline,
                 kHalfQuadNextLutPipeline,
                 kPreloadAllLuts><<<blocks, 128>>>(
        input.data_ptr<uint8_t>(), cycles.data_ptr<int64_t>(),
        witnesses.data_ptr<int32_t>());
}

}  // namespace

std::vector<torch::Tensor> run_dequant_bench(torch::Tensor input, int64_t variant, int64_t blocks) {
    TORCH_CHECK(input.is_cuda() && input.scalar_type() == torch::kUInt8);
    TORCH_CHECK(input.numel() == kRows * kPackedRowBytes);
    auto cycles = torch::empty({blocks}, input.options().dtype(torch::kInt64));
    auto witnesses = torch::empty(
        {kRows, kFp8RowBytes / static_cast<int>(sizeof(int32_t))},
        input.options().dtype(torch::kInt32));
    switch (variant) {
        case 0: launch<false, false, false, false, false, false>(input, cycles, witnesses); break;
        case 1: launch<true,  false, false, false, false, false>(input, cycles, witnesses); break;
        case 2: launch<true,  false, true,  false, false, false>(input, cycles, witnesses); break;
        case 3: launch<true,  false, false, true,  false, false>(input, cycles, witnesses); break;
        case 4: launch<true,  false, true,  true,  false, false>(input, cycles, witnesses); break;
        case 5: launch<true,  false, false, true,  true,  true >(input, cycles, witnesses); break;
        case 6: launch<true,  false, true,  true,  true,  true >(input, cycles, witnesses); break;
        case 7: launch<true,  false, false, false, false, true >(input, cycles, witnesses); break;
        case 8: launch<true,  false, true,  false, false, true >(input, cycles, witnesses); break;
        case 9: launch<true,  true,  false, false, false, false>(input, cycles, witnesses); break;
        case 10: launch<true, true,  false, true,  false, false>(input, cycles, witnesses); break;
        case 11: launch<true, false, false, true,  true,  true, true>(input, cycles, witnesses); break;
        case 12: launch<true, false, false, true,  true,  true, false, true>(input, cycles, witnesses); break;
        case 13: launch<true, false, false, true,  true,  true, false, false, true>(input, cycles, witnesses); break;
        case 14: launch<true, true,  false, true,  false, false, false, true>(input, cycles, witnesses); break;
        case 15: launch<true, true,  false, true,  false, false, false, false, true>(input, cycles, witnesses); break;
        case 16: launch<true, true,  false, true,  false, false, false, false, false, true>(input, cycles, witnesses); break;
        default: TORCH_CHECK(false, "unknown variant");
    }
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return {cycles, witnesses};
}
"""


VARIANTS = {
    0: "empty",
    1: "prmt/current",
    2: "prmt/scale-replica",
    3: "paired-prmt/current",
    4: "paired-prmt/scale-replica",
    5: "paired-dp4a/current",
    6: "paired-dp4a/scale-replica",
    7: "low-hybrid/current",
    8: "low-hybrid/scale-replica",
    9: "braided-selectors",
    10: "paired/braided-selectors",
    11: "paired-dp4a/integer-lut",
    12: "paired-dp4a/next-quad-lut",
    13: "paired-dp4a/half-quad-next-lut",
    14: "braided/next-quad-lut",
    15: "braided/half-quad-next-lut",
    16: "braided/preload-all-luts",
}


def load_extension():
    cpp_src = "std::vector<torch::Tensor> run_dequant_bench(torch::Tensor, int64_t, int64_t);"
    return load_inline(
        name="deepgemm_sm90_nvfp4_dequant_rows_bench",
        cpp_sources=cpp_src,
        cuda_sources=CUDA_SRC,
        functions=["run_dequant_bench"],
        extra_include_paths=[os.path.join(REPO_ROOT, "deep_gemm", "include")],
        extra_cuda_cflags=["-O3", "-lineinfo", "--expt-relaxed-constexpr"],
        verbose=False,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--blocks", type=int, default=624)
    parser.add_argument("--rounds", type=int, default=9)
    parser.add_argument("--variants", type=int, nargs="+", default=list(VARIANTS))
    parser.add_argument(
        "--scale-distribution",
        choices=("uniform", "model", "exhaustive"),
        default="uniform",
    )
    args = parser.parse_args()

    assert torch.cuda.get_device_capability()[0] == 9
    torch.manual_seed(1234)
    rows = torch.randint(0, 256, (128, 80), dtype=torch.uint8)
    if args.scale_distribution == "model":
        rows[:, 64:72] = torch.randint(4, 20, (128, 8), dtype=torch.uint8)
    elif args.scale_distribution == "exhaustive":
        word0 = torch.tensor([0x04, 0x15, 0x26, 0x37], dtype=torch.uint8)
        word1 = torch.tensor([0x8C, 0x9D, 0xAE, 0xBF], dtype=torch.uint8)
        rows[:, :64] = torch.cat([word0, word1] * 8).view(1, 64)
        rows[:, 64:72] = torch.arange(1024).remainder(127).to(torch.uint8).view(128, 8)
    else:
        rows[:, 64:72] = torch.randint(0, 127, (128, 8), dtype=torch.uint8)
    rows[:, 72:80] = rows[:, 64:72]
    rows = rows.cuda()
    packed = rows.contiguous().view(-1)

    braided_rows = rows.clone()
    marlin_bytes = rows[:, :64].view(128, 16, 4)
    values = torch.cat([(marlin_bytes >> 4) & 0x0F, marlin_bytes & 0x0F], dim=-1)
    magnitudes = values & 0x07
    signs = values >> 3
    sign_order = torch.stack(
        [signs[..., 4], signs[..., 0], signs[..., 5], signs[..., 1],
         signs[..., 6], signs[..., 2], signs[..., 7], signs[..., 3]],
        dim=-1,
    )
    selector_nibbles = magnitudes | (sign_order << 3)
    braided_bytes = selector_nibbles[..., 0::2] | (selector_nibbles[..., 1::2] << 4)
    braided_rows[:, :64] = braided_bytes.view(128, 64)
    braided = braided_rows.contiguous().view(-1)
    ext = load_extension()

    reference = None
    variant_ids = args.variants
    if any(variant not in VARIANTS for variant in variant_ids):
        raise ValueError(f"variants must be in {list(VARIANTS)}")
    samples = {variant: [] for variant in variant_ids}
    for round_idx in range(args.rounds):
        order = variant_ids if round_idx % 2 == 0 else list(reversed(variant_ids))
        for variant in order:
            input_rows = braided if variant in (9, 10, 14, 15, 16) else packed
            cycles, witnesses = ext.run_dequant_bench(input_rows, variant, args.blocks)
            torch.cuda.synchronize()
            if variant != 0:
                candidate = witnesses.cpu()
                if reference is None:
                    reference = candidate
                else:
                    torch.testing.assert_close(candidate, reference, rtol=0, atol=0)
            samples[variant].append(float(cycles.float().median().item()))

    empty = statistics.median(samples[0]) if 0 in samples else 0.0
    print(
        f"scales={args.scale_distribution} blocks={args.blocks} "
        f"rounds={args.rounds} empty_median={empty:.1f} cycles"
    )
    for variant in variant_ids:
        name = VARIANTS[variant]
        center = statistics.median(samples[variant])
        net = center - empty
        print(
            f"{variant:2d} {name:30s} raw={center:8.1f} net={net:8.1f} "
            f"rounds={[round(value, 1) for value in samples[variant]]}"
        )


if __name__ == "__main__":
    main()
