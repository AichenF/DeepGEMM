"""Isolated SM90 NVFP4 benchmark for a 112-byte compact selector row.

The candidate keeps all 64 packed E2M1 bytes, adds one 16-bit magnitude
selector per packed word, and keeps the eight standard UE4M3 scales.  It is a
lossless integer-only deployment layout and is not wired into production.
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
constexpr int kCurrentRowBytes = 80;
constexpr int kCompactRowBytes = 112;
constexpr int kFp8RowBytes = 128;

template <bool kUseDp4aHi, bool kUseDp4aLo>
__device__ __forceinline__ uint2 decode_current_word(uint32_t q, const uint2& lut) {
    return deep_gemm::nvfp4::dequant_nvfp4_to_fp8_pair_with_lut<
        kUseDp4aHi, kUseDp4aLo>(q, lut);
}

template <bool kStoreHi, bool kUseDp4aHi, bool kUseDp4aLo>
__device__ __forceinline__ uint2 decode_compact_word(
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

template <bool kPaired, bool kUseDp4aHi, bool kUseDp4aLo>
__device__ __forceinline__ void decode_current_row(
        uint8_t* __restrict__ fp8,
        const uint8_t* __restrict__ packed,
        uint32_t row,
        const uint2* __restrict__ lut_smem) {
    const uint8_t* __restrict__ row_ptr = packed + row * kCurrentRowBytes;
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
        const uint32_t scale0 = (scale_word >> ((scale_i0 & 3) * 8)) & 0xffu;

        if constexpr (kPaired) {
            const uint32_t scale1 = (scale_word >> ((scale_i1 & 3) * 8)) & 0xffu;
            const uint2 lut0 = lut_smem[scale0 & 0x7fu];
            const uint2 lut1 = lut_smem[scale1 & 0x7fu];
            const uint2 q0 = decode_current_word<kUseDp4aHi, kUseDp4aLo>(q.x, lut0);
            const uint2 q1 = decode_current_word<kUseDp4aHi, kUseDp4aLo>(q.y, lut0);
            *reinterpret_cast<uint4*>(fp8_dst + ((scale_i0 * 16) ^ row_swizzle)) =
                make_uint4(q0.x, q0.y, q1.x, q1.y);
            const uint2 q2 = decode_current_word<kUseDp4aHi, kUseDp4aLo>(q.z, lut1);
            const uint2 q3 = decode_current_word<kUseDp4aHi, kUseDp4aLo>(q.w, lut1);
            *reinterpret_cast<uint4*>(fp8_dst + ((scale_i1 * 16) ^ row_swizzle)) =
                make_uint4(q2.x, q2.y, q3.x, q3.y);
        } else {
            const uint2 lut0 = lut_smem[scale0 & 0x7fu];
            const uint2 q0 = decode_current_word<kUseDp4aHi, kUseDp4aLo>(q.x, lut0);
            const uint2 q1 = decode_current_word<kUseDp4aHi, kUseDp4aLo>(q.y, lut0);
            *reinterpret_cast<uint4*>(fp8_dst + ((scale_i0 * 16) ^ row_swizzle)) =
                make_uint4(q0.x, q0.y, q1.x, q1.y);
            const uint32_t scale1 = (scale_word >> ((scale_i1 & 3) * 8)) & 0xffu;
            const uint2 lut1 = lut_smem[scale1 & 0x7fu];
            const uint2 q2 = decode_current_word<kUseDp4aHi, kUseDp4aLo>(q.z, lut1);
            const uint2 q3 = decode_current_word<kUseDp4aHi, kUseDp4aLo>(q.w, lut1);
            *reinterpret_cast<uint4*>(fp8_dst + ((scale_i1 * 16) ^ row_swizzle)) =
                make_uint4(q2.x, q2.y, q3.x, q3.y);
        }
    }
}

template <bool kStoreHi, bool kPaired, bool kUseDp4aHi, bool kUseDp4aLo>
__device__ __forceinline__ void decode_compact_row(
        uint8_t* __restrict__ fp8,
        const uint8_t* __restrict__ packed,
        uint32_t row,
        const uint2* __restrict__ lut_smem) {
    const uint8_t* __restrict__ row_ptr = packed + row * kCompactRowBytes;
    const uint4* __restrict__ fp4_src = reinterpret_cast<const uint4*>(row_ptr);
    uint4 fp4_quads[4];
#pragma unroll
    for (int i = 0; i < 4; ++i)
        fp4_quads[i] = fp4_src[i];

    const uint2 scale_words = *reinterpret_cast<const uint2*>(row_ptr + 96);
    const uint32_t scale_word_lo = scale_words.x;
    const uint32_t scale_word_hi = scale_words.y;
    uint8_t* __restrict__ fp8_dst = fp8 + row * kFp8RowBytes;
    const uint32_t row_swizzle = (row & 7u) << 4;

#pragma unroll
    for (int quad_i = 0; quad_i < 4; ++quad_i) {
        const uint4 q = fp4_quads[quad_i];
        const uint2 selector_pairs =
            *reinterpret_cast<const uint2*>(row_ptr + 64 + quad_i * 8);
        const uint32_t scale_word = quad_i < 2 ? scale_word_lo : scale_word_hi;
        const int scale_i0 = quad_i * 2;
        const int scale_i1 = scale_i0 + 1;
        const uint32_t scale0 = (scale_word >> ((scale_i0 & 3) * 8)) & 0xffu;

        if constexpr (kPaired) {
            const uint32_t scale1 = (scale_word >> ((scale_i1 & 3) * 8)) & 0xffu;
            const uint2 lut0 = lut_smem[scale0 & 0x7fu];
            const uint2 lut1 = lut_smem[scale1 & 0x7fu];
            const uint2 q0 = decode_compact_word<kStoreHi, kUseDp4aHi, kUseDp4aLo>(
                q.x, selector_pairs.x, lut0);
            const uint2 q1 = decode_compact_word<kStoreHi, kUseDp4aHi, kUseDp4aLo>(
                q.y, selector_pairs.x >> 16, lut0);
            *reinterpret_cast<uint4*>(fp8_dst + ((scale_i0 * 16) ^ row_swizzle)) =
                make_uint4(q0.x, q0.y, q1.x, q1.y);
            const uint2 q2 = decode_compact_word<kStoreHi, kUseDp4aHi, kUseDp4aLo>(
                q.z, selector_pairs.y, lut1);
            const uint2 q3 = decode_compact_word<kStoreHi, kUseDp4aHi, kUseDp4aLo>(
                q.w, selector_pairs.y >> 16, lut1);
            *reinterpret_cast<uint4*>(fp8_dst + ((scale_i1 * 16) ^ row_swizzle)) =
                make_uint4(q2.x, q2.y, q3.x, q3.y);
        } else {
            const uint2 lut0 = lut_smem[scale0 & 0x7fu];
            const uint2 q0 = decode_compact_word<kStoreHi, kUseDp4aHi, kUseDp4aLo>(
                q.x, selector_pairs.x, lut0);
            const uint2 q1 = decode_compact_word<kStoreHi, kUseDp4aHi, kUseDp4aLo>(
                q.y, selector_pairs.x >> 16, lut0);
            *reinterpret_cast<uint4*>(fp8_dst + ((scale_i0 * 16) ^ row_swizzle)) =
                make_uint4(q0.x, q0.y, q1.x, q1.y);
            const uint32_t scale1 = (scale_word >> ((scale_i1 & 3) * 8)) & 0xffu;
            const uint2 lut1 = lut_smem[scale1 & 0x7fu];
            const uint2 q2 = decode_compact_word<kStoreHi, kUseDp4aHi, kUseDp4aLo>(
                q.z, selector_pairs.y, lut1);
            const uint2 q3 = decode_compact_word<kStoreHi, kUseDp4aHi, kUseDp4aLo>(
                q.w, selector_pairs.y >> 16, lut1);
            *reinterpret_cast<uint4*>(fp8_dst + ((scale_i1 * 16) ^ row_swizzle)) =
                make_uint4(q2.x, q2.y, q3.x, q3.y);
        }
    }
}

template <bool kDecode, bool kCompact, bool kStoreHi, bool kPaired,
          bool kUseDp4aHi, bool kUseDp4aLo>
__global__ __launch_bounds__(128) void bench_kernel(
        const uint8_t* __restrict__ input,
        int64_t* __restrict__ cycles,
        int32_t* __restrict__ witnesses) {
    constexpr int kRowBytes = kCompact ? kCompactRowBytes : kCurrentRowBytes;
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
        if constexpr (kCompact) {
            decode_compact_row<kStoreHi, kPaired, kUseDp4aHi, kUseDp4aLo>(
                fp8, packed, tid, lut);
        } else {
            decode_current_row<kPaired, kUseDp4aHi, kUseDp4aLo>(
                fp8, packed, tid, lut);
        }
    }
    __syncthreads();
    if (tid == 0)
        cycles[blockIdx.x] = static_cast<int64_t>(clock64() - start);

    const uint32_t* row_words = reinterpret_cast<const uint32_t*>(fp8 + tid * kFp8RowBytes);
    witnesses[blockIdx.x * kRows + tid] = static_cast<int32_t>(row_words[tid & 31u]);
}

template <bool kDecode, bool kCompact, bool kStoreHi, bool kPaired,
          bool kUseDp4aHi, bool kUseDp4aLo>
void launch(const torch::Tensor& input, torch::Tensor& cycles, torch::Tensor& witnesses) {
    constexpr int kRowBytes = kCompact ? kCompactRowBytes : kCurrentRowBytes;
    constexpr int kSharedBytes =
        kRows * (kRowBytes + kFp8RowBytes) + 128 * static_cast<int>(sizeof(uint2));
    bench_kernel<kDecode, kCompact, kStoreHi, kPaired, kUseDp4aHi, kUseDp4aLo>
        <<<static_cast<int>(cycles.numel()), 128, kSharedBytes>>>(
            input.data_ptr<uint8_t>(), cycles.data_ptr<int64_t>(),
            witnesses.data_ptr<int32_t>());
}

}  // namespace

std::vector<torch::Tensor> run_dequant_compact_bench(
        torch::Tensor input, int64_t variant, int64_t blocks) {
    TORCH_CHECK(input.is_cuda() && input.scalar_type() == torch::kUInt8);
    auto cycles = torch::empty({blocks}, input.options().dtype(torch::kInt64));
    auto witnesses = torch::empty({blocks, kRows}, input.options().dtype(torch::kInt32));
    switch (variant) {
        case 0:  launch<false, false, false, false, false, true >(input, cycles, witnesses); break;
        case 1:  launch<true,  false, false, false, false, true >(input, cycles, witnesses); break;
        case 2:  launch<true,  false, false, true,  true,  true >(input, cycles, witnesses); break;
        case 3:  launch<true,  true,  false, false, false, false>(input, cycles, witnesses); break;
        case 4:  launch<true,  true,  false, true,  true,  false>(input, cycles, witnesses); break;
        case 5:  launch<true,  true,  true,  false, false, true >(input, cycles, witnesses); break;
        case 6:  launch<true,  true,  true,  true,  false, true >(input, cycles, witnesses); break;
        case 7:  launch<true,  true,  false, true,  false, false>(input, cycles, witnesses); break;
        case 8:  launch<true,  true,  true,  true,  false, true >(input, cycles, witnesses); break;
        case 9:  launch<true,  true,  false, false, true,  false>(input, cycles, witnesses); break;
        case 10: launch<true,  true,  true,  false, false, true >(input, cycles, witnesses); break;
        default: TORCH_CHECK(false, "unknown variant");
    }
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return {cycles, witnesses};
}
"""


VARIANTS = {
    0: "empty/current-row",
    1: "flash/current-hybrid",
    2: "pro/current-paired-dp4a",
    3: "flash/store-low/sequential",
    4: "pro/store-low/paired",
    5: "flash/store-high/sequential",
    6: "pro/store-high/paired",
    7: "flash/store-low/paired",
    8: "flash/store-high/paired",
    9: "pro/store-low/sequential",
    10: "pro/store-high/sequential",
}


def load_extension():
    cpp_src = "std::vector<torch::Tensor> run_dequant_compact_bench(torch::Tensor, int64_t, int64_t);"
    return load_inline(
        name="deepgemm_sm90_nvfp4_dequant_compact_selector",
        cpp_sources=cpp_src,
        cuda_sources=CUDA_SRC,
        functions=["run_dequant_compact_bench"],
        extra_include_paths=[os.path.join(REPO_ROOT, "deep_gemm", "include")],
        extra_cuda_cflags=["-O3", "-lineinfo", "--expt-relaxed-constexpr"],
        verbose=False,
    )


def make_current_rows(scale_pattern: str) -> torch.Tensor:
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


def make_compact_rows(current: torch.Tensor, store_hi: bool) -> torch.Tensor:
    q_bytes = current[:, :64].contiguous().view(128, 16, 4)
    shifts = torch.tensor([0, 4, 8, 12], dtype=torch.int32, device=current.device)
    magnitudes = ((q_bytes >> 4) & 0x7) if store_hi else (q_bytes & 0x7)
    selectors = (magnitudes.to(torch.int32) << shifts).sum(dim=-1).to(torch.int32)
    selector_pairs = selectors.view(128, 8, 2)
    packed_selectors = (
        selector_pairs[:, :, 0] | (selector_pairs[:, :, 1] << 16)
    ).to(torch.int32)

    rows = torch.empty((128, 112), dtype=torch.uint8, device=current.device)
    rows[:, :64] = current[:, :64]
    rows[:, 64:96] = packed_selectors.contiguous().view(torch.uint8).view(128, 32)
    rows[:, 96:104] = current[:, 64:72]
    rows[:, 104:112] = current[:, 64:72]
    return rows.contiguous()


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

    current_rows = make_current_rows(args.scale_pattern)
    current = current_rows.view(-1)
    compact_low = make_compact_rows(current_rows, store_hi=False).view(-1)
    compact_high = make_compact_rows(current_rows, store_hi=True).view(-1)
    ext = load_extension()
    reference = None
    samples = {variant: [] for variant in args.variants}
    for round_idx in range(args.rounds):
        order = args.variants if round_idx % 2 == 0 else list(reversed(args.variants))
        for variant in order:
            if variant in (3, 4, 7, 9):
                input_rows = compact_low
            elif variant >= 5:
                input_rows = compact_high
            else:
                input_rows = current
            cycles, witnesses = ext.run_dequant_compact_bench(input_rows, variant, args.blocks)
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
            f"{variant:2d} {VARIANTS[variant]:31s} raw={center:8.1f} "
            f"net={center - empty:8.1f} rounds={[round(value, 1) for value in samples[variant]]}"
        )


if __name__ == "__main__":
    main()
