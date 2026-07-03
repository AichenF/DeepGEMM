"""Isolated SM90 NVFP4 benchmark for deployment-time selector metadata.

The candidate row is 144 bytes: 128 bytes of integer selector/sign metadata,
eight standard UE4M3 scale bytes, and eight alignment bytes.  It contains no
FP8 weights and has no production wiring.
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
constexpr int kMetadataRowBytes = 144;
constexpr int kFp8RowBytes = 128;

template <bool kUseDp4aHi, bool kUseDp4aLo>
__device__ __forceinline__ uint2 decode_current_word(uint32_t q, const uint2& lut) {
    return deep_gemm::nvfp4::dequant_nvfp4_to_fp8_pair_with_lut<
        kUseDp4aHi, kUseDp4aLo>(q, lut);
}

__device__ __forceinline__ uint2 decode_metadata_word(
        uint32_t selectors, uint32_t signs, const uint2& lut) {
    uint32_t out_hi = deep_gemm::nvfp4::byte_perm_unchecked(lut.x, lut.y, selectors);
    uint32_t out_lo = deep_gemm::nvfp4::byte_perm_unchecked(lut.x, lut.y, selectors >> 16);
    out_hi |= signs & 0x80808080u;
    out_lo |= (signs << 4) & 0x80808080u;
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

__device__ __forceinline__ void decode_metadata_group(
        uint8_t* __restrict__ fp8_dst,
        uint32_t physical,
        const uint4& metadata,
        const uint2& lut) {
    const uint2 q0 = decode_metadata_word(metadata.x, metadata.y, lut);
    const uint2 q1 = decode_metadata_word(metadata.z, metadata.w, lut);
    *reinterpret_cast<uint4*>(fp8_dst + physical) =
        make_uint4(q0.x, q0.y, q1.x, q1.y);
}

template <bool kPaired>
__device__ __forceinline__ void decode_metadata_row(
        uint8_t* __restrict__ fp8,
        const uint8_t* __restrict__ packed,
        uint32_t row,
        const uint2* __restrict__ lut_smem) {
    const uint8_t* __restrict__ row_ptr = packed + row * kMetadataRowBytes;
    const uint2 scale_words = *reinterpret_cast<const uint2*>(row_ptr + 128);
    const uint32_t scale_word_lo = scale_words.x;
    const uint32_t scale_word_hi = scale_words.y;
    uint8_t* __restrict__ fp8_dst = fp8 + row * kFp8RowBytes;
    const uint32_t row_swizzle = (row & 7u) << 4;

    if constexpr (kPaired) {
#pragma unroll
        for (int pair_i = 0; pair_i < 4; ++pair_i) {
            const int scale_i0 = pair_i * 2;
            const int scale_i1 = scale_i0 + 1;
            const uint32_t scale_word = pair_i < 2 ? scale_word_lo : scale_word_hi;
            const uint32_t scale0 = (scale_word >> ((scale_i0 & 3) * 8)) & 0xffu;
            const uint32_t scale1 = (scale_word >> ((scale_i1 & 3) * 8)) & 0xffu;
            const uint4 metadata0 =
                *reinterpret_cast<const uint4*>(row_ptr + scale_i0 * 16);
            const uint4 metadata1 =
                *reinterpret_cast<const uint4*>(row_ptr + scale_i1 * 16);
            const uint2 lut0 = lut_smem[scale0 & 0x7fu];
            const uint2 lut1 = lut_smem[scale1 & 0x7fu];
            decode_metadata_group(
                fp8_dst, (scale_i0 * 16) ^ row_swizzle, metadata0, lut0);
            decode_metadata_group(
                fp8_dst, (scale_i1 * 16) ^ row_swizzle, metadata1, lut1);
        }
    } else {
#pragma unroll
        for (int scale_i = 0; scale_i < 8; ++scale_i) {
            const uint32_t scale_word = scale_i < 4 ? scale_word_lo : scale_word_hi;
            const uint32_t scale = (scale_word >> ((scale_i & 3) * 8)) & 0xffu;
            const uint4 metadata =
                *reinterpret_cast<const uint4*>(row_ptr + scale_i * 16);
            const uint2 lut = lut_smem[scale & 0x7fu];
            decode_metadata_group(
                fp8_dst, (scale_i * 16) ^ row_swizzle, metadata, lut);
        }
    }
}

template <bool kDecode, bool kMetadata, bool kPaired,
          bool kUseDp4aHi, bool kUseDp4aLo>
__global__ __launch_bounds__(128) void bench_kernel(
        const uint8_t* __restrict__ input,
        int64_t* __restrict__ cycles,
        int32_t* __restrict__ witnesses) {
    constexpr int kRowBytes = kMetadata ? kMetadataRowBytes : kCurrentRowBytes;
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
        if constexpr (kMetadata) {
            decode_metadata_row<kPaired>(fp8, packed, tid, lut);
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

template <bool kDecode, bool kMetadata, bool kPaired,
          bool kUseDp4aHi, bool kUseDp4aLo>
void launch(const torch::Tensor& input, torch::Tensor& cycles, torch::Tensor& witnesses) {
    constexpr int kRowBytes = kMetadata ? kMetadataRowBytes : kCurrentRowBytes;
    constexpr int kSharedBytes =
        kRows * (kRowBytes + kFp8RowBytes) + 128 * static_cast<int>(sizeof(uint2));
    const int blocks = static_cast<int>(cycles.numel());
    bench_kernel<kDecode, kMetadata, kPaired, kUseDp4aHi, kUseDp4aLo>
        <<<blocks, 128, kSharedBytes>>>(
            input.data_ptr<uint8_t>(), cycles.data_ptr<int64_t>(),
            witnesses.data_ptr<int32_t>());
}

}  // namespace

std::vector<torch::Tensor> run_dequant_selector_bench(
        torch::Tensor input, int64_t variant, int64_t blocks) {
    TORCH_CHECK(input.is_cuda() && input.scalar_type() == torch::kUInt8);
    auto cycles = torch::empty({blocks}, input.options().dtype(torch::kInt64));
    auto witnesses = torch::empty({blocks, kRows}, input.options().dtype(torch::kInt32));
    switch (variant) {
        case 0: launch<false, false, false, false, true >(input, cycles, witnesses); break;
        case 1: launch<true,  false, false, false, true >(input, cycles, witnesses); break;
        case 2: launch<true,  false, true,  true,  true >(input, cycles, witnesses); break;
        case 3: launch<true,  true,  false, false, false>(input, cycles, witnesses); break;
        case 4: launch<true,  true,  true,  false, false>(input, cycles, witnesses); break;
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
    3: "metadata/sequential",
    4: "metadata/paired",
}


def load_extension():
    cpp_src = "std::vector<torch::Tensor> run_dequant_selector_bench(torch::Tensor, int64_t, int64_t);"
    return load_inline(
        name="deepgemm_sm90_nvfp4_dequant_selector_metadata",
        cpp_sources=cpp_src,
        cuda_sources=CUDA_SRC,
        functions=["run_dequant_selector_bench"],
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


def make_metadata_rows(current: torch.Tensor) -> torch.Tensor:
    q_bytes = current[:, :64].contiguous().view(128, 16, 4)
    shifts = torch.tensor([0, 4, 8, 12], dtype=torch.int32, device=current.device)
    hi = ((q_bytes >> 4) & 0x7).to(torch.int32)
    lo = (q_bytes & 0x7).to(torch.int32)
    selectors_hi = (hi << shifts).sum(dim=-1)
    selectors_lo = (lo << shifts).sum(dim=-1)
    selectors = selectors_hi | (selectors_lo << 16)
    sign_words = current[:, :64].contiguous().view(torch.int32).view(128, 16)

    metadata_words = torch.empty((128, 16, 2), dtype=torch.int32, device=current.device)
    metadata_words[:, :, 0] = selectors
    metadata_words[:, :, 1] = sign_words
    rows = torch.empty((128, 144), dtype=torch.uint8, device=current.device)
    rows[:, :128] = metadata_words.view(torch.uint8).view(128, 128)
    rows[:, 128:136] = current[:, 64:72]
    rows[:, 136:144] = current[:, 64:72]
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

    current = make_current_rows(args.scale_pattern).view(-1)
    metadata = make_metadata_rows(current.view(128, 80)).view(-1)
    ext = load_extension()
    reference = None
    samples = {variant: [] for variant in args.variants}
    for round_idx in range(args.rounds):
        order = args.variants if round_idx % 2 == 0 else list(reversed(args.variants))
        for variant in order:
            input_rows = metadata if variant >= 3 else current
            cycles, witnesses = ext.run_dequant_selector_bench(input_rows, variant, args.blocks)
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
            f"{variant:2d} {VARIANTS[variant]:29s} raw={center:8.1f} "
            f"net={center - empty:8.1f} rounds={[round(value, 1) for value in samples[variant]]}"
        )


if __name__ == "__main__":
    main()
