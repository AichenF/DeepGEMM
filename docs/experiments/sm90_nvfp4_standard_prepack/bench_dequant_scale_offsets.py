"""Screen zero-growth 16-bit LUT offsets for the braided NVFP4 decoder.

The retained deployment row stores 64 braided E2M1 bytes, eight UE4M3 scale
bytes, and eight padding bytes.  This isolated candidate uses the same 80
bytes but losslessly represents each scale as ``uint16(scale_code * 8)``.
Those values are byte offsets into the shared ``uint2`` LUT.  No FP8 weight
value is materialized and there is no production/runtime wiring here.
"""

import argparse
import os
import statistics
import sys

import torch
from torch.utils.cpp_extension import load_inline


REPO_ROOT = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
)
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)


CUDA_SRC = r"""
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <c10/cuda/CUDAException.h>
#include <deep_gemm/quantization/nvfp4_dequant.cuh>

namespace {

constexpr int kRows = 128;
constexpr int kPackedRowBytes = 80;
constexpr int kFp8RowBytes = 128;

enum ScaleMode : int {
    kScaleBytes = 0,
    kLutOffsets16 = 1,
};

__device__ __forceinline__ uint2 decode_braided_word(
        const uint32_t braided, const uint2 lut) {
    const uint32_t sel0 = braided & 0x00007777u;
    const uint32_t sel1 = (braided >> 16) & 0x00007777u;
    uint32_t out0 = deep_gemm::nvfp4::byte_perm_unchecked(lut.x, lut.y, sel0);
    uint32_t out1 = deep_gemm::nvfp4::byte_perm_unchecked(lut.x, lut.y, sel1);
    out0 |= braided & 0x80808080u;
    out1 |= (braided << 4) & 0x80808080u;
    return make_uint2(out0, out1);
}

__device__ __forceinline__ void decode_braided_quad(
        uint8_t* __restrict__ fp8_dst, const uint4 q,
        const uint2 lut0, const uint2 lut1, const int scale_i0,
        const uint32_t row_swizzle) {
    const uint2 q0 = decode_braided_word(q.x, lut0);
    const uint2 q1 = decode_braided_word(q.y, lut0);
    *reinterpret_cast<uint4*>(fp8_dst + ((scale_i0 * 16) ^ row_swizzle)) =
        make_uint4(q0.x, q0.y, q1.x, q1.y);

    const uint2 q2 = decode_braided_word(q.z, lut1);
    const uint2 q3 = decode_braided_word(q.w, lut1);
    *reinterpret_cast<uint4*>(fp8_dst + (((scale_i0 + 1) * 16) ^ row_swizzle)) =
        make_uint4(q2.x, q2.y, q3.x, q3.y);
}

template <int kMode, int kQuad>
__device__ __forceinline__ uint2 get_lut_keys(const uint4 metadata) {
    if constexpr (kMode == kScaleBytes) {
        const uint32_t scale_word = kQuad < 2 ? metadata.x : metadata.y;
        constexpr int kScaleI0 = kQuad * 2;
        constexpr int kShift0 = (kScaleI0 & 3) * 8;
        constexpr int kShift1 = ((kScaleI0 + 1) & 3) * 8;
        return make_uint2(
            (scale_word >> kShift0) & 0x7fu,
            (scale_word >> kShift1) & 0x7fu);
    } else {
        const uint32_t offset_word =
            kQuad == 0 ? metadata.x :
            kQuad == 1 ? metadata.y :
            kQuad == 2 ? metadata.z : metadata.w;
        return make_uint2(offset_word & 0xffffu, offset_word >> 16);
    }
}

template <int kMode>
__device__ __forceinline__ uint2 load_lut(
        const uint2* __restrict__ lut, const uint32_t key) {
    if constexpr (kMode == kScaleBytes) {
        return lut[key];
    } else {
        const auto* address = reinterpret_cast<const uint8_t*>(lut) + key;
        return *reinterpret_cast<const uint2*>(address);
    }
}

template <int kMode, int kQuad>
__device__ __forceinline__ void decode_lut_window(
        uint8_t* __restrict__ fp8_dst, const uint4 (&fp4_quads)[4],
        const uint4 metadata, const uint2* __restrict__ lut,
        const uint2 lut0, const uint2 lut1, const uint32_t row_swizzle) {
    uint2 next_lut0;
    uint2 next_lut1;
    if constexpr (kQuad + 1 < 4) {
        const uint2 keys = get_lut_keys<kMode, kQuad + 1>(metadata);
        next_lut0 = load_lut<kMode>(lut, keys.x);
        next_lut1 = load_lut<kMode>(lut, keys.y);
    }

    decode_braided_quad(
        fp8_dst, fp4_quads[kQuad], lut0, lut1, kQuad * 2, row_swizzle);

    if constexpr (kQuad + 1 < 4) {
        decode_lut_window<kMode, kQuad + 1>(
            fp8_dst, fp4_quads, metadata, lut,
            next_lut0, next_lut1, row_swizzle);
    }
}

template <int kMode>
__device__ __forceinline__ void decode_row(
        uint8_t* __restrict__ fp8, const uint8_t* __restrict__ packed,
        const uint32_t row, const uint2* __restrict__ lut) {
    const uint8_t* __restrict__ row_ptr = packed + row * kPackedRowBytes;
    const uint4* __restrict__ fp4_src = reinterpret_cast<const uint4*>(row_ptr);
    uint4 fp4_quads[4];
#pragma unroll
    for (int i = 0; i < 4; ++i)
        fp4_quads[i] = fp4_src[i];

    uint4 metadata;
    if constexpr (kMode == kScaleBytes) {
        const uint2 scale_words = *reinterpret_cast<const uint2*>(row_ptr + 64);
        metadata = make_uint4(scale_words.x, scale_words.y, 0u, 0u);
    } else {
        metadata = *reinterpret_cast<const uint4*>(row_ptr + 64);
    }

    const uint2 keys = get_lut_keys<kMode, 0>(metadata);
    const uint2 lut0 = load_lut<kMode>(lut, keys.x);
    const uint2 lut1 = load_lut<kMode>(lut, keys.y);
    decode_lut_window<kMode, 0>(
        fp8 + row * kFp8RowBytes, fp4_quads, metadata, lut,
        lut0, lut1, (row & 7u) << 4);
}

template <bool kDecode, int kMode>
__global__ __launch_bounds__(128) void bench_kernel(
        const uint8_t* __restrict__ input, int64_t* __restrict__ cycles,
        uint8_t* __restrict__ output) {
    extern __shared__ __align__(16) uint8_t smem[];
    uint8_t* packed = smem;
    uint8_t* fp8 = packed + kRows * kPackedRowBytes;
    auto* lut = reinterpret_cast<uint2*>(fp8 + kRows * kFp8RowBytes);
    const uint32_t tid = threadIdx.x;

    for (int i = tid; i < kRows * kPackedRowBytes; i += blockDim.x)
        packed[i] = input[i];
    for (int i = tid; i < kRows * kFp8RowBytes; i += blockDim.x)
        fp8[i] = 0;
    lut[tid] = deep_gemm::nvfp4::kE2M1AndUe4m3ToFp8Lut[tid];
    __syncthreads();

    uint64_t start = 0;
    if (tid == 0)
        start = clock64();
    __syncthreads();
    if constexpr (kDecode)
        decode_row<kMode>(fp8, packed, tid, lut);
    __syncthreads();
    if (tid == 0)
        cycles[blockIdx.x] = static_cast<int64_t>(clock64() - start);

    if (blockIdx.x == 0) {
        for (int i = tid; i < kRows * kFp8RowBytes; i += blockDim.x)
            output[i] = fp8[i];
    }
}

template <bool kDecode, int kMode>
void launch(const torch::Tensor& input, torch::Tensor& cycles, torch::Tensor& output) {
    constexpr int kSharedBytes =
        kRows * (kPackedRowBytes + kFp8RowBytes) + 128 * sizeof(uint2);
    bench_kernel<kDecode, kMode><<<cycles.numel(), 128, kSharedBytes>>>(
        input.data_ptr<uint8_t>(), cycles.data_ptr<int64_t>(),
        output.data_ptr<uint8_t>());
}

}  // namespace

std::vector<torch::Tensor> run_dequant_scale_offset_bench(
        torch::Tensor input, int64_t variant, int64_t blocks) {
    TORCH_CHECK(input.is_cuda() && input.scalar_type() == torch::kUInt8);
    TORCH_CHECK(input.numel() == kRows * kPackedRowBytes);
    auto cycles = torch::empty({blocks}, input.options().dtype(torch::kInt64));
    auto output = torch::empty({kRows, kFp8RowBytes}, input.options());
    switch (variant) {
        case 0: launch<false, kScaleBytes>(input, cycles, output); break;
        case 1: launch<true,  kScaleBytes>(input, cycles, output); break;
        case 2: launch<true,  kLutOffsets16>(input, cycles, output); break;
        default: TORCH_CHECK(false, "unknown variant");
    }
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return {cycles, output};
}
"""


VARIANTS = {
    0: "empty",
    1: "braided/scale-bytes",
    2: "braided/lut-offsets16",
}


def load_extension():
    cpp_src = (
        "std::vector<torch::Tensor> run_dequant_scale_offset_bench("
        "torch::Tensor, int64_t, int64_t);"
    )
    return load_inline(
        name="deepgemm_sm90_nvfp4_dequant_scale_offsets_v1",
        cpp_sources=cpp_src,
        cuda_sources=CUDA_SRC,
        functions=["run_dequant_scale_offset_bench"],
        extra_include_paths=[os.path.join(REPO_ROOT, "deep_gemm", "include")],
        extra_cuda_cflags=["-O3", "-lineinfo", "--expt-relaxed-constexpr"],
        verbose=False,
    )


def braid_rows(rows: torch.Tensor) -> torch.Tensor:
    result = rows.clone()
    packed = result[:, :64].view(128, 16, 4)
    codes = torch.cat(((packed >> 4) & 0x0F, packed & 0x0F), dim=-1)
    magnitudes = codes & 0x07
    signs = codes >> 3
    braided_signs = torch.stack(
        (
            signs[..., 4],
            signs[..., 0],
            signs[..., 5],
            signs[..., 1],
            signs[..., 6],
            signs[..., 2],
            signs[..., 7],
            signs[..., 3],
        ),
        dim=-1,
    )
    braided_nibbles = magnitudes | (braided_signs << 3)
    braided_bytes = braided_nibbles[..., 0::2] | (braided_nibbles[..., 1::2] << 4)
    result[:, :64] = braided_bytes.reshape(128, 64)
    return result


def make_rows(scale_pattern: str) -> tuple[torch.Tensor, torch.Tensor]:
    generator = torch.Generator(device="cpu").manual_seed(1234)
    codes = torch.randint(
        0, 16, (128, 128), dtype=torch.uint8, generator=generator
    )
    if scale_pattern == "exhaustive":
        codes = torch.arange(16, dtype=torch.uint8).repeat(128, 8)
    packed = codes[:, 0::2] | (codes[:, 1::2] << 4)
    rows = torch.zeros((128, 80), dtype=torch.uint8)
    rows[:, :64] = packed

    if scale_pattern == "model":
        torch.manual_seed(1234)
        from deep_gemm.quantization_nvfp4 import fp32_to_ue4m3_ceil

        weights = (
            torch.randn((128, 8, 16), dtype=torch.float32, device="cuda")
            * 0.05
        )
        scales = fp32_to_ue4m3_ceil(weights.abs().amax(dim=-1) / 6.0).cpu()
    elif scale_pattern == "random":
        scales = torch.randint(
            0, 127, (128, 8), dtype=torch.uint8, generator=generator
        )
    elif scale_pattern == "exhaustive":
        scales = (torch.arange(128 * 8, dtype=torch.int64) % 127).to(
            torch.uint8
        ).view(128, 8)
    else:
        raise ValueError(scale_pattern)
    rows[:, 64:72] = scales

    braided = braid_rows(rows)
    offsets = braided.clone()
    offset_words = scales.to(torch.int32).mul(8).to(torch.int16).contiguous()
    offsets[:, 64:80] = offset_words.view(torch.uint8).view(128, 16)
    return braided.cuda().contiguous().view(-1), offsets.cuda().contiguous().view(-1)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--blocks", type=int, default=624)
    parser.add_argument("--rounds", type=int, default=15)
    parser.add_argument(
        "--scale-pattern",
        choices=("model", "random", "exhaustive"),
        default="model",
    )
    parser.add_argument("--variants", type=int, nargs="+", default=list(VARIANTS))
    args = parser.parse_args()

    assert torch.cuda.get_device_capability()[0] == 9
    if any(variant not in VARIANTS for variant in args.variants):
        raise ValueError(f"variants must be in {list(VARIANTS)}")

    scale_bytes, lut_offsets = make_rows(args.scale_pattern)
    ext = load_extension()
    reference = None
    samples = {variant: [] for variant in args.variants}
    for round_idx in range(args.rounds):
        order = args.variants if round_idx % 2 == 0 else list(reversed(args.variants))
        for variant in order:
            packed = lut_offsets if variant == 2 else scale_bytes
            cycles, output = ext.run_dequant_scale_offset_bench(
                packed, variant, args.blocks
            )
            torch.cuda.synchronize()
            if variant != 0:
                candidate = output.cpu()
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
            f"{variant:2d} {VARIANTS[variant]:28s} raw={center:8.1f} "
            f"net={center - empty:8.1f} rounds={[round(value, 1) for value in samples[variant]]}"
        )


if __name__ == "__main__":
    main()
