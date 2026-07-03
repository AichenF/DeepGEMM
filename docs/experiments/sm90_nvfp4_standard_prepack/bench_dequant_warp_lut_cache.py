"""Screen warp-distributed LUT caches for the braided SM90 NVFP4 decoder.

This is an isolated microbenchmark with no runtime or production wiring.  It
compares the retained shared-LUT next-quad window against 16-entry and 32-entry
warp caches.  Out-of-window UE4M3 codes always fall back to the shared LUT.
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
constexpr int kHot16Base = 3;

enum LutMode : int {
    kShared = 0,
    kWarp16 = 1,
    kWarp32 = 2,
};

struct WarpLutCache {
    uint32_t x;
    uint32_t y;
};

template <int kMode>
__device__ __forceinline__ WarpLutCache make_warp_lut_cache(
        const uint2* __restrict__ lut, const uint32_t lane) {
    WarpLutCache cache{0u, 0u};
    if constexpr (kMode == kWarp16) {
        const auto* words = reinterpret_cast<const uint32_t*>(lut);
        const uint32_t scale = kHot16Base + (lane & 15u);
        cache.x = words[scale * 2u + (lane >> 4)];
    } else if constexpr (kMode == kWarp32) {
        const uint2 entry = lut[lane];
        cache.x = entry.x;
        cache.y = entry.y;
    }
    return cache;
}

template <int kMode>
__device__ __forceinline__ uint2 load_lut(
        const uint2* __restrict__ lut, const uint32_t scale,
        const WarpLutCache cache) {
    const uint32_t code = scale & 0x7fu;
    if constexpr (kMode == kWarp16) {
        const uint32_t hot_idx = code - kHot16Base;
        const uint32_t src_lane = hot_idx & 15u;
        // All lanes execute both shuffles, so fixed source lanes remain active
        // even when some requesting lanes take the shared-LUT fallback.
        const uint32_t x = __shfl_sync(0xffffffffu, cache.x, src_lane);
        const uint32_t y = __shfl_sync(0xffffffffu, cache.x, src_lane + 16u);
        if (hot_idx < 16u)
            return make_uint2(x, y);
    } else if constexpr (kMode == kWarp32) {
        const uint32_t src_lane = code & 31u;
        const uint32_t x = __shfl_sync(0xffffffffu, cache.x, src_lane);
        const uint32_t y = __shfl_sync(0xffffffffu, cache.y, src_lane);
        if (code < 32u)
            return make_uint2(x, y);
    }
    return lut[code];
}

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
__device__ __forceinline__ void decode_lut_window(
        uint8_t* __restrict__ fp8_dst, const uint4 (&fp4_quads)[4],
        const uint32_t scale_word_lo, const uint32_t scale_word_hi,
        const uint2* __restrict__ lut, const WarpLutCache cache,
        const uint2 lut0, const uint2 lut1, const uint32_t row_swizzle) {
    uint2 next_lut0;
    uint2 next_lut1;
    if constexpr (kQuad + 1 < 4) {
        constexpr int kNextScaleI0 = (kQuad + 1) * 2;
        constexpr int kNextScaleI1 = kNextScaleI0 + 1;
        const uint32_t word = kQuad + 1 < 2 ? scale_word_lo : scale_word_hi;
        const uint32_t scale0 = (word >> ((kNextScaleI0 & 3) * 8)) & 0x7fu;
        const uint32_t scale1 = (word >> ((kNextScaleI1 & 3) * 8)) & 0x7fu;
        next_lut0 = load_lut<kMode>(lut, scale0, cache);
        next_lut1 = load_lut<kMode>(lut, scale1, cache);
    }

    decode_braided_quad(
        fp8_dst, fp4_quads[kQuad], lut0, lut1, kQuad * 2, row_swizzle);

    if constexpr (kQuad + 1 < 4) {
        decode_lut_window<kMode, kQuad + 1>(
            fp8_dst, fp4_quads, scale_word_lo, scale_word_hi, lut, cache,
            next_lut0, next_lut1, row_swizzle);
    }
}

template <int kMode>
__device__ __forceinline__ void decode_row(
        uint8_t* __restrict__ fp8, const uint8_t* __restrict__ packed,
        const uint32_t row, const uint2* __restrict__ lut,
        const WarpLutCache cache) {
    const uint8_t* __restrict__ row_ptr = packed + row * kPackedRowBytes;
    const uint4* __restrict__ fp4_src = reinterpret_cast<const uint4*>(row_ptr);
    uint4 fp4_quads[4];
#pragma unroll
    for (int i = 0; i < 4; ++i)
        fp4_quads[i] = fp4_src[i];

    const uint2 scale_words = *reinterpret_cast<const uint2*>(row_ptr + 64);
    const uint2 lut0 = load_lut<kMode>(lut, scale_words.x & 0x7fu, cache);
    const uint2 lut1 = load_lut<kMode>(lut, (scale_words.x >> 8) & 0x7fu, cache);
    decode_lut_window<kMode, 0>(
        fp8 + row * kFp8RowBytes, fp4_quads, scale_words.x, scale_words.y,
        lut, cache, lut0, lut1, (row & 7u) << 4);
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
    const uint32_t lane = tid & 31u;

    for (int i = tid; i < kRows * kPackedRowBytes; i += blockDim.x)
        packed[i] = input[i];
    for (int i = tid; i < kRows * kFp8RowBytes; i += blockDim.x)
        fp8[i] = 0;
    lut[tid] = deep_gemm::nvfp4::kE2M1AndUe4m3ToFp8Lut[tid];
    __syncthreads();

    WarpLutCache cache = make_warp_lut_cache<kMode>(lut, lane);
    asm volatile("" : "+r"(cache.x), "+r"(cache.y) :: "memory");
    __syncthreads();

    uint64_t start = 0;
    if (tid == 0)
        start = clock64();
    __syncthreads();
    if constexpr (kDecode)
        decode_row<kMode>(fp8, packed, tid, lut, cache);
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

std::vector<torch::Tensor> run_dequant_warp_lut_cache_bench(
        torch::Tensor input, int64_t variant, int64_t blocks) {
    TORCH_CHECK(input.is_cuda() && input.scalar_type() == torch::kUInt8);
    TORCH_CHECK(input.numel() == kRows * kPackedRowBytes);
    auto cycles = torch::empty({blocks}, input.options().dtype(torch::kInt64));
    auto output = torch::empty({kRows, kFp8RowBytes}, input.options());
    switch (variant) {
        case 0: launch<false, kShared>(input, cycles, output); break;
        case 1: launch<true,  kShared>(input, cycles, output); break;
        case 2: launch<true,  kWarp16>(input, cycles, output); break;
        case 3: launch<true,  kWarp32>(input, cycles, output); break;
        default: TORCH_CHECK(false, "unknown variant");
    }
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return {cycles, output};
}
"""


VARIANTS = {
    0: "empty",
    1: "shared-lut-window",
    2: "warp16-lut-window",
    3: "warp32-lut-window",
}


def load_extension():
    cpp_src = (
        "std::vector<torch::Tensor> run_dequant_warp_lut_cache_bench("
        "torch::Tensor, int64_t, int64_t);"
    )
    return load_inline(
        name="deepgemm_sm90_nvfp4_dequant_warp_lut_cache_v1",
        cpp_sources=cpp_src,
        cuda_sources=CUDA_SRC,
        functions=["run_dequant_warp_lut_cache_bench"],
        extra_include_paths=[os.path.join(REPO_ROOT, "deep_gemm", "include")],
        extra_cuda_cflags=["-O3", "-lineinfo", "--expt-relaxed-constexpr"],
        verbose=False,
    )


def make_rows(scale_pattern: str) -> torch.Tensor:
    num_rows = 128
    generator = torch.Generator(device="cpu").manual_seed(1234)
    codes = torch.randint(
        0, 16, (num_rows, 128), dtype=torch.uint8, generator=generator
    )
    if scale_pattern == "exhaustive":
        codes = torch.arange(16, dtype=torch.uint8).repeat(num_rows, 8)
    packed = codes[:, 0::2] | (codes[:, 1::2] << 4)
    rows = torch.zeros((num_rows, 80), dtype=torch.uint8)
    rows[:, :64] = packed

    if scale_pattern == "model":
        torch.manual_seed(1234)
        from deep_gemm.quantization_nvfp4 import fp32_to_ue4m3_ceil

        weights = (
            torch.randn((num_rows, 8, 16), dtype=torch.float32, device="cuda")
            * 0.05
        )
        scales = fp32_to_ue4m3_ceil(weights.abs().amax(dim=-1) / 6.0).cpu()
    elif scale_pattern == "hot16":
        scales = torch.randint(
            3, 19, (num_rows, 8), dtype=torch.uint8, generator=generator
        )
    elif scale_pattern == "random":
        scales = torch.randint(
            0, 127, (num_rows, 8), dtype=torch.uint8, generator=generator
        )
    elif scale_pattern == "exhaustive":
        scales = (torch.arange(num_rows * 8, dtype=torch.int64) % 127).to(
            torch.uint8
        ).view(num_rows, 8)
    else:
        raise ValueError(scale_pattern)
    rows[:, 64:72] = scales
    return braid_rows(rows).cuda().contiguous().view(-1)


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


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--blocks", type=int, default=624)
    parser.add_argument("--rounds", type=int, default=15)
    parser.add_argument(
        "--scale-pattern",
        choices=("model", "hot16", "random", "exhaustive"),
        default="model",
    )
    parser.add_argument("--variants", type=int, nargs="+", default=list(VARIANTS))
    args = parser.parse_args()

    assert torch.cuda.get_device_capability()[0] == 9
    if any(variant not in VARIANTS for variant in args.variants):
        raise ValueError(f"variants must be in {list(VARIANTS)}")

    packed = make_rows(args.scale_pattern)
    ext = load_extension()
    reference = None
    samples = {variant: [] for variant in args.variants}
    for round_idx in range(args.rounds):
        order = args.variants if round_idx % 2 == 0 else list(reversed(args.variants))
        for variant in order:
            cycles, output = ext.run_dequant_warp_lut_cache_bench(
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
            f"{variant:2d} {VARIANTS[variant]:24s} raw={center:8.1f} "
            f"net={center - empty:8.1f} "
            f"rounds={[round(value, 1) for value in samples[variant]]}"
        )


if __name__ == "__main__":
    main()
