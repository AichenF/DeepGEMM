"""Screen instruction ordering variants of the braided decoder.

This isolated harness keeps the retained 80-byte deployment layout and shared
LUT unchanged. It changes how independent PRMT/sign chains are exposed to
ptxas and whether the next LUT pair is loaded before or after the current
stores. There is no production wiring.
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

enum ScheduleMode : int {
    kWordSerial = 0,
    kPairIlp = 1,
    kQuadIlp = 2,
    kWordStoreFirst = 3,
};

__device__ __forceinline__ uint2 decode_word_serial(
        const uint32_t q, const uint2 lut) {
    const uint32_t sel0 = q & 0x00007777u;
    const uint32_t sel1 = (q >> 16) & 0x00007777u;
    uint32_t out0 = deep_gemm::nvfp4::byte_perm_unchecked(lut.x, lut.y, sel0);
    uint32_t out1 = deep_gemm::nvfp4::byte_perm_unchecked(lut.x, lut.y, sel1);
    out0 |= q & 0x80808080u;
    out1 |= (q << 4) & 0x80808080u;
    return make_uint2(out0, out1);
}

__device__ __forceinline__ uint4 decode_pair_ilp(
        const uint32_t q0, const uint32_t q1, const uint2 lut) {
    const uint32_t q0_sel0 = q0 & 0x00007777u;
    const uint32_t q0_sel1 = (q0 >> 16) & 0x00007777u;
    const uint32_t q1_sel0 = q1 & 0x00007777u;
    const uint32_t q1_sel1 = (q1 >> 16) & 0x00007777u;
    const uint32_t q0_sign1 = q0 << 4;
    const uint32_t q1_sign1 = q1 << 4;

    uint32_t q0_out0 = deep_gemm::nvfp4::byte_perm_unchecked(
        lut.x, lut.y, q0_sel0);
    uint32_t q0_out1 = deep_gemm::nvfp4::byte_perm_unchecked(
        lut.x, lut.y, q0_sel1);
    uint32_t q1_out0 = deep_gemm::nvfp4::byte_perm_unchecked(
        lut.x, lut.y, q1_sel0);
    uint32_t q1_out1 = deep_gemm::nvfp4::byte_perm_unchecked(
        lut.x, lut.y, q1_sel1);

    q0_out0 |= q0 & 0x80808080u;
    q0_out1 |= q0_sign1 & 0x80808080u;
    q1_out0 |= q1 & 0x80808080u;
    q1_out1 |= q1_sign1 & 0x80808080u;
    return make_uint4(q0_out0, q0_out1, q1_out0, q1_out1);
}

__device__ __forceinline__ void decode_quad_ilp(
        uint8_t* __restrict__ dst0, uint8_t* __restrict__ dst1,
        const uint4 q, const uint2 lut0, const uint2 lut1) {
    const uint32_t q0_sel0 = q.x & 0x00007777u;
    const uint32_t q0_sel1 = (q.x >> 16) & 0x00007777u;
    const uint32_t q1_sel0 = q.y & 0x00007777u;
    const uint32_t q1_sel1 = (q.y >> 16) & 0x00007777u;
    const uint32_t q2_sel0 = q.z & 0x00007777u;
    const uint32_t q2_sel1 = (q.z >> 16) & 0x00007777u;
    const uint32_t q3_sel0 = q.w & 0x00007777u;
    const uint32_t q3_sel1 = (q.w >> 16) & 0x00007777u;

    uint32_t q0_out0 = deep_gemm::nvfp4::byte_perm_unchecked(
        lut0.x, lut0.y, q0_sel0);
    uint32_t q0_out1 = deep_gemm::nvfp4::byte_perm_unchecked(
        lut0.x, lut0.y, q0_sel1);
    uint32_t q1_out0 = deep_gemm::nvfp4::byte_perm_unchecked(
        lut0.x, lut0.y, q1_sel0);
    uint32_t q1_out1 = deep_gemm::nvfp4::byte_perm_unchecked(
        lut0.x, lut0.y, q1_sel1);
    uint32_t q2_out0 = deep_gemm::nvfp4::byte_perm_unchecked(
        lut1.x, lut1.y, q2_sel0);
    uint32_t q2_out1 = deep_gemm::nvfp4::byte_perm_unchecked(
        lut1.x, lut1.y, q2_sel1);
    uint32_t q3_out0 = deep_gemm::nvfp4::byte_perm_unchecked(
        lut1.x, lut1.y, q3_sel0);
    uint32_t q3_out1 = deep_gemm::nvfp4::byte_perm_unchecked(
        lut1.x, lut1.y, q3_sel1);

    q0_out0 |= q.x & 0x80808080u;
    q0_out1 |= (q.x << 4) & 0x80808080u;
    q1_out0 |= q.y & 0x80808080u;
    q1_out1 |= (q.y << 4) & 0x80808080u;
    q2_out0 |= q.z & 0x80808080u;
    q2_out1 |= (q.z << 4) & 0x80808080u;
    q3_out0 |= q.w & 0x80808080u;
    q3_out1 |= (q.w << 4) & 0x80808080u;

    *reinterpret_cast<uint4*>(dst0) =
        make_uint4(q0_out0, q0_out1, q1_out0, q1_out1);
    *reinterpret_cast<uint4*>(dst1) =
        make_uint4(q2_out0, q2_out1, q3_out0, q3_out1);
}

template <int kSchedule>
__device__ __forceinline__ void decode_quad(
        uint8_t* __restrict__ fp8_dst, const uint4 q,
        const uint2 lut0, const uint2 lut1, const int scale_i0,
        const uint32_t row_swizzle) {
    uint8_t* dst0 = fp8_dst + ((scale_i0 * 16) ^ row_swizzle);
    uint8_t* dst1 = fp8_dst + (((scale_i0 + 1) * 16) ^ row_swizzle);
    if constexpr (kSchedule == kWordSerial || kSchedule == kWordStoreFirst) {
        const uint2 q0 = decode_word_serial(q.x, lut0);
        const uint2 q1 = decode_word_serial(q.y, lut0);
        *reinterpret_cast<uint4*>(dst0) = make_uint4(q0.x, q0.y, q1.x, q1.y);
        const uint2 q2 = decode_word_serial(q.z, lut1);
        const uint2 q3 = decode_word_serial(q.w, lut1);
        *reinterpret_cast<uint4*>(dst1) = make_uint4(q2.x, q2.y, q3.x, q3.y);
    } else if constexpr (kSchedule == kPairIlp) {
        const uint4 pair0 = decode_pair_ilp(q.x, q.y, lut0);
        const uint4 pair1 = decode_pair_ilp(q.z, q.w, lut1);
        *reinterpret_cast<uint4*>(dst0) = pair0;
        *reinterpret_cast<uint4*>(dst1) = pair1;
    } else {
        decode_quad_ilp(dst0, dst1, q, lut0, lut1);
    }
}

template <int kQuad>
__device__ __forceinline__ void decode_lut_window_store_first(
        uint8_t* __restrict__ fp8_dst, const uint4 (&fp4_quads)[4],
        const uint32_t scale_word_lo, const uint32_t scale_word_hi,
        const uint2* __restrict__ lut, const uint2 lut0, const uint2 lut1,
        const uint32_t row_swizzle) {
    decode_quad<kWordStoreFirst>(
        fp8_dst, fp4_quads[kQuad], lut0, lut1, kQuad * 2, row_swizzle);

    if constexpr (kQuad + 1 < 4) {
        constexpr int kNextScaleI0 = (kQuad + 1) * 2;
        constexpr int kNextScaleI1 = kNextScaleI0 + 1;
        const uint32_t word = kQuad + 1 < 2 ? scale_word_lo : scale_word_hi;
        const uint32_t scale0 =
            (word >> ((kNextScaleI0 & 3) * 8)) & 0x7fu;
        const uint32_t scale1 =
            (word >> ((kNextScaleI1 & 3) * 8)) & 0x7fu;
        const uint2 next_lut0 = lut[scale0];
        const uint2 next_lut1 = lut[scale1];
        decode_lut_window_store_first<kQuad + 1>(
            fp8_dst, fp4_quads, scale_word_lo, scale_word_hi, lut,
            next_lut0, next_lut1, row_swizzle);
    }
}

template <int kSchedule, int kQuad>
__device__ __forceinline__ void decode_lut_window(
        uint8_t* __restrict__ fp8_dst, const uint4 (&fp4_quads)[4],
        const uint32_t scale_word_lo, const uint32_t scale_word_hi,
        const uint2* __restrict__ lut, const uint2 lut0, const uint2 lut1,
        const uint32_t row_swizzle) {
    uint2 next_lut0;
    uint2 next_lut1;
    if constexpr (kQuad + 1 < 4) {
        constexpr int kNextScaleI0 = (kQuad + 1) * 2;
        constexpr int kNextScaleI1 = kNextScaleI0 + 1;
        const uint32_t word = kQuad + 1 < 2 ? scale_word_lo : scale_word_hi;
        const uint32_t scale0 =
            (word >> ((kNextScaleI0 & 3) * 8)) & 0x7fu;
        const uint32_t scale1 =
            (word >> ((kNextScaleI1 & 3) * 8)) & 0x7fu;
        next_lut0 = lut[scale0];
        next_lut1 = lut[scale1];
    }

    decode_quad<kSchedule>(
        fp8_dst, fp4_quads[kQuad], lut0, lut1, kQuad * 2, row_swizzle);

    if constexpr (kQuad + 1 < 4) {
        decode_lut_window<kSchedule, kQuad + 1>(
            fp8_dst, fp4_quads, scale_word_lo, scale_word_hi, lut,
            next_lut0, next_lut1, row_swizzle);
    }
}

template <int kSchedule>
__device__ __forceinline__ void decode_row(
        uint8_t* __restrict__ fp8, const uint8_t* __restrict__ packed,
        const uint32_t row, const uint2* __restrict__ lut) {
    const uint8_t* __restrict__ row_ptr = packed + row * kPackedRowBytes;
    const uint4* __restrict__ fp4_src = reinterpret_cast<const uint4*>(row_ptr);
    uint4 fp4_quads[4];
#pragma unroll
    for (int i = 0; i < 4; ++i)
        fp4_quads[i] = fp4_src[i];
    const uint2 scale_words = *reinterpret_cast<const uint2*>(row_ptr + 64);
    const uint2 lut0 = lut[scale_words.x & 0x7fu];
    const uint2 lut1 = lut[(scale_words.x >> 8) & 0x7fu];
    if constexpr (kSchedule == kWordStoreFirst) {
        decode_lut_window_store_first<0>(
            fp8 + row * kFp8RowBytes, fp4_quads, scale_words.x, scale_words.y,
            lut, lut0, lut1, (row & 7u) << 4);
    } else {
        decode_lut_window<kSchedule, 0>(
            fp8 + row * kFp8RowBytes, fp4_quads, scale_words.x, scale_words.y,
            lut, lut0, lut1, (row & 7u) << 4);
    }
}

template <bool kDecode, int kSchedule>
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
        decode_row<kSchedule>(fp8, packed, tid, lut);
    __syncthreads();
    if (tid == 0)
        cycles[blockIdx.x] = static_cast<int64_t>(clock64() - start);

    if (blockIdx.x == 0) {
        for (int i = tid; i < kRows * kFp8RowBytes; i += blockDim.x)
            output[i] = fp8[i];
    }
}

template <bool kDecode, int kSchedule>
void launch(const torch::Tensor& input, torch::Tensor& cycles, torch::Tensor& output) {
    constexpr int kSharedBytes =
        kRows * (kPackedRowBytes + kFp8RowBytes) + 128 * sizeof(uint2);
    bench_kernel<kDecode, kSchedule><<<cycles.numel(), 128, kSharedBytes>>>(
        input.data_ptr<uint8_t>(), cycles.data_ptr<int64_t>(),
        output.data_ptr<uint8_t>());
}

}  // namespace

std::vector<torch::Tensor> run_dequant_braided_ilp_bench(
        torch::Tensor input, int64_t variant, int64_t blocks) {
    TORCH_CHECK(input.is_cuda() && input.scalar_type() == torch::kUInt8);
    TORCH_CHECK(input.numel() == kRows * kPackedRowBytes);
    auto cycles = torch::empty({blocks}, input.options().dtype(torch::kInt64));
    auto output = torch::empty({kRows, kFp8RowBytes}, input.options());
    switch (variant) {
        case 0: launch<false, kWordSerial>(input, cycles, output); break;
        case 1: launch<true,  kWordSerial>(input, cycles, output); break;
        case 2: launch<true,  kPairIlp>(input, cycles, output); break;
        case 3: launch<true,  kQuadIlp>(input, cycles, output); break;
        case 4: launch<true,  kWordStoreFirst>(input, cycles, output); break;
        default: TORCH_CHECK(false, "unknown variant");
    }
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return {cycles, output};
}
"""


VARIANTS = {
    0: "empty",
    1: "braided/word-serial",
    2: "braided/pair-ilp",
    3: "braided/quad-ilp",
    4: "braided/store-first",
}


def load_extension():
    cpp_src = (
        "std::vector<torch::Tensor> run_dequant_braided_ilp_bench("
        "torch::Tensor, int64_t, int64_t);"
    )
    return load_inline(
        name="deepgemm_sm90_nvfp4_dequant_braided_ilp_v2",
        cpp_sources=cpp_src,
        cuda_sources=CUDA_SRC,
        functions=["run_dequant_braided_ilp_bench"],
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
            signs[..., 4], signs[..., 0], signs[..., 5], signs[..., 1],
            signs[..., 6], signs[..., 2], signs[..., 7], signs[..., 3],
        ),
        dim=-1,
    )
    braided_nibbles = magnitudes | (braided_signs << 3)
    braided_bytes = braided_nibbles[..., 0::2] | (braided_nibbles[..., 1::2] << 4)
    result[:, :64] = braided_bytes.reshape(128, 64)
    return result


def make_rows(scale_pattern: str) -> torch.Tensor:
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
    return braid_rows(rows).cuda().contiguous().view(-1)


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

    packed = make_rows(args.scale_pattern)
    ext = load_extension()
    reference = None
    samples = {variant: [] for variant in args.variants}
    for round_idx in range(args.rounds):
        order = args.variants if round_idx % 2 == 0 else list(reversed(args.variants))
        for variant in order:
            cycles, output = ext.run_dequant_braided_ilp_bench(
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
            f"net={center - empty:8.1f} rounds={[round(value, 1) for value in samples[variant]]}"
        )


if __name__ == "__main__":
    main()
