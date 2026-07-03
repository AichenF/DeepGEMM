"""Isolated SM90 NVFP4 decoder benchmark for LUT storage layouts.

This harness deliberately has no production wiring.  It compares the current
shared AoS LUT against padded replicas, a split x/y LUT, and direct constant
loads for the two small-M selector schedules used by Flash and Pro.
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
constexpr int kPackedRowBytes = 80;
constexpr int kFp8RowBytes = 128;
constexpr int kReplicaStride = 129;

enum LutMode : int {
    kSharedAos = 0,
    kReplicatedAos = 1,
    kSharedSoa = 2,
    kDirectConstant = 3,
    kDualEncoded = 4,
};

template <int kMode, int kCopies>
constexpr int lut_bytes() {
    if constexpr (kMode == kReplicatedAos)
        return kCopies * kReplicaStride * static_cast<int>(sizeof(uint2));
    if constexpr (kMode == kDualEncoded)
        return 256 * static_cast<int>(sizeof(uint2));
    if constexpr (kMode == kDirectConstant)
        return 0;
    return 128 * static_cast<int>(sizeof(uint2));
}

template <int kMode, int kCopies>
__device__ __forceinline__ void init_lut(uint8_t* __restrict__ lut_storage,
                                         uint32_t tid) {
    if constexpr (kMode == kSharedAos) {
        reinterpret_cast<uint2*>(lut_storage)[tid] =
            deep_gemm::nvfp4::kE2M1AndUe4m3ToFp8Lut[tid];
    } else if constexpr (kMode == kReplicatedAos) {
#pragma unroll
        for (int copy = 0; copy < kCopies; ++copy) {
            reinterpret_cast<uint2*>(lut_storage)[copy * kReplicaStride + tid] =
                deep_gemm::nvfp4::kE2M1AndUe4m3ToFp8Lut[tid];
        }
    } else if constexpr (kMode == kSharedSoa) {
        const uint2 entry = deep_gemm::nvfp4::kE2M1AndUe4m3ToFp8Lut[tid];
        auto* words = reinterpret_cast<uint32_t*>(lut_storage);
        words[tid] = entry.x;
        words[128 + tid] = entry.y;
    } else if constexpr (kMode == kDualEncoded) {
        const uint2 entry = deep_gemm::nvfp4::kE2M1AndUe4m3ToFp8Lut[tid];
        auto* lut = reinterpret_cast<uint2*>(lut_storage);
        lut[tid] = entry;
        lut[128 + ((tid + kCopies) & 127)] = entry;
    }
}

template <int kMode, int kCopies>
__device__ __forceinline__ uint2 load_lut(const uint8_t* __restrict__ lut_storage,
                                          uint32_t scale, uint32_t row) {
    if constexpr (kMode == kDualEncoded) {
        return reinterpret_cast<const uint2*>(lut_storage)[scale];
    } else if constexpr (kMode == kSharedAos) {
        scale &= 0x7fu;
        return reinterpret_cast<const uint2*>(lut_storage)[scale];
    } else if constexpr (kMode == kReplicatedAos) {
        scale &= 0x7fu;
        const uint32_t copy = row & (kCopies - 1);
        return reinterpret_cast<const uint2*>(lut_storage)[copy * kReplicaStride + scale];
    } else if constexpr (kMode == kSharedSoa) {
        scale &= 0x7fu;
        const auto* words = reinterpret_cast<const uint32_t*>(lut_storage);
        return make_uint2(words[scale], words[128 + scale]);
    } else {
        scale &= 0x7fu;
        return deep_gemm::nvfp4::kE2M1AndUe4m3ToFp8Lut[scale];
    }
}

template <bool kUseDp4aHi, bool kUseDp4aLo>
__device__ __forceinline__ uint2 decode_word(uint32_t q, const uint2& lut) {
    return deep_gemm::nvfp4::dequant_nvfp4_to_fp8_pair_with_lut<
        kUseDp4aHi, kUseDp4aLo>(q, lut);
}

template <bool kPaired, bool kUseDp4aHi, bool kUseDp4aLo, int kMode, int kCopies>
__device__ __forceinline__ void decode_row(
        uint8_t* __restrict__ fp8,
        const uint8_t* __restrict__ packed,
        uint32_t row,
        const uint8_t* __restrict__ lut_storage) {
    const uint8_t* __restrict__ row_ptr = packed + row * kPackedRowBytes;
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
            const uint2 lut0 = load_lut<kMode, kCopies>(lut_storage, scale0, row);
            const uint2 lut1 = load_lut<kMode, kCopies>(lut_storage, scale1, row);
            const uint2 q0 = decode_word<kUseDp4aHi, kUseDp4aLo>(q.x, lut0);
            const uint2 q1 = decode_word<kUseDp4aHi, kUseDp4aLo>(q.y, lut0);
            *reinterpret_cast<uint4*>(fp8_dst + ((scale_i0 * 16) ^ row_swizzle)) =
                make_uint4(q0.x, q0.y, q1.x, q1.y);
            const uint2 q2 = decode_word<kUseDp4aHi, kUseDp4aLo>(q.z, lut1);
            const uint2 q3 = decode_word<kUseDp4aHi, kUseDp4aLo>(q.w, lut1);
            *reinterpret_cast<uint4*>(fp8_dst + ((scale_i1 * 16) ^ row_swizzle)) =
                make_uint4(q2.x, q2.y, q3.x, q3.y);
        } else {
            const uint2 lut0 = load_lut<kMode, kCopies>(lut_storage, scale0, row);
            const uint2 q0 = decode_word<kUseDp4aHi, kUseDp4aLo>(q.x, lut0);
            const uint2 q1 = decode_word<kUseDp4aHi, kUseDp4aLo>(q.y, lut0);
            *reinterpret_cast<uint4*>(fp8_dst + ((scale_i0 * 16) ^ row_swizzle)) =
                make_uint4(q0.x, q0.y, q1.x, q1.y);
            const uint32_t scale1 = (scale_word >> ((scale_i1 & 3) * 8)) & 0xffu;
            const uint2 lut1 = load_lut<kMode, kCopies>(lut_storage, scale1, row);
            const uint2 q2 = decode_word<kUseDp4aHi, kUseDp4aLo>(q.z, lut1);
            const uint2 q3 = decode_word<kUseDp4aHi, kUseDp4aLo>(q.w, lut1);
            *reinterpret_cast<uint4*>(fp8_dst + ((scale_i1 * 16) ^ row_swizzle)) =
                make_uint4(q2.x, q2.y, q3.x, q3.y);
        }
    }
}

template <bool kDecode, bool kPaired, bool kUseDp4aHi, bool kUseDp4aLo,
          int kMode, int kCopies>
__global__ __launch_bounds__(128) void bench_kernel(
        const uint8_t* __restrict__ input,
        int64_t* __restrict__ cycles,
        int32_t* __restrict__ witnesses) {
    extern __shared__ __align__(16) uint8_t smem[];
    uint8_t* packed = smem;
    uint8_t* fp8 = packed + kRows * kPackedRowBytes;
    uint8_t* lut_storage = fp8 + kRows * kFp8RowBytes;
    const uint32_t tid = threadIdx.x;

    for (int i = tid; i < kRows * kPackedRowBytes; i += blockDim.x)
        packed[i] = input[i];
    for (int i = tid; i < kRows * kFp8RowBytes; i += blockDim.x)
        fp8[i] = 0;
    init_lut<kMode, kCopies>(lut_storage, tid);
    __syncthreads();

    uint64_t start = 0;
    if (tid == 0)
        start = clock64();
    __syncthreads();
    if constexpr (kDecode) {
        decode_row<kPaired, kUseDp4aHi, kUseDp4aLo, kMode, kCopies>(
            fp8, packed, tid, lut_storage);
    }
    __syncthreads();
    if (tid == 0)
        cycles[blockIdx.x] = static_cast<int64_t>(clock64() - start);

    const uint32_t* row_words = reinterpret_cast<const uint32_t*>(fp8 + tid * kFp8RowBytes);
    witnesses[blockIdx.x * kRows + tid] = static_cast<int32_t>(row_words[tid & 31u]);
}

template <bool kDecode, bool kPaired, bool kUseDp4aHi, bool kUseDp4aLo,
          int kMode, int kCopies>
void launch(const torch::Tensor& input, torch::Tensor& cycles, torch::Tensor& witnesses) {
    constexpr int kSharedBytes =
        kRows * (kPackedRowBytes + kFp8RowBytes) + lut_bytes<kMode, kCopies>();
    const int blocks = static_cast<int>(cycles.numel());
    bench_kernel<kDecode, kPaired, kUseDp4aHi, kUseDp4aLo, kMode, kCopies>
        <<<blocks, 128, kSharedBytes>>>(
            input.data_ptr<uint8_t>(), cycles.data_ptr<int64_t>(),
            witnesses.data_ptr<int32_t>());
}

}  // namespace

std::vector<torch::Tensor> run_dequant_lut_bench(
        torch::Tensor input, int64_t variant, int64_t blocks) {
    TORCH_CHECK(input.is_cuda() && input.scalar_type() == torch::kUInt8);
    TORCH_CHECK(input.numel() == kRows * kPackedRowBytes);
    auto cycles = torch::empty({blocks}, input.options().dtype(torch::kInt64));
    auto witnesses = torch::empty({blocks, kRows}, input.options().dtype(torch::kInt32));
    switch (variant) {
        case 0:  launch<false, false, false, true,  kSharedAos,      1>(input, cycles, witnesses); break;
        case 1:  launch<true,  false, false, true,  kSharedAos,      1>(input, cycles, witnesses); break;
        case 2:  launch<true,  true,  true,  true,  kSharedAos,      1>(input, cycles, witnesses); break;
        case 3:  launch<true,  false, false, true,  kReplicatedAos,  2>(input, cycles, witnesses); break;
        case 4:  launch<true,  false, false, true,  kReplicatedAos,  4>(input, cycles, witnesses); break;
        case 5:  launch<true,  false, false, true,  kReplicatedAos,  8>(input, cycles, witnesses); break;
        case 6:  launch<true,  false, false, true,  kReplicatedAos, 16>(input, cycles, witnesses); break;
        case 7:  launch<true,  true,  true,  true,  kReplicatedAos,  2>(input, cycles, witnesses); break;
        case 8:  launch<true,  true,  true,  true,  kReplicatedAos,  4>(input, cycles, witnesses); break;
        case 9:  launch<true,  true,  true,  true,  kReplicatedAos,  8>(input, cycles, witnesses); break;
        case 10: launch<true,  true,  true,  true,  kReplicatedAos, 16>(input, cycles, witnesses); break;
        case 11: launch<true,  false, false, true,  kSharedSoa,      1>(input, cycles, witnesses); break;
        case 12: launch<true,  true,  true,  true,  kSharedSoa,      1>(input, cycles, witnesses); break;
        case 13: launch<true,  false, false, true,  kDirectConstant, 1>(input, cycles, witnesses); break;
        case 14: launch<true,  true,  true,  true,  kDirectConstant, 1>(input, cycles, witnesses); break;
        case 15: launch<true,  false, false, true,  kDualEncoded,    1>(input, cycles, witnesses); break;
        case 16: launch<true,  true,  true,  true,  kDualEncoded,    1>(input, cycles, witnesses); break;
        case 17: launch<true,  false, false, true,  kDualEncoded,    3>(input, cycles, witnesses); break;
        case 18: launch<true,  true,  true,  true,  kDualEncoded,    3>(input, cycles, witnesses); break;
        case 19: launch<true,  false, false, true,  kDualEncoded,    5>(input, cycles, witnesses); break;
        case 20: launch<true,  true,  true,  true,  kDualEncoded,    5>(input, cycles, witnesses); break;
        case 21: launch<true,  false, false, true,  kDualEncoded,    7>(input, cycles, witnesses); break;
        case 22: launch<true,  true,  true,  true,  kDualEncoded,    7>(input, cycles, witnesses); break;
        default: TORCH_CHECK(false, "unknown variant");
    }
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return {cycles, witnesses};
}
"""


VARIANTS = {
    0: "empty",
    1: "flash/shared-aos",
    2: "pro/shared-aos",
    3: "flash/replicated-2",
    4: "flash/replicated-4",
    5: "flash/replicated-8",
    6: "flash/replicated-16",
    7: "pro/replicated-2",
    8: "pro/replicated-4",
    9: "pro/replicated-8",
    10: "pro/replicated-16",
    11: "flash/shared-soa",
    12: "pro/shared-soa",
    13: "flash/direct-constant",
    14: "pro/direct-constant",
    15: "flash/dual-encoded-r1",
    16: "pro/dual-encoded-r1",
    17: "flash/dual-encoded-r3",
    18: "pro/dual-encoded-r3",
    19: "flash/dual-encoded-r5",
    20: "pro/dual-encoded-r5",
    21: "flash/dual-encoded-r7",
    22: "pro/dual-encoded-r7",
}

DUAL_ROTATIONS = {
    15: 1,
    16: 1,
    17: 3,
    18: 3,
    19: 5,
    20: 5,
    21: 7,
    22: 7,
}


def load_extension():
    cpp_src = "std::vector<torch::Tensor> run_dequant_lut_bench(torch::Tensor, int64_t, int64_t);"
    return load_inline(
        name="deepgemm_sm90_nvfp4_dequant_lut_layouts",
        cpp_sources=cpp_src,
        cuda_sources=CUDA_SRC,
        functions=["run_dequant_lut_bench"],
        extra_include_paths=[os.path.join(REPO_ROOT, "deep_gemm", "include")],
        extra_cuda_cflags=["-O3", "-lineinfo", "--expt-relaxed-constexpr"],
        verbose=False,
    )


def make_rows(scale_pattern: str) -> torch.Tensor:
    torch.manual_seed(1234)
    rows = torch.randint(0, 256, (128, 80), dtype=torch.uint8, device="cuda")
    if scale_pattern == "random":
        scales = torch.randint(0, 127, (128, 8), dtype=torch.uint8, device="cuda")
    elif scale_pattern == "clustered":
        scales = torch.randint(48, 56, (128, 8), dtype=torch.uint8, device="cuda")
    elif scale_pattern == "constant":
        scales = torch.full((128, 8), 52, dtype=torch.uint8, device="cuda")
    elif scale_pattern == "model":
        from deep_gemm.quantization_nvfp4 import fp32_to_ue4m3_ceil

        weights = torch.randn((128, 8, 16), dtype=torch.float32, device="cuda") * 0.05
        scales = fp32_to_ue4m3_ceil(weights.abs().amax(dim=-1) / 6.0)
    else:
        raise ValueError(scale_pattern)
    rows[:, 64:72] = scales
    rows[:, 72:80] = scales
    return rows.contiguous().view(-1)


def dual_encode_rows(packed: torch.Tensor, rotation: int) -> torch.Tensor:
    """Choose one of two equivalent LUT indices per warp and scale position."""
    rows = packed.view(128, 80).clone()
    raw_scales = rows[:, 64:72].cpu().to(torch.int64)
    encoded = raw_scales.clone()

    for warp_start in range(0, 128, 32):
        for scale_i in range(8):
            values = raw_scales[warp_start : warp_start + 32, scale_i].tolist()
            frequencies = {value: values.count(value) for value in set(values)}
            distinct = sorted(frequencies, key=lambda value: (-frequencies[value], value))
            occupancy = [0] * 16
            assignment = {}

            for value in distinct:
                classes = (value & 15, (value + rotation) & 15)
                color = 0 if occupancy[classes[0]] <= occupancy[classes[1]] else 1
                assignment[value] = color
                occupancy[classes[color]] += 1

            changed = True
            while changed:
                changed = False
                for value in distinct:
                    old_color = assignment[value]
                    classes = (value & 15, (value + rotation) & 15)
                    occupancy[classes[old_color]] -= 1
                    other_color = 1 - old_color
                    new_color = (
                        other_color
                        if occupancy[classes[other_color]] < occupancy[classes[old_color]]
                        else old_color
                    )
                    occupancy[classes[new_color]] += 1
                    if new_color != old_color:
                        assignment[value] = new_color
                        changed = True

            for lane, value in enumerate(values):
                if assignment[value]:
                    encoded[warp_start + lane, scale_i] = 128 + ((value + rotation) & 127)

    encoded_gpu = encoded.to(device=rows.device, dtype=torch.uint8)
    rows[:, 64:72] = encoded_gpu
    rows[:, 72:80] = encoded_gpu
    return rows.contiguous().view(-1)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--blocks", type=int, default=624)
    parser.add_argument("--rounds", type=int, default=11)
    parser.add_argument(
        "--scale-pattern",
        choices=("random", "clustered", "constant", "model"),
        default="random",
    )
    parser.add_argument("--variants", type=int, nargs="+", default=list(VARIANTS))
    args = parser.parse_args()

    assert torch.cuda.get_device_capability()[0] == 9
    if any(variant not in VARIANTS for variant in args.variants):
        raise ValueError(f"variants must be in {list(VARIANTS)}")

    packed = make_rows(args.scale_pattern)
    packed_inputs = {rotation: dual_encode_rows(packed, rotation) for rotation in (1, 3, 5, 7)}
    ext = load_extension()
    reference = None
    samples = {variant: [] for variant in args.variants}
    for round_idx in range(args.rounds):
        order = args.variants if round_idx % 2 == 0 else list(reversed(args.variants))
        for variant in order:
            input_rows = packed_inputs[DUAL_ROTATIONS[variant]] if variant in DUAL_ROTATIONS else packed
            cycles, witnesses = ext.run_dequant_lut_bench(input_rows, variant, args.blocks)
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
            f"{variant:2d} {VARIANTS[variant]:27s} raw={center:8.1f} "
            f"net={center - empty:8.1f} rounds={[round(value, 1) for value in samples[variant]]}"
        )


if __name__ == "__main__":
    main()
