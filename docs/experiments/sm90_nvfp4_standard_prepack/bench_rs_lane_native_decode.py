"""Screen lane-native NVFP4 prepack layouts for SM90 RS WGMMA.

The harness measures only the register-fragment decoder.  It compares the
previous pair-native strategy, where neighboring lanes consume opposite
nibble halves, with a layout that gives every lane four independent groups of
four E2M1 values in the exact CUTLASS ``ALayout_64x32`` register order.  The
last variant reproduces grouped_shawn.py's two-seed FP8 reconstruction so its
register decode cost can be compared directly with the retained LUT decoder.
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

constexpr int kThreads = 128;
constexpr int kPackedBytes = 1024;
constexpr int kScaleBytes = 128;

__device__ __forceinline__ uint32_t decode_pair_half(
        uint32_t q, const uint2& lut, bool use_low_half) {
    if (use_low_half) {
        const uint32_t selector =
            deep_gemm::nvfp4::pack_nvfp4_magnitude_selector<true>(
                q & 0x07070707u);
        uint32_t out = deep_gemm::nvfp4::byte_perm_unchecked(
            lut.x, lut.y, selector);
        out |= (q << 4) & 0x80808080u;
        return out;
    }
    const uint32_t selector =
        deep_gemm::nvfp4::pack_nvfp4_magnitude_selector<true>(
            (q >> 4) & 0x07070707u);
    uint32_t out = deep_gemm::nvfp4::byte_perm_unchecked(
        lut.x, lut.y, selector);
    out |= q & 0x80808080u;
    return out;
}

__device__ __forceinline__ uint32_t decode_pair_half_branchless(
        uint32_t q, const uint2& lut, bool use_low_half) {
    const uint32_t sel_hi =
        deep_gemm::nvfp4::pack_nvfp4_magnitude_selector<true>(
            (q >> 4) & 0x07070707u);
    const uint32_t sel_lo =
        deep_gemm::nvfp4::pack_nvfp4_magnitude_selector<true>(
            q & 0x07070707u);
    uint32_t out_hi = deep_gemm::nvfp4::byte_perm_unchecked(
        lut.x, lut.y, sel_hi);
    uint32_t out_lo = deep_gemm::nvfp4::byte_perm_unchecked(
        lut.x, lut.y, sel_lo);
    out_hi |= q & 0x80808080u;
    out_lo |= (q << 4) & 0x80808080u;
    return use_low_half ? out_lo : out_hi;
}

__device__ __forceinline__ uint32_t apply_group_signs(
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

__device__ __forceinline__ uint32_t decode_group(
        uint32_t grouped_nibbles, const uint2& lut) {
    const uint32_t selector = grouped_nibbles & 0x7777u;
    const uint32_t magnitudes = deep_gemm::nvfp4::byte_perm_unchecked(
        lut.x, lut.y, selector);
    return apply_group_signs(magnitudes, grouped_nibbles);
}

__device__ __forceinline__ uint2 decode_braided_groups(
        uint32_t braided, const uint2& lut0, const uint2& lut1) {
    const uint32_t sel0 = braided & 0x00007777u;
    const uint32_t sel1 = (braided >> 16) & 0x00007777u;
    uint32_t out0 = deep_gemm::nvfp4::byte_perm_unchecked(
        lut0.x, lut0.y, sel0);
    uint32_t out1 = deep_gemm::nvfp4::byte_perm_unchecked(
        lut1.x, lut1.y, sel1);
    out0 |= braided & 0x80808080u;
    out1 |= (braided << 4) & 0x80808080u;
    return make_uint2(out0, out1);
}

__device__ __forceinline__ uint32_t pack_two_seed_metadata(
        const uint2& lut0, const uint2& lut1) {
    const uint32_t half0 = (lut0.x >> 8) & 0xffu;
    const uint32_t one_half0 = (lut0.x >> 24) & 0xffu;
    const uint32_t half1 = (lut1.x >> 8) & 0xffu;
    const uint32_t one_half1 = (lut1.x >> 24) & 0xffu;
    return half0 | (one_half0 << 8) | (half1 << 16) | (one_half1 << 24);
}

__device__ __forceinline__ uint2 decode_two_seed_groups(
        uint32_t packed_e2m1, uint32_t base_codes) {
    uint32_t buffer10 = __byte_perm(base_codes, 0u, 0x1004) + 0x00080000u;
    uint32_t buffer20 =
        __byte_perm(base_codes, base_codes, 0x1010) + 0x10180810u;
    uint32_t buffer11 = __byte_perm(base_codes, 0u, 0x3224) + 0x00080000u;
    uint32_t buffer21 =
        __byte_perm(base_codes, base_codes, 0x3232) + 0x10180810u;

    uint32_t magnitudes0 = __byte_perm(buffer10, buffer20, packed_e2m1);
    uint32_t magnitudes1 =
        __byte_perm(buffer11, buffer21, packed_e2m1 >> 16);
    uint32_t out0;
    uint32_t out1;
    asm("lop3.b32 %0, %1, %2, 0x80808080, 0xf8;"
        : "=r"(out0) : "r"(magnitudes0), "r"(packed_e2m1 << 4));
    asm("lop3.b32 %0, %1, %2, 0x80808080, 0xf8;"
        : "=r"(out1) : "r"(magnitudes1), "r"(packed_e2m1));
    return make_uint2(out0, out1);
}

template <int kVariant>
__device__ __forceinline__ uint4 decode_fragment(
        const uint8_t* __restrict__ packed,
        const uint8_t* __restrict__ scales,
        const uint2* __restrict__ seeds,
        const uint2* __restrict__ lut,
        uint32_t tid) {
    if constexpr (kVariant == 5) {
        const uint2 q = *reinterpret_cast<const uint2*>(packed + tid * 8u);
        const uint2 base_codes = seeds[tid >> 2];
        const uint2 out01 = decode_two_seed_groups(q.x, base_codes.x);
        const uint2 out23 = decode_two_seed_groups(q.y, base_codes.y);
        return make_uint4(out01.x, out01.y, out23.x, out23.y);
    }

    const uint32_t scale_word =
        *reinterpret_cast<const uint32_t*>(scales + (tid >> 2) * 4u);
    const uint32_t s0 = scale_word & 0x7fu;
    const uint32_t s1 = (scale_word >> 8) & 0x7fu;
    const uint32_t s2 = (scale_word >> 16) & 0x7fu;
    const uint32_t s3 = (scale_word >> 24) & 0x7fu;
    const uint2 lut0 = lut[s0];
    const uint2 lut1 = lut[s1];
    const uint2 lut2 = lut[s2];
    const uint2 lut3 = lut[s3];

    if constexpr (kVariant == 1 || kVariant == 2) {
        const uint4 q = *reinterpret_cast<const uint4*>(
            packed + (tid >> 1) * 16u);
        const bool use_low_half = (tid & 1u) != 0u;
        if constexpr (kVariant == 1) {
            return make_uint4(
                decode_pair_half(q.x, lut0, use_low_half),
                decode_pair_half(q.y, lut1, use_low_half),
                decode_pair_half(q.z, lut2, use_low_half),
                decode_pair_half(q.w, lut3, use_low_half));
        } else {
            return make_uint4(
                decode_pair_half_branchless(q.x, lut0, use_low_half),
                decode_pair_half_branchless(q.y, lut1, use_low_half),
                decode_pair_half_branchless(q.z, lut2, use_low_half),
                decode_pair_half_branchless(q.w, lut3, use_low_half));
        }
    } else if constexpr (kVariant == 3) {
        const uint2 q = *reinterpret_cast<const uint2*>(packed + tid * 8u);
        return make_uint4(
            decode_group(q.x & 0xffffu, lut0),
            decode_group(q.x >> 16, lut1),
            decode_group(q.y & 0xffffu, lut2),
            decode_group(q.y >> 16, lut3));
    } else {
        const uint2 q = *reinterpret_cast<const uint2*>(packed + tid * 8u);
        const uint2 out01 = decode_braided_groups(q.x, lut0, lut1);
        const uint2 out23 = decode_braided_groups(q.y, lut2, lut3);
        return make_uint4(out01.x, out01.y, out23.x, out23.y);
    }
}

template <int kVariant>
__global__ __launch_bounds__(kThreads) void bench_kernel(
        const uint8_t* __restrict__ input_packed,
        const uint8_t* __restrict__ input_scales,
        int64_t* __restrict__ cycles,
        uint32_t* __restrict__ witnesses) {
    __shared__ __align__(16) uint8_t packed[kPackedBytes];
    __shared__ __align__(16) uint8_t scales[kScaleBytes];
    __shared__ __align__(16) uint2 seeds[32];
    __shared__ __align__(16) uint2 lut[128];

    const uint32_t tid = threadIdx.x;
    for (uint32_t i = tid; i < kPackedBytes; i += kThreads)
        packed[i] = input_packed[i];
    scales[tid] = input_scales[tid];
    lut[tid] = deep_gemm::nvfp4::kE2M1AndUe4m3ToFp8Lut[tid];
    if (tid < 32) {
        const uint32_t scale_word =
            *reinterpret_cast<const uint32_t*>(input_scales + tid * 4u);
        const uint2 lut0 =
            deep_gemm::nvfp4::kE2M1AndUe4m3ToFp8Lut[scale_word & 0x7fu];
        const uint2 lut1 =
            deep_gemm::nvfp4::kE2M1AndUe4m3ToFp8Lut[(scale_word >> 8) & 0x7fu];
        const uint2 lut2 =
            deep_gemm::nvfp4::kE2M1AndUe4m3ToFp8Lut[(scale_word >> 16) & 0x7fu];
        const uint2 lut3 =
            deep_gemm::nvfp4::kE2M1AndUe4m3ToFp8Lut[(scale_word >> 24) & 0x7fu];
        seeds[tid] = make_uint2(
            pack_two_seed_metadata(lut0, lut1),
            pack_two_seed_metadata(lut2, lut3));
    }
    __syncthreads();

    uint64_t start = 0;
    if (tid == 0)
        start = clock64();
    __syncthreads();

    uint4 result = {};
    if constexpr (kVariant != 0)
        result = decode_fragment<kVariant>(packed, scales, seeds, lut, tid);
    asm volatile("" : : "r"(result.x), "r"(result.y),
                          "r"(result.z), "r"(result.w) : "memory");
    __syncthreads();
    if (tid == 0)
        cycles[blockIdx.x] = static_cast<int64_t>(clock64() - start);

    const uint64_t output_base =
        (static_cast<uint64_t>(blockIdx.x) * kThreads + tid) * 4u;
    witnesses[output_base + 0] = result.x;
    witnesses[output_base + 1] = result.y;
    witnesses[output_base + 2] = result.z;
    witnesses[output_base + 3] = result.w;
}

template <int kVariant>
void launch(const torch::Tensor& packed, const torch::Tensor& scales,
            torch::Tensor& cycles, torch::Tensor& witnesses) {
    bench_kernel<kVariant><<<static_cast<int>(cycles.numel()), kThreads>>>(
        packed.data_ptr<uint8_t>(), scales.data_ptr<uint8_t>(),
        cycles.data_ptr<int64_t>(), witnesses.data_ptr<uint32_t>());
}

}  // namespace

std::vector<torch::Tensor> run_rs_decode_bench(
        torch::Tensor packed, torch::Tensor scales,
        int64_t variant, int64_t blocks) {
    TORCH_CHECK(packed.is_cuda() && packed.scalar_type() == torch::kUInt8);
    TORCH_CHECK(scales.is_cuda() && scales.scalar_type() == torch::kUInt8);
    TORCH_CHECK(packed.numel() == kPackedBytes);
    TORCH_CHECK(scales.numel() == kScaleBytes);
    auto cycles = torch::empty(
        {blocks}, packed.options().dtype(torch::kInt64));
    auto witnesses = torch::empty(
        {blocks, kThreads, 4}, packed.options().dtype(torch::kUInt32));
    switch (variant) {
        case 0: launch<0>(packed, scales, cycles, witnesses); break;
        case 1: launch<1>(packed, scales, cycles, witnesses); break;
        case 2: launch<2>(packed, scales, cycles, witnesses); break;
        case 3: launch<3>(packed, scales, cycles, witnesses); break;
        case 4: launch<4>(packed, scales, cycles, witnesses); break;
        case 5: launch<5>(packed, scales, cycles, witnesses); break;
        default: TORCH_CHECK(false, "unknown variant");
    }
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return {cycles, witnesses};
}
"""


VARIANTS = {
    0: "empty",
    1: "pair-native/divergent-half",
    2: "pair-native/branchless-both",
    3: "lane-native/grouped-nibble",
    4: "lane-native/braided-signs",
    5: "lane-native/two-seed-shawn",
}


def _lane_coordinates(tid: int) -> tuple[int, int, int]:
    t0 = tid % 4
    t1 = (tid // 4) % 8
    t2 = tid // 32
    row0 = t1 + 16 * t2
    row1 = row0 + 8
    return row0, row1, 4 * t0


def _lane_groups(codes: torch.Tensor) -> torch.Tensor:
    groups = torch.empty((128, 4, 4), dtype=torch.uint8)
    for tid in range(128):
        row0, row1, k0 = _lane_coordinates(tid)
        groups[tid, 0] = codes[row0, k0 : k0 + 4]
        groups[tid, 1] = codes[row1, k0 : k0 + 4]
        groups[tid, 2] = codes[row0, k0 + 16 : k0 + 20]
        groups[tid, 3] = codes[row1, k0 + 16 : k0 + 20]
    return groups


def _pack_pair_native(groups: torch.Tensor) -> torch.Tensor:
    packed = torch.empty((64, 4, 4), dtype=torch.uint8)
    for pair in range(64):
        even = groups[pair * 2].to(torch.int16)
        odd = groups[pair * 2 + 1].to(torch.int16)
        packed[pair] = ((even << 4) | odd).to(torch.uint8)
    return packed.contiguous().view(-1)


def _pack_lane_native(groups: torch.Tensor) -> torch.Tensor:
    packed = torch.zeros((128, 8), dtype=torch.uint8)
    for tid in range(128):
        for group_idx in range(4):
            values = groups[tid, group_idx].to(torch.int32)
            word = sum(int(values[i].item()) << (4 * i) for i in range(4))
            byte_offset = group_idx * 2
            packed[tid, byte_offset] = word & 0xFF
            packed[tid, byte_offset + 1] = (word >> 8) & 0xFF
    return packed.contiguous().view(-1)


def _pack_lane_native_braided(groups: torch.Tensor) -> torch.Tensor:
    packed = torch.zeros((128, 8), dtype=torch.uint8)
    for tid in range(128):
        for pair_idx in range(2):
            group0 = groups[tid, pair_idx * 2].to(torch.int32)
            group1 = groups[tid, pair_idx * 2 + 1].to(torch.int32)
            values = torch.cat([group0, group1])
            magnitudes = values & 0x7
            signs = values >> 3
            sign_order = torch.stack(
                [signs[4], signs[0], signs[5], signs[1],
                 signs[6], signs[2], signs[7], signs[3]]
            )
            nibbles = magnitudes | (sign_order << 3)
            word = sum(int(nibbles[i].item()) << (4 * i) for i in range(8))
            byte_offset = pair_idx * 4
            for byte_idx in range(4):
                packed[tid, byte_offset + byte_idx] = (word >> (8 * byte_idx)) & 0xFF
    return packed.contiguous().view(-1)


def _pack_lane_native_two_seed(groups: torch.Tensor) -> torch.Tensor:
    """Pack magnitude selectors and the sign plane expected by Shawn's E91."""
    packed = torch.zeros((128, 8), dtype=torch.uint8)
    for tid in range(128):
        for pair_idx in range(2):
            group0 = groups[tid, pair_idx * 2].to(torch.int32)
            group1 = groups[tid, pair_idx * 2 + 1].to(torch.int32)
            values = torch.cat([group0, group1])
            magnitudes = values & 0x7
            signs = values >> 3
            sign_order = torch.stack(
                [signs[0], signs[4], signs[1], signs[5],
                 signs[2], signs[6], signs[3], signs[7]]
            )
            nibbles = magnitudes | (sign_order << 3)
            word = sum(int(nibbles[i].item()) << (4 * i) for i in range(8))
            byte_offset = pair_idx * 4
            for byte_idx in range(4):
                packed[tid, byte_offset + byte_idx] = (word >> (8 * byte_idx)) & 0xFF
    return packed.contiguous().view(-1)


def _pack_scale_records(scales: torch.Tensor) -> torch.Tensor:
    records = torch.empty((32, 4), dtype=torch.uint8)
    for group in range(32):
        row0, row1, _ = _lane_coordinates(group * 4)
        records[group] = torch.tensor(
            [scales[row0, 0], scales[row1, 0],
             scales[row0, 1], scales[row1, 1]],
            dtype=torch.uint8,
        )
    return records.contiguous().view(-1)


def _make_inputs(exhaustive: bool) -> tuple[torch.Tensor, ...]:
    if exhaustive:
        codes = torch.arange(32, dtype=torch.uint8).view(1, 32).expand(64, 32) & 0xF
        # The exponent-shift reconstruction is exact only away from FP8
        # subnormal and saturation boundaries.  grouped_shawn.py enforces the
        # upper bound by expert-tensor normalization; 16..105 screens the
        # lossless interior separately from that model-level policy.
        scale_ids = (
            torch.arange(128, dtype=torch.int16).remainder(90) + 16
        ).to(torch.uint8)
        scales = scale_ids.view(64, 2)
    else:
        generator = torch.Generator().manual_seed(1234)
        codes = torch.randint(0, 16, (64, 32), dtype=torch.uint8, generator=generator)
        scales = torch.randint(32, 96, (64, 2), dtype=torch.uint8, generator=generator)
    groups = _lane_groups(codes)
    return (
        _pack_pair_native(groups).cuda(),
        _pack_lane_native(groups).cuda(),
        _pack_lane_native_braided(groups).cuda(),
        _pack_lane_native_two_seed(groups).cuda(),
        _pack_scale_records(scales).cuda(),
    )


def load_extension():
    cpp_src = (
        "std::vector<torch::Tensor> run_rs_decode_bench("
        "torch::Tensor, torch::Tensor, int64_t, int64_t);"
    )
    return load_inline(
        name="deepgemm_sm90_nvfp4_rs_lane_native_decode_bench_two_seed",
        cpp_sources=cpp_src,
        cuda_sources=CUDA_SRC,
        functions=["run_rs_decode_bench"],
        extra_include_paths=[os.path.join(REPO_ROOT, "deep_gemm", "include")],
        extra_cuda_cflags=["-O3", "-lineinfo", "--expt-relaxed-constexpr"],
        verbose=False,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--blocks", type=int, default=624)
    parser.add_argument("--rounds", type=int, default=9)
    parser.add_argument("--variants", type=int, nargs="+", default=list(VARIANTS))
    args = parser.parse_args()

    assert torch.cuda.get_device_capability()[0] == 9
    ext = load_extension()

    for exhaustive in (True, False):
        pair_packed, lane_packed, braided_packed, two_seed_packed, scales = _make_inputs(exhaustive)
        reference = None
        samples = {variant: [] for variant in args.variants}
        for round_idx in range(args.rounds):
            order = args.variants if round_idx % 2 == 0 else list(reversed(args.variants))
            for variant in order:
                if variant == 3:
                    packed = lane_packed
                elif variant == 4:
                    packed = braided_packed
                elif variant == 5:
                    packed = two_seed_packed
                else:
                    packed = pair_packed
                cycles, witnesses = ext.run_rs_decode_bench(
                    packed, scales, variant, args.blocks)
                torch.cuda.synchronize()
                if variant != 0:
                    candidate = witnesses[0].cpu()
                    if reference is None:
                        reference = candidate
                    else:
                        torch.testing.assert_close(candidate, reference, rtol=0, atol=0)
                samples[variant].append(float(cycles.float().median().item()))

        empty = statistics.median(samples[0]) if 0 in samples else 0.0
        label = "safe-exhaustive" if exhaustive else "safe-model-like"
        print(f"input={label} blocks={args.blocks} rounds={args.rounds} empty={empty:.1f}")
        for variant in args.variants:
            center = statistics.median(samples[variant])
            net = center - empty
            print(
                f"{variant:2d} {VARIANTS[variant]:34s} raw={center:8.1f} "
                f"net={net:8.1f} rounds={[round(value, 1) for value in samples[variant]]}"
            )


if __name__ == "__main__":
    main()
