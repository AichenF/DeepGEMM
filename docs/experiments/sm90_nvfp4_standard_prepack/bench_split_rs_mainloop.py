"""Screen a seven-stage BN128 NVFP4 RS-WGMMA mainloop on SM90.

This is an isolated experiment.  It models the two math warpgroups used by the
optimized W8A8 split kernel and compares shared-source W8A8 with lane-native
NVFP4 register-source schedules.  The Shawn two-seed variant keeps the same
1280-byte fragment footprint but replaces four UE4M3 scale bytes plus padding
with eight E4M3 seed bytes, so its decoder needs no shared LUT reads.  It does
not expose a production API.
"""

import argparse
import os
import statistics

import torch
from torch.utils.cpp_extension import load_inline


REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
os.environ.setdefault("TORCH_CUDA_ARCH_LIST", "9.0a")


CUDA_SRC = r"""
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <c10/cuda/CUDAException.h>

#include <cute/algorithm/cooperative_gemm.hpp>
#include <deep_gemm/common/math.cuh>
#include <deep_gemm/mma/sm90.cuh>
#include <deep_gemm/ptx/wgmma.cuh>
#include <deep_gemm/quantization/nvfp4_dequant.cuh>

namespace {

constexpr int kThreads = 384;
constexpr int kMathThreadOffset = 128;
constexpr int kMathThreads = 256;
constexpr int kStages = 7;
constexpr int kBlockK = 128;
constexpr int kActStageBytes = 64 * kBlockK;
constexpr int kW8StageBytes = 128 * kBlockK;
constexpr int kRSFragmentBytes = 1280;
constexpr int kRSHalfStageBytes = 4 * kRSFragmentBytes;
constexpr int kRSStageBytes = 2 * kRSHalfStageBytes;
constexpr int kDynamicSmemBytes = 176 * 1024;

template <int N_, typename MMA>
struct FP8MMARS {
    template <size_t... Idx>
    __device__ __forceinline__ static void call_fma_impl(
            const uint4& a, const uint64_t& desc_b, float* d,
            const bool scale_d, cute::index_sequence<Idx...>) {
        using namespace cute::SM90::GMMA;
        MMA::fma(a.x, a.y, a.z, a.w, desc_b, d[Idx]...,
                 scale_d ? ScaleOut::One : ScaleOut::Zero);
    }

    __device__ __forceinline__ static void wgmma(
            const uint4& a, const uint64_t& desc_b, float* d,
            const bool scale_d) {
        call_fma_impl(a, desc_b, d, scale_d,
                      cute::make_index_sequence<N_ / 2>{});
    }

    static constexpr int K = 32;
    static constexpr int kNumAccum = N_ / 2;
};

template <int N>
struct FP8MMARSSelector {
    static constexpr auto select_mma() {
        using namespace cute::SM90::GMMA;
        if constexpr (N == 8) return MMA_64x8x32_F32E4M3E4M3_RS_TN();
        if constexpr (N == 16) return MMA_64x16x32_F32E4M3E4M3_RS_TN();
        if constexpr (N == 24) return MMA_64x24x32_F32E4M3E4M3_RS_TN();
        if constexpr (N == 32) return MMA_64x32x32_F32E4M3E4M3_RS_TN();
        if constexpr (N == 64) return MMA_64x64x32_F32E4M3E4M3_RS_TN();
        if constexpr (N == 128) return MMA_64x128x32_F32E4M3E4M3_RS_TN();
    }

    using type = FP8MMARS<N, decltype(select_mma())>;
};

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

__device__ __forceinline__ uint4 decode_a_fragment(
        const uint8_t* __restrict__ fragment,
        const uint32_t thread_idx_in_wg,
        const uint2* __restrict__ lut_smem) {
    const uint2 q = *reinterpret_cast<const uint2*>(
        fragment + thread_idx_in_wg * 8u);
    const uint32_t scale_word = *reinterpret_cast<const uint32_t*>(
        fragment + 1024u + (thread_idx_in_wg >> 2u) * 4u);
    const uint2 lut0 = lut_smem[(scale_word >> 0) & 0x7fu];
    const uint2 lut1 = lut_smem[(scale_word >> 8) & 0x7fu];
    const uint2 lut2 = lut_smem[(scale_word >> 16) & 0x7fu];
    const uint2 lut3 = lut_smem[(scale_word >> 24) & 0x7fu];
    const uint2 out01 = decode_braided_groups(q.x, lut0, lut1);
    const uint2 out23 = decode_braided_groups(q.y, lut2, lut3);
    return make_uint4(out01.x, out01.y, out23.x, out23.y);
}

__device__ __forceinline__ uint2 decode_two_seed_groups(
        const uint32_t packed_e2m1, const uint32_t base_codes) {
    const uint32_t buffer10 =
        __byte_perm(base_codes, 0u, 0x1004) + 0x00080000u;
    const uint32_t buffer20 =
        __byte_perm(base_codes, base_codes, 0x1010) + 0x10180810u;
    const uint32_t buffer11 =
        __byte_perm(base_codes, 0u, 0x3224) + 0x00080000u;
    const uint32_t buffer21 =
        __byte_perm(base_codes, base_codes, 0x3232) + 0x10180810u;
    const uint32_t magnitudes0 =
        __byte_perm(buffer10, buffer20, packed_e2m1);
    const uint32_t magnitudes1 =
        __byte_perm(buffer11, buffer21, packed_e2m1 >> 16);
    uint32_t out0;
    uint32_t out1;
    // The lane-native benchmark stores signs as [group1_i, group0_i] in
    // each byte, opposite to Humming mode-2.  Keep Shawn's magnitude
    // construction but apply the sign masks for this retained transport.
    asm("lop3.b32 %0, %1, %2, 0x80808080, 0xf8;"
        : "=r"(out0) : "r"(magnitudes0), "r"(packed_e2m1));
    asm("lop3.b32 %0, %1, %2, 0x80808080, 0xf8;"
        : "=r"(out1) : "r"(magnitudes1), "r"(packed_e2m1 << 4));
    return make_uint2(out0, out1);
}

__device__ __forceinline__ uint4 decode_two_seed_a_fragment(
        const uint8_t* __restrict__ fragment,
        const uint32_t thread_idx_in_wg) {
    const uint2 q = *reinterpret_cast<const uint2*>(
        fragment + thread_idx_in_wg * 8u);
    const uint2 base_codes = *reinterpret_cast<const uint2*>(
        fragment + 1024u + (thread_idx_in_wg >> 2u) * 8u);
    const uint2 out01 = decode_two_seed_groups(q.x, base_codes.x);
    const uint2 out23 = decode_two_seed_groups(q.y, base_codes.y);
    return make_uint4(out01.x, out01.y, out23.x, out23.y);
}

__device__ __forceinline__ void fence_a_fragment(uint4& a) {
    asm volatile("" : "+r"(a.x), "+r"(a.y), "+r"(a.z), "+r"(a.w) :: "memory");
}

template <int kVariant, int N>
__global__ __launch_bounds__(kThreads, 1) void bench_kernel(
        const uint8_t* __restrict__ input_acts,
        const uint8_t* __restrict__ input_w8,
        const uint8_t* __restrict__ input_rs,
        const uint8_t* __restrict__ input_rs_two_seed,
        int64_t* __restrict__ cycles,
        float* __restrict__ witnesses,
        int repeats) {
    extern __shared__ __align__(1024) uint8_t smem[];
    uint8_t* smem_acts = smem;
    constexpr int kActBytes = kStages * kActStageBytes;
    constexpr int kWeightStageBytes = kVariant == 1 ? kW8StageBytes : kRSStageBytes;
    constexpr int kWeightBytes = kStages * kWeightStageBytes;
    uint8_t* smem_weights = smem_acts + kActBytes;
    uint2* smem_lut = reinterpret_cast<uint2*>(smem_weights + kWeightBytes);

    for (int i = threadIdx.x; i < kActBytes; i += kThreads)
        smem_acts[i] = input_acts[i];
    const uint8_t* input_weights = kVariant == 1 ? input_w8 :
        (kVariant >= 5 ? input_rs_two_seed : input_rs);
    for (int i = threadIdx.x; i < kWeightBytes; i += kThreads)
        smem_weights[i] = input_weights[i];
    if (threadIdx.x < 128)
        smem_lut[threadIdx.x] =
            deep_gemm::nvfp4::kE2M1AndUe4m3ToFp8Lut[threadIdx.x];
    __syncthreads();

    if (threadIdx.x < kMathThreadOffset)
        return;

    const uint32_t math_tid = threadIdx.x - kMathThreadOffset;
    const uint32_t wg_idx = math_tid >> 7u;
    const uint32_t tid_in_wg = math_tid & 127u;
    using SSMMA = typename deep_gemm::mma::sm90::FP8MMASelector<N>::type;
    using RSMMA = typename FP8MMARSSelector<N>::type;
    constexpr int kNumAccum = N / 2;
    float accum[kNumAccum] = {};

    asm volatile("bar.sync 1, %0;" : : "n"(kMathThreads));
    uint64_t start = 0;
    if (tid_in_wg == 0)
        start = clock64();
    asm volatile("bar.sync 1, %0;" : : "n"(kMathThreads));

    for (int repeat = 0; repeat < repeats; ++repeat) {
        const uint32_t stage = static_cast<uint32_t>(repeat % kStages);
        #pragma unroll
        for (int i = 0; i < kNumAccum; ++i)
            deep_gemm::ptx::warpgroup_fence_operand(accum[i]);

        if constexpr (kVariant == 1) {
            deep_gemm::ptx::warpgroup_arrive();
            #pragma unroll
            for (uint32_t k = 0; k < 4; ++k) {
                auto desc_a = deep_gemm::mma::sm90::make_smem_desc(
                    smem_weights + stage * kW8StageBytes +
                    wg_idx * 64u * kBlockK + k * 32u, 1);
                auto desc_b = deep_gemm::mma::sm90::make_smem_desc(
                    smem_acts + stage * kActStageBytes + k * 32u, 1);
                SSMMA::wgmma(desc_a, desc_b, accum, k != 0u);
            }
        } else if constexpr (kVariant == 2) {
            uint4 fragments[4];
            #pragma unroll
            for (uint32_t k = 0; k < 4; ++k) {
                const uint8_t* fragment = smem_weights +
                    stage * kRSStageBytes + wg_idx * kRSHalfStageBytes +
                    k * kRSFragmentBytes;
                fragments[k] = decode_a_fragment(fragment, tid_in_wg, smem_lut);
                fence_a_fragment(fragments[k]);
            }
            deep_gemm::ptx::warpgroup_arrive();
            #pragma unroll
            for (uint32_t k = 0; k < 4; ++k) {
                auto desc_b = deep_gemm::mma::sm90::make_smem_desc(
                    smem_acts + stage * kActStageBytes + k * 32u, 1);
                RSMMA::wgmma(fragments[k], desc_b, accum, k != 0u);
            }
        } else if constexpr (kVariant == 3) {
            #pragma unroll
            for (uint32_t k = 0; k < 4; ++k) {
                const uint8_t* fragment = smem_weights +
                    stage * kRSStageBytes + wg_idx * kRSHalfStageBytes +
                    k * kRSFragmentBytes;
                uint4 a = decode_a_fragment(fragment, tid_in_wg, smem_lut);
                fence_a_fragment(a);
                deep_gemm::ptx::warpgroup_arrive();
                auto desc_b = deep_gemm::mma::sm90::make_smem_desc(
                    smem_acts + stage * kActStageBytes + k * 32u, 1);
                RSMMA::wgmma(a, desc_b, accum, k != 0u);
            }
        } else if constexpr (kVariant == 4) {
            uint4 fragments[2];
            #pragma unroll
            for (uint32_t pair = 0; pair < 2; ++pair) {
                #pragma unroll
                for (uint32_t i = 0; i < 2; ++i) {
                    const uint32_t k = pair * 2u + i;
                    const uint8_t* fragment = smem_weights +
                        stage * kRSStageBytes + wg_idx * kRSHalfStageBytes +
                        k * kRSFragmentBytes;
                    fragments[i] = decode_a_fragment(fragment, tid_in_wg, smem_lut);
                    fence_a_fragment(fragments[i]);
                }
                deep_gemm::ptx::warpgroup_arrive();
                #pragma unroll
                for (uint32_t i = 0; i < 2; ++i) {
                    const uint32_t k = pair * 2u + i;
                    auto desc_b = deep_gemm::mma::sm90::make_smem_desc(
                        smem_acts + stage * kActStageBytes + k * 32u, 1);
                    RSMMA::wgmma(fragments[i], desc_b, accum, k != 0u);
                }
            }
        } else if constexpr (kVariant == 5) {
            uint4 fragments[4];
            #pragma unroll
            for (uint32_t k = 0; k < 4; ++k) {
                const uint8_t* fragment = smem_weights +
                    stage * kRSStageBytes + wg_idx * kRSHalfStageBytes +
                    k * kRSFragmentBytes;
                fragments[k] = decode_two_seed_a_fragment(
                    fragment, tid_in_wg);
                fence_a_fragment(fragments[k]);
            }
            deep_gemm::ptx::warpgroup_arrive();
            #pragma unroll
            for (uint32_t k = 0; k < 4; ++k) {
                auto desc_b = deep_gemm::mma::sm90::make_smem_desc(
                    smem_acts + stage * kActStageBytes + k * 32u, 1);
                RSMMA::wgmma(fragments[k], desc_b, accum, k != 0u);
            }
        } else if constexpr (kVariant == 6) {
            uint4 fragments[2];
            #pragma unroll
            for (uint32_t pair = 0; pair < 2; ++pair) {
                #pragma unroll
                for (uint32_t i = 0; i < 2; ++i) {
                    const uint32_t k = pair * 2u + i;
                    const uint8_t* fragment = smem_weights +
                        stage * kRSStageBytes + wg_idx * kRSHalfStageBytes +
                        k * kRSFragmentBytes;
                    fragments[i] = decode_two_seed_a_fragment(
                        fragment, tid_in_wg);
                    fence_a_fragment(fragments[i]);
                }
                deep_gemm::ptx::warpgroup_arrive();
                #pragma unroll
                for (uint32_t i = 0; i < 2; ++i) {
                    const uint32_t k = pair * 2u + i;
                    auto desc_b = deep_gemm::mma::sm90::make_smem_desc(
                        smem_acts + stage * kActStageBytes + k * 32u, 1);
                    RSMMA::wgmma(fragments[i], desc_b, accum, k != 0u);
                }
            }
        } else {
            #pragma unroll
            for (uint32_t k = 0; k < 4; ++k) {
                const uint8_t* fragment = smem_weights +
                    stage * kRSStageBytes + wg_idx * kRSHalfStageBytes +
                    k * kRSFragmentBytes;
                uint4 a = decode_two_seed_a_fragment(fragment, tid_in_wg);
                fence_a_fragment(a);
                deep_gemm::ptx::warpgroup_arrive();
                auto desc_b = deep_gemm::mma::sm90::make_smem_desc(
                    smem_acts + stage * kActStageBytes + k * 32u, 1);
                RSMMA::wgmma(a, desc_b, accum, k != 0u);
            }
        }

        deep_gemm::ptx::warpgroup_commit_batch();
        #pragma unroll
        for (int i = 0; i < kNumAccum; ++i)
            deep_gemm::ptx::warpgroup_fence_operand(accum[i]);
        deep_gemm::ptx::warpgroup_wait<0>();
    }

    asm volatile("bar.sync 1, %0;" : : "n"(kMathThreads));
    if (tid_in_wg == 0)
        cycles[blockIdx.x * 2u + wg_idx] =
            static_cast<int64_t>(clock64() - start);

    if (blockIdx.x == 0) {
        const uint32_t base = math_tid * kNumAccum;
        #pragma unroll
        for (int i = 0; i < kNumAccum; ++i)
            witnesses[base + i] = accum[i];
    }
}

template <int kVariant, int N>
void launch(const torch::Tensor& acts, const torch::Tensor& w8,
            const torch::Tensor& rs, const torch::Tensor& rs_two_seed,
            torch::Tensor& cycles,
            torch::Tensor& witnesses, int repeats) {
    C10_CUDA_CHECK(cudaFuncSetAttribute(
        bench_kernel<kVariant, N>, cudaFuncAttributeMaxDynamicSharedMemorySize,
        kDynamicSmemBytes));
    bench_kernel<kVariant, N><<<cycles.size(0), kThreads, kDynamicSmemBytes>>>(
        acts.data_ptr<uint8_t>(), w8.data_ptr<uint8_t>(), rs.data_ptr<uint8_t>(),
        rs_two_seed.data_ptr<uint8_t>(),
        cycles.data_ptr<int64_t>(), witnesses.data_ptr<float>(), repeats);
}

template <int N>
void dispatch_variant(int variant, const torch::Tensor& acts,
                      const torch::Tensor& w8, const torch::Tensor& rs,
                      const torch::Tensor& rs_two_seed,
                      torch::Tensor& cycles, torch::Tensor& witnesses,
                      int repeats) {
    if (variant == 1) launch<1, N>(acts, w8, rs, rs_two_seed, cycles, witnesses, repeats);
    else if (variant == 2) launch<2, N>(acts, w8, rs, rs_two_seed, cycles, witnesses, repeats);
    else if (variant == 3) launch<3, N>(acts, w8, rs, rs_two_seed, cycles, witnesses, repeats);
    else if (variant == 4) launch<4, N>(acts, w8, rs, rs_two_seed, cycles, witnesses, repeats);
    else if (variant == 5) launch<5, N>(acts, w8, rs, rs_two_seed, cycles, witnesses, repeats);
    else if (variant == 6) launch<6, N>(acts, w8, rs, rs_two_seed, cycles, witnesses, repeats);
    else if (variant == 7) launch<7, N>(acts, w8, rs, rs_two_seed, cycles, witnesses, repeats);
    else TORCH_CHECK(false, "unknown variant");
}

}  // namespace

std::vector<torch::Tensor> run_split_rs_mainloop(
        torch::Tensor acts, torch::Tensor w8, torch::Tensor rs,
        torch::Tensor rs_two_seed,
        int64_t variant, int64_t n, int64_t blocks, int64_t repeats) {
    TORCH_CHECK(acts.is_cuda() && acts.scalar_type() == torch::kUInt8);
    TORCH_CHECK(w8.is_cuda() && w8.scalar_type() == torch::kUInt8);
    TORCH_CHECK(rs.is_cuda() && rs.scalar_type() == torch::kUInt8);
    TORCH_CHECK(rs_two_seed.is_cuda() &&
                rs_two_seed.scalar_type() == torch::kUInt8);
    TORCH_CHECK(acts.numel() == kStages * kActStageBytes);
    TORCH_CHECK(w8.numel() == kStages * kW8StageBytes);
    TORCH_CHECK(rs.numel() == kStages * kRSStageBytes);
    TORCH_CHECK(rs_two_seed.numel() == kStages * kRSStageBytes);
    auto cycles = torch::empty({blocks, 2}, acts.options().dtype(torch::kInt64));
    auto witnesses = torch::empty({256, n / 2}, acts.options().dtype(torch::kFloat32));
    if (n == 8) dispatch_variant<8>(variant, acts, w8, rs, rs_two_seed, cycles, witnesses, repeats);
    else if (n == 16) dispatch_variant<16>(variant, acts, w8, rs, rs_two_seed, cycles, witnesses, repeats);
    else if (n == 24) dispatch_variant<24>(variant, acts, w8, rs, rs_two_seed, cycles, witnesses, repeats);
    else if (n == 32) dispatch_variant<32>(variant, acts, w8, rs, rs_two_seed, cycles, witnesses, repeats);
    else if (n == 64) dispatch_variant<64>(variant, acts, w8, rs, rs_two_seed, cycles, witnesses, repeats);
    else if (n == 128) dispatch_variant<128>(variant, acts, w8, rs, rs_two_seed, cycles, witnesses, repeats);
    else TORCH_CHECK(false, "unsupported N");
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return {cycles, witnesses};
}
"""


VARIANTS = {
    1: "w8/shared-source",
    2: "nvfp4/rs-predecode",
    3: "nvfp4/rs-interleaved",
    4: "nvfp4/rs-pairwise",
    5: "nvfp4/shawn-two-seed-predecode",
    6: "nvfp4/shawn-two-seed-pairwise",
    7: "nvfp4/shawn-two-seed-interleaved",
}


def _lane_coordinates(tid: int) -> tuple[int, int, int]:
    t0 = tid % 4
    t1 = (tid // 4) % 8
    t2 = tid // 32
    return t1 + 16 * t2, t1 + 16 * t2 + 8, 4 * t0


def _pack_braided_half(
    codes: torch.Tensor,
    scales: torch.Tensor,
    two_seed: bool = False,
) -> torch.Tensor:
    result = torch.zeros((4, 1280), dtype=torch.uint8)
    for k32 in range(4):
        k_base = k32 * 32
        for tid in range(128):
            row0, row1, lane_k = _lane_coordinates(tid)
            groups = torch.stack(
                [codes[row0, k_base + lane_k : k_base + lane_k + 4],
                 codes[row1, k_base + lane_k : k_base + lane_k + 4],
                 codes[row0, k_base + lane_k + 16 : k_base + lane_k + 20],
                 codes[row1, k_base + lane_k + 16 : k_base + lane_k + 20]]
            )
            for pair_idx in range(2):
                values = torch.cat([groups[pair_idx * 2], groups[pair_idx * 2 + 1]]).to(torch.int32)
                magnitudes = values & 0x7
                signs = values >> 3
                sign_order = torch.stack(
                    [signs[4], signs[0], signs[5], signs[1],
                     signs[6], signs[2], signs[7], signs[3]]
                )
                nibbles = magnitudes | (sign_order << 3)
                word = sum(int(nibbles[i].item()) << (4 * i) for i in range(8))
                offset = tid * 8 + pair_idx * 4
                for byte_idx in range(4):
                    result[k32, offset + byte_idx] = (word >> (8 * byte_idx)) & 0xFF

        for group in range(32):
            row0, row1, _ = _lane_coordinates(group * 4)
            scale_values = [
                scales[row0, k32 * 2], scales[row1, k32 * 2],
                scales[row0, k32 * 2 + 1], scales[row1, k32 * 2 + 1],
            ]
            if two_seed:
                scale_tensor = torch.tensor(scale_values, dtype=torch.uint8)
                exp_bits = (scale_tensor.to(torch.int32) >> 3) - 7
                mant = (scale_tensor.to(torch.int32) & 7).to(torch.float32)
                scale_fp32 = (1.0 + mant * 0.125) * torch.exp2(
                    exp_bits.to(torch.float32)
                )
                seed_half = (scale_fp32 * 0.5).to(
                    torch.float8_e4m3fn
                ).view(torch.uint8)
                seed_one_half = (scale_fp32 * 1.5).to(
                    torch.float8_e4m3fn
                ).view(torch.uint8)
                metadata = torch.stack(
                    (seed_half, seed_one_half), dim=-1
                ).reshape(-1)
                result[k32, 1024 + group * 8 : 1024 + group * 8 + 8] = metadata
            else:
                result[k32, 1024 + group * 4 : 1024 + group * 4 + 4] = torch.tensor(
                    scale_values, dtype=torch.uint8
                )
    return result.view(-1)


def make_inputs() -> tuple[
    torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor
]:
    generator = torch.Generator().manual_seed(1234)
    acts = torch.randint(0, 127, (7, 64, 128), dtype=torch.uint8, generator=generator)
    w8 = torch.randint(0, 127, (7, 128, 128), dtype=torch.uint8, generator=generator)
    rs_stages = []
    rs_two_seed_stages = []
    for _ in range(7):
        codes = torch.randint(0, 16, (128, 128), dtype=torch.uint8, generator=generator)
        # Keep R and 6R in the normal finite E4M3 range required by the
        # exponent-shift construction used by Shawn's E91 decoder.
        scales = torch.randint(24, 97, (128, 8), dtype=torch.uint8, generator=generator)
        rs_stages.append(torch.cat([
            _pack_braided_half(codes[:64], scales[:64]),
            _pack_braided_half(codes[64:], scales[64:]),
        ]))
        rs_two_seed_stages.append(torch.cat([
            _pack_braided_half(codes[:64], scales[:64], two_seed=True),
            _pack_braided_half(codes[64:], scales[64:], two_seed=True),
        ]))
    rs = torch.stack(rs_stages)
    rs_two_seed = torch.stack(rs_two_seed_stages)
    return (
        acts.contiguous().view(-1).cuda(),
        w8.contiguous().view(-1).cuda(),
        rs.contiguous().view(-1).cuda(),
        rs_two_seed.contiguous().view(-1).cuda(),
    )


def load_extension():
    cpp_src = (
        "std::vector<torch::Tensor> run_split_rs_mainloop("
        "torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, "
        "int64_t, int64_t, int64_t, int64_t);"
    )
    return load_inline(
        name="deepgemm_sm90_nvfp4_split_rs_mainloop_bench",
        cpp_sources=cpp_src,
        cuda_sources=CUDA_SRC,
        functions=["run_split_rs_mainloop"],
        extra_include_paths=[
            os.path.join(REPO_ROOT, "deep_gemm", "include"),
            os.path.join(REPO_ROOT, "third-party", "cutlass", "include"),
        ],
        extra_cuda_cflags=["-O3", "-lineinfo", "--expt-relaxed-constexpr"],
        verbose=False,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--blocks", type=int, default=78)
    parser.add_argument("--repeats", type=int, default=28)
    parser.add_argument("--rounds", type=int, default=11)
    parser.add_argument(
        "--n", type=int, nargs="+", default=[8, 16, 24, 32, 64, 128]
    )
    parser.add_argument(
        "--variants", type=int, nargs="+", default=list(VARIANTS)
    )
    args = parser.parse_args()
    selected_variants = {
        variant: VARIANTS[variant] for variant in args.variants
    }

    assert torch.cuda.get_device_capability()[0] == 9
    acts, w8, rs, rs_two_seed = make_inputs()
    ext = load_extension()

    for n in args.n:
        reference = None
        samples = {variant: [] for variant in selected_variants}
        for round_idx in range(args.rounds):
            order = list(selected_variants)
            if round_idx % 2:
                order.reverse()
            for variant in order:
                cycles, witnesses = ext.run_split_rs_mainloop(
                    acts, w8, rs, rs_two_seed,
                    variant, n, args.blocks, args.repeats
                )
                torch.cuda.synchronize()
                if variant in (2, 3, 4, 5, 6, 7):
                    candidate = witnesses.cpu()
                    if reference is None:
                        reference = candidate
                    else:
                        torch.testing.assert_close(candidate, reference, rtol=0, atol=0)
                per_stage = cycles.float().amax(dim=1).median().item() / args.repeats
                samples[variant].append(float(per_stage))

        print(f"N={n} blocks={args.blocks} repeats={args.repeats} rounds={args.rounds}")
        for variant, name in selected_variants.items():
            center = statistics.median(samples[variant])
            print(
                f"{variant} {name:28s} cycles/stage={center:8.1f} "
                f"rounds={[round(value, 1) for value in samples[variant]]}"
            )


if __name__ == "__main__":
    main()
