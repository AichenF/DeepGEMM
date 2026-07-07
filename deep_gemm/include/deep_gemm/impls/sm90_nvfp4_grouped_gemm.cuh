#pragma once

#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wunknown-attributes"

#include <cuda_bf16.h>
#include <cuda_fp8.h>

#include <cute/numeric/integer_sequence.hpp>
#include <cutlass/arch/barrier.h>
#include <cutlass/arch/reg_reconfig.h>

#include <deep_gemm/comm/barrier.cuh>
#include <deep_gemm/common/exception.cuh>
#include <deep_gemm/common/math.cuh>
#include <deep_gemm/common/tma_copy.cuh>
#include <deep_gemm/mma/sm90.cuh>
#include <deep_gemm/ptx/ld_st.cuh>
#include <deep_gemm/ptx/utils.cuh>
#include <deep_gemm/ptx/wgmma.cuh>
#include <deep_gemm/quantization/nvfp4_dequant.cuh>

namespace deep_gemm {

namespace detail {

CUTLASS_DEVICE void mma_sync_m16n8k32_fp8(
        float (&d)[4], const uint32_t (&a)[4], const uint32_t (&b)[2]) {
    asm volatile(
        "mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 "
        "{%0, %1, %2, %3}, "
        "{%4, %5, %6, %7}, "
        "{%8, %9}, "
        "{%0, %1, %2, %3};\n"
        : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3])
        : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]),
          "r"(b[0]), "r"(b[1]));
}

CUTLASS_DEVICE uint32_t dequant_nvfp4x4_to_fp8(
        const uint32_t packed16, const uint2& lut) {
    const uint32_t magnitudes = packed16 & 0x7777u;
    const uint32_t low_signs = packed16 & 0x88u;
    const uint32_t high_signs = (packed16 >> 8u) & 0x88u;
    const uint32_t sign_mask =
        ((low_signs * 0x110u) & 0x00008080u) |
        (((high_signs * 0x110u) & 0x00008080u) << 16u);
    return nvfp4::byte_perm_unchecked(
        lut.x, lut.y, magnitudes) ^ sign_mask;
}

CUTLASS_DEVICE uint2 dequant_nvfp4x8_bitwoven_to_fp8(
        const uint32_t packed, const uint2& lut) {
    const uint32_t first = nvfp4::byte_perm_unchecked(
        lut.x, lut.y, packed & 0x7777u) ^
        (packed & 0x80808080u);
    const uint32_t second = nvfp4::byte_perm_unchecked(
        lut.x, lut.y, (packed >> 16u) & 0x7777u) ^
        ((packed << 4u) & 0x80808080u);
    return make_uint2(first, second);
}

}  // namespace detail

// Low-token path.  Transpose the logical GEMM so mma.sync's N8 dimension is
// the token dimension: W[N16, K32] @ A^T[K32, M8].  Each warp owns N16 for
// one expert and reuses every decoded weight fragment across all M8 groups.
template <bool kPretransformedWeights>
CUTLASS_GLOBAL __launch_bounds__(128, 4) void
sm90_nvfp4_grouped_gemm_small_m_mma_impl(
        __nv_bfloat16* __restrict__ d,
        const __nv_fp8_e4m3* __restrict__ a,
        const float* __restrict__ a_scale,
        const uint8_t* __restrict__ w_packed,
        const uint8_t* __restrict__ block_scale,
        const float* __restrict__ global_scale,
        const int* __restrict__ offsets,
        uint32_t shape_m, uint32_t shape_n, uint32_t shape_k,
        uint32_t num_groups) {
#if (defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900)) || defined(__CLION_IDE__)
    constexpr uint32_t kBlockN = 64;
    constexpr uint32_t kWarpN = 16;
    constexpr uint32_t kMmaM = 16;
    constexpr uint32_t kMmaN = 8;
    constexpr uint32_t kMmaK = 32;
    constexpr uint32_t kMaxTokenGroups = 16;

    extern __shared__ __align__(16) uint8_t smem_buffer[];
    auto* smem_lut = reinterpret_cast<uint2*>(smem_buffer);
    const uint32_t tid = threadIdx.x;
    const uint32_t lane = tid & 31u;
    const uint32_t warp = tid >> 5u;
    const uint32_t group_id = lane >> 2u;
    const uint32_t lane_in_group = lane & 3u;

    if (tid < 64u) {
        reinterpret_cast<uint4*>(smem_lut)[tid] =
            reinterpret_cast<const uint4*>(
                nvfp4::kE2M1AndUe4m3ToFp8Lut)[tid];
    }
    __syncthreads();

    const uint32_t n_tiles = shape_n / kBlockN;
    const uint32_t tile_idx = blockIdx.x;
    const uint32_t expert = tile_idx / n_tiles;
    if (expert >= num_groups)
        return;
    const uint32_t block_n = (tile_idx - expert * n_tiles) * kBlockN;
    const uint32_t warp_n = block_n + warp * kWarpN;
    const uint32_t expert_begin = static_cast<uint32_t>(offsets[expert]);
    const uint32_t expert_end = static_cast<uint32_t>(offsets[expert + 1]);
    const uint32_t expert_tokens = expert_end - expert_begin;
    if (expert_tokens == 0u)
        return;
    const uint32_t token_groups =
        math::ceil_div(expert_tokens, kMmaN);
    DG_DEVICE_ASSERT(token_groups <= kMaxTokenGroups);

    const uint32_t packed_k = shape_k / 2u;
    const uint32_t scale_k = shape_k / 16u;
    float accum[kMaxTokenGroups][4] = {};

    auto load_weight_fragment = [&] (
            const uint32_t weight_row, const uint32_t scale_group,
            const uint32_t packed_byte_in_group) {
        uint32_t scale_code = 0u;
        const uint8_t* packed_ptr;
        if constexpr (kPretransformedWeights) {
            const size_t scale_offset =
                (static_cast<size_t>(expert) * scale_k + scale_group) *
                    shape_n + weight_row;
            packed_ptr = w_packed + scale_offset * 8u +
                packed_byte_in_group;
            if (lane_in_group == 0u)
                scale_code = block_scale[scale_offset];
        } else {
            const size_t row_offset =
                static_cast<size_t>(expert) * shape_n + weight_row;
            packed_ptr = w_packed + row_offset * packed_k +
                scale_group * 8u + packed_byte_in_group;
            if (lane_in_group == 0u)
                scale_code = block_scale[row_offset * scale_k + scale_group];
        }
        scale_code = __shfl_sync(0xffffffffu, scale_code, 0, 4) & 0x7fu;
        const uint32_t packed =
            *reinterpret_cast<const uint16_t*>(packed_ptr);
        const uint2 decoded =
            nvfp4::dequant_nvfp4_to_fp8_pair_with_lut(
                packed, smem_lut[scale_code]);
        return __byte_perm(decoded.y, decoded.x, 0x5140u);
    };

    for (uint32_t k_begin = 0; k_begin < shape_k; k_begin += kMmaK) {
        const uint32_t row0 = warp_n + group_id;
        const uint32_t row1 = row0 + 8u;
        const uint32_t scale_group0 = k_begin / 16u;
        const uint32_t scale_group1 = scale_group0 + 1u;
        const uint32_t packed_byte = lane_in_group * 2u;
        const uint32_t weight_frag[4] = {
            load_weight_fragment(row0, scale_group0, packed_byte),
            load_weight_fragment(row1, scale_group0, packed_byte),
            load_weight_fragment(row0, scale_group1, packed_byte),
            load_weight_fragment(row1, scale_group1, packed_byte),
        };

        #pragma unroll
        for (uint32_t token_group = 0;
             token_group < kMaxTokenGroups; ++token_group) {
            if (token_group >= token_groups)
                break;
            const uint32_t token =
                expert_begin + token_group * kMmaN + group_id;
            uint32_t activation_frag[2] = {0u, 0u};
            if (token < expert_end) {
                const auto* activation_row = reinterpret_cast<const uint8_t*>(
                    a + static_cast<size_t>(token) * shape_k + k_begin);
                activation_frag[0] = *reinterpret_cast<const uint32_t*>(
                    activation_row + lane_in_group * 4u);
                activation_frag[1] = *reinterpret_cast<const uint32_t*>(
                    activation_row + 16u + lane_in_group * 4u);
            }
            detail::mma_sync_m16n8k32_fp8(
                accum[token_group], weight_frag, activation_frag);
        }
    }

    const uint32_t output_n0 = warp_n + group_id;
    const uint32_t output_n1 = output_n0 + 8u;
    const float expert_scale = global_scale[expert] * 8.0f;
    #pragma unroll
    for (uint32_t token_group = 0;
         token_group < kMaxTokenGroups; ++token_group) {
        if (token_group >= token_groups)
            break;
        const uint32_t token0 =
            expert_begin + token_group * kMmaN + lane_in_group * 2u;
        const uint32_t token1 = token0 + 1u;
        if (token0 < expert_end) {
            const float scale = a_scale[token0] * expert_scale;
            d[static_cast<size_t>(token0) * shape_n + output_n0] =
                __float2bfloat16_rn(accum[token_group][0] * scale);
            d[static_cast<size_t>(token0) * shape_n + output_n1] =
                __float2bfloat16_rn(accum[token_group][2] * scale);
        }
        if (token1 < expert_end) {
            const float scale = a_scale[token1] * expert_scale;
            d[static_cast<size_t>(token1) * shape_n + output_n0] =
                __float2bfloat16_rn(accum[token_group][1] * scale);
            d[static_cast<size_t>(token1) * shape_n + output_n1] =
                __float2bfloat16_rn(accum[token_group][3] * scale);
        }
    }
    (void)shape_m;
    (void)shape_n;
    (void)shape_k;
    (void)num_groups;
#else
    if (blockIdx.x == 0 && threadIdx.x == 0)
        DG_DEVICE_ASSERT(false && "This kernel only supports sm_90a");
#endif
}

// Hopper-native low-token path. WGMMA M64 covers output columns while N8
// covers tokens. A producer warpgroup decodes one N64 weight tile and stages
// every token row once; the consumer reuses that tile for all M8 groups.
template <bool kPretransformedWeights, bool kExactByteFusedWeights>
CUTLASS_GLOBAL __launch_bounds__(384, 2) void
sm90_nvfp4_grouped_gemm_small_m_wgmma_impl(
        __nv_bfloat16* __restrict__ d,
        const __nv_fp8_e4m3* __restrict__ a,
        const float* __restrict__ a_scale,
        const uint8_t* __restrict__ w_packed,
        const uint8_t* __restrict__ block_scale,
        const float* __restrict__ global_scale,
        const int* __restrict__ offsets,
        uint32_t shape_m, uint32_t shape_n, uint32_t shape_k,
        uint32_t num_groups) {
#if (defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900)) || defined(__CLION_IDE__)
    constexpr uint32_t kBlockN = 64;
    constexpr uint32_t kBlockK = 128;
    constexpr uint32_t kTokenGroup = 8;
    constexpr uint32_t kGroupsPerPass = 4;
    constexpr uint32_t kTokensPerPass = kTokenGroup * kGroupsPerPass;
    constexpr uint32_t kNumStages = 2;
    constexpr uint32_t kWeightBytes = kBlockN * kBlockK;
    constexpr uint32_t kActivationBytes = kTokensPerPass * kBlockK;
    using WGMMA = typename mma::sm90::FP8MMASelector<kTokenGroup>::type;
    using Barrier = cutlass::arch::ClusterTransactionBarrier;
    DG_STATIC_ASSERT(WGMMA::M == kBlockN && WGMMA::N == kTokenGroup &&
                     WGMMA::K == 32 && WGMMA::kNumAccum == 4,
                     "Unexpected small-M WGMMA shape");
    DG_STATIC_ASSERT(!(kPretransformedWeights && kExactByteFusedWeights),
                     "Weight layouts are mutually exclusive");

    extern __shared__ __align__(1024) uint8_t smem_buffer[];
    auto* smem_weight_base =
        reinterpret_cast<__nv_fp8_e4m3*>(smem_buffer);
    auto* smem_activation_base = reinterpret_cast<__nv_fp8_e4m3*>(
        smem_buffer + kNumStages * kWeightBytes);
    auto* smem_lut = reinterpret_cast<uint2*>(
        smem_buffer + kNumStages * (kWeightBytes + kActivationBytes));
    auto* full_barriers = reinterpret_cast<Barrier*>(smem_lut + 128);
    auto* empty_barriers = full_barriers + kNumStages;

    const uint32_t tid = threadIdx.x;
    const uint32_t wg_tid = tid & 127u;
    const uint32_t lane = wg_tid & 31u;
    const uint32_t n_tiles = shape_n / kBlockN;
    const uint32_t tile_idx = blockIdx.x;
    const uint32_t expert = tile_idx / n_tiles;
    if (expert >= num_groups)
        return;
    const uint32_t block_n = (tile_idx - expert * n_tiles) * kBlockN;
    const uint32_t expert_begin = static_cast<uint32_t>(offsets[expert]);
    const uint32_t expert_end = static_cast<uint32_t>(offsets[expert + 1]);
    const uint32_t expert_tokens = expert_end - expert_begin;
    if (expert_tokens == 0u)
        return;
    const uint32_t total_token_groups =
        math::ceil_div(expert_tokens, kTokenGroup);

    if (tid < 64u) {
        reinterpret_cast<uint4*>(smem_lut)[tid] =
            reinterpret_cast<const uint4*>(
                nvfp4::kE2M1AndUe4m3ToFp8Lut)[tid];
    }
    if (tid == 0u) {
        #pragma unroll
        for (uint32_t stage = 0; stage < kNumStages; ++stage) {
            full_barriers[stage].init(2);
            empty_barriers[stage].init(4);
        }
        cutlass::arch::fence_barrier_init();
    }
    __syncthreads();

    uint32_t stage_idx = 0u;
    uint32_t phase = 0u;
    auto advance_pipeline = [&] () {
        stage_idx = stage_idx == kNumStages - 1u ? 0u : stage_idx + 1u;
        phase ^= static_cast<uint32_t>(stage_idx == 0u);
    };

    const uint32_t packed_k = shape_k / 2u;
    const uint32_t scale_k = shape_k / 16u;
    const uint32_t k_blocks = shape_k / kBlockK;

    if (tid >= 128u) {
        const uint32_t producer_group = (tid - 128u) >> 7u;
        const uint32_t producer_tid = tid - 128u;
        for (uint32_t token_group_base = 0u;
             token_group_base < total_token_groups;
             token_group_base += kGroupsPerPass) {
            const uint32_t remaining_groups =
                total_token_groups - token_group_base;
            const uint32_t pass_groups = remaining_groups < kGroupsPerPass
                ? remaining_groups : kGroupsPerPass;
            for (uint32_t k_block = 0; k_block < k_blocks; ++k_block) {
                if (wg_tid == 0u)
                    empty_barriers[stage_idx].wait(phase ^ 1u);
                ptx::sync_aligned(128, 1u + producer_group);

            const uint32_t k_begin = k_block * kBlockK;
            auto* smem_weight =
                smem_weight_base + stage_idx * kWeightBytes;
            auto* smem_activation =
                smem_activation_base + stage_idx * kActivationBytes;

            const uint32_t weight_row =
                (wg_tid & 31u) + producer_group * 32u;
            const uint32_t packed_vector = wg_tid >> 5u;
            const uint32_t k_half = packed_vector >> 1u;
            const uint32_t global_weight = block_n + weight_row;
            const uint32_t row_swizzle = (weight_row & 7u) << 4u;
            uint4 fused_packed = make_uint4(0u, 0u, 0u, 0u);
            uint32_t fused_scales = 0u;
            if constexpr (kExactByteFusedWeights) {
                constexpr uint32_t kPackedRowBytes = kBlockK / 2u;
                constexpr uint32_t kScaleRowBytes = kBlockK / 16u;
                constexpr uint32_t kTileRowBytes =
                    kPackedRowBytes + kScaleRowBytes;
                const size_t tile_offset =
                    (static_cast<size_t>(expert) * k_blocks + k_block) *
                    shape_n * kTileRowBytes;
                const uint8_t* tile_base = w_packed + tile_offset;
                const size_t packed_vector_offset =
                    static_cast<size_t>(packed_vector) * shape_n +
                    global_weight;
                fused_packed = *reinterpret_cast<const uint4*>(
                    tile_base + packed_vector_offset * sizeof(uint4));
                fused_scales = *reinterpret_cast<const uint32_t*>(
                    tile_base + static_cast<size_t>(shape_n) *
                        kPackedRowBytes +
                    (static_cast<size_t>(k_half) * shape_n +
                     global_weight) * sizeof(uint32_t));
            }
            #pragma unroll
            for (uint32_t local_scale = 0; local_scale < 2u;
                 ++local_scale) {
                const uint32_t scale_in_block =
                    packed_vector * 2u + local_scale;
                const uint32_t global_scale_k =
                    k_begin / 16u + scale_in_block;
                uint32_t scale;
                uint32_t packed0;
                uint32_t packed1;
                if constexpr (kExactByteFusedWeights) {
                    scale = (fused_scales >>
                             ((scale_in_block & 3u) * 8u)) & 0xffu;
                    if (local_scale == 0u) {
                        packed0 = fused_packed.x;
                        packed1 = fused_packed.y;
                    } else {
                        packed0 = fused_packed.z;
                        packed1 = fused_packed.w;
                    }
                } else if constexpr (kPretransformedWeights) {
                    const size_t scale_offset =
                        (static_cast<size_t>(expert) * scale_k +
                         global_scale_k) * shape_n + global_weight;
                    const auto* packed_words =
                        reinterpret_cast<const uint32_t*>(
                            w_packed + scale_offset * 8u);
                    packed0 = packed_words[0];
                    packed1 = packed_words[1];
                    scale = block_scale[scale_offset];
                } else {
                    const size_t row_offset =
                        static_cast<size_t>(expert) * shape_n + global_weight;
                    const auto* packed_words =
                        reinterpret_cast<const uint32_t*>(
                            w_packed + row_offset * packed_k +
                            k_begin / 2u + scale_in_block * 8u);
                    packed0 = packed_words[0];
                    packed1 = packed_words[1];
                    scale = block_scale[row_offset * scale_k + global_scale_k];
                }
                const uint2 lut = smem_lut[scale & 0x7fu];
                const uint32_t physical =
                    scale_in_block * 16u ^ row_swizzle;
                *reinterpret_cast<uint4*>(
                    reinterpret_cast<uint8_t*>(
                        smem_weight + weight_row * kBlockK) + physical) =
                    make_uint4(
                        detail::dequant_nvfp4x4_to_fp8(packed0, lut),
                        detail::dequant_nvfp4x4_to_fp8(packed0 >> 16u, lut),
                        detail::dequant_nvfp4x4_to_fp8(packed1, lut),
                        detail::dequant_nvfp4x4_to_fp8(packed1 >> 16u, lut));
            }

            constexpr uint32_t kChunksPerRow = kBlockK / sizeof(uint4);
            const uint32_t activation_chunks =
                pass_groups * kTokenGroup * kChunksPerRow;
            for (uint32_t chunk = producer_tid; chunk < activation_chunks;
                 chunk += 256u) {
                const uint32_t local_token = chunk / kChunksPerRow;
                const uint32_t k_chunk =
                    (chunk % kChunksPerRow) * sizeof(uint4);
                const uint32_t global_token = expert_begin +
                    token_group_base * kTokenGroup + local_token;
                uint4 value = make_uint4(0u, 0u, 0u, 0u);
                if (global_token < expert_end) {
                    value = *reinterpret_cast<const uint4*>(
                        a + static_cast<size_t>(global_token) * shape_k +
                        k_begin + k_chunk);
                }
                const uint32_t physical_k =
                    k_chunk ^ ((local_token & 7u) << 4u);
                *reinterpret_cast<uint4*>(
                    smem_activation + local_token * kBlockK + physical_k) =
                    value;
            }

                cutlass::arch::fence_view_async_shared();
                ptx::sync_aligned(128, 1u + producer_group);
                if (wg_tid == 0u)
                    full_barriers[stage_idx].arrive();
                advance_pipeline();
            }
        }
        return;
    }

    const uint32_t warp = wg_tid >> 5u;
    const uint32_t output_n0 = block_n + warp * 16u + lane / 4u;
    const uint32_t output_n1 = output_n0 + 8u;
    const uint32_t token_in_group0 = 2u * (lane & 3u);
    const uint32_t token_in_group1 = token_in_group0 + 1u;
    const float expert_scale = global_scale[expert] * 8.0f;
    for (uint32_t token_group_base = 0u;
         token_group_base < total_token_groups;
         token_group_base += kGroupsPerPass) {
        const uint32_t remaining_groups =
            total_token_groups - token_group_base;
        const uint32_t pass_groups = remaining_groups < kGroupsPerPass
            ? remaining_groups : kGroupsPerPass;
        float accum[kGroupsPerPass][WGMMA::kNumAccum] = {};

        for (uint32_t k_block = 0; k_block < k_blocks; ++k_block) {
            full_barriers[stage_idx].wait(phase);
            auto* smem_weight =
                smem_weight_base + stage_idx * kWeightBytes;
            auto* smem_activation =
                smem_activation_base + stage_idx * kActivationBytes;
            auto weight_desc = mma::sm90::make_smem_desc(smem_weight, 1);
            const uint32_t weight_desc_base = weight_desc.reg32_[0];

            #pragma unroll
            for (uint32_t batch_base = 0u;
                 batch_base < kGroupsPerPass; batch_base += 2u) {
                if (batch_base >= pass_groups)
                    break;
                #pragma unroll
                for (uint32_t batch_item = 0u; batch_item < 2u;
                     ++batch_item) {
                    const uint32_t token_group = batch_base + batch_item;
                    if (token_group >= pass_groups)
                        break;
                    #pragma unroll
                    for (uint32_t i = 0; i < WGMMA::kNumAccum; ++i)
                        ptx::warpgroup_fence_operand(accum[token_group][i]);
                }
                ptx::warpgroup_arrive();
                #pragma unroll
                for (uint32_t batch_item = 0u; batch_item < 2u;
                     ++batch_item) {
                    const uint32_t token_group = batch_base + batch_item;
                    if (token_group >= pass_groups)
                        break;
                    auto activation_desc = mma::sm90::make_smem_desc(
                        smem_activation +
                            token_group * kTokenGroup * kBlockK, 1);
                    const uint32_t activation_desc_base =
                        activation_desc.reg32_[0];
                    #pragma unroll
                    for (uint32_t k_inner = 0;
                         k_inner < kBlockK / WGMMA::K; ++k_inner) {
                        weight_desc.reg32_[0] =
                            weight_desc_base + k_inner * WGMMA::K / 16u;
                        activation_desc.reg32_[0] =
                            activation_desc_base + k_inner * WGMMA::K / 16u;
                        WGMMA::wgmma(weight_desc, activation_desc,
                                     accum[token_group],
                                     k_block != 0u || k_inner != 0u);
                    }
                }
                ptx::warpgroup_commit_batch();
                ptx::warpgroup_wait<0>();
                #pragma unroll
                for (uint32_t batch_item = 0u; batch_item < 2u;
                     ++batch_item) {
                    const uint32_t token_group = batch_base + batch_item;
                    if (token_group >= pass_groups)
                        break;
                    #pragma unroll
                    for (uint32_t i = 0; i < WGMMA::kNumAccum; ++i)
                        ptx::warpgroup_fence_operand(accum[token_group][i]);
                }
            }
            if (lane == 0u)
                empty_barriers[stage_idx].arrive();
            advance_pipeline();
        }

        #pragma unroll
        for (uint32_t token_group = 0;
             token_group < kGroupsPerPass; ++token_group) {
            if (token_group >= pass_groups)
                break;
            const uint32_t global_group =
                token_group_base + token_group;
            const uint32_t token0 = expert_begin +
                global_group * kTokenGroup + token_in_group0;
            const uint32_t token1 = expert_begin +
                global_group * kTokenGroup + token_in_group1;
            if (token0 < expert_end) {
                const float scale = a_scale[token0] * expert_scale;
                d[static_cast<size_t>(token0) * shape_n + output_n0] =
                    __float2bfloat16_rn(accum[token_group][0] * scale);
                d[static_cast<size_t>(token0) * shape_n + output_n1] =
                    __float2bfloat16_rn(accum[token_group][2] * scale);
            }
            if (token1 < expert_end) {
                const float scale = a_scale[token1] * expert_scale;
                d[static_cast<size_t>(token1) * shape_n + output_n0] =
                    __float2bfloat16_rn(accum[token_group][1] * scale);
                d[static_cast<size_t>(token1) * shape_n + output_n1] =
                    __float2bfloat16_rn(accum[token_group][3] * scale);
            }
        }
    }
    (void)shape_m;
#else
    if (blockIdx.x == 0 && threadIdx.x == 0)
        DG_DEVICE_ASSERT(false && "This kernel only supports sm_90a");
#endif
}

// A single warpgroup cooperatively decodes each packed-FP4 K tile, stages
// the active token rows, and then consumes the tile with narrow-N WGMMA.  Independent
// CTAs provide latency hiding, avoiding a dedicated producer warpgroup and the
// full/empty pipeline barriers required by the warp-specialized variant.
template <bool kPretransformedWeights, bool kExactByteFusedWeights,
          uint32_t kBlockK, bool kActivationMulticast = false,
          uint32_t kStaticShapeN = 0, uint32_t kStaticShapeK = 0,
          uint32_t kStaticNumGroups = 0>
CUTLASS_GLOBAL __launch_bounds__(128, 6) void
sm90_nvfp4_grouped_gemm_small_m_cooperative_impl(
        __nv_bfloat16* __restrict__ d,
        const __grid_constant__ cute::TmaDescriptor tensor_map_a,
        const float* __restrict__ a_scale,
        const uint8_t* __restrict__ w_packed,
        const uint8_t* __restrict__ block_scale,
        const float* __restrict__ global_scale,
        const int* __restrict__ offsets,
        uint32_t shape_m, uint32_t shape_n, uint32_t shape_k,
        uint32_t num_groups) {
#if (defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900)) || defined(__CLION_IDE__)
    DG_STATIC_ASSERT(kStaticShapeN > 0 && kStaticShapeK > 0 &&
                     kStaticNumGroups > 0,
                     "Small-M dimensions must be JIT constants");
    constexpr uint32_t kShapeN = kStaticShapeN;
    constexpr uint32_t kShapeK = kStaticShapeK;
    constexpr uint32_t kNumGroups = kStaticNumGroups;
    constexpr uint32_t kBlockN = 64;
    constexpr uint32_t kTokenGroup = 32;
    constexpr uint32_t kGroupsPerPass = 1;
    constexpr uint32_t kPassSplits = 2;
    constexpr uint32_t kMaxTailExperts = 3;
    constexpr uint32_t kTokensPerPass = kTokenGroup * kGroupsPerPass;
    constexpr uint32_t kDescriptorK = 128;
    constexpr uint32_t kKSubtiles = kBlockK / kDescriptorK;
    constexpr uint32_t kWeightSubtileBytes = kBlockN * kDescriptorK;
    constexpr uint32_t kActivationSubtileBytes =
        kTokensPerPass * kDescriptorK;
    constexpr uint32_t kWeightBytes = kBlockN * kBlockK;
    constexpr uint32_t kActivationBytes = kTokensPerPass * kBlockK;
    constexpr bool kUseCpAsyncPackedPipeline = false;
    constexpr uint32_t kRawPackedBytes =
        kUseCpAsyncPackedPipeline && kExactByteFusedWeights &&
        kBlockK == 256 ? 4096u : 0u;
    constexpr uint32_t kLutBytes = 1024;
    constexpr uint32_t kNumActivationMulticast =
        kActivationMulticast ? 2u : 1u;
    using Barrier = cutlass::arch::ClusterTransactionBarrier;
    using WGMMA = typename mma::sm90::FP8MMASelector<kTokenGroup>::type;
    using WGMMA8 = typename mma::sm90::FP8MMASelector<8>::type;
    using WGMMA16 = typename mma::sm90::FP8MMASelector<16>::type;
    using WGMMA24 = typename mma::sm90::FP8MMASelector<24>::type;
    DG_STATIC_ASSERT(WGMMA::M == kBlockN && WGMMA::N == kTokenGroup &&
                     WGMMA::K == 32 && WGMMA::kNumAccum == 16,
                     "Unexpected cooperative small-M WGMMA shape");
    DG_STATIC_ASSERT(WGMMA16::M == kBlockN && WGMMA16::N == 16 &&
                     WGMMA16::K == 32 && WGMMA16::kNumAccum == 8,
                     "Unexpected cooperative small-M N16 WGMMA shape");
    DG_STATIC_ASSERT(WGMMA8::kNumAccum == 4 &&
                     WGMMA24::kNumAccum == 12,
                     "Unexpected adaptive small-M WGMMA shape");
    DG_STATIC_ASSERT(!(kPretransformedWeights && kExactByteFusedWeights),
                     "Weight layouts are mutually exclusive");
    DG_STATIC_ASSERT(kBlockK == 128 || kBlockK == 256,
                     "Cooperative K tile must be K128 or K256");

    extern __shared__ __align__(1024) uint8_t smem_buffer[];
    auto* smem_weight =
        reinterpret_cast<__nv_fp8_e4m3*>(smem_buffer);
    auto* smem_activation = reinterpret_cast<__nv_fp8_e4m3*>(
        smem_buffer + kWeightBytes);
    auto* smem_raw_packed =
        smem_buffer + kWeightBytes + kActivationBytes;
    auto* smem_lut = reinterpret_cast<uint2*>(
        smem_buffer + kWeightBytes + kActivationBytes + kRawPackedBytes);
    auto* activation_barrier = reinterpret_cast<Barrier*>(
        smem_buffer + kWeightBytes + kActivationBytes + kRawPackedBytes +
        kLutBytes);

    const uint32_t tid = threadIdx.x;
    const uint32_t lane = tid & 31u;
    const uint32_t warp = tid >> 5u;
    constexpr uint32_t n_tiles = kShapeN / kBlockN;
    constexpr uint32_t total_tiles = kNumGroups * n_tiles;
    uint32_t pass_idx = 0u;
    uint32_t expert = kNumGroups;
    uint32_t block_n = 0u;
    if (blockIdx.x < total_tiles) {
        const uint32_t tile_idx = blockIdx.x;
        expert = tile_idx / n_tiles;
        block_n = (tile_idx - expert * n_tiles) * kBlockN;
    } else {
        pass_idx = 1u;
        const uint32_t tail_idx = blockIdx.x - total_tiles;
        const uint32_t long_expert_rank = tail_idx / n_tiles;
        block_n = (tail_idx - long_expert_rank * n_tiles) * kBlockN;

        // All lanes need the same mapping, but only one lane per warp scans
        // the tiny offsets table.  Broadcasting avoids 128 redundant scans
        // without adding a CTA-wide synchronization to the tail path.
        if (lane == 0u && long_expert_rank < kMaxTailExperts) {
            uint32_t seen_long_experts = 0u;
            for (uint32_t candidate = 0u; candidate < kNumGroups;
                 ++candidate) {
                const uint32_t candidate_begin =
                    static_cast<uint32_t>(offsets[candidate]);
                const uint32_t candidate_end =
                    static_cast<uint32_t>(offsets[candidate + 1u]);
                if (candidate_end - candidate_begin > kTokenGroup) {
                    if (seen_long_experts == long_expert_rank) {
                        expert = candidate;
                        break;
                    }
                    ++seen_long_experts;
                }
            }
        }
        expert = __shfl_sync(0xffffffffu, expert, 0);
    }
    if (expert >= kNumGroups)
        return;
    const uint32_t expert_begin = static_cast<uint32_t>(offsets[expert]);
    const uint32_t expert_end = static_cast<uint32_t>(offsets[expert + 1]);
    const uint32_t expert_tokens = expert_end - expert_begin;
    if (expert_tokens == 0u)
        return;
    const uint32_t total_token_groups =
        math::ceil_div(expert_tokens, kTokenGroup);
    const uint32_t first_token_group = pass_idx * kGroupsPerPass;
    if (first_token_group >= total_token_groups)
        return;

    if (tid == 0u) {
        cute::prefetch_tma_descriptor(&tensor_map_a);
        activation_barrier->init(1);
        cutlass::arch::fence_barrier_init();
    }
    if (tid < 64u) {
        reinterpret_cast<uint4*>(smem_lut)[tid] =
            reinterpret_cast<const uint4*>(
                nvfp4::kE2M1AndUe4m3ToFp8Lut)[tid];
    }
    __syncthreads();
    if constexpr (kActivationMulticast)
        comm::cluster_sync_with_relaxed_arrive();

    constexpr uint32_t packed_k = kShapeK / 2u;
    constexpr uint32_t scale_k = kShapeK / 16u;
    constexpr uint32_t k_blocks = kShapeK / kBlockK;
    const uint32_t weight_row = tid & 63u;
    const uint32_t k_half = tid >> 6u;
    const uint32_t global_weight = block_n + weight_row;
    const uint32_t row_swizzle = (weight_row & 7u) << 4u;

    const uint32_t output_n0 = block_n + warp * 16u + lane / 4u;
    const uint32_t output_n1 = output_n0 + 8u;
    const uint32_t token_in_eight0 = 2u * (lane & 3u);
    const uint32_t token_in_eight1 = token_in_eight0 + 1u;
    const float expert_scale = global_scale[expert] * 8.0f;
    uint32_t activation_phase = 0u;

    for (uint32_t token_group_base = first_token_group;
         token_group_base < total_token_groups;
         token_group_base += kGroupsPerPass * kPassSplits) {
        const uint32_t remaining_groups =
            total_token_groups - token_group_base;
        const uint32_t pass_groups = remaining_groups < kGroupsPerPass
            ? remaining_groups : kGroupsPerPass;
        const uint32_t pass_token_begin = expert_begin +
            token_group_base * kTokenGroup;
        const uint32_t active_tokens = expert_end - pass_token_begin <
                kTokenGroup
            ? expert_end - pass_token_begin : kTokenGroup;
        const uint32_t active_token_eights = math::ceil_div(active_tokens, 8u);
        float accum[kGroupsPerPass][WGMMA::kNumAccum] = {};

        // TMA skips global OOB positions. Clear the reused activation tile
        // once per token pass so the final M tail is deterministic.
        for (uint32_t chunk = tid;
             chunk < kActivationBytes / sizeof(uint4); chunk += 128u) {
            reinterpret_cast<uint4*>(smem_activation)[chunk] =
                make_uint4(0u, 0u, 0u, 0u);
        }
        __syncthreads();

        auto run_k_loop = [&]<typename NarrowWGMMA>() {
        if constexpr (kUseCpAsyncPackedPipeline &&
                      kExactByteFusedWeights && kBlockK == 256) {
            constexpr uint32_t kPreparedPackedRowBytes = 64u;
            constexpr uint32_t kPreparedScaleRowBytes = 8u;
            constexpr uint32_t kPreparedTileRowBytes =
                kPreparedPackedRowBytes + kPreparedScaleRowBytes;
            constexpr uint32_t kPreparedNTile = 64u;
            constexpr uint32_t kPreparedNTileBytes =
                kPreparedNTile * kPreparedTileRowBytes;
            constexpr uint32_t prepared_k_blocks =
                kShapeK / kDescriptorK;
            const uint32_t prepared_n_tile = block_n / kPreparedNTile;
            const size_t packed_vector0 =
                (static_cast<size_t>(k_half) * 2u) * kPreparedNTile +
                weight_row;
            const size_t packed_vector1 =
                packed_vector0 + kPreparedNTile;

            auto get_tile_base = [&](uint32_t prepared_k_block) {
                const size_t tile_offset =
                    ((static_cast<size_t>(expert) * prepared_k_blocks +
                      prepared_k_block) * (kShapeN / kPreparedNTile) +
                     prepared_n_tile) * kPreparedNTileBytes;
                return w_packed + tile_offset;
            };
            auto prefetch_packed = [&](uint32_t prepared_k_block) {
                const uint8_t* tile_base =
                    get_tile_base(prepared_k_block);
                ptx::cp_async_global_to_shared_16(
                    smem_raw_packed + packed_vector0 * sizeof(uint4),
                    tile_base + packed_vector0 * sizeof(uint4));
                ptx::cp_async_global_to_shared_16(
                    smem_raw_packed + packed_vector1 * sizeof(uint4),
                    tile_base + packed_vector1 * sizeof(uint4));
                ptx::cp_async_commit_group();
            };
            auto decode_packed = [&](uint32_t prepared_k_block,
                                     uint32_t decoded_slot) {
                const uint8_t* tile_base =
                    get_tile_base(prepared_k_block);
                const auto packed0 = ptx::ld_shared(
                    reinterpret_cast<const uint4*>(
                        smem_raw_packed +
                        packed_vector0 * sizeof(uint4)));
                const auto packed1 = ptx::ld_shared(
                    reinterpret_cast<const uint4*>(
                        smem_raw_packed +
                        packed_vector1 * sizeof(uint4)));
                const uint32_t fused_packed[8] = {
                    packed0.x, packed0.y, packed0.z, packed0.w,
                    packed1.x, packed1.y, packed1.z, packed1.w};
                const uint32_t fused_scales = ptx::ld_global_stream(
                    reinterpret_cast<const uint32_t*>(
                        tile_base +
                        static_cast<size_t>(kPreparedNTile) *
                            kPreparedPackedRowBytes +
                        (static_cast<size_t>(k_half) * kPreparedNTile +
                         weight_row) * sizeof(uint32_t)));
                #pragma unroll
                for (uint32_t local_scale = 0; local_scale < 4u;
                     ++local_scale) {
                    const uint32_t scale =
                        (fused_scales >> (local_scale * 8u)) & 0x7fu;
                    const uint2 lut = smem_lut[scale];
                    const uint2 decoded0 =
                        detail::dequant_nvfp4x8_bitwoven_to_fp8(
                            fused_packed[local_scale * 2u], lut);
                    const uint2 decoded1 =
                        detail::dequant_nvfp4x8_bitwoven_to_fp8(
                            fused_packed[local_scale * 2u + 1u], lut);
                    const uint32_t scale_in_subtile =
                        k_half * 4u + local_scale;
                    const uint32_t physical =
                        scale_in_subtile * 16u ^ row_swizzle;
                    auto* decoded_target = reinterpret_cast<uint4*>(
                        reinterpret_cast<uint8_t*>(
                            smem_weight +
                            decoded_slot * kWeightSubtileBytes +
                            weight_row * kDescriptorK) + physical);
                    *decoded_target = make_uint4(
                        decoded0.x, decoded0.y,
                        decoded1.x, decoded1.y);
                }
            };
            auto issue_wgmma_subtile = [&](uint32_t k_block,
                                            uint32_t k_subtile) {
                auto weight_desc = mma::sm90::make_smem_desc(
                    smem_weight + k_subtile * kWeightSubtileBytes, 1);
                auto activation_desc = mma::sm90::make_smem_desc(
                    smem_activation +
                        k_subtile * kActivationSubtileBytes, 1);
                const uint32_t weight_desc_base =
                    weight_desc.reg32_[0];
                const uint32_t activation_desc_base =
                    activation_desc.reg32_[0];
                #pragma unroll
                for (uint32_t k_inner = 0;
                     k_inner < kDescriptorK / NarrowWGMMA::K;
                     ++k_inner) {
                    weight_desc.reg32_[0] = weight_desc_base +
                        k_inner * NarrowWGMMA::K / 16u;
                    activation_desc.reg32_[0] = activation_desc_base +
                        k_inner * NarrowWGMMA::K / 16u;
                    NarrowWGMMA::wgmma(
                        weight_desc, activation_desc, accum[0],
                        k_block != 0u || k_subtile != 0u ||
                            k_inner != 0u);
                }
            };

            prefetch_packed(0u);
            for (uint32_t k_block = 0; k_block < k_blocks; ++k_block) {
                const uint32_t k_begin = k_block * kBlockK;
                if (tid == 0u) {
                    tma::copy<kBlockK, kTokensPerPass, 128>(
                        &tensor_map_a, activation_barrier, smem_activation,
                        k_begin,
                        expert_begin + token_group_base * kTokenGroup,
                        kNumActivationMulticast);
                    activation_barrier->arrive_and_expect_tx(
                        kActivationBytes);
                }

                ptx::cp_async_wait_group<0>();
                decode_packed(k_block * 2u, 0u);
                prefetch_packed(k_block * 2u + 1u);

                activation_barrier->wait(activation_phase);
                activation_phase ^= 1u;
                cutlass::arch::fence_view_async_shared();
                __syncthreads();

                #pragma unroll
                for (uint32_t i = 0; i < WGMMA::kNumAccum; ++i)
                    ptx::warpgroup_fence_operand(accum[0][i]);
                ptx::warpgroup_arrive();
                issue_wgmma_subtile(k_block, 0u);
                ptx::warpgroup_commit_batch();

                ptx::cp_async_wait_group<0>();
                decode_packed(k_block * 2u + 1u, 1u);
                if (k_block + 1u < k_blocks)
                    prefetch_packed((k_block + 1u) * 2u);
                __syncthreads();

                issue_wgmma_subtile(k_block, 1u);
                ptx::warpgroup_commit_batch();
                ptx::warpgroup_wait<0>();
                #pragma unroll
                for (uint32_t i = 0; i < WGMMA::kNumAccum; ++i)
                    ptx::warpgroup_fence_operand(accum[0][i]);
            }
        } else {
        for (uint32_t k_block = 0; k_block < k_blocks; ++k_block) {
            const uint32_t k_begin = k_block * kBlockK;
            if (tid == 0u) {
                tma::copy<kBlockK, kTokensPerPass, 128>(
                    &tensor_map_a, activation_barrier, smem_activation,
                    k_begin,
                    expert_begin + token_group_base * kTokenGroup,
                    kNumActivationMulticast);
                activation_barrier->arrive_and_expect_tx(kActivationBytes);
            }
            #pragma unroll
            for (uint32_t k_subtile = 0;
                 k_subtile < kKSubtiles; ++k_subtile) {
                uint32_t fused_packed[8];
                uint32_t fused_scales = 0u;
                if constexpr (kExactByteFusedWeights) {
                    constexpr uint32_t kPreparedPackedRowBytes = 64u;
                    constexpr uint32_t kPreparedScaleRowBytes = 8u;
                    constexpr uint32_t kPreparedTileRowBytes =
                        kPreparedPackedRowBytes + kPreparedScaleRowBytes;
                    constexpr uint32_t kPreparedNTile = 64u;
                    constexpr uint32_t kPreparedNTileBytes =
                        kPreparedNTile * kPreparedTileRowBytes;
                    constexpr uint32_t prepared_k_blocks = kShapeK / 128u;
                    const uint32_t prepared_k_block =
                        k_block * kKSubtiles + k_subtile;
                    const uint32_t prepared_n_tile = block_n / kPreparedNTile;
                    const size_t tile_offset =
                        ((static_cast<size_t>(expert) * prepared_k_blocks +
                          prepared_k_block) * (kShapeN / kPreparedNTile) +
                         prepared_n_tile) * kPreparedNTileBytes;
                    const uint8_t* tile_base = w_packed + tile_offset;
                    const size_t packed_vector0 =
                        (static_cast<size_t>(k_half) * 2u) *
                            kPreparedNTile + weight_row;
                    const size_t packed_vector1 = packed_vector0 + kPreparedNTile;
                    const auto packed0 = ptx::ld_global_stream(
                        reinterpret_cast<const uint4*>(
                            tile_base + packed_vector0 * sizeof(uint4)));
                    const auto packed1 = ptx::ld_global_stream(
                        reinterpret_cast<const uint4*>(
                            tile_base + packed_vector1 * sizeof(uint4)));
                    fused_packed[0] = packed0.x;
                    fused_packed[1] = packed0.y;
                    fused_packed[2] = packed0.z;
                    fused_packed[3] = packed0.w;
                    fused_packed[4] = packed1.x;
                    fused_packed[5] = packed1.y;
                    fused_packed[6] = packed1.z;
                    fused_packed[7] = packed1.w;
                    fused_scales = ptx::ld_global_stream(
                        reinterpret_cast<const uint32_t*>(
                            tile_base + static_cast<size_t>(kPreparedNTile) *
                                kPreparedPackedRowBytes +
                            (static_cast<size_t>(k_half) * kPreparedNTile +
                             weight_row) * sizeof(uint32_t)));
                }

                #pragma unroll
                for (uint32_t local_scale = 0; local_scale < 4u;
                     ++local_scale) {
                    const uint32_t scale_in_subtile =
                        k_half * 4u + local_scale;
                    const uint32_t scale_in_block =
                        k_subtile * 8u + scale_in_subtile;
                    const uint32_t global_scale_k =
                        k_begin / 16u + scale_in_block;
                    uint32_t scale;
                    uint32_t packed0;
                    uint32_t packed1;
                    if constexpr (kExactByteFusedWeights) {
                        scale =
                            (fused_scales >> (local_scale * 8u)) & 0xffu;
                        packed0 = fused_packed[local_scale * 2u];
                        packed1 = fused_packed[local_scale * 2u + 1u];
                    } else if constexpr (kPretransformedWeights) {
                        const size_t scale_offset =
                            (static_cast<size_t>(expert) * scale_k +
                             global_scale_k) * kShapeN + global_weight;
                        const auto* packed_words =
                            reinterpret_cast<const uint32_t*>(
                                w_packed + scale_offset * 8u);
                        packed0 = packed_words[0];
                        packed1 = packed_words[1];
                        scale = block_scale[scale_offset];
                    } else {
                        const size_t row_offset =
                            static_cast<size_t>(expert) * kShapeN +
                            global_weight;
                        const auto* packed_words =
                            reinterpret_cast<const uint32_t*>(
                                w_packed + row_offset * packed_k +
                                k_begin / 2u + scale_in_block * 8u);
                        packed0 = packed_words[0];
                        packed1 = packed_words[1];
                        scale = block_scale[
                            row_offset * scale_k + global_scale_k];
                    }
                    const uint2 lut = smem_lut[scale & 0x7fu];
                    const uint32_t physical =
                        scale_in_subtile * 16u ^ row_swizzle;
                    auto* decoded_target = reinterpret_cast<uint4*>(
                        reinterpret_cast<uint8_t*>(
                            smem_weight +
                            k_subtile * kWeightSubtileBytes +
                            weight_row * kDescriptorK) + physical);
                    if constexpr (kExactByteFusedWeights) {
                        const uint2 decoded0 =
                            detail::dequant_nvfp4x8_bitwoven_to_fp8(
                                packed0, lut);
                        const uint2 decoded1 =
                            detail::dequant_nvfp4x8_bitwoven_to_fp8(
                                packed1, lut);
                        *decoded_target = make_uint4(
                            decoded0.x, decoded0.y,
                            decoded1.x, decoded1.y);
                    } else {
                        *decoded_target = make_uint4(
                            detail::dequant_nvfp4x4_to_fp8(packed0, lut),
                            detail::dequant_nvfp4x4_to_fp8(
                                packed0 >> 16u, lut),
                            detail::dequant_nvfp4x4_to_fp8(packed1, lut),
                            detail::dequant_nvfp4x4_to_fp8(
                                packed1 >> 16u, lut));
                    }
                }
            }

            activation_barrier->wait(activation_phase);
            activation_phase ^= 1u;
            cutlass::arch::fence_view_async_shared();
            __syncthreads();

            #pragma unroll
            for (uint32_t batch_base = 0u;
                 batch_base < kGroupsPerPass; batch_base += 2u) {
                if (batch_base >= pass_groups)
                    break;
                #pragma unroll
                for (uint32_t batch_item = 0u; batch_item < 2u;
                     ++batch_item) {
                    const uint32_t token_group = batch_base + batch_item;
                    if (token_group >= pass_groups)
                        break;
                    #pragma unroll
                    for (uint32_t i = 0; i < WGMMA::kNumAccum; ++i)
                        ptx::warpgroup_fence_operand(accum[token_group][i]);
                }
                ptx::warpgroup_arrive();
                #pragma unroll
                for (uint32_t batch_item = 0u; batch_item < 2u;
                     ++batch_item) {
                    const uint32_t token_group = batch_base + batch_item;
                    if (token_group >= pass_groups)
                        break;
                    #pragma unroll
                    for (uint32_t k_subtile = 0;
                         k_subtile < kKSubtiles; ++k_subtile) {
                        auto weight_desc = mma::sm90::make_smem_desc(
                            smem_weight +
                                k_subtile * kWeightSubtileBytes, 1);
                        auto activation_desc = mma::sm90::make_smem_desc(
                            smem_activation +
                                k_subtile * kActivationSubtileBytes, 1);
                        const uint32_t weight_desc_base =
                            weight_desc.reg32_[0];
                        const uint32_t activation_desc_base =
                            activation_desc.reg32_[0];
                        #pragma unroll
                        for (uint32_t k_inner = 0;
                             k_inner < kDescriptorK / NarrowWGMMA::K;
                             ++k_inner) {
                            weight_desc.reg32_[0] = weight_desc_base +
                                k_inner * NarrowWGMMA::K / 16u;
                            activation_desc.reg32_[0] =
                                activation_desc_base +
                                k_inner * NarrowWGMMA::K / 16u;
                            NarrowWGMMA::wgmma(
                                weight_desc, activation_desc,
                                accum[token_group],
                                k_block != 0u || k_subtile != 0u ||
                                    k_inner != 0u);
                        }
                    }
                }
                ptx::warpgroup_commit_batch();
                ptx::warpgroup_wait<0>();
                #pragma unroll
                for (uint32_t batch_item = 0u; batch_item < 2u;
                     ++batch_item) {
                    const uint32_t token_group = batch_base + batch_item;
                    if (token_group >= pass_groups)
                        break;
                    #pragma unroll
                    for (uint32_t i = 0; i < WGMMA::kNumAccum; ++i)
                        ptx::warpgroup_fence_operand(accum[token_group][i]);
                }
            }
        }
        }
        };
        if (active_token_eights == 1u)
            run_k_loop.template operator()<WGMMA8>();
        else if (active_token_eights == 2u)
            run_k_loop.template operator()<WGMMA16>();
        else if (active_token_eights == 3u)
            run_k_loop.template operator()<WGMMA24>();
        else
            run_k_loop.template operator()<WGMMA>();

        #pragma unroll
        for (uint32_t token_group = 0;
             token_group < kGroupsPerPass; ++token_group) {
            if (token_group >= pass_groups)
                break;
            const uint32_t global_group =
                token_group_base + token_group;
            #pragma unroll
            for (uint32_t token_quarter = 0;
                 token_quarter < 4u; ++token_quarter) {
                if (token_quarter >= active_token_eights)
                    break;
                const uint32_t token0 = expert_begin +
                    global_group * kTokenGroup + token_quarter * 8u +
                    token_in_eight0;
                const uint32_t token1 = expert_begin +
                    global_group * kTokenGroup + token_quarter * 8u +
                    token_in_eight1;
                const uint32_t accum_base = token_quarter * 4u;
                if (token0 < expert_end) {
                    const float scale = a_scale[token0] * expert_scale;
                    d[static_cast<size_t>(token0) * kShapeN + output_n0] =
                        __float2bfloat16_rn(
                            accum[token_group][accum_base] * scale);
                    d[static_cast<size_t>(token0) * kShapeN + output_n1] =
                        __float2bfloat16_rn(
                            accum[token_group][accum_base + 2u] * scale);
                }
                if (token1 < expert_end) {
                    const float scale = a_scale[token1] * expert_scale;
                    d[static_cast<size_t>(token1) * kShapeN + output_n0] =
                        __float2bfloat16_rn(
                            accum[token_group][accum_base + 1u] * scale);
                    d[static_cast<size_t>(token1) * kShapeN + output_n1] =
                        __float2bfloat16_rn(
                            accum[token_group][accum_base + 3u] * scale);
                }
            }
        }
    }
    (void)shape_m;
#else
    if (blockIdx.x == 0 && threadIdx.x == 0)
        DG_DEVICE_ASSERT(false && "This kernel only supports sm_90a");
#endif
}

template <uint32_t BLOCK_M, uint32_t BLOCK_N, uint32_t BLOCK_K,
          bool kWarpSpecialized, bool kPretransformedWeights>
CUTLASS_GLOBAL __launch_bounds__(kWarpSpecialized ? 256 : 128, 1) void
sm90_nvfp4_grouped_gemm_impl(
        __nv_bfloat16* __restrict__ d,
        const __nv_fp8_e4m3* __restrict__ a,
        const float* __restrict__ a_scale,
        const uint8_t* __restrict__ w_packed,
        const uint8_t* __restrict__ block_scale,
        const float* __restrict__ global_scale,
        const int* __restrict__ offsets,
        uint32_t shape_m, uint32_t shape_n, uint32_t shape_k,
        uint32_t num_groups) {
#if (defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900)) || defined(__CLION_IDE__)
    DG_STATIC_ASSERT(BLOCK_M == 64, "The initial SM90 path uses one WGMMA M tile");
    DG_STATIC_ASSERT(BLOCK_N == 128 || BLOCK_N == 256,
                     "The optimized SM90 path uses WGMMA N=128 or N=256");
    DG_STATIC_ASSERT(BLOCK_K == 128, "FP8 swizzle atoms require K=128");

    using WGMMA = typename mma::sm90::FP8MMASelector<BLOCK_N>::type;
    DG_STATIC_ASSERT(WGMMA::M == BLOCK_M && WGMMA::K == 32,
                     "Unexpected WGMMA instruction shape");

    constexpr uint32_t kThreads = 128;
    constexpr uint32_t kVectorBytes = 16;
    constexpr uint32_t kABytes = BLOCK_M * BLOCK_K;
    constexpr uint32_t kBBytes = BLOCK_N * BLOCK_K;
    constexpr uint32_t kNumStages = 2;

    extern __shared__ __align__(1024) uint8_t smem_buffer[];
    auto* smem_a_base = reinterpret_cast<__nv_fp8_e4m3*>(smem_buffer);
    auto* smem_b_primary_base = reinterpret_cast<__nv_fp8_e4m3*>(
        smem_buffer + kNumStages * kABytes);
    auto* smem_primary_lut = reinterpret_cast<uint2*>(
        smem_buffer + kNumStages * (kABytes + kBBytes));

    const uint32_t tid = threadIdx.x;
    const uint32_t n_tiles = shape_n / BLOCK_N;
    const uint32_t k_blocks = shape_k / BLOCK_K;
    const uint32_t packed_k = shape_k / 2;
    const uint32_t scale_k = shape_k / 16;
    (void)shape_m;

    // Stage the 1 KiB primary dequant table in shared memory once per
    // persistent CTA so divergent scale codes remain single-cycle loads.
    if (tid < 64) {
        reinterpret_cast<uint4*>(smem_primary_lut)[tid] =
            reinterpret_cast<const uint4*>(
                nvfp4::kE2M1AndUe4m3ToFp8Lut)[tid];
    }
    __syncthreads();

    if constexpr (kWarpSpecialized) {
        using Barrier = cutlass::arch::ClusterTransactionBarrier;
        auto* barrier_start = reinterpret_cast<Barrier*>(smem_primary_lut + 128);
        auto* full_barriers = barrier_start;
        auto* empty_barriers = barrier_start + kNumStages;

        if (threadIdx.x == 0) {
            #pragma unroll
            for (uint32_t stage = 0; stage < kNumStages; ++stage) {
                full_barriers[stage].init(1);
                empty_barriers[stage].init(4);
            }
            cutlass::arch::fence_barrier_init();
        }
        __syncthreads();

        const uint32_t wg_tid = threadIdx.x & 127u;
        const uint32_t lane = wg_tid & 31u;
        uint32_t stage_idx = 0;
        uint32_t phase = 0;
        auto advance_pipeline = [&]() {
            stage_idx = stage_idx == kNumStages - 1 ? 0 : stage_idx + 1;
            phase ^= stage_idx == 0;
        };

        auto map_tile = [&](const uint32_t tile_idx,
                            uint32_t& expert, uint32_t& expert_begin,
                            uint32_t& expert_end, uint32_t& row_begin,
                            uint32_t& col_begin) {
            expert = num_groups;
            uint32_t local_tile = tile_idx;
            #pragma unroll 1
            for (uint32_t e = 0; e < num_groups; ++e) {
                const uint32_t begin = static_cast<uint32_t>(offsets[e]);
                const uint32_t end = static_cast<uint32_t>(offsets[e + 1]);
                const uint32_t m_tiles = math::ceil_div(end - begin, BLOCK_M);
                const uint32_t expert_tiles = m_tiles * n_tiles;
                if (local_tile < expert_tiles) {
                    expert = e;
                    expert_begin = begin;
                    expert_end = end;
                    const uint32_t m_tile = local_tile / n_tiles;
                    const uint32_t n_tile = local_tile - m_tile * n_tiles;
                    row_begin = begin + m_tile * BLOCK_M;
                    col_begin = n_tile * BLOCK_N;
                    return true;
                }
                local_tile -= expert_tiles;
            }
            return false;
        };

        if (threadIdx.x >= 128) {
            // Producer warpgroup: keep the two-stage ring filled with decoded
            // A/B tiles. Named barrier 0 synchronizes only these 128 threads.
            for (uint32_t tile_idx = blockIdx.x;; tile_idx += gridDim.x) {
                uint32_t expert, expert_begin, expert_end, row_begin, col_begin;
                if (!map_tile(tile_idx, expert, expert_begin, expert_end,
                              row_begin, col_begin))
                    break;
                (void)expert_begin;

                for (uint32_t k_block = 0; k_block < k_blocks; ++k_block) {
                    if (wg_tid == 0)
                        empty_barriers[stage_idx].wait(phase ^ 1u);
                    ptx::sync_aligned(128, 0);

                    const uint32_t k_begin = k_block * BLOCK_K;
                    auto* smem_a = smem_a_base + stage_idx * BLOCK_M * BLOCK_K;
                    auto* smem_b_primary =
                        smem_b_primary_base + stage_idx * BLOCK_N * BLOCK_K;

                    constexpr uint32_t kAChunks = kABytes / kVectorBytes;
                    #pragma unroll
                    for (uint32_t chunk = wg_tid; chunk < kAChunks;
                         chunk += kThreads) {
                        const uint32_t row = chunk / (BLOCK_K / kVectorBytes);
                        const uint32_t k_chunk =
                            (chunk % (BLOCK_K / kVectorBytes)) * kVectorBytes;
                        const uint32_t physical_k =
                            k_chunk ^ ((row & 7u) << 4u);
                        uint4 value = make_uint4(0u, 0u, 0u, 0u);
                        const uint32_t global_row = row_begin + row;
                        if (global_row < expert_end) {
                            const auto* src = reinterpret_cast<const uint4*>(
                                a + static_cast<size_t>(global_row) * shape_k +
                                k_begin + k_chunk);
                            value = *src;
                        }
                        *reinterpret_cast<uint4*>(
                            smem_a + row * BLOCK_K + physical_k) = value;
                    }

                    for (uint32_t b_row = wg_tid; b_row < BLOCK_N;
                         b_row += kThreads) {
                        const uint32_t global_n = col_begin + b_row;
                        const auto* packed_row = w_packed +
                            (static_cast<size_t>(expert) * shape_n + global_n) *
                                packed_k +
                            k_begin / 2;
                        const auto* scale_row = block_scale +
                            (static_cast<size_t>(expert) * shape_n + global_n) *
                                scale_k +
                            k_begin / 16;
                        auto* dst_primary = reinterpret_cast<uint8_t*>(
                            smem_b_primary + b_row * BLOCK_K);
                        const uint32_t row_swizzle = (b_row & 7u) << 4u;

                        #pragma unroll
                        for (uint32_t scale_idx = 0;
                             scale_idx < BLOCK_K / 16; ++scale_idx) {
                            const uint32_t k_group = k_begin / 16 + scale_idx;
                            const uint8_t* packed_group =
                                packed_row + scale_idx * 8u;
                            uint32_t scale = scale_row[scale_idx];
                            if constexpr (kPretransformedWeights) {
                                packed_group = w_packed +
                                    ((static_cast<size_t>(expert) * scale_k +
                                      k_group) * shape_n + global_n) * 8u;
                                scale = block_scale[
                                    (static_cast<size_t>(expert) * scale_k +
                                     k_group) * shape_n + global_n];
                            }
                            const uint32_t physical =
                                scale_idx * 16u ^ row_swizzle;
                            const uint2 primary_lut =
                                smem_primary_lut[scale & 0x7fu];
                            const auto* packed_words =
                                reinterpret_cast<const uint32_t*>(
                                    packed_group);
                            const uint2 q0 =
                                nvfp4::dequant_nvfp4_to_fp8_pair_with_lut(
                                    packed_words[0], primary_lut);
                            const uint2 q1 =
                                nvfp4::dequant_nvfp4_to_fp8_pair_with_lut(
                                    packed_words[1], primary_lut);
                            *reinterpret_cast<uint4*>(dst_primary + physical) =
                                make_uint4(
                                    __byte_perm(q0.y, q0.x, 0x5140),
                                    __byte_perm(q0.y, q0.x, 0x7362),
                                    __byte_perm(q1.y, q1.x, 0x5140),
                                    __byte_perm(q1.y, q1.x, 0x7362));
                        }
                    }

                    cutlass::arch::fence_view_async_shared();
                    ptx::sync_aligned(128, 0);
                    if (wg_tid == 0)
                        full_barriers[stage_idx].arrive();
                    advance_pipeline();
                }
            }
            return;
        }

        // Consumer warpgroup: wait on decoded stages, run WGMMA, and store.
        for (uint32_t tile_idx = blockIdx.x;; tile_idx += gridDim.x) {
            uint32_t expert, expert_begin, expert_end, row_begin, col_begin;
            if (!map_tile(tile_idx, expert, expert_begin, expert_end,
                          row_begin, col_begin))
                break;
            (void)expert_begin;

            float accum[WGMMA::kNumAccum] = {0.0f};
            for (uint32_t k_block = 0; k_block < k_blocks; ++k_block) {
                full_barriers[stage_idx].wait(phase);
                auto* smem_a = smem_a_base + stage_idx * BLOCK_M * BLOCK_K;
                auto* smem_b_primary =
                    smem_b_primary_base + stage_idx * BLOCK_N * BLOCK_K;
                auto a_desc = mma::sm90::make_smem_desc(smem_a, 1);
                auto b_primary_desc =
                    mma::sm90::make_smem_desc(smem_b_primary, 1);
                const uint32_t a_desc_base = a_desc.reg32_[0];
                const uint32_t b_primary_desc_base =
                    b_primary_desc.reg32_[0];
                #pragma unroll
                for (uint32_t i = 0; i < WGMMA::kNumAccum; ++i)
                    ptx::warpgroup_fence_operand(accum[i]);
                ptx::warpgroup_arrive();
                #pragma unroll
                for (uint32_t k_inner = 0;
                     k_inner < BLOCK_K / WGMMA::K; ++k_inner) {
                    a_desc.reg32_[0] =
                        a_desc_base + k_inner * WGMMA::K / 16;
                    b_primary_desc.reg32_[0] =
                        b_primary_desc_base + k_inner * WGMMA::K / 16;
                    WGMMA::wgmma(a_desc, b_primary_desc, accum,
                                 k_block != 0 || k_inner != 0);
                }
                ptx::warpgroup_commit_batch();
                #pragma unroll
                for (uint32_t i = 0; i < WGMMA::kNumAccum; ++i)
                    ptx::warpgroup_fence_operand(accum[i]);
                ptx::warpgroup_wait<0>();
                if (lane == 0)
                    empty_barriers[stage_idx].arrive();
                advance_pipeline();
            }

            const uint32_t warp = wg_tid >> 5u;
            const uint32_t local_row_0 = warp * 16u + lane / 4u;
            const uint32_t local_row_1 = local_row_0 + 8u;
            const uint32_t global_row_0 = row_begin + local_row_0;
            const uint32_t global_row_1 = row_begin + local_row_1;
            const float expert_scale = global_scale[expert] * 8.0f;
            const float row_scale_0 = global_row_0 < expert_end
                ? a_scale[global_row_0] * expert_scale : 0.0f;
            const float row_scale_1 = global_row_1 < expert_end
                ? a_scale[global_row_1] * expert_scale : 0.0f;

            #pragma unroll
            for (uint32_t group = 0; group < BLOCK_N / 16; ++group) {
                const uint32_t local_col_0 =
                    group * 16u + 2u * (wg_tid & 3u);
                const uint32_t local_col_1 = local_col_0 + 8u;
                const uint32_t base = group * 8u;
                if (global_row_0 < expert_end) {
                    auto* row_ptr = d +
                        static_cast<size_t>(global_row_0) * shape_n + col_begin;
                    row_ptr[local_col_0] =
                        __float2bfloat16_rn(accum[base + 0] * row_scale_0);
                    row_ptr[local_col_0 + 1] =
                        __float2bfloat16_rn(accum[base + 1] * row_scale_0);
                    row_ptr[local_col_1] =
                        __float2bfloat16_rn(accum[base + 4] * row_scale_0);
                    row_ptr[local_col_1 + 1] =
                        __float2bfloat16_rn(accum[base + 5] * row_scale_0);
                }
                if (global_row_1 < expert_end) {
                    auto* row_ptr = d +
                        static_cast<size_t>(global_row_1) * shape_n + col_begin;
                    row_ptr[local_col_0] =
                        __float2bfloat16_rn(accum[base + 2] * row_scale_1);
                    row_ptr[local_col_0 + 1] =
                        __float2bfloat16_rn(accum[base + 3] * row_scale_1);
                    row_ptr[local_col_1] =
                        __float2bfloat16_rn(accum[base + 6] * row_scale_1);
                    row_ptr[local_col_1 + 1] =
                        __float2bfloat16_rn(accum[base + 7] * row_scale_1);
                }
            }
        }
        return;
    }

    for (uint32_t tile_idx = blockIdx.x;; tile_idx += gridDim.x) {
        // Map a linear tile to (expert, local M tile, N tile). The number of
        // experts is at most 32 for the target workload, so a device-side scan
        // is cheaper and safer than synchronizing offsets to the host.
        uint32_t expert = num_groups;
        uint32_t local_tile = tile_idx;
        uint32_t expert_begin = 0;
        uint32_t expert_end = 0;

        #pragma unroll 1
        for (uint32_t e = 0; e < num_groups; ++e) {
            const uint32_t begin = static_cast<uint32_t>(offsets[e]);
            const uint32_t end = static_cast<uint32_t>(offsets[e + 1]);
            const uint32_t m_tiles = math::ceil_div(end - begin, BLOCK_M);
            const uint32_t expert_tiles = m_tiles * n_tiles;
            if (local_tile < expert_tiles) {
                expert = e;
                expert_begin = begin;
                expert_end = end;
                break;
            }
            local_tile -= expert_tiles;
        }

        if (expert == num_groups)
            break;

        const uint32_t m_tile = local_tile / n_tiles;
        const uint32_t n_tile = local_tile - m_tile * n_tiles;
        const uint32_t row_begin = expert_begin + m_tile * BLOCK_M;
        const uint32_t col_begin = n_tile * BLOCK_N;

        float accum[WGMMA::kNumAccum] = {0.0f};

        auto load_stage = [&](const uint32_t k_block, const uint32_t stage) {
            const uint32_t k_begin = k_block * BLOCK_K;
            auto* smem_a = smem_a_base + stage * BLOCK_M * BLOCK_K;
            auto* smem_b_primary =
                smem_b_primary_base + stage * BLOCK_N * BLOCK_K;

            // Load A into the 128-byte-swizzled K-major layout expected by
            // Hopper WGMMA. Each iteration moves one aligned uint4.
            constexpr uint32_t kAChunks = kABytes / kVectorBytes;
            #pragma unroll
            for (uint32_t chunk = tid; chunk < kAChunks; chunk += kThreads) {
                const uint32_t row = chunk / (BLOCK_K / kVectorBytes);
                const uint32_t k_chunk =
                    (chunk % (BLOCK_K / kVectorBytes)) * kVectorBytes;
                const uint32_t physical_k =
                    k_chunk ^ ((row & 7u) << 4u);

                uint4 value = make_uint4(0u, 0u, 0u, 0u);
                const uint32_t global_row = row_begin + row;
                if (global_row < expert_end) {
                    const auto* src = reinterpret_cast<const uint4*>(
                        a + static_cast<size_t>(global_row) * shape_k + k_begin + k_chunk);
                    value = *src;
                }
                *reinterpret_cast<uint4*>(
                    smem_a + row * BLOCK_K + physical_k) = value;
            }

            // Threads cooperatively expand B rows. Canonical packing stores even K
            // in the low nibble and odd K in the high nibble. Each per-16
            // scale is applied before the expanded FP8 byte is written into
            // the same 128-byte swizzle used by WGMMA.
            for (uint32_t b_row = tid; b_row < BLOCK_N; b_row += kThreads) {
                const uint32_t global_n = col_begin + b_row;
                const auto* packed_row = w_packed +
                    (static_cast<size_t>(expert) * shape_n + global_n) * packed_k +
                    k_begin / 2;
                const auto* scale_row = block_scale +
                    (static_cast<size_t>(expert) * shape_n + global_n) * scale_k +
                    k_begin / 16;
                auto* dst_primary = reinterpret_cast<uint8_t*>(
                    smem_b_primary + b_row * BLOCK_K);
                const uint32_t row_swizzle = (b_row & 7u) << 4u;

                #pragma unroll
                for (uint32_t scale_idx = 0; scale_idx < BLOCK_K / 16; ++scale_idx) {
                    const uint32_t k_group = k_begin / 16 + scale_idx;
                    const uint8_t* packed_group = packed_row + scale_idx * 8u;
                    uint32_t scale = scale_row[scale_idx];
                    if constexpr (kPretransformedWeights) {
                        packed_group = w_packed +
                            ((static_cast<size_t>(expert) * scale_k + k_group) *
                             shape_n + global_n) * 8u;
                        scale = block_scale[
                            (static_cast<size_t>(expert) * scale_k + k_group) *
                            shape_n + global_n];
                    }
                    const uint32_t physical = scale_idx * 16u ^ row_swizzle;
                    const uint2 primary_lut = smem_primary_lut[scale & 0x7fu];
                    const auto* packed_words = reinterpret_cast<const uint32_t*>(
                        packed_group);
                    const uint2 q0 = nvfp4::dequant_nvfp4_to_fp8_pair_with_lut(
                        packed_words[0], primary_lut);
                    const uint2 q1 = nvfp4::dequant_nvfp4_to_fp8_pair_with_lut(
                        packed_words[1], primary_lut);

                    const uint4 expanded = make_uint4(
                        __byte_perm(q0.y, q0.x, 0x5140),
                        __byte_perm(q0.y, q0.x, 0x7362),
                        __byte_perm(q1.y, q1.x, 0x5140),
                        __byte_perm(q1.y, q1.x, 0x7362));
                    *reinterpret_cast<uint4*>(dst_primary + physical) = expanded;
                }
            }
        };

        load_stage(0, 0);
        __syncthreads();

        #pragma unroll 1
        for (uint32_t k_block = 0; k_block < k_blocks; ++k_block) {
            const uint32_t stage = k_block & 1u;
            auto* smem_a = smem_a_base + stage * BLOCK_M * BLOCK_K;
            auto* smem_b_primary =
                smem_b_primary_base + stage * BLOCK_N * BLOCK_K;
            auto a_desc = mma::sm90::make_smem_desc(smem_a, 1);
            auto b_primary_desc = mma::sm90::make_smem_desc(smem_b_primary, 1);
            const uint32_t a_desc_base = a_desc.reg32_[0];
            const uint32_t b_primary_desc_base = b_primary_desc.reg32_[0];
            #pragma unroll
            for (uint32_t i = 0; i < WGMMA::kNumAccum; ++i)
                ptx::warpgroup_fence_operand(accum[i]);
            ptx::warpgroup_arrive();
            #pragma unroll
            for (uint32_t k_inner = 0; k_inner < BLOCK_K / WGMMA::K; ++k_inner) {
                a_desc.reg32_[0] = a_desc_base + k_inner * WGMMA::K / 16;
                b_primary_desc.reg32_[0] =
                    b_primary_desc_base + k_inner * WGMMA::K / 16;
                WGMMA::wgmma(a_desc, b_primary_desc, accum,
                             k_block != 0 || k_inner != 0);
            }
            ptx::warpgroup_commit_batch();
            #pragma unroll
            for (uint32_t i = 0; i < WGMMA::kNumAccum; ++i)
                ptx::warpgroup_fence_operand(accum[i]);

            // WGMMA is asynchronous. Decode the next K slice into the other
            // stage while tensor cores consume the current stage.
            if (k_block + 1 < k_blocks) {
                load_stage(k_block + 1, stage ^ 1u);
                __syncthreads();
            }
            ptx::warpgroup_wait<0>();
        }

        // WGMMA distributes a 64x64 result as eight values per 16-column
        // group in every thread. Apply scales in FP32 and store only rows
        // belonging to this expert, which handles arbitrary unpadded tails.
        const uint32_t lane = tid & 31u;
        const uint32_t warp = tid >> 5u;
        const uint32_t local_row_0 = warp * 16u + lane / 4u;
        const uint32_t local_row_1 = local_row_0 + 8u;
        const uint32_t global_row_0 = row_begin + local_row_0;
        const uint32_t global_row_1 = row_begin + local_row_1;
        // Compensate for the /8 normalization used while expanding B to FP8.
        const float expert_scale = global_scale[expert] * 8.0f;
        const float row_scale_0 = global_row_0 < expert_end
            ? a_scale[global_row_0] * expert_scale : 0.0f;
        const float row_scale_1 = global_row_1 < expert_end
            ? a_scale[global_row_1] * expert_scale : 0.0f;

        #pragma unroll
        for (uint32_t group = 0; group < BLOCK_N / 16; ++group) {
            const uint32_t local_col_0 = group * 16u + 2u * (tid & 3u);
            const uint32_t local_col_1 = local_col_0 + 8u;
            const uint32_t base = group * 8u;

            if (global_row_0 < expert_end) {
                auto* row_ptr = d + static_cast<size_t>(global_row_0) * shape_n + col_begin;
                row_ptr[local_col_0] = __float2bfloat16_rn(accum[base + 0] * row_scale_0);
                row_ptr[local_col_0 + 1] = __float2bfloat16_rn(accum[base + 1] * row_scale_0);
                row_ptr[local_col_1] = __float2bfloat16_rn(accum[base + 4] * row_scale_0);
                row_ptr[local_col_1 + 1] = __float2bfloat16_rn(accum[base + 5] * row_scale_0);
            }
            if (global_row_1 < expert_end) {
                auto* row_ptr = d + static_cast<size_t>(global_row_1) * shape_n + col_begin;
                row_ptr[local_col_0] = __float2bfloat16_rn(accum[base + 2] * row_scale_1);
                row_ptr[local_col_0 + 1] = __float2bfloat16_rn(accum[base + 3] * row_scale_1);
                row_ptr[local_col_1] = __float2bfloat16_rn(accum[base + 6] * row_scale_1);
                row_ptr[local_col_1 + 1] = __float2bfloat16_rn(accum[base + 7] * row_scale_1);
            }
        }
        __syncthreads();
    }
#else
    if (blockIdx.x == 0 && threadIdx.x == 0)
        DG_DEVICE_ASSERT(false && "This kernel only supports sm_90a");
#endif
}

// Large-M dense throughput path. One producer decodes each packed N256 weight
// tile once while two consumers compute independent M64 tiles against it.
template <uint32_t BLOCK_N, bool kPretransformedWeights>
CUTLASS_GLOBAL __launch_bounds__(384, 1) void
sm90_nvfp4_grouped_gemm_dual_m_ws_impl(
        __nv_bfloat16* __restrict__ d,
        const __nv_fp8_e4m3* __restrict__ a,
        const float* __restrict__ a_scale,
        const uint8_t* __restrict__ w_packed,
        const uint8_t* __restrict__ block_scale,
        const float* __restrict__ global_scale,
        const int* __restrict__ offsets,
        uint32_t shape_m, uint32_t shape_n, uint32_t shape_k,
        uint32_t num_groups) {
#if (defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900)) || defined(__CLION_IDE__)
    constexpr uint32_t kBlockM = 64;
    constexpr uint32_t kBlockK = 128;
    constexpr uint32_t kNumConsumers = 2;
    constexpr uint32_t kBlockTokens = kBlockM * kNumConsumers;
    constexpr uint32_t kNumStages = 2;
    constexpr uint32_t kABytes = kBlockTokens * kBlockK;
    constexpr uint32_t kBBytes = BLOCK_N * kBlockK;
    using WGMMA = typename mma::sm90::FP8MMASelector<BLOCK_N>::type;
    using Barrier = cutlass::arch::ClusterTransactionBarrier;
    DG_STATIC_ASSERT(BLOCK_N == 256, "Dual-M dense path targets WGMMA N256");
    DG_STATIC_ASSERT(WGMMA::M == kBlockM && WGMMA::K == 32,
                     "Unexpected WGMMA instruction shape");

    extern __shared__ __align__(1024) uint8_t smem_buffer[];
    auto* smem_a_base = reinterpret_cast<__nv_fp8_e4m3*>(smem_buffer);
    auto* smem_b_base = reinterpret_cast<__nv_fp8_e4m3*>(
        smem_buffer + kNumStages * kABytes);
    auto* smem_primary_lut = reinterpret_cast<uint2*>(
        smem_buffer + kNumStages * (kABytes + kBBytes));
    auto* full_barriers = reinterpret_cast<Barrier*>(smem_primary_lut + 128);
    auto* empty_barriers = full_barriers + kNumStages;

    const uint32_t tid = threadIdx.x;
    const uint32_t wg_tid = tid & 127u;
    const uint32_t lane = wg_tid & 31u;
    const uint32_t n_tiles = shape_n / BLOCK_N;
    const uint32_t k_blocks = shape_k / kBlockK;
    const uint32_t packed_k = shape_k / 2;
    const uint32_t scale_k = shape_k / 16;
    (void)shape_m;

    if (tid < 64) {
        reinterpret_cast<uint4*>(smem_primary_lut)[tid] =
            reinterpret_cast<const uint4*>(
                nvfp4::kE2M1AndUe4m3ToFp8Lut)[tid];
    }
    if (tid == 0) {
        #pragma unroll
        for (uint32_t stage = 0; stage < kNumStages; ++stage) {
            full_barriers[stage].init(1 + kNumConsumers);
            empty_barriers[stage].init(4 * kNumConsumers);
        }
        cutlass::arch::fence_barrier_init();
    }
    __syncthreads();

    auto map_tile = [&](const uint32_t tile_idx,
                        uint32_t& expert, uint32_t& expert_begin,
                        uint32_t& expert_end, uint32_t& token_begin,
                        uint32_t& col_begin) {
        expert = num_groups;
        uint32_t local_tile = tile_idx;
        #pragma unroll 1
        for (uint32_t e = 0; e < num_groups; ++e) {
            const uint32_t begin = static_cast<uint32_t>(offsets[e]);
            const uint32_t end = static_cast<uint32_t>(offsets[e + 1]);
            const uint32_t token_tiles =
                math::ceil_div(end - begin, kBlockTokens);
            const uint32_t expert_tiles = token_tiles * n_tiles;
            if (local_tile < expert_tiles) {
                expert = e;
                expert_begin = begin;
                expert_end = end;
                const uint32_t token_tile = local_tile / n_tiles;
                const uint32_t n_tile = local_tile - token_tile * n_tiles;
                token_begin = begin + token_tile * kBlockTokens;
                col_begin = n_tile * BLOCK_N;
                return true;
            }
            local_tile -= expert_tiles;
        }
        return false;
    };

    uint32_t stage_idx = 0;
    uint32_t phase = 0;
    auto advance_pipeline = [&]() {
        stage_idx = stage_idx == kNumStages - 1 ? 0 : stage_idx + 1;
        phase ^= stage_idx == 0;
    };

    if (tid >= 128u * kNumConsumers) {
        for (uint32_t tile_idx = blockIdx.x;; tile_idx += gridDim.x) {
            uint32_t expert, expert_begin, expert_end, token_begin, col_begin;
            if (!map_tile(tile_idx, expert, expert_begin, expert_end,
                          token_begin, col_begin))
                break;
            (void)expert_begin;
            (void)expert_end;
            (void)token_begin;

            for (uint32_t k_block = 0; k_block < k_blocks; ++k_block) {
                if (wg_tid == 0)
                    empty_barriers[stage_idx].wait(phase ^ 1u);
                ptx::sync_aligned(128, 0);

                const uint32_t k_begin = k_block * kBlockK;
                auto* smem_b =
                    smem_b_base + stage_idx * BLOCK_N * kBlockK;
                for (uint32_t b_row = wg_tid; b_row < BLOCK_N;
                     b_row += 128u) {
                    const uint32_t global_n = col_begin + b_row;
                    const auto* packed_row = w_packed +
                        (static_cast<size_t>(expert) * shape_n + global_n) *
                            packed_k + k_begin / 2;
                    const auto* scale_row = block_scale +
                        (static_cast<size_t>(expert) * shape_n + global_n) *
                            scale_k + k_begin / 16;
                    auto* dst = reinterpret_cast<uint8_t*>(
                        smem_b + b_row * kBlockK);
                    const uint32_t row_swizzle = (b_row & 7u) << 4u;

                    #pragma unroll
                    for (uint32_t scale_idx = 0;
                         scale_idx < kBlockK / 16; ++scale_idx) {
                        const uint32_t k_group = k_begin / 16 + scale_idx;
                        const uint8_t* packed_group =
                            packed_row + scale_idx * 8u;
                        uint32_t scale = scale_row[scale_idx];
                        if constexpr (kPretransformedWeights) {
                            packed_group = w_packed +
                                ((static_cast<size_t>(expert) * scale_k +
                                  k_group) * shape_n + global_n) * 8u;
                            scale = block_scale[
                                (static_cast<size_t>(expert) * scale_k +
                                 k_group) * shape_n + global_n];
                        }
                        const uint32_t physical =
                            scale_idx * 16u ^ row_swizzle;
                        const uint2 lut =
                            smem_primary_lut[scale & 0x7fu];
                        const auto* packed_words =
                            reinterpret_cast<const uint32_t*>(packed_group);
                        const uint2 q0 =
                            nvfp4::dequant_nvfp4_to_fp8_pair_with_lut(
                                packed_words[0], lut);
                        const uint2 q1 =
                            nvfp4::dequant_nvfp4_to_fp8_pair_with_lut(
                                packed_words[1], lut);
                        *reinterpret_cast<uint4*>(dst + physical) = make_uint4(
                            __byte_perm(q0.y, q0.x, 0x5140),
                            __byte_perm(q0.y, q0.x, 0x7362),
                            __byte_perm(q1.y, q1.x, 0x5140),
                            __byte_perm(q1.y, q1.x, 0x7362));
                    }
                }

                cutlass::arch::fence_view_async_shared();
                ptx::sync_aligned(128, 0);
                if (wg_tid == 0)
                    full_barriers[stage_idx].arrive();
                advance_pipeline();
            }
        }
        return;
    }

    const uint32_t consumer_idx = tid / 128u;
    for (uint32_t tile_idx = blockIdx.x;; tile_idx += gridDim.x) {
        uint32_t expert, expert_begin, expert_end, token_begin, col_begin;
        if (!map_tile(tile_idx, expert, expert_begin, expert_end,
                      token_begin, col_begin))
            break;
        (void)expert_begin;

        float accum[WGMMA::kNumAccum] = {0.0f};
        auto load_activation_stage = [&] (
                const uint32_t k_block, const uint32_t load_stage_idx,
                const uint32_t load_phase) {
            if (wg_tid == 0)
                empty_barriers[load_stage_idx].wait(load_phase ^ 1u);
            ptx::sync_aligned(128, 1u + consumer_idx);

            auto* smem_a = smem_a_base +
                load_stage_idx * kABytes +
                consumer_idx * kBlockM * kBlockK;
            constexpr uint32_t kChunksPerRow = kBlockK / sizeof(uint4);
            const uint32_t k_begin = k_block * kBlockK;
            for (uint32_t chunk = wg_tid;
                 chunk < kBlockM * kChunksPerRow; chunk += 128u) {
                const uint32_t local_row = chunk / kChunksPerRow;
                const uint32_t k_chunk =
                    (chunk % kChunksPerRow) * sizeof(uint4);
                const uint32_t global_row = token_begin +
                    consumer_idx * kBlockM + local_row;
                uint4 value = make_uint4(0u, 0u, 0u, 0u);
                if (global_row < expert_end)
                    value = *reinterpret_cast<const uint4*>(
                        a + static_cast<size_t>(global_row) * shape_k +
                        k_begin + k_chunk);
                const uint32_t physical_k =
                    k_chunk ^ ((local_row & 7u) << 4u);
                *reinterpret_cast<uint4*>(
                    smem_a + local_row * kBlockK + physical_k) = value;
            }
            cutlass::arch::fence_view_async_shared();
            ptx::sync_aligned(128, 1u + consumer_idx);
            if (wg_tid == 0)
                full_barriers[load_stage_idx].arrive();
        };

        load_activation_stage(0, stage_idx, phase);
        for (uint32_t k_block = 0; k_block < k_blocks; ++k_block) {
            full_barriers[stage_idx].wait(phase);
            auto* smem_a = smem_a_base + stage_idx * kABytes +
                consumer_idx * kBlockM * kBlockK;
            auto* smem_b =
                smem_b_base + stage_idx * BLOCK_N * kBlockK;
            auto a_desc = mma::sm90::make_smem_desc(smem_a, 1);
            auto b_desc = mma::sm90::make_smem_desc(smem_b, 1);
            const uint32_t a_desc_base = a_desc.reg32_[0];
            const uint32_t b_desc_base = b_desc.reg32_[0];

            #pragma unroll
            for (uint32_t i = 0; i < WGMMA::kNumAccum; ++i)
                ptx::warpgroup_fence_operand(accum[i]);
            ptx::warpgroup_arrive();
            #pragma unroll
            for (uint32_t k_inner = 0;
                 k_inner < kBlockK / WGMMA::K; ++k_inner) {
                a_desc.reg32_[0] =
                    a_desc_base + k_inner * WGMMA::K / 16;
                b_desc.reg32_[0] =
                    b_desc_base + k_inner * WGMMA::K / 16;
                WGMMA::wgmma(a_desc, b_desc, accum,
                             k_block != 0 || k_inner != 0);
            }
            ptx::warpgroup_commit_batch();
            if (k_block + 1u < k_blocks) {
                const uint32_t next_stage_idx =
                    stage_idx == kNumStages - 1 ? 0 : stage_idx + 1;
                const uint32_t next_phase =
                    phase ^ static_cast<uint32_t>(next_stage_idx == 0);
                load_activation_stage(
                    k_block + 1u, next_stage_idx, next_phase);
            }
            ptx::warpgroup_wait<0>();
            #pragma unroll
            for (uint32_t i = 0; i < WGMMA::kNumAccum; ++i)
                ptx::warpgroup_fence_operand(accum[i]);
            if (lane == 0)
                empty_barriers[stage_idx].arrive();
            advance_pipeline();
        }

        const uint32_t warp = wg_tid >> 5u;
        const uint32_t local_row_0 = warp * 16u + lane / 4u;
        const uint32_t local_row_1 = local_row_0 + 8u;
        const uint32_t global_row_0 = token_begin +
            consumer_idx * kBlockM + local_row_0;
        const uint32_t global_row_1 = token_begin +
            consumer_idx * kBlockM + local_row_1;
        const float expert_scale = global_scale[expert] * 8.0f;
        const float row_scale_0 = global_row_0 < expert_end
            ? a_scale[global_row_0] * expert_scale : 0.0f;
        const float row_scale_1 = global_row_1 < expert_end
            ? a_scale[global_row_1] * expert_scale : 0.0f;

        #pragma unroll
        for (uint32_t group = 0; group < BLOCK_N / 16; ++group) {
            const uint32_t local_col_0 =
                group * 16u + 2u * (wg_tid & 3u);
            const uint32_t local_col_1 = local_col_0 + 8u;
            const uint32_t base = group * 8u;
            if (global_row_0 < expert_end) {
                auto* row_ptr = d +
                    static_cast<size_t>(global_row_0) * shape_n + col_begin;
                row_ptr[local_col_0] =
                    __float2bfloat16_rn(accum[base + 0] * row_scale_0);
                row_ptr[local_col_0 + 1] =
                    __float2bfloat16_rn(accum[base + 1] * row_scale_0);
                row_ptr[local_col_1] =
                    __float2bfloat16_rn(accum[base + 4] * row_scale_0);
                row_ptr[local_col_1 + 1] =
                    __float2bfloat16_rn(accum[base + 5] * row_scale_0);
            }
            if (global_row_1 < expert_end) {
                auto* row_ptr = d +
                    static_cast<size_t>(global_row_1) * shape_n + col_begin;
                row_ptr[local_col_0] =
                    __float2bfloat16_rn(accum[base + 2] * row_scale_1);
                row_ptr[local_col_0 + 1] =
                    __float2bfloat16_rn(accum[base + 3] * row_scale_1);
                row_ptr[local_col_1] =
                    __float2bfloat16_rn(accum[base + 6] * row_scale_1);
                row_ptr[local_col_1 + 1] =
                    __float2bfloat16_rn(accum[base + 7] * row_scale_1);
            }
        }
    }
#else
    if (blockIdx.x == 0 && threadIdx.x == 0)
        DG_DEVICE_ASSERT(false && "This kernel only supports sm_90a");
#endif
}

}  // namespace deep_gemm

#pragma clang diagnostic pop
