#pragma once

#include <cstdint>

#include <deep_gemm/layout/mega_moe.cuh>

#include "../../utils/exception.hpp"
#include "sm90.hpp"

namespace deep_gemm {

struct SM90NVFP4SmallMConfig {
    static constexpr int kBlockN = 256;
    static constexpr int kBlockK = 128;
    static constexpr int kWeightStoragePerKBlock = 80;
    static constexpr int kSwizzleActsMode = 128;
    static constexpr int kClusterSize = 1;
    static constexpr int kNumDispatchThreads = 64;
    static constexpr int kNumNonEpilogueThreads = 64;
    static constexpr int kNumEpilogueThreads = 256;
    static constexpr int kNumThreads =
        kNumDispatchThreads + kNumNonEpilogueThreads + kNumEpilogueThreads;

    int block_m;
    int num_max_pool_tokens;
    int num_padded_sf_pool_tokens;
    int num_experts_per_wave;
    int num_stages;
    int smem_size;
};

struct SM90NVFP4SmallMInput {
    int num_ranks;
    int num_experts;
    int num_experts_per_rank;
    int num_max_tokens_per_rank;
    int num_tokens;
    int num_topk;
    int hidden;
    int intermediate_hidden;
    int num_padded_sf_pool_tokens;
};

struct SM90NVFP4SmallMPlan {
    SM90NVFP4SmallMConfig config;
    bool swap_ab;
    bool use_mode2_row_decoder;
    bool single_active_dispatch_warp;
};

static int get_sm90_nvfp4_wave_divisor(
        const int num_experts_per_rank, const int target) {
    int wave = target < num_experts_per_rank ?
        target : num_experts_per_rank;
    while (wave < num_experts_per_rank &&
           num_experts_per_rank % wave != 0)
        ++wave;
    return wave;
}

static int get_sm90_nvfp4_small_m_smem_size(
        const SM90NVFP4SmallMInput& input,
        const int block_m,
        const int num_stages,
        const bool swap_ab,
        const bool single_active_dispatch_warp) {
    const auto align_up = [](const int value, const int alignment) {
        return ((value + alignment - 1) / alignment) * alignment;
    };
    constexpr int kSmemAlignment = 1024;
    constexpr int kNumDispatchWarps =
        SM90NVFP4SmallMConfig::kNumDispatchThreads / 32;
    constexpr int kNumEpilogueWarps =
        SM90NVFP4SmallMConfig::kNumEpilogueThreads / 32;
    constexpr int kNumEpilogueWarpgroups = kNumEpilogueWarps / 4;
    const int num_active_dispatch_warps =
        single_active_dispatch_warp ? 1 : kNumDispatchWarps;

    const int smem_expert_count = align_up(
        input.num_experts * static_cast<int>(sizeof(uint32_t)),
        kSmemAlignment);
    const int smem_send_buffers = align_up(
        input.hidden * num_active_dispatch_warps, kSmemAlignment);
    constexpr int kSmemNVFP4LUT = 1024;

    const int smem_cd_l1 =
        kNumEpilogueWarpgroups * block_m * 64;
    const int smem_cd_l2 =
        swap_ab ? block_m * SM90NVFP4SmallMConfig::kBlockN * 2 : 0;
    const int smem_cd_sf_slots =
        kNumEpilogueWarpgroups * block_m;
    const int smem_cd_swap_slots =
        swap_ab ? block_m * kNumEpilogueWarps : 0;
    const int smem_cd_extra =
        (smem_cd_sf_slots > smem_cd_swap_slots ?
         smem_cd_sf_slots : smem_cd_swap_slots) *
        static_cast<int>(sizeof(float));
    const int smem_cd = align_up(
        (smem_cd_l1 > smem_cd_l2 ? smem_cd_l1 : smem_cd_l2) +
        smem_cd_extra,
        kSmemAlignment);

    const int smem_sfa_per_stage = align_up(
        block_m * static_cast<int>(sizeof(float)), 128);
    const int smem_per_stage =
        block_m * SM90NVFP4SmallMConfig::kBlockK +
        SM90NVFP4SmallMConfig::kBlockN *
            SM90NVFP4SmallMConfig::kBlockK +
        SM90NVFP4SmallMConfig::kBlockN *
            SM90NVFP4SmallMConfig::kWeightStoragePerKBlock +
        smem_sfa_per_stage;
    const int num_barriers =
        kNumDispatchWarps + 2 * num_stages + 2 * kNumEpilogueWarps;
    const int smem_barriers =
        num_barriers * static_cast<int>(sizeof(uint64_t));
    return align_up(
        smem_expert_count + smem_send_buffers + kSmemNVFP4LUT +
        smem_cd + num_stages * smem_per_stage + smem_barriers,
        256);
}

static SM90NVFP4SmallMPlan select_sm90_nvfp4_small_m(
        const SM90NVFP4SmallMInput& input) {
    DG_HOST_ASSERT(input.num_ranks > 0);
    DG_HOST_ASSERT(input.num_experts_per_rank > 0);
    DG_HOST_ASSERT(
        input.num_experts ==
        input.num_experts_per_rank * input.num_ranks);
    DG_HOST_ASSERT(input.num_max_tokens_per_rank > 0);
    DG_HOST_ASSERT(
        input.num_tokens > 0 &&
        input.num_tokens <= input.num_max_tokens_per_rank);
    DG_HOST_ASSERT(input.num_topk > 0 && input.num_topk <= 32);
    DG_HOST_ASSERT(
        input.hidden % SM90NVFP4SmallMConfig::kBlockN == 0);
    DG_HOST_ASSERT(
        (2 * input.intermediate_hidden) %
            SM90NVFP4SmallMConfig::kBlockN == 0);
    DG_HOST_ASSERT(
        input.hidden % SM90NVFP4SmallMConfig::kBlockK == 0);
    DG_HOST_ASSERT(
        input.intermediate_hidden %
            SM90NVFP4SmallMConfig::kBlockK == 0);
    DG_HOST_ASSERT(input.num_padded_sf_pool_tokens > 0);

    const int64_t routed_tokens =
        static_cast<int64_t>(input.num_tokens) * input.num_topk;
    const int64_t local_experts = input.num_experts_per_rank;
    const bool load_at_most_1p5 =
        2 * routed_tokens <= 3 * local_experts;
    const bool load_at_most_3 =
        routed_tokens <= 3 * local_experts;
    const bool load_at_most_6 =
        routed_tokens <= 6 * local_experts;
    const bool load_at_most_12 =
        routed_tokens <= 12 * local_experts;

    const int block_m =
        load_at_most_3 ? 8 :
        load_at_most_6 ? 16 :
        load_at_most_12 ? 24 : 64;
    const int target_experts_per_wave =
        load_at_most_1p5 ? 16 :
        load_at_most_3 ? 24 :
        input.num_experts_per_rank;
    const int num_experts_per_wave = get_sm90_nvfp4_wave_divisor(
        input.num_experts_per_rank, target_experts_per_wave);
    int num_stages = block_m == 8 ? 4 : 3;
    const bool swap_ab = load_at_most_12;
    const bool use_mode2_row_decoder =
        load_at_most_6 || !load_at_most_12;
    const bool single_active_dispatch_warp =
        load_at_most_3 || (load_at_most_12 && !load_at_most_6);
    int smem_size = get_sm90_nvfp4_small_m_smem_size(
        input, block_m, num_stages, swap_ab,
        single_active_dispatch_warp);
    if (num_stages == 4 && smem_size > SM90ArchSpec::smem_capacity) {
        num_stages = 3;
        smem_size = get_sm90_nvfp4_small_m_smem_size(
            input, block_m, num_stages, swap_ab,
            single_active_dispatch_warp);
    }

    DG_HOST_ASSERT(
        input.num_experts_per_rank % num_experts_per_wave == 0);
    DG_HOST_ASSERT(smem_size <= SM90ArchSpec::smem_capacity);
    return {
        {
            block_m,
            layout::get_num_max_pool_tokens(
                input.num_ranks,
                input.num_max_tokens_per_rank,
                input.num_topk,
                input.num_experts_per_rank),
            input.num_padded_sf_pool_tokens,
            num_experts_per_wave,
            num_stages,
            smem_size,
        },
        swap_ab,
        use_mode2_row_decoder,
        single_active_dispatch_warp,
    };
}

}  // namespace deep_gemm
