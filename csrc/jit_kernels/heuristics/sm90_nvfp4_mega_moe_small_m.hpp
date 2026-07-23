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

struct SM90NVFP4SmallMLoad {
    int64_t routed_tokens;
    int64_t local_experts;

    bool valid() const {
        return routed_tokens > 0 and local_experts > 0;
    }

    bool less_equal(const int64_t value) const {
        return routed_tokens <= value * local_experts;
    }

    bool less_equal_ratio(
            const int64_t numerator,
            const int64_t denominator) const {
        return denominator * routed_tokens <= numerator * local_experts;
    }
};

struct SM90NVFP4SmallMScheduleTuning {
    int block_m;
    int target_num_experts_per_wave;
    int num_stages;
    bool swap_ab;
    bool use_mode2_row_decoder;
    bool single_active_dispatch_warp;
};

struct SM90NVFP4SmallMPlan {
    SM90NVFP4SmallMConfig config;
    bool swap_ab;
    bool use_mode2_row_decoder;
    bool single_active_dispatch_warp;
};

static SM90NVFP4SmallMLoad get_sm90_nvfp4_small_m_load(
        const SM90NVFP4SmallMInput& input) {
    const SM90NVFP4SmallMLoad load {
        static_cast<int64_t>(input.num_tokens) * input.num_topk,
        input.num_experts_per_rank,
    };
    DG_HOST_ASSERT(load.valid());
    return load;
}

static int derive_sm90_nvfp4_small_m_epw(
        const int num_experts_per_rank, const int target) {
    int wave = target < num_experts_per_rank ?
        target : num_experts_per_rank;
    while (wave < num_experts_per_rank &&
           num_experts_per_rank % wave != 0)
        ++wave;
    return wave;
}

static SM90NVFP4SmallMScheduleTuning
select_sm90_nvfp4_small_m_tuning(const SM90NVFP4SmallMLoad& load) {
    DG_HOST_ASSERT(load.valid());
    const int all_local_experts = static_cast<int>(load.local_experts);

    if (load.less_equal(3)) {
        return {
            8,
            load.less_equal_ratio(3, 2) ? 16 : 24,
            4,
            true,
            true,
            true,
        };
    }
    if (load.less_equal(6)) {
        return {
            16,
            all_local_experts,
            3,
            true,
            true,
            false,
        };
    }
    if (load.less_equal(12)) {
        return {
            24,
            all_local_experts,
            3,
            true,
            false,
            true,
        };
    }
    if (load.less_equal(18)) {
        return {
            24,
            all_local_experts,
            3,
            true,
            true,
            false,
        };
    }
    if (load.less_equal(24)) {
        return {
            64,
            16,
            3,
            false,
            true,
            false,
        };
    }
    return {
        64,
        all_local_experts,
        3,
        false,
        true,
        false,
    };
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

static SM90NVFP4SmallMPlan materialize_sm90_nvfp4_small_m_tuning(
        const SM90NVFP4SmallMInput& input,
        const SM90NVFP4SmallMScheduleTuning& tuning) {
    const int num_experts_per_wave = derive_sm90_nvfp4_small_m_epw(
        input.num_experts_per_rank,
        tuning.target_num_experts_per_wave);
    int num_stages = tuning.num_stages;
    int smem_size = get_sm90_nvfp4_small_m_smem_size(
        input, tuning.block_m, num_stages, tuning.swap_ab,
        tuning.single_active_dispatch_warp);
    if (num_stages == 4 && smem_size > SM90ArchSpec::smem_capacity) {
        num_stages = 3;
        smem_size = get_sm90_nvfp4_small_m_smem_size(
            input, tuning.block_m, num_stages, tuning.swap_ab,
            tuning.single_active_dispatch_warp);
    }

    return {
        {
            tuning.block_m,
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
        tuning.swap_ab,
        tuning.use_mode2_row_decoder,
        tuning.single_active_dispatch_warp,
    };
}

static bool is_sm90_nvfp4_small_m_plan_legal(
        const SM90NVFP4SmallMInput& input,
        const SM90NVFP4SmallMPlan& plan) {
    const auto& config = plan.config;
    const bool supported_block_m =
        config.block_m == 8 || config.block_m == 16 ||
        config.block_m == 24 || config.block_m == 64;
    return supported_block_m &&
        config.num_experts_per_wave > 0 &&
        config.num_experts_per_wave <= input.num_experts_per_rank &&
        input.num_experts_per_rank % config.num_experts_per_wave == 0 &&
        config.num_stages >= 3 && config.num_stages <= 4 &&
        config.smem_size > 0 &&
        config.smem_size <= SM90ArchSpec::smem_capacity;
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

    const auto load = get_sm90_nvfp4_small_m_load(input);
    const auto tuning = select_sm90_nvfp4_small_m_tuning(load);
    const auto plan = materialize_sm90_nvfp4_small_m_tuning(input, tuning);
    DG_HOST_ASSERT(is_sm90_nvfp4_small_m_plan_legal(input, plan));
    return plan;
}

}  // namespace deep_gemm
