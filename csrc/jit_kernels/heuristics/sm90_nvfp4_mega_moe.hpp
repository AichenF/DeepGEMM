#pragma once

#include <deep_gemm/layout/mega_moe.cuh>

#include "../../utils/exception.hpp"
#include "sm90.hpp"

namespace deep_gemm {

static constexpr int kSM90NVFP4BStoragePerKBlock = 80;

struct SM90NVFP4H200MimoFusedConfig {
    static constexpr int kBlockN = 256;
    static constexpr int kBlockK = 128;
    static constexpr int kSwizzleActsMode = 128;
    static constexpr int kNumDispatchThreads = 64;
    static constexpr int kNumNonEpilogueThreads = 64;
    static constexpr int kNumEpilogueThreads = 256;
    static constexpr int kNumThreads =
        kNumDispatchThreads + kNumNonEpilogueThreads + kNumEpilogueThreads;

    int block_m;
    int num_max_pool_tokens;
    int num_padded_sf_pool_tokens;
    int num_experts_per_wave;
    int num_stages, smem_size;
};

struct SM90NVFP4H200MimoFusedShape {
    static constexpr int kH200NumSMs = 132;
    static constexpr int kMimoNumRanks = 8;
    static constexpr int kMimoExpertsPerRank = 48;
    static constexpr int kMimoTopk = 8;
    static constexpr int kMimoHidden = 6144;
    static constexpr int kMimoIntermediateHidden = 2048;

    int num_sms;
    int num_ranks;
    int num_experts;
    int num_topk;
    int hidden;
    int intermediate_hidden;

    static constexpr bool is_supported_batch(const int num_tokens) noexcept {
        return num_tokens > 0;
    }

    constexpr bool is_h200_mimo_pro() const noexcept {
        return num_sms == kH200NumSMs &&
            num_ranks == kMimoNumRanks &&
            num_experts == kMimoExpertsPerRank * kMimoNumRanks &&
            num_topk == kMimoTopk &&
            hidden == kMimoHidden &&
            intermediate_hidden == kMimoIntermediateHidden;
    }
};

struct SM90NVFP4H200MimoFusedInput {
    int num_sms;
    int num_ranks, num_experts, num_experts_per_rank;
    int num_max_tokens_per_rank, num_tokens, num_topk;
    int hidden, intermediate_hidden;
    int num_padded_sf_pool_tokens;

    SM90NVFP4H200MimoFusedShape shape() const noexcept {
        return {
            num_sms, num_ranks, num_experts, num_topk,
            hidden, intermediate_hidden};
    }
};

struct SM90NVFP4H200MimoFusedPlan {
    SM90NVFP4H200MimoFusedConfig config;
    bool swap_ab;
    bool use_mode2_row_decoder;
    bool single_active_dispatch_warp;
};

static SM90NVFP4H200MimoFusedPlan
select_sm90_nvfp4_h200_mimo_fused(
        const SM90NVFP4H200MimoFusedInput& input) {
    DG_HOST_ASSERT(input.shape().is_h200_mimo_pro());
    DG_HOST_ASSERT(
        input.num_experts_per_rank ==
        SM90NVFP4H200MimoFusedShape::kMimoExpertsPerRank);
    DG_HOST_ASSERT(input.num_experts ==
                   input.num_experts_per_rank * input.num_ranks);
    DG_HOST_ASSERT(input.num_max_tokens_per_rank > 0);
    DG_HOST_ASSERT(input.num_tokens <= input.num_max_tokens_per_rank);
    DG_HOST_ASSERT(
        SM90NVFP4H200MimoFusedShape::is_supported_batch(input.num_tokens));
    DG_HOST_ASSERT(input.num_padded_sf_pool_tokens > 0);

    struct Tuning {
        int block_m;
        int num_experts_per_wave;
        int num_stages;
        int smem_size;
        bool swap_ab;
        bool use_mode2_row_decoder;
        bool single_active_dispatch_warp;
    } tuning {};

    if (input.num_tokens <= 8)
        tuning = {8, 16, 4, SM90ArchSpec::smem_capacity,
                  true, true, true};
    else if (input.num_tokens <= 16)
        tuning = {8, 24, 4, SM90ArchSpec::smem_capacity,
                  true, true, true};
    else if (input.num_tokens <= 32)
        tuning = {16, 48, 3, SM90ArchSpec::smem_capacity,
                  true, true, false};
    else if (input.num_tokens <= 64)
        tuning = {24, 48, 3, 229312,
                  true, false, true};
    else
        tuning = {64, 48, 3, 209856,
                  false, true, false};

    DG_HOST_ASSERT(
        input.num_experts_per_rank % tuning.num_experts_per_wave == 0);
    DG_HOST_ASSERT(tuning.smem_size <= SM90ArchSpec::smem_capacity);
    return {
        {
            tuning.block_m,
            layout::get_num_max_pool_tokens(
                input.num_ranks, input.num_max_tokens_per_rank,
                input.num_topk, input.num_experts_per_rank),
            input.num_padded_sf_pool_tokens,
            tuning.num_experts_per_wave,
            tuning.num_stages,
            tuning.smem_size,
        },
        tuning.swap_ab,
        tuning.use_mode2_row_decoder,
        tuning.single_active_dispatch_warp,
    };
}

}  // namespace deep_gemm
