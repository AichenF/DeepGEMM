#pragma once

#include <cstdlib>

#include <deep_gemm/layout/mega_moe.cuh>

#include "../../utils/exception.hpp"
#include "sm90.hpp"

namespace deep_gemm {

static constexpr int kSM90MXFP4BStoragePerKBlock = 80;

struct SM90MXFP4H200FusedConfig {
    static constexpr int kBlockK = 128;
    static constexpr int kSwizzleActsMode = 128;
    static constexpr int kNumDispatchThreads = 64;
    static constexpr int kNumNonEpilogueThreads = 64;
    static constexpr int kNumEpilogueThreads = 256;
    static constexpr int kNumThreads =
        kNumDispatchThreads + kNumNonEpilogueThreads + kNumEpilogueThreads;

    int block_m, block_n;
    int num_max_pool_tokens;
    int num_padded_sf_pool_tokens;
    int num_experts_per_wave;
    int num_stages, smem_size;
};

struct SM90MXFP4H200FusedShape {
    // kNumRanks is still fixed: the kernel takes layout::SymBuffer<8> and the
    // nvlink barrier instantiations are sized for 8 peers.
    static constexpr int kNumRanks = 8;

    int num_sms;
    int num_ranks;
    int num_experts;
    int num_topk;
    int hidden;
    int intermediate_hidden;

    static constexpr bool is_supported_batch(const int num_tokens) noexcept {
        return num_tokens > 0;
    }

    // Shape used to be pinned to the H200 384-expert / 6144-hidden model.
    // kNumSMs became a kernel template parameter in 1b23095, and hidden /
    // intermediate_hidden / num_experts / num_topk followed, so the only
    // constraints left are the ones the kernel body genuinely needs:
    //   - 8 ranks (SymBuffer<8>)
    //   - experts divide evenly across ranks
    //   - hidden and intermediate_hidden are whole BLOCK_K (128) tiles
    //   - topk fits in one warp (the dispatch loop maps lanes to topk slots)
    // This admits DeepSeek-V4-Flash (4096 / 2048 / 256 experts / topk 6).
    constexpr bool is_supported_shape() const noexcept {
        return num_sms > 0 &&
            num_ranks == kNumRanks &&
            num_experts > 0 &&
            num_experts % kNumRanks == 0 &&
            num_topk > 0 && num_topk <= 32 &&
            hidden > 0 && hidden % 128 == 0 &&
            intermediate_hidden > 0 && intermediate_hidden % 128 == 0;
    }

    constexpr int experts_per_rank() const noexcept {
        return num_experts / kNumRanks;
    }
};

// num_experts_per_wave must divide num_experts_per_rank exactly (the kernel
// walks experts in whole waves). The shipped table was written for 48
// experts/rank, where 48/24/16 are all valid; DeepSeek-V4-Flash has 32, where
// 48 and 24 are not. Clamp to the largest divisor <= the requested value so a
// table entry stays meaningful across expert counts.
static constexpr int largest_divisor_at_most(const int n, const int cap) noexcept {
    for (int d = (cap < n ? cap : n); d >= 1; --d) {
        if (n % d == 0)
            return d;
    }
    return 1;
}

struct SM90MXFP4H200FusedInput {
    int num_sms;
    int num_ranks, num_experts, num_experts_per_rank;
    int num_max_tokens_per_rank, num_tokens, num_topk;
    int hidden, intermediate_hidden;
    int num_padded_sf_pool_tokens;

    SM90MXFP4H200FusedShape shape() const noexcept {
        return {
            num_sms, num_ranks, num_experts, num_topk,
            hidden, intermediate_hidden};
    }
};

struct SM90MXFP4H200FusedPlan {
    SM90MXFP4H200FusedConfig config;
    bool swap_ab;
    bool use_mode2_row_decoder;
    bool single_active_dispatch_warp;
    bool use_interleaved_scheduler;
};

static SM90MXFP4H200FusedPlan
select_sm90_mxfp4_h200_fused(
        const SM90MXFP4H200FusedInput& input) {
    DG_HOST_ASSERT(input.shape().is_supported_shape());
    DG_HOST_ASSERT(input.num_experts ==
                   input.num_experts_per_rank * input.num_ranks);
    DG_HOST_ASSERT(input.num_max_tokens_per_rank > 0);
    DG_HOST_ASSERT(input.num_tokens <= input.num_max_tokens_per_rank);
    DG_HOST_ASSERT(
        SM90MXFP4H200FusedShape::is_supported_batch(input.num_tokens));
    DG_HOST_ASSERT(input.num_padded_sf_pool_tokens > 0);

    struct Tuning {
        int block_m, block_n;
        int num_experts_per_wave;
        int num_stages;
        int smem_size;
        bool swap_ab;
        bool use_mode2_row_decoder;
        bool single_active_dispatch_warp;
    } tuning {};

    if (input.num_tokens <= 1)
        tuning = {8, 256, 24, 4, SM90ArchSpec::smem_capacity,
                  true, true, true};
    else if (input.num_tokens <= 8)
        tuning = {8, 256, 16, 4, SM90ArchSpec::smem_capacity,
                  true, true, true};
    else if (input.num_tokens <= 16)
        tuning = {8, 256, 24, 4, SM90ArchSpec::smem_capacity,
                  true, true, true};
    else if (input.num_tokens <= 32)
        tuning = {16, 256, 48, 3, SM90ArchSpec::smem_capacity,
                  true, true, false};
    else if (input.num_tokens <= 64)
        tuning = {24, 256, 48, 3, 229312,
                  true, false, true};
    else if (input.num_tokens <= 256)
        tuning = {64, 256, 48, 3, 209856,
                  false, true, false};
    else
        tuning = {128, 128, 48, 6, SM90ArchSpec::smem_capacity,
                  false, true, false};

    // Tuning override hook. The table above was measured on H200's 132 SMs;
    // H20 has 78, so the tiers have to be re-swept there. Reading the knobs
    // from the environment lets one build serve a whole sweep instead of
    // recompiling this header per candidate. Unset vars keep the table value.
    auto env_int = [](const char* name, int fallback) {
        const char* v = std::getenv(name);
        if (v == nullptr || *v == '\0')
            return fallback;
        return std::atoi(v);
    };
    tuning.block_m = env_int("DG_MXFP4_BLOCK_M", tuning.block_m);
    tuning.block_n = env_int("DG_MXFP4_BLOCK_N", tuning.block_n);
    tuning.num_experts_per_wave =
        env_int("DG_MXFP4_EPW", tuning.num_experts_per_wave);
    tuning.num_stages = env_int("DG_MXFP4_STAGES", tuning.num_stages);
    tuning.smem_size = env_int("DG_MXFP4_SMEM", tuning.smem_size);
    tuning.swap_ab = env_int("DG_MXFP4_SWAP_AB", tuning.swap_ab ? 1 : 0) != 0;
    tuning.use_mode2_row_decoder =
        env_int("DG_MXFP4_MODE2_ROW", tuning.use_mode2_row_decoder ? 1 : 0) != 0;
    tuning.single_active_dispatch_warp =
        env_int("DG_MXFP4_SINGLE_DISPATCH",
                tuning.single_active_dispatch_warp ? 1 : 0) != 0;

    tuning.num_experts_per_wave = largest_divisor_at_most(
        input.num_experts_per_rank, tuning.num_experts_per_wave);
    DG_HOST_ASSERT(
        input.num_experts_per_rank % tuning.num_experts_per_wave == 0);
    DG_HOST_ASSERT(tuning.smem_size <= SM90ArchSpec::smem_capacity);
    return {
        {
            tuning.block_m,
            tuning.block_n,
            layout::get_num_max_pool_tokens(
                input.num_ranks, input.num_max_tokens_per_rank,
                input.num_topk, input.num_experts_per_rank),
            input.num_padded_sf_pool_tokens,
            tuning.num_experts_per_wave,
            tuning.num_stages,
            cute::min(tuning.smem_size +
                          layout::kSM90InterleavedSchedulerSMEMBytes,
                      SM90ArchSpec::smem_capacity),
        },
        tuning.swap_ab,
        tuning.use_mode2_row_decoder,
        tuning.single_active_dispatch_warp,
        true,
    };
}

}  // namespace deep_gemm
