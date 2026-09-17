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
    // Source the WGMMA's A operand (the weight tile under swapAB) from
    // registers instead of decoding it into shared memory first. Removes the
    // decoded-B ring, the store/load round trip and the decode barrier.
    bool rs_swap_ab;
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
        bool rs_swap_ab;
    } tuning {};

    // The transposed (swapAB) tile packs tokens into the WGMMA's N dimension, so
    // it is the right shape exactly while a local expert's tokens still fit one
    // tile. Routing spreads `num_tokens * num_ranks * num_topk` slots over
    // `num_experts` experts, so that holds up to:
    const int64_t routed_slots_per_expert_num =
        static_cast<int64_t>(input.num_ranks) * input.num_topk;
    const auto max_tokens_for_block_m = [&](const int block_m) {
        return static_cast<int>(
            (static_cast<int64_t>(block_m) * input.num_experts) /
            routed_slots_per_expert_num);
    };
    // ...but fitting is necessary, not sufficient. The transposed tile issues a
    // WGMMA of N_SWAP <= 24 where the straight tile issues one of 128, so it
    // needs roughly five times the MMA instructions for the same work. That is
    // affordable only while each SM still has enough tokens for the decode, not
    // the MMA, to be the critical path -- so the bound shrinks as the part gets
    // wider. Measured on MiMo: BM24 wins by 6.5 % at M=128 and loses by 57 % at
    // M=256 on H20's 78 SMs; on H200's 132 it already loses by 23 % at M=128 and
    // wins by 4.6 % at M=64.
    static constexpr int kSwapABReferenceNumSMs = 78;
    const int swap_ab_max_tokens =
        max_tokens_for_block_m(24) * kSwapABReferenceNumSMs / input.num_sms;

    if (input.num_tokens <= 1)
        tuning = {8, 256, 24, 8, SM90ArchSpec::smem_capacity,
                  true, true, true, true};
    else if (input.num_tokens <= 8)
        tuning = {8, 256, 16, 8, SM90ArchSpec::smem_capacity,
                  true, true, true, true};
    else if (input.num_tokens <= 16)
        tuning = {8, 256, 24, 8, SM90ArchSpec::smem_capacity,
                  true, true, true, true};
    else if (input.num_tokens <= 32)
        tuning = {16, 256, 48, 6, SM90ArchSpec::smem_capacity,
                  true, true, true, true};
    else if (input.num_tokens <= swap_ab_max_tokens)
        tuning = {24, 256, 48, 8, SM90ArchSpec::smem_capacity,
                  true, true, true, true};
    else if (input.num_tokens <= 256)
        tuning = {64, 256, 48, 3, SM90ArchSpec::smem_capacity,
                  false, true, false, false};
    else
        tuning = {128, 128, 48, 6, SM90ArchSpec::smem_capacity,
                  false, true, false, false};

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
    // The same width argument applies to RS itself. Trading a shared-memory
    // round trip for cross-lane shuffles wins wherever the decode is on the
    // critical path -- 2 to 6.7 % at every swapAB tier on H20 -- but the BM8
    // tiers on a 132-SM part are short enough that the shuffle latency is not
    // hidden, and there it costs 1 to 2 %.
    if (input.num_sms > kSwapABReferenceNumSMs && tuning.block_m < 16)
        tuning.rs_swap_ab = false;
    tuning.rs_swap_ab =
        env_int("DG_MXFP4_RS", tuning.rs_swap_ab ? 1 : 0) != 0;
    // Only the swapAB mainloop has a register-source form.
    tuning.rs_swap_ab = tuning.rs_swap_ab && tuning.swap_ab;

    // The per-tier shared-memory numbers above were the exact layout size of
    // one fixed stage count. Once pipeline depth became tunable they became a
    // trap: a deeper plan lays out past the launch's dynamic allocation and the
    // kernel faults with an illegal address rather than failing to compile.
    // These are persistent one-CTA-per-SM kernels, so requesting the whole
    // capacity costs no occupancy and removes the entire failure mode.
    // A swapAB tier that falls back to shared memory allocates a decoded-B tile
    // again and wants the depth it was originally measured at, not the RS one.
    if (tuning.swap_ab && !tuning.rs_swap_ab)
        tuning.num_stages = tuning.block_m == 8 ? 4 : 3;

    tuning.smem_size = SM90ArchSpec::smem_capacity;

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
        tuning.rs_swap_ab,
    };
}

}  // namespace deep_gemm
