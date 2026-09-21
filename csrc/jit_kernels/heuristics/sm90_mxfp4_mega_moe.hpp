#pragma once

#include <cstdlib>

#include <deep_gemm/layout/mega_moe.cuh>

#include "../../utils/exception.hpp"
#include "sm90.hpp"

namespace deep_gemm {

// Weight tiles are stored contiguously at a fixed row count, so the layout is
// independent of the BLOCK_N the selector picks. A tile holds 64 B of packed
// E2M1 and 4 B of E8M0 per row, with no padding.
static constexpr int kSM90MXFP4BTileRows = 128;
static constexpr int kSM90MXFP4BTileBytes = kSM90MXFP4BTileRows * (64 + 4);

// Below this tile height the WGMMA is too short to hide a decode running beside
// it. Must stay in step with sm90_mxfp4_split_decode() in the kernel header.
static constexpr int kSplitDecodeMinBlockM = 16;

struct SM90MXFP4H200FusedConfig {
    static constexpr int kBlockK = 128;
    static constexpr int kSwizzleActsMode = 128;
    static constexpr int kNumDispatchThreads = 64;
    static constexpr int kNumNonEpilogueThreads = 64;
    static constexpr int kNumDecodeThreads = 256;
    static constexpr int kNumEpilogueThreads = 256;
    // Decode occupies its own warps only on a swapAB tile tall enough to hide
    // it. Must stay in step with sm90_mxfp4_num_threads() in the kernel header.
    static constexpr int num_threads(const bool swap_ab, const bool use_rs,
                                     const int block_m) {
        const bool split = swap_ab and not use_rs and block_m >= kSplitDecodeMinBlockM;
        return kNumDispatchThreads + kNumNonEpilogueThreads + kNumEpilogueThreads +
               (split ? kNumDecodeThreads : 0);
    }

    int block_m, block_n;
    int num_max_pool_tokens;
    int num_padded_sf_pool_tokens;
    int num_experts_per_wave;
    int num_stages, smem_size;
};

struct SM90MXFP4H200FusedShape {
    // The dispatch loop keeps per-rank state in one warp, so the width has to
    // stay within a warp; everything else is templated on it.
    static constexpr int kMaxNumRanks = 32;

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
    //   - a power-of-two rank count the dispatch loop can address
    //   - experts divide evenly across ranks
    //   - hidden and intermediate_hidden are whole BLOCK_K (128) tiles
    //   - topk fits in one warp (the dispatch loop maps lanes to topk slots)
    // This admits DeepSeek-V4-Flash (4096 / 2048 / 256 experts / topk 6).
    constexpr bool is_supported_shape() const noexcept {
        return num_sms > 0 &&
            num_ranks > 0 && num_ranks <= kMaxNumRanks &&
            (num_ranks & (num_ranks - 1)) == 0 &&
            num_experts > 0 &&
            num_experts % num_ranks == 0 &&
            num_topk > 0 && num_topk <= 32 &&
            hidden > 0 && hidden % 128 == 0 &&
            intermediate_hidden > 0 && intermediate_hidden % 128 == 0;
    }

    constexpr int experts_per_rank() const noexcept {
        return num_experts / num_ranks;
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
    // Within that range, take the *narrowest* transposed tile that still covers
    // one local expert's tokens. A wider tile pads the WGMMA's token dimension
    // with slots that carry nothing, and the tile is the token dimension under
    // swapAB. Measured on EP1/48 experts against the register-source path, where
    // routing puts M/6 slots on an expert: BM8 beats BM16 by 1.8 % at M=32,
    // BM16 beats BM24 by 1.7 % at M=64, and going one step too narrow is far
    // worse than one too wide -- BM16 at M=128 needs a second m-block per
    // expert and costs 45 %.
    // BM16 is left out. The bound above assumes routing spreads tokens evenly,
    // so it lets BM16 run to 16 tokens per expert; real routing is lumpy, and
    // by the top of that range enough experts overflow the tile to need the
    // second m-block this rule exists to avoid -- 40% at M=96 on both EP4 and
    // EP8. BM16 is also only ever picked where the decode runs in its own
    // warps, and there the wider tile wins anyway by giving that decode a
    // longer WGMMA to hide behind.
    // max_tokens_for_block_m is a mean, and a tile has to hold the busiest
    // expert, not the average one. Leave a quarter of it as headroom for that
    // imbalance: measured on MiMo, BM8 is still the better tile at M=32 but
    // costs 10% at M=40 and M=48 against its nominal bound of 48, because by
    // then enough experts need a second m-block.
    const auto tokens_per_block_m = [&](const int block_m) {
        return max_tokens_for_block_m(block_m) * 3 / 4;
    };
    const auto swap_ab_block_m = [&]() {
        for (const int block_m : {8, 24, 32}) {
            if (input.num_tokens <= tokens_per_block_m(block_m))
                return block_m;
        }
        // Past every bound the widest transposed tile is the least bad, not the
        // narrowest: it spills the busiest experts into the fewest extra
        // m-blocks. Returning 24 here cost 7-10% at M=128/EP8, where BM32's
        // derated bound is 127 and the batch misses it by one.
        return 32;
    }();

    if (input.num_tokens <= 1)
        tuning = {swap_ab_block_m, 256, 24, 8, SM90ArchSpec::smem_capacity,
                  true, true, true, true};
    else if (input.num_tokens <= 8)
        tuning = {swap_ab_block_m, 256, 16, 8, SM90ArchSpec::smem_capacity,
                  true, true, true, true};
    else if (input.num_tokens <= 16)
        tuning = {swap_ab_block_m, 256, 24, 8, SM90ArchSpec::smem_capacity,
                  true, true, true, true};
    else if (input.num_tokens <= swap_ab_max_tokens)
        tuning = {swap_ab_block_m, 256, 48, 8, SM90ArchSpec::smem_capacity,
                  true, true, true, true};
    // Everything above the transposed range takes the straight BN256 tile. The
    // BM128/BN128 split-M plan used to serve M > 256, but that bound was a bare
    // constant among otherwise shape-derived ones, and the plan loses to BN256
    // at every batch measured on both shapes -- MiMo 34/19/31 % at M=384/512/768
    // and DeepSeek-V4-Flash 55/55/24/16/37/26 % at M=384..2048. Against FP8 it
    // turns a 22-112 % loss into a 2 % win. It stays compiled and reachable
    // through DG_MXFP4_BLOCK_M=128; it is simply never selected.
    else
        tuning = {64, 256, 48, 3, SM90ArchSpec::smem_capacity,
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
    // critical path -- 2 to 6.7 % at every swapAB tier on H20 -- but the very
    // small batches on a 132-SM part are short enough that the shuffle latency
    // is not hidden, and there it costs 1 to 2 %. That is a property of how
    // little work the tier has, not of BLOCK_M: once the rule above sends M=32
    // to BM8, the same tile is 4.3 % *faster* with RS than without.
    if (input.num_sms > kSwapABReferenceNumSMs && tuning.block_m < 16 &&
        input.num_tokens <= 16)
        tuning.rs_swap_ab = false;
    // Decoding in its own warps costs a shared-memory round trip and buys
    // overlap with the WGMMA, so it pays only where the WGMMA is long enough to
    // hide the decode behind it. Measured on MiMo at both EP4 and EP8: BM24
    // gains 4-5% and BM16 3.8-7.7%, while BM8 issues a third as many WGMMAs as
    // BM24 and the round trip costs it 2-3% instead.
    if (tuning.swap_ab && tuning.block_m >= kSplitDecodeMinBlockM)
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
