#pragma once

#include <algorithm>
#include <cctype>
#include <cmath>
#include <cstdint>
#include <string>
#include <unordered_set>

#include "../../utils/exception.hpp"

#include <deep_gemm/common/types.cuh>
#include <deep_gemm/layout/mega_moe.cuh>

#include "../../utils/math.hpp"
#include "../../utils/system.hpp"
#include "sm100.hpp"
#include "sm90.hpp"

namespace deep_gemm {

struct MegaMoEConfig {
    // Block tiling
    int block_m, block_n, block_k;
    int load_block_m, load_block_n;
    int store_block_m;

    // SF block sizes (UTCCP 128-aligned)
    int sf_block_m, sf_block_n;

    // Pool capacity and SF-padded token count
    int num_max_pool_tokens;
    int num_padded_sf_pool_tokens;

    // Swizzle modes for TMA descriptors
    int swizzle_acts_mode, swizzle_weights_mode;

    // Number of experts to process per wave
    int num_experts_per_wave;

    // Pipeline stages and shared memory
    int num_stages, smem_size;

    // Thread layout
    int num_dispatch_threads, num_non_epilogue_threads, num_epilogue_threads;

    // Dispatch pull config
    int num_bytes_per_pull;

    friend std::ostream& operator << (std::ostream& os, const MegaMoEConfig& config) {
        os << "MegaMoEConfig("
           << "block_m=" << config.block_m << ", block_n=" << config.block_n << ", block_k=" << config.block_k
           << ", load_block_m=" << config.load_block_m << ", load_block_n=" << config.load_block_n
           << ", store_block_m=" << config.store_block_m
           << ", sf_block_m=" << config.sf_block_m << ", sf_block_n=" << config.sf_block_n
           << ", num_max_pool_tokens=" << config.num_max_pool_tokens
           << ", num_padded_sf_pool_tokens=" << config.num_padded_sf_pool_tokens
           << ", swizzle_acts_mode=" << config.swizzle_acts_mode << ", swizzle_weights_mode=" << config.swizzle_weights_mode
           << ", num_experts_per_wave=" << config.num_experts_per_wave
           << ", num_stages=" << config.num_stages << ", smem_size=" << config.smem_size
           << ", num_dispatch_threads=" << config.num_dispatch_threads
           << ", num_non_epilogue_threads=" << config.num_non_epilogue_threads
           << ", num_epilogue_threads=" << config.num_epilogue_threads
           << ", num_bytes_per_pull=" << config.num_bytes_per_pull << ")";
        return os;
    }
};

static std::tuple<int, int, int, int, int> get_block_config_for_mega_moe(
    const int& num_ranks, const int& num_experts,
    const int& num_max_tokens_per_rank, const int& num_topk,
    const int& num_tokens) {
    auto [cluster_size, block_m, store_block_m, block_k, num_epilogue_warpgroups] = [&]() -> std::tuple<int, int, int, int, int> {
        float num_expected_tokens_per_expert = static_cast<float>(num_tokens) * num_ranks * num_topk / num_experts;
        if (num_expected_tokens_per_expert <= 8.5) {
            // Really small token-per-expert (e.g. RL long-tail rollout), use the smallest block_m and larger BLOCK_K for less synchronization
            return {2, 16, 8, 256, 2};
        } else if (num_expected_tokens_per_expert <= 16.5) {
            // Small batch size, small EP, decoding, e.g. 6/384 experts, EP8, bsz 128
            return {2, 32, 16, 128, 2};
        } else if (num_expected_tokens_per_expert <= 32.5) {
            // Medium batch size, small EP, decoding, e.g. 6/384 experts, EP8, bsz 256
            return {2, 64, 32, 128, 1};
        } else if (num_expected_tokens_per_expert <= 64.5) {
            // Large batch size, small EP, decoding, e.g. 6/384 experts, EP8, bsz 512
            return {2, 96, 16, 128, 2};
        } else if (num_expected_tokens_per_expert <= 96.5) {
            // Medium batch size, Medium EP, decoding, e.g. 6/384 experts, EP16, bsz 256, or EP32, bsz128
            return {2, 128, 32, 128, 2};
        } else {
            // Prefill, or large EP decoding
            return {2, 192, 32, 128, 2};
        }
    }();

    // Check whether our `block_m` lies in `kCandidateBlockM`
    DG_HOST_ASSERT(std::any_of(
        layout::kCandidateBlockM, layout::kCandidateBlockM + layout::kNumCandidateBlockMs,
        [=](const auto& candidate) { return candidate == block_m; })
    );

    // Return configs
    return {cluster_size, block_m, store_block_m, block_k, num_epilogue_warpgroups * 128};
}

static int get_num_experts_per_wave_for_mega_moe(
    const int& num_experts_per_rank, const int& num_tokens, const int& num_topk,
    const int& intermediate_hidden, const int& block_m, const int& block_n, const int& num_sms) {

    float expected_tokens_per_expert = static_cast<float>(num_tokens) * num_topk / num_experts_per_rank;
    if (expected_tokens_per_expert < 1) {
        // Most experts don't have tokens, calculate all experts at once
        return num_experts_per_rank;
    }

    // Reduce per-expert block count by this factor since uneven routing leaves some experts with fewer tokens
    constexpr int kImbalanceFactor = 2;

    // Count L1 blocks per expert assuming tokens are evenly spread across experts
    const int num_m_blocks = ceil_div(static_cast<int>(std::ceil(expected_tokens_per_expert)), block_m);
    const int num_n_blocks = (2 * intermediate_hidden) / block_n;
    const int num_l1_blocks_per_expert = num_m_blocks * num_n_blocks;

    // Pick the smallest value whose total blocks (after imbalance reduction) can keep all SMs busy
    int min_num_experts_per_wave = num_l1_blocks_per_expert > 0
        ? ceil_div(kImbalanceFactor * num_sms, num_l1_blocks_per_expert) : 1;
    if (min_num_experts_per_wave >= num_experts_per_rank)
        return num_experts_per_rank;

    // When each expert nearly fills all SMs, use the smallest wave to maximize L2 cache reuse
    if (num_l1_blocks_per_expert >= num_sms)
        return min_num_experts_per_wave;

    // Otherwise search [min_num_experts_per_wave, min_num_experts_per_wave * 2] for a value where the last partial
    // wave has as many experts as possible relative to a full wave
    const int max_num_experts_per_wave = std::min(num_experts_per_rank, min_num_experts_per_wave * 2);
    int best_num_experts_per_wave = min_num_experts_per_wave;
    float best_tail_ratio = -1.0f;

    for (int num_experts_per_wave = min_num_experts_per_wave; num_experts_per_wave <= max_num_experts_per_wave; ++ num_experts_per_wave) {
        int remainder = num_experts_per_rank % num_experts_per_wave;
        float tail_ratio = (remainder == 0) ? 1.0f : static_cast<float>(remainder) / num_experts_per_wave;

        if (tail_ratio > best_tail_ratio) {
            best_tail_ratio = tail_ratio;
            best_num_experts_per_wave = num_experts_per_wave;
        }
    }
    return best_num_experts_per_wave;
}

static std::pair<int, int> get_pipeline_config_for_mega_moe(
    const int& smem_capacity,
    const int& num_experts, const int& hidden,
    const int& block_m, const int& block_n, const int& block_k,
    const int& num_bytes_per_pull, const int& store_block_m,
    const int& sf_block_m, const int& sf_block_n, const int& gran_k,
    const int& num_dispatch_warps, const int& num_epilogue_warps) {
    constexpr int kSmemAlignment = 1024;
    constexpr int kNumEpilogueStages = 2;
    constexpr int kNumTMAStoreStages = 2;

    // Always multicast on A
    const int load_block_m = block_m / 2;

    // Dispatch region
    const int smem_expert_count_size = align(
        num_experts * static_cast<int>(sizeof(uint32_t)), kSmemAlignment);
    const int smem_send_buffers_size = align(
        static_cast<int>(layout::Buffer(layout::Data(num_bytes_per_pull), num_dispatch_warps, 1).get_num_bytes()),
        kSmemAlignment);
    const int smem_dispatch_size = smem_expert_count_size + smem_send_buffers_size;

    // C/D output region: max of L1 FP8 (2 TMA stages, BLOCK_N/2 post-SwiGLU) and L2 BF16 (1 stage)
    const auto num_epilogue_warpgroups = num_epilogue_warps / 4;
    const int smem_cd_l1 = num_epilogue_warpgroups * store_block_m * (block_n / 2) * kNumTMAStoreStages;
    const int smem_cd_l2 = num_epilogue_warpgroups * store_block_m * block_n * static_cast<int>(sizeof(nv_bfloat16));
    const int smem_cd = align(std::max(smem_cd_l1, smem_cd_l2), kSmemAlignment);

    // Barriers (stage-independent): dispatch + tensor memory full/empty + combine (2 per epilogue warp)
    const int smem_barriers = (num_dispatch_warps + kNumEpilogueStages * 2 + num_epilogue_warps * 2) * 8;

    // Amax warp-pair reduction buffer
    const int smem_amax_reduction = store_block_m * num_epilogue_warps * static_cast<int>(sizeof(float));

    // Tensor memory pointer
    const int smem_tmem_ptr = 4;

    // SF is aligned to UTCCP 128-element granularity
    const int smem_sfa_per_stage = sf_block_m * (block_k / gran_k);
    const int smem_sfb_per_stage = sf_block_n * (block_k / gran_k);

    // Per-stage: A tile + B tile + SF tiles + full/empty barriers
    const int smem_a_size_per_stage = load_block_m * block_k;
    const int smem_b_size_per_stage = block_n * block_k;
    const int smem_size_per_stage = smem_a_size_per_stage + smem_b_size_per_stage + smem_sfa_per_stage + smem_sfb_per_stage + 2 * 8;

    // Fixed total
    const int smem_fixed = smem_dispatch_size + smem_cd + smem_amax_reduction + smem_barriers + smem_tmem_ptr;

    // Select maximum number of stages
    const int num_stages = (smem_capacity - smem_fixed) / smem_size_per_stage;
    DG_HOST_ASSERT(num_stages >= 2);

    return {num_stages, smem_fixed + num_stages * smem_size_per_stage};
}

static MegaMoEConfig get_mega_moe_config(
    const int& num_ranks, const int& num_experts, const int& num_experts_per_rank,
    const int& num_max_tokens_per_rank, const int& num_tokens, const int& num_topk,
    const int& hidden, const int& intermediate_hidden,
    const int& num_padded_sf_pool_tokens) {

    // Block config
    const auto [cluster_size, block_m, store_block_m, block_k, num_epilogue_threads] =
        get_block_config_for_mega_moe(num_ranks, num_experts, num_max_tokens_per_rank, num_topk, num_tokens);
    const int block_n = 128;
    const int load_block_m = block_m / 2;
    const int load_block_n = block_n;
    const auto [sf_block_m, sf_block_n] =
        SM100ArchSpec::get_sf_uttcp_aligned_block_sizes(block_m, block_n, MmaKind::MXFP8FP4);
    const int num_max_pool_tokens = layout::get_num_max_pool_tokens(
        num_ranks, num_max_tokens_per_rank, num_topk, num_experts_per_rank);
    // NOTES: FP8 activations and FP4 weights (unpacked to 8-bit in smem) both use 128B swizzle
    const int swizzle_acts_mode = 128;
    const int swizzle_weights_mode = 128;
    const int gran_k = 32;

    // Waves
    const int num_sms = device_runtime->get_num_sms();
    const int num_experts_per_wave = get_num_experts_per_wave_for_mega_moe(
        num_experts_per_rank, num_tokens, num_topk,
        intermediate_hidden, block_m, block_n, num_sms);

    // Thread layout
    const int num_dispatch_threads = 128;
    const int num_non_epilogue_threads = 128;

    // Pull: divide token bytes by 2 until <= kPullThreshold
    constexpr int kPullThreshold = 4096;
    int num_bytes_per_pull = hidden;
    while (num_bytes_per_pull > kPullThreshold) {
        DG_HOST_ASSERT(num_bytes_per_pull % 2 == 0);
        num_bytes_per_pull /= 2;
    }

    // Pipeline
    const auto [num_stages, smem_size] = get_pipeline_config_for_mega_moe(
        SM100ArchSpec::smem_capacity,
        num_experts, hidden,
        block_m, block_n, block_k, num_bytes_per_pull, store_block_m,
        sf_block_m, sf_block_n, gran_k,
        num_dispatch_threads / 32, num_epilogue_threads / 32);

    const auto config = MegaMoEConfig {
        block_m, block_n, block_k,
        load_block_m, load_block_n, store_block_m,
        sf_block_m, sf_block_n,
        num_max_pool_tokens, num_padded_sf_pool_tokens,
        swizzle_acts_mode, swizzle_weights_mode,
        num_experts_per_wave,
        num_stages, smem_size,
        num_dispatch_threads, num_non_epilogue_threads, num_epilogue_threads,
        num_bytes_per_pull
    };

    // Print configs for the first time
    if (get_env<int>("DG_JIT_DEBUG") or get_env<int>("DG_PRINT_CONFIGS")) {
        const auto key = fmt::format(
            "MegaMoEConfig(num_ranks={}, num_experts={}, hidden={}, intermediate_hidden={}, num_max_tokens_per_rank={}, num_tokens={}, num_topk={})",
            num_ranks, num_experts, hidden, intermediate_hidden, num_max_tokens_per_rank, num_tokens, num_topk);
        static std::unordered_set<std::string> printed;
        if (printed.count(key) == 0) {
            std::cout << key << ": " << config << std::endl;
            printed.insert(key);
        }
    }
    return config;
}

// ============================================================================
// SM90 (Hopper) MegaMoE configuration
// ----------------------------------------------------------------------------
// SM90 differs from SM100 in:
//   - No tensor memory (TMEM): WGMMA accumulators live in registers.
//   - No FP4: weights are FP8 e4m3, scales are per-128 channel float.
//   - No 2-CTA cluster MMA or TMA multicast.
//   - SF for activations is float (not UE8M0 int) and per-128 (not per-32).
// The kernel is in `deep_gemm/impls/sm90_fp8_mega_moe.cuh`; this config is
// what the host runtime reads when instantiating a shape-specialized variant.
// ============================================================================

struct MegaMoESM90Config {
    // Block tiling (no STORE_BLOCK_M / SF_BLOCK_M concept on SM90)
    int block_m, block_n, block_k;

    // Pool capacity, allocated SF capacity, and per-config logical SF stride.
    int num_max_pool_tokens;
    int num_padded_sf_pool_tokens, sf_pool_stride_tokens;

    // Swizzle modes for TMA descriptors (acts/weights). Both are 128B on FP8 K-major.
    int swizzle_acts_mode, swizzle_weights_mode;

    // Number of experts to process per wave
    int num_experts_per_wave;

    // Number of SMs used by this phase
    int num_sms;

    // Pipeline stages and shared memory
    int num_stages, smem_size;

    // Thread layout: dispatch + non-epilogue (TMA) + epilogue (math)
    int num_dispatch_threads, num_non_epilogue_threads, num_epilogue_threads;

    // Chosen scheduler / epilogue modes.  Keeping these in the config makes the
    // SM90 path follow the same single-source-of-truth style as regular GEMM
    // configs: the selector chooses a complete phase config, then launch consumes it.
    bool direct_l2_scatter, nmajor_schedule, one_warp_cleanup, swap_ab;

    friend std::ostream& operator << (std::ostream& os, const MegaMoESM90Config& config) {
        os << "MegaMoESM90Config("
           << "block_m=" << config.block_m << ", block_n=" << config.block_n << ", block_k=" << config.block_k
           << ", num_max_pool_tokens=" << config.num_max_pool_tokens
           << ", num_padded_sf_pool_tokens=" << config.num_padded_sf_pool_tokens
           << ", sf_pool_stride_tokens=" << config.sf_pool_stride_tokens
           << ", swizzle_acts_mode=" << config.swizzle_acts_mode << ", swizzle_weights_mode=" << config.swizzle_weights_mode
           << ", num_experts_per_wave=" << config.num_experts_per_wave
           << ", num_sms=" << config.num_sms
           << ", num_stages=" << config.num_stages << ", smem_size=" << config.smem_size
           << ", num_dispatch_threads=" << config.num_dispatch_threads
           << ", num_non_epilogue_threads=" << config.num_non_epilogue_threads
           << ", num_epilogue_threads=" << config.num_epilogue_threads
           << ", direct_l2_scatter=" << config.direct_l2_scatter
           << ", nmajor_schedule=" << config.nmajor_schedule
           << ", one_warp_cleanup=" << config.one_warp_cleanup
           << ", swap_ab=" << config.swap_ab << ")";
        return os;
    }
};

enum class Sm90MoeHardwareProfile {
    Generic,
    LowSm,
    HighSm
};

struct Sm90MoeHeuristicInput {
    int launch_num_sms;

    int num_ranks, num_experts, num_experts_per_rank;
    int num_max_tokens_per_rank, num_tokens, num_topk;
    int hidden, intermediate_hidden;
    int num_padded_sf_pool_tokens;

    bool fp8_combine;
    bool eplb_hint, skew_hint, masked_hint;
    std::string device_profile_override;
};

struct Sm90MoeNumericalConfig {
    bool fp8_combine = false;
    bool bf16_scaled_accum = false;
};

struct Sm90MoeLaunchConfig {
    MegaMoESM90Config l1;
    MegaMoESM90Config l2;
    Sm90MoeNumericalConfig numerical;
};

static std::string get_sm90_moe_lowercase(std::string value) {
    std::transform(value.begin(), value.end(), value.begin(), [](const unsigned char c) {
        return static_cast<char>(std::tolower(c));
    });
    return value;
}

static Sm90MoeHardwareProfile classify_sm90_moe_hardware(
    const int launch_num_sms,
    const std::string& profile_override) {
    const auto forced = get_sm90_moe_lowercase(profile_override);
    if (not forced.empty() and forced != "auto") {
        DG_HOST_ASSERT(forced == "generic" or forced == "low_sm" or forced == "high_sm");
        if (forced == "low_sm")
            return Sm90MoeHardwareProfile::LowSm;
        if (forced == "high_sm")
            return Sm90MoeHardwareProfile::HighSm;
        return Sm90MoeHardwareProfile::Generic;
    }

    DG_HOST_ASSERT(launch_num_sms > 0);
    return launch_num_sms < 100
        ? Sm90MoeHardwareProfile::LowSm
        : Sm90MoeHardwareProfile::HighSm;
}

enum class Sm90MoeShapeFamily {
    Generic,
    Compact,
    Wide
};

struct Sm90MoeLoad {
    int64_t routed_tokens;
    int64_t local_experts;

    bool valid() const {
        return routed_tokens >= 0 and local_experts > 0;
    }

    bool equals(const int64_t value) const {
        return routed_tokens == value * local_experts;
    }

    bool less_than(const int64_t value) const {
        return routed_tokens < value * local_experts;
    }

    bool less_equal(const int64_t value) const {
        return routed_tokens <= value * local_experts;
    }

    bool greater_than(const int64_t value) const {
        return routed_tokens > value * local_experts;
    }

    bool greater_equal(const int64_t value) const {
        return routed_tokens >= value * local_experts;
    }

    bool in_closed(const int64_t low, const int64_t high) const {
        return greater_equal(low) and less_equal(high);
    }

    bool in_open_closed(const int64_t low, const int64_t high) const {
        return greater_than(low) and less_equal(high);
    }

    bool less_equal_fraction(const int64_t numerator, const int64_t denominator) const {
        DG_HOST_ASSERT(denominator > 0);
        return routed_tokens * denominator <= numerator * local_experts;
    }

    template <typename... Values>
    bool is_one_of(const Values... values) const {
        return (equals(static_cast<int64_t>(values)) or ...);
    }
};

static Sm90MoeLoad get_sm90_moe_load(const Sm90MoeHeuristicInput& input) {
    const Sm90MoeLoad load {
        static_cast<int64_t>(input.num_tokens) * input.num_topk,
        input.num_experts_per_rank,
    };
    DG_HOST_ASSERT(load.valid());
    return load;
}

static Sm90MoeShapeFamily classify_sm90_moe_shape(
    const int hidden, const int intermediate_hidden) {
    if (hidden >= 3072 and hidden < 5120 and
        intermediate_hidden >= 1536 and intermediate_hidden < 2560)
        return Sm90MoeShapeFamily::Compact;
    if (hidden >= 5120 and hidden <= 8192 and
        intermediate_hidden >= 2560 and intermediate_hidden <= 4096)
        return Sm90MoeShapeFamily::Wide;
    return Sm90MoeShapeFamily::Generic;
}

static bool should_use_swap_ab_for_mega_moe_sm90(
    const Sm90MoeHeuristicInput& input,
    const Sm90MoeLoad& load,
    const Sm90MoeShapeFamily shape_family,
    const int block_m,
    const int num_epilogue_threads) {
    const int max_load =
        shape_family == Sm90MoeShapeFamily::Compact and input.num_topk == 6
            ? 24
            : 16;
    const bool decode_split_n_path =
        block_m == 64 and num_epilogue_threads == 256;
    return decode_split_n_path and load.greater_than(0) and
           load.less_equal(max_load);
}

struct Sm90MoeProfileTuning {
    bool selected = false;
    int num_experts_per_wave = 0;
    int num_stages = 4;
    bool direct_l2_scatter = false;
    bool l2_nmajor_schedule = false;
    bool one_warp_cleanup = false;
};

static Sm90MoeProfileTuning lookup_sm90_moe_profile_tuning(
    const Sm90MoeHardwareProfile hardware_profile,
    const Sm90MoeShapeFamily shape_family,
    const Sm90MoeHeuristicInput& input,
    const Sm90MoeLoad& load,
    const int block_m,
    const int block_n) {
    Sm90MoeProfileTuning tuning;
    if (block_m != 64 or block_n != 256)
        return tuning;

    const bool compact_topk8 =
        shape_family == Sm90MoeShapeFamily::Compact and
        input.num_experts_per_rank == 32 and input.num_topk == 8;
    const bool wide_topk6 =
        shape_family == Sm90MoeShapeFamily::Wide and
        input.num_experts_per_rank == 48 and input.num_topk == 6;
    if (not compact_topk8 and not wide_topk6)
        return tuning;

    tuning.selected = true;
    if (compact_topk8 and hardware_profile == Sm90MoeHardwareProfile::HighSm) {
        if (load.less_equal(3)) {
            tuning = {true, 32, 4, true,  true,  false};
        } else if (load.less_equal(6)) {
            tuning = {true, 32, 4, false, true,  true};
        } else if (load.less_equal(12)) {
            tuning = {true, 32, 4, true,  false, true};
        } else if (load.less_equal(24)) {
            tuning = {true, 32, 4, false, true,  true};
        } else if (load.less_equal(48)) {
            tuning = {true, 32, 4, true,  false, true};
        } else if (load.less_equal_fraction(129, 2)) {
            tuning = {true, 32, 4, false, true,  true};
        } else if (load.less_equal(240)) {
            tuning = {true, 32, 4, false, true,  false};
        } else if (load.less_equal(384)) {
            tuning = {true, 16, 4, false, true,  false};
        } else if (load.less_equal(640)) {
            tuning = {true, 32, 4, false, true,  true};
        } else if (load.less_equal(896)) {
            tuning = {true, 32, 4, false, true,  false};
        } else if (load.less_equal(1536)) {
            tuning = {true, 32, 4, false, true,  true};
        } else {
            tuning = {true, 32, 4, false, true,  false};
        }
        return tuning;
    }

    if (compact_topk8) {
        if (load.equals(128)) {
            tuning.num_experts_per_wave = 8;
        } else if ((load.greater_equal(192) and load.less_than(512)) or
                   (load.greater_than(512) and load.less_equal(768))) {
            tuning.num_experts_per_wave = 16;
        }

        const bool direct_l2_mid_suppressed =
            load.greater_than(64) and load.less_than(256) and
            not load.equals(128);
        tuning.direct_l2_scatter =
            not direct_l2_mid_suppressed and
            (load.is_one_of(2, 4, 8, 16, 32, 88, 128) or
             load.in_closed(64, 80) or load.in_closed(96, 120) or
             load.greater_equal(144));
        tuning.l2_nmajor_schedule =
            load.greater_equal(256) and
            not (load.equals(256) and input.eplb_hint) and
            not input.skew_hint;
        tuning.one_warp_cleanup =
            (load.less_equal(80) and not load.equals(64)) or
            load.equals(128);
        const bool hinted_load64 =
            (input.eplb_hint or input.skew_hint or input.masked_hint) and
            load.equals(64);
        const bool stage5 = tuning.direct_l2_scatter and
            (load.is_one_of(2, 4, 16, 32, 128) or hinted_load64 or
             load.greater_equal(192));
        tuning.num_stages = stage5 ? 5 : 4;
        return tuning;
    }

    if (load.in_closed(8, 32))
        tuning.num_experts_per_wave = 16;
    tuning.direct_l2_scatter =
        load.in_closed(61, 62) or load.greater_equal(64);
    tuning.one_warp_cleanup =
        (input.masked_hint and load.equals(64)) or
        load.is_one_of(80, 128);
    const bool stage5 = tuning.direct_l2_scatter and
        (load.equals(64) or load.in_closed(76, 96) or
         (load.greater_equal(128) and load.less_than(240)) or
         load.greater_equal(384));
    tuning.num_stages = stage5 ? 5 : 4;
    return tuning;
}

static int get_num_experts_per_wave_for_mega_moe_sm90(
    const Sm90MoeLoad& load,
    const int& num_experts_per_rank, const int& num_tokens, const int& num_topk,
    const int& intermediate_hidden, const int& block_m, const int& block_n, const int& num_sms) {
    if (const int forced = get_env<int>("DG_SM90_MOE_EXPERTS_PER_WAVE"); forced > 0) {
        DG_HOST_ASSERT(forced <= num_experts_per_rank);
        DG_HOST_ASSERT(num_experts_per_rank % forced == 0);
        return forced;
    }

    if (block_m == 64 and (load.less_than(1) or load.greater_than(4))) {
        return num_experts_per_rank;
    }
    return get_num_experts_per_wave_for_mega_moe(
        num_experts_per_rank, num_tokens, num_topk,
        intermediate_hidden, block_m, block_n, num_sms);
}

static std::pair<int, int> get_pipeline_config_for_mega_moe_sm90(
    const int& smem_capacity,
    const int& num_experts, const int& hidden,
    const int& block_m, const int& block_n, const int& block_k,
    const int& num_dispatch_warps, const int& num_epilogue_warps,
    const bool& direct_l2_scatter_enabled = false,
    const int& default_num_stages = 0,
    const bool& swap_ab = false,
    const bool& fp8_combine = false,
    const bool& require_exact_default_stages = false) {
    constexpr int kSmemAlignment = 1024;

    // Dispatch region (same as SM100)
    const int smem_expert_count_size = align(
        num_experts * static_cast<int>(sizeof(uint32_t)), kSmemAlignment);
    const int smem_send_buffers_size = align(
        static_cast<int>(layout::Buffer(layout::Data(hidden), num_dispatch_warps, 1).get_num_bytes()),
        kSmemAlignment);
    const int smem_dispatch_size = smem_expert_count_size + smem_send_buffers_size;

    // C/D output region: max of L1 FP8 (single-buffered, BLOCK_N/2 post-SwiGLU)
    // and L2 BF16, then 1024-byte aligned (matches kernel's SMEM_CD_SIZE).
    const auto num_epilogue_warpgroups = num_epilogue_warps / 4;
    const auto wg_layout = layout::get_sm90_moe_warpgroup_layout(
        block_m, block_n, num_epilogue_warpgroups);
    const int wg_block_m = static_cast<int>(wg_layout.block_m);
    const int wg_block_n = static_cast<int>(wg_layout.block_n);
    const int smem_cd_l1 = num_epilogue_warpgroups * wg_block_m * (wg_block_n / 2);  // 1 byte/elem (FP8)
    const bool direct_l2_scatter =
        direct_l2_scatter_enabled and not swap_ab and wg_block_n == 128;
    const int combine_element_bytes = fp8_combine ?
        1 : static_cast<int>(sizeof(nv_bfloat16));
    const int smem_cd_l2 = direct_l2_scatter ? 0 :
        num_epilogue_warpgroups * wg_block_m * wg_block_n * combine_element_bytes;
    const int smem_cd_swap_l1 = swap_ab
        ? block_m * (block_n / 2) *
              (static_cast<int>(sizeof(float)) + static_cast<int>(sizeof(uint8_t)))
        : 0;
    const int smem_cd = align(
        std::max(std::max(smem_cd_l1, smem_cd_l2), smem_cd_swap_l1),
        kSmemAlignment);

    // SF on SM90:
    //   * SFA per stage must hold one aligned BLOCK_M-float vector for every
    //     per-64-K L2 scale group (two for BK128, four for BK256)
    //   * SFB is loaded directly from global by the math warpgroup (block-(128,128)
    //     weight quantization), so no SMEM is reserved for it.
    const int smem_sfa_half_stride_bytes = align(block_m * static_cast<int>(sizeof(float)), 128);
    const int smem_sfa_per_stage =
        (block_k / 64) * smem_sfa_half_stride_bytes;
    const int smem_sfb_per_stage = 0;

    // Per-stage: A tile + B tile + SFA tile + SFB tile
    const int smem_per_stage = block_m * block_k + block_n * block_k +
                               smem_sfa_per_stage + smem_sfb_per_stage;

    // Barriers (8 bytes each):
    //   * dispatch: num_dispatch_warps
    //   * GEMM full + empty: 2 * num_stages
    //   * combine: 2 * num_epilogue_warps
    const int smem_barriers_fixed = (num_dispatch_warps + 2 * num_epilogue_warps) * 8;
    const int smem_barriers_per_stage = 2 * 8;

    // Fixed total
    const int smem_fixed = smem_dispatch_size + smem_cd + smem_barriers_fixed;

    // Select the retained stage count for the current shape.
    const int max_num_stages = (smem_capacity - smem_fixed) /
                               (smem_per_stage + smem_barriers_per_stage);
    if (max_num_stages < 2)
        return {0, 0};
    const bool prefer_bn256_n_tile = block_n == 256;
    const int forced_num_stages = get_env<int>("DG_SM90_MOE_NUM_STAGES");
    if (require_exact_default_stages and forced_num_stages <= 0 and
        default_num_stages > max_num_stages)
        return {0, 0};
    const int preferred_num_stages = default_num_stages > 0
        ? std::min(default_num_stages, max_num_stages)
        : (prefer_bn256_n_tile ? std::min(4, max_num_stages) : 0);
    const int num_stages = forced_num_stages > 0
        ? std::min(forced_num_stages, max_num_stages)
        : (preferred_num_stages > 0 ? preferred_num_stages : max_num_stages);
    if (num_stages < 2 or num_stages > max_num_stages)
        return {0, 0};
    return {num_stages,
            smem_fixed + num_stages * (smem_per_stage + smem_barriers_per_stage)};
}

static bool get_sm90_moe_bool_config(
    const std::string& env_name, const bool default_value) {
    const int forced = get_env<int>(env_name, -1);
    DG_HOST_ASSERT(forced == -1 or forced == 0 or forced == 1);
    return forced == -1 ? default_value : forced != 0;
}

static bool supports_sm90_moe_bf16_scaled_accum(
    const MegaMoESM90Config& config) {
    const int num_epilogue_warpgroups = config.num_epilogue_threads / 128;
    if (config.swap_ab) {
        return config.block_m == 64 and config.block_n == 128 and
               config.block_k == 128 and num_epilogue_warpgroups == 2;
    }
    const auto wg_layout = layout::get_sm90_moe_warpgroup_layout(
        config.block_m, config.block_n, num_epilogue_warpgroups);
    const int wg_block_n = static_cast<int>(wg_layout.block_n);
    return config.block_m == 64 and
           (wg_block_n == 128 or wg_block_n == 256) and
           (config.block_k == 128 or config.block_k == 256);
}

static bool is_sm90_moe_phase_config_legal(
    const Sm90MoeHeuristicInput& input,
    const MegaMoESM90Config& config,
    const int phase_n,
    const int phase_num_sms) {
    if ((config.block_m != 64 and config.block_m != 128) or
        (config.block_n != 128 and config.block_n != 256 and config.block_n != 512) or
        (config.block_k != 128 and config.block_k != 256))
        return false;
    if (phase_n % config.block_n != 0 or
        input.hidden % config.block_k != 0 or
        input.intermediate_hidden % config.block_k != 0)
        return false;
    if (phase_num_sms <= 0 or phase_num_sms > input.launch_num_sms)
        return false;
    if (config.num_experts_per_wave <= 0 or
        config.num_experts_per_wave > input.num_experts_per_rank or
        input.num_experts_per_rank % config.num_experts_per_wave != 0)
        return false;
    if ((config.num_dispatch_threads != 64 and config.num_dispatch_threads % 128 != 0) or
        (config.num_non_epilogue_threads != 64 and config.num_non_epilogue_threads != 128) or
        config.num_epilogue_threads <= 0 or config.num_epilogue_threads % 128 != 0 or
        config.num_dispatch_threads + config.num_non_epilogue_threads +
            config.num_epilogue_threads > 1024)
        return false;

    const int num_epilogue_warpgroups = config.num_epilogue_threads / 128;
    const auto wg_layout = layout::get_sm90_moe_warpgroup_layout(
        config.block_m, config.block_n, num_epilogue_warpgroups);
    const int split_m = static_cast<int>(wg_layout.split_m);
    const int split_n_count = static_cast<int>(wg_layout.split_n_count);
    if (split_m <= 0 or split_n_count <= 0 or
        config.block_m % split_m != 0 or config.block_n % split_n_count != 0)
        return false;
    const int wg_block_m = static_cast<int>(wg_layout.block_m);
    const int wg_block_n = static_cast<int>(wg_layout.block_n);
    if (wg_block_m != 64 or
        (wg_block_n != 64 and wg_block_n != 128 and wg_block_n != 256))
        return false;
    // The only N64 consumer retained in production is the validated swap-AB
    // path. The paired-WG scale-factor path is deliberately not selectable.
    if (wg_block_n == 64 and not config.swap_ab)
        return false;
    if (config.block_m == 64 and config.block_n == 256 and
        num_epilogue_warpgroups != 2)
        return false;
    if (config.swap_ab and not (
            config.block_m == 64 and config.block_n == 128 and
            config.block_k == 128 and num_epilogue_warpgroups == 2))
        return false;
    if (config.direct_l2_scatter and (config.swap_ab or wg_block_n != 128))
        return false;

    const auto [num_stages, smem_size] = get_pipeline_config_for_mega_moe_sm90(
        SM90ArchSpec::smem_capacity,
        input.num_experts, input.hidden,
        config.block_m, config.block_n, config.block_k,
        config.num_dispatch_threads / 32, config.num_epilogue_threads / 32,
        config.direct_l2_scatter,
        config.num_stages,
        config.swap_ab,
        input.fp8_combine,
        true);
    return num_stages == config.num_stages and smem_size == config.smem_size and
           smem_size > 0 and smem_size <= SM90ArchSpec::smem_capacity;
}

static bool is_sm90_moe_launch_config_legal(
    const Sm90MoeHeuristicInput& input,
    const Sm90MoeLaunchConfig& config) {
    if (not is_sm90_moe_phase_config_legal(
            input, config.l1, 2 * input.intermediate_hidden, config.l1.num_sms) or
        not is_sm90_moe_phase_config_legal(
            input, config.l2, input.hidden, config.l2.num_sms))
        return false;
    // The four-math-warpgroup BN512 L1 path relies on packed BF16 partial
    // accumulation to stay within the CTA register budget.
    if (config.l1.block_n == 512 and
        not config.numerical.bf16_scaled_accum)
        return false;
    return not config.numerical.bf16_scaled_accum or
           (supports_sm90_moe_bf16_scaled_accum(config.l1) and
            supports_sm90_moe_bf16_scaled_accum(config.l2));
}

static MegaMoESM90Config make_generic_mega_moe_config_sm90(
    const Sm90MoeHeuristicInput& input,
    const bool conservative = false) {
    const int forced_block_m = conservative ? 0 :
        get_env<int>("DG_SM90_MOE_FORCE_BLOCK_M", 0);
    const int forced_epilogue_warpgroups = get_env<int>(
        "DG_SM90_MOE_FORCE_EPILOGUE_WG", 0);
    DG_HOST_ASSERT(forced_block_m == 0 or forced_block_m == 64 or forced_block_m == 128);
    DG_HOST_ASSERT(forced_epilogue_warpgroups == 0 or
                   forced_epilogue_warpgroups == 1 or
                   forced_epilogue_warpgroups == 2 or
                   forced_epilogue_warpgroups == 4);

    const bool use_bn256_split_n_env =
        not conservative and get_env<int>("DG_SM90_MOE_BN256_2WG", 1) != 0 and
        forced_block_m != 128;
    const bool swap_ab_env_enabled =
        not conservative and get_env<int>("DG_SM90_MOE_SWAP_AB", 1) != 0 and
        forced_block_m != 128;
    const int block_m = forced_block_m > 0 ? forced_block_m : 64;
    const int num_max_pool_tokens = layout::get_num_max_pool_tokens(
        input.num_ranks, input.num_max_tokens_per_rank,
        input.num_topk, input.num_experts_per_rank);
    const int block_k = 128;
    const auto load = get_sm90_moe_load(input);
    const auto shape_family = classify_sm90_moe_shape(
        input.hidden, input.intermediate_hidden);

    const bool prefer_swap_ab_block =
        swap_ab_env_enabled and block_m == 64 and
        should_use_swap_ab_for_mega_moe_sm90(
            input, load, shape_family, block_m, 256);
    const int block_n = prefer_swap_ab_block ? 128 :
        (block_m == 64 and use_bn256_split_n_env ? 256 : 128);
    const bool prefer_swap_ab_shape = prefer_swap_ab_block and block_n == 128;
    const int default_epilogue_warpgroups = block_m == 128 ? 2 :
        ((block_n == 256 or prefer_swap_ab_shape) ? 2 : 1);
    const int num_epilogue_warpgroups = conservative ? default_epilogue_warpgroups :
        (forced_epilogue_warpgroups > 0 ?
            forced_epilogue_warpgroups : default_epilogue_warpgroups);
    DG_HOST_ASSERT(block_m % num_epilogue_warpgroups == 0);
    DG_HOST_ASSERT(block_m != 128 or
                   (block_n == 128 and num_epilogue_warpgroups == 2));
    DG_HOST_ASSERT(block_m != 64 or block_n != 256 or
                   num_epilogue_warpgroups == 2);
    const int num_epilogue_threads = num_epilogue_warpgroups * 128;
    const bool swap_ab =
        swap_ab_env_enabled and block_n == 128 and
        should_use_swap_ab_for_mega_moe_sm90(
            input, load, shape_family, block_m, num_epilogue_threads);

    const bool compact_frontend = block_n >= 256 or swap_ab;
    const int forced_dispatch_warps = conservative ? -1 :
        get_env<int>("DG_SM90_MOE_DISPATCH_WARPS", -1);
    DG_HOST_ASSERT(forced_dispatch_warps == -1 or forced_dispatch_warps == 0 or
                   forced_dispatch_warps == 2 or forced_dispatch_warps == 4 or
                   forced_dispatch_warps == 8);
    const int num_dispatch_warps = forced_dispatch_warps > 0 ?
        forced_dispatch_warps : (compact_frontend ? 2 : 4);
    DG_HOST_ASSERT(not compact_frontend or num_dispatch_warps == 2);
    const int num_dispatch_threads = num_dispatch_warps * 32;
    const int num_non_epilogue_threads = compact_frontend ? 64 : 128;
    DG_HOST_ASSERT((num_dispatch_threads + num_non_epilogue_threads) % 128 == 0);

    const bool direct_l2_scatter_legal =
        (not swap_ab) and
        ((block_m == 64 and block_n == 256 and num_epilogue_warpgroups == 2) or
         block_n == 128);
    const bool direct_l2_scatter = conservative ? false :
        get_sm90_moe_bool_config(
            "DG_SM90_MOE_DIRECT_L2_SCATTER", false);
    DG_HOST_ASSERT(not direct_l2_scatter or direct_l2_scatter_legal);
    const bool l2_nmajor_schedule = conservative ? false :
        get_sm90_moe_bool_config(
            "DG_SM90_MOE_L2_NMAJOR", false);
    const bool one_warp_cleanup = conservative ? false :
        get_sm90_moe_bool_config(
            "DG_SM90_MOE_ONE_WARP_CLEANUP", false);

    const int num_experts_per_wave = conservative ?
        get_num_experts_per_wave_for_mega_moe(
            input.num_experts_per_rank, input.num_tokens, input.num_topk,
            input.intermediate_hidden, block_m, block_n, input.launch_num_sms) :
        get_num_experts_per_wave_for_mega_moe_sm90(
            load,
            input.num_experts_per_rank, input.num_tokens, input.num_topk,
            input.intermediate_hidden, block_m, block_n, input.launch_num_sms);
    DG_HOST_ASSERT(num_experts_per_wave > 0 and
                   num_experts_per_wave <= input.num_experts_per_rank and
                   input.num_experts_per_rank % num_experts_per_wave == 0);

    const auto [num_stages, smem_size] = get_pipeline_config_for_mega_moe_sm90(
        SM90ArchSpec::smem_capacity,
        input.num_experts, input.hidden,
        block_m, block_n, block_k,
        num_dispatch_threads / 32, num_epilogue_threads / 32,
        direct_l2_scatter,
        0,
        swap_ab,
        input.fp8_combine);
    DG_HOST_ASSERT(num_stages >= 2 and smem_size > 0);
    const int sf_pool_stride_tokens =
        layout::get_num_padded_sf_pool_tokens(num_max_pool_tokens, block_m);
    return {
        block_m, block_n, block_k,
        num_max_pool_tokens, input.num_padded_sf_pool_tokens, sf_pool_stride_tokens,
        128, 128,
        num_experts_per_wave,
        input.launch_num_sms,
        num_stages, smem_size,
        num_dispatch_threads, num_non_epilogue_threads, num_epilogue_threads,
        direct_l2_scatter, l2_nmajor_schedule, one_warp_cleanup, swap_ab
    };
}

static bool select_sm90_moe_high_sm_bf16_scaled_accum(
    const Sm90MoeHeuristicInput& input,
    const Sm90MoeShapeFamily shape_family,
    const Sm90MoeLoad& load) {
    if (input.num_topk != 6)
        return false;
    if (shape_family == Sm90MoeShapeFamily::Compact and
        input.num_experts_per_rank == 32) {
        return load.less_equal(3) or
               (load.greater_than(6) and load.less_equal(384));
    }
    if (shape_family == Sm90MoeShapeFamily::Wide and
        input.num_experts_per_rank == 48) {
        return load.less_equal(4) or load.greater_than(8);
    }
    return false;
}

struct Sm90MoePhaseTuning {
    int block_m, block_n, block_k;
    int num_experts_per_wave, num_sms, num_stages;
    int num_dispatch_threads, num_non_epilogue_threads, num_epilogue_threads;
    bool direct_l2_scatter, nmajor_schedule, one_warp_cleanup, swap_ab;
};

struct Sm90MoeScheduleTuning {
    bool selected;
    Sm90MoePhaseTuning l1, l2;
};

static Sm90MoePhaseTuning make_sm90_moe_phase_tuning(
    const MegaMoESM90Config& config) {
    return {
        config.block_m, config.block_n, config.block_k,
        config.num_experts_per_wave, config.num_sms, config.num_stages,
        config.num_dispatch_threads, config.num_non_epilogue_threads,
        config.num_epilogue_threads,
        config.direct_l2_scatter, config.nmajor_schedule,
        config.one_warp_cleanup, config.swap_ab,
    };
}

static void set_sm90_moe_specialized_phase_tuning(
    Sm90MoePhaseTuning& tuning,
    const int block_n,
    const int block_k,
    const int num_experts_per_wave,
    const int num_stages,
    const int num_sms) {
    tuning.block_m = 64;
    tuning.block_n = block_n;
    tuning.block_k = block_k;
    tuning.num_experts_per_wave = num_experts_per_wave;
    tuning.num_sms = num_sms;
    tuning.num_stages = num_stages;
    tuning.num_dispatch_threads = block_n >= 256 ? 64 : 128;
    tuning.num_non_epilogue_threads = block_n >= 256 ? 64 : 128;
    tuning.num_epilogue_threads = block_n >= 256 ? block_n : 128;
    tuning.direct_l2_scatter = false;
    tuning.nmajor_schedule = false;
    tuning.one_warp_cleanup = false;
    tuning.swap_ab = false;
}

static Sm90MoeScheduleTuning lookup_sm90_moe_tuning(
    const Sm90MoeHeuristicInput& input,
    const Sm90MoeHardwareProfile hardware_profile,
    const Sm90MoeShapeFamily shape_family,
    const Sm90MoeLoad& load,
    const Sm90MoeLaunchConfig& generic) {
    Sm90MoeScheduleTuning tuning {
        false,
        make_sm90_moe_phase_tuning(generic.l1),
        make_sm90_moe_phase_tuning(generic.l2),
    };

    const auto profile_tuning = lookup_sm90_moe_profile_tuning(
        hardware_profile, shape_family, input, load,
        tuning.l1.block_m, tuning.l1.block_n);
    if (profile_tuning.selected) {
        tuning.selected = true;
        if (profile_tuning.num_experts_per_wave > 0) {
            tuning.l1.num_experts_per_wave = profile_tuning.num_experts_per_wave;
            tuning.l2.num_experts_per_wave = profile_tuning.num_experts_per_wave;
        }
        tuning.l1.num_stages = tuning.l2.num_stages = profile_tuning.num_stages;
        tuning.l1.direct_l2_scatter = tuning.l2.direct_l2_scatter =
            profile_tuning.direct_l2_scatter;
        tuning.l1.nmajor_schedule = false;
        tuning.l2.nmajor_schedule = profile_tuning.l2_nmajor_schedule;
        tuning.l1.one_warp_cleanup = tuning.l2.one_warp_cleanup =
            profile_tuning.one_warp_cleanup;
    }

    if (hardware_profile != Sm90MoeHardwareProfile::HighSm)
        return tuning;

    int l1_block_n = 256, l1_block_k = 128, l1_epw = 0, l1_stages = 0;
    int l2_block_n = 256, l2_block_k = 128, l2_epw = 0, l2_stages = 0;
    int phase_num_sms = input.launch_num_sms;
    bool l1_nmajor = false, one_warp_cleanup = false;
    bool specialized = false;

    if (shape_family == Sm90MoeShapeFamily::Compact and load.in_open_closed(12, 32)) {
        specialized = true;
        l1_epw = l2_epw = 4;
        l1_stages = l2_stages = 3;
    } else if (shape_family == Sm90MoeShapeFamily::Compact and load.in_open_closed(128, 256)) {
        specialized = true;
        l1_epw = l2_epw = 32;
        l1_stages = l2_stages = 4;
    } else if (shape_family == Sm90MoeShapeFamily::Compact and load.greater_than(1024)) {
        specialized = true;
        l1_epw = l2_epw = 32;
        // Keep the previously validated value. A 4 -> 3 stage retune is a
        // performance change and must not be hidden inside this refactor.
        l1_stages = l2_stages = 4;
    } else if (shape_family == Sm90MoeShapeFamily::Wide and load.in_open_closed(8, 24)) {
        specialized = true;
        l1_block_n = 512;
        l1_epw = l2_epw = 16;
        l1_stages = 2;
        l2_stages = 3;
    } else if (shape_family == Sm90MoeShapeFamily::Wide and load.in_open_closed(24, 48)) {
        specialized = true;
        l1_block_k = 256;
        l1_epw = 8;
        l2_epw = 48;
        l1_stages = 2;
        l2_stages = 3;
        phase_num_sms = 128;
    } else if (shape_family == Sm90MoeShapeFamily::Wide and load.in_open_closed(48, 96)) {
        specialized = true;
        l1_block_n = 512;
        l1_epw = l2_epw = 16;
        l1_stages = 2;
        l2_stages = 3;
        l1_nmajor = true;
    } else if (shape_family == Sm90MoeShapeFamily::Wide and load.in_open_closed(96, 192)) {
        specialized = true;
        l1_block_n = 512;
        l1_epw = l2_epw = 16;
        l1_stages = 2;
        l2_stages = 3;
        l1_nmajor = true;
        one_warp_cleanup = true;
    } else if (shape_family == Sm90MoeShapeFamily::Wide and load.greater_than(192)) {
        specialized = true;
        l1_block_n = 512;
        l1_epw = l2_epw = 16;
        l1_stages = 2;
        l2_stages = 3;
        one_warp_cleanup = true;
    }
    if (not specialized)
        return tuning;

    tuning.selected = true;
    set_sm90_moe_specialized_phase_tuning(
        tuning.l1, l1_block_n, l1_block_k, l1_epw, l1_stages, phase_num_sms);
    set_sm90_moe_specialized_phase_tuning(
        tuning.l2, l2_block_n, l2_block_k, l2_epw, l2_stages, phase_num_sms);
    tuning.l1.nmajor_schedule = l1_nmajor;
    tuning.l2.nmajor_schedule = true;
    tuning.l2.one_warp_cleanup = one_warp_cleanup;
    return tuning;
}

static bool try_materialize_sm90_moe_phase_tuning(
    const Sm90MoeHeuristicInput& input,
    MegaMoESM90Config& config,
    const Sm90MoePhaseTuning& tuning) {
    config.block_m = tuning.block_m;
    config.block_n = tuning.block_n;
    config.block_k = tuning.block_k;
    config.num_experts_per_wave = tuning.num_experts_per_wave;
    config.num_sms = tuning.num_sms;
    config.num_dispatch_threads = tuning.num_dispatch_threads;
    config.num_non_epilogue_threads = tuning.num_non_epilogue_threads;
    config.num_epilogue_threads = tuning.num_epilogue_threads;
    config.direct_l2_scatter = tuning.direct_l2_scatter;
    config.nmajor_schedule = tuning.nmajor_schedule;
    config.one_warp_cleanup = tuning.one_warp_cleanup;
    config.swap_ab = tuning.swap_ab;
    config.sf_pool_stride_tokens = layout::get_num_padded_sf_pool_tokens(
        config.num_max_pool_tokens, config.block_m);

    const auto [num_stages, smem_size] = get_pipeline_config_for_mega_moe_sm90(
        SM90ArchSpec::smem_capacity,
        input.num_experts, input.hidden,
        config.block_m, config.block_n, config.block_k,
        config.num_dispatch_threads / 32, config.num_epilogue_threads / 32,
        config.direct_l2_scatter,
        tuning.num_stages,
        config.swap_ab,
        input.fp8_combine,
        true);
    if (num_stages == 0 or smem_size == 0)
        return false;
    config.num_stages = num_stages;
    config.smem_size = smem_size;
    return true;
}

static bool try_apply_sm90_moe_tuning(
    const Sm90MoeHeuristicInput& input,
    Sm90MoeScheduleTuning tuning,
    Sm90MoeLaunchConfig& config) {
    if (not tuning.selected)
        return true;
    if (const int forced = get_env<int>("DG_SM90_MOE_EXPERTS_PER_WAVE"); forced > 0) {
        DG_HOST_ASSERT(forced <= input.num_experts_per_rank);
        DG_HOST_ASSERT(input.num_experts_per_rank % forced == 0);
        tuning.l1.num_experts_per_wave = forced;
        tuning.l2.num_experts_per_wave = forced;
    }
    const bool direct_l2_scatter = get_sm90_moe_bool_config(
        "DG_SM90_MOE_DIRECT_L2_SCATTER", tuning.l2.direct_l2_scatter);
    tuning.l1.direct_l2_scatter = direct_l2_scatter;
    tuning.l2.direct_l2_scatter = direct_l2_scatter;
    tuning.l2.nmajor_schedule = get_sm90_moe_bool_config(
        "DG_SM90_MOE_L2_NMAJOR", tuning.l2.nmajor_schedule);
    const bool one_warp_cleanup = get_sm90_moe_bool_config(
        "DG_SM90_MOE_ONE_WARP_CLEANUP", tuning.l2.one_warp_cleanup);
    tuning.l1.one_warp_cleanup = one_warp_cleanup;
    tuning.l2.one_warp_cleanup = one_warp_cleanup;
    return try_materialize_sm90_moe_phase_tuning(input, config.l1, tuning.l1) and
           try_materialize_sm90_moe_phase_tuning(input, config.l2, tuning.l2);
}

static Sm90MoeLaunchConfig select_mega_moe_sm90(
    const Sm90MoeHeuristicInput& input) {
    const auto hardware_profile = classify_sm90_moe_hardware(
        input.launch_num_sms,
        input.device_profile_override);
    const auto shape_family = classify_sm90_moe_shape(
        input.hidden, input.intermediate_hidden);
    const auto load = get_sm90_moe_load(input);

    auto generic = make_generic_mega_moe_config_sm90(input);
    Sm90MoeLaunchConfig result {
        generic, generic,
        {input.fp8_combine, false}
    };
    result.l1.nmajor_schedule = false;

    // The computed generic route is always available. If debug ENV overrides
    // make it illegal, replace it with the conservative computed route before
    // applying any profile tuning.
    if (not is_sm90_moe_launch_config_legal(input, result)) {
        generic = make_generic_mega_moe_config_sm90(input, true);
        result.l1 = result.l2 = generic;
        result.l1.nmajor_schedule = false;
    }
    DG_HOST_ASSERT(is_sm90_moe_launch_config_legal(input, result));

    if (hardware_profile != Sm90MoeHardwareProfile::Generic) {
        auto candidate = result;
        candidate.numerical.bf16_scaled_accum =
            hardware_profile == Sm90MoeHardwareProfile::HighSm and
            select_sm90_moe_high_sm_bf16_scaled_accum(
                input, shape_family, load);
        const auto tuning = lookup_sm90_moe_tuning(
            input, hardware_profile, shape_family, load, result);
        if (try_apply_sm90_moe_tuning(input, tuning, candidate) and
            is_sm90_moe_launch_config_legal(input, candidate)) {
            result = candidate;
        }
    }
    DG_HOST_ASSERT(is_sm90_moe_launch_config_legal(input, result));

    if (get_env<int>("DG_JIT_DEBUG") or get_env<int>("DG_PRINT_CONFIGS")) {
        const auto key = fmt::format(
            "Sm90MoeLaunchConfig(num_ranks={}, num_experts={}, hidden={}, intermediate_hidden={}, num_max_tokens_per_rank={}, num_tokens={}, num_topk={})",
            input.num_ranks, input.num_experts, input.hidden,
            input.intermediate_hidden, input.num_max_tokens_per_rank,
            input.num_tokens, input.num_topk);
        static std::unordered_set<std::string> printed;
        if (printed.count(key) == 0) {
            std::cout << key << ": profile=" << static_cast<int>(hardware_profile)
                      << ", shape_family=" << static_cast<int>(shape_family)
                      << ", l1=" << result.l1
                      << ", l2=" << result.l2
                      << ", fp8_combine=" << result.numerical.fp8_combine
                      << ", bf16_scaled_accum=" << result.numerical.bf16_scaled_accum
                      << std::endl;
            printed.insert(key);
        }
    }
    return result;
}

} // namespace deep_gemm
