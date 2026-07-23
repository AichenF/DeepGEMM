#pragma once

#include <algorithm>
#include <cstdint>
#include <iostream>
#include <string>
#include <tuple>
#include <unordered_set>
#include <utility>

#include <deep_gemm/layout/mega_moe.cuh>

#include "../../utils/exception.hpp"
#include "../../utils/system.hpp"
#include "mega_moe.hpp"
#include "sm90.hpp"

namespace deep_gemm {

// The framework and API have already selected the BN128 split family before
// this selector runs. This header chooses only that family's concrete launch
// schedule; it never changes the weight layout or selects the fused family.

struct SM90NVFP4MegaMoEConfig {
    int block_m, block_n, block_k;
    int cluster_size;

    int num_max_pool_tokens;
    int num_padded_sf_pool_tokens;

    int swizzle_acts_mode;
    int num_experts_per_wave;

    int num_stages, smem_size;

    int num_dispatch_threads;
    int num_non_epilogue_threads;
    int num_epilogue_threads;

    friend std::ostream& operator << (
            std::ostream& os,
            const SM90NVFP4MegaMoEConfig& config) {
        os << "SM90NVFP4MegaMoEConfig("
           << "block_m=" << config.block_m
           << ", block_n=" << config.block_n
           << ", block_k=" << config.block_k
           << ", cluster_size=" << config.cluster_size
           << ", num_max_pool_tokens=" << config.num_max_pool_tokens
           << ", num_padded_sf_pool_tokens="
           << config.num_padded_sf_pool_tokens
           << ", swizzle_acts_mode=" << config.swizzle_acts_mode
           << ", num_experts_per_wave=" << config.num_experts_per_wave
           << ", num_stages=" << config.num_stages
           << ", smem_size=" << config.smem_size
           << ", num_dispatch_threads=" << config.num_dispatch_threads
           << ", num_non_epilogue_threads="
           << config.num_non_epilogue_threads
           << ", num_epilogue_threads=" << config.num_epilogue_threads
           << ")";
        return os;
    }
};

struct SM90NVFP4MegaMoEInput {
    int launch_num_sms;

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

struct SM90NVFP4MegaMoELoad {
    int64_t routed_tokens;
    int64_t local_experts;
    float expected_tokens_per_local_expert;

    bool valid() const {
        return routed_tokens > 0 && local_experts > 0;
    }

    bool greater_equal(const int64_t value) const {
        return routed_tokens >= value * local_experts;
    }
};

struct SM90NVFP4MegaMoEPhaseTuning {
    int block_m;
    int block_n;
    int block_k;
    int cluster_size;
    int num_active_dispatch_warps;
    int num_dispatch_threads;
    int num_non_epilogue_threads;
    int num_epilogue_threads;
    int acts_sf_gran_k;
    bool l2_only;
};

struct SM90NVFP4MegaMoEScheduleTuning {
    SM90NVFP4MegaMoEPhaseTuning l1;
    SM90NVFP4MegaMoEPhaseTuning l2;
    bool dispatch_dequant;
    bool l2_arrival_counter;
};

struct SM90NVFP4MegaMoEPlan {
    SM90NVFP4MegaMoEConfig l1_config;
    SM90NVFP4MegaMoEConfig l2_config;
    bool dispatch_dequant;
    bool l2_arrival_counter;
};

static SM90NVFP4MegaMoELoad get_sm90_nvfp4_mega_moe_load(
        const SM90NVFP4MegaMoEInput& input) {
    const int64_t routed_tokens =
        static_cast<int64_t>(input.num_tokens) * input.num_topk;
    const SM90NVFP4MegaMoELoad load {
        routed_tokens,
        input.num_experts_per_rank,
        static_cast<float>(routed_tokens) / input.num_experts_per_rank,
    };
    DG_HOST_ASSERT(load.valid());
    return load;
}

static int get_num_experts_per_wave_for_sm90_nvfp4_mega_moe(
        const SM90NVFP4MegaMoEInput& input,
        const SM90NVFP4MegaMoEPhaseTuning& phase) {
    const float expected_tokens_per_expert =
        static_cast<float>(input.num_tokens) *
        input.num_topk /
        input.num_experts_per_rank;
    if (phase.block_m == 64 &&
        (expected_tokens_per_expert <= 1.0f ||
         expected_tokens_per_expert >= 4.0f))
        return input.num_experts_per_rank;
    return get_num_experts_per_wave_for_mega_moe(
        input.num_experts_per_rank,
        input.num_tokens,
        input.num_topk,
        input.intermediate_hidden,
        phase.block_m,
        phase.block_n,
        input.launch_num_sms);
}

static SM90NVFP4MegaMoEScheduleTuning
select_sm90_nvfp4_mega_moe_tuning(
        const SM90NVFP4MegaMoELoad& load) {
    constexpr SM90NVFP4MegaMoEPhaseTuning l1 {
        128, 128, 128,
        2,
        2,
        128, 128, 256,
        // This is only the proven split-L1 shared-memory reservation. Kernel
        // activation-scale semantics remain per-128.
        64,
        false,
    };
    constexpr SM90NVFP4MegaMoEPhaseTuning l2 {
        128, 128, 128,
        1,
        0,
        0, 128, 256,
        128,
        true,
    };

    return {
        l1,
        l2,
        load.greater_equal(256),
        load.expected_tokens_per_local_expert <= 32.0f ||
            load.expected_tokens_per_local_expert >= 128.0f,
    };
}

static std::pair<int, int>
get_sm90_nvfp4_mega_moe_pipeline_config(
        const SM90NVFP4MegaMoEInput& input,
        const SM90NVFP4MegaMoELoad& load,
        const SM90NVFP4MegaMoEPhaseTuning& phase,
        const SM90NVFP4MegaMoEConfig& config) {
    const auto align = [](int value, int alignment) {
        return ((value + alignment - 1) / alignment) * alignment;
    };
    constexpr int kSmemAlignment = 1024;

    DG_HOST_ASSERT(phase.num_active_dispatch_warps >= 0);
    DG_HOST_ASSERT(
        phase.num_active_dispatch_warps <=
        config.num_dispatch_threads / 32);
    const int num_dispatch_warps = phase.num_active_dispatch_warps;
    const int num_epilogue_warps = config.num_epilogue_threads / 32;
    const int num_epilogue_warpgroups = num_epilogue_warps / 4;

    const bool split_n_warpgroups =
        config.block_m == 64 && config.block_n == 256 &&
        num_epilogue_warpgroups == 2;
    const bool split_mn_warpgroups =
        config.block_m == 128 && config.block_n == 256 &&
        num_epilogue_warpgroups == 4;
    const int wg_split_m = split_n_warpgroups ? 1 :
        (split_mn_warpgroups ? 2 : num_epilogue_warpgroups);
    const int wg_split_n = split_n_warpgroups ?
        num_epilogue_warpgroups : (split_mn_warpgroups ? 2 : 1);
    DG_HOST_ASSERT(
        wg_split_m * wg_split_n == num_epilogue_warpgroups);
    const int wg_block_m = config.block_m / wg_split_m;
    const int wg_block_n = config.block_n / wg_split_n;
    const int wg_l1_out_block_n = wg_block_n / 2;

    const int smem_expert_count_size = align(
        input.num_experts * static_cast<int>(sizeof(uint32_t)),
        kSmemAlignment);
    const int smem_send_buffers_size = align(
        static_cast<int>(layout::Buffer(
            layout::Data(input.hidden),
            num_dispatch_warps,
            1).get_num_bytes()),
        kSmemAlignment);
    const int smem_dispatch_size =
        smem_expert_count_size + smem_send_buffers_size;
    const int smem_nvfp4_lut = align(128 * 8, kSmemAlignment);

    const int smem_cd_l1 =
        num_epilogue_warpgroups * wg_block_m * wg_l1_out_block_n;
    DG_HOST_ASSERT(
        phase.acts_sf_gran_k == 64 || phase.acts_sf_gran_k == 128);
    const bool split_n_shares_l1_sf =
        split_n_warpgroups &&
        wg_l1_out_block_n < phase.acts_sf_gran_k;
    const int smem_cd_l1_shared_sf_slots = split_n_shares_l1_sf ?
        num_epilogue_warpgroups * config.block_m : 0;
    const int smem_cd_l1_shared_sf =
        smem_cd_l1_shared_sf_slots * static_cast<int>(sizeof(float));
    const int smem_cd = phase.l2_only ? 0 : align(
        smem_cd_l1 + smem_cd_l1_shared_sf,
        kSmemAlignment);

    const int num_sfa_groups_per_bk =
        config.block_k / phase.acts_sf_gran_k;
    const int smem_sfa_per_stage = align(
        num_sfa_groups_per_bk * config.block_m *
            static_cast<int>(sizeof(float)),
        128);
    const int smem_per_stage =
        config.block_m * config.block_k +
        config.block_n * config.block_k +
        smem_sfa_per_stage;
    const int smem_barriers_fixed =
        (num_dispatch_warps + 2 * num_epilogue_warps) * 8;
    // The BN128 split family always dequantizes B in the loader warpgroup, so
    // every stage owns full, empty, and dequant barriers.
    const int smem_barriers_per_stage = 3 * 8;
    const int smem_fixed =
        smem_dispatch_size +
        smem_nvfp4_lut +
        smem_cd +
        smem_barriers_fixed;
    const int max_num_stages =
        (SM90ArchSpec::smem_capacity - smem_fixed) /
        (smem_per_stage + smem_barriers_per_stage);
    const int num_stages =
        !phase.l2_only &&
        config.block_n == 128 &&
        load.expected_tokens_per_local_expert > 8.0f &&
        max_num_stages > 6 ?
        6 : max_num_stages;
    DG_HOST_ASSERT(max_num_stages >= 2);
    DG_HOST_ASSERT(num_stages >= 2 && num_stages <= max_num_stages);
    return {
        num_stages,
        smem_fixed +
            num_stages *
            (smem_per_stage + smem_barriers_per_stage),
    };
}

static SM90NVFP4MegaMoEConfig materialize_sm90_nvfp4_mega_moe_phase(
        const SM90NVFP4MegaMoEInput& input,
        const SM90NVFP4MegaMoELoad& load,
        const SM90NVFP4MegaMoEPhaseTuning& phase) {
    SM90NVFP4MegaMoEConfig config {
        phase.block_m,
        phase.block_n,
        phase.block_k,
        phase.cluster_size,
        layout::get_num_max_pool_tokens(
            input.num_ranks,
            input.num_max_tokens_per_rank,
            input.num_topk,
            input.num_experts_per_rank),
        input.num_padded_sf_pool_tokens,
        128,
        get_num_experts_per_wave_for_sm90_nvfp4_mega_moe(
            input,
            phase),
        0,
        0,
        phase.num_dispatch_threads,
        phase.num_non_epilogue_threads,
        phase.num_epilogue_threads,
    };
    std::tie(config.num_stages, config.smem_size) =
        get_sm90_nvfp4_mega_moe_pipeline_config(
            input,
            load,
            phase,
            config);
    return config;
}

static SM90NVFP4MegaMoEPlan materialize_sm90_nvfp4_mega_moe_tuning(
        const SM90NVFP4MegaMoEInput& input,
        const SM90NVFP4MegaMoELoad& load,
        const SM90NVFP4MegaMoEScheduleTuning& tuning) {
    const auto l1_config = materialize_sm90_nvfp4_mega_moe_phase(
        input,
        load,
        tuning.l1);
    const auto l2_config = materialize_sm90_nvfp4_mega_moe_phase(
        input,
        load,
        tuning.l2);
    return {
        l1_config,
        l2_config,
        tuning.dispatch_dequant,
        tuning.l2_arrival_counter,
    };
}

static bool is_sm90_nvfp4_mega_moe_plan_legal(
        const SM90NVFP4MegaMoEInput& input,
        const SM90NVFP4MegaMoEPlan& plan) {
    const auto valid_phase = [&](const SM90NVFP4MegaMoEConfig& config) {
        return config.block_m > 0 &&
            config.block_n > 0 &&
            config.block_k > 0 &&
            config.num_experts_per_wave > 0 &&
            config.num_experts_per_wave <= input.num_experts_per_rank &&
            input.num_experts_per_rank % config.num_experts_per_wave == 0 &&
            config.num_stages >= 2 &&
            config.smem_size > 0 &&
            config.smem_size <= SM90ArchSpec::smem_capacity;
    };
    return valid_phase(plan.l1_config) &&
        valid_phase(plan.l2_config) &&
        plan.l1_config.block_m == 128 &&
        plan.l1_config.block_n == 128 &&
        plan.l1_config.block_k == 128 &&
        plan.l1_config.cluster_size == 2 &&
        plan.l2_config.block_m == 128 &&
        plan.l2_config.block_n == 128 &&
        plan.l2_config.block_k == 128 &&
        plan.l2_config.cluster_size == 1;
}

static SM90NVFP4MegaMoEPlan select_sm90_nvfp4_split_mega_moe(
        const SM90NVFP4MegaMoEInput& input) {
    DG_HOST_ASSERT(input.launch_num_sms > 0);
    DG_HOST_ASSERT(input.num_ranks > 0);
    DG_HOST_ASSERT(input.num_experts_per_rank > 0);
    DG_HOST_ASSERT(
        input.num_experts ==
        input.num_experts_per_rank * input.num_ranks);
    DG_HOST_ASSERT(input.num_max_tokens_per_rank > 0);
    DG_HOST_ASSERT(
        input.num_tokens > 0 &&
        input.num_tokens <= input.num_max_tokens_per_rank);
    DG_HOST_ASSERT(input.num_topk > 0);
    DG_HOST_ASSERT(input.hidden > 0 && input.hidden % 128 == 0);
    DG_HOST_ASSERT(
        input.intermediate_hidden > 0 &&
        input.intermediate_hidden % 128 == 0);
    DG_HOST_ASSERT(input.num_padded_sf_pool_tokens > 0);

    const auto load = get_sm90_nvfp4_mega_moe_load(input);
    const auto tuning = select_sm90_nvfp4_mega_moe_tuning(load);
    const auto plan =
        materialize_sm90_nvfp4_mega_moe_tuning(input, load, tuning);
    DG_HOST_ASSERT(is_sm90_nvfp4_mega_moe_plan_legal(input, plan));

    if (get_env<int>("DG_JIT_DEBUG") ||
        get_env<int>("DG_PRINT_CONFIGS")) {
        const auto key = fmt::format(
            "SM90NVFP4MegaMoEPlan(num_ranks={}, num_experts={}, "
            "hidden={}, intermediate_hidden={}, "
            "num_max_tokens_per_rank={}, num_tokens={}, num_topk={}, "
            "layout_block_n=128)",
            input.num_ranks,
            input.num_experts,
            input.hidden,
            input.intermediate_hidden,
            input.num_max_tokens_per_rank,
            input.num_tokens,
            input.num_topk);
        static std::unordered_set<std::string> printed;
        if (printed.count(key) == 0) {
            std::cout << key
                      << ": l1=" << plan.l1_config
                      << ", l2=" << plan.l2_config
                      << std::endl;
            printed.insert(key);
        }
    }
    return plan;
}

}  // namespace deep_gemm
