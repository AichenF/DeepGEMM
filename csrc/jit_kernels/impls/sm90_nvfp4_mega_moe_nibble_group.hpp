#pragma once

#include "sm90_nvfp4_mega_moe.hpp"

namespace deep_gemm {

class SM90NVFP4NibbleGroupMegaMoERuntime final
    : public LaunchRuntime<SM90NVFP4NibbleGroupMegaMoERuntime> {
public:
    using Args = SM90NVFP4MegaMoERuntime::Args;

    static std::string generate_impl(const Args& args) {
        DG_HOST_ASSERT(args.phase_mode == SM90NVFP4MegaMoERuntime::kFusedPhaseMode);
        const std::string phase_template_args = fmt::format(
            "/* kPhaseProfileRequested */ {},\n"
            "        /* kLoaderDequantRequested */ {},\n"
            "        /* kSwapABRequested */ {},\n"
            "        /* kDp4aSelectorPack */ {},\n"
            "        /* kHybridLowSelectorPack */ {},\n"
            "        /* kSingleActiveDispatchWarp */ {}",
            args.phase_profile ? "true" : "false",
            args.loader_dequant ? "true" : "false",
            args.swap_ab ? "true" : "false",
            args.dp4a_selector_pack ? "true" : "false",
            args.hybrid_low_selector_pack ? "true" : "false",
            args.single_active_dispatch_warp ? "true" : "false");
        return fmt::format(R"(
#include <deep_gemm/impls/sm90_nvfp4_mega_moe_nibble_group.cuh>

using namespace deep_gemm;

static void __instantiate_kernel() {{
    auto ptr = reinterpret_cast<void*>(&sm90_nvfp4_mega_moe_nibble_group_fused_impl<
        {},
        {}, {},
        {}, {},
        {},
        {}, {}, {},
        {},
        {},
        {},
        {}, {}, {},
        {},
        {}, {},
        {},
        {},
        {}
    >);
}};
)",
            args.num_max_tokens_per_rank,
            args.hidden, args.intermediate_hidden,
            args.num_experts, args.num_topk,
            args.config.num_experts_per_wave,
            args.config.block_m, args.config.block_n, args.config.block_k,
            args.config.num_max_pool_tokens,
            args.config.num_padded_sf_pool_tokens,
            args.config.num_stages,
            args.config.num_dispatch_threads, args.config.num_non_epilogue_threads,
            args.config.num_epilogue_threads,
            args.config.cluster_size,
            args.launch_args.grid_dim.first, args.num_ranks,
            to_string(args.activation_clamp),
            args.fast_math ? "true" : "false",
            phase_template_args);
    }

    static void launch_impl(
            const KernelHandle& kernel, const LaunchConfigHandle& config, Args args) {
        DG_CUDA_UNIFIED_CHECK(launch_kernel(
            kernel, config,
            args.y,
            args.cumulative_local_expert_recv_stats,
            args.num_tokens,
            args.sym_buffer_ptrs,
            args.tensor_map_l1_acts,
            args.tensor_map_l1_acts_sf,
            args.tensor_map_l1_weights,
            args.tensor_map_l1_output,
            args.tensor_map_l2_acts,
            args.tensor_map_l2_acts_sf,
            args.tensor_map_l2_weights,
            args.l1_global_scales,
            args.l2_global_scales));
    }
};

static void sm90_nvfp4_nibble_group_mega_moe(
    const torch::Tensor& y,
    const torch::Tensor& l1_acts, const torch::Tensor& l1_acts_sf,
    const torch::Tensor& l2_acts, const torch::Tensor& l2_acts_sf,
    const torch::Tensor& l1_weights, const torch::Tensor& l2_weights,
    const torch::Tensor& l1_weights_sf, const torch::Tensor& l2_weights_sf,
    const std::optional<torch::Tensor> cumulative_local_expert_recv_stats,
    const std::optional<torch::Tensor> l1_global_scales,
    const std::optional<torch::Tensor> l2_global_scales,
    const std::vector<int64_t>& sym_buffer_ptrs,
    const int& rank_idx, const int& num_max_tokens_per_rank,
    const int& num_experts_per_rank,
    const int& num_tokens, const int& num_topk,
    const int& hidden, const int& intermediate_hidden,
    const float& activation_clamp,
    const bool& fast_math
) {
    DG_HOST_ASSERT(intermediate_hidden <= 2048);
    const int num_ranks = static_cast<int>(sym_buffer_ptrs.size());
    const int num_experts = num_experts_per_rank * num_ranks;
    const int num_padded_sf_pool_tokens = static_cast<int>(l1_acts_sf.size(0));
    const int block_n_from_layout = static_cast<int>(l1_weights_sf.size(3));
    DG_HOST_ASSERT(block_n_from_layout == 256);
    const auto plan = get_nvfp4_mega_moe_plan_sm90(
        num_ranks, num_experts, num_experts_per_rank,
        num_max_tokens_per_rank, num_tokens, num_topk,
        hidden, intermediate_hidden, num_padded_sf_pool_tokens,
        block_n_from_layout);
    DG_HOST_ASSERT(plan.use_fused_phase);
    auto config = plan.l1_or_fused_config;
    DG_HOST_ASSERT(config.block_m == 64 && config.block_n == 256);
    const int routed_tokens = num_tokens * num_topk;
    const float expected_tokens_per_local_expert =
        static_cast<float>(routed_tokens) / num_experts_per_rank;
    // Match the production swapAB cutoff. The candidate's original cutoff-16
    // was validated on Flash-like shapes only; 8-rank ABBA medians on
    // MiMo-Pro (48 local experts) show +2.0%/+8.5% regressions at expected
    // 9/16 vs the default path, mirroring the production-decoder E4 result.
    // Revisit per-geometry if Flash-like shapes land on this path.
    const bool candidate_swap_ab =
        routed_tokens <= 8 * num_experts_per_rank;
    // The kernel now carries kSingleActiveDispatchWarp, so a verbatim copy of a
    // single-active-warp 3-stage plan is consistent. When this candidate widens
    // swapAB to expected 9..16 the plan config was built without swapAB —
    // recompute, then attempt the same 3-stage single-active-warp upgrade the
    // production plan builder applies at expected<=8.
    bool single_active_dispatch_warp = plan.single_active_dispatch_warp;
    if (candidate_swap_ab != plan.swap_ab) {
        single_active_dispatch_warp = false;
        std::tie(config.num_stages, config.smem_size) =
            get_nvfp4_pipeline_config_for_mega_moe_sm90(
                num_experts, hidden, expected_tokens_per_local_expert,
                config, plan.loader_dequant, true, candidate_swap_ab, false);
        if (candidate_swap_ab && config.num_stages == 2) {
            MegaMoESM90Config upgraded = config;
            std::tie(upgraded.num_stages, upgraded.smem_size) =
                get_nvfp4_pipeline_config_for_mega_moe_sm90(
                    num_experts, hidden, expected_tokens_per_local_expert,
                    upgraded, plan.loader_dequant, true, candidate_swap_ab, false,
                    true);
            if (upgraded.num_stages > config.num_stages) {
                config = upgraded;
                single_active_dispatch_warp = true;
            }
        }
    }
    const int weight_storage_k = static_cast<int>(l1_weights.size(2));
    const int weight_k_blocks = static_cast<int>(l1_weights_sf.size(2));
    DG_HOST_ASSERT(weight_storage_k == weight_k_blocks * kSM90NVFP4BStoragePerKBlock);

    constexpr int kGranK = 128;
    constexpr int kL2ActsSFGranK = 64;
    const auto tensor_map_l1_acts = make_tma_2d_desc(
        l1_acts, hidden, config.num_max_pool_tokens,
        config.block_k, config.block_m,
        static_cast<int>(l1_acts.stride(-2)), config.swizzle_acts_mode);
    const auto tensor_map_l1_acts_sf = make_tma_sf_desc(
        cute::UMMA::Major::MN, l1_acts_sf,
        config.num_padded_sf_pool_tokens, hidden,
        config.block_m, kGranK, 1, 0);
    const auto tensor_map_l1_weights = make_tma_2d_desc(
        l1_weights, static_cast<int>(l1_weights.size(2)),
        num_experts_per_rank * intermediate_hidden * 2,
        kSM90NVFP4BStoragePerKBlock, config.block_n,
        static_cast<int>(l1_weights.stride(-2)), 0);

    const int num_epilogue_warpgroups = config.num_epilogue_threads / 128;
    const bool split_n_warpgroups = config.block_m == 64 && config.block_n == 256 &&
        num_epilogue_warpgroups == 2;
    DG_HOST_ASSERT(split_n_warpgroups);
    const int wg_block_m = config.block_m;
    const int wg_block_n = config.block_n / num_epilogue_warpgroups;
    const int l1_output_store_block_n = config.block_n / 2;
    const auto tensor_map_l1_output = make_tma_2d_desc(
        l2_acts, intermediate_hidden, config.num_max_pool_tokens,
        l1_output_store_block_n, wg_block_m,
        static_cast<int>(l2_acts.stride(-2)), 0);
    const auto tensor_map_l2_acts = make_tma_2d_desc(
        l2_acts, intermediate_hidden, config.num_max_pool_tokens,
        config.block_k, config.block_m,
        static_cast<int>(l2_acts.stride(-2)), config.swizzle_acts_mode);
    const auto tensor_map_l2_acts_sf = make_tma_sf_desc(
        cute::UMMA::Major::MN, l2_acts_sf,
        config.num_padded_sf_pool_tokens, intermediate_hidden,
        config.block_m, kL2ActsSFGranK, 1, 0);
    const auto tensor_map_l2_weights = make_tma_2d_desc(
        l2_weights, static_cast<int>(l2_weights.size(2)),
        num_experts_per_rank * hidden,
        kSM90NVFP4BStoragePerKBlock, config.block_n,
        static_cast<int>(l2_weights.stride(-2)), 0);
    (void)wg_block_n;

    int* cumulative_stats_ptr = cumulative_local_expert_recv_stats.has_value() ?
        cumulative_local_expert_recv_stats->data_ptr<int>() : nullptr;
    const float* l1_global_scales_ptr = l1_global_scales.has_value() ?
        l1_global_scales->data_ptr<float>() : nullptr;
    const float* l2_global_scales_ptr = l2_global_scales.has_value() ?
        l2_global_scales->data_ptr<float>() : nullptr;
    const int num_sms = device_runtime->get_num_sms();

    const SM90NVFP4NibbleGroupMegaMoERuntime::Args args = {
        .num_max_tokens_per_rank = num_max_tokens_per_rank,
        .hidden = hidden,
        .intermediate_hidden = intermediate_hidden,
        .num_experts = num_experts,
        .num_topk = num_topk,
        .num_ranks = num_ranks,
        .activation_clamp = activation_clamp,
        .fast_math = fast_math,
        .l2_dual_accum = false,
        .phase_profile = get_env<int>("DG_SM90_MOE_PHASE_PROFILE", 0) != 0,
        .l2_arrival_counter = false,
        .loader_dequant = plan.loader_dequant,
        .swap_ab = candidate_swap_ab,
        // Both selector-pack variants are unused by the grouped-nibble decode
        // helpers; force them off so they don't multiply JIT variants.
        .dp4a_selector_pack = false,
        .hybrid_low_selector_pack = false,
        .single_active_dispatch_warp = single_active_dispatch_warp,
        .phase_mode = SM90NVFP4MegaMoERuntime::kFusedPhaseMode,
        .config = config,
        .y = y.data_ptr(),
        .cumulative_local_expert_recv_stats = cumulative_stats_ptr,
        .num_tokens = num_tokens,
        .sym_buffer_ptrs = layout::SymBuffer<>(sym_buffer_ptrs, rank_idx),
        .tensor_map_l1_acts = tensor_map_l1_acts,
        .tensor_map_l1_acts_sf = tensor_map_l1_acts_sf,
        .tensor_map_l1_weights = tensor_map_l1_weights,
        .tensor_map_l1_output = tensor_map_l1_output,
        .tensor_map_l2_acts = tensor_map_l2_acts,
        .tensor_map_l2_acts_sf = tensor_map_l2_acts_sf,
        .tensor_map_l2_weights = tensor_map_l2_weights,
        .l1_global_scales = l1_global_scales_ptr,
        .l2_global_scales = l2_global_scales_ptr,
        .launch_args = LaunchArgs(
            num_sms,
            config.num_dispatch_threads + config.num_non_epilogue_threads +
                config.num_epilogue_threads,
            config.smem_size, config.cluster_size)
    };

    const auto code = SM90NVFP4NibbleGroupMegaMoERuntime::generate(args);
    const auto runtime = compiler->build("sm90_nvfp4_mega_moe_nibble_group", code);
    SM90NVFP4NibbleGroupMegaMoERuntime::launch(runtime, args);
}

}  // namespace deep_gemm
