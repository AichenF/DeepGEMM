#pragma once

#include "sm90_nvfp4_mega_moe.hpp"

namespace deep_gemm {

class SM90NVFP4Per128ProBraided3StageMegaMoERuntime final
    : public LaunchRuntime<SM90NVFP4Per128ProBraided3StageMegaMoERuntime> {
public:
    using Args = SM90NVFP4MegaMoERuntime::Args;

    static std::string generate_impl(const Args& args) {
        DG_HOST_ASSERT(args.phase_mode == SM90NVFP4MegaMoERuntime::kFusedPhaseMode);
        // Exact MiMo Pro M32/BM16 has room for both full dispatch buffers and
        // benefits from parallel token pulls.  Keep one active dispatch warp
        // for every other deployment shape, including generic Pro M32.
        const bool mimo_pro_m32_dual_dispatch =
            args.num_tokens == 32 && args.hidden == 6144 &&
            args.intermediate_hidden == 2048 && args.num_topk == 8 &&
            args.num_ranks > 0 && args.num_experts == 48 * args.num_ranks &&
            args.config.block_m == 16 && args.config.num_stages == 3;
        const bool single_active_dispatch_warp = !mimo_pro_m32_dual_dispatch;
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
            single_active_dispatch_warp ? "true" : "false");
        return fmt::format(R"(
#include <deep_gemm/impls/sm90_nvfp4_mega_moe_per128_pro_braided.cuh>

using namespace deep_gemm;

static void __instantiate_kernel() {{
    auto ptr = reinterpret_cast<void*>(&sm90_nvfp4_mega_moe_per128_pro_braided_fused_impl<
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

static void sm90_nvfp4_per128_pro_braided_3stage_mega_moe(
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
    const bool mimo_pro_small_m_candidate =
        (num_tokens == 8 || num_tokens == 16 ||
         num_tokens == 32 || num_tokens == 64) &&
        num_topk == 8 && num_experts_per_rank == 48 &&
        hidden == 6144 && intermediate_hidden == 2048;
    DG_HOST_ASSERT(intermediate_hidden >= 3072 || mimo_pro_small_m_candidate);
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
    const bool mimo_pro_m8_epw16_candidate =
        num_tokens == 8 && num_topk == 8 && num_experts_per_rank == 48 &&
        hidden == 6144 && intermediate_hidden == 2048;
    if (mimo_pro_m8_epw16_candidate)
        config.num_experts_per_wave = 16;
    const bool mimo_pro_m32_dual_dispatch_candidate =
        num_tokens == 32 && num_topk == 8 && num_experts_per_rank == 48 &&
        hidden == 6144 && intermediate_hidden == 2048;
    DG_HOST_ASSERT(num_experts_per_rank % config.num_experts_per_wave == 0);
    DG_HOST_ASSERT(config.block_m == 64 && config.block_n == 256);
    DG_HOST_ASSERT(plan.swap_ab &&
                   (plan.dp4a_selector_pack || mimo_pro_small_m_candidate));
    DG_HOST_ASSERT(!plan.loader_dequant && config.num_stages == 2);

    const auto align_up = [](const int value, const int alignment) {
        return ((value + alignment - 1) / alignment) * alignment;
    };
    constexpr int kSmemAlignment = 1024;
    const int num_dispatch_warps = config.num_dispatch_threads / 32;
    DG_HOST_ASSERT(num_dispatch_warps == 2);
    const int full_send_buffers_size = align_up(
        static_cast<int>(layout::Buffer(
            layout::Data(hidden), num_dispatch_warps, 1).get_num_bytes()),
        kSmemAlignment);
    const int active_send_buffers_size = align_up(
        static_cast<int>(layout::Buffer(
            layout::Data(hidden), 1, 1).get_num_bytes()),
        kSmemAlignment);
    const int smem_sfa_per_stage = align_up(
        2 * config.block_m * static_cast<int>(sizeof(float)), 128);
    const int smem_per_stage = config.block_m * config.block_k +
        config.block_n * config.block_k +
        config.block_n * kSM90NVFP4BStoragePerKBlock +
        smem_sfa_per_stage;
    constexpr int kSmemBarriersPerStage = 2 * static_cast<int>(sizeof(uint64_t));
    config.num_stages = 3;
    config.smem_size += active_send_buffers_size - full_send_buffers_size +
        smem_per_stage + kSmemBarriersPerStage;
    constexpr int kPer128SFStageSaving = 64 * static_cast<int>(sizeof(float));
    config.smem_size -= config.num_stages * kPer128SFStageSaving;
    DG_HOST_ASSERT(config.smem_size <= SM90ArchSpec::smem_capacity);
    // Match the scheduler M tile to the observed per-expert route envelope.
    // Extra-heavy experts remain legal because the scheduler emits more tiles.
    // Keep the conservative BM64 launch allocation for this isolated screen.
    if (num_tokens == 8 || num_tokens == 16)
        config.block_m = 8;
    else if (num_tokens == 32)
        config.block_m = 16;
    else if (num_tokens == 64)
        config.block_m = 24;
    if (num_tokens == 8 || num_tokens == 16) {
        // BM8 leaves just enough shared memory for a fourth A/B pipeline
        // stage (232144 bytes including barriers). Reserve the full SM90
        // opt-in capacity so the dynamic allocation covers that layout.
        config.num_stages = 4;
        config.smem_size = SM90ArchSpec::smem_capacity;
    } else if (mimo_pro_m32_dual_dispatch_candidate) {
        // Route-sized BM16 needs 191040 bytes with both dispatch pull warps
        // active and the retained three GEMM stages.  Allocate the full
        // opt-in capacity so the generated two-buffer layout is covered
        // without changing occupancy (still one CTA/SM).
        config.smem_size = SM90ArchSpec::smem_capacity;
    }
    const int weight_storage_k = static_cast<int>(l1_weights.size(2));
    const int weight_k_blocks = static_cast<int>(l1_weights_sf.size(2));
    DG_HOST_ASSERT(weight_storage_k == weight_k_blocks * kSM90NVFP4BStoragePerKBlock);

    constexpr int kGranK = 128;
    constexpr int kL2ActsSFGranK = 128;
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
    const bool split_n_warpgroups =
        (config.block_m == 8 || config.block_m == 16 ||
         config.block_m == 24 || config.block_m == 32 || config.block_m == 64) &&
        config.block_n == 256 && num_epilogue_warpgroups == 2;
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

    const SM90NVFP4Per128ProBraided3StageMegaMoERuntime::Args args = {
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
        .swap_ab = plan.swap_ab,
        .dp4a_selector_pack = plan.dp4a_selector_pack,
        .hybrid_low_selector_pack = plan.hybrid_low_selector_pack,
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

    const auto code = SM90NVFP4Per128ProBraided3StageMegaMoERuntime::generate(args);
    const auto runtime = compiler->build("sm90_nvfp4_mega_moe_per128_pro_braided_3stage", code);
    SM90NVFP4Per128ProBraided3StageMegaMoERuntime::launch(runtime, args);
}

}  // namespace deep_gemm
