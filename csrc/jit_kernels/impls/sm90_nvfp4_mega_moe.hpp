#pragma once

#include <torch/python.h>
#include "../../jit/compiler.hpp"
#include "../../jit/kernel_runtime.hpp"
#include "../../utils/exception.hpp"
#include "../../utils/format.hpp"
#include "runtime_utils.hpp"

#include <deep_gemm/layout/mega_moe.cuh>
#include <deep_gemm/layout/sym_buffer.cuh>

#include "../heuristics/sm90_nvfp4_mega_moe.hpp"

namespace deep_gemm {

// ============================================================================
// SM90 (Hopper) NVFP4 MegaMoE host runtime
// ----------------------------------------------------------------------------
// SM90 runtime for the NVFP4 MegaMoE bridge path. Packed E2M1 weights are
// TMA-loaded into shared memory, expanded to FP8 with per-16 UE4M3 scales by
// the math warpgroup, then consumed by Hopper WGMMA.
//
// Differences from SM100 native FP4 path:
//   * Hopper has no native block-scaled FP4 MMA, so B is manually dequantized
//     to FP8 in shared memory before WGMMA.
//   * Activation SF is logically per-128 for both input and L1-to-L2
//     intermediates. The intermediate SF allocation deliberately retains its
//     former per-64 physical capacity until the separate compaction change.
//   * Weight SF is raw UE4M3 uint8 prepacked as (E, N/128, K/128, 128, 8).
//   * Cluster size is at most 2 (TMA multicast on A); no 2-CTA UMMA.
// ============================================================================

class SM90NVFP4SplitMegaMoERuntime final
    : public LaunchRuntime<SM90NVFP4SplitMegaMoERuntime> {
public:
    static constexpr int kL1PhaseMode = 0;
    static constexpr int kL2PhaseMode = 1;

    struct Args {
        // Templated arguments
        int num_max_tokens_per_rank;
        int hidden, intermediate_hidden;
        int num_experts, num_topk;
        int num_ranks;
        float activation_clamp;
        bool fast_math;
        bool phase_profile;
        bool l2_arrival_counter;
        bool dispatch_dequant;
        int phase_mode;
        SM90NVFP4MegaMoEConfig config;

        // Runtime arguments
        void* y;
        int* cumulative_local_expert_recv_stats;
        int num_tokens;
        layout::SymBuffer<> sym_buffer_ptrs;

        // Tensormaps for activations and fused packed weights.
        CUtensorMap tensor_map_l1_acts;
        CUtensorMap tensor_map_l1_acts_sf;
        CUtensorMap tensor_map_l1_weights;
        CUtensorMap tensor_map_l1_output;
        CUtensorMap tensor_map_l2_acts;
        CUtensorMap tensor_map_l2_acts_sf;
        CUtensorMap tensor_map_l2_weights;
        const float* l1_global_scales;
        const float* l2_global_scales;

        // Launch configs
        LaunchArgs launch_args;
    };

    static std::string generate_impl(const Args& args) {
        DG_HOST_ASSERT(
            args.phase_mode == kL1PhaseMode ||
            args.phase_mode == kL2PhaseMode);
        const char* kernel_symbol = args.phase_mode == kL1PhaseMode ?
            "sm90_nvfp4_mega_moe_split_l1_impl" :
            "sm90_nvfp4_mega_moe_split_l2_impl";
        const std::string phase_template_args =
            args.phase_mode == kL1PhaseMode ?
            fmt::format("/* kPhaseProfileRequested */ {},\n"
                        "        /* kL2ArrivalCounterRequested */ {},\n"
                        "        /* kDispatchDequantRequested */ {}",
                        args.phase_profile ? "true" : "false",
                        args.l2_arrival_counter ? "true" : "false",
                        args.dispatch_dequant ? "true" : "false") :
            fmt::format("/* kPhaseProfileRequested */ {}",
                        args.phase_profile ? "true" : "false");
        return fmt::format(R"(
#include <deep_gemm/impls/sm90_nvfp4_mega_moe.cuh>

using namespace deep_gemm;

static void __instantiate_kernel() {{
    auto ptr = reinterpret_cast<void*>(&{}<
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
    kernel_symbol,
    args.num_max_tokens_per_rank,
    args.hidden, args.intermediate_hidden,
    args.num_experts, args.num_topk,
    args.config.num_experts_per_wave,
    args.config.block_m, args.config.block_n, args.config.block_k,
    args.config.num_max_pool_tokens,
    args.config.num_padded_sf_pool_tokens,
    args.config.num_stages,
    args.config.num_dispatch_threads, args.config.num_non_epilogue_threads, args.config.num_epilogue_threads,
    args.config.cluster_size,
    args.launch_args.grid_dim.first, args.num_ranks,
    to_string(args.activation_clamp),
    args.fast_math ? "true" : "false",
    phase_template_args);
    }

    static void launch_impl(const KernelHandle& kernel, const LaunchConfigHandle& config, Args args) {
        if (args.phase_mode == kL1PhaseMode) {
            DG_CUDA_UNIFIED_CHECK(launch_kernel(kernel, config,
                args.cumulative_local_expert_recv_stats,
                args.num_tokens,
                args.sym_buffer_ptrs,
                args.tensor_map_l1_acts,
                args.tensor_map_l1_acts_sf,
                args.tensor_map_l1_weights,
                args.tensor_map_l1_output,
                args.l1_global_scales
            ));
            return;
        }
        DG_HOST_ASSERT(args.phase_mode == kL2PhaseMode);
        DG_CUDA_UNIFIED_CHECK(launch_kernel(kernel, config,
            args.y,
            args.cumulative_local_expert_recv_stats,
            args.num_tokens,
            args.sym_buffer_ptrs,
            args.tensor_map_l2_acts,
            args.tensor_map_l2_acts_sf,
            args.tensor_map_l2_weights,
            args.l2_global_scales
        ));
    }
};

static constexpr int kSM90NVFP4BStoragePerKBlock = 80;

static void sm90_nvfp4_split_mega_moe(
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
    const auto num_ranks = static_cast<int>(sym_buffer_ptrs.size());
    const auto num_experts = num_experts_per_rank * num_ranks;
    const auto num_padded_sf_pool_tokens = static_cast<int>(l1_acts_sf.size(0));

    // The public SM90 entry routes BN256 to the dedicated per-128 small-M
    // runtime. This general runtime owns only the BN128 deployment layout.
    const int block_n_from_layout = static_cast<int>(l1_weights_sf.size(3));
    DG_HOST_ASSERT(block_n_from_layout == 128);
    const auto num_sms = device_runtime->get_num_sms();
    const SM90NVFP4MegaMoEInput heuristic_input {
        num_sms,
        num_ranks,
        num_experts,
        num_experts_per_rank,
        num_max_tokens_per_rank,
        num_tokens,
        num_topk,
        hidden,
        intermediate_hidden,
        num_padded_sf_pool_tokens,
    };
    const auto plan =
        select_sm90_nvfp4_split_mega_moe(heuristic_input);
    const auto& config = plan.l1_config;
    const auto& l2_config = plan.l2_config;
    const int weight_storage_k = static_cast<int>(l1_weights.size(2));
    const int weight_k_blocks = static_cast<int>(l1_weights_sf.size(2));
    const bool layout_fused_b_scale =
        weight_storage_k == weight_k_blocks * kSM90NVFP4BStoragePerKBlock;
    DG_HOST_ASSERT(layout_fused_b_scale);

    // Tensormap construction
    // Acts/weights: standard 2D TMA descriptors (FP8 K-major).
    // Activation SF: per-128 channel float for both L1 and L2
    // (MN-major, no swizzle). The physical L2-SF tensor retains its old
    // per-64 capacity in this stage; only the dense first half is addressed.
    // Weight SF: per-16 K raw UE4M3 pointer (no TMA descriptor).
    constexpr int kGranK = 128;
    constexpr int kL2ActsSFGranK = 128;
    const auto tensor_map_l1_acts = make_tma_2d_desc(l1_acts,
                                                     hidden, config.num_max_pool_tokens,
                                                     config.block_k, config.block_m,
                                                     static_cast<int>(l1_acts.stride(-2)),
                                                     config.swizzle_acts_mode);
    const auto tensor_map_l1_acts_sf = make_tma_sf_desc(cute::UMMA::Major::MN, l1_acts_sf,
                                                        config.num_padded_sf_pool_tokens, hidden,
                                                        config.block_m, kGranK,
                                                        1, 0);
    // NVFP4: packed FP4 weight, optionally fused with row-local UE4M3 scale
    // bytes as 80B per BK128 row. No swizzle since dequant restages into the
    // existing FP8 smem buffer which keeps its 128B swizzle for WGMMA.
    const auto tensor_map_l1_weights = make_tma_2d_desc(l1_weights,
                                                        static_cast<int>(l1_weights.size(2)),
                                                        num_experts_per_rank * intermediate_hidden * 2,
                                                        kSM90NVFP4BStoragePerKBlock, config.block_n,
                                                        static_cast<int>(l1_weights.stride(-2)),
                                                        0);
    // UE4M3 SF: accessed via raw uint8 pointer rather than TMA, since the
    // per-K-block box (block_k/16 = 8 bytes) is below TMA's 16-byte minimum.
    // L1 output (post-SwiGLU FP8): N is halved. The SM90 epilogue writes this
    // staging tile to SMEM as plain row-major bytes, so the TMA store descriptor
    // must use no shared-memory swizzle. Later L2 TMA loads may still swizzle
    // from this row-major global buffer into their own SMEM tile.
    // The default TMA store is issued per warpgroup, each writing a WG_BLOCK_M
    // row tile. The default BM128/BN128 path uses two M warpgroups; adjacent
    // CTAs own the two 64-column halves of a per-128 group.
    const int num_epilogue_warpgroups_h = config.num_epilogue_threads / 128;
    const bool split_n_warpgroups_h =
        config.block_m == 64 and config.block_n == 256 and num_epilogue_warpgroups_h == 2;
    const bool split_mn_warpgroups_h =
        config.block_m == 128 and config.block_n == 256 and num_epilogue_warpgroups_h == 4;
    const int wg_split_m_h = split_n_warpgroups_h ? 1 :
        (split_mn_warpgroups_h ? 2 : num_epilogue_warpgroups_h);
    const int wg_split_n_h = split_n_warpgroups_h ? num_epilogue_warpgroups_h :
        (split_mn_warpgroups_h ? 2 : 1);
    DG_HOST_ASSERT(wg_split_m_h * wg_split_n_h == num_epilogue_warpgroups_h);
    const int wg_block_m = config.block_m / wg_split_m_h;
    const int wg_block_n = config.block_n / wg_split_n_h;
    const int wg_l1_out_block_n = wg_block_n / 2;
    const int l1_output_store_block_n = split_n_warpgroups_h ? config.block_n / 2 : wg_l1_out_block_n;
    const auto tensor_map_l1_output = make_tma_2d_desc(l2_acts,
                                                       intermediate_hidden, config.num_max_pool_tokens,
                                                       l1_output_store_block_n, wg_block_m,
                                                       static_cast<int>(l2_acts.stride(-2)),
                                                       0);
    const auto tensor_map_l2_acts = make_tma_2d_desc(l2_acts,
                                                     intermediate_hidden, l2_config.num_max_pool_tokens,
                                                     l2_config.block_k, l2_config.block_m,
                                                     static_cast<int>(l2_acts.stride(-2)),
                                                     l2_config.swizzle_acts_mode);
    const auto tensor_map_l2_acts_sf = make_tma_sf_desc(cute::UMMA::Major::MN, l2_acts_sf,
                                                        l2_config.num_padded_sf_pool_tokens, intermediate_hidden,
                                                        l2_config.block_m, kL2ActsSFGranK,
                                                        1, 0);
    // NVFP4: packed FP4 weight, optionally fused with row-local UE4M3 scale
    // bytes as 80B per BK128 row. No swizzle (see L1).
    const auto tensor_map_l2_weights = make_tma_2d_desc(l2_weights,
                                                        static_cast<int>(l2_weights.size(2)),
                                                        num_experts_per_rank * hidden,
                                                        kSM90NVFP4BStoragePerKBlock, l2_config.block_n,
                                                        static_cast<int>(l2_weights.stride(-2)),
                                                        0);

    // Stats can be optional
    int* cumulative_local_expert_recv_stats_ptr = nullptr;
    if (cumulative_local_expert_recv_stats.has_value())
        cumulative_local_expert_recv_stats_ptr = cumulative_local_expert_recv_stats->data_ptr<int>();
    const float* l1_global_scales_ptr = l1_global_scales.has_value() ? l1_global_scales->data_ptr<float>() : nullptr;
    const float* l2_global_scales_ptr = l2_global_scales.has_value() ? l2_global_scales->data_ptr<float>() : nullptr;

    // Launch
    DG_HOST_ASSERT(config.block_m == 128 &&
                   config.block_n == 128 && config.cluster_size == 2 &&
                   l2_config.block_m == 128 && l2_config.block_n == 128);

    const SM90NVFP4SplitMegaMoERuntime::Args args = {
        .num_max_tokens_per_rank = num_max_tokens_per_rank,
        .hidden = hidden, .intermediate_hidden = intermediate_hidden,
        .num_experts = num_experts, .num_topk = num_topk,
        .num_ranks = num_ranks,
        .activation_clamp = activation_clamp,
        .fast_math = fast_math,
        .phase_profile = false,
        .l2_arrival_counter = plan.l2_arrival_counter,
        .dispatch_dequant = plan.dispatch_dequant,
        .phase_mode = SM90NVFP4SplitMegaMoERuntime::kL1PhaseMode,
        .config = config,
        .y = y.data_ptr(),
        .cumulative_local_expert_recv_stats = cumulative_local_expert_recv_stats_ptr,
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
        .launch_args = LaunchArgs(num_sms, config.num_dispatch_threads + config.num_non_epilogue_threads + config.num_epilogue_threads,
                                  config.smem_size, config.cluster_size)
    };

    const auto refresh_launch_args = [&](SM90NVFP4SplitMegaMoERuntime::Args& phase_args) {
        phase_args.launch_args = LaunchArgs(
            num_sms,
            phase_args.config.num_dispatch_threads + phase_args.config.num_non_epilogue_threads +
                phase_args.config.num_epilogue_threads,
            phase_args.config.smem_size, phase_args.config.cluster_size);
    };

    const auto build_and_launch = [&](const SM90NVFP4SplitMegaMoERuntime::Args& phase_args,
                                      const std::string& kernel_name) {
        const auto code = SM90NVFP4SplitMegaMoERuntime::generate(phase_args);
        const auto runtime = compiler->build(kernel_name, code);
        SM90NVFP4SplitMegaMoERuntime::launch(runtime, phase_args);
    };

    const auto launch_split_l1 = [&]() {
        auto phase_args = args;
        phase_args.phase_mode = SM90NVFP4SplitMegaMoERuntime::kL1PhaseMode;
        DG_HOST_ASSERT(phase_args.config.block_m == 128 &&
                       phase_args.config.block_n == 128 &&
                       phase_args.config.cluster_size == 2);
        build_and_launch(phase_args, "sm90_nvfp4_mega_moe_l1");
    };

    const auto launch_split_l2 = [&]() {
        auto phase_args = args;
        phase_args.phase_mode = SM90NVFP4SplitMegaMoERuntime::kL2PhaseMode;
        phase_args.config = plan.l2_config;
        DG_HOST_ASSERT(phase_args.config.block_n == 128);
        refresh_launch_args(phase_args);
        build_and_launch(phase_args, "sm90_nvfp4_mega_moe_l2");
    };

    launch_split_l1();
    launch_split_l2();
}

} // namespace deep_gemm
