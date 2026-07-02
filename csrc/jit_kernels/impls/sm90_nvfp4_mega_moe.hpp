#pragma once

#include <torch/python.h>
#include "../../jit/compiler.hpp"
#include "../../jit/kernel_runtime.hpp"
#include "../../utils/exception.hpp"
#include "../../utils/format.hpp"
#include "runtime_utils.hpp"

#include <deep_gemm/layout/mega_moe.cuh>
#include <deep_gemm/layout/sym_buffer.cuh>

#include "../heuristics/mega_moe.hpp"

namespace deep_gemm {

// ============================================================================
// SM90 (Hopper) FP8 MegaMoE host runtime
// ----------------------------------------------------------------------------
// SM90 runtime for the NVFP4 MegaMoE bridge path. Packed E2M1 weights are
// TMA-loaded into shared memory, expanded to FP8 with per-16 UE4M3 scales by
// the math warpgroup, then consumed by Hopper WGMMA.
//
// Differences from SM100 native FP4 path:
//   * Hopper has no native block-scaled FP4 MMA, so B is manually dequantized
//     to FP8 in shared memory before WGMMA.
//   * Activation SF remains the SM90 FP8 per-128/per-64 float path.
//   * Weight SF is raw UE4M3 uint8 prepacked as (E, N/128, K/128, 128, 8).
//   * Cluster size is at most 2 (TMA multicast on A); no 2-CTA UMMA.
// ============================================================================

class SM90NVFP4MegaMoERuntime final : public LaunchRuntime<SM90NVFP4MegaMoERuntime> {
public:
    static constexpr int kFusedPhaseMode = 0;
    static constexpr int kSplitL1PhaseMode = 1;
    static constexpr int kSplitL2PhaseMode = 2;

    struct Args {
        // Templated arguments
        int num_max_tokens_per_rank;
        int hidden, intermediate_hidden;
        int num_experts, num_topk;
        int num_ranks;
        float activation_clamp;
        bool fast_math;
        bool l2_dual_accum;
        bool phase_profile;
        bool l2_arrival_counter;
        bool loader_dequant;
        bool swap_ab;
        bool dp4a_selector_pack;
        bool hybrid_low_selector_pack;
        int phase_mode;
        MegaMoESM90Config config;

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
        DG_HOST_ASSERT(args.phase_mode >= kFusedPhaseMode && args.phase_mode <= kSplitL2PhaseMode);
        const char* kernel_symbol = args.phase_mode == kFusedPhaseMode ? "sm90_nvfp4_mega_moe_fused_impl" :
            (args.phase_mode == kSplitL1PhaseMode ?
                "sm90_nvfp4_mega_moe_split_l1_impl" : "sm90_nvfp4_mega_moe_split_l2_impl");
        const std::string phase_template_args = args.phase_mode == kFusedPhaseMode ?
            fmt::format("/* kPhaseProfileRequested */ {},\n"
                        "        /* kLoaderDequantRequested */ {},\n"
                        "        /* kSwapABRequested */ {},\n"
                        "        /* kDp4aSelectorPack */ {},\n"
                        "        /* kHybridLowSelectorPack */ {}",
                        args.phase_profile ? "true" : "false",
                        args.loader_dequant ? "true" : "false",
                        args.swap_ab ? "true" : "false",
                        args.dp4a_selector_pack ? "true" : "false",
                        args.hybrid_low_selector_pack ? "true" : "false") :
            (args.phase_mode == kSplitL1PhaseMode ?
                fmt::format("/* kPhaseProfileRequested */ {},\n"
                            "        /* kL2ArrivalCounterRequested */ {},\n"
                            "        /* kLoaderDequantRequested */ {}",
                            args.phase_profile ? "true" : "false",
                            args.l2_arrival_counter ? "true" : "false",
                            args.loader_dequant ? "true" : "false") :
                fmt::format("/* kL2DualAccumRequested */ {},\n"
                            "        /* kPhaseProfileRequested */ {},\n"
                            "        /* kLoaderDequantRequested */ {}",
                            args.l2_dual_accum ? "true" : "false",
                            args.phase_profile ? "true" : "false",
                            args.loader_dequant ? "true" : "false"));
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
        if (args.phase_mode == kFusedPhaseMode) {
            DG_CUDA_UNIFIED_CHECK(launch_kernel(kernel, config,
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
                args.l2_global_scales
            ));
            return;
        }
        if (args.phase_mode == kSplitL1PhaseMode) {
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
        DG_HOST_ASSERT(args.phase_mode == kSplitL2PhaseMode);
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

// The prepacked weight layout selects one deployment-time phase plan. Build
// each phase's final launch config here instead of mutating the generic FP8 one.
struct SM90NVFP4MegaMoEPlan {
    MegaMoESM90Config l1_or_fused_config;
    MegaMoESM90Config split_l2_config;
    bool use_fused_phase;
    bool loader_dequant;
    bool swap_ab;
    bool dp4a_selector_pack;
    bool hybrid_low_selector_pack;
    bool l2_dual_accum;
    bool l2_arrival_counter;
};

static std::pair<int, int> get_nvfp4_pipeline_config_for_mega_moe_sm90(
    const int num_experts, const int hidden,
    const float expected_tokens_per_local_expert,
    const MegaMoESM90Config& config,
    const bool loader_dequant, const bool packed_b_scratch,
    const bool swap_ab, const bool l2_only
) {
    const auto align = [](int x, int a) { return ((x + a - 1) / a) * a; };
    constexpr int kSmemAlignment = 1024;
    const int num_dispatch_warps = config.num_dispatch_threads / 32;
    const int num_epilogue_warps = config.num_epilogue_threads / 32;
    const int num_epilogue_warpgroups = num_epilogue_warps / 4;

    const bool split_n_warpgroups = config.block_m == 64 && config.block_n == 256 &&
        num_epilogue_warpgroups == 2;
    const bool split_mn_warpgroups = config.block_m == 128 && config.block_n == 256 &&
        num_epilogue_warpgroups == 4;
    const int wg_split_m = split_n_warpgroups ? 1 :
        (split_mn_warpgroups ? 2 : num_epilogue_warpgroups);
    const int wg_split_n = split_n_warpgroups ? num_epilogue_warpgroups :
        (split_mn_warpgroups ? 2 : 1);
    DG_HOST_ASSERT(wg_split_m * wg_split_n == num_epilogue_warpgroups);
    const int wg_block_m = config.block_m / wg_split_m;
    const int wg_block_n = config.block_n / wg_split_n;
    const int wg_l1_out_block_n = wg_block_n / 2;

    const int smem_expert_count_size = align(
        num_experts * static_cast<int>(sizeof(uint32_t)), kSmemAlignment);
    const int smem_send_buffers_size = align(
        static_cast<int>(layout::Buffer(
            layout::Data(hidden), num_dispatch_warps, 1).get_num_bytes()),
        kSmemAlignment);
    const int smem_dispatch_size = smem_expert_count_size + smem_send_buffers_size;
    const int smem_nvfp4_lut = align(128 * 8, kSmemAlignment);

    const int smem_cd_l1 = num_epilogue_warpgroups * wg_block_m * wg_l1_out_block_n;
    const bool split_n_shares_l1_sf = split_n_warpgroups && wg_l1_out_block_n < 64;
    const int smem_cd_l1_shared_sf_slots = split_n_shares_l1_sf ?
        num_epilogue_warpgroups * config.block_m : 0;
    const int smem_cd_l1_swap_amax_slots = swap_ab ?
        config.block_m * num_epilogue_warps : 0;
    const int smem_cd_l1_extra_float_slots = std::max(
        smem_cd_l1_shared_sf_slots, smem_cd_l1_swap_amax_slots);
    const int smem_cd_l1_shared_sf =
        smem_cd_l1_extra_float_slots * static_cast<int>(sizeof(float));
    const int smem_cd_l2 = swap_ab ? config.block_m * config.block_n * 2 : 0;
    const int smem_cd = l2_only ? 0 : align(
        std::max(smem_cd_l1, smem_cd_l2) + smem_cd_l1_shared_sf,
        kSmemAlignment);

    const int smem_sfa_per_stage = align(
        2 * config.block_m * static_cast<int>(sizeof(float)), 128);
    const int smem_packed_b_per_stage = packed_b_scratch ?
        config.block_n * kSM90NVFP4BStoragePerKBlock : 0;
    const int smem_per_stage = config.block_m * config.block_k +
        config.block_n * config.block_k + smem_packed_b_per_stage + smem_sfa_per_stage;
    const int smem_barriers_fixed = (num_dispatch_warps + 2 * num_epilogue_warps) * 8;
    const int smem_barriers_per_stage = (loader_dequant ? 3 : 2) * 8;
    const int smem_fixed = smem_dispatch_size + smem_nvfp4_lut +
        smem_cd + smem_barriers_fixed;
    const int max_num_stages = (SM90ArchSpec::smem_capacity - smem_fixed) /
        (smem_per_stage + smem_barriers_per_stage);
    const int num_stages = !l2_only && config.block_n == 128 &&
        expected_tokens_per_local_expert > 8.0f && max_num_stages > 6 ?
        6 : max_num_stages;
    DG_HOST_ASSERT(max_num_stages >= 2);
    DG_HOST_ASSERT(num_stages >= 2 && num_stages <= max_num_stages);
    return {
        num_stages,
        smem_fixed + num_stages * (smem_per_stage + smem_barriers_per_stage)
    };
}

static SM90NVFP4MegaMoEPlan get_nvfp4_mega_moe_plan_sm90(
    const int num_ranks, const int num_experts, const int num_experts_per_rank,
    const int num_max_tokens_per_rank, const int num_tokens, const int num_topk,
    const int hidden, const int intermediate_hidden,
    const int num_padded_sf_pool_tokens, const int block_n_from_layout
) {
    DG_HOST_ASSERT(block_n_from_layout == 128 || block_n_from_layout == 256);
    const int routed_tokens = num_tokens * num_topk;
    const float expected_tokens_per_local_expert =
        static_cast<float>(routed_tokens) / num_experts_per_rank;
    const auto expected_eq = [=](int expected) {
        return routed_tokens == expected * num_experts_per_rank;
    };
    const auto expected_in_closed_range = [=](int lower, int upper) {
        return routed_tokens >= lower * num_experts_per_rank &&
               routed_tokens <= upper * num_experts_per_rank;
    };

    const bool use_fused_phase = block_n_from_layout == 256;
    const int block_m = !use_fused_phase && expected_tokens_per_local_expert >= 64.0f ?
        128 : 64;
    const int block_n = block_n_from_layout;
    const int block_k = 128;
    const int cluster_size = 1;
    const int num_dispatch_threads = use_fused_phase ? 64 : 128;
    const int num_non_epilogue_threads = use_fused_phase ? 64 : 128;
    const int num_epilogue_threads = use_fused_phase || block_m == 128 ? 256 : 128;

    DG_HOST_ASSERT(block_m == 64 || block_m == 128);
    DG_HOST_ASSERT((block_n == 128 &&
                    (num_epilogue_threads == 128 || num_epilogue_threads == 256)) ||
                   (block_n == 256 && num_epilogue_threads == 256));
    DG_HOST_ASSERT((num_dispatch_threads + num_non_epilogue_threads) % 128 == 0);

    int num_experts_per_wave = get_num_experts_per_wave_for_mega_moe_sm90(
        num_experts_per_rank, num_tokens, num_topk,
        intermediate_hidden, block_m, block_n,
        device_runtime->get_num_sms());
    if (block_m == 128 && block_n == 128) {
        if (expected_eq(256))
            num_experts_per_wave = num_experts_per_rank;
        if (expected_eq(512) && num_experts_per_rank % 4 == 0)
            num_experts_per_wave = 4;
    }

    const int m_tiles = (num_tokens + block_m - 1) / block_m;
    const int loader_dequant_default = use_fused_phase ? (m_tiles >= 2 ? 1 : 0) : 1;
    const bool loader_dequant_requested =
        get_env<int>("DG_SM90_NVFP4_LOADER_DEQUANT", loader_dequant_default) != 0;
    const bool bn256_packed_loader_dequant =
        use_fused_phase && num_non_epilogue_threads == 64;
    const bool loader_dequant = loader_dequant_requested &&
        (num_non_epilogue_threads == 128 || bn256_packed_loader_dequant);
    const bool swap_ab = use_fused_phase && block_m == 64 &&
        num_epilogue_threads == 256 && expected_in_closed_range(0, 8);
    const bool fused_policy_eligible = use_fused_phase && block_m == 64;
    const bool dp4a_selector_pack = fused_policy_eligible &&
        intermediate_hidden >= 3072 && expected_in_closed_range(0, 8);
    const bool hybrid_low_selector_pack = fused_policy_eligible &&
        intermediate_hidden <= 2048 && expected_in_closed_range(3, 8);
    const bool packed_b_scratch = use_fused_phase;
    DG_HOST_ASSERT(loader_dequant || packed_b_scratch);

    MegaMoESM90Config l1_or_fused_config = {
        block_m, block_n, block_k,
        cluster_size,
        layout::get_num_max_pool_tokens(
            num_ranks, num_max_tokens_per_rank, num_topk, num_experts_per_rank),
        num_padded_sf_pool_tokens,
        128, 128,
        num_experts_per_wave,
        0, 0,
        num_dispatch_threads, num_non_epilogue_threads, num_epilogue_threads
    };
    std::tie(l1_or_fused_config.num_stages, l1_or_fused_config.smem_size) =
        get_nvfp4_pipeline_config_for_mega_moe_sm90(
            num_experts, hidden, expected_tokens_per_local_expert,
            l1_or_fused_config, loader_dequant, packed_b_scratch, swap_ab, false);

    MegaMoESM90Config split_l2_config = l1_or_fused_config;
    if (!use_fused_phase) {
        const auto align = [](int x, int a) { return ((x + a - 1) / a) * a; };
        constexpr int kSmemAlignment = 1024;
        const int full_send_buffers_size = align(
            static_cast<int>(layout::Buffer(layout::Data(hidden), 4, 1).get_num_bytes()),
            kSmemAlignment);
        const int active_send_buffers_size = align(
            static_cast<int>(layout::Buffer(layout::Data(hidden), 2, 1).get_num_bytes()),
            kSmemAlignment);
        l1_or_fused_config.smem_size -= full_send_buffers_size - active_send_buffers_size;

        split_l2_config.num_dispatch_threads = 0;
        split_l2_config.num_non_epilogue_threads = 128;
        std::tie(split_l2_config.num_stages, split_l2_config.smem_size) =
            get_nvfp4_pipeline_config_for_mega_moe_sm90(
                num_experts, hidden, expected_tokens_per_local_expert,
                split_l2_config, loader_dequant, false, false, true);
    }

    const bool l2_dual_accum = !use_fused_phase &&
        (expected_tokens_per_local_expert <= 64.0f ||
         expected_tokens_per_local_expert >= 128.0f);
    const bool l2_arrival_counter = !use_fused_phase &&
        (expected_tokens_per_local_expert <= 32.0f ||
         expected_tokens_per_local_expert >= 128.0f);

    if (get_env<int>("DG_JIT_DEBUG") || get_env<int>("DG_PRINT_CONFIGS")) {
        const auto key = fmt::format(
            "SM90NVFP4MegaMoEPlan(num_ranks={}, num_experts={}, hidden={}, intermediate_hidden={}, "
            "num_max_tokens_per_rank={}, num_tokens={}, num_topk={}, block_n={})",
            num_ranks, num_experts, hidden, intermediate_hidden,
            num_max_tokens_per_rank, num_tokens, num_topk, block_n);
        static std::unordered_set<std::string> printed;
        if (printed.count(key) == 0) {
            std::cout << key << ": l1_or_fused=" << l1_or_fused_config;
            if (!use_fused_phase)
                std::cout << ", split_l2=" << split_l2_config;
            std::cout << std::endl;
            printed.insert(key);
        }
    }

    return {
        l1_or_fused_config,
        split_l2_config,
        use_fused_phase,
        loader_dequant,
        swap_ab,
        dp4a_selector_pack,
        hybrid_low_selector_pack,
        l2_dual_accum,
        l2_arrival_counter
    };
}

static void sm90_nvfp4_mega_moe(
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

    // The framework chooses the serving phase by passing the corresponding
    // prepacked weight layout: BN256 means fused, BN128 means split L1/L2.
    const int block_n_from_layout = static_cast<int>(l1_weights_sf.size(3));
    const auto plan = get_nvfp4_mega_moe_plan_sm90(
        num_ranks, num_experts, num_experts_per_rank,
        num_max_tokens_per_rank, num_tokens, num_topk,
        hidden, intermediate_hidden, num_padded_sf_pool_tokens,
        block_n_from_layout);
    const auto& config = plan.l1_or_fused_config;
    const bool use_fused_phase = plan.use_fused_phase;
    const bool split_l1_l2 = !use_fused_phase;
    const int weight_storage_k = static_cast<int>(l1_weights.size(2));
    const int weight_k_blocks = static_cast<int>(l1_weights_sf.size(2));
    const bool layout_fused_b_scale =
        weight_storage_k == weight_k_blocks * kSM90NVFP4BStoragePerKBlock;
    DG_HOST_ASSERT(layout_fused_b_scale);

    // Tensormap construction
    // Acts/weights: standard 2D TMA descriptors (FP8 K-major).
    // Activation SF: per-128 channel float for L1, per-64 for L2 (MN-major, no swizzle).
    // Weight SF: per-16 K raw UE4M3 pointer (no TMA descriptor).
    constexpr int kGranK = 128;
    constexpr int kL2ActsSFGranK = 64;
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
    // row tile. The split-N experiment has two WGs produce different N halves
    // of the same M rows, then one TMA store writes the full 64x128 post-SwiGLU tile.
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
                                                     intermediate_hidden, config.num_max_pool_tokens,
                                                     config.block_k, config.block_m,
                                                     static_cast<int>(l2_acts.stride(-2)),
                                                     config.swizzle_acts_mode);
    const auto tensor_map_l2_acts_sf = make_tma_sf_desc(cute::UMMA::Major::MN, l2_acts_sf,
                                                        config.num_padded_sf_pool_tokens, intermediate_hidden,
                                                        config.block_m, kL2ActsSFGranK,
                                                        1, 0);
    // NVFP4: packed FP4 weight, optionally fused with row-local UE4M3 scale
    // bytes as 80B per BK128 row. No swizzle (see L1).
    const auto tensor_map_l2_weights = make_tma_2d_desc(l2_weights,
                                                        static_cast<int>(l2_weights.size(2)),
                                                        num_experts_per_rank * hidden,
                                                        kSM90NVFP4BStoragePerKBlock, config.block_n,
                                                        static_cast<int>(l2_weights.stride(-2)),
                                                        0);

    // Stats can be optional
    int* cumulative_local_expert_recv_stats_ptr = nullptr;
    if (cumulative_local_expert_recv_stats.has_value())
        cumulative_local_expert_recv_stats_ptr = cumulative_local_expert_recv_stats->data_ptr<int>();
    const float* l1_global_scales_ptr = l1_global_scales.has_value() ? l1_global_scales->data_ptr<float>() : nullptr;
    const float* l2_global_scales_ptr = l2_global_scales.has_value() ? l2_global_scales->data_ptr<float>() : nullptr;

    // Launch
    const auto num_sms = device_runtime->get_num_sms();
    DG_HOST_ASSERT(use_fused_phase == (config.block_n == 256));
    DG_HOST_ASSERT(split_l1_l2 == (config.block_n == 128));

    const SM90NVFP4MegaMoERuntime::Args args = {
        .num_max_tokens_per_rank = num_max_tokens_per_rank,
        .hidden = hidden, .intermediate_hidden = intermediate_hidden,
        .num_experts = num_experts, .num_topk = num_topk,
        .num_ranks = num_ranks,
        .activation_clamp = activation_clamp,
        .fast_math = fast_math,
        .l2_dual_accum = plan.l2_dual_accum,
        .phase_profile = get_env<int>("DG_SM90_MOE_PHASE_PROFILE", 0) != 0,
        .l2_arrival_counter = plan.l2_arrival_counter,
        .loader_dequant = plan.loader_dequant,
        .swap_ab = plan.swap_ab,
        .dp4a_selector_pack = plan.dp4a_selector_pack,
        .hybrid_low_selector_pack = plan.hybrid_low_selector_pack,
        .phase_mode = SM90NVFP4MegaMoERuntime::kFusedPhaseMode,
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

    const auto refresh_launch_args = [&](SM90NVFP4MegaMoERuntime::Args& phase_args) {
        phase_args.launch_args = LaunchArgs(
            num_sms,
            phase_args.config.num_dispatch_threads + phase_args.config.num_non_epilogue_threads +
                phase_args.config.num_epilogue_threads,
            phase_args.config.smem_size, phase_args.config.cluster_size);
    };

    const auto build_and_launch = [&](const SM90NVFP4MegaMoERuntime::Args& phase_args,
                                      const std::string& kernel_name) {
        const auto code = SM90NVFP4MegaMoERuntime::generate(phase_args);
        const auto runtime = compiler->build(kernel_name, code);
        SM90NVFP4MegaMoERuntime::launch(runtime, phase_args);
    };

    const auto launch_fused = [&]() {
        auto phase_args = args;
        phase_args.phase_mode = SM90NVFP4MegaMoERuntime::kFusedPhaseMode;
        build_and_launch(phase_args, "sm90_nvfp4_mega_moe");
    };

    const auto launch_split_l1 = [&]() {
        auto phase_args = args;
        phase_args.phase_mode = SM90NVFP4MegaMoERuntime::kSplitL1PhaseMode;
        DG_HOST_ASSERT(!use_fused_phase && phase_args.config.block_n == 128);
        build_and_launch(phase_args, "sm90_nvfp4_mega_moe_l1");
    };

    const auto launch_split_l2 = [&]() {
        auto phase_args = args;
        phase_args.phase_mode = SM90NVFP4MegaMoERuntime::kSplitL2PhaseMode;
        phase_args.config = plan.split_l2_config;
        DG_HOST_ASSERT(!use_fused_phase && phase_args.config.block_n == 128);
        refresh_launch_args(phase_args);
        build_and_launch(phase_args, "sm90_nvfp4_mega_moe_l2");
    };

    if (split_l1_l2) {
        launch_split_l1();
        launch_split_l2();
        return;
    }

    launch_fused();
}

} // namespace deep_gemm
