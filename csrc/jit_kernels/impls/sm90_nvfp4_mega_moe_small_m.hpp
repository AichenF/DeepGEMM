#pragma once

#include <torch/python.h>

#include "../../jit/compiler.hpp"
#include "../../jit/kernel_runtime.hpp"
#include "../../utils/exception.hpp"
#include "../../utils/format.hpp"
#include "runtime_utils.hpp"

#include <deep_gemm/layout/mega_moe.cuh>
#include <deep_gemm/layout/sym_buffer.cuh>

#include "../heuristics/sm90_nvfp4_mega_moe_allm.hpp"
#include "../heuristics/sm90_nvfp4_mega_moe_small_m.hpp"

namespace deep_gemm {

class SM90NVFP4SmallMFusedRuntime final
    : public LaunchRuntime<SM90NVFP4SmallMFusedRuntime> {
public:
    struct Args {
        int num_max_tokens_per_rank;
        int hidden;
        int intermediate_hidden;
        int num_experts;
        int num_topk;
        int num_ranks;
        int num_sms;
        float activation_clamp;
        bool fast_math;
        bool swap_ab;
        bool rs_swap_ab;
        bool rs_batch4;
        bool rs_vector_scale;
        bool rs_direct_lut;
        bool rs_compact_smem;
        bool use_mode2_lop3_decoder;
        bool single_active_dispatch_warp;
        SM90NVFP4SmallMConfig config;

        void* y;
        int* cumulative_local_expert_recv_stats;
        int num_tokens;
        layout::SymBuffer<> sym_buffer_ptrs;
        CUtensorMap tensor_map_l1_acts;
        CUtensorMap tensor_map_l1_acts_sf;
        CUtensorMap tensor_map_l1_weights;
        CUtensorMap tensor_map_l1_output;
        CUtensorMap tensor_map_l2_acts;
        CUtensorMap tensor_map_l2_acts_sf;
        CUtensorMap tensor_map_l2_weights;
        const float* l1_global_scales;
        const float* l2_global_scales;
        LaunchArgs launch_args;
    };

    static std::string generate_impl(const Args& args) {
        const std::string kernel_header =
            "#define DG_NVLINK_BARRIER_TRAP_ONLY_TIMEOUT 1\n"
            "#include <deep_gemm/impls/"
            "sm90_nvfp4_mega_moe_small_m.cuh>";
        return fmt::format(R"(
{}

using namespace deep_gemm;

static void __instantiate_kernel() {{
    auto ptr = reinterpret_cast<void*>(&sm90_nvfp4_mega_moe_small_m_fused_impl<
        /* kNumMaxTokensPerRank */ {},
        /* kHidden */ {},
        /* kIntermediateHidden */ {},
        /* kNumExperts */ {},
        /* kNumTopk */ {},
        /* kNumExpertsPerWave */ {},
        /* BLOCK_M */ {},
        /* kNumMaxPoolTokens */ {},
        /* kNumPaddedSFPoolTokens */ {},
        /* kNumStages */ {},
        /* kNumSMs */ {},
        /* kNumRanks */ {},
        /* kActivationClamp */ {},
        /* kFastMath */ {},
        /* kSwapABRequested */ {},
        /* kRSSwapABRequested */ {},
        /* kRSBatch4Requested */ {},
        /* kRSVectorScaleRequested */ {},
        /* kRSDirectLutRequested */ {},
        /* kRSCompactSmemRequested */ {},
        /* kSingleActiveDispatchWarp */ {},
        /* kUseMode2Lop3Decoder */ {}
    >);
}};
)",
            kernel_header,
            args.num_max_tokens_per_rank,
            args.hidden,
            args.intermediate_hidden,
            args.num_experts,
            args.num_topk,
            args.config.num_experts_per_wave,
            args.config.block_m,
            args.config.num_max_pool_tokens,
            args.config.num_padded_sf_pool_tokens,
            args.config.num_stages,
            args.num_sms,
            args.num_ranks,
            to_string(args.activation_clamp),
            args.fast_math ? "true" : "false",
            args.swap_ab ? "true" : "false",
            args.rs_swap_ab ? "true" : "false",
            args.rs_batch4 ? "true" : "false",
            args.rs_vector_scale ? "true" : "false",
            args.rs_direct_lut ? "true" : "false",
            args.rs_compact_smem ? "true" : "false",
            args.single_active_dispatch_warp ? "true" : "false",
            args.use_mode2_lop3_decoder ? "true" : "false");
    }

    static void launch_impl(
            const KernelHandle& kernel,
            const LaunchConfigHandle& config,
            Args args) {
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

static void sm90_nvfp4_small_m_fused_mega_moe(
    const torch::Tensor& y,
    const torch::Tensor& l1_acts,
    const torch::Tensor& l1_acts_sf,
    const torch::Tensor& l2_acts,
    const torch::Tensor& l2_acts_sf,
    const torch::Tensor& l1_weights,
    const torch::Tensor& l2_weights,
    const torch::Tensor& l1_weights_sf,
    const torch::Tensor& l2_weights_sf,
    const std::optional<torch::Tensor> cumulative_local_expert_recv_stats,
    const std::optional<torch::Tensor> l1_global_scales,
    const std::optional<torch::Tensor> l2_global_scales,
    const std::vector<int64_t>& sym_buffer_ptrs,
    const int& rank_idx,
    const int& num_max_tokens_per_rank,
    const int& num_experts_per_rank,
    const int& num_tokens,
    const int& num_topk,
    const int& hidden,
    const int& intermediate_hidden,
    const float& activation_clamp,
    const bool& fast_math,
    const bool& use_kf424_schedule = false
) {
    const int num_ranks = static_cast<int>(sym_buffer_ptrs.size());
    const int num_experts = num_experts_per_rank * num_ranks;
    const int num_sms = device_runtime->get_num_sms();
    DG_HOST_ASSERT(num_sms > 1);
    const int num_padded_sf_pool_tokens =
        static_cast<int>(l1_acts_sf.size(0));
    const SM90NVFP4SmallMInput heuristic_input {
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
    if (use_kf424_schedule)
        DG_HOST_ASSERT(num_sms == 132);
    const auto plan = use_kf424_schedule ?
        select_sm90_nvfp4_small_m_kf424(heuristic_input) :
        select_sm90_nvfp4_small_m(heuristic_input);
    const auto& config = plan.config;
    using KernelConfig = SM90NVFP4SmallMConfig;

    const int l1_weight_k_blocks =
        static_cast<int>(l1_weights_sf.size(2));
    const int l2_weight_k_blocks =
        static_cast<int>(l2_weights_sf.size(2));
    DG_HOST_ASSERT(
        l1_weights.size(2) ==
        l1_weight_k_blocks * KernelConfig::kWeightStoragePerKBlock);
    DG_HOST_ASSERT(
        l2_weights.size(2) ==
        l2_weight_k_blocks * KernelConfig::kWeightStoragePerKBlock);

    constexpr int kActsSFGranK = 128;
    const auto tensor_map_l1_acts = make_tma_2d_desc(
        l1_acts,
        hidden,
        config.num_max_pool_tokens,
        KernelConfig::kBlockK,
        config.block_m,
        static_cast<int>(l1_acts.stride(-2)),
        KernelConfig::kSwizzleActsMode);
    const auto tensor_map_l1_acts_sf = make_tma_sf_desc(
        cute::UMMA::Major::MN,
        l1_acts_sf,
        config.num_padded_sf_pool_tokens,
        hidden,
        config.block_m,
        kActsSFGranK,
        1,
        0);
    const auto tensor_map_l1_weights = make_tma_2d_desc(
        l1_weights,
        static_cast<int>(l1_weights.size(2)),
        num_experts_per_rank * intermediate_hidden * 2,
        KernelConfig::kWeightStoragePerKBlock,
        KernelConfig::kBlockN,
        static_cast<int>(l1_weights.stride(-2)),
        0);

    constexpr int kL1OutputStoreBlockN = KernelConfig::kBlockN / 2;
    const auto tensor_map_l1_output = make_tma_2d_desc(
        l2_acts,
        intermediate_hidden,
        config.num_max_pool_tokens,
        kL1OutputStoreBlockN,
        config.block_m,
        static_cast<int>(l2_acts.stride(-2)),
        0);
    const auto tensor_map_l2_acts = make_tma_2d_desc(
        l2_acts,
        intermediate_hidden,
        config.num_max_pool_tokens,
        KernelConfig::kBlockK,
        config.block_m,
        static_cast<int>(l2_acts.stride(-2)),
        KernelConfig::kSwizzleActsMode);
    const auto tensor_map_l2_acts_sf = make_tma_sf_desc(
        cute::UMMA::Major::MN,
        l2_acts_sf,
        config.num_padded_sf_pool_tokens,
        intermediate_hidden,
        config.block_m,
        kActsSFGranK,
        1,
        0);
    const auto tensor_map_l2_weights = make_tma_2d_desc(
        l2_weights,
        static_cast<int>(l2_weights.size(2)),
        num_experts_per_rank * hidden,
        KernelConfig::kWeightStoragePerKBlock,
        KernelConfig::kBlockN,
        static_cast<int>(l2_weights.stride(-2)),
        0);

    int* cumulative_stats_ptr =
        cumulative_local_expert_recv_stats.has_value() ?
        cumulative_local_expert_recv_stats->data_ptr<int>() : nullptr;
    const float* l1_global_scales_ptr =
        l1_global_scales.has_value() ?
        l1_global_scales->data_ptr<float>() : nullptr;
    const float* l2_global_scales_ptr =
        l2_global_scales.has_value() ?
        l2_global_scales->data_ptr<float>() : nullptr;

    const SM90NVFP4SmallMFusedRuntime::Args args = {
        .num_max_tokens_per_rank = num_max_tokens_per_rank,
        .hidden = hidden,
        .intermediate_hidden = intermediate_hidden,
        .num_experts = num_experts,
        .num_topk = num_topk,
        .num_ranks = num_ranks,
        .num_sms = num_sms,
        .activation_clamp = activation_clamp,
        .fast_math = fast_math,
        .swap_ab = plan.swap_ab,
        .rs_swap_ab = plan.rs_swap_ab,
        .rs_batch4 = plan.rs_batch4,
        .rs_vector_scale = plan.rs_vector_scale,
        .rs_direct_lut = plan.rs_direct_lut,
        .rs_compact_smem = plan.rs_compact_smem,
        .use_mode2_lop3_decoder = plan.use_mode2_lop3_decoder,
        .single_active_dispatch_warp =
            plan.single_active_dispatch_warp,
        .config = config,
        .y = y.data_ptr(),
        .cumulative_local_expert_recv_stats = cumulative_stats_ptr,
        .num_tokens = num_tokens,
        .sym_buffer_ptrs =
            layout::SymBuffer<>(sym_buffer_ptrs, rank_idx),
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
            KernelConfig::kNumThreads,
            config.smem_size,
            KernelConfig::kClusterSize,
            false),
    };

    const auto code = SM90NVFP4SmallMFusedRuntime::generate(args);
    const auto runtime = compiler->build(
        plan.rs_compact_smem ?
            "sm90_nvfp4_small_m_kf424_rs_mode5_compact" :
        plan.rs_swap_ab ?
            "sm90_nvfp4_small_m_kf424_rs_mode5" :
        plan.use_mode2_lop3_decoder ?
            "sm90_nvfp4_small_m_mode2_lop3" :
            "sm90_nvfp4_small_m_mode2_braided_lut",
        code,
        use_kf424_schedule ?
            get_sm90_nvfp4_allm_small_jit_flags(fast_math) : std::string());
    SM90NVFP4SmallMFusedRuntime::launch(runtime, args);
}

}  // namespace deep_gemm
