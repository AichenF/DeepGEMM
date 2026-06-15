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

class SM90NVFP4MegaMoESplitRuntime final : public LaunchRuntime<SM90NVFP4MegaMoESplitRuntime> {
public:
    struct Args {
        // Templated arguments
        int num_max_tokens_per_rank;
        int hidden, intermediate_hidden;
        int num_experts, num_topk;
        int num_ranks;
        float activation_clamp;
        bool fast_math;
        bool async_l1_tma_store;
        bool split_sfa_tma;
        bool direct_l2_scatter;
        bool l2_dual_accum;
        bool phase_profile;
        bool l2_arrival_counter;
        bool direct_scatter_metadata_broadcast;
        bool l1_dual_k_accum;
        bool l2_nmajor_schedule;
        bool l1_nmajor_schedule;
        bool k2_direct_accum;
        bool loader_dequant;
        bool direct_scale_gmem;
        bool packed_b_scratch;
        bool split_dequant_barrier;
        bool strided_b_gmem_load;
        bool fused_b_scale_layout;
        bool combine_7chunk;
        bool skip_direct_scatter_sync;
        bool run_l1_phase;
        bool run_l2_phase;
        MegaMoESM90Config config;

        // Runtime arguments
        void* y;
        int* cumulative_local_expert_recv_stats;
        int num_tokens;
        layout::SymBuffer<> sym_buffer_ptrs;

        // Tensormaps for activations and packed weights. UE4M3 weight scale
        // factors are raw uint8 pointers with tile-major shape
        // (E, N/128, K/128, 128, 8); the math warpgroup loads them directly
        // from global memory.
        CUtensorMap tensor_map_l1_acts;
        CUtensorMap tensor_map_l1_acts_sf;
        CUtensorMap tensor_map_l1_weights;
        const uint8_t* l1_weights_gmem;
        const uint8_t* l1_weights_sf;
        CUtensorMap tensor_map_l1_output;
        CUtensorMap tensor_map_l2_acts;
        CUtensorMap tensor_map_l2_acts_sf;
        CUtensorMap tensor_map_l2_weights;
        const uint8_t* l2_weights_gmem;
        const uint8_t* l2_weights_sf;

        // Launch configs
        LaunchArgs launch_args;
    };

    static std::string generate_impl(const Args& args) {
        return fmt::format(R"(
#include <deep_gemm/impls/sm90_nvfp4_mega_moe.cuh>

using namespace deep_gemm;

static void __instantiate_kernel() {{
    auto ptr = reinterpret_cast<void*>(&sm90_nvfp4_mega_moe_impl<
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
        {}, {}, {}, {}, {}, {}, {}, {}, {},
        {}, {}, {}, {}, {}, {}, {}, {}, {}, {},
        {}, {}, {}, {}, {}, {}, {}, {}, {}, {},
        {}, {}, {}, {}
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
    args.config.num_dispatch_threads, args.config.num_non_epilogue_threads, args.config.num_epilogue_threads,
    args.config.cluster_size,
    args.launch_args.grid_dim.first, args.num_ranks,
    to_string(args.activation_clamp),
    args.fast_math ? "true" : "false",
    args.async_l1_tma_store ? "true" : "false", args.split_sfa_tma ? "true" : "false", args.direct_l2_scatter ? "true" : "false", args.l2_dual_accum ? "true" : "false", args.phase_profile ? "true" : "false", args.l2_arrival_counter ? "true" : "false", args.direct_scatter_metadata_broadcast ? "true" : "false", args.l1_dual_k_accum ? "true" : "false", args.l2_nmajor_schedule ? "true" : "false", args.l1_nmajor_schedule ? "true" : "false",
    args.k2_direct_accum ? "true" : "false", args.loader_dequant ? "true" : "false", args.direct_scale_gmem ? "true" : "false", args.packed_b_scratch ? "true" : "false", args.split_dequant_barrier ? "true" : "false", args.strided_b_gmem_load ? "true" : "false", args.fused_b_scale_layout ? "true" : "false", args.combine_7chunk ? "true" : "false", args.skip_direct_scatter_sync ? "true" : "false",
    args.intermediate_hidden * 2, args.hidden, args.hidden, args.intermediate_hidden,
    args.config.num_dispatch_threads / 32, args.config.num_non_epilogue_threads / 32, args.config.num_epilogue_threads / 32, (args.config.num_epilogue_threads / 32) / 4, args.config.num_dispatch_threads + args.config.num_non_epilogue_threads + args.config.num_epilogue_threads, 32 / args.num_topk, args.num_experts / args.num_ranks,
    args.run_l1_phase ? "true" : "false", args.run_l2_phase ? "true" : "false",
    (args.run_l1_phase ? 1 : 0) | (args.run_l2_phase ? 2 : 0));
    }

    static void launch_impl(const KernelHandle& kernel, const LaunchConfigHandle& config, Args args) {
        DG_CUDA_UNIFIED_CHECK(launch_kernel(kernel, config,
            args.y,
            args.cumulative_local_expert_recv_stats,
            args.num_tokens,
            args.sym_buffer_ptrs,
            args.tensor_map_l1_acts,
            args.tensor_map_l1_acts_sf,
            args.tensor_map_l1_weights,
            args.l1_weights_gmem,
            args.l1_weights_sf,
            args.tensor_map_l1_output,
            args.tensor_map_l2_acts,
            args.tensor_map_l2_acts_sf,
            args.tensor_map_l2_weights,
            args.l2_weights_gmem,
            args.l2_weights_sf
        ));
    }
};

static void sm90_nvfp4_mega_moe(
    const torch::Tensor& y,
    const torch::Tensor& l1_acts, const torch::Tensor& l1_acts_sf,
    const torch::Tensor& l2_acts, const torch::Tensor& l2_acts_sf,
    const torch::Tensor& l1_weights, const torch::Tensor& l2_weights,
    const torch::Tensor& l1_weights_sf, const torch::Tensor& l2_weights_sf,
    const std::optional<torch::Tensor> cumulative_local_expert_recv_stats,
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

    // Heuristics
    auto config = get_mega_moe_config_sm90(
        num_ranks, num_experts, num_experts_per_rank,
        num_max_tokens_per_rank, num_tokens, num_topk,
        hidden, intermediate_hidden, num_padded_sf_pool_tokens);

    // BN128 remains the default layout. BN256 is opt-in for the verified
    // small-M packed-scratch path, where it halves the N-block count.
    const int nvfp4_block_n_from_layout = static_cast<int>(l1_weights_sf.size(3));
    DG_HOST_ASSERT(nvfp4_block_n_from_layout == 128 || nvfp4_block_n_from_layout == 256);
    const int nvfp4_block_n = get_env<int>("DG_SM90_NVFP4_BLOCK_N", nvfp4_block_n_from_layout);
    DG_HOST_ASSERT(nvfp4_block_n == 128 || nvfp4_block_n == 256);
    DG_HOST_ASSERT(nvfp4_block_n == nvfp4_block_n_from_layout);
    config.block_m = 64;
    config.block_n = nvfp4_block_n;
    config.block_k = 128;
    config.cluster_size = 1;
    const int nvfp4_env_block_m = get_env<int>("DG_SM90_MOE_BLOCK_M", 0);
    if (nvfp4_env_block_m > 0)
        config.block_m = nvfp4_env_block_m;
    config.num_epilogue_threads = get_env<int>(
        "DG_SM90_NVFP4_EPILOGUE_THREADS", config.block_n == 256 ? 256 : 128);
    DG_HOST_ASSERT((config.block_n == 128 and (config.num_epilogue_threads == 128 or config.num_epilogue_threads == 256)) or
                   (config.block_n == 256 and config.num_epilogue_threads == 256));

    const bool nvfp4_user_shape_override =
        get_env<int>("DG_SM90_MOE_BLOCK_M", 0) > 0 ||
        get_env<int>("DG_SM90_MOE_EPILOGUE_WG", 0) > 0 ||
        get_env<int>("DG_SM90_MOE_BLOCK_N", 128) != 128 ||
        get_env<int>("DG_SM90_NVFP4_EPILOGUE_THREADS", 0) > 0;
    const bool nvfp4_bm128_heuristic = get_env<int>("DG_SM90_NVFP4_BM128_HEURISTIC", 1) != 0;
    const bool nvfp4_bm128_main_m =
        num_tokens == 256 || num_tokens == 512 || num_tokens == 1024 || num_tokens == 2048 ||
        num_tokens == 4096 || num_tokens == 8192;
    if (!nvfp4_user_shape_override && nvfp4_bm128_heuristic && nvfp4_bm128_main_m && config.block_n == 128) {
        config.block_m = 128;
        config.num_epilogue_threads = 256;
    }

    config.num_experts_per_wave = get_num_experts_per_wave_for_mega_moe_sm90(
        num_experts_per_rank, num_tokens, num_topk,
        intermediate_hidden, config.block_m, config.block_n,
        device_runtime->get_num_sms());

    // NVFP4 dequant defaults to the idle-warps loader path for the supported
    // SM90 bridge shape (4 dispatch + 4 non-epilogue + 4 math warps). Set
    // DG_SM90_NVFP4_LOADER_DEQUANT=0 to use the math-side fallback.
    // The BLOCK_M=16/32 mma.sync branch is not fully wired for NVFP4.
    DG_HOST_ASSERT(config.block_m >= 64);
    DG_HOST_ASSERT(config.block_m == 64 or config.block_m == 128);
    // UE4M3 scale tensors are prepacked as (E, N/block_n, K/128, block_n, 8).
    DG_HOST_ASSERT(config.block_n == 128 or config.block_n == 256);
    const int nvfp4_dispatch_threads_default = config.block_n == 256 ? 64 : 128;
    const int nvfp4_non_epilogue_threads_default = config.block_n == 256 ? 64 : 128;
    const int nvfp4_dispatch_threads = get_env<int>("DG_SM90_NVFP4_DISPATCH_THREADS", nvfp4_dispatch_threads_default);
    const int nvfp4_non_epilogue_threads = get_env<int>("DG_SM90_NVFP4_NON_EPILOGUE_THREADS", nvfp4_non_epilogue_threads_default);
    DG_HOST_ASSERT(nvfp4_dispatch_threads == 64 || nvfp4_dispatch_threads == 128);
    DG_HOST_ASSERT(nvfp4_non_epilogue_threads == 64 || nvfp4_non_epilogue_threads == 128);
    const bool split_sfa_tma = get_env<int>("DG_SM90_MOE_SPLIT_SFA_TMA", 0) != 0;
    const int nvfp4_loader_dequant_default = config.block_n == 256 ?
        (num_tokens == 128 ? 1 : 0) :
        1;
    const bool nvfp4_loader_dequant_requested =
        get_env<int>("DG_SM90_NVFP4_LOADER_DEQUANT", nvfp4_loader_dequant_default) != 0 && !split_sfa_tma;
    const int nvfp4_packed_b_scratch_default = config.block_n == 256 ? 1 : 0;
    const bool nvfp4_packed_b_scratch_requested =
        get_env<int>("DG_SM90_NVFP4_PACKED_B_SCRATCH", nvfp4_packed_b_scratch_default) != 0;
    const bool nvfp4_bn256_packed_loader_dequant =
        config.block_n == 256 && nvfp4_non_epilogue_threads == 64 && nvfp4_packed_b_scratch_requested;
    const bool nvfp4_loader_dequant = nvfp4_loader_dequant_requested &&
        (nvfp4_non_epilogue_threads == 128 || nvfp4_bn256_packed_loader_dequant);
    const bool nvfp4_direct_scale_gmem = get_env<int>("DG_SM90_NVFP4_DIRECT_SCALE_GMEM", 0) != 0 &&
        !nvfp4_loader_dequant;
    const bool nvfp4_packed_b_scratch = nvfp4_packed_b_scratch_requested &&
        (!nvfp4_loader_dequant || nvfp4_bn256_packed_loader_dequant);
    const bool nvfp4_split_dequant_barrier = get_env<int>("DG_SM90_NVFP4_SPLIT_DEQUANT_BARRIER", 0) != 0 &&
        !nvfp4_loader_dequant && !nvfp4_packed_b_scratch;
    const bool nvfp4_strided_b_gmem_load = get_env<int>("DG_SM90_NVFP4_STRIDED_B_GMEM_LOAD", 0) != 0 &&
        !nvfp4_loader_dequant && !nvfp4_packed_b_scratch;
    const int weight_storage_k = static_cast<int>(l1_weights.size(2));
    const int weight_k_blocks = static_cast<int>(l1_weights_sf.size(2));
    const bool layout_fused_b_scale = weight_storage_k == weight_k_blocks * 80;
    const bool layout_plain_b_scale = weight_storage_k == weight_k_blocks * (config.block_k / 2);
    DG_HOST_ASSERT(layout_fused_b_scale || layout_plain_b_scale);
    const auto fused_b_scale_env = get_env<std::string>("DG_SM90_NVFP4_FUSED_B_SCALE", "");
    const bool nvfp4_fused_b_scale_layout = fused_b_scale_env.empty() ?
        layout_fused_b_scale : (get_env<int>("DG_SM90_NVFP4_FUSED_B_SCALE", 0) != 0);
    DG_HOST_ASSERT(nvfp4_fused_b_scale_layout == layout_fused_b_scale);
    DG_HOST_ASSERT(config.block_n == 128 || nvfp4_fused_b_scale_layout || !nvfp4_loader_dequant);
    DG_HOST_ASSERT(!nvfp4_fused_b_scale_layout || nvfp4_loader_dequant || nvfp4_packed_b_scratch);
    const int nvfp4_b_storage_per_k_block = nvfp4_fused_b_scale_layout ? 80 : config.block_k / 2;
    config.num_dispatch_threads = nvfp4_dispatch_threads;
    config.num_non_epilogue_threads = nvfp4_non_epilogue_threads;
    DG_HOST_ASSERT((config.num_dispatch_threads + config.num_non_epilogue_threads) % 128 == 0);
    DG_HOST_ASSERT(!(config.block_n == 256 && nvfp4_packed_b_scratch && num_tokens > 128));
    const int direct_l2_scatter_default =
        (config.block_n == 128 || (config.block_n == 256 && num_tokens <= 128)) ? 1 : 0;
    const bool direct_l2_scatter = get_env<int>("DG_SM90_NVFP4_DIRECT_L2_SCATTER", direct_l2_scatter_default) != 0;
    DG_HOST_ASSERT(!direct_l2_scatter || config.block_n == 128 ||
                   (config.block_n == 256 && config.block_m == 64 && config.num_epilogue_threads == 256));
    {
        auto align = [](int x, int a) { return ((x + a - 1) / a) * a; };
        constexpr int kSmemAlignment = 1024;
        const int num_dispatch_warps = config.num_dispatch_threads / 32;
        const int num_epilogue_warps = config.num_epilogue_threads / 32;
        const int num_epilogue_warpgroups = num_epilogue_warps / 4;
        const bool split_n_warpgroups = config.block_m == 64 && config.block_n == 256 &&
            num_epilogue_warpgroups == 2;
        const bool split_mn_warpgroups = config.block_m == 128 && config.block_n == 256 && num_epilogue_warpgroups == 4;
        const int wg_split_m = split_n_warpgroups ? 1 : (split_mn_warpgroups ? 2 : num_epilogue_warpgroups);
        const int wg_split_n = split_n_warpgroups ? num_epilogue_warpgroups : (split_mn_warpgroups ? 2 : 1);
        DG_HOST_ASSERT(wg_split_m * wg_split_n == num_epilogue_warpgroups);
        const int wg_block_m = config.block_m / wg_split_m;
        const int wg_block_n = config.block_n / wg_split_n;
        const int wg_l1_out_block_n = wg_block_n / 2;
        const int smem_expert_count_size = align(num_experts * static_cast<int>(sizeof(uint32_t)), kSmemAlignment);
        const int smem_send_buffers_size = align(
            static_cast<int>(layout::Buffer(layout::Data(hidden), num_dispatch_warps, 1).get_num_bytes()),
            kSmemAlignment);
        const int smem_dispatch_size = smem_expert_count_size + smem_send_buffers_size;
        const int smem_nvfp4_lut = align(128 * 8, kSmemAlignment);
        const int smem_cd_l1 = num_epilogue_warpgroups * wg_block_m * wg_l1_out_block_n;
        const int smem_cd_l2 = direct_l2_scatter ? 0 :
            num_epilogue_warpgroups * wg_block_m * wg_block_n * static_cast<int>(sizeof(nv_bfloat16));
        const int smem_cd = align(std::max(smem_cd_l1, smem_cd_l2), kSmemAlignment);
        const int smem_sfa_per_stage = align(2 * config.block_m * static_cast<int>(sizeof(float)), 128);
        const int smem_sfb_per_stage = (nvfp4_direct_scale_gmem || nvfp4_fused_b_scale_layout) ? 0 :
            align(config.block_n * (config.block_k / 16), 128);
        const int smem_packed_b_per_stage = nvfp4_packed_b_scratch ?
            config.block_n * nvfp4_b_storage_per_k_block : 0;
        const int smem_per_stage = config.block_m * config.block_k + config.block_n * config.block_k +
                                   smem_packed_b_per_stage + smem_sfa_per_stage + smem_sfb_per_stage;
        const int smem_barriers_fixed = (num_dispatch_warps + 2 * num_epilogue_warps) * 8;
        const int smem_barriers_per_stage = ((nvfp4_loader_dequant || nvfp4_split_dequant_barrier) ? 3 : 2) * 8;
        const int smem_fixed = smem_dispatch_size + smem_nvfp4_lut + smem_cd + smem_barriers_fixed;
        const int max_num_stages = (SM90ArchSpec::smem_capacity - smem_fixed) /
                                   (smem_per_stage + smem_barriers_per_stage);
        const int default_num_stages = (config.block_n == 128 && num_tokens > 32 && max_num_stages > 6) ?
            6 : max_num_stages;
        const int requested_num_stages = get_env<int>("DG_SM90_NVFP4_NUM_STAGES", default_num_stages);
        DG_HOST_ASSERT(max_num_stages >= 2);
        DG_HOST_ASSERT(requested_num_stages >= 2 && requested_num_stages <= max_num_stages);
        config.num_stages = requested_num_stages;
        config.smem_size = smem_fixed + config.num_stages * (smem_per_stage + smem_barriers_per_stage);
    }

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
                                                        nvfp4_b_storage_per_k_block, config.block_n,
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
                                                        nvfp4_b_storage_per_k_block, config.block_n,
                                                        static_cast<int>(l2_weights.stride(-2)),
                                                        0);

    // Stats can be optional
    int* cumulative_local_expert_recv_stats_ptr = nullptr;
    if (cumulative_local_expert_recv_stats.has_value())
        cumulative_local_expert_recv_stats_ptr = cumulative_local_expert_recv_stats->data_ptr<int>();

    // Launch
    const auto num_sms = device_runtime->get_num_sms();
    const bool fused_default = (num_tokens == 4096 && config.block_m == 64);
    const bool true_fused_small_m_default = (num_tokens == 32 || num_tokens == 64 || num_tokens == 128);
    const int split_l1_l2_default = (fused_default || true_fused_small_m_default) ? 0 : 1;
    const bool split_l1_l2 = get_env<int>("DG_SM90_MOE_SPLIT_L1_L2", split_l1_l2_default) != 0;
    const bool true_fused_l1_l2 = !split_l1_l2;
    DG_HOST_ASSERT(!(config.block_n == 256 && nvfp4_packed_b_scratch && split_l1_l2));
    const bool l1_dual_k_default =
        (hidden / config.block_k) % 2 == 0 && num_tokens != 32 && num_tokens < 512 &&
        !true_fused_l1_l2;
    const int l2_dual_accum_default = true_fused_l1_l2 ? 0 :
        ((num_tokens == 256 || num_tokens == 4096 || num_tokens == 8192) ? 1 : 0);
    const bool l2_arrival_counter_default = config.block_n == 128 &&
        (num_tokens == 32 || num_tokens == 128 || num_tokens == 512 || num_tokens == 1024 ||
         num_tokens == 2048 || num_tokens == 4096 || num_tokens == 8192) &&
        !(true_fused_l1_l2 && (num_tokens == 512 || num_tokens == 2048));

    const bool direct_scatter_metadata_broadcast_default =
        direct_l2_scatter && num_tokens >= 256;
    const int combine_7chunk_default =
        (num_tokens == 512 || num_tokens == 1024) ? 1 : 0;
    const bool async_l1_tma_store_requested = get_env<int>("DG_SM90_MOE_ASYNC_L1_STORE", 0) != 0;
    DG_HOST_ASSERT(!async_l1_tma_store_requested && "DG_SM90_MOE_ASYNC_L1_STORE is not supported for NVFP4 yet");
    const SM90NVFP4MegaMoESplitRuntime::Args args = {
        .num_max_tokens_per_rank = num_max_tokens_per_rank,
        .hidden = hidden, .intermediate_hidden = intermediate_hidden,
        .num_experts = num_experts, .num_topk = num_topk,
        .num_ranks = num_ranks,
        .activation_clamp = activation_clamp,
        .fast_math = fast_math,
        .async_l1_tma_store = false,
        .split_sfa_tma = split_sfa_tma,
        .direct_l2_scatter = direct_l2_scatter,
        .l2_dual_accum = get_env<int>("DG_SM90_MOE_L2_DUAL_ACCUM", l2_dual_accum_default) != 0,
        .phase_profile = get_env<int>("DG_SM90_MOE_PHASE_PROFILE", 0) != 0,
        .l2_arrival_counter = get_env<int>(
            "DG_SM90_NVFP4_L2_ARRIVAL_COUNTER", l2_arrival_counter_default ? 1 : 0) != 0,
        .direct_scatter_metadata_broadcast = get_env<int>(
            "DG_SM90_NVFP4_DIRECT_SCATTER_METADATA_BCAST",
            direct_scatter_metadata_broadcast_default ? 1 : 0) != 0,
        .l1_dual_k_accum = get_env<int>("DG_SM90_MOE_L1_DUAL_K", l1_dual_k_default ? 1 : 0) != 0,
        .l2_nmajor_schedule = get_env<int>("DG_SM90_MOE_L2_NMAJOR", 1) != 0,
        .l1_nmajor_schedule = get_env<int>("DG_SM90_MOE_L1_NMAJOR", 0) != 0,
        .k2_direct_accum = get_env<int>("DG_SM90_MOE_K2_DIRECT_ACCUM", 0) != 0,
        .loader_dequant = nvfp4_loader_dequant,
        .direct_scale_gmem = nvfp4_direct_scale_gmem,
        .packed_b_scratch = nvfp4_packed_b_scratch,
        .split_dequant_barrier = nvfp4_split_dequant_barrier,
        .strided_b_gmem_load = nvfp4_strided_b_gmem_load,
        .fused_b_scale_layout = nvfp4_fused_b_scale_layout,
        .combine_7chunk = get_env<int>("DG_SM90_NVFP4_COMBINE_7CHUNK", combine_7chunk_default) != 0,
        .skip_direct_scatter_sync = get_env<int>(
            "DG_SM90_NVFP4_SKIP_DIRECT_SCATTER_SYNC",
            (true_fused_l1_l2 && config.block_n == 256 && num_tokens == 128) ? 1 : 0) != 0,
        .run_l1_phase = true,
        .run_l2_phase = true,
        .config = config,
        .y = y.data_ptr(),
        .cumulative_local_expert_recv_stats = cumulative_local_expert_recv_stats_ptr,
        .num_tokens = num_tokens,
        .sym_buffer_ptrs = layout::SymBuffer<>(sym_buffer_ptrs, rank_idx),
        .tensor_map_l1_acts = tensor_map_l1_acts,
        .tensor_map_l1_acts_sf = tensor_map_l1_acts_sf,
        .tensor_map_l1_weights = tensor_map_l1_weights,
        .l1_weights_gmem = l1_weights.data_ptr<uint8_t>(),
        .l1_weights_sf = l1_weights_sf.data_ptr<uint8_t>(),
        .tensor_map_l1_output = tensor_map_l1_output,
        .tensor_map_l2_acts = tensor_map_l2_acts,
        .tensor_map_l2_acts_sf = tensor_map_l2_acts_sf,
        .tensor_map_l2_weights = tensor_map_l2_weights,
        .l2_weights_gmem = l2_weights.data_ptr<uint8_t>(),
        .l2_weights_sf = l2_weights_sf.data_ptr<uint8_t>(),
        .launch_args = LaunchArgs(num_sms, config.num_dispatch_threads + config.num_non_epilogue_threads + config.num_epilogue_threads,
                                  config.smem_size, config.cluster_size)
    };
    const int l2_no_dispatch_pipeline_default =
        (num_tokens == 128 || num_tokens == 256 || num_tokens == 512 || num_tokens == 1024 || num_tokens == 2048 || num_tokens == 4096 || num_tokens == 8192) ? 1 : 0;
    const bool l2_no_dispatch_pipeline =
        get_env<int>("DG_SM90_MOE_L2_NO_DISPATCH_PIPELINE", l2_no_dispatch_pipeline_default) != 0;

    const auto launch_with_phases = [&](const bool run_l1_phase,
                                        const bool run_l2_phase,
                                        const std::string& kernel_name) {
        auto phase_args = args;
        phase_args.run_l1_phase = run_l1_phase;
        phase_args.run_l2_phase = run_l2_phase;
        if (!run_l1_phase)
            phase_args.async_l1_tma_store = false;
        if (!run_l1_phase && run_l2_phase && l2_no_dispatch_pipeline) {
            phase_args.config.num_dispatch_threads = 0;
            phase_args.config.num_non_epilogue_threads = 128;
            std::tie(phase_args.config.num_stages, phase_args.config.smem_size) =
                get_pipeline_config_for_mega_moe_sm90(
                    SM90ArchSpec::smem_capacity,
                    num_experts, hidden,
                    phase_args.config.block_m, phase_args.config.block_n, phase_args.config.block_k,
                    phase_args.config.num_dispatch_threads / 32,
                    phase_args.config.num_epilogue_threads / 32);
            if (phase_args.direct_l2_scatter) {
                auto align = [](int x, int a) { return ((x + a - 1) / a) * a; };
                constexpr int kSmemAlignment = 1024;
                const int num_dispatch_warps = phase_args.config.num_dispatch_threads / 32;
                const int num_epilogue_warps = phase_args.config.num_epilogue_threads / 32;
                const int num_epilogue_warpgroups = num_epilogue_warps / 4;
                const int smem_expert_count_size = align(num_experts * static_cast<int>(sizeof(uint32_t)), kSmemAlignment);
                const int smem_send_buffers_size = align(
                    static_cast<int>(layout::Buffer(layout::Data(hidden), num_dispatch_warps, 1).get_num_bytes()),
                    kSmemAlignment);
                const int smem_dispatch_size = smem_expert_count_size + smem_send_buffers_size;
                const int smem_nvfp4_lut = align(128 * 8, kSmemAlignment);
                const int smem_cd_l1 = num_epilogue_warpgroups * phase_args.config.block_m * (phase_args.config.block_n / 2);
                const int smem_cd = align(smem_cd_l1, kSmemAlignment);
                const int smem_sfa_per_stage = align(2 * phase_args.config.block_m * static_cast<int>(sizeof(float)), 128);
                const int smem_sfb_per_stage = (phase_args.direct_scale_gmem || phase_args.fused_b_scale_layout) ? 0 :
                    align(phase_args.config.block_n * (phase_args.config.block_k / 16), 128);
                const int smem_packed_b_per_stage = phase_args.packed_b_scratch ?
                    phase_args.config.block_n * (phase_args.fused_b_scale_layout ? 80 : phase_args.config.block_k / 2) : 0;
                const int smem_per_stage = phase_args.config.block_m * phase_args.config.block_k +
                                           phase_args.config.block_n * phase_args.config.block_k +
                                           smem_packed_b_per_stage + smem_sfa_per_stage + smem_sfb_per_stage;
                const int smem_barriers_fixed = (num_dispatch_warps + 2 * num_epilogue_warps) * 8;
                const int smem_barriers_per_stage = ((phase_args.loader_dequant || phase_args.split_dequant_barrier) ? 3 : 2) * 8;
                const int smem_fixed = smem_dispatch_size + smem_nvfp4_lut + smem_cd + smem_barriers_fixed;
                phase_args.config.num_stages = (SM90ArchSpec::smem_capacity - smem_fixed) /
                    (smem_per_stage + smem_barriers_per_stage);
                DG_HOST_ASSERT(phase_args.config.num_stages >= 2);
                phase_args.config.smem_size = smem_fixed + phase_args.config.num_stages *
                    (smem_per_stage + smem_barriers_per_stage);
            }
            phase_args.launch_args = LaunchArgs(
                num_sms,
                phase_args.config.num_dispatch_threads + phase_args.config.num_non_epilogue_threads +
                    phase_args.config.num_epilogue_threads,
                phase_args.config.smem_size, phase_args.config.cluster_size);
        }
        const auto code = SM90NVFP4MegaMoESplitRuntime::generate(phase_args);
        const auto runtime = compiler->build(kernel_name, code);
        SM90NVFP4MegaMoESplitRuntime::launch(runtime, phase_args);
    };

    if (split_l1_l2) {
        launch_with_phases(true, false, "sm90_nvfp4_mega_moe_l1");
        launch_with_phases(
            false, true,
            l2_no_dispatch_pipeline ? "sm90_nvfp4_mega_moe_l2_nodisp" : "sm90_nvfp4_mega_moe_l2");
        return;
    }

    launch_with_phases(true, true, "sm90_nvfp4_mega_moe");
}

} // namespace deep_gemm
