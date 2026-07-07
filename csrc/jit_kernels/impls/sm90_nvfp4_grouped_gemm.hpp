#pragma once

#include <torch/python.h>

#include "../../jit/compiler.hpp"
#include "../../jit/device_runtime.hpp"
#include "../../jit/kernel_runtime.hpp"
#include "../../utils/exception.hpp"
#include "../../utils/format.hpp"
#include "runtime_utils.hpp"

namespace deep_gemm {

class SM90NVFP4GroupedGemmRuntime final
    : public LaunchRuntime<SM90NVFP4GroupedGemmRuntime> {
public:
    struct Args {
        LaunchArgs launch_args;
        void* d;
        void* a;
        void* a_scale;
        void* w_packed;
        void* block_scale;
        void* global_scale;
        void* offsets;
        int m;
        int n;
        int k;
        int num_groups;
        int block_n;
        int small_block_k;
        bool small_m_warp_mma;
        bool warp_specialized;
        bool dual_m_consumers;
        bool pretransformed_weights;
        bool exact_byte_fused_weights;
        CUtensorMap tensor_map_a;
    };

    static std::string generate_impl(const Args& args) {
        if (args.small_m_warp_mma) {
            return fmt::format(R"(
#include <deep_gemm/impls/sm90_nvfp4_grouped_gemm.cuh>

using namespace deep_gemm;

static void __instantiate_kernel() {{
    auto ptr = reinterpret_cast<void*>(&sm90_nvfp4_grouped_gemm_small_m_cooperative_impl<
        {}, {}, {}, false, {}, {}, {}>);
}}
)", args.pretransformed_weights ? "true" : "false",
            args.exact_byte_fused_weights ? "true" : "false",
            args.small_block_k,
            args.n, args.k, args.num_groups);
        }
        if (args.dual_m_consumers) {
            return fmt::format(R"(
#include <deep_gemm/impls/sm90_nvfp4_grouped_gemm.cuh>

using namespace deep_gemm;

static void __instantiate_kernel() {{
    auto ptr = reinterpret_cast<void*>(&sm90_nvfp4_grouped_gemm_dual_m_ws_impl<
        {}, {}>);
}}
)", args.block_n,
            args.pretransformed_weights ? "true" : "false");
        }
        return fmt::format(R"(
#include <deep_gemm/impls/sm90_nvfp4_grouped_gemm.cuh>

using namespace deep_gemm;

static void __instantiate_kernel() {{
    auto ptr = reinterpret_cast<void*>(&sm90_nvfp4_grouped_gemm_impl<
        64, {}, 128, {}, {}>);
}}
)", args.block_n,
        args.warp_specialized ? "true" : "false",
        args.pretransformed_weights ? "true" : "false");
    }

    static void launch_impl(const KernelHandle& kernel,
                            const LaunchConfigHandle& config,
                            Args args) {
        if (args.small_m_warp_mma) {
            DG_CUDA_UNIFIED_CHECK(launch_kernel(
                kernel, config,
                args.d, args.tensor_map_a, args.a_scale,
                args.w_packed, args.block_scale, args.global_scale,
                args.offsets, args.m, args.n, args.k, args.num_groups));
            return;
        }
        DG_CUDA_UNIFIED_CHECK(launch_kernel(
            kernel, config,
            args.d, args.a, args.a_scale,
            args.w_packed, args.block_scale, args.global_scale, args.offsets,
            args.m, args.n, args.k, args.num_groups));
    }
};

static void sm90_m_grouped_nvfp4_gemm_nt_contiguous(
        const torch::Tensor& a,
        const torch::Tensor& a_scale,
        const torch::Tensor& w_packed,
        const torch::Tensor& block_scale,
        const torch::Tensor& global_scale,
        const torch::Tensor& d,
        const torch::Tensor& offsets,
        const int m, const int n, const int k, const int num_groups,
        const bool pretransformed_weights,
        const bool exact_byte_fused_weights) {
    constexpr int kBlockM = 64;
    constexpr int kBlockK = 128;
    constexpr int kLutBytes = 1024;
    const int num_sms = device_runtime->get_num_sms();
    const int dense_m_tiles = (m + kBlockM - 1) / kBlockM;
    const int expert_tail_tiles = m < num_groups ? m : num_groups;
    const int estimated_m_tiles =
        dense_m_tiles > expert_tail_tiles ? dense_m_tiles : expert_tail_tiles;
    const int estimated_n256_tiles = estimated_m_tiles * (n / 256);
    const bool enough_n256_tiles =
        estimated_n256_tiles * 4 >= num_sms * 3;
    const int average_tokens_per_group = (m + num_groups - 1) / num_groups;
    const bool small_m_warp_mma = m <= 128;
    const int small_block_k = k % 256 == 0 ? 256 : 128;
    const int block_n = enough_n256_tiles ? 256 : 128;
    const bool dual_m_consumers = !small_m_warp_mma && block_n == 256 &&
        average_tokens_per_group >= 128;
    const bool warp_specialized = !small_m_warp_mma &&
        (dual_m_consumers ||
         (block_n == 256 && estimated_n256_tiles >= num_sms));
    const int num_stages = 2;
    const int num_threads = small_m_warp_mma ? 128 :
        dual_m_consumers ? 384 :
        (warp_specialized ? 256 : 128);
    const int barrier_bytes = warp_specialized ? 2 * num_stages * 8 : 0;
    constexpr int kSmallMMaxTokens = 32;
    constexpr int kSmallMBarrierBytes = 8;
    const int small_m_residency_pad = small_block_k == 256 ? 4096 : 0;
    const int small_m_smem_size =
        (64 + kSmallMMaxTokens) * small_block_k + kLutBytes +
        kSmallMBarrierBytes + small_m_residency_pad;
    const int smem_size = small_m_warp_mma ? small_m_smem_size :
        num_stages * ((dual_m_consumers ? 2 : 1) * kBlockM + block_n) *
            kBlockK + kLutBytes + barrier_bytes;
    // With at most 128 tokens, no more than floor(128 / 33) experts can
    // require the odd token-group pass.  Keep the first pass dense so every
    // expert starts promptly, but compact the tail to those three possible
    // long-expert ranks instead of launching another full expert grid.
    constexpr int kSmallMMaxTailExperts = 3;
    const int grid_dim = small_m_warp_mma
        ? (num_groups + kSmallMMaxTailExperts) * (n / 64) : num_sms;
    CUtensorMap tensor_map_a{};
    if (small_m_warp_mma) {
        tensor_map_a = make_tma_2d_desc(
            a, k, m, 128, kSmallMMaxTokens, k, 128);
    }

    const SM90NVFP4GroupedGemmRuntime::Args args = {
        .launch_args = LaunchArgs(
            grid_dim, num_threads, smem_size, 1, false),
        .d = d.data_ptr(),
        .a = a.data_ptr(),
        .a_scale = a_scale.data_ptr(),
        .w_packed = w_packed.data_ptr(),
        .block_scale = block_scale.data_ptr(),
        .global_scale = global_scale.data_ptr(),
        .offsets = offsets.data_ptr(),
        .m = m,
        .n = n,
        .k = k,
        .num_groups = num_groups,
        .block_n = block_n,
        .small_block_k = small_block_k,
        .small_m_warp_mma = small_m_warp_mma,
        .warp_specialized = warp_specialized,
        .dual_m_consumers = dual_m_consumers,
        .pretransformed_weights = pretransformed_weights,
        .exact_byte_fused_weights = exact_byte_fused_weights,
        .tensor_map_a = tensor_map_a,
    };

    const auto code = SM90NVFP4GroupedGemmRuntime::generate(args);
    const auto runtime = compiler->build(
        small_m_warp_mma ? "sm90_nvfp4_grouped_gemm_small_m" :
                           "sm90_nvfp4_grouped_gemm",
        code);
    SM90NVFP4GroupedGemmRuntime::launch(runtime, args);
}

}  // namespace deep_gemm
