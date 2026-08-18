#pragma once

#include "../../jit/compiler.hpp"
#include "../../jit/device_runtime.hpp"
#include "../../jit/kernel_runtime.hpp"
#include "../heuristics/sm100.hpp"
#include "runtime_utils.hpp"

namespace deep_gemm {

class SM100MQALogitsF16WeightsRuntime final
    : public LaunchRuntime<SM100MQALogitsF16WeightsRuntime> {
public:
    struct Args {
        int seq_len;
        int seq_len_kv;
        int max_seqlen_k;
        int stride_logits;
        int num_heads, head_dim;
        bool is_compressed_logits;

        int num_q_stages;
        int num_kv_stages;
        int block_q;
        int block_kv;

        uint32_t* cu_seq_len_k_start_and_end;
        void* logits;

        CUtensorMap tensor_map_q;
        CUtensorMap tensor_map_kv;
        CUtensorMap tensor_map_kv_scales;
        CUtensorMap tensor_map_weights;
        at::ScalarType logits_dtype;

        int num_specialized_threads;
        int num_math_threads;

        LaunchArgs launch_args;
    };

    static std::string generate_impl(const Args& args) {
        DG_HOST_ASSERT(128 % args.num_heads == 0);

        return fmt::format(R"(
#include <deep_gemm/impls/sm100_fp8_mqa_logits_f16_weights.cuh>

using namespace deep_gemm;

static void __instantiate_kernel() {{
    auto ptr = reinterpret_cast<void*>(&sm100_fp8_mqa_logits_f16_weights<
        {}, {},
        {},
        {}, {},
        {}, {},
        {}, {},
        {}
    >);
}};
)", args.num_heads, args.head_dim,
    args.is_compressed_logits,
    args.block_q, args.block_kv,
    args.num_q_stages, args.num_kv_stages,
    args.num_specialized_threads, args.num_math_threads,
    to_string(args.logits_dtype));
    }

    static void launch_impl(const KernelHandle& kernel,
                            const LaunchConfigHandle& config, Args args) {
        DG_CUDA_UNIFIED_CHECK(launch_kernel(
            kernel, config, args.seq_len, args.seq_len_kv, args.max_seqlen_k,
            static_cast<uint64_t>(args.stride_logits),
            args.cu_seq_len_k_start_and_end, args.logits, args.tensor_map_q,
            args.tensor_map_kv, args.tensor_map_kv_scales,
            args.tensor_map_weights));
    }
};

static void sm100_mqa_logits_f16_weights(
    const torch::Tensor& q, const torch::Tensor& kv,
    const torch::Tensor& kv_scales, const torch::Tensor& weights,
    const torch::Tensor& cu_seq_len_k_start,
    const torch::Tensor& cu_seq_len_k_end, const torch::Tensor& logits,
    const at::ScalarType& logits_dtype, const int& seq_len,
    const int& seq_len_kv, const int& max_seqlen_k, const int& stride_logits,
    const int& num_heads, const int& head_dim, const int& block_q,
    const int& block_kv) {
    DG_HOST_ASSERT(device_runtime->get_arch_major() == 10);

    constexpr int num_specialized_threads = 128;
    constexpr int num_q_stages = 5, num_kv_stages = 8;
    constexpr int num_math_threads = 256;
    const bool is_compressed_logits = (max_seqlen_k > 0);
    auto weights_f16 = weights.to(torch::kFloat16).contiguous();

    // The two CTAs split each KV tile in half and share the same Q/weights tile.
    const auto tensor_map_q =
        make_tma_2d_desc(q, head_dim, seq_len * num_heads, head_dim,
                         block_q * num_heads, head_dim, head_dim);
    const auto tensor_map_kv = make_tma_2d_desc(
        kv, head_dim, seq_len_kv, head_dim, block_kv / 2, head_dim, head_dim);
    const auto tensor_map_kv_scales = make_tma_2d_desc(
        kv_scales,
        get_tma_aligned_size(seq_len_kv,
                             static_cast<int>(kv_scales.element_size())),
        1, block_kv / 2, 1, 0, 0);
    const auto tensor_map_weights = make_tma_2d_desc(
        weights_f16, num_heads, seq_len, num_heads, block_q, num_heads, 0);

    const int block_q_2cta = block_q * 2;
    const int smem_q_per_stage = block_q * num_heads * head_dim;
    const int smem_weight_per_stage = block_q_2cta * num_heads * 2;
    const int smem_kv_per_stage = (block_kv / 2) * head_dim;
    const int smem_kv_scale_raw =
        (block_kv / 2) * static_cast<int>(kv_scales.element_size());
    const int smem_kv_scale_per_stage =
        (smem_kv_scale_raw + 511) / 512 * 512;
    const int smem_kv_offset_per_stage = block_q_2cta * 8;
    const int num_umma_stages = 512 / (block_q_2cta * num_heads);
    const int num_barriers =
        num_q_stages * 2 + num_kv_stages * 2 + num_umma_stages * 2;

    int smem_size = 0;
    smem_size += num_q_stages * smem_q_per_stage;
    smem_size += num_q_stages * smem_weight_per_stage;
    smem_size += num_kv_stages * smem_kv_per_stage;
    smem_size += num_kv_stages * smem_kv_scale_per_stage;
    smem_size += num_q_stages * smem_kv_offset_per_stage;
    smem_size += num_barriers * 8;
    smem_size += 4;
    DG_HOST_ASSERT(smem_size <= SM100ArchSpec::smem_capacity);

    // The device kernel bulk-copies a full two-CTA Q tile without TMA
    // out-of-bounds zero fill. Pad the offsets buffer to that tile size;
    // {UINT32_MAX, 0} is neutral for the device-side min(start)/max(end)
    // reduction and suppresses compressed stores for padded rows.
    const int aligned_offset_rows = align(seq_len, block_q_2cta);
    torch::Tensor cu_seq_len_k_start_and_end = torch::empty(
        {aligned_offset_rows, 2}, cu_seq_len_k_start.options());
    cu_seq_len_k_start_and_end.select(1, 0).fill_(-1);
    cu_seq_len_k_start_and_end.select(1, 1).zero_();
    auto valid_offsets = cu_seq_len_k_start_and_end.narrow(0, 0, seq_len);
    valid_offsets.select(1, 0).copy_(cu_seq_len_k_start);
    valid_offsets.select(1, 1).copy_(cu_seq_len_k_end);
    cu_seq_len_k_start_and_end = cu_seq_len_k_start_and_end.reshape({-1}).contiguous();

    const SM100MQALogitsF16WeightsRuntime::Args args = {
        .seq_len = seq_len,
        .seq_len_kv = seq_len_kv,
        .max_seqlen_k = max_seqlen_k,
        .stride_logits = stride_logits,
        .num_heads = num_heads,
        .head_dim = head_dim,
        .is_compressed_logits = is_compressed_logits,
        .num_q_stages = num_q_stages,
        .num_kv_stages = num_kv_stages,
        .block_q = block_q,
        .block_kv = block_kv,
        .cu_seq_len_k_start_and_end = reinterpret_cast<uint32_t*>(
            cu_seq_len_k_start_and_end.data_ptr<int>()),
        .logits = logits.data_ptr(),
        .tensor_map_q = tensor_map_q,
        .tensor_map_kv = tensor_map_kv,
        .tensor_map_kv_scales = tensor_map_kv_scales,
        .tensor_map_weights = tensor_map_weights,
        .logits_dtype = logits_dtype,
        .num_specialized_threads = num_specialized_threads,
        .num_math_threads = num_math_threads,
        .launch_args = LaunchArgs(device_runtime->get_num_sms(),
                                  num_specialized_threads + num_math_threads,
                                  smem_size, 2)
    };
    const auto code = SM100MQALogitsF16WeightsRuntime::generate(args);
    const auto runtime =
        compiler->build("sm100_mqa_logits_f16_weights", code);
    SM100MQALogitsF16WeightsRuntime::launch(runtime, args);
}

} // namespace deep_gemm
