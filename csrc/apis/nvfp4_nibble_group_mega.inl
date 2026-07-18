static void nvfp4_nibble_group_mega_moe(
    const torch::Tensor& y,
    const std::tuple<torch::Tensor, torch::Tensor>& l1_weights_tuple,
    const std::tuple<torch::Tensor, torch::Tensor>& l2_weights_tuple,
    const std::optional<torch::Tensor>& cumulative_local_expert_recv_stats,
    const std::optional<torch::Tensor>& l1_global_scales,
    const std::optional<torch::Tensor>& l2_global_scales,
    const torch::Tensor& sym_buffer,
    const std::vector<int64_t>& sym_buffer_ptrs,
    const int& rank_idx,
    const int& num_max_tokens_per_rank,
    const int& num_experts,
    const int& num_topk,
    const std::tuple<int, int, int>& recipe,
    const std::string& activation,
    const std::optional<float>& activation_clamp_opt,
    const bool& fast_math,
    const bool& mode2_nibble_weights
) {
    const auto [l1_weights, l1_weights_sf] = l1_weights_tuple;
    const auto [l2_weights, l2_weights_sf] = l2_weights_tuple;
    DG_HOST_ASSERT(device_runtime->get_arch_major() == 9);
    const int num_tokens = static_cast<int>(y.size(0));
    const auto [rm, rn, rk] = recipe;
    DG_HOST_ASSERT(rm == 128 && rn == 128 && rk == 128);
    DG_HOST_ASSERT(activation == "swiglu");
    const float activation_clamp =
        activation_clamp_opt.value_or(std::numeric_limits<float>::infinity());
    DG_HOST_ASSERT(activation_clamp >= 0);
    DG_HOST_ASSERT(get_major_type_ab(l1_weights) == cute::UMMA::Major::K);
    DG_HOST_ASSERT(get_major_type_ab(l2_weights) == cute::UMMA::Major::K);
    DG_HOST_ASSERT(l1_weights.scalar_type() == torch::kUInt8);
    DG_HOST_ASSERT(l2_weights.scalar_type() == torch::kUInt8);
    DG_HOST_ASSERT(l1_weights_sf.scalar_type() == torch::kUInt8);
    DG_HOST_ASSERT(l2_weights_sf.scalar_type() == torch::kUInt8);
    DG_HOST_ASSERT(l1_weights_sf.dim() == 5 && l2_weights_sf.dim() == 5);

    constexpr int kBlockN = 256;
    const auto [num_experts_per_rank, intermediate_hidden_2, hidden_storage] =
        get_shape<3>(l1_weights);
    const auto [num_experts_per_rank_l2, hidden_l2, intermediate_hidden_storage] =
        get_shape<3>(l2_weights);
    const int hidden = static_cast<int>(l1_weights_sf.size(2)) * 128;
    const int intermediate_hidden = static_cast<int>(l2_weights_sf.size(2)) * 128;
    DG_HOST_ASSERT(intermediate_hidden <= 2048);
    DG_HOST_ASSERT(l1_weights_sf.size(3) == kBlockN);
    DG_HOST_ASSERT(l2_weights_sf.size(3) == kBlockN);
    DG_HOST_ASSERT(hidden_storage == (hidden / 128) * kSM90NVFP4BStoragePerKBlock);
    DG_HOST_ASSERT(intermediate_hidden_storage ==
                   (intermediate_hidden / 128) * kSM90NVFP4BStoragePerKBlock);
    DG_HOST_ASSERT(num_tokens <= num_max_tokens_per_rank);
    DG_HOST_ASSERT(num_experts_per_rank == num_experts_per_rank_l2);
    DG_HOST_ASSERT(hidden == hidden_l2);
    DG_HOST_ASSERT(intermediate_hidden_2 == 2 * intermediate_hidden);
    DG_HOST_ASSERT(l1_weights.is_contiguous() && l2_weights.is_contiguous());
    DG_HOST_ASSERT(hidden % 128 == 0 && intermediate_hidden % 128 == 0);

    DG_HOST_ASSERT(l1_weights_sf.size(0) == num_experts_per_rank);
    DG_HOST_ASSERT(l1_weights_sf.size(1) == intermediate_hidden * 2 / kBlockN);
    DG_HOST_ASSERT(l1_weights_sf.size(2) == hidden / 128);
    DG_HOST_ASSERT(l1_weights_sf.size(4) == 8);
    DG_HOST_ASSERT(l1_weights_sf.is_contiguous());
    DG_HOST_ASSERT(l2_weights_sf.size(0) == num_experts_per_rank);
    DG_HOST_ASSERT(l2_weights_sf.size(1) == hidden / kBlockN);
    DG_HOST_ASSERT(l2_weights_sf.size(2) == intermediate_hidden / 128);
    DG_HOST_ASSERT(l2_weights_sf.size(4) == 8);
    DG_HOST_ASSERT(l2_weights_sf.is_contiguous());

    if (cumulative_local_expert_recv_stats.has_value()) {
        DG_HOST_ASSERT(cumulative_local_expert_recv_stats->scalar_type() == torch::kInt);
        const auto stats_numel = cumulative_local_expert_recv_stats->numel();
        const bool phase_profile = get_env<int>("DG_SM90_MOE_PHASE_PROFILE", 0) != 0;
        DG_HOST_ASSERT(stats_numel == num_experts_per_rank ||
                       (phase_profile && stats_numel >= num_experts_per_rank + 64));
        DG_HOST_ASSERT(cumulative_local_expert_recv_stats->is_contiguous());
    }
    if (l1_global_scales.has_value()) {
        DG_HOST_ASSERT(l1_global_scales->scalar_type() == torch::kFloat32);
        DG_HOST_ASSERT(l1_global_scales->numel() == num_experts_per_rank);
        DG_HOST_ASSERT(l1_global_scales->is_contiguous());
        DG_HOST_ASSERT(l1_global_scales->device() == y.device());
    }
    if (l2_global_scales.has_value()) {
        DG_HOST_ASSERT(l2_global_scales->scalar_type() == torch::kFloat32);
        DG_HOST_ASSERT(l2_global_scales->numel() == num_experts_per_rank);
        DG_HOST_ASSERT(l2_global_scales->is_contiguous());
        DG_HOST_ASSERT(l2_global_scales->device() == y.device());
    }

    const int num_ranks = static_cast<int>(sym_buffer_ptrs.size());
    DG_HOST_ASSERT(num_experts == num_experts_per_rank * num_ranks);
    const auto [num_required_bytes, slice] = get_symm_buffer_size_for_mega_moe(
        num_ranks, num_experts, num_max_tokens_per_rank, num_topk,
        hidden, intermediate_hidden, true, activation);
    DG_HOST_ASSERT(sym_buffer.nbytes() >= static_cast<size_t>(num_required_bytes));
    const auto [x, x_sf, topk_idx, topk_weights,
                l1_acts, l1_acts_sf, l2_acts, l2_acts_sf] = slice(sym_buffer);

    // Reuse the validated per-128 schedule while decoding the public
    // grouped-nibble weight layout.  Gate on the full 132-SM launch so H20
    // keeps its existing small-M selector and tuning.
    const bool full_sm_mimo_small_m_per128 =
        device_runtime->get_num_sms() >= 132 && num_ranks == 8 &&
        (num_tokens == 8 || num_tokens == 16 ||
         num_tokens == 32 || num_tokens == 64) &&
        num_topk == 8 && num_experts_per_rank == 48 &&
        hidden == 6144 && intermediate_hidden == 2048;
    DG_HOST_ASSERT(!mode2_nibble_weights ||
        (device_runtime->get_num_sms() >= 132 && num_ranks == 8 &&
         num_topk == 8 && num_experts_per_rank == 48 &&
         hidden == 6144 && intermediate_hidden == 2048));
    if (full_sm_mimo_small_m_per128) {
        // Mode-2 M64 is fastest with the existing braided LUT-window decoder;
        // M8/16/32 retain the grouped decoder with mode-2 sign extraction.
        const bool grouped_nibble_decoder =
            !mode2_nibble_weights || num_tokens != 64;
        sm90_nvfp4_per128_pro_braided_3stage_mega_moe(
            y,
            l1_acts, l1_acts_sf,
            l2_acts, l2_acts_sf,
            l1_weights, l2_weights,
            l1_weights_sf, l2_weights_sf,
            cumulative_local_expert_recv_stats,
            l1_global_scales, l2_global_scales,
            sym_buffer_ptrs,
            rank_idx, num_max_tokens_per_rank,
            num_experts_per_rank,
            num_tokens, num_topk,
            hidden, intermediate_hidden,
            activation_clamp, fast_math,
            grouped_nibble_decoder,
            mode2_nibble_weights);
    } else {
        sm90_nvfp4_nibble_group_mega_moe(
            y,
            l1_acts, l1_acts_sf,
            l2_acts, l2_acts_sf,
            l1_weights, l2_weights,
            l1_weights_sf, l2_weights_sf,
            cumulative_local_expert_recv_stats,
            l1_global_scales, l2_global_scales,
            sym_buffer_ptrs,
            rank_idx, num_max_tokens_per_rank,
            num_experts_per_rank,
            num_tokens, num_topk,
            hidden, intermediate_hidden,
            activation_clamp, fast_math,
            mode2_nibble_weights);
    }
    if (get_env<int>("DG_COMM_KERNEL_DEBUG"))
        sym_buffer.zero_();
}
