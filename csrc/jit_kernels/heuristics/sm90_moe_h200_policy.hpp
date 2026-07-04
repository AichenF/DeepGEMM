#pragma once

#include <ostream>

namespace deep_gemm {

enum class Sm90MoeH200Workload {
    None,
    Flash,
    Pro
};

struct Sm90MoeH200Policy {
    Sm90MoeH200Workload workload = Sm90MoeH200Workload::None;

    // Shared/base configuration. Zero means inherit the existing selector;
    // boolean selectors use -1 for inherit.
    int block_m = 0, block_n = 0, block_k = 0;
    int cluster_size = 0, dispatch_warps = 0, epilogue_warpgroups = 0;
    int direct_l2_scatter = -1, l2_nmajor_schedule = -1;
    int one_warp_cleanup = -1;

    // Phase-local configuration. Zero means inherit the shared configuration.
    int l1_block_n = 0, l2_block_n = 0;
    int l1_block_k = 0, l2_block_k = 0;
    int l1_experts_per_wave = 0, l2_experts_per_wave = 0;
    int l1_num_stages = 0, l2_num_stages = 0;
    int l1_num_sms = 0, l2_num_sms = 0;

    // Global execution features selected by the retained candidates.
    int fp8_combine = 0;
    int bf16_scaled_accum = 0;

    bool enabled() const {
        return workload != Sm90MoeH200Workload::None;
    }

    friend std::ostream& operator << (std::ostream& os, const Sm90MoeH200Policy& policy) {
        const char* workload = policy.workload == Sm90MoeH200Workload::Flash ? "flash" :
                               policy.workload == Sm90MoeH200Workload::Pro ? "pro" : "none";
        os << "Sm90MoeH200Policy(workload=" << workload
           << ", block_m=" << policy.block_m
           << ", block_n=" << policy.block_n
           << ", block_k=" << policy.block_k
           << ", cluster_size=" << policy.cluster_size
           << ", dispatch_warps=" << policy.dispatch_warps
           << ", epilogue_warpgroups=" << policy.epilogue_warpgroups
           << ", direct_l2_scatter=" << policy.direct_l2_scatter
           << ", l2_nmajor_schedule=" << policy.l2_nmajor_schedule
           << ", one_warp_cleanup=" << policy.one_warp_cleanup
           << ", l1_block_n=" << policy.l1_block_n
           << ", l2_block_n=" << policy.l2_block_n
           << ", l1_block_k=" << policy.l1_block_k
           << ", l2_block_k=" << policy.l2_block_k
           << ", l1_experts_per_wave=" << policy.l1_experts_per_wave
           << ", l2_experts_per_wave=" << policy.l2_experts_per_wave
           << ", l1_num_stages=" << policy.l1_num_stages
           << ", l2_num_stages=" << policy.l2_num_stages
           << ", l1_num_sms=" << policy.l1_num_sms
           << ", l2_num_sms=" << policy.l2_num_sms
           << ", fp8_combine=" << policy.fp8_combine
           << ", bf16_scaled_accum=" << policy.bf16_scaled_accum << ")";
        return os;
    }
};

// Pure table lookup kept independent of CUDA/device state so its conservative
// coverage and fallthrough behavior can be tested on a CPU host.
inline Sm90MoeH200Policy select_sm90_moe_h200_policy(
    const bool h200_enabled,
    const int num_ranks, const int num_experts, const int num_experts_per_rank,
    const int num_tokens, const int num_topk,
    const int hidden, const int intermediate_hidden) {
    Sm90MoeH200Policy policy;
    if (not h200_enabled)
        return policy;

    const bool is_flash =
        num_ranks == 8 and num_experts == 256 and num_experts_per_rank == 32 and
        num_topk == 6 and hidden == 4096 and intermediate_hidden == 2048;
    const bool is_pro =
        num_ranks == 8 and num_experts == 384 and num_experts_per_rank == 48 and
        num_topk == 6 and hidden == 7168 and intermediate_hidden == 3072;

    const auto set_common = [&](const Sm90MoeH200Workload workload) {
        policy.workload = workload;
        policy.block_m = 64;
        policy.block_n = 256;
        policy.block_k = 128;
        policy.cluster_size = 1;
        policy.dispatch_warps = 2;
        policy.epilogue_warpgroups = 2;
        policy.direct_l2_scatter = 0;
        policy.l2_nmajor_schedule = 1;
        policy.fp8_combine = 1;
    };

    if (is_flash and num_tokens == 128) {
        set_common(Sm90MoeH200Workload::Flash);
        policy.one_warp_cleanup = 0;
        policy.l1_experts_per_wave = policy.l2_experts_per_wave = 4;
        policy.l1_num_stages = policy.l2_num_stages = 3;
    } else if (is_flash and num_tokens == 8192) {
        set_common(Sm90MoeH200Workload::Flash);
        policy.one_warp_cleanup = 0;
        policy.l1_experts_per_wave = policy.l2_experts_per_wave = 32;
        policy.l1_num_stages = policy.l2_num_stages = 3;
    } else if (is_pro and num_tokens == 128) {
        set_common(Sm90MoeH200Workload::Pro);
        policy.one_warp_cleanup = 0;
        policy.l1_block_n = 512;
        policy.l2_block_n = 256;
        policy.l1_experts_per_wave = policy.l2_experts_per_wave = 16;
        policy.l1_num_stages = 2;
        policy.l2_num_stages = 3;
        policy.bf16_scaled_accum = 1;
    } else if (is_pro and num_tokens == 256) {
        set_common(Sm90MoeH200Workload::Pro);
        policy.one_warp_cleanup = 0;
        policy.l1_block_n = policy.l2_block_n = 256;
        policy.l1_block_k = 256;
        policy.l2_block_k = 128;
        policy.l1_experts_per_wave = 8;
        policy.l2_experts_per_wave = 48;
        policy.l1_num_stages = 2;
        policy.l2_num_stages = 3;
        policy.l1_num_sms = policy.l2_num_sms = 128;
    } else if (is_pro and
               (num_tokens == 1024 or num_tokens == 2048 or
                num_tokens == 4096 or num_tokens == 8192)) {
        set_common(Sm90MoeH200Workload::Pro);
        policy.l1_block_n = 512;
        policy.l2_block_n = 256;
        policy.l1_experts_per_wave = policy.l2_experts_per_wave = 16;
        policy.l1_num_stages = 2;
        policy.l2_num_stages = 3;
        policy.bf16_scaled_accum = 1;
    }
    return policy;
}

} // namespace deep_gemm
