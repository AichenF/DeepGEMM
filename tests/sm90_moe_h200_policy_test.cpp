#include <cassert>
#include <initializer_list>

#include "../csrc/jit_kernels/heuristics/sm90_moe_h200_policy.hpp"

using deep_gemm::Sm90MoeH200Workload;
using deep_gemm::select_sm90_moe_h200_policy;

static auto flash(const bool enabled, const int m) {
    return select_sm90_moe_h200_policy(enabled, 8, 256, 32, m, 6, 4096, 2048);
}

static auto pro(const bool enabled, const int m) {
    return select_sm90_moe_h200_policy(enabled, 8, 384, 48, m, 6, 7168, 3072);
}

int main() {
    // Non-H200 devices must never receive an automatic policy.
    assert(not flash(false, 128).enabled());
    assert(not pro(false, 8192).enabled());

    for (const int m : {8, 16, 32, 64, 256, 512, 1024, 2048, 4096, 4097})
        assert(not flash(true, m).enabled());
    for (const int m : {8, 16, 32, 64, 512, 513})
        assert(not pro(true, m).enabled());

    const auto flash128 = flash(true, 128);
    assert(flash128.workload == Sm90MoeH200Workload::Flash);
    assert(flash128.block_m == 64 and flash128.block_n == 256 and flash128.block_k == 128);
    assert(flash128.cluster_size == 1 and flash128.dispatch_warps == 2);
    assert(flash128.epilogue_warpgroups == 2 and flash128.direct_l2_scatter == 0);
    assert(flash128.l2_nmajor_schedule == 1 and flash128.one_warp_cleanup == 0);
    assert(flash128.l1_experts_per_wave == 4 and flash128.l2_experts_per_wave == 4);
    assert(flash128.l1_num_stages == 3 and flash128.l2_num_stages == 3);
    assert(flash128.fp8_combine == 1 and flash128.bf16_scaled_accum == 0);

    const auto flash8192 = flash(true, 8192);
    assert(flash8192.l1_experts_per_wave == 32 and flash8192.l2_experts_per_wave == 32);
    assert(flash8192.l1_num_stages == 3 and flash8192.l2_num_stages == 3);
    assert(flash8192.fp8_combine == 1 and flash8192.bf16_scaled_accum == 0);

    const auto pro128 = pro(true, 128);
    assert(pro128.workload == Sm90MoeH200Workload::Pro);
    assert(pro128.l1_block_n == 512 and pro128.l2_block_n == 256);
    assert(pro128.l1_experts_per_wave == 16 and pro128.l2_experts_per_wave == 16);
    assert(pro128.l1_num_stages == 2 and pro128.l2_num_stages == 3);
    assert(pro128.one_warp_cleanup == 0 and pro128.fp8_combine == 1);
    assert(pro128.bf16_scaled_accum == 1);

    const auto pro256 = pro(true, 256);
    assert(pro256.l1_block_n == 256 and pro256.l2_block_n == 256);
    assert(pro256.l1_block_k == 256 and pro256.l2_block_k == 128);
    assert(pro256.l1_experts_per_wave == 8 and pro256.l2_experts_per_wave == 48);
    assert(pro256.l1_num_stages == 2 and pro256.l2_num_stages == 3);
    assert(pro256.l1_num_sms == 128 and pro256.l2_num_sms == 128);
    assert(pro256.fp8_combine == 1 and pro256.bf16_scaled_accum == 0);

    for (const int m : {1024, 2048, 4096, 8192}) {
        const auto policy = pro(true, m);
        assert(policy.workload == Sm90MoeH200Workload::Pro);
        assert(policy.l1_block_n == 512 and policy.l2_block_n == 256);
        assert(policy.l1_experts_per_wave == 16 and policy.l2_experts_per_wave == 16);
        assert(policy.l1_num_stages == 2 and policy.l2_num_stages == 3);
        assert(policy.one_warp_cleanup == -1);
        assert(policy.fp8_combine == 1 and policy.bf16_scaled_accum == 1);
    }

    // Every workload identity field is part of the match.
    assert(not select_sm90_moe_h200_policy(true, 7, 384, 48, 128, 6, 7168, 3072).enabled());
    assert(not select_sm90_moe_h200_policy(true, 8, 383, 48, 128, 6, 7168, 3072).enabled());
    assert(not select_sm90_moe_h200_policy(true, 8, 384, 47, 128, 6, 7168, 3072).enabled());
    assert(not select_sm90_moe_h200_policy(true, 8, 384, 48, 128, 8, 7168, 3072).enabled());
    assert(not select_sm90_moe_h200_policy(true, 8, 384, 48, 128, 6, 6144, 3072).enabled());
    assert(not select_sm90_moe_h200_policy(true, 8, 384, 48, 128, 6, 7168, 2048).enabled());
}
