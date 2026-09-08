#pragma once

#include <deep_gemm/layout/sm120_routed_moe.cuh>

namespace deep_gemm {

static bool can_use_sm120_routed_moe_fast_path(
    int num_ranks,
    int num_experts,
    int num_topk,
    int hidden,
    int intermediate_hidden,
    int num_external_shared_experts,
    int num_tokens,
    float activation_clamp,
    bool fast_math) {
    using Shape = sm120_routed_moe::Shape;
    return num_ranks == Shape::kWorldSize and
           num_experts == Shape::kExperts and
           num_topk == Shape::kTopK and
           hidden == Shape::kHidden and
           intermediate_hidden == Shape::kIntermediate and
           num_external_shared_experts == 1 and
           num_tokens >= 1 and num_tokens <= Shape::kMaxRows and
           activation_clamp == Shape::kActivationClamp and
           fast_math == Shape::kFastMath;
}

} // namespace deep_gemm
