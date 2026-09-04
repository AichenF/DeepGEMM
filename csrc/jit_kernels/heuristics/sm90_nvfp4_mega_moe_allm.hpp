#pragma once

#include <cstdint>
#include <string>

namespace deep_gemm {

// The all-M small/fused path was accepted with ptxas' default register-usage
// aggressiveness (5).
// DeepGEMM's global JIT default is 10, which regressed the long-lived RS
// fragments at Pro M64.  Appending 5 restores the dev-m build policy for the
// entire small/fused family; fast-math is included only when requested.
static std::string get_sm90_nvfp4_allm_small_jit_flags(
        const bool fast_math) {
    const std::string register_flags =
        "--ptxas-options=--register-usage-level=5";
    return fast_math ? "--use_fast_math " + register_flags : register_flags;
}

// Host-level composition of the dev-m dynamic small/fused baseline, accepted
// D40 pointwise overrides, and the merged bigM split family.  D40 is
// intentionally restricted to the physical 8xH200 Flash/Pro geometry and its
// measured M=8..128 scope.
enum class SM90NVFP4AllMArm : uint32_t {
    DevMDynamic,
    D40DynamicRS,
    D40KF424StaticRS,
    BigMSplit,
};

struct SM90NVFP4AllMPolicyInput {
    int num_sms;
    int num_ranks;
    int num_experts;
    int num_tokens;
    int num_topk;
    int hidden;
    int intermediate_hidden;
    int selected_kernel_block_n;
};

static constexpr SM90NVFP4AllMArm select_sm90_nvfp4_allm_arm(
        const SM90NVFP4AllMPolicyInput& input) {
    if (input.selected_kernel_block_n == 128)
        return SM90NVFP4AllMArm::BigMSplit;

    const bool is_h200 = input.num_sms == 132;
    const bool is_flash =
        is_h200 && input.num_ranks == 8 && input.num_experts == 256 &&
        input.num_topk == 6 && input.hidden == 4096 &&
        input.intermediate_hidden == 2048;
    const bool is_pro =
        is_h200 && input.num_ranks == 8 && input.num_experts == 384 &&
        input.num_topk == 6 && input.hidden == 7168 &&
        input.intermediate_hidden == 3072;
    if (is_flash) {
        switch (input.num_tokens) {
            case 8:
                return SM90NVFP4AllMArm::D40DynamicRS;
            case 16:
            case 32:
            case 64:
                return SM90NVFP4AllMArm::D40KF424StaticRS;
            case 128:
                return SM90NVFP4AllMArm::DevMDynamic;
            default:
                // D40 forbids interpolation between measured anchors.
                return SM90NVFP4AllMArm::DevMDynamic;
        }
    }

    if (is_pro) {
        switch (input.num_tokens) {
            case 8:
            case 16:
            case 64:
                return SM90NVFP4AllMArm::D40DynamicRS;
            case 32:
                return SM90NVFP4AllMArm::DevMDynamic;
            case 128:
                return SM90NVFP4AllMArm::D40KF424StaticRS;
            default:
                return SM90NVFP4AllMArm::DevMDynamic;
        }
    }

    // Every fused/small-M point not replaced by an accepted exact-key arm is
    // based on the dev-m dynamic scheduler.  Unsupported dev-m hardware or
    // model geometries fail closed in its own host heuristic instead of
    // silently switching back to the bigM static small-M implementation.
    return SM90NVFP4AllMArm::DevMDynamic;
}

static constexpr const char* sm90_nvfp4_allm_arm_name(
        const SM90NVFP4AllMArm arm) {
    switch (arm) {
        case SM90NVFP4AllMArm::DevMDynamic:
            return "devm-dynamic";
        case SM90NVFP4AllMArm::D40DynamicRS:
            return "d40-dynamic-rs-mode5";
        case SM90NVFP4AllMArm::D40KF424StaticRS:
            return "d40-kf424-static-rs";
        case SM90NVFP4AllMArm::BigMSplit:
            return "bigm-split-mode4";
    }
    return "unknown";
}

static_assert(select_sm90_nvfp4_allm_arm(
    {132, 8, 256, 8, 6, 4096, 2048, 256}) ==
    SM90NVFP4AllMArm::D40DynamicRS);
static_assert(select_sm90_nvfp4_allm_arm(
    {132, 8, 256, 64, 6, 4096, 2048, 256}) ==
    SM90NVFP4AllMArm::D40KF424StaticRS);
static_assert(select_sm90_nvfp4_allm_arm(
    {132, 8, 256, 128, 6, 4096, 2048, 256}) ==
    SM90NVFP4AllMArm::DevMDynamic);
static_assert(select_sm90_nvfp4_allm_arm(
    {132, 8, 256, 256, 6, 4096, 2048, 256}) ==
    SM90NVFP4AllMArm::DevMDynamic);
static_assert(select_sm90_nvfp4_allm_arm(
    {132, 8, 384, 512, 8, 6144, 2048, 256}) ==
    SM90NVFP4AllMArm::DevMDynamic);
static_assert(select_sm90_nvfp4_allm_arm(
    {132, 8, 320, 64, 5, 5120, 2560, 256}) ==
    SM90NVFP4AllMArm::DevMDynamic);
static_assert(select_sm90_nvfp4_allm_arm(
    {132, 8, 384, 64, 6, 7168, 3072, 256}) ==
    SM90NVFP4AllMArm::D40DynamicRS);
static_assert(select_sm90_nvfp4_allm_arm(
    {132, 8, 384, 128, 6, 7168, 3072, 256}) ==
    SM90NVFP4AllMArm::D40KF424StaticRS);
static_assert(select_sm90_nvfp4_allm_arm(
    {132, 8, 256, 2048, 6, 4096, 2048, 128}) ==
    SM90NVFP4AllMArm::BigMSplit);

}  // namespace deep_gemm
