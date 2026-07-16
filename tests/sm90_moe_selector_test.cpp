#include <algorithm>
#include <array>
#include <cstdio>
#include <cstdlib>

#include "../csrc/jit_kernels/heuristics/sm90_mega_moe.hpp"

using deep_gemm::MegaMoESM90Config;
using deep_gemm::Sm90MoeHardwareProfile;
using deep_gemm::Sm90MoeHeuristicInput;
using deep_gemm::Sm90MoeLaunchConfig;
using deep_gemm::classify_sm90_moe_hardware;
using deep_gemm::is_sm90_moe_launch_config_legal;
using deep_gemm::select_mega_moe_sm90;

constexpr int kH20Sms = 78;
constexpr int kH200Sms = 132;

[[noreturn]] static void fail_check(
    const char* expression, const char* file, const int line) {
    std::fprintf(stderr, "%s:%d: check failed: %s\n", file, line, expression);
    std::exit(EXIT_FAILURE);
}

#define TEST_CHECK(expression) \
    do { \
        if (not (expression)) \
            fail_check(#expression, __FILE__, __LINE__); \
    } while (false)

static void clear_debug_env() {
    for (const char* name : {"DG_JIT_DEBUG", "DG_PRINT_CONFIGS"}) {
        unsetenv(name);
    }
}

static Sm90MoeHeuristicInput make_input(
    const int launch_sms,
    const int num_ranks,
    const int num_experts_per_rank,
    const int num_tokens,
    const int num_topk,
    const int hidden,
    const int intermediate_hidden) {
    const int num_experts = num_ranks * num_experts_per_rank;
    const int num_max_tokens_per_rank = std::max(128, num_tokens);
    const int num_max_pool_tokens = deep_gemm::layout::get_num_max_pool_tokens(
        num_ranks, num_max_tokens_per_rank, num_topk, num_experts_per_rank);
    const int num_padded_sf_pool_tokens =
        deep_gemm::layout::get_num_padded_sf_pool_tokens(num_max_pool_tokens, 64);
    return {
        launch_sms,
        num_ranks, num_experts, num_experts_per_rank,
        num_max_tokens_per_rank, num_tokens, num_topk,
        hidden, intermediate_hidden,
        num_padded_sf_pool_tokens,
    };
}

static bool same_phase_schedule(
    const MegaMoESM90Config& lhs,
    const MegaMoESM90Config& rhs) {
    return lhs.block_m == rhs.block_m and
           lhs.block_n == rhs.block_n and
           lhs.block_k == rhs.block_k and
           lhs.num_experts_per_wave == rhs.num_experts_per_wave and
           lhs.num_sms == rhs.num_sms and
           lhs.num_stages == rhs.num_stages and
           lhs.num_dispatch_threads == rhs.num_dispatch_threads and
           lhs.num_non_epilogue_threads == rhs.num_non_epilogue_threads and
           lhs.num_epilogue_threads == rhs.num_epilogue_threads and
           lhs.direct_l2_scatter == rhs.direct_l2_scatter and
           lhs.nmajor_schedule == rhs.nmajor_schedule and
           lhs.one_warp_cleanup == rhs.one_warp_cleanup and
           lhs.swap_ab == rhs.swap_ab;
}

static bool has_specialized_schedule(
    const Sm90MoeHeuristicInput& input,
    const Sm90MoeLaunchConfig& config) {
    auto generic_phase = deep_gemm::make_generic_mega_moe_config_sm90(input);
    Sm90MoeLaunchConfig generic {generic_phase, generic_phase, {}};
    generic.l1.nmajor_schedule = false;
    if (not is_sm90_moe_launch_config_legal(input, generic, false)) {
        generic_phase = deep_gemm::make_generic_mega_moe_config_sm90(input, true);
        generic.l1 = generic.l2 = generic_phase;
        generic.l1.nmajor_schedule = false;
    }
    return not (same_phase_schedule(config.l1, generic.l1) and
                same_phase_schedule(config.l2, generic.l2));
}

struct HighSmGolden {
    int m;
    bool specialized;
    int l1_block_n;
    int l1_block_k;
    int l1_epw;
    int l2_epw;
    int l1_stages;
    int l2_stages;
    bool cleanup;
    bool l1_nmajor = false;
};

static void check_high_sm_golden(
    const Sm90MoeHeuristicInput& input,
    const Sm90MoeLaunchConfig& config,
    const HighSmGolden& golden) {
    TEST_CHECK(is_sm90_moe_launch_config_legal(input, config));
    TEST_CHECK(config.l1.num_sms == input.launch_num_sms);
    TEST_CHECK(config.l2.num_sms == input.launch_num_sms);
    TEST_CHECK(has_specialized_schedule(input, config) == golden.specialized);
    if (not golden.specialized)
        return;
    TEST_CHECK(config.l1.block_m == 64 and config.l2.block_m == 64);
    TEST_CHECK(config.l1.block_n == golden.l1_block_n);
    TEST_CHECK(config.l2.block_n == 256);
    TEST_CHECK(config.l1.block_k == golden.l1_block_k);
    TEST_CHECK(config.l2.block_k == 128);
    TEST_CHECK(config.l1.num_experts_per_wave == golden.l1_epw);
    TEST_CHECK(config.l2.num_experts_per_wave == golden.l2_epw);
    TEST_CHECK(config.l1.num_stages == golden.l1_stages);
    TEST_CHECK(config.l2.num_stages == golden.l2_stages);
    TEST_CHECK(config.l1.nmajor_schedule == golden.l1_nmajor);
    TEST_CHECK(not config.l2.direct_l2_scatter);
    TEST_CHECK(config.l2.nmajor_schedule);
    TEST_CHECK(config.l2.one_warp_cleanup == golden.cleanup);
}

struct LowSmGolden {
    int m;
    int block_n;
    int epw;
    int stages;
    bool direct_l2_scatter;
    bool l2_nmajor;
    bool one_warp_cleanup;
    bool swap_ab;
};

static void check_low_sm_golden(
    const Sm90MoeHeuristicInput& input,
    const Sm90MoeLaunchConfig& config,
    const LowSmGolden& golden) {
    TEST_CHECK(is_sm90_moe_launch_config_legal(input, config));
    for (const auto* phase : {&config.l1, &config.l2}) {
        TEST_CHECK(phase->block_m == 64);
        TEST_CHECK(phase->block_n == golden.block_n);
        TEST_CHECK(phase->block_k == 128);
        TEST_CHECK(phase->num_experts_per_wave == golden.epw);
        TEST_CHECK(phase->num_sms == input.launch_num_sms);
        TEST_CHECK(phase->num_stages == golden.stages);
        TEST_CHECK(phase->direct_l2_scatter == golden.direct_l2_scatter);
        TEST_CHECK(phase->one_warp_cleanup == golden.one_warp_cleanup);
        TEST_CHECK(phase->swap_ab == golden.swap_ab);
    }
    TEST_CHECK(not config.l1.nmajor_schedule);
    TEST_CHECK(config.l2.nmajor_schedule == golden.l2_nmajor);
    TEST_CHECK(not config.numerical.bf16_scaled_accum);
}

int main() {
    clear_debug_env();

    TEST_CHECK(classify_sm90_moe_hardware(kH20Sms) ==
           Sm90MoeHardwareProfile::LowSm);
    TEST_CHECK(classify_sm90_moe_hardware(99) ==
           Sm90MoeHardwareProfile::LowSm);
    TEST_CHECK(classify_sm90_moe_hardware(100) ==
           Sm90MoeHardwareProfile::HighSm);
    TEST_CHECK(classify_sm90_moe_hardware(kH200Sms) ==
           Sm90MoeHardwareProfile::HighSm);

    constexpr std::array<HighSmGolden, 11> compact_golden {{
        {8, false, 0, 0, 0, 0, 0, 0, false},
        {16, false, 0, 0, 0, 0, 0, 0, false},
        {32, false, 0, 0, 0, 0, 0, 0, false},
        {64, false, 0, 0, 0, 0, 0, 0, false},
        {128, true, 256, 128, 4, 4, 3, 3, false},
        {256, false, 0, 0, 0, 0, 0, 0, false},
        {512, false, 0, 0, 0, 0, 0, 0, false},
        {1024, true, 256, 128, 32, 32, 4, 4, false},
        {2048, false, 0, 0, 0, 0, 0, 0, false},
        {4096, false, 0, 0, 0, 0, 0, 0, false},
        {8192, true, 256, 128, 32, 32, 4, 4, false},
    }};
    for (const auto& golden : compact_golden) {
        const auto input = make_input(
            kH200Sms, 8, 32, golden.m, 6, 4096, 2048);
        check_high_sm_golden(input, select_mega_moe_sm90(input), golden);
    }

    constexpr std::array<HighSmGolden, 11> wide_golden {{
        {8, false, 0, 0, 0, 0, 0, 0, false},
        {16, false, 0, 0, 0, 0, 0, 0, false},
        {32, false, 0, 0, 0, 0, 0, 0, false},
        {64, false, 0, 0, 0, 0, 0, 0, false},
        {128, true, 512, 128, 16, 16, 2, 3, false},
        {256, true, 256, 256, 8, 48, 2, 3, false},
        {512, true, 512, 128, 16, 16, 2, 3, false, true},
        {1024, true, 512, 128, 16, 16, 2, 3, true, true},
        {2048, true, 512, 128, 16, 16, 2, 3, true},
        {4096, true, 512, 128, 16, 16, 2, 3, true},
        {8192, true, 512, 128, 16, 16, 2, 3, true},
    }};
    for (const auto& golden : wide_golden) {
        const auto input = make_input(
            kH200Sms, 8, 48, golden.m, 6, 7168, 3072);
        check_high_sm_golden(input, select_mega_moe_sm90(input), golden);
    }

    TEST_CHECK(select_mega_moe_sm90(
        make_input(kH200Sms, 8, 32, 16, 6, 4096, 2048))
        .numerical.bf16_scaled_accum);  // compact load = 3
    TEST_CHECK(not select_mega_moe_sm90(
        make_input(kH200Sms, 8, 32, 32, 6, 4096, 2048))
        .numerical.bf16_scaled_accum);  // compact load = 6
    TEST_CHECK(select_mega_moe_sm90(
        make_input(kH200Sms, 8, 32, 2048, 6, 4096, 2048))
        .numerical.bf16_scaled_accum);  // compact load = 384
    TEST_CHECK(not select_mega_moe_sm90(
        make_input(kH200Sms, 8, 32, 4096, 6, 4096, 2048))
        .numerical.bf16_scaled_accum);
    TEST_CHECK(select_mega_moe_sm90(
        make_input(kH200Sms, 8, 48, 32, 6, 7168, 3072))
        .numerical.bf16_scaled_accum);  // wide load = 4
    TEST_CHECK(not select_mega_moe_sm90(
        make_input(kH200Sms, 8, 48, 64, 6, 7168, 3072))
        .numerical.bf16_scaled_accum);  // wide load = 8
    TEST_CHECK(select_mega_moe_sm90(
        make_input(kH200Sms, 8, 48, 128, 6, 7168, 3072))
        .numerical.bf16_scaled_accum);

    // Equal routed load selects the same tile/pipeline family without an
    // expert-count or top-k identity match. EPW is resolved to a legal divisor.
    const auto compact_alt_input = make_input(
        kH200Sms, 4, 24, 72, 8, 4096, 2048);  // load = 24
    const auto compact_alt = select_mega_moe_sm90(compact_alt_input);
    TEST_CHECK(has_specialized_schedule(compact_alt_input, compact_alt));
    TEST_CHECK(compact_alt.l1.block_n == 256 and compact_alt.l1.num_stages == 3);
    TEST_CHECK(compact_alt.l1.num_experts_per_wave == 4);
    TEST_CHECK(compact_alt.numerical.bf16_scaled_accum);

    const auto wide_alt_input = make_input(
        kH200Sms, 4, 32, 128, 8, 7168, 3072);  // load = 32
    const auto wide_alt = select_mega_moe_sm90(wide_alt_input);
    TEST_CHECK(has_specialized_schedule(wide_alt_input, wide_alt));
    TEST_CHECK(wide_alt.l1.block_k == 256);
    TEST_CHECK(wide_alt.l1.num_experts_per_wave == 8);
    TEST_CHECK(wide_alt.l2.num_experts_per_wave == 32);
    TEST_CHECK(wide_alt.l1.num_sms == kH200Sms and wide_alt.l2.num_sms == kH200Sms);
    TEST_CHECK(wide_alt.numerical.bf16_scaled_accum);

    constexpr std::array<LowSmGolden, 11> h20_flash_golden {{
        {8,    128, 8,  8, false, false, false, true},
        {16,   128, 8,  8, false, false, false, true},
        {32,   128, 32, 8, false, false, false, true},
        {64,   128, 16, 8, false, false, false, true},
        {128,  128, 32, 8, false, false, false, true},
        {256,  256, 32, 4, false, false, false, false},
        {512,  256, 8,  5, true,  false, true,  false},
        {1024, 256, 16, 5, true,  false, false, false},
        {2048, 256, 16, 5, true,  true,  false, false},
        {4096, 256, 32, 5, true,  true,  false, false},
        {8192, 256, 32, 5, true,  true,  false, false},
    }};
    for (const auto& golden : h20_flash_golden) {
        const auto input = make_input(
            kH20Sms, 8, 32, golden.m, 6, 4096, 2048);
        check_low_sm_golden(input, select_mega_moe_sm90(input), golden);
    }

    constexpr std::array<LowSmGolden, 11> h20_mimo_golden {{
        {8,    128, 8,  8, false, false, false, true},
        {16,   128, 8,  8, false, false, false, true},
        {32,   128, 32, 8, false, false, false, true},
        {64,   128, 32, 8, false, false, false, true},
        {128,  256, 32, 5, true,  false, true,  false},
        {256,  256, 32, 4, true,  false, false, false},
        {512,  256, 8,  5, true,  false, true,  false},
        {1024, 256, 16, 5, true,  true,  false, false},
        {2048, 256, 32, 5, true,  true,  false, false},
        {4096, 256, 32, 5, true,  true,  false, false},
        {8192, 256, 32, 5, true,  true,  false, false},
    }};
    for (const auto& golden : h20_mimo_golden) {
        const auto input = make_input(
            kH20Sms, 8, 32, golden.m, 8, 4096, 2048);
        check_low_sm_golden(input, select_mega_moe_sm90(input), golden);
    }

    constexpr std::array<LowSmGolden, 3> h20_load96_boundary {{
        {380, 256, 32, 4, true, false, false, false},
        {384, 256, 8,  5, true, false, true,  false},
        {388, 256, 8,  5, true, false, true,  false},
    }};
    for (const auto& golden : h20_load96_boundary) {
        const auto input = make_input(
            kH20Sms, 8, 32, golden.m, 8, 4096, 2048);
        check_low_sm_golden(input, select_mega_moe_sm90(input), golden);
    }

    constexpr std::array<LowSmGolden, 11> h20_pro_golden {{
        {8,    128, 4,  7, false, false, false, true},
        {16,   128, 4,  7, false, false, false, true},
        {32,   128, 4,  7, false, false, false, true},
        {64,   128, 48, 7, false, false, false, true},
        {128,  128, 48, 7, false, false, false, true},
        {256,  256, 16, 4, false, false, false, false},
        {512,  256, 48, 5, true,  false, false, false},
        {1024, 256, 48, 5, true,  false, true,  false},
        {2048, 256, 48, 4, true,  false, false, false},
        {4096, 256, 48, 5, true,  false, false, false},
        {8192, 256, 48, 5, true,  false, false, false},
    }};
    for (const auto& golden : h20_pro_golden) {
        const auto input = make_input(
            kH20Sms, 8, 48, golden.m, 6, 7168, 3072);
        check_low_sm_golden(input, select_mega_moe_sm90(input), golden);
    }

    for (const int m : {1, 127, 128, 129, 255, 256, 257, 767, 768, 769}) {
        const auto input = make_input(kH200Sms, 3, 30, m, 7, 4608, 2304);
        const auto config = select_mega_moe_sm90(input);
        TEST_CHECK(is_sm90_moe_launch_config_legal(input, config));
        TEST_CHECK(config.l1.num_sms == kH200Sms and
                   config.l2.num_sms == kH200Sms);
    }
}
