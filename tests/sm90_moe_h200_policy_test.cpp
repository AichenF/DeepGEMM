#include <array>
#include <cassert>

#include "../csrc/jit_kernels/heuristics/sm90_moe_h200_policy.hpp"

using deep_gemm::Sm90MoeExpectedLoad;
using deep_gemm::Sm90MoeH200SchedulePolicy;
using deep_gemm::Sm90MoeH200ShapeBand;
using deep_gemm::is_sm90_moe_h200_schedule_legal;
using deep_gemm::select_sm90_moe_h200_policy;
using deep_gemm::select_sm90_moe_h200_schedule_policy;

constexpr int kH200Sms = 132;

static auto compact(const bool enabled, const int m) {
    return select_sm90_moe_h200_policy(
        enabled, kH200Sms, 8, 256, 32, m, 6, 4096, 2048);
}

static auto wide(const bool enabled, const int m) {
    return select_sm90_moe_h200_policy(
        enabled, kH200Sms, 8, 384, 48, m, 6, 7168, 3072);
}

static bool same_schedule(
    const Sm90MoeH200SchedulePolicy& lhs,
    const Sm90MoeH200SchedulePolicy& rhs) {
    return lhs.selected == rhs.selected and
           lhs.requires_bf16_scaled_accum == rhs.requires_bf16_scaled_accum and
           lhs.block_m == rhs.block_m and
           lhs.cluster_size == rhs.cluster_size and
           lhs.l1.block_n == rhs.l1.block_n and
           lhs.l1.block_k == rhs.l1.block_k and
           lhs.l1.experts_per_wave == rhs.l1.experts_per_wave and
           lhs.l1.num_stages == rhs.l1.num_stages and
           lhs.l1.num_sms == rhs.l1.num_sms and
           lhs.l1.nmajor == rhs.l1.nmajor and
           lhs.l2.block_n == rhs.l2.block_n and
           lhs.l2.block_k == rhs.l2.block_k and
           lhs.l2.experts_per_wave == rhs.l2.experts_per_wave and
           lhs.l2.num_stages == rhs.l2.num_stages and
           lhs.l2.num_sms == rhs.l2.num_sms and
           lhs.l2.nmajor == rhs.l2.nmajor and
           lhs.l2_execution.direct_scatter == rhs.l2_execution.direct_scatter and
           lhs.l2_execution.one_warp_cleanup == rhs.l2_execution.one_warp_cleanup;
}

struct ScheduleGolden {
    int m;
    bool selected;
    int l1_block_n;
    int l1_block_k;
    int l1_epw;
    int l2_epw;
    int l1_stages;
    int l2_stages;
    int num_sms;
    int cleanup;
    int l1_nmajor = -1;
};

static void check_golden(
    const Sm90MoeH200SchedulePolicy& schedule,
    const ScheduleGolden& golden) {
    assert(schedule.selected == golden.selected);
    if (not golden.selected)
        return;
    assert(schedule.block_m == 64);
    assert(schedule.cluster_size == 1);
    assert(schedule.l1.block_n == golden.l1_block_n);
    assert(schedule.l2.block_n == 256);
    assert(schedule.l1.block_k == golden.l1_block_k);
    assert(schedule.l2.block_k == 128);
    assert(schedule.l1.experts_per_wave == golden.l1_epw);
    assert(schedule.l2.experts_per_wave == golden.l2_epw);
    assert(schedule.l1.num_stages == golden.l1_stages);
    assert(schedule.l2.num_stages == golden.l2_stages);
    assert(schedule.l1.num_sms == golden.num_sms);
    assert(schedule.l2.num_sms == golden.num_sms);
    assert(schedule.l1.nmajor == golden.l1_nmajor);
    assert(schedule.l2_execution.direct_scatter == 0);
    assert(schedule.l2.nmajor == 1);
    assert(schedule.l2_execution.one_warp_cleanup == golden.cleanup);
}

int main() {
    // Non-H200 devices never receive an automatic H200 policy.
    assert(not compact(false, 128).recognized());
    assert(not compact(false, 128).enabled());
    assert(not wide(false, 8192).recognized());
    assert(not wide(false, 8192).enabled());

    // Shape classification is independent of M and model identity.
    for (const int m : {8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192}) {
        const auto compact_policy = compact(true, m);
        assert(compact_policy.recognized());
        assert(compact_policy.shape_band == Sm90MoeH200ShapeBand::CompactFfn);
        assert(compact_policy.numerical.selected);
        const bool compact_fp32_exception =
            m == 32 or m == 4096 or m == 8192;
        assert(compact_policy.numerical.bf16_scaled_accum ==
               (compact_fp32_exception ? 0 : 1));

        const auto wide_policy = wide(true, m);
        assert(wide_policy.recognized());
        assert(wide_policy.shape_band == Sm90MoeH200ShapeBand::WideFfn);
        assert(wide_policy.numerical.selected);
        assert(wide_policy.numerical.bf16_scaled_accum == (m == 64 ? 0 : 1));
    }

    // Golden schedule table: all 22 validated points keep the previous
    // effective choices. Values formerly inherited from the generic selector
    // are materialized here so H200 range rules remain self-contained.
    constexpr std::array<ScheduleGolden, 11> compact_golden {{
        {8, false, 0, 0, 0, 0, 0, 0, 0, -1},
        {16, false, 0, 0, 0, 0, 0, 0, 0, -1},
        {32, false, 0, 0, 0, 0, 0, 0, 0, -1},
        {64, false, 0, 0, 0, 0, 0, 0, 0, -1},
        {128, true, 256, 128, 4, 4, 3, 3, 0, 0},
        {256, false, 0, 0, 0, 0, 0, 0, 0, -1},
        {512, false, 0, 0, 0, 0, 0, 0, 0, -1},
        {1024, true, 256, 128, 32, 32, 4, 4, 0, 0},
        {2048, false, 0, 0, 0, 0, 0, 0, 0, -1},
        {4096, false, 0, 0, 0, 0, 0, 0, 0, -1},
        {8192, true, 256, 128, 32, 32, 3, 3, 0, 0},
    }};
    for (const auto& golden : compact_golden)
        check_golden(compact(true, golden.m).schedule, golden);

    constexpr std::array<ScheduleGolden, 11> wide_golden {{
        {8, false, 0, 0, 0, 0, 0, 0, 0, -1},
        {16, false, 0, 0, 0, 0, 0, 0, 0, -1},
        {32, false, 0, 0, 0, 0, 0, 0, 0, -1},
        {64, false, 0, 0, 0, 0, 0, 0, 0, -1},
        {128, true, 512, 128, 16, 16, 2, 3, 0, 0},
        {256, true, 256, 256, 8, 48, 2, 3, 128, 0},
        {512, true, 512, 128, 16, 16, 2, 3, 0, 0, 1},
        {1024, true, 512, 128, 16, 16, 2, 3, 0, 1, 1},
        {2048, true, 512, 128, 16, 16, 2, 3, 0, 1},
        {4096, true, 512, 128, 16, 16, 2, 3, 0, 1},
        {8192, true, 512, 128, 16, 16, 2, 3, 0, 1},
    }};
    for (const auto& golden : wide_golden)
        check_golden(wide(true, golden.m).schedule, golden);

    // The selector uses expected load, not top-k/local-expert identity.
    const auto compact_expected24_alt = select_sm90_moe_h200_policy(
        true, kH200Sms, 4, 128, 32, 96, 8, 4096, 2048);
    assert(same_schedule(compact(true, 128).schedule,
                         compact_expected24_alt.schedule));
    assert(not compact_expected24_alt.numerical.selected);

    const auto wide_expected16_alt = select_sm90_moe_h200_policy(
        true, kH200Sms, 4, 128, 32, 64, 8, 7168, 3072);
    assert(not wide_expected16_alt.numerical.selected);
    assert(not wide_expected16_alt.schedule.enabled());
    const auto wide_expected16_raw = select_sm90_moe_h200_schedule_policy(
        Sm90MoeH200ShapeBand::WideFfn, Sm90MoeExpectedLoad {512, 32},
        7168, 3072, 32, kH200Sms);
    assert(same_schedule(wide(true, 128).schedule, wide_expected16_raw));

    // Exact cross-multiplied boundaries: compact (12, 32].
    assert(not select_sm90_moe_h200_schedule_policy(
        Sm90MoeH200ShapeBand::CompactFfn, Sm90MoeExpectedLoad {120, 10},
        4096, 2048, 10, kH200Sms).enabled());
    assert(not select_sm90_moe_h200_schedule_policy(
        Sm90MoeH200ShapeBand::CompactFfn, Sm90MoeExpectedLoad {121, 10},
        4096, 2048, 10, kH200Sms).enabled()); // EPW4 is illegal for 10 experts.
    assert(select_sm90_moe_h200_schedule_policy(
        Sm90MoeH200ShapeBand::CompactFfn, Sm90MoeExpectedLoad {49, 4},
        4096, 2048, 4, kH200Sms).enabled());
    assert(select_sm90_moe_h200_schedule_policy(
        Sm90MoeH200ShapeBand::CompactFfn, Sm90MoeExpectedLoad {124, 4},
        4096, 2048, 4, kH200Sms).enabled());
    assert(select_sm90_moe_h200_schedule_policy(
        Sm90MoeH200ShapeBand::CompactFfn, Sm90MoeExpectedLoad {128, 4},
        4096, 2048, 4, kH200Sms).enabled());
    assert(not select_sm90_moe_h200_schedule_policy(
        Sm90MoeH200ShapeBand::CompactFfn, Sm90MoeExpectedLoad {129, 4},
        4096, 2048, 4, kH200Sms).enabled());

    // Compact medium and high buckets preserve open/closed endpoints.
    assert(not select_sm90_moe_h200_schedule_policy(
        Sm90MoeH200ShapeBand::CompactFfn, Sm90MoeExpectedLoad {4096, 32},
        4096, 2048, 32, kH200Sms).enabled());
    assert(select_sm90_moe_h200_schedule_policy(
        Sm90MoeH200ShapeBand::CompactFfn, Sm90MoeExpectedLoad {4097, 32},
        4096, 2048, 32, kH200Sms).enabled());
    assert(select_sm90_moe_h200_schedule_policy(
        Sm90MoeH200ShapeBand::CompactFfn, Sm90MoeExpectedLoad {8192, 32},
        4096, 2048, 32, kH200Sms).enabled());
    assert(not select_sm90_moe_h200_schedule_policy(
        Sm90MoeH200ShapeBand::CompactFfn, Sm90MoeExpectedLoad {8193, 32},
        4096, 2048, 32, kH200Sms).enabled());
    assert(not select_sm90_moe_h200_schedule_policy(
        Sm90MoeH200ShapeBand::CompactFfn, Sm90MoeExpectedLoad {4096, 4},
        4096, 2048, 4, kH200Sms).enabled());
    assert(select_sm90_moe_h200_schedule_policy(
        Sm90MoeH200ShapeBand::CompactFfn, Sm90MoeExpectedLoad {32769, 32},
        4096, 2048, 32, kH200Sms).enabled());

    // Wide boundaries switch configs at 24, enable the tuned N-major parent
    // in (48, 96], and retain N-major above 96.
    const auto wide_at24 = select_sm90_moe_h200_schedule_policy(
        Sm90MoeH200ShapeBand::WideFfn, Sm90MoeExpectedLoad {24, 1},
        7168, 3072, 48, kH200Sms);
    const auto wide_above24 = select_sm90_moe_h200_schedule_policy(
        Sm90MoeH200ShapeBand::WideFfn, Sm90MoeExpectedLoad {25, 1},
        7168, 3072, 48, kH200Sms);
    assert(wide_at24.enabled() and wide_at24.l1.block_n == 512);
    assert(wide_above24.enabled() and wide_above24.l1.block_k == 256);
    assert(select_sm90_moe_h200_schedule_policy(
        Sm90MoeH200ShapeBand::WideFfn, Sm90MoeExpectedLoad {48, 1},
        7168, 3072, 48, kH200Sms).enabled());
    const auto wide_above48 = select_sm90_moe_h200_schedule_policy(
        Sm90MoeH200ShapeBand::WideFfn, Sm90MoeExpectedLoad {49, 1},
        7168, 3072, 48, kH200Sms);
    assert(wide_above48.enabled() and wide_above48.l1.nmajor == 1);
    assert(wide_above48.requires_bf16_scaled_accum);
    const auto wide_at96 = select_sm90_moe_h200_schedule_policy(
        Sm90MoeH200ShapeBand::WideFfn, Sm90MoeExpectedLoad {96, 1},
        7168, 3072, 48, kH200Sms);
    assert(wide_at96.enabled() and wide_at96.l1.nmajor == 1);
    assert(wide_at96.requires_bf16_scaled_accum);
    const auto wide_above96 = select_sm90_moe_h200_schedule_policy(
        Sm90MoeH200ShapeBand::WideFfn, Sm90MoeExpectedLoad {97, 1},
        7168, 3072, 48, kH200Sms);
    assert(wide_above96.enabled() and wide_above96.l1.nmajor == 1);
    assert(wide_above96.requires_bf16_scaled_accum);
    const auto wide_above192 = select_sm90_moe_h200_schedule_policy(
        Sm90MoeH200ShapeBand::WideFfn, Sm90MoeExpectedLoad {193, 1},
        7168, 3072, 48, kH200Sms);
    assert(wide_above192.enabled() and wide_above192.l1.nmajor == -1);
    assert(wide_above192.requires_bf16_scaled_accum);

    // H/I bands use both dimensions and do not encode model names.
    assert(select_sm90_moe_h200_policy(
        true, kH200Sms, 1, 4, 4, 96, 1, 3072, 1536).recognized());
    assert(select_sm90_moe_h200_policy(
        true, kH200Sms, 1, 4, 4, 96, 1, 5119, 2559).recognized());
    assert(not select_sm90_moe_h200_policy(
        true, kH200Sms, 1, 4, 4, 96, 1, 5120, 2048).recognized());
    assert(not select_sm90_moe_h200_policy(
        true, kH200Sms, 1, 4, 4, 96, 1, 4096, 2560).recognized());
    assert(select_sm90_moe_h200_policy(
        true, kH200Sms, 1, 4, 4, 96, 1, 5120, 2560).recognized());
    assert(select_sm90_moe_h200_policy(
        true, kH200Sms, 1, 4, 4, 96, 1, 8192, 4096).recognized());
    assert(not select_sm90_moe_h200_policy(
        true, kH200Sms, 1, 4, 4, 96, 1, 8193, 4096).recognized());

    // Matching but illegal range requests fall back to the generic selector.
    const auto illegal_tile = select_sm90_moe_h200_policy(
        true, kH200Sms, 1, 48, 48, 128, 6, 5200, 2600);
    assert(illegal_tile.recognized());
    assert(not illegal_tile.schedule.enabled());
    const auto insufficient_sms = select_sm90_moe_h200_policy(
        true, 120, 8, 384, 48, 256, 6, 7168, 3072);
    assert(insufficient_sms.recognized());
    assert(not insufficient_sms.schedule.enabled());

    // Numerical policy keeps the exact old E5M2/BF16 validation matrix.
    for (const int m : {8, 16, 32, 64, 256, 512, 2048, 4096})
        assert(compact(true, m).numerical.fp8_combine == 0);
    assert(compact(true, 128).numerical.fp8_combine == 1);
    assert(compact(true, 1024).numerical.fp8_combine == 1);
    assert(compact(true, 8192).numerical.fp8_combine == 1);
    for (const int m : {8, 16, 32, 64, 512})
        assert(wide(true, m).numerical.fp8_combine == 0);
    for (const int m : {128, 256, 1024, 2048, 4096, 8192})
        assert(wide(true, m).numerical.fp8_combine == 1);

    // Compact M512 remains generic; wide M512 uses the validated N-major
    // schedule without broadening numerical policy.
    assert(not compact(true, 512).schedule.enabled());
    assert(compact(true, 512).numerical.bf16_scaled_accum == 1);
    assert(wide(true, 512).schedule.enabled());
    assert(wide(true, 512).schedule.l1.nmajor == 1);
    assert(wide(true, 512).numerical.bf16_scaled_accum == 1);
    const auto wide768 = wide(true, 768);
    assert(wide768.recognized());
    assert(not wide768.numerical.enabled());
    assert(not wide768.schedule.enabled());

    // Unvalidated M retains shape recognition. Range schedule and numerical
    // selection remain independent.
    const auto compact4097 = compact(true, 4097);
    assert(compact4097.recognized());
    assert(not compact4097.schedule.enabled());
    assert(not compact4097.numerical.enabled());
    const auto wide1025 = wide(true, 1025);
    assert(wide1025.recognized());
    assert(not wide1025.schedule.enabled());
    assert(not wide1025.numerical.enabled());
}
