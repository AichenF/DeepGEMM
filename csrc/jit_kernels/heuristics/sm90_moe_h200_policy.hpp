#pragma once

#include <cstdint>
#include <ostream>

namespace deep_gemm {

enum class Sm90MoeH200ShapeBand {
    None,
    CompactFfn,
    WideFfn
};

struct Sm90MoeExpectedLoad {
    int64_t routed_tokens = 0;
    int local_experts = 0;

    bool valid() const {
        return routed_tokens >= 0 and local_experts > 0;
    }

    bool greater_than(const int threshold) const {
        return valid() and
               routed_tokens > static_cast<int64_t>(threshold) * local_experts;
    }

    bool less_equal(const int threshold) const {
        return valid() and
               routed_tokens <= static_cast<int64_t>(threshold) * local_experts;
    }

    bool in_open_closed(const int low, const int high) const {
        return greater_than(low) and less_equal(high);
    }

    double value() const {
        return valid() ? static_cast<double>(routed_tokens) / local_experts : 0.0;
    }
};

inline Sm90MoeExpectedLoad make_sm90_moe_expected_load(
    const int num_tokens, const int num_topk, const int num_experts_per_rank) {
    if (num_tokens < 0 or num_topk <= 0 or num_experts_per_rank <= 0)
        return {};
    return {
        static_cast<int64_t>(num_tokens) * num_topk,
        num_experts_per_rank
    };
}

struct Sm90MoeH200PhaseSchedulePatch {
    // Zero means inherit the generic SM90 config.
    int block_n = 0;
    int block_k = 0;
    int experts_per_wave = 0;
    int num_stages = 0;
    int num_sms = 0;

    // -1 means inherit the generic phase schedule.
    int nmajor = -1;
};

struct Sm90MoeH200L2SchedulePatch {
    // -1 means inherit the generic SM90 config.
    int direct_scatter = -1;
    int one_warp_cleanup = -1;
};

struct Sm90MoeH200SchedulePolicy {
    bool selected = false;
    bool requires_bf16_scaled_accum = false;

    // The split implementation shares these values between L1 and L2.
    // Zero means inherit the generic SM90 config.
    int block_m = 0;
    int cluster_size = 0;

    Sm90MoeH200PhaseSchedulePatch l1;
    Sm90MoeH200PhaseSchedulePatch l2;
    Sm90MoeH200L2SchedulePatch l2_execution;

    bool enabled() const {
        return selected;
    }

    // Candidate generation needs one complete seed config before the split
    // phase patches are applied. Prefer L2 because all current H200 rules use
    // the retained L2 tile as that seed.
    int seed_block_n() const {
        return l2.block_n > 0 ? l2.block_n : l1.block_n;
    }

    int seed_block_k() const {
        return l2.block_k > 0 ? l2.block_k : l1.block_k;
    }
};

struct Sm90MoeH200NumericalPolicy {
    bool selected = false;

    // Numerical representation choices remain independent of schedule rules.
    int fp8_combine = 0;
    int bf16_scaled_accum = 0;

    bool enabled() const {
        return selected;
    }
};

struct Sm90MoeH200Policy {
    Sm90MoeH200ShapeBand shape_band = Sm90MoeH200ShapeBand::None;
    Sm90MoeExpectedLoad expected_load;
    Sm90MoeH200SchedulePolicy schedule;
    Sm90MoeH200NumericalPolicy numerical;

    bool recognized() const {
        return shape_band != Sm90MoeH200ShapeBand::None;
    }

    bool enabled() const {
        return schedule.enabled() or numerical.enabled();
    }

    friend std::ostream& operator << (std::ostream& os, const Sm90MoeH200Policy& policy) {
        const char* shape_band =
            policy.shape_band == Sm90MoeH200ShapeBand::CompactFfn ? "compact_ffn" :
            policy.shape_band == Sm90MoeH200ShapeBand::WideFfn ? "wide_ffn" : "none";
        const auto& schedule = policy.schedule;
        const auto& numerical = policy.numerical;
        os << "Sm90MoeH200Policy(shape_band=" << shape_band
           << ", expected_tokens_per_expert=" << policy.expected_load.value()
           << ", schedule_selected=" << schedule.selected
           << ", schedule_requires_bf16=" << schedule.requires_bf16_scaled_accum
           << ", schedule={block_m=" << schedule.block_m
           << ", cluster_size=" << schedule.cluster_size
           << ", l1={block_n=" << schedule.l1.block_n
           << ", block_k=" << schedule.l1.block_k
           << ", experts_per_wave=" << schedule.l1.experts_per_wave
           << ", num_stages=" << schedule.l1.num_stages
           << ", num_sms=" << schedule.l1.num_sms
           << ", nmajor=" << schedule.l1.nmajor
           << "}, l2={block_n=" << schedule.l2.block_n
           << ", block_k=" << schedule.l2.block_k
           << ", experts_per_wave=" << schedule.l2.experts_per_wave
           << ", num_stages=" << schedule.l2.num_stages
           << ", num_sms=" << schedule.l2.num_sms
           << ", nmajor=" << schedule.l2.nmajor
           << "}, l2_execution={direct_scatter=" << schedule.l2_execution.direct_scatter
           << ", one_warp_cleanup=" << schedule.l2_execution.one_warp_cleanup
           << "}}, numerical={selected=" << numerical.selected
           << ", fp8_combine=" << numerical.fp8_combine
           << ", bf16_scaled_accum=" << numerical.bf16_scaled_accum << "})";
        return os;
    }
};

inline Sm90MoeH200ShapeBand classify_sm90_moe_h200_shape_band(
    const bool h200_enabled, const int hidden, const int intermediate_hidden) {
    if (not h200_enabled)
        return Sm90MoeH200ShapeBand::None;

    if (hidden >= 3072 and hidden < 5120 and
        intermediate_hidden >= 1536 and intermediate_hidden < 2560)
        return Sm90MoeH200ShapeBand::CompactFfn;

    if (hidden >= 5120 and hidden <= 8192 and
        intermediate_hidden >= 2560 and intermediate_hidden <= 4096)
        return Sm90MoeH200ShapeBand::WideFfn;

    return Sm90MoeH200ShapeBand::None;
}

inline bool is_sm90_moe_h200_phase_patch_legal(
    const Sm90MoeH200PhaseSchedulePatch& phase,
    const int phase_n, const int hidden, const int intermediate_hidden,
    const int num_experts_per_rank, const int device_sms) {
    if (phase.block_n != 0 and
        (phase.block_n != 128 and phase.block_n != 256 and
         phase.block_n != 512))
        return false;
    if (phase.block_n > 0 and phase_n % phase.block_n != 0)
        return false;

    if (phase.block_k != 0 and phase.block_k != 128 and phase.block_k != 256)
        return false;
    // The current SM90 kernel template requires either phase's BLOCK_K to
    // divide both dimensions even though L1 and L2 launch independently.
    if (phase.block_k > 0 and
        (hidden % phase.block_k != 0 or intermediate_hidden % phase.block_k != 0))
        return false;

    if (phase.experts_per_wave > 0 and
        (phase.experts_per_wave > num_experts_per_rank or
         num_experts_per_rank % phase.experts_per_wave != 0))
        return false;
    if (phase.num_stages != 0 and phase.num_stages < 2)
        return false;
    if (phase.num_sms < 0 or phase.num_sms > device_sms)
        return false;
    if (phase.nmajor < -1 or phase.nmajor > 1)
        return false;
    return true;
}

inline bool is_sm90_moe_h200_schedule_legal(
    const Sm90MoeH200SchedulePolicy& schedule,
    const int hidden, const int intermediate_hidden,
    const int num_experts_per_rank, const int device_sms) {
    if (not schedule.enabled())
        return true;
    if (device_sms <= 0)
        return false;
    if (schedule.block_m != 0 and schedule.block_m != 64 and schedule.block_m != 128)
        return false;
    if (schedule.cluster_size != 0 and
        (schedule.cluster_size != 1 and schedule.cluster_size != 2))
        return false;
    if (schedule.cluster_size > 0 and device_sms % schedule.cluster_size != 0)
        return false;
    if (schedule.l2_execution.direct_scatter < -1 or
        schedule.l2_execution.direct_scatter > 1 or
        schedule.l2_execution.one_warp_cleanup < -1 or
        schedule.l2_execution.one_warp_cleanup > 1)
        return false;
    return is_sm90_moe_h200_phase_patch_legal(
               schedule.l1, 2 * intermediate_hidden, hidden, intermediate_hidden,
               num_experts_per_rank, device_sms) and
           is_sm90_moe_h200_phase_patch_legal(
               schedule.l2, hidden, hidden, intermediate_hidden,
               num_experts_per_rank, device_sms);
}

inline Sm90MoeH200SchedulePolicy select_sm90_moe_h200_schedule_policy(
    const Sm90MoeH200ShapeBand shape_band,
    const Sm90MoeExpectedLoad& expected_load,
    const int hidden, const int intermediate_hidden,
    const int num_experts_per_rank, const int device_sms) {
    Sm90MoeH200SchedulePolicy schedule;
    if (shape_band == Sm90MoeH200ShapeBand::None or not expected_load.valid())
        return schedule;

    const auto set_common = [&]() {
        schedule.selected = true;
        schedule.block_m = 64;
        schedule.cluster_size = 1;
        schedule.l1.block_n = schedule.l2.block_n = 256;
        schedule.l1.block_k = schedule.l2.block_k = 128;
        schedule.l2_execution.direct_scatter = 0;
        schedule.l2.nmajor = 1;
    };

    if (shape_band == Sm90MoeH200ShapeBand::CompactFfn and
        expected_load.in_open_closed(12, 32)) {
        set_common();
        schedule.l2_execution.one_warp_cleanup = 0;
        schedule.l1.experts_per_wave = schedule.l2.experts_per_wave = 4;
        schedule.l1.num_stages = schedule.l2.num_stages = 3;
    } else if (shape_band == Sm90MoeH200ShapeBand::CompactFfn and
               expected_load.in_open_closed(128, 256)) {
        set_common();
        schedule.l2_execution.one_warp_cleanup = 0;
        schedule.l1.experts_per_wave = schedule.l2.experts_per_wave = 32;
        schedule.l1.num_stages = schedule.l2.num_stages = 4;
    } else if (shape_band == Sm90MoeH200ShapeBand::CompactFfn and
               expected_load.greater_than(1024)) {
        set_common();
        schedule.l2_execution.one_warp_cleanup = 0;
        schedule.l1.experts_per_wave = schedule.l2.experts_per_wave = 32;
        schedule.l1.num_stages = schedule.l2.num_stages = 3;
    } else if (shape_band == Sm90MoeH200ShapeBand::WideFfn and
               expected_load.in_open_closed(8, 24)) {
        set_common();
        schedule.l2_execution.one_warp_cleanup = 0;
        schedule.l1.block_n = 512;
        schedule.l1.experts_per_wave = schedule.l2.experts_per_wave = 16;
        schedule.l1.num_stages = 2;
        schedule.l2.num_stages = 3;
        schedule.requires_bf16_scaled_accum = true;
    } else if (shape_band == Sm90MoeH200ShapeBand::WideFfn and
               expected_load.in_open_closed(24, 48)) {
        set_common();
        schedule.l2_execution.one_warp_cleanup = 0;
        schedule.l1.block_k = 256;
        schedule.l1.experts_per_wave = 8;
        schedule.l2.experts_per_wave = 48;
        schedule.l1.num_stages = 2;
        schedule.l2.num_stages = 3;
        schedule.l1.num_sms = schedule.l2.num_sms = 128;
    } else if (shape_band == Sm90MoeH200ShapeBand::WideFfn and
               expected_load.in_open_closed(48, 96)) {
        set_common();
        schedule.l2_execution.one_warp_cleanup = 0;
        schedule.l1.block_n = 512;
        schedule.l1.experts_per_wave = schedule.l2.experts_per_wave = 16;
        schedule.l1.num_stages = 2;
        schedule.l2.num_stages = 3;
        schedule.l1.nmajor = 1;
        schedule.requires_bf16_scaled_accum = true;
    } else if (shape_band == Sm90MoeH200ShapeBand::WideFfn and
               expected_load.in_open_closed(96, 192)) {
        set_common();
        schedule.l2_execution.one_warp_cleanup = 1;
        schedule.l1.block_n = 512;
        schedule.l1.experts_per_wave = schedule.l2.experts_per_wave = 16;
        schedule.l1.num_stages = 2;
        schedule.l2.num_stages = 3;
        schedule.l1.nmajor = 1;
        schedule.requires_bf16_scaled_accum = true;
    } else if (shape_band == Sm90MoeH200ShapeBand::WideFfn and
               expected_load.greater_than(192)) {
        set_common();
        schedule.l2_execution.one_warp_cleanup = 1;
        schedule.l1.block_n = 512;
        schedule.l1.experts_per_wave = schedule.l2.experts_per_wave = 16;
        schedule.l1.num_stages = 2;
        schedule.l2.num_stages = 3;
        schedule.requires_bf16_scaled_accum = true;
    }

    // Range rules deliberately fall back to the generic selector when an
    // otherwise matching shape cannot instantiate the requested kernel.
    if (not is_sm90_moe_h200_schedule_legal(
            schedule, hidden, intermediate_hidden,
            num_experts_per_rank, device_sms))
        return {};
    return schedule;
}

inline bool is_sm90_moe_h200_validated_m(const int num_tokens) {
    return num_tokens == 8 or num_tokens == 16 or num_tokens == 32 or
           num_tokens == 64 or num_tokens == 128 or num_tokens == 256 or
           num_tokens == 512 or num_tokens == 1024 or num_tokens == 2048 or
           num_tokens == 4096 or num_tokens == 8192;
}

inline Sm90MoeH200NumericalPolicy select_sm90_moe_h200_numerical_policy(
    const bool h200_enabled,
    const int num_ranks, const int num_experts, const int num_experts_per_rank,
    const int num_tokens, const int num_topk,
    const int hidden, const int intermediate_hidden) {
    Sm90MoeH200NumericalPolicy numerical;
    if (not h200_enabled or not is_sm90_moe_h200_validated_m(num_tokens))
        return numerical;

    // Numerical coverage stays on the exact previously validated topology.
    // These predicates are intentionally independent of the H/I range-based
    // schedule selector.
    const bool compact_validated_shape =
        num_ranks == 8 and num_experts == 256 and num_experts_per_rank == 32 and
        num_topk == 6 and hidden == 4096 and intermediate_hidden == 2048;
    const bool wide_validated_shape =
        num_ranks == 8 and num_experts == 384 and num_experts_per_rank == 48 and
        num_topk == 6 and hidden == 7168 and intermediate_hidden == 3072;
    if (not compact_validated_shape and not wide_validated_shape)
        return numerical;
    numerical.selected = true;

    numerical.fp8_combine =
        (compact_validated_shape and
         (num_tokens == 128 or num_tokens == 1024 or num_tokens == 8192)) or
        (wide_validated_shape and
         (num_tokens == 128 or num_tokens == 256 or num_tokens == 1024 or
          num_tokens == 2048 or num_tokens == 4096 or num_tokens == 8192));

    const bool compact_fp32_exception =
        compact_validated_shape and
        (num_tokens == 32 or num_tokens == 4096 or num_tokens == 8192);
    const bool wide_fp32_exception =
        wide_validated_shape and num_tokens == 64;
    numerical.bf16_scaled_accum =
        not (compact_fp32_exception or wide_fp32_exception);
    return numerical;
}

// Pure policy composition is independent of CUDA state, so shape/load range
// matching, topology-independent schedule selection, fallback, and the exact
// numerical gate are CPU-testable.
inline Sm90MoeH200Policy select_sm90_moe_h200_policy(
    const bool h200_enabled, const int device_sms,
    const int num_ranks, const int num_experts, const int num_experts_per_rank,
    const int num_tokens, const int num_topk,
    const int hidden, const int intermediate_hidden) {
    Sm90MoeH200Policy policy;
    policy.shape_band = classify_sm90_moe_h200_shape_band(
        h200_enabled, hidden, intermediate_hidden);
    policy.expected_load = make_sm90_moe_expected_load(
        num_tokens, num_topk, num_experts_per_rank);
    policy.numerical = select_sm90_moe_h200_numerical_policy(
        h200_enabled,
        num_ranks, num_experts, num_experts_per_rank,
        num_tokens, num_topk, hidden, intermediate_hidden);
    policy.schedule = select_sm90_moe_h200_schedule_policy(
        policy.shape_band, policy.expected_load,
        hidden, intermediate_hidden, num_experts_per_rank, device_sms);
    if (policy.schedule.requires_bf16_scaled_accum and
        not policy.numerical.bf16_scaled_accum)
        policy.schedule = {};
    return policy;
}

} // namespace deep_gemm
