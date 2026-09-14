#pragma once

#include <array>
#include <cstdint>
#include <iostream>
#include <string>
#include <tuple>
#include <unordered_set>
#include <utility>

#include <deep_gemm/layout/mega_moe.cuh>

#include "../../utils/exception.hpp"
#include "../../utils/system.hpp"
#include "mega_moe.hpp"
#include "sm90.hpp"

namespace deep_gemm {

// The framework and API have already selected the BN128 split family before
// this selector runs. This header chooses only that family's concrete launch
// schedule; it never changes the weight layout or selects the fused family.

struct SM90NVFP4MegaMoEConfig {
    static constexpr int kBlockM = 128;
    static constexpr int kBlockN = 128;
    static constexpr int kBlockK = 128;
    static constexpr int kWeightStoragePerKBlock = 80;
    static constexpr int kSwizzleActsMode = 128;
    static constexpr int kL1ClusterSize = 2;
    static constexpr int kL2ClusterSize = 1;
    static constexpr int kL1NumDispatchThreads = 128;
    static constexpr int kL2NumDispatchThreads = 0;
    static constexpr int kL1NumActiveDispatchWarps = 2;
    static constexpr int kL2NumActiveDispatchWarps = 0;
    static constexpr int kL1ActsSFReservationGranK = 64;
    static constexpr int kL2ActsSFReservationGranK = 128;
    static constexpr int kNumNonEpilogueThreads = 128;
    static constexpr int kNumEpilogueThreads = 256;

    int cluster_size;

    int num_max_pool_tokens;
    int num_padded_sf_pool_tokens;

    int num_experts_per_wave;

    int num_stages, smem_size;

    int num_dispatch_threads;

    friend std::ostream& operator << (
            std::ostream& os,
            const SM90NVFP4MegaMoEConfig& config) {
        os << "SM90NVFP4MegaMoEConfig("
           << "block_m=" << kBlockM
           << ", block_n=" << kBlockN
           << ", block_k=" << kBlockK
           << ", cluster_size=" << config.cluster_size
           << ", num_max_pool_tokens=" << config.num_max_pool_tokens
           << ", num_padded_sf_pool_tokens="
           << config.num_padded_sf_pool_tokens
           << ", swizzle_acts_mode=" << kSwizzleActsMode
           << ", num_experts_per_wave=" << config.num_experts_per_wave
           << ", num_stages=" << config.num_stages
           << ", smem_size=" << config.smem_size
           << ", num_dispatch_threads=" << config.num_dispatch_threads
           << ", num_non_epilogue_threads="
           << kNumNonEpilogueThreads
           << ", num_epilogue_threads=" << kNumEpilogueThreads
           << ")";
        return os;
    }
};

struct SM90NVFP4MegaMoEInput {
    int launch_num_sms;

    int num_ranks;
    int num_experts;
    int num_experts_per_rank;
    int num_max_tokens_per_rank;
    int num_tokens;
    int num_topk;
    int hidden;
    int intermediate_hidden;
    int num_padded_sf_pool_tokens;
};

struct SM90NVFP4MegaMoELoad {
    int64_t routed_tokens;
    int64_t local_experts;
    float expected_tokens_per_local_expert;

    bool valid() const {
        return routed_tokens > 0 && local_experts > 0;
    }

    bool greater_equal(const int64_t value) const {
        return routed_tokens >= value * local_experts;
    }
};

enum class SM90NVFP4MegaMoEPhase {
    L1,
    L2,
};

struct SM90NVFP4MegaMoEPlan {
    SM90NVFP4MegaMoEConfig l1_config;
    SM90NVFP4MegaMoEConfig l2_config;
    bool dispatch_dequant;
    bool l2_arrival_counter;
};

static SM90NVFP4MegaMoELoad get_sm90_nvfp4_mega_moe_load(
        const SM90NVFP4MegaMoEInput& input) {
    const int64_t routed_tokens =
        static_cast<int64_t>(input.num_tokens) * input.num_topk;
    const SM90NVFP4MegaMoELoad load {
        routed_tokens,
        input.num_experts_per_rank,
        static_cast<float>(routed_tokens) / input.num_experts_per_rank,
    };
    DG_HOST_ASSERT(load.valid());
    return load;
}

static int get_num_experts_per_wave_for_sm90_nvfp4_mega_moe(
        const SM90NVFP4MegaMoEInput& input) {
    return get_num_experts_per_wave_for_mega_moe(
        input.num_experts_per_rank,
        input.num_tokens,
        input.num_topk,
        input.intermediate_hidden,
        SM90NVFP4MegaMoEConfig::kBlockM,
        SM90NVFP4MegaMoEConfig::kBlockN,
        input.launch_num_sms);
}

static std::pair<int, int>
get_sm90_nvfp4_mega_moe_pipeline_config(
        const SM90NVFP4MegaMoEInput& input,
        const SM90NVFP4MegaMoELoad& load,
        const SM90NVFP4MegaMoEPhase phase) {
    const auto align = [](int value, int alignment) {
        return ((value + alignment - 1) / alignment) * alignment;
    };
    constexpr int kSmemAlignment = 1024;
    const bool is_l2 = phase == SM90NVFP4MegaMoEPhase::L2;
    const int num_dispatch_threads = is_l2 ?
        SM90NVFP4MegaMoEConfig::kL2NumDispatchThreads :
        SM90NVFP4MegaMoEConfig::kL1NumDispatchThreads;
    const int num_active_dispatch_warps = is_l2 ?
        SM90NVFP4MegaMoEConfig::kL2NumActiveDispatchWarps :
        SM90NVFP4MegaMoEConfig::kL1NumActiveDispatchWarps;
    // L1 retains only the proven per-64 physical reservation. Kernel
    // activation-scale semantics remain per-128.
    const int acts_sf_gran_k = is_l2 ?
        SM90NVFP4MegaMoEConfig::kL2ActsSFReservationGranK :
        SM90NVFP4MegaMoEConfig::kL1ActsSFReservationGranK;

    DG_HOST_ASSERT(
        num_active_dispatch_warps <= num_dispatch_threads / 32);
    const int num_dispatch_warps = num_active_dispatch_warps;
    constexpr int kNumEpilogueWarps = 8;
    constexpr int kNumEpilogueWarpgroups = 2;
    constexpr int kWGBlockM = 64;
    constexpr int kWGL1OutBlockN = 64;

    const int smem_expert_count_size = align(
        input.num_experts * static_cast<int>(sizeof(uint32_t)),
        kSmemAlignment);
    const int smem_send_buffers_size = align(
        static_cast<int>(layout::Buffer(
            layout::Data(input.hidden),
            num_dispatch_warps,
            1).get_num_bytes()),
        kSmemAlignment);
    const int smem_dispatch_size =
        smem_expert_count_size + smem_send_buffers_size;
    const int smem_nvfp4_lut = align(128 * 8, kSmemAlignment);

    const int smem_cd_l1 =
        kNumEpilogueWarpgroups * kWGBlockM * kWGL1OutBlockN;
    // L2 stages its remote scatter through one warp-private 8-row half tile so
    // that each destination row leaves as a single contiguous 16-byte-wide
    // burst. Must match `kL2StageRows` / `kL2StageRowStride` in the kernel.
    constexpr int kWGBlockN = 128;
    constexpr int kL2StageRows = 8;
    constexpr int kL2StageRowPad = 8;
    const int smem_cd_l2 = kNumEpilogueWarps * kL2StageRows *
        (kWGBlockN + kL2StageRowPad) * static_cast<int>(sizeof(uint16_t));
    const int smem_cd = align(
        is_l2 ? smem_cd_l2 : smem_cd_l1,
        kSmemAlignment);

    const int num_sfa_groups_per_bk =
        SM90NVFP4MegaMoEConfig::kBlockK / acts_sf_gran_k;
    const int smem_sfa_per_stage = align(
        num_sfa_groups_per_bk * SM90NVFP4MegaMoEConfig::kBlockM *
            static_cast<int>(sizeof(float)),
        128);
    const int smem_per_stage =
        SM90NVFP4MegaMoEConfig::kBlockM *
            SM90NVFP4MegaMoEConfig::kBlockK +
        SM90NVFP4MegaMoEConfig::kBlockN *
            SM90NVFP4MegaMoEConfig::kBlockK +
        smem_sfa_per_stage;
    const int smem_barriers_fixed =
        (num_dispatch_warps + 2 * kNumEpilogueWarps) * 8;
    // Both phases own full, empty, and dequant barriers. Production L1 is
    // always half-stream mode4 and therefore owns one additional K[64:128]
    // publication mbarrier per stage. Account for it while selecting stages,
    // rather than appending bytes after a max-stage decision.
    const int smem_barriers_per_stage = (is_l2 ? 3 : 4) * 8;
    const int smem_fixed =
        smem_dispatch_size +
        smem_nvfp4_lut +
        smem_cd +
        smem_barriers_fixed;
    const int max_num_stages =
        (SM90ArchSpec::smem_capacity - smem_fixed) /
        (smem_per_stage + smem_barriers_per_stage);
    const int num_stages =
        !is_l2 &&
        load.expected_tokens_per_local_expert > 8.0f &&
        max_num_stages > 6 ?
        6 : max_num_stages;
    DG_HOST_ASSERT(max_num_stages >= 2);
    DG_HOST_ASSERT(num_stages >= 2 && num_stages <= max_num_stages);
    return {
        num_stages,
        smem_fixed +
            num_stages *
            (smem_per_stage + smem_barriers_per_stage),
    };
}

static SM90NVFP4MegaMoEConfig materialize_sm90_nvfp4_mega_moe_phase(
        const SM90NVFP4MegaMoEInput& input,
        const SM90NVFP4MegaMoELoad& load,
        const SM90NVFP4MegaMoEPhase phase) {
    const bool is_l2 = phase == SM90NVFP4MegaMoEPhase::L2;
    SM90NVFP4MegaMoEConfig config {
        is_l2 ?
            SM90NVFP4MegaMoEConfig::kL2ClusterSize :
            SM90NVFP4MegaMoEConfig::kL1ClusterSize,
        layout::get_num_max_pool_tokens(
            input.num_ranks,
            input.num_max_tokens_per_rank,
            input.num_topk,
            input.num_experts_per_rank),
        input.num_padded_sf_pool_tokens,
        get_num_experts_per_wave_for_sm90_nvfp4_mega_moe(
            input),
        0,
        0,
        is_l2 ?
            SM90NVFP4MegaMoEConfig::kL2NumDispatchThreads :
            SM90NVFP4MegaMoEConfig::kL1NumDispatchThreads,
    };
    std::tie(config.num_stages, config.smem_size) =
        get_sm90_nvfp4_mega_moe_pipeline_config(
            input,
            load,
            phase);
    return config;
}

static bool is_sm90_nvfp4_mega_moe_plan_legal(
        const SM90NVFP4MegaMoEInput& input,
        const SM90NVFP4MegaMoEPlan& plan) {
    const auto valid_phase = [&](const SM90NVFP4MegaMoEConfig& config) {
        return config.num_experts_per_wave > 0 &&
            config.num_experts_per_wave <= input.num_experts_per_rank &&
            input.num_experts_per_rank % config.num_experts_per_wave == 0 &&
            config.num_stages >= 2 &&
            config.smem_size > 0 &&
            config.smem_size <= SM90ArchSpec::smem_capacity;
    };
    return valid_phase(plan.l1_config) &&
        valid_phase(plan.l2_config) &&
        plan.l1_config.cluster_size ==
            SM90NVFP4MegaMoEConfig::kL1ClusterSize &&
        plan.l1_config.num_dispatch_threads ==
            SM90NVFP4MegaMoEConfig::kL1NumDispatchThreads &&
        plan.l2_config.cluster_size ==
            SM90NVFP4MegaMoEConfig::kL2ClusterSize &&
        plan.l2_config.num_dispatch_threads ==
            SM90NVFP4MegaMoEConfig::kL2NumDispatchThreads;
}

static SM90NVFP4MegaMoEPlan select_sm90_nvfp4_split_mega_moe(
        const SM90NVFP4MegaMoEInput& input) {
    DG_HOST_ASSERT(input.launch_num_sms > 0);
    DG_HOST_ASSERT(input.num_ranks > 0);
    DG_HOST_ASSERT(input.num_experts_per_rank > 0);
    DG_HOST_ASSERT(
        input.num_experts ==
        input.num_experts_per_rank * input.num_ranks);
    DG_HOST_ASSERT(input.num_max_tokens_per_rank > 0);
    DG_HOST_ASSERT(
        input.num_tokens > 0 &&
        input.num_tokens <= input.num_max_tokens_per_rank);
    DG_HOST_ASSERT(input.num_topk > 0);
    DG_HOST_ASSERT(input.hidden > 0 && input.hidden % 128 == 0);
    DG_HOST_ASSERT(
        input.intermediate_hidden > 0 &&
        input.intermediate_hidden % 128 == 0);
    DG_HOST_ASSERT(input.num_padded_sf_pool_tokens > 0);

    const auto load = get_sm90_nvfp4_mega_moe_load(input);
    const SM90NVFP4MegaMoEPlan plan {
        materialize_sm90_nvfp4_mega_moe_phase(
            input, load, SM90NVFP4MegaMoEPhase::L1),
        materialize_sm90_nvfp4_mega_moe_phase(
            input, load, SM90NVFP4MegaMoEPhase::L2),
        // Every production BN128 split uses the accepted L1 mode4+remap
        // implementation. That implementation requires the independent odd-K
        // dispatch dequant team, including in the former rho<256 interval.
        true,
        load.expected_tokens_per_local_expert <= 32.0f ||
            load.expected_tokens_per_local_expert >= 128.0f,
    };
    DG_HOST_ASSERT(is_sm90_nvfp4_mega_moe_plan_legal(input, plan));

    if (get_env<int>("DG_JIT_DEBUG") ||
        get_env<int>("DG_PRINT_CONFIGS")) {
        const auto key = fmt::format(
            "SM90NVFP4MegaMoEPlan(num_ranks={}, num_experts={}, "
            "hidden={}, intermediate_hidden={}, "
            "num_max_tokens_per_rank={}, num_tokens={}, num_topk={}, "
            "layout_block_n=128)",
            input.num_ranks,
            input.num_experts,
            input.hidden,
            input.intermediate_hidden,
            input.num_max_tokens_per_rank,
            input.num_tokens,
            input.num_topk);
        static std::unordered_set<std::string> printed;
        if (printed.count(key) == 0) {
            std::cout << key
                      << ": l1=" << plan.l1_config
                      << ", l2=" << plan.l2_config
                      << std::endl;
            printed.insert(key);
        }
    }
    return plan;
}

// Exact-key static-RS small-M implementation.  This is a separate physical
// schedule from the dev-m dynamic fused kernel, but its H20/H200 selection
// policy and material configuration live in this one SM90 heuristic file.
struct SM90NVFP4SmallMConfig {
    static constexpr int kBlockN = 256;
    static constexpr int kBlockK = 128;
    static constexpr int kWeightStoragePerKBlock = 80;
    static constexpr int kSwizzleActsMode = 128;
    static constexpr int kClusterSize = 1;
    static constexpr int kNumDispatchThreads = 64;
    static constexpr int kNumNonEpilogueThreads = 64;
    static constexpr int kNumEpilogueThreads = 256;
    static constexpr int kNumThreads =
        kNumDispatchThreads + kNumNonEpilogueThreads + kNumEpilogueThreads;

    int block_m;
    int num_max_pool_tokens;
    int num_padded_sf_pool_tokens;
    int num_experts_per_wave;
    int num_stages;
    int smem_size;
};

struct SM90NVFP4SmallMInput {
    int num_sms;
    int num_ranks;
    int num_experts;
    int num_experts_per_rank;
    int num_max_tokens_per_rank;
    int num_tokens;
    int num_topk;
    int hidden;
    int intermediate_hidden;
    int num_padded_sf_pool_tokens;
};

struct SM90NVFP4SmallMPlan {
    SM90NVFP4SmallMConfig config;
    bool swap_ab;
    bool rs_swap_ab;
    bool rs_batch4;
    bool rs_vector_scale;
    bool rs_direct_lut;
    bool rs_compact_smem;
    bool use_mode2_lop3_decoder;
    bool single_active_dispatch_warp;
};

static bool is_sm90_nvfp4_small_m_plan_legal(
        const SM90NVFP4SmallMInput& input,
        const SM90NVFP4SmallMPlan& plan) {
    const auto& config = plan.config;
    const bool supported_block_m =
        config.block_m == 8 || config.block_m == 16 ||
        config.block_m == 24;
    return (input.num_sms == 78 || input.num_sms == 132) &&
        input.num_ranks == 8 &&
        input.num_experts == input.num_experts_per_rank * input.num_ranks &&
        input.num_tokens > 0 &&
        input.num_tokens <= input.num_max_tokens_per_rank &&
        input.num_topk > 0 && input.num_topk <= 32 &&
        input.hidden % SM90NVFP4SmallMConfig::kBlockN == 0 &&
        (2 * input.intermediate_hidden) %
            SM90NVFP4SmallMConfig::kBlockN == 0 &&
        input.hidden % SM90NVFP4SmallMConfig::kBlockK == 0 &&
        input.intermediate_hidden %
            SM90NVFP4SmallMConfig::kBlockK == 0 &&
        input.num_padded_sf_pool_tokens > 0 &&
        supported_block_m &&
        config.num_experts_per_wave > 0 &&
        config.num_experts_per_wave <= input.num_experts_per_rank &&
        input.num_experts_per_rank % config.num_experts_per_wave == 0 &&
        config.num_stages >= 3 && config.num_stages <= 4 &&
        config.smem_size > 0 &&
        config.smem_size <= SM90ArchSpec::smem_capacity;
}

// Materialize only the exact H20/H200 KF424 buckets admitted by the all-M
// table below.  Architecture is part of the key; no neighboring M or SKU
// inherits one of these schedules.
static SM90NVFP4SmallMPlan select_sm90_nvfp4_small_m_kf424(
        const SM90NVFP4SmallMInput& input) {
    const bool is_h20 = input.num_sms == 78;
    const bool is_h200 = input.num_sms == 132;
    const bool is_flash =
        input.num_ranks == 8 && input.num_experts == 256 &&
        input.num_experts_per_rank == 32 && input.num_topk == 6 &&
        input.hidden == 4096 && input.intermediate_hidden == 2048;
    const bool is_pro =
        input.num_ranks == 8 && input.num_experts == 384 &&
        input.num_experts_per_rank == 48 && input.num_topk == 6 &&
        input.hidden == 7168 && input.intermediate_hidden == 3072;

    int block_m = 0;
    int num_experts_per_wave = 0;
    int num_stages = 0;
    int smem_size = 0;
    bool single_active_dispatch_warp = false;
    bool compact_smem = false;

    if (is_h200 && is_flash && input.num_tokens == 16) {
        block_m = 8;
        num_experts_per_wave = 32;
        num_stages = 4;
        smem_size = 229120;
        single_active_dispatch_warp = true;
    } else if ((is_h20 || is_h200) && is_flash &&
               input.num_tokens == 32) {
        block_m = 8;
        num_experts_per_wave = 32;
        num_stages = 3;
        smem_size = 186112;
    } else if (is_h200 && is_flash && input.num_tokens == 64) {
        block_m = 16;
        num_experts_per_wave = 32;
        num_stages = 3;
        smem_size = 189184;
        single_active_dispatch_warp = true;
    } else if ((is_h20 || is_h200) && is_pro &&
               input.num_tokens == 128) {
        block_m = 24;
        num_experts_per_wave = 48;
        num_stages = 4;
        smem_size = 193280;
        single_active_dispatch_warp = true;
        compact_smem = true;
    } else if (is_h20 && is_pro && input.num_tokens == 8) {
        block_m = 8;
        num_experts_per_wave = 48;
        num_stages = 3;
        smem_size = 178944;
        single_active_dispatch_warp = true;
    } else if (is_h20 && is_pro && input.num_tokens == 32) {
        block_m = 8;
        num_experts_per_wave = 48;
        num_stages = 3;
        smem_size = 193280;
    } else {
        DG_HOST_UNREACHABLE(
            "Point is not an admitted H20/H200 KF424 bucket");
    }

    const SM90NVFP4SmallMPlan plan {
        {
            block_m,
            layout::get_num_max_pool_tokens(
                input.num_ranks,
                input.num_max_tokens_per_rank,
                input.num_topk,
                input.num_experts_per_rank),
            input.num_padded_sf_pool_tokens,
            num_experts_per_wave,
            num_stages,
            smem_size,
        },
        true,
        true,
        true,
        true,
        false,
        compact_smem,
        true,
        single_active_dispatch_warp,
    };
    DG_HOST_ASSERT(is_sm90_nvfp4_small_m_plan_legal(input, plan));
    return plan;
}

// Common dev-m dynamic fused kernel.  H20 and H200 have the same SM90
// instruction, register, and shared-memory surface; num_sms changes the
// persistent grid and the exact small-M bucket selected below.
static constexpr int kSM90NVFP4BStoragePerKBlock = 80;

struct SM90NVFP4FusedConfig {
    static constexpr int kBlockK = 128;
    static constexpr int kSwizzleActsMode = 128;
    static constexpr int kNumDispatchThreads = 64;
    static constexpr int kNumNonEpilogueThreads = 64;
    static constexpr int kNumEpilogueThreads = 256;
    static constexpr int kNumThreads =
        kNumDispatchThreads + kNumNonEpilogueThreads + kNumEpilogueThreads;

    int block_m, block_n;
    int num_max_pool_tokens;
    int num_padded_sf_pool_tokens;
    int num_experts_per_wave;
    int num_stages, smem_size;
};

struct SM90NVFP4FusedShape {
    static constexpr int kH20NumSMs = 78;
    static constexpr int kH200NumSMs = 132;
    static constexpr int kNumRanks = 8;

    int num_sms;
    int num_ranks;
    int num_experts;
    int num_topk;
    int hidden;
    int intermediate_hidden;

    static constexpr bool is_supported_batch(const int num_tokens) noexcept {
        return num_tokens > 0;
    }

    constexpr bool is_supported_shape() const noexcept {
        if ((num_sms != kH20NumSMs && num_sms != kH200NumSMs) ||
            num_ranks != kNumRanks)
            return false;
        const bool flash =
            num_experts == 256 && num_topk == 6 &&
            hidden == 4096 && intermediate_hidden == 2048;
        const bool pro =
            num_experts == 384 && num_topk == 6 &&
            hidden == 7168 && intermediate_hidden == 3072;
        const bool mimo =
            num_experts == 384 && num_topk == 8 &&
            hidden == 6144 && intermediate_hidden == 2048;
        return flash || pro || mimo;
    }
};

struct SM90NVFP4FusedInput {
    int num_sms;
    int num_ranks, num_experts, num_experts_per_rank;
    int num_max_tokens_per_rank, num_tokens, num_topk;
    int hidden, intermediate_hidden;
    int num_padded_sf_pool_tokens;

    SM90NVFP4FusedShape shape() const noexcept {
        return {
            num_sms, num_ranks, num_experts, num_topk,
            hidden, intermediate_hidden};
    }
};

struct SM90NVFP4FusedPlan {
    SM90NVFP4FusedConfig config;
    bool swap_ab;
    bool use_mode2_row_decoder;
    bool single_active_dispatch_warp;
    bool use_interleaved_scheduler;
};

struct SM90NVFP4FusedBucket {
    int hidden;
    int min_tokens;
    int max_tokens;
    int block_m;
    int num_experts_per_wave;
    int num_stages;
    bool swap_ab;
    bool use_mode2_row_decoder;
    bool single_active_dispatch_warp;
};

// H20's range-keyed small-M physical buckets.  These continuous ranges were
// selected from physical H20 crossover measurements; the arm table
// below sends every point outside them to the unmodified dev-m scheduler.
// H200 never reads this table.
static constexpr std::array<SM90NVFP4FusedBucket, 12>
kSM90NVFP4H20FusedBuckets {{
    {4096,  1,   8,  8, 16, 6, true, true, true},
    {4096,  9,  16,  8, 32, 6, true, true, true},
    {4096, 17,  64, 24, 32, 6, true, true, true},
    {4096, 65, 128, 24, 16, 6, true, true, true},
    {7168,  1,   8,  8, 48, 6, true, true, true},
    {7168,  9,  16,  8, 24, 6, true, true, true},
    {7168, 17,  64, 24, 48, 6, true, true, true},
    {7168, 65, 191, 24, 16, 6, true, true, true},
    {6144,  1,   8,  8, 48, 6, true, true, true},
    {6144,  9,  16,  8, 24, 6, true, true, true},
    {6144, 17,  64, 24, 48, 6, true, true, true},
    {6144, 65, 144, 24, 16, 6, true, true, true},
}};

// H200 physical buckets for the measured optimized ranges.  A missing range
// intentionally falls through to the unmodified dev-m physical selector
// below (most importantly its BM64 bucket beginning at M65).
static constexpr std::array<SM90NVFP4FusedBucket, 11>
kSM90NVFP4H200FusedBuckets {{
    {4096,  1,   8,  8, 16, 4, true, true, true},
    {4096,  9,  16,  8, 32, 4, true, true, true},
    {4096, 17,  32, 24, 32, 3, true, true, false},
    {4096, 33,  64, 24, 32, 3, true, true, true},
    {7168,  1,   8,  8, 16, 3, true, true, true},
    {7168,  9,  16,  8, 24, 3, true, true, true},
    {7168, 17,  32, 24, 48, 3, true, true, false},
    {7168, 33, 128, 24, 48, 3, true, true, true},
    {6144,  1,   8,  8, 16, 4, true, true, true},
    {6144,  9,  16,  8, 24, 4, true, true, true},
    {6144, 33,  96, 24, 48, 3, true, true, true},
}};

static SM90NVFP4FusedPlan select_sm90_nvfp4_fused(
        const SM90NVFP4FusedInput& input,
        const bool use_interleaved_scheduler = true) {
    DG_HOST_ASSERT(input.shape().is_supported_shape());
    DG_HOST_ASSERT(input.num_experts ==
                   input.num_experts_per_rank * input.num_ranks);
    DG_HOST_ASSERT(input.num_experts_per_rank == 32 ||
                   input.num_experts_per_rank == 48);
    DG_HOST_ASSERT(input.num_max_tokens_per_rank > 0);
    DG_HOST_ASSERT(input.num_tokens <= input.num_max_tokens_per_rank);
    DG_HOST_ASSERT(
        SM90NVFP4FusedShape::is_supported_batch(input.num_tokens));
    DG_HOST_ASSERT(input.num_padded_sf_pool_tokens > 0);

    struct Tuning {
        int block_m, block_n;
        int num_experts_per_wave;
        int num_stages;
        int smem_size;
        bool swap_ab;
        bool use_mode2_row_decoder;
        bool single_active_dispatch_warp;
    } tuning {};

    bool range_bucket = false;
    const auto find_range_bucket = [&](const auto& buckets) {
        for (const auto& bucket : buckets) {
            if (bucket.hidden == input.hidden &&
                input.num_tokens >= bucket.min_tokens &&
                input.num_tokens <= bucket.max_tokens) {
                tuning = {
                    bucket.block_m,
                    256,
                    bucket.num_experts_per_wave,
                    bucket.num_stages,
                    SM90ArchSpec::smem_capacity,
                    bucket.swap_ab,
                    bucket.use_mode2_row_decoder,
                    bucket.single_active_dispatch_warp,
                };
                range_bucket = true;
                return;
            }
        }
    };
    if (input.num_sms == SM90NVFP4FusedShape::kH20NumSMs)
        find_range_bucket(kSM90NVFP4H20FusedBuckets);
    else if (input.num_sms == SM90NVFP4FusedShape::kH200NumSMs)
        find_range_bucket(kSM90NVFP4H200FusedBuckets);

    if (!range_bucket && input.num_tokens <= 1)
        tuning = {8, 256, 24, 4, SM90ArchSpec::smem_capacity,
                  true, true, true};
    else if (!range_bucket && input.num_tokens <= 8)
        tuning = {8, 256, 16, 4, SM90ArchSpec::smem_capacity,
                  true, true, true};
    else if (!range_bucket && input.num_tokens <= 16)
        tuning = {8, 256, 24, 4, SM90ArchSpec::smem_capacity,
                  true, true, true};
    else if (!range_bucket && input.num_tokens <= 32)
        tuning = {16, 256, 48, 3, SM90ArchSpec::smem_capacity,
                  true, true, false};
    else if (!range_bucket && input.num_tokens <= 64)
        tuning = {24, 256, 48, 3, 229312,
                  true, false, true};
    else if (!range_bucket && input.num_tokens <= 256)
        tuning = {64, 256, 48, 3, 209856,
                  false, true, false};
    else if (!range_bucket)
        tuning = {128, 128, 48, 6, SM90ArchSpec::smem_capacity,
                  false, true, false};

    tuning.num_experts_per_wave = cute::min(
        tuning.num_experts_per_wave, input.num_experts_per_rank);
    while (tuning.num_experts_per_wave < input.num_experts_per_rank &&
           input.num_experts_per_rank % tuning.num_experts_per_wave != 0)
        ++tuning.num_experts_per_wave;

    // The unified interleaved scheduler appends a 96-byte mailbox after the
    // original dev-m barriers.  In particular, the inherited MiMo BM64
    // 209856-byte launch is 96 bytes short of the new 209952-byte end.  Use
    // the full SM90 capacity for every fused shape; all plans are persistent
    // one-CTA-per-SM kernels, so this does not reduce resident CTA count.
    tuning.smem_size = SM90ArchSpec::smem_capacity;

    const bool is_pro =
        input.num_experts == 384 && input.num_topk == 6 &&
        input.hidden == 7168 && input.intermediate_hidden == 3072;
    if (is_pro && tuning.block_m == 8 && tuning.num_stages == 4)
        tuning.num_stages = 3;

    DG_HOST_ASSERT(
        input.num_experts_per_rank % tuning.num_experts_per_wave == 0);
    DG_HOST_ASSERT(tuning.smem_size <= SM90ArchSpec::smem_capacity);
    return {
        {
            tuning.block_m,
            tuning.block_n,
            layout::get_num_max_pool_tokens(
                input.num_ranks, input.num_max_tokens_per_rank,
                input.num_topk, input.num_experts_per_rank),
            input.num_padded_sf_pool_tokens,
            tuning.num_experts_per_wave,
            tuning.num_stages,
            use_interleaved_scheduler ?
                cute::min(
                    tuning.smem_size +
                        layout::kSM90InterleavedSchedulerSMEMBytes,
                    SM90ArchSpec::smem_capacity) :
                tuning.smem_size,
        },
        tuning.swap_ab,
        tuning.use_mode2_row_decoder,
        tuning.single_active_dispatch_warp,
        use_interleaved_scheduler,
    };
}

static std::string get_sm90_nvfp4_small_jit_flags(const bool fast_math) {
    const std::string register_flags =
        "--ptxas-options=--register-usage-level=5";
    return fast_math ? "--use_fast_math " + register_flags : register_flags;
}

enum class SM90NVFP4Target : uint32_t {
    Unsupported,
    H20,
    H200,
};

enum class SM90NVFP4Model : uint32_t {
    Unsupported,
    Flash,
    Pro,
    MiMo,
};

enum class SM90NVFP4AllMArm : uint32_t {
    DevMDynamic,
    DynamicRS,
    StaticSS,
    KF424StaticRS,
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

struct SM90NVFP4SmallMBucket {
    SM90NVFP4Target target;
    SM90NVFP4Model model;
    int min_tokens;
    int max_tokens;
    SM90NVFP4AllMArm arm;
};

static constexpr SM90NVFP4Target get_sm90_nvfp4_target(
        const int num_sms) {
    return num_sms == 78 ? SM90NVFP4Target::H20 :
           num_sms == 132 ? SM90NVFP4Target::H200 :
           SM90NVFP4Target::Unsupported;
}

static constexpr SM90NVFP4Model get_sm90_nvfp4_model(
        const SM90NVFP4AllMPolicyInput& input) {
    if (input.num_ranks != 8)
        return SM90NVFP4Model::Unsupported;
    if (input.num_experts == 256 && input.num_topk == 6 &&
        input.hidden == 4096 && input.intermediate_hidden == 2048)
        return SM90NVFP4Model::Flash;
    if (input.num_experts == 384 && input.num_topk == 6 &&
        input.hidden == 7168 && input.intermediate_hidden == 3072)
        return SM90NVFP4Model::Pro;
    if (input.num_experts == 384 && input.num_topk == 8 &&
        input.hidden == 6144 && input.intermediate_hidden == 2048)
        return SM90NVFP4Model::MiMo;
    return SM90NVFP4Model::Unsupported;
}

// One table owns every architecture-specific small-M arm bucket. Both targets
// use inclusive ranges; their physical BM/EPW/stage tables remain separate
// above. The large-M split arm is deliberately absent because H20 and H200
// share the same bigM implementation and runtime heuristic.
static constexpr std::array<SM90NVFP4SmallMBucket, 15>
kSM90NVFP4SmallMBuckets {{
    {SM90NVFP4Target::H20,  SM90NVFP4Model::Flash,   1, 128,
     SM90NVFP4AllMArm::DynamicRS},
    {SM90NVFP4Target::H20,  SM90NVFP4Model::Flash, 129, 0x7fffffff,
     SM90NVFP4AllMArm::DevMDynamic},
    {SM90NVFP4Target::H20,  SM90NVFP4Model::Pro,     1, 191,
     SM90NVFP4AllMArm::DynamicRS},
    {SM90NVFP4Target::H20,  SM90NVFP4Model::Pro,   192, 0x7fffffff,
     SM90NVFP4AllMArm::DevMDynamic},
    {SM90NVFP4Target::H20,  SM90NVFP4Model::MiMo,    1, 144,
     SM90NVFP4AllMArm::DynamicRS},
    {SM90NVFP4Target::H20,  SM90NVFP4Model::MiMo,  145, 0x7fffffff,
     SM90NVFP4AllMArm::DevMDynamic},
    {SM90NVFP4Target::H200, SM90NVFP4Model::Flash,   1,  35,
     SM90NVFP4AllMArm::DynamicRS},
    {SM90NVFP4Target::H200, SM90NVFP4Model::Flash,  36,  64,
     SM90NVFP4AllMArm::StaticSS},
    {SM90NVFP4Target::H200, SM90NVFP4Model::Flash,  65, 0x7fffffff,
     SM90NVFP4AllMArm::DevMDynamic},
    {SM90NVFP4Target::H200, SM90NVFP4Model::Pro,     1, 128,
     SM90NVFP4AllMArm::DynamicRS},
    {SM90NVFP4Target::H200, SM90NVFP4Model::Pro,   129, 0x7fffffff,
     SM90NVFP4AllMArm::DevMDynamic},
    {SM90NVFP4Target::H200, SM90NVFP4Model::MiMo,    1,  16,
     SM90NVFP4AllMArm::DynamicRS},
    {SM90NVFP4Target::H200, SM90NVFP4Model::MiMo,   17,  32,
     SM90NVFP4AllMArm::DevMDynamic},
    {SM90NVFP4Target::H200, SM90NVFP4Model::MiMo,   33,  96,
     SM90NVFP4AllMArm::DynamicRS},
    {SM90NVFP4Target::H200, SM90NVFP4Model::MiMo,   97, 0x7fffffff,
     SM90NVFP4AllMArm::DevMDynamic},
}};

static constexpr SM90NVFP4AllMArm select_sm90_nvfp4_allm_arm(
        const SM90NVFP4AllMPolicyInput& input) {
    // Resolve the physical SM90 target first. Both supported targets share
    // bigM, while their fused/small-M portfolios differ below.
    const auto target = get_sm90_nvfp4_target(input.num_sms);
    if (input.selected_kernel_block_n == 128)
        return SM90NVFP4AllMArm::BigMSplit;

    const auto model = get_sm90_nvfp4_model(input);
    for (const auto& bucket : kSM90NVFP4SmallMBuckets) {
        if (bucket.target == target && bucket.model == model &&
            input.num_tokens >= bucket.min_tokens &&
            input.num_tokens <= bucket.max_tokens)
            return bucket.arm;
    }

    // The default fused path is always dev-m dynamic.  Its own selector
    // rejects unsupported H20/H200 hardware or model geometry fail-closed.
    return SM90NVFP4AllMArm::DevMDynamic;
}

static constexpr const char* sm90_nvfp4_allm_arm_name(
        const SM90NVFP4AllMArm arm) {
    switch (arm) {
        case SM90NVFP4AllMArm::DevMDynamic:
            return "devm-dynamic";
        case SM90NVFP4AllMArm::DynamicRS:
            return "dynamic-rs-mode5";
        case SM90NVFP4AllMArm::StaticSS:
            return "static-ss";
        case SM90NVFP4AllMArm::KF424StaticRS:
            return "kf424-static-rs";
        case SM90NVFP4AllMArm::BigMSplit:
            return "bigm-split-mode4";
    }
    return "unknown";
}

static_assert(select_sm90_nvfp4_allm_arm(
    {78, 8, 256, 37, 6, 4096, 2048, 256}) ==
    SM90NVFP4AllMArm::DynamicRS);
static_assert(select_sm90_nvfp4_allm_arm(
    {78, 8, 384, 191, 6, 7168, 3072, 256}) ==
    SM90NVFP4AllMArm::DynamicRS);
static_assert(select_sm90_nvfp4_allm_arm(
    {78, 8, 384, 192, 6, 7168, 3072, 256}) ==
    SM90NVFP4AllMArm::DevMDynamic);
static_assert(select_sm90_nvfp4_allm_arm(
    {78, 8, 256, 128, 6, 4096, 2048, 256}) ==
    SM90NVFP4AllMArm::DynamicRS);
static_assert(select_sm90_nvfp4_allm_arm(
    {78, 8, 256, 129, 6, 4096, 2048, 256}) ==
    SM90NVFP4AllMArm::DevMDynamic);
static_assert(select_sm90_nvfp4_allm_arm(
    {78, 8, 384, 144, 8, 6144, 2048, 256}) ==
    SM90NVFP4AllMArm::DynamicRS);
static_assert(select_sm90_nvfp4_allm_arm(
    {78, 8, 384, 145, 8, 6144, 2048, 256}) ==
    SM90NVFP4AllMArm::DevMDynamic);
static_assert(select_sm90_nvfp4_allm_arm(
    {132, 8, 256, 35, 6, 4096, 2048, 256}) ==
    SM90NVFP4AllMArm::DynamicRS);
static_assert(select_sm90_nvfp4_allm_arm(
    {132, 8, 256, 36, 6, 4096, 2048, 256}) ==
    SM90NVFP4AllMArm::StaticSS);
static_assert(select_sm90_nvfp4_allm_arm(
    {132, 8, 256, 64, 6, 4096, 2048, 256}) ==
    SM90NVFP4AllMArm::StaticSS);
static_assert(select_sm90_nvfp4_allm_arm(
    {132, 8, 256, 65, 6, 4096, 2048, 256}) ==
    SM90NVFP4AllMArm::DevMDynamic);
static_assert(select_sm90_nvfp4_allm_arm(
    {132, 8, 384, 128, 6, 7168, 3072, 256}) ==
    SM90NVFP4AllMArm::DynamicRS);
static_assert(select_sm90_nvfp4_allm_arm(
    {132, 8, 384, 129, 6, 7168, 3072, 256}) ==
    SM90NVFP4AllMArm::DevMDynamic);
static_assert(select_sm90_nvfp4_allm_arm(
    {132, 8, 384, 16, 8, 6144, 2048, 256}) ==
    SM90NVFP4AllMArm::DynamicRS);
static_assert(select_sm90_nvfp4_allm_arm(
    {132, 8, 384, 17, 8, 6144, 2048, 256}) ==
    SM90NVFP4AllMArm::DevMDynamic);
static_assert(select_sm90_nvfp4_allm_arm(
    {132, 8, 384, 33, 8, 6144, 2048, 256}) ==
    SM90NVFP4AllMArm::DynamicRS);
static_assert(select_sm90_nvfp4_allm_arm(
    {132, 8, 384, 96, 8, 6144, 2048, 256}) ==
    SM90NVFP4AllMArm::DynamicRS);
static_assert(select_sm90_nvfp4_allm_arm(
    {132, 8, 384, 97, 8, 6144, 2048, 256}) ==
    SM90NVFP4AllMArm::DevMDynamic);
static_assert(select_sm90_nvfp4_allm_arm(
    {78, 8, 384, 2048, 6, 7168, 3072, 128}) ==
    SM90NVFP4AllMArm::BigMSplit);

}  // namespace deep_gemm
