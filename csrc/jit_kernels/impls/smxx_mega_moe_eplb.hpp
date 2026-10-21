#pragma once

#include <cstdio>
#include <cstdlib>
#include <string>
#include <torch/python.h>

#include "../../jit/compiler.hpp"
#include "../../jit/kernel_runtime.hpp"
#include "../../utils/exception.hpp"
#include "../../utils/format.hpp"

#include <deep_gemm/layout/mega_moe_eplb.cuh>
#include <deep_gemm/layout/sym_buffer.cuh>

namespace deep_gemm {

// `DG_EPLB_PROFILE=1` instantiates the planner kernels with phase stamps
// written to the workspace's profile records (see `eplb::PlanProfile`).
static bool eplb_profile_enabled() {
    const auto* env = std::getenv("DG_EPLB_PROFILE");
    return env != nullptr and std::string(env) == "1";
}

class MegaMoEEPLBPlanRuntime final: public LaunchRuntime<MegaMoEEPLBPlanRuntime> {
public:
    struct Args {
        int num_ranks, num_experts, num_helper_slots;
        layout::SymBuffer<> workspace;
        const int64_t* topk_idx;
        int num_tokens, num_topk;
        int block_m, open_gate_blocks;
        unsigned generation;
        LaunchArgs launch_args;
    };

    static std::string generate_impl(const Args& args) {
        return fmt::format(R"(
{}#include <deep_gemm/impls/smxx_mega_moe_eplb.cuh>

using namespace deep_gemm;

static void __instantiate_kernel() {{
    auto ptr = reinterpret_cast<void*>(&eplb::eplb_plan_kernel<
        {}, {}, {}, {}, {}
    >);
}};
)", std::getenv("DG_EPLB_DEBUG") ? "#define DG_EPLB_DEBUG 1\n" : "",
    args.num_ranks, args.num_experts, args.num_helper_slots, args.launch_args.num_threads,
    eplb_profile_enabled() ? "true" : "false");
    }

    static void launch_impl(const KernelHandle& kernel, const LaunchConfigHandle& config, Args args) {
        if (std::getenv("DG_EPLB_DEBUG"))
            printf("EPLB plan launch: rank=%d generation=%u num_tokens=%d block_m=%d\n",
                   args.workspace.rank_idx, args.generation, args.num_tokens, args.block_m);
        DG_CUDA_UNIFIED_CHECK(launch_kernel(kernel, config,
            args.workspace, args.topk_idx,
            static_cast<uint32_t>(args.num_tokens), static_cast<uint32_t>(args.num_topk),
            static_cast<uint32_t>(args.block_m), static_cast<uint32_t>(args.open_gate_blocks),
            static_cast<uint32_t>(args.generation)));
    }
};

// Plan and remap in one launch (`eplb_plan_remap_kernel`); `topk_idx` is
// rewritten in place.
class MegaMoEEPLBPlanRemapRuntime final: public LaunchRuntime<MegaMoEEPLBPlanRemapRuntime> {
public:
    struct Args {
        int num_ranks, num_experts, num_helper_slots;
        layout::SymBuffer<> workspace;
        int64_t* topk_idx;
        int num_tokens, num_topk;
        int block_m, open_gate_blocks;
        unsigned generation;
        LaunchArgs launch_args;
    };

    static std::string generate_impl(const Args& args) {
        return fmt::format(R"(
#include <deep_gemm/impls/smxx_mega_moe_eplb.cuh>

using namespace deep_gemm;

static void __instantiate_kernel() {{
    auto ptr = reinterpret_cast<void*>(&eplb::eplb_plan_remap_kernel<
        {}, {}, {}, {}, {}
    >);
}};
)", args.num_ranks, args.num_experts, args.num_helper_slots, args.launch_args.num_threads,
    eplb_profile_enabled() ? "true" : "false");
    }

    static void launch_impl(const KernelHandle& kernel, const LaunchConfigHandle& config, Args args) {
        DG_CUDA_UNIFIED_CHECK(launch_kernel(kernel, config,
            args.workspace, args.topk_idx,
            static_cast<uint32_t>(args.num_tokens), static_cast<uint32_t>(args.num_topk),
            static_cast<uint32_t>(args.block_m), static_cast<uint32_t>(args.open_gate_blocks),
            static_cast<uint32_t>(args.generation)));
    }
};

class MegaMoEEPLBRemapRuntime final: public LaunchRuntime<MegaMoEEPLBRemapRuntime> {
public:
    struct Args {
        int num_ranks, num_experts, num_helper_slots;
        uint32_t* workspace;
        int rank_idx;
        int64_t* topk_idx;
        int num_tokens, num_topk, block_m;
        LaunchArgs launch_args;
    };

    static std::string generate_impl(const Args& args) {
        return fmt::format(R"(
#include <deep_gemm/impls/smxx_mega_moe_eplb.cuh>

using namespace deep_gemm;

static void __instantiate_kernel() {{
    auto ptr = reinterpret_cast<void*>(&eplb::eplb_remap_kernel<
        {}, {}, {}, {}
    >);
}};
)", args.num_ranks, args.num_experts, args.num_helper_slots, args.launch_args.num_threads);
    }

    static void launch_impl(const KernelHandle& kernel, const LaunchConfigHandle& config, Args args) {
        DG_CUDA_UNIFIED_CHECK(launch_kernel(kernel, config,
            args.workspace, static_cast<uint32_t>(args.rank_idx), args.topk_idx,
            static_cast<uint32_t>(args.num_tokens), static_cast<uint32_t>(args.num_topk),
            static_cast<uint32_t>(args.block_m)));
    }
};

class MegaMoEEPLBCopyWeightsRuntime final: public LaunchRuntime<MegaMoEEPLBCopyWeightsRuntime> {
public:
    struct Args {
        int num_ranks;
        layout::SymBuffer<> arena;
        eplb::CopyPlanes planes;
        const int2* copy_list;
        const int32_t* copy_count;
        int num_home_experts;
        LaunchArgs launch_args;
    };

    static std::string generate_impl(const Args& args) {
        return fmt::format(R"(
#include <deep_gemm/impls/smxx_mega_moe_eplb.cuh>

using namespace deep_gemm;

static void __instantiate_kernel() {{
    auto ptr = reinterpret_cast<void*>(&eplb::eplb_copy_weights_kernel<
        {}, {}
    >);
}};
)", args.num_ranks, args.launch_args.num_threads);
    }

    static void launch_impl(const KernelHandle& kernel, const LaunchConfigHandle& config, Args args) {
        DG_CUDA_UNIFIED_CHECK(launch_kernel(kernel, config,
            args.arena, args.planes, args.copy_list, args.copy_count,
            static_cast<uint32_t>(args.num_home_experts)));
    }
};

// Builds the MegaMoE kernel's fused-copy argument. `l1_planes` and
// `l2_planes` list (byte offset into the arena, bytes per slot) of the planes
// the L1 and the L2 GEMM read, respectively; the kernel gates each group on
// its own counter.
static eplb::FusedCopy<> make_eplb_fused_copy(
        const torch::Tensor& workspace,
        const torch::Tensor& arena, const std::vector<int64_t>& arena_ptrs, const int& rank_idx,
        const int& num_experts, const int& num_helper_slots,
        const std::vector<std::tuple<int64_t, int64_t>>& l1_planes,
        const std::vector<std::tuple<int64_t, int64_t>>& l2_planes) {
    const auto num_ranks = static_cast<int>(arena_ptrs.size());
    DG_HOST_ASSERT(num_ranks > 0 and num_experts % num_ranks == 0 and num_helper_slots > 0);
    DG_HOST_ASSERT(num_helper_slots <= 32);
    DG_HOST_ASSERT(rank_idx >= 0 and rank_idx < num_ranks);
    DG_HOST_ASSERT(arena.is_cuda() and arena.is_contiguous());
    DG_HOST_ASSERT(arena_ptrs[rank_idx] == reinterpret_cast<int64_t>(arena.data_ptr()));
    DG_HOST_ASSERT(workspace.is_cuda() and workspace.is_contiguous());
    const eplb::Workspace ws(num_ranks, num_experts, num_helper_slots);
    DG_HOST_ASSERT(workspace.nbytes() >= ws.num_bytes());
    const int num_home_experts = num_experts / num_ranks;
    const int num_slots = num_home_experts + num_helper_slots;
    const auto num_planes = l1_planes.size() + l2_planes.size();
    DG_HOST_ASSERT(num_planes > 0 and num_planes <= eplb::CopyPlanes::kMaxPlanes);

    eplb::FusedCopy<> fc = {};
    fc.arena = layout::SymBuffer<>(arena_ptrs, rank_idx);
    fc.planes.num_planes = static_cast<uint32_t>(num_planes);
    size_t k = 0;
    for (const auto* group : {&l1_planes, &l2_planes}) {
        for (const auto& [offset, bytes_per_slot] : *group) {
            DG_HOST_ASSERT(offset >= 0 and bytes_per_slot > 0 and bytes_per_slot % 4 == 0);
            DG_HOST_ASSERT(offset + bytes_per_slot * num_slots <= static_cast<int64_t>(arena.nbytes()));
            const bool vec16 = offset % 16 == 0 and bytes_per_slot % 16 == 0;
            DG_HOST_ASSERT(vec16 or offset % 4 == 0);
            fc.planes.planes[k ++] = {static_cast<uint64_t>(offset), static_cast<uint32_t>(bytes_per_slot),
                                      vec16 ? 16u : 4u};
        }
    }
    auto* ws_base = static_cast<uint32_t*>(workspace.data_ptr());
    fc.copy_list = reinterpret_cast<const int2*>(ws_base + ws.copy_list_offset());
    fc.copy_count = reinterpret_cast<int32_t*>(ws_base + ws.copy_count_offset());
    fc.copy_done = ws_base + ws.copy_done_offset();
    fc.num_home_experts = static_cast<uint32_t>(num_home_experts);
    fc.num_helper_slots = static_cast<uint32_t>(num_helper_slots);
    fc.num_l1_planes = static_cast<uint32_t>(l1_planes.size());
    return fc;
}

} // namespace deep_gemm
