#pragma once

#include <map>
#include <tuple>

#include <pybind11/stl.h>

#include "../jit/device_runtime.hpp"
#include "../utils/layout.hpp"
#include "../jit_kernels/impls/smxx_mega_moe_eplb.hpp"
#include "../jit_kernels/heuristics/sm90_mxfp4_mega_moe.hpp"

namespace deep_gemm::mega {

// Dynamic expert load balancing: planner workspace layout, the three
// kernels that turn logical routing into physical slot routing, and the
// single-launch plan + remap (the logical-to-physical remap of the top-k
// ids). See
// `deep_gemm/include/deep_gemm/impls/smxx_mega_moe_eplb.cuh`.

// The generated stub of each EPLB kernel depends only on its template
// parameters, so the built runtime is cached per parameter tuple and a
// steady-state launch is a map lookup plus the driver launch. Without this,
// every call paid the stub formatting, the include hash prefix and two
// signature hashes before `compiler->build` could hit its own cache (about
// 15 us of host time per launch, see `dynamic_eplb/PROFILE_eplb_stages.md`).
// The last element is the profile flag (`DG_EPLB_PROFILE`), which changes
// the generated stub of the two planner kernels.
using EPLBRuntimeKey = std::tuple<int, int, int, int, int>;

template <typename Runtime>
static std::shared_ptr<KernelRuntime> get_eplb_runtime(const char* name, const EPLBRuntimeKey& key,
                                                        const typename Runtime::Args& args) {
    static std::map<EPLBRuntimeKey, std::shared_ptr<KernelRuntime>> cache;
    if (const auto it = cache.find(key); it != cache.end())
        return it->second;
    const auto code = Runtime::generate(args);
    const auto runtime = compiler->build(name, code);
    cache.emplace(key, runtime);
    return runtime;
}

static void check_eplb_geometry(const int& num_ranks, const int& num_experts, const int& num_redundant_slots) {
    DG_HOST_ASSERT(num_ranks > 0 and num_ranks <= static_cast<int>(eplb::Workspace::kMaxRanks));
    DG_HOST_ASSERT(num_experts > 0 and num_experts % num_ranks == 0);
    DG_HOST_ASSERT(num_redundant_slots > 0);
}

static pybind11::dict get_mega_moe_eplb_workspace_layout(
        const int& num_ranks, const int& num_experts, const int& num_redundant_slots) {
    check_eplb_geometry(num_ranks, num_experts, num_redundant_slots);
    const eplb::Workspace ws(num_ranks, num_experts, num_redundant_slots);
    pybind11::dict d;
    d["flags"] = ws.flags_offset();
    d["hist"] = ws.hist_offset();
    d["slot_expert"] = ws.slot_expert_offset();
    d["copies"] = ws.copies_offset();
    d["expert_base"] = ws.expert_base_offset();
    d["local_counter"] = ws.local_counter_offset();
    d["copy_list"] = ws.copy_list_offset();
    d["copy_count"] = ws.copy_count_offset();
    d["copy_done"] = ws.copy_done_offset();
    d["remap_done"] = ws.remap_done_offset();
    d["local_hist"] = ws.local_hist_offset();
    d["plan_ticket"] = ws.plan_ticket_offset();
    d["plan_ready"] = ws.plan_ready_offset();
    d["profile"] = ws.profile_offset();
    d["num_profile_slots"] = eplb::Workspace::kNumProfileSlots;
    d["num_profile_words"] = eplb::Workspace::kNumProfileWords;
    d["num_words"] = ws.num_words();
    d["num_bytes"] = ws.num_bytes();
    return d;
}

static void check_eplb_workspace(const torch::Tensor& workspace, const int& num_ranks,
                                 const int& num_experts, const int& num_redundant_slots) {
    const eplb::Workspace ws(num_ranks, num_experts, num_redundant_slots);
    DG_HOST_ASSERT(workspace.is_cuda() and workspace.is_contiguous());
    DG_HOST_ASSERT(workspace.nbytes() >= ws.num_bytes());
    DG_HOST_ASSERT(reinterpret_cast<uintptr_t>(workspace.data_ptr()) % 16 == 0);
}

static void check_eplb_topk_idx(const torch::Tensor& topk_idx) {
    DG_HOST_ASSERT(topk_idx.is_cuda() and topk_idx.dim() == 2);
    DG_HOST_ASSERT(topk_idx.scalar_type() == torch::kInt64);
    DG_HOST_ASSERT(topk_idx.is_contiguous());
}

static void mega_moe_eplb_plan(
        const torch::Tensor& workspace, const std::vector<int64_t>& workspace_ptrs, const int& rank_idx,
        const torch::Tensor& topk_idx,
        const int& num_experts, const int& num_redundant_slots,
        const int& block_m, const int& open_gate_blocks, const int64_t& generation) {
    const auto num_ranks = static_cast<int>(workspace_ptrs.size());
    check_eplb_geometry(num_ranks, num_experts, num_redundant_slots);
    check_eplb_workspace(workspace, num_ranks, num_experts, num_redundant_slots);
    check_eplb_topk_idx(topk_idx);
    DG_HOST_ASSERT(rank_idx >= 0 and rank_idx < num_ranks);
    DG_HOST_ASSERT(block_m > 0 and open_gate_blocks >= 0);
    DG_HOST_ASSERT(generation > 0 and generation <= 0xffffffffll);
    DG_HOST_ASSERT(workspace_ptrs[rank_idx] == reinterpret_cast<int64_t>(workspace.data_ptr()));
    const auto [num_tokens, num_topk] = get_shape<2>(topk_idx);

    // One CTA counts every token of the rank; its histogram is bound by the
    // SM's shared-atomic throughput (512 threads measured the same), so the
    // critical path uses the grid-wide `mega_moe_eplb_plan_and_remap`.
    constexpr int kNumThreads = 256;
    const auto smem_bytes = static_cast<int>(eplb::plan_smem_bytes(num_ranks, num_experts, num_redundant_slots, kNumThreads / 32));
    const MegaMoEEPLBPlanRuntime::Args args = {
        .num_ranks = num_ranks, .num_experts = num_experts, .num_redundant_slots = num_redundant_slots,
        .workspace = layout::SymBuffer<>(workspace_ptrs, rank_idx),
        .topk_idx = topk_idx.data_ptr<int64_t>(),
        .num_tokens = num_tokens, .num_topk = num_topk,
        .block_m = block_m, .open_gate_blocks = open_gate_blocks,
        .generation = static_cast<unsigned>(generation),
        .launch_args = LaunchArgs(1, kNumThreads, smem_bytes)
    };
    const auto runtime = get_eplb_runtime<MegaMoEEPLBPlanRuntime>(
        "mega_moe_eplb_plan", {num_ranks, num_experts, num_redundant_slots, kNumThreads, eplb_profile_enabled()}, args);
    MegaMoEEPLBPlanRuntime::launch(runtime, args);
}

// Plan and remap in one launch: `topk_idx` (logical) is rewritten to physical
// slot ids in place and the tables are published as by `mega_moe_eplb_plan`.
// Grid of at most one CTA per SM; see `eplb_plan_remap_kernel`.
static void mega_moe_eplb_plan_and_remap(
        const torch::Tensor& workspace, const std::vector<int64_t>& workspace_ptrs, const int& rank_idx,
        const torch::Tensor& topk_idx,
        const int& num_experts, const int& num_redundant_slots,
        const int& block_m, const int& open_gate_blocks, const int64_t& generation) {
    const auto num_ranks = static_cast<int>(workspace_ptrs.size());
    check_eplb_geometry(num_ranks, num_experts, num_redundant_slots);
    check_eplb_workspace(workspace, num_ranks, num_experts, num_redundant_slots);
    check_eplb_topk_idx(topk_idx);
    DG_HOST_ASSERT(rank_idx >= 0 and rank_idx < num_ranks);
    DG_HOST_ASSERT(block_m > 0 and open_gate_blocks >= 0);
    DG_HOST_ASSERT(generation > 0 and generation <= 0xffffffffll);
    DG_HOST_ASSERT(workspace_ptrs[rank_idx] == reinterpret_cast<int64_t>(workspace.data_ptr()));
    const auto [num_tokens, num_topk] = get_shape<2>(topk_idx);

    constexpr int kNumThreads = 256;
    const int num_blocks = std::max(1, std::min(device_runtime->get_num_sms(),
                                                (num_tokens * num_topk + kNumThreads - 1) / kNumThreads));
    const auto smem_bytes = static_cast<int>(eplb::plan_smem_bytes(num_ranks, num_experts, num_redundant_slots, kNumThreads / 32));
    const MegaMoEEPLBPlanRemapRuntime::Args args = {
        .num_ranks = num_ranks, .num_experts = num_experts, .num_redundant_slots = num_redundant_slots,
        .workspace = layout::SymBuffer<>(workspace_ptrs, rank_idx),
        .topk_idx = topk_idx.data_ptr<int64_t>(),
        .num_tokens = num_tokens, .num_topk = num_topk,
        .block_m = block_m, .open_gate_blocks = open_gate_blocks,
        .generation = static_cast<unsigned>(generation),
        .launch_args = LaunchArgs(num_blocks, kNumThreads, smem_bytes)
    };
    const auto runtime = get_eplb_runtime<MegaMoEEPLBPlanRemapRuntime>(
        "mega_moe_eplb_plan_and_remap", {num_ranks, num_experts, num_redundant_slots, kNumThreads, eplb_profile_enabled()}, args);
    MegaMoEEPLBPlanRemapRuntime::launch(runtime, args);
}

static void mega_moe_eplb_remap(
        const torch::Tensor& workspace, const int& rank_idx, const int& num_ranks,
        const torch::Tensor& topk_idx,
        const int& num_experts, const int& num_redundant_slots, const int& block_m) {
    check_eplb_geometry(num_ranks, num_experts, num_redundant_slots);
    check_eplb_workspace(workspace, num_ranks, num_experts, num_redundant_slots);
    check_eplb_topk_idx(topk_idx);
    DG_HOST_ASSERT(rank_idx >= 0 and rank_idx < num_ranks);
    DG_HOST_ASSERT(block_m > 0);
    const auto [num_tokens, num_topk] = get_shape<2>(topk_idx);

    constexpr int kNumThreads = 256;
    const int num_blocks = std::max(1, std::min(device_runtime->get_num_sms(),
                                                (num_tokens * num_topk + kNumThreads - 1) / kNumThreads));
    const MegaMoEEPLBRemapRuntime::Args args = {
        .num_ranks = num_ranks, .num_experts = num_experts, .num_redundant_slots = num_redundant_slots,
        .workspace = static_cast<uint32_t*>(workspace.data_ptr()),
        .rank_idx = rank_idx,
        .topk_idx = topk_idx.data_ptr<int64_t>(),
        .num_tokens = num_tokens, .num_topk = num_topk, .block_m = block_m,
        .launch_args = LaunchArgs(num_blocks, kNumThreads)
    };
    const auto runtime = get_eplb_runtime<MegaMoEEPLBRemapRuntime>(
        "mega_moe_eplb_remap", {num_ranks, num_experts, num_redundant_slots, kNumThreads, 0}, args);
    MegaMoEEPLBRemapRuntime::launch(runtime, args);
}

// `planes` lists (byte offset into the arena, bytes per slot) for every weight
// plane; each plane holds `num_experts / num_ranks + num_redundant_slots` slots.
static void mega_moe_eplb_copy_weights(
        const torch::Tensor& workspace,
        const torch::Tensor& arena, const std::vector<int64_t>& arena_ptrs, const int& rank_idx,
        const int& num_experts, const int& num_redundant_slots,
        const std::vector<std::tuple<int64_t, int64_t>>& planes) {
    const auto num_ranks = static_cast<int>(arena_ptrs.size());
    check_eplb_geometry(num_ranks, num_experts, num_redundant_slots);
    check_eplb_workspace(workspace, num_ranks, num_experts, num_redundant_slots);
    DG_HOST_ASSERT(rank_idx >= 0 and rank_idx < num_ranks);
    DG_HOST_ASSERT(arena.is_cuda() and arena.is_contiguous());
    DG_HOST_ASSERT(arena_ptrs[rank_idx] == reinterpret_cast<int64_t>(arena.data_ptr()));
    DG_HOST_ASSERT(not planes.empty() and planes.size() <= eplb::CopyPlanes::kMaxPlanes);
    const int num_resident_experts = num_experts / num_ranks;
    const int num_slots = num_resident_experts + num_redundant_slots;

    eplb::CopyPlanes copy_planes = {};
    copy_planes.num_planes = static_cast<uint32_t>(planes.size());
    for (size_t i = 0; i < planes.size(); ++ i) {
        const auto [offset, bytes_per_slot] = planes[i];
        DG_HOST_ASSERT(offset >= 0 and bytes_per_slot > 0 and bytes_per_slot % 4 == 0);
        DG_HOST_ASSERT(offset + bytes_per_slot * num_slots <= static_cast<int64_t>(arena.nbytes()));
        const bool vec16 = offset % 16 == 0 and bytes_per_slot % 16 == 0;
        DG_HOST_ASSERT(vec16 or offset % 4 == 0);
        copy_planes.planes[i] = {static_cast<uint64_t>(offset), static_cast<uint32_t>(bytes_per_slot),
                                 vec16 ? 16u : 4u};
    }

    const eplb::Workspace ws(num_ranks, num_experts, num_redundant_slots);
    auto* ws_base = static_cast<uint32_t*>(workspace.data_ptr());
    constexpr int kNumThreads = 256;
    // Default grid unchanged; `DG_EPLB_COPY_BLOCKS_PER_SLOT` widens it for
    // comparison with the in-kernel weight copy, which uses every SM.
    const int kNumBlocksPerSlot = std::max(1, get_env<int>("DG_EPLB_COPY_BLOCKS_PER_SLOT", 32));
    const MegaMoEEPLBCopyWeightsRuntime::Args args = {
        .num_ranks = num_ranks,
        .arena = layout::SymBuffer<>(arena_ptrs, rank_idx),
        .planes = copy_planes,
        .copy_list = reinterpret_cast<const int2*>(ws_base + ws.copy_list_offset()),
        .copy_count = reinterpret_cast<const int32_t*>(ws_base + ws.copy_count_offset()),
        .num_resident_experts = num_resident_experts,
        .launch_args = LaunchArgs({num_redundant_slots, kNumBlocksPerSlot}, kNumThreads)
    };
    const auto runtime = get_eplb_runtime<MegaMoEEPLBCopyWeightsRuntime>(
        "mega_moe_eplb_copy_weights", {num_ranks, 0, 0, kNumThreads, 0}, args);
    MegaMoEEPLBCopyWeightsRuntime::launch(runtime, args);
}

// The token block size the SM90 MXFP4 kernel will use for this batch, so the
// planner counts the same blocks the kernel schedules.
static int sm90_mxfp4_mega_moe_block_m(
        const int& num_ranks, const int& num_experts, const int& num_topk,
        const int& hidden, const int& intermediate_hidden,
        const int& num_max_tokens_per_rank, const int& num_tokens) {
    const SM90MXFP4FusedInput input {
        device_runtime->get_num_sms(),
        num_ranks, num_experts, num_experts / num_ranks,
        num_max_tokens_per_rank, num_tokens, num_topk,
        hidden, intermediate_hidden, 1,
    };
    return select_sm90_mxfp4_fused(input).config.block_m;
}

static void register_eplb_apis(pybind11::module_& m) {
#if DG_TENSORMAP_COMPATIBLE
    m.def("get_mega_moe_eplb_workspace_layout", &get_mega_moe_eplb_workspace_layout);
    m.def("mega_moe_eplb_plan", &mega_moe_eplb_plan);
    m.def("mega_moe_eplb_plan_and_remap", &mega_moe_eplb_plan_and_remap);
    m.def("mega_moe_eplb_remap", &mega_moe_eplb_remap);
    m.def("mega_moe_eplb_copy_weights", &mega_moe_eplb_copy_weights);
    m.def("sm90_mxfp4_mega_moe_block_m", &sm90_mxfp4_mega_moe_block_m);
#endif
}

} // namespace deep_gemm::mega
