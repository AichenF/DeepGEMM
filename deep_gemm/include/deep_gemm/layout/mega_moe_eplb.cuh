#pragma once

#include <deep_gemm/common/math.cuh>
#include <deep_gemm/common/exception.cuh>

// Host/device shared layout for MegaMoE dynamic expert load balancing. The
// kernels live in `deep_gemm/impls/smxx_mega_moe_eplb.cuh`.

namespace deep_gemm::eplb {

// Symmetric planner workspace. All offsets are in 32-bit words so the host
// and the kernels slice the same buffer the same way.
struct Workspace {
    static constexpr uint32_t kMaxRanks = 32;

    uint32_t num_ranks, num_experts, num_helper_slots;

    CUTLASS_HOST_DEVICE
    Workspace(const uint32_t& num_ranks, const uint32_t& num_experts, const uint32_t& num_helper_slots):
        num_ranks(num_ranks), num_experts(num_experts), num_helper_slots(num_helper_slots) {}

    // Generation flags, double-buffered by generation parity: [2][kMaxRanks]
    CUTLASS_HOST_DEVICE uint32_t flags_offset() const { return 0; }
    // Expert histograms pushed by every rank: [2][num_ranks][num_experts]
    CUTLASS_HOST_DEVICE uint32_t hist_offset() const { return flags_offset() + 2 * kMaxRanks; }
    // Helper slot table, identical on all ranks: [num_ranks][num_helper_slots]
    CUTLASS_HOST_DEVICE uint32_t slot_expert_offset() const { return hist_offset() + 2 * num_ranks * num_experts; }
    // Per-expert copies as int2 {rank, cumulative blocks}: [num_experts][num_ranks]
    CUTLASS_HOST_DEVICE uint32_t copies_offset() const { return slot_expert_offset() + math::align(num_ranks * num_helper_slots, 2u); }
    // This rank's ordinal base per expert: [num_experts]
    CUTLASS_HOST_DEVICE uint32_t expert_base_offset() const { return copies_offset() + 2 * num_experts * num_ranks; }
    // Remap counters, zeroed by the planner: [num_experts]
    CUTLASS_HOST_DEVICE uint32_t local_counter_offset() const { return expert_base_offset() + num_experts; }
    // This rank's weight copies as int2 {helper slot, expert}: [num_helper_slots]
    CUTLASS_HOST_DEVICE uint32_t copy_list_offset() const { return local_counter_offset() + math::align(num_experts, 2u); }
    CUTLASS_HOST_DEVICE uint32_t copy_count_offset() const { return copy_list_offset() + 2 * num_helper_slots; }
    CUTLASS_HOST_DEVICE uint32_t num_words() const { return math::align(copy_count_offset() + 1, 4u); }
    CUTLASS_HOST_DEVICE uint64_t num_bytes() const { return static_cast<uint64_t>(num_words()) * sizeof(uint32_t); }
};

// Shared memory the planner needs, in bytes.
CUTLASS_HOST_DEVICE constexpr uint32_t plan_smem_bytes(const uint32_t& num_ranks, const uint32_t& num_experts,
                                                       const uint32_t& num_helper_slots) {
    const auto r = num_ranks, e = num_experts, s = num_helper_slots;
    uint32_t bytes = 0;
    bytes += r * e * sizeof(int32_t);         // hist
    bytes += e * r * sizeof(int32_t);         // q
    bytes += e * r * sizeof(int32_t);         // q snapshot for pruning
    bytes += e * sizeof(int32_t);             // b
    bytes += r * sizeof(int32_t) * 4;         // load, nrep, visited, pad
    bytes += r * s * sizeof(int32_t) * 2;     // previous and new slot tables
    bytes += math::constexpr_align(r * e, 4u); // holds
    bytes += math::constexpr_align(e * r, 4u); // done marks
    bytes += math::constexpr_align(e, 4u);     // tried marks
    return bytes;
}

// One weight plane of the symmetric weight arena: `bytes_per_slot` contiguous
// bytes per local slot, starting `offset` bytes into the arena.
struct CopyPlane {
    uint64_t offset;
    uint32_t bytes_per_slot;
    uint32_t vec_bytes;
};

struct CopyPlanes {
    static constexpr uint32_t kMaxPlanes = 8;
    CopyPlane planes[kMaxPlanes];
    uint32_t num_planes;
};

} // namespace deep_gemm::eplb
