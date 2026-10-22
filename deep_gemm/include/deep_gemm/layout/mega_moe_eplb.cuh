#pragma once

#include <deep_gemm/common/math.cuh>
#include <deep_gemm/common/exception.cuh>
#include <deep_gemm/layout/sym_buffer.cuh>

// Host/device shared layout for MegaMoE dynamic expert load balancing. The
// kernels live in `deep_gemm/impls/smxx_mega_moe_eplb.cuh`.

namespace deep_gemm::eplb {

// Symmetric planner workspace. All offsets are in 32-bit words so the host
// and the kernels slice the same buffer the same way.
struct Workspace {
    static constexpr uint32_t kMaxRanks = 32;
    static constexpr uint32_t kNumProfileSlots = 8;
    static constexpr uint32_t kNumProfileWords = 40;

    uint32_t num_ranks, num_experts, num_redundant_slots;

    CUTLASS_HOST_DEVICE
    Workspace(const uint32_t& num_ranks, const uint32_t& num_experts, const uint32_t& num_redundant_slots):
        num_ranks(num_ranks), num_experts(num_experts), num_redundant_slots(num_redundant_slots) {}

    // Generation flags, double-buffered by generation parity: [2][kMaxRanks]
    CUTLASS_HOST_DEVICE uint32_t flags_offset() const { return 0; }
    // Expert histograms pushed by every rank: [2][num_ranks][num_experts]
    CUTLASS_HOST_DEVICE uint32_t hist_offset() const { return flags_offset() + 2 * kMaxRanks; }
    // Redundant slot table, identical on all ranks: [num_ranks][num_redundant_slots]
    CUTLASS_HOST_DEVICE uint32_t slot_expert_offset() const { return hist_offset() + 2 * num_ranks * num_experts; }
    // Per-expert copies as int2 {rank, cumulative blocks}: [num_experts][num_ranks]
    CUTLASS_HOST_DEVICE uint32_t copies_offset() const { return slot_expert_offset() + math::align(num_ranks * num_redundant_slots, 2u); }
    // This rank's ordinal base per expert: [num_experts]
    CUTLASS_HOST_DEVICE uint32_t expert_base_offset() const { return copies_offset() + 2 * num_experts * num_ranks; }
    // Remap (the logical-to-physical remap of the top-k ids) counters,
    // zeroed by the planner: [num_experts]
    CUTLASS_HOST_DEVICE uint32_t local_counter_offset() const { return expert_base_offset() + num_experts; }
    // This rank's weight copies as int2 {redundant slot, expert}: [num_redundant_slots]
    CUTLASS_HOST_DEVICE uint32_t copy_list_offset() const { return local_counter_offset() + math::align(num_experts, 2u); }
    CUTLASS_HOST_DEVICE uint32_t copy_count_offset() const { return copy_list_offset() + 2 * num_redundant_slots; }
    // Fused-copy readiness counters, one pair per redundant slot: [num_redundant_slots][2]
    // (group 0: the L1 planes, group 1: the L2 planes). The planner zeroes the
    // pair of every slot it lists; each CTA of the MegaMoE kernel adds one per
    // group once its share of the slot is written, so a slot is ready when the
    // counter reaches the grid size. Unused by the standalone copy kernel.
    CUTLASS_HOST_DEVICE uint32_t copy_done_offset() const { return math::align(copy_count_offset() + 1, 2u); }
    // Remap completion ticket: the last CTA of the remap kernel to finish
    // zeroes the remap counters and this word, so no separate fill launch
    // is needed between two remaps.
    CUTLASS_HOST_DEVICE uint32_t remap_done_offset() const { return copy_done_offset() + 2 * num_redundant_slots; }
    // Single-launch prepare (`eplb_plan_remap_kernel`): every CTA adds its
    // share of this rank's histogram here with device atomics; the last CTA
    // to arrive reads it, clears it and plans. [num_experts]
    CUTLASS_HOST_DEVICE uint32_t local_hist_offset() const { return remap_done_offset() + 1; }
    // Arrival ticket of the single-launch kernel, cleared by the planning CTA.
    CUTLASS_HOST_DEVICE uint32_t plan_ticket_offset() const { return local_hist_offset() + num_experts; }
    // Generation whose tables are published, released by the planning CTA
    // and acquired by the remapping CTAs of the same launch.
    CUTLASS_HOST_DEVICE uint32_t plan_ready_offset() const { return plan_ticket_offset() + 1; }
    // Planner timing records (profile builds only, `DG_EPLB_PROFILE=1`):
    // `kNumProfileSlots` records of `kNumProfileWords`, indexed by generation.
    CUTLASS_HOST_DEVICE uint32_t profile_offset() const { return math::align(plan_ready_offset() + 1, 2u); }
    CUTLASS_HOST_DEVICE uint32_t num_words() const {
        return math::align(profile_offset() + kNumProfileSlots * kNumProfileWords, 4u);
    }
    CUTLASS_HOST_DEVICE uint64_t num_bytes() const { return static_cast<uint64_t>(num_words()) * sizeof(uint32_t); }
};

// One planner timing record (profile builds). Clock stamps are absolute
// `clock64` values of the planning CTA's thread 0 at the end of each phase;
// `globaltimer` (ns) is sampled at entry, after the peer wait and at the end
// so the SM clock can be converted and ranks compared.
struct PlanProfile {
    enum Stamp : uint32_t {
        kEntry = 0,     // after the grid dependency sync
        kHistogram,     // this CTA's share of the histogram is counted
        kArrived,       // single launch: last CTA to arrive, local histogram read
        kPublished,     // peer histograms stored, flags released
        kPeersReady,    // every peer's flag of this generation seen
        kSetup,         // b, q, holds and loads ready
        kGreedy,        // greedy rounds done
        kPruned,        // pruning and extreme moves done
        kSlots,         // redundant slot assignment done
        kTables,        // tables published (plan_ready released)
        kRemapped,      // single launch: this CTA's remap share done
        kEnd,
        kNumStamps
    };
    enum Counter : uint32_t {
        kVisits = 0,    // donor visits in the greedy rounds
        kPicks,         // expert picks (inner iterations) in the greedy rounds
        kMoves,         // block moves in the greedy rounds
        kPruneSteps,    // replicas considered by the pruning pass
        kExtremeMoves,  // single-block moves between the extremes
        kGridSize,      // CTAs of the launch
        kNumCounters
    };
    // Word layout: [0, 2 * kNumStamps) clock stamps as 64-bit pairs,
    // then globaltimer at entry, at kPeersReady and at the end (64-bit each),
    // then the counters.
    static constexpr uint32_t kClockWords = 2 * kNumStamps;
    static constexpr uint32_t kTimerEntryWord = kClockWords;
    static constexpr uint32_t kTimerPeersWord = kClockWords + 2;
    static constexpr uint32_t kTimerEndWord = kClockWords + 4;
    static constexpr uint32_t kCounterWord = kClockWords + 6;
    static_assert(kCounterWord + kNumCounters <= Workspace::kNumProfileWords, "Profile record too small");
};

// Shared memory the planner needs, in bytes; the regions of `PlanSmem` in
// `impls/smxx_mega_moe_eplb.cuh`, in order. `num_warps` private histograms
// are kept while the tokens are counted, so hot experts do not serialise one
// shared atomic stream.
CUTLASS_HOST_DEVICE constexpr uint32_t plan_smem_bytes(const uint32_t& num_ranks, const uint32_t& num_experts,
                                                       const uint32_t& num_redundant_slots, const uint32_t& num_warps = 8) {
    const auto r = num_ranks, e = num_experts, s = num_redundant_slots;
    uint32_t bytes = 0;
    bytes += r * e * sizeof(int32_t);                       // hist
    bytes += num_warps * e * sizeof(int32_t);               // per-warp histograms
    bytes += e * r * sizeof(int32_t);                       // q
    bytes += 2 * r * s * sizeof(int32_t);                   // replica list for pruning: index, snapshot
    bytes += e * sizeof(int32_t);                           // b
    bytes += math::constexpr_align(r, 4u) * sizeof(int32_t); // resident loads
    bytes += 2 * r * s * sizeof(int32_t);                   // previous and new slot tables
    bytes += math::constexpr_align(r * e, 4u);              // holds
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

// Everything the MegaMoE kernel needs to copy the changed redundant slots itself
// (`kInKernelWeightCopy`): the symmetric weight arena, its planes ordered L1 group
// first (`num_l1_planes` of them) then L2 group, and the planner's copy list,
// count and readiness counters inside the workspace. Only `num_resident_experts`
// and `num_redundant_slots` describe the slot geometry; the kernel's
// `kNumExpertsPerRank` equals their sum.
template <uint32_t kNumRanks = layout::kNumMaxRanks>
struct InKernelCopy {
    layout::SymBuffer<kNumRanks> arena;
    CopyPlanes planes;
    const int2* copy_list;
    // Cleared by the kernel once consumed, so a later launch without a new
    // plan (remap-only decode steps) copies nothing.
    int32_t* copy_count;
    uint32_t* copy_done;
    uint32_t num_resident_experts;
    uint32_t num_redundant_slots;
    uint32_t num_l1_planes;
};

} // namespace deep_gemm::eplb
