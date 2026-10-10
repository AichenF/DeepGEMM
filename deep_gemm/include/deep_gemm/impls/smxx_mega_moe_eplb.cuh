#pragma once

#include <cstdio>
#include <cuda_bf16.h>

#include <deep_gemm/common/math.cuh>
#include <deep_gemm/common/utils.cuh>
#include <deep_gemm/layout/sym_buffer.cuh>
#include <deep_gemm/layout/mega_moe_eplb.cuh>
#include <deep_gemm/ptx/ld_st.cuh>

// Dynamic expert load balancing for MegaMoE.
//
// Every rank owns `H = E / R` home experts in local slots [0, H) and `S`
// helper slots in [H, H + S). The MegaMoE kernel only ever sees physical slot
// ids `rank * (H + S) + local_slot`. Three kernels turn a logical routing
// table into a physical one:
//   * plan:   exchange per-rank expert histograms over symmetric memory and
//             run the same deterministic block-granular planner on every rank
//             (the Python reference is `deep_gemm/mega/eplb_planner.py`);
//   * copy:   fetch the weights of experts newly assigned to this rank's
//             helper slots from their owners over NVLink;
//   * remap:  rewrite `topk_idx` from logical experts to physical slots,
//             splitting each expert's token blocks across its copies.

namespace deep_gemm::eplb {

__device__ __forceinline__ void st_release_sys(uint32_t* ptr, const uint32_t& value) {
    asm volatile("st.release.sys.global.u32 [%0], %1;" :: "l"(ptr), "r"(value) : "memory");
}

// Lane-cooperative argmin over a 64-bit key; returns the winning key or
// `kNone` when no lane holds a candidate. Keys pack (primary, secondary).
static constexpr uint64_t kNoneKey = ~0ull;

__device__ __forceinline__ uint64_t warp_min_key(uint64_t key) {
    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        const auto other = __shfl_xor_sync(0xffffffffu, key, offset);
        key = other < key ? other : key;
    }
    return key;
}

__device__ __forceinline__ uint64_t make_key(const uint32_t& primary, const uint32_t& secondary) {
    return (static_cast<uint64_t>(primary) << 32) | secondary;
}

template <uint32_t kNumRanks, uint32_t kNumExperts, uint32_t kNumHelperSlots, uint32_t kNumThreads>
CUTLASS_GLOBAL void __launch_bounds__(kNumThreads, 1)
eplb_plan_kernel(const __grid_constant__ layout::SymBuffer<kNumRanks> ws,
                 const int64_t* __restrict__ topk_idx,
                 const uint32_t num_tokens, const uint32_t num_topk,
                 const uint32_t block_m, const uint32_t open_gate_blocks,
                 const uint32_t generation) {
    DG_STATIC_ASSERT(kNumRanks <= Workspace::kMaxRanks and kNumRanks <= 32, "Too many ranks");
    DG_STATIC_ASSERT(kNumExperts % kNumRanks == 0, "Experts must divide across ranks");
    constexpr uint32_t R = kNumRanks, E = kNumExperts, S = kNumHelperSlots;
    constexpr uint32_t H = E / R;
    const Workspace layout(R, E, S);
    const uint32_t thread_idx = threadIdx.x, lane = thread_idx & 31, warp_idx = thread_idx >> 5;
    const uint32_t rank = ws.rank_idx;
    const uint32_t parity = generation & 1;

    extern __shared__ __align__(16) uint8_t smem_buffer[];
    auto* hist = reinterpret_cast<int32_t*>(smem_buffer);                 // [R][E]
    auto* q = hist + R * E;                                               // [E][R]
    auto* rep_idx = q + E * R;                                            // [R * S]
    auto* rep_v = rep_idx + R * S;                                        // [R * S]
    auto* b = rep_v + R * S;                                              // [E]
    auto* load = b + E;                                                   // [R]
    auto* nrep = load + R;                                                // [R]
    auto* visited = nrep + R;                                             // [R]
    auto* prev_slot = visited + 2 * R;                                    // [R][S]
    auto* new_slot = prev_slot + R * S;                                   // [R][S]
    auto* holds = reinterpret_cast<uint8_t*>(new_slot + R * S);           // [R][E]
    auto* tried = holds + math::constexpr_align(R * E, 4u);               // [E]

    auto* ws_base = ws.template get_base_ptr<uint32_t*>();
    auto* flags = ws_base + layout.flags_offset() + parity * Workspace::kMaxRanks;
    auto* g_hist = reinterpret_cast<int32_t*>(ws_base + layout.hist_offset()) + parity * R * E;
    auto* g_slot_expert = reinterpret_cast<int32_t*>(ws_base + layout.slot_expert_offset());
    auto* g_copies = reinterpret_cast<int2*>(ws_base + layout.copies_offset());
    auto* g_expert_base = reinterpret_cast<int32_t*>(ws_base + layout.expert_base_offset());
    auto* g_local_counter = ws_base + layout.local_counter_offset();
    auto* g_copy_list = reinterpret_cast<int2*>(ws_base + layout.copy_list_offset());
    auto* g_copy_count = reinterpret_cast<int32_t*>(ws_base + layout.copy_count_offset());

    cudaGridDependencySynchronize();
#ifdef DG_EPLB_DEBUG
    if (thread_idx == 0)
        printf("EPLB plan entry: rank=%u generation=%u parity=%u num_tokens=%u block_m=%u\n", rank, generation, parity, num_tokens, block_m);
#endif

    // Local histogram
    for (uint32_t i = thread_idx; i < E; i += kNumThreads)
        hist[rank * E + i] = 0;
    __syncthreads();
    for (uint32_t i = thread_idx; i < num_tokens * num_topk; i += kNumThreads) {
        const auto e = __ldg(topk_idx + i);
        if (e >= 0)
            atomicAdd_block(hist + rank * E + static_cast<uint32_t>(e), 1);
    }
    __syncthreads();

    // Push it to every rank, then publish this generation
    for (uint32_t i = thread_idx; i < R * E; i += kNumThreads) {
        const uint32_t dst_rank = i / E, e = i % E;
        *ws.map(g_hist + rank * E + e, dst_rank) = hist[rank * E + e];
    }
    __syncthreads();
    if (thread_idx == 0) {
        __threadfence_system();
        for (uint32_t r = 0; r < R; ++ r)
            st_release_sys(ws.map(flags + rank, r), generation);
    }
    if (thread_idx < R) {
        const auto start_clock = clock64();
        while (ptx::ld_acq_sys(flags + thread_idx) < generation) {
            if (clock64() - start_clock >= 30ll * 2000000000ll) {
                printf("DeepGEMM EPLB plan timeout: rank=%u waiting for rank=%u generation=%u\n",
                       rank, thread_idx, generation);
                asm volatile("trap;");
            }
        }
    }
    __syncthreads();
    for (uint32_t i = thread_idx; i < R * E; i += kNumThreads)
        hist[i] = ptx::ld_volatile(reinterpret_cast<const uint32_t*>(g_hist + i));
    for (uint32_t i = thread_idx; i < R * S; i += kNumThreads)
        prev_slot[i] = g_slot_expert[i];
    for (uint32_t i = thread_idx; i < R * E; i += kNumThreads)
        holds[i] = 0;
    __syncthreads();
    for (uint32_t i = thread_idx; i < R * S; i += kNumThreads) {
        const auto e = prev_slot[i];
        if (e >= 0)
            holds[(i / S) * E + e] = 1;
    }
    for (uint32_t e = thread_idx; e < E; e += kNumThreads) {
        int32_t total = 0;
        #pragma unroll 4
        for (uint32_t r = 0; r < R; ++ r)
            total += hist[r * E + e];
        b[e] = (total + block_m - 1) / block_m;
        for (uint32_t r = 0; r < R; ++ r)
            q[e * R + r] = r == e / H ? b[e] : 0;
        holds[(e / H) * E + e] = 1;
    }
    __syncthreads();

    // The sequential planner runs on warp 0; everything else waits.
    if (warp_idx == 0) {
        // Every expert starts on its owner, so a rank's load is the block sum
        // of its home range.
        if (lane < R) {
            int32_t sum = 0;
            for (uint32_t e = lane * H; e < (lane + 1) * H; ++ e)
                sum += b[e];
            load[lane] = sum;
            nrep[lane] = 0;
        }
        int32_t total_blocks = 0;
        for (uint32_t e = lane; e < E; e += 32)
            total_blocks += b[e];
        total_blocks = __reduce_add_sync(0xffffffffu, total_blocks);
        const int32_t tau = (total_blocks + R - 1) / R;
        const int32_t gate = open_gate_blocks > 1 ? static_cast<int32_t>(open_gate_blocks) : 1;
        __syncwarp();

        auto holds_fn = [&](const uint32_t& r, const uint32_t& e) {
            return q[e * R + r] > 0 or holds[r * E + e] != 0;
        };
        auto move = [&](const uint32_t& e, const uint32_t& from, const uint32_t& to, const int32_t& v) {
            if (lane == 0) {
                if (q[e * R + to] == 0 and to != e / H)
                    nrep[to] += 1;
                q[e * R + from] -= v;
                q[e * R + to] += v;
                load[from] -= v;
                load[to] += v;
            }
            __syncwarp();
        };

        // Greedy rounds: shift blocks from overloaded owners towards tau.
        for (uint32_t round = 0; round < S + 2; ++ round) {
            bool moved = false;
            if (lane < R)
                visited[lane] = 0;
            __syncwarp();
            for (uint32_t k = 0; k < R; ++ k) {
                uint64_t key = kNoneKey;
                if (lane < R and not visited[lane] and load[lane] > tau)
                    key = make_key(static_cast<uint32_t>(0x7fffffff - (load[lane] - tau)), lane);
                key = warp_min_key(key);
                if (key == kNoneKey)
                    break;
                const uint32_t d = static_cast<uint32_t>(key);
                if (lane == 0)
                    visited[d] = 1;
                for (uint32_t e = d * H + lane; e < (d + 1) * H; e += 32)
                    tried[e] = 0;
                __syncwarp();
                while (load[d] > tau) {
                    uint64_t best = kNoneKey;
                    for (uint32_t e = d * H + lane; e < (d + 1) * H; e += 32) {
                        if (not tried[e] and q[e * R + d] > 0) {
                            const auto cand = make_key(static_cast<uint32_t>(0x7fffffff - q[e * R + d]), e);
                            best = cand < best ? cand : best;
                        }
                    }
                    best = warp_min_key(best);
                    if (best == kNoneKey)
                        break;
                    const uint32_t e = static_cast<uint32_t>(best);
                    if (lane == 0)
                        tried[e] = 1;
                    __syncwarp();
                    const int32_t avail = q[e * R + d];
                    // Existing copies first: active replicas and cached slots.
                    uint64_t pick = kNoneKey;
                    int32_t v = 0;
                    if (lane < R and lane != d and holds_fn(lane, e)) {
                        const int32_t room = tau - load[lane];
                        v = avail < room ? avail : room;
                        if (v > 0 and (q[e * R + lane] > 0 or nrep[lane] < static_cast<int32_t>(S)))
                            pick = make_key(static_cast<uint32_t>(load[lane]), lane);
                    }
                    uint64_t chosen = warp_min_key(pick);
                    if (chosen == kNoneKey) {
                        pick = kNoneKey;
                        v = 0;
                        if (lane < R and lane != d and not holds_fn(lane, e) and load[lane] < tau and
                            nrep[lane] < static_cast<int32_t>(S)) {
                            const int32_t room = tau - load[lane];
                            v = avail < room ? avail : room;
                            if (v >= gate)
                                pick = make_key(static_cast<uint32_t>(load[lane]), lane);
                        }
                        chosen = warp_min_key(pick);
                    }
                    if (chosen == kNoneKey)
                        continue;
                    const uint32_t to = static_cast<uint32_t>(chosen);
                    const int32_t move_v = __shfl_sync(0xffffffffu, v, to);
                    move(e, d, to, move_v);
                    moved = true;
                }
            }
            if (not moved)
                break;
        }

        // Pruning: a replica whose allocation fits elsewhere under the current
        // maximum load is folded back, owner first, then the least loaded copy.
        int32_t max_load = lane < R ? load[lane] : 0;
        max_load = __reduce_max_sync(0xffffffffu, max_load);
        // Each rank hosts at most S replicas, so the list fits R * S entries;
        // one lane per rank collects its own.
        constexpr uint32_t kNumReplicaSlots = R * S;
        constexpr uint32_t kNumReplicasPerLane = math::constexpr_ceil_div(kNumReplicaSlots, 32u);
        if (lane < R) {
            uint32_t n = 0;
            for (uint32_t e = 0; e < E and n < S; ++ e) {
                if (e / H != lane and q[e * R + lane] > 0) {
                    rep_idx[lane * S + n] = static_cast<int32_t>(e * R + lane);
                    rep_v[lane * S + n] = q[e * R + lane];
                    ++ n;
                }
            }
            for (; n < S; ++ n)
                rep_v[lane * S + n] = 0;
        }
        __syncwarp();
        while (true) {
            uint64_t best = kNoneKey;
            #pragma unroll
            for (uint32_t j = 0; j < kNumReplicasPerLane; ++ j) {
                const uint32_t k = lane + 32 * j;
                if (k < kNumReplicaSlots and rep_v[k] > 0) {
                    // Order by (snapshot allocation, expert, rank); both fit in 32 bits.
                    const auto cand = make_key(static_cast<uint32_t>(rep_v[k]), static_cast<uint32_t>(rep_idx[k]));
                    best = cand < best ? cand : best;
                }
            }
            best = warp_min_key(best);
            if (best == kNoneKey)
                break;
            const uint32_t i = static_cast<uint32_t>(best), e = i / R, r = i % R;
            const int32_t v = static_cast<int32_t>(best >> 32);
            if (lane < R) {
                for (uint32_t n = 0; n < S; ++ n)
                    if (rep_idx[lane * S + n] == static_cast<int32_t>(i) and rep_v[lane * S + n] == v)
                        rep_v[lane * S + n] = 0;
            }
            __syncwarp();
            if (q[i] != v)
                continue;
            const uint32_t owner = e / H;
            uint32_t target = ~0u;
            if (load[owner] + v <= max_load) {
                target = owner;
            } else {
                uint64_t pick = kNoneKey;
                if (lane < R and lane != r and lane != owner and q[e * R + lane] > 0 and
                    load[lane] + v <= max_load)
                    pick = make_key(static_cast<uint32_t>(load[lane]), lane);
                pick = warp_min_key(pick);
                if (pick != kNoneKey)
                    target = static_cast<uint32_t>(pick);
            }
            if (target != ~0u)
                move(e, r, target, v);
        }

        // Single-block moves between the extremes while they share an expert.
        for (uint32_t iter = 0; iter < E * R; ++ iter) {
            uint64_t hi_key = lane < R ? make_key(static_cast<uint32_t>(0x7fffffff - load[lane]), lane) : kNoneKey;
            uint64_t lo_key = lane < R ? make_key(static_cast<uint32_t>(load[lane]), lane) : kNoneKey;
            hi_key = warp_min_key(hi_key);
            lo_key = warp_min_key(lo_key);
            const uint32_t hi = static_cast<uint32_t>(hi_key), lo = static_cast<uint32_t>(lo_key);
            if (load[hi] - load[lo] <= 1)
                break;
            uint32_t shared = ~0u;
            for (uint32_t e = lane; e < E; e += 32) {
                if (q[e * R + hi] > 0 and q[e * R + lo] > 0) {
                    shared = e;
                    break;
                }
            }
            shared = __reduce_min_sync(0xffffffffu, shared);
            if (shared == ~0u)
                break;
            move(shared, hi, lo, 1);
        }

        // Helper slot assignment, one lane per rank: keep cached slots that are
        // still used, then fill free slots with new experts in expert order.
        if (lane < R) {
            const uint32_t r = lane;
            for (uint32_t s = 0; s < S; ++ s) {
                const auto e = prev_slot[r * S + s];
                new_slot[r * S + s] = (e >= 0 and q[e * R + r] > 0) ? e : -1;
            }
            for (uint32_t e = 0; e < E; ++ e) {
                if (e / H == r or q[e * R + r] == 0)
                    continue;
                bool present = false;
                for (uint32_t s = 0; s < S; ++ s)
                    present |= new_slot[r * S + s] == static_cast<int32_t>(e);
                if (present)
                    continue;
                for (uint32_t s = 0; s < S; ++ s) {
                    if (new_slot[r * S + s] < 0) {
                        new_slot[r * S + s] = static_cast<int32_t>(e);
                        break;
                    }
                }
            }
        }
        __syncwarp();

        // Publish: copies, ordinal bases, counters, slot table and copy list.
        for (uint32_t e = lane; e < E; e += 32) {
            const uint32_t owner = e / H;
            uint32_t k = 0;
            int32_t cum = q[e * R + owner];
            g_copies[e * R + k ++] = make_int2(static_cast<int32_t>(owner), cum);
            for (uint32_t r = 0; r < R; ++ r) {
                if (r != owner and q[e * R + r] > 0) {
                    cum += q[e * R + r];
                    g_copies[e * R + k ++] = make_int2(static_cast<int32_t>(r), cum);
                }
            }
            if (k < R)
                g_copies[e * R + k] = make_int2(-1, cum);
            int32_t base = 0;
            for (uint32_t r = 0; r < rank; ++ r)
                base += hist[r * E + e];
            g_expert_base[e] = base;
            g_local_counter[e] = 0;
        }
        if (lane == 0) {
            int32_t count = 0;
            for (uint32_t s = 0; s < S; ++ s) {
                const auto e = new_slot[rank * S + s];
                if (e >= 0 and e != prev_slot[rank * S + s])
                    g_copy_list[count ++] = make_int2(static_cast<int32_t>(s), e);
            }
            *g_copy_count = count;
        }
        for (uint32_t i = lane; i < R * S; i += 32)
            g_slot_expert[i] = new_slot[i];
    }
}

template <uint32_t kNumRanks, uint32_t kNumExperts, uint32_t kNumHelperSlots, uint32_t kNumThreads>
CUTLASS_GLOBAL void __launch_bounds__(kNumThreads)
eplb_remap_kernel(uint32_t* __restrict__ ws_base, const uint32_t rank,
                  int64_t* topk_idx, const uint32_t num_tokens, const uint32_t num_topk,
                  const uint32_t block_m) {
    constexpr uint32_t R = kNumRanks, E = kNumExperts, S = kNumHelperSlots;
    constexpr uint32_t H = E / R;
    const Workspace layout(R, E, S);
    const auto* slot_expert = reinterpret_cast<const int32_t*>(ws_base + layout.slot_expert_offset());
    const auto* copies = reinterpret_cast<const int2*>(ws_base + layout.copies_offset());
    const auto* expert_base = reinterpret_cast<const int32_t*>(ws_base + layout.expert_base_offset());
    auto* local_counter = ws_base + layout.local_counter_offset();

    cudaGridDependencySynchronize();

    const uint32_t total = num_tokens * num_topk;
    for (uint32_t i = blockIdx.x * kNumThreads + threadIdx.x; i < total; i += gridDim.x * kNumThreads) {
        const auto e64 = topk_idx[i];
        if (e64 < 0)
            continue;
        const uint32_t e = static_cast<uint32_t>(e64);
        const uint32_t ordinal = expert_base[e] + atomicAdd(local_counter + e, 1u);
        const int32_t block = static_cast<int32_t>(ordinal / block_m);
        int32_t dest = e / H;
        #pragma unroll 1
        for (uint32_t k = 0; k < R; ++ k) {
            const auto copy = copies[e * R + k];
            if (copy.x < 0)
                break;
            dest = copy.x;
            if (block < copy.y)
                break;
        }
        uint32_t local_slot = e - dest * H;
        if (static_cast<uint32_t>(dest) != e / H) {
            local_slot = H;
            for (uint32_t s = 0; s < S; ++ s)
                local_slot = slot_expert[dest * S + s] == static_cast<int32_t>(e) ? H + s : local_slot;
        }
        topk_idx[i] = static_cast<int64_t>(dest) * (H + S) + local_slot;
    }
}

template <uint32_t kNumRanks, uint32_t kNumThreads>
CUTLASS_GLOBAL void __launch_bounds__(kNumThreads)
eplb_copy_weights_kernel(const __grid_constant__ layout::SymBuffer<kNumRanks> arena,
                         const __grid_constant__ CopyPlanes planes,
                         const int2* __restrict__ copy_list,
                         const int32_t* __restrict__ copy_count,
                         const uint32_t num_home_experts) {
    cudaGridDependencySynchronize();
    if (static_cast<int32_t>(blockIdx.x) >= *copy_count)
        return;
    const auto entry = copy_list[blockIdx.x];
    const uint32_t dst_slot = num_home_experts + entry.x;
    const uint32_t expert = entry.y;
    const uint32_t owner = expert / num_home_experts;
    const uint32_t src_slot = expert - owner * num_home_experts;
    auto* base = arena.template get_base_ptr<uint8_t*>();

    for (uint32_t p = 0; p < planes.num_planes; ++ p) {
        const auto& plane = planes.planes[p];
        const uint8_t* src = arena.map(base + plane.offset + static_cast<uint64_t>(src_slot) * plane.bytes_per_slot, owner);
        uint8_t* dst = base + plane.offset + static_cast<uint64_t>(dst_slot) * plane.bytes_per_slot;
        const uint32_t stride = gridDim.y * kNumThreads;
        const uint32_t start = blockIdx.y * kNumThreads + threadIdx.x;
        if (plane.vec_bytes == 16) {
            const auto n = plane.bytes_per_slot / 16;
            for (uint32_t i = start; i < n; i += stride)
                reinterpret_cast<uint4*>(dst)[i] = __ldg(reinterpret_cast<const uint4*>(src) + i);
        } else {
            const auto n = plane.bytes_per_slot / 4;
            for (uint32_t i = start; i < n; i += stride)
                reinterpret_cast<uint32_t*>(dst)[i] = __ldg(reinterpret_cast<const uint32_t*>(src) + i);
        }
    }
}

} // namespace deep_gemm::eplb
