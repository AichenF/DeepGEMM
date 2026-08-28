// Per-CTA phase attribution for the fused SM120 MegaMoE kernel.
//
// CUPTI sees one kernel activity, so it can bound the iteration but cannot say
// where the time inside it goes. This tracer accumulates `%globaltimer` deltas
// per CTA per phase, which decomposes the critical path into dispatch, GIN
// service, W1, SwiGLU/requantization, W2, result return, combine, acknowledgement
// and the grid rendezvous waits between them.
//
// One thread per CTA reads the clock and accumulates into its own slot, so there
// is no contention and no cross-CTA serialization. Build with
// `-DCAKE_MOE_PHASE_TRACE=1` for diagnosis; formal performance runs use the
// default untraced build.

#pragma once

#include <cstdint>

#ifndef CAKE_MOE_PHASE_TRACE
#define CAKE_MOE_PHASE_TRACE 0
#endif

namespace cake_moe {
namespace trace {

enum Phase : int {
  kReset = 0,
  kEpochBaseline,
  kDispatch,
  kTaskBuild,
  kGroupedLayout,
  kValidate,
  kW1Chunk,
  kEpilogue,
  kW2Chunk,
  kWorkerScan,
  kWorkerIdle,
  kCombineWait,
  kCombine,
  kServicePrecombine,
  kServicePostcombine,
  kGridSync,
  kPhaseCount,
};

inline const char* phase_name(int phase) {
  switch (phase) {
    case kReset: return "reset";
    case kEpochBaseline: return "epoch_baseline";
    case kDispatch: return "dispatch";
    case kTaskBuild: return "task_build";
    case kGroupedLayout: return "grouped_layout";
    case kValidate: return "validate";
    case kW1Chunk: return "w1_chunk";
    case kEpilogue: return "swiglu_requant";
    case kW2Chunk: return "w2_chunk";
    case kWorkerScan: return "worker_claim";
    case kWorkerIdle: return "worker_idle";
    case kCombineWait: return "combine_result_wait";
    case kCombine: return "combine";
    case kServicePrecombine: return "gin_result_service";
    case kServicePostcombine: return "gin_ack_service";
    case kGridSync: return "grid_sync";
    default: return "unknown";
  }
}

#if CAKE_MOE_PHASE_TRACE
__device__ __forceinline__ unsigned long long now_ns() {
  unsigned long long stamp;
  asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(stamp) :: "memory");
  return stamp;
}
#endif

}  // namespace trace
}  // namespace cake_moe

#if CAKE_MOE_PHASE_TRACE

#define CAKE_PHASE_BEGIN(token)                                              \
  const unsigned long long token =                                           \
      ((int)threadIdx.x == 0) ? cake_moe::trace::now_ns() : 0ull

#define CAKE_PHASE_END(token, phase, params)                                 \
  do {                                                                       \
    if ((int)threadIdx.x == 0 && (params)->phase_ns != nullptr) {            \
      const unsigned long long cake_phase_end_stamp = cake_moe::trace::now_ns(); \
      const int cake_phase_slot =                                            \
          (int)blockIdx.x * cake_moe::trace::kPhaseCount + (phase);          \
      (params)->phase_ns[cake_phase_slot] += cake_phase_end_stamp - (token); \
      (params)->phase_count[cake_phase_slot] += 1u;                          \
    }                                                                        \
  } while (0)

// Takes the caller's own grid group so the untraced expansion is exactly the
// call site it replaced.
#define CAKE_PHASE_GRID_SYNC(grid_group, params)                             \
  do {                                                                       \
    CAKE_PHASE_BEGIN(cake_grid_sync_stamp);                                  \
    (grid_group).sync();                                                     \
    CAKE_PHASE_END(cake_grid_sync_stamp, cake_moe::trace::kGridSync, params);\
  } while (0)

#else

#define CAKE_PHASE_BEGIN(token) ((void)0)
#define CAKE_PHASE_END(token, phase, params) ((void)0)
#define CAKE_PHASE_GRID_SYNC(grid_group, params) (grid_group).sync()

#endif
