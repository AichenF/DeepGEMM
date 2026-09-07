// Exact Hopper/H20 outer-pipeline experiment.
//
// The MXFP4 math, TP-local workspace protocol, ordered route reduction, and
// embedded TP communication remain in the proven native body.  This include
// makes the outer scheduling contract explicit and compile-time checked so
// the experiment cannot silently fall back to the production interleaved,
// two-CTA/register-dequant specialization.
DG_STATIC_ASSERT(K_NATIVE_H20_EXACT_OUTER,
                 "exact H20 body requires its dedicated JIT identity");
DG_STATIC_ASSERT(kNumSMs == 78, "exact H20 body requires one CTA per H20 SM");
DG_STATIC_ASSERT(kNumDispatchThreads == 64 &&
                 kNumNonEpilogueThreads == 64 &&
                 kNumEpilogueThreads == 256,
                 "exact H20 body requires the 64/64/256 role split");
DG_STATIC_ASSERT(!kUseInterleavedScheduler,
                 "exact H20 body requires the reference static scheduler");
DG_STATIC_ASSERT(!K_NATIVE_REGISTER_DEQUANT,
                 "exact H20 body requires shared-memory Mode2 decode");
DG_STATIC_ASSERT(!K_NATIVE_TWO_CTA_PER_SM,
                 "exact H20 body requires one resident CTA per SM");
DG_STATIC_ASSERT(kNumStages == 4,
                 "DeepSeek-V4-Flash TP small-M exact plan is four-stage");
DG_STATIC_ASSERT(kSwapABRequested && kUseMode2RowDecoder,
                 "exact H20 body requires swap-AB and Mode2 decode");
DG_STATIC_ASSERT(kSingleActiveDispatchWarp,
                 "exact H20 small-M plan requires one active dispatch warp");
DG_STATIC_ASSERT(kNumExpertsPerWave == 16 || kNumExpertsPerWave == 32,
                 "exact H20 TP plan uses EPW16 or EPW32");

#include "v4_flash_tp_native_body.inl"
