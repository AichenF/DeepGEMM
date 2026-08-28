// Single source of shape truth for the SM120 MegaMoE family.
//
// The kernel, the correctness host and the performance host all derive every
// window size, task bound, tile count and protocol stride from the knobs
// below, so one translation unit serves every supported routed-expert shape
// and every expert-parallel width. Override a knob on the nvcc command line,
// for example `-DCAKE_MOE_LOCAL_EXPERTS=64 -DCAKE_MOE_PHYSICAL_RANKS=4` for
// EP4 at 256 global experts.
//
// The defaults describe DeepSeek V4 Flash routed experts on eight ranks.

#pragma once

#include <cstddef>
#include <cstdint>

#ifndef CAKE_MOE_HIDDEN
#define CAKE_MOE_HIDDEN 4096
#endif

#ifndef CAKE_MOE_INTERMEDIATE
#define CAKE_MOE_INTERMEDIATE 4096
#endif

#ifndef CAKE_MOE_OUTPUT
#define CAKE_MOE_OUTPUT CAKE_MOE_HIDDEN
#endif

#ifndef CAKE_MOE_TOPK
#define CAKE_MOE_TOPK 6
#endif

#ifndef CAKE_MOE_LOCAL_EXPERTS
#define CAKE_MOE_LOCAL_EXPERTS 32
#endif

#ifndef CAKE_MOE_PHYSICAL_RANKS
#define CAKE_MOE_PHYSICAL_RANKS 8
#endif

#ifndef CAKE_MOE_MAX_ROWS
#define CAKE_MOE_MAX_ROWS 2048
#endif

#ifndef CAKE_MOE_RING_SLOTS
#define CAKE_MOE_RING_SLOTS 2
#endif

#ifndef CAKE_MOE_COMBINE_CTAS
#define CAKE_MOE_COMBINE_CTAS 110
#endif

namespace cake_moe {

// ---------------------------------------------------------------- knobs ---

constexpr int kHidden = CAKE_MOE_HIDDEN;
constexpr int kIntermediate = CAKE_MOE_INTERMEDIATE;
constexpr int kOutput = CAKE_MOE_OUTPUT;
constexpr int kTopK = CAKE_MOE_TOPK;
constexpr int kLocalExperts = CAKE_MOE_LOCAL_EXPERTS;
constexpr int kPhysicalRanks = CAKE_MOE_PHYSICAL_RANKS;
constexpr int kMaxRows = CAKE_MOE_MAX_ROWS;
constexpr int kRingSlots = CAKE_MOE_RING_SLOTS;
constexpr int kCombineCtas = CAKE_MOE_COMBINE_CTAS;

// ------------------------------------------------------- math tile shape ---

// The canonical SM120 donor is instantiated at these tile bounds. `kTaskM` is
// both the donor `BLOCK_M` and the row-padding granularity of one task.
constexpr int kTaskM = 64;
constexpr int kBlockN = 128;
constexpr int kBlockK = 128;
constexpr int kNumStages = 3;
constexpr int kNumTmaThreads = 128;
constexpr int kNumMathThreads = 256;
constexpr int kWorkerCtas = kCombineCtas - 1;

// Scale-factor granularity along K for both operands (UE8M0 over 32 values).
constexpr int kScaleGranularityK = 32;

// ------------------------------------------------------- expert topology ---

constexpr int kGlobalExperts = kPhysicalRanks * kLocalExperts;

// -------------------------------------------------------- routing bounds ---

constexpr int kMaxRoutesPerPeer = kMaxRows * kTopK;
constexpr int kMaxRoutesAllPeers = kPhysicalRanks * kMaxRoutesPerPeer;

// Worst case padding: every expert may hold a partially filled final task.
constexpr int kMaxTasks = kMaxRoutesAllPeers / kTaskM + kLocalExperts;
constexpr int kMaxPaddedRows = kMaxTasks * kTaskM;

// --------------------------------------------------- dispatch record ABI ---

constexpr int kHeaderWords = 8;
constexpr int kHeaderBytes = 4 * kHeaderWords;

// One dispatch record carries a 128-byte header, the MXFP8 E4M3 activation row
// and its UE8M0 K32 scale vector, padded to a 128-byte boundary.
constexpr int kRecordHeaderBytes = 128;
constexpr int kActivationBytesPerRow = kHidden;
constexpr int kActivationScaleBytesPerRow = kHidden / kScaleGranularityK;
constexpr int kRecordPayloadBytes =
    kRecordHeaderBytes + kActivationBytesPerRow + kActivationScaleBytesPerRow;
constexpr int kRecordBytes = (kRecordPayloadBytes + 127) / 128 * 128;

constexpr int kRecordWords = kRecordBytes / 4;
constexpr int kRecordHeaderWords = kRecordHeaderBytes / 4;
constexpr int kActivationWordsPerRow = kActivationBytesPerRow / 4;
constexpr int kActivationScaleWordsPerRow = kActivationScaleBytesPerRow / 4;
constexpr int kRecordScaleWordOffset =
    kRecordHeaderWords + kActivationWordsPerRow;

// ------------------------------------------------------ transport windows ---

constexpr std::size_t kDispatchPeerSlotBytes =
    static_cast<std::size_t>(kMaxRows) * kRecordBytes;
constexpr std::size_t kDispatchWindowBytes =
    static_cast<std::size_t>(kPhysicalRanks) * kRingSlots * kDispatchPeerSlotBytes;

constexpr std::size_t kResultElementsPerPeer =
    static_cast<std::size_t>(kMaxRoutesPerPeer) * kOutput;
constexpr std::size_t kResultWindowElements =
    static_cast<std::size_t>(kPhysicalRanks) * kRingSlots * kResultElementsPerPeer;

// Largest addressable result element, used by the fail-closed bound checks.
constexpr std::size_t kMaxResultElementIndex =
    static_cast<std::size_t>(kPhysicalRanks * kRingSlots - 1) * kResultElementsPerPeer +
    static_cast<std::size_t>(kMaxRoutesPerPeer - 1) * kOutput +
    static_cast<std::size_t>(kOutput - 1);

// ------------------------------------------------------------ GEMM shape ---

// W1 fuses the gate and up projections into one physical N extent.
constexpr int kW1PhysicalN = 2 * kIntermediate;
constexpr int kW1ShapeK = kHidden;
constexpr int kW2ShapeN = kOutput;
constexpr int kW2ShapeK = kIntermediate;

// Physical N128 tiles that one M64 task must retire in each stage.
constexpr int kW1TilesPerTask = kW1PhysicalN / kBlockN;
constexpr int kW2TilesPerTask = kW2ShapeN / kBlockN;

// Work is claimed in groups of eight adjacent physical tiles.
constexpr int kTilesPerChunk = 8;
constexpr int kW1ChunksPerTask = (kW1TilesPerTask + kTilesPerChunk - 1) / kTilesPerChunk;
constexpr int kW2ChunksPerTask = (kW2TilesPerTask + kTilesPerChunk - 1) / kTilesPerChunk;

// --------------------------------------------------------- weight bounds ---

constexpr int kW1KBlocks = kHidden / kBlockK;
constexpr int kW2KBlocks = kIntermediate / kBlockK;

constexpr std::size_t kW1WeightBytes =
    static_cast<std::size_t>(kLocalExperts) * kW1PhysicalN * kHidden / 2;
constexpr std::size_t kW1WeightSfWords =
    static_cast<std::size_t>(kLocalExperts) * kW1KBlocks * kW1PhysicalN;
constexpr std::size_t kW2WeightBytes =
    static_cast<std::size_t>(kLocalExperts) * kOutput * kIntermediate / 2;
constexpr std::size_t kW2WeightSfWords =
    static_cast<std::size_t>(kLocalExperts) * kW2KBlocks * kOutput;

// ------------------------------------------------------ activation pools ---

constexpr std::size_t kPoolActivationWords =
    static_cast<std::size_t>(kMaxPaddedRows) * kActivationWordsPerRow;
constexpr std::size_t kPoolScaleWords =
    static_cast<std::size_t>(kMaxPaddedRows) * kActivationScaleWordsPerRow;

constexpr std::size_t kIntermediateBytesPerTask =
    static_cast<std::size_t>(kTaskM) * kIntermediate;
constexpr std::size_t kIntermediateSfBytesPerTask =
    static_cast<std::size_t>(kTaskM) * (kIntermediate / kScaleGranularityK);

// --------------------------------------------------------- static checks ---

static_assert(kHidden % 128 == 0, "hidden size must be a multiple of 128");
static_assert(kIntermediate % 128 == 0,
              "intermediate size must be a multiple of 128");
static_assert(kOutput % 128 == 0, "output size must be a multiple of 128");
static_assert(kHidden % kScaleGranularityK == 0,
              "hidden size must cover whole UE8M0 scale groups");
static_assert(kW1PhysicalN % kBlockN == 0,
              "fused gate/up extent must tile evenly by BLOCK_N");
static_assert(kW2ShapeN % kBlockN == 0,
              "output extent must tile evenly by BLOCK_N");
static_assert(kRecordBytes % 128 == 0, "dispatch records stay 128-byte aligned");
static_assert(kMaxRows > 0 && kTopK > 0 && kLocalExperts > 0,
              "routing extents must be positive");
static_assert(kPhysicalRanks > 0 && (kPhysicalRanks & (kPhysicalRanks - 1)) == 0,
              "physical rank capacity must be a power of two");

}  // namespace cake_moe
