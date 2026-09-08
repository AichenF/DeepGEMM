#pragma once

#include <cstdint>

namespace deep_gemm::sm120_routed_moe {

struct Shape {
    static constexpr int kWorldSize = 8;
    static constexpr int kExperts = 256;
    static constexpr int kExpertsPerRank = kExperts / kWorldSize;
    static constexpr int kTopK = 6;
    static constexpr int kHidden = 4096;
    static constexpr int kIntermediate = 2048;
    static constexpr int kMaxRows = 8192;
    static constexpr int kTaskRows = 128;
    static constexpr int kMmaTileN = 128;
    static constexpr int kThreads = 384;
    static constexpr int kMaxGridCTAs = 110;
    static constexpr int kDynamicSharedMemoryBytes = 101376;
    static constexpr float kActivationClamp = 10.0f;
    static constexpr bool kFastMath = true;
};

struct WorkspaceLayout {
    static constexpr int kPoolRows = 397280;
    static constexpr int kMaxTasks = 3104;
    static constexpr int kMailboxEntries = 8192;
};

struct TraceLayout {
    static constexpr int kTimestampCount = 33;
};

struct ResultCodecLayout {
    static constexpr int kBlockValues = 64;
    static constexpr int kBlocksPerRow = Shape::kHidden / kBlockValues;
    static constexpr int kEncodedRowBytes = 6272;
    static constexpr int kEncodedRowWords = kEncodedRowBytes / 4;
    static constexpr int kRawValuesInBody = 48;
    static constexpr int kRawTailValues = kBlockValues - kRawValuesInBody;
    static constexpr int kRawTailBytes = kRawTailValues * 2;
    static constexpr int kRawBlocksPerRow = kBlocksPerRow;
    static constexpr int kRoutePitchBytes =
        kEncodedRowBytes + kRawBlocksPerRow * kRawTailBytes;
    static constexpr int kRoutePitchWords = kRoutePitchBytes / 4;
    static constexpr int kMaxRoutesPerPeer = Shape::kMaxRows * Shape::kTopK;
    static constexpr int kMaxChunksPerPeer = 769;
    static constexpr std::int64_t kRouteStorageBytes =
        static_cast<std::int64_t>(kMaxRoutesPerPeer) * kRoutePitchBytes;
    static constexpr int kSlotBytes =
        kRouteStorageBytes + kMaxRoutesPerPeer * sizeof(int);
    static constexpr int kSlotWords = kSlotBytes / 4;
    static constexpr int kRouteMapOffsetWords = kRouteStorageBytes / 4;
};

struct CommunicationLayout {
    static constexpr int kRingSlots = 2;
    static constexpr int kHeaderWords = 8;
    static constexpr int kHeaderSlotBytes =
        (kHeaderWords + 2 * Shape::kExpertsPerRank) * sizeof(std::int32_t);
    static constexpr int kDispatchRecordBytes = 4352;
    static constexpr std::int64_t kHeaderWindowBytes =
        Shape::kWorldSize * kRingSlots * kHeaderSlotBytes;
    static constexpr std::int64_t kPayloadWindowBytes =
        Shape::kWorldSize * kRingSlots * Shape::kMaxRows * kDispatchRecordBytes;
    static constexpr std::int64_t kResultWindowBytes =
        Shape::kWorldSize * kRingSlots *
        static_cast<std::int64_t>(ResultCodecLayout::kSlotBytes);
    static constexpr std::int64_t kAckWindowBytes = Shape::kWorldSize;

    static constexpr int kGinContextCount = 2;
    static constexpr int kDispatchChunks = 8;
    static constexpr int kDispatchSignalBase = 24;
    static constexpr int kGinSignalCount =
        kDispatchSignalBase + Shape::kWorldSize * kDispatchChunks;
    static constexpr int kWorldBarrierCount = 1;
};

static_assert(Shape::kExperts % Shape::kWorldSize == 0);
static_assert(Shape::kExpertsPerRank == 32);
static_assert(ResultCodecLayout::kRawBlocksPerRow >= ResultCodecLayout::kBlocksPerRow);
static_assert(ResultCodecLayout::kRawValuesInBody + ResultCodecLayout::kRawTailValues ==
              ResultCodecLayout::kBlockValues);
static_assert(ResultCodecLayout::kRawValuesInBody * 2 == 96);
static_assert(ResultCodecLayout::kRawTailBytes == 32);
static_assert(Shape::kTaskRows * ResultCodecLayout::kRawBlocksPerRow <= 32768);
static_assert(ResultCodecLayout::kRoutePitchBytes == 8320);
static_assert(ResultCodecLayout::kSlotBytes == 409141248);
static_assert(CommunicationLayout::kHeaderWindowBytes == 4608);
static_assert(CommunicationLayout::kPayloadWindowBytes == 570425344);
static_assert(CommunicationLayout::kResultWindowBytes == 6546259968);

} // namespace deep_gemm::sm120_routed_moe
