#pragma once

#include <cstdint>

namespace deep_gemm::sm120_routed_moe {

template <int WorldSize>
struct ShapeT {
    static_assert(WorldSize == 4 or WorldSize == 8);
    static constexpr int kWorldSize = WorldSize;
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

using Shape = ShapeT<8>;

template <int WorldSize>
struct WorkspaceLayoutT {
    using Shape = ShapeT<WorldSize>;
    static constexpr int kRouteCandidates =
        Shape::kWorldSize * Shape::kMaxRows * Shape::kTopK;
    static constexpr int kPoolRows =
        kRouteCandidates + Shape::kExpertsPerRank * (Shape::kTaskRows - 1);
    static constexpr int kMaxTasks =
        (kRouteCandidates + Shape::kTaskRows - 1) / Shape::kTaskRows +
        Shape::kExpertsPerRank;
    static constexpr int kMailboxEntries = 8192;
};

using WorkspaceLayout = WorkspaceLayoutT<8>;

struct TraceLayout {
    static constexpr int kTimestampCount = 33;
};

template <int WorldSize>
struct ResultCodecLayoutT {
    using Shape = ShapeT<WorldSize>;
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

using ResultCodecLayout = ResultCodecLayoutT<8>;

template <int WorldSize>
struct CommunicationLayoutT {
    using Codec = ResultCodecLayoutT<WorldSize>;
    using Shape = ShapeT<WorldSize>;
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
        static_cast<std::int64_t>(Codec::kSlotBytes);
    static constexpr std::int64_t kAckWindowBytes = Shape::kWorldSize;

    static constexpr int kGinContextCount = 2;
    static constexpr int kDispatchChunks = 8;
    static constexpr int kHeaderSignalBase = 0;
    static constexpr int kResultSignalBase = Shape::kWorldSize;
    static constexpr int kAckSignalBase = 2 * Shape::kWorldSize;
    static constexpr int kDispatchSignalBase = 3 * Shape::kWorldSize;
    static constexpr int kResultSecondContextSignalBase =
        kDispatchSignalBase + Shape::kWorldSize * kDispatchChunks;
    static constexpr int kGinSignalCount = WorldSize == 4 ?
        kResultSecondContextSignalBase + Shape::kWorldSize :
        kDispatchSignalBase + Shape::kWorldSize * kDispatchChunks;
    static constexpr int kWorldBarrierCount = 1;
};

using CommunicationLayout = CommunicationLayoutT<8>;

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
static_assert(CommunicationLayout::kGinSignalCount == 88);

using ShapeEP4 = ShapeT<4>;
using WorkspaceLayoutEP4 = WorkspaceLayoutT<4>;
using ResultCodecLayoutEP4 = ResultCodecLayoutT<4>;
using CommunicationLayoutEP4 = CommunicationLayoutT<4>;

static_assert(ShapeEP4::kExpertsPerRank == 64);
static_assert(WorkspaceLayoutEP4::kPoolRows == 204736);
static_assert(WorkspaceLayoutEP4::kMaxTasks == 1600);
static_assert(CommunicationLayoutEP4::kHeaderWindowBytes == 4352);
static_assert(CommunicationLayoutEP4::kPayloadWindowBytes == 285212672);
static_assert(CommunicationLayoutEP4::kResultWindowBytes == 3273129984);
static_assert(CommunicationLayoutEP4::kGinSignalCount == 48);

} // namespace deep_gemm::sm120_routed_moe
