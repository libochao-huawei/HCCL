#ifndef HCCL_ALLGATHERBATCH_COMMON_H
#define HCCL_ALLGATHERBATCH_COMMON_H

#include <chrono>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <vector>

#include "acl/acl_rt.h"
#include "hccl/hccl_comm.h"
#include "hccl/hccl_res.h"
#include "hccl/hccl_types.h"
#include "hccl/hcomm_primitives.h"
#include "log.h"

namespace ops_hccl_allgatherbatch {

constexpr uint32_t kAllGatherBatchMaxItems = 8;
constexpr uint32_t kAllGatherBatchControlNotifyNum = 2;
constexpr uint32_t kAllGatherBatchControlNotifyStart = 0;
constexpr uint32_t kAllGatherBatchControlNotifyDone = 1;
constexpr uint32_t SubThreadNum = 3;
constexpr uint32_t kAllGatherBatchNotifyIdxAck = 0;
constexpr uint32_t kAllGatherBatchNotifyIdxDataSignal = 1;
constexpr uint32_t kAllGatherBatchNotifyIdxFinAck = 2;
constexpr u32 NOTIFY_IDX_ACK = 0;
constexpr u32 NOTIFY_IDX_DATA_SIGNAL = 1;
constexpr u32 NOTIFY_IDX_FIN_ACK = 2;
constexpr uint32_t CUSTOM_TIMEOUT = 1800;
constexpr uint32_t kAllGatherBatchOpNameLength = 64;
constexpr uint32_t kAllGatherBatchTagLength = HCCL_RES_TAG_MAX_LEN + 1;
constexpr char kAllGatherBatchCtxTag[] = "allgatherbatch";
constexpr char kAllGatherBatchKernelName[] = "HcclAllGatherBatchAicpuKernel";

enum class BatchKernelOpType : uint32_t {
    kAllGatherBatch = 1,
};

enum class BatchCommMode : uint32_t {
    kUnknown = 0,
    kSingleServer = 1,
    kCrossServer = 2,
};

/**
 * @enum HcclMemType
 * @brief 内存类型枚举定义
 */
typedef enum {
    HCCL_MEM_TYPE_DEVICE, ///< 设备侧内存（如NPU等）
    HCCL_MEM_TYPE_HOST,   ///< 主机侧内存
    HCCL_MEM_TYPE_NUM     ///< 内存类型数量
} HcclMemType;

struct HcclMem {
    HcclMemType type = HcclMemType::HCCL_MEM_TYPE_DEVICE;
    void* addr = nullptr;
    uint64_t size = 0;
};

struct ExecMem {
    u64 count{0};
    HcclDataType dataType{0};
    HcclMem inputMem;           /* 单算子模式时是InCCLMem, 图模式时是InUserMem */
    HcclMem outputMem;          /* 单算子模式时是OutCCLMem, 图模式时是OutUserMem */
    HcclMem scratchMem;
    void *inputPtr = nullptr;   /* InUserMem的地址，图模式时与inputMem的地址相同 */
    void *outputPtr = nullptr;  /* OutUserMem的地址，图模式时与outputMem的地址相同 */
};

struct Slice {
    u64 offset{0}; // Slice相对于input/output的偏移字节数，gather类操作取output，scatter类操作取input
    u64 size{0};    // Slice的数据大小，单位：字节
};

struct CommBuffer {
    void *addr = nullptr;
    uint64_t size = 0;
    uint64_t offset = 0;
};

struct ChannelResource {
    ChannelHandle handle = 0;
    uint32_t remoteRank = 0;
    uint32_t remoteServerIdx = 0;
    uint32_t remoteSuperPodIdx = 0;
    CommProtocol protocol = COMM_PROTOCOL_RESERVED;
    uint32_t localNotifyIdx = 0;
    uint32_t remoteNotifyIdx = 0;
    CommBuffer remoteBuffer {};
};

struct BatchTopoInfo {
    uint32_t rank = 0;
    uint32_t rankSize = 0;
    uint32_t serverIdx = 0;
    uint32_t serverCount = 0;
    uint32_t superPodIdx = 0;
    uint32_t reserved = 0;
};

// Host 侧准备并拷到 Device 的资源上下文。
// 采用“固定头 + 变长尾部 channel 区”的布局，避免固定 channel 上限。
struct AlgResourceCtx {
    // threadHandle 仅作为旧控制链兼容字段保留，正式资源合同以 mainThreadHandle 为准。
    ThreadHandle threadHandle = 0;
    ThreadHandle mainThreadHandle = 0;
    uint32_t lastTwoWorkerCount = 0;
    uint32_t reserved0 = 0;
    ThreadHandle subThreadHandles[SubThreadNum] = {0};
    uint32_t mainNotifyIds[SubThreadNum] = {0};
    uint32_t subNotifyIds[SubThreadNum] = {0};
    uint32_t channelCount = 0;
    uint32_t channelOffset = 0;
    CommBuffer localBuffer {};
};

struct BatchCallProfiling {
    uint32_t rank = 0;
    uint32_t rankSize = 0;
    uint32_t itemCount = 0;
    uint32_t windowCount = 0;
    BatchCommMode commMode = BatchCommMode::kUnknown;
    uint64_t totalInputBytes = 0;
    uint64_t localBufferBytes = 0;
    uint64_t maxWindowBytes = 0;
    uint64_t kernelUs = 0;
    uint64_t execUs = 0;
    uint64_t packUs = 0;
    uint64_t hdStageUs = 0;
    uint64_t unpackUs = 0;
};

struct BatchItemParam {
    void *sendBuf = nullptr;
    void *recvBuf = nullptr;
    uint64_t sendCount = 0;
    HcclDataType dataType = HCCL_DATA_TYPE_RESERVED;
    uint64_t elementSize = 0;
    uint64_t sendBytes = 0;
};

// Host 下发到 Device 的 launch 参数。
// 里面包含拓扑信息、通信模式、展开后的 item 元数据以及资源上下文指针。
struct OpParam {
    char tag[kAllGatherBatchTagLength] = {0};
    char commName[COMM_NAME_MAX_LENGTH] = {0};
    BatchTopoInfo topoInfo {};
    BatchKernelOpType opType = BatchKernelOpType::kAllGatherBatch;
    BatchCommMode commMode = BatchCommMode::kUnknown;
    uint32_t itemCount = 0;
    uint32_t intraServerRankCount = 0;
    uint32_t crossServerRankCount = 0;
    uint32_t reserved0 = 0;
    uint32_t controlNotifyIds[kAllGatherBatchControlNotifyNum] = {0};
    uint64_t totalInputBytes = 0;
    uint64_t totalOutputBytes = 0;
    uint64_t windowBytes = 0;
    BatchItemParam items[kAllGatherBatchMaxItems] {};
    AlgResourceCtx *resCtx = nullptr;
};

struct WindowStageSlice {
    uint32_t rank = 0;
    uint32_t itemIdx = 0;
    uint64_t itemOffsetBytes = 0;
    uint64_t rankOffsetBytes = 0;
    uint64_t stageOffsetBytes = 0;
    uint64_t size = 0;
};

struct WindowStageLayout {
    uint32_t rankSize = 0;
    uint32_t powerSteps = 0;
    uint32_t powerFactor = 1;
    uint32_t noPower = 1;
    uint64_t packedBytes = 0;
    uint64_t totalBytes = 0;
    std::vector<uint64_t> rankBaseOffsets;
    std::vector<WindowStageSlice> localSlices;
    std::vector<WindowStageSlice> perRankSlices;
};

inline HcclMem HcclMemRange(HcclMem inMem, u64 offset, u64 size)
{
    HcclMem outMem;
    if (inMem.addr == nullptr) {
        HCCL_ERROR("HcclMem addr is null");
        return outMem;
    }
    if (offset + size > inMem.size){
        HCCL_ERROR("HcclMem request range[%llu] is out of size[%llu]", offset + size, inMem.size);
        return outMem;
    }
    outMem.type = inMem.type;
    outMem.addr = static_cast<void *>(static_cast<u8 *>(inMem.addr) + offset);
    outMem.size = size;
    return outMem;
}

inline uint32_t CalcStagePowerSteps(uint32_t rankSize)
{
    uint32_t steps = 0;
    while ((rankSize & 1U) == 0U && rankSize > 1U) {
        rankSize >>= 1U;
        ++steps;
    }
    return steps;
}

inline uint32_t ReverseLowerBits(uint32_t value, uint32_t bitCount)
{
    uint32_t reversed = 0;
    for (uint32_t bit = 0; bit < bitCount; ++bit) {
        reversed = (reversed << 1U) | ((value >> bit) & 1U);
    }
    return reversed;
}

inline void ReorderNoPowerSequence(
    uint32_t start,
    uint32_t end,
    uint32_t len,
    std::vector<uint32_t> &tree,
    std::vector<uint32_t> &tmp)
{
    for (uint32_t idx = start; idx < end; ++idx) {
        const uint32_t offset = idx - start;
        if ((offset & 1U) == 0U) {
            tmp[start + offset / 2U] = tree[idx];
        } else {
            tmp[start + (offset + len) / 2U] = tree[idx];
        }
    }
}

inline uint32_t GetNoPowerMappedIndex(uint32_t noPower, uint32_t group)
{
    if (noPower <= 1U) {
        return group;
    }
    std::vector<uint32_t> tree;
    tree.reserve(noPower);
    for (uint32_t idx = 0; idx < noPower; ++idx) {
        tree.push_back(idx);
    }

    std::vector<uint32_t> tmp(noPower, 0U);
    uint32_t nSteps = 0;
    for (uint32_t value = noPower - 1U; value != 0U; value >>= 1U) {
        ++nSteps;
    }
    uint32_t len = noPower;
    for (uint32_t step = 0; step < nSteps; ++step) {
        const uint32_t nSlices = (noPower - 1U + (1U << step)) / (1U << (step + 1U));
        if (nSlices <= 1U) {
            break;
        }

        bool endFlag = false;
        for (uint32_t part = 0; part * len < noPower; ++part) {
            const uint32_t start = part * len;
            const uint32_t end = std::min(start + len, noPower);
            ReorderNoPowerSequence(start, end, len, tree, tmp);
            if (((end - start) & 1U) == 1U) {
                endFlag = true;
            }
        }
        tree = tmp;
        if (endFlag) {
            break;
        }
        len >>= 1U;
    }

    std::vector<uint32_t> mapped(noPower, 0U);
    for (uint32_t idx = 0; idx < noPower; ++idx) {
        mapped[tree[idx]] = idx;
    }
    return mapped[group];
}

inline uint32_t GetStageRankIndex(const WindowStageLayout &layout, uint32_t rank);
inline uint64_t GetStageRankBaseOffset(const WindowStageLayout &layout, uint32_t rank);

inline WindowStageLayout BuildWindowStageLayout(uint32_t rankSize)
{
    WindowStageLayout layout;
    layout.rankSize = rankSize;
    layout.powerSteps = CalcStagePowerSteps(rankSize);
    layout.powerFactor = (layout.powerSteps == 0U) ? 1U : (1U << layout.powerSteps);
    layout.noPower = (layout.powerFactor == 0U) ? 0U : (rankSize / layout.powerFactor);
    return layout;
}

inline WindowStageLayout BuildSingleItemStageLayout(uint32_t rankSize, uint32_t localRank, uint64_t packedBytes)
{
    WindowStageLayout layout = BuildWindowStageLayout(rankSize);
    layout.packedBytes = packedBytes;
    layout.totalBytes = packedBytes * rankSize;
    layout.rankBaseOffsets.reserve(rankSize);
    for (uint32_t rank = 0; rank < rankSize; ++rank) {
        layout.rankBaseOffsets.push_back(packedBytes * GetStageRankIndex(layout, rank));
    }
    if (packedBytes == 0U || localRank >= rankSize) {
        return layout;
    }

    const uint64_t localRankBase = GetStageRankBaseOffset(layout, localRank);
    WindowStageSlice localSlice;
    localSlice.rank = localRank;
    localSlice.itemIdx = 0;
    localSlice.itemOffsetBytes = 0;
    localSlice.rankOffsetBytes = 0;
    localSlice.stageOffsetBytes = localRankBase;
    localSlice.size = packedBytes;
    layout.localSlices.push_back(localSlice);

    layout.perRankSlices.reserve(rankSize);
    for (uint32_t rank = 0; rank < rankSize; ++rank) {
        WindowStageSlice rankSlice = localSlice;
        rankSlice.rank = rank;
        rankSlice.stageOffsetBytes = GetStageRankBaseOffset(layout, rank);
        layout.perRankSlices.push_back(rankSlice);
    }
    return layout;
}

inline bool IsValidWindowStageLayout(const WindowStageLayout &layout)
{
    if (layout.rankSize == 0U ||
        layout.powerFactor == 0U ||
        layout.noPower == 0U ||
        (layout.powerFactor * layout.noPower != layout.rankSize)) {
        return false;
    }

    if (!layout.rankBaseOffsets.empty() && layout.rankBaseOffsets.size() != layout.rankSize) {
        return false;
    }
    if (layout.packedBytes == 0) {
        return layout.totalBytes == 0 &&
            layout.rankBaseOffsets.empty() &&
            layout.localSlices.empty() &&
            layout.perRankSlices.empty();
    }
    if (layout.totalBytes != (layout.packedBytes * layout.rankSize)) {
        return false;
    }
    if (layout.rankBaseOffsets.size() != layout.rankSize) {
        return false;
    }
    if (layout.localSlices.empty()) {
        return false;
    }
    return !layout.perRankSlices.empty();
}

inline uint32_t GetStageRankIndex(const WindowStageLayout &layout, uint32_t rank)
{
    const uint32_t group = rank / layout.powerFactor;
    const uint32_t groupIdx = rank % layout.powerFactor;
    const uint32_t stageGroupIdx = ReverseLowerBits(groupIdx, layout.powerSteps);
    const uint32_t mappedGroup = GetNoPowerMappedIndex(layout.noPower, group);
    return (stageGroupIdx * layout.noPower) + mappedGroup;
}

inline uint64_t GetStageRankBaseOffset(const WindowStageLayout &layout, uint32_t rank)
{
    if (!layout.rankBaseOffsets.empty()) {
        return layout.rankBaseOffsets[rank];
    }
    return layout.packedBytes * GetStageRankIndex(layout, rank);
}


inline const char *ToCommModeString(BatchCommMode commMode)
{
    switch (commMode) {
        case BatchCommMode::kSingleServer:
            return "single-server";
        case BatchCommMode::kCrossServer:
            return "cross-server";
        default:
            return "unknown";
    }
}

inline const char *ToProtocolString(CommProtocol protocol)
{
    switch (protocol) {
        case COMM_PROTOCOL_HCCS:
            return "HCCS";
        case COMM_PROTOCOL_PCIE:
            return "PCIE";
        case COMM_PROTOCOL_ROCE:
            return "ROCE";
        case COMM_PROTOCOL_SIO:
            return "SIO";
        default:
            return "RESERVED";
    }
}

inline uint64_t GetCurrentTimeUs()
{
    const auto now = std::chrono::steady_clock::now().time_since_epoch();
    return static_cast<uint64_t>(std::chrono::duration_cast<std::chrono::microseconds>(now).count());
}

inline uint32_t GetExpectedFullMeshChannelCount(const OpParam &param)
{
    return (param.topoInfo.rankSize > 0U) ? (param.topoInfo.rankSize - 1U) : 0U;
}

inline ChannelResource *GetChannelArray(AlgResourceCtx &resCtx)
{
    return reinterpret_cast<ChannelResource *>(reinterpret_cast<uint8_t *>(&resCtx) + resCtx.channelOffset);
}

inline const ChannelResource *GetChannelArray(const AlgResourceCtx &resCtx)
{
    return reinterpret_cast<const ChannelResource *>(reinterpret_cast<const uint8_t *>(&resCtx) + resCtx.channelOffset);
}

inline ChannelResource &GetChannel(AlgResourceCtx &resCtx, uint32_t idx)
{
    return GetChannelArray(resCtx)[idx];
}

inline const ChannelResource &GetChannel(const AlgResourceCtx &resCtx, uint32_t idx)
{
    return GetChannelArray(resCtx)[idx];
}

inline uint64_t GetPerRankWindowCapacity(const OpParam &param, const AlgResourceCtx &resCtx)
{
    if (param.topoInfo.rankSize == 0U) {
        return 0U;
    }
    return resCtx.localBuffer.size / param.topoInfo.rankSize;
}

inline uint64_t GetMaxWindowBytes(const OpParam &param, const AlgResourceCtx &resCtx)
{
    const uint64_t perRankCapacity = GetPerRankWindowCapacity(param, resCtx);
    if (perRankCapacity == 0U || param.windowBytes == 0U) {
        return 0U;
    }
    return (param.windowBytes < perRankCapacity) ? param.windowBytes : perRankCapacity;
}

inline HcclResult ValidateBasicOpParam(const OpParam &param, const char *tag)
{
    if (param.topoInfo.rankSize == 0U) {
        HCCL_ERROR("%s rankSize is zero", tag);
        return HCCL_E_PARA;
    }
    if (param.topoInfo.rank >= param.topoInfo.rankSize) {
        HCCL_ERROR("%s rank is out of range, rank=%u, rankSize=%u", tag, param.topoInfo.rank, param.topoInfo.rankSize);
        return HCCL_E_PARA;
    }
    if (param.itemCount == 0U || param.itemCount > kAllGatherBatchMaxItems) {
        HCCL_ERROR("%s itemCount is invalid, itemCount=%u", tag, param.itemCount);
        return HCCL_E_PARA;
    }
    if (param.windowBytes == 0U) {
        HCCL_ERROR("%s windowBytes is zero", tag);
        return HCCL_E_PARA;
    }
    if (param.resCtx == nullptr) {
        HCCL_ERROR("%s resCtx is null", tag);
        return HCCL_E_PTR;
    }
    if (param.resCtx->localBuffer.addr == nullptr || param.resCtx->localBuffer.size == 0U) {
        HCCL_ERROR("%s localBuffer is invalid, addr=%p, size=%llu",
            tag,
            param.resCtx->localBuffer.addr,
            static_cast<unsigned long long>(param.resCtx->localBuffer.size));
        return HCCL_E_PARA;
    }
    return HCCL_SUCCESS;
}

constexpr uint32_t SIZE_TABLE[HCCL_DATA_TYPE_RESERVED] = {sizeof(int8_t), sizeof(int16_t), sizeof(int32_t),
    2, sizeof(float), sizeof(int64_t), sizeof(uint64_t), sizeof(uint8_t), sizeof(uint16_t), sizeof(uint32_t),
    8, 2, 16, 2, 1, 1, 1, 1};

}  // namespace ops_hccl_allgatherbatch

#endif
