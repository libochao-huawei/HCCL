#ifndef HCCL_ALLGATHERBATCH_COMMON_H
#define HCCL_ALLGATHERBATCH_COMMON_H

#include <chrono>
#include <cstddef>
#include <cstdint>
#include <cstring>

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
constexpr uint32_t kAllGatherBatchCustomTimeoutMs = 1800;
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

struct CommBuffer {
    void *addr = nullptr;
    uint64_t size = 0;
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
    ThreadHandle threadHandle = 0;
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
// 里面包含拓扑信息、通信模式、展平后的 item 元数据以及资源上下文指针。
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

inline uint64_t GetCurrentTimeUs()
{
    const auto now = std::chrono::steady_clock::now().time_since_epoch();
    return static_cast<uint64_t>(std::chrono::duration_cast<std::chrono::microseconds>(now).count());
}

inline uint64_t GetDataTypeSize(HcclDataType dataType)
{
    switch (dataType) {
        case HCCL_DATA_TYPE_INT8:
        case HCCL_DATA_TYPE_UINT8:
        case HCCL_DATA_TYPE_HIF8:
        case HCCL_DATA_TYPE_FP8E4M3:
        case HCCL_DATA_TYPE_FP8E5M2:
        case HCCL_DATA_TYPE_FP8E8M0:
            return 1;
        case HCCL_DATA_TYPE_INT16:
        case HCCL_DATA_TYPE_FP16:
        case HCCL_DATA_TYPE_UINT16:
        case HCCL_DATA_TYPE_BFP16:
            return 2;
        case HCCL_DATA_TYPE_INT32:
        case HCCL_DATA_TYPE_FP32:
        case HCCL_DATA_TYPE_UINT32:
            return 4;
        case HCCL_DATA_TYPE_INT64:
        case HCCL_DATA_TYPE_UINT64:
        case HCCL_DATA_TYPE_FP64:
            return 8;
        case HCCL_DATA_TYPE_INT128:
            return 16;
        default:
            return 0;
    }
}

inline HcommDataType ToHcommDataType(HcclDataType dataType)
{
    return static_cast<HcommDataType>(dataType);
}

inline bool IsSupportedDataType(HcclDataType dataType)
{
    return GetDataTypeSize(dataType) != 0;
}

inline bool IsAligned32(const void *ptr)
{
    return ptr != nullptr && ((reinterpret_cast<uintptr_t>(ptr) & 0x1fU) == 0);
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
        case COMM_PROTOCOL_ROCE:
            return "ROCE";
        case COMM_PROTOCOL_PCIE:
            return "PCIE";
        case COMM_PROTOCOL_SIO:
            return "SIO";
        default:
            return "RESERVED";
    }
}

inline bool IsValidCommMode(BatchCommMode commMode)
{
    return commMode == BatchCommMode::kSingleServer || commMode == BatchCommMode::kCrossServer;
}

// 当前版本明确假设 fullmesh：除了本 rank 之外，其余每个 rank 都有一条 channel。
inline uint32_t GetExpectedFullMeshChannelCount(uint32_t rankSize)
{
    return (rankSize > 0) ? (rankSize - 1) : 0;
}

inline uint32_t GetExpectedFullMeshChannelCount(const OpParam &param)
{
    return GetExpectedFullMeshChannelCount(param.topoInfo.rankSize);
}

// 动态 channel 区统一通过 helper 访问，避免 Host 和 Device 继续依赖固定数组布局。
inline const ChannelResource *GetChannelArray(const AlgResourceCtx &resCtx)
{
    if (resCtx.channelCount == 0) {
        return nullptr;
    }
    return reinterpret_cast<const ChannelResource *>(
        reinterpret_cast<const uint8_t *>(&resCtx) + resCtx.channelOffset);
}

inline ChannelResource *GetChannelArray(AlgResourceCtx &resCtx)
{
    if (resCtx.channelCount == 0) {
        return nullptr;
    }
    return reinterpret_cast<ChannelResource *>(
        reinterpret_cast<uint8_t *>(&resCtx) + resCtx.channelOffset);
}

inline const ChannelResource &GetChannel(const AlgResourceCtx &resCtx, uint32_t idx)
{
    return GetChannelArray(resCtx)[idx];
}

inline ChannelResource &GetChannel(AlgResourceCtx &resCtx, uint32_t idx)
{
    return GetChannelArray(resCtx)[idx];
}

inline uint64_t GetPerRankWindowCapacity(const OpParam &param, const AlgResourceCtx &resCtx)
{
    if (param.topoInfo.rankSize == 0) {
        return 0;
    }
    return resCtx.localBuffer.size / param.topoInfo.rankSize;
}

inline uint64_t GetMaxWindowBytes(const OpParam &param, const AlgResourceCtx &resCtx)
{
    const uint64_t perRankCapacity = GetPerRankWindowCapacity(param, resCtx);
    if (param.windowBytes == 0 || perRankCapacity == 0) {
        return 0;
    }
    return (param.windowBytes < perRankCapacity) ? param.windowBytes : perRankCapacity;
}

inline uint32_t CountChannelsByProtocol(const AlgResourceCtx &resCtx, CommProtocol protocol)
{
    uint32_t count = 0;
    for (uint32_t idx = 0; idx < resCtx.channelCount; ++idx) {
        if (GetChannel(resCtx, idx).protocol == protocol) {
            ++count;
        }
    }
    return count;
}

inline uint32_t CountCrossServerChannels(const BatchTopoInfo &topoInfo, const AlgResourceCtx &resCtx)
{
    uint32_t count = 0;
    for (uint32_t idx = 0; idx < resCtx.channelCount; ++idx) {
        if (GetChannel(resCtx, idx).remoteServerIdx != topoInfo.serverIdx) {
            ++count;
        }
    }
    return count;
}

inline uint32_t CountRecognizedProtocols(const AlgResourceCtx &resCtx)
{
    return CountChannelsByProtocol(resCtx, COMM_PROTOCOL_HCCS) +
        CountChannelsByProtocol(resCtx, COMM_PROTOCOL_ROCE) +
        CountChannelsByProtocol(resCtx, COMM_PROTOCOL_PCIE) +
        CountChannelsByProtocol(resCtx, COMM_PROTOCOL_SIO);
}

struct ResourceStats {
    uint32_t crossServerChannels = 0;
    uint32_t intraServerChannels = 0;
    uint32_t hccsChannels = 0;
    uint32_t roceChannels = 0;
    uint32_t pcieChannels = 0;
    uint32_t sioChannels = 0;
    uint32_t recognizedProtocols = 0;
    uint64_t perRankCapacity = 0;
    uint64_t maxWindowBytes = 0;
};

inline ResourceStats CollectResourceStats(const OpParam &param, const AlgResourceCtx &resCtx)
{
    ResourceStats stats;
    stats.crossServerChannels = CountCrossServerChannels(param.topoInfo, resCtx);
    stats.intraServerChannels = resCtx.channelCount - stats.crossServerChannels;
    stats.hccsChannels = CountChannelsByProtocol(resCtx, COMM_PROTOCOL_HCCS);
    stats.roceChannels = CountChannelsByProtocol(resCtx, COMM_PROTOCOL_ROCE);
    stats.pcieChannels = CountChannelsByProtocol(resCtx, COMM_PROTOCOL_PCIE);
    stats.sioChannels = CountChannelsByProtocol(resCtx, COMM_PROTOCOL_SIO);
    stats.recognizedProtocols = stats.hccsChannels + stats.roceChannels + stats.pcieChannels + stats.sioChannels;
    stats.perRankCapacity = GetPerRankWindowCapacity(param, resCtx);
    stats.maxWindowBytes = GetMaxWindowBytes(param, resCtx);
    return stats;
}

inline bool HasConsistentRankDistribution(const OpParam &param)
{
    return param.topoInfo.rankSize != 0 &&
        param.intraServerRankCount != 0 &&
        (param.intraServerRankCount + param.crossServerRankCount == param.topoInfo.rankSize) &&
        ((param.commMode == BatchCommMode::kSingleServer && param.crossServerRankCount == 0) ||
         (param.commMode == BatchCommMode::kCrossServer && param.crossServerRankCount != 0));
}

inline HcclResult ValidateBasicOpParam(const OpParam &param, const char *stageTag)
{
    if (!IsValidCommMode(param.commMode)) {
        HCCL_ERROR("%s commMode is invalid", stageTag);
        return HCCL_E_INTERNAL;
    }
    if (param.itemCount == 0 || param.itemCount > kAllGatherBatchMaxItems) {
        HCCL_ERROR("%s itemCount=%u is invalid", stageTag, param.itemCount);
        return HCCL_E_INTERNAL;
    }
    if (param.topoInfo.rankSize == 0 || param.topoInfo.rank >= param.topoInfo.rankSize) {
        HCCL_ERROR("%s topo rank/rankSize is invalid, rank=%u, rankSize=%u",
            stageTag,
            param.topoInfo.rank,
            param.topoInfo.rankSize);
        return HCCL_E_INTERNAL;
    }
    if (!HasConsistentRankDistribution(param)) {
        HCCL_ERROR("%s rank distribution is inconsistent, commMode=%s, intra=%u, cross=%u, rankSize=%u",
            stageTag,
            ToCommModeString(param.commMode),
            param.intraServerRankCount,
            param.crossServerRankCount,
            param.topoInfo.rankSize);
        return HCCL_E_INTERNAL;
    }
    if (param.controlNotifyIds[kAllGatherBatchControlNotifyStart] == 0 ||
        param.controlNotifyIds[kAllGatherBatchControlNotifyDone] == 0) {
        HCCL_ERROR("%s control notify ids are invalid, start=%u, done=%u",
            stageTag,
            param.controlNotifyIds[kAllGatherBatchControlNotifyStart],
            param.controlNotifyIds[kAllGatherBatchControlNotifyDone]);
        return HCCL_E_INTERNAL;
    }
    if (param.totalInputBytes == 0 || param.totalOutputBytes == 0 || param.windowBytes == 0) {
        HCCL_ERROR("%s byte sizes are invalid, input=%llu, output=%llu, window=%llu",
            stageTag,
            static_cast<unsigned long long>(param.totalInputBytes),
            static_cast<unsigned long long>(param.totalOutputBytes),
            static_cast<unsigned long long>(param.windowBytes));
        return HCCL_E_INTERNAL;
    }
    return HCCL_SUCCESS;
}

// 基础资源校验在 Host、kernel 入口、executor 等多层复用，确保大家对 fullmesh 资源形态的理解一致。
inline HcclResult ValidateBasicResourceCtx(const OpParam &param, const AlgResourceCtx &resCtx, const char *stageTag)
{
    const ResourceStats stats = CollectResourceStats(param, resCtx);
    const uint32_t expectedChannelCount = GetExpectedFullMeshChannelCount(param);

    if (resCtx.threadHandle == 0) {
        HCCL_ERROR("%s threadHandle is invalid", stageTag);
        return HCCL_E_INTERNAL;
    }
    if (resCtx.localBuffer.addr == nullptr || resCtx.localBuffer.size == 0) {
        HCCL_ERROR("%s localBuffer is invalid", stageTag);
        return HCCL_E_INTERNAL;
    }
    if (resCtx.channelCount > 0 && resCtx.channelOffset < sizeof(AlgResourceCtx)) {
        HCCL_ERROR("%s channelOffset=%u is invalid", stageTag, resCtx.channelOffset);
        return HCCL_E_INTERNAL;
    }
    if (resCtx.channelCount != expectedChannelCount) {
        HCCL_ERROR("%s fullmesh channelCount mismatch, actual=%u, expected=%u",
            stageTag,
            resCtx.channelCount,
            expectedChannelCount);
        return HCCL_E_INTERNAL;
    }
    if (stats.intraServerChannels + stats.crossServerChannels != resCtx.channelCount) {
        HCCL_ERROR("%s channel split is inconsistent, intra=%u, cross=%u, channelCount=%u",
            stageTag,
            stats.intraServerChannels,
            stats.crossServerChannels,
            resCtx.channelCount);
        return HCCL_E_INTERNAL;
    }
    if (stats.maxWindowBytes == 0) {
        HCCL_ERROR("%s maxWindowBytes is zero", stageTag);
        return HCCL_E_INTERNAL;
    }
    if (stats.recognizedProtocols != resCtx.channelCount) {
        HCCL_ERROR("%s protocol distribution mismatch, recognized=%u, channelCount=%u",
            stageTag,
            stats.recognizedProtocols,
            resCtx.channelCount);
        return HCCL_E_INTERNAL;
    }
    if (param.commMode == BatchCommMode::kSingleServer && stats.crossServerChannels != 0) {
        HCCL_ERROR("%s unexpectedly has crossServerChannels=%u in single-server mode",
            stageTag,
            stats.crossServerChannels);
        return HCCL_E_INTERNAL;
    }
    if (param.commMode == BatchCommMode::kCrossServer && stats.crossServerChannels != param.crossServerRankCount) {
        HCCL_ERROR("%s cross-server channel mismatch, channels=%u, expected=%u",
            stageTag,
            stats.crossServerChannels,
            param.crossServerRankCount);
        return HCCL_E_INTERNAL;
    }
    return HCCL_SUCCESS;
}

inline HcclResult ValidateRemoteChannelResources(const OpParam &param, const AlgResourceCtx &resCtx, const char *stageTag)
{
    const ResourceStats stats = CollectResourceStats(param, resCtx);

    // 这一层专门约束每条远端 channel 的 endpoint / buffer 元数据，Host 和 Device 共用同一套规则。
    for (uint32_t idx = 0; idx < resCtx.channelCount; ++idx) {
        const ChannelResource &channel = GetChannel(resCtx, idx);
        if (channel.protocol == COMM_PROTOCOL_RESERVED) {
            HCCL_ERROR("%s channel %u has reserved protocol", stageTag, idx);
            return HCCL_E_INTERNAL;
        }
        if (channel.remoteRank == param.topoInfo.rank || channel.remoteRank >= param.topoInfo.rankSize) {
            HCCL_ERROR("%s channel %u remoteRank=%u is invalid", stageTag, idx, channel.remoteRank);
            return HCCL_E_INTERNAL;
        }
        if (channel.remoteSuperPodIdx != param.topoInfo.superPodIdx) {
            HCCL_ERROR("%s channel %u crosses superPod unexpectedly, local=%u, remote=%u",
                stageTag,
                idx,
                param.topoInfo.superPodIdx,
                channel.remoteSuperPodIdx);
            return HCCL_E_INTERNAL;
        }
        if (channel.remoteBuffer.addr == nullptr || channel.remoteBuffer.size == 0) {
            HCCL_ERROR("%s channel %u remoteBuffer is invalid", stageTag, idx);
            return HCCL_E_INTERNAL;
        }
        if (channel.remoteBuffer.size < (stats.maxWindowBytes * param.topoInfo.rankSize)) {
            HCCL_ERROR("%s channel %u remoteBuffer too small, need=%llu, actual=%llu",
                stageTag,
                idx,
                static_cast<unsigned long long>(stats.maxWindowBytes * param.topoInfo.rankSize),
                static_cast<unsigned long long>(channel.remoteBuffer.size));
            return HCCL_E_INTERNAL;
        }
    }
    return HCCL_SUCCESS;
}

}  // namespace ops_hccl_allgatherbatch

#endif
