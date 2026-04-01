#ifndef HCCL_ALLGATHERBATCH_COMMON_H
#define HCCL_ALLGATHERBATCH_COMMON_H

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
constexpr uint32_t kAllGatherBatchMaxChannels = 32;
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

struct AlgResourceCtx {
    ThreadHandle threadHandle = 0;
    uint32_t controlNotifyIds[kAllGatherBatchControlNotifyNum] = {0};
    uint32_t channelCount = 0;
    uint32_t reserved = 0;
    CommBuffer localBuffer {};
    ChannelResource channels[kAllGatherBatchMaxChannels] {};
};

struct BatchItemParam {
    void *sendBuf = nullptr;
    void *recvBuf = nullptr;
    uint64_t sendCount = 0;
    HcclDataType dataType = HCCL_DATA_TYPE_RESERVED;
    uint64_t elementSize = 0;
    uint64_t sendBytes = 0;
};

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
    uint64_t appendedItemBytes = 0;
    uint64_t totalInputBytes = 0;
    uint64_t totalOutputBytes = 0;
    uint64_t windowBytes = 0;
    BatchItemParam items[kAllGatherBatchMaxItems] {};
    AlgResourceCtx *resCtx = nullptr;
};

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
        if (resCtx.channels[idx].protocol == protocol) {
            ++count;
        }
    }
    return count;
}

inline uint32_t CountCrossServerChannels(const BatchTopoInfo &topoInfo, const AlgResourceCtx &resCtx)
{
    uint32_t count = 0;
    for (uint32_t idx = 0; idx < resCtx.channelCount; ++idx) {
        if (resCtx.channels[idx].remoteServerIdx != topoInfo.serverIdx) {
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

}  // namespace ops_hccl_allgatherbatch

#endif
