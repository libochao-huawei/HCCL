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
constexpr uint32_t SubThreadNum = 3;
constexpr u32 NOTIFY_IDX_ACK = 0;
constexpr u32 NOTIFY_IDX_DATA_SIGNAL = 1;
constexpr u32 NOTIFY_IDX_FIN_ACK = 2;
constexpr uint32_t CUSTOM_TIMEOUT = 120;
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
    HcclDataType dataType{HCCL_DATA_TYPE_RESERVED};
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

// addr 表示 `CCLIn` 的起始地址
// addr + offset 表示 `CCLOut` 的起始地址
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
    ThreadHandle mainThreadHandle = 0;
    ThreadHandle cpuThreadOnAicpu = 0;
    uint32_t subThreadCount = 0;
    uint32_t reserved0 = 0;
    ThreadHandle subThreadHandles[SubThreadNum] = {0};
    uint32_t mainNotifyIds[SubThreadNum] = {0};
    uint32_t subNotifyIds[SubThreadNum] = {0};
    uint32_t channelCount = 0;
    uint32_t channelOffset = 0;
    CommBuffer localBuffer {};
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
    ThreadHandle cpuThread = 0;
    ThreadHandle aicpuThreadOnCpu = 0;
    uint32_t startThreadNotifyIdx = 0;
    uint64_t totalInputBytes = 0;
    uint64_t totalOutputBytes = 0;
    uint64_t windowBytes = 0;
    BatchItemParam items[kAllGatherBatchMaxItems] {};
    AlgResourceCtx *resCtx = nullptr;
};

inline HcclMem HcclMemRange(HcclMem inMem, u64 offset, u64 size)
{
    HcclMem outMem;
    if (inMem.addr == nullptr) {
        HCCL_ERROR("HcclMem addr is null");
        return outMem;
    }
    if (offset + size > inMem.size) {
        HCCL_ERROR("HcclMem request range[%llu] is out of size[%llu]", offset + size, inMem.size);
        return outMem;
    }
    outMem.type = inMem.type;
    outMem.addr = static_cast<void *>(static_cast<u8 *>(inMem.addr) + offset);
    outMem.size = size;
    return outMem;
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
