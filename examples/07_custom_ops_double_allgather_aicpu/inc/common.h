#ifndef OPS_HCCL_DOUBLE_ALLGATHER_COMMON_H
#define OPS_HCCL_DOUBLE_ALLGATHER_COMMON_H

#include <cstddef>
#include <cstdint>
#include "hccl/hccl_types.h"
#include "hccl/hccl_res.h"
#include "hccl/hcomm_primitives.h"
#include "acl/acl_rt.h"
#include "log.h"

namespace ops_hccl_double_allgather {

constexpr uint32_t CUSTOM_TIMEOUT = 1800;
constexpr uint32_t COMM_INDENTIFIER_MAX_LENGTH = 128;
constexpr uint32_t OP_NAME_LENGTH = 64;
constexpr uint32_t TAG_LENGTH = OP_NAME_LENGTH + COMM_INDENTIFIER_MAX_LENGTH;
constexpr uint32_t AICPU_CONTROL_NOTIFY_NUM = 2;
constexpr uint32_t NOTIFY_IDX_READY = 0;
constexpr uint32_t NOTIFY_IDX_DONE = 1;

struct CommBuffer {
    void *addr = nullptr;
    uint64_t size = 0;
};

struct DirectionChannelCtx {
    ChannelHandle handle = 0;
    CommBuffer remoteBuffer;
    uint32_t remoteRank = 0;
};

struct AlgResourceCtx {
    ThreadHandle threadHandle = 0;
    CommBuffer localBuffer;
    DirectionChannelCtx prevChannel;
    DirectionChannelCtx nextChannel;
    uint32_t notifyIds[AICPU_CONTROL_NOTIFY_NUM] = {0};
    uint32_t prevRank = 0;
    uint32_t nextRank = 0;
};

struct GatherDesc {
    void *inputPtr = nullptr;
    void *outputPtr = nullptr;
    uint64_t count = 0;
    uint32_t dataType = 0;
    uint32_t elemBytes = 0;
};

struct DoubleAllGatherParam {
    char tag[TAG_LENGTH];
    char commName[COMM_INDENTIFIER_MAX_LENGTH];
    uint32_t rank = 0;
    uint32_t rankSize = 0;
    GatherDesc gather0;
    GatherDesc gather1;
    AlgResourceCtx *resCtx = nullptr;
};

constexpr uint32_t SIZE_TABLE[HCCL_DATA_TYPE_RESERVED] = {
    sizeof(int8_t), sizeof(int16_t), sizeof(int32_t), 2, sizeof(float), sizeof(int64_t), sizeof(uint64_t),
    sizeof(uint8_t), sizeof(uint16_t), sizeof(uint32_t), 8, 2, 16, 2, 1, 1, 1, 1
};

inline bool IsSupportedDataType(HcclDataType dataType)
{
    return dataType == HCCL_DATA_TYPE_FP16 || dataType == HCCL_DATA_TYPE_FP32 || dataType == HCCL_DATA_TYPE_INT32;
}

inline uint32_t GetElemBytes(HcclDataType dataType)
{
    return (dataType < HCCL_DATA_TYPE_RESERVED) ? SIZE_TABLE[dataType] : 0;
}

inline bool GetBytesByCount(uint64_t count, HcclDataType dataType, size_t &bytes)
{
    uint32_t elemBytes = GetElemBytes(dataType);
    if (elemBytes == 0 || count > static_cast<uint64_t>(SIZE_MAX / elemBytes)) {
        return false;
    }
    bytes = static_cast<size_t>(count) * elemBytes;
    return true;
}

} // namespace ops_hccl_double_allgather

#endif
