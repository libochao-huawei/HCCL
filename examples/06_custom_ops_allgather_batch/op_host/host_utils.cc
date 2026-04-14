#include "host_utils.h"

#include <cstdint>

namespace ops_hccl_allgather_batch {

std::string BuildContextKey(const char *commName, uint32_t itemCount)
{
    return std::string(commName) + "_batch_" + std::to_string(itemCount);
}

HcclResult CheckItemValid(const HcclAllGatherItem &item, uint32_t index)
{
    CHK_PTR_NULL(item.sendBuf);
    CHK_PTR_NULL(item.recvBuf);
    CHK_PRT_RET(item.sendCount == 0,
                HCCL_ERROR("[HcclAllGatherBatch] item[%u] sendCount is 0", index),
                HCCL_E_PARA);
    CHK_PRT_RET(!IsSupportedDataType(item.dataType),
                HCCL_ERROR("[HcclAllGatherBatch] item[%u] dataType=%d unsupported", index, item.dataType),
                HCCL_E_PARA);

    const uint32_t elemSize = GetDataTypeSize(item.dataType);
    CHK_PRT_RET(item.sendCount > UINT64_MAX / elemSize,
                HCCL_ERROR("[HcclAllGatherBatch] item[%u] sendCount overflow, count=%lu elemSize=%u",
                           index, item.sendCount, elemSize),
                HCCL_E_PARA);
    return HCCL_SUCCESS;
}

uint64_t GetSliceSizeBytes(const HcclAllGatherItem &item)
{
    return item.sendCount * static_cast<uint64_t>(GetDataTypeSize(item.dataType));
}

} // namespace ops_hccl_allgather_batch
