#include "host_utils.h"

#include <cstdint>

#include "ccu_kernel.h"

namespace ops_hccl_allgather_batch {

namespace {

uint64_t GetLaunchToken(uint64_t inputAddr, uint64_t sliceSize)
{
#ifdef HCCL_BATCH_HOST_UT
    uint64_t token = inputAddr ^ (sliceSize << 1);
    return token == 0 ? 1 : token;
#else
    return hcomm::CcuRep::GetTokenInfo(inputAddr, sliceSize);
#endif
}

} // namespace

std::string BuildContextKey(const char *tag, uintptr_t commKey, uint32_t itemCount)
{
    return std::string(tag) + "_comm_" + std::to_string(commKey) + "_n" + std::to_string(itemCount);
}

std::string BuildEngineCtxTag(const char *tag, uint32_t itemCount)
{
    return std::string(tag) + "_n" + std::to_string(itemCount);
}

HcclResult CheckItemValid(const HcclAllGatherItem &item, uint32_t index, uint32_t rank, uint32_t rankSize)
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

    const uint64_t sliceSize = item.sendCount * static_cast<uint64_t>(elemSize);
    CHK_PRT_RET(rankSize != 0 && sliceSize > UINT64_MAX / rankSize,
                HCCL_ERROR("[HcclAllGatherBatch] item[%u] recv span overflow, sliceSize=%lu rankSize=%u",
                           index, sliceSize, rankSize),
                HCCL_E_PARA);
    CHK_PRT_RET(rank != 0 && sliceSize > UINT64_MAX / rank,
                HCCL_ERROR("[HcclAllGatherBatch] item[%u] offset overflow, sliceSize=%lu rank=%u",
                           index, sliceSize, rank),
                HCCL_E_PARA);
    return HCCL_SUCCESS;
}

uint64_t GetSliceSizeBytes(const HcclAllGatherItem &item)
{
    return item.sendCount * static_cast<uint64_t>(GetDataTypeSize(item.dataType));
}

void PackBatchItemsForLaunch(const OpParam &param, PackedBatchItem *packedItems, uint32_t capacity)
{
    if (packedItems == nullptr) {
        return;
    }

    const uint32_t packCount = (param.itemCount < capacity) ? param.itemCount : capacity;
    for (uint32_t i = 0; i < packCount; ++i) {
        const HcclAllGatherItem &item = param.items[i];
        const uint64_t sliceSize = GetSliceSizeBytes(item);
        packedItems[i].inputAddr = reinterpret_cast<uint64_t>(item.sendBuf);
        packedItems[i].outputAddr = reinterpret_cast<uint64_t>(item.recvBuf);
        packedItems[i].token = GetLaunchToken(packedItems[i].inputAddr, sliceSize);
        packedItems[i].offset = static_cast<uint64_t>(param.rank) * sliceSize;
        packedItems[i].sliceSize = sliceSize;
    }
}

} // namespace ops_hccl_allgather_batch
