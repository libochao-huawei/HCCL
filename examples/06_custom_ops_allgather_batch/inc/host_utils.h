#ifndef OPS_HCCL_CUSTOM_ALLGATHER_BATCH_HOST_UTILS_H
#define OPS_HCCL_CUSTOM_ALLGATHER_BATCH_HOST_UTILS_H

#include <cstdint>
#include <string>

#include "common.h"

namespace ops_hccl_allgather_batch {

struct PackedBatchItem {
    uint64_t inputAddr = 0;
    uint64_t outputAddr = 0;
    uint64_t token = 0;
    uint64_t offset = 0;
    uint64_t sliceSize = 0;
};

std::string BuildContextKey(const char *tag, uintptr_t commKey, uint32_t itemCount);
std::string BuildEngineCtxTag(const char *tag, uint32_t itemCount);
HcclResult CheckItemValid(const HcclAllGatherItem &item, uint32_t index, uint32_t rank, uint32_t rankSize);
uint64_t GetSliceSizeBytes(const HcclAllGatherItem &item);
void PackBatchItemsForLaunch(const OpParam &param, PackedBatchItem *packedItems, uint32_t capacity);

}

#endif
