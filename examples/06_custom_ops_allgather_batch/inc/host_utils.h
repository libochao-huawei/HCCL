#ifndef OPS_HCCL_CUSTOM_ALLGATHER_BATCH_HOST_UTILS_H
#define OPS_HCCL_CUSTOM_ALLGATHER_BATCH_HOST_UTILS_H

#include <string>

#include "common.h"

namespace ops_hccl_allgather_batch {

std::string BuildContextKey(const char *commName, uint32_t itemCount);
HcclResult CheckItemValid(const HcclAllGatherItem &item, uint32_t index);
uint64_t GetSliceSizeBytes(const HcclAllGatherItem &item);

}

#endif
