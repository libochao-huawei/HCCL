#ifndef HCCL_ALLGATHERBATCH_TASK_EXCEPTION_INFO_H
#define HCCL_ALLGATHERBATCH_TASK_EXCEPTION_INFO_H

#include "common.h"

namespace ops_hccl_allgatherbatch {

constexpr uint32_t kAllGatherBatchAlgTagLength = 64;
constexpr uint32_t kAllGatherBatchAlgNameLength = 64;

struct AllGatherBatchOpInfo {
    char algTag[kAllGatherBatchAlgTagLength] = {0};
    char algName[kAllGatherBatchAlgNameLength] = {0};
    char tag[kAllGatherBatchTagLength] = {0};
    char commName[COMM_NAME_MAX_LENGTH] = {0};
    uint32_t rank = 0;
    uint32_t rankSize = 0;
    uint32_t itemCount = 0;
    BatchKernelOpType opType = BatchKernelOpType::kAllGatherBatch;
    BatchCommMode commMode = BatchCommMode::kUnknown;
    uint64_t totalInputBytes = 0;
};

HcclResult CreateAllGatherBatchOpInfo(const OpParam &param, AllGatherBatchOpInfo &opInfo);

void GetAllGatherBatchOpInfo(const void *opInfo, char *output, size_t size);

}  // namespace ops_hccl_allgatherbatch

#endif
