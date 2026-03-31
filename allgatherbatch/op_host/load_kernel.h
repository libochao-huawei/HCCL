#ifndef HCCL_ALLGATHERBATCH_LOAD_KERNEL_H
#define HCCL_ALLGATHERBATCH_LOAD_KERNEL_H

#include <string>

#include "common.h"

namespace ops_hccl_allgatherbatch {

HcclResult LoadAICPUKernel();
HcclResult GetKernelConfigPath(std::string &jsonPath);
extern thread_local aclrtBinHandle g_allGatherBatchKernelHandle;

}  // namespace ops_hccl_allgatherbatch

#endif
