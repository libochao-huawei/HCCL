#ifndef HCCL_ALLGATHERBATCH_LAUNCH_KERNEL_H
#define HCCL_ALLGATHERBATCH_LAUNCH_KERNEL_H

#include "common.h"

namespace ops_hccl_allgatherbatch {

HcclResult LaunchKernel(const OpParam &param, aclrtStream stream);
extern thread_local aclrtNotify g_allGatherBatchNotifies[kAllGatherBatchControlNotifyNum];

}  // namespace ops_hccl_allgatherbatch

#endif
