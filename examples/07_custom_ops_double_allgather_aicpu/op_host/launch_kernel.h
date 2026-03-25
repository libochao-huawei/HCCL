#ifndef OPS_HCCL_DOUBLE_ALLGATHER_LAUNCH_KERNEL_H
#define OPS_HCCL_DOUBLE_ALLGATHER_LAUNCH_KERNEL_H

#include "common.h"

enum KernelLaunchMode {
    KERNEL_LAUNCH_ACLRT,
    KERNEL_LAUNCH_ASC
};

namespace ops_hccl_double_allgather {

extern thread_local aclrtNotify g_notifies[AICPU_CONTROL_NOTIFY_NUM];
extern HcclResult LaunchKernelAsc(DoubleAllGatherParam &param, aclrtStream stream);
HcclResult LaunchKernel(DoubleAllGatherParam &param, aclrtStream stream);

}

#endif
