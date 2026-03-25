#ifndef OPS_HCCL_DOUBLE_ALLGATHER_LOAD_KERNEL_H
#define OPS_HCCL_DOUBLE_ALLGATHER_LOAD_KERNEL_H

#include <string>
#include "common.h"

namespace ops_hccl_double_allgather {

HcclResult LoadAICPUKernel(void);
extern thread_local aclrtBinHandle g_binKernelHandle;

}

#endif
