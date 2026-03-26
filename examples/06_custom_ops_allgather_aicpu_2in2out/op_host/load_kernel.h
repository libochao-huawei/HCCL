#ifndef OPS_HCCL_ALLGATHER_2IN2OUT_LOAD_KERNEL_H
#define OPS_HCCL_ALLGATHER_2IN2OUT_LOAD_KERNEL_H

#include <string>
#include "common.h"

namespace ops_hccl_allgather_2in2out {

HcclResult LoadAICPUKernel(void);
extern thread_local aclrtBinHandle g_binKernelHandle;

} // namespace ops_hccl_allgather_2in2out

#endif
