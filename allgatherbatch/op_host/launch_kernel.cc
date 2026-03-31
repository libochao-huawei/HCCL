#include "launch_kernel.h"

#include "log.h"

namespace ops_hccl_allgatherbatch {

HcclResult LaunchKernel(const OpParam &param, aclrtStream stream)
{
    (void)param;
    (void)stream;
    HCCL_BATCH_INFO("phase 1 stub: LaunchKernel not implemented yet");
    return HCCL_E_NOT_SUPPORT;
}

}  // namespace ops_hccl_allgatherbatch
