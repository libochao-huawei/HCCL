#include <aicpu_api.h>
#include "common.h"
#include "exec_double_all_gather.h"

using namespace ops_hccl_double_allgather;

#ifdef __cplusplus
extern "C" {
#endif

__global__ __aicpu__ uint32_t HcclLaunchDoubleAllGatherAicpuKernel(void *args)
{
    DoubleAllGatherParam *param = reinterpret_cast<DoubleAllGatherParam *>(args);
    if (HcommAcquireComm(param->commName) != HCCL_SUCCESS) {
        return 1;
    }
    ThreadHandle thread = param->resCtx->threadHandle;
    if (HcommBatchModeStart(param->tag) != HCCL_SUCCESS) {
        return 1;
    }
    if (HcommAclrtNotifyWaitOnThread(thread, param->resCtx->notifyIds[0], CUSTOM_TIMEOUT) != HCCL_SUCCESS) {
        return 1;
    }
    if (ExecDoubleAllGather(*param, param->resCtx) != HCCL_SUCCESS) {
        return 1;
    }
    if (HcommAclrtNotifyRecordOnThread(thread, param->resCtx->notifyIds[1]) != HCCL_SUCCESS) {
        return 1;
    }
    if (HcommBatchModeEnd(param->tag) != HCCL_SUCCESS) {
        return 1;
    }
    if (HcommReleaseComm(param->commName) != HCCL_SUCCESS) {
        return 1;
    }
    return 0;
}

#ifdef __cplusplus
}
#endif

