#include <hccl/hcomm_primitives.h>
#include "common.h"
#include "exec_op.h"

using namespace ops_hccl_allgather_2in2out;

extern "C" unsigned int HcclLaunchAllGather2In2OutAicpuKernel(OpParam *param)
{
    if (param == nullptr || param->resCtx == nullptr) {
        HCCL_ERROR("[HcclLaunchAllGather2In2OutAicpuKernel] param or resCtx is nullptr");
        return 1;
    }

    HCCL_INFO("[HcclLaunchAllGather2In2OutAicpuKernel] Entry, commName[%s], tag[%s]",
        param->commName, param->tag);

    ThreadHandle mainThread = param->resCtx->threads[0];
    bool commAcquired = false;
    bool batchStarted = false;
    bool hostNotified = false;
    unsigned int retCode = 1;

    // 这段 cleanup 逻辑很重要：
    // 就算 device 侧执行失败，也尽量回一个 Host completion notify，
    // 避免 Host 一直卡在 wait notify 上，定位问题时也更直观。
    if (HcommAcquireComm(param->commName) != HCCL_SUCCESS) {
        HCCL_ERROR("[HcclLaunchAllGather2In2OutAicpuKernel] HcommAcquireComm failed");
        goto CLEANUP;
    }
    commAcquired = true;

    if (HcommBatchModeStart(param->tag) != HCCL_SUCCESS) {
        HCCL_ERROR("[HcclLaunchAllGather2In2OutAicpuKernel] HcommBatchModeStart failed");
        goto CLEANUP;
    }
    batchStarted = true;

    // AICPU 侧先等待 Host stream 的启动 notify，避免过早读取输入。
    if (HcommAclrtNotifyWaitOnThread(mainThread,
            param->resCtx->notifyIds[0],
            kCustomTimeout) != HCCL_SUCCESS) {
        HCCL_ERROR("[HcclLaunchAllGather2In2OutAicpuKernel] wait host notify failed");
        goto CLEANUP;
    }

    if (ExecOp(*param, param->resCtx) != HCCL_SUCCESS) {
        HCCL_ERROR("[HcclLaunchAllGather2In2OutAicpuKernel] ExecOp failed");
        goto CLEANUP;
    }

    retCode = 0;

CLEANUP:
    if (mainThread != nullptr) {
        if (HcommAclrtNotifyRecordOnThread(mainThread,
                param->resCtx->notifyIds[1]) != HCCL_SUCCESS) {
            HCCL_ERROR("[HcclLaunchAllGather2In2OutAicpuKernel] record host notify failed in cleanup");
        } else {
            hostNotified = true;
        }
    }

    if (batchStarted) {
        if (HcommBatchModeEnd(param->tag) != HCCL_SUCCESS) {
            HCCL_ERROR("[HcclLaunchAllGather2In2OutAicpuKernel] HcommBatchModeEnd failed in cleanup");
            retCode = 1;
        }
    }

    if (commAcquired) {
        if (HcommReleaseComm(param->commName) != HCCL_SUCCESS) {
            HCCL_ERROR("[HcclLaunchAllGather2In2OutAicpuKernel] HcommReleaseComm failed in cleanup");
            retCode = 1;
        }
    }

    if (!hostNotified) {
        HCCL_ERROR("[HcclLaunchAllGather2In2OutAicpuKernel] host completion notify was not recorded");
    }
    return retCode;
}
