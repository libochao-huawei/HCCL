#include "common.h"
#include "diag_dl.h"
#include "exec_op.h"
#include "profiling.h"
#include "task_exception_info.h"

extern "C" unsigned int HcclAllGatherBatchAicpuKernel(
    ops_hccl_allgatherbatch::OpParam *param)
{
    using namespace ops_hccl_allgatherbatch;

    if (param == nullptr || param->resCtx == nullptr) {
        return 1;
    }
    if (ValidateBasicOpParam(*param, "AICPU kernel param") != HCCL_SUCCESS) {
        return 1;
    }

    if (HcommAcquireComm(param->commName) != HCCL_SUCCESS) {
        HCCL_ERROR("HcommAcquireComm failed, commName=%s", param->commName);
        return 1;
    }

    AllGatherBatchOpInfo opInfo {};
    if (CreateAllGatherBatchOpInfo(*param, opInfo) != HCCL_SUCCESS) {
        HCCL_ERROR("CreateAllGatherBatchOpInfo failed, commName=%s, tag=%s", param->commName, param->tag);
        (void)HcommReleaseComm(param->commName);
        return 1;
    }
    if (HcommIsSupportHcommRegOpInfo() &&
        HcommRegOpInfo(param->commName, reinterpret_cast<void *>(&opInfo), sizeof(AllGatherBatchOpInfo)) != HCCL_SUCCESS) {
        HCCL_ERROR("HcommRegOpInfo failed, commName=%s, tag=%s, algTag=%s",
            param->commName,
            param->tag,
            opInfo.algTag);
        (void)HcommReleaseComm(param->commName);
        return 1;
    }
    if (HcommIsSupportHcommRegOpTaskException() &&
        HcommRegOpTaskException(param->commName, ops_hccl_allgatherbatch::GetAllGatherBatchOpInfo) != HCCL_SUCCESS) {
        HCCL_ERROR("HcommRegOpTaskException failed, commName=%s, tag=%s, algTag=%s",
            param->commName,
            param->tag,
            opInfo.algTag);
        (void)HcommReleaseComm(param->commName);
        return 1;
    }

    ThreadHandle thread = param->resCtx->mainThreadHandle;
    ThreadHandle profilingThreads[1 + SubThreadNum] = {0};
    const uint32_t profilingThreadNum = BuildProfilingThreadList(
        *param->resCtx, profilingThreads, 1 + SubThreadNum);
    const uint32_t slaveThreadNum = (profilingThreadNum > 0) ? (profilingThreadNum - 1) : 0;
    bool profilingInitialized = false;
    auto EndProfilingIfNeeded = [&]() {
        if (!profilingInitialized) {
            return;
        }
        if (HcommProfilingEnd(profilingThreads, profilingThreadNum) != HCCL_SUCCESS) {
            HCCL_WARNING("HcommProfilingEnd failed, rank=%u, tag=%s", param->topoInfo.rank, param->tag);
        }
        profilingInitialized = false;
    };
    if (HcommBatchModeStart(param->tag) != HCCL_SUCCESS) {
        HCCL_ERROR("HcommBatchModeStart failed, tag=%s", param->tag);
        (void)HcommReleaseComm(param->commName);
        return 1;
    }

    if (profilingThreadNum > 0) {
        if (HcommProfilingInit(profilingThreads, profilingThreadNum) == HCCL_SUCCESS) {
            profilingInitialized = true;
            if (HcommProfilingReportMainStreamAndFirstTask(thread) != HCCL_SUCCESS) {
                HCCL_WARNING("HcommProfilingReportMainStreamAndFirstTask failed, rank=%u, tag=%s",
                    param->topoInfo.rank, param->tag);
            }
        } else {
            HCCL_WARNING("HcommProfilingInit failed, rank=%u, tag=%s", param->topoInfo.rank, param->tag);
        }
    }

    if (HcommThreadNotifyWaitOnThread(
            thread,
            param->startThreadNotifyIdx,
            CUSTOM_TIMEOUT) != HCCL_SUCCESS) {
        HCCL_ERROR("wait host start notify failed, tag=%s", param->tag);
        EndProfilingIfNeeded();
        (void)HcommBatchModeEnd(param->tag);
        (void)HcommReleaseComm(param->commName);
        return 1;
    }

    HcclResult ret = ExecOp(*param, param->resCtx);
    if (ret != HCCL_SUCCESS) {
        HCCL_ERROR("ExecOp failed, ret=%d", static_cast<int>(ret));
        EndProfilingIfNeeded();
        (void)HcommBatchModeEnd(param->tag);
        (void)HcommReleaseComm(param->commName);
        return 1;
    }

    if (profilingInitialized) {
        HcomProInfoTmp info {};
        FillProfilingInfo(info, *param, 0, slaveThreadNum);
        if (HcommProfilingReportDeviceHcclOpInfo(info) != HCCL_SUCCESS) {
            HCCL_WARNING("HcommProfilingReportDeviceHcclOpInfo failed, rank=%u, tag=%s",
                param->topoInfo.rank, param->tag);
        }
    }

    if (HcommThreadNotifyRecordOnThread(
        thread, param->resCtx->cpuThreadOnAicpu, 0) != HCCL_SUCCESS) {
        HCCL_ERROR("record host done notify failed, tag=%s", param->tag);
        EndProfilingIfNeeded();
        (void)HcommBatchModeEnd(param->tag);
        (void)HcommReleaseComm(param->commName);
        return 1;
    }

    if (profilingInitialized) {
        if (HcommProfilingReportMainStreamAndLastTask(thread) != HCCL_SUCCESS) {
            HCCL_WARNING("HcommProfilingReportMainStreamAndLastTask failed, rank=%u, tag=%s",
                param->topoInfo.rank, param->tag);
        }
    }
    if (HcommBatchModeEnd(param->tag) != HCCL_SUCCESS) {
        HCCL_ERROR("HcommBatchModeEnd failed, tag=%s", param->tag);
        EndProfilingIfNeeded();
        (void)HcommReleaseComm(param->commName);
        return 1;
    }

    EndProfilingIfNeeded();

    if (HcommReleaseComm(param->commName) != HCCL_SUCCESS) {
        HCCL_ERROR("HcommReleaseComm failed, commName=%s", param->commName);
        return 1;
    }

    return 0;
}
