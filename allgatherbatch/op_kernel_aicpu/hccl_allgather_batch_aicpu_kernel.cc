#include "common.h"
#include "exec_op.h"
#include "profiling.h"

namespace {

using namespace ops_hccl_allgatherbatch;

void ResetBatchCallProfiling(BatchCallProfiling &profiling, const OpParam &param, const AlgResourceCtx &resCtx)
{
    profiling = {};
    profiling.rank = param.topoInfo.rank;
    profiling.rankSize = param.topoInfo.rankSize;
    profiling.itemCount = param.itemCount;
    profiling.commMode = param.commMode;
    profiling.totalInputBytes = param.totalInputBytes;
    profiling.localBufferBytes = resCtx.localBuffer.size;
    profiling.maxWindowBytes = GetMaxWindowBytes(param, resCtx);
}

void PrintBatchCallProfiling(const BatchCallProfiling &profiling)
{
    HCCL_RUN_INFO(
        "AGB_CALL rank=%u/%u items=%u windows=%u comm=%s us{kernel=%llu exec=%llu pack=%llu hd=%llu unpack=%llu} bytes{input=%llu local=%llu maxWin=%llu}",
        profiling.rank,
        profiling.rankSize,
        profiling.itemCount,
        profiling.windowCount,
        ToCommModeString(profiling.commMode),
        static_cast<unsigned long long>(profiling.kernelUs),
        static_cast<unsigned long long>(profiling.execUs),
        static_cast<unsigned long long>(profiling.packUs),
        static_cast<unsigned long long>(profiling.hdStageUs),
        static_cast<unsigned long long>(profiling.unpackUs),
        static_cast<unsigned long long>(profiling.totalInputBytes),
        static_cast<unsigned long long>(profiling.localBufferBytes),
        static_cast<unsigned long long>(profiling.maxWindowBytes));
}

}  // namespace

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

    BatchCallProfiling profiling {};
    ResetBatchCallProfiling(profiling, *param, *param->resCtx);

    if (HcommAcquireComm(param->commName) != HCCL_SUCCESS) {
        HCCL_ERROR("HcommAcquireComm failed, commName=%s", param->commName);
        return 1;
    }

    ThreadHandle thread = param->resCtx->mainThreadHandle;
    const bool profilingOn = IsProfilingEnabled();
    ThreadHandle profilingThreads[1] = {thread};
    bool profilingInitialized = false;
    auto EndProfilingIfNeeded = [&]() {
        if (!profilingInitialized) {
            return;
        }
        if (HcommProfilingEnd(profilingThreads, 1) != HCCL_SUCCESS) {
            HCCL_WARNING("HcommProfilingEnd failed, rank=%u, tag=%s", param->topoInfo.rank, param->tag);
        }
        profilingInitialized = false;
    };
    if (HcommBatchModeStart(param->tag) != HCCL_SUCCESS) {
        HCCL_ERROR("HcommBatchModeStart failed, tag=%s", param->tag);
        (void)HcommReleaseComm(param->commName);
        return 1;
    }

    if (HcommAclrtNotifyWaitOnThread(
            thread,
            param->controlNotifyIds[kAllGatherBatchControlNotifyStart],
            CUSTOM_TIMEOUT) != HCCL_SUCCESS) {
        HCCL_ERROR("wait host start notify failed, tag=%s", param->tag);
        (void)HcommBatchModeEnd(param->tag);
        (void)HcommReleaseComm(param->commName);
        return 1;
    }

    if (profilingOn) {
        if (HcommProfilingInit(profilingThreads, 1) == HCCL_SUCCESS) {
            profilingInitialized = true;
        } else {
            HCCL_WARNING("HcommProfilingInit failed, rank=%u, tag=%s", param->topoInfo.rank, param->tag);
        }
        if (profilingInitialized) {
            if (HcommProfilingReportMainStreamAndFirstTask(thread) != HCCL_SUCCESS) {
                HCCL_WARNING("HcommProfilingReportMainStreamAndFirstTask failed, rank=%u, tag=%s",
                    param->topoInfo.rank, param->tag);
            }
            const uint64_t deviceBeginTime = HcommGetProfilingSysCycleTime();
            HcomProInfoTmp info {};
            FillProfilingInfo(info, *param, deviceBeginTime, 0);
            if (HcommProfilingReportDeviceHcclOpInfo(info) != HCCL_SUCCESS) {
                HCCL_WARNING("HcommProfilingReportDeviceHcclOpInfo failed, rank=%u, tag=%s",
                    param->topoInfo.rank, param->tag);
            }
        }
    }

    const uint64_t kernelStartUs = GetCurrentTimeUs();
    HcclResult ret = ExecOp(*param, param->resCtx, profiling);
    if (ret != HCCL_SUCCESS) {
        HCCL_ERROR("ExecOp failed, ret=%d", static_cast<int>(ret));
        EndProfilingIfNeeded();
        (void)HcommBatchModeEnd(param->tag);
        (void)HcommReleaseComm(param->commName);
        return 1;
    }

    HCCL_INFO("kernel done notify begin: rank=%u, tag=%s",
        param->topoInfo.rank,
        param->tag);
    if (HcommAclrtNotifyRecordOnThread(
            thread,
            param->controlNotifyIds[kAllGatherBatchControlNotifyDone]) != HCCL_SUCCESS) {
        HCCL_ERROR("record host done notify failed, tag=%s", param->tag);
        EndProfilingIfNeeded();
        (void)HcommBatchModeEnd(param->tag);
        (void)HcommReleaseComm(param->commName);
        return 1;
    }

    HCCL_INFO("kernel done notify end: rank=%u, tag=%s",
        param->topoInfo.rank,
        param->tag);
    if (profilingOn) {
        if (profilingInitialized) {
            if (HcommProfilingReportMainStreamAndLastTask(thread) != HCCL_SUCCESS) {
                HCCL_WARNING("HcommProfilingReportMainStreamAndLastTask failed, rank=%u, tag=%s",
                    param->topoInfo.rank, param->tag);
            }
        }
        EndProfilingIfNeeded();
    }
    if (HcommBatchModeEnd(param->tag) != HCCL_SUCCESS) {
        HCCL_ERROR("HcommBatchModeEnd failed, tag=%s", param->tag);
        (void)HcommReleaseComm(param->commName);
        return 1;
    }

    if (HcommReleaseComm(param->commName) != HCCL_SUCCESS) {
        HCCL_ERROR("HcommReleaseComm failed, commName=%s", param->commName);
        return 1;
    }

    profiling.kernelUs += (GetCurrentTimeUs() - kernelStartUs);
    PrintBatchCallProfiling(profiling);
    return 0;
}
