#include "common.h"
#include "exec_op.h"

extern "C" unsigned int HcclAllGatherBatchAicpuKernel(
    ops_hccl_allgatherbatch::OpParam *param)
{
    using namespace ops_hccl_allgatherbatch;

    if (param == nullptr || param->resCtx == nullptr) {
        return 1;
    }
    if (!IsValidCommMode(param->commMode)) {
        HCCL_ERROR("AICPU kernel received invalid commMode");
        return 1;
    }
    if (param->itemCount == 0 || param->itemCount > kAllGatherBatchMaxItems) {
        HCCL_ERROR("AICPU kernel received invalid itemCount=%u", param->itemCount);
        return 1;
    }
    if (param->topoInfo.rankSize == 0 || param->topoInfo.rank >= param->topoInfo.rankSize) {
        HCCL_ERROR("AICPU kernel received invalid rank/rankSize, rank=%u, rankSize=%u",
            param->topoInfo.rank,
            param->topoInfo.rankSize);
        return 1;
    }
    if (param->intraServerRankCount + param->crossServerRankCount != param->topoInfo.rankSize) {
        HCCL_ERROR("AICPU kernel received inconsistent rank distribution, intra=%u, cross=%u, rankSize=%u",
            param->intraServerRankCount,
            param->crossServerRankCount,
            param->topoInfo.rankSize);
        return 1;
    }
    if (param->resCtx->threadHandle == 0) {
        HCCL_ERROR("AICPU kernel received invalid threadHandle");
        return 1;
    }

    HCCL_INFO("AICPU kernel enter: rank=%u, rankSize=%u, commMode=%s, serverIdx=%u, serverCount=%u, intraServerRankCount=%u, crossServerRankCount=%u, channelCount=%u, totalInputBytes=%llu, windowBytes=%llu",
        param->topoInfo.rank,
        param->topoInfo.rankSize,
        ToCommModeString(param->commMode),
        param->topoInfo.serverIdx,
        param->topoInfo.serverCount,
        param->intraServerRankCount,
        param->crossServerRankCount,
        param->resCtx->channelCount,
        static_cast<unsigned long long>(param->totalInputBytes),
        static_cast<unsigned long long>(param->windowBytes));

    // Device 入口负责把 Host 下发的控制协议转成完整的设备侧执行时序。
    if (HcommAcquireComm(param->commName) != HCCL_SUCCESS) {
        HCCL_ERROR("HcommAcquireComm failed, commName=%s", param->commName);
        return 1;
    }

    ThreadHandle thread = param->resCtx->threadHandle;
    if (HcommBatchModeStart(param->tag) != HCCL_SUCCESS) {
        HCCL_ERROR("HcommBatchModeStart failed, tag=%s", param->tag);
        (void)HcommReleaseComm(param->commName);
        return 1;
    }

    if (HcommAclrtNotifyWaitOnThread(
            thread,
            param->resCtx->controlNotifyIds[kAllGatherBatchControlNotifyStart],
            kAllGatherBatchCustomTimeoutMs) != HCCL_SUCCESS) {
        HCCL_ERROR("wait host start notify failed, tag=%s", param->tag);
        (void)HcommBatchModeEnd(param->tag);
        (void)HcommReleaseComm(param->commName);
        return 1;
    }

    HcclResult ret = ExecOp(*param, param->resCtx);
    if (ret != HCCL_SUCCESS) {
        HCCL_ERROR("ExecOp failed, ret=%d", static_cast<int>(ret));
        (void)HcommBatchModeEnd(param->tag);
        (void)HcommReleaseComm(param->commName);
        return 1;
    }

    if (HcommAclrtNotifyRecordOnThread(
            thread,
            param->resCtx->controlNotifyIds[kAllGatherBatchControlNotifyDone]) != HCCL_SUCCESS) {
        HCCL_ERROR("record host done notify failed, tag=%s", param->tag);
        (void)HcommBatchModeEnd(param->tag);
        (void)HcommReleaseComm(param->commName);
        return 1;
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

    HCCL_INFO("AICPU kernel done: rank=%u, commMode=%s, itemCount=%u",
        param->topoInfo.rank,
        ToCommModeString(param->commMode),
        param->itemCount);
    return 0;
}
