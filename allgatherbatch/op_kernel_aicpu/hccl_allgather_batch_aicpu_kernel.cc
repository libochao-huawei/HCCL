#include "common.h"
#include "exec_op.h"

namespace {

using namespace ops_hccl_allgatherbatch;

// kernel 入口先把 Host 下发的动态资源协议再核一遍，避免旧 cache 或错误 launch 参数把问题带进 Device 主循环。
HcclResult ValidateKernelResourceCtx(const OpParam &param)
{
    const AlgResourceCtx &resCtx = *param.resCtx;

    HCCL_CHK_RET(ValidateBasicResourceCtx(param, resCtx, "AICPU kernel resCtx"));
    return ValidateRemoteChannelResources(param, resCtx, "AICPU kernel");
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
    if (ValidateKernelResourceCtx(*param) != HCCL_SUCCESS) {
        return 1;
    }

    const ResourceStats stats = CollectResourceStats(*param, *param->resCtx);
    HCCL_INFO("AICPU kernel enter: rank=%u, rankSize=%u, commMode=%s, serverIdx=%u, serverCount=%u, intraServerRankCount=%u, crossServerRankCount=%u, channelCount=%u, crossServerChannels=%u, perRankCapacity=%llu, maxWindowBytes=%llu, totalInputBytes=%llu, windowBytes=%llu, hccs=%u, roce=%u, pcie=%u, sio=%u",
        param->topoInfo.rank,
        param->topoInfo.rankSize,
        ToCommModeString(param->commMode),
        param->topoInfo.serverIdx,
        param->topoInfo.serverCount,
        param->intraServerRankCount,
        param->crossServerRankCount,
        param->resCtx->channelCount,
        stats.crossServerChannels,
        static_cast<unsigned long long>(stats.perRankCapacity),
        static_cast<unsigned long long>(stats.maxWindowBytes),
        static_cast<unsigned long long>(param->totalInputBytes),
        static_cast<unsigned long long>(param->windowBytes),
        stats.hccsChannels,
        stats.roceChannels,
        stats.pcieChannels,
        stats.sioChannels);

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






