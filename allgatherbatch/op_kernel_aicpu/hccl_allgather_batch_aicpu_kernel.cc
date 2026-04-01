#include "common.h"
#include "exec_op.h"

namespace {

using namespace ops_hccl_allgatherbatch;

HcclResult ValidateKernelResourceCtx(const OpParam &param)
{
    const AlgResourceCtx &resCtx = *param.resCtx;
    const uint32_t crossServerChannels = CountCrossServerChannels(param.topoInfo, resCtx);
    const uint64_t perRankCapacity = GetPerRankWindowCapacity(param, resCtx);
    const uint64_t maxWindowBytes = GetMaxWindowBytes(param, resCtx);
    if (resCtx.threadHandle == 0) {
        HCCL_ERROR("AICPU kernel received invalid threadHandle");
        return HCCL_E_INTERNAL;
    }
    if (resCtx.localBuffer.addr == nullptr || resCtx.localBuffer.size == 0) {
        HCCL_ERROR("AICPU kernel received invalid localBuffer");
        return HCCL_E_INTERNAL;
    }
    if (param.topoInfo.rankSize > 1 && resCtx.channelCount + 1 < param.topoInfo.rankSize) {
        HCCL_ERROR("AICPU kernel channelCount=%u is insufficient for rankSize=%u",
            resCtx.channelCount,
            param.topoInfo.rankSize);
        return HCCL_E_INTERNAL;
    }
    if (maxWindowBytes == 0) {
        HCCL_ERROR("AICPU kernel maxWindowBytes is zero");
        return HCCL_E_INTERNAL;
    }

    if (CountRecognizedProtocols(resCtx) != resCtx.channelCount) {
        HCCL_ERROR("AICPU kernel protocol distribution mismatch, recognized=%u, channelCount=%u",
            CountRecognizedProtocols(resCtx),
            resCtx.channelCount);
        return HCCL_E_INTERNAL;
    }
    if (param.commMode == BatchCommMode::kSingleServer && crossServerChannels != 0) {
        HCCL_ERROR("AICPU kernel single-server mode unexpectedly has crossServerChannels=%u", crossServerChannels);
        return HCCL_E_INTERNAL;
    }
    if (param.commMode == BatchCommMode::kCrossServer && crossServerChannels != param.crossServerRankCount) {
        HCCL_ERROR("AICPU kernel cross-server channel mismatch, channels=%u, expected=%u",
            crossServerChannels,
            param.crossServerRankCount);
        return HCCL_E_INTERNAL;
    }

    // Device 入口在真正执行前把资源容量再核一遍，避免窗口大小和远端 buffer 大小不一致时进入通信原语。
    for (uint32_t idx = 0; idx < resCtx.channelCount; ++idx) {
        const ChannelResource &channel = resCtx.channels[idx];
        if (channel.remoteRank == param.topoInfo.rank || channel.remoteRank >= param.topoInfo.rankSize) {
            HCCL_ERROR("AICPU kernel channel %u remoteRank=%u is invalid", idx, channel.remoteRank);
            return HCCL_E_INTERNAL;
        }
        if (channel.remoteSuperPodIdx != param.topoInfo.superPodIdx) {
            HCCL_ERROR("AICPU kernel channel %u crosses superPod unexpectedly, local=%u, remote=%u",
                idx,
                param.topoInfo.superPodIdx,
                channel.remoteSuperPodIdx);
            return HCCL_E_INTERNAL;
        }
        if (channel.remoteBuffer.addr == nullptr || channel.remoteBuffer.size == 0) {
            HCCL_ERROR("AICPU kernel channel %u remoteBuffer is invalid", idx);
            return HCCL_E_INTERNAL;
        }
        if (channel.remoteBuffer.size < (maxWindowBytes * param.topoInfo.rankSize)) {
            HCCL_ERROR("AICPU kernel channel %u remoteBuffer too small, need=%llu, actual=%llu",
                idx,
                static_cast<unsigned long long>(maxWindowBytes * param.topoInfo.rankSize),
                static_cast<unsigned long long>(channel.remoteBuffer.size));
            return HCCL_E_INTERNAL;
        }
    }
    return HCCL_SUCCESS;
}

}  // namespace

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
    if (ValidateKernelResourceCtx(*param) != HCCL_SUCCESS) {
        return 1;
    }

    HCCL_INFO("AICPU kernel enter: rank=%u, rankSize=%u, commMode=%s, serverIdx=%u, serverCount=%u, intraServerRankCount=%u, crossServerRankCount=%u, channelCount=%u, crossServerChannels=%u, perRankCapacity=%llu, totalInputBytes=%llu, windowBytes=%llu, hccs=%u, roce=%u, pcie=%u, sio=%u",
        param->topoInfo.rank,
        param->topoInfo.rankSize,
        ToCommModeString(param->commMode),
        param->topoInfo.serverIdx,
        param->topoInfo.serverCount,
        param->intraServerRankCount,
        param->crossServerRankCount,
        param->resCtx->channelCount,
        CountCrossServerChannels(param->topoInfo, *param->resCtx),
        static_cast<unsigned long long>(GetPerRankWindowCapacity(*param, *param->resCtx)),
        static_cast<unsigned long long>(param->totalInputBytes),
        static_cast<unsigned long long>(param->windowBytes),
        CountChannelsByProtocol(*param->resCtx, COMM_PROTOCOL_HCCS),
        CountChannelsByProtocol(*param->resCtx, COMM_PROTOCOL_ROCE),
        CountChannelsByProtocol(*param->resCtx, COMM_PROTOCOL_PCIE),
        CountChannelsByProtocol(*param->resCtx, COMM_PROTOCOL_SIO));

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

