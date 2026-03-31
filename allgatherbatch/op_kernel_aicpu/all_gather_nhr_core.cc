#include "all_gather_nhr_core.h"

#include "log.h"

namespace ops_hccl_allgatherbatch {

namespace {

const char *ToCommModeString(BatchCommMode commMode)
{
    switch (commMode) {
        case BatchCommMode::kSingleServer:
            return "single-server";
        case BatchCommMode::kCrossServer:
            return "cross-server";
        default:
            return "unknown";
    }
}

}  // namespace

AllGatherNHRCore::AllGatherNHRCore(const OpParam &param, AlgResourceCtx &resCtx, uint64_t packedBytes)
    : param_(param), resCtx_(resCtx), packedBytes_(packedBytes)
{
}

HcclResult AllGatherNHRCore::ValidateCommState() const
{
    if (param_.topoInfo.rankSize == 0) {
        HCCL_ERROR("rankSize is zero");
        return HCCL_E_PARA;
    }
    if (param_.topoInfo.rank >= param_.topoInfo.rankSize) {
        HCCL_ERROR("rank=%u is out of range, rankSize=%u", param_.topoInfo.rank, param_.topoInfo.rankSize);
        return HCCL_E_PARA;
    }
    if (packedBytes_ == 0) {
        HCCL_ERROR("packedBytes is zero");
        return HCCL_E_PARA;
    }
    if (param_.commMode == BatchCommMode::kUnknown) {
        HCCL_ERROR("commMode is unknown");
        return HCCL_E_INTERNAL;
    }

    const uint64_t totalBytes = packedBytes_ * param_.topoInfo.rankSize;
    if (resCtx_.localBuffer.addr == nullptr || resCtx_.localBuffer.size < totalBytes) {
        HCCL_ERROR("localBuffer is too small, need=%llu, actual=%llu",
            static_cast<unsigned long long>(totalBytes),
            static_cast<unsigned long long>(resCtx_.localBuffer.size));
        return HCCL_E_INTERNAL;
    }
    if (param_.topoInfo.rankSize > 1 && resCtx_.channelCount + 1 < param_.topoInfo.rankSize) {
        HCCL_WARNING("channelCount=%u is insufficient for rankSize=%u", resCtx_.channelCount, param_.topoInfo.rankSize);
        return HCCL_E_NOT_SUPPORT;
    }
    return HCCL_SUCCESS;
}

uint32_t AllGatherNHRCore::CalcStepNum(uint32_t rankSize) const
{
    uint32_t nSteps = 0;
    for (uint32_t tmp = rankSize - 1; tmp != 0; tmp >>= 1) {
        ++nSteps;
    }
    return nSteps;
}

HcclResult AllGatherNHRCore::GetStepInfo(uint32_t step, uint32_t nSteps, NHRStepInfo &stepInfo) const
{
    const uint32_t rank = param_.topoInfo.rank;
    const uint32_t rankSize = param_.topoInfo.rankSize;
    const uint32_t deltaRank = 1U << (nSteps - 1U - step);
    const uint32_t fromRank = (rank + rankSize - deltaRank) % rankSize;
    const uint32_t toRank = (rank + deltaRank) % rankSize;

    stepInfo.step = step;
    stepInfo.fromRank = fromRank;
    stepInfo.toRank = toRank;
    stepInfo.sliceCount = 1;
    stepInfo.txItemOrder.clear();
    stepInfo.rxItemOrder.clear();

    // 当前 public-header 方案还没有做 slice merge，因此先按 item 顺序建立最小步进计划。
    for (uint32_t itemIdx = 0; itemIdx < param_.itemCount; ++itemIdx) {
        stepInfo.txItemOrder.push_back(itemIdx);
        stepInfo.rxItemOrder.push_back(itemIdx);
    }
    return HCCL_SUCCESS;
}

HcclResult AllGatherNHRCore::BuildStepPlan(std::vector<NHRStepInfo> &stepPlan) const
{
    const uint32_t rankSize = param_.topoInfo.rankSize;
    if (rankSize <= 1) {
        stepPlan.clear();
        return HCCL_SUCCESS;
    }

    const uint32_t nSteps = CalcStepNum(rankSize);
    stepPlan.clear();
    stepPlan.reserve(nSteps);
    for (uint32_t step = 0; step < nSteps; ++step) {
        NHRStepInfo stepInfo;
        HCCL_CHK_RET(GetStepInfo(step, nSteps, stepInfo));
        stepPlan.push_back(stepInfo);
    }
    return HCCL_SUCCESS;
}

const ChannelResource *AllGatherNHRCore::FindChannel(uint32_t remoteRank) const
{
    for (uint32_t idx = 0; idx < resCtx_.channelCount; ++idx) {
        if (resCtx_.channels[idx].remoteRank == remoteRank) {
            return &resCtx_.channels[idx];
        }
    }
    return nullptr;
}

uint8_t *AllGatherNHRCore::GetRankBuffer(uint32_t rank) const
{
    return static_cast<uint8_t *>(resCtx_.localBuffer.addr) + (packedBytes_ * rank);
}

bool AllGatherNHRCore::IsCrossServerChannel(const ChannelResource &channel) const
{
    return channel.remoteServerIdx != param_.topoInfo.serverIdx;
}

uint32_t AllGatherNHRCore::CountChannelsByScope(bool crossServer) const
{
    uint32_t count = 0;
    for (uint32_t idx = 0; idx < resCtx_.channelCount; ++idx) {
        if (IsCrossServerChannel(resCtx_.channels[idx]) == crossServer) {
            ++count;
        }
    }
    return count;
}

HcclResult AllGatherNHRCore::NotifyReadyByScope(bool crossServer) const
{
    const char *scope = crossServer ? "cross-server" : "intra-server";

    // 先按 scope 分开处理，对后面区分 server 内和 server 间不同同步策略更友好。
    for (uint32_t remoteRank = 0; remoteRank < param_.topoInfo.rankSize; ++remoteRank) {
        if (remoteRank == param_.topoInfo.rank) {
            continue;
        }
        const ChannelResource *channel = FindChannel(remoteRank);
        if (channel == nullptr) {
            HCCL_ERROR("channel to remoteRank=%u is missing", remoteRank);
            return HCCL_E_NOT_FOUND;
        }
        if (IsCrossServerChannel(*channel) != crossServer) {
            continue;
        }

        const int32_t ret = HcommChannelNotifyRecordOnThread(
            resCtx_.threadHandle,
            channel->handle,
            channel->remoteNotifyIdx);
        if (ret != HCCL_SUCCESS) {
            HCCL_ERROR("%s notify record failed, remoteRank=%u, ret=%d", scope, remoteRank, ret);
            return static_cast<HcclResult>(ret);
        }
    }
    return HCCL_SUCCESS;
}

HcclResult AllGatherNHRCore::ReadRemoteRanksByScope(bool crossServer) const
{
    const char *scope = crossServer ? "cross-server" : "intra-server";

    // 当前同 server 和跨 server 仍复用同一套 remote-read 原语，但这里已经把两类链路分 scope 串起来了。
    for (uint32_t remoteRank = 0; remoteRank < param_.topoInfo.rankSize; ++remoteRank) {
        if (remoteRank == param_.topoInfo.rank) {
            continue;
        }
        const ChannelResource *channel = FindChannel(remoteRank);
        if (channel == nullptr) {
            HCCL_ERROR("channel to remoteRank=%u is missing", remoteRank);
            return HCCL_E_NOT_FOUND;
        }
        if (IsCrossServerChannel(*channel) != crossServer) {
            continue;
        }
        if (channel->remoteBuffer.addr == nullptr || channel->remoteBuffer.size < (packedBytes_ * param_.topoInfo.rankSize)) {
            HCCL_ERROR("%s remote buffer for rank=%u is too small", scope, remoteRank);
            return HCCL_E_INTERNAL;
        }

        int32_t ret = HcommChannelNotifyWaitOnThread(
            resCtx_.threadHandle,
            channel->handle,
            channel->localNotifyIdx,
            kAllGatherBatchCustomTimeoutMs);
        if (ret != HCCL_SUCCESS) {
            HCCL_ERROR("%s notify wait failed, remoteRank=%u, ret=%d", scope, remoteRank, ret);
            return static_cast<HcclResult>(ret);
        }

        void *dst = GetRankBuffer(remoteRank);
        const void *src = static_cast<const uint8_t *>(channel->remoteBuffer.addr) + (packedBytes_ * remoteRank);
        ret = HcommReadOnThread(resCtx_.threadHandle, channel->handle, dst, src, packedBytes_);
        if (ret != HCCL_SUCCESS) {
            HCCL_ERROR("%s remote read failed, remoteRank=%u, ret=%d", scope, remoteRank, ret);
            return static_cast<HcclResult>(ret);
        }
    }
    return HCCL_SUCCESS;
}

HcclResult AllGatherNHRCore::RunAsync()
{
    if (param_.topoInfo.rankSize == 1) {
        HCCL_INFO("NHR core fast path: rankSize=1, no inter-rank communication needed");
        return HCCL_SUCCESS;
    }

    HCCL_CHK_RET(ValidateCommState());

    std::vector<NHRStepInfo> stepPlan;
    HCCL_CHK_RET(BuildStepPlan(stepPlan));

    const uint32_t intraServerChannels = CountChannelsByScope(false);
    const uint32_t crossServerChannels = CountChannelsByScope(true);
    HCCL_INFO("NHR core step plan ready: rank=%u, rankSize=%u, commMode=%s, intraServerRankCount=%u, crossServerRankCount=%u, steps=%u, packedBytes=%llu, channelCount=%u, intraServerChannels=%u, crossServerChannels=%u",
        param_.topoInfo.rank,
        param_.topoInfo.rankSize,
        ToCommModeString(param_.commMode),
        param_.intraServerRankCount,
        param_.crossServerRankCount,
        static_cast<unsigned int>(stepPlan.size()),
        static_cast<unsigned long long>(packedBytes_),
        resCtx_.channelCount,
        intraServerChannels,
        crossServerChannels);

    // 这里先保留 NHR 的控制层边界，但数据面已经按同 server / 跨 server 两个 scope 分开组织。
    HCCL_CHK_RET(NotifyReadyByScope(false));
    HCCL_CHK_RET(ReadRemoteRanksByScope(false));
    if (param_.commMode == BatchCommMode::kCrossServer) {
        HCCL_CHK_RET(NotifyReadyByScope(true));
        HCCL_CHK_RET(ReadRemoteRanksByScope(true));
    }
    return HCCL_SUCCESS;
}

}  // namespace ops_hccl_allgatherbatch
