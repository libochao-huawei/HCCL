#include "all_gather_nhr_core.h"

#include "log.h"

namespace ops_hccl_allgatherbatch {

namespace {

const char *ToScopeString(bool crossServer)
{
    return crossServer ? "cross-server" : "intra-server";
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
    if (!IsValidCommMode(param_.commMode)) {
        HCCL_ERROR("commMode is invalid");
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

HcclResult AllGatherNHRCore::ValidateChannelMetadata() const
{
    const uint32_t intraServerChannels = CountChannelsByScope(false);
    const uint32_t crossServerChannels = CountChannelsByScope(true);
    const uint32_t expectedIntraServerChannels = (param_.intraServerRankCount == 0) ? 0 : (param_.intraServerRankCount - 1);

    // 这里再从通信层视角收一次 Host 下发的模式和资源元数据，避免协议或 server 归属异常直接进入 notify/read。
    for (uint32_t idx = 0; idx < resCtx_.channelCount; ++idx) {
        const ChannelResource &channel = resCtx_.channels[idx];
        if (channel.protocol == COMM_PROTOCOL_RESERVED) {
            HCCL_ERROR("channel %u has reserved protocol", idx);
            return HCCL_E_INTERNAL;
        }
        if (channel.remoteSuperPodIdx != param_.topoInfo.superPodIdx) {
            HCCL_ERROR("channel %u crosses superPod unexpectedly, local=%u remote=%u",
                idx,
                param_.topoInfo.superPodIdx,
                channel.remoteSuperPodIdx);
            return HCCL_E_INTERNAL;
        }
        if (channel.remoteRank >= param_.topoInfo.rankSize) {
            HCCL_ERROR("channel %u remoteRank=%u is out of range, rankSize=%u",
                idx,
                channel.remoteRank,
                param_.topoInfo.rankSize);
            return HCCL_E_INTERNAL;
        }
    }

    if (intraServerChannels + crossServerChannels != resCtx_.channelCount) {
        HCCL_ERROR("channel scope split is inconsistent, intra=%u, cross=%u, channelCount=%u",
            intraServerChannels,
            crossServerChannels,
            resCtx_.channelCount);
        return HCCL_E_INTERNAL;
    }
    if (intraServerChannels != expectedIntraServerChannels) {
        HCCL_ERROR("intra-server channel count mismatch, expected=%u, actual=%u",
            expectedIntraServerChannels,
            intraServerChannels);
        return HCCL_E_INTERNAL;
    }
    if (crossServerChannels != param_.crossServerRankCount) {
        HCCL_ERROR("cross-server channel count mismatch, expected=%u, actual=%u",
            param_.crossServerRankCount,
            crossServerChannels);
        return HCCL_E_INTERNAL;
    }
    if (param_.commMode == BatchCommMode::kSingleServer && crossServerChannels != 0) {
        HCCL_ERROR("single-server mode unexpectedly has cross-server channels=%u", crossServerChannels);
        return HCCL_E_INTERNAL;
    }
    if (param_.commMode == BatchCommMode::kCrossServer && crossServerChannels == 0) {
        HCCL_ERROR("cross-server mode has no cross-server channels");
        return HCCL_E_INTERNAL;
    }
    return HCCL_SUCCESS;
}

HcclResult AllGatherNHRCore::ValidateProtocolDistribution() const
{
    const uint32_t intraRecognized = CountRecognizedChannelsByScope(false);
    const uint32_t crossRecognized = CountRecognizedChannelsByScope(true);
    const uint32_t intraChannels = CountChannelsByScope(false);
    const uint32_t crossChannels = CountChannelsByScope(true);

    // 协议维度已经进入控制流，这里要求 scope 内的 channel 都能被已知协议解释，不再只做日志统计。
    if (intraRecognized != intraChannels) {
        HCCL_ERROR("intra-server protocol distribution mismatch, recognized=%u, actual=%u",
            intraRecognized,
            intraChannels);
        return HCCL_E_INTERNAL;
    }
    if (crossRecognized != crossChannels) {
        HCCL_ERROR("cross-server protocol distribution mismatch, recognized=%u, actual=%u",
            crossRecognized,
            crossChannels);
        return HCCL_E_INTERNAL;
    }
    if (resCtx_.channelCount != (intraRecognized + crossRecognized)) {
        HCCL_ERROR("protocol distribution total mismatch, recognized=%u, channelCount=%u",
            intraRecognized + crossRecognized,
            resCtx_.channelCount);
        return HCCL_E_INTERNAL;
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

uint32_t AllGatherNHRCore::CountChannelsByProtocol(bool crossServer, CommProtocol protocol) const
{
    uint32_t count = 0;
    for (uint32_t idx = 0; idx < resCtx_.channelCount; ++idx) {
        const ChannelResource &channel = resCtx_.channels[idx];
        if (IsCrossServerChannel(channel) == crossServer && channel.protocol == protocol) {
            ++count;
        }
    }
    return count;
}

uint32_t AllGatherNHRCore::CountRecognizedChannelsByScope(bool crossServer) const
{
    return CountChannelsByProtocol(crossServer, COMM_PROTOCOL_HCCS) +
        CountChannelsByProtocol(crossServer, COMM_PROTOCOL_ROCE) +
        CountChannelsByProtocol(crossServer, COMM_PROTOCOL_PCIE) +
        CountChannelsByProtocol(crossServer, COMM_PROTOCOL_SIO);
}

HcclResult AllGatherNHRCore::NotifyReadyByScopeAndProtocol(bool crossServer, CommProtocol protocol) const
{
    const char *scope = ToScopeString(crossServer);
    const char *protocolName = ToProtocolString(protocol);

    // 当前不同协议仍复用同一套原语，但先把协议维度变成显式分支点，后面细化链路策略时不用重拆控制骨架。
    for (uint32_t remoteRank = 0; remoteRank < param_.topoInfo.rankSize; ++remoteRank) {
        if (remoteRank == param_.topoInfo.rank) {
            continue;
        }
        const ChannelResource *channel = FindChannel(remoteRank);
        if (channel == nullptr) {
            HCCL_ERROR("channel to remoteRank=%u is missing", remoteRank);
            return HCCL_E_NOT_FOUND;
        }
        if (IsCrossServerChannel(*channel) != crossServer || channel->protocol != protocol) {
            continue;
        }

        const int32_t ret = HcommChannelNotifyRecordOnThread(
            resCtx_.threadHandle,
            channel->handle,
            channel->remoteNotifyIdx);
        if (ret != HCCL_SUCCESS) {
            HCCL_ERROR("%s/%s notify record failed, remoteRank=%u, ret=%d", scope, protocolName, remoteRank, ret);
            return static_cast<HcclResult>(ret);
        }
    }
    return HCCL_SUCCESS;
}

HcclResult AllGatherNHRCore::ReadRemoteRanksByScopeAndProtocol(bool crossServer, CommProtocol protocol) const
{
    const char *scope = ToScopeString(crossServer);
    const char *protocolName = ToProtocolString(protocol);

    for (uint32_t remoteRank = 0; remoteRank < param_.topoInfo.rankSize; ++remoteRank) {
        if (remoteRank == param_.topoInfo.rank) {
            continue;
        }
        const ChannelResource *channel = FindChannel(remoteRank);
        if (channel == nullptr) {
            HCCL_ERROR("channel to remoteRank=%u is missing", remoteRank);
            return HCCL_E_NOT_FOUND;
        }
        if (IsCrossServerChannel(*channel) != crossServer || channel->protocol != protocol) {
            continue;
        }
        if (channel->remoteBuffer.addr == nullptr || channel->remoteBuffer.size < (packedBytes_ * param_.topoInfo.rankSize)) {
            HCCL_ERROR("%s/%s remote buffer for rank=%u is too small", scope, protocolName, remoteRank);
            return HCCL_E_INTERNAL;
        }

        int32_t ret = HcommChannelNotifyWaitOnThread(
            resCtx_.threadHandle,
            channel->handle,
            channel->localNotifyIdx,
            kAllGatherBatchCustomTimeoutMs);
        if (ret != HCCL_SUCCESS) {
            HCCL_ERROR("%s/%s notify wait failed, remoteRank=%u, ret=%d", scope, protocolName, remoteRank, ret);
            return static_cast<HcclResult>(ret);
        }

        void *dst = GetRankBuffer(remoteRank);
        const void *src = static_cast<const uint8_t *>(channel->remoteBuffer.addr) + (packedBytes_ * remoteRank);
        ret = HcommReadOnThread(resCtx_.threadHandle, channel->handle, dst, src, packedBytes_);
        if (ret != HCCL_SUCCESS) {
            HCCL_ERROR("%s/%s remote read failed, remoteRank=%u, ret=%d", scope, protocolName, remoteRank, ret);
            return static_cast<HcclResult>(ret);
        }
    }
    return HCCL_SUCCESS;
}

HcclResult AllGatherNHRCore::RunScope(bool crossServer) const
{
    static const CommProtocol kProtocolOrder[] = {
        COMM_PROTOCOL_HCCS,
        COMM_PROTOCOL_ROCE,
        COMM_PROTOCOL_PCIE,
        COMM_PROTOCOL_SIO,
    };

    const char *scope = ToScopeString(crossServer);
    for (CommProtocol protocol : kProtocolOrder) {
        const uint32_t channelCount = CountChannelsByProtocol(crossServer, protocol);
        if (channelCount == 0) {
            continue;
        }

        HCCL_INFO("NHR scope dispatch: scope=%s, protocol=%s, channels=%u, packedBytes=%llu",
            scope,
            ToProtocolString(protocol),
            channelCount,
            static_cast<unsigned long long>(packedBytes_));
        HCCL_CHK_RET(NotifyReadyByScopeAndProtocol(crossServer, protocol));
        HCCL_CHK_RET(ReadRemoteRanksByScopeAndProtocol(crossServer, protocol));
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
    HCCL_CHK_RET(ValidateChannelMetadata());
    HCCL_CHK_RET(ValidateProtocolDistribution());

    std::vector<NHRStepInfo> stepPlan;
    HCCL_CHK_RET(BuildStepPlan(stepPlan));

    const uint32_t intraServerChannels = CountChannelsByScope(false);
    const uint32_t crossServerChannels = CountChannelsByScope(true);
    HCCL_INFO("NHR core step plan ready: rank=%u, rankSize=%u, commMode=%s, intraServerRankCount=%u, crossServerRankCount=%u, steps=%u, packedBytes=%llu, channelCount=%u, intraServerChannels=%u, crossServerChannels=%u, hccs=%u, roce=%u, pcie=%u, sio=%u",
        param_.topoInfo.rank,
        param_.topoInfo.rankSize,
        ToCommModeString(param_.commMode),
        param_.intraServerRankCount,
        param_.crossServerRankCount,
        static_cast<unsigned int>(stepPlan.size()),
        static_cast<unsigned long long>(packedBytes_),
        resCtx_.channelCount,
        intraServerChannels,
        crossServerChannels,
        CountChannelsByProtocol(false, COMM_PROTOCOL_HCCS) + CountChannelsByProtocol(true, COMM_PROTOCOL_HCCS),
        CountChannelsByProtocol(false, COMM_PROTOCOL_ROCE) + CountChannelsByProtocol(true, COMM_PROTOCOL_ROCE),
        CountChannelsByProtocol(false, COMM_PROTOCOL_PCIE) + CountChannelsByProtocol(true, COMM_PROTOCOL_PCIE),
        CountChannelsByProtocol(false, COMM_PROTOCOL_SIO) + CountChannelsByProtocol(true, COMM_PROTOCOL_SIO));

    // 这里先保留 NHR 的控制层边界，但数据面已经按 scope + protocol 两个维度分开组织。
    HCCL_CHK_RET(RunScope(false));
    if (param_.commMode == BatchCommMode::kCrossServer) {
        HCCL_CHK_RET(RunScope(true));
    }
    return HCCL_SUCCESS;
}

}  // namespace ops_hccl_allgatherbatch
