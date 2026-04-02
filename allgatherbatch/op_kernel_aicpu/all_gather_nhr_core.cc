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
    const ResourceStats stats = CollectResourceStats(param_, resCtx_);

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
    if (!HasConsistentRankDistribution(param_)) {
        HCCL_ERROR("NHR rank distribution is inconsistent, commMode=%s, intra=%u, cross=%u, rankSize=%u",
            ToCommModeString(param_.commMode),
            param_.intraServerRankCount,
            param_.crossServerRankCount,
            param_.topoInfo.rankSize);
        return HCCL_E_INTERNAL;
    }

    if (param_.windowBytes == 0) {
        HCCL_ERROR("windowBytes is zero");
        return HCCL_E_INTERNAL;
    }
    if (packedBytes_ > param_.windowBytes) {
        HCCL_ERROR("packedBytes=%llu exceeds param windowBytes=%llu",
            static_cast<unsigned long long>(packedBytes_),
            static_cast<unsigned long long>(param_.windowBytes));
        return HCCL_E_INTERNAL;
    }
    if (packedBytes_ > stats.maxWindowBytes) {
        HCCL_ERROR("packedBytes=%llu exceeds maxWindowBytes=%llu",
            static_cast<unsigned long long>(packedBytes_),
            static_cast<unsigned long long>(stats.maxWindowBytes));
        return HCCL_E_INTERNAL;
    }

    const uint64_t totalBytes = packedBytes_ * param_.topoInfo.rankSize;
    if (resCtx_.localBuffer.addr == nullptr || resCtx_.localBuffer.size < totalBytes) {
        HCCL_ERROR("localBuffer is too small, need=%llu, actual=%llu",
            static_cast<unsigned long long>(totalBytes),
            static_cast<unsigned long long>(resCtx_.localBuffer.size));
        return HCCL_E_INTERNAL;
    }
    // NHR 当前也是 fullmesh 前提，通信核心自己再收一次 channel 数，避免 Host/Device 协议漂移。
    if (resCtx_.channelCount != GetExpectedFullMeshChannelCount(param_)) {
        HCCL_ERROR("channelCount=%u mismatches expected fullmesh count=%u",
            resCtx_.channelCount,
            GetExpectedFullMeshChannelCount(param_));
        return HCCL_E_INTERNAL;
    }
    return HCCL_SUCCESS;
}

HcclResult AllGatherNHRCore::ValidateChannelMetadata() const
{
    const ResourceStats stats = CollectResourceStats(param_, resCtx_);
    const uint32_t expectedIntraServerChannels = (param_.intraServerRankCount == 0) ? 0 : (param_.intraServerRankCount - 1);

    // 这里再从通信层视角收一次 Host 下发的模式和资源元数据，避免协议或 server 归属异常直接进入 notify/read。
    for (uint32_t idx = 0; idx < resCtx_.channelCount; ++idx) {
        const ChannelResource &channel = GetChannel(resCtx_, idx);
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

    if (stats.intraServerChannels + stats.crossServerChannels != resCtx_.channelCount) {
        HCCL_ERROR("channel scope split is inconsistent, intra=%u, cross=%u, channelCount=%u",
            stats.intraServerChannels,
            stats.crossServerChannels,
            resCtx_.channelCount);
        return HCCL_E_INTERNAL;
    }
    if (stats.intraServerChannels != expectedIntraServerChannels) {
        HCCL_ERROR("intra-server channel count mismatch, expected=%u, actual=%u",
            expectedIntraServerChannels,
            stats.intraServerChannels);
        return HCCL_E_INTERNAL;
    }
    if (stats.crossServerChannels != param_.crossServerRankCount) {
        HCCL_ERROR("cross-server channel count mismatch, expected=%u, actual=%u",
            param_.crossServerRankCount,
            stats.crossServerChannels);
        return HCCL_E_INTERNAL;
    }
    if (param_.commMode == BatchCommMode::kSingleServer && stats.crossServerChannels != 0) {
        HCCL_ERROR("single-server mode unexpectedly has cross-server channels=%u", stats.crossServerChannels);
        return HCCL_E_INTERNAL;
    }
    if (param_.commMode == BatchCommMode::kCrossServer && stats.crossServerChannels == 0) {
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

HcclResult AllGatherNHRCore::ValidateStepPlan(const std::vector<NHRStepInfo> &stepPlan) const
{
    if (param_.topoInfo.rankSize <= 1) {
        return HCCL_SUCCESS;
    }
    if (stepPlan.empty()) {
        HCCL_ERROR("stepPlan is empty for rankSize=%u", param_.topoInfo.rankSize);
        return HCCL_E_INTERNAL;
    }

    // 这层检查把“算出来的 step 计划”正式变成执行前置条件，而不只是调试信息。
    for (const NHRStepInfo &stepInfo : stepPlan) {
        if (stepInfo.fromRank >= param_.topoInfo.rankSize || stepInfo.toRank >= param_.topoInfo.rankSize) {
            HCCL_ERROR("step[%u] rank mapping is invalid, fromRank=%u, toRank=%u, rankSize=%u",
                stepInfo.step,
                stepInfo.fromRank,
                stepInfo.toRank,
                param_.topoInfo.rankSize);
            return HCCL_E_INTERNAL;
        }
        if (stepInfo.txItemOrder.size() != param_.itemCount || stepInfo.rxItemOrder.size() != param_.itemCount) {
            HCCL_ERROR("step[%u] item order size mismatch, tx=%u, rx=%u, itemCount=%u",
                stepInfo.step,
                static_cast<uint32_t>(stepInfo.txItemOrder.size()),
                static_cast<uint32_t>(stepInfo.rxItemOrder.size()),
                param_.itemCount);
            return HCCL_E_INTERNAL;
        }
        for (uint32_t itemIdx : stepInfo.txItemOrder) {
            if (itemIdx >= param_.itemCount) {
                HCCL_ERROR("step[%u] tx item index=%u is invalid", stepInfo.step, itemIdx);
                return HCCL_E_INTERNAL;
            }
        }
        for (uint32_t itemIdx : stepInfo.rxItemOrder) {
            if (itemIdx >= param_.itemCount) {
                HCCL_ERROR("step[%u] rx item index=%u is invalid", stepInfo.step, itemIdx);
                return HCCL_E_INTERNAL;
            }
        }
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
        if (GetChannel(resCtx_, idx).remoteRank == remoteRank) {
            return &GetChannel(resCtx_, idx);
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

bool AllGatherNHRCore::MatchChannel(const ChannelResource &channel, bool crossServer, CommProtocol protocol) const
{
    return IsCrossServerChannel(channel) == crossServer && channel.protocol == protocol;
}

uint32_t AllGatherNHRCore::CountChannelsByScope(bool crossServer) const
{
    uint32_t count = 0;
    for (uint32_t idx = 0; idx < resCtx_.channelCount; ++idx) {
        if (IsCrossServerChannel(GetChannel(resCtx_, idx)) == crossServer) {
            ++count;
        }
    }
    return count;
}

uint32_t AllGatherNHRCore::CountChannelsByProtocol(bool crossServer, CommProtocol protocol) const
{
    uint32_t count = 0;
    for (uint32_t idx = 0; idx < resCtx_.channelCount; ++idx) {
        const ChannelResource &channel = GetChannel(resCtx_, idx);
        if (MatchChannel(channel, crossServer, protocol)) {
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

HcclResult AllGatherNHRCore::NotifyReadyToRank(uint32_t remoteRank, bool crossServer, CommProtocol protocol) const
{
    const char *scope = ToScopeString(crossServer);
    const char *protocolName = ToProtocolString(protocol);
    const ChannelResource *channel = FindChannel(remoteRank);
    if (channel == nullptr) {
        HCCL_ERROR("channel to remoteRank=%u is missing", remoteRank);
        return HCCL_E_NOT_FOUND;
    }
    if (!MatchChannel(*channel, crossServer, protocol)) {
        return HCCL_SUCCESS;
    }

    const int32_t ret = HcommChannelNotifyRecordOnThread(
        resCtx_.threadHandle,
        channel->handle,
        channel->remoteNotifyIdx);
    if (ret != HCCL_SUCCESS) {
        HCCL_ERROR("%s/%s notify record failed, remoteRank=%u, ret=%d", scope, protocolName, remoteRank, ret);
        return static_cast<HcclResult>(ret);
    }
    return HCCL_SUCCESS;
}

HcclResult AllGatherNHRCore::ReadFromRank(uint32_t remoteRank, bool crossServer, CommProtocol protocol) const
{
    const char *scope = ToScopeString(crossServer);
    const char *protocolName = ToProtocolString(protocol);
    const ChannelResource *channel = FindChannel(remoteRank);
    if (channel == nullptr) {
        HCCL_ERROR("channel to remoteRank=%u is missing", remoteRank);
        return HCCL_E_NOT_FOUND;
    }
    if (!MatchChannel(*channel, crossServer, protocol)) {
        return HCCL_SUCCESS;
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
    return HCCL_SUCCESS;
}

HcclResult AllGatherNHRCore::RunProtocolStep(const NHRStepInfo &stepInfo, bool crossServer, CommProtocol protocol) const
{
    const char *scope = ToScopeString(crossServer);
    const char *protocolName = ToProtocolString(protocol);

    // 这一层开始真正使用 step plan：每一步都显式记录 sendTo / recvFrom，
    // 后面如果要引入更像正式实现的 slice 组合或双向策略，就可以直接在 step 维度细化。
    HCCL_INFO("NHR protocol step: step=%u, scope=%s, protocol=%s, sendTo=%u, recvFrom=%u, txItems=%u, rxItems=%u",
        stepInfo.step,
        scope,
        protocolName,
        stepInfo.toRank,
        stepInfo.fromRank,
        static_cast<uint32_t>(stepInfo.txItemOrder.size()),
        static_cast<uint32_t>(stepInfo.rxItemOrder.size()));

    HCCL_CHK_RET(NotifyReadyToRank(stepInfo.toRank, crossServer, protocol));
    HCCL_CHK_RET(ReadFromRank(stepInfo.fromRank, crossServer, protocol));
    return HCCL_SUCCESS;
}

HcclResult AllGatherNHRCore::RunProtocol(bool crossServer, CommProtocol protocol, const std::vector<NHRStepInfo> &stepPlan) const
{
    const uint32_t channelCount = CountChannelsByProtocol(crossServer, protocol);
    if (channelCount == 0) {
        return HCCL_SUCCESS;
    }

    const char *scope = ToScopeString(crossServer);
    HCCL_INFO("NHR protocol dispatch: scope=%s, protocol=%s, channels=%u, steps=%u, packedBytes=%llu",
        scope,
        ToProtocolString(protocol),
        channelCount,
        static_cast<uint32_t>(stepPlan.size()),
        static_cast<unsigned long long>(packedBytes_));

    for (const NHRStepInfo &stepInfo : stepPlan) {
        HCCL_CHK_RET(RunProtocolStep(stepInfo, crossServer, protocol));
    }
    return HCCL_SUCCESS;
}

HcclResult AllGatherNHRCore::RunScope(bool crossServer, const std::vector<NHRStepInfo> &stepPlan) const
{
    static const CommProtocol kProtocolOrder[] = {
        COMM_PROTOCOL_HCCS,
        COMM_PROTOCOL_ROCE,
        COMM_PROTOCOL_PCIE,
        COMM_PROTOCOL_SIO,
    };

    const char *scope = ToScopeString(crossServer);
    HCCL_INFO("NHR scope dispatch begins: scope=%s, channels=%u, steps=%u",
        scope,
        CountChannelsByScope(crossServer),
        static_cast<uint32_t>(stepPlan.size()));
    for (CommProtocol protocol : kProtocolOrder) {
        HCCL_CHK_RET(RunProtocol(crossServer, protocol, stepPlan));
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
    HCCL_CHK_RET(ValidateStepPlan(stepPlan));

    const ResourceStats stats = CollectResourceStats(param_, resCtx_);
    HCCL_INFO("NHR core step plan ready: rank=%u, rankSize=%u, commMode=%s, intraServerRankCount=%u, crossServerRankCount=%u, steps=%u, packedBytes=%llu, paramWindowBytes=%llu, maxWindowBytes=%llu, channelCount=%u, intraServerChannels=%u, crossServerChannels=%u, hccs=%u, roce=%u, pcie=%u, sio=%u",
        param_.topoInfo.rank,
        param_.topoInfo.rankSize,
        ToCommModeString(param_.commMode),
        param_.intraServerRankCount,
        param_.crossServerRankCount,
        static_cast<unsigned int>(stepPlan.size()),
        static_cast<unsigned long long>(packedBytes_),
        static_cast<unsigned long long>(param_.windowBytes),
        static_cast<unsigned long long>(stats.maxWindowBytes),
        resCtx_.channelCount,
        stats.intraServerChannels,
        stats.crossServerChannels,
        stats.hccsChannels,
        stats.roceChannels,
        stats.pcieChannels,
        stats.sioChannels);

    // 这里先保留 NHR 的控制层边界，但执行顺序已经按 step -> scope -> protocol 三层组织。
    HCCL_CHK_RET(RunScope(false, stepPlan));
    if (param_.commMode == BatchCommMode::kCrossServer) {
        HCCL_CHK_RET(RunScope(true, stepPlan));
    }
    return HCCL_SUCCESS;
}

}  // namespace ops_hccl_allgatherbatch







