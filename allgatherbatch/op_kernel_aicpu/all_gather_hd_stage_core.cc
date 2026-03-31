#include "all_gather_hd_stage_core.h"

#include "all_gather_nhr_core.h"
#include "log.h"

namespace ops_hccl_allgatherbatch {

namespace {

uint32_t CalcTrailingPowerSteps(uint32_t rankSize)
{
    uint32_t steps = 0;
    while ((rankSize & 1U) == 0U && rankSize > 1U) {
        rankSize >>= 1U;
        ++steps;
    }
    return steps;
}

uint32_t CountChannelsByScope(const OpParam &param, const AlgResourceCtx &resCtx, bool crossServer)
{
    uint32_t count = 0;
    for (uint32_t idx = 0; idx < resCtx.channelCount; ++idx) {
        const bool isCrossServer = (resCtx.channels[idx].remoteServerIdx != param.topoInfo.serverIdx);
        if (isCrossServer == crossServer) {
            ++count;
        }
    }
    return count;
}

uint32_t CountChannelsByProtocol(const AlgResourceCtx &resCtx, CommProtocol protocol)
{
    uint32_t count = 0;
    for (uint32_t idx = 0; idx < resCtx.channelCount; ++idx) {
        if (resCtx.channels[idx].protocol == protocol) {
            ++count;
        }
    }
    return count;
}

uint32_t CountRecognizedChannels(const AlgResourceCtx &resCtx)
{
    return CountChannelsByProtocol(resCtx, COMM_PROTOCOL_HCCS) +
        CountChannelsByProtocol(resCtx, COMM_PROTOCOL_ROCE) +
        CountChannelsByProtocol(resCtx, COMM_PROTOCOL_PCIE) +
        CountChannelsByProtocol(resCtx, COMM_PROTOCOL_SIO);
}

}  // namespace

AllGatherHDStageCore::AllGatherHDStageCore(const OpParam &param, AlgResourceCtx &resCtx, uint64_t packedBytes)
    : param_(param), resCtx_(resCtx), packedBytes_(packedBytes)
{
}

HcclResult AllGatherHDStageCore::ValidateStageInput() const
{
    if (!IsValidCommMode(param_.commMode)) {
        HCCL_ERROR("HDStage commMode is invalid");
        return HCCL_E_INTERNAL;
    }
    if (param_.topoInfo.rankSize == 0) {
        HCCL_ERROR("HDStage rankSize is zero");
        return HCCL_E_PARA;
    }
    if (packedBytes_ == 0) {
        HCCL_ERROR("HDStage packedBytes is zero");
        return HCCL_E_PARA;
    }

    const uint64_t totalBytes = packedBytes_ * param_.topoInfo.rankSize;
    if (resCtx_.localBuffer.addr == nullptr || resCtx_.localBuffer.size < totalBytes) {
        HCCL_ERROR("HDStage localBuffer is too small, need=%llu, actual=%llu",
            static_cast<unsigned long long>(totalBytes),
            static_cast<unsigned long long>(resCtx_.localBuffer.size));
        return HCCL_E_INTERNAL;
    }
    return HCCL_SUCCESS;
}

HcclResult AllGatherHDStageCore::BuildStagePlan(HDStagePlan &plan) const
{
    const uint32_t rankSize = param_.topoInfo.rankSize;
    if (rankSize == 0) {
        HCCL_ERROR("rankSize is zero");
        return HCCL_E_PARA;
    }

    plan.powerSteps = CalcTrailingPowerSteps(rankSize);
    const uint32_t powerFactor = (plan.powerSteps == 0) ? 1U : (1U << plan.powerSteps);
    plan.noPower = rankSize / powerFactor;
    plan.finalSteps = 0;
    plan.needNoPowerPath = (plan.noPower > 1U);
    plan.needPowerPath = (plan.powerSteps > plan.finalSteps);
    return HCCL_SUCCESS;
}

HcclResult AllGatherHDStageCore::ValidateStagePlan(const HDStagePlan &plan) const
{
    const uint32_t crossServerChannels = CountChannelsByScope(param_, resCtx_, true);
    const uint32_t intraServerChannels = CountChannelsByScope(param_, resCtx_, false);
    const uint32_t powerFactor = (plan.powerSteps == 0) ? 1U : (1U << plan.powerSteps);

    // HDStage 这一层把 stage 计划和链路 scope 再对一次，避免“前后层都能跑，但 stage 假设不成立”的问题。
    if (plan.noPower == 0) {
        HCCL_ERROR("HDStage noPower is zero");
        return HCCL_E_INTERNAL;
    }
    if ((plan.noPower * powerFactor) != param_.topoInfo.rankSize) {
        HCCL_ERROR("HDStage plan mismatch, noPower=%u, powerFactor=%u, rankSize=%u",
            plan.noPower,
            powerFactor,
            param_.topoInfo.rankSize);
        return HCCL_E_INTERNAL;
    }
    if (intraServerChannels + crossServerChannels != resCtx_.channelCount) {
        HCCL_ERROR("HDStage channel scope split is inconsistent, intra=%u, cross=%u, channelCount=%u",
            intraServerChannels,
            crossServerChannels,
            resCtx_.channelCount);
        return HCCL_E_INTERNAL;
    }
    if (param_.commMode == BatchCommMode::kSingleServer && crossServerChannels != 0) {
        HCCL_ERROR("HDStage single-server mode unexpectedly has cross-server channels=%u", crossServerChannels);
        return HCCL_E_INTERNAL;
    }
    if (param_.commMode == BatchCommMode::kCrossServer && crossServerChannels == 0) {
        HCCL_ERROR("HDStage cross-server mode has no cross-server channels");
        return HCCL_E_INTERNAL;
    }
    return HCCL_SUCCESS;
}

HcclResult AllGatherHDStageCore::ValidateProtocolDistribution() const
{
    const uint32_t recognizedChannels = CountRecognizedChannels(resCtx_);

    // Stage 层也要求资源里的协议分布是自洽的，避免把“未知协议”静默带入 NHR 子模板。
    if (recognizedChannels != resCtx_.channelCount) {
        HCCL_ERROR("HDStage protocol distribution mismatch, recognized=%u, channelCount=%u",
            recognizedChannels,
            resCtx_.channelCount);
        return HCCL_E_INTERNAL;
    }
    return HCCL_SUCCESS;
}

HcclResult AllGatherHDStageCore::RunNoPowerPath(const HDStagePlan &plan) const
{
    HCCL_INFO("HDStage noPower path: rank=%u, noPower=%u, powerSteps=%u, packedBytes=%llu",
        param_.topoInfo.rank,
        plan.noPower,
        plan.powerSteps,
        static_cast<unsigned long long>(packedBytes_));

    AllGatherNHRCore nhrCore(param_, resCtx_, packedBytes_);
    return nhrCore.RunAsync();
}

HcclResult AllGatherHDStageCore::RunPowerPath(const HDStagePlan &plan) const
{
    HCCL_INFO("HDStage power path planned: rank=%u, rankSize=%u, commMode=%s, powerSteps=%u, packedBytes=%llu",
        param_.topoInfo.rank,
        param_.topoInfo.rankSize,
        ToCommModeString(param_.commMode),
        plan.powerSteps,
        static_cast<unsigned long long>(packedBytes_));

    // 当前 custom-op 方案先让 power 路径复用同一套最小 NHR 数据面，保证所有 rank 都能回收到完整窗口。
    AllGatherNHRCore nhrCore(param_, resCtx_, packedBytes_);
    return nhrCore.RunAsync();
}

HcclResult AllGatherHDStageCore::RunAsync()
{
    if (param_.topoInfo.rankSize == 1) {
        HCCL_INFO("HDStage fast path: rankSize=1, no staged communication needed");
        return HCCL_SUCCESS;
    }

    HCCL_CHK_RET(ValidateStageInput());

    HDStagePlan plan;
    HCCL_CHK_RET(BuildStagePlan(plan));
    HCCL_CHK_RET(ValidateStagePlan(plan));
    HCCL_CHK_RET(ValidateProtocolDistribution());

    const uint32_t crossServerChannels = CountChannelsByScope(param_, resCtx_, true);
    const uint32_t intraServerChannels = CountChannelsByScope(param_, resCtx_, false);
    HCCL_INFO("HDStage plan ready: rank=%u, rankSize=%u, commMode=%s, serverIdx=%u, intraServerRankCount=%u, crossServerRankCount=%u, noPower=%u, powerSteps=%u, packedBytes=%llu, intraServerChannels=%u, crossServerChannels=%u, hccs=%u, roce=%u, pcie=%u, sio=%u",
        param_.topoInfo.rank,
        param_.topoInfo.rankSize,
        ToCommModeString(param_.commMode),
        param_.topoInfo.serverIdx,
        param_.intraServerRankCount,
        param_.crossServerRankCount,
        plan.noPower,
        plan.powerSteps,
        static_cast<unsigned long long>(packedBytes_),
        intraServerChannels,
        crossServerChannels,
        CountChannelsByProtocol(resCtx_, COMM_PROTOCOL_HCCS),
        CountChannelsByProtocol(resCtx_, COMM_PROTOCOL_ROCE),
        CountChannelsByProtocol(resCtx_, COMM_PROTOCOL_PCIE),
        CountChannelsByProtocol(resCtx_, COMM_PROTOCOL_SIO));

    if (plan.needNoPowerPath) {
        HCCL_INFO("HDStage dispatch noPower path first");
        HcclResult ret = RunNoPowerPath(plan);
        if (ret != HCCL_SUCCESS) {
            return ret;
        }
    }

    if (plan.needPowerPath) {
        HCCL_INFO("HDStage dispatch power path after noPower=%s", plan.needNoPowerPath ? "true" : "false");
        return RunPowerPath(plan);
    }

    HCCL_INFO("HDStage completes without extra power path");
    return HCCL_SUCCESS;
}

}  // namespace ops_hccl_allgatherbatch
