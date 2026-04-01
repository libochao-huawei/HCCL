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

uint32_t CalcFinalSteps(uint32_t powerSteps)
{
    if (powerSteps >= 2U) {
        return 2U;
    }
    if (powerSteps >= 1U) {
        return 1U;
    }
    return 0U;
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

uint32_t CountPrimaryPaths(const HDStagePlan &plan)
{
    return static_cast<uint32_t>(plan.useNoPowerAsPrimary) +
        static_cast<uint32_t>(plan.usePowerAsPrimary) +
        static_cast<uint32_t>(plan.useFinalAsPrimary);
}

const char *SelectPrimaryPath(const HDStagePlan &plan)
{
    if (plan.useNoPowerAsPrimary) {
        return "noPower";
    }
    if (plan.usePowerAsPrimary) {
        return "power";
    }
    if (plan.useFinalAsPrimary) {
        return "final";
    }
    return "none";
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

    // 这里沿用设计稿和 hcomm HDStage 的拆解思路：
    // 1. noPower 负责非 2 次幂部分。
    // 2. power 负责 powerSteps 中去掉 finalSteps 后的主阶段。
    // 3. final 负责最后 0/1/2 步的收尾语义。
    plan.powerSteps = CalcTrailingPowerSteps(rankSize);
    plan.powerFactor = (plan.powerSteps == 0U) ? 1U : (1U << plan.powerSteps);
    plan.noPower = rankSize / plan.powerFactor;
    plan.finalSteps = CalcFinalSteps(plan.powerSteps);
    plan.remainingPowerSteps = (plan.powerSteps > plan.finalSteps) ? (plan.powerSteps - plan.finalSteps) : 0U;
    plan.needNoPowerPath = (plan.noPower > 1U);
    plan.needPowerPath = (plan.remainingPowerSteps >= 1U);
    plan.needFinalPath = (rankSize > 1U);

    // 当前 public-header 方案下，NHR 仍是唯一的数据面实现，因此三个阶段里只选一个主路径真正落通信。
    // 这样既保留 HDStage 的阶段语义，也避免最小数据面被重复执行多次。
    plan.useNoPowerAsPrimary = plan.needNoPowerPath;
    plan.usePowerAsPrimary = (!plan.useNoPowerAsPrimary && plan.needPowerPath);
    plan.useFinalAsPrimary = (!plan.useNoPowerAsPrimary && !plan.usePowerAsPrimary && plan.needFinalPath);
    return HCCL_SUCCESS;
}

HcclResult AllGatherHDStageCore::ValidateStagePlan(const HDStagePlan &plan) const
{
    const uint32_t crossServerChannels = CountChannelsByScope(param_, resCtx_, true);
    const uint32_t intraServerChannels = CountChannelsByScope(param_, resCtx_, false);

    // HDStage 这一层把 stage 计划和链路 scope 再对一次，避免“前后层都能跑，但 stage 假设不成立”的问题。
    if (plan.noPower == 0) {
        HCCL_ERROR("HDStage noPower is zero");
        return HCCL_E_INTERNAL;
    }
    if ((plan.noPower * plan.powerFactor) != param_.topoInfo.rankSize) {
        HCCL_ERROR("HDStage plan mismatch, noPower=%u, powerFactor=%u, rankSize=%u",
            plan.noPower,
            plan.powerFactor,
            param_.topoInfo.rankSize);
        return HCCL_E_INTERNAL;
    }
    if (plan.finalSteps > plan.powerSteps) {
        HCCL_ERROR("HDStage finalSteps exceeds powerSteps, finalSteps=%u, powerSteps=%u",
            plan.finalSteps,
            plan.powerSteps);
        return HCCL_E_INTERNAL;
    }
    if (plan.remainingPowerSteps + plan.finalSteps != plan.powerSteps) {
        HCCL_ERROR("HDStage power split mismatch, remainingPowerSteps=%u, finalSteps=%u, powerSteps=%u",
            plan.remainingPowerSteps,
            plan.finalSteps,
            plan.powerSteps);
        return HCCL_E_INTERNAL;
    }
    if (CountPrimaryPaths(plan) != 1U) {
        HCCL_ERROR("HDStage primary path selection is invalid, primaryCount=%u",
            CountPrimaryPaths(plan));
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

HcclResult AllGatherHDStageCore::RunNHR(const char *pathTag) const
{
    HCCL_INFO("HDStage delegates data movement to NHR: path=%s, rank=%u, packedBytes=%llu",
        pathTag,
        param_.topoInfo.rank,
        static_cast<unsigned long long>(packedBytes_));
    AllGatherNHRCore nhrCore(param_, resCtx_, packedBytes_);
    return nhrCore.RunAsync();
}

HcclResult AllGatherHDStageCore::RunNoPowerPath(const HDStagePlan &plan) const
{
    HCCL_INFO("HDStage noPower path: rank=%u, noPower=%u, powerSteps=%u, primary=%s, packedBytes=%llu",
        param_.topoInfo.rank,
        plan.noPower,
        plan.powerSteps,
        plan.useNoPowerAsPrimary ? "true" : "false",
        static_cast<unsigned long long>(packedBytes_));

    if (!plan.useNoPowerAsPrimary) {
        HCCL_INFO("HDStage noPower path is structural only in current implementation");
        return HCCL_SUCCESS;
    }
    return RunNHR("noPower");
}

HcclResult AllGatherHDStageCore::RunPowerPath(const HDStagePlan &plan) const
{
    HCCL_INFO("HDStage power path: rank=%u, rankSize=%u, commMode=%s, remainingPowerSteps=%u, finalSteps=%u, primary=%s, packedBytes=%llu",
        param_.topoInfo.rank,
        param_.topoInfo.rankSize,
        ToCommModeString(param_.commMode),
        plan.remainingPowerSteps,
        plan.finalSteps,
        plan.usePowerAsPrimary ? "true" : "false",
        static_cast<unsigned long long>(packedBytes_));

    if (!plan.usePowerAsPrimary) {
        HCCL_INFO("HDStage power path is structural only in current implementation");
        return HCCL_SUCCESS;
    }
    return RunNHR("power");
}

HcclResult AllGatherHDStageCore::RunFinalDirect(const HDStagePlan &plan) const
{
    HCCL_INFO("HDStage final direct path: rank=%u, primary=%s, packedBytes=%llu",
        param_.topoInfo.rank,
        plan.useFinalAsPrimary ? "true" : "false",
        static_cast<unsigned long long>(packedBytes_));

    if (!plan.useFinalAsPrimary) {
        return HCCL_SUCCESS;
    }
    return RunNHR("final-direct");
}

HcclResult AllGatherHDStageCore::RunFinalLastOne(const HDStagePlan &plan) const
{
    HCCL_INFO("HDStage final last-one path: rank=%u, powerSteps=%u, primary=%s, packedBytes=%llu",
        param_.topoInfo.rank,
        plan.powerSteps,
        plan.useFinalAsPrimary ? "true" : "false",
        static_cast<unsigned long long>(packedBytes_));

    if (!plan.useFinalAsPrimary) {
        return HCCL_SUCCESS;
    }
    return RunNHR("final-last-one");
}

HcclResult AllGatherHDStageCore::RunFinalLastTwo(const HDStagePlan &plan) const
{
    HCCL_INFO("HDStage final last-two path: rank=%u, powerSteps=%u, primary=%s, packedBytes=%llu",
        param_.topoInfo.rank,
        plan.powerSteps,
        plan.useFinalAsPrimary ? "true" : "false",
        static_cast<unsigned long long>(packedBytes_));

    if (!plan.useFinalAsPrimary) {
        return HCCL_SUCCESS;
    }
    return RunNHR("final-last-two");
}

HcclResult AllGatherHDStageCore::RunFinalPath(const HDStagePlan &plan) const
{
    // final path 负责表达 HDStage 最后 0/1/2 步的收尾语义。
    // 在当前 custom-op 版本里，如果主通信已经在 noPower/power 阶段完成，这里只保留结构和日志；
    // 如果前面没有主通信路径，则由 final path 接管真正的数据面执行。
    if (plan.finalSteps >= 2U) {
        return RunFinalLastTwo(plan);
    }
    if (plan.finalSteps == 1U) {
        return RunFinalLastOne(plan);
    }
    return RunFinalDirect(plan);
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
    HCCL_INFO("HDStage plan ready: rank=%u, rankSize=%u, commMode=%s, serverIdx=%u, intraServerRankCount=%u, crossServerRankCount=%u, noPower=%u, powerFactor=%u, powerSteps=%u, remainingPowerSteps=%u, finalSteps=%u, primaryPath=%s, packedBytes=%llu, intraServerChannels=%u, crossServerChannels=%u, hccs=%u, roce=%u, pcie=%u, sio=%u",
        param_.topoInfo.rank,
        param_.topoInfo.rankSize,
        ToCommModeString(param_.commMode),
        param_.topoInfo.serverIdx,
        param_.intraServerRankCount,
        param_.crossServerRankCount,
        plan.noPower,
        plan.powerFactor,
        plan.powerSteps,
        plan.remainingPowerSteps,
        plan.finalSteps,
        SelectPrimaryPath(plan),
        static_cast<unsigned long long>(packedBytes_),
        intraServerChannels,
        crossServerChannels,
        CountChannelsByProtocol(resCtx_, COMM_PROTOCOL_HCCS),
        CountChannelsByProtocol(resCtx_, COMM_PROTOCOL_ROCE),
        CountChannelsByProtocol(resCtx_, COMM_PROTOCOL_PCIE),
        CountChannelsByProtocol(resCtx_, COMM_PROTOCOL_SIO));

    if (plan.needNoPowerPath) {
        HCCL_CHK_RET(RunNoPowerPath(plan));
    }
    if (plan.needPowerPath) {
        HCCL_CHK_RET(RunPowerPath(plan));
    }
    if (plan.needFinalPath) {
        HCCL_CHK_RET(RunFinalPath(plan));
    }

    HCCL_INFO("HDStage finished: rank=%u, primaryPath=%s", param_.topoInfo.rank, SelectPrimaryPath(plan));
    return HCCL_SUCCESS;
}

}  // namespace ops_hccl_allgatherbatch
