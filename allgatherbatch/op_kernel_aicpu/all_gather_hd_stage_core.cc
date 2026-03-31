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

uint32_t CountCrossServerChannels(const OpParam &param, const AlgResourceCtx &resCtx)
{
    uint32_t count = 0;
    for (uint32_t idx = 0; idx < resCtx.channelCount; ++idx) {
        if (resCtx.channels[idx].remoteServerIdx != param.topoInfo.serverIdx) {
            ++count;
        }
    }
    return count;
}

}  // namespace

AllGatherHDStageCore::AllGatherHDStageCore(const OpParam &param, AlgResourceCtx &resCtx, uint64_t packedBytes)
    : param_(param), resCtx_(resCtx), packedBytes_(packedBytes)
{
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

HcclResult AllGatherHDStageCore::RunNoPowerPath(const HDStagePlan &plan) const
{
    (void)plan;
    AllGatherNHRCore nhrCore(param_, resCtx_, packedBytes_);
    return nhrCore.RunAsync();
}

HcclResult AllGatherHDStageCore::RunPowerPath(const HDStagePlan &plan) const
{
    HCCL_INFO("HDStage power path planned: rank=%u, rankSize=%u, powerSteps=%u, packedBytes=%llu",
        param_.topoInfo.rank,
        param_.topoInfo.rankSize,
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

    HDStagePlan plan;
    HCCL_CHK_RET(BuildStagePlan(plan));

    const uint32_t crossServerChannels = CountCrossServerChannels(param_, resCtx_);
    const uint32_t intraServerChannels = resCtx_.channelCount - crossServerChannels;
    HCCL_INFO("HDStage plan ready: rank=%u, rankSize=%u, serverIdx=%u, noPower=%u, powerSteps=%u, packedBytes=%llu, intraServerChannels=%u, crossServerChannels=%u",
        param_.topoInfo.rank,
        param_.topoInfo.rankSize,
        param_.topoInfo.serverIdx,
        plan.noPower,
        plan.powerSteps,
        static_cast<unsigned long long>(packedBytes_),
        intraServerChannels,
        crossServerChannels);

    if (plan.needNoPowerPath) {
        HcclResult ret = RunNoPowerPath(plan);
        if (ret != HCCL_SUCCESS) {
            return ret;
        }
    }

    if (plan.needPowerPath) {
        return RunPowerPath(plan);
    }

    return HCCL_SUCCESS;
}

}  // namespace ops_hccl_allgatherbatch
