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

}  // namespace

AllGatherHDStageCore::AllGatherHDStageCore(const OpParam &param, AlgResourceCtx &resCtx)
    : param_(param), resCtx_(resCtx)
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
    AllGatherNHRCore nhrCore(param_, resCtx_);
    return nhrCore.RunAsync();
}

HcclResult AllGatherHDStageCore::RunPowerPath(const HDStagePlan &plan) const
{
    HCCL_INFO("HDStage power path planned: rank=%u, rankSize=%u, powerSteps=%u",
        param_.topoInfo.rank,
        param_.topoInfo.rankSize,
        plan.powerSteps);

    // 阶段 4 先把 power 路径的决策层建起来，实际的数据交换留到后续阶段接入。
    return HCCL_E_NOT_SUPPORT;
}

HcclResult AllGatherHDStageCore::RunAsync()
{
    if (param_.topoInfo.rankSize == 1) {
        HCCL_INFO("HDStage fast path: rankSize=1, no staged communication needed");
        return HCCL_SUCCESS;
    }

    HDStagePlan plan;
    HCCL_CHK_RET(BuildStagePlan(plan));
    HCCL_INFO("HDStage plan ready: rank=%u, rankSize=%u, noPower=%u, powerSteps=%u",
        param_.topoInfo.rank,
        param_.topoInfo.rankSize,
        plan.noPower,
        plan.powerSteps);

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
