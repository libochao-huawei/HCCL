#include "all_gather_nhr_core.h"

#include "log.h"

namespace ops_hccl_allgatherbatch {

AllGatherNHRCore::AllGatherNHRCore(const OpParam &param, AlgResourceCtx &resCtx)
    : param_(param), resCtx_(resCtx)
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
    if (param_.topoInfo.rankSize > 1 && resCtx_.channelCount == 0) {
        HCCL_WARNING("NHR core needs channels for rankSize=%u, but channelCount is 0", param_.topoInfo.rankSize);
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

    // 当前阶段先按 item 级顺序建最小计划，后续接 slice merge 时再细化到子切片。
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

HcclResult AllGatherNHRCore::RunAsync()
{
    if (param_.topoInfo.rankSize == 1) {
        HCCL_INFO("NHR core fast path: rankSize=1, no inter-rank communication needed");
        return HCCL_SUCCESS;
    }

    HcclResult validateRet = ValidateCommState();
    if (validateRet != HCCL_SUCCESS) {
        return validateRet;
    }

    std::vector<NHRStepInfo> stepPlan;
    HCCL_CHK_RET(BuildStepPlan(stepPlan));

    HCCL_INFO("NHR core step plan ready: rank=%u, rankSize=%u, steps=%u, channelCount=%u",
        param_.topoInfo.rank,
        param_.topoInfo.rankSize,
        static_cast<unsigned int>(stepPlan.size()),
        resCtx_.channelCount);

    // 阶段 4 先把 NHR 的步进计划和调用关系立起来。
    // 真正的 channel notify/read/write 数据面会在后续阶段与 Pack/Unpack 一起接入。
    return HCCL_E_NOT_SUPPORT;
}

}  // namespace ops_hccl_allgatherbatch
