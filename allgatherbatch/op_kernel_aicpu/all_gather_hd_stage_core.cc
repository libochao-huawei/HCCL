#include "all_gather_hd_stage_core.h"

#include <vector>

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

}  // namespace

AllGatherHDStageCore::AllGatherHDStageCore(const OpParam &param, AlgResourceCtx &resCtx, uint64_t packedBytes)
    : param_(param), resCtx_(resCtx), packedBytes_(packedBytes)
{
}

HcclResult AllGatherHDStageCore::ValidateStageInput() const
{

    if (param_.topoInfo.rankSize == 0) {
        HCCL_ERROR("HDStage rankSize is zero");
        return HCCL_E_PARA;
    }
    if (packedBytes_ == 0) {
        HCCL_ERROR("HDStage packedBytes is zero");
        return HCCL_E_PARA;
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
    plan.powerFactor = (plan.powerSteps == 0U) ? 1U : (1U << plan.powerSteps);
    plan.noPower = rankSize / plan.powerFactor;
    plan.finalSteps = CalcFinalSteps(plan.powerSteps);
    plan.remainingPowerSteps = (plan.powerSteps > plan.finalSteps) ? (plan.powerSteps - plan.finalSteps) : 0U;
    plan.needNoPowerPath = (plan.noPower > 1U);
    plan.needPowerPath = (plan.remainingPowerSteps >= 1U);
    plan.needFinalPath = (rankSize > 1U);
    return HCCL_SUCCESS;
}

HcclResult AllGatherHDStageCore::ValidateStagePlan(const HDStagePlan &plan) const
{
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

    return HCCL_SUCCESS;
}

HcclResult AllGatherHDStageCore::BuildNHRRunCtx(const HDStagePlan &plan, NHRRunCtx &runCtx) const
{
    if (!plan.needNoPowerPath) {
        HCCL_ERROR("HDStage noPower subgroup is requested while noPower path is disabled");
        return HCCL_E_INTERNAL;
    }
    if (plan.noPower <= 1U || plan.powerFactor == 0U) {
        HCCL_ERROR("HDStage noPower subgroup is invalid, noPower=%u, powerFactor=%u",
            plan.noPower,
            plan.powerFactor);
        return HCCL_E_INTERNAL;
    }

    const uint32_t rank = param_.topoInfo.rank;
    const uint32_t group = rank / plan.powerFactor;
    const uint32_t groupIdx = rank % plan.powerFactor;
    runCtx.rank = group;
    runCtx.rankSize = plan.noPower;
    runCtx.packedBytes = packedBytes_;
    runCtx.baseOffset = 0;
    runCtx.inputBase = static_cast<uint8_t *>(resCtx_.localBuffer.addr);
    runCtx.outputBase = static_cast<uint8_t *>(resCtx_.localBuffer.addr);
    runCtx.keepOrder = true;
    runCtx.subgroupRanks.clear();
    runCtx.slices.clear();
    runCtx.subgroupRanks.reserve(plan.noPower);
    runCtx.slices.reserve(plan.noPower);
    for (uint32_t idx = 0; idx < plan.noPower; ++idx) {
        const uint32_t globalRank = idx * plan.powerFactor + groupIdx;
        runCtx.subgroupRanks.push_back(globalRank);
        runCtx.slices.push_back(LocalSlice { packedBytes_ * globalRank, packedBytes_ });
    }
    if (runCtx.rank >= runCtx.rankSize) {
        HCCL_ERROR("HDStage subgroupRank=%u is out of range, subgroupSize=%u",
            runCtx.rank,
            runCtx.rankSize);
        return HCCL_E_INTERNAL;
    }
    HCCL_INFO("HDStage built NHR run ctx: rank=%u, group=%u, groupIdx=%u, subgroupRank=%u, subgroupSize=%u, sliceCount=%u",
        rank,
        group,
        groupIdx,
        runCtx.rank,
        runCtx.rankSize,
        static_cast<uint32_t>(runCtx.slices.size()));
    return HCCL_SUCCESS;
}

HcclResult AllGatherHDStageCore::RunPreCopy() const
{
    uint8_t *localRankBase = static_cast<uint8_t *>(resCtx_.localBuffer.addr) +
        (packedBytes_ * param_.topoInfo.rank);
    if (localRankBase == nullptr) {
        HCCL_ERROR("HDStage pre-copy base is null");
        return HCCL_E_INTERNAL;
    }
    HCCL_INFO("HDStage pre-copy ready: rank=%u, packedBytes=%llu, localRankBase=%p",
        param_.topoInfo.rank,
        static_cast<unsigned long long>(packedBytes_),
        localRankBase);
    return HCCL_SUCCESS;
}

HcclResult AllGatherHDStageCore::RunNHR(const char *pathTag, const NHRRunCtx &runCtx) const
{
    HCCL_INFO("HDStage delegates subgroup to NHR: path=%s, rank=%u, packedBytes=%llu, subgroupRank=%u, subgroupSize=%u, sliceCount=%u",
        pathTag,
        param_.topoInfo.rank,
        static_cast<unsigned long long>(packedBytes_),
        runCtx.rank,
        runCtx.rankSize,
        static_cast<uint32_t>(runCtx.slices.size()));
    AllGatherNHRCore nhrCore(param_, resCtx_, packedBytes_, runCtx);
    return nhrCore.RunAsync();
}

const ChannelResource *AllGatherHDStageCore::FindChannel(uint32_t remoteRank) const
{
    for (uint32_t idx = 0; idx < resCtx_.channelCount; ++idx) {
        const ChannelResource &channel = GetChannel(resCtx_, idx);
        if (channel.remoteRank == remoteRank) {
            return &channel;
        }
    }
    return nullptr;
}

HcclResult AllGatherHDStageCore::BuildPartnerRanksForBit(
    const HDStagePlan &plan, uint32_t bit, std::vector<uint32_t> &partnerRanks) const
{
    if (plan.powerFactor == 0U || bit >= plan.powerSteps) {
        HCCL_ERROR("HDStage power bit is invalid, bit=%u, powerSteps=%u, powerFactor=%u",
            bit,
            plan.powerSteps,
            plan.powerFactor);
        return HCCL_E_INTERNAL;
    }
    const uint32_t groupIdx = param_.topoInfo.rank % plan.powerFactor;
    const uint32_t lowMask = (bit == 0U) ? 0U : ((1U << bit) - 1U);
    const uint32_t flippedBit = (((groupIdx >> bit) & 1U) ^ 1U);
    partnerRanks.clear();
    for (uint32_t column = 0; column < plan.powerFactor; ++column) {
        if (bit != 0U && ((column & lowMask) != (groupIdx & lowMask))) {
            continue;
        }
        if (((column >> bit) & 1U) != flippedBit) {
            continue;
        }
        for (uint32_t group = 0; group < plan.noPower; ++group) {
            partnerRanks.push_back(group * plan.powerFactor + column);
        }
    }
    if (partnerRanks.empty()) {
        HCCL_ERROR("HDStage partner rank set is empty, bit=%u, groupIdx=%u", bit, groupIdx);
        return HCCL_E_INTERNAL;
    }
    return HCCL_SUCCESS;
}

HcclResult AllGatherHDStageCore::ReadPartnerRanks(
    uint32_t partnerRank, const std::vector<uint32_t> &partnerRanks, const char *stageTag) const
{
    const ChannelResource *channel = FindChannel(partnerRank);
    if (channel == nullptr) {
        HCCL_ERROR("HDStage %s channel to partnerRank=%u is missing", stageTag, partnerRank);
        return HCCL_E_NOT_FOUND;
    }
    }
    int32_t ret = HcommChannelNotifyRecordOnThread(
        resCtx_.threadHandle,
        channel->handle,
        channel->remoteNotifyIdx);
    if (ret != HCCL_SUCCESS) {
        HCCL_ERROR("HDStage %s notify record failed, partnerRank=%u, ret=%d", stageTag, partnerRank, ret);
        return static_cast<HcclResult>(ret);
    }
    ret = HcommChannelNotifyWaitOnThread(
        resCtx_.threadHandle,
        channel->handle,
        channel->localNotifyIdx,
        kAllGatherBatchCustomTimeoutMs);
    if (ret != HCCL_SUCCESS) {
        HCCL_ERROR("HDStage %s notify wait failed, partnerRank=%u, ret=%d", stageTag, partnerRank, ret);
        return static_cast<HcclResult>(ret);
    }
    for (uint32_t remoteRank : partnerRanks) {
        void *dst = static_cast<uint8_t *>(resCtx_.localBuffer.addr) + (packedBytes_ * remoteRank);
        const void *src = static_cast<const uint8_t *>(channel->remoteBuffer.addr) + (packedBytes_ * remoteRank);
        ret = HcommReadOnThread(resCtx_.threadHandle, channel->handle, dst, src, packedBytes_);
        if (ret != HCCL_SUCCESS) {
            HCCL_ERROR("HDStage %s remote read failed, partnerRank=%u, remoteRank=%u, ret=%d",
                stageTag,
                partnerRank,
                remoteRank,
                ret);
            return static_cast<HcclResult>(ret);
        }
    }
    return HCCL_SUCCESS;
}

HcclResult AllGatherHDStageCore::RunPowerBit(const HDStagePlan &plan, uint32_t bit, const char *stageTag) const
{
    const uint32_t rank = param_.topoInfo.rank;
    const uint32_t group = rank / plan.powerFactor;
    const uint32_t groupIdx = rank % plan.powerFactor;
    const uint32_t partnerGroupIdx = groupIdx ^ (1U << bit);
    const uint32_t partnerRank = group * plan.powerFactor + partnerGroupIdx;
    std::vector<uint32_t> partnerRanks;
    HCCL_CHK_RET(BuildPartnerRanksForBit(plan, bit, partnerRanks));
    HCCL_INFO("HDStage %s bit exchange: rank=%u, bit=%u, partnerRank=%u, partnerRanks=%u, packedBytes=%llu",
        stageTag,
        rank,
        bit,
        partnerRank,
        static_cast<uint32_t>(partnerRanks.size()),
        static_cast<unsigned long long>(packedBytes_));
    return ReadPartnerRanks(partnerRank, partnerRanks, stageTag);
}

HcclResult AllGatherHDStageCore::RunAllGatherNoPower(const HDStagePlan &plan) const
{
    HCCL_INFO("HDStage noPower stage: rank=%u, noPower=%u, powerSteps=%u, packedBytes=%llu",
        param_.topoInfo.rank,
        plan.noPower,
        plan.powerSteps,
        static_cast<unsigned long long>(packedBytes_));
    if (!plan.needNoPowerPath) {
        HCCL_INFO("HDStage noPower stage is skipped because noPower <= 1");
        return HCCL_SUCCESS;
    }
    NHRRunCtx runCtx;
    HCCL_CHK_RET(BuildNHRRunCtx(plan, runCtx));
    return RunNHR("noPower", runCtx);
}

HcclResult AllGatherHDStageCore::RunAllGatherPower(const HDStagePlan &plan) const
{
    HCCL_INFO("HDStage power stage: rank=%u, rankSize=%u, commMode=%s, remainingPowerSteps=%u, finalSteps=%u, packedBytes=%llu",
        param_.topoInfo.rank,
        param_.topoInfo.rankSize,
        ToCommModeString(param_.commMode),
        plan.remainingPowerSteps,
        plan.finalSteps,
        static_cast<unsigned long long>(packedBytes_));
    if (!plan.needPowerPath) {
        HCCL_INFO("HDStage power stage is skipped because remainingPowerSteps is zero");
        return HCCL_SUCCESS;
    }
    for (uint32_t step = 0; step < plan.remainingPowerSteps; ++step) {
        const uint32_t bit = plan.powerSteps - 1U - step;
        HCCL_CHK_RET(RunPowerBit(plan, bit, "power"));
    }
    return HCCL_SUCCESS;
}

bool AllGatherHDStageCore::CanSkipLastStage(const HDStagePlan &plan) const
{
    return plan.needFinalPath &&
        plan.finalSteps == 0U &&
        !plan.needPowerPath &&
        plan.needNoPowerPath &&
        plan.powerSteps == 0U &&
        plan.powerFactor == 1U &&
        plan.noPower == param_.topoInfo.rankSize;
}

HcclResult AllGatherHDStageCore::RunAllGatherLast(const HDStagePlan &plan) const
{
    HCCL_INFO("HDStage last stage: rank=%u, needFinal=%s, packedBytes=%llu",
        param_.topoInfo.rank,
        plan.needFinalPath ? "true" : "false",
        static_cast<unsigned long long>(packedBytes_));
    if (CanSkipLastStage(plan)) {
        HCCL_INFO("HDStage last stage is a legal no-op because noPower already covers all rank slots");
        return HCCL_SUCCESS;
    }
    HCCL_ERROR("HDStage last stage is reached with unfinished rank-slot convergence, rank=%u, rankSize=%u, noPower=%u, powerFactor=%u, powerSteps=%u, remainingPowerSteps=%u, finalSteps=%u",
        param_.topoInfo.rank,
        param_.topoInfo.rankSize,
        plan.noPower,
        plan.powerFactor,
        plan.powerSteps,
        plan.remainingPowerSteps,
        plan.finalSteps);
    return HCCL_E_INTERNAL;
}

HcclResult AllGatherHDStageCore::RunAllGatherLastOne(const HDStagePlan &plan) const
{
    HCCL_INFO("HDStage last-one stage: rank=%u, powerSteps=%u, needFinal=%s, packedBytes=%llu",
        param_.topoInfo.rank,
        plan.powerSteps,
        plan.needFinalPath ? "true" : "false",
        static_cast<unsigned long long>(packedBytes_));
    if (!plan.needFinalPath) {
        return HCCL_SUCCESS;
    }
    HCCL_CHK_RET(RunPowerBit(plan, 0U, "lastOne"));
    return HCCL_SUCCESS;
}

HcclResult AllGatherHDStageCore::RunAllGatherLastTwo(const HDStagePlan &plan) const
{
    HCCL_INFO("HDStage last-two stage: rank=%u, powerSteps=%u, needFinal=%s, packedBytes=%llu",
        param_.topoInfo.rank,
        plan.powerSteps,
        plan.needFinalPath ? "true" : "false",
        static_cast<unsigned long long>(packedBytes_));
    if (!plan.needFinalPath) {
        return HCCL_SUCCESS;
    }
    HCCL_CHK_RET(RunPowerBit(plan, 1U, "lastTwo-1"));
    HCCL_CHK_RET(RunPowerBit(plan, 0U, "lastTwo-0"));
    return HCCL_SUCCESS;
}

HcclResult AllGatherHDStageCore::RunAllGatherFinal(const HDStagePlan &plan) const
{
    if (!plan.needFinalPath) {
        HCCL_INFO("HDStage final stage is skipped because rankSize <= 1");
        return HCCL_SUCCESS;
    }
    if (plan.finalSteps >= 2U) {
        return RunAllGatherLastTwo(plan);
    }
    if (plan.finalSteps == 1U) {
        return RunAllGatherLastOne(plan);
    }
    return RunAllGatherLast(plan);
}

HcclResult AllGatherHDStageCore::RunAllGatherStage(const HDStagePlan &plan) const
{
    HCCL_CHK_RET(RunPreCopy());
    if (plan.needNoPowerPath) {
        HCCL_CHK_RET(RunAllGatherNoPower(plan));
    }
    if (plan.needPowerPath) {
        HCCL_CHK_RET(RunAllGatherPower(plan));
    }
    if (plan.needFinalPath) {
        HCCL_CHK_RET(RunAllGatherFinal(plan));
    }
    return HCCL_SUCCESS;
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
    HCCL_CHK_RET(RunAllGatherStage(plan));
    HCCL_INFO("HDStage finished: rank=%u, noPower=%s, power=%s, final=%s",
        param_.topoInfo.rank,
        plan.needNoPowerPath ? "true" : "false",
        plan.needPowerPath ? "true" : "false",
        plan.needFinalPath ? "true" : "false");
    return HCCL_SUCCESS;
}

}  // namespace ops_hccl_allgatherbatch
