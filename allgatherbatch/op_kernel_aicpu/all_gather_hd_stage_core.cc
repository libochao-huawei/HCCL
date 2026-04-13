#include "all_gather_hd_stage_core.h"

#include <algorithm>
#include <vector>

#include "all_gather_nhr_core.h"
#include "log.h"

namespace ops_hccl_allgatherbatch {

namespace {

uint32_t CalcPowerFactor(uint32_t rankSize)
{
    return rankSize & (~rankSize + 1U);
}

uint32_t CalcPowerSteps(uint32_t powerFactor)
{
    uint32_t steps = 0;
    while (powerFactor > 1U) {
        powerFactor >>= 1U;
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

uint32_t CalcMergedSliceCount(const PowerStepTask &stepTask)
{
    return (stepTask.size == 0) ? 0U : 1U;
}

bool CanMergeStageCopySlice(const StageCopySlice &lhs, const StageCopySlice &rhs)
{
    return lhs.remoteRank == rhs.remoteRank &&
        (lhs.localOffset + lhs.size) == rhs.localOffset &&
        (lhs.remoteOffset + lhs.size) == rhs.remoteOffset;
}

void MergeStageCopySlices(std::vector<StageCopySlice> &slices)
{
    if (slices.size() <= 1U) {
        return;
    }
    std::sort(slices.begin(), slices.end(), [](const StageCopySlice &lhs, const StageCopySlice &rhs) {
        if (lhs.remoteRank != rhs.remoteRank) {
            return lhs.remoteRank < rhs.remoteRank;
        }
        if (lhs.localOffset != rhs.localOffset) {
            return lhs.localOffset < rhs.localOffset;
        }
        return lhs.remoteOffset < rhs.remoteOffset;
    });
    std::vector<StageCopySlice> merged;
    merged.reserve(slices.size());
    merged.push_back(slices.front());
    for (size_t idx = 1; idx < slices.size(); ++idx) {
        StageCopySlice &tail = merged.back();
        const StageCopySlice &cur = slices[idx];
        if (CanMergeStageCopySlice(tail, cur)) {
            tail.size += cur.size;
            continue;
        }
        merged.push_back(cur);
    }
    slices.swap(merged);
}

}  // namespace

AllGatherHDStageCore::AllGatherHDStageCore(const OpParam &param, AlgResourceCtx &resCtx, const WindowStageLayout &layout)
    : param_(param), resCtx_(resCtx), packedBytes_(layout.packedBytes), layout_(layout)
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
    if (!IsValidWindowStageLayout(layout_)) {
        HCCL_ERROR("HDStage stage layout is invalid, rankSize=%u, powerSteps=%u, powerFactor=%u, noPower=%u",
            layout_.rankSize,
            layout_.powerSteps,
            layout_.powerFactor,
            layout_.noPower);
        return HCCL_E_INTERNAL;
    }
    if (layout_.packedBytes != packedBytes_) {
        HCCL_ERROR("HDStage packedBytes mismatch, layout=%llu, local=%llu",
            static_cast<unsigned long long>(layout_.packedBytes),
            static_cast<unsigned long long>(packedBytes_));
        return HCCL_E_INTERNAL;

    }
    if (layout_.totalBytes > resCtx_.localBuffer.size) {
        HCCL_ERROR("HDStage layout exceeds localBuffer, totalBytes=%llu, localBuffer=%llu",
            static_cast<unsigned long long>(layout_.totalBytes),
            static_cast<unsigned long long>(resCtx_.localBuffer.size));
        return HCCL_E_INTERNAL;
    }
    return HCCL_SUCCESS;
}

HcclResult AllGatherHDStageCore::BuildStagePlan(HDStagePlan &plan) const
{
    const uint32_t rankSize = param_.topoInfo.rankSize;
    plan = {};
    if (rankSize == 0U) {
        HCCL_ERROR("HDStage rankSize is zero while building stage plan");
        return HCCL_E_PARA;
    }

    const bool needFinalCopy = true;
    plan.powerFactor = rankSize & (~rankSize + 1U);
    plan.powerSteps = CalcPowerSteps(plan.powerFactor);
    plan.noPower = rankSize / plan.powerFactor;
    plan.finalSteps = needFinalCopy ? CalcFinalSteps(plan.powerSteps) : 0U;
    plan.remainingPowerSteps = plan.powerSteps - plan.finalSteps;
    plan.needNoPowerPath = (plan.noPower > 1U);
    plan.needPowerPath = (plan.remainingPowerSteps >= 1U);
    plan.needFinalPath = (rankSize > 1U);
    return HCCL_SUCCESS;
}

HcclResult AllGatherHDStageCore::ValidateStagePlan(const HDStagePlan &plan) const
{
    if (plan.noPower == 0U) {
        HCCL_ERROR("HDStage noPower is zero");
        return HCCL_E_INTERNAL;
    }
    if (plan.powerFactor != CalcPowerFactor(param_.topoInfo.rankSize)) {
        HCCL_ERROR("HDStage powerFactor mismatch, actual=%u, expected=%u, rankSize=%u",
            plan.powerFactor,
            CalcPowerFactor(param_.topoInfo.rankSize),
            param_.topoInfo.rankSize);
        return HCCL_E_INTERNAL;
    }
    if ((plan.noPower * plan.powerFactor) != param_.topoInfo.rankSize) {
        HCCL_ERROR("HDStage plan mismatch, noPower=%u, powerFactor=%u, rankSize=%u",
            plan.noPower,
            plan.powerFactor,
            param_.topoInfo.rankSize);
        return HCCL_E_INTERNAL;
    }
    if (plan.finalSteps > plan.powerSteps || plan.finalSteps > 2U) {
        HCCL_ERROR("HDStage finalSteps is invalid, finalSteps=%u, powerSteps=%u",
            plan.finalSteps,
            plan.powerSteps);
        return HCCL_E_INTERNAL;
    }
    if ((plan.remainingPowerSteps + plan.finalSteps) != plan.powerSteps) {
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
    const uint32_t revIdx = ReverseLowerBits(groupIdx, plan.powerSteps);
    const uint32_t batchRankCount = param_.topoInfo.rankSize >> plan.finalSteps;
    if (batchRankCount == 0U) {
        HCCL_ERROR("HDStage batchRankCount is zero, rankSize=%u, finalSteps=%u",
            param_.topoInfo.rankSize,
            plan.finalSteps);
        return HCCL_E_INTERNAL;
    }

    const uint64_t subgroupBaseOffset =
        (static_cast<uint64_t>((revIdx * plan.noPower) % batchRankCount) * packedBytes_);
    if ((subgroupBaseOffset + static_cast<uint64_t>(plan.noPower) * packedBytes_) > resCtx_.localBuffer.size) {
        HCCL_ERROR("HDStage subgroup buffer exceeds localBuffer, baseOffset=%llu, subgroupBytes=%llu, localBuffer=%llu",
            static_cast<unsigned long long>(subgroupBaseOffset),
            static_cast<unsigned long long>(static_cast<uint64_t>(plan.noPower) * packedBytes_),
            static_cast<unsigned long long>(resCtx_.localBuffer.size));
        return HCCL_E_INTERNAL;
    }

    runCtx = {};
    runCtx.rank = group;
    runCtx.rankSize = plan.noPower;
    runCtx.packedBytes = packedBytes_;
    runCtx.baseOffset = subgroupBaseOffset;
    runCtx.inputBase = static_cast<uint8_t *>(resCtx_.localBuffer.addr) + subgroupBaseOffset;
    runCtx.outputBase = static_cast<uint8_t *>(resCtx_.localBuffer.addr) + subgroupBaseOffset;
    runCtx.needMerge = true;
    runCtx.keepOrder = false;
    runCtx.subgroupRanks.reserve(plan.noPower);
    runCtx.rankBaseOffsets.assign(plan.noPower, 0U);
    runCtx.sliceTemplate.push_back(LocalSlice { 0, packedBytes_ });
    for (uint32_t idx = 0; idx < plan.noPower; ++idx) {
        runCtx.subgroupRanks.push_back(idx * plan.powerFactor + groupIdx);
        const uint32_t mappedIdx = GetNoPowerMappedIndex(plan.noPower, idx);
        runCtx.rankBaseOffsets[mappedIdx] = static_cast<uint64_t>(mappedIdx) * packedBytes_;
    }
    if (runCtx.rank >= runCtx.rankSize) {
        HCCL_ERROR("HDStage subgroupRank=%u is out of range, subgroupSize=%u",
            runCtx.rank,
            runCtx.rankSize);
        return HCCL_E_INTERNAL;
    }

    HCCL_CHK_RET(ValidatePreparedNHRLayout(plan, runCtx));
    HCCL_INFO("HDStage built NHR run ctx: rank=%u, group=%u, groupIdx=%u, revIdx=%u, subgroupRank=%u, subgroupSize=%u, baseOffset=%llu",
        rank,
        group,
        groupIdx,
        revIdx,
        runCtx.rank,
        runCtx.rankSize,
        static_cast<unsigned long long>(runCtx.baseOffset));
    return HCCL_SUCCESS;
}

HcclResult AllGatherHDStageCore::ValidatePreparedNHRLayout(const HDStagePlan &plan, const NHRRunCtx &runCtx) const
{
    if (runCtx.sliceTemplate.empty() || runCtx.rankBaseOffsets.size() != runCtx.rankSize) {
        HCCL_ERROR("HDStage prepared NHR layout is incomplete, subgroupSize=%u, sliceTemplate=%u, rankBaseOffsets=%u",
            runCtx.rankSize,
            static_cast<uint32_t>(runCtx.sliceTemplate.size()),
            static_cast<uint32_t>(runCtx.rankBaseOffsets.size()));
        return HCCL_E_INTERNAL;
    }

    uint64_t coveredBytes = 0;
    uint64_t expectedOffset = 0;
    for (const LocalSlice &slice : runCtx.sliceTemplate) {
        if (slice.offset != expectedOffset) {
            HCCL_ERROR("HDStage prepared NHR layout is not contiguous, subgroupSize=%u, offset=%llu, expected=%llu",
                runCtx.rankSize,
                static_cast<unsigned long long>(slice.offset),
                static_cast<unsigned long long>(expectedOffset));
            return HCCL_E_NOT_SUPPORT;
        }
        coveredBytes += slice.size;
        expectedOffset += slice.size;
    }
    if (coveredBytes != packedBytes_) {
        HCCL_ERROR("HDStage prepared NHR layout size mismatch, subgroupSize=%u, coveredBytes=%llu, packedBytes=%llu",
            runCtx.rankSize,
            static_cast<unsigned long long>(coveredBytes),
            static_cast<unsigned long long>(packedBytes_));
        return HCCL_E_INTERNAL;
    }

    const uint8_t *expectedWorkspaceBase = static_cast<const uint8_t *>(resCtx_.localBuffer.addr) + runCtx.baseOffset;
    if (runCtx.inputBase != expectedWorkspaceBase || runCtx.outputBase != expectedWorkspaceBase) {
        HCCL_ERROR("HDStage prepared NHR workspace base mismatch, baseOffset=%llu, input=%p, output=%p, expected=%p",
            static_cast<unsigned long long>(runCtx.baseOffset),
            runCtx.inputBase,
            runCtx.outputBase,
            expectedWorkspaceBase);
        return HCCL_E_INTERNAL;
    }

    const uint32_t expectedSubgroupRank = param_.topoInfo.rank / plan.powerFactor;
    if (runCtx.rank != expectedSubgroupRank || runCtx.rankSize != plan.noPower) {
        HCCL_ERROR("HDStage prepared NHR subgroup identity mismatch, subgroupRank=%u, expectedRank=%u, subgroupSize=%u, expectedSize=%u",
            runCtx.rank,
            expectedSubgroupRank,
            runCtx.rankSize,
            plan.noPower);
        return HCCL_E_INTERNAL;
    }
    if (runCtx.subgroupRanks.size() != plan.noPower) {
        HCCL_ERROR("HDStage prepared NHR subgroup rank list mismatch, subgroupRanks=%u, expected=%u",
            static_cast<uint32_t>(runCtx.subgroupRanks.size()),
            plan.noPower);
        return HCCL_E_INTERNAL;
    }

    const uint32_t groupIdx = param_.topoInfo.rank % plan.powerFactor;
    for (uint32_t idx = 0; idx < plan.noPower; ++idx) {
        const uint32_t globalRank = idx * plan.powerFactor + groupIdx;
        if (runCtx.subgroupRanks[idx] != globalRank) {
            HCCL_ERROR("HDStage prepared NHR subgroup rank mismatch, subgroupIdx=%u, globalRank=%u, expected=%u",
                idx,
                runCtx.subgroupRanks[idx],
                globalRank);
            return HCCL_E_INTERNAL;
        }
        const uint32_t mappedIdx = GetNoPowerMappedIndex(plan.noPower, idx);
        const uint64_t expectedAbsoluteBase = GetStageRankBaseOffset(layout_, globalRank);
        const uint64_t actualAbsoluteBase = runCtx.baseOffset + runCtx.rankBaseOffsets[mappedIdx];
        if (actualAbsoluteBase != expectedAbsoluteBase) {
            HCCL_ERROR("HDStage prepared NHR rank base mismatch, subgroupIdx=%u, mappedIdx=%u, globalRank=%u, base=%llu, expected=%llu",
                idx,
                mappedIdx,
                globalRank,
                static_cast<unsigned long long>(actualAbsoluteBase),
                static_cast<unsigned long long>(expectedAbsoluteBase));
            return HCCL_E_INTERNAL;
        }
    }

    return HCCL_SUCCESS;
}
HcclResult AllGatherHDStageCore::RunPreCopy() const
{
    if (layout_.localSlices.size() != 1U) {
        HCCL_ERROR("HDStage pre-copy expects one local slice, rank=%u, localSlices=%u",
            param_.topoInfo.rank,
            static_cast<uint32_t>(layout_.localSlices.size()));
        return HCCL_E_INTERNAL;
    }

    const uint32_t stageRankIdx = GetStageRankIndex(layout_, param_.topoInfo.rank);
    const uint64_t localRankBaseOffset = GetStageRankBaseOffset(layout_, param_.topoInfo.rank);
    const WindowStageSlice &slice = layout_.localSlices.front();
    if (slice.rank != param_.topoInfo.rank || slice.itemIdx != 0U || slice.itemOffsetBytes != 0U ||
        slice.rankOffsetBytes != 0U || slice.size != packedBytes_ || slice.stageOffsetBytes != localRankBaseOffset) {
        HCCL_ERROR("HDStage pre-copy slice mismatch, rank=%u, sliceRank=%u, item=%u, itemOffset=%llu, rankOffset=%llu, size=%llu, expectedSize=%llu, stageOffset=%llu, expectedStageOffset=%llu",
            param_.topoInfo.rank,
            slice.rank,
            slice.itemIdx,
            static_cast<unsigned long long>(slice.itemOffsetBytes),
            static_cast<unsigned long long>(slice.rankOffsetBytes),
            static_cast<unsigned long long>(slice.size),
            static_cast<unsigned long long>(packedBytes_),
            static_cast<unsigned long long>(slice.stageOffsetBytes),
            static_cast<unsigned long long>(localRankBaseOffset));
        return HCCL_E_INTERNAL;
    }
    HCCL_INFO("HDStage pre-copy ready: rank=%u, stageRankIdx=%u, packedBytes=%llu, localRankBaseOffset=%llu",
        param_.topoInfo.rank,
        stageRankIdx,
        static_cast<unsigned long long>(packedBytes_),
        static_cast<unsigned long long>(localRankBaseOffset));
    return HCCL_SUCCESS;
}

HcclResult AllGatherHDStageCore::RunNHR(const char *pathTag, const NHRRunCtx &runCtx) const
{
    const uint32_t logicalSliceCount = static_cast<uint32_t>(runCtx.rankBaseOffsets.size() * runCtx.sliceTemplate.size());
    HCCL_INFO("HDStage delegates subgroup to NHR: path=%s, rank=%u, packedBytes=%llu, subgroupRank=%u, subgroupSize=%u, sliceCount=%u",
        pathTag,
        param_.topoInfo.rank,
        static_cast<unsigned long long>(packedBytes_),
        runCtx.rank,
        runCtx.rankSize,
        logicalSliceCount);
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

HcclResult AllGatherHDStageCore::BuildPowerStepTask(
    const HDStagePlan &plan, uint32_t bit, PowerStepTask &stepTask) const
{
    if (plan.powerFactor == 0U || bit >= plan.powerSteps) {
        HCCL_ERROR("HDStage power bit is invalid, bit=%u, powerSteps=%u, powerFactor=%u",
            bit,
            plan.powerSteps,
            plan.powerFactor);
        return HCCL_E_INTERNAL;
    }

    const uint32_t rank = param_.topoInfo.rank;
    const uint32_t group = rank / plan.powerFactor;
    const uint32_t groupIdx = rank % plan.powerFactor;
    const uint32_t partnerGroupIdx = groupIdx ^ (1U << bit);
    const uint32_t partnerRank = group * plan.powerFactor + partnerGroupIdx;
    const uint32_t revGroup = ReverseLowerBits(groupIdx, plan.powerSteps);
    const uint32_t step = plan.powerSteps - 1U - bit;
    const uint32_t sliceNum = 1U << step;
    const uint32_t offsetGroup = ((revGroup ^ (1U << step)) / sliceNum) * sliceNum;
    const uint64_t sliceUnitBytes = static_cast<uint64_t>(plan.noPower) * packedBytes_;
    const uint64_t batchBytes = (static_cast<uint64_t>(param_.topoInfo.rankSize) * packedBytes_) >> plan.finalSteps;
    const uint64_t sliceOffset = (static_cast<uint64_t>(offsetGroup) * sliceUnitBytes) % batchBytes;
    const uint64_t sliceBytes = static_cast<uint64_t>(sliceNum) * sliceUnitBytes;

    if (sliceBytes == 0 || batchBytes == 0 || (sliceOffset + sliceBytes) > resCtx_.localBuffer.size) {
        HCCL_ERROR("HDStage power slice is invalid, bit=%u, sliceOffset=%llu, sliceBytes=%llu, batchBytes=%llu, localBuffer=%llu",
            bit,
            static_cast<unsigned long long>(sliceOffset),
            static_cast<unsigned long long>(sliceBytes),
            static_cast<unsigned long long>(batchBytes),
            static_cast<unsigned long long>(resCtx_.localBuffer.size));
        return HCCL_E_INTERNAL;
    }

    stepTask.bit = bit;
    stepTask.remoteRank = partnerRank;
    stepTask.totalBytes = sliceBytes;
    stepTask.localOffset = sliceOffset;
    stepTask.remoteOffset = sliceOffset;
    stepTask.size = sliceBytes;
    return HCCL_SUCCESS;
}



HcclResult AllGatherHDStageCore::RunPowerBit(const PowerStepTask &stepTask, const char *stageTag) const
{
    HCCL_INFO("HDStage %s bit exchange: rank=%u, bit=%u, remoteRank=%u, mergedSlices=%u, totalBytes=%llu, packedBytes=%llu",
        stageTag,
        param_.topoInfo.rank,
        stepTask.bit,
        stepTask.remoteRank,
        CalcMergedSliceCount(stepTask),
        static_cast<unsigned long long>(stepTask.totalBytes),
        static_cast<unsigned long long>(packedBytes_));
    if (stepTask.size == 0) {
        HCCL_ERROR("HDStage %s step task is empty, bit=%u, remoteRank=%u", stageTag, stepTask.bit, stepTask.remoteRank);
        return HCCL_E_INTERNAL;
    }
    const ChannelResource *channel = FindChannel(stepTask.remoteRank);
    if (channel == nullptr) {
        HCCL_ERROR("HDStage %s channel to remoteRank=%u is missing", stageTag, stepTask.remoteRank);
        return HCCL_E_NOT_FOUND;
    }
    HCCL_RUN_INFO("HDStage %s power runinfo: rank=%u, bit=%u, remoteRank=%u, phase=notify-record-begin, localNotify=%u, remoteNotify=%u, localOffset=%llu, remoteOffset=%llu, size=%llu",
        stageTag,
        param_.topoInfo.rank,
        stepTask.bit,
        stepTask.remoteRank,
        channel->localNotifyIdx,
        channel->remoteNotifyIdx,
        static_cast<unsigned long long>(stepTask.localOffset),
        static_cast<unsigned long long>(stepTask.remoteOffset),
        static_cast<unsigned long long>(stepTask.size));
    int32_t ret = HcommChannelNotifyRecordOnThread(
        resCtx_.mainThreadHandle,
        channel->handle,
        channel->remoteNotifyIdx);
    if (ret != HCCL_SUCCESS) {
        HCCL_ERROR("HDStage %s notify record failed, remoteRank=%u, ret=%d", stageTag, stepTask.remoteRank, ret);
        return static_cast<HcclResult>(ret);
    }
    HCCL_RUN_INFO("HDStage %s power runinfo: rank=%u, bit=%u, remoteRank=%u, phase=notify-wait-begin, localNotify=%u, remoteNotify=%u",
        stageTag,
        param_.topoInfo.rank,
        stepTask.bit,
        stepTask.remoteRank,
        channel->localNotifyIdx,
        channel->remoteNotifyIdx);
    ret = HcommChannelNotifyWaitOnThread(
        resCtx_.mainThreadHandle,
        channel->handle,
        channel->localNotifyIdx,
        kAllGatherBatchCustomTimeoutMs);
    if (ret != HCCL_SUCCESS) {
        HCCL_ERROR("HDStage %s notify wait failed, remoteRank=%u, ret=%d", stageTag, stepTask.remoteRank, ret);
        return static_cast<HcclResult>(ret);
    }
    HCCL_RUN_INFO("HDStage %s power runinfo: rank=%u, bit=%u, remoteRank=%u, phase=notify-wait-end",
        stageTag,
        param_.topoInfo.rank,
        stepTask.bit,
        stepTask.remoteRank);
    if ((stepTask.localOffset + stepTask.size) > resCtx_.localBuffer.size ||
        (stepTask.remoteOffset + stepTask.size) > channel->remoteBuffer.size) {
        HCCL_ERROR("HDStage %s slice is out of buffer range, remoteRank=%u, localOffset=%llu, remoteOffset=%llu, size=%llu, localBuffer=%llu, remoteBuffer=%llu",
            stageTag,
            stepTask.remoteRank,
            static_cast<unsigned long long>(stepTask.localOffset),
            static_cast<unsigned long long>(stepTask.remoteOffset),
            static_cast<unsigned long long>(stepTask.size),
            static_cast<unsigned long long>(resCtx_.localBuffer.size),
            static_cast<unsigned long long>(channel->remoteBuffer.size));
        return HCCL_E_INTERNAL;
    }
    void *dst = static_cast<uint8_t *>(resCtx_.localBuffer.addr) + stepTask.localOffset;
    const void *src = static_cast<const uint8_t *>(channel->remoteBuffer.addr) + stepTask.remoteOffset;
    HCCL_RUN_INFO("HDStage %s power runinfo: rank=%u, bit=%u, remoteRank=%u, phase=read-begin, localOffset=%llu, remoteOffset=%llu, size=%llu",
        stageTag,
        param_.topoInfo.rank,
        stepTask.bit,
        stepTask.remoteRank,
        static_cast<unsigned long long>(stepTask.localOffset),
        static_cast<unsigned long long>(stepTask.remoteOffset),
        static_cast<unsigned long long>(stepTask.size));
    ret = HcommReadOnThread(resCtx_.mainThreadHandle, channel->handle, dst, src, stepTask.size);
    if (ret != HCCL_SUCCESS) {
        HCCL_ERROR("HDStage %s slice read failed, remoteRank=%u, localOffset=%llu, remoteOffset=%llu, size=%llu, ret=%d",
            stageTag,
            stepTask.remoteRank,
            static_cast<unsigned long long>(stepTask.localOffset),
            static_cast<unsigned long long>(stepTask.remoteOffset),
            static_cast<unsigned long long>(stepTask.size),
            ret);
        return static_cast<HcclResult>(ret);
    }
    HCCL_RUN_INFO("HDStage %s power runinfo: rank=%u, bit=%u, remoteRank=%u, phase=read-end",
        stageTag,
        param_.topoInfo.rank,
        stepTask.bit,
        stepTask.remoteRank);
    return HCCL_SUCCESS;
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
        PowerStepTask stepTask {};
        HCCL_CHK_RET(BuildPowerStepTask(plan, bit, stepTask));
        HCCL_CHK_RET(RunPowerBit(stepTask, "power"));
    }
    return HCCL_SUCCESS;
}

HcclResult AllGatherHDStageCore::RunAllGatherFinal(const HDStagePlan &plan) const
{
    switch (plan.finalSteps) {
        case 2U:
            return RunAllGatherLastTwo(plan);
        case 1U:
            return RunAllGatherLastOne(plan);
        case 0U:
            return RunAllGatherLast(plan);
        default:
            HCCL_ERROR("HDStage final stage uses unexpected finalSteps=%u", plan.finalSteps);
            return HCCL_E_INTERNAL;
    }
}
HcclResult AllGatherHDStageCore::RunAllGatherLast(const HDStagePlan &plan) const
{
    HCCL_INFO("HDStage last stage: rank=%u, needFinal=%s, packedBytes=%llu",
        param_.topoInfo.rank,
        plan.needFinalPath ? "true" : "false",
        static_cast<unsigned long long>(packedBytes_));
    if (plan.finalSteps != 0U) {
        HCCL_ERROR("HDStage last stage is called with unexpected finalSteps=%u", plan.finalSteps);
        return HCCL_E_INTERNAL;
    }

    return HCCL_SUCCESS;
}

HcclResult AllGatherHDStageCore::RunAllGatherLastOne(const HDStagePlan &plan) const
{
    HCCL_INFO("HDStage last-one stage: rank=%u, powerSteps=%u, needFinal=%s, packedBytes=%llu",
        param_.topoInfo.rank,
        plan.powerSteps,
        plan.needFinalPath ? "true" : "false",
        static_cast<unsigned long long>(packedBytes_));
    if (plan.finalSteps != 1U) {
        HCCL_ERROR("HDStage last-one stage is called with unexpected finalSteps=%u", plan.finalSteps);
        return HCCL_E_INTERNAL;
    }
    std::vector<LastTwoCopyTask> tasks;
    HCCL_CHK_RET(BuildFinalCopyTasks(2U, tasks));
    const LastTwoCopyTask *localTask = nullptr;
    const LastTwoCopyTask *remoteTask = nullptr;
    for (const LastTwoCopyTask &task : tasks) {
        if (task.isLocal) {
            if (localTask != nullptr) {
                HCCL_ERROR("HDStage lastOne duplicated local task, rank=%u", param_.topoInfo.rank);
                return HCCL_E_INTERNAL;
            }
            localTask = &task;
        } else {
            if (remoteTask != nullptr) {
                HCCL_ERROR("HDStage lastOne duplicated remote task, rank=%u", param_.topoInfo.rank);
                return HCCL_E_INTERNAL;
            }
            remoteTask = &task;
        }
    }
    if (localTask == nullptr || remoteTask == nullptr) {
        HCCL_ERROR("HDStage lastOne final tasks are incomplete, rank=%u", param_.topoInfo.rank);
        return HCCL_E_INTERNAL;
    }
    HCCL_CHK_RET(RunLastTwoLocalTask(*localTask, "lastOne"));
    HCCL_CHK_RET(RunRemoteTaskOnMainThread(*remoteTask, "lastOne"));
    return HCCL_SUCCESS;
}

HcclResult AllGatherHDStageCore::BuildFinalCopyTasks(uint32_t subgroupSize, std::vector<LastTwoCopyTask> &tasks) const
{
    if (subgroupSize == 0U || (subgroupSize & (subgroupSize - 1U)) != 0U) {
        HCCL_ERROR("HDStage final subgroupSize is invalid, subgroupSize=%u", subgroupSize);
        return HCCL_E_INTERNAL;
    }
    if ((layout_.totalBytes % subgroupSize) != 0U) {
        HCCL_ERROR("HDStage final chunk size is invalid, totalBytes=%llu, subgroupSize=%u",
            static_cast<unsigned long long>(layout_.totalBytes),
            subgroupSize);
        return HCCL_E_INTERNAL;
    }

    const uint32_t subgroupRankBase = (param_.topoInfo.rank / subgroupSize) * subgroupSize;
    const uint32_t subgroupRankIdx = param_.topoInfo.rank % subgroupSize;
    const uint64_t chunkBytes = layout_.totalBytes / subgroupSize;
    static const uint32_t kLastTwoResMap[4] = {0U, 2U, 1U, 3U};

    tasks.clear();
    tasks.resize(subgroupSize);
    for (uint32_t subgroupIdx = 0; subgroupIdx < subgroupSize; ++subgroupIdx) {
        LastTwoCopyTask &task = tasks[subgroupIdx];
        task.peerRank = subgroupRankBase + subgroupIdx;
        task.isLocal = (subgroupIdx == subgroupRankIdx);

        uint32_t chunkIdx = subgroupIdx;
        if (subgroupSize == 4U) {
            chunkIdx = kLastTwoResMap[subgroupIdx];
        }
        const uint64_t chunkOffset = static_cast<uint64_t>(chunkIdx) * chunkBytes;
        if ((chunkOffset + chunkBytes) > layout_.totalBytes) {
            HCCL_ERROR("HDStage final chunk exceeds layout, peerRank=%u, chunkIdx=%u, chunkOffset=%llu, chunkBytes=%llu, totalBytes=%llu",
                task.peerRank,
                chunkIdx,
                static_cast<unsigned long long>(chunkOffset),
                static_cast<unsigned long long>(chunkBytes),
                static_cast<unsigned long long>(layout_.totalBytes));
            return HCCL_E_INTERNAL;
        }

        StageCopySlice taskSlice;
        taskSlice.remoteRank = task.peerRank;
        taskSlice.localOffset = chunkOffset;
        taskSlice.remoteOffset = 0;
        taskSlice.size = chunkBytes;
        task.slices.push_back(taskSlice);
    }
    return ValidateFinalCopyTasks(subgroupSize, tasks);
}

HcclResult AllGatherHDStageCore::ValidateFinalCopyTasks(uint32_t subgroupSize, const std::vector<LastTwoCopyTask> &tasks) const
{
    if (tasks.size() != subgroupSize) {
        HCCL_ERROR("HDStage final task count mismatch, subgroupSize=%u, tasks=%u", subgroupSize, static_cast<uint32_t>(tasks.size()));
        return HCCL_E_INTERNAL;
    }

    if ((layout_.totalBytes % subgroupSize) != 0U) {
        HCCL_ERROR("HDStage final validation found invalid chunk split, totalBytes=%llu, subgroupSize=%u",
            static_cast<unsigned long long>(layout_.totalBytes),
            subgroupSize);
        return HCCL_E_INTERNAL;
    }

    const uint64_t chunkBytes = layout_.totalBytes / subgroupSize;
    const uint32_t subgroupRankBase = (param_.topoInfo.rank / subgroupSize) * subgroupSize;
    uint32_t localTaskCount = 0;
    std::vector<bool> covered(subgroupSize, false);
    for (const LastTwoCopyTask &task : tasks) {
        if (task.peerRank < subgroupRankBase || task.peerRank >= (subgroupRankBase + subgroupSize)) {
            HCCL_ERROR("HDStage final peer rank is out of subgroup, peerRank=%u, subgroupBase=%u, subgroupSize=%u",
                task.peerRank,
                subgroupRankBase,
                subgroupSize);
            return HCCL_E_INTERNAL;
        }
        if (task.isLocal) {
            ++localTaskCount;
            if (task.peerRank != param_.topoInfo.rank) {
                HCCL_ERROR("HDStage final local task rank mismatch, rank=%u, peerRank=%u",
                    param_.topoInfo.rank,
                    task.peerRank);
                return HCCL_E_INTERNAL;
            }
        }
        if (task.slices.size() != 1U) {
            HCCL_ERROR("HDStage final task slice count mismatch, peerRank=%u, slices=%u",
                task.peerRank,
                static_cast<uint32_t>(task.slices.size()));
            return HCCL_E_INTERNAL;
        }

        const StageCopySlice &slice = task.slices.front();
        if (slice.remoteRank != task.peerRank || slice.remoteOffset != 0U || slice.size != chunkBytes) {
            HCCL_ERROR("HDStage final slice layout mismatch, peerRank=%u, remoteRank=%u, remoteOffset=%llu, size=%llu, chunkBytes=%llu",
                task.peerRank,
                slice.remoteRank,
                static_cast<unsigned long long>(slice.remoteOffset),
                static_cast<unsigned long long>(slice.size),
                static_cast<unsigned long long>(chunkBytes));
            return HCCL_E_INTERNAL;
        }
        if ((slice.localOffset % chunkBytes) != 0U || slice.localOffset >= layout_.totalBytes) {
            HCCL_ERROR("HDStage final local offset is invalid, peerRank=%u, localOffset=%llu, chunkBytes=%llu, totalBytes=%llu",
                task.peerRank,
                static_cast<unsigned long long>(slice.localOffset),
                static_cast<unsigned long long>(chunkBytes),
                static_cast<unsigned long long>(layout_.totalBytes));
            return HCCL_E_INTERNAL;
        }
        const uint32_t chunkIdx = static_cast<uint32_t>(slice.localOffset / chunkBytes);
        if (chunkIdx >= subgroupSize || covered[chunkIdx]) {
            HCCL_ERROR("HDStage final chunk coverage is invalid, peerRank=%u, chunkIdx=%u, subgroupSize=%u, alreadyCovered=%s",
                task.peerRank,
                chunkIdx,
                subgroupSize,
                (chunkIdx < subgroupSize && covered[chunkIdx]) ? "true" : "false");
            return HCCL_E_INTERNAL;
        }
        const uint64_t stageBaseOffset = GetStageRankBaseOffset(layout_, task.peerRank);
        const uint32_t stageChunkIdx = static_cast<uint32_t>(stageBaseOffset / chunkBytes);
        if (stageChunkIdx != chunkIdx || (stageBaseOffset + packedBytes_) > (slice.localOffset + slice.size)) {
            HCCL_ERROR("HDStage final chunk mapping mismatches stage layout, peerRank=%u, chunkIdx=%u, stageChunkIdx=%u, localOffset=%llu, stageBaseOffset=%llu, packedBytes=%llu, chunkBytes=%llu",
                task.peerRank,
                chunkIdx,
                stageChunkIdx,
                static_cast<unsigned long long>(slice.localOffset),
                static_cast<unsigned long long>(stageBaseOffset),
                static_cast<unsigned long long>(packedBytes_),
                static_cast<unsigned long long>(chunkBytes));
            return HCCL_E_INTERNAL;
        }
        covered[chunkIdx] = true;
    }

    if (localTaskCount != 1U) {
        HCCL_ERROR("HDStage final local task count mismatch, rank=%u, localTasks=%u",
            param_.topoInfo.rank,
            localTaskCount);
        return HCCL_E_INTERNAL;
    }
    for (uint32_t idx = 0; idx < subgroupSize; ++idx) {
        if (!covered[idx]) {
            HCCL_ERROR("HDStage final chunk coverage is incomplete, missingChunk=%u, subgroupSize=%u", idx, subgroupSize);
            return HCCL_E_INTERNAL;
        }
    }
    return HCCL_SUCCESS;
}

HcclResult AllGatherHDStageCore::BuildLastTwoTasks(const HDStagePlan &plan, std::vector<LastTwoCopyTask> &tasks) const
{
    if (plan.finalSteps != 2U) {
        HCCL_ERROR("HDStage lastTwo requires finalSteps=2, actual=%u", plan.finalSteps);
        return HCCL_E_INTERNAL;
    }
    return BuildFinalCopyTasks(4U, tasks);
}

HcclResult AllGatherHDStageCore::RunLastTwoLocalTask(const LastTwoCopyTask &task, const char *stageTag) const
{
    if (!task.isLocal) {
        HCCL_ERROR("HDStage %s local task is marked as remote, peerRank=%u", stageTag, task.peerRank);
        return HCCL_E_INTERNAL;
    }
    if (task.peerRank != param_.topoInfo.rank) {
        HCCL_ERROR("HDStage %s local task peerRank mismatch, rank=%u, peerRank=%u",
            stageTag,
            param_.topoInfo.rank,
            task.peerRank);
        return HCCL_E_INTERNAL;
    }

    for (const StageCopySlice &slice : task.slices) {
        if ((slice.localOffset + slice.size) > resCtx_.localBuffer.size ||
            (slice.remoteOffset + slice.size) > resCtx_.localBuffer.size) {
            HCCL_ERROR("HDStage %s local slice is out of range, localOffset=%llu, remoteOffset=%llu, size=%llu, localBuffer=%llu",
                stageTag,
                static_cast<unsigned long long>(slice.localOffset),
                static_cast<unsigned long long>(slice.remoteOffset),
                static_cast<unsigned long long>(slice.size),
                static_cast<unsigned long long>(resCtx_.localBuffer.size));
            return HCCL_E_INTERNAL;
        }
        if (slice.localOffset == slice.remoteOffset) {
            continue;
        }
        void *dst = static_cast<uint8_t *>(resCtx_.localBuffer.addr) + slice.localOffset;
        const void *src = static_cast<const uint8_t *>(resCtx_.localBuffer.addr) + slice.remoteOffset;
        const int32_t ret = HcommLocalCopyOnThread(resCtx_.mainThreadHandle, dst, src, slice.size);
        if (ret != HCCL_SUCCESS) {
            HCCL_ERROR("HDStage %s local copy failed, localOffset=%llu, remoteOffset=%llu, size=%llu, ret=%d",
                stageTag,
                static_cast<unsigned long long>(slice.localOffset),
                static_cast<unsigned long long>(slice.remoteOffset),
                static_cast<unsigned long long>(slice.size),
                ret);
            return static_cast<HcclResult>(ret);
        }
    }
    return HCCL_SUCCESS;
}

HcclResult AllGatherHDStageCore::RunRemoteTaskOnMainThread(const LastTwoCopyTask &task, const char *stageTag) const
{
    if (task.isLocal) {
        HCCL_ERROR("HDStage %s remote task is marked as local, peerRank=%u", stageTag, task.peerRank);
        return HCCL_E_INTERNAL;
    }
    const ChannelResource *channel = FindChannel(task.peerRank);
    if (channel == nullptr) {
        HCCL_ERROR("HDStage %s channel to remoteRank=%u is missing", stageTag, task.peerRank);
        return HCCL_E_NOT_FOUND;
    }

    HCCL_RUN_INFO("HDStage %s runinfo: rank=%u, peerRank=%u, phase=notify-record-begin, localNotify=%u, remoteNotify=%u, sliceCount=%u",
        stageTag,
        param_.topoInfo.rank,
        task.peerRank,
        channel->localNotifyIdx,
        channel->remoteNotifyIdx,
        static_cast<uint32_t>(task.slices.size()));
    int32_t ret = HcommChannelNotifyRecordOnThread(resCtx_.mainThreadHandle, channel->handle, channel->remoteNotifyIdx);
    if (ret != HCCL_SUCCESS) {
        HCCL_ERROR("HDStage %s notify record failed, peerRank=%u, ret=%d", stageTag, task.peerRank, ret);
        return static_cast<HcclResult>(ret);
    }
    HCCL_RUN_INFO("HDStage %s runinfo: rank=%u, peerRank=%u, phase=notify-wait-begin, localNotify=%u, remoteNotify=%u",
        stageTag,
        param_.topoInfo.rank,
        task.peerRank,
        channel->localNotifyIdx,
        channel->remoteNotifyIdx);
    ret = HcommChannelNotifyWaitOnThread(resCtx_.mainThreadHandle, channel->handle, channel->localNotifyIdx, kAllGatherBatchCustomTimeoutMs);
    if (ret != HCCL_SUCCESS) {
        HCCL_ERROR("HDStage %s notify wait failed, peerRank=%u, ret=%d", stageTag, task.peerRank, ret);
        return static_cast<HcclResult>(ret);
    }
    HCCL_RUN_INFO("HDStage %s runinfo: rank=%u, peerRank=%u, phase=notify-wait-end",
        stageTag,
        param_.topoInfo.rank,
        task.peerRank);

    uint32_t sliceIdx = 0;
    for (const StageCopySlice &slice : task.slices) {
        if ((slice.localOffset + slice.size) > resCtx_.localBuffer.size ||
            (slice.remoteOffset + slice.size) > channel->remoteBuffer.size) {
            HCCL_ERROR("HDStage %s remote slice is out of range, peerRank=%u, localOffset=%llu, remoteOffset=%llu, size=%llu, localBuffer=%llu, remoteBuffer=%llu",
                stageTag,
                task.peerRank,
                static_cast<unsigned long long>(slice.localOffset),
                static_cast<unsigned long long>(slice.remoteOffset),
                static_cast<unsigned long long>(slice.size),
                static_cast<unsigned long long>(resCtx_.localBuffer.size),
                static_cast<unsigned long long>(channel->remoteBuffer.size));
            return HCCL_E_INTERNAL;
        }
        void *dst = static_cast<uint8_t *>(resCtx_.localBuffer.addr) + slice.localOffset;
        const void *src = static_cast<const uint8_t *>(channel->remoteBuffer.addr) + slice.remoteOffset;
        HCCL_RUN_INFO("HDStage %s runinfo: rank=%u, peerRank=%u, phase=read-begin, sliceIdx=%u, localOffset=%llu, remoteOffset=%llu, size=%llu",
            stageTag,
            param_.topoInfo.rank,
            task.peerRank,
            sliceIdx,
            static_cast<unsigned long long>(slice.localOffset),
            static_cast<unsigned long long>(slice.remoteOffset),
            static_cast<unsigned long long>(slice.size));
        ret = HcommReadOnThread(resCtx_.mainThreadHandle, channel->handle, dst, src, slice.size);
        if (ret != HCCL_SUCCESS) {
            HCCL_ERROR("HDStage %s remote read failed, peerRank=%u, localOffset=%llu, remoteOffset=%llu, size=%llu, ret=%d",
                stageTag,
                task.peerRank,
                static_cast<unsigned long long>(slice.localOffset),
                static_cast<unsigned long long>(slice.remoteOffset),
                static_cast<unsigned long long>(slice.size),
                ret);
            return static_cast<HcclResult>(ret);
        }
        HCCL_RUN_INFO("HDStage %s runinfo: rank=%u, peerRank=%u, phase=read-end, sliceIdx=%u",
            stageTag,
            param_.topoInfo.rank,
            task.peerRank,
            sliceIdx);
        ++sliceIdx;
    }
    return HCCL_SUCCESS;
}

HcclResult AllGatherHDStageCore::RunAllGatherLastTwo(const HDStagePlan &plan) const
{
    HCCL_INFO("HDStage last-two stage: rank=%u, powerSteps=%u, needFinal=%s, packedBytes=%llu",
        param_.topoInfo.rank,
        plan.powerSteps,
        plan.needFinalPath ? "true" : "false",
        static_cast<unsigned long long>(packedBytes_));
    if (plan.finalSteps != 2U) {
        HCCL_ERROR("HDStage last-two stage is called with unexpected finalSteps=%u", plan.finalSteps);
        return HCCL_E_INTERNAL;
    }

    std::vector<LastTwoCopyTask> tasks;
    HCCL_CHK_RET(BuildLastTwoTasks(plan, tasks));

    const LastTwoCopyTask *localTask = nullptr;
    for (const LastTwoCopyTask &task : tasks) {
        if (task.isLocal) {
            if (localTask != nullptr) {
                HCCL_ERROR("HDStage lastTwo duplicated local task, rank=%u", param_.topoInfo.rank);
                return HCCL_E_INTERNAL;
            }
            localTask = &task;
            continue;
        }
    }
    if (localTask == nullptr) {
        HCCL_ERROR("HDStage lastTwo local task is missing, rank=%u", param_.topoInfo.rank);
        return HCCL_E_INTERNAL;
    }

    // Prefer the hcomm baseline semantics here: finalize local and remote subgroup chunks
    // on the main thread, instead of routing the formal path through custom worker parallelism.
    HCCL_RUN_INFO("HDStage lastTwo runinfo: rank=%u, phase=local-task-begin, subgroupSize=%u",
        param_.topoInfo.rank,
        static_cast<uint32_t>(tasks.size()));
    HCCL_CHK_RET(RunLastTwoLocalTask(*localTask, "lastTwo"));
    HCCL_RUN_INFO("HDStage lastTwo runinfo: rank=%u, phase=local-task-end",
        param_.topoInfo.rank);
    const uint32_t subgroupSize = static_cast<uint32_t>(tasks.size());
    const uint32_t subgroupRankIdx = param_.topoInfo.rank % subgroupSize;
    for (uint32_t round = 1; round < subgroupSize; ++round) {
        const uint32_t peerSubgroupIdx = (subgroupRankIdx + subgroupSize - round) % subgroupSize;
        HCCL_RUN_INFO("HDStage lastTwo runinfo: rank=%u, phase=remote-task-begin, round=%u, peerRank=%u, peerSubgroupIdx=%u",
            param_.topoInfo.rank,
            round,
            tasks[peerSubgroupIdx].peerRank,
            peerSubgroupIdx);
        HCCL_CHK_RET(RunRemoteTaskOnMainThread(tasks[peerSubgroupIdx], "lastTwo"));
        HCCL_RUN_INFO("HDStage lastTwo runinfo: rank=%u, phase=remote-task-end, round=%u, peerRank=%u, peerSubgroupIdx=%u",
            param_.topoInfo.rank,
            round,
            tasks[peerSubgroupIdx].peerRank,
            peerSubgroupIdx);
    }
    return HCCL_SUCCESS;
}

HcclResult AllGatherHDStageCore::RunAllGatherStage(const HDStagePlan &plan) const
{
    HCCL_CHK_RET(RunPreCopy());

    if (plan.noPower > 1U) {
        HCCL_CHK_RET(RunAllGatherNoPower(plan));
    }
    if (plan.remainingPowerSteps >= 1U) {
        HCCL_CHK_RET(RunAllGatherPower(plan));
    }

    if (plan.finalSteps == 2U) {
        HCCL_CHK_RET(RunAllGatherLastTwo(plan));
    } else if (plan.finalSteps == 1U) {
        HCCL_CHK_RET(RunAllGatherLastOne(plan));
    } else {
        HCCL_CHK_RET(RunAllGatherLast(plan));
    }
    return HCCL_SUCCESS;
}

HcclResult AllGatherHDStageCore::RunAsync()
{
    if (param_.topoInfo.rankSize == 1U) {
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
