#include "all_gather_hd_stage_core.h"

#include <algorithm>
#include <vector>

#include "all_gather_nhr_core.h"
#include "log.h"

namespace ops_hccl_allgatherbatch {

namespace {

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

uint64_t CalcTotalSliceBytes(const std::vector<StageCopySlice> &stepSlices)
{
    uint64_t totalBytes = 0;
    for (const StageCopySlice &slice : stepSlices) {
        totalBytes += slice.size;
    }
    return totalBytes;
}


uint32_t CalcMergedSliceCount(const std::vector<PowerPeerTask> &peers)
{
    uint32_t sliceCount = 0;
    for (const PowerPeerTask &peer : peers) {
        sliceCount += static_cast<uint32_t>(peer.slices.size());
    }
    return sliceCount;
}

bool CanMergeStageCopySlice(const StageCopySlice &lhs, const StageCopySlice &rhs)
{
    return lhs.remoteRank == rhs.remoteRank &&
        (lhs.localOffset + lhs.size) == rhs.localOffset &&
        (lhs.remoteOffset + lhs.size) == rhs.remoteOffset;
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
    plan.powerSteps = layout_.powerSteps;
    plan.powerFactor = layout_.powerFactor;
    plan.noPower = layout_.noPower;
    plan.finalSteps = CalcFinalSteps(plan.powerSteps);
    plan.remainingPowerSteps = (plan.powerSteps > plan.finalSteps) ? (plan.powerSteps - plan.finalSteps) : 0U;
    plan.needNoPowerPath = (plan.noPower > 1U);
    plan.needPowerPath = (plan.remainingPowerSteps >= 1U);
    plan.needFinalPath = (param_.topoInfo.rankSize > 1U);
    return HCCL_SUCCESS;
}

HcclResult AllGatherHDStageCore::ValidateStagePlan(const HDStagePlan &plan) const
{
    if (plan.noPower == 0U) {
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
    runCtx.rank = group;
    runCtx.rankSize = plan.noPower;
    runCtx.sliceGroupSize = static_cast<uint32_t>(layout_.localSlices.size());
    runCtx.packedBytes = packedBytes_;
    runCtx.baseOffset = 0;
    runCtx.inputBase = static_cast<uint8_t *>(resCtx_.localBuffer.addr);
    runCtx.outputBase = static_cast<uint8_t *>(resCtx_.localBuffer.addr);
    runCtx.keepOrder = true;
    runCtx.preparedOutputLayout = true;
    runCtx.subgroupRanks.clear();
    runCtx.rankBaseOffsets.clear();
    runCtx.sliceTemplate.clear();
    runCtx.slices.clear();
    runCtx.subgroupRanks.reserve(plan.noPower);
    runCtx.rankBaseOffsets.reserve(plan.noPower);
    runCtx.sliceTemplate.reserve(layout_.localSlices.size());
    runCtx.slices.reserve(plan.noPower * layout_.localSlices.size());
    for (const WindowStageSlice &slice : layout_.localSlices) {
        runCtx.sliceTemplate.push_back(LocalSlice { slice.rankOffsetBytes, slice.size });
    }
    for (uint32_t idx = 0; idx < plan.noPower; ++idx) {
        const uint32_t globalRank = idx * plan.powerFactor + groupIdx;
        const uint64_t rankBaseOffset = GetStageRankBaseOffset(layout_, globalRank);
        runCtx.subgroupRanks.push_back(globalRank);
        runCtx.rankBaseOffsets.push_back(rankBaseOffset);
        for (const WindowStageSlice &slice : layout_.localSlices) {
            runCtx.slices.push_back(LocalSlice { rankBaseOffset + slice.rankOffsetBytes, slice.size });
        }
    }
    if (runCtx.rank >= runCtx.rankSize) {
        HCCL_ERROR("HDStage subgroupRank=%u is out of range, subgroupSize=%u",
            runCtx.rank,
            runCtx.rankSize);
        return HCCL_E_INTERNAL;
    }
    HCCL_INFO("HDStage built NHR run ctx: rank=%u, group=%u, groupIdx=%u, subgroupRank=%u, subgroupSize=%u, sliceCount=%u, sliceGroupSize=%u",
        rank,
        group,
        groupIdx,
        runCtx.rank,
        runCtx.rankSize,
        static_cast<uint32_t>(runCtx.slices.size()),
        static_cast<uint32_t>(layout_.localSlices.size()));
    return HCCL_SUCCESS;
}

HcclResult AllGatherHDStageCore::RunPreCopy() const
{
    const uint32_t stageRankIdx = GetStageRankIndex(layout_, param_.topoInfo.rank);
    uint8_t *localRankBase = static_cast<uint8_t *>(resCtx_.localBuffer.addr) + GetStageRankBaseOffset(layout_, param_.topoInfo.rank);
    if (localRankBase == nullptr) {
        HCCL_ERROR("HDStage pre-copy base is null");
        return HCCL_E_INTERNAL;
    }
    HCCL_INFO("HDStage pre-copy ready: rank=%u, stageRankIdx=%u, packedBytes=%llu, localRankBase=%p, localSlices=%u",
        param_.topoInfo.rank,
        stageRankIdx,
        static_cast<unsigned long long>(packedBytes_),
        localRankBase,
        static_cast<uint32_t>(layout_.localSlices.size()));
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
    const uint32_t lowMask = (bit == 0U) ? 0U : ((1U << bit) - 1U);
    const uint32_t flippedBit = (((groupIdx >> bit) & 1U) ^ 1U);
    const uint32_t localSliceCount = static_cast<uint32_t>(layout_.localSlices.size());

    stepTask.bit = bit;
    stepTask.totalBytes = 0;
    stepTask.peers.clear();
    stepTask.peers.reserve(1);
    stepTask.peers.push_back(PowerPeerTask {});
    PowerPeerTask &peerTask = stepTask.peers.back();
    peerTask.remoteRank = partnerRank;

    uint32_t selectedColumns = 0;
    for (uint32_t column = 0; column < plan.powerFactor; ++column) {
        if (bit != 0U && ((column & lowMask) != (groupIdx & lowMask))) {
            continue;
        }
        if (((column >> bit) & 1U) != flippedBit) {
            continue;
        }
        ++selectedColumns;
    }
    peerTask.slices.clear();
    peerTask.slices.reserve(static_cast<size_t>(selectedColumns) * plan.noPower * localSliceCount);

    for (uint32_t column = 0; column < plan.powerFactor; ++column) {
        if (bit != 0U && ((column & lowMask) != (groupIdx & lowMask))) {
            continue;
        }
        if (((column >> bit) & 1U) != flippedBit) {
            continue;
        }
        for (uint32_t subgroupIdx = 0; subgroupIdx < plan.noPower; ++subgroupIdx) {
            const uint32_t targetRank = subgroupIdx * plan.powerFactor + column;
            const uint64_t rankBaseOffset = GetStageRankBaseOffset(layout_, targetRank);
            for (const WindowStageSlice &slice : layout_.localSlices) {
                peerTask.slices.push_back(StageCopySlice {
                    partnerRank,
                    rankBaseOffset + slice.rankOffsetBytes,
                    rankBaseOffset + slice.rankOffsetBytes,
                    slice.size });
                stepTask.totalBytes += slice.size;
            }
        }
    }

    if (peerTask.slices.empty()) {
        HCCL_ERROR("HDStage step slice plan is empty, bit=%u, rank=%u, groupIdx=%u",
            bit,
            rank,
            groupIdx);
        return HCCL_E_INTERNAL;
    }

    MergeContiguousSlices(peerTask.slices);
    return HCCL_SUCCESS;
}

HcclResult AllGatherHDStageCore::ReadPowerPeerTask(const PowerPeerTask &peerTask, const char *stageTag) const
{
    if (peerTask.slices.empty()) {
        HCCL_ERROR("HDStage %s peer task is empty, remoteRank=%u", stageTag, peerTask.remoteRank);
        return HCCL_E_INTERNAL;
    }

    const ChannelResource *channel = FindChannel(peerTask.remoteRank);
    if (channel == nullptr) {
        HCCL_ERROR("HDStage %s channel to remoteRank=%u is missing", stageTag, peerTask.remoteRank);
        return HCCL_E_NOT_FOUND;
    }

    int32_t ret = HcommChannelNotifyRecordOnThread(
        resCtx_.mainThreadHandle,
        channel->handle,
        channel->remoteNotifyIdx);
    if (ret != HCCL_SUCCESS) {
        HCCL_ERROR("HDStage %s notify record failed, remoteRank=%u, ret=%d", stageTag, peerTask.remoteRank, ret);
        return static_cast<HcclResult>(ret);
    }
    ret = HcommChannelNotifyWaitOnThread(
        resCtx_.mainThreadHandle,
        channel->handle,
        channel->localNotifyIdx,
        kAllGatherBatchCustomTimeoutMs);
    if (ret != HCCL_SUCCESS) {
        HCCL_ERROR("HDStage %s notify wait failed, remoteRank=%u, ret=%d", stageTag, peerTask.remoteRank, ret);
        return static_cast<HcclResult>(ret);
    }

    for (const StageCopySlice &slice : peerTask.slices) {
        if ((slice.localOffset + slice.size) > resCtx_.localBuffer.size ||
            (slice.remoteOffset + slice.size) > channel->remoteBuffer.size) {
            HCCL_ERROR("HDStage %s slice is out of buffer range, remoteRank=%u, localOffset=%llu, remoteOffset=%llu, size=%llu, localBuffer=%llu, remoteBuffer=%llu",
                stageTag,
                peerTask.remoteRank,
                static_cast<unsigned long long>(slice.localOffset),
                static_cast<unsigned long long>(slice.remoteOffset),
                static_cast<unsigned long long>(slice.size),
                static_cast<unsigned long long>(resCtx_.localBuffer.size),
                static_cast<unsigned long long>(channel->remoteBuffer.size));
            return HCCL_E_INTERNAL;
        }

        void *dst = static_cast<uint8_t *>(resCtx_.localBuffer.addr) + slice.localOffset;
        const void *src = static_cast<const uint8_t *>(channel->remoteBuffer.addr) + slice.remoteOffset;
        ret = HcommReadOnThread(resCtx_.mainThreadHandle, channel->handle, dst, src, slice.size);
        if (ret != HCCL_SUCCESS) {
            HCCL_ERROR("HDStage %s slice read failed, remoteRank=%u, localOffset=%llu, remoteOffset=%llu, size=%llu, ret=%d",
                stageTag,
                peerTask.remoteRank,
                static_cast<unsigned long long>(slice.localOffset),
                static_cast<unsigned long long>(slice.remoteOffset),
                static_cast<unsigned long long>(slice.size),
                ret);
            return static_cast<HcclResult>(ret);
        }
    }

    return HCCL_SUCCESS;
}

HcclResult AllGatherHDStageCore::RunPowerBit(const PowerStepTask &stepTask, const char *stageTag) const
{
    HCCL_INFO("HDStage %s bit exchange: rank=%u, bit=%u, peerTasks=%u, mergedSlices=%u, totalBytes=%llu, packedBytes=%llu",
        stageTag,
        param_.topoInfo.rank,
        stepTask.bit,
        static_cast<uint32_t>(stepTask.peers.size()),
        CalcMergedSliceCount(stepTask.peers),
        static_cast<unsigned long long>(stepTask.totalBytes),
        static_cast<unsigned long long>(packedBytes_));
    for (const PowerPeerTask &peerTask : stepTask.peers) {
        HCCL_CHK_RET(ReadPowerPeerTask(peerTask, stageTag));
    }
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
    std::vector<PowerStepTask> stepTasks;
    stepTasks.reserve(plan.remainingPowerSteps);
    for (uint32_t step = 0; step < plan.remainingPowerSteps; ++step) {
        const uint32_t bit = plan.powerSteps - 1U - step;
        stepTasks.push_back(PowerStepTask {});
        HCCL_CHK_RET(BuildPowerStepTask(plan, bit, stepTasks.back()));
    }
    for (const PowerStepTask &stepTask : stepTasks) {
        HCCL_CHK_RET(RunPowerBit(stepTask, "power"));
    }
    return HCCL_SUCCESS;
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
    PowerStepTask stepTask {};
    HCCL_CHK_RET(BuildPowerStepTask(plan, 0U, stepTask));
    HCCL_CHK_RET(RunPowerBit(stepTask, "lastOne"));
    return HCCL_SUCCESS;
}

HcclResult AllGatherHDStageCore::BuildLastTwoTasks(const HDStagePlan &plan, std::vector<LastTwoCopyTask> &tasks) const
{
    if (plan.finalSteps != 2U) {
        HCCL_ERROR("HDStage lastTwo requires finalSteps=2, actual=%u", plan.finalSteps);
        return HCCL_E_INTERNAL;
    }

    const uint32_t subgroupSize = 4U;
    const uint32_t rankSize = layout_.rankSize;
    const uint32_t stageRankIdx = GetStageRankIndex(layout_, param_.topoInfo.rank);
    const uint32_t subgroupBase = (stageRankIdx / subgroupSize) * subgroupSize;
    const uint32_t localChunkPos = stageRankIdx % subgroupSize;
    std::vector<uint32_t> stageIdxToRank(rankSize, UINT32_MAX);
    for (uint32_t rank = 0; rank < rankSize; ++rank) {
        const uint32_t idx = GetStageRankIndex(layout_, rank);
        if (idx >= rankSize || stageIdxToRank[idx] != UINT32_MAX) {
            HCCL_ERROR("HDStage lastTwo stage rank mapping is invalid, rank=%u, stageRankIdx=%u", rank, idx);
            return HCCL_E_INTERNAL;
        }
        stageIdxToRank[idx] = rank;
    }

    tasks.clear();
    tasks.resize(subgroupSize);
    for (uint32_t chunkPos = 0; chunkPos < subgroupSize; ++chunkPos) {
        const uint32_t peerStageIdx = subgroupBase + chunkPos;
        if (peerStageIdx >= rankSize || stageIdxToRank[peerStageIdx] == UINT32_MAX) {
            HCCL_ERROR("HDStage lastTwo peer stage idx is invalid, subgroupBase=%u, chunkPos=%u, peerStageIdx=%u",
                subgroupBase,
                chunkPos,
                peerStageIdx);
            return HCCL_E_INTERNAL;
        }
        tasks[chunkPos].peerRank = stageIdxToRank[peerStageIdx];
        tasks[chunkPos].isLocal = (chunkPos == localChunkPos);
    }

    for (const WindowStageSlice &slice : layout_.perRankSlices) {
        const uint32_t targetStageIdx = GetStageRankIndex(layout_, slice.rank);
        if (targetStageIdx < subgroupBase || targetStageIdx >= (subgroupBase + subgroupSize)) {
            continue;
        }
        const uint32_t chunkPos = targetStageIdx - subgroupBase;
        StageCopySlice taskSlice;
        taskSlice.remoteRank = tasks[chunkPos].peerRank;
        taskSlice.localOffset = slice.stageOffsetBytes;
        taskSlice.remoteOffset = slice.stageOffsetBytes;
        taskSlice.size = slice.size;
        tasks[chunkPos].slices.push_back(taskSlice);
    }

    for (uint32_t chunkPos = 0; chunkPos < subgroupSize; ++chunkPos) {
        if (tasks[chunkPos].slices.empty()) {
            HCCL_ERROR("HDStage lastTwo task is empty, chunkPos=%u, peerRank=%u", chunkPos, tasks[chunkPos].peerRank);
            return HCCL_E_INTERNAL;
        }
    }
    return HCCL_SUCCESS;
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

HcclResult AllGatherHDStageCore::RunLastTwoWorkerTask(
    const LastTwoCopyTask &task,
    ThreadHandle workerThread,
    uint32_t workerNotifyIdx,
    uint32_t doneNotifyIdx,
    const char *stageTag) const
{
    if (task.isLocal) {
        return HCCL_SUCCESS;
    }
    if (workerThread == 0) {
        HCCL_ERROR("HDStage %s workerThread is invalid for peerRank=%u", stageTag, task.peerRank);
        return HCCL_E_INTERNAL;
    }

    const ChannelResource *channel = FindChannel(task.peerRank);
    if (channel == nullptr) {
        HCCL_ERROR("HDStage %s channel to remoteRank=%u is missing", stageTag, task.peerRank);
        return HCCL_E_NOT_FOUND;
    }

    int32_t ret = HcommThreadNotifyWaitOnThread(workerThread, workerNotifyIdx, kAllGatherBatchCustomTimeoutMs);
    if (ret != HCCL_SUCCESS) {
        HCCL_ERROR("HDStage %s worker start wait failed, peerRank=%u, ret=%d", stageTag, task.peerRank, ret);
        return static_cast<HcclResult>(ret);
    }
    ret = HcommChannelNotifyRecordOnThread(workerThread, channel->handle, channel->remoteNotifyIdx);
    if (ret != HCCL_SUCCESS) {
        HCCL_ERROR("HDStage %s worker notify record failed, peerRank=%u, ret=%d", stageTag, task.peerRank, ret);
        return static_cast<HcclResult>(ret);
    }
    ret = HcommChannelNotifyWaitOnThread(workerThread, channel->handle, channel->localNotifyIdx, kAllGatherBatchCustomTimeoutMs);
    if (ret != HCCL_SUCCESS) {
        HCCL_ERROR("HDStage %s worker notify wait failed, peerRank=%u, ret=%d", stageTag, task.peerRank, ret);
        return static_cast<HcclResult>(ret);
    }

    for (const StageCopySlice &slice : task.slices) {
        if ((slice.localOffset + slice.size) > resCtx_.localBuffer.size ||
            (slice.remoteOffset + slice.size) > channel->remoteBuffer.size) {
            HCCL_ERROR("HDStage %s worker slice is out of range, peerRank=%u, localOffset=%llu, remoteOffset=%llu, size=%llu, localBuffer=%llu, remoteBuffer=%llu",
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
        ret = HcommReadOnThread(workerThread, channel->handle, dst, src, slice.size);
        if (ret != HCCL_SUCCESS) {
            HCCL_ERROR("HDStage %s worker read failed, peerRank=%u, localOffset=%llu, remoteOffset=%llu, size=%llu, ret=%d",
                stageTag,
                task.peerRank,
                static_cast<unsigned long long>(slice.localOffset),
                static_cast<unsigned long long>(slice.remoteOffset),
                static_cast<unsigned long long>(slice.size),
                ret);
            return static_cast<HcclResult>(ret);
        }
    }

    ret = HcommThreadNotifyRecordOnThread(workerThread, resCtx_.mainThreadHandle, doneNotifyIdx);
    if (ret != HCCL_SUCCESS) {
        HCCL_ERROR("HDStage %s worker done notify failed, peerRank=%u, ret=%d", stageTag, task.peerRank, ret);
        return static_cast<HcclResult>(ret);
    }
    return HCCL_SUCCESS;
}

HcclResult AllGatherHDStageCore::WaitLastTwoWorkers(uint32_t workerCount, const char *stageTag) const
{
    for (uint32_t idx = 0; idx < workerCount; ++idx) {
        const int32_t ret = HcommThreadNotifyWaitOnThread(
            resCtx_.mainThreadHandle,
            resCtx_.lastTwoMainNotifyIds[idx],
            kAllGatherBatchCustomTimeoutMs);
        if (ret != HCCL_SUCCESS) {
            HCCL_ERROR("HDStage %s main wait worker[%u] failed, ret=%d", stageTag, idx, ret);
            return static_cast<HcclResult>(ret);
        }
    }
    return HCCL_SUCCESS;
}

HcclResult AllGatherHDStageCore::RunAllGatherLastTwoParallel(const HDStagePlan &plan) const
{
    if (resCtx_.lastTwoWorkerCount != kAllGatherBatchLastTwoWorkerCount) {
        HCCL_ERROR("HDStage lastTwo worker count mismatch, actual=%u, expected=%u",
            resCtx_.lastTwoWorkerCount,
            kAllGatherBatchLastTwoWorkerCount);
        return HCCL_E_INTERNAL;
    }

    std::vector<LastTwoCopyTask> tasks;
    HCCL_CHK_RET(BuildLastTwoTasks(plan, tasks));

    const LastTwoCopyTask *localTask = nullptr;
    uint32_t workerIdx = 0;
    for (const LastTwoCopyTask &task : tasks) {
        if (task.isLocal) {
            if (localTask != nullptr) {
                HCCL_ERROR("HDStage lastTwo duplicated local task, rank=%u", param_.topoInfo.rank);
                return HCCL_E_INTERNAL;
            }
            localTask = &task;
            continue;
        }
        if (workerIdx >= resCtx_.lastTwoWorkerCount) {
            HCCL_ERROR("HDStage lastTwo worker overflow, workerIdx=%u, workerCount=%u",
                workerIdx,
                resCtx_.lastTwoWorkerCount);
            return HCCL_E_INTERNAL;
        }
        HCCL_CHK_RET(RunLastTwoWorkerTask(
            task,
            resCtx_.lastTwoWorkerThreads[workerIdx],
            resCtx_.lastTwoWorkerNotifyIds[workerIdx],
            resCtx_.lastTwoMainNotifyIds[workerIdx],
            "lastTwo"));
        ++workerIdx;
    }
    if (localTask == nullptr) {
        HCCL_ERROR("HDStage lastTwo local task is missing, rank=%u", param_.topoInfo.rank);
        return HCCL_E_INTERNAL;
    }
    HCCL_CHK_RET(RunLastTwoLocalTask(*localTask, "lastTwo"));
    if (workerIdx != resCtx_.lastTwoWorkerCount) {
        HCCL_ERROR("HDStage lastTwo scheduled worker count mismatch, scheduled=%u, expected=%u",
            workerIdx,
            resCtx_.lastTwoWorkerCount);
        return HCCL_E_INTERNAL;
    }

    for (uint32_t idx = 0; idx < workerIdx; ++idx) {
        const int32_t ret = HcommThreadNotifyRecordOnThread(
            resCtx_.mainThreadHandle,
            resCtx_.lastTwoWorkerThreads[idx],
            resCtx_.lastTwoWorkerNotifyIds[idx]);
        if (ret != HCCL_SUCCESS) {
            HCCL_ERROR("HDStage lastTwo start worker[%u] failed, ret=%d", idx, ret);
            return static_cast<HcclResult>(ret);
        }
    }
    HCCL_CHK_RET(WaitLastTwoWorkers(workerIdx, "lastTwo"));
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
    return RunAllGatherLastTwoParallel(plan);
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



