#include "all_gather_nhr_core.h"

#include <algorithm>

#include "log.h"

namespace ops_hccl_allgatherbatch {

namespace {

bool IsRoceProtocol(CommProtocol protocol)
{
    return protocol == COMM_PROTOCOL_ROCE;
}

bool ShouldUseSdmaPath(CommProtocol leftProtocol, CommProtocol rightProtocol)
{
    return !IsRoceProtocol(leftProtocol) && !IsRoceProtocol(rightProtocol);
}

}  // namespace

AllGatherNHRCore::AllGatherNHRCore(
    const OpParam &param,
    AlgResourceCtx &resCtx,
    uint64_t packedBytes,
    const NHRRunCtx &runCtx)
    : param_(param), resCtx_(resCtx), packedBytes_(packedBytes), runCtx_(runCtx)
{
    InitDefaultRunCtx();
}

void AllGatherNHRCore::BuildDefaultSlices()
{
    runCtx_.sliceTemplate.clear();
    runCtx_.sliceTemplate.push_back(LocalSlice { 0, runCtx_.packedBytes });
    runCtx_.rankBaseOffsets.clear();
    runCtx_.rankBaseOffsets.reserve(runCtx_.rankSize);
    for (uint32_t rank = 0; rank < runCtx_.rankSize; ++rank) {
        runCtx_.rankBaseOffsets.push_back(static_cast<uint64_t>(rank) * runCtx_.packedBytes);
    }
}

void AllGatherNHRCore::BuildEffectiveSlices()
{
    effectiveSlices_.clear();
    effectiveSlices_.reserve(runCtx_.rankBaseOffsets.size() * runCtx_.sliceTemplate.size());
    for (size_t rankIdx = 0; rankIdx < runCtx_.rankBaseOffsets.size(); ++rankIdx) {
        const uint64_t rankBaseOffset = runCtx_.rankBaseOffsets[rankIdx];
        for (const LocalSlice &slice : runCtx_.sliceTemplate) {
            effectiveSlices_.push_back(LocalSlice { rankBaseOffset + slice.offset, slice.size });
        }
    }
}



void AllGatherNHRCore::InitDefaultRunCtx()
{
    if (runCtx_.packedBytes == 0) {
        runCtx_.packedBytes = packedBytes_;
    }
    if (runCtx_.rankSize == 0) {
        runCtx_.rank = param_.topoInfo.rank;
        runCtx_.rankSize = param_.topoInfo.rankSize;
    }
    if (runCtx_.inputBase == nullptr) {
        runCtx_.inputBase = static_cast<uint8_t *>(resCtx_.localBuffer.addr);
    }
    if (runCtx_.outputBase == nullptr) {
        runCtx_.outputBase = static_cast<uint8_t *>(resCtx_.localBuffer.addr);
    }
    if (runCtx_.needMerge) {
        runCtx_.keepOrder = false;
    }
    if (runCtx_.subgroupRanks.empty()) {
        runCtx_.subgroupRanks.reserve(runCtx_.rankSize);
        for (uint32_t rank = 0; rank < runCtx_.rankSize; ++rank) {
            runCtx_.subgroupRanks.push_back(rank);
        }
    }
    if (runCtx_.sliceTemplate.empty() && runCtx_.rankBaseOffsets.empty()) {
        BuildDefaultSlices();
    }
    BuildEffectiveSlices();
}

uint32_t AllGatherNHRCore::GetEffectiveRank() const
{
    return runCtx_.rank;
}

uint32_t AllGatherNHRCore::GetEffectiveRankSize() const
{
    return runCtx_.rankSize;
}

uint32_t AllGatherNHRCore::GetSliceGroupSize() const
{
    return static_cast<uint32_t>(runCtx_.sliceTemplate.size());
}

const std::vector<LocalSlice> &AllGatherNHRCore::GetEffectiveSlices() const
{
    return effectiveSlices_;
}

HcclResult AllGatherNHRCore::ValidateCommState() const
{
    if (param_.topoInfo.rankSize == 0) {
        HCCL_ERROR("NHR rankSize is zero");
        return HCCL_E_PARA;
    }
    if (packedBytes_ == 0) {
        HCCL_ERROR("NHR packedBytes is zero");
        return HCCL_E_PARA;
    }
    if (runCtx_.rankSize == 0 || runCtx_.rank >= runCtx_.rankSize) {
        HCCL_ERROR("NHR subgroup is invalid, rank=%u, rankSize=%u", runCtx_.rank, runCtx_.rankSize);
        return HCCL_E_INTERNAL;
    }
    if (runCtx_.subgroupRanks.size() != runCtx_.rankSize) {
        HCCL_ERROR("NHR subgroup rank list mismatch, expected=%u, actual=%u",
            runCtx_.rankSize,
            static_cast<uint32_t>(runCtx_.subgroupRanks.size()));
        return HCCL_E_INTERNAL;
    }
    if (runCtx_.inputBase == nullptr || runCtx_.outputBase == nullptr) {
        HCCL_ERROR("NHR input/output base is null, input=%p, output=%p", runCtx_.inputBase, runCtx_.outputBase);
        return HCCL_E_INTERNAL;
    }
    if (runCtx_.sliceTemplate.empty() || runCtx_.rankBaseOffsets.size() != runCtx_.rankSize) {
        HCCL_ERROR("NHR formal layout is incomplete, sliceTemplate=%u, rankBaseOffsets=%u, rankSize=%u",
            static_cast<uint32_t>(runCtx_.sliceTemplate.size()),
            static_cast<uint32_t>(runCtx_.rankBaseOffsets.size()),
            runCtx_.rankSize);
        return HCCL_E_INTERNAL;
    }
    if (runCtx_.rankBaseOffsets.size() != runCtx_.subgroupRanks.size()) {
        HCCL_ERROR("NHR rankBaseOffsets mismatch, baseOffsets=%u, subgroupRanks=%u",
            static_cast<uint32_t>(runCtx_.rankBaseOffsets.size()),
            static_cast<uint32_t>(runCtx_.subgroupRanks.size()));
        return HCCL_E_INTERNAL;
    }

    const uint32_t sliceGroupSize = GetSliceGroupSize();
    if (sliceGroupSize == 0U) {
        HCCL_ERROR("NHR effective slice group size is zero");
        return HCCL_E_INTERNAL;
    }
    if (!runCtx_.sliceTemplate.empty() && runCtx_.sliceTemplate.size() != sliceGroupSize) {
        HCCL_ERROR("NHR sliceTemplate size mismatch, template=%u, sliceGroupSize=%u",
            static_cast<uint32_t>(runCtx_.sliceTemplate.size()),
            sliceGroupSize);
        return HCCL_E_INTERNAL;
    }
    const std::vector<LocalSlice> &effectiveSlices = GetEffectiveSlices();
    if (effectiveSlices.empty() || effectiveSlices.size() != (runCtx_.rankSize * sliceGroupSize)) {
        HCCL_ERROR("NHR effective slices are invalid, sliceCount=%u, rankSize=%u, sliceGroupSize=%u",
            static_cast<uint32_t>(effectiveSlices.size()),
            runCtx_.rankSize,
            sliceGroupSize);
        return HCCL_E_INTERNAL;
    }

    for (size_t idx = 0; idx < runCtx_.subgroupRanks.size(); ++idx) {
        if (runCtx_.subgroupRanks[idx] >= param_.topoInfo.rankSize) {
            HCCL_ERROR("NHR subgroup rank is out of range, subgroupIdx=%u, globalRank=%u, rankSize=%u",
                static_cast<uint32_t>(idx),
                runCtx_.subgroupRanks[idx],
                param_.topoInfo.rankSize);
            return HCCL_E_INTERNAL;
        }
    }
    for (const LocalSlice &slice : effectiveSlices) {
        if (slice.size == 0) {
            HCCL_ERROR("NHR slice size is zero, offset=%llu", static_cast<unsigned long long>(slice.offset));
            return HCCL_E_INTERNAL;
        }
        if ((runCtx_.baseOffset + slice.offset + slice.size) > resCtx_.localBuffer.size) {
            HCCL_ERROR("NHR slice exceeds local buffer, baseOffset=%llu, sliceOffset=%llu, sliceSize=%llu, localBuffer=%llu",
                static_cast<unsigned long long>(runCtx_.baseOffset),
                static_cast<unsigned long long>(slice.offset),
                static_cast<unsigned long long>(slice.size),
                static_cast<unsigned long long>(resCtx_.localBuffer.size));
            return HCCL_E_INTERNAL;
        }
    }
    return HCCL_SUCCESS;
}
HcclResult AllGatherNHRCore::LocalDataCopy()
{
    if (sliceMap_.size() != runCtx_.rankSize) {
        HCCL_ERROR("NHR sliceMap size mismatch, sliceMap=%u, rankSize=%u",
            static_cast<uint32_t>(sliceMap_.size()),
            runCtx_.rankSize);
        return HCCL_E_INTERNAL;
    }
    if (runCtx_.sliceTemplate.empty() || runCtx_.rankBaseOffsets.size() != runCtx_.subgroupRanks.size() ||
        runCtx_.rank >= runCtx_.rankBaseOffsets.size()) {
        HCCL_ERROR("NHR local copy layout is incomplete, rank=%u, template=%u, baseOffsets=%u, subgroupRanks=%u",
            runCtx_.rank,
            static_cast<uint32_t>(runCtx_.sliceTemplate.size()),
            static_cast<uint32_t>(runCtx_.rankBaseOffsets.size()),
            static_cast<uint32_t>(runCtx_.subgroupRanks.size()));
        return HCCL_E_INTERNAL;
    }

    const uint32_t sourceRankIdx = runCtx_.rank;
    const uint32_t targetRankIdx = sliceMap_[runCtx_.rank];
    if (targetRankIdx >= runCtx_.rankBaseOffsets.size()) {
        HCCL_ERROR("NHR local copy target idx is out of range, source=%u, target=%u, baseOffsets=%u",
            sourceRankIdx,
            targetRankIdx,
            static_cast<uint32_t>(runCtx_.rankBaseOffsets.size()));
        return HCCL_E_INTERNAL;
    }

    if (runCtx_.inputBase == runCtx_.outputBase) {
        // Align with hcomm noPower main path: PreCopy has already placed the local rank data
        // into the noPowerMap[group] workspace slot, so in-place input/output needs no extra local move.
        return HCCL_SUCCESS;
    }


    const uint64_t targetBaseOffset = runCtx_.rankBaseOffsets[targetRankIdx];
    for (const LocalSlice &slice : runCtx_.sliceTemplate) {
        void *dst = static_cast<uint8_t *>(runCtx_.outputBase) + targetBaseOffset + slice.offset;
        const void *src = static_cast<const uint8_t *>(runCtx_.inputBase) + slice.offset;
        const int32_t ret = HcommLocalCopyOnThread(resCtx_.mainThreadHandle, dst, src, slice.size);
        if (ret != HCCL_SUCCESS) {
            HCCL_ERROR("NHR local copy failed, rank=%u, targetBase=%llu, offset=%llu, size=%llu, ret=%d",
                runCtx_.rank,
                static_cast<unsigned long long>(targetBaseOffset),
                static_cast<unsigned long long>(slice.offset),
                static_cast<unsigned long long>(slice.size),
                ret);
            return static_cast<HcclResult>(ret);
        }
    }
    return HCCL_SUCCESS;
}

HcclResult AllGatherNHRCore::PostLocalCopy() const
{
    // 当前 custom-op 的 NHR 输出缓冲区就是后续 HDStage 继续消费的正式工作区，
    // 对应 hcomm 指令版的 post-copy 语义，这一轮不需要额外把 scratch 写回到别处。
    return HCCL_SUCCESS;
}
void AllGatherNHRCore::GetRankMapping(uint32_t rankSize, bool keepOrder)
{
    std::vector<uint32_t> tree;
    tree.reserve(rankSize);
    for (uint32_t rank = 0; rank < rankSize; ++rank) {
        tree.push_back(rank);
    }
    if (keepOrder) {
        sliceMap_ = tree;
        return;
    }

    std::vector<uint32_t> tmp(rankSize);
    uint32_t nSteps = GetStepNumInterServer(rankSize);
    uint32_t len = rankSize;
    for (uint32_t step = 0; step < nSteps; ++step) {
        uint32_t nSlices = (rankSize - 1U + (1U << step)) / (1U << (step + 1U));
        if (nSlices <= 1U) {
            break;
        }
        bool endFlag = false;
        for (uint32_t part = 0; part * len < rankSize; ++part) {
            const uint32_t start = part * len;
            const uint32_t end = std::min(start + len, rankSize);
            ReorderSequence(start, end, len, tree, tmp);
            if (((end - start) & 1U) == 1U) {
                endFlag = true;
            }
        }
        tree = tmp;
        if (endFlag) {
            break;
        }
        len >>= 1U;
    }

    sliceMap_.assign(rankSize, 0U);
    for (uint32_t idx = 0; idx < rankSize; ++idx) {
        sliceMap_[tree[idx]] = idx;
    }
}

void AllGatherNHRCore::ReorderSequence(
    uint32_t start,
    uint32_t end,
    uint32_t len,
    std::vector<uint32_t> &tree,
    std::vector<uint32_t> &tmp) const
{
    for (uint32_t idx = start; idx < end; ++idx) {
        const uint32_t offset = idx - start;
        if ((offset & 1U) == 0U) {
            tmp[start + offset / 2U] = tree[idx];
        } else {
            tmp[start + (offset + len) / 2U] = tree[idx];
        }
    }
}

uint32_t AllGatherNHRCore::GetStepNumInterServer(uint32_t rankSize) const
{
    uint32_t nSteps = 0;
    for (uint32_t tmp = rankSize - 1U; tmp != 0U; tmp >>= 1U) {
        ++nSteps;
    }
    return nSteps;
}

HcclResult AllGatherNHRCore::GetStepInfo(uint32_t step, uint32_t nSteps, InterServerAlgoStep &stepInfo) const
{
    const uint32_t rank = GetEffectiveRank();
    const uint32_t rankSize = GetEffectiveRankSize();
    const uint32_t sliceGroupSize = GetSliceGroupSize();

    stepInfo.step = step;
    stepInfo.myRank = rank;
    stepInfo.txSliceIdxs.clear();
    stepInfo.rxSliceIdxs.clear();

    const uint32_t deltaRank = 1U << (nSteps - 1U - step);
    const uint32_t recvFrom = (rank + rankSize - deltaRank) % rankSize;
    const uint32_t sendTo = (rank + deltaRank) % rankSize;
    const uint32_t nSlices = (rankSize - 1U + (1U << (nSteps - 1U - step))) / (1U << (nSteps - step));
    const uint32_t deltaSliceIndex = 1U << (nSteps - step);
    uint32_t txSliceIdx = rank;
    uint32_t rxSliceIdx = (rank + rankSize - deltaRank) % rankSize;

    stepInfo.nSlices = nSlices * sliceGroupSize;
    stepInfo.toRank = sendTo;
    stepInfo.fromRank = recvFrom;
    stepInfo.txSliceIdxs.reserve(stepInfo.nSlices);
    stepInfo.rxSliceIdxs.reserve(stepInfo.nSlices);

    for (uint32_t idx = 0; idx < nSlices; ++idx) {
        for (uint32_t subIdx = 0; subIdx < sliceGroupSize; ++subIdx) {
            const uint32_t targetTxSliceIdx = sliceMap_[txSliceIdx];
            const uint32_t targetRxSliceIdx = sliceMap_[rxSliceIdx];
            stepInfo.txSliceIdxs.push_back(targetTxSliceIdx * sliceGroupSize + subIdx);
            stepInfo.rxSliceIdxs.push_back(targetRxSliceIdx * sliceGroupSize + subIdx);
        }
        txSliceIdx = (txSliceIdx + rankSize - deltaSliceIndex) % rankSize;
        rxSliceIdx = (rxSliceIdx + rankSize - deltaSliceIndex) % rankSize;
    }
    return HCCL_SUCCESS;
}

HcclResult AllGatherNHRCore::BuildStepSlices(
    const InterServerAlgoStep &stepInfo,
    std::vector<LocalSlice> &txSlices,
    std::vector<LocalSlice> &rxSlices) const
{
    const std::vector<LocalSlice> &effectiveSlices = GetEffectiveSlices();
    txSlices.clear();
    rxSlices.clear();
    txSlices.reserve(stepInfo.txSliceIdxs.size());
    rxSlices.reserve(stepInfo.rxSliceIdxs.size());
    for (uint32_t sliceIdx : stepInfo.txSliceIdxs) {
        if (sliceIdx >= effectiveSlices.size()) {
            HCCL_ERROR("NHR tx slice index is out of range, sliceIdx=%u, sliceCount=%u",
                sliceIdx,
                static_cast<uint32_t>(effectiveSlices.size()));
            return HCCL_E_INTERNAL;
        }
        txSlices.push_back(effectiveSlices[sliceIdx]);
    }
    for (uint32_t sliceIdx : stepInfo.rxSliceIdxs) {
        if (sliceIdx >= effectiveSlices.size()) {
            HCCL_ERROR("NHR rx slice index is out of range, sliceIdx=%u, sliceCount=%u",
                sliceIdx,
                static_cast<uint32_t>(effectiveSlices.size()));
            return HCCL_E_INTERNAL;
        }
        rxSlices.push_back(effectiveSlices[sliceIdx]);
    }
    MergeSlices(txSlices);
    MergeSlices(rxSlices);
    return HCCL_SUCCESS;
}

void AllGatherNHRCore::MergeSlices(std::vector<LocalSlice> &slices) const
{
    if (!runCtx_.needMerge || slices.size() <= 1U) {
        return;
    }
    std::sort(slices.begin(), slices.end(), [](const LocalSlice &lhs, const LocalSlice &rhs) {
        return lhs.offset < rhs.offset;
    });
    std::vector<LocalSlice> merged;
    merged.reserve(slices.size());
    merged.push_back(slices.front());
    for (size_t idx = 1; idx < slices.size(); ++idx) {
        LocalSlice &tail = merged.back();
        const LocalSlice &current = slices[idx];
        if ((tail.offset + tail.size) == current.offset) {
            tail.size += current.size;
            continue;
        }
        merged.push_back(current);
    }
    slices.swap(merged);
}

const ChannelResource *AllGatherNHRCore::FindChannelByGlobalRank(uint32_t remoteRank) const
{
    for (uint32_t idx = 0; idx < resCtx_.channelCount; ++idx) {
        const ChannelResource &channel = GetChannel(resCtx_, idx);
        if (channel.remoteRank == remoteRank) {
            return &channel;
        }
    }
    return nullptr;
}

const ChannelResource *AllGatherNHRCore::FindChannelBySubgroupRank(uint32_t subgroupRank) const
{
    if (subgroupRank >= runCtx_.subgroupRanks.size()) {
        return nullptr;
    }
    const uint32_t remoteGlobalRank = runCtx_.subgroupRanks[subgroupRank];
    if (remoteGlobalRank == param_.topoInfo.rank) {
        return nullptr;
    }
    return FindChannelByGlobalRank(remoteGlobalRank);
}

HcclResult AllGatherNHRCore::Tx(const ChannelResource &channel, const std::vector<LocalSlice> &txSlices) const
{
    for (const LocalSlice &slice : txSlices) {
        void *dst = static_cast<uint8_t *>(channel.remoteBuffer.addr) + runCtx_.baseOffset + slice.offset;
        void *src = static_cast<uint8_t *>(runCtx_.outputBase) + slice.offset;
        const int32_t ret = HcommWriteOnThread(resCtx_.mainThreadHandle, channel.handle, dst, src, slice.size);
        if (ret != HCCL_SUCCESS) {
            HCCL_ERROR("NHR tx failed, remoteRank=%u, offset=%llu, size=%llu, ret=%d",
                channel.remoteRank,
                static_cast<unsigned long long>(slice.offset),
                static_cast<unsigned long long>(slice.size),
                ret);
            return static_cast<HcclResult>(ret);
        }
    }
    return HCCL_SUCCESS;
}

HcclResult AllGatherNHRCore::Rx(const ChannelResource &channel, const std::vector<LocalSlice> &rxSlices) const
{
    for (const LocalSlice &slice : rxSlices) {
        void *dst = static_cast<uint8_t *>(runCtx_.outputBase) + slice.offset;
        const void *src = static_cast<const uint8_t *>(channel.remoteBuffer.addr) + runCtx_.baseOffset + slice.offset;
        const int32_t ret = HcommReadOnThread(resCtx_.mainThreadHandle, channel.handle, dst, src, slice.size);
        if (ret != HCCL_SUCCESS) {
            HCCL_ERROR("NHR rx failed, remoteRank=%u, offset=%llu, size=%llu, ret=%d",
                channel.remoteRank,
                static_cast<unsigned long long>(slice.offset),
                static_cast<unsigned long long>(slice.size),
                ret);
            return static_cast<HcclResult>(ret);
        }
    }
    return HCCL_SUCCESS;
}

HcclResult AllGatherNHRCore::SdmaRx(
    const ChannelResource *channelLeft,
    const ChannelResource *channelRight,
    const std::vector<LocalSlice> &rxSlices,
    const InterServerAlgoStep &stepInfo) const
{
    if (channelRight != nullptr) {
        const int32_t ret = HcommChannelNotifyRecordOnThread(
            resCtx_.mainThreadHandle,
            channelRight->handle,
            kAllGatherBatchNotifyIdxAck);
        if (ret != HCCL_SUCCESS) {
            HCCL_ERROR("NHR step[%u] ack record failed, toRank=%u, ret=%d",
                stepInfo.step,
                channelRight->remoteRank,
                ret);
            return static_cast<HcclResult>(ret);
        }
    }
    if (channelLeft != nullptr) {
        int32_t ret = HcommChannelNotifyWaitOnThread(
            resCtx_.mainThreadHandle,
            channelLeft->handle,
            kAllGatherBatchNotifyIdxAck,
            CUSTOM_TIMEOUT);
        if (ret != HCCL_SUCCESS) {
            HCCL_ERROR("NHR step[%u] ack wait failed, fromRank=%u, ret=%d",
                stepInfo.step,
                channelLeft->remoteRank,
                ret);
            return static_cast<HcclResult>(ret);
        }
        HCCL_CHK_RET(Rx(*channelLeft, rxSlices));
        ret = HcommChannelNotifyRecordOnThread(
            resCtx_.mainThreadHandle,
            channelLeft->handle,
            kAllGatherBatchNotifyIdxDataSignal);
        if (ret != HCCL_SUCCESS) {
            HCCL_ERROR("NHR step[%u] data record failed, fromRank=%u, ret=%d",
                stepInfo.step,
                channelLeft->remoteRank,
                ret);
            return static_cast<HcclResult>(ret);
        }
    }
    if (channelRight != nullptr) {
        const int32_t ret = HcommChannelNotifyWaitOnThread(
            resCtx_.mainThreadHandle,
            channelRight->handle,
            kAllGatherBatchNotifyIdxDataSignal,
            CUSTOM_TIMEOUT);
        if (ret != HCCL_SUCCESS) {
            HCCL_ERROR("NHR step[%u] data wait failed, toRank=%u, ret=%d",
                stepInfo.step,
                channelRight->remoteRank,
                ret);
            return static_cast<HcclResult>(ret);
        }
    }
    return HCCL_SUCCESS;
}

HcclResult AllGatherNHRCore::RdmaTxRx(
    const ChannelResource *channelLeft,
    const ChannelResource *channelRight,
    const std::vector<LocalSlice> &txSlices,
    const std::vector<LocalSlice> &rxSlices,
    const InterServerAlgoStep &stepInfo) const
{
    if (channelLeft != nullptr) {
        const int32_t ret = HcommChannelNotifyRecordOnThread(
            resCtx_.mainThreadHandle,
            channelLeft->handle,
            kAllGatherBatchNotifyIdxAck);
        if (ret != HCCL_SUCCESS) {
            HCCL_ERROR("NHR step[%u] left ack record failed, fromRank=%u, ret=%d",
                stepInfo.step,
                channelLeft->remoteRank,
                ret);
            return static_cast<HcclResult>(ret);
        }
    }
    if (channelRight != nullptr) {
        int32_t ret = HcommChannelNotifyWaitOnThread(
            resCtx_.mainThreadHandle,
            channelRight->handle,
            kAllGatherBatchNotifyIdxAck,
            CUSTOM_TIMEOUT);
        if (ret != HCCL_SUCCESS) {
            HCCL_ERROR("NHR step[%u] right ack wait failed, toRank=%u, ret=%d",
                stepInfo.step,
                channelRight->remoteRank,
                ret);
            return static_cast<HcclResult>(ret);
        }
        HCCL_CHK_RET(Tx(*channelRight, txSlices));
        ret = HcommChannelNotifyRecordOnThread(
            resCtx_.mainThreadHandle,
            channelRight->handle,
            kAllGatherBatchNotifyIdxDataSignal);
        if (ret != HCCL_SUCCESS) {
            HCCL_ERROR("NHR step[%u] right data record failed, toRank=%u, ret=%d",
                stepInfo.step,
                channelRight->remoteRank,
                ret);
            return static_cast<HcclResult>(ret);
        }
    }
    if (channelLeft != nullptr) {
        int32_t ret = HcommChannelNotifyWaitOnThread(
            resCtx_.mainThreadHandle,
            channelLeft->handle,
            kAllGatherBatchNotifyIdxDataSignal,
            CUSTOM_TIMEOUT);
        if (ret != HCCL_SUCCESS) {
            HCCL_ERROR("NHR step[%u] left data wait failed, fromRank=%u, ret=%d",
                stepInfo.step,
                channelLeft->remoteRank,
                ret);
            return static_cast<HcclResult>(ret);
        }
        HCCL_CHK_RET(Rx(*channelLeft, rxSlices));
        ret = HcommChannelNotifyRecordOnThread(
            resCtx_.mainThreadHandle,
            channelLeft->handle,
            kAllGatherBatchNotifyIdxFinAck);
        if (ret != HCCL_SUCCESS) {
            HCCL_ERROR("NHR step[%u] left fin record failed, fromRank=%u, ret=%d",
                stepInfo.step,
                channelLeft->remoteRank,
                ret);
            return static_cast<HcclResult>(ret);
        }
    }
    if (channelRight != nullptr) {
        const int32_t ret = HcommChannelNotifyWaitOnThread(
            resCtx_.mainThreadHandle,
            channelRight->handle,
            kAllGatherBatchNotifyIdxFinAck,
            CUSTOM_TIMEOUT);
        if (ret != HCCL_SUCCESS) {
            HCCL_ERROR("NHR step[%u] right fin wait failed, toRank=%u, ret=%d",
                stepInfo.step,
                channelRight->remoteRank,
                ret);
            return static_cast<HcclResult>(ret);
        }
    }
    return HCCL_SUCCESS;
}

HcclResult AllGatherNHRCore::RunAllGather() const
{
    const uint32_t rankSize = GetEffectiveRankSize();
    const uint32_t nSteps = GetStepNumInterServer(rankSize);
    std::vector<LocalSlice> txSlices;
    std::vector<LocalSlice> rxSlices;
    for (uint32_t step = 0; step < nSteps; ++step) {
        InterServerAlgoStep stepInfo;
        HCCL_CHK_RET(GetStepInfo(step, nSteps, stepInfo));

        const ChannelResource *channelLeft = FindChannelBySubgroupRank(stepInfo.fromRank);
        const ChannelResource *channelRight = FindChannelBySubgroupRank(stepInfo.toRank);
        if (channelLeft == nullptr || channelRight == nullptr) {
            HCCL_ERROR("NHR step[%u] channel is missing, from=%u, to=%u",
                stepInfo.step,
                stepInfo.fromRank,
                stepInfo.toRank);
            return HCCL_E_NOT_FOUND;
        }

        HCCL_CHK_RET(BuildStepSlices(stepInfo, txSlices, rxSlices));

        HCCL_INFO("NHR step ready: subgroupRank=%u, subgroupSize=%u, step=%u, fromRank=%u(global=%u), toRank=%u(global=%u), mergedTxSlices=%u, mergedRxSlices=%u, protocolLeft=%s, protocolRight=%s",
            GetEffectiveRank(),
            GetEffectiveRankSize(),
            stepInfo.step,
            stepInfo.fromRank,
            runCtx_.subgroupRanks[stepInfo.fromRank],
            stepInfo.toRank,
            runCtx_.subgroupRanks[stepInfo.toRank],
            static_cast<uint32_t>(txSlices.size()),
            static_cast<uint32_t>(rxSlices.size()),
            ToProtocolString(channelLeft->protocol),
            ToProtocolString(channelRight->protocol));

        if (ShouldUseSdmaPath(channelLeft->protocol, channelRight->protocol)) {
            HCCL_CHK_RET(SdmaRx(channelLeft, channelRight, rxSlices, stepInfo));
        } else {
            HCCL_CHK_RET(RdmaTxRx(channelLeft, channelRight, txSlices, rxSlices, stepInfo));
        }
    }
    return HCCL_SUCCESS;
}

HcclResult AllGatherNHRCore::RunAsync()
{
    HCCL_CHK_RET(ValidateCommState());

    const uint32_t rankSize = GetEffectiveRankSize();
    const bool keepOrder = runCtx_.needMerge ? false : runCtx_.keepOrder;
    GetRankMapping(rankSize, keepOrder);

    HCCL_CHK_RET(LocalDataCopy());

    if (rankSize <= 1U) {
        HCCL_INFO("NHR fast path: rankSize=%u", rankSize);
        return PostLocalCopy();
    }

    const uint32_t stepCount = GetStepNumInterServer(rankSize);
    HCCL_INFO("NHR plan ready: rank=%u, subgroupRank=%u, subgroupSize=%u, stepCount=%u, sliceCount=%u, sliceGroupSize=%u, packedBytes=%llu, needMerge=%s, keepOrder=%s, localCopyMode=%s",
        param_.topoInfo.rank,
        GetEffectiveRank(),
        rankSize,
        stepCount,
        static_cast<uint32_t>(GetEffectiveSlices().size()),
        GetSliceGroupSize(),
        static_cast<unsigned long long>(packedBytes_),
        runCtx_.needMerge ? "true" : "false",
        keepOrder ? "true" : "false",
        (runCtx_.inputBase == runCtx_.outputBase) ? "inplace" : "out-of-place");

    HCCL_CHK_RET(RunAllGather());
    HCCL_CHK_RET(PostLocalCopy());
    HCCL_INFO("NHR finished: rank=%u, subgroupRank=%u, subgroupSize=%u",
        param_.topoInfo.rank,
        GetEffectiveRank(),
        rankSize);
    return HCCL_SUCCESS;
}

}  // namespace ops_hccl_allgatherbatch
