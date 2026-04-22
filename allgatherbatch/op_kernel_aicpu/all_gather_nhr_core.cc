#include "all_gather_nhr_core.h"

#include <algorithm>

#include "log.h"
#include "stage_rank_mapping.h"

namespace ops_hccl_allgatherbatch {

AllGatherNHRCore::AllGatherNHRCore(AlgResourceCtx &resCtx,
    ExecMem &execMem,
    u64 baseOffset,
    u64 totalSize,
    const std::vector<ChannelResource> &channels)
    : resCtx_(resCtx), execMem_(execMem), baseOffset_(baseOffset), totalSize_(totalSize), channels_(channels)
{
}

HcclResult AllGatherNHRCore::Prepare(bool needMerge)
{
    isNeedMerge_ = needMerge;
    return HCCL_SUCCESS;
}

void AllGatherNHRCore::SetInputPreparedInOutput(bool enabled)
{
    inputPreparedInOutput_ = enabled;
}


HcclResult AllGatherNHRCore::RunAsync(const u32 rank, const u32 rankSize)
{
    HCCL_INFO("[AllGatherNHRCore][RunAsync] rank[%u] rankSize[%u] inputPtr[%p] outputMem[%p] count[%llu] baseOffset[%llu]",
        rank, rankSize, execMem_.inputPtr, execMem_.outputMem.addr, execMem_.count,
        static_cast<unsigned long long>(baseOffset_));

    CHK_PTR_NULL(execMem_.inputPtr);
    CHK_PTR_NULL(execMem_.outputMem.addr);
    CHK_PRT_RET(resCtx_.mainThreadHandle == 0,
        HCCL_ERROR("[AllGatherNHRCore][RunAsync] mainThreadHandle is invalid"),
        HCCL_E_PTR);
    CHK_PRT_RET(rank >= rankSize,
        HCCL_ERROR("[AllGatherNHRCore][RunAsync] rank[%u] is out of range rankSize[%u]", rank, rankSize),
        HCCL_E_PARA);

    CHK_PRT_RET(channels_.size() < rankSize,
        HCCL_ERROR("[AllGatherNHRCore][RunAsync] link size[%llu] is less than rankSize[%u]",
            static_cast<unsigned long long>(channels_.size()), rankSize),
        HCCL_E_PARA);

    if (isNeedMerge_) {
        GetRankMapping(rankSize, sliceMap_);
    }
    if (sliceMap_.size() != rankSize) {
        GetRankMapping(rankSize, sliceMap_, true);
    }

    std::vector<Slice> slices(rankSize);
    for (u32 i = 0; i < rankSize; ++i) {
        slices[i].offset = static_cast<u64>(i) * totalSize_;
        slices[i].size = totalSize_;
    }

    if (!inputPreparedInOutput_) {
        const Slice &localSlice = slices[sliceMap_[rank]];
        void *localDst = static_cast<u8 *>(execMem_.outputMem.addr) + baseOffset_ + localSlice.offset;
        if (execMem_.inputPtr != localDst) {
            CHK_RET(HcommLocalCopyOnThread(resCtx_.mainThreadHandle, localDst, execMem_.inputPtr, totalSize_));
        }
    }

    if (rankSize == 1) {
        return HCCL_SUCCESS;
    }

    return RunAllGather(rank, rankSize, slices, channels_);
}

HcclResult AllGatherNHRCore::RunAllGather(u32 rank, u32 rankSize,
    const std::vector<Slice> &outputSlices,
    const std::vector<ChannelResource> &links)
{
    const u32 nSteps = GetStepNumInterServer(rankSize);
    for (u32 step = 0; step < nSteps; ++step) {
        InterServerAlgoStep stepInfo;
        CHK_RET(GetStepInfo(step, nSteps, rank, rankSize, stepInfo));

        const ChannelResource &linkLeft = links[stepInfo.fromRank];
        const ChannelResource &linkRight = links[stepInfo.toRank];

        std::vector<Slice> txSlices;
        std::vector<Slice> rxSlices;
        txSlices.reserve(stepInfo.nSlices);
        rxSlices.reserve(stepInfo.nSlices);
        for (u32 idx = 0; idx < stepInfo.nSlices; ++idx) {
            txSlices.push_back(outputSlices[stepInfo.txSliceIdxs[idx]]);
            rxSlices.push_back(outputSlices[stepInfo.rxSliceIdxs[idx]]);
        }

        MergeSlices(txSlices);
        MergeSlices(rxSlices);

        const bool useSdma = (linkLeft.protocol != COMM_PROTOCOL_ROCE) && (linkRight.protocol != COMM_PROTOCOL_ROCE);
        if (useSdma) {
            CHK_RET(SdmaRx(linkLeft, linkRight, rxSlices));
        } else {
            CHK_RET(RdmaTxRx(linkLeft, linkRight, stepInfo, txSlices, rxSlices));
        }
    }
    return HCCL_SUCCESS;
}

HcclResult AllGatherNHRCore::SdmaRx(const ChannelResource &linkLeft,
    const ChannelResource &linkRight,
    std::vector<Slice> &rxSlices)
{
    CHK_RET(HcommChannelNotifyRecordOnThread(
        resCtx_.mainThreadHandle, linkRight.handle, NOTIFY_IDX_ACK));
    CHK_RET(HcommChannelNotifyWaitOnThread(
        resCtx_.mainThreadHandle, linkLeft.handle, NOTIFY_IDX_ACK, CUSTOM_TIMEOUT));

    const u8 *remoteBase = static_cast<const u8 *>(linkLeft.remoteBuffer.addr) +
        linkLeft.remoteBuffer.offset + baseOffset_;
    u8 *localBase = static_cast<u8 *>(execMem_.outputMem.addr) + baseOffset_;
    for (const Slice &rxSlice : rxSlices) {
        void *dst = localBase + rxSlice.offset;
        const void *src = remoteBase + rxSlice.offset;
        CHK_RET(HcommReadOnThread(resCtx_.mainThreadHandle, linkLeft.handle, dst, src, rxSlice.size));
    }

    CHK_RET(HcommChannelNotifyRecordOnThread(
        resCtx_.mainThreadHandle, linkLeft.handle, NOTIFY_IDX_DATA_SIGNAL));
    CHK_RET(HcommChannelNotifyWaitOnThread(
        resCtx_.mainThreadHandle, linkRight.handle, NOTIFY_IDX_DATA_SIGNAL, CUSTOM_TIMEOUT));
    return HCCL_SUCCESS;
}

HcclResult AllGatherNHRCore::RdmaTxRx(const ChannelResource &linkLeft,
    const ChannelResource &linkRight,
    InterServerAlgoStep &stepInfo,
    std::vector<Slice> &txSlices,
    std::vector<Slice> &rxSlices)
{
    (void)stepInfo;
    CHK_RET(HcommChannelNotifyRecordOnThread(
        resCtx_.mainThreadHandle, linkLeft.handle, NOTIFY_IDX_ACK));
    CHK_RET(HcommChannelNotifyWaitOnThread(
        resCtx_.mainThreadHandle, linkRight.handle, NOTIFY_IDX_ACK, CUSTOM_TIMEOUT));

    CHK_RET(Tx(linkRight, txSlices));
    CHK_RET(Rx(linkLeft, rxSlices));

    CHK_RET(HcommChannelNotifyRecordOnThread(
        resCtx_.mainThreadHandle, linkRight.handle, NOTIFY_IDX_DATA_SIGNAL));
    CHK_RET(HcommChannelNotifyWaitOnThread(
        resCtx_.mainThreadHandle, linkLeft.handle, NOTIFY_IDX_DATA_SIGNAL, CUSTOM_TIMEOUT));
    CHK_RET(HcommChannelNotifyRecordOnThread(
        resCtx_.mainThreadHandle, linkLeft.handle, NOTIFY_IDX_FIN_ACK));
    CHK_RET(HcommChannelNotifyWaitOnThread(
        resCtx_.mainThreadHandle, linkRight.handle, NOTIFY_IDX_FIN_ACK, CUSTOM_TIMEOUT));
    return HCCL_SUCCESS;
}

HcclResult AllGatherNHRCore::Tx(const ChannelResource &link, std::vector<Slice> &txSlices)
{
    const u8 *localBase = static_cast<const u8 *>(execMem_.outputMem.addr) + baseOffset_;
    u8 *remoteBase = static_cast<u8 *>(link.remoteBuffer.addr) + link.remoteBuffer.offset + baseOffset_;
    for (const Slice &txSlice : txSlices) {
        const void *src = localBase + txSlice.offset;
        void *dst = remoteBase + txSlice.offset;
        CHK_RET(HcommWriteOnThread(resCtx_.mainThreadHandle, link.handle, dst, src, txSlice.size));
    }
    return HCCL_SUCCESS;
}

HcclResult AllGatherNHRCore::Rx(const ChannelResource &link, std::vector<Slice> &rxSlices)
{
    const u8 *remoteBase = static_cast<const u8 *>(link.remoteBuffer.addr) + link.remoteBuffer.offset + baseOffset_;
    u8 *localBase = static_cast<u8 *>(execMem_.outputMem.addr) + baseOffset_;
    for (const Slice &rxSlice : rxSlices) {
        const void *src = remoteBase + rxSlice.offset;
        void *dst = localBase + rxSlice.offset;
        CHK_RET(HcommReadOnThread(resCtx_.mainThreadHandle, link.handle, dst, src, rxSlice.size));
    }
    return HCCL_SUCCESS;
}

HcclResult AllGatherNHRCore::GetStepInfo(u32 step, u32 nSteps, u32 rank, u32 rankSize,
    InterServerAlgoStep &stepInfo)
{
    stepInfo.txSliceIdxs.clear();
    stepInfo.rxSliceIdxs.clear();
    stepInfo.step = step;
    stepInfo.myRank = rank;

    const u32 deltaRank = 1U << (nSteps - 1U - step);
    const u32 recvFrom = (rank + rankSize - deltaRank) % rankSize;
    const u32 sendTo = (rank + deltaRank) % rankSize;
    const u32 nSlices = (rankSize - 1U + (1U << (nSteps - 1U - step))) / (1U << (nSteps - step));
    const u32 deltaSliceIndex = 1U << (nSteps - step);
    u32 txSliceIdx = rank;
    u32 rxSliceIdx = (rank + rankSize - (1U << (nSteps - 1U - step))) % rankSize;

    stepInfo.nSlices = nSlices;
    stepInfo.toRank = sendTo;
    stepInfo.fromRank = recvFrom;

    stepInfo.txSliceIdxs.reserve(stepInfo.nSlices);
    stepInfo.rxSliceIdxs.reserve(stepInfo.nSlices);
    for (u32 i = 0; i < nSlices; ++i) {
        stepInfo.txSliceIdxs.push_back(sliceMap_[txSliceIdx]);
        stepInfo.rxSliceIdxs.push_back(sliceMap_[rxSliceIdx]);
        txSliceIdx = (txSliceIdx + rankSize - deltaSliceIndex) % rankSize;
        rxSliceIdx = (rxSliceIdx + rankSize - deltaSliceIndex) % rankSize;
    }
    return HCCL_SUCCESS;
}



void AllGatherNHRCore::MergeSlices(std::vector<Slice> &slices)
{
    if (!isNeedMerge_ || slices.size() <= 1U) {
        return;
    }

    std::sort(slices.begin(), slices.end(), [](const Slice &lhs, const Slice &rhs) {
        return lhs.offset < rhs.offset;
    });

    std::vector<Slice> merged;
    merged.reserve(slices.size());
    Slice current = slices[0];
    for (size_t idx = 1; idx < slices.size(); ++idx) {
        const Slice &next = slices[idx];
        if (current.offset + current.size == next.offset) {
            current.size += next.size;
            continue;
        }
        merged.push_back(current);
        current = next;
    }
    merged.push_back(current);
    slices.swap(merged);
}

}  // namespace ops_hccl_allgatherbatch

