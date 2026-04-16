#include "allgather_batch_small_count_executor.h"

#include <algorithm>

#include "all_gather_hd_stage_core.h"
#include "log.h"

namespace ops_hccl_allgatherbatch {

AllGatherBatchSmallCountExecutor::AllGatherBatchSmallCountExecutor(
    const OpParam &param, AlgResourceCtx &resCtx, BatchCallProfiling &profiling)
    : param_(param), resCtx_(resCtx), profiling_(profiling)
{
}

HcclResult AllGatherBatchSmallCountExecutor::Orchestrate()
{
    const uint64_t startus = GetCurrentTimeUs();

    std::vector<ChannelResource> channels_(param_.topoInfo.rankSize);
    for (uint32_t idx = 0; idx < resCtx_.channelCount; ++idx) {
        ChannelResource &channel = GetChannel(resCtx_, idx);
        channels_[channel.remoteRank] = channel;
    }
    CHK_RET(RunLoop(channels_));

    HCCL_INFO("tag[%s], Allgather executor orchestrate success, take time [%llu]us",
        param_.tag, GetCurrentTimeUs() - startus);
    return HCCL_SUCCESS;
}

HcclResult AllGatherBatchSmallCountExecutor::BuildWindowRange(
    const WindowRange &current, u64 maxWindowBytes, WindowRange &range, WindowRange &next) const
{
    CHK_PRT_RET(current.startDescIdx >= param_.itemCount,
        HCCL_ERROR("[AllGatherBatchSmallCountExecutor][BuildWindowRange]tag[%s], startDescIdx[%u] out of range, itemCount[%u]",
            param_.tag, current.startDescIdx, param_.itemCount),
        HCCL_E_PARA);
    CHK_PRT_RET(maxWindowBytes == 0,
        HCCL_ERROR("[AllGatherBatchSmallCountExecutor][BuildWindowRange]tag[%s], maxWindowBytes is zero", param_.tag),
        HCCL_E_PARA);

    range = {};
    range.startDescIdx = current.startDescIdx;
    range.startOffset = current.startOffset;
    next = {};

    u64 remaining = maxWindowBytes;
    u64 packedSize = 0;
    uint32_t itemIdx = current.startDescIdx;
    u64 itemOffset = current.startOffset;

    while (itemIdx < param_.itemCount) {
        const BatchItemParam &item = param_.items[itemIdx];
        CHK_PRT_RET(item.sendBuf == nullptr || item.recvBuf == nullptr || item.sendBytes == 0,
            HCCL_ERROR("[AllGatherBatchSmallCountExecutor][BuildWindowRange]tag[%s], invalid item[%u], sendBuf[%p], recvBuf[%p], sendBytes[%llu]",
                param_.tag,
                itemIdx,
                item.sendBuf,
                item.recvBuf,
                static_cast<unsigned long long>(item.sendBytes)),
            HCCL_E_PARA);
        CHK_PRT_RET(itemOffset > item.sendBytes,
            HCCL_ERROR("[AllGatherBatchSmallCountExecutor][BuildWindowRange]tag[%s], item[%u] offset[%llu] exceeds sendBytes[%llu]",
                param_.tag,
                itemIdx,
                static_cast<unsigned long long>(itemOffset),
                static_cast<unsigned long long>(item.sendBytes)),
            HCCL_E_PARA);

        if (itemOffset == item.sendBytes) {
            ++itemIdx;
            itemOffset = 0;
            continue;
        }

        const u64 itemRemain = item.sendBytes - itemOffset;
        const u64 takeBytes = std::min(itemRemain, remaining);
        packedSize += takeBytes;
        remaining -= takeBytes;

        range.endDescIdx = itemIdx;
        range.endOffset = itemOffset + takeBytes;
        range.packedSize = packedSize;

        if (remaining == 0 || range.endOffset < item.sendBytes || itemIdx == (param_.itemCount - 1U)) {
            if (range.endOffset < item.sendBytes) {
                next.startDescIdx = range.endDescIdx;
                next.startOffset = range.endOffset;
            } else if (range.endDescIdx + 1U >= param_.itemCount) {
                next.startDescIdx = param_.itemCount;
                next.startOffset = 0;
            } else {
                next.startDescIdx = range.endDescIdx + 1U;
                next.startOffset = 0;
            }
            return HCCL_SUCCESS;
        }

        ++itemIdx;
        itemOffset = 0;
    }

    CHK_PRT_RET(range.packedSize == 0,
        HCCL_ERROR("[AllGatherBatchSmallCountExecutor][BuildWindowRange]tag[%s], packedSize is zero at startDescIdx[%u], startOffset[%llu]",
            param_.tag,
            current.startDescIdx,
            static_cast<unsigned long long>(current.startOffset)),
        HCCL_E_INTERNAL);
    return HCCL_SUCCESS;
}

HcclResult AllGatherBatchSmallCountExecutor::PackWindowToCCLIn(const WindowRange &range, void *commInputPtr)
{
    CHK_PTR_NULL(commInputPtr);
    u8 *packedPtr = static_cast<u8 *>(commInputPtr);
    u64 packedOffset = 0;

    for (uint32_t itemIdx = range.startDescIdx; itemIdx <= range.endDescIdx; ++itemIdx) {
        const BatchItemParam &item = param_.items[itemIdx];
        const u64 startOffset = (itemIdx == range.startDescIdx) ? range.startOffset : 0;
        const u64 endOffset = (itemIdx == range.endDescIdx) ? range.endOffset : item.sendBytes;
        CHK_PRT_RET(endOffset < startOffset || endOffset > item.sendBytes,
            HCCL_ERROR("[AllGatherBatchSmallCountExecutor][PackWindowToCCLIn]tag[%s], invalid range on item[%u], startOffset[%llu], endOffset[%llu], sendBytes[%llu]",
                param_.tag,
                itemIdx,
                static_cast<unsigned long long>(startOffset),
                static_cast<unsigned long long>(endOffset),
                static_cast<unsigned long long>(item.sendBytes)),
            HCCL_E_PARA);

        const u64 sizeBytes = endOffset - startOffset;
        if (sizeBytes == 0) {
            continue;
        }

        void *dstPtr = packedPtr + packedOffset;
        void *srcPtr = static_cast<u8 *>(item.sendBuf) + startOffset;
        CHK_RET(HcommLocalCopyOnThread(resCtx_.mainThreadHandle, dstPtr, srcPtr, sizeBytes));
        packedOffset += sizeBytes;
    }

    CHK_PRT_RET(packedOffset != range.packedSize,
        HCCL_ERROR("[AllGatherBatchSmallCountExecutor][PackWindowToCCLIn]tag[%s], packedOffset[%llu] != packedSize[%llu]",
            param_.tag,
            static_cast<unsigned long long>(packedOffset),
            static_cast<unsigned long long>(range.packedSize)),
        HCCL_E_INTERNAL);
    return HCCL_SUCCESS;
}

HcclResult AllGatherBatchSmallCountExecutor::UnpackWindowFromCCLOut(const WindowRange &range, u8 *commOutputPtr)
{
    CHK_PTR_NULL(commOutputPtr);

    for (u32 rank = 0; rank < param_.topoInfo.rankSize; ++rank) {
        u64 packedOffset = 0;
        for (uint32_t itemIdx = range.startDescIdx; itemIdx <= range.endDescIdx; ++itemIdx) {
            const BatchItemParam &item = param_.items[itemIdx];
            const u64 startOffset = (itemIdx == range.startDescIdx) ? range.startOffset : 0;
            const u64 endOffset = (itemIdx == range.endDescIdx) ? range.endOffset : item.sendBytes;
            CHK_PRT_RET(endOffset < startOffset || endOffset > item.sendBytes,
                HCCL_ERROR("[AllGatherBatchSmallCountExecutor][UnpackWindowFromCCLOut]tag[%s], invalid range on item[%u], startOffset[%llu], endOffset[%llu], sendBytes[%llu]",
                    param_.tag,
                    itemIdx,
                    static_cast<unsigned long long>(startOffset),
                    static_cast<unsigned long long>(endOffset),
                    static_cast<unsigned long long>(item.sendBytes)),
                HCCL_E_PARA);

            const u64 sizeBytes = endOffset - startOffset;
            if (sizeBytes == 0) {
                continue;
            }

            void *srcPtr = commOutputPtr + rank * range.packedSize + packedOffset;
            void *dstPtr = static_cast<u8 *>(item.recvBuf) + rank * item.sendBytes + startOffset;
            CHK_RET(HcommLocalCopyOnThread(resCtx_.mainThreadHandle, dstPtr, srcPtr, sizeBytes));
            packedOffset += sizeBytes;
        }

        CHK_PRT_RET(packedOffset != range.packedSize,
            HCCL_ERROR("[AllGatherBatchSmallCountExecutor][UnpackWindowFromCCLOut]tag[%s], rank[%u] packedOffset[%llu] != packedSize[%llu]",
                param_.tag,
                rank,
                static_cast<unsigned long long>(packedOffset),
                static_cast<unsigned long long>(range.packedSize)),
            HCCL_E_INTERNAL);
    }
    return HCCL_SUCCESS;
}

HcclResult AllGatherBatchSmallCountExecutor::RunLoop(std::vector<ChannelResource> &channels)
{
    void *commInputPtr = resCtx_.localBuffer.addr;
    u8 *commOutputPtr = static_cast<u8 *>(resCtx_.localBuffer.addr) + resCtx_.localBuffer.offset;
    CHK_PTR_NULL(commInputPtr);
    CHK_PTR_NULL(commOutputPtr);

    const u64 inputCapacity = resCtx_.localBuffer.offset;
    const u64 outputCapacity = resCtx_.localBuffer.size - resCtx_.localBuffer.offset;
    const u64 maxWindowBytes = outputCapacity / param_.topoInfo.rankSize;
    CHK_PRT_RET(maxWindowBytes == 0,
        HCCL_ERROR("[AllGatherBatchSmallCountExecutor][RunLoop]tag[%s], maxWindowBytes is zero, inputCapacity[%llu], outputCapacity[%llu], rankSize[%u]",
            param_.tag,
            static_cast<unsigned long long>(inputCapacity),
            static_cast<unsigned long long>(outputCapacity),
            param_.topoInfo.rankSize),
        HCCL_E_PARA);
    profiling_.localBufferBytes = resCtx_.localBuffer.size;
    profiling_.maxWindowBytes = maxWindowBytes;

    WindowRange current;
    current.startDescIdx = 0;
    current.startOffset = 0;

    while (current.startDescIdx < param_.itemCount) {
        WindowRange range;
        WindowRange next;
        CHK_RET(BuildWindowRange(current, maxWindowBytes, range, next));
        CHK_PRT_RET(range.packedSize > inputCapacity,
            HCCL_ERROR("[AllGatherBatchSmallCountExecutor][RunLoop]tag[%s], packedSize[%llu] exceeds inputCapacity[%llu]",
                param_.tag,
                static_cast<unsigned long long>(range.packedSize),
                static_cast<unsigned long long>(inputCapacity)),
            HCCL_E_INTERNAL);
        CHK_PRT_RET(range.packedSize * param_.topoInfo.rankSize > outputCapacity,
            HCCL_ERROR("[AllGatherBatchSmallCountExecutor][RunLoop]tag[%s], outputBytes[%llu] exceeds outputCapacity[%llu]",
                param_.tag,
                static_cast<unsigned long long>(range.packedSize * param_.topoInfo.rankSize),
                static_cast<unsigned long long>(outputCapacity)),
            HCCL_E_INTERNAL);
        ++profiling_.windowCount;

        const uint64_t packStartUs = GetCurrentTimeUs();
        CHK_RET(PackWindowToCCLIn(range, commInputPtr));
        profiling_.packUs += (GetCurrentTimeUs() - packStartUs);

        ExecMem execMem;
        execMem.count = range.packedSize;
        execMem.dataType = HCCL_DATA_TYPE_INT8;
        execMem.inputMem = {HCCL_MEM_TYPE_DEVICE, commInputPtr, range.packedSize};
        execMem.outputMem = {HCCL_MEM_TYPE_DEVICE, commOutputPtr, range.packedSize * param_.topoInfo.rankSize};
        execMem.inputPtr = commInputPtr;
        execMem.outputPtr = commOutputPtr;

        HcclResult ret = KernelRun(execMem, channels);
        CHK_PRT_RET(ret != HCCL_SUCCESS,
            HCCL_ERROR("[AllGatherBatchSmallCountExecutor][RunLoop]errNo[0x%016llx]kernel run error, tag[%s], inputMem ptr[%p], outputMem ptr[%p], packedBytes[%llu]",
                HCCL_ERROR_CODE(ret),
                param_.tag,
                commInputPtr,
                commOutputPtr,
                static_cast<unsigned long long>(range.packedSize)),
            ret);

        const uint64_t unpackStartUs = GetCurrentTimeUs();
        CHK_RET(UnpackWindowFromCCLOut(range, commOutputPtr));
        profiling_.unpackUs += (GetCurrentTimeUs() - unpackStartUs);

        current = next;
    }
    return HCCL_SUCCESS;
}

HcclResult AllGatherBatchSmallCountExecutor::KernelRun(ExecMem &execMem, std::vector<ChannelResource> &channels)
{
    AllGatherHDStage hdStageCore(param_, resCtx_, execMem, channels);
    const uint64_t hdStageStartUs = GetCurrentTimeUs();
    HcclResult commRet = hdStageCore.RunAsync();
    profiling_.hdStageUs += (GetCurrentTimeUs() - hdStageStartUs);
    CHK_RET(commRet);

    return HCCL_SUCCESS;
}

}  // namespace ops_hccl_allgatherbatch
