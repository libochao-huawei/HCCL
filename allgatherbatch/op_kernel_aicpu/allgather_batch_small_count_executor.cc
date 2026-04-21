#include "allgather_batch_small_count_executor.h"

#include <algorithm>

#include "all_gather_hd_stage_core.h"
#include "log.h"

namespace ops_hccl_allgatherbatch {

namespace {

constexpr u64 kDualThreadUnpackMinBytes = 256 * 1024;
constexpr u32 kQuadThreadWorkerCount = 4;

void BuildUnpackRankChunks(
    u32 rankSize,
    std::vector<u32> &beginRanks,
    std::vector<u32> &endRanks)
{
    u32 counts[kQuadThreadWorkerCount] = {0};
    const u32 remain = rankSize - 1;
    const u32 base = remain / kQuadThreadWorkerCount;
    const u32 extra = remain % kQuadThreadWorkerCount;

    counts[0] = 1 + base;
    for (u32 i = 1; i < kQuadThreadWorkerCount; ++i) {
        counts[i] = base;
    }
    for (u32 i = 0; i < extra; ++i) {
        counts[i] += 1;
    }

    for (u32 tid = 0; tid < kQuadThreadWorkerCount; ++tid) {
        if (tid > 0) {
            beginRanks[tid] = endRanks[tid - 1];
        }
        endRanks[tid] = beginRanks[tid] + counts[tid];
    }
}

HcclResult UnpackRankRangeOnThread(
    ThreadHandle thread,
    const OpParam &param,
    const std::vector<WindowPart> &parts,
    u64 packedSize,
    u8 *commOutputPtr,
    uint32_t beginRank,
    uint32_t endRank)
{
    CHK_PTR_NULL(commOutputPtr);
    CHK_PRT_RET(beginRank > endRank || endRank > param.topoInfo.rankSize,
        HCCL_ERROR("[AllGatherBatchSmallCountExecutor][UnpackRankRangeOnThread]tag[%s], invalid rank range [%u, %u), rankSize[%u]",
            param.tag, beginRank, endRank, param.topoInfo.rankSize),
        HCCL_E_PARA);

    for (u32 rank = beginRank; rank < endRank; ++rank) {
        for (const WindowPart &part : parts) {
            const BatchItemParam &item = param.items[part.itemIdx];
            void *srcPtr = commOutputPtr + rank * packedSize + part.packedOffset;
            void *dstPtr = static_cast<u8 *>(item.recvBuf) + rank * item.sendBytes + part.startOffset;
            CHK_RET(HcommLocalCopyOnThread(thread, dstPtr, srcPtr, part.sizeBytes));
        }
    }
    return HCCL_SUCCESS;
}

} // namespace

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

HcclResult AllGatherBatchSmallCountExecutor::BuildWindowPlan(
    const WindowRange &current,
    u64 maxWindowBytes,
    WindowRange &range,
    WindowRange &next,
    std::vector<WindowPart> &parts) const
{
    CHK_PRT_RET(current.startDescIdx >= param_.itemCount,
        HCCL_ERROR("[AllGatherBatchSmallCountExecutor][BuildWindowPlan]tag[%s], startDescIdx[%u] out of range, itemCount[%u]",
            param_.tag, current.startDescIdx, param_.itemCount),
        HCCL_E_PARA);
    CHK_PRT_RET(maxWindowBytes == 0,
        HCCL_ERROR("[AllGatherBatchSmallCountExecutor][BuildWindowPlan]tag[%s], maxWindowBytes is zero", param_.tag),
        HCCL_E_PARA);

    range = {};
    range.startDescIdx = current.startDescIdx;
    range.startOffset = current.startOffset;
    next = {};
    parts.clear();

    u64 remaining = maxWindowBytes;
    u64 packedSize = 0;
    uint32_t itemIdx = current.startDescIdx;
    u64 itemOffset = current.startOffset;

    while (itemIdx < param_.itemCount) {
        const BatchItemParam &item = param_.items[itemIdx];
        CHK_PRT_RET(item.sendBuf == nullptr || item.recvBuf == nullptr || item.sendBytes == 0,
            HCCL_ERROR("[AllGatherBatchSmallCountExecutor][BuildWindowPlan]tag[%s], invalid item[%u], sendBuf[%p], recvBuf[%p], sendBytes[%llu]",
                param_.tag,
                itemIdx,
                item.sendBuf,
                item.recvBuf,
                static_cast<unsigned long long>(item.sendBytes)),
            HCCL_E_PARA);
        CHK_PRT_RET(itemOffset > item.sendBytes,
            HCCL_ERROR("[AllGatherBatchSmallCountExecutor][BuildWindowPlan]tag[%s], item[%u] offset[%llu] exceeds sendBytes[%llu]",
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
        CHK_PRT_RET(takeBytes == 0,
            HCCL_ERROR("[AllGatherBatchSmallCountExecutor][BuildWindowPlan]tag[%s], takeBytes is zero on item[%u]",
                param_.tag, itemIdx),
            HCCL_E_INTERNAL);

        WindowPart part;
        part.itemIdx = itemIdx;
        part.startOffset = itemOffset;
        part.sizeBytes = takeBytes;
        part.packedOffset = packedSize;
        parts.push_back(part);

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

            CHK_PRT_RET(parts.empty(),
                HCCL_ERROR("[AllGatherBatchSmallCountExecutor][BuildWindowPlan]tag[%s], parts is empty", param_.tag),
                HCCL_E_INTERNAL);
            CHK_PRT_RET(packedSize != range.packedSize,
                HCCL_ERROR("[AllGatherBatchSmallCountExecutor][BuildWindowPlan]tag[%s], packedSize[%llu] != range.packedSize[%llu]",
                    param_.tag,
                    static_cast<unsigned long long>(packedSize),
                    static_cast<unsigned long long>(range.packedSize)),
                HCCL_E_INTERNAL);
            return HCCL_SUCCESS;
        }

        ++itemIdx;
        itemOffset = 0;
    }

    CHK_PRT_RET(range.packedSize == 0,
        HCCL_ERROR("[AllGatherBatchSmallCountExecutor][BuildWindowPlan]tag[%s], packedSize is zero at startDescIdx[%u], startOffset[%llu]",
            param_.tag,
            current.startDescIdx,
            static_cast<unsigned long long>(current.startOffset)),
        HCCL_E_INTERNAL);
    return HCCL_SUCCESS;
}

HcclResult AllGatherBatchSmallCountExecutor::PackWindowToCCLIn(
    const std::vector<WindowPart> &parts, void *commInputPtr)
{
    CHK_PTR_NULL(commInputPtr);

    u8 *packedPtr = static_cast<u8 *>(commInputPtr);
    for (const WindowPart &part : parts) {
        const BatchItemParam &item = param_.items[part.itemIdx];
        void *dstPtr = packedPtr + part.packedOffset;
        void *srcPtr = static_cast<u8 *>(item.sendBuf) + part.startOffset;
        CHK_RET(HcommLocalCopyOnThread(resCtx_.mainThreadHandle, dstPtr, srcPtr, part.sizeBytes));
    }
    return HCCL_SUCCESS;
}

HcclResult AllGatherBatchSmallCountExecutor::UnpackWindowFromCCLOut(
    const std::vector<WindowPart> &parts, u64 packedSize, u8 *commOutputPtr)
{
    CHK_PTR_NULL(commOutputPtr);

    const u32 rankSize = param_.topoInfo.rankSize;
    const u64 totalUnpackBytes = packedSize * rankSize;
    const bool useQuadThreadUnpack = (rankSize > 1) &&
        (resCtx_.subThreadHandles[0] != 0) &&
        (resCtx_.subThreadHandles[1] != 0) &&
        (resCtx_.subThreadHandles[2] != 0) &&
        (totalUnpackBytes >= kDualThreadUnpackMinBytes);

    if (!useQuadThreadUnpack) {
        return UnpackRankRangeOnThread(
            resCtx_.mainThreadHandle, param_, parts, packedSize, commOutputPtr, 0, rankSize);
    }

    std::vector<u32> beginRanks(kQuadThreadWorkerCount, 0);
    std::vector<u32> endRanks(kQuadThreadWorkerCount, 0);
    BuildUnpackRankChunks(rankSize, beginRanks, endRanks);
    const auto hasRankChunk = [&](u32 tid) {
        return beginRanks[tid] < endRanks[tid];
    };

    for (u32 tid = 1; tid < kQuadThreadWorkerCount; ++tid) {
        if (hasRankChunk(tid)) {
            CHK_RET(HcommThreadNotifyRecordOnThread(
                resCtx_.mainThreadHandle, resCtx_.subThreadHandles[tid - 1], resCtx_.subNotifyIds[tid - 1]));
        }
    }
    for (u32 tid = 1; tid < kQuadThreadWorkerCount; ++tid) {
        if (hasRankChunk(tid)) {
            CHK_RET(HcommThreadNotifyWaitOnThread(
                resCtx_.subThreadHandles[tid - 1], resCtx_.subNotifyIds[tid - 1], CUSTOM_TIMEOUT));
        }
    }

    if (hasRankChunk(0)) {
        CHK_RET(UnpackRankRangeOnThread(
            resCtx_.mainThreadHandle, param_, parts, packedSize, commOutputPtr, beginRanks[0], endRanks[0]));
    }
    for (u32 tid = 1; tid < kQuadThreadWorkerCount; ++tid) {
        if (hasRankChunk(tid)) {
            CHK_RET(UnpackRankRangeOnThread(
                resCtx_.subThreadHandles[tid - 1], param_, parts, packedSize, commOutputPtr, beginRanks[tid], endRanks[tid]));
        }
    }

    for (u32 tid = 1; tid < kQuadThreadWorkerCount; ++tid) {
        if (hasRankChunk(tid)) {
            CHK_RET(HcommThreadNotifyRecordOnThread(
                resCtx_.subThreadHandles[tid - 1], resCtx_.mainThreadHandle, resCtx_.mainNotifyIds[tid - 1]));
        }
    }
    for (u32 tid = 1; tid < kQuadThreadWorkerCount; ++tid) {
        if (hasRankChunk(tid)) {
            CHK_RET(HcommThreadNotifyWaitOnThread(
                resCtx_.mainThreadHandle, resCtx_.mainNotifyIds[tid - 1], CUSTOM_TIMEOUT));
        }
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

    std::vector<WindowPart> parts;
    parts.reserve(param_.itemCount);
    while (current.startDescIdx < param_.itemCount) {
        WindowRange range;
        WindowRange next;
        CHK_RET(BuildWindowPlan(current, maxWindowBytes, range, next, parts));
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
        CHK_RET(PackWindowToCCLIn(parts, commInputPtr));
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
        CHK_RET(UnpackWindowFromCCLOut(parts, range.packedSize, commOutputPtr));
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

