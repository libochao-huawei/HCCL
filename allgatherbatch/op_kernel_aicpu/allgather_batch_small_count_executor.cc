#include "allgather_batch_small_count_executor.h"

#include <algorithm>

#include "all_gather_hd_stage_core.h"
#include "log.h"

namespace ops_hccl_allgatherbatch {

namespace {

const char *ToWindowScopeString(BatchCommMode commMode)
{
    return (commMode == BatchCommMode::kCrossServer) ? "cross-server" : "single-server";
}

}  // namespace

AllGatherBatchSmallCountExecutor::AllGatherBatchSmallCountExecutor(
    const OpParam &param, AlgResourceCtx &resCtx, BatchCallProfiling &profiling)
    : param_(param), resCtx_(resCtx), profiling_(profiling)
{
}

const BatchItemParam &AllGatherBatchSmallCountExecutor::GetInputItem() const
{
    return param_.items[0];
}

HcclResult AllGatherBatchSmallCountExecutor::ValidateParam() const
{
    if (param_.itemCount == 0) {
        HCCL_ERROR("executor itemCount is zero");
        return HCCL_E_PARA;
    }

    const BatchItemParam &item = GetInputItem();
    if (item.sendBuf == nullptr || item.recvBuf == nullptr) {
        HCCL_ERROR("item[0] buffer is null");
        return HCCL_E_PTR;
    }
    if (item.sendBytes == 0) {
        HCCL_ERROR("item[0] sendBytes is zero");
        return HCCL_E_PARA;
    }

    const uint64_t maxWindowBytes = GetMaxWindowBytes(param_, resCtx_);
    if (maxWindowBytes == 0) {
        HCCL_ERROR("maxWindowBytes is zero, localBuffer=%llu, rankSize=%u, windowBytes=%llu",
            static_cast<unsigned long long>(resCtx_.localBuffer.size),
            param_.topoInfo.rankSize,
            static_cast<unsigned long long>(param_.windowBytes));
        return HCCL_E_INTERNAL;
    }
    return HCCL_SUCCESS;
}

HcclResult AllGatherBatchSmallCountExecutor::ValidateWindow(uint64_t windowOffset, uint64_t currentWindowBytes) const
{
    const BatchItemParam &item = GetInputItem();
    if (windowOffset >= item.sendBytes) {
        HCCL_ERROR("window offset=%llu exceeds item[0] bytes=%llu",
            static_cast<unsigned long long>(windowOffset),
            static_cast<unsigned long long>(item.sendBytes));
        return HCCL_E_INTERNAL;
    }
    if (currentWindowBytes == 0U) {
        HCCL_ERROR("currentWindowBytes is zero, windowOffset=%llu",
            static_cast<unsigned long long>(windowOffset));
        return HCCL_E_INTERNAL;
    }
    if ((windowOffset + currentWindowBytes) > item.sendBytes) {
        HCCL_ERROR("window range is invalid, offset=%llu, bytes=%llu, itemBytes=%llu",
            static_cast<unsigned long long>(windowOffset),
            static_cast<unsigned long long>(currentWindowBytes),
            static_cast<unsigned long long>(item.sendBytes));
        return HCCL_E_INTERNAL;
    }
    if (currentWindowBytes > GetPerRankWindowCapacity()) {
        HCCL_ERROR("window bytes=%llu exceeds per-rank capacity=%llu",
            static_cast<unsigned long long>(currentWindowBytes),
            static_cast<unsigned long long>(GetPerRankWindowCapacity()));
        return HCCL_E_INTERNAL;
    }
    if (currentWindowBytes > param_.windowBytes) {
        HCCL_ERROR("window bytes=%llu exceeds param windowBytes=%llu",
            static_cast<unsigned long long>(currentWindowBytes),
            static_cast<unsigned long long>(param_.windowBytes));
        return HCCL_E_INTERNAL;
    }
    return HCCL_SUCCESS;
}

uint64_t AllGatherBatchSmallCountExecutor::GetPerRankWindowCapacity() const
{
    return ops_hccl_allgatherbatch::GetPerRankWindowCapacity(param_, resCtx_);
}

WindowStageLayout AllGatherBatchSmallCountExecutor::BuildStageLayout(uint64_t currentWindowBytes) const
{
    return BuildSingleItemStageLayout(param_.topoInfo.rankSize, param_.topoInfo.rank, currentWindowBytes);
}

HcclResult AllGatherBatchSmallCountExecutor::Pack(
    uint64_t windowOffset, uint64_t currentWindowBytes, const WindowStageLayout &layout) const
{
    HCCL_CHK_RET(ValidateWindow(windowOffset, currentWindowBytes));

    const BatchItemParam &item = GetInputItem();
    const uint64_t localRankBase = GetStageRankBaseOffset(layout, param_.topoInfo.rank);
    const void *src = static_cast<const uint8_t *>(item.sendBuf) + windowOffset;
    void *dst = static_cast<uint8_t *>(resCtx_.localBuffer.addr) + localRankBase;
    const int32_t ret = HcommLocalCopyOnThread(resCtx_.mainThreadHandle, dst, src, currentWindowBytes);
    if (ret != HCCL_SUCCESS) {
        HCCL_ERROR("pack local copy failed, rank=%u, itemOffset=%llu, stageOffset=%llu, size=%llu, ret=%d",
            param_.topoInfo.rank,
            static_cast<unsigned long long>(windowOffset),
            static_cast<unsigned long long>(localRankBase),
            static_cast<unsigned long long>(currentWindowBytes),
            ret);
        return static_cast<HcclResult>(ret);
    }
    return HCCL_SUCCESS;
}

HcclResult AllGatherBatchSmallCountExecutor::Unpack(
    uint64_t windowOffset, uint64_t currentWindowBytes, const WindowStageLayout &layout) const
{
    HCCL_CHK_RET(ValidateWindow(windowOffset, currentWindowBytes));

    const BatchItemParam &item = GetInputItem();
    for (uint32_t rank = 0; rank < param_.topoInfo.rankSize; ++rank) {
        const uint64_t rankBase = GetStageRankBaseOffset(layout, rank);
        const void *src = static_cast<const uint8_t *>(resCtx_.localBuffer.addr) + rankBase;
        uint8_t *dst = static_cast<uint8_t *>(item.recvBuf) + (static_cast<uint64_t>(rank) * item.sendBytes) + windowOffset;
        const int32_t ret = HcommLocalCopyOnThread(resCtx_.mainThreadHandle, dst, src, currentWindowBytes);
        if (ret != HCCL_SUCCESS) {
            HCCL_ERROR("unpack local copy failed, rank=%u, itemOffset=%llu, stageOffset=%llu, size=%llu, ret=%d",
                rank,
                static_cast<unsigned long long>(windowOffset),
                static_cast<unsigned long long>(rankBase),
                static_cast<unsigned long long>(currentWindowBytes),
                ret);
            return static_cast<HcclResult>(ret);
        }
    }
    return HCCL_SUCCESS;
}

HcclResult AllGatherBatchSmallCountExecutor::Orchestrate()
{
    HCCL_CHK_RET(ValidateParam());

    const BatchItemParam &item = GetInputItem();
    const uint64_t maxWindowBytes = GetMaxWindowBytes(param_, resCtx_);
    uint64_t windowOffset = 0;
    while (windowOffset < item.sendBytes) {
        const uint64_t currentWindowBytes = std::min(item.sendBytes - windowOffset, maxWindowBytes);
        HCCL_CHK_RET(ValidateWindow(windowOffset, currentWindowBytes));

        const WindowStageLayout layout = BuildStageLayout(currentWindowBytes);
        if (!IsValidWindowStageLayout(layout)) {
            HCCL_ERROR("window stage layout is invalid, packedBytes=%llu, rankSize=%u, localSlices=%u, perRankSlices=%u",
                static_cast<unsigned long long>(layout.packedBytes),
                layout.rankSize,
                static_cast<uint32_t>(layout.localSlices.size()),
                static_cast<uint32_t>(layout.perRankSlices.size()));
            return HCCL_E_INTERNAL;
        }
        if (layout.totalBytes > resCtx_.localBuffer.size) {
            HCCL_ERROR("window stage layout exceeds localBuffer, totalBytes=%llu, localBuffer=%llu",
                static_cast<unsigned long long>(layout.totalBytes),
                static_cast<unsigned long long>(resCtx_.localBuffer.size));
            return HCCL_E_INTERNAL;
        }

        ++profiling_.windowCount;
        HCCL_INFO("executor window ready: scope=%s, item=0, offset=%llu, endOffset=%llu, packedBytes=%llu, itemBytes=%llu, paramWindowBytes=%llu, perRankCapacity=%llu, maxWindowBytes=%llu, rankSize=%u, stagePowerSteps=%u, stageNoPower=%u, localSlices=%u",
            ToWindowScopeString(param_.commMode),
            static_cast<unsigned long long>(windowOffset),
            static_cast<unsigned long long>(windowOffset + currentWindowBytes),
            static_cast<unsigned long long>(currentWindowBytes),
            static_cast<unsigned long long>(item.sendBytes),
            static_cast<unsigned long long>(param_.windowBytes),
            static_cast<unsigned long long>(GetPerRankWindowCapacity()),
            static_cast<unsigned long long>(maxWindowBytes),
            param_.topoInfo.rankSize,
            layout.powerSteps,
            layout.noPower,
            static_cast<uint32_t>(layout.localSlices.size()));

        const uint64_t packStartUs = GetCurrentTimeUs();
        HCCL_CHK_RET(Pack(windowOffset, currentWindowBytes, layout));
        profiling_.packUs += (GetCurrentTimeUs() - packStartUs);

        AllGatherHDStageCore hdStageCore(param_, resCtx_, layout);
        const uint64_t hdStageStartUs = GetCurrentTimeUs();
        HcclResult commRet = hdStageCore.RunAsync();
        profiling_.hdStageUs += (GetCurrentTimeUs() - hdStageStartUs);
        if (commRet != HCCL_SUCCESS) {
            return commRet;
        }

        const uint64_t unpackStartUs = GetCurrentTimeUs();
        HCCL_CHK_RET(Unpack(windowOffset, currentWindowBytes, layout));
        profiling_.unpackUs += (GetCurrentTimeUs() - unpackStartUs);

        windowOffset += currentWindowBytes;
    }
    return HCCL_SUCCESS;
}

}  // namespace ops_hccl_allgatherbatch


