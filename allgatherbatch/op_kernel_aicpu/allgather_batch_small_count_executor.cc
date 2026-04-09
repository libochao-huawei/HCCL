#include "allgather_batch_small_count_executor.h"

#include <algorithm>

#include "all_gather_hd_stage_core.h"
#include "log.h"

namespace ops_hccl_allgatherbatch {

namespace {

uint64_t CalcRemainingBytes(const BatchItemParam &item, uint64_t offsetBytes)
{
    return (offsetBytes < item.sendBytes) ? (item.sendBytes - offsetBytes) : 0;
}

const char *ToWindowScopeString(BatchCommMode commMode)
{
    return (commMode == BatchCommMode::kCrossServer) ? "cross-server" : "single-server";
}

}  // namespace

AllGatherBatchSmallCountExecutor::AllGatherBatchSmallCountExecutor(
    const OpParam &param, AlgResourceCtx &resCtx, BatchCallProfiling &profiling)
    : param_(param), resCtx_(resCtx), profiling_(profiling), stageLayout_(BuildWindowStageLayout(param.topoInfo.rankSize))
{
}

HcclResult AllGatherBatchSmallCountExecutor::ValidateParam() const
{
    if (!IsValidWindowStageLayout(stageLayout_)) {
        HCCL_ERROR("stage layout is invalid, rankSize=%u, powerSteps=%u, powerFactor=%u, noPower=%u",
            stageLayout_.rankSize,
            stageLayout_.powerSteps,
            stageLayout_.powerFactor,
            stageLayout_.noPower);
        return HCCL_E_INTERNAL;
    }
    for (uint32_t itemIdx = 0; itemIdx < param_.itemCount; ++itemIdx) {
        const BatchItemParam &item = param_.items[itemIdx];
        if (item.sendBuf == nullptr || item.recvBuf == nullptr) {
            HCCL_ERROR("item %u buffer is null", itemIdx);
            return HCCL_E_PTR;
        }
        if (item.sendBytes == 0) {
            HCCL_ERROR("item %u sendBytes is zero", itemIdx);
            return HCCL_E_PARA;
        }
    }
    return HCCL_SUCCESS;
}

HcclResult AllGatherBatchSmallCountExecutor::ValidateWindow(const WindowRange &window) const
{
    if (window.packedBytes == 0) {
        HCCL_ERROR("window packedBytes is zero");
        return HCCL_E_INTERNAL;
    }
    if (window.startItemIdx >= param_.itemCount) {
        HCCL_ERROR("window startItemIdx=%u is out of range, itemCount=%u",
            window.startItemIdx,
            param_.itemCount);
        return HCCL_E_INTERNAL;
    }
    if (window.startOffsetBytes >= param_.items[window.startItemIdx].sendBytes) {
        HCCL_ERROR("window startOffset=%llu is invalid for item=%u, itemBytes=%llu",
            static_cast<unsigned long long>(window.startOffsetBytes),
            window.startItemIdx,
            static_cast<unsigned long long>(param_.items[window.startItemIdx].sendBytes));
        return HCCL_E_INTERNAL;
    }
    if (window.endItemIdx > param_.itemCount) {
        HCCL_ERROR("window endItemIdx=%u exceeds itemCount=%u", window.endItemIdx, param_.itemCount);
        return HCCL_E_INTERNAL;
    }
    if (window.endItemIdx == param_.itemCount) {
        if (window.endOffsetBytes != 0) {
            HCCL_ERROR("terminal window endOffsetBytes=%llu must be zero",
                static_cast<unsigned long long>(window.endOffsetBytes));
            return HCCL_E_INTERNAL;
        }
    } else if (window.endOffsetBytes >= param_.items[window.endItemIdx].sendBytes) {
        HCCL_ERROR("window endOffset=%llu is invalid for item=%u, itemBytes=%llu",
            static_cast<unsigned long long>(window.endOffsetBytes),
            window.endItemIdx,
            static_cast<unsigned long long>(param_.items[window.endItemIdx].sendBytes));
        return HCCL_E_INTERNAL;
    }

    const uint64_t coveredBytes = CalcWindowCoveredBytes(window);
    if (coveredBytes != window.packedBytes) {
        HCCL_ERROR("window bytes mismatch, covered=%llu, packed=%llu",
            static_cast<unsigned long long>(coveredBytes),
            static_cast<unsigned long long>(window.packedBytes));
        return HCCL_E_INTERNAL;
    }
    if (window.packedBytes > GetPerRankWindowCapacity()) {
        HCCL_ERROR("window packedBytes=%llu exceeds per-rank capacity=%llu",
            static_cast<unsigned long long>(window.packedBytes),
            static_cast<unsigned long long>(GetPerRankWindowCapacity()));
        return HCCL_E_INTERNAL;
    }
    if (window.packedBytes > param_.windowBytes) {
        HCCL_ERROR("window packedBytes=%llu exceeds param windowBytes=%llu",
            static_cast<unsigned long long>(window.packedBytes),
            static_cast<unsigned long long>(param_.windowBytes));
        return HCCL_E_INTERNAL;
    }
    return HCCL_SUCCESS;
}

uint64_t AllGatherBatchSmallCountExecutor::CalcWindowCoveredBytes(const WindowRange &window) const
{
    uint64_t coveredBytes = 0;
    uint32_t itemIdx = window.startItemIdx;
    uint64_t offsetBytes = window.startOffsetBytes;

    while (itemIdx < param_.itemCount) {
        if (itemIdx == window.endItemIdx) {
            coveredBytes += (window.endOffsetBytes - offsetBytes);
            return coveredBytes;
        }

        coveredBytes += CalcRemainingBytes(param_.items[itemIdx], offsetBytes);
        ++itemIdx;
        offsetBytes = 0;
    }
    return coveredBytes;
}

uint64_t AllGatherBatchSmallCountExecutor::GetPerRankWindowCapacity() const
{
    return ops_hccl_allgatherbatch::GetPerRankWindowCapacity(param_, resCtx_);
}

uint8_t *AllGatherBatchSmallCountExecutor::GetStageRankWindowBase(const WindowStageLayout &layout, uint32_t rank) const
{
    return static_cast<uint8_t *>(resCtx_.localBuffer.addr) + GetStageRankBaseOffset(layout, rank);
}

HcclResult AllGatherBatchSmallCountExecutor::AdvancePosition(uint32_t &itemIdx, uint64_t &offsetBytes) const
{
    while (itemIdx < param_.itemCount && offsetBytes >= param_.items[itemIdx].sendBytes) {
        ++itemIdx;
        offsetBytes = 0;
    }
    return HCCL_SUCCESS;
}

HcclResult AllGatherBatchSmallCountExecutor::LocateWindowEnd(
    uint32_t startItemIdx,
    uint64_t startOffsetBytes,
    uint64_t packedBytes,
    uint32_t &endItemIdx,
    uint64_t &endOffsetBytes) const
{
    uint32_t itemIdx = startItemIdx;
    uint64_t offsetBytes = startOffsetBytes;
    uint64_t remaining = packedBytes;

    while (remaining > 0 && itemIdx < param_.itemCount) {
        const uint64_t itemRemaining = CalcRemainingBytes(param_.items[itemIdx], offsetBytes);
        const uint64_t currentBytes = std::min(itemRemaining, remaining);
        remaining -= currentBytes;
        offsetBytes += currentBytes;
        HCCL_CHK_RET(AdvancePosition(itemIdx, offsetBytes));
    }

    endItemIdx = itemIdx;
    endOffsetBytes = offsetBytes;
    return HCCL_SUCCESS;
}

HcclResult AllGatherBatchSmallCountExecutor::BuildFirstWindow(WindowRange &window) const
{
    const uint64_t packedBytes = std::min(param_.totalInputBytes, GetMaxWindowBytes(param_, resCtx_));
    window.startItemIdx = 0;
    window.startOffsetBytes = 0;
    window.packedBytes = packedBytes;
    HCCL_CHK_RET(LocateWindowEnd(0, 0, packedBytes, window.endItemIdx, window.endOffsetBytes));
    return ValidateWindow(window);
}

HcclResult AllGatherBatchSmallCountExecutor::BuildNextWindow(
    const WindowRange &current, WindowRange &next, bool &hasNext) const
{
    uint32_t nextItemIdx = current.endItemIdx;
    uint64_t nextOffsetBytes = current.endOffsetBytes;
    HCCL_CHK_RET(AdvancePosition(nextItemIdx, nextOffsetBytes));

    if (nextItemIdx >= param_.itemCount) {
        hasNext = false;
        return HCCL_SUCCESS;
    }

    uint64_t remainingBytes = 0;
    for (uint32_t itemIdx = nextItemIdx; itemIdx < param_.itemCount; ++itemIdx) {
        remainingBytes += CalcRemainingBytes(param_.items[itemIdx], (itemIdx == nextItemIdx) ? nextOffsetBytes : 0);
    }

    next.startItemIdx = nextItemIdx;
    next.startOffsetBytes = nextOffsetBytes;
    next.packedBytes = std::min(remainingBytes, GetMaxWindowBytes(param_, resCtx_));
    HCCL_CHK_RET(LocateWindowEnd(nextItemIdx, nextOffsetBytes, next.packedBytes, next.endItemIdx, next.endOffsetBytes));
    HCCL_CHK_RET(ValidateWindow(next));
    hasNext = true;
    return HCCL_SUCCESS;
}

HcclResult AllGatherBatchSmallCountExecutor::Pack(const WindowStageLayout &layout) const
{
    if (!IsValidWindowStageLayout(layout)) {
        HCCL_ERROR("pack layout is invalid");
        return HCCL_E_INTERNAL;
    }

    for (const WindowStageSlice &slice : layout.localSlices) {
        const BatchItemParam &item = param_.items[slice.itemIdx];
        const void *src = static_cast<const uint8_t *>(item.sendBuf) + slice.itemOffsetBytes;
        void *dst = static_cast<uint8_t *>(resCtx_.localBuffer.addr) + slice.stageOffsetBytes;
        const int32_t ret = HcommLocalCopyOnThread(resCtx_.threadHandle, dst, src, slice.size);
        if (ret != HCCL_SUCCESS) {
            HCCL_ERROR("pack local copy failed, item=%u, rankOffset=%llu, size=%llu, ret=%d",
                slice.itemIdx,
                static_cast<unsigned long long>(slice.rankOffsetBytes),
                static_cast<unsigned long long>(slice.size),
                ret);
            return static_cast<HcclResult>(ret);
        }
    }
    return HCCL_SUCCESS;
}

HcclResult AllGatherBatchSmallCountExecutor::Unpack(const WindowStageLayout &layout) const
{
    if (!IsValidWindowStageLayout(layout)) {
        HCCL_ERROR("unpack layout is invalid");
        return HCCL_E_INTERNAL;
    }

    for (const WindowStageSlice &slice : layout.perRankSlices) {
        const BatchItemParam &item = param_.items[slice.itemIdx];
        const void *src = static_cast<const uint8_t *>(resCtx_.localBuffer.addr) + slice.stageOffsetBytes;
        uint8_t *dst = static_cast<uint8_t *>(item.recvBuf) + (slice.rank * item.sendBytes) + slice.itemOffsetBytes;
        const int32_t ret = HcommLocalCopyOnThread(resCtx_.threadHandle, dst, src, slice.size);
        if (ret != HCCL_SUCCESS) {
            HCCL_ERROR("unpack local copy failed, rank=%u, item=%u, itemOffset=%llu, size=%llu, ret=%d",
                slice.rank,
                slice.itemIdx,
                static_cast<unsigned long long>(slice.itemOffsetBytes),
                static_cast<unsigned long long>(slice.size),
                ret);
            return static_cast<HcclResult>(ret);
        }
    }
    return HCCL_SUCCESS;
}

HcclResult AllGatherBatchSmallCountExecutor::Orchestrate()
{
    HCCL_CHK_RET(ValidateParam());

    WindowRange window;
    HCCL_CHK_RET(BuildFirstWindow(window));

    while (true) {
        const WindowStageLayout layout = BuildWindowStageLayout(param_, window);
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
        HCCL_INFO("executor window ready: scope=%s, start=(%u,%llu), end=(%u,%llu), packedBytes=%llu, paramWindowBytes=%llu, perRankCapacity=%llu, maxWindowBytes=%llu, rankSize=%u, stagePowerSteps=%u, stageNoPower=%u, localSlices=%u",
            ToWindowScopeString(param_.commMode),
            window.startItemIdx,
            static_cast<unsigned long long>(window.startOffsetBytes),
            window.endItemIdx,
            static_cast<unsigned long long>(window.endOffsetBytes),
            static_cast<unsigned long long>(window.packedBytes),
            static_cast<unsigned long long>(param_.windowBytes),
            static_cast<unsigned long long>(GetPerRankWindowCapacity()),
            static_cast<unsigned long long>(GetMaxWindowBytes(param_, resCtx_)),
            param_.topoInfo.rankSize,
            layout.powerSteps,
            layout.noPower,
            static_cast<uint32_t>(layout.localSlices.size()));

        const uint64_t packStartUs = GetCurrentTimeUs();
        HCCL_CHK_RET(Pack(layout));
        profiling_.packUs += (GetCurrentTimeUs() - packStartUs);

        AllGatherHDStageCore hdStageCore(param_, resCtx_, layout);
        const uint64_t hdStageStartUs = GetCurrentTimeUs();
        HcclResult commRet = hdStageCore.RunAsync();
        profiling_.hdStageUs += (GetCurrentTimeUs() - hdStageStartUs);
        if (commRet != HCCL_SUCCESS) {
            return commRet;
        }

        const uint64_t unpackStartUs = GetCurrentTimeUs();
        HCCL_CHK_RET(Unpack(layout));
        profiling_.unpackUs += (GetCurrentTimeUs() - unpackStartUs);

        bool hasNext = false;
        WindowRange nextWindow;
        HCCL_CHK_RET(BuildNextWindow(window, nextWindow, hasNext));
        if (!hasNext) {
            return HCCL_SUCCESS;
        }
        window = nextWindow;
    }
}

}  // namespace ops_hccl_allgatherbatch
