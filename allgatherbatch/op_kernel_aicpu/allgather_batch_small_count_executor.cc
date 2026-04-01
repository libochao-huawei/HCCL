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

AllGatherBatchSmallCountExecutor::AllGatherBatchSmallCountExecutor(const OpParam &param, AlgResourceCtx &resCtx)
    : param_(param), resCtx_(resCtx)
{
}

HcclResult AllGatherBatchSmallCountExecutor::ValidateParam() const
{
    if (param_.itemCount == 0 || param_.itemCount > kAllGatherBatchMaxItems) {
        HCCL_ERROR("invalid itemCount=%u", param_.itemCount);
        return HCCL_E_PARA;
    }
    if (param_.resCtx == nullptr) {
        HCCL_ERROR("param.resCtx is null");
        return HCCL_E_PTR;
    }
    if (!IsValidCommMode(param_.commMode)) {
        HCCL_ERROR("commMode is invalid");
        return HCCL_E_INTERNAL;
    }
    if (resCtx_.threadHandle == 0) {
        HCCL_ERROR("threadHandle is invalid");
        return HCCL_E_INTERNAL;
    }
    if (resCtx_.localBuffer.addr == nullptr || resCtx_.localBuffer.size == 0) {
        HCCL_ERROR("localBuffer is not ready");
        return HCCL_E_INTERNAL;
    }
    if (GetPerRankWindowCapacity() == 0) {
        HCCL_ERROR("localBuffer capacity per rank is zero, localBufferSize=%llu, rankSize=%u",
            static_cast<unsigned long long>(resCtx_.localBuffer.size),
            param_.topoInfo.rankSize);
        return HCCL_E_INTERNAL;
    }
    if (param_.totalInputBytes == 0) {
        HCCL_ERROR("totalInputBytes is zero");
        return HCCL_E_PARA;
    }
    if (param_.windowBytes == 0) {
        HCCL_ERROR("windowBytes is zero");
        return HCCL_E_INTERNAL;
    }
    if (CountRecognizedProtocols(resCtx_) != resCtx_.channelCount) {
        HCCL_ERROR("recognized protocol count=%u does not match channelCount=%u",
            CountRecognizedProtocols(resCtx_),
            resCtx_.channelCount);
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

uint32_t AllGatherBatchSmallCountExecutor::CountCrossServerChannels() const
{
    return ops_hccl_allgatherbatch::CountCrossServerChannels(param_.topoInfo, resCtx_);
}

HcclResult AllGatherBatchSmallCountExecutor::ValidateModeConsistency() const
{
    const ResourceStats stats = CollectResourceStats(param_, resCtx_);

    // Host 已经把 commMode 和 rank 分布写进 OpParam，这里再在 Device 入口收一层，避免后续窗口循环建立在错误前提上。
    if (!HasConsistentRankDistribution(param_)) {
        HCCL_ERROR("rank distribution is inconsistent, commMode=%s, intra=%u, cross=%u, rankSize=%u",
            ToCommModeString(param_.commMode),
            param_.intraServerRankCount,
            param_.crossServerRankCount,
            param_.topoInfo.rankSize);
        return HCCL_E_INTERNAL;
    }
    if (stats.intraServerChannels + stats.crossServerChannels != resCtx_.channelCount) {
        HCCL_ERROR("channel scope split is inconsistent, intra=%u, cross=%u, channelCount=%u",
            stats.intraServerChannels,
            stats.crossServerChannels,
            resCtx_.channelCount);
        return HCCL_E_INTERNAL;
    }
    if (param_.commMode == BatchCommMode::kSingleServer && stats.crossServerChannels != 0) {
        HCCL_ERROR("single-server mode unexpectedly has cross-server channels=%u", stats.crossServerChannels);
        return HCCL_E_INTERNAL;
    }
    if (param_.commMode == BatchCommMode::kCrossServer && stats.crossServerChannels == 0) {
        HCCL_ERROR("cross-server mode has zero cross-server channels");
        return HCCL_E_INTERNAL;
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

uint8_t *AllGatherBatchSmallCountExecutor::GetRankWindowBase(const WindowRange &window, uint32_t rank) const
{
    return static_cast<uint8_t *>(resCtx_.localBuffer.addr) + (window.packedBytes * rank);
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
    // gathered 结果要按 rank 拆槽放回同一个 localBuffer，因此每轮窗口最多只能占用 localBuffer/rankSize。
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
    HCCL_CHK_RET(ValidateWindow(current));

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

HcclResult AllGatherBatchSmallCountExecutor::Pack(const WindowRange &window) const
{
    HCCL_CHK_RET(ValidateWindow(window));

    // Pack 现在把本 rank 的窗口打到 localBuffer 的“本 rank 槽位”，后续通信层会补齐其它 rank 的槽位。
    uint8_t *dst = GetRankWindowBase(window, param_.topoInfo.rank);
    uint64_t packedOffset = 0;
    uint32_t itemIdx = window.startItemIdx;
    uint64_t offsetBytes = window.startOffsetBytes;

    while (packedOffset < window.packedBytes && itemIdx < param_.itemCount) {
        const BatchItemParam &item = param_.items[itemIdx];
        const uint64_t itemRemaining = CalcRemainingBytes(item, offsetBytes);
        const uint64_t copyBytes = std::min(itemRemaining, window.packedBytes - packedOffset);
        const void *src = static_cast<const uint8_t *>(item.sendBuf) + offsetBytes;
        int32_t ret = HcommLocalCopyOnThread(resCtx_.threadHandle, dst + packedOffset, src, copyBytes);
        if (ret != HCCL_SUCCESS) {
            HCCL_ERROR("pack local copy failed, ret=%d", ret);
            return static_cast<HcclResult>(ret);
        }

        packedOffset += copyBytes;
        offsetBytes += copyBytes;
        HCCL_CHK_RET(AdvancePosition(itemIdx, offsetBytes));
    }

    if (packedOffset != window.packedBytes || itemIdx != window.endItemIdx || offsetBytes != window.endOffsetBytes) {
        HCCL_ERROR("pack window mismatch, packedOffset=%llu/%llu, end=(%u,%llu), expectedEnd=(%u,%llu)",
            static_cast<unsigned long long>(packedOffset),
            static_cast<unsigned long long>(window.packedBytes),
            itemIdx,
            static_cast<unsigned long long>(offsetBytes),
            window.endItemIdx,
            static_cast<unsigned long long>(window.endOffsetBytes));
        return HCCL_E_INTERNAL;
    }
    return HCCL_SUCCESS;
}

HcclResult AllGatherBatchSmallCountExecutor::Unpack(const WindowRange &window) const
{
    HCCL_CHK_RET(ValidateWindow(window));

    // gathered 结果以“rank 槽位 + 槽内 packed 顺序”存放在 localBuffer 中，这里再拆回每个 item 的 recvBuf。
    for (uint32_t rank = 0; rank < param_.topoInfo.rankSize; ++rank) {
        uint8_t *src = GetRankWindowBase(window, rank);
        uint64_t packedOffset = 0;
        uint32_t itemIdx = window.startItemIdx;
        uint64_t offsetBytes = window.startOffsetBytes;

        while (packedOffset < window.packedBytes && itemIdx < param_.itemCount) {
            const BatchItemParam &item = param_.items[itemIdx];
            const uint64_t itemRemaining = CalcRemainingBytes(item, offsetBytes);
            const uint64_t copyBytes = std::min(itemRemaining, window.packedBytes - packedOffset);
            uint8_t *dst = static_cast<uint8_t *>(item.recvBuf) + (rank * item.sendBytes) + offsetBytes;
            int32_t ret = HcommLocalCopyOnThread(resCtx_.threadHandle, dst, src + packedOffset, copyBytes);
            if (ret != HCCL_SUCCESS) {
                HCCL_ERROR("unpack local copy failed, rank=%u, ret=%d", rank, ret);
                return static_cast<HcclResult>(ret);
            }

            packedOffset += copyBytes;
            offsetBytes += copyBytes;
            HCCL_CHK_RET(AdvancePosition(itemIdx, offsetBytes));
        }

        if (packedOffset != window.packedBytes || itemIdx != window.endItemIdx || offsetBytes != window.endOffsetBytes) {
            HCCL_ERROR("unpack window mismatch, rank=%u, packedOffset=%llu/%llu, end=(%u,%llu), expectedEnd=(%u,%llu)",
                rank,
                static_cast<unsigned long long>(packedOffset),
                static_cast<unsigned long long>(window.packedBytes),
                itemIdx,
                static_cast<unsigned long long>(offsetBytes),
                window.endItemIdx,
                static_cast<unsigned long long>(window.endOffsetBytes));
            return HCCL_E_INTERNAL;
        }
    }
    return HCCL_SUCCESS;
}

HcclResult AllGatherBatchSmallCountExecutor::Orchestrate()
{
    HCCL_CHK_RET(ValidateParam());
    HCCL_CHK_RET(ValidateModeConsistency());

    WindowRange window;
    HCCL_CHK_RET(BuildFirstWindow(window));

    const uint32_t crossServerChannels = CountCrossServerChannels();
    const ResourceStats stats = CollectResourceStats(param_, resCtx_);
    HCCL_INFO("executor start: rank=%u, rankSize=%u, commMode=%s, windowScope=%s, intraServerRankCount=%u, crossServerRankCount=%u, perRankCapacity=%llu, maxWindowBytes=%llu, crossServerChannels=%u",
        param_.topoInfo.rank,
        param_.topoInfo.rankSize,
        ToCommModeString(param_.commMode),
        ToWindowScopeString(param_.commMode),
        param_.intraServerRankCount,
        param_.crossServerRankCount,
        static_cast<unsigned long long>(stats.perRankCapacity),
        static_cast<unsigned long long>(stats.maxWindowBytes),
        crossServerChannels);

    while (true) {
        HCCL_CHK_RET(ValidateWindow(window));
        HCCL_INFO("executor window ready: scope=%s, start=(%u,%llu), end=(%u,%llu), packedBytes=%llu, paramWindowBytes=%llu, perRankCapacity=%llu, maxWindowBytes=%llu, rankSize=%u",
            ToWindowScopeString(param_.commMode),
            window.startItemIdx,
            static_cast<unsigned long long>(window.startOffsetBytes),
            window.endItemIdx,
            static_cast<unsigned long long>(window.endOffsetBytes),
            static_cast<unsigned long long>(window.packedBytes),
            static_cast<unsigned long long>(param_.windowBytes),
            static_cast<unsigned long long>(GetPerRankWindowCapacity()),
            static_cast<unsigned long long>(GetMaxWindowBytes(param_, resCtx_)),
            param_.topoInfo.rankSize);

        HCCL_CHK_RET(Pack(window));

        // 这轮起，通信层按窗口大小真正收齐所有 rank 的槽位，再由 Unpack 拆回每个 item。
        AllGatherHDStageCore hdStageCore(param_, resCtx_, window.packedBytes);
        HcclResult commRet = hdStageCore.RunAsync();
        if (commRet != HCCL_SUCCESS) {
            return commRet;
        }

        HCCL_CHK_RET(Unpack(window));

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

