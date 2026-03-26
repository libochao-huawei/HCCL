#include <array>
#include <cstdint>
#include <vector>
#include "exec_op.h"

namespace ops_hccl_allgather_2in2out {

namespace {

struct PeerSlice {
    uint32_t begin = 0;
    uint32_t end = 0;
};

inline uint8_t *BytePtr(void *ptr)
{
    return reinterpret_cast<uint8_t *>(ptr);
}

inline const uint8_t *BytePtr(const void *ptr)
{
    return reinterpret_cast<const uint8_t *>(ptr);
}

uint64_t CalcSliceOffsetBytes(const RouteParam &route, uint32_t rankId, uint64_t loopOffsetCount)
{
    return (static_cast<uint64_t>(rankId) * route.outputSliceStride + loopOffsetCount) * route.unitSize;
}

std::array<PeerSlice, kSubThreadNum> SplitPeersForSubThreads(uint32_t peerNum)
{
    std::array<PeerSlice, kSubThreadNum> ranges {};
    const uint32_t base = (kSubThreadNum == 0) ? 0 : (peerNum / kSubThreadNum);
    const uint32_t remain = (kSubThreadNum == 0) ? 0 : (peerNum % kSubThreadNum);

    uint32_t cursor = 0;
    for (uint32_t idx = 0; idx < kSubThreadNum; ++idx) {
        const uint32_t extra = (idx < remain) ? 1 : 0;
        ranges[idx].begin = cursor;
        ranges[idx].end = cursor + base + extra;
        cursor = ranges[idx].end;
    }
    return ranges;
}

HcclResult RoutePreCopy(ThreadHandle thread, const OpParam &param, const RouteParam &route, AlgResourceCtx *resCtx,
    uint64_t loopOffsetCount, uint64_t currentCount)
{
    const uint64_t bytes = currentCount * route.unitSize;
    CHK_PRT_RET(bytes > resCtx->localCclBuffer.size,
        HCCL_ERROR("[RoutePreCopy] route[%u] bytes[%llu] exceeds localCclBufferSize[%llu]",
            route.routeId,
            static_cast<unsigned long long>(bytes),
            static_cast<unsigned long long>(resCtx->localCclBuffer.size)),
        HCCL_E_INTERNAL);

    void *localScratch = resCtx->localCclBuffer.addr;
    const void *routeInput = BytePtr(route.inputPtr) + loopOffsetCount * route.unitSize;
    void *selfOutput = BytePtr(route.outputPtr) + CalcSliceOffsetBytes(route, param.rankId, loopOffsetCount);

    // 先把本地数据放到 ccl buffer，供其它 rank 通过 channel 读取。
    CHK_RET(HcommLocalCopyOnThread(thread, localScratch, routeInput, bytes));
    // 同时把自己的 slice 写到最终输出里的“自己槽位”。
    CHK_RET(HcommLocalCopyOnThread(thread, selfOutput, routeInput, bytes));
    return HCCL_SUCCESS;
}

HcclResult BroadcastAckToPeers(ThreadHandle thread, AlgResourceCtx *resCtx)
{
    for (uint32_t idx = 0; idx < resCtx->peerNum; ++idx) {
        const CommPeerRes &peer = resCtx->peerRes[idx];
        CHK_RET(HcommChannelNotifyRecordOnThread(thread, peer.channelHandle, peer.ackNotifyIdx));
    }
    return HCCL_SUCCESS;
}

HcclResult ReadPeerRange(ThreadHandle thread, const RouteParam &route, const CommPeerRes *peerRes,
    uint32_t begin, uint32_t end, uint64_t loopOffsetCount, uint64_t currentCount)
{
    const uint64_t bytes = currentCount * route.unitSize;
    for (uint32_t idx = begin; idx < end; ++idx) {
        const CommPeerRes &peer = peerRes[idx];
        void *dst = BytePtr(route.outputPtr) + CalcSliceOffsetBytes(route, peer.peerRank, loopOffsetCount);
        const void *src = peer.remoteBuffer.addr;

        // 先等对端确认“它的本地 ccl buffer 已经准备好”，再从 remote buffer 把该 peer 的 slice 读回来。
        CHK_RET(HcommChannelNotifyWaitOnThread(thread, peer.channelHandle, peer.ackNotifyIdx, kCustomTimeout));
        CHK_RET(HcommReadOnThread(thread, peer.channelHandle, dst, src, bytes));
        // 读完以后回一个 DATA_SIGNAL，告诉对端“我已经消费完你这一路这一轮的数据”。
        CHK_RET(HcommChannelNotifyRecordOnThread(thread, peer.channelHandle, peer.dataNotifyIdx));
    }
    return HCCL_SUCCESS;
}

HcclResult WaitPeersReadDone(ThreadHandle thread, AlgResourceCtx *resCtx)
{
    for (uint32_t idx = 0; idx < resCtx->peerNum; ++idx) {
        const CommPeerRes &peer = resCtx->peerRes[idx];
        // 只有等所有 peer 都回了 DATA_SIGNAL，当前 rank 才能安全复用自己的 local ccl buffer。
        CHK_RET(HcommChannelNotifyWaitOnThread(thread, peer.channelHandle, peer.dataNotifyIdx, kCustomTimeout));
    }
    return HCCL_SUCCESS;
}

HcclResult MainKickSubThreads(ThreadHandle mainThread, AlgResourceCtx *resCtx)
{
    for (uint32_t idx = 0; idx < kSubThreadNum; ++idx) {
        CHK_RET(HcommThreadNotifyRecordOnThread(mainThread,
            resCtx->threads[idx + 1],
            resCtx->threadSync.mainToSub[idx]));
    }
    return HCCL_SUCCESS;
}

HcclResult MainWaitSubThreads(ThreadHandle mainThread, AlgResourceCtx *resCtx)
{
    for (uint32_t idx = 0; idx < kSubThreadNum; ++idx) {
        CHK_RET(HcommThreadNotifyWaitOnThread(mainThread,
            resCtx->threadSync.subToMain[idx],
            kCustomTimeout));
    }
    return HCCL_SUCCESS;
}

HcclResult SubThreadsExchangePeers(const RouteParam &route, AlgResourceCtx *resCtx,
    uint64_t loopOffsetCount, uint64_t currentCount)
{
    const auto peerRanges = SplitPeersForSubThreads(resCtx->peerNum);

    for (uint32_t workerIdx = 0; workerIdx < kSubThreadNum; ++workerIdx) {
        ThreadHandle workerThread = resCtx->threads[workerIdx + 1];
        const PeerSlice &range = peerRanges[workerIdx];

        // 子线程先等待主线程放行，再处理自己负责的 peer 子集，最后通知主线程“我这半边做完了”。
        CHK_RET(HcommThreadNotifyWaitOnThread(workerThread,
            resCtx->threadSync.mainToSub[workerIdx],
            kCustomTimeout));
        CHK_RET(ReadPeerRange(workerThread, route, resCtx->peerRes,
            range.begin, range.end, loopOffsetCount, currentCount));
        CHK_RET(HcommThreadNotifyRecordOnThread(workerThread,
            resCtx->threads[0],
            resCtx->threadSync.subToMain[workerIdx]));
    }
    return HCCL_SUCCESS;
}

HcclResult RunRouteLoop(ThreadHandle mainThread, const OpParam &param, const RouteParam &route, AlgResourceCtx *resCtx)
{
    const uint64_t loopMaxCount = route.loopState.loopMaxCount;
    CHK_PRT_RET(loopMaxCount == 0,
        HCCL_ERROR("[RunRouteLoop] route[%u] loopMaxCount is zero", route.routeId), HCCL_E_PARA);

    for (uint64_t loopIdx = 0; loopIdx < route.loopState.totalLoopNum; ++loopIdx) {
        const uint64_t loopOffsetCount = loopIdx * loopMaxCount;
        const uint64_t remainCount = route.count - loopOffsetCount;
        const uint64_t currentCount = (remainCount < loopMaxCount) ? remainCount : loopMaxCount;

        HCCL_INFO("[RunRouteLoop] route[%u] loop[%llu/%llu] offsetCount[%llu] currentCount[%llu]",
            route.routeId,
            static_cast<unsigned long long>(loopIdx),
            static_cast<unsigned long long>(route.loopState.totalLoopNum),
            static_cast<unsigned long long>(loopOffsetCount),
            static_cast<unsigned long long>(currentCount));

        CHK_RET(RoutePreCopy(mainThread, param, route, resCtx, loopOffsetCount, currentCount));
        CHK_RET(BroadcastAckToPeers(mainThread, resCtx));
        CHK_RET(SubThreadsExchangePeers(route, resCtx, loopOffsetCount, currentCount));
        CHK_RET(MainKickSubThreads(mainThread, resCtx));
        CHK_RET(MainWaitSubThreads(mainThread, resCtx));
        CHK_RET(WaitPeersReadDone(mainThread, resCtx));
    }
    return HCCL_SUCCESS;
}

HcclResult RunRouteAllGather(ThreadHandle mainThread, const OpParam &param, const RouteParam &route, AlgResourceCtx *resCtx)
{
    HCCL_INFO("[RunRouteAllGather] route[%u] count[%llu] unitSize[%llu] loopMaxCount[%llu] totalLoopNum[%llu]",
        route.routeId,
        static_cast<unsigned long long>(route.count),
        static_cast<unsigned long long>(route.unitSize),
        static_cast<unsigned long long>(route.loopState.loopMaxCount),
        static_cast<unsigned long long>(route.loopState.totalLoopNum));

    if (route.count == 0) {
        return HCCL_SUCCESS;
    }
    return RunRouteLoop(mainThread, param, route, resCtx);
}

} // namespace

HcclResult ExecOp(OpParam &param, AlgResourceCtx *resCtx)
{
    CHK_PTR_NULL(resCtx);
    CHK_PRT_RET(resCtx->threadNum < kThreadNum || resCtx->threads[0] == nullptr ||
            resCtx->threads[1] == nullptr || resCtx->threads[2] == nullptr,
        HCCL_ERROR("[ExecOp] threads are not ready for stage6"), HCCL_E_PTR);

    ThreadHandle mainThread = resCtx->threads[0];

    // 第 6 阶段开始把 peer 交换拆给两个子线程：
    // mainThread 负责 pre-copy、发 ACK、等待子线程收敛、等待 peer 读完；
    // thread1/thread2 分别处理不同 peer 子集的 read + DATA_SIGNAL。
    CHK_RET(RunRouteAllGather(mainThread, param, param.routes[0], resCtx));
    CHK_RET(RunRouteAllGather(mainThread, param, param.routes[1], resCtx));
    return HCCL_SUCCESS;
}

} // namespace ops_hccl_allgather_2in2out
