#include "exec_double_all_gather.h"

namespace ops_hccl_double_allgather {

static HcclResult CopyLocalSegment(AlgResourceCtx *resCtx, const GatherDesc &gather, uint32_t rank)
{
    const uint64_t segBytes = gather.count * gather.elemBytes;
    char *dst = static_cast<char *>(gather.outputPtr) + rank * segBytes;
    return static_cast<HcclResult>(HcommLocalCopyOnThread(resCtx->threadHandle, dst, gather.inputPtr, segBytes));
}

static HcclResult RunGatherStep(AlgResourceCtx *resCtx, const GatherDesc &gather, uint32_t rank, uint32_t rankSize, uint32_t step)
{
    const uint64_t segBytes = gather.count * gather.elemBytes;
    const uint32_t sendSegIdx = (rank + rankSize - step) % rankSize;
    const uint32_t recvSegIdx = (rank + rankSize - step - 1) % rankSize;

    char *outputBase = static_cast<char *>(gather.outputPtr);
    char *sendSrc = outputBase + static_cast<uint64_t>(sendSegIdx) * segBytes;
    char *recvDst = outputBase + static_cast<uint64_t>(recvSegIdx) * segBytes;

    CHK_RET(HcommLocalCopyOnThread(resCtx->threadHandle, resCtx->localBuffer.addr, sendSrc, segBytes));
    CHK_RET(HcommChannelNotifyRecordOnThread(resCtx->threadHandle, resCtx->nextChannel.handle, NOTIFY_IDX_READY));
    CHK_RET(HcommChannelNotifyWaitOnThread(resCtx->threadHandle, resCtx->prevChannel.handle, NOTIFY_IDX_READY, CUSTOM_TIMEOUT));
    CHK_RET(HcommReadOnThread(resCtx->threadHandle, resCtx->prevChannel.handle, recvDst, resCtx->prevChannel.remoteBuffer.addr, segBytes));
    CHK_RET(HcommChannelNotifyRecordOnThread(resCtx->threadHandle, resCtx->prevChannel.handle, NOTIFY_IDX_DONE));
    CHK_RET(HcommChannelNotifyWaitOnThread(resCtx->threadHandle, resCtx->nextChannel.handle, NOTIFY_IDX_DONE, CUSTOM_TIMEOUT));
    return HCCL_SUCCESS;
}

HcclResult ExecDoubleAllGather(DoubleAllGatherParam &param, AlgResourceCtx *resCtx)
{
    CHK_RET(CopyLocalSegment(resCtx, param.gather0, param.rank));
    CHK_RET(CopyLocalSegment(resCtx, param.gather1, param.rank));
    if (param.rankSize <= 1) {
        return HCCL_SUCCESS;
    }

    for (uint32_t step = 0; step < param.rankSize - 1; ++step) {
        CHK_RET(RunGatherStep(resCtx, param.gather0, param.rank, param.rankSize, step));
        CHK_RET(RunGatherStep(resCtx, param.gather1, param.rank, param.rankSize, step));
    }
    return HCCL_SUCCESS;
}

}
