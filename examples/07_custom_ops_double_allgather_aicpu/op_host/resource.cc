#include "resource.h"
#include "launch_kernel.h"

#include <vector>

namespace ops_hccl_double_allgather {

static HcclResult AcquireSingleChannel(HcclComm comm, uint32_t remoteRank, ChannelHandle &handle, CommBuffer &remoteBuffer)
{
    HcclChannelDesc channelDesc;
    CHK_RET(HcclChannelDescInit(&channelDesc, 1));
    channelDesc.remoteRank = remoteRank;
    channelDesc.channelProtocol = CommProtocol::COMM_PROTOCOL_HCCS;
    channelDesc.notifyNum = 2;
    CHK_RET(HcclChannelAcquire(comm, CommEngine::COMM_ENGINE_AICPU, &channelDesc, 1, &handle));
    CHK_RET(HcclChannelGetHcclBuffer(comm, handle, &remoteBuffer.addr, &remoteBuffer.size));
    return HCCL_SUCCESS;
}

HcclResult PrepareResources(HcclComm comm, DoubleAllGatherParam &param, aclrtStream stream)
{
    (void)stream;
    void *ctx = nullptr;
    uint64_t size = sizeof(AlgResourceCtx);
    if (HcclEngineCtxGet(comm, param.tag, CommEngine::COMM_ENGINE_AICPU, &ctx, &size) == HCCL_SUCCESS) {
        param.resCtx = static_cast<AlgResourceCtx *>(ctx);
        return HCCL_SUCCESS;
    }

    CHK_RET(HcclEngineCtxCreate(comm, param.tag, CommEngine::COMM_ENGINE_AICPU, sizeof(AlgResourceCtx), &ctx));
    param.resCtx = static_cast<AlgResourceCtx *>(ctx);

    AlgResourceCtx hostCtx = {};
    ACLCHECK(aclrtCreateNotify(&(g_notifies[0]), ACL_NOTIFY_DEFAULT));
    ACLCHECK(aclrtCreateNotify(&(g_notifies[1]), ACL_NOTIFY_DEFAULT));
    for (uint32_t idx = 0; idx < AICPU_CONTROL_NOTIFY_NUM; ++idx) {
        ACLCHECK(aclrtGetNotifyId(g_notifies[idx], &(hostCtx.notifyIds[idx])));
    }

    CHK_RET(HcclThreadAcquire(comm, CommEngine::COMM_ENGINE_AICPU, 1, 0, &(hostCtx.threadHandle)));
    CHK_RET(HcclGetHcclBuffer(comm, &(hostCtx.localBuffer.addr), &(hostCtx.localBuffer.size)));

    hostCtx.prevRank = (param.rank + param.rankSize - 1) % param.rankSize;
    hostCtx.nextRank = (param.rank + 1) % param.rankSize;
    hostCtx.prevChannel.remoteRank = hostCtx.prevRank;
    hostCtx.nextChannel.remoteRank = hostCtx.nextRank;

    if (param.rankSize > 1) {
        if (hostCtx.prevRank == hostCtx.nextRank) {
            CHK_RET(AcquireSingleChannel(comm, hostCtx.nextRank, hostCtx.nextChannel.handle, hostCtx.nextChannel.remoteBuffer));
            hostCtx.prevChannel.handle = hostCtx.nextChannel.handle;
            hostCtx.prevChannel.remoteBuffer = hostCtx.nextChannel.remoteBuffer;
        } else {
            CHK_RET(AcquireSingleChannel(comm, hostCtx.prevRank, hostCtx.prevChannel.handle, hostCtx.prevChannel.remoteBuffer));
            CHK_RET(AcquireSingleChannel(comm, hostCtx.nextRank, hostCtx.nextChannel.handle, hostCtx.nextChannel.remoteBuffer));
        }
    }

    ACLCHECK(aclrtMemcpy(param.resCtx, sizeof(AlgResourceCtx), &hostCtx, sizeof(AlgResourceCtx), ACL_MEMCPY_HOST_TO_DEVICE));
    return HCCL_SUCCESS;
}

}
