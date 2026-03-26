#include <cstddef>
#include <cstdio>
#include <cstring>
#include <vector>
#include <hccl_rank_graph.h>
#include "resource.h"
#include "launch_kernel.h"

namespace ops_hccl_allgather_2in2out {

namespace {

constexpr CommEngine kCustomEngine = CommEngine::COMM_ENGINE_AICPU_TS;
constexpr uint32_t kThreadNotifyNumPerThread = 4;
constexpr uint32_t kChannelNotifyNum = 2;

uint64_t CalcResCtxSize(uint32_t peerNum)
{
    return sizeof(AlgResourceCtx) + sizeof(CommPeerRes) * peerNum;
}

HcclResult BuildCtxTag(const CommMeta &meta, char *tag, size_t tagSize)
{
    int ret = std::snprintf(tag, tagSize, "AllGather2In2Out_%s_AICPU", meta.commName);
    CHK_PRT_RET(ret <= 0 || static_cast<size_t>(ret) >= tagSize,
        HCCL_ERROR("[BuildCtxTag] failed to build ctx tag, commName[%s]", meta.commName), HCCL_E_INTERNAL);
    return HCCL_SUCCESS;
}

HcclResult FillThreadSyncRes(ThreadSyncRes &syncRes)
{
    // 这四个索引是后续 main/sub thread 做握手时要用的 notify 槽位约定。
    syncRes.mainToSub[0] = 0;
    syncRes.subToMain[0] = 1;
    syncRes.mainToSub[1] = 2;
    syncRes.subToMain[1] = 3;
    return HCCL_SUCCESS;
}

HcclResult FillHostControlNotifyIds(AlgResourceCtx &resCtx)
{
    CHK_RET(EnsureHostControlNotifiesCreated());
    for (uint32_t idx = 0; idx < kControlNotifyNum; ++idx) {
        ACLCHECK(aclrtGetNotifyId(g_notifies[idx], &resCtx.notifyIds[idx]));
    }
    return HCCL_SUCCESS;
}

HcclResult UpdateDeviceNotifyIds(AlgResourceCtx *resCtxDevice)
{
    AlgResourceCtx notifyShadow {};
    CHK_RET(FillHostControlNotifyIds(notifyShadow));

    void *dst = reinterpret_cast<uint8_t *>(resCtxDevice) + offsetof(AlgResourceCtx, notifyIds);
    ACLCHECK(aclrtMemcpy(dst,
        sizeof(notifyShadow.notifyIds),
        notifyShadow.notifyIds,
        sizeof(notifyShadow.notifyIds),
        ACL_MEMCPY_HOST_TO_DEVICE));
    return HCCL_SUCCESS;
}

HcclResult BuildChannelDescByPeer(HcclComm comm, uint32_t myRank, uint32_t peerRank, HcclChannelDesc &desc)
{
    // 这里不硬编码协议，而是从 rank graph 中拿真实链路描述，尽量兼容单机和超节点内多机场景。
    CHK_RET(HcclChannelDescInit(&desc, 1));
    desc.remoteRank = peerRank;
    desc.notifyNum = kChannelNotifyNum;

    uint32_t *netLayers = nullptr;
    uint32_t netLayerNum = 0;
    CHK_RET(HcclRankGraphGetLayers(comm, &netLayers, &netLayerNum));

    for (uint32_t idx = 0; idx < netLayerNum; ++idx) {
        CommLink *linkList = nullptr;
        uint32_t listSize = 0;
        CHK_RET(HcclRankGraphGetLinks(comm, netLayers[idx], myRank, peerRank, &linkList, &listSize));
        if (listSize == 0) {
            continue;
        }

        // 首版每个 peer 只挑第一条可用链路，先把资源模型跑通。
        const CommLink &link = linkList[0];
        desc.localEndpoint.protocol = link.srcEndpointDesc.protocol;
        desc.localEndpoint.commAddr = link.srcEndpointDesc.commAddr;
        desc.localEndpoint.loc = link.srcEndpointDesc.loc;
        desc.remoteEndpoint.protocol = link.dstEndpointDesc.protocol;
        desc.remoteEndpoint.commAddr = link.dstEndpointDesc.commAddr;
        desc.remoteEndpoint.loc = link.dstEndpointDesc.loc;
        desc.channelProtocol = link.linkAttr.linkProtocol;
        return HCCL_SUCCESS;
    }

    HCCL_ERROR("[BuildChannelDescByPeer] no link found between rank[%u] and peer[%u]", myRank, peerRank);
    return HCCL_E_NOT_FOUND;
}

HcclResult PreparePeerChannels(HcclComm comm,
    const CommMeta &meta,
    const std::vector<uint32_t> &peers,
    AlgResourceCtx &resCtx,
    CommPeerRes *peerResHost)
{
    if (peers.empty()) {
        resCtx.peerNum = 0;
        return HCCL_SUCCESS;
    }

    std::vector<HcclChannelDesc> channelDescs(peers.size());
    std::vector<ChannelHandle> channelHandles(peers.size(), nullptr);
    for (size_t idx = 0; idx < peers.size(); ++idx) {
        CHK_RET(BuildChannelDescByPeer(comm, meta.rankId, peers[idx], channelDescs[idx]));
    }

    CHK_RET(HcclChannelAcquire(comm, kCustomEngine,
        channelDescs.data(), static_cast<uint32_t>(channelDescs.size()), channelHandles.data()));

    for (size_t idx = 0; idx < peers.size(); ++idx) {
        peerResHost[idx].peerRank = peers[idx];
        peerResHost[idx].channelHandle = channelHandles[idx];
        peerResHost[idx].ackNotifyIdx = kNotifyAck;
        peerResHost[idx].dataNotifyIdx = kNotifyData;
        CHK_RET(HcclChannelGetHcclBuffer(comm, channelHandles[idx],
            &peerResHost[idx].remoteBuffer.addr,
            &peerResHost[idx].remoteBuffer.size));
    }

    resCtx.peerNum = static_cast<uint32_t>(peers.size());
    return HCCL_SUCCESS;
}

HcclResult PrepareResourceCtx(HcclComm comm,
    const CommMeta &meta,
    const std::vector<uint32_t> &peers,
    const char *ctxTag,
    AlgResourceCtx *&resCtxDevice)
{
    uint64_t ctxSize = CalcResCtxSize(static_cast<uint32_t>(peers.size()));
    void *ctx = nullptr;
    if (HcclEngineCtxGet(comm, ctxTag, kCustomEngine, &ctx, &ctxSize) == HCCL_SUCCESS && ctx != nullptr) {
        resCtxDevice = static_cast<AlgResourceCtx *>(ctx);
        // 即使 device context 已经存在，也要把本次 Host 创建出来的 notify id 刷进去。
        CHK_RET(UpdateDeviceNotifyIds(resCtxDevice));
        return HCCL_SUCCESS;
    }

    CHK_RET(HcclEngineCtxCreate(comm, ctxTag, kCustomEngine, ctxSize, &ctx));
    resCtxDevice = static_cast<AlgResourceCtx *>(ctx);

    const size_t alignSize = sizeof(std::max_align_t);
    const size_t alignCount = static_cast<size_t>((ctxSize + alignSize - 1) / alignSize);
    std::vector<std::max_align_t> hostCtxBuffer(alignCount);
    std::memset(hostCtxBuffer.data(), 0, alignCount * alignSize);

    auto *resCtxHost = reinterpret_cast<AlgResourceCtx *>(hostCtxBuffer.data());
    auto *peerResHost = reinterpret_cast<CommPeerRes *>(reinterpret_cast<uint8_t *>(resCtxHost) + sizeof(AlgResourceCtx));

    // 注意这里写入的是 device context 中 peerRes 数组的地址，而不是 host 临时 buffer 的地址。
    resCtxHost->peerRes = reinterpret_cast<CommPeerRes *>(reinterpret_cast<uint8_t *>(ctx) + sizeof(AlgResourceCtx));
    resCtxHost->threadNum = kThreadNum;
    CHK_RET(FillThreadSyncRes(resCtxHost->threadSync));
    CHK_RET(FillHostControlNotifyIds(*resCtxHost));
    CHK_RET(HcclThreadAcquire(comm, kCustomEngine, kThreadNum, kThreadNotifyNumPerThread, resCtxHost->threads));
    CHK_RET(HcclGetHcclBuffer(comm, &resCtxHost->localCclBuffer.addr, &resCtxHost->localCclBuffer.size));
    CHK_RET(PreparePeerChannels(comm, meta, peers, *resCtxHost, peerResHost));

    ACLCHECK(aclrtMemcpy(ctx, ctxSize, resCtxHost, ctxSize, ACL_MEMCPY_HOST_TO_DEVICE));
    return HCCL_SUCCESS;
}

void FillRouteParam(RouteParam &route,
    void *inputPtr,
    void *outputPtr,
    uint64_t count,
    uint64_t loopMaxCount,
    uint32_t routeId,
    HcclDataType dataType)
{
    route.inputPtr = inputPtr;
    route.outputPtr = outputPtr;
    route.count = count;
    route.unitSize = SIZE_TABLE[dataType];
    route.totalBytes = route.unitSize * count;
    route.inputSliceStride = count;
    route.outputSliceStride = count;
    route.loopState.loopMaxCount = loopMaxCount;
    route.loopState.totalLoopNum = (loopMaxCount == 0) ? 0 : ((count + loopMaxCount - 1) / loopMaxCount);
    route.routeId = routeId;
}

} // namespace

HcclResult BuildFusedOpParam(
    void *sendBuf0,
    void *sendBuf1,
    void *recvBuf0,
    void *recvBuf1,
    uint64_t sendCount0,
    uint64_t sendCount1,
    HcclDataType dataType,
    const SmallCountDecision &decision,
    const CommMeta &meta,
    const std::vector<uint32_t> &peers,
    HcclComm comm,
    OpParam &param)
{
    char ctxTag[kTagLength] = {0};
    CHK_RET(BuildCtxTag(meta, ctxTag, sizeof(ctxTag)));
    AlgResourceCtx *resCtxDevice = nullptr;
    CHK_RET(PrepareResourceCtx(comm, meta, peers, ctxTag, resCtxDevice));

    std::memset(&param, 0, sizeof(param));
    int ret = std::snprintf(param.tag, sizeof(param.tag), "%s", ctxTag);
    CHK_PRT_RET(ret <= 0 || static_cast<size_t>(ret) >= sizeof(param.tag),
        HCCL_ERROR("[BuildFusedOpParam] failed to fill tag"), HCCL_E_INTERNAL);
    ret = std::snprintf(param.commName, sizeof(param.commName), "%s", meta.commName);
    CHK_PRT_RET(ret <= 0 || static_cast<size_t>(ret) >= sizeof(param.commName),
        HCCL_ERROR("[BuildFusedOpParam] failed to fill commName"), HCCL_E_INTERNAL);

    param.dataType = dataType;
    param.rankId = meta.rankId;
    param.rankSize = meta.rankSize;
    param.deviceType = meta.deviceType;
    param.workflowMode = meta.workflowMode;
    param.topologyType = meta.topologyType;
    param.pathType = PATH_FUSED_SMALLCOUNT;
    param.resCtx = resCtxDevice;

    FillRouteParam(param.routes[0], sendBuf0, recvBuf0, sendCount0, decision.loopMaxCount, 0, dataType);
    FillRouteParam(param.routes[1], sendBuf1, recvBuf1, sendCount1, decision.loopMaxCount, 1, dataType);

    void *localCclBufferAddr = nullptr;
    uint64_t localCclBufferSize = 0;
    CHK_RET(HcclGetHcclBuffer(comm, &localCclBufferAddr, &localCclBufferSize));
    (void)localCclBufferAddr;

    HCCL_INFO("[BuildFusedOpParam] rank[%u] peerNum[%u] threadNum[%u] localCclBufferSize[%llu] route0Count[%llu] route1Count[%llu]",
        meta.rankId,
        static_cast<uint32_t>(peers.size()),
        kThreadNum,
        static_cast<unsigned long long>(localCclBufferSize),
        static_cast<unsigned long long>(param.routes[0].count),
        static_cast<unsigned long long>(param.routes[1].count));
    return HCCL_SUCCESS;
}

} // namespace ops_hccl_allgather_2in2out
