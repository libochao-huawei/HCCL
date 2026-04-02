#include "allgather_batch.h"

#include <cstdio>
#include <memory>
#include <vector>

#include "common.h"
#include "hccl/hccl_rank_graph.h"
#include "launch_kernel.h"
#include "load_kernel.h"
#include "log.h"
#include "resource_request.h"

namespace ops_hccl_allgatherbatch {

namespace {

struct ChannelLinkInfo {
    CommProtocol protocol = COMM_PROTOCOL_RESERVED;
    uint32_t localServerIdx = 0;
    uint32_t localSuperPodIdx = 0;
    uint32_t remoteServerIdx = 0;
    uint32_t remoteSuperPodIdx = 0;
};

HcclResult EnsureControlNotifies(AlgResourceCtx &resCtx)
{
    // Host 侧保留两类控制 notify：启动 device 执行、等待 device 完成。
    for (uint32_t idx = 0; idx < kAllGatherBatchControlNotifyNum; ++idx) {
        if (g_allGatherBatchNotifies[idx] == nullptr) {
            ACL_CHK(aclrtCreateNotify(&g_allGatherBatchNotifies[idx], ACL_NOTIFY_DEFAULT));
        }
        ACL_CHK(aclrtGetNotifyId(g_allGatherBatchNotifies[idx], &resCtx.controlNotifyIds[idx]));
    }
    return HCCL_SUCCESS;
}

HcclResult QueryServerInstSizeList(HcclComm comm, uint32_t **instSizeList, uint32_t *instListSize)
{
    HCCL_CHK_PTR(instSizeList);
    HCCL_CHK_PTR(instListSize);
    return HcclRankGraphGetInstSizeListByLayer(comm, 0, instSizeList, instListSize);
}

HcclResult QueryNetLayers(HcclComm comm, uint32_t **netLayers, uint32_t *netLayerNum)
{
    HCCL_CHK_PTR(netLayers);
    HCCL_CHK_PTR(netLayerNum);
    return HcclRankGraphGetLayers(comm, netLayers, netLayerNum);
}

HcclResult QueryChannelLinkInfo(HcclComm comm, uint32_t localRank, uint32_t remoteRank, ChannelLinkInfo &info)
{
    uint32_t *netLayers = nullptr;
    uint32_t netLayerNum = 0;
    HCCL_CHK_RET(QueryNetLayers(comm, &netLayers, &netLayerNum));

    for (uint32_t idx = 0; idx < netLayerNum; ++idx) {
        CommLink *links = nullptr;
        uint32_t linkNum = 0;
        HcclResult ret = HcclRankGraphGetLinks(comm, netLayers[idx], localRank, remoteRank, &links, &linkNum);
        if (ret == HCCL_SUCCESS && links != nullptr && linkNum > 0) {
            info.protocol = links[0].linkAttr.linkProtocol;
            info.localServerIdx = links[0].srcEndpointDesc.loc.device.serverIdx;
            info.localSuperPodIdx = links[0].srcEndpointDesc.loc.device.superPodIdx;
            info.remoteServerIdx = links[0].dstEndpointDesc.loc.device.serverIdx;
            info.remoteSuperPodIdx = links[0].dstEndpointDesc.loc.device.superPodIdx;
            return HCCL_SUCCESS;
        }
    }

    HCCL_ERROR("failed to query link info between rank=%u and remoteRank=%u", localRank, remoteRank);
    return HCCL_E_NOT_FOUND;
}

HcclResult FillServerGroupInfo(HcclComm comm, BatchTopoInfo &topoInfo)
{
    uint32_t *instSizeList = nullptr;
    uint32_t instListSize = 0;
    HcclResult ret = QueryServerInstSizeList(comm, &instSizeList, &instListSize);
    if (ret != HCCL_SUCCESS || instSizeList == nullptr || instListSize == 0) {
        topoInfo.serverCount = 1;
        topoInfo.serverIdx = 0;
        return HCCL_SUCCESS;
    }

    topoInfo.serverCount = instListSize;
    uint32_t rankBase = 0;
    for (uint32_t serverIdx = 0; serverIdx < instListSize; ++serverIdx) {
        const uint32_t rankEnd = rankBase + instSizeList[serverIdx];
        if (topoInfo.rank < rankEnd) {
            topoInfo.serverIdx = serverIdx;
            return HCCL_SUCCESS;
        }
        rankBase = rankEnd;
    }

    topoInfo.serverIdx = 0;
    return HCCL_SUCCESS;
}

HcclResult FillEndpointLocation(HcclComm comm, BatchTopoInfo &topoInfo)
{
    uint32_t *serverRanks = nullptr;
    uint32_t serverRankNum = 0;
    HcclResult ret = HcclRankGraphGetRanksByLayer(comm, 0, &serverRanks, &serverRankNum);
    if (ret != HCCL_SUCCESS || serverRanks == nullptr || serverRankNum == 0) {
        return HCCL_SUCCESS;
    }

    uint32_t peerRank = topoInfo.rank;
    for (uint32_t idx = 0; idx < serverRankNum; ++idx) {
        if (serverRanks[idx] != topoInfo.rank) {
            peerRank = serverRanks[idx];
            break;
        }
    }
    if (peerRank == topoInfo.rank && topoInfo.rankSize > 1) {
        peerRank = (topoInfo.rank == 0) ? 1 : 0;
    }
    if (peerRank == topoInfo.rank) {
        return HCCL_SUCCESS;
    }

    ChannelLinkInfo linkInfo;
    ret = QueryChannelLinkInfo(comm, topoInfo.rank, peerRank, linkInfo);
    if (ret != HCCL_SUCCESS) {
        return HCCL_SUCCESS;
    }

    topoInfo.serverIdx = linkInfo.localServerIdx;
    topoInfo.superPodIdx = linkInfo.localSuperPodIdx;
    return HCCL_SUCCESS;
}

BatchCommMode DetermineCommMode(const BatchTopoInfo &topoInfo)
{
    return (topoInfo.serverCount > 1U) ? BatchCommMode::kCrossServer : BatchCommMode::kSingleServer;
}

HcclResult QueryLocalServerRankCount(HcclComm comm, const BatchTopoInfo &topoInfo, uint32_t &localServerRankCount)
{
    uint32_t *instSizeList = nullptr;
    uint32_t instListSize = 0;
    HcclResult ret = QueryServerInstSizeList(comm, &instSizeList, &instListSize);
    if (ret != HCCL_SUCCESS || instSizeList == nullptr || instListSize == 0) {
        localServerRankCount = topoInfo.rankSize;
        return HCCL_SUCCESS;
    }
    if (topoInfo.serverIdx >= instListSize) {
        HCCL_ERROR("serverIdx=%u is out of range for instListSize=%u", topoInfo.serverIdx, instListSize);
        return HCCL_E_INTERNAL;
    }

    uint64_t totalRanks = 0;
    for (uint32_t idx = 0; idx < instListSize; ++idx) {
        totalRanks += instSizeList[idx];
    }
    if (totalRanks != topoInfo.rankSize) {
        HCCL_ERROR("rank graph inst size sum=%llu mismatches rankSize=%u",
            static_cast<unsigned long long>(totalRanks),
            topoInfo.rankSize);
        return HCCL_E_INTERNAL;
    }

    localServerRankCount = instSizeList[topoInfo.serverIdx];
    if (localServerRankCount == 0 || localServerRankCount > topoInfo.rankSize) {
        HCCL_ERROR("local server rank count=%u is invalid, rankSize=%u", localServerRankCount, topoInfo.rankSize);
        return HCCL_E_INTERNAL;
    }
    return HCCL_SUCCESS;
}

HcclResult FillCommModeInfo(HcclComm comm, OpParam &param)
{
    param.commMode = DetermineCommMode(param.topoInfo);
    HCCL_CHK_RET(QueryLocalServerRankCount(comm, param.topoInfo, param.intraServerRankCount));
    if (param.intraServerRankCount > param.topoInfo.rankSize) {
        HCCL_ERROR("intraServerRankCount=%u exceeds rankSize=%u",
            param.intraServerRankCount,
            param.topoInfo.rankSize);
        return HCCL_E_INTERNAL;
    }

    param.crossServerRankCount = param.topoInfo.rankSize - param.intraServerRankCount;
    if (param.commMode == BatchCommMode::kSingleServer) {
        param.intraServerRankCount = param.topoInfo.rankSize;
        param.crossServerRankCount = 0;
    }
    return HCCL_SUCCESS;
}

HcclResult ValidateItems(const HcclAllGatherItem *items, uint32_t itemCount, HcclComm comm, aclrtStream stream)
{
    HCCL_CHK_PTR(items);
    HCCL_CHK_PTR(comm);
    HCCL_CHK_PTR(stream);
    if (itemCount == 0 || itemCount > kAllGatherBatchMaxItems) {
        HCCL_ERROR("invalid itemCount=%u", itemCount);
        return HCCL_E_PARA;
    }

    for (uint32_t idx = 0; idx < itemCount; ++idx) {
        HCCL_CHK_PTR(items[idx].sendBuf);
        if (!IsAligned32(items[idx].sendBuf)) {
            HCCL_ERROR("item %u sendBuf is not 32B aligned", idx);
            return HCCL_E_PARA;
        }
        HCCL_CHK_PTR(items[idx].recvBuf);
        if (items[idx].sendCount == 0) {
            HCCL_ERROR("item %u sendCount is zero", idx);
            return HCCL_E_PARA;
        }
        if (!IsSupportedDataType(items[idx].dataType)) {
            HCCL_ERROR("item %u dataType=%d is unsupported", idx, static_cast<int>(items[idx].dataType));
            return HCCL_E_NOT_SUPPORT;
        }
    }
    return HCCL_SUCCESS;
}

HcclResult ValidateTopo(const BatchTopoInfo &topoInfo)
{
    if (topoInfo.rankSize == 0) {
        HCCL_ERROR("rankSize is zero after topo preparation");
        return HCCL_E_PARA;
    }
    if (topoInfo.serverCount == 0) {
        HCCL_ERROR("serverCount is zero after topo preparation");
        return HCCL_E_INTERNAL;
    }
    if (topoInfo.serverCount > topoInfo.rankSize) {
        HCCL_ERROR("serverCount=%u exceeds rankSize=%u", topoInfo.serverCount, topoInfo.rankSize);
        return HCCL_E_INTERNAL;
    }
    if (topoInfo.serverIdx >= topoInfo.serverCount) {
        HCCL_ERROR("serverIdx=%u is out of range, serverCount=%u", topoInfo.serverIdx, topoInfo.serverCount);
        return HCCL_E_INTERNAL;
    }
    return HCCL_SUCCESS;
}

HcclResult PrepareTopoInfo(HcclComm comm, BatchTopoInfo &topoInfo)
{
    HCCL_CHK_RET(HcclGetRankId(comm, &topoInfo.rank));
    HCCL_CHK_RET(HcclGetRankSize(comm, &topoInfo.rankSize));

    topoInfo.serverCount = 1;
    topoInfo.serverIdx = 0;
    topoInfo.superPodIdx = 0;

    HCCL_CHK_RET(FillServerGroupInfo(comm, topoInfo));
    HCCL_CHK_RET(FillEndpointLocation(comm, topoInfo));
    return HCCL_SUCCESS;
}

HcclResult PrepareOpParam(const HcclAllGatherItem *items, uint32_t itemCount, HcclComm comm, OpParam &param)
{
    std::snprintf(param.tag, sizeof(param.tag), "%s", kAllGatherBatchCtxTag);
    HCCL_CHK_RET(HcclGetCommName(comm, param.commName));
    HCCL_CHK_RET(PrepareTopoInfo(comm, param.topoInfo));
    HCCL_CHK_RET(FillCommModeInfo(comm, param));

    param.itemCount = itemCount;
    param.totalInputBytes = 0;
    param.totalOutputBytes = 0;

    for (uint32_t idx = 0; idx < itemCount; ++idx) {
        BatchItemParam &itemParam = param.items[idx];
        itemParam.sendBuf = items[idx].sendBuf;
        itemParam.recvBuf = items[idx].recvBuf;
        itemParam.sendCount = items[idx].sendCount;
        itemParam.dataType = items[idx].dataType;
        itemParam.elementSize = GetDataTypeSize(items[idx].dataType);
        itemParam.sendBytes = itemParam.sendCount * itemParam.elementSize;

        param.totalInputBytes += itemParam.sendBytes;
        param.totalOutputBytes += itemParam.sendBytes * param.topoInfo.rankSize;
    }

    param.windowBytes = param.totalInputBytes;
    return HCCL_SUCCESS;
}

HcclResult ValidatePreparedParam(const OpParam &param)
{
    return ValidateBasicOpParam(param, "prepared param");
}

HcclResult ValidatePreparedResourceCtx(const OpParam &param)
{
    HCCL_CHK_PTR(param.resCtx);
    const AlgResourceCtx &resCtx = *param.resCtx;

    HCCL_CHK_RET(ValidateBasicResourceCtx(param, resCtx, "prepared resCtx"));
    return ValidateRemoteChannelResources(param, resCtx, "prepared");
}

// 资源请求阶段先把“需要和哪些 rank 建链”算清楚，再交给 GetAlgRes 真正申请资源。
HcclResult CalcFullMeshResourceRequest(HcclComm comm, const OpParam &param, BatchResourceRequest &request)
{
    request.threadNum = 1;
    request.controlNotifyNum = kAllGatherBatchControlNotifyNum;
    request.localBufferBytes = 0;
    request.commMode = param.commMode;
    request.channelCount = GetExpectedFullMeshChannelCount(param);
    request.channels.clear();
    request.channels.reserve(request.channelCount);

    for (uint32_t remoteRank = 0; remoteRank < param.topoInfo.rankSize; ++remoteRank) {
        if (remoteRank == param.topoInfo.rank) {
            continue;
        }

        ChannelLinkInfo linkInfo;
        HCCL_CHK_RET(QueryChannelLinkInfo(comm, param.topoInfo.rank, remoteRank, linkInfo));
        if (linkInfo.localSuperPodIdx != linkInfo.remoteSuperPodIdx) {
            HCCL_ERROR("cross-superpod link is not supported, rank=%u(superPod=%u) remoteRank=%u(superPod=%u)",
                param.topoInfo.rank,
                linkInfo.localSuperPodIdx,
                remoteRank,
                linkInfo.remoteSuperPodIdx);
            return HCCL_E_NOT_SUPPORT;
        }

        ChannelRequest channelRequest;
        channelRequest.remoteRank = remoteRank;
        channelRequest.remoteServerIdx = linkInfo.remoteServerIdx;
        channelRequest.remoteSuperPodIdx = linkInfo.remoteSuperPodIdx;
        channelRequest.protocol = linkInfo.protocol;
        channelRequest.notifyNum = 2;
        request.channels.push_back(channelRequest);
    }

    request.channelCount = static_cast<uint32_t>(request.channels.size());
    if (request.channelCount != GetExpectedFullMeshChannelCount(param)) {
        HCCL_ERROR("fullmesh request channelCount mismatch, actual=%u, expected=%u",
            request.channelCount,
            GetExpectedFullMeshChannelCount(param));
        return HCCL_E_INTERNAL;
    }
    return HCCL_SUCCESS;
}

// context 大小由真实 channel 数决定，避免固定 32 条 channel 的静态上限。
uint64_t CalcAlgResourceCtxSize(const BatchResourceRequest &request)
{
    return sizeof(AlgResourceCtx) +
        static_cast<uint64_t>(request.channelCount) * sizeof(ChannelResource);
}

// 先初始化固定头，尾部 channel 区由 AllocAlgResource 继续填充。
void InitAlgResourceCtxHeader(const BatchResourceRequest &request, AlgResourceCtx &resCtx)
{
    resCtx.channelCount = request.channelCount;
    resCtx.channelOffset = sizeof(AlgResourceCtx);
}

// 真正的资源申请只消费 request，不再自己重算拓扑和 remote rank 集合。
HcclResult AllocAlgResource(
    HcclComm comm,
    const OpParam &param,
    const BatchResourceRequest &request,
    AlgResourceCtx &resCtx)
{
    void *localBuffer = nullptr;
    HCCL_CHK_RET(EnsureControlNotifies(resCtx));
    HCCL_CHK_RET(HcclThreadAcquire(comm, COMM_ENGINE_AICPU, request.threadNum, 0, &resCtx.threadHandle));
    HCCL_CHK_RET(HcclGetHcclBuffer(comm, &localBuffer, &resCtx.localBuffer.size));
    resCtx.localBuffer.addr = localBuffer;

    if (request.channelCount == 0) {
        return HCCL_SUCCESS;
    }

    std::vector<HcclChannelDesc> channelDescs(request.channelCount);
    std::vector<ChannelHandle> channelHandles(request.channelCount, 0);
    HcclChannelDescInit(channelDescs.data(), request.channelCount);

    for (uint32_t idx = 0; idx < request.channelCount; ++idx) {
        channelDescs[idx].remoteRank = request.channels[idx].remoteRank;
        channelDescs[idx].channelProtocol = request.channels[idx].protocol;
        channelDescs[idx].notifyNum = request.channels[idx].notifyNum;
    }

    HCCL_CHK_RET(HcclChannelAcquire(
        comm,
        COMM_ENGINE_AICPU,
        channelDescs.data(),
        request.channelCount,
        channelHandles.data()));

    ChannelResource *channels = GetChannelArray(resCtx);
    for (uint32_t idx = 0; idx < request.channelCount; ++idx) {
        channels[idx].handle = channelHandles[idx];
        channels[idx].remoteRank = request.channels[idx].remoteRank;
        channels[idx].remoteServerIdx = request.channels[idx].remoteServerIdx;
        channels[idx].remoteSuperPodIdx = request.channels[idx].remoteSuperPodIdx;
        channels[idx].protocol = request.channels[idx].protocol;
        channels[idx].localNotifyIdx = 0;
        channels[idx].remoteNotifyIdx = 0;
        HCCL_CHK_RET(HcclChannelGetHcclBuffer(
            comm,
            channelHandles[idx],
            &channels[idx].remoteBuffer.addr,
            &channels[idx].remoteBuffer.size));
    }

    const uint32_t expectedChannelCount = GetExpectedFullMeshChannelCount(param);
    if (resCtx.channelCount != expectedChannelCount) {
        HCCL_ERROR("allocated channelCount mismatch, actual=%u, expected=%u",
            resCtx.channelCount,
            expectedChannelCount);
        return HCCL_E_INTERNAL;
    }
    return HCCL_SUCCESS;
}

// Host 侧资源链现在固定为：CalcFullMeshResourceRequest -> 动态 ctxSize -> AllocAlgResource。
HcclResult GetAlgRes(HcclComm comm, const OpParam &param, AlgResourceCtx **resCtx)
{
    HCCL_CHK_PTR(resCtx);

    BatchResourceRequest request;
    HCCL_CHK_RET(CalcFullMeshResourceRequest(comm, param, request));

    void *ctx = nullptr;
    uint64_t ctxSize = CalcAlgResourceCtxSize(request);
    const uint64_t expectedCtxSize = ctxSize;
    const CommEngine engine = COMM_ENGINE_AICPU;

    HcclResult ret = HcclEngineCtxGet(comm, param.tag, engine, &ctx, &ctxSize);
    if (ret == HCCL_SUCCESS) {
        if (ctxSize != expectedCtxSize) {
            HCCL_ERROR("cached ctx size mismatch, actual=%llu, expected=%llu",
                static_cast<unsigned long long>(ctxSize),
                static_cast<unsigned long long>(expectedCtxSize));
            return HCCL_E_INTERNAL;
        }
        *resCtx = static_cast<AlgResourceCtx *>(ctx);
        return HCCL_SUCCESS;
    }
    if (ret != HCCL_E_NOT_FOUND && ret != HCCL_E_UNAVAIL) {
        return ret;
    }

    HCCL_CHK_RET(HcclEngineCtxCreate(comm, param.tag, engine, expectedCtxSize, &ctx));

    std::unique_ptr<uint8_t[]> hostResBytes(new (std::nothrow) uint8_t[expectedCtxSize]());
    if (!hostResBytes) {
        HCCL_ERROR("failed to allocate host resource buffer, size=%llu",
            static_cast<unsigned long long>(expectedCtxSize));
        return HCCL_E_MEMORY;
    }

    AlgResourceCtx *hostResCtx = reinterpret_cast<AlgResourceCtx *>(hostResBytes.get());
    InitAlgResourceCtxHeader(request, *hostResCtx);
    HCCL_CHK_RET(AllocAlgResource(comm, param, request, *hostResCtx));
    HCCL_CHK_RET(HcclEngineCtxCopy(comm, engine, param.tag, hostResCtx, expectedCtxSize, 0));

    *resCtx = static_cast<AlgResourceCtx *>(ctx);
    return HCCL_SUCCESS;
}

HcclResult LoadAndLaunch(const OpParam &param, aclrtStream stream)
{
    HCCL_CHK_RET(LoadAICPUKernel());
    HCCL_CHK_RET(LaunchKernel(param, stream));
    return HCCL_SUCCESS;
}

}  // namespace

}  // namespace ops_hccl_allgatherbatch

HcclResult HcclAllGatherBatch(
    const HcclAllGatherItem *items, uint32_t itemCount, HcclComm comm, aclrtStream stream)
{
    using namespace ops_hccl_allgatherbatch;

    // 对外入口先只保留轻量日志，详细拓扑/模式信息交给 Host 主链路在拿到 topo 后再打印。
    HCCL_INFO("HcclAllGatherBatch invoked: itemCount=%u", itemCount);

    HCCL_CHK_RET(ValidateItems(items, itemCount, comm, stream));

    OpParam param;
    HCCL_CHK_RET(PrepareOpParam(items, itemCount, comm, param));
    HCCL_CHK_RET(ValidateTopo(param.topoInfo));
    HCCL_CHK_RET(ValidatePreparedParam(param));

    HCCL_INFO("Host op prepared: rank=%u, rankSize=%u, commMode=%s, serverIdx=%u, serverCount=%u, superPodIdx=%u, intraServerRankCount=%u, crossServerRankCount=%u",
        param.topoInfo.rank,
        param.topoInfo.rankSize,
        ToCommModeString(param.commMode),
        param.topoInfo.serverIdx,
        param.topoInfo.serverCount,
        param.topoInfo.superPodIdx,
        param.intraServerRankCount,
        param.crossServerRankCount);

    AlgResourceCtx *resCtx = nullptr;
    HCCL_CHK_RET(GetAlgRes(comm, param, &resCtx));
    param.resCtx = resCtx;
    HCCL_CHK_RET(ValidatePreparedResourceCtx(param));

    const ResourceStats stats = CollectResourceStats(param, *param.resCtx);
    HCCL_INFO("Host resources ready: rank=%u, commMode=%s, channelCount=%u, crossServerChannels=%u, perRankCapacity=%llu, maxWindowBytes=%llu, hccs=%u, roce=%u, pcie=%u, sio=%u",
        param.topoInfo.rank,
        ToCommModeString(param.commMode),
        param.resCtx->channelCount,
        stats.crossServerChannels,
        static_cast<unsigned long long>(stats.perRankCapacity),
        static_cast<unsigned long long>(stats.maxWindowBytes),
        stats.hccsChannels,
        stats.roceChannels,
        stats.pcieChannels,
        stats.sioChannels);

    HCCL_CHK_RET(LoadAndLaunch(param, stream));
    return HCCL_SUCCESS;
}


