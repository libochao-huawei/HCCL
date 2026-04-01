#include "allgather_batch_op.h"

#include <cstdio>

#include "hccl/hccl_rank_graph.h"
#include "launch_kernel.h"
#include "load_kernel.h"
#include "log.h"

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

HcclResult ValidatePreparedParam(const OpParam &param)
{
    if (!IsValidCommMode(param.commMode)) {
        HCCL_ERROR("prepared param commMode is invalid");
        return HCCL_E_INTERNAL;
    }
    if (param.itemCount == 0 || param.itemCount > kAllGatherBatchMaxItems) {
        HCCL_ERROR("prepared param itemCount=%u is invalid", param.itemCount);
        return HCCL_E_INTERNAL;
    }
    if (param.topoInfo.rankSize == 0 || param.topoInfo.rank >= param.topoInfo.rankSize) {
        HCCL_ERROR("prepared topo rank/rankSize is invalid, rank=%u, rankSize=%u",
            param.topoInfo.rank,
            param.topoInfo.rankSize);
        return HCCL_E_INTERNAL;
    }
    if (!HasConsistentRankDistribution(param)) {
        HCCL_ERROR("prepared rank distribution is inconsistent, commMode=%s, intra=%u, cross=%u, rankSize=%u",
            ToCommModeString(param.commMode),
            param.intraServerRankCount,
            param.crossServerRankCount,
            param.topoInfo.rankSize);
        return HCCL_E_INTERNAL;
    }
    if (param.totalInputBytes == 0 || param.totalOutputBytes == 0 || param.windowBytes == 0) {
        HCCL_ERROR("prepared byte sizes are invalid, input=%llu, output=%llu, window=%llu",
            static_cast<unsigned long long>(param.totalInputBytes),
            static_cast<unsigned long long>(param.totalOutputBytes),
            static_cast<unsigned long long>(param.windowBytes));
        return HCCL_E_INTERNAL;
    }
    return HCCL_SUCCESS;
}

HcclResult ValidatePreparedResourceCtx(const OpParam &param)
{
    HCCL_CHK_PTR(param.resCtx);
    const AlgResourceCtx &resCtx = *param.resCtx;
    const ResourceStats stats = CollectResourceStats(param, resCtx);

    if (resCtx.threadHandle == 0) {
        HCCL_ERROR("prepared resCtx threadHandle is invalid");
        return HCCL_E_INTERNAL;
    }
    if (resCtx.localBuffer.addr == nullptr || resCtx.localBuffer.size == 0) {
        HCCL_ERROR("prepared resCtx localBuffer is invalid");
        return HCCL_E_INTERNAL;
    }
    if (param.topoInfo.rankSize > 1 && resCtx.channelCount + 1 < param.topoInfo.rankSize) {
        HCCL_ERROR("prepared resCtx channelCount=%u is insufficient for rankSize=%u",
            resCtx.channelCount,
            param.topoInfo.rankSize);
        return HCCL_E_INTERNAL;
    }
    if (stats.intraServerChannels + stats.crossServerChannels != resCtx.channelCount) {
        HCCL_ERROR("prepared resCtx channel split is inconsistent, intra=%u, cross=%u, channelCount=%u",
            stats.intraServerChannels,
            stats.crossServerChannels,
            resCtx.channelCount);
        return HCCL_E_INTERNAL;
    }
    if (param.commMode == BatchCommMode::kSingleServer && stats.crossServerChannels != 0) {
        HCCL_ERROR("single-server resCtx unexpectedly has crossServerChannels=%u", stats.crossServerChannels);
        return HCCL_E_INTERNAL;
    }
    if (param.commMode == BatchCommMode::kCrossServer && stats.crossServerChannels != param.crossServerRankCount) {
        HCCL_ERROR("cross-server resCtx mismatch, channels=%u, expected=%u",
            stats.crossServerChannels,
            param.crossServerRankCount);
        return HCCL_E_INTERNAL;
    }
    if (stats.maxWindowBytes == 0) {
        HCCL_ERROR("prepared maxWindowBytes is zero");
        return HCCL_E_INTERNAL;
    }

    // Host 在 launch 前把资源协议也收一遍，尽量让错误停在资源准备阶段而不是设备执行阶段。
    for (uint32_t idx = 0; idx < resCtx.channelCount; ++idx) {
        const ChannelResource &channel = resCtx.channels[idx];
        if (channel.protocol == COMM_PROTOCOL_RESERVED) {
            HCCL_ERROR("prepared channel %u has reserved protocol", idx);
            return HCCL_E_INTERNAL;
        }
        if (channel.remoteRank == param.topoInfo.rank || channel.remoteRank >= param.topoInfo.rankSize) {
            HCCL_ERROR("prepared channel %u remoteRank=%u is invalid", idx, channel.remoteRank);
            return HCCL_E_INTERNAL;
        }
        if (channel.remoteSuperPodIdx != param.topoInfo.superPodIdx) {
            HCCL_ERROR("prepared channel %u crosses superPod unexpectedly, local=%u, remote=%u",
                idx,
                param.topoInfo.superPodIdx,
                channel.remoteSuperPodIdx);
            return HCCL_E_INTERNAL;
        }
        if (channel.remoteBuffer.addr == nullptr || channel.remoteBuffer.size == 0) {
            HCCL_ERROR("prepared channel %u remoteBuffer is invalid", idx);
            return HCCL_E_INTERNAL;
        }
        if (channel.remoteBuffer.size < (stats.maxWindowBytes * param.topoInfo.rankSize)) {
            HCCL_ERROR("prepared channel %u remoteBuffer too small, need=%llu, actual=%llu",
                idx,
                static_cast<unsigned long long>(stats.maxWindowBytes * param.topoInfo.rankSize),
                static_cast<unsigned long long>(channel.remoteBuffer.size));
            return HCCL_E_INTERNAL;
        }
    }
    return HCCL_SUCCESS;
}

HcclResult BuildFreshResourceCtx(HcclComm comm, const BatchTopoInfo &topoInfo, AlgResourceCtx &resCtx)
{
    void *localBuffer = nullptr;
    HCCL_CHK_RET(EnsureControlNotifies(resCtx));
    HCCL_CHK_RET(HcclThreadAcquire(comm, COMM_ENGINE_AICPU, 1, 0, &resCtx.threadHandle));
    HCCL_CHK_RET(HcclGetHcclBuffer(comm, &localBuffer, &resCtx.localBuffer.size));
    resCtx.localBuffer.addr = localBuffer;
    resCtx.channelCount = 0;

    if (topoInfo.rankSize <= 1) {
        return HCCL_SUCCESS;
    }
    if (topoInfo.rankSize - 1 > kAllGatherBatchMaxChannels) {
        HCCL_ERROR("rankSize=%u exceeds max channel capacity=%u", topoInfo.rankSize, kAllGatherBatchMaxChannels);
        return HCCL_E_NOT_SUPPORT;
    }

    HcclChannelDesc channelDescs[kAllGatherBatchMaxChannels];
    ChannelHandle channelHandles[kAllGatherBatchMaxChannels] = {0};
    ChannelLinkInfo linkInfos[kAllGatherBatchMaxChannels];
    HcclChannelDescInit(channelDescs, kAllGatherBatchMaxChannels);

    uint32_t channelCount = 0;
    for (uint32_t remoteRank = 0; remoteRank < topoInfo.rankSize; ++remoteRank) {
        if (remoteRank == topoInfo.rank) {
            continue;
        }

        HCCL_CHK_RET(QueryChannelLinkInfo(comm, topoInfo.rank, remoteRank, linkInfos[channelCount]));
        if (linkInfos[channelCount].localSuperPodIdx != linkInfos[channelCount].remoteSuperPodIdx) {
            HCCL_ERROR("cross-superpod link is not supported, rank=%u(superPod=%u) remoteRank=%u(superPod=%u)",
                topoInfo.rank,
                linkInfos[channelCount].localSuperPodIdx,
                remoteRank,
                linkInfos[channelCount].remoteSuperPodIdx);
            return HCCL_E_NOT_SUPPORT;
        }

        channelDescs[channelCount].remoteRank = remoteRank;
        channelDescs[channelCount].channelProtocol = linkInfos[channelCount].protocol;
        channelDescs[channelCount].notifyNum = 2;
        ++channelCount;
    }

    HCCL_CHK_RET(HcclChannelAcquire(
        comm,
        COMM_ENGINE_AICPU,
        channelDescs,
        channelCount,
        channelHandles));

    for (uint32_t idx = 0; idx < channelCount; ++idx) {
        resCtx.channels[idx].handle = channelHandles[idx];
        resCtx.channels[idx].remoteRank = channelDescs[idx].remoteRank;
        resCtx.channels[idx].remoteServerIdx = linkInfos[idx].remoteServerIdx;
        resCtx.channels[idx].remoteSuperPodIdx = linkInfos[idx].remoteSuperPodIdx;
        resCtx.channels[idx].protocol = linkInfos[idx].protocol;
        resCtx.channels[idx].localNotifyIdx = 0;
        resCtx.channels[idx].remoteNotifyIdx = 0;
        HCCL_CHK_RET(HcclChannelGetHcclBuffer(
            comm,
            channelHandles[idx],
            &resCtx.channels[idx].remoteBuffer.addr,
            &resCtx.channels[idx].remoteBuffer.size));
    }
    resCtx.channelCount = channelCount;
    return HCCL_SUCCESS;
}

}  // namespace

HcclResult AllGatherBatchOp::Exec(
    const HcclAllGatherItem *items, uint32_t itemCount, HcclComm comm, aclrtStream stream)
{
    HCCL_CHK_RET(Validate(items, itemCount, comm, stream));

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

HcclResult AllGatherBatchOp::Validate(
    const HcclAllGatherItem *items, uint32_t itemCount, HcclComm comm, aclrtStream stream) const
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

HcclResult AllGatherBatchOp::ValidateTopo(const BatchTopoInfo &topoInfo) const
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

HcclResult AllGatherBatchOp::PrepareTopoInfo(HcclComm comm, BatchTopoInfo &topoInfo) const
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

HcclResult AllGatherBatchOp::PrepareOpParam(
    const HcclAllGatherItem *items, uint32_t itemCount, HcclComm comm, OpParam &param) const
{
    std::snprintf(param.tag, sizeof(param.tag), "%s", kAllGatherBatchCtxTag);
    HCCL_CHK_RET(HcclGetCommName(comm, param.commName));
    HCCL_CHK_RET(PrepareTopoInfo(comm, param.topoInfo));
    HCCL_CHK_RET(FillCommModeInfo(comm, param));

    param.itemCount = itemCount;
    param.appendedItemBytes = static_cast<uint64_t>(itemCount) * sizeof(BatchItemParam);
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

HcclResult AllGatherBatchOp::GetAlgRes(
    HcclComm comm, const OpParam &param, AlgResourceCtx **resCtx) const
{
    HCCL_CHK_PTR(resCtx);

    void *ctx = nullptr;
    uint64_t ctxSize = sizeof(AlgResourceCtx);
    const CommEngine engine = COMM_ENGINE_AICPU;

    HcclResult ret = HcclEngineCtxGet(comm, param.tag, engine, &ctx, &ctxSize);
    if (ret == HCCL_SUCCESS) {
        *resCtx = static_cast<AlgResourceCtx *>(ctx);
        return HCCL_SUCCESS;
    }
    if (ret != HCCL_E_NOT_FOUND && ret != HCCL_E_UNAVAIL) {
        return ret;
    }

    HCCL_CHK_RET(HcclEngineCtxCreate(comm, param.tag, engine, sizeof(AlgResourceCtx), &ctx));

    AlgResourceCtx hostResCtx;
    HCCL_CHK_RET(BuildFreshResourceCtx(comm, param.topoInfo, hostResCtx));
    HCCL_CHK_RET(HcclEngineCtxCopy(comm, engine, param.tag, &hostResCtx, sizeof(AlgResourceCtx), 0));

    *resCtx = static_cast<AlgResourceCtx *>(ctx);
    return HCCL_SUCCESS;
}

HcclResult AllGatherBatchOp::LoadAndLaunch(const OpParam &param, aclrtStream stream) const
{
    HCCL_CHK_RET(LoadAICPUKernel());
    HCCL_CHK_RET(LaunchKernel(param, stream));
    return HCCL_SUCCESS;
}

}  // namespace ops_hccl_allgatherbatch

