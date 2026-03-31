#include "allgather_batch_op.h"

#include <cstdio>

#include "hccl/hccl_rank_graph.h"
#include "launch_kernel.h"
#include "load_kernel.h"
#include "log.h"

namespace ops_hccl_allgatherbatch {

namespace {

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

HcclResult QueryNetLayers(HcclComm comm, uint32_t **netLayers, uint32_t *netLayerNum)
{
    HCCL_CHK_PTR(netLayers);
    HCCL_CHK_PTR(netLayerNum);
    return HcclRankGraphGetLayers(comm, netLayers, netLayerNum);
}

HcclResult QueryChannelProtocol(HcclComm comm, uint32_t localRank, uint32_t remoteRank, CommProtocol &protocol)
{
    uint32_t *netLayers = nullptr;
    uint32_t netLayerNum = 0;
    HCCL_CHK_RET(QueryNetLayers(comm, &netLayers, &netLayerNum));

    for (uint32_t idx = 0; idx < netLayerNum; ++idx) {
        CommLink *links = nullptr;
        uint32_t linkNum = 0;
        HcclResult ret = HcclRankGraphGetLinks(comm, netLayers[idx], localRank, remoteRank, &links, &linkNum);
        if (ret == HCCL_SUCCESS && links != nullptr && linkNum > 0) {
            protocol = links[0].linkAttr.linkProtocol;
            return HCCL_SUCCESS;
        }
    }

    HCCL_ERROR("failed to query link protocol between rank=%u and remoteRank=%u", localRank, remoteRank);
    return HCCL_E_NOT_FOUND;
}

HcclResult FillServerGroupInfo(HcclComm comm, BatchTopoInfo &topoInfo)
{
    uint32_t *instSizeList = nullptr;
    uint32_t instListSize = 0;
    HcclResult ret = HcclRankGraphGetInstSizeListByLayer(comm, 0, &instSizeList, &instListSize);
    if (ret != HCCL_SUCCESS || instSizeList == nullptr || instListSize == 0) {
        // 单机或简单场景下 rank graph 可能拿不到更细粒度 server 信息，这里回退到默认值。
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

    CommLink *links = nullptr;
    uint32_t linkNum = 0;
    ret = HcclRankGraphGetLinks(comm, 0, topoInfo.rank, peerRank, &links, &linkNum);
    if (ret != HCCL_SUCCESS || links == nullptr || linkNum == 0) {
        return HCCL_SUCCESS;
    }

    topoInfo.serverIdx = links[0].srcEndpointDesc.loc.device.serverIdx;
    topoInfo.superPodIdx = links[0].srcEndpointDesc.loc.device.superPodIdx;
    return HCCL_SUCCESS;
}

HcclResult BuildFreshResourceCtx(HcclComm comm, const BatchTopoInfo &topoInfo, AlgResourceCtx &resCtx)
{
    // 先准备主线程和本地 HCCL buffer，它们是 Pack/Unpack 和 AICPU 控制流的最小前提。
    void *localBuffer = nullptr;
    HCCL_CHK_RET(EnsureControlNotifies(resCtx));
    HCCL_CHK_RET(HcclThreadAcquire(comm, COMM_ENGINE_AICPU, 1, 0, &resCtx.threadHandle));
    HCCL_CHK_RET(HcclGetHcclBuffer(comm, &localBuffer, &resCtx.localBuffer.size));
    resCtx.localBuffer.addr = localBuffer;
    resCtx.channelCount = 0;

    // 多 rank 时，按“本 rank 到其它 rank 各建 1 条 channel”的最小模型申请资源。
    // 这里不再硬编码 HCCS，而是按 rank graph 查询每个对端的真实链路协议，为跨 server 场景留出正确协议入口。
    if (topoInfo.rankSize <= 1) {
        return HCCL_SUCCESS;
    }
    if (topoInfo.rankSize - 1 > kAllGatherBatchMaxChannels) {
        HCCL_ERROR("rankSize=%u exceeds max channel capacity=%u", topoInfo.rankSize, kAllGatherBatchMaxChannels);
        return HCCL_E_NOT_SUPPORT;
    }

    HcclChannelDesc channelDescs[kAllGatherBatchMaxChannels];
    ChannelHandle channelHandles[kAllGatherBatchMaxChannels] = {0};
    HcclChannelDescInit(channelDescs, kAllGatherBatchMaxChannels);

    uint32_t channelCount = 0;
    for (uint32_t remoteRank = 0; remoteRank < topoInfo.rankSize; ++remoteRank) {
        if (remoteRank == topoInfo.rank) {
            continue;
        }

        CommProtocol protocol = COMM_PROTOCOL_RESERVED;
        HCCL_CHK_RET(QueryChannelProtocol(comm, topoInfo.rank, remoteRank, protocol));
        channelDescs[channelCount].remoteRank = remoteRank;
        channelDescs[channelCount].channelProtocol = protocol;
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
    // Host 侧主流程先固定成三段：校验/封装 -> 资源准备 -> load + launch。
    HCCL_CHK_RET(Validate(items, itemCount, comm, stream));

    OpParam param;
    HCCL_CHK_RET(PrepareOpParam(items, itemCount, comm, param));

    AlgResourceCtx *resCtx = nullptr;
    HCCL_CHK_RET(GetAlgRes(comm, param, &resCtx));
    param.resCtx = resCtx;

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

HcclResult AllGatherBatchOp::PrepareTopoInfo(HcclComm comm, BatchTopoInfo &topoInfo) const
{
    HCCL_CHK_RET(HcclGetRankId(comm, &topoInfo.rank));
    HCCL_CHK_RET(HcclGetRankSize(comm, &topoInfo.rankSize));

    topoInfo.serverCount = 1;
    topoInfo.serverIdx = 0;
    topoInfo.superPodIdx = 0;

    // 这里开始使用 hcomm 的公开 rank graph 接口，补齐跨 server 场景至少需要的 server 归属与分组规模。
    HCCL_CHK_RET(FillServerGroupInfo(comm, topoInfo));
    HCCL_CHK_RET(FillEndpointLocation(comm, topoInfo));
    return HCCL_SUCCESS;
}

HcclResult AllGatherBatchOp::PrepareOpParam(
    const HcclAllGatherItem *items, uint32_t itemCount, HcclComm comm, OpParam &param) const
{
    // OpParam 在这一阶段先承载稳定的 launch 入参：tag、comm 名称、topo 基本信息、总字节量。
    std::snprintf(param.tag, sizeof(param.tag), "%s", kAllGatherBatchCtxTag);
    HCCL_CHK_RET(HcclGetCommName(comm, param.commName));
    HCCL_CHK_RET(PrepareTopoInfo(comm, param.topoInfo));

    param.itemCount = itemCount;
    param.appendedItemBytes = static_cast<uint64_t>(itemCount) * sizeof(BatchItemParam);
    param.totalInputBytes = 0;
    param.totalOutputBytes = 0;

    // 当前 custom-op 方案把 item 描述符内联在 OpParam 里，先保证 Host/Device 协议简单稳定。
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

    // 后续执行器会根据 buffer 和算法约束重算窗口，这里先把总输入量作为初始窗口上界。
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

    // 优先复用 engine context 里的 device 侧资源，避免同一 comm 反复申请。
    HcclResult ret = HcclEngineCtxGet(comm, param.tag, engine, &ctx, &ctxSize);
    if (ret == HCCL_SUCCESS) {
        *resCtx = static_cast<AlgResourceCtx *>(ctx);
        return HCCL_SUCCESS;
    }
    if (ret != HCCL_E_NOT_FOUND && ret != HCCL_E_UNAVAIL) {
        return ret;
    }

    // 缓存未命中时，先创建 device context，再把 host 临时组织的资源信息拷过去。
    HCCL_CHK_RET(HcclEngineCtxCreate(comm, param.tag, engine, sizeof(AlgResourceCtx), &ctx));

    AlgResourceCtx hostResCtx;
    HCCL_CHK_RET(BuildFreshResourceCtx(comm, param.topoInfo, hostResCtx));
    HCCL_CHK_RET(HcclEngineCtxCopy(comm, engine, param.tag, &hostResCtx, sizeof(AlgResourceCtx), 0));

    *resCtx = static_cast<AlgResourceCtx *>(ctx);
    return HCCL_SUCCESS;
}

HcclResult AllGatherBatchOp::LoadAndLaunch(const OpParam &param, aclrtStream stream) const
{
    // load 和 launch 先保持独立，后面接 Device 真正入口时不用再拆 Host 主流程。
    HCCL_CHK_RET(LoadAICPUKernel());
    HCCL_CHK_RET(LaunchKernel(param, stream));
    return HCCL_SUCCESS;
}

}  // namespace ops_hccl_allgatherbatch
