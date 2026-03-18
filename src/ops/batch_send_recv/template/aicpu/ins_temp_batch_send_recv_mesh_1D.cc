/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "aicpu/ins_temp_batch_send_recv_mesh_1D.h"

namespace ops_hccl {

InsTempBatchSendRecvMesh1D::InsTempBatchSendRecvMesh1D(
    const OpParam &param, const u32 rankId,
    const std::vector<std::vector<u32>> &subCommRanks)
    : InsAlgTemplateBase(param, rankId, subCommRanks)
{
}

HcclResult InsTempBatchSendRecvMesh1D::CalcRes(HcclComm comm, const OpParam &param,
    const TopoInfoWithNetLayerDetails *topoInfo, AlgResourceRequest &resourceRequest)
{
    resourceRequest.slaveThreadNum = 1;
    resourceRequest.notifyNumPerThread = {1};
    resourceRequest.notifyNumOnMainThread = 1;

    CHK_PTR_NULL(param.batchSendRecvDataDes.sendRecvItemsPtr);
    const HcclSendRecvItem *itemPtr = param.batchSendRecvDataDes.sendRecvItemsPtr;
    u32 itemNum = param.batchSendRecvDataDes.itemNum;

    std::set<u32> commTargetUserRankSet;
    for (u32 i = 0; i < itemNum; i++) {
        commTargetUserRankSet.insert((itemPtr + i)->remoteRank);
    }

    std::vector<HcclChannelDesc> channelLevel0;
    for (const u32 &remoteRank : commTargetUserRankSet) {
        if (remoteRank == static_cast<uint32_t>(topoInfo->userRank)) {
            continue;
        }
        std::vector<HcclChannelDesc> channelByRank;
        CHK_RET(CreateChannelRequestByRankId(comm, topoInfo->userRank,
            remoteRank, channelByRank, CHANNEL_NUM_PER_RANK_PAIR));
        channelLevel0.insert(channelLevel0.end(), channelByRank.begin(), channelByRank.end());
    }
    resourceRequest.channels.push_back(channelLevel0);
    return HCCL_SUCCESS;
}

u64 InsTempBatchSendRecvMesh1D::CalcScratchMultiple(BufferType inBuffType, BufferType outBuffType)
{
    return 0;
}

void InsTempBatchSendRecvMesh1D::SetBatchSendRecvInfo(const BatchSendRecvInfo &info)
{
    batchSendRecvInfo_ = info;
}

void InsTempBatchSendRecvMesh1D::GetNotifyIdxMainToSub(std::vector<u32> &notifyIdxMainToSub)
{
    notifyIdxMainToSub.clear();
    notifyIdxMainToSub.push_back(0);
}

void InsTempBatchSendRecvMesh1D::GetNotifyIdxSubToMain(std::vector<u32> &notifyIdxSubToMain)
{
    notifyIdxSubToMain.clear();
    notifyIdxSubToMain.push_back(0);
}

HcclResult InsTempBatchSendRecvMesh1D::ProcessSelfSendRecvTasks(const ThreadHandle &thread)
{
    auto &sendToSelfSlices = batchSendRecvInfo_.sendToSelfSlices;
    auto &recvFromSelfSlices = batchSendRecvInfo_.recvFromSelfSlices;
    while (!sendToSelfSlices.empty() && !recvFromSelfSlices.empty()) {
        DataSlice srcSlice(sendToSelfSlices.front().addr, 0, sendToSelfSlices.front().size);
        DataSlice dstSlice(recvFromSelfSlices.front().addr, 0, recvFromSelfSlices.front().size);
        CHK_RET(LocalCopy(thread, srcSlice, dstSlice));
        HCCL_DEBUG("[InsTempBatchSendRecvMesh1D][ProcessSelfSendRecvTasks] inputData[%p], "\
            "outputData[%p], dataSize[%llu]", sendToSelfSlices.front().addr,
            recvFromSelfSlices.front().addr, sendToSelfSlices.front().size);
        sendToSelfSlices.pop_front();
        recvFromSelfSlices.pop_front();
    }
    return HCCL_SUCCESS;
}

HcclResult InsTempBatchSendRecvMesh1D::GetSendChannel(u32 remoteRank,
    const std::map<u32, std::vector<ChannelInfo>> &channels, ChannelInfo &sendChannel) const
{
    auto it = channels.find(remoteRank);
    if (it == channels.end()) {
        HCCL_ERROR("[InsTempBatchSendRecvMesh1D][GetSendChannel] Cannot find channel for remoteRank[%u]",
            remoteRank);
        return HCCL_E_INTERNAL;
    }
    if (it->second.size() < CHANNEL_NUM_PER_RANK_PAIR) {
        HCCL_ERROR("[InsTempBatchSendRecvMesh1D][GetSendChannel] Channel number[%zu] is less than expected[%u]",
            it->second.size(), CHANNEL_NUM_PER_RANK_PAIR);
        return HCCL_E_INTERNAL;
    }
    if (remoteRank < static_cast<u32>(myRank_)) {
        sendChannel = it->second[0];
    } else {
        sendChannel = it->second[1];
    }
    return HCCL_SUCCESS;
}

HcclResult InsTempBatchSendRecvMesh1D::GetRecvChannel(u32 remoteRank,
    const std::map<u32, std::vector<ChannelInfo>> &channels, ChannelInfo &recvChannel) const
{
    auto it = channels.find(remoteRank);
    if (it == channels.end()) {
        HCCL_ERROR("[InsTempBatchSendRecvMesh1D][GetRecvChannel] Cannot find channel for remoteRank[%u]",
            remoteRank);
        return HCCL_E_INTERNAL;
    }
    if (it->second.size() < CHANNEL_NUM_PER_RANK_PAIR) {
        HCCL_ERROR("[InsTempBatchSendRecvMesh1D][GetRecvChannel] Channel number[%zu] is less than expected[%u]",
            it->second.size(), CHANNEL_NUM_PER_RANK_PAIR);
        return HCCL_E_INTERNAL;
    }
    if (remoteRank > static_cast<u32>(myRank_)) {
        recvChannel = it->second[0];
    } else {
        recvChannel = it->second[1];
    }
    return HCCL_SUCCESS;
}

HcclResult InsTempBatchSendRecvMesh1D::ProcessSendDataSlice(SendRecvSlice &sendSlice,
    const ThreadHandle &thread, const HcclMem &cclMem,
    const std::map<u32, std::vector<ChannelInfo>> &channels) const
{
    // 数据从 input mem copy 到 CCL buffer 上
    DataSlice srcSlice(sendSlice.addr, 0, sendSlice.size);
    DataSlice dstSlice(cclMem.addr, 0, sendSlice.size);
    CHK_RET(LocalCopy(thread, srcSlice, dstSlice));

    // 发送数据
    ChannelInfo sendChannel;
    CHK_RET(GetSendChannel(sendSlice.remoteRank, channels, sendChannel));
    HCCL_DEBUG("[InsTempBatchSendRecvMesh1D][ProcessSendDataSlice] myRank[%u], remoteRank[%u], "\
        "send channel handle[0x%llx]", myRank_, sendSlice.remoteRank, sendChannel.handle);
    SlicesList slices({}, {});
    DataInfo sendDataInfo(sendChannel, slices);
    SendRead(sendDataInfo, thread);
    return HCCL_SUCCESS;
}

HcclResult InsTempBatchSendRecvMesh1D::ProcessRecvDataSlice(SendRecvSlice &recvSlice,
    const ThreadHandle &thread,
    const std::map<u32, std::vector<ChannelInfo>> &channels) const
{
    ChannelInfo recvChannel;
    CHK_RET(GetRecvChannel(recvSlice.remoteRank, channels, recvChannel));
    HCCL_DEBUG("[InsTempBatchSendRecvMesh1D][ProcessRecvDataSlice] myRank[%u], remoteRank[%u], "\
        "recv channel handle[0x%llx]", myRank_, recvSlice.remoteRank, recvChannel.handle);

    void *remoteCclBuffAddr = recvChannel.remoteCclMem.addr;
    DataSlice recvSrcDataSlice(remoteCclBuffAddr, 0, recvSlice.size);
    DataSlice recvDstDataSlice(recvSlice.addr, 0, recvSlice.size);
    SlicesList slices({recvSrcDataSlice}, {recvDstDataSlice});
    DataInfo recvDataInfo(recvChannel, slices);
    RecvRead(recvDataInfo, thread);
    return HCCL_SUCCESS;
}

HcclResult InsTempBatchSendRecvMesh1D::RunLoopSendRecv(const std::vector<ThreadHandle> &threads,
    const HcclMem &cclMem, const std::map<u32, std::vector<ChannelInfo>> &channels)
{
    // 前同步
    std::vector<ThreadHandle> subThreads = {threads[1]};
    std::vector<u32> notifyIdxMainToSub = {0};
    CHK_RET(PreSyncInterThreads(threads[0], subThreads, notifyIdxMainToSub));

    auto &sendSlices = batchSendRecvInfo_.sendSlices;
    auto &recvSlices = batchSendRecvInfo_.recvSlices;
    while (!sendSlices.empty() || !recvSlices.empty()) {
        if (!sendSlices.empty()) {
            CHK_RET(ProcessSendDataSlice(sendSlices.front(), threads[0], cclMem, channels));
            sendSlices.pop_front();
        }
        if (!recvSlices.empty()) {
            CHK_RET(ProcessRecvDataSlice(recvSlices.front(), threads[1], channels));
            recvSlices.pop_front();
        }
    }
    HCCL_INFO("[InsTempBatchSendRecvMesh1D][RunLoopSendRecv] Process all tasks finish.");

    // 后同步
    std::vector<u32> notifyIdxSubToMain = {0};
    CHK_RET(PostSyncInterThreads(threads[0], subThreads, notifyIdxSubToMain));
    HCCL_INFO("[InsTempBatchSendRecvMesh1D][RunLoopSendRecv] post sync success.");
    return HCCL_SUCCESS;
}

HcclResult InsTempBatchSendRecvMesh1D::KernelRun(const OpParam &param,
    const TemplateDataParams &tempAlgParams, const TemplateResource &templateResource)
{
    HCCL_INFO("[InsTempBatchSendRecvMesh1D][KernelRun] Start.");

    // 执行自收发任务
    CHK_RET(ProcessSelfSendRecvTasks(templateResource.threads[0]));

    // 如果有跨 rank 通信任务，执行循环收发
    if (!batchSendRecvInfo_.sendSlices.empty() || !batchSendRecvInfo_.recvSlices.empty()) {
        CHK_RET(RunLoopSendRecv(templateResource.threads, tempAlgParams.buffInfo.hcclBuff,
            templateResource.channels));
    }

    HCCL_INFO("[InsTempBatchSendRecvMesh1D][KernelRun] End.");
    return HCCL_SUCCESS;
}

} // namespace ops_hccl
