/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "aicpu/ins_temp_all_to_all_v_mesh_1D.h"

#include <string>

#define NET_NUM 2

namespace ops_hccl {
namespace {
constexpr u32 VMESH_2X2_RANK_NUM = 4;
constexpr u32 VMESH_2X2_ROW_NUM = 2;
constexpr u32 VMESH_2X2_COL_NUM = 2;
constexpr u32 VMESH_4X4_RANK_NUM = 16;
constexpr u32 VMESH_4X4_ROW_NUM = 4;
constexpr u32 VMESH_4X4_COL_NUM = 4;
constexpr u32 VMESH_8X8_RANK_NUM = 64;
constexpr u32 VMESH_8X8_ROW_NUM = 8;
constexpr u32 VMESH_8X8_COL_NUM = 8;

std::string FormatRankList(const std::vector<u32> &ranks)
{
    std::string rankList = "[";
    for (u32 index = 0; index < ranks.size(); index++) {
        if (index != 0) {
            rankList += ",";
        }
        rankList += std::to_string(ranks[index]);
    }
    rankList += "]";
    return rankList;
}
} // namespace

InsTempAlltoAllVMesh1D::InsTempAlltoAllVMesh1D(
    const OpParam& param, const u32 rankId, // 传通信域的rankId，userRank
    const std::vector<std::vector<u32>> &subCommRanks)
    : InsAlgTemplateBase(param, rankId, subCommRanks)
{
}

InsTempAlltoAllVMesh1D::~InsTempAlltoAllVMesh1D()
{
}

HcclResult InsTempAlltoAllVMesh1D::CalcRes(HcclComm comm, const OpParam& param, const TopoInfoWithNetLayerDetails* topoInfo,
    AlgResourceRequest& resourceRequest)
{
    if (topoInfo->level0Topo == Level0Shape::MESH_1D_CLOS && topoInfo->topoLevelNums > 1 && !topoInfo->level0PcieMix) {
        CHK_PRT_RET(subCommRanks_.size() != NET_NUM,
                    HCCL_ERROR("[InsTempAlltoAllVMesh1D][CalcRes] subCommRankNum[%zu] is not [%u]",
                               subCommRanks_.size(), NET_NUM),
                    HCCL_E_PARA);
        subCommRanks_ = {subCommRanks_[1]};
        templateRankSize_ = subCommRanks_[1].size();
    }
    HCCL_INFO("[InsTempAlltoAllVMesh1D][CalcRes] topoLevelNums[%u], level0Topo[%u], level0PcieMix[%u], "
        "templateRankSize[%u], subCommRanksSize[%zu], subCommRanks0%s.",
        topoInfo->topoLevelNums, static_cast<u32>(topoInfo->level0Topo), topoInfo->level0PcieMix,
        templateRankSize_, subCommRanks_.size(), subCommRanks_.empty() ? "[]" : FormatRankList(subCommRanks_[0]).c_str());
    u32 rowNum = 0;
    u32 colNum = 0;
    CHK_RET(GetVmeshShape(rowNum, colNum));
    HCCL_INFO("[InsTempAlltoAllVMesh1D][CalcRes] vmesh shape rowNum[%u], colNum[%u].", rowNum, colNum);

    std::vector<HcclChannelDesc> level0Channels;
    if(topoInfo->level0Topo == Level0Shape::MESH_1D_CLOS && !topoInfo->level0PcieMix) {
        std::vector<HcclChannelDesc> myChannelDescs;
        CHK_RET(CalcChannelRequestMesh1D(comm, param, topoInfo, subCommRanks_, myChannelDescs));
        HCCL_INFO("[InsTempAlltoAllVMesh1D][CalcRes] raw channel count[%zu].", myChannelDescs.size());
        for(auto channel : myChannelDescs) {
            HCCL_INFO("[InsTempAlltoAllVMesh1D][CalcRes] raw channel remoteRank[%u], protocol[%u], "
                "localPhyId[%u], remotePhyId[%u].",
                channel.remoteRank, channel.channelProtocol, channel.localEndpoint.loc.device.devPhyId,
                channel.remoteEndpoint.loc.device.devPhyId);
            if(channel.channelProtocol == COMM_PROTOCOL_UBC_CTP) {
                level0Channels.push_back(channel);
            }
        }
        HCCL_INFO("[InsTempAlltoAllVMesh1D][CalcRes] channel count after UBC_CTP filter[%zu].", level0Channels.size());
        HCCL_DEBUG("[InsTempAlltoAllVMesh1D::CalcRes] Get Channel Success!");
    } else {
        CHK_RET(CalcChannelRequestMesh1D(comm, param, topoInfo, subCommRanks_, level0Channels));
    }
    resourceRequest.channels.push_back(level0Channels);
    for (const auto &channel : level0Channels) {
        HCCL_INFO("[InsTempAlltoAllVMesh1D][CalcRes] valid channel remoteRank[%u], protocol[%u], notifyNum[%u].",
            channel.remoteRank, channel.channelProtocol, channel.notifyNum);
    }
    channelsPerRank_ = 1;
    u32 slotNum = rowNum + 1;
    HCCL_INFO("[InsTempAlltoAllVMesh1D][CalcRes] channelsPerRank[%u], slotNum[%u].",
        channelsPerRank_, slotNum);
    resourceRequest.slaveThreadNum = slotNum * channelsPerRank_;
    for (u32 index = 0; index < resourceRequest.slaveThreadNum; index++) {
        resourceRequest.notifyNumPerThread.push_back(channelsPerRank_);
    }
    resourceRequest.notifyNumOnMainThread = resourceRequest.slaveThreadNum;
    return HCCL_SUCCESS;
}

u64 InsTempAlltoAllVMesh1D::CalcScratchMultiple(BufferType inBuffType, BufferType outBuffType)
{
    (void) inBuffType;
    (void) outBuffType;
    u32 rowNum = 0;
    u32 colNum = 0;
    HcclResult ret = GetVmeshShape(rowNum, colNum);
    if (ret != HCCL_SUCCESS) {
        HCCL_ERROR("[InsTempAlltoAllVMesh1D][CalcScratchMultiple] failed to get vmesh shape, ret[%u].", ret);
        return 0;
    }
    concurrentSendRecvNum_ = rowNum + 1;
    HCCL_INFO("[InsTempAlltoAllVMesh1D][CalcScratchMultiple] templateRankSize[%u], rowNum[%u], "
        "colNum[%u], concurrentSendRecvNum[%u].", templateRankSize_, rowNum, colNum, concurrentSendRecvNum_);
    return concurrentSendRecvNum_;
}

HcclResult InsTempAlltoAllVMesh1D::CalcCommRankBySlot(const u32 roundIdx, const u32 slotIdx,
    u32 &remoteRank) const
{
    remoteRank = 0;
    u32 rowNum = 0;
    u32 colNum = 0;
    CHK_RET(GetVmeshShape(rowNum, colNum));
    if (roundIdx >= colNum - 1) {
        HCCL_ERROR("[InsTempAlltoAllVMesh1D][CalcCommRankBySlot] roundIdx[%u] is invalid, colNum[%u].",
            roundIdx, colNum);
        return HCCL_E_PARA;
    }
    if (slotIdx > rowNum) {
        HCCL_ERROR("[InsTempAlltoAllVMesh1D][CalcCommRankBySlot] slotIdx[%u] is invalid, rowNum[%u].",
            slotIdx, rowNum);
        return HCCL_E_PARA;
    }

    u32 myMeshRow = 0;
    u32 myMeshCol = 0;
    CHK_RET(GetMesh1DClosCoord(myRank_, myMeshRow, myMeshCol));
    u32 mask = roundIdx + 1;
    u32 remoteMeshRow = slotIdx;
    u32 remoteMeshCol = myMeshCol ^ mask;
    if (slotIdx == rowNum) {
        remoteMeshRow = myMeshRow ^ mask;
        remoteMeshCol = myMeshCol;
    }
    CHK_RET(GetRankByMesh1DClosCoord(remoteMeshRow, remoteMeshCol, remoteRank));
    HCCL_INFO("[InsTempAlltoAllVMesh1D][CalcCommRankBySlot] roundIdx[%u], slotIdx[%u], myRank[%u], "
        "rowNum[%u], colNum[%u], myMeshRow[%u], myMeshCol[%u], remoteMeshRow[%u], remoteMeshCol[%u], "
        "remoteRank[%u].",
        roundIdx, slotIdx, myRank_, rowNum, colNum, myMeshRow, myMeshCol, remoteMeshRow, remoteMeshCol,
        remoteRank);
    return HCCL_SUCCESS;
}

HcclResult InsTempAlltoAllVMesh1D::CalcCommLoops(u32 &commLoops) const
{
    commLoops = 0;
    u32 rowNum = 0;
    u32 colNum = 0;
    CHK_RET(GetVmeshShape(rowNum, colNum));
    commLoops = colNum - 1;
    HCCL_INFO("[InsTempAlltoAllVMesh1D][CalcCommLoops] rowNum[%u], colNum[%u], commLoops[%u].",
        rowNum, colNum, commLoops);
    return HCCL_SUCCESS;
}

HcclResult InsTempAlltoAllVMesh1D::CalcCclBuffIdx(u32 remoteRank, u32 &myRankCclBuffIdx, u32 &remoteCclBuffIdx) const
{
    myRankCclBuffIdx = 0;
    remoteCclBuffIdx = 0;
    u32 rowNum = 0;
    u32 colNum = 0;
    CHK_RET(GetVmeshShape(rowNum, colNum));

    u32 myRankIndex = 0;
    u32 remoteRankIndex = 0;
    CHK_RET(GetRankIndexInSubComm(myRank_, myRankIndex));
    CHK_RET(GetRankIndexInSubComm(remoteRank, remoteRankIndex));
    if (myRankIndex == remoteRankIndex) {
        HCCL_ERROR("[InsTempAlltoAllVMesh1D][CalcCclBuffIdx] myRank[%u] and remoteRank[%u] must be different.",
            myRank_, remoteRank);
        return HCCL_E_PARA;
    }

    u32 myMeshRow = 0;
    u32 myMeshCol = 0;
    u32 remoteMeshRow = 0;
    u32 remoteMeshCol = 0;
    CHK_RET(GetMesh1DClosCoord(myRank_, myMeshRow, myMeshCol));
    CHK_RET(GetMesh1DClosCoord(remoteRank, remoteMeshRow, remoteMeshCol));
    if (myMeshCol == remoteMeshCol) {
        myRankCclBuffIdx = rowNum;
        remoteCclBuffIdx = rowNum;
    } else {
        myRankCclBuffIdx = remoteMeshRow;
        remoteCclBuffIdx = myMeshRow;
    }
    HCCL_DEBUG("[InsTempAlltoAllVMesh1D][CalcCclBuffIdx] myRank[%u], remoteRank[%u], "
        "myRankIndex[%u], remoteRankIndex[%u], rowNum[%u], colNum[%u], myMeshRow[%u], myMeshCol[%u], "
        "remoteMeshRow[%u], remoteMeshCol[%u], myRankCclBuffIdx[%u], remoteCclBuffIdx[%u].",
        myRank_, remoteRank, myRankIndex, remoteRankIndex, rowNum, colNum, myMeshRow, myMeshCol, remoteMeshRow,
        remoteMeshCol, myRankCclBuffIdx, remoteCclBuffIdx);
    return HCCL_SUCCESS;
}

HcclResult InsTempAlltoAllVMesh1D::KernelRun(const OpParam& param,
    const TemplateDataParams& tempAlgParams,
    TemplateResource& templateResource)
{
    HCCL_INFO("[InsTempAlltoAllVMesh1D][KernelRun] Run Start");
    threadNum_ = templateResource.threads.size();
    dataType_ = param.all2AllVDataDes.sendType;
    dataTypeSize_ = SIZE_TABLE[dataType_];
    HCCL_INFO("[InsTempAlltoAllVMesh1D][KernelRun] threadNum[%u], channelRemoteRankNum[%zu], "
        "templateRankSize[%u], subCommRanks0%s.", threadNum_, templateResource.channels.size(), templateRankSize_,
        subCommRanks_.empty() ? "[]" : FormatRankList(subCommRanks_[0]).c_str());
    u32 rowNum = 0;
    u32 colNum = 0;
    CHK_RET(GetVmeshShape(rowNum, colNum));
    HCCL_INFO("[InsTempAlltoAllVMesh1D][KernelRun] vmesh shape rowNum[%u], colNum[%u].", rowNum, colNum);
    for (const auto &channelsByRank : templateResource.channels) {
        HCCL_INFO("[InsTempAlltoAllVMesh1D][KernelRun] remoteRank[%u], channelSize[%zu].",
            channelsByRank.first, channelsByRank.second.size());
        for (u32 channelIdx = 0; channelIdx < channelsByRank.second.size(); channelIdx++) {
            const auto &channel = channelsByRank.second[channelIdx];
            HCCL_INFO("[InsTempAlltoAllVMesh1D][KernelRun] remoteRank[%u], channelIdx[%u], protocol[%u], "
                "portGroupSize[%u], remoteCclAddr[%p], remoteCclSize[%llu].",
                channelsByRank.first, channelIdx, channel.protocol, channel.portGroupSize,
                channel.remoteCclMem.addr, channel.remoteCclMem.size);
        }
    }

    bool isPcieProtocal = IsPcieProtocol(templateResource.channels);  // 判断是否存在pcie链路
    isDmaRead_ = isPcieProtocal;  // 是否使用Read模式
    HCCL_DEBUG("[InsTempAlltoAllVMesh1D][KernelRun] Use Dma Read[%d]", isDmaRead_);

    u32 myAlgRank = 0;
    auto iter = std::find(subCommRanks_[0].begin(), subCommRanks_[0].end(), myRank_);
    if (iter != subCommRanks_[0].end()) {
        myAlgRank = std::distance(subCommRanks_[0].begin(), iter);
    } else {
        HCCL_ERROR("[InsTempAlltoAllVMesh1D][KernelRun] subCommRanks_ or myRank_ is error.");
        return HCCL_E_INTERNAL;
    }
    CHK_RET(RunALLtoALL(templateResource.channels, templateResource.threads, tempAlgParams, myAlgRank));

    HCCL_INFO("[InsTempAlltoAllVMesh1D][KernelRun] Run End");
    return HcclResult::HCCL_SUCCESS;
}

HcclResult InsTempAlltoAllVMesh1D::LocalCopyForMyRank(const TemplateDataParams &tempAlgParams,
    const ThreadHandle &thread, const u32 myAlgRank, const u32 queIdx) const
{
    DataSlice srcSlice = DataSlice(tempAlgParams.buffInfo.inputPtr,
        tempAlgParams.sdispls[myAlgRank] * dataTypeSize_,
        tempAlgParams.sendCounts[myAlgRank] * dataTypeSize_, tempAlgParams.sendCounts[myAlgRank]);
    DataSlice dstSlice = DataSlice(tempAlgParams.buffInfo.outputPtr,
        tempAlgParams.rdispls[myAlgRank] * dataTypeSize_,
        tempAlgParams.recvCounts[myAlgRank] * dataTypeSize_, tempAlgParams.recvCounts[myAlgRank]);

    if (tempAlgParams.sendCounts[myAlgRank] > 0) {
        CHK_RET(static_cast<HcclResult>(LocalCopy(thread, srcSlice, dstSlice)));
        HCCL_DEBUG("[InsTempAlltoAllVMesh1D][RunALLtoALL] do local copy on thread[%u], data size[%llu].",
            queIdx, tempAlgParams.sendCounts[myAlgRank] * dataTypeSize_);
    }
    return HCCL_SUCCESS;
}

HcclResult InsTempAlltoAllVMesh1D::RunALLtoALL(
    const std::map<u32, std::vector<ChannelInfo>> &channels, const std::vector<ThreadHandle> &threads,
    const TemplateDataParams &tempAlgParams, const u32 myAlgRank)
{
    // 计算通信轮数
    u32 commLoops = 0;
    CHK_RET(CalcCommLoops(commLoops));
    u32 remainRankSize = templateRankSize_ - 1;
    channelsPerRank_ = 1;
    HCCL_INFO("[InsTempAlltoAllVMesh1D][RunALLtoALL] commLoops[%u], remainRankSize[%u], "
        "channelsPerRank[%u], threadNum[%u], isDmaRead[%u].",
        commLoops, remainRankSize, channelsPerRank_, threadNum_, isDmaRead_);
    u32 rowNum = 0;
    u32 colNum = 0;
    CHK_RET(GetVmeshShape(rowNum, colNum));
    u32 slotNum = rowNum + 1;
    if (threadNum_ <= slotNum || threads.size() <= slotNum) {
        HCCL_ERROR("[InsTempAlltoAllVMesh1D][RunALLtoALL] threadNum[%u], threadsSize[%zu] is not enough for "
            "slotNum[%u].", threadNum_, threads.size(), slotNum);
        return HCCL_E_PARA;
    }

    std::vector<ThreadHandle> subThreads;
    if (threadNum_ > 1) {
        subThreads.assign(threads.begin() + 1, threads.end());
        GetNotifyIdxMainToSub(notifyIdxMainToSub_);
        CHK_RET(PreSyncInterThreads(threads[0], subThreads, notifyIdxMainToSub_));
    }
    CHK_RET(LocalCopyForMyRank(tempAlgParams, threads[0], myAlgRank, 0));
    for (u32 slotIdx = 0; slotIdx < slotNum; slotIdx++) {
        CHK_RET(RunSendRecvBySlot(slotIdx, tempAlgParams, channels, threads, commLoops));
    }
    if (threadNum_ > 1) {
        GetNotifyIdxSubToMain(notifyIdxSubToMain_);
        CHK_RET(PostSyncInterThreads(threads[0], subThreads, notifyIdxSubToMain_));
    }
    return HCCL_SUCCESS;
}

HcclResult InsTempAlltoAllVMesh1D::RunSendRecvBySlot(const u32 slotIdx,
    const TemplateDataParams &tempAlgParams,
    const std::map<u32, std::vector<ChannelInfo>> &channels,
    const std::vector<ThreadHandle> &threads, const u32 commLoops)
{
    for (u32 roundIdx = 0; roundIdx < commLoops; roundIdx++) {
        u32 remoteRank = 0;
        CHK_RET(CalcCommRankBySlot(roundIdx, slotIdx, remoteRank));
        HCCL_INFO("[InsTempAlltoAllVMesh1D][RunSendRecvBySlot] slotIdx[%u], roundIdx[%u], "
            "remoteRank[%u], commLoops[%u].", slotIdx, roundIdx, remoteRank, commLoops);
        CHK_RET(RunSendRecvByRemoteRank(tempAlgParams, roundIdx, channels, threads, remoteRank, slotIdx));
    }
    return HcclResult::HCCL_SUCCESS;
}

HcclResult InsTempAlltoAllVMesh1D::RunSendRecvByRemoteRank(const TemplateDataParams &tempAlgParams,
    const u32 roundIdx,
    const std::map<u32, std::vector<ChannelInfo>> &channels,
    const std::vector<ThreadHandle> &threads, const u32 remoteRank, const u32 slotIdx)
{
    if (channels.find(remoteRank) == channels.end()) {
        HCCL_ERROR("[InsTempAlltoAllVMesh1D][RunSendRecvByRemoteRank] remoteRank[%u] "
            "does not exist in channels map!", remoteRank);
        return HCCL_E_PARA;
    }
    const std::vector<ChannelInfo> &allChannels = channels.at(remoteRank);
    HCCL_INFO("[InsTempAlltoAllVMesh1D][RunSendRecvByRemoteRank] round[%u], remoteRank[%u], "
        "allChannelSize[%zu].", roundIdx, remoteRank, allChannels.size());
    std::vector<ChannelInfo> curChannels;
    CHK_RET(SelectChannel(remoteRank, allChannels, curChannels));
    HCCL_INFO("[InsTempAlltoAllVMesh1D][RunSendRecvByRemoteRank] round[%u], remoteRank[%u], "
        "selectedChannelSize[%zu].", roundIdx, remoteRank, curChannels.size());
    if (curChannels.size() != 1) {
        HCCL_ERROR("[InsTempAlltoAllVMesh1D][RunSendRecvByRemoteRank] selectedChannelSize[%zu] must be 1 for "
            "slot sync, remoteRank[%u].", curChannels.size(), remoteRank);
        return HCCL_E_PARA;
    }
    CHK_RET(CalcDataSplitByPortGroupCommon(tempAlgParams.sendCounts[remoteRank], dataTypeSize_, curChannels,
        sendCountsSplit_, sendSizeSplit_, sendOffsetSplit_, static_cast<u32>(curChannels.size())));
    CHK_RET(CalcDataSplitByPortGroupCommon(tempAlgParams.recvCounts[remoteRank], dataTypeSize_, curChannels,
        recvCountsSplit_, recvSizeSplit_, recvOffsetSplit_, static_cast<u32>(curChannels.size())));
    HCCL_INFO("[InsTempAlltoAllVMesh1D][RunSendRecvByRemoteRank] remoteRank[%u], sendCount[%llu], "
        "recvCount[%llu], sendSizeSplit0[%llu], recvSizeSplit0[%llu].", remoteRank,
        tempAlgParams.sendCounts[remoteRank], tempAlgParams.recvCounts[remoteRank],
        sendSizeSplit_.empty() ? 0 : sendSizeSplit_[0], recvSizeSplit_.empty() ? 0 : recvSizeSplit_[0]);
    CHK_RET(RunSendRecvByChannel(tempAlgParams, roundIdx, curChannels, remoteRank, threads, slotIdx));
    return HcclResult::HCCL_SUCCESS;
}

HcclResult InsTempAlltoAllVMesh1D::RunSendRecvByChannel(const TemplateDataParams &tempAlgParams, const u32 roundIdx,
    const std::vector<ChannelInfo> &curChannels, const u32 remoteRank,
    const std::vector<ThreadHandle> &threads, const u32 slotIdx) const
{
    u32 myRankCclBuffIdx = 0; // myRank与remoteRank交互时myRank提供的cclbuffer index
    u32 remoteCclBuffIdx = 0; // myRank与remoteRank交互时remoteRank提供的cclbuffer index
    CHK_RET(CalcCclBuffIdx(remoteRank, myRankCclBuffIdx, remoteCclBuffIdx));
    u32 queIdx = myRankCclBuffIdx * channelsPerRank_ + 1;
    if (myRankCclBuffIdx != slotIdx) {
        HCCL_ERROR("[InsTempAlltoAllVMesh1D][RunSendRecvByChannel] myRankCclBuffIdx[%u] does not match "
            "slotIdx[%u], remoteRank[%u].", myRankCclBuffIdx, slotIdx, remoteRank);
        return HCCL_E_PARA;
    }
    HCCL_INFO("[InsTempAlltoAllVMesh1D][RunSendRecvByChannel] round[%u], remoteRank[%u], "
        "myRankCclBuffIdx[%u], remoteCclBuffIdx[%u], channelsPerRank[%u], startQueIdx[%u], curChannelSize[%zu].",
        roundIdx, remoteRank, myRankCclBuffIdx, remoteCclBuffIdx, channelsPerRank_, queIdx, curChannels.size());
    for (u32 channelId = 0; channelId < curChannels.size(); channelId++) {
        if (isDmaRead_ && sendSizeSplit_[channelId] > 0) {
            CHK_RET(static_cast<HcclResult>(PreCopy(tempAlgParams, threads[queIdx], myRankCclBuffIdx, remoteRank,
                sendSizeSplit_[channelId], sendCountsSplit_[channelId], sendOffsetSplit_[channelId])));
        }
        const ChannelInfo &channelSend = curChannels[channelId]; // 发给哪个rank
        const ChannelInfo &channelRecv = curChannels[channelId]; // 收哪个rank的数据
        HCCL_INFO("[InsTempAlltoAllVMesh1D][RunSendRecvByChannel] remoteRank[%u], channelId[%u], "
            "queIdx[%u], protocol[%u], portGroupSize[%u], sendSize[%llu], recvSize[%llu].",
            remoteRank, channelId, queIdx, channelSend.protocol, channelSend.portGroupSize,
            sendSizeSplit_[channelId], recvSizeSplit_[channelId]);
        std::vector<DataSlice> txSrcSlices;
        std::vector<DataSlice> txDstSlices;
        std::vector<DataSlice> rxSrcSlices;
        std::vector<DataSlice> rxDstSlices;

        void* remoteCclBuffAddr = channelRecv.remoteCclMem.addr;
        // write模式下，本端src数据input buffer slice
        DataSlice txSrcSlice = DataSlice(tempAlgParams.buffInfo.inputPtr, tempAlgParams.sdispls[remoteRank]
            * dataTypeSize_ + sendOffsetSplit_[channelId], sendSizeSplit_[channelId], sendCountsSplit_[channelId]);
        // write模式下，远端dst数据ccl buffer slice
        DataSlice txDstSlice = DataSlice(remoteCclBuffAddr, remoteCclBuffIdx * tempAlgParams.inputSliceStride
            + tempAlgParams.buffInfo.hcclBuffBaseOff + sendOffsetSplit_[channelId], sendSizeSplit_[channelId], sendCountsSplit_[channelId]);
        // read模式下，远端src数据ccl buffer slice
        DataSlice rxSrcSlice = DataSlice(remoteCclBuffAddr, remoteCclBuffIdx * tempAlgParams.inputSliceStride
            + tempAlgParams.buffInfo.hcclBuffBaseOff + recvOffsetSplit_[channelId], recvSizeSplit_[channelId], recvCountsSplit_[channelId]);
        // read模式下，本端dst数据output buffer slice
        DataSlice rxDstSlice = DataSlice(tempAlgParams.buffInfo.outputPtr, tempAlgParams.rdispls[remoteRank]
            * dataTypeSize_ + recvOffsetSplit_[channelId], recvSizeSplit_[channelId], recvCountsSplit_[channelId]);

        txSrcSlices.push_back(txSrcSlice);
        txDstSlices.push_back(txDstSlice);
        rxSrcSlices.push_back(rxSrcSlice);
        rxDstSlices.push_back(rxDstSlice);

        DataInfo sendInfo{channelSend, {txSrcSlices, txDstSlices}, dataType_};
        DataInfo recvInfo{channelRecv, {rxSrcSlices, rxDstSlices}, dataType_};
        SendRecvInfo sendRecvInfo{{channelSend, channelRecv},
            {{txSrcSlices, txDstSlices}, {rxSrcSlices, rxDstSlices}}, dataType_};
        CHK_RET(RunSendRecv(tempAlgParams, sendRecvInfo, sendInfo, recvInfo, threads[queIdx], channelId));
        HCCL_INFO("[InsTempAlltoAllVMesh1D][RunSendRecvByChannel] do send recv on thread[%u], channelId[%u], "\
            "send size[%llu], recv size[%llu], remote rank[%u].",
            queIdx, channelId, sendSizeSplit_[channelId], recvSizeSplit_[channelId], remoteRank);
        if (!isDmaRead_ && recvSizeSplit_[channelId] > 0) {
            CHK_RET(PostCopy(tempAlgParams, threads[queIdx], myRankCclBuffIdx, remoteRank,
                recvSizeSplit_[channelId], recvCountsSplit_[channelId], recvOffsetSplit_[channelId]));
        }
        queIdx++;
    }
    return HcclResult::HCCL_SUCCESS;
}

HcclResult InsTempAlltoAllVMesh1D::RunSendRecv(const TemplateDataParams &tempAlgParams,
    const SendRecvInfo &sendRecvInfo, const DataInfo &sendInfo, const DataInfo &recvInfo,
    const ThreadHandle& thread, const u32 channelId) const
{
    if (isDmaRead_) {
        if (sendSizeSplit_[channelId] > 0 && recvSizeSplit_[channelId] > 0) {
            CHK_PRT_RET(SendRecvRead(sendRecvInfo, thread),
                HCCL_ERROR("[InsTempAlltoAllVMesh1D] RunALLtoALL SendRecvInfo failed"),
                HcclResult::HCCL_E_INTERNAL);
        } else { // 其中一个或者两个为0
            if (sendSizeSplit_[channelId] > 0) {
                CHK_PRT_RET(SendRead(sendInfo, thread),
                    HCCL_ERROR("[InsTempAlltoAllVMesh1D] RunALLtoALL sendInfo failed"),
                    HcclResult::HCCL_E_INTERNAL);
            } else if (recvSizeSplit_[channelId] > 0) {
                CHK_PRT_RET(RecvRead(recvInfo, thread),
                    HCCL_ERROR("[InsTempAlltoAllVMesh1D] RunALLtoALL recvInfo failed"),
                    HcclResult::HCCL_E_INTERNAL);
            }
        }
    } else {
        if (sendSizeSplit_[channelId] > 0 && recvSizeSplit_[channelId] > 0) {
            CHK_PRT_RET(SendRecvWrite(sendRecvInfo, thread),
                HCCL_ERROR("[InsTempAlltoAllVMesh1D] RunALLtoALL SendRecvInfo failed"),
                HcclResult::HCCL_E_INTERNAL);
        } else { // 其中一个或者两个为0
            if (sendSizeSplit_[channelId] > 0) {
                CHK_PRT_RET(SendWrite(sendInfo, thread),
                    HCCL_ERROR("[InsTempAlltoAllVMesh1D] RunALLtoALL sendInfo failed"),
                    HcclResult::HCCL_E_INTERNAL);
            }
            if (recvSizeSplit_[channelId] > 0) {
                CHK_PRT_RET(RecvWrite(recvInfo, thread),
                    HCCL_ERROR("[InsTempAlltoAllVMesh1D] RunALLtoALL recvInfo failed"),
                    HcclResult::HCCL_E_INTERNAL);
            }
        }
    }
    return HcclResult::HCCL_SUCCESS;
}

HcclResult InsTempAlltoAllVMesh1D::PreCopy(const TemplateDataParams &tempAlgParams, const ThreadHandle &thread,
    const u32 myRankCclBuffIdx, const u32 remoteRank, const u64 &sendSize,
    const u64 &sendCount, const u64 &sendOffset) const
{
    DataSlice localCopySrcSlice = DataSlice(tempAlgParams.buffInfo.inputPtr,
        tempAlgParams.sdispls[remoteRank] * dataTypeSize_ + sendOffset, sendSize, sendCount);
    DataSlice localCopyDstSlice = DataSlice(tempAlgParams.buffInfo.hcclBuff.addr,
        myRankCclBuffIdx * tempAlgParams.inputSliceStride + tempAlgParams.buffInfo.hcclBuffBaseOff + sendOffset,
        sendSize, sendCount);
    CHK_RET(static_cast<HcclResult>(LocalCopy(thread, localCopySrcSlice, localCopyDstSlice)));
    return HcclResult::HCCL_SUCCESS;
}

HcclResult InsTempAlltoAllVMesh1D::PostCopy(const TemplateDataParams &tempAlgParams, const ThreadHandle &thread,
    const u32 myRankCclBuffIdx, const u32 remoteRank, const u64 &recvSize,
    const u64 &recvCount, const u64 &recvOffset) const
{
    // ccl buffer的数据搬运到usrout
    // 远端的数据发送到本端ccl buffer的slice
    DataSlice localCopySrcSlice = DataSlice(tempAlgParams.buffInfo.hcclBuff.addr,
        myRankCclBuffIdx * tempAlgParams.inputSliceStride + tempAlgParams.buffInfo.hcclBuffBaseOff +
        recvOffset, recvSize, recvCount);
    // 本端output buffer slice
    DataSlice localCopyDstSlice = DataSlice(tempAlgParams.buffInfo.outputPtr,
        tempAlgParams.rdispls[remoteRank] * dataTypeSize_ + recvOffset,
        recvSize, recvCount);
    CHK_RET(static_cast<HcclResult>(LocalCopy(thread, localCopySrcSlice, localCopyDstSlice)));
    return HcclResult::HCCL_SUCCESS;
}

void InsTempAlltoAllVMesh1D::GetNotifyIdxMainToSub(std::vector<u32> &notifyIdxMianToSub)
{
    notifyIdxMianToSub.clear();
    if (threadNum_ <= 1) {
        return;
    }
    u32 slaveThreadNum = threadNum_ - 1;
    for (u32 slaveThreadIdx = 0; slaveThreadIdx < slaveThreadNum; slaveThreadIdx++) {
        notifyIdxMianToSub.push_back(0);
    }
}

void InsTempAlltoAllVMesh1D::GetNotifyIdxSubToMain(std::vector<u32> &notifyIdxSubToMain)
{
    notifyIdxSubToMain.clear();
    u32 notifyNum = threadNum_ - 1;
    for (u32 notifyIdx = 0; notifyIdx < notifyNum; notifyIdx++) {
        notifyIdxSubToMain.push_back(notifyIdx);
    }
}

HcclResult InsTempAlltoAllVMesh1D::GetVmeshShape(u32 &rowNum, u32 &colNum) const
{
    rowNum = 0;
    colNum = 0;
    if (subCommRanks_.size() != 1) {
        HCCL_ERROR("[InsTempAlltoAllVMesh1D][GetVmeshShape] subCommRanksSize[%zu] is not 1.",
            subCommRanks_.size());
        return HCCL_E_PARA;
    }
    if (subCommRanks_[0].size() != templateRankSize_) {
        HCCL_ERROR("[InsTempAlltoAllVMesh1D][GetVmeshShape] subCommRanks0Size[%zu] is not templateRankSize[%u].",
            subCommRanks_[0].size(), templateRankSize_);
        return HCCL_E_PARA;
    }
    if (templateRankSize_ == VMESH_2X2_RANK_NUM) {
        rowNum = VMESH_2X2_ROW_NUM;
        colNum = VMESH_2X2_COL_NUM;
        return HCCL_SUCCESS;
    }
    if (templateRankSize_ == VMESH_4X4_RANK_NUM) {
        rowNum = VMESH_4X4_ROW_NUM;
        colNum = VMESH_4X4_COL_NUM;
        return HCCL_SUCCESS;
    }
    if (templateRankSize_ == VMESH_8X8_RANK_NUM) {
        rowNum = VMESH_8X8_ROW_NUM;
        colNum = VMESH_8X8_COL_NUM;
        return HCCL_SUCCESS;
    }
    HCCL_ERROR("[InsTempAlltoAllVMesh1D][GetVmeshShape] unsupported templateRankSize[%u].", templateRankSize_);
    return HCCL_E_PARA;
}

HcclResult InsTempAlltoAllVMesh1D::GetMesh1DClosCoord(u32 rank, u32 &meshRow, u32 &meshCol) const
{
    meshRow = 0;
    meshCol = 0;
    u32 rowNum = 0;
    u32 colNum = 0;
    CHK_RET(GetVmeshShape(rowNum, colNum));
    u32 rankIndex = 0;
    CHK_RET(GetRankIndexInSubComm(rank, rankIndex));
    if (rankIndex >= templateRankSize_) {
        HCCL_ERROR("[InsTempAlltoAllVMesh1D][GetMesh1DClosCoord] rankIndex[%u] is invalid for rank[%u], "
            "templateRankSize[%u].", rankIndex, rank, templateRankSize_);
        return HCCL_E_PARA;
    }
    meshRow = rankIndex % rowNum;
    meshCol = rankIndex / rowNum;
    return HCCL_SUCCESS;
}

HcclResult InsTempAlltoAllVMesh1D::GetRankByMesh1DClosCoord(u32 meshRow, u32 meshCol, u32 &rank) const
{
    rank = 0;
    u32 rowNum = 0;
    u32 colNum = 0;
    CHK_RET(GetVmeshShape(rowNum, colNum));
    if (meshRow >= rowNum || meshCol >= colNum) {
        HCCL_ERROR("[InsTempAlltoAllVMesh1D][GetRankByMesh1DClosCoord] invalid coord meshRow[%u], meshCol[%u], "
            "rowNum[%u], colNum[%u].", meshRow, meshCol, rowNum, colNum);
        return HCCL_E_PARA;
    }
    rank = subCommRanks_[0][meshCol * rowNum + meshRow];
    return HCCL_SUCCESS;
}

HcclResult InsTempAlltoAllVMesh1D::GetRankIndexInSubComm(u32 rank, u32 &rankIndex) const
{
    if (subCommRanks_.empty()) {
        HCCL_ERROR("[InsTempAlltoAllVMesh1D][GetRankIndexInSubComm] subCommRanks_ is empty.");
        return HCCL_E_PARA;
    }
    auto iter = std::find(subCommRanks_[0].begin(), subCommRanks_[0].end(), rank);
    if (iter == subCommRanks_[0].end()) {
        HCCL_ERROR("[InsTempAlltoAllVMesh1D][GetRankIndexInSubComm] rank[%u] is not in subCommRanks_[0].", rank);
        return HCCL_E_PARA;
    }
    rankIndex = static_cast<u32>(std::distance(subCommRanks_[0].begin(), iter));
    return HCCL_SUCCESS;
}

HcclResult InsTempAlltoAllVMesh1D::SelectChannel(u32 remoteRank, const std::vector<ChannelInfo> &allChannels,
    std::vector<ChannelInfo> &selectedChannels) const
{
    selectedChannels.clear();
    if (allChannels.empty()) {
        HCCL_ERROR("[InsTempAlltoAllVMesh1D][SelectChannel] channel is empty for remoteRank[%u].", remoteRank);
        return HCCL_E_PARA;
    }
    u32 rowNum = 0;
    u32 colNum = 0;
    CHK_RET(GetVmeshShape(rowNum, colNum));

    u32 myMeshRow = 0;
    u32 myMeshCol = 0;
    u32 remoteMeshRow = 0;
    u32 remoteMeshCol = 0;
    CHK_RET(GetMesh1DClosCoord(myRank_, myMeshRow, myMeshCol));
    CHK_RET(GetMesh1DClosCoord(remoteRank, remoteMeshRow, remoteMeshCol));
    HCCL_INFO("[InsTempAlltoAllVMesh1D][SelectChannel] myRank[%u], remoteRank[%u], rowNum[%u], colNum[%u], "
        "myMeshRow[%u], myMeshCol[%u], remoteMeshRow[%u], remoteMeshCol[%u], allChannelSize[%zu].",
        myRank_, remoteRank, rowNum, colNum, myMeshRow, myMeshCol, remoteMeshRow, remoteMeshCol,
        allChannels.size());
    if (myMeshCol == remoteMeshCol) {
        selectedChannels.push_back(allChannels[0]);
        HCCL_INFO("[InsTempAlltoAllVMesh1D][SelectChannel] same mesh column selects channelIndex[0], protocol[%u].",
            allChannels[0].protocol);
        return HCCL_SUCCESS;
    }
    u32 channelIndex = remoteMeshRow;
    if (channelIndex >= allChannels.size()) {
        HCCL_ERROR("[InsTempAlltoAllVMesh1D][SelectChannel] invalid channel index[%u], channel size[%zu], "
            "remoteRank[%u], remoteMeshRow[%u].", channelIndex, allChannels.size(), remoteRank, remoteMeshRow);
        return HCCL_E_PARA;
    }
    selectedChannels.push_back(allChannels[channelIndex]);
    HCCL_INFO("[InsTempAlltoAllVMesh1D][SelectChannel] cross mesh column selects channelIndex[%u], protocol[%u].",
        channelIndex, allChannels[channelIndex].protocol);
    return HCCL_SUCCESS;
}
} // namespace Hccl