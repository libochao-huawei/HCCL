/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#include "ins_temp_all_gather_mesh_1D_Z_axis_detour.h"
#include "alg_data_trans_wrapper.h"
#include "template_utils.h"

namespace ops_hccl {
bool InsTempAllGatherMesh1D1DZAxisDetour::isNew;
InsTempAllGatherMesh1D1DZAxisDetour::InsTempAllGatherMesh1D1DZAxisDetour(const OpParam &param, const u32 rankId,
                                               const std::vector<std::vector<u32>> &subCommRanks)
    : InsTempAllGatherMesh1D(param, rankId, subCommRanks)
{
}
InsTempAllGatherMesh1D1DZAxisDetour::~InsTempAllGatherMesh1D1DZAxisDetour() {}

HcclResult InsTempAllGatherMesh1D1DZAxisDetour::CalcRes(HcclComm comm, const OpParam &param, const TopoInfoWithNetLayerDetails *topoInfo,
                                           AlgResourceRequest &resourceRequest)
{
    HCCL_INFO("[InsTempAllGatherMesh1D1DZAxisDetour][CalcRes] start");
    std::vector<HcclChannelDesc> level0Channels;
    CHK_RET(CalcChannelRequestMesh1DLevel0(comm, param, topoInfo, subCommRanks_, level0Channels));
    std::vector<HcclChannelDesc> level1Channels;
    CHK_RET(CalcChannelRequestMesh1DLevel1(comm, param, topoInfo, subCommRanks_, level1Channels));
    level0ChannelNumPerRank_ = level0Channels.empty() ? 0 : CalcChannelsPerRank(level0Channels);
    level1ChannelNumPerRank_ = level1Channels.empty() ? 0 : CalcChannelsPerRank(level1Channels);
    channelsPerRank_ = level0ChannelNumPerRank_ + level1ChannelNumPerRank_;
    std::vector<HcclChannelDesc> mergedChannels;
    HCCL_INFO("level0Channels[%d]level1Channels[%d]\n", level0Channels.size(), level1Channels.size());
    mergedChannels.insert(mergedChannels.end(), level0Channels.begin(), level0Channels.end());
    mergedChannels.insert(mergedChannels.end(), level1Channels.begin(), level1Channels.end());
    resourceRequest.channels.push_back(mergedChannels);
    HCCL_INFO("mergedChannels[%d]\n", mergedChannels.size());
    channelsSize = mergedChannels.size();

    if(subCommRanks_.size() <= COMM_LEVEL0) {
        return HCCL_E_PARA;
    }
    GetRes(resourceRequest);
    return HCCL_SUCCESS;
}
HcclResult InsTempAllGatherMesh1D1DZAxisDetour::GetRes(AlgResourceRequest &resourceRequest) const
{
    u32 threadNum = templateRankSize_ > 1 ? ((templateRankSize_ - 1) * channelsPerRank_) : 1;
    HCCL_INFO("[InsTempAllGatherMesh1D1DZAxisDetour][GetRes] threadNum[%u]", threadNum);
    resourceRequest.slaveThreadNum = threadNum - 1;
    resourceRequest.notifyNumPerThread.assign(resourceRequest.slaveThreadNum, 1);
    resourceRequest.notifyNumOnMainThread = threadNum - 1;
    return HCCL_SUCCESS;
}

u64 InsTempAllGatherMesh1D1DZAxisDetour::GetThreadNum() const
{
    u32 threadNum = templateRankSize_ > 1 ? ((templateRankSize_ - 1) * channelsPerRank_) : 1;
    HCCL_INFO("[InsTempAllGatherMesh1D1DZAxisDetour][GetThreadNum] templateRankSize_[%u] channelsPerRank_[%u] threadNum[%u]", templateRankSize_, channelsPerRank_, threadNum);
    return threadNum;
}

HcclResult InsTempAllGatherMesh1D1DZAxisDetour::CalcDataSplitByPortGroup(
    const u64 totalDataCount, const u64 dataTypeSize,
    const std::vector<ChannelInfo> &channels,
    std::vector<u64> &elemCountOut, std::vector<u64> &sizeOut,
    std::vector<u64> &elemOffset)
{
    HCCL_INFO("[InsTempAllGatherMesh1D1DZAxisDetour][CalcDataSplitByPortGroup] Run Start[%u][%u][%f]\n", level0ChannelNumPerRank_, level1ChannelNumPerRank_, level0DataRatio_);
    return CalcDataSplitByPortGroupZAxisDetour(totalDataCount, dataTypeSize, channels,
        elemCountOut, sizeOut, elemOffset,
        level0ChannelNumPerRank_, level1ChannelNumPerRank_, level0DataRatio_);
}

HcclResult InsTempAllGatherMesh1D1DZAxisDetour::SetchannelsPerRank(
    const std::map<u32, std::vector<ChannelInfo>> &channels)
{
    CHK_PRT_RET(channels.empty(), HCCL_ERROR("[SetchannelsPerRank] channels is empty."), HCCL_E_INTERNAL);
    channelsPerRank_ = CalcChannelsPerRank(channels);
    if (channelsPerRank_ > 1) {
        level0ChannelNumPerRank_ = MESH_CHANNELS_NUM;
        level1ChannelNumPerRank_ = channelsPerRank_ - level0ChannelNumPerRank_;
        level0DataRatio_ = 0.5f;
    }
    HCCL_INFO("[InsTempAllGatherMesh1D1DZAxisDetour][SetchannelsPerRank], channelsPerRank_[%u], "
              "level0ChannelNumPerRank_[%u], level1ChannelNumPerRank_[%u], level0DataRatio_[%.2f]",
              channelsPerRank_, level0ChannelNumPerRank_, level1ChannelNumPerRank_, level0DataRatio_);
    return HCCL_SUCCESS;
}

std::vector<ChannelInfo> InsTempAllGatherMesh1D1DZAxisDetour::PrepareMergedChannels(
    const std::map<u32, std::vector<ChannelInfo>> &channels)
{
    std::vector<ChannelInfo> mergedChannels;
    auto& ranks = subCommRanks_[COMM_LEVEL0];
    for (u32 i = 0; i < ranks.size(); i++) {
        if (ranks[i] == myRank_) {
            continue;
        }
        mergedChannels.insert(mergedChannels.end(), channels.at(ranks[i]).begin(), channels.at(ranks[i]).end());
    }
    return mergedChannels;
}

u64 InsTempAllGatherMesh1D1DZAxisDetour::CalcSliceSizeForChannel(u32 myAlgRank, u32 connectedAlgRank, bool dmaRead) const
{
    u64 sliceSize = tempAlgParams_.sliceSize;
    HCCL_INFO("[InsTempAllGatherMesh1D1DZAxisDetour] sliceSize[%u]\n", sliceSize);
    if (dmaRead) {
        if (tempAlgParams_.tailSize != 0 && connectedAlgRank == templateRankSize_ - 1) {
            sliceSize = tempAlgParams_.tailSize;
            HCCL_INFO("[InsTempAllGatherMesh1D1DZAxisDetour] sliceSize[%u]\n", sliceSize);
        }
    } else if (tempAlgParams_.tailSize != 0 && myAlgRank == templateRankSize_ - 1) {
        sliceSize = tempAlgParams_.tailSize;
        HCCL_INFO("[InsTempAllGatherMesh1D1DZAxisDetour] sliceSize[%u]\n", sliceSize);
    }
    return sliceSize;
}

void InsTempAllGatherMesh1D1DZAxisDetour::BuildDataSlicesForChannel(
    u32 connectedRank, u32 myAlgRank, u32 connectedAlgRank, u32 idx,
    const ChannelInfo &linkRemote, void *remoteCclBuffAddr,
    std::vector<DataSlice> &txSrcSlicesAll, std::vector<DataSlice> &txDstSlicesAll,
    std::vector<DataSlice> &rxDstSlicesAll, std::vector<DataSlice> &rxSrcSlicesAll)
{
    for (u32 rpt = 0; rpt < tempAlgParams_.repeatNum; ++rpt) {
        const u64 outBaseOff = tempAlgParams_.buffInfo.outBuffBaseOff + rpt * tempAlgParams_.outputRepeatStride;
        const u64 scratchRepeatStride = tempAlgParams_.sliceSize * templateRankSize_;
        u64 scratchBase = tempAlgParams_.buffInfo.hcclBuffBaseOff + rpt * scratchRepeatStride;
        if (tempAlgParams_.buffInfo.inBuffType == BufferType::HCCL_BUFFER) {
            scratchBase = tempAlgParams_.buffInfo.hcclBuffBaseOff + rpt * tempAlgParams_.inputRepeatStride;
        }
        u64 txOutOffset = tempAlgParams_.outputSliceStride * myAlgRank + outBaseOff + elemOffset_[idx];
        u64 txScratchOffset = scratchBase + tempAlgParams_.sliceSize * myAlgRank + elemOffset_[idx];
        u64 txDstOffset = (!enableRemoteMemAccess_) ? txScratchOffset : txOutOffset;
        u64 rxOutOffset = tempAlgParams_.outputSliceStride * connectedAlgRank + outBaseOff + elemOffset_[idx];
        u64 rxScratchOffset = scratchBase + tempAlgParams_.inputSliceStride * connectedAlgRank + elemOffset_[idx];
        u64 rxSrcOffset = (!enableRemoteMemAccess_) ? rxScratchOffset : rxOutOffset;
        void *txSrcPtr = tempAlgParams_.buffInfo.outputPtr;
        void *txDstPtr = (!enableRemoteMemAccess_) ? remoteCclBuffAddr : linkRemote.remoteOutputGraphMode.addr;
        void *rxSrcPtr = (!enableRemoteMemAccess_) ? remoteCclBuffAddr : linkRemote.remoteOutputGraphMode.addr;
        void *rxDstPtr = tempAlgParams_.buffInfo.outputPtr;
        txSrcSlicesAll.emplace_back(txSrcPtr, txOutOffset, sizeOut_[idx], elemCountOut_[idx]);
        txDstSlicesAll.emplace_back(txDstPtr, txDstOffset, sizeOut_[idx], elemCountOut_[idx]);
        rxDstSlicesAll.emplace_back(rxDstPtr, rxOutOffset, sizeOut_[idx], elemCountOut_[idx]);
        rxSrcSlicesAll.emplace_back(rxSrcPtr, rxSrcOffset, sizeOut_[idx], elemCountOut_[idx]);
        HCCL_DEBUG("[InsTempAllGatherMesh1D1DZAxisDetour][RunAllGatherMesh] rankId [%d] connectedRank [%d] rpt [%d] txSrcSlices: "
                   "offset[%d] sliceSize[%d] count[%d].", myRank_, connectedRank, rpt, txOutOffset, sizeOut_[idx], elemCountOut_[idx]);
        HCCL_DEBUG("[InsTempAllGatherMesh1D1DZAxisDetour][RunAllGatherMesh] rankId [%d] connectedRank [%d] rpt [%d] txDstSlices: "
                   "offset[%d] sliceSize[%d] count[%d].", myRank_, connectedRank, rpt, txDstOffset, sizeOut_[idx], elemCountOut_[idx]);
        HCCL_DEBUG("[InsTempAllGatherMesh1D1DZAxisDetour][RunAllGatherMesh] rankId [%d] connectedRank [%d] rpt [%d] rxSrcSlices: "
                   "offset[%d] sliceSize[%d] count[%d].", myRank_, connectedRank, rpt, rxOutOffset, sizeOut_[idx], elemCountOut_[idx]);
        HCCL_DEBUG("[InsTempAllGatherMesh1D1DZAxisDetour][RunAllGatherMesh] rankId [%d] connectedRank [%d] rpt [%d] rxDrcSlices: "
                   "offset[%d] sliceSize[%d] count[%d].", myRank_, connectedRank, rpt, rxSrcOffset, sizeOut_[idx], elemCountOut_[idx]);
    }
}

HcclResult InsTempAllGatherMesh1D1DZAxisDetour::ExecuteSendRecvForChannel(
    u32 threadIdx, bool dmaRead, const std::vector<ThreadHandle> &threads,
    const ChannelInfo &linkRemote,
    const std::vector<DataSlice> &txSrcSlicesAll, const std::vector<DataSlice> &txDstSlicesAll,
    const std::vector<DataSlice> &rxSrcSlicesAll, const std::vector<DataSlice> &rxDstSlicesAll)
{
    TxRxSlicesList sendRecvSlicesList({txSrcSlicesAll, txDstSlicesAll}, {rxSrcSlicesAll, rxDstSlicesAll});
    TxRxChannels sendRecvChannels(linkRemote, linkRemote);
    SendRecvInfo sendRecvInfo(sendRecvChannels, sendRecvSlicesList);
    if (dmaRead) {
        CHK_PRT_RET(SendRecvRead(sendRecvInfo, threads[threadIdx]),
                    HCCL_ERROR("[InsTempAllGatherMesh1D1DZAxisDetour] RunAllGather Send failed"), HcclResult::HCCL_E_INTERNAL);
    } else {
        CHK_PRT_RET(SendRecvWrite(sendRecvInfo, threads[threadIdx]),
                    HCCL_ERROR("[InsTempAllGatherMesh1D1DZAxisDetour] RunAllGather Send failed"), HcclResult::HCCL_E_INTERNAL);
    }
    return HCCL_SUCCESS;
}

HcclResult InsTempAllGatherMesh1D1DZAxisDetour::ProcessSingleChannel(
    u32 threadIdx, u32 myAlgRank, bool dmaRead, const u32 dataTypeSize,
    const std::vector<ThreadHandle> &threads,
    const std::map<u32, std::vector<ChannelInfo>> &channels,
    const std::vector<ChannelInfo> &mergedChannels)
{
    u32 connectedRank = mergedChannels[threadIdx].remoteRank;
    const std::vector<ChannelInfo> &curChannels = channels.at(connectedRank);
    channelsPerRank_ = curChannels.size();
    HCCL_INFO("[RunAllGatherMesh]channelsPerRank_[%u]\n", channelsPerRank_);
    u32 idx = (channelsPerRank_ == 0) ? 0 : (threadIdx % channelsPerRank_);
    u32 connectedAlgRank = 0;
    CHK_RET(GetAlgRank(connectedRank, subCommRanks_[0], connectedAlgRank));
    HCCL_INFO("[InsTempAllGatherMesh1D1DZAxisDetour] RunAllGatherMesh RankIDs[%d], connectedRank[%d], connectedAlgRank[%d].",
              myRank_, connectedRank, connectedAlgRank);
    u64 sliceSize = CalcSliceSizeForChannel(myAlgRank, connectedAlgRank, dmaRead);
    u64 sliceCount = sliceSize / dataTypeSize;
    HCCL_INFO("[InsTempAllGatherMesh1D1DZAxisDetour] silceCount[%u]\n", sliceCount);
    elemCountOut_.clear();
    sizeOut_.clear();
    elemOffset_.clear();
    CHK_RET(CalcDataSplitByPortGroup(sliceCount, dataTypeSize, curChannels, elemCountOut_, sizeOut_, elemOffset_));
    CHK_PRT_RET(threadIdx >= threads.size() || channels.count(connectedRank) == 0 || channels.at(connectedRank).empty(),
                HCCL_ERROR("[InsTempAllGatherMesh1D1DZAxisDetour][RankID]=%u threadIdx=%u invalid params", myRank_, threadIdx),
                HcclResult::HCCL_E_INTERNAL);
    const ChannelInfo &linkRemote = channels.at(connectedRank)[idx];
    void *remoteCclBuffAddr = linkRemote.remoteCclMem.addr;
    std::vector<DataSlice> txSrcSlicesAll, txDstSlicesAll, rxDstSlicesAll, rxSrcSlicesAll;
    BuildDataSlicesForChannel(connectedRank, myAlgRank, connectedAlgRank, idx, linkRemote, remoteCclBuffAddr,
                              txSrcSlicesAll, txDstSlicesAll, rxDstSlicesAll, rxSrcSlicesAll);
    return ExecuteSendRecvForChannel(threadIdx, dmaRead, threads, linkRemote, txSrcSlicesAll, txDstSlicesAll, rxSrcSlicesAll, rxDstSlicesAll);
}

HcclResult InsTempAllGatherMesh1D1DZAxisDetour::RunAllGatherMesh(const std::vector<ThreadHandle> &threads,
                                                    const std::map<u32, std::vector<ChannelInfo>> &channels)
{
    HCCL_INFO("[InsTempAllGatherMesh1D1DZAxisDetour] RunAllGatherMesh RankIDs[%d].", myRank_);
    u32 myAlgRank = 0;
    CHK_RET(GetAlgRank(myRank_, subCommRanks_[0], myAlgRank));
    const u32 dataTypeSize = DATATYPE_SIZE_TABLE[dataType_];
    bool dmaRead = (tempAlgParams_.buffInfo.inBuffType == BufferType::HCCL_BUFFER && 
                    tempAlgParams_.buffInfo.outBuffType != BufferType::HCCL_BUFFER);
    std::vector<ChannelInfo> mergedChannels = PrepareMergedChannels(channels);
    u32 threadNum = mergedChannels.size();
    HCCL_DEBUG("threadNum %u\n", threadNum);
    for (u32 threadIdx = 0; threadIdx < threadNum; threadIdx++) {
        CHK_RET(ProcessSingleChannel(threadIdx, myAlgRank, dmaRead, dataTypeSize, threads, channels, mergedChannels));
    }
    return HcclResult::HCCL_SUCCESS;
}

}  // namespace ops_hccl