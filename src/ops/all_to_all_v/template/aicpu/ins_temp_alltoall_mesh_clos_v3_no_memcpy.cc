/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "ins_temp_alltoall_mesh_clos_v3_no_memcpy.h"
#include "alg_data_trans_wrapper.h"
#include "template_utils.h"
#include "channel.h"
#include <algorithm>
namespace ops_hccl {
namespace {
constexpr u32 COPY_THREAD_NUM = 0;
constexpr u32 COPY_NOTIFY_BASE_IDX = 1;
}

InsTempAlltoAllMeshClosV3NoMemcpy::InsTempAlltoAllMeshClosV3NoMemcpy(const OpParam &param, const u32 rankId,
                                                     const std::vector<std::vector<u32>> &subCommRanks)
    : InsTempAlltoAllMesh2DV3NoMemcpy(param, rankId, subCommRanks)
{
}

InsTempAlltoAllMeshClosV3NoMemcpy::~InsTempAlltoAllMeshClosV3NoMemcpy() {}

std::string InsTempAlltoAllMeshClosV3NoMemcpy::Describe() const
{
    std::string info = "Template of alltoall MeshClosV2 (hash-based link selection) with tempRankSize ";
    info += std::to_string(templateRankSize_);
    return info;
}

u64 InsTempAlltoAllMeshClosV3NoMemcpy::GetThreadNum() const
{
    return GetClosSlotNum() + COPY_THREAD_NUM;
}

u32 InsTempAlltoAllMeshClosV3NoMemcpy::GetClosSlotNum() const
{
    u32 rowNum = GetMatrixRowNum();
    if (rowNum > 0) {
        return rowNum;
    }
    return std::max(channelsPerRank_, 1u);
}

u32 InsTempAlltoAllMeshClosV3NoMemcpy::GetMatrixRowNum() const
{
    return meshSize_;
}

u32 InsTempAlltoAllMeshClosV3NoMemcpy::GetMatrixColNum() const
{
    if (rankSize_ == 0 || meshSize_ == 0) {
        return 0;
    }
    return rankSize_ / meshSize_;
}

u32 InsTempAlltoAllMeshClosV3NoMemcpy::GetCopyNotifySlotCount() const
{
    u32 commThreadNum = GetClosSlotNum();
    u32 rowNum = GetMatrixRowNum();
    u32 colNum = GetMatrixColNum();
    if (rowNum == 0 && channelsPerRank_ > 0) {
        rowNum = channelsPerRank_;
        colNum = templateRankSize_ > 1 ? (templateRankSize_ - 1) / rowNum + 1 : 1;
    }
    if (rowNum == 0 || colNum <= 1) {
        return 1;
    }
    return (colNum - 1) * commThreadNum;
}

HcclResult InsTempAlltoAllMeshClosV3NoMemcpy::GetRes(AlgResourceRequest &resourceRequest) const
{
    u32 threadNum = GetThreadNum();
    resourceRequest.slaveThreadNum = threadNum > 1 ? threadNum - 1 : 0;
    if (resourceRequest.slaveThreadNum > 0) {
        resourceRequest.notifyNumPerThread.assign(resourceRequest.slaveThreadNum, COPY_NOTIFY_BASE_IDX);
        resourceRequest.notifyNumPerThread.back() = COPY_NOTIFY_BASE_IDX + GetCopyNotifySlotCount();
    }
    resourceRequest.notifyNumOnMainThread = threadNum > 1 ? threadNum - 1 : 0;
    return HCCL_SUCCESS;
}

HcclResult InsTempAlltoAllMeshClosV3NoMemcpy::CalcRes(HcclComm comm, const OpParam &param,
                                               const TopoInfoWithNetLayerDetails *topoInfo,
                                               AlgResourceRequest &resourceRequest)
{
    HCCL_INFO("[InsTempAlltoAllMeshClosV3NoMemcpy][CalcRes] start");
    std::vector<HcclChannelDesc> levelChannels;
    CHK_RET(CalcChannelRequestMesh1D(comm, param, topoInfo, subCommRanks_, levelChannels));
    resourceRequest.channels.push_back(levelChannels);

    channelsPerRank_ = levelChannels.empty() ? 1 : CalcChannelsPerRank(levelChannels);
    HCCL_INFO("[InsTempAlltoAllMeshClosV3NoMemcpy][CalcRes] totalLinks[%u], channelCount[%zu]",
              channelsPerRank_, levelChannels.size());

    CHK_RET(GetRes(resourceRequest));
    return HCCL_SUCCESS;
}

HcclResult InsTempAlltoAllMeshClosV3NoMemcpy::RunAlltoAllMesh(
    const std::vector<ThreadHandle> &threads,
    const std::map<u32, std::vector<ChannelInfo>> &channels)
{
    HCCL_WARNING("[ALLTOALL_V2_DEBUG][MeshClos][RunAlltoAllMesh] Entry: rank=%d templateRankSize=%u totalLinks=%u "
              "hierarchy: xRank=%u yRank=%u totalRank=%u myRank_=%u sliceSize=%llu",
              myRank_, templateRankSize_, channelsPerRank_,
              meshSize_, closSize_, rankSize_, myRank_,
              tempAlgParams_.sliceSize);
    if (templateRankSize_ <= 1) {
        return HCCL_SUCCESS;
    }
    u32 closSlotNum = GetClosSlotNum();
    u32 rowNum = GetMatrixRowNum();
    u32 colNum = GetMatrixColNum();
    CHK_PRT_RET(rowNum == 0 || colNum == 0 || rankSize_ == 0 || rankSize_ % rowNum != 0,
                HCCL_ERROR("[ALLTOALL_NO_MEMCPY][MeshClos][RunAlltoAllMesh] invalid mesh dims. "
                           "rowNum=%u colNum=%u meshSize=%u closSize=%u rankSize=%u myRank=%d",
                           rowNum, colNum, meshSize_, closSize_, rankSize_, myRank_),
                HcclResult::HCCL_E_INTERNAL);
    CHK_PRT_RET(myRank_ >= rankSize_,
                HCCL_ERROR("[ALLTOALL_NO_MEMCPY][MeshClos][RunAlltoAllMesh] myRank[%d] >= rankSize[%u].",
                           myRank_, rankSize_),
                HcclResult::HCCL_E_INTERNAL);
    CHK_PRT_RET(threads.size() < closSlotNum + COPY_THREAD_NUM,
                HCCL_ERROR("[ALLTOALL_V2_DEBUG][MeshClos][RunAlltoAllMesh] threads[%zu] < required[%u]. "
                           "commThreads=%u copyThreads=%u myRank=%d",
                           threads.size(), closSlotNum + COPY_THREAD_NUM, closSlotNum, COPY_THREAD_NUM,
                           myRank_),
                HcclResult::HCCL_E_INTERNAL);

    const bool isPcie = IsPcieProtocol(channels);
    CHK_PRT_RET(isPcie,
                HCCL_ERROR("[ALLTOALL_NO_MEMCPY][MeshClos] pcie/read protocol is not supported."),
                HcclResult::HCCL_E_NOT_SUPPORT);

    const u32 dataTypeSize = DATATYPE_SIZE_TABLE[dataType_];
    u64 totalSliceSize = tempAlgParams_.sliceSize;
    u64 actualChunkSize = (totalSliceSize + rankSize_ - 1) / rankSize_;
    u64 chunkCount = actualChunkSize / dataTypeSize;
    HCCL_WARNING("[ALLTOALL_NO_MEMCPY][MeshClos][RunAlltoAllMesh] matrix clos schedule: "
                 "rank=%d rowNum=%u colNum=%u meshSize=%u closSize=%u rankSize=%u slotNum=%u "
                 "actualChunkSize=%llu chunkCount=%llu",
                 myRank_, rowNum, colNum, meshSize_, closSize_, rankSize_, closSlotNum,
                 actualChunkSize, chunkCount);

    std::vector<ThreadHandle> commThreads(threads.begin(), threads.begin() + closSlotNum);
    std::vector<ThreadHandle> subCommThreads;
    std::vector<u32> roundNotifyIdx;
    if (commThreads.size() > 1) {
        subCommThreads.assign(commThreads.begin() + 1, commThreads.end());
        for (u32 notifyIdx = 0; notifyIdx < subCommThreads.size(); notifyIdx++) {
            roundNotifyIdx.push_back(notifyIdx);
        }
    }
    for (u32 round = 1; round < colNum; round++) {
        std::vector<ClosNoMemcpySlot> slotPlans;
        CHK_RET(CalcClosNoMemcpyRoundPlan(round, slotPlans));
        CHK_PRT_RET(slotPlans.size() > commThreads.size(),
                    HCCL_ERROR("[ALLTOALL_NO_MEMCPY][MeshClos][RunAlltoAllMesh] slotNum[%zu] > commThreads[%zu]. "
                               "round=%u myRank=%d",
                               slotPlans.size(), commThreads.size(), round, myRank_),
                    HcclResult::HCCL_E_INTERNAL);
        for (u32 slotIdx = 0; slotIdx < slotPlans.size(); slotIdx++) {
            CHK_RET(RunClosNoMemcpySlot(channels, slotPlans[slotIdx], commThreads[slotIdx], round,
                                        actualChunkSize, chunkCount, isPcie));
        }
        if (!subCommThreads.empty()) {
            CHK_RET(PostSyncInterThreads(commThreads[0], subCommThreads, roundNotifyIdx));
        }
    }

    for (u32 i = 0; i < failedRanks_.size(); i++) {
        if (failedRanks_[i]) {
            HCCL_ERROR("[ALLTOALL_V2_DEBUG][MeshClos][RunAlltoAllMesh] Failed rank[%u] detected. "
                       "templateRank=%u myRank=%d totalLinks=%u",
                       i, templateRankSize_, myRank_, channelsPerRank_);
            return HcclResult::HCCL_E_INTERNAL;
        }
    }
    return HCCL_SUCCESS;
}

HcclResult InsTempAlltoAllMeshClosV3NoMemcpy::CalcClosNoMemcpyRoundPlan(
    u32 round, std::vector<ClosNoMemcpySlot> &slotPlans) const
{
    u32 rowNum = GetMatrixRowNum();
    u32 colNum = GetMatrixColNum();
    CHK_PRT_RET(rowNum == 0 || colNum == 0 || rankSize_ == 0 || rankSize_ % rowNum != 0,
                HCCL_ERROR("[ALLTOALL_NO_MEMCPY][MeshClos][CalcRoundPlan] invalid mesh dims. "
                           "rowNum=%u colNum=%u meshSize=%u closSize=%u rankSize=%u myRank=%d",
                           rowNum, colNum, meshSize_, closSize_, rankSize_, myRank_),
                HcclResult::HCCL_E_INTERNAL);
    CHK_PRT_RET(myRank_ >= rankSize_,
                HCCL_ERROR("[ALLTOALL_NO_MEMCPY][MeshClos][CalcRoundPlan] myRank[%d] >= rankSize[%u].",
                           myRank_, rankSize_),
                HcclResult::HCCL_E_INTERNAL);
    CHK_PRT_RET(round == 0 || round >= colNum,
                HCCL_ERROR("[ALLTOALL_NO_MEMCPY][MeshClos][CalcRoundPlan] invalid round[%u], colNum[%u]. "
                           "myRank=%d",
                           round, colNum, myRank_),
                HcclResult::HCCL_E_PARA);

    slotPlans.clear();
    u32 myRow = myRank_ % rowNum;
    u32 myCol = myRank_ / rowNum;
    u32 txCol = (myCol + round) % colNum;
    u32 rxCol = (myCol + colNum - round) % colNum;
    for (u32 rowOffset = 0; rowOffset < rowNum; rowOffset++) {
        u32 peerRow = (myRow + rowOffset) % rowNum;
        u64 txRank64 = static_cast<u64>(txCol) * rowNum + peerRow;
        u64 rxRank64 = static_cast<u64>(rxCol) * rowNum + peerRow;
        CHK_PRT_RET(txRank64 >= rankSize_ || rxRank64 >= rankSize_,
                    HCCL_ERROR("[ALLTOALL_NO_MEMCPY][MeshClos][CalcRoundPlan] slot rank out of range. "
                               "round=%u rowOffset=%u peerRow=%u txRank=%llu rxRank=%llu rankSize=%u "
                               "myRank=%d myRow=%u myCol=%u",
                               round, rowOffset, peerRow, txRank64, rxRank64, rankSize_, myRank_, myRow, myCol),
                    HcclResult::HCCL_E_INTERNAL);
        u32 txRank = static_cast<u32>(txRank64);
        u32 rxRank = static_cast<u32>(rxRank64);
        u32 channelIdx = (myRow + peerRow) % rowNum;
        slotPlans.push_back({txRank, rxRank, channelIdx, channelIdx});
    }
    HCCL_INFO("[ALLTOALL_NO_MEMCPY][MeshClos][CalcRoundPlan] myRank=%d round=%u myRow=%u myCol=%u "
              "txCol=%u rxCol=%u slotNum=%zu",
              myRank_, round, myRow, myCol, txCol, rxCol, slotPlans.size());
    return HCCL_SUCCESS;
}

HcclResult InsTempAlltoAllMeshClosV3NoMemcpy::SelectClosNoMemcpyChannel(
    const std::map<u32, std::vector<ChannelInfo>> &channels, u32 remoteRank, u32 channelIdx, ChannelInfo &channel) const
{
    auto iter = channels.find(remoteRank);
    CHK_PRT_RET(iter == channels.end(),
                HCCL_ERROR("[ALLTOALL_NO_MEMCPY][MeshClos][SelectChannel] remoteRank[%u] not found. "
                           "myRank=%d channelIdx=%u",
                           remoteRank, myRank_, channelIdx),
                HcclResult::HCCL_E_PARA);
    const std::vector<ChannelInfo> &remoteChannels = iter->second;
    u32 rowNum = GetMatrixRowNum();
    CHK_PRT_RET(rowNum == 0,
                HCCL_ERROR("[ALLTOALL_NO_MEMCPY][MeshClos][SelectChannel] invalid rowNum[0]. "
                           "remoteRank=%u myRank=%d channelIdx=%u",
                           remoteRank, myRank_, channelIdx),
                HcclResult::HCCL_E_INTERNAL);
    CHK_PRT_RET(channelIdx >= rowNum || remoteChannels.size() < rowNum,
                HCCL_ERROR("[ALLTOALL_NO_MEMCPY][MeshClos][SelectChannel] channelIdx[%u] invalid or channelNum[%zu] < rowNum[%u]. "
                           "remoteRank=%u myRank=%d channelIdx=%u rowNum=%u",
                           channelIdx, remoteChannels.size(), rowNum, remoteRank, myRank_, channelIdx, rowNum),
                HcclResult::HCCL_E_PARA);
    u32 resolvedIdx = channelIdx;
    CHK_PRT_RET(resolvedIdx >= remoteChannels.size(),
                HCCL_ERROR("[ALLTOALL_NO_MEMCPY][MeshClos][SelectChannel] resolvedIdx[%u] out of range. "
                           "remoteRank=%u channelNum=%zu channelIdx=%u myRank=%d",
                           resolvedIdx, remoteRank, remoteChannels.size(), channelIdx, myRank_),
                HcclResult::HCCL_E_PARA);
    channel = remoteChannels[resolvedIdx];
    CHK_PRT_RET(channel.remoteOutputGraphMode.addr == nullptr,
                HCCL_ERROR("[ALLTOALL_NO_MEMCPY][MeshClos][SelectChannel] remote output is null. "
                           "remoteRank=%u resolvedIdx=%u myRank=%d",
                           remoteRank, resolvedIdx, myRank_),
                HcclResult::HCCL_E_INTERNAL);
    HCCL_DEBUG("[ALLTOALL_NO_MEMCPY][MeshClos][SelectChannel] myRank=%d remoteRank=%u "
               "channelIdx=%u resolvedIdx=%u channelNum=%zu",
               myRank_, remoteRank, channelIdx, resolvedIdx, remoteChannels.size());
    return HCCL_SUCCESS;
}

HcclResult InsTempAlltoAllMeshClosV3NoMemcpy::RunClosNoMemcpySlot(
    const std::map<u32, std::vector<ChannelInfo>> &channels,
    const ClosNoMemcpySlot &slotPlan,
    const ThreadHandle &thread,
    u32 round,
    u64 actualChunkSize,
    u64 chunkCount,
    bool isPcie)
{
    CHK_PRT_RET(isPcie,
                HCCL_ERROR("[ALLTOALL_NO_MEMCPY][MeshClos] pcie/read protocol is not supported."),
                HcclResult::HCCL_E_NOT_SUPPORT);
    CHK_PRT_RET(!enableRemoteMemAccess_,
                HCCL_ERROR("[ALLTOALL_NO_MEMCPY][MeshClos][RunSlot] remote output access is disabled. "
                           "myRank=%d txRank=%u rxRank=%u round=%u txChannelIdx=%u rxChannelIdx=%u",
                           myRank_, slotPlan.txRank, slotPlan.rxRank, round,
                           slotPlan.txChannelIdx, slotPlan.rxChannelIdx),
                HcclResult::HCCL_E_INTERNAL);

    ChannelInfo txChannel;
    ChannelInfo rxChannel;
    CHK_RET(SelectClosNoMemcpyChannel(channels, slotPlan.txRank, slotPlan.txChannelIdx, txChannel));
    CHK_RET(SelectClosNoMemcpyChannel(channels, slotPlan.rxRank, slotPlan.rxChannelIdx, rxChannel));

    u64 txSrcOffset = tempAlgParams_.buffInfo.inBuffBaseOff + static_cast<u64>(slotPlan.txRank) * actualChunkSize;
    u64 txDstOffset = tempAlgParams_.buffInfo.outBuffBaseOff + static_cast<u64>(myRank_) * actualChunkSize;
    CHK_PRT_RET(txSrcOffset + actualChunkSize > tempAlgParams_.buffInfo.inputSize ||
                    txDstOffset + actualChunkSize > txChannel.remoteOutputGraphMode.size,
                HCCL_ERROR("[ALLTOALL_NO_MEMCPY][MeshClos][RunSlot] tx slice out of registered range. "
                           "myRank=%d txRank=%u rxRank=%u round=%u txChannelIdx=%u rxChannelIdx=%u "
                           "txSrcOff=%llu txDstOff=%llu chunk=%llu inputSize=%llu remoteOutputSize=%llu",
                           myRank_, slotPlan.txRank, slotPlan.rxRank, round,
                           slotPlan.txChannelIdx, slotPlan.rxChannelIdx,
                           txSrcOffset, txDstOffset, actualChunkSize,
                           tempAlgParams_.buffInfo.inputSize, txChannel.remoteOutputGraphMode.size),
                HcclResult::HCCL_E_INTERNAL);

    u64 rxSrcOffset = tempAlgParams_.buffInfo.outBuffBaseOff + static_cast<u64>(myRank_) * actualChunkSize;
    u64 rxDstOffset = tempAlgParams_.buffInfo.outBuffBaseOff + static_cast<u64>(slotPlan.rxRank) * actualChunkSize;
    CHK_PRT_RET(rxSrcOffset + actualChunkSize > rxChannel.remoteOutputGraphMode.size ||
                    rxDstOffset + actualChunkSize > tempAlgParams_.buffInfo.outputSize,
                HCCL_ERROR("[ALLTOALL_NO_MEMCPY][MeshClos][RunSlot] rx slice out of registered range. "
                           "myRank=%d txRank=%u rxRank=%u round=%u txChannelIdx=%u rxChannelIdx=%u "
                           "rxSrcOff=%llu rxDstOff=%llu chunk=%llu remoteOutputSize=%llu outputSize=%llu",
                           myRank_, slotPlan.txRank, slotPlan.rxRank, round,
                           slotPlan.txChannelIdx, slotPlan.rxChannelIdx,
                           rxSrcOffset, rxDstOffset, actualChunkSize,
                           rxChannel.remoteOutputGraphMode.size, tempAlgParams_.buffInfo.outputSize),
                HcclResult::HCCL_E_INTERNAL);

    std::vector<DataSlice> txSrcSlices;
    std::vector<DataSlice> txDstSlices;
    std::vector<DataSlice> rxSrcSlices;
    std::vector<DataSlice> rxDstSlices;

    txSrcSlices.emplace_back(tempAlgParams_.buffInfo.inputPtr, txSrcOffset, actualChunkSize, chunkCount);
    txDstSlices.emplace_back(txChannel.remoteOutputGraphMode.addr, txDstOffset, actualChunkSize, chunkCount);
    rxSrcSlices.emplace_back(rxChannel.remoteOutputGraphMode.addr, rxSrcOffset, actualChunkSize, chunkCount);
    rxDstSlices.emplace_back(tempAlgParams_.buffInfo.outputPtr, rxDstOffset, actualChunkSize, chunkCount);

    TxRxSlicesList sendRecvSlicesList({txSrcSlices, txDstSlices}, {rxSrcSlices, rxDstSlices});
    TxRxChannels sendRecvChannels(txChannel, rxChannel);
    SendRecvInfo sendRecvInfo(sendRecvChannels, sendRecvSlicesList, dataType_);

    HCCL_WARNING("[ALLTOALL_NO_MEMCPY][MeshClos][RunSlot] myRank=%d round=%u txChannelIdx=%u rxChannelIdx=%u "
                 "txRank=%u rxRank=%u txSrcOff=%llu txDstOff=%llu rxSrcOff=%llu rxDstOff=%llu chunk=%llu",
                 myRank_, round, slotPlan.txChannelIdx, slotPlan.rxChannelIdx,
                 slotPlan.txRank, slotPlan.rxRank,
                 txSrcOffset, txDstOffset, rxSrcOffset, rxDstOffset, actualChunkSize);

    HcclResult dmaResult = SendRecvWrite(sendRecvInfo, thread);
    if (dmaResult == HcclResult::HCCL_E_INTERNAL) {
        u32 failedAlgRank = 0;
        if (GetAlgRank(slotPlan.txRank, subCommRanks_[0], failedAlgRank) == HCCL_SUCCESS &&
            failedAlgRank < failedRanks_.size()) {
            failedRanks_[failedAlgRank] = 1;
        }
        if (GetAlgRank(slotPlan.rxRank, subCommRanks_[0], failedAlgRank) == HCCL_SUCCESS &&
            failedAlgRank < failedRanks_.size()) {
            failedRanks_[failedAlgRank] = 1;
        }
        HCCL_WARNING("[ALLTOALL_NO_MEMCPY][MeshClos][RunSlot] peer timed out. "
                     "myRank=%d txRank=%u rxRank=%u round=%u txChannelIdx=%u rxChannelIdx=%u",
                     myRank_, slotPlan.txRank, slotPlan.rxRank, round,
                     slotPlan.txChannelIdx, slotPlan.rxChannelIdx);
        return HCCL_SUCCESS;
    }
    CHK_PRT_RET(dmaResult != HCCL_SUCCESS,
                HCCL_ERROR("[ALLTOALL_NO_MEMCPY][MeshClos][RunSlot] SendRecvWrite failed. "
                           "myRank=%d txRank=%u rxRank=%u round=%u txChannelIdx=%u rxChannelIdx=%u err=0x%x",
                           myRank_, slotPlan.txRank, slotPlan.rxRank, round,
                           slotPlan.txChannelIdx, slotPlan.rxChannelIdx, dmaResult),
                dmaResult);
    return HCCL_SUCCESS;
}

HcclResult InsTempAlltoAllMeshClosV3NoMemcpy::RunAlltoAllOnLink(
    const std::vector<ThreadHandle> &commThreads,
    const std::map<u32, std::vector<ChannelInfo>> &channels,
    u32 linkIdx, u32 step, u32 numSteps)
{
    CHK_PRT_RET(linkIdx >= commThreads.size(),
                HCCL_ERROR("[ALLTOALL_V2_DEBUG][MeshClos][RunAlltoAllOnLink] linkIdx[%u] >= commThreads.size()[%zu] "
                           "myRank=%d templateRank=%u",
                           linkIdx, commThreads.size(), myRank_, templateRankSize_),
                HcclResult::HCCL_E_INTERNAL);
    u32 myAlgRank = 0;
    CHK_RET(GetAlgRank(myRank_, subCommRanks_[0], myAlgRank));

    bool isPcie = IsPcieProtocol(channels);
    const u32 dataTypeSize = DATATYPE_SIZE_TABLE[dataType_];
    u64 totalSliceSize = tempAlgParams_.sliceSize;
    u64 actualChunkSize = (totalSliceSize + rankSize_ - 1) / rankSize_;
    u64 chunkCount = actualChunkSize / dataTypeSize;
    HCCL_WARNING("[ALLTOALL_V2_DEBUG][MeshClos][RunAlltoAllOnLink] Stride config: "
        "inputSliceStride=%llu outputSliceStride=%llu inBuffType=%d outBuffType=%d "
        "inBuffBaseOff=%llu outBuffBaseOff=%llu outputSize=%llu actualChunkSize=%llu",
        tempAlgParams_.inputSliceStride, tempAlgParams_.outputSliceStride,
        static_cast<int>(tempAlgParams_.buffInfo.inBuffType),
        static_cast<int>(tempAlgParams_.buffInfo.outBuffType),
        tempAlgParams_.buffInfo.inBuffBaseOff,
        tempAlgParams_.buffInfo.outBuffBaseOff,
        tempAlgParams_.buffInfo.outputSize,
        actualChunkSize);


    for (u32 peer = 0; peer < subCommRanks_[0].size(); ++peer) {
        u32 rankInSubcomm = subCommRanks_[0][peer];
        if (rankInSubcomm == myRank_) {
            continue;
        }
        if (channels.count(rankInSubcomm) == 0 || channels.at(rankInSubcomm).empty()) {
            HCCL_ERROR("[ALLTOALL_V2_DEBUG][MeshClos][RunAlltoAllOnLink] Peer %u has no channels. "
                       "myRank=%d linkIdx=%u channels.size=%zu templateRank=%u",
                       rankInSubcomm, myRank_, linkIdx, channels.size(), templateRankSize_);
            return HcclResult::HCCL_E_INTERNAL;
        }
    }

    for (u32 neighborIdx = 0; neighborIdx < subCommRanks_[0].size() - 1; neighborIdx++) {
        u32 connectedRank = subCommRanks_[0][(myAlgRank + 1 + neighborIdx) % subCommRanks_[0].size()];

        if ((myRank_ ^ connectedRank) % numSteps != step) {
            continue;
        }

        u32 connectedAlgRank = 0;
        CHK_RET(GetAlgRank(connectedRank, subCommRanks_[0], connectedAlgRank));

        if (failedRanks_[connectedAlgRank]) {
            HCCL_WARNING("[ALLTOALL_V2_DEBUG][MeshClos] linkIdx[%u] rank[%d] peer[%u] already failed, skipping.",
                      linkIdx, myRank_, connectedRank);
            continue;
        }

        auto it = channels.find(connectedRank);
        if (it == channels.end() || it->second.empty()) {
            HCCL_ERROR("[ALLTOALL_V2_DEBUG][MeshClos][RunAlltoAllOnLink] Rank[%d] connectedRank[%u] has no channels. "
                       "linkIdx=%u channels.size=%zu templateRank=%u",
                       myRank_, connectedRank, linkIdx, channels.size(), templateRankSize_);
            return HcclResult::HCCL_E_INTERNAL;
        }

        u32 totalLinksToNeighbor = it->second.size();
        u32 selectedLinkIdx = (myRank_ ^ connectedRank) % commThreads.size();

        if (selectedLinkIdx >= it->second.size()) {
            HCCL_ERROR("[ALLTOALL_V2_DEBUG][MeshClos][RunAlltoAllOnLink] selectedLinkIdx OOB: "
                       "selectedLinkIdx=%u >= channels[%u].size()=%zu. myRank=%d connectedRank=%u",
                       selectedLinkIdx, connectedRank, it->second.size(), myRank_, connectedRank);
            continue;
        }

        if (!it->second[selectedLinkIdx].remoteCclMem.addr) {
            HCCL_ERROR("[ALLTOALL_V2_DEBUG][MeshClos][RunAlltoAllOnLink] remoteCclMem.addr is NULL at selectedLinkIdx: "
                       "selectedLinkIdx=%u connectedRank=%u myRank=%d",
                       selectedLinkIdx, connectedRank, myRank_);
            continue;
        }

        if (selectedLinkIdx != linkIdx) {
            continue;
        }

        HCCL_WARNING("[ALLTOALL_V2_DEBUG][MeshClos][RunAlltoAllOnLink] linkIdx[%u] matched: "
                  "myRank=%d connectedRank=%u selectedLinkIdx=%u/%u threads=%zu "
                  "enableRemoteMemAccess=%d isPcie=%d",
                  linkIdx, myRank_, connectedRank, selectedLinkIdx,
                  totalLinksToNeighbor, commThreads.size(),
                  enableRemoteMemAccess_, isPcie);

        if (linkIdx >= channels.at(connectedRank).size()) {
            HCCL_ERROR("[ALLTOALL_V2_DEBUG][MeshClos][RunAlltoAllOnLink] channel index OOB: linkIdx[%u] >= "
                       "channels[%u].size()=%zu. myRank=%d templateRank=%u",
                       linkIdx, connectedRank, channels.at(connectedRank).size(), myRank_, templateRankSize_);
            return HCCL_E_INTERNAL;
        }

        const ChannelInfo &linkRemote = it->second[linkIdx];
        void *remoteCclBuffAddr = linkRemote.remoteCclMem.addr;
        if (!remoteCclBuffAddr) {
            HCCL_ERROR("[ALLTOALL_V2_DEBUG][MeshClos][RunAlltoAllOnLink] remoteCclMem.addr is NULL for peer %u. "
                       "myRank=%d connectedRank=%u linkIdx=%u templateRank=%u",
                       connectedRank, myRank_, connectedRank, linkIdx, templateRankSize_);
            return HCCL_E_INTERNAL;
        }

        std::vector<DataSlice> txSrcSlicesAll;
        std::vector<DataSlice> txDstSlicesAll;
        std::vector<DataSlice> rxDstSlicesAll;
        std::vector<DataSlice> rxSrcSlicesAll;

        // tx 远端写
        void *txSrcPtr = tempAlgParams_.buffInfo.inputPtr;
        u64 txSrcOffset = tempAlgParams_.buffInfo.inBuffBaseOff + connectedRank * actualChunkSize;
        txSrcSlicesAll.emplace_back(txSrcPtr, txSrcOffset, actualChunkSize, chunkCount);

        CHK_PRT_RET(!enableRemoteMemAccess_ || linkRemote.remoteOutputGraphMode.addr == nullptr,
                    HCCL_ERROR("[ALLTOALL_NO_MEMCPY][MeshClos] remote output is unavailable. "
                               "myRank=%d connectedRank=%u linkIdx=%u enableRemoteMemAccess=%d",
                               myRank_, connectedRank, linkIdx, enableRemoteMemAccess_),
                    HcclResult::HCCL_E_INTERNAL);

        void *txDstPtr = linkRemote.remoteOutputGraphMode.addr;
        u64 txDstOffset = tempAlgParams_.buffInfo.outBuffBaseOff + myRank_ * actualChunkSize;
        CHK_PRT_RET(txSrcOffset + actualChunkSize > tempAlgParams_.buffInfo.inputSize ||
                        txDstOffset + actualChunkSize > linkRemote.remoteOutputGraphMode.size,
                    HCCL_ERROR("[ALLTOALL_NO_MEMCPY][MeshClos] slice out of registered range. "
                               "myRank=%d peer=%u txSrcOff=%llu txDstOff=%llu chunk=%llu "
                               "inputSize=%llu remoteOutputSize=%llu",
                               myRank_, connectedRank, txSrcOffset, txDstOffset, actualChunkSize,
                               tempAlgParams_.buffInfo.inputSize, linkRemote.remoteOutputGraphMode.size),
                    HcclResult::HCCL_E_INTERNAL);
        txDstSlicesAll.emplace_back(txDstPtr, txDstOffset, actualChunkSize, chunkCount);

        // no-memcpy mode只使用远端写；rx slice仅用于SendRecvInfo占位。
        void *rxSrcPtr = linkRemote.remoteOutputGraphMode.addr;
        u64 rxSrcOffset = tempAlgParams_.buffInfo.outBuffBaseOff + myRank_ * actualChunkSize;
        rxSrcSlicesAll.emplace_back(rxSrcPtr, rxSrcOffset, actualChunkSize, chunkCount);
        
        void *rxDstPtr = tempAlgParams_.buffInfo.outputPtr;
        u64 rxOutOffset = tempAlgParams_.buffInfo.outBuffBaseOff + connectedRank * actualChunkSize;
        rxDstSlicesAll.emplace_back(rxDstPtr, rxOutOffset, actualChunkSize, chunkCount);

        HCCL_WARNING(
            "[ALLTOALL_V3_DEBUG][Mesh2D][RunAlltoAllMesh] rank[%d]->peer[%d] "
            "txSrcOff=%llu txDstOff=%llu rxSrcOff=%llu rxDstOff=%llu "
            "actualSz=%llu",
            myRank_, connectedRank,
            txSrcOffset, txDstOffset, rxSrcOffset, rxOutOffset,
            actualChunkSize
        );

        TxRxSlicesList sendRecvSlicesList({txSrcSlicesAll, txDstSlicesAll},
                                          {rxSrcSlicesAll, rxDstSlicesAll});
        TxRxChannels sendRecvChannels(linkRemote, linkRemote);
        SendRecvInfo sendRecvInfo(sendRecvChannels, sendRecvSlicesList, dataType_);

        CHK_PRT_RET(isPcie,
                    HCCL_ERROR("[ALLTOALL_NO_MEMCPY][MeshClos] pcie/read protocol is not supported."),
                    HcclResult::HCCL_E_NOT_SUPPORT);
        HcclResult dmaResult = SendRecvWrite(sendRecvInfo, commThreads[linkIdx]);

        if (dmaResult == HcclResult::HCCL_E_INTERNAL) {
            failedRanks_[connectedAlgRank] = 1;
            HCCL_WARNING("[ALLTOALL_V2_DEBUG][MeshClos] linkIdx[%u] peer %u timed out. "
                         "myRank=%d templateRank=%u",
                         linkIdx, connectedRank, myRank_, templateRankSize_);
            continue;
        }

        if (dmaResult != HCCL_SUCCESS) {
            HCCL_ERROR("[ALLTOALL_V2_DEBUG][MeshClos] send/recv FAILED: linkIdx=%u connectedRank=%u err=0x%x "
                       "myRank=%d templateRank=%u actualChunkSize=%llu",
                       linkIdx, connectedRank, dmaResult, myRank_, templateRankSize_, actualChunkSize);
            return dmaResult;
        }

    }
    return HCCL_SUCCESS;
}


HcclResult InsTempAlltoAllMeshClosV3NoMemcpy::LocalDataCopy(const std::vector<ThreadHandle> &threads)
{
    if (threads.empty()) {
        return HcclResult::HCCL_E_INTERNAL;
    }

    if (rankSize_ == 0) {
        HCCL_ERROR("[ALLTOALL_V3_DEBUG][Mesh2D][LocalDataCopy] totalRankSize_ is 0.");
        return HCCL_E_INTERNAL;
    }
    const u32 dataTypeSize = DATATYPE_SIZE_TABLE[dataType_];
    u64 totalSize = tempAlgParams_.sliceSize;
    u64 cellSize = (totalSize + rankSize_ - 1) / rankSize_;
    u64 cellCount = cellSize / dataTypeSize;

    u32 myRank = myRank_;
    u32 meshDataOffset = (myRank / meshSize_) * meshSize_;

    if (meshDataOffset > 0) {
        u64 inputOffset = tempAlgParams_.buffInfo.inBuffBaseOff;
        u64 scratchOffset = tempAlgParams_.buffInfo.hcclBuffBaseOff;

        DataSlice srcSlice(tempAlgParams_.buffInfo.inputPtr, inputOffset, cellSize * meshDataOffset, cellCount * meshDataOffset);
        DataSlice dstSlice(tempAlgParams_.buffInfo.hcclBuff.addr, scratchOffset, cellSize * meshDataOffset, cellCount * meshDataOffset);

        CHK_RET(LocalCopy(threads[0], srcSlice, dstSlice));
    }

    if (meshDataOffset < rankSize_ / meshSize_ - 1) {
        u64 inputOffset = tempAlgParams_.buffInfo.inBuffBaseOff + cellSize * (meshDataOffset + meshSize_);
        u64 scratchOffset = tempAlgParams_.buffInfo.hcclBuffBaseOff + cellSize * (meshDataOffset + meshSize_);
        u32 reamindCount = (rankSize_ / meshSize_ - 1 - meshDataOffset) * meshDataOffset;

        DataSlice srcSlice(tempAlgParams_.buffInfo.inputPtr, inputOffset, cellSize * reamindCount, cellCount * reamindCount);
        DataSlice dstSlice(tempAlgParams_.buffInfo.hcclBuff.addr, scratchOffset, cellSize * reamindCount, cellCount * reamindCount);

        CHK_RET(LocalCopy(threads[0], srcSlice, dstSlice));
    }

    return HcclResult::HCCL_SUCCESS;
}

HcclResult InsTempAlltoAllMeshClosV3NoMemcpy::PostLocalCopy(const std::vector<ThreadHandle> &threads)
{
    if (threads.empty()) {
        return HcclResult::HCCL_E_INTERNAL;
    }

    if (rankSize_ == 0) {
        HCCL_ERROR("[ALLTOALL_V3_DEBUG][Mesh2D][LocalDataCopy] totalRankSize_ is 0.");
        return HCCL_E_INTERNAL;
    }
    const u32 dataTypeSize = DATATYPE_SIZE_TABLE[dataType_];
    u64 totalSize = tempAlgParams_.sliceSize;
    u64 cellSize = (totalSize + rankSize_ - 1) / rankSize_;
    u64 cellCount = cellSize / dataTypeSize;

    u32 myRank = myRank_;
    u32 meshDataOffset = (myRank / meshSize_) * meshSize_;

    u64 scratchOffset = tempAlgParams_.buffInfo.hcclBuffBaseOff + cellSize * meshDataOffset;
    u64 outputOffset = tempAlgParams_.buffInfo.outBuffBaseOff + cellSize * meshDataOffset;

    DataSlice srcSlice(tempAlgParams_.buffInfo.hcclBuff.addr, scratchOffset, cellSize * meshSize_, cellCount * meshSize_);
    DataSlice dstSlice(tempAlgParams_.buffInfo.outputPtr, outputOffset, cellSize * meshSize_, cellCount * meshSize_);

    CHK_RET(LocalCopy(threads[0], srcSlice, dstSlice));

    if (meshDataOffset > 0) {
        u64 scratchOffset = tempAlgParams_.buffInfo.hcclBuffBaseOff;
        u64 outputOffset = tempAlgParams_.buffInfo.outBuffBaseOff;

        DataSlice srcSlice(tempAlgParams_.buffInfo.hcclBuff.addr, scratchOffset, cellSize * meshDataOffset, cellCount * meshDataOffset);
        DataSlice dstSlice(tempAlgParams_.buffInfo.outputPtr, outputOffset, cellSize * meshDataOffset, cellCount * meshDataOffset);
        
        CHK_RET(LocalCopy(threads[0], srcSlice, dstSlice));
    }

    if (meshDataOffset < rankSize_ / meshSize_ - 1) {
        u64 scratchOffset = tempAlgParams_.buffInfo.hcclBuffBaseOff + cellSize * (meshDataOffset + meshSize_);
        u64 outputOffset = tempAlgParams_.buffInfo.outBuffBaseOff + cellSize * (meshDataOffset + meshSize_);
        u32 reamindCount = (rankSize_ / meshSize_ - 1 - meshDataOffset) * meshDataOffset;

        DataSlice srcSlice(tempAlgParams_.buffInfo.hcclBuff.addr, scratchOffset, cellSize * reamindCount, cellCount * reamindCount);
        DataSlice dstSlice(tempAlgParams_.buffInfo.outputPtr, outputOffset, cellSize * reamindCount, cellCount * reamindCount);

        CHK_RET(LocalCopy(threads[0], srcSlice, dstSlice));
    }

    return HcclResult::HCCL_SUCCESS;
}

}  // namespace ops_hccl
