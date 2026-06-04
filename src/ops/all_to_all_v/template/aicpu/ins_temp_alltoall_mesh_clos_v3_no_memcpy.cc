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
    return std::max(GetClosSlotNum(), 1u) + COPY_THREAD_NUM;
}

u32 InsTempAlltoAllMeshClosV3NoMemcpy::GetClosSlotNum() const
{
    return meshSize_ == 0 ? channelsPerRank_ : meshSize_;
}

u32 InsTempAlltoAllMeshClosV3NoMemcpy::GetCopyNotifySlotCount() const
{
    u32 commThreadNum = GetClosSlotNum() == 0 ? 1 : GetClosSlotNum();
    u32 peerNum = templateRankSize_ > 0 ? templateRankSize_ - 1 : 0;
    if (peerNum == 0) {
        return 1;
    }
    u32 stepNum = (peerNum + commThreadNum - 1) / commThreadNum;
    return stepNum * commThreadNum;
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
    HCCL_WARNING("[ALLTOALL_NO_MEMCPY][MeshClos][RunAlltoAllMesh] Entry: rank=%d templateRankSize=%u "
                 "totalLinks=%u hierarchy: xRank=%u yRank=%u totalRank=%u myRank_=%u sliceSize=%llu",
                 myRank_, templateRankSize_, channelsPerRank_, meshSize_, closSize_, rankSize_, myRank_,
                 tempAlgParams_.sliceSize);
    if (templateRankSize_ <= 1) {
        return HCCL_SUCCESS;
    }
    CHK_PRT_RET(IsPcieProtocol(channels),
                HCCL_ERROR("[ALLTOALL_NO_MEMCPY][MeshClos][RunAlltoAllMesh] pcie/read protocol is not supported."),
                HcclResult::HCCL_E_NOT_SUPPORT);

    u32 columnNum = 0;
    CHK_RET(ValidateClosTopology(columnNum));
    u32 slotNum = GetClosSlotNum();
    CHK_PRT_RET(threads.size() < slotNum + COPY_THREAD_NUM,
                HCCL_ERROR("[ALLTOALL_NO_MEMCPY][MeshClos][RunAlltoAllMesh] threads[%zu] < required[%u]. "
                           "slotNum=%u copyThreads=%u myRank=%d",
                           threads.size(), slotNum + COPY_THREAD_NUM, slotNum, COPY_THREAD_NUM, myRank_),
                HcclResult::HCCL_E_INTERNAL);

    std::vector<ThreadHandle> commThreads(threads.begin(), threads.begin() + slotNum);
    for (u32 round = 1; round < columnNum; round++) {
        std::vector<ClosNoMemcpySlot> slotPlans;
        CHK_RET(CalcClosRoundPlan(round, slotPlans));
        CHK_PRT_RET(slotPlans.size() > commThreads.size(),
                    HCCL_ERROR("[ALLTOALL_NO_MEMCPY][MeshClos][RunAlltoAllMesh] slotPlans[%zu] > threads[%zu]. "
                               "round=%u myRank=%d",
                               slotPlans.size(), commThreads.size(), round, myRank_),
                    HcclResult::HCCL_E_INTERNAL);
        for (u32 slotIdx = 0; slotIdx < slotPlans.size(); slotIdx++) {
            CHK_RET(RunAlltoAllSlot(commThreads[slotIdx], channels, slotPlans[slotIdx], round, slotIdx));
        }
    }

    for (u32 i = 0; i < failedRanks_.size(); i++) {
        if (failedRanks_[i]) {
            HCCL_ERROR("[ALLTOALL_NO_MEMCPY][MeshClos][RunAlltoAllMesh] Failed rank[%u] detected. "
                       "templateRank=%u myRank=%d slotNum=%u",
                       i, templateRankSize_, myRank_, slotNum);
            return HcclResult::HCCL_E_INTERNAL;
        }
    }
    return HCCL_SUCCESS;
}

HcclResult InsTempAlltoAllMeshClosV3NoMemcpy::ValidateClosTopology(u32 &columnNum) const
{
    columnNum = 0;
    CHK_PRT_RET(meshSize_ == 0,
                HCCL_ERROR("[ALLTOALL_NO_MEMCPY][MeshClos][ValidateClosTopology] meshSize is 0. myRank=%d",
                           myRank_),
                HcclResult::HCCL_E_INTERNAL);
    CHK_PRT_RET(rankSize_ == 0 || rankSize_ % meshSize_ != 0,
                HCCL_ERROR("[ALLTOALL_NO_MEMCPY][MeshClos][ValidateClosTopology] invalid rank layout. "
                           "rankSize=%u meshSize=%u myRank=%d templateRankSize=%u",
                           rankSize_, meshSize_, myRank_, templateRankSize_),
                HcclResult::HCCL_E_INTERNAL);
    columnNum = rankSize_ / meshSize_;
    CHK_PRT_RET(columnNum == 0,
                HCCL_ERROR("[ALLTOALL_NO_MEMCPY][MeshClos][ValidateClosTopology] columnNum is 0. "
                           "rankSize=%u meshSize=%u myRank=%d templateRankSize=%u",
                           rankSize_, meshSize_, myRank_, templateRankSize_),
                HcclResult::HCCL_E_INTERNAL);
    return HCCL_SUCCESS;
}

HcclResult InsTempAlltoAllMeshClosV3NoMemcpy::CalcClosRoundPlan(
    u32 round, std::vector<ClosNoMemcpySlot> &slotPlans) const
{
    u32 columnNum = 0;
    CHK_RET(ValidateClosTopology(columnNum));
    CHK_PRT_RET(round == 0 || round >= columnNum,
                HCCL_ERROR("[ALLTOALL_NO_MEMCPY][MeshClos][CalcClosRoundPlan] invalid round[%u], columnNum[%u]. "
                           "myRank=%d",
                           round, columnNum, myRank_),
                HcclResult::HCCL_E_INTERNAL);

    slotPlans.clear();
    u32 myRow = myRank_ % meshSize_;
    u32 myCol = myRank_ / meshSize_;
    u32 txCol = (myCol + round) % columnNum;
    u32 rxCol = (myCol + columnNum - round) % columnNum;
    for (u32 row = 0; row < meshSize_; row++) {
        ClosNoMemcpySlot slotPlan;
        slotPlan.txRank = txCol * meshSize_ + row;
        slotPlan.rxRank = rxCol * meshSize_ + row;
        slotPlan.channelIdx = (row + myRow) % meshSize_;
        CHK_PRT_RET(slotPlan.txRank == myRank_ || slotPlan.rxRank == myRank_,
                    HCCL_ERROR("[ALLTOALL_NO_MEMCPY][MeshClos][CalcClosRoundPlan] unexpected self slot. "
                               "round=%u row=%u myRank=%d txRank=%u rxRank=%u",
                               round, row, myRank_, slotPlan.txRank, slotPlan.rxRank),
                    HcclResult::HCCL_E_INTERNAL);
        slotPlans.push_back(slotPlan);
    }
    HCCL_INFO("[ALLTOALL_NO_MEMCPY][MeshClos][CalcClosRoundPlan] myRank=%d myRow=%u myCol=%u round=%u "
              "txCol=%u rxCol=%u slotNum=%zu",
              myRank_, myRow, myCol, round, txCol, rxCol, slotPlans.size());
    return HCCL_SUCCESS;
}

HcclResult InsTempAlltoAllMeshClosV3NoMemcpy::SelectClosChannel(
    const std::map<u32, std::vector<ChannelInfo>> &channels,
    u32 remoteRank, u32 channelIdx, ChannelInfo &channel) const
{
    auto iter = channels.find(remoteRank);
    CHK_PRT_RET(iter == channels.end() || iter->second.empty(),
                HCCL_ERROR("[ALLTOALL_NO_MEMCPY][MeshClos][SelectClosChannel] remoteRank[%u] has no channel. "
                           "myRank=%d channels.size=%zu",
                           remoteRank, myRank_, channels.size()),
                HcclResult::HCCL_E_INTERNAL);
    const std::vector<ChannelInfo> &remoteChannels = iter->second;
    u32 slotNum = GetClosSlotNum();
    CHK_PRT_RET(slotNum == 0 || remoteChannels.size() < slotNum,
                HCCL_ERROR("[ALLTOALL_NO_MEMCPY][MeshClos][SelectClosChannel] channelNum[%zu] < slotNum[%u]. "
                           "myRank=%d remoteRank=%u",
                           remoteChannels.size(), slotNum, myRank_, remoteRank),
                HcclResult::HCCL_E_INTERNAL);
    u32 closOffset = static_cast<u32>(remoteChannels.size()) - slotNum;
    u32 resolvedIdx = closOffset + channelIdx;
    CHK_PRT_RET(resolvedIdx >= remoteChannels.size(),
                HCCL_ERROR("[ALLTOALL_NO_MEMCPY][MeshClos][SelectClosChannel] channelIdx OOB. "
                           "myRank=%d remoteRank=%u channelIdx=%u resolvedIdx=%u channelNum=%zu slotNum=%u",
                           myRank_, remoteRank, channelIdx, resolvedIdx, remoteChannels.size(), slotNum),
                HcclResult::HCCL_E_INTERNAL);
    channel = remoteChannels[resolvedIdx];
    CHK_PRT_RET(channel.remoteCclMem.addr == nullptr,
                HCCL_ERROR("[ALLTOALL_NO_MEMCPY][MeshClos][SelectClosChannel] remoteCclMem.addr is null. "
                           "myRank=%d remoteRank=%u resolvedIdx=%u",
                           myRank_, remoteRank, resolvedIdx),
                HcclResult::HCCL_E_INTERNAL);
    return HCCL_SUCCESS;
}

HcclResult InsTempAlltoAllMeshClosV3NoMemcpy::RunAlltoAllSlot(
    const ThreadHandle &thread,
    const std::map<u32, std::vector<ChannelInfo>> &channels,
    const ClosNoMemcpySlot &slotPlan, u32 round, u32 slotIdx)
{
    ChannelInfo txChannel;
    ChannelInfo rxChannel;
    CHK_RET(SelectClosChannel(channels, slotPlan.txRank, slotPlan.channelIdx, txChannel));
    CHK_RET(SelectClosChannel(channels, slotPlan.rxRank, slotPlan.channelIdx, rxChannel));

    u32 txAlgRank = 0;
    u32 rxAlgRank = 0;
    CHK_RET(GetAlgRank(slotPlan.txRank, subCommRanks_[0], txAlgRank));
    CHK_RET(GetAlgRank(slotPlan.rxRank, subCommRanks_[0], rxAlgRank));
    if (failedRanks_[txAlgRank] || failedRanks_[rxAlgRank]) {
        HCCL_WARNING("[ALLTOALL_NO_MEMCPY][MeshClos][RunAlltoAllSlot] skip failed rank. "
                     "myRank=%d round=%u slotIdx=%u txRank=%u rxRank=%u txFailed=%u rxFailed=%u",
                     myRank_, round, slotIdx, slotPlan.txRank, slotPlan.rxRank,
                     failedRanks_[txAlgRank], failedRanks_[rxAlgRank]);
        return HCCL_SUCCESS;
    }

    const u32 dataTypeSize = DATATYPE_SIZE_TABLE[dataType_];
    u64 totalSliceSize = tempAlgParams_.sliceSize;
    u64 actualChunkSize = (totalSliceSize + rankSize_ - 1) / rankSize_;
    u64 chunkCount = actualChunkSize / dataTypeSize;

    std::vector<DataSlice> txSrcSlicesAll;
    std::vector<DataSlice> txDstSlicesAll;
    std::vector<DataSlice> rxSrcSlicesAll;
    std::vector<DataSlice> rxDstSlicesAll;

    void *txSrcPtr = tempAlgParams_.buffInfo.inputPtr;
    u64 txSrcOffset = tempAlgParams_.buffInfo.inBuffBaseOff + slotPlan.txRank * actualChunkSize;
    txSrcSlicesAll.emplace_back(txSrcPtr, txSrcOffset, actualChunkSize, chunkCount);

    CHK_PRT_RET(!enableRemoteMemAccess_ || txChannel.remoteOutputGraphMode.addr == nullptr,
                HCCL_ERROR("[ALLTOALL_NO_MEMCPY][MeshClos][RunAlltoAllSlot] remote output is unavailable. "
                           "myRank=%d txRank=%u round=%u slotIdx=%u enableRemoteMemAccess=%d",
                           myRank_, slotPlan.txRank, round, slotIdx, enableRemoteMemAccess_),
                HcclResult::HCCL_E_INTERNAL);

    void *txDstPtr = txChannel.remoteOutputGraphMode.addr;
    u64 txDstOffset = tempAlgParams_.buffInfo.outBuffBaseOff + myRank_ * actualChunkSize;
    CHK_PRT_RET(txSrcOffset + actualChunkSize > tempAlgParams_.buffInfo.inputSize ||
                    txDstOffset + actualChunkSize > txChannel.remoteOutputGraphMode.size,
                HCCL_ERROR("[ALLTOALL_NO_MEMCPY][MeshClos][RunAlltoAllSlot] tx slice out of registered range. "
                           "myRank=%d txRank=%u txSrcOff=%llu txDstOff=%llu chunk=%llu "
                           "inputSize=%llu remoteOutputSize=%llu",
                           myRank_, slotPlan.txRank, txSrcOffset, txDstOffset, actualChunkSize,
                           tempAlgParams_.buffInfo.inputSize, txChannel.remoteOutputGraphMode.size),
                HcclResult::HCCL_E_INTERNAL);
    txDstSlicesAll.emplace_back(txDstPtr, txDstOffset, actualChunkSize, chunkCount);

    void *rxSrcPtr = rxChannel.remoteOutputGraphMode.addr;
    u64 rxSrcOffset = tempAlgParams_.buffInfo.outBuffBaseOff + myRank_ * actualChunkSize;
    rxSrcSlicesAll.emplace_back(rxSrcPtr, rxSrcOffset, actualChunkSize, chunkCount);

    void *rxDstPtr = tempAlgParams_.buffInfo.outputPtr;
    u64 rxOutOffset = tempAlgParams_.buffInfo.outBuffBaseOff + slotPlan.rxRank * actualChunkSize;
    CHK_PRT_RET(rxOutOffset + actualChunkSize > tempAlgParams_.buffInfo.outputSize,
                HCCL_ERROR("[ALLTOALL_NO_MEMCPY][MeshClos][RunAlltoAllSlot] rx slice out of registered range. "
                           "myRank=%d rxRank=%u rxOutOff=%llu chunk=%llu outputSize=%llu",
                           myRank_, slotPlan.rxRank, rxOutOffset, actualChunkSize,
                           tempAlgParams_.buffInfo.outputSize),
                HcclResult::HCCL_E_INTERNAL);
    rxDstSlicesAll.emplace_back(rxDstPtr, rxOutOffset, actualChunkSize, chunkCount);

    HCCL_WARNING("[ALLTOALL_NO_MEMCPY][MeshClos][RunAlltoAllSlot] myRank=%d round=%u slotIdx=%u "
                 "txRank=%u rxRank=%u channelIdx=%u txSrcOff=%llu txDstOff=%llu rxSrcOff=%llu rxDstOff=%llu "
                 "actualSz=%llu",
                 myRank_, round, slotIdx, slotPlan.txRank, slotPlan.rxRank, slotPlan.channelIdx,
                 txSrcOffset, txDstOffset, rxSrcOffset, rxOutOffset, actualChunkSize);

    TxRxSlicesList sendRecvSlicesList({txSrcSlicesAll, txDstSlicesAll},
                                      {rxSrcSlicesAll, rxDstSlicesAll});
    TxRxChannels sendRecvChannels(txChannel, rxChannel);
    SendRecvInfo sendRecvInfo(sendRecvChannels, sendRecvSlicesList, dataType_);

    HcclResult dmaResult = SendRecvWrite(sendRecvInfo, thread);
    if (dmaResult == HcclResult::HCCL_E_INTERNAL) {
        failedRanks_[txAlgRank] = 1;
        failedRanks_[rxAlgRank] = 1;
        HCCL_WARNING("[ALLTOALL_NO_MEMCPY][MeshClos][RunAlltoAllSlot] timeout. "
                     "myRank=%d round=%u slotIdx=%u txRank=%u rxRank=%u",
                     myRank_, round, slotIdx, slotPlan.txRank, slotPlan.rxRank);
        return HCCL_SUCCESS;
    }
    CHK_PRT_RET(dmaResult != HCCL_SUCCESS,
                HCCL_ERROR("[ALLTOALL_NO_MEMCPY][MeshClos][RunAlltoAllSlot] send/recv failed. "
                           "myRank=%d round=%u slotIdx=%u txRank=%u rxRank=%u err=0x%x actualChunkSize=%llu",
                           myRank_, round, slotIdx, slotPlan.txRank, slotPlan.rxRank, dmaResult, actualChunkSize),
                dmaResult);
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
