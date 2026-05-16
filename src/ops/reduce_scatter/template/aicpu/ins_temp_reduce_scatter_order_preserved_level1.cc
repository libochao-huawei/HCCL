/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You can not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "ins_temp_reduce_scatter_order_preserved_level1.h"

namespace ops_hccl {

InsTempReduceScatterOrderPreservedLevel1::InsTempReduceScatterOrderPreservedLevel1(const OpParam &param,
    const u32 rankId, const std::vector<std::vector<u32>> &subCommRanks)
    : InsAlgTemplateBase(param, rankId, subCommRanks)
{
    deterministicStrict_ = (GetLocalDeterministicConfig() == static_cast<u8>(DeterministicEnableLevel::DETERMINISTIC_STRICT));
    all2allOffset_ = 0;
}

InsTempReduceScatterOrderPreservedLevel1::~InsTempReduceScatterOrderPreservedLevel1()
{}

HcclResult InsTempReduceScatterOrderPreservedLevel1::CalcRes(
    HcclComm comm, const OpParam &param, const TopoInfoWithNetLayerDetails *topoInfo,
    AlgResourceRequest &resourceRequest)
{
    u32 threadNum = templateRankSize_ > 1 ? templateRankSize_ : 1;
    threadNum = std::min(threadNum, REDUCE_SCATTER_MAX_STREAM_NUM_ORDER_PRESERVED);
    resourceRequest.slaveThreadNum = threadNum - 1;

    for (u32 index = 0; index < threadNum - 1; index++) {
        resourceRequest.notifyNumPerThread.push_back(2);
    }
    resourceRequest.notifyNumOnMainThread = threadNum - 1;

    std::vector<HcclChannelDesc> level0Channels;
    CHK_RET(CalcChannelRequestMesh1D(comm, param, topoInfo, subCommRanks_, level0Channels));
    resourceRequest.channels.push_back(level0Channels);

    HCCL_INFO("[InsTempReduceScatterOrderPreservedLevel1][CalcRes] myRank[%u], threadNum[%u], "
        "notifyNumOnMainThread[%u], slaveThreadNum[%u]",
        myRank_, threadNum, resourceRequest.notifyNumOnMainThread, resourceRequest.slaveThreadNum);
    return HCCL_SUCCESS;
}

u64 InsTempReduceScatterOrderPreservedLevel1::CalcScratchMultiple(BufferType inBuffType, BufferType outBuffType)
{
    u64 scratchMultiple = templateRankSize_;
    return scratchMultiple;
}

HcclResult InsTempReduceScatterOrderPreservedLevel1::KernelRun(
    const OpParam& param, const TemplateDataParams& tempAlgParams, TemplateResource& templateResource)
{
    if (tempAlgParams.sliceSize == 0 && tempAlgParams.tailSize == 0) {
        HCCL_DEBUG("[InsTempReduceScatterOrderPreservedLevel1] myRank[%u] sliceSize and tailSize are 0, skip.", myRank_);
        return HCCL_SUCCESS;
    }

    threadNum_ = templateResource.threads.size();
    dataType_ = param.DataDes.dataType;
    reduceOp_ = param.reduceType;
    processSize_ = tempAlgParams.sliceSize;
    count_ = tempAlgParams.sliceSize / DATATYPE_SIZE_TABLE[dataType_];

    HCCL_INFO("[InsTempReduceScatterOrderPreservedLevel1][KernelRun] Start, threadNum[%u], count[%llu], "
        "dataType[%u], deterministicStrict[%d]", threadNum_, count_, dataType_, deterministicStrict_);

    if (threadNum_ > 1) {
        std::vector<ThreadHandle> subThreads(templateResource.threads.begin() + 1, templateResource.threads.end());
        GetNotifyIdxMainToSub(notifyIdxMainToSub_);
        CHK_RET(PreSyncInterThreads(templateResource.threads[0], subThreads, notifyIdxMainToSub_));
    }

    CHK_RET(PreLocalCopy(tempAlgParams, templateResource.threads));

    CHK_RET(RunAllToAll(templateResource.channels, templateResource.threads, tempAlgParams));

    CHK_RET(RunLocalReduce(templateResource.threads, tempAlgParams));

    CHK_RET(PostCopy(tempAlgParams, templateResource.threads));

    if (threadNum_ > 1) {
        std::vector<ThreadHandle> subThreads(templateResource.threads.begin() + 1, templateResource.threads.end());
        GetNotifyIdxSubToMain(notifyIdxSubToMain_);
        CHK_RET(PostSyncInterThreads(templateResource.threads[0], subThreads, notifyIdxSubToMain_));
    }

    HCCL_INFO("[InsTempReduceScatterOrderPreservedLevel1][KernelRun] End");
    return HCCL_SUCCESS;
}

HcclResult InsTempReduceScatterOrderPreservedLevel1::GetRes(AlgResourceRequest &resourceRequest) const
{
    u32 threadNum = GetThreadNum();
    resourceRequest.slaveThreadNum = threadNum - 1;
    for (u32 index = 0; index < threadNum - 1; index++) {
        resourceRequest.notifyNumPerThread.push_back(2);
    }
    resourceRequest.notifyNumOnMainThread = threadNum - 1;
    return HCCL_SUCCESS;
}

u64 InsTempReduceScatterOrderPreservedLevel1::GetThreadNum() const
{
    return templateRankSize_ > 1 ? templateRankSize_ : 1;
}

void InsTempReduceScatterOrderPreservedLevel1::GetNotifyIdxMainToSub(std::vector<u32> &notifyIdxMainToSub)
{
    notifyIdxMainToSub.clear();
    u32 threadNum = GetThreadNum();
    u32 slaveThreadNum = threadNum - 1;
    for (u32 slaveThreadIdx = 0; slaveThreadIdx < slaveThreadNum; slaveThreadIdx++) {
        notifyIdxMainToSub.push_back(0);
    }
}

void InsTempReduceScatterOrderPreservedLevel1::GetNotifyIdxSubToMain(std::vector<u32> &notifyIdxSubToMain)
{
    notifyIdxSubToMain.clear();
    u32 threadNum = GetThreadNum();
    u32 notifyNum = threadNum - 1;
    for (u32 notifyIdx = 0; notifyIdx < notifyNum; notifyIdx++) {
        notifyIdxSubToMain.push_back(notifyIdx);
    }
}

u32 InsTempReduceScatterOrderPreservedLevel1::CalcOutputIndex(const u32 round, const u32 localRank)
{
    return (all2allOffset_ + round + localRank) % templateRankSize_;
}

bool InsTempReduceScatterOrderPreservedLevel1::IsLastBlockData(const u32 outputIndex)
{
    return outputIndex == templateRankSize_ - 1;
}

bool InsTempReduceScatterOrderPreservedLevel1::IsLastRank(const u32 rankId)
{
    return rankId == templateRankSize_ - 1;
}

HcclResult InsTempReduceScatterOrderPreservedLevel1::PreLocalCopy(
    const TemplateDataParams &tempAlgParams, const std::vector<ThreadHandle> &threads)
{
    const MemBlockInfo &memBlockInfo = memBlockInfo_;
    u32 myAlgRank = 0;
    CHK_RET(GetAlgRank(myRank_, subCommRanks_[0], myAlgRank));

    u64 sliceSize = memBlockInfo.size[myAlgRank];
    if (sliceSize == 0) {
        HCCL_DEBUG("[PreLocalCopy] myAlgRank[%u] sliceSize is 0, skip.", myAlgRank);
        return HCCL_SUCCESS;
    }

    u64 srcOffset = memBlockInfo.userInputOffsets[myAlgRank];
    u32 outputIndex = CalcOutputIndex(myAlgRank, myAlgRank);
    u64 dstOffset = memBlockInfo.outputOffsets[outputIndex];

    HCCL_INFO("[PreLocalCopy] myAlgRank[%u], sliceSize[%llu], srcOffset[%llu], dstOffset[%llu], outputIndex[%u]",
        myAlgRank, sliceSize, srcOffset, dstOffset, outputIndex);

    DataSlice srcSlice(tempAlgParams.buffInfo.inputPtr, srcOffset, sliceSize);
    DataSlice dstSlice(tempAlgParams.buffInfo.hcclBuff.addr, dstOffset, sliceSize);
    CHK_RET(LocalCopy(threads[0], srcSlice, dstSlice));

    return HCCL_SUCCESS;
}

HcclResult InsTempReduceScatterOrderPreservedLevel1::RunAllToAll(
    const std::map<u32, std::vector<ChannelInfo>> &channels,
    const std::vector<ThreadHandle> &threads, const TemplateDataParams &tempAlgParams)
{
    HCCL_INFO("[RunAllToAll] Start");

    const MemBlockInfo &memBlockInfo = memBlockInfo_;
    u32 myAlgRank = 0;
    CHK_RET(GetAlgRank(myRank_, subCommRanks_[0], myAlgRank));

    u32 queIdx = 1;
    for (u32 rankIdx = 1; rankIdx < templateRankSize_; rankIdx++) {
        u32 nextRank = (myAlgRank + rankIdx) % templateRankSize_;
        u32 remoteRank = subCommRanks_[0][nextRank];

        auto channelIter = channels.find(remoteRank);
        CHK_PRT_RET(channelIter == channels.end(),
            HCCL_ERROR("[RunAllToAll] channel not found for nextRank[%u], remoteRank[%u]", nextRank, remoteRank), HCCL_E_INTERNAL);

        const std::vector<ChannelInfo> &curChannels = channelIter->second;
        CHK_PRT_RET(curChannels.empty(),
            HCCL_ERROR("[RunAllToAll] curChannels empty for nextRank[%u]", nextRank), HCCL_E_INTERNAL);

        u64 sliceSize = memBlockInfo.size[nextRank];
        if (sliceSize == 0) {
            queIdx++;
            continue;
        }

        for (u32 channelIdx = 0; channelIdx < curChannels.size(); channelIdx++) {
            const ChannelInfo &linkSend = curChannels[channelIdx];
            const ChannelInfo &linkRecv = curChannels[channelIdx];
            ThreadHandle thread = threads[queIdx];

            void* remoteCclBuffAddr = linkSend.remoteCclMem.addr;

            u32 txOutputIndex = CalcOutputIndex(nextRank, myAlgRank);
            u64 txSrcOffset = memBlockInfo.userInputOffsets[nextRank];
            u64 txDstOffset = memBlockInfo.outputOffsets[txOutputIndex];

            u32 rxOutputIndex = CalcOutputIndex(nextRank, myAlgRank);
            u64 rxSrcOffset = memBlockInfo.outputOffsets[rxOutputIndex];
            u64 rxDstOffset = memBlockInfo.outputOffsets[rxOutputIndex];

            DataSlice txSrcSlice(tempAlgParams.buffInfo.inputPtr, txSrcOffset, sliceSize);
            DataSlice txDstSlice(remoteCclBuffAddr, txDstOffset, sliceSize);
            DataSlice rxSrcSlice(remoteCclBuffAddr, rxSrcOffset, sliceSize);
            DataSlice rxDstSlice(tempAlgParams.buffInfo.hcclBuff.addr, rxDstOffset, sliceSize);

            std::vector<DataSlice> txSrcSlices = {txSrcSlice};
            std::vector<DataSlice> txDstSlices = {txDstSlice};
            std::vector<DataSlice> rxSrcSlices = {rxSrcSlice};
            std::vector<DataSlice> rxDstSlices = {rxDstSlice};

            SendRecvInfo sendRecvInfo{
                TxRxChannels{linkSend, linkRecv},
                TxRxSlicesList{SlicesList{txSrcSlices, txDstSlices}, SlicesList{rxSrcSlices, rxDstSlices}}
            };
            CHK_RET(SendRecvWrite(sendRecvInfo, thread));
        }
        queIdx++;
    }

    HCCL_INFO("[RunAllToAll] End");
    return HCCL_SUCCESS;
}

HcclResult InsTempReduceScatterOrderPreservedLevel1::RunLocalReduce(
    const std::vector<ThreadHandle> &threads, const TemplateDataParams &tempAlgParams)
{
    HCCL_INFO("[RunLocalReduce] Start, deterministicStrict[%d], templateRankSize[%u]",
        deterministicStrict_, templateRankSize_);

    if (templateRankSize_ <= 1) {
        HCCL_INFO("[RunLocalReduce] Skip for single rank");
        return HCCL_SUCCESS;
    }

    const MemBlockInfo &memBlockInfo = memBlockInfo_;
    u32 myAlgRank = 0;
    CHK_RET(GetAlgRank(myRank_, subCommRanks_[0], myAlgRank));

    u64 sliceSize = memBlockInfo.size[myAlgRank];
    u64 count = sliceSize / DATATYPE_SIZE_TABLE[dataType_];

    u32 myOutputIndex = CalcOutputIndex(myAlgRank, myAlgRank);
    u64 dstOffset = memBlockInfo.outputOffsets[myOutputIndex];

    HCCL_DEBUG("[RunLocalReduce] myAlgRank[%u], myOutputIndex[%u], dstOffset[%llu], sliceSize[%llu]",
        myAlgRank, myOutputIndex, dstOffset, sliceSize);

    for (u32 round = 1; round < templateRankSize_; round++) {
        u32 peerRank = (myAlgRank + round) % templateRankSize_;

        u64 peerSliceSize = memBlockInfo.size[peerRank];
        if (peerSliceSize == 0) {
            continue;
        }

        u32 peerOutputIndex = CalcOutputIndex(peerRank, myAlgRank);
        u64 peerOffset = memBlockInfo.outputOffsets[peerOutputIndex];

        DataSlice srcSlice(tempAlgParams.buffInfo.hcclBuff.addr, peerOffset, peerSliceSize);
        DataSlice dstSlice(tempAlgParams.buffInfo.hcclBuff.addr, dstOffset, sliceSize);

        HCCL_INFO("[RunLocalReduce] round[%u], peerRank[%u], peerOutputIndex[%u], peerOffset[%llu], dstOffset[%llu]",
            round, peerRank, peerOutputIndex, peerOffset, dstOffset);

        CHK_RET(LocalReduce(threads[0], srcSlice, dstSlice, dataType_, reduceOp_));
    }

    HCCL_INFO("[RunLocalReduce] End");
    return HCCL_SUCCESS;
}

HcclResult InsTempReduceScatterOrderPreservedLevel1::PostCopy(
    const TemplateDataParams &tempAlgParams, const std::vector<ThreadHandle> &threads)
{
    const MemBlockInfo &memBlockInfo = memBlockInfo_;
    u32 myAlgRank = 0;
    CHK_RET(GetAlgRank(myRank_, subCommRanks_[0], myAlgRank));

    u64 sliceSize = memBlockInfo.size[myAlgRank];
    if (sliceSize == 0) {
        HCCL_DEBUG("[PostCopy] myAlgRank[%u] sliceSize is 0, skip.", myAlgRank);
        return HCCL_SUCCESS;
    }

    u32 outputIndex = CalcOutputIndex(myAlgRank, myAlgRank);
    u64 srcOffset = memBlockInfo.outputOffsets[outputIndex];
    u64 dstOffset = 0;

    HCCL_INFO("[PostCopy] myAlgRank[%u], sliceSize[%llu], srcOffset[%llu], dstOffset[%llu]",
        myAlgRank, sliceSize, srcOffset, dstOffset);

    DataSlice srcSlice(tempAlgParams.buffInfo.hcclBuff.addr, srcOffset, sliceSize);
    DataSlice dstSlice(tempAlgParams.buffInfo.outputPtr, dstOffset, sliceSize);
    CHK_RET(LocalCopy(threads[0], srcSlice, dstSlice));

    return HCCL_SUCCESS;
}

REGISTER_TEMPLATE_V2("InsTempReduceScatterOrderPreservedLevel1", InsTempReduceScatterOrderPreservedLevel1);
}