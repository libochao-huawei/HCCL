/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "ins_temp_reduce_scatter_mesh_1D_two_shot.h"

#include <algorithm>
#include <numeric>

namespace ops_hccl {
InsTempReduceScatterMesh1DTwoShot::InsTempReduceScatterMesh1DTwoShot(const OpParam &param, const u32 rankId,
    const std::vector<std::vector<u32>> &subCommRanks)
    : InsAlgTemplateBase(param, rankId, subCommRanks)
{}

InsTempReduceScatterMesh1DTwoShot::~InsTempReduceScatterMesh1DTwoShot()
{}

HcclResult InsTempReduceScatterMesh1DTwoShot::CalcRes(HcclComm comm, const OpParam &param,
    const TopoInfoWithNetLayerDetails *topoInfo, AlgResourceRequest &resourceRequest)
{
    CHK_PRT_RET(topoInfo == nullptr,
        HCCL_ERROR("[InsTempReduceScatterMesh1DTwoShot][CalcRes] topoInfo is nullptr"), HCCL_E_PARA);
    std::vector<HcclChannelDesc> level0Channels;
    CHK_RET(CalcChannelRequestMesh1D(comm, param, topoInfo, subCommRanks_, level0Channels));

    channelsPerRank_ = level0Channels.empty() ? 1 : CalcChannelsPerRank(level0Channels);
    u32 threadNum = GetThreadNum();
    resourceRequest.slaveThreadNum = threadNum - 1;
    for (u32 index = 0; index < threadNum - 1; index++) {
        resourceRequest.notifyNumPerThread.push_back(1);
    }
    resourceRequest.notifyNumOnMainThread = threadNum - 1;
    resourceRequest.channels.push_back(level0Channels);
    return HCCL_SUCCESS;
}

HcclResult InsTempReduceScatterMesh1DTwoShot::GetRes(AlgResourceRequest &resourceRequest) const
{
    u32 threadNum = GetThreadNum();
    resourceRequest.slaveThreadNum = threadNum - 1;
    for (u32 index = 0; index < threadNum - 1; index++) {
        resourceRequest.notifyNumPerThread.push_back(1);
    }
    resourceRequest.notifyNumOnMainThread = threadNum - 1;
    return HCCL_SUCCESS;
}

u64 InsTempReduceScatterMesh1DTwoShot::CalcScratchMultiple(BufferType inBuffType, BufferType outBuffType)
{
    (void)inBuffType;
    (void)outBuffType;
    return templateRankSize_;
}

u64 InsTempReduceScatterMesh1DTwoShot::GetThreadNum() const
{
    return templateRankSize_ > 1 ? (templateRankSize_ - 1) * channelsPerRank_ + 1 : 1;
}

HcclResult InsTempReduceScatterMesh1DTwoShot::KernelRun(const OpParam &param,
    const TemplateDataParams &tempAlgParams, TemplateResource &templateResource)
{
    if (tempAlgParams.sliceSize == 0 && tempAlgParams.tailSize == 0) {
        HCCL_DEBUG("[InsTempReduceScatterMesh1DTwoShot] myRank[%u] sliceSize and tailSize are 0, skip.", myRank_);
        return HCCL_SUCCESS;
    }
    CHK_PRT_RET(templateResource.threads.empty(),
        HCCL_ERROR("[InsTempReduceScatterMesh1DTwoShot][KernelRun] threads is empty."), HCCL_E_INTERNAL);
    CHK_PRT_RET(subCommRanks_.empty() || subCommRanks_[0].empty(),
        HCCL_ERROR("[InsTempReduceScatterMesh1DTwoShot][KernelRun] subCommRanks is empty."), HCCL_E_INTERNAL);

    threadNum_ = GetThreadNum();
    dataType_ = param.DataDes.dataType;
    reduceOp_ = param.reduceType;
    dataTypeSize_ = DATATYPE_SIZE_TABLE[dataType_];
    CHK_PRT_RET(dataTypeSize_ == 0,
        HCCL_ERROR("[InsTempReduceScatterMesh1DTwoShot][KernelRun] dataTypeSize is 0."), HCCL_E_INTERNAL);

    HCCL_INFO("[InsTempReduceScatterMesh1DTwoShot] Run Start");
    if (!CanRunTwoShot(tempAlgParams)) {
        if (threadNum_ > 1) {
            std::vector<ThreadHandle> subThreads(templateResource.threads.begin() + 1, templateResource.threads.end());
            GetNotifyIdxMainToSub(notifyIdxMainToSub_);
            CHK_RET(PreSyncInterThreads(templateResource.threads[0], subThreads, notifyIdxMainToSub_));
        }
        CHK_RET(RunReduceScatterOneShot(tempAlgParams, templateResource.channels, templateResource.threads));
        if (threadNum_ > 1) {
            std::vector<ThreadHandle> subThreads(templateResource.threads.begin() + 1, templateResource.threads.end());
            GetNotifyIdxSubToMain(notifyIdxSubToMain_);
            CHK_RET(PostSyncInterThreads(templateResource.threads[0], subThreads, notifyIdxSubToMain_));
        }
        CHK_RET(PostCopyOneShot(param, tempAlgParams, templateResource.threads));
        return HCCL_SUCCESS;
    }

    CHK_RET(RunReduceScatterTwoShot(tempAlgParams, templateResource.channels, templateResource.threads));
    CHK_RET(PostCopyTwoShot(param, tempAlgParams, templateResource.threads));
    HCCL_INFO("[InsTempReduceScatterMesh1DTwoShot] Run End");
    return HCCL_SUCCESS;
}

HcclResult InsTempReduceScatterMesh1DTwoShot::RunReduceScatterOneShot(const TemplateDataParams &tempAlgParams,
    const std::map<u32, std::vector<ChannelInfo>> &channels, const std::vector<ThreadHandle> &threads)
{
    u32 myAlgRank = 0;
    CHK_RET(GetAlgRank(myRank_, subCommRanks_[0], myAlgRank));
    u32 threadIdx = 1;
    for (u32 rankIdx = 0; rankIdx < templateRankSize_ - 1; rankIdx++) {
        u32 nextRank = (myAlgRank + 1 + rankIdx) % templateRankSize_;
        u32 remoteRank = subCommRanks_[0][nextRank];
        const std::vector<ChannelInfo> &curChannels = channels.at(remoteRank);
        u64 sendSlice = GetSliceSize(nextRank, tempAlgParams);
        u64 recvSlice = GetSliceSize(myAlgRank, tempAlgParams);
        std::vector<u64> sendElemCount;
        std::vector<u64> sendSize;
        std::vector<u64> sendOffset;
        std::vector<u64> recvElemCount;
        std::vector<u64> recvSize;
        std::vector<u64> recvOffset;
        CHK_RET(CalcDataSplitByPortGroup(sendSlice / dataTypeSize_, dataTypeSize_, curChannels,
            sendElemCount, sendSize, sendOffset));
        CHK_RET(CalcDataSplitByPortGroup(recvSlice / dataTypeSize_, dataTypeSize_, curChannels,
            recvElemCount, recvSize, recvOffset));
        for (u32 channelIdx = 0; channelIdx < curChannels.size(); channelIdx++) {
            CHK_PRT_RET(threadIdx >= threads.size(),
                HCCL_ERROR("[InsTempReduceScatterMesh1DTwoShot][RunReduceScatterOneShot] threadIdx[%u] >= size[%u].",
                    threadIdx, threads.size()),
                HCCL_E_INTERNAL);
            const ChannelInfo &channel = curChannels[channelIdx];
            void *remoteCclBuffAddr = channel.remoteCclMem.addr;
            std::vector<DataSlice> txSrcSlices;
            std::vector<DataSlice> txDstSlices;
            std::vector<DataSlice> rxSrcSlices;
            std::vector<DataSlice> rxDstSlices;
            for (u32 repeatIdx = 0; repeatIdx < tempAlgParams.repeatNum; repeatIdx++) {
                DataSlice txSrcSlice(tempAlgParams.buffInfo.inputPtr,
                    tempAlgParams.buffInfo.inBuffBaseOff + repeatIdx * tempAlgParams.inputRepeatStride +
                        nextRank * tempAlgParams.inputSliceStride + sendOffset[channelIdx],
                    sendSize[channelIdx], sendElemCount[channelIdx]);
                DataSlice txDstSlice(remoteCclBuffAddr,
                    GetScratchSlotOffset(tempAlgParams, repeatIdx, myAlgRank) + sendOffset[channelIdx],
                    sendSize[channelIdx], sendElemCount[channelIdx]);
                DataSlice rxSrcSlice(remoteCclBuffAddr,
                    tempAlgParams.buffInfo.inBuffBaseOff + repeatIdx * tempAlgParams.inputRepeatStride +
                        myAlgRank * tempAlgParams.inputSliceStride + recvOffset[channelIdx],
                    recvSize[channelIdx], recvElemCount[channelIdx]);
                DataSlice rxDstSlice(tempAlgParams.buffInfo.hcclBuff.addr,
                    GetScratchSlotOffset(tempAlgParams, repeatIdx, nextRank) + recvOffset[channelIdx],
                    recvSize[channelIdx], recvElemCount[channelIdx]);
                txSrcSlices.push_back(txSrcSlice);
                txDstSlices.push_back(txDstSlice);
                rxSrcSlices.push_back(rxSrcSlice);
                rxDstSlices.push_back(rxDstSlice);
            }
            SendRecvInfo sendRecvInfo{{channel, channel}, {{txSrcSlices, txDstSlices}, {rxSrcSlices, rxDstSlices}}, dataType_};
            CHK_PRT_RET(SendRecvBatchWrite(sendRecvInfo, threads[threadIdx]),
                HCCL_ERROR("[InsTempReduceScatterMesh1DTwoShot] RunReduceScatterOneShot SendRecv failed"),
                HCCL_E_INTERNAL);
            threadIdx++;
        }
    }
    return HCCL_SUCCESS;
}

HcclResult InsTempReduceScatterMesh1DTwoShot::PostCopyOneShot(const OpParam &param,
    const TemplateDataParams &tempAlgParams, const std::vector<ThreadHandle> &threads)
{
    u32 rankIdx = 0;
    CHK_RET(GetAlgRank(myRank_, subCommRanks_[0], rankIdx));
    u64 sliceSize = GetSliceSize(rankIdx, tempAlgParams);
    u64 sliceCount = GetSliceCount(sliceSize);
    for (u32 repeatIdx = 0; repeatIdx < tempAlgParams.repeatNum; repeatIdx++) {
        DataSlice myRankSlice(tempAlgParams.buffInfo.inputPtr,
            tempAlgParams.buffInfo.inBuffBaseOff + repeatIdx * tempAlgParams.inputRepeatStride +
                rankIdx * tempAlgParams.inputSliceStride,
            sliceSize, sliceCount);
        DataSlice outputSlice(tempAlgParams.buffInfo.outputPtr,
            tempAlgParams.buffInfo.outBuffBaseOff + repeatIdx * tempAlgParams.outputRepeatStride,
            sliceSize, sliceCount);
        if (!(tempAlgParams.buffInfo.inBuffType == tempAlgParams.buffInfo.outBuffType &&
                myRankSlice.GetOffset() == outputSlice.GetOffset())) {
            CHK_RET(LocalCopy(threads[0], myRankSlice, outputSlice));
        }
        if (dataType_ == HCCL_DATA_TYPE_INT64 || dataType_ == HCCL_DATA_TYPE_UINT64 ||
            dataType_ == HCCL_DATA_TYPE_FP64 || param.reduceType == HcclReduceOp::HCCL_REDUCE_PROD) {
            CHK_RET(static_cast<HcclResult>(HcommBatchModeEnd(param.algTag)));
            CHK_RET(static_cast<HcclResult>(HcommBatchModeStart(param.algTag)));
            for (const auto &thread : threads) {
                CHK_RET(static_cast<HcclResult>(HcommThreadJoin(thread, CUSTOM_TIMEOUT)));
            }
        }
        for (u32 tmpRank = 0; tmpRank < templateRankSize_; tmpRank++) {
            if (tmpRank == rankIdx) {
                continue;
            }
            DataSlice srcSlice(tempAlgParams.buffInfo.hcclBuff.addr,
                GetScratchSlotOffset(tempAlgParams, repeatIdx, tmpRank), sliceSize, sliceCount);
            CHK_RET(LocalReduce(threads[0], srcSlice, outputSlice, dataType_, reduceOp_));
        }
    }
    return HCCL_SUCCESS;
}

HcclResult InsTempReduceScatterMesh1DTwoShot::RunReduceScatterTwoShot(const TemplateDataParams &tempAlgParams,
    const std::map<u32, std::vector<ChannelInfo>> &channels, const std::vector<ThreadHandle> &threads)
{
    if (threadNum_ > 1) {
        std::vector<ThreadHandle> subThreads(threads.begin() + 1, threads.end());
        GetNotifyIdxMainToSub(notifyIdxMainToSub_);
        CHK_RET(PreSyncInterThreads(threads[0], subThreads, notifyIdxMainToSub_));
    }
    CHK_RET(RunReduceScatterShot(tempAlgParams, channels, threads, false));
    if (threadNum_ > 1) {
        std::vector<ThreadHandle> subThreads(threads.begin() + 1, threads.end());
        GetNotifyIdxSubToMain(notifyIdxSubToMain_);
        CHK_RET(PostSyncInterThreads(threads[0], subThreads, notifyIdxSubToMain_));
        GetNotifyIdxMainToSub(notifyIdxMainToSub_);
        CHK_RET(PreSyncInterThreads(threads[0], subThreads, notifyIdxMainToSub_));
    }
    CHK_RET(RunReduceScatterShot(tempAlgParams, channels, threads, true));
    if (threadNum_ > 1) {
        std::vector<ThreadHandle> subThreads(threads.begin() + 1, threads.end());
        GetNotifyIdxSubToMain(notifyIdxSubToMain_);
        CHK_RET(PostSyncInterThreads(threads[0], subThreads, notifyIdxSubToMain_));
    }
    return HCCL_SUCCESS;
}

HcclResult InsTempReduceScatterMesh1DTwoShot::RunReduceScatterShot(const TemplateDataParams &tempAlgParams,
    const std::map<u32, std::vector<ChannelInfo>> &channels, const std::vector<ThreadHandle> &threads, bool secondShot)
{
    u32 myAlgRank = 0;
    CHK_RET(GetAlgRank(myRank_, subCommRanks_[0], myAlgRank));
    u32 threadIdx = 1;
    for (u32 rankIdx = 0; rankIdx < templateRankSize_ - 1; rankIdx++) {
        u32 nextRank = (myAlgRank + 1 + rankIdx) % templateRankSize_;
        bool currIsUnpaired = IsUnpairedRemote(myAlgRank, nextRank);
        u32 remoteRank = subCommRanks_[0][nextRank];
        const std::vector<ChannelInfo> &curChannels = channels.at(remoteRank);
        u64 sendFullSlice = GetSliceSize(nextRank, tempAlgParams);
        u64 recvFullSlice = GetSliceSize(myAlgRank, tempAlgParams);
        u32 txGroupSlot = GetGroupSlotIdx(nextRank, myAlgRank);
        u32 rxGroupSlot = GetGroupSlotIdx(myAlgRank, nextRank);
        u64 sendOffset = GetShotOffset(sendFullSlice, nextRank, myAlgRank, secondShot);
        u64 recvOffset = GetShotOffset(recvFullSlice, myAlgRank, nextRank, secondShot);
        u64 sendDstOffset = (secondShot && !currIsUnpaired) ?
            GetPutReduceDstOffset(sendFullSlice, nextRank, myAlgRank) : sendOffset;
        u64 sendSlice = GetShotSize(sendFullSlice, nextRank, myAlgRank, secondShot);
        u64 recvSlice = GetShotSize(recvFullSlice, myAlgRank, nextRank, secondShot);
        std::vector<u64> sendElemCount;
        std::vector<u64> sendSize;
        std::vector<u64> sendSplitOffset;
        std::vector<u64> recvElemCount;
        std::vector<u64> recvSize;
        std::vector<u64> recvSplitOffset;
        CHK_RET(CalcDataSplitByPortGroup(sendSlice / dataTypeSize_, dataTypeSize_, curChannels,
            sendElemCount, sendSize, sendSplitOffset));
        CHK_RET(CalcDataSplitByPortGroup(recvSlice / dataTypeSize_, dataTypeSize_, curChannels,
            recvElemCount, recvSize, recvSplitOffset));
        for (u32 channelIdx = 0; channelIdx < curChannels.size(); channelIdx++) {
            CHK_PRT_RET(threadIdx >= threads.size(),
                HCCL_ERROR("[InsTempReduceScatterMesh1DTwoShot][RunReduceScatterShot] threadIdx[%u] >= size[%u].",
                    threadIdx, threads.size()),
                HCCL_E_INTERNAL);
            const ChannelInfo &channel = curChannels[channelIdx];
            void *remoteCclBuffAddr = channel.remoteCclMem.addr;
            std::vector<DataSlice> txSrcSlices;
            std::vector<DataSlice> txDstSlices;
            std::vector<DataSlice> rxSrcSlices;
            std::vector<DataSlice> rxDstSlices;
            for (u32 repeatIdx = 0; repeatIdx < tempAlgParams.repeatNum; repeatIdx++) {
                DataSlice txSrcSlice(tempAlgParams.buffInfo.inputPtr,
                    tempAlgParams.buffInfo.inBuffBaseOff + repeatIdx * tempAlgParams.inputRepeatStride +
                        nextRank * tempAlgParams.inputSliceStride + sendOffset + sendSplitOffset[channelIdx],
                    sendSize[channelIdx], sendElemCount[channelIdx]);
                DataSlice txDstSlice(remoteCclBuffAddr,
                    GetScratchSlotOffset(tempAlgParams, repeatIdx, txGroupSlot) + sendDstOffset + sendSplitOffset[channelIdx],
                    sendSize[channelIdx], sendElemCount[channelIdx]);
                DataSlice rxSrcSlice(remoteCclBuffAddr,
                    tempAlgParams.buffInfo.inBuffBaseOff + repeatIdx * tempAlgParams.inputRepeatStride +
                        myAlgRank * tempAlgParams.inputSliceStride + recvOffset + recvSplitOffset[channelIdx],
                    recvSize[channelIdx], recvElemCount[channelIdx]);
                DataSlice rxDstSlice(tempAlgParams.buffInfo.hcclBuff.addr,
                    GetScratchSlotOffset(tempAlgParams, repeatIdx, rxGroupSlot) + recvOffset + recvSplitOffset[channelIdx],
                    recvSize[channelIdx], recvElemCount[channelIdx]);
                txSrcSlices.push_back(txSrcSlice);
                txDstSlices.push_back(txDstSlice);
                rxSrcSlices.push_back(rxSrcSlice);
                rxDstSlices.push_back(rxDstSlice);
            }
            if (secondShot && !currIsUnpaired) {
                SendRecvReduceInfo info{{channel, channel}, {{txSrcSlices, txDstSlices}, {rxSrcSlices, rxDstSlices}},
                    dataType_, reduceOp_};
                CHK_PRT_RET(SendRecvBatchWriteReduce(info, threads[threadIdx]),
                    HCCL_ERROR("[InsTempReduceScatterMesh1DTwoShot] RunReduceScatterShot SendRecvReduce failed"),
                    HCCL_E_INTERNAL);
            } else {
                SendRecvInfo info{{channel, channel}, {{txSrcSlices, txDstSlices}, {rxSrcSlices, rxDstSlices}}, dataType_};
                CHK_PRT_RET(SendRecvBatchWrite(info, threads[threadIdx]),
                    HCCL_ERROR("[InsTempReduceScatterMesh1DTwoShot] RunReduceScatterShot SendRecv failed"),
                    HCCL_E_INTERNAL);
            }
            threadIdx++;
        }
    }
    return HCCL_SUCCESS;
}

HcclResult InsTempReduceScatterMesh1DTwoShot::PostCopyTwoShot(const OpParam &param,
    const TemplateDataParams &tempAlgParams, const std::vector<ThreadHandle> &threads)
{
    u32 rankIdx = 0;
    CHK_RET(GetAlgRank(myRank_, subCommRanks_[0], rankIdx));
    u64 sliceSize = GetSliceSize(rankIdx, tempAlgParams);
    u64 sliceCount = GetSliceCount(sliceSize);
    u32 repeatThreadNum = static_cast<u32>(std::min<size_t>(threads.size(), tempAlgParams.repeatNum));
    std::vector<ThreadHandle> postCopyThreads(threads.begin(), threads.begin() + repeatThreadNum);
    if (repeatThreadNum > 1) {
        std::vector<ThreadHandle> subThreads(postCopyThreads.begin() + 1, postCopyThreads.end());
        GetNotifyIdxMainToSub(notifyIdxMainToSub_);
        CHK_RET(PreSyncInterThreads(postCopyThreads[0], subThreads, notifyIdxMainToSub_));
    }
    for (u32 repeatIdx = 0; repeatIdx < tempAlgParams.repeatNum; repeatIdx++) {
        ThreadHandle currThread = postCopyThreads[repeatIdx % repeatThreadNum];
        DataSlice myRankSlice(tempAlgParams.buffInfo.inputPtr,
            tempAlgParams.buffInfo.inBuffBaseOff + repeatIdx * tempAlgParams.inputRepeatStride +
                rankIdx * tempAlgParams.inputSliceStride,
            sliceSize, sliceCount);
        DataSlice outputSlice(tempAlgParams.buffInfo.outputPtr,
            tempAlgParams.buffInfo.outBuffBaseOff + repeatIdx * tempAlgParams.outputRepeatStride,
            sliceSize, sliceCount);
        if (!(tempAlgParams.buffInfo.inBuffType == tempAlgParams.buffInfo.outBuffType &&
                myRankSlice.GetOffset() == outputSlice.GetOffset())) {
            CHK_RET(LocalCopy(currThread, myRankSlice, outputSlice));
        }
        if (dataType_ == HCCL_DATA_TYPE_INT64 || dataType_ == HCCL_DATA_TYPE_UINT64 ||
            dataType_ == HCCL_DATA_TYPE_FP64 || param.reduceType == HcclReduceOp::HCCL_REDUCE_PROD) {
            CHK_RET(static_cast<HcclResult>(HcommBatchModeEnd(param.algTag)));
            CHK_RET(static_cast<HcclResult>(HcommBatchModeStart(param.algTag)));
            for (const auto &thread : threads) {
                CHK_RET(static_cast<HcclResult>(HcommThreadJoin(thread, CUSTOM_TIMEOUT)));
            }
        }
        for (u32 orderIdx = 0; orderIdx < templateRankSize_ - 1; orderIdx++) {
            u32 remoteIdx = (rankIdx + 1 + orderIdx) % templateRankSize_;
            if (!IsUnpairedRemote(rankIdx, remoteIdx) && IsSecondShotRemote(rankIdx, remoteIdx)) {
                continue;
            }
            DataSlice srcSlice(tempAlgParams.buffInfo.hcclBuff.addr,
                GetScratchSlotOffset(tempAlgParams, repeatIdx, GetGroupSlotIdx(rankIdx, remoteIdx)), sliceSize, sliceCount);
            CHK_RET(LocalReduce(currThread, srcSlice, outputSlice, dataType_, reduceOp_));
        }
    }
    if (repeatThreadNum > 1) {
        std::vector<ThreadHandle> subThreads(postCopyThreads.begin() + 1, postCopyThreads.end());
        GetNotifyIdxSubToMain(notifyIdxSubToMain_);
        CHK_RET(PostSyncInterThreads(postCopyThreads[0], subThreads, notifyIdxSubToMain_));
    }
    return HCCL_SUCCESS;
}

u64 InsTempReduceScatterMesh1DTwoShot::GetSliceSize(u32 rankIdx, const TemplateDataParams &tempAlgParams) const
{
    if (rankIdx == templateRankSize_ - 1 && tempAlgParams.tailSize != 0) {
        return tempAlgParams.tailSize;
    }
    return tempAlgParams.sliceSize;
}

u64 InsTempReduceScatterMesh1DTwoShot::GetSliceCount(u64 sliceSize) const
{
    return dataTypeSize_ == 0 ? 0 : sliceSize / dataTypeSize_;
}

u64 InsTempReduceScatterMesh1DTwoShot::GetScratchSlotOffset(
    const TemplateDataParams &tempAlgParams, u32 repeatIdx, u32 slotIdx) const
{
    return tempAlgParams.buffInfo.hcclBuffBaseOff + repeatIdx * tempAlgParams.outputRepeatStride +
        slotIdx * tempAlgParams.sliceSize;
}

u32 InsTempReduceScatterMesh1DTwoShot::GetRemoteOrderIdx(u32 ownerIdx, u32 srcIdx) const
{
    return (srcIdx + templateRankSize_ - ownerIdx - 1) % templateRankSize_;
}

u32 InsTempReduceScatterMesh1DTwoShot::GetPairedRemoteOrderIdx(u32 ownerIdx, u32 srcIdx) const
{
    u32 orderIdx = GetRemoteOrderIdx(ownerIdx, srcIdx);
    u32 unpairedOrderIdx = templateRankSize_ / 2 - 1;
    if (orderIdx == unpairedOrderIdx) {
        return orderIdx;
    }
    return (orderIdx < unpairedOrderIdx) ? orderIdx : (orderIdx - 1);
}

u32 InsTempReduceScatterMesh1DTwoShot::GetPairedSrcIdx(u32 ownerIdx, u32 srcIdx) const
{
    u32 pairedOrderIdx = GetPairedRemoteOrderIdx(ownerIdx, srcIdx) ^ 1U;
    u32 unpairedOrderIdx = templateRankSize_ / 2 - 1;
    u32 remoteOrderIdx = pairedOrderIdx;
    if (remoteOrderIdx >= unpairedOrderIdx) {
        remoteOrderIdx++;
    }
    return (ownerIdx + 1 + remoteOrderIdx) % templateRankSize_;
}

u32 InsTempReduceScatterMesh1DTwoShot::GetGroupSlotIdx(u32 ownerIdx, u32 srcIdx) const
{
    if (IsUnpairedRemote(ownerIdx, srcIdx)) {
        return srcIdx;
    }
    u32 unpairedOrderIdx = templateRankSize_ / 2 - 1;
    u32 orderIdx = GetPairedRemoteOrderIdx(ownerIdx, srcIdx);
    u32 groupFirstOrderIdx = (orderIdx / 2) * 2;
    if (groupFirstOrderIdx >= unpairedOrderIdx) {
        groupFirstOrderIdx++;
    }
    return (ownerIdx + 1 + groupFirstOrderIdx) % templateRankSize_;
}

bool InsTempReduceScatterMesh1DTwoShot::IsSecondShotRemote(u32 ownerIdx, u32 srcIdx) const
{
    return (GetPairedRemoteOrderIdx(ownerIdx, srcIdx) % 2) == 1;
}

bool InsTempReduceScatterMesh1DTwoShot::IsUnpairedRemote(u32 ownerIdx, u32 srcIdx) const
{
    return GetRemoteOrderIdx(ownerIdx, srcIdx) == (templateRankSize_ / 2 - 1);
}

bool InsTempReduceScatterMesh1DTwoShot::CanRunTwoShot(const TemplateDataParams &tempAlgParams) const
{
    if (templateRankSize_ <= 2 || (templateRankSize_ % 2) != 0) {
        return false;
    }
    for (u32 rankIdx = 0; rankIdx < templateRankSize_; rankIdx++) {
        u64 sliceSize = GetSliceSize(rankIdx, tempAlgParams);
        u64 firstHalfSize = GetFirstHalfSize(sliceSize);
        if (firstHalfSize == 0 || firstHalfSize == sliceSize) {
            HCCL_INFO("[InsTempReduceScatterMesh1DTwoShot] fallback to one-shot for sliceSize[%llu].", sliceSize);
            return false;
        }
    }
    return true;
}

u64 InsTempReduceScatterMesh1DTwoShot::GetFirstHalfSize(u64 sliceSize) const
{
    if (dataTypeSize_ == 0) {
        return sliceSize / 2;
    }
    u64 elementNum = sliceSize / dataTypeSize_;
    return (elementNum / 2) * dataTypeSize_;
}

u64 InsTempReduceScatterMesh1DTwoShot::GetPutReduceDstOffset(u64 sliceSize, u32 ownerIdx, u32 srcIdx) const
{
    u32 pairedSrcIdx = GetPairedSrcIdx(ownerIdx, srcIdx);
    return GetShotOffset(sliceSize, ownerIdx, pairedSrcIdx, false);
}

u64 InsTempReduceScatterMesh1DTwoShot::GetShotOffset(u64 sliceSize, u32 ownerIdx, u32 srcIdx, bool secondShot) const
{
    if (IsUnpairedRemote(ownerIdx, srcIdx)) {
        return secondShot ? GetFirstHalfSize(sliceSize) : 0;
    }
    bool secondShotRemote = IsSecondShotRemote(ownerIdx, srcIdx);
    if (secondShotRemote == secondShot) {
        return 0;
    }
    return GetFirstHalfSize(sliceSize);
}

u64 InsTempReduceScatterMesh1DTwoShot::GetShotSize(u64 sliceSize, u32 ownerIdx, u32 srcIdx, bool secondShot) const
{
    u64 firstHalfSize = GetFirstHalfSize(sliceSize);
    return (GetShotOffset(sliceSize, ownerIdx, srcIdx, secondShot) == 0) ? firstHalfSize : (sliceSize - firstHalfSize);
}

void InsTempReduceScatterMesh1DTwoShot::GetNotifyIdxMainToSub(std::vector<u32> &notifyIdxMainToSub)
{
    notifyIdxMainToSub.clear();
    u32 threadNum = GetThreadNum();
    for (u32 slaveThreadIdx = 0; slaveThreadIdx < threadNum - 1; slaveThreadIdx++) {
        notifyIdxMainToSub.push_back(0);
    }
}

void InsTempReduceScatterMesh1DTwoShot::GetNotifyIdxSubToMain(std::vector<u32> &notifyIdxSubToMain)
{
    notifyIdxSubToMain.clear();
    u32 threadNum = GetThreadNum();
    notifyIdxSubToMain.resize(threadNum - 1);
    std::iota(notifyIdxSubToMain.begin(), notifyIdxSubToMain.end(), 0);
}
} // namespace ops_hccl
