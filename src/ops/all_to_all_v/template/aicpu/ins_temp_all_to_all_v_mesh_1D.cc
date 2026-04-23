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

namespace ops_hccl {
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
    std::vector<HcclChannelDesc> level0Channels;
    CHK_RET(CalcChannelRequestMesh1D(comm, param, topoInfo, subCommRanks_, level0Channels));
    resourceRequest.channels.push_back(level0Channels);
    u32 channelsPerRank = CalcChannelsPerRank(level0Channels);
    resourceRequest.slaveThreadNum = ALLTOALLV_DIRECT_FULLMESH_CONCURRENT_SIZE * channelsPerRank;
    for (u32 index = 0; index < resourceRequest.slaveThreadNum; index++) {
        resourceRequest.notifyNumPerThread.push_back(1);
    }
    resourceRequest.notifyNumOnMainThread = resourceRequest.slaveThreadNum;
    return HCCL_SUCCESS;
}

u64 InsTempAlltoAllVMesh1D::CalcScratchMultiple(BufferType inBuffType, BufferType outBuffType)
{
    (void) inBuffType;
    (void) outBuffType;
    // 分组fullmesh，每轮最多通信maxConcurrentSize_个
    concurrentSendRecvNum_ = std::min(ALLTOALLV_DIRECT_FULLMESH_CONCURRENT_SIZE, templateRankSize_ - 1);
    return concurrentSendRecvNum_;
}

void InsTempAlltoAllVMesh1D::CalcCommRankSetForOneLoop(u32 roundIdx, const u32 remainRankSize,
    std::vector<u32> &commRanks) const
{
    commRanks.clear();
    u32 pairNumPerRound = (concurrentSendRecvNum_ + 1) / 2;
    u32 pairSize = (remainRankSize < concurrentSendRecvNum_) ? (remainRankSize +  1) / 2: pairNumPerRound;
    for (u32 i = roundIdx * pairNumPerRound + 1; i < (roundIdx * pairNumPerRound + pairSize + 1); i++) {
        u32 leftRemoteRank = (myRank_ + templateRankSize_ - i) % templateRankSize_;
        u32 rightRemoteRank = (myRank_ + i) % templateRankSize_;
        if (leftRemoteRank == rightRemoteRank) {
            commRanks.push_back(leftRemoteRank);
            break;
        } else {
            commRanks.push_back(leftRemoteRank);
            commRanks.push_back(rightRemoteRank);
        }
    }
    return;
}

u32 InsTempAlltoAllVMesh1D::CalcCommLoops() const
{
    u32 totalCommRankSize = templateRankSize_ - 1; // 除去本rank
    return (totalCommRankSize + concurrentSendRecvNum_ - 1) / concurrentSendRecvNum_;
}

void InsTempAlltoAllVMesh1D::CalcRemoteCclBuffIdx(u32 remoteRank, u32 &myRankRecvCclBuffIdx, u32 &remoteRecvCclBuffIdx) const
{
    u32 pairNum = (concurrentSendRecvNum_ + 1) / 2;
    // 以myRank为基准，计算remoteRank相对于它的gapRight和gapLeft
    // 反过来就是myRank相对于remoteRank的gapLeft和gapRight
    u32 gapRight = (templateRankSize_ + remoteRank - myRank_) % templateRankSize_;
    u32 gapLeft = (templateRankSize_ + myRank_ - remoteRank) % templateRankSize_;
    u32 remoteCclBuffIdx = 0;
    if (gapLeft < gapRight) {
        // remoteRank是myRank左边的rank，myRank是remoteRank右边的rank
        u32 gap = gapLeft;
        myRankRecvCclBuffIdx = pairNum - 1 - ((gap - 1) % pairNum);
        remoteRecvCclBuffIdx = pairNum + ((gap - 1) % pairNum);
    } else if (gapLeft > gapRight) {
        // remoteRank是myRank右边的rank，myRank是remoteRank右边的rank
        u32 gap = gapRight;
        myRankRecvCclBuffIdx = pairNum + ((gap - 1) % pairNum);
        remoteRecvCclBuffIdx = pairNum - 1 - ((gap - 1) % pairNum);
    } else {
        myRankRecvCclBuffIdx = 0;
        remoteRecvCclBuffIdx = 0;
    }
    HCCL_DEBUG("[InsTempAlltoAllVMesh1D][CalcRemoteCclBuffIdx] For my rank[%u] and remote rank[%u], "\
        "my ccl buff idx is [%u], remote ccl buff idx is [%u].",
        myRank_, remoteRank, myRankRecvCclBuffIdx, remoteRecvCclBuffIdx);
    return remoteCclBuffIdx;
}

HcclResult InsTempAlltoAllVMesh1D::KernelRun(const OpParam& param,
    const TemplateDataParams& tempAlgParams,
    TemplateResource& templateResource)
{
    threadNum_ = templateResource.threads.size();
    processSize_ = tempAlgParams.sliceSize;
    count_ = tempAlgParams.count;
    dataType_ = param.all2AllVDataDes.sendType;
    dataTypeSize_ = SIZE_TABLE[dataType_];
    HCCL_INFO("[InsTempAlltoAllVMesh1D] Run Start");

    u32 myAlgRank = 0;
    auto iter = std::find(subCommRanks_[0].begin(), subCommRanks_[0].end(), myRank_);
    if (iter != subCommRanks_[0].end()) {
        myAlgRank = std::distance(subCommRanks_[0].begin(), iter);
    } else {
        HCCL_ERROR("[InsTempAlltoAllVMesh1D][KernelRun] subCommRanks_ or myRank_ is error.");
        return HCCL_E_INTERNAL;
    }

    if (threadNum_ > 1) {
        std::vector<ThreadHandle> subThreads(templateResource.threads.begin() + 1, templateResource.threads.end());
        GetNotifyIdxMainToSub(notifyIdxMainToSub_);
        CHK_RET(PreSyncInterThreads(templateResource.threads[0], subThreads, notifyIdxMainToSub_));
    }
    CHK_RET(RunALLtoALL(templateResource.channels, templateResource.threads, tempAlgParams, myAlgRank));
    if (threadNum_ > 1) {
        std::vector<ThreadHandle> subThreads(templateResource.threads.begin() + 1, templateResource.threads.end());
        GetNotifyIdxSubToMain(notifyIdxSubToMain_);
        CHK_RET(PostSyncInterThreads(templateResource.threads[0], subThreads, notifyIdxSubToMain_));
    }

    HCCL_INFO("[InsTempAlltoAllVMesh1D] Run End");
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
    const std::map<u32, std::vector<ChannelInfo>> &channels,
    const std::vector<ThreadHandle> &threads,
    const TemplateDataParams &tempAlgParams,
    const u32 myAlgRank)
{
    // 第0条流做本卡local copy
    u32 queIdx = 0;
    CHK_RET(LocalCopyForMyRank(tempAlgParams, threads[queIdx], myAlgRank, queIdx));
    // 计算通信轮数
    u32 commLoops = CalcCommLoops();
    u32 remainRankSize = templateRankSize_ - 1;
    std::vector<u32> commRanks;

    for (u32 roundIdx = 0; roundIdx < commLoops && remainRankSize > 0; roundIdx++) {
        queIdx = 1; // 每轮通信都从第1条流开始
        CHK_RET(CalcCommRankSetForOneLoop(roundIdx, remainRankSize, commRanks));
        CHK_RET(RunSendRecvByLoop(commRanks, tempAlgParams, channels, threads, queIdx));
        remainRankSize -= commRanks.size();
        HCCL_DEBUG("[InsTempAlltoAllVMesh1D][RunALLtoALL] round[%u] finish, commRank size is [%zu], "\
            "remainRankSize is [%u].", roundIdx, commRanks.size(), remainRankSize);
    }
    return HCCL_SUCCESS;
}

HcclResult InsTempAlltoAllVMesh1D::RunSendRecvByLoop(const std::vector<u32> &commRanks,
    const TemplateDataParams &tempAlgParams,
    const std::map<u32, std::vector<ChannelInfo>> &channels,
    const std::vector<ThreadHandle> &threads, u32 &queIdx) const
{
    // 遍历本次通信的所有rank
    for (u32 rankIdx = 0; rankIdx < commRanks.size(); rankIdx++) {
        u32 remoteRank = commRanks[rankIdx];
        // 计算本端发送的数据在远端cclbuffer中的index
        u32 myRankRecvCclBuffIdx = 0;
        u32 remoteRecvCclBuffIdx = 0;
        CalcRemoteCclBuffIdx(remoteRank, myRankRecvCclBuffIdx, remoteRecvCclBuffIdx);
        // 取出本次通信对端的channel
        if (channels.find(remoteRank) == channels.end()) {
            HCCL_ERROR("[InsTempAlltoAllVMesh1D][RunSendRecvByLoop] remoteRank[%u] "\
                "does not exist in channels map!", remoteRank);
            return HCCL_E_PARA;
        }
        const std::vector<ChannelInfo> &curChannels = channels.at(remoteRank);
        // send数据按照channel分片
        std::vector<u64> sendCountsSplit;
        std::vector<u64> sendSizeSplit;
        std::vector<u64> sendOffsetSplit;
        CHK_RET(CalcDataSplitByPortGroup(tempAlgParams.sendCounts[remoteRank], dataTypeSize_, curChannels,
            sendCountsSplit, sendSizeSplit, sendOffsetSplit));
        // recv数据按照channel分片
        std::vector<u64> recvCountsSplit;
        std::vector<u64> recvSizeSplit;
        std::vector<u64> recvOffsetSplit;
        CHK_RET(CalcDataSplitByPortGroup(tempAlgParams.recvCounts[remoteRank], dataTypeSize_, curChannels,
            recvCountsSplit, recvSizeSplit, recvOffsetSplit));
        for (u32 channelId = 0; channelId < curChannels.size(); channelId++) {
            const ChannelInfo &channelSend = curChannels[channelId]; // 发给哪个rank
            const ChannelInfo &channelRecv = curChannels[channelId]; // 收哪个rank的数据
            std::vector<DataSlice> txSrcSlices;
            std::vector<DataSlice> txDstSlices;
            std::vector<DataSlice> rxSrcSlices; // 在write模式下用不到rxSlice，直接给空的
            std::vector<DataSlice> rxDstSlices;

            void* remoteCclBuffAddr = channelSend.remoteCclMem.addr;
            // 本端input slice
            DataSlice txSrcSlice = DataSlice(tempAlgParams.buffInfo.inputPtr,
                tempAlgParams.sdispls[remoteRank] * dataTypeSize_ + sendOffsetSplit[channelId],
                sendSizeSplit[channelId], sendCountsSplit[channelId]);
            // 远端ccl buffer slice
            DataSlice txDstSlice = DataSlice(remoteCclBuffAddr,
                remoteRecvCclBuffIdx * tempAlgParams.inputSliceStride + sendOffsetSplit[channelId] +
                tempAlgParams.buffInfo.hcclBuffBaseOff,
                sendSizeSplit[channelId], sendCountsSplit[channelId]);
            txSrcSlices.push_back(txSrcSlice);
            txDstSlices.push_back(txDstSlice);

            if (sendSizeSplit[channelId] > 0 && recvSizeSplit[channelId] > 0) {
                SendRecvInfo sendRecvInfo{{channelSend, channelRecv},
                    {{txSrcSlices, txDstSlices}, {rxSrcSlices, rxDstSlices}}};
                CHK_PRT_RET(SendRecvWrite(sendRecvInfo, threads[queIdx]),
                    HCCL_ERROR("[InsTempAlltoAllVMesh1D] RunALLtoALL SendRecvInfo failed"),
                    HcclResult::HCCL_E_INTERNAL);
            } else { // 如果send/recv只有一个数据量不为0，调用单独的send/recv接口
                if (sendSizeSplit[channelId] > 0) {
                    DataInfo sendInfo{channelSend, {txSrcSlices, txDstSlices}};
                    CHK_PRT_RET(SendWrite(sendInfo, threads[queIdx]),
                        HCCL_ERROR("[InsTempAlltoAllVMesh1D] RunALLtoALL sendInfo failed"),
                        HcclResult::HCCL_E_INTERNAL);
                } else if (recvSizeSplit[channelId] > 0) {
                    DataInfo recvInfo{channelRecv, {rxSrcSlices, rxDstSlices}};
                    CHK_PRT_RET(RecvWrite(recvInfo, threads[queIdx]),
                        HCCL_ERROR("[InsTempAlltoAllVMesh1D] RunALLtoALL recvInfo failed"),
                        HcclResult::HCCL_E_INTERNAL);
                }
            }
            HCCL_DEBUG("[InsTempAlltoAllVMesh1D][RunSendRecvByLoop] do send recv write on thread[%u], "\
                "send size[%llu], recv size[%llu], remote rank[%u].",
                queIdx, sendSizeSplit[channelId], recvSizeSplit[channelId], remoteRank);
            if (recvSizeSplit[channelId] > 0) {
                CHK_RET(PostCopy(tempAlgParams, threads[queIdx], myRankRecvCclBuffIdx,
                    recvSizeSplit[channelId], recvCountsSplit[channelId], recvOffsetSplit[channelId]));
            }
            queIdx++;
        }
    }
    return HcclResult::HCCL_SUCCESS;
}

HcclResult InsTempAlltoAllVMesh1D::PostCopy(const TemplateDataParams &tempAlgParams, const ThreadHandle &thread,
    const u32 myRankRecvCclBuffIdx, const u64 &recvSize, const u64 &recvCount, const u64 &recvOffset) const
{
    // ccl buffer的数据搬运到usrout
    // 远端的数据发送到本端ccl buffer的slice
    DataSlice localCopySrcSlice = DataSlice(tempAlgParams.buffInfo.hcclBuff.addr,
        myRankRecvCclBuffIdx * tempAlgParams.inputSliceStride + tempAlgParams.buffInfo.hcclBuffBaseOff +
        recvOffset, recvSize, recvCount);
    // 本端output buffer slice
    DataSlice localCopyDstSlice = DataSlice(tempAlgParams.buffInfo.outputPtr,
        tempAlgParams.rdispls[myRankRecvCclBuffIdx] * dataTypeSize_ + recvOffset,
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
} // namespace Hccl