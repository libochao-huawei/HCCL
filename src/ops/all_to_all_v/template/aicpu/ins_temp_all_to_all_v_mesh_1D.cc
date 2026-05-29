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

#define NET_NUM 2

namespace ops_hccl {
namespace {
constexpr u32 VMESH_4X4_RANK_NUM = 16;
constexpr u32 VMESH_4X4_ROW_NUM = 4;
constexpr u32 VMESH_4X4_COL_NUM = 4;
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

    std::vector<HcclChannelDesc> level0Channels;
    if(topoInfo->level0Topo == Level0Shape::MESH_1D_CLOS && !topoInfo->level0PcieMix) {
        std::vector<HcclChannelDesc> myChannelDescs;
        CHK_RET(CalcChannelRequestMesh1DWithPriorityTopo(comm, param, topoInfo, subCommRanks_, myChannelDescs, CommTopo::COMM_TOPO_1DMESH));
        for(auto channel : myChannelDescs) {
            if(channel.channelProtocol == COMM_PROTOCOL_UBC_CTP) {
                level0Channels.push_back(channel);
            }
        }
        HCCL_DEBUG("[InsTempAlltoAllVMesh1D::CalcRes] Get Channel Success!");
    } else {
        CHK_RET(CalcChannelRequestMesh1D(comm, param, topoInfo, subCommRanks_, level0Channels));
    }
    resourceRequest.channels.push_back(level0Channels);
    channelsPerRank_ = CalcChannelsPerRank(level0Channels);
    HCCL_INFO("[InsTempAlltoAllVMesh1D][CalcRes] channelsPerRank_ is [%u]", channelsPerRank_);
    resourceRequest.slaveThreadNum = std::min(ALLTOALLV_DIRECT_FULLMESH_CONCURRENT_SIZE, templateRankSize_ - 1) * channelsPerRank_;
    for (u32 index = 0; index < resourceRequest.slaveThreadNum; index++) {
        // 从流的notify数量以rank间channel数的最大值为准，用于和主流同步以及同一个rank多条链路间的同步
        resourceRequest.notifyNumPerThread.push_back(channelsPerRank_);
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

void InsTempAlltoAllVMesh1D::CalcCommRankSetForOneLoop(const u32 roundIdx, const u32 remainRankSize,
    std::vector<u32> &commRanks) const
{
    (void) remainRankSize;
    commRanks.clear();
    if (!IsVmesh4X4Mode()) {
        HCCL_ERROR("[InsTempAlltoAllVMesh1D][CalcCommRankSetForOneLoop] only support 4x4 vmesh mode.");
        return;
    }
    u32 myRankIndex = 0;
    if (GetRankIndexInSubComm(myRank_, myRankIndex) != HCCL_SUCCESS) {
        return;
    }
    if (roundIdx >= VMESH_4X4_ROW_NUM) {
        HCCL_ERROR("[InsTempAlltoAllVMesh1D][CalcCommRankSetForOneLoop] roundIdx[%u] is invalid.", roundIdx);
        return;
    }
    u32 myRowIdx = myRankIndex / VMESH_4X4_COL_NUM;
    u32 rowOffset = (roundIdx + 1) % VMESH_4X4_ROW_NUM;
    u32 targetRowIdx = 0;
    if ((myRowIdx % 2) == 1) {
        targetRowIdx = (myRowIdx + rowOffset) % VMESH_4X4_ROW_NUM;
    } else {
        targetRowIdx = (myRowIdx + VMESH_4X4_ROW_NUM - rowOffset) % VMESH_4X4_ROW_NUM;
    }
    for (u32 targetColIdx = 0; targetColIdx < VMESH_4X4_COL_NUM; targetColIdx++) {
        u32 remoteRankIndex = targetRowIdx * VMESH_4X4_COL_NUM + targetColIdx;
        u32 remoteRank = subCommRanks_[0][remoteRankIndex];
        if (remoteRank != myRank_) {
            commRanks.push_back(remoteRank);
        }
    }
    return;
}

u32 InsTempAlltoAllVMesh1D::CalcCommLoops() const
{
    return VMESH_4X4_ROW_NUM;
}

void InsTempAlltoAllVMesh1D::CalcCclBuffIdx(u32 remoteRank, u32 &myRankCclBuffIdx, u32 &remoteCclBuffIdx) const
{
    myRankCclBuffIdx = 0;
    remoteCclBuffIdx = 0;
    if (!IsVmesh4X4Mode()) {
        HCCL_ERROR("[InsTempAlltoAllVMesh1D][CalcCclBuffIdx] only support 4x4 vmesh mode.");
        return;
    }

    u32 myRankIndex = 0;
    u32 remoteRankIndex = 0;
    if (GetRankIndexInSubComm(myRank_, myRankIndex) != HCCL_SUCCESS ||
        GetRankIndexInSubComm(remoteRank, remoteRankIndex) != HCCL_SUCCESS) {
        return;
    }
    if (myRankIndex == remoteRankIndex) {
        HCCL_ERROR("[InsTempAlltoAllVMesh1D][CalcCclBuffIdx] 本端rank[%u]和对端rank[%u]不能相同.",
            myRank_, remoteRank);
        return;
    }

    u32 myRowIdx = myRankIndex / VMESH_4X4_COL_NUM;
    u32 myColIdx = myRankIndex % VMESH_4X4_COL_NUM;
    u32 remoteRowIdx = remoteRankIndex / VMESH_4X4_COL_NUM;
    u32 remoteColIdx = remoteRankIndex % VMESH_4X4_COL_NUM;
    if (myRowIdx == remoteRowIdx) {
        u32 rowInnerIdx = remoteColIdx < myColIdx ? remoteColIdx : remoteColIdx - 1;
        u32 remoteRowInnerIdx = myColIdx < remoteColIdx ? myColIdx : myColIdx - 1;
        myRankCclBuffIdx = VMESH_4X4_COL_NUM * (VMESH_4X4_ROW_NUM - 1) + rowInnerIdx;
        remoteCclBuffIdx = VMESH_4X4_COL_NUM * (VMESH_4X4_ROW_NUM - 1) + remoteRowInnerIdx;
    } else {
        u32 myRowOffset = 0;
        if ((myRowIdx % 2) == 1) {
            myRowOffset = (remoteRowIdx + VMESH_4X4_ROW_NUM - myRowIdx) % VMESH_4X4_ROW_NUM;
        } else {
            myRowOffset = (myRowIdx + VMESH_4X4_ROW_NUM - remoteRowIdx) % VMESH_4X4_ROW_NUM;
        }
        u32 remoteRowOffset = 0;
        if ((remoteRowIdx % 2) == 1) {
            remoteRowOffset = (myRowIdx + VMESH_4X4_ROW_NUM - remoteRowIdx) % VMESH_4X4_ROW_NUM;
        } else {
            remoteRowOffset = (remoteRowIdx + VMESH_4X4_ROW_NUM - myRowIdx) % VMESH_4X4_ROW_NUM;
        }
        if (myRowOffset == 0 || remoteRowOffset == 0) {
            HCCL_ERROR("[InsTempAlltoAllVMesh1D][CalcCclBuffIdx] invalid row offset, myRowIdx[%u], "
                "remoteRowIdx[%u], myRank[%u], remoteRank[%u].", myRowIdx, remoteRowIdx, myRank_, remoteRank);
            return;
        }
        myRankCclBuffIdx = VMESH_4X4_COL_NUM * (myRowOffset - 1) + remoteColIdx;
        remoteCclBuffIdx = VMESH_4X4_COL_NUM * (remoteRowOffset - 1) + myColIdx;
    }
    HCCL_DEBUG("[InsTempAlltoAllVMesh1D][CalcCclBuffIdx] For my rank[%u] and remote rank[%u], "\
        "my ccl buff idx is [%u], remote ccl buff idx is [%u].",
        myRank_, remoteRank, myRankCclBuffIdx, remoteCclBuffIdx);
    return;
}

HcclResult InsTempAlltoAllVMesh1D::KernelRun(const OpParam& param,
    const TemplateDataParams& tempAlgParams,
    TemplateResource& templateResource)
{
    HCCL_INFO("[InsTempAlltoAllVMesh1D][KernelRun] Run Start");
    threadNum_ = templateResource.threads.size();
    dataType_ = param.all2AllVDataDes.sendType;
    dataTypeSize_ = SIZE_TABLE[dataType_];

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
    u32 commLoops = CalcCommLoops();
    u32 remainRankSize = templateRankSize_ - 1;
    channelsPerRank_ = CalcChannelsPerRank(channels); // 每个rank的channel数量的最大值
    std::vector<u32> commRanks;

    std::vector<ThreadHandle> subThreads;
    if (threadNum_ > 1) {
        // 只做一次全量的前同步
        subThreads.assign(threads.begin() + 1, threads.end());
        GetNotifyIdxMainToSub(notifyIdxMainToSub_);
        CHK_RET(PreSyncInterThreads(threads[0], subThreads, notifyIdxMainToSub_));
    }
    for (u32 roundIdx = 0; roundIdx < commLoops && remainRankSize > 0; roundIdx++) {
        CalcCommRankSetForOneLoop(roundIdx, remainRankSize, commRanks); // 计算本轮通信rank
        if (isDmaRead_) {
            if (roundIdx == 0) {
                // 如果是read模式，第一轮做统一的前拷贝
                CHK_RET(PreCopyByLoop(commRanks, channels, threads, tempAlgParams, myAlgRank));
                if (threadNum_ > 1) {
                    GetNotifyIdxSubToMain(notifyIdxSubToMain_);
                    CHK_RET(PostSyncInterThreads(threads[0], subThreads, notifyIdxSubToMain_)); // 第1轮通信中将前拷贝与本卡数据拷贝错开
                    CHK_RET(PreSyncInterThreads(threads[0], subThreads, notifyIdxMainToSub_));
                }
                CHK_RET(LocalCopyForMyRank(tempAlgParams, threads[0], myAlgRank, 0)); // 在第1轮通信中用0号流做本卡数据拷贝
            }
            CHK_RET(RunSendRecvByLoop(commRanks, tempAlgParams, channels, threads, roundIdx, commLoops));
            remainRankSize -= commRanks.size();
        } else {
            if (roundIdx == 0) {
                CHK_RET(LocalCopyForMyRank(tempAlgParams, threads[0], myAlgRank, 0)); // 在第1轮通信中用0号流做本卡数据拷贝
            }
            CHK_RET(RunSendRecvByLoop(commRanks, tempAlgParams, channels, threads, roundIdx, commLoops));
            remainRankSize -= commRanks.size();
            HCCL_DEBUG("[InsTempAlltoAllVMesh1D][RunALLtoALL] round[%u] finish, commRank size is [%zu], "\
                "remainRankSize is [%u].", roundIdx, commRanks.size(), remainRankSize);
        }
    }
    if (threadNum_ > 1) {
        // 只做一次全量的后同步
        GetNotifyIdxSubToMain(notifyIdxSubToMain_);
        CHK_RET(PostSyncInterThreads(threads[0], subThreads, notifyIdxSubToMain_));
    }
    return HCCL_SUCCESS;
}

HcclResult InsTempAlltoAllVMesh1D::RunSendRecvByLoop(const std::vector<u32> &commRanks,
    const TemplateDataParams &tempAlgParams,
    const std::map<u32, std::vector<ChannelInfo>> &channels,
    const std::vector<ThreadHandle> &threads, const u32 roundIdx, const u32 commLoops)
{
    // 遍历本次通信的所有rank
    for (u32 rankIdx = 0; rankIdx < commRanks.size(); rankIdx++) {
        u32 remoteRank = commRanks[rankIdx];
        // 取出本次通信对端的channel
        if (channels.find(remoteRank) == channels.end()) {
            HCCL_ERROR("[InsTempAlltoAllVMesh1D][RunSendRecvByLoop] remoteRank[%u] "\
                "does not exist in channels map!", remoteRank);
            return HCCL_E_PARA;
        }
        const std::vector<ChannelInfo> &allChannels = channels.at(remoteRank);
        std::vector<ChannelInfo> curChannels;
        CHK_RET(SelectChannel(remoteRank, allChannels, curChannels));
        // send数据按照channel分片
        CHK_RET(CalcDataSplitByPortGroupCommon(tempAlgParams.sendCounts[remoteRank], dataTypeSize_, curChannels,
            sendCountsSplit_, sendSizeSplit_, sendOffsetSplit_, static_cast<u32>(curChannels.size())));
        // recv数据按照channel分片
        CHK_RET(CalcDataSplitByPortGroupCommon(tempAlgParams.recvCounts[remoteRank], dataTypeSize_, curChannels,
            recvCountsSplit_, recvSizeSplit_, recvOffsetSplit_, static_cast<u32>(curChannels.size())));
        CHK_RET(RunSendRecvByChannel(tempAlgParams, roundIdx, curChannels, remoteRank, threads, commLoops));
    }
    return HcclResult::HCCL_SUCCESS;
}

HcclResult InsTempAlltoAllVMesh1D::PreSyncInterThreadsPerRank(const ThreadHandle &mainThreadCurRank,
    const std::vector<ThreadHandle> &subThreadsCurRank) const
{
    std::vector<u32> notifyIdxMainToSubCurRank;
    for (u32 subThreadIdx = 0; subThreadIdx < subThreadsCurRank.size(); subThreadIdx++) {
        notifyIdxMainToSubCurRank.emplace_back(1); // 第0个用于和全局的主流通信，第1个用于和rank内部的主流通信
    }
    CHK_RET(PreSyncInterThreads(mainThreadCurRank, subThreadsCurRank, notifyIdxMainToSubCurRank));
    return HcclResult::HCCL_SUCCESS;
}

HcclResult InsTempAlltoAllVMesh1D::PostSyncInterThreadsPerRank(const ThreadHandle &mainThreadCurRank,
    const std::vector<ThreadHandle> &subThreadsCurRank) const
{
    std::vector<u32> notifyIdxSubToMainCurRank;
    for (u32 subThreadIdx = 0; subThreadIdx < subThreadsCurRank.size(); subThreadIdx++) {
        notifyIdxSubToMainCurRank.emplace_back(subThreadIdx + 1); // 第0个用于和全局的主流通信，从第1个开始用于和rank内部的从流通信
    }
    CHK_RET(PostSyncInterThreads(mainThreadCurRank, subThreadsCurRank, notifyIdxSubToMainCurRank));
    return HcclResult::HCCL_SUCCESS;
}

HcclResult InsTempAlltoAllVMesh1D::RunSendRecvByChannel(const TemplateDataParams &tempAlgParams, const u32 roundIdx,
    const std::vector<ChannelInfo> &curChannels, const u32 remoteRank, const std::vector<ThreadHandle> &threads, const u32 commLoops) const
{
    u32 myRankCclBuffIdx = 0; // myRank与remoteRank交互时myRank提供的cclbuffer index
    u32 remoteCclBuffIdx = 0; // myRank与remoteRank交互时remoteRank提供的cclbuffer index
    CalcCclBuffIdx(remoteRank, myRankCclBuffIdx, remoteCclBuffIdx);
    u32 queIdx = myRankCclBuffIdx * channelsPerRank_ + 1;
    const ThreadHandle &mainThreadCurRank = threads[queIdx]; // 当前rank分配到的第一条流（rank内主流）
    std::vector<ThreadHandle> subThreadsCurRank; // 当前rank的rank内从流
    if (curChannels.size() > 1 && roundIdx != 0) {
        subThreadsCurRank.assign(threads.begin() + queIdx + 1, threads.begin() + queIdx + curChannels.size());
        PreSyncInterThreadsPerRank(mainThreadCurRank, subThreadsCurRank);
    }
    for (u32 channelId = 0; channelId < curChannels.size(); channelId++) {
        if (roundIdx != 0 && isDmaRead_ && sendSizeSplit_[channelId] > 0) {
            CHK_RET(static_cast<HcclResult>(PreCopy(tempAlgParams, threads[queIdx], myRankCclBuffIdx, remoteRank,
                sendSizeSplit_[channelId], sendCountsSplit_[channelId], sendOffsetSplit_[channelId])));
        }
        const ChannelInfo &channelSend = curChannels[channelId]; // 发给哪个rank
        const ChannelInfo &channelRecv = curChannels[channelId]; // 收哪个rank的数据
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
        HCCL_INFO("[InsTempAlltoAllVMesh1D][RunSendRecvByLoop] do send recv write on thread[%u], channelId[%u], "\
            "send size[%llu], recv size[%llu], remote rank[%u].",
            queIdx, channelId, sendSizeSplit_[channelId], recvSizeSplit_[channelId], remoteRank);
        if (!isDmaRead_ && recvSizeSplit_[channelId] > 0) {
            CHK_RET(PostCopy(tempAlgParams, threads[queIdx], myRankCclBuffIdx, remoteRank,
                recvSizeSplit_[channelId], recvCountsSplit_[channelId], recvOffsetSplit_[channelId]));
        }
        queIdx++;
    }
    if (curChannels.size() > 1 && roundIdx != commLoops - 1) {
        PostSyncInterThreadsPerRank(mainThreadCurRank, subThreadsCurRank);
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

HcclResult InsTempAlltoAllVMesh1D::PreCopyByLoop(const std::vector<u32> &commRanks,
    const std::map<u32, std::vector<ChannelInfo>> &channels, const std::vector<ThreadHandle> &threads,
    const TemplateDataParams &tempAlgParams, const u32 myAlgRank)
{
    for (u32 rankIdx = 0; rankIdx < commRanks.size(); rankIdx++) {
        u32 remoteRank = commRanks[rankIdx];
        u32 myRankCclBuffIdx = 0; // myRank与remoteRank交互时myRank提供的cclbuffer index
        u32 remoteCclBuffIdx = 0; // myRank与remoteRank交互时remoteRank提供的cclbuffer index
        CalcCclBuffIdx(remoteRank, myRankCclBuffIdx, remoteCclBuffIdx);
        u32 queIdx = myRankCclBuffIdx * channelsPerRank_ + 1;
        if (channels.find(remoteRank) == channels.end()) {
            HCCL_ERROR("[InsTempAlltoAllVMesh1D][PreCopy] remoteRank[%u] does not exist in channels map!",
                remoteRank);
            return HCCL_E_PARA;
        }
        const std::vector<ChannelInfo> &allChannels = channels.at(remoteRank);
        std::vector<ChannelInfo> curChannels;
        CHK_RET(SelectChannel(remoteRank, allChannels, curChannels));
        // send数据按照channel分片
        CHK_RET(CalcDataSplitByPortGroupCommon(tempAlgParams.sendCounts[remoteRank], dataTypeSize_, curChannels,
            sendCountsSplit_, sendSizeSplit_, sendOffsetSplit_, static_cast<u32>(curChannels.size())));
        for (u32 channelId = 0; channelId < curChannels.size(); channelId++) {
            if (sendSizeSplit_[channelId] > 0) {
                CHK_RET(static_cast<HcclResult>(PreCopy(tempAlgParams, threads[queIdx], myRankCclBuffIdx, remoteRank,
                    sendSizeSplit_[channelId], sendCountsSplit_[channelId], sendOffsetSplit_[channelId])));
            }
            queIdx++;
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

bool InsTempAlltoAllVMesh1D::IsVmesh4X4Mode() const
{
    return subCommRanks_.size() == 1 && subCommRanks_[0].size() == VMESH_4X4_RANK_NUM &&
        templateRankSize_ == VMESH_4X4_RANK_NUM;
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
    if (!IsVmesh4X4Mode()) {
        selectedChannels = allChannels;
        return HCCL_SUCCESS;
    }

    u32 myRankIndex = 0;
    u32 remoteRankIndex = 0;
    CHK_RET(GetRankIndexInSubComm(myRank_, myRankIndex));
    CHK_RET(GetRankIndexInSubComm(remoteRank, remoteRankIndex));

    u32 myColIdx = myRankIndex % VMESH_4X4_COL_NUM;
    u32 remoteColIdx = remoteRankIndex % VMESH_4X4_COL_NUM;
    if (myColIdx == remoteColIdx) {
        for (const auto &channel : allChannels) {
            if (channel.protocol == COMM_PROTOCOL_UBC_CTP) {
                selectedChannels.push_back(channel);
                return HCCL_SUCCESS;
            }
        }
        HCCL_ERROR("[InsTempAlltoAllVMesh1D][SelectChannel] no UBC_CTP channel for remoteRank[%u].", remoteRank);
        return HCCL_E_PARA;
    }

    u32 colDistance = (remoteColIdx + VMESH_4X4_COL_NUM - myColIdx) % VMESH_4X4_COL_NUM;
    if (colDistance == 0) {
        HCCL_ERROR("[InsTempAlltoAllVMesh1D][SelectChannel] invalid col distance, myColIdx[%u], "
            "remoteColIdx[%u], remoteRank[%u].", myColIdx, remoteColIdx, remoteRank);
        return HCCL_E_PARA;
    }
    u32 channelIndex = colDistance - 1;
    if (channelIndex >= allChannels.size()) {
        HCCL_ERROR("[InsTempAlltoAllVMesh1D][SelectChannel] invalid channel index[%u], channel size[%zu], "
            "myColIdx[%u], remoteColIdx[%u], remoteRank[%u].",
            channelIndex, allChannels.size(), myColIdx, remoteColIdx, remoteRank);
        return HCCL_E_PARA;
    }
    selectedChannels.push_back(allChannels[channelIndex]);
    return HCCL_SUCCESS;
}
} // namespace Hccl