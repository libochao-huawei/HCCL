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
    CHK_RET(CalcChannelRequestMesh1D(comm, param, topoInfo, subCommRanks_, level0Channels));
    resourceRequest.channels.push_back(level0Channels);
    resourceRequest.slaveThreadNum = level0Channels.size();
    for (u32 index = 0; index < resourceRequest.slaveThreadNum; index++) {
        resourceRequest.notifyNumPerThread.push_back(1);
    }
    resourceRequest.notifyNumOnMainThread = resourceRequest.slaveThreadNum;
    return HCCL_SUCCESS;
}

u64 InsTempAlltoAllVMesh1D::CalcScratchMultiple(BufferType inBuffType, BufferType outBuffType)
{
    // usrIn和cclBuffer大小相同
    return 1;
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

    bool isPcieProtocal = IsPcieProtocol(templateResource.channels);  // 判断是否存在pcie链路
    isDmaRead_ = isPcieProtocal;  // 是否使用Read模式
    HCCL_DEBUG("[InsTempAlltoAllVMesh1D] Use Dma Read[%d]", isDmaRead_);

    HCCL_INFO("[InsTempAlltoAllVMesh1D] Run Start");

    u32 myAlgRank = 0;
    auto iter = std::find(subCommRanks_[0].begin(), subCommRanks_[0].end(), myRank_);
    if (iter != subCommRanks_[0].end()) {
        myAlgRank = std::distance(subCommRanks_[0].begin(), iter);
    } else {
        HCCL_ERROR("[InsTempAlltoAllVMesh1D][KernelRun] subCommRanks_ or myRank_ is error.");
        return HCCL_E_INTERNAL;
    }

    if (isDmaRead_) {
        if (threadNum_ > 1) {
            std::vector<ThreadHandle> subThreads(templateResource.threads.begin() + 1, templateResource.threads.end());
            GetNotifyIdxMainToSub(notifyIdxMainToSub_);
            CHK_RET(PreSyncInterThreads(templateResource.threads[0], subThreads, notifyIdxMainToSub_));
        }
        CHK_RET(PreCopy(templateResource.channels, templateResource.threads, tempAlgParams, myAlgRank));
        if (threadNum_ > 1) {
            std::vector<ThreadHandle> subThreads(templateResource.threads.begin() + 1, templateResource.threads.end());
            GetNotifyIdxSubToMain(notifyIdxSubToMain_);
            CHK_RET(PostSyncInterThreads(templateResource.threads[0], subThreads, notifyIdxSubToMain_));
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
    } else {
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
    u32 queIdx = 0;
    for (u32 rankId = 0; rankId < templateRankSize_; rankId++) {
        if (rankId == myAlgRank) {
            // 做本卡local copy
            CHK_RET(LocalCopyForMyRank(tempAlgParams, threads[queIdx], myAlgRank, queIdx));
            queIdx++;
            continue;
        }

        u32 remoteRank = subCommRanks_[0][rankId]; // 物理rank
        if (channels.find(remoteRank) == channels.end()) {
            HCCL_ERROR("[InsTempAlltoAllVMesh1D] remoteRank[%u] does not exist in channels map!", remoteRank);
            return HCCL_E_PARA;
        }
        const std::vector<ChannelInfo> &curChannels = channels.at(remoteRank);
        channelsPerRank_ = curChannels.size();
        // send数据按照channel分片
        std::vector<u64> sendCountsSplit;
        std::vector<u64> sendSizeSplit;
        std::vector<u64> sendOffsetSplit;
        CHK_RET(CalcDataSplitByPortGroup(tempAlgParams.sendCounts[rankId], dataTypeSize_, curChannels,
            sendCountsSplit, sendSizeSplit, sendOffsetSplit));
        // recv数据按照channel分片
        std::vector<u64> recvCountsSplit;
        std::vector<u64> recvSizeSplit;
        std::vector<u64> recvOffsetSplit;
        CHK_RET(CalcDataSplitByPortGroup(tempAlgParams.recvCounts[rankId], dataTypeSize_, curChannels,
            recvCountsSplit, recvSizeSplit, recvOffsetSplit));
        for (u32 channelId = 0; channelId < curChannels.size(); channelId++) {
            const ChannelInfo &channelSend = curChannels[channelId]; // 发给哪个rank
            const ChannelInfo &channelRecv = curChannels[channelId]; // 收哪个rank的数据
            std::vector<DataSlice> txSrcSlices;
            std::vector<DataSlice> txDstSlices;
            std::vector<DataSlice> rxSrcSlices;
            std::vector<DataSlice> rxDstSlices;

            void* remoteCclBuffAddr = channelRecv.remoteCclMem.addr;
            // repeatNum为1，所以这里不考虑重复场景
            DataSlice txSrcSlice = DataSlice(tempAlgParams.buffInfo.inputPtr,
                tempAlgParams.sdispls[rankId] * dataTypeSize_ + sendOffsetSplit[channelId],
                sendSizeSplit[channelId], sendCountsSplit[channelId]);
            DataSlice txDstSlice = DataSlice(remoteCclBuffAddr,
                myAlgRank * tempAlgParams.inputSliceStride + tempAlgParams.buffInfo.hcclBuffBaseOff +
                sendOffsetSplit[channelId], sendSizeSplit[channelId], sendCountsSplit[channelId]);

            DataSlice rxSrcSlice = DataSlice(remoteCclBuffAddr,
                myAlgRank * tempAlgParams.inputSliceStride + tempAlgParams.buffInfo.hcclBuffBaseOff +
                recvOffsetSplit[channelId], recvSizeSplit[channelId], recvCountsSplit[channelId]);
            DataSlice rxDstSlice = DataSlice(tempAlgParams.buffInfo.outputPtr,
                tempAlgParams.rdispls[rankId] * dataTypeSize_ + recvOffsetSplit[channelId],
                recvSizeSplit[channelId], recvCountsSplit[channelId]);

            txSrcSlices.push_back(txSrcSlice);
            txDstSlices.push_back(txDstSlice);
            rxSrcSlices.push_back(rxSrcSlice);
            rxDstSlices.push_back(rxDstSlice);

            DataInfo sendInfo{channelSend, {txSrcSlices, txDstSlices}};
            DataInfo recvInfo{channelRecv, {rxSrcSlices, rxDstSlices}};
            SendRecvInfo sendRecvInfo{{channelSend, channelRecv},
                {{txSrcSlices, txDstSlices}, {rxSrcSlices, rxDstSlices}}};
            if (isDmaRead_) {
                if (sendSizeSplit[channelId] > 0 && recvSizeSplit[channelId] > 0) {
                    CHK_PRT_RET(SendRecvRead(sendRecvInfo, threads[queIdx]),
                        HCCL_ERROR("[InsTempAlltoAllVMesh1D] RunALLtoALL SendRecvInfo failed"),
                        HcclResult::HCCL_E_INTERNAL);
                } else { // 其中一个或者两个为0
                    if (sendSizeSplit[channelId] > 0) {
                        CHK_PRT_RET(SendRead(sendInfo, threads[queIdx]),
                            HCCL_ERROR("[InsTempAlltoAllVMesh1D] RunALLtoALL sendInfo failed"),
                            HcclResult::HCCL_E_INTERNAL);
                    } else if (recvSizeSplit[channelId] > 0) {
                        CHK_PRT_RET(RecvRead(recvInfo, threads[queIdx]),
                            HCCL_ERROR("[InsTempAlltoAllVMesh1D] RunALLtoALL recvInfo failed"),
                            HcclResult::HCCL_E_INTERNAL);
                    }
                }
            } else {
                if (sendSizeSplit[channelId] > 0 && recvSizeSplit[channelId] > 0) {
                    CHK_PRT_RET(SendRecvWrite(sendRecvInfo, threads[queIdx]),
                        HCCL_ERROR("[InsTempAlltoAllVMesh1D] RunALLtoALL SendRecvInfo failed"),
                        HcclResult::HCCL_E_INTERNAL);
                } else { // 其中一个或者两个为0
                    if (sendSizeSplit[channelId] > 0) {
                        CHK_PRT_RET(SendWrite(sendInfo, threads[queIdx]),
                            HCCL_ERROR("[InsTempAlltoAllVMesh1D] RunALLtoALL sendInfo failed"),
                            HcclResult::HCCL_E_INTERNAL);
                    }
                    if (recvSizeSplit[channelId] > 0) {
                        CHK_PRT_RET(RecvWrite(recvInfo, threads[queIdx]),
                            HCCL_ERROR("[InsTempAlltoAllVMesh1D] RunALLtoALL recvInfo failed"),
                            HcclResult::HCCL_E_INTERNAL);
                    }
                }
            }
            HCCL_DEBUG("[InsTempAlltoAllVMesh1D][RunALLtoALL] do send recv write on thread[%u], "\
                "send size[%llu], recv size[%llu], remote rank[%u].",
                queIdx, sendSizeSplit[channelId], recvSizeSplit[channelId], remoteRank);
            if (!isDmaRead_ && recvSizeSplit[channelId] > 0) {
                CHK_RET(PostCopy(tempAlgParams, threads[queIdx], rankId, recvSizeSplit[channelId],
                    recvCountsSplit[channelId], recvOffsetSplit[channelId]));
            }
            queIdx++;
        }
    }
    return HcclResult::HCCL_SUCCESS;
}

HcclResult InsTempAlltoAllVMesh1D::PreCopy(const std::map<u32, std::vector<ChannelInfo>> &channels,
    const std::vector<ThreadHandle> &threads, const TemplateDataParams &tempAlgParams,
    const u32 myAlgRank)
{
    // local copy
    u32 queIdx = 0;
    for (u32 rankId = 0; rankId < templateRankSize_; rankId++) {
        if (rankId == myAlgRank) {
            // 跳过本卡
            queIdx++;
            continue;
        }
        if (channels.find(rankId) == channels.end()) {
            HCCL_ERROR("[InsTempAlltoAllVMesh1D] remoteRank[%u] does not exist in channels map!", rankId);
            return HCCL_E_PARA;
        }
        const std::vector<ChannelInfo> &curChannels = channels.at(rankId);
        channelsPerRank_ = curChannels.size();
        // send数据按照channel分片
        std::vector<u64> sendCountsSplit;
        std::vector<u64> sendSizeSplit;
        std::vector<u64> sendOffsetSplit;
        CHK_RET(CalcDataSplitByPortGroup(tempAlgParams.sendCounts[rankId], dataTypeSize_, curChannels,
            sendCountsSplit, sendSizeSplit, sendOffsetSplit));
        // recv数据按照channel分片
        std::vector<u64> recvCountsSplit;
        std::vector<u64> recvSizeSplit;
        std::vector<u64> recvOffsetSplit;
        CHK_RET(CalcDataSplitByPortGroup(tempAlgParams.recvCounts[rankId], dataTypeSize_, curChannels,
            recvCountsSplit, recvSizeSplit, recvOffsetSplit));
        for (u32 channelId = 0; channelId < curChannels.size(); channelId++) {
            DataSlice srcSlice = DataSlice(tempAlgParams.buffInfo.inputPtr,
                tempAlgParams.sdispls[rankId] * dataTypeSize_ + sendOffsetSplit[channelId],
                sendSizeSplit[channelId], sendCountsSplit[channelId]);
            DataSlice dstSlice = DataSlice(tempAlgParams.buffInfo.hcclBuff.addr,
                rankId * tempAlgParams.inputSliceStride + tempAlgParams.buffInfo.hcclBuffBaseOff +
                sendOffsetSplit[channelId], sendSizeSplit[channelId], sendCountsSplit[channelId]);
            if (sendSizeSplit[channelId] > 0) {
                CHK_RET(static_cast<HcclResult>(LocalCopy(threads[queIdx], srcSlice, dstSlice)));
            }
            queIdx++;
        }
    }
    return HcclResult::HCCL_SUCCESS;
}

HcclResult InsTempAlltoAllVMesh1D::PostCopy(const TemplateDataParams &tempAlgParams, const ThreadHandle &thread,
    const u32 rankId, const u64 &recvSize, const u64 &recvCount, const u64 &recvOffset) const
{
    // ccl buffer的数据搬运到usrout
    DataSlice localCopySrcSlice = DataSlice(tempAlgParams.buffInfo.hcclBuff.addr,
        rankId * tempAlgParams.inputSliceStride + tempAlgParams.buffInfo.hcclBuffBaseOff + recvOffset,
        recvSize, recvCount);
    DataSlice localCopyDstSlice = DataSlice(tempAlgParams.buffInfo.outputPtr,
        tempAlgParams.rdispls[rankId] * dataTypeSize_ + recvOffset,
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