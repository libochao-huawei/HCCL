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
            const ChannelInfo &channelSend = channels.at(remoteRank)[0]; // 发给哪个rank
            const ChannelInfo &channelRecv = channels.at(remoteRank)[0]; // 收哪个rank的数据
            std::vector<DataSlice> txSrcSlices; // 在read模式下用不到txSlice，直接给空的
            std::vector<DataSlice> txDstSlices;
            std::vector<DataSlice> rxSrcSlices;
            std::vector<DataSlice> rxDstSlices;

            void* remoteCclBuffAddr = channelRecv.remoteCclMem.addr;
            // repeatNum为1，所以这里不考虑重复场景
            DataSlice rxSrcSlice = DataSlice(remoteCclBuffAddr,
                myAlgRank * tempAlgParams.inputSliceStride + tempAlgParams.buffInfo.hcclBuffBaseOff +
                recvOffsetSplit[channelId], recvSizeSplit[channelId], recvCountsSplit[channelId]);
            DataSlice rxDstSlice = DataSlice(tempAlgParams.buffInfo.outputPtr,
                tempAlgParams.rdispls[rankId] * dataTypeSize_,
                recvSizeSplit[channelId], recvCountsSplit[channelId]);

            rxSrcSlices.push_back(rxSrcSlice);
            rxDstSlices.push_back(rxDstSlice);

            // 先做前拷贝
            if (sendSizeSplit[channelId] > 0) {
                CHK_RET(PreCopy(tempAlgParams, threads[queIdx], rankId, sendSizeSplit[channelId],
                    sendCountsSplit[channelId], sendOffsetSplit[channelId]));
            }
            if (sendSizeSplit[channelId] > 0 && recvSizeSplit[channelId] > 0) {
                SendRecvInfo sendRecvInfo{{channelSend, channelRecv},
                    {{txSrcSlices, txDstSlices}, {rxSrcSlices, rxDstSlices}}};
                CHK_PRT_RET(SendRecvRead(sendRecvInfo, threads[queIdx]),
                    HCCL_ERROR("[InsTempAlltoAllVMesh1D] RunALLtoALL SendRecvInfo failed"),
                    HcclResult::HCCL_E_INTERNAL);
            } else { // 其中一个或者两个为0
                if (sendSizeSplit[channelId] > 0) {
                    DataInfo sendInfo{channelSend, {txSrcSlices, txDstSlices}};
                    CHK_PRT_RET(SendRead(sendInfo, threads[queIdx]),
                        HCCL_ERROR("[InsTempAlltoAllVMesh1D] RunALLtoALL sendInfo failed"),
                        HcclResult::HCCL_E_INTERNAL);
                } else if (recvSizeSplit[channelId] > 0) {
                    DataInfo recvInfo{channelRecv, {rxSrcSlices, rxDstSlices}};
                    CHK_PRT_RET(RecvRead(recvInfo, threads[queIdx]),
                        HCCL_ERROR("[InsTempAlltoAllVMesh1D] RunALLtoALL recvInfo failed"),
                        HcclResult::HCCL_E_INTERNAL);
                }
            }
            HCCL_DEBUG("[InsTempAlltoAllVMesh1D][RunALLtoALL] do send recv write on thread[%u], "\
                "send size[%llu], recv size[%llu], remote rank[%u].",
                queIdx, sendSizeSplit[channelId], recvSizeSplit[channelId], remoteRank);
            queIdx++;
        }
    }
    return HcclResult::HCCL_SUCCESS;
}

HcclResult InsTempAlltoAllVMesh1D::PreCopy(const TemplateDataParams &tempAlgParams, const ThreadHandle &thread,
    const u32 rankId, const u64 &sendSize, const u64 &sendCount, const u64 &sendOffset) const
{
    // local copy
    DataSlice srcSlice = DataSlice(tempAlgParams.buffInfo.inputPtr,
        tempAlgParams.sdispls[rankId] * dataTypeSize_ + sendOffset, sendSize, sendCount);
    DataSlice dstSlice = DataSlice(tempAlgParams.buffInfo.hcclBuff.addr,
        rankId * tempAlgParams.inputSliceStride + tempAlgParams.buffInfo.hcclBuffBaseOff + sendOffset,
        sendSize, sendCount);
    CHK_RET(static_cast<HcclResult>(LocalCopy(thread, srcSlice, dstSlice)));
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