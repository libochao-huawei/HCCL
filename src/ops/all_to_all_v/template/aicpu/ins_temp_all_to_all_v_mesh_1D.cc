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
    HCCL_WARNING("Resource calculation is temporarily not performed in the template.");
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
    cclBufferCountPerRank_ = tempAlgParams.inputSliceStride / dataTypeSize_;
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
        const std::vector<ChannelInfo> &curChannels = channels.at(remoteRank);
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
            const ChannelInfo &linkSend = curChannels[channelId]; // 发给哪个rank
            const ChannelInfo &linkRecv = curChannels[channelId]; // 收哪个rank的数据
            std::vector<DataSlice> txSrcSlices;
            std::vector<DataSlice> txDstSlices;
            std::vector<DataSlice> rxSrcSlices; // 在write模式下用不到rxSlice，直接给空的
            std::vector<DataSlice> rxDstSlices;

            void* remoteCclBuffAddr = linkSend.remoteCclMem.addr;
            // repeatNum为1，所以这里不考虑重复场景
            DataSlice txSrcSlice = DataSlice(tempAlgParams.buffInfo.inputPtr,
                tempAlgParams.sdispls[rankId] * dataTypeSize_ + sendOffsetSplit[channelId],
                sendSizeSplit[channelId], sendCountsSplit[channelId]);
            DataSlice txDstSlice = DataSlice(remoteCclBuffAddr,
                myAlgRank * cclBufferCountPerRank_ * dataTypeSize_ + sendOffsetSplit[channelId] + tempAlgParams.buffInfo.hcclBuffBaseOff,
                sendSizeSplit[channelId], sendCountsSplit[channelId]);
            txSrcSlices.push_back(txSrcSlice);
            txDstSlices.push_back(txDstSlice);

            // 不用SendRecvWrite接口里面，因为recv 0 也会去等
            DataInfo sendInfo{linkSend, {txSrcSlices, txDstSlices}};
            DataInfo recvInfo{linkRecv, {rxSrcSlices, rxDstSlices}};
            SendRecvInfo sendRecvInfo{{linkSend, linkRecv},
                                {{txSrcSlices, txDstSlices},{rxSrcSlices, rxDstSlices}}};
            if (sendSizeSplit[channelId] > 0 && recvSizeSplit[channelId] > 0) {
                CHK_PRT_RET(SendRecvWrite(sendRecvInfo, threads[queIdx]),
                    HCCL_ERROR("[InsTempAlltoAllVMesh1D] RunALLtoALL SendRecvInfo failed"), HcclResult::HCCL_E_INTERNAL);
            } else { // 其中一个或者两个为0
                if (sendSizeSplit[channelId] > 0) {
                    CHK_PRT_RET(SendWrite(sendInfo, threads[queIdx]),
                        HCCL_ERROR("[InsTempAlltoAllVMesh1D] RunALLtoALL sendInfo failed"), HcclResult::HCCL_E_INTERNAL);
                }
                if (recvSizeSplit[channelId] > 0) {
                    CHK_PRT_RET(RecvWrite(recvInfo, threads[queIdx]),
                        HCCL_ERROR("[InsTempAlltoAllVMesh1D] RunALLtoALL recvInfo failed"), HcclResult::HCCL_E_INTERNAL);
                }
            }
            HCCL_DEBUG("[InsTempAlltoAllVMesh1D][RunALLtoALL] do send recv write on thread[%u], send size[%llu], "\
                "recv size[%llu], remote rank[%u].", queIdx, sendSizeSplit[channelId], recvSizeSplit[channelId], remoteRank);
            if (recvSizeSplit[channelId] > 0) {
                CHK_RET(PostCopy(tempAlgParams, threads[queIdx], rankId, recvSizeSplit[channelId],
                    recvCountsSplit[channelId], recvOffsetSplit[channelId]));
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
        rankId * cclBufferCountPerRank_ * dataTypeSize_ + tempAlgParams.buffInfo.hcclBuffBaseOff + recvOffset,
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