/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "ins_temp_all_gather_symmetry_memory_mesh_1D.h"
#include "alg_data_trans_wrapper.h"
#include "template_utils.h"
namespace ops_hccl {
InsTempAllGatherSymmetryMemoryMesh1D::InsTempAllGatherSymmetryMemoryMesh1D(const OpParam &param, const u32 rankId,
                                               const std::vector<std::vector<u32>> &subCommRanks)
    : InsAlgTemplateBase(param, rankId, subCommRanks)
{
}
InsTempAllGatherSymmetryMemoryMesh1D::~InsTempAllGatherSymmetryMemoryMesh1D() {}

HcclResult InsTempAllGatherSymmetryMemoryMesh1D::CalcRes(HcclComm comm, const OpParam &param, const TopoInfoWithNetLayerDetails *topoInfo,
                                           AlgResourceRequest &resourceRequest)
{
    HCCL_INFO("[InsTempAllGatherSymmetryMemoryMesh1D][CalcRes] start");
    GetRes(resourceRequest);
    std::vector<HcclChannelDesc> level0Channels;
    CHK_RET(CalcChannelRequestMesh1D(comm, param, topoInfo, subCommRanks_, level0Channels));
    resourceRequest.channels.push_back(level0Channels);
    return HCCL_SUCCESS;
}
HcclResult InsTempAllGatherSymmetryMemoryMesh1D::GetRes(AlgResourceRequest &resourceRequest) const
{
    u32 level0RankSize = templateRankSize_;
    u32 threadNum = level0RankSize > 1 ? level0RankSize - 1 : 1;
    resourceRequest.slaveThreadNum = threadNum - 1;
    resourceRequest.notifyNumPerThread.assign(resourceRequest.slaveThreadNum, 1);
    resourceRequest.notifyNumOnMainThread = threadNum - 1;
    return HCCL_SUCCESS;
}

u64 InsTempAllGatherSymmetryMemoryMesh1D::GetThreadNum() const
{
    return templateRankSize_ > 1 ? templateRankSize_ - 1 : 1;
}

u64 InsTempAllGatherSymmetryMemoryMesh1D::CalcScratchMultiple(BufferType inBuffType, BufferType outBuffType)
{
    (void)inBuffType;
    (void)outBuffType;
    u64 scratchMultiple = 0;
    if (opMode_ == OpMode::OPBASE){
        scratchMultiple = templateRankSize_;
    }
    return scratchMultiple;
}

HcclResult InsTempAllGatherSymmetryMemoryMesh1D::KernelRun(const OpParam &param, const TemplateDataParams &tempAlgParams,
                                             TemplateResource &templateResource)
{
    enableRemoteMemAccess_ = tempAlgParams.enableRemoteMemAccess;
    HCCL_INFO("[InsTempAllGatherSymmetryMemoryMesh1D] Run start");
    if (tempAlgParams.sliceSize == 0) {
        HCCL_INFO("[InsTempAllGatherSymmetryMemoryMesh1D] Rank [%d], get slicesize zero.", myRank_);
        return HCCL_SUCCESS;
    }
    outSymWinHandle_ = param.outSymWinHandle;
    outOffset_ = param.outOffset;
    threadNum_ = templateResource.threads.size();
    tempAlgParams_ = tempAlgParams;
    dataType_ = param.DataDes.dataType;
    HCCL_DEBUG("[InsTempAllGatherSymmetryMemoryMesh1D] Rank [%d], get threadNum_[%d].", myRank_, threadNum_);
    CHK_RET(LocalDataCopy(templateResource.threads));
    if (templateRankSize_ == 1) {
        return HcclResult::HCCL_SUCCESS;
    }
    if (threadNum_ > 1) {
        std::vector<ThreadHandle> subThreads(templateResource.threads.begin() + 1, templateResource.threads.end());
        GetNotifyIdxMainToSub(notifyIdxMainToSub_);
        CHK_RET(PreSyncInterThreads(templateResource.threads[0], subThreads, notifyIdxMainToSub_));
    }

    CHK_RET(RunAllGatherMesh(templateResource.threads, templateResource.channels));

    if (threadNum_ > 1) {
        std::vector<ThreadHandle> subThreads(templateResource.threads.begin() + 1, templateResource.threads.end());
        GetNotifyIdxSubToMain(notifyIdxSubToMain_);
        CHK_RET(PostSyncInterThreads(templateResource.threads[0], subThreads, notifyIdxSubToMain_));
    }
    HCCL_INFO("[InsTempAllGatherSymmetryMemoryMesh1D] Run End");
    return HcclResult::HCCL_SUCCESS;
}

// 从outputPtr的对应位置（LocalDataCopy拷贝到的）将本端数据发送到所有对端
HcclResult InsTempAllGatherSymmetryMemoryMesh1D::RunAllGatherMesh(const std::vector<ThreadHandle> &threads,
                                                    const std::map<u32, std::vector<ChannelInfo>> &channels)
{
    HCCL_INFO("[InsTempAllGatherSymmetryMemoryMesh1D] RunAllGatherMesh RankIDs[%d].", myRank_);
    const u32 dataTypeSize = DATATYPE_SIZE_TABLE[dataType_];
    for (u32 threadIdx = 0; threadIdx < subCommRanks_[0].size() - 1; threadIdx++) {
        // 对端rank
        u32 connectedRank = subCommRanks_[0][(myRank_ + 1 + threadIdx) % subCommRanks_[0].size()];
        void* remotePtr;

        HCCL_INFO("[InsTempAllGatherSymmetryMemoryMesh1D] RunAllGatherMesh RankIDs[%d], connectedRank[%d].",
                    myRank_, connectedRank);

        CHK_PRT_RET(threadIdx >= threads.size() || channels.count(connectedRank) == 0 ||
                    channels.at(connectedRank).empty(),
                    HCCL_ERROR("[InsTempAllGatherSymmetryMemoryMesh1D][RankID]=%u threadIdx=%u, threads.size=%u, "
                                "connectedRank=%d, channels.size=%u",
                                myRank_, threadIdx, threads.size(), connectedRank, channels.size()),
                    HcclResult::HCCL_E_INTERNAL);

        const ChannelInfo &linkRemote = channels.at(connectedRank)[0];
        HcclResult ret = HcclSymWinGetPeerPointer(outSymWinHandle_, outOffset_, connectedRank, &remotePtr);
        if (ret != HCCL_SUCCESS)
        {
            HCCL_ERROR("[InsTempAllGatherSymmetryMemoryMesh1D] HcclSymWinGetPeerPointer failed", HcclResult::HCCL_E_INTERNAL);
        }

        std::vector<DataSlice> txSrcSlicesAll;
        std::vector<DataSlice> txDstSlicesAll;
        std::vector<DataSlice> rxDstSlicesAll;
        std::vector<DataSlice> rxSrcSlicesAll;

        for (u32 rpt = 0; rpt < tempAlgParams_.repeatNum; ++rpt) {
            const u64 outBaseOff = tempAlgParams_.buffInfo.outBuffBaseOff + rpt * tempAlgParams_.outputRepeatStride;
            const u64 scratchRepeatStride = tempAlgParams_.sliceSize * templateRankSize_;
            const u64 scratchBase = tempAlgParams_.buffInfo.hcclBuffBaseOff + rpt * scratchRepeatStride;

            u64 sliceSize = tempAlgParams_.sliceSize;
            if (tempAlgParams_.tailSize != 0 && connectedRank == templateRankSize_ - 1) {
                sliceSize = tempAlgParams_.tailSize;
            }

            u64 txSrcDstOffset = tempAlgParams_.outputSliceStride * myRank_ + outBaseOff;
            u64 rxSrcDstOffset = tempAlgParams_.outputSliceStride * connectedRank + outBaseOff;

            void *txSrcPtr = tempAlgParams_.buffInfo.outputPtr; // 本端发送数据起始地址
            void *txDstPtr = remotePtr; // 对端接收数据起始地址
            void *rxSrcPtr = remotePtr; // 对端发送数据起始地址
            void *rxDstPtr = tempAlgParams_.buffInfo.outputPtr; // 本端接收数据起始地址

            u64 sliceCount = sliceSize / dataTypeSize;

            txSrcSlicesAll.emplace_back(txSrcPtr, txSrcDstOffset, sliceSize, sliceCount); // 就是LocalDataCopy拷贝到的outputPtr地址。
            txDstSlicesAll.emplace_back(txDstPtr, txSrcDstOffset, sliceSize, sliceCount);
            rxDstSlicesAll.emplace_back(rxDstPtr, rxSrcDstOffset, sliceSize, sliceCount);
            rxSrcSlicesAll.emplace_back(rxSrcPtr, rxSrcDstOffset, sliceSize, sliceCount);

            HCCL_DEBUG("[InsTempAllGatherSymmetryMemoryMesh1D][RunAllGatherMesh] rankId [%d] connectedRank [%d] rpt [%d] txSrcDstSlices: "
                        "offset[%d] sliceSize[%d] count[%d].",
                        myRank_, connectedRank, rpt, txSrcDstOffset, sliceSize, sliceCount);

            HCCL_DEBUG("[InsTempAllGatherSymmetryMemoryMesh1D][RunAllGatherMesh] rankId [%d] connectedRank [%d] rpt [%d] rxSrcDstSlices: "
                        "offset[%d] sliceSize[%d] count[%d].",
                        myRank_, connectedRank, rpt, rxSrcDstOffset, sliceSize, sliceCount);

        }

        TxRxSlicesList sendRecvSlicesList({txSrcSlicesAll, txDstSlicesAll}, {rxSrcSlicesAll, rxDstSlicesAll});
        TxRxChannels sendRecvChannels(linkRemote, linkRemote);
        SendRecvInfo sendRecvInfo(sendRecvChannels, sendRecvSlicesList);
        CHK_PRT_RET(SendRecvRead(sendRecvInfo, threads[threadIdx]),
                    HCCL_ERROR("[InsTempAllGatherSymmetryMemoryMesh1D] RunAllGather Send failed"), HcclResult::HCCL_E_INTERNAL);

        }
    return HcclResult::HCCL_SUCCESS;
}

// 将inputPtr的发送数据通过LocalCopy拷贝到outputPtr的对应位置。
HcclResult InsTempAllGatherSymmetryMemoryMesh1D::LocalDataCopy(const std::vector<ThreadHandle> &threads)
{
    HCCL_INFO("[InsTempAllGatherSymmetryMemoryMesh1D] LocalDataCopy.");
    if (threads.empty()) {
        return HcclResult::HCCL_E_INTERNAL;
    }

    const u32 dataTypeSize = DATATYPE_SIZE_TABLE[dataType_];
    u64 sliceSize = tempAlgParams_.sliceSize;

    u64 sliceCount = sliceSize / dataTypeSize;
    for (u32 rpt = 0; rpt < tempAlgParams_.repeatNum; ++rpt) {
        // 要发送数据的起始偏移
        const u64 inBaseOff = tempAlgParams_.buffInfo.inBuffBaseOff + rpt * tempAlgParams_.inputRepeatStride;
        const u64 outBaseOff = tempAlgParams_.buffInfo.outBuffBaseOff + rpt * tempAlgParams_.outputRepeatStride;
        const u64 inOff = tempAlgParams_.inputSliceStride * myRank_ + inBaseOff;
        const u64 outOff = tempAlgParams_.outputSliceStride * myRank_ + outBaseOff;

        // 整个要发送的数据都拷贝到outputptr对应的myRank_位置
        DataSlice srcSlice(tempAlgParams_.buffInfo.inputPtr, inOff, sliceSize, sliceCount);
        bool skipOutCopy = (tempAlgParams_.buffInfo.inputPtr == tempAlgParams_.buffInfo.outputPtr && inOff == outOff);
        if (!skipOutCopy) {
            DataSlice dstSlice(tempAlgParams_.buffInfo.outputPtr, outOff, sliceSize, sliceCount);
            HCCL_DEBUG("[InsTempAllGatherSymmetryMemoryMesh1D][LocalDataCopy] RankID [%d] srcSlice: inBaseOff[%llu] inOff[%llu] "
                       "sliceSize[%llu] count[%llu].",
                       myRank_, inBaseOff, inOff, sliceSize, sliceCount);
            HCCL_DEBUG("[InsTempAllGatherSymmetryMemoryMesh1D][LocalDataCopy] RankID [%d] dstSlice: outBaseoff[%llu] "
                       "outOff[%llu] sliceSize[%llu] count[%llu].",
                       myRank_, outBaseOff, outOff, sliceSize, sliceCount);
            LocalCopy(threads[0], srcSlice, dstSlice);
        }
    }
    return HcclResult::HCCL_SUCCESS;
}

void InsTempAllGatherSymmetryMemoryMesh1D::GetNotifyIdxMainToSub(std::vector<u32> &notifyIdxMianToSub)
{
    notifyIdxMianToSub.clear();
    u32 threadNum = GetThreadNum();
    u32 slaveThreadNum = threadNum - 1;
    for (u32 slaveThreadIdx = 0; slaveThreadIdx < slaveThreadNum; slaveThreadIdx++) {
        notifyIdxMianToSub.push_back(0);
    }
}

void InsTempAllGatherSymmetryMemoryMesh1D::GetNotifyIdxSubToMain(std::vector<u32> &notifyIdxSubToMain)
{
    notifyIdxSubToMain.clear();
    u32 threadNum = GetThreadNum();
    u32 notifyNum = threadNum - 1;
    for (u32 notifyIdx = 0; notifyIdx < notifyNum; notifyIdx++) {
        notifyIdxSubToMain.push_back(notifyIdx);
    }
}

}  // namespace ops_hccl