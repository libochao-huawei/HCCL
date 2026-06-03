/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "ins_temp_all_gather_mesh_1D.h"
#include "alg_data_trans_wrapper.h"
#include "template_utils.h"
namespace ops_hccl {
InsTempAllGatherMesh1D::InsTempAllGatherMesh1D(const OpParam &param, const u32 rankId,
                                               const std::vector<std::vector<u32>> &subCommRanks)
    : InsAlgTemplateBase(param, rankId, subCommRanks)
{
}
InsTempAllGatherMesh1D::~InsTempAllGatherMesh1D() {}

HcclResult InsTempAllGatherMesh1D::CalcRes(HcclComm comm, const OpParam &param, const TopoInfoWithNetLayerDetails *topoInfo,
                                           AlgResourceRequest &resourceRequest)
{
    HCCL_INFO("[InsTempAllGatherMesh1D][CalcRes] start");
    enableSymmetryMemory_ = param.enableSymmetryMemory;
    GetRes(resourceRequest);
    std::vector<HcclChannelDesc> level0Channels;
    CHK_RET(CalcChannelRequestMesh1D(comm, param, topoInfo, subCommRanks_, level0Channels));
    resourceRequest.channels.push_back(level0Channels);
    return HCCL_SUCCESS;
}
HcclResult InsTempAllGatherMesh1D::GetRes(AlgResourceRequest &resourceRequest) const
{
    u32 level0RankSize = templateRankSize_;
    u32 threadNum = level0RankSize > 1 ? level0RankSize - 1 : 1;
    u32 slaveThreadNum = threadNum - 1;
    // 开启对称内存时，由于不需要等主线程完成本地copy就可以开始通信，因此通信时需要多一个线程并发。
    if (enableSymmetryMemory_ && level0RankSize > 1) {
        slaveThreadNum += 1;
    }
    resourceRequest.slaveThreadNum = slaveThreadNum;
    resourceRequest.notifyNumPerThread.assign(resourceRequest.slaveThreadNum, 1);
    resourceRequest.notifyNumOnMainThread = slaveThreadNum;
    return HCCL_SUCCESS;
}

u64 InsTempAllGatherMesh1D::GetThreadNum() const
{
    u64 threadNum = templateRankSize_ > 1 ? templateRankSize_ - 1 : 1;
    if (enableSymmetryMemory_ && templateRankSize_ > 1) {
        return threadNum + 1;
    }
    return threadNum;
}

u64 InsTempAllGatherMesh1D::CalcScratchMultiple(BufferType inBuffType, BufferType outBuffType)
{
    (void)inBuffType;
    (void)outBuffType;
    u64 scratchMultiple = 0;
    if (opMode_ == OpMode::OPBASE){
        scratchMultiple = templateRankSize_;
    }
    return scratchMultiple;
}

HcclResult InsTempAllGatherMesh1D::KernelRun(const OpParam &param, const TemplateDataParams &tempAlgParams,
                                             TemplateResource &templateResource)
{
    enableRemoteMemAccess_ = tempAlgParams.enableRemoteMemAccess;
    HCCL_INFO("[InsTempAllGatherMesh1D] Run start");
    if (tempAlgParams.sliceSize == 0 && tempAlgParams.tailSize ==0) {
        HCCL_INFO("[InsTempAllGatherMesh1D] Rank [%d], get slicesize zero.", myRank_);
        return HCCL_SUCCESS;
    }
    inSymWinHandle_ = param.inSymWinHandle;
    outSymWinHandle_ = param.outSymWinHandle;
    inOffset_ = param.inOffset;
    outOffset_ = param.outOffset;
    enableSymmetryMemory_ = param.enableSymmetryMemory;
    threadNum_ = templateResource.threads.size();
    tempAlgParams_ = tempAlgParams;
    dataType_ = param.DataDes.dataType;
    HCCL_DEBUG("[InsTempAllGatherMesh1D] Rank [%d], get threadNum_[%d].", myRank_, threadNum_);
    CHK_RET(LocalDataCopy(templateResource.threads));
    if (templateRankSize_ == 1) {
        return HcclResult::HCCL_SUCCESS;
    }

    // 非对称内存场景需要在通信前做主/从线程同步。等待主线程完成本地copy后从线程才开始通信
    if (!enableSymmetryMemory_ && threadNum_ > 1) {
        std::vector<ThreadHandle> subThreads(templateResource.threads.begin() + 1, templateResource.threads.end());
        GetNotifyIdxMainToSub(notifyIdxMainToSub_);
        CHK_RET(PreSyncInterThreads(templateResource.threads[0], subThreads, notifyIdxMainToSub_));
    }

    // 对称内存下，主线程将数据拷贝到output和从线程从其他rank的input接收数据到本端output可以并发进行。
    // 因此通信和本地copy的先后顺序不确定，不需要前同步，直接进入通信流程。
    CHK_RET(RunAllGatherMesh(templateResource.threads, templateResource.channels));

    // 通信完成后主线程需要等从线程完成本地copy才算真正完成，因此需要后同步。
    if (threadNum_ > 1) {
        std::vector<ThreadHandle> subThreads(templateResource.threads.begin() + 1, templateResource.threads.end());
        GetNotifyIdxSubToMain(notifyIdxSubToMain_);
        CHK_RET(PostSyncInterThreads(templateResource.threads[0], subThreads, notifyIdxSubToMain_));
    }
    HCCL_INFO("[InsTempAllGatherMesh1D] Run End");
    return HcclResult::HCCL_SUCCESS;
}

HcclResult InsTempAllGatherMesh1D::RunAllGatherMesh(const std::vector<ThreadHandle> &threads,
                                                    const std::map<u32, std::vector<ChannelInfo>> &channels)
{
    HCCL_INFO("[InsTempAllGatherMesh1D] RunAllGatherMesh RankIDs[%d].", myRank_);

    u32 myAlgRank = 0;
    CHK_RET(GetAlgRank(myRank_, subCommRanks_[0], myAlgRank));
    const u32 dataTypeSize = DATATYPE_SIZE_TABLE[dataType_];
    for (u32 threadIdx = 0; threadIdx < subCommRanks_[0].size() - 1; threadIdx++) {
        u32 connectedRank = subCommRanks_[0][(myAlgRank + 1 + threadIdx) % subCommRanks_[0].size()];

        u32 connectedAlgRank = 0;
        CHK_RET(GetAlgRank(connectedRank, subCommRanks_[0], connectedAlgRank));
        HCCL_INFO("[InsTempAllGatherMesh1D] RunAllGatherMesh RankIDs[%d], connectedRank[%d], connectedAlgRank[%d].",
                    myRank_, connectedRank, connectedAlgRank);

        CHK_PRT_RET(threadIdx >= threads.size() || channels.count(connectedRank) == 0 ||
                    channels.at(connectedRank).empty(),
                    HCCL_ERROR("[InsTempAllGatherMesh1D][RankID]=%u threadIdx=%u, threads.size=%u, "
                                "connectedRank=%d, channels.size=%u",
                                myRank_, threadIdx, threads.size(), connectedRank, channels.size()),
                    HcclResult::HCCL_E_INTERNAL);

        const ChannelInfo &linkRemote = channels.at(connectedRank)[0];
        void *remoteCclBuffAddr = linkRemote.remoteCclMem.addr;
        
        // 对称内存下，远端地址需要通过HcclSymWinGetPeerPointer获取
        // 直接从远端input开始位置，大小为sliceSize大小的内存区域作为接收数据源地址
        void *remoteInSymmtryPtr = nullptr;
        void *remoteOutSymmtryPtr = nullptr;
        if (enableSymmetryMemory_) {
            HcclResult inRet = HcclSymWinGetPeerPointer(inSymWinHandle_, inOffset_, connectedRank, &remoteInSymmtryPtr);
            HcclResult outRet = HcclSymWinGetPeerPointer(outSymWinHandle_, outOffset_, connectedRank, &remoteOutSymmtryPtr);
            if (inRet != HCCL_SUCCESS || outRet != HCCL_SUCCESS)
            {
                HCCL_ERROR("[InsTempAllGatherSymmetryMemoryMesh1D] HcclSymWinGetPeerPointer failed, inRet=%d, outRet=%d", inRet, outRet);
                return HcclResult::HCCL_E_INTERNAL;
            }
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
            if (tempAlgParams_.tailSize != 0 && connectedAlgRank == templateRankSize_ - 1) {
                sliceSize = tempAlgParams_.tailSize;
            }
            u64 txOutOffset = 0;
            u64 txDstOffset = 0;
            u64 rxOutOffset = 0;
            u64 rxSrcOffset = 0;
            void *txSrcPtr = enableSymmetryMemory_ ? tempAlgParams_.buffInfo.inputPtr : tempAlgParams_.buffInfo.outputPtr;
            void *txDstPtr = nullptr;
            void *rxSrcPtr = nullptr;
            void *rxDstPtr = tempAlgParams_.buffInfo.outputPtr;

            if (!enableSymmetryMemory_) {
                txOutOffset = tempAlgParams_.outputSliceStride * myAlgRank + outBaseOff;
                u64 txScratchOffset = scratchBase + tempAlgParams_.sliceSize * myAlgRank;
                txDstOffset = (!enableRemoteMemAccess_) ? txScratchOffset : txOutOffset;

                rxOutOffset = tempAlgParams_.outputSliceStride * connectedAlgRank + outBaseOff;
                u64 rxScratchOffset = scratchBase + tempAlgParams_.sliceSize * connectedAlgRank;
                rxSrcOffset = (!enableRemoteMemAccess_) ? rxScratchOffset : rxOutOffset;

                txDstPtr = (!enableRemoteMemAccess_) ? remoteCclBuffAddr : linkRemote.remoteOutputGraphMode.addr;
                rxSrcPtr = (!enableRemoteMemAccess_) ? remoteCclBuffAddr : linkRemote.remoteOutputGraphMode.addr;
            } else {
                // 对称内存下，读取远端rank的input不需要偏移，写入本端output才需要偏移
                txDstOffset = tempAlgParams_.outputSliceStride * myAlgRank + outBaseOff;
                rxOutOffset = tempAlgParams_.outputSliceStride * connectedAlgRank + outBaseOff;

                txDstPtr = remoteOutSymmtryPtr;
                rxSrcPtr = remoteInSymmtryPtr;
            }
            
            u64 sliceCount = sliceSize / dataTypeSize;

            txSrcSlicesAll.emplace_back(txSrcPtr, txOutOffset, sliceSize, sliceCount);
            txDstSlicesAll.emplace_back(txDstPtr, txDstOffset, sliceSize, sliceCount);
            rxDstSlicesAll.emplace_back(rxDstPtr, rxOutOffset, sliceSize, sliceCount);
            rxSrcSlicesAll.emplace_back(rxSrcPtr, rxSrcOffset, sliceSize, sliceCount);

            HCCL_DEBUG("[InsTempAllGatherMesh1D][RunAllGatherMesh] rankId [%d] connectedRank [%d] rpt [%d] txSrcSlices: "
                        "offset[%d] sliceSize[%d] count[%d].",
                        myRank_, connectedRank, rpt, txOutOffset, sliceSize, sliceCount);

            HCCL_DEBUG("[InsTempAllGatherMesh1D][RunAllGatherMesh] rankId [%d] connectedRank [%d] rpt [%d] txDstSlices: "
                        "offset[%d] sliceSize[%d] count[%d].",
                        myRank_, connectedRank, rpt, txDstOffset, sliceSize, sliceCount);

            HCCL_DEBUG("[InsTempAllGatherMesh1D][RunAllGatherMesh] rankId [%d] connectedRank [%d] rpt [%d] rxSrcSlices: "
                        "offset[%d] sliceSize[%d] count[%d].",
                        myRank_, connectedRank, rpt, rxOutOffset, sliceSize, sliceCount);

            HCCL_DEBUG("[InsTempAllGatherMesh1D][RunAllGatherMesh] rankId [%d] connectedRank [%d] rpt [%d] rxDrcSlices: "
                        "offset[%d] sliceSize[%d] count[%d].",
                        myRank_, connectedRank, rpt, rxSrcOffset, sliceSize, sliceCount);
        }

        TxRxSlicesList sendRecvSlicesList({txSrcSlicesAll, txDstSlicesAll}, {rxSrcSlicesAll, rxDstSlicesAll});
        TxRxChannels sendRecvChannels(linkRemote, linkRemote);
        SendRecvInfo sendRecvInfo(sendRecvChannels, sendRecvSlicesList);
        CHK_PRT_RET(SendRecvRead(sendRecvInfo, enableSymmetryMemory_ ? threads[threadIdx + 1] : threads[threadIdx]),
                    HCCL_ERROR("[InsTempAllGatherMesh1D] RunAllGather Send failed"), HcclResult::HCCL_E_INTERNAL);
        }
    return HcclResult::HCCL_SUCCESS;
}

HcclResult InsTempAllGatherMesh1D::LocalDataCopy(const std::vector<ThreadHandle> &threads)
{
    HCCL_INFO("[InsTempAllGatherMesh1D] LocalDataCopy.");
    if (threads.empty()) {
        return HcclResult::HCCL_E_INTERNAL;
    }

    u32 myAlgRank;
    CHK_RET(GetAlgRank(myRank_, subCommRanks_[0], myAlgRank));
    const u32 dataTypeSize = DATATYPE_SIZE_TABLE[dataType_];
    u64 sliceSize = tempAlgParams_.sliceSize;

    if (tempAlgParams_.tailSize !=0 && myAlgRank == templateRankSize_ -1) {
        sliceSize = tempAlgParams_.tailSize;
    }

    u64 sliceCount = sliceSize / dataTypeSize;
    for (u32 rpt = 0; rpt < tempAlgParams_.repeatNum; ++rpt) {
        const u64 inBaseOff = tempAlgParams_.buffInfo.inBuffBaseOff + rpt * tempAlgParams_.inputRepeatStride;
        const u64 outBaseOff = tempAlgParams_.buffInfo.outBuffBaseOff + rpt * tempAlgParams_.outputRepeatStride;
        const u64 inOff = tempAlgParams_.inputSliceStride * myAlgRank + inBaseOff;
        const u64 outOff = tempAlgParams_.outputSliceStride * myAlgRank + outBaseOff;

        DataSlice srcSlice(tempAlgParams_.buffInfo.inputPtr, inOff, sliceSize, sliceCount);
        bool skipOutCopy = (tempAlgParams_.buffInfo.inputPtr == tempAlgParams_.buffInfo.outputPtr && inOff == outOff);

        if (!skipOutCopy) {
            DataSlice dstSlice(tempAlgParams_.buffInfo.outputPtr, outOff, sliceSize, sliceCount);
            HCCL_DEBUG("[InsTempAllGatherMesh1D][LocalDataCopy] RankID [%d] AlgRank [%d] srcSlice: inBaseOff[%llu] inOff[%llu] "
                       "sliceSize[%llu] count[%llu].",
                       myRank_, myAlgRank, inBaseOff, inOff, sliceSize, sliceCount);
            HCCL_DEBUG("[InsTempAllGatherMesh1D][LocalDataCopy] RankID [%d] AlgRank [%d] dstSlice: outBaseoff[%llu] "
                       "outOff[%llu] sliceSize[%llu] count[%llu].",
                       myRank_, myAlgRank, outBaseOff, outOff, sliceSize, sliceCount);
            LocalCopy(threads[0], srcSlice, dstSlice);
        }

        if (!enableRemoteMemAccess_) {
            const u64 scratchRepeatStride = tempAlgParams_.sliceSize * templateRankSize_;
            const u64 cclBaseOff = tempAlgParams_.buffInfo.hcclBuffBaseOff + rpt * scratchRepeatStride;
            u64 cclOff = cclBaseOff + tempAlgParams_.sliceSize * myAlgRank;
            DataSlice cclDstSlice(tempAlgParams_.buffInfo.hcclBuff.addr, cclOff, sliceSize, sliceCount);
            bool skipCclCopy = (tempAlgParams_.buffInfo.inputPtr == tempAlgParams_.buffInfo.hcclBuff.addr &&
                                inOff == cclOff);
            if (!skipCclCopy) {
                HCCL_DEBUG("[InsTempAllGatherMesh1D][LocalDataCopy] RankID [%d] AlgRank [%d] copy to ccl: "
                        "cclBaseOff[%llu] cclOff[%llu] sliceSize[%llu] count[%llu].",
                        myRank_, myAlgRank, cclBaseOff, cclOff, sliceSize, sliceCount);
                LocalCopy(threads[0], srcSlice, cclDstSlice);
            }
        }
    }
    return HcclResult::HCCL_SUCCESS;
}

HcclResult InsTempAllGatherMesh1D::PostLocalCopy(const std::vector<ThreadHandle> &threads)
{
    HCCL_INFO("[InsTempAllGatherMesh1D] PostLocalCopy.");
    if (tempAlgParams_.buffInfo.outBuffType == BufferType::HCCL_BUFFER) {
        HCCL_INFO("[InsTempAllGatherMesh1D] PostLocalCopy skip because output is scratch" );
        return HcclResult::HCCL_SUCCESS;
    }
    if (tempAlgParams_.buffInfo.inBuffType == BufferType::HCCL_BUFFER) {
        HCCL_INFO("[InsTempAllGatherMesh1D] PostLocalCopy skip because input is scratch and should be read to output" );
        return HcclResult::HCCL_SUCCESS;
    }
    const u32 dataTypeSize = DATATYPE_SIZE_TABLE[dataType_];
    u64 sliceSize = tempAlgParams_.sliceSize;
    for (u32 rpt = 0; rpt < tempAlgParams_.repeatNum; ++rpt) {
        const u64 outBaseOff = tempAlgParams_.buffInfo.outBuffBaseOff + rpt * tempAlgParams_.outputRepeatStride;
        const u64 scratchRepeatStride = tempAlgParams_.sliceSize * templateRankSize_;
        const u64 scratchBase = tempAlgParams_.buffInfo.hcclBuffBaseOff + rpt * scratchRepeatStride;

        for (auto rank : subCommRanks_[0]) {
            if (rank == myRank_) {
                continue;
            }
            u32 algRank = 0;
            CHK_RET(GetAlgRank(rank, subCommRanks_[0], algRank));
            // 尾块模式
            if (tempAlgParams_.tailSize !=0 && algRank == templateRankSize_ -1) {
                sliceSize = tempAlgParams_.tailSize;
            }
            u64 scratchOffset = tempAlgParams_.sliceSize * algRank + scratchBase;
            u64 outOffset = tempAlgParams_.outputSliceStride * algRank + outBaseOff;
            u64 sliceCount = sliceSize / dataTypeSize;
            DataSlice srcSlice(tempAlgParams_.buffInfo.hcclBuff.addr, scratchOffset, sliceSize, sliceCount);
            DataSlice dstSlice(tempAlgParams_.buffInfo.outputPtr, outOffset, sliceSize, sliceCount);
            HCCL_DEBUG("[InsTempAllGatherMesh1D] LocalDataCopy RankID [%d] dataRank [%d] dataAlgRank[%d] "
                       "scratchBase[%d] outBaseOff[%d] scratchOffset[%d] outOffset[%d].",
                       myRank_, rank, algRank, outBaseOff, outBaseOff, scratchOffset, outOffset);
            LocalCopy(threads[0], srcSlice, dstSlice);
        }
    }
    return HcclResult::HCCL_SUCCESS;
}

void InsTempAllGatherMesh1D::GetNotifyIdxMainToSub(std::vector<u32> &notifyIdxMianToSub)
{
    notifyIdxMianToSub.clear();
    u32 threadNum = GetThreadNum();
    u32 slaveThreadNum = threadNum - 1;
    for (u32 slaveThreadIdx = 0; slaveThreadIdx < slaveThreadNum; slaveThreadIdx++) {
        notifyIdxMianToSub.push_back(0);
    }
}

void InsTempAllGatherMesh1D::GetNotifyIdxSubToMain(std::vector<u32> &notifyIdxSubToMain)
{
    notifyIdxSubToMain.clear();
    u32 threadNum = GetThreadNum();
    u32 notifyNum = threadNum - 1;
    for (u32 notifyIdx = 0; notifyIdx < notifyNum; notifyIdx++) {
        notifyIdxSubToMain.push_back(notifyIdx);
    }
}

}  // namespace ops_hccl