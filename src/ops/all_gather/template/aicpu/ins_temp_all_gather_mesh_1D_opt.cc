/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "ins_temp_all_gather_mesh_1D_opt.h"
#include "alg_data_trans_wrapper.h"
#include "template_utils.h"
namespace ops_hccl {
InsTempAllGatherMesh1DOpt::InsTempAllGatherMesh1DOpt(const OpParam &param, const u32 rankId,
                                               const std::vector<std::vector<u32>> &subCommRanks)
    : InsAlgTemplateBase(param, rankId, subCommRanks)
{
}
InsTempAllGatherMesh1DOpt::~InsTempAllGatherMesh1DOpt() {}

HcclResult InsTempAllGatherMesh1DOpt::CalcRes(HcclComm comm, const OpParam &param, const TopoInfoWithNetLayerDetails *topoInfo,
                                           AlgResourceRequest &resourceRequest)
{
    HCCL_INFO("[InsTempAllGatherMesh1DOpt][CalcRes] start");
    GetRes(resourceRequest);
    std::vector<HcclChannelDesc> level0Channels;
    CHK_RET(CalcChannelRequestMesh1D(comm, param, topoInfo, subCommRanks_, level0Channels));
    resourceRequest.channels.push_back(level0Channels);
    return HCCL_SUCCESS;
}
HcclResult InsTempAllGatherMesh1DOpt::GetRes(AlgResourceRequest &resourceRequest) const
{
    u32 level0RankSize = templateRankSize_;
    u32 threadNum = level0RankSize > 1 ? level0RankSize - 1 : 1;
    resourceRequest.slaveThreadNum = threadNum - 1;
    resourceRequest.notifyNumPerThread.assign(resourceRequest.slaveThreadNum, 1);
    resourceRequest.notifyNumOnMainThread = threadNum - 1;
    return HCCL_SUCCESS;
}

u64 InsTempAllGatherMesh1DOpt::GetThreadNum() const
{
    return templateRankSize_ > 1 ? templateRankSize_ - 1 : 1;
}

u64 InsTempAllGatherMesh1DOpt::CalcScratchMultiple(BufferType inBuffType, BufferType outBuffType)
{
    (void)inBuffType;
    (void)outBuffType;
    u64 scratchMultiple = 0;
    if (opMode_ == OpMode::OPBASE){
        scratchMultiple = templateRankSize_;
    }
    return scratchMultiple;
}

HcclResult InsTempAllGatherMesh1DOpt::KernelRun(const OpParam &param, const TemplateDataParams &tempAlgParams,
                                             TemplateResource &templateResource)
{
    enableRemoteMemAccess_ = tempAlgParams.enableRemoteMemAccess;
    HCCL_INFO("[InsTempAllGatherMesh1DOpt] Run start");
    if (tempAlgParams.sliceSize == 0 && tempAlgParams.tailSize ==0) {
        HCCL_INFO("[InsTempAllGatherMesh1DOpt] Rank [%d], get slicesize zero.", myRank_);
        return HCCL_SUCCESS;
    }
    threadNum_ = templateResource.threads.size();
    tempAlgParams_ = tempAlgParams;
    dataType_ = param.DataDes.dataType;
    HCCL_DEBUG("[InsTempAllGatherMesh1DOpt] Rank [%d], get threadNum_[%d].", myRank_, threadNum_);
    // CHK_RET(LocalDataCopy(templateResource.threads));
    if (templateRankSize_ == 1) {
        return HcclResult::HCCL_SUCCESS;
    }
    if (threadNum_ > 1) {
        std::vector<ThreadHandle> subThreads(templateResource.threads.begin() + 1, templateResource.threads.end());
        GetNotifyIdxMainToSub(notifyIdxMainToSub_);
        CHK_RET(PreSyncInterThreads(templateResource.threads[0], subThreads, notifyIdxMainToSub_));
    }

    for (u32 rpt = 0; rpt < tempAlgParams_.repeatNum; ++rpt) {
        const u32 dataTypeSize = DATATYPE_SIZE_TABLE[dataType_];
        if (remoteWrite) {
            u64 sliceSize = tempAlgParams_.buffInfo.inputSize;
            u64 sliceCount = sliceSize / dataTypeSize;
            u64 inputOffset = tempAlgParams_.buffInfo.inBuffBaseOff;
            u64 scratchOffset = tempAlgParams_.buffInfo.hcclBuffBaseOff + myRank_ * tempAlgParams_.outputSliceStride;
            DataSlice srcSlice(tempAlgParams_.buffInfo.inputPtr, inputOffset, sliceSize, sliceCount);
            DataSlice dstSlice(tempAlgParams_.buffInfo.hcclBuff.addr, scratchOffset, sliceSize, sliceCount);
            CHK_RET(LocalCopy(templateResource.threads[0], srcSlice, dstSlice));
            break;
        } else {
            u64 sliceSize = tempAlgParams_.buffInfo.inputSize;
            u64 sliceCount = sliceSize / dataTypeSize;

            u32 localMeshRank = ((myRank_ + rpt * rankSize_ / meshSize_) % rankSize_);
            u64 scratchOffset = tempAlgParams_.buffInfo.hcclBuffBaseOff + localMeshRank * tempAlgParams_.outputSliceStride;;
            u64 outputOffset = tempAlgParams_.buffInfo.outBuffBaseOff + localMeshRank * tempAlgParams_.outputSliceStride;

            DataSlice srcSlice(tempAlgParams_.buffInfo.hcclBuff.addr, scratchOffset, sliceSize, sliceCount);
            DataSlice dstSlice(tempAlgParams_.buffInfo.outputPtr, outputOffset, sliceSize, sliceCount);
            CHK_RET(LocalCopy(templateResource.threads[0], srcSlice, dstSlice));
        }
    }
    CHK_RET(RunAllGatherMesh(templateResource.threads, templateResource.channels));

    if (threadNum_ > 1) {
        std::vector<ThreadHandle> subThreads(templateResource.threads.begin() + 1, templateResource.threads.end());
        GetNotifyIdxSubToMain(notifyIdxSubToMain_);
        CHK_RET(PostSyncInterThreads(templateResource.threads[0], subThreads, notifyIdxSubToMain_));
    }
    HCCL_INFO("[InsTempAllGatherMesh1DOpt] Run End");
    return HcclResult::HCCL_SUCCESS;
}

HcclResult InsTempAllGatherMesh1DOpt::RunAllGatherMesh(const std::vector<ThreadHandle> &threads,
                                                    const std::map<u32, std::vector<ChannelInfo>> &channels)
{
    HCCL_INFO("[InsTempAllGatherMesh1DOpt] RunAllGatherMesh RankIDs[%d].", myRank_);

    u32 myAlgRank = 0;
    CHK_RET(GetAlgRank(myRank_, subCommRanks_[0], myAlgRank));
    const u32 dataTypeSize = DATATYPE_SIZE_TABLE[dataType_];
    for (u32 threadIdx = 0; threadIdx < subCommRanks_[0].size() - 1; threadIdx++) {
        u32 connectedRank = subCommRanks_[0][(myAlgRank + 1 + threadIdx) % subCommRanks_[0].size()];

        u32 connectedAlgRank = 0;
        CHK_RET(GetAlgRank(connectedRank, subCommRanks_[0], connectedAlgRank));
        HCCL_INFO("[InsTempAllGatherMesh1DOpt] RunAllGatherMesh RankIDs[%d], connectedRank[%d], connectedAlgRank[%d].",
                    myRank_, connectedRank, connectedAlgRank);

        CHK_PRT_RET(threadIdx >= threads.size() || channels.count(connectedRank) == 0 ||
                    channels.at(connectedRank).empty(),
                    HCCL_ERROR("[InsTempAllGatherMesh1DOpt][RankID]=%u threadIdx=%u, threads.size=%u, "
                                "connectedRank=%d, channels.size=%u",
                                myRank_, threadIdx, threads.size(), connectedRank, channels.size()),
                    HcclResult::HCCL_E_INTERNAL);

        const ChannelInfo &linkRemote = channels.at(connectedRank)[0];
        void *remoteCclBuffAddr = linkRemote.remoteCclMem.addr;

        std::vector<DataSlice> txSrcSlicesAll;
        std::vector<DataSlice> txDstSlicesAll;
        std::vector<DataSlice> rxDstSlicesAll;
        std::vector<DataSlice> rxSrcSlicesAll;
        
        u64 sliceSize = tempAlgParams_.buffInfo.inputSize;
        u64 sliceCount = sliceSize / dataTypeSize;
        u64 outputSliceStride = tempAlgParams_.outputSliceStride;
        u64 inputSliceStride = tempAlgParams_.inputSliceStride;

        if (remoteWrite) {
            // 阶段1 远端写， 从本地的 input ptr 到 远端的 hccl buffer
            // 需要确保 当前场景下配置的repeatNum 为 1

            // tx 远端写
            void *txSrcPtr = tempAlgParams_.buffInfo.inputPtr;
            u64 txSrcOffset = tempAlgParams_.buffInfo.inBuffBaseOff;
            txSrcSlicesAll.emplace_back(txSrcPtr, txSrcOffset, sliceSize, sliceCount);

            void *txDstPtr = (!enableRemoteMemAccess_) ? remoteCclBuffAddr : linkRemote.remoteOutputGraphMode.addr;
            u64 txDstOffset = tempAlgParams_.buffInfo.hcclBuffBaseOff + myRank_ * outputSliceStride;
            txDstSlicesAll.emplace_back(txDstPtr, txDstOffset, sliceSize, sliceCount);

            // rx 远端读，不应该启动
            void *rxSrcPtr = (!enableRemoteMemAccess_) ? remoteCclBuffAddr : linkRemote.remoteOutputGraphMode.addr;
            u64 rxSrcOffset = tempAlgParams_.buffInfo.hcclBuffBaseOff + connectedRank * outputSliceStride;
            rxSrcSlicesAll.emplace_back(rxSrcPtr, rxSrcOffset, sliceSize, sliceCount);
            
            void *rxDstPtr = tempAlgParams_.buffInfo.outputPtr;
            u64 rxOutOffset = tempAlgParams_.buffInfo.outBuffBaseOff + connectedRank * outputSliceStride;
            rxDstSlicesAll.emplace_back(rxDstPtr, rxOutOffset, sliceSize, sliceCount);

            TxRxSlicesList sendRecvSlicesList({txSrcSlicesAll, txDstSlicesAll}, {rxSrcSlicesAll, rxDstSlicesAll});
            TxRxChannels sendRecvChannels(linkRemote, linkRemote);
            SendRecvInfo sendRecvInfo(sendRecvChannels, sendRecvSlicesList);
            CHK_PRT_RET(SendRecvWrite(sendRecvInfo, threads[threadIdx]),
                        HCCL_ERROR("[InsTempAllGatherMesh1DOpt] RunAllGather SendRecvWrite failed"), HcclResult::HCCL_E_INTERNAL);

        } else {
            // 阶段 2 远端读， 从远端的 hccl buffer 到 本地的 output buffer
            for (u32 rpt = 0; rpt < tempAlgParams_.repeatNum; ++rpt) {
                
                // tx 远端写, 不应该启动
                void *txSrcPtr = tempAlgParams_.buffInfo.inputPtr;
                u64 txSrcOffset = tempAlgParams_.buffInfo.inBuffBaseOff;
                txSrcSlicesAll.emplace_back(txSrcPtr, txSrcOffset, sliceSize, sliceCount);

                void *txDstPtr = (!enableRemoteMemAccess_) ? remoteCclBuffAddr : linkRemote.remoteOutputGraphMode.addr;
                u64 txDstOffset = tempAlgParams_.buffInfo.hcclBuffBaseOff + myRank_ * outputSliceStride;
                txDstSlicesAll.emplace_back(txDstPtr, txDstOffset, sliceSize, sliceCount);


                // rx 远端读
                void *rxSrcPtr = (!enableRemoteMemAccess_) ? remoteCclBuffAddr : linkRemote.remoteOutputGraphMode.addr;
                u64 rxSrcOffset = tempAlgParams_.buffInfo.hcclBuffBaseOff + ((connectedRank + rpt * rankSize_ / meshSize_) % rankSize_) * outputSliceStride;
                rxSrcSlicesAll.emplace_back(rxSrcPtr, rxSrcOffset, sliceSize, sliceCount);
                
                void *rxDstPtr = tempAlgParams_.buffInfo.outputPtr;
                u64 rxOutOffset = tempAlgParams_.buffInfo.outBuffBaseOff + ((connectedRank + rpt * rankSize_ / meshSize_) % rankSize_) * outputSliceStride;
                rxDstSlicesAll.emplace_back(rxDstPtr, rxOutOffset, sliceSize, sliceCount);

                TxRxSlicesList sendRecvSlicesList({txSrcSlicesAll, txDstSlicesAll}, {rxSrcSlicesAll, rxDstSlicesAll});
                TxRxChannels sendRecvChannels(linkRemote, linkRemote);
                SendRecvInfo sendRecvInfo(sendRecvChannels, sendRecvSlicesList);
                CHK_PRT_RET(SendRecvRead(sendRecvInfo, threads[threadIdx]),
                            HCCL_ERROR("[InsTempAllGatherMesh1DOpt] RunAllGather SendRecvRead failed"), HcclResult::HCCL_E_INTERNAL);
            }
        }
    }    
    return HcclResult::HCCL_SUCCESS;
}

void InsTempAllGatherMesh1DOpt::GetNotifyIdxMainToSub(std::vector<u32> &notifyIdxMianToSub)
{
    notifyIdxMianToSub.clear();
    u32 threadNum = GetThreadNum();
    u32 slaveThreadNum = threadNum - 1;
    for (u32 slaveThreadIdx = 0; slaveThreadIdx < slaveThreadNum; slaveThreadIdx++) {
        notifyIdxMianToSub.push_back(0);
    }
}

void InsTempAllGatherMesh1DOpt::GetNotifyIdxSubToMain(std::vector<u32> &notifyIdxSubToMain)
{
    notifyIdxSubToMain.clear();
    u32 threadNum = GetThreadNum();
    u32 notifyNum = threadNum - 1;
    for (u32 notifyIdx = 0; notifyIdx < notifyNum; notifyIdx++) {
        notifyIdxSubToMain.push_back(notifyIdx);
    }
}

}  // namespace ops_hccl