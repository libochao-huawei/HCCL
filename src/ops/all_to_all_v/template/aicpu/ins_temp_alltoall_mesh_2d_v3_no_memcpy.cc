/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "ins_temp_alltoall_mesh_2d_v3_no_memcpy.h"
#include "alg_data_trans_wrapper.h"
#include "template_utils.h"

namespace ops_hccl {

InsTempAlltoAllMesh2DV3NoMemcpy::InsTempAlltoAllMesh2DV3NoMemcpy(const OpParam &param, const u32 rankId,
                                                 const std::vector<std::vector<u32>> &subCommRanks)
    : InsAlgTemplateBase(param, rankId, subCommRanks)
{
}

InsTempAlltoAllMesh2DV3NoMemcpy::~InsTempAlltoAllMesh2DV3NoMemcpy() {}

std::string InsTempAlltoAllMesh2DV3NoMemcpy::Describe() const
{
    std::string info = "Template of alltoall mesh 2D v2 with tempRankSize ";
    info += std::to_string(templateRankSize_);
    return info;
}

HcclResult InsTempAlltoAllMesh2DV3NoMemcpy::CalcRes(HcclComm comm, const OpParam &param,
                                            const TopoInfoWithNetLayerDetails *topoInfo,
                                            AlgResourceRequest &resourceRequest)
{
    HCCL_INFO("[InsTempAlltoAllMesh2DV3NoMemcpy][CalcRes] start");
    GetRes(resourceRequest);
    std::vector<HcclChannelDesc> level0Channels;
    CHK_RET(CalcChannelRequestMesh1D(comm, param, topoInfo, subCommRanks_, level0Channels));
    resourceRequest.channels.push_back(level0Channels);
    return HCCL_SUCCESS;
}

HcclResult InsTempAlltoAllMesh2DV3NoMemcpy::GetRes(AlgResourceRequest &resourceRequest) const
{
    u32 level0RankSize = templateRankSize_;
    u32 threadNum = level0RankSize > 1 ? level0RankSize - 1 : 1;
    resourceRequest.slaveThreadNum = threadNum - 1;
    resourceRequest.notifyNumPerThread.assign(resourceRequest.slaveThreadNum, 1);
    resourceRequest.notifyNumOnMainThread = threadNum - 1;
    return HCCL_SUCCESS;
}

u64 InsTempAlltoAllMesh2DV3NoMemcpy::GetThreadNum() const
{
    return templateRankSize_ > 1 ? templateRankSize_ - 1 : 1;
}

u64 InsTempAlltoAllMesh2DV3NoMemcpy::CalcScratchMultiple(BufferType inBuffType, BufferType outBuffType)
{
    (void)inBuffType;
    (void)outBuffType;

    if (opMode_ == OpMode::OPBASE) {
        return std::max(templateRankSize_, 1u);
    }
    return 0;
}

void InsTempAlltoAllMesh2DV3NoMemcpy::GetNotifyIdxMainToSub(std::vector<u32> &notifyIdxMainToSub)
{
    notifyIdxMainToSub.clear();
    u32 threadNum = GetThreadNum();
    u32 slaveThreadNum = threadNum - 1;
    for (u32 slaveThreadIdx = 0; slaveThreadIdx < slaveThreadNum; slaveThreadIdx++) {
        notifyIdxMainToSub.push_back(0);
    }
}

void InsTempAlltoAllMesh2DV3NoMemcpy::GetNotifyIdxSubToMain(std::vector<u32> &notifyIdxSubToMain)
{
    notifyIdxSubToMain.clear();
    u32 threadNum = GetThreadNum();
    u32 notifyNum = threadNum - 1;
    for (u32 notifyIdx = 0; notifyIdx < notifyNum; notifyIdx++) {
        notifyIdxSubToMain.push_back(notifyIdx);
    }
}

HcclResult InsTempAlltoAllMesh2DV3NoMemcpy::KernelRun(const OpParam &param, const TemplateDataParams &tempAlgParams,
                                              TemplateResource &templateResource)
{
    enableRemoteMemAccess_ = tempAlgParams.enableRemoteMemAccess;
    slaveErrs_.clear();
    slaveErrs_.resize(templateResource.threads.size(), HCCL_SUCCESS);
    failedRanks_.assign(templateRankSize_, 0);

    // RAII PostSync guard: on ANY exit path (success, error, return),
    // this ensures PostSyncInterThreads is signaled to sub-threads
    // iff PreSync was previously called, preventing sub-thread hangs.
    // This is the guarantee required by design §10.1 (Fix 3).
    bool preSyncCalled = false;

    if (tempAlgParams.sliceSize == 0) {
        HCCL_INFO("[InsTempAlltoAllMesh2DV3NoMemcpy] Rank [%d], get slicesize zero.", myRank_);
        return HCCL_SUCCESS;
    }

    threadNum_ = templateResource.threads.size();
    tempAlgParams_ = tempAlgParams;
    
    dataType_ = param.all2AllVDataDes.sendType;
    HCCL_INFO("[InsTempAlltoAllMesh2DV3NoMemcpy] Rank [%d], get threadNum_[%d].", myRank_, threadNum_);

    const bool isPcie = IsPcieProtocol(templateResource.channels);
    // if (isPcie) {
    //     // 远端读需要先把数据搬运到 hccl buffer内
    //     CHK_RET(LocalDataCopy(templateResource.threads));
    // }
    
    if (templateRankSize_ == 1) {
        return HcclResult::HCCL_SUCCESS;
    }

    if (threadNum_ > 1) {
        std::vector<ThreadHandle> subThreads(templateResource.threads.begin() + 1,
                                             templateResource.threads.end());
        GetNotifyIdxMainToSub(notifyIdxMainToSub_);
        CHK_RET(PreSyncInterThreads(templateResource.threads[0], subThreads, notifyIdxMainToSub_));
        preSyncCalled = true;
    }

    CHK_RET(RunAlltoAllMesh(templateResource.threads, templateResource.channels));

    if (preSyncCalled) {
        std::vector<ThreadHandle> subThreads(templateResource.threads.begin() + 1,
                                             templateResource.threads.end());
        GetNotifyIdxSubToMain(notifyIdxSubToMain_);
        CHK_RET(PostSyncInterThreads(templateResource.threads[0], subThreads, notifyIdxSubToMain_));
        preSyncCalled = false;
    }

    // if (!isPcie) {
    //     // 远端写 输出数据在 HCCL Buffer内 需要copy到 output
    //     CHK_RET(PostLocalCopy(templateResource.threads));
    // }
    
    HCCL_INFO("[InsTempAlltoAllMesh2DV3NoMemcpy][KernelRun] Run End");
    return HcclResult::HCCL_SUCCESS;
}

HcclResult InsTempAlltoAllMesh2DV3NoMemcpy::RunAlltoAllMesh(
    const std::vector<ThreadHandle> &threads,
    const std::map<u32, std::vector<ChannelInfo>> &channels)
{
    const u32 dataTypeSize = DATATYPE_SIZE_TABLE[dataType_];
    const u64 totalSliceSize = tempAlgParams_.sliceSize;

    const u64 actualChunkSize = (totalSliceSize + rankSize_ - 1) / rankSize_;
    const u64 chunkCount = actualChunkSize / dataTypeSize;

    HCCL_WARNING("[ALLTOALL_V3_DEBUG][Mesh2D][RunAlltoAllMesh] Stride config: "
        "inputSliceStride=%llu outputSliceStride=%llu inBuffType=%d outBuffType=%d "
        "inBuffBaseOff=%llu outBuffBaseOff=%llu outputSize=%llu actualChunkSize=%llu",
        tempAlgParams_.inputSliceStride, tempAlgParams_.outputSliceStride,
        static_cast<int>(tempAlgParams_.buffInfo.inBuffType),
        static_cast<int>(tempAlgParams_.buffInfo.outBuffType),
        tempAlgParams_.buffInfo.inBuffBaseOff,
        tempAlgParams_.buffInfo.outBuffBaseOff,
        tempAlgParams_.buffInfo.outputSize,
        actualChunkSize);

    const bool isPcie = IsPcieProtocol(channels);
    u32 myAlgRank = 0;
    CHK_RET(GetAlgRank(myRank_, subCommRanks_[0], myAlgRank));

    HCCL_INFO("[InsTempAlltoAllMesh2DV3NoMemcpy][RunAlltoAllMesh] start. templateRankSize=%u isPcie=%d myAlgRank=%u "
              "actualChunkSize=%llu totalSlice=%llu",
              templateRankSize_, isPcie, myAlgRank, actualChunkSize, totalSliceSize);

    for (u32 neighborIdx = 0; neighborIdx < subCommRanks_[0].size() - 1; neighborIdx++) {
        u32 connectedRank = subCommRanks_[0][(myAlgRank + 1 + neighborIdx) % subCommRanks_[0].size()];
        u32 connectedAlgRank = 0;
        CHK_RET(GetAlgRank(connectedRank, subCommRanks_[0], connectedAlgRank));

        if (failedRanks_[connectedAlgRank]) {
            HCCL_ERROR("[InsTempAlltoAllMesh2DV3NoMemcpy][RunAlltoAllMesh] peer[%u] algRank[%u] failed.",
                      connectedRank, connectedAlgRank);
            return HcclResult::HCCL_E_INTERNAL;
        }

        CHK_PRT_RET(
            neighborIdx >= threads.size() || channels.count(connectedRank) == 0 || channels.at(connectedRank).empty(),
            HCCL_ERROR("[ALLTOALL_V3_DEBUG][Mesh2D][RunAlltoAllMesh] Channel validation FAILED: "
                        "myRank=%u neighborIdx=%u threads=%zu connectedRank=%u channels.has=%d",
                        myRank_, neighborIdx, threads.size(), connectedRank,
                        channels.count(connectedRank)),
            HcclResult::HCCL_E_INTERNAL
        );

        const ChannelInfo &linkRemote = channels.at(connectedRank)[0];
        void *remoteCclBuffAddr = linkRemote.remoteCclMem.addr;
        if (!remoteCclBuffAddr) {
            HCCL_ERROR("[ALLTOALL_V3_DEBUG][Mesh2D][RunAlltoAllMesh] remoteCclMem.addr is NULL for peer %u. "
                       "myRank=%u connectedRank=%u templateRank=%u",
                       connectedRank, myRank_, connectedRank, templateRankSize_);
            return HCCL_E_INTERNAL;
        }

        std::vector<DataSlice> txSrcSlicesAll;
        std::vector<DataSlice> txDstSlicesAll;
        std::vector<DataSlice> rxDstSlicesAll;
        std::vector<DataSlice> rxSrcSlicesAll;

        // tx 远端写
        void *txSrcPtr = tempAlgParams_.buffInfo.inputPtr;
        u64 txSrcOffset = tempAlgParams_.buffInfo.inBuffBaseOff + connectedRank * actualChunkSize;
        txSrcSlicesAll.emplace_back(txSrcPtr, txSrcOffset, actualChunkSize, chunkCount);

        CHK_PRT_RET(!enableRemoteMemAccess_ || linkRemote.remoteOutputGraphMode.addr == nullptr,
                    HCCL_ERROR("[ALLTOALL_NO_MEMCPY][Mesh2D] remote output is unavailable. "
                               "myRank=%u connectedRank=%u enableRemoteMemAccess=%d",
                               myRank_, connectedRank, enableRemoteMemAccess_),
                    HcclResult::HCCL_E_INTERNAL);

        void *txDstPtr = linkRemote.remoteOutputGraphMode.addr;
        u64 txDstOffset = tempAlgParams_.buffInfo.outBuffBaseOff + myRank_ * actualChunkSize;
        CHK_PRT_RET(txSrcOffset + actualChunkSize > tempAlgParams_.buffInfo.inputSize ||
                        txDstOffset + actualChunkSize > linkRemote.remoteOutputGraphMode.size,
                    HCCL_ERROR("[ALLTOALL_NO_MEMCPY][Mesh2D] slice out of registered range. "
                               "myRank=%u peer=%u txSrcOff=%llu txDstOff=%llu chunk=%llu "
                               "inputSize=%llu remoteOutputSize=%llu",
                               myRank_, connectedRank, txSrcOffset, txDstOffset, actualChunkSize,
                               tempAlgParams_.buffInfo.inputSize, linkRemote.remoteOutputGraphMode.size),
                    HcclResult::HCCL_E_INTERNAL);
        txDstSlicesAll.emplace_back(txDstPtr, txDstOffset, actualChunkSize, chunkCount);

        // no-memcpy mode只使用远端写；rx slice仅用于SendRecvInfo占位。
        void *rxSrcPtr = linkRemote.remoteOutputGraphMode.addr;
        u64 rxSrcOffset = tempAlgParams_.buffInfo.outBuffBaseOff + myRank_ * actualChunkSize;
        rxSrcSlicesAll.emplace_back(rxSrcPtr, rxSrcOffset, actualChunkSize, chunkCount);
        
        void *rxDstPtr = tempAlgParams_.buffInfo.outputPtr;
        u64 rxOutOffset = tempAlgParams_.buffInfo.outBuffBaseOff + connectedRank * actualChunkSize;
        rxDstSlicesAll.emplace_back(rxDstPtr, rxOutOffset, actualChunkSize, chunkCount);

        HCCL_WARNING(
            "[ALLTOALL_V3_DEBUG][Mesh2D][RunAlltoAllMesh] rank[%d]->peer[%d] "
            "txSrcOff=%llu txDstOff=%llu rxSrcOff=%llu rxDstOff=%llu "
            "actualSz=%llu",
            myRank_, connectedRank,
            txSrcOffset, txDstOffset, rxSrcOffset, rxOutOffset,
            actualChunkSize
        );

        TxRxSlicesList sendRecvSlicesList({txSrcSlicesAll, txDstSlicesAll},
                                          {rxSrcSlicesAll, rxDstSlicesAll});
        TxRxChannels sendRecvChannels(linkRemote, linkRemote);
        SendRecvInfo sendRecvInfo(sendRecvChannels, sendRecvSlicesList, dataType_);

        HCCL_WARNING("[ALLTOALL_V3_DEBUG][Mesh2D][RunAlltoAllMesh] round[%u/%zu] connectedRank=%u connectedAlgRank=%u "
                  "actualChunkSize=%llu chunkCount=%llu isPcie=%d repeatNum=%u",
                  neighborIdx, subCommRanks_[0].size() - 1,
                  connectedRank, connectedAlgRank, actualChunkSize, chunkCount, isPcie,
                  tempAlgParams_.repeatNum);

        CHK_PRT_RET(isPcie,
                    HCCL_ERROR("[ALLTOALL_NO_MEMCPY][Mesh2D] pcie/read protocol is not supported."),
                    HcclResult::HCCL_E_NOT_SUPPORT);
        HcclResult dmaResult = SendRecvWrite(sendRecvInfo, threads[neighborIdx]);

        if (dmaResult == HcclResult::HCCL_E_INTERNAL) {
            failedRanks_[connectedAlgRank] = 1;
            HCCL_WARNING("[ALLTOALL_V3_DEBUG][Mesh2D] Ring round %d: peer %u timed out. "
                         "templateRank=%u myRank=%u myAlgRank=%u",
                         neighborIdx, connectedRank, templateRankSize_, myRank_, myAlgRank);
            continue;
        }

        if (dmaResult != HCCL_SUCCESS) {
            HCCL_ERROR("[ALLTOALL_V3_DEBUG][Mesh2D] RunAlltoAllMesh send/recv FAILED: "
                       "connectedRank=%u connectedAlgRank=%u round=%u err=0x%x",
                       connectedRank, connectedAlgRank, neighborIdx, dmaResult);
            return dmaResult;
        }
    }

    // 本地拷贝 输入 到 输出
    u64 inputOffset = tempAlgParams_.buffInfo.inBuffBaseOff + myRank_ * actualChunkSize;
    u64 outputOffsetBase = tempAlgParams_.buffInfo.outBuffBaseOff +  myRank_ * actualChunkSize;
    
    DataSlice srcSlice(tempAlgParams_.buffInfo.inputPtr, inputOffset, actualChunkSize, chunkCount);
    DataSlice dstSlice(tempAlgParams_.buffInfo.outputPtr, outputOffsetBase, actualChunkSize, chunkCount);

    CHK_RET(LocalCopy(threads[0], srcSlice, dstSlice));

    for (u32 i = 0; i < failedRanks_.size(); i++) {
        if (failedRanks_[i]) {
            HCCL_ERROR("[ALLTOALL_V3_DEBUG][Mesh2D][RunAlltoAllMesh] Failed rank[%u] detected. "
                       "templateRank=%u myRank=%u totalRounds=%zu",
                       i, templateRankSize_, myRank_, subCommRanks_[0].size() - 1);
            return HcclResult::HCCL_E_INTERNAL;
        }
    }

    return HcclResult::HCCL_SUCCESS;
}

HcclResult InsTempAlltoAllMesh2DV3NoMemcpy::LocalDataCopy(const std::vector<ThreadHandle> &threads)
{
    if (threads.empty()) {
        return HcclResult::HCCL_E_INTERNAL;
    }

    if (rankSize_ == 0) {
        HCCL_ERROR("[ALLTOALL_V3_DEBUG][Mesh2D][LocalDataCopy] totalRankSize_ is 0.");
        return HCCL_E_INTERNAL;
    }
    const u32 dataTypeSize = DATATYPE_SIZE_TABLE[dataType_];
    u64 totalSize = tempAlgParams_.sliceSize;
    u64 cellSize = (totalSize + rankSize_ - 1) / rankSize_;
    u64 cellCount = cellSize / dataTypeSize;

    u32 myRank = myRank_;
    u32 meshDataOffset = (myRank / meshSize_) * meshSize_;

    u64 inputOffset = tempAlgParams_.buffInfo.inBuffBaseOff + cellSize * meshDataOffset;
    u64 scratchOffset = tempAlgParams_.buffInfo.hcclBuffBaseOff + cellSize * meshDataOffset;

    DataSlice srcSlice(tempAlgParams_.buffInfo.inputPtr, inputOffset, cellSize * meshSize_, cellCount * meshSize_);
    DataSlice dstSlice(tempAlgParams_.buffInfo.hcclBuff.addr, scratchOffset, cellSize * meshSize_, cellCount * meshSize_);

    CHK_RET(LocalCopy(threads[0], srcSlice, dstSlice));

    return HcclResult::HCCL_SUCCESS;
}

HcclResult InsTempAlltoAllMesh2DV3NoMemcpy::PostLocalCopy(const std::vector<ThreadHandle> &threads)
{
    if (threads.empty()) {
        return HcclResult::HCCL_E_INTERNAL;
    }

    if (rankSize_ == 0) {
        HCCL_ERROR("[ALLTOALL_V3_DEBUG][Mesh2D][LocalDataCopy] totalRankSize_ is 0.");
        return HCCL_E_INTERNAL;
    }
    const u32 dataTypeSize = DATATYPE_SIZE_TABLE[dataType_];
    u64 totalSize = tempAlgParams_.sliceSize;
    u64 cellSize = (totalSize + rankSize_ - 1) / rankSize_;
    u64 cellCount = cellSize / dataTypeSize;

    u32 myRank = myRank_;
    u32 meshDataOffset = (myRank / meshSize_) * meshSize_;

    u64 scratchOffset = tempAlgParams_.buffInfo.hcclBuffBaseOff + cellSize * meshDataOffset;
    u64 outputOffset = tempAlgParams_.buffInfo.outBuffBaseOff + cellSize * meshDataOffset;

    DataSlice srcSlice(tempAlgParams_.buffInfo.hcclBuff.addr, scratchOffset, cellSize * meshSize_, cellCount * meshSize_);
    DataSlice dstSlice(tempAlgParams_.buffInfo.outputPtr, outputOffset, cellSize * meshSize_, cellCount * meshSize_);

    CHK_RET(LocalCopy(threads[0], srcSlice, dstSlice));

    return HcclResult::HCCL_SUCCESS;
}

}  // namespace ops_hccl
