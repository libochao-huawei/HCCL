/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "ins_temp_alltoall_mesh_2d_v2.h"
#include "alg_data_trans_wrapper.h"
#include "template_utils.h"

namespace ops_hccl {

InsTempAlltoAllMesh2DV2::InsTempAlltoAllMesh2DV2(const OpParam &param, const u32 rankId,
                                                 const std::vector<std::vector<u32>> &subCommRanks)
    : InsAlgTemplateBase(param, rankId, subCommRanks)
{
}

InsTempAlltoAllMesh2DV2::~InsTempAlltoAllMesh2DV2() {}

std::string InsTempAlltoAllMesh2DV2::Describe() const
{
    std::string info = "Template of alltoall mesh 2D v2 with tempRankSize ";
    info += std::to_string(templateRankSize_);
    return info;
}

HcclResult InsTempAlltoAllMesh2DV2::CalcRes(HcclComm comm, const OpParam &param,
                                            const TopoInfoWithNetLayerDetails *topoInfo,
                                            AlgResourceRequest &resourceRequest)
{
    HCCL_INFO("[InsTempAlltoAllMesh2DV2][CalcRes] start");
    GetRes(resourceRequest);
    std::vector<HcclChannelDesc> level0Channels;
    CHK_RET(CalcChannelRequestMesh1D(comm, param, topoInfo, subCommRanks_, level0Channels));
    resourceRequest.channels.push_back(level0Channels);
    return HCCL_SUCCESS;
}

HcclResult InsTempAlltoAllMesh2DV2::GetRes(AlgResourceRequest &resourceRequest) const
{
    u32 level0RankSize = templateRankSize_;
    u32 threadNum = level0RankSize > 1 ? level0RankSize - 1 : 1;
    resourceRequest.slaveThreadNum = threadNum - 1;
    resourceRequest.notifyNumPerThread.assign(resourceRequest.slaveThreadNum, 1);
    resourceRequest.notifyNumOnMainThread = threadNum - 1;
    return HCCL_SUCCESS;
}

u64 InsTempAlltoAllMesh2DV2::GetThreadNum() const
{
    return templateRankSize_ > 1 ? templateRankSize_ - 1 : 1;
}

u64 InsTempAlltoAllMesh2DV2::CalcScratchMultiple(BufferType inBuffType, BufferType outBuffType)
{
    (void)inBuffType;
    (void)outBuffType;

    if (opMode_ == OpMode::OPBASE) {
        return std::max(templateRankSize_, 1u);
    }
    return 0;
}

void InsTempAlltoAllMesh2DV2::GetNotifyIdxMainToSub(std::vector<u32> &notifyIdxMainToSub)
{
    notifyIdxMainToSub.clear();
    u32 threadNum = GetThreadNum();
    u32 slaveThreadNum = threadNum - 1;
    for (u32 slaveThreadIdx = 0; slaveThreadIdx < slaveThreadNum; slaveThreadIdx++) {
        notifyIdxMainToSub.push_back(0);
    }
}

void InsTempAlltoAllMesh2DV2::GetNotifyIdxSubToMain(std::vector<u32> &notifyIdxSubToMain)
{
    notifyIdxSubToMain.clear();
    u32 threadNum = GetThreadNum();
    u32 notifyNum = threadNum - 1;
    for (u32 notifyIdx = 0; notifyIdx < notifyNum; notifyIdx++) {
        notifyIdxSubToMain.push_back(notifyIdx);
    }
}

HcclResult InsTempAlltoAllMesh2DV2::KernelRun(const OpParam &param, const TemplateDataParams &tempAlgParams,
                                              TemplateResource &templateResource)
{
    enableRemoteMemAccess_ = tempAlgParams.enableRemoteMemAccess;
    HCCL_INFO("[InsTempAlltoAllMesh2DV2][KernelRun] Run start. sliceSize=%llu templateRankSize=%u threadNum=%zu "
              "xRank=%u yRank=%u totalRank=%u inBuffBase=%llu outBuffBase=%llu hcclBase=%llu",
              tempAlgParams.sliceSize, templateRankSize_, templateResource.threads.size(),
              xRankSize_, yRankSize_, totalRankSize_,
              tempAlgParams.buffInfo.inBuffBaseOff,
              tempAlgParams.buffInfo.outBuffBaseOff,
              tempAlgParams.buffInfo.hcclBuffBaseOff);

    slaveErrs_.clear();
    slaveErrs_.resize(templateResource.threads.size(), HCCL_SUCCESS);
    failedRanks_.assign(templateRankSize_, 0);

    // RAII PostSync guard: on ANY exit path (success, error, return),
    // this ensures PostSyncInterThreads is signaled to sub-threads
    // iff PreSync was previously called, preventing sub-thread hangs.
    // This is the guarantee required by design §10.1 (Fix 3).
    bool preSyncCalled = false;

    if (tempAlgParams.sliceSize == 0) {
        HCCL_INFO("[InsTempAlltoAllMesh2DV2] Rank [%d], get slicesize zero.", myRank_);
        return HCCL_SUCCESS;
    }

    threadNum_ = templateResource.threads.size();
    tempAlgParams_ = tempAlgParams;

    // v3.0 Fix A: Element-level guard — fall back to local-only if chunk < one element
    u32 dataTypeSize = DATATYPE_SIZE_TABLE[param.DataDes.dataType];
    u64 localPerPeerChunkSize = (tempAlgParams.sliceSize + totalRankSize_ - 1) / totalRankSize_;
    if (localPerPeerChunkSize < static_cast<u64>(dataTypeSize) && tempAlgParams.sliceSize > 0) {
        HCCL_INFO("[InsTempAlltoAllMesh2DV2] perPeerChunkSize[%llu] < dataTypeSize[%u], fallback to local-only.",
                  localPerPeerChunkSize, dataTypeSize);
        // Temporarily override inBuffType to INPUT so isScratchToOutput guard
        // in LocalDataCopy allows output copy (needed when fallback fires in Stage 2
        // where inBuffType=HCCL_BUFFER would otherwise block the output write)
        auto savedInBuffType = tempAlgParams_.buffInfo.inBuffType;
        tempAlgParams_.buffInfo.inBuffType = BufferType::INPUT;
        CHK_RET(LocalDataCopy(templateResource.threads));
        tempAlgParams_.buffInfo.inBuffType = savedInBuffType;
        return HCCL_SUCCESS;
    }
    dataType_ = param.DataDes.dataType;
    HCCL_DEBUG("[InsTempAlltoAllMesh2DV2] Rank [%d], get threadNum_[%d].", myRank_, threadNum_);

    CHK_RET(LocalDataCopy(templateResource.threads));

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

    HcclResult ringErr = RunAlltoAllMesh(templateResource.threads, templateResource.channels);
    if (ringErr != HCCL_SUCCESS) {
        slaveErrs_[0] = ringErr;
    }

    if (preSyncCalled) {
        std::vector<ThreadHandle> subThreads(templateResource.threads.begin() + 1,
                                             templateResource.threads.end());
        GetNotifyIdxSubToMain(notifyIdxSubToMain_);
        CHK_RET(PostSyncInterThreads(templateResource.threads[0], subThreads, notifyIdxSubToMain_));
        preSyncCalled = false;
    }

    if (ringErr != HCCL_SUCCESS) {
        for (size_t i = 0; i < slaveErrs_.size(); i++) {
            if (slaveErrs_[i] != HCCL_SUCCESS) {
                HCCL_ERROR("[InsTempAlltoAllMesh2DV2] Ring exchange failed, skip post-copy.");
                return slaveErrs_[i];
            }
        }
    }

    CHK_RET(PostLocalCopy(templateResource.threads));
    HCCL_INFO("[InsTempAlltoAllMesh2DV2][KernelRun] Run End");
    return HcclResult::HCCL_SUCCESS;
}

HcclResult InsTempAlltoAllMesh2DV2::RunAlltoAllMesh(
    const std::vector<ThreadHandle> &threads,
    const std::map<u32, std::vector<ChannelInfo>> &channels)
{
    const u32 dataTypeSize = DATATYPE_SIZE_TABLE[dataType_];
    const u64 totalSliceSize = tempAlgParams_.sliceSize;
    const u64 perPeerChunkSize = (totalSliceSize + templateRankSize_ - 1) / templateRankSize_;
    u64 lastPeerSize = totalSliceSize - perPeerChunkSize * (templateRankSize_ - 1);
    if (lastPeerSize <= 0) { lastPeerSize = perPeerChunkSize; }
    const u64 lastPeerIndex = templateRankSize_ - 1;
    const bool isPcie = IsPcieProtocol(channels);
    u32 myAlgRank = 0;
    CHK_RET(GetAlgRank(myRank_, subCommRanks_[0], myAlgRank));

    HCCL_INFO("[InsTempAlltoAllMesh2DV2][RunAlltoAllMesh] start. templateRankSize=%u isPcie=%d myAlgRank=%u "
              "perPeerChunk=%llu totalSlice=%llu lastPeerSize=%llu lastPeerIndex=%llu",
              templateRankSize_, isPcie, myAlgRank, perPeerChunkSize, totalSliceSize,
              lastPeerSize, lastPeerIndex);

    for (u32 neighborIdx = 0; neighborIdx < subCommRanks_[0].size() - 1; neighborIdx++) {
        u32 connectedRank = subCommRanks_[0][(myAlgRank + 1 + neighborIdx) % subCommRanks_[0].size()];
        u32 connectedAlgRank = 0;
        CHK_RET(GetAlgRank(connectedRank, subCommRanks_[0], connectedAlgRank));

        if (failedRanks_[connectedAlgRank]) {
            HCCL_INFO("[InsTempAlltoAllMesh2DV2][RunAlltoAllMesh] peer[%u] algRank[%u] already failed, skip.",
                      connectedRank, connectedAlgRank);
            continue;
        }

        CHK_PRT_RET(neighborIdx >= threads.size() || channels.count(connectedRank) == 0 ||
                    channels.at(connectedRank).empty(),
                    HCCL_ERROR("[InsTempAlltoAllMesh2DV2][RankID]=%u neighborIdx=%u, threads.size=%u, "
                                "connectedRank=%d, channels.size=%u",
                                myRank_, neighborIdx, threads.size(), connectedRank, channels.size()),
                    HcclResult::HCCL_E_INTERNAL);

        const ChannelInfo &linkRemote = channels.at(connectedRank)[0];
        void *remoteCclBuffAddr = linkRemote.remoteCclMem.addr;

        std::vector<DataSlice> txSrcSlicesAll;
        std::vector<DataSlice> txDstSlicesAll;
        std::vector<DataSlice> rxDstSlicesAll;
        std::vector<DataSlice> rxSrcSlicesAll;

        u64 actualChunkSize = (connectedAlgRank == lastPeerIndex) ? lastPeerSize : perPeerChunkSize;
        u64 chunkCount = actualChunkSize / dataTypeSize;

        // v3.0 Fix B (caveat #3): per-peer clamping for each peer
        u64 offsetInSlice = connectedAlgRank * perPeerChunkSize;
        u64 remainingAtOffset = (offsetInSlice < totalSliceSize) ? (totalSliceSize - offsetInSlice) : 0;
        actualChunkSize = std::min(actualChunkSize, remainingAtOffset);
        if (actualChunkSize == 0) {
            continue;
        }
        chunkCount = actualChunkSize / dataTypeSize;

        for (u32 rpt = 0; rpt < tempAlgParams_.repeatNum; ++rpt) {
            const u64 outBaseOff = tempAlgParams_.buffInfo.outBuffBaseOff +
                                   rpt * tempAlgParams_.outputRepeatStride;
            const u64 scratchRepeatStride = tempAlgParams_.sliceSize * templateRankSize_;
            const u64 scratchBase = tempAlgParams_.buffInfo.hcclBuffBaseOff +
                                    rpt * scratchRepeatStride;

            u64 txSrcInputOffset = tempAlgParams_.outputSliceStride * connectedAlgRank + outBaseOff;
            u64 txSrcScratchOffset = tempAlgParams_.buffInfo.inBuffBaseOff +
                                     perPeerChunkSize * connectedAlgRank;
            u64 txScratchOffset = scratchBase + perPeerChunkSize * connectedAlgRank;
            u64 txDstOffset = (!enableRemoteMemAccess_) ? txScratchOffset : txSrcInputOffset;

            u64 rxOutOffset = tempAlgParams_.outputSliceStride * myAlgRank + outBaseOff;
            u64 rxScratchOffset = scratchBase + perPeerChunkSize * myAlgRank;
            u64 rxSrcOffset = (!enableRemoteMemAccess_) ? rxScratchOffset : rxOutOffset;

            void *txSrcPtr = tempAlgParams_.buffInfo.inputPtr;
            void *txDstPtr = (!enableRemoteMemAccess_) ? remoteCclBuffAddr
                                                          : linkRemote.remoteOutputGraphMode.addr;
            void *rxSrcPtr = (!enableRemoteMemAccess_) ? remoteCclBuffAddr
                                                          : linkRemote.remoteOutputGraphMode.addr;
            void *rxDstPtr = tempAlgParams_.buffInfo.outputPtr;

            bool readingFromScratch = (tempAlgParams_.buffInfo.inBuffType == BufferType::HCCL_BUFFER);
            u64 txSrcOffset = readingFromScratch ? txSrcScratchOffset : txSrcInputOffset;

            txSrcSlicesAll.emplace_back(txSrcPtr, txSrcOffset, actualChunkSize, chunkCount);
            txDstSlicesAll.emplace_back(txDstPtr, txDstOffset, actualChunkSize, chunkCount);
            rxDstSlicesAll.emplace_back(rxDstPtr, rxOutOffset, actualChunkSize, chunkCount);
            rxSrcSlicesAll.emplace_back(rxSrcPtr, rxSrcOffset, actualChunkSize, chunkCount);

            HCCL_DEBUG("[InsTempAlltoAllMesh2DV2][RunAlltoAllMesh] rankId [%d] connectedRank [%d] rpt [%d] "
                        "txSrc offset[%llu] sliceSize[%llu] count[%llu] fromScratch[%d].",
                        myRank_, connectedRank, rpt, txSrcOffset, actualChunkSize, chunkCount,
                        readingFromScratch);
        }

        TxRxSlicesList sendRecvSlicesList({txSrcSlicesAll, txDstSlicesAll},
                                          {rxSrcSlicesAll, rxDstSlicesAll});
        TxRxChannels sendRecvChannels(linkRemote, linkRemote);
        SendRecvInfo sendRecvInfo(sendRecvChannels, sendRecvSlicesList);

        HCCL_INFO("[InsTempAlltoAllMesh2DV2][RunAlltoAllMesh] round[%u] connectedRank=%u connectedAlgRank=%u "
                  "actualChunkSize=%llu isPcie=%d",
                  neighborIdx, connectedRank, connectedAlgRank, actualChunkSize, isPcie);

        HcclResult dmaResult;
        if (isPcie) {
            dmaResult = SendRecvRead(sendRecvInfo, threads[neighborIdx]);
        } else {
            dmaResult = SendRecvWrite(sendRecvInfo, threads[neighborIdx]);
        }

        if (dmaResult == HcclResult::HCCL_E_INTERNAL) {
            failedRanks_[connectedAlgRank] = 1;
            HCCL_WARNING("[InsTempAlltoAllMesh2DV2] Ring round %d: peer %u timed out, skipping.",
                         neighborIdx, connectedRank);
            continue;
        }

        if (dmaResult != HCCL_SUCCESS) {
            HCCL_ERROR("[InsTempAlltoAllMesh2DV2] RunAlltoAllMesh send/recv failed for connectedRank[%u]",
                       connectedRank);
            return dmaResult;
        }
    }

    for (u32 i = 0; i < failedRanks_.size(); i++) {
        if (failedRanks_[i]) {
            return HcclResult::HCCL_E_INTERNAL;
        }
    }

    return HcclResult::HCCL_SUCCESS;
}

HcclResult InsTempAlltoAllMesh2DV2::LocalDataCopy(const std::vector<ThreadHandle> &threads)
{
    HCCL_INFO("[InsTempAlltoAllMesh2DV2][LocalDataCopy] Start. totalRankSize=%u xRank=%u yRank=%u "
              "totalSliceSize=%llu chunkPerPeer=%llu",
              totalRankSize_, xRankSize_, yRankSize_,
              tempAlgParams_.sliceSize,
              (tempAlgParams_.sliceSize + totalRankSize_ - 1) / totalRankSize_);
    if (threads.empty()) {
        return HcclResult::HCCL_E_INTERNAL;
    }

    const u32 dataTypeSize = DATATYPE_SIZE_TABLE[dataType_];

    if (totalRankSize_ == 0) {
        HCCL_ERROR("[InsTempAlltoAllMesh2DV2][LocalDataCopy] totalRankSize_ is 0.");
        return HCCL_E_INTERNAL;
    }

    u64 totalSliceSize = tempAlgParams_.sliceSize;
    u64 cellSize = (totalSliceSize + totalRankSize_ - 1) / totalRankSize_;
    if (cellSize == 0) {
        cellSize = totalSliceSize;
    }

    u64 perPeerMeshSize = (totalSliceSize + xRankSize_ - 1) / xRankSize_;
    if (perPeerMeshSize == 0) {
        perPeerMeshSize = totalSliceSize;
    }

    for (u32 d = 0; d < totalRankSize_; d++) {
        u32 dx = d % xRankSize_;
        u32 dy = d / xRankSize_;

        u64 offsetInSlice = cellSize * d;
        u64 remainingAtOffset = (offsetInSlice < totalSliceSize) ? (totalSliceSize - offsetInSlice) : 0;
        u64 actualChunkSize = std::min(cellSize, remainingAtOffset);
        if (actualChunkSize == 0) {
            continue;
        }
        u64 chunkCount = actualChunkSize / dataTypeSize;

        u64 inOff = tempAlgParams_.buffInfo.inBuffBaseOff + offsetInSlice;

        DataSlice srcSlice(tempAlgParams_.buffInfo.inputPtr, inOff, actualChunkSize, chunkCount);

        u64 outOff = tempAlgParams_.buffInfo.outBuffBaseOff + tempAlgParams_.outputSliceStride * d;
        bool skipOutCopy = (tempAlgParams_.buffInfo.inputPtr == tempAlgParams_.buffInfo.outputPtr &&
                            inOff == outOff);
        bool isScratchToOutput = (tempAlgParams_.buffInfo.inBuffType == BufferType::HCCL_BUFFER &&
                                   tempAlgParams_.buffInfo.outBuffType == BufferType::OUTPUT);
        if (!skipOutCopy && !isScratchToOutput && tempAlgParams_.buffInfo.outBuffType != BufferType::HCCL_BUFFER) {
            DataSlice dstOutSlice(tempAlgParams_.buffInfo.outputPtr, outOff, actualChunkSize, chunkCount);
            LocalCopy(threads[0], srcSlice, dstOutSlice);
        }

        u64 cclOff = tempAlgParams_.buffInfo.hcclBuffBaseOff +
                     perPeerMeshSize * dx + cellSize * dy;
        bool skipCclCopy = (tempAlgParams_.buffInfo.inputPtr == tempAlgParams_.buffInfo.hcclBuff.addr &&
                            inOff == cclOff);
        if (!skipCclCopy) {
            DataSlice cclDstSlice(tempAlgParams_.buffInfo.hcclBuff.addr, cclOff, actualChunkSize, chunkCount);
            LocalCopy(threads[0], srcSlice, cclDstSlice);
        }
    }
    return HcclResult::HCCL_SUCCESS;
}

HcclResult InsTempAlltoAllMesh2DV2::PostLocalCopy(const std::vector<ThreadHandle> &threads)
{
    HCCL_INFO("[InsTempAlltoAllMesh2DV2][PostLocalCopy] Start.");
    if (tempAlgParams_.buffInfo.outBuffType == BufferType::HCCL_BUFFER) {
        HCCL_INFO("[InsTempAlltoAllMesh2DV2][PostLocalCopy] skip because output is scratch");
        return HcclResult::HCCL_SUCCESS;
    }

    const u32 dataTypeSize = DATATYPE_SIZE_TABLE[dataType_];

    if (xRankSize_ == 0 || yRankSize_ == 0 || totalRankSize_ == 0) {
        HCCL_ERROR("[InsTempAlltoAllMesh2DV2][PostLocalCopy] invalid rank sizes.");
        return HCCL_E_INTERNAL;
    }

    u64 totalSliceSize = tempAlgParams_.sliceSize;
    u64 cellSize = (totalSliceSize + totalRankSize_ - 1) / totalRankSize_;
    u64 perPeerSize = (totalSliceSize + xRankSize_ - 1) / xRankSize_;

    for (auto rank : subCommRanks_[0]) {
        if (rank == myRank_) {
            continue;
        }
        u32 algRank = 0;
        CHK_RET(GetAlgRank(rank, subCommRanks_[0], algRank));

        u32 sx = rank % xRankSize_;
        u32 sy = rank / xRankSize_;

        for (u32 dx = 0; dx < xRankSize_; dx++) {
            u32 d = dx + sy * xRankSize_;
            u64 offsetInSlice = cellSize * d;
            u64 remainingAtOffset = (offsetInSlice < totalSliceSize) ? (totalSliceSize - offsetInSlice) : 0;
            u64 actualChunkSize = std::min(cellSize, remainingAtOffset);
            if (actualChunkSize == 0) {
                continue;
            }
            u64 chunkCount = actualChunkSize / dataTypeSize;

            u64 scratchOffset = tempAlgParams_.buffInfo.hcclBuffBaseOff +
                                dx * perPeerSize + sy * cellSize;
            u64 outOffset = tempAlgParams_.buffInfo.outBuffBaseOff +
                            tempAlgParams_.outputSliceStride * d;

            DataSlice srcSlice(tempAlgParams_.buffInfo.hcclBuff.addr, scratchOffset,
                               actualChunkSize, chunkCount);
            DataSlice dstSlice(tempAlgParams_.buffInfo.outputPtr, outOffset,
                               actualChunkSize, chunkCount);
            LocalCopy(threads[0], srcSlice, dstSlice);
        }
    }
    return HcclResult::HCCL_SUCCESS;
}

}  // namespace ops_hccl
