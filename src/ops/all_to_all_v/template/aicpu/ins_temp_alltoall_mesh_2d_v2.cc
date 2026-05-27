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
    HCCL_WARNING("[ALLTOALL_V2_DEBUG][Mesh2D][KernelRun] Entry: templateRankSize=%u myRank=%u "
              "xRank=%u yRank=%u totalRank=%u myXRank=%u myYRank=%u "
              "sliceSize=%llu threadNum=%zu inBuffBase=%llu outBuffBase=%llu hcclBase=%llu "
              "inBuffType=%d outBuffType=%d",
              templateRankSize_, myRank_, xRankSize_, yRankSize_, totalRankSize_,
              myXRank_, myYRank_,
              tempAlgParams.sliceSize, templateResource.threads.size(),
              tempAlgParams.buffInfo.inBuffBaseOff,
              tempAlgParams.buffInfo.outBuffBaseOff,
              tempAlgParams.buffInfo.hcclBuffBaseOff,
              static_cast<int>(tempAlgParams.buffInfo.inBuffType),
              static_cast<int>(tempAlgParams.buffInfo.outBuffType));

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
    u32 dataTypeSize = DATATYPE_SIZE_TABLE[param.all2AllVDataDes.sendType];
    u64 localPerPeerChunkSize = (tempAlgParams.sliceSize + totalRankSize_ - 1) / totalRankSize_;
    if (localPerPeerChunkSize < static_cast<u64>(dataTypeSize) && tempAlgParams.sliceSize > 0) {
        HCCL_WARNING("[ALLTOALL_V2_DEBUG][Mesh2D][KernelRun] perPeerChunkSize[%llu] < dataTypeSize[%u], fallback to local-only. "
                  "sliceSize=%llu totalRank=%u",
                  localPerPeerChunkSize, dataTypeSize, tempAlgParams.sliceSize, totalRankSize_);
        // Temporarily override inBuffType to INPUT so isScratchToOutput guard
        // in LocalDataCopy allows output copy (needed when fallback fires in Stage 2
        // where inBuffType=HCCL_BUFFER would otherwise block the output write)
        auto savedInBuffType = tempAlgParams_.buffInfo.inBuffType;
        tempAlgParams_.buffInfo.inBuffType = BufferType::INPUT;
        CHK_RET(LocalDataCopy(templateResource.threads));
        tempAlgParams_.buffInfo.inBuffType = savedInBuffType;
        return HCCL_SUCCESS;
    }
    dataType_ = param.all2AllVDataDes.sendType;
    HCCL_INFO("[InsTempAlltoAllMesh2DV2] Rank [%d], get threadNum_[%d].", myRank_, threadNum_);

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
                HCCL_ERROR("[ALLTOALL_V2_DEBUG][Mesh2D] Ring exchange failed: slave[%zu] err=0x%x. "
                           "templateRank=%u myRank=%u sliceSize=%llu",
                           i, slaveErrs_[i], templateRankSize_, myRank_, tempAlgParams_.sliceSize);
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
    HCCL_WARNING("[ALLTOALL_V2_DEBUG][Mesh2D][RunAlltoAllMesh] Stride config: "
        "inputSliceStride=%llu outputSliceStride=%llu inBuffType=%d outBuffType=%d "
        "inBuffBaseOff=%llu outBuffBaseOff=%llu outputSize=%llu perPeerChunk=%llu",
        tempAlgParams_.inputSliceStride, tempAlgParams_.outputSliceStride,
        static_cast<int>(tempAlgParams_.buffInfo.inBuffType),
        static_cast<int>(tempAlgParams_.buffInfo.outBuffType),
        tempAlgParams_.buffInfo.inBuffBaseOff,
        tempAlgParams_.buffInfo.outBuffBaseOff,
        tempAlgParams_.buffInfo.outputSize,
        perPeerChunkSize);
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
                    HCCL_ERROR("[ALLTOALL_V2_DEBUG][Mesh2D][RunAlltoAllMesh] Channel validation FAILED: "
                               "myRank=%u neighborIdx=%u threads=%zu connectedRank=%u channels.has=%d",
                               myRank_, neighborIdx, threads.size(), connectedRank,
                               channels.count(connectedRank)),
                    HcclResult::HCCL_E_INTERNAL);

        const ChannelInfo &linkRemote = channels.at(connectedRank)[0];
        void *remoteCclBuffAddr = linkRemote.remoteCclMem.addr;
        if (!remoteCclBuffAddr) {
            HCCL_ERROR("[ALLTOALL_V2_DEBUG][Mesh2D][RunAlltoAllMesh] remoteCclMem.addr is NULL for peer %u. "
                       "myRank=%u connectedRank=%u templateRank=%u",
                       connectedRank, myRank_, connectedRank, templateRankSize_);
            return HCCL_E_INTERNAL;
        }

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

            bool readingFromScratch = (tempAlgParams_.buffInfo.inBuffType == BufferType::HCCL_BUFFER);

            u64 actualInputStride = (tempAlgParams_.inputSliceStride != 0)
                ? tempAlgParams_.inputSliceStride : tempAlgParams_.outputSliceStride;
            u64 txSrcInputOffset = tempAlgParams_.buffInfo.inBuffBaseOff +
                                   actualInputStride * connectedAlgRank;

            u64 txSrcScratchOffset = tempAlgParams_.buffInfo.inBuffBaseOff +
                                     perPeerChunkSize * connectedAlgRank;
            u64 txScratchOffset = scratchBase + perPeerChunkSize * connectedAlgRank;
            u64 txDstOffset = (!enableRemoteMemAccess_) ? txScratchOffset : txSrcInputOffset;

            u64 rxOutOffset = tempAlgParams_.outputSliceStride * myAlgRank + outBaseOff;
            // v1.12 Fix C: extend OOB guard to BufferType::OUTPUT (Stage 2 writes to user output buffer)
            if (tempAlgParams_.buffInfo.outBuffType == BufferType::HCCL_BUFFER ||
                tempAlgParams_.buffInfo.outBuffType == BufferType::OUTPUT) {
                u64 maxRxWritePos = rxOutOffset + actualChunkSize;
                if (maxRxWritePos > tempAlgParams_.buffInfo.outputSize) {
                    HCCL_ERROR("[ALLTOALL_V2_DEBUG][Mesh2D] RX destination OOB! "
                        "rxOutOffset=%llu + actualChunkSize=%llu = %llu > outputSize=%llu "
                        "myAlgRank=%u myRank=%d outBuffBaseOff=%llu outputSliceStride=%llu",
                        rxOutOffset, actualChunkSize, maxRxWritePos,
                        tempAlgParams_.buffInfo.outputSize, myAlgRank, myRank_,
                        tempAlgParams_.buffInfo.outBuffBaseOff,
                        tempAlgParams_.outputSliceStride);
                    return HcclResult::HCCL_E_INTERNAL;
                }
            }
            u64 rxScratchOffset = scratchBase + perPeerChunkSize * myAlgRank;
            u64 rxSrcOffset = (!enableRemoteMemAccess_) ? rxScratchOffset : rxOutOffset;

            void *txSrcPtr = tempAlgParams_.buffInfo.inputPtr;
            void *txDstPtr = (!enableRemoteMemAccess_) ? remoteCclBuffAddr
                                                          : linkRemote.remoteOutputGraphMode.addr;
            void *rxSrcPtr = (!enableRemoteMemAccess_) ? remoteCclBuffAddr
                                                          : linkRemote.remoteOutputGraphMode.addr;
            void *rxDstPtr = tempAlgParams_.buffInfo.outputPtr;

            u64 txSrcOffset = readingFromScratch ? txSrcScratchOffset : txSrcInputOffset;

            txSrcSlicesAll.emplace_back(txSrcPtr, txSrcOffset, actualChunkSize, chunkCount);
            txDstSlicesAll.emplace_back(txDstPtr, txDstOffset, actualChunkSize, chunkCount);
            rxDstSlicesAll.emplace_back(rxDstPtr, rxOutOffset, actualChunkSize, chunkCount);
            rxSrcSlicesAll.emplace_back(rxSrcPtr, rxSrcOffset, actualChunkSize, chunkCount);

            HCCL_WARNING("[ALLTOALL_V2_DEBUG][Mesh2D][RunAlltoAllMesh] rank[%d]->peer[%d] rpt[%u] "
                      "txSrcOff=%llu txDstOff=%llu rxSrcOff=%llu rxDstOff=%llu "
                      "actualSz=%llu fromScratch=%d",
                      myRank_, connectedRank, rpt,
                      txSrcOffset, txDstOffset, rxSrcOffset, rxOutOffset,
                      actualChunkSize, readingFromScratch);
        }

        TxRxSlicesList sendRecvSlicesList({txSrcSlicesAll, txDstSlicesAll},
                                          {rxSrcSlicesAll, rxDstSlicesAll});
        TxRxChannels sendRecvChannels(linkRemote, linkRemote);
        SendRecvInfo sendRecvInfo(sendRecvChannels, sendRecvSlicesList);

        HCCL_WARNING("[ALLTOALL_V2_DEBUG][Mesh2D][RunAlltoAllMesh] round[%u/%zu] connectedRank=%u connectedAlgRank=%u "
                  "actualChunkSize=%llu chunkCount=%llu isPcie=%d repeatNum=%u",
                  neighborIdx, subCommRanks_[0].size() - 1,
                  connectedRank, connectedAlgRank, actualChunkSize, chunkCount, isPcie,
                  tempAlgParams_.repeatNum);

        HcclResult dmaResult;
        if (isPcie) {
            dmaResult = SendRecvRead(sendRecvInfo, threads[neighborIdx]);
        } else {
            dmaResult = SendRecvWrite(sendRecvInfo, threads[neighborIdx]);
        }

        if (dmaResult == HcclResult::HCCL_E_INTERNAL) {
            failedRanks_[connectedAlgRank] = 1;
            HCCL_WARNING("[ALLTOALL_V2_DEBUG][Mesh2D] Ring round %d: peer %u timed out. "
                         "templateRank=%u myRank=%u myAlgRank=%u",
                         neighborIdx, connectedRank, templateRankSize_, myRank_, myAlgRank);
            continue;
        }

        if (dmaResult != HCCL_SUCCESS) {
            HCCL_ERROR("[ALLTOALL_V2_DEBUG][Mesh2D] RunAlltoAllMesh send/recv FAILED: "
                       "connectedRank=%u connectedAlgRank=%u round=%u err=0x%x",
                       connectedRank, connectedAlgRank, neighborIdx, dmaResult);
            return dmaResult;
        }
    }

    for (u32 i = 0; i < failedRanks_.size(); i++) {
        if (failedRanks_[i]) {
            HCCL_ERROR("[ALLTOALL_V2_DEBUG][Mesh2D][RunAlltoAllMesh] Failed rank[%u] detected. "
                       "templateRank=%u myRank=%u totalRounds=%zu",
                       i, templateRankSize_, myRank_, subCommRanks_[0].size() - 1);
            return HcclResult::HCCL_E_INTERNAL;
        }
    }

    return HcclResult::HCCL_SUCCESS;
}

HcclResult InsTempAlltoAllMesh2DV2::LocalDataCopy(const std::vector<ThreadHandle> &threads)
{
    if (threads.empty()) {
        return HcclResult::HCCL_E_INTERNAL;
    }

    const u32 dataTypeSize = DATATYPE_SIZE_TABLE[dataType_];

    if (totalRankSize_ == 0) {
        HCCL_ERROR("[ALLTOALL_V2_DEBUG][Mesh2D][LocalDataCopy] totalRankSize_ is 0.");
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

    HCCL_WARNING("[ALLTOALL_V2_DEBUG][Mesh2D][LocalDataCopy] Start: totalRank=%u xRank=%u yRank=%u "
              "totalSlice=%llu chunkPerPeer=%llu cellSize=%llu perPeerMesh=%llu myXRank=%u myYRank=%u",
              totalRankSize_, xRankSize_, yRankSize_,
              tempAlgParams_.sliceSize,
              (tempAlgParams_.sliceSize + totalRankSize_ - 1) / totalRankSize_,
              cellSize, perPeerMeshSize,
              myXRank_, myYRank_);

    bool readingFromInput = (tempAlgParams_.buffInfo.inBuffType == BufferType::INPUT &&
                              !tempAlgParams_.sendCounts.empty());
    u64 perPeerInputChunkSize = 0;
    if (readingFromInput) {
        u64 totalCount = 0;
        for (auto c : tempAlgParams_.sendCounts) {
            totalCount += c;
        }
        perPeerInputChunkSize = (totalCount > 0 && totalRankSize_ > 0)
            ? (totalCount * dataTypeSize / totalRankSize_) : 0;
    }

    for (u32 d = 0; d < totalRankSize_; d++) {
        u32 dx = d % xRankSize_;
        u32 dy = d / xRankSize_;

        u64 inOff;
        u64 actualChunkSize;
        u64 chunkCount;

        if (readingFromInput) {
            u64 peerStartInInput = perPeerInputChunkSize * d;
            u64 inputBase = tempAlgParams_.buffInfo.inBuffBaseOff;
            inOff = inputBase + peerStartInInput;
            u64 sliceEndInInput = inputBase + totalSliceSize;
            u64 available = (inOff < sliceEndInInput)
                ? std::min(sliceEndInInput - inOff, perPeerInputChunkSize) : 0;
            actualChunkSize = std::min(cellSize, available);
        } else {
            u64 offsetInSlice = cellSize * d;
            u64 remainingAtOffset = (offsetInSlice < totalSliceSize) ? (totalSliceSize - offsetInSlice) : 0;
            actualChunkSize = std::min(cellSize, remainingAtOffset);
            inOff = tempAlgParams_.buffInfo.inBuffBaseOff + offsetInSlice;
        }
        if (actualChunkSize == 0) {
            continue;
        }
        chunkCount = actualChunkSize / dataTypeSize;

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
            // v1.14 C-R2-4 fix: protect against null hcclBuff.addr that
            // produces a fake non-null address when offset is added
            if (!tempAlgParams_.buffInfo.hcclBuff.addr) {
                HCCL_ERROR("[ALLTOALL_V2_DEBUG][Mesh2D][LocalDataCopy] hcclBuff.addr is NULL. "
                           "d=%u dx=%u dy=%u myXRank=%u myYRank=%u myRank=%u",
                           d, dx, dy, myXRank_, myYRank_, myRank_);
                return HCCL_E_INTERNAL;
            }
            DataSlice cclDstSlice(tempAlgParams_.buffInfo.hcclBuff.addr, cclOff, actualChunkSize, chunkCount);
            LocalCopy(threads[0], srcSlice, cclDstSlice);
        }
    }
    return HcclResult::HCCL_SUCCESS;
}

HcclResult InsTempAlltoAllMesh2DV2::PostLocalCopy(const std::vector<ThreadHandle> &threads)
{
    if (tempAlgParams_.buffInfo.outBuffType == BufferType::HCCL_BUFFER) {
        HCCL_WARNING("[ALLTOALL_V2_DEBUG][Mesh2D][PostLocalCopy] skip because output is scratch");
        return HcclResult::HCCL_SUCCESS;
    }

    const u32 dataTypeSize = DATATYPE_SIZE_TABLE[dataType_];

    if (xRankSize_ == 0 || yRankSize_ == 0 || totalRankSize_ == 0) {
        HCCL_ERROR("[ALLTOALL_V2_DEBUG][Mesh2D][PostLocalCopy] invalid rank sizes: xRank=%u yRank=%u totalRank=%u",
                   xRankSize_, yRankSize_, totalRankSize_);
        return HCCL_E_INTERNAL;
    }

    u64 totalSliceSize = tempAlgParams_.sliceSize;
    u64 cellSize = (totalSliceSize + totalRankSize_ - 1) / totalRankSize_;
    u64 perPeerSize = (totalSliceSize + xRankSize_ - 1) / xRankSize_;
    u64 perPeerOutputStride = (tempAlgParams_.buffInfo.outputSize + totalRankSize_ - 1) / totalRankSize_;

    HCCL_WARNING("[ALLTOALL_V2_DEBUG][Mesh2D][PostLocalCopy] Start: templateRank=%u xRank=%u yRank=%u totalRank=%u "
              "sliceSize=%llu cellSize=%llu perPeerSize=%llu",
              templateRankSize_, xRankSize_, yRankSize_, totalRankSize_,
              tempAlgParams_.sliceSize, cellSize, perPeerSize);

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
                            perPeerOutputStride * d;

            // v1.12 Fix D: OOB guard for PostLocalCopy output writes
            u64 maxWritePos = outOffset + actualChunkSize;
            if (maxWritePos > tempAlgParams_.buffInfo.outputSize) {
                HCCL_ERROR("[ALLTOALL_V2_DEBUG][Mesh2D][PostLocalCopy] Output OOB! "
                    "outOffset=%llu + actualChunkSize=%llu = %llu > outputSize=%llu "
                    "d=%u rank=%d outBuffBaseOff=%llu outputSliceStride=%llu",
                    outOffset, actualChunkSize, maxWritePos,
                    tempAlgParams_.buffInfo.outputSize, d, myRank_,
                    tempAlgParams_.buffInfo.outBuffBaseOff,
                    tempAlgParams_.outputSliceStride);
                return HCCL_E_INTERNAL;
            }

            // v1.14 C-R2-4 fix: protect against null hcclBuff.addr
            if (!tempAlgParams_.buffInfo.hcclBuff.addr) {
                HCCL_ERROR("[ALLTOALL_V2_DEBUG][Mesh2D][PostLocalCopy] hcclBuff.addr is NULL. "
                           "d=%u rank=%d outBuffBaseOff=%llu",
                           d, myRank_, tempAlgParams_.buffInfo.outBuffBaseOff);
                return HCCL_E_INTERNAL;
            }
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
