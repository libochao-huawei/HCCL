/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "ins_temp_alltoall_mesh_clos_v2.h"
#include "alg_data_trans_wrapper.h"
#include "template_utils.h"
#include "channel.h"
namespace ops_hccl {

InsTempAlltoAllMeshClosV2::InsTempAlltoAllMeshClosV2(const OpParam &param, const u32 rankId,
                                                     const std::vector<std::vector<u32>> &subCommRanks)
    : InsTempAlltoAllMesh2DV2(param, rankId, subCommRanks)
{
}

InsTempAlltoAllMeshClosV2::~InsTempAlltoAllMeshClosV2() {}

std::string InsTempAlltoAllMeshClosV2::Describe() const
{
    std::string info = "Template of alltoall MeshClosV2 (hash-based link selection) with tempRankSize ";
    info += std::to_string(templateRankSize_);
    return info;
}

u64 InsTempAlltoAllMeshClosV2::GetThreadNum() const
{
    return channelsPerRank_;
}

HcclResult InsTempAlltoAllMeshClosV2::GetRes(AlgResourceRequest &resourceRequest) const
{
    u32 threadNum = GetThreadNum();
    resourceRequest.slaveThreadNum = threadNum > 1 ? threadNum - 1 : 0;
    if (resourceRequest.slaveThreadNum > 0) {
        resourceRequest.notifyNumPerThread.assign(resourceRequest.slaveThreadNum, 1);
    }
    resourceRequest.notifyNumOnMainThread = threadNum > 1 ? threadNum - 1 : 0;
    return HCCL_SUCCESS;
}

HcclResult InsTempAlltoAllMeshClosV2::CalcRes(HcclComm comm, const OpParam &param,
                                               const TopoInfoWithNetLayerDetails *topoInfo,
                                               AlgResourceRequest &resourceRequest)
{
    HCCL_INFO("[InsTempAlltoAllMeshClosV2][CalcRes] start");
    std::vector<HcclChannelDesc> levelChannels;
    CHK_RET(CalcChannelRequestNhr(comm, param, topoInfo, subCommRanks_, levelChannels));
    resourceRequest.channels.push_back(levelChannels);

    channelsPerRank_ = levelChannels.empty() ? 1 : CalcChannelsPerRank(levelChannels);
    HCCL_INFO("[InsTempAlltoAllMeshClosV2][CalcRes] totalLinks[%u], channelCount[%zu]",
              channelsPerRank_, levelChannels.size());

    CHK_RET(GetRes(resourceRequest));
    return HCCL_SUCCESS;
}

HcclResult InsTempAlltoAllMeshClosV2::RunAlltoAllMesh(
    const std::vector<ThreadHandle> &threads,
    const std::map<u32, std::vector<ChannelInfo>> &channels)
{
    HCCL_WARNING("[ALLTOALL_V2_DEBUG][MeshClos][RunAlltoAllMesh] Entry: rank=%d templateRankSize=%u totalLinks=%u "
              "hierarchy: xRank=%u yRank=%u totalRank=%u myXRank=%u myYRank=%u sliceSize=%llu",
              myRank_, templateRankSize_, channelsPerRank_,
              xRankSize_, yRankSize_, totalRankSize_, myXRank_, myYRank_,
              tempAlgParams_.sliceSize);
    if (templateRankSize_ <= 1) {
        return HCCL_SUCCESS;
    }

    for (u32 linkIdx = 0; linkIdx < threads.size(); linkIdx++) {
        CHK_RET(RunAlltoAllOnLink(threads, channels, linkIdx));
    }
    for (u32 i = 0; i < failedRanks_.size(); i++) {
        if (failedRanks_[i]) {
            HCCL_ERROR("[ALLTOALL_V2_DEBUG][MeshClos][RunAlltoAllMesh] Failed rank[%u] detected. "
                       "templateRank=%u myRank=%d totalLinks=%u",
                       i, templateRankSize_, myRank_, channelsPerRank_);
            return HcclResult::HCCL_E_INTERNAL;
        }
    }
    return HCCL_SUCCESS;
}

HcclResult InsTempAlltoAllMeshClosV2::RunAlltoAllOnLink(
    const std::vector<ThreadHandle> &threads,
    const std::map<u32, std::vector<ChannelInfo>> &channels,
    u32 linkIdx)
{
    CHK_PRT_RET(linkIdx >= threads.size(),
                HCCL_ERROR("[ALLTOALL_V2_DEBUG][MeshClos][RunAlltoAllOnLink] linkIdx[%u] >= threads.size()[%zu] "
                           "myRank=%d templateRank=%u",
                           linkIdx, threads.size(), myRank_, templateRankSize_),
                HcclResult::HCCL_E_INTERNAL);

    u32 myAlgRank = 0;
    CHK_RET(GetAlgRank(myRank_, subCommRanks_[0], myAlgRank));

    bool isPcie = IsPcieProtocol(channels);
    const u32 dataTypeSize = DATATYPE_SIZE_TABLE[dataType_];
    u64 totalSliceSize = tempAlgParams_.sliceSize;
    u64 perPeerChunkSize = (totalSliceSize + templateRankSize_ - 1) / templateRankSize_;
    HCCL_WARNING("[ALLTOALL_V2_DEBUG][MeshClos][RunAlltoAllOnLink] Stride config: "
        "inputSliceStride=%llu outputSliceStride=%llu inBuffType=%d outBuffType=%d "
        "inBuffBaseOff=%llu outBuffBaseOff=%llu outputSize=%llu perPeerChunk=%llu",
        tempAlgParams_.inputSliceStride, tempAlgParams_.outputSliceStride,
        static_cast<int>(tempAlgParams_.buffInfo.inBuffType),
        static_cast<int>(tempAlgParams_.buffInfo.outBuffType),
        tempAlgParams_.buffInfo.inBuffBaseOff,
        tempAlgParams_.buffInfo.outBuffBaseOff,
        tempAlgParams_.buffInfo.outputSize,
        perPeerChunkSize);

    // v3.0 Fix B: per-link lastPeerSize for ceiling over-shoot
    u64 lastPeerSize = totalSliceSize - perPeerChunkSize * (templateRankSize_ - 1);
    if (lastPeerSize <= 0) {
        lastPeerSize = perPeerChunkSize;
    }
    u32 lastPeerIndex = templateRankSize_ - 1;

    for (u32 peer = 0; peer < subCommRanks_[0].size(); ++peer) {
        u32 rankInSubcomm = subCommRanks_[0][peer];
        if (rankInSubcomm == myRank_) {
            continue;
        }
        if (channels.count(rankInSubcomm) == 0 || channels.at(rankInSubcomm).empty()) {
            HCCL_ERROR("[ALLTOALL_V2_DEBUG][MeshClos][RunAlltoAllOnLink] Peer %u has no channels. "
                       "myRank=%d linkIdx=%u channels.size=%zu templateRank=%u",
                       rankInSubcomm, myRank_, linkIdx, channels.size(), templateRankSize_);
            return HcclResult::HCCL_E_INTERNAL;
        }
    }

    for (u32 neighborIdx = 0; neighborIdx < subCommRanks_[0].size() - 1; neighborIdx++) {
        u32 connectedRank = subCommRanks_[0][(myAlgRank + 1 + neighborIdx) % subCommRanks_[0].size()];
        u32 connectedAlgRank = 0;
        CHK_RET(GetAlgRank(connectedRank, subCommRanks_[0], connectedAlgRank));

        if (failedRanks_[connectedAlgRank]) {
            HCCL_WARNING("[ALLTOALL_V2_DEBUG][MeshClos] linkIdx[%u] rank[%d] peer[%u] already failed, skipping.",
                      linkIdx, myRank_, connectedRank);
            continue;
        }

        auto it = channels.find(connectedRank);
        if (it == channels.end() || it->second.empty()) {
            HCCL_ERROR("[ALLTOALL_V2_DEBUG][MeshClos][RunAlltoAllOnLink] Rank[%d] connectedRank[%u] has no channels. "
                       "linkIdx=%u channels.size=%zu templateRank=%u",
                       myRank_, connectedRank, linkIdx, channels.size(), templateRankSize_);
            return HcclResult::HCCL_E_INTERNAL;
        }

        u32 totalLinksToNeighbor = it->second.size();
        u32 selectedLinkIdx = (myAlgRank + connectedAlgRank) % threads.size();

        if (selectedLinkIdx >= it->second.size()) {
            HCCL_ERROR("[ALLTOALL_V2_DEBUG][MeshClos][RunAlltoAllOnLink] selectedLinkIdx OOB: "
                       "selectedLinkIdx=%u >= channels[%u].size()=%zu. myRank=%d connectedRank=%u",
                       selectedLinkIdx, connectedRank, it->second.size(), myRank_, connectedRank);
            continue;
        }

        if (!it->second[selectedLinkIdx].remoteCclMem.addr) {
            HCCL_ERROR("[ALLTOALL_V2_DEBUG][MeshClos][RunAlltoAllOnLink] remoteCclMem.addr is NULL at selectedLinkIdx: "
                       "selectedLinkIdx=%u connectedRank=%u myRank=%d",
                       selectedLinkIdx, connectedRank, myRank_);
            continue;
        }

        if (selectedLinkIdx != linkIdx) {
            continue;
        }

        HCCL_WARNING("[ALLTOALL_V2_DEBUG][MeshClos][RunAlltoAllOnLink] linkIdx[%u] matched: "
                  "myRank=%d connectedRank=%u selectedLinkIdx=%u/%u threads=%zu "
                  "enableRemoteMemAccess=%d isPcie=%d",
                  linkIdx, myRank_, connectedRank, selectedLinkIdx,
                  totalLinksToNeighbor, threads.size(),
                  enableRemoteMemAccess_, isPcie);

        if (linkIdx >= channels.at(connectedRank).size()) {
            HCCL_ERROR("[ALLTOALL_V2_DEBUG][MeshClos][RunAlltoAllOnLink] channel index OOB: linkIdx[%u] >= "
                       "channels[%u].size()=%zu. myRank=%d templateRank=%u",
                       linkIdx, connectedRank, channels.at(connectedRank).size(), myRank_, templateRankSize_);
            return HCCL_E_INTERNAL;
        }

        const ChannelInfo &linkRemote = it->second[linkIdx];
        void *remoteCclBuffAddr = linkRemote.remoteCclMem.addr;
        if (!remoteCclBuffAddr) {
            HCCL_ERROR("[ALLTOALL_V2_DEBUG][MeshClos][RunAlltoAllOnLink] remoteCclMem.addr is NULL for peer %u. "
                       "myRank=%d connectedRank=%u linkIdx=%u templateRank=%u",
                       connectedRank, myRank_, connectedRank, linkIdx, templateRankSize_);
            return HCCL_E_INTERNAL;
        }

        std::vector<DataSlice> txSrcSlicesAll;
        std::vector<DataSlice> txDstSlicesAll;
        std::vector<DataSlice> rxDstSlicesAll;
        std::vector<DataSlice> rxSrcSlicesAll;

        u64 actualChunkSize = (connectedAlgRank == lastPeerIndex) ? lastPeerSize : perPeerChunkSize;
        u64 offsetInSlice = connectedAlgRank * perPeerChunkSize;
        u64 remainingAtOffset = (offsetInSlice < totalSliceSize) ? (totalSliceSize - offsetInSlice) : 0;
        actualChunkSize = std::min(actualChunkSize, remainingAtOffset);
        if (actualChunkSize == 0) {
            continue;
        }
        u64 chunkCount = actualChunkSize / dataTypeSize;

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
                    HCCL_ERROR("[ALLTOALL_V2_DEBUG][MeshClos] RX destination OOB! "
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

            HCCL_WARNING("[ALLTOALL_V2_DEBUG][MeshClos][RunAlltoAllOnLink] rank[%d]->peer[%d] linkIdx[%u] rpt[%u] "
                      "txSrcOff=%llu txDstOff=%llu rxSrcOff=%llu rxDstOff=%llu "
                      "actualSz=%llu fromScratch=%d",
                      myRank_, connectedRank, linkIdx, rpt,
                      txSrcOffset, txDstOffset, rxSrcOffset, rxOutOffset,
                      actualChunkSize, readingFromScratch);
        }

        TxRxSlicesList sendRecvSlicesList({txSrcSlicesAll, txDstSlicesAll},
                                          {rxSrcSlicesAll, rxDstSlicesAll});
        TxRxChannels sendRecvChannels(linkRemote, linkRemote);
        SendRecvInfo sendRecvInfo(sendRecvChannels, sendRecvSlicesList);

        HcclResult dmaResult;
        if (isPcie) {
            dmaResult = SendRecvRead(sendRecvInfo, threads[linkIdx]);
        } else {
            dmaResult = SendRecvWrite(sendRecvInfo, threads[linkIdx]);
        }

        if (dmaResult == HcclResult::HCCL_E_INTERNAL) {
            failedRanks_[connectedAlgRank] = 1;
            HCCL_WARNING("[ALLTOALL_V2_DEBUG][MeshClos] linkIdx[%u] peer %u timed out. "
                         "myRank=%d templateRank=%u",
                         linkIdx, connectedRank, myRank_, templateRankSize_);
            continue;
        }

        if (dmaResult != HCCL_SUCCESS) {
            HCCL_ERROR("[ALLTOALL_V2_DEBUG][MeshClos] send/recv FAILED: linkIdx=%u connectedRank=%u err=0x%x "
                       "myRank=%d templateRank=%u actualChunkSize=%llu",
                       linkIdx, connectedRank, dmaResult, myRank_, templateRankSize_, actualChunkSize);
            return dmaResult;
        }

        // v1.17 FIX: Removed HcommFenceOnThread(threads[linkIdx]) — not supported
        // on AICPU (returns HCCL_E_NOT_SUPPORT=5) and unnecessary in the clos ring
        // because each peer maps to a unique linkIdx via (myAlgRank+connectedAlgRank)%threads.size(),
        // and SendRecvWrite/SendRecvRead already provide notify-based DMA serialization.
    }

    return HCCL_SUCCESS;
}

HcclResult InsTempAlltoAllMeshClosV2::LocalDataCopy(const std::vector<ThreadHandle> &threads)
{
    if (threads.empty()) {
        return HcclResult::HCCL_E_INTERNAL;
    }

    const u32 dataTypeSize = DATATYPE_SIZE_TABLE[dataType_];

    if (totalRankSize_ == 0 || yRankSize_ == 0) {
        HCCL_ERROR("[ALLTOALL_V2_DEBUG][MeshClos][LocalDataCopy] invalid rank sizes: totalRank=%u xRank=%u yRank=%u",
                   totalRankSize_, xRankSize_, yRankSize_);
        return HCCL_E_INTERNAL;
    }

    u64 totalSliceSize = tempAlgParams_.sliceSize;
    u64 cellSize = (totalSliceSize + totalRankSize_ - 1) / totalRankSize_;
    if (cellSize == 0) {
        cellSize = totalSliceSize;
    }

    u64 perPeerClosSize = (totalSliceSize + yRankSize_ - 1) / yRankSize_;
    if (perPeerClosSize == 0) {
        perPeerClosSize = totalSliceSize;
    }

    HCCL_WARNING("[ALLTOALL_V2_DEBUG][MeshClos][LocalDataCopy] Start: totalRank=%u xRank=%u yRank=%u "
              "totalSlice=%llu cellSize=%llu perPeerClos=%llu myXRank=%u myYRank=%u",
              totalRankSize_, xRankSize_, yRankSize_,
              tempAlgParams_.sliceSize, cellSize, perPeerClosSize,
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
        u32 sx = d % xRankSize_;
        u32 sy = d / xRankSize_;

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
                     perPeerClosSize * sy + cellSize * sx;
        bool skipCclCopy = (tempAlgParams_.buffInfo.inputPtr == tempAlgParams_.buffInfo.hcclBuff.addr &&
                            inOff == cclOff);
        if (!skipCclCopy) {
            // v1.14 C-R2-4 fix: protect against null hcclBuff.addr that
            // produces a fake non-null address when offset is added
            if (!tempAlgParams_.buffInfo.hcclBuff.addr) {
                HCCL_ERROR("[ALLTOALL_V2_DEBUG][MeshClos][LocalDataCopy] hcclBuff.addr is NULL. "
                           "d=%u sx=%u sy=%u myXRank=%u myYRank=%u myRank=%d",
                           d, sx, sy, myXRank_, myYRank_, myRank_);
                return HCCL_E_INTERNAL;
            }
            DataSlice cclDstSlice(tempAlgParams_.buffInfo.hcclBuff.addr, cclOff, actualChunkSize, chunkCount);
            LocalCopy(threads[0], srcSlice, cclDstSlice);
        }
    }
    return HcclResult::HCCL_SUCCESS;
}

HcclResult InsTempAlltoAllMeshClosV2::PostLocalCopy(const std::vector<ThreadHandle> &threads)
{
    if (tempAlgParams_.buffInfo.outBuffType == BufferType::HCCL_BUFFER) {
        HCCL_WARNING("[ALLTOALL_V2_DEBUG][MeshClos][PostLocalCopy] skip because output is scratch");
        return HcclResult::HCCL_SUCCESS;
    }

    const u32 dataTypeSize = DATATYPE_SIZE_TABLE[dataType_];

    if (xRankSize_ == 0 || yRankSize_ == 0 || totalRankSize_ == 0) {
        HCCL_ERROR("[ALLTOALL_V2_DEBUG][MeshClos][PostLocalCopy] invalid rank sizes: xRank=%u yRank=%u totalRank=%u",
                   xRankSize_, yRankSize_, totalRankSize_);
        return HCCL_E_INTERNAL;
    }

    u64 totalSliceSize = tempAlgParams_.sliceSize;
    u64 cellSize = (totalSliceSize + totalRankSize_ - 1) / totalRankSize_;
    u64 perPeerSize = (totalSliceSize + yRankSize_ - 1) / yRankSize_;
    u64 perPeerOutputStride = (tempAlgParams_.buffInfo.outputSize + totalRankSize_ - 1) / totalRankSize_;

    HCCL_WARNING("[ALLTOALL_V2_DEBUG][MeshClos][PostLocalCopy] Start: templateRank=%u xRank=%u yRank=%u totalRank=%u "
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
                                sy * perPeerSize + dx * cellSize;
            u64 outOffset = tempAlgParams_.buffInfo.outBuffBaseOff +
                            perPeerOutputStride * d;

            // v1.12 Fix D: OOB guard for PostLocalCopy output writes
            u64 maxWritePos = outOffset + actualChunkSize;
            if (maxWritePos > tempAlgParams_.buffInfo.outputSize) {
                HCCL_ERROR("[ALLTOALL_V2_DEBUG][MeshClos][PostLocalCopy] Output OOB! "
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
                HCCL_ERROR("[ALLTOALL_V2_DEBUG][MeshClos][PostLocalCopy] hcclBuff.addr is NULL. "
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
