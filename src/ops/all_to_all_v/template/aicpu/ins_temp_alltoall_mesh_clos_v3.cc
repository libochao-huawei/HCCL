/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "ins_temp_alltoall_mesh_clos_v3.h"
#include "alg_data_trans_wrapper.h"
#include "template_utils.h"
#include "channel.h"
#include <cstdlib>

namespace ops_hccl {

namespace {
constexpr float kDefaultSharedRatio = 0.2f;
constexpr u64 kHcclMinSliceAlign = 128;

float GetSharedLinkRatio()
{
    const char *env = std::getenv("HCCL_CLOS_SHARED_LINK_RATIO");
    if (env != nullptr) {
        float ratio = std::atof(env);
        if (ratio > 0.0f && ratio < 1.0f) {
            return ratio;
        }
    }
    return kDefaultSharedRatio;
}
}  // namespace

InsTempAlltoAllMeshClosV3::InsTempAlltoAllMeshClosV3(const OpParam &param, const u32 rankId,
                                                      const std::vector<std::vector<u32>> &subCommRanks)
    : InsTempAlltoAllMesh2DV3(param, rankId, subCommRanks)
{
}

InsTempAlltoAllMeshClosV3::~InsTempAlltoAllMeshClosV3() {}

u64 InsTempAlltoAllMeshClosV3::GetThreadNum() const
{
    return channelsPerRank_;
}

HcclResult InsTempAlltoAllMeshClosV3::GetRes(AlgResourceRequest &resourceRequest) const
{
    u32 threadNum = GetThreadNum();
    resourceRequest.slaveThreadNum = threadNum > 1 ? threadNum - 1 : 0;
    if (resourceRequest.slaveThreadNum > 0) {
        resourceRequest.notifyNumPerThread.assign(resourceRequest.slaveThreadNum, 1);
    }
    resourceRequest.notifyNumOnMainThread = threadNum > 1 ? threadNum - 1 : 0;
    return HCCL_SUCCESS;
}

HcclResult InsTempAlltoAllMeshClosV3::CalcRes(HcclComm comm, const OpParam &param,
                                               const TopoInfoWithNetLayerDetails *topoInfo,
                                               AlgResourceRequest &resourceRequest)
{
    HCCL_INFO("[InsTempAlltoAllMeshClosV3][CalcRes] start");
    std::vector<HcclChannelDesc> levelChannels;
    CHK_RET(CalcChannelRequestNhr(comm, param, topoInfo, subCommRanks_, levelChannels));
    resourceRequest.channels.push_back(levelChannels);

    channelsPerRank_ = levelChannels.empty() ? 1 : CalcChannelsPerRank(levelChannels);
    HCCL_INFO("[InsTempAlltoAllMeshClosV3][CalcRes] totalLinks[%u], channelCount[%zu]",
              channelsPerRank_, levelChannels.size());

    CHK_RET(GetRes(resourceRequest));
    return HCCL_SUCCESS;
}

bool InsTempAlltoAllMeshClosV3::IsSharedLink(u32 linkIdx) const
{
    return (linkIdx == 0 && sharedPortMode_);
}

HcclResult InsTempAlltoAllMeshClosV3::RunAlltoAllMesh(
    const std::vector<ThreadHandle> &threads,
    const std::map<u32, std::vector<ChannelInfo>> &channels)
{
    HCCL_INFO("[InsTempAlltoAllMeshClosV3][RunAlltoAllMesh] Rank[%d] templateRankSize[%u] "
              "channelsPerRank[%u] sharedPort[%d].",
              myRank_, templateRankSize_, channelsPerRank_, sharedPortMode_);
    if (templateRankSize_ <= 1) {
        return HCCL_SUCCESS;
    }

    for (u32 linkIdx = 0; linkIdx < threads.size(); linkIdx++) {
        CHK_RET(RunAlltoAllOnLink(threads, channels, linkIdx));
    }
    for (u32 i = 0; i < failedRanks_.size(); i++) {
        if (failedRanks_[i]) {
            return HcclResult::HCCL_E_INTERNAL;
        }
    }
    return HCCL_SUCCESS;
}

HcclResult InsTempAlltoAllMeshClosV3::RunAlltoAllOnLink(
    const std::vector<ThreadHandle> &threads,
    const std::map<u32, std::vector<ChannelInfo>> &channels,
    u32 linkIdx)
{
    CHK_PRT_RET(linkIdx >= threads.size(),
                HCCL_ERROR("[InsTempAlltoAllMeshClosV3] linkIdx[%u] >= threads.size()[%zu].",
                           linkIdx, threads.size()),
                HcclResult::HCCL_E_INTERNAL);

    u32 myAlgRank = 0;
    CHK_RET(GetAlgRank(myRank_, subCommRanks_[0], myAlgRank));

    bool isPcie = IsPcieProtocol(channels);
    const u32 dataTypeSize = DATATYPE_SIZE_TABLE[dataType_];
    u64 totalSliceSize = tempAlgParams_.sliceSize;
    u64 perPeerChunkSize = (totalSliceSize + templateRankSize_ - 1) / templateRankSize_;

    u64 lastPeerSize = totalSliceSize - perPeerChunkSize * (templateRankSize_ - 1);
    if (lastPeerSize <= 0) {
        lastPeerSize = perPeerChunkSize;
    }
    u32 lastPeerIndex = templateRankSize_ - 1;

    const bool linkIsShared = IsSharedLink(linkIdx);
    const float sharedRatio = GetSharedLinkRatio();
    u32 nNeighbors = subCommRanks_[0].size();

    for (u32 neighborIdx = 0; neighborIdx < nNeighbors - 1; neighborIdx++) {
        u32 connectedRank = subCommRanks_[0][(myAlgRank + 1 + neighborIdx) % nNeighbors];
        u32 connectedAlgRank = 0;
        CHK_RET(GetAlgRank(connectedRank, subCommRanks_[0], connectedAlgRank));

        if (failedRanks_[connectedAlgRank]) {
            continue;
        }

        auto it = channels.find(connectedRank);
        if (it == channels.end() || it->second.empty()) {
            HCCL_ERROR("[InsTempAlltoAllMeshClosV3] Rank[%d] connectedRank[%u] has no channels.",
                       myRank_, connectedRank);
            return HcclResult::HCCL_E_INTERNAL;
        }

        if (!linkIsShared) {
            u32 selectedLinkIdx;
            u32 totalRanks = nNeighbors;
            bool isPowerOfTwo = ((totalRanks & (totalRanks - 1)) == 0);

            if (isPowerOfTwo) {
                selectedLinkIdx = myAlgRank ^ connectedAlgRank;
            } else {
                u32 nDedicated = threads.size() - 1;
                selectedLinkIdx = (myAlgRank ^ connectedAlgRank) % nDedicated + 1;
            }

            if (selectedLinkIdx >= channelsPerRank_) { continue; }

            if (selectedLinkIdx != linkIdx) {
                continue;
            }
        }

        u64 actualChunkSize = (connectedAlgRank == lastPeerIndex) ? lastPeerSize : perPeerChunkSize;
        u64 offsetInSlice = connectedAlgRank * perPeerChunkSize;
        u64 remainingAtOffset = (offsetInSlice < totalSliceSize) ? (totalSliceSize - offsetInSlice) : 0;
        actualChunkSize = std::min(actualChunkSize, remainingAtOffset);
        if (actualChunkSize == 0) {
            continue;
        }

        const ChannelInfo &linkRemote = it->second[linkIdx];
        void *remoteCclBuffAddr = linkRemote.remoteCclMem.addr;

        u64 chunkSize;
        u64 srcOffset;
        u64 sharedChunkSize = 0;

        if (linkIsShared) {
            sharedChunkSize = (actualChunkSize * sharedRatio) / dataTypeSize * dataTypeSize;
            sharedChunkSize = (sharedChunkSize / kHcclMinSliceAlign) * kHcclMinSliceAlign;
            if (sharedChunkSize == 0 && actualChunkSize > 0) {
                sharedChunkSize = dataTypeSize;
            }

            srcOffset = connectedAlgRank * perPeerChunkSize +
                        (actualChunkSize - sharedChunkSize);
            chunkSize = sharedChunkSize;
        } else {
            u64 mainChunkSize = actualChunkSize * (1.0f - sharedRatio);
            mainChunkSize = (mainChunkSize / dataTypeSize) * dataTypeSize;
            mainChunkSize = (mainChunkSize / kHcclMinSliceAlign) * kHcclMinSliceAlign;
            if (mainChunkSize == 0) {
                mainChunkSize = actualChunkSize;
            }

            srcOffset = connectedAlgRank * perPeerChunkSize;
            chunkSize = mainChunkSize;
        }

        if (chunkSize == 0) {
            continue;
        }
        u64 chunkCount = chunkSize / dataTypeSize;

        std::vector<DataSlice> txSrcSlicesAll;
        std::vector<DataSlice> txDstSlicesAll;
        std::vector<DataSlice> rxDstSlicesAll;
        std::vector<DataSlice> rxSrcSlicesAll;

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

            u64 txLocalOffset;
            if (readingFromScratch) {
                txLocalOffset = txSrcScratchOffset;
            } else if (linkIsShared) {
                txLocalOffset = srcOffset + tempAlgParams_.buffInfo.inBuffBaseOff;
            } else {
                txLocalOffset = txSrcInputOffset;
            }

            u64 effectiveTxScratch = txScratchOffset;
            u64 effectiveTxDst = txDstOffset;

            if (linkIsShared && !readingFromScratch) {
                effectiveTxScratch = scratchBase + perPeerChunkSize * connectedAlgRank +
                                     (actualChunkSize - sharedChunkSize);
                effectiveTxDst = (!enableRemoteMemAccess_) ? effectiveTxScratch : txLocalOffset;
            }

            txSrcSlicesAll.emplace_back(txSrcPtr, txLocalOffset, chunkSize, chunkCount);
            txDstSlicesAll.emplace_back(txDstPtr, effectiveTxDst, chunkSize, chunkCount);
            rxDstSlicesAll.emplace_back(rxDstPtr, rxOutOffset, chunkSize, chunkCount);
            rxSrcSlicesAll.emplace_back(rxSrcPtr, rxSrcOffset, chunkSize, chunkCount);
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
            HCCL_WARNING("[InsTempAlltoAllMeshClosV3] linkIdx[%u] peer %u timed out.", linkIdx, connectedRank);
            continue;
        }

        if (dmaResult != HCCL_SUCCESS) {
            HCCL_ERROR("[InsTempAlltoAllMeshClosV3] send/recv failed linkIdx[%u] connectedRank[%u]",
                       linkIdx, connectedRank);
            return dmaResult;
        }
    }

    return HCCL_SUCCESS;
}

HcclResult InsTempAlltoAllMeshClosV3::LocalDataCopy(const std::vector<ThreadHandle> &threads)
{
    if (threads.empty()) {
        return HcclResult::HCCL_E_INTERNAL;
    }

    const u32 dataTypeSize = DATATYPE_SIZE_TABLE[dataType_];

    if (totalRankSize_ == 0 || yRankSize_ == 0) {
        HCCL_ERROR("[InsTempAlltoAllMeshClosV3][LocalDataCopy] invalid rank sizes.");
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

    u64 effectiveCellStride = (tempAlgParams_.buffInfo.outBuffType == BufferType::HCCL_BUFFER)
        ? tempAlgParams_.outputSliceStride : cellSize;

    for (u32 d = 0; d < totalRankSize_; d++) {
        u32 sx = d % xRankSize_;
        u32 sy = d / xRankSize_;

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
                     perPeerClosSize * sy + effectiveCellStride * sx;
        bool skipCclCopy = (tempAlgParams_.buffInfo.inputPtr == tempAlgParams_.buffInfo.hcclBuff.addr &&
                            inOff == cclOff);
        if (!skipCclCopy) {
            DataSlice cclDstSlice(tempAlgParams_.buffInfo.hcclBuff.addr, cclOff, actualChunkSize, chunkCount);
            LocalCopy(threads[0], srcSlice, cclDstSlice);
        }
    }
    return HcclResult::HCCL_SUCCESS;
}

HcclResult InsTempAlltoAllMeshClosV3::PostLocalCopy(const std::vector<ThreadHandle> &threads)
{
    if (tempAlgParams_.buffInfo.outBuffType == BufferType::HCCL_BUFFER) {
        HCCL_INFO("[InsTempAlltoAllMeshClosV3][PostLocalCopy] skip because output is scratch");
        return HcclResult::HCCL_SUCCESS;
    }

    const u32 dataTypeSize = DATATYPE_SIZE_TABLE[dataType_];

    if (xRankSize_ == 0 || yRankSize_ == 0 || totalRankSize_ == 0) {
        HCCL_ERROR("[InsTempAlltoAllMeshClosV3][PostLocalCopy] invalid rank sizes.");
        return HCCL_E_INTERNAL;
    }

    u64 totalSliceSize = tempAlgParams_.sliceSize;
    u64 cellSize = (totalSliceSize + totalRankSize_ - 1) / totalRankSize_;
    u64 perPeerSize = (totalSliceSize + yRankSize_ - 1) / yRankSize_;

    u64 effectiveCellStride = (xRankSize_ > 0)
        ? tempAlgParams_.inputSliceStride / xRankSize_ : cellSize;

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
                                sy * perPeerSize + dx * effectiveCellStride;
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
