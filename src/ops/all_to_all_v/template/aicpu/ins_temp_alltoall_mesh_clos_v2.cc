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
    HCCL_INFO("[InsTempAlltoAllMeshClosV2][RunAlltoAllMesh] Rank[%d] templateRankSize[%u] totalLinks[%u].",
              myRank_, templateRankSize_, channelsPerRank_);
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

HcclResult InsTempAlltoAllMeshClosV2::RunAlltoAllOnLink(
    const std::vector<ThreadHandle> &threads,
    const std::map<u32, std::vector<ChannelInfo>> &channels,
    u32 linkIdx)
{
    CHK_PRT_RET(linkIdx >= threads.size(),
                HCCL_ERROR("[InsTempAlltoAllMeshClosV2] linkIdx[%u] >= threads.size()[%zu].",
                           linkIdx, threads.size()),
                HcclResult::HCCL_E_INTERNAL);

    u32 myAlgRank = 0;
    CHK_RET(GetAlgRank(myRank_, subCommRanks_[0], myAlgRank));

    bool isPcie = IsPcieProtocol(channels);
    const u32 dataTypeSize = DATATYPE_SIZE_TABLE[dataType_];
    u64 totalSliceSize = tempAlgParams_.sliceSize;
    u64 perPeerChunkSize = (totalSliceSize + templateRankSize_ - 1) / templateRankSize_;

    // v3.0 Fix B: per-link lastPeerSize for ceiling over-shoot
    u64 lastPeerSize = totalSliceSize - perPeerChunkSize * (templateRankSize_ - 1);
    if (lastPeerSize <= 0) {
        lastPeerSize = perPeerChunkSize;
    }
    u32 lastPeerIndex = templateRankSize_ - 1;

    for (u32 neighborIdx = 0; neighborIdx < subCommRanks_[0].size() - 1; neighborIdx++) {
        u32 connectedRank = subCommRanks_[0][(myAlgRank + 1 + neighborIdx) % subCommRanks_[0].size()];
        u32 connectedAlgRank = 0;
        CHK_RET(GetAlgRank(connectedRank, subCommRanks_[0], connectedAlgRank));

        if (failedRanks_[connectedAlgRank]) {
            continue;
        }

        auto it = channels.find(connectedRank);
        if (it == channels.end() || it->second.empty()) {
            HCCL_ERROR("[InsTempAlltoAllMeshClosV2] Rank[%d] connectedRank[%u] has no channels.",
                       myRank_, connectedRank);
            return HcclResult::HCCL_E_INTERNAL;
        }

        u32 totalLinksToNeighbor = it->second.size();
        u32 selectedLinkIdx = (myAlgRank + connectedAlgRank) % threads.size();

        if (selectedLinkIdx != linkIdx) {
            continue;
        }

        HCCL_INFO("[InsTempAlltoAllMeshClosV2] Rank[%d] linkIdx[%u] matched connectedRank[%u] "
                  "selectedLinkIdx[%u] totalLinks[%u] totalThreads[%u] enableRemoteMemAccess[%d]",
                  myRank_, linkIdx, connectedRank, selectedLinkIdx, totalLinksToNeighbor, threads.size(),
                  enableRemoteMemAccess_);

        const ChannelInfo &linkRemote = it->second[linkIdx];
        void *remoteCclBuffAddr = linkRemote.remoteCclMem.addr;

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

            u64 txOutOffset = tempAlgParams_.outputSliceStride * connectedAlgRank + outBaseOff;
            u64 txScratchOffset = scratchBase + perPeerChunkSize * connectedAlgRank;
            u64 txDstOffset = (!enableRemoteMemAccess_) ? txScratchOffset : txOutOffset;

            u64 rxOutOffset = tempAlgParams_.outputSliceStride * myAlgRank + outBaseOff;
            u64 rxScratchOffset = scratchBase + perPeerChunkSize * myAlgRank;
            u64 rxSrcOffset = (!enableRemoteMemAccess_) ? rxScratchOffset : rxOutOffset;

            void *txSrcPtr = tempAlgParams_.buffInfo.inputPtr;
            void *txDstPtr = (!enableRemoteMemAccess_) ? remoteCclBuffAddr
                                                         : linkRemote.remoteOutputGraphMode.addr;
            void *rxSrcPtr = (!enableRemoteMemAccess_) ? remoteCclBuffAddr
                                                         : linkRemote.remoteOutputGraphMode.addr;
            void *rxDstPtr = tempAlgParams_.buffInfo.outputPtr;

            txSrcSlicesAll.emplace_back(txSrcPtr, txOutOffset, actualChunkSize, chunkCount);
            txDstSlicesAll.emplace_back(txDstPtr, txDstOffset, actualChunkSize, chunkCount);
            rxDstSlicesAll.emplace_back(rxDstPtr, rxOutOffset, actualChunkSize, chunkCount);
            rxSrcSlicesAll.emplace_back(rxSrcPtr, rxSrcOffset, actualChunkSize, chunkCount);
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
            HCCL_WARNING("[InsTempAlltoAllMeshClosV2] linkIdx[%u] peer %u timed out.", linkIdx, connectedRank);
            continue;
        }

        if (dmaResult != HCCL_SUCCESS) {
            HCCL_ERROR("[InsTempAlltoAllMeshClosV2] send/recv failed linkIdx[%u] connectedRank[%u]",
                       linkIdx, connectedRank);
            return dmaResult;
        }
    }

    return HCCL_SUCCESS;
}

}  // namespace ops_hccl
