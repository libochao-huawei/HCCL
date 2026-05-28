/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "ins_temp_all_gather_mesh_clos_v3.h"
#include "alg_data_trans_wrapper.h"
#include "template_utils.h"
#include "channel.h"

namespace ops_hccl {

InsTempAllGatherMeshClosV3::InsTempAllGatherMeshClosV3(const OpParam &param, const u32 rankId,
                                                         const std::vector<std::vector<u32>> &subCommRanks)
    : InsTempAllGatherMeshClosV2(param, rankId, subCommRanks)
{
}

InsTempAllGatherMeshClosV3::~InsTempAllGatherMeshClosV3() {}

u64 InsTempAllGatherMeshClosV3::GetThreadNum() const
{
    return channelsPerRank_;
}

HcclResult InsTempAllGatherMeshClosV3::GetRes(AlgResourceRequest &resourceRequest) const
{
    u32 threadNum = static_cast<u32>(GetThreadNum());
    resourceRequest.slaveThreadNum = threadNum > 1 ? threadNum - 1 : 0;
    if (resourceRequest.slaveThreadNum > 0) {
        resourceRequest.notifyNumPerThread.assign(resourceRequest.slaveThreadNum, 1);
    }
    resourceRequest.notifyNumOnMainThread = threadNum > 1 ? threadNum - 1 : 0;
    return HCCL_SUCCESS;
}

void InsTempAllGatherMeshClosV3::SetStageConfig(u32 stage, bool sharedPortMode)
{
    stage_ = stage;
    if (stage_ == 1 && sharedPortMode) {
        u32 nNeighbors = static_cast<u32>(subCommRanks_[0].size());
        channelsPerRank_ = nNeighbors;
    } else {
        channelsPerRank_ = 3;
    }
    HCCL_INFO("[InsTempAllGatherMeshClosV3] SetStageConfig: stage=%u, sharedPortMode=%d, channelsPerRank_=%u",
              stage_, sharedPortMode, channelsPerRank_);
}

bool InsTempAllGatherMeshClosV3::IsSharedLink(u32 linkIdx) const
{
    return (stage_ == 1 && linkIdx == 0 && channelsPerRank_ == 4);
}

HcclResult InsTempAllGatherMeshClosV3::RunAllGatherMesh(
    const std::vector<ThreadHandle> &threads,
    const std::map<u32, std::vector<ChannelInfo>> &channels)
{
    HCCL_INFO("[InsTempAllGatherMeshClosV3][RunAllGatherMesh] Rank[%d] templateRankSize[%u] channelsPerRank_[%u] stage_[%u].",
              myRank_, templateRankSize_, channelsPerRank_, stage_);
    if (templateRankSize_ <= 1) {
        return HCCL_SUCCESS;
    }

    for (u32 linkIdx = 0; linkIdx < threads.size(); linkIdx++) {
        CHK_RET(RunAllGatherOnLink(threads, channels, linkIdx));
    }
    return HCCL_SUCCESS;
}

HcclResult InsTempAllGatherMeshClosV3::RunAllGatherOnLink(
    const std::vector<ThreadHandle> &threads,
    const std::map<u32, std::vector<ChannelInfo>> &channels,
    u32 linkIdx)
{
    CHK_PRT_RET(linkIdx >= threads.size(),
                 HCCL_ERROR("[InsTempAllGatherMeshClosV3] linkIdx[%u] >= threads.size()[%zu].",
                            linkIdx, threads.size()),
                 HcclResult::HCCL_E_INTERNAL);

    u32 myAlgRank = 0;
    CHK_RET(GetAlgRank(myRank_, subCommRanks_[0], myAlgRank));
    const u32 dataTypeSize = DATATYPE_SIZE_TABLE[dataType_];
    u32 nNeighbors = static_cast<u32>(subCommRanks_[0].size());

    bool isSharedLink = IsSharedLink(linkIdx);

    if (isSharedLink) {
        for (u32 neighborIdx = 0; neighborIdx < nNeighbors; neighborIdx++) {
            u32 connectedRank = subCommRanks_[0][(myAlgRank + 1 + neighborIdx) % nNeighbors];

            u32 connectedAlgRank = 0;
            CHK_RET(GetAlgRank(connectedRank, subCommRanks_[0], connectedAlgRank));

            if (connectedRank == myRank_) {
                continue;
            }

            auto it = channels.find(connectedRank);
            if (it == channels.end() || it->second.empty()) {
                HCCL_ERROR("[InsTempAllGatherMeshClosV3] Rank[%d] connectedRank[%u] has no channels (shared link).",
                           myRank_, connectedRank);
                return HcclResult::HCCL_E_INTERNAL;
            }

            u64 totalSliceSize = tempAlgParams_.sliceSize;
            if (tempAlgParams_.tailSize != 0 && connectedAlgRank == templateRankSize_ - 1) {
                totalSliceSize = tempAlgParams_.tailSize;
            }

            u64 mainSliceSize = static_cast<u64>(totalSliceSize * sharedLinkRatio_);
            mainSliceSize = (mainSliceSize / dataTypeSize) * dataTypeSize;

            u64 sharedSliceSize = totalSliceSize - mainSliceSize;
            if (sharedSliceSize == 0) {
                continue;
            }

            u64 scratchSliceOffset = mainSliceSize;

            CHK_RET(HandleSendRecv(threads, channels, 0, connectedRank, connectedAlgRank,
                                   sharedSliceSize, scratchSliceOffset, sharedLinkRatio_));
        }
        return HCCL_SUCCESS;
    }

    for (u32 neighborIdx = 0; neighborIdx < nNeighbors; neighborIdx++) {
        u32 connectedRank = subCommRanks_[0][(myAlgRank + 1 + neighborIdx) % nNeighbors];

        u32 connectedAlgRank = 0;
        CHK_RET(GetAlgRank(connectedRank, subCommRanks_[0], connectedAlgRank));

        if (connectedRank == myRank_) {
            continue;
        }

        auto it = channels.find(connectedRank);
        if (it == channels.end() || it->second.empty()) {
            HCCL_ERROR("[InsTempAllGatherMeshClosV3] Rank[%d] connectedRank[%u] has no channels.",
                       myRank_, connectedRank);
            return HcclResult::HCCL_E_INTERNAL;
        }

        u32 selectedLinkIdx;
        u32 totalRanks = nNeighbors;
        if ((totalRanks & (totalRanks - 1)) == 0) {
            selectedLinkIdx = myAlgRank ^ connectedAlgRank;
        } else {
            u32 nDedicated = channelsPerRank_ - 1;
            selectedLinkIdx = (myAlgRank ^ connectedAlgRank) % nDedicated + 1;
        }

        if (selectedLinkIdx != linkIdx) {
            continue;
        }

        u64 totalSliceSize = tempAlgParams_.sliceSize;
        if (tempAlgParams_.tailSize != 0 && connectedAlgRank == templateRankSize_ - 1) {
            totalSliceSize = tempAlgParams_.tailSize;
        }

        u64 mainSliceSize;
        if (stage_ == 1 && channelsPerRank_ == 4) {
            mainSliceSize = static_cast<u64>(totalSliceSize * sharedLinkRatio_);
            mainSliceSize = (mainSliceSize / dataTypeSize) * dataTypeSize;
        } else {
            mainSliceSize = totalSliceSize;
        }

        if (mainSliceSize == 0) {
            continue;
        }

        CHK_RET(HandleSendRecv(threads, channels, linkIdx, connectedRank, connectedAlgRank,
                               mainSliceSize, 0, sharedLinkRatio_));
    }

    return HCCL_SUCCESS;
}

HcclResult InsTempAllGatherMeshClosV3::HandleSendRecv(
    const std::vector<ThreadHandle> &threads,
    const std::map<u32, std::vector<ChannelInfo>> &channels,
    u32 linkIdx, u32 connectedRank, u32 connectedAlgRank,
    u64 sliceSize, u64 scratchSliceOffset, double sharedLinkRatio)
{
    (void)sharedLinkRatio;

    auto it = channels.find(connectedRank);
    if (it == channels.end() || linkIdx >= it->second.size()) {
        HCCL_ERROR("[InsTempAllGatherMeshClosV3] invalid channel access: rank=%u linkIdx=%u",
                   connectedRank, linkIdx);
        return HcclResult::HCCL_E_INTERNAL;
    }

    const ChannelInfo &linkRemote = it->second[linkIdx];
    void *remoteCclBuffAddr = linkRemote.remoteCclMem.addr;

    const u32 dataTypeSize = DATATYPE_SIZE_TABLE[dataType_];

    std::vector<DataSlice> txSrcSlicesAll;
    std::vector<DataSlice> txDstSlicesAll;
    std::vector<DataSlice> rxDstSlicesAll;
    std::vector<DataSlice> rxSrcSlicesAll;

    for (u32 rpt = 0; rpt < tempAlgParams_.repeatNum; ++rpt) {
        const u64 outBaseOff = tempAlgParams_.buffInfo.outBuffBaseOff + rpt * tempAlgParams_.outputRepeatStride;
        const u64 scratchRepeatStride = tempAlgParams_.sliceSize * templateRankSize_;
        const u64 scratchBase = tempAlgParams_.buffInfo.hcclBuffBaseOff + rpt * scratchRepeatStride;

        u64 txOutOffset = tempAlgParams_.outputSliceStride * myAlgRank + outBaseOff;
        u64 txScratchOffset = scratchBase + tempAlgParams_.sliceSize * myAlgRank + scratchSliceOffset;
        u64 txDstOffset = (!enableRemoteMemAccess_) ? txScratchOffset : txOutOffset;

        u64 rxOutOffset = tempAlgParams_.outputSliceStride * connectedAlgRank + outBaseOff;
        u64 rxScratchOffset = scratchBase + tempAlgParams_.sliceSize * connectedAlgRank + scratchSliceOffset;
        u64 rxSrcOffset = (!enableRemoteMemAccess_) ? rxScratchOffset : rxOutOffset;

        void *txSrcPtr = tempAlgParams_.buffInfo.outputPtr;
        void *txDstPtr = (!enableRemoteMemAccess_) ? remoteCclBuffAddr : linkRemote.remoteOutputGraphMode.addr;
        void *rxSrcPtr = (!enableRemoteMemAccess_) ? remoteCclBuffAddr : linkRemote.remoteOutputGraphMode.addr;
        void *rxDstPtr = tempAlgParams_.buffInfo.outputPtr;
        u64 sliceCount = sliceSize / dataTypeSize;

        txSrcSlicesAll.emplace_back(txSrcPtr, txOutOffset, sliceSize, sliceCount);
        txDstSlicesAll.emplace_back(txDstPtr, txDstOffset, sliceSize, sliceCount);
        rxDstSlicesAll.emplace_back(rxDstPtr, rxOutOffset, sliceSize, sliceCount);
        rxSrcSlicesAll.emplace_back(rxSrcPtr, rxSrcOffset, sliceSize, sliceCount);
    }

    TxRxSlicesList sendRecvSlicesList({txSrcSlicesAll, txDstSlicesAll},
                                      {rxSrcSlicesAll, rxDstSlicesAll});
    TxRxChannels sendRecvChannels(linkRemote, linkRemote);
    SendRecvInfo sendRecvInfo(sendRecvChannels, sendRecvSlicesList);

    CHK_PRT_RET(SendRecvRead(sendRecvInfo, threads[linkIdx]),
                 HCCL_ERROR("[InsTempAllGatherMeshClosV3] SendRecvRead failed on linkIdx[%u] connectedRank[%u]",
                            linkIdx, connectedRank),
                 HcclResult::HCCL_E_INTERNAL);

    HCCL_DEBUG("[InsTempAllGatherMeshClosV3] Rank[%d] linkIdx[%u] connectedRank[%u] sliceSize[%llu] scratchOffset[%llu] done",
               myRank_, linkIdx, connectedRank, sliceSize, scratchSliceOffset);

    return HCCL_SUCCESS;
}

}  // namespace ops_hccl
