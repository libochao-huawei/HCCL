/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "ins_temp_all_gather_mesh_clos_v2.h"
#include "alg_data_trans_wrapper.h"
#include "template_utils.h"
#include "channel.h"

namespace ops_hccl {

InsTempAllGatherMeshClosV2::InsTempAllGatherMeshClosV2(const OpParam &param, const u32 rankId,
                                                       const std::vector<std::vector<u32>> &subCommRanks)
    : InsTempAllGatherMesh1D(param, rankId, subCommRanks)
{
}

InsTempAllGatherMeshClosV2::~InsTempAllGatherMeshClosV2() {}

u64 InsTempAllGatherMeshClosV2::GetThreadNum() const
{
    return channelsPerRank_;
    // 最多使用 min(链路数, 邻居数) 个线程
    // u32 numNeighbors = std::max(1u, templateRankSize_ - 1);
    // return std::min(channelsPerRank_, numNeighbors);
}

HcclResult InsTempAllGatherMeshClosV2::GetRes(AlgResourceRequest &resourceRequest) const
{   
    u32 threadNum = GetThreadNum();
    resourceRequest.slaveThreadNum = threadNum > 1 ? threadNum - 1 : 0;
    if (resourceRequest.slaveThreadNum > 0) {
        resourceRequest.notifyNumPerThread.assign(resourceRequest.slaveThreadNum, 1);
    }
    resourceRequest.notifyNumOnMainThread = threadNum > 1 ? threadNum - 1 : 0;
    return HCCL_SUCCESS;
}

HcclResult InsTempAllGatherMeshClosV2::CalcRes(HcclComm comm, const OpParam &param,
                                               const TopoInfoWithNetLayerDetails *topoInfo,
                                               AlgResourceRequest &resourceRequest)
{
    HCCL_INFO("[InsTempAllGatherMeshClosV2][CalcRes] start");
    std::vector<HcclChannelDesc> levelChannels;
    CHK_RET(CalcChannelRequestNhr(comm, param, topoInfo, subCommRanks_, levelChannels));
    resourceRequest.channels.push_back(levelChannels);

    channelsPerRank_ = levelChannels.empty() ? 1 : CalcChannelsPerRank(levelChannels);
    HCCL_INFO("[InsTempAllGatherMeshClosV2][CalcRes] totalLinks[%u], channelCount[%zu]",
              channelsPerRank_, levelChannels.size());

    CHK_RET(GetRes(resourceRequest));
    return HCCL_SUCCESS;
}

HcclResult InsTempAllGatherMeshClosV2::RunAllGatherMesh(
    const std::vector<ThreadHandle> &threads,
    const std::map<u32, std::vector<ChannelInfo>> &channels)
{
    HCCL_INFO("[InsTempAllGatherMeshClosV2][RunAllGatherMesh] Rank[%d] templateRankSize[%u] totalLinks[%u].",
              myRank_, templateRankSize_, channelsPerRank_);
    // ========== 新增日志 ==========
    HCCL_INFO("[InsTempAllGatherMeshClosV2][RunAllGatherMesh] threads.size=%zu, channels.size=%zu",
              threads.size(), channels.size());
    // ============================
    if (templateRankSize_ <= 1) {
        return HCCL_SUCCESS;
    }

    for (u32 linkIdx = 0; linkIdx < threads.size(); linkIdx++) {
        CHK_RET(RunAllGatherOnLink(threads, channels, linkIdx));
    }
    return HCCL_SUCCESS;
}

HcclResult InsTempAllGatherMeshClosV2::RunAllGatherOnLink(
    const std::vector<ThreadHandle> &threads,
    const std::map<u32, std::vector<ChannelInfo>> &channels,
    u32 linkIdx)
{
    CHK_PRT_RET(linkIdx >= threads.size(),
                HCCL_ERROR("[InsTempAllGatherMeshClosV2] linkIdx[%u] >= threads.size()[%zu].",
                           linkIdx, threads.size()),
                HcclResult::HCCL_E_INTERNAL);

    u32 myAlgRank = 0;
    CHK_RET(GetAlgRank(myRank_, subCommRanks_[0], myAlgRank));
    const u32 dataTypeSize = DATATYPE_SIZE_TABLE[dataType_];

    for (u32 neighborIdx = 0; neighborIdx < subCommRanks_[0].size() - 1; neighborIdx++) {
        u32 connectedRank = subCommRanks_[0][(myAlgRank + 1 + neighborIdx) % subCommRanks_[0].size()];

        u32 connectedAlgRank = 0;
        CHK_RET(GetAlgRank(connectedRank, subCommRanks_[0], connectedAlgRank));

        auto it = channels.find(connectedRank);
        if (it == channels.end() || it->second.empty()) {
            HCCL_ERROR("[InsTempAllGatherMeshClosV2] Rank[%d] connectedRank[%u] has no channels.",
                       myRank_, connectedRank);
            return HcclResult::HCCL_E_INTERNAL;
        }

        u32 totalLinksToNeighbor = it->second.size();
        u32 selectedLinkIdx = (myAlgRank + connectedAlgRank) % threads.size();

        
        if (selectedLinkIdx != linkIdx) {
            continue;
        }

        HCCL_INFO("[InsTempAllGatherMeshClosV2 0] Rank[%d] linkIdx[%u] matched connectedRank[%u] "
                  "selectedLinkIdx[%u] totalLinks[%u] totalThreads[%u] enableRemoteMemAccess[%d]",
                  myRank_, linkIdx, connectedRank, selectedLinkIdx, totalLinksToNeighbor, threads.size(),
                  enableRemoteMemAccess_);

        const ChannelInfo &linkRemote = it->second[linkIdx];
        void *remoteCclBuffAddr = linkRemote.remoteCclMem.addr;

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

            u64 sliceSize = tempAlgParams_.sliceSize;
            if (tempAlgParams_.tailSize != 0 && connectedAlgRank == templateRankSize_ - 1) {
                sliceSize = tempAlgParams_.tailSize;
            }

            u64 txOutOffset = tempAlgParams_.outputSliceStride * myAlgRank + outBaseOff;
            u64 txScratchOffset = scratchBase + tempAlgParams_.sliceSize * myAlgRank;
            u64 txDstOffset = (!enableRemoteMemAccess_) ? txScratchOffset : txOutOffset;

            u64 rxOutOffset = tempAlgParams_.outputSliceStride * connectedAlgRank + outBaseOff;
            u64 rxScratchOffset = scratchBase + tempAlgParams_.sliceSize * connectedAlgRank;
            u64 rxSrcOffset = (!enableRemoteMemAccess_) ? rxScratchOffset : rxOutOffset;

            void *txSrcPtr = tempAlgParams_.buffInfo.outputPtr;
            void *txDstPtr = (!enableRemoteMemAccess_) ? remoteCclBuffAddr
                                                        : linkRemote.remoteOutputGraphMode.addr;
            void *rxSrcPtr = (!enableRemoteMemAccess_) ? remoteCclBuffAddr
                                                        : linkRemote.remoteOutputGraphMode.addr;
            void *rxDstPtr = tempAlgParams_.buffInfo.outputPtr;
            u64 sliceCount = sliceSize / dataTypeSize;

            HCCL_INFO("[InsTempAllGatherMeshClosV2 1] Rank[%d] linkIdx[%u] rpt[%u] Send: "
                      "srcRank[%d] -> dstRank[%u], "
                      "srcPtr[%p] + offset[%llu], dstPtr[%p] + offset[%llu], "
                      "size[%llu] count[%llu]",
                      myRank_, linkIdx, rpt,
                      myRank_, connectedRank,
                      txSrcPtr, (unsigned long long)txOutOffset,
                      txDstPtr, (unsigned long long)txDstOffset,
                      (unsigned long long)sliceSize, (unsigned long long)sliceCount);

            HCCL_INFO("[InsTempAllGatherMeshClosV2 2] Rank[%d] linkIdx[%u] rpt[%u] Recv: "
                      "srcRank[%u] -> dstRank[%d], "
                      "srcPtr[%p] + offset[%llu], dstPtr[%p] + offset[%llu], "
                      "size[%llu] count[%llu]",
                      myRank_, linkIdx, rpt,
                      connectedRank, myRank_,
                      rxSrcPtr, (unsigned long long)rxSrcOffset,
                      rxDstPtr, (unsigned long long)rxOutOffset,
                      (unsigned long long)sliceSize, (unsigned long long)sliceCount);

            txSrcSlicesAll.emplace_back(txSrcPtr, txOutOffset, sliceSize, sliceCount);
            txDstSlicesAll.emplace_back(txDstPtr, txDstOffset, sliceSize, sliceCount);
            rxDstSlicesAll.emplace_back(rxDstPtr, rxOutOffset, sliceSize, sliceCount);
            rxSrcSlicesAll.emplace_back(rxSrcPtr, rxSrcOffset, sliceSize, sliceCount);
        }

        TxRxSlicesList sendRecvSlicesList({txSrcSlicesAll, txDstSlicesAll},
                                          {rxSrcSlicesAll, rxDstSlicesAll});
        TxRxChannels sendRecvChannels(linkRemote, linkRemote);
        SendRecvInfo sendRecvInfo(sendRecvChannels, sendRecvSlicesList);
        
        HCCL_INFO("[InsTempAllGatherMeshClosV2] Rank[%d] linkIdx[%u] before SendRecvRead, "
                  "connectedRank[%u] slices[%zu]",
                  myRank_, linkIdx, connectedRank, txSrcSlicesAll.size());

        CHK_PRT_RET(SendRecvRead(sendRecvInfo, threads[linkIdx]),
                    HCCL_ERROR("[InsTempAllGatherMeshClosV2] SendRecvRead failed on linkIdx[%u] "
                               "connectedRank[%u]", linkIdx, connectedRank),
                    HcclResult::HCCL_E_INTERNAL);

        HCCL_INFO("[InsTempAllGatherMeshClosV2] Rank[%d] linkIdx[%u] after SendRecvRead successfully",
                  myRank_, linkIdx);
    }
    return HCCL_SUCCESS;
}

}  // namespace ops_hccl
