/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "ins_temp_all_gather_mesh_1d_v2.h"
#include "alg_data_trans_wrapper.h"
#include "template_utils.h"

namespace ops_hccl {

InsTempAllGatherMesh1DV2::InsTempAllGatherMesh1DV2(const OpParam &param, const u32 rankId,
                                                     const std::vector<std::vector<u32>> &subCommRanks)
    : InsTempAllGatherMesh1D(param, rankId, subCommRanks)
{
}

InsTempAllGatherMesh1DV2::~InsTempAllGatherMesh1DV2() {}

u64 InsTempAllGatherMesh1DV2::GetThreadNum() const
{
    if (borrowEnabled_) {
        return portCount_;
    }
    return InsTempAllGatherMesh1D::GetThreadNum();
}

HcclResult InsTempAllGatherMesh1DV2::GetRes(AlgResourceRequest &resourceRequest) const
{
    u32 threadNum = static_cast<u32>(GetThreadNum());
    resourceRequest.slaveThreadNum = threadNum > 1 ? threadNum - 1 : 0;
    if (resourceRequest.slaveThreadNum > 0) {
        resourceRequest.notifyNumPerThread.assign(resourceRequest.slaveThreadNum, 1);
    }
    resourceRequest.notifyNumOnMainThread = threadNum > 1 ? threadNum - 1 : 0;
    return HCCL_SUCCESS;
}

HcclResult InsTempAllGatherMesh1DV2::RunAllGatherMesh(
    const std::vector<ThreadHandle> &threads,
    const std::map<u32, std::vector<ChannelInfo>> &channels)
{
    HCCL_INFO("[InsTempAllGatherMesh1DV2] RunAllGatherMesh RankIDs[%d].", myRank_);

    if (threads.empty()) {
        return HCCL_SUCCESS;
    }

    u32 myAlgRank = 0;
    CHK_RET(GetAlgRank(myRank_, subCommRanks_[0], myAlgRank));
    const u32 dataTypeSize = DATATYPE_SIZE_TABLE[dataType_];

    u32 numNeighbors = static_cast<u32>(subCommRanks_[0].size());
    if (numNeighbors <= 1) {
        return HCCL_SUCCESS;
    }

    u32 threadCount = static_cast<u32>(threads.size());

    for (u32 threadIdx = 0; threadIdx < threadCount; threadIdx++) {
        u32 connectedRank = INVALID_VALUE_RANKID;
        u32 connectedAlgRank = 0;

        if (!borrowEnabled_ || threadIdx < portCount_ - 1) {
            u32 neighborOffset = threadIdx % (numNeighbors - 1);
            connectedRank = subCommRanks_[0][(myAlgRank + 1 + neighborOffset) % numNeighbors];
            CHK_RET(GetAlgRank(connectedRank, subCommRanks_[0], connectedAlgRank));
        } else {
            connectedRank = doubleLinkedNeighbor_;
            CHK_RET(GetAlgRank(connectedRank, subCommRanks_[0], connectedAlgRank));
        }

        auto it = channels.find(connectedRank);
        if (it == channels.end() || it->second.empty()) {
            HCCL_ERROR("[InsTempAllGatherMesh1DV2] Rank[%d] connectedRank[%u] has no channels.",
                       myRank_, connectedRank);
            return HcclResult::HCCL_E_INTERNAL;
        }

        u32 channelIdx = 0;
        if (borrowEnabled_ && connectedRank == doubleLinkedNeighbor_ && it->second.size() >= 2) {
            if (threadIdx < portCount_ - 1) {
                channelIdx = 0;
            } else {
                channelIdx = 1;
            }
        }

        CHK_PRT_RET(channelIdx >= it->second.size(),
                     HCCL_ERROR("[InsTempAllGatherMesh1DV2] channelIdx[%u] >= channels[%zu] for rank[%u]",
                                channelIdx, it->second.size(), connectedRank),
                     HcclResult::HCCL_E_INTERNAL);

        const ChannelInfo &linkRemote = it->second[channelIdx];
        void *remoteCclBuffAddr = linkRemote.remoteCclMem.addr;

        std::vector<DataSlice> txSrcSlicesAll;
        std::vector<DataSlice> txDstSlicesAll;
        std::vector<DataSlice> rxDstSlicesAll;
        std::vector<DataSlice> rxSrcSlicesAll;

        for (u32 rpt = 0; rpt < tempAlgParams_.repeatNum; ++rpt) {
            const u64 outBaseOff = tempAlgParams_.buffInfo.outBuffBaseOff + rpt * tempAlgParams_.outputRepeatStride;
            const u64 scratchRepeatStride = tempAlgParams_.sliceSize * templateRankSize_;
            const u64 scratchBase = tempAlgParams_.buffInfo.hcclBuffBaseOff + rpt * scratchRepeatStride;

            u64 neighborSliceSize = tempAlgParams_.sliceSize;
            if (tempAlgParams_.tailSize != 0 && connectedAlgRank == templateRankSize_ - 1) {
                neighborSliceSize = tempAlgParams_.tailSize;
            }

            u64 scratchChunkOffset = 0;
            if (borrowEnabled_ && connectedRank == doubleLinkedNeighbor_) {
                constexpr double BORROW_RATIO = 50.0 / 200.0;
                constexpr double DIRECT_RATIO = 150.0 / 200.0;
                if (channelIdx == 0) {
                    neighborSliceSize = static_cast<u64>(neighborSliceSize * DIRECT_RATIO);
                    neighborSliceSize = (neighborSliceSize / dataTypeSize) * dataTypeSize;
                } else {
                    u64 mainSliceSize = static_cast<u64>(
                        (tempAlgParams_.tailSize != 0 && connectedAlgRank == templateRankSize_ - 1
                             ? tempAlgParams_.tailSize
                             : tempAlgParams_.sliceSize) *
                        DIRECT_RATIO);
                    mainSliceSize = (mainSliceSize / dataTypeSize) * dataTypeSize;
                    neighborSliceSize = static_cast<u64>(
                        (tempAlgParams_.tailSize != 0 && connectedAlgRank == templateRankSize_ - 1
                             ? tempAlgParams_.tailSize
                             : tempAlgParams_.sliceSize) *
                        BORROW_RATIO);
                    neighborSliceSize = (neighborSliceSize / dataTypeSize) * dataTypeSize;
                    scratchChunkOffset = mainSliceSize;
                }
            }

            if (neighborSliceSize == 0) {
                continue;
            }

            u64 txOutOffset = tempAlgParams_.outputSliceStride * myAlgRank + outBaseOff;
            u64 txScratchOffset = scratchBase + tempAlgParams_.sliceSize * myAlgRank + scratchChunkOffset;
            u64 txDstOffset = (!enableRemoteMemAccess_) ? txScratchOffset : txOutOffset;

            u64 rxOutOffset = tempAlgParams_.outputSliceStride * connectedAlgRank + outBaseOff;
            u64 rxScratchOffset = scratchBase + tempAlgParams_.sliceSize * connectedAlgRank + scratchChunkOffset;
            u64 rxSrcOffset = (!enableRemoteMemAccess_) ? rxScratchOffset : rxOutOffset;

            void *txSrcPtr = tempAlgParams_.buffInfo.outputPtr;
            void *txDstPtr = (!enableRemoteMemAccess_) ? remoteCclBuffAddr : linkRemote.remoteOutputGraphMode.addr;
            void *rxSrcPtr = (!enableRemoteMemAccess_) ? remoteCclBuffAddr : linkRemote.remoteOutputGraphMode.addr;
            void *rxDstPtr = tempAlgParams_.buffInfo.outputPtr;
            u64 sliceCount = neighborSliceSize / dataTypeSize;

            txSrcSlicesAll.emplace_back(txSrcPtr, txOutOffset, neighborSliceSize, sliceCount);
            txDstSlicesAll.emplace_back(txDstPtr, txDstOffset, neighborSliceSize, sliceCount);
            rxDstSlicesAll.emplace_back(rxDstPtr, rxOutOffset, neighborSliceSize, sliceCount);
            rxSrcSlicesAll.emplace_back(rxSrcPtr, rxSrcOffset, neighborSliceSize, sliceCount);
        }

        if (txSrcSlicesAll.empty()) {
            continue;
        }

        TxRxSlicesList sendRecvSlicesList({txSrcSlicesAll, txDstSlicesAll},
                                          {rxSrcSlicesAll, rxDstSlicesAll});
        TxRxChannels sendRecvChannels(linkRemote, linkRemote);
        SendRecvInfo sendRecvInfo(sendRecvChannels, sendRecvSlicesList);
        CHK_PRT_RET(SendRecvRead(sendRecvInfo, threads[threadIdx]),
                     HCCL_ERROR("[InsTempAllGatherMesh1DV2] SendRecvRead failed on threadIdx[%u] connectedRank[%u]",
                                threadIdx, connectedRank),
                     HcclResult::HCCL_E_INTERNAL);
    }

    return HcclResult::HCCL_SUCCESS;
}

}  // namespace ops_hccl
