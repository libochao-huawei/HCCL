/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "ins_temp_all_gather_mesh_clos_v2_no_memcpy.h"
#include "alg_data_trans_wrapper.h"
#include "channel.h"
#include "template_utils.h"

namespace ops_hccl {
namespace {
HcclResult CheckRemoteOutputRange(const char *tag, u32 myRank, u32 peer, const ChannelInfo &linkRemote,
                                  u64 offset, u64 size)
{
    CHK_PRT_RET(linkRemote.remoteOutputGraphMode.addr == nullptr,
                HCCL_ERROR("[%s] Rank[%u] peer[%u] remote output unavailable.", tag, myRank, peer),
                HcclResult::HCCL_E_INTERNAL);
    CHK_PRT_RET(offset + size > linkRemote.remoteOutputGraphMode.size,
                HCCL_ERROR("[%s] Rank[%u] peer[%u] remote output range overflow. off[%llu] size[%llu] "
                           "remoteOutputSize[%llu]",
                           tag, myRank, peer, offset, size, linkRemote.remoteOutputGraphMode.size),
                HcclResult::HCCL_E_INTERNAL);
    return HCCL_SUCCESS;
}
}  // namespace

InsTempAllGatherMeshClosV2NoMemcpy::InsTempAllGatherMeshClosV2NoMemcpy(
    const OpParam &param, const u32 rankId, const std::vector<std::vector<u32>> &subCommRanks)
    : InsTempAllGatherMesh1DNoMemcpy(param, rankId, subCommRanks)
{
}

InsTempAllGatherMeshClosV2NoMemcpy::~InsTempAllGatherMeshClosV2NoMemcpy() {}

u64 InsTempAllGatherMeshClosV2NoMemcpy::GetThreadNum() const
{
    return channelsPerRank_;
}

HcclResult InsTempAllGatherMeshClosV2NoMemcpy::GetRes(AlgResourceRequest &resourceRequest) const
{
    u32 threadNum = GetThreadNum();
    resourceRequest.slaveThreadNum = threadNum > 1 ? threadNum - 1 : 0;
    if (resourceRequest.slaveThreadNum > 0) {
        resourceRequest.notifyNumPerThread.assign(resourceRequest.slaveThreadNum, 1);
    }
    resourceRequest.notifyNumOnMainThread = threadNum > 1 ? threadNum - 1 : 0;
    return HCCL_SUCCESS;
}

HcclResult InsTempAllGatherMeshClosV2NoMemcpy::CalcRes(HcclComm comm, const OpParam &param,
                                                       const TopoInfoWithNetLayerDetails *topoInfo,
                                                       AlgResourceRequest &resourceRequest)
{
    HCCL_INFO("[InsTempAllGatherMeshClosV2NoMemcpy][CalcRes] start");
    std::vector<HcclChannelDesc> levelChannels;
    CHK_RET(CalcChannelRequestMesh1D(comm, param, topoInfo, subCommRanks_, levelChannels));
    resourceRequest.channels.push_back(levelChannels);

    channelsPerRank_ = levelChannels.empty() ? 1 : CalcChannelsPerRank(levelChannels);
    HCCL_INFO("[InsTempAllGatherMeshClosV2NoMemcpy][CalcRes] totalLinks[%u], channelCount[%zu]",
              channelsPerRank_, levelChannels.size());
    CHK_RET(GetRes(resourceRequest));
    return HCCL_SUCCESS;
}

HcclResult InsTempAllGatherMeshClosV2NoMemcpy::RunAllGatherMesh(
    const std::vector<ThreadHandle> &threads, const std::map<u32, std::vector<ChannelInfo>> &channels)
{
    HCCL_INFO("[InsTempAllGatherMeshClosV2NoMemcpy][RunAllGatherMesh] Rank[%u] templateRankSize[%u] totalLinks[%u] "
              "threads[%zu] channels[%zu].",
              myRank_, templateRankSize_, channelsPerRank_, threads.size(), channels.size());
    if (templateRankSize_ <= 1) {
        return HCCL_SUCCESS;
    }

    for (u32 linkIdx = 0; linkIdx < threads.size(); linkIdx++) {
        CHK_RET(RunAllGatherOnLink(threads, channels, linkIdx));
    }
    return HCCL_SUCCESS;
}

HcclResult InsTempAllGatherMeshClosV2NoMemcpy::RunAllGatherOnLink(
    const std::vector<ThreadHandle> &threads, const std::map<u32, std::vector<ChannelInfo>> &channels, u32 linkIdx)
{
    CHK_PRT_RET(linkIdx >= threads.size(),
                HCCL_ERROR("[InsTempAllGatherMeshClosV2NoMemcpy] linkIdx[%u] >= threads[%zu].", linkIdx,
                           threads.size()),
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
            HCCL_ERROR("[InsTempAllGatherMeshClosV2NoMemcpy] Rank[%u] connectedRank[%u] has no channels.",
                       myRank_, connectedRank);
            return HcclResult::HCCL_E_INTERNAL;
        }

        u32 selectedLinkIdx = (myAlgRank + connectedAlgRank) % threads.size();
        if (selectedLinkIdx != linkIdx) {
            continue;
        }
        CHK_PRT_RET(linkIdx >= it->second.size(),
                    HCCL_ERROR("[InsTempAllGatherMeshClosV2NoMemcpy] Rank[%u] peer[%u] linkIdx[%u] "
                               "peerChannels[%zu] threads[%zu].",
                               myRank_, connectedRank, linkIdx, it->second.size(), threads.size()),
                    HcclResult::HCCL_E_INTERNAL);

        const ChannelInfo &linkRemote = it->second[linkIdx];
        std::vector<DataSlice> txSrcSlicesAll;
        std::vector<DataSlice> txDstSlicesAll;
        std::vector<DataSlice> rxSrcSlicesAll;
        std::vector<DataSlice> rxDstSlicesAll;

        for (u32 rpt = 0; rpt < tempAlgParams_.repeatNum; ++rpt) {
            const u64 outBaseOff = tempAlgParams_.buffInfo.outBuffBaseOff + rpt * tempAlgParams_.outputRepeatStride;
            u64 sliceSize = tempAlgParams_.sliceSize;
            if (tempAlgParams_.tailSize != 0 && myAlgRank == templateRankSize_ - 1) {
                sliceSize = tempAlgParams_.tailSize;
            }
            u64 peerSliceSize = tempAlgParams_.sliceSize;
            if (tempAlgParams_.tailSize != 0 && connectedAlgRank == templateRankSize_ - 1) {
                peerSliceSize = tempAlgParams_.tailSize;
            }

            const u64 txOutOffset = tempAlgParams_.outputSliceStride * myAlgRank + outBaseOff;
            const u64 txDstOffset = txOutOffset;
            CHK_RET(CheckRemoteOutputRange("InsTempAllGatherMeshClosV2NoMemcpy", myRank_, connectedRank,
                                           linkRemote, txDstOffset, sliceSize));
            txSrcSlicesAll.emplace_back(tempAlgParams_.buffInfo.outputPtr, txOutOffset, sliceSize,
                                        sliceSize / dataTypeSize);
            txDstSlicesAll.emplace_back(linkRemote.remoteOutputGraphMode.addr, txDstOffset, sliceSize,
                                        sliceSize / dataTypeSize);

            const u64 rxOutOffset = tempAlgParams_.outputSliceStride * connectedAlgRank + outBaseOff;
            rxSrcSlicesAll.emplace_back(linkRemote.remoteOutputGraphMode.addr, rxOutOffset, peerSliceSize,
                                        peerSliceSize / dataTypeSize);
            rxDstSlicesAll.emplace_back(tempAlgParams_.buffInfo.outputPtr, rxOutOffset, peerSliceSize,
                                        peerSliceSize / dataTypeSize);
        }

        TxRxSlicesList sendRecvSlicesList({txSrcSlicesAll, txDstSlicesAll}, {rxSrcSlicesAll, rxDstSlicesAll});
        TxRxChannels sendRecvChannels(linkRemote, linkRemote);
        SendRecvInfo sendRecvInfo(sendRecvChannels, sendRecvSlicesList, dataType_);
        HcclResult ret = SendRecvWrite(sendRecvInfo, threads[linkIdx]);
        CHK_PRT_RET(ret != HCCL_SUCCESS,
                    HCCL_ERROR("[InsTempAllGatherMeshClosV2NoMemcpy] SendRecvWrite failed. rank[%u] peer[%u] "
                               "linkIdx[%u] ret[%d]",
                               myRank_, connectedRank, linkIdx, ret),
                    ret);
    }
    return HCCL_SUCCESS;
}

}  // namespace ops_hccl
