/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "ins_temp_all_gather_mesh_1D_no_memcpy.h"
#include "alg_data_trans_wrapper.h"
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

InsTempAllGatherMesh1DNoMemcpy::InsTempAllGatherMesh1DNoMemcpy(
    const OpParam &param, const u32 rankId, const std::vector<std::vector<u32>> &subCommRanks)
    : InsTempAllGatherMesh1D(param, rankId, subCommRanks)
{
}

InsTempAllGatherMesh1DNoMemcpy::~InsTempAllGatherMesh1DNoMemcpy() {}

HcclResult InsTempAllGatherMesh1DNoMemcpy::RunAllGatherMesh(
    const std::vector<ThreadHandle> &threads, const std::map<u32, std::vector<ChannelInfo>> &channels)
{
    HCCL_INFO("[InsTempAllGatherMesh1DNoMemcpy] RunAllGatherMesh Rank[%u].", myRank_);
    if (templateRankSize_ <= 1) {
        return HCCL_SUCCESS;
    }

    u32 myAlgRank = 0;
    CHK_RET(GetAlgRank(myRank_, subCommRanks_[0], myAlgRank));
    const u32 dataTypeSize = DATATYPE_SIZE_TABLE[dataType_];

    for (u32 threadIdx = 0; threadIdx < subCommRanks_[0].size() - 1; threadIdx++) {
        u32 connectedRank = subCommRanks_[0][(myAlgRank + 1 + threadIdx) % subCommRanks_[0].size()];
        u32 connectedAlgRank = 0;
        CHK_RET(GetAlgRank(connectedRank, subCommRanks_[0], connectedAlgRank));

        CHK_PRT_RET(threadIdx >= threads.size() || channels.count(connectedRank) == 0 ||
                        channels.at(connectedRank).empty(),
                    HCCL_ERROR("[InsTempAllGatherMesh1DNoMemcpy] Rank[%u] invalid peer/thread. threadIdx[%u] "
                               "threads[%zu] peer[%u] channels[%zu]",
                               myRank_, threadIdx, threads.size(), connectedRank, channels.size()),
                    HcclResult::HCCL_E_INTERNAL);

        const ChannelInfo &linkRemote = channels.at(connectedRank)[0];
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
            CHK_RET(CheckRemoteOutputRange("InsTempAllGatherMesh1DNoMemcpy", myRank_, connectedRank, linkRemote,
                                           txDstOffset, sliceSize));
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
        HcclResult ret = SendRecvWrite(sendRecvInfo, threads[threadIdx]);
        CHK_PRT_RET(ret != HCCL_SUCCESS,
                    HCCL_ERROR("[InsTempAllGatherMesh1DNoMemcpy] SendRecvWrite failed. rank[%u] peer[%u] "
                               "threadIdx[%u] ret[%d]",
                               myRank_, connectedRank, threadIdx, ret),
                    ret);
    }
    return HCCL_SUCCESS;
}

}  // namespace ops_hccl
