/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "aicpu/ins_temp_reduce_scatter_mesh_1D_Z_axis_detour.h"

namespace ops_hccl {

InsTempReduceScatterMesh1DZAxisDetour::InsTempReduceScatterMesh1DZAxisDetour(
    const OpParam& param, const u32 rankId,
    const std::vector<std::vector<u32>> &subCommRanks)
    : InsTempReduceScatterMesh1D(param, rankId, subCommRanks)
{
}

InsTempReduceScatterMesh1DZAxisDetour::~InsTempReduceScatterMesh1DZAxisDetour()
{
}

HcclResult InsTempReduceScatterMesh1DZAxisDetour::CalcRes(
    HcclComm comm, const OpParam& param, const TopoInfoWithNetLayerDetails* topoInfo,
    AlgResourceRequest& resourceRequest)
{
    CHK_PRT_RET(topoInfo == nullptr,
        HCCL_ERROR("[InsTempReduceScatterMesh1DZAxisDetour][CalcRes] topoInfo is nullptr"), HCCL_E_PARA);
    std::vector<HcclChannelDesc> level0Channels;
    CHK_RET(CalcChannelRequestMesh1DLevel0(comm, param, topoInfo, subCommRanks_, level0Channels));
    std::vector<HcclChannelDesc> level1Channels;
    CHK_RET(CalcChannelRequestMesh1DLevel1(comm, param, topoInfo, subCommRanks_, level1Channels));
    std::vector<HcclChannelDesc> mergedChannels;
    mergedChannels.insert(mergedChannels.end(), level0Channels.begin(), level0Channels.end());
    mergedChannels.insert(mergedChannels.end(), level1Channels.begin(), level1Channels.end());
    resourceRequest.channels.push_back(mergedChannels);
    channelsPerRank_ = CalcChannelsPerRank(mergedChannels);
    level0ChannelNumPerRank_ = CalcChannelsPerRank(level0Channels);
    level1ChannelNumPerRank_ = CalcChannelsPerRank(level1Channels);
    CHK_RET(GetRes(resourceRequest));
    HCCL_DEBUG("[InsTempReduceScatterMesh1DZAxisDetour][CalcRes] myRank[%u], channelsPerRank_[%u], "
               "level0ChannelNum[%zu], level1ChannelNum[%zu], notifyNumOnMainThread[%u], slaveThreadNum[%u]",
               myRank_, channelsPerRank_, level0Channels.size(), level1Channels.size(),
               resourceRequest.notifyNumOnMainThread, resourceRequest.slaveThreadNum);
    return HCCL_SUCCESS;
}

u64 InsTempReduceScatterMesh1DZAxisDetour::GetThreadNum() const
{
    if (templateRankSize_ <= 1) {
        return 1;
    }
    return (templateRankSize_ - 1) * channelsPerRank_ + 1;
}

HcclResult InsTempReduceScatterMesh1DZAxisDetour::CalcDataSplitByPortGroup(
    const u64 totalDataCount, const u64 dataTypeSize,
    const std::vector<ChannelInfo> &channels,
    std::vector<u64> &elemCountOut, std::vector<u64> &sizeOut,
    std::vector<u64> &elemOffset)
{
    return CalcDataSplitByPortGroupZAxisDetour(totalDataCount, dataTypeSize, channels,
        elemCountOut, sizeOut, elemOffset,
        level0ChannelNumPerRank_, level1ChannelNumPerRank_, level0DataRatio_);
}

} // namespace ops_hccl
