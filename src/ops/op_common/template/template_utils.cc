/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "template_utils.h"
namespace ops_hccl {

HcclResult GetAlgRank(const u32 virtRank, const std::vector<u32> &rankIds, u32 &algRank)
{
    std::vector<u32>::const_iterator topoVecIter = std::find(rankIds.begin(), rankIds.end(), virtRank);
    CHK_PRT_RET(topoVecIter == rankIds.end(), HCCL_ERROR("[GetAlgRank] Invalid virtual Rank!"),
                HcclResult::HCCL_E_PARA);
    algRank = distance(rankIds.begin(), topoVecIter);

    return HcclResult::HCCL_SUCCESS;
}

u32 GetNHRStepNum(u32 rankSize)
{
    u32 nSteps = 0;
    for (u32 tmp = rankSize - 1; tmp != 0; tmp >>= 1, nSteps++) {
    }
    HCCL_DEBUG("[NHRBase][GetStepNumInterServer] rankSize[%u] nSteps[%u]", rankSize, nSteps);

    return nSteps;
}

HcclResult CalcDataSplitRateForLinks(const std::vector<ChannelInfo> &channels, std::vector<float> &datasplitRate) 
{
    //取第一个对端的link数量作为数据切分的依据
    std::vector<u8> channelPortGroupSizes;
    channelPortGroupSizes.resize(channels.size());
    for(u32 channelIdx = 0; channelIdx < channels.size(); channelIdx++) {
        const ChannelInfo& channelInfo = channels[channelIdx];
        channelPortGroupSizes[channelIdx] = channelInfo.portGroupSize;
    }
    u32 totalPortNum = std::accumulate(channelPortGroupSizes.begin(), channelPortGroupSizes.end(), 0);
    if (totalPortNum == 0) {
        HCCL_ERROR("[CalcDataSplitRateForLinks] totalPortNum is zero");
        return HcclResult::HCCL_E_INTERNAL;
    }
    for (u32 channelIdx = 0; channelIdx < channelPortGroupSizes.size(); channelIdx++) {
        datasplitRate[channelIdx] = static_cast<float>(channelPortGroupSizes[channelIdx]) / totalPortNum;
    }
    return HcclResult::HCCL_SUCCESS;
}

DataSlice CalcDataSliceForLinks(const DataSlice& recvSrcSliceAllLinks, std::vector<flost> dataSplitRate, u32 j, HcclDataType dataType_)
{
    void* addr = recvSrcSliceAllLinks.addr_;
    u64 offset = recvSrcSliceAllLinks.offset_;
    u64 size = recvSrcSliceAllLinks.size_;
    u64 count = recvSrcSliceAllLinks.count_;
    u64 accSize = 0;
    u64 typeSize = DATATYPE_SIZE_TABLE[dataType_];
    u64 channelNum = dataSplitRate.size();
    std::vector<DataSlice> dataSliceForLinks(channelNum);
    HCCL_INFO("[CalcDataSliceForLinks] Slice data for channels");
    for (u32 channelIdx = 0; channelIdx < channelNum; channelIdx++) { 
        if (channelIdx != channelNum - 1) {
            dataSliceForLinks[channelIdx].size_ = static_cast<u64>(static_cast<float>(count) * dataSplitRate[channelIdx]) * typeSize;
        } else {
            dataSliceForLinks[channelIdx].size_ = size - accSize;
        }
        dataSliceForLinks[channelIdx].offset_ = offset + accSize;
        accSize += dataSliceForLinks[channelIdx].size_;
        dataSliceForLinks[channelIdx].addr_ = addr;
    }
    return dataSliceForLinks[j];
}
}