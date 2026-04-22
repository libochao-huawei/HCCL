/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef COMMON_ALG_TEMPLATE_BASE_H
#define COMMON_ALG_TEMPLATE_BASE_H

#include "template_utils.h"
#include "alg_param.h"

namespace ops_hccl {

class CommonAlgTemplateBase {
public:
    explicit CommonAlgTemplateBase() = default;
    explicit CommonAlgTemplateBase(const OpParam& param, const u32 rankId, const std::vector<std::vector<u32>> &subCommRanks);
    virtual ~CommonAlgTemplateBase() = default;

    virtual std::string Describe() const = 0;
    virtual HcclResult CalcRes(HcclComm comm, const OpParam& param, const TopoInfoWithNetLayerDetails* topoInfo,
        AlgResourceRequest& resourceRequest)
        = 0;
    virtual HcclResult GetRes(AlgResourceRequest& resourceRequest) const = 0;
    virtual u64 GetThreadNum() const = 0;
    virtual u64 CalcScratchMultiple(BufferType inBuffType, BufferType outBuffType) = 0;

    virtual HcclResult KernelRun(
        const OpParam& param, const TemplateDataParams& tempAlgParams, TemplateResource& templateResource)
        = 0;
    virtual HcclResult FastLaunch(const OpParam& param, const TemplateFastLaunchCtx& tempFastLaunchCtx) = 0;
    virtual HcclResult CalcDataSplitByPortGroup(const u64 totalDataCount, const u64 dataTypeSize,
                                                const std::vector<ChannelInfo> &channels,
                                                std::vector<u64> &elemCountOut, std::vector<u64> &sizeOut,
                                                std::vector<u64> &elemOffset)
    {
        elemCountOut.clear();
        sizeOut.clear();
        elemOffset.clear();

        std::vector<u32> portGroups;
        u32 totalPorts = 0;
        u32 taskCount = std::min(channelsPerRank_, (int)channels.size());
        for (u32 i = 0; i < taskCount; i++) {
            const auto &ch = channels[i];
            portGroups.push_back(ch.portGroupSize);
            totalPorts += ch.portGroupSize;
            HCCL_INFO("[CalcDataSplitByPortGroup] ch.portGroupSize[%u], totalPorts[%u], channelsPerRank_[%u]",
                    ch.portGroupSize, totalPorts, channelsPerRank_);
        }

        u32 channelsize = portGroups.size();
        u64 accumCount = 0;
        u64 offset = 0;
        for (u32 channelIdx = 0; channelIdx < channelsize; channelIdx++) {
            u64 elemCount = 0;
            u64 elemSize = 0;
            if (channelIdx == channelsize - 1) {
                elemCount = totalDataCount - accumCount;
            } else {
                CHK_PRT_RET(totalPorts == 0,
                            HCCL_ERROR("[CalcDataSplitByPortGroup] totalPorts [%u] is 0.", totalPorts),
                            HcclResult::HCCL_E_INTERNAL);
                elemCount = static_cast<u64>((totalDataCount * portGroups[channelIdx]) / totalPorts);
            }
            elemOffset.push_back(offset);
            elemCountOut.push_back(elemCount);
            elemSize = elemCount * dataTypeSize;
            sizeOut.push_back(elemSize);
            offset += elemSize;
            accumCount += elemCount;
        }

        return HcclResult::HCCL_SUCCESS;
    }

    protected:
    u32             channelsPerRank_    = 1;

    
};

} // namespace Hccl

#endif // COMMON_ALG_TEMPLATE_BASE_H