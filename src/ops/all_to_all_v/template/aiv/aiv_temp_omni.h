/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef HCCL_AIV_TEMP_OMNI_H
#define HCCL_AIV_TEMP_OMNI_H

#include "aiv_alg_template_base.h"

namespace ops_hccl {
struct XmlInfo;
struct OmniChannelInfo;

class AivTempOmni : public AivAlgTemplateBase {
public:
    explicit AivTempOmni(const OpParam& param, const u32 rankId, const std::vector<std::vector<u32>> &subCommRanks);
    ~AivTempOmni() override;

    std::string Describe() const override
    {
        return StringFormat("Template of aiv omni with tempRankSize [%u].", tempRankSize_);
    }

    HcclResult CalcRes(HcclComm comm, const OpParam& param, const TopoInfoWithNetLayerDetails* topoInfo,
        AlgResourceRequest& resourceRequest) override;
    HcclResult CalcRes(HcclComm comm, const OpParam& param, const TopoInfoWithNetLayerDetails* topoInfo,
        AlgResourceRequest& resourceRequest, const XmlInfo& xmlInfo);
    HcclResult KernelRun(const OpParam& param, const TemplateDataParams& templateDataParams,
        const TemplateResource& templateResource) override;
    HcclResult CalNumBlocks(u32& numBlocks, u64 dataSize, u32 numBlocksLimit) override;
    u64 CalcScratchMultiple(BufferType inBufferType, BufferType outBufferType) override;

private:
    HcclResult CalcChannelRequestOmni(HcclComm comm, const OpParam& param,
        const TopoInfoWithNetLayerDetails* topoInfo, const std::vector<std::vector<u32>>& subcommInfo,
        const std::vector<std::map<u32, OmniChannelInfo>>& mapchannelInfo, std::vector<HcclChannelDesc>& channels);
    HcclResult BuildInstructionBuffer(HcclComm comm, const OpParam& param, const XmlInfo& xmlInfo);

    u32 tempRankSize_ = 0;
    u32 numBlocks_ = 1;
};
} // namespace ops_hccl

#endif // HCCL_AIV_TEMP_OMNI_H
