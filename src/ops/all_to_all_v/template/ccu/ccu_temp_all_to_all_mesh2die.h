/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef HCCL_CCU_TEMP_ALLTOALL_MESH_2DIE_H
#define HCCL_CCU_TEMP_ALLTOALL_MESH_2DIE_H

#include "utils.h"
#include "ccu_alg_template_base.h"
#include "ccu_kernel_alg_base.h"
#include "template_utils.h"

namespace ops_hccl {

class CcuTempAllToAllMesh2Die : public CcuAlgTemplateBase {
public:
    CcuTempAllToAllMesh2Die() = default;
    CcuTempAllToAllMesh2Die(const OpParam &param, RankId rankId, const std::vector<std::vector<u32>> &subCommRanks);
    ~CcuTempAllToAllMesh2Die() override;

    std::string Describe() const override
    {
        return StringFormat("Template of alltoall ccu mesh 2Die with rankSize[%u]", templateRankSize_);
    }

    HcclResult CalcRes(HcclComm comm, const OpParam &param, const TopoInfoWithNetLayerDetails *topoInfo,
        AlgResourceRequest &resourceRequest) override;

    HcclResult KernelRun(const OpParam &param, const TemplateDataParams &templateDataParams,
        TemplateResource& templateResource) override;

    HcclResult FastLaunch(const OpParam& param, const TemplateFastLaunchCtx& tempFastLaunchCtx) override;

    u64 CalcScratchMultiple(BufferType inBuffType, BufferType outBuffType) override;

private:
    HcclResult PartitionChannels(HcclComm comm, const std::vector<HcclChannelDesc> &channelDescs);

    constexpr static uint32_t DIE_NUM = 2;

    std::vector<std::vector<HcclChannelDesc>> channels_;
    std::vector<std::vector<uint32_t>> rankGroup_;
};

}

#endif
