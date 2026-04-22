/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef HCCLV2_CCU_TEMP_ALL_TO_ALL_MESH_1D_2DIE_H_
#define HCCLV2_CCU_TEMP_ALL_TO_ALL_MESH_1D_2DIE_H_

#include "utils.h"
#include "ccu_alg_template_base.h"

namespace ops_hccl {

class CcuTempAllToAllMesh1D2Die : public CcuAlgTemplateBase {
public:
    CcuTempAllToAllMesh1D2Die() = default;
    explicit CcuTempAllToAllMesh1D2Die(const OpParam &param, RankId rankId, const std::vector<std::vector<u32>> &subCommRanks);
    ~CcuTempAllToAllMesh1D2Die() override;

    std::string Describe() const override
    {
        return StringFormat("Template of alltoall ccu mesh 2Die with rankSize[%u]", templateRankSize_);
    }

    HcclResult CalcRes(HcclComm comm, const OpParam &param, const TopoInfoWithNetLayerDetails *topoInfo,
        AlgResourceRequest &resourceRequest) override;

    HcclResult KernelRun(const OpParam &param, const TemplateDataParams &templateDataParams,
        TemplateResource& templateResource) override;
private:
    HcclResult PartitionChannels(HcclComm comm, const std::vector<HcclChannelDesc> &channelDescs, uint32_t &meshDieId);

    const uint32_t DIE_NUM = 2; // 2Die

    std::map<uint32_t, std::vector<HcclChannelDesc>> channels_; // key is DieId
    std::map<uint32_t, RankGroup> rankGroup_;
};

} // namespace Hccl
#endif // HCCLV2_CCU_TEMP_ALL_TO_ALL_MESH_1D_2DIE_H_