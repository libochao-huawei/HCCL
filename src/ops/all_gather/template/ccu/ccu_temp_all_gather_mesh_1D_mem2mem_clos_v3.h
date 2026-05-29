/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef HCCL_CCU_TEMP_ALL_GATHER_MESH_1D_MEM2MEM_CLOS_V3_H
#define HCCL_CCU_TEMP_ALL_GATHER_MESH_1D_MEM2MEM_CLOS_V3_H

#include "utils.h"
#include "ccu_alg_template_base.h"

namespace ops_hccl {

class CcuTempAllGatherMesh1DMem2MemClosV3 : public CcuAlgTemplateBase {
public:
    CcuTempAllGatherMesh1DMem2MemClosV3() = default;
    explicit  CcuTempAllGatherMesh1DMem2MemClosV3(const OpParam& param,
                                                const u32 rankId, // 传通信域的rankId，userRank
                                                const std::vector<std::vector<u32>> &subCommRanks);

    ~CcuTempAllGatherMesh1DMem2MemClosV3() override;

    std::string Describe() const override
    {
        return StringFormat("Template of All Gather ccu mesh 1D Mem2Mem ClosV3 with tempRankSize [%u].",
                            subCommRanks_[0].size());
    }

    HcclResult CalcRes(HcclComm comm, const OpParam& param, const TopoInfoWithNetLayerDetails* topoInfo,
                       AlgResourceRequest& resourceRequest) override;
    HcclResult GetRes(AlgResourceRequest& resourceRequest) const override;
    HcclResult KernelRun(const OpParam& param,
                         const TemplateDataParams& templateDataParams,
                         TemplateResource& templateResource) override;
    u64 GetThreadNum() const override;
    HcclResult FastLaunch(const OpParam& param, const TemplateFastLaunchCtx& tempFastLaunchCtx) override;
    u64 CalcScratchMultiple(BufferType inBuffType, BufferType outBuffType) override;

private:
    uint32_t mySubCommRank_ = 0;
};

}// namespace ops_hccl

#endif// HCCL_CCU_TEMP_ALL_GATHER_MESH_1D_MEM2MEM_CLOS_V3_H
