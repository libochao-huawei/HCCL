/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef HCCL_CCU_TEMP_ALL_GATHER_OMNIPIPE_MESH_1D_MEM2MEM_H
#define HCCL_CCU_TEMP_ALL_GATHER_OMNIPIPE_MESH_1D_MEM2MEM_H

#include "ccu_alg_template_base.h"
#include "ccu_temp_all_gather_omnipipe_common.h"
#include "utils.h"

namespace ops_hccl {

class CcuTempAllGatherOmniPipeMesh1DMem2Mem : public CcuAlgTemplateBase {
public:
    CcuTempAllGatherOmniPipeMesh1DMem2Mem() = default;
    explicit CcuTempAllGatherOmniPipeMesh1DMem2Mem(const OpParam &param, const u32 rankId,
                                                   const std::vector<std::vector<u32>> &subCommRanks);
    ~CcuTempAllGatherOmniPipeMesh1DMem2Mem() override;

    std::string Describe() const override
    {
        return StringFormat("Template of All Gather CCU OmniPipe Mesh1D Mem2Mem skeleton with tempRankSize [%u].",
                            templateRankSize_);
    }

    HCCL_CCU_OMNIPIPE_TEMPLATE_METHODS();
};

} // namespace ops_hccl

#endif // HCCL_CCU_TEMP_ALL_GATHER_OMNIPIPE_MESH_1D_MEM2MEM_H
