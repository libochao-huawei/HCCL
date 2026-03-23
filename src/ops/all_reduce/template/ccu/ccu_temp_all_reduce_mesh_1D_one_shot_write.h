/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef HCCL_CCU_TEMP_ALL_REDUCE_MESH_1D_ONE_SHOT_WRITE_H_
#define HCCL_CCU_TEMP_ALL_REDUCE_MESH_1D_ONE_SHOT_WRITE_H_

#include "ccu_alg_template_base.h"
#include "utils.h"

namespace ops_hccl {

// write 模式 AllReduce Mesh-1D OneShot 模板：
// 使用 MsWriteNb 替代 ReadNb，降低小数据（≤512KB）延迟。
class CcuTempAllReduceMesh1DOneShotWrite : public CcuAlgTemplateBase {
public:
    explicit CcuTempAllReduceMesh1DOneShotWrite(const OpParam &param,
                                                const u32 rankId,
                                                const std::vector<std::vector<u32>> &subCommRanks);
    ~CcuTempAllReduceMesh1DOneShotWrite() override;

    HcclResult CalcRes(HcclComm comm, const OpParam &param, const TopoInfoWithNetLayerDetails *topoInfo,
                       AlgResourceRequest &resourceRequest) override;

    HcclResult KernelRun(const OpParam &param,
                         const TemplateDataParams &templateDataParams,
                         const TemplateResource &templateResource) override;

    std::string Describe() const override
    {
        return StringFormat("Template of CcuTempAllReduceMesh1DOneShotWrite subCommRanks_[0].size() [%u].",
            subCommRanks_[0].size());
    }

    u64 CalcScratchMultiple(BufferType inBuffType, BufferType outBuffType) override;

private:
    uint32_t mySubCommRank_ = 0;
};

} // namespace ops_hccl

#endif // HCCL_CCU_TEMP_ALL_REDUCE_MESH_1D_ONE_SHOT_WRITE_H_
