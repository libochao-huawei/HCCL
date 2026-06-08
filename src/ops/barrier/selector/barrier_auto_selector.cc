/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "barrier_auto_selector.h"
#include "selector_registry.h"

namespace ops_hccl {

// Barrier 新流程仅在「框间 host-DPU」场景启用，该场景下 Select() 只会走 SelectDPUAlgo；
// 其余场景在 BarrierOutPlace 中已提前回退到 HcclBarrierInner，不会进入算法选择，
// 因此这里不再提供 SelectAicpuAlgo（基类默认 NOT_MATCH 即可）。
SelectorStatus BarrierAutoSelector::SelectDPUAlgo(
    const TopoInfoWithNetLayerDetails *topoInfo, const OpParam &opParam,
    const std::map<HcclCMDType, std::vector<HcclAlgoType>> &configAlgMap,
    std::string &selectAlgName) const
{
    (void)opParam;
    (void)configAlgMap;
    HCCL_DEBUG("[BarrierAutoSelector][%s] start, topoLevelNums[%u]", __func__, topoInfo->topoLevelNums);
    selectAlgName = "InsBarrierMeshNhrDPU";
    HCCL_DEBUG("[BarrierAutoSelector][%s] Algo match[%s]", __func__, selectAlgName.c_str());
    return SelectorStatus::MATCH;
}

REGISTER_SELECTOR_BY_OPTYPE(HcclCMDType::HCCL_CMD_BARRIER, 18, BarrierAutoSelector);

}  // namespace ops_hccl
