/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software; you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef TOPO_MATCH_MULTILEVEL_MESH1D
#define TOPO_MATCH_MULTILEVEL_MESH1D

#include "topo_match_base.h"

namespace ops_hccl {
class TopoMatchMultilevelMesh1D : public TopoMatchBase {
public:
    explicit TopoMatchMultilevelMesh1D();
    ~TopoMatchMultilevelMesh1D() override;

    std::string Describe() const override
    {
        return "Topo Match for AllGatherV MultilevelMesh1D: flatten multi-layer topo into single 1D group.";
    }

    HcclResult MatchTopo(const HcclComm comm, TopoInfoWithNetLayerDetails* topoInfo,
        AlgHierarchyInfoForAllLevel& algHierarchyInfo) override;
};
}  // namespace ops_hccl

#endif  // !TOPO_MATCH_MULTILEVEL_MESH1D