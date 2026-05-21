/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef TOPO_MATCH_CLOS_MESH_2D_UBX_V2_H
#define TOPO_MATCH_CLOS_MESH_2D_UBX_V2_H

#include "topo_match_base.h"

namespace ops_hccl {

class TopoMatchClosMesh2DUBXV2 : public TopoMatchBase {
public:
    explicit TopoMatchClosMesh2DUBXV2();
    ~TopoMatchClosMesh2DUBXV2() override;

    std::string Describe() const override
    {
        return "TopoMatchClosMesh2DUBXV2: UBX 4x4 2D (layer0 Mesh X, layer1 Clos Y).";
    }

    HcclResult MatchTopo(const HcclComm comm, TopoInfoWithNetLayerDetails* topoInfo,
                         AlgHierarchyInfoForAllLevel& algHierarchyInfo) override;
};

}  // namespace ops_hccl

#endif
