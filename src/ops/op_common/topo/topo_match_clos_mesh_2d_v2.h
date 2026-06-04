/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef TOPO_MATCH_CLOS_MESH_2D_V2_H
#define TOPO_MATCH_CLOS_MESH_2D_V2_H

#include "topo_match_base.h"

namespace ops_hccl {

class TopoMatchClosMesh2DV2 : public TopoMatchBase {
public:
    explicit TopoMatchClosMesh2DV2();
    ~TopoMatchClosMesh2DV2() override;

    std::string Describe() const override
    {
        return "TopoMatchClosMesh2DV2: layer 0 intra-pod (X), layer 1 inter-pod (Y).";
    }

    HcclResult MatchTopo(const HcclComm comm, TopoInfoWithNetLayerDetails* topoInfo,
                         AlgHierarchyInfoForAllLevel& algHierarchyInfo) override;

private:
    HcclResult BuildXAxis(const HcclComm comm, const u32 myRank, AlgHierarchyInfoForAllLevel& algHierarchyInfo);
    HcclResult BuildYAxis(const HcclComm comm, const u32 myRank, u32 xRankSize,
                          AlgHierarchyInfoForAllLevel& algHierarchyInfo);

    template<typename T>
    std::string PrintCArray(const T* values, const u32 valueNum) const
    {
        std::ostringstream oss;
        for (u32 i = 0; i < valueNum; i++) {
            oss << values[i] << " ";
        }
        return oss.str();
    }
};

}  // namespace ops_hccl

#endif
