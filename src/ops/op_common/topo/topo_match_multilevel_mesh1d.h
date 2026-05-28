/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
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
/**
 * TopoMatchMultilevelMesh1D: 拓扑匹配器，支持最多3层拓扑组网（跨超节点），
 * 每层始终使用 mesh1D 算法。
 *
 * 层级定义:
 *   Layer 0: 节点内（pod内）Mesh1D — 同一超节点内的卡间通信
 *   Layer 1: 节点间（跨pod）Mesh1D — 不同超节点间同序号卡的通信
 *   Layer 2: 超节点间 Mesh1D — 超节点组间同子序号卡的通信（可选，取决于拓扑层级数）
 */
class TopoMatchMultilevelMesh1D : public TopoMatchBase {
public:
    explicit TopoMatchMultilevelMesh1D();
    ~TopoMatchMultilevelMesh1D() override;

    std::string Describe() const override
    {
        return "Topo Match for AllGatherV Multi-level Mesh1D: up to 3 layers (cross-supernode), always mesh1D.";
    }

    HcclResult MatchTopo(const HcclComm comm, TopoInfoWithNetLayerDetails* topoInfo,
        AlgHierarchyInfoForAllLevel& algHierarchyInfo) override;

private:
    HcclResult TopoForLayer0(const HcclComm comm, uint32_t& layer0Size, const uint32_t myRank,
        AlgHierarchyInfoForAllLevel& algHierarchyInfo, uint32_t gcdInstSize = 0) const;
    HcclResult TopoForLayer1(const HcclComm comm, uint32_t netLayer, uint32_t& layer0Size,
        const uint32_t myRank, AlgHierarchyInfoForAllLevel& algHierarchyInfo) const;
    HcclResult TopoForLayer2(const HcclComm comm, uint32_t netLayer, uint32_t& layer0Size,
        uint32_t& layer1Size, const uint32_t myRank, AlgHierarchyInfoForAllLevel& algHierarchyInfo) const;

    bool CheckVecElementAllSame(const uint32_t* instSizeList, uint32_t listSize) const;
    uint32_t GcdTwo(uint32_t a, uint32_t b) const;
    uint32_t GcdOfInstSizeList(const uint32_t* instSizeList, uint32_t listSize) const;

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

#endif  // !TOPO_MATCH_MULTILEVEL_MESH1D