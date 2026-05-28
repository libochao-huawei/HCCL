/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software; you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "topo_match_multilevel_mesh1d.h"
#include "op_common.h"

namespace ops_hccl {
TopoMatchMultilevelMesh1D::TopoMatchMultilevelMesh1D()
    : TopoMatchBase()
{
}

TopoMatchMultilevelMesh1D::~TopoMatchMultilevelMesh1D()
{
}

HcclResult TopoMatchMultilevelMesh1D::MatchTopo(const HcclComm comm, TopoInfoWithNetLayerDetails* topoInfo,
    AlgHierarchyInfoForAllLevel& algHierarchyInfo)
{
#ifndef AICPU_COMPILE
    // 1层拓扑由TopoMatch1D处理
    CHK_PRT_RET(topoInfo->topoLevelNums < COMM_LAYER_SIZE_2 || topoInfo->topoLevelNums > COMM_LAYER_SIZE_3,
        HCCL_ERROR("[TopoMatchMultilevelMesh1d] topoLevelNum[%u] is invalid, must be 2-3.",
            topoInfo->topoLevelNums),
        HCCL_E_INTERNAL);

    uint32_t myRank;
    CHK_RET(HcclGetRankId(comm, &myRank));
    #ifdef MACRO_DEV_TYPE_NEW
    CHK_PRT_RET(topoInfo->deviceType != DevType::DEV_TYPE_950,
    #else
    CHK_PRT_RET(topoInfo->deviceType != DevType::DEV_TYPE_910_95,
    #endif
        HCCL_ERROR("[TopoMatchMultilevelMesh1d] Rank [%d], deviceType not supported yet.",
            myRank),
        HcclResult::HCCL_E_PARA);

    CHK_PRT_RET((topoInfo->userRankSize == 0),
                HCCL_ERROR("[TopoMatchMultilevelMesh1d] Rank [%d], rankSize is 0.", myRank),
                HcclResult::HCCL_E_PARA);

    // 打平所有拓扑层为单一1D组：所有rank在一个扁平组中，使用mesh1D做通信
    // CalcChannelRequestMesh1D会自动搜索所有netLayer找到正确的链路
    // （同pod用intra-pod链路，跨pod用inter-pod链路）
    std::vector<uint32_t> rankIds;
    for (uint32_t rankId = 0; rankId < topoInfo->userRankSize; rankId++) {
        rankIds.push_back(rankId);
    }
    algHierarchyInfo.infos.resize(1);
    algHierarchyInfo.infos[0].resize(1);
    algHierarchyInfo.infos[0][0] = rankIds;

    HCCL_INFO("[TopoMatchMultilevelMesh1d] Rank [%d], flattened to single 1D group with [%u] ranks, "
              "topoLevelNums [%u]",
        myRank, topoInfo->userRankSize, topoInfo->topoLevelNums);
#endif
    return HcclResult::HCCL_SUCCESS;
}

}  // namespace ops_hccl