/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "topo_match_ubx_1d.h"

namespace ops_hccl {
TopoMatchUBX1d::TopoMatchUBX1d()
    : TopoMatchUBX()
{
}

TopoMatchUBX1d::~TopoMatchUBX1d()
{
}

HcclResult TopoMatchUBX1d::MatchTopo(const HcclComm comm,
                                        TopoInfoWithNetLayerDetails* topoInfo,
                                        AlgHierarchyInfoForAllLevel& algHierarchyInfo)
{
    return TopoMatchUBX::MatchTopo(comm, topoInfo, algHierarchyInfo);
}

HcclResult TopoMatchUBX1d::TopoForLayer1(const HcclComm comm, uint32_t layer0Size, const uint32_t myRank,
                                                  AlgHierarchyInfoForAllLevel& algHierarchyInfo) const
{
    HCCL_DEBUG("[TopoMatchUBX1d::MeshTopoForLayer1] layer0Size [%d]", layer0Size);
#ifndef AICPU_COMPILE
    // 1. 查出layer 1的所有ranks
    uint32_t *topoInsts;
    uint32_t topoInstNum = 0;
    CHK_RET(HcclRankGraphGetTopoInstsByLayer(comm, 1, &topoInsts, &topoInstNum));
    CHK_PRT_RET(
        (topoInstNum != NET_INST_NUM_1),
        HCCL_ERROR("[TopoMatchUBX1d::MeshTopoForLayer1] layer1 topoInstNum [%d], Invalid topo.", topoInstNum),
        HcclResult::HCCL_E_PARA);
    uint32_t* ranks;
    uint32_t rankNum;
    CHK_RET(HcclRankGraphGetRanksByTopoInst(comm, 1, topoInsts[0], &ranks, &rankNum));
    HCCL_DEBUG("[TopoMatchUBX1d::MeshTopoForLayer1] Rank [%d], all [%u] ranks in layer1", myRank, rankNum);
    // 2. 取出每张卡，作为layer1的ranks
    std::vector<uint32_t> rankVecLayer1;
    for (uint32_t i = 0; i < rankNum; i++) {
        uint32_t rankId = ranks[i];
        if (myRank == rankId) {
            rankVecLayer1.push_back(rankId);
            continue;
        }

        CommLink *links;
        uint32_t linkNum = 0;
        HcclRankGraphGetLinks(comm, 1, myRank, rankId, &links, &linkNum);
        if (linkNum == 0) {
            continue;
        }
        rankVecLayer1.push_back(rankId);
    }
    algHierarchyInfo.infos[1].push_back({rankVecLayer1});
#endif
    return HcclResult::HCCL_SUCCESS;
}

}  // namespace ops_hccl