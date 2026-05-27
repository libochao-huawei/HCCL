/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "topo_match_clos_mesh_2d_v2.h"
#include "hccl_rank_graph_dl.h"
#include "hccl_comm.h"
#include <sstream>
#include <set>

namespace ops_hccl {

TopoMatchClosMesh2DV2::TopoMatchClosMesh2DV2()
    : TopoMatchBase()
{
}

TopoMatchClosMesh2DV2::~TopoMatchClosMesh2DV2()
{
}

HcclResult TopoMatchClosMesh2DV2::MatchTopo(const HcclComm comm, TopoInfoWithNetLayerDetails* topoInfo,
                                            AlgHierarchyInfoForAllLevel& algHierarchyInfo)
{
    u32 myRank;
    CHK_RET(HcclGetRankId(comm, &myRank));

    algHierarchyInfo.infos.resize(COMM_LAYER_SIZE_2);

    CHK_RET(BuildXAxis(comm, myRank, algHierarchyInfo));
    u32 xRankSize = algHierarchyInfo.infos[0].empty() ? 0 : algHierarchyInfo.infos[0][0].size();

    if (topoInfo->topoLevelNums > 1) {
        CHK_RET(BuildYAxis(comm, myRank, xRankSize, algHierarchyInfo));
    } else {
        algHierarchyInfo.infos[1].push_back({myRank});
    }

    HCCL_INFO("[TopoMatchClosMesh2DV2] Rank[%u] matched. X=%u, Y=%u.",
              myRank, xRankSize,
              algHierarchyInfo.infos[1].empty() ? 0 : algHierarchyInfo.infos[1][0].size());
    return HCCL_SUCCESS;
}

HcclResult TopoMatchClosMesh2DV2::BuildXAxis(const HcclComm comm, const u32 myRank,
                                             AlgHierarchyInfoForAllLevel& algHierarchyInfo)
{
    u32 *topoInsts;
    u32 topoInstNum = 0;
    CHK_RET(HcclRankGraphGetTopoInstsByLayer(comm, 0, &topoInsts, &topoInstNum));

    u32 *ranks = nullptr;
    u32 rankNum = 0;
    bool found = false;

    for (u32 idx = 0; idx < topoInstNum; idx++) {
        u32 *tmpRanks;
        u32 tmpRankNum;
        CHK_RET(HcclRankGraphGetRanksByTopoInst(comm, 0, topoInsts[idx], &tmpRanks, &tmpRankNum));
        for (u32 k = 0; k < tmpRankNum; k++) {
            if (tmpRanks[k] == myRank) {
                ranks = tmpRanks;
                rankNum = tmpRankNum;
                found = true;
                break;
            }
        }
        if (found) {
            break;
        }
    }
    CHK_PRT_RET(!found,
                HCCL_ERROR("[TopoMatchClosMesh2DV2][BuildXAxis] Rank[%u] not in any Layer0 instance.", myRank),
                HCCL_E_PARA);

    std::vector<u32> rowRanks(ranks, ranks + rankNum);
    std::sort(rowRanks.begin(), rowRanks.end());
    algHierarchyInfo.infos[0].push_back(rowRanks);

    HCCL_DEBUG("[TopoMatchClosMesh2DV2][BuildXAxis] Rank[%u] X-axis[%u ranks]: [%s]",
               myRank, rankNum, PrintCArray<u32>(ranks, rankNum).c_str());
    return HCCL_SUCCESS;
}

HcclResult TopoMatchClosMesh2DV2::BuildYAxis(const HcclComm comm, const u32 myRank, u32 xRankSize,
                                             AlgHierarchyInfoForAllLevel& algHierarchyInfo)
{
    if (xRankSize == 0) {
        return HCCL_SUCCESS;
    }

    u32 *topoInsts;
    u32 topoInstNum = 0;
    CHK_RET(HcclRankGraphGetTopoInstsByLayer(comm, 1, &topoInsts, &topoInstNum));
    CHK_PRT_RET(topoInstNum == 0,
                HCCL_ERROR("[TopoMatchClosMesh2DV2][BuildYAxis] layer1 has no topo instances."),
                HCCL_E_PARA);

    u32 mySlotIdx = myRank % xRankSize;
    std::set<u32> colRankSet;

    for (u32 idx = 0; idx < topoInstNum; idx++) {
        u32 *ranks;
        u32 rankNum;
        CHK_RET(HcclRankGraphGetRanksByTopoInst(comm, 1, topoInsts[idx], &ranks, &rankNum));
        for (u32 i = 0; i < rankNum; i++) {
            if (ranks[i] % xRankSize == mySlotIdx) {
                colRankSet.insert(ranks[i]);
            }
        }
    }

    std::vector<u32> colRanks(colRankSet.begin(), colRankSet.end());
    std::sort(colRanks.begin(), colRanks.end());

    HCCL_DEBUG("[TopoMatchClosMesh2DV2][BuildYAxis] Rank[%u] slotIdx[%u] Y-axis[%zu ranks] from %u instances.",
               myRank, mySlotIdx, colRanks.size(), topoInstNum);
    algHierarchyInfo.infos[1].push_back(colRanks);
    return HCCL_SUCCESS;
}

}  // namespace ops_hccl
