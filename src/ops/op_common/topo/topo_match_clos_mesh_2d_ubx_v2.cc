/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "topo_match_clos_mesh_2d_ubx_v2.h"
#include "hccl_rank_graph_dl.h"
#include "hccl_comm.h"
#include <set>

namespace ops_hccl {

TopoMatchClosMesh2DUBXV2::TopoMatchClosMesh2DUBXV2()
    : TopoMatchBase()
{
}

TopoMatchClosMesh2DUBXV2::~TopoMatchClosMesh2DUBXV2()
{
}

HcclResult TopoMatchClosMesh2DUBXV2::MatchTopo(const HcclComm comm, TopoInfoWithNetLayerDetails* topoInfo,
                                               AlgHierarchyInfoForAllLevel& algHierarchyInfo)
{
    u32 myRank;
    CHK_RET(HcclGetRankId(comm, &myRank));

    algHierarchyInfo.infos.resize(COMM_LAYER_SIZE_2);

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
                HCCL_ERROR("[TopoMatchClosMesh2DUBXV2] Rank[%u] not in any Layer0 instance.", myRank),
                HCCL_E_PARA);

    std::vector<u32> rowRanks(ranks, ranks + rankNum);
    std::sort(rowRanks.begin(), rowRanks.end());
    algHierarchyInfo.infos[0].push_back(rowRanks);
    u32 xRankSize = rankNum;

    CHK_RET(HcclRankGraphGetTopoInstsByLayer(comm, 1, &topoInsts, &topoInstNum));
    CHK_PRT_RET(topoInstNum == 0,
                HCCL_ERROR("[TopoMatchClosMesh2DUBXV2] layer1 has no topo instances."),
                HCCL_E_PARA);

    u32 mySlotIdx = myRank % xRankSize;
    std::set<u32> colRankSet;
    for (u32 idx = 0; idx < topoInstNum; idx++) {
        u32 *tmpRanks;
        u32 tmpRankNum;
        CHK_RET(HcclRankGraphGetRanksByTopoInst(comm, 1, topoInsts[idx], &tmpRanks, &tmpRankNum));
        for (u32 i = 0; i < tmpRankNum; i++) {
            if (tmpRanks[i] % xRankSize == mySlotIdx) {
                colRankSet.insert(tmpRanks[i]);
            }
        }
    }

    std::vector<u32> colRanks(colRankSet.begin(), colRankSet.end());
    std::sort(colRanks.begin(), colRanks.end());
    algHierarchyInfo.infos[1].push_back(colRanks);

    HCCL_INFO("[TopoMatchClosMesh2DUBXV2] Rank[%u] matched. X=%u, Y=%zu.",
              myRank, xRankSize, colRanks.size());
    return HCCL_SUCCESS;
}

}  // namespace ops_hccl
