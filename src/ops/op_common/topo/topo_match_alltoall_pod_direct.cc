/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "topo_match_alltoall_pod_direct.h"
#include <algorithm>
#include <sstream>

namespace ops_hccl {

TopoMatchAlltoAllPodDirect::TopoMatchAlltoAllPodDirect()
    : TopoMatchBase()
{
}

TopoMatchAlltoAllPodDirect::~TopoMatchAlltoAllPodDirect()
{
}

HcclResult TopoMatchAlltoAllPodDirect::BuildLocalPod(const HcclComm comm, const u32 myRank, u32 &localPodSize,
                                                     AlgHierarchyInfoForAllLevel &algHierarchyInfo) const
{
#ifndef AICPU_COMPILE
    u32 *topoInsts;
    u32 topoInstNum = 0;
    CHK_RET(HcclRankGraphGetTopoInstsByLayer(comm, 0, &topoInsts, &topoInstNum));

    if (topoInstNum == NET_INST_NUM_0) {
        algHierarchyInfo.infos[0].push_back({myRank});
        localPodSize = 1;
        return HCCL_SUCCESS;
    }

    for (u32 idx = 0; idx < topoInstNum; idx++) {
        u32 *ranks;
        u32 rankNum;
        CHK_RET(HcclRankGraphGetRanksByTopoInst(comm, 0, topoInsts[idx], &ranks, &rankNum));
        auto iter = std::find(ranks, ranks + rankNum, myRank);
        if (iter == ranks + rankNum) {
            continue;
        }

        std::vector<u32> localPodRanks(ranks, ranks + rankNum);
        std::sort(localPodRanks.begin(), localPodRanks.end());
        algHierarchyInfo.infos[0].push_back(localPodRanks);
        localPodSize = static_cast<u32>(localPodRanks.size());
        HCCL_INFO("[TopoMatchAlltoAllPodDirect] Rank[%u] local pod size[%u] ranks[%s]",
                  myRank, localPodSize, PrintCArray(localPodRanks.data(), localPodSize).c_str());
        return HCCL_SUCCESS;
    }

    HCCL_ERROR("[TopoMatchAlltoAllPodDirect] Rank[%u] not found in layer0 topo instances.", myRank);
    return HcclResult::HCCL_E_PARA;
#else
    (void)comm;
    (void)myRank;
    (void)localPodSize;
    (void)algHierarchyInfo;
    return HCCL_SUCCESS;
#endif
}

HcclResult TopoMatchAlltoAllPodDirect::BuildDirectInterPeers(const HcclComm comm, const u32 myRank,
                                                             const u32 localPodSize,
                                                             AlgHierarchyInfoForAllLevel &algHierarchyInfo) const
{
#ifndef AICPU_COMPILE
    if (localPodSize == 0) {
        HCCL_ERROR("[TopoMatchAlltoAllPodDirect] localPodSize is 0.");
        return HcclResult::HCCL_E_INTERNAL;
    }

    u32 *topoInsts;
    u32 topoInstNum = 0;
    CHK_RET(HcclRankGraphGetTopoInstsByLayer(comm, 1, &topoInsts, &topoInstNum));
    CHK_PRT_RET(topoInstNum == NET_INST_NUM_0,
                HCCL_ERROR("[TopoMatchAlltoAllPodDirect] layer1 has no topo instances."),
                HcclResult::HCCL_E_PARA);

    std::vector<u32> interRanks;
    interRanks.push_back(myRank);
    const u32 myPodIdx = myRank / localPodSize;

    for (u32 instIdx = 0; instIdx < topoInstNum; instIdx++) {
        u32 *ranks;
        u32 rankNum;
        CHK_RET(HcclRankGraphGetRanksByTopoInst(comm, 1, topoInsts[instIdx], &ranks, &rankNum));
        for (u32 i = 0; i < rankNum; i++) {
            u32 rankId = ranks[i];
            if (rankId == myRank || rankId / localPodSize == myPodIdx) {
                continue;
            }

            CommLink *links;
            u32 linkNum = 0;
            CHK_RET(HcclRankGraphGetLinks(comm, 1, myRank, rankId, &links, &linkNum));
            if (linkNum == 0) {
                continue;
            }
            interRanks.push_back(rankId);
        }
    }

    std::sort(interRanks.begin(), interRanks.end());
    interRanks.erase(std::unique(interRanks.begin(), interRanks.end()), interRanks.end());
    algHierarchyInfo.infos[1].push_back(interRanks);
    HCCL_INFO("[TopoMatchAlltoAllPodDirect] Rank[%u] direct inter size[%zu] ranks[%s]",
              myRank, interRanks.size(),
              interRanks.empty() ? "" : PrintCArray(interRanks.data(), static_cast<u32>(interRanks.size())).c_str());
    return HCCL_SUCCESS;
#else
    (void)comm;
    (void)myRank;
    (void)localPodSize;
    (void)algHierarchyInfo;
    return HCCL_SUCCESS;
#endif
}

HcclResult TopoMatchAlltoAllPodDirect::MatchTopo(const HcclComm comm, TopoInfoWithNetLayerDetails *topoInfo,
                                                 AlgHierarchyInfoForAllLevel &algHierarchyInfo)
{
#ifndef AICPU_COMPILE
    CHK_PRT_RET(topoInfo->topoLevelNums == 0 || topoInfo->topoLevelNums > COMM_LAYER_SIZE_2,
                HCCL_ERROR("[TopoMatchAlltoAllPodDirect] topoLevelNum[%u] is invalid.", topoInfo->topoLevelNums),
                HcclResult::HCCL_E_INTERNAL);

    u32 myRank;
    CHK_RET(HcclGetRankId(comm, &myRank));
#ifdef MACRO_DEV_TYPE_NEW
    CHK_PRT_RET(topoInfo->deviceType != DevType::DEV_TYPE_950,
#else
    CHK_PRT_RET(topoInfo->deviceType != DevType::DEV_TYPE_910_95,
#endif
                HCCL_ERROR("[TopoMatchAlltoAllPodDirect] Rank[%u], deviceType not supported yet.", myRank),
                HcclResult::HCCL_E_PARA);

    algHierarchyInfo.infos.resize(COMM_LAYER_SIZE_2);
    u32 localPodSize = 0;
    CHK_RET(BuildLocalPod(comm, myRank, localPodSize, algHierarchyInfo));

    if (topoInfo->topoLevelNums >= COMM_LAYER_SIZE_2) {
        CHK_RET(BuildDirectInterPeers(comm, myRank, localPodSize, algHierarchyInfo));
    } else {
        algHierarchyInfo.infos[1].push_back({myRank});
    }
    HCCL_INFO("[TopoMatchAlltoAllPodDirect] Rank[%u] matched level0[%u] level1[%zu].",
              myRank, localPodSize, algHierarchyInfo.infos[1][0].size());
#else
    (void)comm;
    (void)topoInfo;
    (void)algHierarchyInfo;
#endif
    return HCCL_SUCCESS;
}

}  // namespace ops_hccl
