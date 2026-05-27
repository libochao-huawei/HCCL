/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "topo_match_ubx_v2.h"

namespace ops_hccl {

TopoMatchUBX_V2::TopoMatchUBX_V2()
    : TopoMatchUBX()
{
}

TopoMatchUBX_V2::~TopoMatchUBX_V2()
{
}

HcclResult TopoMatchUBX_V2::MatchTopo(const HcclComm comm, TopoInfoWithNetLayerDetails *topoInfo,
                                       AlgHierarchyInfoForAllLevel &algHierarchyInfo)
{
    CHK_RET(TopoMatchUBX::MatchTopo(comm, topoInfo, algHierarchyInfo));

    u32 layer0Size = 0;
    myRank_ = topoInfo->userRank;

    if (!algHierarchyInfo.infos.empty() && !algHierarchyInfo.infos[0].empty()) {
        layer0Size = static_cast<u32>(algHierarchyInfo.infos[0][0].size());
    }

    CHK_RET(BuildTwoStageTopo(comm, layer0Size, myRank_, algHierarchyInfo));

    return HCCL_SUCCESS;
}

HcclResult TopoMatchUBX_V2::BuildTwoStageTopo(const HcclComm comm, u32 layer0Size, u32 myRank,
                                               const AlgHierarchyInfoForAllLevel &algHierarchyInfo)
{
    if (algHierarchyInfo.infos.empty() || algHierarchyInfo.infos[0].empty()) {
        HCCL_DEBUG("[TopoMatchUBX_V2] empty hierarchy info, skip two-stage build");
        return HCCL_SUCCESS;
    }

    const auto &infosLevel0 = algHierarchyInfo.infos[0];
    twoStageInfo_.stage1IntraRanks.clear();
    twoStageInfo_.stage1InterRanks.clear();
    twoStageInfo_.stage2IntraRanks.clear();
    twoStageInfo_.stage2InterRanks.clear();
    twoStageInfo_.borrowRank = INVALID_VALUE_RANKID;
    twoStageInfo_.borrowLinkIdx = 0;
    twoStageInfo_.borrowChannel = ChannelInfo();

    if (infosLevel0.size() < 2) {
        HCCL_DEBUG("[TopoMatchUBX_V2] insufficient levels for two-stage, size=%zu", infosLevel0.size());
        return HCCL_SUCCESS;
    }

    twoStageInfo_.stage1IntraRanks.push_back(infosLevel0[0]);

    std::vector<u32> closRanks;
    u32 meshSize = infosLevel0[0].size();
    for (auto rank : infosLevel0[1]) {
        if (rank % meshSize == myRank % meshSize) {
            closRanks.push_back(rank);
        }
    }
    twoStageInfo_.stage1InterRanks.push_back(closRanks);

    if (closRanks.size() <= 1) {
        HCCL_DEBUG("[TopoMatchUBX_V2] closRanks size=%zu, no borrow possible", closRanks.size());
        return HCCL_SUCCESS;
    }

    myRank_ = myRank;

    u32 borrowRank = INVALID_VALUE_RANKID;
    for (auto rank : closRanks) {
        if (rank / meshSize != myRank / meshSize) {
            borrowRank = rank;
            break;
        }
    }

    if (borrowRank == INVALID_VALUE_RANKID) {
        HCCL_DEBUG("[TopoMatchUBX_V2] no suitable borrow rank found");
        return HCCL_SUCCESS;
    }

    twoStageInfo_.borrowRank = borrowRank;
    twoStageInfo_.borrowLinkIdx = 0;

#ifndef AICPU_COMPILE
    CommLink *links = nullptr;
    uint32_t linkNum = 0;
    HcclResult linkRet = HcclRankGraphGetLinks(comm, 1, myRank, borrowRank, &links, &linkNum);
    if (linkRet == HCCL_SUCCESS && links != nullptr && linkNum > 0) {
        twoStageInfo_.borrowChannel.isValid = true;
        twoStageInfo_.borrowChannel.remoteRank = borrowRank;
    }
#endif

    twoStageInfo_.stage2IntraRanks = twoStageInfo_.stage1IntraRanks;
    if (!twoStageInfo_.stage2IntraRanks.empty()) {
        bool alreadyInIntra = false;
        for (auto rank : twoStageInfo_.stage2IntraRanks[0]) {
            if (rank == borrowRank) {
                alreadyInIntra = true;
                break;
            }
        }
        if (!alreadyInIntra) {
            twoStageInfo_.stage2IntraRanks[0].push_back(borrowRank);
        }
    }

    twoStageInfo_.stage2InterRanks = twoStageInfo_.stage1InterRanks;
    for (auto &group : twoStageInfo_.stage2InterRanks) {
        group.erase(std::remove(group.begin(), group.end(), borrowRank), group.end());
    }

    HCCL_INFO("[TopoMatchUBX_V2] Two-stage built: borrowRank=%u, "
              "stage2Intra[0].size=%zu, stage2Inter[0].size=%zu",
              borrowRank,
              twoStageInfo_.stage2IntraRanks.empty() ? 0 : twoStageInfo_.stage2IntraRanks[0].size(),
              twoStageInfo_.stage2InterRanks.empty() ? 0 : twoStageInfo_.stage2InterRanks[0].size());

    return HCCL_SUCCESS;
}

}  // namespace ops_hccl
