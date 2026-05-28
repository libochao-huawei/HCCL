/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef TOPO_MATCH_UBX_V2
#define TOPO_MATCH_UBX_V2

#include "topo_match_ubx.h"
#include "alg_param.h"

namespace ops_hccl {

struct TwoStageTopoInfo {
    // Stage 1: standard UBX decomposition (unchanged)
    std::vector<std::vector<u32>> stage1IntraRanks;  // FM: same-pod ranks
    std::vector<std::vector<u32>> stage1InterRanks;  // Clos: cross-pod ranks

    // Stage 2: asymmetric topology
    std::vector<std::vector<u32>> stage2IntraRanks;  // FM + borrow_rank
    std::vector<std::vector<u32>> stage2InterRanks;  // Clos - borrow_rank

    u32 borrowRank = INVALID_VALUE_RANKID;  // the Clos rank whose link 0 is borrowed
    u32 borrowLinkIdx = 0;                  // always 0 (link 0)

    // Channel info for the borrowed link
    ChannelInfo borrowChannel;
};

class TopoMatchUBX_V2 : public TopoMatchUBX {
public:
    explicit TopoMatchUBX_V2();
    ~TopoMatchUBX_V2() override;

    std::string Describe() const override
    {
        return "Topo Match UBX V2: 2-stage topology with asymmetric borrow for AllGather.";
    }

    // Override MatchTopo to also compute two-stage info
    HcclResult MatchTopo(const HcclComm comm, TopoInfoWithNetLayerDetails *topoInfo,
                         AlgHierarchyInfoForAllLevel &algHierarchyInfo) override;

    // Get the two-stage topology result
    const TwoStageTopoInfo &GetTwoStageInfo() const { return twoStageInfo_; }

protected:
    // Build Stage 2 topology from Stage 1 result + borrow rank
    HcclResult BuildTwoStageTopo(const HcclComm comm, u32 layer0Size, u32 myRank,
                                 const AlgHierarchyInfoForAllLevel &algHierarchyInfo);

private:
    TwoStageTopoInfo twoStageInfo_;
};

}  // namespace ops_hccl

#endif  // TOPO_MATCH_UBX_V2
