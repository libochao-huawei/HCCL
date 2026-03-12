/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef TASK_GRAPH_REVAMP_BILATERAL_CCU_H
#define TASK_GRAPH_REVAMP_BILATERAL_CCU_H

#include <map>
#include <set>
#include <vector>
#include <memory>
#include <queue>

#include "task_def.h"
#include "task_graph_revamp_base.h"

namespace HcclSim {

class GraphRevampBilateralCcu : public GraphRevampBase {
public:
    HcclResult Revamp(TaskNodePtr dummyStart) override;
    HcclResult RevampGraph4Rank(TaskNodePtr ccuHead, RankId rankId) override;

private:
    HcclResult ProcCcuRWNode(TaskNodePtr ccuNode, TaskNodePtr asyncNode, TaskNodePtr waitNode, TaskNodePtr postNode);
    HcclResult ProcAsyncCcuRWNode(
        TaskNodePtr ccuNode, TaskNodePtr asyncNode, TaskNodePtr waitNode, TaskNodePtr postNode);
    HcclResult SearchBackwardCcuRW(
        TaskNodePtr ccuNode, TaskNodePtr asyncNode, TaskNodePtr waitNode, TaskNodePtr &beingRWNode);
    HcclResult SearchForwardCcuRW(
        TaskNodePtr ccuNode, TaskNodePtr asyncNode, TaskNodePtr postNode, TaskNodePtr &beingRWNode);

    HcclResult AddBeingRWNode2CcuHead(
        TaskNodePtr ccuNode, TaskNodePtr currNode, TaskNodePtr &beingRWNode, RankId peerRank);
    HcclResult AddVirtualFlowFirstHalfPart(
        TaskNodePtr headNode, TaskNodePtr headChild, TaskNodePtr currNode, TaskNodePtr &beingRWNode, TaskStub *beingRW);

    HcclResult CreateNewVirtualStream(
        TaskNodePtr waitNode, TaskNodePtr currNode, TaskNodePtr &beingRWNode, RankId peerRank);
    HcclResult CloseLoopNewVirtualStream(
        TaskNodePtr waitNode, TaskNodePtr currNode, TaskNodePtr beingRWNode, RankId peerRank);
};
} // namespace HcclSim

#endif