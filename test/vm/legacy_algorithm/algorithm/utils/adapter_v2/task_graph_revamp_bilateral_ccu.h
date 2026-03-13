/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2024-2024. All rights reserved.
 * Description: graph revamp for parallel header file
 * Create: 2025-10-17
 */
#ifndef HCCLV1_GRAPH_REVAMP_BILATERAL_CCU_H
#define HCCLV1_GRAPH_REVAMP_BILATERAL_CCU_H

#include <map>
#include <set>
#include <vector>
#include <memory>
#include <queue>

#include "task_def.h"
#include "task_graph_revamp_base.h"

namespace checker {

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
} // namespace checker

#endif