/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef TASK_GRAPH_REVAMP_BASE_H
#define TASK_GRAPH_REVAMP_BASE_H

#include <map>
#include <set>
#include <queue>
#include "sim_task.h"
#include "task_def.h"

namespace HcclSim {

class GraphRevampBase {
public:
    virtual HcclResult Revamp(TaskNodePtr dummyStart) = 0;
    virtual ~GraphRevampBase();
    virtual HcclResult RevampGraph4Rank(TaskNodePtr ccuHead, RankId rankId);

    HcclResult RevampGraph(TaskNodePtr dummyStart);

    HcclResult GetPeerRankByTaskNode(TaskNodePtr currNode, RankId &peerRank);
    HcclResult GetLinkProtoStubByTaskNode(TaskNodePtr currNode, LinkProtoStub &link);
    TaskStub* GenTaskStubBeingReadOrWrittern(TaskNodePtr currNode);
    void RemoveNodeRelation(TaskNodePtr parent, TaskNodePtr child);    
    void AddNodeRelation(TaskNodePtr parent, TaskNodePtr child);
    void SearchGraphByRank(
        TaskNodePtr currNode, std::queue<TaskNodePtr> &graphNodeQue, std::set<TaskNodePtr> &isVisited, RankId rankId);
    void SearchGraphByQueueId(
        TaskNodePtr currNode, std::queue<TaskNodePtr> &graphNodeQue, std::set<TaskNodePtr> &isVisited, uint32_t queIdx);

public:
    uint32_t rankSize_{0};
    static map<RankId, u32> rank2QueSize_;
    std::vector<TaskStub*> toDeleteTaskResource_;
    std::vector<TaskNodePtr> toDeleteTaskNodeResource_;
};
} // namespace HcclSim

#endif