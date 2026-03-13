/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2024-2024. All rights reserved.
 * Description: graph revamp base header file
 * Create: 2025-10-17
 */
#ifndef HCCLV1_GRAPH_REVAMP_BASE_H
#define HCCLV1_GRAPH_REVAMP_BASE_H

#include <map>
#include <queue>
#include "task_stub.h"
#include "task_def.h"

namespace checker {

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
} // namespace checker

#endif