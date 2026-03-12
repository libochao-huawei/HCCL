/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2024-2024. All rights reserved.
 * Description: graph revamp for parallel header file
 * Create: 2025-10-17
 */
#ifndef HCCLV1_GRAPH_REVAMP_PARALLEL_H
#define HCCLV1_GRAPH_REVAMP_PARALLEL_H

#include <map>
#include <set>
#include <vector>
#include <memory>
#include <queue>
#include "task_def.h"
#include "task_graph_revamp_base.h"

namespace checker {
using LoopHeadTailPairs = std::vector<std::pair<TaskNodePtr, TaskNodePtr>>;

struct LoopPostWaitNodes {
    TaskNodePtr localPostHead;
    TaskNodePtr localWaitHead;
    TaskNodePtr localPostTailNode;
    TaskNodePtr localWaitTailNode;
};

class GraphRevampParallel : public GraphRevampBase {
public:
    HcclResult Revamp(TaskNodePtr dummyStart) override;
    HcclResult RevampGraph4Rank(TaskNodePtr ccuHead, RankId rankId) override;

private:
    HcclResult ProcCcuNode4Loop(TaskNodePtr ccuNode, RankId rankId);
    HcclResult ProcCcuNode4Async(TaskNodePtr ccuNode, RankId rankId);
    HcclResult ProcAsyncNode(TaskNodePtr ccuHead, TaskNodePtr currNode, uint32_t virtualQueId);

    HcclResult ProcLoopNode(TaskNodePtr ccuHead, LoopHeadTailPairs &loopNodes, RankId rankId);
    HcclResult PorcessLoopHeadTailNode(const LoopHeadTailPairs &loopNodes, uint32_t loopCnt, TaskNodePtr &beforLoopNode,
        TaskNodePtr &afterLoopNode, RankId rankId);
    HcclResult CreateLoopVirStream(RankId currRank, TaskNodePtr ccuHead, const LoopHeadTailPairs &loopNodes,
        std::vector<LoopPostWaitNodes> &loopHeadTailNodes);
    HcclResult ModifyNodeQueIdx(
        uint32_t virtualQueId, uint32_t relQueId, TaskNodePtr loopStart, LoopPostWaitNodes &loopPostWaitNodes);
    HcclResult PorcessAsyncHeadTailNode(TaskNodePtr currNode, TaskNodePtr &locPostNode,
        std::vector<TaskNodePtr> &beforeAsynNodes, std::vector<TaskNodePtr> &locPostAsynChild);
    HcclResult RestorRealFlowConnection4Async(TaskNodePtr ccuHead, TaskNodePtr locPostNode,
        TaskNodePtr localPostHeadNode, std::vector<TaskNodePtr> &locPostAsynChild);

private:
    std::set<TaskTypeStub> locAsyncNodes_{TaskTypeStub::LOCAL_COPY, TaskTypeStub::LOCAL_REDUCE};
    std::set<TaskTypeStub> rmtAsyncNodes_{TaskTypeStub::READ, TaskTypeStub::READ_REDUCE, TaskTypeStub::WRITE, TaskTypeStub::WRITE_REDUCE};
    std::set<TaskTypeStub> asyncNodes_{TaskTypeStub::LOCAL_COPY,
        TaskTypeStub::LOCAL_REDUCE,
        TaskTypeStub::READ,
        TaskTypeStub::READ_REDUCE,
        TaskTypeStub::WRITE,
        TaskTypeStub::WRITE_REDUCE};
};
} // namespace checker

#endif