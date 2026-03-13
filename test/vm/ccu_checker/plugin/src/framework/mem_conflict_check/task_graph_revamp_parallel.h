/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef TASK_GRAPH_REVAMP_PARALLEL_H
#define TASK_GRAPH_REVAMP_PARALLEL_H

#include <map>
#include <set>
#include <vector>
#include <memory>
#include <queue>
#include "task_def.h"
#include "task_graph_revamp_base.h"

namespace HcclSim {
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
    HcclResult ProcAsyncNode(TaskNodePtr currNode, uint32_t virtualQueId);

    HcclResult ProcLoopNode(TaskNodePtr ccuHead, LoopHeadTailPairs &loopNodes, RankId rankId);
    HcclResult PorcessLoopHeadTailNode(const LoopHeadTailPairs &loopNodes, uint32_t loopCnt, TaskNodePtr &beforLoopNode,
        TaskNodePtr &afterLoopNode, RankId rankId);
    HcclResult CreateLoopVirStream(RankId currRank, TaskNodePtr ccuHead, const LoopHeadTailPairs &loopNodes,
        std::vector<LoopPostWaitNodes> &loopHeadTailNodes);
    HcclResult ModifyNodeQueIdx(
        uint32_t virtualQueId, uint32_t relQueId, TaskNodePtr loopStart, LoopPostWaitNodes &loopPostWaitNodes);
    HcclResult PorcessAsyncHeadTailNode(TaskNodePtr currNode, TaskNodePtr &locPostNode,
        std::vector<TaskNodePtr> &beforeAsynNodes, std::vector<TaskNodePtr> &locPostAsynChild);
    HcclResult RestorRealFlowConnection4Async(
        TaskNodePtr locPostNode, TaskNodePtr localPostHeadNode, std::vector<TaskNodePtr> &locPostAsynChild);

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
} // namespace HcclSim

#endif