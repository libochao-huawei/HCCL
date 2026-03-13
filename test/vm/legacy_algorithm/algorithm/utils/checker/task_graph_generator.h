/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2024-2024. All rights reserved.
 * Description: TaskGraphGenerator header file
 * Author: shenyutian
 * Create: 2024-07-29
 */

#ifndef HCCLV1_TASK_GRAPH_GENERATOR_H
#define HCCLV1_TASK_GRAPH_GENERATOR_H

#include <algorithm>
#include <vector>
#include <map>
#include <memory>
#include <tuple>

#include "task_stub.h"
#include "task_queue_stub.h"
#include "log.h"
#include "aiv_task_queue_stub.h"
#include "task_def.h"

namespace checker {
using namespace std;

using SeenLocalPost = std::vector<TaskNodePtr>;
using SeenInterRankPosts = std::map<RankId, std::map<RankId, std::vector<TaskNodePtr>>>; // seen Post nodes
using SeenInterRankSendSync = std::map<int32_t*, std::tuple<int32_t, std::vector<TaskNodePtr>, bool>>; // seen SendSync or SendSyncReduce nodes

class TaskGraphGenerator {
public:
    HcclResult GenGraph(const TaskQueueStub &taskQueues, TaskNodePtr dummyStart);

private:
    HcclResult GenGraph4Rank(const SingleRankTaskQueues *rankTaskQueues, const RankId rankIdx,
                             TaskNodePtr dummyStart);
    HcclResult InitRankNodeQue(const SingleRankTaskQueues *rankTaskQueues, const RankId rankIdx, TaskNodePtr dummyStart,
                               std::vector<TaskNodePtr> &rankNodeQue);
    void       LocateUnmatchedNode(const std::vector<TaskNodePtr> &rankNodeQue);
    HcclResult ExecFlitPrim(const SingleRankTaskQueues *rankTaskQueues, TaskNodePtr currNode,
                            std::vector<TaskNodePtr> &rankNodeQue, u64 &unmatchedCnt);
    HcclResult ConnectNextAndPushInQue(const SingleRankTaskQueues *rankTaskQueues, TaskNodePtr currNode,
                                       std::vector<TaskNodePtr> &rankNodeQue);
    HcclResult ExecLocalPostPrim(const SingleRankTaskQueues *rankTaskQueues, TaskNodePtr currNode,
                                 std::vector<TaskNodePtr> &rankNodeQue, SeenLocalPost &seenLocalPosts,
                                 u64 &unmatchedCnt);
    HcclResult ExecLocalWaitPrim(const SingleRankTaskQueues *rankTaskQueues, TaskNodePtr currNode,
                                 std::vector<TaskNodePtr> &rankNodeQue, SeenLocalPost &seenLocalPosts,
                                 u64 &unmatchedCnt);
    bool       IsSemPeer(const TaskNodePtr postNode, const TaskNodePtr waitNode);

    HcclResult GenGraphInterRanks(TaskNodePtr dummyStart);
    HcclResult ExecNode4Graph(TaskNodePtr node, std::vector<TaskNodePtr> &graphNodeQue);
    HcclResult ProcNode4Graph(TaskNodePtr currNode, std::vector<TaskNodePtr> &graphNodeQue,
                              SeenInterRankPosts &seenInterRankPosts, u64 &unmatchedCnt);
    bool       IsExecutable(TaskNodePtr currNode);
    HcclResult ProcInterRankPostNode4Graph(TaskNodePtr currNode, std::vector<TaskNodePtr> &graphNodeQue,
                                           SeenInterRankPosts &seenInterRankPosts, u64 &unmatchedCnt);
    HcclResult ProcInterRankWaitNode4Graph(TaskNodePtr currNode, std::vector<TaskNodePtr> &graphNodeQue,
                                           SeenInterRankPosts &seenInterRankPosts, u64 &unmatchedCnt);
    bool       IsPostWaitPeer(const TaskNodePtr postNode, const TaskNodePtr waitNode);

    //用于处理aiv task相关
    HcclResult ExecAivTaskPrim(const SingleRankTaskQueues *rankTaskQueues, TaskNodePtr currNode,
                               std::vector<TaskNodePtr> &rankNodeQue, u64 &unmatchedCnt);
    HcclResult GenGraph4Aiv(const AivSingleBlockTaskQues* aivTaskQueues,  const RankId rankIdx,
                            const BlockId blockIdx, TaskNodePtr aivStart, TaskNodePtr aivEnd);
    HcclResult ConnectNextAivTaskNodeAndPushInQue(const AivSingleBlockTaskQues* aivTaskQueues, TaskNodePtr currNode,
                                                  std::vector<TaskNodePtr> &aivNodeQue, TaskNodePtr aivEnd);
    HcclResult ExecAivFlitPrim(const AivSingleBlockTaskQues* aivTaskQueues, TaskNodePtr currNode,
                               std::vector<TaskNodePtr> &aivNodeQue, u64 &unmatchedCnt, TaskNodePtr aivEnd);
    HcclResult ExecSetFlagPrim(const AivSingleBlockTaskQues* aivTaskQueues, TaskNodePtr currNode,
                               std::vector<TaskNodePtr> &aivNodeQue, std::vector<TaskNodePtr> &SeenSetFlag,
                               u64 &unmatchedCnt, TaskNodePtr aivEnd);
    HcclResult ExecWaitFlagPrim(const AivSingleBlockTaskQues* aivTaskQueues, TaskNodePtr currNode,
                                std::vector<TaskNodePtr> &aivNodeQue, std::vector<TaskNodePtr> &SeenSetFlag,
                                u64 &unmatchedCnt, TaskNodePtr aivEnd);
    HcclResult ExecPipeBarrierPrim(const AivSingleBlockTaskQues* aivTaskQueues, TaskNodePtr currNode,
                                   std::vector<TaskNodePtr> &aivNodeQue, u64 &unmatchedCnt, TaskNodePtr aivEnd);
    bool IsSetWaitPeer(const TaskNodePtr setFlagNode, const TaskNodePtr waitFlagNode);
    HcclResult GenGraphInterAivs(std::map<RankId, std::vector<TaskNode*>> &rank2AivTask);
    HcclResult ExecAivNode4Graph(std::map<RankId, std::vector<TaskNode*>> &rank2AivTask, std::vector<TaskNodePtr> &graphNodeQue);
    HcclResult ProcAivNode4Graph(TaskNodePtr currNode, std::vector<TaskNodePtr> &graphNodeQue,
                                 SeenInterRankSendSync &seenInterRankSendSync, u64 &unmatchedCnt);
    HcclResult ProcInterAivSendSyncNode4Graph(TaskNodePtr currNode, std::vector<TaskNodePtr> &graphNodeQue,
                                                           SeenInterRankSendSync &seenInterRankSendSync, u64 &unmatchedCnt);
    HcclResult ProcInterAivRecvSyncNode4Graph(TaskNodePtr currNode, std::vector<TaskNodePtr> &graphNodeQue,
                                                           SeenInterRankSendSync &seenInterRankSendSync, u64 &unmatchedCnt);

    // 用于收集分配的node节点并进行析构
    // 如果直接在TaskNode的parent或children节点中放入共享指针，在图规模很大的时候，递归析构可能导致堆栈溢出
    std::vector<std::shared_ptr<TaskNode>> nodes_;
};

} // namespace hccl

#endif