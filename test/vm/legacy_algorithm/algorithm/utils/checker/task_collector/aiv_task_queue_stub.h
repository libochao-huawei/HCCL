/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2025-2025. All rights reserved.
 * Description: aiv task queue stub header file
 * Author: luyongshu
 * Create: 2025-05-30
 */
#ifndef HCCLV1_AIV_TASK_QUEUE_STUB_H
#define HCCLV1_AIV_TASK_QUEUE_STUB_H
#include "aiv_task_stub.h"
#include "aiv_base_stub.h"

namespace checker {

struct AivSingleBlockTaskQues {
    //taskQueues[0]:scalar; taskQueues[1]:MTE2; taskQueues[2]:MTE3
    std::vector<std::vector<std::shared_ptr<TaskStub>>> taskQueues{3};

    void AppendAivTask(pipe_t pipeId, std::shared_ptr<TaskStub> task);
    std::shared_ptr<TaskStub> GetTask(pipe_t pipeId, u32 pos) const;
    std::vector<std::shared_ptr<TaskStub>> GetQueTasks(pipe_t pipeId) const;
    u32 GetPipeTaskNum(pipe_t pipeId) const;
};

struct TaskNode;
struct AllAivTaskQueues {
    std::map<RankId, std::map<BlockId, std::vector<AivSingleBlockTaskQues*>>> rsb2AivTaskQueues;
    std::map<TaskStub*, std::vector<TaskNode*>> pipeBarrierAllRecord;
    std::map<RankId, std::vector<TaskNode*>> rank2AivTask;
    std::map<RankId, std::vector<TaskNode*>> copyRank2AivTask;
    std::map<RankId, u32> aivNumforRank;
    std::set<RankId> hasAivTask;
    std::vector<std::shared_ptr<TaskStub>> headAndTailResource;
    void Clear();
    void AppendAivTask(RankId rankId, BlockId blockId, pipe_t pipeId, std::shared_ptr<TaskStub> task);
    AivSingleBlockTaskQues *GetTaskQueueOfAiv(RankId rankId, BlockId blockId, u32 aivTaskIdx) const;
    std::map<BlockId, std::vector<AivSingleBlockTaskQues*>> GetAllAivTaskOfRank(RankId rankId) const;
};

class AivTaskQueueStub {
public:
    static AivTaskQueueStub* Global();
    void Reset();

    static void AppendAivTask(RankId rankId, BlockId blockId, pipe_t pipeId, std::shared_ptr<TaskStub> task);
    AivSingleBlockTaskQues *GetTaskQueueOfAiv(RankId rankId, BlockId blockId) const;
    AllAivTaskQueues& GetAllAivTasks();
    static void PrintAivTask();
    static void SetRank2AivStart(RankId rankId, TaskNode* aivStart);
    static void SetAllCopyAivStart(RankId rankId, TaskNode *aivStart);
    static void AppendAivTaskStubInMainStream(RankId rankId);
    static bool HasAivTask(RankId rankId);
private:
    AllAivTaskQueues allAivTaskQueues;
};

}
#endif