/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "task_graph_generator.h"

namespace HcclSim {

HcclResult TaskGraphGenerator::ExecAivTaskPrim(const SingleRankTaskQueues *rankTaskQueues, TaskNodePtr currNode,
                                               std::vector<TaskNodePtr> &rankNodeQue, u64 &unmatchedCnt)
{
    //init aivStart for rank, record aiv tasknode to map
    auto curRankPos = ((AivTaskStub*)(currNode->task))->GetRankPos();
    std::shared_ptr<TaskStub> aivStartTask = std::make_shared<TaskStubAivStart>(currNode->rankIdx, curRankPos);
    auto aivStart = std::make_shared<TaskNode>(aivStartTask.get(), currNode->rankIdx, curRankPos, -1, -1, -2);
    nodes_.push_back(aivStart);
    ((AivTaskStub *)currNode->task)->SetAivStart(aivStart.get());
    AivTaskQueueStub::Global()->SetRank2AivStart(currNode->rankIdx, aivStart.get());
    AivTaskQueueStub::Global()->GetAllAivTasks().headAndTailResource.push_back(aivStartTask);

    std::shared_ptr<TaskStub> aivEndTask = std::make_shared<TaskStubAivEnd>(currNode->rankIdx, curRankPos);
    auto aivEnd = std::make_shared<TaskNode>(aivEndTask.get(), currNode->rankIdx, curRankPos, -1, -1, -3);
    nodes_.push_back(aivEnd);
    ((AivTaskStub *)currNode->task)->SetAivEnd(aivEnd.get());
    AivTaskQueueStub::Global()->GetAllAivTasks().headAndTailResource.push_back(aivEndTask);
    

    //match the dependency relationships between the same pipe and different pipes in the same AIV
    for (auto& curRankAivTask : AivTaskQueueStub::Global()->GetAllAivTasks().rsb2AivTaskQueues[currNode->rankIdx]) {
        auto curBlock = curRankAivTask.first;
        AivSingleBlockTaskQues* curBlockAivTaskQueuesPtr = curRankAivTask.second[curRankPos];
        std::shared_ptr<TaskStub> blockStartTask = std::make_shared<TaskStubBlockStart>(currNode->rankIdx, curBlock);
        auto blockStart = std::make_shared<TaskNode>(blockStartTask.get(), currNode->rankIdx, curRankPos, curBlock, -1, -1);
        nodes_.push_back(blockStart);
        AivTaskQueueStub::Global()->GetAllAivTasks().headAndTailResource.push_back(blockStartTask);
        aivStart->children.push_back(blockStart.get());
        blockStart->parents.push_back(aivStart.get());
        CHK_PRT_RET(GenGraph4Aiv(curBlockAivTaskQueuesPtr, currNode->rankIdx, curBlock, blockStart.get(), aivEnd.get()) != HcclResult::HCCL_SUCCESS,
                    HCCL_ERROR("[TaskGraphGenerator] Rank [%d], fail to generate dependency aiv graph: TaskType [%s].",
                               currNode->rankIdx, currNode->task->GetType().Describe().c_str()), HcclResult::HCCL_E_INTERNAL);
    }

    //merge PipeBarrierAll Node
    std::map<TaskStub*, std::vector<TaskNode*>> *barrierRecord = &AivTaskQueueStub::Global()->GetAllAivTasks().pipeBarrierAllRecord;
    for (auto& taskNodes : *barrierRecord) {
        TaskNode* firstNode = taskNodes.second[0];
        TaskStubPipeBarrier* firstStub = (TaskStubPipeBarrier*)firstNode->task;
        firstStub->SetPipeToPos((pipe_t)firstNode->pipeIdx, firstNode->pipePos);

        for (int idx = 1; idx < taskNodes.second.size(); idx++) {
            firstStub->SetPipeToPos((pipe_t)taskNodes.second[idx]->pipeIdx, taskNodes.second[idx]->pipePos);

            for (auto& taskNode : taskNodes.second[idx]->parents) {
                firstNode->parents.push_back(taskNode);
                taskNode->children.push_back(firstNode);
                taskNode->children.erase(std::remove(taskNode->children.begin(), taskNode->children.end(), taskNodes.second[idx]), taskNode->children.end());
            }

            for (auto& taskNode : taskNodes.second[idx]->children) {
                firstNode->children.push_back(taskNode);
                taskNode->parents.push_back(firstNode);
                taskNode->parents.erase(std::remove(taskNode->parents.begin(), taskNode->parents.end(), taskNodes.second[idx]), taskNode->parents.end());
            }

        }
    }

    barrierRecord->clear();

    CHK_RET(ExecFlitPrim(rankTaskQueues, currNode, rankNodeQue, unmatchedCnt));
    unmatchedCnt = 0;
    return HcclResult::HCCL_SUCCESS;
}

HcclResult TaskGraphGenerator::GenGraph4Aiv(const AivSingleBlockTaskQues* aivTaskQueues, RankId rankIdx,
                                            BlockId blockIdx, TaskNodePtr blockStart, TaskNodePtr aivEnd)
{
    std::vector<TaskNodePtr> aivNodeQue;
    std::vector<TaskNodePtr> SeenSetFlag;
    u64 unmatchedCnt = 0;

    //init the first tasknode of each pipe in the same block
    auto pipeNum = aivTaskQueues->taskQueues.size();
    for (int currPipe = 0; currPipe < pipeNum; currPipe++) {
        if(aivTaskQueues->GetPipeTaskNum((pipe_t)currPipe) == 0) {
            continue;
        }

        auto currNode = std::make_shared<TaskNode>(aivTaskQueues->GetTask((pipe_t)currPipe, 0).get(), rankIdx, blockStart->rankPos, blockIdx, currPipe, 0);
        CHK_PTR_NULL(currNode);
        nodes_.push_back(currNode);
        blockStart->children.push_back(currNode.get());
        currNode->parents.push_back(blockStart);
        aivNodeQue.push_back(currNode.get());
    }

    while (!aivNodeQue.empty()) {
        if (unmatchedCnt >= aivNodeQue.size()) {
            for (auto& aivNodeUnmatch : aivNodeQue) {
                aivNodeUnmatch->unmatch = true;
            }
            HCCL_ERROR("[TaskGraphGenerator] deadLocking occurs due to mismatch of setFlag and WaitFlag.");
            LocateUnmatchedNode(aivNodeQue);
            return HcclResult::HCCL_E_INTERNAL;
        }

        TaskNodePtr currNode = aivNodeQue[0];
        aivNodeQue.erase(aivNodeQue.begin());

        switch (currNode->task->GetType()) {
            case TaskTypeStub::LOCAL_COPY:
            case TaskTypeStub::LOCAL_REDUCE:
            case TaskTypeStub::READ:
            case TaskTypeStub::READ_REDUCE:
            case TaskTypeStub::WRITE:
            case TaskTypeStub::WRITE_REDUCE:
            case TaskTypeStub::SEND_SYNC:
            case TaskTypeStub::SEND_SYNC_REDUCE:
            case TaskTypeStub::RECV_SYNC:
            case TaskTypeStub::COMP_VALUE:
            case TaskTypeStub::SET_VALUE:
                CHK_RET(ExecAivFlitPrim(aivTaskQueues, currNode, aivNodeQue, unmatchedCnt, aivEnd));
                break;

            case TaskTypeStub::SET_FLAG:
                CHK_RET(ExecSetFlagPrim(aivTaskQueues, currNode, aivNodeQue, SeenSetFlag, unmatchedCnt, aivEnd));
                break;

            case TaskTypeStub::WAIT_FLAG:
                CHK_RET(ExecWaitFlagPrim(aivTaskQueues, currNode, aivNodeQue, SeenSetFlag, unmatchedCnt, aivEnd));
                break;

            case TaskTypeStub::PIPE_BARRIER:
                CHK_RET(ExecPipeBarrierPrim(aivTaskQueues, currNode, aivNodeQue, unmatchedCnt, aivEnd));
                break;

            default:
                HCCL_ERROR("[TaskGraphGenerator] taskType not supported.");
                return HcclResult::HCCL_E_INTERNAL;

        }
    }

    if (!SeenSetFlag.empty()) {
        for (auto& setFlag : SeenSetFlag) {
            bool isGenFromfree = ((TaskStubSetFlag*)(setFlag->task))->IsGenFromFree();
            if (!isGenFromfree) {
                setFlag->unmatch = true;
                HCCL_ERROR("[TaskGraphGenerator] unmatched setFlag: rankId=%d, blockId=%d, pipeId=%s, pipePOs=%d, %s", setFlag->rankIdx, 
                            setFlag->blockIdx, GetPipeName((pipe_t)(setFlag->pipeIdx)).c_str(), setFlag->pipePos, setFlag->task->Describe().c_str());
                return HcclResult::HCCL_E_INTERNAL;
            }
        }
    }
    return HcclResult::HCCL_SUCCESS;
}

HcclResult TaskGraphGenerator::ExecAivFlitPrim(const AivSingleBlockTaskQues* aivTaskQueues, TaskNodePtr currNode,
                                            std::vector<TaskNodePtr> &aivNodeQue, u64 &unmatchedCnt, TaskNodePtr aivEnd)
{
    // curr -> its nxt, push nxt to nodeQue
    CHK_PRT_RET(ConnectNextAivTaskNodeAndPushInQue(aivTaskQueues, currNode, aivNodeQue, aivEnd) != HcclResult::HCCL_SUCCESS,
                HCCL_ERROR("[TaskGraphGenerator] Rank [%d], RankPos [%d], Block [%d], Pipe [%d], fail to generate dependency graph: TaskType [%s].",
                           currNode->rankIdx, currNode->rankPos, currNode->blockIdx, currNode->pipeIdx, currNode->task->GetType().Describe().c_str()),
                HcclResult::HCCL_E_INTERNAL);
    unmatchedCnt = 0;

    return HcclResult::HCCL_SUCCESS;
}

HcclResult TaskGraphGenerator::ConnectNextAivTaskNodeAndPushInQue(const AivSingleBlockTaskQues* aivTaskQueues, TaskNodePtr currNode,
                                                                  std::vector<TaskNodePtr> &aivNodeQue, TaskNodePtr aivEnd)
{
    if (currNode->pipePos < aivTaskQueues->GetPipeTaskNum((pipe_t)(currNode->pipeIdx)) - 1) {
        auto nxtNode = std::make_shared<TaskNode>(aivTaskQueues->GetTask((pipe_t)(currNode->pipeIdx), currNode->pipePos + 1).get(),
                                                  currNode->rankIdx, currNode->rankPos, currNode->blockIdx, currNode->pipeIdx, currNode->pipePos + 1);
        CHK_PTR_NULL(nxtNode);
        nodes_.push_back(nxtNode);
        nxtNode->parents.push_back(currNode);
        currNode->children.push_back(nxtNode.get());
        aivNodeQue.push_back(nxtNode.get());
        return HcclResult::HCCL_SUCCESS;
    }

    if (currNode->pipePos == aivTaskQueues->GetPipeTaskNum((pipe_t)(currNode->pipeIdx)) - 1) {
        currNode->children.push_back(aivEnd);
        aivEnd->parents.push_back(currNode);
        return HcclResult::HCCL_SUCCESS;
    }

    HCCL_DEBUG("[TaskGraphGenerator] Rank [%d], Block [%d], Pipe [%d], end of current Pipe [%d]: TrimType [%s].", currNode->rankIdx,
                   currNode->blockIdx, currNode->pipeIdx, currNode->pipeIdx, currNode->task->GetType().Describe().c_str());
    return HcclResult::HCCL_SUCCESS;
}

HcclResult TaskGraphGenerator::ExecSetFlagPrim(const AivSingleBlockTaskQues* aivTaskQueues, TaskNodePtr currNode,
                                               std::vector<TaskNodePtr> &aivNodeQue, std::vector<TaskNodePtr> &SeenSetFlag,
                                               u64 &unmatchedCnt, TaskNodePtr aivEnd)
{
     CHK_PRT_RET(ConnectNextAivTaskNodeAndPushInQue(aivTaskQueues, currNode, aivNodeQue, aivEnd) != HcclResult::HCCL_SUCCESS,
                HCCL_ERROR("[TaskGraphGenerator] Rank [%d], RankPos [%d], Block [%d], Pipe [%d], fail to generate dependency graph: TaskType [%s].",
                           currNode->rankIdx, currNode->rankPos, currNode->blockIdx, currNode->pipeIdx, currNode->task->GetType().Describe().c_str()),
                HcclResult::HCCL_E_INTERNAL);

    SeenSetFlag.push_back(currNode);
    unmatchedCnt = 0;
    return HcclResult::HCCL_SUCCESS;
}

HcclResult TaskGraphGenerator::ExecWaitFlagPrim(const AivSingleBlockTaskQues* aivTaskQueues, TaskNodePtr currNode,
                                               std::vector<TaskNodePtr> &aivNodeQue, std::vector<TaskNodePtr> &SeenSetFlag,
                                               u64 &unmatchedCnt, TaskNodePtr aivEnd)
{
    std::vector<TaskNodePtr>::iterator setFlagIter;
    for (setFlagIter = SeenSetFlag.begin(); setFlagIter != SeenSetFlag.end(); setFlagIter++) {
        if(IsSetWaitPeer((*setFlagIter), currNode)) {
            (*setFlagIter)->children.push_back(currNode);
            currNode->parents.push_back((*setFlagIter));
            (SeenSetFlag).erase(setFlagIter);
             CHK_PRT_RET(ConnectNextAivTaskNodeAndPushInQue(aivTaskQueues, currNode, aivNodeQue, aivEnd) != HcclResult::HCCL_SUCCESS,
                HCCL_ERROR("[TaskGraphGenerator] Rank [%d], RankPos [%d], Block [%d], Pipe [%d], fail to generate dependency graph: TaskType [%s].",
                           currNode->rankIdx, currNode->rankPos, currNode->blockIdx, currNode->pipeIdx, currNode->task->GetType().Describe().c_str()),
                HcclResult::HCCL_E_INTERNAL);
            unmatchedCnt = 0;
            return HcclResult::HCCL_SUCCESS;
        }
    }

    aivNodeQue.push_back(currNode);
    unmatchedCnt++;

    return HcclResult::HCCL_SUCCESS;
}

HcclResult TaskGraphGenerator::ExecPipeBarrierPrim(const AivSingleBlockTaskQues* aivTaskQueues, TaskNodePtr currNode,
                                                   std::vector<TaskNodePtr> &aivNodeQue, u64 &unmatchedCnt, TaskNodePtr aivEnd)
{
    TaskStubPipeBarrier* currBarrierStub = (TaskStubPipeBarrier*)(currNode->task);
    if (currBarrierStub->IsPipeBarrierAll()) {
        map<TaskStub*, std::vector<TaskNode*>> *pipeRecord = &(AivTaskQueueStub::Global()->GetAllAivTasks().pipeBarrierAllRecord);
        (*pipeRecord)[currNode->task].push_back(currNode);
    }

     CHK_PRT_RET(ConnectNextAivTaskNodeAndPushInQue(aivTaskQueues, currNode, aivNodeQue, aivEnd) != HcclResult::HCCL_SUCCESS,
                HCCL_ERROR("[TaskGraphGenerator] Rank [%d], RankPos [%d], Block [%d], Pipe [%d], fail to generate dependency graph: TaskType [%s].",
                           currNode->rankIdx, currNode->rankPos, currNode->blockIdx, currNode->pipeIdx, currNode->task->GetType().Describe().c_str()),
                HcclResult::HCCL_E_INTERNAL);

    unmatchedCnt = 0;
    return HcclResult::HCCL_SUCCESS;
}

bool TaskGraphGenerator::IsSetWaitPeer(const TaskNodePtr setFlagNode, const TaskNodePtr waitFlagNode)
{
    if ((setFlagNode->task->GetType() != TaskTypeStub::SET_FLAG)
        || (waitFlagNode->task->GetType() != TaskTypeStub::WAIT_FLAG)) {
        return false;
    }

    TaskStubSetFlag *setFlag = dynamic_cast<TaskStubSetFlag*>(setFlagNode->task);
    TaskStubWaitFlag *waitFlag = dynamic_cast<TaskStubWaitFlag*>(waitFlagNode->task);

    if (setFlag->GetBlockId() != waitFlag->GetBlockId()) {
        return false;
    }

    if (setFlag->GetSrcPipe() != waitFlag->GetSrcPipe()) {
        return false;
    }

    if (setFlag->GetDstPipe() != waitFlag->GetDstPipe()) {
        return false;
    }

    if (setFlag->IsGenFromFree() != waitFlag->IsGenFromAlloc()) {
        return false;
    }

    return (setFlag->GetEventId() == waitFlag->GetEventId());
}

HcclResult TaskGraphGenerator::GenGraphInterAivs(std::map<RankId, std::vector<TaskNode*>> &rank2AivTask)
{
    std::vector<TaskNodePtr> graphNodeQue;
    SeenInterRankSendSync seenInterAivSendSync;
    u64 unmatchedCnt = 0;

    CHK_PRT_RET(ExecAivNode4Graph(rank2AivTask, graphNodeQue) != HcclResult::HCCL_SUCCESS,
                HCCL_ERROR("[TaskGraphGenerator] Fail to init graphAivNodeQue."), HcclResult::HCCL_E_INTERNAL);

    while (!graphNodeQue.empty()) {
        if (unmatchedCnt >= graphNodeQue.size()) {
            for (auto& aivNodeUnmatch : graphNodeQue) {
                aivNodeUnmatch->unmatch = true;
            }
            HCCL_ERROR("[TaskGraphGenerator] deadLocking occurs due to mismatch of SendSync/Reduce and RecvSync.");
            LocateUnmatchedNode(graphNodeQue);
            return HcclResult::HCCL_E_INTERNAL;
        }

        TaskNodePtr currNode = graphNodeQue[0];
        graphNodeQue.erase(graphNodeQue.begin());

        CHK_PRT_RET(ProcAivNode4Graph(currNode, graphNodeQue, seenInterAivSendSync, unmatchedCnt)
                        != HcclResult::HCCL_SUCCESS,
                    HCCL_ERROR("[TaskGraphGenerator] Rank [%d], fail to proceed taskNode.", currNode->rankIdx),
                    HcclResult::HCCL_E_INTERNAL);
    }

    if (!seenInterAivSendSync.empty()) {
        for (auto& curGmAddrRecord : seenInterAivSendSync) {
            if (!std::get<2>(curGmAddrRecord.second)) {
                for (auto& sendSyncNode : std::get<1>(curGmAddrRecord.second)) {
                    sendSyncNode->unmatch = true;
                    HCCL_WARNING("[TaskGraphGenerator] unmatched inter-aiv sendSyncNode: GmAddr [%d],  ",
                               curGmAddrRecord.first);
                }
            }
        }
    }

    return HcclResult::HCCL_SUCCESS;
}

HcclResult TaskGraphGenerator::ExecAivNode4Graph(std::map<RankId, std::vector<TaskNode*>> &rank2AivTask, std::vector<TaskNodePtr> &graphNodeQue)
{
    for (auto& aivPair : rank2AivTask) {
        for (auto& aivStart : aivPair.second) {
            aivStart->execFlag = true;
            for (auto& blockStart : aivStart->children) {
                CHK_PRT_RET(ExecNode4Graph(blockStart, graphNodeQue) != HcclResult::HCCL_SUCCESS,
                            HCCL_ERROR("[TaskGraphGenerator] Fail to init graphAivNodeQue."), HcclResult::HCCL_E_INTERNAL);
            }
        }
    }
    return HcclResult::HCCL_SUCCESS; 
}

HcclResult TaskGraphGenerator::ProcAivNode4Graph(TaskNodePtr currNode, std::vector<TaskNodePtr> &graphNodeQue,
                                                 SeenInterRankSendSync &seenInterRankSendSync,
                                                 u64 &unmatchedCnt)
{
    if (!IsExecutable(currNode)) {
        graphNodeQue.push_back(currNode);
        unmatchedCnt++;
        HCCL_DEBUG("[TaskGraphGenerator] taskNode not executable, push back to the queue.");
        return HcclResult::HCCL_SUCCESS;
    }
    if(currNode->task == nullptr){
        currNode->execFlag = true;
        HCCL_DEBUG("[TaskGraphGenerator] aiv end node.");
        return HcclResult::HCCL_SUCCESS;
    }
    switch (currNode->task->GetType()) {
        case TaskTypeStub::LOCAL_COPY:
        case TaskTypeStub::LOCAL_REDUCE:
        case TaskTypeStub::READ:
        case TaskTypeStub::READ_REDUCE:
        case TaskTypeStub::WRITE:
        case TaskTypeStub::WRITE_REDUCE:
        case TaskTypeStub::COMP_VALUE:
        case TaskTypeStub::SET_VALUE:
        case TaskTypeStub::SET_FLAG:
        case TaskTypeStub::WAIT_FLAG:
        case TaskTypeStub::PIPE_BARRIER:
        case TaskTypeStub::AIV_END:
            CHK_PRT_RET(ExecNode4Graph(currNode, graphNodeQue) != HcclResult::HCCL_SUCCESS,
                        HCCL_ERROR("[TaskGraphGenerator] Rank [%d], fail to execute taskNode.", currNode->rankIdx),
                        HcclResult::HCCL_E_INTERNAL);
            unmatchedCnt = 0;
            return HcclResult::HCCL_SUCCESS;

        case TaskTypeStub::SEND_SYNC:
        case TaskTypeStub::SEND_SYNC_REDUCE:
            CHK_RET(ProcInterAivSendSyncNode4Graph(currNode, graphNodeQue, seenInterRankSendSync, unmatchedCnt));
            return HcclResult::HCCL_SUCCESS;

        case TaskTypeStub::RECV_SYNC:
            CHK_RET(ProcInterAivRecvSyncNode4Graph(currNode, graphNodeQue, seenInterRankSendSync, unmatchedCnt));
            return HcclResult::HCCL_SUCCESS;

        default:
            HCCL_ERROR("[TaskGraphGenerator] taskType not supported.");
            return HcclResult::HCCL_E_INTERNAL;
    }

    return HcclResult::HCCL_SUCCESS;
}

HcclResult TaskGraphGenerator::ProcInterAivSendSyncNode4Graph(TaskNodePtr currNode, std::vector<TaskNodePtr> &graphNodeQue,
                                                           SeenInterRankSendSync &seenInterRankSendSync, u64 &unmatchedCnt)
{
    int32_t* flagAddr;
    int32_t  flagValue;
    switch (currNode->task->GetType()) {
        case TaskTypeStub::SEND_SYNC:
            flagAddr = ((TaskStubSendSync*)(currNode->task))->GetFlagAddr();
            flagValue = ((TaskStubSendSync*)(currNode->task))->GetFlagValue();
            break;
        case TaskTypeStub::SEND_SYNC_REDUCE:
            flagAddr = ((TaskStubSendSyncReduce*)(currNode->task))->GetFlagAddr();
            flagValue = ((TaskStubSendSyncReduce*)(currNode->task))->GetFlagValue();
            break;
        default:
            HCCL_ERROR("Node is not expected tpye, SendSync or SendSyncReduce.");
            return HcclResult::HCCL_E_INTERNAL;
    }
    if (seenInterRankSendSync.find(flagAddr) == seenInterRankSendSync.end()) {
        std::pair<int32_t*, std::tuple<int32_t, std::vector<TaskNodePtr>, bool>> sendAndRecvFlag
            = std::make_pair(flagAddr, std::make_tuple(flagValue, std::vector<TaskNodePtr>(), false));
        seenInterRankSendSync.insert(sendAndRecvFlag);
    } else {
        if (std::get<2>(seenInterRankSendSync[flagAddr])) {
            std::get<0>(seenInterRankSendSync[flagAddr]) = flagValue;
            std::get<1>(seenInterRankSendSync[flagAddr]).clear();
            std::get<2>(seenInterRankSendSync[flagAddr]) = false;
        } else {
            if (currNode->task->GetType() == TaskTypeStub::SEND_SYNC) {
                std::get<0>(seenInterRankSendSync[flagAddr]) = flagValue;
            } else if (currNode->task->GetType() == TaskTypeStub::SEND_SYNC_REDUCE) {
                std::get<0>(seenInterRankSendSync[flagAddr]) += flagValue;
            }
        }
    }
    std::get<1>(seenInterRankSendSync[flagAddr]).push_back(currNode);

    CHK_PRT_RET(ExecNode4Graph(currNode, graphNodeQue) != HcclResult::HCCL_SUCCESS,
                        HCCL_ERROR("[TaskGraphGenerator] Fail to init graphAivNodeQue."), HcclResult::HCCL_E_INTERNAL);

    unmatchedCnt = 0;
    return HcclResult::HCCL_SUCCESS;
}

HcclResult TaskGraphGenerator::ProcInterAivRecvSyncNode4Graph(TaskNodePtr currNode, std::vector<TaskNodePtr> &graphNodeQue,
                                                              SeenInterRankSendSync &seenInterRankSendSync, u64 &unmatchedCnt)
{
    int32_t* flagAddr = ((TaskStubRecvSync*)(currNode->task))->GetFlagAddr();
    int32_t  flagValue = ((TaskStubRecvSync*)(currNode->task))->GetFlagValue();

    if (seenInterRankSendSync.find(flagAddr) == seenInterRankSendSync.end()) {
        graphNodeQue.push_back(currNode);
        unmatchedCnt++;
        return HcclResult::HCCL_SUCCESS;
    } else {
        auto sendSyncFlagValue = std::get<0>(seenInterRankSendSync[flagAddr]);
        if (sendSyncFlagValue == flagValue) {
            for (auto& sendNodePtr : std::get<1>(seenInterRankSendSync[flagAddr])) {
                sendNodePtr->children.push_back(currNode);
                currNode->parents.push_back(sendNodePtr);
            }
            CHK_PRT_RET(ExecNode4Graph(currNode, graphNodeQue) != HcclResult::HCCL_SUCCESS,
                        HCCL_ERROR("[TaskGraphGenerator] Fail to init graphAivNodeQue."), HcclResult::HCCL_E_INTERNAL);
            std::get<2>(seenInterRankSendSync[flagAddr]) = true;
            unmatchedCnt = 0;
            return HcclResult::HCCL_SUCCESS;
        }    
    }

    graphNodeQue.push_back(currNode);
    unmatchedCnt++;
    return HcclResult::HCCL_SUCCESS;
}

} // namespace hccl
