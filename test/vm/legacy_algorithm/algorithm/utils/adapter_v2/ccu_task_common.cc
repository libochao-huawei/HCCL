/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2025-2025. All rights reserved.
 * Description: ccu instruction transform to checker task
 * Author: huangweihao
 * Create: 2025-06-19
 */

#include <stack>
#include "ccu_task_common.h"
#include "transport_utils.h"
#include "type_conversion.h"
#include "mem_layout.h"
#include "task_graph_generator.h"
#include "ccu_all_rank_param_recorder.h"
#include "task_stub.h"

using namespace checker;

namespace {
    std::set<TaskTypeStub> g_locAsyncNodes{TaskTypeStub::LOCAL_COPY, TaskTypeStub::LOCAL_REDUCE, TaskTypeStub::LOCAL_BATCH_REDUCE};
    static std::set<TaskTypeStub> g_rmtAsyncNodes{TaskTypeStub::READ, TaskTypeStub::READ_REDUCE, TaskTypeStub::WRITE, TaskTypeStub::WRITE_REDUCE};
}

namespace Hccl {
void PrintGraphOneNode(TaskNodePtr curNode)
{
    printf("\n");
    if (curNode->task != nullptr) {
        if ((curNode->task->GetType() == TaskTypeStub::CCU_GRAPH || curNode->task->GetType() == TaskTypeStub::SUB_GRAPH_END)) {
            std::cout<<"currNode is "<<curNode->task->GetType()<<std::endl;
        } else {
            printf("curNode[%llx] is %s, queueId is %d, parent size=%d, child size=%d\n",
                curNode,
                curNode->task->Describe().c_str(),
                curNode->queIdx,
                curNode->parents.size(),
                curNode->children.size());
        }
    }

    printf("-----------------------\n");
    for (int i = 0; i < curNode->parents.size(); i++) {
        if (curNode->parents[i]->rankIdx != curNode->rankIdx) {
            continue;
        }
        if (curNode->parents[i]->task->GetType() == TaskTypeStub::CCU_GRAPH || curNode->parents[i]->task->GetType() == TaskTypeStub::SUB_GRAPH_END) {
            std::cout<<"currNode is "<<curNode->parents[i]->task->GetType()<<std::endl;
            continue;
        }
        if (curNode->parents[i]->task != nullptr) {
            printf("parents[%d][%llx] is %s, queueId is %d\n",
                i,
                curNode->parents[i],
                curNode->parents[i]->task->Describe().c_str(),
                curNode->parents[i]->queIdx);
        }
    }
    printf("-----------------------\n");
    for (int i = 0; i < curNode->children.size(); i++) {
        if (curNode->children[i]->rankIdx != curNode->rankIdx) {
            continue;
        }
        if (curNode->children[i]->task->GetType() == TaskTypeStub::CCU_GRAPH || curNode->children[i]->task->GetType() == TaskTypeStub::SUB_GRAPH_END) {
            std::cout<<"currNode is "<<curNode->children[i]->task->GetType()<<std::endl;
            continue;
        }
        printf("children[%d][%llx] is %s, queueId is %d\n",
            i,
            curNode->children[i],
            curNode->children[i]->task->Describe().c_str(),
            curNode->children[i]->queIdx);
    }
    printf("-----------------------\n");
    printf("\n");
}

void PrintNodeByTypes(TaskNodePtr curNode, std::set<TaskTypeStub> types)
{
    if (curNode->task != nullptr) {
        if (types.count(curNode->task->GetType()) != 0) {
            printf("-----------------------\n");
            std::cout<<"currNode is: "<<curNode<<": "<<curNode->task->Describe()<<std::endl;
            printf("-----------------------\n");
        }
    }
}


void PrintGraphRevamp(TaskNodePtr head)
{
    std::vector<TaskNodePtr> candTaskNodePtr;
    std::set<TaskNodePtr> printedNode;
    std::set<TaskNodePtr> isVisited;
    PrintGraphOneNode(head);
    for(int i = 0; i < head->children.size(); i++) {
        printedNode.insert(head->children[i]);
        candTaskNodePtr.push_back(head->children[i]);
    }

    while(!candTaskNodePtr.empty()) {
        TaskNodePtr curNode = candTaskNodePtr[0];
        candTaskNodePtr.erase(candTaskNodePtr.begin());

        if (isVisited.count(curNode) != 0) {
            continue;
        }
        isVisited.insert(curNode);
        // print node info
        PrintGraphOneNode(curNode);
        for (int i = 0; i < curNode->children.size(); i++) {
            if (curNode->children[i]->rankIdx != curNode->rankIdx) {
                continue;
            }
            std::set<TaskNodePtr> ::iterator it = printedNode.find(curNode->children[i]);
            if (it == printedNode.end()) {
                candTaskNodePtr.push_back(curNode->children[i]);
                printedNode.insert(curNode->children[i]);
            }
        }
    }
    return;
}

void PrintGraphRevampByTypes(TaskNodePtr head, std::set<TaskTypeStub> types)
{
    std::vector<TaskNodePtr> candTaskNodePtr;
    std::set<TaskNodePtr> printedNode;
    std::set<TaskNodePtr> isVisited;
    for(int i = 0; i < head->children.size(); i++) {
        printedNode.insert(head->children[i]);
        candTaskNodePtr.push_back(head->children[i]);
    }

    while(!candTaskNodePtr.empty()) {
        TaskNodePtr curNode = candTaskNodePtr[0];
        candTaskNodePtr.erase(candTaskNodePtr.begin());

        if (isVisited.count(curNode) != 0 || curNode->task->GetType() == TaskTypeStub::POST) {
            continue;
        }
        isVisited.insert(curNode);
        // print node info
        PrintNodeByTypes(curNode, types);
        for (int i = 0; i < curNode->children.size(); i++) {
            if (curNode->children[i]->rankIdx != curNode->rankIdx) {
                continue;
            }
            std::set<TaskNodePtr> ::iterator it = printedNode.find(curNode->children[i]);
            if (it == printedNode.end()) {
                candTaskNodePtr.push_back(curNode->children[i]);
                printedNode.insert(curNode->children[i]);
            }
        }
    }
    return;
}

void PrintGraphRevampByQueue(TaskNodePtr head, std::set<uint32_t> queList)
{
    std::vector<TaskNodePtr> candTaskNodePtr;
    std::set<TaskNodePtr> printedNode;
    std::set<TaskNodePtr> isVisited;
    for(int i = 0; i < head->children.size(); i++) {
        printedNode.insert(head->children[i]);
        candTaskNodePtr.push_back(head->children[i]);
    }

    while(!candTaskNodePtr.empty()) {
        TaskNodePtr curNode = candTaskNodePtr[0];
        candTaskNodePtr.erase(candTaskNodePtr.begin());

        if (isVisited.count(curNode) != 0 || queList.count(curNode->queIdx) == 0) {
            continue;
        }
        isVisited.insert(curNode);
        // print node info
        PrintGraphOneNode(curNode);
        for (int i = 0; i < curNode->children.size(); i++) {
            if (curNode->children[i]->rankIdx != curNode->rankIdx) {
                continue;
            }
            std::set<TaskNodePtr> ::iterator it = printedNode.find(curNode->children[i]);
            if (it == printedNode.end()) {
                candTaskNodePtr.push_back(curNode->children[i]);
                printedNode.insert(curNode->children[i]);
            }
        }
    }
    return;
}

HcclResult CollectBilateralWaitInfo(TaskStubCcuGraph *curCcuTask, uint32_t queId, TaskNodePtr node)
{
    if (node->task->GetType() == TaskTypeStub::WAIT) {
        TaskStubWait *wait = dynamic_cast<TaskStubWait *>(node->task);
        auto peerRank = wait->GetRemoteRank();
        curCcuTask->waitInfoTmp_[peerRank].waitNodes.push_back(node);
    } else {
        RankId peerRank;
        CHK_RET(GetPeerRankByTaskNode(node, peerRank));
        auto waitSize = curCcuTask->waitInfoTmp_[peerRank].waitNodes.size();
        TaskNodePtr waitNode = nullptr;
        if (waitSize > 0) {
            // 取最近的一个wait节点
            waitNode = curCcuTask->waitInfoTmp_[peerRank].waitNodes[waitSize - 1];
        }
        curCcuTask->bilateralPart1_[queId].insert(std::make_pair(node, waitNode));
    }
    return HCCL_SUCCESS;
}

HcclResult CollectBilateralPostInfo(TaskStubCcuGraph *curCcuTask, uint32_t queId, TaskNodePtr node, bool isLast)
{
    if (isLast) {
        for (uint32_t rankId = 0; rankId < curCcuTask->postInfoTmp_.size(); rankId++) {
            if (rankId == curCcuTask->rankId) {
                continue;
            }
            for (auto &asynNode : curCcuTask->postInfoTmp_[rankId].asyncNodes) {
                curCcuTask->bilateralPart2_[queId].insert(std::make_pair(asynNode, nullptr));
            }
            curCcuTask->postInfoTmp_[rankId].asyncNodes.clear();
        }
        return HCCL_SUCCESS;
    }

    if (node->task->GetType() == TaskTypeStub::POST) {
        TaskStubPost *post = dynamic_cast<TaskStubPost *>(node->task);
        auto peerRank = post->GetRemoteRank();
        auto asynNodeSize = curCcuTask->postInfoTmp_[peerRank].asyncNodes.size();
        if (asynNodeSize > 0) {
            for (auto &asynNode : curCcuTask->postInfoTmp_[peerRank].asyncNodes) {
                curCcuTask->bilateralPart2_[queId].insert(std::make_pair(asynNode, node));
            }
            curCcuTask->postInfoTmp_[peerRank].asyncNodes.clear();
        }
    } else {
        RankId peerRank;
        CHK_RET(GetPeerRankByTaskNode(node, peerRank));
        curCcuTask->postInfoTmp_[peerRank].asyncNodes.push_back(node);
    }
    return HCCL_SUCCESS;
}

void AddNodeRelation(TaskNodePtr parent, TaskNodePtr child)
{
    auto childIter = std::find(parent->children.begin(), parent->children.end(), child);
    if (childIter == parent->children.end()) {
        parent->children.push_back(child);
    }

    auto parentIter = std::find(child->parents.begin(), child->parents.end(), parent);
    if (parentIter == child->parents.end()) {
        child->parents.push_back(parent);
    }
}

HcclResult AppendTailNode(TaskStubCcuGraph *curCcuTask, uint32_t queId, TaskNodePtr node)
{
    TaskNodePtr preNode = (curCcuTask->tailNodes[queId] == nullptr) ? \
        curCcuTask->ccuHeadTaskNode : curCcuTask->tailNodes[queId];
    AddNodeRelation(preNode, node);
    curCcuTask->tailNodes[queId] = node;

    // 收集双边语义节点信息
    if (node->task->GetType() == TaskTypeStub::WAIT || g_rmtAsyncNodes.count(node->task->GetType()) != 0) {
        CHK_RET(CollectBilateralWaitInfo(curCcuTask, queId, node));
    }
    if (node->task->GetType() == TaskTypeStub::POST || g_rmtAsyncNodes.count(node->task->GetType()) != 0) {
        CHK_RET(CollectBilateralPostInfo(curCcuTask, queId, node));
    }
    if (g_rmtAsyncNodes.count(node->task->GetType()) != 0 || g_locAsyncNodes.count(node->task->GetType()) != 0) {
        curCcuTask->parallelNodes_.push_back(node);
    }
    return HCCL_SUCCESS;
}

HcclResult GetPeerRankByTaskNode(TaskNodePtr currNode, RankId &peerRank)
{
    if (currNode->task->GetType() == TaskTypeStub::READ) {
        TaskStubRead *read = dynamic_cast<TaskStubRead *>(currNode->task);
        peerRank = read->GetRemoteRank();
    } else if (currNode->task->GetType() == TaskTypeStub::READ_REDUCE) {
        TaskStubReadReduce *read = dynamic_cast<TaskStubReadReduce *>(currNode->task);
        peerRank = read->GetRemoteRank();
    } else if (currNode->task->GetType() == TaskTypeStub::WRITE) {
        TaskStubWrite *write = dynamic_cast<TaskStubWrite *>(currNode->task);
        peerRank = write->GetRemoteRank();
    } else if (currNode->task->GetType() == TaskTypeStub::WRITE_REDUCE) {
        TaskStubWriteReduce *write = dynamic_cast<TaskStubWriteReduce *>(currNode->task);
        peerRank = write->GetRemoteRank();
    } else {
        HCCL_ERROR("[GetPeerRankByTaskNode] Get peer rank by task node failed, task node type [%d] is not support.",
            static_cast<int>(currNode->task->GetType()));
        return HCCL_E_INTERNAL;
    }
    return HCCL_SUCCESS;
}

// 本端异步节点
void AddLocalCopy(uint32_t rankId, uint32_t queId, TaskStubCcuGraph *curCcuTask, const checker::DataSlice &srcSlice,
    const checker::DataSlice &dstSlice)
{
    TaskStub *localCopy = new TaskStubLocalCopy(srcSlice, dstSlice);
    auto localCopyNode = new TaskNode(localCopy, rankId, queId, curCcuTask->microCodePosInQue[queId]);
    curCcuTask->toDeleteTask_.push_back(localCopy);
    curCcuTask->toDeleteTaskNode_.push_back(localCopyNode);
    (void)AppendTailNode(curCcuTask, queId, localCopyNode);
}

void AddLocalReduce(uint32_t rankId, uint32_t queId, TaskStubCcuGraph *curCcuTask, const checker::DataSlice &srcSlice,
    const checker::DataSlice &dstSlice, CheckerDataType checkerDataType, CheckerReduceOp checkerReduceOp)
{
    TaskStub *localReduceTask = new TaskStubLocalReduce(srcSlice, dstSlice, checkerDataType, checkerReduceOp);
    auto localReduceNode = new TaskNode(localReduceTask, rankId, queId, curCcuTask->microCodePosInQue[queId]);
    curCcuTask->toDeleteTask_.push_back(localReduceTask);
    curCcuTask->toDeleteTaskNode_.push_back(localReduceNode);
    (void)AppendTailNode(curCcuTask, queId, localReduceNode);
}

void AddLocalBatchReduce(uint32_t rankId, uint32_t queId, TaskStubCcuGraph *curCcuTask,
    const std::vector<checker::DataSlice> &srcSlices, const checker::DataSlice &dstSlice, DataType hcclDataType)
{
    TaskStub *localBatchReduce = new TaskStubLocalBatchReduce(srcSlices, dstSlice,
        g_DataType2CheckerDataType_aicpu[hcclDataType], CheckerReduceOp::REDUCE_SUM);
    auto localBatchReduceNode = new TaskNode(localBatchReduce, rankId, queId, curCcuTask->microCodePosInQue[queId]);
    curCcuTask->toDeleteTask_.push_back(localBatchReduce);
    curCcuTask->toDeleteTaskNode_.push_back(localBatchReduceNode);
    (void)AppendTailNode(curCcuTask, queId, localBatchReduceNode);
}

// 远端异步节点
void AddWrite(uint32_t rankId, uint32_t rmtRankId, uint32_t queId, TaskStubCcuGraph *curCcuTask,
    const checker::DataSlice &srcSlice, const checker::DataSlice &dstSlice)
{
    LinkInfoStub link(LinkProtoStub::CCU);
    TaskStub *writeTask = new TaskStubWrite(rmtRankId, link, srcSlice, dstSlice);
    auto writeNode = new TaskNode(writeTask, rankId, queId, curCcuTask->microCodePosInQue[queId]);
    curCcuTask->toDeleteTask_.push_back(writeTask);
    curCcuTask->toDeleteTaskNode_.push_back(writeNode);
    (void)AppendTailNode(curCcuTask, queId, writeNode);
}

void AddRead(uint32_t rankId, uint32_t rmtRankId, uint32_t queId, TaskStubCcuGraph *curCcuTask,
    const checker::DataSlice &srcSlice, const checker::DataSlice &dstSlice)
{
    LinkInfoStub link(LinkProtoStub::CCU);
    TaskStub *readTask = new TaskStubRead(rmtRankId, link, dstSlice, srcSlice);
    auto readNode = new TaskNode(readTask, rankId, queId, curCcuTask->microCodePosInQue[queId]);
    curCcuTask->toDeleteTask_.push_back(readTask);
    curCcuTask->toDeleteTaskNode_.push_back(readNode);
    (void)AppendTailNode(curCcuTask, queId, readNode);
}

void AddWriteReduce(uint32_t rankId, uint32_t rmtRankId, uint32_t queId, TaskStubCcuGraph *curCcuTask,
    const checker::DataSlice &srcSlice, const checker::DataSlice &dstSlice, CheckerDataType checkerDataType, CheckerReduceOp checkerReduceOp)
{
    LinkInfoStub link(LinkProtoStub::CCU);
    TaskStub *writeTask = new TaskStubWriteReduce(rmtRankId, link, srcSlice, dstSlice, checkerDataType, checkerReduceOp);
    auto writeNode = new TaskNode(writeTask, rankId, queId, curCcuTask->microCodePosInQue[queId]);
    curCcuTask->toDeleteTask_.push_back(writeTask);
    curCcuTask->toDeleteTaskNode_.push_back(writeNode);
    (void)AppendTailNode(curCcuTask, queId, writeNode);
}

void AddReadReduce(uint32_t rankId, uint32_t rmtRankId, uint32_t queId, TaskStubCcuGraph *curCcuTask,
    const checker::DataSlice &srcSlice, const checker::DataSlice &dstSlice, CheckerDataType checkerDataType, CheckerReduceOp checkerReduceOp)
{
    LinkInfoStub link(LinkProtoStub::CCU);
    TaskStub *readTask = new TaskStubReadReduce(rmtRankId, link, dstSlice, srcSlice, checkerDataType, checkerReduceOp);
    auto readNode = new TaskNode(readTask, rankId, queId, curCcuTask->microCodePosInQue[queId]);
    curCcuTask->toDeleteTask_.push_back(readTask);
    curCcuTask->toDeleteTaskNode_.push_back(readNode);
    (void)AppendTailNode(curCcuTask, queId, readNode);
}

// Loop相关节点
TaskNodePtr AddLoopStartTask(uint32_t queId, uint32_t loopIdx, uint32_t loopGroupIdx, TaskStubCcuGraph *curCcuTask)
{
    TaskStub *loopStart = new TaskStubLoopStart(loopIdx, loopGroupIdx);
    auto loopStartNode = new TaskNode(loopStart, curCcuTask->GetRankId(), queId, curCcuTask->microCodePosInQue[queId]);
    curCcuTask->toDeleteTask_.push_back(loopStart);
    curCcuTask->toDeleteTaskNode_.push_back(loopStartNode);

    // 进行微码内连边
    (void)AppendTailNode(curCcuTask, queId, loopStartNode);
    LoopInfo loopInfo(loopStartNode, nullptr);
    if (loopIdx == 0) {
        std::vector<LoopInfo> loopTmp = {loopInfo};
        curCcuTask->loopGroupInfo_.push_back(loopTmp);
    } else {
        curCcuTask->loopGroupInfo_[loopGroupIdx].push_back(loopInfo);
    }

    return loopStartNode;
}

TaskNodePtr AddLoopEndTask(uint32_t queId, uint32_t loopIdx, uint32_t loopGroupIdx, TaskStubCcuGraph *curCcuTask)
{
    TaskStub *loopEnd = new TaskStubLoopEnd(loopIdx, loopGroupIdx);
    auto loopEndNode = new TaskNode(loopEnd, curCcuTask->GetRankId(), queId, curCcuTask->microCodePosInQue[queId]);
    curCcuTask->toDeleteTask_.push_back(loopEnd);
    curCcuTask->toDeleteTaskNode_.push_back(loopEndNode);

    // 进行微码内连边
    (void)AppendTailNode(curCcuTask, queId, loopEndNode);
    curCcuTask->loopGroupInfo_[loopGroupIdx][loopIdx].loopEnd = loopEndNode;

    return loopEndNode;
}

// Post、Wait节点
void AddPost(uint32_t rankId, uint32_t rmtRankId, uint32_t queId, TaskStubCcuGraph *curCcuTask,
    uint32_t rmtDieId, uint16_t rmtCKEId, uint16_t setRmtCKEMask)
{
    LinkInfoStub link(LinkProtoStub::CCU);
    std::string tag = "CCU_TASK";
    TaskStub *postTo = new TaskStubPost(rmtRankId, link, setRmtCKEMask, NotifyTypeStub::CCU, tag, curCcuTask);
    auto postToNode = new TaskNode(postTo, rankId, queId, curCcuTask->microCodePosInQue[queId]);
    curCcuTask->toDeleteTask_.push_back(postTo);
    curCcuTask->toDeleteTaskNode_.push_back(postToNode);
    // 进行微码内连边
    (void)AppendTailNode(curCcuTask, queId, postToNode);
    AllRankParamRecorder::Global()->seenPost[rmtRankId][rmtDieId][rmtCKEId].insert(postToNode);
}

TaskNodePtr AddWait(uint32_t rankId, uint32_t queId, TaskStubCcuGraph *curCcuTask, uint16_t remoteWaitMask)
{
    LinkInfoStub link(LinkProtoStub::CCU);
    std::string tag = "CCU_TASK";
    // TODO: 这边remoteRank是多少，其实并不清楚，需要后续匹配的时候进行确定
    TaskStub *remoteWaitTask = new TaskStubWait(rankId, link, remoteWaitMask, NotifyTypeStub::CCU, tag, curCcuTask);
    auto remoteWaitNode = new TaskNode(remoteWaitTask, rankId, queId, curCcuTask->microCodePosInQue[queId]);
    curCcuTask->toDeleteTask_.push_back(remoteWaitTask);
    curCcuTask->toDeleteTaskNode_.push_back(remoteWaitNode);
    (void)AppendTailNode(curCcuTask, queId, remoteWaitNode);
    return remoteWaitNode;
}

// localPost/localWait节点
void AddLocalPost(uint32_t rankId, uint32_t queId, uint32_t dieId, TaskStubCcuGraph *curCcuTask, uint16_t setCKEId,
    uint16_t setCKEMask, bool invalidPost)
{
    // TODO：这边的wait queId当前没有填，后续需要关注一下
    TaskStub *localPostTo = new TaskStubLocalPostTo(setCKEMask, queId, INVALID_QID, invalidPost);
    // TODO: 这边记录的pos为微码序列中的位置
    auto localPostToNode = new TaskNode(localPostTo, rankId, queId, curCcuTask->microCodePosInQue[queId]);
    curCcuTask->toDeleteTask_.push_back(localPostTo);
    curCcuTask->toDeleteTaskNode_.push_back(localPostToNode);
    // 进行微码内连边
    (void)AppendTailNode(curCcuTask, queId, localPostToNode);
    AllRankParamRecorder::Global()->seenPost[rankId][dieId][setCKEId].insert(localPostToNode);
}

TaskNodePtr AddLocalWait(uint32_t rankId, uint32_t queId, TaskStubCcuGraph *curCcuTask, uint16_t localWaitMask)
{
    TaskStub *localWaitTask = new TaskStubLocalWaitFrom(localWaitMask, INVALID_QID, queId);
    auto localWaitNode = new TaskNode(localWaitTask, rankId, queId, curCcuTask->microCodePosInQue[queId]);
    curCcuTask->toDeleteTask_.push_back(localWaitTask);
    curCcuTask->toDeleteTaskNode_.push_back(localWaitNode);
    (void)AppendTailNode(curCcuTask, queId, localWaitNode);
    return localWaitNode;
}

// CCU Graph整图尾节点
void AddCcuSubGraphEnd(TaskNodePtr node, TaskStubCcuGraph *curCcuTask)
{
    TaskStubSubGraphEnd *subGraphEndTask = new TaskStubSubGraphEnd(node);
    TaskNodePtr subGraphEndNode = new TaskNode(subGraphEndTask, curCcuTask->GetRankId(), -1, -1);
    curCcuTask->toDeleteTask_.push_back(subGraphEndTask);
    curCcuTask->toDeleteTaskNode_.push_back(subGraphEndNode);

    for (auto& tailNode : curCcuTask->tailNodes) {
        tailNode->children.push_back(subGraphEndNode);
        subGraphEndNode->parents.push_back(tailNode);
    }
}

} // namespace Hccl