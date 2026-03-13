/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2024-2024. All rights reserved.
 * Description: Graph Revamp for Parallelization
 * Create: 2025-10-15
 */
#include "task_graph_revamp_parallel.h"
#include <set>
#include <thread>
#include "task_ccu.h"
#include "alg_adapt_v2_interface.h"
#include "task_queue_stub.h"
#include "rank_info_recorder.h"
#include "ccu_task_common.h"

using namespace std;
using namespace Hccl;
namespace checker {

HcclResult GraphRevampParallel::Revamp(TaskNodePtr dummyStart)
{
    // revamp the graph for loop and async node parallel revamp
    CHK_PRT_RET(RevampGraph(dummyStart) != HcclResult::HCCL_SUCCESS,
        HCCL_ERROR("[GraphRevampBilateralCcu] Unable to revamp graph."),
        HcclResult::HCCL_E_INTERNAL);

    return HcclResult::HCCL_SUCCESS;
}

HcclResult GraphRevampParallel::RevampGraph4Rank(TaskNodePtr ccuHead, RankId rankId)
{
    HCCL_INFO("[RevampGraph4Rank] Start ccu parallel revamp for rank [%d].", rankId);
    // revamp the graph for loop nodes parallelization
    CHK_PRT_RET(ProcCcuNode4Loop(ccuHead, rankId) != HcclResult::HCCL_SUCCESS,
        HCCL_ERROR("[GraphRevampParallel] Rank[%d] Unable to revamp graph for loop.", rankId),
        HcclResult::HCCL_E_INTERNAL);
    
    // revamp the graph for async node parallelization
    CHK_PRT_RET(ProcCcuNode4Async(ccuHead, rankId) != HcclResult::HCCL_SUCCESS,
        HCCL_ERROR("[GraphRevampParallel] Rank[%d] Unable to revamp graph for async node.", rankId),
        HcclResult::HCCL_E_INTERNAL);

    HCCL_INFO("[RevampGraph4Rank] ccu parallel revamp success for rank [%d].", rankId);
    return HcclResult::HCCL_SUCCESS;
}

HcclResult GraphRevampParallel::ProcCcuNode4Loop(TaskNodePtr ccuHead, RankId rankId)
{
    TaskStubCcuGraph *curCcuTask = dynamic_cast<TaskStubCcuGraph *>(ccuHead->task);

    for (auto &groupInfo : curCcuTask->loopGroupInfo_) {
        LoopHeadTailPairs loopGroup;
        for (auto& loopInfo : groupInfo) {
            loopGroup.push_back(std::make_pair(loopInfo.loopStart, loopInfo.loopEnd));
        }
        CHK_RET(ProcLoopNode(ccuHead, loopGroup, rankId));
    }
    return HcclResult::HCCL_SUCCESS;
}

HcclResult GraphRevampParallel::ProcCcuNode4Async(TaskNodePtr ccuHead, RankId rankId)
{
    TaskStubCcuGraph *curCcuTask = dynamic_cast<TaskStubCcuGraph *>(ccuHead->task);
    for (auto &asyncNode : curCcuTask->parallelNodes_) {
        CHK_RET(ProcAsyncNode(ccuHead, asyncNode, curCcuTask->queueNum_++));
    }
    return HcclResult::HCCL_SUCCESS;
}

HcclResult GraphRevampParallel::ProcAsyncNode(TaskNodePtr ccuHead, TaskNodePtr currNode, uint32_t virtualQueId)
{
    RankId currRank = currNode->rankIdx;
    uint32_t queId = currNode->queIdx;
    // 获取currNode相关的{localPost, localWait}节点、以及在真实流上首尾节点
    TaskNodePtr locPostNode = nullptr;
    std::vector<TaskNodePtr> beforeAsynNodes;
    std::vector<TaskNodePtr> locPostAsynChild;
    CHK_RET(PorcessAsyncHeadTailNode(currNode, locPostNode, beforeAsynNodes, locPostAsynChild));

    // 修改真实流：头尾节点断开连接
    for (auto &child : beforeAsynNodes) {
        RemoveNodeRelation(child, currNode);
    }

    // 创建虚拟流：依次添加localWait(新增节点) -> currNode -> postNode
    // 创建虚拟流：生成loop头节点{localPost, localWait}
    TaskStub *localPostHead = new TaskStubLocalPostTo(currRank, queId, queId);
    auto localPostHeadNode  = new TaskNode(localPostHead, currRank, queId, beforeAsynNodes[0]->pos + 1);
    TaskStub *localWaitHead = new TaskStubLocalWaitFrom(currRank, virtualQueId, virtualQueId);
    auto localWaitHeadNode  = new TaskNode(localWaitHead, currRank, virtualQueId, 0);

    localPostHeadNode->asynParallel = true;
    localWaitHeadNode->asynParallel = true;
    toDeleteTaskResource_.push_back(localPostHead);
    toDeleteTaskNodeResource_.push_back(localPostHeadNode);
    toDeleteTaskResource_.push_back(localWaitHead);
    toDeleteTaskNodeResource_.push_back(localWaitHeadNode);

    AddNodeRelation(localPostHeadNode, localWaitHeadNode);
    AddNodeRelation(localWaitHeadNode, currNode);
    
    // 恢复真实流中原本与locPostNode的连接关系
    if (locPostNode != nullptr) {
        CHK_RET(RestorRealFlowConnection4Async(ccuHead, locPostNode, localPostHeadNode, locPostAsynChild));
    } else {
        for (auto childIter = currNode->children.begin(); childIter != currNode->children.end(); childIter++) {
            if ((*childIter)->queIdx != currNode->queIdx) {
                continue;
            }
            AddNodeRelation(localPostHeadNode, (*childIter));
        }
    }

    // 修改节点流ID
    currNode->queIdx      = virtualQueId;
    if (locPostNode != nullptr) {
        locPostNode->queIdx   = virtualQueId;
    }

    // 修改节点POS
    currNode->pos         = 1;
    if (locPostNode != nullptr) {
        locPostNode->pos      = currNode->pos + 1;
    }

    // 修改真实流：currNode和postNode断开连接，替换为localPost(新增节点)
    for (auto &child : beforeAsynNodes) {
        AddNodeRelation(child, localPostHeadNode);
    }

    return HcclResult::HCCL_SUCCESS;
}

HcclResult GraphRevampParallel::ProcLoopNode(TaskNodePtr ccuHead, LoopHeadTailPairs &loopNodes, RankId rankId)
{
    if (loopNodes.empty()) {
        return HcclResult::HCCL_SUCCESS;
    }

    // loop前节点：第一个LoopStart的前一个节点
    TaskNodePtr beforLoopNode = nullptr;
    TaskNodePtr afterLoopNode = nullptr;
    auto loopCnt = loopNodes.size();
    CHK_RET(PorcessLoopHeadTailNode(loopNodes, loopCnt, beforLoopNode, afterLoopNode, rankId));

    // 原来loopStart --> loopEnd连接关系断开
    for (int idx = 0; idx < loopCnt - 1; idx++) {
        RemoveNodeRelation(loopNodes[idx].second, loopNodes[idx + 1].first);
    }

    // 生成loop虚拟流：头尾节点放入队列栈中
    std::vector<LoopPostWaitNodes> loopPostWaitNodes;
    CHK_RET(CreateLoopVirStream(rankId, ccuHead, loopNodes, loopPostWaitNodes));

    auto beforLoopNodeTmp = beforLoopNode;
    // 修改真实流：头尾节点连接
    for (int idx = loopCnt - 1; idx >= 0; idx--) {
        auto loopInfo = loopPostWaitNodes[idx];
        AddNodeRelation(beforLoopNode, loopInfo.localPostHead);
        AddNodeRelation(loopInfo.localWaitTailNode, afterLoopNode);

        beforLoopNode = loopInfo.localPostHead;
        afterLoopNode = loopInfo.localWaitTailNode;
    }

    AddNodeRelation(beforLoopNode, afterLoopNode);

    return HcclResult::HCCL_SUCCESS;
}

HcclResult GraphRevampParallel::RestorRealFlowConnection4Async(TaskNodePtr ccuHead, TaskNodePtr locPostNode,
    TaskNodePtr localPostHeadNode, std::vector<TaskNodePtr> &locPostAsynChild)
{
    if (locPostAsynChild.empty()) {
        return HcclResult::HCCL_SUCCESS;
    }

    // 单Async节点场景：locPost只有一个配对locWait子节点 ———— 需要恢复真实流locPostHead --> locWait连接关系
    auto locPost = dynamic_cast<TaskStubLocalPostTo*>(locPostNode->task);
    if (locPostAsynChild.size() == 1) {
        if (locPostAsynChild[0]->task->GetType() != TaskTypeStub::LOCAL_WAIT_FROM) {
            HCCL_ERROR("[ProcAsyncNode] unexpected error: single async node has no localWait child");
            return HcclResult::HCCL_E_INTERNAL;
        }
        AddNodeRelation(localPostHeadNode, locPostAsynChild[0]);
        return HcclResult::HCCL_SUCCESS;
    }

    /**-----------------------------------------------------------------------------------
    连续Async节点场景： 
       1. 配对locWait子节点  ———— 不删除原有连接关系；
       2. 非配对子节点 ———— 删除原有连接关系，并与localPostHeadNode建立新连接关系
    --------------------------------------------------------------------------------------**/
    for (auto &child : locPostAsynChild) {
        TaskStubCcuGraph *curCcuTask = dynamic_cast<TaskStubCcuGraph *>(ccuHead->task);
        if (child->task->GetType() == TaskTypeStub::LOCAL_WAIT_FROM) {
            auto locWaitIt = curCcuTask->localPostWaitPairs_.find(locPostNode);
            if (locWaitIt == curCcuTask->localPostWaitPairs_.end()) {
                HCCL_ERROR("[ProcAsyncNode] unexpected error: local post node has no pair node(local wait).");
                return HcclResult::HCCL_E_INTERNAL;
            }
            // 跳过local post配对local wait节点
            if (locWaitIt->second == child) {
                continue;
            }
        }
        // 非配对子节点：需要恢复真实流中与locPostHead的连接关系
        RemoveNodeRelation(locPostNode, child);
        AddNodeRelation(localPostHeadNode, child);
    }
    return HcclResult::HCCL_SUCCESS;
}

HcclResult GraphRevampParallel::PorcessLoopHeadTailNode(
    const LoopHeadTailPairs &loopNodes, uint32_t loopCnt, TaskNodePtr &beforLoopNode, TaskNodePtr &afterLoopNode, RankId rankId)
{
    // 查找真实流中，loop块(以{LoopStart, LoopEnd}包含的指令块)的前后节点
    auto firstLoopStart = loopNodes[0].first;
    for (auto parentIter = firstLoopStart->parents.begin(); parentIter != firstLoopStart->parents.end(); parentIter++) {
        if ((*parentIter)->queIdx == firstLoopStart->queIdx) {
            beforLoopNode = (*parentIter);
            beforLoopNode->loopParallel = true;
        }
    }

    auto lastLoopEnd = loopNodes[loopCnt - 1].second;
    for (auto childIter = lastLoopEnd->children.begin(); childIter != lastLoopEnd->children.end(); childIter++) {
        if ((*childIter)->queIdx == lastLoopEnd->queIdx) {
            afterLoopNode = (*childIter);
            afterLoopNode->loopParallel = true;
        }
    }
    // 修改真实流：头尾节点断开连接
    if (beforLoopNode == nullptr) {
        HCCL_ERROR("[PorcessLoopHeadTailNode] unexpected beforLoopNode is null");
        return HcclResult::HCCL_E_INTERNAL;
    }
    if (afterLoopNode == nullptr) {
        HCCL_ERROR("[PorcessLoopHeadTailNode] unexpected afterLoopNode is null");
        return HcclResult::HCCL_E_INTERNAL;
    }

    RemoveNodeRelation(beforLoopNode, firstLoopStart);
    RemoveNodeRelation(lastLoopEnd, afterLoopNode);

    return HcclResult::HCCL_SUCCESS;
}

HcclResult GraphRevampParallel::CreateLoopVirStream(RankId currRank, TaskNodePtr ccuHead, const LoopHeadTailPairs &loopNodes, std::vector<LoopPostWaitNodes> &loopHeadTailNodes)
{
    TaskStubCcuGraph *curCcuTask = dynamic_cast<TaskStubCcuGraph *>(ccuHead->task);
    for (auto loopPair : loopNodes) {
        auto loopStart = loopPair.first;
        auto loopEnd = loopPair.second;
        // 创建虚拟流：生成loop头节点{localPost, localWait}
        TaskStub *localPostHead = new TaskStubLocalPostTo(currRank, loopStart->queIdx, loopStart->queIdx);
        auto localPostHeadNode  = new TaskNode(localPostHead, currRank, loopStart->queIdx, loopStart->pos + 1);
        TaskStub *localWaitHead = new TaskStubLocalWaitFrom(currRank, loopStart->queIdx, loopStart->queIdx);
        auto localWaitHeadNode  = new TaskNode(localWaitHead, currRank, loopStart->queIdx, 0); // 虚拟流的头结点
        toDeleteTaskResource_.push_back(localPostHead);
        toDeleteTaskNodeResource_.push_back(localPostHeadNode);
        toDeleteTaskResource_.push_back(localWaitHead);
        toDeleteTaskNodeResource_.push_back(localWaitHeadNode);

        // 创建虚拟流：生成loop尾节点{localPost, localWait}
        TaskStub *localPostTail = new TaskStubLocalPostTo(currRank, loopStart->queIdx, loopStart->queIdx);
        auto localPostTailNode  = new TaskNode(localPostTail, currRank, loopStart->queIdx, loopStart->pos + 1);
        TaskStub *localWaitTail = new TaskStubLocalWaitFrom(currRank, loopStart->queIdx, loopStart->queIdx);
        auto localWaitTailNode  = new TaskNode(localWaitTail, currRank, loopStart->queIdx, 0); // todo: 真实流中的pos

        // 修改节点流ID、节点pos
        LoopPostWaitNodes loopPostWaitNodes = {localPostHeadNode, localWaitHeadNode, localPostTailNode, localWaitTailNode};
        CHK_RET(ModifyNodeQueIdx(curCcuTask->queueNum_++, loopStart->queIdx, loopStart, loopPostWaitNodes));

        // 新建节点与原loop指令块建立连接关系
        AddNodeRelation(localPostHeadNode, localWaitHeadNode);
        AddNodeRelation(localWaitHeadNode, loopStart);
        AddNodeRelation(loopEnd, localPostTailNode);
        AddNodeRelation(localPostTailNode, localWaitTailNode);

        toDeleteTaskResource_.push_back(localPostTail);
        toDeleteTaskNodeResource_.push_back(localPostTailNode);
        toDeleteTaskResource_.push_back(localWaitTail);
        toDeleteTaskNodeResource_.push_back(localWaitTailNode);
    
        loopHeadTailNodes.push_back({localPostHeadNode, localWaitHeadNode, localPostTailNode, localWaitTailNode});
    }
    return HcclResult::HCCL_SUCCESS;
}

HcclResult GraphRevampParallel::ModifyNodeQueIdx(uint32_t virtualQueId, uint32_t relQueId, TaskNodePtr loopStart, LoopPostWaitNodes &loopPostWaitNodes)
{
    // 首尾的localPost和localWait节点，仍在真实流上
    loopPostWaitNodes.localPostHead->queIdx = relQueId;
    loopPostWaitNodes.localPostHead->loopParallel = true;
    loopPostWaitNodes.localWaitTailNode->queIdx = relQueId;
    loopPostWaitNodes.localWaitTailNode->loopParallel = true;

    // 虚拟流的首节点
    loopPostWaitNodes.localWaitHead->queIdx = virtualQueId;
    loopPostWaitNodes.localWaitHead->loopParallel = true;
    loopPostWaitNodes.localWaitHead->pos = 0;

    auto virtualPos = loopPostWaitNodes.localWaitHead->pos + 1;

    loopStart->queIdx = virtualQueId;
    loopStart->loopParallel = true;
    loopStart->pos = virtualPos++;

    // loopStart -> loopEnd均搬移到虚拟流，遍历修改流ID
    std::queue<TaskNodePtr> graphNodeQue;
    // queue结构体无法直接查找是否存在某个元素，此处利用set结构体查找是否存在某个元素
    std::set<TaskNodePtr> isVisited;
    // init graphNodeQue
    SearchGraphByQueueId(loopStart, graphNodeQue, isVisited, relQueId);
    while (!graphNodeQue.empty()) {
        TaskNodePtr currNode = graphNodeQue.front();
        graphNodeQue.pop();

        currNode->loopParallel = true;
        currNode->queIdx = virtualQueId;
        currNode->pos = virtualPos++;

        if (currNode->task->GetType() == TaskTypeStub::LOOP_END) {
            continue;
        }
        SearchGraphByQueueId(currNode, graphNodeQue, isVisited, relQueId);
    }

    // 虚拟流的尾节点
    loopPostWaitNodes.localPostTailNode->queIdx = virtualQueId;
    loopPostWaitNodes.localPostTailNode->loopParallel = true;
    loopPostWaitNodes.localPostTailNode->pos = virtualPos;

    return HcclResult::HCCL_SUCCESS;
}

HcclResult GraphRevampParallel::PorcessAsyncHeadTailNode(TaskNodePtr currNode, TaskNodePtr &locPostNode,
    std::vector<TaskNodePtr> &beforeAsynNodes, std::vector<TaskNodePtr> &locPostAsynChild)
{
    // 获取真实流中asyncNode相关的首节点
    for (auto parentIter = currNode->parents.begin(); parentIter != currNode->parents.end(); parentIter++) {
        if ((*parentIter)->queIdx == currNode->queIdx) {
            (*parentIter)->asynParallel = true;
            beforeAsynNodes.push_back((*parentIter));
        }
    }

    if (beforeAsynNodes.empty()) {
        HCCL_ERROR("[PorcessAsyncHeadTailNode] Unexpecred error: No head node found before async node.");
        return HcclResult::HCCL_E_INTERNAL;
    }

    // 查找async节点对应的{localPost, localWait}节点
    for (auto childIter = currNode->children.begin(); childIter != currNode->children.end(); childIter++) {
        if ((*childIter)->queIdx != currNode->queIdx) {
            continue;
        }
        if ((*childIter)->task->GetType() == TaskTypeStub::LOCAL_POST_TO) {
            (*childIter)->asynParallel = true;
            locPostNode = (*childIter);
            break;
        }
    }
    if (locPostNode == nullptr) {
        HCCL_WARNING("[PorcessAsyncHeadTailNode] Unexpecred warning: No local post node found after async node.");
        return HCCL_SUCCESS;
    }

    // 查找localPost的子节点（除配对的localWait外）
    for (auto childIter = locPostNode->children.begin(); childIter != locPostNode->children.end(); childIter++) {
        if ((*childIter)->queIdx != currNode->queIdx) {
            continue;
        }
        (*childIter)->asynParallel = true;
        locPostAsynChild.push_back((*childIter));
    }
    return HcclResult::HCCL_SUCCESS;
}

} // namespace checker