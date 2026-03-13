/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2024-2024. All rights reserved.
 * Description: Graph ReVamp
 * Author: huangweihao
 * Create: 2024-10-29
 */
#include "task_graph_revamp.h"
#include <queue>
#include <set>
#include "log.h"

#ifdef HCCL_ALG_ANALYZER_DAVID
#include "alg_adapt_v2_interface.h"
#endif

using namespace Hccl;

namespace checker {

HcclResult GraphRevampBilateralSemantics::Revamp(TaskNodePtr dummyStart)
{
    std::map<RankId, TaskNodePtr> rank2Head;
    CHK_PRT_RET(InitRankHead(dummyStart, rank2Head) != HcclResult::HCCL_SUCCESS,
                HCCL_ERROR("[GraphRevampBilateralSemantics] Unable to initialize head nodes for each rank."),
                HcclResult::HCCL_E_INTERNAL);

    // revamp the graph for two-side semantics and rdma doorbell specs
    CHK_PRT_RET(RevampGraph(dummyStart, rank2Head) != HcclResult::HCCL_SUCCESS,
                HCCL_ERROR("[GraphRevampBilateralSemantics] Unable to revamp graph."), HcclResult::HCCL_E_INTERNAL);

    return HcclResult::HCCL_SUCCESS;
}

HcclResult GraphRevampBilateralSemantics::InitRankHead(TaskNodePtr dummyStart, std::map<RankId, TaskNodePtr> &rank2Head)
{
    if (GraphRevampBase::rank2QueSize_.empty()) {
        auto childIter = dummyStart->children.begin();
        for (; childIter != dummyStart->children.end(); childIter++) {
            RankId myRank = (*childIter)->rankIdx;
            GraphRevampBase::rank2QueSize_[myRank] = TaskQueueStub::Global()->GetAllRankTasks().rank2TaskQueues[myRank]->taskQueues.size();
        }
    }

    auto childIter = dummyStart->children.begin();
    for (; childIter != dummyStart->children.end(); childIter++) {
        RankId myRank = (*childIter)->rankIdx;
        rank2Head.insert(std::make_pair(myRank, (*childIter)));
    }

    if (dummyStart->hasAivTask) {
        rank2AivStart_.clear();
        rank2AivStartSize_.clear();
        auto aivStartSet = AivTaskQueueStub::Global()->GetAllAivTasks().copyRank2AivTask;
        for (auto aivPos2AivStart : aivStartSet) {
            u32 totalSize = aivPos2AivStart.second.size();
            RankId rankId = aivPos2AivStart.first;
            for (u32 idx = 0; idx < totalSize; idx++) {
                auto aivStart = aivPos2AivStart.second[idx];
                auto blockSize = aivStart->children.size();
                auto aivRankPos = aivStart->rankPos;
                rank2AivStart_[rankId][aivRankPos] = aivStart;
                rank2AivStartSize_[rankId][aivRankPos] = blockSize;
            }
        }
    }

    return HcclResult::HCCL_SUCCESS;
}

HcclResult GraphRevampBilateralSemantics::RevampGraph(TaskNodePtr dummyStart, std::map<RankId, TaskNodePtr> &rank2Head)
{
    VirtQueMgr virtQueManager;
    std::queue<TaskNodePtr> graphNodeQue;
    // queue结构体无法直接查找是否存在某个元素，此处利用set结构体查找是否存在某个元素
    std::set<TaskNodePtr> isVisited;
    // init graphNodeQue
    dummyStart->procFlag = true;
    CHK_RET(ProceedNode(dummyStart, graphNodeQue, isVisited));
    while (!graphNodeQue.empty()) {
        TaskNodePtr currNode = graphNodeQue.front();
        graphNodeQue.pop();
        // 这边塞回队列的逻辑放在IsProceedParentNode里面，会影响可读性
        if (IsProceedParentNode(currNode, graphNodeQue, isVisited) && !currNode->procFlag) {
            currNode->procFlag = true;
            CHK_RET(ProceedNode(currNode, graphNodeQue, isVisited));
            switch (currNode->task->GetType()) {
                case TaskTypeStub::READ:
                case TaskTypeStub::READ_REDUCE:
                    CHK_RET(ProcReadNode(dummyStart, currNode, rank2Head, virtQueManager));
                    break;
                case TaskTypeStub::WRITE:
                case TaskTypeStub::WRITE_REDUCE:
                    CHK_RET(ProcWriteNode(dummyStart, currNode, rank2Head, virtQueManager));
                    break;
                case TaskTypeStub::AIV_TASK:
                    CHK_RET(ProcAivNode(currNode));
                    break;
                default:
                    break;
            }
        }
    }
    return HcclResult::HCCL_SUCCESS;
}

HcclResult GraphRevampBilateralSemantics::ProceedNode(TaskNodePtr currNode, std::queue<TaskNodePtr> &graphNodeQue,
                                                      std::set<TaskNodePtr> &isVisited)
{
    for (auto childIter = currNode->children.begin(); childIter != currNode->children.end(); childIter++) {
        if (!(*childIter)->procFlag) {
            graphNodeQue.push((*childIter));
            if (!isVisited.count(*childIter)) {
                isVisited.insert((*childIter));
            }
        }
    }
    return HcclResult::HCCL_SUCCESS;
}

bool GraphRevampBilateralSemantics::IsProceedParentNode(TaskNodePtr currNode, std::queue<TaskNodePtr> &graphNodeQue,
                                                        std::set<TaskNodePtr> &isVisited)
{
    for (auto parentIter = currNode->parents.begin(); parentIter != currNode->parents.end(); parentIter++) {
        if (!(*parentIter)->procFlag) {
            graphNodeQue.push(currNode);
            if (!isVisited.count(*parentIter)) {
                graphNodeQue.push((*parentIter));
                isVisited.insert(*parentIter);
            }
            return false;
        }
    }
    return true;
}

HcclResult GraphRevampBilateralSemantics::ProcReadNode(TaskNodePtr dummyStart, TaskNodePtr currNode,
    std::map<RankId, TaskNodePtr> &rank2Head, VirtQueMgr &virtQueManager)
{
    LinkProtoStub link;
    RankId peerRank;
    CHK_RET(GetPeerRankByTaskNode(currNode, peerRank));
    // aiv产生read write 给的链路类型
    CHK_RET(GetLinkProtoStubByTaskNode(currNode, link));

    if (link == LinkProtoStub::SDMA) {
        CHK_PRT_RET(
            ProcSdmaRWNode(dummyStart, currNode, rank2Head, virtQueManager) != HcclResult::HCCL_SUCCESS,
            HCCL_ERROR("[GraphRevampBilateralSemantics] fail to proceed READ taskNode locates in Rank [%d] - Que [%u] - Pos [%u], "
                "reading from Rank [%u].", currNode->rankIdx, currNode->queIdx, currNode->pos, peerRank),
            HcclResult::HCCL_E_INTERNAL);
    } else {
        HCCL_ERROR("[GraphRevampBilateralSemantics] Rank [%d], linkProto not supported yet.", currNode->rankIdx);
        return HcclResult::HCCL_E_INTERNAL;
    }
    return HcclResult::HCCL_SUCCESS;
}

HcclResult GraphRevampBilateralSemantics::ProcSdmaRWNode(TaskNodePtr dummyStart, TaskNodePtr currNode,
    std::map<RankId, TaskNodePtr> &rank2Head, VirtQueMgr &virtQueManager)
{
    // Search backward and add beingRead in peerRank
    CHK_RET(SearchBackwardSdmaRW(dummyStart, currNode, rank2Head, virtQueManager));

    // Search forward and see if beingRead can be terminated
    CHK_RET(SearchForwardSdmaRW(dummyStart, currNode, rank2Head, virtQueManager));

    return HcclResult::HCCL_SUCCESS;
}

HcclResult GraphRevampBilateralSemantics::SearchBackwardSdmaRW(TaskNodePtr dummyStart, TaskNodePtr currNode,
    std::map<RankId, TaskNodePtr> &rank2Head, VirtQueMgr &virtQueManager)
{
    RankId peerRank;
    CHK_RET(GetPeerRankByTaskNode(currNode, peerRank));

    std::queue<TaskNodePtr> candParents;
    std::set<TaskNodePtr> isVisited;
    if (currNode->parents.size() != 1) {
        HCCL_ERROR("[GraphRevampBilateralSemantics] read taskNode parent num is not 1, is [%d].", currNode->parents.size());
        return HcclResult::HCCL_E_INTERNAL;
    }
    candParents.push(currNode->parents[0]);
    isVisited.insert(currNode->parents[0]);
    // Search backward till wait(peerRank) or Read/Write(peerRank) being Found
    while (!candParents.empty()) {
        TaskNodePtr candNode = candParents.front();
        candParents.pop();
        if (candNode->task->GetType() == TaskTypeStub::WAIT) {
            TaskStubWait *candWait = dynamic_cast<TaskStubWait *>(candNode->task);
            if (candWait->GetRemoteRank() == peerRank) {
                // get Wait(peerRank)
                CHK_RET(AddBeingRWNodeToVirtualQueWithWait(candNode, currNode, dummyStart, rank2Head, virtQueManager));
                return HcclResult::HCCL_SUCCESS;
            }
        } else if (IsReadWriteWithSameRank(peerRank, candNode)) {
            // get read(currank)
            CHK_RET(AddBeingRWNodeToVirtualQue(currNode, dummyStart, rank2Head, virtQueManager));
            return HcclResult::HCCL_SUCCESS;
        }

        // update candParents
        auto candParentIter = candNode->parents.begin();
        for (; candParentIter != candNode->parents.end(); candParentIter++) {
            TaskNodePtr tmpCandParent = (*candParentIter);
            if ((tmpCandParent->rankIdx == currNode->rankIdx) && (!isVisited.count(tmpCandParent))) {
                candParents.push(tmpCandParent);
                isVisited.insert(tmpCandParent);
            }
        }
    }

    // no Wait(peerRank)
    CHK_RET(AddBeingRWNodeToVirtualQue(currNode, dummyStart, rank2Head, virtQueManager));
    return HcclResult::HCCL_SUCCESS;
}

HcclResult GraphRevampBilateralSemantics::SearchForwardSdmaRW(TaskNodePtr dummyStart, TaskNodePtr currNode,
    std::map<RankId, TaskNodePtr> &rank2Head, VirtQueMgr &virtQueManager)
{
    RankId peerRank;
    CHK_RET(GetPeerRankByTaskNode(currNode, peerRank));

    // Search forward till post(peerRank) or Read(peerRank) being Found
    std::queue<TaskNodePtr> candChildren;
    std::set<TaskNodePtr> isVisited;
    if (currNode->children.size() != 1) {
        HCCL_ERROR("[GraphRevampBilateralSemantics] read taskNode child num is not 1, is [%d].", currNode->children.size());
        return HcclResult::HCCL_E_INTERNAL;
    }
    candChildren.push(currNode->children[0]);
    isVisited.insert(currNode->parents[0]);
    while (!candChildren.empty()) {
        TaskNodePtr candNode = candChildren.front();
        candChildren.pop();
        if (candNode->task->GetType() == TaskTypeStub::POST) {
            TaskStubPost *candPost = dynamic_cast<TaskStubPost *>(candNode->task);
            if (candPost->GetRemoteRank() == peerRank) {
                // get Post(peerRank)
                CHK_RET(AddTerminalNodePeerRankVirtualQue(candNode, currNode, dummyStart, rank2Head, virtQueManager));
                return HcclResult::HCCL_SUCCESS;
            }
        } else if (IsReadWriteWithSameRank(peerRank, candNode)) {
            // no Post(peerRank)
            return HcclResult::HCCL_SUCCESS;
        }

        // update candChildren
        auto candChildrenIter = candNode->children.begin();
        for (; candChildrenIter != candNode->children.end(); candChildrenIter++) {
            TaskNodePtr tmpCandChildren = (*candChildrenIter);
            if ((tmpCandChildren->rankIdx == currNode->rankIdx) && (!isVisited.count(tmpCandChildren))) {
                candChildren.push(tmpCandChildren);
                isVisited.insert(tmpCandChildren);
            }
        }
    }

    // no Post(peerRank)
    return HcclResult::HCCL_SUCCESS;
}

HcclResult GraphRevampBilateralSemantics::AddBeingRWNodeToVirtualQue(TaskNodePtr currNode, TaskNodePtr dummyStart,
    std::map<RankId, TaskNodePtr> &rank2Head, VirtQueMgr &virtQueManager)
{
    RankId currRank = currNode->rankIdx;
    RankId peerRank;
    CHK_RET(GetPeerRankByTaskNode(currNode, peerRank));

    TaskStub *beingRWNode = GenTaskStubBeingReadOrWrittern(currNode);
    if (beingRWNode == nullptr) {
        HCCL_ERROR("[Generate Being Read Or Written Node failed]");
        return HCCL_E_PARA;
    }

    CHK_RET(PrepAvailVirtQueTail(peerRank, currRank, dummyStart, rank2Head, virtQueManager));
    TaskNodePtr peerRankVirtQueTailNode = virtQueManager[peerRank][currRank][0];
    TaskNodePtr beingReadNode = new TaskNode(
        beingRWNode, peerRank, peerRankVirtQueTailNode->queIdx, peerRankVirtQueTailNode->pos + 1);
    beingReadNode->procFlag = true;
    beingReadNode->realPeerNode = currNode;
    toDeleteTaskNodeResource_.push_back(beingReadNode);
    // virtQueTail --> beingRead
    peerRankVirtQueTailNode->children.push_back(beingReadNode);
    beingReadNode->parents.push_back(peerRankVirtQueTailNode);

    // update virtQueManageer
    virtQueManager[peerRank][currRank][0] = beingReadNode;
    return HcclResult::HCCL_SUCCESS;
}

HcclResult GraphRevampBilateralSemantics::AddBeingRWNodeToVirtualQueWithWait(TaskNodePtr waitNode, TaskNodePtr currNode, TaskNodePtr dummyStart,
    std::map<RankId, TaskNodePtr> &rank2Head, VirtQueMgr &virtQueManager)
{
    RankId currRank = currNode->rankIdx;
    RankId peerRank;
    CHK_RET(GetPeerRankByTaskNode(currNode, peerRank));
    TaskStub *beingRW = GenTaskStubBeingReadOrWrittern(currNode);
    if (beingRW == nullptr) {
        HCCL_ERROR("[Generate Being Read Or Written Node failed]");
        return HCCL_E_PARA;
    }

    // find Post(currRank) from parents of candidate Wait(PeerRank)
    auto waitParentIter = waitNode->parents.begin();
    for (; waitParentIter != waitNode->parents.end(); waitParentIter++) {
        TaskNodePtr waitParent = *waitParentIter;
        CHK_RET(PrepAvailVirtQueTail(peerRank, currRank, dummyStart, rank2Head, virtQueManager));
        TaskNodePtr peerRankVirtQueTailNode = virtQueManager[peerRank][currRank][0];
        if ((waitParent->rankIdx == peerRank) && (waitParent->task->GetType() == TaskTypeStub::POST)) {
            // post为什么会在虚拟流上面呢？除非是RDMA场景
            if (waitParent->queIdx != peerRankVirtQueTailNode->queIdx) {
                // Add localWaitFromShadow and beingBeing to virtual que of peerRank
                TaskStub *waitFromShadow = new TaskStubLocalWaitFromShadow(
                    currRank, peerRankVirtQueTailNode->queIdx, waitParent->queIdx);
                auto waitFromShadowNode = new TaskNode(
                    waitFromShadow, peerRank, peerRankVirtQueTailNode->queIdx, peerRankVirtQueTailNode->pos + 1);
                waitFromShadowNode->procFlag = true;
                toDeleteTaskResource_.push_back(waitFromShadow);
                toDeleteTaskNodeResource_.push_back(waitFromShadowNode);

                auto beingRWNode = new TaskNode(
                    beingRW, peerRank, peerRankVirtQueTailNode->queIdx, peerRankVirtQueTailNode->pos + 1);
                beingRWNode->procFlag = true;
                beingRWNode->realPeerNode = currNode;
                toDeleteTaskNodeResource_.push_back(beingRWNode);

                peerRankVirtQueTailNode->children.push_back(waitFromShadowNode);
                waitFromShadowNode->parents.push_back(peerRankVirtQueTailNode);
                waitFromShadowNode->children.push_back(beingRWNode);
                beingRWNode->parents.push_back(waitFromShadowNode);

                // Add localPostToShadow after Post(currRank) of peerRank
                TaskStub *postToShadow = new TaskStubLocalPostToShadow(
                    currRank, waitParent->queIdx, peerRankVirtQueTailNode->queIdx);
                auto postToShadowNode = new TaskNode(postToShadow, peerRank, waitParent->queIdx, waitParent->pos + 1);
                postToShadowNode->procFlag = true;
                toDeleteTaskResource_.push_back(postToShadow);
                toDeleteTaskNodeResource_.push_back(postToShadowNode);

                // Post(currRank) --> localPostToShadow --> original node
                CHK_PRT_RET(InsertNode(waitParent, postToShadowNode) != HcclResult::HCCL_SUCCESS,
                            HCCL_ERROR("[GraphRevampBilateralSemantics] Not able to insert localPostToShadow between Post(currRank) and "
                                    "original next node."),
                            HcclResult::HCCL_E_INTERNAL);

                // localPostToShadow --> LocalWaitFromShadow
                postToShadowNode->children.push_back(waitFromShadowNode);
                waitFromShadowNode->parents.push_back(postToShadowNode);

                // update virtQueManageer
                virtQueManager[peerRank][currRank][0] = beingRWNode;
                return HcclResult::HCCL_SUCCESS;
            } else {  // 这个场景是什么？
                auto beingRWNode = new TaskNode(
                    beingRW, peerRank, peerRankVirtQueTailNode->queIdx, peerRankVirtQueTailNode->pos + 1);
                beingRWNode->procFlag = true;
                beingRWNode->realPeerNode = currNode;
                toDeleteTaskNodeResource_.push_back(beingRWNode);

                beingRWNode->parents.push_back(waitParent);
                waitParent->children.push_back(beingRWNode);
                virtQueManager[peerRank][currRank][0] = beingRWNode;
                return HcclResult::HCCL_SUCCESS;
            }
        }
    }
    HCCL_ERROR("[GraphRevampBilateralSemantics] No Post(Rank [%d]) in parents of Wait(Rank [%d]).", peerRank, currRank);
    return HcclResult::HCCL_SUCCESS;
}

// 非虚拟队列下，在某个节点后插入一个节点
HcclResult GraphRevampBilateralSemantics::InsertNode(TaskNodePtr headNode, TaskNodePtr insertNode)
{
    // headNode --> insertNode --> originalNxtNode
    auto childIter = headNode->children.begin();
    for (; childIter != headNode->children.end(); childIter++) {
        TaskNodePtr originalNxtNode = (*childIter);
        if ((originalNxtNode->rankIdx == headNode->rankIdx) && (originalNxtNode->queIdx == headNode->queIdx)
            && (IsVirtualTask(originalNxtNode))) {
            InsertNode(originalNxtNode, insertNode);
            return HcclResult::HCCL_SUCCESS;
        } else if ((originalNxtNode->rankIdx == headNode->rankIdx) && (originalNxtNode->queIdx == headNode->queIdx)) {
            headNode->children.erase(childIter);

            // remove headNode from parents of originalNxtNode
            auto removeIter = std::remove(originalNxtNode->parents.begin(), originalNxtNode->parents.end(), headNode);
            originalNxtNode->parents.erase(removeIter, originalNxtNode->parents.end());

            headNode->children.push_back(insertNode);
            insertNode->parents.push_back(headNode);
            insertNode->children.push_back(originalNxtNode);
            originalNxtNode->parents.push_back(insertNode);
            return HcclResult::HCCL_SUCCESS;
        }
    }

    // headNode is last node of currQue
    headNode->children.push_back(insertNode);
    insertNode->parents.push_back(headNode);
    return HcclResult::HCCL_SUCCESS;
}

HcclResult GraphRevampBilateralSemantics::AddTerminalNodePeerRankVirtualQue(TaskNodePtr candNode, TaskNodePtr currNode, TaskNodePtr dummyStart,
    std::map<RankId, TaskNodePtr> &rank2Head, VirtQueMgr &virtQueManager)
{
    TaskStubPost *post = dynamic_cast<TaskStubPost *>(candNode->task);
    RankId currRank = currNode->rankIdx;
    RankId peerRank = post->GetRemoteRank();

    // find Wait(currRank) from children of candidate Post(peerRank)
    auto postChildIter = candNode->children.begin();
    for (; postChildIter != candNode->children.end(); postChildIter++) {
        TaskNodePtr postChild = (*postChildIter);
        if ((postChild->rankIdx == peerRank) && (postChild->task->GetType() == TaskTypeStub::WAIT)) {
            // Add LocalPostToShadow after BeingRead to virtual que of peerRank
            CHK_RET(PrepAvailVirtQueTail(peerRank, currRank, dummyStart, rank2Head, virtQueManager));
            TaskNodePtr peerRankVirtQueTailNode = virtQueManager[peerRank][currRank][0];
            // 什么情况下会不是呢？
            if (!IsBeingReadOrWrittenTask(peerRankVirtQueTailNode)) {
                return HCCL_SUCCESS;
            }

            TaskStub *postToShadow = new TaskStubLocalPostToShadow(
                currRank, peerRankVirtQueTailNode->queIdx, postChild->queIdx);
            auto postToShadowNode = new TaskNode
                (postToShadow, peerRank, peerRankVirtQueTailNode->queIdx, peerRankVirtQueTailNode->pos + 1);
            postToShadowNode->procFlag = true;
            toDeleteTaskResource_.push_back(postToShadow);
            toDeleteTaskNodeResource_.push_back(postToShadowNode);

            peerRankVirtQueTailNode->children.push_back(postToShadowNode);
            postToShadowNode->parents.push_back(peerRankVirtQueTailNode);
            virtQueManager[peerRank][currRank][0] = postToShadowNode;
            // Add LocalWaitFromShadow before Wait(currRank)
            TaskStub *waitFromShadow = new TaskStubLocalWaitFromShadow(
                currRank, postChild->queIdx, peerRankVirtQueTailNode->queIdx);
            auto waitFromShadowNode =
                new TaskNode(waitFromShadow, peerRank, postChild->queIdx, postToShadowNode->pos);
            waitFromShadowNode->procFlag = true;
            toDeleteTaskResource_.push_back(waitFromShadow);
            toDeleteTaskNodeResource_.push_back(waitFromShadowNode);

            auto waitParentIter = postChild->parents.begin();
            for (; waitParentIter != postChild->parents.end(); waitParentIter++) {
                if ((*waitParentIter)->rankIdx == peerRank) {
                    CHK_PRT_RET(InsertNode(*waitParentIter, waitFromShadowNode) != HcclResult::HCCL_SUCCESS,
                                HCCL_ERROR("[GraphRevampBilateralSemantics] Not able to insert localWaitFreom between Wait(peerRank) and "
                                           "original previous node."),
                                HcclResult::HCCL_E_INTERNAL);
                    break;
                }
            }
            // Connect the LocalPostToShadow and LocalWaitFromShadow
            postToShadowNode->children.push_back(waitFromShadowNode);
            waitFromShadowNode->parents.push_back(postToShadowNode);

            return HcclResult::HCCL_SUCCESS;
        }
    }

    HCCL_ERROR("[GraphRevampBilateralSemantics] No Wait(peerRank [%d]) in children of Post(currRank [%d]).", peerRank, currRank);
    return HcclResult::HCCL_SUCCESS;
}

HcclResult GraphRevampBilateralSemantics::PrepAvailVirtQueTail(const RankId myRank, const RankId peerRank, TaskNodePtr dummyStart,
    std::map<RankId, TaskNodePtr> &rank2Head, VirtQueMgr &virtQueManager)
{
    // virtQueManager: {myRank: {peerRank: {0: waitFromHeadQueNode}}}
    if (virtQueManager.find(myRank) == virtQueManager.end()) {
        // my rank has empty virtQues
        u32 VirtualQueId = GraphRevampBase::rank2QueSize_[myRank]++;
        TaskStub *waitFromHeadQue = new TaskStubLocalWaitFromShadow(peerRank, VirtualQueId, 0);
        auto waitFromHeadQueNode = new TaskNode(waitFromHeadQue, myRank, VirtualQueId, 0);
        waitFromHeadQueNode->procFlag = true;
        toDeleteTaskResource_.push_back(waitFromHeadQue);
        toDeleteTaskNodeResource_.push_back(waitFromHeadQueNode);
        CHK_RET(AddVirtQueTail(myRank, peerRank, dummyStart, rank2Head, waitFromHeadQueNode));

        std::vector<TaskNodePtr> tmpQueTails;
        tmpQueTails.push_back(waitFromHeadQueNode);
        std::map<RankId, std::vector<TaskNodePtr>> tmpRankQueTails;
        tmpRankQueTails.insert(std::make_pair(peerRank, tmpQueTails));
        virtQueManager.insert(std::make_pair(myRank, tmpRankQueTails));
    } else {
        if (virtQueManager.at(myRank).find(peerRank) == virtQueManager.at(myRank).end()) {
            // target rank has empty virtQues
            u32 VirtualQueId = GraphRevampBase::rank2QueSize_[myRank]++;
            TaskStub *waitFromHeadQue = new TaskStubLocalWaitFromShadow(peerRank, VirtualQueId, 0);
            auto waitFromHeadQueNode = new TaskNode(waitFromHeadQue, myRank, VirtualQueId, 0);
            waitFromHeadQueNode->procFlag = true;
            toDeleteTaskResource_.push_back(waitFromHeadQue);
            toDeleteTaskNodeResource_.push_back(waitFromHeadQueNode);
            CHK_RET(AddVirtQueTail(myRank, peerRank, dummyStart, rank2Head, waitFromHeadQueNode));

            std::vector<TaskNodePtr> tmpQueTails;
            tmpQueTails.push_back(waitFromHeadQueNode);
            virtQueManager[myRank].insert(std::make_pair(peerRank, tmpQueTails));
        }
    }

    return HcclResult::HCCL_SUCCESS;
}

HcclResult GraphRevampBilateralSemantics::AddVirtQueTail(const RankId myRank, const RankId remRank, TaskNodePtr dummyStart,
    std::map<RankId, TaskNodePtr> &rank2Head, TaskNodePtr waitFromHeadQueNode)
{
    // insert postToVirtQueNode before myRankHead and dummyStart
    TaskNodePtr myRankHead = rank2Head.at(myRank);
    TaskStub *postToVirtQue = new TaskStubLocalPostToShadow(remRank, myRankHead->queIdx, waitFromHeadQueNode->queIdx);
    auto postToVirtQueNode = new TaskNode(postToVirtQue, myRank, myRankHead->queIdx, 0);
    postToVirtQueNode->procFlag = true;
    toDeleteTaskResource_.push_back(postToVirtQue);
    toDeleteTaskNodeResource_.push_back(postToVirtQueNode);

    // dummyStart --X--> myRankHead
    auto dummyStartIter = std::find(myRankHead->parents.begin(), myRankHead->parents.end(), dummyStart);
    if (dummyStartIter != myRankHead->parents.end()) {
        myRankHead->parents.erase(dummyStartIter);
    } else {
        HCCL_ERROR("[GraphRevampBilateralSemantics] Rank [%d], rank Head ptr of myRank is not children of dummyStart.", myRank);
        return HcclResult::HCCL_E_INTERNAL;
    }

    auto myRankHeadIter = std::find(dummyStart->children.begin(), dummyStart->children.end(), myRankHead);
    if (myRankHeadIter != dummyStart->children.end()) {
        dummyStart->children.erase(myRankHeadIter);
    } else {
        HCCL_ERROR("[GraphRevampBilateralSemantics] Rank [%d], dummyStart is not parent of rank Head ptr.", myRank);
        return HcclResult::HCCL_E_INTERNAL;
    }

    // dummyStart --> postToVirtQueNode, replace myRankHead with postToVirtQueNode
    dummyStart->children.push_back(postToVirtQueNode);
    postToVirtQueNode->parents.push_back(dummyStart);
    rank2Head.erase(myRank);
    rank2Head.insert(std::make_pair(myRank, postToVirtQueNode));

    // postToVirtQueNode --> currRankHead
    postToVirtQueNode->children.push_back(myRankHead);
    myRankHead->parents.push_back(postToVirtQueNode);

    // postToVirtQueNode --> waitFromHeadQueNode
    postToVirtQueNode->children.push_back(waitFromHeadQueNode);
    waitFromHeadQueNode->parents.push_back(postToVirtQueNode);
    return HcclResult::HCCL_SUCCESS;
}

HcclResult GraphRevampBilateralSemantics::ProcWriteNode(TaskNodePtr dummyStart, TaskNodePtr currNode,
                                     std::map<RankId, TaskNodePtr> &rank2Head, VirtQueMgr &virtQueManager)
{
    LinkProtoStub link;
    RankId peerRank;
    CHK_RET(GetPeerRankByTaskNode(currNode, peerRank));
    CHK_RET(GetLinkProtoStubByTaskNode(currNode, link));

    if (link == LinkProtoStub::RDMA) {
        CHK_PRT_RET(
            ProcRdmaWriteNode(dummyStart, currNode, rank2Head, virtQueManager) != HcclResult::HCCL_SUCCESS,
            HCCL_ERROR("[GraphRevampBilateralSemantics] fail to proceed WRITE taskNode locates in Rank [%d] - Que [%u] - Pos [%u], "
                       "reading from Rank [%u].",
                       currNode->rankIdx, currNode->queIdx, currNode->pos, peerRank),
            HcclResult::HCCL_E_INTERNAL);
    } else if (link == LinkProtoStub::SDMA) {
        CHK_PRT_RET(
            ProcSdmaRWNode(dummyStart, currNode, rank2Head, virtQueManager) != HcclResult::HCCL_SUCCESS,
            HCCL_ERROR("[GraphRevampBilateralSemantics] fail to proceed WRITE taskNode locates in Rank [%d] - Que [%u] - Pos [%u], "
                       "reading from Rank [%u].",
                       currNode->rankIdx, currNode->queIdx, currNode->pos, peerRank),
            HcclResult::HCCL_E_INTERNAL);
    } else {
        HCCL_ERROR("[GraphRevampBilateralSemantics] Rank [%d], linkProto not supported yet.", currNode->rankIdx);
        return HcclResult::HCCL_E_INTERNAL;
    }
    return HcclResult::HCCL_SUCCESS;
}

HcclResult GraphRevampBilateralSemantics::ProcRdmaWriteNode(TaskNodePtr dummyStart, TaskNodePtr currNode,
                                          std::map<RankId, TaskNodePtr> &rank2Head, VirtQueMgr &virtQueManager)
{
    // Search backward and add beingRead in peerRank
    CHK_RET(SearchBackwardRdmaWrite(dummyStart, currNode, rank2Head, virtQueManager));

    // Search forward and see if beingRead can be terminated
    // In this Function transfer write to virtual queue
    CHK_RET(SearchForwardRdmaWrite(dummyStart, currNode, rank2Head, virtQueManager));

    return HcclResult::HCCL_SUCCESS;
}

HcclResult GraphRevampBilateralSemantics::SearchBackwardRdmaWrite(TaskNodePtr dummyStart, TaskNodePtr currNode,
                                   std::map<RankId, TaskNodePtr> &rank2Head, VirtQueMgr &virtQueManager)
{
    RankId peerRank;
    CHK_RET(GetPeerRankByTaskNode(currNode, peerRank));

    // Search backward till Read/Write(peerRank) being Found
    std::queue<TaskNodePtr> candParents;
    std::set<TaskNodePtr> isVisited;
    if (currNode->parents.size() != 1) {
        HCCL_ERROR("[GraphRevampBilateralSemantics] write taskNode parent num is not 1, is [%d].", currNode->parents.size());
        return HcclResult::HCCL_E_INTERNAL;
    }
    candParents.push(currNode->parents[0]);
    isVisited.insert(currNode->parents[0]);
    while (!candParents.empty()) {
        TaskNodePtr candNode = candParents.front();
        candParents.pop();
        if (candNode->task->GetType() == TaskTypeStub::WAIT) {
            TaskStubWait *candWait = dynamic_cast<TaskStubWait *>(candNode->task);
            if (candWait->GetRemoteRank() == peerRank) {
                CHK_RET(AddBeingRWNodeToVirtualQueWithWait(candNode, currNode, dummyStart, rank2Head, virtQueManager));
                CHK_RET(AddWaitToCurRankVitualQue(currNode, dummyStart, rank2Head, virtQueManager));
                return HcclResult::HCCL_SUCCESS;
            }
        } else if (candNode->task->GetType() == TaskTypeStub::LOCAL_POST_TO_SHADOW) {
            TaskStubLocalPostToShadow *candPostToShadow = dynamic_cast<TaskStubLocalPostToShadow *>(candNode->task);
            if (candPostToShadow->GetNeighborRank() == peerRank) {
                CHK_RET(AddBeingRWNodeToVirtualQue(currNode, dummyStart, rank2Head, virtQueManager));
                return HcclResult::HCCL_SUCCESS;
            }
        }

        // update candChildren
        auto candParentsIter = candNode->parents.begin();
        for (; candParentsIter != candNode->parents.end(); candParentsIter++) {
            TaskNodePtr tmpCandParents = (*candParentsIter);
            if ((tmpCandParents->rankIdx == currNode->rankIdx) && (!isVisited.count(tmpCandParents))) {
                candParents.push(tmpCandParents);
                isVisited.insert(tmpCandParents);
            }
        }
    }

    CHK_RET(AddBeingRWNodeToVirtualQue(currNode, dummyStart, rank2Head, virtQueManager));
    return HcclResult::HCCL_SUCCESS;
}

HcclResult GraphRevampBilateralSemantics::AddWaitToCurRankVitualQue(TaskNodePtr currNode, TaskNodePtr dummyStart,
    std::map<RankId, TaskNodePtr> &rank2Head, VirtQueMgr &virtQueManager)
{
    RankId currRank = currNode->rankIdx;
    RankId peerRank;
    CHK_RET(GetPeerRankByTaskNode(currNode, peerRank));

    CHK_RET(PrepAvailVirtQueTail(currRank, peerRank, dummyStart, rank2Head, virtQueManager));
    TaskNodePtr peerRankVirtQueTailNode = virtQueManager[currRank][peerRank][0];

    TaskNodePtr candNode = currNode->parents[0];
    TaskStub *waitFromShadow = new TaskStubLocalWaitFromShadow(peerRank, peerRankVirtQueTailNode->queIdx, candNode->queIdx);
    auto waitFromShadowNode = new TaskNode(
        waitFromShadow, currRank, peerRankVirtQueTailNode->queIdx, peerRankVirtQueTailNode->pos + 1);
    waitFromShadowNode->procFlag = true;
    toDeleteTaskResource_.push_back(waitFromShadow);
    toDeleteTaskNodeResource_.push_back(waitFromShadowNode);

    TaskStub *postToShadow = new TaskStubLocalPostToShadow(peerRank, candNode->queIdx, peerRankVirtQueTailNode->queIdx);
    auto postToShadowNode = new TaskNode(
        postToShadow, currRank, candNode->queIdx, candNode->pos + 1);
    postToShadowNode->procFlag = true;
    toDeleteTaskResource_.push_back(postToShadow);
    toDeleteTaskNodeResource_.push_back(postToShadowNode);

    CHK_RET(InsertNode(candNode, postToShadowNode));

    postToShadowNode->children.push_back(waitFromShadowNode);
    waitFromShadowNode->parents.push_back(postToShadowNode);
    waitFromShadowNode->parents.push_back(peerRankVirtQueTailNode);
    peerRankVirtQueTailNode->children.push_back(waitFromShadowNode);

    virtQueManager[currRank][peerRank][0] = waitFromShadowNode;
    return HcclResult::HCCL_SUCCESS;
}

// Written -> VirtualQue
// PostFin -> VirtualQue
HcclResult GraphRevampBilateralSemantics::TransferCurNodeToVitualQue(TaskNodePtr currNode, TaskNodePtr dummyStart,
    std::map<RankId, TaskNodePtr> &rank2Head, VirtQueMgr &virtQueManager)
{
    RankId currRank = currNode->rankIdx;
    RankId peerRank;
    if (currNode->task->GetType() == TaskTypeStub::WRITE) {
        TaskStubWrite *write = dynamic_cast<TaskStubWrite *>(currNode->task);
        peerRank = write->GetRemoteRank();
    } else if (currNode->task->GetType() == TaskTypeStub::POST) {
        TaskStubPost *postFin = dynamic_cast<TaskStubPost *>(currNode->task);
        peerRank = postFin->GetRemoteRank();
    } else if (currNode->task->GetType() == TaskTypeStub::WRITE_REDUCE) {
        TaskStubWriteReduce *writeReduce = dynamic_cast<TaskStubWriteReduce *>(currNode->task);
        peerRank = writeReduce->GetRemoteRank();
    } else {
        HCCL_ERROR("[TransferCurNodeToVitualQue] this node type should not be here");
        return HCCL_E_INTERNAL;
    }

    CHK_RET(PrepAvailVirtQueTail(currRank, peerRank, dummyStart, rank2Head, virtQueManager));
    TaskNodePtr peerRankVirtQueTailNode = virtQueManager[currRank][peerRank][0];

    TaskNodePtr parentNode;
    parentNode = currNode->parents[0];

    TaskNodePtr childNode = nullptr; // 对于postFin，有两个子节点，也可能只有一个
    std::vector<TaskNodePtr> candChildren;
    for (int i = 0; i < currNode->children.size(); i++) {
        if (currNode->children[i]->rankIdx == currRank) {
            childNode = currNode->children[i];
            auto removeIter = std::remove(currNode->children.begin(), currNode->children.end(), childNode);
            currNode->children.erase(removeIter, currNode->children.end());
            break;
        }
    }

    auto removeIter = std::remove(parentNode->children.begin(), parentNode->children.end(), currNode);
    parentNode->children.erase(removeIter, parentNode->children.end());
    if (childNode != nullptr) {
        parentNode->children.push_back(childNode);
    }

    peerRankVirtQueTailNode->children.push_back(currNode);
    currNode->parents.clear();
    currNode->parents.push_back(peerRankVirtQueTailNode);

    if (childNode != nullptr) {
        removeIter = std::remove(childNode->parents.begin(), childNode->parents.end(), currNode);
        childNode->parents.erase(removeIter, childNode->parents.end());
        childNode->parents.push_back(parentNode);
    }

    currNode->realqueId = currNode->queIdx;
    currNode->queIdx = peerRankVirtQueTailNode->queIdx;
    virtQueManager[currRank][peerRank][0] = currNode;
    return HcclResult::HCCL_SUCCESS;
}

HcclResult GraphRevampBilateralSemantics::SearchForwardRdmaWrite(TaskNodePtr dummyStart, TaskNodePtr currNode,
                                  std::map<RankId, TaskNodePtr> &rank2Head, VirtQueMgr &virtQueManager)
{
    RankId peerRank;
    CHK_RET(GetPeerRankByTaskNode(currNode, peerRank));

    std::queue<TaskNodePtr> candChildren;
    std::set<TaskNodePtr> isVisited;
    if (currNode->children.size() != 1) {
        HCCL_ERROR("[GraphRevampBilateralSemantics] write taskNode children num is not 1, is [%d].", currNode->children.size());
        return HcclResult::HCCL_E_INTERNAL;
    }
    candChildren.push(currNode->children[0]);
    isVisited.insert(currNode->parents[0]);

    CHK_RET(TransferCurNodeToVitualQue(currNode, dummyStart, rank2Head, virtQueManager));
    while (!candChildren.empty()) {
        TaskNodePtr candNode = candChildren.front();
        candChildren.pop();
        if (candNode->task->GetType() == TaskTypeStub::POST) {
            TaskStubPost *candPost = dynamic_cast<TaskStubPost *>(candNode->task);
            if (candPost->GetRemoteRank() == peerRank) {
                CHK_RET(AddTerminalNodePeerRankVirtualQue(candNode, currNode, dummyStart, rank2Head, virtQueManager));
            }
        } else if (candNode->task->GetType() == TaskTypeStub::WAIT) {
            TaskStubWait *candWait = dynamic_cast<TaskStubWait *>(candNode->task);
            if (candWait->GetRemoteRank() == peerRank) {
                CHK_RET(AddTerminalNodeCurRankVirtualQue(candNode, peerRank, dummyStart, rank2Head, virtQueManager));
                return HcclResult::HCCL_SUCCESS;
            }
        } else if (IsReadWriteWithSameRank(peerRank, candNode)) {
            return HcclResult::HCCL_SUCCESS;
        }

        // update candChildren
        auto candChildrenIter = candNode->children.begin();
        for (; candChildrenIter != candNode->children.end(); candChildrenIter++) {
            TaskNodePtr tmpCandChildren = (*candChildrenIter);
            if ((tmpCandChildren->rankIdx == currNode->rankIdx) && (!isVisited.count(tmpCandChildren))) {
                candChildren.push(tmpCandChildren);
                isVisited.insert(tmpCandChildren);
            }
        }

        if (candNode->task->GetType() == TaskTypeStub::POST) {
            TaskStubPost *candPost = dynamic_cast<TaskStubPost *>(candNode->task);
            if (candPost->GetRemoteRank() == peerRank) {
                CHK_RET(TransferCurNodeToVitualQue(candNode, dummyStart, rank2Head, virtQueManager));
            }
        }
    }
    return HcclResult::HCCL_SUCCESS;
}

HcclResult GraphRevampBilateralSemantics::AddTerminalNodeCurRankVirtualQue(TaskNodePtr candNode, RankId peerRank, TaskNodePtr dummyStart,
    std::map<RankId, TaskNodePtr> &rank2Head, VirtQueMgr &virtQueManager)
{
    RankId currRank = candNode->rankIdx;

    CHK_RET(PrepAvailVirtQueTail(currRank, peerRank, dummyStart, rank2Head, virtQueManager));
    TaskNodePtr peerRankVirtQueTailNode = virtQueManager[currRank][peerRank][0];
    CHK_PRT_RET(peerRankVirtQueTailNode->task->GetType() != TaskTypeStub::POST,
                HCCL_ERROR("[GraphRevampBilateralSemantics] The previous node of the virutal queue must be POST."),
                HcclResult::HCCL_E_INTERNAL);

    TaskStub *postToShadow = new TaskStubLocalPostToShadow(
        peerRank, peerRankVirtQueTailNode->queIdx, candNode->queIdx);
    auto postToShadowNode = new TaskNode
        (postToShadow, currRank, peerRankVirtQueTailNode->queIdx, peerRankVirtQueTailNode->pos + 1);
    postToShadowNode->procFlag = true;
    toDeleteTaskResource_.push_back(postToShadow);
    toDeleteTaskNodeResource_.push_back(postToShadowNode);

    TaskStub *waitFromShadow = new TaskStubLocalWaitFromShadow(
        peerRank, candNode->queIdx, peerRankVirtQueTailNode->queIdx);
    auto waitFromShadowNode = new TaskNode
        (waitFromShadow, currRank, candNode->queIdx, peerRankVirtQueTailNode->pos + 1);
    waitFromShadowNode->procFlag = true;
    toDeleteTaskResource_.push_back(waitFromShadow);
    toDeleteTaskNodeResource_.push_back(waitFromShadowNode);

    auto waitParentIter = candNode->parents.begin();
    for (; waitParentIter != candNode->parents.end(); waitParentIter++) {
        if ((*waitParentIter)->rankIdx == currRank) {
            CHK_PRT_RET(InsertNode(*waitParentIter, waitFromShadowNode) != HcclResult::HCCL_SUCCESS,
                HCCL_ERROR("[GraphRevampBilateralSemantics] Not able to insert localWaitFreom between Wait(peerRank) and "
                            "original previous node."),
                HcclResult::HCCL_E_INTERNAL);
            break;
        }
    }
    peerRankVirtQueTailNode->children.push_back(postToShadowNode);

    postToShadowNode->parents.push_back(peerRankVirtQueTailNode);
    postToShadowNode->children.push_back(waitFromShadowNode);

    waitFromShadowNode->parents.push_back(postToShadowNode);

    virtQueManager[currRank][peerRank][0] = postToShadowNode;
    return HcclResult::HCCL_SUCCESS;
}



bool GraphRevampBilateralSemantics::IsVirtualTask(TaskNodePtr node)
{
    switch (node->task->GetType()) {
        case TaskTypeStub::BEING_READ:
        case TaskTypeStub::BEING_READ_REDUCE:
        case TaskTypeStub::BEING_WRITTEN:
        case TaskTypeStub::BEING_WRITTEN_REDUCE:
        case TaskTypeStub::LOCAL_POST_TO_SHADOW:
        case TaskTypeStub::LOCAL_WAIT_FROM_SHADOW:
        case TaskTypeStub::SET_FLAG_SHADOW:
        case TaskTypeStub::WAIT_FLAG_SHADOW:
            return true;
        default:
            return false;
    }
}

bool GraphRevampBilateralSemantics::IsReadWriteWithSameRank(RankId peerRank, TaskNodePtr candNode)
{
    if (candNode->task->GetType() == TaskTypeStub::READ) {
        TaskStubRead *candRead = dynamic_cast<TaskStubRead *>(candNode->task);
        if (candRead->GetRemoteRank() == peerRank) {
            return true;
        }
    } else if (candNode->task->GetType() == TaskTypeStub::WRITE) {
        TaskStubWrite *candWrite = dynamic_cast<TaskStubWrite *>(candNode->task);
        if (candWrite->GetRemoteRank() == peerRank) {
            return true;
        }
    } else if (candNode->task->GetType() == TaskTypeStub::READ_REDUCE) {
        TaskStubReadReduce *candReadReduce = dynamic_cast<TaskStubReadReduce *>(candNode->task);
        if (candReadReduce->GetRemoteRank() == peerRank) {
            return true;
        }
    } else if (candNode->task->GetType() == TaskTypeStub::WRITE_REDUCE) {
        TaskStubWriteReduce *candWriteReduce = dynamic_cast<TaskStubWriteReduce *>(candNode->task);
        if (candWriteReduce->GetRemoteRank() == peerRank) {
            return true;
        }
    }
    return false;
}

bool GraphRevampBilateralSemantics::IsBeingReadOrWrittenTask(TaskNodePtr candNode)
{
    if (candNode->task->GetType() == TaskTypeStub::BEING_READ ||
        candNode->task->GetType() == TaskTypeStub::BEING_READ_REDUCE ||
        candNode->task->GetType() == TaskTypeStub::BEING_WRITTEN ||
        candNode->task->GetType() == TaskTypeStub::BEING_WRITTEN_REDUCE) {
            return true;
    }
    return false;
}

HcclResult GraphRevampBilateralSemantics::ProcAivNode(TaskNodePtr aivTaskNode)
{
    VirtAivBlockMgr virtAivBlockManager;
    std::queue<TaskNodePtr> aivNodeQue;
    std::set<TaskNodePtr> aivIsVisited;

    TaskNodePtr aivStart = ((AivTaskStub*)(aivTaskNode->task))->GetAivStart();
    aivStart->procFlag = true;
    aivIsVisited.insert(aivStart);
    for (auto startChild : aivStart->children) {
        startChild->procFlag = true;
        aivIsVisited.insert(startChild);
        CHK_RET(ProceedAivNode(startChild, aivNodeQue, aivIsVisited));
        while(!aivNodeQue.empty()) {
            TaskNodePtr currNode = aivNodeQue.front();
            aivNodeQue.pop();
       
            if (IsProceedAivParentNode(currNode, aivNodeQue, aivIsVisited) && !currNode->procFlag) {
                currNode->procFlag = true;
                CHK_RET(ProceedAivNode(currNode, aivNodeQue, aivIsVisited));
                switch (currNode->task->GetType()) {
                    case TaskTypeStub::READ:
                    case TaskTypeStub::READ_REDUCE:
                    case TaskTypeStub::WRITE:
                    case TaskTypeStub::WRITE_REDUCE:
                        if (IsGenFromSync(currNode->task)) {
                            CHK_RET(AddBeingAivRWNodeToVirtualQue(currNode, virtAivBlockManager));
                        } else {
                            CHK_RET(ProcAivRWNode(currNode, virtAivBlockManager));
                        }
                        break;
                    default:
                        break;
                }
            }
        }
    }

    return HcclResult::HCCL_SUCCESS;
}

HcclResult GraphRevampBilateralSemantics::ProcAivRWNode(TaskNodePtr currNode, VirtAivBlockMgr& virtAivBlockManager)
{
    LinkProtoStub link;
    RankId peerRank;
    CHK_RET(GetPeerRankByTaskNode(currNode, peerRank));
    CHK_RET(GetLinkProtoStubByTaskNode(currNode, link));
    if (link == LinkProtoStub::SDMA) {
        CHK_PRT_RET(
            ProcSdmaAivRWNode(currNode, virtAivBlockManager) != HcclResult::HCCL_SUCCESS,
            HCCL_ERROR("[GraphRevampBilateralSemantics] fail to proceed Aiv READ taskNode locates in Rank [%d] - RankPos [%u] - Block [%u] - Pipe [%s] - PipePOs [%d], "
                "reading from Rank [%u].", currNode->rankIdx, currNode->rankPos, currNode->blockIdx, GetPipeName((pipe_t)currNode->pipeIdx).c_str(), currNode->pipePos, peerRank),
            HcclResult::HCCL_E_INTERNAL);
    } else {
        HCCL_ERROR("[GraphRevampBilateralSemantics] Rank [%d], linkProto not supported yet.", currNode->rankIdx);
        return HcclResult::HCCL_E_INTERNAL;
    }
    return HcclResult::HCCL_SUCCESS;
}

HcclResult GraphRevampBilateralSemantics::ProcSdmaAivRWNode(TaskNodePtr currNode, VirtAivBlockMgr& virtAivBlockManager)
{
    CHK_RET(SearchBackwardSdmaAivRW(currNode, virtAivBlockManager));
    CHK_RET(SearchForwardSdmaAivRW(currNode, virtAivBlockManager));

    return HcclResult::HCCL_SUCCESS;    
}

HcclResult GraphRevampBilateralSemantics::SearchBackwardSdmaAivRW(TaskNodePtr currNode, VirtAivBlockMgr& virtAivBlockManager)
{
    RankId peerRank;
    CHK_RET(GetPeerRankByTaskNode(currNode, peerRank));

    std::queue<TaskNodePtr> candParents;
    std::set<TaskNodePtr> isVisited;
    if (currNode->parents.size() != 1) {
        HCCL_ERROR("[GraphRevampBilateralSemantics] read taskNode parent num is not 1, is [%d].", currNode->parents.size());
        return HcclResult::HCCL_E_INTERNAL;
    }
    candParents.push(currNode->parents[0]);
    isVisited.insert(currNode->parents[0]);

    while (!candParents.empty()) {
        TaskNodePtr candNode = candParents.front();
        candParents.pop();
        if (candNode->task->GetType() == TaskTypeStub::RECV_SYNC) {
            TaskStubRecvSync *candRecvSync = (TaskStubRecvSync*)(candNode->task);
            auto flagValue = candRecvSync->GetFlagValue();
            auto flagAddr = candRecvSync->GetFlagAddr();
            bool isFindSend = false;
            for (auto& recvParent : candNode->parents) {
                bool isSend = (recvParent->task->GetType() == TaskTypeStub::SEND_SYNC || recvParent->task->GetType() == TaskTypeStub::SEND_SYNC_REDUCE);
                if (isSend) {
                    int32_t sendFlagValue;
                    int32_t* sendFlagAddr;
                    bool isMatched;

                    switch (recvParent->task->GetType()) {
                        case TaskTypeStub::SEND_SYNC:
                            sendFlagValue = ((TaskStubSendSync*)(recvParent->task))->GetFlagValue();
                            sendFlagAddr = ((TaskStubSendSync*)(recvParent->task))->GetFlagAddr();
                            isMatched = (sendFlagValue == flagValue && sendFlagAddr == flagAddr && recvParent->rankIdx == peerRank);
                            break;
                        case TaskTypeStub::SEND_SYNC_REDUCE:
                            sendFlagValue = ((TaskStubSendSyncReduce*)(recvParent->task))->GetFlagValue();
                            sendFlagAddr = ((TaskStubSendSyncReduce*)(recvParent->task))->GetFlagAddr();
                            isMatched = (sendFlagAddr == flagAddr && recvParent->rankIdx == peerRank);
                            break;
                        default:
                            break;
                    }
                    
                    if (isMatched) {
                        isFindSend = true;
                        CHK_RET(AddBeingAivRWNodeToVirtualQueWithRecvSync(candNode, recvParent, currNode, virtAivBlockManager));
                    }
                }
            }

            if (isFindSend) {
                return HcclResult::HCCL_SUCCESS;
            } 

        } else if (IsReadWriteWithSameRank(peerRank, candNode) && !IsGenFromSync(candNode->task)) {
            CHK_RET(AddBeingAivRWNodeToVirtualQue(currNode, virtAivBlockManager));
            return HcclResult::HCCL_SUCCESS;
        }

        auto candParentIter = candNode->parents.begin();
        for (; candParentIter != candNode->parents.end(); candParentIter++) {
            TaskNodePtr tmpCandParent = (*candParentIter);
            if ((tmpCandParent->rankIdx == currNode->rankIdx) && tmpCandParent->rankPos == currNode->rankPos 
                    && tmpCandParent->blockIdx ==  currNode->blockIdx && (!isVisited.count(tmpCandParent))) {
                candParents.push(tmpCandParent);
                isVisited.insert(tmpCandParent);
            }
        }
    }

    CHK_RET(AddBeingAivRWNodeToVirtualQue(currNode, virtAivBlockManager));
    return HcclResult::HCCL_SUCCESS;
}

HcclResult GraphRevampBilateralSemantics::SearchForwardSdmaAivRW(TaskNodePtr currNode, VirtAivBlockMgr& virtAivBlockManager)
{
    RankId peerRank;
    CHK_RET(GetPeerRankByTaskNode(currNode, peerRank));
    std::queue<TaskNodePtr> candChildren;
    std::set<TaskNodePtr> isVisited;
    if (currNode->children.size() != 1) {
        HCCL_ERROR("[GraphRevampBilateralSemantics] read taskNode children num is not 1, is [%d].", currNode->children.size());
        return HcclResult::HCCL_E_INTERNAL;
    }
    candChildren.push(currNode->children[0]);
    isVisited.insert(currNode->children[0]);
    while (!candChildren.empty()) {
        TaskNodePtr candNode = candChildren.front();
        candChildren.pop();
        bool isSend = ((candNode->task->GetType() == TaskTypeStub::SEND_SYNC || candNode->task->GetType() == TaskTypeStub::SEND_SYNC_REDUCE));
        if (isSend) {
            int32_t sendFlagValue;
            int32_t* sendFlagAddr;
            bool isMatched = false;
            
            switch (candNode->task->GetType()) {
                case TaskTypeStub::SEND_SYNC:
                    sendFlagValue = ((TaskStubSendSync*)(candNode->task))->GetFlagValue();
                    sendFlagAddr = ((TaskStubSendSync*)(candNode->task))->GetFlagAddr();
                    break;
                case TaskTypeStub::SEND_SYNC_REDUCE:
                    sendFlagValue = ((TaskStubSendSyncReduce*)(candNode->task))->GetFlagValue();
                    sendFlagAddr = ((TaskStubSendSyncReduce*)(candNode->task))->GetFlagAddr();
                    break;
            }

            for (auto& sendChildren : candNode->children) {
                if (sendChildren->task->GetType() == TaskTypeStub::RECV_SYNC) {
                    TaskStubRecvSync*  sendParent = (TaskStubRecvSync*)(sendChildren->task);
                    auto flagValue = sendParent->GetFlagValue();
                    auto flagAddr = sendParent->GetFlagAddr();
                    switch (candNode->task->GetType()) {
                        case TaskTypeStub::SEND_SYNC:
                            isMatched = (sendFlagValue == flagValue && sendFlagAddr == flagAddr && sendChildren->rankIdx == peerRank);
                            break;
                        case TaskTypeStub::SEND_SYNC_REDUCE:
                            isMatched = (sendFlagAddr == flagAddr && sendChildren->rankIdx == peerRank);
                            break;
                    }
                    
                    if (isMatched) {
                        CHK_RET(AddAivTerminalNodePeerRankVirtualQue(candNode, sendChildren, currNode, virtAivBlockManager));
                    }
                }
            }
            return HcclResult::HCCL_SUCCESS; 
        } else if (IsReadWriteWithSameRank(peerRank,  candNode) && !IsGenFromSync(candNode->task)) {
            return HcclResult::HCCL_SUCCESS; 
        }
        auto candChildrentIter = candNode->children.begin();
        for (; candChildrentIter != candNode->children.end(); candChildrentIter++) {
            TaskNodePtr tmpCandChildren = (*candChildrentIter);
            if ((tmpCandChildren->rankIdx == currNode->rankIdx) && tmpCandChildren->rankPos == currNode->rankPos 
                    && tmpCandChildren->blockIdx ==  currNode->blockIdx && (!isVisited.count(tmpCandChildren))) {
                candChildren.push(tmpCandChildren);
                isVisited.insert(tmpCandChildren);
            }
        }
    }

    return HcclResult::HCCL_SUCCESS;    
}

HcclResult GraphRevampBilateralSemantics::AddBeingAivRWNodeToVirtualQueWithRecvSync(TaskNodePtr recvNode, TaskNodePtr sendNode, 
    TaskNodePtr currNode, VirtAivBlockMgr& virtAivBlockManager)
{
    RankId currRank = currNode->rankIdx;
    RankId peerRank = sendNode->rankIdx;
    u32 currRankPos = currNode->rankPos;
    u32 peerRankPos = sendNode->rankPos;
    std::pair<RankId, u32> currRankAndPosPair = std::make_pair(currRank, currRankPos);
    std::pair<RankId, u32> peerRankAndPosPair = std::make_pair(peerRank, peerRankPos);

    TaskStub *beingRW = GenTaskStubBeingReadOrWrittern(currNode);
    if (beingRW == nullptr) {
        HCCL_ERROR("[Generate Being Read Or Written Node failed]");
        return HCCL_E_PARA;
    }
    CHK_RET(PrepAvailAivVirtQueTail(peerRankAndPosPair, currRankAndPosPair, virtAivBlockManager));
    TaskNodePtr peerRankVirBlockTailNode = virtAivBlockManager[peerRankAndPosPair][currRankAndPosPair][0];
    u32 peerRankBLock = sendNode->blockIdx;
    u32 peerRankVirBlock = peerRankVirBlockTailNode->blockIdx;
    u32 peerRankPipe = sendNode->pipeIdx;
    u32 peerRankVirPipe = 0;

    TaskStub* waitFlagShadow =  new TaskStubWaitFlagShadow(currRank, peerRankPipe, peerRankVirPipe, 
                                                           peerRankBLock, peerRankVirBlock, true);
    auto waitFlagShadowNode = new TaskNode(waitFlagShadow, peerRank, peerRankPos, peerRankVirBlock, 
                                           peerRankVirPipe, peerRankVirBlockTailNode->pipePos + 1);
    waitFlagShadowNode->procFlag = true;
    toDeleteTaskResource_.push_back(waitFlagShadow);
    toDeleteTaskNodeResource_.push_back(waitFlagShadowNode);

    auto beginRWNode = new TaskNode(beingRW, peerRank, peerRankPos, peerRankVirBlock, 
                                    peerRankVirPipe, peerRankVirBlockTailNode->pipePos + 2);
    beginRWNode->procFlag = true;
    beginRWNode->realPeerNode = currNode;
    toDeleteTaskNodeResource_.push_back(beginRWNode);

    peerRankVirBlockTailNode->children.push_back(waitFlagShadowNode);
    waitFlagShadowNode->parents.push_back(peerRankVirBlockTailNode);
    waitFlagShadowNode->children.push_back(beginRWNode);
    beginRWNode->parents.push_back(waitFlagShadowNode);
    
    TaskStub* setFlagShadow = new TaskStubSetFlagShadow(currRank, peerRankPipe, peerRankVirPipe, 
                                                        peerRankBLock, peerRankVirBlock, false);
    auto setFlagShadowNode = new TaskNode(setFlagShadow, peerRank, peerRankPos, peerRankBLock, 
                                          peerRankPipe, sendNode->pipePos + 1);
    toDeleteTaskResource_.push_back(setFlagShadow);
    toDeleteTaskNodeResource_.push_back(setFlagShadowNode);

    CHK_PRT_RET(InsertAivNode(sendNode, setFlagShadowNode) != HcclResult::HCCL_SUCCESS,
                            HCCL_ERROR("[GraphRevampBilateralSemantics] Not able to insert SetFlagShadow between SendSync and "
                                       "original next node."),
                                       HcclResult::HCCL_E_INTERNAL);
    setFlagShadowNode->children.push_back(waitFlagShadowNode);
    waitFlagShadowNode->parents.push_back(setFlagShadowNode);

    virtAivBlockManager[peerRankAndPosPair][currRankAndPosPair][0] = beginRWNode;

    return HcclResult::HCCL_SUCCESS;                                                                          
}

HcclResult GraphRevampBilateralSemantics::AddBeingAivRWNodeToVirtualQue(TaskNodePtr currNode, VirtAivBlockMgr& virtAivBlockManager)
{
    RankId currRank = currNode->rankIdx;
    u32 currRankPos = currNode->rankPos;
    std::pair<RankId, u32> currRankAndPosPair = std::make_pair(currRank, currRankPos);
    RankId peerRank;
    CHK_RET(GetPeerRankByTaskNode(currNode, peerRank));
    u32 peerRankPos = currRankPos;
    std::pair<RankId, u32> peerRankAndPosPair = std::make_pair(peerRank, peerRankPos);

    TaskStub *beingRW = GenTaskStubBeingReadOrWrittern(currNode);
    if (beingRW == nullptr) {
        HCCL_ERROR("[Generate Being Read Or Written Node failed]");
        return HCCL_E_PARA;
    }

    CHK_RET(PrepAvailAivVirtQueTail(peerRankAndPosPair, currRankAndPosPair, virtAivBlockManager));
    TaskNodePtr peerRankVirBlockTailNode = virtAivBlockManager[peerRankAndPosPair][currRankAndPosPair][0];
    auto beginRWNode = new TaskNode(beingRW, peerRank, peerRankPos, peerRankVirBlockTailNode->blockIdx, 
                                    peerRankVirBlockTailNode->pipeIdx, peerRankVirBlockTailNode->pipePos + 1);
    beginRWNode->procFlag = true;
    beginRWNode->realPeerNode = currNode;
    toDeleteTaskNodeResource_.push_back(beginRWNode);

    peerRankVirBlockTailNode->children.push_back(beginRWNode);
    beginRWNode->parents.push_back(peerRankVirBlockTailNode);
    virtAivBlockManager[peerRankAndPosPair][currRankAndPosPair][0] = beginRWNode;

    return HcclResult::HCCL_SUCCESS;  
}

HcclResult GraphRevampBilateralSemantics::AddAivTerminalNodePeerRankVirtualQue(TaskNodePtr sendNode, TaskNodePtr recvNode, 
    TaskNodePtr currNode, VirtAivBlockMgr& virtAivBlockManager)
{
    RankId currRank = sendNode->rankIdx;
    RankId peerRank = recvNode->rankIdx;
    u32 currRankPos = sendNode->rankPos;
    u32 peerRankPos = recvNode->rankPos;
    std::pair<RankId, u32> currRankAndPosPair = std::make_pair(currRank, currRankPos);
    std::pair<RankId, u32> peerRankAndPosPair = std::make_pair(peerRank, peerRankPos);
    CHK_RET(PrepAvailAivVirtQueTail(peerRankAndPosPair, currRankAndPosPair, virtAivBlockManager));
    TaskNodePtr peerRankVirBlockTailNode = virtAivBlockManager[peerRankAndPosPair][currRankAndPosPair][0];

    u32 peerRankBlock = recvNode->blockIdx;
    u32 peerRankVirBlock = peerRankVirBlockTailNode->blockIdx;
    u32 peerRankPipe = recvNode->pipeIdx;
    u32 peerRankVirPipe = 0;
    TaskStub* setFlagShadow = new TaskStubSetFlagShadow(currRank, peerRankVirPipe, peerRankPipe, 
                                                        peerRankVirBlock, peerRankBlock, true);
    auto setFlagShadowNode = new TaskNode(setFlagShadow, peerRank, peerRankPos, peerRankVirBlock, 
                                      peerRankVirPipe, peerRankVirBlockTailNode->pipePos + 1);
    setFlagShadowNode->procFlag = true;
    toDeleteTaskResource_.push_back(setFlagShadow);
    toDeleteTaskNodeResource_.push_back(setFlagShadowNode);

    peerRankVirBlockTailNode->children.push_back(setFlagShadowNode);
    setFlagShadowNode->parents.push_back(peerRankVirBlockTailNode);
    virtAivBlockManager[peerRankAndPosPair][currRankAndPosPair][0] = setFlagShadowNode;

    TaskStub* waitFromShadow = new TaskStubWaitFlagShadow(currRank, peerRankVirPipe, peerRankPipe, 
                                                          peerRankVirBlock, peerRankBlock, false);
    auto waitFromShadowNode = new TaskNode(waitFromShadow, peerRank, peerRankPos, peerRankBlock, 
                                           peerRankPipe, recvNode->pipePos);
    toDeleteTaskResource_.push_back(waitFromShadow);
    toDeleteTaskNodeResource_.push_back(waitFromShadowNode);

    auto recvParentsIter = recvNode->parents.begin();
    for (; recvParentsIter != recvNode->parents.end(); recvParentsIter++) {
        if((*recvParentsIter)->rankIdx == peerRank && (*recvParentsIter)->rankPos == peerRankPos 
                && (*recvParentsIter)->blockIdx == peerRankBlock) {
            CHK_PRT_RET(InsertAivNode((*recvParentsIter), waitFromShadowNode) != HcclResult::HCCL_SUCCESS,
                        HCCL_ERROR("[GraphRevampBilateralSemantics] Not able to insert WaitFlagShadow between recvSync and "
                                   "original next node."),
                        HcclResult::HCCL_E_INTERNAL);
            break;        
        }
    }

    waitFromShadowNode->parents.push_back(setFlagShadowNode);
    setFlagShadowNode->children.push_back(waitFromShadowNode);

    return HcclResult::HCCL_SUCCESS;
}

HcclResult GraphRevampBilateralSemantics::PrepAvailAivVirtQueTail(std::pair<RankId, u32> peerRankAndPosPair, 
    std::pair<RankId, u32> currRankAndPosPair, VirtAivBlockMgr& virtAivBlockManager)
{
    RankId currRank = currRankAndPosPair.first;
    RankId peerRank = peerRankAndPosPair.first;
    u32 currRankPos = currRankAndPosPair.second;
    u32 peerRankPos = peerRankAndPosPair.second;
    TaskNode* peerRankAivStart = rank2AivStart_[peerRank][peerRankPos];

    if (virtAivBlockManager.find(peerRankAndPosPair) == virtAivBlockManager.end()) {
        u32 VirtualBlockId = rank2AivStartSize_[peerRank][peerRankPos]++;
        TaskStub* virtualRankStartStub = new TaskStubVirtualRankStart(currRank);
        TaskNode* virtualRankStartNode = new TaskNode(virtualRankStartStub, peerRank, peerRankPos, VirtualBlockId, 0, -1);
        virtualRankStartNode->procFlag = true;
        toDeleteTaskResource_.push_back(virtualRankStartStub);
        toDeleteTaskNodeResource_.push_back(virtualRankStartNode);

        std::vector<TaskNodePtr> tmpBlockTails;
        tmpBlockTails.push_back(virtualRankStartNode);
        std::map<std::pair<RankId, u32>, std::vector<TaskNodePtr>> tmpRankBlockTails;
        tmpRankBlockTails.insert(std::make_pair(currRankAndPosPair, tmpBlockTails));
        virtAivBlockManager.insert(std::make_pair(peerRankAndPosPair, tmpRankBlockTails));
        peerRankAivStart->children.push_back(virtualRankStartNode);
        virtualRankStartNode->parents.push_back(peerRankAivStart);

    } else {
        if (virtAivBlockManager.at(peerRankAndPosPair).find(currRankAndPosPair) == virtAivBlockManager.at(peerRankAndPosPair).end()) {
            u32 VirtualBlockId = rank2AivStartSize_[peerRank][peerRankPos]++;
            TaskStub* virtualRankStartStub = new TaskStubVirtualRankStart(currRank);
            TaskNode* virtualRankStartNode = new TaskNode(virtualRankStartStub, peerRank, peerRankPos, VirtualBlockId, 0, -1);
            virtualRankStartNode->procFlag = true;
            toDeleteTaskResource_.push_back(virtualRankStartStub);
            toDeleteTaskNodeResource_.push_back(virtualRankStartNode);
            std::vector<TaskNodePtr> tmpBlockTails;
            tmpBlockTails.push_back(virtualRankStartNode);
            virtAivBlockManager[peerRankAndPosPair].insert(std::make_pair(currRankAndPosPair, tmpBlockTails));
            peerRankAivStart->children.push_back(virtualRankStartNode);
            virtualRankStartNode->parents.push_back(peerRankAivStart);
        }
    }
    return HcclResult::HCCL_SUCCESS;
}

HcclResult GraphRevampBilateralSemantics::InsertAivNode(TaskNodePtr headNode, TaskNodePtr insertNode)
{
    if (headNode->task->GetType() == TaskTypeStub::PIPE_BARRIER && ((TaskStubPipeBarrier*)(headNode->task))->IsPipeBarrierAll()) {
        auto targetPipe =  insertNode->pipeIdx;
        for (auto barrierChild : headNode->children) {
            bool isPipeBarrierAll = barrierChild->task->GetType() == TaskTypeStub::PIPE_BARRIER 
                                    && ((TaskStubPipeBarrier*)(barrierChild->task))->IsPipeBarrierAll();
            if(barrierChild->pipeIdx == targetPipe && !isPipeBarrierAll) {
                auto removeIter = std::remove(headNode->children.begin(), headNode->children.end(), barrierChild);
                headNode->children.erase(removeIter);
                removeIter = std::remove(barrierChild->parents.begin(), barrierChild->parents.end(), headNode);
                barrierChild->parents.erase(removeIter);

                headNode->children.push_back(insertNode);
                insertNode->parents.push_back(headNode);
                insertNode->children.push_back(barrierChild);
                barrierChild->parents.push_back(insertNode);
                return HcclResult::HCCL_SUCCESS;
            }
        }
    } else {
        auto childIter = headNode->children.begin();
        for (; childIter != headNode->children.end(); childIter++) {
            TaskNodePtr originalNxtNode = (*childIter);
            bool isExpectedPos = (originalNxtNode->rankIdx == headNode->rankIdx) && (originalNxtNode->rankPos == headNode->rankPos) 
                                    && (originalNxtNode->blockIdx == headNode->blockIdx);
            bool isSamePipe = (originalNxtNode->task->GetType() == TaskTypeStub::PIPE_BARRIER && ((TaskStubPipeBarrier*)(originalNxtNode->task))->IsPipeBarrierAll())
                                || (originalNxtNode->pipeIdx == headNode->pipeIdx);
            bool isAivEnd =  originalNxtNode->task->GetType() == TaskTypeStub::AIV_END;
            if (IsVirtualTask(originalNxtNode) && isExpectedPos) {
                InsertAivNode(originalNxtNode, insertNode);
                return HcclResult::HCCL_SUCCESS;
            } else if ((isExpectedPos && isSamePipe) || isAivEnd) {
                headNode->children.erase(childIter);
                auto removeIter = std::remove(originalNxtNode->parents.begin(), originalNxtNode->parents.end(), headNode);
                originalNxtNode->parents.erase(removeIter, originalNxtNode->parents.end());

                headNode->children.push_back(insertNode);
                insertNode->parents.push_back(headNode);
                insertNode->children.push_back(originalNxtNode);
                originalNxtNode->parents.push_back(insertNode);
                return HcclResult::HCCL_SUCCESS;
            }
        }
    }

    headNode->children.push_back(insertNode);
    insertNode->parents.push_back(headNode);
    return HcclResult::HCCL_SUCCESS;
}

HcclResult GraphRevampBilateralSemantics::ProceedAivNode(TaskNodePtr currNode, std::queue<TaskNodePtr> &graphNodeQue,
                                                      std::set<TaskNodePtr> &isVisited)
{
    for (auto childIter = currNode->children.begin(); childIter != currNode->children.end(); childIter++) {
        if (!(*childIter)->procFlag && (*childIter)->rankIdx == currNode->rankIdx) {
            graphNodeQue.push((*childIter));
            if (!isVisited.count(*childIter)) {
                isVisited.insert((*childIter));
            }
        }
    }
    return HcclResult::HCCL_SUCCESS;
}

bool GraphRevampBilateralSemantics::IsProceedAivParentNode(TaskNodePtr currNode, std::queue<TaskNodePtr> &graphNodeQue,
                                                        std::set<TaskNodePtr> &isVisited)
{
    for (auto parentIter = currNode->parents.begin(); parentIter != currNode->parents.end(); parentIter++) {
        if (!(*parentIter)->procFlag && (*parentIter)->rankIdx == currNode->rankIdx) {
            graphNodeQue.push(currNode);
            if (!isVisited.count(*parentIter)) {
                graphNodeQue.push((*parentIter));
                isVisited.insert(*parentIter);
            }
            return false;
        }
    }
    return true;
}

} // namespace Hccl