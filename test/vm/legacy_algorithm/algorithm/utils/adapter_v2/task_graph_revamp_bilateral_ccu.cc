/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2024-2024. All rights reserved.
 * Description: Graph Revamp for Parallelization
 * Create: 2025-10-15
 */
#include "task_graph_revamp_bilateral_ccu.h"
#include <set>
#include "task_ccu.h"
#include "alg_adapt_v2_interface.h"
#include "task_queue_stub.h"
#include "rank_info_recorder.h"
#include "ccu_task_common.h"

using namespace Hccl;
namespace checker {

HcclResult GraphRevampBilateralCcu::Revamp(TaskNodePtr dummyStart)
{
    // revamp the graph for two-side semantics and rdma doorbell specs
    CHK_PRT_RET(RevampGraph(dummyStart) != HcclResult::HCCL_SUCCESS,
        HCCL_ERROR("[GraphRevampBilateralCcu] Unable to revamp graph."),
        HcclResult::HCCL_E_INTERNAL);

    return HcclResult::HCCL_SUCCESS;
}

HcclResult GraphRevampBilateralCcu::RevampGraph4Rank(TaskNodePtr ccuHead, RankId rankId)
{
    HCCL_INFO("[RevampGraph4Rank] Start ccu bilateral revamp for rank [%d].", rankId);

    TaskStubCcuGraph *curCcuTask = dynamic_cast<TaskStubCcuGraph *>(ccuHead->task);

    for (uint32_t queId = 0; queId < curCcuTask->bilateralPart1_.size(); queId++) {
        for (auto &waitIter : curCcuTask->bilateralPart1_[queId]) {
            auto postRes = curCcuTask->bilateralPart2_[queId].find(waitIter.first);
            if (postRes == curCcuTask->bilateralPart2_[queId].end()) {
                HCCL_ERROR("[RevampGraph4Rank] Unexpected error: cannot find node: %s.", waitIter.first->task->Describe().c_str());
                return HcclResult::HCCL_E_INTERNAL;
            }
            BilateralNode bilateralNode(waitIter.first, waitIter.second, postRes->second);
            curCcuTask->bilateralNodes_[queId].push_back(bilateralNode);
        }
    }

    curCcuTask->waitInfoTmp_.clear();
    curCcuTask->postInfoTmp_.clear();
    curCcuTask->bilateralPart1_.clear();
    curCcuTask->bilateralPart2_.clear();

    for (uint32_t queId = 0; queId < curCcuTask->bilateralNodes_.size(); queId++) {
        for (auto &bilateralNode : curCcuTask->bilateralNodes_[queId]) {
            CHK_RET(ProcCcuRWNode(ccuHead, bilateralNode.asyncNode, bilateralNode.backwardWait, bilateralNode.forwardPost));
        }
    }
    HCCL_INFO("[RevampGraph4Rank] ccu bilateral revamp success for rank [%d].", rankId);
    return HcclResult::HCCL_SUCCESS;
}

HcclResult GraphRevampBilateralCcu::ProcCcuRWNode(TaskNodePtr ccuNode, TaskNodePtr asyncNode, TaskNodePtr waitNode, TaskNodePtr postNode)
{
    LinkProtoStub link;
    RankId peerRank;
    CHK_RET(GetPeerRankByTaskNode(asyncNode, peerRank));
    CHK_RET(GetLinkProtoStubByTaskNode(asyncNode, link));
    CHK_PRT_RET(ProcAsyncCcuRWNode(ccuNode, asyncNode, waitNode, postNode) != HcclResult::HCCL_SUCCESS,
        HCCL_ERROR("[GraphRevampBilateralCcu] fail to proceed Ccu async taskNode locates in Rank [%d] - RankPos "
                   "[%u] - PeerRank[%u], ", asyncNode->rankIdx, asyncNode->rankPos, peerRank),
        HcclResult::HCCL_E_INTERNAL);

    return HcclResult::HCCL_SUCCESS;
}

HcclResult GraphRevampBilateralCcu::ProcAsyncCcuRWNode(TaskNodePtr ccuNode, TaskNodePtr asyncNode, TaskNodePtr waitNode, TaskNodePtr postNode)
{
    TaskNodePtr beingRWNode = nullptr;
    CHK_RET(SearchBackwardCcuRW(ccuNode, asyncNode, waitNode, beingRWNode));
    if (beingRWNode == nullptr) {
        HCCL_ERROR("[GraphRevampBilateralCcu] fail to create beingRWNode, rank: [%u].", asyncNode->rankIdx);
        return HcclResult::HCCL_E_INTERNAL;
    }

    CHK_RET(SearchForwardCcuRW(ccuNode, asyncNode, postNode, beingRWNode));

    return HcclResult::HCCL_SUCCESS;    
}

HcclResult GraphRevampBilateralCcu::SearchBackwardCcuRW(
    TaskNodePtr ccuNode, TaskNodePtr asyncNode, TaskNodePtr waitNode, TaskNodePtr &beingRWNode)
{
    RankId peerRank;
    CHK_RET(GetPeerRankByTaskNode(asyncNode, peerRank));

    if (waitNode != nullptr) {
        CHK_RET(CreateNewVirtualStream(waitNode, asyncNode, beingRWNode, peerRank));
    } else {
        // 找不到WAIT节点场景：虚拟流直接挂在ccu头节点
        HCCL_WARNING("[SearchBackwardCcuRW] Asyn node [type=%d] does not have a matching wait node.",
            static_cast<int>(asyncNode->task->GetType()));
        CHK_RET(AddBeingRWNode2CcuHead(ccuNode, asyncNode, beingRWNode, peerRank));
    }
    return HcclResult::HCCL_SUCCESS;
}

HcclResult GraphRevampBilateralCcu::SearchForwardCcuRW(
    TaskNodePtr ccuNode, TaskNodePtr asyncNode, TaskNodePtr postNode, TaskNodePtr &beingRWNode)
{
    RankId peerRank;
    CHK_RET(GetPeerRankByTaskNode(asyncNode, peerRank));

    if (postNode != nullptr) {
        CHK_RET(CloseLoopNewVirtualStream(postNode, asyncNode, beingRWNode, peerRank));
    } else {
        HCCL_WARNING("[SearchBackwardCcuRWNew] Asyn node [type=%d] does not have a matching post node.",
            static_cast<int>(asyncNode->task->GetType()));
    }
    return HcclResult::HCCL_SUCCESS;
}

// 找到RW节点，并确定虚拟流在真实流中的前半部分插入位置
// 若往前找得到Wait节点，则在对应对端rank的POST节点插入虚拟流 ———— headNode = Post节点
// 若往前找不到Wait节点，则在对应对端rank的CCU头节点插入虚拟流 ———— headNode = CcuHead节点
HcclResult GraphRevampBilateralCcu::AddVirtualFlowFirstHalfPart(
    TaskNodePtr headNode, TaskNodePtr headChild, TaskNodePtr currNode, TaskNodePtr &beingRWNode, TaskStub *beingRW)
{
    RankId peerRank;
    CHK_RET(GetPeerRankByTaskNode(currNode, peerRank));
    RankId currRank = currNode->rankIdx;

    // 在真实流中某个节点(post或ccuNode)插入虚拟流
    RemoveNodeRelation(headNode, headChild);

    uint32_t virQueId = 0;
    if (headNode->task->GetType() == TaskTypeStub::POST) {
        auto rmtPostNode = dynamic_cast<TaskStubPost*>(headNode->task);
        if (rmtPostNode->ccuTaskPtr_ == 0) {
            HCCL_ERROR("[AddVirtualFlowFirstHalfPart] cannot find ccu subgraph head node by post node");
            return HCCL_E_INTERNAL;
        }
        TaskStubCcuGraph *rmtCcuTask = reinterpret_cast<TaskStubCcuGraph *>(rmtPostNode->ccuTaskPtr_);
        virQueId = rmtCcuTask->queueNum_++;
    } else {
        TaskStubCcuGraph *rmtCcuTask = dynamic_cast<TaskStubCcuGraph *>(headNode->task);
        virQueId = rmtCcuTask->queueNum_++;
        headNode = Hccl::GetCcuTaskHead(headNode);
    }

    TaskStub *waitFromShadow = new TaskStubLocalWaitFromShadow(currRank, virQueId, virQueId);
    auto waitFromShadowNode = new TaskNode(waitFromShadow, peerRank, virQueId, 0);
    waitFromShadowNode->ccuBilateral = true;
    toDeleteTaskResource_.push_back(waitFromShadow);
    toDeleteTaskNodeResource_.push_back(waitFromShadowNode);

    beingRWNode = new TaskNode(beingRW, peerRank, virQueId, 1);
    beingRWNode->ccuBilateral = true;
    beingRWNode->realPeerNode = currNode;
    toDeleteTaskNodeResource_.push_back(beingRWNode);

    auto relQueId = headNode->queIdx;
    TaskStub *postToShadow = new TaskStubLocalPostToShadow(currRank, relQueId, relQueId);
    auto postToShadowNode = new TaskNode(postToShadow, peerRank, relQueId, headNode->pos + 1);
    postToShadowNode->ccuBilateral = true;
    toDeleteTaskResource_.push_back(postToShadow);
    toDeleteTaskNodeResource_.push_back(postToShadowNode);

    // 重新连接
    AddNodeRelation(postToShadowNode, headChild);
    AddNodeRelation(headNode, postToShadowNode);
    AddNodeRelation(postToShadowNode, waitFromShadowNode);
    AddNodeRelation(waitFromShadowNode, beingRWNode);

    return HcclResult::HCCL_SUCCESS;
}

HcclResult GraphRevampBilateralCcu::AddBeingRWNode2CcuHead(TaskNodePtr ccuHead, TaskNodePtr currNode, TaskNodePtr &beingRWNode, RankId peerRank)
{
    TaskStub *beingRW = GenTaskStubBeingReadOrWrittern(currNode);
    if (beingRW == nullptr) {
        HCCL_ERROR("[Generate Being Read Or Written Node failed]");
        return HCCL_E_PARA;
    }

    TaskNodePtr ccuChild = nullptr;
    auto ccuNode = Hccl::GetCcuTaskHead(ccuHead);
    for (auto childIter = ccuNode->children.begin(); childIter != ccuNode->children.end(); childIter++) {
        if ((*childIter)->queIdx == ccuNode->queIdx) {
            ccuChild = *childIter;
        }
    }

    if (ccuChild == nullptr) {
        HCCL_ERROR("[AddBeingRWNode2CcuHead] Find ccu rel child is null, rank: [%u], peerRank: [%u].", currNode->rankIdx, peerRank);
        return HcclResult::HCCL_E_INTERNAL;
    }

    CHK_RET(AddVirtualFlowFirstHalfPart(ccuHead, ccuChild, currNode, beingRWNode, beingRW));
    return HcclResult::HCCL_SUCCESS;
}

HcclResult GraphRevampBilateralCcu::CreateNewVirtualStream(TaskNodePtr waitNode, TaskNodePtr currNode, TaskNodePtr &beingRWNode, RankId peerRank)
{
    RankId currRank = currNode->rankIdx;
    TaskStub *beingRW = GenTaskStubBeingReadOrWrittern(currNode);
    if (beingRW == nullptr) {
        HCCL_ERROR("[Generate Being Read Or Written Node failed]");
        return HCCL_E_PARA;
    }
    auto waitParentIter = waitNode->parents.begin();
    for (; waitParentIter != waitNode->parents.end(); waitParentIter++) {
        TaskNodePtr waitParent = *waitParentIter;
        if ((waitParent->rankIdx != peerRank) || (waitParent->task->GetType() != TaskTypeStub::POST)) {
            continue;
        }
        TaskNodePtr postRelChild = nullptr;
        for (auto postChildIter = waitParent->children.begin(); postChildIter != waitParent->children.end(); postChildIter++) {
            if ((*postChildIter)->queIdx == waitParent->queIdx && (*postChildIter)->rankIdx == waitParent->rankIdx) {
                postRelChild = *postChildIter;
            }
        }
        if (postRelChild == nullptr) {
            HCCL_ERROR("[CreateNewVirtualStream] Find post rel child is null, rank: [%u], peerRank: [%u].", currRank, peerRank);
            return HcclResult::HCCL_E_INTERNAL;
        }
        
        CHK_RET(AddVirtualFlowFirstHalfPart(waitParent, postRelChild, currNode, beingRWNode, beingRW));
    }
    return HcclResult::HCCL_SUCCESS;    
}

HcclResult GraphRevampBilateralCcu::CloseLoopNewVirtualStream(TaskNodePtr postNode, TaskNodePtr currNode, TaskNodePtr beingRWNode, RankId peerRank)
{
    RankId currRank = currNode->rankIdx;
    auto childIter = postNode->children.begin();
    for (; childIter != postNode->children.end(); childIter++) {
        TaskNodePtr postChildWait = *childIter;
        if ((postChildWait->rankIdx != peerRank) || (postChildWait->task->GetType() != TaskTypeStub::WAIT)) {
            continue;
        }
        TaskNodePtr waitRelParent = nullptr;
        for (auto waitParentIter = postChildWait->parents.begin(); waitParentIter != postChildWait->parents.end(); waitParentIter++) {
            if ((*waitParentIter)->queIdx == postChildWait->queIdx && (*waitParentIter)->rankIdx == postChildWait->rankIdx) {
                waitRelParent = *waitParentIter;
            }
        }
        if (waitRelParent == nullptr) {
            HCCL_ERROR("[CloseLoopNewVirtualStream] Find wait rel parent is null, rank: [%u], peerRank: [%u].", currRank, peerRank);
            return HcclResult::HCCL_E_INTERNAL;
        }
        // WAIT与原父节点断开连接
        RemoveNodeRelation(waitRelParent, postChildWait);

        auto virQueId = beingRWNode->queIdx;
        auto relQueId = waitRelParent->queIdx;
        TaskStub *waitFromShadow = new TaskStubLocalWaitFromShadow(currRank, relQueId, relQueId);
        auto waitFromShadowNode = new TaskNode(waitFromShadow, peerRank, relQueId, waitRelParent->pos + 1);
        waitFromShadowNode->ccuBilateral = true;
        toDeleteTaskResource_.push_back(waitFromShadow);
        toDeleteTaskNodeResource_.push_back(waitFromShadowNode);

        TaskStub *postToShadow = new TaskStubLocalPostToShadow(currRank, virQueId, virQueId);
        auto postToShadowNode = new TaskNode(postToShadow, peerRank, virQueId, beingRWNode->pos + 1);
        postToShadowNode->ccuBilateral = true;
        toDeleteTaskResource_.push_back(postToShadow);
        toDeleteTaskNodeResource_.push_back(postToShadowNode);

        // 修改wait节点pos (疑问)
        postChildWait->pos = waitFromShadowNode->pos + 1;

        // 重新连接
        AddNodeRelation(waitRelParent, waitFromShadowNode);
        AddNodeRelation(waitFromShadowNode, postChildWait);
        AddNodeRelation(postToShadowNode, waitFromShadowNode);
        AddNodeRelation(beingRWNode, postToShadowNode);
    }
    return HcclResult::HCCL_SUCCESS;    
}

} // namespace checker