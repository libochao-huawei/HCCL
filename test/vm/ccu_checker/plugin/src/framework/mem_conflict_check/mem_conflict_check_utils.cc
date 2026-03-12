/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2025-2025. All rights reserved.
 * Description: 2.0适配器对外提供的编排接口
 * Author: yinding
 * Create: 2025-06-14
 */

#include "mem_conflict_check_utils.h"
#include "task_ccu.h"
#include "log.h"

namespace HcclSim {

TaskNodePtr GetCcuTaskHead(TaskNodePtr node)
{
    TaskNode* retNode = node;
    if (node->task != nullptr && node->task->GetType() == TaskTypeStub::CCU_GRAPH) {
        // 首次进入子图
        TaskStubCcuGraph *curCcuTask = dynamic_cast<TaskStubCcuGraph *>(node->task);
        retNode = curCcuTask->ccuHeadTaskNode;
    } else if (node->task != nullptr && node->task->GetType() == TaskTypeStub::SUB_GRAPH_END) {
        // 走到子图的最后一个子节点了，就回到整图
        TaskStubSubGraphEnd *subGraphEnd = dynamic_cast<TaskStubSubGraphEnd *>(node->task);
        retNode = subGraphEnd->subGraphNode;
    }
    return retNode;
}

HcclResult CopyCcuSubGraph(TaskStub *originCcu, TaskStub **newCcu)
{
    if (originCcu->GetType() != TaskTypeStub::CCU_GRAPH) {
        HCCL_ERROR("origin node is not ccu graph");
        return HcclResult::HCCL_E_INTERNAL;
    }

    TaskStubCcuGraph *oriCcuTask = dynamic_cast<TaskStubCcuGraph *>(originCcu);
    *newCcu = new TaskStubCcuGraph(oriCcuTask);
    TaskStubCcuGraph *newCcuTask = dynamic_cast<TaskStubCcuGraph *>(*newCcu);

    std::map<TaskNodePtr, TaskNodePtr> originNode2copyNode; // 用来收录原节点到新节点的映射
    // 拷贝节点
    for (auto &oriNode : oriCcuTask->toDeleteTaskNode_) {
        auto newNode = new TaskNode(oriNode->task, oriNode->rankIdx, oriNode->queIdx, oriNode->pos);
        originNode2copyNode[oriNode] = newNode;
        newCcuTask->toDeleteTaskNode_.push_back(newNode);
    }
    // 按原节点，拷贝副本连接关系
    for (auto &oriNode : oriCcuTask->toDeleteTaskNode_) {
        for (auto &parent : oriNode->parents) {
            originNode2copyNode[oriNode]->parents.push_back(originNode2copyNode[parent]);
        }
        for (auto &child : oriNode->children) {
            originNode2copyNode[oriNode]->children.push_back(originNode2copyNode[child]);
        }
    }

    // 恢复内存冲突改造所需的成员变量
    // 拷贝loop节点 —— loop并行化改造
    for (const auto &loopGroupInfo : oriCcuTask->loopGroupInfo_) {
        std::vector<LoopInfo> loopGroup;
        for (const auto &loop : loopGroupInfo) {
            loopGroup.push_back(LoopInfo(originNode2copyNode[loop.loopStart], originNode2copyNode[loop.loopEnd]));
        }
        newCcuTask->loopGroupInfo_.push_back(loopGroup);
    }
    // 拷贝双边语义节点 —— 单边转双边改造
    for (const auto &waitInfo : oriCcuTask->waitInfoTmp_) {
        BilateralWaitInfo newWaitInfo;
        for (const auto &wait : waitInfo.waitNodes) {
            newWaitInfo.waitNodes.push_back(originNode2copyNode[wait]);
        }
        newCcuTask->waitInfoTmp_.push_back(newWaitInfo);
    }
    for (const auto &postInfo : oriCcuTask->postInfoTmp_) {
        BilateralPostInfo newPostInfo;
        for (const auto &asynNode : postInfo.asyncNodes) {
            newPostInfo.asyncNodes.push_back(originNode2copyNode[asynNode]);
        }
        newCcuTask->postInfoTmp_.push_back(newPostInfo);
    }

    // 拷贝异步节点 —— 并行化改造
    for (const auto &parNode : oriCcuTask->parallelNodes_) {
        newCcuTask->parallelNodes_.push_back(originNode2copyNode[parNode]);
    }
    return HcclResult::HCCL_SUCCESS;
}

}