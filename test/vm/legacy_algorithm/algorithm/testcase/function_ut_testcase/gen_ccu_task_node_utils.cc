/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */


#include "gen_ccu_task_node_utils.h"
#include "ccu_ins.h"
#include "instruction.h"
#include "ccu_task_common.h"

using namespace checker;

namespace hccl {

GenCcuTaskNodeGraphBase::~GenCcuTaskNodeGraphBase()
{
    for (auto& ele : toDeleteTaskNodeResource_) {
        if (ele == nullptr) {
            continue;
        }
        delete ele;
    }
}

void GenCcuTaskNodeGraphBase::PrintRankGraph(RankId rankId)
{
    std::cout<<"============================Print Rank"<<rankId<<" graph start============================"<<std::endl;
    Hccl::PrintGraphRevamp(GetCcuTaskHead(ccuNode[rankId].get()));
    std::cout<<"============================Print Rank"<<rankId<<" graph end============================"<<std::endl;
}

TaskNodePtr GenCcuTaskNodeGraphBase::GetCcuTaskHead(TaskNodePtr node)
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

void GenCcuTaskNodeGraphBase::LinkNode(TaskNodePtr parent, TaskNodePtr node)
{
    parent->children.push_back(node);
    node->parents.push_back(parent);
    return;
}

TaskNodePtr GenCcuTaskNodeGraphBase::CreateHeadNode()
{
    head = new TaskNode(nullptr, -1, 0, 0);
    toDeleteTaskNodeResource_.push_back(head);
    return head;
}

TaskNodePtr GenCcuTaskNodeGraphBase::CreateCcuHeadNode(RankId rankId, uint32_t queueId)
{
    auto taskCcuTmp = std::make_shared<TaskStubCcuGraph>(rankId);
    taskCCU.push_back(taskCcuTmp);
    auto ccuTmp = std::make_shared<TaskNode>(taskCcuTmp.get(), rankId, queueId, 0);
    ccuNode.push_back(ccuTmp);

    return ccuTmp.get();
}

TaskNodePtr GenCcuTaskNodeGraphBase::AddCcuLocalCopy(RankId rankId, uint32_t queueId, DataSlice srcSlice, DataSlice dstSlice, TaskStubCcuGraph *curCcuTask)
{
    TaskStub *localCopyTask = new TaskStubLocalCopy(srcSlice, dstSlice);
    auto localCopyNode = new TaskNode(localCopyTask, rankId, queueId, 0);
    curCcuTask->toDeleteTask_.push_back(localCopyTask);
    curCcuTask->toDeleteTaskNode_.push_back(localCopyNode);
    Hccl::AppendTailNode(curCcuTask, queueId, localCopyNode);
    return localCopyNode;
}

TaskNodePtr GenCcuTaskNodeGraphBase::AddCcuWrite(RankId rankId, RankId rmtRankId, uint32_t queueId, DataSlice srcSlice, DataSlice dstSlice, TaskStubCcuGraph *curCcuTask)
{
    LinkInfoStub link(LinkProtoStub::CCU);
    TaskStub *writeTask = new TaskStubWrite(rmtRankId, link, srcSlice, dstSlice);
    auto writeNode = new TaskNode(writeTask, rankId, queueId, 0);
    curCcuTask->toDeleteTask_.push_back(writeTask);
    curCcuTask->toDeleteTaskNode_.push_back(writeNode);
    Hccl::AppendTailNode(curCcuTask, queueId, writeNode);
    return writeNode;
}

TaskNodePtr GenCcuTaskNodeGraphBase::AddCcuRead(RankId rankId, RankId rmtRankId, uint32_t queueId, DataSlice srcSlice, DataSlice dstSlice, TaskStubCcuGraph *curCcuTask)
{
    LinkInfoStub link(LinkProtoStub::CCU);
    TaskStub *readTask = new TaskStubRead(rmtRankId, link, dstSlice, srcSlice);
    auto readNode = new TaskNode(readTask, rankId, queueId, 0);
    curCcuTask->toDeleteTask_.push_back(readTask);
    curCcuTask->toDeleteTaskNode_.push_back(readNode);
    Hccl::AppendTailNode(curCcuTask, queueId, readNode);
    return readNode;
}

TaskNodePtr GenCcuTaskNodeGraphBase::AddCcuLocalWait(RankId rankId, uint32_t queueId, uint32_t topicId, TaskStubCcuGraph *curCcuTask)
{
    TaskStub *localWaitTask = new TaskStubLocalWaitFrom(topicId, INVALID_QID, queueId);
    auto localWaitNode = new TaskNode(localWaitTask, rankId, queueId, 0);
    curCcuTask->toDeleteTask_.push_back(localWaitTask);
    curCcuTask->toDeleteTaskNode_.push_back(localWaitNode);
    Hccl::AppendTailNode(curCcuTask, queueId, localWaitNode);
    return localWaitNode;
}

TaskNodePtr GenCcuTaskNodeGraphBase::AddCcuLocalPost(RankId rankId, uint32_t queueId, uint32_t topicId, TaskStubCcuGraph *curCcuTask)
{
    TaskStub *localPostTo = new TaskStubLocalPostTo(topicId, queueId);
    auto localPostToNode = new TaskNode(localPostTo, rankId, queueId, 0);
    curCcuTask->toDeleteTask_.push_back(localPostTo);
    curCcuTask->toDeleteTaskNode_.push_back(localPostToNode);
    Hccl::AppendTailNode(curCcuTask, queueId, localPostToNode);
    return localPostToNode;
}

TaskNodePtr GenCcuTaskNodeGraphBase::AddCcuWait(RankId rankId, RankId rmtRankId, uint32_t queueId, uint32_t topicId, TaskStubCcuGraph *curCcuTask)
{
    LinkInfoStub link(LinkProtoStub::CCU);
    std::string tag = "CCU_TASK";
    TaskStub *remoteWaitTask = new TaskStubWait(rmtRankId, link, topicId, NotifyTypeStub::CCU, tag, curCcuTask);
    auto remoteWaitNode = new TaskNode(remoteWaitTask, rankId, queueId, 0);
    curCcuTask->toDeleteTask_.push_back(remoteWaitTask);
    curCcuTask->toDeleteTaskNode_.push_back(remoteWaitNode);
    Hccl::AppendTailNode(curCcuTask, queueId, remoteWaitNode);
    return remoteWaitNode;
}

TaskNodePtr GenCcuTaskNodeGraphBase::AddCcuPost(RankId rankId, RankId rmtRankId, uint32_t queueId, uint32_t topicId, TaskStubCcuGraph *curCcuTask)
{
    LinkInfoStub link(LinkProtoStub::CCU);
    std::string tag = "CCU_TASK";
    TaskStub *postTo = new TaskStubPost(rmtRankId, link, topicId, NotifyTypeStub::CCU, tag, curCcuTask);
    auto postToNode = new TaskNode(postTo, rankId, queueId, 0);
    curCcuTask->toDeleteTask_.push_back(postTo);
    curCcuTask->toDeleteTaskNode_.push_back(postToNode);
    Hccl::AppendTailNode(curCcuTask, queueId, postToNode);
    return postToNode;
}

void GenCcuTaskNodeGraphBase::CreateCcuEndNode(RankId rankId, TaskNodePtr &node, TaskStubCcuGraph *curCcuTask)
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

void GenCcuTaskNodeGraphBase::Init(TaskStubCcuGraph *curCcuTask, uint32_t rankSize, uint32_t queNum)
{
    curCcuTask->tailNodes.resize(queNum);
    curCcuTask->instrInfo.resize(queNum);
    curCcuTask->queueNum_ = queNum;
    curCcuTask->microCodePosInQue.resize(queNum);
    curCcuTask->bilateralPart1_.resize(queNum);
    curCcuTask->bilateralPart2_.resize(queNum);
    curCcuTask->bilateralNodes_.resize(queNum);
    curCcuTask->waitInfoTmp_.resize(rankSize);
    curCcuTask->postInfoTmp_.resize(rankSize);
}

}