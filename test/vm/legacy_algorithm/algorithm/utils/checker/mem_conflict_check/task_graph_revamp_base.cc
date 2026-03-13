/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2024-2024. All rights reserved.
 * Description: Graph Revamp base
 * Create: 2025-10-15
 */
#include "task_graph_revamp_base.h"
#include "log.h"
#include <algorithm>

using namespace Hccl;

namespace checker {

map<RankId, u32> GraphRevampBase::rank2QueSize_;

GraphRevampBase::~GraphRevampBase()
{
    for (auto& ele : toDeleteTaskResource_) {
        if (ele == nullptr) {
            continue;
        }
        delete ele;
    }
    for (auto& ele : toDeleteTaskNodeResource_) {
        if (ele == nullptr) {
            continue;
        }
        delete ele;
    }
    rank2QueSize_.clear();
}

HcclResult GraphRevampBase::GetPeerRankByTaskNode(TaskNodePtr currNode, RankId &peerRank)
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
    }
    return HCCL_SUCCESS;
}

HcclResult GraphRevampBase::GetLinkProtoStubByTaskNode(TaskNodePtr currNode, LinkProtoStub &link)
{
    if (currNode->task->GetType() == TaskTypeStub::READ) {
        TaskStubRead *read = dynamic_cast<TaskStubRead *>(currNode->task);
        link = read->GetLinkType();
    } else if (currNode->task->GetType() == TaskTypeStub::READ_REDUCE) {
        TaskStubReadReduce *read = dynamic_cast<TaskStubReadReduce *>(currNode->task);
        link = read->GetLinkType();
    } else if (currNode->task->GetType() == TaskTypeStub::WRITE) {
        TaskStubWrite *write = dynamic_cast<TaskStubWrite *>(currNode->task);
        link = write->GetLinkType();
    } else if (currNode->task->GetType() == TaskTypeStub::WRITE_REDUCE) {
        TaskStubWriteReduce *write = dynamic_cast<TaskStubWriteReduce *>(currNode->task);
        link = write->GetLinkType();
    }
    return HCCL_SUCCESS;
}

TaskStub* GraphRevampBase::GenTaskStubBeingReadOrWrittern(TaskNodePtr currNode)
{
    TaskStub *res = nullptr;
    if (currNode->task->GetType() == TaskTypeStub::READ) {
        TaskStubRead *read = dynamic_cast<TaskStubRead *>(currNode->task);
        res = new TaskStubBeingRead
            (currNode->rankIdx, read->GetLinkInfo(), read->GetRemoteSlice(), read->GetLocalSlice(), read->IsGenFromSync());
    } else if (currNode->task->GetType() == TaskTypeStub::WRITE) {
        TaskStubWrite *write = dynamic_cast<TaskStubWrite *>(currNode->task);
        res = new TaskStubBeingWritten
            (currNode->rankIdx, write->GetLinkInfo(), write->GetRemoteSlice(), write->GetLocalSlice(), write->IsGenFromSync());
    } else if (currNode->task->GetType() == TaskTypeStub::READ_REDUCE) {
        TaskStubReadReduce *readReduce = dynamic_cast<TaskStubReadReduce *>(currNode->task);
        res = new TaskStubBeingReadReduce
            (currNode->rankIdx, readReduce->GetLinkInfo(), readReduce->GetRemoteSlice(), readReduce->GetLocalSlice(),
            readReduce->GetDataType(), readReduce->GetReduceOp(), readReduce->IsGenFromSync());
    } else if (currNode->task->GetType() == TaskTypeStub::WRITE_REDUCE) {
        TaskStubWriteReduce *writeReduce = dynamic_cast<TaskStubWriteReduce *>(currNode->task);
        res = new TaskStubBeingWrittenReduce
            (currNode->rankIdx, writeReduce->GetLinkInfo(), writeReduce->GetRemoteSlice(), writeReduce->GetLocalSlice(),
            writeReduce->GetDataType(), writeReduce->GetReduceOp(), writeReduce->IsGenFromSync());
    }
    toDeleteTaskResource_.push_back(res);
    return res;
}

void GraphRevampBase::RemoveNodeRelation(TaskNodePtr parent, TaskNodePtr child)
{
    auto childIter = std::remove(parent->children.begin(), parent->children.end(), child);
    parent->children.erase(childIter, parent->children.end());
    auto parIter = std::remove(child->parents.begin(), child->parents.end(), parent);
    child->parents.erase(parIter, child->parents.end());
}

void GraphRevampBase::AddNodeRelation(TaskNodePtr parent, TaskNodePtr child)
{
    parent->children.push_back(child);
    child->parents.push_back(parent);
}

void GraphRevampBase::SearchGraphByRank(
    TaskNodePtr currNode, std::queue<TaskNodePtr> &graphNodeQue, std::set<TaskNodePtr> &isVisited, RankId rankId)
{
    for (auto childIter = currNode->children.begin(); childIter != currNode->children.end(); childIter++) {
        if ((*childIter)->rankIdx != rankId) {
            continue;
        }
        if (!isVisited.count(*childIter)) {
            graphNodeQue.push((*childIter));
            isVisited.insert((*childIter));
        }
    }
}

void GraphRevampBase::SearchGraphByQueueId(
    TaskNodePtr currNode, std::queue<TaskNodePtr> &graphNodeQue, std::set<TaskNodePtr> &isVisited, uint32_t queIdx)
{
    for (auto childIter = currNode->children.begin(); childIter != currNode->children.end(); childIter++) {
        if ((*childIter)->queIdx != queIdx) {
            continue;
        }
        if (!isVisited.count(*childIter)) {
            graphNodeQue.push((*childIter));
            isVisited.insert((*childIter));
        }
    }
}

HcclResult GraphRevampBase::RevampGraph4Rank(TaskNodePtr ccuHead, RankId rankId)
{
    return HcclResult::HCCL_SUCCESS;
}

HcclResult GraphRevampBase::RevampGraph(TaskNodePtr dummyStart)
{
    std::set<TaskNodePtr> isVisited;
    std::queue<TaskNodePtr> graphNodeQue;
    for (auto childIter = dummyStart->children.begin(); childIter != dummyStart->children.end(); childIter++) {
        auto rankId = (*childIter)->rankIdx;
        graphNodeQue.push(*childIter);

        while (!graphNodeQue.empty()) {
            TaskNodePtr currNode = graphNodeQue.front();
            graphNodeQue.pop();
            SearchGraphByRank(currNode, graphNodeQue, isVisited, rankId);

            if (currNode->task->GetType() == TaskTypeStub::CCU_GRAPH) {
                CHK_RET(RevampGraph4Rank(currNode, rankId));
            }
        }
    }

    HCCL_INFO("[RevampGraph] All Rank revamp success.");
    return HcclResult::HCCL_SUCCESS;
}

} // namespace checker