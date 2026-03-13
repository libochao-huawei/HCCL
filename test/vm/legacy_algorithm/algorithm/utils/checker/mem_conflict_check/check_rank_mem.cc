/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2024-2024. All rights reserved.
 * Description: 内存冲突校验类实现
 * Author: huangweihao
 * Create: 2024-11-11
 */
#include "check_rank_mem.h"

#include <queue>
#include <set>

#include "check_utils.h"
#include "log.h"
#include "analysis_result.pb.h"
#include "data_dumper.h"
#include "semantics_utils.h"
#include "aiv_task_stub.h"

#ifdef HCCL_ALG_ANALYZER_DAVID
#include "alg_adapt_v2_interface.h"
#endif

namespace checker {

MemoryStatus operator|(MemoryStatus a, MemoryStatus b)
{
    return static_cast<MemoryStatus>(static_cast<u32>(a) | static_cast<u32>(b));
}

MemoryStatus operator&(MemoryStatus a, MemoryStatus b)
{
    return static_cast<MemoryStatus>(static_cast<u32>(a) & static_cast<u32>(b));
}

MemoryStatus &operator|=(MemoryStatus &a, MemoryStatus b)
{
    return a = a | b;
}

// 边界节点，用于将一个原语队列切分为多个碎片
bool IsBoardType(TaskTypeStub type)
{
    const std::set<TaskTypeStub> boardTypes = {TaskTypeStub::LOCAL_POST_TO,
                                               TaskTypeStub::LOCAL_WAIT_FROM,
                                               TaskTypeStub::LOCAL_POST_TO_SHADOW,
                                               TaskTypeStub::LOCAL_WAIT_FROM_SHADOW,
                                               TaskTypeStub::SET_FLAG,
                                               TaskTypeStub::WAIT_FLAG,
                                               TaskTypeStub::SET_FLAG_SHADOW,
                                               TaskTypeStub::WAIT_FLAG_SHADOW,
                                               TaskTypeStub::PIPE_BARRIER,
                                               TaskTypeStub::SEND_SYNC,
                                               TaskTypeStub::RECV_SYNC,
                                               TaskTypeStub::SEND_SYNC_REDUCE};
    return boardTypes.count(type) != 0;
}

std::string GenFragQueueMemDes(FragQueueMemStatus &fragQueMemStatus)
{
    std::stringstream ret;
    for (auto iter = fragQueMemStatus.begin(); iter != fragQueMemStatus.end(); iter++) {
        BufferType type = iter->first;
        ret << FOUR_INDENT_SPACE << FOUR_INDENT_SPACE << "BufferType is " << type.Describe() << std::endl;
        for (auto &ele : iter->second) {
            ret << FOUR_INDENT_SPACE << FOUR_INDENT_SPACE << FOUR_INDENT_SPACE << ele.Describe();
        }
    }
    return ret.str();
}

bool CheckRankMem::CheckCcuInvalidNode(bool isCcuGraph, TaskNode *node)
{
    if (node == nullptr) {
        return false;
    }
    if (!isCcuGraph) {
        return false;
    }
    if (node->task->GetType() == TaskTypeStub::LOCAL_POST_TO) {
        auto locPost = dynamic_cast<TaskStubLocalPostTo*>(node->task);
        return locPost->IsInvalidPost();
    }
    if (node->parents.size() != 1) {
        return false;
    }
    return false;
}

void CheckRankMem::GenFragQueueInOneQueue(TaskNode *head, std::set<u32> &seenQueues)
{
    TaskNode *fragStart = nullptr;
    TaskNode *fragEnd = nullptr;

    std::set<TaskNode *> visitedNodes;
    visitedNodes.insert(head);
    std::queue<TaskNode *> walkQue;
    walkQue.push(head);

    // 出于灵活性考虑，一个queue的头节点不一定是Post/Wait类型
    if (IsBoardType(head->task->GetType())) {
        fragStart = head;
    }

    while (!walkQue.empty()) {
        TaskNode *curNode = walkQue.front();
        walkQue.pop();
        for (auto &child : curNode->children) {
            // 不是同一个rank上的不考虑
            if (child->rankIdx != head->rankIdx) {
                continue;
            }

            if ((child->queIdx != head->queIdx) && (seenQueues.count(child->queIdx) == 0)) {
                bool isHeadNode = true;
                for (int i = 0; i < child->parents.size(); i++) {
                    if (child->parents[i]->queIdx == child->queIdx) {
                        isHeadNode = false;
                        break;
                    }
                }
                if (isHeadNode) {
                    seenQueues.insert(child->queIdx);
                    GenFragQueueInOneQueue(child, seenQueues);
                }
            }

            if (child->queIdx != head->queIdx) {
                continue;
            }

            if (visitedNodes.count(child) == 0) {
                walkQue.push(child);
                visitedNodes.insert(child);
            }
        }

        if (curNode == fragStart) {
            continue;
        }

        // 1）遇到边界节点 2）待循环队列已经为空
        if (IsBoardType(curNode->task->GetType()) || walkQue.empty()) {
            if (IsBoardType(curNode->task->GetType())) {
                fragEnd = curNode;
            }
            FragmentQueue ele{head->queIdx, 0, 0, false, fragStart, fragEnd};
            rank2FragQueue_[head->rankIdx].insert(ele);
            fragStart = curNode;
            fragEnd   = nullptr;
        }
    }
    return;
}

void CheckRankMem::GenFragQueueInOneRank(TaskNode *node)
{
    u32 queueId = node->queIdx;
    // 头结点链接的都应该是主流，queIdx=0
    if (queueId != 0) {
        HCCL_ERROR("The node connecting the head node should be the mainstream.");
    }
    std::set<u32> seenQueues;
    seenQueues.insert(queueId);
    GenFragQueueInOneQueue(node, seenQueues);
    return;
}

void CheckRankMem::GenFragQueue()
{
    // 头节点的每个child应该代表了一个rank
    for (auto &child : graphHead_->children) {
        GenFragQueueInOneRank(child);
    }
    return;
}

TaskNode* CheckRankMem::GetPipeBarrierChildNode(TaskNode *pipeeBarrier, s32 pipeIdx)
{
    // 优先找同流水上的child节点(非PipeBarrier(all))
    for (auto &child : pipeeBarrier->children) {
        if (child->pipeIdx == pipeIdx && GetNodeType(child) != TaskTypeStub::PIPE_BARRIER) {
            return child;
        }
    }

    // 若没有同流水上的child节点，在child里找有没有直连的PipeBarrier(all)
    for (auto &child : pipeeBarrier->children) {
        if (GetNodeType(child) == TaskTypeStub::PIPE_BARRIER &&
        ((TaskStubPipeBarrier*)child->task)->IsPipeBarrierAll()) {
            return child;
        }
    }

    return nullptr;
}

void CheckRankMem::GenAivFragQueueInOnePipe(TaskNode *head)
{
    TaskNode *fragStart = nullptr;
    TaskNode *fragEnd = nullptr;

    std::set<TaskNode *> visitedNodes;
    visitedNodes.insert(head);
    std::queue<TaskNode *> walkQue;
    walkQue.push(head);

    // 出于灵活性考虑，一个queue的头节点不一定是Post/Wait类型
    if (IsBoardType(GetNodeType(head))) {
        fragStart = head;
    }

    while (!walkQue.empty()) {
        TaskNode *curNode = walkQue.front();
        walkQue.pop();
        
        if (GetNodeType(curNode) == TaskTypeStub::PIPE_BARRIER && ((TaskStubPipeBarrier*)curNode->task)->IsPipeBarrierAll()) {
            TaskNode *child = GetPipeBarrierChildNode(curNode, head->pipeIdx);
            if (child != nullptr) {
                walkQue.push(child);
                visitedNodes.insert(child);
            }
        } else {
            for (auto &child : curNode->children) {
                // 跳过AivEnd节点
                if (GetNodeType(child) == TaskTypeStub::AIV_END) {
                    continue;
                }
                // PipeBarrier(ALL)可能跨流水特殊处理，不跳过，其他节点不是同一个rank上、跨block、跨pipe的不考虑
                if (GetNodeType(child) == TaskTypeStub::PIPE_BARRIER && ((TaskStubPipeBarrier*)child->task)->IsPipeBarrierAll()) {
                    ;  //不处理
                } else if (child->rankIdx != head->rankIdx || child->blockIdx != head->blockIdx || child->pipeIdx != head->pipeIdx) {
                    continue;
                }

                if (visitedNodes.count(child) == 0) {
                    walkQue.push(child);
                    // PipeBarrier(ALL)特殊处理，不能添加到visitedNodes
                    if (GetNodeType(child) == TaskTypeStub::PIPE_BARRIER) {
                        if (!((TaskStubPipeBarrier*)child->task)->IsPipeBarrierAll()) {
                            visitedNodes.insert(child);
                        }
                    } else {
                        visitedNodes.insert(child);
                    }
                }
            }
        }

        if (curNode == fragStart) {
            continue;
        }

        // 1）遇到边界节点 2）待循环队列已经为空
        if ((IsBoardType(curNode->task->GetType())) || walkQue.empty()) {
            if (IsBoardType(curNode->task->GetType())) {
                fragEnd = curNode;
            }
            FragmentQueue ele{0, head->blockIdx, head->pipeIdx, true, fragStart, fragEnd};
            rank2FragQueue_[head->rankIdx].insert(ele);

            fragStart = curNode;
            fragEnd   = nullptr;
        }
    }
    return;
}

void CheckRankMem::GenAivFragQueue(TaskNode* aivStart)
{
    for (auto &child : aivStart->children) {
        if (GetNodeType(child) == TaskTypeStub::BLOCK_START) {
            for (auto &childpipe : child->children) {
                GenAivFragQueueInOnePipe(childpipe);
            }
        } else if (GetNodeType(child) == TaskTypeStub::VIRTUAL_RANK_START) {
            GenAivFragQueueInOnePipe(child);
        }
    }
    return;
}

void CheckRankMem::FindPostWaitNode(TaskNode *node, std::set<TaskNode *> &postNodes, std::set<TaskNode *> &waitNodes) const
{
    if (node == nullptr) {
        return;
    }
    switch (GetNodeType(node)) {
        case TaskTypeStub::LOCAL_POST_TO:
        case TaskTypeStub::LOCAL_POST_TO_SHADOW:
            postNodes.insert(node);
            break;
        case TaskTypeStub::LOCAL_WAIT_FROM:
        case TaskTypeStub::LOCAL_WAIT_FROM_SHADOW:
            waitNodes.insert(node);
            break;
        default:
            return;
    }
    return;
}

HcclResult CheckRankMem::FindPostWaitPair(RankId rankId, bool isCcuGraph)
{
    std::set<TaskNode *> postNodes;
    std::set<TaskNode *> waitNodes;
    for (auto &ele : rank2FragQueue_[rankId]) {
        if (!CheckCcuInvalidNode(isCcuGraph, ele.head)) {
            FindPostWaitNode(ele.head, postNodes, waitNodes);
        }
        if (!CheckCcuInvalidNode(isCcuGraph, ele.tail)) {
            FindPostWaitNode(ele.tail, postNodes, waitNodes);
        }
    }

    for (auto &post : postNodes) {
        TaskNode *wait = nullptr;
        for (auto &child : post->children) {
            if (child->queIdx == post->queIdx) {
                continue;
            }
            if (waitNodes.count(child) == 1) {
                wait = child;
            }
        }
        if (wait == nullptr) {
            HCCL_ERROR("Can not find corresponding wait node for post node:%p", post);
            return HcclResult::HCCL_E_PARA;
        }

        rank2PostWaitPairs_[rankId][post] = wait;
        rank2PostWaitPairs_[rankId][wait] = post;
    }
    return HcclResult::HCCL_SUCCESS;
}

void CheckRankMem::FindAivPostWaitNode(TaskNode *node, std::set<TaskNode *> &postNodes, std::set<TaskNode *> &waitNodes) const
{
    if (node == nullptr) {
        return;
    }

    switch (GetNodeType(node)) {
        case TaskTypeStub::SET_FLAG:
        case TaskTypeStub::SET_FLAG_SHADOW:
        case TaskTypeStub::SEND_SYNC:
        case TaskTypeStub::SEND_SYNC_REDUCE:
            postNodes.insert(node);
            break;
        case TaskTypeStub::WAIT_FLAG:
        case TaskTypeStub::WAIT_FLAG_SHADOW:
        case TaskTypeStub::RECV_SYNC:
            waitNodes.insert(node);
            break;
        default:
            return;
    }
    return;
}

HcclResult CheckRankMem::FindAivSyncPair(RankId rankId)
{
    std::set<TaskNode *> postNodes;
    std::set<TaskNode *> waitNodes;

    for (auto &ele : rank2FragQueue_[rankId]) {
        FindAivPostWaitNode(ele.head, postNodes, waitNodes);
        FindAivPostWaitNode(ele.tail, postNodes, waitNodes);
    }

    for (auto &post : postNodes) {
        TaskTypeStub nodeType = GetNodeType(post);
        TaskNode *wait = nullptr;
        for (auto &child : post->children) {
            if (child->blockIdx == post->blockIdx && child->pipeIdx == post->pipeIdx) {
                continue;
            }

            if (waitNodes.count(child) == 1) {
                wait = child;
                if (nodeType == TaskTypeStub::SEND_SYNC || nodeType == TaskTypeStub::SEND_SYNC_REDUCE) {
                    rank2AivSendRecvPairs_[rankId][post].insert(wait);
                    rank2AivSendRecvPairs_[rankId][wait].insert(post);
                } else {
                    rank2AivPostWaitPairs_[rankId][post] = wait;
                    rank2AivPostWaitPairs_[rankId][wait] = post;
                }
            }   
            
        }
        if (wait == nullptr) {
            continue;
            // HCCL_ERROR("node[%d, %d, %d, %d] Can not find corresponding WaitFlag node for SetFlag node, or RecvSync node for SendSync node", post->rankIdx, post->blockIdx, post->pipeIdx, post->pipePos);
            // return HcclResult::HCCL_E_PARA;
        }

        
    }

    return HcclResult::HCCL_SUCCESS;
}

void CheckRankMem::ProcessEqualToTargetStartAddr(u64 &sliceStartAddr, u64 sliceEndAddr,
                                                 std::vector<SliceMemoryStatus> &addedEles, MemoryStatus sliceStatus,
                                                 std::set<SliceMemoryStatus>::iterator target) const
{
    u64 eleEndAddr = target->startAddr + target->size;
    // 已经打过相同的标记位，不需要重复打
    if ((static_cast<u32>(target->status) & static_cast<u32>(sliceStatus)) != 0) {
        sliceStartAddr = eleEndAddr;
        return;
    }

    if (sliceEndAddr < eleEndAddr) {
        SliceMemoryStatus sliceMemStatus{sliceEndAddr, eleEndAddr - sliceEndAddr, target->status};
        addedEles.push_back(sliceMemStatus);
        target->size = sliceEndAddr - target->startAddr;
        target->status |= sliceStatus;
        sliceStartAddr = sliceEndAddr;
    } else if (sliceEndAddr == eleEndAddr) {
        target->status |= sliceStatus;
        sliceStartAddr = sliceEndAddr;
    } else { // sliceEndAddr > eleEndAddr
        target->status |= sliceStatus;
        sliceStartAddr = eleEndAddr;
    }
}

void CheckRankMem::ProcessGreatThanTargetStartAddr(u64 &sliceStartAddr, u64 sliceEndAddr,
                                                   std::vector<SliceMemoryStatus> &addedEles, MemoryStatus sliceStatus,
                                                   std::set<SliceMemoryStatus>::iterator target) const
{
    u64 eleEndAddr = target->startAddr + target->size;
    // 已经打过相同的标记位，不需要重复打
    if ((static_cast<u32>(target->status) & static_cast<u32>(sliceStatus)) != 0) {
        sliceStartAddr = eleEndAddr;
        return;
    }

    if (sliceEndAddr < eleEndAddr) {
        SliceMemoryStatus sliceMemStatus{sliceStartAddr, sliceEndAddr - sliceStartAddr, target->status | sliceStatus};
        addedEles.push_back(sliceMemStatus);

        SliceMemoryStatus tmp{sliceEndAddr, eleEndAddr - sliceEndAddr, target->status};
        addedEles.push_back(tmp);

        target->size   = sliceStartAddr - target->startAddr;
        sliceStartAddr = sliceEndAddr;
    } else if (sliceEndAddr == eleEndAddr) {
        SliceMemoryStatus sliceMemStatus{sliceStartAddr, sliceEndAddr - sliceStartAddr, target->status | sliceStatus};
        addedEles.push_back(sliceMemStatus);

        target->size   = sliceStartAddr - target->startAddr;
        sliceStartAddr = sliceEndAddr;
    } else { // sliceEndAddr > eleEndAddr
        SliceMemoryStatus sliceMemStatus{sliceStartAddr, eleEndAddr - sliceStartAddr, target->status | sliceStatus};
        addedEles.push_back(sliceMemStatus);

        target->size   = sliceStartAddr - target->startAddr;
        sliceStartAddr = eleEndAddr;
    }
}

void CheckRankMem::GenSliceMemoryInfo(DataSlice &slice, MemoryStatus sliceStatus, FragQueueMemStatus &result)
{
    BufferType sliceBufferType = slice.GetType();
    u64        sliceStartAddr  = slice.GetOffset(); // offset
    u64        sliceEndAddr    = sliceStartAddr + slice.GetSize();

    std::vector<SliceMemoryStatus> addedEles;
    for (auto ele = result[sliceBufferType].begin(); ele != result[sliceBufferType].end(); ele++) {
        u64 eleEndAddr = ele->startAddr + ele->size;
        // 下面两个判断保证了slice和ele有交集部分
        if ((sliceStartAddr >= eleEndAddr) || (sliceEndAddr <= ele->startAddr)) {
            continue;
        }

        if (sliceStartAddr < ele->startAddr) {
            SliceMemoryStatus sliceMemStatus{sliceStartAddr, ele->startAddr - sliceStartAddr, sliceStatus};
            addedEles.push_back(sliceMemStatus);
            sliceStartAddr = ele->startAddr;
        } else if (sliceStartAddr == ele->startAddr) {
            ProcessEqualToTargetStartAddr(sliceStartAddr, sliceEndAddr, addedEles, sliceStatus, ele);
        } else { // sliceStartAddr > ele->startAddr
            ProcessGreatThanTargetStartAddr(sliceStartAddr, sliceEndAddr, addedEles, sliceStatus, ele);
        }
    }

    if (sliceEndAddr > sliceStartAddr) {
        SliceMemoryStatus sliceMemStatus{sliceStartAddr, sliceEndAddr - sliceStartAddr, sliceStatus};
        addedEles.push_back(sliceMemStatus);
    }

    // 将addedElem给刷新上去
    for (auto &ele : addedEles) {
        result[sliceBufferType].insert(ele);
    }
    return;
}

HcclResult CheckRankMem::GenPrimNodeMemoryInfo(TaskNode *node, FragQueueMemStatus &result)
{
    std::vector<DataSlice> readSlices;
    std::vector<DataSlice> writeSlices;
    GetReadSlice(node, readSlices);
    GetWriteSlice(node, writeSlices);

    for (auto &ele : readSlices) {
        GenSliceMemoryInfo(ele, MemoryStatus::READ, result);
    }

    for (auto &ele : writeSlices) {
        GenSliceMemoryInfo(ele, MemoryStatus::WRITE, result);
    }

    return HcclResult::HCCL_SUCCESS;
}

HcclResult CheckRankMem::GenAivTaskNodeMemoryInfo(TaskNode *node, FragQueueMemStatus &result)
{
    std::set<TaskNode *> visitedNodes;
    std::queue<TaskNode *> walkQue;

    TaskNode* aivStart = ((AivTaskStub*)(node->task))->GetAivStart();
    visitedNodes.insert(aivStart);
    walkQue.push(aivStart);

    while (!walkQue.empty()) {
        TaskNode *curNode = walkQue.front();
        walkQue.pop();

        for (auto &child : curNode->children) {
            if (visitedNodes.count(child) == 0) {
                walkQue.push(child);
                visitedNodes.insert(child);
            }
        }

        // 跳过AivStart和BlockStart
        if (GetNodeType(curNode) != TaskTypeStub::AIV_START && GetNodeType(curNode) != TaskTypeStub::BLOCK_START) {
            GenPrimNodeMemoryInfo(curNode, result);
        }
    }

    return HcclResult::HCCL_SUCCESS;
}

HcclResult CheckRankMem::GenFragQueueMemoryInfo(FragmentQueue &fragQueue, FragQueueMemStatus &result)
{
    std::queue<TaskNode *> walkQueue;
    walkQueue.push(fragQueue.head);

    std::set<TaskNode *> visitedNodes;
    visitedNodes.insert(fragQueue.head);

    while (!walkQueue.empty()) {
        TaskNode *curNode = walkQueue.front();
        walkQueue.pop();

        // 只有主流最前面的原语碎片才会出现头节点为空的情况，主流最前面的原语碎片不会和其他的原语碎片冲突，不生成内存信息也没关系
        if (curNode == nullptr) {
            continue;
        }

        for (auto &child : curNode->children) {
            if (GetNodeType(curNode) == TaskTypeStub::PIPE_BARRIER && ((TaskStubPipeBarrier*)curNode->task)->IsPipeBarrierAll()) {
                TaskNode *child = GetPipeBarrierChildNode(curNode, fragQueue.pipeIdx);
                if (child != nullptr) {
                    walkQueue.push(child);
                    visitedNodes.insert(child);
                }
            } else {
                if (curNode->isAivNode) {
                    if (child->rankIdx != curNode->rankIdx or child->blockIdx != curNode->blockIdx or child->pipeIdx != curNode->pipeIdx) {
                        continue;
                    }
                }
                else {
                    if (child->rankIdx != curNode->rankIdx or child->queIdx != curNode->queIdx) {
                        continue;
                    }
                }

                if (visitedNodes.count(child) == 0) {
                    walkQueue.push(child);
                    visitedNodes.insert(child);
                }
            }
        }

        if (GetNodeType(curNode) == TaskTypeStub::AIV_TASK) {
            GenAivTaskNodeMemoryInfo(curNode, result);
        } else {
            GenPrimNodeMemoryInfo(curNode, result);
        }

        if (curNode == fragQueue.tail) {
            break;
        }
    }
    return HcclResult::HCCL_SUCCESS;
}

HcclResult CheckRankMem::CompareBufferTypeMemoryInfo(std::set<SliceMemoryStatus> &left,
                                                     std::set<SliceMemoryStatus> &right)
{
    std::set<SliceMemoryStatus>::iterator leftIter  = left.begin();
    std::set<SliceMemoryStatus>::iterator rightIter = right.begin();

    while (leftIter != left.end() && rightIter != right.end()) {
        if (leftIter->size == 0) {
            leftIter++;
            continue;
        }

        if (rightIter->size == 0) {
            rightIter++;
            continue;
        }

        if (leftIter->startAddr + leftIter->size <= rightIter->startAddr) {
            leftIter++;
            continue;
        }

        if (rightIter->startAddr + rightIter->size <= leftIter->startAddr) {
            rightIter++;
            continue;
        }

        if ((leftIter->status & MemoryStatus::WRITE) == MemoryStatus::WRITE
            or (rightIter->status & MemoryStatus::WRITE) == MemoryStatus::WRITE) {
            DUMP_AND_ERROR("there is memory use confilict in two SliceMemoryStatus");
            DUMP_AND_ERROR("one is %s", leftIter->Describe().c_str());
            DUMP_AND_ERROR("another is %s", rightIter->Describe().c_str());
            return HcclResult::HCCL_E_MEMORY;
        }

        if (leftIter->startAddr + leftIter->size <= rightIter->startAddr + rightIter->size) {
            leftIter++;
        } else {
            rightIter++;
        }
    }

    return HcclResult::HCCL_SUCCESS;
}

// 仅内部dump使用，不需要对外提供
HcclResult CheckRankMem::CompareBufferTypeMemoryInfo(std::set<SliceMemoryStatus> &left,
                                                     std::set<SliceMemoryStatus> &right,
                                                     SliceMemoryStatus &conflictEleA,
                                                     SliceMemoryStatus &conflictEleB)
{
    std::set<SliceMemoryStatus>::iterator leftIter  = left.begin();
    std::set<SliceMemoryStatus>::iterator rightIter = right.begin();

    while (leftIter != left.end() && rightIter != right.end()) {
        if (leftIter->size == 0) {
            leftIter++;
            continue;
        }

        if (rightIter->size == 0) {
            rightIter++;
            continue;
        }

        if (leftIter->startAddr + leftIter->size <= rightIter->startAddr) {
            leftIter++;
            continue;
        }

        if (rightIter->startAddr + rightIter->size <= leftIter->startAddr) {
            rightIter++;
            continue;
        }

        if ((leftIter->status & MemoryStatus::WRITE) == MemoryStatus::WRITE
            or (rightIter->status & MemoryStatus::WRITE) == MemoryStatus::WRITE) {
            conflictEleA = *leftIter;
            conflictEleB = *rightIter;
            return HcclResult::HCCL_E_MEMORY;
        }

        if (leftIter->startAddr + leftIter->size <= rightIter->startAddr + rightIter->size) {
            leftIter++;
        } else {
            rightIter++;
        }
    }

    return HcclResult::HCCL_SUCCESS;
}

HcclResult CheckRankMem::CompareSliceMemoryInfo(FragQueueMemStatus &left, FragQueueMemStatus &right)
{
    for (auto iter = left.begin(); iter != left.end(); iter++) {
        BufferType type = iter->first;
        if (right.count(type) != 0) {
            auto ret = CompareBufferTypeMemoryInfo(iter->second, right[type]);
            if (ret != HcclResult::HCCL_SUCCESS) {
                HCCL_ERROR("failed to check memory %s", type.Describe().c_str());
                return ret;
            }
        }
    }
    return HcclResult::HCCL_SUCCESS;
}

std::string GenConflictDetailInfo(TaskNode *node)
{
    if (node->realPeerNode) {
        return node->realPeerNode->GenPosInfo();
    }
    return node->GenPosInfo();
}

HcclResult CheckRankMem::CompareFragQueStatus(u32 fragQueueSize, std::vector<FragmentQueue> &index2FragQueue,
                                              std::vector<std::vector<bool>> &fragQueueMatrix)
{
    std::map<FragmentQueue, FragQueueMemStatus> fragQueue2MemStatus;
    for (u32 i = 0; i < fragQueueSize; i++) {
        for (u32 j = i + 1; j < fragQueueSize; j++) {
            if (fragQueueMatrix[i][j] == false) {
                continue;
            }
            auto getOrGenMemInfo = [&](const FragmentQueue& key) -> FragQueueMemStatus& {
                auto it = fragQueue2MemStatus.find(key);
                if (it == fragQueue2MemStatus.end()) {
                    // 使用 emplace 并获取迭代器
                    auto result = fragQueue2MemStatus.emplace(
                        std::piecewise_construct,
                        std::forward_as_tuple(key),
                        std::forward_as_tuple()
                    );
                    
                    // 初始化新元素
                    HcclResult ret = GenFragQueueMemoryInfo(const_cast<FragmentQueue&>(key), result.first->second);
                    if (ret != HCCL_SUCCESS) {
                        fragQueue2MemStatus.erase(result.first);
                        throw std::runtime_error("GenFragQueueMemoryInfo failed");
                    }
                    
                    it = result.first;
                }
                return it->second;
            };

            auto& memStatei = getOrGenMemInfo(index2FragQueue[i]);
            auto& memStatej = getOrGenMemInfo(index2FragQueue[j]);

            HcclResult ret;
            ret = CompareSliceMemoryInfo(memStatei, memStatej);

            if (ret != HcclResult::HCCL_SUCCESS) {
                for (TaskNode *nodeA = index2FragQueue[i].head; nodeA != index2FragQueue[i].tail;) {
                    for (TaskNode *nodeB = index2FragQueue[j].head; nodeB != index2FragQueue[j].tail;) {
                        // 判断是否有冲突，如果有，就dump数据
                        SliceMemoryStatus conflictEleA;
                        SliceMemoryStatus conflictEleB;
                        if (IsConfilictBetweenTwoNodes(nodeA, nodeB, conflictEleA, conflictEleB)) {
                            DUMP_AND_ERROR("memory conflict between node[%p] %s and node[%p] %s",
                                nodeA, nodeA->GenPosInfo().c_str(), nodeB, nodeB->GenPosInfo().c_str());
                            DataDumper::Global()->DumpMemConflictInfo(nodeA, nodeB, conflictEleA, conflictEleB);
                            break;
                        }
                        auto nodeBOld = nodeB;
                        for (auto &child : nodeB->children) {
                            if (nodeB->isAivNode) {
                                if (GetNodeType(nodeB) == TaskTypeStub::PIPE_BARRIER && ((TaskStubPipeBarrier*)nodeB->task)->IsPipeBarrierAll()) {
                                    TaskNode *child = GetPipeBarrierChildNode(nodeB, index2FragQueue[j].pipeIdx);
                                    if (child != nullptr) {
                                        nodeB = child;
                                        break;
                                    }
                                } else {
                                    if (child->rankIdx != nodeB->rankIdx || child->blockIdx != nodeB->blockIdx || child->pipeIdx != nodeB->pipeIdx) {
                                        if (GetNodeType(child) == TaskTypeStub::PIPE_BARRIER && ((TaskStubPipeBarrier*)child->task)->IsPipeBarrierAll()) {
                                            nodeB = child;
                                            break;
                                        }
                                        continue;
                                    }
                                    nodeB = child;
                                    break;
                                }
                            } else {
                                if (child->rankIdx != nodeB->rankIdx || child->queIdx != nodeB->queIdx) {
                                    continue;
                                }
                                nodeB = child;
                                break;
                            }
                        }

                        if (nodeB->children.size() == 0 || nodeB == nodeBOld) {
                            break;
                        }
                    }

                    auto nodeAOld = nodeA;
                    for (auto &child : nodeA->children) {
                        if (nodeA->isAivNode) {
                            if (GetNodeType(nodeA) == TaskTypeStub::PIPE_BARRIER && ((TaskStubPipeBarrier*)nodeA->task)->IsPipeBarrierAll()) {
                                TaskNode *child = GetPipeBarrierChildNode(nodeA, index2FragQueue[i].pipeIdx);
                                if (child != nullptr) {
                                    nodeA = child;
                                    break;
                                }
                            } else {
                                if (child->rankIdx != nodeA->rankIdx || child->blockIdx != nodeA->blockIdx || child->pipeIdx != nodeA->pipeIdx) {
                                    if (GetNodeType(child) == TaskTypeStub::PIPE_BARRIER && ((TaskStubPipeBarrier*)child->task)->IsPipeBarrierAll()) {
                                        nodeA = child;
                                        break;
                                    }
                                    continue;
                                }
                                nodeA = child;
                                break;
                            }
                        } else {
                            if (child->rankIdx != nodeA->rankIdx || child->queIdx != nodeA->queIdx) {
                                continue;
                            }
                            nodeA = child;
                            break;
                        }
                    }

                    if (nodeA->children.size() == 0 || nodeA == nodeAOld) {
                        break;
                    }
                }
                return ret;
            }
        }
    }

    return HcclResult::HCCL_SUCCESS;
}

bool CheckRankMem::IsConfilictBetweenTwoNodes(TaskNode* nodeA, TaskNode* nodeB,
                                              SliceMemoryStatus &conflictEleA, SliceMemoryStatus &conflictEleB)
{
    FragQueueMemStatus resultA;
    FragQueueMemStatus resultB;
    if (GetNodeType(nodeA) == TaskTypeStub::AIV_TASK) {
        GenAivTaskNodeMemoryInfo(nodeA, resultA);
    } else {
        GenPrimNodeMemoryInfo(nodeA, resultA);
    }

    if (GetNodeType(nodeB) == TaskTypeStub::AIV_TASK) {
        GenAivTaskNodeMemoryInfo(nodeB, resultB);
    } else {
        GenPrimNodeMemoryInfo(nodeB, resultB);
    }

    HcclResult ret;
    for (auto iter = resultA.begin(); iter != resultA.end(); iter++) {
        BufferType type = iter->first;
        if (resultB.count(type) != 0) {
            ret = CompareBufferTypeMemoryInfo(iter->second, resultB[type], conflictEleA, conflictEleB);
            if (ret != HcclResult::HCCL_SUCCESS) {
                break;
            }
        }
    }

    if (ret != HcclResult::HCCL_SUCCESS) {
        return true;
    }
    return false;
}

// 判断是否是连接AivEnd的最后一个TaskNode
bool CheckRankMem::IsLastTaskNode(TaskNode* node)
{
    for (auto &child : node->children) {
        if (GetNodeType(child) == TaskTypeStub::AIV_END) {
            return true;
        }
    }
    return false;
}

HcclResult CheckRankMem::GenFragQueConcurrencyMatrixAndCompare(RankId rankId)
{
    u32 fragQueueSize = rank2FragQueue_[rankId].size();
    std::vector<std::vector<bool>> fragQueueMatrix(fragQueueSize, std::vector<bool>(fragQueueSize, true));
    // 自己不能与自己比较
    for (u32 i = 0; i < fragQueueSize; i++) {
        fragQueueMatrix[i][i] = false;
    }

    std::vector<FragmentQueue> index2FragQueue(fragQueueSize);
    std::map<TaskNode*, FragmentQueue> headNode2FragQueue;
    std::map<TaskNode*, u32> headNode2Index;
    u32 index = 0;
    for (auto &fragQueue : rank2FragQueue_[rankId]) {
        index2FragQueue[index] = fragQueue;
        headNode2FragQueue[fragQueue.head] = fragQueue;
        headNode2Index[fragQueue.head] = index;
        index++;
    }

    clock_t ccuStart = clock();
    for (int i = 0; i < fragQueueSize; i++) {
        TaskNode* curTailNode = index2FragQueue[i].tail;
        if (curTailNode == nullptr) {
            continue;
        }
        
        std::queue<TaskNode*> walkQue;
        std::set<TaskNode*> visitedNodes;
        walkQue.push(curTailNode);
        while(!walkQue.empty()) {
            TaskNode* curNode = walkQue.front();
            walkQue.pop();
            if (curNode == nullptr) {
                continue;
            }
            auto type = GetNodeType(curNode);
            auto nodeIdx = headNode2Index[curNode];
            TaskNode* nodeTail = nullptr;
            auto node2Frag = headNode2FragQueue.find(curNode);
            if (node2Frag != headNode2FragQueue.end()) {
                nodeTail = node2Frag->second.tail;
            }
            if (type == TaskTypeStub::LOCAL_POST_TO || type == TaskTypeStub::LOCAL_POST_TO_SHADOW ||
                type == TaskTypeStub::SET_FLAG || type == TaskTypeStub::SET_FLAG_SHADOW ||
                type == TaskTypeStub::SEND_SYNC || type == TaskTypeStub::SEND_SYNC_REDUCE) {
                // 找到以该post节点为起点的碎片队列，并打上不可能并行的标签
                // 将以该post节点为起点的碎片队列的结束点加进walkQue队列中
                // 将该post节点对应的wait节点加进walkQue队列中
                if (!curNode->isAivNode) {
                    fragQueueMatrix[i][nodeIdx] = false;
                    fragQueueMatrix[nodeIdx][i] = false;
                } else {
                    // 对于AIV，如果是连接AivEnd的最后一个TaskNode，headNode2Index找不到对应的片段，不应该置false
                    if (IsLastTaskNode(curNode) == false) {
                        fragQueueMatrix[i][nodeIdx] = false;
                        fragQueueMatrix[nodeIdx][i] = false;
                    }
                }

                if (nodeTail != nullptr && !visitedNodes.count(nodeTail)) {
                    walkQue.push(nodeTail);
                    visitedNodes.insert(nodeTail);
                }

                if (curNode->isAivNode) {
                    if (type == TaskTypeStub::SEND_SYNC || type == TaskTypeStub::SEND_SYNC_REDUCE) {
                        for (auto recvNode : rank2AivSendRecvPairs_[rankId][curNode]) {
                            if (!visitedNodes.count(recvNode)) {
                                walkQue.push(recvNode);
                                visitedNodes.insert(recvNode);
                            }
                        }
                    } else {
                        if (!visitedNodes.count(rank2AivPostWaitPairs_[rankId][curNode])) {
                            walkQue.push(rank2AivPostWaitPairs_[rankId][curNode]);
                            visitedNodes.insert(rank2AivPostWaitPairs_[rankId][curNode]);
                        }
                    }
                } else {
                    if (!visitedNodes.count(rank2PostWaitPairs_[rankId][curNode])) {
                        walkQue.push(rank2PostWaitPairs_[rankId][curNode]);
                        visitedNodes.insert(rank2PostWaitPairs_[rankId][curNode]);
                    }
                }

            } else if (type == TaskTypeStub::LOCAL_WAIT_FROM || type == TaskTypeStub::LOCAL_WAIT_FROM_SHADOW ||
                       type == TaskTypeStub::WAIT_FLAG || type == TaskTypeStub::WAIT_FLAG_SHADOW ||
                       type == TaskTypeStub::RECV_SYNC) {
                // 找到以该wait节点为起点的碎片队列，并打上不可能并行的标签
                // 将该wait节点加进walkQue队列中
                fragQueueMatrix[i][nodeIdx] = false;
                fragQueueMatrix[nodeIdx][i] = false;

                if (nodeTail != nullptr && !visitedNodes.count(nodeTail)) {
                    walkQue.push(nodeTail);
                    visitedNodes.insert(nodeTail);
                }
            } else if (type == TaskTypeStub::PIPE_BARRIER) {
                for (int j = 0; j < fragQueueSize; j++) {
                    if (index2FragQueue[j].head == curNode) {
                        fragQueueMatrix[j][i] = false;
                        fragQueueMatrix[i][j] = false;

                        walkQue.push(index2FragQueue[j].tail);
                        visitedNodes.insert(index2FragQueue[j].tail);
                    }
                }
            }
        }
    }
    clock_t ccuEnd1 = clock();

    auto ret = CompareFragQueStatus(fragQueueSize, index2FragQueue, fragQueueMatrix);

    clock_t ccuEnd2 = clock();

    double time1 = double(ccuEnd1 - ccuStart) / CLOCKS_PER_SEC;
    double time2 = double(ccuEnd2 - ccuEnd1) / CLOCKS_PER_SEC;
    if (time1 >= 2 || time2 >= 2) {
        std::cout<<"CCU revamp cost time: find impossiable= "<<time1<<", compare= "<<time2<<std::endl;
    }

    return ret;
}

// 被读的内存块
void CheckRankMem::GetReadSlice(TaskNode *node, std::vector<DataSlice> &slices)
{
    TaskTypeStub type = node->task->GetType();
    bool isGenFromSync = IsGenFromSync(node->task);
    if (type == TaskTypeStub::LOCAL_COPY) {
        auto task = dynamic_cast<TaskStubLocalCopy *>(node->task);
        if (task->GetSrcSlice().GetType() == BufferType::OUTPUT_AIV && isGenFromSync) {
            return;
        }
        slices.push_back(task->GetSrcSlice());
    } else if (type == TaskTypeStub::LOCAL_REDUCE) {
        auto task = dynamic_cast<TaskStubLocalReduce *>(node->task);
        slices.push_back(task->GetSrcSlice());
    } else if (type == TaskTypeStub::BEING_READ && !isGenFromSync) {
        auto task = dynamic_cast<TaskStubBeingRead *>(node->task);
        slices.push_back(task->GetLocalSlice());
    } else if (type == TaskTypeStub::WRITE) {
        auto task = dynamic_cast<TaskStubWrite *>(node->task);
        slices.push_back(task->GetLocalSlice());
    } else if (type == TaskTypeStub::BEING_READ_REDUCE && !isGenFromSync) {
        auto task = dynamic_cast<TaskStubBeingReadReduce *>(node->task);
        slices.push_back(task->GetLocalSlice());
    } else if (type == TaskTypeStub::WRITE_REDUCE) {
        auto task = dynamic_cast<TaskStubWriteReduce *>(node->task);
        slices.push_back(task->GetLocalSlice());
    }
    return;
}

void CheckRankMem::GetWriteSlice(TaskNode *node, std::vector<DataSlice> &slices)
{
    TaskTypeStub type = node->task->GetType();
    bool isGenFromSync = IsGenFromSync(node->task);
    if (type == TaskTypeStub::LOCAL_COPY) {
        auto task = dynamic_cast<TaskStubLocalCopy *>(node->task);
        if (task->GetDstSlice().GetType() == BufferType::OUTPUT_AIV && isGenFromSync) {
            return;
        }
        slices.push_back(task->GetDstSlice());
    } else if (type == TaskTypeStub::LOCAL_REDUCE && !isGenFromSync) {
        auto task = dynamic_cast<TaskStubLocalReduce *>(node->task);
        slices.push_back(task->GetDstSlice());
    } else if (type == TaskTypeStub::BEING_WRITTEN && !isGenFromSync) {
        auto task = dynamic_cast<TaskStubBeingWritten *>(node->task);
        slices.push_back(task->GetLocalSlice());
    } else if (type == TaskTypeStub::READ) {
        auto task = dynamic_cast<TaskStubRead *>(node->task);
        slices.push_back(task->GetLocalSlice());
    } else if (type == TaskTypeStub::BEING_WRITTEN_REDUCE && !isGenFromSync) {
        auto task = dynamic_cast<TaskStubBeingWrittenReduce *>(node->task);
        slices.push_back(task->GetLocalSlice());
    } else if (type == TaskTypeStub::READ_REDUCE) {
        auto task = dynamic_cast<TaskStubReadReduce *>(node->task);
        slices.push_back(task->GetLocalSlice());
    }
    return;
}

HcclResult CheckRankMem::ExecuteAiv(TaskNode* aivStart)
{
    // 先从整图中提取信息
    GenAivFragQueue(aivStart);

    RankId rankId = aivStart->rankIdx;
    CHK_RET(FindAivSyncPair(rankId));
    auto ret = GenFragQueConcurrencyMatrixAndCompare(rankId);
    ClearAIVData();
    if (ret != HcclResult::HCCL_SUCCESS) {
        DataDumper::Global()->SetResultStatus(gui::ResultStatus::MEMORY_CONFLICT);
        HCCL_ERROR("check rank memory conflict failed for rank %d", rankId);
        return ret;
    }

    return HcclResult::HCCL_SUCCESS;
}

#ifdef HCCL_ALG_ANALYZER_DAVID
HcclResult CheckRankMem::CcuGraphMemCheck()
{
    for (auto &child : graphHead_->children) {
        uint32_t queueId = child->queIdx;
        RankId   rankId  = child->rankIdx;
        // 头结点链接的都应该是主流，queIdx=0
        if (queueId != 0) {
            HCCL_ERROR("The node connecting the head node should be the mainstream.");
            return HcclResult::HCCL_E_PARA;
        }

        std::queue<TaskNodePtr> graphNodeQue;
        std::set<TaskNodePtr> isVisited;
        graphNodeQue.push(child);
        isVisited.insert(child);

        while (!graphNodeQue.empty()) {
            TaskNodePtr currNode = graphNodeQue.front();
            graphNodeQue.pop();
            CHK_RET(ProceedNode(currNode, graphNodeQue, isVisited));
            if (currNode->task->GetType() == TaskTypeStub::CCU_GRAPH) {
                auto ccuNode = Hccl::GetCcuTaskHead(currNode);
                CHK_RET(CcuGraphMemCheckProc(rankId, queueId, ccuNode));
            }
        }
        std::cout<<"zhf-rank "<<rankId<<" CcuGraphMemCheck proc success."<<std::endl;
    }

    return HcclResult::HCCL_SUCCESS;
}

// CCU子图内存冲突校验
HcclResult CheckRankMem::CcuGraphMemCheckProc(RankId rankId, uint32_t queueId, TaskNodePtr ccuNode)
{
    // 收集ccugraph中碎片
    std::set<u32> seenQueues;
    seenQueues.insert(queueId);
    for (auto &child : ccuNode->children) {
        clock_t start = clock();
        GenFragQueueInOneQueue(child, seenQueues);
        clock_t afterGenFrag = clock();

        // 从每个rank中提取post/wait队列
        CHK_RET(FindPostWaitPair(rankId, true));
        clock_t afterFindPos = clock();
        auto ret = GenFragQueConcurrencyMatrixAndCompare(rankId);
        if (ret != HcclResult::HCCL_SUCCESS) {
            DataDumper::Global()->SetResultStatus(gui::ResultStatus::MEMORY_CONFLICT);
            HCCL_ERROR("check rank memory conflict failed for rank %d", rankId);
            return ret;
        }
        clock_t afterCheck = clock();
        double genFrag = double(afterGenFrag - start) / CLOCKS_PER_SEC;
        double findPos = double(afterFindPos - afterGenFrag) / CLOCKS_PER_SEC;
        double checkMem = double(afterCheck - afterFindPos) / CLOCKS_PER_SEC;
        if (genFrag >= 2 || findPos >= 2 || checkMem >= 2) {
            std::cout<<"[CcuGraphMemCheckProc]Cost time: genFrag= "<<genFrag<<", findPos= "<<findPos<<", checkMem= "<<checkMem<<std::endl;
        }
        // 清除数据，否则会影响下一个ccu子图的内存冲突校验
        ClearCcuData();
    }

    return HcclResult::HCCL_SUCCESS;
}

HcclResult CheckRankMem::ProceedNode(
    TaskNodePtr currNode, std::queue<TaskNodePtr> &graphNodeQue, std::set<TaskNodePtr> &isVisited)
{
    for (auto childIter = currNode->children.begin(); childIter != currNode->children.end(); childIter++) {
        if (!isVisited.count(*childIter) && (*childIter)->rankIdx == currNode->rankIdx) {
            graphNodeQue.push((*childIter));
            isVisited.insert((*childIter));
        }
    }
    return HcclResult::HCCL_SUCCESS;
}

void CheckRankMem::ClearCcuData()
{
    for (auto& fragQueue : rank2FragQueue_) {
        fragQueue.second.clear();
    }
    rank2FragQueue_.clear();

    for (auto& postWaitPairs : rank2PostWaitPairs_) {
        postWaitPairs.second.clear();
    }
    rank2PostWaitPairs_.clear();
}
#endif

void CheckRankMem::ClearAIVData()
{
    for (auto& fragQueue : rank2FragQueue_) {
        fragQueue.second.clear();
    }
    rank2FragQueue_.clear();

    for (auto& postWaitPairs : rank2AivPostWaitPairs_) {
        postWaitPairs.second.clear();
    }
    rank2AivPostWaitPairs_.clear();
    rank2AivSendRecvPairs_.clear();
}

HcclResult CheckRankMem::Execute()
{
    // 如果有AivTask，不管是纯AIV算法还是混编算法，均先处理AIV子图内层的冲突校验
    if (graphHead_->hasAivTask) {
        auto allAivStartSet = AivTaskQueueStub::Global()->GetAllAivTasks().copyRank2AivTask;
        for (auto &aivStartSet : allAivStartSet) {
            for (auto &aivStart : aivStartSet.second) {
                CHK_RET(ExecuteAiv(aivStart));
            }
        }
    }

#ifdef HCCL_ALG_ANALYZER_DAVID
    // 如果有ccu子图，先处理ccu子图冲突校验
    if (graphHead_->hasCcuTask) {
        CHK_RET(CcuGraphMemCheck());
    }
#endif

    // 先从整图中提取信息
    GenFragQueue();

    // 从每个rank中提取post/wait队列
    for (auto &child : graphHead_->children) {
        RankId rankId = child->rankIdx;
        CHK_RET(FindPostWaitPair(rankId));
        auto ret = GenFragQueConcurrencyMatrixAndCompare(rankId);
        if (ret != HcclResult::HCCL_SUCCESS) {
            DataDumper::Global()->SetResultStatus(gui::ResultStatus::MEMORY_CONFLICT);
            HCCL_ERROR("check rank memory conflict failed for rank %d", rankId);
            return ret;
        }
    }
    return HcclResult::HCCL_SUCCESS;
}

} // namespace Hccl