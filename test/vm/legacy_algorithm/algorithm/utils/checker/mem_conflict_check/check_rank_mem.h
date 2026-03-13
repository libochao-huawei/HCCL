/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2024-2024. All rights reserved.
 * Description: 内存冲突校验类文件头
 * Author: huangweihao
 * Create: 2024-11-11
 */
#ifndef HCCLV1_CHECK_RANK_MEM_H
#define HCCLV1_CHECK_RANK_MEM_H

#include <map>
#include <set>
#include <vector>
#include <memory>
#include <queue>

#include "check_utils.h"

namespace checker {

struct FragmentQueue {
    u32 queIdx;
    s32 blockIdx;  // for AIV
    s32 pipeIdx;   // for AIV
    bool isAIV;    // for AIV
    TaskNode *head;
    TaskNode *tail;

    inline bool operator<(const FragmentQueue &another) const
    {
        if (isAIV) {
            if (blockIdx < another.blockIdx) {
                return true;
            } else if (blockIdx > another.blockIdx) {
                return false;
            } else if (blockIdx == another.blockIdx) {
                if (pipeIdx != another.pipeIdx) {
                    return pipeIdx < another.pipeIdx;
                } else {
                    if (head < another.head) {
                        return true;
                    }
                    if (head > another.head) {
                        return false;
                    }
                    if (tail < another.tail) {
                        return true;
                    }
                    if (tail > another.tail) {
                        return false;
                    }
                    return false;
                }
            }
        } else {
            if (queIdx != another.queIdx) {
                return queIdx < another.queIdx;
            }
            return head < another.head; // todo: 此处返回两个指针的大小比较是什么意思？
        }
        return false;
    }

    std::string Describe() const
    {
        std::stringstream ret;
        if (isAIV) {
            ret << "blockIdx is " << to_string(blockIdx) << ", pipeIdx is " << to_string(pipeIdx) << ". ";
        } else {
            ret << "queId is " << to_string(queIdx) << ". ";
        }
        
        if (head != nullptr && head->task != nullptr) {
            ret << "head ptr is " << head->task->Describe() << ". ";
        }
        if (tail != nullptr && tail->task != nullptr) {
            ret << "tail ptr is " << tail->task->Describe() << "." << std::endl;
        }
        return ret.str();
    }
};

enum class MemoryStatus {
    READ  = 1,
    WRITE = 2,
};

struct SliceMemoryStatus {
    u64                  startAddr;
    mutable u64          size;
    mutable MemoryStatus status;

    inline bool operator<(const SliceMemoryStatus &another) const
    {
        return startAddr < another.startAddr;
    }

    std::string Describe() const
    {
        std::stringstream ret;
        ret << "startAddr is " << to_string(startAddr) << ", ";
        ret << "size  is " << size << ", ";
        ret << "status is ";

        if (status == MemoryStatus::READ) {
            ret << "READ";
        } else if (status == MemoryStatus::WRITE) {
            ret << "WRITE";
        } else {
            ret << "READ|WRITE";
        }

        ret << "." << std::endl;
        return ret.str();
    }
};

using FragQueueMemStatus = std::map<BufferType, std::set<SliceMemoryStatus>>;

struct BlockIdxPipeIdx {
    s32 blockIdx;
    s32 pipeIdx;

    inline bool operator < (const BlockIdxPipeIdx &another) const
    {
        //return (blockIdx < another.blockIdx) && (pipeIdx < another.pipeIdx);
        if (blockIdx < another.blockIdx) {
            return true;
        } else if (blockIdx == another.blockIdx) {
            return pipeIdx < another.pipeIdx;
        } else {
            return false;
        }
    }
};

class CheckRankMem {
public:
    explicit CheckRankMem(TaskNodePtr head) : graphHead_(head)
    {
    }
    HcclResult Execute();
    HcclResult ExecuteAiv(TaskNode* aivStart);

private:
    // 生成每个rank上的原语碎片队列
    void GenFragQueueInOneQueue(TaskNode *head, std::set<u32> &seenQueues);
    void GenFragQueueInOneRank(TaskNode *node);
    void GenFragQueue();
    bool CheckCcuInvalidNode(bool isCcuGraph, TaskNode *head);
    // 生成AIV每个rank上的原语碎片队列
    TaskNode* GetPipeBarrierChildNode(TaskNode *pipeeBarrier, s32 pipeIdx);
    void GenAivFragQueueInOnePipe(TaskNode *head);
    void GenAivFragQueue(TaskNode* aivStart);

#ifdef HCCL_ALG_ANALYZER_DAVID
    HcclResult CcuGraphMemCheck();
    HcclResult CcuGraphMemCheckProc(RankId rankId, uint32_t queueId, TaskNodePtr ccuNode);
    void ClearCcuData();
    HcclResult ProceedNode(
        TaskNodePtr currNode, std::queue<TaskNodePtr> &graphNodeQue, std::set<TaskNodePtr> &isVisited);
#endif

    // 找出一个rank上的Post/Wait节点
    void       FindPostWaitNode(TaskNode *node, std::set<TaskNode *> &postNodes, std::set<TaskNode *> &waitNodes) const;
    HcclResult FindPostWaitPair(RankId rankId, bool isCcuGraph = false);

    // 找出一个rank上的SetFlag/WaitFlag SendSync/RecvSync节点
    void FindAivPostWaitNode(TaskNode *node, std::set<TaskNode *> &postNodes, std::set<TaskNode *> &waitNodes) const;
    HcclResult FindAivSyncPair(RankId rankId);

    void ProcessEqualToTargetStartAddr(u64 &sliceStartAddr, u64 sliceEndAddr, std::vector<SliceMemoryStatus> &addedEles,
                                       MemoryStatus sliceStatus, std::set<SliceMemoryStatus>::iterator target) const;
    void ProcessGreatThanTargetStartAddr(u64 &sliceStartAddr, u64 sliceEndAddr,
                                         std::vector<SliceMemoryStatus> &addedEles, MemoryStatus sliceStatus,
                                         std::set<SliceMemoryStatus>::iterator target) const;
    // 针对每个内存碎片队列，产生其内存使用状态
    void       GenSliceMemoryInfo(DataSlice &slice, MemoryStatus sliceStatus, FragQueueMemStatus &result);
    HcclResult GenPrimNodeMemoryInfo(TaskNode *node, FragQueueMemStatus &result);
    HcclResult GenFragQueueMemoryInfo(FragmentQueue &fragQueue, FragQueueMemStatus &result);
    HcclResult GenAivTaskNodeMemoryInfo(TaskNode *node, FragQueueMemStatus &result);

    // 比较两个原语内存碎片的内存使用状态，看内存使用是否冲突
    HcclResult CompareBufferTypeMemoryInfo(std::set<SliceMemoryStatus> &left, std::set<SliceMemoryStatus> &right);
    HcclResult CompareBufferTypeMemoryInfo(std::set<SliceMemoryStatus> &left, std::set<SliceMemoryStatus> &right, 
                                           SliceMemoryStatus &conflictEleA, SliceMemoryStatus &conflictEleB);
    HcclResult CompareSliceMemoryInfo(FragQueueMemStatus &left, FragQueueMemStatus &right);

    // 产生原语碎片队列的冲突矩阵，并对可能冲突的原语碎片队列进行内存使用校验
    HcclResult CompareFragQueStatus(u32 fragQueueSize, std::vector<FragmentQueue> &index2FragQueue,
                                    std::vector<std::vector<bool>> &fragQueueMatrix);
    HcclResult GenFragQueConcurrencyMatrixAndCompare(RankId rankId);
    void GetReadSlice(TaskNode *node, std::vector<DataSlice> &slices);
    void GetWriteSlice(TaskNode *node, std::vector<DataSlice> &slices);
    bool IsConfilictBetweenTwoNodes(TaskNode* nodeA, TaskNode* nodeB, SliceMemoryStatus &conflictEleA, SliceMemoryStatus &conflictEleB);
    bool IsLastTaskNode(TaskNode* node);

    // AIV多子图复用map结构，每个子图执行完清理一次
    void ClearAIVData();

    TaskNodePtr graphHead_;
    std::map<RankId, std::set<FragmentQueue>> rank2FragQueue_;
    std::map<RankId, std::map<TaskNode *, TaskNode *>> rank2PostWaitPairs_;
    std::map<RankId, std::map<TaskNode *, TaskNode *>> rank2AivPostWaitPairs_;  //AIV的PostWait对，包含SetFlag/WaitFlag
    std::map<RankId, std::map<TaskNode*, std::set<TaskNode*>>> rank2AivSendRecvPairs_; //AIV的SendSync/RecvSync
};
} // namespace Hccl

#endif