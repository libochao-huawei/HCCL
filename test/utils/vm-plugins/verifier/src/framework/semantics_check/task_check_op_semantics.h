/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef TASK_CHECK_OP_SEMANTICS_H
#define TASK_CHECK_OP_SEMANTICS_H

#include <map>
#include <set>
#include <vector>
#include <memory>
#include <queue>
#include "sim_task.h"
#include "check_utils.h"
#include "task_graph_generator.h"
#include "task_def.h"

namespace HcclSim {

enum class SliceOp {
    OVERRIDE,
    REDUCE,
};

struct SliceOpPair {
    RankId    srcRank;
    RankId    dstRank;
    DataSlice srcSlice;
    DataSlice dtsSlice;
    SliceOp   sliceOp;

    std::string Describe() const
    {
        std::stringstream ret;
        ret << "src rank is " << srcRank << ", ";
        ret << "dst rank is " << dstRank << ", ";
        ret << "src slice is " << srcSlice.Describe() << ", ";
        ret << "dst slice is " << dtsSlice.Describe() << ", ";
        if (sliceOp == SliceOp::OVERRIDE) {
            ret << "sliceOp is "
                << "SliceOp::OVERRIDE" << std::endl;
        } else {
            ret << "sliceOp is "
                << "SliceOp::REDUCE" << std::endl;
        }

        return ret.str();
    }
};

struct LocalStep {
    RankId rankId;
    u32 localStep;
};

class TaskCheckOpSemantics {
public:
    TaskCheckOpSemantics(TaskNodePtr head, CheckerOpParam &checkerOpParam, u32 rankSize)
        : graphHead_(head), opType_(checkerOpParam.opType), reduceType_(checkerOpParam.reduceType),
          checkerOpParam_(&checkerOpParam), root_(checkerOpParam.root), srcRank_(checkerOpParam.srcRank),
          dstRank_(checkerOpParam.dstRank), rankSize_(rankSize)
    {
        CalcDataSize(checkerOpParam, dataSize_);
    }
    HcclResult Execute();

private:
    void InitInputBuffer();
    void InitInputBuffer(RankId root);
    void UpdateStep(TaskNode *simNode);
    u32 GetCurLocalStep();
    HcclResult DumpNodeSemantics(TaskNode *simNode);

    void       GetSrcIntersectionAddr(SliceOpPair &slicePair, const BufferSemantic &srcBufSemantic, u64 &srcStartAddr,
                                      u64 &srcEndAddr) const;
    HcclResult CheckBufSemantics(std::vector<BufferSemantic *> &bufSemantics, u64 startAddr, u64 size, bool ignoreError = false) const;
    void RemoveAffectedBufSemantics(SliceOpPair &slicePair, std::vector<BufferSemantic *> &affectedDstBufSemantics);
    void       ApplyOverrideSrcBufSemantic(SliceOpPair &slicePair, const BufferSemantic srcBufSemantic);
    HcclResult ReduceToAffectedBufSemantic(const BufferSemantic         &srcBufSemantic,
                                           std::vector<BufferSemantic *> toAddReduceInfoSemantics, u64 srcStartAddr);
    HcclResult ApplyReduceSrcBufSemantic(SliceOpPair &slicePair, const BufferSemantic &srcBufSemantic,
                                         std::vector<BufferSemantic *> &affectedDstBufSemantics);
    void       GetAffectedBufSemantics(SliceOpPair &slicePair, const BufferSemantic &srcBufSemantic,
                                       std::vector<BufferSemantic *> &affectedDstBufSemantics);
    void GetAffectedBufSemantics(SliceOpPair &slicePair, std::vector<BufferSemantic *> &affectedDstBufSemantics);
    HcclResult ApplySrcBufSemanticsToDst(SliceOpPair &slicePair, std::vector<BufferSemantic *> srcBufSemantics);

    HcclResult ProcessSliceOpPair(SliceOpPair &slicePair);
    void       GetSliceOpPair(TaskNode *simNodes, std::vector<SliceOpPair> &sliceOpPairs) const;
    HcclResult ProcessNodeSemantics(TaskNode *simNode);

    bool       IsReadyForSimulate(const TaskNode *node, std::set<TaskNode *> &simulatedNodes) const;
    bool       IsNeedSimulateSameTime(TaskNode *node) const;
    HcclResult FindCorrespondingPairNode(TaskNode *node, std::vector<TaskNode *> &result) const;

    void       AddChildrenToQueue(TaskNode *node, std::set<TaskNode *> &visitedNodes,
                                  std::queue<TaskNode *> &walkQue, std::set<TaskNode *>& simulatedNodes) const;
    void       AddAivChildrenToQueue(TaskNode *node, std::set<TaskNode *> &visitedNodes,
                                  std::queue<TaskNode *> &walkQue, std::set<TaskNode *>& simulatedNodes) const;
    HcclResult GenMemSemantics();
    HcclResult GenAivMemSemantics();

    TaskNodePtr                           graphHead_;
    CheckerOpType                         opType_ = CheckerOpType::INVALID;
    CheckerReduceOp                       reduceType_ = CheckerReduceOp::REDUCE_RESERVED;
    CheckerOpParam                        *checkerOpParam_;
    u64                                   dataSize_ = 0;
    u64                                   inputDataSize_ = 0;
    u64                                   outputDataSize_ = 0;
    std::map<RankId, RankMemorySemantics> allRankMemSemantics_;
    RankId                                root_ = 0;
    RankId                                srcRank_ = 0;
    RankId                                dstRank_ = 1;
    u32                                   rankSize_;

    // 语义信息dump相关的环境变量
    u32 globalStep_ = 0;
    std::map<RankId, u32> localStep_;
    std::map<RankId, std::map<u32, u32>> localStep2GlobalStep_;
    std::map<u32, LocalStep> globalStep2LocalStep_;
    std::map<RankId, bool> memSemanticsChange_;
};

} // namespace HcclSim

#endif