/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2024-2024. All rights reserved.
 * Description: 内存冲突校验/语义校验辅助函数文件头
 * Author: yinding
 * Create: 2024-02-04
 */
#ifndef HCCLV1_CHECK_UTILS_H
#define HCCLV1_CHECK_UTILS_H

#include <map>
#include <set>
#include <vector>
#include <memory>

#include "base.h"
#include "log.h"
#include "task_stub.h"
#include "task_def.h"
#include "checker_def.h"

namespace checker {

extern const std::string FOUR_INDENT_SPACE;

struct SrcBufDes {
    RankId      rankId;  // 数据源的rankId
    BufferType  bufType; // 数据源的内存类型
    mutable u64 srcAddr; // 数据源的地址
    SrcBufDes(RankId id, BufferType type, u64 addr) : rankId(id), bufType(type), srcAddr(addr)
    {
    }
    inline bool operator<(const SrcBufDes &another) const
    {
        return rankId < another.rankId;
    }

    std::string Describe() const
    {
        std::stringstream ret;
        ret << "rankId is " << rankId << ", ";
        ret << "bufType  is " << bufType << ", ";
        ret << "srcAddr is " << srcAddr << std::endl;

        return ret.str();
    }
};

struct BufferSemantic {
    u64                         startAddr;  // 起始地址
    mutable u64                 size;       // 大小
    mutable bool                isReduce;   // 是否做了reduce操作
    mutable CheckerReduceOp     reduceType; // reduce操作的类型
    mutable std::set<SrcBufDes> srcBufs;    // 这块数据来自哪个或哪些rank
    std::vector<u32>            affectedGlobalSteps;  // 表示这个语义块被哪个、哪些节点影响了，用于图形化界面展示

    BufferSemantic(u64 startAddr, u64 size, bool isReduce = false,
        CheckerReduceOp reduceType = CheckerReduceOp::REDUCE_RESERVED)
        : startAddr(startAddr), size(size), isReduce(isReduce), reduceType(reduceType)
    {
    }

    BufferSemantic(u64 startAddr, u64 size, bool isReduce, CheckerReduceOp reduceType, std::set<SrcBufDes> srcBufs)
        : startAddr(startAddr), size(size), isReduce(isReduce), reduceType(reduceType), srcBufs(srcBufs)
    {
    }

    inline bool operator<(const BufferSemantic &another) const
    {
        return startAddr < another.startAddr;
    }

    std::string Describe() const
    {
        std::stringstream ret;
        ret << "startAddr is " << startAddr << ", ";
        ret << "size is " << size << ", ";
        if (isReduce) {
            ret << "HcclReduceOp is " << reduceType << "." << std::endl;
        } else {
            ret << "no reduce" << std::endl;
        }
        for (auto &ele : srcBufs) {
            ret << FOUR_INDENT_SPACE << FOUR_INDENT_SPACE << FOUR_INDENT_SPACE << ele.Describe();
        }
        return ret.str();
    }
};

using RankMemorySemantics = std::map<BufferType, std::set<BufferSemantic>>;

TaskTypeStub GetNodeType(const TaskNode *node);
void CalcInputOutputSize(const CheckerOpParam &opParam, u32 ranksize, u64 &inputSize, u64 &outputSize, RankId myRank);
void CalcDataSize(const CheckerOpParam &opParam, u64 &dataSize);
bool IsAllToAllSeries(CheckerOpType opType);

} // namespace checker

#endif