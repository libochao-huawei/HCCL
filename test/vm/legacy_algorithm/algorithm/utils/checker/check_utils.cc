/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2024-2024. All rights reserved.
 * Description: 内存冲突校验/语义校验辅助函数实现
 * Author: yinding
 * Create: 2024-02-04
 */
#include "check_utils.h"

#include <vector>

#include "task_stub.h"
#include "mem_layout.h"

using namespace Hccl;

namespace checker {

const std::string FOUR_INDENT_SPACE = "    ";

// 获取原语的类型
TaskTypeStub GetNodeType(const TaskNode *node)
{
    return node->task->GetType();
}

bool IsAllToAllSeries(CheckerOpType opType)
{
    return (opType == CheckerOpType::ALLTOALL || opType == CheckerOpType::ALLTOALLV ||
            opType == CheckerOpType::ALLTOALLVC);
}

bool IsSendRecvType(CheckerOpType opType)
{
    return opType == CheckerOpType::SEND || opType == CheckerOpType::RECEIVE;
}

void CalcInputOutputSize(const CheckerOpParam &opParam, u32 ranksize, u64 &inputSize, u64 &outputSize, RankId myRank)
{
    u32 unitSize = 0;
    if (!IsAllToAllSeries(opParam.opType) && opParam.opType != CheckerOpType::BATCH_SEND_RECV &&
        opParam.opType != CheckerOpType::REDUCE_SCATTER_V && opParam.opType != CheckerOpType::ALLGATHER_V) {
        unitSize = CHECK_SIZE_TABLE[opParam.DataDes.dataType];
    }

    u64 count = opParam.DataDes.count;
    if (opParam.opType == CheckerOpType::ALLREDUCE) {
        inputSize = count * unitSize;
        outputSize = count * unitSize;
    } else if (opParam.opType == CheckerOpType::BROADCAST) {
        inputSize = count * unitSize;
        outputSize = count * unitSize;
    } else if (IsSendRecvType(opParam.opType) && myRank == opParam.srcRank) {
        inputSize = count * unitSize;
        outputSize = 0;
    } else if (IsSendRecvType(opParam.opType) && myRank == opParam.dstRank) {
        inputSize = 0;
        outputSize = count * unitSize;
    } else if (opParam.opType == CheckerOpType::REDUCE) {
        if (myRank == opParam.root) {
            outputSize = count * unitSize;
        } else {
            // 当前代码中非root节点还是会用到OUTPUT内存块
            outputSize = count * unitSize;
        }
        inputSize = count * unitSize;
    } else if (opParam.opType == CheckerOpType::ALLGATHER) {
        inputSize = count * unitSize;
        outputSize = count * unitSize * ranksize;
    } else if (opParam.opType == CheckerOpType::REDUCE_SCATTER) {
        inputSize = count * unitSize * ranksize;
        outputSize = count * unitSize;
    } else if (opParam.opType == CheckerOpType::ALLTOALL || opParam.opType == CheckerOpType::ALLTOALLVC) {
        u64 curSendOffset = 0;
        u64 curRecvOffset = 0;
        void *sendCountMatrix = static_cast<void *>(const_cast<u64*>(opParam.All2AllDataDes.sendCountMatrix.data()));
        // 对于AllToAllV/AllToAllVC来说，当前checker还不支持不均匀的数据收发，每个rank收发的数据量是一样的，
        // 所以这边以rank0来计算即可
        RankId curRank = 0;
        // sendCountMatrix[i * ranksize + j] 代表rank i发送到rank j的count参数
        for (u32 j = 0; j < ranksize; j++) {
            u64 curSendCounts = *(static_cast<const u64 *>(sendCountMatrix) + curRank * ranksize + j);
            u64 curSendLength = curSendCounts * CHECK_SIZE_TABLE[opParam.All2AllDataDes.sendType];
            curSendOffset += curSendLength;

            u64 curRecvCounts = *(static_cast<const u64 *>(sendCountMatrix) + curRank + ranksize * j);
            u64 curRecvLength = curRecvCounts * CHECK_SIZE_TABLE[opParam.All2AllDataDes.recvType];
            curRecvOffset += curRecvLength;
        }
        inputSize = curSendOffset;
        outputSize = curRecvOffset;
    } else if (opParam.opType == CheckerOpType::ALLTOALLV) {
        void* sendCounts = static_cast<void *>(const_cast<u64*>(opParam.All2AllDataDes.sendCounts.data()));
        void* recvCounts = static_cast<void *>(const_cast<u64*>(opParam.All2AllDataDes.recvCounts.data()));

        u64 curSendOffset = 0;
        u64 curRecvOffset = 0;
        for (u32 i = 0; i < ranksize; i++) {
            u64 curSendCounts = *(static_cast<const u64 *>(sendCounts) + i);
            u64 curSendLength = curSendCounts * CHECK_SIZE_TABLE[opParam.All2AllDataDes.sendType];
            curSendOffset += curSendLength;

            u64 curRecvCounts = *(static_cast<const u64 *>(recvCounts) + i);
            u64 curRecvLength = curRecvCounts * CHECK_SIZE_TABLE[opParam.All2AllDataDes.recvType];
            curRecvOffset += curRecvLength;
        }
        inputSize = curSendOffset;
        outputSize = curRecvOffset;
    } else if (opParam.opType == CheckerOpType::SCATTER) {
        inputSize = count * unitSize * ranksize;
        outputSize = count * unitSize;
    } else if (opParam.opType == CheckerOpType::BATCH_SEND_RECV) {
        if (opParam.allRanksSendRecvInfoVec.size() == 0 || opParam.allRanksSendRecvInfoVec[0].size() == 0) {
            HCCL_ERROR("BatchSendRecv allRanksSendRecvInfoVec is empty.");
            return;
        }
        u32 unitSizePerTask = CHECK_SIZE_TABLE[opParam.allRanksSendRecvInfoVec[0][0].dataType];
        u64 countPerTask = opParam.allRanksSendRecvInfoVec[0][0].count;
        inputSize = ranksize * countPerTask * unitSizePerTask;
        outputSize = ranksize * countPerTask * unitSizePerTask;
    } else if (opParam.opType == CheckerOpType::REDUCE_SCATTER_V) {
        void* counts = static_cast<void *>(const_cast<u64*>(opParam.VDataDes.counts.data()));
        inputSize = 0;
        for (u32 i = 0; i < ranksize; i++) {
            u64 curCounts = *(static_cast<const u64 *>(counts) + i);
            u64 curLength = curCounts * CHECK_SIZE_TABLE[opParam.VDataDes.dataType];
            inputSize += curLength;
        }
        outputSize = static_cast<const u64 *>(counts)[myRank] * CHECK_SIZE_TABLE[opParam.VDataDes.dataType];
    } else if (opParam.opType == CheckerOpType::ALLGATHER_V) {
        void* counts = static_cast<void *>(const_cast<u64*>(opParam.VDataDes.counts.data()));
        outputSize = 0;
        for (u32 i = 0; i < ranksize; i++) {
            u64 curCounts = *(static_cast<const u64 *>(counts) + i);
            u64 curLength = curCounts * CHECK_SIZE_TABLE[opParam.VDataDes.dataType];
            outputSize += curLength;
        }
        inputSize = static_cast<const u64 *>(counts)[myRank] * CHECK_SIZE_TABLE[opParam.VDataDes.dataType];
    }
    return;
}

// 如果输入、输出的count的大小不一样的话，那么opParam中的count是指较小的那个值
// 比如对于AllGather算子，count指输入；对于ReduceScatter算子，count指输入
// 如果输入、输出的count大小一样的话，那么opParam中的count既可以指代输入，也可以指代输出
void CalcDataSize(const CheckerOpParam &opParam, u64 &dataSize)
{
    if (opParam.opType == CheckerOpType::BATCH_SEND_RECV) {
        u32 unitSize = CHECK_SIZE_TABLE[opParam.allRanksSendRecvInfoVec[0][0].dataType];
        u64 count = opParam.allRanksSendRecvInfoVec[0][0].count;
        dataSize = count * unitSize;
        return;
    }
    // 当前AllToAll系列以及不等长算子不使用dataSize，如果后续使用的话，需要适配这个地方
    if (!IsAllToAllSeries(opParam.opType) && opParam.opType != CheckerOpType::REDUCE_SCATTER_V &&
        opParam.opType != CheckerOpType::ALLGATHER_V) {
        u32 unitSize = CHECK_SIZE_TABLE[opParam.DataDes.dataType];
        u64 count = opParam.DataDes.count;
        dataSize = count * unitSize;
    }
    return;
}

std::vector<std::string> SplitString(const std::string &str, const char c)
{
    std::string::size_type startPos = 0;
    std::string::size_type foundPos = str.find(c);
 
    std::vector<std::string> strVector;
    while (foundPos != std::string::npos) {
        strVector.push_back(str.substr(startPos, foundPos - startPos));
        startPos = foundPos + 1;
        foundPos = str.find(c, startPos);
    }
    if (startPos != str.length()) {
        strVector.push_back(str.substr(startPos));
    }
    return strVector;
}

bool DataSliceSizeIsEqual(std::unique_ptr<DataSlice> &a, std::unique_ptr<DataSlice> &b)
{
    return a->GetSize() == b->GetSize();
}
 
bool DataSliceSizeIsEqual(std::unique_ptr<DataSlice> &a, std::unique_ptr<DataSlice> &b, std::unique_ptr<DataSlice> &c)
{
    return (a->GetSize() == b->GetSize()) && (b->GetSize() == c->GetSize());
}

} // namespace hccl