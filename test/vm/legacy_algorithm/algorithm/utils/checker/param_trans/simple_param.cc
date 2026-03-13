/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2024-2024. All rights reserved.
 * Description: 参数转换头文件，用于对接可视化工具及命令行工具
 * Author: yinding
 * Create: 2025-01-15
 */

#include "simple_param.h"
#include <vector>
#include "check_utils.h"

#ifndef HCCL_ALG_ANALYZER_DAVID
using namespace hccl;
#endif

using namespace Hccl;

namespace checker {

std::string HcclCMDType2Tag(CheckerOpType optype)
{
    std::string tag;
    switch (optype) {
        case CheckerOpType::ALLGATHER:
            tag = "AllGather";
            break;
        case CheckerOpType::ALLREDUCE:
            tag = "AllReduce";
            break;
        case CheckerOpType::ALLTOALL:
        case CheckerOpType::ALLTOALLV:
        case CheckerOpType::ALLTOALLVC:
            tag = "AllToAll";
            break;
        case CheckerOpType::BROADCAST:
            tag = "BroadCast";
            break;
        case CheckerOpType::REDUCE_SCATTER:
            tag = "ReduceScatter";
            break;
        case CheckerOpType::REDUCE:
            tag = "Reduce";
            break;
        case CheckerOpType::SCATTER:
            tag = "Scatter";
            break;
        case CheckerOpType::SEND:
        case CheckerOpType::RECEIVE:
            tag = "SendRecv";
            break;
        case CheckerOpType::ALLGATHER_V:
            tag = "AllGatherV";
            break;
        case CheckerOpType::REDUCE_SCATTER_V:
            tag = "ReduceScatterV";
            break;
        case CheckerOpType::BATCH_WRITE:
            tag = "batch_write";
            break;
        default:
            tag = "invalid";
    }

    return tag;
}

std::vector<u64> GenerateSendCountMatrix(u64 count, u32 rankSize)
{
    std::vector<u64> sendCountMatrix(rankSize * rankSize, count);
    return sendCountMatrix;
}

void GenAllToAllVParams(u32 rankSize, u64 count, std::vector<u64>& sendCounts, std::vector<u64>& sdispls,
                        std::vector<u64>& recvCounts, std::vector<u64>& rdispls)
{
    u64 sendDisplacement = 0;
    u64 recvDisplacement = 0;
    for (u32 i = 0; i < rankSize; i++) {
        sendCounts.push_back(count);
        sdispls.push_back(sendDisplacement);
        recvCounts.push_back(count);
        rdispls.push_back(recvDisplacement);
        sendDisplacement += count;
        recvDisplacement += count;
    }
    return;
}

HcclResult GenTestOpParams(u32 rankSize, const SimpleParam& uiParam, CheckerOpParam& testOpParam)
{
    testOpParam.opType = uiParam.opType;
    testOpParam.tag = HcclCMDType2Tag(uiParam.opType);
    testOpParam.algName = uiParam.algName;
    testOpParam.opMode = uiParam.opMode;
    testOpParam.reduceType = uiParam.reduceType;
    testOpParam.devtype = uiParam.devtype;
    testOpParam.is310P3V = uiParam.is310P3V;

    if (uiParam.opType == CheckerOpType::BROADCAST || uiParam.opType == CheckerOpType::REDUCE ||
        uiParam.opType == CheckerOpType::GATHER || uiParam.opType == CheckerOpType::SCATTER) {
        testOpParam.root = uiParam.root;
    }

    testOpParam.dstRank = uiParam.dstRank;
    testOpParam.srcRank = uiParam.srcRank;

    if (IsAllToAllSeries(testOpParam.opType)) {
        if (uiParam.count % rankSize != 0) {
            HCCL_WARNING("count[%llu] is not divisible by rankSize[%u]", uiParam.count, rankSize);
        }
    }

    if (testOpParam.opType == CheckerOpType::ALLTOALL || testOpParam.opType == CheckerOpType::ALLTOALLVC) {
        u64 count = uiParam.count / rankSize;
        testOpParam.All2AllDataDes.sendCountMatrix = GenerateSendCountMatrix(count, rankSize);
        testOpParam.All2AllDataDes.sendCount = count;
        testOpParam.All2AllDataDes.sendType = uiParam.dataType;
        testOpParam.All2AllDataDes.recvType = uiParam.dataType;
    } else if (testOpParam.opType == CheckerOpType::ALLTOALLV) {
        u64 count = uiParam.count / rankSize;
        GenAllToAllVParams(rankSize, count, testOpParam.All2AllDataDes.sendCounts,
            testOpParam.All2AllDataDes.sdispls,
            testOpParam.All2AllDataDes.recvCounts, testOpParam.All2AllDataDes.rdispls);
        testOpParam.All2AllDataDes.sendType = uiParam.dataType;
        testOpParam.All2AllDataDes.recvType = uiParam.dataType;
    } else if (testOpParam.opType == CheckerOpType::ALLGATHER_V ||
               testOpParam.opType == CheckerOpType::REDUCE_SCATTER_V) {
        u64 displacement = 0;
        for (u32 i = 0; i < rankSize; i++) {
            testOpParam.VDataDes.counts.push_back(uiParam.count);
            testOpParam.VDataDes.displs.push_back(displacement);
            displacement += uiParam.count;
        }
        testOpParam.VDataDes.dataType = uiParam.dataType;
    } else {
        testOpParam.DataDes.count = uiParam.count;
        testOpParam.DataDes.dataType = uiParam.dataType;
    }

    if (testOpParam.opType == CheckerOpType::BATCH_SEND_RECV) {
        testOpParam.allRanksSendRecvInfoVec.resize(rankSize);
    }

    return HCCL_SUCCESS;
}

} // namespace hccl