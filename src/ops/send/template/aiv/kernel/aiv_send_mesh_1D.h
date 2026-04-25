/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "aiv_communication_base_v2.h"

using namespace AscendC;

template<typename T>
// todo 简化参数
class AivSendMesh1D : public AivCommBase { // 单核、多核都能运行
public:

    __aicore__ inline AivSendMesh1D() {
    }

    __aicore__ inline void InitCoreInfo(uint64_t processedDataCount, uint64_t currDataCount)
    {
        coreIndex = block_idx;  // 每个核在当前coreNumPerRank里面的排序

        // 发送数据的编排
        uint64_t dataPerCore = currDataCount / coreNumPerRank; // 数据量很少的时候，dataPerCore为0
        uint64_t remainder = currDataCount % coreNumPerRank;
        // 数据对不齐的情况
        uint64_t innerDispls = 0;
        if (coreIndex < remainder) { // 这部分核需要多处理一个数据
            innerDispls = coreIndex * dataPerCore + coreIndex;
            sendCurCount = dataPerCore + 1;
        } else {
            innerDispls = coreIndex * dataPerCore + remainder;
            sendCurCount = dataPerCore;
        }
        sendInputOffset = input_ + (processedDataCount + innerDispls) * sizeof(T);
        sendOutputOffset = reinterpret_cast<uint64_t>(GM_IN[rank_]) + innerDispls * sizeof(T);
    }

    __aicore__ inline void Producer()
    {
        if (sendCurCount == 0) {
            return;
        }
        uint64_t flag_offset = block_idx;
        WaitFlag(rank_, flag_offset, 0);

        CpGM2GM((__gm__ T *)sendOutputOffset, (__gm__ T *)sendInputOffset, sendCurCount);
        PipeBarrier<PIPE_ALL>();

        Record(rank_, flag_offset, curTag);
    }

    __aicore__ inline void Process(uint64_t len, uint32_t sliceId)
    {
        // 目标卡只有一个，一个或多个核去处理
        coreNumPerRank = numBlocks_;
        if (coreNumPerRank < 1) {
            return;
        }

        coreCount = coreNumPerRank;
        if (block_idx >= coreCount) { // 负责每个rank的核数相同，方便读写都能并行
            return;
        }

        targetRank = sendRecvRemoteRank_; // 每个核负责哪个rank的数据

        curTag = (static_cast<uint32_t>(tag_) << AIV_TAG_MOVE_RIGHT_BITS) | (sliceId & LOW_16_BITS);
        cclBufferCountPerRank = inputSliceStride_; // 整个cclBuffer给一张卡用

        // 运行过程中用到的所有flag，先置为0，后面会复用，recv侧不用做这个事情，动作是send端发起，所以send端去置位
        Record(rank_, block_idx, 0);
        PipeBarrier<PIPE_ALL>();

        uint64_t processedDataCount = 0;
        // 每张卡的loopTimes可能是不一样的
        uint64_t loopTimes = len / cclBufferCountPerRank +
            static_cast<uint64_t>(len % cclBufferCountPerRank != 0);
        for (uint64_t loop = 0; loop < loopTimes; loop++) {
            uint64_t currDataCount = (loop == loopTimes - 1) ? len - processedDataCount : cclBufferCountPerRank;
            InitCoreInfo(processedDataCount, currDataCount);
            Producer(); // 写数据
            processedDataCount += currDataCount;
        }
    }

    int32_t curTag;
    uint32_t coreCount;
    uint32_t coreNumPerRank;
    uint32_t targetRank;
    uint32_t coreIndex;
    uint64_t sendInputOffset;
    uint64_t sendOutputOffset;
    uint64_t sendCurCount;
    uint64_t recvInputOffset;
    uint64_t recvOutputOffset;
    uint64_t recvCurCount;
    uint64_t cclBufferCountPerRank;
};

template<typename T>
__aicore__ inline void AivSendV2Mesh1D(EXTERN_KERNEL_ARGS_DEF_V2)
{
    AivSendMesh1D<T> op;
    op.Init(KERNEL_CLASS_INIT, true);
    SyncAll<true>();
    if (op.IsFirstOP(sliceId)) {
        op.SendRecvBarrierForFirstOP(rank, sendRecvRemoteRank);
    }
    SyncAll<true>();
    op.Process(len, sliceId);
    op.SendRecvBarrierAll(rank, sendRecvRemoteRank);
}
