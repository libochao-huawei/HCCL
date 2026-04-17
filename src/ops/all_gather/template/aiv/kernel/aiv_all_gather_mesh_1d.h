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
class AivAllGatherMesh1D : public AivCommBase {
public:
    __aicore__ inline AivAllGatherMesh1D() {}
 
    __aicore__ inline void Process(uint64_t count, uint64_t sliceId, uint64_t stride)
    {
        curTag_ = (static_cast<uint32_t>(tag_) << AIV_TAG_MOVE_RIGHT_BITS) | (sliceId & LOW_16_BITS);
        RunCtrlCore(count, stride);
    }
 
    __aicore__ inline void RunCtrlCore(uint64_t count, uint64_t stride)
    {
        // 核数小于ranksize
        if (numBlocks_ > rankSize_) {
            numBlocks_ = rankSize_;
        }
        if (block_idx >= numBlocks_) {
            SyncAll<true>();
            return;
        }
        // 分核把数据从input搬到gm
        auto input = reinterpret_cast<__gm__ T *>(input_);
        uint64_t dataTypeSize = sizeof(T);
        uint64_t countPerCore = count / numBlocks_;
        uint64_t curCountCore = block_idx == numBlocks_ - 1 ? count - countPerCore * (numBlocks_ - 1) : countPerCore;
        auto gmIn = reinterpret_cast<__gm__ T *>(reinterpret_cast<uint64_t>(GM_IN[rank_]) + block_idx * countPerCore * dataTypeSize);
        CpGM2GM(gmIn, input + block_idx * countPerCore, curCountCore);
        SyncAll<true>();

        // 每个核分配多个rank搬运数据从gm到对端output
        uint32_t perCoreRankNum = rankSize_ / numBlocks_;
        uint32_t remainRankNum = rankSize_ % numBlocks_;
        uint32_t curCoreRankNum = block_idx < remainRankNum ? perCoreRankNum + 1 : perCoreRankNum;
        uint32_t startRank = block_idx < remainRankNum ? (perCoreRankNum + 1) * block_idx : perCoreRankNum * block_idx + remainRankNum;
        for (uint32_t rank = startRank; rank < startRank + curCoreRankNum; rank++) {
            Record(rank, rank_, curTag_);
        }
        for (uint32_t rank = startRank; rank < startRank + curCoreRankNum; rank++) {
            auto gmOthers = reinterpret_cast<__gm__ T *>(reinterpret_cast<uint64_t>(GM_IN[rank]));
            auto output = reinterpret_cast<__gm__ T *>(output_ + rank * stride);
            WaitFlag(rank_, rank, curTag_);
            CpGM2GM(output, gmOthers, count);
            PipeBarrier<PIPE_ALL>();
            Record(rank, rank_ + rankSize_, curTag_);
        }
        for (uint32_t rank = startRank; rank < startRank + curCoreRankNum; rank++) {
            WaitFlag(rank_, rank + rankSize_, curTag_);
        }
    }
    uint64_t coreOffset;
    uint64_t curCount;
};
 
template<typename T>
__aicore__ inline void AivAllGatherV2Mesh1D(EXTERN_KERNEL_ARGS_DEF_V2)
{
    AivAllGatherMesh1D<T> op;
    op.Init(KERNEL_CLASS_INIT, true);
    SyncAll<true>();
    if (op.IsFirstOP(sliceId)) {
        op.BarrierForFirstOP();
    }
    SyncAll<true>();
 
    op.Process(len, sliceId, outputSliceStride);
    // 执行barrier全同步
    op.BarrierAll();
}