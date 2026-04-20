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
 
    // 如果让rankSize个核去写本地，写的时候切成多份去写
    // 然后剩下的核，切成rankSize份，去读
    // 需要知道写本地和读远端的时间比例，才可以调节怎么切分
    __aicore__ inline void InitCoreInfo(uint64_t len) //len is input length
    {
        // 每个核处理 len/curStageCoreNum 个数据
        uint64_t targetRank = GetBlockIdx();
        uint64_t dataPerRank = len / rankSize_;
        uint64_t remainderPerRank = len % rankSize_;
        // 数据对不齐的情况
        uint64_t innerDisplsPerRank = 0;
        uint64_t sendCurCountPerRank = 0;
        if (targetRank < remainderPerRank) { // 这部分核需要多处理一个数据
            innerDisplsPerRank = targetRank * dataPerRank + targetRank;
            sendCurCountPerRank = dataPerRank + 1;
        } else {
            innerDisplsPerRank = targetRank * dataPerRank + remainderPerRank;
            sendCurCountPerRank = dataPerRank;
        }

        uint64_t dataPerCore = sendCurCountPerRank / cutNum;
        uint64_t remainder = sendCurCountPerRank % cutNum;
        // 这里是要切三分，还是直接一份，然后多个flag？
        for (int64_t coreIndex = 0; coreIndex < cutNum; coreIndex++) {
            // 数据对不齐的情况
            uint64_t innerDispls = 0;
            uint64_t sendCurCount = 0;
            if (coreIndex < remainder) { // 这部分核需要多处理一个数据
                innerDispls = coreIndex * dataPerCore + coreIndex;
                sendCurCount = dataPerCore + 1;
            } else {
                innerDispls = coreIndex * dataPerCore + remainder;
                sendCurCount = dataPerCore;
            }

            if (sendCurCount > 0) {
                uint64_t usrInOffset = input_ + (innerDisplsPerRank + innerDispls) * sizeof(T);
                uint64_t cclInOffset = reinterpret_cast<uint64_t>(GM_IN[rank_]) + (innerDisplsPerRank + innerDispls) * sizeof(T);
                CpGM2GM((__gm__ T *)cclInOffset, (__gm__ T *)usrInOffset, sendCurCount);
                PipeBarrier<PIPE_ALL>();
                // 每个核写flag
                Record(rank_, GetBlockIdx() * cutNum + coreIndex, curTag_);
            }
        }
    }
 
    __aicore__ inline void Run(uint64_t len, uint64_t stride)
    {
        uint64_t blockInedx = GetBlockIdx() - coreNumStage1;
        // 然后stage2的核分成rankSize份，每一份有cutNum个核，每个核去读对端的rankSize份数据
        uint32_t coreNumPerRank = coreNumStage2 / rankSize_;
        uint32_t targetRank = blockInedx / coreNumPerRank;
        uint32_t coreIndex = (blockInedx - (targetRank * coreNumPerRank))  % coreNumPerRank;
        for (uint32_t idx = 0; idx < rankSize_; idx++) {
            // 这里根据写数据时候的编排，去计算每次读数据时候的编排
            uint64_t innerIndex = (idx + rank_) % rankSize_; // 做一点偏移，不要出现好多张卡读同一块数据的情况，却似有一点点效果
            uint64_t dataPerRank = len / rankSize_;
            uint64_t remainderPerRank = len % rankSize_;
            // 数据对不齐的情况
            uint64_t innerDisplsPerRank = 0;
            uint64_t sendCurCountPerRank = 0;
            if (innerIndex < remainderPerRank) { // 这部分核需要多处理一个数据
                innerDisplsPerRank = innerIndex * dataPerRank + innerIndex;
                sendCurCountPerRank = dataPerRank + 1;
            } else {
                innerDisplsPerRank = innerIndex * dataPerRank + remainderPerRank;
                sendCurCountPerRank = dataPerRank;
            }

            // 然后每个核分数据
            uint64_t dataPerCore = sendCurCountPerRank / coreNumPerRank;
            uint64_t remainder = sendCurCountPerRank % coreNumPerRank;
            // 数据对不齐的情况
            uint64_t innerDispls = 0;
            uint64_t sendCurCount = 0;
            if (coreIndex < remainder) { // 这部分核需要多处理一个数据
                innerDispls = coreIndex * dataPerCore + coreIndex;
                sendCurCount = dataPerCore + 1;
            } else {
                innerDispls = coreIndex * dataPerCore + remainder;
                sendCurCount = dataPerCore;
            }

            if (sendCurCount > 0) {
                // 先去wait flag，然后把数据拷贝过来
                WaitFlag(targetRank, innerIndex * coreNumPerRank + coreIndex, curTag_);

                uint64_t srcCclInOffset = reinterpret_cast<uint64_t>(GM_IN[targetRank]) + (innerDisplsPerRank + innerDispls )* sizeof(T);
                uint64_t usrOutOffset = output_ + targetRank * stride + (innerDisplsPerRank + innerDispls) * sizeof(T);
                CpGM2GM((__gm__ T *)usrOutOffset, (__gm__ T *)srcCclInOffset, sendCurCount);
                PipeBarrier<PIPE_ALL>();
            }
        }
    }
 
    __aicore__ inline void Process(uint64_t count, uint64_t sliceId, uint64_t stride)
    {
        curTag_ = (static_cast<uint32_t>(tag_) << AIV_TAG_MOVE_RIGHT_BITS) | (sliceId & LOW_16_BITS);
        if (count * sizeof(T) >= DATA_LIMIT && numBlocks_ >= 2 * rankSize_) {
            // 核数大于等于2倍ranksize
            curStageCoreNum = numBlocks_ / rankSize_ * rankSize_; // 总的核数
            coreNumStage1 = rankSize_;
            coreNumStage2 = curStageCoreNum - coreNumStage1;
            cutNum = coreNumStage2 / rankSize_;

            if (GetBlockIdx() < coreNumStage1) {
                InitCoreInfo(count);
            } else if (GetBlockIdx() < curStageCoreNum) {
                Run(count, stride);
            }
        } else {
            // 核数小于ranksize
            RunCtrlCore(count, stride);
        }
    }
 
    __aicore__ inline void RunCtrlCore(uint64_t count, uint64_t stride)
    {
        if (numBlocks_ > rankSize_) {
            numBlocks_ = rankSize_;
        }
        // 核数小于ranksize
        if (block_idx >= numBlocks_) {
            return;
        }
        // 分核把数据从input搬到gm
        auto input = reinterpret_cast<__gm__ T *>(input_);
        uint64_t dataTypeSize = sizeof(T);
        uint64_t countPerCore = count / numBlocks_;
        uint64_t curCountCore = block_idx == numBlocks_ - 1 ? count - countPerCore * (numBlocks_ - 1) : countPerCore;
        auto gmIn = reinterpret_cast<__gm__ T *>(reinterpret_cast<uint64_t>(GM_IN[rank_]) + block_idx * countPerCore * dataTypeSize);
        CpGM2GM(gmIn, input + block_idx * countPerCore * dataTypeSize, curCountCore);
        PipeBarrier<PIPE_ALL>();
        Record(rank_, block_idx, curTag_);
        for (uint32_t idx = 0; idx < numBlocks_; idx++) {
            WaitFlag(rank_, idx, curTag_);
            Record(rank_, idx, 0);
        }
        if (block_idx == 0) {
            Record(rank_, rank_, curTag_);
        }
        // 每个核分配多个rank搬运数据从gm到对端output
        uint32_t perCoreRankNum = rankSize_ / numBlocks_;
        uint32_t curCoreRankNum = block_idx == numBlocks_ - 1 ? rankSize_ - perCoreRankNum * (numBlocks_ - 1) : perCoreRankNum;
        uint32_t startRank = block_idx * perCoreRankNum;
        for (uint32_t rank = startRank; rank < startRank + curCoreRankNum; rank++) {
            auto gmOthers = reinterpret_cast<__gm__ T *>(reinterpret_cast<uint64_t>(GM_IN[rank]));
            auto output = reinterpret_cast<__gm__ T *>(output_ + rank * stride);
            WaitFlag(rank, rank, curTag_);
            CpGM2GM(output, gmOthers, count);
            PipeBarrier<PIPE_ALL>();
        }
    }
    uint64_t coreOffset;
    uint64_t curCount;
    uint64_t coreNumStage1;
    uint64_t coreNumStage2;
    uint64_t cutNum;
    uint64_t curStageCoreNum;
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