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
 
// todo 简化参数
 
class AivBroadcastMesh1D : public AivCommBase {

public:
    __aicore__ inline AivBroadcastMesh1D() {}

    template<typename T>
    __aicore__ inline void Process(uint64_t curCount, uint64_t sliceId, uint64_t stride);

    template<typename T>
    __aicore__ inline void ProcessBigData(uint64_t curCount, uint64_t sliceId, uint64_t stride);

private:
    template<typename T>
    __aicore__ inline void ScatterPhase(bool isMidData16p, uint64_t usedCoreNumAll, 
                                        uint64_t countPerCore, __gm__ T *rootCclGM, __gm__ T *inputGM,
                                        __gm__ T *cclGM, uint32_t tag) {
        if (rank_ == root_) {
            CpGM2GM(rootCclGM, inputGM, countPerCore);
            PipeBarrier<PIPE_ALL>();
            if (isMidData16p) {
                Record(root_, blockIdx_, tag);
                for (uint32_t i = 0; i < rankSize_; i++) {
                    Record(root_, usedCoreNumAll + blockIdx_ * rankSize_ + i, tag);
                }
            } else {
                for (uint32_t i = 0; i < rankSize_; i++) {
                    Record(i, blockIdx_, tag);
                    Record(i, usedCoreNumAll + blockIdx_, tag);
                }
            }
        } else if (blockIdx_ / (numBlocks_ / rankSize_) == rank_) {
            WaitFlag(isMidData16p ? root_ : rank_, blockIdx_, tag);
            CpGM2GM(cclGM, rootCclGM, countPerCore);
            PipeBarrier<PIPE_ALL>();
            if (isMidData16p) {
                for (uint32_t i = 0; i < rankSize_; i++) {
                    Record(rank_, usedCoreNumAll + blockIdx_ * rankSize_ + i, tag);
                }
            } else {
                for (uint32_t i = 0; i < rankSize_; i++) {
                    Record(i, usedCoreNumAll + blockIdx_, tag);
                }
            }
        }
    }

    template<typename T>
    __aicore__ inline void AllgatherPhase(bool isMidData16p, uint64_t usedCoreNumAll,
                                          uint64_t countPerCore, __gm__ T *inputGM, __gm__ T *cclGM, uint32_t tag) {
        if (rank_ != root_) {
            uint64_t peerRank = blockIdx_ / (numBlocks_ / rankSize_);
            uint64_t waitFlag = isMidData16p ? usedCoreNumAll + blockIdx_ * rankSize_ + rank_ : usedCoreNumAll + blockIdx_;
            uint32_t waitRank = isMidData16p ? peerRank : rank_;
            WaitFlag(waitRank, waitFlag, tag);
            CpGM2GM(inputGM, cclGM, countPerCore);
            PipeBarrier<PIPE_ALL>();
        }
    }
};
 
template<typename T>
__aicore__ inline void AivBroadcastMesh1D::Process(uint64_t curCount, uint64_t sliceId, uint64_t stride)
{
    curTag_ = (static_cast<uint32_t>(tag_) << AIV_TAG_MOVE_RIGHT_BITS) | (sliceId & LOW_16_BITS);
    uint64_t dataTypeSize = sizeof(T);
    uint64_t coreNumPerRank = 1;
    uint64_t curStageCoreNum = coreNumPerRank * rankSize_;
    if (blockIdx_ >= curStageCoreNum) {
        return;
    }

    uint64_t peerRank = blockIdx_ / coreNumPerRank;
    uint64_t offsetPerCore = curCount / curStageCoreNum * dataTypeSize;
    uint64_t dataOffset = offsetPerCore * blockIdx_;
    uint64_t countPerCore = blockIdx_ == curStageCoreNum - 1 ? curCount - (curStageCoreNum - 1) * (curCount / curStageCoreNum)
                                    : curCount / curStageCoreNum;
    uint64_t flag_offset = blockIdx_;
    __gm__ T *inputGM = (__gm__ T *)(input_ + dataOffset);
    __gm__ T *cclGM = (__gm__ T *)(GM_IN[peerRank] + dataOffset);
    // scatter
    if (rank_ == root_) {
        CpGM2GM(cclGM, inputGM, countPerCore);
        PipeBarrier<PIPE_ALL>();
        // 避免多核同时访问一个flag
        for (uint32_t i = 0; i < rankSize_; i++) {
            Record(i, flag_offset, curTag_);
        }
    }
 
    // allgather
    WaitFlag(rank_, flag_offset, curTag_);
    CpGM2GM(inputGM, cclGM, countPerCore);
    PipeBarrier<PIPE_ALL>();
}

template<typename T>
__aicore__ inline void AivBroadcastMesh1D::ProcessBigData(uint64_t curCount, uint64_t sliceId, uint64_t stride)
{
    curTag_ = (static_cast<uint32_t>(tag_) << AIV_TAG_MOVE_RIGHT_BITS) | (sliceId & LOW_16_BITS);
    uint64_t dataTypeSize = sizeof(T);
    uint64_t coreNumPerRank = numBlocks_ / rankSize_;
    uint64_t usedCoreNumAll = numBlocks_ / rankSize_ * rankSize_;
    bool isMidData16p = rankSize_ == 16 && curCount * dataTypeSize >= 8 * 1024 * 1024 &&
                        curCount * dataTypeSize <= 16 * 1024 * 1024;
    if (blockIdx_ >= usedCoreNumAll) {
        return;
    }
    uint64_t offsetPerCore = curCount / usedCoreNumAll * dataTypeSize;
    uint64_t dataOffset = offsetPerCore * blockIdx_;
    uint64_t countPerCore = blockIdx_ == usedCoreNumAll - 1 ? curCount - (usedCoreNumAll - 1) * (curCount / usedCoreNumAll)
                                    : curCount / usedCoreNumAll;
    __gm__ T *inputGM = (__gm__ T *)(input_ + dataOffset);
    __gm__ T *cclGM = (__gm__ T *)(GM_IN[blockIdx_ / coreNumPerRank] + dataOffset);
    __gm__ T *rootCclGM = (__gm__ T *)(GM_IN[root_] + dataOffset);

    ScatterPhase<T>(isMidData16p, usedCoreNumAll, countPerCore, rootCclGM, inputGM, cclGM, curTag_);
    AllgatherPhase<T>(isMidData16p, usedCoreNumAll, countPerCore, inputGM, cclGM, curTag_);
}
 
template<typename T>
__aicore__ inline void AivBroadcastV2Mesh1D(KERNEL_ARGS_DEF)
{
    AivBroadcastMesh1D op;
    op.Init(KERNEL_CLASS_INIT, true);
    if (op.IsFirstOP(sliceId)) {
        op.BarrierForFirstOP();
    }
    if (len * sizeof(T) >= DATA_LIMIT) {
        op.ProcessBigData<T>(len, sliceId, inputSliceStride);
    } else {
        op.Process<T>(len, sliceId, inputSliceStride);
    }
    op.BarrierAll();
}