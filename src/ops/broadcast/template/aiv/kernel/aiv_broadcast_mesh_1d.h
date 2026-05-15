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
    constexpr static uint64_t CORE_NUMS_ALL = 2;
 
public:
    __aicore__ inline AivBroadcastMesh1D() {}
 
    template<typename T>
    __aicore__ inline void Process(uint64_t curCount, uint64_t sliceId);

    template<typename T>
    __aicore__ inline void ProcessBigData(uint64_t curCount, uint64_t sliceId);
};
 
template<typename T>
__aicore__ inline void AivBroadcastMesh1D::ProcessBigData(uint64_t curCount, uint64_t sliceId)
{
    curTag_ = (static_cast<uint32_t>(tag_) << AIV_TAG_MOVE_RIGHT_BITS) | (sliceId & LOW_16_BITS);
    uint64_t dataTypeSize = sizeof(T);
    uint64_t usedCoreNumsAll = numBlocks_ / rankSize_ * rankSize_;
    if (block_idx >= usedCoreNumsAll) {
        return;
    }
    uint64_t coreNumPerRank = usedCoreNumsAll / rankSize_;
    uint32_t peerRank = block_idx / coreNumPerRank;
    uint64_t offsetPerCore = curCount / usedCoreNumsAll * dataTypeSize;
    uint64_t dataOffset = offsetPerCore * block_idx;
    uint64_t countPerCore = block_idx == usedCoreNumsAll - 1 ? curCount - (usedCoreNumsAll - 1) * (curCount / usedCoreNumsAll)
                                    : curCount / usedCoreNumsAll;
    uint64_t flag_offset = block_idx;
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
__aicore__ inline void AivBroadcastMesh1D::Process(uint64_t curCount, uint64_t sliceId) {
    curTag_ = (static_cast<uint32_t>(tag_) << AIV_TAG_MOVE_RIGHT_BITS) | (sliceId & LOW_16_BITS);
    uint64_t dataTypeSize = sizeof(T);
    uint64_t usedCoreNumsAll = 2;
    if (block_idx >= usedCoreNumsAll) {
        return;
    }
    uint64_t offsetPerCore = curCount / usedCoreNumsAll * dataTypeSize;
    uint64_t dataOffset = offsetPerCore * block_idx;
    uint64_t countPerCore = block_idx == usedCoreNumsAll - 1 ? curCount - (usedCoreNumsAll - 1) * (curCount / usedCoreNumsAll)
                                    : curCount / usedCoreNumsAll;
    uint64_t flag_offset = block_idx;
    __gm__ T *inputGM = (__gm__ T *)(input_ + dataOffset);

    if (rank_ == root_) {
        for (uint32_t i = 0; i < rankSize_; i++) {
            if (i == root_) {
                continue;
            }
            __gm__ T *cclGM = (__gm__ T *)(GM_IN[i] + dataOffset);
            CpGM2GM(cclGM, inputGM, countPerCore);
            PipeBarrier<PIPE_ALL>();
        }
        // 避免多核同时访问一个flag
        for (uint32_t i = 0; i < rankSize_; i++) {
            Record(i, flag_offset, curTag_);
        }
    } else {
        WaitFlag(rank_, flag_offset, curTag_);
    }
}
 
template<typename T>
__aicore__ inline void AivBroadcastV2Mesh1D(EXTERN_KERNEL_ARGS_DEF_V2)
{
    AivBroadcastMesh1D op;
    op.Init(KERNEL_CLASS_INIT, true);
    SyncAll<true>();
    if (op.IsFirstOP(sliceId)) {
        op.BarrierForFirstOP();
    }
    SyncAll<true>();
    if (len * sizeof(T) >= DATA_LIMIT) {
        op.ProcessBigData<T>(len, sliceId);
    } else {
        op.Process<T>(len, sliceId);
    }
    op.BarrierAll();
}