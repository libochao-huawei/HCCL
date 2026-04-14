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
class AivReduceMesh1DTwoShot : public AivCommBase {
public:
    __aicore__ inline AivReduceMesh1DTwoShot()
    {
    }

    uint32_t coreNumPerRank;
    uint32_t coreNumFirstStage;
    uint32_t coreNumTotal;
    uint32_t innerId;
    uint32_t targetRank;
    uint64_t rankChunkSize;
    uint64_t innerChunkSize;
    uint64_t rankChunkStride;
    uint64_t innerChunkStride;
    uint64_t ipcReduceFlagOffset {1024};

    __aicore__ inline void InitCoreInfo(uint32_t stepTag)
    {
        curTag_ = (static_cast<uint32_t>(tag_) << AIV_TAG_MOVE_RIGHT_BITS) | (stepTag & LOW_16_BITS);
        uint64_t dataCount = len_;
        coreNumPerRank = numBlocks_ / (rankSize_ + 1);
        coreNumFirstStage = coreNumPerRank * rankSize_;
        coreNumTotal = coreNumPerRank * (rankSize_ + 1);

        innerChunkStride = RoundUpAiv(dataCount, coreNumFirstStage);
        rankChunkStride = innerChunkStride * coreNumPerRank;
        if (GetBlockIdx() < coreNumFirstStage) {
            targetRank = GetBlockIdx() / coreNumPerRank;
            rankChunkSize = ((targetRank + 1) * rankChunkStride <= dataCount)
                ? rankChunkStride
                : (dataCount <= targetRank * rankChunkStride ? 0 : (dataCount - targetRank * rankChunkStride));
        } else if (GetBlockIdx() < coreNumTotal) {
            targetRank = rank_;
            rankChunkSize = ((rank_ + 1) * rankChunkStride <= dataCount)
                ? rankChunkStride
                : (dataCount <= rank_ * rankChunkStride ? 0 : (dataCount - rank_ * rankChunkStride));
        }

        if (GetBlockIdx() < coreNumTotal) {
            innerId = GetBlockIdx() % coreNumPerRank;
            innerChunkSize = ((innerId + 1) * innerChunkStride <= rankChunkSize)
                ? innerChunkStride
                : (rankChunkSize <= innerId * innerChunkStride ? 0 : (rankChunkSize - innerId * innerChunkStride));
        }
        ipcReduceFlagOffset = coreNumFirstStage;
    }

    __aicore__ inline void ReduceScatter()
    {
        if (GetBlockIdx() < coreNumFirstStage) {
            if (innerChunkSize > 0) {
                uint64_t inputOffset =
                    input_ + (targetRank * rankChunkStride + innerId * innerChunkStride) * sizeof(T);
                uint64_t outputOffset =
                    reinterpret_cast<uint64_t>(GM_IN[targetRank]) + (rank_ * rankChunkSize + innerId * innerChunkStride) * sizeof(T);
                CpGM2GM((__gm__ T *)outputOffset, (__gm__ T *)inputOffset, innerChunkSize);
                pipe_barrier(PIPE_ALL);
            }
            Record(targetRank, rank_ * coreNumPerRank + innerId, curTag_);
        } else if (GetBlockIdx() < coreNumTotal) {
            if (innerChunkSize > 0) {
                for (uint32_t i = 0; i < rankSize_; i++) {
                    WaitFlag(rank_, i * coreNumPerRank + innerId, curTag_);
                    if (i == 0) {
                        continue;
                    }
                    uint64_t inputOffset =
                        reinterpret_cast<uint64_t>(GM_IN[rank_]) + (i * rankChunkSize + innerId * innerChunkStride) * sizeof(T);
                    uint64_t outputOffset =
                        reinterpret_cast<uint64_t>(GM_IN[rank_]) + (innerId * innerChunkStride) * sizeof(T);
                    CpGM2GM((__gm__ T *)outputOffset, (__gm__ T *)inputOffset, innerChunkSize, reduceOp_);
                    pipe_barrier(PIPE_ALL);
                }
            }
            Record(rank_, ipcReduceFlagOffset + innerId, curTag_);
        }
    }

    __aicore__ inline void GatherToRoot()
    {
        if (rank_ != root_ || GetBlockIdx() >= coreNumFirstStage || innerChunkSize == 0) {
            return;
        }
        WaitFlag(targetRank, ipcReduceFlagOffset + innerId, curTag_);
        uint64_t inputOffset =
            reinterpret_cast<uint64_t>(GM_IN[targetRank]) + (innerId * innerChunkStride) * sizeof(T);
        uint64_t outputOffset = output_ + (targetRank * rankChunkStride + innerId * innerChunkStride) * sizeof(T);
        CpGM2GM((__gm__ T *)outputOffset, (__gm__ T *)inputOffset, innerChunkSize);
        pipe_barrier(PIPE_ALL);
    }

    __aicore__ inline uint64_t RoundUpAiv(uint64_t dividend, uint64_t divisor)
    {
        return dividend / divisor + ((dividend % divisor != 0) ? 1 : 0);
    }

    __aicore__ inline void SmallCoreReduceScatter(uint32_t stepTag)
    {
        uint64_t dataCount = len_;
        rankChunkStride = RoundUpAiv(dataCount, rankSize_);
        ipcReduceFlagOffset = rankSize_;
        curTag_ = (static_cast<uint32_t>(tag_) << AIV_TAG_MOVE_RIGHT_BITS) | (stepTag & LOW_16_BITS);

        for (uint32_t i = 0; block_idx + i * numBlocks_ < rankSize_; i++) {
            targetRank = block_idx + i * numBlocks_;
            rankChunkSize = ((targetRank + 1) * rankChunkStride <= dataCount)
                ? rankChunkStride
                : (dataCount <= targetRank * rankChunkStride ? 0 : (dataCount - targetRank * rankChunkStride));

            if (rankChunkSize > 0) {
                uint64_t inputOffset = input_ + (targetRank * rankChunkStride) * sizeof(T);
                uint64_t outputOffset = reinterpret_cast<uint64_t>(GM_IN[targetRank]) + (rank_ * rankChunkSize) * sizeof(T);
                CpGM2GM((__gm__ T *)outputOffset, (__gm__ T *)inputOffset, rankChunkSize);
                pipe_barrier(PIPE_ALL);
            }
            Record(targetRank, rank_, curTag_);
        }

        if (block_idx == numBlocks_ - 1) {
            uint64_t myRankChunkSize = ((rank_ + 1) * rankChunkStride <= dataCount)
                ? rankChunkStride
                : (dataCount <= rank_ * rankChunkStride ? 0 : (dataCount - rank_ * rankChunkStride));
            if (myRankChunkSize > 0) {
                for (uint32_t i = 0; i < rankSize_; i++) {
                    WaitFlag(rank_, i, curTag_);
                    if (i == 0) {
                        continue;
                    }
                    uint64_t inputOffset =
                        reinterpret_cast<uint64_t>(GM_IN[rank_]) + (i * myRankChunkSize) * sizeof(T);
                    uint64_t outputOffset = reinterpret_cast<uint64_t>(GM_IN[rank_]);
                    CpGM2GM((__gm__ T *)outputOffset, (__gm__ T *)inputOffset, myRankChunkSize, reduceOp_);
                    pipe_barrier(PIPE_ALL);
                }
            }
            Record(rank_, ipcReduceFlagOffset + 1, curTag_);
        }
    }

    __aicore__ inline void SmallCoreGatherToRoot()
    {
        if (rank_ != root_) {
            return;
        }

        uint64_t dataCount = len_;
        for (uint32_t i = 0; block_idx + i * numBlocks_ < rankSize_; i++) {
            targetRank = block_idx + i * numBlocks_;
            rankChunkSize = ((targetRank + 1) * rankChunkStride <= dataCount)
                ? rankChunkStride
                : (dataCount <= targetRank * rankChunkStride ? 0 : (dataCount - targetRank * rankChunkStride));

            if (rankChunkSize == 0) {
                continue;
            }

            WaitFlag(targetRank, ipcReduceFlagOffset + 1, curTag_);
            uint64_t inputOffset = reinterpret_cast<uint64_t>(GM_IN[targetRank]);
            uint64_t outputOffset = output_ + (targetRank * rankChunkStride) * sizeof(T);
            CpGM2GM((__gm__ T *)outputOffset, (__gm__ T *)inputOffset, rankChunkSize);
            pipe_barrier(PIPE_ALL);
        }
    }
};

template<typename T>
__aicore__ inline void AivReduceV2Mesh1DTwoShot(EXTERN_KERNEL_ARGS_DEF_V2)
{
    (void)extraArgs;
    AivReduceMesh1DTwoShot<T> op;
    op.Init(KERNEL_CLASS_INIT, true);
    SyncAll<true>();
    if (op.IsFirstOP(sliceId)) {
        op.BarrierForFirstOP();
    }
    SyncAll<true>();
    if (rankSize + 1 <= block_num) {
        op.InitCoreInfo(sliceId);
        op.ReduceScatter();
        op.GatherToRoot();
    } else {
        op.SmallCoreReduceScatter(sliceId);
        op.SmallCoreGatherToRoot();
    }
    op.BarrierAll();
}
