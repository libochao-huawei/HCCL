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
#include "aiv_kernel_omni.h"

using namespace AscendC;
using namespace ops_hccl;

template<typename T>
class AivOmniV2 : public AivCommBase {
public:
    __aicore__ inline AivOmniV2()
    {
    }

    __aicore__ inline void Process(uint64_t len, uint32_t tag, ExtraArgs &extraArgs)
    {
        if (numBlocks_ != 1 || block_idx != 0 || extraArgs.omniInfoAddr == 0) {
            return;
        }
        __gm__ AivOmniInfoHeader *header = reinterpret_cast<__gm__ AivOmniInfoHeader *>(extraArgs.omniInfoAddr);
        __gm__ AivOmniSendRecvInfo *infos = reinterpret_cast<__gm__ AivOmniSendRecvInfo *>(extraArgs.omniInfoAddr +
            sizeof(AivOmniInfoHeader));

        curTag_ = static_cast<int32_t>(tag);
        if (tag >> AIV_TAG_MOVE_RIGHT_BITS == 1 && (tag & LOW_16_BITS) == 1) {
            BarrierForFirstOP();
        }

        for (u64 infoIdx = 0; infoIdx < header->infoNum; infoIdx++) {
            const AivOmniSendRecvInfo &info = infos[infoIdx];
            curSliceCount_ = (info.sliceNum == 0) ? len : len / info.sliceNum;
            ExecuteInstruction(info);
            BarrierAll();
        }
    }

private:
    __aicore__ inline bool IsReduceOp(u32 opType) const
    {
        return opType == AIV_OMNI_OP_LOCAL_REDUCE ||
            opType == AIV_OMNI_OP_SEND_RECV_WRITE_REDUCE ||
            opType == AIV_OMNI_OP_SEND_WRITE_REDUCE ||
            opType == AIV_OMNI_OP_RECV_WRITE_REDUCE ||
            opType == AIV_OMNI_OP_SEND_RECV_READ_REDUCE ||
            opType == AIV_OMNI_OP_SEND_READ_REDUCE ||
            opType == AIV_OMNI_OP_RECV_READ_REDUCE ||
            opType == AIV_OMNI_OP_GROUP_REDUCE;
    }

    __aicore__ inline __gm__ T *ResolveSliceAddr(const AivOmniSliceInfo &slice, bool isSrc)
    {
        u64 baseAddr = 0;
        if (slice.sliceType == 0) {
            baseAddr = (slice.remoteRank == rank_ || isSrc) ? input_ : reinterpret_cast<u64>(GM_IN[slice.remoteRank]);
        } else if (slice.sliceType == 1) {
            baseAddr = (slice.remoteRank == rank_ || !isSrc) ? output_ : reinterpret_cast<u64>(GM_IN[slice.remoteRank]);
        } else {
            const u64 remoteRank = (slice.remoteRank < rankSize_) ? slice.remoteRank : rank_;
            baseAddr = reinterpret_cast<u64>(GM_IN[remoteRank]);
        }
        return reinterpret_cast<__gm__ T *>(baseAddr + slice.sliceIdx * curSliceCount_ * sizeof(T));
    }

    __aicore__ inline void CopySlice(const AivOmniSliceInfo &src, const AivOmniSliceInfo &dst, u32 reduceType, bool reduce)
    {
        if (curSliceCount_ == 0) {
            return;
        }
        __gm__ T *srcAddr = ResolveSliceAddr(src, true);
        __gm__ T *dstAddr = ResolveSliceAddr(dst, false);
        if (reduce) {
            CpGM2GM(dstAddr, srcAddr, curSliceCount_, reduceType);
        } else {
            CpGM2GM(dstAddr, srcAddr, curSliceCount_);
        }
        PipeBarrier<PIPE_ALL>();
    }

    __aicore__ inline void ExecutePairwise(const AivOmniSendRecvInfo &info)
    {
        const bool reduce = IsReduceOp(info.opType);
        if (info.srcSliceNum == 0 || info.dstSliceNum == 0) {
            return;
        }
        if (info.opType == AIV_OMNI_OP_GROUP_BROAD_CAST) {
            for (u32 dstIdx = 0; dstIdx < info.dstSliceNum; dstIdx++) {
                CopySlice(info.srcSliceInfo[0], info.dstSliceInfo[dstIdx], info.reduceType, false);
            }
            return;
        }
        if (info.opType == AIV_OMNI_OP_GROUP_REDUCE) {
            CopySlice(info.srcSliceInfo[0], info.dstSliceInfo[0], info.reduceType, false);
            for (u32 srcIdx = 1; srcIdx < info.srcSliceNum; srcIdx++) {
                CopySlice(info.srcSliceInfo[srcIdx], info.dstSliceInfo[0], info.reduceType, true);
            }
            return;
        }
        if (info.srcSliceNum == 1 && info.dstSliceNum > 1) {
            for (u32 dstIdx = 0; dstIdx < info.dstSliceNum; dstIdx++) {
                CopySlice(info.srcSliceInfo[0], info.dstSliceInfo[dstIdx], info.reduceType, reduce);
            }
            return;
        }
        if (info.dstSliceNum == 1 && info.srcSliceNum > 1) {
            CopySlice(info.srcSliceInfo[0], info.dstSliceInfo[0], info.reduceType, false);
            for (u32 srcIdx = 1; srcIdx < info.srcSliceNum; srcIdx++) {
                CopySlice(info.srcSliceInfo[srcIdx], info.dstSliceInfo[0], info.reduceType, reduce);
            }
            return;
        }

        const u32 pairCount = info.srcSliceNum < info.dstSliceNum ? info.srcSliceNum : info.dstSliceNum;
        for (u32 pairIdx = 0; pairIdx < pairCount; pairIdx++) {
            CopySlice(info.srcSliceInfo[pairIdx], info.dstSliceInfo[pairIdx], info.reduceType, reduce);
        }
    }

    __aicore__ inline void ExecuteInstruction(const AivOmniSendRecvInfo &info)
    {
        ExecutePairwise(info);
    }

    int32_t curTag_ = 0;
    u64 curSliceCount_ = 0;
};

template<typename T>
__aicore__ inline void AivOmniV2Entry(EXTERN_KERNEL_ARGS_DEF_V2)
{
    AivOmniV2<T> op;
    op.Init(KERNEL_CLASS_INIT, true);
    SyncAll<true>();
    op.Process(len, tag, extraArgs);
    op.BarrierAll();
}
