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

template<bool Value>
struct OmniBoolType {};

template<typename T>
struct OmniIsAtomicReduceType {
    using type = OmniBoolType<false>;
};

template<>
struct OmniIsAtomicReduceType<float> {
    using type = OmniBoolType<true>;
};

template<>
struct OmniIsAtomicReduceType<half> {
    using type = OmniBoolType<true>;
};

template<>
struct OmniIsAtomicReduceType<int16_t> {
    using type = OmniBoolType<true>;
};

template<>
struct OmniIsAtomicReduceType<int32_t> {
    using type = OmniBoolType<true>;
};

template<>
struct OmniIsAtomicReduceType<int8_t> {
    using type = OmniBoolType<true>;
};

template<>
struct OmniIsAtomicReduceType<bfloat16_t> {
    using type = OmniBoolType<true>;
};

template<typename T>
class AivOmniV2 : public AivCommBase {
public:
    __aicore__ inline AivOmniV2()
    {
    }

    __aicore__ inline void Process(uint64_t len, uint32_t sliceId, ExtraArgs &extraArgs)
    {
        if (extraArgs.omniInfoAddr == 0) {
            return;
        }
        __gm__ AivOmniInfoHeader *header = reinterpret_cast<__gm__ AivOmniInfoHeader *>(extraArgs.omniInfoAddr);
        __gm__ AivOmniSendRecvInfo *infos = reinterpret_cast<__gm__ AivOmniSendRecvInfo *>(extraArgs.omniInfoAddr +
            sizeof(AivOmniInfoHeader));

        curTag_ = (static_cast<uint32_t>(tag_) << AIV_TAG_MOVE_RIGHT_BITS) | (sliceId & LOW_16_BITS);
        if (IsFirstOP(sliceId)) {
            BarrierForFirstOP();
        }

        for (uint64_t infoIdx = 0; infoIdx < header->infoNum; infoIdx++) {
            AivOmniSendRecvInfo info = {};
            LoadInfoFromGm(info, infos + infoIdx);
            curSliceCount_ = (info.sliceNum == 0) ? len : len / info.sliceNum;
            ExecuteInstruction(info);
            BarrierAll();
        }
    }

private:
    __aicore__ inline void LoadInfoFromGm(AivOmniSendRecvInfo &dst, const __gm__ AivOmniSendRecvInfo *src)
    {
        dst.opType = src->opType;
        dst.inputDataType = src->inputDataType;
        dst.outputDataType = src->outputDataType;
        dst.reduceType = src->reduceType;
        dst.srcSliceNum = src->srcSliceNum;
        dst.dstSliceNum = src->dstSliceNum;
        dst.sliceNum = src->sliceNum;
        dst.linkType = src->linkType;
        dst.threadIdx = src->threadIdx;
        for (uint32_t idx = 0; idx < AIV_OMNI_MAX_SLICE_CNT; idx++) {
            dst.srcSliceInfo[idx].sliceType = src->srcSliceInfo[idx].sliceType;
            dst.srcSliceInfo[idx].sliceIdx = src->srcSliceInfo[idx].sliceIdx;
            dst.srcSliceInfo[idx].remoteRank = src->srcSliceInfo[idx].remoteRank;
            dst.dstSliceInfo[idx].sliceType = src->dstSliceInfo[idx].sliceType;
            dst.dstSliceInfo[idx].sliceIdx = src->dstSliceInfo[idx].sliceIdx;
            dst.dstSliceInfo[idx].remoteRank = src->dstSliceInfo[idx].remoteRank;
        }
    }

    __aicore__ inline void CopySliceReduce(__gm__ T *dstAddr, __gm__ T *srcAddr, uint32_t reduceType, OmniBoolType<true>)
    {
        CpGM2GM(dstAddr, srcAddr, curSliceCount_, reduceType);
    }

    __aicore__ inline void CopySliceReduce(__gm__ T *dstAddr, __gm__ T *srcAddr, uint32_t reduceType, OmniBoolType<false>)
    {
        (void)reduceType;
        // OMNI first version falls back to copy for data types without AIV atomic reduce support.
        CpGM2GM(dstAddr, srcAddr, curSliceCount_);
    }

    __aicore__ inline bool IsReduceOp(uint32_t opType) const
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
        uint64_t baseAddr = 0;
        uint64_t effectiveSliceIdx = slice.sliceIdx;
        if (slice.sliceType == AIV_OMNI_BUFFER_HCCL) {
            baseAddr = reinterpret_cast<uint64_t>(GM_IN[slice.remoteRank]);
            effectiveSliceIdx = 0;
        } else if (slice.sliceType == AIV_OMNI_BUFFER_INPUT) {
            baseAddr = input_;
        } else if (slice.sliceType == AIV_OMNI_BUFFER_OUTPUT) {
            baseAddr = output_;
        } else {
            const uint64_t remoteRank = (slice.remoteRank < rankSize_) ? slice.remoteRank : rank_;
            baseAddr = reinterpret_cast<uint64_t>(GM_IN[remoteRank]);
            effectiveSliceIdx = 0;
        }
        return reinterpret_cast<__gm__ T *>(baseAddr + effectiveSliceIdx * curSliceCount_ * sizeof(T));
    }

    __aicore__ inline bool ShouldHandleSlice(const AivOmniSliceInfo &dstSlice) const
    {
        if (numBlocks_ <= 1) {
            return true;
        }
        return (dstSlice.sliceIdx % static_cast<uint64_t>(numBlocks_)) == static_cast<uint64_t>(GetBlockIdx());
    }

    __aicore__ inline void CopySlice(const AivOmniSliceInfo &src, const AivOmniSliceInfo &dst, uint32_t reduceType, bool reduce)
    {
        if (curSliceCount_ == 0) {
            return;
        }
        if (!ShouldHandleSlice(dst)) {
            return;
        }
        __gm__ T *srcAddr = ResolveSliceAddr(src, true);
        __gm__ T *dstAddr = ResolveSliceAddr(dst, false);
        if (reduce) {
            CopySliceReduce(dstAddr, srcAddr, reduceType, typename OmniIsAtomicReduceType<T>::type{});
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
            for (uint32_t dstIdx = 0; dstIdx < info.dstSliceNum; dstIdx++) {
                CopySlice(info.srcSliceInfo[0], info.dstSliceInfo[dstIdx], info.reduceType, false);
            }
            return;
        }
        if (info.opType == AIV_OMNI_OP_GROUP_REDUCE) {
            CopySlice(info.srcSliceInfo[0], info.dstSliceInfo[0], info.reduceType, false);
            for (uint32_t srcIdx = 1; srcIdx < info.srcSliceNum; srcIdx++) {
                CopySlice(info.srcSliceInfo[srcIdx], info.dstSliceInfo[0], info.reduceType, true);
            }
            return;
        }
        if (info.srcSliceNum == 1 && info.dstSliceNum > 1) {
            for (uint32_t dstIdx = 0; dstIdx < info.dstSliceNum; dstIdx++) {
                CopySlice(info.srcSliceInfo[0], info.dstSliceInfo[dstIdx], info.reduceType, reduce);
            }
            return;
        }
        if (info.dstSliceNum == 1 && info.srcSliceNum > 1) {
            CopySlice(info.srcSliceInfo[0], info.dstSliceInfo[0], info.reduceType, false);
            for (uint32_t srcIdx = 1; srcIdx < info.srcSliceNum; srcIdx++) {
                CopySlice(info.srcSliceInfo[srcIdx], info.dstSliceInfo[0], info.reduceType, reduce);
            }
            return;
        }

        const uint32_t pairCount = info.srcSliceNum < info.dstSliceNum ? info.srcSliceNum : info.dstSliceNum;
        for (uint32_t pairIdx = 0; pairIdx < pairCount; pairIdx++) {
            CopySlice(info.srcSliceInfo[pairIdx], info.dstSliceInfo[pairIdx], info.reduceType, reduce);
        }
    }

    __aicore__ inline void ExecuteInstruction(const AivOmniSendRecvInfo &info)
    {
        ExecutePairwise(info);
    }

    int32_t curTag_ = 0;
    uint64_t curSliceCount_ = 0;
};

template<typename T>
__aicore__ inline void AivOmniV2Entry(EXTERN_KERNEL_ARGS_DEF_V2)
{
    AivOmniV2<T> op;
    op.Init(KERNEL_CLASS_INIT, true);
    SyncAll<true>();
    // op.Process(len, sliceId, extraArgs);
    op.BarrierAll();
}
