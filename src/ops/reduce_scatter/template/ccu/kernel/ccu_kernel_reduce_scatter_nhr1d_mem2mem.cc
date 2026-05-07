/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "ccu_kernel_reduce_scatter_nhr1d_mem2mem.h"

namespace ops_hccl {

constexpr uint16_t INPUT_XN_ID      = 0;
constexpr uint16_t TOKEN_XN_ID      = 1;
constexpr uint16_t POST_SYNC_ID     = 2;
constexpr uint16_t STEP_PRE_SYNC_ID = 3;
constexpr uint16_t STEP_POST_SYNC_ID= 4;
constexpr uint16_t CKE_IDX_0        = 0;
constexpr uint16_t LINK_SIZE        = 2;

static CcuResult ParseKernelArg(ReduceScatterNHR1DMem2MemContext &ctx, CcuKernelArgReduceScatterNHR1D *kernelArg)
{
    ctx.mySubCommRankId = kernelArg->mySubCommRankId_;
    ctx.axisId          = kernelArg->axisId_;
    ctx.dimSize         = kernelArg->dimSize_;
    ctx.stepInfoVector  = kernelArg->stepInfoVector_;
    ctx.rank2ChannelIdx = kernelArg->rank2ChannelIdx_;
    ctx.localSize       = ctx.rank2ChannelIdx.size();
    ctx.myRankIdx       = ctx.rank2ChannelIdx.size();
    ctx.reduceOp        = kernelArg->opParam_.reduceType;
    ctx.dataType        = kernelArg->opParam_.DataDes.dataType;
    ctx.outputDataType  = kernelArg->opParam_.DataDes.outputType;
    ctx.axisSize        = kernelArg->axisSize_;

    if (ctx.outputDataType == HcclDataType::HCCL_DATA_TYPE_RESERVED) {
        ctx.outputDataType = ctx.dataType;
        HCCL_DEBUG("[CcuKernelReduceScatterNHR1DMem2Mem] outputDataType is [INVALID], set outputDataType to[%d]",
                   ctx.outputDataType);
    }

    HCCL_INFO("[CcuKernelReduceScatterNHR1DMem2Mem] KernelArg: mySubCommRankId_[%u], myRankIdx_[%d], axisId_[%u], dimSize_[%u], localSize_[%u], "
              "dataType[%d], outputDataType[%d], reduceOp[%d]",
              ctx.mySubCommRankId, ctx.myRankIdx, ctx.axisId, ctx.dimSize, ctx.localSize, ctx.dataType, ctx.outputDataType, ctx.reduceOp);
    return CCU_SUCCESS;
}

static CcuResult InitResources(ReduceScatterNHR1DMem2MemContext &ctx)
{
    const auto *arg = ctx.arg;

    CCU_CHK_RET(ccu::Alloc(&ctx.die0Size));
    CCU_CHK_RET(ccu::Alloc(&ctx.die1Size));
    CCU_CHK_RET(ccu::Alloc(&ctx.die0LastSliceSize));
    CCU_CHK_RET(ccu::Alloc(&ctx.die1LastSliceSize));
    CCU_CHK_RET(ccu::Alloc(&ctx.sliceSize));
    CCU_CHK_RET(ccu::Alloc(&ctx.inputSliceStride));
    CCU_CHK_RET(ccu::Alloc(&ctx.currentRankSliceOutputOffset));
    CCU_CHK_RET(ccu::Alloc(&ctx.inputRepeatStride));
    CCU_CHK_RET(ccu::Alloc(&ctx.outputRepeatStride));
    CCU_CHK_RET(ccu::Alloc(&ctx.repeatNumVar));
    CCU_CHK_RET(ccu::Alloc(&ctx.repeatNumVarTemp));
    CCU_CHK_RET(ccu::Alloc(&ctx.isInputOutputEqual));
    CCU_CHK_RET(ccu::Alloc(&ctx.event));

    CCU_CHK_RET(ccu::Alloc(&ctx.output));

    ctx.input.resize(ctx.localSize + 1);
    ctx.token.resize(ctx.localSize + 1);

    for (uint32_t channelIdx = 0; channelIdx < ctx.localSize; channelIdx++) {
        HCCL_INFO("[CcuKernelReduceScatterNHR1DMem2Mem] mySubCommRankId[%u], channelId[%u]", ctx.mySubCommRankId, channelIdx);
        CCU_CHK_RET(ccu::CreateByChannel(arg->channels[channelIdx], INPUT_XN_ID, &ctx.input[channelIdx]));
        CCU_CHK_RET(ccu::CreateByChannel(arg->channels[channelIdx], TOKEN_XN_ID, &ctx.token[channelIdx]));
    }
    CCU_CHK_RET(ccu::Alloc(&ctx.input[ctx.localSize]));
    CCU_CHK_RET(ccu::Alloc(&ctx.token[ctx.localSize]));

    CCU_CHK_RET(ccu::Alloc(&ctx.repeatInputOffset));
    CCU_CHK_RET(ccu::Alloc(&ctx.repeatOutputOffset));

    CCU_CHK_RET(ccu::Alloc(&ctx.localSrc));
    CCU_CHK_RET(ccu::Alloc(&ctx.localDst));
    CCU_CHK_RET(ccu::Alloc(&ctx.remoteDst));
    CCU_CHK_RET(ccu::Alloc(&ctx.isRepeatIter));

    HCCL_INFO("[CcuKernelReduceScatterNHR1DMem2Mem] InitResources finished");
    return CCU_SUCCESS;
}

static CcuResult LoadArgs(ReduceScatterNHR1DMem2MemContext &ctx)
{
    CCU_CHK_RET(ccu::LoadArg(ctx.input[ctx.myRankIdx]));
    CCU_CHK_RET(ccu::LoadArg(ctx.output));
    CCU_CHK_RET(ccu::LoadArg(ctx.token[ctx.myRankIdx]));
    CCU_CHK_RET(ccu::LoadArg(ctx.die0Size));
    CCU_CHK_RET(ccu::LoadArg(ctx.die1Size));
    CCU_CHK_RET(ccu::LoadArg(ctx.die0LastSliceSize));
    CCU_CHK_RET(ccu::LoadArg(ctx.die1LastSliceSize));
    CCU_CHK_RET(ccu::LoadArg(ctx.inputSliceStride));
    CCU_CHK_RET(ccu::LoadArg(ctx.currentRankSliceOutputOffset));
    CCU_CHK_RET(ccu::LoadArg(ctx.inputRepeatStride));
    CCU_CHK_RET(ccu::LoadArg(ctx.outputRepeatStride));
    CCU_CHK_RET(ccu::LoadArg(ctx.repeatNumVar));
    CCU_CHK_RET(ccu::LoadArg(ctx.isInputOutputEqual));

    ctx.repeatNumVarTemp = ctx.repeatNumVar;
    HCCL_INFO("[CcuKernelReduceScatterNHR1DMem2Mem] LoadArgs run finished");
    return CCU_SUCCESS;
}

static void PreSync(ReduceScatterNHR1DMem2MemContext &ctx)
{
    const auto *arg = ctx.arg;
    HCCL_INFO("[CcuKernelReduceScatterNHR1DMem2Mem] PreSync start");

    for (uint32_t i = 0; i < arg->channelCount; i++) {
        ccu::WriteVariableWithNotify(arg->channels[i], ctx.input[ctx.myRankIdx], INPUT_XN_ID, CKE_IDX_0, 1 << INPUT_XN_ID);
        ccu::WriteVariableWithNotify(arg->channels[i], ctx.token[ctx.myRankIdx], TOKEN_XN_ID, CKE_IDX_0, 1 << TOKEN_XN_ID);
    }

    uint32_t allBit = 1 << INPUT_XN_ID | 1 << TOKEN_XN_ID;
    for (uint32_t i = 0; i < arg->channelCount; i++) {
        ccu::NotifyWait(arg->channels[i], CKE_IDX_0, allBit);
    }

    HCCL_INFO("[CcuKernelReduceScatterNHR1DMem2Mem] PreSync end");
}

static void PostSync(ReduceScatterNHR1DMem2MemContext &ctx)
{
    const auto *arg = ctx.arg;

    for (uint32_t i = 0; i < arg->channelCount; i++) {
        ccu::NotifyRecord(arg->channels[i], CKE_IDX_0, 1 << POST_SYNC_ID);
    }
    for (uint32_t i = 0; i < arg->channelCount; i++) {
        ccu::NotifyWait(arg->channels[i], CKE_IDX_0, 1 << POST_SYNC_ID);
    }

    HCCL_INFO("[CcuKernelReduceScatterNHR1DMem2Mem] PostSync run finished");
}

static CcuResult DoRepeatWriteReduceSlices(ReduceScatterNHR1DMem2MemContext &ctx, const u32 &toRank, 
    ccu::LocalAddr &src, ccu::RemoteAddr &dst, const bool islastSlice)
{
    const auto *arg = ctx.arg;

    CcuVariable repeatNumAdd;
    CCU_CHK_RET(ccu::Alloc(&repeatNumAdd));
    repeatNumAdd = 1;
    ctx.isRepeatIter = 0;

    ChannelHandle sendChannel = arg->channels[ctx.rank2ChannelIdx[toRank]];
    ctx.repeatNumVarTemp = ctx.repeatNumVar;

    CCU_WHILE(ctx.repeatNumVarTemp != UINT64_MAX) {
        CCU_IF_ONLY(ctx.repeatNumVarTemp != UINT64_MAX) {
            ctx.repeatNumVarTemp = ctx.repeatNumVarTemp + repeatNumAdd;
        }

        CCU_IF_ONLY(ctx.isRepeatIter == 1) {
            src.addr += ctx.inputRepeatStride;
            dst.addr += ctx.inputRepeatStride;
        }
        CCU_IF_ONLY(ctx.isRepeatIter == 0) {
            if (ctx.axisId == 1) {
                src.addr += ctx.die0Size;
                dst.addr += ctx.die0Size;
            }
        }

        ctx.sliceSize = (ctx.axisId == 0) ? (islastSlice ? ctx.die0LastSliceSize : ctx.die0Size)
                                          : (islastSlice ? ctx.die1LastSliceSize : ctx.die1Size);

        ctx.event.setMask(1);
        CCU_IF_ONLY(ctx.sliceSize != 0) {
            ccu::WriteReduceNb(sendChannel, dst, src, ctx.sliceSize, ctx.dataType, ctx.reduceOp, ctx.event);
        }
        CCU_IF_ONLY(ctx.sliceSize == 0) {
            ccu::RecordEvent(ctx.event);
        }
        ccu::WaitEvent(ctx.event);
        ctx.isRepeatIter = 1;
    }
    ctx.isRepeatIter = 0;

    return CCU_SUCCESS;
}

static CcuResult DoRepeatReduceScatterNHRSingleStep(ReduceScatterNHR1DMem2MemContext &ctx, 
    const NHRStepInfo &nhrStepInfo, const std::vector<CcuVariable> &inputSliceOffset)
{
    const auto *arg = ctx.arg;

    u32& toRankIdx = ctx.rank2ChannelIdx[nhrStepInfo.toRank];
    u32& fromRankIdx = ctx.rank2ChannelIdx[nhrStepInfo.fromRank];
    ChannelHandle sendChannel = arg->channels[toRankIdx];
    ChannelHandle recvChannel = arg->channels[fromRankIdx];
    const std::vector<u32> &sendSliceIdxList = nhrStepInfo.txSliceIdxs;

    ctx.remoteDst.token = ctx.token[toRankIdx];
    ctx.localSrc.token = ctx.token[ctx.myRankIdx];

    bool islastSlice = false;

    ccu::NotifyRecord(recvChannel, CKE_IDX_0, 1 << STEP_PRE_SYNC_ID);
    ccu::NotifyWait(sendChannel, CKE_IDX_0, 1 << STEP_PRE_SYNC_ID);

    for (const u32 &sendSliceIdx : sendSliceIdxList) {
        ctx.remoteDst.addr = ctx.input[toRankIdx];
        ctx.remoteDst.addr += inputSliceOffset[sendSliceIdx];
        ctx.localSrc.addr = ctx.input[ctx.myRankIdx];
        ctx.localSrc.addr += inputSliceOffset[sendSliceIdx];

        islastSlice = (sendSliceIdx + 1 == ctx.dimSize);
        CCU_CHK_RET(DoRepeatWriteReduceSlices(ctx, nhrStepInfo.toRank, ctx.localSrc, ctx.remoteDst, islastSlice));
    }

    ccu::NotifyRecord(sendChannel, CKE_IDX_0, 1 << STEP_POST_SYNC_ID);
    ccu::NotifyWait(recvChannel, CKE_IDX_0, 1 << STEP_POST_SYNC_ID);

    return CCU_SUCCESS;
}

static CcuResult DoRepeatReduceScatterNHR(ReduceScatterNHR1DMem2MemContext &ctx)
{
    CcuVariable tmpSliceOffset;
    CCU_CHK_RET(ccu::Alloc(&tmpSliceOffset));
    tmpSliceOffset = 0;

    std::vector<CcuVariable> inputSliceOffset;
    for (u64 i = 0; i < ctx.dimSize; i++) {
        CcuVariable offset;
        CCU_CHK_RET(ccu::Alloc(&offset));
        offset = tmpSliceOffset;
        inputSliceOffset.push_back(offset);
        tmpSliceOffset += ctx.inputSliceStride;
    }

    for (auto &nhrStepInfo : ctx.stepInfoVector) {
        CCU_CHK_RET(DoRepeatReduceScatterNHRSingleStep(ctx, nhrStepInfo, inputSliceOffset));
    }

    ctx.localSrc.addr  = ctx.input[ctx.myRankIdx];
    ctx.localSrc.addr += inputSliceOffset[ctx.mySubCommRankId];
    ctx.localSrc.token = ctx.token[ctx.myRankIdx];
    ctx.localDst.addr  = ctx.output;
    ctx.localDst.addr += ctx.currentRankSliceOutputOffset;
    ctx.localDst.token = ctx.token[ctx.myRankIdx];

    bool islastSlice = (ctx.mySubCommRankId + 1 == ctx.dimSize);

    CcuVariable repeatNumAdd2;
    CCU_CHK_RET(ccu::Alloc(&repeatNumAdd2));
    repeatNumAdd2 = 1;

    CCU_WHILE(ctx.repeatNumVar != UINT64_MAX) {
        ctx.repeatNumVar = ctx.repeatNumVar + repeatNumAdd2;

        CCU_IF_ONLY(ctx.isRepeatIter == 1) {
            ctx.localSrc.addr += ctx.inputRepeatStride;
            ctx.localDst.addr += ctx.outputRepeatStride;
        }
        CCU_IF_ONLY(ctx.isRepeatIter == 0) {
            if (ctx.axisId == 1) {
                ctx.localSrc.addr += ctx.die0Size;
                ctx.localDst.addr += ctx.die0Size;
            }
        }

        CcuVariable &localSliceSize = (ctx.axisId == 0) ? (islastSlice ? ctx.die0LastSliceSize : ctx.die0Size)
                                                        : (islastSlice ? ctx.die1LastSliceSize : ctx.die1Size);

        ctx.event.setMask(1);
        CCU_IF_ONLY(localSliceSize != 0) {
            CCU_IF_ONLY(ctx.isInputOutputEqual == 0) {
                ccu::LocalCopyNb(ctx.localDst, ctx.localSrc, localSliceSize, ctx.event);
            }
            CCU_IF_ONLY(ctx.isInputOutputEqual != 0) {
                ccu::RecordEvent(ctx.event);
            }
        }
        CCU_IF_ONLY(localSliceSize == 0) {
            ccu::RecordEvent(ctx.event);
        }
        ccu::WaitEvent(ctx.event);
        ctx.isRepeatIter = 1;
    }

    return CCU_SUCCESS;
}

CcuResult CcuReduceScatterNHR1DMem2MemKernel(CcuKernelArg arg)
{
    auto *kernelArg = static_cast<CcuKernelArgReduceScatterNHR1D *>(arg);

    ReduceScatterNHR1DMem2MemContext ctx;
    ctx.arg = kernelArg;

    HCCL_INFO("[CcuKernelReduceScatterNHR1DMem2Mem] CcuKernelReduceScatterNHR1DMem2Mem run.");

    CCU_CHK_RET(ParseKernelArg(ctx, kernelArg));
    CCU_CHK_RET(InitResources(ctx));
    CCU_CHK_RET(LoadArgs(ctx));
    PreSync(ctx);
    CCU_CHK_RET(DoRepeatReduceScatterNHR(ctx));
    PostSync(ctx);

    HCCL_INFO("[CcuKernelReduceScatterNHR1DMem2Mem] CcuKernelReduceScatterNHR1DMem2Mem end.");
    return CCU_SUCCESS;
}

} // namespace ops_hccl