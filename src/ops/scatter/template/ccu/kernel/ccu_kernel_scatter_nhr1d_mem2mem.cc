/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "ccu_kernel_scatter_nhr1d_mem2mem.h"
#include "ccu_kernel_alg_base.h"

namespace ops_hccl {

constexpr uint16_t SCRATCH_XN_ID       = 1;
constexpr uint16_t TOKEN_XN_ID         = 2;
constexpr uint16_t STEP_POST_SYNC_ID   = 3;
constexpr uint16_t CKE_IDX_0           = 0;
constexpr uint16_t RANK_NUM_PER_CKE    = 16;

static CcuResult ParseKernelArg(ScatterNHR1DContext &ctx, CcuKernelArgScatterNHRMem2Mem1D *kernelArg)
{
    ctx.arg = kernelArg;
    ctx.rankSize = kernelArg->rankSize;
    ctx.rankId = kernelArg->rankId;
    ctx.rootId = kernelArg->rootId;
    ctx.axisId = kernelArg->axisId;
    ctx.axisSize = kernelArg->axisSize;
    ctx.stepInfoVector = kernelArg->stepInfoVector;
    ctx.rank2ChannelIdx = kernelArg->rank2ChannelIdx;
    ctx.channels = kernelArg->channels;
    ctx.dataType = kernelArg->opParam.DataDes.dataType;

    ctx.localSize = static_cast<uint32_t>(ctx.rank2ChannelIdx.size());
    ctx.myRankIdx = ctx.localSize;
    ctx.signalNum = static_cast<uint32_t>((ctx.rankSize + RANK_NUM_PER_CKE - 1) / RANK_NUM_PER_CKE);
    return CCU_SUCCESS;
}

static CcuResult InitResource(ScatterNHR1DContext &ctx)
{
    CCU_CHK_RET(ccu::Alloc(&ctx.input));
    CCU_CHK_RET(ccu::Alloc(&ctx.output));

    CCU_CHK_RET(ccu::Alloc(&ctx.die0Size));
    CCU_CHK_RET(ccu::Alloc(&ctx.die1Size));
    CCU_CHK_RET(ccu::Alloc(&ctx.inputSliceStride));
    CCU_CHK_RET(ccu::Alloc(&ctx.outputSliceStride));
    CCU_CHK_RET(ccu::Alloc(&ctx.curScratchStride));
    CCU_CHK_RET(ccu::Alloc(&ctx.inputRepeatStride));
    CCU_CHK_RET(ccu::Alloc(&ctx.outputRepeatStride));
    CCU_CHK_RET(ccu::Alloc(&ctx.repeatNumVar));
    CCU_CHK_RET(ccu::Alloc(&ctx.isOutputScratch));
    CCU_CHK_RET(ccu::Alloc(&ctx.isInputOutputEqual));
    CCU_CHK_RET(ccu::Alloc(&ctx.die0TailSize));
    CCU_CHK_RET(ccu::Alloc(&ctx.die1TailSize));
    CCU_CHK_RET(ccu::Alloc(&ctx.isSliceSizeZero));

    // remote ranks scratch/token
    ctx.scratch.clear();
    ctx.token.clear();
    ctx.scratch.reserve(ctx.localSize + 1);
    ctx.token.reserve(ctx.localSize + 1);

    uint16_t channelIdx = 0;
    for (uint32_t i = 0; i < ctx.localSize; i++) {
        if (ctx.channels.size() <= channelIdx) {
            HCCL_ERROR("[CcuScatterNHR1DMem2MemKernel] channels size[%llu] < localSize[%u]", ctx.channels.size(), ctx.localSize);
            return CCU_E_INTERNAL;
        }
        CcuVariable scratchVar;
        CcuVariable tokenVar;
        CCU_CHK_RET(ccu::CreateByChannel(ctx.channels[channelIdx], SCRATCH_XN_ID, &scratchVar));
        CCU_CHK_RET(ccu::CreateByChannel(ctx.channels[channelIdx], TOKEN_XN_ID, &tokenVar));
        ctx.scratch.push_back(scratchVar);
        ctx.token.push_back(tokenVar);
        channelIdx++;
    }
    // local scratch/token
    CcuVariable localScratch;
    CcuVariable localToken;
    CCU_CHK_RET(ccu::Alloc(&localScratch));
    CCU_CHK_RET(ccu::Alloc(&localToken));
    ctx.scratch.push_back(localScratch);
    ctx.token.push_back(localToken);

    CCU_CHK_RET(ccu::Alloc(&ctx.srcMem));
    CCU_CHK_RET(ccu::Alloc(&ctx.dstMem));
    CCU_CHK_RET(ccu::Alloc(&ctx.dstRemoteMem));
    CCU_CHK_RET(ccu::Alloc(&ctx.event));
    return CCU_SUCCESS;
}

static CcuResult LoadArgs(ScatterNHR1DContext &ctx)
{
    CCU_CHK_RET(ccu::LoadArg(ctx.input));
    CCU_CHK_RET(ccu::LoadArg(ctx.output));
    CCU_CHK_RET(ccu::LoadArg(ctx.token[ctx.myRankIdx]));
    CCU_CHK_RET(ccu::LoadArg(ctx.scratch[ctx.myRankIdx]));

    CCU_CHK_RET(ccu::LoadArg(ctx.die0Size));
    CCU_CHK_RET(ccu::LoadArg(ctx.die1Size));
    CCU_CHK_RET(ccu::LoadArg(ctx.inputSliceStride));
    CCU_CHK_RET(ccu::LoadArg(ctx.outputSliceStride));
    CCU_CHK_RET(ccu::LoadArg(ctx.curScratchStride));
    CCU_CHK_RET(ccu::LoadArg(ctx.inputRepeatStride));
    CCU_CHK_RET(ccu::LoadArg(ctx.outputRepeatStride));
    CCU_CHK_RET(ccu::LoadArg(ctx.repeatNumVar));
    CCU_CHK_RET(ccu::LoadArg(ctx.isOutputScratch));
    CCU_CHK_RET(ccu::LoadArg(ctx.isInputOutputEqual));
    CCU_CHK_RET(ccu::LoadArg(ctx.die0TailSize));
    CCU_CHK_RET(ccu::LoadArg(ctx.die1TailSize));
    CCU_CHK_RET(ccu::LoadArg(ctx.isSliceSizeZero));
    return CCU_SUCCESS;
}

static CcuResult PreSync(ScatterNHR1DContext &ctx)
{
    uint32_t allBit = (1 << SCRATCH_XN_ID) | (1 << TOKEN_XN_ID);
    for (auto ch : ctx.channels) {
        ccu::WriteVariableWithNotify(ch, ctx.scratch[ctx.myRankIdx], SCRATCH_XN_ID, CKE_IDX_0, 1 << SCRATCH_XN_ID);
        ccu::WriteVariableWithNotify(ch, ctx.token[ctx.myRankIdx], TOKEN_XN_ID, CKE_IDX_0, 1 << TOKEN_XN_ID);
    }
    for (auto ch : ctx.channels) {
        ccu::NotifyWait(ch, CKE_IDX_0, allBit);
    }
    return CCU_SUCCESS;
}

static void DoLocalCopyNb(ScatterNHR1DContext &ctx, CcuLocalAddr &dst, CcuLocalAddr &src, CcuVariable &sliceSize)
{
    ctx.event.mask = 1 << ctx.rankId;
    CCU_IF_ONLY(sliceSize == 0)
    {
        ccu::RecordEvent(ctx.event);
    }
    CCU_IF_ONLY(sliceSize != 0)
    {
        ccu::LocalCopyNb(dst, src, sliceSize, ctx.event);
    }
    ccu::WaitEvent(ctx.event);
}

static void DoWriteNb(ScatterNHR1DContext &ctx, ChannelHandle sendChannel, CcuRemoteAddr &dst, CcuLocalAddr &src,
                      CcuVariable &sliceSize, uint32_t signalIndex)
{
    ctx.event.mask = 1 << signalIndex;
    CCU_IF_ONLY(sliceSize == 0)
    {
        ccu::RecordEvent(ctx.event);
    }
    CCU_IF_ONLY(sliceSize != 0)
    {
        ccu::WriteNb(sendChannel, dst, src, sliceSize, ctx.event);
    }
    ccu::WaitEvent(ctx.event);
}

static CcuResult DoSendRecvSlice(ScatterNHR1DContext &ctx, const u32 &toRank, CcuLocalAddr &src, CcuRemoteAddr &dst,
                                 u32 signalIndex, bool isLastSlice)
{
    if (ctx.rank2ChannelIdx.count(toRank) == 0) {
        return CCU_SUCCESS;
    }
    u32 toRankIdx = ctx.rank2ChannelIdx[toRank];
    if (toRankIdx >= ctx.channels.size()) {
        return CCU_SUCCESS;
    }
    ChannelHandle sendChannel = ctx.channels[toRankIdx];

    CcuVariable repeatNumAdd;
    CCU_CHK_RET(ccu::Alloc(&repeatNumAdd));
    repeatNumAdd = 1;
    CCU_CHK_RET(ccu::Alloc(&ctx.repeatTimeFlag));
    ctx.repeatTimeFlag = 0;
    CCU_CHK_RET(ccu::Alloc(&ctx.repeatNumVarTemp));
    ctx.repeatNumVarTemp = ctx.repeatNumVar;

    CCU_WHILE(ctx.repeatNumVarTemp != UINT64_MAX)
    {
        ctx.repeatNumVarTemp = ctx.repeatNumVarTemp + repeatNumAdd;
        CCU_IF_ONLY(ctx.repeatTimeFlag == 1)
        {
            if (ctx.rankId == ctx.rootId) {
                src.addr = src.addr + ctx.inputRepeatStride;
            } else {
                src.addr = src.addr + ctx.outputRepeatStride;
            }
            dst.addr = dst.addr + ctx.outputRepeatStride;
        }
        CCU_IF_ONLY(ctx.repeatTimeFlag == 0)
        {
            CCU_IF_ONLY(ctx.axisId == 1)
            {
                if (isLastSlice) {
                    src.addr = src.addr + ctx.die0TailSize;
                    dst.addr = dst.addr + ctx.die0TailSize;
                } else {
                    src.addr = src.addr + ctx.die0Size;
                    dst.addr = dst.addr + ctx.die0Size;
                }
            }
        }

        if (isLastSlice) {
            ctx.curSliceSize = (ctx.axisId == 0) ? ctx.die0TailSize : ctx.die1TailSize;
        } else {
            ctx.curSliceSize = (ctx.axisId == 0) ? ctx.die0Size : ctx.die1Size;
        }
        DoWriteNb(ctx, sendChannel, dst, src, ctx.curSliceSize, signalIndex);
        ctx.repeatTimeFlag = 1;
    }
    return CCU_SUCCESS;
}

static CcuResult DoScatterNHRSingleStep(ScatterNHR1DContext &ctx, const NHRStepInfo &nhrStepInfo)
{
    const auto &sendSliceIdxList = nhrStepInfo.txSliceIdxs;
    const auto &recvSliceIdxList = nhrStepInfo.rxSliceIdxs;

    if (!recvSliceIdxList.empty()) {
        if (ctx.rank2ChannelIdx.count(nhrStepInfo.fromRank) != 0) {
            u32 fromRankIdx = ctx.rank2ChannelIdx[nhrStepInfo.fromRank];
            if (fromRankIdx < ctx.channels.size()) {
                ChannelHandle recvChannel = ctx.channels[fromRankIdx];
                ccu::NotifyWait(recvChannel, CKE_IDX_0, 1 << STEP_POST_SYNC_ID);
            }
        }
    }

    if (!sendSliceIdxList.empty()) {
        if (ctx.rank2ChannelIdx.count(nhrStepInfo.toRank) == 0) {
            return;
        }
        u32 toRankIdx = ctx.rank2ChannelIdx[nhrStepInfo.toRank];
        if (toRankIdx >= ctx.channels.size()) {
            return;
        }
        ChannelHandle sendChannel = ctx.channels[toRankIdx];

        for (u32 i = 0; i < sendSliceIdxList.size(); i++) {
            u32 sendSliceIdx = sendSliceIdxList[i];
            bool isLastSlice = (sendSliceIdx == ctx.rankSize - 1);

            if (i != 0 && i % RANK_NUM_PER_CKE == 0) {
                ctx.event.mask = (1 << RANK_NUM_PER_CKE) - 1;
                ccu::WaitEvent(ctx.event);
            }

            if (ctx.rankId == ctx.rootId) {
                ctx.srcMem.addr = ctx.input;
                ctx.srcMem.addr = ctx.srcMem.addr + ctx.inputOffset[sendSliceIdx];
            } else {
                ctx.srcMem.addr = ctx.scratch[ctx.myRankIdx];
                ctx.srcMem.addr = ctx.srcMem.addr + ctx.scratchOffset[sendSliceIdx];
            }
            ctx.srcMem.token = ctx.token[ctx.myRankIdx];

            ctx.dstRemoteMem.token = ctx.token[toRankIdx];
            ctx.dstRemoteMem.addr = ctx.scratch[toRankIdx];
            ctx.dstRemoteMem.addr = ctx.dstRemoteMem.addr + ctx.scratchOffset[sendSliceIdx];

            CCU_CHK_RET(DoSendRecvSlice(ctx, nhrStepInfo.toRank, ctx.srcMem, ctx.dstRemoteMem,
                                        i % RANK_NUM_PER_CKE, isLastSlice));
        }

        ccu::NotifyRecord(sendChannel, CKE_IDX_0, 1 << STEP_POST_SYNC_ID);
    }
    return CCU_SUCCESS;
}

static CcuResult DoScatterNHR(ScatterNHR1DContext &ctx)
{
    CCU_CHK_RET(ccu::Alloc(&ctx.curInputOffset));
    CCU_CHK_RET(ccu::Alloc(&ctx.curScratchOffset));
    ctx.curInputOffset = 0;
    ctx.curScratchOffset = 0;

    ctx.inputOffset.resize(ctx.rankSize);
    ctx.scratchOffset.resize(ctx.rankSize);
    for (u64 i = 0; i < ctx.rankSize; i++) {
        CCU_CHK_RET(ccu::Alloc(&ctx.inputOffset[i]));
        ctx.inputOffset[i] = ctx.curInputOffset;
        ctx.curInputOffset = ctx.curInputOffset + ctx.inputSliceStride;
    }
    for (u64 i = 0; i < ctx.rankSize; i++) {
        CCU_CHK_RET(ccu::Alloc(&ctx.scratchOffset[i]));
        ctx.scratchOffset[i] = ctx.curScratchOffset;
        ctx.curScratchOffset = ctx.curScratchOffset + ctx.curScratchStride;
    }

    for (auto &step : ctx.stepInfoVector) {
        CCU_CHK_RET(DoScatterNHRSingleStep(ctx, step));
    }

    // final local copy to output
    if (ctx.rankId == ctx.rootId) {
        ctx.srcMem.addr = ctx.input;
        ctx.srcMem.addr = ctx.srcMem.addr + ctx.inputOffset[ctx.rankId];
    } else {
        ctx.srcMem.addr = ctx.scratch[ctx.myRankIdx];
        ctx.srcMem.addr = ctx.srcMem.addr + ctx.scratchOffset[ctx.rankId];
    }
    ctx.dstMem.addr = ctx.output;
    ctx.srcMem.token = ctx.token[ctx.myRankIdx];
    ctx.dstMem.token = ctx.token[ctx.myRankIdx];

    CcuVariable repeatNumAdd;
    CCU_CHK_RET(ccu::Alloc(&repeatNumAdd));
    repeatNumAdd = 1;
    CCU_CHK_RET(ccu::Alloc(&ctx.repeatTimeFlag));
    ctx.repeatTimeFlag = 0;

    CCU_WHILE(ctx.repeatNumVar != UINT64_MAX)
    {
        ctx.repeatNumVar = ctx.repeatNumVar + repeatNumAdd;
        CCU_IF_ONLY(ctx.repeatTimeFlag != 0)
        {
            if (ctx.rankId == ctx.rootId) {
                ctx.srcMem.addr = ctx.srcMem.addr + ctx.inputRepeatStride;
            } else {
                ctx.srcMem.addr = ctx.srcMem.addr + ctx.outputRepeatStride;
            }
            ctx.dstMem.addr = ctx.dstMem.addr + ctx.outputRepeatStride;
        }
        CCU_IF_ONLY(ctx.repeatTimeFlag == 0)
        {
            CCU_IF_ONLY(ctx.axisId == 1)
            {
                if (ctx.rankId != ctx.rankSize - 1) {
                    ctx.srcMem.addr = ctx.srcMem.addr + ctx.die0Size;
                    ctx.dstMem.addr = ctx.dstMem.addr + ctx.die0Size;
                } else {
                    ctx.srcMem.addr = ctx.srcMem.addr + ctx.die0TailSize;
                    ctx.dstMem.addr = ctx.dstMem.addr + ctx.die0TailSize;
                }
            }
        }

        if (ctx.rankId != ctx.rankSize - 1) {
            ctx.curSliceSize = (ctx.axisId == 0) ? ctx.die0Size : ctx.die1Size;
        } else {
            ctx.curSliceSize = (ctx.axisId == 0) ? ctx.die0TailSize : ctx.die1TailSize;
        }

        CCU_IF_ONLY(ctx.isOutputScratch == 1)
        {
            CCU_IF_ONLY(ctx.outputSliceStride == 0)
            {
                // special-case: rootId != 0 && rankId == 0
                if (ctx.rootId != 0 && ctx.rankId == 0) {
                    ctx.event.mask = 1 << ctx.rankId;
                    ccu::RecordEvent(ctx.event);
                    ccu::WaitEvent(ctx.event);
                } else {
                    CCU_IF_ONLY(ctx.isInputOutputEqual != 1)
                    {
                        if (ctx.rankId == ctx.rootId) {
                            DoLocalCopyNb(ctx, ctx.dstMem, ctx.srcMem, ctx.curSliceSize);
                        } else {
                            CCU_IF_ONLY(ctx.isSliceSizeZero != 1)
                            {
                                DoLocalCopyNb(ctx, ctx.dstMem, ctx.srcMem, ctx.curSliceSize);
                            }
                            CCU_IF_ONLY(ctx.isSliceSizeZero == 1)
                            {
                                ctx.event.mask = 1 << ctx.rankId;
                                ccu::RecordEvent(ctx.event);
                                ccu::WaitEvent(ctx.event);
                            }
                        }
                    }
                    CCU_IF_ONLY(ctx.isInputOutputEqual == 1)
                    {
                        ctx.event.mask = 1 << ctx.rankId;
                        ccu::RecordEvent(ctx.event);
                        ccu::WaitEvent(ctx.event);
                    }
                }
            }
            CCU_IF_ONLY(ctx.outputSliceStride != 0)
            {
                if (ctx.rankId == ctx.rootId) {
                    CCU_IF_ONLY(ctx.isInputOutputEqual != 1)
                    {
                        for (uint32_t i = 0; i < ctx.rootId; i++) {
                            ctx.dstMem.addr = ctx.dstMem.addr + ctx.outputSliceStride;
                        }
                        DoLocalCopyNb(ctx, ctx.dstMem, ctx.srcMem, ctx.curSliceSize);
                    }
                    CCU_IF_ONLY(ctx.isInputOutputEqual == 1)
                    {
                        ctx.event.mask = 1 << ctx.rankId;
                        ccu::RecordEvent(ctx.event);
                        ccu::WaitEvent(ctx.event);
                    }
                } else {
                    ctx.event.mask = 1 << ctx.rankId;
                    ccu::RecordEvent(ctx.event);
                    ccu::WaitEvent(ctx.event);
                }
            }
        }
        CCU_IF_ONLY(ctx.isOutputScratch != 1)
        {
            DoLocalCopyNb(ctx, ctx.dstMem, ctx.srcMem, ctx.curSliceSize);
        }

        ctx.repeatTimeFlag = 1;
    }

    return CCU_SUCCESS;
}

CcuResult CcuScatterNHR1DMem2MemKernel(CcuKernelArg arg)
{
    auto *kernelArg = static_cast<CcuKernelArgScatterNHRMem2Mem1D *>(arg);
    ScatterNHR1DContext ctx{};

    CCU_CHK_RET(ParseKernelArg(ctx, kernelArg));
    CCU_CHK_RET(InitResource(ctx));
    CCU_CHK_RET(LoadArgs(ctx));

    CCU_CHK_RET(PreSync(ctx));
    CCU_CHK_RET(DoScatterNHR(ctx));
    return CCU_SUCCESS;
}

} // namespace ops_hccl
