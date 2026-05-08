/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "ccu_kernel_all_gather_nhr1d_multi_jetty_mem2mem.h"

namespace ops_hccl {

constexpr uint16_t OUTPUT_XN_ID       = 1;
constexpr uint16_t TOKEN_XN_ID        = 2;
constexpr uint16_t POST_SYNC_ID       = 3;
constexpr uint16_t STEP_PRE_SYNC_ID   = 4;
constexpr uint16_t STEP_POST_SYNC_ID = 5;
constexpr uint16_t CKE_IDX_0          = 0;
constexpr uint16_t BIT_NUM_PER_CKE    = 16;

static CcuResult ParseKernelArg(AllGatherNHR1DMultiJettyMem2MemContext &ctx,
    CcuKernelArgAllGatherNHR1DMultiJettyMem2Mem *kernelArg)
{
    ctx.arg = kernelArg;
    ctx.localSize = kernelArg->rank2ChannelIdx.size();
    ctx.myRankIdx = kernelArg->rank2ChannelIdx.size();
    return CCU_SUCCESS;
}

static CcuResult InitResource(AllGatherNHR1DMultiJettyMem2MemContext &ctx)
{
    const auto *arg = ctx.arg;

    if (arg->channelCount == 0) {
        HCCL_ERROR("[CcuKernelAllGatherNHR1DMultiJettyMem2Mem] channels is empty!");
        return CcuResult::CCU_E_INTERNAL;
    }

    CCU_CHK_RET(ccu::Alloc(&ctx.input));
    ctx.output.resize(ctx.localSize + 1);
    ctx.token.resize(ctx.localSize + 1);

    for (uint32_t channelIdx = 0; channelIdx < arg->channelCount; channelIdx++) {
        HCCL_DEBUG("[CcuKernelAllGatherNHR1DMultiJettyMem2Mem] MyRank[%u], channelIdx[%u]",
            arg->rankId, channelIdx);
        CCU_CHK_RET(ccu::CreateByChannel(
            arg->channels[channelIdx], OUTPUT_XN_ID, &ctx.output[channelIdx]));
        CCU_CHK_RET(ccu::CreateByChannel(
            arg->channels[channelIdx], TOKEN_XN_ID, &ctx.token[channelIdx]));
    }
    
    CCU_CHK_RET(ccu::Alloc(&ctx.output[ctx.myRankIdx]));
    CCU_CHK_RET(ccu::Alloc(&ctx.token[ctx.myRankIdx]));

    CCU_CHK_RET(ccu::Alloc(&ctx.sliceSize));
    CCU_CHK_RET(ccu::Alloc(&ctx.sliceSizePerJetty));
    CCU_CHK_RET(ccu::Alloc(&ctx.lastSliceSizePerJetty));
    CCU_CHK_RET(ccu::Alloc(&ctx.repeatNumInv));
    CCU_CHK_RET(ccu::Alloc(&ctx.inputSliceStride));
    CCU_CHK_RET(ccu::Alloc(&ctx.outputSliceStride));
    CCU_CHK_RET(ccu::Alloc(&ctx.inputRepeatStride));
    CCU_CHK_RET(ccu::Alloc(&ctx.outputRepeatStride));
    CCU_CHK_RET(ccu::Alloc(&ctx.isInputOutputEqual));
    CCU_CHK_RET(ccu::Alloc(&ctx.groupOpSize.addrOffset));
    CCU_CHK_RET(ccu::Alloc(&ctx.groupOpSize.loopParam));
    CCU_CHK_RET(ccu::Alloc(&ctx.groupOpSize.parallelParam));
    CCU_CHK_RET(ccu::Alloc(&ctx.groupOpSize.residual));

    CCU_CHK_RET(ccu::CreateLoopExecutor(&ctx.enginePool, 8)); // 待修改

    ctx.resourceAllocated = false;
    ctx.loopRegistered = false;

    CCU_CHK_RET(ccu::Alloc(&ctx.event));

    ctx.outputSliceOffset.resize(arg->rankSize);
    for (u64 i = 0; i < arg->rankSize; i++) {
        CCU_CHK_RET(ccu::Alloc(&ctx.outputSliceOffset[i]));
    }

    CCU_CHK_RET(ccu::Alloc(&ctx.constVar1));
    ctx.constVar1 = 1;

    CCU_CHK_RET(ccu::Alloc(&ctx.srcMem));
    CCU_CHK_RET(ccu::Alloc(&ctx.dstMem));
    CCU_CHK_RET(ccu::Alloc(&ctx.myDstMem));

    CCU_CHK_RET(ccu::Alloc(&ctx.repeatTimeflag));
    ctx.repeatTimeflag = 0;

    CCU_CHK_RET(ccu::Alloc(&ctx.tmpCopyRepeatNumInv));

    return CCU_SUCCESS;
}

static CcuResult LoadArgs(AllGatherNHR1DMultiJettyMem2MemContext &ctx)
{
    const auto *arg = ctx.arg;

    CCU_CHK_RET(ccu::LoadArg(ctx.input));
    CCU_CHK_RET(ccu::LoadArg(ctx.output[ctx.myRankIdx]));
    CCU_CHK_RET(ccu::LoadArg(ctx.token[ctx.myRankIdx]));
    CCU_CHK_RET(ccu::LoadArg(ctx.sliceSize));
    CCU_CHK_RET(ccu::LoadArg(ctx.sliceSizePerJetty));
    CCU_CHK_RET(ccu::LoadArg(ctx.lastSliceSizePerJetty));
    CCU_CHK_RET(ccu::LoadArg(ctx.repeatNumInv));
    CCU_CHK_RET(ccu::LoadArg(ctx.inputSliceStride));
    CCU_CHK_RET(ccu::LoadArg(ctx.outputSliceStride));
    CCU_CHK_RET(ccu::LoadArg(ctx.inputRepeatStride));
    CCU_CHK_RET(ccu::LoadArg(ctx.outputRepeatStride));
    CCU_CHK_RET(ccu::LoadArg(ctx.isInputOutputEqual));
    CCU_CHK_RET(ccu::LoadArg(ctx.groupOpSize.addrOffset));
    CCU_CHK_RET(ccu::LoadArg(ctx.groupOpSize.loopParam));
    CCU_CHK_RET(ccu::LoadArg(ctx.groupOpSize.parallelParam));
    CCU_CHK_RET(ccu::LoadArg(ctx.groupOpSize.residual));

    HCCL_DEBUG("[CcuKernelAllGatherNHR1DMultiJettyMem2Mem] LoadArgs run finished");
    return CCU_SUCCESS;
}

static CcuResult PreSync(AllGatherNHR1DMultiJettyMem2MemContext &ctx)
{
    const auto *arg = ctx.arg;

    HCCL_DEBUG("[CcuKernelAllGatherNHR1DMultiJettyMem2Mem] PreSync start");
    for (uint32_t i = 0; i < arg->channelCount; i++) {
        CCU_CHK_RET(ccu::WriteVariableWithNotify(arg->channels[i], ctx.output[ctx.myRankIdx],
            OUTPUT_XN_ID, CKE_IDX_0, 1 << OUTPUT_XN_ID));
        CCU_CHK_RET(ccu::WriteVariableWithNotify(arg->channels[i], ctx.token[ctx.myRankIdx],
            TOKEN_XN_ID, CKE_IDX_0, 1 << TOKEN_XN_ID));
    }

    uint16_t allBit = 1 << OUTPUT_XN_ID | 1 << TOKEN_XN_ID;
    for (uint32_t i = 0; i < arg->channelCount; i++) {
        CCU_CHK_RET(ccu::NotifyWait(arg->channels[i], CKE_IDX_0, allBit));
    }
    HCCL_DEBUG("[CcuKernelAllGatherNHR1DMultiJettyMem2Mem] PreSync end");
    return CCU_SUCCESS;
}

static CcuResult PostSync(AllGatherNHR1DMultiJettyMem2MemContext &ctx)
{
    const auto *arg = ctx.arg;

    HCCL_DEBUG("[CcuKernelAllGatherNHR1DMultiJettyMem2Mem] PostSync start");
    for (uint32_t i = 0; i < arg->channelCount; i++) {
        CCU_CHK_RET(ccu::NotifyRecord(arg->channels[i], CKE_IDX_0, 1 << POST_SYNC_ID));
    }

    for (uint32_t i = 0; i < arg->channelCount; i++) {
        CCU_CHK_RET(ccu::NotifyWait(arg->channels[i], CKE_IDX_0, 1 << POST_SYNC_ID));
    }
    HCCL_DEBUG("[CcuKernelAllGatherNHR1DMultiJettyMem2Mem] PostSync end");
    return CCU_SUCCESS;
}

static CcuResult DoSendRecvSlices(AllGatherNHR1DMultiJettyMem2MemContext &ctx,
    const uint32_t &toRank, ccu::LocalAddr &srcMem, ccu::RemoteAddr &dstMem)
{
    const auto *arg = ctx.arg;
    ChannelHandle sendChannel = arg->channels[arg->rank2ChannelIdx.at(toRank)];
    ccu::LocalAddr srcMemTmp;
    ccu::RemoteAddr dstMemTmp;
    CCU_CHK_RET(ccu::Alloc(&srcMemTmp));
    CCU_CHK_RET(ccu::Alloc(&dstMemTmp));
    srcMemTmp.addr = srcMem.addr;
    srcMemTmp.token = srcMem.token;
    dstMemTmp.addr = dstMem.addr;
    dstMemTmp.token = dstMem.token;

    CCU_IF_ONLY(ctx.sliceSizePerJetty != 0)
    {
        for (uint32_t i = 0; i < arg->jettyNum - 1; ++i) {
            CCU_CHK_RET(ccu::SetMask(ctx.event, 1 << i));
            CCU_CHK_RET(ccu::WriteNb(sendChannel, dstMemTmp, srcMemTmp, ctx.sliceSizePerJetty, ctx.event));
            srcMemTmp.addr += ctx.sliceSizePerJetty;
            dstMemTmp.addr += ctx.sliceSizePerJetty;
        }
    }
    CCU_IF_ONLY(ctx.sliceSizePerJetty == 0)
    {
        for (uint32_t i = 0; i < arg->jettyNum - 1; ++i) {
            CCU_CHK_RET(ccu::SetMask(ctx.event, 1 << i));
            CCU_CHK_RET(ccu::RecordEvent(ctx.event));
        }
    }
    CCU_IF_ONLY(ctx.lastSliceSizePerJetty != 0)
    {
        CCU_CHK_RET(ccu::SetMask(ctx.event, 1 << (arg->jettyNum - 1)));
        CCU_CHK_RET(ccu::WriteNb(sendChannel, dstMemTmp, srcMemTmp, ctx.lastSliceSizePerJetty, ctx.event));
    }
    CCU_IF_ONLY(ctx.lastSliceSizePerJetty == 0)
    {
        CCU_CHK_RET(ccu::SetMask(ctx.event, 1 << (arg->jettyNum - 1)));
        CCU_CHK_RET(ccu::RecordEvent(ctx.event));
    }

    uint16_t sendBit = (1 << arg->jettyNum) - 1;
    CCU_CHK_RET(ccu::SetMask(ctx.event, sendBit));
    CCU_CHK_RET(ccu::WaitEvent(ctx.event));

    return CCU_SUCCESS;
}

static CcuResult DoRepeatAllGatherNHRSingleStep(AllGatherNHR1DMultiJettyMem2MemContext &ctx,
    const NHRStepInfo &nhrStepInfo)
{
    const auto *arg = ctx.arg;
    const u32 &toRankIdx = arg->rank2ChannelIdx.at(nhrStepInfo.toRank);
    const u32 &fromRankIdx = arg->rank2ChannelIdx.at(nhrStepInfo.fromRank);
    u32 sendSliceIdx = 0;
    const ChannelHandle &sendChannel = arg->channels[toRankIdx];
    const ChannelHandle &recvChannel = arg->channels[fromRankIdx];
    const std::vector<u32> &sendSliceIdxList = nhrStepInfo.txSliceIdxs;

    ctx.srcMem.token = ctx.token[ctx.myRankIdx];
    ctx.dstMem.token = ctx.token[toRankIdx];

    CCU_CHK_RET(ccu::NotifyRecord(recvChannel, CKE_IDX_0, 1 << STEP_PRE_SYNC_ID)); // 待修改
    CCU_CHK_RET(ccu::NotifyWait(sendChannel, CKE_IDX_0, 1 << STEP_PRE_SYNC_ID));

    for (u32 i = 0; i < sendSliceIdxList.size(); i++) {
        sendSliceIdx = sendSliceIdxList[i];
        ctx.srcMem.addr = ctx.output[ctx.myRankIdx];
        ctx.srcMem.addr += ctx.outputSliceOffset[sendSliceIdx];
        ctx.dstMem.addr = ctx.output[toRankIdx];
        ctx.dstMem.addr += ctx.outputSliceOffset[sendSliceIdx];

        ctx.repeatTimeflag = 0;
        ctx.tmpCopyRepeatNumInv = ctx.repeatNumInv;

        CCU_DO_WHILE(ctx.tmpCopyRepeatNumInv != UINT64_MAX)
        {
            ctx.tmpCopyRepeatNumInv += ctx.constVar1;
            CCU_IF_ONLY(ctx.repeatTimeflag == 1)
            {
                ctx.srcMem.addr += ctx.inputRepeatStride;
                ctx.dstMem.addr += ctx.outputRepeatStride;
            }
            CCU_CHK_RET(DoSendRecvSlices(ctx, nhrStepInfo.toRank, ctx.srcMem, ctx.dstMem));
            ctx.repeatTimeflag = 1;
        }
    }
    CCU_CHK_RET(ccu::NotifyRecord(sendChannel, CKE_IDX_0, 1 << STEP_POST_SYNC_ID));
    CCU_CHK_RET(ccu::NotifyWait(recvChannel, CKE_IDX_0, 1 << STEP_POST_SYNC_ID));

    return CCU_SUCCESS;
}

static CcuResult DoRepeatAllGatherNHR(AllGatherNHR1DMultiJettyMem2MemContext &ctx)
{
    const auto *arg = ctx.arg;

    CcuVariable tmpSliceOffset;
    CCU_CHK_RET(ccu::Alloc(&tmpSliceOffset));
    tmpSliceOffset = 0;

    CcuVariable myrankInputSliceOffset;
    CCU_CHK_RET(ccu::Alloc(&myrankInputSliceOffset));
    myrankInputSliceOffset = 0;

    for (u64 i = 0; i < arg->rankId; i++) {
        myrankInputSliceOffset += ctx.inputSliceStride;
    }
    for (u64 i = 0; i < arg->rankSize; i++) {
        ctx.outputSliceOffset[i] = tmpSliceOffset;
        tmpSliceOffset += ctx.outputSliceStride;
    }

    ctx.srcMem.addr = ctx.input;
    ctx.srcMem.addr += myrankInputSliceOffset;
    ctx.myDstMem.addr = ctx.output[ctx.myRankIdx];
    ctx.myDstMem.addr += ctx.outputSliceOffset[arg->rankId];
    ctx.srcMem.token = ctx.token[ctx.myRankIdx];
    ctx.myDstMem.token = ctx.token[ctx.myRankIdx];

    ctx.tmpCopyRepeatNumInv = ctx.repeatNumInv;

    CCU_DO_WHILE(ctx.tmpCopyRepeatNumInv != UINT64_MAX)
    {
        ctx.tmpCopyRepeatNumInv += ctx.constVar1;
        CCU_IF_ONLY(ctx.repeatTimeflag != 0)
        {
            ctx.srcMem.addr += ctx.inputRepeatStride;
            ctx.myDstMem.addr += ctx.outputRepeatStride;
        }
        CCU_CHK_RET(ccu::SetMask(ctx.event, 1 << arg->rankId));
        CCU_IF_ONLY(ctx.isInputOutputEqual == 0)
        {
            CCU_CHK_RET(GroupCopy(ctx, ctx.myDstMem, ctx.srcMem, ctx.groupOpSize));
            CCU_CHK_RET(ccu::RecordEvent(ctx.event));
        }
        CCU_IF_ONLY(ctx.isInputOutputEqual != 0)
        {
            CCU_CHK_RET(ccu::RecordEvent(ctx.event));
        }
        CCU_CHK_RET(ccu::SetMask(ctx.event, 1 << arg->rankId));
        CCU_CHK_RET(ccu::WaitEvent(ctx.event));
        ctx.repeatTimeflag = 1;
    }

    for (auto &nhrStepInfo : arg->stepInfoVector) {
        CCU_CHK_RET(DoRepeatAllGatherNHRSingleStep(ctx, nhrStepInfo));
    }

    return CCU_SUCCESS;
}

CcuResult CcuAllGatherNHR1DMultiJettyMem2MemKernel(CcuKernelArg arg)
{
    auto *kernelArg = static_cast<CcuKernelArgAllGatherNHR1DMultiJettyMem2Mem *>(arg);

    AllGatherNHR1DMultiJettyMem2MemContext ctx;
    ctx.resourceAllocated = false;
    ctx.loopRegistered = false;
    ctx.moConfig.msInterleave = 0;
    ctx.moConfig.loopCount = 0;
    ctx.moConfig.memSlice = 0;
    ctx.moRes.eventCount = 0;
    ctx.moRes.bufCount = 0;
    ctx.enginePool = 0;

    HCCL_INFO("[CcuKernelAllGatherNHR1DMultiJettyMem2Mem] AllGatherNHR1DMultiJettyMem2Mem start");
    CCU_CHK_RET(ParseKernelArg(ctx, kernelArg));
    CCU_CHK_RET(InitResource(ctx));
    CCU_CHK_RET(LoadArgs(ctx));

    CCU_CHK_RET(PreSync(ctx));

    CCU_CHK_RET(DoRepeatAllGatherNHR(ctx));

    CCU_CHK_RET(PostSync(ctx));
    HCCL_INFO("[CcuKernelAllGatherNHR1DMultiJettyMem2Mem] AllGatherNHR1DMultiJettyMem2Mem end");

    return CCU_SUCCESS;
}

} // namespace ops_hccl
