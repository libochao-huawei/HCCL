/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "ccu_kernel_broadcast_nhr1d_mem2mem.h"
#include "ccu_kernel_alg_base.h"

namespace ops_hccl {

constexpr uint16_t OUTPUT_XN_ID     = 0;
constexpr uint16_t TOKEN_XN_ID      = 1;
constexpr uint16_t POST_SYNC_ID     = 2;
constexpr uint16_t STEP_PRE_SYNC_ID = 3;
constexpr uint16_t STEP_POST_SYNC_ID= 4;

constexpr uint16_t CKE_IDX_0        = 0;
constexpr uint16_t RANK_NUM_PER_CKE = 16;

static CcuResult ParseKernelArg(BroadcastNhr1DMem2MemContext &ctx, CcuKernelArgBroadcastNhr1DMem2Mem *kernelArg)
{
    ctx.arg = kernelArg;
    ctx.rankId = kernelArg->rankId;
    ctx.axisId = kernelArg->axisId;
    ctx.axisSize = kernelArg->axisSize;
    ctx.dimSize = kernelArg->dimSize.size();
    ctx.stepInfoVector = kernelArg->stepInfoVector;
    ctx.rank2ChannelIdx = kernelArg->rank2ChannelIdx;
    ctx.localSize = kernelArg->rank2ChannelIdx.size();
    ctx.myRankIdx = kernelArg->rank2ChannelIdx.size();
    ctx.dataType = kernelArg->opParam.DataDes.dataType;
    HCCL_INFO("[CcuBroadcastNhr1DMem2Mem] CtxArg: rankId[%u], axisId[%u], axisSize[%u], dimSize[%u], "
              "stepInfoVectorSize[%zu], localSize[%zu], dataType[%u] channelsSize[%u]",
              ctx.rankId, ctx.axisId, ctx.axisSize, ctx.dimSize, ctx.stepInfoVector.size(),
              ctx.localSize, ctx.dataType, ctx.arg->channelCount);
    return CCU_SUCCESS;
}

static CcuResult InitResources(BroadcastNhr1DMem2MemContext &ctx)
{
    if (ctx.arg->channelCount == 0) {
        HCCL_ERROR("[CcuBroadcastNhr1DMem2Mem] channels is empty!");
        return CCU_E_INTERNAL;
    }

    // 按照rank号从小到大遍历channels，遇到本rank就填充本地资源，否则依次取远端资源
    for (uint32_t channelIdx = 0; channelIdx < ctx.localSize; channelIdx++) {
        ccu::Variable outputVar = ccu::GetResByChannel<ccu::Variable>(ctx.arg->channels[channelIdx], OUTPUT_XN_ID);
        ctx.output.push_back(outputVar);
        ccu::Variable tokenVar = ccu::GetResByChannel<ccu::Variable>(ctx.arg->channels[channelIdx], TOKEN_XN_ID);
        ctx.token.push_back(tokenVar);
    }
    ctx.output.push_back(ccu::Variable());
    ctx.token.push_back(ccu::Variable());

    ccu::Variable tmpSliceOffset;
    tmpSliceOffset = 0;
    for (u64 i = 0; i < ctx.dimSize; i++) {
        ctx.sliceOffset.push_back(ccu::Variable());
        ctx.sliceOffset[i] = tmpSliceOffset;
        tmpSliceOffset += ctx.axisId == 0 ? ctx.die0SliceSize : ctx.die1SliceSize;
    }

    ctx.resourceAllocated = false;
    HCCL_INFO("[CcuBroadcastNhr1DMem2Mem][InitResources] InitResources end");
    return CCU_SUCCESS;
}

static CcuResult LoadArgs(BroadcastNhr1DMem2MemContext &ctx)
{
    uint32_t argId = 0;
    CCU_CHK_RET(ccu::LoadArg(ctx.input, argId++));
    CCU_CHK_RET(ccu::LoadArg(ctx.output[ctx.myRankIdx], argId++));
    CCU_CHK_RET(ccu::LoadArg(ctx.token[ctx.myRankIdx], argId++));
    CCU_CHK_RET(ccu::LoadArg(ctx.die0Size, argId++));
    CCU_CHK_RET(ccu::LoadArg(ctx.die1Size, argId++));
    CCU_CHK_RET(ccu::LoadArg(ctx.die0SliceSize, argId++));
    CCU_CHK_RET(ccu::LoadArg(ctx.die1SliceSize, argId++));
    CCU_CHK_RET(ccu::LoadArg(ctx.die0LastSliceSize, argId++));
    CCU_CHK_RET(ccu::LoadArg(ctx.die1LastSliceSize, argId++));
    HCCL_INFO("[CcuBroadcastNhr1DMem2Mem] LoadArgs run finished");
    return CCU_SUCCESS;
}

static CcuResult PreSync(BroadcastNhr1DMem2MemContext &ctx)
{
    for (uint32_t i = 0; i < ctx.arg->channelCount; i++) {
        ccu::WriteVariableWithNotify(ctx.arg->channels[i], ctx.output[ctx.myRankIdx], OUTPUT_XN_ID, CKE_IDX_0, 1 << OUTPUT_XN_ID);
        ccu::WriteVariableWithNotify(ctx.arg->channels[i], ctx.token[ctx.myRankIdx], TOKEN_XN_ID, CKE_IDX_0, 1 << TOKEN_XN_ID);
    }
    uint32_t allBit = 1 << OUTPUT_XN_ID | 1 << TOKEN_XN_ID;
    for (uint32_t i = 0; i < ctx.arg->channelCount; i++) {
        ccu::NotifyWait(ctx.arg->channels[i], CKE_IDX_0, allBit);
    }
    HCCL_INFO("[CcuBroadcastNhr1DMem2Mem] BroadcastNhr1D wait all end");
    return CCU_SUCCESS;
}

static CcuResult PostSync(BroadcastNhr1DMem2MemContext &ctx)
{
    for (uint32_t i = 0; i < ctx.arg->channelCount; i++) {
        ccu::NotifyRecord(ctx.arg->channels[i], CKE_IDX_0, 1 << POST_SYNC_ID);
    }
    for (uint32_t i = 0; i < ctx.arg->channelCount; i++) {
        ccu::NotifyWait(ctx.arg->channels[i], CKE_IDX_0, 1 << POST_SYNC_ID);
    }
    HCCL_INFO("[CcuBroadcastNhr1DMem2Mem] BroadcastNhr1D groupwait end");
    return CCU_SUCCESS;
}

static CcuResult DoSendRecvSlice(BroadcastNhr1DMem2MemContext &ctx, const u32 &toRank,
    ccu::LocalAddr &src, ccu::RemoteAddr &dst,
    const u32 &sendSliceIdx, u32 signalIndex)
{
    HCCL_INFO("[CcuBroadcastNhr1DMem2Mem][DoSendRecvSlice] toRank[%u] sendSliceIdx[%u] signalIndex[%u]",
                toRank, sendSliceIdx, signalIndex);
    size_t sendChannel = ctx.arg->channels[ctx.rank2ChannelIdx[toRank]];
    bool islastSlice;

    // 添加 die1 偏移
    if (ctx.axisId == 1) {
        src.addr += ctx.die0Size;
        dst.addr += ctx.die0Size;
        ctx.localDst.addr += ctx.die0Size;
    }

    islastSlice = (sendSliceIdx + 1 == ctx.dimSize);
    const ccu::Variable &sliceSize = ctx.axisId == 0 ? (islastSlice ? ctx.die0LastSliceSize : ctx.die0SliceSize)
                                                    : (islastSlice ? ctx.die1LastSliceSize : ctx.die1SliceSize);
    ctx.event.SetMask(1 << signalIndex);
    CCU_IF(sliceSize != 0)
    {
        ccu::Write(sendChannel, dst, src, sliceSize, ctx.event, 1 << signalIndex);
    }
    CCU_IF(sliceSize == 0)
    {
        ccu::EventRecord(ctx.event, 1 << signalIndex);
    }
    return CCU_SUCCESS;
}

static CcuResult DoScatterNHRSingleStep(BroadcastNhr1DMem2MemContext &ctx, const NHRStepInfo &nhrStepInfo)
{
    const std::vector<u32> &sendSliceIdxList = nhrStepInfo.txSliceIdxs;
    const std::vector<u32> &recvSliceIdxList = nhrStepInfo.rxSliceIdxs;
    HCCL_INFO("[CcuBroadcastNhr1DMem2Mem][DoScatterNHRSingleStep] sendSliceIdxListSize[%zu] recvSliceIdxList[%zu] "
                "step[%u] myRank[%u] nSlices[%u] toRank[%u] fromRank[%u]", sendSliceIdxList.size(), recvSliceIdxList.size(),
                nhrStepInfo.step, nhrStepInfo.myRank, nhrStepInfo.nSlices, nhrStepInfo.toRank, nhrStepInfo.fromRank);
    // 只需要发
    if (sendSliceIdxList.size() != 0) {
        u32 toRankIdx = ctx.rank2ChannelIdx[nhrStepInfo.toRank];
        u32 sendSliceIdx = 0;
        size_t sendChannel = ctx.arg->channels[toRankIdx];
        ctx.localSrc.token = ctx.token[ctx.myRankIdx];
        ctx.remoteDst.token = ctx.token[toRankIdx];
        ctx.localDst.token = ctx.token[toRankIdx];
        for (u32 i = 0; i < sendSliceIdxList.size(); i++) {
            sendSliceIdx = sendSliceIdxList[i];
            HCCL_INFO("[CcuBroadcastNhr1DMem2Mem][DoScatterNHRSingleStep] sendSliceIdx[%u]", sendSliceIdx);
            if (i != 0) {
                if (i % RANK_NUM_PER_CKE == 0) {
                    ctx.event.SetMask((1 << RANK_NUM_PER_CKE) - 1);
                    ccu::EventWait(ctx.event, (1 << RANK_NUM_PER_CKE) - 1);
                }
            }
            if (nhrStepInfo.step == 0) {
                // 只有第0步的源数据从input中取
                ctx.localSrc.addr = ctx.input;
                ctx.localSrc.addr += ctx.sliceOffset[sendSliceIdx];
            } else {
                ctx.localSrc.addr = ctx.output[ctx.myRankIdx];
                ctx.localSrc.addr += ctx.sliceOffset[sendSliceIdx];
            }
            ctx.remoteDst.addr = ctx.output[toRankIdx];
            ctx.remoteDst.addr += ctx.sliceOffset[sendSliceIdx];
            ctx.localDst.addr = ctx.output[toRankIdx];
            ctx.localDst.addr += ctx.sliceOffset[sendSliceIdx];
            CCU_CHK_RET(DoSendRecvSlice(ctx, nhrStepInfo.toRank, ctx.localSrc, ctx.remoteDst, sendSliceIdx, i % RANK_NUM_PER_CKE));
        }
        ctx.event.SetMask((1 << (sendSliceIdxList.size() % RANK_NUM_PER_CKE)) - 1);
        ccu::EventWait(ctx.event, (1 << (sendSliceIdxList.size() % RANK_NUM_PER_CKE)) - 1);
        // 通知toRank数据写入完毕
        ccu::NotifyRecord(sendChannel, CKE_IDX_0, 1 << STEP_PRE_SYNC_ID);
    }
    // 只需要收
    if (recvSliceIdxList.size() != 0) {
        u32 fromRankIdx = ctx.rank2ChannelIdx[nhrStepInfo.fromRank];
        size_t recvChannel = ctx.arg->channels[fromRankIdx];
        ccu::NotifyWait(recvChannel, CKE_IDX_0, 1 << STEP_PRE_SYNC_ID);
    }
    return CCU_SUCCESS;
}

static CcuResult DoScatterNHR(BroadcastNhr1DMem2MemContext &ctx)
{
    const uint32_t NHR_NUM = 2;
    for (u64 i = 0; i < ctx.stepInfoVector.size() / NHR_NUM; i++) {
        const NHRStepInfo &nhrStepInfo = ctx.stepInfoVector[i];
        CCU_CHK_RET(DoScatterNHRSingleStep(ctx, nhrStepInfo));
    }
    return CCU_SUCCESS;
}

static CcuResult DoAllGatherNHRSingleStep(BroadcastNhr1DMem2MemContext &ctx, const NHRStepInfo &nhrStepInfo)
{
    u32 toRankIdx = ctx.rank2ChannelIdx[nhrStepInfo.toRank];
    u32 fromRankIdx = ctx.rank2ChannelIdx[nhrStepInfo.fromRank];
    u32 sendSliceIdx = 0;
    size_t sendChannel = ctx.arg->channels[toRankIdx];
    size_t recvChannel = ctx.arg->channels[fromRankIdx];
    const std::vector<u32> &sendSliceIdxList = nhrStepInfo.txSliceIdxs;
    ctx.localSrc.token = ctx.token[ctx.myRankIdx];
    ctx.remoteDst.token = ctx.token[toRankIdx];
    ctx.localDst.token = ctx.token[toRankIdx];

    for (u32 i = 0; i < sendSliceIdxList.size(); i++) {
        sendSliceIdx = sendSliceIdxList[i];

        if (i != 0) {
            if (i % RANK_NUM_PER_CKE == 0) {
                ctx.event.SetMask((1 << RANK_NUM_PER_CKE) - 1);
                ccu::EventWait(ctx.event, (1 << RANK_NUM_PER_CKE) - 1);
            }
        }

        ctx.localSrc.addr = ctx.output[ctx.myRankIdx];
        ctx.localSrc.addr += ctx.sliceOffset[sendSliceIdx];

        ctx.remoteDst.addr = ctx.output[toRankIdx];
        ctx.remoteDst.addr += ctx.sliceOffset[sendSliceIdx];
        ctx.localDst.addr = ctx.output[toRankIdx];
        ctx.localDst.addr += ctx.sliceOffset[sendSliceIdx];
        CCU_CHK_RET(DoSendRecvSlice(ctx, nhrStepInfo.toRank, ctx.localSrc, ctx.remoteDst, sendSliceIdx, i % RANK_NUM_PER_CKE));
    }

    ctx.event.SetMask((1 << (sendSliceIdxList.size() % RANK_NUM_PER_CKE)) - 1);
    ccu::EventWait(ctx.event, (1 << (sendSliceIdxList.size() % RANK_NUM_PER_CKE)) - 1);

    if (nhrStepInfo.step + 1 != ctx.stepInfoVector.size()) {   // 最后一步不需要同步
        // 通知toRank，写入完毕
        ccu::NotifyRecord(sendChannel, CKE_IDX_0, 1 << STEP_POST_SYNC_ID);
        // 等待fromRank通知写入完毕
        ccu::NotifyWait(recvChannel, CKE_IDX_0, 1 << STEP_POST_SYNC_ID);
    }

    HCCL_DEBUG("[CcuBroadcastNhr1DMem2Mem][DoAllGatherNHRSingleStep] rank %u step %u, toRank=%u, fromRank=%u, "
                "nSlice=%lu toRankIdx=%u, fromRankIdx=%u",
                ctx.rankId, nhrStepInfo.step, nhrStepInfo.toRank, nhrStepInfo.fromRank,
                sendSliceIdxList.size(), toRankIdx, fromRankIdx);
    return CCU_SUCCESS;
}

static CcuResult DoAllGatherNHR(BroadcastNhr1DMem2MemContext &ctx)
{
    const uint32_t NHR_NUM = 2;
    for (u64 i = ctx.stepInfoVector.size() / NHR_NUM; i < ctx.stepInfoVector.size(); i++) {
        const NHRStepInfo &nhrStepInfo = ctx.stepInfoVector[i];
        CCU_CHK_RET(DoAllGatherNHRSingleStep(ctx, nhrStepInfo));
    }
    return CCU_SUCCESS;
}

CcuResult CcuBroadcastNhr1DMem2MemKernel(CcuKernelArg arg)
{
    auto *kernelArg = static_cast<CcuKernelArgBroadcastNhr1DMem2Mem *>(arg);
    BroadcastNhr1DMem2MemContext ctx;
    ctx.arg = kernelArg;
    ctx.resourceAllocated = false;
    ctx.moConfig.msInterleave = 0;
    ctx.moConfig.loopCount = 0;
    ctx.moConfig.memSlice = 0;
    ctx.moRes.eventCount = 0;
    ctx.moRes.bufCount = 0;
    ctx.enginePool = 0;

    HCCL_INFO("[CcuBroadcastNhr1DMem2Mem] BroadcastNHR1D run");

    CCU_CHK_RET(ParseKernelArg(ctx, kernelArg));
    CCU_CHK_RET(InitResources(ctx));
    CCU_CHK_RET(LoadArgs(ctx));

    CCU_CHK_RET(PreSync(ctx));
    CCU_CHK_RET(DoScatterNHR(ctx));
    CCU_CHK_RET(DoAllGatherNHR(ctx));
    CCU_CHK_RET(PostSync(ctx));

    HCCL_INFO("[CcuBroadcastNhr1DMem2Mem] BroadcastNHR1D end");
    return CCU_SUCCESS;
}

} // namespace ops_hccl