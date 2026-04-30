/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#include "ccu_kernel_reduce_scatter_nhr1d_multi_jetty_mem2mem.h"
#include "ccu_kernel_alg_base.h"

namespace ops_hccl {

constexpr int INPUT_XN_ID   = 0;
constexpr int TOKEN_XN_ID   = 1;

constexpr int CKE_IDX_INPUT = 0;
constexpr int CKE_IDX_TOKEN = 1;
constexpr int CKE_IDX_READY = 2;
constexpr int CKE_IDX_DONE  = 3;
constexpr int POST_XN_ID    = 4;
constexpr uint16_t BIT_NUM_PER_CKE = 16;

static CcuResult ParseKernelArg(ReduceScatterNhrMultiJettyMem2Mem1DContext &ctx,
                                CcuKernelArgReduceScatterNhrMultiJettyMem2Mem1D *kernelArg)
{
    ctx.dataType       = kernelArg->opParam.DataDes.dataType;
    ctx.outputDataType = kernelArg->opParam.DataDes.outputType;
    if (ctx.outputDataType == HcclDataType::HCCL_DATA_TYPE_RESERVED) {
        ctx.outputDataType = ctx.dataType;
        HCCL_DEBUG("[CcuReduceScatterNhrMultiJettyMem2Mem1D] outputDataType is [INVALID], set outputDataType to[%d]",
                   ctx.dataType);
    }
    ctx.reduceOp = kernelArg->opParam.reduceType;
    return CCU_SUCCESS;
}

static CcuResult InitResource(ReduceScatterNhrMultiJettyMem2Mem1DContext &ctx)
{
    const auto *arg = ctx.arg;

    if (arg->channelCount == 0) {
        HCCL_ERROR("[CcuReduceScatterNhrMultiJettyMem2Mem1D] channels is empty!");
        return CcuResult::CCU_E_INTERNAL;
    }

    CCU_CHK_RET(ccu::Alloc(&ctx.output));
    CCU_CHK_RET(ccu::Alloc(&ctx.sliceSize));
    CCU_CHK_RET(ccu::Alloc(&ctx.inputSliceStride));
    CCU_CHK_RET(ccu::Alloc(&ctx.inputRepeatStride));
    CCU_CHK_RET(ccu::Alloc(&ctx.outputRepeatStride));
    CCU_CHK_RET(ccu::Alloc(&ctx.sliceOneJettySize));
    CCU_CHK_RET(ccu::Alloc(&ctx.sliceLastJettySize));
    CCU_CHK_RET(ccu::Alloc(&ctx.repeatNumVar));
    CCU_CHK_RET(ccu::Alloc(&ctx.repeatNumVarTemp));
    CCU_CHK_RET(ccu::Alloc(&ctx.flag));

    uint32_t localSize = arg->rank2ChannelIdx.size();
    ctx.input.resize(localSize + 1);
    ctx.token.resize(localSize + 1);

    for (uint32_t channelIdx = 0; channelIdx < localSize; channelIdx++) {
        CCU_CHK_RET(ccu::CreateByChannel(
            arg->channels[channelIdx], INPUT_XN_ID, &ctx.input[channelIdx]));
        CCU_CHK_RET(ccu::CreateByChannel(
            arg->channels[channelIdx], TOKEN_XN_ID, &ctx.token[channelIdx]));
    }

    CCU_CHK_RET(ccu::Alloc(&ctx.input[localSize]));
    CCU_CHK_RET(ccu::Alloc(&ctx.token[localSize]));

    CCU_CHK_RET(ccu::Alloc(&ctx.localSrc));
    CCU_CHK_RET(ccu::Alloc(&ctx.localDst));
    CCU_CHK_RET(ccu::Alloc(&ctx.remoteDst));

    CCU_CHK_RET(ccu::Alloc(&ctx.event));

    ctx.jettyEvents.resize(arg->portSize);
    for (uint32_t jettyId = 0; jettyId < arg->portSize; jettyId++) {
        CCU_CHK_RET(ccu::Alloc(&ctx.jettyEvents[jettyId]));
    }

    ctx.resourceAllocated = false;

    HCCL_INFO("[CcuReduceScatterNhrMultiJettyMem2Mem1D] InitResource success!");
    return CCU_SUCCESS;
}

static CcuResult LoadArgs(ReduceScatterNhrMultiJettyMem2Mem1DContext &ctx)
{
    const auto *arg = ctx.arg;
    uint32_t myRankIdx = arg->rank2ChannelIdx.size();

    CCU_CHK_RET(ccu::LoadArg(ctx.input[myRankIdx]));
    CCU_CHK_RET(ccu::LoadArg(ctx.output));
    CCU_CHK_RET(ccu::LoadArg(ctx.token[myRankIdx]));
    CCU_CHK_RET(ccu::LoadArg(ctx.sliceSize));
    CCU_CHK_RET(ccu::LoadArg(ctx.inputSliceStride));
    CCU_CHK_RET(ccu::LoadArg(ctx.sliceOneJettySize));
    CCU_CHK_RET(ccu::LoadArg(ctx.sliceLastJettySize));
    CCU_CHK_RET(ccu::LoadArg(ctx.repeatNumVar));
    CCU_CHK_RET(ccu::LoadArg(ctx.inputRepeatStride));
    CCU_CHK_RET(ccu::LoadArg(ctx.outputRepeatStride));
    ctx.repeatNumVarTemp = ctx.repeatNumVar;

    HCCL_INFO("[CcuReduceScatterNhrMultiJettyMem2Mem1D] LoadArgs success!");
    return CCU_SUCCESS;
}

static uint32_t GetSignalIndex(const int signalBit)
{
    // 一个CKE有16位，可以处理16个用途
    return static_cast<uint32_t>(signalBit) / BIT_NUM_PER_CKE;
}

static uint16_t GetSignalMask(const int signalBit)
{
    return (1 << (static_cast<uint32_t>(signalBit) % BIT_NUM_PER_CKE));
}

static void PreSync(ReduceScatterNhrMultiJettyMem2Mem1DContext &ctx)
{
    const auto *arg = ctx.arg;
    uint32_t myRankIdx = arg->rank2ChannelIdx.size();

    HCCL_INFO("[CcuReduceScatterNhrMultiJettyMem2Mem1D] PreSync start");

    const uint16_t signalBitInput = GetSignalMask(CKE_IDX_INPUT);
    const uint16_t signalBitToken = GetSignalMask(CKE_IDX_TOKEN);
    const uint32_t signalIndexInput = GetSignalIndex(CKE_IDX_INPUT);
    const uint32_t signalIndexToken = GetSignalIndex(CKE_IDX_TOKEN);

    for (uint32_t i = 0; i < arg->channelCount; i++) {
        ccu::WriteVariableWithNotify(arg->channels[i], ctx.input[myRankIdx],
            INPUT_XN_ID, signalIndexInput, signalBitInput);
        ccu::WriteVariableWithNotify(arg->channels[i], ctx.token[myRankIdx],
            TOKEN_XN_ID, signalIndexToken, signalBitToken);
    }

    // 等所有对端
    const uint16_t waitMask = signalBitInput | signalBitToken;
    std::set<uint32_t> signalIdxes{signalIndexInput, signalIndexToken};
    for (uint32_t i = 0; i < arg->channelCount; i++) {
        for (uint32_t signalIdx : signalIdxes) {
            ccu::NotifyWait(arg->channels[i], signalIdx, waitMask);
        }
    }

    HCCL_INFO("[CcuReduceScatterNhrMultiJettyMem2Mem1D] PreSync end");
}

static void PostSync(ReduceScatterNhrMultiJettyMem2Mem1DContext &ctx)
{
    const auto *arg = ctx.arg;

    HCCL_INFO("[CcuReduceScatterNhrMultiJettyMem2Mem1D] PostSync start");

    const uint16_t selfBitInput = GetSignalMask(POST_XN_ID);
    const uint32_t signalIndexInput = GetSignalIndex(POST_XN_ID);

    // 通知所有对端
    for (uint32_t i = 0; i < arg->channelCount; i++) {
        ccu::NotifyRecord(arg->channels[i], signalIndexInput, selfBitInput);
    }

    // 等待所有需要的对端
    for (uint32_t i = 0; i < arg->channelCount; i++) {
        ccu::NotifyWait(arg->channels[i], signalIndexInput, selfBitInput);
    }

    HCCL_INFO("[CcuReduceScatterNhrMultiJettyMem2Mem1D] PostSync end");
}

static CcuResult DoRepeatSendRecvSlices(ReduceScatterNhrMultiJettyMem2Mem1DContext &ctx,
                                        const u32 &toRank, ccu::LocalAddr &src, ccu::RemoteAddr &dst)
{
    const auto *arg = ctx.arg;
    const u32 toRankIdx = arg->rank2ChannelIdx.at(toRank);
    const ChannelHandle sendChannel = arg->channels[toRankIdx];

    CcuVariable repeatNumAdd;
    CCU_CHK_RET(ccu::Alloc(&repeatNumAdd));
    repeatNumAdd = 1;
    ctx.flag = 0;
    ctx.repeatNumVarTemp = ctx.repeatNumVar;

    CCU_WHILE(ctx.repeatNumVarTemp != UINT64_MAX) {
        CCU_IF(ctx.repeatNumVarTemp != UINT64_MAX) {
            ctx.repeatNumVarTemp += repeatNumAdd;
        }
        CCU_IF(ctx.flag == 1) {
            src.addr += ctx.inputRepeatStride;
            dst.addr += ctx.inputRepeatStride;
        }

        ccu::LocalAddr tempSrc;
        ccu::RemoteAddr tempDst;
        CCU_CHK_RET(ccu::Alloc(&tempSrc));
        CCU_CHK_RET(ccu::Alloc(&tempDst));
        tempSrc.addr = src.addr;
        tempSrc.token = src.token;
        tempDst.addr = dst.addr;
        tempDst.token = dst.token;

        CCU_IF(ctx.sliceOneJettySize == 0) {
            for (u32 jettyId = 0; jettyId < arg->portSize - 1; jettyId++) {
                ctx.jettyEvents[jettyId].mask = 1;
                ccu::RecordEvent(ctx.jettyEvents[jettyId]);
            }
        }
        CCU_IF(ctx.sliceOneJettySize != 0) {
            for (u32 jettyId = 0; jettyId < arg->portSize - 1; jettyId++) {
                ctx.jettyEvents[jettyId].mask = 1;
                CCU_CHK_RET(ccu::WriteReduceNb(sendChannel, tempDst, tempSrc, ctx.sliceOneJettySize,
                    ctx.dataType, ctx.reduceOp, ctx.jettyEvents[jettyId]));
                tempDst.addr += ctx.sliceOneJettySize;
                tempSrc.addr += ctx.sliceOneJettySize;
            }
        }
        CCU_IF(ctx.sliceLastJettySize == 0) {
            ctx.jettyEvents[arg->portSize - 1].mask = 1;
            ccu::RecordEvent(ctx.jettyEvents[arg->portSize - 1]);
        }
        CCU_IF(ctx.sliceLastJettySize != 0) {
            u32 jettyId = arg->portSize - 1;
            ctx.jettyEvents[jettyId].mask = 1;
            CCU_CHK_RET(ccu::WriteReduceNb(sendChannel, tempDst, tempSrc, ctx.sliceLastJettySize,
                    ctx.dataType, ctx.reduceOp, ctx.jettyEvents[jettyId]));
        }
        for (u32 jettyId = 0; jettyId < arg->portSize; jettyId++) {
            ccu::WaitEvent(ctx.jettyEvents[jettyId]);
        }
        ctx.flag = 1;
    }
    ctx.flag = 0;
    return CCU_SUCCESS;
}

static CcuResult DoRepeatReduceScatterNHRSingleStep(ReduceScatterNhrMultiJettyMem2Mem1DContext &ctx,
    const NHRStepInfo &nhrStepInfo, const std::vector<CcuVariable> &inputSliceOffset)
{
    const auto *arg = ctx.arg;
    const u32 toRankIdx = arg->rank2ChannelIdx.at(nhrStepInfo.toRank);
    const u32 fromRankIdx = arg->rank2ChannelIdx.at(nhrStepInfo.fromRank);
    const ChannelHandle sendChannel = arg->channels[toRankIdx];
    const ChannelHandle recvChannel = arg->channels[fromRankIdx];
    const std::vector<u32> &sendSliceIdxList = nhrStepInfo.txSliceIdxs;

    uint32_t myRankIdx = arg->rank2ChannelIdx.size();
    ctx.remoteDst.token = ctx.token[toRankIdx];
    ctx.localSrc.token = ctx.token[myRankIdx];

    const uint32_t signalIdxReady = GetSignalIndex(CKE_IDX_READY);
    const uint32_t signalIdxDone = GetSignalIndex(CKE_IDX_DONE);
    const uint16_t signalBitReady = GetSignalMask(CKE_IDX_READY);
    const uint16_t signalBitDone = GetSignalMask(CKE_IDX_DONE);

    if (nhrStepInfo.step != 0) {
        ccu::NotifyRecord(recvChannel, signalIdxReady, signalBitReady);
        ccu::NotifyWait(sendChannel, signalIdxReady, signalBitReady);
    }

    for (const u32 &sendSliceIdx : sendSliceIdxList) {
        ctx.remoteDst.addr = ctx.input[toRankIdx];
        ctx.remoteDst.addr += inputSliceOffset[sendSliceIdx];
        ctx.localSrc.addr = ctx.input[myRankIdx];
        ctx.localSrc.addr += inputSliceOffset[sendSliceIdx];
        CCU_CHK_RET(DoRepeatSendRecvSlices(ctx, nhrStepInfo.toRank, ctx.localSrc, ctx.remoteDst));
        HCCL_INFO("[CcuReduceScatterNhrMultiJettyMem2Mem1D] DoRepeatSendRecvSlices success");
    }

    ccu::NotifyRecord(sendChannel, signalIdxDone, signalBitDone);
    ccu::NotifyWait(recvChannel, signalIdxDone, signalBitDone);

    HCCL_INFO("[CcuReduceScatterNhrMultiJettyMem2Mem1D] DoRepeatReduceScatterNHRSingleStep success");
    return CCU_SUCCESS;
}

static CcuResult DoRepeatReduceScatter(ReduceScatterNhrMultiJettyMem2Mem1DContext &ctx)
{
    const auto *arg = ctx.arg;
    uint32_t myRankIdx = arg->rank2ChannelIdx.size();

    CcuVariable tmpSliceOffset;
    CCU_CHK_RET(ccu::Alloc(&tmpSliceOffset));
    tmpSliceOffset = 0;

    std::vector<CcuVariable> inputSliceOffset;
    inputSliceOffset.resize(arg->dimSize);
    for (u64 i = 0; i < arg->dimSize; i++) {
        CCU_CHK_RET(ccu::Alloc(&inputSliceOffset[i]));
        inputSliceOffset[i] = tmpSliceOffset;
        tmpSliceOffset += ctx.inputSliceStride;
    }

    for (auto &nhrStepInfo : arg->stepInfoVector) {
        CCU_CHK_RET(DoRepeatReduceScatterNHRSingleStep(ctx, nhrStepInfo, inputSliceOffset));
    }

    ctx.localDst.addr = ctx.output;
    ctx.localDst.token = ctx.token[myRankIdx];
    ctx.localSrc.addr = ctx.input[myRankIdx];
    ctx.localSrc.addr += inputSliceOffset[arg->rankId];
    ctx.localSrc.token = ctx.token[myRankIdx];

    CcuVariable repeatNumAdd2;
    CCU_CHK_RET(ccu::Alloc(&repeatNumAdd2));
    repeatNumAdd2 = 1;

    CCU_WHILE(ctx.repeatNumVar != UINT64_MAX) {
        ctx.repeatNumVar += repeatNumAdd2;
        CCU_IF(ctx.flag == 1) {
            ctx.localSrc.addr += ctx.inputRepeatStride;
            ctx.localDst.addr += ctx.outputRepeatStride;
        }
        ctx.event.mask = 1;
        CCU_IF(ctx.sliceSize == 0) {
            ccu::RecordEvent(ctx.event);
        }
        CCU_IF(ctx.sliceSize != 0) {
            CCU_CHK_RET(ccu::LocalCopyNb(ctx.localDst, ctx.localSrc, ctx.sliceSize, ctx.event));
        }
        ccu::WaitEvent(ctx.event);
        ctx.flag = 1;
    }

    HCCL_INFO("[CcuReduceScatterNhrMultiJettyMem2Mem1D] DoRepeatReduceScatter success");
    return CCU_SUCCESS;
}

CcuResult CcuReduceScatterNhrMultiJettyMem2Mem1DKernel(CcuKernelArg arg)
{
    auto *kernelArg = static_cast<CcuKernelArgReduceScatterNhrMultiJettyMem2Mem1D *>(arg);

    ReduceScatterNhrMultiJettyMem2Mem1DContext ctx;
    ctx.arg = kernelArg;
    ctx.resourceAllocated = false;
    ctx.moConfig.msInterleave = 0;
    ctx.moConfig.loopCount    = 0;
    ctx.moConfig.memSlice     = 0;
    ctx.moRes.eventCount      = 0;
    ctx.moRes.bufCount        = 0;
    ctx.enginePool            = 0;

    HCCL_INFO("[CcuReduceScatterNhrMultiJettyMem2Mem1D] ReduceScatterNhrMultiJettyMem2Mem1D run");
    CCU_CHK_RET(ParseKernelArg(ctx, kernelArg));
    CCU_CHK_RET(InitResource(ctx));
    CCU_CHK_RET(LoadArgs(ctx));

    PreSync(ctx);

    CCU_CHK_RET(DoRepeatReduceScatter(ctx));

    PostSync(ctx);
    HCCL_INFO("[CcuReduceScatterNhrMultiJettyMem2Mem1D] ReduceScatterNhrMultiJettyMem2Mem1D end");

    return CCU_SUCCESS;
}

} // namespace ops_hccl