/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "ccu_kernel_all_to_all_v_mesh2die.h"

namespace ops_hccl {

constexpr uint64_t MAX_TRANSPORT_SIZE = UB_MAX_TRANS_SIZE;
constexpr uint32_t CKE_IDX_0 = 0;
constexpr uint32_t CKE_IDX_1 = 1;
constexpr uint32_t CKE_IDX_2 = 2;
constexpr uint32_t INPUT_XN_ID = 0;
constexpr uint32_t OUTPUT_XN_ID = 1;
constexpr uint32_t TOKEN_XN_ID = 2;
constexpr uint32_t CONST_ONE = 1;

static CcuResult ParseKernelArg(AlltoAllVMesh2DieContext &ctx, CcuKernelArg arg)
{
    auto *kernelArg = static_cast<CcuKernelArgAlltoAllVMesh2Die *>(arg);
    ctx.arg = kernelArg;
    ctx.rankId = kernelArg->rankId;
    ctx.withMyRank = kernelArg->withMyRank;
    ctx.localSize = kernelArg->localSize;
    ctx.localId = kernelArg->localId;
    ctx.peerSize = kernelArg->peerSize;
    ctx.logicId = kernelArg->logicId;
    ctx.rankGroup = kernelArg->rankGroup;
    ctx.selfBit = 1 << ctx.localId;
    ctx.allBit = (1 << ctx.localSize) - 1;
    return CCU_SUCCESS;
}

static CcuResult InitResource(AlltoAllVMesh2DieContext &ctx)
{
    const auto *arg = ctx.arg;

    ctx.output.resize(arg->localSize);
    ctx.token.resize(arg->localSize);
    ctx.sendRecvInfo.resize(arg->peerSize);
    ctx.src.resize(arg->peerSize);
    ctx.dst.resize(arg->peerSize);

    uint32_t channelIdx = 0;
    for (uint32_t i = 0; i < arg->localSize; i++) {
        if (i != arg->localId) {
            ctx.output[i] = ccu::GetResByChannel<ccu::Variable>(arg->channels[channelIdx], OUTPUT_XN_ID);
            ctx.token[i] = ccu::GetResByChannel<ccu::Variable>(arg->channels[channelIdx], TOKEN_XN_ID);
            channelIdx++;
        }
    }

    for (uint32_t peerId = 0; peerId < arg->peerSize; peerId++) {
        uint32_t rankIdx = arg->rankGroup[peerId];
        if (rankIdx != arg->localId) {
            uint32_t chIdx = (rankIdx < arg->localId) ? rankIdx : rankIdx - 1;
            ctx.dst[peerId] = ccu::GetResByChannel<ccu::RemoteAddr>(arg->channels[chIdx], OUTPUT_XN_ID);
        }
    }

    LoopGroupConfig config{};
    config.msInterleave = CCU_MS_INTERLEAVE;
    config.loopCount = CCU_MS_LOCAL_COPY_LOOP_COUNT;
    config.memSlice = LOCAL_COPY_MS_PER_LOOP * CCU_MS_SIZE;
    ctx.moConfig = config;
    CCU_CHK_RET(AllocGoResource(ctx.moConfig, ctx.moRes, ctx.resourceAllocated,
        CCU_MS_LOCAL_COPY_LOOP_COUNT, LOCAL_COPY_MS_PER_LOOP));

    return CCU_SUCCESS;
}

static CcuResult LoadArgs(AlltoAllVMesh2DieContext &ctx)
{
    const auto *arg = ctx.arg;
    uint16_t argIdx = 0;

    CCU_CHK_RET(ccu::LoadArg(ctx.input, argIdx++));
    CCU_CHK_RET(ccu::LoadArg(ctx.output[arg->localId], argIdx++));
    CCU_CHK_RET(ccu::LoadArg(ctx.token[arg->localId], argIdx++));

    CCU_CHK_RET(ccu::LoadArg(ctx.xnMaxTransportGoSize.addrOffset, argIdx++));
    CCU_CHK_RET(ccu::LoadArg(ctx.xnMaxTransportGoSize.loopParam, argIdx++));
    CCU_CHK_RET(ccu::LoadArg(ctx.xnMaxTransportGoSize.parallelParam, argIdx++));
    CCU_CHK_RET(ccu::LoadArg(ctx.xnMaxTransportGoSize.residual, argIdx++));

    for (uint32_t peerId = 0; peerId < arg->peerSize; peerId++) {
        CCU_CHK_RET(ccu::LoadArg(ctx.sendRecvInfo[peerId].sendOffset, argIdx++));
        CCU_CHK_RET(ccu::LoadArg(ctx.sendRecvInfo[peerId].recvOffset, argIdx++));
        CCU_CHK_RET(ccu::LoadArg(ctx.sendRecvInfo[peerId].sendTailSize, argIdx++));
        CCU_CHK_RET(ccu::LoadArg(ctx.sendRecvInfo[peerId].sendTailGoSize.addrOffset, argIdx++));
        CCU_CHK_RET(ccu::LoadArg(ctx.sendRecvInfo[peerId].sendTailGoSize.loopParam, argIdx++));
        CCU_CHK_RET(ccu::LoadArg(ctx.sendRecvInfo[peerId].sendTailGoSize.parallelParam, argIdx++));
        CCU_CHK_RET(ccu::LoadArg(ctx.sendRecvInfo[peerId].sendTailGoSize.residual, argIdx++));
        CCU_CHK_RET(ccu::LoadArg(ctx.sendRecvInfo[peerId].sendLoopNum, argIdx++));
    }

    return CCU_SUCCESS;
}

static CcuResult ExchangeInfoSync(AlltoAllVMesh2DieContext &ctx)
{
    const auto *arg = ctx.arg;

    for (uint32_t peerId = 0; peerId < arg->localSize; peerId++) {
        if (peerId == arg->localId) {
            continue;
        }
        uint32_t chIdx = (peerId < arg->localId) ? peerId : peerId - 1;
        ccu::WriteVariableWithNotify(arg->channels[chIdx], ctx.localDst, OUTPUT_XN_ID,
            CKE_IDX_1, ctx.selfBit);
        ccu::WriteVariableWithNotify(arg->channels[chIdx], ctx.token[arg->localId], TOKEN_XN_ID,
            CKE_IDX_2, ctx.selfBit);
    }

    for (uint32_t peerId = 0; peerId < arg->localSize; peerId++) {
        if (peerId == arg->localId) {
            continue;
        }
        uint32_t chIdx2 = (peerId < arg->localId) ? peerId : peerId - 1;
        ccu::NotifyWait(arg->channels[chIdx2], CKE_IDX_1, 1 << peerId);
        ccu::NotifyWait(arg->channels[chIdx2], CKE_IDX_2, 1 << peerId);
    }

    return CCU_SUCCESS;
}

static CcuResult CalcGroupSrcDst(AlltoAllVMesh2DieContext &ctx)
{
    const auto *arg = ctx.arg;

    for (uint32_t peerId = 0; peerId < arg->peerSize; peerId++) {
        uint32_t rankIdx = arg->rankGroup[peerId];
        if (rankIdx == arg->localId) {
            ctx.localSrc = ccu::LocalAddr(ctx.input, ctx.sendRecvInfo[peerId].sendOffset);
            ctx.localDst = ccu::LocalAddr(ctx.output[arg->localId], ctx.sendRecvInfo[peerId].recvOffset);
        } else {
            ctx.src[peerId] = ccu::LocalAddr(ctx.input, ctx.sendRecvInfo[peerId].sendOffset);
            ctx.dst[peerId] = ccu::RemoteAddr(ctx.output[rankIdx], ctx.sendRecvInfo[peerId].recvOffset);
        }
    }

    return CCU_SUCCESS;
}

static CcuResult WriteToDstOutput(AlltoAllVMesh2DieContext &ctx, uint32_t peerId)
{
    const auto *arg = ctx.arg;
    uint32_t rankIdx = arg->rankGroup[peerId];
    uint32_t chIdx = (rankIdx < arg->localId) ? rankIdx : rankIdx - 1;
    uint16_t mask = 1 << peerId;

    CCU_WHILE(ctx.completedRankCount < ctx.sendRecvInfo[peerId].sendLoopNum) {
        ccu::Write(arg->channels[chIdx], ctx.dst[peerId], ctx.src[peerId],
            ctx.xnMaxTransportSize, ctx.event, mask);
        ccu::EventRecord(ctx.event, mask);
        ccu::EventWait(ctx.event, mask);
        ctx.completedRankCount = ctx.completedRankCount + ctx.xnConst1;
    }
    CCU_END_WHILE();

    ccu::Write(arg->channels[chIdx], ctx.dst[peerId], ctx.src[peerId],
        ctx.sendRecvInfo[peerId].sendTailSize, ctx.event, mask);
    ccu::EventRecord(ctx.event, mask);

    return CCU_SUCCESS;
}

static CcuResult GroupCopyToDstOutput(AlltoAllVMesh2DieContext &ctx)
{
    uint16_t mask = 1 << ctx.localId;

    CCU_CHK_RET(GroupCopy(ctx, ctx.localDst, ctx.localSrc, ctx.sendRecvInfo[ctx.localId].sendTailGoSize));
    ccu::EventRecord(ctx.event, mask);

    return CCU_SUCCESS;
}

static CcuResult LoopStep(AlltoAllVMesh2DieContext &ctx)
{
    const auto *arg = ctx.arg;

    for (uint32_t peerId = 0; peerId < arg->peerSize; peerId++) {
        uint32_t rankIdx = arg->rankGroup[peerId];
        if (rankIdx == arg->localId) {
            continue;
        }
        CCU_CHK_RET(WriteToDstOutput(ctx, peerId));
    }

    if (arg->withMyRank) {
        CCU_CHK_RET(GroupCopyToDstOutput(ctx));
    }

    ccu::EventWait(ctx.event, ctx.allBit);

    return CCU_SUCCESS;
}

static CcuResult DoAll2AllVMultiLoop(AlltoAllVMesh2DieContext &ctx)
{
    const auto *arg = ctx.arg;

    ctx.xnConst1 = 1;
    ctx.completedRankCount = 0;
    ctx.xnMaxTransportSize = MAX_TRANSPORT_SIZE;

    CCU_CHK_RET(LoopStep(ctx));

    return CCU_SUCCESS;
}

static CcuResult PostSync(AlltoAllVMesh2DieContext &ctx)
{
    const auto *arg = ctx.arg;

    for (uint32_t i = 0; i < arg->localSize; i++) {
        if (i == arg->localId) {
            continue;
        }
        uint32_t chIdx = (i < arg->localId) ? i : i - 1;
        ccu::NotifyRecord(arg->channels[chIdx], CKE_IDX_0, ctx.selfBit);
    }

    for (uint32_t i = 0; i < arg->localSize; i++) {
        if (i == arg->localId) {
            continue;
        }
        uint32_t chIdx2 = (i < arg->localId) ? i : i - 1;
        ccu::NotifyWait(arg->channels[chIdx2], CKE_IDX_0, 1 << i);
    }

    return CCU_SUCCESS;
}

CcuResult CcuAlltoAllVMesh2DieKernel(CcuKernelArg arg)
{
    AlltoAllVMesh2DieContext ctx;

    CCU_CHK_RET(ParseKernelArg(ctx, arg));
    CCU_CHK_RET(InitResource(ctx));
    CCU_CHK_RET(LoadArgs(ctx));
    CCU_CHK_RET(ExchangeInfoSync(ctx));
    CCU_CHK_RET(CalcGroupSrcDst(ctx));
    CCU_CHK_RET(DoAll2AllVMultiLoop(ctx));
    CCU_CHK_RET(PostSync(ctx));

    return CCU_SUCCESS;
}

}
