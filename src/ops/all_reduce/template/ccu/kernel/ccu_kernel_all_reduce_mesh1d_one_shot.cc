/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "ccu_kernel_all_reduce_mesh1d_one_shot.h"

namespace ops_hccl {

constexpr int INPUT_XN_ID  = 0;
constexpr int TOKEN_XN_ID  = 2;
constexpr int POST_SYNC_CKE_IDX   = 0;
constexpr int PRE_SYNC_CKE_IDX    = 1;
constexpr uint16_t POST_CKE_BIT0  = 0;

static CcuResult ParseKernelArg(AllReduceMesh1DOneShotContext &ctx, CcuKernelArgAllReduceMesh1DOneShot *kernelArg)
{
    // ctx.rankId          = kernelArg->rankId;
    // ctx.rankSize        = kernelArg->rankSize;
    // ctx.channels       = kernelArg->channels;
    ctx.dataType        = kernelArg->opParam.DataDes.dataType;
    ctx.outputDataType  = kernelArg->opParam.DataDes.outputType;
    if (ctx.outputDataType == HcclDataType::HCCL_DATA_TYPE_RESERVED) {
        ctx.outputDataType = ctx.dataType;
        HCCL_DEBUG("[CcuKernelAllReduceMesh1DOneShot] outputDataType is [INVALID], set outputDataType to[%d]",
            ctx.dataType);
    }
    ctx.reduceOp = kernelArg->opParam.reduceType;
    return CCU_SUCCESS;
}

static CcuResult InitResource(AllReduceMesh1DOneShotContext &ctx)
{
    HCCL_INFO("[CcuKernelAllReduceMesh1DOneShot] InitResource start");
    CCU_CHK_RET(ccu::Alloc(&ctx.output));
    const auto *arg = ctx.arg;
    uint32_t channelIdx = 0;

    if (arg->channelCount == 0) {
        HCCL_ERROR("[CcuKernelAllReduceMesh1DOneShot] channels is empty!");
        return CcuResult::CCU_E_INTERNAL;
    }

    // 按照rank号从小到大遍历channels，遇到本rank就填充本地资源，否则依次取远端资源，要求算法返回的Link同样是按顺序排列的
    ctx.input.resize(arg->rankSize);
    ctx.token.resize(arg->rankSize);
    for (uint64_t peerId = 0; peerId < arg->rankSize; peerId++) {
        if (peerId == arg->rankId) {
            CCU_CHK_RET(ccu::Alloc(&ctx.input[peerId]));
            CCU_CHK_RET(ccu::Alloc(&ctx.token[peerId]));
        } else {
            CCU_CHK_RET(ccu::CreateByChannel(
                arg->channels[channelIdx], INPUT_XN_ID, &ctx.input[peerId]));
            CCU_CHK_RET(ccu::CreateByChannel(
                arg->channels[channelIdx], TOKEN_XN_ID, &ctx.token[peerId]));
            channelIdx++;
        }
    }
    CCU_CHK_RET(ccu::Alloc(&ctx.goSize.addrOffset));
    CCU_CHK_RET(ccu::Alloc(&ctx.goSize.loopParam));
    CCU_CHK_RET(ccu::Alloc(&ctx.goSize.parallelParam));
    CCU_CHK_RET(ccu::Alloc(&ctx.goSize.residual));
    CCU_CHK_RET(ccu::CreateLoopExecutor(&ctx.enginePool, RS_MAX_RANK_SIZE + 1)); // todo

    ctx.resourceAllocated = false;  // todo
    ctx.loopRegistered    = false;  // todo

    return CCU_SUCCESS;
}

static CcuResult LoadArgs(AllReduceMesh1DOneShotContext &ctx)
{
    HCCL_INFO("[CcuKernelAllReduceMesh1DOneShot] LoadArgs start");
    const auto *arg = ctx.arg;

    CCU_CHK_RET(ccu::LoadArg(ctx.input[arg->rankId]));
    CCU_CHK_RET(ccu::LoadArg(ctx.output));
    CCU_CHK_RET(ccu::LoadArg(ctx.token[arg->rankId]));
    CCU_CHK_RET(ccu::LoadArg(ctx.goSize.addrOffset));
    CCU_CHK_RET(ccu::LoadArg(ctx.goSize.loopParam));
    CCU_CHK_RET(ccu::LoadArg(ctx.goSize.parallelParam));
    CCU_CHK_RET(ccu::LoadArg(ctx.goSize.residual));

    HCCL_INFO("[CcuKernelAllReduceMesh1DOneShot] LoadArgs end");
    return CCU_SUCCESS;
}

static void PreSync(AllReduceMesh1DOneShotContext &ctx)
{
    HCCL_INFO("[CcuKernelAllReduceMesh1DOneShot] Presync start");
    const auto *arg = ctx.arg;

    for (uint32_t i = 0; i < arg->channelCount; i++) {
        ccu::WriteVariableWithNotify(arg->channels[i], ctx.input[arg->rankId],
            INPUT_XN_ID, PRE_SYNC_CKE_IDX, 1 << INPUT_XN_ID);
        ccu::WriteVariableWithNotify(arg->channels[i], ctx.token[arg->rankId],
            TOKEN_XN_ID, PRE_SYNC_CKE_IDX, 1 << TOKEN_XN_ID);
    }

    uint32_t allBit = (1 << INPUT_XN_ID) | (1 << TOKEN_XN_ID);
    for (uint32_t i = 0; i < arg->channelCount; i++) {
        ccu::NotifyWait(arg->channels[i], PRE_SYNC_CKE_IDX, allBit);
    }

    HCCL_INFO("[CcuKernelAllReduceMesh1DOneShot] Presync end");
}

static void PostSync(AllReduceMesh1DOneShotContext &ctx)
{
    HCCL_INFO("[CcuKernelAllReduceMesh1DOneShot] Postsync start");
    const auto *arg = ctx.arg;

    for (uint32_t i = 0; i < arg->channelCount; i++) {
        ccu::NotifyRecord(arg->channels[i], POST_SYNC_CKE_IDX, 1 << POST_CKE_BIT0);
    }
    for (uint32_t i = 0; i < arg->channelCount; i++) {
        ccu::NotifyWait(arg->channels[i], POST_SYNC_CKE_IDX, 1 << POST_CKE_BIT0);
    }
    HCCL_INFO("[CcuKernelAllReduceMesh1DOneShot] Postsync end");
}

static CcuResult DoGroupReduce(AllReduceMesh1DOneShotContext &ctx)
{
    const auto *arg = ctx.arg;
    std::vector<ccu::RemoteAddr> reduceSrc;
    ccu::LocalAddr localSrc;
    reduceSrc.resize(arg->rankSize);
    for (uint32_t rankIdx = 0; rankIdx < arg->rankSize - 1; rankIdx++) {
        CCU_CHK_RET(ccu::Alloc(&reduceSrc[rankIdx]));
    }
    CCU_CHK_RET(ccu::Alloc(&localSrc));
    ccu::LocalAddr reduceDst;
    CCU_CHK_RET(ccu::Alloc(&reduceDst));
    reduceDst.addr  = ctx.output;
    reduceDst.token = ctx.token[arg->rankId];
    uint32_t curId = 0;
    for (uint32_t rankIdx = 0; rankIdx < arg->rankSize; rankIdx++) {
        if (rankIdx != arg->rankId) {
            reduceSrc[curId].addr = ctx.input[rankIdx];
            reduceSrc[curId].token = ctx.token[rankIdx];
            curId++;
        } else {
            localSrc.addr = ctx.input[rankIdx];
            localSrc.token = ctx.token[rankIdx];
        }
    }

    HCCL_INFO("ctx.loopRegistered 5 [%d]", (int)ctx.loopRegistered);
    GroupReduce(ctx, arg->channels, arg->channelCount, reduceDst, reduceSrc, localSrc, ctx.goSize, ctx.dataType, ctx.outputDataType, ctx.reduceOp);

    return CCU_SUCCESS;
}

// ============================================================================
// 主入口 Kernel 函数
// ============================================================================
CcuResult CcuAllReduceMesh1DOneShotKernel(CcuKernelArg arg)
{
    auto *kernelArg = static_cast<CcuKernelArgAllReduceMesh1DOneShot *>(arg);

    AllReduceMesh1DOneShotContext ctx;
    ctx.arg = kernelArg;
    ctx.resourceAllocated = false;
    ctx.loopRegistered = false;
    ctx.moConfig.msInterleave = 0;
    ctx.moConfig.loopCount = 0;
    ctx.moConfig.memSlice = 0;
    ctx.moRes.eventCount = 0;
    ctx.moRes.bufCount = 0;
    ctx.enginePool = 0;

    HCCL_INFO("[CcuKernelAllReduceMesh1DOneShot] AllReduceMesh1DOneShot start");
    CCU_CHK_RET(ParseKernelArg(ctx, kernelArg));
    CCU_CHK_RET(InitResource(ctx));
    CCU_CHK_RET(LoadArgs(ctx));

    PreSync(ctx);

    CCU_CHK_RET(DoGroupReduce(ctx));

    PostSync(ctx);
    HCCL_INFO("[CcuKernelAllReduceMesh1DOneShot] AllReduceMesh1DOneShot end");

    return CCU_SUCCESS;
}
}