/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "ccu_kernel_reduce_scatter_mesh1d.h"

namespace ops_hccl {

constexpr int INPUT_XN_ID  = 0;
constexpr int TOKEN_XN_ID  = 1;
constexpr int POST_SYNC_ID = 2;
constexpr int CKE_IDX_0    = 0;

static CcuResult ParseKernelArg(ReduceScatterContext &ctx, CcuKernelArgReduceScatterMesh1D *kernelArg)
{
    // ctx.rankId          = kernelArg->rankId;
    // ctx.rankSize        = kernelArg->rankSize;
    // ctx.channels       = kernelArg->channels;
    ctx.dataType        = kernelArg->opParam.DataDes.dataType;
    ctx.outputDataType  = kernelArg->opParam.DataDes.outputType;
    if (ctx.outputDataType == HcclDataType::HCCL_DATA_TYPE_RESERVED) {
        ctx.outputDataType = ctx.dataType;
        HCCL_DEBUG("[CcuKernelReduceScatterMesh1D] outputDataType is [INVALID], set outputDataType to[%d]",
            ctx.dataType);
    }
    ctx.reduceOp = kernelArg->opParam.reduceType;
    return CCU_SUCCESS;
}

static CcuResult InitResource(ReduceScatterContext &ctx)
{
    const auto *arg = ctx.arg;
    uint32_t channelIdx = 0;

    if (arg->channelCount == 0) {
        HCCL_ERROR("[CcuKernelReduceScatterMesh1D] channels is empty!");
        return CcuResult::CCU_E_INTERNAL;
    }

    CCU_CHK_RET(ccu::Alloc(&ctx.output));
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

    CCU_CHK_RET(ccu::Alloc(&ctx.offset));

    CCU_CHK_RET(ccu::Alloc(&ctx.goSize.addrOffset));
    CCU_CHK_RET(ccu::Alloc(&ctx.goSize.loopParam));
    CCU_CHK_RET(ccu::Alloc(&ctx.goSize.parallelParam));
    CCU_CHK_RET(ccu::Alloc(&ctx.goSize.residual));

    CCU_CHK_RET(ccu::CreateLoopExecutor(&ctx.enginePool, RS_MAX_RANK_SIZE + 1));

    ctx.resourceAllocated = false;
    ctx.loopRegistered    = false;

    return CCU_SUCCESS;
}

static CcuResult LoadArgs(ReduceScatterContext &ctx)
{
    const auto *arg = ctx.arg;

    CCU_CHK_RET(ccu::LoadArg(ctx.input[arg->rankId]));
    CCU_CHK_RET(ccu::LoadArg(ctx.output));
    CCU_CHK_RET(ccu::LoadArg(ctx.token[arg->rankId]));
    CCU_CHK_RET(ccu::LoadArg(ctx.offset));
    CCU_CHK_RET(ccu::LoadArg(ctx.goSize.addrOffset));
    CCU_CHK_RET(ccu::LoadArg(ctx.goSize.loopParam));
    CCU_CHK_RET(ccu::LoadArg(ctx.goSize.parallelParam));
    CCU_CHK_RET(ccu::LoadArg(ctx.goSize.residual));

    return CCU_SUCCESS;
}

static void PreSync(ReduceScatterContext &ctx)
{
    const auto *arg = ctx.arg;

    for (uint32_t i = 0; i < arg->channelCount; i++) {
        ccu::WriteVariableWithNotify(arg->channels[i], ctx.input[arg->rankId],
            INPUT_XN_ID, CKE_IDX_0, 1 << INPUT_XN_ID);
        ccu::WriteVariableWithNotify(arg->channels[i], ctx.token[arg->rankId],
            TOKEN_XN_ID, CKE_IDX_0, 1 << TOKEN_XN_ID);
    }

    uint32_t allBit = (1 << INPUT_XN_ID) | (1 << TOKEN_XN_ID);
    for (uint32_t i = 0; i < arg->channelCount; i++) {
        ccu::NotifyWait(arg->channels[i], CKE_IDX_0, allBit);
    }
}

static void PostSync(ReduceScatterContext &ctx)
{
    const auto *arg = ctx.arg;

    for (uint32_t i = 0; i < arg->channelCount; i++) {
        ccu::NotifyRecord(arg->channels[i], CKE_IDX_0, 1 << POST_SYNC_ID);
    }
    for (uint32_t i = 0; i < arg->channelCount; i++) {
        ccu::NotifyWait(arg->channels[i], CKE_IDX_0, 1 << POST_SYNC_ID);
    }
}

static CcuResult DoReduceScatter(ReduceScatterContext &ctx)
{
    const auto *arg = ctx.arg;
    std::vector<ccu::RemoteAddr> src;
    ccu::LocalAddr localSrc;
    src.resize(arg->rankSize);
    for (uint32_t rankIdx = 0; rankIdx < arg->rankSize - 1; rankIdx++) {
        CCU_CHK_RET(ccu::Alloc(&src[rankIdx]));
    }
    CCU_CHK_RET(ccu::Alloc(&localSrc));
    ccu::LocalAddr dst;
    CCU_CHK_RET(ccu::Alloc(&dst));
    dst.addr  = ctx.output; // GSA[400] + Xn[0] to GSA[2](dst.addr)
    dst.token = ctx.token[arg->rankId]; // Load Xn[2] + Xn[413] to Xn[10]
    uint32_t curId = 0;
    for (uint32_t rankIdx = 0; rankIdx < arg->rankSize; rankIdx++) {
        if (rankIdx != arg->rankId) {
            src[curId].addr = ctx.input[rankIdx]; // Load GSA[400] + Xn[1] to GSA[1] | GSA[400] + Xn[400] to GSA[0]
            src[curId].addr += ctx.offset; //  Load GSA[1] + Xn[3] to GSA[1]         | GSA[0] + Xn[3] to GSA[0]
            src[curId].token = ctx.token[rankIdx]; // Load Xn[2] + Xn[413] to Xn[9]  | Xn[401] + Xn[413] to Xn[8]
            curId++;
        } else {
            localSrc.addr = ctx.input[rankIdx];
            localSrc.addr += ctx.offset;
            localSrc.token = ctx.token[rankIdx];
        }
    }

    HCCL_INFO("ctx.loopRegistered 5 [%d]", (int)ctx.loopRegistered);
    GroupReduce(ctx, arg->channels, arg->channelCount, dst, src, localSrc, ctx.goSize, ctx.dataType, ctx.outputDataType, ctx.reduceOp);

    return CCU_SUCCESS;
}

// ============================================================================
// 主入口 Kernel 函数
// ============================================================================
CcuResult CcuReduceScatterMesh1DKernel(CcuKernelArg arg)
{
    auto *kernelArg = static_cast<CcuKernelArgReduceScatterMesh1D *>(arg);

    ReduceScatterContext ctx;
    ctx.arg = kernelArg;
    ctx.resourceAllocated = false;
    ctx.loopRegistered = false;
    ctx.moConfig.msInterleave = 0;
    ctx.moConfig.loopCount = 0;
    ctx.moConfig.memSlice = 0;
    ctx.moRes.eventCount = 0;
    ctx.moRes.bufCount = 0;
    ctx.enginePool = 0;

    HCCL_INFO("[CcuKernelReduceScatterMesh1D] ReduceScatterMesh1D run");
    HCCL_INFO("ctx.loopRegistered 1 [%d]", (int)ctx.loopRegistered);
    CCU_CHK_RET(ParseKernelArg(ctx, kernelArg));
    CCU_CHK_RET(InitResource(ctx));
    CCU_CHK_RET(LoadArgs(ctx));
    HCCL_INFO("ctx.loopRegistered 2 [%d]", (int)ctx.loopRegistered);

    PreSync(ctx);

    HCCL_INFO("ctx.loopRegistered 3 [%d]", (int)ctx.loopRegistered);
    CCU_CHK_RET(DoReduceScatter(ctx));
    HCCL_INFO("ctx.loopRegistered 4 [%d]", (int)ctx.loopRegistered);


    PostSync(ctx);
    HCCL_INFO("[CcuKernelReduceScatterMesh1D] ReduceScatterMesh1D end");

    return CCU_SUCCESS;
}
} // namespace ops_hccl
