/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "ccu_kernel_all_reduce_mesh1d.h"

namespace ops_hccl {

constexpr int INPUT_XN_ID  = 0;
constexpr int OUTPUT_XN_ID = 1;
constexpr int TOKEN_XN_ID  = 2;
constexpr int POST_SYNC_ID = 3;
constexpr int CKE_IDX_0    = 0;

static CcuResult ParseKernelArg(AllReduceMesh1DContext &ctx, CcuKernelArgAllReduceMesh1D *kernelArg)
{
    ctx.dataType        = kernelArg->opParam.DataDes.dataType;
    ctx.outputDataType  = kernelArg->opParam.DataDes.outputType;
    if (ctx.outputDataType == HcclDataType::HCCL_DATA_TYPE_RESERVED) {
        ctx.outputDataType = ctx.dataType;
        HCCL_DEBUG("[CcuAllReduceMesh1D] outputDataType is [INVALID], set outputDataType to[%d]",
            ctx.dataType);
    }
    ctx.reduceOp = kernelArg->opParam.reduceType;
    return CCU_SUCCESS;
}

static CcuResult InitResource(AllReduceMesh1DContext &ctx)
{
    const auto *arg = ctx.arg;
    uint32_t channelIdx = 0;

    if (arg->channelCount == 0) {
        HCCL_ERROR("[CcuAllReduceMesh1D] channels is empty!");
        return CcuResult::CCU_E_INTERNAL;
    }

    ctx.input.resize(arg->rankSize);
    ctx.output.resize(arg->rankSize);
    ctx.token.resize(arg->rankSize);
    for (uint64_t peerId = 0; peerId < arg->rankSize; peerId++) {
        if (peerId == arg->rankId) {
            CCU_CHK_RET(ccu::Alloc(&ctx.input[peerId]));
            CCU_CHK_RET(ccu::Alloc(&ctx.output[peerId]));
            CCU_CHK_RET(ccu::Alloc(&ctx.token[peerId]));
        } else {
            CCU_CHK_RET(ccu::CreateByChannel(
                arg->channels[channelIdx], INPUT_XN_ID, &ctx.input[peerId]));
            CCU_CHK_RET(ccu::CreateByChannel(
                arg->channels[channelIdx], OUTPUT_XN_ID, &ctx.output[peerId]));
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

    CCU_CHK_RET(ccu::CreateLoopExecutor(&ctx.enginePool, arg->rankSize + 1));

    ctx.resourceAllocated = false;
    ctx.loopRegistered    = false;

    return CCU_SUCCESS;
}

static CcuResult LoadArgs(AllReduceMesh1DContext &ctx)
{
    const auto *arg = ctx.arg;

    CCU_CHK_RET(ccu::LoadArg(ctx.input[arg->rankId]));
    CCU_CHK_RET(ccu::LoadArg(ctx.output[arg->rankId]));
    CCU_CHK_RET(ccu::LoadArg(ctx.token[arg->rankId]));
    CCU_CHK_RET(ccu::LoadArg(ctx.offset));
    CCU_CHK_RET(ccu::LoadArg(ctx.goSize.addrOffset));
    CCU_CHK_RET(ccu::LoadArg(ctx.goSize.loopParam));
    CCU_CHK_RET(ccu::LoadArg(ctx.goSize.parallelParam));
    CCU_CHK_RET(ccu::LoadArg(ctx.goSize.residual));

    return CCU_SUCCESS;
}

static void PreSync(AllReduceMesh1DContext &ctx)
{
    const auto *arg = ctx.arg;

    for (uint32_t i = 0; i < arg->channelCount; i++) {
        ccu::WriteVariableWithNotify(arg->channels[i], ctx.input[arg->rankId],
            INPUT_XN_ID, CKE_IDX_0, 1 << INPUT_XN_ID);
        ccu::WriteVariableWithNotify(arg->channels[i], ctx.output[arg->rankId],
            OUTPUT_XN_ID, CKE_IDX_0, 1 << OUTPUT_XN_ID);
        ccu::WriteVariableWithNotify(arg->channels[i], ctx.token[arg->rankId],
            TOKEN_XN_ID, CKE_IDX_0, 1 << TOKEN_XN_ID);
    }

    uint32_t allBit = (1 << INPUT_XN_ID) | (1 << OUTPUT_XN_ID) | (1 << TOKEN_XN_ID);
    for (uint32_t i = 0; i < arg->channelCount; i++) {
        ccu::NotifyWait(arg->channels[i], CKE_IDX_0, allBit);
    }
}

static void PostSync(AllReduceMesh1DContext &ctx)
{
    const auto *arg = ctx.arg;

    for (uint32_t i = 0; i < arg->channelCount; i++) {
        ccu::NotifyRecord(arg->channels[i], CKE_IDX_0, 1 << POST_SYNC_ID);
    }
    for (uint32_t i = 0; i < arg->channelCount; i++) {
        ccu::NotifyWait(arg->channels[i], CKE_IDX_0, 1 << POST_SYNC_ID);
    }
}

static CcuResult DoAllReduce(AllReduceMesh1DContext &ctx)
{
    const auto *arg = ctx.arg;
    
    // ReduceScatter phase
    std::vector<ccu::RemoteAddr> reduceScatterSrc;
    reduceScatterSrc.resize(arg->rankSize - 1);
    for (uint32_t rankIdx = 0; rankIdx < arg->rankSize - 1; rankIdx++) {
        CCU_CHK_RET(ccu::Alloc(&reduceScatterSrc[rankIdx]));
    }
    ccu::LocalAddr reduceScatterDst;
    CCU_CHK_RET(ccu::Alloc(&reduceScatterDst));
    reduceScatterDst.addr  = ctx.output[arg->rankId];
    reduceScatterDst.addr += ctx.offset;
    reduceScatterDst.token = ctx.token[arg->rankId];

    ccu::LocalAddr localSrc;
    CCU_CHK_RET(ccu::Alloc(&localSrc));
    
    uint32_t curId = 0;
    for (uint64_t rankIdx = 0; rankIdx < arg->rankSize; rankIdx++) {
        if (rankIdx != arg->rankId) {
            reduceScatterSrc[curId].addr = ctx.input[rankIdx];
            reduceScatterSrc[curId].addr += ctx.offset;
            reduceScatterSrc[curId].token = ctx.token[rankIdx];
            curId++;
        } else {
            localSrc.addr = ctx.input[rankIdx];
            localSrc.addr += ctx.offset;
            localSrc.token = ctx.token[rankIdx];
        }
    }
    GroupReduce(ctx, arg->channels, arg->channelCount, reduceScatterDst, reduceScatterSrc, localSrc, ctx.goSize, ctx.dataType, ctx.outputDataType, ctx.reduceOp);

    // AllGather phase
    ccu::LocalAddr allGatherSrc;
    CCU_CHK_RET(ccu::Alloc(&allGatherSrc));
    allGatherSrc.addr  = ctx.output[arg->rankId];
    allGatherSrc.addr += ctx.offset;
    allGatherSrc.token = ctx.token[arg->rankId];
    
    std::vector<ccu::RemoteAddr> allGatherDst;
    allGatherDst.resize(arg->rankSize - 1);
    for (uint32_t rankIdx = 0; rankIdx < arg->rankSize - 1; rankIdx++) {
        CCU_CHK_RET(ccu::Alloc(&allGatherDst[rankIdx]));
    }
    
    ccu::LocalAddr localDst;
    CCU_CHK_RET(ccu::Alloc(&localDst));
    
    curId = 0;
    for (uint64_t rankIdx = 0; rankIdx < arg->rankSize; rankIdx++) {
        if (rankIdx != arg->rankId) {
            allGatherDst[curId].addr = ctx.output[rankIdx];
            allGatherDst[curId].addr += ctx.offset;
            allGatherDst[curId].token = ctx.token[rankIdx];
            curId++;
        } else {
            localDst.addr = ctx.output[rankIdx];
            localDst.addr += ctx.offset;
            localDst.token = ctx.token[rankIdx];
        }
    }
    GroupBroadcast(ctx, arg->channels, arg->channelCount, localDst, allGatherDst, allGatherSrc, ctx.goSize);

    return CCU_SUCCESS;
}

// ============================================================================
// 主入口 Kernel 函数
// ============================================================================
CcuResult CcuAllReduceMesh1DKernel(CcuKernelArg arg)
{
    auto *kernelArg = static_cast<CcuKernelArgAllReduceMesh1D *>(arg);

    AllReduceMesh1DContext ctx;
    ctx.arg = kernelArg;
    ctx.resourceAllocated = false;
    ctx.loopRegistered = false;
    ctx.moConfig.msInterleave = 0;
    ctx.moConfig.loopCount = 0;
    ctx.moConfig.memSlice = 0;
    ctx.moRes.eventCount = 0;
    ctx.moRes.bufCount = 0;
    ctx.enginePool = 0;

    HCCL_INFO("[CcuAllReduceMesh1D] AllReduceMesh1D run");
    CCU_CHK_RET(ParseKernelArg(ctx, kernelArg));
    CCU_CHK_RET(InitResource(ctx));
    CCU_CHK_RET(LoadArgs(ctx));

    PreSync(ctx);

    CCU_CHK_RET(DoAllReduce(ctx));

    PostSync(ctx);
    HCCL_INFO("[CcuAllReduceMesh1D] AllReduceMesh1D end");

    return CCU_SUCCESS;
}
} // namespace ops_hccl