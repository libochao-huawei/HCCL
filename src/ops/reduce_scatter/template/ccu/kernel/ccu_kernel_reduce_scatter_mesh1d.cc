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

static HcclResult ParseKernelArg(ReduceScatterContext &ctx, CcuKernelArgReduceScatterMesh1D *kernelArg)
{
    ctx.rankId          = kernelArg->rankId;
    ctx.rankSize        = kernelArg->rankSize;
    ctx.channels       = kernelArg->channels;
    ctx.dataType        = kernelArg->opParam.DataDes.dataType;
    ctx.outputDataType  = kernelArg->opParam.DataDes.outputType;
    if (ctx.outputDataType == HcclDataType::HCCL_DATA_TYPE_RESERVED) {
        ctx.outputDataType = ctx.dataType;
        HCCL_DEBUG("[CcuKernelReduceScatterMesh1D] outputDataType is [INVALID], set outputDataType to[%d]",
            ctx.dataType);
    }
    ctx.reduceOp = kernelArg->opParam.reduceType;
    return HCCL_SUCCESS;
}

static HcclResult InitResource(ReduceScatterContext &ctx)
{
    uint32_t channelIdx = 0;

    if (ctx.channels.size() == 0) {
        HCCL_ERROR("[CcuKernelReduceScatterMesh1D] channels is empty!");
        return HcclResult::HCCL_E_INTERNAL;
    }

    CCU_CHK_RET(ccu::Create(&ctx.output));
    // 按照rank号从小到大遍历channels，遇到本rank就填充本地资源，否则依次取远端资源，要求算法返回的Link同样是按顺序排列的
    ctx.input.resize(ctx.rankSize);
    ctx.token.resize(ctx.rankSize);
    for (uint64_t peerId = 0; peerId < ctx.rankSize; peerId++) {
        if (peerId == ctx.rankId) {
            CCU_CHK_RET(ccu::Create(&ctx.input[peerId]));
            CCU_CHK_RET(ccu::Create(&ctx.token[peerId]));
        } else {
            CCU_CHK_RET(CcuVariableCreateFromChannel(
                ctx.channels[channelIdx], INPUT_XN_ID, &ctx.input[peerId]));
            CCU_CHK_RET(CcuVariableCreateFromChannel(
                ctx.channels[channelIdx], TOKEN_XN_ID, &ctx.token[peerId]));
            channelIdx++;
        }
    }

    CCU_CHK_RET(ccu::Create(&ctx.offset));

    CCU_CHK_RET(ccu::Create(&ctx.goSize.addrOffset));
    CCU_CHK_RET(ccu::Create(&ctx.goSize.loopParam));
    CCU_CHK_RET(ccu::Create(&ctx.goSize.parallelParam));
    CCU_CHK_RET(ccu::Create(&ctx.goSize.residual));

    ctx.resourceAllocated = false;
    ctx.loopRegistered    = false;

    return HCCL_SUCCESS;
}

static HcclResult LoadArgs(ReduceScatterContext &ctx)
{
    CCU_CHK_RET(CcuLoadArg(ctx.input[ctx.rankId]));
    CCU_CHK_RET(CcuLoadArg(ctx.output));
    CCU_CHK_RET(CcuLoadArg(ctx.token[ctx.rankId]));
    CCU_CHK_RET(CcuLoadArg(ctx.offset));
    CCU_CHK_RET(CcuLoadArg(ctx.goSize.addrOffset));
    CCU_CHK_RET(CcuLoadArg(ctx.goSize.loopParam));
    CCU_CHK_RET(CcuLoadArg(ctx.goSize.parallelParam));
    CCU_CHK_RET(CcuLoadArg(ctx.goSize.residual));

    return HCCL_SUCCESS;
}

static void PreSync(ReduceScatterContext &ctx)
{
    for (auto ch : ctx.channels) {
        CcuWriteVariableWithNotify(ch, ctx.input[ctx.rankId],
            INPUT_XN_ID, CKE_IDX_0, 1 << INPUT_XN_ID);
        CcuWriteVariableWithNotify(ch, ctx.token[ctx.rankId],
            TOKEN_XN_ID, CKE_IDX_0, 1 << TOKEN_XN_ID);
    }

    uint32_t allBit = (1 << INPUT_XN_ID) | (1 << TOKEN_XN_ID);
    for (auto ch : ctx.channels) {
        CcuNotifyWait(ch, CKE_IDX_0, allBit);
    }
}

static void PostSync(ReduceScatterContext &ctx)
{
    for (auto ch : ctx.channels) {
        CcuWriteNotify(ch, CKE_IDX_0, 1 << POST_SYNC_ID);
    }
    for (auto ch : ctx.channels) {
        CcuNotifyWait(ch, CKE_IDX_0, 1 << POST_SYNC_ID);
    }
}

static HcclResult DoReduceScatter(ReduceScatterContext &ctx)
{
    const auto *arg = ctx.arg;
    std::vector<CcuRemoteAddr> src;
    src.resize(ctx.rankSize);
    for (uint32_t rankIdx = 0; rankIdx < ctx.rankSize; rankIdx++) {
        CCU_CHK_RET(ccu::Create(&src[rankIdx]));
    }
    CcuLocalAddr dst;
    CCU_CHK_RET(ccu::Create(&dst));
    dst.addr  = ctx.output;
    dst.token = ctx.token[ctx.rankSize];
    uint32_t dstId = 0;
    uint32_t curId = 0;
    for (uint32_t rankIdx = 0; rankIdx < ctx.rankSize; rankIdx++) {
        if (rankIdx != ctx.rankId) {
            curId = dstId;
            dstId++;
        } else {
            // 本端input放在数组末尾
            curId = ctx.rankSize - 1;
        }
        // 其中本端input为LocalAddr，但适配接口，写入RemoteAddr中
        src[curId].addr = ctx.input[rankIdx];
        src[curId].addr += ctx.offset;
        src[curId].token = ctx.token[rankIdx];
    }

    GroupReduce(ctx, ctx.channels, dst, src, ctx.goSize, ctx.dataType, ctx.outputDataType, ctx.reduceOp);

    return HCCL_SUCCESS;
}

// ============================================================================
// 主入口 Kernel 函数
// ============================================================================
HcclResult CcuReduceScatterMesh1DKernel(CcuKernelArg arg)
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

    HCCL_INFO("[CcuKernelReduceScatterMesh1D] ReduceScatterMesh1D run");
    CHK_RET(ParseKernelArg(ctx, kernelArg));
    CHK_RET(InitResource(ctx));
    CHK_RET(LoadArgs(ctx));

    PreSync(ctx);

    CHK_RET(DoReduceScatter(ctx));

    PostSync(ctx);
    HCCL_INFO("[CcuKernelReduceScatterMesh1D] ReduceScatterMesh1D end");

    return HCCL_SUCCESS;
}
} // namespace ops_hccl
