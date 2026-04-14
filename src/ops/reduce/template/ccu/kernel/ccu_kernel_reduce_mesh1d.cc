/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "ccu_kernel_reduce_mesh1d.h"
#include "ccu_control_api.h"

namespace ops_hccl {

constexpr int INPUT_XN_ID  = 0;
constexpr int OUTPUT_XN_ID = 1;
constexpr int TOKEN_XN_ID  = 2;
constexpr int POST_SYNC_ID  = 3;
constexpr int CKE_IDX_0    = 0;

static CcuResult ParseKernelArg(ReduceMesh1DContext &ctx, CcuKernelArgReduceMesh1D *kernelArg)

{
    // ctx.rankId          = kernelArg->rankId;
    // ctx.rankSize        = kernelArg->rankSize;
    // ctx.channels       = kernelArg->channels;
    ctx.dataType        = kernelArg->opParam.DataDes.dataType;
    ctx.outputDataType  = kernelArg->opParam.DataDes.outputType;
    if (ctx.outputDataType == HcclDataType::HCCL_DATA_TYPE_RESERVED) {
        ctx.outputDataType = ctx.dataType;

        HCCL_DEBUG("[CcuKernelReduceMesh1D] outputDataType is [INVALID], set outputDataType to[%d]",
            ctx.dataType);
    }
    ctx.reduceOp = kernelArg->opParam.reduceType;
    return CCU_SUCCESS;
}

static CcuResult InitResource(ReduceMesh1DContext &ctx)
{
    const auto *arg = ctx.arg;
    uint32_t channelIdx = 0;

    if (arg->channelCount == 0) {
        HCCL_ERROR("[CcuKernelReduceMesh1D] channels is empty!");
        return CcuResult::CCU_E_INTERNAL;
    }
    HCCL_INFO("[CcuKernelReduceMesh1D] channels.size: [%u]", arg->channelCount);

    // 按照rank号从小到大遍历channels，遇到本rank就填充本地资源，否则依次取远端资源，要求算法返回的Link同样是按顺序排列的
    ctx.input.resize(arg->rankSize);
    ctx.output.resize(arg->rankSize);
    ctx.token.resize(arg->rankSize);
    for (uint64_t peerId = 0; peerId < arg->rankSize; peerId++) {
        if (peerId == arg->rankId) {
            CCU_CHK_RET(ccu::Alloc(&ctx.input[peerId]));
            CCU_CHK_RET(ccu::Alloc(&ctx.output[peerId]));
            CCU_CHK_RET(ccu::Alloc(&ctx.token[peerId]));
        } else {
            CCU_CHK_RET(ccu::CreateByChannel(arg->channels[channelIdx], INPUT_XN_ID, &ctx.input[peerId]));
            CCU_CHK_RET(ccu::CreateByChannel(arg->channels[channelIdx], OUTPUT_XN_ID, &ctx.output[peerId]));
            CCU_CHK_RET(ccu::CreateByChannel(arg->channels[channelIdx], TOKEN_XN_ID, &ctx.token[peerId]));
            channelIdx++;
        }
    }

    CCU_CHK_RET(ccu::Alloc(&ctx.groupOpSize.addrOffset));
    CCU_CHK_RET(ccu::Alloc(&ctx.groupOpSize.loopParam));
    CCU_CHK_RET(ccu::Alloc(&ctx.groupOpSize.parallelParam));
    CCU_CHK_RET(ccu::Alloc(&ctx.groupOpSize.residual));

    CCU_CHK_RET(ccu::Alloc(&ctx.currentRankSliceInputOffset));
    CCU_CHK_RET(ccu::Alloc(&ctx.currentRankSliceOutputOffset));
    CCU_CHK_RET(ccu::Alloc(&ctx.repeatNum));
    CCU_CHK_RET(ccu::Alloc(&ctx.inputRepeatStride));
    CCU_CHK_RET(ccu::Alloc(&ctx.outputRepeatStride));
    CCU_CHK_RET(ccu::Alloc(&ctx.normalSliceSize));
    CCU_CHK_RET(ccu::Alloc(&ctx.lastSliceSize));
    CCU_CHK_RET(ccu::Alloc(&ctx.repeatNumVar));
    CCU_CHK_RET(ccu::Alloc(&ctx.flag));
    CCU_CHK_RET(ccu::Alloc(&ctx.event));

    CCU_CHK_RET(ccu::CreateLoopExecutor(&ctx.enginePool, RS_MAX_RANK_SIZE + 1));

    ctx.resourceAllocated = false;
    ctx.loopRegistered    = false;

    return CCU_SUCCESS;
}

static CcuResult LoadArgs(ReduceMesh1DContext &ctx)
{
    const auto *arg = ctx.arg;
    CCU_CHK_RET(ccu::LoadArg(ctx.input[arg->rankId]));
    CCU_CHK_RET(ccu::LoadArg(ctx.output[arg->rankId]));
    CCU_CHK_RET(ccu::LoadArg(ctx.token[arg->rankId]));
    CCU_CHK_RET(ccu::LoadArg(ctx.currentRankSliceInputOffset));
    CCU_CHK_RET(ccu::LoadArg(ctx.currentRankSliceOutputOffset));
    CCU_CHK_RET(ccu::LoadArg(ctx.repeatNum));
    CCU_CHK_RET(ccu::LoadArg(ctx.inputRepeatStride));
    CCU_CHK_RET(ccu::LoadArg(ctx.outputRepeatStride));
    CCU_CHK_RET(ccu::LoadArg(ctx.normalSliceSize));
    CCU_CHK_RET(ccu::LoadArg(ctx.lastSliceSize));
    CCU_CHK_RET(ccu::LoadArg(ctx.repeatNumVar));
    CCU_CHK_RET(ccu::LoadArg(ctx.groupOpSize.addrOffset));
    CCU_CHK_RET(ccu::LoadArg(ctx.groupOpSize.loopParam));
    CCU_CHK_RET(ccu::LoadArg(ctx.groupOpSize.parallelParam));
    CCU_CHK_RET(ccu::LoadArg(ctx.groupOpSize.residual));
    return CCU_SUCCESS;
}

static void PreSync(ReduceMesh1DContext &ctx)
{
    HCCL_INFO("[CcuKernelReduceMesh1D] ReduceMesh1D PreSync begin");
    const auto *arg = ctx.arg;
    for (uint32_t i = 0; i < arg->channelCount; i++) {
        ccu::WriteVariableWithNotify(arg->channels[i], ctx.input[arg->rankId], INPUT_XN_ID, CKE_IDX_0, 1 << INPUT_XN_ID);
        ccu::WriteVariableWithNotify(arg->channels[i], ctx.output[arg->rankId], OUTPUT_XN_ID, CKE_IDX_0, 1 << OUTPUT_XN_ID);
        ccu::WriteVariableWithNotify(arg->channels[i], ctx.token[arg->rankId], TOKEN_XN_ID, CKE_IDX_0, 1 << TOKEN_XN_ID);
    }

    uint32_t allBit = (1 << INPUT_XN_ID) | (1 << OUTPUT_XN_ID) | (1 << TOKEN_XN_ID);
    for (uint32_t i = 0; i < arg->channelCount; i++) {
        ccu::NotifyWait(arg->channels[i], CKE_IDX_0, allBit);
    }
    HCCL_INFO("[CcuKernelReduceMesh1D] ReduceMesh1D PreSync end");
}

static void PostSync(ReduceMesh1DContext &ctx)
{
    HCCL_INFO("[CcuKernelReduceMesh1D] ReduceMesh1D post sync start");
    const auto *arg = ctx.arg;
    for (uint32_t i = 0; i < arg->channelCount; i++) {
        ccu::NotifyRecord(arg->channels[i], CKE_IDX_0, 1 << POST_SYNC_ID);
    }
    for (uint32_t i = 0; i < arg->channelCount; i++) {
        ccu::NotifyWait(arg->channels[i], CKE_IDX_0, 1 << POST_SYNC_ID);
    }
    HCCL_INFO("[CcuKernelReduceMesh1D] ReduceMesh1D post sync end");
}

static CcuResult DoRepeatReduce(ReduceMesh1DContext &ctx)
{
    const auto *arg = ctx.arg;

    std::vector<CcuRemoteAddr> remoteSrc; // GSA[0]
    remoteSrc.resize(arg->rankSize);
    for (uint64_t rankIdx = 0; rankIdx < arg->rankSize - 1; rankIdx++) {
        CCU_CHK_RET(ccu::Alloc(&remoteSrc[rankIdx]));
    }

    CcuLocalAddr localSrc;  // GSA[1]
    CCU_CHK_RET(ccu::Alloc(&localSrc));

    CcuLocalAddr dst; // GSA[2]
    CCU_CHK_RET(ccu::Alloc(&dst));


    dst.addr = ctx.output[arg->rankId];   // GSA[400] + Xn[1] to GSA[2]
    dst.token = ctx.token[arg->rankId]; // Xn[2] + Xn[413] to Xn[19]
    uint32_t curId = 0;
    for (uint32_t rankIdx = 0; rankIdx < arg->rankSize; rankIdx++) {
        if (rankIdx != arg->rootId) {
            remoteSrc[curId].addr  = ctx.input[rankIdx]; // GSA[400] + Xn[400] to GSA[0]
            remoteSrc[curId].token = ctx.token[rankIdx];
            curId++;
        } else {
            continue;
        }
    }
    localSrc.addr = ctx.input[arg->rankId];
    localSrc.token = ctx.token[arg->rankId];

    CCU_IF_ONLY (ctx.flag != 0) {
        // 非第一轮执行时，remoteSrc 和 dst 已经初始化，需要添加偏移量
        dst.addr += ctx.outputRepeatStride;
        for (auto &s : remoteSrc) {
            s.addr += ctx.inputRepeatStride;
        }
    }
    GroupReduce(ctx, arg->channels, arg->channelCount, dst, remoteSrc, localSrc,
                ctx.groupOpSize, ctx.dataType, ctx.outputDataType, ctx.reduceOp);

    return CCU_SUCCESS;
}

// ============================================================================
// 主入口 Kernel 函数
// ============================================================================
CcuResult CcuReduceMesh1DKernel(CcuKernelArg arg)
{
    auto *kernelArg = static_cast<CcuKernelArgReduceMesh1D *>(arg);

    ReduceMesh1DContext ctx;
    ctx.arg = kernelArg;
    ctx.resourceAllocated = false;
    ctx.loopRegistered = false;
    ctx.moConfig.msInterleave = 0;
    ctx.moConfig.loopCount = 0;
    ctx.moConfig.memSlice = 0;
    ctx.moRes.eventCount = 0;
    ctx.moRes.bufCount = 0;
    ctx.enginePool = 0;

    HCCL_INFO("[CcuKernelReduceMesh1D] ReduceMesh1D run");
    CCU_CHK_RET(ParseKernelArg(ctx, kernelArg));
    CCU_CHK_RET(InitResource(ctx));
    CCU_CHK_RET(LoadArgs(ctx));

    PreSync(ctx);

    if (kernelArg->rankId == kernelArg->rootId) {
        CcuVariable repeatNumAdd;
        CCU_CHK_RET(ccu::Alloc(&repeatNumAdd));
        repeatNumAdd  = 1;
        ctx.flag = 0;
        CCU_DO_WHILE(ctx.repeatNumVar != UINT64_MAX) { // 循环repeatNum_次
            CCU_CHK_RET(DoRepeatReduce(ctx));
            ctx.repeatNumVar += repeatNumAdd;
            ctx.flag = 1;
        }
    }

    PostSync(ctx);
    HCCL_INFO("[CcuKernelReduceMesh1D] ReduceMesh1D end");

    return CCU_SUCCESS;
}

} // namespace ops_hccl