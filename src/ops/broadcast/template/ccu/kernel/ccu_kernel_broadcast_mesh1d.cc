/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "ccu_kernel_broadcast_mesh1d.h"
#include "ccu_kernel_alg_base.h"

namespace ops_hccl {

constexpr int INPUT_XN_ID  = 0;
constexpr int OUTPUT_XN_ID = 1;
constexpr int TOKEN_XN_ID  = 2;
constexpr int POST_SYNC_ID = 3;
constexpr int CKE_IDX_0 = 0;

static CcuResult ParseKernelArg(BroadcastMesh1DContext &ctx, CcuKernelArgBroadcastMesh1D *kernelArg)
{
    ctx.arg = kernelArg;
    ctx.rankSize = kernelArg->rankSize;
    ctx.rankId = kernelArg->rankId;
    ctx.rootId = kernelArg->rootId;
    ctx.dataType = kernelArg->opParam.DataDes.dataType;
    ctx.outputDataType = kernelArg->opParam.DataDes.outputType;
    if (ctx.outputDataType == HcclDataType::HCCL_DATA_TYPE_RESERVED) {
        ctx.outputDataType = ctx.dataType;
    }
    HCCL_INFO("[CcuBroadcastMesh1D] Init, KernelArgs are rootId[%u] rankId[%u], rankSize[%u], dataType[%d], "
        "outputDataType[%d]",
        ctx.rootId, ctx.rankId, ctx.rankSize, ctx.dataType, ctx.outputDataType);
    return CCU_SUCCESS;
}

static CcuResult InitResource(BroadcastMesh1DContext &ctx)
{
    uint16_t channelIdx = 0;
    if (ctx.arg->channelCount == 0) {
        HCCL_ERROR("[CcuBroadcastMesh1D] channels is empty!");
        return CCU_E_INTERNAL;
    }
    ctx.output.resize(ctx.rankSize);
    ctx.token.resize(ctx.rankSize);
    for (uint64_t peerId = 0; peerId < ctx.rankSize; peerId++) {
        if (peerId != ctx.rankId) {
            ctx.output[peerId] = ccu::GetResByChannel<ccu::Variable>(ctx.arg->channels[channelIdx], OUTPUT_XN_ID);
            ctx.token[peerId] = ccu::GetResByChannel<ccu::Variable>(ctx.arg->channels[channelIdx], TOKEN_XN_ID);
            channelIdx++;
        }
    }

    ctx.resourceAllocated = false;
    return CCU_SUCCESS;
}

static CcuResult LoadArgs(BroadcastMesh1DContext &ctx)
{
    uint32_t argId = 0;
    CCU_CHK_RET(ccu::LoadArg(ctx.input, argId++));
    CCU_CHK_RET(ccu::LoadArg(ctx.output[ctx.rankId], argId++));
    CCU_CHK_RET(ccu::LoadArg(ctx.token[ctx.rankId], argId++));
    CCU_CHK_RET(ccu::LoadArg(ctx.offSet, argId++));
    CCU_CHK_RET(ccu::LoadArg(ctx.goSize.addrOffset, argId++));
    CCU_CHK_RET(ccu::LoadArg(ctx.goSize.loopParam, argId++));
    CCU_CHK_RET(ccu::LoadArg(ctx.goSize.parallelParam, argId++));
    CCU_CHK_RET(ccu::LoadArg(ctx.goSize.residual, argId++));
    return CCU_SUCCESS;
}

static CcuResult PreSync(BroadcastMesh1DContext &ctx)
{
    for (uint32_t i = 0; i < ctx.arg->channelCount; i++) {
        ccu::WriteVariableWithNotify(ctx.arg->channels[i], ctx.output[ctx.rankId], OUTPUT_XN_ID, CKE_IDX_0, 1 << OUTPUT_XN_ID);
        ccu::WriteVariableWithNotify(ctx.arg->channels[i], ctx.token[ctx.rankId], TOKEN_XN_ID, CKE_IDX_0, 1 << TOKEN_XN_ID);
    }
    uint32_t allBit = 1 << OUTPUT_XN_ID | 1 << TOKEN_XN_ID;
    for (uint32_t i = 0; i < ctx.arg->channelCount; i++) {
        ccu::NotifyWait(ctx.arg->channels[i], CKE_IDX_0, allBit);
    }
    HCCL_INFO("[CcuBroadcastMesh1D] BroadcastMesh1D wait all end");
    return CCU_SUCCESS;
}

static CcuResult PostSync(BroadcastMesh1DContext &ctx)
{
    for (uint32_t i = 0; i < ctx.arg->channelCount; i++) {
        ccu::NotifyRecord(ctx.arg->channels[i], CKE_IDX_0, 1 << POST_SYNC_ID);
    }
    for (uint32_t i = 0; i < ctx.arg->channelCount; i++) {
        ccu::NotifyWait(ctx.arg->channels[i], CKE_IDX_0, 1 << POST_SYNC_ID);
    }
    HCCL_INFO("[CcuBroadcastMesh1D] BroadcastMesh1D groupwait end");
    return CCU_SUCCESS;
}

static CcuResult BroadcastFromRootToAll(BroadcastMesh1DContext &ctx)
{
    std::vector<ccu::RemoteAddr> dst;
    dst.resize(ctx.rankSize - 1);

    ccu::LocalAddr src;
    src.addr = ctx.input;
    src.addr += ctx.offSet;
    src.token = ctx.token[ctx.rankId];

    uint32_t curId = 0;
    for (uint32_t rankIdx = 0; rankIdx < ctx.rankSize; rankIdx++) {
        if (rankIdx != ctx.rootId) {
            dst[curId].addr = ctx.output[rankIdx];
            dst[curId].addr += ctx.offSet;
            dst[curId].token = ctx.token[rankIdx];
            curId++;
        }
    }

    ccu::LocalAddr localDst;
    localDst.addr = ctx.output[ctx.rankId];
    localDst.addr += ctx.offSet;
    localDst.token = ctx.token[ctx.rankId];

    GroupBroadcast(ctx, ctx.arg->channels, ctx.arg->channelCount, localDst, dst, src, ctx.goSize);
    HCCL_INFO("[CcuBroadcastMesh1D] BroadcastMesh1D GroupBroadcast end");
    return CCU_SUCCESS;
}

CcuResult CcuBroadcastMesh1DKernel(CcuKernelArg arg)
{
    auto *kernelArg = static_cast<CcuKernelArgBroadcastMesh1D *>(arg);
    BroadcastMesh1DContext ctx;
    ctx.arg = kernelArg;
    ctx.resourceAllocated = false;
    ctx.moConfig.msInterleave = 0;
    ctx.moConfig.loopCount = 0;
    ctx.moConfig.memSlice = 0;
    ctx.moRes.eventCount = 0;
    ctx.moRes.bufCount = 0;
    ctx.enginePool = 0;

    HCCL_INFO("[CcuBroadcastMesh1D] BroadcastMesh1D run");
    CCU_CHK_RET(ParseKernelArg(ctx, kernelArg));
    CCU_CHK_RET(InitResource(ctx));
    CCU_CHK_RET(LoadArgs(ctx));

    CCU_CHK_RET(PreSync(ctx));

    if (ctx.rankId == ctx.rootId) {
        CCU_CHK_RET(BroadcastFromRootToAll(ctx));
    }
    CCU_CHK_RET(PostSync(ctx));

    HCCL_INFO("[CcuBroadcastMesh1D] BroadcastMesh1D end");
    return CCU_SUCCESS;
}

} // namespace ops_hccl