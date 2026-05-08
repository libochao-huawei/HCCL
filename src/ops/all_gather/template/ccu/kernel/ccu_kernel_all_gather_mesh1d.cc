/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "ccu_kernel_all_gather_mesh1d.h"

namespace ops_hccl {

constexpr int OUTPUT_XN_ID = 1;
constexpr int TOKEN_XN_ID = 2;
constexpr int CKE_IDX_0 = 0;
constexpr int POST_SYNC_ID = 3;

static CcuResult ParseKernelArg(AllGatherMesh1DContext &ctx, CcuKernelArgAllGatherMesh1D *kernelArg)
{
    ctx.arg = kernelArg;
    return CCU_SUCCESS;
}

static CcuResult InitResource(AllGatherMesh1DContext &ctx)
{
    const auto *arg = ctx.arg;
    uint32_t channelIdx = 0;

    if (arg->channelCount == 0) {
        HCCL_ERROR("[CcuKernelAllGatherMesh1D] channels is empty!");
        return CcuResult::CCU_E_INTERNAL;
    }

    CCU_CHK_RET(ccu::Alloc(&ctx.input));
    ctx.output.resize(arg->rankSize);
    ctx.token.resize(arg->rankSize);

    for (uint64_t peerId = 0; peerId < arg->rankSize; peerId++) {
        if (peerId == arg->rankId) {
            CCU_CHK_RET(ccu::Alloc(&ctx.output[peerId]));
            CCU_CHK_RET(ccu::Alloc(&ctx.token[peerId]));
        } else {
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

    CCU_CHK_RET(ccu::CreateLoopExecutor(&ctx.enginePool, CCU_MS_DEFAULT_LOOP_COUNT)); // 待修改

    ctx.resourceAllocated = false;
    ctx.loopRegistered = false;

    return CCU_SUCCESS;
}

static CcuResult LoadArgs(AllGatherMesh1DContext &ctx)
{
    const auto *arg = ctx.arg;

    CCU_CHK_RET(ccu::LoadArg(ctx.input));
    CCU_CHK_RET(ccu::LoadArg(ctx.output[arg->rankId]));
    CCU_CHK_RET(ccu::LoadArg(ctx.token[arg->rankId]));
    CCU_CHK_RET(ccu::LoadArg(ctx.offset));
    CCU_CHK_RET(ccu::LoadArg(ctx.goSize.addrOffset));
    CCU_CHK_RET(ccu::LoadArg(ctx.goSize.loopParam));
    CCU_CHK_RET(ccu::LoadArg(ctx.goSize.parallelParam));
    CCU_CHK_RET(ccu::LoadArg(ctx.goSize.residual));

    return CCU_SUCCESS;
}

static CcuResult PreSync(AllGatherMesh1DContext &ctx)
{
    const auto *arg = ctx.arg;

    for (uint32_t i = 0; i < arg->channelCount; i++) {
        CCU_CHK_RET(ccu::WriteVariableWithNotify(arg->channels[i], ctx.output[arg->rankId],
            OUTPUT_XN_ID, CKE_IDX_0, 1 << OUTPUT_XN_ID));
        CCU_CHK_RET(ccu::WriteVariableWithNotify(arg->channels[i], ctx.token[arg->rankId],
            TOKEN_XN_ID, CKE_IDX_0, 1 << TOKEN_XN_ID));
    }

    uint32_t allBit = (1 << OUTPUT_XN_ID) | (1 << TOKEN_XN_ID);
    for (uint32_t i = 0; i < arg->channelCount; i++) {
        CCU_CHK_RET(ccu::NotifyWait(arg->channels[i], CKE_IDX_0, allBit));
    }
    return CCU_SUCCESS;
}

static CcuResult PostSync(AllGatherMesh1DContext &ctx)
{
    const auto *arg = ctx.arg;

    for (uint32_t i = 0; i < arg->channelCount; i++) {
        ccu::NotifyRecord(arg->channels[i], CKE_IDX_0, 1 << POST_SYNC_ID);
    }
    for (uint32_t i = 0; i < arg->channelCount; i++) {
        ccu::NotifyWait(arg->channels[i], CKE_IDX_0, 1 << POST_SYNC_ID);
    }
    return CCU_SUCCESS;
}

static CcuResult DoAllGather(AllGatherMesh1DContext &ctx)
{
    const auto *arg = ctx.arg;
    ccu::LocalAddr src;
    ccu::LocalAddr localdst;
    std::vector<ccu::RemoteAddr> dst;
    dst.resize(arg->rankSize - 1);

    CCU_CHK_RET(ccu::Alloc(&src));
    CCU_CHK_RET(ccu::Alloc(&localdst));
    for (uint32_t rankIdx = 0; rankIdx < arg->rankSize - 1; rankIdx++) {
        CCU_CHK_RET(ccu::Alloc(&dst[rankIdx]));
    }

    src.addr = ctx.input;
    src.token = ctx.token[arg->rankId];

    uint32_t dstId = 0;
    uint32_t curId = 0;
    for (uint64_t rankIdx = 0; rankIdx < arg->rankSize; rankIdx++) {
        if (rankIdx != arg->rankId) {
            curId = dstId;
            dst[curId].addr = ctx.output[rankIdx];
            dst[curId].addr += ctx.offset;
            dst[curId].token = ctx.token[rankIdx];
            dstId++;
        } else {
            localdst.addr = ctx.output[rankIdx];
            localdst.addr += ctx.offset;
            localdst.token = ctx.token[rankIdx];
        }
    }

    CCU_CHK_RET(GroupBroadcast(ctx, arg->channels, arg->channelCount, dst, src, localdst, ctx.goSize));

    return CCU_SUCCESS;
}

CcuResult CcuAllGatherMesh1DKernel(CcuKernelArg arg)
{
    auto *kernelArg = static_cast<CcuKernelArgAllGatherMesh1D *>(arg);

    AllGatherMesh1DContext ctx;
    ctx.resourceAllocated = false;
    ctx.loopRegistered = false;
    ctx.moConfig.msInterleave = 0;
    ctx.moConfig.loopCount = 0;
    ctx.moConfig.memSlice = 0;
    ctx.moRes.eventCount = 0;
    ctx.moRes.bufCount = 0;
    ctx.enginePool = 0;

    HCCL_INFO("[CcuKernelAllGatherMesh1D] AllGatherMesh1D run");
    CCU_CHK_RET(ParseKernelArg(ctx, kernelArg));
    CCU_CHK_RET(InitResource(ctx));
    CCU_CHK_RET(LoadArgs(ctx));

    CCU_CHK_RET(PreSync(ctx));

    CCU_CHK_RET(DoAllGather(ctx));

    CCU_CHK_RET(PostSync(ctx));
    HCCL_INFO("[CcuKernelAllGatherMesh1D] AllGatherMesh1D end");

    return CCU_SUCCESS;
}

} // namespace ops_hccl
