/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "ccu_kernel_all_to_all_mesh2die.h"

namespace ops_hccl {

constexpr uint32_t CKE_IDX_0 = 0;
constexpr uint32_t OUTPUT_XN_ID = 1;
constexpr uint32_t TOKEN_XN_ID = 2;
constexpr uint32_t POST_SYNC_ID = 3;

static CcuResult ParseKernelArg(AllToAllMesh2DieContext &ctx, CcuKernelArg arg)
{
    auto *kernelArg = static_cast<CcuKernelArgAllToAllMesh2Die *>(arg);
    ctx.arg = kernelArg;
    ctx.rankSize = kernelArg->rankSize;
    ctx.rankId = kernelArg->rankId;
    ctx.withMyRank = kernelArg->withMyRank;
    ctx.localSize = kernelArg->localSize;
    ctx.localId = kernelArg->localId;
    ctx.rankGroup = kernelArg->rankGroup;
    ctx.selfBit = 1 << ctx.localId;
    ctx.allBit = (1 << ctx.localSize) - 1;
    return CCU_SUCCESS;
}

static CcuResult InitResource(AllToAllMesh2DieContext &ctx)
{
    const auto *arg = ctx.arg;

    ctx.output.resize(arg->localSize);
    ctx.token.resize(arg->localSize);
    ctx.inputOffsets.resize(arg->localSize);

    uint32_t channelIdx = 0;
    for (uint32_t i = 0; i < arg->localSize; i++) {
        if (i != arg->localId) {
            ctx.output[i] = ccu::GetResByChannel<ccu::Variable>(arg->channels[channelIdx], OUTPUT_XN_ID);
            ctx.token[i] = ccu::GetResByChannel<ccu::Variable>(arg->channels[channelIdx], TOKEN_XN_ID);
            channelIdx++;
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

static CcuResult LoadArgs(AllToAllMesh2DieContext &ctx)
{
    const auto *arg = ctx.arg;
    uint16_t argIdx = 0;

    CCU_CHK_RET(ccu::LoadArg(ctx.input, argIdx++));
    CCU_CHK_RET(ccu::LoadArg(ctx.output[arg->localId], argIdx++));
    CCU_CHK_RET(ccu::LoadArg(ctx.token[arg->localId], argIdx++));
    CCU_CHK_RET(ccu::LoadArg(ctx.sliceSize, argIdx++));
    CCU_CHK_RET(ccu::LoadArg(ctx.inputSliceStride, argIdx++));
    CCU_CHK_RET(ccu::LoadArg(ctx.outputoffset, argIdx++));
    CCU_CHK_RET(ccu::LoadArg(ctx.groupOpSize.addrOffset, argIdx++));
    CCU_CHK_RET(ccu::LoadArg(ctx.groupOpSize.loopParam, argIdx++));
    CCU_CHK_RET(ccu::LoadArg(ctx.groupOpSize.parallelParam, argIdx++));
    CCU_CHK_RET(ccu::LoadArg(ctx.groupOpSize.residual, argIdx++));

    for (uint32_t i = 0; i < arg->localSize; i++) {
        CCU_CHK_RET(ccu::LoadArg(ctx.inputOffsets[i], argIdx++));
    }

    return CCU_SUCCESS;
}

static CcuResult ExchangeInfoSync(AllToAllMesh2DieContext &ctx)
{
    const auto *arg = ctx.arg;

    for (uint32_t chIdx = 0; chIdx < arg->channelCount; chIdx++) {
        ccu::WriteVariableWithNotify(arg->channels[chIdx], ctx.output[arg->localId], OUTPUT_XN_ID,
            CKE_IDX_0, 1 << OUTPUT_XN_ID);
        ccu::WriteVariableWithNotify(arg->channels[chIdx], ctx.token[arg->localId], TOKEN_XN_ID,
            CKE_IDX_0, 1 << TOKEN_XN_ID);
    }

    uint32_t waitBits = (1 << OUTPUT_XN_ID) | (1 << TOKEN_XN_ID);
    for (uint32_t chIdx = 0; chIdx < arg->channelCount; chIdx++) {
        ccu::NotifyWait(arg->channels[chIdx], CKE_IDX_0, waitBits);
    }

    return CCU_SUCCESS;
}

static CcuResult DoAllToAll(AllToAllMesh2DieContext &ctx)
{
    const auto *arg = ctx.arg;

    uint32_t channelsIdx = 0;
    for (uint32_t r = 0; r < arg->localSize; r++) {
        if (r == arg->localId) {
            if (arg->withMyRank) {
                ccu::LocalAddr localSrc(ctx.input, ctx.inputOffsets[r]);
                ccu::LocalAddr localDst(ctx.output[r], ctx.outputoffset);
                CCU_CHK_RET(GroupCopy(ctx, localDst, localSrc, ctx.groupOpSize));
                ccu::EventRecord(ctx.event, ctx.selfBit);
            }
            continue;
        }
        ccu::LocalAddr src(ctx.input, ctx.inputOffsets[r]);
        ccu::RemoteAddr dst(ctx.output[r], ctx.outputoffset);
        uint16_t mask = 1 << r;
        ccu::Write(arg->channels[channelsIdx], dst, src, ctx.sliceSize, ctx.event, mask);
        ccu::EventRecord(ctx.event, mask);
        channelsIdx++;
    }

    uint16_t waitMask = arg->withMyRank ? ctx.allBit : (ctx.allBit & ~ctx.selfBit);
    ccu::EventWait(ctx.event, waitMask);

    return CCU_SUCCESS;
}

static CcuResult PostSync(AllToAllMesh2DieContext &ctx)
{
    const auto *arg = ctx.arg;

    for (uint32_t chIdx = 0; chIdx < arg->channelCount; chIdx++) {
        ccu::NotifyRecord(arg->channels[chIdx], CKE_IDX_0, 1 << POST_SYNC_ID);
    }
    for (uint32_t chIdx = 0; chIdx < arg->channelCount; chIdx++) {
        ccu::NotifyWait(arg->channels[chIdx], CKE_IDX_0, 1 << POST_SYNC_ID);
    }

    return CCU_SUCCESS;
}

CcuResult CcuAllToAllMesh2DieKernel(CcuKernelArg arg)
{
    AllToAllMesh2DieContext ctx;

    CCU_CHK_RET(ParseKernelArg(ctx, arg));
    CCU_CHK_RET(InitResource(ctx));
    CCU_CHK_RET(LoadArgs(ctx));
    CCU_CHK_RET(ExchangeInfoSync(ctx));
    CCU_CHK_RET(DoAllToAll(ctx));
    CCU_CHK_RET(PostSync(ctx));

    return CCU_SUCCESS;
}

}
