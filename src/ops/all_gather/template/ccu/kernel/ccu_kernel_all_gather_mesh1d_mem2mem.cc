/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "ccu_kernel_all_gather_mesh1d_mem2mem.h"

namespace ops_hccl {

constexpr int OUTPUT_XN_ID = 1;
constexpr int TOKEN_XN_ID = 2;
constexpr int CKE_IDX_0 = 0;
constexpr int POST_SYNC_ID = 3;

static CcuResult ParseKernelArg(AllGatherMesh1DMem2MemContext &ctx, CcuKernelArgAllGatherMesh1DMem2Mem *kernelArg)
{
    ctx.arg = kernelArg;
    return CCU_SUCCESS;
}

static CcuResult InitResource(AllGatherMesh1DMem2MemContext &ctx)
{
    const auto *arg = ctx.arg;
    uint32_t channelIdx = 0;

    if (arg->channelCount == 0) {
        HCCL_ERROR("[CcuKernelAllGatherMesh1DMem2Mem] channels is empty!");
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

    CCU_CHK_RET(ccu::Alloc(&ctx.currentRankSliceInputOffset));
    CCU_CHK_RET(ccu::Alloc(&ctx.currentRankSliceOutputOffset));
    CCU_CHK_RET(ccu::Alloc(&ctx.inputRepeatStride));
    CCU_CHK_RET(ccu::Alloc(&ctx.outputRepeatStride));
    CCU_CHK_RET(ccu::Alloc(&ctx.tmpRepeatNum));
    CCU_CHK_RET(ccu::Alloc(&ctx.normalSliceSize));
    CCU_CHK_RET(ccu::Alloc(&ctx.lastSliceSize));
    CCU_CHK_RET(ccu::Alloc(&ctx.isInputOutputEqual));

    CCU_CHK_RET(ccu::Alloc(&ctx.goSize.addrOffset));
    CCU_CHK_RET(ccu::Alloc(&ctx.goSize.loopParam));
    CCU_CHK_RET(ccu::Alloc(&ctx.goSize.parallelParam));
    CCU_CHK_RET(ccu::Alloc(&ctx.goSize.residual));

    CCU_CHK_RET(ccu::CreateLoopExecutor(&ctx.enginePool, CCU_MS_LOCAL_COPY_LOOP_COUNT)); // 待修改

    ctx.resourceAllocated = false;
    ctx.loopRegistered = false;

    CCU_CHK_RET(ccu::Alloc(&ctx.event));
    CCU_CHK_RET(ccu::Alloc(&ctx.src_loccopy));
    CCU_CHK_RET(ccu::Alloc(&ctx.localDst));

    return CCU_SUCCESS;
}

static CcuResult LoadArgs(AllGatherMesh1DMem2MemContext &ctx)
{
    const auto *arg = ctx.arg;

    CCU_CHK_RET(ccu::LoadArg(ctx.input));
    CCU_CHK_RET(ccu::LoadArg(ctx.output[arg->rankId]));
    CCU_CHK_RET(ccu::LoadArg(ctx.token[arg->rankId]));
    CCU_CHK_RET(ccu::LoadArg(ctx.currentRankSliceInputOffset));
    CCU_CHK_RET(ccu::LoadArg(ctx.currentRankSliceOutputOffset));
    CCU_CHK_RET(ccu::LoadArg(ctx.tmpRepeatNum));
    CCU_CHK_RET(ccu::LoadArg(ctx.inputRepeatStride));
    CCU_CHK_RET(ccu::LoadArg(ctx.outputRepeatStride));
    CCU_CHK_RET(ccu::LoadArg(ctx.normalSliceSize));
    CCU_CHK_RET(ccu::LoadArg(ctx.lastSliceSize));
    CCU_CHK_RET(ccu::LoadArg(ctx.isInputOutputEqual));
    CCU_CHK_RET(ccu::LoadArg(ctx.goSize.addrOffset));
    CCU_CHK_RET(ccu::LoadArg(ctx.goSize.loopParam));
    CCU_CHK_RET(ccu::LoadArg(ctx.goSize.parallelParam));
    CCU_CHK_RET(ccu::LoadArg(ctx.goSize.residual));

    return CCU_SUCCESS;
}

static CcuResult PreSync(AllGatherMesh1DMem2MemContext &ctx)
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

static CcuResult PostSync(AllGatherMesh1DMem2MemContext &ctx)
{
    const auto *arg = ctx.arg;

    for (uint32_t i = 0; i < arg->channelCount; i++) {
        CCU_CHK_RET(ccu::NotifyRecord(arg->channels[i], CKE_IDX_0, 1 << POST_SYNC_ID));
    }
    for (uint32_t i = 0; i < arg->channelCount; i++) {
        CCU_CHK_RET(ccu::NotifyWait(arg->channels[i], CKE_IDX_0, 1 << POST_SYNC_ID));
    }
    return CCU_SUCCESS;
}

static CcuResult DoAllGather(AllGatherMesh1DMem2MemContext &ctx, const ccu::LocalAddr &src, const std::vector<ccu::RemoteAddr> &dst, const CcuVariable &sliceSize)
{
    const auto *arg = ctx.arg;
    uint32_t channelId = 0;

    for (uint64_t rankIdx = 0; rankIdx < arg->rankSize; rankIdx++) {
        if (rankIdx == arg->rankId) {
            CCU_CHK_RET(ccu::SetMask(ctx.event, 1 << rankIdx));
            CCU_CHK_RET(ccu::RecordEvent(ctx.event));
        } else {
            // 处理非本卡情况
            CCU_CHK_RET(ccu::SetMask(ctx.event, 1 << rankIdx));
            CCU_CHK_RET(ccu::WriteNb(arg->channels[channelId], dst[rankIdx], src, sliceSize, ctx.event));
            channelId++;
        }
    }
    CCU_IF_ONLY(ctx.isInputOutputEqual == 0)
    {
        // 处理本卡情况
        CCU_CHK_RET(GroupCopy(ctx, ctx.localDst, ctx.src_loccopy, ctx.goSize));
    }
    CCU_CHK_RET(ccu::SetMask(ctx.event, (1 << arg->rankSize) - 1));
    CCU_CHK_RET(ccu::WaitEvent(ctx.event));
    return CCU_SUCCESS;
}

static CcuResult DoRepeatAllGather(AllGatherMesh1DMem2MemContext &ctx)
{
    const auto *arg = ctx.arg;

    ccu::LocalAddr src;
    std::vector<ccu::RemoteAddr> dst;

    dst.resize(arg->rankSize);
    CCU_CHK_RET(ccu::Alloc(&src));
    for (uint32_t rankIdx = 0; rankIdx < arg->rankSize; rankIdx++) {
        if (rankIdx != arg->rankId) {
            CCU_CHK_RET(ccu::Alloc(&dst[rankIdx]));
        }
    }

    src.addr = ctx.input;
    src.addr += ctx.currentRankSliceInputOffset;
    src.token = ctx.token[arg->rankId];

    ctx.src_loccopy.addr = ctx.input;
    ctx.src_loccopy.addr += ctx.currentRankSliceInputOffset;
    ctx.src_loccopy.token = ctx.token[arg->rankId];

    for (uint32_t rankIdx = 0; rankIdx < arg->rankSize; rankIdx++) {
        if (rankIdx == arg->rankId) {
            ctx.localDst.addr = ctx.output[arg->rankId];
            ctx.localDst.addr += ctx.currentRankSliceOutputOffset;
            ctx.localDst.token = ctx.token[arg->rankId];
        } else {
            dst[rankIdx].addr = ctx.output[rankIdx];
            dst[rankIdx].addr += ctx.currentRankSliceOutputOffset;
            dst[rankIdx].token = ctx.token[rankIdx];
        }
    }

    CcuVariable constVar1;
    CcuVariable repeatTimeflag;
    CCU_CHK_RET(ccu::Alloc(&constVar1));
    CCU_CHK_RET(ccu::Alloc(&repeatTimeflag));
    constVar1 = 1;
    repeatTimeflag = 0;

    CCU_DO_WHILE(ctx.tmpRepeatNum != UINT64_MAX)
    {
        ctx.tmpRepeatNum += constVar1;
        CCU_IF_ONLY(repeatTimeflag != 0)
        {
            src.addr += ctx.inputRepeatStride;
            for (uint32_t rankIdx = 0; rankIdx < arg->rankSize; rankIdx++) {
                if (rankIdx == arg->rankId) {
                    ctx.localDst.addr += ctx.outputRepeatStride;
                } else {
                    dst[rankIdx].addr += ctx.outputRepeatStride;
                }
            }
        }
        CCU_IF_ONLY(ctx.normalSliceSize != 0)
        {
            CCU_CHK_RET(DoAllGather(ctx, src, dst, ctx.normalSliceSize));
        }
        repeatTimeflag = 1;
    }

    return CCU_SUCCESS;
}

CcuResult CcuAllGatherMesh1DMem2MemKernel(CcuKernelArg arg)
{
    auto *kernelArg = static_cast<CcuKernelArgAllGatherMesh1DMem2Mem *>(arg);

    AllGatherMesh1DMem2MemContext ctx;
    ctx.resourceAllocated = false;
    ctx.loopRegistered = false;
    ctx.moConfig.msInterleave = 0;
    ctx.moConfig.loopCount = 0;
    ctx.moConfig.memSlice = 0;
    ctx.moRes.eventCount = 0;
    ctx.moRes.bufCount = 0;
    ctx.enginePool = 0;

    HCCL_INFO("[CcuKernelAllGatherMesh1DMem2Mem] AllGatherMesh1DMem2Mem run");
    CCU_CHK_RET(ParseKernelArg(ctx, kernelArg));
    CCU_CHK_RET(InitResource(ctx));
    CCU_CHK_RET(LoadArgs(ctx));

    CCU_CHK_RET(PreSync(ctx));

    CCU_CHK_RET(DoRepeatAllGather(ctx));

    CCU_CHK_RET(PostSync(ctx));
    HCCL_INFO("[CcuKernelAllGatherMesh1DMem2Mem] AllGatherMesh1DMem2Mem end");

    return CCU_SUCCESS;
}

} // namespace ops_hccl
