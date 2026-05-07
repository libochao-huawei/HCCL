/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "ccu_kernel_reduce_scatter_mesh1d_mem2mem.h"

namespace ops_hccl {

constexpr int INPUT_XN_ID   = 0;
constexpr int SCRATCH_XN_ID = 1;
constexpr int TOKEN_XN_ID   = 2;
constexpr int POST_SYNC_ID  = 3;
constexpr int CKE_IDX_0     = 0;

static CcuResult ParseKernelArg(ReduceScatterMesh1DMem2MemContext &ctx, CcuKernelArgReduceScatterMesh1DMem2Mem *kernelArg)
{
    ctx.dataType        = kernelArg->opParam_.DataDes.dataType;
    ctx.outputDataType  = kernelArg->opParam_.DataDes.outputType;
    if (ctx.outputDataType == HcclDataType::HCCL_DATA_TYPE_RESERVED) {
        ctx.outputDataType = ctx.dataType;
        HCCL_DEBUG("[CcuKernelReduceScatterMesh1DMem2Mem] outputDataType is [INVALID], set outputDataType to[%d]",
            ctx.dataType);
    }
    ctx.reduceOp = kernelArg->opParam_.reduceType;
    return CCU_SUCCESS;
}

static CcuResult InitResource(ReduceScatterMesh1DMem2MemContext &ctx)
{
    const auto *arg = ctx.arg;
    uint32_t channelIdx = 0;

    if (arg->channelCount == 0) {
        HCCL_ERROR("[CcuKernelReduceScatterMesh1DMem2Mem] channels is empty!");
        return CcuResult::CCU_E_INTERNAL;
    }

    for (uint64_t peerId = 0; peerId < arg->rankSize; peerId++) {
        if (peerId == arg->rankId) {
            CCU_CHK_RET(ccu::Alloc(&ctx.input[peerId]));
            CCU_CHK_RET(ccu::Alloc(&ctx.scratch[peerId]));
            CCU_CHK_RET(ccu::Alloc(&ctx.token[peerId]));
        } else {
            CCU_CHK_RET(ccu::CreateByChannel(
                arg->channels[channelIdx], INPUT_XN_ID, &ctx.input[peerId]));
            CCU_CHK_RET(ccu::CreateByChannel(
                arg->channels[channelIdx], SCRATCH_XN_ID, &ctx.scratch[peerId]));
            CCU_CHK_RET(ccu::CreateByChannel(
                arg->channels[channelIdx], TOKEN_XN_ID, &ctx.token[peerId]));
            channelIdx++;
        }
    }

    CCU_CHK_RET(ccu::Alloc(&ctx.output));
    CCU_CHK_RET(ccu::Alloc(&ctx.currentRankSliceInputOffset));
    CCU_CHK_RET(ccu::Alloc(&ctx.currentRankSliceOutputOffset));
    CCU_CHK_RET(ccu::Alloc(&ctx.normalSliceSize));
    CCU_CHK_RET(ccu::Alloc(&ctx.lastSliceSize));
    CCU_CHK_RET(ccu::Alloc(&ctx.inputRepeatStride));
    CCU_CHK_RET(ccu::Alloc(&ctx.outputRepeatStride));
    CCU_CHK_RET(ccu::Alloc(&ctx.repeatNum));
    CCU_CHK_RET(ccu::Alloc(&ctx.flag));

    CCU_CHK_RET(ccu::Alloc(&ctx.goSize.addrOffset));
    CCU_CHK_RET(ccu::Alloc(&ctx.goSize.loopParam));
    CCU_CHK_RET(ccu::Alloc(&ctx.goSize.parallelParam));
    CCU_CHK_RET(ccu::Alloc(&ctx.goSize.residual));

    ctx.selfBit = 1 << arg->rankId;
    ctx.allBit  = ((1 << arg->rankSize) - 1) & (~(1 << arg->rankId));

    for (uint64_t rankIdx = 0; rankIdx < arg->rankSize; rankIdx++) {
        CCU_CHK_RET(ccu::Alloc(&ctx.scratchMem[rankIdx]));
        if (rankIdx == arg->rankId) {
            CCU_CHK_RET(ccu::Alloc(&ctx.myInput));
        } else {
            CCU_CHK_RET(ccu::Alloc(&ctx.remoteInput[rankIdx]));
        }
    }

    CCU_CHK_RET(ccu::Alloc(&ctx.event));

    ctx.resourceAllocated = false;
    ctx.loopRegistered    = false;

    return CCU_SUCCESS;
}

static CcuResult LoadArgs(ReduceScatterMesh1DMem2MemContext &ctx)
{
    const auto *arg = ctx.arg;

    CCU_CHK_RET(ccu::LoadArg(ctx.input[arg->rankId]));
    CCU_CHK_RET(ccu::LoadArg(ctx.output));
    CCU_CHK_RET(ccu::LoadArg(ctx.token[arg->rankId]));
    CCU_CHK_RET(ccu::LoadArg(ctx.scratch[arg->rankId]));
    CCU_CHK_RET(ccu::LoadArg(ctx.currentRankSliceInputOffset));
    CCU_CHK_RET(ccu::LoadArg(ctx.currentRankSliceOutputOffset));
    CCU_CHK_RET(ccu::LoadArg(ctx.inputRepeatStride));
    CCU_CHK_RET(ccu::LoadArg(ctx.outputRepeatStride));
    CCU_CHK_RET(ccu::LoadArg(ctx.normalSliceSize));
    CCU_CHK_RET(ccu::LoadArg(ctx.lastSliceSize));
    CCU_CHK_RET(ccu::LoadArg(ctx.repeatNum));

    CCU_CHK_RET(ccu::LoadArg(ctx.goSize.addrOffset));
    CCU_CHK_RET(ccu::LoadArg(ctx.goSize.loopParam));
    CCU_CHK_RET(ccu::LoadArg(ctx.goSize.parallelParam));
    CCU_CHK_RET(ccu::LoadArg(ctx.goSize.residual));

    return CCU_SUCCESS;
}

static void PreSync(ReduceScatterMesh1DMem2MemContext &ctx)
{
    const auto *arg = ctx.arg;

    for (uint32_t i = 0; i < arg->channelCount; i++) {
        ccu::WriteVariableWithNotify(arg->channels[i], ctx.input[arg->rankId],
            INPUT_XN_ID, CKE_IDX_0, 1 << INPUT_XN_ID);
        ccu::WriteVariableWithNotify(arg->channels[i], ctx.scratch[arg->rankId],
            SCRATCH_XN_ID, CKE_IDX_0, 1 << SCRATCH_XN_ID);
        ccu::WriteVariableWithNotify(arg->channels[i], ctx.token[arg->rankId],
            TOKEN_XN_ID, CKE_IDX_0, 1 << TOKEN_XN_ID);
    }

    uint32_t allBit = (1 << INPUT_XN_ID) | (1 << SCRATCH_XN_ID) | (1 << TOKEN_XN_ID);
    for (uint32_t i = 0; i < arg->channelCount; i++) {
        ccu::NotifyWait(arg->channels[i], CKE_IDX_0, allBit);
    }
}

static void PostSync(ReduceScatterMesh1DMem2MemContext &ctx)
{
    const auto *arg = ctx.arg;

    for (uint32_t i = 0; i < arg->channelCount; i++) {
        ccu::NotifyRecord(arg->channels[i], CKE_IDX_0, 1 << POST_SYNC_ID);
    }
    for (uint32_t i = 0; i < arg->channelCount; i++) {
        ccu::NotifyWait(arg->channels[i], CKE_IDX_0, 1 << POST_SYNC_ID);
    }
}

static CcuResult CreateReduceLoop(ReduceScatterMesh1DMem2MemContext &ctx)
{
    const auto *arg = ctx.arg;
    const uint32_t size = arg->rankSize;

    constexpr uint32_t LOOP_NUM = 16;
    CHK_RET(AllocGoResource(ctx.moConfig, ctx.moRes, ctx.resourceAllocated, LOOP_NUM));

    if (ctx.loopRegistered) {
        return CCU_SUCCESS;
    }

    uint32_t expansionNum = GetReduceExpansionNum(arg->reduceOp, arg->dataType, arg->outputDataType);
    uint32_t usedBufNum   = size > expansionNum ? size : expansionNum;

    for (int32_t index = 0; index < 2; index++) {
        CCU_CHK_RET(ccu::Alloc(&ctx.loopDst[index]));
        CCU_CHK_RET(ccu::Alloc(&ctx.loopSrc[index]));
        for (uint32_t i = 0; i < size; i++) {
            CCU_CHK_RET(ccu::Alloc(&ctx.loopScratch[index][i]));
        }
        CCU_CHK_RET(ccu::Alloc(&ctx.loopLen[index]));
        CCU_CHK_RET(ccu::Alloc(&ctx.loopLenExp[index]));

        uint32_t bufBase = index * ctx.moConfig.msInterleave;

        CcuEvent loopEvt = ctx.moRes.completedEvent[index];

        CCU_LOOP(ctx.reduceLoops[index]) {
            for (uint32_t i = 0; i < size; i++) {
                loopEvt.setMask(1 << i);
                if (i == arg->rankId) {
                    CcuLocalCopyHBMToBuffer(
                        ctx.moRes.ccuBuf[bufBase + i], ctx.loopSrc[index],
                        ctx.loopLen[index], loopEvt);
                } else {
                    CcuLocalCopyHBMToBuffer(
                        ctx.moRes.ccuBuf[bufBase + i], ctx.loopScratch[index][i],
                        ctx.loopLen[index], loopEvt);
                }
            }
            loopEvt.setMask((1 << size) - 1);
            WaitEvent(loopEvt);

            if (size > 1) {
                loopEvt.setMask(1);
                LocalReduceNb(
                    &ctx.moRes.ccuBuf[bufBase], size,
                    arg->dataType, arg->outputDataType, arg->reduceOp,
                    ctx.loopLen[index], loopEvt);
                WaitEvent(loopEvt);
            }

            loopEvt.setMask(1);
            CcuLocalCopyBufferToHBM(
                ctx.loopDst[index], ctx.moRes.ccuBuf[bufBase],
                ctx.loopLenExp[index], loopEvt);
            WaitEvent(loopEvt);
        }
    }

    ctx.loopRegistered = true;
    return CCU_SUCCESS;
}

static CcuResult ReduceLoopGroup(ReduceScatterMesh1DMem2MemContext &ctx,
    ccu::LocalAddr outDstOrg, ccu::LocalAddr srcOrg,
    ccu::LocalAddr *scratchOrg, uint32_t scratchCount,
    GroupOpSizeVars &goSize)
{
    const auto *arg = ctx.arg;
    const uint32_t size = scratchCount;

    ccu::LocalAddr dst;
    CCU_CHK_RET(ccu::Alloc(&dst));
    dst.addr  = outDstOrg.addr;
    dst.token = outDstOrg.token;

    ccu::LocalAddr src;
    CCU_CHK_RET(ccu::Alloc(&src));
    src.addr  = srcOrg.addr;
    src.token = srcOrg.token;

    ccu::LocalAddr scratch[CCU_MAX_RANK_SIZE];
    for (uint32_t idx = 0; idx < size; idx++) {
        CCU_CHK_RET(ccu::Alloc(&scratch[idx]));
        scratch[idx].addr  = scratchOrg[idx].addr;
        scratch[idx].token = scratchOrg[idx].token;
    }

    CCU_CHK_RET(CreateReduceLoop(ctx));

    uint32_t expansionNum = GetReduceExpansionNum(arg->reduceOp, arg->dataType, arg->outputDataType);
    CcuVariable sliceSizeExpansion;
    CCU_CHK_RET(ccu::Alloc(&sliceSizeExpansion));

    if (expansionNum != 1) {
        CcuVariable tmp;
        CCU_CHK_RET(ccu::Alloc(&tmp));
        tmp = GetExpansionParam(expansionNum);
        dst.token = dst.token + tmp;
    }

    CCU_IF_ONLY(goSize.loopParam != 0) {
        CcuVariable loopParam;
        CCU_CHK_RET(ccu::Alloc(&loopParam));
        loopParam = GetLoopParam(0, ctx.moConfig.memSlice * ctx.moConfig.loopCount, 0);
        loopParam = loopParam + goSize.loopParam;

        CcuVariable sliceSize;
        CCU_CHK_RET(ccu::Alloc(&sliceSize));
        sliceSize          = ctx.moConfig.memSlice;
        sliceSizeExpansion = ctx.moConfig.memSlice * expansionNum;

        ctx.loopDst[0].addr  = dst.addr;
        ctx.loopDst[0].token = dst.token;
        ctx.loopSrc[0].addr  = src.addr;
        ctx.loopSrc[0].token = src.token;
        for (uint32_t i = 0; i < size; i++) {
            ctx.loopScratch[0][i].addr  = scratch[i].addr;
            ctx.loopScratch[0][i].token = scratch[i].token;
        }
        ctx.loopLen[0]    = sliceSize;
        ctx.loopLenExp[0] = sliceSizeExpansion;

        CcuVariable paraCfg;
        CCU_CHK_RET(ccu::Alloc(&paraCfg));
        paraCfg = GetParallelParam(ctx.moConfig.loopCount - 1, 0, 1);

        CcuVariable offsetCfg;
        CCU_CHK_RET(ccu::Alloc(&offsetCfg));
        offsetCfg = GetOffsetParam(ctx.moConfig.memSlice, ctx.moConfig.msInterleave, 1);

        CcuLoopGroup group;
        CcuLoopExecutors enginePool;
        CCU_CHK_RET(ccu::CreateLoopGroup(&group, &paraCfg, &offsetCfg, enginePool));
        CCU_CHK_RET(ccu::AddLoop(group, ctx.reduceLoops[0], &loopParam));
    }

    CCU_IF_ONLY(goSize.parallelParam != 0) {
        for (uint32_t i = 0; i < size; i++) {
            scratch[i].addr += goSize.addrOffset;
        }
        src.addr += goSize.addrOffset;
        for (uint32_t i = 0; i < expansionNum; i++) {
            dst.addr += goSize.addrOffset;
        }

        sliceSizeExpansion = 0;
        for (uint32_t i = 0; i < expansionNum; i++) {
            sliceSizeExpansion = sliceSizeExpansion + goSize.residual;
        }

        ctx.loopDst[0].addr  = dst.addr;
        ctx.loopDst[0].token = dst.token;
        ctx.loopSrc[0].addr  = src.addr;
        ctx.loopSrc[0].token = src.token;
        for (uint32_t i = 0; i < size; i++) {
            ctx.loopScratch[0][i].addr  = scratch[i].addr;
            ctx.loopScratch[0][i].token = scratch[i].token;
        }
        ctx.loopLen[0]    = goSize.residual;
        ctx.loopLenExp[0] = sliceSizeExpansion;

        for (uint32_t i = 0; i < size; i++) {
            scratch[i].addr += goSize.residual;
        }
        src.addr += goSize.residual;
        for (uint32_t i = 0; i < expansionNum; i++) {
            dst.addr += goSize.residual;
        }

        CcuVariable sliceSize;
        CCU_CHK_RET(ccu::Alloc(&sliceSize));
        sliceSize          = ctx.moConfig.memSlice;
        sliceSizeExpansion = ctx.moConfig.memSlice * expansionNum;

        ctx.loopDst[1].addr  = dst.addr;
        ctx.loopDst[1].token = dst.token;
        ctx.loopSrc[1].addr  = src.addr;
        ctx.loopSrc[1].token = src.token;
        for (uint32_t i = 0; i < size; i++) {
            ctx.loopScratch[1][i].addr  = scratch[i].addr;
            ctx.loopScratch[1][i].token = scratch[i].token;
        }
        ctx.loopLen[1]    = sliceSize;
        ctx.loopLenExp[1] = sliceSizeExpansion;

        CcuVariable loopCfg0;
        CCU_CHK_RET(ccu::Alloc(&loopCfg0));
        loopCfg0 = GetLoopParam(0, 0, 1);

        CcuVariable loopCfg1;
        CCU_CHK_RET(ccu::Alloc(&loopCfg1));
        loopCfg1 = GetLoopParam(0, 0, 1);

        CcuVariable offsetCfg;
        CCU_CHK_RET(ccu::Alloc(&offsetCfg));
        offsetCfg = GetOffsetParam(ctx.moConfig.memSlice, ctx.moConfig.msInterleave, 1);

        CcuLoopGroup group;
        CcuLoopExecutors enginePool;
        CCU_CHK_RET(ccu::CreateLoopGroup(&group, &goSize.parallelParam, &offsetCfg, enginePool));
        CCU_CHK_RET(ccu::AddLoop(group, ctx.reduceLoops[0], &loopCfg0));
        CCU_CHK_RET(ccu::AddLoop(group, ctx.reduceLoops[1], &loopCfg1));
    }

    return CCU_SUCCESS;
}

static CcuResult DoReduceScatter(ReduceScatterMesh1DMem2MemContext &ctx)
{
    const auto *arg = ctx.arg;
    uint32_t channelId = 0;

    ccu::LocalAddr myOutput;
    CCU_CHK_RET(ccu::Alloc(&myOutput));
    myOutput.addr  = ctx.output;
    myOutput.addr += ctx.currentRankSliceOutputOffset;
    myOutput.token = ctx.token[arg->rankId];

    CcuVariable sliceSize;
    CCU_CHK_RET(ccu::Alloc(&sliceSize));
    sliceSize = (arg->rankId == (arg->rankSize - 1)) ? ctx.lastSliceSize : ctx.normalSliceSize;

    CCU_IF_ONLY(sliceSize != 0) {
        for (uint32_t rankIdx = 0; rankIdx < arg->rankSize; rankIdx++) {
            ctx.event.setMask(1 << rankIdx);
            if (rankIdx == arg->rankId) {
                CcuRecordEvent(ctx.event);
            } else {
                CcuReadHBMToHBM(
                    arg->channels[channelId],
                    ctx.scratchMem[rankIdx],
                    ctx.remoteInput[rankIdx],
                    sliceSize, ctx.event);
                channelId++;
            }
        }

        ctx.event.setMask((1 << arg->rankSize) - 1);
        WaitEvent(ctx.event);

        ReduceLoopGroup(ctx, myOutput, ctx.myInput,
            ctx.scratchMem, arg->rankSize, ctx.goSize);
    }

    return CCU_SUCCESS;
}

static CcuResult DoRepeatReduceScatter(ReduceScatterMesh1DMem2MemContext &ctx)
{
    const auto *arg = ctx.arg;

    CcuVariable scratchOffset;
    CCU_CHK_RET(ccu::Alloc(&scratchOffset));
    scratchOffset = 0;

    for (uint32_t rankIdx = 0; rankIdx < arg->rankSize; rankIdx++) {
        if (rankIdx == arg->rankId) {
            ctx.myInput.addr  = ctx.input[rankIdx];
            ctx.myInput.addr += ctx.currentRankSliceInputOffset;
            ctx.myInput.token = ctx.token[rankIdx];
        } else {
            ctx.remoteInput[rankIdx].addr  = ctx.input[rankIdx];
            ctx.remoteInput[rankIdx].addr += ctx.currentRankSliceInputOffset;
            ctx.remoteInput[rankIdx].token = ctx.token[rankIdx];
        }

        ctx.scratchMem[rankIdx].addr  = ctx.scratch[arg->rankId];
        ctx.scratchMem[rankIdx].addr += scratchOffset;
        scratchOffset = scratchOffset + ctx.normalSliceSize;
        ctx.scratchMem[rankIdx].token = ctx.token[arg->rankId];
    }

    CcuVariable repeatNumAdd;
    CCU_CHK_RET(ccu::Alloc(&repeatNumAdd));
    repeatNumAdd = 1;
    ctx.flag     = 0;

    CCU_WHILE(ctx.repeatNum != UINT64_MAX) {
        ctx.repeatNum = ctx.repeatNum + repeatNumAdd;

        CCU_IF_ONLY(ctx.flag == 1) {
            for (uint64_t rankIdx = 0; rankIdx < arg->rankSize; rankIdx++) {
                if (rankIdx == arg->rankId) {
                    ctx.myInput.addr += ctx.inputRepeatStride;
                } else {
                    ctx.remoteInput[rankIdx].addr += ctx.inputRepeatStride;
                }
            }
            ctx.output = ctx.output + ctx.outputRepeatStride;
        }

        DoReduceScatter(ctx);
        ctx.flag = 1;
    }

    return CCU_SUCCESS;
}

CcuResult CcuReduceScatterMesh1dMem2MemKernel(CcuKernelArg arg)
{
    auto *kernelArg = static_cast<CcuKernelArgReduceScatterMesh1DMem2Mem *>(arg);

    ReduceScatterMesh1DMem2MemContext ctx;
    ctx.arg = kernelArg;
    ctx.selfBit = 0;
    ctx.allBit = 0;
    ctx.resourceAllocated = false;
    ctx.loopRegistered = false;
    ctx.moConfig.msInterleave = 0;
    ctx.moConfig.loopCount = 0;
    ctx.moConfig.memSlice = 0;
    ctx.moRes.eventCount = 0;
    ctx.moRes.bufCount = 0;

    HCCL_INFO("[CcuKernelReduceScatterMesh1DMem2Mem] ReduceScatterMesh1DMem2Mem run");
    CCU_CHK_RET(ParseKernelArg(ctx, kernelArg));
    CCU_CHK_RET(InitResource(ctx));
    CCU_CHK_RET(LoadArgs(ctx));

    PreSync(ctx);

    CCU_CHK_RET(DoRepeatReduceScatter(ctx));

    PostSync(ctx);

    HCCL_INFO("[CcuKernelReduceScatterMesh1DMem2Mem] ReduceScatterMesh1DMem2Mem end");
    return CCU_SUCCESS;
}

} // namespace ops_hccl