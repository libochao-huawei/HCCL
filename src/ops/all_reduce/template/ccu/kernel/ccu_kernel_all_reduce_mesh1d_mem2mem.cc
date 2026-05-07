/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "ccu_kernel_all_reduce_mesh1d_mem2mem.h"

namespace ops_hccl {

constexpr int INPUT_XN_ID   = 0;
constexpr int OUTPUT_XN_ID  = 1;
constexpr int SCRATCH_XN_ID = 2;
constexpr int TOKEN_XN_ID   = 3;
constexpr int POST_SYNC_ID   = 4;
constexpr int CKE_IDX_0     = 0;

struct GroupReduceMem2MemVar {
    ccu::LocalAddr loopDst[2];
    ccu::LocalAddr loopSrc[2];
    std::array<std::vector<ccu::LocalAddr>, 2> loopScratch;
    CcuVariable  loopLen[2];
    CcuVariable  loopLenExp[2];
};

static CcuResult ParseKernelArg(AllReduceMeshMem2Mem1DContext &ctx, CcuKernelArgAllReduceMeshMem2Mem1D *kernelArg)
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

static CcuResult InitResource(AllReduceMeshMem2Mem1DContext &ctx)
{
    uint16_t channelIdx = 0;
    if (ctx.arg->channelCount == 0) {
        HCCL_ERROR("[CcuAllReduceMeshMem2Mem1D] channels is empty!");
        return CcuResult::CCU_E_INTERNAL;
    }
    const auto *arg = ctx.arg;
    ctx.input.resize(arg->rankSize);
    ctx.output.resize(arg->rankSize);
    ctx.scratch.resize(arg->rankSize);
    ctx.token.resize(arg->rankSize);
    for (uint64_t peerId = 0; peerId < arg->rankSize; peerId++) {
        if (peerId == arg->rankId) {
            CCU_CHK_RET(ccu::Alloc(&ctx.input[peerId]));
            CCU_CHK_RET(ccu::Alloc(&ctx.output[peerId]));
            CCU_CHK_RET(ccu::Alloc(&ctx.scratch[peerId]));
            CCU_CHK_RET(ccu::Alloc(&ctx.token[peerId]));
        } else {
            CCU_CHK_RET(ccu::CreateByChannel(
                arg->channels[channelIdx], INPUT_XN_ID, &ctx.input[peerId]));
            CCU_CHK_RET(ccu::CreateByChannel(
                arg->channels[channelIdx], OUTPUT_XN_ID, &ctx.output[peerId]));
            CCU_CHK_RET(ccu::CreateByChannel(
                arg->channels[channelIdx], TOKEN_XN_ID, &ctx.token[peerId]));
            CCU_CHK_RET(ccu::CreateByChannel(
                arg->channels[channelIdx], SCRATCH_XN_ID, &ctx.scratch[peerId]));
            channelIdx++;
        }
    }

    CCU_CHK_RET(ccu::Alloc(&ctx.currentRankSliceInputOffset));
    CCU_CHK_RET(ccu::Alloc(&ctx.currentRankSliceOutputOffset));
    CCU_CHK_RET(ccu::Alloc(&ctx.normalSliceSize));
    CCU_CHK_RET(ccu::Alloc(&ctx.lastSliceSize));
    CCU_CHK_RET(ccu::Alloc(&ctx.mySliceSize));
    CCU_CHK_RET(ccu::Alloc(&ctx.sliceOffset));
    CCU_CHK_RET(ccu::Alloc(&ctx.isInputOutputEqual));
    CCU_CHK_RET(ccu::Alloc(&ctx.locEvent));
    CCU_CHK_RET(ccu::Alloc(&ctx.srcMem));
    CCU_CHK_RET(ccu::Alloc(&ctx.localDstMem));
    CCU_CHK_RET(ccu::Alloc(&ctx.remoteDstMem));
    ctx.reduceScatterSrc.reserve(arg->rankSize);
    for (uint32_t rankIdx = 0; rankIdx < arg->rankSize; rankIdx++) {
        CCU_CHK_RET(ccu::Alloc(&ctx.reduceScatterSrc[rankIdx]));
    }
    ctx.reduceScatterDst.reserve(arg->rankSize);
    for (uint32_t rankIdx = 0; rankIdx < arg->rankSize; rankIdx++) {
        CCU_CHK_RET(ccu::Alloc(&ctx.reduceScatterDst[rankIdx]));
    }
    CCU_CHK_RET(ccu::Alloc(&ctx.sliceSize));
    CCU_CHK_RET(ccu::Alloc(&ctx.goSize.addrOffset));
    CCU_CHK_RET(ccu::Alloc(&ctx.goSize.loopParam));
    CCU_CHK_RET(ccu::Alloc(&ctx.goSize.parallelParam));
    CCU_CHK_RET(ccu::Alloc(&ctx.goSize.residual));

    CCU_CHK_RET(ccu::CreateLoopExecutor(&ctx.enginePool, arg->rankSize + 1));
    ctx.resourceAllocated = false;
    return CCU_SUCCESS;
}
 
static CcuResult CreateReduceLoop(AllReduceMeshMem2Mem1DContext &ctx, 
    GroupReduceMem2MemVar &var, uint32_t size, HcclDataType dataType, HcclDataType outputDataType,
    HcclReduceOp opType)
{
    constexpr uint32_t LOOP_NUM = 16;
    AllocGoResource(ctx.moConfig, ctx.moRes, ctx.resourceAllocated, LOOP_NUM);
    if (ctx.IsLoopRegistered("reduce_mesh1d_mem2mem")) {
        return CCU_SUCCESS;
    }
    ctx.CreateLoop("reduce_mesh1d_mem2mem");
    auto &loops = ctx.loopMap["reduce_mesh1d_mem2mem"];

    uint32_t expansionNum = GetReduceExpansionNum(opType, dataType, outputDataType);
    uint32_t usedBufNum   = size > expansionNum ? size : expansionNum;

    for (int32_t index = 0; index < 2; index++) { // 需要实例化2个Loop
        CCU_CHK_RET(ccu::Alloc(&var.loopDst[index]));
        CCU_CHK_RET(ccu::Alloc(&var.loopSrc[index]));
        for (uint32_t i = 0; i < size; i++) {
            CCU_CHK_RET(ccu::Alloc(&var.loopScratch[index][i]));
        }
        CCU_CHK_RET(ccu::Alloc(&var.loopLen[index]));
        CCU_CHK_RET(ccu::Alloc(&var.loopLenExp[index]));
        uint32_t bufBase = index * ctx.moConfig.msInterleave;
        CcuEvent             e  = ctx.moRes.completedEvent[index];
        CCU_LOOP(loops[index]) {
            for (uint32_t i = 0; i < size; i++) {
                e.setMask(1 << i);
                if (i == ctx.arg->rankId) {
                    ccu::LocalCopyNb(ctx.moRes.ccuBuf[bufBase + i], var.loopSrc[index], var.loopLen[index], e);
                } else {
                    ccu::LocalCopyNb(ctx.moRes.ccuBuf[bufBase + i], var.loopScratch[index][i], var.loopLen[index], e);
                }
            }
            e.setMask((1 << size) - 1);
            ccu::WaitEvent(e);
            e.setMask(1);
            if (size > 1) {
                ccu::LocalReduceNb(&ctx.moRes.ccuBuf[bufBase], size, dataType, outputDataType, opType, var.loopLen[index], e); 
                ccu::WaitEvent(e);
            }

            ccu::LocalCopyNb(var.loopDst[index], ctx.moRes.ccuBuf[bufBase], var.loopLenExp[index], e);
            ccu::WaitEvent(e);
        }
    }
    return CCU_SUCCESS;
}

static CcuResult ReduceLoopGroup(AllReduceMeshMem2Mem1DContext &ctx, ccu::LocalAddr srcOrg)
{
    const uint32_t size = ctx.reduceScatterDst.size();

    ccu::LocalAddr dst;
    CCU_CHK_RET(ccu::Alloc(&dst));
    dst.addr = ctx.localDstMem.addr;
    dst.token = ctx.localDstMem.token;

    ccu::LocalAddr src;
    CCU_CHK_RET(ccu::Alloc(&src));
    src.addr = srcOrg.addr;
    src.token = srcOrg.token;

    std::vector<ccu::LocalAddr> scratch;
    for (uint32_t idx = 0; idx < size; idx++) {
        CCU_CHK_RET(ccu::Alloc(&scratch[idx]));
        scratch[idx].addr = ctx.reduceScatterDst[idx].addr;
        scratch[idx].token = ctx.reduceScatterDst[idx].token;
    }
    GroupReduceMem2MemVar var;
    CCU_CHK_RET(CreateReduceLoop(ctx, var, size, ctx.dataType, ctx.outputDataType, ctx.reduceOp));
    auto &loops = ctx.loopMap["reduce_mesh1d_mem2mem"];

    uint32_t         expansionNum = GetReduceExpansionNum(ctx.reduceOp, ctx.dataType, ctx.outputDataType);
    CcuVariable sliceSizeExpansion;
    CCU_CHK_RET(ccu::Alloc(&sliceSizeExpansion));

    if (expansionNum != 1) {
        CcuVariable tmp;
        CCU_CHK_RET(ccu::Alloc(&tmp));
        tmp = GetExpansionParam(expansionNum);
        dst.token = dst.token + tmp;
    }

    // m部分
    CCU_IF_ONLY(ctx.goSize.loopParam != 0)                   // goSize1
    {
        CcuVariable loopParam;
        CCU_CHK_RET(ccu::Alloc(&loopParam));
        loopParam = GetLoopParam(0, ctx.moConfig.memSlice * ctx.moConfig.loopCount, 0);
        loopParam = loopParam + ctx.goSize.loopParam;

        CcuVariable sliceSize;
        CCU_CHK_RET(ccu::Alloc(&sliceSize));
        sliceSize          = ctx.moConfig.memSlice;
        sliceSizeExpansion = ctx.moConfig.memSlice * expansionNum;

        for (uint32_t i = 0; i < size; ++i) {
            var.loopScratch[0][i].addr = scratch[i].addr;
            var.loopScratch[0][i].token = scratch[i].token;
        }
        var.loopSrc[0].addr = src.addr;
        var.loopSrc[0].token = src.token;
        var.loopDst[0].addr  = dst.addr;
        var.loopDst[0].token = dst.token;
        var.loopLen[0]       = sliceSize;
        var.loopLenExp[0]    = sliceSizeExpansion;

        CcuVariable paraCfg;
        CCU_CHK_RET(ccu::Alloc(&paraCfg));
        paraCfg = GetParallelParam(ctx.moConfig.loopCount - 1, 0, 1);

        CcuVariable offsetCfg;
        CCU_CHK_RET(ccu::Alloc(&offsetCfg));
        offsetCfg = GetOffsetParam(ctx.moConfig.memSlice, ctx.moConfig.msInterleave, 1);

        CcuLoopGroup group;
        CCU_CHK_RET(ccu::CreateLoopGroup(&group, &paraCfg, &offsetCfg, ctx.enginePool));
        CCU_CHK_RET(ccu::AddLoop(group, loops[0], &loopParam));
    }

    CCU_IF_ONLY(ctx.goSize.parallelParam != 0)               // goSize2
    {
        // p部分，加m的偏移
        for (uint32_t i = 0; i < size; i++) {
            scratch[i].addr += ctx.goSize.addrOffset;
        }
        src.addr += ctx.goSize.addrOffset;              // goSize0
        for (uint32_t i = 0; i < expansionNum; i++) {
            dst.addr += ctx.goSize.addrOffset;
        }

        sliceSizeExpansion = 0;
        for (uint32_t i = 0; i < expansionNum; i++) {
            sliceSizeExpansion += ctx.goSize.residual;  // goSize3
        }

        for (uint32_t i = 0; i < size; ++i) {
            var.loopScratch[0][i].addr = scratch[i].addr;
            var.loopScratch[0][i].token = scratch[i].token;
        }
        var.loopSrc[0].addr = src.addr;
        var.loopSrc[0].token = src.token;
        var.loopDst[0].addr  = dst.addr;
        var.loopDst[0].token = dst.token;
        var.loopLen[0]       = ctx.goSize.residual;
        var.loopLenExp[0]    = sliceSizeExpansion;
        
        // n部分，再加p的偏移
        for (uint32_t i = 0; i < size; i++) {
            scratch[i].addr += ctx.goSize.residual;
        }
        src.addr += ctx.goSize.residual;
        for (uint32_t i = 0; i < expansionNum; i++) {
            dst.addr += ctx.goSize.residual;
        }

        CcuVariable sliceSize;
        CCU_CHK_RET(ccu::Alloc(&sliceSize));
        sliceSize          = ctx.moConfig.memSlice;
        sliceSizeExpansion = ctx.moConfig.memSlice * expansionNum;

        var.loopSrc[1].addr = src.addr;
        var.loopSrc[1].token = src.token;
        var.loopDst[1].addr  = dst.addr;
        var.loopDst[1].token = dst.token;
        var.loopLen[1]       = sliceSize;
        var.loopLenExp[1]    = sliceSizeExpansion;

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
        CCU_CHK_RET(ccu::CreateLoopGroup(&group, &ctx.goSize.parallelParam, &offsetCfg, ctx.enginePool));
        CCU_CHK_RET(ccu::AddLoop(group, loops[0], &loopCfg0));
        CCU_CHK_RET(ccu::AddLoop(group, loops[1], &loopCfg1));
    }
    return CCU_SUCCESS;
}

static CcuResult LoadArgs(AllReduceMeshMem2Mem1DContext &ctx)
{
    const auto *arg = ctx.arg;

    CCU_CHK_RET(ccu::LoadArg(ctx.input[arg->rankId]));
    CCU_CHK_RET(ccu::LoadArg(ctx.output[arg->rankId]));
    CCU_CHK_RET(ccu::LoadArg(ctx.token[arg->rankId]));
    CCU_CHK_RET(ccu::LoadArg(ctx.scratch[arg->rankId]));
    CCU_CHK_RET(ccu::LoadArg(ctx.currentRankSliceInputOffset));
    CCU_CHK_RET(ccu::LoadArg(ctx.currentRankSliceOutputOffset));
    CCU_CHK_RET(ccu::LoadArg(ctx.normalSliceSize));
    CCU_CHK_RET(ccu::LoadArg(ctx.lastSliceSize));
    CCU_CHK_RET(ccu::LoadArg(ctx.mySliceSize));
    CCU_CHK_RET(ccu::LoadArg(ctx.sliceOffset));
    CCU_CHK_RET(ccu::LoadArg(ctx.isInputOutputEqual));
    CCU_CHK_RET(ccu::LoadArg(ctx.goSize.addrOffset));
    CCU_CHK_RET(ccu::LoadArg(ctx.goSize.loopParam));
    CCU_CHK_RET(ccu::LoadArg(ctx.goSize.parallelParam));
    CCU_CHK_RET(ccu::LoadArg(ctx.goSize.residual));
    return CCU_SUCCESS;
}

static CcuResult PreSync(AllReduceMeshMem2Mem1DContext &ctx)
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
    return CCU_SUCCESS;
}

static CcuResult PostSync(AllReduceMeshMem2Mem1DContext &ctx)
{
    const auto *arg = ctx.arg;

    for (uint32_t i = 0; i < arg->channelCount; i++) {
        ccu::NotifyRecord(arg->channels[i], CKE_IDX_0, 1 << POST_SYNC_ID);
    }
    for (uint32_t i = 0; i < arg->channelCount; i++) {
        ccu::NotifyWait(arg->channels[i], CKE_IDX_0, 1 << POST_SYNC_ID);
    }
    HCCL_INFO("[CcuKernelAllReduceMeshMem2Mem1D] AllReduceMeshMem2Mem1D AllReduce GroupWait end");
    return CCU_SUCCESS;
}

static CcuResult BcastLocToRmt(AllReduceMeshMem2Mem1DContext &ctx, const CcuVariable &srcAddr,
                   const std::vector<CcuVariable> &dstAddr)
{
    const auto *arg = ctx.arg;
    if (dstAddr.size() != arg->channelCount + 1) {
         HCCL_ERROR("[BcastLocToRmt] srcAddr.size[%zu] != channels_ size[%zu] + 1", dstAddr.size(), arg->channelCount); 
         return CCU_SUCCESS;
    }
    ctx.srcMem.addr = srcAddr;
    ctx.srcMem.addr += ctx.sliceOffset;
    ctx.srcMem.token = ctx.token[arg->rankId];

    uint32_t channelIdx = 0;
    for (uint32_t rmtId = 0; rmtId < dstAddr.size(); rmtId++) {
        ctx.locEvent.setMask(1 << rmtId);
        if (rmtId == arg->rankId) {
            ccu::RecordEvent(ctx.locEvent);
            continue;
        }
        ctx.remoteDstMem.addr = dstAddr[rmtId];
        ctx.remoteDstMem.addr += ctx.sliceOffset;
        ctx.remoteDstMem.token = ctx.token[rmtId];

        ccu::WriteNb(arg->channels[channelIdx], ctx.remoteDstMem, ctx.srcMem, ctx.sliceSize, ctx.locEvent);
        channelIdx++;
    }
    ctx.locEvent.setMask((1 << arg->rankSize) - 1);
    ccu::WaitEvent(ctx.locEvent);
    return CCU_SUCCESS;
}

static CcuResult ReduceRmtToLoc(AllReduceMeshMem2Mem1DContext &ctx, const std::vector<CcuVariable> &srcAddr,
                    const CcuVariable              &dstAddr)
{
    const auto *arg = ctx.arg;
    if (srcAddr.size() != arg->channelCount + 1) {
        HCCL_ERROR("[ReduceRmtToLoc] srcAddr.size[%zu] != channels_ size[%zu] +1", srcAddr.size(), arg->channelCount);
        return CCU_SUCCESS;
    }

    ctx.localDstMem.addr = dstAddr;
    ctx.localDstMem.addr += ctx.sliceOffset;
    ctx.localDstMem.token = ctx.token[arg->rankId];

    CcuVariable scratchOffset;
    CCU_CHK_RET(ccu::Alloc(&scratchOffset));

    scratchOffset                  = 0;
    for (uint32_t rankIdx = 0; rankIdx < arg->rankSize; rankIdx++) {      
        ctx.reduceScatterSrc[rankIdx].addr = srcAddr[rankIdx];
        ctx.reduceScatterSrc[rankIdx].addr += ctx.sliceOffset;
        ctx.reduceScatterSrc[rankIdx].token = ctx.token[rankIdx];

        ctx.reduceScatterDst[rankIdx].addr = ctx.scratch[arg->rankId];
        ctx.reduceScatterDst[rankIdx].addr += scratchOffset;
        scratchOffset += ctx.normalSliceSize;
        ctx.reduceScatterDst[rankIdx].token = ctx.token[rankIdx];
    }

    uint32_t channelIdx = 0;
    for (uint32_t rankIdx = 0; rankIdx < arg->rankSize; rankIdx++) {
        ctx.locEvent.setMask(1 << rankIdx);
        if (rankIdx == arg->rankId) {
            ccu::RecordEvent(ctx.locEvent);
        } else {    
            ccu::ReadNb(arg->channels[channelIdx], ctx.reduceScatterDst[rankIdx], ctx.reduceScatterSrc[rankIdx], ctx.sliceSize, ctx.locEvent);
            channelIdx++;
        }
    }
    ctx.locEvent.setMask((1 << arg->rankSize) - 1);
    ccu::WaitEvent(ctx.locEvent);
    ccu::LocalAddr  srcLoc;
    CCU_CHK_RET(ccu::Alloc(&srcLoc));
    srcLoc.addr = ctx.reduceScatterSrc[arg->rankId].addr;
    srcLoc.token = ctx.reduceScatterSrc[arg->rankId].token;
    CCU_CHK_RET(ReduceLoopGroup(ctx, srcLoc));
    return CCU_SUCCESS;
}

static CcuResult DoRepeatAllReduce(AllReduceMeshMem2Mem1DContext &ctx)
{
    const auto *arg = ctx.arg;
    if (arg->rankId != arg->rankSize - 1) {
        ctx.sliceSize = ctx.normalSliceSize;
    } else {
        ctx.sliceSize = ctx.lastSliceSize;
    }
    CCU_CHK_RET(ReduceRmtToLoc(ctx, ctx.input, ctx.output[arg->rankId]));
    CCU_CHK_RET(BcastLocToRmt(ctx, ctx.output[arg->rankId], ctx.output));
    return CCU_SUCCESS;
}

CcuResult CcuAllReduceMeshMem2Mem1DKernel(CcuKernelArg arg)
{
 auto *kernelArg = static_cast<CcuKernelArgAllReduceMeshMem2Mem1D *>(arg);

    AllReduceMeshMem2Mem1DContext ctx;
    ctx.arg = kernelArg;
    ctx.resourceAllocated = false;
    ctx.moConfig.msInterleave = 0;
    ctx.moConfig.loopCount = 0;
    ctx.moConfig.memSlice = 0;
    ctx.moRes.eventCount = 0;
    ctx.moRes.bufCount = 0;
    ctx.enginePool = 0;

    HCCL_INFO("[CcuKernelAllReduceMeshMem2Mem1D] AllReduceMeshMem2Mem1D run");
    CCU_CHK_RET(ParseKernelArg(ctx, kernelArg));
    CCU_CHK_RET(InitResource(ctx));
    CCU_CHK_RET(LoadArgs(ctx));

    CCU_CHK_RET(PreSync(ctx));

    CCU_IF_ONLY(ctx.mySliceSize != 0)
    {
        CCU_CHK_RET(DoRepeatAllReduce(ctx));
    }

    CCU_CHK_RET(PostSync(ctx));
    HCCL_INFO("[CcuKernelAllReduceMeshMem2Mem1D] AllReduceMeshMem2Mem1D end");

    return CCU_SUCCESS;
}
} // namespace ops_hccl