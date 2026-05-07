/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "ccu_kernel_alg_base.h"
#include "ccu_kernel_reduce_scatter_mesh1d_mem2mem.h"

namespace ops_hccl {
using namespace hcomm;

// bit序号，每种信号用一个bit
constexpr int INPUT_XN_ID   = 0;
constexpr int SCRATCH_XN_ID = 1;
constexpr int TOKEN_XN_ID   = 2;
constexpr int POST_SYNC_ID   = 3;  
// cke序号
constexpr int CKE_IDX_0     = 0;

CcuKernelReduceScatterMesh1DMem2Mem::CcuKernelReduceScatterMesh1DMem2Mem(const CcuKernelArg &arg)
    : CcuKernelAlgBase(arg)
{
    const CcuKernelArgReduceScatterMesh1DMem2Mem *kernelArg
        = dynamic_cast<const CcuKernelArgReduceScatterMesh1DMem2Mem *>(&arg);
    rankId_         = kernelArg->rankId_;
    rankSize_       = kernelArg->dimSize_;
    channels_       = kernelArg->channels;
    dataType_       = kernelArg->opParam_.DataDes.dataType;
    outputDataType_ = kernelArg->opParam_.DataDes.outputType;
    if (outputDataType_ == HcclDataType::HCCL_DATA_TYPE_RESERVED) {
        outputDataType_ = dataType_;
        HCCL_DEBUG(
            "[CcuKernelReduceScatterMesh1DMem2Mem] outputDataType is [INVALID], set outputDataType to[%d]",
            outputDataType_);
    }
    reduceOp_       = kernelArg->opParam_.reduceType;
    HCCL_INFO(
        "[CcuKernelReduceScatterMesh1DMem2Mem] Init, KernelArgs are rankId[%u], rankSize_[%u], dataType[%d], "
        "outputDataType[%d], reduceOp[%d]",
        rankId_, rankSize_, dataType_, outputDataType_, reduceOp_);
}

static HcclResult InitResource(ReduceScatterMesh1DMem2MemContext &ctx)
{
    const auto *arg = ctx.arg;
    uint32_t channelIdx = 0;

    if (arg->channelCount == 0) {
        return CCU_E_PARA;
    }

    for (uint64_t peerId = 0; peerId < arg->rankSize; peerId++) {
        if (peerId == arg->rankId) {
            CHK_RET(ccu::Alloc(&ctx.input[peerId]));
            CHK_RET(ccu::Alloc(&ctx.scratch[peerId]));
            CHK_RET(ccu::Alloc(&ctx.token[peerId]));
        } else {
            CHK_RET(ccu::CreateByChannel(
                arg->channels[channelIdx], RS_INPUT_XN_ID, &ctx.input[peerId]));
            CHK_RET(ccu::CreateByChannel(
                arg->channels[channelIdx], RS_SCRATCH_XN_ID, &ctx.scratch[peerId]));
            CHK_RET(ccu::CreateByChannel(
                arg->channels[channelIdx], RS_TOKEN_XN_ID, &ctx.token[peerId]));
            channelIdx++;
        }
    }

    CHK_RET(ccu::Alloc(&ctx.output));
    CHK_RET(ccu::Alloc(&ctx.currentRankSliceInputOffset));
    CHK_RET(ccu::Alloc(&ctx.currentRankSliceOutputOffset));
    CHK_RET(ccu::Alloc(&ctx.normalSliceSize));
    CHK_RET(ccu::Alloc(&ctx.lastSliceSize));
    CHK_RET(ccu::Alloc(&ctx.inputRepeatStride));
    CHK_RET(ccu::Alloc(&ctx.outputRepeatStride));
    CHK_RET(ccu::Alloc(&ctx.repeatNum));
    CHK_RET(ccu::Alloc(&ctx.flag));

    CHK_RET(ccu::Alloc(&ctx.goSize.addrOffset));
    CHK_RET(ccu::Alloc(&ctx.goSize.loopParam));
    CHK_RET(ccu::Alloc(&ctx.goSize.parallelParam));
    CHK_RET(ccu::Alloc(&ctx.goSize.residual));

    ctx.selfBit = 1 << arg->rankId;
    ctx.allBit  = ((1 << arg->rankSize) - 1) & (~(1 << arg->rankId));

    for (uint64_t rankIdx = 0; rankIdx < arg->rankSize; rankIdx++) {
        CHK_RET(ccu::Alloc(&ctx.scratchMem[rankIdx]));
        if (rankIdx == arg->rankId) {
            CHK_RET(ccu::Alloc(&ctx.myInput));
        } else {
            CHK_RET(ccu::Alloc(&ctx.remoteInput[rankIdx]));
        }
    }

    CHK_RET(ccu::Alloc(&ctx.event));

    ctx.resourceAllocated = false;
    ctx.loopRegistered    = false;

    return HCCL_SUCCESS;
}

// ============================================================================
// LoadArgs
// ============================================================================

static HcclResult LoadArgs(ReduceScatterMesh1DMem2MemContext &ctx)
{
    const auto *arg = ctx.arg;

    CHK_RET(ccu::LoadArg(ctx.input[arg->rankId]));
    CHK_RET(ccu::LoadArg(ctx.output));
    CHK_RET(ccu::LoadArg(ctx.token[arg->rankId]));
    CHK_RET(ccu::LoadArg(ctx.scratch[arg->rankId]));
    CHK_RET(ccu::LoadArg(ctx.currentRankSliceInputOffset));
    CHK_RET(ccu::LoadArg(ctx.currentRankSliceOutputOffset));
    CHK_RET(ccu::LoadArg(ctx.inputRepeatStride));
    CHK_RET(ccu::LoadArg(ctx.outputRepeatStride));
    CHK_RET(ccu::LoadArg(ctx.normalSliceSize));
    CHK_RET(ccu::LoadArg(ctx.lastSliceSize));
    CHK_RET(ccu::LoadArg(ctx.repeatNum));

    CHK_RET(ccu::LoadArg(ctx.goSize.addrOffset));
    CHK_RET(ccu::LoadArg(ctx.goSize.loopParam));
    CHK_RET(ccu::LoadArg(ctx.goSize.parallelParam));
    CHK_RET(ccu::LoadArg(ctx.goSize.residual));

    return HCCL_SUCCESS;
}

static void PreSync(ReduceScatterMesh1DMem2MemContext &ctx)
{
    const auto *arg = ctx.arg;

    for (uint32_t i = 0; i < arg->channelCount; i++) {
        ccu::WriteVariableWithNotify(arg->channels[i], ctx.input[arg->rankId],
            RS_INPUT_XN_ID, RS_CKE_IDX_0, 1 << RS_INPUT_XN_ID);
        ccu::WriteVariableWithNotify(arg->channels[i], ctx.scratch[arg->rankId],
            RS_SCRATCH_XN_ID, RS_CKE_IDX_0, 1 << RS_SCRATCH_XN_ID);
        ccu::WriteVariableWithNotify(arg->channels[i], ctx.token[arg->rankId],
            RS_TOKEN_XN_ID, RS_CKE_IDX_0, 1 << RS_TOKEN_XN_ID);
    }

    uint32_t allBit = (1 << RS_INPUT_XN_ID) | (1 << RS_SCRATCH_XN_ID) | (1 << RS_TOKEN_XN_ID);
    for (uint32_t i = 0; i < arg->channelCount; i++) {
        ccu::NotifyWait(arg->channels[i], RS_CKE_IDX_0, allBit);
    }
}

static void PostSync(ReduceScatterMesh1DMem2MemContext &ctx)
{
    const auto *arg = ctx.arg;

    for (uint32_t i = 0; i < arg->channelCount; i++) {
        ccu::NotifyRecord(arg->channels[i], RS_CKE_IDX_0, 1 << RS_POST_SYNC_ID);
    }
    for (uint32_t i = 0; i < arg->channelCount; i++) {
        ccu::NotifyWait(arg->channels[i], RS_CKE_IDX_0, 1 << RS_POST_SYNC_ID);
    }
}

// ============================================================================
// DoReduceScatter
// ============================================================================

static HcclResult DoReduceScatter(ReduceScatterMesh1DMem2MemContext &ctx)
{
    const auto *arg = ctx.arg;
    uint32_t channelId = 0;

    ccu::LocalAddr myOutput;
    CHK_RET(ccu::Alloc(&myOutput));
    myOutput.addr  = ctx.output;
    myOutput.addr += ctx.currentRankSliceOutputOffset;
    myOutput.token = ctx.token[arg->rankId];

    CcuVariable sliceSize;
    CHK_RET(ccu::Alloc(&sliceSize));
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

    return HCCL_SUCCESS;
}


// ============================================================================
// DoRepeatReduceScatter
// ============================================================================

static HcclResult DoRepeatReduceScatter(ReduceScatterMesh1DMem2MemContext &ctx)
{
    const auto *arg = ctx.arg;

    CcuVariable scratchOffset;
    CHK_RET(ccu::Alloc(&scratchOffset));
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
    CHK_RET(ccu::Alloc(&repeatNumAdd));
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

    return HCCL_SUCCESS;
}

std::string CcuKernelReduceScatterMesh1DMem2Mem::GetLoopBlockTag(std::string loopType, int32_t index)
{
    return loopType + LOOP_BLOCK_TAG + std::to_string(index);
}

// ============================================================================
// CreateReduceLoop
// ============================================================================

static HcclResult CreateReduceLoop(ReduceScatterMesh1DMem2MemContext &ctx)
{
    const auto *arg = ctx.arg;
    const uint32_t size = arg->rankSize;

    constexpr uint32_t LOOP_NUM = 16;
    CHK_RET(AllocGoResource(ctx.moConfig, ctx.moRes, ctx.resourceAllocated, LOOP_NUM));

    if (ctx.loopRegistered) {
        return HCCL_SUCCESS;
    }

    uint32_t expansionNum = GetReduceExpansionNum(arg->reduceOp, arg->dataType, arg->outputDataType);
    uint32_t usedBufNum   = size > expansionNum ? size : expansionNum;

    for (int32_t index = 0; index < 2; index++) {
        CHK_RET(ccu::Alloc(&ctx.loopDst[index]));
        CHK_RET(ccu::Alloc(&ctx.loopSrc[index]));
        for (uint32_t i = 0; i < size; i++) {
            CHK_RET(ccu::Alloc(&ctx.loopScratch[index][i]));
        }
        CHK_RET(ccu::Alloc(&ctx.loopLen[index]));
        CHK_RET(ccu::Alloc(&ctx.loopLenExp[index]));

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
    return HCCL_SUCCESS;
}
 
// ============================================================================
// ReduceLoopGroup
// ============================================================================

static HcclResult ReduceLoopGroup(ReduceScatterMesh1DMem2MemContext &ctx,
    ccu::LocalAddr outDstOrg, ccu::LocalAddr srcOrg,
    ccu::LocalAddr *scratchOrg, uint32_t scratchCount,
    GroupOpSizeVars &goSize)
{
    const auto *arg = ctx.arg;
    const uint32_t size = scratchCount;

    ccu::LocalAddr dst;
    CHK_RET(ccu::Alloc(&dst));
    dst.addr  = outDstOrg.addr;
    dst.token = outDstOrg.token;

    ccu::LocalAddr src;
    CHK_RET(ccu::Alloc(&src));
    src.addr  = srcOrg.addr;
    src.token = srcOrg.token;

    ccu::LocalAddr scratch[RS_MAX_RANK_SIZE];
    for (uint32_t idx = 0; idx < size; idx++) {
        CHK_RET(ccu::Alloc(&scratch[idx]));
        scratch[idx].addr  = scratchOrg[idx].addr;
        scratch[idx].token = scratchOrg[idx].token;
    }

    CHK_RET(CreateReduceLoop(ctx));

    uint32_t expansionNum = GetReduceExpansionNum(arg->reduceOp, arg->dataType, arg->outputDataType);
    CcuVariable sliceSizeExpansion;
    CHK_RET(ccu::Alloc(&sliceSizeExpansion));

    if (expansionNum != 1) {
        CcuVariable tmp;
        CHK_RET(ccu::Alloc(&tmp));
        tmp = GetExpansionParam(expansionNum);
        dst.token = dst.token + tmp;
    }

    // m 部分
    CCU_IF_ONLY(goSize.loopParam != 0) {
        CcuVariable loopParam;
        CHK_RET(ccu::Alloc(&loopParam));
        loopParam = GetLoopParam(0, ctx.moConfig.memSlice * ctx.moConfig.loopCount, 0);
        loopParam = loopParam + goSize.loopParam;

        CcuVariable sliceSize;
        CHK_RET(ccu::Alloc(&sliceSize));
        sliceSize          = ctx.moConfig.memSlice;
        sliceSizeExpansion = ctx.moConfig.memSlice * expansionNum;

        // 绑定 loop0 的外部 LocalAddr 和 Variable
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
        CHK_RET(ccu::Alloc(&paraCfg));
        paraCfg = GetParallelParam(ctx.moConfig.loopCount - 1, 0, 1);

        CcuVariable offsetCfg;
        CHK_RET(ccu::Alloc(&offsetCfg));
        offsetCfg = GetOffsetParam(ctx.moConfig.memSlice, ctx.moConfig.msInterleave, 1);

        CcuLoopGroup group;
        CcuLoopExecutors enginePool;
        CHK_RET(ccu::CreateLoopGroup(&group, &paraCfg, &offsetCfg, enginePool));
        CHK_RET(ccu::AddLoop(group, ctx.reduceLoops[0], &loopParam));
    }

    // n+p 部分
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

        // 绑定 loop0 参数 (p 部分)
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

        // n 部分偏移
        for (uint32_t i = 0; i < size; i++) {
            scratch[i].addr += goSize.residual;
        }
        src.addr += goSize.residual;
        for (uint32_t i = 0; i < expansionNum; i++) {
            dst.addr += goSize.residual;
        }

        CcuVariable sliceSize;
        CHK_RET(ccu::Alloc(&sliceSize));
        sliceSize          = ctx.moConfig.memSlice;
        sliceSizeExpansion = ctx.moConfig.memSlice * expansionNum;

        // 绑定 loop1 参数 (n 部分)
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
        CHK_RET(ccu::Alloc(&loopCfg0));
        loopCfg0 = GetLoopParam(0, 0, 1);

        CcuVariable loopCfg1;
        CHK_RET(ccu::Alloc(&loopCfg1));
        loopCfg1 = GetLoopParam(0, 0, 1);

        CcuVariable offsetCfg;
        CHK_RET(ccu::Alloc(&offsetCfg));
        offsetCfg = GetOffsetParam(ctx.moConfig.memSlice, ctx.moConfig.msInterleave, 1);

        CcuLoopGroup group;
        CcuLoopExecutors enginePool;
        CHK_RET(ccu::CreateLoopGroup(&group, &goSize.parallelParam, &offsetCfg, enginePool));
        CHK_RET(ccu::AddLoop(group, ctx.reduceLoops[0], &loopCfg0));
        CHK_RET(ccu::AddLoop(group, ctx.reduceLoops[1], &loopCfg1));
    }

    return HCCL_SUCCESS;
}

// ============================================================================
// 主入口 Kernel 函数
// ============================================================================

HcclResult CcuReduceScatterMesh1dMem2MemKernel(CcuKernelArg arg)
{
    auto *kernelArg = static_cast<ReduceScatterKernelArg *>(arg);

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

    CHK_RET(InitResource(ctx));
    CHK_RET(LoadArgs(ctx));

    PreSync(ctx);

    CHK_RET(DoRepeatReduceScatter(ctx));

    PostSync(ctx);

    return HCCL_SUCCESS;
}

std::vector<uint64_t> CcuKernelReduceScatterMesh1DMem2Mem::GeneArgs(const CcuTaskArg &arg)
{
    const CcuTaskArgReduceScatterMesh1DMem2Mem *taskArg
        = dynamic_cast<const CcuTaskArgReduceScatterMesh1DMem2Mem *>(&arg);
    uint64_t inputAddr                   = taskArg->inputAddr_;
    uint64_t outputAddr                  = taskArg->outputAddr_;
    uint64_t tokenInfo                   = taskArg->token_;
    uint64_t scratchAddr                 = taskArg->scratchAddr_;
    uint64_t currentRankSliceInputOffset = taskArg->inputSliceStride_ * rankId_;
    uint64_t currentRankSliceOutputOffset= taskArg->outputSliceStride_ * rankId_;
    uint64_t inputRepeatStride           = taskArg->inputRepeatStride_;
    uint64_t outputRepeatStride          = taskArg->outputRepeatStride_;
    uint64_t normalSliceSize             = taskArg->normalSliceSize_;
    uint64_t lastSliceSize               = taskArg->lastSliceSize_;
    uint64_t repeatNum                   = taskArg->repeatNum_;
    auto     GoSize                      = (rankId_ == (rankSize_ - 1)) ? CalGoSize(lastSliceSize) 
                                                                       : CalGoSize(normalSliceSize);

    std::vector<uint64_t> taskArgs = {
        inputAddr,         outputAddr,         tokenInfo,
        scratchAddr,       currentRankSliceInputOffset,
        currentRankSliceOutputOffset,          inputRepeatStride,
        outputRepeatStride, normalSliceSize,   lastSliceSize,
        repeatNum
    };

    HCCL_INFO("[CcuKernelReduceScatterMesh1DMem2Mem] TaskArgs: inputAddr[%llu], outputAddr[%llu], "
               "scratchAddr[%llu], currentRankSliceInputOffset[%llu], currentRankSliceOutputOffset[%llu], "
               "inputRepeatStride[%llu], outputRepeatStride[%llu], "
               "normalSliceSize[%llu], lastSliceSize[%llu], repeatNum[%llu]",
               inputAddr, outputAddr, scratchAddr, currentRankSliceInputOffset, currentRankSliceOutputOffset,
               inputRepeatStride, outputRepeatStride, normalSliceSize, lastSliceSize, UINT64_MAX - repeatNum);
               
    taskArgs.insert(taskArgs.cend(), GoSize.cbegin(), GoSize.cend());
    return taskArgs;
}

} // namespace ops_hccl