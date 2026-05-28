/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "ccu_kernel_reduce_scatter_mesh1d_2die_mem2mem.h"
#include "ccu_kernel_utils.h"

namespace ops_hccl {

constexpr int INPUT_XN_ID   = 0;
constexpr int SCRATCH_XN_ID = 1;
constexpr int TOKEN_XN_ID   = 2;
constexpr int POST_SYNC_ID  = 3;
constexpr int CKE_IDX_0     = 0;
constexpr uint16_t BIT_NUM_PER_CKE = 16;

// 前向声明
static CcuResult ReduceLoopGroup(ReduceScatterMesh1D2DieMem2MemContext &ctx, ccu::LocalAddr outDstOrg,
    ccu::LocalAddr srcOrg, std::vector<ccu::LocalAddr> &scratchOrg);
static CcuResult PairwiseLocalReduce(ReduceScatterMesh1D2DieMem2MemContext &ctx, ccu::LocalAddr myOutput,
    std::vector<ccu::LocalAddr> &inputVec, ccu::Variable sliceSize);

static CcuResult ParseKernelArg(ReduceScatterMesh1D2DieMem2MemContext &ctx, CcuKernelArgReduceScatterMesh1D2DieMem2Mem *kernelArg)
{
    ctx.dataType            = kernelArg->opParam.DataDes.dataType;
    ctx.outputDataType      = kernelArg->opParam.DataDes.outputType;
    if (ctx.outputDataType == HcclDataType::HCCL_DATA_TYPE_RESERVED) {
        ctx.outputDataType = ctx.dataType;
        HCCL_DEBUG("[CcuKernelReduceScatterMesh1D2DieMem2Mem] outputDataType is [INVALID], set outputDataType to[%d]",
            ctx.outputDataType);
    }
    ctx.reduceOp            = kernelArg->opParam.reduceType;
    ctx.subRankGroup        = kernelArg->subRankGroup;
    ctx.isReduceToOutput    = kernelArg->isReduceToOutput;
    return CCU_SUCCESS;
}

static CcuResult InitResource(ReduceScatterMesh1D2DieMem2MemContext &ctx)
{
    const auto *arg = ctx.arg;
    uint16_t channelIdx = 0;

    if (arg->channelCount == 0) {
        HCCL_ERROR("[CcuKernelReduceScatterMesh1D2DieMem2Mem] channels is empty!");
        return CcuResult::CCU_E_INTERNAL;
    }

    // 按照rank号从小到大遍历channels，遇到本rank就填充本地资源，否则依次取远端资源
    ctx.input.resize(arg->rankSize);
    ctx.scratch.resize(arg->rankSize);
    ctx.token.resize(arg->rankSize);

    for (uint64_t rankIdx = 0; rankIdx < arg->rankSize; rankIdx++) {
        if (ctx.subRankGroup[rankIdx] == arg->rankId) {
            // 本地资源
        } else {
            ctx.input[rankIdx] = ccu::GetResByChannel<ccu::Variable>(arg->channels[channelIdx], INPUT_XN_ID);
            ctx.scratch[rankIdx] = ccu::GetResByChannel<ccu::Variable>(arg->channels[channelIdx], SCRATCH_XN_ID);
            ctx.token[rankIdx] = ccu::GetResByChannel<ccu::Variable>(arg->channels[channelIdx], TOKEN_XN_ID);
            channelIdx++;
        }
    }
    if (!ctx.isReduceToOutput) {
        // 需要额外的本地资源做中转
    }
    ctx.myRankIdx = ctx.input.size() - 1;

    ctx.remoteInput.reserve(arg->rankSize);
    ctx.scratchMem.reserve(arg->rankSize);
    for (uint64_t rankIdx = 0; rankIdx < arg->rankSize; rankIdx++) {
        ctx.scratchMem.push_back(ccu::LocalAddr());
        if (ctx.subRankGroup[rankIdx] == arg->rankId) {
            ctx.myInput = ccu::LocalAddr();
            ctx.remoteInput.push_back(ccu::RemoteAddr());
        } else {
            ctx.remoteInput.push_back(ccu::RemoteAddr());
        }
    }

    ctx.moConfig.loopCount = 16;
    ctx.moConfig.msInterleave = REDUCE_2DIE_MS_CNT;
    ctx.moConfig.memSlice = CCU_MS_SIZE;
    ctx.resourceAllocated = false;

    // 创建events数组
    uint32_t eventNum = (arg->rankSize + BIT_NUM_PER_CKE - 1) / BIT_NUM_PER_CKE;
    ctx.events.resize(eventNum);

    return CCU_SUCCESS;
}

static CcuResult LoadArgs(ReduceScatterMesh1D2DieMem2MemContext &ctx)
{
    const auto *arg = ctx.arg;
    uint32_t cnt = 0;

    CCU_CHK_RET(ccu::LoadArg(ctx.input[ctx.myRankIdx], cnt++));
    CCU_CHK_RET(ccu::LoadArg(ctx.output, cnt++));
    CCU_CHK_RET(ccu::LoadArg(ctx.token[ctx.myRankIdx], cnt++));
    CCU_CHK_RET(ccu::LoadArg(ctx.scratch[ctx.myRankIdx], cnt++));
    CCU_CHK_RET(ccu::LoadArg(ctx.currentRankSliceInputOffset, cnt++));
    CCU_CHK_RET(ccu::LoadArg(ctx.inputRepeatStride, cnt++));
    CCU_CHK_RET(ccu::LoadArg(ctx.outputRepeatStride, cnt++));
    CCU_CHK_RET(ccu::LoadArg(ctx.sliceSize, cnt++));
    CCU_CHK_RET(ccu::LoadArg(ctx.repeatNum, cnt++));
    CCU_CHK_RET(ccu::LoadArg(ctx.goSize.addrOffset, cnt++));
    CCU_CHK_RET(ccu::LoadArg(ctx.goSize.loopParam, cnt++));
    CCU_CHK_RET(ccu::LoadArg(ctx.goSize.parallelParam, cnt++));
    CCU_CHK_RET(ccu::LoadArg(ctx.goSize.residual, cnt++));

    return CCU_SUCCESS;
}

static CcuResult PreSync(ReduceScatterMesh1D2DieMem2MemContext &ctx)
{
    const auto *arg = ctx.arg;

    for (uint32_t i = 0; i < arg->channelCount; i++) {
        CCU_CHK_RET(ccu::WriteVariableWithNotify(arg->channels[i], ctx.input[ctx.myRankIdx],
            INPUT_XN_ID, CKE_IDX_0, 1 << INPUT_XN_ID));
        CCU_CHK_RET(ccu::WriteVariableWithNotify(arg->channels[i], ctx.scratch[ctx.myRankIdx],
            SCRATCH_XN_ID, CKE_IDX_0, 1 << SCRATCH_XN_ID));
        CCU_CHK_RET(ccu::WriteVariableWithNotify(arg->channels[i], ctx.token[ctx.myRankIdx],
            TOKEN_XN_ID, CKE_IDX_0, 1 << TOKEN_XN_ID));
    }

    uint32_t allBit = (1 << INPUT_XN_ID) | (1 << SCRATCH_XN_ID) | (1 << TOKEN_XN_ID);
    for (uint32_t i = 0; i < arg->channelCount; i++) {
        CCU_CHK_RET(ccu::NotifyWait(arg->channels[i], CKE_IDX_0, allBit));
    }
    return CCU_SUCCESS;
}

static CcuResult PostSync(ReduceScatterMesh1D2DieMem2MemContext &ctx)
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

static CcuResult RmtReduce(ReduceScatterMesh1D2DieMem2MemContext &ctx)
{
    const auto *arg = ctx.arg;

    // 计算scratchOffset
    ccu::Variable scratchOffset;
    scratchOffset = 0;
    std::vector<ccu::Variable> scratchOffsetVec;

    for (uint32_t gRankIdx = 0; gRankIdx < arg->gRankSize; gRankIdx++) {
        ccu::Variable scratchOffTmp;
        scratchOffTmp = scratchOffset;
        scratchOffsetVec.push_back(scratchOffTmp);
        scratchOffset += ctx.sliceSize;
    }

    ctx.outputTmp.addr = ctx.scratch[ctx.myRankIdx];
    ctx.outputTmp.addr += scratchOffsetVec[arg->gRankSize / 2];
    ctx.outputTmp.token = ctx.token[ctx.myRankIdx];

    for (uint64_t rankIdx = 0; rankIdx < arg->rankSize; rankIdx++) {
        if (ctx.subRankGroup[rankIdx] != arg->rankId) {
            ctx.remoteInput[rankIdx].addr = ctx.input[rankIdx];
            ctx.remoteInput[rankIdx].addr += ctx.currentRankSliceInputOffset;
            ctx.remoteInput[rankIdx].token = ctx.token[rankIdx];
        }
        ctx.scratchMem[rankIdx].addr = ctx.scratch[ctx.myRankIdx];
        ctx.scratchMem[rankIdx].addr += scratchOffsetVec[ctx.subRankGroup[rankIdx]];
        ctx.scratchMem[rankIdx].token = ctx.token[ctx.myRankIdx];
    }

    ccu::Variable repeatNumAdd;
    repeatNumAdd = 1;
    ctx.flag = 0;

    CCU_WHILE(ctx.repeatNum != UINT64_MAX) {
        ctx.repeatNum += repeatNumAdd;
        CCU_IF(ctx.flag == 1) {
            for (uint64_t rankIdx = 0; rankIdx < arg->rankSize; rankIdx++) {
                if (ctx.subRankGroup[rankIdx] == arg->rankId) {
                    ctx.myInput.addr += ctx.inputRepeatStride;
                } else {
                    ctx.remoteInput[rankIdx].addr += ctx.inputRepeatStride;
                }
            }
            ctx.output += ctx.outputRepeatStride;
        }
        CCU_IF(ctx.sliceSize != 0) {
            CCU_CHK_RET(DoReduceScatter_2Die(ctx));
        }
        ctx.flag = 1;
    }

    return CCU_SUCCESS;
}

static CcuResult DoReduceScatter_2Die(ReduceScatterMesh1D2DieMem2MemContext &ctx)
{
    const auto *arg = ctx.arg;
    uint32_t channelId = 0;

    ccu::LocalAddr myOutput;
    myOutput.addr   = ctx.output;
    myOutput.token  = ctx.token[ctx.myRankIdx];

    if (arg->rankId < arg->rankSize) {
        if (!ctx.isReduceToOutput) {
            myOutput.addr = ctx.outputTmp.addr;
        }
    } else {
        if (ctx.isReduceToOutput) {
            myOutput.addr = ctx.outputTmp.addr;
        }
    }

    const uint32_t eventNum = (arg->rankSize + BIT_NUM_PER_CKE - 1) / BIT_NUM_PER_CKE;

    for (uint64_t rankIdx = 0; rankIdx < arg->rankSize; rankIdx++) {
        const uint16_t eventIdx = rankIdx / BIT_NUM_PER_CKE;
        const uint16_t rankMask = 1 << (rankIdx % BIT_NUM_PER_CKE);

        if (ctx.subRankGroup[rankIdx] == arg->rankId) {
            CCU_CHK_RET(ccu::EventRecord(ctx.events[eventIdx], rankMask));
            ctx.scratchMem[rankIdx].addr = ctx.input[ctx.myRankIdx];
            ctx.scratchMem[rankIdx].addr += ctx.currentRankSliceInputOffset;
            ctx.scratchMem[rankIdx].token = ctx.token[ctx.myRankIdx];
        } else {
            CCU_CHK_RET(ccu::Read(arg->channels[channelId], ctx.scratchMem[rankIdx],
                ctx.remoteInput[rankIdx], ctx.sliceSize, ctx.events[eventIdx], rankMask));
            channelId++;
        }
    }

    // 等读完所有对端
    for (uint32_t i = 0; i < eventNum; i++) {
        uint32_t sigNum;
        if (arg->rankSize % BIT_NUM_PER_CKE != 0 && i == (eventNum - 1)) {
            sigNum = arg->rankSize % BIT_NUM_PER_CKE;
        } else {
            sigNum = BIT_NUM_PER_CKE;
        }
        uint32_t allBit = (1 << sigNum) - 1;
        CCU_CHK_RET(ccu::EventWait(ctx.events[i], allBit));
    }

    // 做reduce
    if (arg->rankSize <= REDUCE_SCATTER_2DIE_GROUP_REDUCE_MAX_PIECE_CNT) {
        ReduceLoopGroup(ctx, myOutput, ctx.myInput, ctx.scratchMem);
    } else {
        PairwiseLocalReduce(ctx, myOutput, ctx.scratchMem, ctx.sliceSize);
    }

    return CCU_SUCCESS;
}

static CcuResult PairwiseLocalReduce(ReduceScatterMesh1D2DieMem2MemContext &ctx, ccu::LocalAddr myOutput,
    std::vector<ccu::LocalAddr> &inputVec, ccu::Variable sliceSize)
{
    const auto *arg = ctx.arg;
    ccu::Variable len;

    uint32_t remainPieces = arg->rankSize;
    while (remainPieces > 1) {
        uint32_t reducePieces = remainPieces / 2;
        uint32_t srcIdx = remainPieces - reducePieces;
        uint32_t dstIdx = 0;

        len = sliceSize;
        for (uint32_t i = 0; i < reducePieces - 1; i++) {
            len += sliceSize;
        }

        ccu::LocalReduce(inputVec[dstIdx], inputVec[srcIdx], len, ctx.dataType, ctx.reduceOp, ctx.events[0]);
        ccu::EventWait(ctx.events[0]);

        remainPieces -= reducePieces;
    }

    ccu::LocalCopy(myOutput, inputVec[0], sliceSize, ctx.events[0]);
    ccu::EventWait(ctx.events[0]);

    return CCU_SUCCESS;
}

static CcuResult CreateReduceLoop(ReduceScatterMesh1D2DieMem2MemContext &ctx)
{
    AllocGoResource(ctx.moConfig, ctx.moRes, ctx.resourceAllocated, 16);

    if (ctx.IsLoopEntityRegistered("reduceScatter2DieLocalReduce")) {
        return CCU_SUCCESS;
    }
    ctx.CreateLoopEntity("reduceScatter2DieLocalReduce");
    auto &loops = ctx.loopMap["reduceScatter2DieLocalReduce"];

    const auto *arg = ctx.arg;
    uint32_t size = arg->rankSize;
    uint32_t expansionNum = GetReduceExpansionNum(arg->opParam.reduceType, ctx.dataType, ctx.outputDataType);
    uint32_t usedBufNum   = size > expansionNum ? size : expansionNum;

    for (int32_t index = 0; index < 2; index++) {
        ctx.loopScratch[index].resize(size);

        uint32_t bufBase = index * ctx.moConfig.msInterleave;
        ccu::Event loopEvt = ctx.moRes.completedEvent[index];

        loops.body[index].reset(new ccu::Func(
            [&ctx, index, bufBase, loopEvt, size, expansionNum, usedBufNum]() {
            // Step 1: 将数据copy到ccuBuf
            for (uint32_t i = 0; i < size; i++) {
                if (i == ctx.myRankIdx) {
                    ccu::LocalCopy(ctx.moRes.ccuBuf[bufBase + i], ctx.loopSrc[index], ctx.loopLen[index], loopEvt, 1 << i);
                } else {
                    ccu::LocalCopy(ctx.moRes.ccuBuf[bufBase + i], ctx.loopScratch[index][i], ctx.loopLen[index], loopEvt, 1 << i);
                }
            }
            ccu::EventWait(loopEvt, (1 << size) - 1);

            // Step 2: LocalReduce
            if (size > 1) {
                ccu::LocalReduce(&ctx.moRes.ccuBuf[bufBase], size, ctx.dataType, ctx.outputDataType, ctx.reduceOp,
                    ctx.loopLen[index], loopEvt, 1);
                ccu::EventWait(loopEvt, 1);
            }

            // Step 3: Copy结果到dst
            ccu::LocalCopy(ctx.loopDst[index], ctx.moRes.ccuBuf[bufBase], ctx.loopLenExp[index], loopEvt, 1);
            ccu::EventWait(loopEvt, 1);
        }));

        loops.loops[index].reset(
            new ccu::Loop(loops.loopParam[index], *loops.body[index]));
    }

    return CCU_SUCCESS;
}

static CcuResult ReduceLoopGroup(ReduceScatterMesh1D2DieMem2MemContext &ctx, ccu::LocalAddr outDstOrg,
    ccu::LocalAddr srcOrg, std::vector<ccu::LocalAddr> &scratchOrg)
{
    const auto *arg = ctx.arg;
    const uint32_t size = scratchOrg.size();

    ccu::LocalAddr dst;
    dst.addr  = outDstOrg.addr;
    dst.token = outDstOrg.token;

    ccu::LocalAddr src;
    src.addr  = srcOrg.addr;
    src.token = srcOrg.token;

    std::vector<ccu::LocalAddr> scratch;
    for (uint32_t idx = 0; idx < size; idx++) {
        ccu::LocalAddr scratchAddr;
        scratchAddr.addr  = scratchOrg[idx].addr;
        scratchAddr.token = scratchOrg[idx].token;
        scratch.push_back(scratchAddr);
    }

    CCU_CHK_RET(CreateReduceLoop(ctx));
    auto &loops = ctx.loopMap["reduceScatter2DieLocalReduce"];

    uint32_t expansionNum = GetReduceExpansionNum(arg->opParam.reduceType, ctx.dataType, ctx.outputDataType);
    ccu::Variable sliceSizeExpansion;

    if (expansionNum != 1) {
        ccu::Variable tmp;
        tmp = GetExpansionParam(expansionNum);
        dst.token = dst.token + tmp;
    }

    // 第一个loopgroup，处理m部分数据
    CCU_IF(ctx.goSize.loopParam != 0) {
        ccu::Variable loopParam;
        loopParam = GetLoopParam(0, ctx.moConfig.memSlice * ctx.moConfig.loopCount, 0);
        loopParam = loopParam + ctx.goSize.loopParam;

        ccu::Variable sliceSize;
        sliceSize          = ctx.moConfig.memSlice;
        sliceSizeExpansion = ctx.moConfig.memSlice * expansionNum;

        for (uint32_t i = 0; i < size; i++) {
            ctx.loopScratch[0][i].addr  = scratch[i].addr;
            ctx.loopScratch[0][i].token = scratch[i].token;
        }
        ctx.loopSrc[0].addr  = src.addr;
        ctx.loopSrc[0].token = src.token;
        ctx.loopDst[0].addr  = dst.addr;
        ctx.loopDst[0].token = dst.token;
        ctx.loopLen[0]       = sliceSize;
        ctx.loopLenExp[0]    = sliceSizeExpansion;

        ccu::Variable paraCfg;
        paraCfg = GetParallelParam(ctx.moConfig.loopCount - 1, 0, 1);

        ccu::Variable offsetCfg;
        offsetCfg = GetOffsetParam(ctx.moConfig.memSlice, ctx.moConfig.msInterleave, 1);

        loops.loopParam[0] = loopParam;
        std::vector<ccu::Loop> grpLoops{ *loops.loops[0] };
        ccu::LoopGroup group(paraCfg, offsetCfg, 1, grpLoops);
    }

    // 第二个loopgroup，处理n和p部分数据
    CCU_IF(ctx.goSize.parallelParam != 0) {
        for (uint32_t i = 0; i < size; i++) {
            scratch[i].addr += ctx.goSize.addrOffset;
        }
        src.addr += ctx.goSize.addrOffset;
        for (uint32_t i = 0; i < expansionNum; i++) {
            dst.addr += ctx.goSize.addrOffset;
        }

        sliceSizeExpansion = 0;
        for (uint32_t i = 0; i < expansionNum; i++) {
            sliceSizeExpansion = sliceSizeExpansion + ctx.goSize.residual;
        }

        // 绑定loop0参数 (p部分)
        for (uint32_t i = 0; i < size; i++) {
            ctx.loopScratch[0][i].addr  = scratch[i].addr;
            ctx.loopScratch[0][i].token = scratch[i].token;
        }
        ctx.loopSrc[0].addr  = src.addr;
        ctx.loopSrc[0].token = src.token;
        ctx.loopDst[0].addr  = dst.addr;
        ctx.loopDst[0].token = dst.token;
        ctx.loopLen[0]    = ctx.goSize.residual;
        ctx.loopLenExp[0] = sliceSizeExpansion;

        for (uint32_t i = 0; i < size; i++) {
            scratch[i].addr += ctx.goSize.residual;
        }
        src.addr += ctx.goSize.residual;
        for (uint32_t i = 0; i < expansionNum; i++) {
            dst.addr += ctx.goSize.residual;
        }

        ccu::Variable sliceSize;
        sliceSize          = ctx.moConfig.memSlice;
        sliceSizeExpansion = ctx.moConfig.memSlice * expansionNum;

        // 绑定loop1参数 (n部分)
        for (uint32_t i = 0; i < size; i++) {
            ctx.loopScratch[1][i].addr  = scratch[i].addr;
            ctx.loopScratch[1][i].token = scratch[i].token;
        }
        ctx.loopSrc[1].addr  = src.addr;
        ctx.loopSrc[1].token = src.token;
        ctx.loopDst[1].addr  = dst.addr;
        ctx.loopDst[1].token = dst.token;
        ctx.loopLen[1]    = sliceSize;
        ctx.loopLenExp[1] = sliceSizeExpansion;

        ccu::Variable loopCfg0;
        loopCfg0 = GetLoopParam(0, 0, 1);

        ccu::Variable loopCfg1;
        loopCfg1 = GetLoopParam(0, 0, 1);

        ccu::Variable offsetCfg;
        offsetCfg = GetOffsetParam(ctx.moConfig.memSlice, ctx.moConfig.msInterleave, 1);

        loops.loopParam[0] = loopCfg0;
        loops.loopParam[1] = loopCfg1;
        std::vector<ccu::Loop> grpLoops{ *loops.loops[0], *loops.loops[1] };
        ccu::LoopGroup group(ctx.goSize.parallelParam, offsetCfg, 2, grpLoops);
    }

    return CCU_SUCCESS;
}

CcuResult CcuReduceScatterMesh1D2DieMem2MemKernel(CcuKernelArg arg)
{
    auto *kernelArg = static_cast<CcuKernelArgReduceScatterMesh1D2DieMem2Mem *>(arg);

    ReduceScatterMesh1D2DieMem2MemContext ctx;
    ctx.resourceAllocated = false;
    ctx.moConfig.msInterleave = 0;
    ctx.moConfig.loopCount = 0;
    ctx.moConfig.memSlice = 0;
    ctx.moRes.eventCount = 0;
    ctx.moRes.bufCount = 0;
    ctx.enginePool = 0;

    HCCL_INFO("[CcuKernelReduceScatterMesh1D2DieMem2Mem] Algorithm start");
    CCU_CHK_RET(ParseKernelArg(ctx, kernelArg));
    CCU_CHK_RET(InitResource(ctx));
    CCU_CHK_RET(LoadArgs(ctx));

    CCU_CHK_RET(PreSync(ctx));
    CCU_CHK_RET(RmtReduce(ctx));
    CCU_CHK_RET(PostSync(ctx));

    HCCL_INFO("[CcuKernelReduceScatterMesh1D2DieMem2Mem] Algorithm end");
    return CCU_SUCCESS;
}

} // namespace ops_hccl