/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "ccu_kernel_all_reduce_mesh1d_mem2mem_2die_oneshot.h"
#include "ccu_control_api.h"

namespace ops_hccl {

constexpr int INPUT_XN_ID        = 0;
constexpr int TOKEN_XN_ID        = 1;
constexpr int POST_SYNC_ID       = 2;

constexpr int CKE_IDX_0          = 0;
constexpr int CKE_IDX_1          = 1;

constexpr int MISSION_NUM        = 2;

constexpr uint32_t LOCAL_REDUCE_LOOP_COUNT  = 8;
constexpr uint32_t LOCAL_REDUCE_MS_PER_LOOP = 8;
const std::string LOCAL_REDUCE_LOOP_BLOCK_TAG{"_local_reduce_loop_"};

struct LocalReduceVar {
    ccu::LocalAddr loopDst[2];
    std::vector<ccu::LocalAddr> loopSrc[2];
    ccu::Variable loopLen[2];
    ccu::Variable loopLenExp[2];
};

static CcuResult ParseKernelArg(AllReduceMesh1DMem2Mem2DieOneShotContext &ctx)
{
    const auto *arg = ctx.arg;
    
    ctx.rankSize = arg->rankSize;
    ctx.rankId = arg->rankId;
    ctx.rmtReduceWithMyRank = arg->rmtReduceWithMyRank;
    ctx.rmtReduceRankNum = arg->channels.size() + (ctx.rmtReduceWithMyRank ? 1 : 0);

    ctx.dataType = arg->opParam.DataDes.dataType;
    ctx.outputDataType = arg->opParam.DataDes.outputType;
    if (ctx.outputDataType == HcclDataType::HCCL_DATA_TYPE_RESERVED) {
        ctx.outputDataType = ctx.dataType;
        HCCL_DEBUG("[CcuKernelAllReduceMesh1DMem2Mem2DieOneShot] outputDataType is [INVALID], set outputDataType to[%d]",
            ctx.outputDataType);
    }
    ctx.reduceOp = arg->opParam.reduceType;

    ctx.missionSyncMybit = 1 << (ctx.rmtReduceWithMyRank ? 1 : 0);
    ctx.missionSyncWaitBit = 1 << (!ctx.rmtReduceWithMyRank ? 1 : 0);

    HCCL_INFO("[CcuKernelAllReduceMesh1DMem2Mem2DieOneShot] Init, KernelArgs are rankId[%u], rankSize[%u], dataType[%d], "
        "outputDataType[%d], reduceOp[%d], rmtReduceWithMyRank[%d]",
        ctx.rankId, ctx.rankSize, ctx.dataType, ctx.outputDataType, ctx.reduceOp, ctx.rmtReduceWithMyRank);
    
    return CCU_SUCCESS;
}

static CcuResult InitResource(AllReduceMesh1DMem2Mem2DieOneShotContext &ctx)
{
    const auto *arg = ctx.arg;

    ctx.myInput = ccu::GetResByChannel<ccu::Variable>(arg->channels[0], INPUT_XN_ID);
    ctx.myOutput = ccu::GetResByChannel<ccu::Variable>(arg->channels[0], 0);
    ctx.myScratch = ccu::GetResByChannel<ccu::Variable>(arg->channels[0], 2);
    ctx.myToken = ccu::GetResByChannel<ccu::Variable>(arg->channels[0], TOKEN_XN_ID);

    ctx.peerInput.resize(arg->channels.size());
    ctx.peerToken.resize(arg->channels.size());
    for (size_t i = 0; i < arg->channels.size(); i++) {
        ctx.peerInput[i] = ccu::GetResByChannel<ccu::Variable>(arg->channels[i], INPUT_XN_ID);
        ctx.peerToken[i] = ccu::GetResByChannel<ccu::Variable>(arg->channels[i], TOKEN_XN_ID);
    }

    ctx.event = ccu::CreateEvent(1);

    return CCU_SUCCESS;
}

static CcuResult LoadArgs(AllReduceMesh1DMem2Mem2DieOneShotContext &ctx)
{
    uint32_t argId = 0;

    CCU_CHK_RET(ccu::LoadArg(ctx.myInput, argId++));
    CCU_CHK_RET(ccu::LoadArg(ctx.myOutput, argId++));
    CCU_CHK_RET(ccu::LoadArg(ctx.myScratch, argId++));
    CCU_CHK_RET(ccu::LoadArg(ctx.myToken, argId++));

    CCU_CHK_RET(ccu::LoadArg(ctx.normalSliceSize, argId++));

    CCU_CHK_RET(ccu::LoadArg(ctx.scratchBaseOffset0, argId++));
    CCU_CHK_RET(ccu::LoadArg(ctx.scratchBaseOffset1, argId++));

    CCU_CHK_RET(ccu::LoadArg(ctx.localReduceSliceOffset0, argId++));
    CCU_CHK_RET(ccu::LoadArg(ctx.localReduceSliceOffset1, argId++));

    CCU_CHK_RET(ccu::LoadArg(ctx.localReduceGoSize.addrOffset, argId++));
    CCU_CHK_RET(ccu::LoadArg(ctx.localReduceGoSize.loopParam, argId++));
    CCU_CHK_RET(ccu::LoadArg(ctx.localReduceGoSize.parallelParam, argId++));
    CCU_CHK_RET(ccu::LoadArg(ctx.localReduceGoSize.residual, argId++));

    CCU_CHK_RET(ccu::LoadArg(ctx.localReduceGoSize0.addrOffset, argId++));
    CCU_CHK_RET(ccu::LoadArg(ctx.localReduceGoSize0.loopParam, argId++));
    CCU_CHK_RET(ccu::LoadArg(ctx.localReduceGoSize0.parallelParam, argId++));
    CCU_CHK_RET(ccu::LoadArg(ctx.localReduceGoSize0.residual, argId++));

    CCU_CHK_RET(ccu::LoadArg(ctx.localReduceGoSize1.addrOffset, argId++));
    CCU_CHK_RET(ccu::LoadArg(ctx.localReduceGoSize1.loopParam, argId++));
    CCU_CHK_RET(ccu::LoadArg(ctx.localReduceGoSize1.parallelParam, argId++));
    CCU_CHK_RET(ccu::LoadArg(ctx.localReduceGoSize1.residual, argId++));

    return CCU_SUCCESS;
}

static CcuResult PreSync(AllReduceMesh1DMem2Mem2DieOneShotContext &ctx)
{
    const auto *arg = ctx.arg;

    for (auto channel : arg->channels) {
        ccu::WriteVariableWithNotify(channel, ctx.myInput, INPUT_XN_ID, CKE_IDX_0, 1 << INPUT_XN_ID);
        ccu::WriteVariableWithNotify(channel, ctx.myToken, TOKEN_XN_ID, CKE_IDX_0, 1 << TOKEN_XN_ID);
    }

    uint32_t allBit = 1 << INPUT_XN_ID | 1 << TOKEN_XN_ID;
    for (auto channel : arg->channels) {
        ccu::NotifyWait(channel, CKE_IDX_0, allBit);
    }
    return CCU_SUCCESS;
}

static CcuResult PostSync(AllReduceMesh1DMem2Mem2DieOneShotContext &ctx)
{
    const auto *arg = ctx.arg;

    for (auto &ch : arg->channels) {
        ccu::NotifyRecord(ch, CKE_IDX_0, 1 << POST_SYNC_ID);
    }
    for (auto &ch : arg->channels) {
        ccu::NotifyWait(ch, CKE_IDX_0, 1 << POST_SYNC_ID);
    }
    return CCU_SUCCESS;
}

static CcuResult CreateReduceLoop(AllReduceMesh1DMem2Mem2DieOneShotContext &ctx,
    LocalReduceVar &var, uint32_t size, HcclDataType dataType, HcclDataType outputDataType,
    HcclReduceOp opType, const std::string &loopName)
{
    constexpr uint32_t LOOP_NUM = 16;
    AllocGoResource(ctx.moConfig, ctx.moRes, ctx.resourceAllocated, LOOP_NUM);

    const std::string loopType = "all_reduce_mesh1d_mem2mem_2die_oneshot_local_reduce";

    if (ctx.IsLoopEntityRegistered(loopType)) {
        return CCU_SUCCESS;
    }
    ctx.CreateLoopEntity(loopType);
    auto &loops = ctx.loopMap[loopType];

    uint32_t expansionNum = GetReduceExpansionNum(opType, dataType, outputDataType);
    uint32_t usedBufNum = size > expansionNum ? size : expansionNum;

    for (int32_t index = 0; index < 2; index++) {
        var.loopSrc[index].resize(size);
        uint32_t bufBase = index * ctx.moConfig.msInterleave;
        ccu::Event e = ctx.moRes.completedEvent[index];

        loops.body[index].reset(new ccu::Func(
            [&ctx, index, bufBase, e, size, &var, dataType, outputDataType, opType]() {
            for (uint32_t i = 0; i < size; i++) {
                ccu::LocalCopy(ctx.moRes.ccuBuf[bufBase + i], var.loopSrc[index][i], var.loopLen[index], e, 1 << i);
            }
            ccu::EventWait(e, (1 << size) - 1);

            if (size > 1) {
                ccu::LocalReduce(&ctx.moRes.ccuBuf[bufBase], size, dataType, outputDataType, opType, var.loopLen[index], e, 1);
                ccu::EventWait(e, 1);
            }

            ccu::LocalCopy(var.loopDst[index], ctx.moRes.ccuBuf[bufBase], var.loopLenExp[index], e, 1);
            ccu::EventWait(e, 1);
        }));

        loops.loops[index].reset(new ccu::Loop(loops.loopParam[index], *loops.body[index]));
    }

    return CCU_SUCCESS;
}

static CcuResult ReduceLoopGroup(AllReduceMesh1DMem2Mem2DieOneShotContext &ctx,
    ccu::LocalAddr &outDstOrg, std::vector<ccu::LocalAddr> &srcOrg,
    GroupOpSize goSize, HcclDataType dataType, HcclDataType outputDataType,
    HcclReduceOp opType, const std::string &loopName)
{
    const uint32_t size = srcOrg.size();

    ccu::LocalAddr dst;
    dst = outDstOrg;

    std::vector<ccu::LocalAddr> src;
    src.resize(size);
    for (uint32_t idx = 0; idx < size; idx++) {
        src[idx] = srcOrg[idx];
    }

    LocalReduceVar var;
    CCU_CHK_RET(CreateReduceLoop(ctx, var, size, dataType, outputDataType, opType, loopName));

    const std::string loopType = "all_reduce_mesh1d_mem2mem_2die_oneshot_local_reduce";
    auto &loops = ctx.loopMap[loopType];

    uint32_t expansionNum = GetReduceExpansionNum(opType, dataType, outputDataType);
    ccu::Variable sliceSizeExpansion;

    if (expansionNum != 1) {
        ccu::Variable tmp;
        tmp = GetExpansionParam(expansionNum);
        dst.token = dst.token + tmp;
    }

    CCU_IF(goSize.loopParam != 0) {
        ccu::Variable loopParam;
        loopParam = GetLoopParam(0, ctx.moConfig.memSlice * ctx.moConfig.loopCount, 0);
        loopParam = loopParam + goSize.loopParam;

        ccu::Variable sliceSize;
        sliceSize = ctx.moConfig.memSlice;
        sliceSizeExpansion = ctx.moConfig.memSlice * expansionNum;

        for (uint32_t i = 0; i < size; ++i) {
            var.loopSrc[0][i] = src[i];
        }
        var.loopDst[0] = dst;
        var.loopLen[0] = sliceSize;
        var.loopLenExp[0] = sliceSizeExpansion;

        ccu::Variable paraCfg;
        paraCfg = GetParallelParam(ctx.moConfig.loopCount - 1, 0, 1);
        ccu::Variable offsetCfg;
        offsetCfg = GetOffsetParam(ctx.moConfig.memSlice, ctx.moConfig.msInterleave, 1);

        loops.loopParam[0] = loopParam;
        std::vector<ccu::Loop> grpLoops{ *loops.loops[0] };
        ccu::LoopGroup group(paraCfg, offsetCfg, 1, grpLoops);
    }

    CCU_IF(goSize.parallelParam != 0) {
        for (uint32_t i = 0; i < size; i++) {
            src[i].addr = src[i].addr + goSize.addrOffset;
        }
        for (uint32_t i = 0; i < expansionNum; i++) {
            dst.addr = dst.addr + goSize.addrOffset;
        }

        sliceSizeExpansion = 0;
        for (uint32_t i = 0; i < expansionNum; i++) {
            sliceSizeExpansion = sliceSizeExpansion + goSize.residual;
        }

        for (uint32_t i = 0; i < size; ++i) {
            var.loopSrc[0][i] = src[i];
        }
        var.loopDst[0] = dst;
        var.loopLen[0] = goSize.residual;
        var.loopLenExp[0] = sliceSizeExpansion;

        for (uint32_t i = 0; i < size; i++) {
            src[i].addr = src[i].addr + goSize.residual;
        }
        for (uint32_t i = 0; i < expansionNum; i++) {
            dst.addr = dst.addr + goSize.residual;
        }

        ccu::Variable sliceSize;
        sliceSize = ctx.moConfig.memSlice;
        sliceSizeExpansion = ctx.moConfig.memSlice * expansionNum;

        for (uint32_t i = 0; i < size; ++i) {
            var.loopSrc[1][i] = src[i];
        }
        var.loopDst[1] = dst;
        var.loopLen[1] = sliceSize;
        var.loopLenExp[1] = sliceSizeExpansion;

        ccu::Variable loopCfg0;
        loopCfg0 = GetLoopParam(0, 0, 1);
        ccu::Variable loopCfg1;
        loopCfg1 = GetLoopParam(0, 0, 1);
        ccu::Variable offsetCfg;
        offsetCfg = GetOffsetParam(ctx.moConfig.memSlice, ctx.moConfig.msInterleave, 1);

        loops.loopParam[0] = loopCfg0;
        loops.loopParam[1] = loopCfg1;
        std::vector<ccu::Loop> grpLoops{ *loops.loops[0], *loops.loops[1] };
        ccu::LoopGroup group(goSize.parallelParam, offsetCfg, 2, grpLoops);
    }

    return CCU_SUCCESS;
}

static CcuResult RmtReduce(AllReduceMesh1DMem2Mem2DieOneShotContext &ctx)
{
    const auto *arg = ctx.arg;

    std::vector<ccu::RemoteAddr> remoteInput;
    uint32_t channelIdx = 0;
    remoteInput.reserve(ctx.rmtReduceRankNum);
    for (uint32_t rankIdx = 0; rankIdx < ctx.rmtReduceRankNum; rankIdx++) {
        if (ctx.rmtReduceWithMyRank && rankIdx == ctx.rankId % ctx.rmtReduceRankNum) {
            remoteInput.push_back({});
        } else {
            ccu::RemoteAddr addr;
            addr.addr = ctx.peerInput[channelIdx];
            addr.token = ctx.peerToken[channelIdx];
            remoteInput.push_back(addr);
            channelIdx++;
        }
    }

    std::vector<ccu::LocalAddr> scratchDst;
    scratchDst.reserve(ctx.rmtReduceRankNum);
    ccu::Variable scratchOffset;
    scratchOffset = 0;
    for (uint32_t rankIdx = 0; rankIdx < ctx.rmtReduceRankNum; rankIdx++) {
        ccu::LocalAddr addr;
        addr.addr = ctx.myScratch;
        addr.addr = addr.addr + (ctx.rmtReduceWithMyRank ? ctx.scratchBaseOffset0 : ctx.scratchBaseOffset1);
        addr.addr = addr.addr + scratchOffset;
        addr.token = ctx.myToken;
        scratchDst.push_back(addr);
        scratchOffset = scratchOffset + ctx.normalSliceSize;
    }

    uint32_t channelId = 0;
    for (uint32_t rankIdx = 0; rankIdx < ctx.rmtReduceRankNum; rankIdx++) {
        if (ctx.rmtReduceWithMyRank && rankIdx == ctx.rankId % ctx.rmtReduceRankNum) {
            ccu::EventRecord(ctx.event, 1 << rankIdx);
        } else {
            ccu::Read(arg->channels[channelId], scratchDst[rankIdx], remoteInput[rankIdx], ctx.normalSliceSize, ctx.event, 1 << rankIdx);
            channelId++;
        }
    }

    ccu::EventWait(ctx.event, (1 << ctx.rmtReduceRankNum) - 1);

    ccu::LocalAddr output;
    output.token = ctx.myToken;
    output.addr = ctx.myOutput;

    if (ctx.rmtReduceWithMyRank) {
        scratchDst[ctx.rankId % ctx.rmtReduceRankNum].addr = ctx.myInput;
        scratchDst[ctx.rankId % ctx.rmtReduceRankNum].token = ctx.myToken;
        CCU_CHK_RET(ReduceLoopGroup(ctx, output, scratchDst, ctx.localReduceGoSize, ctx.dataType, ctx.outputDataType, ctx.reduceOp, "all_reduce_2die_localreduce1_"));
    } else {
        CCU_CHK_RET(ReduceLoopGroup(ctx, scratchDst[0], scratchDst, ctx.localReduceGoSize, ctx.dataType, ctx.outputDataType, ctx.reduceOp, "all_reduce_2die_localreduce1_"));
    }

    return CCU_SUCCESS;
}

static CcuResult DoLocalReduce(AllReduceMesh1DMem2Mem2DieOneShotContext &ctx)
{
    std::vector<ccu::LocalAddr> src;
    src.reserve(MISSION_NUM);
    for (uint32_t i = 0; i < MISSION_NUM; i++) {
        ccu::LocalAddr addr;
        addr.token = ctx.myToken;
        addr.addr = (i == 0 ? ctx.myOutput : ctx.myScratch);
        addr.addr = addr.addr + (i == 0 ? ctx.scratchBaseOffset0 : ctx.scratchBaseOffset1);
        addr.addr = addr.addr + (ctx.rmtReduceWithMyRank ? ctx.localReduceSliceOffset0 : ctx.localReduceSliceOffset1);
        src.push_back(addr);
    }

    ccu::LocalAddr dst;
    dst.token = ctx.myToken;
    dst.addr = ctx.myOutput;
    dst.addr = dst.addr + (ctx.rmtReduceWithMyRank ? ctx.localReduceSliceOffset0 : ctx.localReduceSliceOffset1);

    if (ctx.rmtReduceWithMyRank) {
        CCU_CHK_RET(ReduceLoopGroup(ctx, dst, src, ctx.localReduceGoSize0, ctx.dataType, ctx.outputDataType, ctx.reduceOp, "all_reduce_2die_localreduce2_"));
    } else {
        CCU_CHK_RET(ReduceLoopGroup(ctx, dst, src, ctx.localReduceGoSize1, ctx.dataType, ctx.outputDataType, ctx.reduceOp, "all_reduce_2die_localreduce2_"));
    }

    return CCU_SUCCESS;
}

static CcuResult MissionSync(AllReduceMesh1DMem2Mem2DieOneShotContext &ctx, uint32_t maskIndex)
{
    uint32_t coreIdx = ctx.rmtReduceWithMyRank ? 1 : 0;
    CCU_CHK_RET(ccu::LocalNotifyRecord(1 - coreIdx, CKE_IDX_1, ctx.missionSyncMybit << (MISSION_NUM * maskIndex)));
    CCU_CHK_RET(ccu::LocalNotifyWait(coreIdx, CKE_IDX_1, ctx.missionSyncWaitBit << (MISSION_NUM * maskIndex)));
    return CCU_SUCCESS;
}

CcuResult CcuAllReduceMesh1DMem2Mem2DieOneShotKernel(CcuKernelArg arg)
{
    auto *kernelArg = static_cast<CcuKernelArgAllReduceMesh1DMem2Mem2DieOneShot *>(arg);

    AllReduceMesh1DMem2Mem2DieOneShotContext ctx;
    ctx.arg = kernelArg;
    ctx.resourceAllocated = false;
    ctx.moConfig.msInterleave = 0;
    ctx.moConfig.loopCount = 0;
    ctx.moConfig.memSlice = 0;
    ctx.moRes.eventCount = 0;
    ctx.moRes.bufCount = 0;
    ctx.enginePool = 0;

    HCCL_INFO("[CcuKernelAllReduceMesh1DMem2Mem2DieOneShot] AllReduceMesh1DMem2Mem2DieOneShot run");

    CCU_CHK_RET(ParseKernelArg(ctx));
    CCU_CHK_RET(InitResource(ctx));
    CCU_CHK_RET(LoadArgs(ctx));

    CCU_CHK_RET(PreSync(ctx));
    CCU_CHK_RET(RmtReduce(ctx));
    CCU_CHK_RET(PostSync(ctx));

    CCU_CHK_RET(MissionSync(ctx, 0));
    CCU_CHK_RET(DoLocalReduce(ctx));
    CCU_CHK_RET(MissionSync(ctx, 1));

    HCCL_INFO("[CcuKernelAllReduceMesh1DMem2Mem2DieOneShot] AllReduceMesh1DMem2Mem2DieOneShot end");

    return CCU_SUCCESS;
}

} // namespace ops_hccl
