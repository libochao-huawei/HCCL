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
#include "ccu_kernel_alg_base.h"

namespace ops_hccl {

constexpr int INPUT_XN_ID  = 0;
constexpr int OUTPUT_XN_ID = 1;
constexpr int TOKEN_XN_ID  = 2;
constexpr int CKE_IDX_0    = 0;
constexpr int CKE_IDX_1    = 1;
constexpr int CKE_IDX_2    = 2;
constexpr int POST_SYNC_ID = 3;

static CcuResult ParseKernelArg(AllToAllMesh2DieContext &ctx, CcuKernelArgAllToAllMesh2Die *kernelArg)
{
    ctx.rankSize = kernelArg->rankSize;
    ctx.rankId = kernelArg->rankId;
    ctx.withMyRank = kernelArg->withMyRank;
    ctx.rankGroup = kernelArg->rankGroup;
    ctx.channels.assign(kernelArg->channels, kernelArg->channels + kernelArg->channelCount);
    return CCU_SUCCESS;
}

static CcuResult InitResource(AllToAllMesh2DieContext &ctx)
{
    if (ctx.channels.size() == 0) {
        HCCL_ERROR("[CcuKernelAllToAllMesh2Die] RankId[%u] channels is empty", ctx.rankId);
        return CCU_E_INTERNAL;
    }
    ctx.virRankSize = ctx.channels.size() + 1;

    ctx.output.resize(ctx.channels.size());
    ctx.token.resize(ctx.channels.size());
    for (u64 id = 0; id < ctx.channels.size(); id++) {
        ctx.output[id] = ccu::GetResByChannel<ccu::Variable>(ctx.channels[id], OUTPUT_XN_ID);
        ctx.token[id] = ccu::GetResByChannel<ccu::Variable>(ctx.channels[id], TOKEN_XN_ID);
    }

    ctx.output.emplace_back(ccu::Variable());
    ctx.token.emplace_back(ccu::Variable());

    ctx.logicRankSize = ctx.withMyRank ? ctx.channels.size() + 1 : ctx.channels.size();

    return CCU_SUCCESS;
}

static CcuResult LoadArgs(AllToAllMesh2DieContext &ctx)
{
    CCU_CHK_RET(ccu::LoadArg(ctx.input, 0));
    CCU_CHK_RET(ccu::LoadArg(ctx.output[ctx.virRankSize - 1], 1));
    CCU_CHK_RET(ccu::LoadArg(ctx.token[ctx.virRankSize - 1], 2));
    CCU_CHK_RET(ccu::LoadArg(ctx.sliceSize, 3));
    CCU_CHK_RET(ccu::LoadArg(ctx.inputSliceStride, 4));
    CCU_CHK_RET(ccu::LoadArg(ctx.outputSliceStride, 5));
    CCU_CHK_RET(ccu::LoadArg(ctx.groupOpSize.addrOffset, 6));
    CCU_CHK_RET(ccu::LoadArg(ctx.groupOpSize.loopParam, 7));
    CCU_CHK_RET(ccu::LoadArg(ctx.groupOpSize.parallelParam, 8));
    CCU_CHK_RET(ccu::LoadArg(ctx.groupOpSize.residual, 9));

    return CCU_SUCCESS;
}

static void PreSync(AllToAllMesh2DieContext &ctx)
{
    for (const ChannelHandle &channel : ctx.channels) {
        ccu::WriteVariableWithNotify(channel, ctx.output[ctx.virRankSize - 1], OUTPUT_XN_ID, CKE_IDX_0, 1 << OUTPUT_XN_ID);
        ccu::WriteVariableWithNotify(channel, ctx.token[ctx.virRankSize - 1], TOKEN_XN_ID, CKE_IDX_0, 1 << TOKEN_XN_ID);
    }
    uint32_t waitBits = (1 << OUTPUT_XN_ID) | (1 << TOKEN_XN_ID);
    for (const ChannelHandle &channel : ctx.channels) {
        ccu::NotifyWait(channel, CKE_IDX_0, waitBits);
    }
}

static void PostSync(AllToAllMesh2DieContext &ctx)
{
    for (const auto &channel : ctx.channels) {
        ccu::NotifyRecord(channel, CKE_IDX_0, 1 << POST_SYNC_ID);
    }
    for (const auto &channel : ctx.channels) {
        ccu::NotifyWait(channel, CKE_IDX_0, 1 << POST_SYNC_ID);
    }
}

static uint32_t CalcDstRank(AllToAllMesh2DieContext &ctx, uint32_t peerId)
{
    if (peerId > ctx.rankGroup.size()) {
        HCCL_ERROR("[CcuKernelAllToAllMesh2Die][CalcDstRank] Unexpected peerId[%u]", peerId);
    }
    return ctx.rankGroup[peerId];
}

static CcuResult DoRepeatAllToAll(AllToAllMesh2DieContext &ctx)
{
    std::vector<ccu::LocalAddr> src(ctx.logicRankSize);
    std::vector<ccu::RemoteAddr> dst(ctx.logicRankSize);

    for (uint64_t r = 0; r < ctx.logicRankSize; r++) {
        const u32 dstRank = CalcDstRank(ctx, r);

        src[r].token = ctx.token[r];
        dst[r].token = ctx.token[r];

        src[r].addr = ctx.input;

        dst[r].addr = ctx.output[r];
        dst[r].addr += ctx.outputSliceStride;
        for(uint64_t i = 0; i < dstRank; i++){
            src[r].addr += ctx.inputSliceStride;
        }
    }

    ccu::LocalAddr localSrc;
    ccu::LocalAddr localDst;
    if(ctx.withMyRank){
        const u32 with_dstRank = CalcDstRank(ctx, ctx.logicRankSize - 1);

        localSrc.token = ctx.token[ctx.logicRankSize - 1];
        localSrc.addr = ctx.input;

        localDst.token = ctx.token[ctx.logicRankSize - 1];
        localDst.addr = ctx.output[ctx.logicRankSize - 1];
        localDst.addr += ctx.outputSliceStride;
        for(uint64_t i = 0; i < with_dstRank; i++){
            localSrc.addr += ctx.inputSliceStride;
        }
    }

    u32 channelsIdx = 0;
    for (uint64_t r = 0; r < ctx.logicRankSize; r++) {
        if (ctx.withMyRank && r == ctx.logicRankSize - 1) {
            GroupCopy(ctx, localDst, localSrc, ctx.groupOpSize);
            continue;
        }
        ccu::Write(ctx.channels[channelsIdx], dst[r], src[r], ctx.sliceSize, ctx.event, 1 << r);
        channelsIdx++;
    }
    uint16_t waitMask = ctx.withMyRank ? ((1 << ctx.logicRankSize) - 1) & (~(1 << ctx.channels.size())) : (1 << ctx.logicRankSize) - 1;
    ccu::EventWait(ctx.event, waitMask);

    return CCU_SUCCESS;
}

CcuResult CcuAllToAllMesh2DieKernel(CcuKernelArg arg)
{
    auto *kernelArg = static_cast<CcuKernelArgAllToAllMesh2Die *>(arg);
    AllToAllMesh2DieContext ctx;
    ctx.arg = kernelArg;
    ctx.resourceAllocated = false;
    ctx.moConfig.msInterleave = 0;
    ctx.moConfig.loopCount = 0;
    ctx.moConfig.memSlice = 0;
    ctx.moRes.eventCount = 0;
    ctx.moRes.bufCount = 0;
    ctx.enginePool = 0;

    HCCL_INFO("[ccuAllToAllMesh2Die_kernel] AllToAllMesh2Die run.");
    CCU_CHK_RET(ParseKernelArg(ctx, kernelArg));
    CCU_CHK_RET(InitResource(ctx));
    CCU_CHK_RET(LoadArgs(ctx));
    PreSync(ctx);
    CCU_CHK_RET(DoRepeatAllToAll(ctx));
    PostSync(ctx);
    HCCL_INFO("[ccuAllToAllMesh2Die_kernel] AllToAllMesh2Die end.");
    return CCU_SUCCESS;
}

}// namespace ops_hccl
