/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "ccu_kernel_scatter_mesh1d.h"
#include "ccu_kernel_alg_base.h"

namespace ops_hccl {

constexpr uint16_t OUTPUT_XN_ID   = 0;
constexpr uint16_t TOKEN_XN_ID    = 1;
constexpr uint16_t POST_SYNC_ID   = 2;
constexpr uint16_t CKE_IDX_0      = 0;

static CcuResult ParseKernelArg(ScatterMesh1DContext &ctx, CcuKernelArgScatterMesh1D *kernelArg)
{
    ctx.arg = kernelArg;
    ctx.rankSize = kernelArg->rankSize;
    ctx.rankId = kernelArg->rankId;
    ctx.rootId = kernelArg->rootId;
    ctx.channels = kernelArg->channels;
    ctx.dataType = kernelArg->opParam.DataDes.dataType;
    ctx.outputDataType = kernelArg->opParam.DataDes.outputType;
    if (ctx.outputDataType == HcclDataType::HCCL_DATA_TYPE_RESERVED) {
        ctx.outputDataType = ctx.dataType;
    }
    return CCU_SUCCESS;
}

static CcuResult InitResource(ScatterMesh1DContext &ctx)
{
    uint16_t channelIdx = 0;
    if (ctx.channels.size() == 0) {
        HCCL_ERROR("[CcuScatterMesh1DKernel] channels is empty!");
        return CCU_E_INTERNAL;
    }

    ctx.output.resize(ctx.rankSize);
    ctx.token.resize(ctx.rankSize);
    for (uint64_t peerId = 0; peerId < ctx.rankSize; peerId++) {
        if (peerId == ctx.rankId) {
            CCU_CHK_RET(ccu::Alloc(&ctx.output[peerId]));
            CCU_CHK_RET(ccu::Alloc(&ctx.token[peerId]));
        } else {
            CCU_CHK_RET(ccu::CreateByChannel(ctx.channels[channelIdx], OUTPUT_XN_ID, &ctx.output[peerId]));
            CCU_CHK_RET(ccu::CreateByChannel(ctx.channels[channelIdx], TOKEN_XN_ID, &ctx.token[peerId]));
            channelIdx++;
        }
    }

    CCU_CHK_RET(ccu::Alloc(&ctx.input));
    CCU_CHK_RET(ccu::Alloc(&ctx.currentRankSliceInputOffset));
    CCU_CHK_RET(ccu::Alloc(&ctx.outputSliceStride));
    CCU_CHK_RET(ccu::Alloc(&ctx.inputRepeatStride));
    CCU_CHK_RET(ccu::Alloc(&ctx.outputRepeatStride));
    CCU_CHK_RET(ccu::Alloc(&ctx.normalSliceSize));
    CCU_CHK_RET(ccu::Alloc(&ctx.lastSliceSize));
    CCU_CHK_RET(ccu::Alloc(&ctx.repeatNum));
    CCU_CHK_RET(ccu::Alloc(&ctx.isInputOutputEqual));
    CCU_CHK_RET(ccu::Alloc(&ctx.flag));
    ctx.flag = 0;

    ctx.inputMem.resize(ctx.rankSize);
    ctx.outputMem.resize(ctx.rankSize);
    for (uint64_t i = 0; i < ctx.rankSize; i++) {
        CCU_CHK_RET(ccu::Alloc(&ctx.inputMem[i]));
        CCU_CHK_RET(ccu::Alloc(&ctx.outputMem[i]));
    }
    CCU_CHK_RET(ccu::Alloc(&ctx.event));
    return CCU_SUCCESS;
}

static CcuResult LoadArgs(ScatterMesh1DContext &ctx)
{
    CCU_CHK_RET(ccu::LoadArg(ctx.input));
    CCU_CHK_RET(ccu::LoadArg(ctx.output[ctx.rankId]));
    CCU_CHK_RET(ccu::LoadArg(ctx.token[ctx.rankId]));
    CCU_CHK_RET(ccu::LoadArg(ctx.currentRankSliceInputOffset));
    CCU_CHK_RET(ccu::LoadArg(ctx.outputSliceStride));
    CCU_CHK_RET(ccu::LoadArg(ctx.inputRepeatStride));
    CCU_CHK_RET(ccu::LoadArg(ctx.outputRepeatStride));
    CCU_CHK_RET(ccu::LoadArg(ctx.normalSliceSize));
    CCU_CHK_RET(ccu::LoadArg(ctx.lastSliceSize));
    CCU_CHK_RET(ccu::LoadArg(ctx.repeatNum));
    CCU_CHK_RET(ccu::LoadArg(ctx.isInputOutputEqual));
    return CCU_SUCCESS;
}

static CcuResult PreSync(ScatterMesh1DContext &ctx)
{
    uint32_t allBit = (1 << OUTPUT_XN_ID) | (1 << TOKEN_XN_ID);
    for (auto ch : ctx.channels) {
        ccu::WriteVariableWithNotify(ch, ctx.output[ctx.rankId], OUTPUT_XN_ID, CKE_IDX_0, 1 << OUTPUT_XN_ID);
        ccu::WriteVariableWithNotify(ch, ctx.token[ctx.rankId], TOKEN_XN_ID, CKE_IDX_0, 1 << TOKEN_XN_ID);
    }
    for (auto ch : ctx.channels) {
        ccu::NotifyWait(ch, CKE_IDX_0, allBit);
    }
    return CCU_SUCCESS;
}

static CcuResult PostSync(ScatterMesh1DContext &ctx)
{
    for (auto ch : ctx.channels) {
        ccu::NotifyRecord(ch, CKE_IDX_0, 1 << POST_SYNC_ID);
    }
    for (auto ch : ctx.channels) {
        ccu::NotifyWait(ch, CKE_IDX_0, 1 << POST_SYNC_ID);
    }
    return CCU_SUCCESS;
}

static CcuResult DoScatterOnce(ScatterMesh1DContext &ctx)
{
    uint32_t channelId = 0;

    CcuLocalAddr myOutput;
    CCU_CHK_RET(ccu::Alloc(&myOutput));
    myOutput.addr = ctx.outputMem[ctx.rankId].addr;
    myOutput.token = ctx.outputMem[ctx.rankId].token;

    CcuVariable sliceSize;
    CCU_CHK_RET(ccu::Alloc(&sliceSize));

    for (uint64_t rankIdx = 0; rankIdx < ctx.rankSize; rankIdx++) {
        ctx.event.mask = 1 << rankIdx;
        sliceSize = (rankIdx == ctx.rankSize - 1) ? ctx.lastSliceSize : ctx.normalSliceSize;
        CCU_IF_ONLY(sliceSize != 0)
        {
            if (rankIdx == ctx.rankId) {
                CCU_IF_ONLY(ctx.isInputOutputEqual == 0)
                {
                    ccu::LocalCopyNb(myOutput, ctx.inputMem[rankIdx], sliceSize, ctx.event);
                }
                CCU_IF_ONLY(ctx.isInputOutputEqual != 0)
                {
                    ccu::RecordEvent(ctx.event);
                }
            } else {
                ccu::WriteNb(ctx.channels[channelId], ctx.outputMem[rankIdx], ctx.inputMem[rankIdx], sliceSize, ctx.event);
                channelId++;
            }
        }
        CCU_IF_ONLY(sliceSize == 0)
        {
            ccu::RecordEvent(ctx.event);
        }
    }

    ctx.event.mask = (1 << ctx.rankSize) - 1;
    ccu::WaitEvent(ctx.event);
    return CCU_SUCCESS;
}

static CcuResult DoRepeatScatter(ScatterMesh1DContext &ctx)
{
    CcuVariable repeatNumAdd;
    CCU_CHK_RET(ccu::Alloc(&repeatNumAdd));
    repeatNumAdd = 1;

    // 初始化每张卡 input/output 逻辑地址
    for (uint64_t curId = 0; curId < ctx.rankSize; curId++) {
        ctx.inputMem[curId].token = ctx.token[curId];
        ctx.outputMem[curId].token = ctx.token[curId];

        ctx.inputMem[curId].addr = ctx.input;
        ctx.outputMem[curId].addr = ctx.output[curId];
        for (uint64_t i = 0; i < curId; i++) {
            ctx.inputMem[curId].addr = ctx.inputMem[curId].addr + ctx.currentRankSliceInputOffset;
            ctx.outputMem[curId].addr = ctx.outputMem[curId].addr + ctx.outputSliceStride;
        }
    }

    if (ctx.rankId != ctx.rootId) {
        return CCU_SUCCESS;
    }

    CCU_WHILE(ctx.repeatNum != UINT64_MAX)
    {
        CCU_IF_ONLY(ctx.flag != 0)
        {
            for (auto &i : ctx.inputMem) {
                i.addr = i.addr + ctx.inputRepeatStride;
            }
            for (auto &r : ctx.outputMem) {
                r.addr = r.addr + ctx.outputRepeatStride;
            }
        }
        CCU_CHK_RET(DoScatterOnce(ctx));
        ctx.repeatNum = ctx.repeatNum + repeatNumAdd;
        ctx.flag = 1;
    }
    return CCU_SUCCESS;
}

CcuResult CcuScatterMesh1DKernel(CcuKernelArg arg)
{
    auto *kernelArg = static_cast<CcuKernelArgScatterMesh1D *>(arg);
    ScatterMesh1DContext ctx{};

    CCU_CHK_RET(ParseKernelArg(ctx, kernelArg));
    CCU_CHK_RET(InitResource(ctx));
    CCU_CHK_RET(LoadArgs(ctx));

    CCU_CHK_RET(PreSync(ctx));
    CCU_CHK_RET(DoRepeatScatter(ctx));
    CCU_CHK_RET(PostSync(ctx));
    return CCU_SUCCESS;
}

} // namespace ops_hccl