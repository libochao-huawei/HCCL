/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "ccu_kernel_broadcast_mesh1d_mem2mem.h"
#include "ccu_kernel_alg_base.h"

namespace ops_hccl {

constexpr int INPUT_XN_ID  = 0;
constexpr int OUTPUT_XN_ID = 1;
constexpr int TOKEN_XN_ID  = 2;
constexpr int CKE_IDX_0    = 0;
constexpr int CKE_IDX_3    = 3;
constexpr int CKE_IDX_4    = 4;

static CcuResult ParseKernelArg(BroadcastMesh1DMem2MemContext &ctx, CcuKernelArgBroadcastMesh1DMem2Mem *kernelArg)
{
    ctx.arg = kernelArg;
    ctx.rankSize = kernelArg->rankSize;
    ctx.rankId = kernelArg->rankId;
    ctx.rootId = kernelArg->rootId;
    ctx.dataType = kernelArg->opParam.DataDes.dataType;
    ctx.outputDataType = kernelArg->opParam.DataDes.outputType;
    if (ctx.outputDataType == HcclDataType::HCCL_DATA_TYPE_RESERVED) {
        ctx.outputDataType = ctx.dataType;
        HCCL_DEBUG(
            "[CcuBroadcastMesh1DMem2Mem] outputDataType is [INVALID], set outputDataType to[%d]",
            ctx.outputDataType);
    }
    HCCL_INFO("[CcuBroadcastMesh1DMem2Mem] Init, KernelArgs are rootId[%u] rankId[%u], rankSize_[%u], dataType[%d], "
        "outputDataType[%d] channelsSize[%u]",
        ctx.rootId, ctx.rankId, ctx.rankSize, ctx.dataType, ctx.outputDataType, ctx.arg->channelCount);
    return CCU_SUCCESS;
}

static CcuResult InitResource(BroadcastMesh1DMem2MemContext &ctx)
{
    uint16_t channelIdx = 0;
    if (ctx.arg->channelCount == 0) {
        HCCL_ERROR("[CcuBroadcastMesh1DMem2Mem] channels is empty!");
        return CCU_E_INTERNAL;
    }

    ctx.input.resize(ctx.rankSize);
    ctx.output.resize(ctx.rankSize);
    ctx.token.resize(ctx.rankSize);
    for (uint64_t peerId = 0; peerId < ctx.rankSize; peerId++) {
        if (peerId != ctx.rankId) {
            ctx.input[peerId] = ccu::GetResByChannel<ccu::Variable>(ctx.arg->channels[channelIdx], INPUT_XN_ID);
            ctx.output[peerId] = ccu::GetResByChannel<ccu::Variable>(ctx.arg->channels[channelIdx], OUTPUT_XN_ID);
            ctx.token[peerId] = ccu::GetResByChannel<ccu::Variable>(ctx.arg->channels[channelIdx], TOKEN_XN_ID);
            channelIdx++;
        }
    }

    ctx.scatterdstMem.reserve(ctx.rankSize);
    ctx.allgatherdstMem.reserve(ctx.rankSize);
    ctx.scattersrcMem.reserve(ctx.rankSize);
    for (uint64_t rankIdx = 0; rankIdx < ctx.rankSize; rankIdx++) {
        ctx.scattersrcMem.push_back(ccu::LocalAddr());
        if (rankIdx == ctx.rankId) {
            ctx.myScatterDst = ccu::LocalAddr();
            ctx.scatterdstMem.push_back({});
            ctx.allgatherdstMem.push_back({});
        } else {
            ctx.allgatherdstMem.push_back(ccu::RemoteAddr());
            ctx.scatterdstMem.push_back(ccu::RemoteAddr());
        }
    }

    ctx.resourceAllocated = false;
    return CCU_SUCCESS;
}

static CcuResult LoadArgs(BroadcastMesh1DMem2MemContext &ctx)
{
    uint32_t argId = 0;
    CCU_CHK_RET(ccu::LoadArg(ctx.input[ctx.rankId], argId++));
    CCU_CHK_RET(ccu::LoadArg(ctx.output[ctx.rankId], argId++));
    CCU_CHK_RET(ccu::LoadArg(ctx.token[ctx.rankId], argId++));
    CCU_CHK_RET(ccu::LoadArg(ctx.currentRankSliceInputOffset, argId++));
    CCU_CHK_RET(ccu::LoadArg(ctx.currentRankSliceOutputOffset, argId++));
    CCU_CHK_RET(ccu::LoadArg(ctx.inputRepeatStride, argId++));
    CCU_CHK_RET(ccu::LoadArg(ctx.outputRepeatStride, argId++));
    CCU_CHK_RET(ccu::LoadArg(ctx.normalSliceSize, argId++));
    CCU_CHK_RET(ccu::LoadArg(ctx.lastSliceSize, argId++));
    CCU_CHK_RET(ccu::LoadArg(ctx.allgatherOffset, argId++));
    CCU_CHK_RET(ccu::LoadArg(ctx.repeatNumVar, argId++));
    return CCU_SUCCESS;
}

static CcuResult PreSync(BroadcastMesh1DMem2MemContext &ctx)
{
    for (uint32_t i = 0; i < ctx.arg->channelCount; i++) {
        ccu::WriteVariableWithNotify(ctx.arg->channels[i], ctx.input[ctx.rankId], INPUT_XN_ID, CKE_IDX_0, 1 << INPUT_XN_ID);
        ccu::WriteVariableWithNotify(ctx.arg->channels[i], ctx.output[ctx.rankId], OUTPUT_XN_ID, CKE_IDX_0, 1 << OUTPUT_XN_ID);
        ccu::WriteVariableWithNotify(ctx.arg->channels[i], ctx.token[ctx.rankId], TOKEN_XN_ID, CKE_IDX_0, 1 << TOKEN_XN_ID);
    }
    uint32_t allBit = 1 << INPUT_XN_ID | 1 << OUTPUT_XN_ID | 1 << TOKEN_XN_ID;
    for (uint32_t i = 0; i < ctx.arg->channelCount; i++) {
        ccu::NotifyWait(ctx.arg->channels[i], CKE_IDX_0, allBit);
    }
    HCCL_INFO("[CcuBroadcastMesh1DMem2Mem] BroadcastMesh1D wait all end");
    return CCU_SUCCESS;
}

static CcuResult PostSync(BroadcastMesh1DMem2MemContext &ctx, int CKE_id)
{
    for (uint32_t i = 0; i < ctx.arg->channelCount; i++) {
        ccu::NotifyRecord(ctx.arg->channels[i], CKE_IDX_0, 1 << CKE_id);
    }
    for (uint32_t i = 0; i < ctx.arg->channelCount; i++) {
        ccu::NotifyWait(ctx.arg->channels[i], CKE_IDX_0, 1 << CKE_id);
    }
    HCCL_INFO("[CcuBroadcastMesh1DMem2Mem] BroadcastMesh1D PostSync end");
    return CCU_SUCCESS;
}

static CcuResult DoScatter(BroadcastMesh1DMem2MemContext &ctx, const std::vector<ccu::RemoteAddr> &dst)
{
    HCCL_INFO("[CcuBroadcastMesh1DMem2Mem][DoScatter] DoScatter");
    uint64_t channelId = 0;
    for (uint32_t rankIdx = 0; rankIdx < ctx.rankSize; rankIdx++) {
        ccu::Variable sliceSize = (rankIdx + 1 == ctx.rankSize) ? ctx.lastSliceSize : ctx.normalSliceSize;
        ctx.event.SetMask(1 << rankIdx);
        CCU_IF(sliceSize != 0)
        {
            if (rankIdx == ctx.rankId) {
                ccu::EventRecord(ctx.event, 1 << rankIdx);
            } else {
                ccu::Write(ctx.arg->channels[channelId], dst[rankIdx],
                    ctx.scattersrcMem[rankIdx], sliceSize, ctx.event, 1 << rankIdx);
                HCCL_INFO("[CcuBroadcastMesh1DMem2Mem][DoScatter] channelsId[%llu] rankIdx[%u]",
                            channelId, rankIdx);
                channelId++;
            }
        }
        CCU_IF(sliceSize == 0)
        {
            ccu::EventRecord(ctx.event, 1 << rankIdx);
        }
    }

    ctx.event.SetMask((1 << ctx.rankSize) - 1);
    ccu::EventWait(ctx.event, (1 << ctx.rankSize) - 1);
    return CCU_SUCCESS;
}

static CcuResult DoRepeaScatterMem2Mem(BroadcastMesh1DMem2MemContext &ctx)
{
    if (ctx.rankId != ctx.rootId) {
        return CCU_SUCCESS;
    }
    HCCL_INFO("[CcuBroadcastMesh1DMem2Mem][DoRepeaScatterMem2Mem]rankId[%u] rankSize[%u]", ctx.rankId, ctx.rankSize);
    std::vector<ccu::RemoteAddr> &dst = ctx.scatterdstMem;
    ccu::Variable SliceOffset;
    SliceOffset = 0;
    for (uint32_t rankIdx = 0; rankIdx < ctx.rankSize; rankIdx++) {
        if (rankIdx == 0) {
            SliceOffset = 0;
        } else {
            SliceOffset += ctx.normalSliceSize;
        }
        ctx.scattersrcMem[rankIdx].addr = ctx.input[ctx.rankId];
        ctx.scattersrcMem[rankIdx].addr += ctx.currentRankSliceInputOffset;
        ctx.scattersrcMem[rankIdx].addr += SliceOffset;
        ctx.scattersrcMem[rankIdx].token = ctx.token[ctx.rankId];
        if (rankIdx == ctx.rankId) {
            ctx.myScatterDst.addr = ctx.output[rankIdx];
            ctx.myScatterDst.addr += ctx.currentRankSliceOutputOffset;
            ctx.myScatterDst.addr += SliceOffset;
            ctx.myScatterDst.token = ctx.token[rankIdx];
        } else {
            dst[rankIdx].addr = ctx.output[rankIdx];
            dst[rankIdx].addr += ctx.currentRankSliceOutputOffset;
            dst[rankIdx].addr += SliceOffset;
            dst[rankIdx].token = ctx.token[rankIdx];
        }

        CCU_IF(ctx.flag != 0)
        {
            // 非第一轮执行时，src 和 dst 已经初始化，需要添加偏移量
            ctx.myScatterDst.addr += ctx.outputRepeatStride;
            for (uint32_t curId = 0; curId < ctx.rankSize; curId++) {
                ctx.scattersrcMem[curId].addr += ctx.inputRepeatStride;
                if (curId != ctx.rankId) {
                    dst[curId].addr += ctx.outputRepeatStride;
                }
            }
        }
    }
    CCU_CHK_RET(DoScatter(ctx, dst));
    return CCU_SUCCESS;
}

static CcuResult DoAllGather(BroadcastMesh1DMem2MemContext &ctx, const ccu::LocalAddr &src,
    const std::vector<ccu::RemoteAddr> &dst)
{
    uint64_t channelId = 0;
    ccu::Variable sliceSize = (ctx.rankId + 1 == ctx.rankSize) ? ctx.lastSliceSize : ctx.normalSliceSize;
    CCU_IF(sliceSize != 0)
    {
        for (uint64_t rankIdx = 0; rankIdx < ctx.rankSize; rankIdx++) {
            ctx.event.SetMask(1 << rankIdx);
            if (rankIdx == ctx.rankId) {
                ccu::EventRecord(ctx.event, 1 << rankIdx);
            } else {
                ccu::Write(ctx.arg->channels[channelId], dst[rankIdx], src, sliceSize, ctx.event, 1 << rankIdx);
                channelId++;
            }
        }
        ctx.event.SetMask((1 << ctx.rankSize) - 1);
        ccu::EventWait(ctx.event, (1 << ctx.rankSize) - 1);
    }
    return CCU_SUCCESS;
}

static CcuResult DoRepeatAllGatherMem2Mem(BroadcastMesh1DMem2MemContext &ctx)
{
    ccu::LocalAddr &src = ctx.myScatterDst;
    std::vector<ccu::RemoteAddr> &dst = ctx.allgatherdstMem;
    src.addr = ctx.output[ctx.rankId];
    src.addr += ctx.currentRankSliceOutputOffset;
    src.addr += ctx.allgatherOffset;
    src.token = ctx.token[ctx.rankId];

    for (uint32_t rankIdx = 0; rankIdx < ctx.rankSize; rankIdx++) {
        if (rankIdx != ctx.rankId) {
            dst[rankIdx].addr = ctx.output[rankIdx];
            dst[rankIdx].addr += ctx.currentRankSliceOutputOffset;
            dst[rankIdx].addr += ctx.allgatherOffset;
            dst[rankIdx].token = ctx.token[rankIdx];
        }
    }
    CCU_IF(ctx.flag != 0)
    {
        //  非第一轮执行时，src 和 dst 已经初始化，需要添加偏移量
        src.addr += ctx.inputRepeatStride;
        for (uint32_t curId = 0; curId < ctx.rankSize; curId++) {
            if (curId != ctx.rankId) {
                dst[curId].addr += ctx.outputRepeatStride;
            }
        }
    }

    CCU_CHK_RET(DoAllGather(ctx, src, dst));
    return CCU_SUCCESS;
}

CcuResult CcuBroadcastMesh1DMem2MemKernel(CcuKernelArg arg)
{
    auto *kernelArg = static_cast<CcuKernelArgBroadcastMesh1DMem2Mem *>(arg);
    BroadcastMesh1DMem2MemContext ctx;
    ctx.arg = kernelArg;
    ctx.resourceAllocated = false;
    ctx.moConfig.msInterleave = 0;
    ctx.moConfig.loopCount = 0;
    ctx.moConfig.memSlice = 0;
    ctx.moRes.eventCount = 0;
    ctx.moRes.bufCount = 0;
    ctx.enginePool = 0;

    HCCL_INFO("[CcuBroadcastMesh1DMem2Mem] BroadcastMesh1D run");
    CCU_CHK_RET(ParseKernelArg(ctx, kernelArg));
    CCU_CHK_RET(InitResource(ctx));
    CCU_CHK_RET(LoadArgs(ctx));

    CCU_CHK_RET(PreSync(ctx));

    ccu::Variable repeatNumAdd;
    repeatNumAdd = 1;
    ctx.flag = 0;
    CCU_WHILE(ctx.repeatNumVar != UINT64_MAX)
    {
        // 循环repeatNum_次
        CCU_CHK_RET(DoRepeaScatterMem2Mem(ctx));
        CCU_CHK_RET(PostSync(ctx, CKE_IDX_3));
        CCU_CHK_RET(DoRepeatAllGatherMem2Mem(ctx));
        CCU_CHK_RET(PostSync(ctx, CKE_IDX_4));
        ctx.repeatNumVar += repeatNumAdd;
        ctx.flag = 1;
    }

    HCCL_INFO("[CcuBroadcastMesh1DMem2Mem] BroadcastMesh1D end");
    return CCU_SUCCESS;
}

} // namespace ops_hccl