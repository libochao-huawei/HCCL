/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "ccu_kernel_broadcast_mesh1d.h"

namespace ops_hccl {

constexpr int INPUT_XN_ID  = 0;
constexpr int OUTPUT_XN_ID = 1;
constexpr int TOKEN_XN_ID  = 2;
constexpr int POST_SYNC_ID = 3;
constexpr int CKE_IDX_0 = 0;

static CcuResult ParseKernelArg(BroadcastMesh1DContext &ctx, CcuKernelArgBroadcastMesh1D *kernelArg)
{
    ctx.dataType       = kernelArg->opParam.DataDes.dataType;
    ctx.outputDataType = kernelArg->opParam.DataDes.outputType;
    if (ctx.outputDataType == HcclDataType::HCCL_DATA_TYPE_RESERVED) {
        ctx.outputDataType = ctx.dataType;
        HCCL_DEBUG(
            "[CcuKernelBroadcastMesh1D] outputDataType is [INVALID], set outputDataType to[%d]",
            ctx.dataType);
    }
    return CCU_SUCCESS;
}

static CcuResult InitResource(BroadcastMesh1DContext &ctx)
{
    const auto *arg = ctx.arg;
    uint32_t channelIdx = 0;
    if (arg->channelCount == 0) {
        HCCL_ERROR("[CcuKernelBroadcastMesh1D] channels is empty!");
        return CcuResult::CCU_E_INTERNAL;
    }
    // 按照rank号从小到大遍历channels，遇到本rank就填充本地资源，否则依次取远端资源，要求给框架返回的Link同样是按顺序排列的
    ctx.output.resize(arg->rankSize);
    ctx.token.resize(arg->rankSize);
    for (uint64_t peerId = 0; peerId < arg->rankSize; peerId++) {
        if (peerId != arg->rankId) {
            ctx.output[peerId] = ccu::GetResByChannel<ccu::Variable>(arg->channels[channelIdx], OUTPUT_XN_ID);
            ctx.token[peerId] = ccu::GetResByChannel<ccu::Variable>(arg->channels[channelIdx], TOKEN_XN_ID);
            channelIdx++;
        }
    }
    ctx.resourceAllocated = false;
    return CCU_SUCCESS;
}

static CcuResult LoadArgs(BroadcastMesh1DContext &ctx)
{
    const auto *arg = ctx.arg;
    uint32_t argId = 0;

    CCU_CHK_RET(ccu::LoadArg(ctx.input, argId++));
    CCU_CHK_RET(ccu::LoadArg(ctx.output[arg->rankId], argId++));
    CCU_CHK_RET(ccu::LoadArg(ctx.token[arg->rankId], argId++));
    CCU_CHK_RET(ccu::LoadArg(ctx.offset, argId++));
    CCU_CHK_RET(ccu::LoadArg(ctx.slicesize, argId++));
    CCU_CHK_RET(ccu::LoadArg(ctx.groupOpSize.addrOffset, argId++));
    CCU_CHK_RET(ccu::LoadArg(ctx.groupOpSize.loopParam, argId++));
    CCU_CHK_RET(ccu::LoadArg(ctx.groupOpSize.parallelParam, argId++));
    CCU_CHK_RET(ccu::LoadArg(ctx.groupOpSize.residual, argId++));

    return CCU_SUCCESS;
}

static CcuResult PreSync(BroadcastMesh1DContext &ctx)
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

static CcuResult PostSync(BroadcastMesh1DContext &ctx)
{
    const auto *arg = ctx.arg;

    for (uint32_t i = 0; i < arg->channelCount; i++) {
        ccu::NotifyRecord(arg->channels[i], CKE_IDX_0, 1 << POST_SYNC_ID);
    }
    for (uint32_t i = 0; i < arg->channelCount; i++) {
        ccu::NotifyWait(arg->channels[i], CKE_IDX_0, 1 << POST_SYNC_ID);
    }
    return CCU_SUCCESS;
}

// static CcuResult DoBroadcast(BroadcastMesh1DContext &ctx)
// {
//     const auto *arg = ctx.arg;
//     std::vector<CcuRep::RemoteAddr> dst;
//     CcuRep::LocalAddr src = CreateLocalAddr();
//     for (uint32_t index = 0; index < rankSize_; index++) {
//         dst.emplace_back(CreateRemoteAddr());
//     }

//     src.addr = input_;
//     src.token = token_[rankId_];
//     uint32_t curId = 0;
//     for (uint32_t rankIdx = 0; rankIdx < rankSize_; rankIdx++) {
//         if (rankIdx != rootId_) {
//             dst[curId].addr  = output_[rankIdx];
//             dst[curId].token = token_[rankIdx];
//             curId++;
//         }
//     }
//     dst[rankSize_ - 1].addr = output_[rankId_];
//     dst[rankSize_ - 1].token = token_[rankId_];
//     GroupBroadcast(channels_, dst, src, groupOpSize_);
//     HCCL_INFO("[CcuKernelBroadcastMesh1D] BroadcastMesh1D GroupBroadcast end");
//     return;
// }

static CcuResult DoBroadcast(BroadcastMesh1DContext &ctx)
{
    const auto *arg = ctx.arg;
    ccu::LocalAddr src;
    ccu::LocalAddr localdst;
    std::vector<ccu::RemoteAddr> dst;   
    dst.resize(arg->rankSize - 1);

    src.addr = ctx.input;
    src.token = ctx.token[arg->rankId];
    uint32_t curId = 0;

    for (uint32_t rankIdx = 0; rankIdx < arg->rankSize; rankIdx++) { // ?
        if (rankIdx != arg->rootId) {
            dst[curId].addr  = ctx.output[rankIdx];
            dst[curId].token = ctx.token[rankIdx];
            curId++;
        } else {
            localdst.addr = ctx.output[rankIdx];
            localdst.addr += ctx.offset;
            localdst.token = ctx.token[rankIdx];
        }
    }
    dst[arg->rankSize - 1].addr = ctx.output[arg->rankId];
    dst[arg->rankSize - 1].token = ctx.token[arg->rankId];
    CCU_CHK_RET(GroupBroadcast(ctx, arg->channels, arg->channelCount, localdst, dst, src, ctx.groupOpSize));
    HCCL_INFO("[CcuKernelBroadcastMesh1D] BroadcastMesh1D GroupBroadcast end");
    return CCU_SUCCESS;
}

CcuResult CcuBroadcastMesh1DKernel(CcuKernelArg arg)
{
    auto *kernelArg = static_cast<CcuKernelArgBroadcastMesh1D *>(arg);

    BroadcastMesh1DContext ctx;
    ctx.resourceAllocated = false;
    ctx.moConfig.msInterleave = 0;
    ctx.moConfig.loopCount = 0;
    ctx.moConfig.memSlice = 0;
    ctx.moRes.eventCount = 0;
    ctx.moRes.bufCount = 0;
    ctx.enginePool = 0;

    HCCL_INFO("[CcuKernelBroadcastMesh1D] BroadcastMesh1D run");
    CCU_CHK_RET(ParseKernelArg(ctx, kernelArg));
    CCU_CHK_RET(InitResource(ctx));
    CCU_CHK_RET(LoadArgs(ctx));
    CCU_CHK_RET(PreSync(ctx));
    if (ctx.arg->rankId == ctx.arg->rootId) {
        CCU_CHK_RET(DoBroadcast(ctx));
    }
    CCU_CHK_RET(PostSync(ctx));
    HCCL_INFO("[CcuKernelBroadcastMesh1D] BroadcastMesh1D end");
    return CCU_SUCCESS;
}

} // namespace ops_hccl