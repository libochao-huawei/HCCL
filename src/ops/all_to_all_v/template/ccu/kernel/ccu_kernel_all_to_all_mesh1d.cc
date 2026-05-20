/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "ccu_kernel_all_to_all_mesh1d.h"

namespace ops_hccl {

constexpr int INPUT_XN_ID  = 0;
constexpr int OUTPUT_XN_ID = 1;
constexpr int TOKEN_XN_ID  = 2;
constexpr int CKE_IDX_0    = 0;
constexpr int CKE_IDX_1    = 1;
constexpr int CKE_IDX_2    = 2;

constexpr uint64_t LOCAL_COPY_MS = 8;

static CcuResult ParseKernelArg(AlltoAllMesh1DContext &ctx, CcuKernelArgAlltoAllMesh1D *kernelArg)
{
    // ctx.rankId          = kernelArg->rankId;
    // ctx.rankSize        = kernelArg->rankSize;
    // ctx.channels       = kernelArg->channels;
    return CCU_SUCCESS;
}

static CcuResult InitResource(AlltoAllMesh1DContext &ctx)
{
    const auto *arg = ctx.arg;
    uint32_t channelIdx = 0;

    if (arg->channelCount == 0) {
        HCCL_ERROR("[CcuKernelAlltoAllMesh1D] channels is empty!");
        return CcuResult::CCU_E_INTERNAL;
    }

    // 按照rank号从小到大遍历channels，遇到本rank就填充本地资源，否则依次取远端资源，要求算法返回的Link同样是按顺序排列的
    ctx.input.resize(arg->rankSize);
    ctx.output.resize(arg->rankSize);
    ctx.token.resize(arg->rankSize);
    for (uint64_t peerId = 0; peerId < arg->rankSize; peerId++) {
        if (peerId != arg->rankId) {
            ctx.input[peerId] = ccu::GetResByChannel(arg->channels[channelIdx], INPUT_XN_ID);
            ctx.output[peerId] = ccu::GetResByChannel(arg->channels[channelIdx], OUTPUT_XN_ID);
            ctx.token[peerId] = ccu::GetResByChannel(arg->channels[channelIdx], TOKEN_XN_ID);
            channelIdx++;
        }
    }

    CCU_CHK_RET(ccu::CreateLoopExecutor(&ctx.enginePool, MAX_RANK_SIZE));

    ctx.resourceAllocated = false;
    ctx.loopRegistered    = false;

    return CCU_SUCCESS;
}

static CcuResult LoadArgs(AlltoAllMesh1DContext &ctx)
{
    const auto *arg = ctx.arg;

    CCU_CHK_RET(ccu::LoadArg(ctx.input[arg->rankId], 0));
    CCU_CHK_RET(ccu::LoadArg(ctx.output[arg->rankId], 1));
    CCU_CHK_RET(ccu::LoadArg(ctx.token[arg->rankId], 2));
    CCU_CHK_RET(ccu::LoadArg(ctx.sliceSize, 3));
    CCU_CHK_RET(ccu::LoadArg(ctx.srcStride, 4));
    CCU_CHK_RET(ccu::LoadArg(ctx.srcOffset, 5));
    CCU_CHK_RET(ccu::LoadArg(ctx.dstOffset, 6));
    CCU_CHK_RET(ccu::LoadArg(ctx.goSize.addrOffset, 7));
    CCU_CHK_RET(ccu::LoadArg(ctx.goSize.loopParam, 8));
    CCU_CHK_RET(ccu::LoadArg(ctx.goSize.parallelParam, 9));
    CCU_CHK_RET(ccu::LoadArg(ctx.goSize.residual, 10));

    ctx.srcOffset += ctx.input[arg->rankId];

    return CCU_SUCCESS;
}

static void PreSync(AlltoAllMesh1DContext &ctx)
{
    HCCL_INFO("[CcuKernelAlltoAllMesh1D] AlltoAllMesh1D PreSync begin.");
    const auto *arg = ctx.arg;

    for (uint32_t i = 0; i < arg->channelCount; i++) {
        ccu::WriteVariableWithNotify(arg->channels[i], ctx.output[arg->rankId],
            OUTPUT_XN_ID, CKE_IDX_0, 1 << OUTPUT_XN_ID);
        ccu::WriteVariableWithNotify(arg->channels[i], ctx.token[arg->rankId],
            TOKEN_XN_ID, CKE_IDX_0, 1 << TOKEN_XN_ID);
    }

    uint32_t allBit = (1 << OUTPUT_XN_ID) | (1 << TOKEN_XN_ID);
    for (uint32_t i = 0; i < arg->channelCount; i++) {
        ccu::NotifyWait(arg->channels[i], CKE_IDX_0, allBit);
    }
    HCCL_INFO("[CcuKernelAlltoAllMesh1D] AlltoAllMesh1D PreSync end.");
}

static void PostSync(AlltoAllMesh1DContext &ctx)
{
    HCCL_INFO("[CcuKernelAlltoAllMesh1D] AlltoAllMesh1D PostSync begin.");
    const auto *arg = ctx.arg;

    uint16_t postBit = 1 << 5;
    for (uint32_t i = 0; i < arg->channelCount; i++) {
        ccu::NotifyRecord(arg->channels[i], CKE_IDX_1, postBit);
    }
    for (uint32_t i = 0; i < arg->channelCount; i++) {
        ccu::NotifyWait(arg->channels[i], CKE_IDX_1, postBit);
    }
    HCCL_INFO("[CcuKernelAlltoAllMesh1D] AlltoAllMesh1D PostSync end.");
}

static CcuResult DoAlltoAll(AlltoAllMesh1DContext &ctx)
{
    HCCL_INFO("DoAlltoAll Start.");
    const auto *arg = ctx.arg;
    std::vector<ccu::LocalAddr> src(arg->rankSize);
    std::vector<ccu::RemoteAddr> dst(arg->rankSize);
    ccu::LocalAddr localDst;

    for (uint32_t rankIdx = 0; rankIdx < arg->rankSize; rankIdx++) {
        if (rankIdx != arg->rankId) {
            dst[rankIdx].token = ctx.token[rankIdx];
            dst[rankIdx].addr = ctx.output[rankIdx];
            dst[rankIdx].addr += ctx.dstOffset;
        } else {
            localDst.token = ctx.token[rankIdx];
            localDst.addr = ctx.output[rankIdx];
            localDst.addr += ctx.dstOffset;
        }

        src[rankIdx].addr = ctx.srcOffset;
        src[rankIdx].token = ctx.token[rankIdx];
        for (uint64_t i = 0; i < rankIdx; i++) {
            src[rankIdx].addr += ctx.srcStride;
        }
    }

    uint32_t channelId = 0;
    uint16_t allBit = ((1 << arg->rankSize) - 1) & (~(1 << arg->rankId)); // 仅rankid位为0，其他位为1，代表远端准备好了

    if (arg->loadFromMem) {
        for(uint64_t r = 0; r < arg->rankSize; r++) {
            if (r == arg->rankId) {
                ccu::LocalCopy(localDst, src[r], ctx.sliceSize, ctx.event, 1 << r);
            }
            else {
                ccu::Write(arg->channels[channelId], dst[r], src[r], ctx.sliceSize, ctx.event, 1 << r);
                channelId++;
            }
        }
        // 等读完所有对端
        ccu::EventWait(ctx.event, (1 << arg->rankSize) - 1);
    } else {
        for(uint64_t r = 0; r < arg->rankSize; r++) {
            if (r != arg->rankId) {
                ccu::Write(arg->channels[channelId], dst[r], src[r], ctx.sliceSize, ctx.event, 1 << r);
                channelId++;
            }
        }
        GroupCopy(ctx, localDst, src[arg->rankId], ctx.goSize);
        ccu::EventWait(ctx.event, allBit);
    }

    return CCU_SUCCESS;
}

// ============================================================================
// 主入口 Kernel 函数
// ============================================================================
CcuResult CcuAlltoAllMesh1DKernel(CcuKernelArg arg)
{
    auto *kernelArg = static_cast<CcuKernelArgAlltoAllMesh1D *>(arg);

    AlltoAllMesh1DContext ctx;
    ctx.arg = kernelArg;
    ctx.resourceAllocated = false;
    ctx.loopRegistered = false;
    ctx.moConfig.msInterleave = 0;
    ctx.moConfig.loopCount = 0;
    ctx.moConfig.memSlice = 0;
    ctx.moRes.eventCount = 0;
    ctx.moRes.bufCount = 0;
    ctx.enginePool = 0;

    HCCL_INFO("[CcuKernelAlltoAllMesh1D] AlltoAllMesh1D run");
    CCU_CHK_RET(ParseKernelArg(ctx, kernelArg));
    CCU_CHK_RET(InitResource(ctx));
    CCU_CHK_RET(LoadArgs(ctx));

    PreSync(ctx);

    CCU_CHK_RET(DoAlltoAll(ctx));


    PostSync(ctx);
    HCCL_INFO("[CcuKernelAlltoAllMesh1D] AlltoAllMesh1D end");

    return CCU_SUCCESS;
}

}// namespace ops_hccl