/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "ccu_kernel_all_gather_v_mesh1d_mem2mem.h"

namespace ops_hccl {

constexpr int OUTPUT_XN_ID = 1;
constexpr int TOKEN_XN_ID = 2;
constexpr int CKE_IDX_0 = 0;
constexpr int CKE_IDX_1 = 1;
constexpr uint64_t CCU_MS_SIZE = 4096;
constexpr uint64_t LOCAL_COPY_MS = 8;
constexpr int POST_SYNC_ID = 3;

static CcuResult InitResource(AllGatherVMesh1DMem2MemContext &ctx)
{
    const auto *arg = ctx.arg;
    uint32_t channelIdx = 0;
    if (arg->channelCount == 0) {
        HCCL_ERROR("[CcuKernelAllGatherVMesh1DMem2Mem] channels is empty!");
        return CcuResult::CCU_E_INTERNAL;
    }
    HCCL_INFO("[CcuKernelAllGatherVMesh1DMem2Mem] channels.size: [%u]", arg->channelCount);
    // 按照rank号从小到大遍历channels，遇到本rank就填充本地资源，否则依次取远端资源，要求算法返回的Link同样是按顺序排列的
    ctx.output.resize(arg->rankSize);
    ctx.token.resize(arg->rankSize);
    for (uint64_t peerId = 0; peerId < arg->rankSize; peerId++) {
        if (peerId == arg->rankId) {
            CCU_CHK_RET(ccu::Alloc(&ctx.output[peerId]));
            CCU_CHK_RET(ccu::Alloc(&ctx.token[peerId]));
        } else {
            CCU_CHK_RET(ccu::CreateByChannel(arg->channels[channelIdx], OUTPUT_XN_ID, &ctx.input[peerId]));
            CCU_CHK_RET(ccu::CreateByChannel(arg->channels[channelIdx], TOKEN_XN_ID, &ctx.token[peerId]));
            channelIdx++;
        }
    }
    CCU_CHK_RET(ccu::Alloc(&ctx.input));
    CCU_CHK_RET(ccu::Alloc(&ctx.output));
    CCU_CHK_RET(ccu::Alloc(&ctx.token));
    CCU_CHK_RET(ccu::Alloc(&ctx.sliceSize));
    CCU_CHK_RET(ccu::Alloc(&ctx.mySliceSize));
    CCU_CHK_RET(ccu::Alloc(&ctx.mySliceSizeOutputOffset));
    CCU_CHK_RET(ccu::Alloc(&ctx.localGoSize.addrOffset));
    CCU_CHK_RET(ccu::Alloc(&ctx.localGoSize.loopParam));
    CCU_CHK_RET(ccu::Alloc(&ctx.localGoSize.parallelParam));
    CCU_CHK_RET(ccu::Alloc(&ctx.localGoSize.residual));
    CCU_CHK_RET(ccu::Alloc(&ctx.event));
    ctx.resourceAllocated = true; // base里面的
 	ctx.loopRegistered    = false;
    return CCU_SUCCESS;
}

static CcuResult LoadArgs(AllGatherVMesh1DMem2MemContext &ctx)
{
    const auto *arg = ctx.arg;
    CCU_CHK_RET(ccu::LoadArg(ctx.input));
    CCU_CHK_RET(ccu::LoadArg(ctx.output[arg->rankId]));
    CCU_CHK_RET(ccu::LoadArg(ctx.token[arg->rankId]));
    CCU_CHK_RET(ccu::LoadArg(ctx.mySliceSize));
    CCU_CHK_RET(ccu::LoadArg(ctx.mySliceSizeOutputOffset));
    CCU_CHK_RET(ccu::LoadArg(ctx.localGoSize.addrOffset));
    CCU_CHK_RET(ccu::LoadArg(ctx.localGoSize.loopParam));
    CCU_CHK_RET(ccu::LoadArg(ctx.localGoSize.parallelParam));
    CCU_CHK_RET(ccu::LoadArg(ctx.localGoSize.residual));
    return CCU_SUCCESS;
}

static void PreSync(AllGatherVMesh1DMem2MemContext &ctx)
{
    HCCL_INFO("[CcuKernelAllGatherVMesh1DMem2Mem] AllgatherVMesh1D PreSync begin");
 	const auto *arg = ctx.arg;
    for (uint32_t i = 0; i < arg->channelCount; i++) {
        ccu::WriteVariableWithNotify(
            arg->channels[i], ctx.output[arg->rankId], OUTPUT_XN_ID, CKE_IDX_0, 1 << OUTPUT_XN_ID);
        ccu::WriteVariableWithNotify(
            arg->channels[i], ctx.token[arg->rankId], TOKEN_XN_ID, CKE_IDX_0, 1 << TOKEN_XN_ID);
    }
    uint32_t allBit = (1 << OUTPUT_XN_ID) | (1 << TOKEN_XN_ID);
    for (uint32_t i = 0; i < arg->channelCount; i++) {
 	    ccu::NotifyWait(arg->channels[i], CKE_IDX_0, allBit);
 	}
    HCCL_INFO("[CcuKernelAllGatherVMesh1DMem2Mem] AllgatherVMesh1D PreSync end");
}

static void PostSync(AllGatherVMesh1DMem2MemContext &ctx)
{
    HCCL_INFO("[CcuKernelAllGatherVMesh1DMem2Mem] AllgatherVMesh1D post sync start");
    const auto *arg = ctx.arg;
    for (uint32_t i = 0; i < arg->channelCount; i++) {
        ccu::NotifyRecord(arg->channels[i], CKE_IDX_0, 1 << POST_SYNC_ID);
    }
    for (uint32_t i = 0; i < arg->channelCount; i++) {
        ccu::NotifyWait(arg->channels[i], CKE_IDX_0, 1 << POST_SYNC_ID);
    }
    HCCL_INFO("[CcuKernelAllGatherVMesh1DMem2Mem] AllgatherVMesh1D post sync end");
}

static void CcuKernelAllGatherVMesh1DMem2Mem::DoAllGatherV(AllGatherVMesh1DMem2MemContext &ctx)
{
    const auto *arg = ctx.arg;
    uint32_t channelId = 0;
    CCU_IF(ctx.mySliceSize != 0) {
        for (uint32_t rankIdx = 0; rankIdx < arg->rankSize; rankIdx++) {
            ctx.src.addr = ctx.input;
            ctx.src.token = ctx.token[arg->rankId];
            if (rankIdx == arg->rankId) {
                ctx.localDst.addr = output[arg->rankId];
                ctx.localDst.addr += ctx.mySliceSizeOutputOffset;
                ctx.localDst.token = token[arg->rankId];
                ctx.event.setMask(1 << rankIdx);
                GroupCopy(ctx, ctx.localDst, ctx.src, ctx.localGoSize); // fix
                ccu::RecordEvent(ctx.event);
            } else {
                ctx.dst[rankIdx].addr = ctx.output[rankIdx];
                ctx.dst[rankIdx].addr += ctx.mySliceSizeOutputOffset;
                ctx.dst[rankIdx].token = ctx.token[rankIdx];
                CCU_IF(ctx.mySliceSize != 0)
                {
                    ctx.event.setMask(1 << rankIdx);
                    WriteNb(arg->channels[channelId], ctx.dst[rankIdx], ctx.src, ctx.mySliceSize, ctx.event); // fix
                }
                channelId++;
            }
        }
        ctx.event.SetMask((1 << arg->rankSize) - 1);
        ccu::WaitEvent(ctx.event);
    }
}

CcuResult CcuAllGatherVMesh1DMem2MemKernel(CcuKernelArg arg)
{
    auto *kernelArg = static_cast<CcuKernelArgAllGatherVMesh1DMem2Mem *>(arg);
    AllGatherVMesh1DMem2MemContext ctx;
    ctx.arg = kernelArg;
    HCCL_INFO("[CcuKernelAllGatherVMesh1DMem2Mem] AllgatherVMesh1D run");
    CCU_CHK_RET(InitResource(ctx));
    LoadArgs(ctx);
    PreSync(ctx);
    DoAllGatherV(ctx); // fix
    PostSync(ctx);
    HCCL_INFO("[CcuKernelAllGatherVMesh1DMem2Mem] AllgatherVMesh1D end");

    return CCU_SUCCESS;
}

} // namespace ops_hccl