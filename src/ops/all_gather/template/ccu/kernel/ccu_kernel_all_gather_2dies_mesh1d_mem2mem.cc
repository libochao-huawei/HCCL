/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "ccu_kernel_all_gather_2dies_mesh1d_mem2mem.h"
#include "ccu_kernel_utils.h"

namespace ops_hccl {

constexpr int OUTPUT_XN_ID = 1;
constexpr int TOKEN_XN_ID = 2;
constexpr int CKE_IDX_0 = 0;
constexpr int POST_SYNC_ID = 3;
constexpr uint16_t BIT_NUM_PER_CKE = 16;

static CcuResult ParseKernelArg(AllGather2DiesMesh1DMem2MemContext &ctx, CcuKernelArgAllGather2DiesMesh1DMem2Mem *kernelArg)
{
    ctx.arg = kernelArg;
    return CCU_SUCCESS;
}

static CcuResult InitResource(AllGather2DiesMesh1DMem2MemContext &ctx)
{
    const auto *arg = ctx.arg;
    uint16_t channelIdx = 0;

    if (arg->channelCount == 0) {
        HCCL_ERROR("[CcuKernelAllGather2DiesMesh1DMem2Mem] channels is empty!");
        return CcuResult::CCU_E_INTERNAL;
    }
    HCCL_INFO("[CcuKernelAllGather2DiesMesh1DMem2Mem] channels.size: [%u]", arg->channelCount);

    ctx.output.resize(arg->dimSize);
    ctx.token.resize(arg->dimSize);

    for (uint64_t peerId = 0; peerId < arg->dimSize; peerId++) {
        if (peerId == arg->rankId) {
            // 本地资源，后续创建
        } else if (peerId != arg->rankIdGroup[channelIdx]) {
            // 该peer不在本kernel的channel组中
        } else {
            // 对端有对应的channel
            ctx.output[peerId] = ccu::GetResByChannel<ccu::Variable>(arg->channels[channelIdx], OUTPUT_XN_ID);
            ctx.token[peerId] = ccu::GetResByChannel<ccu::Variable>(arg->channels[channelIdx], TOKEN_XN_ID);
        }
        if (peerId >= arg->rankIdGroup[channelIdx] && channelIdx < arg->rankIdGroup.size() - 1) {
            channelIdx++;
        }
    }

    const uint32_t eventNum = (arg->dimSize + BIT_NUM_PER_CKE - 1) / BIT_NUM_PER_CKE;
    ctx.events.resize(eventNum);

    ctx.resourceAllocated = false;
    return CCU_SUCCESS;
}

static CcuResult LoadArgs(AllGather2DiesMesh1DMem2MemContext &ctx)
{
    const auto *arg = ctx.arg;
    uint32_t cnt = 0;

    CCU_CHK_RET(ccu::LoadArg(ctx.input, cnt++));
    CCU_CHK_RET(ccu::LoadArg(ctx.output[arg->rankId], cnt++));
    CCU_CHK_RET(ccu::LoadArg(ctx.sliceSize, cnt++));
    CCU_CHK_RET(ccu::LoadArg(ctx.offSet, cnt++));
    CCU_CHK_RET(ccu::LoadArg(ctx.token[arg->rankId], cnt++));
    CCU_CHK_RET(ccu::LoadArg(ctx.isInputOutputEqual, cnt++));
    CCU_CHK_RET(ccu::LoadArg(ctx.goSize.addrOffset, cnt++));
    CCU_CHK_RET(ccu::LoadArg(ctx.goSize.loopParam, cnt++));
    CCU_CHK_RET(ccu::LoadArg(ctx.goSize.parallelParam, cnt++));
    CCU_CHK_RET(ccu::LoadArg(ctx.goSize.residual, cnt++));

    return CCU_SUCCESS;
}

static CcuResult PreSync(AllGather2DiesMesh1DMem2MemContext &ctx)
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

static CcuResult PostSync(AllGather2DiesMesh1DMem2MemContext &ctx)
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

static CcuResult DoAllGather(AllGather2DiesMesh1DMem2MemContext &ctx)
{
    const auto *arg = ctx.arg;

    ccu::LocalAddr src;
    src.addr = ctx.input;
    src.token = ctx.token[arg->rankId];

    ctx.src_loccopy.addr = ctx.input;
    ctx.src_loccopy.token = ctx.token[arg->rankId];

    std::vector<ccu::RemoteAddr> remoteDst;
    remoteDst.resize(arg->rankIdGroup.size());

    for (uint64_t rankIdx = 0; rankIdx < arg->rankIdGroup.size(); rankIdx++) {
        ctx.events[0].SetMask(1 << arg->rankIdGroup[rankIdx]);
        remoteDst[rankIdx].addr = ctx.output[arg->rankIdGroup[rankIdx]];
        remoteDst[rankIdx].addr += ctx.offSet;
        remoteDst[rankIdx].token = ctx.token[arg->rankIdGroup[rankIdx]];
    }

    uint32_t channelId = 0;
    const uint32_t eventNum = (arg->dimSize + BIT_NUM_PER_CKE - 1) / BIT_NUM_PER_CKE;

    for (uint64_t rankIdx = 0; rankIdx < arg->rankIdGroup.size(); rankIdx++) {
        const uint16_t eventIdx = arg->rankIdGroup[rankIdx] / BIT_NUM_PER_CKE;
        const uint16_t rankMask = 1 << (arg->rankIdGroup[rankIdx] % BIT_NUM_PER_CKE);

        CCU_IF(ctx.sliceSize != 0) {
            CCU_CHK_RET(ccu::Write(arg->channels[channelId], remoteDst[rankIdx],
                src, ctx.sliceSize, ctx.events[eventIdx], rankMask));
        }
        CCU_IF(ctx.sliceSize == 0) {
            CCU_CHK_RET(ccu::EventRecord(ctx.events[eventIdx], rankMask));
        }
        channelId++;
    }

    if (arg->ifHandleSelfRank) {
        ctx.localDst.addr = ctx.output[arg->rankId];
        ctx.localDst.addr += ctx.offSet;
        ctx.localDst.token = ctx.token[arg->rankId];

        CCU_IF(ctx.isInputOutputEqual == 0) {
            CCU_CHK_RET(GroupCopy(ctx, ctx.localDst, ctx.src_loccopy, ctx.goSize));
        }
    }

    uint16_t rankMask = 0x0000;
    for (uint64_t rankIdx = 0; rankIdx < arg->rankIdGroup.size(); rankIdx++) {
        rankMask |= (1 << arg->rankIdGroup[rankIdx]);
    }

    for (uint32_t i = 0; i < eventNum; i++) {
        uint32_t sigNum;
        if (arg->dimSize % BIT_NUM_PER_CKE != 0 && i == (eventNum - 1)) {
            sigNum = arg->dimSize % BIT_NUM_PER_CKE;
        } else {
            sigNum = BIT_NUM_PER_CKE;
        }
        uint32_t allBit = (1 << sigNum) - 1;
        CCU_CHK_RET(ccu::EventWait(ctx.events[i], allBit));
    }

    return CCU_SUCCESS;
}

CcuResult CcuAllGather2DiesMesh1DMem2MemKernel(CcuKernelArg arg)
{
    auto *kernelArg = static_cast<CcuKernelArgAllGather2DiesMesh1DMem2Mem *>(arg);

    AllGather2DiesMesh1DMem2MemContext ctx;
    ctx.resourceAllocated = false;
    ctx.moConfig.msInterleave = 0;
    ctx.moConfig.loopCount = 0;
    ctx.moConfig.memSlice = 0;
    ctx.moRes.eventCount = 0;
    ctx.moRes.bufCount = 0;
    ctx.enginePool = 0;

    HCCL_INFO("[CcuKernelAllGather2DiesMesh1DMem2Mem] AllGather2DiesMesh1DMem2Mem run");

    if (kernelArg->rankIdGroup.size() == 0) {
        HCCL_INFO("[CcuKernelAllGather2DiesMesh1DMem2Mem] rankIdGroup is empty, skip.");
        return CCU_SUCCESS;
    }

    CCU_CHK_RET(ParseKernelArg(ctx, kernelArg));
    CCU_CHK_RET(InitResource(ctx));
    CCU_CHK_RET(LoadArgs(ctx));

    CCU_CHK_RET(PreSync(ctx));

    CCU_CHK_RET(DoAllGather(ctx));

    CCU_CHK_RET(PostSync(ctx));

    HCCL_INFO("[CcuKernelAllGather2DiesMesh1DMem2Mem] AllGather2DiesMesh1DMem2Mem end");

    return CCU_SUCCESS;
}

} // namespace ops_hccl