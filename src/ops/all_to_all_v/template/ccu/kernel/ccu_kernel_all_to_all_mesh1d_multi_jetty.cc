/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "ccu_kernel_all_to_all_mesh1d_multi_jetty.h"
#include "ccu_kernel_alg_base.h"
#include "ccu_control_api.h"

namespace ops_hccl {

constexpr int INPUT_XN_ID   = 0;
constexpr int OUTPUT_XN_ID = 1;
constexpr int TOKEN_XN_ID   = 2;
constexpr int POST_SYNC_ID = 3;
constexpr int CKE_IDX_0     = 0;

static CcuResult InitResource(AllToAllMesh1DMultiJettyContext &ctx)
{
    const auto *arg = ctx.arg;
    uint32_t channelIdx = 0;

    if (arg->channelCount == 0) {
        HCCL_ERROR("[CcuKernelAllToAllMesh1DMultiJetty] channels is empty!");
        return CcuResult::CCU_E_INTERNAL;
    }

    ctx.peerInput.resize(arg->rankSize);
    ctx.peerOutput.resize(arg->rankSize);
    ctx.peerToken.resize(arg->rankSize);

    for (uint64_t peerId = 0; peerId < arg->rankSize; peerId++) {
        if (peerId == arg->rankId) {
            ctx.peerInput[peerId] = ccu::CreateVariable();
            ctx.peerOutput[peerId] = ccu::CreateVariable();
            ctx.peerToken[peerId] = ccu::CreateVariable();
        } else {
            HCCL_DEBUG("[CcuKernelAllToAllMesh1DMultiJetty] MyRank[%u], PeerId[%u], ChannelId[%u]",
                       arg->rankId, peerId, channelIdx);
            ctx.peerInput[peerId] = ccu::GetResByChannel<ccu::Variable>(arg->channels[channelIdx], INPUT_XN_ID);
            ctx.peerOutput[peerId] = ccu::GetResByChannel<ccu::Variable>(arg->channels[channelIdx], OUTPUT_XN_ID);
            ctx.peerToken[peerId] = ccu::GetResByChannel<ccu::Variable>(arg->channels[channelIdx], TOKEN_XN_ID);
            channelIdx++;
        }
    }

    ctx.sliceSize   = ccu::CreateVariable();
    ctx.srcStride   = ccu::CreateVariable();
    ctx.srcOffset   = ccu::CreateVariable();
    ctx.dstOffset   = ccu::CreateVariable();

    ctx.eventList.resize(arg->rankSize);
    ctx.jettySlice.resize(arg->rankSize);
    ctx.jettySliceTail.resize(arg->rankSize);
    for (uint64_t peerId = 0; peerId < arg->rankSize; peerId++) {
        ctx.eventList[peerId] = ccu::CreateEvent();
        ctx.jettySlice[peerId] = ccu::CreateVariable();
        ctx.jettySliceTail[peerId] = ccu::CreateVariable();
    }

    ctx.resourceAllocated = false;
    HCCL_INFO("[CcuKernelAllToAllMesh1DMultiJetty] InitResource success!");
    return CCU_SUCCESS;
}

static CcuResult LoadArgs(AllToAllMesh1DMultiJettyContext &ctx)
{
    const auto *arg = ctx.arg;
    uint32_t argId = 0;

    CCU_CHK_RET(ccu::LoadArg(ctx.peerInput[arg->rankId], argId++));
    CCU_CHK_RET(ccu::LoadArg(ctx.peerOutput[arg->rankId], argId++));
    CCU_CHK_RET(ccu::LoadArg(ctx.peerToken[arg->rankId], argId++));
    CCU_CHK_RET(ccu::LoadArg(ctx.sliceSize, argId++));
    CCU_CHK_RET(ccu::LoadArg(ctx.srcStride, argId++));
    CCU_CHK_RET(ccu::LoadArg(ctx.srcOffset, argId++));
    CCU_CHK_RET(ccu::LoadArg(ctx.dstOffset, argId++));
    CCU_CHK_RET(ccu::LoadArg(ctx.goSize.addrOffset, argId++));
    CCU_CHK_RET(ccu::LoadArg(ctx.goSize.loopParam, argId++));
    CCU_CHK_RET(ccu::LoadArg(ctx.goSize.parallelParam, argId++));
    CCU_CHK_RET(ccu::LoadArg(ctx.goSize.residual, argId++));

    for (uint64_t i = 0; i < arg->rankSize; i++) {
        CCU_CHK_RET(ccu::LoadArg(ctx.jettySlice[i], argId++));
    }
    for (uint64_t i = 0; i < arg->rankSize; i++) {
        CCU_CHK_RET(ccu::LoadArg(ctx.jettySliceTail[i], argId++));
    }

    HCCL_INFO("[CcuKernelAllToAllMesh1DMultiJetty] LoadArgs success, total argId=%u!", argId);
    return CCU_SUCCESS;
}

static CcuResult PreSync(AllToAllMesh1DMultiJettyContext &ctx)
{
    const auto *arg = ctx.arg;

    for (uint32_t i = 0; i < arg->channelCount; i++) {
        CCU_CHK_RET(ccu::WriteVariableWithNotify(arg->channels[i], ctx.peerInput[arg->rankId],
            INPUT_XN_ID, CKE_IDX_0, 1 << INPUT_XN_ID));
        CCU_CHK_RET(ccu::WriteVariableWithNotify(arg->channels[i], ctx.peerOutput[arg->rankId],
            OUTPUT_XN_ID, CKE_IDX_0, 1 << OUTPUT_XN_ID));
        CCU_CHK_RET(ccu::WriteVariableWithNotify(arg->channels[i], ctx.peerToken[arg->rankId],
            TOKEN_XN_ID, CKE_IDX_0, 1 << TOKEN_XN_ID));
    }

    uint32_t allBit = (1 << INPUT_XN_ID) | (1 << OUTPUT_XN_ID) | (1 << TOKEN_XN_ID);
    for (uint32_t i = 0; i < arg->channelCount; i++) {
        CCU_CHK_RET(ccu::NotifyWait(arg->channels[i], CKE_IDX_0, allBit));
    }

    HCCL_INFO("[CcuKernelAllToAllMesh1DMultiJetty] PreSync success!");
    return CCU_SUCCESS;
}

static CcuResult DoAllToAll(AllToAllMesh1DMultiJettyContext &ctx)
{
    const auto *arg = ctx.arg;

    std::vector<ccu::LocalAddr> remoteSrc(arg->rankSize);
    std::vector<ccu::RemoteAddr> remoteDst(arg->rankSize);
    ccu::LocalAddr localSrc;
    ccu::LocalAddr localDst;

    for (uint32_t rankIdx = 0; rankIdx < arg->rankSize; rankIdx++) {
        if (rankIdx == arg->rankId) {
            localSrc.addr = ctx.srcOffset;
            for (uint32_t i = 0; i < rankIdx; i++) {
                localSrc.addr += ctx.srcStride;
            }
            localSrc.token = ctx.peerToken[rankIdx];
            localDst.addr = ctx.peerOutput[rankIdx];
            localDst.token = ctx.peerToken[rankIdx];
            localDst.addr += ctx.dstOffset;
        } else {
            remoteSrc[rankIdx].addr = ctx.srcOffset;
            for (uint32_t i = 0; i < rankIdx; i++) {
                remoteSrc[rankIdx].addr += ctx.srcStride;
            }
            remoteSrc[rankIdx].token = ctx.peerToken[rankIdx];
            remoteDst[rankIdx].addr = ctx.peerOutput[rankIdx];
            remoteDst[rankIdx].token = ctx.peerToken[rankIdx];
            remoteDst[rankIdx].addr += ctx.dstOffset;
        }
    }

    uint32_t channelId = 0;
    for (uint64_t r = 0; r < arg->rankSize; r++) {
        if (r == arg->rankId) {
            continue;
        }
        for (uint32_t jettyIdx = 0; jettyIdx < arg->jettyNums[r]; jettyIdx++) {
            ctx.eventList[r].mask = 1 << jettyIdx;
            if (jettyIdx == (arg->jettyNums[r] - 1)) {
                CCU_IF(ctx.jettySliceTail[r] != 0) {
                    CCU_CHK_RET(ccu::WriteNb(arg->channels[channelId], remoteDst[r], remoteSrc[r],
                                             ctx.jettySliceTail[r], ctx.eventList[r]));
                }
                CCU_IF(ctx.jettySliceTail[r] == 0) {
                    CCU_CHK_RET(ccu::RecordEvent(ctx.eventList[r]));
                }
            } else {
                CCU_IF(ctx.jettySlice[r] != 0) {
                    CCU_CHK_RET(ccu::WriteNb(arg->channels[channelId], remoteDst[r], remoteSrc[r],
                                             ctx.jettySlice[r], ctx.eventList[r]));
                }
                CCU_IF(ctx.jettySlice[r] == 0) {
                    CCU_CHK_RET(ccu::RecordEvent(ctx.eventList[r]));
                }
            }
            remoteDst[r].addr += ctx.jettySlice[r];
            remoteSrc[r].addr += ctx.jettySlice[r];
        }
        channelId++;
    }

    CCU_CHK_RET(GroupCopy(ctx, localDst, localSrc, ctx.goSize));
    ctx.eventList[arg->rankId].mask = 1;
    CCU_CHK_RET(ccu::RecordEvent(ctx.eventList[arg->rankId]));

    for (uint64_t r = 0; r < arg->rankSize; r++) {
        if (r == arg->rankId) {
            ctx.eventList[r].mask = 1;
        } else {
            ctx.eventList[r].mask = (1 << arg->jettyNums[r]) - 1;
        }
        CCU_CHK_RET(ccu::WaitEvent(ctx.eventList[r]));
    }

    HCCL_INFO("[CcuKernelAllToAllMesh1DMultiJetty] DoAllToAll success!");
    return CCU_SUCCESS;
}

static CcuResult PostSync(AllToAllMesh1DMultiJettyContext &ctx)
{
    const auto *arg = ctx.arg;

    for (uint32_t i = 0; i < arg->channelCount; i++) {
        CCU_CHK_RET(ccu::NotifyRecord(arg->channels[i], CKE_IDX_0, 1 << POST_SYNC_ID));
    }
    for (uint32_t i = 0; i < arg->channelCount; i++) {
        CCU_CHK_RET(ccu::NotifyWait(arg->channels[i], CKE_IDX_0, 1 << POST_SYNC_ID));
    }

    HCCL_INFO("[CcuKernelAllToAllMesh1DMultiJetty] PostSync success!");
    return CCU_SUCCESS;
}

CcuResult CcuAllToAllMesh1DMultiJettyKernel(CcuKernelArg arg)
{
    auto *kernelArg = static_cast<CcuKernelArgAllToAllMesh1DMultiJetty *>(arg);

    AllToAllMesh1DMultiJettyContext ctx;
    ctx.arg = kernelArg;
    ctx.resourceAllocated = false;
    ctx.moConfig.msInterleave = 0;
    ctx.moConfig.loopCount = 0;
    ctx.moConfig.memSlice = 0;
    ctx.moRes.eventCount = 0;
    ctx.moRes.bufCount = 0;
    ctx.enginePool = 0;

    HCCL_INFO("[CcuKernelAllToAllMesh1DMultiJetty] AllToAllMesh1DMultiJetty run, rankSize=%llu, rankId=%u",
              kernelArg->rankSize, kernelArg->rankId);
    CCU_CHK_RET(InitResource(ctx));
    CCU_CHK_RET(LoadArgs(ctx));
    CCU_CHK_RET(PreSync(ctx));
    CCU_CHK_RET(DoAllToAll(ctx));
    CCU_CHK_RET(PostSync(ctx));
    HCCL_INFO("[CcuKernelAllToAllMesh1DMultiJetty] AllToAllMesh1DMultiJetty end");

    return CCU_SUCCESS;
}

} // namespace ops_hccl
