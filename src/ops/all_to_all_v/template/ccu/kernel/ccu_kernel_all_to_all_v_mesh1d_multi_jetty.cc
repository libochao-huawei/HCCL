/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "ccu_kernel_all_to_all_v_mesh1d_multi_jetty.h"
#include "ccu_control_api.h"

namespace ops_hccl {
using namespace hcomm;

constexpr int OUTPUT_XN_ID = 1;
constexpr int TOKEN_XN_ID  = 2;
constexpr int POST_SYNC_ID  = 3;
constexpr int CKE_IDX_0    = 0;

static CcuResult PreSync(AllToAllVMesh1DMultiJettyContext &ctx)
{
    const auto& arg = ctx.arg;
    uint32_t channelIdx = 0;
    for (uint32_t id = 0; id < arg->rankSize; id++) {
        if (id == arg->rankId) {
            continue;
        }
        ccu::Variable tempDst;
        tempDst = ctx.output[arg->rankId];
        tempDst += ctx.sendRecvInfo[id].recvOffset;
        CCU_CHK_RET(ccu::NotifyRecord(ctx.channels[channelIdx], CKE_IDX_0, OUTPUT_XN_ID, tempDst, 1 << OUTPUT_XN_ID));
        CCU_CHK_RET(ccu::NotifyRecord(ctx.channels[channelIdx], CKE_IDX_0, TOKEN_XN_ID, ctx.token[arg->rankId], 1 << TOKEN_XN_ID));
        channelIdx++;
    }

    uint32_t allBit = (1 << OUTPUT_XN_ID) | (1 << TOKEN_XN_ID);
    for (const auto& ch : ctx.channels) {
        CCU_CHK_RET(ccu::NotifyWait(ch, CKE_IDX_0, allBit));
    }
    return CCU_SUCCESS;
}

static CcuResult PostSync(AllToAllVMesh1DMultiJettyContext &ctx)
{
    for (const auto& ch : ctx.channels) {
        CCU_CHK_RET(ccu::NotifyRecord(ch, CKE_IDX_0, 1 << POST_SYNC_ID));
    }
    for (const auto& ch : ctx.channels) {
        CCU_CHK_RET(ccu::NotifyWait(ch, CKE_IDX_0, 1 << POST_SYNC_ID));
    }
    HCCL_INFO("[CcuKernelAllToAllVMesh1DMultiJetty] AllToAllV GroupWait end");
    return CCU_SUCCESS;
}

static CcuResult InitResource(AllToAllVMesh1DMultiJettyContext &ctx)
{
    const auto& arg = ctx.arg;
    uint32_t channelIdx = 0;

    if (ctx.channels.empty()) {
        HCCL_ERROR("[CcuKernelAllToAllVMesh1DMultiJetty] channels is empty!");
        return CCU_E_INTERNAL;
    }
    HCCL_INFO("[CcuKernelAllToAllVMesh1DMultiJetty] channels.size: [%u]", ctx.channels.size());

    ctx.input.resize(arg->rankSize);
    ctx.output.resize(arg->rankSize);
    ctx.token.resize(arg->rankSize);
    for (uint32_t id = 0; id < arg->rankSize; id++) {
        if (id != arg->rankId) {
            ctx.input[id] = ccu::GetResByChannel<ccu::Variable>(ctx.channels[channelIdx], OUTPUT_XN_ID);
            ctx.output[id] = ccu::GetResByChannel<ccu::Variable>(ctx.channels[channelIdx], OUTPUT_XN_ID);
            ctx.token[id] = ccu::GetResByChannel<ccu::Variable>(ctx.channels[channelIdx], TOKEN_XN_ID);
            channelIdx++;
        }
    }

    ctx.src.resize(arg->rankSize);
    ctx.remoteDst.resize(arg->rankSize);
    ctx.eventList.resize(arg->rankSize);

    return CCU_SUCCESS;
}

static CcuResult LoadArgs(AllToAllVMesh1DMultiJettyContext &ctx)
{
    const auto& arg = ctx.arg;
    uint32_t argId = 0;

    CCU_CHK_RET(ccu::LoadArg(ctx.input[arg->rankId], argId++));
    CCU_CHK_RET(ccu::LoadArg(ctx.output[arg->rankId], argId++));
    CCU_CHK_RET(ccu::LoadArg(ctx.token[arg->rankId], argId++));
    CCU_CHK_RET(ccu::LoadArg(ctx.srcOffset, argId++));
    CCU_CHK_RET(ccu::LoadArg(ctx.dstOffset, argId++));
    CCU_CHK_RET(ccu::LoadArg(ctx.xnMaxTransportGoSize.addrOffset, argId++));
    CCU_CHK_RET(ccu::LoadArg(ctx.xnMaxTransportGoSize.loopParam, argId++));
    CCU_CHK_RET(ccu::LoadArg(ctx.xnMaxTransportGoSize.parallelParam, argId++));
    CCU_CHK_RET(ccu::LoadArg(ctx.xnMaxTransportGoSize.residual, argId++));

    ctx.sendRecvInfo.resize(arg->rankSize);
    for (uint64_t rankIdx = 0; rankIdx < arg->rankSize; rankIdx++) {
        CCU_CHK_RET(ccu::LoadArg(ctx.sendRecvInfo[rankIdx].sliceSize, argId++));
        CCU_CHK_RET(ccu::LoadArg(ctx.sendRecvInfo[rankIdx].tailSliceSize, argId++));
        CCU_CHK_RET(ccu::LoadArg(ctx.sendRecvInfo[rankIdx].lastSliceSize, argId++));
        CCU_CHK_RET(ccu::LoadArg(ctx.sendRecvInfo[rankIdx].lastTailSliceSize, argId++));
        CCU_CHK_RET(ccu::LoadArg(ctx.sendRecvInfo[rankIdx].loopNum, argId++));
        CCU_CHK_RET(ccu::LoadArg(ctx.sendRecvInfo[rankIdx].sendOffset, argId++));
        CCU_CHK_RET(ccu::LoadArg(ctx.sendRecvInfo[rankIdx].recvOffset, argId++));
        CCU_CHK_RET(ccu::LoadArg(ctx.sendRecvInfo[rankIdx].tailGoSize.addrOffset, argId++));
        CCU_CHK_RET(ccu::LoadArg(ctx.sendRecvInfo[rankIdx].tailGoSize.loopParam, argId++));
        CCU_CHK_RET(ccu::LoadArg(ctx.sendRecvInfo[rankIdx].tailGoSize.parallelParam, argId++));
        CCU_CHK_RET(ccu::LoadArg(ctx.sendRecvInfo[rankIdx].tailGoSize.residual, argId++));
    }
    return CCU_SUCCESS;
}

static CcuResult CalcGroupSrcDst(AllToAllVMesh1DMultiJettyContext &ctx)
{
    const auto& arg = ctx.arg;
    for (uint32_t rankIdx = 0; rankIdx < arg->rankSize; rankIdx++) {
        ctx.src[rankIdx].token = ctx.token[rankIdx];
        ctx.src[rankIdx].addr = ctx.input[0];
        ctx.src[rankIdx].addr += ctx.sendRecvInfo[rankIdx].sendOffset;
        ctx.src[rankIdx].addr += ctx.srcOffset;

        if (rankIdx == arg->rankId) {
            ctx.myDst.token = ctx.token[rankIdx];
            ctx.myDst.addr = ctx.output[rankIdx];
            ctx.myDst.addr += ctx.sendRecvInfo[rankIdx].recvOffset;
            ctx.myDst.addr += ctx.dstOffset;
        } else {
            ctx.remoteDst[rankIdx].token = ctx.token[rankIdx];
            ctx.remoteDst[rankIdx].addr = ctx.output[rankIdx];
            ctx.remoteDst[rankIdx].addr += ctx.sendRecvInfo[rankIdx].recvOffset;
            ctx.remoteDst[rankIdx].addr += ctx.dstOffset;
        }
    }
    return CCU_SUCCESS;
}

static CcuResult DoAll2AllVLastBlock(AllToAllVMesh1DMultiJettyContext &ctx, uint32_t rankIdx, uint32_t channelIdx)
{
    const auto& arg = ctx.arg;
    for (uint32_t i = 0; i < arg->jettyNums[rankIdx]; i++) {
        CCU_IF(ctx.sendRecvInfo[rankIdx].loopNum != UINT64_MAX - 1) {
            CCU_IF(ctx.sendRecvInfo[rankIdx].lastSliceSize == 0) {
                ctx.eventList[rankIdx].SetMask(1 << i);
                CHK_RET(ccu::EventRecord(ctx.eventList[rankIdx]));
            }
            CCU_IF(ctx.sendRecvInfo[rankIdx].lastSliceSize != 0) {
                ctx.eventList[rankIdx].SetMask(1 << i);
                CHK_RET(ccu::WriteNb(ctx.channels[channelIdx], ctx.remoteDst[rankIdx], ctx.src[rankIdx], 
                    ctx.sendRecvInfo[rankIdx].lastSliceSize, ctx.eventList[rankIdx]));
                ctx.src[rankIdx].addr += ctx.sendRecvInfo[rankIdx].lastSliceSize;
                ctx.remoteDst[rankIdx].addr += ctx.sendRecvInfo[rankIdx].lastSliceSize;
            }
        }
        CCU_IF(ctx.sendRecvInfo[rankIdx].loopNum == UINT64_MAX - 1) {
            CCU_IF(ctx.sendRecvInfo[rankIdx].lastTailSliceSize == 0) {
                ctx.eventList[rankIdx].SetMask(1 << i);
                CHK_RET(ccu::EventRecord(ctx.eventList[rankIdx]));
            }
            CCU_IF(ctx.sendRecvInfo[rankIdx].lastTailSliceSize != 0) {
                ctx.eventList[rankIdx].SetMask(1 << i);
                CHK_RET(ccu::WriteNb(ctx.channels[channelIdx], ctx.remoteDst[rankIdx], ctx.src[rankIdx], 
                    ctx.sendRecvInfo[rankIdx].lastTailSliceSize, ctx.eventList[rankIdx]));
            }
            ctx.completedRankCount += ctx.xnConst1;
        }
        ctx.sendRecvInfo[rankIdx].loopNum += ctx.xnConst1;
    }
    return CCU_SUCCESS;
}

static CcuResult DoAll2AllVBlock(AllToAllVMesh1DMultiJettyContext &ctx, uint32_t rankIdx, uint32_t channelIdx)
{
    const auto& arg = ctx.arg;
    for (uint32_t i = 0; i < arg->jettyNums[rankIdx]; i++) {
        if (i != arg->jettyNums[rankIdx] - 1) {
            CCU_IF(ctx.sendRecvInfo[rankIdx].sliceSize == 0) {
                ctx.eventList[rankIdx].SetMask(1 << i);
                CHK_RET(ccu::EventRecord(ctx.eventList[rankIdx]));
            }
            CCU_IF(ctx.sendRecvInfo[rankIdx].sliceSize != 0) {
                ctx.eventList[rankIdx].SetMask(1 << i);
                CHK_RET(ccu::WriteNb(ctx.channels[channelIdx], ctx.remoteDst[rankIdx], ctx.src[rankIdx], 
                    ctx.sendRecvInfo[rankIdx].sliceSize, ctx.eventList[rankIdx]));
                ctx.src[rankIdx].addr += ctx.sendRecvInfo[rankIdx].sliceSize;
                ctx.remoteDst[rankIdx].addr += ctx.sendRecvInfo[rankIdx].sliceSize;
            }
        } else {
            CCU_IF(ctx.sendRecvInfo[rankIdx].tailSliceSize == 0) {
                ctx.eventList[rankIdx].SetMask(1 << i);
                CHK_RET(ccu::EventRecord(ctx.eventList[rankIdx]));
            }
            CCU_IF(ctx.sendRecvInfo[rankIdx].tailSliceSize != 0) {
                ctx.eventList[rankIdx].SetMask(1 << i);
                CHK_RET(ccu::WriteNb(ctx.channels[channelIdx], ctx.remoteDst[rankIdx], ctx.src[rankIdx], 
                    ctx.sendRecvInfo[rankIdx].tailSliceSize, ctx.eventList[rankIdx]));
                ctx.src[rankIdx].addr += ctx.sendRecvInfo[rankIdx].tailSliceSize;
                ctx.remoteDst[rankIdx].addr += ctx.sendRecvInfo[rankIdx].tailSliceSize;
            }
        }
        ctx.sendRecvInfo[rankIdx].loopNum += ctx.xnConst1;
    }
    return CCU_SUCCESS;
}

static CcuResult DoAll2AllVMultiLoop(AllToAllVMesh1DMultiJettyContext &ctx)
{
    const auto& arg = ctx.arg;
    HCCL_DEBUG("[CcuKernelAllToAllVMesh1DMultiJetty] alltoallv mesh 1d use GroupCopy start");
    ctx.xnMaxTransportSize = UB_MAX_TRANS_SIZE;
    ctx.completedRankCount = 0;
    ctx.xnConst1 = 1;
    uint32_t channelIdx = 0;
    CCU_WHILE(ctx.completedRankCount != arg->rankSize) {
        for (uint32_t rankIdx = 0; rankIdx < arg->rankSize; rankIdx++) {
            if (rankIdx == arg->rankId) {
                continue;
            }
            CCU_IF(ctx.sendRecvInfo[rankIdx].loopNum == UINT64_MAX) {
                ctx.eventList[rankIdx].SetMask((1 << arg->jettyNums[rankIdx]) - 1);
                CHK_RET(ccu::EventRecord(ctx.eventList[rankIdx]));
            }
            CCU_IF(ctx.sendRecvInfo[rankIdx].loopNum != UINT64_MAX) {
                CCU_IF(ctx.sendRecvInfo[rankIdx].loopNum != UINT64_MAX - arg->jettyNums[rankIdx]) {
                    CHK_RET(DoAll2AllVBlock(ctx, rankIdx, channelIdx));
                }
                CCU_IF(ctx.sendRecvInfo[rankIdx].loopNum == UINT64_MAX - arg->jettyNums[rankIdx]) {
                    CHK_RET(DoAll2AllVLastBlock(ctx, rankIdx, channelIdx));
                }
            }
            channelIdx++;
        }
        CCU_IF(ctx.sendRecvInfo[arg->rankId].loopNum == UINT64_MAX) {
            ctx.eventList[arg->rankId].SetMask((1 << arg->jettyNums[arg->rankId]) - 1);
            CHK_RET(ccu::EventRecord(ctx.eventList[arg->rankId]));
        }

        CCU_IF(ctx.sendRecvInfo[arg->rankId].loopNum != UINT64_MAX) {
            CCU_IF(ctx.sendRecvInfo[arg->rankId].loopNum == UINT64_MAX - arg->jettyNums[arg->rankId]) {
                CCU_IF(ctx.sendRecvInfo[arg->rankId].lastTailSliceSize == 0) {
                    ctx.eventList[arg->rankId].SetMask((1 << arg->jettyNums[arg->rankId]) - 1);
                    CHK_RET(ccu::EventRecord(ctx.eventList[arg->rankId]));
                }
                CCU_IF(ctx.sendRecvInfo[arg->rankId].lastTailSliceSize != 0) {
                    CHK_RET(GroupCopy(ctx, ctx.myDst, ctx.src[arg->rankId], ctx.sendRecvInfo[arg->rankId].tailGoSize));
                    ctx.eventList[arg->rankId].SetMask((1 << arg->jettyNums[arg->rankId]) - 1);
                    CHK_RET(ccu::EventRecord(ctx.eventList[arg->rankId]));
                }
                ctx.completedRankCount += ctx.xnConst1;
            }
            CCU_IF(ctx.sendRecvInfo[arg->rankId].loopNum != UINT64_MAX - arg->jettyNums[arg->rankId]) {
                CCU_IF(ctx.sendRecvInfo[arg->rankId].tailSliceSize == 0) {
                    ctx.eventList[arg->rankId].SetMask((1 << arg->jettyNums[arg->rankId]) - 1);
                    CHK_RET(ccu::EventRecord(ctx.eventList[arg->rankId]));
                }
                CCU_IF(ctx.sendRecvInfo[arg->rankId].tailSliceSize != 0) {
                    CHK_RET(GroupCopy(ctx, ctx.myDst, ctx.src[arg->rankId], ctx.xnMaxTransportGoSize));
                    ctx.eventList[arg->rankId].SetMask((1 << arg->jettyNums[arg->rankId]) - 1);
                    CHK_RET(ccu::EventRecord(ctx.eventList[arg->rankId]));
                    ctx.src[arg->rankId].addr += ctx.xnMaxTransportSize;
                    ctx.myDst.addr += ctx.xnMaxTransportSize;
                }
            }
            ctx.sendRecvInfo[arg->rankId].loopNum += ctx.xnConst1;
        }
        for (uint32_t rankIdx = 0; rankIdx < arg->rankSize; rankIdx++) {
            ctx.eventList[rankIdx].SetMask((1 << arg->jettyNums[rankIdx]) - 1);
            CHK_RET(ccu::EventWait(ctx.eventList[rankIdx]));
        }
    }
    return CCU_SUCCESS;
}

CcuResult CcuAllToAllVMesh1DMultiJettyKernel(CcuKernelCtxBase *ctxBase)
{
    HCCL_INFO("[AllToAllVAlgo] AllToAllVMesh1DMultiJetty run");
    auto ctx = static_cast<AllToAllVMesh1DMultiJettyContext*>(ctxBase);
    const auto& arg = ctx->arg;

    HCCL_INFO("[CcuKernelAllToAllVMesh1DMultiJetty] Init, KernelArgs are rankId[%u], rankSize[%u]",
        arg->rankId, arg->rankSize);

    CCU_CHK_RET(InitResource(*ctx));
    CCU_CHK_RET(LoadArgs(*ctx));
    CCU_CHK_RET(PreSync(*ctx));
    CCU_CHK_RET(CalcGroupSrcDst(*ctx));
    CCU_CHK_RET(DoAll2AllVMultiLoop(*ctx));
    CCU_CHK_RET(PostSync(*ctx));

    HCCL_INFO("[AllToAllVAlgo] AllToAllVMesh1DMultiJetty end");
    return CCU_SUCCESS;
}

} // namespace ops_hccl