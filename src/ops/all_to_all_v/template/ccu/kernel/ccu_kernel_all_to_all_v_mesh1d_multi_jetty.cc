/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "alg_template_base.h"
#include "ccu_kernel_all_to_all_v_mesh1d_multi_jetty.h"

namespace ops_hccl {
using namespace hcomm;

constexpr int OUTPUT_XN_ID = 1;
constexpr int TOKEN_XN_ID  = 2;
constexpr int POST_SYNC_ID  = 3;
constexpr int CKE_IDX_0    = 0;

static CcuResult PreSync(AllToAllVMesh1DMultiJettyContext &ctx)
{
    const auto& arg = ctx.arg;
    ccu::Variable tempDst;
    u32 channelIdx = 0;
    for (u32 id = 0; id < arg->rankSize; id++) {
        if (id == arg->rankId) {
            continue;
        }
        tempDst = ctx.output[arg->rankId];
        tempDst += ctx.sendRecvInfo[id].recvOffset;
        CHK_RET(ccu::NotifyRecord(ctx.channels[channelIdx], CKE_IDX_0, OUTPUT_XN_ID, tempDst, 1 << OUTPUT_XN_ID));
        CHK_RET(ccu::NotifyRecord(ctx.channels[channelIdx], CKE_IDX_0, TOKEN_XN_ID, ctx.token[arg->rankId], 1 << TOKEN_XN_ID));
        channelIdx++;
    }

    uint32_t allBit = 1 << OUTPUT_XN_ID | 1 << TOKEN_XN_ID;
    for (const auto& ch : ctx.channels) {
        CHK_RET(ccu::NotifyWait(ch, CKE_IDX_0, allBit));
    }
    return CCU_SUCCESS;
}

static CcuResult PostSync(AllToAllVMesh1DMultiJettyContext &ctx)
{
    for (const auto& ch : ctx.channels) {
        CHK_RET(ccu::NotifyRecord(ch, CKE_IDX_0, 1 << POST_SYNC_ID));
    }
    for (const auto& ch : ctx.channels) {
        CHK_RET(ccu::NotifyWait(ch, CKE_IDX_0, 1 << POST_SYNC_ID));
    }
    return CCU_SUCCESS;
}

static CcuResult InitResource(AllToAllVMesh1DMultiJettyContext &ctx)
{
    const auto& arg = ctx.arg;
    u32 channelIdx = 0;
    if (ctx.channels.empty()) {
        HCCL_ERROR("[CcuKernelAllToAllVMesh1DMultiJetty] channels is empty!");
        return CCU_E_INTERNAL;
    }
    ctx.input.push_back(ccu::CreateVariable());
    ctx.output.reserve(arg->rankSize);
    ctx.token.reserve(arg->rankSize);
    for (u32 id = 0; id < arg->rankSize; id++) {
        if (id == arg->rankId) {
            ctx.output.push_back(ccu::CreateVariable());
            ctx.token.push_back(ccu::CreateVariable());
        } else {
            HCCL_DEBUG("[CcuKernelAllToAllVMesh1DMultiJetty] MyRank[%u], PeerId[%u], ChannelId[%u]",
                arg->rankId, id, channelIdx);
            ctx.output.push_back(ccu::GetResByChannel<ccu::Variable>(ctx.channels[channelIdx], OUTPUT_XN_ID));
            ctx.token.push_back(ccu::GetResByChannel<ccu::Variable>(ctx.channels[channelIdx], TOKEN_XN_ID));
            channelIdx++;
        }
    }

    ctx.src.reserve(arg->rankSize);
    ctx.remoteDst.reserve(arg->rankSize);
    for (uint32_t rankIdx = 0; rankIdx < arg->rankSize; rankIdx++) {
        ctx.src.push_back(ccu::CreateLocalAddr());
        if (rankIdx == arg->rankId) {
            ctx.myDst = ccu::CreateLocalAddr();
            ctx.remoteDst.push_back({});
        } else {
            ctx.remoteDst.push_back(ccu::CreateRemoteAddr());
        }
    }

    ctx.srcOffset = ccu::CreateVariable();
    ctx.dstOffset = ccu::CreateVariable();

    ctx.completedRankCount = ccu::CreateVariable();
    ctx.xnMaxTransportSize = ccu::CreateVariable();
    ctx.xnMaxTransportGoSize = ccu::CreateGroupOpSize();
    ctx.xnConst1 = ccu::CreateVariable();

    ctx.eventList.reserve(arg->rankSize);
    for (uint32_t rankIdx = 0; rankIdx < arg->rankSize; rankIdx++) {
        ctx.eventList.push_back(ccu::CreateCompletedEvent());
    }
    return CCU_SUCCESS;
}

static CcuResult LoadArgs(AllToAllVMesh1DMultiJettyContext &ctx)
{
    const auto& arg = ctx.arg;
    ccu::Load(ctx.input[0]);
    ccu::Load(ctx.output[arg->rankId]);
    ccu::Load(ctx.token[arg->rankId]);
    ccu::Load(ctx.srcOffset);
    ccu::Load(ctx.dstOffset);
    ccu::Load(ctx.xnMaxTransportGoSize);

    ctx.sendRecvInfo.resize(arg->rankSize);
    for (uint64_t rankIdx = 0; rankIdx < arg->rankSize; rankIdx++) {
        ctx.sendRecvInfo[rankIdx].sliceSize = ccu::CreateVariable();
        ctx.sendRecvInfo[rankIdx].tailSliceSize = ccu::CreateVariable();
        ctx.sendRecvInfo[rankIdx].lastSliceSize = ccu::CreateVariable();
        ctx.sendRecvInfo[rankIdx].lastTailSliceSize = ccu::CreateVariable();
        ctx.sendRecvInfo[rankIdx].loopNum = ccu::CreateVariable();
        ctx.sendRecvInfo[rankIdx].sendOffset = ccu::CreateVariable();
        ctx.sendRecvInfo[rankIdx].recvOffset = ccu::CreateVariable();
        ctx.sendRecvInfo[rankIdx].tailGoSize = ccu::CreateGroupOpSize();

        ccu::Load(ctx.sendRecvInfo[rankIdx].sliceSize);
        ccu::Load(ctx.sendRecvInfo[rankIdx].tailSliceSize);
        ccu::Load(ctx.sendRecvInfo[rankIdx].lastSliceSize);
        ccu::Load(ctx.sendRecvInfo[rankIdx].lastTailSliceSize);
        ccu::Load(ctx.sendRecvInfo[rankIdx].loopNum);
        ccu::Load(ctx.sendRecvInfo[rankIdx].sendOffset);
        ccu::Load(ctx.sendRecvInfo[rankIdx].recvOffset);
        ccu::Load(ctx.sendRecvInfo[rankIdx].tailGoSize);
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
            ctx.remoteDst[rankIdx].addr += ctx.dstOffset;
        }
    }
    return CCU_SUCCESS;
}

static CcuResult DoAll2AllVLastBlock(AllToAllVMesh1DMultiJettyContext &ctx, u32 rankIdx, u32 channelIdx)
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

static CcuResult DoAll2AllVBlock(AllToAllVMesh1DMultiJettyContext &ctx, u32 rankIdx, u32 channelIdx)
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
    u32 channelIdx = 0;
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
                    CHK_RET(ccu::GroupCopy(ctx.myDst, ctx.src[arg->rankId], ctx.sendRecvInfo[arg->rankId].tailGoSize));
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
                    CHK_RET(ccu::GroupCopy(ctx.myDst, ctx.src[arg->rankId], ctx.xnMaxTransportGoSize));
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

    CHK_RET(InitResource(*ctx));
    CHK_RET(LoadArgs(*ctx));
    CHK_RET(PreSync(*ctx));
    CHK_RET(CalcGroupSrcDst(*ctx));
    CHK_RET(DoAll2AllVMultiLoop(*ctx));
    CHK_RET(PostSync(*ctx));

    HCCL_INFO("[AllToAllVAlgo] AllToAllVMesh1DMultiJetty end");
    return CCU_SUCCESS;
}

} // namespace ops_hccl