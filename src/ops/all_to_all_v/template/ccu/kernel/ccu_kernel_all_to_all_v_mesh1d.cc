/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "ccu_kernel_alg_base.h"
#include "ccu_kernel_all_to_all_v_mesh1d.h"

namespace ops_hccl {
using namespace hcomm;
constexpr int INPUT_XN_ID  = 0;
constexpr int OUTPUT_XN_ID = 1;
constexpr int TOKEN_XN_ID  = 2;
constexpr int CKE_IDX_0    = 0; // pre
constexpr int CKE_IDX_1    = 1; // post
constexpr int CKE_IDX_2    = 2;
constexpr int CONST_ONE    = 1;

static CcuResult ParseKernelArg(AlltoAllVMesh1DContext &ctx, CcuKernelArgAlltoAllVMesh1D *kernelArg)
{
    // ctx.rankId          = kernelArg->rankId;
    // ctx.rankSize        = kernelArg->rankSize;
    // ctx.channels       = kernelArg->channels;
    return CCU_SUCCESS;
}

static CcuResult LoadAll2allSendRecvInfo(AlltoAllVMesh1DContext &ctx, A2AsingleSendRecvInfo &sendRecvInfo)
{
    HCCL_INFO("[CcuKernelAlltoAllVMesh1D] LoadAll2allSendRecvInfo!");
    const auto *arg = ctx.arg;
    CCU_CHK_RET(ccu::Alloc(&sendRecvInfo.tailSize));
    CCU_CHK_RET(ccu::Alloc(&sendRecvInfo.loopNum));
    CCU_CHK_RET(ccu::Alloc(&sendRecvInfo.sendOffset));
    CCU_CHK_RET(ccu::Alloc(&sendRecvInfo.recvOffset));

    CCU_CHK_RET(ccu::Alloc(&sendRecvInfo.tailGoSize.addrOffset));
    CCU_CHK_RET(ccu::Alloc(&sendRecvInfo.tailGoSize.loopParam));
    CCU_CHK_RET(ccu::Alloc(&sendRecvInfo.tailGoSize.parallelParam));
    CCU_CHK_RET(ccu::Alloc(&sendRecvInfo.tailGoSize.residual));

    if (arg->loadFromMem) {
        HCCL_INFO("[CcuKernelAlltoAllVMesh1D] Load Args from Mem");
        sendRecvInfo.loopNum = UINT64_MAX - 1; // MC2 场景 loop num 默认为 1

        // 要求client端排列内存为[size,send,recv][size,send,recv]...
        ccu::LoadVar(ctx.a2avXnAddr, sendRecvInfo.tailSize);
        // sendRecvInfo.tailSize = ctx.a2avXnAddr;
        ctx.a2avXnAddr += ctx.xnLength;

        ccu::LoadVar(ctx.a2avXnAddr, sendRecvInfo.sendOffset);
        // sendRecvInfo.sendOffset = ctx.a2avXnAddr;
        ctx.a2avXnAddr += ctx.xnLength;

        // 跳过recvSize
        ctx.a2avXnAddr += ctx.xnLength;

        ccu::LoadVar(ctx.a2avXnAddr, sendRecvInfo.recvOffset);
        // sendRecvInfo.recvOffset = ctx.a2avXnAddr;
        ctx.a2avXnAddr += ctx.xnLength;
    } else {
        CCU_CHK_RET(ccu::LoadArg(sendRecvInfo.tailSize));
        CCU_CHK_RET(ccu::LoadArg(sendRecvInfo.loopNum));
        CCU_CHK_RET(ccu::LoadArg(sendRecvInfo.sendOffset));
        CCU_CHK_RET(ccu::LoadArg(sendRecvInfo.recvOffset));
        CCU_CHK_RET(ccu::LoadArg(sendRecvInfo.tailGoSize.addrOffset));
        CCU_CHK_RET(ccu::LoadArg(sendRecvInfo.tailGoSize.loopParam));
        CCU_CHK_RET(ccu::LoadArg(sendRecvInfo.tailGoSize.parallelParam));
        CCU_CHK_RET(ccu::LoadArg(sendRecvInfo.tailGoSize.residual));
    }

    return CCU_SUCCESS;
}

static CcuResult InitResource(AlltoAllVMesh1DContext &ctx)
{   
    HCCL_INFO("[CcuKernelAlltoAllVMesh1D] InitResource!");
    const auto *arg = ctx.arg;
    uint32_t channelIdx = 0;
    if (arg->channelCount == 0) {
        HCCL_ERROR("[CcuKernelAlltoAllVMesh1D] channels is empty!");
        return CcuResult::CCU_E_INTERNAL;
    }

    HCCL_INFO("arg->rankSize = %d, arg->rankId = %d", arg->rankSize, arg->rankId);
    ctx.input.resize(arg->rankSize);
    ctx.output.resize(arg->rankSize);
    ctx.token.resize(arg->rankSize);
    for (uint64_t peerId = 0; peerId < arg->rankSize; peerId++) {
        if (peerId == arg->rankId) {
            CCU_CHK_RET(ccu::Alloc(&ctx.input[peerId]));
            CCU_CHK_RET(ccu::Alloc(&ctx.output[peerId]));
            CCU_CHK_RET(ccu::Alloc(&ctx.token[peerId]));
        }
        else { // 非本地，使用远端Variable
            CCU_CHK_RET(ccu::CreateByChannel(arg->channels[channelIdx], INPUT_XN_ID, &ctx.input[peerId]));
            CCU_CHK_RET(ccu::CreateByChannel(arg->channels[channelIdx], OUTPUT_XN_ID, &ctx.output[peerId]));
            CCU_CHK_RET(ccu::CreateByChannel(arg->channels[channelIdx], TOKEN_XN_ID, &ctx.token[peerId]));
            channelIdx++;
        }
    }
    HCCL_INFO("output size: %d, token size: %d", ctx.output.size(), ctx.token.size());

    ctx.src.resize(arg->rankSize);
    ctx.dst.resize(arg->rankSize);
    for (uint64_t peerId = 0; peerId < arg->rankSize; peerId++) {
        CCU_CHK_RET(ccu::Alloc(&ctx.src[peerId]));
        if (peerId == arg->rankId) {
            CCU_CHK_RET(ccu::Alloc(&ctx.myDst));
        } else {
            CCU_CHK_RET(ccu::Alloc(&ctx.dst[peerId]));
        }
    }

    CCU_CHK_RET(ccu::Alloc(&ctx.srcOffset));
    CCU_CHK_RET(ccu::Alloc(&ctx.dstOffset));
    CCU_CHK_RET(ccu::Alloc(&ctx.a2avXnAddr));

    // 前同步。交换信息，将本Rank load的in\out等地址信息写到所有对端的对应Variable中，并同步
    // selfBit_ = 1 << arg->rankId;  // 本rank的mask
    // allBit_  = (1 << arg->rankSize) - 1;  // 等待包含自身的全部对端
    // allOtherBit_ = ((1 << arg->rankSize) - 1) & (~(1 << arg->rankId)); // 等待其他所有对端

    //  all2allv 数据搬运
    CCU_CHK_RET(ccu::Alloc(&ctx.completedRankCount));
    CCU_CHK_RET(ccu::Alloc(&ctx.xnMaxTransportSize));
    CCU_CHK_RET(ccu::Alloc(&ctx.xnMaxTransportGoSize.addrOffset));
    CCU_CHK_RET(ccu::Alloc(&ctx.xnMaxTransportGoSize.loopParam));
    CCU_CHK_RET(ccu::Alloc(&ctx.xnMaxTransportGoSize.parallelParam));
    CCU_CHK_RET(ccu::Alloc(&ctx.xnMaxTransportGoSize.residual));
    CCU_CHK_RET(ccu::Alloc(&ctx.xnConst1));
    CCU_CHK_RET(ccu::Alloc(&ctx.xnLength));
    ctx.xnLength = 8; // xn长度为8byte

    CCU_CHK_RET(ccu::Alloc(&ctx.event));

    return CCU_SUCCESS;
}

static void PreSync(AlltoAllVMesh1DContext &ctx)
{
    HCCL_INFO("[CcuKernelAlltoAllVMesh1D] PreSync!");
    const auto *arg = ctx.arg;
    CcuVariable tempDst;
    CCU_CHK_RET(ccu::Alloc(&tempDst));
    
    u32 channelIdx = 0;
    for (u32 id = 0; id < arg->rankSize; id++) {
        if (id == arg->rankId) {
            continue;
        }
        tempDst = ctx.output[arg->rankId];
        tempDst += ctx.sendRecvInfo[id].recvOffset;
        // index = 0，传递output信息
        ccu::WriteVariableWithNotify(arg->channels[channelIdx], tempDst, CKE_IDX_0, OUTPUT_XN_ID, 1 << OUTPUT_XN_ID);
        // index = 1，传递token信息
        ccu::NotifyRecord(arg->channels[channelIdx], ctx.token[arg->rankId], CKE_IDX_0, TOKEN_XN_ID,  1 << TOKEN_XN_ID);
        channelIdx++;
    }

    uint16_t allBit = 1 << OUTPUT_XN_ID | 1 << TOKEN_XN_ID;
    for (uint32_t i = 0; i < arg->channelCount; i++) {
        ccu::NotifyWait(arg->channels[i], CKE_IDX_0, allBit);
    }
    HCCL_INFO( "[CcuKernelAlltoAllVMesh1D] PreSync end");
}

static void PostSync(AlltoAllVMesh1DContext &ctx)
{
    HCCL_INFO("[CcuKernelAlltoAllVMesh1D] PostSync!");
    const auto *arg = ctx.arg;
    for (uint32_t i = 0; i < arg->channelCount; i++) {
        ccu::NotifyRecord(arg->channels[i], CKE_IDX_1, 1 << CONST_ONE);
    }

    for (uint32_t i = 0; i < arg->channelCount; i++) {
        ccu::NotifyWait(arg->channels[i], CKE_IDX_1, 1 << CONST_ONE);
    }
    HCCL_INFO("[CcuKernelAlltoAllVMesh1D] PostSync End!");
}

static CcuResult LoadArgs(AlltoAllVMesh1DContext &ctx)
{
    HCCL_INFO("[CcuKernelAlltoAllVMesh1D] LoadArgs!");
    // 从SQE load args，本rank需要的input、output地址等信息
    // inputAddr, outputAddr, tokenInfo, srcStride, dstStride, srcOffset, dstOffset
    const auto *arg = ctx.arg;

    CCU_CHK_RET(ccu::LoadArg(ctx.input[0]));
    CCU_CHK_RET(ccu::LoadArg(ctx.output[arg->rankId]));
    CCU_CHK_RET(ccu::LoadArg(ctx.token[arg->rankId]));
    CCU_CHK_RET(ccu::LoadArg(ctx.srcOffset));
    CCU_CHK_RET(ccu::LoadArg(ctx.dstOffset));
    if (arg->loadFromMem) {
        CCU_CHK_RET(ccu::LoadArg(ctx.a2avXnAddr));
    } else {
        CCU_CHK_RET(ccu::LoadArg(ctx.xnMaxTransportGoSize.addrOffset));
        CCU_CHK_RET(ccu::LoadArg(ctx.xnMaxTransportGoSize.loopParam));
        CCU_CHK_RET(ccu::LoadArg(ctx.xnMaxTransportGoSize.parallelParam));
        CCU_CHK_RET(ccu::LoadArg(ctx.xnMaxTransportGoSize.residual));
    }
    // 恢复当前卡对所有卡的收发信息
    ctx.sendRecvInfo.resize(arg->rankSize);
    for (uint64_t peerId = 0; peerId < arg->rankSize; peerId++) {
        CCU_CHK_RET(LoadAll2allSendRecvInfo(ctx, ctx.sendRecvInfo[peerId]));
    }

    return CCU_SUCCESS;
}

static void CalcGroupSrcDst(AlltoAllVMesh1DContext &ctx)
{
    HCCL_INFO("[CcuKernelAlltoAllVMesh1D] CalcGroupSrcDst!");
    const auto *arg = ctx.arg;
    for (uint32_t rankIdx = 0; rankIdx < arg->rankSize; rankIdx++) {
        ctx.src[rankIdx].token = ctx.token[rankIdx];

        // ctx.src[rankIdx] = usrInAddr + sendoffset + srcOffset_
        ctx.src[rankIdx].addr = ctx.input[0];
        ctx.src[rankIdx].addr += ctx.sendRecvInfo[rankIdx].sendOffset;
        ctx.src[rankIdx].addr += ctx.srcOffset;

        // ctx.dst[r] = recvBuf[r] + recvOffset + ctx.dstOffset
        if (rankIdx == arg->rankId) {
            // 写目的端为本端时需要特殊处理：使用接收基地址 + 块地址offset + 已发送数据量
            ctx.myDst.token = ctx.token[rankIdx];
            ctx.myDst.addr = ctx.output[rankIdx];
            ctx.myDst.addr += ctx.sendRecvInfo[rankIdx].recvOffset;
            ctx.myDst.addr += ctx.dstOffset;
        } else {
            // 对端交换的接收块起始地址 + 已接收的数据偏移
            ctx.dst[rankIdx].token = ctx.token[rankIdx];
            ctx.dst[rankIdx].addr = ctx.output[rankIdx];
            ctx.dst[rankIdx].addr += ctx.dstOffset;
        }
    }
}

static CcuResult DoAll2AllVMultiLoop(AlltoAllVMesh1DContext &ctx)
{
    HCCL_DEBUG("[CcuKernelAlltoAllVMesh1D] alltoallv mesh 1d use GroupCopy start");
    const auto *arg = ctx.arg;
    ctx.xnMaxTransportSize = UB_MAX_TRANS_SIZE;
    uint32_t completedRankCount = 0;
    ctx.xnConst1 = 1;
    u32 channelId = 0;
    uint16_t allBit  = (1 << arg->rankSize) - 1;
    CCU_DO_WHILE(completedRankCount != arg->rankSize) {  // 循环发送数据，直到所有对端数据都发送完成
        for(uint32_t rankIdx = 0; rankIdx < arg->rankSize; rankIdx++) {  // 循环发送所有对端数据
            ctx.event.setMask(1 <<rankIdx);
            if (rankIdx == arg->rankId) {
                continue;
            }
            CCU_IF_ONLY(ctx.sendRecvInfo[rankIdx].loopNum == UINT64_MAX) { // 已经完成，直接置位完成信号
                ccu::WaitEvent(ctx.event);
            }
            CCU_IF_ONLY(ctx.sendRecvInfo[rankIdx].loopNum != UINT64_MAX) {  // 还没有完成，则继续循环
                CCU_IF_ONLY(ctx.sendRecvInfo[rankIdx].loopNum == UINT64_MAX - 1) { // 最后一轮循环, 发送尾块数据
                    CCU_IF_ONLY(ctx.sendRecvInfo[rankIdx].tailSize == 0) { // 尾块数据量为 0，则不需要发送尾块数据
                        ccu::WaitEvent(ctx.event);
                    }
                    CCU_IF_ONLY(ctx.sendRecvInfo[rankIdx].tailSize != 0) { // 尾块数据量不为 0，则需要发送尾块数据
                        ccu::WriteNb(arg->channels[channelId], ctx.dst[rankIdx], ctx.src[rankIdx], ctx.sendRecvInfo[rankIdx].tailSize,
                              ctx.event);
                    }
                    completedRankCount += ctx.xnConst1;  // 之后一轮循环完成，更新已完成的rank数
                }
                CCU_IF_ONLY(ctx.sendRecvInfo[rankIdx].loopNum != UINT64_MAX - 1) { // 未完成，则继续循环，发送整块数据
                    ccu::WriteNb(arg->channels[channelId], ctx.dst[rankIdx], ctx.src[rankIdx], ctx.xnMaxTransportSize, ctx.event);
                    // 更新偏移
                    ctx.src[rankIdx].addr += ctx.xnMaxTransportSize;
                    ctx.dst[rankIdx].addr += ctx.xnMaxTransportSize;
                }
                ctx.sendRecvInfo[rankIdx].loopNum += ctx.xnConst1;
            }
                channelId++;
        }

        ctx.event.setMask(1 << arg->rankId);
        CCU_IF_ONLY(ctx.sendRecvInfo[arg->rankId].loopNum == UINT64_MAX) { // 已经完成，直接置位完成信号
                ccu::WaitEvent(ctx.event);
        }

        CCU_IF_ONLY(ctx.sendRecvInfo[arg->rankId].loopNum != UINT64_MAX) {  // 还没有完成，则继续循环
                CCU_IF_ONLY(ctx.sendRecvInfo[arg->rankId].loopNum == UINT64_MAX - 1) { // 最后一轮循环, 发送尾块数据
                    CCU_IF_ONLY(ctx.sendRecvInfo[arg->rankId].tailSize == 0) { // 尾块数据量为 0，则不需要发送尾块数据
                        ccu::WaitEvent(ctx.event);
                    }
                    CCU_IF_ONLY(ctx.sendRecvInfo[arg->rankId].tailSize != 0) { // 尾块数据量不为 0，则需要发送尾块数据
                        if (arg->loadFromMem) {
                            ccu::LocalCopyNb(ctx.myDst, ctx.src[arg->rankId], ctx.sendRecvInfo[arg->rankId].tailSize, ctx.event);
                        } else {
                            GroupCopy(ctx, ctx.myDst, ctx.src[arg->rankId], ctx.sendRecvInfo[arg->rankId].tailGoSize);
                            ccu::WaitEvent(ctx.event);
                        }
                    }
                    completedRankCount += ctx.xnConst1;  // 之后一轮循环完成，更新已完成的rank数
                }
                CCU_IF_ONLY(ctx.sendRecvInfo[arg->rankId].loopNum != UINT64_MAX - 1) { // 未完成，则继续循环，发送整块数据
                    if (arg->loadFromMem) {
                        ccu::LocalCopyNb(ctx.myDst, ctx.src[arg->rankId], ctx.xnMaxTransportSize, ctx.event);
                    } else {
                        GroupCopy(ctx, ctx.myDst, ctx.src[arg->rankId], ctx.xnMaxTransportGoSize);
                        ccu::WaitEvent(ctx.event);
                    }
                    // 更新偏移
                    ctx.src[arg->rankId].addr += ctx.xnMaxTransportSize;
                    ctx.myDst.addr += ctx.xnMaxTransportSize;
                }
                ctx.sendRecvInfo[arg->rankId].loopNum += ctx.xnConst1;
        }
        // 等待本轮发送完成
        ctx.event.setMask(allBit);
        ccu::WaitEvent(ctx.event);
    }

    return CCU_SUCCESS;
}

// ============================================================================
// 主入口 Kernel 函数
// ============================================================================
CcuResult CcuAlltoAllVMesh1DKernel(CcuKernelArg arg)
{
    auto *kernelArg = static_cast<CcuKernelArgAlltoAllVMesh1D *>(arg);

    AlltoAllVMesh1DContext ctx;
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
    CalcGroupSrcDst(ctx);

    CCU_CHK_RET(DoAll2AllVMultiLoop(ctx));

    PostSync(ctx);
    HCCL_INFO("[CcuKernelAlltoAllMesh1D] AlltoAllMesh1D end");

    return CCU_SUCCESS;
}
}