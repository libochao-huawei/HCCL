/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "ccu_kernel_all_reduce_nhr_mem2mem_1D_multi_jetty.h"

#include <set>

#include "ccu_kernel_alg_base.h"

namespace ops_hccl {

constexpr int BIT_NUM_PER_CKE = 16;
constexpr int CKE_IDX_0       = 0;

static CcuResult ParseKernelArg(AllReduceNhrMem2Mem1DMultiJettyContext &ctx,
                                CcuKernelArgAllReduceNhrMem2Mem1DMultiJetty *kernelArg)
{
    ctx.dataType       = kernelArg->opParam.DataDes.dataType;
    ctx.outputDataType = kernelArg->opParam.DataDes.outputType;
    if (ctx.outputDataType == HcclDataType::HCCL_DATA_TYPE_RESERVED) {
        ctx.outputDataType = ctx.dataType;
        HCCL_DEBUG("[CcuAllReduceNhrMem2Mem1DMultiJetty] outputDataType is [INVALID], set outputDataType to[%d]",
                   ctx.dataType);
    }
    ctx.reduceOp = kernelArg->opParam.reduceType;
    return CCU_SUCCESS;
}

static CcuResult InitResource(AllReduceNhrMem2Mem1DMultiJettyContext &ctx)
{
    const auto *arg = ctx.arg;

    if (arg->channelCount == 0) {
        HCCL_ERROR("[CcuAllReduceNhrMem2Mem1DMultiJetty] channels is empty!");
        return CcuResult::CCU_E_INTERNAL;
    }

    // Xns
    CCU_CHK_RET(ccu::Alloc(&ctx.inputAddr));
    CCU_CHK_RET(ccu::Alloc(&ctx.isInplace));
    CCU_CHK_RET(ccu::Alloc(&ctx.dataSizePerRank));
    CCU_CHK_RET(ccu::Alloc(&ctx.dataSizePerPort));
    CCU_CHK_RET(ccu::Alloc(&ctx.lastRankSliceSize));
    CCU_CHK_RET(ccu::Alloc(&ctx.lastPortSliceSize));

    // LocalAddr & RemoteAddr
    CCU_CHK_RET(ccu::Alloc(&ctx.localInput));
    CCU_CHK_RET(ccu::Alloc(&ctx.localOutput));
    CCU_CHK_RET(ccu::Alloc(&ctx.remoteOutput));

    // 初始化
    ctx.outputAddrs.resize(arg->rankSize);
    ctx.outputTokens.resize(arg->rankSize);
    ctx.sliceOffset.resize(arg->rankSize);

    for (uint32_t peerRankId = 0; peerRankId < arg->rankSize; peerRankId++) {
        if (peerRankId == arg->rankId) {
            // 本rank
            CCU_CHK_RET(ccu::Alloc(&ctx.outputAddrs[peerRankId]));
            CCU_CHK_RET(ccu::Alloc(&ctx.outputTokens[peerRankId]));
        } else if (arg->channelIdxMap.find(peerRankId) != arg->channelIdxMap.end()) {
            // 需要通信的对端
            const u32 chIdx = arg->channelIdxMap.at(peerRankId);
            HCCL_DEBUG("[CcuAllReduceNhrMem2Mem1DMultiJetty] MyRank[%u], peerRankId[%u], ChannelId[%u]",
                       arg->rankId, peerRankId, chIdx);
            CCU_CHK_RET(ccu::CreateByChannel(
                arg->channels[chIdx], static_cast<int>(XnId::OUTPUT), &ctx.outputAddrs[peerRankId]));
            CCU_CHK_RET(ccu::CreateByChannel(
                arg->channels[chIdx], static_cast<int>(XnId::TOKEN), &ctx.outputTokens[peerRankId]));
        } else {
            // 不需要通信的对端，填空
            ctx.outputAddrs[peerRankId] = CcuVariable();
            ctx.outputTokens[peerRankId] = CcuVariable();
        }

        CCU_CHK_RET(ccu::Alloc(&ctx.sliceOffset[peerRankId]));
    }

    // 创建goSize的Variables
    CCU_CHK_RET(ccu::Alloc(&ctx.localCopyGoSize.addrOffset));
    CCU_CHK_RET(ccu::Alloc(&ctx.localCopyGoSize.loopParam));
    CCU_CHK_RET(ccu::Alloc(&ctx.localCopyGoSize.parallelParam));
    CCU_CHK_RET(ccu::Alloc(&ctx.localCopyGoSize.residual));

    CCU_CHK_RET(ccu::Alloc(&ctx.localCopyGoSizeLastSlice.addrOffset));
    CCU_CHK_RET(ccu::Alloc(&ctx.localCopyGoSizeLastSlice.loopParam));
    CCU_CHK_RET(ccu::Alloc(&ctx.localCopyGoSizeLastSlice.parallelParam));
    CCU_CHK_RET(ccu::Alloc(&ctx.localCopyGoSizeLastSlice.residual));

    // 需要portNum_个event
    ctx.events.resize(arg->portSize);
    for (uint32_t i = 0; i < arg->portSize; ++i) {
        CCU_CHK_RET(ccu::Alloc(&ctx.events[i]));
    }

    ctx.resourceAllocated = false;

    return CCU_SUCCESS;
}

static CcuResult LoadArgs(AllReduceNhrMem2Mem1DMultiJettyContext &ctx)
{
    const auto *arg = ctx.arg;

    CCU_CHK_RET(ccu::LoadArg(ctx.inputAddr));
    CCU_CHK_RET(ccu::LoadArg(ctx.outputAddrs[arg->rankId]));
    CCU_CHK_RET(ccu::LoadArg(ctx.outputTokens[arg->rankId]));
    CCU_CHK_RET(ccu::LoadArg(ctx.isInplace));
    CCU_CHK_RET(ccu::LoadArg(ctx.dataSizePerRank));
    CCU_CHK_RET(ccu::LoadArg(ctx.dataSizePerPort));
    CCU_CHK_RET(ccu::LoadArg(ctx.lastRankSliceSize));
    CCU_CHK_RET(ccu::LoadArg(ctx.lastPortSliceSize));
    CCU_CHK_RET(ccu::LoadArg(ctx.localCopyGoSize.addrOffset));
    CCU_CHK_RET(ccu::LoadArg(ctx.localCopyGoSize.loopParam));
    CCU_CHK_RET(ccu::LoadArg(ctx.localCopyGoSize.parallelParam));
    CCU_CHK_RET(ccu::LoadArg(ctx.localCopyGoSize.residual));
    CCU_CHK_RET(ccu::LoadArg(ctx.localCopyGoSizeLastSlice.addrOffset));
    CCU_CHK_RET(ccu::LoadArg(ctx.localCopyGoSizeLastSlice.loopParam));
    CCU_CHK_RET(ccu::LoadArg(ctx.localCopyGoSizeLastSlice.parallelParam));
    CCU_CHK_RET(ccu::LoadArg(ctx.localCopyGoSizeLastSlice.residual));

    return CCU_SUCCESS;
}

static uint32_t GetSignalIndex(const SignalBit signalBit)
{
    // 一个CKE有16位，可以处理16个用途
    return static_cast<uint32_t>(signalBit) / BIT_NUM_PER_CKE;
}

static uint16_t GetSignalMask(const SignalBit signalBit)
{
    return (1 << (static_cast<uint32_t>(signalBit) % BIT_NUM_PER_CKE));
}

static CcuResult LocalWaitAllEvent(AllReduceNhrMem2Mem1DMultiJettyContext &ctx, const uint16_t mask)
{
    for (auto &event : ctx.events) {
        event.mask = mask;
        CCU_CHK_RET(ccu::WaitEvent(event));
    }
    return CCU_SUCCESS;
}

static CcuResult PreSync(AllReduceNhrMem2Mem1DMultiJettyContext &ctx)
{
    const auto *arg = ctx.arg;

    HCCL_DEBUG("[CcuAllReduceNhrMem2Mem1DMultiJetty] PreSync start");

    const uint16_t signalBitOutput = GetSignalMask(SignalBit::PRE_SYNC_OUTPUT);
    const uint16_t signalBitToken  = GetSignalMask(SignalBit::PRE_SYNC_TOKEN);
    const uint32_t signalIndexOutput = GetSignalIndex(SignalBit::PRE_SYNC_OUTPUT);
    const uint32_t signalIndexToken  = GetSignalIndex(SignalBit::PRE_SYNC_TOKEN);

    // 通知所有对端，同时写output和token信息
    for (uint32_t i = 0; i < arg->channelCount; i++) {
        ccu::WriteVariableWithNotify(arg->channels[i], ctx.outputAddrs[arg->rankId],
            static_cast<int>(XnId::OUTPUT), signalIndexOutput, signalBitOutput);
        ccu::WriteVariableWithNotify(arg->channels[i], ctx.outputTokens[arg->rankId],
            static_cast<int>(XnId::TOKEN), signalIndexToken, signalBitToken);
    }

    // 等待所有需要通信的对端
    const uint16_t waitMask = signalBitOutput | signalBitToken;
    std::set<uint32_t> signalIdxes{signalIndexOutput, signalIndexToken};
    for (uint32_t i = 0; i < arg->channelCount; i++) {
        for (const auto signalIdx : signalIdxes) {
            ccu::NotifyWait(arg->channels[i], signalIdx, waitMask);
        }
    }

    HCCL_DEBUG("[CcuAllReduceNhrMem2Mem1DMultiJetty] PreSync end");
    return CCU_SUCCESS;
}

static void PostSync(AllReduceNhrMem2Mem1DMultiJettyContext &ctx)
{
    const auto *arg = ctx.arg;

    const uint32_t signalIndex = GetSignalIndex(SignalBit::POST_SYNC);
    const uint16_t signalBit   = GetSignalMask(SignalBit::POST_SYNC);

    for (uint32_t i = 0; i < arg->channelCount; i++) {
        ccu::NotifyRecord(arg->channels[i], signalIndex, signalBit);
    }
    for (uint32_t i = 0; i < arg->channelCount; i++) {
        ccu::NotifyWait(arg->channels[i], signalIndex, signalBit);
    }
}

static std::vector<u32> GetNonTxSliceIdxs(const std::vector<u32> &txSliceIdxs, uint32_t rankSize)
{
    std::vector<bool> isTx(rankSize, false);
    for (u32 idx : txSliceIdxs) {
        if (idx < rankSize) {
            isTx[idx] = true;
        }
    }

    std::vector<u32> nonTxSliceIdxs;
    for (u32 idx = 0; idx < rankSize; ++idx) {
        if (!isTx[idx]) {
            nonTxSliceIdxs.push_back(idx);
        }
    }

    return nonTxSliceIdxs;
}

static CcuResult DoLocalCopySlice(AllReduceNhrMem2Mem1DMultiJettyContext &ctx,
                                  ccu::LocalAddr &src, ccu::LocalAddr &dst,
                                  const u32 &copySliceIdx, CcuEvent &event)
{
    const auto *arg = ctx.arg;
    const bool islastSlice = copySliceIdx + 1 == arg->rankSize;
    CcuVariable sliceSize;
    CHK_RET(ccu::Alloc(&sliceSize));
    sliceSize = islastSlice ? ctx.lastRankSliceSize : ctx.dataSizePerRank;

    CCU_IF_ONLY(sliceSize != 0)
    {
        CCU_CHK_RET(ccu::LocalCopyNb(dst, src, sliceSize, event));
    }
    CCU_IF_ONLY(sliceSize == 0)
    {
        ccu::RecordEvent(event);
    }
    return CCU_SUCCESS;
}

static CcuResult LocalCopySlices(AllReduceNhrMem2Mem1DMultiJettyContext &ctx)
{
    const auto *arg = ctx.arg;

    CcuVariable tmpSliceOffset;
    CCU_CHK_RET(ccu::Alloc(&tmpSliceOffset));
    tmpSliceOffset = 0;
    u32 nonTxSliceIdx = 0;
    for (u64 i = 0; i < arg->rankSize; i++) {
        ctx.sliceOffset[i] = tmpSliceOffset;
        tmpSliceOffset += ctx.dataSizePerRank;
    }
    // 使用一个event
    CcuEvent &event = ctx.events[0];
    // 原地操作时不需要拷贝
    CCU_IF_ONLY(ctx.isInplace == 0)
    {
        // 将step0中不需要写的slice，拷贝到本rank的output中
        const NHRStepInfo &nhrStepInfo = arg->algStepInfoList[0];
        const std::vector<u32> &nonTxSliceIdxList = GetNonTxSliceIdxs(nhrStepInfo.txSliceIdxs, arg->rankSize);

        for (u32 i = 0; i < nonTxSliceIdxList.size(); i++) {
            nonTxSliceIdx = nonTxSliceIdxList[i];

            if (i != 0) { // 每拷贝16块等一次
                if (i % BIT_NUM_PER_CKE == 0) {
                    event.mask = (1 << BIT_NUM_PER_CKE) - 1;
                    ccu::WaitEvent(event);
                }
            }

            ctx.localInput.addr  = ctx.inputAddr;
            ctx.localInput.addr += ctx.sliceOffset[nonTxSliceIdx];
            ctx.localInput.token = ctx.outputTokens[arg->rankId];

            ctx.localOutput.addr  = ctx.outputAddrs[arg->rankId];
            ctx.localOutput.addr += ctx.sliceOffset[nonTxSliceIdx];
            ctx.localOutput.token = ctx.outputTokens[arg->rankId];

            event.mask = 1 << i;
            CCU_CHK_RET(DoLocalCopySlice(ctx, ctx.localInput, ctx.localOutput, nonTxSliceIdx, event));
        }
        event.mask = (1 << (nonTxSliceIdxList.size() % BIT_NUM_PER_CKE)) - 1;
        ccu::WaitEvent(event);
    }
    return CCU_SUCCESS;
}

static CcuResult DoWriteReduceSlice(AllReduceNhrMem2Mem1DMultiJettyContext &ctx,
                                    const u32 toRank, ccu::LocalAddr &src, ccu::RemoteAddr &dst,
                                    const u32 sendSliceIdx, const u32 signalIndex)
{
    const auto *arg = ctx.arg;
    const u32 toRankIdx = arg->channelIdxMap.at(toRank);
    const ChannelHandle sendChannel = arg->channels[toRankIdx];

    const bool islastSlice = sendSliceIdx + 1 == arg->rankSize;
    CcuVariable lastSliceSize;
    CHK_RET(ccu::Alloc(&lastSliceSize));
    lastSliceSize = islastSlice ? ctx.lastPortSliceSize : ctx.dataSizePerPort;

    for (auto &event : ctx.events) {
        event.mask = 1 << signalIndex;
    }

    CCU_IF_ONLY(ctx.dataSizePerPort != 0)
    {
        for (uint32_t i = 0; i < arg->portSize - 1; ++i) {
            CCU_CHK_RET(ccu::WriteReduceNb(sendChannel, dst, src, ctx.dataSizePerPort,
                                            ctx.dataType, ctx.reduceOp, ctx.events[i]));
            src.addr += ctx.dataSizePerPort;
            dst.addr += ctx.dataSizePerPort;
        }
    }
    CCU_IF_ONLY(ctx.dataSizePerPort == 0)
    {
        for (uint32_t i = 0; i < arg->portSize - 1; ++i) {
            ccu::RecordEvent(ctx.events[i]);
        }
    }
    CCU_IF_ONLY(lastSliceSize != 0)
    {
        CCU_CHK_RET(ccu::WriteReduceNb(sendChannel, dst, src, lastSliceSize,
                                        ctx.dataType, ctx.reduceOp, ctx.events[ctx.events.size() - 1]));
    }
    CCU_IF_ONLY(lastSliceSize == 0)
    {
        ccu::RecordEvent(ctx.events[ctx.events.size() - 1]);
    }
    return CCU_SUCCESS;
}

static CcuResult DoReduceScatterNHRSingleStep(AllReduceNhrMem2Mem1DMultiJettyContext &ctx,
                                               const NHRStepInfo &nhrStepInfo)
{
    const auto *arg = ctx.arg;
    const u32 toRankIdx   = arg->channelIdxMap.at(nhrStepInfo.toRank);
    const u32 fromRankIdx = arg->channelIdxMap.at(nhrStepInfo.fromRank);
    const ChannelHandle sendChannel = arg->channels[toRankIdx];
    const ChannelHandle recvChannel = arg->channels[fromRankIdx];
    const std::vector<u32> &sendSliceIdxList = nhrStepInfo.txSliceIdxs;

    ctx.localInput.token  = ctx.outputTokens[arg->rankId];
    ctx.remoteOutput.token = ctx.outputTokens[nhrStepInfo.toRank];

    HCCL_DEBUG("[DoReduceScatterNHRSingleStep] nhrStepInfo{step[%u], myRank[%u], toRank[%u], fromRank[%u]},"
               " toRankIdx[%u], fromRankIdx[%u]",
               nhrStepInfo.step, nhrStepInfo.myRank, nhrStepInfo.toRank, nhrStepInfo.fromRank,
               toRankIdx, fromRankIdx);

    const uint32_t signalIdReady = GetSignalIndex(SignalBit::READY_TO_RECV_RS);
    const uint32_t signalIdDone  = GetSignalIndex(SignalBit::SEND_DONE_RS);
    const uint16_t signalBitReady = GetSignalMask(SignalBit::READY_TO_RECV_RS);
    const uint16_t signalBitDone  = GetSignalMask(SignalBit::SEND_DONE_RS);

    if (nhrStepInfo.step != 0) {
        // 通知fromRank，可以写入
        ccu::NotifyRecord(recvChannel, signalIdReady, signalBitReady);
        // 等待toRank通知其可以写入
        ccu::NotifyWait(sendChannel, signalIdReady, signalBitReady);
    }

    for (u32 i = 0; i < sendSliceIdxList.size(); i++) {
        u32 sendSliceIdx = sendSliceIdxList[i];

        if (i != 0) {
            if (i % BIT_NUM_PER_CKE == 0) {
                CCU_CHK_RET(LocalWaitAllEvent(ctx, (1 << BIT_NUM_PER_CKE) - 1));
            }
        }

        if (nhrStepInfo.step == 0) {
             // 只有第0步的源数据从input中取
            ctx.localInput.addr  = ctx.inputAddr;
            ctx.localInput.addr += ctx.sliceOffset[sendSliceIdx];
        } else {
            ctx.localInput.addr  = ctx.outputAddrs[arg->rankId];
            ctx.localInput.addr += ctx.sliceOffset[sendSliceIdx];
        }

        ctx.remoteOutput.addr  = ctx.outputAddrs[nhrStepInfo.toRank];
        ctx.remoteOutput.addr += ctx.sliceOffset[sendSliceIdx];

        CCU_CHK_RET(DoWriteReduceSlice(ctx, nhrStepInfo.toRank, ctx.localInput, ctx.remoteOutput,
                                        sendSliceIdx, i % BIT_NUM_PER_CKE));
    }
    CCU_CHK_RET(LocalWaitAllEvent(ctx, (1 << (sendSliceIdxList.size() % BIT_NUM_PER_CKE)) - 1));

    // 通知toRank数据写入完毕
    ccu::NotifyRecord(sendChannel, signalIdDone, signalBitDone);
    // 等待fromRank通知其数据写入完毕
    ccu::NotifyWait(recvChannel, signalIdDone, signalBitDone);

    HCCL_DEBUG("[DoReduceScatterNHRSingleStep] rank %u step %u, toRank=%u, fromRank=%u, nSlice=%lu",
               arg->rankId, nhrStepInfo.step, nhrStepInfo.toRank, nhrStepInfo.fromRank, sendSliceIdxList.size());
    return CCU_SUCCESS;
}

static CcuResult DoReduceScatterNHR(AllReduceNhrMem2Mem1DMultiJettyContext &ctx)
{
    const auto *arg = ctx.arg;
    constexpr uint32_t nhrNum = 2;
    for (u64 i = 0; i < arg->algStepInfoList.size() / nhrNum; i++) {
        const NHRStepInfo &nhrStepInfo = arg->algStepInfoList[i];
        CCU_CHK_RET(DoReduceScatterNHRSingleStep(ctx, nhrStepInfo));
    }
    return CCU_SUCCESS;
}

static CcuResult DoSendRecvSlice(AllReduceNhrMem2Mem1DMultiJettyContext &ctx,
                                 const u32 toRank, ccu::LocalAddr &src, ccu::RemoteAddr &dst,
                                 const u32 &sendSliceIdx, u32 signalIndex)
{
    const auto *arg = ctx.arg;
    const u32 toRankIdx = arg->channelIdxMap.at(toRank);
    const ChannelHandle sendChannel = arg->channels[toRankIdx];

    // allreduce切片的最后一块slice，大小可能不一致
    const bool islastSlice = sendSliceIdx + 1 == arg->rankSize;
    CcuVariable lastSliceSize;
    CHK_RET(ccu::Alloc(&lastSliceSize));
    lastSliceSize = islastSlice ? ctx.lastPortSliceSize : ctx.dataSizePerPort;

    // 统一设置一下mask
    for (auto &event : ctx.events) {
        event.mask = 1 << signalIndex;
    }

    CCU_IF_ONLY(ctx.dataSizePerPort != 0)
    {
        for (uint32_t i = 0; i < arg->portSize - 1; ++i) {
            CCU_CHK_RET(ccu::WriteNb(sendChannel, dst, src, ctx.dataSizePerPort, ctx.events[i]));
            src.addr += ctx.dataSizePerPort;
            dst.addr += ctx.dataSizePerPort;
        }
    }
    CCU_IF_ONLY(ctx.dataSizePerPort == 0)
    {
        // 无数据时，直接record event
        for (uint32_t i = 0; i < arg->portSize - 1; ++i) {
            ccu::RecordEvent(ctx.events[i]);
        }
    }
    CCU_IF_ONLY(lastSliceSize != 0)
    {
        CCU_CHK_RET(ccu::WriteNb(sendChannel, dst, src, lastSliceSize, ctx.events[ctx.events.size() - 1]));
    }
    CCU_IF_ONLY(lastSliceSize == 0)
    {
        ccu::RecordEvent(ctx.events[ctx.events.size() - 1]);
    }

    return CCU_SUCCESS;
}

static CcuResult DoAllGatherNHRSingleStep(AllReduceNhrMem2Mem1DMultiJettyContext &ctx,
                                           const NHRStepInfo &nhrStepInfo)
{
    const auto *arg = ctx.arg;
    const u32 toRankIdx   = arg->channelIdxMap.at(nhrStepInfo.toRank);
    const u32 fromRankIdx = arg->channelIdxMap.at(nhrStepInfo.fromRank);
    const ChannelHandle sendChannel = arg->channels[toRankIdx];
    const ChannelHandle recvChannel = arg->channels[fromRankIdx];
    const std::vector<u32> &sendSliceIdxList = nhrStepInfo.txSliceIdxs;

    ctx.localInput.token  = ctx.outputTokens[arg->rankId];
    ctx.remoteOutput.token = ctx.outputTokens[nhrStepInfo.toRank];

    const uint32_t signalIdDone  = GetSignalIndex(SignalBit::SEND_DONE_AG);
    const uint16_t signalBitDone = GetSignalMask(SignalBit::SEND_DONE_AG);

    for (u32 i = 0; i < sendSliceIdxList.size(); i++) {
        u32 sendSliceIdx = sendSliceIdxList[i];

        if (i != 0) {
            if (i % BIT_NUM_PER_CKE == 0) {
                CCU_CHK_RET(LocalWaitAllEvent(ctx, (1 << BIT_NUM_PER_CKE) - 1));
            }
        }

        ctx.localInput.addr  = ctx.outputAddrs[arg->rankId];
        ctx.localInput.addr += ctx.sliceOffset[sendSliceIdx];

        ctx.remoteOutput.addr  = ctx.outputAddrs[nhrStepInfo.toRank];
        ctx.remoteOutput.addr += ctx.sliceOffset[sendSliceIdx];

        CCU_CHK_RET(DoSendRecvSlice(ctx, nhrStepInfo.toRank, ctx.localInput, ctx.remoteOutput,
                                     sendSliceIdx, i % BIT_NUM_PER_CKE));
    }
    CCU_CHK_RET(LocalWaitAllEvent(ctx, (1 << (sendSliceIdxList.size() % BIT_NUM_PER_CKE)) - 1));

    ccu::NotifyRecord(sendChannel, signalIdDone, signalBitDone);
    ccu::NotifyWait(recvChannel, signalIdDone, signalBitDone);

    HCCL_DEBUG("[DoAllGatherNHRSingleStep] rank %u step %u, toRank=%u, fromRank=%u, nSlice=%lu",
               arg->rankId, nhrStepInfo.step, nhrStepInfo.toRank, nhrStepInfo.fromRank, sendSliceIdxList.size());

    return CCU_SUCCESS;
}

static CcuResult DoAllGatherNHR(AllReduceNhrMem2Mem1DMultiJettyContext &ctx)
{
    const auto *arg = ctx.arg;
    constexpr uint32_t nhrNum = 2;
    for (u64 i = arg->algStepInfoList.size() / nhrNum; i < arg->algStepInfoList.size(); i++) {
        const NHRStepInfo &nhrStepInfo = arg->algStepInfoList[i];
        CCU_CHK_RET(DoAllGatherNHRSingleStep(ctx, nhrStepInfo));
    }
    return CCU_SUCCESS;
}

CcuResult CcuAllReduceNhrMem2Mem1DMultiJettyKernel(CcuKernelArg arg)
{
    auto *kernelArg = static_cast<CcuKernelArgAllReduceNhrMem2Mem1DMultiJetty *>(arg);

    AllReduceNhrMem2Mem1DMultiJettyContext ctx;
    ctx.arg = kernelArg;
    ctx.resourceAllocated = false;
    ctx.moConfig.msInterleave = 0;
    ctx.moConfig.loopCount    = 0;
    ctx.moConfig.memSlice     = 0;
    ctx.moRes.eventCount      = 0;
    ctx.moRes.bufCount        = 0;
    ctx.enginePool            = 0;

    HCCL_INFO("[CcuAllReduceNhrMem2Mem1DMultiJetty] AllReduceNhrMem2Mem1DMultiJetty run");
    CCU_CHK_RET(ParseKernelArg(ctx, kernelArg));
    CCU_CHK_RET(InitResource(ctx));
    CCU_CHK_RET(LoadArgs(ctx));
    CCU_CHK_RET(LocalCopySlices(ctx));
    PreSync(ctx);

    CCU_CHK_RET(DoReduceScatterNHR(ctx));
    CCU_CHK_RET(DoAllGatherNHR(ctx));

    PostSync(ctx);
    HCCL_INFO("[CcuAllReduceNhrMem2Mem1DMultiJetty] AllReduceNhrMem2Mem1DMultiJetty end");

    return CCU_SUCCESS;
}

} // namespace ops_hccl
