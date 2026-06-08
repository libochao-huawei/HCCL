/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "ccu_kernel_all_gather_mesh1d_mem2mem.h"
#include "ccu_kernel_alg_base.h"

namespace ops_hccl {
using namespace hcomm;

constexpr int OUTPUT_XN_ID = 1;
constexpr int TOKEN_XN_ID = 2;
constexpr int CKE_IDX_0 = 0;
constexpr int CKE_IDX_1 = 1;
constexpr uint64_t CCU_MS_SIZE = 4096;
constexpr uint64_t LOCAL_COPY_MS = 8;
constexpr int POST_SYNC_ID = 3;  
constexpr uint16_t BIT_NUM_PER_CKE = 16;
constexpr uint32_t UNROLL_NUM = 16; // 最多支持8 * 16 = 128个rank

CcuKernelAllGatherMesh1DMem2Mem::CcuKernelAllGatherMesh1DMem2Mem(const CcuKernelArg &arg)
    : CcuKernelAlgBase(arg)
{
    const CcuKernelArgAllGatherMesh1DMem2Mem *kernelArg
        = dynamic_cast<const CcuKernelArgAllGatherMesh1DMem2Mem *>(&arg);
    rankId_         = kernelArg->rankId_;
    rankSize_       = kernelArg->dimSize_;
    channels_       = kernelArg->channels;

    HCCL_INFO(
        "[CcuKernelAllGatherMesh1DMem2Mem] Init, KernelArgs are rankId[%u], rankSize_[%u]",
        rankId_, rankSize_);
}

HcclResult CcuKernelAllGatherMesh1DMem2Mem::InitResource()
{
    localInput_           = CreateVariable();
    uint16_t channelIdx = 0;
    
    CHK_PRT_RET(channels_.size() == 0, HCCL_ERROR("[CcuKernelAllGatherMesh1DMem2Mem] channels is empty!"), HCCL_E_INTERNAL);

    // 按照rank号从小到大遍历channels，遇到本rank就填充本地资源，否则依次取远端资源，要求给框架返回的Link同样是按顺序排列的
    for (uint64_t peerId = 0; peerId < rankSize_; peerId++) {
        if (peerId == rankId_) {
            output_.push_back(CreateVariable());
            token_.push_back(CreateVariable());
        } else {
            HCCL_DEBUG("[CcuKernelAllGatherMesh1DMem2Mem] MyRank[%u], PeerId[%u], ChannelId[%u]",
                       rankId_, peerId, channelIdx);
            CcuRep::Variable inputVar, scratchVar, tokenVar;
            CHK_RET(CreateVariable(channels_[channelIdx], OUTPUT_XN_ID, &inputVar));
            output_.push_back(inputVar); // 获取channel中id=0的Var来传递output
            CHK_RET(CreateVariable(channels_[channelIdx], TOKEN_XN_ID, &tokenVar));
            token_.push_back(tokenVar);
            channelIdx++;
        }
    }
    currentRankSliceInputOffset_  = CreateVariable();
    currentRankSliceOutputOffset_ = CreateVariable();
    inputRepeatStride_            = CreateVariable();
    outputRepeatStride_           = CreateVariable();
    tmpRepeatNum_                 = CreateVariable();
    normalSliceSize_              = CreateVariable();
    lastSliceSize_                = CreateVariable();
    constVar1_                    = CreateVariable();
    constVar1_                    = 1;
    repeatTimeflag_               = CreateVariable();
    repeatTimeflag_               = 0;
    waitRepeatNum_                = CreateVariable();
    groupCopyRepeatNum_           = CreateVariable();
    isInputOutputEqual_           = CreateVariable();
    localGoSize_                  = CreateGroupOpSize();

    src = CreateLocalAddr();
    src_loccopy = CreateLocalAddr();
    localCopyDst_ = CreateLocalAddr();

    for (uint64_t rankIdx = 0; rankIdx < rankSize_; rankIdx++) {
        if (rankIdx == rankId_) {
            dst.push_back({});
        } else {
            dst.push_back(CreateRemoteAddr());
        }
    }

    uint32_t numEventsPerIter = (rankSize_ + BIT_NUM_PER_CKE - 1)/BIT_NUM_PER_CKE;
    for(uint32_t i = 0; i < UNROLL_NUM * numEventsPerIter; i++){
        event_.push_back(CreateCompletedEvent());
    }
    return HcclResult::HCCL_SUCCESS;
}

void CcuKernelAllGatherMesh1DMem2Mem::LoadArgs()
{
    Load(localInput_);
    Load(output_[rankId_]);
    Load(token_[rankId_]);
    Load(currentRankSliceInputOffset_);
    Load(currentRankSliceOutputOffset_);
    Load(tmpRepeatNum_);
    Load(inputRepeatStride_);
    Load(outputRepeatStride_);
    Load(normalSliceSize_);
    Load(lastSliceSize_);
    Load(isInputOutputEqual_);
    Load(localGoSize_);
    return;
}

void CcuKernelAllGatherMesh1DMem2Mem::PreSync()
{
    for (ChannelHandle channel : channels_) {
        NotifyRecord(channel, CKE_IDX_0, OUTPUT_XN_ID, output_[rankId_], 1 << OUTPUT_XN_ID);
        NotifyRecord(channel, CKE_IDX_0, TOKEN_XN_ID, token_[rankId_], 1 << TOKEN_XN_ID);
    }

    uint16_t allBit  = 1 << OUTPUT_XN_ID | 1 << TOKEN_XN_ID;
    for (ChannelHandle channel : channels_) {
        NotifyWait(channel, CKE_IDX_0, allBit);
    }
    return;
}

void CcuKernelAllGatherMesh1DMem2Mem::PostSync()
{
    for (ChannelHandle channel : channels_) {
        NotifyRecord(channel, CKE_IDX_0, 1 << POST_SYNC_ID);         // bit index = 4, 用作后同步。cke都可以用同一个，所以都是CKE_IDX_0
    }
    for (ChannelHandle channel : channels_) {
        NotifyWait(channel, CKE_IDX_0, 1 << POST_SYNC_ID);
    }
}

void CcuKernelAllGatherMesh1DMem2Mem::DoAllGather(const hcomm::CcuRep::LocalAddr              &src,
                                                             const std::vector<hcomm::CcuRep::RemoteAddr> &dst,
                                                             const CcuRep::Variable            &sliceSize)
{
    uint32_t channelId = 0;
    for (uint64_t rankIdx = 0; rankIdx < rankSize_; rankIdx++) {
        uint32_t eventIdx= rankIdx / BIT_NUM_PER_CKE;
        event_[eventIdx].SetMask(1 << (rankIdx % BIT_NUM_PER_CKE));
        if (rankIdx == rankId_) {
            RecordEvent(event_[eventIdx]);
        } else {
            WriteNb(channels_[channelId], dst[rankIdx], src, sliceSize, event_[eventIdx]);
            channelId++;
        }
    }
    CCU_IF(isInputOutputEqual_ == 0)
    {
        GroupCopy(localCopyDst_, src_loccopy, localGoSize_);
    }
    for(uint32_t i = 0; i < (rankSize_ + BIT_NUM_PER_CKE - 1)/BIT_NUM_PER_CKE; i++){
        if (i == (rankSize_ + BIT_NUM_PER_CKE - 1)/BIT_NUM_PER_CKE - 1) {
            if(rankSize_ % BIT_NUM_PER_CKE == 0){
                event_[i].SetMask((1 << BIT_NUM_PER_CKE) - 1);
            } else {
                event_[i].SetMask((1 << (rankSize_ % BIT_NUM_PER_CKE)) - 1);
            }
        } else {
            event_[i].SetMask((1 << BIT_NUM_PER_CKE) - 1);
        }
        WaitEvent(event_[i]);
    }
}

void CcuKernelAllGatherMesh1DMem2Mem::DoAllGatherWrite(const hcomm::CcuRep::LocalAddr              &src,
                                                                const std::vector<hcomm::CcuRep::RemoteAddr> &dst,
                                                                const CcuRep::Variable            &sliceSize,
                                                                uint32_t unrollIdx)
{
    uint32_t channelId = 0;
    uint32_t numEventsPerIter = (rankSize_ + BIT_NUM_PER_CKE - 1) / BIT_NUM_PER_CKE;
    for (uint64_t rankIdx = 0; rankIdx < rankSize_; rankIdx++) {
        uint32_t eventIdx = unrollIdx * numEventsPerIter + rankIdx / BIT_NUM_PER_CKE;
        event_[eventIdx].SetMask(1 << (rankIdx % BIT_NUM_PER_CKE));
        if (rankIdx == rankId_) {
            RecordEvent(event_[eventIdx]);
        } else {
            WriteNb(channels_[channelId], dst[rankIdx], src, sliceSize, event_[eventIdx]);
            channelId++;
        }
    }
}

void CcuKernelAllGatherMesh1DMem2Mem::DoAllGatherWait(uint32_t unrollIdx)
{
    uint32_t numEventsPerIter = (rankSize_ + BIT_NUM_PER_CKE - 1) / BIT_NUM_PER_CKE;
    for(uint32_t i = 0; i < numEventsPerIter; i++){
        uint32_t eventIdx = unrollIdx * numEventsPerIter + i;
        if (i == numEventsPerIter - 1) {
            if(rankSize_ % BIT_NUM_PER_CKE == 0){
                event_[eventIdx].SetMask((1 << BIT_NUM_PER_CKE) - 1);
            } else {
                event_[eventIdx].SetMask((1 << (rankSize_ % BIT_NUM_PER_CKE)) - 1);
            }
        } else {
            event_[eventIdx].SetMask((1 << BIT_NUM_PER_CKE) - 1);
        }
        WaitEvent(event_[eventIdx]);
    }
}

void CcuKernelAllGatherMesh1DMem2Mem::InitAllGatherAddr()
{
    src.addr = localInput_;
    src.addr += currentRankSliceInputOffset_;
    src.token = token_[rankId_];

    src_loccopy.addr = localInput_;
    src_loccopy.addr += currentRankSliceInputOffset_;
    src_loccopy.token = token_[rankId_];

    for (uint32_t rankIdx = 0; rankIdx < rankSize_; rankIdx++) {
        if (rankIdx == rankId_){
            localCopyDst_.addr = output_[rankId_];
            localCopyDst_.addr += currentRankSliceOutputOffset_;
            localCopyDst_.token = token_[rankId_];
        } else {
            dst[rankIdx].addr = output_[rankIdx];
            dst[rankIdx].addr += currentRankSliceOutputOffset_;
            dst[rankIdx].token = token_[rankIdx];
        }
    }
}

void CcuKernelAllGatherMesh1DMem2Mem::DoAllGatherGroupCopy()
{
    CCU_IF(isInputOutputEqual_ == 0)
    {
        CCU_IF(groupCopyRepeatNum_ != UINT64_MAX)
        {
            repeatTimeflag_ = 0;
            CCU_WHILE(groupCopyRepeatNum_ != UINT64_MAX)
            {
                groupCopyRepeatNum_ += constVar1_;
                CCU_IF(repeatTimeflag_ != 0)
                {
                    localCopyDst_.addr += outputRepeatStride_;
                }
                GroupCopy(localCopyDst_, src_loccopy, localGoSize_);
                repeatTimeflag_ = 1;
            }
        }
    }
}

void CcuKernelAllGatherMesh1DMem2Mem::DoRepeatAllGather()
{
    InitAllGatherAddr();
    waitRepeatNum_ = tmpRepeatNum_;
    groupCopyRepeatNum_ = tmpRepeatNum_;

    // Phase 1: 先下发所有WriteNb（非阻塞，event错开），不包含GroupCopy
    CCU_IF(tmpRepeatNum_ != UINT64_MAX)
    {
        tmpRepeatNum_ += constVar1_;
        CCU_IF(normalSliceSize_ != 0)
        {
            DoAllGatherWrite(src, dst, normalSliceSize_, 0);
        }
    }

    for (uint32_t i = 1; i < UNROLL_NUM; i++) {
        CCU_IF(tmpRepeatNum_ != UINT64_MAX)
        {
            tmpRepeatNum_ += constVar1_;
            src.addr += inputRepeatStride_;
            for (uint32_t rankIdx = 0; rankIdx < rankSize_; rankIdx++) {
                if (rankIdx != rankId_){
                    dst[rankIdx].addr += outputRepeatStride_;
                }
            }
            CCU_IF(normalSliceSize_ != 0)
            {
                DoAllGatherWrite(src, dst, normalSliceSize_, i);
            }
        }
    }

    // Phase 2: GroupCopy使用CCU_WHILE
    DoAllGatherGroupCopy();

    // Phase 3: 批量WaitEvent
    for (uint32_t i = 0; i < UNROLL_NUM; i++) {
        CCU_IF(waitRepeatNum_ != UINT64_MAX)
        {
            waitRepeatNum_ += constVar1_;
            DoAllGatherWait(i);
        }
    }
}

HcclResult CcuKernelAllGatherMesh1DMem2Mem::Algorithm()
{
    HCCL_INFO("[CcuKernelAllGatherMesh1DMem2Mem] AllgatherMesh1D run.");
    InitResource();
    LoadArgs();
    PreSync();
    DoRepeatAllGather();
    PostSync();
    HCCL_INFO("[CcuKernelAllGatherMesh1DMem2Mem] AllgatherMesh1D end.");
    return HcclResult::HCCL_SUCCESS;
}

std::vector<uint64_t> CcuKernelAllGatherMesh1DMem2Mem::GeneArgs(const CcuTaskArg &arg)
{
    const CcuTaskArgAllGatherMesh1DMem2Mem *taskArg
        = dynamic_cast<const CcuTaskArgAllGatherMesh1DMem2Mem *>(&arg);
    uint64_t inputAddr                   = taskArg->inputAddr_;
    uint64_t outputAddr                  = taskArg->outputAddr_;
    uint64_t token                       = taskArg->token_;
    
    uint64_t currentRankSliceInputOffset  = taskArg->inputSliceStride_ * rankId_;
    uint64_t currentRankSliceOutputOffset = taskArg->outputSliceStride_ * rankId_;
    uint64_t tmpRepeatNum                 = UINT64_MAX - taskArg->repeatNum_;
    uint64_t inputRepeatStride            = taskArg->inputRepeatStride_;
    uint64_t outputRepeatStride           = taskArg->outputRepeatStride_;
    uint64_t normalSliceSize              = taskArg->normalSliceSize_;
    uint64_t lastSliceSize                = taskArg->lastSliceSize_;
    uint64_t isInputOutputEqual           = taskArg->isInputOutputEqual_;

    auto goSize                           = CalGoSize(normalSliceSize);

    std::vector<uint64_t> taskArgs = {inputAddr,
                                      outputAddr,
                                      token,
                                      currentRankSliceInputOffset,
                                      currentRankSliceOutputOffset,
                                      tmpRepeatNum,
                                      inputRepeatStride,
                                      outputRepeatStride,
                                      normalSliceSize,
                                      lastSliceSize,
                                      isInputOutputEqual,
                                      goSize[0],
                                      goSize[1],
                                      goSize[2],
                                      goSize[3]};

    HCCL_INFO("[CcuKernelAllGatherMesh1DMem2Mem] TaskArgs: inputAddr[%llu], outputAddr[%llu], "
        "currentRankSliceInputOffset[%llu], currentRankSliceOutputOffset[%llu], "
        "repeatNum[%llu],inputRepeatStride[%llu], outputRepeatStride[%llu], normalSliceSize[%llu], lastSliceSize[%llu]",
        inputAddr, outputAddr, currentRankSliceInputOffset, currentRankSliceOutputOffset, UINT64_MAX - tmpRepeatNum,
        inputRepeatStride, outputRepeatStride, normalSliceSize, lastSliceSize);
    return taskArgs;
}

} // namespace ops_hccl