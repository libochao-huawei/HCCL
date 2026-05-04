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
#include "ccu_kernel_reduce_scatter_mesh1d_flatten.h"

namespace ops_hccl {
using namespace hcomm;

// bit序号，每种信号用一个bit
constexpr int INPUT_XN_ID   = 0;
constexpr int SCRATCH_XN_ID = 1;
constexpr int TOKEN_XN_ID   = 2;
constexpr int POST_SYNC_ID   = 3;
// cke序号
constexpr int CKE_IDX_0     = 0;
constexpr int BIT_NUM_PER_CKE     = 16;

CcuKernelReduceScatterMesh1DFlatten::CcuKernelReduceScatterMesh1DFlatten(const CcuKernelArg &arg)
    : CcuKernelAlgBase(arg)
{
    const CcuKernelArgReduceScatterMesh1DFlatten *kernelArg
        = dynamic_cast<const CcuKernelArgReduceScatterMesh1DFlatten *>(&arg);
    rankId_         = kernelArg->rankId_;
    rankSize_       = kernelArg->dimSize_;
    channels_       = kernelArg->channels;
    dataType_       = kernelArg->opParam_.DataDes.dataType;
    outputDataType_ = kernelArg->opParam_.DataDes.outputType;
    if (outputDataType_ == HcclDataType::HCCL_DATA_TYPE_RESERVED) {
        outputDataType_ = dataType_;
        HCCL_DEBUG(
            "[CcuKernelReduceScatterMesh1DFlatten] outputDataType is [INVALID], set outputDataType to[%d]",
            outputDataType_);
    }
    reduceOp_       = kernelArg->opParam_.reduceType;
    HCCL_INFO(
        "[CcuKernelReduceScatterMesh1DFlatten] Init, KernelArgs are rankId[%u], rankSize_[%u], dataType[%d], "
        "outputDataType[%d], reduceOp[%d]",
        rankId_, rankSize_, dataType_, outputDataType_, reduceOp_);
}

HcclResult CcuKernelReduceScatterMesh1DFlatten::InitResource()
{
    uint16_t channelIdx = 0;
    if (channels_.size() == 0) {
        HCCL_ERROR("[CcuKernelReduceScatterMesh1DFlatten] channels is empty!");
        return HcclResult::HCCL_E_INTERNAL;
    }

    // 按照rank号从小到大遍历channels，遇到本rank就填充本地资源，否则依次取远端资源，要求给框架返回的Link同样是按顺序排列的
    for (uint64_t peerId = 0; peerId < rankSize_; peerId++) {
        if (peerId == rankId_) {
            input_.push_back(CreateVariable());
            scratch_.push_back(CreateVariable());
            token_.push_back(CreateVariable());
        } else {
            HCCL_DEBUG("[CcuKernelReduceScatterMesh1DFlatten] MyRank[%u], PeerId[%u], ChannelId[%u]",
                       rankId_, peerId, channelIdx);

            CcuRep::Variable inputVar, scratchVar, tokenVar;
            CHK_RET(CreateVariable(channels_[channelIdx], INPUT_XN_ID, &inputVar));
            input_.push_back(inputVar); // 获取channel中id=0的Var来传递output
            CHK_RET(CreateVariable(channels_[channelIdx], SCRATCH_XN_ID, &scratchVar));
            scratch_.push_back(scratchVar);
            CHK_RET(CreateVariable(channels_[channelIdx], TOKEN_XN_ID, &tokenVar));
            token_.push_back(tokenVar);
            channelIdx++;
        }
    }
    output_                      = CreateVariable();
    currentRankSliceInputOffset_ = CreateVariable();
    currentRankSliceOutputOffset_= CreateVariable();
    normalSliceSize_             = CreateVariable();
    inputRepeatStride_           = CreateVariable();
    outputRepeatStride_          = CreateVariable();
    repeatNum_                   = CreateVariable();
    lastSliceSize_               = CreateVariable();
    flag_                        = CreateVariable();
    localEvent_                  = CreateCompletedEvent();

    selfBit_ = 1 << rankId_;                              // 仅rankid位为1，其他位为0，代表本端准备好了
    allBit_ = ((1 << rankSize_) - 1) & (~(1 << rankId_)); // 仅rankid位为0，其他位为1，代表远端准备好了
    remoteInput_.reserve(rankSize_);
    scratchMem_.reserve(rankSize_);
    for (uint64_t rankIdx = 0; rankIdx < rankSize_; rankIdx++) {
        scratchMem_.push_back(CreateLocalAddr());
        if (rankIdx == rankId_) {
            myInput_ = CreateLocalAddr();
            remoteInput_.push_back({});
        } else {
            remoteInput_.push_back(CreateRemoteAddr());
        }
    }

    for(uint32_t i = 0; i < (rankSize_ + BIT_NUM_PER_CKE - 1)/BIT_NUM_PER_CKE; i++)
    {
        event_.push_back(CreateCompletedEvent());
    }
    return HcclResult::HCCL_SUCCESS;
}

void CcuKernelReduceScatterMesh1DFlatten::LoadArgs()
{
    Load(input_[rankId_]);
    Load(output_);
    Load(token_[rankId_]);
    Load(scratch_[rankId_]);
    Load(currentRankSliceInputOffset_);
    Load(currentRankSliceOutputOffset_);
    Load(inputRepeatStride_);
    Load(outputRepeatStride_);
    Load(normalSliceSize_);
    Load(lastSliceSize_);
    Load(repeatNum_);
    return;
}

void CcuKernelReduceScatterMesh1DFlatten::PreSync()
{
    for (ChannelHandle channel : channels_) {
        NotifyRecord(channel, CKE_IDX_0, INPUT_XN_ID, input_[rankId_], 1 << INPUT_XN_ID);       // bit index = 1，传递input信息
        NotifyRecord(channel, CKE_IDX_0, SCRATCH_XN_ID, scratch_[rankId_], 1 << SCRATCH_XN_ID); // bit index = 2, 传递scratch信息
        NotifyRecord(channel, CKE_IDX_0, TOKEN_XN_ID, token_[rankId_], 1 << TOKEN_XN_ID);       // bit index = 3，传递output信息
    }
    uint32_t allBit = 1 << INPUT_XN_ID | 1 << SCRATCH_XN_ID | 1 << TOKEN_XN_ID;
    for (ChannelHandle channel : channels_) {
        NotifyWait(channel, CKE_IDX_0, allBit);
    }

    return;
}

void CcuKernelReduceScatterMesh1DFlatten::PostSync()
{
    for (ChannelHandle channel : channels_) {
        NotifyRecord(channel, CKE_IDX_0, 1 << POST_SYNC_ID);         // bit index = 4, 用作后同步。cke都可以用同一个，所以都是CKE_IDX_0
    }
    for (ChannelHandle channel : channels_) {
        NotifyWait(channel, CKE_IDX_0, 1 << POST_SYNC_ID);
    }
}

void CcuKernelReduceScatterMesh1DFlatten::DoReduceScatter()
{
    uint32_t channelId = 0;

    CcuRep::LocalAddr myOutput = CreateLocalAddr();

    myOutput.addr   = output_;
    myOutput.addr  += currentRankSliceOutputOffset_;
    myOutput.token  = token_[rankId_];

    CcuRep::Variable sliceSize = CreateVariable();
    sliceSize = (rankId_ == (rankSize_ - 1)) ? lastSliceSize_: normalSliceSize_;

    CCU_IF(sliceSize != 0)
    {   //input->scratch[rankid],localcopy
        localEvent_.SetMask(1);
        LocalCopyNb(scratchMem_[rankId_], myInput_, sliceSize, localEvent_);
        WaitEvent(localEvent_);
        for (uint32_t rankIdx = 0; rankIdx < rankSize_; rankIdx++) {
            uint32_t eventIdx= rankIdx / BIT_NUM_PER_CKE;
            event_[eventIdx].SetMask(1 << (rankIdx % BIT_NUM_PER_CKE));
            if (rankIdx == rankId_) {
                RecordEvent(event_[eventIdx]);
            } else {
                ReadReduceNb(channels_[channelId], scratchMem_[rankId_], remoteInput_[rankIdx], sliceSize, dataType_, reduceOp_, event_[eventIdx]);
                channelId++;
            }
        }
        // 等读完所有对端
        for(uint32_t i = 0; i < (rankSize_ + BIT_NUM_PER_CKE - 1)/BIT_NUM_PER_CKE; i++){
            if(i == (rankSize_ + BIT_NUM_PER_CKE - 1)/BIT_NUM_PER_CKE - 1){
                if(rankSize_ % BIT_NUM_PER_CKE == 0){
                    event_[i].SetMask((1 << BIT_NUM_PER_CKE) - 1);
                } else {
                    event_[i].SetMask((1 << rankSize_ % BIT_NUM_PER_CKE) - 1);
                }
            } else {
                event_[i].SetMask((1 << BIT_NUM_PER_CKE) - 1);
            }
            WaitEvent(event_[i]);
        }

        LocalCopyNb(myOutput, scratchMem_[rankId_], sliceSize, localEvent_);
        WaitEvent(localEvent_);
    }


}

void CcuKernelReduceScatterMesh1DFlatten::DoRepeatReduceScatter()
{
    CcuRep::Variable scratchOffset = CreateVariable();
    scratchOffset                  = 0;

    for (uint32_t rankIdx = 0; rankIdx < rankSize_; rankIdx++) {
        if (rankIdx == rankId_) {
            myInput_.addr = input_[rankIdx];
            myInput_.addr += currentRankSliceInputOffset_;
            myInput_.token = token_[rankIdx];
        } else {
            remoteInput_[rankIdx].addr = input_[rankIdx];
            remoteInput_[rankIdx].addr += currentRankSliceInputOffset_;
            remoteInput_[rankIdx].token = token_[rankIdx];
        }

        scratchMem_[rankIdx].addr = scratch_[rankId_];
        scratchMem_[rankIdx].addr += scratchOffset;
        scratchOffset += normalSliceSize_;
        scratchMem_[rankIdx].token = token_[rankId_];
    }

    CcuRep::Variable repeatNumAdd = CreateVariable();
    repeatNumAdd  = 1;
    flag_ = 0;
    CCU_WHILE(repeatNum_ != UINT64_MAX) {
        repeatNum_ += repeatNumAdd;
        CCU_IF(flag_ == 1) {
            //  非第一轮执行时，src 和 dst 已经初始化，需要添加偏移量
            for (uint64_t rankIdx = 0; rankIdx < rankSize_; rankIdx++) {
                if (rankIdx == rankId_) {
                    myInput_.addr += inputRepeatStride_;
                } else {
                    remoteInput_[rankIdx].addr += inputRepeatStride_;
                }
            }
            output_ += outputRepeatStride_;
        }
        DoReduceScatter();
        flag_ = 1;
    }
}



HcclResult CcuKernelReduceScatterMesh1DFlatten::Algorithm()
{
    HCCL_INFO("[CcuKernelReduceScatterMesh1DFlatten] ReduceScatterMesh1DFlatten run");

    CHK_RET(InitResource());

    LoadArgs();

    PreSync();

    DoRepeatReduceScatter();

    PostSync();

    HCCL_INFO("[CcuKernelReduceScatterMesh1DFlatten] ReduceScatterMesh1DFlatten end");

    return HcclResult::HCCL_SUCCESS;
}

std::vector<uint64_t> CcuKernelReduceScatterMesh1DFlatten::GeneArgs(const CcuTaskArg &arg)
{
    const CcuTaskArgReduceScatterMesh1DFlatten *taskArg
        = dynamic_cast<const CcuTaskArgReduceScatterMesh1DFlatten *>(&arg);
    uint64_t inputAddr                   = taskArg->inputAddr_;
    uint64_t outputAddr                  = taskArg->outputAddr_;
    uint64_t tokenInfo                   = taskArg->token_;
    uint64_t scratchAddr                 = taskArg->scratchAddr_;
    uint64_t currentRankSliceInputOffset = taskArg->inputSliceStride_ * rankId_;
    uint64_t currentRankSliceOutputOffset= taskArg->outputSliceStride_ * rankId_;
    uint64_t inputRepeatStride           = taskArg->inputRepeatStride_;
    uint64_t outputRepeatStride          = taskArg->outputRepeatStride_;
    uint64_t normalSliceSize             = taskArg->normalSliceSize_;
    uint64_t lastSliceSize               = taskArg->lastSliceSize_;
    uint64_t repeatNum                   = taskArg->repeatNum_;

    std::vector<uint64_t> taskArgs = {
        inputAddr,         outputAddr,         tokenInfo,
        scratchAddr,       currentRankSliceInputOffset,
        currentRankSliceOutputOffset,          inputRepeatStride,
        outputRepeatStride, normalSliceSize,   lastSliceSize,
        repeatNum
    };

    HCCL_INFO("[CcuKernelReduceScatterMesh1DFlatten] TaskArgs: inputAddr[%llu], outputAddr[%llu], "
               "scratchAddr[%llu], currentRankSliceInputOffset[%llu], currentRankSliceOutputOffset[%llu], "
               "inputRepeatStride[%llu], outputRepeatStride[%llu], "
               "normalSliceSize[%llu], lastSliceSize[%llu], repeatNum[%llu]",
               inputAddr, outputAddr, scratchAddr, currentRankSliceInputOffset, currentRankSliceOutputOffset,
               inputRepeatStride, outputRepeatStride, normalSliceSize, lastSliceSize, UINT64_MAX - repeatNum);

    return taskArgs;
}

} // namespace ops_hccl