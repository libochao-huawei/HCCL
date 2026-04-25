/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "ccu_kernel_all_gather_omnipipe_nhr1d_multi_jetty_mem2mem.h"

namespace ops_hccl {

constexpr uint16_t OUTPUT_XN_ID = 1;
constexpr uint16_t TOKEN_XN_ID = 2;
constexpr uint16_t POST_SYNC_ID = 3;
constexpr uint16_t STEP_PRE_SYNC_ID = 4;
constexpr uint16_t STEP_POST_SYNC_ID = 5;
constexpr uint16_t CKE_IDX_0 = 0;

hcomm::CcuKernelSignature CcuKernelArgAllGatherOmniPipeNHR1DMultiJettyMem2Mem::GetKernelSignature() const
{
    hcomm::CcuKernelSignature signature;
    GenerateCcuKernelSignature(signature, "CcuKernelArgAllGatherOmniPipeNHR1DMultiJettyMem2Mem", opParam_,
                               subCommRanks_);
    return signature;
}

CcuKernelAllGatherOmniPipeNHR1DMultiJettyMem2Mem::CcuKernelAllGatherOmniPipeNHR1DMultiJettyMem2Mem(
    const hcomm::CcuKernelArg &arg)
    : CcuKernelAlgBase(arg)
{
    const CcuKernelArgAllGatherOmniPipeNHR1DMultiJettyMem2Mem *kernelArg =
        dynamic_cast<const CcuKernelArgAllGatherOmniPipeNHR1DMultiJettyMem2Mem *>(&arg);
    if (kernelArg != nullptr) {
        rankSize_ = kernelArg->rankSize_;
        rankId_ = kernelArg->rankId_;
        channels_ = kernelArg->channels;
        jettyNum_ = kernelArg->jettyNum_;
        stepInfo_ = kernelArg->stepInfo_;
        toChannelIdx_ = kernelArg->toChannelIdx_;
        fromChannelIdx_ = kernelArg->fromChannelIdx_;
        localSize_ = channels_.size();
        myRankIdx_ = localSize_;
    }
}

HcclResult CcuKernelAllGatherOmniPipeNHR1DMultiJettyMem2Mem::InitResources()
{
    if (rankSize_ > 1 && channels_.empty()) {
        HCCL_ERROR("[CcuKernelAllGatherOmniPipeNHR1DMultiJettyMem2Mem] channels is empty.");
        return HcclResult::HCCL_E_INTERNAL;
    }
    input_ = CreateVariable();
    for (uint32_t channelIdx = 0; channelIdx < localSize_; channelIdx++) {
        CcuRep::Variable outputVar;
        CcuRep::Variable tokenVar;
        CHK_RET(CreateVariable(channels_[channelIdx], OUTPUT_XN_ID, &outputVar));
        CHK_RET(CreateVariable(channels_[channelIdx], TOKEN_XN_ID, &tokenVar));
        output_.push_back(outputVar);
        token_.push_back(tokenVar);
    }
    output_.push_back(CreateVariable());
    token_.push_back(CreateVariable());
    srcOffset_ = CreateVariable();
    dstOffset_ = CreateVariable();
    sliceSize_ = CreateVariable();
    sliceSizePerJetty_ = CreateVariable();
    lastSliceSizePerJetty_ = CreateVariable();
    doStepPreSync_ = CreateVariable();
    doStepPostSync_ = CreateVariable();
    srcMem_ = CreateLocalAddr();
    dstMem_ = CreateRemoteAddr();
    srcMemTmp_ = CreateLocalAddr();
    dstMemTmp_ = CreateRemoteAddr();
    event_ = CreateCompletedEvent();
    return HcclResult::HCCL_SUCCESS;
}

HcclResult CcuKernelAllGatherOmniPipeNHR1DMultiJettyMem2Mem::LoadArgs()
{
    Load(input_);
    Load(output_[myRankIdx_]);
    Load(token_[myRankIdx_]);
    Load(srcOffset_);
    Load(dstOffset_);
    Load(sliceSize_);
    Load(sliceSizePerJetty_);
    Load(lastSliceSizePerJetty_);
    Load(doStepPreSync_);
    Load(doStepPostSync_);
    return HcclResult::HCCL_SUCCESS;
}

HcclResult CcuKernelAllGatherOmniPipeNHR1DMultiJettyMem2Mem::PreSync()
{
    for (ChannelHandle channel : channels_) {
        CHK_RET(NotifyRecord(channel, CKE_IDX_0, OUTPUT_XN_ID, output_[myRankIdx_], 1 << OUTPUT_XN_ID));
        CHK_RET(NotifyRecord(channel, CKE_IDX_0, TOKEN_XN_ID, token_[myRankIdx_], 1 << TOKEN_XN_ID));
    }
    uint16_t allBit = 1 << OUTPUT_XN_ID | 1 << TOKEN_XN_ID;
    for (ChannelHandle channel : channels_) {
        CHK_RET(NotifyWait(channel, CKE_IDX_0, allBit));
    }
    return HcclResult::HCCL_SUCCESS;
}

HcclResult CcuKernelAllGatherOmniPipeNHR1DMultiJettyMem2Mem::PostSync()
{
    for (ChannelHandle channel : channels_) {
        CHK_RET(NotifyRecord(channel, CKE_IDX_0, 1 << POST_SYNC_ID));
    }
    for (ChannelHandle channel : channels_) {
        CHK_RET(NotifyWait(channel, CKE_IDX_0, 1 << POST_SYNC_ID));
    }
    return HcclResult::HCCL_SUCCESS;
}

HcclResult CcuKernelAllGatherOmniPipeNHR1DMultiJettyMem2Mem::DoOmniPipeNHRSendRecv()
{
    uint32_t toRankIdx = toChannelIdx_;
    uint32_t fromRankIdx = fromChannelIdx_;
    if (toRankIdx >= channels_.size() || fromRankIdx >= channels_.size()) {
        HCCL_ERROR("[CcuKernelAllGatherOmniPipeNHR1DMultiJettyMem2Mem] invalid channel idx, to[%u], from[%u], "
                   "channelSize[%zu].", toRankIdx, fromRankIdx, channels_.size());
        return HCCL_E_INTERNAL;
    }
    ChannelHandle sendChannel = channels_[toRankIdx];
    ChannelHandle recvChannel = channels_[fromRankIdx];

    CCU_IF(doStepPreSync_ != 0)
    {
        CHK_RET(NotifyRecord(recvChannel, CKE_IDX_0, 1 << STEP_PRE_SYNC_ID));
        CHK_RET(NotifyWait(sendChannel, CKE_IDX_0, 1 << STEP_PRE_SYNC_ID));
    }

    srcMem_.addr = output_[myRankIdx_];
    srcMem_.addr += srcOffset_;
    srcMem_.token = token_[myRankIdx_];
    dstMem_.addr = output_[toRankIdx];
    dstMem_.addr += dstOffset_;
    dstMem_.token = token_[toRankIdx];
    srcMemTmp_ = srcMem_;
    dstMemTmp_ = dstMem_;

    CCU_IF(sliceSizePerJetty_ != 0)
    {
        for (uint32_t i = 0; i < jettyNum_ - 1; ++i) {
            event_.SetMask(1 << i);
            CHK_RET(WriteNb(sendChannel, dstMemTmp_, srcMemTmp_, sliceSizePerJetty_, event_));
            srcMemTmp_.addr += sliceSizePerJetty_;
            dstMemTmp_.addr += sliceSizePerJetty_;
        }
    }
    CCU_IF(sliceSizePerJetty_ == 0)
    {
        for (uint32_t i = 0; i < jettyNum_ - 1; ++i) {
            event_.SetMask(1 << i);
            CHK_RET(RecordEvent(event_));
        }
    }
    CCU_IF(lastSliceSizePerJetty_ != 0)
    {
        event_.SetMask(1 << (jettyNum_ - 1));
        CHK_RET(WriteNb(sendChannel, dstMemTmp_, srcMemTmp_, lastSliceSizePerJetty_, event_));
    }
    CCU_IF(lastSliceSizePerJetty_ == 0)
    {
        event_.SetMask(1 << (jettyNum_ - 1));
        CHK_RET(RecordEvent(event_));
    }
    event_.SetMask((1 << jettyNum_) - 1);
    CHK_RET(WaitEvent(event_));

    CCU_IF(doStepPostSync_ != 0)
    {
        CHK_RET(NotifyRecord(sendChannel, CKE_IDX_0, 1 << STEP_POST_SYNC_ID));
        CHK_RET(NotifyWait(recvChannel, CKE_IDX_0, 1 << STEP_POST_SYNC_ID));
    }
    return HcclResult::HCCL_SUCCESS;
}

HcclResult CcuKernelAllGatherOmniPipeNHR1DMultiJettyMem2Mem::Algorithm()
{
    CHK_RET(InitResources());
    CHK_RET(LoadArgs());
    CHK_RET(PreSync());
    CHK_RET(DoOmniPipeNHRSendRecv());
    CHK_RET(PostSync());
    return HcclResult::HCCL_SUCCESS;
}

std::vector<uint64_t> CcuKernelAllGatherOmniPipeNHR1DMultiJettyMem2Mem::GeneArgs(const hcomm::CcuTaskArg &arg)
{
    const CcuTaskArgAllGatherOmniPipeNHR1DMultiJettyMem2Mem *taskArg =
        dynamic_cast<const CcuTaskArgAllGatherOmniPipeNHR1DMultiJettyMem2Mem *>(&arg);
    return {taskArg->inputAddr_,      taskArg->outputAddr_,       taskArg->token_,          taskArg->srcOffset_,
            taskArg->dstOffset_,      taskArg->sliceSize_,        taskArg->sliceSizePerJetty_,
            taskArg->lastSliceSizePerJetty_, taskArg->doStepPreSync_, taskArg->doStepPostSync_};
}

} // namespace ops_hccl
