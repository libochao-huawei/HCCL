/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "ccu_kernel_all_gather_omnipipe_mesh1d_mem2mem.h"

namespace ops_hccl {

constexpr int OUTPUT_XN_ID = 1;
constexpr int TOKEN_XN_ID = 2;
constexpr int CKE_IDX_0 = 0;
constexpr int POST_SYNC_ID = 3;

hcomm::CcuKernelSignature CcuKernelArgAllGatherOmniPipeMesh1DMem2Mem::GetKernelSignature() const
{
    hcomm::CcuKernelSignature signature;
    GenerateCcuKernelSignature(signature, "CcuKernelArgAllGatherOmniPipeMesh1DMem2Mem", opParam_, subCommRanks_);
    return signature;
}

CcuKernelAllGatherOmniPipeMesh1DMem2Mem::CcuKernelAllGatherOmniPipeMesh1DMem2Mem(
    const hcomm::CcuKernelArg &arg)
    : CcuKernelAlgBase(arg)
{
    const CcuKernelArgAllGatherOmniPipeMesh1DMem2Mem *kernelArg =
        dynamic_cast<const CcuKernelArgAllGatherOmniPipeMesh1DMem2Mem *>(&arg);
    if (kernelArg != nullptr) {
        rankSize_ = kernelArg->dimSize_;
        rankId_ = kernelArg->rankId_;
        channels_ = kernelArg->channels;
    }
    HCCL_INFO("[CcuKernelAllGatherOmniPipeMesh1DMem2Mem] init, rankId[%u], rankSize[%llu]", rankId_,
               rankSize_);
}

HcclResult CcuKernelAllGatherOmniPipeMesh1DMem2Mem::InitResource()
{
    if (rankSize_ > 1 && channels_.empty()) {
        HCCL_ERROR("[CcuKernelAllGatherOmniPipeMesh1DMem2Mem] channels is empty.");
        return HcclResult::HCCL_E_INTERNAL;
    }
    localInput_ = CreateVariable();
    uint16_t channelIdx = 0;
    for (uint64_t peerId = 0; peerId < rankSize_; peerId++) {
        if (peerId == rankId_) {
            output_.push_back(CreateVariable());
            token_.push_back(CreateVariable());
        } else {
            CcuRep::Variable outputVar;
            CcuRep::Variable tokenVar;
            CHK_RET(CreateVariable(channels_[channelIdx], OUTPUT_XN_ID, &outputVar));
            CHK_RET(CreateVariable(channels_[channelIdx], TOKEN_XN_ID, &tokenVar));
            output_.push_back(outputVar);
            token_.push_back(tokenVar);
            channelIdx++;
        }
    }
    srcOffset_ = CreateVariable();
    dstOffset_ = CreateVariable();
    sliceSize_ = CreateVariable();
    isSrcDstEqual_ = CreateVariable();
    localGoSize_ = CreateGroupOpSize();
    src_ = CreateLocalAddr();
    localDst_ = CreateLocalAddr();
    for (uint64_t rankIdx = 0; rankIdx < rankSize_; rankIdx++) {
        if (rankIdx == rankId_) {
            remoteDst_.push_back({});
        } else {
            remoteDst_.push_back(CreateRemoteAddr());
        }
    }
    event_ = CreateCompletedEvent();
    return HcclResult::HCCL_SUCCESS;
}

void CcuKernelAllGatherOmniPipeMesh1DMem2Mem::LoadArgs()
{
    Load(localInput_);
    Load(output_[rankId_]);
    Load(token_[rankId_]);
    Load(srcOffset_);
    Load(dstOffset_);
    Load(sliceSize_);
    Load(isSrcDstEqual_);
    Load(localGoSize_);
}

void CcuKernelAllGatherOmniPipeMesh1DMem2Mem::PreSync()
{
    for (ChannelHandle channel : channels_) {
        NotifyRecord(channel, CKE_IDX_0, OUTPUT_XN_ID, output_[rankId_], 1 << OUTPUT_XN_ID);
        NotifyRecord(channel, CKE_IDX_0, TOKEN_XN_ID, token_[rankId_], 1 << TOKEN_XN_ID);
    }
    uint16_t allBit = 1 << OUTPUT_XN_ID | 1 << TOKEN_XN_ID;
    for (ChannelHandle channel : channels_) {
        NotifyWait(channel, CKE_IDX_0, allBit);
    }
}

void CcuKernelAllGatherOmniPipeMesh1DMem2Mem::PostSync()
{
    for (ChannelHandle channel : channels_) {
        NotifyRecord(channel, CKE_IDX_0, 1 << POST_SYNC_ID);
    }
    for (ChannelHandle channel : channels_) {
        NotifyWait(channel, CKE_IDX_0, 1 << POST_SYNC_ID);
    }
}

void CcuKernelAllGatherOmniPipeMesh1DMem2Mem::DoOmniPipeMeshAllGather()
{
    src_.addr = localInput_;
    src_.addr += srcOffset_;
    src_.token = token_[rankId_];
    localDst_.addr = output_[rankId_];
    localDst_.addr += dstOffset_;
    localDst_.token = token_[rankId_];

    uint32_t channelId = 0;
    for (uint64_t rankIdx = 0; rankIdx < rankSize_; rankIdx++) {
        if (rankIdx == rankId_) {
            event_.SetMask(1 << rankIdx);
            CCU_IF(isSrcDstEqual_ != 0)
            {
                RecordEvent(event_);
            }
            CCU_IF(isSrcDstEqual_ == 0)
            {
                GroupCopy(localDst_, src_, localGoSize_);
                RecordEvent(event_);
            }
        } else {
            remoteDst_[rankIdx].addr = output_[rankIdx];
            remoteDst_[rankIdx].addr += dstOffset_;
            remoteDst_[rankIdx].token = token_[rankIdx];
            CCU_IF(sliceSize_ != 0)
            {
                event_.SetMask(1 << rankIdx);
                WriteNb(channels_[channelId], remoteDst_[rankIdx], src_, sliceSize_, event_);
            }
            CCU_IF(sliceSize_ == 0)
            {
                event_.SetMask(1 << rankIdx);
                RecordEvent(event_);
            }
            channelId++;
        }
    }
    event_.SetMask((1 << rankSize_) - 1);
    WaitEvent(event_);
}

HcclResult CcuKernelAllGatherOmniPipeMesh1DMem2Mem::Algorithm()
{
    HCCL_INFO("[CcuKernelAllGatherOmniPipeMesh1DMem2Mem][Algorithm] start.");
    CHK_RET(InitResource());
    LoadArgs();
    PreSync();
    DoOmniPipeMeshAllGather();
    PostSync();
    HCCL_INFO("[CcuKernelAllGatherOmniPipeMesh1DMem2Mem][Algorithm] end.");
    return HcclResult::HCCL_SUCCESS;
}

std::vector<uint64_t> CcuKernelAllGatherOmniPipeMesh1DMem2Mem::GeneArgs(const hcomm::CcuTaskArg &arg)
{
    const CcuTaskArgAllGatherOmniPipeMesh1DMem2Mem *taskArg =
        dynamic_cast<const CcuTaskArgAllGatherOmniPipeMesh1DMem2Mem *>(&arg);
    auto goSize = CalGoSize(taskArg->sliceSize_);
    return {taskArg->inputAddr_,       taskArg->outputAddr_, taskArg->token_,      taskArg->srcOffset_,
            taskArg->dstOffset_,       taskArg->sliceSize_,  taskArg->isSrcDstEqual_, goSize[0],
            goSize[1],                 goSize[2],            goSize[3]};
}

} // namespace ops_hccl
