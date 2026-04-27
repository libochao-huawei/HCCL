/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "ccu_kernel_scatter_omnipipe_mesh_1D_mem2mem.h"

namespace ops_hccl {
using namespace hcomm;

constexpr int INPUT_XN_ID   = 0;
constexpr int TOKEN_XN_ID   = 1;
constexpr int POST_SYNC_ID  = 2;
constexpr int CKE_IDX_0     = 0;

CcuKernelScatterOmniPipeMesh1DMem2Mem::CcuKernelScatterOmniPipeMesh1DMem2Mem(const CcuKernelArg &arg)
    : CcuKernelAlgBase(arg)
{
    const CcuKernelArgScatterOmniPipeMesh1DMem2Mem *kernelArg = 
        dynamic_cast<const CcuKernelArgScatterOmniPipeMesh1DMem2Mem *>(&arg);
    rankId_        = kernelArg->rankId_;
    rankSize_      = kernelArg->dimSize_;
    channels_      = kernelArg->channels;
    dataType_      = kernelArg->opParam_.DataDes.dataType;
    rootId_        = kernelArg->rootId_;
    subCommRanks_  = kernelArg->subCommRanks_;
    
    HCCL_INFO("[CcuKernelScatterOmniPipeMesh1DMem2Mem] Init, KernelArgs are rankId[%u], "
        "rankSize_[%u], dataType[%d], rootId[%u]", rankId_, rankSize_, dataType_, rootId_);
}

HcclResult CcuKernelScatterOmniPipeMesh1DMem2Mem::InitResource()
{
    uint16_t channelIdx = 0;
    if (channels_.size() == 0) {
        HCCL_ERROR("[CcuKernelScatterOmniPipeMesh1DMem2Mem] channels is empty!");
        return HcclResult::HCCL_E_INTERNAL;
    }

    for (uint64_t peerId = 0; peerId < rankSize_; peerId++) {
        if (peerId == rankId_) {
            input_.push_back(CreateVariable());
            output_.push_back(CreateVariable());
            token_.push_back(CreateVariable());
        } else {
            HCCL_DEBUG("[CcuKernelScatterOmniPipeMesh1DMem2Mem] MyRank[%u], PeerId[%u], ChannelId[%u]",
                       rankId_, peerId, channelIdx);
            
            CcuRep::Variable inputVar, outputVar, tokenVar;
            CHK_RET(CreateVariable(channels_[channelIdx], INPUT_XN_ID, &inputVar));
            input_.push_back(inputVar);
            CHK_RET(CreateVariable(channels_[channelIdx], INPUT_XN_ID, &outputVar));
            output_.push_back(outputVar);
            CHK_RET(CreateVariable(channels_[channelIdx], TOKEN_XN_ID, &tokenVar));
            token_.push_back(tokenVar);
            channelIdx++;
        }
    }

    offSet_ = CreateVariable();
    sliceSize_ = CreateVariable();
    inputSliceStride_ = CreateVariable();
    outputSliceStride_ = CreateVariable();
    inputOmniPipeSliceStride_ = CreateVariable();
    outputOmniPipeSliceStride_ = CreateVariable();
    localCopyFlag_ = CreateVariable();

    inputMem_.reserve(rankSize_);
    outputMem_.reserve(rankSize_);

    for (uint64_t rankIdx = 0; rankIdx < rankSize_; rankIdx++) {
        inputMem_.push_back(CreateLocalAddr());
        if (rankIdx != rankId_) {
            outputMem_.push_back(CreateRemoteAddr());
        }
    }

    event_ = CreateCompletedEvent();
    return HcclResult::HCCL_SUCCESS;
}

void CcuKernelScatterOmniPipeMesh1DMem2Mem::LoadArgs()
{
    Load(input_[rankId_]);
    Load(output_[rankId_]);
    Load(sliceSize_);
    Load(offSet_);
    Load(token_[rankId_]);
    Load(localCopyFlag_);   
    Load(inputSliceStride_);
    Load(outputSliceStride_);
    Load(inputOmniPipeSliceStride_);
    Load(outputOmniPipeSliceStride_);
}

void CcuKernelScatterOmniPipeMesh1DMem2Mem::PreSync()
{
    uint32_t allBit = 1 << INPUT_XN_ID | 1 << TOKEN_XN_ID;
    for (auto channel : channels_) {
        NotifyRecord(channel, CKE_IDX_0, INPUT_XN_ID, input_[rankId_], 1 << INPUT_XN_ID);
        NotifyRecord(channel, CKE_IDX_0, TOKEN_XN_ID, token_[rankId_], 1 << TOKEN_XN_ID);
    }
    for (auto channel : channels_) {
        NotifyWait(channel, CKE_IDX_0, allBit);
    }
}

void CcuKernelScatterOmniPipeMesh1DMem2Mem::PostSync()
{
    for (auto channel : channels_) {
        NotifyRecord(channel, CKE_IDX_0, 1 << POST_SYNC_ID);
    }
    for (auto channel : channels_) {
        NotifyWait(channel, CKE_IDX_0, 1 << POST_SYNC_ID);
    }
}

void CcuKernelScatterOmniPipeMesh1DMem2Mem::DoScatter()
{
    uint32_t channelId = 0;
    CcuRep::LocalAddr myOutput = CreateLocalAddr();
    myOutput.addr  = output_[rankId_];
    myOutput.token = token_[rankId_];

    for (uint64_t rankIdx = 0; rankIdx < rankSize_; rankIdx++) {
        event_.SetMask(1 << rankIdx);
        
        if (rankIdx == rankId_) {
            CCU_IF(localCopyFlag_ == 0)
            {
                LocalCopyNb(myOutput, inputMem_[rankIdx], sliceSize_, event_);
            }
            CCU_IF(localCopyFlag_ != 0)
            {
                RecordEvent(event_);
            }
        } else {
            CCU_IF(sliceSize_ != 0) {
                WriteNb(channels_[channelId], outputMem_[channelId], inputMem_[rankIdx], sliceSize_, event_);
                channelId++;
            } 
        }
    }

    event_.SetMask((1 << rankSize_) - 1);
    WaitEvent(event_);
}

void CcuKernelScatterOmniPipeMesh1DMem2Mem::DoLocalCopy()
{
    CcuRep::LocalAddr myOutput = CreateLocalAddr();
    myOutput.addr  = output_[rankId_];
    myOutput.token = token_[rankId_];
    LocalCopyNb(myOutput, inputMem_[rankId_], sliceSize_, event_);
}

HcclResult CcuKernelScatterOmniPipeMesh1DMem2Mem::Algorithm()
{
    HCCL_INFO("[CcuKernelScatterOmniPipeMesh1DMem2Mem] ScatterOmniPipeMesh1D Mem2Mem run");

    CHK_RET(InitResource());
    LoadArgs();
    PreSync();

    uint32_t remoteIdx = 0;
    for (uint64_t curId = 0; curId < rankSize_; curId++) {
        inputMem_[curId].token = token_[curId];

        inputMem_[curId].addr = input_[rankId_];
        inputMem_[curId].addr += inputSliceStride_;
        inputMem_[curId].addr += inputOmniPipeSliceStride_;

        if (curId != rankId_) {
            outputMem_[remoteIdx].token = token_[curId];
            outputMem_[remoteIdx].addr = output_[curId];
            outputMem_[remoteIdx].addr += outputSliceStride_;
            outputMem_[remoteIdx].addr += outputOmniPipeSliceStride_;
            remoteIdx++;
        }
    }

    DoScatter();

    PostSync();

    HCCL_INFO("[CcuKernelScatterOmniPipeMesh1DMem2Mem] ScatterOmniPipeMesh1D Mem2Mem end");
    return HcclResult::HCCL_SUCCESS;
}

std::vector<uint64_t> CcuKernelScatterOmniPipeMesh1DMem2Mem::GeneArgs(const CcuTaskArg &arg)
{
    const auto taskArg = dynamic_cast<const CcuTaskArgScatterOmniPipeMesh1DMem2Mem &>(arg);

    std::vector<uint64_t> taskArgs = {
        taskArg.inputAddr_,
        taskArg.outputAddr_,
        taskArg.sliceSize_,
        taskArg.offSet_,
        taskArg.token_,
        taskArg.localCopyFlag_,
        taskArg.inputSliceStride_,
        taskArg.outputSliceStride_,
        taskArg.inputOmniPipeSliceStride_,
        taskArg.outputOmniPipeSliceStride_
    };

    HCCL_INFO("[CcuKernelScatterOmniPipeMesh1DMem2Mem] TaskArgs: inputAddr[%llu], outputAddr[%llu], "
        "sliceSize[%llu], offSet[%llu], localCopyFlag[%llu]", taskArg.inputAddr_, taskArg.outputAddr_,
        taskArg.sliceSize_, taskArg.offSet_, taskArg.localCopyFlag_);

    return taskArgs;
}

} // namespace ops_hccl
