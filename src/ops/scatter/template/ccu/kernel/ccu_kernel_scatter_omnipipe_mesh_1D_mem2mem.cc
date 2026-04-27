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
constexpr int SCRATCH_XN_ID = 1;
constexpr int TOKEN_XN_ID   = 2;
constexpr int POST_SYNC_ID  = 3;
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
    currentStep_   = kernelArg->currentStep_;
    isSameAxisAsRoot_ = kernelArg->isSameAxisAsRoot_;
    totalStep_ = kernelArg->totalStep_;
    
    HCCL_INFO("[CcuKernelScatterOmniPipeMesh1DMem2Mem] Init, KernelArgs are rankId[%u], "
        "rankSize_[%u], dataType[%d], rootId[%u], currentStep[%u], isSameAxisAsRoot[%d], totalStep[%u]", rankId_, rankSize_, dataType_, rootId_, currentStep_, isSameAxisAsRoot_, totalStep_);
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
            scratch_.push_back(CreateVariable());
            token_.push_back(CreateVariable());
        } else {
            HCCL_DEBUG("[CcuKernelScatterOmniPipeMesh1DMem2Mem] MyRank[%u], PeerId[%u], ChannelId[%u]",
                       rankId_, peerId, channelIdx);
            
            CcuRep::Variable inputVar, outputVar, scratchVar, tokenVar;
            CHK_RET(CreateVariable(channels_[channelIdx], INPUT_XN_ID, &inputVar));
            input_.push_back(inputVar);
            CHK_RET(CreateVariable(channels_[channelIdx], INPUT_XN_ID, &outputVar));
            output_.push_back(outputVar);
            CHK_RET(CreateVariable(channels_[channelIdx], SCRATCH_XN_ID, &scratchVar));
            scratch_.push_back(scratchVar);
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
    scratchMem_.reserve(rankSize_);
    remoteScratchMem_.reserve(rankSize_);

    for (uint64_t rankIdx = 0; rankIdx < rankSize_; rankIdx++) {
        inputMem_.push_back(CreateLocalAddr());
        scratchMem_.push_back(CreateLocalAddr());
        if (rankIdx != rankId_) {
            outputMem_.push_back(CreateRemoteAddr());
            remoteScratchMem_.push_back(CreateRemoteAddr());
        }
    }

    event_ = CreateCompletedEvent();
    return HcclResult::HCCL_SUCCESS;
}

void CcuKernelScatterOmniPipeMesh1DMem2Mem::LoadArgs()
{
    Load(input_[rankId_]);
    Load(output_[rankId_]);
    Load(scratch_[rankId_]);
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
    uint32_t allBit = 1 << INPUT_XN_ID | 1 << SCRATCH_XN_ID | 1 << TOKEN_XN_ID;
    for (auto channel : channels_) {
        NotifyRecord(channel, CKE_IDX_0, INPUT_XN_ID, input_[rankId_], 1 << INPUT_XN_ID);
        NotifyRecord(channel, CKE_IDX_0, SCRATCH_XN_ID, scratch_[rankId_], 1 << SCRATCH_XN_ID);
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

void CcuKernelScatterOmniPipeMesh1DMem2Mem::DoRepeatScatter()
{
    uint32_t remoteIdx = 0;
    uint64_t scratchOffset = 0;
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

        scratchMem_[curId].token = token_[rankId_];
        scratchMem_[curId].addr = scratch_[rankId_];
        scratchMem_[curId].addr += outputSliceStride_;
        scratchMem_[curId].addr += outputOmniPipeSliceStride_;
        
    }
    if (currentStep_ == 0) {
        // 第一步：root 发送到 scratch
        if (rankId_ == rootId_) {
            DoScatter();
        }
    } else if (currentStep_ == totalStep_){
        // 最后一步：root 发送到 output，同轴节点转发scratch 中的数据
        if (isSameAxisAsRoot_){
            DoScatterFromScratch();
        }
        if (rankId_ == rootId_){
            DoScatter2Out();
        }
    } else {
        // 中间步骤：root 继续发送对角数据，同轴节点开始转发 scratch 中的数据
        if (isSameAxisAsRoot_){
            DoScatterFromScratch();
        }
        if (rankId_ == rootId_) {
            DoScatter();
        }
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
                WriteNb(channels_[channelId], remoteScratchMem_[channelId], inputMem_[rankIdx], sliceSize_, event_);
                channelId++;
            } 
        }
    }

    event_.SetMask((1 << rankSize_) - 1);
    WaitEvent(event_);
}

void CcuKernelScatterOmniPipeMesh1DMem2Mem::DoScatterFromScratch()
{
    uint32_t channelId = 0;
    CcuRep::LocalAddr myOutput = CreateLocalAddr();
    myOutput.addr  = output_[rankId_];
    myOutput.token = token_[rankId_];

    for (uint64_t rankIdx = 0; rankIdx < rankSize_; rankIdx++) {
        event_.SetMask(1 << rankIdx);
        
        if (rankIdx == rankId_) {
            // 从 scratch 读取自己的数据
            CCU_IF(sliceSize_ != 0) {
                LocalCopyNb(myOutput, scratchMem_[rankIdx], sliceSize_, event_);
            }
        } else {
            // 转发 scratch 中的数据给其他节点
            CCU_IF(sliceSize_ != 0) {
                WriteNb(channels_[channelId], outputMem_[channelId], scratchMem_[rankIdx], sliceSize_, event_);
                channelId++;
            }
        }
    }

    event_.SetMask((1 << rankSize_) - 1);
    WaitEvent(event_);
}

void CcuKernelScatterOmniPipeMesh1DMem2Mem::DoScatter2Out()
{
    uint32_t channelId = 0;
    CcuRep::LocalAddr myOutput = CreateLocalAddr();
    myOutput.addr  = output_[rankId_];
    myOutput.token = token_[rankId_];

    for (uint64_t rankIdx = 0; rankIdx < rankSize_; rankIdx++) {
        event_.SetMask(1 << rankIdx);
        
        if (rankIdx == rankId_) {
            // 从 scratch 读取自己的数据
            CCU_IF(sliceSize_ != 0) {
                LocalCopyNb(myOutput, scratchMem_[rankIdx], sliceSize_, event_);
            }
        } else {
            // 转发 scratch 中的数据给其他节点
            CCU_IF(sliceSize_ != 0) {
                WriteNb(channels_[channelId], outputMem_[channelId], inputMem_[rankIdx], sliceSize_, event_);
                channelId++;
            }
        }
    }

    event_.SetMask((1 << rankSize_) - 1);
    WaitEvent(event_);
}

HcclResult CcuKernelScatterOmniPipeMesh1DMem2Mem::Algorithm()
{
    HCCL_INFO("[CcuKernelScatterOmniPipeMesh1DMem2Mem] ScatterOmniPipeMesh1D Mem2Mem run");

    CHK_RET(InitResource());
    LoadArgs();
    PreSync();
    DoRepeatScatter();
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
        taskArg.scratchAddr_,
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
        "scratchAddr[%llu], sliceSize[%llu], offSet[%llu], localCopyFlag[%llu]", 
        taskArg.inputAddr_, taskArg.outputAddr_, taskArg.scratchAddr_,
        taskArg.sliceSize_, taskArg.offSet_, taskArg.localCopyFlag_);

    return taskArgs;
}

} // namespace ops_hccl
