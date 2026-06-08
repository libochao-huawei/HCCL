/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "ccu_kernel_gather_omnipipe_mesh_1d_mem2mem.h"

namespace ops_hccl {
using namespace hcomm;

constexpr int INPUT_XN_ID   = 0;
// constexpr int OUTPUT_XN_ID  = 1;
// constexpr int SCRATCH_XN_ID = 2;
constexpr int TOKEN_XN_ID   = 1;
constexpr int POST_SYNC_ID   = 4;
constexpr int CKE_IDX_0     = 0;

CcuKernelGatherOmniPipeMesh1DMem2Mem::CcuKernelGatherOmniPipeMesh1DMem2Mem(const hcomm::CcuKernelArg& arg)
    : CcuKernelAlgBase(arg)
{
    HCCL_INFO("[CcuKernelGatherOmniPipeMesh1DMem2Mem] STARTS");
    const CcuKernelArgGatherOmniPipeMesh1DMem2Mem* kernelArg =
        dynamic_cast<const CcuKernelArgGatherOmniPipeMesh1DMem2Mem*>(&arg);
    


    rankId_        = kernelArg->rankId_;
    rankSize_      = kernelArg->dimSize_;
    channels_      = kernelArg->channels;
    rootId_        = kernelArg->rootId_;
    subCommRanks_  = kernelArg->subCommRanks_;
    dataType_      = kernelArg->opParam_.DataDes.dataType;
    outputDataType_ = kernelArg->opParam_.DataDes.outputType;
    subRankIdx2RankIdx_  = kernelArg->subRankIdx2RankIdx_;
    
    for (auto item : subRankIdx2RankIdx_) {
        HCCL_DEBUG("[CcuKernelScatterOmniPipeMesh1DMem2Mem] subRankIdx2RankIdx_: %u -> %u", item.first, item.second);
    }
    // HCCL_DEBUG("[CcuKernelGatherOmniPipeMesh1DMem2Mem] rankSize=%llu, rankIdx=%u, rootIdx=%u",
    //            rankSize_, rankId_, rootId_);
}

HcclResult CcuKernelGatherOmniPipeMesh1DMem2Mem::InitResource()
{
    HCCL_DEBUG("[%s] start", __func__);
    uint16_t channelIdx = 0;
    if (channels_.size() == 0) {
        HCCL_ERROR("[CcuKernelGatherOmniPipeMesh1DMem2Mem] channels is empty!");
        return HcclResult::HCCL_E_INTERNAL;
    }

    // 按照rank号从小到大遍历channels，遇到本rank就填充本地资源，否则依次取远端资源，要求给框架返回的Link同样是按顺序排列的
    for (uint64_t peerId = 0; peerId < rankSize_; peerId++) {
        if (peerId == rankId_) {
            input_.push_back(CreateVariable());
            // output_.push_back(CreateVariable());
            token_.push_back(CreateVariable());
        } else {
            HCCL_DEBUG("[CcuKernelGatherOmniPipeMesh1DMem2Mem] MyRank[%u], PeerId[%u], ChannelId[%u]",
                       rankId_, peerId, channelIdx);
            
            CcuRep::Variable inputVar, tokenVar;
            CHK_RET(CreateVariable(channels_[channelIdx], INPUT_XN_ID, &inputVar));
            input_.push_back(inputVar);
            CHK_RET(CreateVariable(channels_[channelIdx], TOKEN_XN_ID, &tokenVar));
            token_.push_back(tokenVar);
            channelIdx++;
        }
    }

    output_ = CreateVariable();
    sliceSize_ = CreateVariable();
    inputSliceStride_ = CreateVariable();
    outputSliceStride_ = CreateVariable();
    inputOmniPipeSliceStride_ = CreateVariable();
    outputOmniPipeSliceStride_ = CreateVariable();
    localCopyFlag_ = CreateVariable();
    isStepOne_ = CreateVariable();
    isLastStep_ = CreateVariable();
    ifNewRoot_ = CreateVariable();

    inputMem_.reserve(rankSize_);
    outputMem_.reserve(rankSize_);

    for (uint64_t rankIdx = 0; rankIdx < rankSize_; rankIdx++) {
        inputMem_.push_back(CreateRemoteAddr());
        outputMem_.push_back(CreateLocalAddr());
    }

    event_ = CreateCompletedEvent();
    return HcclResult::HCCL_SUCCESS;
}

void CcuKernelGatherOmniPipeMesh1DMem2Mem::LoadArgs()
{
    HCCL_DEBUG("[%s] LoadArgs start", __func__);

    Load(input_[rankId_]);
    Load(output_);
    Load(token_[rankId_]);
    Load(localCopyFlag_);
    Load(sliceSize_);
    Load(inputSliceStride_);
    Load(outputSliceStride_);
    Load(inputOmniPipeSliceStride_);
    Load(outputOmniPipeSliceStride_);
    Load(isStepOne_);
    Load(isLastStep_);
    Load(ifNewRoot_);
    // Load(groupOpSize_); // 干啥的

    HCCL_DEBUG("[%s] LoadArgs end", __func__);
}

void CcuKernelGatherOmniPipeMesh1DMem2Mem::PreSync()
{
    HCCL_DEBUG("[CcuKernelGatherOmniPipeMesh1DMem2Mem::PreSync] start");
    for (ChannelHandle channel : channels_) {
        // NotifyRecord(channel, CKE_IDX_0, OUTPUT_XN_ID, output_[rankId_], 1 << OUTPUT_XN_ID); //input和output有啥区别
        NotifyRecord(channel, CKE_IDX_0, INPUT_XN_ID, input_[rankId_], 1 << INPUT_XN_ID); //input和output有啥区别
        NotifyRecord(channel, CKE_IDX_0, TOKEN_XN_ID, token_[rankId_], 1 << TOKEN_XN_ID);
    }
    // uint32_t allBit = 1 << OUTPUT_XN_ID | 1 << TOKEN_XN_ID;
    uint32_t allBit = 1 << INPUT_XN_ID | 1 << TOKEN_XN_ID;

    for (ChannelHandle channel : channels_) {
        NotifyWait(channel, CKE_IDX_0, allBit);
    }
    HCCL_DEBUG("[CcuKernelGatherOmniPipeMesh1DMem2Mem::PreSync] end");
    return;
}

void CcuKernelGatherOmniPipeMesh1DMem2Mem::PostSync()
{
    HCCL_DEBUG("[CcuKernelGatherOmniPipeMesh1DMem2Mem::PostSync] start");
    // HCCL_DEBUG("[%s] channels_ size %u", __func__, channels_.size());
    for (ChannelHandle channel : channels_) {
        NotifyRecord(channel, CKE_IDX_0, 1 << POST_SYNC_ID);
    }
    for (ChannelHandle channel : channels_) {
        NotifyWait(channel, CKE_IDX_0, 1 << POST_SYNC_ID);
    }
}


std::vector<uint64_t> CcuKernelGatherOmniPipeMesh1DMem2Mem::GeneArgs(const hcomm::CcuTaskArg& arg)
{
    HCCL_INFO("[CcuKernelGatherOmniPipeMesh1DMem2Mem] GeneArgs");
    const auto taskArg = dynamic_cast<const CcuTaskArgGatherOmniPipeMesh1DMem2Mem&>(arg);
    std::vector<uint64_t> args;
    
    args.push_back(taskArg.inputAddr_);
    args.push_back(taskArg.outputAddr_);
    args.push_back(taskArg.token_);
    args.push_back(taskArg.localCopyFlag_);
    args.push_back(taskArg.sliceSize_);
    args.push_back(taskArg.inputSliceStride_);
    args.push_back(taskArg.outputSliceStride_);
    args.push_back(taskArg.inputOmniPipeSliceStride_);
    args.push_back(taskArg.outputOmniPipeSliceStride_);
    args.push_back(taskArg.isStepOne_);
    args.push_back(taskArg.isLastStep_);
    args.push_back(taskArg.ifNewRoot_);
     HCCL_DEBUG("[CcuKernelGatherOmniPipeMesh1DMem2Mem::GeneArgs] taskArg.inputAddr_[%ld] taskArg.outputAddr_[%ld] taskArg.token_[%ld] taskArg.localCopyFlag_[%ld] taskArg.sliceSize_[%ld] taskArg.inputOmniPipeSliceStride_[%ld] taskArg.outputOmniPipeSliceStride_[%ld]"
     " taskArg.isStepOne_[%ld]  taskArg.ifNewRoot_[%ld] rankId_[%d] userRank_[%d]", taskArg.inputAddr_, taskArg.outputAddr_, taskArg.token_,
    taskArg.localCopyFlag_, taskArg.sliceSize_, taskArg.inputOmniPipeSliceStride_, taskArg.outputOmniPipeSliceStride_, taskArg.isStepOne_, taskArg.ifNewRoot_, rankId_, userRank_);
    return args;
}

HcclResult CcuKernelGatherOmniPipeMesh1DMem2Mem::Algorithm()
{
    HCCL_DEBUG("[CcuKernelGatherOmniPipeMesh1DMem2Mem::Algorithm] start");
    CHK_RET(InitResource());
    LoadArgs();
    PreSync();

    DoRepeatGather();

    PostSync();
    HCCL_DEBUG("[CcuKernelGatherOmniPipeMesh1DMem2Mem::Algorithm] end");
    return HcclResult::HCCL_SUCCESS;
}


void CcuKernelGatherOmniPipeMesh1DMem2Mem::DoGather()
{
    HCCL_DEBUG("[CcuKernelGatherOmniPipeMesh1DMem2Mem] DoGather, myRank[%u]", rankId_);
    uint32_t channelId = 0;

    for (uint64_t rankIdx = 0; rankIdx < rankSize_; rankIdx++) {
        event_.SetMask(1 << rankIdx);
        CCU_IF(sliceSize_ != 0) {
            if (rankIdx == rankId_) {
                RecordEvent(event_);
            } else {
                ReadNb(channels_[channelId], outputMem_[rankIdx], inputMem_[rankIdx], sliceSize_, event_);
                channelId++;
            }
        }
        CCU_IF(sliceSize_ == 0) {
                RecordEvent(event_);
        }
    }
    event_.SetMask((1 << rankSize_) - 1);
    WaitEvent(event_);
}

void CcuKernelGatherOmniPipeMesh1DMem2Mem::DoRepeatGather()
{
    HCCL_DEBUG("[%s] DoRepeatGather start", __func__);
    for (uint64_t curId = 0; curId < rankSize_; curId++) {
        if (curId == rankId_) {
            continue;
        }
        inputMem_[curId].token = token_[curId];
        inputMem_[curId].addr = input_[curId];
        inputMem_[curId].addr += inputOmniPipeSliceStride_;

        outputMem_[curId].token = token_[curId];
        outputMem_[curId].addr = output_; //root本卡的基址 + loopoffset
        outputMem_[curId].addr += outputOmniPipeSliceStride_; // 卡间偏移 + 片内偏移
    }

    CCU_IF (ifNewRoot_ == true) {
        DoGather();
    }

    CCU_IF (ifNewRoot_ != true) {
            HCCL_INFO(
                "[%s] DoRepeatGather local rank[%u], root rank[%u], do nothing", __func__, rankId_, rootId_);
    }
}

} // namespace ops_hccl