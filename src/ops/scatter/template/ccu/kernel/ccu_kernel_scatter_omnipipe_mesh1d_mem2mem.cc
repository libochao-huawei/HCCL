/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "ccu_kernel_scatter_omnipipe_mesh1d_mem2mem.h"

namespace ops_hccl {
using namespace hcomm;

constexpr uint16_t OUTPUT_XN_ID = 0;
constexpr uint16_t TOKEN_XN_ID = 1;
constexpr uint16_t POST_SYNC_ID = 2;
constexpr uint16_t CKE_IDX_0 = 0;

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
    subRankIdx2RankIdx_  = kernelArg->subRankIdx2RankIdx_;
    ifRealRoot_    = kernelArg->ifRealRoot_;
    myrealrank_ = kernelArg->myrealrank_;

    for (auto item : subRankIdx2RankIdx_) {
        HCCL_DEBUG("[CcuKernelScatterOmniPipeMesh1DMem2Mem] subRankIdx2RankIdx_: %u -> %u", item.first, item.second);
    }

    HCCL_INFO("[CcuKernelScatterOmniPipeMesh1DMem2Mem] Init, KernelArgs are rankId[%u], "
              "rankSize_[%u], dataType[%d], rootId[%u], ifRealRoot[%d] myRealrank[%u] ",
        rankId_, rankSize_, dataType_, rootId_, ifRealRoot_, myrealrank_);
}

HcclResult CcuKernelScatterOmniPipeMesh1DMem2Mem::InitResource()
{
    uint16_t channelIdx = 0;
    // if (channels_.size() == 0) {
    //     HCCL_ERROR("[CcuKernelScatterOmniPipeMesh1DMem2Mem] channels is empty!");
    //     return HcclResult::HCCL_E_INTERNAL;
    // }

    // 按照rank号从小到大遍历channels，遇到本rank就填充本地资源，否则依次取远端资源，要求给框架返回的Link同样是按顺序排列的
    for (uint64_t peerId = 0; peerId < rankSize_; peerId++) {
        if (peerId == rankId_) {
            output_.push_back(CreateVariable());
            token_.push_back(CreateVariable());
        } else {
            HCCL_DEBUG("[CcuKernelScatterOmniPipeMesh1DMem2Mem] MyRank[%u], PeerId[%u], ChannelId[%u]",
                       rankId_, peerId, channelIdx);
            // 获取channel中id=0的Var来传递output
            CcuRep::Variable outputVar, tokenVar;
            CHK_RET(CreateVariable(channels_[channelIdx], OUTPUT_XN_ID, &outputVar));
            output_.push_back(outputVar);
            CHK_RET(CreateVariable(channels_[channelIdx], TOKEN_XN_ID, &tokenVar));
            token_.push_back(tokenVar);
            channelIdx++;
        }
    }

    input_ = CreateVariable();
    offSet_ = CreateVariable();
    sliceSize_ = CreateVariable();
    isStepOne_ = CreateVariable();
    isLastStep_ = CreateVariable();
    ifNewRoot_ = CreateVariable();
    inputSliceStride_ = CreateVariable();
    outputSliceStride_ = CreateVariable();
    inputOmniPipeSliceStride_ = CreateVariable();
    outputOmniPipeSliceStride_ = CreateVariable();
    localCopyFlag_ = CreateVariable();

    inputMem_.reserve(rankSize_);
    outputMem_.reserve(rankSize_);

    for (uint64_t rankIdx = 0; rankIdx < rankSize_; rankIdx++) {
        inputMem_.push_back(CreateLocalAddr());        
        outputMem_.push_back(CreateRemoteAddr());
    }

    event_ = CreateCompletedEvent();
    return HcclResult::HCCL_SUCCESS;
}

void CcuKernelScatterOmniPipeMesh1DMem2Mem::LoadArgs()
{
    Load(input_);
    Load(output_[rankId_]);
    Load(sliceSize_);
    Load(offSet_);
    Load(token_[rankId_]);
    Load(localCopyFlag_);   
    Load(inputSliceStride_);
    Load(outputSliceStride_);
    Load(inputOmniPipeSliceStride_);
    Load(outputOmniPipeSliceStride_);
    Load(isStepOne_);
    Load(isLastStep_);
    Load(ifNewRoot_);
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

void CcuKernelScatterOmniPipeMesh1DMem2Mem::PreSync()
{
    uint32_t allBit = 1 << OUTPUT_XN_ID | 1 << TOKEN_XN_ID;
    HCCL_DEBUG("channels_.size[%u]", channels_.size());
    for (auto channel : channels_) {
        NotifyRecord(channel, CKE_IDX_0, OUTPUT_XN_ID, output_[rankId_], 1 << OUTPUT_XN_ID);  // 广播output地址
        NotifyRecord(channel, CKE_IDX_0, TOKEN_XN_ID, token_[rankId_], 1 << TOKEN_XN_ID);
    }
    for (auto channel : channels_) {
        NotifyWait(channel, CKE_IDX_0, allBit);
    }
}

void CcuKernelScatterOmniPipeMesh1DMem2Mem::DoRepeatScatter()
{
    HCCL_DEBUG("[CcuKernelScatterOmniPipeMesh1DMem2Mem] DoRepeatScatter, myRank[%u], rootId_[%u] myrealrank_[%u]", rankId_, rootId_, myrealrank_);
    for (uint64_t curId = 0; curId < rankSize_; curId++) {
        if (curId == rankId_) {
            continue;
        }
        inputMem_[curId].token = token_[curId];
        inputMem_[curId].addr = input_;
        inputMem_[curId].addr += inputOmniPipeSliceStride_;

        outputMem_[curId].token = token_[curId];
        outputMem_[curId].addr = output_[curId];
        outputMem_[curId].addr += outputOmniPipeSliceStride_;
    }

    CCU_IF (ifNewRoot_ == true) {
        DoScatter();
    }
    CCU_IF (ifNewRoot_ != true) {
        HCCL_INFO(
            "[CcuKernelScatterOmniPipeMesh1DMem2Mem] DoRepeatScatter local rank[%u], root rank[%u], do nothing", rankId_, rootId_);
    }
}

void CcuKernelScatterOmniPipeMesh1DMem2Mem::DoScatter()
{
    HCCL_DEBUG("[CcuKernelScatterOmniPipeMesh1DMem2Mem] DoScatter, myRank[%u], myrealrank_[%u]", rankId_, myrealrank_);
    uint32_t channelId = 0;

    for (uint64_t rankIdx = 0; rankIdx < rankSize_; rankIdx++) {
        event_.SetMask(1 << rankIdx);
        CCU_IF(sliceSize_ != 0) {
            if (rankIdx == rankId_) {
                RecordEvent(event_);
            } else {
                HCCL_DEBUG("[CcuKernelScatterOmniPipeMesh1DMem2Mem] DoScatter WriteNb, myRank[%u], rankIdx[%u], "
                            "myrealrank_[%u]",
                    rankId_, rankIdx, myrealrank_);
                WriteNb(channels_[channelId], outputMem_[rankIdx], inputMem_[rankIdx], sliceSize_, event_);
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
        taskArg.sliceSize_,
        taskArg.offSet_,
        taskArg.token_,
        taskArg.localCopyFlag_,
        taskArg.inputSliceStride_,
        taskArg.outputSliceStride_,
        taskArg.inputOmniPipeSliceStride_,
        taskArg.outputOmniPipeSliceStride_,
        taskArg.isStepOne_,
        taskArg.isLastStep_,
        taskArg.ifNewRoot_,
    };

    HCCL_INFO("[CcuKernelScatterOmniPipeMesh1DMem2Mem] TaskArgs: inputAddr[%llu], outputAddr[%llu], "
              "sliceSize[%llu],inputSliceStride_[%llu], outputSliceStride_[%llu],"
              "inputOmniPipeSliceStride_[%llu], outputOmniPipeSliceStride_[%llu],"
              "offSet[%llu], localCopyFlag[%llu], isStepOne[%d], isLastStep[%d], ifNewRoot[%d]",
        taskArg.inputAddr_, taskArg.outputAddr_, taskArg.sliceSize_, taskArg.inputSliceStride_,
        taskArg.outputSliceStride_, taskArg.inputOmniPipeSliceStride_, taskArg.outputOmniPipeSliceStride_,
        taskArg.offSet_, taskArg.localCopyFlag_, taskArg.isStepOne_, taskArg.isLastStep_, taskArg.ifNewRoot_);

    return taskArgs;
}

} // namespace ops_hccl
