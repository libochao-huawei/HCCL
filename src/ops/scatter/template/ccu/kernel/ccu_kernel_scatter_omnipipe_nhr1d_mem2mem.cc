/**
* Copyright (c) 2025 Huawei Technologies Co., Ltd.
* This program is free software, you can redistribute it and/or modify it under the terms and conditions of
* CANN Open Software License Agreement Version 2.0 (the "License").
* Please refer to the License for details. You may not use this file except in compliance with the License.
* THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
* INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
* See LICENSE in the root of the software repository for the full text of the License.
*/
#include "ccu_kernel_scatter_omnipipe_nhr1d_mem2mem.h"

namespace ops_hccl {
using namespace hcomm;

constexpr uint16_t OUTPUT_XN_ID = 0;
constexpr uint16_t TOKEN_XN_ID = 1;
constexpr uint16_t POST_SYNC_ID = 2;
constexpr uint16_t STEP_SYNC_ID = 3;
constexpr uint16_t CKE_IDX_0 = 0;

CcuKernelScatterOmniPipeNHR1DMem2Mem::CcuKernelScatterOmniPipeNHR1DMem2Mem(const CcuKernelArg &arg)
    : CcuKernelAlgBase(arg)
{
    const CcuKernelArgScatterOmniPipeNHR1DMem2Mem *kernelArg = 
        dynamic_cast<const CcuKernelArgScatterOmniPipeNHR1DMem2Mem *>(&arg);
    rankId_        = kernelArg->rankId_;
    rankSize_      = kernelArg->dimSize_;
    channels_      = kernelArg->channels;
    dataType_      = kernelArg->opParam_.DataDes.dataType;
    rootId_        = kernelArg->rootId_;
    subCommRanks_  = kernelArg->subCommRanks_;
    ifRealRoot_    = kernelArg->ifRealRoot_;
    myrealrank_    = kernelArg->myrealrank_;

    stepInfoVector_    = kernelArg->stepInfoVector_;
    rank2ChannelIdx_   = kernelArg->rank2ChannelIdx_;
    localSize_         = rank2ChannelIdx_.size(); // nhr 算法通信rank数
    myRankIdx_         = rank2ChannelIdx_.size(); // InitResources中将本端放在末尾 此处为对应的idx

    // TODO stepInfoVector_传递  》》》 

    HCCL_INFO("[CcuKernelScatterOmniPipeNHR1DMem2Mem] Init, KernelArgs are rankId[%u], "
            "rankSize_[%u], dataType[%d], rootId[%u], ifRealRoot[%d] myRealrank[%u] ",
        rankId_, rankSize_, dataType_, rootId_, ifRealRoot_, myrealrank_);
}

HcclResult CcuKernelScatterOmniPipeNHR1DMem2Mem::InitResource()
{
    HCCL_INFO("InitResource start");

    uint16_t channelIdx = 0;
    if (channels_.size() == 0) {
        HCCL_ERROR("[CcuKernelScatterOmniPipeNHR1DMem2Mem] channels is empty!");
        return HcclResult::HCCL_E_INTERNAL;
    }

    // 为每个需要通信的rank创建scratch和token变量
    for (uint32_t i = 0; i < localSize_; i++) {
        HCCL_INFO("InitResource start tag0");
        if (channels_.size() > channelIdx) {
            HCCL_DEBUG(
                "[CcuKernelScatterNHR1DMem2Mem] MyRank[%u], TransportId[%u], ChannelId[%u]", rankId_, i, channelIdx);
            CcuRep::Variable outputVar, tokenVar;
            CHK_RET(CreateVariable(channels_[channelIdx], OUTPUT_XN_ID, &outputVar));
            output_.push_back(outputVar);
            CHK_RET(CreateVariable(channels_[channelIdx], TOKEN_XN_ID, &tokenVar));
            token_.push_back(tokenVar);
            channelIdx++;
        } else {
            HCCL_ERROR(
                "[CcuKernelScatterNHR1DMem2Mem] channels size[%lu] is less than expected[%u]", channels_.size(), i);
            return HcclResult::HCCL_E_INTERNAL;
        }
        HCCL_INFO("InitResource start tag1");
    }
    // 本端的output和token放在最后
    output_.push_back(CreateVariable());
    token_.push_back(CreateVariable());

    input_ = CreateVariable();
    offSet_ = CreateVariable();
    sliceSize_ = CreateVariable();
    inputSliceStride_ = CreateVariable();
    outputSliceStride_ = CreateVariable();
    inputOmniPipeSliceStride_ = CreateVariable();
    outputOmniPipeSliceStride_ = CreateVariable();
    localCopyFlag_ = CreateVariable();
    isStepOne_ = CreateVariable();
    isLastStep_ = CreateVariable();
    ifNewRoot_ = CreateVariable();
    for (uint32_t peerId = 0; peerId < rankSize_; peerId++) {
        inputOmniSliceStrideVec_.push_back(CreateVariable());
    }

    event_ = CreateCompletedEvent();
    HCCL_INFO("InitResource end");
    return HcclResult::HCCL_SUCCESS;
}

void CcuKernelScatterOmniPipeNHR1DMem2Mem::LoadArgs()
{
    Load(input_);
    Load(output_[myRankIdx_]);// NHR
    Load(sliceSize_);
    Load(offSet_);
    Load(token_[myRankIdx_]);// NHR
    Load(localCopyFlag_);   
    Load(inputSliceStride_);
    Load(outputSliceStride_);
    Load(inputOmniPipeSliceStride_);
    Load(outputOmniPipeSliceStride_);

    Load(isStepOne_);
    Load(isLastStep_);
    Load(ifNewRoot_);
    for (int i = 0; i < rankSize_; i++) {
        Load(inputOmniSliceStrideVec_[i]);
    }
}

void CcuKernelScatterOmniPipeNHR1DMem2Mem::PreSync()
{
    HCCL_DEBUG("channels_.size[%u]", channels_.size());
    uint16_t allBit = 1 << OUTPUT_XN_ID | 1 << TOKEN_XN_ID;
    for (ChannelHandle channel : channels_) {
        NotifyRecord(channel, CKE_IDX_0, OUTPUT_XN_ID, output_[myRankIdx_], 1 << OUTPUT_XN_ID); // 同步并置位远端
        NotifyRecord(channel, CKE_IDX_0, TOKEN_XN_ID, token_[myRankIdx_], 1 << TOKEN_XN_ID);
    }
    for (ChannelHandle channel : channels_) {
        NotifyWait(channel, CKE_IDX_0, allBit);
    }
}

void CcuKernelScatterOmniPipeNHR1DMem2Mem::PostSync()
{
    for (auto channel : channels_) {
        NotifyRecord(channel, CKE_IDX_0, 1 << POST_SYNC_ID);
    }
    for (auto channel : channels_) {
        NotifyWait(channel, CKE_IDX_0, 1 << POST_SYNC_ID);
    }
}

HcclResult CcuKernelScatterOmniPipeNHR1DMem2Mem::DoScatterOmniPipeNHR()
{
    for (auto& nhrStepInfo : stepInfoVector_) {
        DoScatterOmniPipeNHRSingleStep(nhrStepInfo);
    }
    return HcclResult::HCCL_SUCCESS;
}

HcclResult CcuKernelScatterOmniPipeNHR1DMem2Mem::DoScatterOmniPipeNHRSingleStep(const NHRStepInfo& nhrStepInfo)
{

    CcuRep::LocalAddr src = CreateLocalAddr();
    CcuRep::RemoteAddr dst = CreateRemoteAddr();
    u32                    &toRankIdx        = rank2ChannelIdx_[nhrStepInfo.toRank];
    u32                    &fromRankIdx      = rank2ChannelIdx_[nhrStepInfo.fromRank];
    ChannelHandle          &sendChannel      = channels_[toRankIdx];
    ChannelHandle          &recvChannel      = channels_[fromRankIdx];
    const std::vector<u32> &sendSliceIdxList = nhrStepInfo.txSliceIdxs;
    const std::vector<u32> &recvSliceIdxList = nhrStepInfo.rxSliceIdxs;

    if (sendSliceIdxList.size() != 0) {
        u32& toRankIdx = rank2ChannelIdx_[nhrStepInfo.toRank]; // nhrStepInfo.toRank是实际的Rank，toRankIdx转换为子通信域的
        u32  sendSliceIdx = 0;
        ChannelHandle sendChannel        = channels_[toRankIdx];
        src.token                        = token_[myRankIdx_];
        dst.token                        = token_[toRankIdx];
        for (u32 i = 0; i < sendSliceIdxList.size(); i++) {
            sendSliceIdx = sendSliceIdxList[i]; //1
            HCCL_INFO("[CcuKernelBroadcastNhr1DMem2Mem][DoScatterNHRSingleStep] sendSliceIdx[%u]", sendSliceIdx);
            
            // 设置源地址：如果是root，从input中取；否则从scratch中取
            if(ifRealRoot_) {
                src.addr = input_;
                src.addr += inputOmniPipeSliceStride_;
            } else {
                src.addr = output_[myRankIdx_];
                src.addr += inputOmniPipeSliceStride_;
            }

            dst.addr = output_[toRankIdx];

            // dst.addr += inputOmniSliceStrideVec_[myrealrank_];  rankId_  myrealrank_ % rankSize_
            // todo :找到对应的映射关系，对应到具体的下标
            // dst.addr += inputOmniSliceStrideVec_[myrealrank_ % rankSize_]; // inputOmniSliceStrideVec_[sendSliceIdx 发给1]; // 收发的偏移地址,输入偏移和输出偏移是一致的
            dst.addr += inputOmniSliceStrideVec_[sendSliceIdx];

            // 分析偏移地址： 0卡 取下标为0的； 1卡取下标为1的
            // 问题：子rank 0 1 -- sendSliceIdx 都是1和预期为0不符合 【泛化拓扑如何处理？？ 】
            // myrealrank_%rankSize_

            event_.SetMask(1);
            CCU_IF(sliceSize_ != 0) {
                WriteNb(sendChannel, dst, src, sliceSize_, event_);
            }
            CCU_IF(sliceSize_ == 0)
            {
                RecordEvent(event_);
            }
            WaitEvent(event_);
        }
        NotifyRecord(channels_[toRankIdx], CKE_IDX_0, 1 << STEP_SYNC_ID);
    }

    if (recvSliceIdxList.size() != 0) {
        u32& fromRankIdx = rank2ChannelIdx_[nhrStepInfo.fromRank];
        ChannelHandle recvChannel = channels_[fromRankIdx];
        NotifyWait(recvChannel, CKE_IDX_0, 1 << STEP_SYNC_ID);
    }

    HCCL_INFO("[DoScatterOmniPipeNHRSingleStep] step %u, toRank=%u, fromRank=%u, sendSliceNum=%lu",
        nhrStepInfo.step, nhrStepInfo.toRank, nhrStepInfo.fromRank, sendSliceIdxList.size());
    return HcclResult::HCCL_SUCCESS;
}

HcclResult CcuKernelScatterOmniPipeNHR1DMem2Mem::Algorithm()
{
    HCCL_INFO("[CcuKernelScatterOmniPipeNHR1DMem2Mem] ScatterOmniPipeNHR1D Mem2Mem run");

    CHK_RET(InitResource());
    LoadArgs();
    PreSync();
    DoScatterOmniPipeNHR();
    PostSync();
    HCCL_INFO("[CcuKernelScatterOmniPipeNHR1DMem2Mem] ScatterOmniPipeNHR1D Mem2Mem end");
    return HcclResult::HCCL_SUCCESS;
}

std::vector<uint64_t> CcuKernelScatterOmniPipeNHR1DMem2Mem::GeneArgs(const CcuTaskArg &arg)
{
    const CcuTaskArgScatterOmniPipeNHR1DMem2Mem *taskArg
        = dynamic_cast<const CcuTaskArgScatterOmniPipeNHR1DMem2Mem *>(&arg);
    uint64_t inputAddr                 = taskArg->inputAddr_;
    uint64_t outputAddr                = taskArg->outputAddr_;
    uint64_t sliceSize                 = taskArg->sliceSize_;
    uint64_t offSet                    = taskArg->offSet_;
    uint64_t token                     = taskArg->token_;
    uint64_t localCopyFlag             = taskArg->localCopyFlag_;
    uint64_t inputSliceStride          = taskArg->inputSliceStride_;
    uint64_t outputSliceStride         = taskArg->outputSliceStride_;
    uint64_t inputOmniPipeSliceStride  = taskArg->inputOmniPipeSliceStride_;
    uint64_t outputOmniPipeSliceStride = taskArg->outputOmniPipeSliceStride_;
    uint64_t isStepOne                 = taskArg->isStepOne_;
    uint64_t isLastStep                = taskArg->isLastStep_;
    uint64_t ifNewRoot                 = taskArg->ifNewRoot_;
    std::vector<uint64_t> inputOmniSliceStrideVec = taskArg->inputOmniSliceStrideVec_;

    std::vector<uint64_t> taskArgs = {inputAddr, outputAddr, sliceSize, offSet, token, localCopyFlag, inputSliceStride,
        outputSliceStride, inputOmniPipeSliceStride, outputOmniPipeSliceStride, isStepOne, isLastStep, ifNewRoot};
    taskArgs.insert(taskArgs.end(), inputOmniSliceStrideVec.begin(), inputOmniSliceStrideVec.end());

    HCCL_INFO("[CcuKernelScatterOmniPipeNHR1DMem2Mem] TaskArgs: inputAddr[%llu], outputAddr[%llu], "
            "sliceSize[%llu],inputSliceStride_[%llu], outputSliceStride_[%llu],"
            "inputOmniPipeSliceStride_[%llu], outputOmniPipeSliceStride_[%llu],"
            "offSet[%llu], localCopyFlag[%llu], isStepOne[%d], isLastStep[%d], ifNewRoot[%d]",
        inputAddr, outputAddr, sliceSize, inputSliceStride,
        outputSliceStride, inputOmniPipeSliceStride, outputOmniPipeSliceStride,
        offSet, localCopyFlag, isStepOne, isLastStep, ifNewRoot);
    
    HCCL_DEBUG("[%s] end", __func__);

    return taskArgs;
}

} // namespace ops_hccl