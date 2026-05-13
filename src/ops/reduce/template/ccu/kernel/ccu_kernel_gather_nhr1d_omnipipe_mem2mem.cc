/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "ccu_kernel_gather_nhr1d_omnipipe_mem2mem.h"
#include "ccu_rep.h"

namespace ops_hccl {

CcuKernelGatherNHROmniPipe1DMem2Mem::CcuKernelGatherNHROmniPipe1DMem2Mem(const hcomm::CcuKernelArg& arg)
    : CcuKernelAlgBase(arg)
{
    const CcuKernelArgGatherNHROmniPipe1D& kernelArg = static_cast<const CcuKernelArgGatherNHROmniPipe1D&>(arg);
    rankSize_ = kernelArg.dimSize_;
    rankIdx_ = kernelArg.mySubCommRankId_;
    rootIdx_ = kernelArg.mySubCommRootId_;
    axisId_ = kernelArg.axisId_;
    axisSize_ = kernelArg.axisSize_;
    stepInfoVector_ = kernelArg.stepInfoVector_;
    rank2ChannelIdx_ = kernelArg.rank2ChannelIdx_;
    subCommRanks_ = kernelArg.subCommRanks_;
    dataType_ = kernelArg.opParam_.DataDes.dataType;
    outputDataType_ = kernelArg.opParam_.outputDataType;

    HCCL_DEBUG("[CcuKernelGatherNHROmniPipe1DMem2Mem] rankSize=%llu, rankIdx=%u, rootIdx=%u, axisId=%u, axisSize=%u, stepNum=%zu",
               rankSize_, rankIdx_, rootIdx_, axisId_, axisSize_, stepInfoVector_.size());
}

HcclResult CcuKernelGatherNHROmniPipe1DMem2Mem::InitResource()
{
    HCCL_DEBUG("[CcuKernelGatherNHROmniPipe1DMem2Mem::InitResource] start");

    for (const auto& pair : rank2ChannelIdx_) {
        ChannelHandle handle = GetChannelHandle(pair.second);
        channels_.push_back(handle);
    }

    event_ = CcuRep::CompletedEvent("gather_nhr_omnipipe_event");

    HCCL_DEBUG("[CcuKernelGatherNHROmniPipe1DMem2Mem::InitResource] channels.size=%zu", channels_.size());
    return HcclResult::HCCL_SUCCESS;
}

void CcuKernelGatherNHROmniPipe1DMem2Mem::LoadArgs()
{
    const CcuTaskArgGatherNHROmniPipe1D& taskArg = static_cast<const CcuTaskArgGatherNHROmniPipe1D&>(GetTaskArg());
    input_.SetValue(taskArg.inputAddr_);
    output_.resize(axisSize_);
    for (u32 i = 0; i < axisSize_; i++) {
        output_[i].SetValue(taskArg.outputAddr_);
    }
    token_.resize(axisSize_);
    for (u32 i = 0; i < axisSize_; i++) {
        token_[i].SetValue(taskArg.token_);
    }
    sliceStride_.SetValue(taskArg.sliceStride_);
    localCopyFlag_.SetValue(taskArg.localCopyFlag_);

    HCCL_DEBUG("[CcuKernelGatherNHROmniPipe1DMem2Mem::LoadArgs] input=%llu, output=%llu, token=%llu, sliceStride=%llu, localCopyFlag=%llu",
               taskArg.inputAddr_, taskArg.outputAddr_, taskArg.token_, taskArg.sliceStride_, taskArg.localCopyFlag_);
}

void CcuKernelGatherNHROmniPipe1DMem2Mem::PreSync()
{
    HCCL_DEBUG("[CcuKernelGatherNHROmniPipe1DMem2Mem::PreSync] start");
}

void CcuKernelGatherNHROmniPipe1DMem2Mem::PostSync()
{
    HCCL_DEBUG("[CcuKernelGatherNHROmniPipe1DMem2Mem::PostSync] start");
}

std::vector<uint64_t> CcuKernelGatherNHROmniPipe1DMem2Mem::GeneArgs(const hcomm::CcuTaskArg& arg)
{
    const CcuTaskArgGatherNHROmniPipe1D& taskArg = static_cast<const CcuTaskArgGatherNHROmniPipe1D&>(arg);
    std::vector<uint64_t> args;
    args.push_back(taskArg.inputAddr_);
    args.push_back(taskArg.outputAddr_);
    args.push_back(taskArg.token_);
    args.push_back(taskArg.sliceStride_);
    args.push_back(taskArg.localCopyFlag_);
    return args;
}

HcclResult CcuKernelGatherNHROmniPipe1DMem2Mem::Algorithm()
{
    HCCL_DEBUG("[CcuKernelGatherNHROmniPipe1DMem2Mem::Algorithm] start");
    LoadArgs();
    CHK_RET(InitResource());
    PreSync();

    DoGatherNHR();

    PostSync();
    HCCL_DEBUG("[CcuKernelGatherNHROmniPipe1DMem2Mem::Algorithm] end");
    return HcclResult::HCCL_SUCCESS;
}

void CcuKernelGatherNHROmniPipe1DMem2Mem::DoGatherNHR()
{
    HCCL_DEBUG("[CcuKernelGatherNHROmniPipe1DMem2Mem::DoGatherNHR] stepNum=%zu, isRoot=%d", 
               stepInfoVector_.size(), (rankIdx_ == rootIdx_));

    for (u32 step = 0; step < stepInfoVector_.size(); step++) {
        DoGatherNHRSingleStep(stepInfoVector_[step]);
    }
}

void CcuKernelGatherNHROmniPipe1DMem2Mem::DoGatherNHRSingleStep(const GatherNHRStepInfo& stepInfo)
{
    HCCL_DEBUG("[CcuKernelGatherNHROmniPipe1DMem2Mem::DoGatherNHRSingleStep] step=%u, toRank=%u, fromRank=%u, nSlices=%u",
               stepInfo.step, stepInfo.toRank, stepInfo.fromRank, stepInfo.nSlices);

    u32 toRank = stepInfo.toRank;
    u32 fromRank = stepInfo.fromRank;

    u32 channelIdxTo = rank2ChannelIdx_[toRank];
    u32 channelIdxFrom = rank2ChannelIdx_[fromRank];

    ChannelHandle channelTo = GetChannelHandle(channelIdxTo);
    ChannelHandle channelFrom = GetChannelHandle(channelIdxFrom);

    for (u32 sliceIdx = 0; sliceIdx < stepInfo.nSlices; sliceIdx++) {
        u32 txSliceIdx = stepInfo.txSliceIdxs[sliceIdx];
        u32 rxSliceIdx = stepInfo.rxSliceIdxs[sliceIdx];

        bool isLastSlice = (sliceIdx == stepInfo.nSlices - 1);
        u32 signalIndex = stepInfo.step * stepInfo.nSlices + sliceIdx;

        if (rankIdx_ != rootIdx_) {
            hcomm::CcuRep::LocalAddr src;
            src.addr = input_;
            src.size = sliceStride_;

            hcomm::CcuRep::RemoteAddr dst;
            dst.addr = output_[axisId_];
            dst.channel = channelTo;

            CcuRep::Variable sizeVar = sliceStride_;
            CcuRep::Variable signalVar = CcuRep::Const(signalIndex);

            CcuRep::Send(dst, src, sizeVar, signalVar);

            HCCL_DEBUG("[CcuKernelGatherNHROmniPipe1DMem2Mem] Send: step=%u, sliceIdx=%u, txSliceIdx=%u, toRank=%u",
                       stepInfo.step, sliceIdx, txSliceIdx, toRank);
        }

        if (isLastSlice) {
            CcuRep::Wait(signalIndex);
        }

        HCCL_DEBUG("[CcuKernelGatherNHROmniPipe1DMem2Mem] step=%u, sliceIdx=%u, txSliceIdx=%u, rxSliceIdx=%u",
                   stepInfo.step, sliceIdx, txSliceIdx, rxSliceIdx);
    }
}

} // namespace ops_hccl