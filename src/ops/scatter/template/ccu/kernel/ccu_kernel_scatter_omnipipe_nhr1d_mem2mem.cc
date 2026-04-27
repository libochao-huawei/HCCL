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
#include "ccu_kernel_alg_base.h"

namespace ops_hccl {
using namespace hcomm;

constexpr uint16_t OUTPUT_XN_ID   = 0;
constexpr uint16_t TOKEN_XN_ID    = 1;
constexpr uint16_t POST_SYNC_ID   = 2;
constexpr uint16_t CKE_IDX_0      = 0;
constexpr uint16_t FST_AXIS_ID    = 0;
constexpr uint16_t SEC_AXIS_ID    = 1;

CcuKernelScatterOmniPipeNHR1DMem2Mem::CcuKernelScatterOmniPipeNHR1DMem2Mem(const CcuKernelArg& arg) : CcuKernelAlgBase(arg)
{
    const CcuKernelArgScatterOmniPipeNHR1DMem2Mem* kernelArg = 
        dynamic_cast<const CcuKernelArgScatterOmniPipeNHR1DMem2Mem*>(&arg);

    rankId_          = kernelArg->rankId_;
    dimSize_         = kernelArg->dimSize_;
    rootId_          = kernelArg->rootId_;
    axisId_          = kernelArg->axisId_;
    axisSize_        = kernelArg->axisSize_;
    stepInfoVector_  = kernelArg->stepInfoVector_;
    rank2ChannelIdx_ = kernelArg->rank2ChannelIdx_;
    channels_        = kernelArg->channels;
    dataType_        = kernelArg->opParam_.DataDes.dataType;
    subCommRanks_    = kernelArg->subCommRanks_;

    localSize_ = rank2ChannelIdx_.size();
    myRankIdx_ = rank2ChannelIdx_.size();

    HCCL_INFO("[CcuKernelScatterOmniPipeNHR1DMem2Mem] Init, rankId[%u], dimSize_[%u], rootId[%u], axisId[%u], "
        "axisSize[%u], localSize_[%u], stepInfoVector.size()[%lu]",
        rankId_, dimSize_, rootId_, axisId_, axisSize_, localSize_, stepInfoVector_.size());
}

HcclResult CcuKernelScatterOmniPipeNHR1DMem2Mem::InitResource()
{
    die0Size_           = CreateVariable();
    die1Size_           = CreateVariable();
    inputSliceStride_   = CreateVariable();
    outputSliceStride_  = CreateVariable();
    inputRepeatStride_  = CreateVariable();
    outputRepeatStride_ = CreateVariable();
    repeatNumVar_       = CreateVariable();
    repeatNumVarTemp_   = CreateVariable();
    repeatTimeflag_     = CreateVariable();
    cursliceSize_       = CreateVariable();
    isInputOutputEqual_ = CreateVariable();
    die0TailSize_       = CreateVariable();
    die1TailSize_       = CreateVariable();
    inputOmniPipeSliceStride_  = CreateVariable();
    outputOmniPipeSliceStride_ = CreateVariable();

    input_  = CreateVariable();
    output_ = CreateVariable();

    uint16_t channelIdx = 0;
    for (uint32_t i = 0; i < localSize_; i++) {
        if (channelIdx < channels_.size()) {
            CcuRep::Variable outputVar, tokenVar;
            CHK_RET(CreateVariable(channels_[channelIdx], OUTPUT_XN_ID, &outputVar));
            outputRemote_.push_back(outputVar);
            CHK_RET(CreateVariable(channels_[channelIdx], TOKEN_XN_ID, &tokenVar));
            token_.push_back(tokenVar);
            channelIdx++;
        } else {
            HCCL_ERROR("[InitResource] channels size[%lu] < expected[%u]", channels_.size(), i);
            return HcclResult::HCCL_E_INTERNAL;
        }
    }
    token_.push_back(CreateVariable());

    srcMem_       = CreateLocalAddr();
    dstMem_       = CreateLocalAddr();
    dstRemoteMem_ = CreateRemoteAddr();
    event_        = CreateCompletedEvent();

    HCCL_INFO("[InitResource] success");
    return HcclResult::HCCL_SUCCESS;
}

HcclResult CcuKernelScatterOmniPipeNHR1DMem2Mem::LoadArgs()
{
    Load(input_);
    Load(output_);
    Load(token_[myRankIdx_]);
    Load(die0Size_);
    Load(die1Size_);
    Load(inputSliceStride_);
    Load(outputSliceStride_);
    Load(inputRepeatStride_);
    Load(outputRepeatStride_);
    Load(repeatNumVar_);
    Load(isInputOutputEqual_);
    Load(die0TailSize_);
    Load(die1TailSize_);
    Load(inputOmniPipeSliceStride_);
    Load(outputOmniPipeSliceStride_);
    HCCL_INFO("[LoadArgs] success");
    return HcclResult::HCCL_SUCCESS;
}

void CcuKernelScatterOmniPipeNHR1DMem2Mem::PreSync()
{
    HCCL_INFO("[PreSync] start");
    const uint16_t signalBitOutput = 1 << OUTPUT_XN_ID;
    const uint16_t signalBitToken = 1 << TOKEN_XN_ID;
    const uint32_t allBit = signalBitOutput | signalBitToken;

    for (ChannelHandle channel : channels_) {
        NotifyRecord(channel, CKE_IDX_0, OUTPUT_XN_ID, output_, signalBitOutput);
        NotifyRecord(channel, CKE_IDX_0, TOKEN_XN_ID, token_[myRankIdx_], signalBitToken);
    }

    for (ChannelHandle channel : channels_) {
        NotifyWait(channel, CKE_IDX_0, allBit);
    }
    HCCL_INFO("[PreSync] end");
}

void CcuKernelScatterOmniPipeNHR1DMem2Mem::PostSync()
{
    HCCL_INFO("[PostSync] start");
    for (ChannelHandle channel : channels_) {
        NotifyRecord(channel, CKE_IDX_0, 1 << POST_SYNC_ID);
    }
    for (ChannelHandle channel : channels_) {
        NotifyWait(channel, CKE_IDX_0, 1 << POST_SYNC_ID);
    }
    HCCL_INFO("[PostSync] end");
}

void CcuKernelScatterOmniPipeNHR1DMem2Mem::DoScatterOmniPipeNHR()
{
    curInputOffset_ = 0;
    curOutputOffset_ = 0;

    for (u64 i = 0; i < dimSize_; i++) {
        inputOffset_.push_back(CreateVariable());
        inputOffset_[i] = curInputOffset_;
        curInputOffset_ += inputSliceStride_;
    }

    for (u64 i = 0; i < dimSize_; i++) {
        outputOffset_.push_back(CreateVariable());
        outputOffset_[i] = curOutputOffset_;
        curOutputOffset_ += outputSliceStride_;
    }

    for (auto& nhrStepInfo : stepInfoVector_) {
        DoScatterOmniPipeNHRSingleStep(nhrStepInfo);
    }

    if (rankId_ == rootId_) {
        srcMem_.addr = input_;
        srcMem_.addr += inputOffset_[rankId_];
        srcMem_.addr += inputOmniPipeSliceStride_;
    } else {
        srcMem_.addr = output_;
        srcMem_.addr += outputOffset_[rankId_];
        srcMem_.addr += outputOmniPipeSliceStride_;
    }
    dstMem_.addr  = output_;
    dstMem_.addr += outputOffset_[rankId_];
    dstMem_.addr += outputOmniPipeSliceStride_;
    srcMem_.token = token_[myRankIdx_];
    dstMem_.token = token_[myRankIdx_];

    CcuRep::Variable repeatNumAdd = CreateVariable();
    repeatNumAdd = 1;
    repeatTimeflag_ = 0;

    CCU_WHILE(repeatNumVar_ != UINT64_MAX) {
        repeatNumVar_ += repeatNumAdd;

        CCU_IF(repeatTimeflag_ == 1) {
            if (rankId_ == rootId_) {
                srcMem_.addr += inputRepeatStride_;
            }
            dstMem_.addr += outputRepeatStride_;
        }

        CCU_IF(repeatTimeflag_ == 0) {
            if (axisId_ == 1) {
                CcuRep::Variable offsetSize = (rankId_ != dimSize_ - 1) ? die0Size_ : die0TailSize_;
                srcMem_.addr += offsetSize;
                dstMem_.addr += offsetSize;
            }
        }

        cursliceSize_ = (rankId_ != dimSize_ - 1) ? 
            ((axisId_ == 0) ? die0Size_ : die1Size_) :
            ((axisId_ == 0) ? die0TailSize_ : die1TailSize_);

        event_.SetMask(1 << rankId_);
        CCU_IF(isInputOutputEqual_ != 1) {
            DoLocalCopyNb(dstMem_, srcMem_, cursliceSize_, event_);
        }
        CCU_IF(isInputOutputEqual_ == 1) {
            RecordEvent(event_);
        }
        WaitEvent(event_);
        repeatTimeflag_ = 1;
    }
}

void CcuKernelScatterOmniPipeNHR1DMem2Mem::DoScatterOmniPipeNHRSingleStep(const NHRStepInfo& nhrStepInfo)
{
    const std::vector<u32>& sendSliceIdxList = nhrStepInfo.txSliceIdxs;
    const std::vector<u32>& recvSliceIdxList = nhrStepInfo.rxSliceIdxs;

    if (recvSliceIdxList.size() != 0) {
        if (rank2ChannelIdx_.count(nhrStepInfo.fromRank) == 0) {
            HCCL_ERROR("[DoScatterOmniPipeNHRSingleStep] fromRank[%u] not found", nhrStepInfo.fromRank);
            return;
        }
        u32 fromRankIdx = rank2ChannelIdx_[nhrStepInfo.fromRank];
        if (fromRankIdx < channels_.size()) {
            NotifyWait(channels_[fromRankIdx], CKE_IDX_0, 1 << POST_SYNC_ID);
        }
    }

    if (sendSliceIdxList.size() != 0) {
        if (rank2ChannelIdx_.count(nhrStepInfo.toRank) == 0) {
            HCCL_ERROR("[DoScatterOmniPipeNHRSingleStep] toRank[%u] not found", nhrStepInfo.toRank);
            return;
        }
        u32 toRankIdx = rank2ChannelIdx_[nhrStepInfo.toRank];
        if (toRankIdx >= channels_.size()) {
            HCCL_ERROR("[DoScatterOmniPipeNHRSingleStep] toRankIdx[%u] >= channels.size()[%lu]", 
                toRankIdx, channels_.size());
            return;
        }

        dstRemoteMem_.token = token_[toRankIdx];
        srcMem_.token = token_[myRankIdx_];

        for (const u32& sendSliceIdx : sendSliceIdxList) {
            bool isLastSlice = (sendSliceIdx == dimSize_ - 1);

            if (rankId_ == rootId_) {
                srcMem_.addr = input_;
                srcMem_.addr += inputOffset_[sendSliceIdx];
                srcMem_.addr += inputOmniPipeSliceStride_;
            } else {
                srcMem_.addr = output_;
                srcMem_.addr += outputOffset_[sendSliceIdx];
                srcMem_.addr += outputOmniPipeSliceStride_;
            }

            dstRemoteMem_.addr = outputRemote_[toRankIdx];
            dstRemoteMem_.addr += outputOffset_[sendSliceIdx];
            dstRemoteMem_.addr += outputOmniPipeSliceStride_;

            DoSendRecvSlice(nhrStepInfo.toRank, srcMem_, dstRemoteMem_, isLastSlice);
        }

        NotifyRecord(channels_[toRankIdx], CKE_IDX_0, 1 << POST_SYNC_ID);
    }

    HCCL_INFO("[DoScatterOmniPipeNHRSingleStep] step %u, toRank=%u, fromRank=%u, sendSliceNum=%lu",
        nhrStepInfo.step, nhrStepInfo.toRank, nhrStepInfo.fromRank, sendSliceIdxList.size());
}

void CcuKernelScatterOmniPipeNHR1DMem2Mem::DoSendRecvSlice(const u32& toRank, CcuRep::LocalAddr& src, 
    CcuRep::RemoteAddr& dst, bool isLastSlice)
{
    if (rank2ChannelIdx_.count(toRank) == 0) {
        HCCL_ERROR("[DoSendRecvSlice] toRank[%u] not found", toRank);
        return;
    }
    u32 toRankIdx = rank2ChannelIdx_[toRank];
    if (toRankIdx >= channels_.size()) {
        HCCL_ERROR("[DoSendRecvSlice] toRankIdx[%u] >= channels.size()[%lu]", toRankIdx, channels_.size());
        return;
    }
    ChannelHandle sendChannel = channels_[toRankIdx];

    CcuRep::Variable repeatNumAdd2 = CreateVariable();
    repeatNumAdd2 = 1;
    repeatTimeflag_ = 0;
    repeatNumVarTemp_ = repeatNumVar_;

    CCU_WHILE(repeatNumVarTemp_ != UINT64_MAX) {
        repeatNumVarTemp_ += repeatNumAdd2;

        CCU_IF(repeatTimeflag_ == 1) {
            if (rankId_ == rootId_) {
                src.addr += inputRepeatStride_;
            }
            dst.addr += outputRepeatStride_;
        }

        CCU_IF(repeatTimeflag_ == 0) {
            if (axisId_ == 1) {
                CcuRep::Variable offsetSize = isLastSlice ? die0TailSize_ : die0Size_;
                src.addr += offsetSize;
                dst.addr += offsetSize;
            }
        }

        cursliceSize_ = isLastSlice ? 
            ((axisId_ == 0) ? die0TailSize_ : die1TailSize_) :
            ((axisId_ == 0) ? die0Size_ : die1Size_);

        event_.SetMask(1);
        DoWriteNb(sendChannel, dst, src, cursliceSize_, event_);
        WaitEvent(event_);
        repeatTimeflag_ = 1;
    }
}

void CcuKernelScatterOmniPipeNHR1DMem2Mem::DoLocalCopyNb(CcuRep::LocalAddr& dst, CcuRep::LocalAddr& src, 
    CcuRep::Variable& sliceSize, CcuRep::CompletedEvent& event_)
{
    CCU_IF(sliceSize == 0) {
        RecordEvent(event_);
    }
    CCU_IF(sliceSize != 0) {
        LocalCopyNb(dst, src, sliceSize, event_);
    }
}

void CcuKernelScatterOmniPipeNHR1DMem2Mem::DoWriteNb(ChannelHandle& sendChannel, CcuRep::RemoteAddr& dst,
    CcuRep::LocalAddr& src, CcuRep::Variable& sliceSize, CcuRep::CompletedEvent& event_)
{
    CCU_IF(sliceSize == 0) {
        RecordEvent(event_);
    }
    CCU_IF(sliceSize != 0) {
        WriteNb(sendChannel, dst, src, sliceSize, event_);
    }
}

HcclResult CcuKernelScatterOmniPipeNHR1DMem2Mem::Algorithm()
{
    HCCL_INFO("[Algorithm] ScatterOmniPipeNHR1D Mem2Mem run");

    CHK_RET(InitResource());
    CHK_RET(LoadArgs());
    PreSync();
    DoScatterOmniPipeNHR();
    PostSync();

    HCCL_INFO("[Algorithm] ScatterOmniPipeNHR1D Mem2Mem end");
    return HcclResult::HCCL_SUCCESS;
}

std::vector<uint64_t> CcuKernelScatterOmniPipeNHR1DMem2Mem::GeneArgs(const CcuTaskArg& arg)
{
    const CcuTaskArgScatterOmniPipeNHR1DMem2Mem* taskArg = 
        dynamic_cast<const CcuTaskArgScatterOmniPipeNHR1DMem2Mem*>(&arg);

    uint64_t inputAddr          = taskArg->inputAddr_;
    uint64_t outputAddr         = taskArg->outputAddr_;
    uint64_t token              = taskArg->token_;
    uint64_t die0Size           = taskArg->die0Size_;
    uint64_t die1Size           = taskArg->die1Size_;
    uint64_t inputSliceStride   = taskArg->inputSliceStride_;
    uint64_t outputSliceStride  = taskArg->outputSliceStride_;
    uint64_t inputRepeatStride  = taskArg->inputRepeatStride_;
    uint64_t outputRepeatStride = taskArg->outputRepeatStride_;
    uint64_t repeatNumVar       = UINT64_MAX - taskArg->repeatNum_;
    uint64_t isInputOutputEqual  = taskArg->isInputOutputEqual_;
    uint64_t die0TailSize       = taskArg->die0TailSize_;
    uint64_t die1TailSize       = taskArg->die1TailSize_;
    uint64_t inputOmniPipeSliceStride  = taskArg->inputOmniPipeSliceStride_;
    uint64_t outputOmniPipeSliceStride = taskArg->outputOmniPipeSliceStride_;

    HCCL_INFO("[GeneArgs] inputAddr[%llu], outputAddr[%llu], die0Size[%llu], die1Size[%llu], "
        "inputSliceStride[%llu], outputSliceStride[%llu], repeatNum[%llu], isInputOutputEqual[%llu], "
        "inputOmniPipeSliceStride[%llu], outputOmniPipeSliceStride[%llu]",
        inputAddr, outputAddr, die0Size, die1Size, inputSliceStride, outputSliceStride,
        taskArg->repeatNum_, isInputOutputEqual, inputOmniPipeSliceStride, outputOmniPipeSliceStride);

    return {
        inputAddr, outputAddr, token, die0Size, die1Size,
        inputSliceStride, outputSliceStride, inputRepeatStride, outputRepeatStride,
        repeatNumVar, isInputOutputEqual, die0TailSize, die1TailSize,
        inputOmniPipeSliceStride, outputOmniPipeSliceStride
    };
}

}  // namespace ops_hccl
