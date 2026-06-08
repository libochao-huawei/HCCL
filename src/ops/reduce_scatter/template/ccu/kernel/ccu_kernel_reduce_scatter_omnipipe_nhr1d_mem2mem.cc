/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "ccu_kernel_reduce_scatter_omnipipe_nhr1d_mem2mem.h"
#include "ccu_kernel_alg_base.h"

namespace ops_hccl {

constexpr uint16_t INPUT_XN_ID   = 0;
constexpr uint16_t TOKEN_XN_ID   = 1;

constexpr uint16_t CKE_IDX_INPUT = 0;
constexpr uint16_t CKE_IDX_TOKEN = 1;
constexpr uint16_t CKE_IDX_READY = 2;
constexpr uint16_t CKE_IDX_DONE  = 3;
constexpr uint16_t POST_XN_ID    = 4;
constexpr uint16_t BIT_NUM_PER_CKE = 16; // CKE的位数，一个CKE可以处理16种信号


CcuKernelReduceScatterOmniPipeNHR1DMem2Mem::CcuKernelReduceScatterOmniPipeNHR1DMem2Mem(const CcuKernelArg &arg)
    : CcuKernelAlgBase(arg)
{
    const auto kernelArg
        = dynamic_cast<const CcuKernelArgReduceScatterOmniPipeNHR1DMem2Mem *>(&arg);
    rankId_            = kernelArg->rankId_;
    rankSize_          = kernelArg->rankSize_; // 子通信域rank数
    channels_          = kernelArg->channels;
    userRank_          = kernelArg->subCommRanks_[0][rankId_];
    stepInfoVector_    = kernelArg->stepInfoVector_;
    rank2ChannelIdx_   = kernelArg->rank2ChannelIdx_;
    localSize_         = rank2ChannelIdx_.size(); // nhr 算法通信rank数
    myRankIdx_         = rank2ChannelIdx_.size(); // InitResources中将本端放在末尾 此处为对应的idx
    // subRankIdx2RankIdx_ = kernelArg->subRankIdx2RankIdx_;

    dataType_        = kernelArg->opParam_.DataDes.dataType;
    reduceOp_       = kernelArg->opParam_.reduceType;
    HCCL_INFO("[%s] kernelArg: rankId[%u] userRank[%u] myRankIdx[%u] rankSize[%u] localSize[%u] channelSize[%u]",
        __func__, rankId_, userRank_, myRankIdx_, rankSize_, localSize_, channels_.size());
}

HcclResult CcuKernelReduceScatterOmniPipeNHR1DMem2Mem::InitResources()
{
    HCCL_INFO("[%s] start", __func__);
    if (channels_.size() == 0) {
        HCCL_ERROR("[%s] channels is empty!", __func__);
        return HcclResult::HCCL_E_INTERNAL;
    }

    output_       = CreateVariable();
    for (uint32_t channelIdx = 0; channelIdx < localSize_; channelIdx++) {
        HCCL_DEBUG("[%s] myRank[%u] rankId[%u] add remote channelIdx[%u]", __func__, userRank_, rankId_, channelIdx);
        CcuRep::Variable inputVar;
        CcuRep::Variable tokenVar;
        CHK_RET(CreateVariable(channels_[channelIdx], INPUT_XN_ID, &inputVar));
        input_.push_back(inputVar);
        CHK_RET(CreateVariable(channels_[channelIdx], TOKEN_XN_ID, &tokenVar));
        token_.push_back(tokenVar);
    }
    input_.push_back(CreateVariable()); // 将本端数据加在末尾
    token_.push_back(CreateVariable());

    sliceStride_               = CreateVariable();
    localCopyFlag_             = CreateVariable();
    sliceSize_                 = CreateVariable();
    event_                     = CreateCompletedEvent();
    inputOmniPipeSliceStride_  = CreateVariable();
    // groupOpSize_               = CreateGroupOpSize();
    for (uint32_t peerId = 0; peerId < rankSize_; peerId++) {
        inputOmniSliceStrideVec_.push_back(CreateVariable());
    }
    inputSliceStride_          = CreateVariable();
    HCCL_INFO("[%s] success!", __func__);
    return HcclResult::HCCL_SUCCESS;
}

HcclResult CcuKernelReduceScatterOmniPipeNHR1DMem2Mem::LoadArgs()
{
    Load(input_[myRankIdx_]);
    Load(output_);
    Load(token_[myRankIdx_]);
    Load(sliceSize_);
    Load(sliceStride_);
    Load(localCopyFlag_);
    Load(inputOmniPipeSliceStride_);
    // Load(groupOpSize_);
    for (int i = 0; i < rankSize_; i++) {
        Load(inputOmniSliceStrideVec_[i]);
    }
    Load(inputSliceStride_);
    HCCL_INFO("[%s] success!", __func__);
    return HcclResult::HCCL_SUCCESS;
}

static uint32_t GetSignalIndex(const int signalBit)
{
    // 一个CKE有16位，可以处理16个用途
    return static_cast<uint32_t>(signalBit) / BIT_NUM_PER_CKE;
}

static uint16_t GetSignalMask(const int signalBit)
{
    return (1 << (static_cast<uint32_t>(signalBit) % BIT_NUM_PER_CKE));
}

HcclResult CcuKernelReduceScatterOmniPipeNHR1DMem2Mem::PreSync()
{
    HCCL_DEBUG("[%s] channels_ size %u", __func__, channels_.size());
    const uint16_t signalBitInput = GetSignalMask(CKE_IDX_INPUT);
    const uint16_t signalBitToken = GetSignalMask(CKE_IDX_TOKEN);
    const uint32_t signalIndexInput = GetSignalIndex(CKE_IDX_INPUT);
    const uint32_t signalIndexToken = GetSignalIndex(CKE_IDX_TOKEN);

    for (ChannelHandle &channel : channels_) {
        CHK_RET(NotifyRecord(channel, signalIndexInput, CKE_IDX_INPUT, input_[myRankIdx_], signalBitInput));
        CHK_RET(NotifyRecord(channel, signalIndexToken, CKE_IDX_TOKEN, token_[myRankIdx_], signalBitToken));
    }
    // 等所有对端
    const uint16_t waitMask = signalBitInput | signalBitToken; // 组合一下mask
    std::set<uint32_t> signalIdxes{signalIndexInput, signalIndexToken}; // 0
    for (ChannelHandle &channel : channels_) {
        for (uint32_t signalIdx : signalIdxes) {
            CHK_RET(NotifyWait(channel, signalIdx, waitMask));
        }
    }
    HCCL_INFO("[%s] success!", __func__);
    return HcclResult::HCCL_SUCCESS;
}

HcclResult CcuKernelReduceScatterOmniPipeNHR1DMem2Mem::PostSync()
{
    const uint16_t selfBitInput = GetSignalMask(POST_XN_ID);
    const uint32_t signalIndexInput = GetSignalIndex(POST_XN_ID);

    // 通知所有对端
    for (ChannelHandle &channel : channels_) {
        CHK_RET(NotifyRecord(channel, signalIndexInput, selfBitInput));
    }

    // 等待所有需要的对端
    for (ChannelHandle &channel : channels_) {
        CHK_RET(NotifyWait(channel, signalIndexInput, selfBitInput));
    }
    HCCL_INFO("[%s] success!", __func__);
    return HcclResult::HCCL_SUCCESS;
}

HcclResult CcuKernelReduceScatterOmniPipeNHR1DMem2Mem::DoRepeatReduceScatterNHR()
{
    // 本地拷贝 应该用不到了?
    // CCU_IF(localCopyFlag_ == 1) {
    //     HCCL_DEBUG("[%s] rankId[%u] localcopy start", __func__, rankId_);
    //     CcuRep::LocalAddr myOutput = CreateLocalAddr();
    //     CcuRep::LocalAddr myInput  = CreateLocalAddr();

    //     CcuRep::Variable outSliceStride = CreateVariable();
    //     outSliceStride = 0;
    //     for (uint32_t i = 0; i < userRank_; i++) {
    //         outSliceStride += sliceStride_;
    //     }

    //     myInput.addr            =  input_[rankId_];
    //     myOutput.addr           =  output_;
    //     myOutput.addr           += outSliceStride;

    //     event_.SetMask(1 << rankId_);
    //     LocalCopyNb(myOutput, myInput, sliceSize_, event_);

    //     HCCL_DEBUG("[%s] rankId[%u] localcopy end", __func__, rankId_);
    //     WaitEvent(event_);
    // }

    for (auto &nhrStepInfo : stepInfoVector_) {
        CHK_RET(DoRepeatReduceScatterNHRSingleStep(nhrStepInfo));
    }
    HCCL_INFO("[%s] success!", __func__);
    return HcclResult::HCCL_SUCCESS;
}

HcclResult CcuKernelReduceScatterOmniPipeNHR1DMem2Mem::DoRepeatReduceScatterNHRSingleStep(const NHRStepInfoRS &nhrStepInfo)
{
    u32                    &toRankIdx        = rank2ChannelIdx_[nhrStepInfo.toRank];
    u32                    &fromRankIdx      = rank2ChannelIdx_[nhrStepInfo.fromRank];
    ChannelHandle          &sendChannel      = channels_[toRankIdx];
    ChannelHandle          &recvChannel      = channels_[fromRankIdx];
    const std::vector<u32> &recvSliceIdxList = nhrStepInfo.rxSliceIdxs;
    const std::vector<u32> &sendSliceIdxList = nhrStepInfo.txSliceIdxs;
    // TODOv remove test
    // myChannelIdx=myRankIdx_ 已测试 均为3 证明自身rank位于最后一个
    HCCL_DEBUG(
        "[%s] myRank[%u] rankId[%u] step[%u] toRank[%u](channelIdx[%u]) fromRank[%u](channelIdx[%u]) SliceSize[%u]",
        __func__, userRank_, rankId_, nhrStepInfo.step, nhrStepInfo.toRank, toRankIdx, nhrStepInfo.fromRank,
        fromRankIdx, recvSliceIdxList.size());

    CcuRep::LocalAddr dst = CreateLocalAddr();
    CcuRep::RemoteAddr src = CreateRemoteAddr();
    dst.token = token_[myRankIdx_];
    src.token = token_[fromRankIdx];

    const uint32_t signalIdxReady = GetSignalIndex(CKE_IDX_READY); // 0
    const uint32_t signalIdxDone = GetSignalIndex(CKE_IDX_DONE);   // 0
    const uint16_t signalBitReady = GetSignalMask(CKE_IDX_READY);  // 准备好的信号
    const uint16_t signalBitDone = GetSignalMask(CKE_IDX_DONE);    // 写完的信号

    // 通知对端rank自己准备好了-前同步
    // 第一步时可跳过
    if (nhrStepInfo.step != 0) {
        CHK_RET(NotifyRecord(recvChannel, signalIdxReady, signalBitReady)); // 通知fromrank可以写入
        CHK_RET(NotifyWait(sendChannel, signalIdxReady, signalBitReady)); // 等待torank准备好
    }

    for (const auto &recvSliceIdx : recvSliceIdxList) {
        HCCL_DEBUG("[%s] sliceIdx[%u]", __func__, recvSliceIdx);
        // 基址都一样
        src.addr = input_[fromRankIdx];
        dst.addr = input_[myRankIdx_];
        // 相同代表可以使用本rank上的偏移
        if (recvSliceIdx == rankId_) {
            src.addr += sliceStride_;
            src.addr += inputOmniPipeSliceStride_;

            dst.addr += sliceStride_;
            dst.addr += inputOmniPipeSliceStride_;
        } else {
            src.addr += inputOmniSliceStrideVec_[recvSliceIdx];
            dst.addr += inputOmniSliceStrideVec_[recvSliceIdx];
        }
        event_.SetMask(1);
        CCU_IF(sliceSize_ != 0) {
            ReadReduceNb(recvChannel, dst, src, sliceSize_, dataType_, reduceOp_, event_);
        }
        CCU_IF(sliceSize_ == 0) {
            RecordEvent(event_);
        }
        WaitEvent(event_);
    }

    // 写之后告诉对端写完了-后同步
    // 告诉toRank数据写完了
    CHK_RET(NotifyRecord(sendChannel, signalIdxDone, signalBitDone));
    // 等待fromRank写完数据
    CHK_RET(NotifyWait(recvChannel, signalIdxDone, signalBitDone));
    HCCL_INFO("[%s] success!", __func__);
    return HcclResult::HCCL_SUCCESS;
}

HcclResult CcuKernelReduceScatterOmniPipeNHR1DMem2Mem::Algorithm()
{
    HCCL_DEBUG("[%s] start", __func__);

    InitResources();
    LoadArgs();
    PreSync();
    DoRepeatReduceScatterNHR();
    PostSync();

    HCCL_DEBUG("[%s] end", __func__);
    return HcclResult::HCCL_SUCCESS;
}

std::vector<uint64_t> CcuKernelReduceScatterOmniPipeNHR1DMem2Mem::GeneArgs(const CcuTaskArg &arg)
{
    const CcuTaskArgReduceScatterOmniPipeNHR1DMem2Mem *taskArg
        = dynamic_cast<const CcuTaskArgReduceScatterOmniPipeNHR1DMem2Mem *>(&arg);

    uint64_t inputAddr                = taskArg->inputAddr_;
    uint64_t outputAddr               = taskArg->outputAddr_;
    uint64_t tokenInfo                = taskArg->token_;
    uint64_t sliceSize                = taskArg->sliceSize_;
    uint64_t sliceStride              = taskArg->sliceStride_;
    uint64_t localCopyFlag            = taskArg->localCopyFlag_;
    uint64_t inputOmniPipeSliceStride = taskArg->inputOmniPipeSliceStride_;
    std::vector<uint64_t> inputOmniSliceStrideVec = taskArg->inputOmniSliceStrideVec_;
    uint64_t inputSliceStride         = taskArg->inputSliceStride_;
    // auto goSize                       = CalGoSize(sliceSize);

    std::vector<uint64_t> taskArgs = {inputAddr, outputAddr, tokenInfo, sliceSize, sliceStride,
        localCopyFlag, inputOmniPipeSliceStride};
    taskArgs.insert(taskArgs.end(), inputOmniSliceStrideVec.begin(), inputOmniSliceStrideVec.end());
    taskArgs.push_back(inputSliceStride);

    // TODOv 删除含token调试日志
    HCCL_DEBUG(
        "[%s] rankId[%u] userRank[%u] TaskArgs(size[%u]): (0)inputAddr[%llu] (1)outputAddr[%llu] (2)token[%llu] "
        "(3)sliceSize[%llu] (4)sliceStride[%llu] (5)localCopyFlag[%llu] (6)inputOmniPipeSliceStride[%llu]",
        __func__, rankId_, userRank_, taskArgs.size(), inputAddr, outputAddr, tokenInfo, sliceSize, sliceStride,
        localCopyFlag, inputOmniPipeSliceStride);

    HCCL_DEBUG("[%s] end", __func__);
    return taskArgs;
}
} // namespace Hccl
