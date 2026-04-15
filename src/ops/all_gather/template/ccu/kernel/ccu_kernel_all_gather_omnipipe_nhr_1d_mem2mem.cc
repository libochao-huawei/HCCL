/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

// #include "ccu_repeat.h"
// #include "ccu_condition.h"
// #include "ccu_template_utils.h"
#include "ccu_kernel_all_gather_omnipipe_nhr_1d_mem2mem.h"
// #include "ccu_alg_data_trans_wrapper.h"


namespace ops_hccl {
using namespace hcomm;

constexpr int INPUT_XN_ID   = 0;
constexpr int OUTPUT_XN_ID  = 1;
constexpr int TOKEN_XN_ID   = 2;
constexpr int CKE_IDX_0     = 0;
constexpr uint16_t STEP_PRE_SYNC_ID = 3;
constexpr uint16_t STEP_POST_SYNC_ID= 4;

CcuKernelAllGatherOmniPipeNHR1DMem2Mem::CcuKernelAllGatherOmniPipeNHR1DMem2Mem(const CcuKernelArg &arg)
    : CcuKernelAlgBase(arg)
{
    const CcuKernelArgAllGatherOmniPipeNHR1DMem2Mem *kernelArg
        = dynamic_cast<const CcuKernelArgAllGatherOmniPipeNHR1DMem2Mem *>(&arg);

    rankIdx_        = kernelArg->rankId_;
    channels_       = kernelArg->channels;
    stepInfoVector_ = kernelArg->stepInfoVector_;
    rank2ChannelIdx_= kernelArg->rank2ChannelIdx_;
    localSize_      = rank2ChannelIdx_.size();
    reduceOp_       = kernelArg->opParam_.reduceType;
    dataType_       = kernelArg->opParam_.DataDes.dataType;
    outputDataType_ = kernelArg->opParam_.DataDes.outputType;
    subCommRanks_   = kernelArg->subCommRanks_;
    rankId_         = subCommRanks_[0][rankIdx_];
    if (outputDataType_ == HcclDataType::HCCL_DATA_TYPE_RESERVED) {
        outputDataType_ = dataType_;
        HCCL_DEBUG("[CcuKernelAllGatherOmniPipeNHR1DMem2Mem] outputDataType is [INVALID], set outputDataType to[%d]",
                   outputDataType_);
    }
    HCCL_INFO("[CcuKernelAllGatherOmniPipeNHR1DMem2Mem] KernelArg: rankId_[%u], rankIdx_[%u], localSize_[%u], "
              "dataType[%s], outputDataType[%s], reduceOp[%s]",
              rankId_, rankIdx_, localSize_, dataType_, outputDataType_, reduceOp_);
}

HcclResult CcuKernelAllGatherOmniPipeNHR1DMem2Mem::InitResource()
{
    uint16_t channelIdx = 0;
    if (channels_.size() == 0) {
        HCCL_ERROR("[CcuKernelAllGatherOmniPipeNHR1DMem2Mem] channels is empty!");
        return HcclResult::HCCL_E_INTERNAL;
    }

    // 按照rank号从小到大遍历channels，遇到本rank就填充本地资源，否则依次取远端资源，要求给框架返回的Link同样是按顺序排列的
    for (uint64_t peerId = 0; peerId < rankSize_; peerId++) {
        if (peerId == rankIdx_) {
            input_ = CreateVariable();
            output_.push_back(CreateVariable());
            token_.push_back(CreateVariable());
        } else {
            HCCL_DEBUG("[CcuKernelAllGatherOmniPipeNHR1DMem2Mem] MyRank[%u], PeerId[%u], ChannelId[%u]", rankIdx_, peerId, channelIdx);
            CcuRep::Variable outputVar, tokenVar;
            CHK_RET(CreateVariable(channels_[channelIdx], OUTPUT_XN_ID, &outputVar));
            output_.push_back(outputVar);
            CHK_RET(CreateVariable(channels_[channelIdx], TOKEN_XN_ID, &tokenVar));
            token_.push_back(tokenVar);
            channelIdx++;
        }
    }
    repeatNum_                   = CreateVariable();
    sliceStride_                 = CreateVariable();
    localCopyFlag_               = CreateVariable();
    sendCount_                   = CreateVariable();
    sliceSize_                   = CreateVariable();
    event_                       = CreateCompletedEvent();

    selfBit_ = 1 << rankIdx_;                              // 仅rankid位为1，其他位为0，代表本端准备好了
    allBit_ = ((1 << rankSize_) - 1) & (~(1 << rankIdx_)); // 仅rankid位为0，其他位为1，代表远端准备好了

    return HcclResult::HCCL_SUCCESS;
}

void CcuKernelAllGatherOmniPipeNHR1DMem2Mem::LoadArgs()
{
    Load(input_);
    Load(output_[rankIdx_]);
    Load(token_[rankIdx_]);
    Load(sliceSize_);
    Load(repeatNum_);
    Load(sliceStride_);
    Load(localCopyFlag_);

    hcomm::CcuRep::Variable repeatNumTmp_ = CreateVariable();
    repeatNumTmp_ = repeatNum_;

    CcuRep::Variable repeatNumAdd = CreateVariable();
    repeatNumAdd  = 1;
    uint64_t idx = 0;
    CCU_WHILE(repeatNumTmp_ != UINT64_MAX) {
        HCCL_INFO("local input omnipipe");
        inputOmniPipeSliceStride_.emplace_back(CreateVariable());
        Load(inputOmniPipeSliceStride_[idx]);
        repeatNumTmp_ += repeatNumAdd;
        idx++;
    }

    idx = 0;
    repeatNumTmp_ = repeatNum_;
    CCU_WHILE(repeatNumTmp_ != UINT64_MAX) {
        HCCL_INFO("local output omnipipe");
        outputOmniPipeSliceStride_.emplace_back(CreateVariable());
        Load(outputOmniPipeSliceStride_[idx]);
        repeatNumTmp_ += repeatNumAdd;
        idx++;
    }

    return;
}

void CcuKernelAllGatherOmniPipeNHR1DMem2Mem::PreSync()
{
    for (ChannelHandle channel : channels_) {
        NotifyRecord(channel, CKE_IDX_0, OUTPUT_XN_ID, output_[rankIdx_], 1 << OUTPUT_XN_ID);
        NotifyRecord(channel, CKE_IDX_0, TOKEN_XN_ID, token_[rankIdx_], 1 << TOKEN_XN_ID);
    }
    uint32_t allBit = 1 << OUTPUT_XN_ID | 1 << TOKEN_XN_ID;
    for (ChannelHandle channel : channels_) {
        NotifyWait(channel, CKE_IDX_0, allBit);
    }

    return;
}

void CcuKernelAllGatherOmniPipeNHR1DMem2Mem::PostSync()
{
    for (ChannelHandle channel : channels_) {
        NotifyRecord(channel, CKE_IDX_0, OUTPUT_XN_ID, output_[rankIdx_], 1 << OUTPUT_XN_ID);
        NotifyRecord(channel, CKE_IDX_0, TOKEN_XN_ID, token_[rankIdx_], 1 << TOKEN_XN_ID);
    }
    uint32_t allBit = 1 << OUTPUT_XN_ID | 1 << TOKEN_XN_ID;
    for (ChannelHandle channel : channels_) {
        NotifyWait(channel, CKE_IDX_0, allBit);
    }
}


void CcuKernelAllGatherOmniPipeNHR1DMem2Mem::DoRepeatAllGatherNHR()
{
    CcuRep::Variable tmpSliceOffset   = CreateVariable();
    tmpSliceOffset                    = 0;
    std::vector<CcuRep::Variable> outputSliceOffset;

    // 本地拷贝
    CCU_IF(localCopyFlag_ == 1) {
        HCCL_INFO("[DoRepeatAllGather] rankid %u, localcopy begin", rankId_);
        CcuRep::LocalAddr myOutput = CreateLocalAddr();
        CcuRep::LocalAddr myInput  = CreateLocalAddr();

        CcuRep::Variable outSliceStride = CreateVariable();
        outSliceStride = 0;
        for (uint32_t i = 0; i < rankId_; i++) {
            outSliceStride += sliceStride_;
        }

        myInput.addr            =  input_;
        myOutput.addr           =  output_[rankIdx_];
        myOutput.addr           += outSliceStride;

        event_.SetMask(1 << rankIdx_);
        LocalCopyNb(myOutput, myInput, sliceSize_, event_);

        HCCL_INFO("[DoRepeatAllGather] rankid %u, localcopy end", rankId_);
        WaitEvent(event_);

        return ; // 只做local copy
    }

    HCCL_INFO("kernel local copy end");

    for (auto &nhrStepInfo : stepInfoVector_) {
        DoRepeatAllGatherNHRSingleStep(nhrStepInfo, outputSliceOffset);
    }
}

void CcuKernelAllGatherOmniPipeNHR1DMem2Mem::DoRepeatAllGatherNHRSingleStep(
        const NHRStepInfo                   &nhrStepInfo,
        const std::vector<CcuRep::Variable> &outputSliceOffset)
{
    u32& toRankIdx = rank2ChannelIdx_[nhrStepInfo.toRank];
    u32& fromRankIdx = rank2ChannelIdx_[nhrStepInfo.fromRank];
    ChannelHandle sendChannel = channels_[toRankIdx];
    ChannelHandle recvChannel = channels_[fromRankIdx];
    const std::vector<u32> &sendSliceIdxList  = nhrStepInfo.txSliceIdxs;

    CcuRep::LocalAddr src = CreateLocalAddr();
    CcuRep::RemoteAddr dst = CreateRemoteAddr();

    // 被写之前告诉写自己的rank自己准备好了-前同步
    NotifyRecord(recvChannel, CKE_IDX_0, 1 << STEP_PRE_SYNC_ID);
    NotifyWait(sendChannel, CKE_IDX_0, 1 << STEP_PRE_SYNC_ID);

    for (const u32 &sendSliceIdx : sendSliceIdxList) {
        CcuRep::Variable repeatNumAdd = CreateVariable();
        repeatNumAdd  = 1;
        uint32_t idx = 0;
        CCU_WHILE(repeatNum_ != UINT64_MAX) {
            repeatNum_ += repeatNumAdd;
            if (sendSliceIdx == rankIdx_) {
                src.addr = output_[rankIdx_];
                for (uint64_t i = 0; i < rankId_; i++) {
                    src.addr += sliceStride_;
                }
                src.addr += inputOmniPipeSliceStride_[idx];
                src.token = token_[rankIdx_];

                dst.addr = output_[toRankIdx];
                for (uint64_t i = 0; i < rankId_; i++) {
                    dst.addr += sliceStride_;
                }
                dst.addr += outputOmniPipeSliceStride_[sendSliceIdx];
                dst.token = token_[rankIdx_];

            } else {
                src.addr = output_[rankIdx_];
                dst.addr = output_[toRankIdx];
                for (u32 i = 0; i < sendSliceIdx; i++) {
                    src.addr += sliceStride_;
                    dst.addr += sliceStride_;
                }
            }

            event_.SetMask(1);
            WriteNb(sendChannel, dst, src, sliceSize_, event_);
            WaitEvent(event_);

            idx++;
        }
    }

    // 写之后告诉对面写完了-后同步
    NotifyRecord(sendChannel, CKE_IDX_0, 1 << STEP_POST_SYNC_ID);
    NotifyWait(recvChannel, CKE_IDX_0, 1 << STEP_POST_SYNC_ID);
}

HcclResult CcuKernelAllGatherOmniPipeNHR1DMem2Mem::Algorithm()
{
    HCCL_INFO("[CcuKernelAllGatherOmniPipeNHR1DMem2Mem] AllGatherOmniPipeNHR1DMem2Mem run");

    CHK_RET(InitResource());
    LoadArgs();
    PreSync();
    DoRepeatAllGatherNHR();
    PostSync();

    HCCL_INFO("[CcuKernelAllGatherOmniPipeNHR1DMem2Mem] AllGatherOmniPipeNHR1DMem2Mem end");

    return HcclResult::HCCL_SUCCESS;
}

std::vector<uint64_t> CcuKernelAllGatherOmniPipeNHR1DMem2Mem::GeneArgs(const CcuTaskArg &arg)
{
    const CcuTaskArgAllGatherOmniPipeNHR1DMem2Mem *taskArg
        = dynamic_cast<const CcuTaskArgAllGatherOmniPipeNHR1DMem2Mem *>(&arg);
    uint64_t inputAddr                   = taskArg->inputAddr_;
    uint64_t outputAddr                  = taskArg->outputAddr_;
    uint64_t tokenInfo                   = taskArg->token_;

    uint64_t sliceStride                 = taskArg->sliceStride_;
    uint64_t sendCount                   = taskArg->sendCount_;
    uint64_t sliceSize                   = taskArg->sliceSize_;
    uint64_t localCopyFlag               = taskArg->localCopyFlag_;
    uint64_t repeatNum                   = taskArg->repeatNum_;
    uint64_t repeatNumTmp                = UINT64_MAX - repeatNum;

    std::vector<uint64_t> taskArgs = {inputAddr, outputAddr, tokenInfo,
                                      sliceSize, repeatNumTmp, sliceStride, localCopyFlag};

    for (int i = 0; i < repeatNum; i++) {
        taskArgs.emplace_back(taskArg->inputOmniPipeSliceStride_[i]);
    }

    for (int i = 0; i < repeatNum; i++) {
        taskArgs.emplace_back(taskArg->outputOmniPipeSliceStride_[i]);
    }

    HCCL_INFO("[CcuKernelAllGatherOmniPipeNHR1DMem2Mem] TaskArgs: inputAddr[%llu], outputAddr[%llu], "
               "sliceSize[%llu], repeatNum[%llu], sliceStride[%llu], localCopyFlag[%llu]",
               inputAddr, outputAddr, sliceSize, repeatNum, sliceStride, localCopyFlag);
    return taskArgs;
}

} // namespace ops_hccl