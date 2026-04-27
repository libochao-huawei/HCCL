/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "ccu_kernel_reduce_scatter_omnipipe_mesh_1D.h"

namespace ops_hccl {
using namespace hcomm;

constexpr int CKE_IDX_0   = 0;
constexpr int INPUT_XN_ID = 1;
constexpr int TOKEN_XN_ID = 2;
constexpr int POST_SYNC_ID = 3;

CcuKernelReduceScatterOmniPipeMesh1D::CcuKernelReduceScatterOmniPipeMesh1D(const CcuKernelArg &arg)
    : CcuKernelAlgBase(arg)
{
    auto kernelArg = dynamic_cast<const CcuKernelArgReduceScatterOmniPipeMesh1D *>(&arg);
    rankId_        = kernelArg->rankId_;
    rankSize_      = kernelArg->dimSize_;
    channels_      = kernelArg->channels;
    dataType_       = kernelArg->opParam_.DataDes.dataType;
    outputDataType_ = kernelArg->opParam_.DataDes.outputType;
    userRank_ = kernelArg->subCommRanks_[0][rankId_];
    if (outputDataType_ == HcclDataType::HCCL_DATA_TYPE_RESERVED) {
        outputDataType_ = dataType_;
        HCCL_WARNING("[%s]outputDataType is [INVALID], set outputDataType to[%d]", __func__, outputDataType_);
    }
    reduceOp_ = kernelArg->opParam_.reduceType;
    HCCL_INFO("[%s]userRank[%u] rankId[%u], rankSize_[%u], dataType[%d], outputDataType[%d], reduceOp[%d]"
        "channels size[%u]", __func__, userRank_, rankId_, rankSize_, dataType_, outputDataType_, reduceOp_,
        channels_.size());
}

HcclResult CcuKernelReduceScatterOmniPipeMesh1D::InitResource()
{
    HCCL_DEBUG("[%s] start", __func__);
    uint16_t channelIdx = 0;
    if (channels_.size() == 0) {
        HCCL_ERROR("[%s]channels is empty!", __func__);
        return HcclResult::HCCL_E_INTERNAL;
    }

    // 按照rank号从小到大遍历channels，遇到本rank就填充本地资源，否则依次取远端资源，要求算法返回的Link同样是按顺序排列的
    for (uint64_t peerId = 0; peerId < rankSize_; peerId++) {
        if (peerId == rankId_) {
            input_.push_back(CreateVariable());
            token_.push_back(CreateVariable());
            HCCL_DEBUG("[%s]channel:local userRank[%u] rankId[%u] PeerId[%llu]",
                __func__, userRank_, rankId_, peerId);
        } else {
            HCCL_DEBUG("[%s]channel:remote userRank[%u] rankId[%u] PeerId[%llu] ChannelId[%u] inputVar[%llu]",
                __func__, userRank_, rankId_, peerId, channelIdx, reinterpret_cast<uint64_t>(channels_[channelIdx]));
            CcuRep::Variable inputVar;
            CcuRep::Variable tokenVar;
            CHK_RET(CreateVariable(channels_[channelIdx], INPUT_XN_ID, &inputVar));
            input_.push_back(inputVar); // 获取channel中id=0的Var来传递output
            CHK_RET(CreateVariable(channels_[channelIdx], TOKEN_XN_ID, &tokenVar));
            token_.push_back(tokenVar);
            channelIdx++;
        }
    }
    output_ = CreateVariable();
    sliceSize_ = CreateVariable();
    inputSliceStride_ = CreateVariable();
    outputSliceStride_ = CreateVariable();
    inputOmniPipeSliceStride_  = CreateVariable();
    localCopyFlag_ = CreateVariable();
    offSet_ = CreateVariable();
    groupOpSize_ = CreateGroupOpSize();
    event_ = CreateCompletedEvent();

    selfBit_ = 1 << rankId_;

    HCCL_DEBUG("[%s] run success", __func__);
    return HcclResult::HCCL_SUCCESS;
}

void CcuKernelReduceScatterOmniPipeMesh1D::LoadArgs()
{
    HCCL_DEBUG("[%s] start", __func__);
    Load(input_[rankId_]);
    Load(output_);
    Load(sliceSize_);
    Load(offSet_);
    Load(token_[rankId_]);
    Load(localCopyFlag_);
    Load(inputSliceStride_);
    Load(outputSliceStride_);
    Load(inputOmniPipeSliceStride_);
    Load(groupOpSize_);

    HCCL_DEBUG("[%s] run success", __func__);
}

void CcuKernelReduceScatterOmniPipeMesh1D::PreSync()
{
    for (ChannelHandle channel : channels_) {
        NotifyRecord(channel, CKE_IDX_0, INPUT_XN_ID, input_[rankId_], 1 << INPUT_XN_ID);
        NotifyRecord(channel, CKE_IDX_0, TOKEN_XN_ID, token_[rankId_], 1 << TOKEN_XN_ID);
    }
    uint16_t allBit = 1 << INPUT_XN_ID | 1 << TOKEN_XN_ID;
    for (ChannelHandle channel : channels_) {
        NotifyWait(channel, CKE_IDX_0, allBit);
    }
}

void CcuKernelReduceScatterOmniPipeMesh1D::PostSync()
{
    for (ChannelHandle channel : channels_) {
        NotifyRecord(channel, CKE_IDX_0, 1 << POST_SYNC_ID);
    }
    for (ChannelHandle channel : channels_) {
        NotifyWait(channel, CKE_IDX_0, 1 << POST_SYNC_ID);
    }
}

void CcuKernelReduceScatterOmniPipeMesh1D::DoRepeatReduceScatter()
{
    HCCL_DEBUG("[%s] userRank[%u] rankId[%u] start", __func__, userRank_, rankId_);
    CCU_IF(localCopyFlag_ == 0) {
        HCCL_DEBUG("[%s] userRank[%u] rankId[%u] do repeat ReduceScatter", __func__, userRank_, rankId_);
        CcuRep::LocalAddr dst = CreateLocalAddr();
        std::vector<CcuRep::RemoteAddr> src;
        for (auto i = 0; i < rankSize_; ++i) {
            src.push_back(CreateRemoteAddr());
        }


        HCCL_DEBUG("[%s] userRank[%u] rankId[%u] one repeat", __func__, userRank_, rankId_);

        dst.addr = input_[rankId_];
        // for (auto i = 0; i < rankId_; ++i) {
        dst.addr += inputSliceStride_;
        // }
        dst.addr += inputOmniPipeSliceStride_;
        dst.token = token_[rankId_];

        src[rankSize_ - 1].addr = dst.addr;
        src[rankSize_ - 1].token = dst.token;
        uint32_t idx = 0;
        for (auto i = 0; i < rankSize_; ++i) {
            if (i == rankId_) {
                continue;
            }
            HCCL_DEBUG("[%s] myRank[%u] mySubCommRank[%u] current i[%d]", __func__, userRank_, rankId_, i);
            src[idx].addr = input_[i];
            // src[idx].addr = output_;
            // for (auto j = 0; j < rankId_; ++j) {
            src[idx].addr += inputSliceStride_;
            // }
            src[idx].addr += inputOmniPipeSliceStride_;
            src[idx].token = token_[i];
            idx++;
        }
        GroupReduce(channels_, dst, src, groupOpSize_, dataType_, outputDataType_, reduceOp_);
    }
    // CCU_IF(localCopyFlag_ == 1) {
        // HCCL_DEBUG("[%s] userRank[%u] rankId[%u] do local copy", __func__, userRank_, rankId_);
        // DoLocalCopy();
    // }
    HCCL_DEBUG("[%s] userRank[%u] rankId[%u] run success", __func__, userRank_, rankId_);
}

void CcuKernelReduceScatterOmniPipeMesh1D::DoLocalCopy()
{
    HCCL_DEBUG("[%s] start", __func__);
    CcuRep::LocalAddr myOutput = CreateLocalAddr();
    CcuRep::LocalAddr myInput  = CreateLocalAddr();

    myOutput.addr           =  output_;
    myInput.addr            =  input_[rankId_];
    myOutput.token          =  token_[rankId_];
    myInput.token           =  token_[rankId_];

    event_.SetMask(1 << rankId_);
    LocalCopyNb(myOutput, myInput, inputSliceStride_, event_);
    WaitEvent(event_);

    HCCL_DEBUG("[%s] rank[%u] local copy end", __func__, rankId_);
}

HcclResult CcuKernelReduceScatterOmniPipeMesh1D::Algorithm()
{
    HCCL_INFO("[%s]start", __func__);
    CHK_RET(InitResource());
    LoadArgs();
    PreSync();
    DoRepeatReduceScatter();
    PostSync();
    HCCL_INFO("[%s]end", __func__);
    return HcclResult::HCCL_SUCCESS;
}

std::vector<uint64_t> CcuKernelReduceScatterOmniPipeMesh1D::GeneArgs(const CcuTaskArg &arg)
{
    const auto taskArg = dynamic_cast<const CcuTaskArgReduceScatterOmniPipeMesh1D *>(&arg);

    uint64_t inputAddr  = taskArg->inputAddr_;
    uint64_t outputAddr = taskArg->outputAddr_;
    uint64_t sliceSize  = taskArg->sliceSize_;
    uint64_t offset     = taskArg->offSet_;
    uint64_t tokenInfo  = taskArg->token_;
    uint64_t localCopyFlag = taskArg->localCopyFlag_;
    uint64_t inputSliceStride = taskArg->inputSliceStride_;
    uint64_t outputSliceStride = taskArg->outputSliceStride_;
    uint64_t inputOmniPipeSliceStride = taskArg->inputOmniPipeSliceStride_;
    auto     goSize     = CalGoSize(sliceSize);

    std::vector<uint64_t> taskArgs = {inputAddr, outputAddr, sliceSize, offset, tokenInfo, localCopyFlag,
        inputSliceStride, outputSliceStride, inputOmniPipeSliceStride, goSize[0], goSize[1], goSize[2], goSize[3]};
    HCCL_INFO(
        "[%s] rankId[%u] userRank[%u] TaskArgs: (0)inputAddr[%llu] (1)outputAddr[%llu] (2)sliceSize[%llu] "
        "(3)offset[%llu] (4)token[%llu] (5)localCopyFlag[%llu] (6)inputSliceStride[%llu] "
        "(7)outputSliceStride[%llu] (8)inputOmniPipeSliceStride[%llu] (9)goSize(0)[%llu] (10)goSize(1)[%llu] "
        "(11)goSize(2)[%llu] (12)goSize(3)[%llu]",
        __func__, rankId_, userRank_, inputAddr, outputAddr, sliceSize, offset, tokenInfo, localCopyFlag,
        inputSliceStride, outputSliceStride, inputOmniPipeSliceStride, goSize[0], goSize[1], goSize[2], goSize[3]);

    HCCL_DEBUG("[%s] rankId[%u] userRank[%u] taskArgs size[%u]", __func__, rankId_, userRank_, taskArgs.size());
    return taskArgs;
}
} // namespace ops_hccl