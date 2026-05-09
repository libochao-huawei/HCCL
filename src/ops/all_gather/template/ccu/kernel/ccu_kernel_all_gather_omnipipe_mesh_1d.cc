/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "ccu_kernel_all_gather_omnipipe_mesh_1d.h"

namespace ops_hccl {
using namespace hcomm;

constexpr int INPUT_XN_ID   = 0;
constexpr int OUTPUT_XN_ID  = 1;
constexpr int TOKEN_XN_ID   = 2;
constexpr int POST_SYNC_ID  = 3;
constexpr int CKE_IDX_0     = 0;

CcuKernelAllGatherOmniPipeMesh1D::CcuKernelAllGatherOmniPipeMesh1D(const CcuKernelArg &arg)
    : CcuKernelAlgBase(arg)
{
    const CcuKernelArgAllGatherOmniPipeMesh1D *kernelArg
        = dynamic_cast<const CcuKernelArgAllGatherOmniPipeMesh1D *>(&arg);
    rankIdx_        = kernelArg->rankId_;
    rankSize_       = kernelArg->dimSize_;
    channels_       = kernelArg->channels;
    dataType_       = kernelArg->opParam_.DataDes.dataType;
    outputDataType_ = kernelArg->opParam_.DataDes.outputType;
    subCommRanks_   = kernelArg->subCommRanks_;
    userRank_       = subCommRanks_[0][rankIdx_];

    if (outputDataType_ == HcclDataType::HCCL_DATA_TYPE_RESERVED) {
        outputDataType_ = dataType_;
        HCCL_DEBUG("[%s] outputDataType is [INVALID], set outputDataType to[%d]", __func__, outputDataType_);
    }
    HCCL_INFO("[%s] Init, KernelArgs are rankIdx[%u], userRank[%u], rankSize_[%u], dataType[%d], "
              "outputDataType[%d]",
        __func__, rankIdx_, userRank_, rankSize_, dataType_, outputDataType_);
}

HcclResult CcuKernelAllGatherOmniPipeMesh1D::InitResource()
{
    HCCL_DEBUG("[%s] start", __func__);
    uint16_t channelIdx = 0;
    if (channels_.size() == 0) {
        HCCL_ERROR("[%s] channels is empty!", __func__);
        return HcclResult::HCCL_E_INTERNAL;
    }

    HCCL_DEBUG("[%s] channels_.size[%u]", __func__, channels_.size());

    // 按照rank号从小到大遍历channels，遇到本rank就填充本地资源，否则依次取远端资源，要求给框架返回的Link同样是按顺序排列的
    for (uint64_t peerId = 0; peerId < rankSize_; peerId++) {
        if (peerId == rankIdx_) {
            input_ = CreateVariable();
            output_.push_back(CreateVariable());
            token_.push_back(CreateVariable());
        } else {
            HCCL_DEBUG("[%s] MyRank[%u], PeerId[%u], ChannelId[%u]", __func__, rankIdx_, peerId, channelIdx);
            CcuRep::Variable outputVar, tokenVar;
            CHK_RET(CreateVariable(channels_[channelIdx], OUTPUT_XN_ID, &outputVar));
            output_.push_back(outputVar);
            CHK_RET(CreateVariable(channels_[channelIdx], TOKEN_XN_ID, &tokenVar));
            token_.push_back(tokenVar);
            channelIdx++;
        }
    }
    sliceStride_               = CreateVariable();
    localCopyFlag_             = CreateVariable();
    sliceSize_                 = CreateVariable();
    event_                     = CreateCompletedEvent();
    inputOmniPipeSliceStride_  = CreateVariable();
    groupOpSize_               = CreateGroupOpSize();

    selfBit_ = 1 << rankIdx_;                              // 仅rankid位为1，其他位为0，代表本端准备好了
    allBit_ = ((1 << rankSize_) - 1) & (~(1 << rankIdx_)); // 仅rankid位为0，其他位为1，代表远端准备好了

    HCCL_DEBUG("[%s] end", __func__);
    return HcclResult::HCCL_SUCCESS;
}

void CcuKernelAllGatherOmniPipeMesh1D::LoadArgs()
{
    HCCL_DEBUG("[%s] start", __func__);

    Load(input_);
    Load(output_[rankIdx_]);
    Load(token_[rankIdx_]);
    Load(sliceSize_);
    Load(sliceStride_);
    Load(localCopyFlag_);
    Load(inputOmniPipeSliceStride_);
    Load(groupOpSize_);

    HCCL_DEBUG("[%s] end", __func__);
}

void CcuKernelAllGatherOmniPipeMesh1D::PreSync()
{
    HCCL_DEBUG("[%s] channels_ size %u", __func__, channels_.size());
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

void CcuKernelAllGatherOmniPipeMesh1D::PostSync()
{
    HCCL_DEBUG("[%s] channels_ size %u", __func__, channels_.size());
    for (ChannelHandle channel : channels_) {
        NotifyRecord(channel, CKE_IDX_0, 1 << POST_SYNC_ID);
    }
    for (ChannelHandle channel : channels_) {
        NotifyWait(channel, CKE_IDX_0, 1 << POST_SYNC_ID);
    }
}

void CcuKernelAllGatherOmniPipeMesh1D::DoRepeatAllGather()
{
    HCCL_DEBUG("[%s] start", __func__);

    // 本地拷贝
    CCU_IF(localCopyFlag_ == 1) {
        HCCL_DEBUG("[%s] rankIdx[%u] userRank[%u[] localcopy begin", __func__, rankIdx_, userRank_);
        CcuRep::LocalAddr myOutput = CreateLocalAddr();
        CcuRep::LocalAddr myInput  = CreateLocalAddr();

        CcuRep::Variable outSliceStride = CreateVariable();
        outSliceStride = 0;
        for (uint32_t i = 0; i < userRank_; i++) {
            outSliceStride += sliceStride_;
        }

        myInput.addr            =  input_;
        myOutput.addr           =  output_[rankIdx_];
        myOutput.addr           += outSliceStride;

#if T_DESC("使用低性能本地拷贝", true)
        event_.SetMask(1 << rankIdx_);
        LocalCopyNb(myOutput, myInput, sliceSize_, event_);
        WaitEvent(event_);
#else
        GroupCopy(myOutput, myInput, groupOpSize_);
#endif
        HCCL_DEBUG("[%s] rankIdx[%u] userRank[%u] localcopy end", __func__, rankIdx_, userRank_);
    }

    CCU_IF(localCopyFlag_ == 0) {
        CcuRep::LocalAddr src = CreateLocalAddr();
        std::vector<CcuRep::RemoteAddr> dst;
        for (uint64_t rankIdx = 0; rankIdx < rankSize_; rankIdx++) {
            dst.push_back(CreateRemoteAddr());
        }

        HCCL_DEBUG("[%s] kernel rankId %llu, rankIdx %llu, rankSize %llu", __func__, userRank_, rankIdx_, rankSize_);

        src.addr = output_[rankIdx_];
        src.addr += sliceStride_;
        src.addr += inputOmniPipeSliceStride_;
        src.token = token_[rankIdx_];

        dst[rankSize_ - 1].addr = src.addr;
        dst[rankSize_ - 1].token = src.token;
        uint32_t curId = 0;
        for (uint64_t rankIdx = 0; rankIdx < rankSize_; rankIdx++) {
            if (rankIdx == rankIdx_) {
                continue;
            }

            dst[curId].addr = output_[rankIdx];
            dst[curId].addr += sliceStride_;
            dst[curId].addr += inputOmniPipeSliceStride_;
            dst[curId].token = token_[rankIdx];
            curId++;
        }

        HCCL_DEBUG("[%s] GroupBroadcast start, dst size[%u]", __func__, dst.size());
        GroupBroadcast(channels_, dst, src, groupOpSize_);
        HCCL_DEBUG("[%s] GroupBroadcast end", __func__);
    }
    HCCL_DEBUG("[%s] end", __func__);
}

HcclResult CcuKernelAllGatherOmniPipeMesh1D::Algorithm()
{
    HCCL_INFO("[%s] start", __func__);
    CHK_RET(InitResource());
    LoadArgs();
    PreSync();
    DoRepeatAllGather();
    PostSync();
    HCCL_INFO("[%s] end", __func__);
    return HcclResult::HCCL_SUCCESS;
}

std::vector<uint64_t> CcuKernelAllGatherOmniPipeMesh1D::GeneArgs(const CcuTaskArg &arg)
{
    HCCL_DEBUG("[%s] start", __func__);
    const CcuTaskArgAllGatherOmniPipeMesh1D *taskArg
        = dynamic_cast<const CcuTaskArgAllGatherOmniPipeMesh1D *>(&arg);
    uint64_t inputAddr                = taskArg->inputAddr_;
    uint64_t outputAddr               = taskArg->outputAddr_;
    uint64_t tokenInfo                = taskArg->token_;
    uint64_t sliceSize                = taskArg->sliceSize_;
    uint64_t sliceStride              = taskArg->sliceStride_;
    uint64_t localCopyFlag            = taskArg->localCopyFlag_;
    uint64_t inputOmniPipeSliceStride = taskArg->inputOmniPipeSliceStride_;
    auto goSize                       = CalGoSize(sliceSize);

    std::vector<uint64_t> taskArgs = {inputAddr, outputAddr, tokenInfo, sliceSize, sliceStride,
        localCopyFlag, inputOmniPipeSliceStride, goSize[0], goSize[1], goSize[2], goSize[3]};

    HCCL_DEBUG(
        "[%s] rankIdx[%u] userRank[%u] TaskArgs(size[%u]): (0)inputAddr[%llu] (1)outputAddr[%llu] (2)token[%llu] "
        "(3)sliceSize[%llu] (4)sliceStride[%llu] (5)localCopyFlag[%llu] (6)inputOmniPipeSliceStride[%llu] "
        "(7)goSize[0][%llu] (8)goSize[1][%llu] (9)goSize[2][%llu] (10)goSize[3][%llu]",
        __func__, rankIdx_, userRank_, taskArgs.size(), inputAddr, outputAddr, tokenInfo, sliceSize, sliceStride,
        localCopyFlag, inputOmniPipeSliceStride, goSize[0], goSize[1], goSize[2], goSize[3]);

    HCCL_DEBUG("[%s] end", __func__);
    return taskArgs;
}

} // namespace ops_hccl