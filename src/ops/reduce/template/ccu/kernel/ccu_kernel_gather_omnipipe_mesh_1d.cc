/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "ccu_kernel_gather_omnipipe_mesh_1d.h"

namespace ops_hccl {
using namespace hcomm;

constexpr int INPUT_XN_ID   = 0;
constexpr int OUTPUT_XN_ID  = 1;
constexpr int TOKEN_XN_ID   = 2;
constexpr int POST_SYNC_ID  = 3;
constexpr int CKE_IDX_0     = 0;

CcuKernelGatherOmniPipeMesh1D::CcuKernelGatherOmniPipeMesh1D(const CcuKernelArg &arg)
    : CcuKernelAlgBase(arg)
{
    const CcuKernelArgGatherOmniPipeMesh1D *kernelArg
        = dynamic_cast<const CcuKernelArgGatherOmniPipeMesh1D *>(&arg);
    rankIdx_        = kernelArg->rankId_;
    rootIdx_        = kernelArg->rootId_;
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
    HCCL_INFO("[%s] Init, KernelArgs are rankIdx[%u], rootIdx[%u], userRank[%u], rankSize_[%u], dataType[%d], "
              "outputDataType[%d]",
        __func__, rankIdx_, rootIdx_, userRank_, rankSize_, dataType_, outputDataType_);
}

HcclResult CcuKernelGatherOmniPipeMesh1D::InitResource()
{
    HCCL_DEBUG("[%s] start", __func__);
    uint16_t channelIdx = 0;
    if (channels_.size() == 0) {
        HCCL_ERROR("[%s] channels is empty!", __func__);
        return HcclResult::HCCL_E_INTERNAL;
    }

    HCCL_DEBUG("[%s] channels_.size[%u]", __func__, channels_.size());

    for (uint64_t peerId = 0; peerId < rankSize_; peerId++) {
        if (peerId == rankIdx_) {
            input_ = CreateVariable();
            output_.push_back(CreateVariable());
            token_.push_back(CreateVariable());
        } else {
            HCCL_DEBUG("[%s] MyRank[%u], PeerId[%u], ChannelId[%u]", __func__, rankIdx_, peerId, channelIdx);
            CcuRep::Variable inputVar, outputVar, tokenVar;
            CHK_RET(CreateVariable(channels_[channelIdx], INPUT_XN_ID, &inputVar));
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

    selfBit_ = 1 << rankIdx_;
    allBit_ = ((1 << rankSize_) - 1) & (~(1 << rankIdx_));

    HCCL_DEBUG("[%s] end", __func__);
    return HcclResult::HCCL_SUCCESS;
}

void CcuKernelGatherOmniPipeMesh1D::LoadArgs()
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

void CcuKernelGatherOmniPipeMesh1D::PreSync()
{
    HCCL_DEBUG("[%s] channels_ size %u", __func__, channels_.size());
    for (ChannelHandle channel : channels_) {
        NotifyRecord(channel, CKE_IDX_0, INPUT_XN_ID, input_, 1 << INPUT_XN_ID);
        NotifyRecord(channel, CKE_IDX_0, OUTPUT_XN_ID, output_[rankIdx_], 1 << OUTPUT_XN_ID);
        NotifyRecord(channel, CKE_IDX_0, TOKEN_XN_ID, token_[rankIdx_], 1 << TOKEN_XN_ID);
    }
    uint32_t allBit = 1 << INPUT_XN_ID | 1 << OUTPUT_XN_ID | 1 << TOKEN_XN_ID;

    for (ChannelHandle channel : channels_) {
        NotifyWait(channel, CKE_IDX_0, allBit);
    }

    return;
}

void CcuKernelGatherOmniPipeMesh1D::PostSync()
{
    HCCL_DEBUG("[%s] channels_ size %u", __func__, channels_.size());
    for (ChannelHandle channel : channels_) {
        NotifyRecord(channel, CKE_IDX_0, 1 << POST_SYNC_ID);
    }
    for (ChannelHandle channel : channels_) {
        NotifyWait(channel, CKE_IDX_0, 1 << POST_SYNC_ID);
    }
}

void CcuKernelGatherOmniPipeMesh1D::DoRepeatGather()
{
    HCCL_DEBUG("[%s] start", __func__);

    CCU_IF(localCopyFlag_ == 1) {
        HCCL_DEBUG("[%s] rankIdx[%u] userRank[%u] localcopy begin", __func__, rankIdx_, userRank_);
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
        if (rankIdx_ == rootIdx_) {
            HCCL_DEBUG("[%s] kernel rankId %llu, rankIdx %llu, rankSize %llu, rootIdx %llu, root receive data", 
                __func__, userRank_, rankIdx_, rankSize_, rootIdx_);

            std::vector<CcuRep::RemoteAddr> src;
            std::vector<CcuRep::LocalAddr> dst;
            for (uint64_t rankIdx = 0; rankIdx < rankSize_; rankIdx++) {
                src.push_back(CreateRemoteAddr());
                dst.push_back(CreateLocalAddr());
            }

            uint32_t channelId = 0;
            for (uint64_t rankIdx = 0; rankIdx < rankSize_; rankIdx++) {
                event_.SetMask(1 << rankIdx);
                
                if (rankIdx == rankIdx_) {
                    dst[rankIdx].addr = output_[rankIdx_];
                    CcuRep::Variable outOffset = CreateVariable();
                    outOffset = 0;
                    for (uint32_t i = 0; i < rankIdx; i++) {
                        outOffset += sliceStride_;
                    }
                    dst[rankIdx].addr += outOffset;
                    dst[rankIdx].addr += inputOmniPipeSliceStride_;
                    dst[rankIdx].token = token_[rankIdx_];
                    
                    CcuRep::LocalAddr myInput = CreateLocalAddr();
                    myInput.addr = input_;
                    myInput.token = token_[rankIdx_];
                    LocalCopyNb(dst[rankIdx], myInput, sliceSize_, event_);
                } else {
                    src[rankIdx].addr = output_[rankIdx];
                    src[rankIdx].addr += inputOmniPipeSliceStride_;
                    src[rankIdx].token = token_[rankIdx];
                    
                    dst[rankIdx].addr = output_[rootIdx_];
                    CcuRep::Variable outOffset = CreateVariable();
                    outOffset = 0;
                    for (uint32_t i = 0; i < rankIdx; i++) {
                        outOffset += sliceStride_;
                    }
                    dst[rankIdx].addr += outOffset;
                    dst[rankIdx].addr += inputOmniPipeSliceStride_;
                    dst[rankIdx].token = token_[rootIdx_];
                    
                    ReadNb(channels_[channelId], dst[rankIdx], src[rankIdx], sliceSize_, event_);
                    channelId++;
                }
            }

            event_.SetMask((1 << rankSize_) - 1);
            WaitEvent(event_);
            HCCL_DEBUG("[%s] root gather end", __func__);
        } else {
            HCCL_DEBUG("[%s] kernel rankId %llu, rankIdx %llu, rankSize %llu, rootIdx %llu, non-root send data", 
                __func__, userRank_, rankIdx_, rankSize_, rootIdx_);

            CcuRep::RemoteAddr dst = CreateRemoteAddr();
            CcuRep::LocalAddr src = CreateLocalAddr();

            dst.addr = output_[rootIdx_];
            CcuRep::Variable outOffset = CreateVariable();
            outOffset = 0;
            for (uint32_t i = 0; i < rankIdx_; i++) {
                outOffset += sliceStride_;
            }
            dst.addr += outOffset;
            dst.addr += inputOmniPipeSliceStride_;
            dst.token = token_[rootIdx_];

            src.addr = input_;
            src.addr += inputOmniPipeSliceStride_;
            src.token = token_[rankIdx_];

            event_.SetMask(1 << rankIdx_);
            WriteNb(channels_[0], dst, src, sliceSize_, event_);
            WaitEvent(event_);
            HCCL_DEBUG("[%s] non-root send end", __func__);
        }
    }
    HCCL_DEBUG("[%s] end", __func__);
}

HcclResult CcuKernelGatherOmniPipeMesh1D::Algorithm()
{
    HCCL_INFO("[%s] start", __func__);
    CHK_RET(InitResource());
    LoadArgs();
    PreSync();
    DoRepeatGather();
    PostSync();
    HCCL_INFO("[%s] end", __func__);
    return HcclResult::HCCL_SUCCESS;
}

std::vector<uint64_t> CcuKernelGatherOmniPipeMesh1D::GeneArgs(const CcuTaskArg &arg)
{
    HCCL_DEBUG("[%s] start", __func__);
    const CcuTaskArgGatherOmniPipeMesh1D *taskArg
        = dynamic_cast<const CcuTaskArgGatherOmniPipeMesh1D *>(&arg);
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
        "[%s] rankIdx[%u] userRank[%u] rootIdx[%u] TaskArgs(size[%u]): (0)inputAddr[%llu] (1)outputAddr[%llu] (2)token[%llu] "
        "(3)sliceSize[%llu] (4)sliceStride[%llu] (5)localCopyFlag[%llu] (6)inputOmniPipeSliceStride[%llu] "
        "(7)goSize[0][%llu] (8)goSize[1][%llu] (9)goSize[2][%llu] (10)goSize[3][%llu]",
        __func__, rankIdx_, userRank_, rootIdx_, taskArgs.size(), inputAddr, outputAddr, tokenInfo, sliceSize, sliceStride,
        localCopyFlag, inputOmniPipeSliceStride, goSize[0], goSize[1], goSize[2], goSize[3]);

    HCCL_DEBUG("[%s] end", __func__);
    return taskArgs;
}

}