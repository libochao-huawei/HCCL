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
    rankId_         = subCommRanks_[0][rankIdx_];

    if (outputDataType_ == HcclDataType::HCCL_DATA_TYPE_RESERVED) {
        outputDataType_ = dataType_;
        HCCL_DEBUG(
            "[CcuKernelArgAllGatherOmniPipeMesh1D] outputDataType is [INVALID], set outputDataType to[%d]",
            outputDataType_);
    }
    HCCL_INFO(
        "[CcuKernelArgAllGatherOmniPipeMesh1D] Init, KernelArgs are rankIdx[%u], rankId[%u], rankSize_[%u], dataType[%d], "
        "outputDataType[%d]",
        rankIdx_, rankId_, rankSize_, dataType_, outputDataType_);
}

HcclResult CcuKernelAllGatherOmniPipeMesh1D::InitResource()
{
    HCCL_INFO("[CcuKernelAllGatherOmniPipeMesh1D] [InitResource] begin");
    uint16_t channelIdx = 0;
    if (channels_.size() == 0) {
        HCCL_ERROR("[CcuKernelAllGatherOmniPipeMesh1D] channels is empty!");
        return HcclResult::HCCL_E_INTERNAL;
    }

    HCCL_INFO("InitResource channels_ %u", channels_.size());

    // 按照rank号从小到大遍历channels，遇到本rank就填充本地资源，否则依次取远端资源，要求给框架返回的Link同样是按顺序排列的
    for (uint64_t peerId = 0; peerId < rankSize_; peerId++) {
        if (peerId == rankIdx_) {
            input_ = CreateVariable();
            output_.push_back(CreateVariable());
            token_.push_back(CreateVariable());
        } else {
            HCCL_DEBUG("[CcuKernelAllGatherOmniPipeMesh1D] MyRank[%u], PeerId[%u], ChannelId[%u]", rankIdx_, peerId, channelIdx);
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
    groupOpSize_                 = CreateGroupOpSize();

    selfBit_ = 1 << rankIdx_;                              // 仅rankid位为1，其他位为0，代表本端准备好了
    allBit_ = ((1 << rankSize_) - 1) & (~(1 << rankIdx_)); // 仅rankid位为0，其他位为1，代表远端准备好了

    HCCL_INFO("[CcuKernelAllGatherOmniPipeMesh1D] [InitResource] end");
    return HcclResult::HCCL_SUCCESS;
}

void CcuKernelAllGatherOmniPipeMesh1D::LoadArgs() //TODO:change
{
    HCCL_INFO("[CcuKernelAllGatherOmniPipeMesh1D] [LoadArgs] begin");

    Load(input_);
    Load(output_[rankIdx_]);
    Load(token_[rankIdx_]);
    Load(sliceSize_);
    Load(repeatNum_);
    Load(sliceStride_);
    Load(localCopyFlag_);
    Load(groupOpSize_);

    for (int i = 0; i < MAX_STEP_NUM; i++) {
        inputOmniPipeSliceStride_.emplace_back(CreateVariable());
        Load(inputOmniPipeSliceStride_[i]);
    }

    for (int i = 0; i < MAX_STEP_NUM; i++) {
        outputOmniPipeSliceStride_.emplace_back(CreateVariable());
        Load(outputOmniPipeSliceStride_[i]);
    }

    HCCL_INFO("[CcuKernelAllGatherOmniPipeMesh1D] [LoadArgs] end");
}

// void CcuKernelAllGatherOmniPipeMesh1D::LoadArgs()
// {
//     HCCL_INFO("[CcuKernelAllGatherOmniPipeMesh1D] [LoadArgs] begin");

//     Load(input_);
//     Load(output_[rankIdx_]);
//     Load(token_[rankIdx_]);
//     Load(sliceSize_);
//     Load(repeatNum_);
//     Load(sliceStride_);
//     Load(localCopyFlag_);
//     Load(groupOpSize_);

//     hcomm::CcuRep::Variable repeatNumTmp_ = CreateVariable();
//     repeatNumTmp_ = repeatNum_;

//     CcuRep::Variable repeatNumAdd = CreateVariable();
//     repeatNumAdd  = 1;
//     uint64_t idx = 0;
//     CCU_WHILE(repeatNumTmp_ != UINT64_MAX) {
//         HCCL_INFO("local input omnipipe");
//         inputOmniPipeSliceStride_.emplace_back(CreateVariable());
//         Load(inputOmniPipeSliceStride_[idx]);
//         repeatNumTmp_ += repeatNumAdd;
//         idx++;
//     }

//     idx = 0;
//     repeatNumTmp_ = repeatNum_;
//     CCU_WHILE(repeatNumTmp_ != UINT64_MAX) {
//         HCCL_INFO("local output omnipipe");
//         outputOmniPipeSliceStride_.emplace_back(CreateVariable());
//         Load(outputOmniPipeSliceStride_[idx]);
//         repeatNumTmp_ += repeatNumAdd;
//         idx++;
//     }

//     HCCL_INFO("[CcuKernelAllGatherOmniPipeMesh1D] [LoadArgs] end");

//     return;
// }

void CcuKernelAllGatherOmniPipeMesh1D::PreSync()
{
    for (ChannelHandle channel : channels_) {
        HCCL_DEBUG("channels_ size %u", channels_.size());
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
    for (ChannelHandle channel : channels_) {
        HCCL_DEBUG("channels_ size %u", channels_.size());
        NotifyRecord(channel, CKE_IDX_0, 1 << POST_SYNC_ID);
    }
    for (ChannelHandle channel : channels_) {
        HCCL_DEBUG("channels_ size %u", channels_.size());
        NotifyWait(channel, CKE_IDX_0, 1 << POST_SYNC_ID);
    }
}

void CcuKernelAllGatherOmniPipeMesh1D::DoRepeatAllGather()
{
    HCCL_INFO("[CcuKernelAllGatherOmniPipeMesh1D] [DoRepeatAllGather] begin");

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

        HCCL_DEBUG("[DoRepeatAllGather] rankid %u, localcopy end", rankId_);
        WaitEvent(event_);

        HCCL_INFO("kernel local copy end");

        // return ; // 只做local copy  写了return 后面微码就不生成了？？
    }

    CCU_IF(localCopyFlag_ == 0) { // 由于 上面 return不能用，所以这边加了这个判断
        CcuRep::LocalAddr src = CreateLocalAddr();
        std::vector<CcuRep::RemoteAddr> dst;
        for (uint64_t rankIdx = 0; rankIdx < rankSize_; rankIdx++) {
            dst.push_back(CreateRemoteAddr());
        }

        CcuRep::Variable repeatNumAdd = CreateVariable();
        repeatNumAdd  = 1;
        uint32_t idx = 0;
        HCCL_DEBUG("kernel rankId %llu, rankIdx %llu, rankSize %llu", rankId_, rankIdx_, rankSize_);
        CCU_WHILE(repeatNum_ != UINT64_MAX) {
            repeatNum_ += repeatNumAdd;
            src.addr = output_[rankIdx_];
            for (uint64_t i = 0; i < rankIdx_; i++) {
                src.addr += sliceStride_;
            }
            src.addr += inputOmniPipeSliceStride_[idx];
            src.token = token_[rankIdx_];

            dst[rankSize_ - 1].addr = src.addr;
            dst[rankSize_ - 1].token = src.token;

            uint32_t curId = 0;

            for (uint64_t rankIdx = 0; rankIdx < rankSize_; rankIdx++) {
                if (rankIdx == rankIdx_) {
                    continue;
                }

                dst[curId].addr = output_[rankIdx];
                for (uint64_t i = 0; i < rankIdx_; i++) {
                    dst[curId].addr += sliceStride_;
                }
                dst[curId].addr += outputOmniPipeSliceStride_[idx];
                dst[curId].token = token_[rankIdx];
                curId++;
            }

            idx ++;
            HCCL_DEBUG("[DoRepeatAllGather] GroupBroadcast begin, dst size %u", dst.size());
            GroupBroadcast(channels_, dst, src, groupOpSize_);

            // CcuRep::CompletedEvent event1_;
            // event1_.SetMask(1);
            // WriteNb(channels_[0], dst[0], src, sliceSize_, event1_);
            // WaitEvent(event1_);
            HCCL_INFO("[DoRepeatAllGather] GroupBroadcast end");
        }
    }

    HCCL_INFO("[CcuKernelAllGatherOmniPipeMesh1D] [DoRepeatAllGather] end");

    return ;
}

HcclResult CcuKernelAllGatherOmniPipeMesh1D::Algorithm()
{
    HCCL_INFO("[CcuKernelArgAllGatherOmniPipeMesh1D] AllGatherOmniPipeMesh1D run");

    CHK_RET(InitResource());
    LoadArgs();
    PreSync();
    DoRepeatAllGather();
    PostSync();

    HCCL_INFO("[CcuKernelArgAllGatherOmniPipeMesh1D] AllGatherOmniPipeMesh1D end");

    return HcclResult::HCCL_SUCCESS;
}

std::vector<uint64_t> CcuKernelAllGatherOmniPipeMesh1D::GeneArgs(const CcuTaskArg &arg)
{
    HCCL_INFO("[CcuKernelAllGatherOmniPipeMesh1D] [GeneArgs] begin");
    const CcuTaskArgAllGatherOmniPipeMesh1D *taskArg
        = dynamic_cast<const CcuTaskArgAllGatherOmniPipeMesh1D *>(&arg);
    uint64_t inputAddr                   = taskArg->inputAddr_;
    uint64_t outputAddr                  = taskArg->outputAddr_;
    uint64_t tokenInfo                   = taskArg->token_;
    uint64_t sliceStride                 = taskArg->sliceStride_;

    uint64_t sendCount                   = taskArg->sendCount_;
    uint64_t sliceSize                   = taskArg->sliceSize_;
    uint64_t localCopyFlag               = taskArg->localCopyFlag_;
    uint64_t repeatNum                   = taskArg->repeatNum_;
    uint64_t repeatNumTmp                = UINT64_MAX - repeatNum;
    auto goSize                          = CalGoSize(sliceSize);

    HCCL_DEBUG("gosize goSize[0] %llu, goSize[1] %llu, goSize[2] %llu, goSize[3] %llu", goSize[0], goSize[1], goSize[2], goSize[3]);

    std::vector<uint64_t> taskArgs = {inputAddr, outputAddr, tokenInfo, sliceSize, repeatNumTmp, sliceStride, localCopyFlag,
                                    goSize[0], goSize[1], goSize[2], goSize[3]};

    for (int i = 0; i < MAX_STEP_NUM; i++) {
        taskArgs.emplace_back(taskArg->inputOmniPipeSliceStride_[i]);
    }

    for (int i = 0; i < MAX_STEP_NUM; i++) {
        taskArgs.emplace_back(taskArg->outputOmniPipeSliceStride_[i]);
    }

    HCCL_INFO("[CcuKernelArgAllGatherOmniPipeMesh1D] TaskArgs: rankIdx_[%u], rankId_[%u], inputAddr[%llu], outputAddr[%llu], "
               "sliceSize[%llu], repeatNum[%llu], UINT64_MAX[%llu], sliceStride[%llu], localCopyFlag[%llu]",
               rankIdx_, rankId_, inputAddr, outputAddr, sliceSize, repeatNumTmp, UINT64_MAX, sliceStride, localCopyFlag);

    HCCL_INFO("[CcuKernelAllGatherOmniPipeMesh1D] [GeneArgs] end");
    return taskArgs;
}

} // namespace ops_hccl