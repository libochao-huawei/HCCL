/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "ccu_kernel_all_reduce_mesh1d_flatten.h"

namespace ops_hccl {

constexpr int INPUT_XN_ID  = 0;
constexpr int TOKEN_XN_ID  = 2;
constexpr int POST_SYNC_CKE_IDX   = 0;
constexpr int PRE_SYNC_CKE_IDX    = 1;
constexpr uint16_t POST_CKE_BIT0  = 0;
constexpr uint16_t BIT_NUM_PER_CKE = 16;

CcuKernelAllReduceMesh1DFlatten::CcuKernelAllReduceMesh1DFlatten(const CcuKernelArg &arg)
    : CcuKernelAlgBase(arg)
{
    const CcuKernelArgAllReduceMesh1DFlatten *kernelArg = dynamic_cast<const CcuKernelArgAllReduceMesh1DFlatten *>(&arg);
    rankId_ = kernelArg->rankId_;
    rankSize_ = kernelArg->dimSize_;
    dataType_ = kernelArg->opParam_.DataDes.dataType;
    outputDataType_ = kernelArg->opParam_.DataDes.outputType;
    reduceOp_ = kernelArg->opParam_.reduceType;
    channels_ = kernelArg->channels;
    if (outputDataType_ == HcclDataType::HCCL_DATA_TYPE_RESERVED) {
        outputDataType_ = dataType_;
        HCCL_INFO("[CcuKernelAllReduceMesh1DFlatten] outputDataType is [INVALID], set outputDataType to[%u]",
            outputDataType_);
    }
    HCCL_INFO("[CcuKernelAllReduceMesh1DFlatten] Init, CtxArgs are rankId[%u], rankSize_[%u], dataType[%u], "
        "outputDataType[%u], reduceOp[%u]", rankId_, rankSize_, dataType_, outputDataType_, reduceOp_);
}

HcclResult CcuKernelAllReduceMesh1DFlatten::Algorithm()
{
    HCCL_INFO("[CcuKernelAllReduceMesh1DFlatten] AllReduceMesh1DFlatten start");
    CHK_RET(InitResource());
    LoadArgs();  // 加载 taskArg 参数
    Presync();  // 跨卡前同步，交换参数信息

    DoGroupReduce();

    Postsync();  // 所有搬运任务结束后，跨卡后同步

    HCCL_INFO("[CcuKernelAllReduceMesh1DFlatten] AllReduceMesh1DFlatten end");
    return HcclResult::HCCL_SUCCESS;
}

HcclResult CcuKernelAllReduceMesh1DFlatten::InitResource()
{
    HCCL_INFO("[CcuKernelAllReduceMesh1DFlatten] InitResource start");
    // 初始化资源
    output_ = CreateVariable();
    localEvent_     = CreateCompletedEvent();
    for(uint32_t i = 0; i < (rankSize_ + BIT_NUM_PER_CKE - 1)/BIT_NUM_PER_CKE; i++){
        event_.push_back(CreateCompletedEvent());
    }
    uint16_t channelIdx = 0;
    if (channels_.size() == 0) {
        // THROW<NullPtrException>(StringFormat("CcuKernelAllReduceMesh1DFlatten channels is empty")); //打印
        return HcclResult::HCCL_E_INTERNAL;
    }
    // 按照rank号从小到大遍历channels，遇到本rank就填充本地资源，否则依次取远端资源，要求给框架返回的Link同样是按顺序排列的
    for (uint64_t peerId = 0; peerId < rankSize_; peerId++) {
        if (peerId == rankId_) {
            input_.push_back(CreateVariable());
            token_.push_back(CreateVariable());
        } else {
            HCCL_INFO("[CcuKernelAllReduceMesh1DFlatten] MyRank[%u], PeerId[%llu], ChannelId[%u]",
                rankId_, peerId, channelIdx);
            CcuRep::Variable inputVar, tokenVar;
            CHK_RET(CreateVariable(channels_[channelIdx], INPUT_XN_ID, &inputVar));
            input_.push_back(inputVar);
            CHK_RET(CreateVariable(channels_[channelIdx], TOKEN_XN_ID, &tokenVar));
            token_.push_back(tokenVar);
            channelIdx++;
        }
    }
    HCCL_INFO("[CcuKernelAllReduceMesh1DFlatten] InitResource end");
    return HcclResult::HCCL_SUCCESS;
}

void CcuKernelAllReduceMesh1DFlatten::LoadArgs()
{
    HCCL_INFO("[CcuKernelAllReduceMesh1DFlatten] LoadArgs start");
    Load(input_[rankId_]);
    Load(output_);
    Load(token_[rankId_]);
    Load(sliceSize_);
    HCCL_INFO("[CcuKernelAllReduceMesh1DFlatten] LoadArgs end");
}

void CcuKernelAllReduceMesh1DFlatten::Presync()
{
    HCCL_INFO("[CcuKernelAllReduceMesh1DFlatten] Presync start");
    for (auto &channel : channels_) {
        NotifyRecord(channel, PRE_SYNC_CKE_IDX, INPUT_XN_ID, input_[rankId_], 1 << INPUT_XN_ID);
        NotifyRecord(channel, PRE_SYNC_CKE_IDX, TOKEN_XN_ID, token_[rankId_], 1 << TOKEN_XN_ID);
    }
    uint32_t allBit = 1 << INPUT_XN_ID | 1 << TOKEN_XN_ID;
    for (auto &chn : channels_) {
        NotifyWait(chn, PRE_SYNC_CKE_IDX, allBit);
    }
    HCCL_INFO("[CcuKernelAllReduceMesh1DFlatten] Presync end");
}

void CcuKernelAllReduceMesh1DFlatten::Postsync()
{
    HCCL_INFO("[CcuKernelAllReduceMesh1DFlatten] Postsync start");
    uint16_t postCkeBit = 1 << POST_CKE_BIT0;
    for (auto &ch : channels_) {
        NotifyRecord(ch, POST_SYNC_CKE_IDX, postCkeBit);
    }
    for (auto &channel : channels_) {
        NotifyWait(channel, POST_SYNC_CKE_IDX, postCkeBit);
    }
    HCCL_INFO("[CcuKernelAllReduceMesh1DFlatten] Postsync end");
}

void CcuKernelAllReduceMesh1DFlatten::DoGroupReduce()
{
    HCCL_INFO("[CcuKernelAllReduceMesh1DFlatten] DoGroupReduce start");
    // 初始化地址寄存器
    std::vector<CcuRep::RemoteAddr> reduceSrc;
    for (uint64_t rankIdx = 0; rankIdx < rankSize_; rankIdx++) {
        reduceSrc.push_back(CreateRemoteAddr());
    }
    CcuRep::LocalAddr reduceDst = CreateLocalAddr();

    // 填充地址
    uint32_t dstId = 0;
    uint32_t curId = 0;
    // SRC
    for (uint64_t rankIdx = 0; rankIdx < rankSize_; rankIdx++) {
        if (rankIdx != rankId_) {
            curId = dstId;
            dstId++;
        } else {
            curId = rankSize_ - 1;
        }
        reduceSrc[curId].addr = input_[rankIdx];
        reduceSrc[curId].token = token_[rankIdx];
    }

    // DST
    reduceDst.addr  = output_;
    reduceDst.token = token_[rankId_];
    //localcopy,input->output
    localEvent_.SetMask(1);
    LocalCopyNb(reduceDst, reduceSrc[rankSize_ - 1], sliceSize_, localEvent_);
    WaitEvent(localEvent_);

    // 执行 reduce 操作,input->output
    for (uint32_t channelId = 0; channelId < channels_.size();channelId++) {
        uint32_t eventIdx= channelId / BIT_NUM_PER_CKE;
        event_[eventIdx].SetMask(1 << (channelId % BIT_NUM_PER_CKE));
        ReadReduceNb(channels_[channelId], reduceDst, reduceSrc[channelId], sliceSize_, dataType_, reduceOp_, event_[eventIdx]);
    }

    for (uint32_t i = 0; i < (rankSize_ + BIT_NUM_PER_CKE - 2)/BIT_NUM_PER_CKE; i++){
        if(i == (rankSize_ + BIT_NUM_PER_CKE - 2)/BIT_NUM_PER_CKE - 1){
            if((rankSize_ - 1) % BIT_NUM_PER_CKE == 0){
                event_[i].SetMask((1 << BIT_NUM_PER_CKE) - 1);
            } else {
                event_[i].SetMask((1 << ((rankSize_ - 1) % BIT_NUM_PER_CKE)) - 1);
            }
        } else {
            event_[i].SetMask((1 << BIT_NUM_PER_CKE) - 1);
        }
        WaitEvent(event_[i]);
    }
    HCCL_INFO("[CcuKernelAllReduceMesh1DFlatten] DoGroupReduce end");
    return;
}

std::vector<uint64_t> CcuKernelAllReduceMesh1DFlatten::GeneArgs(const CcuTaskArg &arg)
{
    HCCL_INFO("[CcuKernelAllReduceMesh1DFlatten] GeneArgs start");
    const CcuTaskArgAllReduceMesh1DFlatten *taskArg    = dynamic_cast<const CcuTaskArgAllReduceMesh1DFlatten *>(&arg);
    uint64_t                                inputAddr  = taskArg->inputAddr_;
    uint64_t                                outputAddr = taskArg->outputAddr_;
    uint64_t                                tokenInfo  = taskArg->token_;
    uint64_t                                sliceSize  = taskArg->sliceSize_;

    HCCL_INFO("[CcuKernelAllReduceMesh1DFlatten] GeneArgs, taskArg are inputAddr[%llu], outputAddr[%llu], "
        "sliceSize[%llu]", inputAddr, outputAddr, sliceSize);

    std::vector<uint64_t> taskArgList{inputAddr, outputAddr, tokenInfo};

    HCCL_INFO("[CcuKernelAllReduceMesh1DFlatten] GeneArgs end");
    return taskArgList;
}
}