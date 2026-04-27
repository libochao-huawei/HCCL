/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "ccu_kernel_scatter_omnipipe_mesh_1D.h"
#include "ccu_kernel_alg_base.h"

namespace ops_hccl {
using namespace hcomm;

constexpr uint16_t OUTPUT_XN_ID = 0;
constexpr uint16_t TOKEN_XN_ID = 1;
constexpr uint16_t POST_SYNC_ID = 2;
constexpr uint16_t CKE_IDX_0 = 0;

CcuKernelScatterOmniPipeMesh1D::CcuKernelScatterOmniPipeMesh1D(const CcuKernelArg &arg)
    : CcuKernelAlgBase(arg)
{
    auto kernelArg = dynamic_cast<const CcuKernelArgScatterOmniPipeMesh1D *>(&arg);
    rankSize_ = kernelArg->dimSize_;
    rankId_ = kernelArg->rankId_;
    channels_ = kernelArg->channels;
    dataType_ = kernelArg->opParam_.DataDes.dataType;
    rootId_ = kernelArg->opParam_.root;
    userRank_ = kernelArg->subCommRanks_[0][rankId_];
    
    // 判断当前节点是否为root节点的同轴线节点
    isSameAxisAsRoot_ = kernelArg->isSameAxisAsRoot;

    HCCL_INFO("[%s]rootId[%u], userRank[%u] rankId[%u], rankSize_[%u], dataType[%d], dataType_[%d], "
        "channels size[%u]", __func__, rootId_, userRank_, rankId_, rankSize_, dataType_, dataType_, 
        channels_.size());
}

std::string CcuKernelArgScatterOmniPipeMesh1D::GetKernelSignature() const
{
    std::stringstream ss;
    ss << "CcuKernelScatterOmniPipeMesh1D" << "." << dimSize_ << "." << rankId_ << "." 
       << static_cast<uint32_t>(opParam_.DataDes.dataType) << "." << opParam_.root;
    return ss.str();
}

HcclResult CcuKernelScatterOmniPipeMesh1D::InitResource()
{
    if (channels_.size() == 0) {
        HCCL_ERROR("[CcuKernelScatterOmniPipeMesh1D] channels is empty!");
        return HcclResult::HCCL_E_INTERNAL;
    }

    // 按照rank号从小到大遍历channels，遇到本rank就填充本地资源，否则依次取远端资源
    for (uint64_t peerId = 0; peerId < rankSize_; peerId++) {
        if (peerId == rankId_) {
            output_.push_back(CreateVariable());
            token_.push_back(CreateVariable());
        } else {
            CcuRep::Variable outputVar;
            CcuRep::Variable tokenVar;
            CHK_RET(CreateVariable(channels_[peerId - (peerId > rankId_ ? 1 : 0)], OUTPUT_XN_ID, &outputVar));
            output_.push_back(outputVar);
            CHK_RET(CreateVariable(channels_[peerId - (peerId > rankId_ ? 1 : 0)], TOKEN_XN_ID, &tokenVar));
            token_.push_back(tokenVar);
        }
    }

    // 初始化变量
    input_ = CreateVariable();
    sliceSize_ = CreateVariable();
    inputSliceStride_ = CreateVariable();
    outputSliceStride_ = CreateVariable();
    inputOmniPipeSliceStride_ = CreateVariable();
    outputOmniPipeSliceStride_ = CreateVariable();
    offset_ = CreateVariable();
    localCopyFlag_ = CreateVariable();
    stepIndex_ = CreateVariable();

    // 初始化内存地址
    inputMem_.reserve(rankSize_);
    outputMem_.reserve(rankSize_);
    for (uint64_t rankIdx = 0; rankIdx < rankSize_; rankIdx++) {
        inputMem_.push_back(CreateLocalAddr());
        outputMem_.push_back(CreateRemoteAddr());
    }

    // 初始化事件
    event_ = CreateCompletedEvent();
    
    return HcclResult::HCCL_SUCCESS;
}

void CcuKernelScatterOmniPipeMesh1D::LoadArgs()
{
    // 使用基类的Load方法加载变量
    Load(input_);
    Load(output_[rankId_]);
    Load(sliceSize_);
    Load(offset_);
    Load(token_[rankId_]);
    Load(localCopyFlag_);
    Load(inputSliceStride_);
    Load(outputSliceStride_);
    Load(inputOmniPipeSliceStride_);
    Load(outputOmniPipeSliceStride_);
    Load(currentStep_);
    
}

void CcuKernelScatterOmniPipeMesh1D::PreSync()
{
    uint32_t allBit = 1 << OUTPUT_XN_ID | 1 << TOKEN_XN_ID;
    for (auto channel : channels_) {                                                           // 遍历所有通道
        NotifyRecord(channel, CKE_IDX_0, OUTPUT_XN_ID, output_[rankId_], 1 << OUTPUT_XN_ID);  // 广播output地址
        NotifyRecord(channel, CKE_IDX_0, TOKEN_XN_ID, token_[rankId_], 1 << TOKEN_XN_ID);     // 广播token
    }
    for (auto channel : channels_) {
        NotifyWait(channel, CKE_IDX_0, allBit);
    }
}

void CcuKernelScatterOmniPipeMesh1D::PostSync()
{
    for (auto channel : channels_) {  // 通知所有通道操作完成
        NotifyRecord(channel, CKE_IDX_0, 1 << POST_SYNC_ID);
    }
    for (auto channel : channels_) {
        NotifyWait(channel, CKE_IDX_0, 1 << POST_SYNC_ID);
    }
}

void CcuKernelScatterOmniPipeMesh1D::DoScatter()
{
    HCCL_DEBUG("[%s] userRank[%u] rankId[%u] start", __func__, userRank_, rankId_);
    // 为root写到自己的地址专门创建一个LocalAddr变量
    CcuRep::LocalAddr myOutput = CreateLocalAddr();
    myOutput.addr = outputMem_[rankId_].addr;
    myOutput.token = outputMem_[rankId_].token;

    uint32_t channelId = 0;

    // root卡的数据发送到所有卡
    for (uint64_t rankIdx = 0; rankIdx < rankSize_; rankIdx++) {
        event_.SetMask(1 << rankIdx);
        
        if (rankIdx == rankId_) {
            // 如果输入输出地址不同，还需要进行一次本地搬运
            CCU_IF(localCopyFlag_ == 0)
            {
                LocalCopyNb(myOutput, inputMem_[rankIdx], sliceSize_, event_);
            }
            // 如果输入输出地址相同，不进行本地搬运，只发送同步信号
            CCU_IF(localCopyFlag_ != 0)
            {
                RecordEvent(event_);
            }
        } else {
            // 发送数据到其他节点
            CCU_IF(sliceSize_ != 0) {
                WriteNb(channels_[channelId], outputMem_[rankIdx], inputMem_[rankIdx], sliceSize, event_);
                channelId++;
            }
        }
    }
}

HcclResult CcuKernelScatterOmniPipeMesh1D::Algorithm()
{
    HCCL_INFO("[CcuKernelScatterOmniPipeMesh1D] ScatterOmniPipeMesh1D run");

    CHK_RET(InitResource());

    LoadArgs();

    PreSync();

    // 设置每张卡输入输出的起始地址
    for (uint64_t curId = 0; curId < rankSize_; curId++) {
        inputMem_[curId].token = token_[curId];  // 设置每张卡的输入token
        outputMem_[curId].token = token_[curId]; // 设置每张卡的输出token

        // 计算输入地址：基础地址 + rank偏移 + OmniPipe步长
        inputMem_[curId].addr = input_ + curId * inputSliceStride_ + inputOmniPipeSliceStride_;

        // 计算输出地址：基础地址 + rank偏移 + OmniPipe步长
        outputMem_[curId].addr = output_[curId] + curId * outputSliceStride_ + outputOmniPipeSliceStride_;
    }

    // 判断是否需要执行发送操作
    // 1. 根节点始终执行发送
    // 2. 同轴线节点在后续步骤执行发送
    bool needToSend = (rankId_ == rootId_) || (isSameAxisAsRoot_ && currentStep_ > 0);

    if (needToSend) {
        // 根节点或同轴线节点在后续步骤执行scatter操作
        DoScatter();
    } else {
        // 同轴线节点在第一步或斜对角节点始终等待接收完成
        event_.SetMask(1 << rankId_);
        WaitEvent(event_);
    }

    PostSync();

    HCCL_INFO("[CcuKernelScatterOmniPipeMesh1D] ScatterOmniPipeMesh1D end");

    return HcclResult::HCCL_SUCCESS;
}

// 生成SQE的参数
std::vector<uint64_t> CcuKernelScatterOmniPipeMesh1D::GeneArgs(const hcomm::CcuTaskArg &arg)
{
    const auto taskArg = dynamic_cast<const CcuTaskArgScatterOmniPipeMesh1D &>(arg);

    // 生成CCU硬件执行参数
    std::vector<uint64_t> taskArgs = {
        taskArg.inputAddr_,
        taskArg.outputAddr_,
        taskArg.sliceSize_,
        taskArg.offset_,
        taskArg.token_,
        taskArg.localCopyFlag_,
        taskArg.inputSliceStride_,
        taskArg.outputSliceStride_,
        taskArg.inputOmniPipeSliceStride_,
        taskArg.outputOmniPipeSliceStride_,
        taskArg.stepIndex_
    };

    return taskArgs;
}

} // namespace ops_hccl
