/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "ccu_kernel_alg_base.h"
#include "ccu_kernel_reduce_scatter_v_mesh1d_mem2mem.h"

namespace ops_hccl {
using namespace hcomm;

// bit序号，每种信号用一个bit
constexpr int INPUT_XN_ID   = 0;
constexpr int SCRATCH_XN_ID = 1;
constexpr int TOKEN_XN_ID   = 2;
constexpr int POST_SYNC_ID   = 3;
// cke序号
constexpr int CKE_IDX_0     = 0;

CcuKernelReduceScatterVMesh1DMem2Mem::CcuKernelReduceScatterVMesh1DMem2Mem(const CcuKernelArg &arg)
    : CcuKernelAlgBase(arg)
{
    const CcuKernelArgReduceScatterVMesh1DMem2Mem *kernelArg
        = dynamic_cast<const CcuKernelArgReduceScatterVMesh1DMem2Mem *>(&arg);
    rankId_         = kernelArg->rankId_;
    rankSize_       = kernelArg->dimSize_;
    channels_       = kernelArg->channels;
    dataType_       = kernelArg->opParam_.vDataDes.dataType;
    outputDataType_ = kernelArg->opParam_.vDataDes.dataType;
    if (outputDataType_ == HcclDataType::HCCL_DATA_TYPE_RESERVED) {
        outputDataType_ = dataType_;
        HCCL_DEBUG(
            "[CcuKernelReduceScatterVMesh1DMem2Mem] outputDataType is [INVALID], set outputDataType to[%d]",
            outputDataType_);
    }
    reduceOp_ = kernelArg->opParam_.reduceType;
    HCCL_INFO(
        "[CcuKernelReduceScatterVMesh1DMem2Mem] Init, KernelArgs are rankId[%u], rankSize_[%u], dataType[%d], "
        "outputDataType[%d], reduceOp[%d]",
        rankId_, rankSize_, dataType_, outputDataType_, reduceOp_);
}

HcclResult CcuKernelReduceScatterVMesh1DMem2Mem::InitResource()
{
    uint16_t channelIdx = 0;
    if (channels_.size() == 0) {
        HCCL_ERROR("[CcuKernelReduceScatterVMesh1DMem2Mem] channels is empty!");
        return HcclResult::HCCL_E_INTERNAL;
    }

    // 按照rank号从小到大遍历channels，遇到本rank就填充本地资源，否则依次取远端资源，要求给框架返回的Link同样是按顺序排列的
    for (uint64_t peerId = 0; peerId < rankSize_; peerId++) {
        if (peerId == rankId_) {
            input_.push_back(CreateVariable());
            token_.push_back(CreateVariable());
        } else {
            HCCL_DEBUG("[CcuKernelReduceScatterVMesh1DMem2Mem] MyRank[%u], PeerId[%u], ChannelId[%u]",
                       rankId_, peerId, channelIdx);
            CcuRep::Variable inputVar, tokenVar;
            CHK_RET(CreateVariable(channels_[channelIdx], INPUT_XN_ID, &inputVar));
            CHK_RET(CreateVariable(channels_[channelIdx], TOKEN_XN_ID, &tokenVar));
            input_.push_back(inputVar);  // 获取channel中id=0的Var来传递input
            token_.push_back(tokenVar);
            channelIdx++;
        }
    }
    output_          = CreateVariable();
    scratch_         = CreateVariable();
    scratchInterval_ = CreateVariable();
    sliceSize_       = CreateVariable();
    offset_          = CreateVariable();
    reduceGosize_ = CreateGroupOpSize();

    src = CreateLocalAddr();
    for (uint64_t rankIdx = 0; rankIdx < rankSize_; rankIdx++) {
        if (rankIdx == rankId_) {
            dst.push_back({});
        }
        else {
            dst.push_back(CreateRemoteAddr());
        }
    }
    event_ = CreateCompletedEvent();

    for (uint32_t rankIdx = 0; rankIdx < rankSize_; rankIdx++) {
        reduceScatterVSrc_.push_back(CreateRemoteAddr());
        reduceScatterVDst_.push_back(CreateLocalAddr());
    }
    return HcclResult::HCCL_SUCCESS;
}

void CcuKernelReduceScatterVMesh1DMem2Mem::LoadArgs()
{
    Load(input_[rankId_]);
    Load(output_);
    Load(token_[rankId_]);
    Load(scratch_);
    Load(scratchInterval_);
    Load(sliceSize_);
    Load(offset_);
    Load(reduceGosize_);
    return;
}

void CcuKernelReduceScatterVMesh1DMem2Mem::PreSync()
{
    for (ChannelHandle channel : channels_) {
        NotifyRecord(channel, CKE_IDX_0, INPUT_XN_ID, input_[rankId_], 1 << INPUT_XN_ID);
        NotifyRecord(channel, CKE_IDX_0, TOKEN_XN_ID, token_[rankId_], 1 << TOKEN_XN_ID);
    }

    uint16_t allBit = 1 << INPUT_XN_ID | 1 << TOKEN_XN_ID;
    for (ChannelHandle channel : channels_) {
        NotifyWait(channel, CKE_IDX_0, allBit);
    }
    return;
}

void CcuKernelReduceScatterVMesh1DMem2Mem::PostSync()
{
    for (ChannelHandle channel : channels_) {
        NotifyRecord(channel, CKE_IDX_0, 1 << POST_SYNC_ID);         // bit index = 4, 用作后同步。cke都可以用同一个，所以都是CKE_IDX_0
    }
    for (ChannelHandle channel : channels_) {
        NotifyWait(channel, CKE_IDX_0, 1 << POST_SYNC_ID);
    }
}

void CcuKernelReduceScatterVMesh1DMem2Mem::CollectAllRanksSlice(std::vector<CcuRep::RemoteAddr>& tmpSrc,
    std::vector<CcuRep::LocalAddr>& tmpDst)
{
    uint32_t channelId = 0;
    for (uint32_t rankIdx = 0; rankIdx < rankSize_; rankIdx++) {
        if (rankIdx == rankId_) {
            // 跳过本卡
            event_.SetMask(1 << rankIdx);
            RecordEvent(event_);
        } else {
            event_.SetMask(1 << rankIdx);
            ReadNb(channels_[channelId], tmpDst[rankIdx], tmpSrc[rankIdx], sliceSize_, event_);
            channelId++;
        }
    }
    // 等读完所有对端
    event_.SetMask((1 << rankSize_) - 1);
    WaitEvent(event_);
}

void CcuKernelReduceScatterVMesh1DMem2Mem::PrepareReduceScatterVData(std::vector<CcuRep::RemoteAddr>& reduceScatterVSrc,
    std::vector<CcuRep::LocalAddr>& reduceScatterVDst)
{
    CcuRep::Variable scratchOffset = CreateVariable();
    scratchOffset = 0;
    for (uint32_t rankIdx = 0; rankIdx < rankSize_; rankIdx++) {
        if (rankIdx == rankId_) {
            // 本卡不使用 scratch 数据依然放在 input 上
            reduceScatterVDst[rankIdx].addr = input_[rankId_];
            reduceScatterVDst[rankIdx].addr += offset_;
            reduceScatterVDst[rankIdx].token = token_[rankId_];
            continue;
        }
        // 从远端的 input 上，读取本卡所需的数据片
        reduceScatterVSrc[rankIdx].addr = input_[rankIdx];
        reduceScatterVSrc[rankIdx].addr += offset_;
        reduceScatterVSrc[rankIdx].token = token_[rankIdx];

        // 将数据放到本卡的 scratch 上
        reduceScatterVDst[rankIdx].addr  = scratch_;
        reduceScatterVDst[rankIdx].addr += scratchOffset;
        scratchOffset += scratchInterval_;
        reduceScatterVDst[rankIdx].token = token_[rankId_];
    }
    return;
}

void CcuKernelReduceScatterVMesh1DMem2Mem::DoReduceScatterV()
{
    CCU_IF(sliceSize_ != 0) {
        PrepareReduceScatterVData(reduceScatterVSrc_, reduceScatterVDst_);
        CollectAllRanksSlice(reduceScatterVSrc_, reduceScatterVDst_);

        CcuRep::LocalAddr outDst = CreateLocalAddr();
        outDst.addr = output_;
        outDst.token = token_[rankId_];

        // 执行reduce操作
        GroupLocalReduce(outDst, reduceScatterVDst_, reduceGosize_,
                        dataType_, outputDataType_, reduceOp_);
    }
}

HcclResult CcuKernelReduceScatterVMesh1DMem2Mem::Algorithm()
{
    HCCL_INFO("[CcuKernelReduceScatterVMesh1DMem2Mem] ReduceScatterVMesh1DMem2Mem run");

    CHK_RET(InitResource());

    LoadArgs();

    PreSync();

    DoReduceScatterV();

    PostSync();

    HCCL_INFO("[CcuKernelReduceScatterVMesh1DMem2Mem] ReduceScatterVMesh1DMem2Mem end");

    return HcclResult::HCCL_SUCCESS;
}

std::vector<uint64_t> CcuKernelReduceScatterVMesh1DMem2Mem::GeneArgs(const CcuTaskArg &arg)
{
    const CcuTaskArgReduceScatterVMesh1DMem2Mem *taskArg
        = dynamic_cast<const CcuTaskArgReduceScatterVMesh1DMem2Mem *>(&arg);
    uint64_t inputAddr       = taskArg->inputAddr_;
    uint64_t outputAddr      = taskArg->outputAddr_;
    uint64_t tokenInfo       = taskArg->token_;
    uint64_t scratchAddr     = taskArg->scratchAddr_;
    uint64_t scratchInterval = taskArg->scratchInterval_;
    uint64_t sliceSize       = taskArg->sliceSize_;
    uint64_t offset          = taskArg->offset_;
    auto goSize = CalGoSize(sliceSize);

    std::vector<uint64_t> taskArgs = {inputAddr, outputAddr, tokenInfo,
                                     scratchAddr, scratchInterval, sliceSize, offset};

    HCCL_INFO("[CcuKernelReduceScatterVMesh1DMem2Mem] TaskArgs: inputAddr[%llu], outputAddr[%llu], "
        "scratchAddr[%llu], scratchInterval[%llu], sliceSize[%llu], offset[%llu]",
        inputAddr, outputAddr, scratchAddr, scratchInterval, sliceSize, offset);
    taskArgs.insert(taskArgs.cend(), goSize.cbegin(), goSize.cend());
    return taskArgs;
}

} // namespace ops_hccl
