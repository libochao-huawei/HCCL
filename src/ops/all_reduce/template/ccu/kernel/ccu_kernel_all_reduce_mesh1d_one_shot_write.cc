/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "ccu_kernel_all_reduce_mesh1d_one_shot_write.h"

namespace ops_hccl {

constexpr int PRESYNC_CKE_IDX  = 0;   // Presync/Postsync 屏障固定使用 CKE 0
constexpr int POSTSYNC_CKE_IDX = 1;

CcuKernelAllReduceMesh1DOneShotWrite::CcuKernelAllReduceMesh1DOneShotWrite(const hcomm::CcuKernelArg &arg)
    : CcuKernelAlgBase(arg)
{
    const auto *kernelArg = dynamic_cast<const CcuKernelArgAllReduceMesh1DOneShotWrite *>(&arg);
    rankId_    = kernelArg->rankId_;
    rankSize_  = kernelArg->dimSize_;
    dataType_  = kernelArg->opParam_.DataDes.dataType;
    outputDataType_ = kernelArg->opParam_.DataDes.outputType;
    reduceOp_  = kernelArg->opParam_.reduceType;
    channels_  = kernelArg->channels;
    if (outputDataType_ == HcclDataType::HCCL_DATA_TYPE_RESERVED) {
        outputDataType_ = dataType_;
        HCCL_INFO("[CcuKernelAllReduceMesh1DOneShotWrite] outputDataType is [INVALID], set outputDataType to[%u]",
            outputDataType_);
    }
    HCCL_INFO("[CcuKernelAllReduceMesh1DOneShotWrite] Init, rankId[%u], rankSize[%lu], dataType[%u], "
        "outputDataType[%u], reduceOp[%u]", rankId_, rankSize_, dataType_, outputDataType_, reduceOp_);
}

HcclResult CcuKernelAllReduceMesh1DOneShotWrite::Algorithm()
{
    HCCL_INFO("[CcuKernelAllReduceMesh1DOneShotWrite] AllReduceMesh1DOneShotWrite start");
    CHK_RET(InitResource());
    LoadArgs();
    // Presync();
    DoGroupWrite();
    // Postsync();
    HCCL_INFO("[CcuKernelAllReduceMesh1DOneShotWrite] AllReduceMesh1DOneShotWrite end");
    return HcclResult::HCCL_SUCCESS;
}

HcclResult CcuKernelAllReduceMesh1DOneShotWrite::InitResource()
{
    HCCL_INFO("[CcuKernelAllReduceMesh1DOneShotWrite] InitResource start");
    if (channels_.empty()) {
        return HcclResult::HCCL_E_INTERNAL;
    }
    input_      = CreateVariable();
    output_     = CreateVariable();
    token_      = CreateVariable();
    groupOpSize_ = CreateGroupOpSize();
    HCCL_INFO("[CcuKernelAllReduceMesh1DOneShotWrite] InitResource end");
    return HcclResult::HCCL_SUCCESS;
}

void CcuKernelAllReduceMesh1DOneShotWrite::LoadArgs()
{
    HCCL_INFO("[CcuKernelAllReduceMesh1DOneShotWrite] LoadArgs start");
    Load(input_);
    Load(output_);
    Load(token_);
    Load(groupOpSize_);
    HCCL_INFO("[CcuKernelAllReduceMesh1DOneShotWrite] LoadArgs end");
}

void CcuKernelAllReduceMesh1DOneShotWrite::Presync()
{
    HCCL_INFO("[CcuKernelAllReduceMesh1DOneShotWrite] Presync start");
    // write 模式不需要交换远端地址，只做同步屏障确保所有 rank 数据就绪（CKE 0, bit 1）
    for (auto &ch : channels_) {
        NotifyRecord(ch, PRESYNC_CKE_IDX, PRE_SYNC_MASK);
    }
    for (auto &ch : channels_) {
        NotifyWait(ch, PRESYNC_CKE_IDX, PRE_SYNC_MASK);
    }
    HCCL_INFO("[CcuKernelAllReduceMesh1DOneShotWrite] Presync end");
}

void CcuKernelAllReduceMesh1DOneShotWrite::Postsync()
{
    HCCL_INFO("[CcuKernelAllReduceMesh1DOneShotWrite] Postsync start");
    // 后同步屏障（CKE 0, bit 0）
    for (auto &ch : channels_) {
        NotifyRecord(ch, POSTSYNC_CKE_IDX, POST_SYNC_MASK);
    }
    for (auto &ch : channels_) {
        NotifyWait(ch, POSTSYNC_CKE_IDX, POST_SYNC_MASK);
    }
    HCCL_INFO("[CcuKernelAllReduceMesh1DOneShotWrite] Postsync end");
}

void CcuKernelAllReduceMesh1DOneShotWrite::DoGroupWrite()
{
    HCCL_INFO("[CcuKernelAllReduceMesh1DOneShotWrite] DoGroupWrite start");
    CcuRep::LocalAddr src = CreateLocalAddr();
    src.addr  = input_;
    src.token = token_;

    CcuRep::LocalAddr dst = CreateLocalAddr();
    dst.addr  = output_;
    dst.token = token_;

    GroupWrite(channels_, rankId_, dst, src, groupOpSize_, dataType_, outputDataType_, reduceOp_);
    HCCL_INFO("[CcuKernelAllReduceMesh1DOneShotWrite] DoGroupWrite end");
}

std::vector<uint64_t> CcuKernelAllReduceMesh1DOneShotWrite::GeneArgs(const hcomm::CcuTaskArg &arg)
{
    HCCL_INFO("[CcuKernelAllReduceMesh1DOneShotWrite] GeneArgs start");
    const auto *taskArg   = dynamic_cast<const CcuTaskArgAllReduceMesh1DOneShotWrite *>(&arg);
    uint64_t    inputAddr  = taskArg->inputAddr_;
    uint64_t    outputAddr = taskArg->outputAddr_;
    uint64_t    tokenInfo  = taskArg->token_;
    uint64_t    sliceSize  = taskArg->sliceSize_;

    auto blockGoSize = CalGoSize(sliceSize);

    HCCL_INFO("[CcuKernelAllReduceMesh1DOneShotWrite] GeneArgs, inputAddr[%llu], outputAddr[%llu], sliceSize[%llu]",
        inputAddr, outputAddr, sliceSize);

    std::vector<uint64_t> taskArgList{inputAddr, outputAddr, tokenInfo};
    for (auto val : blockGoSize) {
        taskArgList.push_back(val);
    }

    HCCL_INFO("[CcuKernelAllReduceMesh1DOneShotWrite] GeneArgs end");
    return taskArgList;
}

} // namespace ops_hccl
