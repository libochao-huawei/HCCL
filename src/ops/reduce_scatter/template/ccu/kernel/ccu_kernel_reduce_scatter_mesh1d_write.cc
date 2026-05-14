/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "ccu_kernel_reduce_scatter_mesh1d_write.h"

namespace ops_hccl {
using namespace hcomm;

constexpr int RS_PRESYNC_CKE_IDX  = 0;
constexpr int RS_POSTSYNC_CKE_IDX = 0;

CcuKernelReduceScatterMesh1DWrite::CcuKernelReduceScatterMesh1DWrite(const CcuKernelArg &arg)
    : CcuKernelAlgBase(arg)
{
    const auto *kernelArg = dynamic_cast<const CcuKernelArgReduceScatterMesh1DWrite *>(&arg);
    rankId_        = kernelArg->rankId_;
    rankSize_      = kernelArg->dimSize_;
    channels_      = kernelArg->channels;
    dataType_      = kernelArg->opParam_.DataDes.dataType;
    outputDataType_ = kernelArg->opParam_.DataDes.outputType;
    if (outputDataType_ == HcclDataType::HCCL_DATA_TYPE_RESERVED) {
        outputDataType_ = dataType_;
    }
    reduceOp_ = kernelArg->opParam_.reduceType;
    HCCL_INFO("[CcuKernelReduceScatterMesh1DWrite] Init, rankId[%u], rankSize[%lu]", rankId_, rankSize_);
}

HcclResult CcuKernelReduceScatterMesh1DWrite::InitResource()
{
    if (channels_.empty()) {
        return HcclResult::HCCL_E_INTERNAL;
    }
    input_       = CreateVariable();
    output_      = CreateVariable();
    token_       = CreateVariable();
    groupOpSize_ = CreateGroupOpSize();
    return HcclResult::HCCL_SUCCESS;
}

void CcuKernelReduceScatterMesh1DWrite::LoadArgs()
{
    Load(input_);
    Load(output_);
    Load(token_);
    Load(groupOpSize_);
}

void CcuKernelReduceScatterMesh1DWrite::Presync()
{
    for (auto &ch : channels_) {
        NotifyRecord(ch, RS_PRESYNC_CKE_IDX, PRE_SYNC_MASK);
    }
    for (auto &ch : channels_) {
        NotifyWait(ch, RS_PRESYNC_CKE_IDX, PRE_SYNC_MASK);
    }
}

void CcuKernelReduceScatterMesh1DWrite::Postsync()
{
    for (auto &ch : channels_) {
        NotifyRecord(ch, RS_POSTSYNC_CKE_IDX, POST_SYNC_MASK);
    }
    for (auto &ch : channels_) {
        NotifyWait(ch, RS_POSTSYNC_CKE_IDX, POST_SYNC_MASK);
    }
}

void CcuKernelReduceScatterMesh1DWrite::DoGroupWrite()
{
    // 穿刺版本：src = 本 rank 的输入 slice（inputAddr 已在 GeneArgs 中加上 rankId * sliceSize + offset）
    // dst = 本 rank 的输出（reduce 结果写入 outputAddr）
    CcuRep::LocalAddr src = CreateLocalAddr();
    src.addr  = input_;
    src.token = token_;

    CcuRep::LocalAddr dst = CreateLocalAddr();
    dst.addr  = output_;
    dst.token = token_;

    GroupWrite(channels_, rankId_, dst, src, groupOpSize_, dataType_, outputDataType_, reduceOp_);
}

HcclResult CcuKernelReduceScatterMesh1DWrite::Algorithm()
{
    HCCL_INFO("[CcuKernelReduceScatterMesh1DWrite] ReduceScatterMesh1DWrite start");
    CHK_RET(InitResource());
    LoadArgs();
    Presync();
    DoGroupWrite();
    Postsync();
    HCCL_INFO("[CcuKernelReduceScatterMesh1DWrite] ReduceScatterMesh1DWrite end");
    return HcclResult::HCCL_SUCCESS;
}

std::vector<uint64_t> CcuKernelReduceScatterMesh1DWrite::GeneArgs(const CcuTaskArg &arg)
{
    const auto *taskArg   = dynamic_cast<const CcuTaskArgReduceScatterMesh1DWrite *>(&arg);
    uint64_t    inputAddr  = taskArg->inputAddr_;  // 已含 rankId * sliceSize + offset
    uint64_t    outputAddr = taskArg->outputAddr_;
    uint64_t    token      = taskArg->token_;
    uint64_t    sliceSize  = taskArg->sliceSize_;

    auto blockGoSize = CalGoSize(sliceSize);

    HCCL_INFO("[CcuKernelReduceScatterMesh1DWrite] GeneArgs, inputAddr[%llu], outputAddr[%llu], sliceSize[%llu]",
        inputAddr, outputAddr, sliceSize);

    std::vector<uint64_t> argList{inputAddr, outputAddr, token};
    for (auto v : blockGoSize) {
        argList.push_back(v);
    }
    return argList;
}

} // namespace ops_hccl
