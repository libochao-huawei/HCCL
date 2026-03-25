/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "ccu_kernel_all_gather_mesh1d_write.h"

namespace ops_hccl {
using namespace hcomm;

constexpr int AG_POST_SYNC_CKE_IDX_W = 0;
constexpr int AG_PRE_SYNC_CKE_IDX_W  = 1;

CcuKernelAllGatherMesh1DWrite::CcuKernelAllGatherMesh1DWrite(const hcomm::CcuKernelArg &arg)
    : CcuKernelAlgBase(arg)
{
    const auto *kernelArg = dynamic_cast<const CcuKernelArgAllGatherMesh1DWrite *>(&arg);
    rankId_   = kernelArg->rankId_;
    rankSize_ = kernelArg->dimSize_;
    channels_ = kernelArg->channels;
    HCCL_INFO("[CcuKernelAllGatherMesh1DWrite] Init, rankId[%u], rankSize[%lu]", rankId_, rankSize_);
}

HcclResult CcuKernelAllGatherMesh1DWrite::InitResource()
{
    if (channels_.empty()) {
        return HcclResult::HCCL_E_INTERNAL;
    }
    input_ = CreateVariable();
    for (uint64_t i = 0; i < rankSize_; i++) {
        output_.push_back(CreateVariable());
    }
    token_       = CreateVariable();
    groupOpSize_ = CreateGroupOpSize();
    return HcclResult::HCCL_SUCCESS;
}

void CcuKernelAllGatherMesh1DWrite::LoadArgs()
{
    Load(input_);
    for (uint64_t i = 0; i < rankSize_; i++) {
        Load(output_[i]);
    }
    Load(token_);
    Load(groupOpSize_);
}

void CcuKernelAllGatherMesh1DWrite::Presync()
{
    for (auto &ch : channels_) {
        NotifyRecord(ch, AG_PRE_SYNC_CKE_IDX_W, 1);
    }
    for (auto &ch : channels_) {
        NotifyWait(ch, AG_PRE_SYNC_CKE_IDX_W, 1);
    }
}

void CcuKernelAllGatherMesh1DWrite::Postsync()
{
    for (auto &ch : channels_) {
        NotifyRecord(ch, AG_POST_SYNC_CKE_IDX_W, 1);
    }
    for (auto &ch : channels_) {
        NotifyWait(ch, AG_POST_SYNC_CKE_IDX_W, 1);
    }
}

void CcuKernelAllGatherMesh1DWrite::DoGroupBroadcastWrite()
{
    CcuRep::LocalAddr src = CreateLocalAddr();
    src.addr  = input_;
    src.token = token_;

    // 构建 dst 数组：dst[R] = rank R 的输出位置，与 bufs[R] 一一对应
    std::vector<CcuRep::RemoteAddr> dst;
    for (uint64_t i = 0; i < rankSize_; i++) {
        CcuRep::LocalAddr tmp = CreateLocalAddr();
        dst.emplace_back(*reinterpret_cast<CcuRep::RemoteAddr *>(&tmp));
    }

    for (uint64_t R = 0; R < rankSize_; R++) {
        dst[R].addr  = output_[R];
        dst[R].token = token_;
    }

    GroupBroadcastWrite(channels_, rankId_, dst, src, groupOpSize_);
}

HcclResult CcuKernelAllGatherMesh1DWrite::Algorithm()
{
    HCCL_INFO("[CcuKernelAllGatherMesh1DWrite] AllGatherMesh1DWrite start");
    CHK_RET(InitResource());
    LoadArgs();
    Presync();
    DoGroupBroadcastWrite();
    Postsync();
    HCCL_INFO("[CcuKernelAllGatherMesh1DWrite] AllGatherMesh1DWrite end");
    return HcclResult::HCCL_SUCCESS;
}

std::vector<uint64_t> CcuKernelAllGatherMesh1DWrite::GeneArgs(const hcomm::CcuTaskArg &arg)
{
    const auto *taskArg = dynamic_cast<const CcuTaskArgAllGatherMesh1DWrite *>(&arg);
    uint64_t inputAddr  = taskArg->inputAddr_;
    uint64_t outputAddr = taskArg->outputAddr_;
    uint64_t token      = taskArg->token_;
    uint64_t sliceSize  = taskArg->sliceSize_;

    auto blockGoSize = CalGoSize(sliceSize);

    HCCL_INFO("[CcuKernelAllGatherMesh1DWrite] GeneArgs, inputAddr[%llu], outputAddr[%llu], sliceSize[%llu]",
        inputAddr, outputAddr, sliceSize);

    std::vector<uint64_t> argList{inputAddr};
    // N 个输出位置：output_[R] = outputAddr + R * sliceSize
    for (uint64_t R = 0; R < rankSize_; R++) {
        argList.push_back(outputAddr + R * sliceSize);
    }
    argList.push_back(token);
    for (auto v : blockGoSize) {
        argList.push_back(v);
    }

    HCCL_INFO("[CcuKernelAllGatherMesh1DWrite] GeneArgs end");
    return argList;
}

} // namespace ops_hccl
