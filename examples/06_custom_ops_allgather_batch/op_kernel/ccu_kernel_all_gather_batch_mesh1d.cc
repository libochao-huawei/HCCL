/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "ccu_kernel_all_gather_batch_mesh1d.h"

namespace ops_hccl_allgather_batch {
using namespace hcomm;

namespace {
constexpr int OUTPUT_XN_ID = 1;
constexpr int TOKEN_XN_ID = 2;
constexpr int CKE_IDX_0 = 0;
constexpr int POST_SYNC_ID = 3;
} // namespace

CcuKernelAllGatherBatchMesh1D::CcuKernelAllGatherBatchMesh1D(const CcuKernelArg &arg)
    : ops_hccl::CcuKernelAlgBase(arg)
{
    const auto *kernelArg = dynamic_cast<const CcuKernelArgAllGatherBatchMesh1D *>(&arg);
    rankSize_ = kernelArg->rankSize_;
    rankId_ = kernelArg->rankId_;
    itemCount_ = kernelArg->itemCount_;
    channels_ = kernelArg->channels;
}

HcclResult CcuKernelAllGatherBatchMesh1D::InitResource()
{
    CHK_PRT_RET(rankSize_ > 1 && channels_.empty(),
                HCCL_ERROR("[CcuKernelAllGatherBatchMesh1D] channels is empty"),
                HCCL_E_INTERNAL);

    items_.reserve(itemCount_);
    for (uint32_t i = 0; i < itemCount_; ++i) {
        KernelAllGatherBatchItem item;
        item.input = CreateVariable();
        item.output.reserve(rankSize_);
        item.token.reserve(rankSize_);

        uint16_t channelIdx = 0;
        for (uint64_t peerId = 0; peerId < rankSize_; ++peerId) {
            if (peerId == rankId_) {
                item.output.push_back(CreateVariable());
                item.token.push_back(CreateVariable());
                continue;
            }

            hcomm::CcuRep::Variable outputVar;
            hcomm::CcuRep::Variable tokenVar;
            CHK_RET(CreateVariable(channels_[channelIdx], OUTPUT_XN_ID, &outputVar));
            CHK_RET(CreateVariable(channels_[channelIdx], TOKEN_XN_ID, &tokenVar));
            item.output.push_back(outputVar);
            item.token.push_back(tokenVar);
            ++channelIdx;
        }
        item.offset = CreateVariable();
        item.groupOpSize = CreateGroupOpSize();
        items_.push_back(item);
    }

    src_ = CreateLocalAddr();
    dst_.reserve(rankSize_);
    for (uint64_t i = 0; i < rankSize_; ++i) {
        dst_.push_back(CreateRemoteAddr());
    }
    return HCCL_SUCCESS;
}

void CcuKernelAllGatherBatchMesh1D::LoadArgs()
{
    for (auto &item : items_) {
        Load(item.input);
        Load(item.output[rankId_]);
        Load(item.token[rankId_]);
        Load(item.offset);
        Load(item.groupOpSize);
    }
}

void CcuKernelAllGatherBatchMesh1D::PreSync(const KernelAllGatherBatchItem &item)
{
    for (ChannelHandle channel : channels_) {
        NotifyRecord(channel, CKE_IDX_0, OUTPUT_XN_ID, item.output[rankId_], 1 << OUTPUT_XN_ID);
        NotifyRecord(channel, CKE_IDX_0, TOKEN_XN_ID, item.token[rankId_], 1 << TOKEN_XN_ID);
    }
    const uint16_t allBit = (1 << OUTPUT_XN_ID) | (1 << TOKEN_XN_ID);
    for (ChannelHandle channel : channels_) {
        NotifyWait(channel, CKE_IDX_0, allBit);
    }
}

void CcuKernelAllGatherBatchMesh1D::DoAllGather(const KernelAllGatherBatchItem &item)
{
    src_.addr = item.input;
    src_.token = item.token[rankId_];
    uint32_t dstId = 0;
    for (uint64_t rankIdx = 0; rankIdx < rankSize_; ++rankIdx) {
        const uint32_t curId = (rankIdx == rankId_) ? (rankSize_ - 1) : dstId++;
        dst_[curId].addr = item.output[rankIdx];
        dst_[curId].addr += item.offset;
        dst_[curId].token = item.token[rankIdx];
    }
    GroupBroadcast(channels_, dst_, src_, item.groupOpSize);
}

void CcuKernelAllGatherBatchMesh1D::PostSync()
{
    for (ChannelHandle channel : channels_) {
        NotifyRecord(channel, CKE_IDX_0, 1 << POST_SYNC_ID);
    }
    for (ChannelHandle channel : channels_) {
        NotifyWait(channel, CKE_IDX_0, 1 << POST_SYNC_ID);
    }
}

HcclResult CcuKernelAllGatherBatchMesh1D::Algorithm()
{
    CHK_RET(InitResource());
    LoadArgs();
    for (const auto &item : items_) {
        PreSync(item);
        DoAllGather(item);
    }
    PostSync();
    return HCCL_SUCCESS;
}

std::vector<uint64_t> CcuKernelAllGatherBatchMesh1D::GeneArgs(const CcuTaskArg &arg)
{
    const auto *taskArg = dynamic_cast<const CcuTaskArgAllGatherBatchMesh1D *>(&arg);
    if (taskArg == nullptr || taskArg->itemCount_ != itemCount_) {
        HCCL_ERROR("[CcuKernelAllGatherBatchMesh1D] itemCount mismatch or task arg invalid");
        return {};
    }

    std::vector<uint64_t> args;
    args.reserve(itemCount_ * 8);
    for (uint32_t i = 0; i < itemCount_; ++i) {
        const auto &item = taskArg->items_[i];
        auto goSize = CalGoSize(item.sliceSize);
        args.push_back(item.inputAddr);
        args.push_back(item.outputAddr);
        args.push_back(item.token);
        args.push_back(item.offset);
        args.insert(args.end(), goSize.begin(), goSize.end());
    }
    return args;
}

} // namespace ops_hccl_allgather_batch
