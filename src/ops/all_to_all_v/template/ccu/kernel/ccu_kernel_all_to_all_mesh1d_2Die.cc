/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "ccu_kernel_all_to_all_mesh1d_2Die.h"
#include "ccu_kernel_alg_base.h"

namespace Hccl {

constexpr int CKE_IDX_0   = 0;
constexpr int CKE_IDX_1   = 1;
constexpr int CKE_IDX_2   = 2;
constexpr int INPUT_XN_ID = 0;
constexpr int OUPUT_XN_ID = 1;
constexpr int TOKEN_XN_ID = 2;
constexpr int POST_SYNC_ID = 3;

constexpr uint64_t CCU_MS_SIZE   = 4096;
constexpr uint64_t LOCAL_COPY_MS = 8;

CcuKernelAllToAllMesh1D2Die::CcuKernelAllToAllMesh1D2Die(const CcuKernelArg &arg)
    : CcuKernelAlgBase(arg)
{
    const CcuKernelArgAllToAllMesh1D2Die *kernelArg = dynamic_cast<const CcuKernelArgAllToAllMesh1D2Die *>(&arg);
    if (kernelArg == nullptr) {
        THROW<NullPtrException>(StringFormat("CcuKernelAllToAllMesh1D2Die::ctxArg ptr is null"));
    }

    channels_ = kernelArg->channels;

    rankId_     = kernelArg->rankId_;
    withMyRank_ = kernelArg->withMyRank_;
    rankGroup_  = kernelArg->rankGroup;
    // if (kernelArg->dimSize_.size() > 0) {
    //     rankSize_ = kernelArg->dimSize_[0];
    // }
    rankSize_ = kernelArg->rankSize_;
    bitNumPerCKE_ = kernelArg->bitNum_;
}

void CcuKernelAllToAllMesh1D2Die::InitResource()
{
    // 创建Variable，用于交换地址及token
    if (channels_.size() == 0) {
        HCCL_ERROR("[CcuKernelAllToAllMesh2Die] RankId[%u] channels_ is empty", rankId_);
    }
    virRankSize = channels_.size() + 1;

    for (u64 id = 0; id < channels_.size(); id++) {
        // 非本地，使用远端Variable
        HCCL_DEBUG("[CcuKernelAllToAllMesh2Die] RankId[%u], Id[%u]", rankId_, id);
        hcomm::CcuRep::Variable input, output, token;
        CHK_RET(CreateVariable(channels_[id], CKE_IDX_0, &input));
        input_.emplace_back(input);
        CHK_RET(CreateVariable(channels_[id], CKE_IDX_1, &output));
        output_.emplace_back(output);
        CHK_RET(CreateVariable(channels_[id], CKE_IDX_2, &token));
        token_.emplace_back(output);
    }
    // 最后一个位置放自己地址
    input_.push_back(CreateVariable());
    output_.push_back(CreateVariable());
    token_.push_back(CreateVariable());

    sliceSize_         = CreateVariable();
    inputSliceStride_  = CreateVariable();
    outputoffset_ = CreateVariable();
    outBuffBaseOff_    = CreateVariable();
    groupOpSize_       = CreateGroupOpSize();

    logicRankSize = withMyRank_ ? channels_.size() + 1 : channels_.size();
    signalNum_    = (rankSize_ + bitNumPerCKE_ - 1) / bitNumPerCKE_;
    event_ = CreateCompletedEvent();
    HCCL_INFO("[CcuKernelAlltoAll2Die] KernelArgs: rankId_[%u], rankSize_[%u], signalNum_[%u]", rankId_, rankSize_, signalNum_);
    return;
}

void CcuKernelAllToAllMesh1D2Die::LoadArgs()
{
    // 从SQE load args，本rank需要的input、output地址等信息
    // inputAddr, outputAddr, tokenInfo, srcStride, srcOffset, dstOffset, groupOpSize
    Load(input_[virRankSize - 1]);
    Load(output_[virRankSize - 1]);
    Load(token_[virRankSize - 1]);
    Load(sliceSize_); // 本轮传输的分片大小
    Load(inputSliceStride_);
    Load(outputoffset_);
    Load(outBuffBaseOff_);
    Load(groupOpSize_);
    return;
}

void CcuKernelAllToAllMesh1D2Die::PreSync()
{
    for (ChannelHandle channel : channels_) {
        NotifyRecord(channel, CKE_IDX_0, OUTPUT_XN_ID, output_[virRankSize - 1], 1 << OUTPUT_XN_ID);
        NotifyRecord(channel, CKE_IDX_0, TOKEN_XN_ID, token_[virRankSize - 1], 1 << TOKEN_XN_ID);
    }

    uint32_t waitBits = (1 << OUTPUT_XN_ID) | (1 << TOKEN_XN_ID);
    for (const ChannelHandle &channel : channels_) {
        NotifyWait(channel, CKE_IDX_0, waitBits);
    }
    return;
}

void CcuKernelAllToAllMesh1D2Die::PostSync()
{
    for (const auto &channel : channels_) {
        NotifyRecord(channel, CKE_IDX_0, 1 << POST_SYNC_ID);
    }
    for (const auto &channel : channels_) {
        NotifyWait(channel, CKE_IDX_0, 1 << POST_SYNC_ID);
    }
    
    return;
}

uint32_t CcuKernelAllToAllMesh1D2Die::CalcDstRank(uint32_t peerId) const
{
    if (peerId > rankGroup_.size()) {
        HCCL_ERROR("[CcuKernelAllToAllMesh1D2Die][CalcDstRank] Unexpected peerId[%u]", peerId);
    }
    return rankGroup_[peerId];
}

void CcuKernelAllToAllMesh1D2Die::DoRepeatAllToAll()
{
    // 创建GSA， src为本地的各片HBM地址GSA列表，dst为所有对端的HBM地址GSA列表
    std::vector<hcomm::CcuRep::LocalAddr> src;
    for (uint64_t rankIdx = 0; rankIdx < logicRankSize; rankIdx++) {
        src.emplace_back(CreateLocalAddr());
    }
    std::vector<hcomm::CcuRep::RemoteAddr> dst;
    for (uint64_t rankIdx = 0; rankIdx < logicRankSize; rankIdx++) {
        dst.emplace_back(CreateRemoteAddr());
    }

    // 考虑stride信息
    for (uint64_t r = 0; r < logicRankSize; r++) {
        const u32 dstRank = CalcDstRank(r);

        src[r].token = token_[r];
        dst[r].token = token_[r];

        src[r].addr = input_[virRankSize - 1];
        dst[r].addr = output_[r];
        dst[r].addr += outputoffset_;
        for(uint64_t i = 0; i < dstRank; i++){
            src[r].addr += inputSliceStride_;
        }
    }

    hcomm::CcuRep::LocalAddr localSrc_ = CreateLocalAddr(); // for LocalCopy
    hcomm::CcuRep::LocalAddr localDst_ = CreateLocalAddr();
    if(withMyRank_){
        const u32 with_dstRank = CalcDstRank(logicRankSize - 1);

        localSrc_.token = token_[logicRankSize - 1];
        localSrc_.addr = input_[logicRankSize - 1];

        localDst_.token = token_[logicRankSize - 1];
        localDst_.addr = output_[logicRankSize - 1];
        localDst_.addr += outputoffset_;
        for(uint64_t i = 0; i < with_dstRank; i++){
            localSrc_.addr += inputSliceStride_;
        }
    }

    //  all2all 数据搬运
    u32 channelsIdx = 0;
    uint64_t allBit_ = withMyRank_ ? ((1 << logicRankSize) - 1) & (~(1 << transports.size())) : (1 << logicRankSize) - 1;
    if (withMyRank_) {
        for (uint64_t r = 0; r < logicRankSize; r++) {
            if (withMyRank_ && r == logicRankSize - 1) {
                GroupCopy(localDst_, localSrc_, groupOpSize_);
                continue;
            }
            event_.SetMask(1 << r);
            WriteNb(channels_[channelsIdx], dst[r], src[r], sliceSize_, event_);
            channelsIdx++;
        }
        event_.SetMask(allBit_);
        WaitEvent(event_);
    } else {
        uint16_t signalNum = logicRankSize / bitNumPerCKE_;
        std::vector<uint16_t> waitBitVector(signalNum, 0);
        for (uint16_t r = 0; r < logicRankSize; r++) {
            event_.SetMask(1 << r);
            WriteNb(channels_[channelsIdx], dst[r], src[r], sliceSize_, event_);
            channelsIdx++;
        }
        event_.SetMask(allBit_);
        WaitEvent(event_);
    }
}

void CcuKernelAllToAllMesh1D2Die::Algorithm()
{
    HCCL_INFO("[ccuAllToAllMesh1D2Die_context] AllToAllMesh1D2Die run.");
    InitResource();

    LoadArgs();

    PreSync();

    DoRepeatAllToAll();

    PostSync();
    HCCL_INFO("[ccuAllToAllMesh1D2Die_context] AllToAllMesh1D2Die end.");
    return;
}

std::vector<uint64_t> CcuKernelAllToAllMesh1D2Die::GeneArgs(const CcuTaskArg &arg)
{
    const CcuTaskArgAllToAllMesh1D2Die *taskArg = dynamic_cast<const CcuTaskArgAllToAllMesh1D2Die *>(&arg);
    if (taskArg == nullptr) {
        THROW<NullPtrException>(StringFormat("CcuKernelAllToAllMesh1D2Die::taskArg ptr is null"));
    }
    uint64_t inputAddr         = taskArg->inputAddr_;
    uint64_t outputAddr        = taskArg->outputAddr_;
    uint64_t tokenInfo         = taskArg->token_;
    uint64_t sliceSize         = taskArg->sliceSize_;
    uint64_t inputSliceStride  = taskArg->inputSliceStride_;
    uint64_t outputSliceStride = taskArg->outputSliceStride_ * rankId_;
    uint64_t outBuffBaseOff    = taskArg->outBuffBaseOff_;

    auto goSize = CalGoSize(sliceSize);
    HCCL_INFO("[CcuKernelAllToAllMesh1D2Die] inputAddr[%llu], outputAddr[%llu], sliceSize[%llu], "
              "inputSliceStride[%llu], outputSliceStride[%llu], outBuffBaseOff[%llu].",
              inputAddr, outputAddr, sliceSize, inputSliceStride, outputSliceStride, outBuffBaseOff);

    return {inputAddr,      outputAddr, tokenInfo, sliceSize, inputSliceStride, outputSliceStride,
            outBuffBaseOff, goSize[0],  goSize[1], goSize[2], goSize[3]};
}

} // namespace Hccl
