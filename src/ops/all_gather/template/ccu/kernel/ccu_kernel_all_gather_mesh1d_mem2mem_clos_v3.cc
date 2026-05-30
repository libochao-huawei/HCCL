/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "ccu_kernel_all_gather_mesh1d_mem2mem_clos_v3.h"
#include "ccu_kernel_alg_base.h"

namespace ops_hccl {
using namespace hcomm;

constexpr int OUTPUT_XN_ID = 1;
constexpr int TOKEN_XN_ID = 2;
constexpr int CKE_IDX_0 = 0;
constexpr int CKE_IDX_1 = 1;
constexpr uint64_t CCU_MS_SIZE = 4096;
constexpr uint64_t LOCAL_COPY_MS = 8;
constexpr int POST_SYNC_ID = 3;
constexpr uint16_t BIT_NUM_PER_CKE = 16;
constexpr uint32_t INVALID_CHANNEL_IDX = static_cast<uint32_t>(-1);

CcuKernelAllGatherMesh1DMem2MemClosV3::CcuKernelAllGatherMesh1DMem2MemClosV3(const CcuKernelArg &arg)
    : CcuKernelAlgBase(arg)
{
    const CcuKernelArgAllGatherMesh1DMem2MemClosV3 *kernelArg
        = dynamic_cast<const CcuKernelArgAllGatherMesh1DMem2MemClosV3 *>(&arg);
    rankId_                 = kernelArg->rankId_;
    rankSize_               = kernelArg->dimSize_;
    channels_               = kernelArg->channels;
    mainChannelIdxByRank_   = kernelArg->mainChannelIdxByRank_;
    sharedChannelIdxByRank_ = kernelArg->sharedChannelIdxByRank_;

    HCCL_INFO(
        "[CcuKernelAllGatherMesh1DMem2MemClosV3] Init, KernelArgs are rankId[%u], rankSize_[%u]",
        rankId_, rankSize_);
}

bool CcuKernelAllGatherMesh1DMem2MemClosV3::HasSharedChannel(uint32_t peerId) const
{
    uint32_t lowRank = std::min(rankId_, peerId);
    uint32_t highRank = std::max(rankId_, peerId);
    uint32_t rankDiff = highRank - lowRank;
    return highRank < 16 && lowRank % 4 == highRank % 4 && rankDiff != 0 && rankDiff % 4 == 0 &&
           rankDiff <= 12;
}

HcclResult CcuKernelAllGatherMesh1DMem2MemClosV3::InitChannelIdxByRank()
{
    CHK_PRT_RET(channels_.empty(), HCCL_ERROR("[CcuKernelAllGatherMesh1DMem2MemClosV3] channels is empty!"),
                HcclResult::HCCL_E_INTERNAL);

    if (mainChannelIdxByRank_.empty()) {
        mainChannelIdxByRank_.assign(rankSize_, INVALID_CHANNEL_IDX);
        sharedChannelIdxByRank_.assign(rankSize_, INVALID_CHANNEL_IDX);
        uint32_t channelIdx = 0;
        uint32_t expectedMainChannelNum = rankSize_ > 0 ? static_cast<uint32_t>(rankSize_ - 1) : 0;
        uint32_t remainingSharedChannelNum = channels_.size() > expectedMainChannelNum ?
            static_cast<uint32_t>(channels_.size() - expectedMainChannelNum) : 0;
        for (uint32_t peerId = 0; peerId < rankSize_; peerId++) {
            if (peerId == rankId_) {
                continue;
            }
            CHK_PRT_RET(channelIdx >= channels_.size(),
                        HCCL_ERROR("[CcuKernelAllGatherMesh1DMem2MemClosV3] channelIdx[%u] exceeds channel size[%zu] "
                                   "when init main channel for peer[%u].",
                                   channelIdx, channels_.size(), peerId),
                        HcclResult::HCCL_E_INTERNAL);
            mainChannelIdxByRank_[peerId] = channelIdx++;
            if (HasSharedChannel(peerId) && remainingSharedChannelNum > 0 && channelIdx < channels_.size()) {
                sharedChannelIdxByRank_[peerId] = channelIdx++;
                remainingSharedChannelNum--;
            }
        }
    }

    CHK_PRT_RET(mainChannelIdxByRank_.size() != rankSize_,
                HCCL_ERROR("[CcuKernelAllGatherMesh1DMem2MemClosV3] mainChannelIdxByRank size[%zu] is not "
                           "rankSize[%llu].",
                           mainChannelIdxByRank_.size(), rankSize_),
                HcclResult::HCCL_E_INTERNAL);
    if (sharedChannelIdxByRank_.empty()) {
        sharedChannelIdxByRank_.assign(rankSize_, INVALID_CHANNEL_IDX);
    }
    CHK_PRT_RET(sharedChannelIdxByRank_.size() != rankSize_,
                HCCL_ERROR("[CcuKernelAllGatherMesh1DMem2MemClosV3] sharedChannelIdxByRank size[%zu] is not "
                           "rankSize[%llu].",
                           sharedChannelIdxByRank_.size(), rankSize_),
                HcclResult::HCCL_E_INTERNAL);
    return HcclResult::HCCL_SUCCESS;
}

HcclResult CcuKernelAllGatherMesh1DMem2MemClosV3::InitResource()
{
    localInput_ = CreateVariable();
    CHK_RET(InitChannelIdxByRank());

    for (uint64_t peerId = 0; peerId < rankSize_; peerId++) {
        if (peerId == rankId_) {
            output_.push_back(CreateVariable());
            token_.push_back(CreateVariable());
            sharedOutput_.push_back(CreateVariable());
            sharedToken_.push_back(CreateVariable());
            continue;
        }

        uint32_t mainChannelIdx = mainChannelIdxByRank_[peerId];
        CHK_PRT_RET(mainChannelIdx >= channels_.size(),
                    HCCL_ERROR("[CcuKernelAllGatherMesh1DMem2MemClosV3] invalid main channel idx[%u] for "
                               "peer[%llu], channel size[%zu].",
                               mainChannelIdx, peerId, channels_.size()),
                    HcclResult::HCCL_E_INTERNAL);
        HCCL_DEBUG("[CcuKernelAllGatherMesh1DMem2MemClosV3] MyRank[%u], PeerId[%llu], MainChannelId[%u]",
                   rankId_, peerId, mainChannelIdx);
        CcuRep::Variable outputVar, tokenVar;
        CHK_RET(CreateVariable(channels_[mainChannelIdx], OUTPUT_XN_ID, &outputVar));
        output_.push_back(outputVar);
        CHK_RET(CreateVariable(channels_[mainChannelIdx], TOKEN_XN_ID, &tokenVar));
        token_.push_back(tokenVar);

        uint32_t sharedChannelIdx = sharedChannelIdxByRank_[peerId];
        if (sharedChannelIdx < channels_.size()) {
            HCCL_DEBUG("[CcuKernelAllGatherMesh1DMem2MemClosV3] MyRank[%u], PeerId[%llu], SharedChannelId[%u]",
                       rankId_, peerId, sharedChannelIdx);
            CcuRep::Variable sharedOutputVar, sharedTokenVar;
            CHK_RET(CreateVariable(channels_[sharedChannelIdx], OUTPUT_XN_ID, &sharedOutputVar));
            sharedOutput_.push_back(sharedOutputVar);
            CHK_RET(CreateVariable(channels_[sharedChannelIdx], TOKEN_XN_ID, &sharedTokenVar));
            sharedToken_.push_back(sharedTokenVar);
        } else {
            sharedOutput_.push_back(CreateVariable());
            sharedToken_.push_back(CreateVariable());
        }
    }

    currentRankSliceInputOffset_  = CreateVariable();
    currentRankSliceOutputOffset_ = CreateVariable();
    inputRepeatStride_            = CreateVariable();
    outputRepeatStride_           = CreateVariable();
    tmpRepeatNum_                 = CreateVariable();
    normalSliceSize_              = CreateVariable();
    lastSliceSize_                = CreateVariable();
    mainSliceSize_                = CreateVariable();
    sharedSliceSize_              = CreateVariable();
    constVar1_                    = CreateVariable();
    constVar1_                    = 1;
    repeatTimeflag_               = CreateVariable();
    repeatTimeflag_               = 0;
    isInputOutputEqual_           = CreateVariable();
    localGoSize_                  = CreateGroupOpSize();

    src = CreateLocalAddr();
    src_loccopy = CreateLocalAddr();
    remote_src = CreateLocalAddr();
    shared_src = CreateLocalAddr();

    for (uint64_t rankIdx = 0; rankIdx < rankSize_; rankIdx++) {
        if (rankIdx == rankId_) {
            dst.push_back({});
            sharedDst.push_back({});
        } else {
            dst.push_back(CreateRemoteAddr());
            if (sharedChannelIdxByRank_[rankIdx] < channels_.size()) {
                sharedDst.push_back(CreateRemoteAddr());
            } else {
                sharedDst.push_back({});
            }
        }
    }

    uint32_t eventNum = (rankSize_ + BIT_NUM_PER_CKE - 1) / BIT_NUM_PER_CKE;
    sharedEventMask_.assign(eventNum, 0);
    for (uint32_t i = 0; i < eventNum; i++) {
        event_.push_back(CreateCompletedEvent());
        sharedEvent_.push_back(CreateCompletedEvent());
    }
    for (uint64_t rankIdx = 0; rankIdx < rankSize_; rankIdx++) {
        if (rankIdx != rankId_ && sharedChannelIdxByRank_[rankIdx] < channels_.size()) {
            sharedEventMask_[rankIdx / BIT_NUM_PER_CKE] |= 1 << (rankIdx % BIT_NUM_PER_CKE);
        }
    }
    return HcclResult::HCCL_SUCCESS;
}

void CcuKernelAllGatherMesh1DMem2MemClosV3::LoadArgs()
{
    Load(localInput_);
    Load(output_[rankId_]);
    Load(token_[rankId_]);
    Load(currentRankSliceInputOffset_);
    Load(currentRankSliceOutputOffset_);
    Load(tmpRepeatNum_);
    Load(inputRepeatStride_);
    Load(outputRepeatStride_);
    Load(normalSliceSize_);
    Load(lastSliceSize_);
    Load(mainSliceSize_);
    Load(sharedSliceSize_);
    Load(isInputOutputEqual_);
    Load(localGoSize_);
}

void CcuKernelAllGatherMesh1DMem2MemClosV3::PreSync()
{
    for (ChannelHandle channel : channels_) {
        NotifyRecord(channel, CKE_IDX_0, OUTPUT_XN_ID, output_[rankId_], 1 << OUTPUT_XN_ID);
        NotifyRecord(channel, CKE_IDX_0, TOKEN_XN_ID, token_[rankId_], 1 << TOKEN_XN_ID);
    }

    uint16_t allBit = 1 << OUTPUT_XN_ID | 1 << TOKEN_XN_ID;
    for (ChannelHandle channel : channels_) {
        NotifyWait(channel, CKE_IDX_0, allBit);
    }
}

void CcuKernelAllGatherMesh1DMem2MemClosV3::PostSync()
{
    for (ChannelHandle channel : channels_) {
        NotifyRecord(channel, CKE_IDX_0, 1 << POST_SYNC_ID);
    }
    for (ChannelHandle channel : channels_) {
        NotifyWait(channel, CKE_IDX_0, 1 << POST_SYNC_ID);
    }
}

void CcuKernelAllGatherMesh1DMem2MemClosV3::DoAllGather()
{
    for (uint64_t rankIdx = 0; rankIdx < rankSize_; rankIdx++) {
        uint32_t eventIdx = rankIdx / BIT_NUM_PER_CKE;
        event_[eventIdx].SetMask(1 << (rankIdx % BIT_NUM_PER_CKE));
        if (rankIdx == rankId_) {
            RecordEvent(event_[eventIdx]);
            continue;
        }

        uint32_t mainChannelIdx = mainChannelIdxByRank_[rankIdx];
        uint32_t sharedChannelIdx = sharedChannelIdxByRank_[rankIdx];
        bool hasSharedChannel = sharedChannelIdx < channels_.size();
        if (hasSharedChannel) {
            CCU_IF(sharedSliceSize_ != 0)
            {
                WriteNb(channels_[mainChannelIdx], dst[rankIdx], src, mainSliceSize_, event_[eventIdx]);
                sharedEvent_[eventIdx].SetMask(1 << (rankIdx % BIT_NUM_PER_CKE));
                WriteNb(channels_[sharedChannelIdx], sharedDst[rankIdx], shared_src, sharedSliceSize_,
                        sharedEvent_[eventIdx]);
            }
            CCU_IF(sharedSliceSize_ == 0)
            {
                WriteNb(channels_[mainChannelIdx], dst[rankIdx], src, normalSliceSize_, event_[eventIdx]);
            }
        } else {
            WriteNb(channels_[mainChannelIdx], dst[rankIdx], src, normalSliceSize_, event_[eventIdx]);
        }
    }

    CCU_IF(isInputOutputEqual_ == 0)
    {
        GroupCopy(remote_src, src_loccopy, localGoSize_);
    }

    for (uint32_t i = 0; i < event_.size(); i++) {
        if (i == event_.size() - 1) {
            if (rankSize_ % BIT_NUM_PER_CKE == 0) {
                event_[i].SetMask((1 << BIT_NUM_PER_CKE) - 1);
            } else {
                event_[i].SetMask((1 << (rankSize_ % BIT_NUM_PER_CKE)) - 1);
            }
        } else {
            event_[i].SetMask((1 << BIT_NUM_PER_CKE) - 1);
        }
        WaitEvent(event_[i]);
    }

    CCU_IF(sharedSliceSize_ != 0)
    {
        for (uint32_t i = 0; i < sharedEvent_.size(); i++) {
            if (sharedEventMask_[i] != 0) {
                sharedEvent_[i].SetMask(sharedEventMask_[i]);
                WaitEvent(sharedEvent_[i]);
            }
        }
    }
}

void CcuKernelAllGatherMesh1DMem2MemClosV3::DoRepeatAllGather()
{
    src.addr = localInput_;
    src.addr += currentRankSliceInputOffset_;
    src.token = token_[rankId_];

    src_loccopy.addr = localInput_;
    src_loccopy.addr += currentRankSliceInputOffset_;
    src_loccopy.token = token_[rankId_];

    shared_src.addr = localInput_;
    shared_src.addr += currentRankSliceInputOffset_;
    shared_src.addr += mainSliceSize_;
    shared_src.token = token_[rankId_];

    for (uint32_t rankIdx = 0; rankIdx < rankSize_; rankIdx++) {
        if (rankIdx == rankId_) {
            remote_src.addr = output_[rankId_];
            remote_src.addr += currentRankSliceOutputOffset_;
            remote_src.token = token_[rankId_];
        } else {
            dst[rankIdx].addr = output_[rankIdx];
            dst[rankIdx].addr += currentRankSliceOutputOffset_;
            dst[rankIdx].token = token_[rankIdx];
            if (sharedChannelIdxByRank_[rankIdx] < channels_.size()) {
                sharedDst[rankIdx].addr = sharedOutput_[rankIdx];
                sharedDst[rankIdx].addr += currentRankSliceOutputOffset_;
                sharedDst[rankIdx].addr += mainSliceSize_;
                sharedDst[rankIdx].token = sharedToken_[rankIdx];
            }
        }
    }
    CCU_WHILE(tmpRepeatNum_ != UINT64_MAX)
    {
        tmpRepeatNum_ += constVar1_;
        CCU_IF(repeatTimeflag_ != 0)
        {
            src.addr += inputRepeatStride_;
            shared_src.addr += inputRepeatStride_;
            for (uint32_t rankIdx = 0; rankIdx < rankSize_; rankIdx++) {
                if (rankIdx == rankId_) {
                    remote_src.addr += outputRepeatStride_;
                } else {
                    dst[rankIdx].addr += outputRepeatStride_;
                    if (sharedChannelIdxByRank_[rankIdx] < channels_.size()) {
                        sharedDst[rankIdx].addr += outputRepeatStride_;
                    }
                }
            }
        }
        CCU_IF(normalSliceSize_ != 0)
        {
            DoAllGather();
        }
        repeatTimeflag_ = 1;
    }
}

HcclResult CcuKernelAllGatherMesh1DMem2MemClosV3::Algorithm()
{
    HCCL_INFO("[CcuKernelAllGatherMesh1DMem2MemClosV3] AllgatherMesh1D run.");
    CHK_RET(InitResource());
    LoadArgs();
    PreSync();
    DoRepeatAllGather();
    PostSync();
    HCCL_INFO("[CcuKernelAllGatherMesh1DMem2MemClosV3] AllgatherMesh1D end.");
    return HcclResult::HCCL_SUCCESS;
}

std::vector<uint64_t> CcuKernelAllGatherMesh1DMem2MemClosV3::GeneArgs(const CcuTaskArg &arg)
{
    const CcuTaskArgAllGatherMesh1DMem2MemClosV3 *taskArg
        = dynamic_cast<const CcuTaskArgAllGatherMesh1DMem2MemClosV3 *>(&arg);
    uint64_t inputAddr                    = taskArg->inputAddr_;
    uint64_t outputAddr                   = taskArg->outputAddr_;
    uint64_t token                        = taskArg->token_;

    uint64_t currentRankSliceInputOffset  = taskArg->inputSliceStride_ * rankId_;
    uint64_t currentRankSliceOutputOffset = taskArg->outputSliceStride_ * rankId_;
    uint64_t tmpRepeatNum                 = UINT64_MAX - taskArg->repeatNum_;
    uint64_t inputRepeatStride            = taskArg->inputRepeatStride_;
    uint64_t outputRepeatStride           = taskArg->outputRepeatStride_;
    uint64_t normalSliceSize              = taskArg->normalSliceSize_;
    uint64_t lastSliceSize                = taskArg->lastSliceSize_;
    uint64_t isInputOutputEqual           = taskArg->isInputOutputEqual_;
    uint64_t mainSliceSize                = taskArg->mainSliceSize_;
    uint64_t sharedSliceSize              = taskArg->sharedSliceSize_;

    auto goSize = CalGoSize(normalSliceSize);

    std::vector<uint64_t> taskArgs = {inputAddr,
                                      outputAddr,
                                      token,
                                      currentRankSliceInputOffset,
                                      currentRankSliceOutputOffset,
                                      tmpRepeatNum,
                                      inputRepeatStride,
                                      outputRepeatStride,
                                      normalSliceSize,
                                      lastSliceSize,
                                      mainSliceSize,
                                      sharedSliceSize,
                                      isInputOutputEqual,
                                      goSize[0],
                                      goSize[1],
                                      goSize[2],
                                      goSize[3]};

    HCCL_INFO("[CcuKernelAllGatherMesh1DMem2MemClosV3] TaskArgs: inputAddr[%llu], outputAddr[%llu], "
        "currentRankSliceInputOffset[%llu], currentRankSliceOutputOffset[%llu], "
        "repeatNum[%llu], inputRepeatStride[%llu], outputRepeatStride[%llu], normalSliceSize[%llu], "
        "lastSliceSize[%llu], mainSliceSize[%llu], sharedSliceSize[%llu]",
        inputAddr, outputAddr, currentRankSliceInputOffset, currentRankSliceOutputOffset, tmpRepeatNum,
        inputRepeatStride, outputRepeatStride, normalSliceSize, lastSliceSize, mainSliceSize, sharedSliceSize);
    return taskArgs;
}

} // namespace ops_hccl
