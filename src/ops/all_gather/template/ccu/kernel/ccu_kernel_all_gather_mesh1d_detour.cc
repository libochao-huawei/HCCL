/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "ccu_kernel_all_gather_mesh1d_detour.h"
#include "ccu_instruction_all_gather_mesh1d_detour.h"
#include "ccu_assist.h"

namespace Hccl {

constexpr int OUTPUT_XN_ID = 1;
constexpr int TOKEN_XN_ID = 2;
constexpr int CKE_IDX_0 = 0;
constexpr int CKE_IDX_1 = 1;
constexpr int CKE_IDX_2 = 2;
constexpr int CKE_IDX_3 = 3;

void CcuKernelAllGatherMeshDetour1D::ProcessChannels(const std::vector<CcuChannel *> &channels)
{
    // 构建detourChannel
    if (channels.size() % (rankSize_ - 1) != 0) {
        THROW<InvalidParamsException>(StringFormat(
            "Invalid ChannelsNum[%u] for rankSize_[%u]", channels.size(), rankSize_));
    }

    for (uint64_t i = 0; i < pathNumPerPeer_; i++) {
        // 到每个对端有pathNum个channel，故detourChannel中共有pathNum组
        detourChannels_.emplace_back(std::vector<CcuChannel*>());
    }
    uint64_t directPathNum = pathNumPerPeer_ - detourPathNum_;
    for (uint64_t i = 0; i < directPathNum; i++) {
        // 有pathNum-detourPathNum组的直连链路，每组重复
        for (uint64_t j = 0; j < rankSize_ - 1; j++) {
            detourChannels_[i].emplace_back(channels[j]);
        }
        HCCL_INFO("Add directChannels[%llu], size[%zu]", i, detourChannels_[i].size());
    }
    for (uint64_t i = 0; i < detourPathNum_; i++) {
        // 有detourPathNum组的绕路链路，只添加sendOnly的channel
        for (uint64_t j = 0; j < rankSize_ - 1; j++) {
            detourChannels_[i + directPathNum].emplace_back(channels[(i + 1) * (rankSize_ - 1) + j]);
        }
        HCCL_INFO("Add detourChannels_[%llu], size[%zu]", i, detourChannels_[i].size());
    }

    return;
}

CcuKernelAllGatherMeshDetour1D::CcuKernelAllGatherMeshDetour1D(const CcuCtxArg       &arg,
                                                     const std::vector<CcuChannel *> &channels,
                                                     const CcuChannelGroup           &group)
    : CcuKernel(arg, channels, group)
{
    HCCL_INFO("[CcuKernelAllGatherMeshDetour1D] Enter Constructor.");
    const CcuCtxArgAllGatherMeshDetour1D *ctxArg = dynamic_cast<const CcuCtxArgAllGatherMeshDetour1D *>(&arg);
    if (ctxArg == nullptr) {
        THROW<NullPtrException>(StringFormat("CcuKernelAllGatherMeshDetour1D::ctxArg ptr is null"));
    }
    rankId_ = ctxArg->rankId_;
    if (ctxArg->dimSize_.size() > 0) {
        rankSize_ = ctxArg->dimSize_[0];
    }
    singleChannelSize_ = ctxArg->singleChannelSize_;
    detourPathNum_ = ctxArg->detourPathNum_;
    pathNumPerPeer_ = ctxArg->pathNumPerPeer_;

    ProcessChannels(channels);

    // 申请资源
    input_ = CreateVariable();
    baseOffset_ = CreateVariable();
    tailOffset_ = CreateVariable();
    loopIterNum_ = CreateVariable();
    groupOpSize_ = CreateGroupOpSize();

    uint16_t channelIdx = 0;
    // 按照rank号从小到大遍历channels，遇到本rank就填充本地资源，否则依次取远端资源，要求给框架返回的Link同样是按顺序排列的
    for (uint64_t peerId = 0; peerId < rankSize_; peerId++) {
        if (peerId == rankId_) {
            output_.push_back(CreateVariable());
            token_.push_back(CreateVariable());
        } else {
            HCCL_INFO("[CcuKernelAllGatherMeshDetour1D] MyRank[%u], PeerId[%llu], ChannelId[%u]",
                rankId_, peerId, channelIdx);
            CHK_PRT_RET(detourChannels_[0][channelIdx] == nullptr,
                HCCL_ERROR("[CcuKernelAllGatherMeshDetour1D] Algorithm channel ptr is null"),);
            output_.push_back(CreateVariable((*detourChannels_[0][channelIdx]), OUTPUT_XN_ID));
            token_.push_back(CreateVariable((*detourChannels_[0][channelIdx]), TOKEN_XN_ID));
            channelIdx++;
        }
    }
    for (uint32_t i = 0; i < pathNumPerPeer_; i++) {
        lengths_.emplace_back(CreateVariable());
    }

    return;
}

void CcuKernelAllGatherMeshDetour1D::AllocDetourRes()
{
    // 预期给每个对端使用的MS数量都相等
    moConfig.loopCount = CcuRep::CCU_MS_DEFAULT_LOOP_COUNT;
    moConfig.msInterleave = pathNumPerPeer_ * 1;  // Bcast为msNum*1，Reduce为msNum*rankSize_
    if (moRes.executor.size() == 0) {
        moRes.executor = CreateBlockExecutor(moConfig.loopCount);
        moRes.maskSignal = CreateBlockMaskSignal(moConfig.loopCount);
        moRes.ccuBuffer = CreateBlockCcuBuffer(moConfig.loopCount * moConfig.msInterleave);
    }
    return;
}

void CcuKernelAllGatherMeshDetour1D::CreateMultiOpBroadcastDetour()
{
    // 设到每个对端有相同数量的多个channel，每个channel需要传输的长度与同下标的lengths中的值对应 <直连--R1,绕路--R1>--<L0,L1>
    // 当每个channel不均等切分时，考虑1.添加重复的channel，每个channel都用1片MS；2.lengths中给每个channel填不同的长度
    AllocDetourRes();

    std::string loopType = "broadcastDetour";
    if (registeredLoop.find(loopType) != registeredLoop.end()) {
        return;
    }

    CcuRep::LoopBlock lb(this, loopType + "_loop");
    {
        // loopblock的形参
        std::vector<CcuRep::Memory> src;  // 每组channel对应一个src
        std::vector<CcuRep::Memory> dst;  // 每组channel对应rankSize_个dst
        std::vector<CcuRep::Variable> lengths;
        for (uint64_t i = 0; i < pathNumPerPeer_; i++) {
            lengths.emplace_back(CreateVariable());
            src.emplace_back(CreateMemory());
            for (uint64_t j = 0; j < rankSize_; j++) {
                dst.emplace_back(CreateMemory());
            }
        }

        lb(src, dst, lengths);
        std::vector<CcuRep::CcuBuffer> bufs;
        std::vector<CcuRep::MaskSignal> sems;
        for (uint32_t i = 0; i < pathNumPerPeer_; i++) {
            bufs.emplace_back(moRes.ccuBuffer[i]);
            sems.emplace_back(moRes.maskSignal[i]);
        }

        // 从本地搬运多片数据到多个MS
        for (uint64_t i = 0; i < pathNumPerPeer_; i++) {
            LocalCopy(bufs[i], src[i], lengths[i], sems[i]);
        }
        // 等待数据搬到MS
        for (uint64_t i = 0; i < pathNumPerPeer_; i++) {
            LocalWait(sems[i]);
        }
        // 给每个peer搬运多个MS上的数据
        for (uint64_t i = 0; i < pathNumPerPeer_; i++) {
            for (uint64_t j = 0; j < rankSize_ - 1; j++) {
                if (detourChannels_[i][j] == nullptr) {
                    THROW<CcuApiException>("channel is nullptr");
                }
                Write(*detourChannels_[i][j], dst[i * rankSize_ + j], bufs[i], lengths[i], sems[i], 1 << j);
            }
            LocalCopy(dst[i * rankSize_ + rankSize_ - 1], bufs[i], lengths[i], sems[i], 1 << (rankSize_ - 1));
        }
        // 等待给所有远端写完数据
        for (uint32_t i = 0; i < pathNumPerPeer_; i++) {
            LocalWait(sems[i], (1 << rankSize_) - 1);
        }
    }

    registeredLoop.insert(loopType);
    return;
}

void CcuKernelAllGatherMeshDetour1D::GroupBroadcastDetour(
    std::vector<CcuRep::Variable> &lengths, std::vector<CcuRep::Memory> &src, std::vector<CcuRep::Memory> &dst)
{
    CreateMultiOpBroadcastDetour();
    uint32_t interLeave = pathNumPerPeer_ * 1;

    CCU_IF(loopIterNum_ != 0) {
        CcuRep::Variable loopParam = CreateVariable();
        CcuRep::Variable paraCfg = CreateVariable();
        CcuRep::Variable offsetCfg = CreateVariable();

        // sliceSize：单次搬运量，4K*msNum，必须与lengths的总和相等（链路间可以划分不同流量）
        loopParam = CcuRep::GetLoopParam(0, singleChannelSize_ * moConfig.loopCount, 0);  // 偏移是单次总搬运量*loopNum
        loopParam += loopIterNum_;  // 加上loop的迭代次数构成完整loop参数
        paraCfg = CcuRep::GetParallelParam(moConfig.loopCount - 1, 0, 1);  // loop固定展开到128个
        offsetCfg = CcuRep::GetOffsetParam(singleChannelSize_, interLeave, 1);  // 下一个loop偏移量
        auto lc = Loop("broadcastDetour_loop")(src, dst, lengths);
        LoopGroup({lc}, {loopParam}, paraCfg, offsetCfg);
    }
    return;
}

void CcuKernelAllGatherMeshDetour1D::FirstStep()
{
    // step1，绕路搬整块
    // 申请memory地址
    std::vector<CcuRep::Memory> src;
    std::vector<CcuRep::Memory> dst;
    for (uint32_t i = 0; i < pathNumPerPeer_; i++) {
        src.emplace_back(CreateMemory());
        for (uint32_t j = 0; j < rankSize_; j++) {
            dst.emplace_back(CreateMemory());
        }
    }

    // 地址计算
    src[0].addr = input_;
    src[0].token = token_[rankId_];
    for (uint32_t i = 1; i < pathNumPerPeer_; i++) {
        src[i].addr = src[i - 1].addr + lengths_[i - 1];
        src[i].token = token_[rankId_];
    }
    uint32_t dstId = 0;
    uint32_t curId = 0;
    for (uint64_t rankIdx = 0; rankIdx < rankSize_; rankIdx++) {
        if (rankIdx != rankId_) {
            curId = dstId;
            dstId++;
        } else {
            curId = rankSize_ - 1;
        }
        dst[curId].addr = output_[rankIdx];  // 直连链路对应的是下标为0*rankSize_+curId的分片
        dst[curId].addr += baseOffset_;
        dst[curId].token = token_[rankIdx];
    }
    for (uint64_t i = 1; i < pathNumPerPeer_; i++) {
        for (uint64_t j = 0; j < rankSize_; j++) {
            dst[i * rankSize_ + j].addr = dst[(i - 1) * rankSize_ + j].addr + lengths_[i - 1];
            dst[i * rankSize_ + j].token = dst[(i - 1) * rankSize_ + j].token;
        }
    }
    GroupBroadcastDetour(lengths_, src, dst);

    return;
}

void CcuKernelAllGatherMeshDetour1D::SecondStep()
{
    // step2，直连搬尾块
    CcuRep::Memory tailSrc = CreateMemory();
    std::vector<CcuRep::Memory> tailDst;
    for (uint32_t i = 0; i < rankSize_; i++) {
        tailDst.emplace_back(CreateMemory());
    }
    tailSrc.addr = input_;
    tailSrc.addr += tailOffset_;
    tailSrc.token = token_[rankId_];
    uint32_t dstId = 0;
    uint32_t curId = 0;
    for (uint64_t rankIdx = 0; rankIdx < rankSize_; rankIdx++) {
        if (rankIdx != rankId_) {
            curId = dstId;
            dstId++;
        } else {
            curId = rankSize_ - 1;
        }
        tailDst[curId].addr = output_[rankIdx];
        tailDst[curId].addr += baseOffset_;
        tailDst[curId].addr += tailOffset_;
        tailDst[curId].token = token_[rankIdx];
    }
    GroupBroadcast(detourChannels_[0], tailDst, tailSrc, groupOpSize_);

    return;
}

void CcuKernelAllGatherMeshDetour1D::Algorithm()
{
    HCCL_INFO("[CcuKernelAllGatherMeshDetour1D] AllGatherMeshDetour1D run.");
    uint16_t selfBit = 1 << rankId_;
    uint16_t allBit  = ((1 << rankSize_) - 1) & (~(1 << rankId_));

    Load(input_);
    Load(output_[rankId_]);
    Load(token_[rankId_]);
    Load(baseOffset_);
    Load(tailOffset_);
    Load(loopIterNum_);
    Load(groupOpSize_);
    for (uint32_t i = 0; i < pathNumPerPeer_; i++) {
        Load(lengths_[i]);
    }

    // 只通过直连链路给对端置位，groupwait仍然关联所有channel
    for (auto t : detourChannels_[0]) {
        WriteVariableWithSignal(*t, output_[rankId_], OUTPUT_XN_ID, CKE_IDX_1, selfBit); // index = 1，传递output信息
        WriteVariableWithSignal(*t, token_[rankId_], TOKEN_XN_ID, CKE_IDX_2, selfBit);  // index = 2，传递token信息
    }
    GroupWait(*channelGroup, CKE_IDX_1, allBit); // index = 1，传递output信息
    GroupWait(*channelGroup, CKE_IDX_2, allBit); // index = 2，传递token信息

    FirstStep();  // 绕路整块搬运
    SecondStep();  // 直连尾块搬运

    for (auto t : detourChannels_[0]) {
        RemotePost(*t, CKE_IDX_0, selfBit);
    }
    GroupWait(*channelGroup, CKE_IDX_0, allBit);
    HCCL_INFO("[CcuKernelAllGatherMeshDetour1D] AllGatherMeshDetour1D end.");
    return;
}

std::vector<uint64_t> CcuKernelAllGatherMeshDetour1D::GeneArgs(const CcuTaskArg &arg)
{
    const CcuTaskArgAllGatherMeshDetour1D *taskArg = dynamic_cast<const CcuTaskArgAllGatherMeshDetour1D *>(&arg);
    if (taskArg == nullptr) {
        THROW<NullPtrException>(StringFormat("CcuKernelAllGatherMeshDetour1D::taskArg ptr is null"));
    }
    uint64_t inputAddr  = taskArg->inputAddr_;
    uint64_t outputAddr = taskArg->outputAddr_;
    uint64_t tokenInfo  = taskArg->token_;
    uint64_t baseOffset = taskArg->baseOffset_;
    uint64_t tailOffset = taskArg->tailOffset_;
    uint64_t loopIterNum = taskArg->loopIterNum_;
    auto goSize         = CalGoSize(taskArg->tailSize_);

    std::vector<uint64_t> sqeArgs = {inputAddr, outputAddr, tokenInfo, baseOffset, tailOffset, loopIterNum,
                                     goSize[0], goSize[1], goSize[2], goSize[3]};
    for (auto len : taskArg->lengths_) {
        sqeArgs.emplace_back(len);
    }
    return sqeArgs;
}
}
