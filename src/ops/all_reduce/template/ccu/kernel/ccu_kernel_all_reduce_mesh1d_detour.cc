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
#include "ccu_kernel_all_reduce_mesh1d_detour.h"
#include "ccu_kernel_utils.h"
#include "ccu_kernel.h"

namespace ops_hccl {
using namespace hcomm;

constexpr int INPUT_XN_ID = 0;
constexpr int OUTPUT_XN_ID = 1;
constexpr int TOKEN_XN_ID = 2;
constexpr int POST_SYNC_ID = 3;
constexpr int CKE_IDX_0 = 0;

CcuKernelAllReduceMesh1DDetour::CcuKernelAllReduceMesh1DDetour(const CcuKernelArg &arg)
    : CcuKernelAlgBase(arg)
{
    const CcuKernelArgAllReduceMesh1DDetour *kernelArg = dynamic_cast<const CcuKernelArgAllReduceMesh1DDetour *>(&arg);
    if (kernelArg == nullptr) {
        HCCL_ERROR("CcuKernelAllReduceMesh1DDetour::kernelArg ptr is null");
    }
    rankId_             = kernelArg->rankId_;
    rankSize_           = kernelArg->dimSize_[0];
    dataType_           = kernelArg->op_.DataDes.dataType;
    outputDataType_     = kernelArg->op_.DataDes.outputType;
    channels_           = kernelArg->channels;
    std::vector<u32> channelsIndexVec = kernelArg->channelsIndexVec_;
    if (outputDataType_ == HcclDataType::HCCL_DATA_TYPE_RESERVED) {
        outputDataType_ = dataType_;
        HCCL_INFO("[CcuKernelAllReduceMesh1DDetour] outputDataType is [HCCL_DATA_TYPE_RESERVED], set outputDataType to[%d]",
            outputDataType_);
    }
    reduceOp_ = kernelArg->op_.reduceType;
    singleTransferSize = kernelArg->singleTransferSize_;
    detourPathNum = kernelArg->detourPathNum_;
    pathNumPerPeer = kernelArg->pathNumPerPeer_;
    HCCL_INFO("[CcuKernelAllReduceMesh1DDetour] Init, kernelArgs are rankId_[%u], rankSize_[%u], dataType[%d], "
        "outputDataType[%d], reduceOp[%d]", rankId_, rankSize_, dataType_,
        outputDataType_, reduceOp_);

    HCCL_INFO("[CcuKernelAllReduceMesh1DDetour] channel.size[%zu]", channels_.size());
    if (channels_.size() < rankSize_ -1) {
        HCCL_ERROR("CcuKernelAllReduceMesh1DDetour channels_ size is less");
    }
    for (uint64_t i = 0; i < pathNumPerPeer; i++) {
        // 到每个对端有pathNum个channel，故detourchannel中共有pathNum组
        detourChannels_.emplace_back(std::vector<ChannelHandle>());
    }
    uint64_t directPathNum = pathNumPerPeer - detourPathNum;
    for (uint64_t i = 0; i < directPathNum; i++) {
        // 有pathNum-detourPathNum组的直连链路，每组重复
        for (uint64_t j = 0; j < rankSize_ - 1; j++) {
            detourChannels_[i].emplace_back(channels_[channelsIndexVec[j]]);
        }
        HCCL_INFO("[CcuKernelAllReduceMesh1DDetour] Add directchannels_[%llu], size[%zu]", i, detourChannels_[i].size());
    }
    for (uint64_t i = 0; i < detourPathNum; i++) {
        for (uint64_t j = 0; j < rankSize_ - 1; j++) {
            detourChannels_[i + directPathNum].emplace_back(channels_[channelsIndexVec[(i + 1) * (rankSize_ - 1) + j]]);
            detourChannels_[i + directPathNum].emplace_back(channels_[channelsIndexVec[(i + 1) * (rankSize_ - 1) + j + detourPathNum * (rankSize_ - 1)]]);
            HCCL_INFO("detourChannels_ emplace_back sendLink[%u], recvLink[%u]",
                (i + 1) * (rankSize_ - 1) + j, (i + 1) * (rankSize_ - 1) + j + detourPathNum * (rankSize_ - 1));
        }
    }
}

void CcuKernelAllReduceMesh1DDetour::CreateResource(uint32_t msInterleave)
{
    moConfig.loopCount = CcuRep::CCU_MS_DEFAULT_LOOP_COUNT;
    moConfig.msInterleave = msInterleave;
    if (moRes.executor.size() == 0) {
        moRes.completedEvent = CreateBlockCompletedEvent(moConfig.loopCount);
        moRes.executor = CreateBlockExecutor(moConfig.loopCount);
        moRes.ccuBuf = CreateBlockCcuBuf(moConfig.loopCount * moConfig.msInterleave);
    }
}

void CcuKernelAllReduceMesh1DDetour::DoLocalReduce(HcclDataType &dataType, HcclDataType &outputDataType, HcclReduceOp &opType, 
                                     std::vector<std::vector<CcuRep::CcuBuf>> &bufs, std::vector<CcuRep::Variable> &lengths, 
                                     std::vector<CcuRep::CompletedEvent> &sems)
{
    for (uint64_t i = 0; i < pathNumPerPeer; i++) {
        sems[i].SetMask(1 << (rankSize_ - 1));
        CcuRep::LocalAddr tmpAddr = CreateLocalAddr();
        tmpAddr.addr = src[i * rankSize_ + rankSize_ - 1].addr;
        tmpAddr.token = src[i * rankSize_ + rankSize_ - 1].token;
        LocalCopyNb(bufs[i][rankSize_ - 1], tmpAddr, lengths[i], sems[i]);
    }
    for (uint64_t i = 0; i < pathNumPerPeer; i++) {
        sems[i].SetMask((1 << rankSize_) - 1);
        WaitEvent(sems[i]);
    }
    if (rankSize_ > 1) {
        for (uint64_t i = 0; i < pathNumPerPeer; i++) {
            LocalReduceNb(bufs[i], rankSize_, dataType, outputDataType, opType, lengths[i], sems[i]);
            WaitEvent(sems[i]);
        }
    }
}

void CcuKernelAllReduceMesh1DDetour::CreateReduceLoop(HcclDataType &dataType, HcclDataType &outputDataType, HcclReduceOp &opType)
{
    CcuRep::LoopBlock lb(this, loopType + "_loop");
    {
        // loopblock的形参
        std::vector<CcuRep::LocalAddr> dst;
        std::vector<CcuRep::RemoteAddr> src;
        std::vector<CcuRep::Variable> lengths;
        for (uint64_t i = 0; i < pathNumPerPeer; i++) {
            lengths.emplace_back(CreateVariable());
            dst.emplace_back(CreateLocalAddr());
            for (uint64_t j = 0; j < rankSize_; j++) {
                src.emplace_back(CreateRemoteAddr());
            }
        }
        lb(src, dst, lengths);
        std::vector<std::vector<CcuRep::CcuBuf>> bufs;
        bufs.resize(pathNumPerPeer);
        std::vector<CcuRep::CompletedEvent> sems;

        for (uint64_t i = 0; i < pathNumPerPeer; i++) {
            for (uint64_t j = 0; j < rankSize_; j++) {
                bufs[i].emplace_back(moRes.ccuBuf[i * rankSize_ + j]);
            }
            sems.emplace_back(moRes.completedEvent[i]);
        }
        // 先读远端直连的到本地MS
        uint64_t directPathNum = pathNumPerPeer - detourPathNum;
        for (uint64_t i = 0; i < directPathNum; i++) {
            for (uint32_t j = 0; j < detourChannels_[i].size(); j++) {
                sems[i].SetMask(1 << j);
                ReadNb(detourChannels_[i][j], bufs[i][j], src[i * rankSize_ + j], lengths[i], sems[i]);
            }
        }
        // 再读远端绕路的到本地MS
        for (uint64_t i = directPathNum; i < pathNumPerPeer; i++) {
            for (uint64_t j = 0; j < rankSize_ - 1; j++) {
                sems[i].SetMask(1 << j);
                ReadNb(detourChannels_[i][j * 2 + 1], bufs[i][j], src[i * rankSize_ + j], lengths[i], sems[i]);
            }
        }
        DoLocalReduce(dataType, outputDataType, opType, bufs, lengths, sems);
        for (uint64_t i = 0; i < pathNumPerPeer; i++) {
            LocalCopyNb(dst[i], bufs[i][0], lengths[i], sems[i]);
            WaitEvent(sems[i]);
        }
    }
}

void CcuKernelAllReduceMesh1DDetour::CreateMultiOpReduceDetour(HcclDataType &dataType, HcclDataType &outputDataType, HcclReduceOp &opType)
{
    CreateResource(pathNumPerPeer * rankSize_);
    std::string loopType = "reduceDetour";
    if (registeredLoop.find(loopType) != registeredLoop.end()) {
        return;
    }
    CreateReduceLoop(dataType, outputDataType, opType);
    registeredLoop.insert(loopType);
    return;
}

void CcuKernelAllReduceMesh1DDetour::GroupReduceDetour(std::vector<CcuRep::RemoteAddr> &src,
    std::vector<CcuRep::LocalAddr> &dst, HcclDataType &dataType, HcclDataType &outputDataType, HcclReduceOp &opType)
{
    CreateMultiOpReduceDetour(dataType, outputDataType, opType);
    uint32_t interLeave = 8;

    CCU_IF(iterNum_ != 0) {
        CcuRep::Variable loopParam = CreateVariable();
        CcuRep::Variable paraCfg = CreateVariable();
        CcuRep::Variable offsetCfg = CreateVariable();

        loopParam = GetLoopParam(0, singleTransferSize * moConfig.loopCount, 0);  // 下次迭代的偏移是单次总搬运量*loopNum
        loopParam += iterNum_;  // 加上loop的迭代次数构成完整loop参数
        paraCfg = GetParallelParam(moConfig.loopCount - 1, 0, 1);  // loop固定展开到128个
        offsetCfg = GetOffsetParam(singleTransferSize, interLeave, pathNumPerPeer);  // 下一个loop偏移量
        auto lc = Loop("reduceDetour_loop")(src, dst, lengths_);
        LoopGroup({lc}, {loopParam}, paraCfg, offsetCfg);
    }
    return;
}

void CcuKernelAllReduceMesh1DDetour::CreateBroadcastLoop()
{
    CcuRep::LoopBlock lb(this, loopType + "_loop");
    {
        // loopblock的形参
        std::vector<CcuRep::LocalAddr> src;
        std::vector<CcuRep::RemoteAddr> dst;
        std::vector<CcuRep::Variable> lengths;
        for (uint64_t i = 0; i < pathNumPerPeer; i++) {
            lengths.emplace_back(CreateVariable());
            src.emplace_back(CreateLocalAddr());
            for (uint64_t j = 0; j < rankSize_; j++) {
                dst.emplace_back(CreateRemoteAddr());
            }
        }

        lb(src, dst, lengths);
        std::vector<CcuRep::CcuBuf> bufs;
        std::vector<CcuRep::CompletedEvent> sems;
        for (uint64_t i = 0; i < pathNumPerPeer; i++) {
            bufs.emplace_back(moRes.ccuBuf[i]);
            sems.emplace_back(moRes.completedEvent[i]);
        }

        // 从本地搬运多片数据到多个MS
        for (uint64_t i = 0; i < pathNumPerPeer; i++) {
            LocalCopyNb(bufs[i], src[i], lengths[i], sems[i]);
        }
        // 等待数据搬到MS
        for (uint64_t i = 0; i < pathNumPerPeer; i++) {
            WaitEvent(sems[i]);
        }
        // 给每个peer搬运多个MS上的数据
        for (uint64_t i = 0; i < pathNumPerPeer; i++) {
            for (uint64_t j = 0; j < rankSize_ - 1; j++) {
                sems[i].SetMask(1 << j);
                WriteNb(detourChannels_[i][j * 2], dst[i * rankSize_ + j], bufs[i], lengths[i], sems[i]);
            }
            sems[i].SetMask(1 << (rankSize_ - 1));
            CcuRep::LocalAddr tmpAddr = CreateLocalAddr();
            tmpAddr.addr = dst[i * rankSize_ + rankSize_ - 1].addr;
            tmpAddr.token = dst[i * rankSize_ + rankSize_ - 1].token;
            LocalCopyNb(tmpAddr, bufs[i], lengths[i], sems[i]);
        }
        // 等待给所有远端写完数据
        for (uint64_t i = 0; i < pathNumPerPeer; i++) {
            sems[i].SetMask((1 << rankSize_) - 1);
            WaitEvent(sems[i]);
        }
    }
}

void CcuKernelAllReduceMesh1DDetour::CreateMultiOpBroadcastDetour()
{
    CreateResource(pathNumPerPeer);
    std::string loopType = "broadcastDetour";
    if (registeredLoop.find(loopType) != registeredLoop.end()) {
        return;
    }
    CreateBroadcastLoop();
    registeredLoop.insert(loopType);
    return;
}

void CcuKernelAllReduceMesh1DDetour::GroupBroadcastDetour(std::vector<CcuRep::Variable> &lengths, std::vector<CcuRep::LocalAddr> &src,
    std::vector<CcuRep::RemoteAddr> &dst)
{
    CreateMultiOpBroadcastDetour();
    uint32_t interLeave = 8;

    CCU_IF(iterNum_ != 0) {
        CcuRep::Variable loopParam = CreateVariable();
        CcuRep::Variable paraCfg = CreateVariable();
        CcuRep::Variable offsetCfg = CreateVariable();

        loopParam = GetLoopParam(0, singleTransferSize * moConfig.loopCount, 0);  // 偏移是单次总搬运量*loopNum
        loopParam += iterNum_;  // 加上loop的迭代次数构成完整loop参数
        paraCfg = GetParallelParam(moConfig.loopCount - 1, 0, 1);  // loop固定展开到128个
        offsetCfg = GetOffsetParam(singleTransferSize, interLeave, pathNumPerPeer);  // 下一个loop偏移量
        auto lc = Loop("broadcastDetour_loop")(src, dst, lengths);
        LoopGroup({lc}, {loopParam}, paraCfg, offsetCfg);
    }
    return;
}

void CcuKernelAllReduceMesh1DDetour::ReduceScatterFirstStep()
{
    std::vector<CcuRep::RemoteAddr> reduceSrc;
    std::vector<CcuRep::LocalAddr> reduceDst;

    // 为每个直连或绕路channel分别准备reduceSrc与reduceDst
    for (uint64_t i = 0; i < pathNumPerPeer; i++) {
        reduceDst.emplace_back(CreateLocalAddr());
        for (uint64_t j = 0; j < rankSize_; j++) {
            reduceSrc.emplace_back(CreateRemoteAddr());
        }
    }

    // reduceDst填充
    reduceDst[0].addr = output_[rankId_];
    reduceDst[0].addr += offset_;
    reduceDst[0].token = token_[rankId_];
    for (uint64_t i = 1; i < pathNumPerPeer; i++) {
        reduceDst[i].addr = reduceDst[i - 1].addr + lengths_[i - 1];
        reduceDst[i].token = token_[rankId_];
    }
    // 直连channel的reduceSrc填充
    uint32_t srcId = 0;
    uint32_t curId = 0;
    for (uint64_t rankIdx = 0; rankIdx < rankSize_; rankIdx++) {
        if (rankIdx != rankId_) {
            curId = srcId;
            srcId++;
        } else {
            curId = rankSize_ - 1;
        }
        reduceSrc[curId].addr = input_[rankIdx];
        reduceSrc[curId].addr += offset_;
        reduceSrc[curId].token = token_[rankIdx];
    }
    // 绕路channel的reduceSrc相比直连src再做偏移
    for (uint64_t i = 1; i < pathNumPerPeer; i++) {
        for (uint64_t j = 0; j < rankSize_; j++) {
            reduceSrc[i * rankSize_ + j].addr = reduceSrc[(i - 1) * rankSize_ + j].addr + lengths_[i - 1];
            reduceSrc[i * rankSize_ + j].token = reduceSrc[(i - 1) * rankSize_ + j].token;
        }
    }

    // 整块数据用绕路Reduce
    GroupReduceDetour(reduceSrc, reduceDst, dataType_, outputDataType_, reduceOp_);
    return;
}

void CcuKernelAllReduceMesh1DDetour::ReduceScatterSecondStep()
{
    // 余下的尾块用直连Reduce
    std::vector<CcuRep::RemoteAddr> tailSrc;
    CcuRep::LocalAddr tailDst = CreateLocalAddr();
    for (uint64_t i = 0; i < rankSize_; i++) {
        tailSrc.emplace_back(CreateRemoteAddr());
    }
    tailDst.addr = output_[rankId_];
    tailDst.addr += offset_;
    tailDst.addr += tailOffset_;
    tailDst.token = token_[rankId_];
    uint32_t srcId = 0;
    uint32_t curId = 0;
    for (uint64_t rankIdx = 0; rankIdx < rankSize_; rankIdx++) {
        if (rankIdx != rankId_) {
            curId = srcId;
            srcId++;
        } else {
            curId = rankSize_ - 1;
        }
        tailSrc[curId].addr = input_[rankIdx];
        tailSrc[curId].addr += tailOffset_;
        tailSrc[curId].addr += offset_;
        tailSrc[curId].token = token_[rankIdx];
    }

    GroupReduce(detourChannels_[0], tailDst, tailSrc, groupOpSize_, dataType_, outputDataType_, reduceOp_);
    return;
}


void CcuKernelAllReduceMesh1DDetour::AllGatherFirstStep()
{
    // 开始AllGather
    std::vector<CcuRep::LocalAddr> allGatherSrc;
    std::vector<CcuRep::RemoteAddr> allGatherDst;

    // 为每个直连或绕路channel分别准备src与dst
    for (uint64_t i = 0; i < pathNumPerPeer; i++) {
        allGatherSrc.emplace_back(CreateLocalAddr());
        for (uint64_t j = 0; j < rankSize_; j++) {
            allGatherDst.emplace_back(CreateRemoteAddr());
        }
    }
    // allGather 的输入就是 reduceScatter 的输出
    allGatherSrc[0].addr = output_[rankId_]; // 直连源地址
    allGatherSrc[0].addr += offset_;
    allGatherSrc[0].token = token_[rankId_];
    for (uint64_t i = 1; i < pathNumPerPeer; i++) {
        allGatherSrc[i].addr = allGatherSrc[i - 1].addr + lengths_[i - 1];
        allGatherSrc[i].token = token_[rankId_];
    }

    // 直连的allGatherDst填充
    uint32_t curId = 0;
    uint32_t dstId = 0;
    for (uint64_t rankIdx = 0; rankIdx < rankSize_; rankIdx++) {
        if (rankIdx != rankId_) {
            curId = dstId;
            dstId++;
        } else {
            curId = rankSize_ - 1;
        }
        allGatherDst[curId].addr = output_[rankIdx];
        allGatherDst[curId].addr += offset_;
        allGatherDst[curId].token = token_[rankIdx];
    }

    // 绕路的allGatherDst填充，相比直连做偏移
    for (uint64_t i = 1; i < pathNumPerPeer; i++) {
        for (uint64_t j = 0; j < rankSize_; j++) {
            allGatherDst[i * rankSize_ + j].addr = allGatherDst[(i - 1) * rankSize_ + j].addr + lengths_[i - 1];
            allGatherDst[i * rankSize_ + j].token = allGatherDst[(i - 1) * rankSize_ + j].token;
        }
    }
    GroupBroadcastDetour(lengths_, allGatherSrc, allGatherDst);
    return;
}

void CcuKernelAllReduceMesh1DDetour::AllGatherSecondStep()
{
    // 余下的尾块用直连channel发送
    CcuRep::LocalAddr bcastTailSrc = CreateLocalAddr();
    std::vector<CcuRep::RemoteAddr> bcastTailDst;
    for (uint64_t i = 0; i < rankSize_; i++) {
        bcastTailDst.emplace_back(CreateRemoteAddr());
    }
    bcastTailSrc.addr = output_[rankId_];
    bcastTailSrc.addr += offset_;
    bcastTailSrc.addr += tailOffset_;
    bcastTailSrc.token = token_[rankId_];
    uint32_t dstId = 0;
    uint32_t curId = 0;
    for (uint64_t rankIdx = 0; rankIdx < rankSize_; rankIdx++) {
        if (rankIdx != rankId_) {
            curId = dstId;
            dstId++;
        } else {
            curId = rankSize_ - 1;
        }
        bcastTailDst[curId].addr = output_[rankIdx];
        bcastTailDst[curId].addr += offset_;
        bcastTailDst[curId].addr += tailOffset_;
        bcastTailDst[curId].token = token_[rankIdx];
    }
    GroupBroadcast(detourChannels_[0], bcastTailDst, bcastTailSrc, groupOpSize_);
    return;
}

void CcuKernelAllReduceMesh1DDetour::InitResources()
{
    // 初始化资源
    uint16_t channelIdx = 0;
    // 按照rank号从小到大遍历channels_，遇到本rank就填充本地资源，否则依次取远端资源，要求给框架返回的Link同样是按顺序排列的
    for (uint64_t peerId = 0; peerId < rankSize_; peerId++) {
        if (peerId == rankId_) {
            input_.push_back(CreateVariable());
            output_.push_back(CreateVariable());
            token_.push_back(CreateVariable());
        } else {
            HCCL_INFO("[CcuKernelAllReduceMesh1DDetour] MyRank[%u], PeerId[%llu], channelId[%u]",
                rankId_, peerId, channelIdx);
            CcuRep::Variable inputVar, tokenVar, outputVar;
            CreateVariable(detourChannels_[0][channelIdx], INPUT_XN_ID, &inputVar);
            CreateVariable(detourChannels_[0][channelIdx], OUTPUT_XN_ID, &outputVar);
            CreateVariable(detourChannels_[0][channelIdx], TOKEN_XN_ID, &tokenVar);
            input_.push_back(inputVar);
            output_.push_back(outputVar);
            token_.push_back(tokenVar);
            channelIdx++;
        }
    }
    offset_ = CreateVariable();
    iterNum_ = CreateVariable();
    tailOffset_ = CreateVariable();
    tailSize_ = CreateVariable();
    groupOpSize_ = CreateGroupOpSize();
    for (uint64_t i = 0; i < pathNumPerPeer; i++) {
        lengths_.emplace_back(CreateVariable());
    }

    Load(input_[rankId_]);
    Load(output_[rankId_]);
    Load(token_[rankId_]);
    Load(offset_);
    Load(iterNum_);
    Load(tailOffset_);
    Load(tailSize_);
    Load(groupOpSize_);
    for (uint64_t i = 0; i < pathNumPerPeer; i++) {
        Load(lengths_[i]);
    }
}

void CcuKernelAllReduceMesh1DDetour::PreSync()
{
    for (auto &t : detourChannels_[0]) {
        NotifyRecord(t, CKE_IDX_0, INPUT_XN_ID, input_[rankId_], 1 << INPUT_XN_ID);
        NotifyRecord(t, CKE_IDX_0, OUTPUT_XN_ID, output_[rankId_], 1 << OUTPUT_XN_ID);
        NotifyRecord(t, CKE_IDX_0, TOKEN_XN_ID, token_[rankId_], 1 << TOKEN_XN_ID);
    }

    uint16_t syncBit = 1 << INPUT_XN_ID | 1 << OUTPUT_XN_ID | 1 << TOKEN_XN_ID;
    for (auto t : detourChannels_[0]) {
        NotifyWait(t, CKE_IDX_0, syncBit);
    }
}

void CcuKernelAllReduceMesh1DDetour::PostSync()
{
    for (auto t : detourChannels_[0]) {
        NotifyRecord(t, CKE_IDX_0, 1 << POST_SYNC_ID);
    }
    for (auto t : detourChannels_[0]) {
        NotifyWait(t, CKE_IDX_0, 1 << POST_SYNC_ID);
    }
}

HcclResult CcuKernelAllReduceMesh1DDetour::Algorithm()
{
    HCCL_INFO("[CcuKernelAllReduceMesh1DDetour] AllReduceMeshDetour1D run.");
    InitResources();
    PreSync();
    ReduceScatterFirstStep();
    ReduceScatterSecondStep();
    AllGatherFirstStep();
    AllGatherSecondStep();
    PostSync();
    HCCL_INFO("[CcuKernelAllReduceMesh1DDetour] AllReduceMeshDetour1D end.");
    return HCCL_SUCCESS;
}

std::vector<uint64_t> CcuKernelAllReduceMesh1DDetour::GeneArgs(const CcuTaskArg &arg)
{
    const CcuTaskArgAllReduceMesh1DDetour *taskArg = dynamic_cast<const CcuTaskArgAllReduceMesh1DDetour *>(&arg);
    if (taskArg == nullptr) {
        HCCL_ERROR("CcuKernelAllReduceMesh1DDetour::taskArg ptr is null");
    }
    uint64_t inputAddr   = taskArg->inputAddr_;
    uint64_t outputAddr  = taskArg->outputAddr_;
    uint64_t tokenInfo   = taskArg->token_;
    uint64_t offset      = taskArg->offset_;
    uint64_t iterNum     = taskArg->iterNum_;
    uint64_t tailOffset  = taskArg->tailOffset_;
    uint64_t tailSize    = taskArg->tailSize_;
    auto     goSize      = CalGoSize(tailSize);

    HCCL_INFO("[CcuKernelAllReduceMesh1DDetour] GeneArgs, taskArg are inputAddr[%llu], outputAddr[%llu], "
        "offset[%llu], iterNum[%llu], tailOffset[%llu], tailSize[%llu]",
        inputAddr, outputAddr, offset, iterNum, tailOffset, tailSize);
    std::vector<uint64_t> sqeArgs = {inputAddr, outputAddr, tokenInfo, offset, iterNum, tailOffset, tailSize,
                                     goSize[0], goSize[1], goSize[2], goSize[3]};
    for (auto len : taskArg->lengths_) {
        sqeArgs.emplace_back(len);
    }
    return sqeArgs;
}

}
