/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "ccu_kernel_reduce_scatter_mesh1d_detour.h"

namespace ops_hccl {
using namespace hcomm;

constexpr int INPUT_XN_ID  = 0;
constexpr int TOKEN_XN_ID  = 1;
constexpr int POST_SYNC_ID = 2;
constexpr int CKE_IDX_0    = 0;

CcuKernelReduceScatterMeshDetour1D::CcuKernelReduceScatterMeshDetour1D(const CcuKernelArg &arg)
    : CcuKernelAlgBase(arg)
{
    const CcuKernelArgReduceScatterMeshDetour1D *kernelArg = dynamic_cast<const CcuKernelArgReduceScatterMeshDetour1D *>(&arg);
    rankId_        = kernelArg->rankId_;
    rankSize_      = kernelArg->dimSize_;
    channels_      = kernelArg->channels;
    dataType_       = kernelArg->opParam_.DataDes.dataType;
    outputDataType_ = kernelArg->opParam_.DataDes.outputType;
    if (outputDataType_ == HcclDataType::HCCL_DATA_TYPE_RESERVED) {
        outputDataType_ = dataType_;
        HCCL_DEBUG("[CcuKernelReduceScatterMeshDetour1D] outputDataType is [INVALID], set outputDataType to[%d]",
            outputDataType_);
    }
    reduceOp_ = kernelArg->opParam_.reduceType;
    singleTransportSize_ = kernelArg->singleTransportSize_;
    detourPathNum_ = kernelArg->detourPathNum_;
    pathNumPerPeer_ = kernelArg->pathNumPerPeer_;
    HCCL_INFO(
        "[CcuKernelArgReduceScatterMeshDetour1D] Init, KernelArgs are rankId[%u], rankSize_[%u], dataType[%d], "
        "outputDataType[%d], reduceOp[%d], singleTransportSize[%d], detourPathNum[%d], pathNumPerPeer[%d]",
        rankId_, rankSize_, dataType_, outputDataType_, reduceOp_, singleTransportSize_, detourPathNum_,
        pathNumPerPeer_);
    for (uint32_t i = 0; i < pathNumPerPeer_; i++) {
        // 到每个对端有pathNum个channel， 故detourChannels_中共有pathNum组
        detourChannels_.emplace_back(std::vector<ChannelHandle>());
    }
    uint64_t directPathNum = pathNumPerPeer_ - detourPathNum_;
    for (uint32_t i = 0; i < directPathNum; i++) {
        // 有pathNum-detourPathNum组的直连链路，每组重复
        for (uint32_t j = 0; j < rankSize_ - 1; j++) {
            // 到每个对端有pathNum个channel， 故detourChannels_中共有pathNum组
            detourChannels_[i].emplace_back(channels_[j]);
        }
        HCCL_DEBUG("[CcuKernelArgReduceScatterMeshDetour1D] add directChannels[%llu], size[%zu]", i , detourChannels_[i].size());
    }
    for (uint32_t i = 0; i < detourPathNum_; i++) {
        for (uint32_t j = 0; j < rankSize_ - 1; j++) {
            detourChannels_[i + directPathNum].emplace_back(channels_[(i + 1) * (rankSize_ - 1) + j]);
            HCCL_DEBUG("[CcuKernelArgReduceScatterMeshDetour1D] detourChannels_ emplace_back directChannels[%llu], size[%zu]",
                       i , detourChannels_[i].size());
            detourChannels_[i + directPathNum].emplace_back(
                channels_[(i + 1) * (rankSize_ - 1) + j + detourPathNum_ * (rankSize_ - 1)]
            );
        }
    }
}

HcclResult CcuKernelReduceScatterMeshDetour1D::InitResource()
{
    output_.push_back(CreateVariable());
    // 初始化资源
    uint16_t channelIdx = 0;
    // 按照rank号从小到大遍历channels，遇到本rank就填充本地资源，否则依次取远端资源，要求给框架返回的Link同样是按顺序排列的
    for (uint64_t peerId = 0; peerId < rankSize_; peerId++) {
        if (peerId == rankId_) {
            input_.push_back(CreateVariable());
            token_.push_back(CreateVariable());
        } else {
            HCCL_INFO("[CcuKernelReduceScatterMeshDetour1D] MyRank[%u], PeerId[%llu], ChannelId[%u]",
                rankId_, peerId, channelIdx);
            CcuRep::Variable inputVar, tokenVar;
            CHK_RET(CreateVariable(detourChannels_[0][channelIdx], INPUT_XN_ID, &inputVar));
            input_.push_back(inputVar);
            CHK_RET(CreateVariable(detourChannels_[0][channelIdx], TOKEN_XN_ID, &tokenVar));
            token_.push_back(tokenVar);
            channelIdx++;
        }
    }
    offset_ = CreateVariable();
    iterNum_ = CreateVariable();
    tailOffset_ = CreateVariable();
    tailSize_ = CreateVariable();
    groupOpSize_ = CreateGroupOpSize();
    for (uint32_t i = 0; i < pathNumPerPeer_; i++) {
        lengths_.emplace_back(CreateVariable());
    }
    return HcclResult::HCCL_SUCCESS;
}

void CcuKernelReduceScatterMeshDetour1D::LoadArgs()
{
    Load(input_[rankId_]);
    Load(output_[0]);
    Load(token_[rankId_]);
    Load(offset_);
    Load(iterNum_);
    Load(tailOffset_);
    Load(tailSize_);
    Load(groupOpSize_);
    for (uint32_t i = 0; i < pathNumPerPeer_; i++) {
        Load(lengths_[i]);
    }
    return;
}

void CcuKernelReduceScatterMeshDetour1D::PreSync()
{
    for (ChannelHandle channel : detourChannels_[0]) {
        NotifyRecord(channel, CKE_IDX_0, INPUT_XN_ID, input_[rankId_], 1 << INPUT_XN_ID);
        NotifyRecord(channel, CKE_IDX_0, TOKEN_XN_ID, token_[rankId_], 1 << TOKEN_XN_ID);
    }
    uint32_t allBit = 1 << INPUT_XN_ID | 1 << TOKEN_XN_ID;
    for (ChannelHandle channel : channels_) {
        NotifyWait(channel, CKE_IDX_0, allBit);
    }
    return;
}

void CcuKernelReduceScatterMeshDetour1D::PostSync()
{
    for (ChannelHandle channel : detourChannels_[0]) {
        NotifyRecord(channel, CKE_IDX_0, 1 << POST_SYNC_ID);
    }
    for (ChannelHandle channel : channels_) {
        NotifyWait(channel, CKE_IDX_0, 1 << POST_SYNC_ID);
    }
    return;
}

void CcuKernelReduceScatterMeshDetour1D::AllocGoResourceDetour()
{
    moConfig.loopCount = CcuRep::CCU_MS_DEFAULT_LOOP_COUNT;
    moConfig.msInterleave = pathNumPerPeer_ * rankSize_;
    if (moRes.executor.size() == 0) {
        moRes.executor = CreateBlockExecutor(moConfig.loopCount);
        moRes.completedEvent = CreateBlockCompletedEvent(moConfig.loopCount);
        moRes.ccuBuf = CreateBlockCcuBuf(moConfig.loopCount * moConfig.msInterleave);
    }
}

HcclResult CcuKernelReduceScatterMeshDetour1D::CreateMultiOpReduceDetour(
    HcclDataType &dataType, HcclDataType &outputDataType, HcclReduceOp &opType)
{
    AllocGoResourceDetour();
    std::string loopType = "reduceDetour";
    if (registeredLoop.find(loopType) != registeredLoop.end()) {
        return HcclResult::HCCL_SUCCESS;
    }
    CcuRep::LoopBlock lb(this, loopType + "_loop");
    {
        std::vector<CcuRep::RemoteAddr> src;
        std::vector<CcuRep::LocalAddr> dst;
        std::vector<CcuRep::Variable> lengths;
        for (uint32_t i = 0; i < pathNumPerPeer_; i++) {
            lengths.emplace_back(CreateVariable());
            dst.emplace_back(CreateLocalAddr());
            for (uint32_t j = 0; j < rankSize_; j++) {
                src.emplace_back(CreateRemoteAddr());
            }
        }

        lb(src, dst, lengths);
        std::vector<std::vector<CcuRep::CcuBuf>> bufs;
        bufs.resize(pathNumPerPeer_);
        std::vector<CcuRep::CompletedEvent> events;

        for (uint32_t i = 0; i < pathNumPerPeer_; i++) {
            for (uint32_t j = 0; j < rankSize_; j++) {
                bufs[i].emplace_back(moRes.ccuBuf[i * rankSize_ + j]);
            }
            events.emplace_back(moRes.completedEvent[i]);
        }

        LoopForReduceDetour(bufs, src, dst, lengths, events, dataType, outputDataType, opType);
    }
    registeredLoop.insert(loopType);
    return HcclResult::HCCL_SUCCESS;
}

void CcuKernelReduceScatterMeshDetour1D::LoopForReduceDetour(std::vector<std::vector<CcuRep::CcuBuf>> &bufs,
    std::vector<CcuRep::RemoteAddr> &src, std::vector<CcuRep::LocalAddr> &dst,
    std::vector<CcuRep::Variable> &lengths, std::vector<CcuRep::CompletedEvent> &events,
    HcclDataType &dataType, HcclDataType &outputDataType, HcclReduceOp &opType)
{
    // 先读远端直连的到本地MS
    uint64_t directPathNum = pathNumPerPeer_ - detourPathNum_;
    for (uint32_t i = 0; i < directPathNum; i++) {
        for (uint32_t j = 0; j < detourChannels_[i].size(); j++) {
            events[i].mask = 1 << j;
            ReadNb(detourChannels_[i][j], bufs[i][j], src[i * rankSize_ + j], lengths[i], events[i]);
        }
    }
    // 再读远端绕路的到本地MS
    for (uint32_t i = directPathNum; i < pathNumPerPeer_; i++) {
        for (uint32_t j = 0; j < rankSize_ - 1; j++) {
            events[i].mask = 1 << j;
            ReadNb(detourChannels_[i][j * 2 + 1], bufs[i][j], src[i * rankSize_ + j], lengths[i], events[i]);
        }
    }

    for (uint32_t i = 0; i < pathNumPerPeer_; i++) {
        events[i].mask = (1 << rankSize_) - 1;
        CcuRep::LocalAddr &localSrc = *reinterpret_cast<CcuRep::LocalAddr*>(&src[i * rankSize_ + rankSize_ - 1]);
        LocalCopyNb(bufs[i][rankSize_ - 1], localSrc, lengths[i], events[i]);
    }
    for (uint32_t i = 0; i < pathNumPerPeer_; i++) {
        events[i].mask = (1 << rankSize_) - 1;
        WaitEvent(events[i]);
    }
    if (rankSize_ > 1) {
        for (uint32_t i = 0; i < pathNumPerPeer_; i++) {
            events[i].mask = 1;
            LocalReduceNb(bufs[i], rankSize_, dataType, outputDataType, opType, lengths[i], events[i]);
            WaitEvent(events[i]);
        }
    }
    for (uint32_t i = 0; i < pathNumPerPeer_; i++) {
        events[i].mask = 1;
        LocalCopyNb(dst[i], bufs[i][0], lengths[i], events[i]);
        WaitEvent(events[i]);
    }
}

HcclResult CcuKernelReduceScatterMeshDetour1D::GroupReduceDetour(std::vector<CcuRep::RemoteAddr> &src,
    std::vector<CcuRep::LocalAddr> &dst, HcclDataType &dataType, HcclDataType &outputDataType, HcclReduceOp &opType)
{
    CreateMultiOpReduceDetour(dataType, outputDataType, opType);
    uint32_t interLeave = 8;

    CCU_IF(iterNum_ != 0) {
        CcuRep::Variable loopParam = CreateVariable();
        CcuRep::Variable paraCfg = CreateVariable();
        CcuRep::Variable offsetCfg = CreateVariable();

        loopParam = GetLoopParam(0, singleTransportSize_ * moConfig.loopCount, 0);  // 下次迭代的偏移是单次总搬运量*loopNum
        loopParam += iterNum_;  // 加上loop的迭代次数构成完整loop参数
        paraCfg = GetParallelParam(moConfig.loopCount - 1, 0, 1);  // loop固定展开到128个
        offsetCfg = GetOffsetParam(singleTransportSize_, interLeave, pathNumPerPeer_);  // 下一个loop偏移量
        auto lc = Loop("reduceDetour_loop")(src, dst, lengths_);
        LoopGroup({lc}, {loopParam}, paraCfg, offsetCfg);
    }
    return HcclResult::HCCL_SUCCESS;
}


HcclResult CcuKernelReduceScatterMeshDetour1D::GroupReduceDetourFunc()
{
    // 如果是4p*2场景，template里可以都传4k进来，channel和length通过<直连4k>, <直连4k>, <绕路4k>这样构造达成数据量2:1的效果
    
    std::vector<CcuRep::RemoteAddr> reduceSrc;
    std::vector<CcuRep::LocalAddr> reduceDst;
    
    // 为每个直连或绕路channel分别准备reduceSrc与reduceDst
    for (uint32_t i = 0; i < pathNumPerPeer_; i++) {
        reduceDst.emplace_back(CreateLocalAddr());
        for (uint32_t j = 0; j < rankSize_; j++) {
            reduceSrc.emplace_back(CreateRemoteAddr());
        }
    }

    // reduceDst填充
    reduceDst[0].addr = output_[0];
    // reduceDst[0].addr += offset_;
    reduceDst[0].token = token_[rankId_];
    for (uint32_t i = 1; i < pathNumPerPeer_; i++) {
        reduceDst[i].addr = reduceDst[i - 1].addr + lengths_[i - 1];
        reduceDst[i].token = token_[rankId_];
    }
    // 直连channel的reduceSrc填充
    uint32_t srcId = 0;
    uint32_t curId = 0;
    for (uint32_t rankIdx = 0; rankIdx < rankSize_; rankIdx++) {
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
    for (uint32_t i = 1; i < pathNumPerPeer_; i++) {
        for (uint32_t j = 0; j < rankSize_; j++) {
            reduceSrc[i * rankSize_ + j].addr = reduceSrc[(i - 1) * rankSize_ + j].addr + lengths_[i - 1];
            reduceSrc[i * rankSize_ + j].token = reduceSrc[(i - 1) * rankSize_ + j].token;
        }
    }
    GroupReduceDetour(reduceSrc, reduceDst, dataType_, outputDataType_, reduceOp_);
    return HcclResult::HCCL_SUCCESS;
}

HcclResult CcuKernelReduceScatterMeshDetour1D::GroupReduceForTailData()
{
    std::vector<CcuRep::RemoteAddr> tailSrc;
    CcuRep::LocalAddr tailDst = CreateLocalAddr();
    for (uint32_t i = 0; i < rankSize_; i++) {
        tailSrc.emplace_back(CreateRemoteAddr());
    }
    tailDst.addr = output_[0];
    // tailDst.addr += offset_;
    tailDst.addr += tailOffset_;
    tailDst.token = token_[rankId_];
    uint32_t srcId = 0;
    uint32_t curId = 0;
    for (uint32_t rankIdx = 0; rankIdx < rankSize_; rankIdx++) {
        if (rankIdx != rankId_) {
            curId = srcId;
            srcId++;
        } else {
            curId = rankSize_ - 1;
        }
        tailSrc[curId].addr = input_[rankIdx];
        tailSrc[curId].addr += offset_;
        tailSrc[curId].addr += tailOffset_;
        tailSrc[curId].token = token_[rankIdx];
    }

    GroupReduce(detourChannels_[0], tailDst, tailSrc, groupOpSize_, dataType_, outputDataType_, reduceOp_);
    return HcclResult::HCCL_SUCCESS;
}

HcclResult CcuKernelReduceScatterMeshDetour1D::Algorithm()
{
    HCCL_INFO("[CcuKernelReduceScatterMeshDetour1D] ReduceScatterMeshDetour1D run");
    
    CHK_RET(InitResource());
    
    LoadArgs();
    
    PreSync();

    CHK_RET(GroupReduceDetourFunc());

    // 余下的尾块用直连Reduce
    CHK_RET(GroupReduceForTailData());

    PostSync();

    HCCL_INFO("[CcuKernelReduceScatterMeshDetour1D] ReduceScatterMeshDetour1D end");
    
    return HcclResult::HCCL_SUCCESS;
}

std::vector<uint64_t> CcuKernelReduceScatterMeshDetour1D::GeneArgs(const CcuTaskArg &arg)
{
    const CcuTaskArgReduceScatterMeshDetour1D *taskArg = dynamic_cast<const CcuTaskArgReduceScatterMeshDetour1D *>(&arg);
    
    uint64_t inputAddr   = taskArg->inputAddr_;
    uint64_t outputAddr  = taskArg->outputAddr_;
    uint64_t tokenInfo   = taskArg->token_;
    uint64_t offset      = taskArg->offset_;
    uint64_t iterNum     = taskArg->iterNum_;
    uint64_t tailOffset  = taskArg->tailOffset_;
    uint64_t tailSize    = taskArg->tailSize_;
    auto     goSize      = CalGoSize(tailSize);

    HCCL_INFO("[CcuKernelReduceScatterMeshDetour1D] GeneArgs, taskArg are inputAddr[%llu], outputAddr[%llu], "
        "offset[%llu], iterNum[%llu], tailOffset[%llu], tailSize[%llu]",
        inputAddr, outputAddr, offset, iterNum, tailOffset, tailSize);
    std::vector<uint64_t> sqeArgs = {inputAddr, outputAddr, tokenInfo, offset, iterNum, tailOffset, tailSize,
                                     goSize[0], goSize[1], goSize[2], goSize[3]};
    for (auto len : taskArg->lengths_) {
        HCCL_INFO("get lengths");
        sqeArgs.emplace_back(len);
    }
    return sqeArgs;
}
} // namespace ops_hccl
