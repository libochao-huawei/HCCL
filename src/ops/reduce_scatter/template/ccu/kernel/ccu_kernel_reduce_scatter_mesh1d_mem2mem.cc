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
#include "ccu_kernel_reduce_scatter_mesh1d_mem2mem.h"

namespace ops_hccl {
using namespace hcomm;

// bit序号，每种信号用一个bit
constexpr int INPUT_XN_ID   = 0;
constexpr int SCRATCH_XN_ID = 1;
constexpr int TOKEN_XN_ID   = 2;
constexpr int POST_SYNC_ID   = 3;  
// cke序号
constexpr int CKE_IDX_0     = 0;

constexpr uint16_t BIT_NUM_PER_CKE = 16;
constexpr uint16_t GROUP_REDUCE_MAX_PIECE_CNT = 8;
constexpr uint32_t UNROLL_NUM = 16; // 最多支持8 * 16 = 128个rank

CcuKernelReduceScatterMesh1DMem2Mem::CcuKernelReduceScatterMesh1DMem2Mem(const CcuKernelArg &arg)
    : CcuKernelAlgBase(arg)
{
    const CcuKernelArgReduceScatterMesh1DMem2Mem *kernelArg
        = dynamic_cast<const CcuKernelArgReduceScatterMesh1DMem2Mem *>(&arg);
    rankId_         = kernelArg->rankId_;
    rankSize_       = kernelArg->dimSize_;
    channels_       = kernelArg->channels;
    dataType_       = kernelArg->opParam_.DataDes.dataType;
    outputDataType_ = kernelArg->opParam_.DataDes.outputType;
    if (outputDataType_ == HcclDataType::HCCL_DATA_TYPE_RESERVED) {
        outputDataType_ = dataType_;
        HCCL_DEBUG(
            "[CcuKernelReduceScatterMesh1DMem2Mem] outputDataType is [INVALID], set outputDataType to[%d]",
            outputDataType_);
    }
    reduceOp_       = kernelArg->opParam_.reduceType;
    HCCL_INFO(
        "[CcuKernelReduceScatterMesh1DMem2Mem] Init, KernelArgs are rankId[%u], rankSize_[%u], dataType[%d], "
        "outputDataType[%d], reduceOp[%d]",
        rankId_, rankSize_, dataType_, outputDataType_, reduceOp_);
}

HcclResult CcuKernelReduceScatterMesh1DMem2Mem::InitResource()
{
    uint16_t channelIdx = 0;
    if (channels_.size() == 0) {
        HCCL_ERROR("[CcuKernelReduceScatterMesh1DMem2Mem] channels is empty!");
        return HcclResult::HCCL_E_INTERNAL;
    }

    // 按照rank号从小到大遍历channels，遇到本rank就填充本地资源，否则依次取远端资源，要求给框架返回的Link同样是按顺序排列的
    for (uint64_t peerId = 0; peerId < rankSize_; peerId++) {
        if (peerId == rankId_) {
            input_.push_back(CreateVariable());
            scratch_.push_back(CreateVariable());
            token_.push_back(CreateVariable());
        } else {
            HCCL_DEBUG("[CcuKernelReduceScatterMesh1DMem2Mem] MyRank[%u], PeerId[%u], ChannelId[%u]",
                       rankId_, peerId, channelIdx);
                
            CcuRep::Variable inputVar, scratchVar, tokenVar;
            CHK_RET(CreateVariable(channels_[channelIdx], INPUT_XN_ID, &inputVar));
            input_.push_back(inputVar); // 获取channel中id=0的Var来传递output
            CHK_RET(CreateVariable(channels_[channelIdx], SCRATCH_XN_ID, &scratchVar));
            scratch_.push_back(scratchVar);
            CHK_RET(CreateVariable(channels_[channelIdx], TOKEN_XN_ID, &tokenVar));
            token_.push_back(tokenVar);
            channelIdx++;
        }
    }
    output_                      = CreateVariable();
    currentRankSliceInputOffset_ = CreateVariable();
    currentRankSliceOutputOffset_= CreateVariable();
    normalSliceSize_             = CreateVariable();
    inputRepeatStride_           = CreateVariable();
    outputRepeatStride_          = CreateVariable();
    repeatNum_                   = CreateVariable();
    lastSliceSize_               = CreateVariable();
    flag_                        = CreateVariable();
    constVar1_                   = CreateVariable();
    constVar1_                   = 1;
    readRepeatNum_               = CreateVariable();
    readRepeatNum_               = 0;
    waitRepeatNum_               = CreateVariable();
    waitRepeatNum_               = 0;
    scratchRepeatStride_         = CreateVariable();
    GoSize_                      = CreateGroupOpSize();

    remoteInput_.reserve(rankSize_);
    scratchMem_.reserve(rankSize_);
    for (uint64_t rankIdx = 0; rankIdx < rankSize_; rankIdx++) {
        scratchMem_.push_back(CreateLocalAddr());
        if (rankIdx == rankId_) {
            myInput_ = CreateLocalAddr();
            remoteInput_.push_back({});
        } else {
            remoteInput_.push_back(CreateRemoteAddr());
        }
    }

    uint32_t numEventsPerIter = (rankSize_ + BIT_NUM_PER_CKE - 1) / BIT_NUM_PER_CKE;
    for (uint32_t i = 0; i < UNROLL_NUM * numEventsPerIter; i++) {
        event_.push_back(CreateCompletedEvent());
    }
    return HcclResult::HCCL_SUCCESS;
}

void CcuKernelReduceScatterMesh1DMem2Mem::LoadArgs()
{
    Load(input_[rankId_]);
    Load(output_);
    Load(token_[rankId_]);
    Load(scratch_[rankId_]);
    Load(currentRankSliceInputOffset_);
    Load(currentRankSliceOutputOffset_);
    Load(inputRepeatStride_);
    Load(outputRepeatStride_);
    Load(normalSliceSize_);
    Load(lastSliceSize_);
    Load(scratchRepeatStride_);
    Load(repeatNum_);
    Load(GoSize_);
    return;
}

void CcuKernelReduceScatterMesh1DMem2Mem::PreSync()
{
    for (ChannelHandle channel : channels_) {
        NotifyRecord(channel, CKE_IDX_0, INPUT_XN_ID, input_[rankId_], 1 << INPUT_XN_ID);       // bit index = 1，传递input信息
        NotifyRecord(channel, CKE_IDX_0, SCRATCH_XN_ID, scratch_[rankId_], 1 << SCRATCH_XN_ID); // bit index = 2, 传递scratch信息
        NotifyRecord(channel, CKE_IDX_0, TOKEN_XN_ID, token_[rankId_], 1 << TOKEN_XN_ID);       // bit index = 3，传递output信息
    }
    uint32_t allBit = 1 << INPUT_XN_ID | 1 << SCRATCH_XN_ID | 1 << TOKEN_XN_ID;
    for (ChannelHandle channel : channels_) {
        NotifyWait(channel, CKE_IDX_0, allBit);
    }
    
    return;
}

void CcuKernelReduceScatterMesh1DMem2Mem::PostSync()
{
    for (ChannelHandle channel : channels_) {
        NotifyRecord(channel, CKE_IDX_0, 1 << POST_SYNC_ID);         // bit index = 4, 用作后同步。cke都可以用同一个，所以都是CKE_IDX_0
    }
    for (ChannelHandle channel : channels_) {
        NotifyWait(channel, CKE_IDX_0, 1 << POST_SYNC_ID);
    }
}

void CcuKernelReduceScatterMesh1DMem2Mem::DoReduceScatterRead(uint32_t unrollIdx)
{
    uint32_t channelId = 0;
    uint32_t numEventsPerIter = (rankSize_ + BIT_NUM_PER_CKE - 1) / BIT_NUM_PER_CKE;
    CcuRep::Variable sliceSize = CreateVariable();
    sliceSize = (rankId_ == (rankSize_ - 1)) ? lastSliceSize_ : normalSliceSize_;

    CCU_IF(sliceSize != 0)
    {
        for (uint32_t rankIdx = 0; rankIdx < rankSize_; rankIdx++) {
            uint32_t eventIdx = unrollIdx * numEventsPerIter + rankIdx / BIT_NUM_PER_CKE;
            event_[eventIdx].SetMask(1 << (rankIdx % BIT_NUM_PER_CKE));
            if (rankIdx == rankId_) {
                if (rankSize_ <= GROUP_REDUCE_MAX_PIECE_CNT) {
                    RecordEvent(event_[eventIdx]);
                } else {
                    LocalCopyNb(scratchMem_[rankIdx], myInput_, sliceSize, event_[eventIdx]);
                }
            } else {
                ReadNb(channels_[channelId], scratchMem_[rankIdx], remoteInput_[rankIdx], sliceSize, event_[eventIdx]);
                channelId++;
            }
        }
    }
}

void CcuKernelReduceScatterMesh1DMem2Mem::DoReduceScatterWait(uint32_t unrollIdx)
{
    uint32_t numEventsPerIter = (rankSize_ + BIT_NUM_PER_CKE - 1) / BIT_NUM_PER_CKE;
    for (uint32_t i = 0; i < numEventsPerIter; i++) {
        uint32_t eventIdx = unrollIdx * numEventsPerIter + i;
        uint32_t sigNum = BIT_NUM_PER_CKE;
        if (rankSize_ % BIT_NUM_PER_CKE != 0 && i == (numEventsPerIter - 1)) {
            sigNum = rankSize_ % BIT_NUM_PER_CKE;
        }
        event_[eventIdx].SetMask((1 << sigNum) - 1);
        WaitEvent(event_[eventIdx]);
    }
}

void CcuKernelReduceScatterMesh1DMem2Mem::DoReduceScatter()
{
    CcuRep::LocalAddr myOutput = CreateLocalAddr();
    myOutput.addr   = output_;
    myOutput.addr  += currentRankSliceOutputOffset_;
    myOutput.token  = token_[rankId_];

    CcuRep::Variable sliceSize = CreateVariable();
    sliceSize = (rankId_ == (rankSize_ - 1)) ? lastSliceSize_: normalSliceSize_;

    CCU_IF(sliceSize != 0)
    {
        if (rankSize_ <= GROUP_REDUCE_MAX_PIECE_CNT) {
            ReduceLoopGroup(myOutput, myInput_, scratchMem_, GoSize_, dataType_, outputDataType_, reduceOp_);
        } else {
            PairwiseLocalReduce(myOutput, scratchMem_, sliceSize, dataType_, outputDataType_, reduceOp_);
        }
    }
}

void CcuKernelReduceScatterMesh1DMem2Mem::PairwiseLocalReduce(CcuRep::LocalAddr myOutput, std::vector<CcuRep::LocalAddr> &inputVec,
    CcuRep::Variable sliceSize, HcclDataType dataType, HcclDataType outputDataType, HcclReduceOp opType)
{
    (void) outputDataType;
    CcuRep::Variable len = CreateVariable();

    // 每轮将数据划分为2组做规约，总规约次数log2(n)
    uint32_t remainPieces = rankSize_;
    while (remainPieces > 1) {
        // 每轮将最后remain/2块，reduce到最前remian/2块
        uint32_t reducePieces = remainPieces / 2;
        uint32_t srcIdx = remainPieces - reducePieces;
        uint32_t dstIdx = 0;
        
        len = sliceSize;
        for (uint32_t i = 0; i < reducePieces - 1; i++) {
            len += sliceSize;
        }

        event_[0].SetMask(1);
        LocalReduceNb(inputVec[dstIdx], inputVec[srcIdx], len, dataType, opType, event_[0]);
        WaitEvent(event_[0]);

        remainPieces -= reducePieces;
    }

    event_[0].SetMask(1);
    LocalCopyNb(myOutput, inputVec[0], sliceSize, event_[0]);
    WaitEvent(event_[0]);
}

void CcuKernelReduceScatterMesh1DMem2Mem::DoRepeatReduceScatter()
{
    CcuRep::Variable scratchOffset = CreateVariable();
    scratchOffset                  = 0;

    for (uint32_t rankIdx = 0; rankIdx < rankSize_; rankIdx++) {
        if (rankIdx == rankId_) {
            myInput_.addr = input_[rankIdx];
            myInput_.addr += currentRankSliceInputOffset_;
            myInput_.token = token_[rankIdx];
        } else {
            remoteInput_[rankIdx].addr = input_[rankIdx];
            remoteInput_[rankIdx].addr += currentRankSliceInputOffset_;
            remoteInput_[rankIdx].token = token_[rankIdx];
        }

        scratchMem_[rankIdx].addr = scratch_[rankId_];
        scratchMem_[rankIdx].addr += scratchOffset;
        scratchOffset += normalSliceSize_;
        scratchMem_[rankIdx].token = token_[rankId_];
    }

    // 软件展开三阶段设计（仅支持repeatNum <= UNROLL_NUM）:
    // Phase 1: 先下发所有ReadNb（非阻塞，event错开），步进scratch和input地址
    // Phase 2: 批量WaitEvent，等所有远端ReadNb数据到齐
    // Phase 3: Reduce串行（CCU_WHILE），恢复scratch/input地址后从头步进
    // 注意: scratchRepeatStride_ = rankSize_ * normalSliceSize_（每轮占rankSize片）

    // 原始Variable在三阶段中不会被修改，Phase 3前直接用它们重算初始地址
    // myInput_.addr = input_[rankId_] + currentRankSliceInputOffset_
    // scratchMem_[rankIdx].addr = scratch_[rankId_] + rankIdx*normalSliceSize_

    // 从repeatNum_派生Phase 2/3计数器
    waitRepeatNum_ += repeatNum_;
    readRepeatNum_ += repeatNum_;

    // Phase 1: 第1轮迭代（不需要地址步进）
    CCU_IF(repeatNum_ != UINT64_MAX)
    {
        repeatNum_ += constVar1_;
        DoReduceScatterRead(0);
    }

    // 第2~UNROLL_NUM轮迭代（需要地址步进）
    for (uint32_t i = 1; i < UNROLL_NUM; i++) {
        CCU_IF(repeatNum_ != UINT64_MAX)
        {
            repeatNum_ += constVar1_;
            for (uint64_t rankIdx = 0; rankIdx < rankSize_; rankIdx++) {
                if (rankIdx == rankId_) {
                    myInput_.addr += inputRepeatStride_; // LocalCopyNb源步进
                } else {
                    remoteInput_[rankIdx].addr += inputRepeatStride_; // ReadNb源步进
                }
                scratchMem_[rankIdx].addr += scratchRepeatStride_; // ReadNb目的地步进
            }
            DoReduceScatterRead(i);
        }
    }

    // Phase 2: 批量WaitEvent，等所有远端ReadNb数据到齐
    for (uint32_t i = 0; i < UNROLL_NUM; i++) {
        CCU_IF(waitRepeatNum_ != UINT64_MAX)
        {
            waitRepeatNum_ += constVar1_;
            DoReduceScatterWait(i);
        }
    }

    // 恢复myInput_/scratchMem_初始地址，为Phase 3 Reset起始位置
    // 用原始Variable重算，不依赖Address保存
    myInput_.addr = input_[rankId_];
    myInput_.addr += currentRankSliceInputOffset_;
    scratchOffset = 0;
    for (uint32_t rankIdx = 0; rankIdx < rankSize_; rankIdx++) {
        scratchMem_[rankIdx].addr = scratch_[rankId_];
        scratchMem_[rankIdx].addr += scratchOffset;
        scratchOffset += normalSliceSize_;
    }

    // Phase 3: Reduce串行，保持原有逻辑不变
    // scratch步进用scratchRepeatStride_（= rankSize * normalSliceSize）
    CCU_IF(readRepeatNum_ != UINT64_MAX)
    {
        flag_ = 0;
        CCU_WHILE(readRepeatNum_ != UINT64_MAX)
        {
            readRepeatNum_ += constVar1_;
            CCU_IF(flag_ != 0)
            {
                for (uint64_t rankIdx = 0; rankIdx < rankSize_; rankIdx++) {
                    scratchMem_[rankIdx].addr += scratchRepeatStride_; // 每轮步进rankSize片
                }
                myInput_.addr += inputRepeatStride_;
                output_ += outputRepeatStride_;
            }
            DoReduceScatter();
            flag_ = 1;
        }
    }
}

std::string CcuKernelReduceScatterMesh1DMem2Mem::GetLoopBlockTag(std::string loopType, int32_t index)
{
    return loopType + LOOP_BLOCK_TAG + std::to_string(index);
}

void CcuKernelReduceScatterMesh1DMem2Mem::CreateReduceLoop(uint32_t size, HcclDataType dataType, HcclDataType outputDataType,
    HcclReduceOp opType)
{
    constexpr uint32_t LOOP_NUM = 16;
    AllocGoResource(LOOP_NUM);
 
    std::string loopType = GetReduceTypeStr(dataType, opType);
    if (registeredLoop.find(loopType) != registeredLoop.end()) {
        return;
    }
 
    uint32_t expansionNum = GetReduceExpansionNum(opType, dataType, outputDataType);
    uint32_t usedBufNum   = size > expansionNum ? size : expansionNum;  // ?
 
    for (int32_t index = 0; index < 2; index++) { // 需要实例化2个Loop
        CcuRep::LocalAddr dst = CreateLocalAddr();
        CcuRep::LocalAddr src = CreateLocalAddr();
        std::vector<CcuRep::LocalAddr> scratch;
        for (uint32_t i = 0; i < size; i++) {
            scratch.emplace_back(CreateLocalAddr());
        }
        CcuRep::Variable            len = CreateVariable();
        CcuRep::Variable            lenForExpansion = CreateVariable();
        CcuRep::LoopBlock           lb(this, GetLoopBlockTag(loopType, index));
        lb(dst, src, scratch, len, lenForExpansion);
 
        std::vector<CcuRep::CcuBuf> bufs = {moRes.ccuBuf.begin() + index * moConfig.msInterleave,
                                            moRes.ccuBuf.begin() + index * moConfig.msInterleave + usedBufNum};
        CcuRep::CompletedEvent     event = moRes.completedEvent[index];
 
        for (uint32_t i = 0; i < size; i++) {
            event.SetMask(1 << i);
            if (i == rankId_) {
                LocalCopyNb(bufs[i], src, len, event);
            } else {
                LocalCopyNb(bufs[i], scratch[i], len, event);
            }
        }
        event.SetMask((1 << size) - 1);
        WaitEvent(event);
 
        if (size > 1) {
            event.SetMask(1);
            LocalReduceNb(bufs, size, dataType, outputDataType, opType, len, event);
            WaitEvent(event);
        }
 
        event.SetMask(1);
        LocalCopyNb(dst, bufs[0], lenForExpansion, event);
        WaitEvent(event);
    }
 
    registeredLoop.insert(loopType);
}
 
void CcuKernelReduceScatterMesh1DMem2Mem::ReduceLoopGroup(CcuRep::LocalAddr outDstOrg, CcuRep::LocalAddr srcOrg,
    std::vector<CcuRep::LocalAddr> &scratchOrg, GroupOpSize goSize, HcclDataType dataType, HcclDataType outputDataType,
    HcclReduceOp opType)
{
    const uint32_t size = scratchOrg.size();
 
    CcuRep::LocalAddr dst = CreateLocalAddr();
    dst = outDstOrg;
 
    CcuRep::LocalAddr src = CreateLocalAddr();
    src = srcOrg;
 
    std::vector<CcuRep::LocalAddr> scratch;
    for (uint32_t idx = 0; idx < size; idx++) {
        scratch.push_back(CreateLocalAddr());
        scratch[idx] = scratchOrg[idx];
    }
 
    CreateReduceLoop(size, dataType, outputDataType, opType);
 
    std::string loopType = GetReduceTypeStr(dataType, opType);
    uint32_t         expansionNum = GetReduceExpansionNum(opType, dataType, outputDataType);
    CcuRep::Variable sliceSizeExpansion = CreateVariable();
 
    if (expansionNum != 1) {
        CcuRep::Variable tmp = CreateVariable();
        tmp = GetExpansionParam(expansionNum);
        dst.token += tmp;
    }
 
    // m部分
    CCU_IF(goSize.loopParam != 0)                   // goSize1
    {
        CcuRep::Variable loopParam = CreateVariable();
        loopParam = GetLoopParam(0, moConfig.memSlice * moConfig.loopCount, 0);
        loopParam += goSize.loopParam;
 
        CcuRep::Variable sliceSize = CreateVariable();
        sliceSize          = moConfig.memSlice;
        sliceSizeExpansion = moConfig.memSlice * expansionNum;
 
        auto lc = Loop(GetLoopBlockTag(loopType, 0))(dst, src, scratch, sliceSize, sliceSizeExpansion);
 
        CcuRep::Variable paraCfg = CreateVariable();
        paraCfg = GetParallelParam(moConfig.loopCount - 1, 0, 1);
        CcuRep::Variable offsetCfg = CreateVariable();
        offsetCfg = GetOffsetParam(moConfig.memSlice, moConfig.msInterleave, 1);
 
        LoopGroup({lc}, {loopParam}, paraCfg, offsetCfg);
    }
 
    CCU_IF(goSize.parallelParam != 0)               // goSize2
    {
        // p部分，加m的偏移
        for (uint32_t i = 0; i < size; i++) {
            scratch[i].addr += goSize.addrOffset;
        }
        src.addr += goSize.addrOffset;              // goSize0
        for (uint32_t i = 0; i < expansionNum; i++) {
            dst.addr += goSize.addrOffset;
        }
 
        sliceSizeExpansion = 0;
        for (uint32_t i = 0; i < expansionNum; i++) {
            sliceSizeExpansion += goSize.residual;  // goSize3
        }
 
        auto lc0 = Loop(GetLoopBlockTag(loopType, 0))(dst, src, scratch, goSize.residual, sliceSizeExpansion);
 
        // n部分，再加p的偏移
        for (uint32_t i = 0; i < size; i++) {
            scratch[i].addr += goSize.residual;
        }
        src.addr += goSize.residual;
        for (uint32_t i = 0; i < expansionNum; i++) {
            dst.addr += goSize.residual;
        }
 
        CcuRep::Variable sliceSize = CreateVariable();
        sliceSize          = moConfig.memSlice;
        sliceSizeExpansion = moConfig.memSlice * expansionNum;
 
        auto lc1 = Loop(GetLoopBlockTag(loopType, 1))(dst, src, scratch, sliceSize, sliceSizeExpansion);
 
        CcuRep::Variable loopCfg0 = CreateVariable();
        loopCfg0 = GetLoopParam(0, 0, 1);
        CcuRep::Variable loopCfg1 = CreateVariable();
        loopCfg1 = GetLoopParam(0, 0, 1);
        CcuRep::Variable offsetCfg = CreateVariable();
        offsetCfg = GetOffsetParam(moConfig.memSlice, moConfig.msInterleave, 1);
 
        LoopGroup({lc0, lc1}, {loopCfg0, loopCfg1}, goSize.parallelParam, offsetCfg);
    }
}

HcclResult CcuKernelReduceScatterMesh1DMem2Mem::Algorithm()
{
    HCCL_INFO("[CcuKernelReduceScatterMesh1DMem2Mem] ReduceScatterMesh1DMem2Mem run");

    CHK_RET(InitResource());

    LoadArgs();

    PreSync();

    DoRepeatReduceScatter();

    PostSync();

    HCCL_INFO("[CcuKernelReduceScatterMesh1DMem2Mem] ReduceScatterMesh1DMem2Mem end");
    
    return HcclResult::HCCL_SUCCESS;
}

std::vector<uint64_t> CcuKernelReduceScatterMesh1DMem2Mem::GeneArgs(const CcuTaskArg &arg)
{
    const CcuTaskArgReduceScatterMesh1DMem2Mem *taskArg
        = dynamic_cast<const CcuTaskArgReduceScatterMesh1DMem2Mem *>(&arg);
    uint64_t inputAddr                   = taskArg->inputAddr_;
    uint64_t outputAddr                  = taskArg->outputAddr_;
    uint64_t tokenInfo                   = taskArg->token_;
    uint64_t scratchAddr                 = taskArg->scratchAddr_;
    uint64_t currentRankSliceInputOffset = taskArg->inputSliceStride_ * rankId_;
    uint64_t currentRankSliceOutputOffset= taskArg->outputSliceStride_ * rankId_;
    uint64_t inputRepeatStride           = taskArg->inputRepeatStride_;
    uint64_t outputRepeatStride          = taskArg->outputRepeatStride_;
    uint64_t normalSliceSize             = taskArg->normalSliceSize_;
    uint64_t lastSliceSize               = taskArg->lastSliceSize_;
    uint64_t scratchRepeatStride         = taskArg->scratchRepeatStride_;
    uint64_t repeatNum                   = taskArg->repeatNum_;
    auto     GoSize                      = (rankId_ == (rankSize_ - 1)) ? CalGoSize(lastSliceSize) 
                                                                       : CalGoSize(normalSliceSize);

    std::vector<uint64_t> taskArgs = {
        inputAddr,         outputAddr,         tokenInfo,
        scratchAddr,       currentRankSliceInputOffset,
        currentRankSliceOutputOffset,          inputRepeatStride,
        outputRepeatStride, normalSliceSize,   lastSliceSize,
        scratchRepeatStride, repeatNum
    };

    HCCL_INFO("[CcuKernelReduceScatterMesh1DMem2Mem] TaskArgs: inputAddr[%llu], outputAddr[%llu], "
               "scratchAddr[%llu], currentRankSliceInputOffset[%llu], currentRankSliceOutputOffset[%llu], "
               "inputRepeatStride[%llu], outputRepeatStride[%llu], "
               "normalSliceSize[%llu], lastSliceSize[%llu], scratchRepeatStride[%llu], repeatNum[%llu]",
               inputAddr, outputAddr, scratchAddr, currentRankSliceInputOffset, currentRankSliceOutputOffset,
               inputRepeatStride, outputRepeatStride, normalSliceSize, lastSliceSize, scratchRepeatStride, UINT64_MAX - repeatNum);
               
    taskArgs.insert(taskArgs.cend(), GoSize.cbegin(), GoSize.cend());
    return taskArgs;
}

} // namespace ops_hccl