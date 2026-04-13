/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "ccu_kernel_omni.h"

namespace ops_hccl {

constexpr int CKE_IDX_0   = 0;
constexpr int INPUT_XN_ID = 0;
constexpr int OUTPUT_XN_ID = 1;
constexpr int TOKEN_XN_ID = 2;
constexpr int POST_SYNC_ID = 3;

const std::string LOCAL_REDUCE_LOOP_BLOCK_TAG{"OMNI"};

CcuKernelOmni::CcuKernelOmni(const hcomm::CcuKernelArg &arg)
    : CcuKernelAlgBase(arg)
{
    const CcuKernelArgOmni *kernelArg = dynamic_cast<const CcuKernelArgOmni *>(&arg);
    if (kernelArg == nullptr) {
        HCCL_ERROR("[CcuKernelOmni] kernelArg ptr is null.");
        return;
    }

    channels_         = kernelArg->channels;
    rankId_           = kernelArg->rankId_;
    rankGroup_        = kernelArg->rankGroup_;
    ccuSendRecvInfo_  = kernelArg->sendRecvInfo_;
    handleSelfRank_  = kernelArg->handleSelfRank_;

    HCCL_INFO("rank[%u] CcuKernelOmni channels_ size %u", rankId_, channels_.size());

    // dataType_       = kernelArg->opParam_.DataDes.dataType;
    // outputDataType_ = kernelArg->opParam_.DataDes.outputType;
    // if (outputDataType_ == HcclDataType::HCCL_DATA_TYPE_RESERVED) {
    //     outputDataType_ = dataType_;
    //     HCCL_DEBUG(
    //         "[CcuKernelAllReduceMesh1DMem2Mem2DieOneShot] outputDataType is [INVALID], set outputDataType to[%d]",
    //         outputDataType_);
    // }
    // reduceOp_       = kernelArg->opParam_.reduceType;
}

void CcuKernelOmni::ReduceLoopGroup(CcuRep::LocalAddr &outDstOrg, std::vector<CcuRep::LocalAddr> &srcOrg,
                                    GroupOpSize goSize, HcclDataType dataType, HcclDataType outputDataType,
                                    HcclReduceOp opType, std::string loopName)
{
    const uint32_t size = srcOrg.size();
 
    CcuRep::LocalAddr dst = CreateLocalAddr();
    dst                = outDstOrg;
 
    std::vector<CcuRep::LocalAddr> src;
    for (uint32_t idx = 0; idx < size; idx++) {
        src.push_back(CreateLocalAddr());
        src[idx] = srcOrg[idx];
    }
 
    CreateReduceLoop(size, dataType, outputDataType, opType, loopName);
 
    std::string      loopType           = GetReduceTypeStr(dataType, opType);
    loopType = loopName + loopType;
    uint32_t         expansionNum       = GetReduceExpansionNum(opType, dataType, outputDataType);
    CcuRep::Variable sliceSizeExpansion = CreateVariable();
 
    if (expansionNum != 1) {
        CcuRep::Variable tmp = CreateVariable();
        tmp                  = GetExpansionParam(expansionNum);
        dst.token += tmp;
    }
 
    // m部分
    CCU_IF (goSize.loopParam != 0) // goSize1
    {
        CcuRep::Variable loopParam = CreateVariable();
        loopParam                  = GetLoopParam(0, moConfig.memSlice * moConfig.loopCount, 0);
        loopParam += goSize.loopParam;
 
        CcuRep::Variable sliceSize = CreateVariable();
        sliceSize                  = moConfig.memSlice;
        sliceSizeExpansion         = moConfig.memSlice * expansionNum;
        auto lc = Loop(GetLoopBlockTag(loopType, 0))(dst, src, sliceSize, sliceSizeExpansion);
 
        CcuRep::Variable paraCfg   = CreateVariable();
        paraCfg                    = GetParallelParam(moConfig.loopCount - 1, 0, 1);
        CcuRep::Variable offsetCfg = CreateVariable();
        offsetCfg                  = GetOffsetParam(moConfig.memSlice, moConfig.msInterleave, 1);
 
        LoopGroup({lc}, {loopParam}, paraCfg, offsetCfg);
    }
 
    CCU_IF (goSize.parallelParam != 0) // goSize2
    {
        // p部分，加m的偏移
        for (uint32_t i = 0; i < size; i++) {
            src[i].addr += goSize.addrOffset;
        }
        for (uint32_t i = 0; i < expansionNum; i++) {
            dst.addr += goSize.addrOffset;
        }
 
        sliceSizeExpansion = 0;
        for (uint32_t i = 0; i < expansionNum; i++) {
            sliceSizeExpansion += goSize.residual; // goSize3
        }
 
        auto lc0 = Loop(GetLoopBlockTag(loopType, 0))(dst, src, goSize.residual, sliceSizeExpansion);
 
        // n部分，再加p的偏移
        for (uint32_t i = 0; i < size; i++) {
            src[i].addr += goSize.residual;
        }
 
        for (uint32_t i = 0; i < expansionNum; i++) {
            dst.addr += goSize.residual;
        }
 
        CcuRep::Variable sliceSize = CreateVariable();
        sliceSize                  = moConfig.memSlice;
        sliceSizeExpansion         = moConfig.memSlice * expansionNum;
 
        auto lc1 = Loop(GetLoopBlockTag(loopType, 1))(dst, src, sliceSize, sliceSizeExpansion);
 
        CcuRep::Variable loopCfg0  = CreateVariable();
        loopCfg0                   = GetLoopParam(0, 0, 1);
        CcuRep::Variable loopCfg1  = CreateVariable();
        loopCfg1                   = GetLoopParam(0, 0, 1);
        CcuRep::Variable offsetCfg = CreateVariable();
        offsetCfg                  = GetOffsetParam(moConfig.memSlice, moConfig.msInterleave, 1);
 
        LoopGroup({lc0, lc1}, {loopCfg0, loopCfg1}, goSize.parallelParam, offsetCfg);
    }
}
 
void CcuKernelOmni::CreateReduceLoop(uint32_t size, HcclDataType dataType, HcclDataType outputDataType,
                                                         HcclReduceOp opType, std::string loopName)
{
    constexpr uint32_t LOOP_NUM = 16;
    AllocGoResource(LOOP_NUM);
    
    std::string loopType = GetReduceTypeStr(dataType, opType);
    loopType = loopName + loopType;
    if (registeredLoop.find(loopType) != registeredLoop.end()) {
        return;
    }
 
    uint32_t expansionNum = GetReduceExpansionNum(opType, dataType, outputDataType);
    uint32_t usedBufNum   = size > expansionNum ? size : expansionNum;
 
    for (int32_t index = 0; index < 2; index++) { // 需要实例化2个Loop
        CcuRep::LocalAddr              dst = CreateLocalAddr();
        std::vector<CcuRep::LocalAddr> src;
        for (uint32_t i = 0; i < size; i++) {
            src.emplace_back(CreateLocalAddr());
        }
        CcuRep::Variable  len             = CreateVariable();
        CcuRep::Variable  lenForExpansion = CreateVariable();
        CcuRep::LoopBlock lb(this, GetLoopBlockTag(loopType, index));
        lb(dst, src, len, lenForExpansion);
 
        std::vector<CcuRep::CcuBuf> bufs = {moRes.ccuBuf.begin() + index * moConfig.msInterleave,
                                               moRes.ccuBuf.begin() + index * moConfig.msInterleave + usedBufNum};
        CcuRep::CompletedEvent &event = moRes.completedEvent[index];
 
        for (uint32_t i = 0; i < size; i++) {
            event.SetMask(1 << i);
            LocalCopyNb(bufs[i], src[i], len, event);
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
 
std::string CcuKernelOmni::GetLoopBlockTag(std::string loopType, int32_t index)
{
    return loopType + LOCAL_REDUCE_LOOP_BLOCK_TAG + std::to_string(index);
}
 

HcclResult CcuKernelOmni::InitResources()
{
    // 创建Variable，用于交换地址及token
    if (channels_.size() == 0) {
        HCCL_ERROR("[CcuKernelOmni] RankId[%u] channels_ is empty", rankId_);
        return HcclResult::HCCL_E_INTERNAL;
    }

    uint32_t idx = 0;
    for (uint32_t i = 0; i < rankGroup_.size(); i++) {

        HCCL_DEBUG("rankId_ [%u], rankGroup_[%u] is [%u]", rankId_, i, rankGroup_[i]);

        if (rankGroup_[i] == rankId_) {
            input_.insert(input_.begin() + rankId_, CreateVariable());
            output_.insert(output_.begin() + rankId_, CreateVariable());
            token_.insert(token_.begin() + rankId_, CreateVariable());
        } else {
            rankId2Channel_[rankGroup_[i]] = channels_[idx];

            CcuRep::Variable inputVar, outputVar, tokenVar;
            CHK_RET(CreateVariable(channels_[idx], INPUT_XN_ID, &inputVar));
            input_.push_back(inputVar);
            CHK_RET(CreateVariable(channels_[idx], OUTPUT_XN_ID, &outputVar));
            output_.push_back(outputVar);
            CHK_RET(CreateVariable(channels_[idx], TOKEN_XN_ID, &tokenVar));
            token_.push_back(tokenVar);
            idx++;
        }
    }

    scratchAddr_                 = CreateVariable();
    repeatNum_                   = CreateVariable();
    inputRepeatStride_           = CreateVariable();
    outputRepeatStride_          = CreateVariable();
    groupOpSize_                 = CreateGroupOpSize();
    sliceSize_                   = CreateVariable();
    event_                       = CreateCompletedEvent();
    
    return HcclResult::HCCL_SUCCESS;
}

void CcuKernelOmni::LoadArgs()
{
    HCCL_INFO("LoadArgs begin rankid is %u", rankId_);

    Load(input_[rankId_]);
    Load(output_[rankId_]);
    Load(scratchAddr_);
    Load(token_[rankId_]);
    Load(sliceSize_);
    Load(repeatNum_);
    Load(inputRepeatStride_);
    Load(outputRepeatStride_);
    Load(groupOpSize_);

    HCCL_INFO("LoadArgs end rankid is %u", rankId_);
}

void CcuKernelOmni::PreSync()
{
    for (ChannelHandle channel : channels_) {
        NotifyRecord(channel, CKE_IDX_0, OUTPUT_XN_ID, output_[rankId_], 1 << OUTPUT_XN_ID);
        NotifyRecord(channel, CKE_IDX_0, TOKEN_XN_ID, token_[rankId_], 1 << TOKEN_XN_ID);
    }
    uint32_t waitBits = (1 << OUTPUT_XN_ID) | (1 << TOKEN_XN_ID);
    for (const ChannelHandle &channel : channels_) {
        NotifyWait(channel, CKE_IDX_0, waitBits);
    }
    return;
}

void CcuKernelOmni::PostSync()
{
    for (const auto &channel : channels_) {
        NotifyRecord(channel, CKE_IDX_0, 1 << POST_SYNC_ID);
    }
    for (const auto &channel : channels_) {
        NotifyWait(channel, CKE_IDX_0, 1 << POST_SYNC_ID);
    }
    return;
}

void CcuKernelOmni::DoRepeatOmni()
{
    for (auto& signalInfo : ccuSendRecvInfo_) {
        if (signalInfo.optype == OP_LOCAL_COPY) { // groupCopy
            HCCL_DEBUG("groupCopy begin");
            CcuRep::LocalAddr myOutput = CreateLocalAddr();
            CcuRep::LocalAddr myInput  = CreateLocalAddr();
            myInput.token = token_[rankId_];
            myOutput.token = token_[rankId_];

            if (signalInfo.srcSliceInfo.size() > 1 || signalInfo.dstSliceInfo.size() > 1) {
                HCCL_WARNING("[DoRepeatOmni] groupCopy slicesize > 1");
            }

            for (auto& sliceInfo : signalInfo.srcSliceInfo) {
                if (sliceInfo.sliceType == BufferTypeTmp::INPUT) { 
                    myInput.addr = input_[sliceInfo.remoteRank];                
                }

                if (sliceInfo.sliceType == BufferTypeTmp::OUTPUT) {
                    myInput.addr = output_[sliceInfo.remoteRank];
                }

                if (sliceInfo.sliceType == BufferTypeTmp::HCCL_BUFFER) {
                    myInput.addr = scratchAddr_;
                }

                for (uint64_t i = 0; i < sliceInfo.sliceIdx; i++) {
                    myInput.addr += sliceSize_; // 一个sliceId 对应一份sliceSize
                }
            }

            for (auto& sliceInfo : signalInfo.dstSliceInfo) {
                if (sliceInfo.sliceType == BufferTypeTmp::INPUT) { // input
                    myOutput.addr = input_[sliceInfo.remoteRank];                
                }

                if (sliceInfo.sliceType == BufferTypeTmp::OUTPUT) {
                    myOutput.addr = output_[sliceInfo.remoteRank];
                }

                if (sliceInfo.sliceType == BufferTypeTmp::HCCL_BUFFER) {
                    myOutput.addr = scratchAddr_;
                }

                for (uint64_t i = 0; i < sliceInfo.sliceIdx; i++) {
                    myOutput.addr += sliceSize_; // 一个sliceId 对应一份sliceSize
                }
            }
    
            GroupCopy(myOutput, myInput, groupOpSize_);
            HCCL_DEBUG("groupCopy end");
        }

        if (signalInfo.optype == OP_LOCAL_REDUCE) { // ReduceLoopGroup
            CcuRep::LocalAddr myOutput = CreateLocalAddr();
            std::vector<CcuRep::LocalAddr> myInput;
            myOutput.token = token_[rankId_];

            if (signalInfo.dstSliceInfo.size() > 1) {
                HCCL_WARNING("[DoRepeatOmni] groupCopy dstSliceInfo_ > 1");
            }

            for (auto& sliceInfo : signalInfo.srcSliceInfo) {
                CcuRep::LocalAddr tmpInput = CreateLocalAddr();
                tmpInput.token = token_[sliceInfo.remoteRank];
                if (sliceInfo.sliceType == BufferTypeTmp::INPUT) { // input
                    tmpInput.addr = input_[sliceInfo.remoteRank];
                }

                if (sliceInfo.sliceType == BufferTypeTmp::OUTPUT) {
                    tmpInput.addr = output_[sliceInfo.remoteRank];
                }

                if (sliceInfo.sliceType == BufferTypeTmp::HCCL_BUFFER) {
                    tmpInput.addr = scratchAddr_;
                }

                for (uint64_t i = 0; i < sliceInfo.sliceIdx; i++) {
                    tmpInput.addr += sliceSize_; // 一个sliceId 对应一份sliceSize
                }

                myInput.push_back(tmpInput);
            }

            for (auto& sliceInfo : signalInfo.dstSliceInfo) {
                if (sliceInfo.sliceType == BufferTypeTmp::INPUT) { // input
                    myOutput.addr = input_[sliceInfo.remoteRank];                
                }

                if (sliceInfo.sliceType == BufferTypeTmp::OUTPUT) {
                    myOutput.addr = output_[sliceInfo.remoteRank];
                }

                if (sliceInfo.sliceType == BufferTypeTmp::HCCL_BUFFER) {
                    myOutput.addr = scratchAddr_;
                }

                for (uint64_t i = 0; i < sliceInfo.sliceIdx; i++) {
                    myOutput.addr += sliceSize_; // 一个sliceId 对应一份sliceSize
                }
            }

            ReduceLoopGroup(myOutput, myInput, groupOpSize_, signalInfo.inputDataType, signalInfo.outputDataType, signalInfo.reduceType, "OMNI");            
        }

        if (signalInfo.optype == OP_SEND_RECV_WRITE) { // WriteNb
            for (uint32_t i = 0; i < signalInfo.srcSliceInfo.size(); i++) {
                CcuRep::RemoteAddr remoteDst = CreateRemoteAddr();
                CcuRep::LocalAddr src  = CreateLocalAddr();
                src.token = token_[signalInfo.srcSliceInfo[i].remoteRank];
                remoteDst.token = token_[signalInfo.dstSliceInfo[i].remoteRank];

                if (signalInfo.srcSliceInfo[i].sliceType == BufferTypeTmp::INPUT) {
                    src.addr = input_[signalInfo.srcSliceInfo[i].remoteRank];                
                }

                if (signalInfo.srcSliceInfo[i].sliceType == BufferTypeTmp::OUTPUT) {
                    src.addr = output_[signalInfo.srcSliceInfo[i].remoteRank];
                }

                if (signalInfo.srcSliceInfo[i].sliceType == BufferTypeTmp::HCCL_BUFFER) {
                    src.addr = scratchAddr_;
                }

                for (uint32_t j = 0; j < signalInfo.srcSliceInfo[i].sliceIdx; j++) {
                    src.addr += sliceSize_; // 一个sliceId 对应一份sliceSize
                }

                if (signalInfo.dstSliceInfo[i].sliceType == BufferTypeTmp::INPUT) {
                    remoteDst.addr = input_[signalInfo.dstSliceInfo[i].remoteRank];                
                }

                if (signalInfo.dstSliceInfo[i].sliceType == BufferTypeTmp::OUTPUT) {
                    remoteDst.addr = output_[signalInfo.dstSliceInfo[i].remoteRank];
                }

                if (signalInfo.dstSliceInfo[i].sliceType == BufferTypeTmp::HCCL_BUFFER) {
                    remoteDst.addr = scratchAddr_;
                }

                for (uint32_t j = 0; j < signalInfo.dstSliceInfo[i].sliceIdx; j++) {
                    remoteDst.addr += sliceSize_; // 一个sliceId 对应一份sliceSize
                }

                event_.SetMask(1 << rankId_);
                WriteNb(rankId2Channel_[signalInfo.dstSliceInfo[i].remoteRank], remoteDst, src, sliceSize_, event_);
                WaitEvent(event_);
            }
        }

        if (signalInfo.optype == OP_SEND_WRITE) { // WriteNb
            for (uint32_t i = 0; i < signalInfo.srcSliceInfo.size(); i++) {
                CcuRep::RemoteAddr remoteDst = CreateRemoteAddr();
                CcuRep::LocalAddr src  = CreateLocalAddr();
                src.token = token_[signalInfo.srcSliceInfo[i].remoteRank];
                remoteDst.token = token_[signalInfo.dstSliceInfo[i].remoteRank];

                if (signalInfo.srcSliceInfo[i].sliceType == BufferTypeTmp::INPUT) { // input
                    src.addr = input_[signalInfo.srcSliceInfo[i].remoteRank];                
                }

                if (signalInfo.srcSliceInfo[i].sliceType == BufferTypeTmp::OUTPUT) {
                    src.addr = output_[signalInfo.srcSliceInfo[i].remoteRank];
                }

                if (signalInfo.srcSliceInfo[i].sliceType == BufferTypeTmp::HCCL_BUFFER) {
                    src.addr = scratchAddr_;
                }

                for (uint32_t j = 0; j < signalInfo.srcSliceInfo[i].sliceIdx; j++) {
                    src.addr += sliceSize_; // 一个sliceId 对应一份sliceSize
                }

                if (signalInfo.dstSliceInfo[i].sliceType == BufferTypeTmp::INPUT) { // input
                    remoteDst.addr = input_[signalInfo.dstSliceInfo[i].remoteRank];                
                }

                if (signalInfo.dstSliceInfo[i].sliceType == BufferTypeTmp::OUTPUT) {
                    remoteDst.addr = output_[signalInfo.dstSliceInfo[i].remoteRank];
                }

                if (signalInfo.dstSliceInfo[i].sliceType == BufferTypeTmp::HCCL_BUFFER) {
                    remoteDst.addr = scratchAddr_;
                }

                for (uint32_t j = 0; j < signalInfo.dstSliceInfo[i].sliceIdx; j++) {
                    remoteDst.addr += sliceSize_; // 一个sliceId 对应一份sliceSize
                }

                event_.SetMask(1 << rankId_);
                WriteNb(rankId2Channel_[signalInfo.dstSliceInfo[i].remoteRank], remoteDst, src, sliceSize_, event_);
                WaitEvent(event_);
            }
        }

        if (signalInfo.optype == OP_RECV_WRITE) { // recvWrite
            // ccu 没有对应的函数
        }

        if (signalInfo.optype == OP_SEND_RECV_WRITE_REDUCE) { // WriteReduceNb
            for (uint32_t i = 0; i < signalInfo.srcSliceInfo.size(); i++) {
                CcuRep::RemoteAddr remoteDst = CreateRemoteAddr();
                CcuRep::LocalAddr src  = CreateLocalAddr();
                src.token = token_[signalInfo.srcSliceInfo[i].remoteRank];
                remoteDst.token = token_[signalInfo.dstSliceInfo[i].remoteRank];

                if (signalInfo.srcSliceInfo[i].sliceType == BufferTypeTmp::INPUT) { // input
                    src.addr = input_[signalInfo.srcSliceInfo[i].remoteRank];                
                }

                if (signalInfo.srcSliceInfo[i].sliceType == BufferTypeTmp::OUTPUT) {
                    src.addr = output_[signalInfo.srcSliceInfo[i].remoteRank];
                }

                if (signalInfo.srcSliceInfo[i].sliceType == BufferTypeTmp::HCCL_BUFFER) {
                    src.addr = scratchAddr_;
                }

                for (uint32_t j = 0; j < signalInfo.srcSliceInfo[i].sliceIdx; j++) {
                    src.addr += sliceSize_; // 一个sliceId 对应一份sliceSize
                }

                if (signalInfo.dstSliceInfo[i].sliceType == BufferTypeTmp::INPUT) { // input
                    remoteDst.addr = input_[signalInfo.dstSliceInfo[i].remoteRank];                
                }

                if (signalInfo.dstSliceInfo[i].sliceType == BufferTypeTmp::OUTPUT) {
                    remoteDst.addr = output_[signalInfo.dstSliceInfo[i].remoteRank];
                }

                if (signalInfo.dstSliceInfo[i].sliceType == BufferTypeTmp::HCCL_BUFFER) {
                    remoteDst.addr = scratchAddr_;
                }

                for (uint32_t j = 0; j < signalInfo.dstSliceInfo[i].sliceIdx; j++) {
                    remoteDst.addr += sliceSize_; // 一个sliceId 对应一份sliceSize
                }

                event_.SetMask(1 << rankId_);
                WriteReduceNb(rankId2Channel_[signalInfo.dstSliceInfo[i].remoteRank], remoteDst, src, sliceSize_, signalInfo.inputDataType, signalInfo.reduceType, event_);
                WaitEvent(event_);
            }
        }

        if (signalInfo.optype == OP_SEND_WRITE_REDUCE) { // WriteReduceNb
            for (uint32_t i = 0; i < signalInfo.srcSliceInfo.size(); i++) {
                CcuRep::RemoteAddr remoteDst = CreateRemoteAddr();
                CcuRep::LocalAddr src  = CreateLocalAddr();
                src.token = token_[signalInfo.srcSliceInfo[i].remoteRank];
                remoteDst.token = token_[signalInfo.dstSliceInfo[i].remoteRank];

                if (signalInfo.srcSliceInfo[i].sliceType == BufferTypeTmp::INPUT) {
                    src.addr = input_[signalInfo.srcSliceInfo[i].remoteRank];                
                }

                if (signalInfo.srcSliceInfo[i].sliceType == BufferTypeTmp::OUTPUT) {
                    src.addr = output_[signalInfo.srcSliceInfo[i].remoteRank];
                }

                if (signalInfo.srcSliceInfo[i].sliceType == BufferTypeTmp::HCCL_BUFFER) {
                    src.addr = scratchAddr_;
                }

                for (uint32_t j = 0; j < signalInfo.srcSliceInfo[i].sliceIdx; j++) {
                    src.addr += sliceSize_; // 一个sliceId 对应一份sliceSize
                }

                if (signalInfo.dstSliceInfo[i].sliceType == BufferTypeTmp::INPUT) {
                    remoteDst.addr = input_[signalInfo.dstSliceInfo[i].remoteRank];                
                }

                if (signalInfo.dstSliceInfo[i].sliceType == BufferTypeTmp::OUTPUT) {
                    remoteDst.addr = output_[signalInfo.dstSliceInfo[i].remoteRank];
                }

                if (signalInfo.dstSliceInfo[i].sliceType == BufferTypeTmp::HCCL_BUFFER) {
                    remoteDst.addr = scratchAddr_;
                }

                for (uint32_t j = 0; j < signalInfo.dstSliceInfo[i].sliceIdx; j++) {
                    remoteDst.addr += sliceSize_; // 一个sliceId 对应一份sliceSize
                }

                event_.SetMask(1 << rankId_);
                WriteReduceNb(rankId2Channel_[signalInfo.dstSliceInfo[i].remoteRank], remoteDst, src, sliceSize_, signalInfo.inputDataType, signalInfo.reduceType, event_);
                WaitEvent(event_);
            }
        }

        if (signalInfo.optype == OP_RECV_WRITE_REDUCE) { // OP_RECV_WRITE_REDUCE
            // ccu 没有对应的函数
        }

        if (signalInfo.optype == OP_SEND_RECV_READ) { // ReadNb
            for (uint32_t i = 0; i < signalInfo.srcSliceInfo.size(); i++) {
                CcuRep::RemoteAddr remoteDst = CreateRemoteAddr();
                CcuRep::LocalAddr src  = CreateLocalAddr();
                src.token = token_[signalInfo.srcSliceInfo[i].remoteRank];
                remoteDst.token = token_[signalInfo.dstSliceInfo[i].remoteRank];

                if (signalInfo.srcSliceInfo[i].sliceType == BufferTypeTmp::INPUT) {
                    src.addr = input_[signalInfo.srcSliceInfo[i].remoteRank];                
                }

                if (signalInfo.srcSliceInfo[i].sliceType == BufferTypeTmp::OUTPUT) {
                    src.addr = output_[signalInfo.srcSliceInfo[i].remoteRank];
                }

                if (signalInfo.srcSliceInfo[i].sliceType == BufferTypeTmp::HCCL_BUFFER) {
                    src.addr = scratchAddr_;
                }

                for (uint32_t j = 0; j < signalInfo.srcSliceInfo[i].sliceIdx; j++) {
                    src.addr += sliceSize_; // 一个sliceId 对应一份sliceSize
                }

                if (signalInfo.dstSliceInfo[i].sliceType == BufferTypeTmp::INPUT) {
                    remoteDst.addr = input_[signalInfo.dstSliceInfo[i].remoteRank];                
                }

                if (signalInfo.dstSliceInfo[i].sliceType == BufferTypeTmp::OUTPUT) {
                    remoteDst.addr = output_[signalInfo.dstSliceInfo[i].remoteRank];
                }

                if (signalInfo.dstSliceInfo[i].sliceType == BufferTypeTmp::HCCL_BUFFER) {
                    remoteDst.addr = scratchAddr_;
                }

                for (uint32_t j = 0; j < signalInfo.dstSliceInfo[i].sliceIdx; j++) {
                    remoteDst.addr += sliceSize_; // 一个sliceId 对应一份sliceSize
                }

                event_.SetMask(1 << rankId_);
                ReadNb(rankId2Channel_[signalInfo.dstSliceInfo[i].remoteRank], src, remoteDst, sliceSize_, event_);
                WaitEvent(event_);
            }
        }

        if (signalInfo.optype == OP_SEND_READ) { // OP_SEND_READ
            // ccu 没有对应的函数
        }

        if (signalInfo.optype == OP_RECV_READ) { // ReadNb
            for (uint32_t i = 0; i < signalInfo.srcSliceInfo.size(); i++) {
                CcuRep::RemoteAddr remoteDst = CreateRemoteAddr();
                CcuRep::LocalAddr src  = CreateLocalAddr();
                src.token = token_[signalInfo.srcSliceInfo[i].remoteRank];
                remoteDst.token = token_[signalInfo.dstSliceInfo[i].remoteRank];

                if (signalInfo.srcSliceInfo[i].sliceType == BufferTypeTmp::INPUT) {
                    src.addr = input_[signalInfo.srcSliceInfo[i].remoteRank];                
                }

                if (signalInfo.srcSliceInfo[i].sliceType == BufferTypeTmp::OUTPUT) {
                    src.addr = output_[signalInfo.srcSliceInfo[i].remoteRank];
                }

                if (signalInfo.srcSliceInfo[i].sliceType == BufferTypeTmp::HCCL_BUFFER) {
                    src.addr = scratchAddr_;
                }

                for (uint32_t j = 0; j < signalInfo.srcSliceInfo[i].sliceIdx; j++) {
                    src.addr += sliceSize_; // 一个sliceId 对应一份sliceSize
                }

                if (signalInfo.dstSliceInfo[i].sliceType == BufferTypeTmp::INPUT) {
                    remoteDst.addr = input_[signalInfo.dstSliceInfo[i].remoteRank];                
                }

                if (signalInfo.dstSliceInfo[i].sliceType == BufferTypeTmp::OUTPUT) {
                    remoteDst.addr = output_[signalInfo.dstSliceInfo[i].remoteRank];
                }

                if (signalInfo.dstSliceInfo[i].sliceType == BufferTypeTmp::HCCL_BUFFER) {
                    remoteDst.addr = scratchAddr_;
                }

                for (uint32_t j = 0; j < signalInfo.dstSliceInfo[i].sliceIdx; j++) {
                    remoteDst.addr += sliceSize_; // 一个sliceId 对应一份sliceSize
                }

                event_.SetMask(1 << rankId_);
                ReadNb(rankId2Channel_[signalInfo.dstSliceInfo[i].remoteRank], src, remoteDst, sliceSize_, event_);
                WaitEvent(event_);
            }
        }

        if (signalInfo.optype == OP_SEND_RECV_READ_REDUCE) { // ReadReduceNb
            for (uint32_t i = 0; i < signalInfo.srcSliceInfo.size(); i++) {
                CcuRep::RemoteAddr remoteDst = CreateRemoteAddr();
                CcuRep::LocalAddr src  = CreateLocalAddr();
                src.token = token_[signalInfo.srcSliceInfo[i].remoteRank];
                remoteDst.token = token_[signalInfo.dstSliceInfo[i].remoteRank];

                if (signalInfo.srcSliceInfo[i].sliceType == BufferTypeTmp::INPUT) {
                    src.addr = input_[rankId_];                
                }

                if (signalInfo.srcSliceInfo[i].sliceType == BufferTypeTmp::OUTPUT) {
                    src.addr = output_[rankId_];
                }

                if (signalInfo.srcSliceInfo[i].sliceType == BufferTypeTmp::HCCL_BUFFER) {
                    src.addr = scratchAddr_;
                }

                for (uint32_t j = 0; j < signalInfo.srcSliceInfo[i].sliceIdx; j++) {
                    src.addr += sliceSize_; // 一个sliceId 对应一份sliceSize
                }

                if (signalInfo.dstSliceInfo[i].sliceType == BufferTypeTmp::INPUT) {
                    remoteDst.addr = input_[signalInfo.dstSliceInfo[i].remoteRank];                
                }

                if (signalInfo.dstSliceInfo[i].sliceType == BufferTypeTmp::OUTPUT) {
                    remoteDst.addr = output_[signalInfo.dstSliceInfo[i].remoteRank];
                }

                if (signalInfo.dstSliceInfo[i].sliceType == BufferTypeTmp::HCCL_BUFFER) {
                    remoteDst.addr = scratchAddr_;
                }

                for (uint32_t j = 0; j < signalInfo.dstSliceInfo[i].sliceIdx; j++) {
                    remoteDst.addr += sliceSize_; // 一个sliceId 对应一份sliceSize
                }

                event_.SetMask(1 << rankId_);
                ReadReduceNb(rankId2Channel_[signalInfo.dstSliceInfo[i].remoteRank], src, remoteDst, sliceSize_, signalInfo.inputDataType, signalInfo.reduceType, event_);
                WaitEvent(event_);
            }
        }

        if (signalInfo.optype == OP_SEND_READ_REDUCE) { // OP_SEND_READ_REDUCE
            // ccu 没有对应的函数
        }

        if (signalInfo.optype == OP_RECV_READ_REDUCE) { // ReadReduceNb
            for (uint32_t i = 0; i < signalInfo.srcSliceInfo.size(); i++) {
                CcuRep::RemoteAddr remoteDst = CreateRemoteAddr();
                CcuRep::LocalAddr src  = CreateLocalAddr();
                src.token = token_[signalInfo.srcSliceInfo[i].remoteRank];
                remoteDst.token = token_[signalInfo.dstSliceInfo[i].remoteRank];

                if (signalInfo.srcSliceInfo[i].sliceType == BufferTypeTmp::INPUT) {
                    src.addr = input_[signalInfo.srcSliceInfo[i].remoteRank];                
                }

                if (signalInfo.srcSliceInfo[i].sliceType == BufferTypeTmp::OUTPUT) {
                    src.addr = output_[signalInfo.srcSliceInfo[i].remoteRank];
                }

                if (signalInfo.srcSliceInfo[i].sliceType == BufferTypeTmp::HCCL_BUFFER) {
                    src.addr = scratchAddr_;
                }

                for (uint32_t j = 0; j < signalInfo.srcSliceInfo[i].sliceIdx; j++) {
                    src.addr += sliceSize_; // 一个sliceId 对应一份sliceSize
                }

                if (signalInfo.dstSliceInfo[i].sliceType == BufferTypeTmp::INPUT) {
                    remoteDst.addr = input_[signalInfo.dstSliceInfo[i].remoteRank];                
                }

                if (signalInfo.dstSliceInfo[i].sliceType == BufferTypeTmp::OUTPUT) {
                    remoteDst.addr = output_[signalInfo.dstSliceInfo[i].remoteRank];
                }

                if (signalInfo.dstSliceInfo[i].sliceType == BufferTypeTmp::HCCL_BUFFER) {
                    remoteDst.addr = scratchAddr_;
                }

                for (uint32_t j = 0; j < signalInfo.dstSliceInfo[i].sliceIdx; j++) {
                    remoteDst.addr += sliceSize_; // 一个sliceId 对应一份sliceSize
                }

                event_.SetMask(1 << rankId_);
                ReadReduceNb(rankId2Channel_[signalInfo.dstSliceInfo[i].remoteRank], src, remoteDst, sliceSize_, signalInfo.inputDataType, signalInfo.reduceType, event_);
                WaitEvent(event_);
            }
        }

        if (signalInfo.optype == OP_GROUP_BROAD_CAST) { // GroupBroadcast
            if (signalInfo.srcSliceInfo.size() > 1) {
                HCCL_WARNING("[DoRepeatOmni] GroupBroadcast srcSliceInfo > 1");
            }

            CcuRep::LocalAddr src = CreateLocalAddr();
            std::vector<CcuRep::RemoteAddr> dst;
            src.token = token_[signalInfo.srcSliceInfo[0].remoteRank];

            for (auto& sliceInfo : signalInfo.srcSliceInfo) {
                if (sliceInfo.sliceType == BufferTypeTmp::INPUT) {
                    src.addr = input_[sliceInfo.remoteRank];                
                }

                if (sliceInfo.sliceType == BufferTypeTmp::OUTPUT) {
                    src.addr = output_[sliceInfo.remoteRank];
                }

                if (sliceInfo.sliceType == BufferTypeTmp::HCCL_BUFFER) {
                    src.addr = scratchAddr_;
                }

                for (uint64_t i = 0; i < sliceInfo.sliceIdx; i++) {
                    src.addr += sliceSize_; // 一个sliceId 对应一份sliceSize
                }
            }
            
            std::vector<ChannelHandle> channels;
            for (auto& sliceInfo : signalInfo.dstSliceInfo) {
                CcuRep::RemoteAddr tmpdst = CreateRemoteAddr();
                tmpdst.token = token_[sliceInfo.remoteRank];
                if (sliceInfo.sliceType == BufferTypeTmp::INPUT) {
                    tmpdst.addr = input_[sliceInfo.remoteRank];
                }

                if (sliceInfo.sliceType == BufferTypeTmp::OUTPUT) {
                    tmpdst.addr = output_[sliceInfo.remoteRank];
                }

                if (sliceInfo.sliceType == BufferTypeTmp::HCCL_BUFFER) {
                    tmpdst.addr = scratchAddr_;
                }

                for (uint64_t i = 0; i < sliceInfo.sliceIdx; i++) {
                    tmpdst.addr += sliceSize_; // 一个sliceId 对应一份sliceSize
                }

                dst.push_back(tmpdst);
                channels.push_back(rankId2Channel_[sliceInfo.remoteRank]);
            }

            // 添加本卡信息加到最后
            CcuRep::RemoteAddr tmpdst = CreateRemoteAddr();
            tmpdst.token = token_[rankId_];

            if (signalInfo.dstSliceInfo[0].sliceType == BufferTypeTmp::INPUT) {
                tmpdst.addr = input_[rankId_];
            }

            if (signalInfo.dstSliceInfo[0].sliceType == BufferTypeTmp::OUTPUT) {
                tmpdst.addr = output_[rankId_];
            }

            if (signalInfo.dstSliceInfo[0].sliceType == BufferTypeTmp::HCCL_BUFFER) {
                tmpdst.addr = scratchAddr_;
            }

            for (uint64_t i = 0; i < signalInfo.srcSliceInfo[0].sliceIdx; i++) {
                tmpdst.addr += sliceSize_; // 一个sliceId 对应一份sliceSize
            }

            dst.push_back(tmpdst);

            //self defined broadcast
            if (handleSelfRank_) {
                GroupBroadcast(channels, dst, src, groupOpSize_);
            } else {
                GroupBroadcastWithoutMyRank(channels, dst, src, groupOpSize_);
            }
        }

        if (signalInfo.optype == OP_GROUP_REDUCE) { // GroupReduce
            if (signalInfo.dstSliceInfo.size() > 1) {
                HCCL_WARNING("[DoRepeatOmni] GroupReduce dstSliceInfo > 1");
            }

            std::vector<CcuRep::RemoteAddr> src;
            CcuRep::LocalAddr dst = CreateLocalAddr();
            dst.token = token_[rankId_];

            for (auto& sliceInfo : signalInfo.dstSliceInfo) {
                if (sliceInfo.sliceType == BufferTypeTmp::INPUT) {
                    dst.addr = input_[sliceInfo.remoteRank];                
                }

                if (sliceInfo.sliceType == BufferTypeTmp::OUTPUT) {
                    dst.addr = output_[sliceInfo.remoteRank];
                }

                if (sliceInfo.sliceType == BufferTypeTmp::HCCL_BUFFER) {
                    dst.addr = scratchAddr_;
                }

                for (uint64_t i = 0; i < sliceInfo.sliceIdx; i++) {
                    dst.addr += sliceSize_; // 一个sliceId 对应一份sliceSize
                }
            }
            
            std::vector<ChannelHandle> channels;
            for (auto& sliceInfo : signalInfo.srcSliceInfo) {
                CcuRep::RemoteAddr tmpdst = CreateRemoteAddr();
                tmpdst.token = token_[sliceInfo.remoteRank];
                if (sliceInfo.sliceType == BufferTypeTmp::INPUT) {
                    tmpdst.addr = input_[sliceInfo.remoteRank];
                }

                if (sliceInfo.sliceType == BufferTypeTmp::OUTPUT) {
                    tmpdst.addr = output_[sliceInfo.remoteRank];
                }

                if (sliceInfo.sliceType == BufferTypeTmp::HCCL_BUFFER) {
                    tmpdst.addr = scratchAddr_;
                }

                for (uint64_t i = 0; i < sliceInfo.sliceIdx; i++) {
                    tmpdst.addr += sliceSize_; // 一个sliceId 对应一份sliceSize
                }

                src.push_back(tmpdst);
                channels.push_back(rankId2Channel_[sliceInfo.remoteRank]);
            }

            if (handleSelfRank_) {
                // 添加本卡信息加到最后
                CcuRep::RemoteAddr tmpsrc = CreateRemoteAddr();
                tmpsrc.token = token_[rankId_];

                if (signalInfo.dstSliceInfo[0].sliceType == BufferTypeTmp::INPUT) {
                    tmpsrc.addr = input_[rankId_];
                }

                if (signalInfo.dstSliceInfo[0].sliceType == BufferTypeTmp::OUTPUT) {
                    tmpsrc.addr = output_[rankId_];
                }

                if (signalInfo.dstSliceInfo[0].sliceType == BufferTypeTmp::HCCL_BUFFER) {
                    tmpsrc.addr = scratchAddr_;
                }

                for (uint64_t i = 0; i < signalInfo.dstSliceInfo[0].sliceIdx; i++) {
                    tmpsrc.addr += sliceSize_; // 一个sliceId 对应一份sliceSize
                }

                src.push_back(tmpsrc);
            }

            //self defined broadcast
            if (handleSelfRank_) {
                GroupReduce(channels, dst, src, groupOpSize_,signalInfo.inputDataType, signalInfo.outputDataType, signalInfo.reduceType);
            } else {
                GroupReduceWithoutMyRank(channels, dst, src, groupOpSize_,signalInfo.inputDataType, signalInfo.outputDataType, signalInfo.reduceType);
            }
        }
    }
}

HcclResult CcuKernelOmni::Algorithm()
{
    HCCL_INFO("[ccuOmni_kernel] RankId[%u] Omni run begin.", rankId_);

    // if (rankGroup_.size() == 0) {
    //     return HcclResult::HCCL_SUCCESS;
    // }

    InitResources(); // 创建变量
    LoadArgs();      // 加载变量 顺序跟args相同
    PreSync();       // 前同步
    DoRepeatOmni();  // 执行数据搬运
    PostSync();      // 后同步

    HCCL_INFO("[ccuOmni_kernel] RankId[%u] Omni run end.", rankId_);
    return HcclResult::HCCL_SUCCESS;
}

std::vector<uint64_t> CcuKernelOmni::GeneArgs(const CcuTaskArg &arg)
{
    HCCL_INFO("GeneArgs begin RankId[%u]", rankId_);

    const CcuTaskArgOmni *taskArg = dynamic_cast<const CcuTaskArgOmni *>(&arg);
    if (taskArg == nullptr) {
        HCCL_ERROR("[CcuKernelOmni] taskArg ptr is null. RankId[%u]", rankId_);
    }


    uint64_t inputAddr         = taskArg->inputAddr_;
    uint64_t outputAddr        = taskArg->outputAddr_;
    uint64_t scratchAddr       = taskArg->scratchAddr_;
    uint64_t tokenInfo         = taskArg->token_;
    uint64_t sliceSize         = taskArg->sliceSize_;
    uint64_t repeatNum         = taskArg->repeatNum_;
    uint64_t inputRepeatStride = taskArg->inputRepeatStride_;
    uint64_t outputRepeatStride = taskArg->outputRepeatStride_;

    std::vector<uint64_t> taskArgs ={inputAddr, outputAddr, scratchAddr, tokenInfo, sliceSize, repeatNum, inputRepeatStride, outputRepeatStride};

    auto goSize = CalGoSize(sliceSize);
    taskArgs.insert(taskArgs.cend(), goSize.cbegin(), goSize.cend());

    HCCL_INFO("[CcuKernelOmni] RankId[%u], inputAddr[%llu], outputAddr[%llu], scratchAddr[%llu], tokenInfo[%llu], sliceSize[%llu], "
              "repeatNum[%llu], inputRepeatStride[%llu], outputRepeatStride[%llu].", rankId_, 
              inputAddr, outputAddr, scratchAddr, tokenInfo, sliceSize, repeatNum, inputRepeatStride, outputRepeatStride);

    HCCL_INFO("GeneArgs end RankId[%u]", rankId_);

    return taskArgs;
}

}// namespace ops_hccl