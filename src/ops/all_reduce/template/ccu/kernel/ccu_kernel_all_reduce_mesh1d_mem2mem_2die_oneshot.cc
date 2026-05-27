/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "ccu_kernel_all_reduce_mesh1d_mem2mem_2die_oneshot.h"

namespace ops_hccl {

constexpr int INPUT_XN_ID = 0;
constexpr int TOKEN_XN_ID = 1;
constexpr int POST_SYNC_ID = 2;

constexpr int CKE_IDX_0 = 0;

constexpr int MISSION_NUM = 2;

constexpr uint32_t LOCAL_REDUCE_LOOP_COUNT = 16;
const std::string LOCAL_REDUCE_LOOP_BLOCK_TAG{"_local_reduce_loop_"};
constexpr const char *MISSION_SYNC_TAG = "allreduce_mesh1d_mem2mem_2die_oneshot_mission_sync";

CcuKernelAllReduceMesh1DMem2Mem2DieOneShot::CcuKernelAllReduceMesh1DMem2Mem2DieOneShot(
    const CcuKernelArgAllReduceMesh1DMem2Mem2DieOneShot *arg)
    : kernelArg_(arg),
      rankSize_(arg->dimSize_),
      rankId_(arg->rankId_),
      rmtReduceWithMyRank_(arg->rmtReduceWithMyRank_),
      rmtReduceRankNum_(arg->channelCount + (arg->rmtReduceWithMyRank_ ? 1 : 0)),
      missionSyncMybit_(1U << (arg->rmtReduceWithMyRank_ ? 1 : 0)),
      missionSyncWaitBit_(1U << (arg->rmtReduceWithMyRank_ ? 0 : 1)),
      dataType_(arg->opParam_.DataDes.dataType),
      outputDataType_(arg->opParam_.DataDes.outputType),
      reduceOp_(arg->opParam_.reduceType)
{
    channels_.assign(arg->channels, arg->channels + arg->channelCount);
    if (outputDataType_ == HcclDataType::HCCL_DATA_TYPE_RESERVED) {
        outputDataType_ = dataType_;
        HCCL_INFO("[CcuKernelAllReduceMesh1DMem2Mem2DieOneShot] outputDataType is invalid, set to[%u]",
            outputDataType_);
    }

    resourceAllocated = false;
    moConfig.msInterleave = 0;
    moConfig.loopCount = 0;
    moConfig.memSlice = 0;
    moRes.eventCount = 0;
    moRes.bufCount = 0;
    enginePool = 0;

    HCCL_INFO("[CcuKernelAllReduceMesh1DMem2Mem2DieOneShot] Init, KernelArgs are rankId[%u], "
        "rankSize_[%llu], dataType[%d], outputDataType[%d], reduceOp[%d], rmtReduceWithMyRank_[%d]",
        rankId_, rankSize_, dataType_, outputDataType_, reduceOp_, rmtReduceWithMyRank_);
}

CcuResult CcuKernelAllReduceMesh1DMem2Mem2DieOneShot::InitResource()
{
    if (channels_.size() == 0) {
        HCCL_ERROR("[CcuKernelAllReduceMesh1DMem2Mem2DieOneShot] channels is empty.");
        return CcuResult::CCU_E_INTERNAL;
    }
    if (rmtReduceRankNum_ == 0) {
        HCCL_ERROR("[CcuKernelAllReduceMesh1DMem2Mem2DieOneShot] rmtReduceRankNum is 0.");
        return CcuResult::CCU_E_INTERNAL;
    }

    peerInput_.resize(channels_.size());
    peerToken_.resize(channels_.size());
    for (uint32_t channelIdx = 0; channelIdx < channels_.size(); channelIdx++) {
        peerInput_[channelIdx] = ccu::GetResByChannel<ccu::Variable>(channels_[channelIdx], INPUT_XN_ID);
        peerToken_[channelIdx] = ccu::GetResByChannel<ccu::Variable>(channels_[channelIdx], TOKEN_XN_ID);
    }

    return CCU_SUCCESS;
}

CcuResult CcuKernelAllReduceMesh1DMem2Mem2DieOneShot::LoadArgs()
{
    uint32_t argId = 0;
    CCU_CHK_RET(ccu::LoadArg(myInput_, argId++));
    CCU_CHK_RET(ccu::LoadArg(myOutput_, argId++));
    CCU_CHK_RET(ccu::LoadArg(myToken_, argId++));
    CCU_CHK_RET(ccu::LoadArg(myScratch_, argId++));
    CCU_CHK_RET(ccu::LoadArg(normalSliceSize_, argId++));
    CCU_CHK_RET(ccu::LoadArg(scratchBaseOffset0_, argId++));
    CCU_CHK_RET(ccu::LoadArg(scratchBaseOffset1_, argId++));
    CCU_CHK_RET(ccu::LoadArg(localReduceSliceOffset0_, argId++));
    CCU_CHK_RET(ccu::LoadArg(localReduceSliceOffset1_, argId++));
    CCU_CHK_RET(ccu::LoadArg(localReduceGoSize_.addrOffset, argId++));
    CCU_CHK_RET(ccu::LoadArg(localReduceGoSize_.loopParam, argId++));
    CCU_CHK_RET(ccu::LoadArg(localReduceGoSize_.parallelParam, argId++));
    CCU_CHK_RET(ccu::LoadArg(localReduceGoSize_.residual, argId++));
    CCU_CHK_RET(ccu::LoadArg(localReduceGoSize0_.addrOffset, argId++));
    CCU_CHK_RET(ccu::LoadArg(localReduceGoSize0_.loopParam, argId++));
    CCU_CHK_RET(ccu::LoadArg(localReduceGoSize0_.parallelParam, argId++));
    CCU_CHK_RET(ccu::LoadArg(localReduceGoSize0_.residual, argId++));
    CCU_CHK_RET(ccu::LoadArg(localReduceGoSize1_.addrOffset, argId++));
    CCU_CHK_RET(ccu::LoadArg(localReduceGoSize1_.loopParam, argId++));
    CCU_CHK_RET(ccu::LoadArg(localReduceGoSize1_.parallelParam, argId++));
    CCU_CHK_RET(ccu::LoadArg(localReduceGoSize1_.residual, argId++));
    return CCU_SUCCESS;
}

void CcuKernelAllReduceMesh1DMem2Mem2DieOneShot::InitDerivedArgs()
{
    scratchBaseOffset1_ = 0;
    for (uint32_t i = 0; i < rmtReduceRankNum_; i++) {
        scratchBaseOffset1_ += normalSliceSize_;
    }
}

CcuResult CcuKernelAllReduceMesh1DMem2Mem2DieOneShot::PreSync()
{
    for (ChannelHandle channel : channels_) {
        CCU_CHK_RET(ccu::WriteVariableWithNotify(channel, myInput_,
            INPUT_XN_ID, CKE_IDX_0, 1 << INPUT_XN_ID));
        CCU_CHK_RET(ccu::WriteVariableWithNotify(channel, myToken_,
            TOKEN_XN_ID, CKE_IDX_0, 1 << TOKEN_XN_ID));
    }

    uint32_t allBit = (1 << INPUT_XN_ID) | (1 << TOKEN_XN_ID);
    for (ChannelHandle channel : channels_) {
        CCU_CHK_RET(ccu::NotifyWait(channel, CKE_IDX_0, allBit));
    }
    return CCU_SUCCESS;
}

CcuResult CcuKernelAllReduceMesh1DMem2Mem2DieOneShot::PostSync()
{
    for (auto &ch : channels_) {
        CCU_CHK_RET(ccu::NotifyRecord(ch, CKE_IDX_0, 1 << POST_SYNC_ID));
    }
    for (auto &ch : channels_) {
        CCU_CHK_RET(ccu::NotifyWait(ch, CKE_IDX_0, 1 << POST_SYNC_ID));
    }
    return CCU_SUCCESS;
}

CcuResult CcuKernelAllReduceMesh1DMem2Mem2DieOneShot::RmtReduce()
{
    std::vector<ccu::RemoteAddr> remoteInput(rmtReduceRankNum_);
    uint32_t channelIdx = 0;
    uint32_t selfIdx = rankId_ % rmtReduceRankNum_;
    for (uint32_t rankIdx = 0; rankIdx < rmtReduceRankNum_; rankIdx++) {
        if (rmtReduceWithMyRank_ && rankIdx == selfIdx) {
            continue;
        }
        if (channelIdx >= peerInput_.size()) {
            HCCL_ERROR("[CcuKernelAllReduceMesh1DMem2Mem2DieOneShot] channelIdx[%u] >= peerInput_.size[%llu].",
                channelIdx, static_cast<unsigned long long>(peerInput_.size()));
            return CcuResult::CCU_E_INTERNAL;
        }
        remoteInput[rankIdx].addr = peerInput_[channelIdx];
        remoteInput[rankIdx].token = peerToken_[channelIdx];
        channelIdx++;
    }

    std::vector<ccu::LocalAddr> scratchDst;
    scratchDst.reserve(rmtReduceRankNum_);
    ccu::Variable scratchOffset;
    scratchOffset = 0;
    for (uint32_t rankIdx = 0; rankIdx < rmtReduceRankNum_; rankIdx++) {
        ccu::LocalAddr dst;
        dst.addr = myScratch_;
        dst.addr += rmtReduceWithMyRank_ ? scratchBaseOffset0_ : scratchBaseOffset1_;
        dst.addr += scratchOffset;
        scratchOffset += normalSliceSize_;
        dst.token = myToken_;
        scratchDst.push_back(dst);
    }

    uint32_t channelId = 0;
    for (uint32_t rankIdx = 0; rankIdx < rmtReduceRankNum_; rankIdx++) {
        uint16_t mask = static_cast<uint16_t>(1U << rankIdx);
        if (rmtReduceWithMyRank_ && rankIdx == selfIdx) {
            scratchDst[rankIdx].addr = myInput_;
            scratchDst[rankIdx].token = myToken_;
            CCU_CHK_RET(ccu::EventRecord(event_, mask));
        } else {
            CCU_CHK_RET(ccu::Read(channels_[channelId], scratchDst[rankIdx], remoteInput[rankIdx],
                normalSliceSize_, event_, mask));
            channelId++;
        }
    }

    CCU_CHK_RET(ccu::EventWait(event_, static_cast<uint16_t>((1U << rmtReduceRankNum_) - 1U)));

    if (rmtReduceWithMyRank_) {
        ccu::LocalAddr output;
        output.token = myToken_;
        output.addr = myOutput_;
        CCU_CHK_RET(ReduceLoopGroup(output, scratchDst, localReduceGoSize_, dataType_, outputDataType_, reduceOp_,
            "all_reduce_2die_localreduce1_"));
    } else {
        CCU_CHK_RET(ReduceLoopGroup(scratchDst[0], scratchDst, localReduceGoSize_, dataType_, outputDataType_, reduceOp_,
            "all_reduce_2die_localreduce1_"));
    }

    return CCU_SUCCESS;
}

CcuResult CcuKernelAllReduceMesh1DMem2Mem2DieOneShot::DoLocalReduce()
{
    std::vector<ccu::LocalAddr> src;
    src.reserve(MISSION_NUM);
    for (uint32_t i = 0; i < MISSION_NUM; i++) {
        ccu::LocalAddr localSrc;
        localSrc.token = myToken_;
        localSrc.addr = i == 0 ? myOutput_ : myScratch_;
        localSrc.addr += i == 0 ? scratchBaseOffset0_ : scratchBaseOffset1_;
        localSrc.addr += rmtReduceWithMyRank_ ? localReduceSliceOffset0_ : localReduceSliceOffset1_;
        src.push_back(localSrc);
    }

    ccu::LocalAddr dst;
    dst.token = myToken_;
    dst.addr = myOutput_;
    dst.addr += rmtReduceWithMyRank_ ? localReduceSliceOffset0_ : localReduceSliceOffset1_;
    if (rmtReduceWithMyRank_) {
        CCU_CHK_RET(ReduceLoopGroup(dst, src, localReduceGoSize0_, dataType_, outputDataType_, reduceOp_,
            "all_reduce_2die_localreduce2_"));
    } else {
        CCU_CHK_RET(ReduceLoopGroup(dst, src, localReduceGoSize1_, dataType_, outputDataType_, reduceOp_,
            "all_reduce_2die_localreduce2_"));
    }
    return CCU_SUCCESS;
}

CcuResult CcuKernelAllReduceMesh1DMem2Mem2DieOneShot::ReduceLoopGroup(ccu::LocalAddr outDstOrg,
    std::vector<ccu::LocalAddr> srcOrg, GroupOpSizeVars goSize, HcclDataType dataType,
    HcclDataType outputDataType, HcclReduceOp opType, const std::string &loopName)
{
    LocalReduceVar var;
    CCU_CHK_RET(CreateReduceLoop(var, srcOrg.size(), dataType, outputDataType, opType, loopName));
    auto &loops = loopMap[loopName];

    ccu::LocalAddr dst = outDstOrg;
    std::vector<ccu::LocalAddr> src = srcOrg;

    uint32_t expansionNum = GetReduceExpansionNum(opType, dataType, outputDataType);
    ccu::Variable sliceSizeExpansion;

    if (expansionNum != 1) {
        ccu::Variable tmp;
        tmp = GetExpansionParam(expansionNum);
        dst.token += tmp;
    }

    CCU_IF(goSize.loopParam != 0)
    {
        ccu::Variable loopParam;
        loopParam = GetLoopParam(0, moConfig.memSlice * moConfig.loopCount, 0);
        loopParam += goSize.loopParam;

        ccu::Variable sliceSize;
        sliceSize = moConfig.memSlice;
        sliceSizeExpansion = moConfig.memSlice * expansionNum;
        SetReduceLoopInput(var, 0, dst, src, sliceSize, sliceSizeExpansion);

        ccu::Variable paraCfg;
        paraCfg = GetParallelParam(moConfig.loopCount - 1, 0, 1);
        ccu::Variable offsetCfg;
        offsetCfg = GetOffsetParam(moConfig.memSlice, moConfig.msInterleave, 1);

        loops.loopParam[0] = loopParam;
        std::vector<ccu::Loop> grpLoops{*loops.loops[0]};
        ccu::LoopGroup group(paraCfg, offsetCfg, 1, grpLoops);
    }

    CCU_IF(goSize.parallelParam != 0)
    {
        for (uint32_t i = 0; i < src.size(); i++) {
            src[i].addr += goSize.addrOffset;
        }
        for (uint32_t i = 0; i < expansionNum; i++) {
            dst.addr += goSize.addrOffset;
        }

        sliceSizeExpansion = 0;
        for (uint32_t i = 0; i < expansionNum; i++) {
            sliceSizeExpansion += goSize.residual;
        }
        SetReduceLoopInput(var, 0, dst, src, goSize.residual, sliceSizeExpansion);

        for (uint32_t i = 0; i < src.size(); i++) {
            src[i].addr += goSize.residual;
        }

        for (uint32_t i = 0; i < expansionNum; i++) {
            dst.addr += goSize.residual;
        }

        ccu::Variable sliceSize;
        sliceSize = moConfig.memSlice;
        sliceSizeExpansion = moConfig.memSlice * expansionNum;
        SetReduceLoopInput(var, 1, dst, src, sliceSize, sliceSizeExpansion);

        ccu::Variable loopCfg0;
        loopCfg0 = GetLoopParam(0, 0, 1);
        ccu::Variable loopCfg1;
        loopCfg1 = GetLoopParam(0, 0, 1);
        ccu::Variable offsetCfg;
        offsetCfg = GetOffsetParam(moConfig.memSlice, moConfig.msInterleave, 1);

        loops.loopParam[0] = loopCfg0;
        loops.loopParam[1] = loopCfg1;
        std::vector<ccu::Loop> grpLoops{*loops.loops[0], *loops.loops[1]};
        ccu::LoopGroup group(goSize.parallelParam, offsetCfg, 2, grpLoops);
    }
    return CCU_SUCCESS;
}

std::string CcuKernelAllReduceMesh1DMem2Mem2DieOneShot::GetLoopBlockTag(std::string loopType, int32_t index) const
{
    return loopType + LOCAL_REDUCE_LOOP_BLOCK_TAG + std::to_string(index);
}

CcuResult CcuKernelAllReduceMesh1DMem2Mem2DieOneShot::CreateReduceLoop(LocalReduceVar &var,
    uint32_t size, HcclDataType dataType, HcclDataType outputDataType, HcclReduceOp opType,
    const std::string &loopName)
{
    CCU_CHK_RET(AllocGoResource(moConfig, moRes, resourceAllocated, LOCAL_REDUCE_LOOP_COUNT));

    if (IsLoopEntityRegistered(loopName)) {
        return CCU_SUCCESS;
    }
    CreateLoopEntity(loopName);
    auto &loops = loopMap[loopName];

    if (size == 0) {
        HCCL_ERROR("[CcuKernelAllReduceMesh1DMem2Mem2DieOneShot] local reduce input size is 0.");
        return CcuResult::CCU_E_INTERNAL;
    }

    uint32_t expansionNum = GetReduceExpansionNum(opType, dataType, outputDataType);
    uint32_t usedBufNum = size > expansionNum ? size : expansionNum;
    if (usedBufNum > moConfig.msInterleave) {
        HCCL_ERROR("[CcuKernelAllReduceMesh1DMem2Mem2DieOneShot] usedBufNum[%u] exceeds msInterleave[%u].",
            usedBufNum, moConfig.msInterleave);
        return CcuResult::CCU_E_INTERNAL;
    }

    for (uint32_t index = 0; index < 2; index++) {
        var.loopSrc[index].resize(size);

        uint32_t bufBase = index * moConfig.msInterleave;
        ccu::Event loopEvt = moRes.completedEvent[index];
        loops.body[index].reset(new ccu::Func(
            [this, index, bufBase, loopEvt, size, dataType, outputDataType, opType, &var]() {
            for (uint32_t i = 0; i < size; i++) {
                ccu::LocalCopy(moRes.ccuBuf[bufBase + i], var.loopSrc[index][i],
                    var.loopLen[index], loopEvt, static_cast<uint16_t>(1U << i));
            }

            ccu::EventWait(loopEvt, static_cast<uint16_t>((1U << size) - 1U));
            if (size > 1) {
                ccu::LocalReduce(&moRes.ccuBuf[bufBase], size, dataType, outputDataType, opType,
                    var.loopLen[index], loopEvt, 1);
                ccu::EventWait(loopEvt, 1);
            }
            ccu::LocalCopy(var.loopDst[index], moRes.ccuBuf[bufBase], var.loopLenExp[index], loopEvt, 1);
            ccu::EventWait(loopEvt, 1);
        }));
        loops.loops[index].reset(new ccu::Loop(loops.loopParam[index], *loops.body[index]));
    }

    return CCU_SUCCESS;
}

void CcuKernelAllReduceMesh1DMem2Mem2DieOneShot::SetReduceLoopInput(LocalReduceVar &var, uint32_t index,
    const ccu::LocalAddr &dst, const std::vector<ccu::LocalAddr> &src, const ccu::Variable &len,
    const ccu::Variable &lenExp)
{
    var.loopDst[index] = dst;
    for (uint32_t i = 0; i < src.size(); i++) {
        var.loopSrc[index][i] = src[i];
    }
    var.loopLen[index] = len;
    var.loopLenExp[index] = lenExp;
}

CcuResult CcuKernelAllReduceMesh1DMem2Mem2DieOneShot::MissionSync(uint32_t maskIndex)
{
    uint16_t recordMask = static_cast<uint16_t>(missionSyncMybit_ << (MISSION_NUM * maskIndex));
    uint16_t waitMask = static_cast<uint16_t>(missionSyncWaitBit_ << (MISSION_NUM * maskIndex));
    CCU_CHK_RET(ccu::EventRecord(MISSION_SYNC_TAG, recordMask));
    CCU_CHK_RET(ccu::EventWait(MISSION_SYNC_TAG, waitMask));
    return CCU_SUCCESS;
}

CcuResult CcuKernelAllReduceMesh1DMem2Mem2DieOneShot::Algorithm()
{
    HCCL_INFO("[CcuKernelAllReduceMesh1DMem2Mem2DieOneShot] AllReduceMesh1DMem2Mem2DieOneShot run");

    CCU_CHK_RET(InitResource());

    CCU_CHK_RET(LoadArgs());

    InitDerivedArgs();

    CCU_CHK_RET(PreSync());

    CCU_CHK_RET(RmtReduce());

    CCU_CHK_RET(PostSync());

    CCU_CHK_RET(MissionSync(0));

    CCU_CHK_RET(DoLocalReduce());

    CCU_CHK_RET(MissionSync(1));

    HCCL_INFO("[CcuKernelAllReduceMesh1DMem2Mem2DieOneShot] AllReduceMesh1DMem2Mem2DieOneShot end");

    return CCU_SUCCESS;
}

CcuResult CcuAllReduceMesh1DMem2Mem2DieOneShotKernel(CcuKernelArg arg)
{
    auto *kernelArg = static_cast<CcuKernelArgAllReduceMesh1DMem2Mem2DieOneShot *>(arg);
    if (kernelArg == nullptr) {
        HCCL_ERROR("[CcuKernelAllReduceMesh1DMem2Mem2DieOneShot] kernelArg is null.");
        return CcuResult::CCU_E_PTR;
    }

    CcuKernelAllReduceMesh1DMem2Mem2DieOneShot kernel(kernelArg);
    return kernel.Algorithm();
}

} // namespace ops_hccl
