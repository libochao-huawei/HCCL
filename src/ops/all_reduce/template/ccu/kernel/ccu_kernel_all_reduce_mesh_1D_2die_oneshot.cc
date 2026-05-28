/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "ccu_kernel_all_reduce_mesh_1D_2die_oneshot.h"

namespace ops_hccl {

constexpr int INPUT_XN_ID = 0;
constexpr int TOKEN_XN_ID = 1;
constexpr int POST_SYNC_ID = 2;

constexpr int CKE_IDX_0 = 0;
constexpr int CKE_IDX_1 = 1;

constexpr int DIE_WORK = 2;
constexpr uint32_t LOOP_NUM = 64;
const std::string LOCAL_REDUCE_LOOP_BLOCK_TAG{"_local_reduce_loop_"};
constexpr const char *MISSION_SYNC_TAG = "allreduce_mesh1d_2die_oneshot_mission_sync";

CcuKernelAllreduceMesh1D2DieOneShot::CcuKernelAllreduceMesh1D2DieOneShot(
    const CcuKernelArgAllreduceMesh1D2DieOneShot *arg)
    : kernelArg_(arg),
      rmtReduceWithMyRank_(arg->rmtReduceWithMyRank_),
      myRankId_(arg->rankId_),
      rankSize_(static_cast<uint32_t>(arg->dimSize_)),
      rmtReduceRankNum_(arg->channelCount + (arg->rmtReduceWithMyRank_ ? 1 : 0)),
      dataType_(arg->opParam_.DataDes.dataType),
      outputDataType_(arg->opParam_.DataDes.outputType),
      reduceOp_(arg->opParam_.reduceType),
      missionSyncMybit_(1U << (arg->rmtReduceWithMyRank_ ? 1 : 0)),
      missionSyncWaitBit_(1U << (arg->rmtReduceWithMyRank_ ? 0 : 1))
{
    channels_.assign(arg->channels, arg->channels + arg->channelCount);
    if (outputDataType_ == HcclDataType::HCCL_DATA_TYPE_RESERVED) {
        outputDataType_ = dataType_;
        HCCL_INFO("[CcuKernelAllreduceMesh1D2DieOneShot] outputDataType is invalid, set to[%u]",
            outputDataType_);
    }

    resourceAllocated = false;
    moConfig.msInterleave = 0;
    moConfig.loopCount = 0;
    moConfig.memSlice = 0;
    moRes.eventCount = 0;
    moRes.bufCount = 0;
    enginePool = 0;

    HCCL_INFO("[CcuKernelAllreduceMesh1D2DieOneShot] Init, KernelArgs are rankId[%u], rankSize_[%u], "
        "dataType[%d], outputDataType[%d], reduceOp[%d], rmtReduceWithMyRank[%d], rmtReduceRankNum[%u]",
        myRankId_, rankSize_, dataType_, outputDataType_, reduceOp_, rmtReduceWithMyRank_, rmtReduceRankNum_);
}

CcuResult CcuKernelAllreduceMesh1D2DieOneShot::InitResource()
{
    if (channels_.size() == 0) {
        HCCL_ERROR("[CcuKernelAllreduceMesh1D2DieOneShot] channels is empty.");
        return CcuResult::CCU_E_INTERNAL;
    }
    if (rmtReduceRankNum_ == 0) {
        HCCL_ERROR("[CcuKernelAllreduceMesh1D2DieOneShot] rmtReduceRankNum is 0.");
        return CcuResult::CCU_E_INTERNAL;
    }

    input_.resize(channels_.size());
    remoteToken_.resize(channels_.size());
    for (uint32_t channelIdx = 0; channelIdx < channels_.size(); channelIdx++) {
        input_[channelIdx] = ccu::GetResByChannel<ccu::Variable>(channels_[channelIdx], INPUT_XN_ID);
        remoteToken_[channelIdx] = ccu::GetResByChannel<ccu::Variable>(channels_[channelIdx], TOKEN_XN_ID);
    }

    HCCL_INFO("[CcuKernelAllreduceMesh1D2DieOneShot] InitResources finished");
    return CCU_SUCCESS;
}

CcuResult CcuKernelAllreduceMesh1D2DieOneShot::LoadArgs()
{
    uint32_t argId = 0;
    CCU_CHK_RET(ccu::LoadArg(myInput_, argId++));
    CCU_CHK_RET(ccu::LoadArg(myOutput_, argId++));
    CCU_CHK_RET(ccu::LoadArg(myToken_, argId++));
    CCU_CHK_RET(ccu::LoadArg(myScratch_, argId++));
    CCU_CHK_RET(ccu::LoadArg(scratchBaseOffset0_, argId++));
    CCU_CHK_RET(ccu::LoadArg(scratchBaseOffset1_, argId++));
    CCU_CHK_RET(ccu::LoadArg(localReduceSliceOffset0_, argId++));
    CCU_CHK_RET(ccu::LoadArg(localReduceSliceOffset1_, argId++));
    CCU_CHK_RET(ccu::LoadArg(rmtReduceGoSize_.addrOffset, argId++));
    CCU_CHK_RET(ccu::LoadArg(rmtReduceGoSize_.loopParam, argId++));
    CCU_CHK_RET(ccu::LoadArg(rmtReduceGoSize_.parallelParam, argId++));
    CCU_CHK_RET(ccu::LoadArg(rmtReduceGoSize_.residual, argId++));
    CCU_CHK_RET(ccu::LoadArg(localReduceGoSize0_.addrOffset, argId++));
    CCU_CHK_RET(ccu::LoadArg(localReduceGoSize0_.loopParam, argId++));
    CCU_CHK_RET(ccu::LoadArg(localReduceGoSize0_.parallelParam, argId++));
    CCU_CHK_RET(ccu::LoadArg(localReduceGoSize0_.residual, argId++));
    CCU_CHK_RET(ccu::LoadArg(localReduceGoSize1_.addrOffset, argId++));
    CCU_CHK_RET(ccu::LoadArg(localReduceGoSize1_.loopParam, argId++));
    CCU_CHK_RET(ccu::LoadArg(localReduceGoSize1_.parallelParam, argId++));
    CCU_CHK_RET(ccu::LoadArg(localReduceGoSize1_.residual, argId++));
    HCCL_INFO("[CcuKernelAllreduceMesh1D2DieOneShot] LoadArgs run finished");
    return CCU_SUCCESS;
}

CcuResult CcuKernelAllreduceMesh1D2DieOneShot::PreSync()
{
    for (uint32_t i = 0; i < channels_.size(); i++) {
        CCU_CHK_RET(ccu::WriteVariableWithNotify(channels_[i], myInput_,
            INPUT_XN_ID, CKE_IDX_0, 1 << INPUT_XN_ID));
        CCU_CHK_RET(ccu::WriteVariableWithNotify(channels_[i], myToken_,
            TOKEN_XN_ID, CKE_IDX_0, 1 << TOKEN_XN_ID));
    }

    uint32_t allBit = (1 << INPUT_XN_ID) | (1 << TOKEN_XN_ID);
    for (uint32_t i = 0; i < channels_.size(); i++) {
        CCU_CHK_RET(ccu::NotifyWait(channels_[i], CKE_IDX_0, allBit));
    }
    HCCL_INFO("[CcuKernelAllreduceMesh1D2DieOneShot] PreSync run finished");
    return CCU_SUCCESS;
}

CcuResult CcuKernelAllreduceMesh1D2DieOneShot::PostSync(uint32_t signalIndex)
{
    for (auto channel : channels_) {
        CCU_CHK_RET(ccu::NotifyRecord(channel, CKE_IDX_1, 1 << signalIndex));
    }

    for (auto channel : channels_) {
        CCU_CHK_RET(ccu::NotifyWait(channel, CKE_IDX_1, 1 << signalIndex));
    }
    HCCL_INFO("[CcuKernelAllreduceMesh1D2DieOneShot] PostSync run finished");
    return CCU_SUCCESS;
}

CcuResult CcuKernelAllreduceMesh1D2DieOneShot::DoRmtReduce()
{
    std::vector<ccu::RemoteAddr> remoteSrc;
    remoteSrc.reserve(channels_.size());
    for (uint32_t peerIdx = 0; peerIdx < channels_.size(); peerIdx++) {
        HCCL_INFO("[CcuKernelAllreduceMesh1D2DieOneShot] DoRmtReduce myRankId_[%u] peerIdx[%u]",
            myRankId_, peerIdx);
        ccu::RemoteAddr src;
        src.token = remoteToken_[peerIdx];
        src.addr = input_[peerIdx];
        remoteSrc.push_back(src);
    }

    std::vector<ccu::LocalAddr> localSrc;
    if (rmtReduceWithMyRank_) {
        ccu::LocalAddr src;
        src.token = myToken_;
        src.addr = myInput_;
        localSrc.push_back(src);
    }

    ccu::LocalAddr dst;
    dst.token = myToken_;
    dst.addr = myScratch_;
    dst.addr += rmtReduceWithMyRank_ ? scratchBaseOffset0_ : scratchBaseOffset1_;

    CCU_CHK_RET(ReduceLoopGroup(dst, remoteSrc, localSrc, rmtReduceGoSize_, dataType_,
        outputDataType_, reduceOp_, "allreduce_2die_rmt_reduce"));
    HCCL_INFO("[CcuKernelAllreduceMesh1D2DieOneShot] Step1 RmtReduce run finished");
    return CCU_SUCCESS;
}

std::string CcuKernelAllreduceMesh1D2DieOneShot::GetLoopBlockTag(std::string loopType, int32_t index) const
{
    return loopType + LOCAL_REDUCE_LOOP_BLOCK_TAG + std::to_string(index);
}

CcuResult CcuKernelAllreduceMesh1D2DieOneShot::CreateReduceLoop(BufferedReduceVar &var,
    const std::string &loopName, uint32_t remoteCount, uint32_t localCount, HcclDataType dataType,
    HcclDataType outputDataType, HcclReduceOp opType)
{
    CCU_CHK_RET(AllocGoResource(moConfig, moRes, resourceAllocated, LOOP_NUM));

    if (IsLoopEntityRegistered(loopName)) {
        return CCU_SUCCESS;
    }
    CreateLoopEntity(loopName);
    auto &loops = loopMap[loopName];

    uint32_t size = remoteCount + localCount;
    if (size == 0) {
        HCCL_ERROR("[CcuKernelAllreduceMesh1D2DieOneShot] reduce input size is 0.");
        return CcuResult::CCU_E_INTERNAL;
    }

    uint32_t expansionNum = GetReduceExpansionNum(opType, dataType, outputDataType);
    uint32_t usedBufNum = size > expansionNum ? size : expansionNum;
    if (usedBufNum > moConfig.msInterleave) {
        HCCL_ERROR("[CcuKernelAllreduceMesh1D2DieOneShot] usedBufNum[%u] exceeds msInterleave[%u].",
            usedBufNum, moConfig.msInterleave);
        return CcuResult::CCU_E_INTERNAL;
    }

    for (uint32_t index = 0; index < 2; index++) {
        var.loopRemoteSrc[index].resize(remoteCount);
        var.loopLocalSrc[index].resize(localCount);

        uint32_t bufBase = index * moConfig.msInterleave;
        ccu::Event loopEvt = moRes.completedEvent[index];
        loops.body[index].reset(new ccu::Func(
            [this, index, bufBase, loopEvt, remoteCount, localCount, size,
             dataType, outputDataType, opType, &var]() {
            for (uint32_t i = 0; i < remoteCount; i++) {
                ccu::Read(channels_[i], moRes.ccuBuf[bufBase + i], var.loopRemoteSrc[index][i],
                    var.loopLen[index], loopEvt, static_cast<uint16_t>(1U << i));
            }
            for (uint32_t i = 0; i < localCount; i++) {
                uint32_t bufIdx = remoteCount + i;
                ccu::LocalCopy(moRes.ccuBuf[bufBase + bufIdx], var.loopLocalSrc[index][i],
                    var.loopLen[index], loopEvt, static_cast<uint16_t>(1U << bufIdx));
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

void CcuKernelAllreduceMesh1D2DieOneShot::SetReduceLoopInput(BufferedReduceVar &var, uint32_t index,
    const ccu::LocalAddr &dst, const std::vector<ccu::RemoteAddr> &remoteSrc,
    const std::vector<ccu::LocalAddr> &localSrc, const ccu::Variable &len, const ccu::Variable &lenExp)
{
    var.loopDst[index] = dst;
    for (uint32_t i = 0; i < remoteSrc.size(); i++) {
        var.loopRemoteSrc[index][i] = remoteSrc[i];
    }
    for (uint32_t i = 0; i < localSrc.size(); i++) {
        var.loopLocalSrc[index][i] = localSrc[i];
    }
    var.loopLen[index] = len;
    var.loopLenExp[index] = lenExp;
}

CcuResult CcuKernelAllreduceMesh1D2DieOneShot::ReduceLoopGroup(ccu::LocalAddr outDstOrg,
    std::vector<ccu::RemoteAddr> remoteSrcOrg, std::vector<ccu::LocalAddr> localSrcOrg,
    GroupOpSizeVars goSize, HcclDataType dataType, HcclDataType outputDataType, HcclReduceOp opType,
    const std::string &loopName)
{
    BufferedReduceVar var;
    CCU_CHK_RET(CreateReduceLoop(var, loopName, remoteSrcOrg.size(), localSrcOrg.size(), dataType,
        outputDataType, opType));
    auto &loops = loopMap[loopName];

    ccu::LocalAddr dst = outDstOrg;
    std::vector<ccu::RemoteAddr> remoteSrc = remoteSrcOrg;
    std::vector<ccu::LocalAddr> localSrc = localSrcOrg;

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
        SetReduceLoopInput(var, 0, dst, remoteSrc, localSrc, sliceSize, sliceSizeExpansion);

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
        for (uint32_t i = 0; i < remoteSrc.size(); i++) {
            remoteSrc[i].addr += goSize.addrOffset;
        }
        for (uint32_t i = 0; i < localSrc.size(); i++) {
            localSrc[i].addr += goSize.addrOffset;
        }
        for (uint32_t i = 0; i < expansionNum; i++) {
            dst.addr += goSize.addrOffset;
        }

        sliceSizeExpansion = 0;
        for (uint32_t i = 0; i < expansionNum; i++) {
            sliceSizeExpansion += goSize.residual;
        }
        SetReduceLoopInput(var, 0, dst, remoteSrc, localSrc, goSize.residual, sliceSizeExpansion);

        for (uint32_t i = 0; i < remoteSrc.size(); i++) {
            remoteSrc[i].addr += goSize.residual;
        }
        for (uint32_t i = 0; i < localSrc.size(); i++) {
            localSrc[i].addr += goSize.residual;
        }
        for (uint32_t i = 0; i < expansionNum; i++) {
            dst.addr += goSize.residual;
        }

        ccu::Variable sliceSize;
        sliceSize = moConfig.memSlice;
        sliceSizeExpansion = moConfig.memSlice * expansionNum;
        SetReduceLoopInput(var, 1, dst, remoteSrc, localSrc, sliceSize, sliceSizeExpansion);

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

CcuResult CcuKernelAllreduceMesh1D2DieOneShot::DoLocalReduce()
{
    std::vector<ccu::LocalAddr> src;
    src.reserve(DIE_WORK);
    for (uint32_t i = 0; i < DIE_WORK; i++) {
        ccu::LocalAddr localSrc;
        localSrc.token = myToken_;
        localSrc.addr = myScratch_;
        localSrc.addr += i == 0 ? scratchBaseOffset0_ : scratchBaseOffset1_;
        localSrc.addr += rmtReduceWithMyRank_ ? localReduceSliceOffset0_ : localReduceSliceOffset1_;
        src.push_back(localSrc);
    }

    ccu::LocalAddr dst;
    dst.token = myToken_;
    dst.addr = myOutput_;
    dst.addr += rmtReduceWithMyRank_ ? localReduceSliceOffset0_ : localReduceSliceOffset1_;

    std::vector<ccu::RemoteAddr> remoteSrc;
    if (rmtReduceWithMyRank_) {
        CcuResult ret = CCU_SUCCESS;
        CCU_IF(localReduceSliceOffset1_ != 0)
        {
            ret = ReduceLoopGroup(dst, remoteSrc, src, localReduceGoSize0_, dataType_,
                outputDataType_, reduceOp_, "allreduce_2die_local_reduce");
        }
        CCU_CHK_RET(ret);
    } else {
        CCU_CHK_RET(ReduceLoopGroup(dst, remoteSrc, src, localReduceGoSize1_, dataType_,
            outputDataType_, reduceOp_, "allreduce_2die_local_reduce"));
    }
    HCCL_INFO("[CcuKernelAllreduceMesh1D2DieOneShot] DoLocalReduce run finished");
    return CCU_SUCCESS;
}

CcuResult CcuKernelAllreduceMesh1D2DieOneShot::MissionSync(uint32_t maskIndex)
{
    HCCL_INFO("[CcuKernelAllreduceMesh1D2DieOneShot] MissionSync, missionSyncMybit_[%u], missionSyncWaitBit_[%u]",
        missionSyncMybit_, missionSyncWaitBit_);
    uint16_t recordMask = static_cast<uint16_t>(missionSyncMybit_ << (DIE_WORK * maskIndex));
    uint16_t waitMask = static_cast<uint16_t>(missionSyncWaitBit_ << (DIE_WORK * maskIndex));
    CCU_CHK_RET(ccu::EventRecord(MISSION_SYNC_TAG, recordMask));
    CCU_CHK_RET(ccu::EventWait(MISSION_SYNC_TAG, waitMask));
    HCCL_INFO("[CcuKernelAllreduceMesh1D2DieOneShot] MissionSync run finished");
    return CCU_SUCCESS;
}

CcuResult CcuKernelAllreduceMesh1D2DieOneShot::Algorithm()
{
    HCCL_INFO("[CcuKernelAllreduceMesh1D2DieOneShot] AllreduceMesh1D2DieOneShot run");
    CCU_CHK_RET(InitResource());

    CCU_CHK_RET(LoadArgs());

    CCU_CHK_RET(PreSync());

    HCCL_INFO("[CcuKernelAllreduceMesh1D2DieOneShot] Algorithm second step rmtReduce begins.");

    CCU_CHK_RET(DoRmtReduce());

    CCU_CHK_RET(PostSync(POST_SYNC_ID));

    CCU_CHK_RET(MissionSync(0));

    HCCL_INFO("[CcuKernelAllreduceMesh1D2DieOneShot] Algorithm second step localreduce begins.");

    CCU_CHK_RET(DoLocalReduce());

    CCU_CHK_RET(MissionSync(1));

    HCCL_INFO("[CcuKernelAllreduceMesh1D2DieOneShot] AllreduceMesh1D2Die end");
    return CCU_SUCCESS;
}

CcuResult CcuAllreduceMesh1D2DieOneShotKernel(CcuKernelArg arg)
{
    auto *kernelArg = static_cast<CcuKernelArgAllreduceMesh1D2DieOneShot *>(arg);
    if (kernelArg == nullptr) {
        HCCL_ERROR("[CcuKernelAllreduceMesh1D2DieOneShot] kernelArg is null.");
        return CcuResult::CCU_E_PTR;
    }

    CcuKernelAllreduceMesh1D2DieOneShot kernel(kernelArg);
    return kernel.Algorithm();
}

} // namespace ops_hccl
