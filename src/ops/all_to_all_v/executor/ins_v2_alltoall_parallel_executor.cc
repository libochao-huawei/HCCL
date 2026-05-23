/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "ins_v2_alltoall_parallel_executor.h"
#include <cmath>
#include "topo_match_clos_mesh_2d_v2.h"
#include "topo_match_clos_mesh_2d_ubx_v2.h"
#include "ins_temp_alltoall_mesh_2d_v2.h"
#include "ins_temp_alltoall_mesh_clos_v2.h"

namespace ops_hccl {

template <typename AlgTopoMatch, typename InsAlgTemplateX, typename InsAlgTemplateY>
HcclResult InsV2AlltoAllParallelExecutor<AlgTopoMatch, InsAlgTemplateX, InsAlgTemplateY>::CalcAlgHierarchyInfo(
    HcclComm comm,
    TopoInfoWithNetLayerDetails* topoInfo,
    AlgHierarchyInfoForAllLevel& algHierarchyInfo)
{
    AlgTopoMatch topoMatch;
    CHK_RET(topoMatch.MatchTopo(comm, topoInfo, algHierarchyInfo));

    CHK_PRT_RET(algHierarchyInfo.infos.size() < 2 || algHierarchyInfo.infos[0].empty() ||
                    algHierarchyInfo.infos[1].empty(),
                HCCL_ERROR("[InsV2AlltoAllParallelExecutor][CalcAlgHierarchyInfo] Invalid hierarchy info, "
                           "infos.size[%zu], infos[0].size[%zu], infos[1].size[%zu]",
                           algHierarchyInfo.infos.size(), algHierarchyInfo.infos[0].size(),
                           algHierarchyInfo.infos[1].size()),
                HcclResult::HCCL_E_INTERNAL);

    xRankSize_ = algHierarchyInfo.infos[0].size();
    yRankSize_ = algHierarchyInfo.infos[1].size();
    algHierarchyInfoCache_ = algHierarchyInfo;

    HCCL_INFO("[InsV2AlltoAllParallelExecutor][CalcAlgHierarchyInfo] xRankSize_[%llu], yRankSize_[%llu]",
              xRankSize_, yRankSize_);
    return HCCL_SUCCESS;
}

template <typename AlgTopoMatch, typename InsAlgTemplateX, typename InsAlgTemplateY>
HcclResult InsV2AlltoAllParallelExecutor<AlgTopoMatch, InsAlgTemplateX, InsAlgTemplateY>::CalcRes(
    HcclComm comm,
    const OpParam& param,
    const TopoInfoWithNetLayerDetails* topoInfo,
    const AlgHierarchyInfoForAllLevel& algHierarchyInfo,
    AlgResourceRequest& resourceRequest)
{
    myRank_ = topoInfo->userRank;

    InsAlgTemplateX templateX(param, topoInfo->userRank, algHierarchyInfo.infos[0]);
    InsAlgTemplateY templateY(param, topoInfo->userRank, algHierarchyInfo.infos[1]);

    AlgResourceRequest resReqX, resReqY;
    CHK_RET(templateX.CalcRes(comm, param, topoInfo, resReqX));
    CHK_RET(templateY.CalcRes(comm, param, topoInfo, resReqY));

    for (auto& chVec : resReqX.channels) {
        resourceRequest.channels.push_back(chVec);
    }
    for (auto& chVec : resReqY.channels) {
        resourceRequest.channels.push_back(chVec);
    }

    resourceRequest.slaveThreadNum = resReqX.slaveThreadNum + resReqY.slaveThreadNum;

    resourceRequest.notifyNumOnMainThread =
        resReqX.notifyNumOnMainThread + resReqY.notifyNumOnMainThread;

    resourceRequest.notifyNumPerThread.clear();
    resourceRequest.notifyNumPerThread.insert(
        resourceRequest.notifyNumPerThread.end(),
        resReqX.notifyNumPerThread.begin(), resReqX.notifyNumPerThread.end());
    resourceRequest.notifyNumPerThread.insert(
        resourceRequest.notifyNumPerThread.end(),
        resReqY.notifyNumPerThread.begin(), resReqY.notifyNumPerThread.end());

    scratchMultipleX_ = templateX.CalcScratchMultiple(BufferType::INPUT, BufferType::OUTPUT);
    scratchMultipleY_ = templateY.CalcScratchMultiple(BufferType::HCCL_BUFFER, BufferType::OUTPUT);
    resourceRequest.scratchMultiple = std::max(scratchMultipleX_, scratchMultipleY_);

    HCCL_DEBUG("[InsV2AlltoAllParallelExecutor][CalcRes] myRank[%u], slaveThreadNum[%u], "
               "notifyNumOnMainThread[%u], channels[%zu], scratchMultiple[%llu]",
               myRank_, resourceRequest.slaveThreadNum, resourceRequest.notifyNumOnMainThread,
               resourceRequest.channels.size(), resourceRequest.scratchMultiple);
    return HCCL_SUCCESS;
}

template <typename AlgTopoMatch, typename InsAlgTemplateX, typename InsAlgTemplateY>
HcclResult InsV2AlltoAllParallelExecutor<AlgTopoMatch, InsAlgTemplateX, InsAlgTemplateY>::PrepareResForTemplate(
    InsAlgTemplateX& templateX,
    InsAlgTemplateY& templateY,
    std::vector<ThreadHandle>& intraThreads,
    std::vector<ThreadHandle>& interThreads)
{
    AlgResourceRequest intraReq, interReq;
    CHK_RET(templateX.GetRes(intraReq));
    CHK_RET(templateY.GetRes(interReq));

    u64 t0ThreadNum = intraReq.slaveThreadNum + 1;
    u64 t1ThreadNum = interReq.slaveThreadNum + 1;

    intraThreads.clear();
    for (u64 i = 0; i < t0ThreadNum; i++) {
        intraThreads.push_back(threads_[i]);
    }

    interThreads.clear();
    for (u64 i = 0; i < t1ThreadNum; i++) {
        interThreads.push_back(threads_[t0ThreadNum + i]);
    }

    mainThread_ = threads_[0];
    templateMainThreads_.emplace_back(intraThreads[0]);
    templateMainThreads_.emplace_back(interThreads[0]);
    syncNotifyOnTemplates_ = {intraReq.notifyNumOnMainThread, interReq.notifyNumOnMainThread};
    syncNotifyOnMain_ = {0, 1};

    HCCL_INFO("[InsV2AlltoAllParallelExecutor][PrepareResForTemplate] intraThreads[%zu], interThreads[%zu]",
              intraThreads.size(), interThreads.size());
    return HCCL_SUCCESS;
}

template <typename AlgTopoMatch, typename InsAlgTemplateX, typename InsAlgTemplateY>
HcclResult InsV2AlltoAllParallelExecutor<AlgTopoMatch, InsAlgTemplateX, InsAlgTemplateY>::RestoreChannelMap(
    const AlgResourceCtxSerializable& resCtx,
    std::vector<std::map<u32, std::vector<ChannelInfo>>>& rankIdToChannelInfo) const
{
    return InsCollAlgBase::RestoreChannelMap(resCtx, rankIdToChannelInfo);
}

template <typename AlgTopoMatch, typename InsAlgTemplateX, typename InsAlgTemplateY>
TemplateResource InsV2AlltoAllParallelExecutor<AlgTopoMatch, InsAlgTemplateX, InsAlgTemplateY>::BuildResource(
    const std::vector<ThreadHandle>& threads,
    const std::map<u32, std::vector<ChannelInfo>>& channels)
{
    TemplateResource res;
    res.threads = threads;
    res.channels = channels;
    res.aivCommInfoPtr = nullptr;
    return res;
}

template <typename AlgTopoMatch, typename InsAlgTemplateX, typename InsAlgTemplateY>
HcclResult InsV2AlltoAllParallelExecutor<AlgTopoMatch, InsAlgTemplateX, InsAlgTemplateY>::Orchestrate(
    const OpParam& param,
    const AlgResourceCtxSerializable& resCtx)
{
    HCCL_INFO("[InsV2AlltoAllParallelExecutor][Orchestrate] Start");

    maxTmpMemSize_ = resCtx.cclMem.size;
    myRank_ = resCtx.topoInfo.userRank;
    rankSize_ = resCtx.topoInfo.userRankSize;
    threads_ = resCtx.threads;

    if (param.engine != CommEngine::COMM_ENGINE_AIV && param.engine != CommEngine::COMM_ENGINE_CCU) {
        CHK_RET(RestoreChannelMap(resCtx, remoteRankToChannelInfo_));
    }

    dataCount_ = param.DataDes.count;
    dataType_ = param.DataDes.dataType;
    dataTypeSize_ = DATATYPE_SIZE_TABLE[param.DataDes.dataType];
    dataSize_ = dataCount_ * dataTypeSize_;

    myXRank_ = myRank_ % xRankSize_;
    myYRank_ = myRank_ / xRankSize_;

    HCCL_INFO("[InsV2AlltoAllParallelExecutor][Orchestrate] rank[%u] myX[%llu] myY[%llu] "
              "xRankSize[%llu] yRankSize[%llu]",
              myRank_, myXRank_, myYRank_, xRankSize_, yRankSize_);

    InsAlgTemplateX templateX(param, resCtx.topoInfo.userRank, algHierarchyInfoCache_.infos[0]);
    InsAlgTemplateY templateY(param, resCtx.topoInfo.userRank, algHierarchyInfoCache_.infos[1]);

    if (param.engine == CommEngine::COMM_ENGINE_AICPU_TS) {
        templateY.SetchannelsPerRank(remoteRankToChannelInfo_[1]);
    }

    std::vector<ThreadHandle> intraThreads, interThreads;
    CHK_RET(PrepareResForTemplate(templateX, templateY, intraThreads, interThreads));

    HcclResult ret = OrchestrateLoop(param, templateX, templateY,
                                     remoteRankToChannelInfo_, intraThreads, interThreads);
    CHK_PRT_RET(ret != HCCL_SUCCESS,
                HCCL_ERROR("[InsV2AlltoAllParallelExecutor][Orchestrate]errNo[0x%016llx] "
                           "AlltoAll parallel executor kernel run failed",
                           HCCL_ERROR_CODE(ret)),
                ret);

    HCCL_INFO("[InsV2AlltoAllParallelExecutor][Orchestrate] End");
    return HCCL_SUCCESS;
}

template <typename AlgTopoMatch, typename InsAlgTemplateX, typename InsAlgTemplateY>
HcclResult InsV2AlltoAllParallelExecutor<AlgTopoMatch, InsAlgTemplateX, InsAlgTemplateY>::OrchestrateLoop(
    const OpParam& param,
    InsAlgTemplateX& templateX,
    InsAlgTemplateY& templateY,
    const std::vector<std::map<u32, std::vector<ChannelInfo>>>& channelMapVec,
    std::vector<ThreadHandle>& intraThreads,
    std::vector<ThreadHandle>& interThreads)
{
    u64 totalCount = dataCount_;
    u64 maxPerLoop = (UB_MAX_DATA_SIZE / dataTypeSize_ / 10) * 10;
    if (maxPerLoop == 0) {
        maxPerLoop = totalCount;
    }
    u64 loopNum = (totalCount + maxPerLoop - 1) / maxPerLoop;

    multipleDimensionSplitRatio_ = param.multipleDimensionSplitRatio;
    if (multipleDimensionSplitRatio_ <= 0.0 || multipleDimensionSplitRatio_ >= 1.0) {
        HCCL_WARNING("[InsV2AlltoAllParallelExecutor][OrchestrateLoop] splitRatio[%f] out of range, fallback to 0.5",
                     multipleDimensionSplitRatio_);
        multipleDimensionSplitRatio_ = 0.5;
    }

    // Scratch partition (separate regions for X/Y-parts)
    u64 scratchMemBlockSize = maxTmpMemSize_;
    u64 xScratchMultiple = std::max(scratchMultipleX_, scratchMultipleY_);
    u64 yScratchMultiple = xScratchMultiple;
    u64 xScratchOffset = 0;
    u64 yScratchOffset = xScratchMultiple * scratchMemBlockSize;

    HCCL_INFO("[InsV2AlltoAllParallelExecutor][OrchestrateLoop] totalCount[%llu] maxPerLoop[%llu] loopNum[%llu] "
              "splitRatio[%f] xScratchOff[%llu] yScratchOff[%llu]",
              totalCount, maxPerLoop, loopNum, multipleDimensionSplitRatio_, xScratchOffset, yScratchOffset);

    for (u64 loopIdx = 0; loopIdx < loopNum; loopIdx++) {
        u64 loopSize = (loopIdx == loopNum - 1) ? (totalCount - loopIdx * maxPerLoop) : maxPerLoop;

        // Data splitting: X-parts (ratio) + Y-parts (1-ratio)
        u64 xDataSize = static_cast<u64>(multipleDimensionSplitRatio_ * loopSize);
        u64 yDataSize = loopSize - xDataSize;

        u64 typeSize = dataTypeSize_;
        u64 loopBaseOff = loopIdx * maxPerLoop * typeSize;

        if (xDataSize == 0 && yDataSize > 0) {
            u64 yInputOff = loopBaseOff;
            u64 yOutputOff = loopBaseOff;

            CHK_RET(PreSyncInterThreads(mainThread_, templateMainThreads_, syncNotifyOnTemplates_));

            TemplateDataParams s1Y;
            CHK_RET(GenAlgParamsStage1(param, loopIdx, yDataSize, yScratchOffset, yInputOff, s1Y));
            TemplateResource s1ResY = BuildResource(interThreads, channelMapVec[1]);
            CHK_RET(templateY.KernelRun(param, s1Y, s1ResY));

            CHK_RET(PostSyncInterThreads(mainThread_, templateMainThreads_, syncNotifyOnMain_));

            CHK_RET(PreSyncInterThreads(mainThread_, templateMainThreads_, syncNotifyOnTemplates_));

            TemplateDataParams s2X;
            CHK_RET(GenAlgParamsStage2(param, loopIdx, yDataSize, yScratchOffset, yOutputOff, s2X));
            TemplateResource s2ResX = BuildResource(intraThreads, channelMapVec[0]);
            CHK_RET(templateX.KernelRun(param, s2X, s2ResX));

            CHK_RET(PostSyncInterThreads(mainThread_, templateMainThreads_, syncNotifyOnMain_));
            continue;
        }

        if (yDataSize == 0 && xDataSize > 0) {
            u64 xInputOff = loopBaseOff;
            u64 xOutputOff = loopBaseOff;

            CHK_RET(PreSyncInterThreads(mainThread_, templateMainThreads_, syncNotifyOnTemplates_));

            TemplateDataParams s1X;
            CHK_RET(GenAlgParamsStage1(param, loopIdx, xDataSize, xScratchOffset, xInputOff, s1X));
            TemplateResource s1ResX = BuildResource(intraThreads, channelMapVec[0]);
            CHK_RET(templateX.KernelRun(param, s1X, s1ResX));

            CHK_RET(PostSyncInterThreads(mainThread_, templateMainThreads_, syncNotifyOnMain_));

            CHK_RET(PreSyncInterThreads(mainThread_, templateMainThreads_, syncNotifyOnTemplates_));

            TemplateDataParams s2Y;
            CHK_RET(GenAlgParamsStage2(param, loopIdx, xDataSize, xScratchOffset, xOutputOff, s2Y));
            TemplateResource s2ResY = BuildResource(interThreads, channelMapVec[1]);
            CHK_RET(templateY.KernelRun(param, s2Y, s2ResY));

            CHK_RET(PostSyncInterThreads(mainThread_, templateMainThreads_, syncNotifyOnMain_));
            continue;
        }

        if (xDataSize == 0 && yDataSize == 0) {
            HCCL_WARNING("[InsV2AlltoAllParallelExecutor][OrchestrateLoop] loopIdx[%llu] both xDataSize and "
                         "yDataSize are zero, skip", loopIdx);
            continue;
        }

        u64 xInputOff = loopBaseOff;
        u64 yInputOff = loopBaseOff + xDataSize * typeSize;

        // Stage 1: parallel — X-template (X-axis) + Y-template (Y-axis)
        CHK_RET(PreSyncInterThreads(mainThread_, templateMainThreads_, syncNotifyOnTemplates_));

        TemplateDataParams stage1ParamsX;
        CHK_RET(GenAlgParamsStage1(param, loopIdx, xDataSize, xScratchOffset, xInputOff, stage1ParamsX));
        TemplateResource stage1ResX = BuildResource(intraThreads, channelMapVec[0]);
        CHK_RET(templateX.KernelRun(param, stage1ParamsX, stage1ResX));

        TemplateDataParams stage1ParamsY;
        CHK_RET(GenAlgParamsStage1(param, loopIdx, yDataSize, yScratchOffset, yInputOff, stage1ParamsY));
        TemplateResource stage1ResY = BuildResource(interThreads, channelMapVec[1]);
        CHK_RET(templateY.KernelRun(param, stage1ParamsY, stage1ResY));

        CHK_RET(PostSyncInterThreads(mainThread_, templateMainThreads_, syncNotifyOnMain_));

        // Stage 2: parallel — Y-template processes X-parts, X-template processes Y-parts
        u64 xOutputOff = loopBaseOff;
        u64 yOutputOff = loopBaseOff + xDataSize * typeSize;

        CHK_RET(PreSyncInterThreads(mainThread_, templateMainThreads_, syncNotifyOnTemplates_));

        TemplateDataParams stage2ParamsX;
        CHK_RET(GenAlgParamsStage2(param, loopIdx, xDataSize, xScratchOffset, xOutputOff, stage2ParamsX));
        TemplateResource stage2ResX = BuildResource(interThreads, channelMapVec[1]);
        CHK_RET(templateY.KernelRun(param, stage2ParamsX, stage2ResX));

        TemplateDataParams stage2ParamsY;
        CHK_RET(GenAlgParamsStage2(param, loopIdx, yDataSize, yScratchOffset, yOutputOff, stage2ParamsY));
        TemplateResource stage2ResY = BuildResource(intraThreads, channelMapVec[0]);
        CHK_RET(templateX.KernelRun(param, stage2ParamsY, stage2ResY));

        CHK_RET(PostSyncInterThreads(mainThread_, templateMainThreads_, syncNotifyOnMain_));
    }

    return HCCL_SUCCESS;
}

template <typename AlgTopoMatch, typename InsAlgTemplateX, typename InsAlgTemplateY>
HcclResult InsV2AlltoAllParallelExecutor<AlgTopoMatch, InsAlgTemplateX, InsAlgTemplateY>::GenAlgParamsStage1(
    const OpParam& param,
    u64 loopIdx,
    u64 loopSize,
    u64 scratchOffset,
    u64 inputDataOffset,
    TemplateDataParams& params)
{
    u64 dataTypeSize = DATATYPE_SIZE_TABLE[param.DataDes.dataType];
    u64 sliceSize = loopSize * dataTypeSize;
    u64 perRankChunk = sliceSize / (xRankSize_ * yRankSize_);

    params.buffInfo.inputPtr = param.inputPtr;
    params.buffInfo.outputPtr = param.hcclBuff.addr;
    params.buffInfo.hcclBuff = param.hcclBuff;
    params.buffInfo.inBuffType = BufferType::INPUT;
    params.buffInfo.outBuffType = BufferType::HCCL_BUFFER;
    params.buffInfo.hcclBuffType = BufferType::HCCL_BUFFER;
    params.buffInfo.inBuffBaseOff = inputDataOffset;
    params.buffInfo.outBuffBaseOff = scratchOffset;
    params.buffInfo.hcclBuffBaseOff = scratchOffset;
    params.buffInfo.inputSize = sliceSize;
    params.buffInfo.outputSize = sliceSize;
    params.buffInfo.hcclBuffSize = sliceSize;

    params.count = loopSize;
    params.sliceSize = sliceSize;
    params.inputSliceStride = perRankChunk;
    params.outputSliceStride = perRankChunk;
    params.repeatNum = 1;
    params.inputRepeatStride = 0;
    params.outputRepeatStride = 0;
    params.tailSize = 0;
    params.dataType = param.DataDes.dataType;
    params.enableRemoteMemAccess = false;

    HCCL_DEBUG(
        "[InsV2AlltoAllParallelExecutor][GenAlgParamsStage1] loopIdx[%llu] loopSize[%llu] "
        "sliceSize[%llu] perRankChunk[%llu] inBuffBaseOff[%llu] outBuffBaseOff[%llu]",
        loopIdx, loopSize, sliceSize, perRankChunk, params.buffInfo.inBuffBaseOff,
        params.buffInfo.outBuffBaseOff);
    return HCCL_SUCCESS;
}

template <typename AlgTopoMatch, typename InsAlgTemplateX, typename InsAlgTemplateY>
HcclResult InsV2AlltoAllParallelExecutor<AlgTopoMatch, InsAlgTemplateX, InsAlgTemplateY>::GenAlgParamsStage2(
    const OpParam& param,
    u64 loopIdx,
    u64 loopSize,
    u64 scratchOffset,
    u64 outputDataOffset,
    TemplateDataParams& params)
{
    u64 dataTypeSize = DATATYPE_SIZE_TABLE[param.DataDes.dataType];
    u64 sliceSize = loopSize * dataTypeSize;
    u64 perRankChunk = sliceSize / (xRankSize_ * yRankSize_);

    params.buffInfo.inputPtr = nullptr;
    params.buffInfo.outputPtr = param.outputPtr;
    params.buffInfo.hcclBuff = param.hcclBuff;
    params.buffInfo.inBuffType = BufferType::HCCL_BUFFER;
    params.buffInfo.outBuffType = BufferType::OUTPUT;
    params.buffInfo.hcclBuffType = BufferType::HCCL_BUFFER;
    params.buffInfo.inBuffBaseOff = scratchOffset;
    params.buffInfo.outBuffBaseOff = outputDataOffset;
    params.buffInfo.hcclBuffBaseOff = scratchOffset;
    params.buffInfo.inputSize = sliceSize;
    params.buffInfo.outputSize = sliceSize;
    params.buffInfo.hcclBuffSize = sliceSize;

    params.count = loopSize;
    params.sliceSize = sliceSize;
    params.inputSliceStride = perRankChunk;
    params.outputSliceStride = perRankChunk;
    params.repeatNum = 1;
    params.inputRepeatStride = 0;
    params.outputRepeatStride = 0;
    params.tailSize = 0;
    params.dataType = param.DataDes.dataType;
    params.enableRemoteMemAccess = false;

    HCCL_DEBUG(
        "[InsV2AlltoAllParallelExecutor][GenAlgParamsStage2] loopIdx[%llu] loopSize[%llu] "
        "sliceSize[%llu] perRankChunk[%llu] inBuffBaseOff[%llu] outBuffBaseOff[%llu]",
        loopIdx, loopSize, sliceSize, perRankChunk, params.buffInfo.inBuffBaseOff,
        params.buffInfo.outBuffBaseOff);
    return HCCL_SUCCESS;
}

REGISTER_EXECUTOR_BY_TWO_TEMPS(HcclCMDType::HCCL_CMD_ALLTOALL,
                               InsAlltoAllParallelMesh2DClosV2,
                               InsV2AlltoAllParallelExecutor,
                               TopoMatchClosMesh2DV2,
                               InsTempAlltoAllMesh2DV2,
                               InsTempAlltoAllMeshClosV2);

REGISTER_EXECUTOR_BY_TWO_TEMPS(HcclCMDType::HCCL_CMD_ALLTOALL,
                               InsAlltoAllParallelMesh2DClosUBXV2,
                               InsV2AlltoAllParallelExecutor,
                               TopoMatchClosMesh2DUBXV2,
                               InsTempAlltoAllMesh2DV2,
                               InsTempAlltoAllMeshClosV2);

}  // namespace ops_hccl
