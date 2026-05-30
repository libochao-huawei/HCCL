/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "ins_v2_all_gather_pipelined_executor.h"
#include <cmath>
#include "alg_data_trans_wrapper.h"
#include "ins_temp_all_gather_mesh_1D.h"
#include "ins_temp_all_gather_nhr.h"
#include "topo_match_multilevel.h"

namespace ops_hccl {

template <typename AlgTopoMatch, typename InsAlgTemplate0, typename InsAlgTemplate1>
InsV2AllGatherPipelinedExecutor<AlgTopoMatch, InsAlgTemplate0, InsAlgTemplate1>::
    InsV2AllGatherPipelinedExecutor()
{
}

template <typename AlgTopoMatch, typename InsAlgTemplate0, typename InsAlgTemplate1>
HcclResult InsV2AllGatherPipelinedExecutor<AlgTopoMatch, InsAlgTemplate0, InsAlgTemplate1>::CalcAlgHierarchyInfo(
    HcclComm comm, TopoInfoWithNetLayerDetails *topoInfo, AlgHierarchyInfoForAllLevel &algHierarchyInfo)
{
    AlgTopoMatch topoMatch;
    CHK_RET(topoMatch.MatchTopo(comm, topoInfo, algHierarchyInfo));
    return HCCL_SUCCESS;
}

template <typename AlgTopoMatch, typename InsAlgTemplate0, typename InsAlgTemplate1>
uint64_t InsV2AllGatherPipelinedExecutor<AlgTopoMatch, InsAlgTemplate0, InsAlgTemplate1>::GetRankSize(
    const std::vector<std::vector<u32>> &vTopo) const
{
    uint64_t count = 1;
    for (const auto &i : vTopo) {
        count *= i.size();
    }
    return count;
}

template <typename AlgTopoMatch, typename InsAlgTemplate0, typename InsAlgTemplate1>
HcclResult InsV2AllGatherPipelinedExecutor<AlgTopoMatch, InsAlgTemplate0, InsAlgTemplate1>::CalcRes(
    HcclComm comm, const OpParam &param, const TopoInfoWithNetLayerDetails *topoInfo,
    const AlgHierarchyInfoForAllLevel &algHierarchyInfo, AlgResourceRequest &resourceRequest)
{
    if (param.engine == COMM_ENGINE_CCU) {
        HCCL_ERROR("[InsV2AllGatherPipelinedExecutor][CalcRes] CCU engine is not supported.");
        return HcclResult::HCCL_E_INTERNAL;
    }

    myRank_ = topoInfo->userRank;
    constexpr u32 TOPO_NUM = 2;
    CHK_PRT_RET(algHierarchyInfo.infos.size() < TOPO_NUM || algHierarchyInfo.infos[0].empty() ||
                    algHierarchyInfo.infos[1].empty(),
                 HCCL_ERROR("[InsV2AllGatherPipelinedExecutor][CalcRes] Invalid topoInfo"),
                 HcclResult::HCCL_E_INTERNAL);
    intraHierarchyInfo_ = algHierarchyInfo.infos[0];
    interHierarchyInfo_ = algHierarchyInfo.infos[1];

    InsAlgTemplate0 intraTempAlg(param, topoInfo->userRank, intraHierarchyInfo_);
    InsAlgTemplate1 interTempAlg(param, topoInfo->userRank, interHierarchyInfo_);

    AlgResourceRequest intraTempRequest;
    AlgResourceRequest interTempRequest;
    CHK_RET(intraTempAlg.CalcRes(comm, param, topoInfo, intraTempRequest));
    CHK_RET(interTempAlg.CalcRes(comm, param, topoInfo, interTempRequest));

    constexpr u32 SUB_MAIN_THREAD_NUM = 2;
    resourceRequest.notifyNumOnMainThread = SUB_MAIN_THREAD_NUM;
    resourceRequest.slaveThreadNum = intraTempRequest.slaveThreadNum + interTempRequest.slaveThreadNum +
                                     SUB_MAIN_THREAD_NUM;
    resourceRequest.notifyNumPerThread.emplace_back(intraTempRequest.notifyNumOnMainThread + 1);
    resourceRequest.notifyNumPerThread.insert(resourceRequest.notifyNumPerThread.end(),
                                              intraTempRequest.notifyNumPerThread.begin(),
                                              intraTempRequest.notifyNumPerThread.end());
    resourceRequest.notifyNumPerThread.emplace_back(interTempRequest.notifyNumOnMainThread + 1);
    resourceRequest.notifyNumPerThread.insert(resourceRequest.notifyNumPerThread.end(),
                                              interTempRequest.notifyNumPerThread.begin(),
                                              interTempRequest.notifyNumPerThread.end());

    CHK_PRT_RET(intraTempRequest.channels.empty() || interTempRequest.channels.empty(),
                 HCCL_ERROR("[InsV2AllGatherPipelinedExecutor][CalcRes] template has empty channels."),
                 HcclResult::HCCL_E_INTERNAL);
    resourceRequest.channels.emplace_back(intraTempRequest.channels[0]);
    resourceRequest.channels.emplace_back(interTempRequest.channels[0]);
    return HCCL_SUCCESS;
}

template <typename AlgTopoMatch, typename InsAlgTemplate0, typename InsAlgTemplate1>
HcclResult InsV2AllGatherPipelinedExecutor<AlgTopoMatch, InsAlgTemplate0, InsAlgTemplate1>::PrepareResForTemplate(
    InsAlgTemplate0 &tempAlgIntra, InsAlgTemplate1 &tempAlgInter)
{
    AlgResourceRequest intraTempRequest;
    AlgResourceRequest interTempRequest;
    tempAlgIntra.GetRes(intraTempRequest);
    tempAlgInter.GetRes(interTempRequest);

    auto intraThreadsNum = intraTempRequest.slaveThreadNum + 1;
    auto interThreadsNum = interTempRequest.slaveThreadNum + 1;
    intraThreads_.assign(threads_.begin() + 1, threads_.begin() + intraThreadsNum + 1);
    interThreads_.assign(threads_.begin() + intraThreadsNum + 1,
                         threads_.begin() + intraThreadsNum + interThreadsNum + 1);

    mainThread_ = threads_.at(0);
    templateMainThreads_.emplace_back(intraThreads_.at(0));
    templateMainThreads_.emplace_back(interThreads_.at(0));
    syncNotifyOnTemplates_ = {intraTempRequest.notifyNumOnMainThread, interTempRequest.notifyNumOnMainThread};
    syncNotifyOnMain_ = {0, 1};
    return HCCL_SUCCESS;
}

template <typename AlgTopoMatch, typename InsAlgTemplate0, typename InsAlgTemplate1>
HcclResult InsV2AllGatherPipelinedExecutor<AlgTopoMatch, InsAlgTemplate0, InsAlgTemplate1>::PreSyncTemplate(
    u32 templateIdx)
{
    std::vector<ThreadHandle> thread{templateMainThreads_.at(templateIdx)};
    std::vector<u32> notify{syncNotifyOnTemplates_.at(templateIdx)};
    return PreSyncInterThreads(mainThread_, thread, notify);
}

template <typename AlgTopoMatch, typename InsAlgTemplate0, typename InsAlgTemplate1>
HcclResult InsV2AllGatherPipelinedExecutor<AlgTopoMatch, InsAlgTemplate0, InsAlgTemplate1>::PostSyncTemplate(
    u32 templateIdx)
{
    std::vector<ThreadHandle> thread{templateMainThreads_.at(templateIdx)};
    std::vector<u32> notify{syncNotifyOnMain_.at(templateIdx)};
    return PostSyncInterThreads(mainThread_, thread, notify);
}

template <typename AlgTopoMatch, typename InsAlgTemplate0, typename InsAlgTemplate1>
void InsV2AllGatherPipelinedExecutor<AlgTopoMatch, InsAlgTemplate0, InsAlgTemplate1>::GenTemplateAlgParamsIntra0(
    const OpParam &param, const AlgResourceCtxSerializable &resCtx, const u64 dataOffset,
    const u64 dataCountPerLoopAixs0, const u64 scratchOffset, TemplateDataParams &tempAlgParamsIntra0) const
{
    tempAlgParamsIntra0.buffInfo.inputPtr = param.inputPtr;
    tempAlgParamsIntra0.buffInfo.outputPtr = param.outputPtr;
    tempAlgParamsIntra0.buffInfo.hcclBuff = resCtx.cclMem;
    tempAlgParamsIntra0.buffInfo.inBuffType = BufferType::INPUT;
    tempAlgParamsIntra0.buffInfo.outBuffType = BufferType::OUTPUT;
    tempAlgParamsIntra0.buffInfo.hcclBuffType = BufferType::HCCL_BUFFER;
    tempAlgParamsIntra0.buffInfo.inputSize = param.inputSize;
    tempAlgParamsIntra0.buffInfo.outputSize = param.outputSize;
    tempAlgParamsIntra0.buffInfo.inBuffBaseOff = dataOffset;
    tempAlgParamsIntra0.buffInfo.outBuffBaseOff = rankIdxLevel1_ * rankSizeLevel0_ * dataSize_ + dataOffset;
    tempAlgParamsIntra0.buffInfo.hcclBuffBaseOff = scratchOffset;
    tempAlgParamsIntra0.sliceSize = dataCountPerLoopAixs0 * dataTypeSize_;
    tempAlgParamsIntra0.count = dataCountPerLoopAixs0;
    tempAlgParamsIntra0.tailSize = tempAlgParamsIntra0.sliceSize;
    tempAlgParamsIntra0.inputSliceStride = 0;
    tempAlgParamsIntra0.outputSliceStride = dataSize_;
    tempAlgParamsIntra0.repeatNum = 1;
    tempAlgParamsIntra0.inputRepeatStride = 0;
    tempAlgParamsIntra0.outputRepeatStride = 0;
    tempAlgParamsIntra0.enableRemoteMemAccess = param.opMode == OpMode::OFFLOAD;
}

template <typename AlgTopoMatch, typename InsAlgTemplate0, typename InsAlgTemplate1>
void InsV2AllGatherPipelinedExecutor<AlgTopoMatch, InsAlgTemplate0, InsAlgTemplate1>::GenTemplateAlgParamsInter0(
    const OpParam &param, const AlgResourceCtxSerializable &resCtx, const u64 dataOffset,
    const u64 dataCountPerLoopAixs0, const u64 scratchOffset, TemplateDataParams &tempAlgParamsInter0) const
{
    tempAlgParamsInter0.buffInfo.inputPtr = param.outputPtr;
    tempAlgParamsInter0.buffInfo.outputPtr = param.outputPtr;
    tempAlgParamsInter0.buffInfo.hcclBuff = resCtx.cclMem;
    tempAlgParamsInter0.buffInfo.inBuffBaseOff = dataOffset;
    tempAlgParamsInter0.buffInfo.outBuffBaseOff = dataOffset;
    tempAlgParamsInter0.buffInfo.hcclBuffBaseOff = scratchOffset;
    tempAlgParamsInter0.buffInfo.inBuffType = BufferType::OUTPUT;
    tempAlgParamsInter0.buffInfo.outBuffType = BufferType::OUTPUT;
    tempAlgParamsInter0.buffInfo.hcclBuffType = BufferType::HCCL_BUFFER;
    tempAlgParamsInter0.buffInfo.inputSize = param.inputSize;
    tempAlgParamsInter0.buffInfo.outputSize = param.outputSize;
    tempAlgParamsInter0.sliceSize = dataCountPerLoopAixs0 * dataTypeSize_;
    tempAlgParamsInter0.count = dataCountPerLoopAixs0;
    tempAlgParamsInter0.tailSize = tempAlgParamsInter0.sliceSize;
    tempAlgParamsInter0.inputSliceStride = dataSize_ * rankSizeLevel0_;
    tempAlgParamsInter0.outputSliceStride = dataSize_ * rankSizeLevel0_;
    tempAlgParamsInter0.repeatNum = rankSizeLevel0_;
    tempAlgParamsInter0.inputRepeatStride = dataSize_;
    tempAlgParamsInter0.outputRepeatStride = dataSize_;
    tempAlgParamsInter0.enableRemoteMemAccess = param.opMode == OpMode::OFFLOAD;
}

template <typename AlgTopoMatch, typename InsAlgTemplate0, typename InsAlgTemplate1>
void InsV2AllGatherPipelinedExecutor<AlgTopoMatch, InsAlgTemplate0, InsAlgTemplate1>::GenTemplateAlgParamsInter1(
    const OpParam &param, const AlgResourceCtxSerializable &resCtx, const u64 dataOffset,
    const u64 dataCountPerLoopAixs1, const u64 scratchOffset, TemplateDataParams &tempAlgParamsInter1) const
{
    tempAlgParamsInter1.buffInfo.inputPtr = param.inputPtr;
    tempAlgParamsInter1.buffInfo.outputPtr = param.outputPtr;
    tempAlgParamsInter1.buffInfo.hcclBuff = resCtx.cclMem;
    tempAlgParamsInter1.buffInfo.inBuffBaseOff = dataOffset;
    tempAlgParamsInter1.buffInfo.outBuffBaseOff = rankIdxLevel0_ * dataSize_ + dataOffset;
    tempAlgParamsInter1.buffInfo.hcclBuffBaseOff = scratchOffset;
    tempAlgParamsInter1.buffInfo.inBuffType = BufferType::INPUT;
    tempAlgParamsInter1.buffInfo.outBuffType = BufferType::OUTPUT;
    tempAlgParamsInter1.buffInfo.hcclBuffType = BufferType::HCCL_BUFFER;
    tempAlgParamsInter1.buffInfo.inputSize = param.inputSize;
    tempAlgParamsInter1.buffInfo.outputSize = param.outputSize;
    tempAlgParamsInter1.sliceSize = dataCountPerLoopAixs1 * dataTypeSize_;
    tempAlgParamsInter1.count = dataCountPerLoopAixs1;
    tempAlgParamsInter1.tailSize = tempAlgParamsInter1.sliceSize;
    tempAlgParamsInter1.inputSliceStride = 0;
    tempAlgParamsInter1.outputSliceStride = dataSize_ * rankSizeLevel0_;
    tempAlgParamsInter1.repeatNum = 1;
    tempAlgParamsInter1.inputRepeatStride = 0;
    tempAlgParamsInter1.outputRepeatStride = 0;
    tempAlgParamsInter1.enableRemoteMemAccess = param.opMode == OpMode::OFFLOAD;
}

template <typename AlgTopoMatch, typename InsAlgTemplate0, typename InsAlgTemplate1>
void InsV2AllGatherPipelinedExecutor<AlgTopoMatch, InsAlgTemplate0, InsAlgTemplate1>::
    GenTemplateAlgParamsIntra1LocalNode(const OpParam &param, const AlgResourceCtxSerializable &resCtx,
                                        const u64 dataOffset, const u64 dataCountPerLoopAixs1,
                                        const u64 scratchOffset, TemplateDataParams &tempAlgParamsIntra1) const
{
    GenTemplateAlgParamsIntra0(param, resCtx, dataOffset, dataCountPerLoopAixs1, scratchOffset, tempAlgParamsIntra1);
}

template <typename AlgTopoMatch, typename InsAlgTemplate0, typename InsAlgTemplate1>
void InsV2AllGatherPipelinedExecutor<AlgTopoMatch, InsAlgTemplate0, InsAlgTemplate1>::
    GenTemplateAlgParamsIntra1FromInterScratch(const OpParam &param, const AlgResourceCtxSerializable &resCtx,
                                               const u32 nodeIdx, const u64 dataOffset,
                                               const u64 dataCountPerLoopAixs1, const u64 intraScratchOffset,
                                               const u64 interScratchOffset,
                                               TemplateDataParams &tempAlgParamsIntra1) const
{
    const u64 sliceSize = dataCountPerLoopAixs1 * dataTypeSize_;
    tempAlgParamsIntra1.buffInfo.inputPtr = resCtx.cclMem.addr;
    tempAlgParamsIntra1.buffInfo.outputPtr = param.outputPtr;
    tempAlgParamsIntra1.buffInfo.hcclBuff = resCtx.cclMem;
    tempAlgParamsIntra1.buffInfo.inBuffType = BufferType::HCCL_BUFFER;
    tempAlgParamsIntra1.buffInfo.outBuffType = BufferType::OUTPUT;
    tempAlgParamsIntra1.buffInfo.hcclBuffType = BufferType::HCCL_BUFFER;
    tempAlgParamsIntra1.buffInfo.inputSize = param.inputSize;
    tempAlgParamsIntra1.buffInfo.outputSize = param.outputSize;
    tempAlgParamsIntra1.buffInfo.inBuffBaseOff = interScratchOffset + nodeIdx * sliceSize;
    tempAlgParamsIntra1.buffInfo.outBuffBaseOff = nodeIdx * rankSizeLevel0_ * dataSize_ + dataOffset;
    tempAlgParamsIntra1.buffInfo.hcclBuffBaseOff = intraScratchOffset + nodeIdx * sliceSize * rankSizeLevel0_;
    tempAlgParamsIntra1.sliceSize = sliceSize;
    tempAlgParamsIntra1.count = dataCountPerLoopAixs1;
    tempAlgParamsIntra1.tailSize = sliceSize;
    tempAlgParamsIntra1.inputSliceStride = 0;
    tempAlgParamsIntra1.outputSliceStride = dataSize_;
    tempAlgParamsIntra1.repeatNum = 1;
    tempAlgParamsIntra1.inputRepeatStride = 0;
    tempAlgParamsIntra1.outputRepeatStride = 0;
    tempAlgParamsIntra1.enableRemoteMemAccess = param.opMode == OpMode::OFFLOAD;
}

template <typename AlgTopoMatch, typename InsAlgTemplate0, typename InsAlgTemplate1>
void InsV2AllGatherPipelinedExecutor<AlgTopoMatch, InsAlgTemplate0, InsAlgTemplate1>::GetParallelDataSplit(
    std::vector<float> &splitDataSize) const
{
    double splitData = multipleDimensionSplitRatio_;
    splitDataSize.push_back(1 - splitData);
    splitDataSize.push_back(splitData);
    HCCL_INFO("[InsV2AllGatherPipelinedExecutor] splitDataSize is %f, %f", splitDataSize[0], splitDataSize[1]);
}

template <typename AlgTopoMatch, typename InsAlgTemplate0, typename InsAlgTemplate1>
HcclResult InsV2AllGatherPipelinedExecutor<AlgTopoMatch, InsAlgTemplate0, InsAlgTemplate1>::Orchestrate(
    const OpParam &param, const AlgResourceCtxSerializable &resCtx)
{
    HCCL_INFO("[InsV2AllGatherPipelinedExecutor][Orchestrate] Orchestrate Start");
    maxTmpMemSize_ = resCtx.cclMem.size;
    myRank_ = resCtx.topoInfo.userRank;
    threads_ = resCtx.threads;
    CHK_RET(RestoreChannelMap(resCtx, remoteRankToChannelInfo_));
    intraLinkMap_ = remoteRankToChannelInfo_[0];
    interLinkMap_ = remoteRankToChannelInfo_[1];
    dataCount_ = param.DataDes.count;
    dataType_ = param.DataDes.dataType;
    dataTypeSize_ = DATATYPE_SIZE_TABLE[param.DataDes.dataType];
    dataSize_ = dataCount_ * dataTypeSize_;
    intraHierarchyInfo_ = resCtx.algHierarchyInfo.infos[0];
    interHierarchyInfo_ = resCtx.algHierarchyInfo.infos[1];
    rankSizeLevel0_ = GetRankSize(intraHierarchyInfo_);
    rankSizeLevel1_ = GetRankSize(interHierarchyInfo_);
    rankIdxLevel0_ = myRank_ % rankSizeLevel0_;
    rankIdxLevel1_ = myRank_ / rankSizeLevel0_;

    InsAlgTemplate0 intraTempAlg(param, resCtx.topoInfo.userRank, intraHierarchyInfo_);
    InsAlgTemplate1 interTempAlg(param, resCtx.topoInfo.userRank, interHierarchyInfo_);
    if (param.engine == CommEngine::COMM_ENGINE_AICPU_TS) {
        interTempAlg.SetchannelsPerRank(interLinkMap_);
    }
    CHK_RET(PrepareResForTemplate(intraTempAlg, interTempAlg));

    HcclResult ret = OrchestrateLoop(param, resCtx, intraTempAlg, interTempAlg);
    CHK_PRT_RET(ret != HCCL_SUCCESS,
        HCCL_ERROR("[InsV2AllGatherPipelinedExecutor][Orchestrate] AllGather executor kernel run failed"),
        ret);
    return HCCL_SUCCESS;
}

template <typename AlgTopoMatch, typename InsAlgTemplate0, typename InsAlgTemplate1>
HcclResult InsV2AllGatherPipelinedExecutor<AlgTopoMatch, InsAlgTemplate0, InsAlgTemplate1>::OrchestrateLoop(
    const OpParam &param, const AlgResourceCtxSerializable &resCtx, InsAlgTemplate0 &tempAlgIntra,
    InsAlgTemplate1 &tempAlgInter)
{
    multipleDimensionSplitRatio_ = param.opConfig.multipleDimensionSplitRatio;
    std::vector<float> dataSplitSize;
    GetParallelDataSplit(dataSplitSize);

    u32 intraScratchMultiple = tempAlgIntra.CalcScratchMultiple(BufferType::INPUT, BufferType::OUTPUT);
    u32 interScratchMultiple = tempAlgInter.CalcScratchMultiple(BufferType::INPUT, BufferType::OUTPUT);
    u32 scratchMultipleIntra =
        static_cast<u32>(std::max(std::ceil(dataSplitSize[0] * intraScratchMultiple),
                                  std::ceil(dataSplitSize[1] * intraScratchMultiple * rankSizeLevel1_)));
    u32 scratchMultipleInter =
        static_cast<u32>(std::max(std::ceil(dataSplitSize[1] * interScratchMultiple),
                                  std::ceil(dataSplitSize[0] * interScratchMultiple * rankSizeLevel0_)));
    u32 totalScratchMultiple = scratchMultipleIntra + scratchMultipleInter;
    u64 scratchMemBlockSize = maxTmpMemSize_;
    if (totalScratchMultiple > 0) {
        scratchMemBlockSize = (maxTmpMemSize_ / HCCL_MIN_SLICE_ALIGN / totalScratchMultiple) * HCCL_MIN_SLICE_ALIGN;
        scratchMemBlockSize = std::min(scratchMemBlockSize, static_cast<u64>(UB_MAX_DATA_SIZE));
    }
    u64 intraScratchOffset = 0;
    u64 interScratchOffset = scratchMultipleIntra * scratchMemBlockSize;
    u64 maxCountPerLoop = (std::min(scratchMemBlockSize, static_cast<u64>(UB_MAX_DATA_SIZE)) / dataTypeSize_ / 10) * 10;

    u64 alignSize = AICPU_ALIGN_SIZE;
    u64 dataCountPerLoopAixs0 = static_cast<u64>(dataSplitSize[0] * maxCountPerLoop);
    u64 dataCountPerLoopAixs1 = maxCountPerLoop - dataCountPerLoopAixs0;
    if (dataCountPerLoopAixs0 * dataTypeSize_ >= alignSize) {
        dataCountPerLoopAixs0 = dataCountPerLoopAixs0 * dataTypeSize_ / alignSize * alignSize / dataTypeSize_;
    }
    if (dataCountPerLoopAixs1 * dataTypeSize_ >= alignSize) {
        dataCountPerLoopAixs1 = dataCountPerLoopAixs1 * dataTypeSize_ / alignSize * alignSize / dataTypeSize_;
    }
    maxCountPerLoop = dataCountPerLoopAixs0 + dataCountPerLoopAixs1;
    u32 loopTimes = dataCount_ / maxCountPerLoop + ((dataCount_ % maxCountPerLoop == 0) ? 0 : 1);

    TemplateResource intraTempAlgRes;
    intraTempAlgRes.channels = intraLinkMap_;
    intraTempAlgRes.threads = intraThreads_;
    intraTempAlgRes.aivCommInfoPtr = resCtx.aivCommInfoPtr;
    TemplateResource interTempAlgRes;
    interTempAlgRes.channels = interLinkMap_;
    interTempAlgRes.threads = interThreads_;
    interTempAlgRes.aivCommInfoPtr = resCtx.aivCommInfoPtr;

    u64 finalDataCountPerLoopAixs0 = dataCountPerLoopAixs0;
    u64 finalDataCountPerLoopAixs1 = dataCountPerLoopAixs1;
    if (loopTimes > 1) {
        u64 finalCount = dataCount_ - (loopTimes - 1) * maxCountPerLoop;
        finalDataCountPerLoopAixs0 = static_cast<u64>(dataSplitSize[0] * finalCount);
        finalDataCountPerLoopAixs1 = finalCount - finalDataCountPerLoopAixs0;
    } else {
        finalDataCountPerLoopAixs0 = static_cast<u64>(dataSplitSize[0] * dataCount_);
        finalDataCountPerLoopAixs1 = dataCount_ - finalDataCountPerLoopAixs0;
    }

    TemplateDataParams tempAlgParamsIntra0;
    TemplateDataParams tempAlgParamsIntra1;
    TemplateDataParams tempAlgParamsInter0;
    TemplateDataParams tempAlgParamsInter1;
    constexpr u32 INTRA_TEMPLATE_IDX = 0;
    constexpr u32 INTER_TEMPLATE_IDX = 1;

    for (u32 loopIndex = 0; loopIndex < loopTimes; loopIndex++) {
        u64 currCountPart0 = (loopIndex == loopTimes - 1) ? finalDataCountPerLoopAixs0 : dataCountPerLoopAixs0;
        u64 currCountPart1 = (loopIndex == loopTimes - 1) ? finalDataCountPerLoopAixs1 : dataCountPerLoopAixs1;
        u64 dataOffset0 = loopIndex * maxCountPerLoop * dataTypeSize_;
        u64 dataOffset1 = dataOffset0 + currCountPart0 * dataTypeSize_;

        if (currCountPart1 > 0) {
            CHK_RET(PreSyncInterThreads(mainThread_, templateMainThreads_, syncNotifyOnTemplates_));
            bool hasPendingIntra = true;

            GenTemplateAlgParamsInter1(param, resCtx, dataOffset1, currCountPart1, interScratchOffset,
                                       tempAlgParamsInter1);
            CHK_RET(tempAlgInter.PrepareStepRun(param, tempAlgParamsInter1, interTempAlgRes));

            if (currCountPart0 > 0) {
                GenTemplateAlgParamsIntra0(param, resCtx, dataOffset0, currCountPart0, intraScratchOffset,
                                           tempAlgParamsIntra0);
                CHK_RET(tempAlgIntra.KernelRun(param, tempAlgParamsIntra0, intraTempAlgRes));
            }

            GenTemplateAlgParamsIntra1LocalNode(param, resCtx, dataOffset1, currCountPart1, intraScratchOffset,
                                                tempAlgParamsIntra1);
            CHK_RET(tempAlgIntra.KernelRun(param, tempAlgParamsIntra1, intraTempAlgRes));

            const u32 nSteps = GetNHRStepNum(rankSizeLevel1_);
            for (u32 step = 0; step < nSteps; ++step) {
                AicpuNHRStepInfo stepInfo;
                for (u32 channelIdx = 0; channelIdx < tempAlgInter.GetThreadNum(); channelIdx++) {
                    AicpuNHRStepInfo currStepInfo;
                    CHK_RET(tempAlgInter.RunNHRStep(interTempAlgRes.threads, interTempAlgRes.channels, channelIdx,
                                                     step, currStepInfo));
                    if (channelIdx == 0) {
                        stepInfo = currStepInfo;
                    }
                }
                if (hasPendingIntra) {
                    CHK_RET(PostSyncTemplate(INTRA_TEMPLATE_IDX));
                    hasPendingIntra = false;
                }
                for (u32 nodeIdx : stepInfo.rxSliceIdxs) {
                    if (hasPendingIntra) {
                        CHK_RET(PostSyncTemplate(INTRA_TEMPLATE_IDX));
                        hasPendingIntra = false;
                    }
                    GenTemplateAlgParamsIntra1FromInterScratch(param, resCtx, nodeIdx, dataOffset1, currCountPart1,
                                                               intraScratchOffset, interScratchOffset,
                                                               tempAlgParamsIntra1);
                    CHK_RET(PreSyncTemplate(INTRA_TEMPLATE_IDX));
                    CHK_RET(tempAlgIntra.KernelRun(param, tempAlgParamsIntra1, intraTempAlgRes));
                    hasPendingIntra = true;
                }
            }
            if (hasPendingIntra) {
                CHK_RET(PostSyncTemplate(INTRA_TEMPLATE_IDX));
            }
            for (u32 channelIdx = 0; channelIdx < tempAlgInter.GetThreadNum(); channelIdx++) {
                CHK_RET(tempAlgInter.FinalizeStepRun(interTempAlgRes.threads, channelIdx, false));
            }
            CHK_RET(PostSyncTemplate(INTER_TEMPLATE_IDX));
        }

        if (currCountPart1 == 0 && currCountPart0 > 0) {
            GenTemplateAlgParamsIntra0(param, resCtx, dataOffset0, currCountPart0, intraScratchOffset,
                                       tempAlgParamsIntra0);
            CHK_RET(PreSyncTemplate(INTRA_TEMPLATE_IDX));
            CHK_RET(tempAlgIntra.KernelRun(param, tempAlgParamsIntra0, intraTempAlgRes));
            CHK_RET(PostSyncTemplate(INTRA_TEMPLATE_IDX));
        }

        if (currCountPart0 > 0) {
            GenTemplateAlgParamsInter0(param, resCtx, dataOffset0, currCountPart0, interScratchOffset,
                                       tempAlgParamsInter0);
            CHK_RET(PreSyncTemplate(INTER_TEMPLATE_IDX));
            CHK_RET(tempAlgInter.KernelRun(param, tempAlgParamsInter0, interTempAlgRes));
            CHK_RET(PostSyncTemplate(INTER_TEMPLATE_IDX));
        }
    }

    HCCL_INFO("[InsV2AllGatherPipelinedExecutor][OrchestrateLoop] End.");
    return HcclResult::HCCL_SUCCESS;
}

#if !defined(HCCL_CANN_COMPAT_850)
REGISTER_EXECUTOR_BY_TWO_TEMPS(HcclCMDType::HCCL_CMD_ALLGATHER, InsAllGatherPipelinedMesh1DNHR,
                               InsV2AllGatherPipelinedExecutor, TopoMatchMultilevel, InsTempAllGatherMesh1D,
                               InsTempAllGatherNHR);
#endif

} // namespace ops_hccl
