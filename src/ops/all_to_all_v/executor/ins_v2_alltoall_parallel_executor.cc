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
#include "alg_data_trans_wrapper.h"
#include "ins_temp_alltoall_mesh_2d_v2.h"
#include "ins_temp_alltoall_mesh_clos_v2.h"

#include "topo_match_clos_mesh_2d_v2.h"
#include "topo_match_clos_mesh_2d_ubx_v2.h"
#include "topo_match_ubx.h"

namespace ops_hccl {

template <typename AlgTopoMatch, typename InsAlgTemplate0, typename InsAlgTemplate1>
InsV2AlltoAllParallelExecutor<AlgTopoMatch, InsAlgTemplate0, InsAlgTemplate1>::InsV2AlltoAllParallelExecutor()
{
}

template <typename AlgTopoMatch, typename InsAlgTemplate0, typename InsAlgTemplate1>
HcclResult InsV2AlltoAllParallelExecutor<AlgTopoMatch, InsAlgTemplate0, InsAlgTemplate1>::CalcAlgHierarchyInfo(
    HcclComm comm, TopoInfoWithNetLayerDetails *topoInfo, AlgHierarchyInfoForAllLevel &algHierarchyInfo)
{
    AlgTopoMatch topoMatch;
    CHK_RET(topoMatch.MatchTopo(comm, topoInfo, algHierarchyInfo));
    return HCCL_SUCCESS;
}

template <typename AlgTopoMatch, typename InsAlgTemplate0, typename InsAlgTemplate1>
HcclResult InsV2AlltoAllParallelExecutor<AlgTopoMatch, InsAlgTemplate0, InsAlgTemplate1>::CalcRes(
    HcclComm comm, const OpParam &param, const TopoInfoWithNetLayerDetails *topoInfo,
    const AlgHierarchyInfoForAllLevel &algHierarchyInfo, AlgResourceRequest &resourceRequest)
{
    myRank_ = topoInfo->userRank;

    // v2.0 Fix 4: transactional CalcRes — zero-initialize before sub-calls
    resourceRequest = AlgResourceRequest{};

    std::vector<std::vector<u32>> intraHierarchyInfo;
    std::vector<std::vector<u32>> interHierarchyInfo;
    if (topoInfo->level0Topo == Level0Shape::MESH_1D_CLOS && !topoInfo->level0PcieMix) {
        intraHierarchyInfo = {algHierarchyInfo.infos[0][0]};
        std::vector<u32> closRanks;
        u32 meshSize = algHierarchyInfo.infos[0][0].size();
        for (auto rank : algHierarchyInfo.infos[0][1]) {
            if (rank % meshSize == topoInfo->userRank % meshSize) {
                closRanks.push_back(rank);
            }
        }
        interHierarchyInfo = {closRanks};
    } else {
        constexpr u32 TOPO_NUM = 2;
        CHK_PRT_RET(algHierarchyInfo.infos.size() < TOPO_NUM || algHierarchyInfo.infos[0].empty() ||
                    algHierarchyInfo.infos[1].empty(),
                    HCCL_ERROR("[InsV2AlltoAllParallelExecutor][CalcRes] Invalid topoInfo"),
                    HcclResult::HCCL_E_INTERNAL);
        intraHierarchyInfo = algHierarchyInfo.infos[0];
        interHierarchyInfo = algHierarchyInfo.infos[1];
    }

    InsAlgTemplate0 intraTempAlg(param, topoInfo->userRank, intraHierarchyInfo);
    InsAlgTemplate1 interTempAlg(param, topoInfo->userRank, interHierarchyInfo);

    // v2.0 Fix 4: separate local requests; merge only if both succeed
    AlgResourceRequest intraTempRequest;
    AlgResourceRequest interTempRequest;
    CHK_RET(intraTempAlg.CalcRes(comm, param, topoInfo, intraTempRequest));
    CHK_RET(interTempAlg.CalcRes(comm, param, topoInfo, interTempRequest));

    constexpr u32 SUB_MAIN_THREAD_NUM = 2;
    resourceRequest.notifyNumOnMainThread = SUB_MAIN_THREAD_NUM;
    resourceRequest.slaveThreadNum =
        intraTempRequest.slaveThreadNum + interTempRequest.slaveThreadNum + SUB_MAIN_THREAD_NUM;
    resourceRequest.notifyNumPerThread.emplace_back(intraTempRequest.notifyNumOnMainThread + 1);
    resourceRequest.notifyNumPerThread.insert(resourceRequest.notifyNumPerThread.end(),
                                              intraTempRequest.notifyNumPerThread.begin(),
                                              intraTempRequest.notifyNumPerThread.end());
    resourceRequest.notifyNumPerThread.emplace_back(interTempRequest.notifyNumOnMainThread + 1);
    resourceRequest.notifyNumPerThread.insert(resourceRequest.notifyNumPerThread.end(),
                                              interTempRequest.notifyNumPerThread.begin(),
                                              interTempRequest.notifyNumPerThread.end());

    // v2.0 Fix 4: Assert invariants after successful merge per design §6.1 step 4
    CHK_PRT_RET(resourceRequest.slaveThreadNum == 0,
                HCCL_ERROR("[InsV2AlltoAllParallelExecutor][CalcRes] slaveThreadNum[%u] must be > 0",
                           resourceRequest.slaveThreadNum),
                HcclResult::HCCL_E_INTERNAL);
    u32 expectedNotifySize = intraTempRequest.notifyNumPerThread.size() +
                             interTempRequest.notifyNumPerThread.size() + 2;  // +2 for two template main threads
    CHK_PRT_RET(resourceRequest.notifyNumPerThread.size() != expectedNotifySize,
                HCCL_ERROR("[InsV2AlltoAllParallelExecutor][CalcRes] notifyNumPerThread.size()[%zu] != expected[%u]",
                           resourceRequest.notifyNumPerThread.size(), expectedNotifySize),
                HcclResult::HCCL_E_INTERNAL);

    if (param.engine != COMM_ENGINE_CCU) {
        CHK_PRT_RET(intraTempRequest.channels.empty() || interTempRequest.channels.empty(),
                    HCCL_ERROR("[InsV2AlltoAllParallelExecutor][CalcRes] intraTemplate or interTemplate has empty channels."),
                    HcclResult::HCCL_E_INTERNAL);
        resourceRequest.channels.emplace_back(intraTempRequest.channels[0]);
        resourceRequest.channels.emplace_back(interTempRequest.channels[0]);
    } else {
        HCCL_INFO("[InsV2AlltoAllParallelExecutor][CalcRes] intraTemplate has [%d] kernels.",
                  intraTempRequest.ccuKernelNum[0]);
        resourceRequest.ccuKernelInfos.insert(resourceRequest.ccuKernelInfos.end(),
                                              intraTempRequest.ccuKernelInfos.begin(),
                                              intraTempRequest.ccuKernelInfos.end());
        resourceRequest.ccuKernelNum.emplace_back(intraTempRequest.ccuKernelNum[0]);
        HCCL_INFO("[InsV2AlltoAllParallelExecutor][CalcRes] interTemplate has [%d] kernels.",
                  interTempRequest.ccuKernelNum[0]);
        resourceRequest.ccuKernelInfos.insert(resourceRequest.ccuKernelInfos.end(),
                                              interTempRequest.ccuKernelInfos.begin(),
                                              interTempRequest.ccuKernelInfos.end());
        resourceRequest.ccuKernelNum.emplace_back(interTempRequest.ccuKernelNum[0]);
    }

    HCCL_DEBUG("[InsV2AlltoAllParallelExecutor][CalcRes] myRank[%u], notifyNumOnMainThread[%u], slaveThreadNum[%u], "
               "channels[%u]",
               myRank_, resourceRequest.notifyNumOnMainThread, resourceRequest.slaveThreadNum,
               resourceRequest.channels.size());
    for (auto i = 0; i < resourceRequest.notifyNumPerThread.size(); i++) {
        HCCL_DEBUG("[InsV2AlltoAllParallelExecutor][CalcRes] myRank[%u], notifyNumPerThread[%u]=[%u]", myRank_, i,
                   resourceRequest.notifyNumPerThread[i]);
    }

    return HCCL_SUCCESS;
}

template <typename AlgTopoMatch, typename InsAlgTemplate0, typename InsAlgTemplate1>
void InsV2AlltoAllParallelExecutor<AlgTopoMatch, InsAlgTemplate0, InsAlgTemplate1>::GenTemplateAlgParamsIntra0(
    const OpParam &param, const AlgResourceCtxSerializable &resCtx, const u64 dataOffset,
    const u64 dataCountPerLoopAxis0, const u64 scratchOffset, TemplateDataParams &tempAlgParamsIntra0) const
{
    // Stage 1: input → INTRA scratch. Ring exchange writes received data to scratch,
    // not user output, so Stage 2 can read it back.
    tempAlgParamsIntra0.buffInfo.inputPtr = param.inputPtr;
    tempAlgParamsIntra0.buffInfo.outputPtr = resCtx.cclMem.addr;
    tempAlgParamsIntra0.buffInfo.hcclBuff = resCtx.cclMem;
    tempAlgParamsIntra0.buffInfo.inBuffType = BufferType::INPUT;
    tempAlgParamsIntra0.buffInfo.outBuffType = BufferType::HCCL_BUFFER;
    tempAlgParamsIntra0.buffInfo.hcclBuffType = BufferType::HCCL_BUFFER;
    tempAlgParamsIntra0.buffInfo.inputSize = param.inputSize;
    tempAlgParamsIntra0.buffInfo.outputSize = param.outputSize;

    tempAlgParamsIntra0.buffInfo.inBuffBaseOff = dataOffset;
    tempAlgParamsIntra0.buffInfo.outBuffBaseOff = scratchOffset;
    tempAlgParamsIntra0.buffInfo.hcclBuffBaseOff = scratchOffset;
    tempAlgParamsIntra0.sliceSize = dataCountPerLoopAxis0 * dataTypeSize_;
    tempAlgParamsIntra0.count = dataCountPerLoopAxis0;
    tempAlgParamsIntra0.tailSize = tempAlgParamsIntra0.sliceSize;

    tempAlgParamsIntra0.inputSliceStride = 0;
    tempAlgParamsIntra0.outputSliceStride = dataSize_;
    tempAlgParamsIntra0.repeatNum = 1;
    tempAlgParamsIntra0.inputRepeatStride = 0;
    tempAlgParamsIntra0.outputRepeatStride = 0;
    tempAlgParamsIntra0.enableRemoteMemAccess = param.opMode == OpMode::OFFLOAD;

    HCCL_DEBUG(
        "[InsV2AlltoAllParallelExecutor][GenTemplateAlgParamsIntra0] rank[%d] inBuffBaseOff[%llu] "
        "outBuffBaseOff[%llu] scratchBuffBaseOff[%llu] sliceSize[%llu] "
        "rankSizeLevel0[%u] rankSizeLevel1[%u] rankIdxLevel0[%u] rankIdxLevel1[%u]",
        myRank_, tempAlgParamsIntra0.buffInfo.inBuffBaseOff, tempAlgParamsIntra0.buffInfo.outBuffBaseOff,
        tempAlgParamsIntra0.buffInfo.hcclBuffBaseOff, tempAlgParamsIntra0.sliceSize,
        rankSizeLevel0_, rankSizeLevel1_, rankIdxLevel0_, rankIdxLevel1_);
    return;
}

template <typename AlgTopoMatch, typename InsAlgTemplate0, typename InsAlgTemplate1>
void InsV2AlltoAllParallelExecutor<AlgTopoMatch, InsAlgTemplate0, InsAlgTemplate1>::GenTemplateAlgParamsInter1(
    const OpParam &param, const AlgResourceCtxSerializable &resCtx, const u64 dataOffset,
    const u64 dataCountPerLoopAxis1, const u64 scratchOffset, TemplateDataParams &tempAlgParamsInter1) const
{
    // Stage 1: input → INTER scratch. Ring exchange writes received data to scratch,
    // not user output, so Stage 2 can read it back.
    tempAlgParamsInter1.buffInfo.inputPtr = param.inputPtr;
    tempAlgParamsInter1.buffInfo.outputPtr = resCtx.cclMem.addr;
    tempAlgParamsInter1.buffInfo.hcclBuff = resCtx.cclMem;
    tempAlgParamsInter1.buffInfo.inBuffType = BufferType::INPUT;
    tempAlgParamsInter1.buffInfo.outBuffType = BufferType::HCCL_BUFFER;
    tempAlgParamsInter1.buffInfo.hcclBuffType = BufferType::HCCL_BUFFER;
    tempAlgParamsInter1.buffInfo.inputSize = param.inputSize;
    tempAlgParamsInter1.buffInfo.outputSize = param.outputSize;

    tempAlgParamsInter1.buffInfo.inBuffBaseOff = dataOffset;
    tempAlgParamsInter1.buffInfo.outBuffBaseOff = scratchOffset;
    tempAlgParamsInter1.buffInfo.hcclBuffBaseOff = scratchOffset;
    tempAlgParamsInter1.sliceSize = dataCountPerLoopAxis1 * dataTypeSize_;
    tempAlgParamsInter1.count = dataCountPerLoopAxis1;
    tempAlgParamsInter1.tailSize = tempAlgParamsInter1.sliceSize;

    tempAlgParamsInter1.inputSliceStride = 0;
    tempAlgParamsInter1.outputSliceStride = dataSize_;
    tempAlgParamsInter1.repeatNum = 1;
    tempAlgParamsInter1.inputRepeatStride = 0;
    tempAlgParamsInter1.outputRepeatStride = 0;
    tempAlgParamsInter1.enableRemoteMemAccess = param.opMode == OpMode::OFFLOAD;

    HCCL_DEBUG("[InsV2AlltoAllParallelExecutor][GenTemplateAlgParamsInter1] rank[%u] inBuffBaseOff[%llu] "
               "outBuffBaseOff[%llu] scratchBuffBaseOff[%llu] sliceSize[%llu]",
               myRank_, tempAlgParamsInter1.buffInfo.inBuffBaseOff, tempAlgParamsInter1.buffInfo.outBuffBaseOff,
               tempAlgParamsInter1.buffInfo.hcclBuffBaseOff, tempAlgParamsInter1.sliceSize);
    return;
}

template <typename AlgTopoMatch, typename InsAlgTemplate0, typename InsAlgTemplate1>
void InsV2AlltoAllParallelExecutor<AlgTopoMatch, InsAlgTemplate0, InsAlgTemplate1>::GenTemplateAlgParamsInter0(
    const OpParam &param, const AlgResourceCtxSerializable &resCtx, const u64 dataOffset,
    const u64 dataCountPerLoopAxis0, const u64 scratchOffset, TemplateDataParams &tempAlgParamsInter0) const
{
    // Stage 2: READS from INTRA scratch → writes to user output buffer.
    // inputPtr = scratch (where Stage 1 Intra0 wrote), outputPtr = user output.
    tempAlgParamsInter0.buffInfo.inputPtr = resCtx.cclMem.addr;
    tempAlgParamsInter0.buffInfo.outputPtr = param.outputPtr;
    tempAlgParamsInter0.buffInfo.hcclBuff = resCtx.cclMem;
    tempAlgParamsInter0.buffInfo.inBuffType = BufferType::HCCL_BUFFER;
    tempAlgParamsInter0.buffInfo.outBuffType = BufferType::OUTPUT;
    tempAlgParamsInter0.buffInfo.hcclBuffType = BufferType::HCCL_BUFFER;
    tempAlgParamsInter0.buffInfo.inputSize = param.inputSize;
    tempAlgParamsInter0.buffInfo.outputSize = param.outputSize;

    tempAlgParamsInter0.buffInfo.inBuffBaseOff = scratchOffset;
    tempAlgParamsInter0.buffInfo.outBuffBaseOff = dataOffset;
    tempAlgParamsInter0.buffInfo.hcclBuffBaseOff = scratchOffset;
    tempAlgParamsInter0.sliceSize = dataCountPerLoopAxis0 * dataTypeSize_;
    tempAlgParamsInter0.count = dataCountPerLoopAxis0;
    tempAlgParamsInter0.tailSize = tempAlgParamsInter0.sliceSize;

    tempAlgParamsInter0.inputSliceStride = dataSize_ * rankSizeLevel0_;
    tempAlgParamsInter0.outputSliceStride = dataSize_;
    tempAlgParamsInter0.repeatNum = 1;
    tempAlgParamsInter0.inputRepeatStride = 0;
    tempAlgParamsInter0.outputRepeatStride = 0;
    tempAlgParamsInter0.enableRemoteMemAccess = param.opMode == OpMode::OFFLOAD;

    HCCL_DEBUG("[InsV2AlltoAllParallelExecutor][GenTemplateAlgParamsInter0] rank[%u] inBuffBaseOff[%llu] "
               "outBuffBaseOff[%llu] scratchBuffBaseOff[%llu] sliceSize[%llu]",
               myRank_, tempAlgParamsInter0.buffInfo.inBuffBaseOff, tempAlgParamsInter0.buffInfo.outBuffBaseOff,
               tempAlgParamsInter0.buffInfo.hcclBuffBaseOff, tempAlgParamsInter0.sliceSize);
    return;
}

template <typename AlgTopoMatch, typename InsAlgTemplate0, typename InsAlgTemplate1>
void InsV2AlltoAllParallelExecutor<AlgTopoMatch, InsAlgTemplate0, InsAlgTemplate1>::GenTemplateAlgParamsIntra1(
    const OpParam &param, const AlgResourceCtxSerializable &resCtx, const u64 dataOffset,
    const u64 dataCountPerLoopAxis1, const u64 scratchOffset, TemplateDataParams &tempAlgParamsIntra1) const
{
    tempAlgParamsIntra1.buffInfo.inBuffBaseOff = scratchOffset;
    tempAlgParamsIntra1.buffInfo.outBuffBaseOff = dataOffset;
    tempAlgParamsIntra1.buffInfo.hcclBuffBaseOff = scratchOffset;
    tempAlgParamsIntra1.buffInfo.inBuffType = BufferType::HCCL_BUFFER;
    tempAlgParamsIntra1.buffInfo.outBuffType = BufferType::OUTPUT;
    tempAlgParamsIntra1.buffInfo.hcclBuffType = BufferType::HCCL_BUFFER;
    tempAlgParamsIntra1.buffInfo.inputSize = param.inputSize;
    tempAlgParamsIntra1.buffInfo.outputSize = param.outputSize;
    tempAlgParamsIntra1.sliceSize = dataCountPerLoopAxis1 * dataTypeSize_;
    tempAlgParamsIntra1.count = dataCountPerLoopAxis1;
    tempAlgParamsIntra1.tailSize = tempAlgParamsIntra1.sliceSize;

    tempAlgParamsIntra1.inputSliceStride = dataSize_;
    tempAlgParamsIntra1.outputSliceStride = dataSize_;
    tempAlgParamsIntra1.repeatNum = 1;
    tempAlgParamsIntra1.inputRepeatStride = 0;
    tempAlgParamsIntra1.outputRepeatStride = 0;
    tempAlgParamsIntra1.enableRemoteMemAccess = param.opMode == OpMode::OFFLOAD;

    HCCL_DEBUG("[InsV2AlltoAllParallelExecutor][GenTemplateAlgParamsIntra1] rank[%u] inBuffBaseOff[%llu] "
               "outBuffBaseOff[%llu] scratchBuffBaseOff[%llu] sliceSize[%llu]",
               myRank_, tempAlgParamsIntra1.buffInfo.inBuffBaseOff, tempAlgParamsIntra1.buffInfo.outBuffBaseOff,
               tempAlgParamsIntra1.buffInfo.hcclBuffBaseOff, tempAlgParamsIntra1.sliceSize);
    return;
}

template <typename AlgTopoMatch, typename InsAlgTemplate0, typename InsAlgTemplate1>
uint64_t InsV2AlltoAllParallelExecutor<AlgTopoMatch, InsAlgTemplate0, InsAlgTemplate1>::GetRankSize(
    const std::vector<std::vector<u32>> &vTopo) const
{
    uint64_t count = 1;
    for (const auto &i : vTopo) {
        count *= i.size();
    }
    return count;
}

template <typename AlgTopoMatch, typename InsAlgTemplate0, typename InsAlgTemplate1>
HcclResult InsV2AlltoAllParallelExecutor<AlgTopoMatch, InsAlgTemplate0, InsAlgTemplate1>::Orchestrate(
    const OpParam &param, const AlgResourceCtxSerializable &resCtx)
{
    HCCL_INFO("[InsV2AlltoAllParallelExecutor][Orchestrate] Orchestrate Start");
    maxTmpMemSize_ = resCtx.cclMem.size;
    myRank_ = resCtx.topoInfo.userRank;
    threads_ = resCtx.threads;

    if (param.engine != CommEngine::COMM_ENGINE_AIV && param.engine != CommEngine::COMM_ENGINE_CCU) {
        CHK_RET(RestoreChannelMap(resCtx, remoteRankToChannelInfo_));
        intraLinkMap_ = remoteRankToChannelInfo_[0];
        interLinkMap_ = remoteRankToChannelInfo_[1];

        HCCL_INFO("[Orchestrate] interLinkMap_ size=%zu", interLinkMap_.size());
        for (auto &kv : interLinkMap_) {
            HCCL_INFO("[Orchestrate] interLinkMap_ rank=%u channels=%zu", kv.first, kv.second.size());
        }
    }
    dataCount_ = param.DataDes.count;
    dataType_ = param.DataDes.dataType;
    dataTypeSize_ = DATATYPE_SIZE_TABLE[param.DataDes.dataType];
    dataSize_ = dataCount_ * dataTypeSize_;

    if (resCtx.topoInfo.level0Topo == Level0Shape::MESH_1D_CLOS && !resCtx.topoInfo.level0PcieMix) {
        intraHierarchyInfo_ = {resCtx.algHierarchyInfo.infos[0][0]};
        std::vector<u32> closRanks;
        u32 meshSize = resCtx.algHierarchyInfo.infos[0][0].size();
        for (auto rank : resCtx.algHierarchyInfo.infos[0][1]) {
            if (rank % meshSize == resCtx.topoInfo.userRank % meshSize) {
                closRanks.push_back(rank);
            }
        }
        interHierarchyInfo_ = {closRanks};
    } else {
        intraHierarchyInfo_ = resCtx.algHierarchyInfo.infos[0];
        interHierarchyInfo_ = resCtx.algHierarchyInfo.infos[1];
    }
    rankSizeLevel0_ = GetRankSize(intraHierarchyInfo_);
    rankSizeLevel1_ = GetRankSize(interHierarchyInfo_);
    rankIdxLevel0_ = myRank_ % rankSizeLevel0_;
    rankIdxLevel1_ = myRank_ / rankSizeLevel0_;

    InsAlgTemplate0 intraTempAlg(param, resCtx.topoInfo.userRank, intraHierarchyInfo_);
    InsAlgTemplate1 interTempAlg(param, resCtx.topoInfo.userRank, interHierarchyInfo_);

    // Set 2D grid dimensions on templates
    intraTempAlg.xRankSize_ = rankSizeLevel0_;
    intraTempAlg.yRankSize_ = rankSizeLevel1_;
    intraTempAlg.totalRankSize_ = rankSizeLevel0_ * rankSizeLevel1_;
    intraTempAlg.myXRank_ = rankIdxLevel0_;
    intraTempAlg.myYRank_ = rankIdxLevel1_;

    interTempAlg.xRankSize_ = rankSizeLevel0_;
    interTempAlg.yRankSize_ = rankSizeLevel1_;
    interTempAlg.totalRankSize_ = rankSizeLevel0_ * rankSizeLevel1_;
    interTempAlg.myXRank_ = rankIdxLevel0_;
    interTempAlg.myYRank_ = rankIdxLevel1_;

    if (param.engine == CommEngine::COMM_ENGINE_AICPU_TS) {
        interTempAlg.SetchannelsPerRank(interLinkMap_);
    }

    PrepareResForTemplate(intraTempAlg, interTempAlg);

    HcclResult ret = OrchestrateLoop(param, resCtx, intraTempAlg, interTempAlg);
    CHK_PRT_RET(
        ret != HCCL_SUCCESS,
        HCCL_ERROR("[InsV2AlltoAllParallelExecutor][Orchestrate]errNo[0x%016llx] AlltoAll executor kernel run failed",
                   HCCL_ERROR_CODE(ret)),
        ret);
    return HCCL_SUCCESS;
}

template <typename AlgTopoMatch, typename InsAlgTemplate0, typename InsAlgTemplate1>
HcclResult InsV2AlltoAllParallelExecutor<AlgTopoMatch, InsAlgTemplate0, InsAlgTemplate1>::PrepareResForTemplate(
    InsAlgTemplate0 &tempAlgIntra, InsAlgTemplate1 &tempAlgInter)
{
    AlgResourceRequest intraTempRequest;
    AlgResourceRequest interTempRequest;
    tempAlgIntra.GetRes(intraTempRequest);
    tempAlgInter.GetRes(interTempRequest);
    auto intraThreadsNum = intraTempRequest.slaveThreadNum + 1;
    auto interThreadsNum = interTempRequest.slaveThreadNum + 1;
    auto intraNotifyOnMainThread = intraTempRequest.notifyNumOnMainThread;
    auto interNotifyOnMainThread = interTempRequest.notifyNumOnMainThread;

    intraThreads_.assign(threads_.begin() + 1, threads_.begin() + intraThreadsNum + 1);
    interThreads_.assign(threads_.begin() + intraThreadsNum + 1, threads_.end());
    mainThread_ = threads_.at(0);
    templateMainThreads_.emplace_back(intraThreads_.at(0));
    templateMainThreads_.emplace_back(interThreads_.at(0));
    syncNotifyOnTemplates_ = {intraNotifyOnMainThread, interNotifyOnMainThread};
    syncNotifyOnMain_ = {0, 1};
    return HCCL_SUCCESS;
}

template <typename AlgTopoMatch, typename InsAlgTemplate0, typename InsAlgTemplate1>
void InsV2AlltoAllParallelExecutor<AlgTopoMatch, InsAlgTemplate0, InsAlgTemplate1>::GetParallelDataSplit(
    std::vector<float> &splitDataSize) const
{
    double splitData = multipleDimensionSplitRatio_;

    // v2.0 Fix 6: adaptive splitRatio for degenerate topologies
    if (rankSizeLevel0_ == 1) {
        splitDataSize.push_back(0.0f);
        splitDataSize.push_back(1.0f);
    } else if (rankSizeLevel1_ == 1) {
        splitDataSize.push_back(1.0f);
        splitDataSize.push_back(0.0f);
    } else {
        splitDataSize.push_back(static_cast<float>(splitData));
        splitDataSize.push_back(1.0f - static_cast<float>(splitData));
    }
    HCCL_INFO("[InsV2AlltoAllParallelExecutor] splitDataSize is %f, %f", splitDataSize[0], splitDataSize[1]);
    return;
}

template <typename AlgTopoMatch, typename InsAlgTemplate0, typename InsAlgTemplate1>
HcclResult InsV2AlltoAllParallelExecutor<AlgTopoMatch, InsAlgTemplate0, InsAlgTemplate1>::OrchestrateLoop(
    const OpParam &param, const AlgResourceCtxSerializable &resCtx, InsAlgTemplate0 &tempAlgIntra,
    InsAlgTemplate1 &tempAlgInter)
{
    HCCL_INFO("[InsV2AlltoAllParallelExecutor] AlgTemplate intra server is [%s]", tempAlgIntra.Describe().c_str());
    HCCL_INFO("[InsV2AlltoAllParallelExecutor] AlgTemplate inter server is [%s]", tempAlgInter.Describe().c_str());
    multipleDimensionSplitRatio_ = param.multipleDimensionSplitRatio;

    std::vector<float> splitDataSize;
    GetParallelDataSplit(splitDataSize);

    u32 intraScratchMultipleStage0 = tempAlgIntra.CalcScratchMultiple(BufferType::INPUT, BufferType::OUTPUT);
    u32 interScratchMultipleStage0 = tempAlgInter.CalcScratchMultiple(BufferType::INPUT, BufferType::OUTPUT);
    u32 intraScratchMultipleStage1 = tempAlgIntra.CalcScratchMultiple(BufferType::HCCL_BUFFER, BufferType::OUTPUT);
    u32 interScratchMultipleStage1 = tempAlgInter.CalcScratchMultiple(BufferType::HCCL_BUFFER, BufferType::OUTPUT);

    u32 scratchMultipleIntra =
        static_cast<u32>(std::max(std::ceil(splitDataSize[0] * intraScratchMultipleStage0),
                                  std::ceil(splitDataSize[1] * intraScratchMultipleStage1)));
    u32 scratchMultipleInter =
        static_cast<u32>(std::max(std::ceil(splitDataSize[1] * interScratchMultipleStage0),
                                  std::ceil(splitDataSize[0] * interScratchMultipleStage1)));
    u32 totalScratchMultiple = scratchMultipleIntra + scratchMultipleInter;

    u64 scratchMemBlockSize = maxTmpMemSize_;
    u64 transportBoundDataSize = UB_MAX_DATA_SIZE;
    if (totalScratchMultiple > 0) {
        scratchMemBlockSize = (maxTmpMemSize_ / HCCL_MIN_SLICE_ALIGN / totalScratchMultiple) * HCCL_MIN_SLICE_ALIGN;
        scratchMemBlockSize = std::min(scratchMemBlockSize, transportBoundDataSize);
    }
    u64 intraScratchOffset = 0;
    u64 interScratchOffset = scratchMultipleIntra * scratchMemBlockSize;

    u64 maxCountPerLoop =
        (std::min(static_cast<u64>(scratchMemBlockSize), static_cast<u64>(UB_MAX_DATA_SIZE)) / dataTypeSize_ / 10) *
        10;

    u64 alignSize = AICPU_ALIGN_SIZE;

    u64 dataCountPerLoopAxis0 = static_cast<u64>(splitDataSize[0] * maxCountPerLoop);
    u64 dataCountPerLoopAxis1 = maxCountPerLoop - dataCountPerLoopAxis0;

    if (dataCountPerLoopAxis0 * dataTypeSize_ >= alignSize) {
        dataCountPerLoopAxis0 = dataCountPerLoopAxis0 * dataTypeSize_ / alignSize * alignSize / dataTypeSize_;
    }
    if (dataCountPerLoopAxis1 * dataTypeSize_ >= alignSize) {
        dataCountPerLoopAxis1 = dataCountPerLoopAxis1 * dataTypeSize_ / alignSize * alignSize / dataTypeSize_;
    }
    maxCountPerLoop = dataCountPerLoopAxis0 + dataCountPerLoopAxis1;

    u32 loopTimes = dataCount_ / maxCountPerLoop + ((dataCount_ % maxCountPerLoop == 0) ? 0 : 1);

    TemplateResource interTempAlgRes;
    interTempAlgRes.channels = interLinkMap_;
    interTempAlgRes.threads = interThreads_;
    interTempAlgRes.aivCommInfoPtr = resCtx.aivCommInfoPtr;

    TemplateResource intraTempAlgRes;
    intraTempAlgRes.channels = intraLinkMap_;
    intraTempAlgRes.threads = intraThreads_;
    intraTempAlgRes.aivCommInfoPtr = resCtx.aivCommInfoPtr;

    TemplateDataParams tempAlgParamsIntra0;
    TemplateDataParams tempAlgParamsInter0;
    TemplateDataParams tempAlgParamsInter1;
    TemplateDataParams tempAlgParamsIntra1;

    // v2.0 Fix 4: scratch size verification assertion
    u64 totalSliceSize = dataSize_;
    u64 intraScratchSize = scratchMultipleIntra * scratchMemBlockSize;
    u64 interScratchSize = scratchMultipleInter * scratchMemBlockSize;
    if (splitDataSize[0] > 0 &&
        intraScratchSize < static_cast<u64>(splitDataSize[0] * totalSliceSize)) {
        HCCL_ERROR("[InsV2AlltoAllParallelExecutor] Intra scratch[%llu] insufficient for routed data[%llu]",
                   intraScratchSize, static_cast<u64>(splitDataSize[0] * totalSliceSize));
        return HcclResult::HCCL_E_INTERNAL;
    }
    if (splitDataSize[1] > 0 &&
        interScratchSize < static_cast<u64>(splitDataSize[1] * totalSliceSize)) {
        HCCL_ERROR("[InsV2AlltoAllParallelExecutor] Inter scratch[%llu] insufficient for routed data[%llu]",
                   interScratchSize, static_cast<u64>(splitDataSize[1] * totalSliceSize));
        return HcclResult::HCCL_E_INTERNAL;
    }

    if (param.engine == COMM_ENGINE_CCU) {
        intraTempAlgRes.ccuKernels.insert(intraTempAlgRes.ccuKernels.end(),
                                          resCtx.ccuKernels.begin(),
                                          resCtx.ccuKernels.begin() + resCtx.ccuKernelNum[0]);
        interTempAlgRes.ccuKernels.insert(interTempAlgRes.ccuKernels.end(),
                                          resCtx.ccuKernels.begin() + resCtx.ccuKernelNum[0],
                                          resCtx.ccuKernels.begin() + resCtx.ccuKernelNum[0] +
                                              resCtx.ccuKernelNum[1]);
    } else {
        intraTempAlgRes.channels = intraLinkMap_;
        interTempAlgRes.channels = interLinkMap_;
    }

    u64 finalDataCountPerLoopAxis0 = dataCountPerLoopAxis0;
    u64 finalDataCountPerLoopAxis1 = dataCountPerLoopAxis1;

    if (loopTimes > 1) {
        u64 finalCount = dataCount_ - (loopTimes - 1) * maxCountPerLoop;
        finalDataCountPerLoopAxis0 = static_cast<u64>(splitDataSize[0] * finalCount);
        finalDataCountPerLoopAxis1 = finalCount - finalDataCountPerLoopAxis0;
    } else {
        finalDataCountPerLoopAxis0 = static_cast<u64>(splitDataSize[0] * dataCount_);
        finalDataCountPerLoopAxis1 = dataCount_ - finalDataCountPerLoopAxis0;
    }

    u64 totalRankCount = rankSizeLevel0_ * rankSizeLevel1_;

    for (u32 loopIndex = 0; loopIndex < loopTimes; loopIndex++) {
        u64 currCountPart0 = (loopIndex == loopTimes - 1) ? finalDataCountPerLoopAxis0 : dataCountPerLoopAxis0;
        u64 currCountPart1 = (loopIndex == loopTimes - 1) ? finalDataCountPerLoopAxis1 : dataCountPerLoopAxis1;

        // Stage 1 PreSync
        CHK_RET(PreSyncInterThreads(mainThread_, templateMainThreads_, syncNotifyOnTemplates_));

        u64 dataOffset0 = loopIndex * maxCountPerLoop * dataTypeSize_;
        u64 dataOffset1 = dataOffset0 + currCountPart0 * dataTypeSize_;

        // v3.0 Fix A: element-level guard for small data
        u64 currPerPeerChunkSize0 =
            (currCountPart0 * dataTypeSize_ + totalRankCount - 1) / totalRankCount;
        u64 currPerPeerChunkSize1 =
            (currCountPart1 * dataTypeSize_ + totalRankCount - 1) / totalRankCount;

        if (currPerPeerChunkSize0 < dataTypeSize_ && currCountPart0 > 0) {
            currPerPeerChunkSize0 = currCountPart0 * dataTypeSize_;
        }
        if (currPerPeerChunkSize1 < dataTypeSize_ && currCountPart1 > 0) {
            currPerPeerChunkSize1 = currCountPart1 * dataTypeSize_;
        }

        // Stage 1: Intra0 (X-axis mesh) + Inter1 (Y-axis clos)
        // Executed sequentially on the orchestrator's main thread.
        // The templates run their internal sub-thread PreSync/PostSync in parallel,
        // but KernelRun calls are sequential by-design (same pattern as AllGather executor).
        // Future optimization: thread-pool dispatch for true intra-template parallelism.
        GenTemplateAlgParamsIntra0(param, resCtx, dataOffset0, currCountPart0, intraScratchOffset,
                                   tempAlgParamsIntra0);
        HcclResult intra0Ret = tempAlgIntra.KernelRun(param, tempAlgParamsIntra0, intraTempAlgRes);

        GenTemplateAlgParamsInter1(param, resCtx, dataOffset1, currCountPart1, interScratchOffset,
                                   tempAlgParamsInter1);
        HcclResult inter1Ret = tempAlgInter.KernelRun(param, tempAlgParamsInter1, interTempAlgRes);

        // Stage 1 PostSync — MUST always execute, even on template errors.
        // The RAII guard in KernelRun guarantees template main threads signal notify,
        // so this PostSync will not hang regardless of intra0Ret/inter1Ret.
        HcclResult syncRet1 = PostSyncInterThreads(mainThread_, templateMainThreads_, syncNotifyOnMain_);

        // v2.0 Fix 3: Error aggregation AFTER PostSync per design §10.2
        // Check template errors first (root cause), then sync errors (framework).
        if (intra0Ret != HCCL_SUCCESS || inter1Ret != HCCL_SUCCESS) {
            if (syncRet1 != HCCL_SUCCESS) {
                HCCL_ERROR("[InsV2AlltoAllParallelExecutor] Stage 1 PostSync also failed: 0x%016llx",
                           HCCL_ERROR_CODE(syncRet1));
            }
            HcclResult templateErr = (intra0Ret != HCCL_SUCCESS) ? intra0Ret : inter1Ret;
            HCCL_ERROR("[InsV2AlltoAllParallelExecutor] Stage 1 template failed: 0x%016llx",
                       HCCL_ERROR_CODE(templateErr));
            return templateErr;
        }
        CHK_RET(syncRet1);

#ifndef AICPU_COMPILE
        if (loopTimes == 1 && param.engine == CommEngine::COMM_ENGINE_CCU) {
            ccuKernelLaunchNumIntra0_ = intraTempAlgRes.submitInfos.size();
            ccuKernelLaunchNumInter1_ = interTempAlgRes.submitInfos.size();
        }
#endif

        // Stage 2 PreSync
        CHK_RET(PreSyncInterThreads(mainThread_, templateMainThreads_, syncNotifyOnTemplates_));

        // v2.0 Fix 2: Inter0 reads from INTRA scratch, Intra1 reads from INTER scratch
        GenTemplateAlgParamsInter0(param, resCtx, dataOffset0, currCountPart0, intraScratchOffset,
                                   tempAlgParamsInter0);
        HcclResult inter0Ret = tempAlgInter.KernelRun(param, tempAlgParamsInter0, interTempAlgRes);

        GenTemplateAlgParamsIntra1(param, resCtx, dataOffset1, currCountPart1, interScratchOffset,
                                   tempAlgParamsIntra1);
        HcclResult intra1Ret = tempAlgIntra.KernelRun(param, tempAlgParamsIntra1, intraTempAlgRes);

        // Stage 2 PostSync — MUST always execute, even on template errors
        HcclResult syncRet2 = PostSyncInterThreads(mainThread_, templateMainThreads_, syncNotifyOnMain_);

        // v2.0 Fix 3: Error aggregation AFTER PostSync per design §10.2
        // Check template errors first (root cause), then sync errors (framework).
        if (inter0Ret != HCCL_SUCCESS || intra1Ret != HCCL_SUCCESS) {
            if (syncRet2 != HCCL_SUCCESS) {
                HCCL_ERROR("[InsV2AlltoAllParallelExecutor] Stage 2 PostSync also failed: 0x%016llx",
                           HCCL_ERROR_CODE(syncRet2));
            }
            HcclResult templateErr = (inter0Ret != HCCL_SUCCESS) ? inter0Ret : intra1Ret;
            HCCL_ERROR("[InsV2AlltoAllParallelExecutor] Stage 2 template failed: 0x%016llx",
                       HCCL_ERROR_CODE(templateErr));
            return templateErr;
        }
        CHK_RET(syncRet2);
    }

#ifndef AICPU_COMPILE
    if (loopTimes == 1 && param.engine == CommEngine::COMM_ENGINE_CCU && param.opMode != OpMode::OFFLOAD) {
        CHK_RET(FastLaunchSaveCtx(param, intraTempAlgRes, interTempAlgRes, resCtx.notifyNumOnMainThread));
    }
#endif

    HCCL_INFO("[InsV2AlltoAllParallelExecutor][OrchestrateLoop] End.");
    return HcclResult::HCCL_SUCCESS;
}

#ifndef AICPU_COMPILE
template <typename AlgTopoMatch, typename InsAlgTemplate0, typename InsAlgTemplate1>
HcclResult InsV2AlltoAllParallelExecutor<AlgTopoMatch, InsAlgTemplate0, InsAlgTemplate1>::FastLaunchSaveCtx(
    const OpParam &param, const TemplateResource &templateAlgResIntra, const TemplateResource &templateAlgResInter,
    u32 notifyNumOnMainThread)
{
    HCCL_INFO("[InsV2AlltoAllParallelExecutor] loopTimes==1, save fast launch ctx.");
    ccuKernelLaunchNumIntra1_ = templateAlgResIntra.submitInfos.size() - ccuKernelLaunchNumIntra0_;
    ccuKernelLaunchNumInter0_ = templateAlgResInter.submitInfos.size() - ccuKernelLaunchNumInter1_;
    u32 threadNum = threads_.size();
    u32 ccuKernelNum =
        ccuKernelLaunchNumIntra1_ + ccuKernelLaunchNumInter0_ + ccuKernelLaunchNumIntra0_ + ccuKernelLaunchNumInter1_;
    if (ccuKernelNum < 1) {
        HCCL_INFO("[InsV2AlltoAllParallelExecutor] ccu kernel num is 0, no need to save.");
        return HCCL_SUCCESS;
    }
    HCCL_INFO("[InsV2AlltoAllParallelExecutor][FastLaunchSaveCtx] threadNum[%llu], ccuKernelNum[%llu]", threadNum,
              ccuKernelNum);

    std::vector<u32> ccuKernelNumList = {ccuKernelLaunchNumIntra0_, ccuKernelLaunchNumInter1_,
                                         ccuKernelLaunchNumInter0_, ccuKernelLaunchNumIntra1_};
    std::vector<std::vector<CcuKernelSubmitInfo>> submitInfosList = {templateAlgResIntra.submitInfos,
                                                                     templateAlgResInter.submitInfos};
    return FastLaunchSaveCtxTwoTemplate(param, threadNum, ccuKernelNum, threads_, ccuKernelNumList, submitInfosList,
                                        notifyNumOnMainThread);
}

template <typename AlgTopoMatch, typename InsAlgTemplate0, typename InsAlgTemplate1>
HcclResult InsV2AlltoAllParallelExecutor<AlgTopoMatch, InsAlgTemplate0, InsAlgTemplate1>::FastLaunch(
    const OpParam &param, const CcuFastLaunchCtx *ctx)
{
    InsAlgTemplate0 intraTempAlg{};
    InsAlgTemplate1 interTempAlg{};

    TemplateFastLaunchCtx tempFastLaunchCtxIntra0, tempFastLaunchCtxInter0;
    TemplateFastLaunchCtx tempFastLaunchCtxInter1, tempFastLaunchCtxIntra1;

    TemplateResource templateAlgResIntra, templateAlgResInter;
    ThreadHandle *threads = ctx->GetThreadHandlePtr();
    threads_.assign(threads, threads + ctx->threadNum);
    PrepareResForTemplate(intraTempAlg, interTempAlg);

    CcuKernelSubmitInfo *ccuKernelSubmitInfos = ctx->GetCcuKernelSubmitInfoPtr();

    HCCL_INFO("[InsV2AlltoAllParallelExecutor][FastLaunch] Intra0 ccuKernelNum[%llu]", ctx->ccuKernelNum[0]);
    CHK_RET(PreSyncInterThreads(mainThread_, templateMainThreads_, syncNotifyOnTemplates_));

    CHK_RET(SetTempFastLaunchAddr(tempFastLaunchCtxIntra0, param.inputPtr, param.outputPtr, param.hcclBuff));
    tempFastLaunchCtxIntra0.threads = intraThreads_;
    tempFastLaunchCtxIntra0.ccuKernelSubmitInfos.assign(ccuKernelSubmitInfos,
                                                        ccuKernelSubmitInfos + ctx->ccuKernelNum[0]);
    ccuKernelSubmitInfos += ctx->ccuKernelNum[0];
    if (ctx->ccuKernelNum[0] > 0) {
        CHK_RET(intraTempAlg.FastLaunch(param, tempFastLaunchCtxIntra0));
    }

    CHK_RET(SetTempFastLaunchAddr(tempFastLaunchCtxInter1, param.inputPtr, param.outputPtr, param.hcclBuff));
    tempFastLaunchCtxInter1.threads = interThreads_;
    tempFastLaunchCtxInter1.ccuKernelSubmitInfos.assign(ccuKernelSubmitInfos,
                                                        ccuKernelSubmitInfos + ctx->ccuKernelNum[1]);
    ccuKernelSubmitInfos += ctx->ccuKernelNum[1];
    if (ctx->ccuKernelNum[1] > 0) {
        CHK_RET(interTempAlg.FastLaunch(param, tempFastLaunchCtxInter1));
    }

    CHK_RET(PostSyncInterThreads(mainThread_, templateMainThreads_, syncNotifyOnMain_));

    CHK_RET(PreSyncInterThreads(mainThread_, templateMainThreads_, syncNotifyOnTemplates_));

    CHK_RET(SetTempFastLaunchAddr(tempFastLaunchCtxInter0, param.outputPtr, param.outputPtr, param.hcclBuff));
    tempFastLaunchCtxInter0.threads = interThreads_;
    tempFastLaunchCtxInter0.ccuKernelSubmitInfos.assign(ccuKernelSubmitInfos,
                                                        ccuKernelSubmitInfos + ctx->ccuKernelNum[2]);
    ccuKernelSubmitInfos += ctx->ccuKernelNum[2];
    if (ctx->ccuKernelNum[2] > 0) {
        CHK_RET(interTempAlg.FastLaunch(param, tempFastLaunchCtxInter0));
    }

    CHK_RET(SetTempFastLaunchAddr(tempFastLaunchCtxIntra1, param.outputPtr, param.outputPtr, param.hcclBuff));
    tempFastLaunchCtxIntra1.threads = intraThreads_;
    tempFastLaunchCtxIntra1.ccuKernelSubmitInfos.assign(ccuKernelSubmitInfos,
                                                        ccuKernelSubmitInfos + ctx->ccuKernelNum[3]);
    if (ctx->ccuKernelNum[3] > 0) {
        CHK_RET(intraTempAlg.FastLaunch(param, tempFastLaunchCtxIntra1));
    }

    CHK_RET(PostSyncInterThreads(mainThread_, templateMainThreads_, syncNotifyOnMain_));

    HCCL_INFO("[InsV2AlltoAllParallelExecutor][FastLaunch] End.");
    return HCCL_SUCCESS;
}
#endif

// UBX topology (8-card boards) — Mesh intra, Clos inter
REGISTER_EXECUTOR_BY_TWO_TEMPS(
    HcclCMDType::HCCL_CMD_ALLTOALL,
    InsAlltoAllParallelMesh2DClosV2,
    InsV2AlltoAllParallelExecutor,
    TopoMatchUBX,
    InsTempAlltoAllMesh2DV2,
    InsTempAlltoAllMeshClosV2);

REGISTER_EXECUTOR_BY_TWO_TEMPS(HcclCMDType::HCCL_CMD_ALLTOALL,
    InsAlltoAllParallelMesh2DClosV2_ClosMesh2D,
    InsV2AlltoAllParallelExecutor,
    TopoMatchClosMesh2DV2,
    InsTempAlltoAllMesh2DV2,
    InsTempAlltoAllMeshClosV2);

REGISTER_EXECUTOR_BY_TWO_TEMPS(HcclCMDType::HCCL_CMD_ALLTOALL,
    InsAlltoAllParallelMesh2DClosV2_ClosMesh2DUBX,
    InsV2AlltoAllParallelExecutor,
    TopoMatchClosMesh2DUBXV2,
    InsTempAlltoAllMesh2DV2,
    InsTempAlltoAllMeshClosV2);

}  // namespace ops_hccl
