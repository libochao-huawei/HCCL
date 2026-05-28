/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "ins_v2_alltoall_parallel_opt_executor.h"
#include <alloca.h>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include "hcomm_primitives.h"
#include "alg_data_trans_wrapper.h"
#include "ins_temp_alltoall_mesh_2d_v3.h"
#include "ins_temp_alltoall_mesh_clos_v3.h"

#include "topo_match_clos_mesh_2d_v2.h"
#include "topo_match_clos_mesh_2d_ubx_v2.h"
#include "topo_match_ubx.h"

namespace ops_hccl {

namespace {
constexpr float kBandwidthFm   = 150.0f;
constexpr float kBandwidthClos = 200.0f;
constexpr float kDefaultBeta   = kBandwidthClos / (kBandwidthFm + kBandwidthClos);

float GetEnvRatio()
{
    const char *env = std::getenv("HCCL_AG_PARALLEL_RATIO");
    if (env != nullptr) {
        float ratio = std::atof(env);
        if (ratio > 0.0f && ratio < 1.0f) {
            return ratio;
        }
    }
    return kDefaultBeta;
}
}  // namespace

template <typename AlgTopoMatch, typename InsAlgTemplate0, typename InsAlgTemplate1>
InsV2AlltoAllParallelOptExecutor<AlgTopoMatch, InsAlgTemplate0, InsAlgTemplate1>::InsV2AlltoAllParallelOptExecutor()
{
}

template <typename AlgTopoMatch, typename InsAlgTemplate0, typename InsAlgTemplate1>
HcclResult InsV2AlltoAllParallelOptExecutor<AlgTopoMatch, InsAlgTemplate0, InsAlgTemplate1>::CalcAlgHierarchyInfo(
    HcclComm comm, TopoInfoWithNetLayerDetails *topoInfo, AlgHierarchyInfoForAllLevel &algHierarchyInfo)
{
    AlgTopoMatch topoMatch;
    CHK_RET(topoMatch.MatchTopo(comm, topoInfo, algHierarchyInfo));
    return HCCL_SUCCESS;
}

template <typename AlgTopoMatch, typename InsAlgTemplate0, typename InsAlgTemplate1>
HcclResult InsV2AlltoAllParallelOptExecutor<AlgTopoMatch, InsAlgTemplate0, InsAlgTemplate1>::CalcRes(
    HcclComm comm, const OpParam &param, const TopoInfoWithNetLayerDetails *topoInfo,
    const AlgHierarchyInfoForAllLevel &algHierarchyInfo, AlgResourceRequest &resourceRequest)
{
    myRank_ = topoInfo->userRank;

    resourceRequest = AlgResourceRequest{};

    std::vector<std::vector<u32>> intraHierarchyInfo;
    std::vector<std::vector<u32>> interHierarchyInfo;
    constexpr u32 TOPO_NUM = 2;
    CHK_PRT_RET(algHierarchyInfo.infos.size() < TOPO_NUM || algHierarchyInfo.infos[0].empty() ||
                algHierarchyInfo.infos[1].empty(),
                HCCL_ERROR("[InsV2AlltoAllParallelOptExecutor][CalcRes] Invalid topoInfo"),
                HcclResult::HCCL_E_INTERNAL);

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
        intraHierarchyInfo = algHierarchyInfo.infos[0];
        interHierarchyInfo = algHierarchyInfo.infos[1];
    }

    InsAlgTemplate0 intraTempAlg(param, topoInfo->userRank, intraHierarchyInfo);
    InsAlgTemplate1 interTempAlg(param, topoInfo->userRank, interHierarchyInfo);

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

    CHK_PRT_RET(resourceRequest.slaveThreadNum == 0,
                HCCL_ERROR("[InsV2AlltoAllParallelOptExecutor][CalcRes] slaveThreadNum[%u] must be > 0",
                           resourceRequest.slaveThreadNum),
                HcclResult::HCCL_E_INTERNAL);
    u32 expectedNotifySize = intraTempRequest.notifyNumPerThread.size() +
                             interTempRequest.notifyNumPerThread.size() + 2;
    CHK_PRT_RET(resourceRequest.notifyNumPerThread.size() != expectedNotifySize,
                HCCL_ERROR("[InsV2AlltoAllParallelOptExecutor][CalcRes] notifyNumPerThread.size()[%zu] != expected[%u]",
                           resourceRequest.notifyNumPerThread.size(), expectedNotifySize),
                HcclResult::HCCL_E_INTERNAL);

    if (param.engine != COMM_ENGINE_CCU) {
        CHK_PRT_RET(intraTempRequest.channels.empty() || interTempRequest.channels.empty(),
                    HCCL_ERROR("[InsV2AlltoAllParallelOptExecutor][CalcRes] empty channels."),
                    HcclResult::HCCL_E_INTERNAL);
        resourceRequest.channels.emplace_back(intraTempRequest.channels[0]);
        resourceRequest.channels.emplace_back(interTempRequest.channels[0]);
    } else {
        resourceRequest.ccuKernelInfos.insert(resourceRequest.ccuKernelInfos.end(),
                                              intraTempRequest.ccuKernelInfos.begin(),
                                              intraTempRequest.ccuKernelInfos.end());
        resourceRequest.ccuKernelNum.emplace_back(intraTempRequest.ccuKernelNum[0]);
        resourceRequest.ccuKernelInfos.insert(resourceRequest.ccuKernelInfos.end(),
                                              interTempRequest.ccuKernelInfos.begin(),
                                              interTempRequest.ccuKernelInfos.end());
        resourceRequest.ccuKernelNum.emplace_back(interTempRequest.ccuKernelNum[0]);
    }

    return HCCL_SUCCESS;
}

template <typename AlgTopoMatch, typename InsAlgTemplate0, typename InsAlgTemplate1>
uint64_t InsV2AlltoAllParallelOptExecutor<AlgTopoMatch, InsAlgTemplate0, InsAlgTemplate1>::GetRankSize(
    const std::vector<std::vector<u32>> &vTopo) const
{
    uint64_t count = 1;
    for (const auto &i : vTopo) {
        count *= i.size();
    }
    return count;
}

template <typename AlgTopoMatch, typename InsAlgTemplate0, typename InsAlgTemplate1>
float InsV2AlltoAllParallelOptExecutor<AlgTopoMatch, InsAlgTemplate0, InsAlgTemplate1>::CalcDynamicSplitRatio() const
{
    return GetEnvRatio();
}

template <typename AlgTopoMatch, typename InsAlgTemplate0, typename InsAlgTemplate1>
void InsV2AlltoAllParallelOptExecutor<AlgTopoMatch, InsAlgTemplate0, InsAlgTemplate1>::GetParallelDataSplit(
    std::vector<float> &splitDataSize) const
{
    double splitData = multipleDimensionSplitRatio_;

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
}

template <typename AlgTopoMatch, typename InsAlgTemplate0, typename InsAlgTemplate1>
HcclResult InsV2AlltoAllParallelOptExecutor<AlgTopoMatch, InsAlgTemplate0, InsAlgTemplate1>::Orchestrate(
    const OpParam &param, const AlgResourceCtxSerializable &resCtx)
{
    HCCL_INFO("[InsV2AlltoAllParallelOptExecutor][Orchestrate] Orchestrate Start");
    maxTmpMemSize_ = resCtx.cclMem.size;
    myRank_ = resCtx.topoInfo.userRank;
    threads_ = resCtx.threads;

    if (param.engine != CommEngine::COMM_ENGINE_AIV && param.engine != CommEngine::COMM_ENGINE_CCU) {
        CHK_RET(RestoreChannelMap(resCtx, remoteRankToChannelInfo_));
        intraLinkMap_ = remoteRankToChannelInfo_[0];
        if (remoteRankToChannelInfo_.size() >= 2) {
            interLinkMap_ = remoteRankToChannelInfo_[1];
        }
    }
    if (param.opType == HcclCMDType::HCCL_CMD_ALLTOALL ||
        param.opType == HcclCMDType::HCCL_CMD_ALLTOALLV) {
        dataType_ = param.all2AllVDataDes.sendType;
        u64* sendCounts = reinterpret_cast<u64*>(param.all2AllVDataDes.sendCounts);
        u64 totalRanks = resCtx.topoInfo.userRankSize;
        u64 totalCount = 0;
        for (u64 i = 0; i < totalRanks; i++) {
            totalCount += sendCounts[i];
        }
        dataCount_ = totalCount;
    } else {
        dataCount_ = param.DataDes.count;
        dataType_ = param.DataDes.dataType;
    }
    dataTypeSize_ = DATATYPE_SIZE_TABLE[dataType_];
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

    multipleDimensionSplitRatio_ = CalcDynamicSplitRatio();
    HCCL_INFO("[InsV2AlltoAllParallelOptExecutor][Orchestrate] dynamic split ratio = %f", multipleDimensionSplitRatio_);

    InsAlgTemplate0 intraTempAlg(param, resCtx.topoInfo.userRank, intraHierarchyInfo_);
    InsAlgTemplate1 interTempAlg(param, resCtx.topoInfo.userRank, interHierarchyInfo_);

    intraTempAlg.SetMeshDimensions(rankSizeLevel0_, rankSizeLevel1_, rankIdxLevel0_, rankIdxLevel1_);
    interTempAlg.SetMeshDimensions(rankSizeLevel0_, rankSizeLevel1_, rankIdxLevel0_, rankIdxLevel1_);

    if (param.engine == CommEngine::COMM_ENGINE_AICPU_TS ||
        param.engine == CommEngine::COMM_ENGINE_AIV) {
        interTempAlg.SetchannelsPerRank(interLinkMap_);
    }

    intraTempAlg.SetPortCount(3);
    interTempAlg.SetPortCount(4);
    interTempAlg.SetSharedPortMode(true);
    interTempAlg.SetSharedLinkRatio(0.8);

    bool hasInterComm = !interLinkMap_.empty();
    HCCL_INFO("[InsV2AlltoAllParallelOptExecutor][Orchestrate] hasInterComm=%d", hasInterComm);

    PrepareResForTemplate(intraTempAlg, interTempAlg);

    HcclResult ret = OrchestrateLoop(param, resCtx, intraTempAlg, interTempAlg, hasInterComm);
    CHK_PRT_RET(
        ret != HCCL_SUCCESS,
        HCCL_ERROR("[InsV2AlltoAllParallelOptExecutor][Orchestrate]errNo[0x%016llx] failed",
                   HCCL_ERROR_CODE(ret)),
        ret);
    return HCCL_SUCCESS;
}

template <typename AlgTopoMatch, typename InsAlgTemplate0, typename InsAlgTemplate1>
HcclResult InsV2AlltoAllParallelOptExecutor<AlgTopoMatch, InsAlgTemplate0, InsAlgTemplate1>::PrepareResForTemplate(
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
HcclResult InsV2AlltoAllParallelOptExecutor<AlgTopoMatch, InsAlgTemplate0, InsAlgTemplate1>::ReorganizeScratches_v1_1(
    ThreadHandle thread, void *intraBuf, void *interBuf,
    u32 xSize, u32 ySize, u64 cellSizeMax)
{
    if (cellSizeMax == 0) return HCCL_SUCCESS;

    uint8_t *intra = static_cast<uint8_t*>(intraBuf);
    uint8_t *inter = static_cast<uint8_t*>(interBuf);

    uint8_t *temp = static_cast<uint8_t*>(alloca(2 * cellSizeMax));
    if (temp == nullptr) return HCCL_E_INTERNAL;

    for (u32 sy = 0; sy < ySize; sy++) {
        for (u32 sx = sy + 1; sx < xSize; sx++) {
            if (sx >= ySize || sy >= xSize) continue;

            u64 intraOff_sx_sy = cellSizeMax * (sy * xSize + sx);
            u64 interOff_sx_sy = cellSizeMax * (sy * xSize + sx);

            u64 intraOff_sy_sx = cellSizeMax * (sx * xSize + sy);
            u64 interOff_sy_sx = cellSizeMax * (sx * xSize + sy);

            HcommLocalCopyOnThread(thread, temp, intra + intraOff_sx_sy, cellSizeMax);

            if (sy < xSize && sx < ySize) {
                HcommLocalCopyOnThread(thread, temp + cellSizeMax,
                    intra + intraOff_sy_sx, cellSizeMax);
            } else {
                memset(temp + cellSizeMax, 0, cellSizeMax);
            }

            if (sy < xSize && sx < ySize) {
                HcommLocalCopyOnThread(thread, intra + intraOff_sy_sx,
                    inter + interOff_sx_sy, cellSizeMax);
            }

            if (sy < xSize && sx < ySize) {
                HcommLocalCopyOnThread(thread, inter + interOff_sy_sx,
                    temp, cellSizeMax);
            }

            HcommLocalCopyOnThread(thread, inter + interOff_sx_sy,
                temp + cellSizeMax, cellSizeMax);
        }
    }
    return HCCL_SUCCESS;
}

template <typename AlgTopoMatch, typename InsAlgTemplate0, typename InsAlgTemplate1>
void InsV2AlltoAllParallelOptExecutor<AlgTopoMatch, InsAlgTemplate0, InsAlgTemplate1>::GenTemplateAlgParamsIntra0(
    const OpParam &param, const AlgResourceCtxSerializable &resCtx,
    u64 dataOffset, u64 dataCountPerLoop,
    u64 scratchOffset, u64 cellSizeMax,
    TemplateDataParams &out) const
{
    out.buffInfo.inputPtr = param.inputPtr;
    out.buffInfo.outputPtr = resCtx.cclMem.addr;
    out.buffInfo.hcclBuff = resCtx.cclMem;
    out.buffInfo.inBuffType = BufferType::INPUT;
    out.buffInfo.outBuffType = BufferType::HCCL_BUFFER;
    out.buffInfo.hcclBuffType = BufferType::HCCL_BUFFER;
    out.buffInfo.inputSize = param.inputSize;
    out.buffInfo.outputSize = param.outputSize;

    out.buffInfo.inBuffBaseOff = dataOffset;
    out.buffInfo.outBuffBaseOff = scratchOffset;
    out.buffInfo.hcclBuffBaseOff = scratchOffset;
    out.sliceSize = dataCountPerLoop * dataTypeSize_;
    out.count = dataCountPerLoop;
    out.tailSize = out.sliceSize;

    out.inputSliceStride = 0;
    out.outputSliceStride = cellSizeMax;
    out.repeatNum = 1;
    out.inputRepeatStride = 0;
    out.outputRepeatStride = 0;
    out.enableRemoteMemAccess = param.opMode == OpMode::OFFLOAD;
}

template <typename AlgTopoMatch, typename InsAlgTemplate0, typename InsAlgTemplate1>
void InsV2AlltoAllParallelOptExecutor<AlgTopoMatch, InsAlgTemplate0, InsAlgTemplate1>::GenTemplateAlgParamsInter1(
    const OpParam &param, const AlgResourceCtxSerializable &resCtx,
    u64 dataOffset, u64 dataCountPerLoop,
    u64 scratchOffset, u64 cellSizeMax,
    TemplateDataParams &out) const
{
    out.buffInfo.inputPtr = param.inputPtr;
    out.buffInfo.outputPtr = resCtx.cclMem.addr;
    out.buffInfo.hcclBuff = resCtx.cclMem;
    out.buffInfo.inBuffType = BufferType::INPUT;
    out.buffInfo.outBuffType = BufferType::HCCL_BUFFER;
    out.buffInfo.hcclBuffType = BufferType::HCCL_BUFFER;
    out.buffInfo.inputSize = param.inputSize;
    out.buffInfo.outputSize = param.outputSize;

    out.buffInfo.inBuffBaseOff = dataOffset;
    out.buffInfo.outBuffBaseOff = scratchOffset;
    out.buffInfo.hcclBuffBaseOff = scratchOffset;
    out.sliceSize = dataCountPerLoop * dataTypeSize_;
    out.count = dataCountPerLoop;
    out.tailSize = out.sliceSize;

    out.inputSliceStride = 0;
    out.outputSliceStride = cellSizeMax;
    out.repeatNum = 1;
    out.inputRepeatStride = 0;
    out.outputRepeatStride = 0;
    out.enableRemoteMemAccess = param.opMode == OpMode::OFFLOAD;
}

template <typename AlgTopoMatch, typename InsAlgTemplate0, typename InsAlgTemplate1>
void InsV2AlltoAllParallelOptExecutor<AlgTopoMatch, InsAlgTemplate0, InsAlgTemplate1>::GenTemplateAlgParamsInter0(
    const OpParam &param, const AlgResourceCtxSerializable &resCtx,
    u64 dataOffset, u64 dataCountPerLoop,
    u64 scratchOffset, u64 cellSizeMax,
    TemplateDataParams &out) const
{
    out.buffInfo.inputPtr = resCtx.cclMem.addr;
    out.buffInfo.outputPtr = param.outputPtr;
    out.buffInfo.hcclBuff = resCtx.cclMem;
    out.buffInfo.inBuffType = BufferType::HCCL_BUFFER;
    out.buffInfo.outBuffType = BufferType::OUTPUT;
    out.buffInfo.hcclBuffType = BufferType::HCCL_BUFFER;
    out.buffInfo.inputSize = param.inputSize;
    out.buffInfo.outputSize = param.outputSize;

    out.buffInfo.inBuffBaseOff = scratchOffset;
    out.buffInfo.outBuffBaseOff = dataOffset;
    out.buffInfo.hcclBuffBaseOff = scratchOffset;
    out.sliceSize = dataCountPerLoop * dataTypeSize_;
    out.count = dataCountPerLoop;
    out.tailSize = out.sliceSize;

    out.inputSliceStride = cellSizeMax * rankSizeLevel0_;
    out.outputSliceStride = dataSize_;
    out.repeatNum = 1;
    out.inputRepeatStride = 0;
    out.outputRepeatStride = 0;
    out.enableRemoteMemAccess = param.opMode == OpMode::OFFLOAD;
}

template <typename AlgTopoMatch, typename InsAlgTemplate0, typename InsAlgTemplate1>
void InsV2AlltoAllParallelOptExecutor<AlgTopoMatch, InsAlgTemplate0, InsAlgTemplate1>::GenTemplateAlgParamsIntra1(
    const OpParam &param, const AlgResourceCtxSerializable &resCtx,
    u64 dataOffset, u64 dataCountPerLoop,
    u64 scratchOffset, u64 cellSizeMax,
    TemplateDataParams &out) const
{
    out.buffInfo.inputPtr = static_cast<uint8_t*>(resCtx.cclMem.addr) + scratchOffset;
    out.buffInfo.outputPtr = param.outputPtr;
    out.buffInfo.hcclBuff = resCtx.cclMem;
    out.buffInfo.inBuffBaseOff = 0;
    out.buffInfo.outBuffBaseOff = dataOffset;
    out.buffInfo.hcclBuffBaseOff = scratchOffset;
    out.buffInfo.inBuffType = BufferType::HCCL_BUFFER;
    out.buffInfo.outBuffType = BufferType::OUTPUT;
    out.buffInfo.hcclBuffType = BufferType::HCCL_BUFFER;
    out.buffInfo.inputSize = param.inputSize;
    out.buffInfo.outputSize = param.outputSize;
    out.sliceSize = dataCountPerLoop * dataTypeSize_;
    out.count = dataCountPerLoop;
    out.tailSize = out.sliceSize;

    out.inputSliceStride = cellSizeMax;
    out.outputSliceStride = dataSize_;
    out.repeatNum = 1;
    out.inputRepeatStride = 0;
    out.outputRepeatStride = 0;
    out.enableRemoteMemAccess = param.opMode == OpMode::OFFLOAD;
}

template <typename AlgTopoMatch, typename InsAlgTemplate0, typename InsAlgTemplate1>
HcclResult InsV2AlltoAllParallelOptExecutor<AlgTopoMatch, InsAlgTemplate0, InsAlgTemplate1>::OrchestrateLoop(
    const OpParam &param, const AlgResourceCtxSerializable &resCtx,
    InsAlgTemplate0 &tempAlgIntra, InsAlgTemplate1 &tempAlgInter, bool hasInterComm)
{
    if (dataCount_ == 0) return HCCL_SUCCESS;

    std::vector<float> splitDataSize;
    if (!hasInterComm) {
        splitDataSize.push_back(1.0f);
        splitDataSize.push_back(0.0f);
    } else {
        GetParallelDataSplit(splitDataSize);
    }

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

    if (maxCountPerLoop == 0) return HCCL_SUCCESS;

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

        u64 dataOffset0 = loopIndex * maxCountPerLoop * dataTypeSize_;
        u64 dataOffset1 = dataOffset0 + currCountPart0 * dataTypeSize_;

        u64 perPeerChunk0 =
            (currCountPart0 * dataTypeSize_ + totalRankCount - 1) / totalRankCount;
        u64 perPeerChunk1 =
            (currCountPart1 * dataTypeSize_ + totalRankCount - 1) / totalRankCount;

        if (perPeerChunk0 < dataTypeSize_ && currCountPart0 > 0) {
            perPeerChunk0 = currCountPart0 * dataTypeSize_;
        }
        if (perPeerChunk1 < dataTypeSize_ && currCountPart1 > 0) {
            perPeerChunk1 = currCountPart1 * dataTypeSize_;
        }

        u64 cellSizeMax = std::max(perPeerChunk0, perPeerChunk1);

        CHK_RET(PreSyncInterThreads(mainThread_, templateMainThreads_, syncNotifyOnTemplates_));

        GenTemplateAlgParamsIntra0(param, resCtx, dataOffset0, currCountPart0,
                                    intraScratchOffset, cellSizeMax, tempAlgParamsIntra0);
        HcclResult intra0Ret = tempAlgIntra.KernelRun(param, tempAlgParamsIntra0, intraTempAlgRes);

        HcclResult inter1Ret = HCCL_SUCCESS;
        if (hasInterComm && currCountPart1 > 0) {
            GenTemplateAlgParamsInter1(param, resCtx, dataOffset1, currCountPart1,
                                        interScratchOffset, cellSizeMax, tempAlgParamsInter1);
            inter1Ret = tempAlgInter.KernelRun(param, tempAlgParamsInter1, interTempAlgRes);
        }

        HcclResult syncRet1 = PostSyncInterThreads(mainThread_, templateMainThreads_, syncNotifyOnMain_);

        if (intra0Ret != HCCL_SUCCESS || inter1Ret != HCCL_SUCCESS) {
            if (syncRet1 != HCCL_SUCCESS) {
                HCCL_ERROR("[InsV2AlltoAllParallelOptExecutor] Stage 1 PostSync also failed: 0x%016llx",
                           HCCL_ERROR_CODE(syncRet1));
            }
            HcclResult templateErr = (intra0Ret != HCCL_SUCCESS) ? intra0Ret : inter1Ret;
            return templateErr;
        }
        CHK_RET(syncRet1);

        {
            u64 meshSliceSize = currCountPart0 * dataTypeSize_;
            u64 closSliceSize = currCountPart1 * dataTypeSize_;

            bool needReorg = (rankSizeLevel0_ > 1 && rankSizeLevel1_ > 1 &&
                             (currCountPart0 > 0 || currCountPart1 > 0));

            if (needReorg) {
                uint8_t *intraBuf = static_cast<uint8_t*>(resCtx.cclMem.addr) + intraScratchOffset;
                uint8_t *interBuf = static_cast<uint8_t*>(resCtx.cclMem.addr) + interScratchOffset;

                HcclResult reorgRet = ReorganizeScratches_v1_1(mainThread_,
                    intraBuf, interBuf, rankSizeLevel0_, rankSizeLevel1_, cellSizeMax);
                CHK_RET(reorgRet);
            }
        }

        tempAlgIntra.SetPortCount(4);
        tempAlgIntra.SetBorrowedLink(true, 0);
        tempAlgInter.SetPortCount(3);
        tempAlgInter.SetSharedPortMode(false);

        CHK_RET(PreSyncInterThreads(mainThread_, templateMainThreads_, syncNotifyOnTemplates_));

        GenTemplateAlgParamsIntra1(param, resCtx, dataOffset1, currCountPart1,
                                    intraScratchOffset, cellSizeMax, tempAlgParamsIntra1);
        HcclResult intra1Ret = tempAlgIntra.KernelRun(param, tempAlgParamsIntra1, intraTempAlgRes);

        HcclResult inter0Ret = HCCL_SUCCESS;
        if (hasInterComm && currCountPart0 > 0) {
            GenTemplateAlgParamsInter0(param, resCtx, dataOffset0, currCountPart0,
                                        interScratchOffset, cellSizeMax, tempAlgParamsInter0);
            inter0Ret = tempAlgInter.KernelRun(param, tempAlgParamsInter0, interTempAlgRes);
        }

        HcclResult syncRet2 = PostSyncInterThreads(mainThread_, templateMainThreads_, syncNotifyOnMain_);

        if (inter0Ret != HCCL_SUCCESS || intra1Ret != HCCL_SUCCESS) {
            if (syncRet2 != HCCL_SUCCESS) {
                HCCL_ERROR("[InsV2AlltoAllParallelOptExecutor] Stage 2 PostSync also failed: 0x%016llx",
                           HCCL_ERROR_CODE(syncRet2));
            }
            HcclResult templateErr = (inter0Ret != HCCL_SUCCESS) ? inter0Ret : intra1Ret;
            return templateErr;
        }
        CHK_RET(syncRet2);

        tempAlgIntra.SetPortCount(3);
        tempAlgIntra.SetBorrowedLink(false, 0);
        tempAlgInter.SetPortCount(4);
        tempAlgInter.SetSharedPortMode(true);
        tempAlgInter.SetSharedLinkRatio(0.8);
    }

    HCCL_INFO("[InsV2AlltoAllParallelOptExecutor][OrchestrateLoop] End.");
    return HcclResult::HCCL_SUCCESS;
}

REGISTER_EXECUTOR_BY_TWO_TEMPS(
    HcclCMDType::HCCL_CMD_ALLTOALL,
    InsAlltoAllParallelMesh2DClosV3Opt,
    InsV2AlltoAllParallelOptExecutor,
    TopoMatchUBX,
    InsTempAlltoAllMesh2DV3,
    InsTempAlltoAllMeshClosV3);

}  // namespace ops_hccl
