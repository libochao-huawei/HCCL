/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "ins_v2_all_gather_parallel_opt_executor.h"
#include <cmath>
#include "alg_data_trans_wrapper.h"
#include "ins_temp_all_gather_mesh_1d_v2.h"
#include "ins_temp_all_gather_mesh_clos_v3.h"

namespace ops_hccl {

template <typename AlgTopoMatch, typename InsAlgTemplate0, typename InsAlgTemplate1,
          typename InsAlgTemplate2, typename InsAlgTemplate3>
InsV2AllGatherParallelOptExecutor<AlgTopoMatch, InsAlgTemplate0, InsAlgTemplate1,
                                   InsAlgTemplate2, InsAlgTemplate3>::InsV2AllGatherParallelOptExecutor()
{
}

template <typename AlgTopoMatch, typename InsAlgTemplate0, typename InsAlgTemplate1,
          typename InsAlgTemplate2, typename InsAlgTemplate3>
HcclResult InsV2AllGatherParallelOptExecutor<AlgTopoMatch, InsAlgTemplate0, InsAlgTemplate1,
                                              InsAlgTemplate2, InsAlgTemplate3>::CalcAlgHierarchyInfo(
    HcclComm comm, TopoInfoWithNetLayerDetails *topoInfo, AlgHierarchyInfoForAllLevel &algHierarchyInfo)
{
    AlgTopoMatch topoMatch;
    CHK_RET(topoMatch.MatchTopo(comm, topoInfo, algHierarchyInfo));
    return HCCL_SUCCESS;
}

template <typename AlgTopoMatch, typename InsAlgTemplate0, typename InsAlgTemplate1,
          typename InsAlgTemplate2, typename InsAlgTemplate3>
HcclResult InsV2AllGatherParallelOptExecutor<AlgTopoMatch, InsAlgTemplate0, InsAlgTemplate1,
                                              InsAlgTemplate2, InsAlgTemplate3>::CalcRes(
    HcclComm comm, const OpParam &param, const TopoInfoWithNetLayerDetails *topoInfo,
    const AlgHierarchyInfoForAllLevel &algHierarchyInfo, AlgResourceRequest &resourceRequest)
{
    myRank_ = topoInfo->userRank;

    std::vector<std::vector<u32>> intraHierarchyInfo;
    std::vector<std::vector<u32>> interHierarchyInfo;
    if (topoInfo->level0Topo == Level0Shape::MESH_1D_CLOS && !topoInfo->level0PcieMix) {
        intraHierarchyInfo = {algHierarchyInfo.infos[0][0]};
        std::vector<u32> closRanks;
        u32 meshSize = static_cast<u32>(algHierarchyInfo.infos[0][0].size());
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
                     HCCL_ERROR("[InsAllGatherParallelOptExecutor][CalcRes] Invalid topoInfo"),
                     HcclResult::HCCL_E_INTERNAL);
        intraHierarchyInfo = algHierarchyInfo.infos[0];
        interHierarchyInfo = algHierarchyInfo.infos[1];
    }

    InsAlgTemplate0 intraTempS1(param, topoInfo->userRank, intraHierarchyInfo);
    InsAlgTemplate1 interTempS1(param, topoInfo->userRank, interHierarchyInfo);
    InsAlgTemplate2 intraTempS2(param, topoInfo->userRank, intraHierarchyInfo);
    InsAlgTemplate3 interTempS2(param, topoInfo->userRank, interHierarchyInfo);

    AlgResourceRequest intraS1Request;
    AlgResourceRequest interS1Request;
    AlgResourceRequest intraS2Request;
    AlgResourceRequest interS2Request;
    CHK_RET(intraTempS1.CalcRes(comm, param, topoInfo, intraS1Request));
    CHK_RET(interTempS1.CalcRes(comm, param, topoInfo, interS1Request));
    CHK_RET(intraTempS2.CalcRes(comm, param, topoInfo, intraS2Request));
    CHK_RET(interTempS2.CalcRes(comm, param, topoInfo, interS2Request));

    constexpr u32 SUB_MAIN_THREAD_NUM = 4;
    resourceRequest.notifyNumOnMainThread = SUB_MAIN_THREAD_NUM;
    resourceRequest.slaveThreadNum = intraS1Request.slaveThreadNum + interS1Request.slaveThreadNum +
                                     intraS2Request.slaveThreadNum + interS2Request.slaveThreadNum + SUB_MAIN_THREAD_NUM;
    resourceRequest.notifyNumPerThread.emplace_back(intraS1Request.notifyNumOnMainThread + 1);
    resourceRequest.notifyNumPerThread.insert(resourceRequest.notifyNumPerThread.end(),
                                              intraS1Request.notifyNumPerThread.begin(),
                                              intraS1Request.notifyNumPerThread.end());
    resourceRequest.notifyNumPerThread.emplace_back(interS1Request.notifyNumOnMainThread + 1);
    resourceRequest.notifyNumPerThread.insert(resourceRequest.notifyNumPerThread.end(),
                                              interS1Request.notifyNumPerThread.begin(),
                                              interS1Request.notifyNumPerThread.end());
    resourceRequest.notifyNumPerThread.emplace_back(intraS2Request.notifyNumOnMainThread + 1);
    resourceRequest.notifyNumPerThread.insert(resourceRequest.notifyNumPerThread.end(),
                                              intraS2Request.notifyNumPerThread.begin(),
                                              intraS2Request.notifyNumPerThread.end());
    resourceRequest.notifyNumPerThread.emplace_back(interS2Request.notifyNumOnMainThread + 1);
    resourceRequest.notifyNumPerThread.insert(resourceRequest.notifyNumPerThread.end(),
                                              interS2Request.notifyNumPerThread.begin(),
                                              interS2Request.notifyNumPerThread.end());

    if (param.engine != COMM_ENGINE_CCU) {
        CHK_PRT_RET(intraS1Request.channels.empty() || interS1Request.channels.empty() ||
                     intraS2Request.channels.empty() || interS2Request.channels.empty(),
                     HCCL_ERROR("[InsAllGatherParallelOptExecutor][CalcRes] empty channels."),
                     HcclResult::HCCL_E_INTERNAL);
        resourceRequest.channels.emplace_back(intraS1Request.channels[0]);
        resourceRequest.channels.emplace_back(interS1Request.channels[0]);
        resourceRequest.channels.emplace_back(intraS2Request.channels[0]);
        resourceRequest.channels.emplace_back(interS2Request.channels[0]);
    } else {
        resourceRequest.ccuKernelInfos.insert(resourceRequest.ccuKernelInfos.end(),
                                              intraS1Request.ccuKernelInfos.begin(),
                                              intraS1Request.ccuKernelInfos.end());
        resourceRequest.ccuKernelNum.emplace_back(intraS1Request.ccuKernelNum[0]);
        resourceRequest.ccuKernelInfos.insert(resourceRequest.ccuKernelInfos.end(),
                                              interS1Request.ccuKernelInfos.begin(),
                                              interS1Request.ccuKernelInfos.end());
        resourceRequest.ccuKernelNum.emplace_back(interS1Request.ccuKernelNum[0]);
        resourceRequest.ccuKernelInfos.insert(resourceRequest.ccuKernelInfos.end(),
                                              intraS2Request.ccuKernelInfos.begin(),
                                              intraS2Request.ccuKernelInfos.end());
        resourceRequest.ccuKernelNum.emplace_back(intraS2Request.ccuKernelNum[0]);
        resourceRequest.ccuKernelInfos.insert(resourceRequest.ccuKernelInfos.end(),
                                              interS2Request.ccuKernelInfos.begin(),
                                              interS2Request.ccuKernelInfos.end());
        resourceRequest.ccuKernelNum.emplace_back(interS2Request.ccuKernelNum[0]);
    }

    return HCCL_SUCCESS;
}

template <typename AlgTopoMatch, typename InsAlgTemplate0, typename InsAlgTemplate1,
          typename InsAlgTemplate2, typename InsAlgTemplate3>
void InsV2AllGatherParallelOptExecutor<AlgTopoMatch, InsAlgTemplate0, InsAlgTemplate1,
                                        InsAlgTemplate2, InsAlgTemplate3>::GenTemplateAlgParamsIntra0(
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

template <typename AlgTopoMatch, typename InsAlgTemplate0, typename InsAlgTemplate1,
          typename InsAlgTemplate2, typename InsAlgTemplate3>
void InsV2AllGatherParallelOptExecutor<AlgTopoMatch, InsAlgTemplate0, InsAlgTemplate1,
                                        InsAlgTemplate2, InsAlgTemplate3>::GenTemplateAlgParamsInter0(
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
    tempAlgParamsInter0.repeatNum = static_cast<u32>(rankSizeLevel0_);
    tempAlgParamsInter0.inputRepeatStride = dataSize_;
    tempAlgParamsInter0.outputRepeatStride = dataSize_;
    tempAlgParamsInter0.enableRemoteMemAccess = param.opMode == OpMode::OFFLOAD;
}

template <typename AlgTopoMatch, typename InsAlgTemplate0, typename InsAlgTemplate1,
          typename InsAlgTemplate2, typename InsAlgTemplate3>
void InsV2AllGatherParallelOptExecutor<AlgTopoMatch, InsAlgTemplate0, InsAlgTemplate1,
                                        InsAlgTemplate2, InsAlgTemplate3>::GenTemplateAlgParamsInter1(
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

template <typename AlgTopoMatch, typename InsAlgTemplate0, typename InsAlgTemplate1,
          typename InsAlgTemplate2, typename InsAlgTemplate3>
void InsV2AllGatherParallelOptExecutor<AlgTopoMatch, InsAlgTemplate0, InsAlgTemplate1,
                                        InsAlgTemplate2, InsAlgTemplate3>::GenTemplateAlgParamsIntra1(
    const OpParam &param, const AlgResourceCtxSerializable &resCtx, const u64 dataOffset,
    const u64 dataCountPerLoopAixs1, const u64 scratchOffset, TemplateDataParams &tempAlgParamsIntra1) const
{
    tempAlgParamsIntra1.buffInfo.inputPtr = param.outputPtr;
    tempAlgParamsIntra1.buffInfo.outputPtr = param.outputPtr;
    tempAlgParamsIntra1.buffInfo.hcclBuff = resCtx.cclMem;
    tempAlgParamsIntra1.buffInfo.inBuffBaseOff = dataOffset;
    tempAlgParamsIntra1.buffInfo.outBuffBaseOff = dataOffset;
    tempAlgParamsIntra1.buffInfo.hcclBuffBaseOff = scratchOffset;
    tempAlgParamsIntra1.buffInfo.inBuffType = BufferType::OUTPUT;
    tempAlgParamsIntra1.buffInfo.outBuffType = BufferType::OUTPUT;
    tempAlgParamsIntra1.buffInfo.hcclBuffType = BufferType::HCCL_BUFFER;
    tempAlgParamsIntra1.buffInfo.inputSize = param.inputSize;
    tempAlgParamsIntra1.buffInfo.outputSize = param.outputSize;
    tempAlgParamsIntra1.sliceSize = dataCountPerLoopAixs1 * dataTypeSize_;
    tempAlgParamsIntra1.count = dataCountPerLoopAixs1;
    tempAlgParamsIntra1.tailSize = tempAlgParamsIntra1.sliceSize;

    tempAlgParamsIntra1.inputSliceStride = dataSize_;
    tempAlgParamsIntra1.outputSliceStride = dataSize_;
    tempAlgParamsIntra1.repeatNum = static_cast<u32>(rankSizeLevel1_);
    tempAlgParamsIntra1.inputRepeatStride = dataSize_ * rankSizeLevel0_;
    tempAlgParamsIntra1.outputRepeatStride = dataSize_ * rankSizeLevel0_;
    tempAlgParamsIntra1.enableRemoteMemAccess = param.opMode == OpMode::OFFLOAD;
}

template <typename AlgTopoMatch, typename InsAlgTemplate0, typename InsAlgTemplate1,
          typename InsAlgTemplate2, typename InsAlgTemplate3>
uint64_t InsV2AllGatherParallelOptExecutor<AlgTopoMatch, InsAlgTemplate0, InsAlgTemplate1,
                                            InsAlgTemplate2, InsAlgTemplate3>::GetRankSize(
    const std::vector<std::vector<u32>> &vTopo) const
{
    uint64_t count = 1;
    for (const auto &i : vTopo) {
        count *= i.size();
    }
    return count;
}

template <typename AlgTopoMatch, typename InsAlgTemplate0, typename InsAlgTemplate1,
          typename InsAlgTemplate2, typename InsAlgTemplate3>
void InsV2AllGatherParallelOptExecutor<AlgTopoMatch, InsAlgTemplate0, InsAlgTemplate1,
                                        InsAlgTemplate2, InsAlgTemplate3>::GetParallelDataSplit(
    std::vector<float> &splitDataSize) const
{
    double splitData = multipleDimensionSplitRatio_;
    splitDataSize.push_back(static_cast<float>(1 - splitData));
    splitDataSize.push_back(static_cast<float>(splitData));
    HCCL_INFO("[InsV2AllGatherParallelOptExecutor] splitDataSize is %f, %f", splitDataSize[0], splitDataSize[1]);
}

template <typename AlgTopoMatch, typename InsAlgTemplate0, typename InsAlgTemplate1,
          typename InsAlgTemplate2, typename InsAlgTemplate3>
HcclResult InsV2AllGatherParallelOptExecutor<AlgTopoMatch, InsAlgTemplate0, InsAlgTemplate1,
                                              InsAlgTemplate2, InsAlgTemplate3>::Orchestrate(
    const OpParam &param, const AlgResourceCtxSerializable &resCtx)
{
    HCCL_INFO("[InsV2AllGatherParallelOptExecutor][Orchestrate] Orchestrate Start");
    maxTmpMemSize_ = resCtx.cclMem.size;
    myRank_ = resCtx.topoInfo.userRank;
    threads_ = resCtx.threads;

    if (param.engine != CommEngine::COMM_ENGINE_AIV && param.engine != CommEngine::COMM_ENGINE_CCU) {
        CHK_RET(RestoreChannelMap(resCtx, remoteRankToChannelInfo_));
        intraLinkMap_ = remoteRankToChannelInfo_[0];
        interLinkMap_ = remoteRankToChannelInfo_[1];
    }

    dataCount_ = param.DataDes.count;
    dataType_ = param.DataDes.dataType;
    dataTypeSize_ = DATATYPE_SIZE_TABLE[param.DataDes.dataType];
    dataSize_ = dataCount_ * dataTypeSize_;

    if (resCtx.topoInfo.level0Topo == Level0Shape::MESH_1D_CLOS && !resCtx.topoInfo.level0PcieMix) {
        intraHierarchyInfo_ = {resCtx.algHierarchyInfo.infos[0][0]};
        std::vector<u32> closRanks;
        u32 meshSize = static_cast<u32>(resCtx.algHierarchyInfo.infos[0][0].size());
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

    const auto &infoCopy = resCtx.algHierarchyInfo;
    TwoStageTopoInfo twoStage;

    if (!infoCopy.infos.empty() && infoCopy.infos[0].size() >= 2) {
        u32 meshSize = static_cast<u32>(infoCopy.infos[0][0].size());
        std::vector<u32> closRanks;
        for (auto rank : infoCopy.infos[0][1]) {
            if (rank % meshSize == myRank_ % meshSize) {
                closRanks.push_back(rank);
            }
        }

        twoStage.stage1IntraRanks.push_back(infoCopy.infos[0][0]);
        twoStage.stage1InterRanks.push_back(closRanks);

        u32 borrowRank = INVALID_VALUE_RANKID;
        for (auto rank : closRanks) {
            if (rank / meshSize != myRank_ / meshSize) {
                borrowRank = rank;
                break;
            }
        }

        if (borrowRank != INVALID_VALUE_RANKID) {
            twoStage.borrowRank = borrowRank;
            twoStage.borrowLinkIdx = 0;
            twoStage.borrowChannel.isValid = true;
            twoStage.borrowChannel.remoteRank = borrowRank;

            twoStage.stage2IntraRanks = twoStage.stage1IntraRanks;
            if (!twoStage.stage2IntraRanks.empty()) {
                bool alreadyInIntra = false;
                for (auto rank : twoStage.stage2IntraRanks[0]) {
                    if (rank == borrowRank) {
                        alreadyInIntra = true;
                        break;
                    }
                }
                if (!alreadyInIntra) {
                    twoStage.stage2IntraRanks[0].push_back(borrowRank);
                }
            }

            twoStage.stage2InterRanks = twoStage.stage1InterRanks;
            for (auto &group : twoStage.stage2InterRanks) {
                group.erase(std::remove(group.begin(), group.end(), borrowRank), group.end());
            }
        }
    }

    intraHierarchyInfo_S2_ = twoStage.stage2IntraRanks.empty() ? intraHierarchyInfo_ : twoStage.stage2IntraRanks;
    interHierarchyInfo_S2_ = twoStage.stage2InterRanks.empty() ? interHierarchyInfo_ : twoStage.stage2InterRanks;

    u32 borrowRank = twoStage.borrowRank;

    double ratioClos = 200.0 / 350.0;
    {
        const char *env = getenv("HCCL_AG_PARALLEL_RATIO");
        if (env != nullptr) {
            ratioClos = std::atof(env);
        }
    }
    double sharedLinkRatio = 0.8;
    {
        const char *env = getenv("HCCL_CLOS_SHARED_LINK_RATIO");
        if (env != nullptr) {
            sharedLinkRatio = std::atof(env);
        }
    }

    multipleDimensionSplitRatio_ = ratioClos;

    InsAlgTemplate0 intraTempS1(param, resCtx.topoInfo.userRank, intraHierarchyInfo_);
    InsAlgTemplate1 interTempS1(param, resCtx.topoInfo.userRank, interHierarchyInfo_);
    InsAlgTemplate2 intraTempS2(param, resCtx.topoInfo.userRank, intraHierarchyInfo_S2_);
    InsAlgTemplate3 interTempS2(param, resCtx.topoInfo.userRank, interHierarchyInfo_S2_);

    if (param.engine == CommEngine::COMM_ENGINE_AICPU_TS) {
        interTempS1.SetchannelsPerRank(interLinkMap_);
        interTempS2.SetchannelsPerRank(interLinkMap_);
    }

    interTempS1.SetStageConfig(1, true);
    interTempS1.SetSharedLinkRatio(sharedLinkRatio);

    interTempS2.SetStageConfig(2, false);
    interTempS2.SetSharedLinkRatio(sharedLinkRatio);

    intraTempS2.SetPortCount(4);
    intraTempS2.SetBorrowEnabled(true);
    if (borrowRank != INVALID_VALUE_RANKID && twoStage.borrowChannel.isValid) {
        intraLinkMap_[borrowRank].push_back(twoStage.borrowChannel);
    }

    if (borrowRank != INVALID_VALUE_RANKID) {
        intraTempS2.SetDoubleLinkedNeighbor(borrowRank);
    }

    PrepareResForTemplates(intraTempS1, interTempS1, intraTempS2, interTempS2);

    CHK_RET(OrchestrateLoop(param, resCtx, intraTempS1, interTempS1, intraTempS2, interTempS2));

    return HCCL_SUCCESS;
}

template <typename AlgTopoMatch, typename InsAlgTemplate0, typename InsAlgTemplate1,
          typename InsAlgTemplate2, typename InsAlgTemplate3>
HcclResult InsV2AllGatherParallelOptExecutor<AlgTopoMatch, InsAlgTemplate0, InsAlgTemplate1,
                                              InsAlgTemplate2, InsAlgTemplate3>::PrepareResForTemplates(
    InsAlgTemplate0 &intraS1, InsAlgTemplate1 &interS1,
    InsAlgTemplate2 &intraS2, InsAlgTemplate3 &interS2)
{
    AlgResourceRequest intraS1Request;
    AlgResourceRequest interS1Request;
    AlgResourceRequest intraS2Request;
    AlgResourceRequest interS2Request;
    intraS1.GetRes(intraS1Request);
    interS1.GetRes(interS1Request);
    intraS2.GetRes(intraS2Request);
    interS2.GetRes(interS2Request);

    auto intraS1ThreadsNum = intraS1Request.slaveThreadNum + 1;
    auto interS1ThreadsNum = interS1Request.slaveThreadNum + 1;
    auto intraS2ThreadsNum = intraS2Request.slaveThreadNum + 1;
    auto interS2ThreadsNum = interS2Request.slaveThreadNum + 1;

    u32 cursor = 1;
    intraThreads_S1.assign(threads_.begin() + cursor, threads_.begin() + cursor + intraS1ThreadsNum);
    cursor += static_cast<u32>(intraS1ThreadsNum);
    interThreads_S1.assign(threads_.begin() + cursor, threads_.begin() + cursor + interS1ThreadsNum);
    cursor += static_cast<u32>(interS1ThreadsNum);
    intraThreads_S2.assign(threads_.begin() + cursor, threads_.begin() + cursor + intraS2ThreadsNum);
    cursor += static_cast<u32>(intraS2ThreadsNum);
    interThreads_S2.assign(threads_.begin() + cursor, threads_.begin() + cursor + interS2ThreadsNum);

    mainThread_ = threads_.at(0);
    templateMainThreads_.emplace_back(intraThreads_S1.at(0));
    templateMainThreads_.emplace_back(interThreads_S1.at(0));
    templateMainThreads_.emplace_back(intraThreads_S2.at(0));
    templateMainThreads_.emplace_back(interThreads_S2.at(0));

    syncNotifyOnTemplates_ = {
        static_cast<u32>(intraS1Request.notifyNumOnMainThread),
        static_cast<u32>(interS1Request.notifyNumOnMainThread),
        static_cast<u32>(intraS2Request.notifyNumOnMainThread),
        static_cast<u32>(interS2Request.notifyNumOnMainThread)};
    syncNotifyOnMain_ = {0, 1, 2, 3};

    return HCCL_SUCCESS;
}

template <typename AlgTopoMatch, typename InsAlgTemplate0, typename InsAlgTemplate1,
          typename InsAlgTemplate2, typename InsAlgTemplate3>
HcclResult InsV2AllGatherParallelOptExecutor<AlgTopoMatch, InsAlgTemplate0, InsAlgTemplate1,
                                              InsAlgTemplate2, InsAlgTemplate3>::OrchestrateLoop(
    const OpParam &param, const AlgResourceCtxSerializable &resCtx,
    InsAlgTemplate0 &intraS1, InsAlgTemplate1 &interS1,
    InsAlgTemplate2 &intraS2, InsAlgTemplate3 &interS2)
{
    HCCL_INFO("[InsV2AllGatherParallelOptExecutor] intraS1: [%s]", intraS1.Describe().c_str());
    HCCL_INFO("[InsV2AllGatherParallelOptExecutor] interS1: [%s]", interS1.Describe().c_str());
    HCCL_INFO("[InsV2AllGatherParallelOptExecutor] intraS2: [%s]", intraS2.Describe().c_str());
    HCCL_INFO("[InsV2AllGatherParallelOptExecutor] interS2: [%s]", interS2.Describe().c_str());

    if (dataCount_ == 0) {
        return HCCL_SUCCESS;
    }

    multipleDimensionSplitRatio_ = param.multipleDimensionSplitRatio;

    std::vector<float> dataSplitSize;
    GetParallelDataSplit(dataSplitSize);

    u32 intraScratchMultS1 = intraS1.CalcScratchMultiple(BufferType::INPUT, BufferType::OUTPUT);
    u32 interScratchMultS1 = interS1.CalcScratchMultiple(BufferType::INPUT, BufferType::OUTPUT);
    u32 intraScratchMultS2 = intraS2.CalcScratchMultiple(BufferType::INPUT, BufferType::OUTPUT);
    u32 interScratchMultS2 = interS2.CalcScratchMultiple(BufferType::INPUT, BufferType::OUTPUT);

    u32 scratchMultipleIntra = static_cast<u32>(std::max(
        std::ceil(dataSplitSize[0] * intraScratchMultS1),
        std::ceil(dataSplitSize[1] * intraScratchMultS2 * rankSizeLevel1_)));
    u32 scratchMultipleInter = static_cast<u32>(std::max(
        std::ceil(dataSplitSize[1] * interScratchMultS1),
        std::ceil(dataSplitSize[0] * interScratchMultS2 * rankSizeLevel0_)));

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
        (std::min(static_cast<u64>(scratchMemBlockSize), static_cast<u64>(UB_MAX_DATA_SIZE)) / dataTypeSize_ / 10) * 10;

    double ratioFM = 1.0 - multipleDimensionSplitRatio_;
    double ratioClos = multipleDimensionSplitRatio_;

    u64 alignSize = AICPU_ALIGN_SIZE;

    u64 unalignedPart0 = static_cast<u64>(ratioFM * maxCountPerLoop);
    u64 unalignedPart1 = maxCountPerLoop - unalignedPart0;

    u64 dataCountPart0 = unalignedPart0;
    u64 lostBytesPart0 = 0;
    if (unalignedPart0 * dataTypeSize_ >= alignSize) {
        u64 bytesBefore = unalignedPart0 * dataTypeSize_;
        dataCountPart0 = (bytesBefore / alignSize) * alignSize / dataTypeSize_;
        lostBytesPart0 = bytesBefore - (dataCountPart0 * dataTypeSize_);
    }

    u64 dataCountPart1 = unalignedPart1;
    u64 lostBytesPart1 = 0;
    if (unalignedPart1 * dataTypeSize_ >= alignSize) {
        u64 bytesBefore = unalignedPart1 * dataTypeSize_;
        dataCountPart1 = (bytesBefore / alignSize) * alignSize / dataTypeSize_;
        lostBytesPart1 = bytesBefore - (dataCountPart1 * dataTypeSize_);
    }

    u64 perLoopLoss = (lostBytesPart0 + lostBytesPart1) / dataTypeSize_;
    maxCountPerLoop = dataCountPart0 + dataCountPart1;

    if (maxCountPerLoop == 0) {
        return HCCL_SUCCESS;
    }

    u32 loopTimes = static_cast<u32>(dataCount_ / maxCountPerLoop);
    if (dataCount_ % maxCountPerLoop != 0) {
        loopTimes += 1;
    }

    u64 finalCountPart0 = dataCountPart0;
    u64 finalCountPart1 = dataCountPart1;
    u32 extraTailIteration = 0;
    u64 accumulatedLoss = 0;

    if (loopTimes > 1) {
        u64 preliminaryProcessed = (static_cast<u64>(loopTimes) - 1) * maxCountPerLoop;
        u64 remainingData = dataCount_ - preliminaryProcessed;
        u64 cumulativeLoss = (static_cast<u64>(loopTimes) - 1) * perLoopLoss;
        u64 adjustedRemaining = remainingData + cumulativeLoss;

        u64 tailPart0Unaligned = static_cast<u64>(ratioFM * adjustedRemaining);
        u64 tailPart0 = tailPart0Unaligned;
        if (tailPart0Unaligned * dataTypeSize_ >= alignSize) {
            u64 tailBytes0Un = tailPart0Unaligned * dataTypeSize_;
            tailPart0 = (tailBytes0Un / alignSize) * alignSize / dataTypeSize_;
        }
        u64 tailPart1 = adjustedRemaining - tailPart0;
        if (tailPart1 * dataTypeSize_ >= alignSize) {
            u64 tailBytes1Un = tailPart1 * dataTypeSize_;
            tailPart1 = (tailBytes1Un / alignSize) * alignSize / dataTypeSize_;
        }

        finalCountPart0 = tailPart0;
        finalCountPart1 = tailPart1;

        u64 processedInTail = finalCountPart0 + finalCountPart1;
        if (processedInTail < adjustedRemaining) {
            accumulatedLoss = adjustedRemaining - processedInTail;
            extraTailIteration = 1;
        } else if (processedInTail > adjustedRemaining) {
            finalCountPart1 -= (processedInTail - adjustedRemaining);
        }
    } else {
        u64 unaligned0 = static_cast<u64>(ratioFM * dataCount_);
        finalCountPart0 = unaligned0;
        if (unaligned0 * dataTypeSize_ >= alignSize) {
            finalCountPart0 = ((unaligned0 * dataTypeSize_) / alignSize) * alignSize / dataTypeSize_;
        }
        u64 unaligned1 = dataCount_ - unaligned0;
        finalCountPart1 = dataCount_ - finalCountPart0;
        if (unaligned1 * dataTypeSize_ >= alignSize) {
            u64 aligned1Bytes = (unaligned1 * dataTypeSize_) / alignSize * alignSize;
            finalCountPart1 = aligned1Bytes / dataTypeSize_;
            finalCountPart0 = dataCount_ - finalCountPart1;
        }
        u64 processed = finalCountPart0 + finalCountPart1;
        if (processed < dataCount_) {
            accumulatedLoss = dataCount_ - processed;
            extraTailIteration = 1;
        }
    }

    TemplateResource intraS1Res;
    intraS1Res.channels = intraLinkMap_;
    intraS1Res.threads = intraThreads_S1;
    intraS1Res.aivCommInfoPtr = resCtx.aivCommInfoPtr;

    TemplateResource interS1Res;
    interS1Res.channels = interLinkMap_;
    interS1Res.threads = interThreads_S1;
    interS1Res.aivCommInfoPtr = resCtx.aivCommInfoPtr;

    TemplateResource intraS2Res;
    intraS2Res.channels = intraLinkMap_;
    intraS2Res.threads = intraThreads_S2;
    intraS2Res.aivCommInfoPtr = resCtx.aivCommInfoPtr;

    TemplateResource interS2Res;
    interS2Res.channels = interLinkMap_;
    interS2Res.threads = interThreads_S2;
    interS2Res.aivCommInfoPtr = resCtx.aivCommInfoPtr;

    TemplateDataParams tempAlgParamsIntra0;
    TemplateDataParams tempAlgParamsInter0;
    TemplateDataParams tempAlgParamsInter1;
    TemplateDataParams tempAlgParamsIntra1;

    u32 totalLoopCount = loopTimes + extraTailIteration;

    for (u32 loopIndex = 0; loopIndex < totalLoopCount; loopIndex++) {
        u64 currPart0, currPart1;
        if (loopIndex < loopTimes - 1) {
            currPart0 = dataCountPart0;
            currPart1 = dataCountPart1;
        } else if (loopIndex == static_cast<u32>(loopTimes - 1)) {
            currPart0 = finalCountPart0;
            currPart1 = finalCountPart1;
        } else {
            currPart0 = static_cast<u64>(ratioFM * accumulatedLoss);
            currPart1 = accumulatedLoss - currPart0;
        }

        if (currPart0 == 0 && currPart1 == 0) {
            continue;
        }

        CHK_RET(PreSyncInterThreads(mainThread_, templateMainThreads_, syncNotifyOnTemplates_));

        u64 dataOffset0 = static_cast<u64>(loopIndex) * maxCountPerLoop * dataTypeSize_;
        u64 dataOffset1 = dataOffset0 + currPart0 * dataTypeSize_;

        GenTemplateAlgParamsIntra0(param, resCtx, dataOffset0, currPart0, intraScratchOffset, tempAlgParamsIntra0);
        CHK_RET(intraS1.KernelRun(param, tempAlgParamsIntra0, intraS1Res));

        GenTemplateAlgParamsInter1(param, resCtx, dataOffset1, currPart1, interScratchOffset, tempAlgParamsInter1);
        CHK_RET(interS1.KernelRun(param, tempAlgParamsInter1, interS1Res));

        CHK_RET(PostSyncInterThreads(mainThread_, templateMainThreads_, syncNotifyOnMain_));

#ifndef AICPU_COMPILE
        if (totalLoopCount == 1 && param.engine == CommEngine::COMM_ENGINE_CCU) {
            ccuKernelLaunchNumIntra0_ = static_cast<u32>(intraS1Res.submitInfos.size());
            ccuKernelLaunchNumInter1_ = static_cast<u32>(interS1Res.submitInfos.size());
        }
#endif

        CHK_RET(PreSyncInterThreads(mainThread_, templateMainThreads_, syncNotifyOnTemplates_));

        GenTemplateAlgParamsInter0(param, resCtx, dataOffset0, currPart0, intraScratchOffset, tempAlgParamsInter0);
        CHK_RET(interS2.KernelRun(param, tempAlgParamsInter0, interS2Res));

        GenTemplateAlgParamsIntra1(param, resCtx, dataOffset1, currPart1, interScratchOffset, tempAlgParamsIntra1);
        CHK_RET(intraS2.KernelRun(param, tempAlgParamsIntra1, intraS2Res));

        CHK_RET(PostSyncInterThreads(mainThread_, templateMainThreads_, syncNotifyOnMain_));
    }

#ifndef AICPU_COMPILE
    if (totalLoopCount == 1 && param.engine == CommEngine::COMM_ENGINE_CCU && param.opMode != OpMode::OFFLOAD) {
        CHK_RET(FastLaunchSaveCtx(param, intraS1Res, interS1Res, intraS2Res, interS2Res,
                                   static_cast<u32>(resCtx.notifyNumOnMainThread)));
    }
#endif

    HCCL_INFO("[InsV2AllGatherParallelOptExecutor][OrchestrateLoop] End.");
    return HcclResult::HCCL_SUCCESS;
}

#ifndef AICPU_COMPILE
template <typename AlgTopoMatch, typename InsAlgTemplate0, typename InsAlgTemplate1,
          typename InsAlgTemplate2, typename InsAlgTemplate3>
HcclResult InsV2AllGatherParallelOptExecutor<AlgTopoMatch, InsAlgTemplate0, InsAlgTemplate1,
                                              InsAlgTemplate2, InsAlgTemplate3>::FastLaunchSaveCtx(
    const OpParam &param, const TemplateResource &intraS1Res, const TemplateResource &interS1Res,
    const TemplateResource &intraS2Res, const TemplateResource &interS2Res, u32 notifyNumOnMainThread)
{
    HCCL_INFO("[InsV2AllGatherParallelOptExecutor] loopTimes==1, save fast launch ctx.");
    ccuKernelLaunchNumIntra1_ = static_cast<u32>(intraS2Res.submitInfos.size());
    ccuKernelLaunchNumInter0_ = static_cast<u32>(interS2Res.submitInfos.size());
    u32 threadNum = static_cast<u32>(threads_.size());
    u32 ccuKernelNum = ccuKernelLaunchNumIntra1_ + ccuKernelLaunchNumInter0_ +
                       ccuKernelLaunchNumIntra1_ + ccuKernelLaunchNumInter1_;
    if (ccuKernelNum < 1) {
        HCCL_INFO("[InsV2AllGatherParallelOptExecutor] ccu kernel num is 0, no need to save.");
        return HCCL_SUCCESS;
    }

    std::vector<u32> ccuKernelNumList = {ccuKernelLaunchNumIntra0_, ccuKernelLaunchNumInter1_,
                                         ccuKernelLaunchNumInter0_, ccuKernelLaunchNumIntra1_};
    std::vector<std::vector<CcuKernelSubmitInfo>> submitInfosList = {
        intraS1Res.submitInfos, interS1Res.submitInfos,
        intraS2Res.submitInfos, interS2Res.submitInfos};
    return FastLaunchSaveCtxTwoTemplate(param, threadNum, ccuKernelNum, threads_, ccuKernelNumList, submitInfosList,
                                        notifyNumOnMainThread);
}

template <typename AlgTopoMatch, typename InsAlgTemplate0, typename InsAlgTemplate1,
          typename InsAlgTemplate2, typename InsAlgTemplate3>
HcclResult InsV2AllGatherParallelOptExecutor<AlgTopoMatch, InsAlgTemplate0, InsAlgTemplate1,
                                              InsAlgTemplate2, InsAlgTemplate3>::FastLaunch(
    const OpParam &param, const CcuFastLaunchCtx *ctx)
{
    InsAlgTemplate0 intraTempS1{};
    InsAlgTemplate1 interTempS1{};
    InsAlgTemplate2 intraTempS2{};
    InsAlgTemplate3 interTempS2{};

    TemplateFastLaunchCtx tempFastLaunchCtxIntra0, tempFastLaunchCtxInter0;
    TemplateFastLaunchCtx tempFastLaunchCtxInter1, tempFastLaunchCtxIntra1;

    TemplateResource intraS1Res, interS1Res, intraS2Res, interS2Res;
    ThreadHandle *threads = ctx->GetThreadHandlePtr();
    threads_.assign(threads, threads + ctx->threadNum);
    PrepareResForTemplates(intraTempS1, interTempS1, intraTempS2, interTempS2);

    CcuKernelSubmitInfo *ccuKernelSubmitInfos = ctx->GetCcuKernelSubmitInfoPtr();

    CHK_RET(PreSyncInterThreads(mainThread_, templateMainThreads_, syncNotifyOnTemplates_));

    CHK_RET(SetTempFastLaunchAddr(tempFastLaunchCtxIntra0, param.inputPtr, param.outputPtr, param.hcclBuff));
    tempFastLaunchCtxIntra0.threads = intraThreads_S1;
    tempFastLaunchCtxIntra0.ccuKernelSubmitInfos.assign(ccuKernelSubmitInfos,
                                                         ccuKernelSubmitInfos + ctx->ccuKernelNum[0]);
    ccuKernelSubmitInfos += ctx->ccuKernelNum[0];
    if (ctx->ccuKernelNum[0] > 0) {
        CHK_RET(intraTempS1.FastLaunch(param, tempFastLaunchCtxIntra0));
    }

    CHK_RET(SetTempFastLaunchAddr(tempFastLaunchCtxInter1, param.inputPtr, param.outputPtr, param.hcclBuff));
    tempFastLaunchCtxInter1.threads = interThreads_S1;
    tempFastLaunchCtxInter1.ccuKernelSubmitInfos.assign(ccuKernelSubmitInfos,
                                                         ccuKernelSubmitInfos + ctx->ccuKernelNum[1]);
    ccuKernelSubmitInfos += ctx->ccuKernelNum[1];
    if (ctx->ccuKernelNum[1] > 0) {
        CHK_RET(interTempS1.FastLaunch(param, tempFastLaunchCtxInter1));
    }

    CHK_RET(PostSyncInterThreads(mainThread_, templateMainThreads_, syncNotifyOnMain_));

    CHK_RET(PreSyncInterThreads(mainThread_, templateMainThreads_, syncNotifyOnTemplates_));

    CHK_RET(SetTempFastLaunchAddr(tempFastLaunchCtxInter0, param.outputPtr, param.outputPtr, param.hcclBuff));
    tempFastLaunchCtxInter0.threads = interThreads_S2;
    tempFastLaunchCtxInter0.ccuKernelSubmitInfos.assign(ccuKernelSubmitInfos,
                                                         ccuKernelSubmitInfos + ctx->ccuKernelNum[2]);
    ccuKernelSubmitInfos += ctx->ccuKernelNum[2];
    if (ctx->ccuKernelNum[2] > 0) {
        CHK_RET(interTempS2.FastLaunch(param, tempFastLaunchCtxInter0));
    }

    CHK_RET(SetTempFastLaunchAddr(tempFastLaunchCtxIntra1, param.outputPtr, param.outputPtr, param.hcclBuff));
    tempFastLaunchCtxIntra1.threads = intraThreads_S2;
    tempFastLaunchCtxIntra1.ccuKernelSubmitInfos.assign(ccuKernelSubmitInfos,
                                                         ccuKernelSubmitInfos + ctx->ccuKernelNum[3]);
    if (ctx->ccuKernelNum[3] > 0) {
        CHK_RET(intraTempS2.FastLaunch(param, tempFastLaunchCtxIntra1));
    }

    CHK_RET(PostSyncInterThreads(mainThread_, templateMainThreads_, syncNotifyOnMain_));

    HCCL_INFO("[InsV2AllGatherParallelOptExecutor][FastLaunch] End.");
    return HCCL_SUCCESS;
}
#endif

REGISTER_EXECUTOR_BY_FOUR_TEMPS(HcclCMDType::HCCL_CMD_ALLGATHER, InsAllGatherParallelMesh1DMeshClosV3Opt,
                                 InsV2AllGatherParallelOptExecutor, TopoMatchUBX_V2,
                                 InsTempAllGatherMesh1D, InsTempAllGatherMeshClosV3,
                                 InsTempAllGatherMesh1DV2, InsTempAllGatherMeshClosV3);

REGISTER_EXECUTOR_BY_FOUR_TEMPS(HcclCMDType::HCCL_CMD_ALLGATHER, InsAllGatherParallelMesh1DMeshClosV3OptMultiJetty,
                                 InsV2AllGatherParallelOptExecutor, TopoMatchUBX_V2,
                                 InsTempAllGatherMesh1D, InsTempAllGatherMeshClosV3,
                                 InsTempAllGatherMesh1DV2, InsTempAllGatherMeshClosV3);

}  // namespace ops_hccl
