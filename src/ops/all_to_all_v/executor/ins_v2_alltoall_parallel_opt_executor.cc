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
#include <cmath>
#include <cstdlib>
#include <cstring>
#include "hcomm_primitives.h"
#include "alg_data_trans_wrapper.h"
#include "ins_temp_alltoall_mesh_2d_v3.h"
#include "ins_temp_alltoall_mesh_clos_v3.h"

#include "topo_match_ubx_v2.h"

namespace ops_hccl {

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

    // v2.0 Fix 4: transactional CalcRes — zero-initialize before sub-calls
    resourceRequest = AlgResourceRequest{};

    std::vector<std::vector<u32>> intraHierarchyInfo;
    std::vector<std::vector<u32>> interHierarchyInfo;

    HCCL_INFO("[AllToAll_V3_DEBUG][CalcRes] algHierarchyInfo.infos.size()=%zu level0Topo=%d level0PcieMix=%d userRank=%u",
              algHierarchyInfo.infos.size(), static_cast<int>(topoInfo->level0Topo),
              static_cast<int>(topoInfo->level0PcieMix), topoInfo->userRank);

    for (size_t i = 0; i < algHierarchyInfo.infos.size(); i++) {
        HCCL_INFO("[AllToAll_V3_DEBUG][CalcRes] infos[%zu].size()=%zu", i, algHierarchyInfo.infos[i].size());
        for (size_t j = 0; j < algHierarchyInfo.infos[i].size(); j++) {
            HCCL_INFO("[AllToAll_V3_DEBUG][CalcRes] infos[%zu][%zu].size()=%zu", i, j,
                      algHierarchyInfo.infos[i][j].size());
        }
    }

    if (topoInfo->level0Topo == Level0Shape::MESH_1D_CLOS && !topoInfo->level0PcieMix) {
        HCCL_WARNING("[ALLTOALL_V3_DEBUG][CalcRes] ClosMesh2D branch: level0Topo=MESH_1D_CLOS level0PcieMix=0 "
                  "infos.size=%zu infos[0].size=%zu",
                  algHierarchyInfo.infos.size(), algHierarchyInfo.infos[0].size());
        CHK_PRT_RET(algHierarchyInfo.infos.size() < 2 || algHierarchyInfo.infos[0].size() < 2 ||
                    algHierarchyInfo.infos[0][0].empty() || algHierarchyInfo.infos[0][1].empty(),
                    HCCL_ERROR("[InsV2AlltoAllParallelOptExecutor][CalcRes] ClosMesh2D: invalid topoInfo. "
                               "infos.size=%zu infos[0].size=%zu",
                               algHierarchyInfo.infos.size(), algHierarchyInfo.infos[0].size()),
                    HcclResult::HCCL_E_INTERNAL);
        intraHierarchyInfo = {algHierarchyInfo.infos[0][0]};
        std::vector<u32> closRanks;
        closRanks.push_back(topoInfo->userRank);  // Ensure self is included in inter-hierarchy group for correctness
        u32 meshSize = algHierarchyInfo.infos[0][0].size();
        for (auto rank : algHierarchyInfo.infos[0][1]) {
            if (rank / meshSize != topoInfo->userRank / meshSize) {
                closRanks.push_back(rank);
            }
        }
        interHierarchyInfo = {closRanks};
        HCCL_WARNING("[ALLTOALL_V3_DEBUG][CalcRes] ClosMesh2D: userRank=%u meshSize=%u intra[0]=%zu closRanks=%zu",
                  topoInfo->userRank, meshSize, intraHierarchyInfo[0].size(), closRanks.size());
    } else {
        HCCL_WARNING("[ALLTOALL_V3_DEBUG][CalcRes] Direct branch: level0Topo=%d level0PcieMix=%d "
                  "infos.size=%zu",
                  static_cast<int>(topoInfo->level0Topo),
                  static_cast<int>(topoInfo->level0PcieMix),
                  algHierarchyInfo.infos.size());
        constexpr u32 TOPO_NUM = 2;
        CHK_PRT_RET(algHierarchyInfo.infos.size() < TOPO_NUM || algHierarchyInfo.infos[0].empty() ||
                    algHierarchyInfo.infos[1].empty(),
                    HCCL_ERROR("[InsV2AlltoAllParallelOptExecutor][CalcRes] Direct path: invalid topoInfo. "
                               "infos.size=%zu infos[0].empty=%d infos[1].empty=%d",
                               algHierarchyInfo.infos.size(),
                               algHierarchyInfo.infos.size() > 0 ? algHierarchyInfo.infos[0].empty() : 1,
                               algHierarchyInfo.infos.size() > 1 ? algHierarchyInfo.infos[1].empty() : 1),
                    HcclResult::HCCL_E_INTERNAL);
        intraHierarchyInfo = algHierarchyInfo.infos[0];
        interHierarchyInfo = algHierarchyInfo.infos[1];
        HCCL_WARNING("[ALLTOALL_V3_DEBUG][CalcRes] Direct: infos[0].size=%zu infos[1].size=%zu",
                  algHierarchyInfo.infos[0].size(), algHierarchyInfo.infos[1].size());
    }

    InsAlgTemplate0 intraTempAlg(param, topoInfo->userRank, intraHierarchyInfo);
    InsAlgTemplate1 interTempAlg(param, topoInfo->userRank, interHierarchyInfo);
    HCCL_INFO("[AllToAll_V3_DEBUG][CalcRes] intra=%s inter=%s",
              intraTempAlg.Describe().c_str(), interTempAlg.Describe().c_str());

    // v2.0 Fix 4: separate local requests; merge only if both succeed
    AlgResourceRequest intraTempRequest;
    AlgResourceRequest interTempRequest;
    CHK_RET(intraTempAlg.CalcRes(comm, param, topoInfo, intraTempRequest));
    CHK_RET(interTempAlg.CalcRes(comm, param, topoInfo, interTempRequest));

    HCCL_INFO("[AllToAll_V3_DEBUG][CalcRes] intra: channels[0]=%zu slaveThreads=%u notifyMain=%u notifyVec=%zu",
              intraTempRequest.channels.empty() ? 0 : intraTempRequest.channels[0].size(),
              intraTempRequest.slaveThreadNum, intraTempRequest.notifyNumOnMainThread,
              intraTempRequest.notifyNumPerThread.size());
    HCCL_INFO("[AllToAll_V3_DEBUG][CalcRes] inter: channels[0]=%zu slaveThreads=%u notifyMain=%u notifyVec=%zu",
              interTempRequest.channels.empty() ? 0 : interTempRequest.channels[0].size(),
              interTempRequest.slaveThreadNum, interTempRequest.notifyNumOnMainThread,
              interTempRequest.notifyNumPerThread.size());

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

   
    u32 expectedNotifySize = intraTempRequest.notifyNumPerThread.size() +
                             interTempRequest.notifyNumPerThread.size() + 2;  // +2 for two template main threads

    CHK_PRT_RET(resourceRequest.notifyNumPerThread.size() != expectedNotifySize,
                HCCL_ERROR("[InsV2AlltoAllParallelOptExecutor][CalcRes] notifyNumPerThread.size()[%zu] != expected[%u]",
                           resourceRequest.notifyNumPerThread.size(), expectedNotifySize),
                HcclResult::HCCL_E_INTERNAL);

    if (param.engine != COMM_ENGINE_CCU) {
        CHK_PRT_RET(intraTempRequest.channels.empty() || interTempRequest.channels.empty(),
                    HCCL_ERROR("[InsV2AlltoAllParallelOptExecutor][CalcRes] intraTemplate or interTemplate has empty channels."),
                    HcclResult::HCCL_E_INTERNAL);
        resourceRequest.channels.emplace_back(intraTempRequest.channels[0]);
        resourceRequest.channels.emplace_back(interTempRequest.channels[0]);
    } else {
        HCCL_INFO("[InsV2AlltoAllParallelOptExecutor][CalcRes] intraTemplate has [%d] kernels.",
                  intraTempRequest.ccuKernelNum[0]);
        resourceRequest.ccuKernelInfos.insert(resourceRequest.ccuKernelInfos.end(),
                                              intraTempRequest.ccuKernelInfos.begin(),
                                              intraTempRequest.ccuKernelInfos.end());
        resourceRequest.ccuKernelNum.emplace_back(intraTempRequest.ccuKernelNum[0]);
        HCCL_INFO("[InsV2AlltoAllParallelOptExecutor][CalcRes] interTemplate has [%d] kernels.",
                  interTempRequest.ccuKernelNum[0]);
        resourceRequest.ccuKernelInfos.insert(resourceRequest.ccuKernelInfos.end(),
                                              interTempRequest.ccuKernelInfos.begin(),
                                              interTempRequest.ccuKernelInfos.end());
        resourceRequest.ccuKernelNum.emplace_back(interTempRequest.ccuKernelNum[0]);
    }

    HCCL_INFO("[AllToAll_V3_DEBUG][InsV2AlltoAllParallelOptExecutor][CalcRes] myRank[%u], notifyNumOnMainThread[%u], slaveThreadNum[%u], "
               "channels[%u]",
               myRank_, resourceRequest.notifyNumOnMainThread, resourceRequest.slaveThreadNum,
               resourceRequest.channels.size());
    for (auto i = 0; i < resourceRequest.notifyNumPerThread.size(); i++) {
        HCCL_INFO("[AllToAll_V3_DEBUG][InsV2AlltoAllParallelOptExecutor][CalcRes] myRank[%u], notifyNumPerThread[%u]=[%u]", myRank_, i,
                   resourceRequest.notifyNumPerThread[i]);
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
HcclResult InsV2AlltoAllParallelOptExecutor<AlgTopoMatch, InsAlgTemplate0, InsAlgTemplate1>::Orchestrate(
    const OpParam &param, const AlgResourceCtxSerializable &resCtx)
{
    HCCL_INFO("[AllToAll_V3_DEBUG][InsV2AlltoAllParallelOptExecutor][Orchestrate] Orchestrate Start");
    maxTmpMemSize_ = resCtx.cclMem.size;
    myRank_ = resCtx.topoInfo.userRank;
    threads_ = resCtx.threads;

    if (param.engine != CommEngine::COMM_ENGINE_AIV && param.engine != CommEngine::COMM_ENGINE_CCU) {
        CHK_RET(RestoreChannelMap(resCtx, remoteRankToChannelInfo_));
        intraLinkMap_ = remoteRankToChannelInfo_[0];
        if (remoteRankToChannelInfo_.size() >= 2) {
            interLinkMap_ = remoteRankToChannelInfo_[1];
        }

        HCCL_INFO("[AllToAll_V3_DEBUG][Orchestrate] intraLinkMap_ size=%zu interLinkMap_ size=%zu",
                  intraLinkMap_.size(), interLinkMap_.size());
        for (auto &kv : intraLinkMap_) {
            HCCL_WARNING("[AllToAll_V3_DEBUG][Orchestrate] intraLinkMap_ rank=%u channels=%zu", kv.first, kv.second.size());
        }
        HCCL_WARNING("[ALLTOALL_V3_DEBUG][Orchestrate] Entering interLinkMap_ log loop. interLinkMap_.size()=%zu",
                     interLinkMap_.size());
        for (auto &kv : interLinkMap_) {
            HCCL_WARNING("[AllToAll_V3_DEBUG][Orchestrate] interLinkMap_ rank=%u channels=%zu", kv.first, kv.second.size());
        }
        HCCL_WARNING("[ALLTOALL_V3_DEBUG][Orchestrate] interLinkMap_ iteration complete. About to enter topology detection.");
    }
    // v1.4 Fix: OpParam uses a union — must read the correct variant based on opType.
    // AlltoAllV (and AlltoAll converted to AlltoAllV) populates all2AllVDataDes, not DataDes.
    if (param.opType == HcclCMDType::HCCL_CMD_ALLTOALL) {
        dataType_ = param.all2AllVDataDes.sendType;
        // AlltoAllV has per-peer counts; compute total dataCount by summing sendCounts
        // For AlltoAll (uniform), sendCounts[i] == sendCount for all i
        u64* sendCounts = reinterpret_cast<u64*>(param.all2AllVDataDes.sendCounts);
        if (!sendCounts) {
            HCCL_ERROR("[Orchestrate] FATAL: all2AllVDataDes.sendCounts is NULL");
            return HcclResult::HCCL_E_INTERNAL;
        }
        u64 totalRanks = resCtx.topoInfo.userRankSize;
        u64 totalCount = 0;
        for (u64 i = 0; i < totalRanks; i++) {
            totalCount += sendCounts[i];
        }
        dataCount_ = totalCount;
        if (static_cast<int>(dataType_) < 0 || dataType_ >= HCCL_DATA_TYPE_RESERVED) {
            HCCL_ERROR("[Orchestrate] FATAL: invalid dataType=%d from all2AllVDataDes.sendType",
                       static_cast<int>(dataType_));
            return HcclResult::HCCL_E_INTERNAL;
        }
    } else {
        dataCount_ = param.DataDes.count;
        dataType_ = param.DataDes.dataType;
        if (static_cast<int>(dataType_) < 0 || dataType_ >= HCCL_DATA_TYPE_RESERVED) {
            HCCL_ERROR("[Orchestrate] FATAL: invalid dataType=%d from DataDes.dataType",
                       static_cast<int>(dataType_));
            return HcclResult::HCCL_E_INTERNAL;
        }
    }
    dataTypeSize_ = DATATYPE_SIZE_TABLE[dataType_];
    dataSize_ = dataCount_ * dataTypeSize_;
    HCCL_WARNING("[AllToAll_V3_DEBUG][L1] dataCount=%llu dataType=%d dataTypeSize=%u dataSize=%llu",
        dataCount_, static_cast<int>(dataType_), dataTypeSize_, dataSize_);

    HCCL_INFO("[AllToAll_V3_DEBUG][InsV2AlltoAllParallelOptExecutor][Orchestrate] myRank=%u dataCount=%llu dataSize=%llu dataTypeSize=%u",
              myRank_, dataCount_, dataSize_, dataTypeSize_);
    HCCL_WARNING("[ALLTOALL_V3_DEBUG][Orchestrate] Topology: level0Topo=%d, level0PcieMix=%d, infos.size=%zu",
                 static_cast<int>(resCtx.topoInfo.level0Topo),
                 static_cast<int>(resCtx.topoInfo.level0PcieMix),
                 resCtx.algHierarchyInfo.infos.size());
    if (resCtx.algHierarchyInfo.infos.size() > 0) {
        HCCL_WARNING("[ALLTOALL_V3_DEBUG][Orchestrate] infos[0].size=%zu",
                     resCtx.algHierarchyInfo.infos[0].size());
        if (resCtx.algHierarchyInfo.infos[0].size() >= 1) {
            HCCL_WARNING("[ALLTOALL_V3_DEBUG][Orchestrate] infos[0][0].size=%zu",
                         resCtx.algHierarchyInfo.infos[0][0].size());
        }
        if (resCtx.algHierarchyInfo.infos[0].size() >= 2) {
            HCCL_WARNING("[ALLTOALL_V3_DEBUG][Orchestrate] infos[0][1].size=%zu",
                         resCtx.algHierarchyInfo.infos[0][1].size());
        }
    }
    if (resCtx.algHierarchyInfo.infos.size() > 1) {
        HCCL_WARNING("[ALLTOALL_V3_DEBUG][Orchestrate] infos[1].size=%zu",
                     resCtx.algHierarchyInfo.infos[1].size());
    }
    for (size_t i = 0; i < resCtx.algHierarchyInfo.infos.size(); i++) {
        HCCL_INFO("[Orchestrate] infos[%zu].size()=%zu", i,
                  resCtx.algHierarchyInfo.infos[i].size());
    }

    HCCL_WARNING("[AllToAll_V3_DEBUG][L2] Entering topology detection. level0Topo=%d level0PcieMix=%d "
        "infos.size=%zu infos[0].size=%zu",
        static_cast<int>(resCtx.topoInfo.level0Topo),
        static_cast<int>(resCtx.topoInfo.level0PcieMix),
        resCtx.algHierarchyInfo.infos.size(),
        resCtx.algHierarchyInfo.infos.empty() ? 0 : resCtx.algHierarchyInfo.infos[0].size());

    // DEFENSIVE: validate infos before accessing
    if (resCtx.algHierarchyInfo.infos.empty() || resCtx.algHierarchyInfo.infos[0].empty()) {
        HCCL_ERROR("[Orchestrate] FATAL: algHierarchyInfo.infos is empty or infos[0] is empty");
        return HcclResult::HCCL_E_INTERNAL;
    }

    // ClosMesh2D: infos[0] has 2+ groups (one per pod). UBX: 1 group.
    if (resCtx.topoInfo.level0Topo == Level0Shape::MESH_1D_CLOS && !resCtx.topoInfo.level0PcieMix) {
        if (resCtx.algHierarchyInfo.infos[0].size() < 2) {
            HCCL_ERROR("[Orchestrate] FATAL: ClosMesh2D path but infos[0].size=%zu < 2",
                resCtx.algHierarchyInfo.infos[0].size());
            return HcclResult::HCCL_E_INTERNAL;
        }
        intraHierarchyInfo_ = {resCtx.algHierarchyInfo.infos[0][0]};
        std::vector<u32> closRanks;
        closRanks.push_back(resCtx.topoInfo.userRank);  // Ensure self is included in inter-hierarchy group for correctness
        u32 meshSize = resCtx.algHierarchyInfo.infos[0][0].size();
        for (auto rank : resCtx.algHierarchyInfo.infos[0][1]) {
            if (rank / meshSize != resCtx.topoInfo.userRank / meshSize) {
                closRanks.push_back(rank);
            }
        }
        interHierarchyInfo_ = {closRanks};
        HCCL_WARNING("[ALLTOALL_V3_DEBUG][Orchestrate] ClosMesh2D: intra[%zu] inter[%zu] meshSize=%u",
                  intraHierarchyInfo_[0].size(), interHierarchyInfo_[0].size(), meshSize);
        // Rebuild interLinkMap_ to match interHierarchyInfo_: merge channels for all ranks in inter group
        if (!interLinkMap_.empty()) {
            HCCL_WARNING("[ALLTOALL_V3_DEBUG][Orchestrate] interLinkMap_ before rebuild: size=%zu keys=",
                      interLinkMap_.size());
            for (auto &kv : interLinkMap_) {
                HCCL_WARNING("[ALLTOALL_V3_DEBUG][Orchestrate] interLinkMap_ before: rank=%u channels=%zu",
                          kv.first, kv.second.size());
            }
            std::map<u32, std::vector<ChannelInfo>> mergedInterMap;
            u32 mergedCount = 0;
            u32 missingCount = 0;
            for (auto rank : interHierarchyInfo_[0]) {
                if (intraLinkMap_.count(rank)) {
                    mergedInterMap[rank] = intraLinkMap_[rank];
                    mergedCount++;
                    HCCL_WARNING("[ALLTOALL_V3_DEBUG][Orchestrate] rebuild inter: rank=%u from intraLinkMap_", rank);
                } else if (interLinkMap_.count(rank)) {
                    mergedInterMap[rank] = interLinkMap_[rank];
                    mergedCount++;
                    HCCL_WARNING("[ALLTOALL_V3_DEBUG][Orchestrate] rebuild inter: rank=%u from interLinkMap_", rank);
                } else {
                    missingCount++;
                    HCCL_ERROR("[ALLTOALL_V3_DEBUG][Orchestrate] rebuild inter: rank=%u MISSING from both maps!", rank);
                }
            }
            interLinkMap_ = mergedInterMap;
            HCCL_WARNING("[ALLTOALL_V3_DEBUG][Orchestrate] interLinkMap_ after rebuild: size=%zu merged=%u missing=%u",
                      interLinkMap_.size(), mergedCount, missingCount);
            for (auto &kv : interLinkMap_) {
                HCCL_WARNING("[ALLTOALL_V3_DEBUG][Orchestrate] interLinkMap_ after: rank=%u channels=%zu",
                          kv.first, kv.second.size());
            }
        }
    } else {
        HCCL_WARNING("[ALLTOALL_V3_DEBUG][Orchestrate] Direct: level0Topo=%d level0PcieMix=%d",
                  static_cast<int>(resCtx.topoInfo.level0Topo),
                  static_cast<int>(resCtx.topoInfo.level0PcieMix));
        intraHierarchyInfo_ = resCtx.algHierarchyInfo.infos[0];
        if (resCtx.algHierarchyInfo.infos.size() < 2) {
            HCCL_ERROR("[Orchestrate] FATAL: Direct path but infos.size=%zu < 2",
                resCtx.algHierarchyInfo.infos.size());
            return HcclResult::HCCL_E_INTERNAL;
        }
        interHierarchyInfo_ = resCtx.algHierarchyInfo.infos[1];
    }
    rankSizeLevel0_ = GetRankSize(intraHierarchyInfo_);
    rankSizeLevel1_ = GetRankSize(interHierarchyInfo_);

    if (rankSizeLevel0_ == 0) {
        HCCL_ERROR("[InsV2AlltoAllParallelOptExecutor][Orchestrate] FATAL: rankSizeLevel0_ is 0. "
                   "intraHierarchyInfo_.size=%zu rankIdxLevel0_=%llu",
                   intraHierarchyInfo_.size(), rankIdxLevel0_);
        return HcclResult::HCCL_E_INTERNAL;
    }

    if (rankSizeLevel1_ == 0) {
        HCCL_ERROR("[InsV2AlltoAllParallelOptExecutor][Orchestrate] FATAL: rankSizeLevel1_ is 0. "
                   "interHierarchyInfo_.size=%zu rankIdxLevel1_=%llu",
                   interHierarchyInfo_.size(), rankIdxLevel1_);
        return HcclResult::HCCL_E_INTERNAL;
    }

    InsAlgTemplate0 intraTempAlg(param, resCtx.topoInfo.userRank, intraHierarchyInfo_);
    InsAlgTemplate1 interTempAlg(param, resCtx.topoInfo.userRank, interHierarchyInfo_);

    intraTempAlg.SetMeshDimensions(rankSizeLevel0_ + rankSizeLevel1_ - 1, myRank_, rankSizeLevel0_, rankSizeLevel1_);
    interTempAlg.SetMeshDimensions(rankSizeLevel0_ + rankSizeLevel1_ - 1, myRank_, rankSizeLevel0_, rankSizeLevel1_);

    if (param.engine == CommEngine::COMM_ENGINE_AICPU_TS ||
        param.engine == CommEngine::COMM_ENGINE_AIV) {
        interTempAlg.SetchannelsPerRank(interLinkMap_);
    }

    bool hasInterComm = !interLinkMap_.empty();
    HCCL_INFO("[InsV2AlltoAllParallelOptExecutor][Orchestrate] hasInterComm=%d", hasInterComm);

    HCCL_WARNING("[ALLTOALL_V3_DEBUG][Orchestrate] L6: calling PrepareResForTemplate. hasInterComm=%d", hasInterComm);

    PrepareResForTemplate(intraTempAlg, interTempAlg);

    HCCL_WARNING("[ALLTOALL_V3_DEBUG][Orchestrate] L7: entering OrchestrateLoop");

    HcclResult ret = OrchestrateLoop(param, resCtx, intraTempAlg, interTempAlg, hasInterComm);
    CHK_PRT_RET(
        ret != HCCL_SUCCESS,
        HCCL_ERROR("[InsV2AlltoAllParallelOptExecutor][Orchestrate]errNo[0x%016llx] AlltoAll executor kernel run failed",
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
void InsV2AlltoAllParallelOptExecutor<AlgTopoMatch, InsAlgTemplate0, InsAlgTemplate1>::GetParallelDataSplit(
    std::vector<float> &splitDataSize) const
{
    // double splitData = multipleDimensionSplitRatio_;
    double splitData = 0.5;
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
    HCCL_INFO("[InsV2AlltoAllParallelOptExecutor] splitDataSize is %f, %f", splitDataSize[0], splitDataSize[1]);
    return;
}

template <typename AlgTopoMatch, typename InsAlgTemplate0, typename InsAlgTemplate1>
HcclResult InsV2AlltoAllParallelOptExecutor<AlgTopoMatch, InsAlgTemplate0, InsAlgTemplate1>::OrchestrateLoop(
    const OpParam &param, const AlgResourceCtxSerializable &resCtx, InsAlgTemplate0 &tempAlgIntra,
    InsAlgTemplate1 &tempAlgInter, bool hasInterComm)
{
    HCCL_INFO("[InsV2AlltoAllParallelOptExecutor][OrchestrateLoop] Entry. maxTmpMem=%llu dataCount=%llu dataSize=%llu "
              "dataTypeSize=%u hasInterComm=%d",
              maxTmpMemSize_, dataCount_, dataSize_, dataTypeSize_, hasInterComm);

    TemplateResource interTempAlgRes;
    interTempAlgRes.channels = interLinkMap_;
    interTempAlgRes.threads = interThreads_;
    interTempAlgRes.aivCommInfoPtr = resCtx.aivCommInfoPtr;

    TemplateResource intraTempAlgRes;
    intraTempAlgRes.channels = intraLinkMap_;
    intraTempAlgRes.threads = intraThreads_;
    intraTempAlgRes.aivCommInfoPtr = resCtx.aivCommInfoPtr;

    // Stage 0 
    TemplateDataParams tempAlgParams;
 
    intraTempAlgRes.channels = intraLinkMap_;
    interTempAlgRes.channels = interLinkMap_;

    // 总 rank 数
    u64 totalRankCount = rankSizeLevel0_ + rankSizeLevel1_ - 1;
    u64 perPeerInputChunkSize = dataSize_ / totalRankCount;


    {
        tempAlgParams.enableRemoteMemAccess = param.opMode == OpMode::OFFLOAD;

        tempAlgParams.buffInfo.inputPtr = param.inputPtr;
        tempAlgParams.buffInfo.inBuffType = BufferType::INPUT;
        tempAlgParams.buffInfo.inputSize = param.inputSize;
        tempAlgParams.buffInfo.inBuffBaseOff = 0;
        tempAlgParams.inputSliceStride = perPeerInputChunkSize;

        tempAlgParams.buffInfo.hcclBuff = resCtx.cclMem;
        tempAlgParams.buffInfo.hcclBuffType = BufferType::HCCL_BUFFER;
        tempAlgParams.buffInfo.hcclBuffSize = resCtx.cclMem.size;
        tempAlgParams.buffInfo.hcclBuffBaseOff = 0;

        // 不需要 copy out
        tempAlgParams.buffInfo.outputPtr = param.outputPtr;
        tempAlgParams.buffInfo.outBuffType = BufferType::OUTPUT;
        tempAlgParams.buffInfo.outputSize = param.outputSize * dataTypeSize_;
        tempAlgParams.buffInfo.outBuffBaseOff = 0;
        tempAlgParams.outputSliceStride = perPeerInputChunkSize;

        tempAlgParams.sliceSize = dataSize_;
        tempAlgParams.count = dataCount_;
    }


    // 开始 Stage 0，Presync，确保所有线程和模板同步准备好进行第一阶段的计算。
    CHK_RET(PreSyncInterThreads(mainThread_, templateMainThreads_, syncNotifyOnTemplates_));
    
    CHK_RET(tempAlgIntra.KernelRun(param, tempAlgParams, intraTempAlgRes));
    CHK_RET(tempAlgInter.KernelRun(param, tempAlgParams, interTempAlgRes));

    // 结束 Stage 0，Postync，确保所有线程和模板同步结束第一阶段的计算。
    CHK_RET(PostSyncInterThreads(mainThread_, templateMainThreads_, syncNotifyOnMain_));

    
    HCCL_INFO("[InsV2AlltoAllParallelOptExecutor][OrchestrateLoop] End.");
    return HcclResult::HCCL_SUCCESS;
}

// UBX topology (8-card boards) — Mesh intra, Clos inter
REGISTER_EXECUTOR_BY_TWO_TEMPS(
    HcclCMDType::HCCL_CMD_ALLTOALL,
    InsAlltoAllParallelMesh2DClosV3,
    InsV2AlltoAllParallelOptExecutor,
    TopoMatchUBX_V2,
    InsTempAlltoAllMesh2DV3,
    InsTempAlltoAllMeshClosV3);

}  // namespace ops_hccl
