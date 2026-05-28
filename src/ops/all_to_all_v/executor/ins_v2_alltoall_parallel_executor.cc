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
#include <cstdlib>
#include <cstring>
#include "hcomm_primitives.h"
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

    HCCL_INFO("[CalcRes] algHierarchyInfo.infos.size()=%zu level0Topo=%d level0PcieMix=%d userRank=%u",
              algHierarchyInfo.infos.size(), static_cast<int>(topoInfo->level0Topo),
              static_cast<int>(topoInfo->level0PcieMix), topoInfo->userRank);
    for (size_t i = 0; i < algHierarchyInfo.infos.size(); i++) {
        HCCL_INFO("[CalcRes] infos[%zu].size()=%zu", i, algHierarchyInfo.infos[i].size());
        for (size_t j = 0; j < algHierarchyInfo.infos[i].size(); j++) {
            HCCL_INFO("[CalcRes] infos[%zu][%zu].size()=%zu", i, j,
                      algHierarchyInfo.infos[i][j].size());
        }
    }

    if (topoInfo->level0Topo == Level0Shape::MESH_1D_CLOS && !topoInfo->level0PcieMix) {
        HCCL_WARNING("[ALLTOALL_V2_DEBUG][CalcRes] ClosMesh2D branch: level0Topo=MESH_1D_CLOS level0PcieMix=0 "
                  "infos.size=%zu infos[0].size=%zu",
                  algHierarchyInfo.infos.size(), algHierarchyInfo.infos[0].size());
        CHK_PRT_RET(algHierarchyInfo.infos.size() < 2 || algHierarchyInfo.infos[0].size() < 2 ||
                    algHierarchyInfo.infos[0][0].empty() || algHierarchyInfo.infos[0][1].empty(),
                    HCCL_ERROR("[InsV2AlltoAllParallelExecutor][CalcRes] ClosMesh2D: invalid topoInfo. "
                               "infos.size=%zu infos[0].size=%zu",
                               algHierarchyInfo.infos.size(), algHierarchyInfo.infos[0].size()),
                    HcclResult::HCCL_E_INTERNAL);
        intraHierarchyInfo = {algHierarchyInfo.infos[0][0]};
        std::vector<u32> closRanks;
        u32 meshSize = algHierarchyInfo.infos[0][0].size();
        for (auto rank : algHierarchyInfo.infos[0][1]) {
            if (rank % meshSize == topoInfo->userRank % meshSize) {
                closRanks.push_back(rank);
            }
        }
        interHierarchyInfo = {closRanks};
        HCCL_WARNING("[ALLTOALL_V2_DEBUG][CalcRes] ClosMesh2D: userRank=%u meshSize=%u intra[0]=%zu closRanks=%zu",
                  topoInfo->userRank, meshSize, intraHierarchyInfo[0].size(), closRanks.size());
    } else {
        HCCL_WARNING("[ALLTOALL_V2_DEBUG][CalcRes] Direct branch: level0Topo=%d level0PcieMix=%d "
                  "infos.size=%zu",
                  static_cast<int>(topoInfo->level0Topo),
                  static_cast<int>(topoInfo->level0PcieMix),
                  algHierarchyInfo.infos.size());
        constexpr u32 TOPO_NUM = 2;
        CHK_PRT_RET(algHierarchyInfo.infos.size() < TOPO_NUM || algHierarchyInfo.infos[0].empty() ||
                    algHierarchyInfo.infos[1].empty(),
                    HCCL_ERROR("[InsV2AlltoAllParallelExecutor][CalcRes] Direct path: invalid topoInfo. "
                               "infos.size=%zu infos[0].empty=%d infos[1].empty=%d",
                               algHierarchyInfo.infos.size(),
                               algHierarchyInfo.infos.size() > 0 ? algHierarchyInfo.infos[0].empty() : 1,
                               algHierarchyInfo.infos.size() > 1 ? algHierarchyInfo.infos[1].empty() : 1),
                    HcclResult::HCCL_E_INTERNAL);
        intraHierarchyInfo = algHierarchyInfo.infos[0];
        interHierarchyInfo = algHierarchyInfo.infos[1];
        HCCL_WARNING("[ALLTOALL_V2_DEBUG][CalcRes] Direct: infos[0].size=%zu infos[1].size=%zu",
                  algHierarchyInfo.infos[0].size(), algHierarchyInfo.infos[1].size());
    }

    InsAlgTemplate0 intraTempAlg(param, topoInfo->userRank, intraHierarchyInfo);
    InsAlgTemplate1 interTempAlg(param, topoInfo->userRank, interHierarchyInfo);
    HCCL_INFO("[CalcRes] intra=%s inter=%s",
              intraTempAlg.Describe().c_str(), interTempAlg.Describe().c_str());

    // v2.0 Fix 4: separate local requests; merge only if both succeed
    AlgResourceRequest intraTempRequest;
    AlgResourceRequest interTempRequest;
    CHK_RET(intraTempAlg.CalcRes(comm, param, topoInfo, intraTempRequest));
    CHK_RET(interTempAlg.CalcRes(comm, param, topoInfo, interTempRequest));

    HCCL_INFO("[CalcRes] intra: channels[0]=%zu slaveThreads=%u notifyMain=%u notifyVec=%zu",
              intraTempRequest.channels.empty() ? 0 : intraTempRequest.channels[0].size(),
              intraTempRequest.slaveThreadNum, intraTempRequest.notifyNumOnMainThread,
              intraTempRequest.notifyNumPerThread.size());
    HCCL_INFO("[CalcRes] inter: channels[0]=%zu slaveThreads=%u notifyMain=%u notifyVec=%zu",
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

    HCCL_INFO("[InsV2AlltoAllParallelExecutor][CalcRes] myRank[%u], notifyNumOnMainThread[%u], slaveThreadNum[%u], "
               "channels[%u]",
               myRank_, resourceRequest.notifyNumOnMainThread, resourceRequest.slaveThreadNum,
               resourceRequest.channels.size());
    for (auto i = 0; i < resourceRequest.notifyNumPerThread.size(); i++) {
        HCCL_INFO("[InsV2AlltoAllParallelExecutor][CalcRes] myRank[%u], notifyNumPerThread[%u]=[%u]", myRank_, i,
                   resourceRequest.notifyNumPerThread[i]);
    }

    return HCCL_SUCCESS;
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
        if (remoteRankToChannelInfo_.size() >= 2) {
            interLinkMap_ = remoteRankToChannelInfo_[1];
        }

        HCCL_INFO("[Orchestrate] intraLinkMap_ size=%zu interLinkMap_ size=%zu",
                  intraLinkMap_.size(), interLinkMap_.size());
        for (auto &kv : intraLinkMap_) {
            HCCL_WARNING("[Orchestrate] intraLinkMap_ rank=%u channels=%zu", kv.first, kv.second.size());
        }
        HCCL_WARNING("[ALLTOALL_V2_DEBUG][Orchestrate] Entering interLinkMap_ log loop. interLinkMap_.size()=%zu",
                     interLinkMap_.size());
        for (auto &kv : interLinkMap_) {
            HCCL_WARNING("[Orchestrate] interLinkMap_ rank=%u channels=%zu", kv.first, kv.second.size());
        }
        HCCL_WARNING("[ALLTOALL_V2_DEBUG][Orchestrate] interLinkMap_ iteration complete. About to enter topology detection.");
    }
    // v1.4 Fix: OpParam uses a union — must read the correct variant based on opType.
    // AlltoAllV (and AlltoAll converted to AlltoAllV) populates all2AllVDataDes, not DataDes.
    if (param.opType == HcclCMDType::HCCL_CMD_ALLTOALL ||
        param.opType == HcclCMDType::HCCL_CMD_ALLTOALLV) {
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
    HCCL_WARNING("[ALLTOALL_V2_CRASH_DIAG][L1] dataCount=%llu dataType=%d dataTypeSize=%u dataSize=%llu",
        dataCount_, static_cast<int>(dataType_), dataTypeSize_, dataSize_);

    HCCL_INFO("[InsV2AlltoAllParallelExecutor][Orchestrate] myRank=%u dataCount=%llu dataSize=%llu dataTypeSize=%u",
              myRank_, dataCount_, dataSize_, dataTypeSize_);
    HCCL_WARNING("[ALLTOALL_V2_DEBUG][Orchestrate] Topology: level0Topo=%d, level0PcieMix=%d, infos.size=%zu",
                 static_cast<int>(resCtx.topoInfo.level0Topo),
                 static_cast<int>(resCtx.topoInfo.level0PcieMix),
                 resCtx.algHierarchyInfo.infos.size());
    if (resCtx.algHierarchyInfo.infos.size() > 0) {
        HCCL_WARNING("[ALLTOALL_V2_DEBUG][Orchestrate] infos[0].size=%zu",
                     resCtx.algHierarchyInfo.infos[0].size());
        if (resCtx.algHierarchyInfo.infos[0].size() >= 1) {
            HCCL_WARNING("[ALLTOALL_V2_DEBUG][Orchestrate] infos[0][0].size=%zu",
                         resCtx.algHierarchyInfo.infos[0][0].size());
        }
        if (resCtx.algHierarchyInfo.infos[0].size() >= 2) {
            HCCL_WARNING("[ALLTOALL_V2_DEBUG][Orchestrate] infos[0][1].size=%zu",
                         resCtx.algHierarchyInfo.infos[0][1].size());
        }
    }
    if (resCtx.algHierarchyInfo.infos.size() > 1) {
        HCCL_WARNING("[ALLTOALL_V2_DEBUG][Orchestrate] infos[1].size=%zu",
                     resCtx.algHierarchyInfo.infos[1].size());
    }
    for (size_t i = 0; i < resCtx.algHierarchyInfo.infos.size(); i++) {
        HCCL_INFO("[Orchestrate] infos[%zu].size()=%zu", i,
                  resCtx.algHierarchyInfo.infos[i].size());
    }

    HCCL_WARNING("[ALLTOALL_V2_CRASH_DIAG][L2] Entering topology detection. level0Topo=%d level0PcieMix=%d "
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
        u32 meshSize = resCtx.algHierarchyInfo.infos[0][0].size();
        for (auto rank : resCtx.algHierarchyInfo.infos[0][1]) {
            if (rank % meshSize == resCtx.topoInfo.userRank % meshSize) {
                closRanks.push_back(rank);
            }
        }
        interHierarchyInfo_ = {closRanks};
        HCCL_WARNING("[ALLTOALL_V2_DEBUG][Orchestrate] ClosMesh2D: intra[%zu] inter[%zu] meshSize=%u",
                  intraHierarchyInfo_[0].size(), interHierarchyInfo_[0].size(), meshSize);
    } else {
        HCCL_WARNING("[ALLTOALL_V2_DEBUG][Orchestrate] Direct: level0Topo=%d level0PcieMix=%d",
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
        HCCL_ERROR("[InsV2AlltoAllParallelExecutor][Orchestrate] FATAL: rankSizeLevel0_ is 0. "
                   "intraHierarchyInfo_.size=%zu rankIdxLevel0_=%llu",
                   intraHierarchyInfo_.size(), rankIdxLevel0_);
        return HcclResult::HCCL_E_INTERNAL;
    }

    if (rankSizeLevel1_ == 0) {
        HCCL_ERROR("[InsV2AlltoAllParallelExecutor][Orchestrate] FATAL: rankSizeLevel1_ is 0. "
                   "interHierarchyInfo_.size=%zu rankIdxLevel1_=%llu",
                   interHierarchyInfo_.size(), rankIdxLevel1_);
        return HcclResult::HCCL_E_INTERNAL;
    }

    rankIdxLevel0_ = myRank_ % rankSizeLevel0_;
    rankIdxLevel1_ = myRank_ / rankSizeLevel0_;

    HCCL_WARNING("[ALLTOALL_V2_CRASH_DIAG][L3] Topology detection complete. "
        "intraHierarchyInfo_.size=%zu interHierarchyInfo_.size=%zu "
        "rankSize0=%llu rankSize1=%llu",
        intraHierarchyInfo_.size(), interHierarchyInfo_.size(),
        rankSizeLevel0_, rankSizeLevel1_);

    HCCL_WARNING("[ALLTOALL_V2_DEBUG][Orchestrate] hierarchy: intra.size=%zu inter.size=%zu "
              "rankSize0=%llu rankSize1=%llu rankIdx0=%llu rankIdx1=%llu",
              intraHierarchyInfo_.size(), interHierarchyInfo_.size(),
              rankSizeLevel0_, rankSizeLevel1_, rankIdxLevel0_, rankIdxLevel1_);

    HCCL_WARNING("[ALLTOALL_V2_DEBUG][Orchestrate] L4: rankSize0=%llu rankSize1=%llu rankIdx0=%llu rankIdx1=%llu",
        rankSizeLevel0_, rankSizeLevel1_, rankIdxLevel0_, rankIdxLevel1_);

    InsAlgTemplate0 intraTempAlg(param, resCtx.topoInfo.userRank, intraHierarchyInfo_);
    InsAlgTemplate1 interTempAlg(param, resCtx.topoInfo.userRank, interHierarchyInfo_);
    HCCL_INFO("[Orchestrate] templates: intra=%s inter=%s",
              intraTempAlg.Describe().c_str(), interTempAlg.Describe().c_str());

    HCCL_WARNING("[ALLTOALL_V2_DEBUG][Orchestrate] L5: templates constructed. intra=%s inter=%s",
        intraTempAlg.Describe().c_str(), interTempAlg.Describe().c_str());

    intraTempAlg.SetMeshDimensions(rankSizeLevel0_, rankSizeLevel1_, rankIdxLevel0_, rankIdxLevel1_);
    interTempAlg.SetMeshDimensions(rankSizeLevel0_, rankSizeLevel1_, rankIdxLevel0_, rankIdxLevel1_);
    HCCL_INFO("[Orchestrate] SetMeshDimensions: rankSize0=%llu rankSize1=%llu rankIdx0=%llu rankIdx1=%llu",
              rankSizeLevel0_, rankSizeLevel1_, rankIdxLevel0_, rankIdxLevel1_);

    HCCL_WARNING("[ALLTOALL_V2_CRASH_DIAG][L4] Templates constructed. intra=%s inter=%s",
        intraTempAlg.Describe().c_str(), interTempAlg.Describe().c_str());

    HCCL_INFO("[InsV2AlltoAllParallelExecutor][Orchestrate] intra template=%s inter template=%s",
              intraTempAlg.Describe().c_str(), interTempAlg.Describe().c_str());

    if (param.engine == CommEngine::COMM_ENGINE_AICPU_TS ||
        param.engine == CommEngine::COMM_ENGINE_AIV) {
        interTempAlg.SetchannelsPerRank(interLinkMap_);
    }

    bool hasInterComm = !interLinkMap_.empty();
    HCCL_INFO("[InsV2AlltoAllParallelExecutor][Orchestrate] hasInterComm=%d", hasInterComm);

    HCCL_WARNING("[ALLTOALL_V2_DEBUG][Orchestrate] L6: calling PrepareResForTemplate. hasInterComm=%d", hasInterComm);

    PrepareResForTemplate(intraTempAlg, interTempAlg);

    HCCL_WARNING("[ALLTOALL_V2_DEBUG][Orchestrate] L7: entering OrchestrateLoop");

    HcclResult ret = OrchestrateLoop(param, resCtx, intraTempAlg, interTempAlg, hasInterComm);
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
    HCCL_INFO("[InsV2AlltoAllParallelExecutor] splitDataSize is %f, %f", splitDataSize[0], splitDataSize[1]);
    return;
}

template <typename AlgTopoMatch, typename InsAlgTemplate0, typename InsAlgTemplate1>
HcclResult InsV2AlltoAllParallelExecutor<AlgTopoMatch, InsAlgTemplate0, InsAlgTemplate1>::OrchestrateLoop(
    const OpParam &param, const AlgResourceCtxSerializable &resCtx, InsAlgTemplate0 &tempAlgIntra,
    InsAlgTemplate1 &tempAlgInter, bool hasInterComm)
{
    HCCL_INFO("[InsV2AlltoAllParallelExecutor][OrchestrateLoop] Entry. maxTmpMem=%llu dataCount=%llu dataSize=%llu "
              "dataTypeSize=%u hasInterComm=%d",
              maxTmpMemSize_, dataCount_, dataSize_, dataTypeSize_, hasInterComm);
    std::vector<float> splitDataSize;
    if (!hasInterComm) {
        splitDataSize.push_back(1.0f);
        splitDataSize.push_back(0.0f);
        HCCL_INFO("[InsV2AlltoAllParallelExecutor][OrchestrateLoop] no inter links, forcing split=[1.0,0.0]");
    } else {
        GetParallelDataSplit(splitDataSize);
    }

    TemplateResource interTempAlgRes;
    interTempAlgRes.channels = interLinkMap_;
    interTempAlgRes.threads = interThreads_;
    interTempAlgRes.aivCommInfoPtr = resCtx.aivCommInfoPtr;

    TemplateResource intraTempAlgRes;
    intraTempAlgRes.channels = intraLinkMap_;
    intraTempAlgRes.threads = intraThreads_;
    intraTempAlgRes.aivCommInfoPtr = resCtx.aivCommInfoPtr;

    // Stage 0 
    TemplateDataParams tempAlgParamsIntra0;
    TemplateDataParams tempAlgParamsInter0;
    // Stage 1
    TemplateDataParams tempAlgParamsIntra1;
    TemplateDataParams tempAlgParamsInter1;
 
    intraTempAlgRes.channels = intraLinkMap_;
    interTempAlgRes.channels = interLinkMap_;
    
    // Stage 0 的 Full-Mesh 数据量 和 Stage 0 的 Clos 数据量
    u64 finalDataCountPerLoopAxis0 = static_cast<u64>(splitDataSize[0] * dataCount_);;
    u64 finalDataCountPerLoopAxis1 = dataCount_ - finalDataCountPerLoopAxis0;;

    // 总 rank 数
    u64 totalRankCount = rankSizeLevel0_ * rankSizeLevel1_;
    u64 perPeerInputChunkSize = dataSize_ / totalRankCount;

    HcclMem hcclBuff = resCtx.cclMem;
    u64 hcclBuffSize = resCtx.cclMem.size;

    // 只支持 all to all 场景， 也就是 发送量 = 接受量

    u64 scratchBufferIntraOffset0 = 0;
    u64 scratchBufferIntraSize0 = finalDataCountPerLoopAxis0 * dataTypeSize_;

    u64 scratchBufferInterOffset0 = scratchBufferIntraOffset0 + 2 * scratchBufferIntraSize0;
    u64 scratchBufferInterSize0 = finalDataCountPerLoopAxis1 * dataTypeSize_;
    
    u64 scratchBufferIntraOffset1 = scratchBufferInterOffset0 + 2 * scratchBufferInterSize0;
    u64 scratchBufferIntraSize1 = scratchBufferInterSize0;

    u64 scratchBufferInterOffset1 = scratchBufferIntraOffset1 + 2 * scratchBufferIntraSize1;
    u64 scratchBufferInterSize1 = scratchBufferIntraSize0;

    if (4 * dataSize_ > hcclBuffSize) {
        HCCL_ERROR("[InsV2AlltoAllParallelExecutor][OrchestrateLoop] FATAL: HCCL buffer too small for double buffering. "
                   "dataSize=%llu hcclBuffSize=%llu", dataSize_, hcclBuffSize);
        return HcclResult::HCCL_E_INTERNAL;
    } else {
        HCCL_INFO("[InsV2AlltoAllParallelExecutor][OrchestrateLoop] HCCL buffer size=%llu sufficient for double buffering dataSize=%llu",
                  hcclBuffSize, dataSize_);
        HCCL_INFO("[InsV2AlltoAllParallelExecutor][OrchestrateLoop] Scratch buffer offsets and sizes: "
                  "intra0 off=%llu size=%llu inter0 off=%llu size=%llu "
                  "intra1 off=%llu size=%llu inter1 off=%llu size=%llu",
                  scratchBufferIntraOffset0, scratchBufferIntraSize0,
                  scratchBufferInterOffset0, scratchBufferInterSize0,
                  scratchBufferIntraOffset1, scratchBufferIntraSize1,
                  scratchBufferInterOffset1, scratchBufferInterSize1);
    }

    // Stage 0 数据预处理和模板参数准备
    // 00 01 10 11 20 21 30 31 40 41 50 51 60 61 70 71
    // tempAlgParamsIntra0 输入是 inputPtr， LocalCopy 到 scratchBufferIntra0，输出是 scratchBufferIntra0 + scratchBufferIntraSize0
    // 00 40 10 50 20 60 30 70 是 intra0 的输入
    {
        tempAlgParamsIntra0.enableRemoteMemAccess = param.opMode == OpMode::OFFLOAD;
        tempAlgParamsIntra0.count = totalRankCount;

        tempAlgParamsIntra0.buffInfo.inputPtr = param.inputPtr;
        tempAlgParamsIntra0.buffInfo.inBuffType = BufferType::INPUT;
        tempAlgParamsIntra0.buffInfo.inputSize = param.inputSize;
        tempAlgParamsIntra0.buffInfo.inBuffBaseOff = 0;
        tempAlgParamsIntra0.inputSliceStride = perPeerInputChunkSize;

        tempAlgParamsIntra0.buffInfo.hcclBuff = resCtx.cclMem;
        tempAlgParamsIntra0.buffInfo.hcclBuffType = BufferType::HCCL_BUFFER;
        tempAlgParamsIntra0.buffInfo.hcclBuffSize = 2 * scratchBufferIntraSize0;
        tempAlgParamsIntra0.buffInfo.hcclBuffBaseOff = scratchBufferIntraOffset0;

        // 不需要 copy out
        tempAlgParamsIntra0.buffInfo.outputPtr = nullptr;

        tempAlgParamsIntra0.sliceSize = scratchBufferIntraSize0;
        tempAlgParamsIntra0.count = finalDataCountPerLoopAxis0;
    }
    
    // tempAlgParamsInter0 输入是 inputPtr， LocalCopy 到 scratchBufferInter0，输出是 scratchBufferInter0 + scratchBufferInterSize0
    {
        tempAlgParamsInter0.enableRemoteMemAccess = param.opMode == OpMode::OFFLOAD;
        tempAlgParamsInter0.count = totalRankCount;

        tempAlgParamsInter0.buffInfo.inputPtr = param.inputPtr;
        tempAlgParamsInter0.buffInfo.inBuffType = BufferType::INPUT;
        tempAlgParamsInter0.buffInfo.inputSize = param.inputSize;
        tempAlgParamsInter0.buffInfo.inBuffBaseOff = static_cast<u64>(splitDataSize[0] * perPeerInputChunkSize);
        tempAlgParamsInter0.inputSliceStride = perPeerInputChunkSize;

        tempAlgParamsInter0.buffInfo.hcclBuff = resCtx.cclMem;
        tempAlgParamsInter0.buffInfo.hcclBuffType = BufferType::HCCL_BUFFER;
        tempAlgParamsInter0.buffInfo.hcclBuffSize = 2 * scratchBufferInterSize0;
        tempAlgParamsInter0.buffInfo.hcclBuffBaseOff = scratchBufferInterOffset0;

        // 不需要 copy out
        tempAlgParamsInter0.buffInfo.outputPtr = nullptr;

        tempAlgParamsInter0.sliceSize = scratchBufferInterSize0;
        tempAlgParamsInter0.count = finalDataCountPerLoopAxis1;
    }

    // 开始 Stage 0，Presync，确保所有线程和模板同步准备好进行第一阶段的计算。
    CHK_RET(PreSyncInterThreads(mainThread_, templateMainThreads_, syncNotifyOnTemplates_));

    if (splitDataSize[0] > 0.0f) {
        HCCL_INFO("[InsV2AlltoAllParallelExecutor][OrchestrateLoop] Running intra template for Stage 0 with dataCount=%llu",
                  finalDataCountPerLoopAxis0);
         CHK_RET(tempAlgIntra.KernelRun(param, tempAlgParamsIntra0, intraTempAlgRes));
    }
   
    if (splitDataSize[1] > 0.0f) {
        HCCL_INFO("[InsV2AlltoAllParallelExecutor][OrchestrateLoop] Running inter template for Stage 0 with dataCount=%llu",
                  finalDataCountPerLoopAxis1);
        CHK_RET(tempAlgInter.KernelRun(param, tempAlgParamsInter0, interTempAlgRes));
    }

    // 结束 Stage 0，Postync，确保所有线程和模板同步结束第一阶段的计算。
    CHK_RET(PostSyncInterThreads(mainThread_, templateMainThreads_, syncNotifyOnMain_));


    // Stage 1 数据预处理和模板参数准备
    // tempAlgParamsIntra1 输入是 inputPtr， LocalCopy 到 scratchBufferIntra0，输出是 scratchBufferIntra0 + scratchBufferIntraSize0
    {
        tempAlgParamsIntra1.enableRemoteMemAccess = param.opMode == OpMode::OFFLOAD;
        tempAlgParamsIntra1.count = totalRankCount;

        tempAlgParamsIntra1.buffInfo.inputPtr = resCtx.cclMem.addr;
        tempAlgParamsIntra1.buffInfo.inBuffType = BufferType::HCCL_BUFFER;
        tempAlgParamsIntra1.buffInfo.inputSize = scratchBufferInterSize0;
        tempAlgParamsIntra1.buffInfo.inBuffBaseOff = scratchBufferInterOffset0 + scratchBufferInterSize0; // double buffer: read from the other half of the HCCL buffer
        tempAlgParamsIntra1.inputSliceStride = perPeerInputChunkSize - static_cast<u64>(splitDataSize[0] * perPeerInputChunkSize);

        tempAlgParamsIntra1.buffInfo.hcclBuff = resCtx.cclMem;
        tempAlgParamsIntra1.buffInfo.hcclBuffType = BufferType::HCCL_BUFFER;
        tempAlgParamsIntra1.buffInfo.hcclBuffSize = 2 * scratchBufferIntraSize1;
        tempAlgParamsIntra1.buffInfo.hcclBuffBaseOff = scratchBufferIntraOffset1;

        tempAlgParamsIntra1.buffInfo.outputPtr = param.outputPtr;
        tempAlgParamsIntra1.buffInfo.outBuffType = BufferType::OUTPUT;
        tempAlgParamsIntra1.buffInfo.outputSize = param.outputSize;
        tempAlgParamsIntra1.buffInfo.outBuffBaseOff = static_cast<u64>(splitDataSize[0] * perPeerInputChunkSize);
        tempAlgParamsIntra1.outputSliceStride = perPeerInputChunkSize;

        tempAlgParamsIntra1.sliceSize = scratchBufferIntraSize1;
        tempAlgParamsIntra1.count = finalDataCountPerLoopAxis1;
    }
    
    // tempAlgParamsInter1 输入是 inputPtr， LocalCopy 到 scratchBufferInter0，输出是 scratchBufferInter0 + scratchBufferInterSize0
    {
        tempAlgParamsInter1.enableRemoteMemAccess = param.opMode == OpMode::OFFLOAD;
        tempAlgParamsInter1.count = totalRankCount;

        tempAlgParamsInter1.buffInfo.inputPtr = resCtx.cclMem.addr;
        tempAlgParamsInter1.buffInfo.inBuffType = BufferType::HCCL_BUFFER;
        tempAlgParamsInter1.buffInfo.inputSize = scratchBufferIntraSize0;
        tempAlgParamsInter1.buffInfo.inBuffBaseOff = scratchBufferIntraOffset0 + scratchBufferIntraSize0; // double buffer: read from the other half of the HCCL buffer
        tempAlgParamsInter1.inputSliceStride = static_cast<u64>(splitDataSize[0] * perPeerInputChunkSize);

        tempAlgParamsInter1.buffInfo.hcclBuff = resCtx.cclMem;
        tempAlgParamsInter1.buffInfo.hcclBuffType = BufferType::HCCL_BUFFER;
        tempAlgParamsInter1.buffInfo.hcclBuffSize = 2 * scratchBufferInterSize1;
        tempAlgParamsInter1.buffInfo.hcclBuffBaseOff = scratchBufferInterOffset1;

        // 不需要 copy out
        tempAlgParamsInter1.buffInfo.outputPtr = param.outputPtr;
        tempAlgParamsInter1.buffInfo.outBuffType = BufferType::OUTPUT;
        tempAlgParamsInter1.buffInfo.outputSize = param.outputSize;
        tempAlgParamsInter1.buffInfo.outBuffBaseOff = 0;
        tempAlgParamsInter1.outputSliceStride = perPeerInputChunkSize;

        tempAlgParamsInter1.sliceSize = scratchBufferInterSize1;
        tempAlgParamsInter1.count = finalDataCountPerLoopAxis0;
    }

    // 开始 Stage 1，Presync，确保所有线程和模板同步准备好进行第一阶段的计算。
    CHK_RET(PreSyncInterThreads(mainThread_, templateMainThreads_, syncNotifyOnTemplates_));

    if (splitDataSize[1] > 0.0f) {
        HCCL_INFO("[InsV2AlltoAllParallelExecutor][OrchestrateLoop] Running intra template for Stage 1 with dataCount=%llu",
                  finalDataCountPerLoopAxis1);
         CHK_RET(tempAlgIntra.KernelRun(param, tempAlgParamsIntra1, intraTempAlgRes));
    }
   
    if (splitDataSize[0] > 0.0f) {
        HCCL_INFO("[InsV2AlltoAllParallelExecutor][OrchestrateLoop] Running inter template for Stage 1 with dataCount=%llu",
                  finalDataCountPerLoopAxis0);
        CHK_RET(tempAlgInter.KernelRun(param, tempAlgParamsInter1, interTempAlgRes));
    }
    
    // 结束 Stage 1，Postync，确保所有线程和模板同步结束第一阶段的计算。
    CHK_RET(PostSyncInterThreads(mainThread_, templateMainThreads_, syncNotifyOnMain_));
    
    HCCL_INFO("[InsV2AlltoAllParallelExecutor][OrchestrateLoop] End.");
    return HcclResult::HCCL_SUCCESS;
}

// UBX topology (8-card boards) — Mesh intra, Clos inter
REGISTER_EXECUTOR_BY_TWO_TEMPS(
    HcclCMDType::HCCL_CMD_ALLTOALL,
    InsAlltoAllParallelMesh2DClosV2,
    InsV2AlltoAllParallelExecutor,
    TopoMatchUBX,
    InsTempAlltoAllMesh2DV2,
    InsTempAlltoAllMeshClosV2);

}  // namespace ops_hccl
