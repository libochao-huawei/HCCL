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
#include "hcomm_primitives.h"
#include "alg_data_trans_wrapper.h"
#include "ins_temp_alltoall_mesh_2d_v2.h"
#include "ins_temp_alltoall_mesh_clos_v2.h"

#include "topo_match_clos_mesh_2d_v2.h"
#include "topo_match_clos_mesh_2d_ubx_v2.h"
#include "topo_match_ubx.h"

namespace ops_hccl {

/**
 * Reorganize INTRA↔INTER scratch buffers with dual-save DMA cell swap.
 * Processes the upper triangle (dy > sx) only to avoid RAW hazards.
 * Both INTRA(sx,dy) and INTRA(dy,sx) are saved to temp1/temp2 BEFORE
 * any writes occur, eliminating the symmetric-pair corruption bug (D-1).
 *
 * INTRA: column-major source-by-dx → dest-by-dx
 * INTER: row-major source-by-dy → dest-by-dy
 *
 * Uses per-direction cellSize (D-2 fix): intraCellSize vs interCellSize.
 *
 * Steps per pair (sx,dy):
 *   ① SAVE temp1 = INTRA(sx,dy)
 *   ② SAVE temp2 = INTRA(dy,sx)  (if dy < xSize)
 *   ③ MOVE INTER(sx,dy) → INTRA(dy,sx)  (if sx < ySize && dy < xSize)
 *   ④ MOVE INTER(dy,sx) → INTRA(sx,dy)  (if sz_inter_dysx > 0)
 *   ⑤ RESTORE temp1 → INTER(dy,sx)
 *   ⑥ RESTORE temp2 → INTER(sx,dy)  (if sx < ySize && dy < xSize)
 */
static HcclResult ReorganizeScratches(
    ThreadHandle thread,
    void *intraBuf, void *interBuf,
    u32 xSize, u32 ySize,
    u64 intraSliceSize, u64 interSliceSize,
    u64 perPeerMesh, u64 perPeerClos)
{
    // CONSTRAINT: splitRatio MUST be 0.5 for this function to produce correct
    // results. When intraCellSize ≠ interCellSize (splitRatio ≠ 0.5), the MOVE
    // and RESTORE DMA copy sizes differ from destination cell capacities, causing
    // buffer overflow. This is a known limitation (tracked as D-3).
    // The executor's GetParallelDataSplit hardcodes splitRatio=0.5, guaranteeing
    // intraCellSize == interCellSize at all call sites.
    // Future redesign (v1.4) will remove this constraint by clamping copy sizes.
    uint8_t *intra = static_cast<uint8_t*>(intraBuf);
    uint8_t *inter = static_cast<uint8_t*>(interBuf);
    u64 totalRanks = xSize * ySize;
    u64 intraCellSize = (intraSliceSize + totalRanks - 1) / totalRanks;
    u64 interCellSize = (interSliceSize + totalRanks - 1) / totalRanks;

    if (intraCellSize != interCellSize) {
        HCCL_ERROR("[ReorganizeScratches] splitRatio violation: intraCell[%llu] != interCell[%llu] "
                  "intraSlice[%llu] interSlice[%llu] totalRanks[%llu]",
                  intraCellSize, interCellSize,
                  intraSliceSize, interSliceSize, static_cast<u64>(totalRanks));
        return HCCL_E_INTERNAL;
    }

    // Dual-save: need 2× the max cell for temp1+temp2
    u64 maxCellSize = std::max(intraCellSize, interCellSize);
    if (maxCellSize == 0) return HCCL_SUCCESS;
    std::vector<uint8_t> tempBuf(2 * maxCellSize);
    uint8_t *temp = tempBuf.data();
    uint8_t *temp1 = temp;
    uint8_t *temp2 = temp + maxCellSize;
    HCCL_WARNING("[ALLTOALL_V2_DEBUG][ReorganizeScratches] xSize=%u ySize=%u intraSlice=%llu interSlice=%llu "
                 "perPeerMesh=%llu perPeerClos=%llu intraCell=%llu interCell=%llu maxCell=%llu temp=%zu",
                 xSize, ySize, intraSliceSize, interSliceSize,
                 perPeerMesh, perPeerClos, intraCellSize, interCellSize, maxCellSize, tempBuf.size());

    for (u32 sx = 0; sx < xSize; sx++) {
        // Intra block info for source row sx
        u64 intraBlockSize = (sx == xSize - 1)
            ? intraSliceSize - perPeerMesh * (xSize - 1) : perPeerMesh;
        if (intraBlockSize <= 0) intraBlockSize = perPeerMesh;

        // Inter block info for source row sx (valid only if sx < ySize)
        u64 interBlockSize = 0;
        bool interValid = (sx < ySize);
        if (interValid) {
            interBlockSize = (sx == ySize - 1)
                ? interSliceSize - perPeerClos * (ySize - 1) : perPeerClos;
            if (interBlockSize <= 0) interBlockSize = perPeerClos;
        }

        // Upper triangle only — avoids RAW hazard on symmetric pairs
        for (u32 dy = sx + 1; dy < ySize; dy++) {
            if (sx >= ySize && dy >= xSize) continue;

            // Intra block info for source row dy (for INTRA(dy,sx) cell)
            u64 dyIntraBlockSize = (dy == xSize - 1)
                ? intraSliceSize - perPeerMesh * (xSize - 1) : perPeerMesh;
            if (dyIntraBlockSize <= 0) dyIntraBlockSize = perPeerMesh;

            // ===== INTRA(sx,dy) size (always valid: sx < xSize) =====
            u64 intraSrcOff_sxdy = sx * perPeerMesh + dy * intraCellSize;
            u64 intraRemaining_sxdy = (intraSrcOff_sxdy < sx * perPeerMesh + intraBlockSize)
                ? (sx * perPeerMesh + intraBlockSize - intraSrcOff_sxdy) : 0;
            u64 sz_intra_sxdy = std::min(intraCellSize, intraRemaining_sxdy);

            // ===== INTRA(dy,sx) size (valid only if dy < xSize) =====
            u64 sz_intra_dysx = 0;
            u64 intraSrcOff_dysx = 0;
            if (dy < xSize) {
                intraSrcOff_dysx = dy * perPeerMesh + sx * intraCellSize;
                u64 intraRemaining_dysx = (intraSrcOff_dysx < dy * perPeerMesh + dyIntraBlockSize)
                    ? (dy * perPeerMesh + dyIntraBlockSize - intraSrcOff_dysx) : 0;
                sz_intra_dysx = std::min(intraCellSize, intraRemaining_dysx);
            }

            // Skip if both INTRA cells are empty
            if (sz_intra_sxdy == 0 && sz_intra_dysx == 0) continue;

            // ===== INTER(sx,dy) size (valid only if sx < ySize) =====
            u64 sz_inter_sxdy = 0;
            u64 interSrcOff_sxdy = 0;
            if (interValid) {
                interSrcOff_sxdy = sx * perPeerClos + dy * interCellSize;
                u64 interRemaining_sxdy = (interSrcOff_sxdy < sx * perPeerClos + interBlockSize)
                    ? (sx * perPeerClos + interBlockSize - interSrcOff_sxdy) : 0;
                sz_inter_sxdy = std::min(interCellSize, interRemaining_sxdy);
            }

            // ===== INTER(dy,sx) size (valid: dy < ySize from loop bound) =====
            u64 sz_inter_dysx = 0;
            u64 interSrcOff_dysx = 0;
            u64 dyInterBlockSize = (dy == ySize - 1)
                ? interSliceSize - perPeerClos * (ySize - 1) : perPeerClos;
            if (dyInterBlockSize <= 0) dyInterBlockSize = perPeerClos;
            interSrcOff_dysx = dy * perPeerClos + sx * interCellSize;
            u64 interRemaining_dysx = (interSrcOff_dysx < dy * perPeerClos + dyInterBlockSize)
                ? (dy * perPeerClos + dyInterBlockSize - interSrcOff_dysx) : 0;
            sz_inter_dysx = std::min(interCellSize, interRemaining_dysx);

            int32_t rc;

            // ① SAVE temp1 = INTRA(sx, dy) — first save before any writes
            if (sz_intra_sxdy > 0) {
                rc = HcommLocalCopyOnThread(thread, temp1, intra + intraSrcOff_sxdy, sz_intra_sxdy);
                if (rc != 0) return HCCL_E_INTERNAL;
            }

            // ② SAVE temp2 = INTRA(dy, sx) — second save before any writes
            if (sz_intra_dysx > 0) {
                rc = HcommLocalCopyOnThread(thread, temp2, intra + intraSrcOff_dysx, sz_intra_dysx);
                if (rc != 0) return HCCL_E_INTERNAL;
            }

            // ③ MOVE INTER(sx,dy) → INTRA(dy,sx) (if INTER source & INTRA dest valid)
            if (sz_inter_sxdy > 0 && dy < xSize) {
                rc = HcommLocalCopyOnThread(thread, intra + intraSrcOff_dysx, inter + interSrcOff_sxdy,
                                             sz_inter_sxdy);
                if (rc != 0) return HCCL_E_INTERNAL;
            }

            // ④ MOVE INTER(dy,sx) → INTRA(sx,dy) (if INTER source has data)
            if (sz_inter_dysx > 0) {
                rc = HcommLocalCopyOnThread(thread, intra + intraSrcOff_sxdy, inter + interSrcOff_dysx,
                                             sz_inter_dysx);
                if (rc != 0) return HCCL_E_INTERNAL;
            }

            // ⑤ RESTORE temp1 → INTER(dy,sx)
            if (sz_intra_sxdy > 0) {
                rc = HcommLocalCopyOnThread(thread, inter + interSrcOff_dysx, temp1, sz_intra_sxdy);
                if (rc != 0) return HCCL_E_INTERNAL;
            }

            // ⑥ RESTORE temp2 → INTER(sx,dy) (only if INTER dest exists: sx < ySize)
            if (sz_intra_dysx > 0 && interValid) {
                rc = HcommLocalCopyOnThread(thread, inter + interSrcOff_sxdy, temp2, sz_intra_dysx);
                if (rc != 0) return HCCL_E_INTERNAL;
            }
        }
    }
    return HCCL_SUCCESS;
}

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
    tempAlgParamsIntra0.buffInfo.outputSize = resCtx.cclMem.size;

    tempAlgParamsIntra0.buffInfo.inBuffBaseOff = dataOffset;
    tempAlgParamsIntra0.buffInfo.outBuffBaseOff = scratchOffset;
    tempAlgParamsIntra0.buffInfo.hcclBuffBaseOff = scratchOffset;
    tempAlgParamsIntra0.sliceSize = dataCountPerLoopAxis0 * dataTypeSize_;
    tempAlgParamsIntra0.count = dataCountPerLoopAxis0;
    tempAlgParamsIntra0.tailSize = tempAlgParamsIntra0.sliceSize;

    u64 totalRankCount = rankSizeLevel0_ * rankSizeLevel1_;
    u64 perPeerInputChunkSize = dataSize_ / totalRankCount;
    tempAlgParamsIntra0.inputSliceStride = perPeerInputChunkSize;
    tempAlgParamsIntra0.outputSliceStride = tempAlgParamsIntra0.sliceSize;
    tempAlgParamsIntra0.repeatNum = 1;
    tempAlgParamsIntra0.inputRepeatStride = 0;
    tempAlgParamsIntra0.outputRepeatStride = 0;
    tempAlgParamsIntra0.enableRemoteMemAccess = param.opMode == OpMode::OFFLOAD;

    u64* sendCountsData = reinterpret_cast<u64*>(param.all2AllVDataDes.sendCounts);
    tempAlgParamsIntra0.sendCounts.assign(sendCountsData, sendCountsData + totalRankCount);
    u64* sdisplsData = reinterpret_cast<u64*>(param.all2AllVDataDes.sdispls);
    tempAlgParamsIntra0.sdispls.assign(sdisplsData, sdisplsData + totalRankCount);

    HCCL_INFO(
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
    tempAlgParamsInter1.buffInfo.outputSize = resCtx.cclMem.size;

    u64 totalRankCount = rankSizeLevel0_ * rankSizeLevel1_;
    u64 perPeerInputChunkSize = dataSize_ / totalRankCount;
    tempAlgParamsInter1.buffInfo.inBuffBaseOff = perPeerInputChunkSize / 2;
    tempAlgParamsInter1.buffInfo.outBuffBaseOff = scratchOffset;
    tempAlgParamsInter1.buffInfo.hcclBuffBaseOff = scratchOffset;
    tempAlgParamsInter1.sliceSize = dataCountPerLoopAxis1 * dataTypeSize_;
    tempAlgParamsInter1.count = dataCountPerLoopAxis1;
    tempAlgParamsInter1.tailSize = tempAlgParamsInter1.sliceSize;

    tempAlgParamsInter1.inputSliceStride = perPeerInputChunkSize * rankSizeLevel0_;
    tempAlgParamsInter1.outputSliceStride = tempAlgParamsInter1.sliceSize;
    tempAlgParamsInter1.repeatNum = 1;
    tempAlgParamsInter1.inputRepeatStride = 0;
    tempAlgParamsInter1.outputRepeatStride = 0;
    tempAlgParamsInter1.enableRemoteMemAccess = param.opMode == OpMode::OFFLOAD;

    u64* sendCountsData = reinterpret_cast<u64*>(param.all2AllVDataDes.sendCounts);
    tempAlgParamsInter1.sendCounts.assign(sendCountsData, sendCountsData + totalRankCount);
    u64* sdisplsData = reinterpret_cast<u64*>(param.all2AllVDataDes.sdispls);
    tempAlgParamsInter1.sdispls.assign(sdisplsData, sdisplsData + totalRankCount);

    HCCL_INFO("[InsV2AlltoAllParallelExecutor][GenTemplateAlgParamsInter1] rank[%u] inBuffBaseOff[%llu] "
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
    tempAlgParamsInter0.buffInfo.outputSize = param.outputSize * dataTypeSize_;

    tempAlgParamsInter0.buffInfo.inBuffBaseOff = scratchOffset;
    // v1.12 Fix B: dataOffset (= dataOffset0 = 0 for first loop iteration) happens
    // to equal the correct interleaved X-data offset (0 within each peer output chunk).
    // X-data starts at offset 0 within each peer chunk in the interleaved output layout.
    tempAlgParamsInter0.buffInfo.outBuffBaseOff = dataOffset;
    tempAlgParamsInter0.buffInfo.hcclBuffBaseOff = scratchOffset;
    tempAlgParamsInter0.sliceSize = dataCountPerLoopAxis0 * dataTypeSize_;
    tempAlgParamsInter0.count = dataCountPerLoopAxis0;
    tempAlgParamsInter0.tailSize = tempAlgParamsInter0.sliceSize;

    tempAlgParamsInter0.inputSliceStride = dataSize_ * rankSizeLevel0_;
    u64 totalRankCount = rankSizeLevel0_ * rankSizeLevel1_;
    u64 perPeerOutputChunkSize = (dataSize_ + totalRankCount - 1) / totalRankCount;
    tempAlgParamsInter0.outputSliceStride = rankSizeLevel0_ * perPeerOutputChunkSize;
    tempAlgParamsInter0.repeatNum = 1;
    tempAlgParamsInter0.inputRepeatStride = 0;
    tempAlgParamsInter0.outputRepeatStride = 0;
    tempAlgParamsInter0.enableRemoteMemAccess = param.opMode == OpMode::OFFLOAD;

    HCCL_INFO("[InsV2AlltoAllParallelExecutor][GenTemplateAlgParamsInter0] rank[%u] inBuffBaseOff[%llu] "
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
    tempAlgParamsIntra1.buffInfo.inputPtr = static_cast<uint8_t*>(resCtx.cclMem.addr) + scratchOffset;
    tempAlgParamsIntra1.buffInfo.outputPtr = param.outputPtr;
    tempAlgParamsIntra1.buffInfo.hcclBuff = resCtx.cclMem;
    tempAlgParamsIntra1.buffInfo.inBuffBaseOff = 0;

    // v1.12 Fix A: Interleaved output layout — Y-data within each peer chunk
    // is at perPeerOutputChunkSize/2 offset, NOT dataOffset (blocked-layout midpoint).
    // dataSize_ reflects total send data; perPeerOutputChunkSize computed from dataSize_
    // (not param.outputSize) to match the actual data distribution for asymmetric AlltoAllV.
    u64 totalRankCount = rankSizeLevel0_ * rankSizeLevel1_;
    u64 perPeerOutputChunkSize = (dataSize_ + totalRankCount - 1) / totalRankCount;
    tempAlgParamsIntra1.buffInfo.outBuffBaseOff = perPeerOutputChunkSize / 2;

    tempAlgParamsIntra1.buffInfo.hcclBuffBaseOff = scratchOffset;
    tempAlgParamsIntra1.buffInfo.inBuffType = BufferType::HCCL_BUFFER;
    tempAlgParamsIntra1.buffInfo.outBuffType = BufferType::OUTPUT;
    tempAlgParamsIntra1.buffInfo.hcclBuffType = BufferType::HCCL_BUFFER;
    tempAlgParamsIntra1.buffInfo.inputSize = param.inputSize;
    tempAlgParamsIntra1.buffInfo.outputSize = param.outputSize * dataTypeSize_;
    tempAlgParamsIntra1.sliceSize = dataCountPerLoopAxis1 * dataTypeSize_;
    tempAlgParamsIntra1.count = dataCountPerLoopAxis1;
    tempAlgParamsIntra1.tailSize = tempAlgParamsIntra1.sliceSize;

    tempAlgParamsIntra1.inputSliceStride = dataSize_;
    // totalRankCount and perPeerOutputChunkSize already computed above (v1.12 Fix A)
    tempAlgParamsIntra1.outputSliceStride = perPeerOutputChunkSize;
    tempAlgParamsIntra1.repeatNum = 1;
    tempAlgParamsIntra1.inputRepeatStride = 0;
    tempAlgParamsIntra1.outputRepeatStride = 0;
    tempAlgParamsIntra1.enableRemoteMemAccess = param.opMode == OpMode::OFFLOAD;

    HCCL_INFO("[InsV2AlltoAllParallelExecutor][GenTemplateAlgParamsIntra1] rank[%u] inBuffBaseOff[%llu] "
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

    if (maxCountPerLoop == 0) {
        HCCL_ERROR("[InsV2AlltoAllParallelExecutor][OrchestrateLoop] FATAL: maxCountPerLoop is 0. "
                   "scratchMemBlockSize=%llu dataTypeSize_=%u",
                   scratchMemBlockSize, dataTypeSize_);
        return HcclResult::HCCL_E_INTERNAL;
    }

    u32 loopTimes = dataCount_ / maxCountPerLoop + ((dataCount_ % maxCountPerLoop == 0) ? 0 : 1);

    HCCL_INFO("[InsV2AlltoAllParallelExecutor][OrchestrateLoop] scratch: intraMulti=%u interMulti=%u total=%u "
              "scratchBlock=%llu intraOff=%llu interOff=%llu intraSz=%llu interSz=%llu",
              scratchMultipleIntra, scratchMultipleInter, totalScratchMultiple,
              scratchMemBlockSize, intraScratchOffset, interScratchOffset,
              scratchMultipleIntra * scratchMemBlockSize, scratchMultipleInter * scratchMemBlockSize);
    HCCL_INFO("[InsV2AlltoAllParallelExecutor][OrchestrateLoop] loop: maxCountPerLoop=%llu axis0=%llu axis1=%llu "
              "loopTimes=%u",
              maxCountPerLoop, dataCountPerLoopAxis0, dataCountPerLoopAxis1, loopTimes);

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

        HCCL_WARNING("[ALLTOALL_V2_DEBUG][OrchestrateLoop] Loop[%u/%u] start: currPart0=%llu currPart1=%llu "
                  "splitData=[%.4f,%.4f]",
                  loopIndex, loopTimes, currCountPart0, currCountPart1,
                  splitDataSize[0], splitDataSize[1]);

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

        HCCL_INFO("[InsV2AlltoAllParallelExecutor][OrchestrateLoop] loop[%u/%u] Stage1 start. "
                  "currPart0=%llu currPart1=%llu dataOff0=%llu dataOff1=%llu "
                  "peerChunk0=%llu peerChunk1=%llu",
                  loopIndex, loopTimes, currCountPart0, currCountPart1,
                  dataOffset0, dataOffset1, currPerPeerChunkSize0, currPerPeerChunkSize1);

        // Stage 1: Intra0 (X-axis mesh) + Inter1 (Y-axis clos)
        // Executed sequentially on the orchestrator's main thread.
        // The templates run their internal sub-thread PreSync/PostSync in parallel,
        // but KernelRun calls are sequential by-design (same pattern as AllGather executor).
        // Future optimization: thread-pool dispatch for true intra-template parallelism.
        GenTemplateAlgParamsIntra0(param, resCtx, dataOffset0, currCountPart0, intraScratchOffset,
                                   tempAlgParamsIntra0);
        HCCL_INFO("[OrchestrateLoop] Stage1-Intra0: sliceSize=%llu count=%llu dataOffset=%llu scratchOff=%llu",
                  tempAlgParamsIntra0.sliceSize, tempAlgParamsIntra0.count,
                  tempAlgParamsIntra0.buffInfo.inBuffBaseOff, tempAlgParamsIntra0.buffInfo.hcclBuffBaseOff);
        HCCL_WARNING("[ALLTOALL_V2_DEBUG][OrchestrateLoop] Stage1 Intra0: "
            "inputSliceStride=%llu outputSliceStride=%llu inBuffBase=%llu outBuffBase=%llu "
            "sliceSize=%llu rankSize0=%llu rankSize1=%llu",
            tempAlgParamsIntra0.inputSliceStride, tempAlgParamsIntra0.outputSliceStride,
            tempAlgParamsIntra0.buffInfo.inBuffBaseOff, tempAlgParamsIntra0.buffInfo.outBuffBaseOff,
            tempAlgParamsIntra0.sliceSize, rankSizeLevel0_, rankSizeLevel1_);
        HcclResult intra0Ret = HCCL_SUCCESS;
        if (currCountPart0 > 0 && tempAlgParamsIntra0.sliceSize > 0) {
            intra0Ret = tempAlgIntra.KernelRun(param, tempAlgParamsIntra0, intraTempAlgRes);
        } else {
            HCCL_INFO("[OrchestrateLoop] Stage1-Intra0 skipped: currPart0=%llu sliceSize=%llu",
                      currCountPart0, tempAlgParamsIntra0.sliceSize);
        }
        HCCL_INFO("[OrchestrateLoop] Stage1-Intra0: ret=0x%x", intra0Ret);

        HcclResult inter1Ret = HCCL_SUCCESS;
        if (hasInterComm && currCountPart1 > 0) {
            GenTemplateAlgParamsInter1(param, resCtx, dataOffset1, currCountPart1, interScratchOffset,
                                       tempAlgParamsInter1);
            HCCL_INFO("[OrchestrateLoop] Stage1-Inter1: sliceSize=%llu count=%llu dataOffset=%llu scratchOff=%llu hasInterComm=%d",
                      tempAlgParamsInter1.sliceSize, tempAlgParamsInter1.count,
                      tempAlgParamsInter1.buffInfo.inBuffBaseOff, tempAlgParamsInter1.buffInfo.hcclBuffBaseOff,
                      hasInterComm);
            HCCL_WARNING("[ALLTOALL_V2_DEBUG][OrchestrateLoop] Stage1 Inter1: "
                "inputSliceStride=%llu outputSliceStride=%llu inBuffBase=%llu outBuffBase=%llu "
                "sliceSize=%llu rankSize0=%llu rankSize1=%llu",
                tempAlgParamsInter1.inputSliceStride, tempAlgParamsInter1.outputSliceStride,
                tempAlgParamsInter1.buffInfo.inBuffBaseOff, tempAlgParamsInter1.buffInfo.outBuffBaseOff,
                tempAlgParamsInter1.sliceSize, rankSizeLevel0_, rankSizeLevel1_);
            if (tempAlgParamsInter1.sliceSize > 0) {
                inter1Ret = tempAlgInter.KernelRun(param, tempAlgParamsInter1, interTempAlgRes);
            } else {
                HCCL_INFO("[OrchestrateLoop] Stage1-Inter1 skipped: sliceSize=%llu",
                          tempAlgParamsInter1.sliceSize);
            }
            HCCL_INFO("[OrchestrateLoop] Stage1-Inter1: ret=0x%x", inter1Ret);
        }

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

        // v1.18 Fix: Replace HcommFenceOnThread with HcommThreadSynchronize
        // HcommFenceOnThread → HcommFlushV2() may interact poorly with AICPU-managed threads
        // HcommThreadSynchronize is a per-thread stream sync that catches hardware errors
        // and ensures all DMA on the thread's stream is complete before ReorganizeScratches.
        for (auto &thread : intraThreads_) {
            int32_t syncRet = HcommThreadSynchronize(thread);
            if (syncRet != 0) {
                HCCL_ERROR("[InsV2AlltoAllParallelExecutor] Stage 1 intra thread[0x%016llx] sync failed: %d",
                           static_cast<u64>(thread), syncRet);
                return HcclResult::HCCL_E_INTERNAL;
            }
        }
        for (auto &thread : interThreads_) {
            int32_t syncRet = HcommThreadSynchronize(thread);
            if (syncRet != 0) {
                HCCL_ERROR("[InsV2AlltoAllParallelExecutor] Stage 1 inter thread[0x%016llx] sync failed: %d",
                           static_cast<u64>(thread), syncRet);
                return HcclResult::HCCL_E_INTERNAL;
            }
        }

#ifndef AICPU_COMPILE
        if (loopTimes == 1 && param.engine == CommEngine::COMM_ENGINE_CCU) {
            ccuKernelLaunchNumIntra0_ = intraTempAlgRes.submitInfos.size();
            ccuKernelLaunchNumInter1_ = interTempAlgRes.submitInfos.size();
        }
#endif

        // v1.3 C-8 fix: Cross-copy reorganization between Stage 1 and Stage 2.
        // Transforms scratch regions: INTRA (source-by-dx) → INTER (dest-by-dy),
        // INTER (source-by-dy) → INTRA (dest-by-dx).
        // This swaps ownership so Stage 2 rings read from the correct layout.
        {
            u64 meshSliceSize = currCountPart0 * dataTypeSize_;
            u64 closSliceSize = currCountPart1 * dataTypeSize_;
            u64 perPeerMesh = (meshSliceSize + rankSizeLevel0_ - 1) / rankSizeLevel0_;
            u64 perPeerClos = (closSliceSize + rankSizeLevel1_ - 1) / rankSizeLevel1_;

            bool needIntraToInter = (rankSizeLevel0_ > 1 && rankSizeLevel1_ > 1 && currCountPart0 > 0);
            bool needInterToIntra = (rankSizeLevel0_ > 1 && rankSizeLevel1_ > 1 && currCountPart1 > 0);

            if (needIntraToInter || needInterToIntra) {
                uint8_t *intraBuf = static_cast<uint8_t*>(resCtx.cclMem.addr) + intraScratchOffset;
                uint8_t *interBuf = static_cast<uint8_t*>(resCtx.cclMem.addr) + interScratchOffset;

                HCCL_WARNING("[ALLTOALL_V2_DEBUG][OrchestrateLoop] ReorganizeScratches: "
                          "meshSliceSize=%llu closSliceSize=%llu perPeerMesh=%llu perPeerClos=%llu "
                          "rankSize0=%llu rankSize1=%llu intraBuf@0x%llx interBuf@0x%llx "
                          "needIntraToInter=%d needInterToIntra=%d",
                          meshSliceSize, closSliceSize, perPeerMesh, perPeerClos,
                          rankSizeLevel0_, rankSizeLevel1_,
                          reinterpret_cast<uint64_t>(intraBuf),
                          reinterpret_cast<uint64_t>(interBuf),
                          needIntraToInter, needInterToIntra);
                HcclResult reorgRet = ReorganizeScratches(mainThread_,
                    intraBuf, interBuf, rankSizeLevel0_, rankSizeLevel1_,
                    meshSliceSize, closSliceSize, perPeerMesh, perPeerClos);
                CHK_RET(reorgRet);
                HCCL_WARNING("[ALLTOALL_V2_DEBUG][OrchestrateLoop] Reorganization complete.");
            } else {
                HCCL_WARNING("[ALLTOALL_V2_DEBUG][OrchestrateLoop] Skipping reorganization: rankSize0=%llu rankSize1=%llu "
                          "currPart0=%llu currPart1=%llu",
                          rankSizeLevel0_, rankSizeLevel1_, currCountPart0, currCountPart1);
            }
        }

        // Stage 2 PreSync
        CHK_RET(PreSyncInterThreads(mainThread_, templateMainThreads_, syncNotifyOnTemplates_));

        HCCL_INFO("[InsV2AlltoAllParallelExecutor][OrchestrateLoop] loop[%u/%u] Stage2 start. "
                  "Intra1 reads INTRA scratch (dest-by-dx), Inter0 reads INTER scratch (dest-by-dy).",
                  loopIndex, loopTimes);

        // v1.3 C-8 fix: cross-copy swaps scratch ownership.
        // Intra1 reads from INTRA (now dest-by-dx), Inter0 reads from INTER (now dest-by-dy).
        GenTemplateAlgParamsIntra1(param, resCtx, dataOffset1, currCountPart1, intraScratchOffset,
                                   tempAlgParamsIntra1);
        HCCL_INFO("[OrchestrateLoop] Stage2-Intra1: sliceSize=%llu count=%llu scratchOff=%llu",
                  tempAlgParamsIntra1.sliceSize, tempAlgParamsIntra1.count,
                  tempAlgParamsIntra1.buffInfo.inBuffBaseOff);
        HcclResult intra1Ret = tempAlgIntra.KernelRun(param, tempAlgParamsIntra1, intraTempAlgRes);
        HCCL_INFO("[OrchestrateLoop] Stage2-Intra1: ret=0x%x", intra1Ret);

        HcclResult inter0Ret = HCCL_SUCCESS;
        if (hasInterComm && currCountPart0 > 0) {
            GenTemplateAlgParamsInter0(param, resCtx, dataOffset0, currCountPart0, interScratchOffset,
                                       tempAlgParamsInter0);
            HCCL_INFO("[OrchestrateLoop] Stage2-Inter0: sliceSize=%llu count=%llu scratchOff=%llu hasInterComm=%d",
                      tempAlgParamsInter0.sliceSize, tempAlgParamsInter0.count,
                      tempAlgParamsInter0.buffInfo.inBuffBaseOff, hasInterComm);
            inter0Ret = tempAlgInter.KernelRun(param, tempAlgParamsInter0, interTempAlgRes);
            HCCL_INFO("[OrchestrateLoop] Stage2-Inter0: ret=0x%x", inter0Ret);
        }

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

}  // namespace ops_hccl
