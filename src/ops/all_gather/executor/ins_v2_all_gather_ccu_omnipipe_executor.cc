/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "ins_v2_all_gather_ccu_omnipipe_executor.h"

#ifndef AICPU_COMPILE
#include <algorithm>
#include <cstdlib>
#include <limits>
#include "topo_match_ubx.h"
#include "ccu_temp_all_gather_omnipipe_mesh_1d_mem2mem.h"
#include "ccu_temp_all_gather_omnipipe_nhr_1d_multi_jetty_mem2mem.h"
#include "ccu_temp_all_gather_omnipipe_local_copy.h"
#include "alg_data_trans_wrapper.h"

namespace ops_hccl {
namespace {
constexpr u32 FAST_ACTION_SEG = 0;
constexpr u32 FAST_ACTION_COPY_IN = 1;
constexpr u32 FAST_ACTION_LEVEL2 = 2;
constexpr u32 FAST_ACTION_LEVEL0 = 3;
constexpr u32 FAST_ACTION_LEVEL1 = 4;
constexpr u32 FAST_ACTION_COPY_OUT = 5;
constexpr u32 FAST_ACTION_PRE_SYNC_Z = 6;
constexpr u32 FAST_ACTION_POST_SYNC_Z = 7;
constexpr u32 FAST_ACTION_PRE_SYNC_XY = 8;
constexpr u32 FAST_ACTION_POST_SYNC_XY = 9;

} // namespace

template <typename AlgTopoMatch, typename CcuAlgTemplate0, typename CcuAlgTemplate1, typename CcuAlgTemplate2>
InsV2AllGatherCcuOmniPipeExecutor<AlgTopoMatch, CcuAlgTemplate0, CcuAlgTemplate1,
                                  CcuAlgTemplate2>::InsV2AllGatherCcuOmniPipeExecutor()
{
}

template <typename AlgTopoMatch, typename CcuAlgTemplate0, typename CcuAlgTemplate1, typename CcuAlgTemplate2>
HcclResult InsV2AllGatherCcuOmniPipeExecutor<AlgTopoMatch, CcuAlgTemplate0, CcuAlgTemplate1,
                                             CcuAlgTemplate2>::InitCommInfo(
    const OpParam &param, const TopoInfoWithNetLayerDetails *topoInfo,
    const AlgHierarchyInfoForAllLevel &algHierarchyInfo)
{
    myRank_ = topoInfo->userRank;
    rankSize_ = topoInfo->userRankSize;
    devType_ = topoInfo->deviceType;
    dataType_ = param.DataDes.dataType;
    dataCount_ = param.DataDes.count;
    dataTypeSize_ = DATATYPE_SIZE_TABLE[param.DataDes.dataType];
    dataSize_ = dataCount_ * dataTypeSize_;
    opMode_ = param.opMode;
    algHierarchyInfo_ = algHierarchyInfo;
    return HCCL_SUCCESS;
}

template <typename AlgTopoMatch, typename CcuAlgTemplate0, typename CcuAlgTemplate1, typename CcuAlgTemplate2>
HcclResult InsV2AllGatherCcuOmniPipeExecutor<AlgTopoMatch, CcuAlgTemplate0, CcuAlgTemplate1,
                                             CcuAlgTemplate2>::InitExecutorInfo()
{
    CHK_PRT_RET(algHierarchyInfo_.infos.empty() || algHierarchyInfo_.infos[0].size() < 2,
                HCCL_ERROR("[InsV2AllGatherCcuOmniPipeExecutor] invalid hierarchy info."), HCCL_E_PARA);
    rankSizeLevel_.assign(OMNIPIPE_LEVEL_NUM, 1);
    rankIdxLevel_.assign(OMNIPIPE_LEVEL_NUM, 0);
    rankSizeLevel_[OMNIPIPE_LEVEL0] = algHierarchyInfo_.infos[0][0].size();
    rankSizeLevel_[OMNIPIPE_LEVEL1] = algHierarchyInfo_.infos[0][1].size() / rankSizeLevel_[OMNIPIPE_LEVEL0];
    CHK_PRT_RET(rankSizeLevel_[OMNIPIPE_LEVEL0] == 0 || rankSizeLevel_[OMNIPIPE_LEVEL1] == 0,
                HCCL_ERROR("[InsV2AllGatherCcuOmniPipeExecutor] invalid rank size level0[%llu], level1[%llu].",
                           rankSizeLevel_[OMNIPIPE_LEVEL0], rankSizeLevel_[OMNIPIPE_LEVEL1]), HCCL_E_PARA);
    if (algHierarchyInfo_.infos.size() > 1 && !algHierarchyInfo_.infos[1].empty() &&
        !algHierarchyInfo_.infos[1][0].empty()) {
        rankSizeLevel_[OMNIPIPE_LEVEL2] = algHierarchyInfo_.infos[1][0].size();
    }
    rankIdxLevel_[OMNIPIPE_LEVEL0] = myRank_ % rankSizeLevel_[OMNIPIPE_LEVEL0];
    rankIdxLevel_[OMNIPIPE_LEVEL1] =
        myRank_ % (rankSizeLevel_[OMNIPIPE_LEVEL0] * rankSizeLevel_[OMNIPIPE_LEVEL1]) / rankSizeLevel_[OMNIPIPE_LEVEL0];
    rankIdxLevel_[OMNIPIPE_LEVEL2] = myRank_ / (rankSizeLevel_[OMNIPIPE_LEVEL0] * rankSizeLevel_[OMNIPIPE_LEVEL1]);
    HCCL_INFO("[InsV2AllGatherCcuOmniPipeExecutor] rankSizeLevel[%llu,%llu,%llu], rankIdxLevel[%llu,%llu,%llu].",
              rankSizeLevel_[0], rankSizeLevel_[1], rankSizeLevel_[2], rankIdxLevel_[0], rankIdxLevel_[1],
              rankIdxLevel_[2]);
    return HCCL_SUCCESS;
}

template <typename AlgTopoMatch, typename CcuAlgTemplate0, typename CcuAlgTemplate1, typename CcuAlgTemplate2>
HcclResult InsV2AllGatherCcuOmniPipeExecutor<AlgTopoMatch, CcuAlgTemplate0, CcuAlgTemplate1,
                                             CcuAlgTemplate2>::CalcAlgHierarchyInfo(
    HcclComm comm, TopoInfoWithNetLayerDetails *topoInfo, AlgHierarchyInfoForAllLevel &algHierarchyInfo)
{
    AlgTopoMatch topoMatch;
    CHK_RET(topoMatch.MatchTopo(comm, topoInfo, algHierarchyInfo));
    return HCCL_SUCCESS;
}

template <typename AlgTopoMatch, typename CcuAlgTemplate0, typename CcuAlgTemplate1, typename CcuAlgTemplate2>
HcclResult InsV2AllGatherCcuOmniPipeExecutor<AlgTopoMatch, CcuAlgTemplate0, CcuAlgTemplate1,
                                             CcuAlgTemplate2>::CalcResLevel(
    HcclComm comm, const OpParam &param, const TopoInfoWithNetLayerDetails *topoInfo,
    const std::shared_ptr<CommonAlgTemplateBase> &tempAlg, AlgResourceRequest &resourceRequest)
{
    AlgResourceRequest levelRequest;
    CHK_RET(tempAlg->CalcRes(comm, param, topoInfo, levelRequest));
    resourceRequest.slaveThreadNum += levelRequest.slaveThreadNum + 1;
    resourceRequest.notifyNumOnMainThread += 1;
    resourceRequest.notifyNumPerThread.emplace_back(levelRequest.notifyNumOnMainThread + 1);
    resourceRequest.notifyNumPerThread.insert(resourceRequest.notifyNumPerThread.end(),
                                              levelRequest.notifyNumPerThread.begin(),
                                              levelRequest.notifyNumPerThread.end());
    resourceRequest.ccuKernelInfos.insert(resourceRequest.ccuKernelInfos.end(), levelRequest.ccuKernelInfos.begin(),
                                          levelRequest.ccuKernelInfos.end());
    resourceRequest.ccuKernelNum.insert(resourceRequest.ccuKernelNum.end(), levelRequest.ccuKernelNum.begin(),
                                        levelRequest.ccuKernelNum.end());
    return HCCL_SUCCESS;
}

template <typename AlgTopoMatch, typename CcuAlgTemplate0, typename CcuAlgTemplate1, typename CcuAlgTemplate2>
HcclResult InsV2AllGatherCcuOmniPipeExecutor<AlgTopoMatch, CcuAlgTemplate0, CcuAlgTemplate1,
                                             CcuAlgTemplate2>::CalcRes(
    HcclComm comm, const OpParam &param, const TopoInfoWithNetLayerDetails *topoInfo,
    const AlgHierarchyInfoForAllLevel &algHierarchyInfo, AlgResourceRequest &resourceRequest)
{
    CHK_RET(InitCommInfo(param, topoInfo, algHierarchyInfo));
    CHK_RET(InitExecutorInfo());
    std::vector<std::vector<u32>> subCommRanks0{algHierarchyInfo.infos[0][0]};
    std::vector<std::vector<u32>> subCommRanks1(1);
    for (u32 i = myRank_ % rankSizeLevel_[OMNIPIPE_LEVEL0]; i < algHierarchyInfo.infos[0][1].size();
         i += rankSizeLevel_[OMNIPIPE_LEVEL0]) {
        subCommRanks1[0].push_back(algHierarchyInfo.infos[0][1][i]);
    }
    std::vector<std::vector<u32>> subCommRanks2;
    if (algHierarchyInfo.infos.size() > 1 && !algHierarchyInfo.infos[1].empty() && !algHierarchyInfo.infos[1][0].empty()) {
        subCommRanks2.push_back(algHierarchyInfo.infos[1][0]);
    } else {
        subCommRanks2.push_back(std::vector<u32>{myRank_});
    }

    if (rankSizeLevel_[OMNIPIPE_LEVEL0] > 1) {
        CHK_RET(CalcResLevel(comm, param, topoInfo, std::make_shared<CcuAlgTemplate0>(param, myRank_, subCommRanks0),
                             resourceRequest));
    }
    if (rankSizeLevel_[OMNIPIPE_LEVEL1] > 1) {
        CHK_RET(CalcResLevel(comm, param, topoInfo, std::make_shared<CcuAlgTemplate1>(param, myRank_, subCommRanks1),
                             resourceRequest));
    }
    if (rankSizeLevel_[OMNIPIPE_LEVEL2] > 1) {
        CHK_RET(CalcResLevel(comm, param, topoInfo,
                             std::make_shared<CcuTempAllGatherOmniPipeNHR1DMultiJettyMem2Mem>(
                                 param, myRank_, subCommRanks2, CommTopo::COMM_TOPO_CUSTOM),
                             resourceRequest));
    }
    CHK_RET(CalcResLevel(comm, param, topoInfo,
                         std::make_shared<CcuTempAllGatherOmniPipeLocalCopy>(param, myRank_, subCommRanks0),
                         resourceRequest));
    return HCCL_SUCCESS;
}

template <typename AlgTopoMatch, typename CcuAlgTemplate0, typename CcuAlgTemplate1, typename CcuAlgTemplate2>
HcclResult InsV2AllGatherCcuOmniPipeExecutor<AlgTopoMatch, CcuAlgTemplate0, CcuAlgTemplate1,
                                              CcuAlgTemplate2>::PrepareResForTemplateLevel(
    u32 level, const std::shared_ptr<CommonAlgTemplateBase> &tempAlg, u32 &ccuKernelNumIdx, u32 &ccuKernelHandleOffset,
    const AlgResourceCtxSerializable &resCtx, TemplateResource &templateResource)
{
    const u32 levelThreadNum = tempAlg->GetThreadNum();
    u32 threadStart = 1;
    const u32 levelThreadIdx = level < OMNIPIPE_LEVEL_NUM ? level : OMNIPIPE_LEVEL_NUM;
    for (u32 i = 0; i < levelThreadIdx; ++i) {
        if (i < levelThreads_.size()) {
            threadStart += levelThreads_[i].size();
        }
    }
    CHK_PRT_RET(levelThreadIdx >= levelThreads_.size() || threadStart + levelThreadNum > threads_.size(),
                HCCL_ERROR("[InsV2AllGatherCcuOmniPipeExecutor][PrepareResForTemplateLevel] invalid level[%u], "
                           "threadStart[%u], levelThreadNum[%u], threadNum[%zu].", level, threadStart, levelThreadNum,
                           threads_.size()), HCCL_E_PARA);
    levelThreads_[levelThreadIdx].assign(threads_.begin() + threadStart,
                                         threads_.begin() + threadStart + levelThreadNum);
    AlgResourceRequest levelRequest;
    CHK_RET(tempAlg->GetRes(levelRequest));
    if (level < OMNIPIPE_LEVEL2) {
        tempMainThreadsXY_.push_back(levelThreads_[levelThreadIdx].at(0));
        ntfIdxCtrlToTempXY_.push_back(levelRequest.notifyNumOnMainThread);
        ntfIdxTempToCtrlXY_.push_back(tempMainThreadsXY_.size() - 1);
    } else if (level == OMNIPIPE_LEVEL2) {
        tempMainThreadsZ_.push_back(levelThreads_[levelThreadIdx].at(0));
        ntfIdxCtrlToTempZ_.push_back(levelRequest.notifyNumOnMainThread);
        ntfIdxTempToCtrlZ_.push_back(tempMainThreadsXY_.size() + tempMainThreadsZ_.size() - 1);
    }
    templateResource.threads = levelThreads_[levelThreadIdx];
    if (ccuKernelNumIdx < resCtx.ccuKernelNum.size()) {
        u32 kernelNum = resCtx.ccuKernelNum[ccuKernelNumIdx];
        CHK_PRT_RET(ccuKernelHandleOffset + kernelNum > resCtx.ccuKernels.size(),
                    HCCL_ERROR("[InsV2AllGatherCcuOmniPipeExecutor][PrepareResForTemplateLevel] invalid kernel "
                               "offset[%u], num[%u], total[%zu].", ccuKernelHandleOffset, kernelNum,
                               resCtx.ccuKernels.size()), HCCL_E_PARA);
        templateResource.ccuKernels.insert(templateResource.ccuKernels.end(),
                                           resCtx.ccuKernels.begin() + ccuKernelHandleOffset,
                                           resCtx.ccuKernels.begin() + ccuKernelHandleOffset + kernelNum);
        ccuKernelNumIdx++;
        ccuKernelHandleOffset += kernelNum;
    }
    return HCCL_SUCCESS;
}

template <typename AlgTopoMatch, typename CcuAlgTemplate0, typename CcuAlgTemplate1, typename CcuAlgTemplate2>
HcclResult InsV2AllGatherCcuOmniPipeExecutor<AlgTopoMatch, CcuAlgTemplate0, CcuAlgTemplate1,
                                             CcuAlgTemplate2>::GenTemplateAlgParamsByDimData(
    TemplateDataParams &tempAlgParams, const StepSliceInfo &stepSliceInfo) const
{
    tempAlgParams.buffInfo.inBuffType = BufferType::HCCL_BUFFER;
    tempAlgParams.buffInfo.outBuffType = BufferType::HCCL_BUFFER;
    tempAlgParams.buffInfo.inputPtr = tempAlgParams.buffInfo.hcclBuff.addr;
    tempAlgParams.buffInfo.outputPtr = tempAlgParams.buffInfo.hcclBuff.addr;
    tempAlgParams.buffInfo.inBuffBaseOff = stepSliceInfo.buffInfo.inBuffBaseOff;
    tempAlgParams.buffInfo.outBuffBaseOff = stepSliceInfo.buffInfo.outBuffBaseOff;
    tempAlgParams.buffInfo.hcclBuffBaseOff = stepSliceInfo.buffInfo.hcclBuffBaseOff;
    tempAlgParams.stepSliceInfo = stepSliceInfo;
    return HCCL_SUCCESS;
}

template <typename AlgTopoMatch, typename CcuAlgTemplate0, typename CcuAlgTemplate1, typename CcuAlgTemplate2>
HcclResult InsV2AllGatherCcuOmniPipeExecutor<AlgTopoMatch, CcuAlgTemplate0, CcuAlgTemplate1,
                                             CcuAlgTemplate2>::RunCcuLocalCopy(
    const OpParam &param, void *srcPtr, void *dstPtr, const HcclMem &hcclBuff, u64 srcOffset, u64 dstOffset, u64 copySize,
    TemplateResource &copyResource, TemplateResource *fastLaunchResource) const
{
    CcuTempAllGatherOmniPipeLocalCopy copyTemp;
    TemplateDataParams copyParams;
    copyParams.buffInfo.inputPtr = srcPtr;
    copyParams.buffInfo.outputPtr = dstPtr;
    copyParams.buffInfo.hcclBuff = hcclBuff;
    copyParams.buffInfo.inputSize = (srcPtr == hcclBuff.addr) ? hcclBuff.size : param.inputSize;
    copyParams.buffInfo.outputSize = (dstPtr == hcclBuff.addr) ? hcclBuff.size : param.outputSize;
    copyParams.buffInfo.hcclBuffSize = hcclBuff.size;
    copyParams.buffInfo.inBuffBaseOff = 0;
    copyParams.buffInfo.outBuffBaseOff = 0;
    copyParams.inputSliceStride = srcOffset;
    copyParams.outputSliceStride = dstOffset;
    copyParams.sliceSize = copySize;
    u32 oldSubmitSize = copyResource.submitInfos.size();
    CHK_RET(copyTemp.KernelRun(param, copyParams, copyResource));
    if (fastLaunchResource != nullptr) {
        fastLaunchResource->submitInfos.insert(fastLaunchResource->submitInfos.end(),
                                               copyResource.submitInfos.begin() + oldSubmitSize,
                                               copyResource.submitInfos.end());
    }
    return HCCL_SUCCESS;
}

template <typename AlgTopoMatch, typename CcuAlgTemplate0, typename CcuAlgTemplate1, typename CcuAlgTemplate2>
HcclResult InsV2AllGatherCcuOmniPipeExecutor<AlgTopoMatch, CcuAlgTemplate0, CcuAlgTemplate1,
                                              CcuAlgTemplate2>::FastLaunchSaveCtx(
    const OpParam &param, const TemplateResource &orderedSubmitInfos)
{
    if (orderedSubmitInfos.submitInfos.empty()) {
        return HCCL_SUCCESS;
    }
    CHK_PRT_RET(orderedSubmitInfos.submitInfos.size() > static_cast<size_t>(std::numeric_limits<u32>::max()),
                HCCL_ERROR("[InsV2AllGatherCcuOmniPipeExecutor][FastLaunchSaveCtx] submitInfo size[%zu] exceeds u32.",
                           orderedSubmitInfos.submitInfos.size()), HCCL_E_PARA);
    std::vector<u32> ccuKernelNumList(MAX_TEMP_NUM_IN_ALGO, 0);
    ccuKernelNumList[FAST_ACTION_SEG] = orderedSubmitInfos.submitInfos.size();
    std::vector<std::vector<CcuKernelSubmitInfo>> submitInfosList;
    submitInfosList.resize(MAX_TEMP_NUM_IN_ALGO);
    submitInfosList[FAST_ACTION_SEG] = orderedSubmitInfos.submitInfos;
    return FastLaunchSaveCtxMultiTemplate(param, threads_.size(), orderedSubmitInfos.submitInfos.size(), threads_,
                                          ccuKernelNumList, submitInfosList);
}

namespace {
void AppendFastLaunchInfos(const TemplateResource &from, u32 oldSubmitSize, TemplateResource &to, u32 action)
{
    for (u32 idx = oldSubmitSize; idx < from.submitInfos.size(); ++idx) {
        CcuKernelSubmitInfo submitInfo = from.submitInfos[idx];
        submitInfo.action = action;
        to.submitInfos.push_back(submitInfo);
    }
}

void AppendFastLaunchMarker(TemplateResource &to, u32 action)
{
    CcuKernelSubmitInfo submitInfo{};
    submitInfo.action = action;
    to.submitInfos.push_back(submitInfo);
}
} // namespace

template <typename AlgTopoMatch, typename CcuAlgTemplate0, typename CcuAlgTemplate1, typename CcuAlgTemplate2>
HcclResult InsV2AllGatherCcuOmniPipeExecutor<AlgTopoMatch, CcuAlgTemplate0, CcuAlgTemplate1,
                                             CcuAlgTemplate2>::Orchestrate(
    const OpParam &param, const AlgResourceCtxSerializable &resCtx)
{
    CHK_RET(InitCommInfo(param, &resCtx.topoInfo, resCtx.algHierarchyInfo));
    maxTmpMemSize_ = resCtx.cclMem.size;
    CHK_RET(InitExecutorInfo());
    threads_ = resCtx.threads;
    controlThread_ = threads_.at(0);
    levelThreads_.resize(OMNIPIPE_LEVEL_NUM + 1);

    std::vector<std::vector<u32>> subCommRanks0{resCtx.algHierarchyInfo.infos[0][0]};
    std::vector<std::vector<u32>> subCommRanks1(1);
    for (u32 i = myRank_ % rankSizeLevel_[OMNIPIPE_LEVEL0]; i < resCtx.algHierarchyInfo.infos[0][1].size();
         i += rankSizeLevel_[OMNIPIPE_LEVEL0]) {
        subCommRanks1[0].push_back(resCtx.algHierarchyInfo.infos[0][1][i]);
    }
    std::vector<std::vector<u32>> subCommRanks2;
    if (resCtx.algHierarchyInfo.infos.size() > 1 && !resCtx.algHierarchyInfo.infos[1].empty() &&
        !resCtx.algHierarchyInfo.infos[1][0].empty()) {
        subCommRanks2.push_back(resCtx.algHierarchyInfo.infos[1][0]);
    } else {
        subCommRanks2.push_back(std::vector<u32>{myRank_});
    }

    std::map<u32, std::shared_ptr<CommonAlgTemplateBase>> tempMap;
    if (rankSizeLevel_[OMNIPIPE_LEVEL0] > 1) {
        tempMap[OMNIPIPE_LEVEL0] = std::make_shared<CcuAlgTemplate0>(param, myRank_, subCommRanks0);
    }
    if (rankSizeLevel_[OMNIPIPE_LEVEL1] > 1) {
        tempMap[OMNIPIPE_LEVEL1] = std::make_shared<CcuAlgTemplate1>(param, myRank_, subCommRanks1);
    }
    if (rankSizeLevel_[OMNIPIPE_LEVEL2] > 1) {
        tempMap[OMNIPIPE_LEVEL2] = std::make_shared<CcuTempAllGatherOmniPipeNHR1DMultiJettyMem2Mem>(
            param, myRank_, subCommRanks2, CommTopo::COMM_TOPO_CUSTOM);
    }
    tempMap[OMNIPIPE_LEVEL_NUM] = std::make_shared<CcuTempAllGatherOmniPipeLocalCopy>(param, myRank_, subCommRanks0);

    std::map<u32, TemplateResource> tempResMap;
    u32 ccuKernelNumIdx = 0;
    u32 ccuKernelHandleOffset = 0;
    for (auto &temp : tempMap) {
        CHK_RET(PrepareResForTemplateLevel(temp.first, temp.second, ccuKernelNumIdx, ccuKernelHandleOffset, resCtx,
                                           tempResMap[temp.first]));
    }
    return OrchestrateLoop(param, resCtx, tempMap, tempResMap);
}

template <typename AlgTopoMatch, typename CcuAlgTemplate0, typename CcuAlgTemplate1, typename CcuAlgTemplate2>
HcclResult InsV2AllGatherCcuOmniPipeExecutor<AlgTopoMatch, CcuAlgTemplate0, CcuAlgTemplate1,
                                             CcuAlgTemplate2>::OrchestrateLoop(
    const OpParam &param, const AlgResourceCtxSerializable &resCtx,
    std::map<u32, std::shared_ptr<CommonAlgTemplateBase>> &tempMap, std::map<u32, TemplateResource> &tempResMap)
{
    std::vector<std::vector<EndpointAttrBwCoeff>> endpointAttrBw;
    CHK_RET(CalAllLevelEndpointAttrBwCoeff(param.hcclComm, myRank_, OMNIPIPE_LEVEL_NUM, endpointAttrBw));
    std::vector<EndpointAttrBwCoeff> endpointAttrBwNew;
    for (u32 level = 0; level < endpointAttrBw.size(); ++level) {
        for (u32 idx = 0; idx < endpointAttrBw[level].size(); ++idx) {
            EndpointAttrBwCoeff bw = endpointAttrBw[level][idx];
            if (level == OMNIPIPE_LEVEL0 && rankSizeLevel_[OMNIPIPE_LEVEL1] > 1 && idx != 0) {
                bw /= (rankSizeLevel_[OMNIPIPE_LEVEL1] - 1);
            } else if (level != OMNIPIPE_LEVEL0 && algHierarchyInfo_.infos.size() > level &&
                       algHierarchyInfo_.infos[level][idx].size() > 1) {
                bw /= (algHierarchyInfo_.infos[level][idx].size() - 1);
            }
            endpointAttrBwNew.push_back(bw);
        }
    }
    u64 scratchBoundDataCount = maxTmpMemSize_ / rankSize_ / HCCL_MIN_SLICE_ALIGN * HCCL_MIN_SLICE_ALIGN / dataTypeSize_;
    u64 maxCountPerLoop = std::min(scratchBoundDataCount, UB_MAX_DATA_SIZE / dataTypeSize_);
    maxCountPerLoop = std::min(maxCountPerLoop, dataCount_);
    CHK_PRT_RET(maxCountPerLoop == 0, HCCL_ERROR("[InsV2AllGatherCcuOmniPipeExecutor] maxCountPerLoop is 0."),
                HCCL_E_PARA);
    u64 loopTimes = dataCount_ / maxCountPerLoop + static_cast<u64>(dataCount_ % maxCountPerLoop != 0);

    auto calcSliceInfo = [&](u64 currDataCount) {
        u64 perLoopSize = currDataCount * dataTypeSize_;
        OmniPipeSliceParam sliceParam;
        sliceParam.levelRankSize = {rankSizeLevel_[0], rankSizeLevel_[1], rankSizeLevel_[2]};
        sliceParam.endpointAttrBw = endpointAttrBwNew;
        sliceParam.dataSizePerLoop.assign(rankSize_, perLoopSize);
        sliceParam.dataWholeSize.assign(rankSize_, perLoopSize);
        sliceParam.dataTypeSize = dataTypeSize_;
        sliceParam.levelRankId = {rankIdxLevel_[0], rankIdxLevel_[1], rankIdxLevel_[2]};
        sliceParam.opMode = opMode_;
        sliceParam.engine = CommEngine::COMM_ENGINE_CCU;
        return CalcAGOmniPipeSliceInfo(sliceParam);
    };

    TemplateDataParams paramsLevel0;
    TemplateDataParams paramsLevel1;
    TemplateDataParams paramsLevel2;
    paramsLevel0.buffInfo.hcclBuff = resCtx.cclMem;
    paramsLevel1.buffInfo.hcclBuff = resCtx.cclMem;
    paramsLevel2.buffInfo.hcclBuff = resCtx.cclMem;
    TemplateResource orderedFastLaunchInfos;

    u64 processedDataCount = 0;
    for (u64 loop = 0; loop < loopTimes; ++loop) {
        u64 currDataCount = (loop == loopTimes - 1) ? dataCount_ - processedDataCount : maxCountPerLoop;
        u64 currBytes = currDataCount * dataTypeSize_;
        u32 oldSubmitSize = tempResMap[OMNIPIPE_LEVEL_NUM].submitInfos.size();
        CHK_RET(RunCcuLocalCopy(param, param.inputPtr, resCtx.cclMem.addr, resCtx.cclMem,
                                processedDataCount * dataTypeSize_, myRank_ * currBytes, currBytes,
                                tempResMap[OMNIPIPE_LEVEL_NUM]));
        AppendFastLaunchInfos(tempResMap[OMNIPIPE_LEVEL_NUM], oldSubmitSize, orderedFastLaunchInfos,
                              FAST_ACTION_COPY_IN);
        tempResMap[OMNIPIPE_LEVEL0].submitInfos.clear();
        tempResMap[OMNIPIPE_LEVEL1].submitInfos.clear();
        tempResMap[OMNIPIPE_LEVEL2].submitInfos.clear();

        OmniPipeSliceInfo sliceInfo = calcSliceInfo(currDataCount);
        u32 level2StepCount = std::max<u32>(sliceInfo.dataSliceLevel2.size(), 1);
        u32 level0StepCount = sliceInfo.dataSliceLevel0.size() / level2StepCount;
        CHK_PRT_RET(sliceInfo.dataSliceLevel2.empty() || level0StepCount == 0 ||
                        sliceInfo.dataSliceLevel1.size() < sliceInfo.dataSliceLevel0.size(),
                    HCCL_ERROR("[InsV2AllGatherCcuOmniPipeExecutor][OrchestrateLoop] invalid slice info, "
                               "level0[%zu], level1[%zu], level2StepCount[%u].", sliceInfo.dataSliceLevel0.size(),
                               sliceInfo.dataSliceLevel1.size(), level2StepCount), HCCL_E_PARA);
        for (u32 stepZ = 0; stepZ < level2StepCount; ++stepZ) {
            if (tempMap.count(OMNIPIPE_LEVEL2) != 0) {
                CHK_RET(GenTemplateAlgParamsByDimData(paramsLevel2, sliceInfo.dataSliceLevel2[stepZ]));
                CHK_RET(PreSyncInterThreads(controlThread_, tempMainThreadsZ_, ntfIdxCtrlToTempZ_));
                AppendFastLaunchMarker(orderedFastLaunchInfos, FAST_ACTION_PRE_SYNC_Z);
                oldSubmitSize = tempResMap[OMNIPIPE_LEVEL2].submitInfos.size();
                CHK_RET(tempMap[OMNIPIPE_LEVEL2]->KernelRun(param, paramsLevel2, tempResMap[OMNIPIPE_LEVEL2]));
                AppendFastLaunchInfos(tempResMap[OMNIPIPE_LEVEL2], oldSubmitSize, orderedFastLaunchInfos,
                                      FAST_ACTION_LEVEL2);
            }
            for (u32 stepXY = 0; stepXY < level0StepCount; ++stepXY) {
                u32 idx = stepZ * level0StepCount + stepXY;
                CHK_RET(PreSyncInterThreads(controlThread_, tempMainThreadsXY_, ntfIdxCtrlToTempXY_));
                AppendFastLaunchMarker(orderedFastLaunchInfos, FAST_ACTION_PRE_SYNC_XY);
                if (tempMap.count(OMNIPIPE_LEVEL0) != 0) {
                    CHK_RET(GenTemplateAlgParamsByDimData(paramsLevel0, sliceInfo.dataSliceLevel0[idx]));
                    oldSubmitSize = tempResMap[OMNIPIPE_LEVEL0].submitInfos.size();
                    CHK_RET(tempMap[OMNIPIPE_LEVEL0]->KernelRun(param, paramsLevel0, tempResMap[OMNIPIPE_LEVEL0]));
                    AppendFastLaunchInfos(tempResMap[OMNIPIPE_LEVEL0], oldSubmitSize, orderedFastLaunchInfos,
                                          FAST_ACTION_LEVEL0);
                }
                if (tempMap.count(OMNIPIPE_LEVEL1) != 0) {
                    CHK_RET(GenTemplateAlgParamsByDimData(paramsLevel1, sliceInfo.dataSliceLevel1[idx]));
                    oldSubmitSize = tempResMap[OMNIPIPE_LEVEL1].submitInfos.size();
                    CHK_RET(tempMap[OMNIPIPE_LEVEL1]->KernelRun(param, paramsLevel1, tempResMap[OMNIPIPE_LEVEL1]));
                    AppendFastLaunchInfos(tempResMap[OMNIPIPE_LEVEL1], oldSubmitSize, orderedFastLaunchInfos,
                                          FAST_ACTION_LEVEL1);
                }
                CHK_RET(PostSyncInterThreads(controlThread_, tempMainThreadsXY_, ntfIdxTempToCtrlXY_));
                AppendFastLaunchMarker(orderedFastLaunchInfos, FAST_ACTION_POST_SYNC_XY);
            }
            if (tempMap.count(OMNIPIPE_LEVEL2) != 0) {
                CHK_RET(PostSyncInterThreads(controlThread_, tempMainThreadsZ_, ntfIdxTempToCtrlZ_));
                AppendFastLaunchMarker(orderedFastLaunchInfos, FAST_ACTION_POST_SYNC_Z);
            }
        }

        for (u32 rank = 0; rank < rankSize_; ++rank) {
            oldSubmitSize = tempResMap[OMNIPIPE_LEVEL_NUM].submitInfos.size();
            CHK_RET(RunCcuLocalCopy(param, resCtx.cclMem.addr, param.outputPtr, resCtx.cclMem, rank * currBytes,
                                    (rank * dataCount_ + processedDataCount) * dataTypeSize_, currBytes,
                                    tempResMap[OMNIPIPE_LEVEL_NUM]));
            AppendFastLaunchInfos(tempResMap[OMNIPIPE_LEVEL_NUM], oldSubmitSize, orderedFastLaunchInfos,
                                  FAST_ACTION_COPY_OUT);
        }
        processedDataCount += currDataCount;
    }
#ifndef AICPU_COMPILE
    if (loopTimes == 1) {
        CHK_RET(FastLaunchSaveCtx(param, orderedFastLaunchInfos));
    }
#endif
    return HCCL_SUCCESS;
}

template <typename AlgTopoMatch, typename CcuAlgTemplate0, typename CcuAlgTemplate1, typename CcuAlgTemplate2>
HcclResult InsV2AllGatherCcuOmniPipeExecutor<AlgTopoMatch, CcuAlgTemplate0, CcuAlgTemplate1,
                                             CcuAlgTemplate2>::FastLaunch(
    const OpParam &param, const CcuFastLaunchCtx *ctx)
{
    threads_.assign(ctx->GetThreadHandlePtr(), ctx->GetThreadHandlePtr() + ctx->threadNum);
    TemplateFastLaunchCtx tempCtx;
    CcuKernelSubmitInfo *submitInfos = ctx->GetCcuKernelSubmitInfoPtr();
    CcuTempAllGatherOmniPipeLocalCopy copyTemp;
    CcuAlgTemplate2 temp2;
    CcuAlgTemplate0 temp0;
    CcuAlgTemplate1 temp1;
    HcclMem hcclBuff = param.hcclBuff;

    std::map<u32, std::vector<ThreadHandle>> actionThreads;
    bool hasLevel0 = false;
    bool hasLevel1 = false;
    bool hasLevel2 = false;
    for (u32 idx = 0; idx < ctx->ccuKernelNum[FAST_ACTION_SEG]; ++idx) {
        hasLevel0 = hasLevel0 || submitInfos[idx].action == FAST_ACTION_LEVEL0;
        hasLevel1 = hasLevel1 || submitInfos[idx].action == FAST_ACTION_LEVEL1;
        hasLevel2 = hasLevel2 || submitInfos[idx].action == FAST_ACTION_LEVEL2;
    }
    CHK_PRT_RET(threads_.empty(), HCCL_ERROR("[InsV2AllGatherCcuOmniPipeExecutor][FastLaunch] threads is empty."),
                HCCL_E_PARA);
    controlThread_ = threads_[0];
    u32 threadIdx = 1;
    if (hasLevel0) {
        CHK_PRT_RET(threadIdx >= threads_.size(), HCCL_ERROR("[InsV2AllGatherCcuOmniPipeExecutor][FastLaunch] "
                                                             "missing level0 thread."), HCCL_E_PARA);
        actionThreads[FAST_ACTION_LEVEL0] = {threads_[threadIdx++]};
    }
    if (hasLevel1) {
        CHK_PRT_RET(threadIdx >= threads_.size(), HCCL_ERROR("[InsV2AllGatherCcuOmniPipeExecutor][FastLaunch] "
                                                             "missing level1 thread."), HCCL_E_PARA);
        actionThreads[FAST_ACTION_LEVEL1] = {threads_[threadIdx++]};
    }
    if (hasLevel2) {
        CHK_PRT_RET(threadIdx >= threads_.size(), HCCL_ERROR("[InsV2AllGatherCcuOmniPipeExecutor][FastLaunch] "
                                                             "missing level2 thread."), HCCL_E_PARA);
        actionThreads[FAST_ACTION_LEVEL2] = {threads_[threadIdx++]};
    }
    CHK_PRT_RET(threadIdx >= threads_.size(), HCCL_ERROR("[InsV2AllGatherCcuOmniPipeExecutor][FastLaunch] "
                                                         "missing local copy thread."), HCCL_E_PARA);
    actionThreads[FAST_ACTION_COPY_IN] = {threads_[threadIdx]};
    actionThreads[FAST_ACTION_COPY_OUT] = {threads_[threadIdx]};

    tempMainThreadsXY_.clear();
    ntfIdxCtrlToTempXY_.clear();
    ntfIdxTempToCtrlXY_.clear();
    if (hasLevel0) {
        tempMainThreadsXY_.push_back(actionThreads[FAST_ACTION_LEVEL0][0]);
        ntfIdxCtrlToTempXY_.push_back(0);
        ntfIdxTempToCtrlXY_.push_back(ntfIdxTempToCtrlXY_.size());
    }
    if (hasLevel1) {
        tempMainThreadsXY_.push_back(actionThreads[FAST_ACTION_LEVEL1][0]);
        ntfIdxCtrlToTempXY_.push_back(0);
        ntfIdxTempToCtrlXY_.push_back(ntfIdxTempToCtrlXY_.size());
    }
    tempMainThreadsZ_.clear();
    ntfIdxCtrlToTempZ_.clear();
    ntfIdxTempToCtrlZ_.clear();
    if (hasLevel2) {
        tempMainThreadsZ_.push_back(actionThreads[FAST_ACTION_LEVEL2][0]);
        ntfIdxCtrlToTempZ_.push_back(0);
        ntfIdxTempToCtrlZ_.push_back(tempMainThreadsXY_.size());
    }

    for (u32 idx = 0; idx < ctx->ccuKernelNum[FAST_ACTION_SEG]; ++idx) {
        CcuKernelSubmitInfo submitInfo = submitInfos[idx];
        u32 action = submitInfo.action;
        tempCtx.ccuKernelSubmitInfos.assign(1, submitInfo);
        if (action == FAST_ACTION_COPY_IN) {
            tempCtx.threads = actionThreads[FAST_ACTION_COPY_IN];
            CHK_RET(SetTempFastLaunchAddr(tempCtx, param.inputPtr, hcclBuff.addr, hcclBuff));
            tempCtx.buffInfo.inputSize = param.inputSize;
            tempCtx.buffInfo.outputSize = hcclBuff.size;
            tempCtx.buffInfo.hcclBuffSize = hcclBuff.size;
            CHK_RET(copyTemp.FastLaunch(param, tempCtx));
        } else if (action == FAST_ACTION_LEVEL2) {
            tempCtx.threads = actionThreads[FAST_ACTION_LEVEL2];
            CHK_RET(SetTempFastLaunchAddr(tempCtx, hcclBuff.addr, hcclBuff.addr, hcclBuff));
            tempCtx.buffInfo.inputSize = hcclBuff.size;
            tempCtx.buffInfo.outputSize = hcclBuff.size;
            tempCtx.buffInfo.hcclBuffSize = hcclBuff.size;
            CHK_RET(temp2.FastLaunch(param, tempCtx));
        } else if (action == FAST_ACTION_LEVEL0) {
            tempCtx.threads = actionThreads[FAST_ACTION_LEVEL0];
            CHK_RET(SetTempFastLaunchAddr(tempCtx, hcclBuff.addr, hcclBuff.addr, hcclBuff));
            tempCtx.buffInfo.inputSize = hcclBuff.size;
            tempCtx.buffInfo.outputSize = hcclBuff.size;
            tempCtx.buffInfo.hcclBuffSize = hcclBuff.size;
            CHK_RET(temp0.FastLaunch(param, tempCtx));
        } else if (action == FAST_ACTION_LEVEL1) {
            tempCtx.threads = actionThreads[FAST_ACTION_LEVEL1];
            CHK_RET(SetTempFastLaunchAddr(tempCtx, hcclBuff.addr, hcclBuff.addr, hcclBuff));
            tempCtx.buffInfo.inputSize = hcclBuff.size;
            tempCtx.buffInfo.outputSize = hcclBuff.size;
            tempCtx.buffInfo.hcclBuffSize = hcclBuff.size;
            CHK_RET(temp1.FastLaunch(param, tempCtx));
        } else if (action == FAST_ACTION_COPY_OUT) {
            tempCtx.threads = actionThreads[FAST_ACTION_COPY_OUT];
            CHK_RET(SetTempFastLaunchAddr(tempCtx, hcclBuff.addr, param.outputPtr, hcclBuff));
            tempCtx.buffInfo.inputSize = hcclBuff.size;
            tempCtx.buffInfo.outputSize = param.outputSize;
            tempCtx.buffInfo.hcclBuffSize = hcclBuff.size;
            CHK_RET(copyTemp.FastLaunch(param, tempCtx));
        } else if (action == FAST_ACTION_PRE_SYNC_Z) {
            CHK_RET(PreSyncInterThreads(controlThread_, tempMainThreadsZ_, ntfIdxCtrlToTempZ_));
        } else if (action == FAST_ACTION_POST_SYNC_Z) {
            CHK_RET(PostSyncInterThreads(controlThread_, tempMainThreadsZ_, ntfIdxTempToCtrlZ_));
        } else if (action == FAST_ACTION_PRE_SYNC_XY) {
            CHK_RET(PreSyncInterThreads(controlThread_, tempMainThreadsXY_, ntfIdxCtrlToTempXY_));
        } else if (action == FAST_ACTION_POST_SYNC_XY) {
            CHK_RET(PostSyncInterThreads(controlThread_, tempMainThreadsXY_, ntfIdxTempToCtrlXY_));
        } else {
            HCCL_ERROR("[InsV2AllGatherCcuOmniPipeExecutor][FastLaunch] invalid action[%u].", action);
            return HCCL_E_PARA;
        }
    }
    return HCCL_SUCCESS;
}

REGISTER_EXEC_V2_MULTI(HcclCMDType::HCCL_CMD_ALLGATHER, CcuAllGatherOmniPipeMesh1DNHRNHR,
                       InsV2AllGatherCcuOmniPipeExecutor, TopoMatchUBX,
                       CcuTempAllGatherOmniPipeMesh1DMem2Mem,
                       CcuTempAllGatherOmniPipeNHR1DMultiJettyMem2Mem,
                       CcuTempAllGatherOmniPipeNHR1DMultiJettyMem2Mem);

REGISTER_EXEC_V2_MULTI(HcclCMDType::HCCL_CMD_ALLGATHER, CcuAllGatherOmniPipeMesh1DNHR,
                       InsV2AllGatherCcuOmniPipeExecutor, TopoMatchUBX,
                       CcuTempAllGatherOmniPipeMesh1DMem2Mem,
                       CcuTempAllGatherOmniPipeNHR1DMultiJettyMem2Mem,
                       CcuTempAllGatherOmniPipeNHR1DMultiJettyMem2Mem);

} // namespace ops_hccl
#endif
