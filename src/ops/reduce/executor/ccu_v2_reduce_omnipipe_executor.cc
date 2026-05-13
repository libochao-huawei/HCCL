/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
 
#include "ccu_v2_reduce_omnipipe_executor.h"
#include "alg_data_trans_wrapper.h"
#ifndef AICPU_COMPILE
#include "ccu_temp_reduce_scatter_omnipipe_mesh_1D.h"
#include "ccu_temp_gather_omnipipe_mesh_1d.h"
#endif

namespace ops_hccl {

constexpr u32 CCU_OMNIPIPE_LEVEL0 = 0;
constexpr u32 CCU_OMNIPIPE_LEVEL1 = 1;
constexpr u32 CCU_OMNIPIPE_LEVEL_NUM = 2;

template <typename AlgTopoMatch, typename CcuRsAlgTemplateX, typename CcuRsAlgTemplateY, typename CcuAgAlgTemplateX, typename CcuAgAlgTemplateY>
CcuV2ReduceOmniPipeExecutor<AlgTopoMatch, CcuRsAlgTemplateX, CcuRsAlgTemplateY, CcuAgAlgTemplateX, CcuAgAlgTemplateY>::CcuV2ReduceOmniPipeExecutor()
{
}


template <typename AlgTopoMatch, typename CcuRsAlgTemplateX, typename CcuRsAlgTemplateY, typename CcuAgAlgTemplateX, typename CcuAgAlgTemplateY>
HcclResult CcuV2ReduceOmniPipeExecutor<AlgTopoMatch, CcuRsAlgTemplateX, CcuRsAlgTemplateY, CcuAgAlgTemplateX, CcuAgAlgTemplateY>::CalcAlgHierarchyInfo(HcclComm comm, 
            TopoInfoWithNetLayerDetails* topoInfo, AlgHierarchyInfoForAllLevel& algHierarchyInfo)
{
    u32 userrank = topoInfo->userRank;
    HCCL_DEBUG("[%s] myRank[%u]", __func__, userrank);

#if T_DESC("测试拓扑2*2", false)
    if (userrank == 0 || userrank == 1) {
        algHierarchyInfo.infos = {{{0, 1}, {0, 1, 2, 3}}};
    } else {
        algHierarchyInfo.infos = {{{2, 3}, {0, 1, 2, 3}}};
    }
#elif T_DESC("4x2用例", true)
    HCCL_DEBUG("[%s] 4x2-TestCase", __func__);
    if (userrank == 0 || userrank == 1 || userrank == 2 || userrank == 3) {
        algHierarchyInfo.infos = {{{0, 1, 2, 3}, {0, 1, 2, 3, 4, 5, 6, 7}}};
    } else {
        algHierarchyInfo.infos = {{{4, 5, 6, 7}, {0, 1, 2, 3, 4, 5, 6, 7}}};
    }
#elif T_DESC("2x4用例", false)
    HCCL_DEBUG("[%s] 2x4-TestCase", __func__);
    if (userrank == 0 || userrank == 1) {
        algHierarchyInfo.infos = {{{0, 1}, {0, 1, 2, 3, 4, 5, 6, 7}}};
    } else if (userrank == 2 || userrank == 3) {
        algHierarchyInfo.infos = {{{2, 3}, {0, 1, 2, 3, 4, 5, 6, 7}}};
    } else if (userrank == 4 || userrank == 5) {
        algHierarchyInfo.infos = {{{4, 5}, {0, 1, 2, 3, 4, 5, 6, 7}}};
    } else if (userrank == 6 || userrank == 7) {
        algHierarchyInfo.infos = {{{6, 7}, {0, 1, 2, 3, 4, 5, 6, 7}}};
    }
#elif T_DESC("4x3用例", false)
    if (userrank == 0 || userrank == 1 || userrank == 2 || userrank == 3) {
        algHierarchyInfo.infos = {{{0, 1, 2, 3}, {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11}}};
    } else if (userrank == 4 || userrank == 5 || userrank == 6 || userrank == 7) {
        algHierarchyInfo.infos = {{{4, 5, 6, 7}, {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11}}};
    } else if (userrank == 8 || userrank == 9 || userrank == 10 || userrank == 11) {
        algHierarchyInfo.infos = {{{8, 9, 10, 11}, {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11}}};
    }
#elif T_DESC("4x4用例", false)
    if (userrank == 0 || userrank == 1 || userrank == 2 || userrank == 3) {
        algHierarchyInfo.infos = {{{0, 1, 2, 3}, {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15}}};
    } else if (userrank == 4 || userrank == 5 || userrank == 6 || userrank == 7) {
        algHierarchyInfo.infos = {{{4, 5, 6, 7}, {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15}}};
    } else if (userrank == 8 || userrank == 9 || userrank == 10 || userrank == 11) {
        algHierarchyInfo.infos = {{{8, 9, 10, 11}, {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15}}};
    } else if (userrank == 12 || userrank == 13 || userrank == 14 || userrank == 15) {
        algHierarchyInfo.infos = {{{12, 13, 14, 15}, {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15}}};
    }
#else
    AlgTopoMatch topoMatch;
    CHK_RET(topoMatch.MatchTopo(comm, topoInfo, algHierarchyInfo));
#endif

    for (auto i = 0; i < algHierarchyInfo.infos.size(); ++i) {
        for (auto j = 0; j < algHierarchyInfo.infos[i].size(); ++j) {
            for (auto k = 0; k < algHierarchyInfo.infos[i][j].size(); ++k) {
                HCCL_INFO("[%s] myRank[%u] (%d, %d, %d) %u", __func__, topoInfo->userRank, i, j, k,
                    algHierarchyInfo.infos[i][j][k]);
            }
        }
    }

    return HCCL_SUCCESS;
}

template <typename AlgTopoMatch, typename CcuRsAlgTemplateX, typename CcuRsAlgTemplateY, typename CcuAgAlgTemplateX, typename CcuAgAlgTemplateY>
HcclResult CcuV2ReduceOmniPipeExecutor<AlgTopoMatch, CcuRsAlgTemplateX, CcuRsAlgTemplateY, CcuAgAlgTemplateX, CcuAgAlgTemplateY>::InitCommInfo(
            const OpParam& param, const TopoInfoWithNetLayerDetails* topoInfo, const AlgHierarchyInfoForAllLevel& algHierarchyInfo)
{
    myRank_ = topoInfo->userRank;
    rankSize_ = topoInfo->userRankSize;
    devType_ = topoInfo->deviceType;
    reduceOp_ = param.reduceType;
    dataType_ = param.DataDes.dataType;
    dataCount_ = param.DataDes.count;
    dataTypeSize_ =  SIZE_TABLE[param.DataDes.dataType];
    dataSize_ = dataCount_ * dataTypeSize_;
    root_ = param.root;

    rankSizeLevel0_ = algHierarchyInfo.infos[0][0].size();
    rankSizeLevel1_ = algHierarchyInfo.infos[0][1].size() / rankSizeLevel0_;

    rankIdxLevel0_ = myRank_ % rankSizeLevel0_;
    rankIdxLevel1_ = myRank_ / rankSizeLevel0_;

    u64 rootXAixs = root_ % rankSizeLevel0_;
    u64 rootYAixs = (root_ / rankSizeLevel0_) % rankSizeLevel1_;
    u64 rootZAixs = root_ / (rankSizeLevel0_ * rankSizeLevel1_);

    u64 currentRankXAixs = myRank_ % rankSizeLevel0_;
    u64 currentRankYAixs = (myRank_ / rankSizeLevel0_) % rankSizeLevel1_;
    u64 currentRankZAixs = myRank_ / (rankSizeLevel0_ * rankSizeLevel1_);

    isRoot = (myRank_ == param.root);
 	isSameXAxis = (myRankx == rootx) && !isRoot;
    isSameYAxis = (myRanky == rooty) && !isRoot;
    

    HCCL_DEBUG("[%s]myRank[%u] rankSize[%u] rankSizeLevel0[%u] rankSizeLevel1[%u] rankIdxLevel0[%u] "
        "rankIdxLevel1[%u] devType[%u] dataCount[%u] dataType[%u] dataTypeSize[%u]",
        __func__, myRank_, rankSize_, rankSizeLevel0_, rankSizeLevel1_, rankIdxLevel0_, rankIdxLevel1_, devType_,
        dataCount_, dataType_, dataTypeSize_);
    return HcclResult::HCCL_SUCCESS;
}

template <typename AlgTopoMatch, typename CcuRsAlgTemplateX, typename CcuRsAlgTemplateY, typename CcuAgAlgTemplateX, typename CcuAgAlgTemplateY>
HcclResult CcuV2ReduceOmniPipeExecutor<AlgTopoMatch, CcuRsAlgTemplateX, CcuRsAlgTemplateY, CcuAgAlgTemplateX, CcuAgAlgTemplateY>::CalcResLevel(
            HcclComm comm, const OpParam& param, const TopoInfoWithNetLayerDetails* topoInfo,
            std::shared_ptr<CcuAlgTemplateBase> tempAlg, AlgResourceRequest& resourceReq, const int& curLevel)
{
    AlgResourceRequest resReqlevel;
    CHK_RET(tempAlg->CalcRes(comm, param, topoInfo, resReqlevel));

    resourceReq.slaveThreadNum += resReqlevel.slaveThreadNum;
    resourceReq.notifyNumOnMainThread += resReqlevel.notifyNumOnMainThread;
    resourceReq.notifyNumPerThread.insert(resourceReq.notifyNumPerThread.end(),
                                            resReqlevel.notifyNumPerThread.begin(),
                                            resReqlevel.notifyNumPerThread.end());
    HCCL_DEBUG("[%s] currTemplate has [%d] kernels.", __func__, resReqlevel.ccuKernelNum[0]);
    if (curLevel == OMNIPIPE_RS_LEVEL0 || curLevel == OMNIPIPE_RS_LEVEL1) {
        std::for_each(resReqlevel.ccuKernelInfos.begin(), resReqlevel.ccuKernelInfos.end(), [](CcuKernelInfo &info) {
            info.resGroup = 0;
        });
    } else {
        std::for_each(resReqlevel.ccuKernelInfos.begin(), resReqlevel.ccuKernelInfos.end(), [](CcuKernelInfo &info) {
            info.resGroup = 1;
        });
    }
    resourceReq.ccuKernelInfos.insert(resourceReq.ccuKernelInfos.end(), resReqlevel.ccuKernelInfos.begin(), resReqlevel.ccuKernelInfos.end());
    resourceReq.ccuKernelNum.insert(resourceReq.ccuKernelNum.end(), resReqlevel.ccuKernelNum.begin(), resReqlevel.ccuKernelNum.end());

    return HCCL_SUCCESS;
}

template <typename AlgTopoMatch, typename CcuRsAlgTemplateX, typename CcuRsAlgTemplateY, typename CcuAgAlgTemplateX, typename CcuAgAlgTemplateY>
HcclResult CcuV2ReduceOmniPipeExecutor<AlgTopoMatch, CcuRsAlgTemplateX, CcuRsAlgTemplateY, CcuAgAlgTemplateX, CcuAgAlgTemplateY>::InitSubCommRanks(
            std::vector<std::vector<u32>>& subCommRanks0, std::vector<std::vector<u32>>& subCommRanks1, const AlgHierarchyInfoForAllLevel& algHierarchyInfo)
{
    subCommRanks0.clear();
    subCommRanks1.clear();

    subCommRanks0.push_back(algHierarchyInfo.infos[0][0]);
    subCommRanks1.resize(1);
    for (auto i = myRank_ % rankSizeLevel0_; i < algHierarchyInfo.infos[0][1].size(); i += rankSizeLevel0_) {
        subCommRanks1[0].push_back(algHierarchyInfo.infos[0][1][i]);
    }

    return HCCL_SUCCESS;
}
 
template <typename AlgTopoMatch, typename CcuRsAlgTemplateX, typename CcuRsAlgTemplateY, typename CcuAgAlgTemplateX, typename CcuAgAlgTemplateY>
HcclResult CcuV2ReduceOmniPipeExecutor<AlgTopoMatch, CcuRsAlgTemplateX, CcuRsAlgTemplateY, CcuAgAlgTemplateX, CcuAgAlgTemplateY>::CalcRes(
    HcclComm comm, const OpParam& param, const TopoInfoWithNetLayerDetails* topoInfo, const AlgHierarchyInfoForAllLevel& algHierarchyInfo,
    AlgResourceRequest& resourceRequest)
{
    // 初始化一些基本成员变量
    CHK_RET(InitCommInfo(param, topoInfo, algHierarchyInfo));

    // 初始化通信域subCommRanks
    std::vector<std::vector<u32>> subCommRanks0;
    std::vector<std::vector<u32>> subCommRanks1;
    CHK_RET(InitSubCommRanks(subCommRanks0, subCommRanks1, algHierarchyInfo));

    // 初始化template [jjy][todo] 暂时没有考虑level1size=0的情况，后续再考虑
    // [jjy][todo] 这里RS和AG的资源好像没办法复用？因为notifyNumPerThread不一样，看AICPU那边就没复用？
    std::map<u32, std::shared_ptr<CcuAlgTemplateBase>> tempMap;
    tempMap[OMNIPIPE_RS_LEVEL0] = std::make_shared<CcuRsAlgTemplateX>(param, myRank_, subCommRanks0, isRoot, isSameXAxis, isSameYAxis);
    tempMap[OMNIPIPE_RS_LEVEL1] = std::make_shared<CcuRsAlgTemplateY>(param, myRank_, subCommRanks1, isRoot, isSameXAxis, isSameYAxis);
    tempMap[OMNIPIPE_AG_LEVEL0] = std::make_shared<CcuAgAlgTemplateX>(param, myRank_, subCommRanks0, isRoot, isSameXAxis, isSameYAxis);
    tempMap[OMNIPIPE_AG_LEVEL1] = std::make_shared<CcuAgAlgTemplateY>(param, myRank_, subCommRanks1, isRoot, isSameXAxis, isSameYAxis);

    resourceRequest.slaveThreadNum = 0;
    resourceRequest.notifyNumOnMainThread = 0;
    for (int level = 0; level < OMNIPIPE_AR_LEVEL_NUM; level++) {
        if (tempMap.count(level) > 0) {
            CHK_RET(CalcResLevel(comm, param, topoInfo, tempMap[level], resourceRequest, level));
        }
    }
    resourceRequest.slaveThreadNum += 1; // 需要一个主流和一个从流来并行2d
    resourceRequest.notifyNumOnMainThread += 1;
    resourceRequest.notifyNumPerThread.assign(resourceRequest.slaveThreadNum, 1);
    // resourceRequest.notifyNumPerThread.emplace_back(1);
    HCCL_DEBUG("[%s] slaveThreadNum:%d, notifyNumOnMainThread:%d", __func__, resourceRequest.slaveThreadNum, resourceRequest.notifyNumOnMainThread);

    return HCCL_SUCCESS;
}
 
template <typename AlgTopoMatch, typename CcuRsAlgTemplateX, typename CcuRsAlgTemplateY, typename CcuAgAlgTemplateX, typename CcuAgAlgTemplateY>
HcclResult CcuV2ReduceOmniPipeExecutor<AlgTopoMatch, CcuRsAlgTemplateX, CcuRsAlgTemplateY, CcuAgAlgTemplateX, CcuAgAlgTemplateY>::Orchestrate(
    const OpParam& param, const AlgResourceCtxSerializable& resCtx)
{
    HCCL_INFO("[CcuV2ReduceOmniPipeExecutor][Orchestrate] Orchestrate Start");
    // 参数填充
    // maxTmpMemSize_设定为cclIn的大小，op中将申请的cclBuff全给了cclIn
    maxTmpMemSize_ = resCtx.cclMem.size;
    threads_ = resCtx.threads;

    // 算法展开
    HcclResult ret = OrchestrateLoop(param, resCtx);
    CHK_PRT_RET(ret != HCCL_SUCCESS,
        HCCL_ERROR("[CcuV2ReduceOmniPipeExecutor][Orchestrate]errNo[0x%016llx] excutor kernel run failed",
            HCCL_ERROR_CODE(ret)), ret);

    return HCCL_SUCCESS;
}

template <typename AlgTopoMatch, typename CcuRsAlgTemplateX, typename CcuRsAlgTemplateY, typename CcuAgAlgTemplateX, typename CcuAgAlgTemplateY>
HcclResult CcuV2ReduceOmniPipeExecutor<AlgTopoMatch, CcuRsAlgTemplateX, CcuRsAlgTemplateY, CcuAgAlgTemplateX, CcuAgAlgTemplateY>::InitTemplate(
            const OpParam& param, std::map<u32, std::shared_ptr<CcuAlgTemplateBase>>& tempMap,
            const std::vector<std::vector<u32>>& subCommRanks0, const std::vector<std::vector<u32>>& subCommRanks1)
{
    if (rankSizeLevel0_ > 1) {
        tempMap[OMNIPIPE_RS_LEVEL0] = std::make_shared<CcuRsAlgTemplateX>(param, myRank_, subCommRanks0);
        tempMap[OMNIPIPE_AG_LEVEL0] = std::make_shared<CcuAgAlgTemplateX>(param, myRank_, subCommRanks0);
    }
    if (rankSizeLevel1_ > 1) {
        tempMap[OMNIPIPE_RS_LEVEL1] = std::make_shared<CcuRsAlgTemplateY>(param, myRank_, subCommRanks1);
        tempMap[OMNIPIPE_AG_LEVEL1] = std::make_shared<CcuAgAlgTemplateY>(param, myRank_, subCommRanks1);
    }
    HCCL_DEBUG("[CcuV2ReduceOmniPipeExecutor][%s] tempMap.size:%u", __func__, tempMap.size());
    
    HCCL_DEBUG("[CcuV2ReduceOmniPipeExecutor][%s] threads_.size:%d", __func__, threads_.size());
    levelThreads_.resize(CCU_OMNIPIPE_LEVEL_NUM);

    levelThreads_[CCU_OMNIPIPE_LEVEL0].push_back(threads_[0]);
    levelThreads_[CCU_OMNIPIPE_LEVEL1].push_back(threads_[1]);
    return HCCL_SUCCESS;
}

template <typename AlgTopoMatch, typename CcuRsAlgTemplateX, typename CcuRsAlgTemplateY, typename CcuAgAlgTemplateX, typename CcuAgAlgTemplateY>
HcclResult CcuV2ReduceOmniPipeExecutor<AlgTopoMatch, CcuRsAlgTemplateX, CcuRsAlgTemplateY, CcuAgAlgTemplateX, CcuAgAlgTemplateY>::InitTemplateParams(
            const OpParam& param, const AlgResourceCtxSerializable& resCtx,
            const std::map<u32, std::shared_ptr<CcuAlgTemplateBase>>& tempMap,
            std::map<u32, TemplateResource>& tempResMap,
            std::map<u32, TemplateDataParams>& tempAlgParamMap)
{
    // CCU不用在TemplateResource填充channel（填充在kernelInfo中）
    u32 kernelOffset = 0;
    for (int level = 0; level < OMNIPIPE_AR_LEVEL_NUM; level++) {
        if (tempMap.count(level) > 0) {
            if (level < OMNIPIPE_AG_LEVEL0) {
                // [RS-level0, RS-level1]
                // tempResMap[level].threads = levelThreads_[level];
                tempResMap[level].ccuKernels.insert(tempResMap[level].ccuKernels.end(),
                    resCtx.ccuKernels.begin() + kernelOffset,
                    resCtx.ccuKernels.begin() + kernelOffset + resCtx.ccuKernelNum[level]); 
            } else {
                // [AG-level0, AG-level1]
                // tempResMap[level].threads = levelThreads_[level - OMNIPIPE_AG_LEVEL0];
                tempResMap[level].ccuKernels.insert(tempResMap[level].ccuKernels.end(),
                    resCtx.ccuKernels.begin() + kernelOffset,
                    resCtx.ccuKernels.begin() + kernelOffset + resCtx.ccuKernelNum[1]);
            }
            kernelOffset += resCtx.ccuKernelNum[level];

            tempAlgParamMap[level].buffInfo.inputPtr = param.inputPtr;
            tempAlgParamMap[level].buffInfo.outputPtr = param.outputPtr;
            tempAlgParamMap[level].buffInfo.inputSize = param.inputSize;
            tempAlgParamMap[level].buffInfo.outputSize = param.outputSize;
            tempAlgParamMap[level].buffInfo.hcclBuff = resCtx.cclMem;
            tempAlgParamMap[level].inputSliceStride = dataSize_;
            tempAlgParamMap[level].outputSliceStride = dataSize_;
            tempAlgParamMap[level].localCopyFlag = 0;
            tempAlgParamMap[level].root = param.root;
        }
    }
    return HCCL_SUCCESS;
}

template <typename AlgTopoMatch, typename CcuRsAlgTemplateX, typename CcuRsAlgTemplateY, typename CcuAgAlgTemplateX, typename CcuAgAlgTemplateY>
HcclResult CcuV2ReduceOmniPipeExecutor<AlgTopoMatch, CcuRsAlgTemplateX, CcuRsAlgTemplateY, CcuAgAlgTemplateX, CcuAgAlgTemplateY>::InitOmniPipeScratchParam(
            OmniPipeScratchParam& scratchParam, const OpParam& param,
            const std::vector<double>& endpointAttrBwAvg,
            std::map<u32, std::shared_ptr<CcuAlgTemplateBase>>& tempMap)
{
    //scratchParam.dataSizePerLoop\ scratchParam.dataWholeSize 在外部赋值
    scratchParam.levelRankSize = {rankSizeLevel0_, rankSizeLevel1_, 1};
    scratchParam.endpointAttrBw = endpointAttrBwAvg;
    scratchParam.levelAlgType = {1, 1, 1}; // [jjy][todo]rs说后面再修改？
    scratchParam.dataSize = dataSize_;
    scratchParam.dataTypeSize = dataTypeSize_;
    scratchParam.maxTmpMemSize = 200 * 1024 * 1024;
    scratchParam.opMode = param.opMode;
    scratchParam.engine = param.engine;
    return HCCL_SUCCESS;
}

template <typename AlgTopoMatch, typename CcuRsAlgTemplateX, typename CcuRsAlgTemplateY, typename CcuAgAlgTemplateX, typename CcuAgAlgTemplateY>
HcclResult CcuV2ReduceOmniPipeExecutor<AlgTopoMatch, CcuRsAlgTemplateX, CcuRsAlgTemplateY, CcuAgAlgTemplateX, CcuAgAlgTemplateY>::InitOmniPipeSliceParam(
            OmniPipeSliceParam& sliceParam, const OpParam& param,
            const std::vector<double>& endpointAttrBwAvg,
            std::map<u32, std::shared_ptr<CcuAlgTemplateBase>>& tempMap)
{
    //sliceParam.dataSizePerLoop\ sliceParam.dataWholeSize 在外部赋值
    sliceParam.endpointAttrBw = endpointAttrBwAvg;
    sliceParam.levelRankSize = {rankSizeLevel0_, rankSizeLevel1_, 1};
    sliceParam.levelRankId = {rankIdxLevel0_, rankIdxLevel1_, 0};
    sliceParam.levelAlgType = {1, 1, 1}; // [jjy][todo]rs说后面再修改？
    sliceParam.dataTypeSize = dataTypeSize_;
    sliceParam.opMode = param.opMode;
    sliceParam.engine = param.engine;
    return HCCL_SUCCESS;
}

// 将计算出的单步slice信息初始化到templateParam中
template <typename AlgTopoMatch, typename CcuRsAlgTemplateX, typename CcuRsAlgTemplateY, typename CcuAgAlgTemplateX, typename CcuAgAlgTemplateY>
HcclResult CcuV2ReduceOmniPipeExecutor<AlgTopoMatch, CcuRsAlgTemplateX, CcuRsAlgTemplateY, CcuAgAlgTemplateX, CcuAgAlgTemplateY>::GenTemplateAlgParamsByDimData(
            TemplateDataParams &tempAlgParams, StepSliceInfo &stepSliceInfo, u64 processedDataCount)
{
    tempAlgParams.count = 0;

    tempAlgParams.stepSliceInfo = stepSliceInfo;
    tempAlgParams.stepSliceInfo.buffInfo.inBuffBaseOff
        = stepSliceInfo.buffInfo.inBuffBaseOff + processedDataCount * dataTypeSize_;
    tempAlgParams.stepSliceInfo.buffInfo.outBuffBaseOff
        = stepSliceInfo.buffInfo.outBuffBaseOff + processedDataCount * dataTypeSize_;

    HCCL_DEBUG("[%s]myRank[%u] inBuffBaseOff[%llu] processedDataCount[%llu] end inBuffBaseOff[%llu]", __func__,
        myRank_, stepSliceInfo.buffInfo.inBuffBaseOff, processedDataCount, tempAlgParams.stepSliceInfo.buffInfo.inBuffBaseOff);

    HCCL_DEBUG("[%s]myRank[%u] outBuffBaseOff[%llu] processedDataCount[%llu] end outBuffBaseOff[%llu]", __func__,
        myRank_, stepSliceInfo.buffInfo.outBuffBaseOff, processedDataCount, tempAlgParams.stepSliceInfo.buffInfo.outBuffBaseOff);

    tempAlgParams.inputSliceStride = 0;
    tempAlgParams.outputSliceStride = 0;
    tempAlgParams.sliceSize = 0;
    tempAlgParams.localCopyFlag = 0;
    return HcclResult::HCCL_SUCCESS;
}
 
template <typename AlgTopoMatch, typename CcuRsAlgTemplateX, typename CcuRsAlgTemplateY, typename CcuAgAlgTemplateX, typename CcuAgAlgTemplateY>
HcclResult CcuV2ReduceOmniPipeExecutor<AlgTopoMatch, CcuRsAlgTemplateX, CcuRsAlgTemplateY, CcuAgAlgTemplateX, CcuAgAlgTemplateY>::OrchestrateLoop(
            const OpParam& param, const AlgResourceCtxSerializable& resCtx)
{
    HCCL_INFO("[%s] Start", __func__);

    // 初始化通信域subCommRanks
    std::vector<std::vector<u32>> subCommRanks0;
    std::vector<std::vector<u32>> subCommRanks1;
    auto& algHierarchyInfo_local = const_cast<ops_hccl::AlgHierarchyInfoForAllLevel&>(resCtx.algHierarchyInfo);
    CHK_RET(InitSubCommRanks(subCommRanks0, subCommRanks1, algHierarchyInfo_local));

    // 初始化template
    std::map<u32, std::shared_ptr<CcuAlgTemplateBase>> tempMap;
    CHK_RET(InitTemplate(param, tempMap, subCommRanks0, subCommRanks1));

    // 初始化资源TemplateResource\TemplateDataParams
    std::map<u32, TemplateResource> tempResMap;
    std::map<u32, TemplateDataParams> tempAlgParamMap;
    CHK_RET(InitTemplateParams(param, resCtx, tempMap, tempResMap, tempAlgParamMap));

    // 1、计算带宽 平均带宽还是总带宽,如果是总带宽这边要处理成平均带宽 // [jjy][todo]计算带宽打桩
    std::vector<std::vector<double>> endpointAttrBw;
    std::vector<double> endpointAttrBwAvg;
#if T_DESC("计算带宽实现", false)
    CHK_RET(CalAllLevelEndpointAttrBwCoeff(param.hcclComm, myRank_, 3, endpointAttrBw));
    // 需要转化成平均带宽
    u64 bwIndex = 0;
    for (u64 i = 0; i < endpointAttrBw.size(); i++) {
        for (u64 j = 0; j < endpointAttrBw[i].size(); ++j) {
            endpointAttrBw[i][j] /= algHierarchyInfo.infos[i][j].size() - 1;
            endpointAttrBwAvg[bwIndex++] = endpointAttrBw[i][j];
        }
    }
#else
    // endpointAttrBwAvg = {1,1,1};
    endpointAttrBwAvg = {3,4,1};
#endif

    // 2.1 获取每个rank切分的数据量count
    auto allRankSplitData = OmniPipeSplitData(rankSize_, dataCount_, dataTypeSize_);
    for (int i=0;i<allRankSplitData.size();i++){
        HCCL_INFO("[jjy]rankId[%d], allRankSplitData[%d]:%d", myRank_, i, allRankSplitData[i]);
    }

    // 2.2 计算loop次数
#if 0
    OmniPipeScratchParam scratchParam;
    CHK_RET(InitOmniPipeScratchParam(scratchParam, param, endpointAttrBwAvg, tempMap));
    scratchParam.maxTmpMemSize = resCtx.cclMem.size;
    // 将数据量切分count转化为dataSize，传给scratchParam
    scratchParam.dataSize = CalcCountToDataSize(allRankSplitData, dataTypeSize_);
    std::vector<u64> loopInfo = CalcOmniPipeScratchInfo(scratchParam); // [jjy][todo]待考虑是否要这样计算？
    // 中转内存(/UB Bound)单次最多能够接受的output count，注意是count不是size
    u64 maxCountPerLoop = loopInfo[0];
    u64 loopTimes = loopInfo[1];
    HCCL_DEBUG("[%s]maxCountPerLoop[%u], loopTimes[%u]", __func__, maxCountPerLoop, loopTimes);
#else
    // u64 maxCountPerLoop = static_cast<u64>(UB_MAX_DATA_SIZE) / dataTypeSize_;
    u64 maxCountPerLoop = static_cast<u64>(256) / dataTypeSize_;
    u32 loopTimes = allRankSplitData[0] / maxCountPerLoop + ((allRankSplitData[0] % maxCountPerLoop == 0) ? 0 : 1);
    HCCL_DEBUG("[%s] myRank[%u] loopTimes[%llu]", __func__, myRank_, loopTimes);
#endif

    // 2.3 获取每个rank，每个loop切分的数据量count
    auto multiLoopAllRankSplitData =
        OmniPipeSplitRankDataLoop(allRankSplitData, maxCountPerLoop, loopTimes, dataTypeSize_);
    for (int i=0;i<multiLoopAllRankSplitData.size();i++){
        for(int j=0;j<multiLoopAllRankSplitData[i].size();j++){
            HCCL_INFO("[jjy]rankId[%d], multiLoopAllRankSplitData[%d][%d]:%d", myRank_, i, j, multiLoopAllRankSplitData[i][j]);
        }
    }

    // 3.1 计算n-1次loop的slice信息
    u64 perLoopSize = multiLoopAllRankSplitData[0][0] * dataTypeSize_;
    perLoopSize = dataSize_ > perLoopSize ? perLoopSize : dataSize_;
    HCCL_DEBUG("[%s][jjy] perLoopSize[%u]", __func__, perLoopSize);
    std::vector<u64> dataSizePerLoop(rankSize_, perLoopSize);
    // std::vector<u64> dataWholeSize(rankSize_, perLoopSize);
    std::vector<u64> dataWholeSize(rankSize_, allRankSplitData[myRank_] * dataTypeSize_);
    OmniPipeSliceParam sliceParam;
    sliceParam.dataSizePerLoop = dataSizePerLoop;
    sliceParam.dataWholeSize = dataWholeSize;
    // sliceParam.endpointAttrBw =  {1, 1, 1};
    sliceParam.endpointAttrBw =  {3, 4, 1};
    sliceParam.levelRankId = {rankIdxLevel0_, rankIdxLevel1_, 0};
    sliceParam.levelRankSize = {rankSizeLevel0_, rankSizeLevel1_, 1};
    std::vector<u64> levelAlgType{1, 1, 1};
    sliceParam.levelAlgType = levelAlgType;
    sliceParam.dataTypeSize = dataTypeSize_;
    sliceParam.opMode = param.opMode;
    sliceParam.engine = CommEngine::COMM_ENGINE_CCU;
    OmniPipeSliceInfo alignSliceInfoRS = CalcRSOmniPipeSliceInfo(sliceParam);
    OmniPipeSliceInfo alignSliceInfoAG = CalcAGOmniPipeSliceInfo(sliceParam);
    OmniPipeSliceInfo alignSliceInfoG = CalcGatherOmniPipeSliceInfo(sliceParam, root_);
    HCCL_INFO("[jjy]alignSliceInfoRS.dataSliceLevel0.size()=%d", alignSliceInfoRS.dataSliceLevel0.size());

    // 3.2 计算第n次的loop的slice信息
    OmniPipeSliceInfo tailSliceInfoRS;
    OmniPipeSliceInfo tailSliceInfoAG;
    if (allRankSplitData[0] % maxCountPerLoop != 0) {
        HCCL_INFO("jjy0000");
        sliceParam.dataSizePerLoop = CalcCountToDataSize(multiLoopAllRankSplitData[loopTimes - 1], dataTypeSize_);
        sliceParam.dataWholeSize = sliceParam.dataSizePerLoop;
        tailSliceInfoRS = CalcRSOmniPipeSliceInfo(sliceParam);
        tailSliceInfoAG = CalcAGOmniPipeSliceInfo(sliceParam);
        tailSliceInfoG = CalcGatherOmniPipeSliceInfo(sliceParam, root_);
    }

    for(int i=0;i<alignSliceInfoRS.dataSliceLevel0.size();i++){
        for(int j=0;j<alignSliceInfoRS.dataSliceLevel0[i].inputOmniPipeSliceStride.size();j++){
            for(int k=0;k<alignSliceInfoRS.dataSliceLevel0[i].inputOmniPipeSliceStride[j].size();k++){
                HCCL_INFO("[jjy][dataSliceLevel0]myRank[%d][inputOmniPipeSliceStride][%d][%d][%d]:%d",myRank_,i,j,k,alignSliceInfoRS.dataSliceLevel0[i].inputOmniPipeSliceStride[j][k]);
            }
        } 
    }
    for(int i=0;i<alignSliceInfoRS.dataSliceLevel1.size();i++){
        for(int j=0;j<alignSliceInfoRS.dataSliceLevel1[i].inputOmniPipeSliceStride.size();j++){
            for(int k=0;k<alignSliceInfoRS.dataSliceLevel1[i].inputOmniPipeSliceStride[j].size();k++){
                HCCL_INFO("[jjy][dataSliceLevel1]myRank[%d][inputOmniPipeSliceStride][%d][%d][%d]:%d",myRank_,i,j,k,alignSliceInfoRS.dataSliceLevel1[i].inputOmniPipeSliceStride[j][k]);
            }
        } 
    }
    

    // 4 进行一次loop的数据处理
    u64 processedDataCount = 0;
    OmniPipeSliceInfo omniPipeSliceInfoRS;
    OmniPipeSliceInfo omniPipeSliceInfoAG;
    for (u64 loop = 0; loop < loopTimes; loop++) {//loopTimes
        CHK_PRT_RET(
            multiLoopAllRankSplitData.size() <= loop,
            HCCL_ERROR("[CcuV2ReduceOmniPipeExecutor][Orchestrate] multiLoopAllRankSplitData.size() <= loop"),
            HCCL_E_PARA);

        // 4.1 确定当前是前n-1次loop的slice结果，还是存在尾块时最后一次loop的slice结果
        if (loop == loopTimes - 1 && !tailSliceInfoRS.isEmpty()) {
            HCCL_INFO("jjy1");
            omniPipeSliceInfoRS = tailSliceInfoRS;
            omniPipeSliceInfoAG = tailSliceInfoAG;
        } else {
            HCCL_INFO("jjy2");
            omniPipeSliceInfoRS = alignSliceInfoRS;
            omniPipeSliceInfoAG = alignSliceInfoAG;
        }

        u64 currDataCount = multiLoopAllRankSplitData[loop][myRank_];       
        HCCL_DEBUG("[%s] dataCount_ %llu, processedDataCount %llu, maxCountPerLoop %llu, currDataCount %llu",
                        __func__, dataCount_, processedDataCount, maxCountPerLoop, currDataCount);

        // 4.2 RS的通信步数
        auto level0StepCountRS = omniPipeSliceInfoRS.dataSliceLevel0.size();
        HCCL_DEBUG("[%s] myRank[%u] level0StepCountRS[%u]", __func__, myRank_, level0StepCountRS);

        if (omniPipeSliceInfoRS.isEmpty()) {
            HCCL_DEBUG("[%s] myRank[%u] omniPipeSliceInfo is Empty!", __func__, myRank_);
        } else {
            auto l0StepNum = omniPipeSliceInfoRS.dataSliceLevel0;
            auto l1StepNum = omniPipeSliceInfoRS.dataSliceLevel1;
            HCCL_DEBUG("[%s] myRank[%u] L0 stepNum[%u]", __func__, myRank_, l0StepNum.size());
            HCCL_DEBUG("[%s] myRank[%u] L1 stepNum[%u]", __func__, myRank_, l1StepNum.size());
        }


        // 4.3 RS for内层2d
        // template间同步所需信息计算
        ThreadHandle mainThread = threads_[0];
        std::vector<ThreadHandle> syncThreads{threads_[1]};
        std::vector<u32> notifyIdxesMainToSub{0};
        std::vector<u32> notifyIdxesSubToMain{0};
        for (auto i = 0; i < level0StepCountRS; ++i) {
            // 第一步开始前同步
            CHK_RET(PreSyncInterThreads(mainThread, syncThreads, notifyIdxesMainToSub));
            
            // level0
            GenTemplateAlgParamsByDimData(tempAlgParamMap[OMNIPIPE_RS_LEVEL0], omniPipeSliceInfoRS.dataSliceLevel0[i], processedDataCount);
            tempResMap[OMNIPIPE_RS_LEVEL0].threads.clear();
            tempResMap[OMNIPIPE_RS_LEVEL0].threads.emplace_back(threads_[0]);
            CHK_RET(tempMap[OMNIPIPE_RS_LEVEL0]->KernelRun(param, tempAlgParamMap[OMNIPIPE_RS_LEVEL0], tempResMap[OMNIPIPE_RS_LEVEL0]));
            
            // level1
            GenTemplateAlgParamsByDimData(tempAlgParamMap[OMNIPIPE_RS_LEVEL1], omniPipeSliceInfoRS.dataSliceLevel1[i], processedDataCount);
            tempResMap[OMNIPIPE_RS_LEVEL1].threads.clear();
            tempResMap[OMNIPIPE_RS_LEVEL1].threads.emplace_back(threads_[1]);
            CHK_RET(tempMap[OMNIPIPE_RS_LEVEL1]->KernelRun(param, tempAlgParamMap[OMNIPIPE_RS_LEVEL1], tempResMap[OMNIPIPE_RS_LEVEL1]));

            // 第一步做完后回到主流做尾同步
            CHK_RET(PostSyncInterThreads(mainThread, syncThreads, notifyIdxesSubToMain));
        }

        // 4.4 AG本地拷贝
        HCCL_DEBUG("[%s] AG local copy start, myRank[%d], currDataCount %llu, processedDataCount %llu",
                        __func__, myRank_, currDataCount, processedDataCount);
        CHK_RET(PreSyncInterThreads(mainThread, syncThreads, notifyIdxesMainToSub));
        // 本地拷贝
        TemplateDataParams tempAlgParamLocalCopy;
        tempAlgParamLocalCopy.buffInfo.inputPtr = param.inputPtr;
        tempAlgParamLocalCopy.buffInfo.outputPtr = param.outputPtr;
        tempAlgParamLocalCopy.buffInfo.inputSize = param.inputSize;
        tempAlgParamLocalCopy.buffInfo.outputSize = param.outputSize;
        tempAlgParamLocalCopy.buffInfo.hcclBuff = resCtx.cclMem;
        tempAlgParamLocalCopy.buffInfo.inBuffType = BufferType::INPUT;
        tempAlgParamLocalCopy.count = currDataCount;
        tempAlgParamLocalCopy.stepSliceInfo.buffInfo.inBuffBaseOff = myRank_ * allRankSplitData[myRank_] * dataTypeSize_ + processedDataCount * dataTypeSize_;
        tempAlgParamLocalCopy.stepSliceInfo.buffInfo.outBuffBaseOff = processedDataCount * dataTypeSize_;
        tempAlgParamLocalCopy.inputSliceStride = allRankSplitData[myRank_] * dataTypeSize_;
        tempAlgParamLocalCopy.outputSliceStride = allRankSplitData[myRank_] * dataTypeSize_;
        tempAlgParamLocalCopy.repeatNum = rankSize_;
        tempAlgParamLocalCopy.sliceSize = currDataCount * dataTypeSize_;
        tempAlgParamLocalCopy.localCopyFlag = 1;

        tempResMap[OMNIPIPE_AG_LEVEL0].threads.clear();
        tempResMap[OMNIPIPE_AG_LEVEL0].threads.emplace_back(threads_[1]);
        CHK_RET(tempMap[OMNIPIPE_AG_LEVEL0]->KernelRun(param, tempAlgParamLocalCopy, tempResMap[OMNIPIPE_AG_LEVEL0]));
        CHK_RET(PostSyncInterThreads(mainThread, syncThreads, notifyIdxesSubToMain));
        HCCL_DEBUG("[%s] AG local copy end", __func__);

        // 4.5 AG for内层2d
        u32 level0StepCountAG = omniPipeSliceInfoAG.dataSliceLevel0.size();
        HCCL_DEBUG("[%s] level0StepCountAG %u", __func__, level0StepCountAG);
        for (u32 i = 0; i < level0StepCountAG; i++) {
            // 初始化机内template param
            GenTemplateAlgParamsByDimData(tempAlgParamMap[OMNIPIPE_AG_LEVEL0], omniPipeSliceInfoAG.dataSliceLevel0[i], processedDataCount);
            GenTemplateAlgParamsByDimData(tempAlgParamMap[OMNIPIPE_AG_LEVEL1], omniPipeSliceInfoAG.dataSliceLevel1[i], processedDataCount);

            //第一步开始前同步
            CHK_RET(PreSyncInterThreads(mainThread, syncThreads, notifyIdxesMainToSub));

            // 执行机内template任务
            // level0
            tempResMap[OMNIPIPE_AG_LEVEL0].threads.clear();
            tempResMap[OMNIPIPE_AG_LEVEL0].threads.emplace_back(threads_[1]);
            CHK_RET(tempMap[OMNIPIPE_AG_LEVEL0]->KernelRun(param, tempAlgParamMap[OMNIPIPE_AG_LEVEL0], tempResMap[OMNIPIPE_AG_LEVEL0]));
            // level1
            tempResMap[OMNIPIPE_AG_LEVEL1].threads.clear();
            tempResMap[OMNIPIPE_AG_LEVEL1].threads.emplace_back(threads_[0]);
            CHK_RET(tempMap[OMNIPIPE_AG_LEVEL1]->KernelRun(param, tempAlgParamMap[OMNIPIPE_AG_LEVEL1], tempResMap[OMNIPIPE_AG_LEVEL1]));

            //第一步做完后回到主流做尾同步
            CHK_RET(PostSyncInterThreads(mainThread, syncThreads, notifyIdxesSubToMain));
        }
        

        processedDataCount += currDataCount;
    }

    HCCL_INFO("[%s][OrchestrateLoop] End.", __func__);
    return HCCL_SUCCESS;
}

#ifndef AICPU_COMPILE // [jjy][todo] 这里的TopoMatchUBX还是TopoMatchMultilevel？这里需要写ifndef吗？
REGISTER_EXEC_V2_MULTI(HcclCMDType::HCCL_CMD_Reduce, CcuV2ReduceOmniPipe2D,
                                CcuV2ReduceOmniPipeExecutor, TopoMatchUBX, 
                                CcuTempReduceScatterOmniPipeMesh1D, CcuTempReduceScatterOmniPipeMesh1D, 
                                CcuTempGatherOmniPipeMesh1D, CcuTempGatherOmniPipeMesh1D);
#endif
}