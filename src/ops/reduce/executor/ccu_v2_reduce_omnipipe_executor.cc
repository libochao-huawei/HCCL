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
#include "ccu_temp_reduce_scatter_omnipipe_mesh1d_mem2mem.h"
#include "ccu_temp_gather_omnipipe_mesh_1d_mem2mem.h"
#include "ccu_temp_gather_omnipipe_mesh_1d_mem2memY.h"
// #include "ccu_temp_gather_omnipipe_nhr_1d_mem2mem.h"
#endif
#include "ccu_alg_template_base.h"
namespace ops_hccl {

constexpr u32 CCU_OMNIPIPE_LEVEL0 = 0;
constexpr u32 CCU_OMNIPIPE_LEVEL1 = 1;
constexpr u32 CCU_OMNIPIPE_LEVEL_NUM = 2;

template <typename AlgTopoMatch, typename CcuRsAlgTemplateX, typename CcuRsAlgTemplateY, typename CcuGAlgTemplateX, typename CcuGAlgTemplateY>
CcuV2ReduceOmniPipeExecutor<AlgTopoMatch, CcuRsAlgTemplateX, CcuRsAlgTemplateY, CcuGAlgTemplateX, CcuGAlgTemplateY>::CcuV2ReduceOmniPipeExecutor()
{
}


template <typename AlgTopoMatch, typename CcuRsAlgTemplateX, typename CcuRsAlgTemplateY, typename CcuGAlgTemplateX, typename CcuGAlgTemplateY>
HcclResult CcuV2ReduceOmniPipeExecutor<AlgTopoMatch, CcuRsAlgTemplateX, CcuRsAlgTemplateY, CcuGAlgTemplateX, CcuGAlgTemplateY>::CalcAlgHierarchyInfo(HcclComm comm, 
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
#elif T_DESC("4x2用例", false)
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
#endif
    AlgTopoMatch topoMatch;
    CHK_RET(topoMatch.MatchTopo(comm, topoInfo, algHierarchyInfo));

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

// 获取同跟root同X轴的节点（获取当前子通信域里的root）
template <typename AlgTopoMatch, typename CcuRsAlgTemplateX, typename CcuRsAlgTemplateY, typename CcuGAlgTemplateX, typename CcuGAlgTemplateY>
u64 CcuV2ReduceOmniPipeExecutor<AlgTopoMatch, CcuRsAlgTemplateX, CcuRsAlgTemplateY, CcuGAlgTemplateX, CcuGAlgTemplateY>::GetXRoot()
{
    HCCL_INFO("[CcuTempGatherOmniPipeMesh1DMem2Mem][GetRoot] myRank_ [%u] ", myRank_);
    u64 res = myRank_;
    std::vector<u32> ranks = subCommRanks0[0];
    std::string ranksStr = "";
    for (auto r : ranks) { ranksStr += std::to_string(r) + " "; }
    for (auto r : ranks) { 
        u64 subX = r % rankSizeLevel0_;
        if (subX == rootXAixs) {
            res = r;
        }
    }
    // return myRank_;

    auto itRoot = std::find(ranks.begin(), ranks.end(), res);
    if (itRoot != ranks.end()) {
        return  std::distance(ranks.begin(), itRoot);
    }
    return myRank_;
}

// 获取同跟root同Y轴的节点（获取当前子通信域里的root）
template <typename AlgTopoMatch, typename CcuRsAlgTemplateX, typename CcuRsAlgTemplateY, typename CcuGAlgTemplateX, typename CcuGAlgTemplateY>
u64 CcuV2ReduceOmniPipeExecutor<AlgTopoMatch, CcuRsAlgTemplateX, CcuRsAlgTemplateY, CcuGAlgTemplateX, CcuGAlgTemplateY>::GetYRoot()
{
    HCCL_INFO("[CcuTempGatherOmniPipeMesh1DMem2Mem][GetRoot] myRank_ [%u] ", myRank_);
    u64 res = myRank_;
    std::vector<u32> ranks = subCommRanks1[0];
    std::string ranksStr = "";
    for (auto r : ranks) { ranksStr += std::to_string(r) + " "; }
    for (auto r : ranks) { 
        u64 subY = r / rankSizeLevel0_;
        if (subY == rootYAixs) {
            res = r;
        }
    }
    // return res;


    auto itRoot = std::find(ranks.begin(), ranks.end(), res);
    if (itRoot != ranks.end()) {
        return  std::distance(ranks.begin(), itRoot);
    }
    return myRank_;
}
template <typename AlgTopoMatch, typename CcuRsAlgTemplateX, typename CcuRsAlgTemplateY, typename CcuGAlgTemplateX, typename CcuGAlgTemplateY>
HcclResult CcuV2ReduceOmniPipeExecutor<AlgTopoMatch, CcuRsAlgTemplateX, CcuRsAlgTemplateY, CcuGAlgTemplateX, CcuGAlgTemplateY>::InitCommInfo(
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

    rootXAixs = root_ % rankSizeLevel0_;
    rootYAixs = root_ / rankSizeLevel0_;

    isRoot = (myRank_ == root_);
 	// isSameXAxis = (rankIdxLevel0_ == rootXAixs) && !isRoot;
    // isSameYAxis = (rankIdxLevel1_ == rootYAixs) && !isRoot;
    isSameXAxis = (rankIdxLevel0_ == rootXAixs && !isRoot); // 同x，走NHR
    isSameYAxis = (rankIdxLevel1_ == rootYAixs && !isRoot); // 同y，走mesh
    

    HCCL_DEBUG("[%s]myRank[%u] rankSize[%u] rankSizeLevel0[%u] rankSizeLevel1[%u] rankIdxLevel0[%u] "
        "rankIdxLevel1[%u] devType[%u] dataCount[%u] dataType[%u] dataTypeSize[%u]",
        __func__, myRank_, rankSize_, rankSizeLevel0_, rankSizeLevel1_, rankIdxLevel0_, rankIdxLevel1_, devType_,
        dataCount_, dataType_, dataTypeSize_);
    return HcclResult::HCCL_SUCCESS;
}

template <typename AlgTopoMatch, typename CcuRsAlgTemplateX, typename CcuRsAlgTemplateY, typename CcuGAlgTemplateX, typename CcuGAlgTemplateY>
HcclResult CcuV2ReduceOmniPipeExecutor<AlgTopoMatch, CcuRsAlgTemplateX, CcuRsAlgTemplateY, CcuGAlgTemplateX, CcuGAlgTemplateY>::CalcResLevel(
            HcclComm comm, const OpParam& param, const TopoInfoWithNetLayerDetails* topoInfo,
            AlgResourceRequest& resReqlevel, AlgResourceRequest& resourceReq, const int& curLevel)
{
    // AlgResourceRequest resReqlevel;
    // CHK_RET(tempAlg->CalcRes(comm, param, topoInfo, resReqlevel)); //每个template自己资源的计算，结果算入resReqlevel

    resourceReq.slaveThreadNum += resReqlevel.slaveThreadNum; // 从流数 一般是0
    resourceReq.notifyNumOnMainThread += resReqlevel.notifyNumOnMainThread; // 一般是0
    resourceReq.notifyNumPerThread.insert(resourceReq.notifyNumPerThread.end(), // 一般是0
                                            resReqlevel.notifyNumPerThread.begin(),
                                            resReqlevel.notifyNumPerThread.end());
    
    // resourceReq.channels.push_back(resReqlevel.channels[0]); // [jjy][todo] 不是说executor中都不用放channel了吗？为什么RS和AG都写了这个？暂时写上，后面在考虑
    //资源组的值一样就一起申请，资源组的值不一样就串行申请，前一个销毁后后一个申请
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
    HCCL_INFO("[%s] 190190.", __func__);
    resourceReq.ccuKernelInfos.insert(resourceReq.ccuKernelInfos.end(), resReqlevel.ccuKernelInfos.begin(), resReqlevel.ccuKernelInfos.end()); //不需要改，无脑放
    resourceReq.ccuKernelNum.insert(resourceReq.ccuKernelNum.end(), resReqlevel.ccuKernelNum.begin(), resReqlevel.ccuKernelNum.end());

    return HCCL_SUCCESS;
}

template <typename AlgTopoMatch, typename CcuRsAlgTemplateX, typename CcuRsAlgTemplateY, typename CcuGAlgTemplateX, typename CcuGAlgTemplateY>
HcclResult CcuV2ReduceOmniPipeExecutor<AlgTopoMatch, CcuRsAlgTemplateX, CcuRsAlgTemplateY, CcuGAlgTemplateX, CcuGAlgTemplateY>::InitSubCommRanks(
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



 
template <typename AlgTopoMatch, typename CcuRsAlgTemplateX, typename CcuRsAlgTemplateY, typename CcuGAlgTemplateX, typename CcuGAlgTemplateY>
HcclResult CcuV2ReduceOmniPipeExecutor<AlgTopoMatch, CcuRsAlgTemplateX, CcuRsAlgTemplateY, CcuGAlgTemplateX, CcuGAlgTemplateY>::CalcRes(
    HcclComm comm, const OpParam& param, const TopoInfoWithNetLayerDetails* topoInfo, const AlgHierarchyInfoForAllLevel& algHierarchyInfo,
    AlgResourceRequest& resourceRequest)
{
    // 初始化一些基本成员变量
    CHK_RET(InitCommInfo(param, topoInfo, algHierarchyInfo));

    // 初始化通信域subCommRanks
    // std::vector<std::vector<u32>> subCommRanks0;
    // std::vector<std::vector<u32>> subCommRanks1;
    CHK_RET(InitSubCommRanks(subCommRanks0, subCommRanks1, algHierarchyInfo));


    CcuRsAlgTemplateX rsAlgTempLevelX(param, myRank_, subCommRanks0);
	CcuRsAlgTemplateY rsAlgTempLevelY(param, myRank_, subCommRanks1);
    CcuGAlgTemplateX gAlgTempLevelX(param, myRank_, subCommRanks0);
	CcuGAlgTemplateY gAlgTempLevelY(param, myRank_, subCommRanks1);


    // 初始化template [jjy][todo] 暂时没有考虑level1size=0的情况，后续再考虑
    // [jjy][todo] 这里RS和AG的资源好像没办法复用？因为notifyNumPerThread不一样，看AICPU那边就没复用？
    // std::map<u32, std::shared_ptr<CcuAlgTemplateBase>> tempMap; 

    // tempMap[OMNIPIPE_RS_LEVEL0] = std::make_shared<CcuRsAlgTemplateX>(param, myRank_, subCommRanks0);
    // tempMap[OMNIPIPE_RS_LEVEL1] = std::make_shared<CcuRsAlgTemplateY>(param, myRank_, subCommRanks1);
    // tempMap[OMNIPIPE_AG_LEVEL0] = std::make_shared<CcuGAlgTemplateX>(param, myRank_, subCommRanks0);
    // tempMap[OMNIPIPE_AG_LEVEL1] = std::make_shared<CcuGAlgTemplateY>(param, myRank_, subCommRanks1);
    HCCL_INFO("[calcRes] start");
    // 计算调用每一个template的资源
    resourceRequest.slaveThreadNum = 0; //ccu内部没有从流和notify
    resourceRequest.notifyNumOnMainThread = 0;

    AlgResourceRequest resRsReqLevelX;
    CHK_RET(rsAlgTempLevelX.CalcRes(comm, param, topoInfo, resRsReqLevelX));
    AlgResourceRequest resRsReqLevelY;
    CHK_RET(rsAlgTempLevelY.CalcRes(comm, param, topoInfo, resRsReqLevelY));
    AlgResourceRequest resGReqLevelX;
    CHK_RET(gAlgTempLevelX.CalcRes(comm, param, topoInfo, resGReqLevelX));
    AlgResourceRequest resGReqLevelY;
    CHK_RET(gAlgTempLevelY.CalcRes(comm, param, topoInfo, resGReqLevelY));

    CHK_RET(CalcResLevel(comm, param, topoInfo, resRsReqLevelX, resourceRequest, 0));
    CHK_RET(CalcResLevel(comm, param, topoInfo, resRsReqLevelY, resourceRequest, 1));
    CHK_RET(CalcResLevel(comm, param, topoInfo, resGReqLevelX, resourceRequest, 2));
    CHK_RET(CalcResLevel(comm, param, topoInfo, resGReqLevelY, resourceRequest, 3));


    // HCCL_INFO("[CalcResLevel] tempMap.size():[%d]", tempMap.size());
    // for (int level = 0; level < OMNIPIPE_AR_LEVEL_NUM; level++) {
    //     HCCL_INFO("[CalcResLevel] tempMap.count(level):[%d] level:[%d]", tempMap.count(level), level);
    //     if (tempMap.count(level) > 0) {
    //         CHK_RET(CalcResLevel(comm, param, topoInfo, tempMap[level], resourceRequest, level));
    //     }
    // }
    resourceRequest.slaveThreadNum += 1; // 需要一个主流和一个从流来并行2d   
    resourceRequest.notifyNumOnMainThread += 1; 
    resourceRequest.notifyNumPerThread.assign(resourceRequest.slaveThreadNum, 1);
    // resourceRequest.notifyNumPerThread.emplace_back(1);
    HCCL_DEBUG("[%s] slaveThreadNum:%d, notifyNumOnMainThread:%d", __func__, resourceRequest.slaveThreadNum, resourceRequest.notifyNumOnMainThread);

    return HCCL_SUCCESS;
}
 
template <typename AlgTopoMatch, typename CcuRsAlgTemplateX, typename CcuRsAlgTemplateY, typename CcuGAlgTemplateX, typename CcuGAlgTemplateY>
HcclResult CcuV2ReduceOmniPipeExecutor<AlgTopoMatch, CcuRsAlgTemplateX, CcuRsAlgTemplateY, CcuGAlgTemplateX, CcuGAlgTemplateY>::Orchestrate(
    const OpParam& param, const AlgResourceCtxSerializable& resCtx)
{
    HCCL_DEBUG("[%s] start", __func__);
    threads_ = resCtx.threads;
    HCCL_DEBUG("[%s]threads size: %u", __func__, threads_.size());
    myRank_ = resCtx.topoInfo.userRank;
    rankSize_ = resCtx.topoInfo.userRankSize;
    dataCount_ = param.DataDes.count;
    dataType_ = param.DataDes.dataType;
    dataTypeSize_ = DATATYPE_SIZE_TABLE[param.DataDes.dataType];
    dataSize_ = dataCount_ * dataTypeSize_;
    rankSizeLevel0_ = resCtx.algHierarchyInfo.infos[0][0].size();
    if (rankSizeLevel0_ == 0) {
        HCCL_ERROR("[%s] rankSizeLevel0 is 0", __func__);
        return HcclResult::HCCL_E_PARA;
    }

    rankSizeLevel1_ = resCtx.algHierarchyInfo.infos[0][1].size() / rankSizeLevel0_;
    if (rankSizeLevel1_ == 0) {
        HCCL_ERROR("[%s] rankSizeLevel1 is 0", __func__);
        return HcclResult::HCCL_E_PARA;
    }
    rankIdxLevel1_ = myRank_ / rankSizeLevel0_;
    rankIdxLevel0_ = myRank_ % rankSizeLevel0_;
    
    HCCL_DEBUG("[%s] myRank[%u] rankSizeLevel0[%u] rankSizeLevel1[%u] rankIdxLevel0[%u] rankIdxLevel1[%u]",
        __func__, myRank_, rankSizeLevel0_, rankSizeLevel1_, rankIdxLevel0_, rankIdxLevel1_);





    // 算法展开
    HcclResult ret = OrchestrateLoop(param, resCtx);
    CHK_PRT_RET(ret != HCCL_SUCCESS,
        HCCL_ERROR("[CcuV2ReduceOmniPipeExecutor][Orchestrate]errNo[0x%016llx] excutor kernel run failed",
            HCCL_ERROR_CODE(ret)), ret);

    return HCCL_SUCCESS;
}

template <typename AlgTopoMatch, typename CcuRsAlgTemplateX, typename CcuRsAlgTemplateY, typename CcuGAlgTemplateX, typename CcuGAlgTemplateY>
HcclResult CcuV2ReduceOmniPipeExecutor<AlgTopoMatch, CcuRsAlgTemplateX, CcuRsAlgTemplateY, CcuGAlgTemplateX, CcuGAlgTemplateY>::InitOmniPipeScratchParam(
            OmniPipeScratchParam& scratchParam, const OpParam& param,
            const std::vector<double>& endpointAttrBwAvg)
{
    //scratchParam.dataSizePerLoop\ scratchParam.dataWholeSize 在外部赋值
    scratchParam.levelRankSize = {rankSizeLevel0_, rankSizeLevel1_, 1};
    scratchParam.endpointAttrBw = endpointAttrBwAvg;
    scratchParam.levelAlgType = {1, 1, 1}; // [jjy][todo]rs说后面再修改？

    // std::vector<u64> dataSizeVec;
    // for (int i = 0; i < rankSize_; i++) {
    //     dataSizeVec.push_back(dataSize_);
    // }
    
    // scratchParam.dataSize = CalcCountToDataSize(allRankSplitData, dataTypeSize_);
    // scratchParam.dataSize = dataSizeVec;
    scratchParam.dataSize = dataSize_;
    scratchParam.dataTypeSize = dataTypeSize_;
    scratchParam.maxTmpMemSize = 200 * 1024 * 1024;
    scratchParam.opMode = param.opMode;
    scratchParam.engine = param.engine;
    return HCCL_SUCCESS;
}

template <typename AlgTopoMatch, typename CcuRsAlgTemplateX, typename CcuRsAlgTemplateY, typename CcuGAlgTemplateX, typename CcuGAlgTemplateY>
HcclResult CcuV2ReduceOmniPipeExecutor<AlgTopoMatch, CcuRsAlgTemplateX, CcuRsAlgTemplateY, CcuGAlgTemplateX, CcuGAlgTemplateY>::InitOmniPipeSliceParam(
            OmniPipeSliceParam& sliceParam, const OpParam& param,
            const std::vector<double>& endpointAttrBwAvg)
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
template <typename AlgTopoMatch, typename CcuRsAlgTemplateX, typename CcuRsAlgTemplateY, typename CcuGAlgTemplateX, typename CcuGAlgTemplateY>
HcclResult CcuV2ReduceOmniPipeExecutor<AlgTopoMatch, CcuRsAlgTemplateX, CcuRsAlgTemplateY, CcuGAlgTemplateX, CcuGAlgTemplateY>::GenTemplateAlgParamsByDimData(
            TemplateDataParams &tempAlgParams, StepSliceInfo &stepSliceInfo, u64 processedDataCount)
{
    tempAlgParams.count = 0;

    tempAlgParams.stepSliceInfo = stepSliceInfo;
    tempAlgParams.buffInfo.inBuffBaseOff = stepSliceInfo.buffInfo.inBuffBaseOff + processedDataCount * dataTypeSize_;
    tempAlgParams.buffInfo.outBuffBaseOff = stepSliceInfo.buffInfo.outBuffBaseOff + processedDataCount * dataTypeSize_;
    // tempAlgParams.stepSliceInfo.buffInfo.inBuffBaseOff
    //     = stepSliceInfo.buffInfo.inBuffBaseOff + 512;
    // tempAlgParams.stepSliceInfo.buffInfo.outBuffBaseOff
    //     = stepSliceInfo.buffInfo.outBuffBaseOff + 512;

    // HCCL_DEBUG("[%s]myRank[%u] inBuffBaseOff[%llu] processedDataCount[%llu] end inBuffBaseOff[%llu]", __func__,
    //     myRank_, stepSliceInfo.buffInfo.inBuffBaseOff, processedDataCount, tempAlgParams.stepSliceInfo.buffInfo.inBuffBaseOff);

    // HCCL_DEBUG("[%s]myRank[%u] outBuffBaseOff[%llu] processedDataCount[%llu] end outBuffBaseOff[%llu]", __func__,
    //     myRank_, stepSliceInfo.buffInfo.outBuffBaseOff, processedDataCount, tempAlgParams.stepSliceInfo.buffInfo.outBuffBaseOff);

    tempAlgParams.inputSliceStride = 0;
    tempAlgParams.outputSliceStride = 0;
    tempAlgParams.sliceSize = 0;
    tempAlgParams.localCopyFlag = 0;
    return HcclResult::HCCL_SUCCESS;
}
template <typename AlgTopoMatch, typename CcuRsAlgTemplateX, typename CcuRsAlgTemplateY, typename CcuGAlgTemplateX, typename CcuGAlgTemplateY>
HcclResult CcuV2ReduceOmniPipeExecutor<AlgTopoMatch, CcuRsAlgTemplateX, CcuRsAlgTemplateY, CcuGAlgTemplateX, CcuGAlgTemplateY>::GenTempAlgParamsIn2HCCLBuff(
    TemplateDataParams &tempAlgParams, StepSliceInfo &stepSliceInfo, u64 processedDataCount, const AlgResourceCtxSerializable &resCtx, const OpParam &param)
{
    tempAlgParams.count = 0;
    stepSliceInfo.buffInfo.hcclBuff = resCtx.cclMem;
    stepSliceInfo.buffInfo.inputPtr = param.inputPtr;
    stepSliceInfo.buffInfo.outputPtr = resCtx.cclMem.addr;
    stepSliceInfo.buffInfo.inBuffType = BufferType::INPUT;
    stepSliceInfo.buffInfo.outBuffType = BufferType::HCCL_BUFFER;
    stepSliceInfo.buffInfo.hcclBuffType = BufferType::HCCL_BUFFER;

    // stepSliceInfo.buffInfo.inBuffBaseOff =  processedDataCount * dataTypeSize_ + stepSliceInfo.buffInfo.inBuffBaseOff; // 64 * 4  
    // stepSliceInfo.buffInfo.outBuffBaseOff = processedDataCount * dataTypeSize_ + stepSliceInfo.buffInfo.outBuffBaseOff; // 64 * 4
    // HCCL_INFO("[%s] GenTempAlgParamsIn2HCCLBuff Start stepSliceInfo.buffInfo.inBuffBaseOff[%d] stepSliceInfo.buffInfo.outBuffBaseOff[%d]", __func__, stepSliceInfo.buffInfo.inBuffBaseOff, stepSliceInfo.buffInfo.outBuffBaseOff);
    tempAlgParams.buffInfo = stepSliceInfo.buffInfo;
    tempAlgParams.stepSliceInfo = stepSliceInfo;
    tempAlgParams.buffInfo.inBuffBaseOff = processedDataCount * dataTypeSize_ + stepSliceInfo.buffInfo.inBuffBaseOff;
    tempAlgParams.buffInfo.outBuffBaseOff = processedDataCount * dataTypeSize_ + stepSliceInfo.buffInfo.outBuffBaseOff;
    
    tempAlgParams.inputSliceStride = 0;
    tempAlgParams.outputSliceStride = 0;
    tempAlgParams.sliceSize = 0;
    tempAlgParams.root = param.root;
    // tempAlgParams.subRoot = param.root;
    // tempAlgParams.isSameXAxis = false;
    // tempAlgParams.isSameYAxis = false;
    tempAlgParams.localCopyFlag = 0;
    tempAlgParams.repeatNum = stepSliceInfo.stepCount.size();

    return HcclResult::HCCL_SUCCESS;
}

template <typename AlgTopoMatch, typename CcuRsAlgTemplateX, typename CcuRsAlgTemplateY, typename CcuGAlgTemplateX, typename CcuGAlgTemplateY>
HcclResult CcuV2ReduceOmniPipeExecutor<AlgTopoMatch, CcuRsAlgTemplateX, CcuRsAlgTemplateY, CcuGAlgTemplateX, CcuGAlgTemplateY>::GenTempAlgParamsHCCLBuff2HCCLBuff(
    TemplateDataParams &tempAlgParams, StepSliceInfo &stepSliceInfo, u64 processedDataCount, const AlgResourceCtxSerializable &resCtx, const OpParam &param)
{
    tempAlgParams.count = 0;
    stepSliceInfo.buffInfo.hcclBuff = resCtx.cclMem;
    stepSliceInfo.buffInfo.inputPtr = resCtx.cclMem.addr;
    stepSliceInfo.buffInfo.outputPtr = resCtx.cclMem.addr;
    stepSliceInfo.buffInfo.inBuffType = BufferType::HCCL_BUFFER;
    stepSliceInfo.buffInfo.outBuffType = BufferType::HCCL_BUFFER;
    stepSliceInfo.buffInfo.hcclBuffType = BufferType::HCCL_BUFFER;
    tempAlgParams.buffInfo = stepSliceInfo.buffInfo;
    tempAlgParams.stepSliceInfo = stepSliceInfo;
    tempAlgParams.buffInfo.inBuffBaseOff = processedDataCount * dataTypeSize_ + stepSliceInfo.buffInfo.inBuffBaseOff;
    tempAlgParams.buffInfo.outBuffBaseOff = processedDataCount * dataTypeSize_ + stepSliceInfo.buffInfo.outBuffBaseOff;
    tempAlgParams.inputSliceStride = 0;
    tempAlgParams.outputSliceStride = 0;
    tempAlgParams.sliceSize = 0;
    tempAlgParams.root = param.root;
    // tempAlgParams.subRoot = param.root;
    // tempAlgParams.isSameXAxis = false;
    // tempAlgParams.isSameYAxis = false;
    tempAlgParams.localCopyFlag = 0;
    tempAlgParams.repeatNum = stepSliceInfo.stepCount.size();

    return HcclResult::HCCL_SUCCESS;
}

template <typename AlgTopoMatch, typename CcuRsAlgTemplateX, typename CcuRsAlgTemplateY, typename CcuGAlgTemplateX, typename CcuGAlgTemplateY>
HcclResult CcuV2ReduceOmniPipeExecutor<AlgTopoMatch, CcuRsAlgTemplateX, CcuRsAlgTemplateY, CcuGAlgTemplateX, CcuGAlgTemplateY>::OrchestrateLoop(
            const OpParam& param, const AlgResourceCtxSerializable& resCtx)
{
    HCCL_INFO("[%s] Start", __func__);

    // 初始化通信域subCommRanks
    std::vector<std::vector<u32>> subCommRanks0;
    std::vector<std::vector<u32>> subCommRanks1;
    auto& algHierarchyInfo_local = const_cast<ops_hccl::AlgHierarchyInfoForAllLevel&>(resCtx.algHierarchyInfo);
    CHK_RET(InitSubCommRanks(subCommRanks0, subCommRanks1, algHierarchyInfo_local));
    bool isRoot = (myRank_ == param.root);

    // 初始化template
    CcuRsAlgTemplateX rsAlgTempX(param, myRank_, subCommRanks0);
	CcuRsAlgTemplateY rsAlgTempY(param, myRank_, subCommRanks1);
    CcuGAlgTemplateX gAlgTempX(param, myRank_, subCommRanks0);
	CcuGAlgTemplateY gAlgTempY(param, myRank_, subCommRanks1);
    levelThreads_.resize(CCU_OMNIPIPE_LEVEL_NUM);
    levelThreads_[CCU_OMNIPIPE_LEVEL0].push_back(threads_[0]);
    levelThreads_[CCU_OMNIPIPE_LEVEL1].push_back(threads_[1]);

    // 公共参数初始化
	TemplateDataParams tempAlgParamsCommon;
	tempAlgParamsCommon.buffInfo.inputPtr = param.inputPtr;
	tempAlgParamsCommon.buffInfo.outputPtr = param.outputPtr;
	tempAlgParamsCommon.buffInfo.inputSize = param.inputSize;
	tempAlgParamsCommon.buffInfo.outputSize = param.outputSize;
	tempAlgParamsCommon.buffInfo.hcclBuff = resCtx.cclMem;
	tempAlgParamsCommon.inputSliceStride = dataSize_;
	tempAlgParamsCommon.outputSliceStride = dataSize_;

    // 资源模板初始化
    TemplateResource templateResourceCommon;
    TemplateResource templateResourceRsX = templateResourceCommon;
    templateResourceRsX.ccuKernels.insert(templateResourceRsX.ccuKernels.end(),
        resCtx.ccuKernels.begin(),
        resCtx.ccuKernels.begin() + resCtx.ccuKernelNum[0]);
    TemplateResource templateResourceRsY = templateResourceCommon;
    templateResourceRsY.ccuKernels.insert(templateResourceRsY.ccuKernels.end(),
        resCtx.ccuKernels.begin() + resCtx.ccuKernelNum[0],
        resCtx.ccuKernels.begin() + resCtx.ccuKernelNum[0] + resCtx.ccuKernelNum[1]);

    TemplateResource templateResourceGX = templateResourceCommon;
    templateResourceGX.ccuKernels.insert(templateResourceGX.ccuKernels.end(), 
        resCtx.ccuKernels.begin() + resCtx.ccuKernelNum[0] + resCtx.ccuKernelNum[1],
        resCtx.ccuKernels.begin() + resCtx.ccuKernelNum[0] + resCtx.ccuKernelNum[1]+ resCtx.ccuKernelNum[2]);
    TemplateResource templateResourceGY = templateResourceCommon;
    templateResourceGY.ccuKernels.insert(templateResourceGY.ccuKernels.end(),
        resCtx.ccuKernels.begin() + resCtx.ccuKernelNum[0] + resCtx.ccuKernelNum[1] + resCtx.ccuKernelNum[2],
        resCtx.ccuKernels.begin() + resCtx.ccuKernelNum[0] + resCtx.ccuKernelNum[1] + resCtx.ccuKernelNum[2]+ resCtx.ccuKernelNum[3]);

    templateResourceRsX.threads.emplace_back(threads_[0]);
    templateResourceRsY.threads.emplace_back(threads_[1]);
    templateResourceGX.threads.emplace_back(threads_[0]);
    templateResourceGY.threads.emplace_back(threads_[1]);

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

    // 2.1 获取每个rank切分的数据量count TODO:不需要修改？每个卡数据量，总数据量 / 卡数
    auto allRankSplitData = OmniPipeSplitData(rankSize_, dataCount_, dataTypeSize_);


    // 2.2 计算loop次数
#if 0
    OmniPipeScratchParam scratchParam;
    CHK_RET(InitOmniPipeScratchParam(scratchParam, param, endpointAttrBwAvg));
    scratchParam.maxTmpMemSize = resCtx.cclMem.size;
    // 将数据量切分count转化为dataSize，传给scratchParam
    scratchParam.dataSize = CalcCountToDataSize(allRankSplitData, dataTypeSize_);
    std::vector<u64> loopInfo = CalcOmniPipeScratchInfo(scratchParam); // [jjy][todo]待考虑是否要这样计算？
    // 中转内存(/UB Bound)单次最多能够接受的output count，注意是count不是size
    u64 maxCountPerLoop = loopInfo[0];
    u64 loopTimes = loopInfo[1];
    HCCL_DEBUG("[%s]maxCountPerLoop[%u], loopTimes[%u]", __func__, maxCountPerLoop, loopTimes);
#else
    // u64 maxCountPerLoop = static_cast<u64>(UB_MAX_DATA_SIZE) / dataTypeSize_; // UB传输的限制
    u64 maxCountPerLoop = static_cast<u64>(256) / dataTypeSize_; 
    u32 loopTimes = allRankSplitData[0] / maxCountPerLoop + ((allRankSplitData[0] % maxCountPerLoop == 0) ? 0 : 1); //总的需要传输的数据量 / UB限制
    // maxCountPerLoop = static_cast<u64>(256) / dataTypeSize_; 
    // loopTimes = allRankSplitData[0] / maxCountPerLoop + ((allRankSplitData[0] % maxCountPerLoop == 0) ? 0 : 1); //总的需要传输的数据量 / UB限制
    HCCL_DEBUG("[%s] myRank[%u] maxCountPerLoop[%u] loopTimes[%llu]", __func__, myRank_, maxCountPerLoop, loopTimes);
#endif

    // 2.3 获取每个rank，每个loop切分的数据量count
    auto multiLoopAllRankSplitData =
        OmniPipeSplitRankDataLoop(allRankSplitData, maxCountPerLoop, loopTimes, dataTypeSize_);
    HCCL_DEBUG("[%s]maxCountPerLoop[%u], loopTimes[%u]", __func__, maxCountPerLoop, loopTimes);
    for (int i=0;i<multiLoopAllRankSplitData.size();i++){
        for(int j=0;j<multiLoopAllRankSplitData[i].size();j++){
            HCCL_INFO("[jjy]rankId[%d],allRankSplitData[%d][%d]:%d multiLoopAllRankSplitData[%d][%d]:%d", myRank_, i, j, allRankSplitData[i], i, j, multiLoopAllRankSplitData[i][j]);
        }
    }

    // 3.1 计算n-1次loop的slice信息
    u64 perLoopSize = multiLoopAllRankSplitData[0][0] * dataTypeSize_;
    perLoopSize = dataSize_ > perLoopSize ? perLoopSize : dataSize_;
    HCCL_DEBUG("[%s][jjy] perLoopSize[%u] dataSize_[%u]", __func__, perLoopSize, dataSize_);
    std::vector<u64> dataSizePerLoop(rankSize_, perLoopSize); //注意的参数
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
    
    // 4 进行一次loop的数据处理
    u64 processedDataCount = 0;
    TemplateDataParams tempRsAlgParamsX = tempAlgParamsCommon;
	TemplateDataParams tempRsAlgParamsY = tempAlgParamsCommon;
    TemplateDataParams tempGAlgParamsX = tempAlgParamsCommon;
	TemplateDataParams tempGAlgParamsY = tempAlgParamsCommon;
    

    OmniPipeSliceInfo omniPipeSliceInfoRS;
    OmniPipeSliceInfo omniPipeSliceInfoG;
    for (u64 loop = 0; loop < loopTimes; loop++) {//loopTimes
        CHK_PRT_RET(
            multiLoopAllRankSplitData.size() <= loop,
            HCCL_ERROR("[CcuV2ReduceOmniPipeExecutor][Orchestrate] multiLoopAllRankSplitData.size() <= loop"),
            HCCL_E_PARA);

        sliceParam.dataSizePerLoop = CalcCountToDataSize(multiLoopAllRankSplitData[loop], dataTypeSize_);
        sliceParam.dataWholeSize = CalcCountToDataSize(allRankSplitData, dataTypeSize_);
        omniPipeSliceInfoRS = CalcRSOmniPipeSliceInfo(sliceParam);
        omniPipeSliceInfoG = CalcGatherOmniPipeSliceInfo(sliceParam);
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
            GenTemplateAlgParamsByDimData(tempRsAlgParamsX, omniPipeSliceInfoRS.dataSliceLevel0[i], processedDataCount);
            CHK_RET(rsAlgTempX.KernelRun(param, tempRsAlgParamsX, templateResourceRsX));
            // 第一步做完后回到主流做尾同步
            // level1
            GenTemplateAlgParamsByDimData(tempRsAlgParamsY, omniPipeSliceInfoRS.dataSliceLevel1[i], processedDataCount);
            CHK_RET(rsAlgTempY.KernelRun(param, tempRsAlgParamsY, templateResourceRsY));

            CHK_RET(PostSyncInterThreads(mainThread, syncThreads, notifyIdxesSubToMain)); 
        }

        // // 4.4 AG本地拷贝 input-->buff
        // if (myRank_ == param.root) { 
        //     HCCL_DEBUG("[%s] AG local copy start, myRank[%d], currDataCount %llu, processedDataCount %llu",
        //                     __func__, myRank_, currDataCount, processedDataCount);
        //     CHK_RET(PreSyncInterThreads(mainThread, syncThreads, notifyIdxesMainToSub));
        //     // 本地拷贝
        //     TemplateDataParams tempAlgParamLocalCopy;
        //     tempAlgParamLocalCopy.buffInfo.inputPtr = param.inputPtr;
        //     tempAlgParamLocalCopy.buffInfo.outputPtr = resCtx.cclMem.addr;
        //     tempAlgParamLocalCopy.buffInfo.inputSize = param.inputSize;
        //     tempAlgParamLocalCopy.buffInfo.outputSize = param.outputSize;
        //     tempAlgParamLocalCopy.buffInfo.hcclBuff = resCtx.cclMem;
        //     tempAlgParamLocalCopy.buffInfo.inBuffType = BufferType::INPUT;
        //     tempAlgParamLocalCopy.buffInfo.outBuffType = BufferType::HCCL_BUFFER;
        //     tempAlgParamLocalCopy.count = currDataCount;
        //     // tempAlgParamLocalCopy.stepSliceInfo.buffInfo.inBuffBaseOff = myRank_ * multiLoopAllRankSplitData[loop][0] * dataTypeSize_ + processedDataCount * dataTypeSize_;
        //     // tempAlgParamLocalCopy.stepSliceInfo.buffInfo.outBuffBaseOff = processedDataCount * dataTypeSize_;
        //     // tempAlgParamLocalCopy.buffInfo.inBuffBaseOff = myRank_ * multiLoopAllRankSplitData[loop][0] * dataTypeSize_ + processedDataCount * dataTypeSize_;
        //     // tempAlgParamLocalCopy.stepSliceInfo.buffInfo.inBuffBaseOff = myRank_ * allRankSplitData[0] * dataTypeSize_ + processedDataCount * dataTypeSize_;
        //     // tempAlgParamLocalCopy.stepSliceInfo.buffInfo.outBuffBaseOff = processedDataCount * dataTypeSize_;
        //     tempAlgParamLocalCopy.buffInfo.inBuffBaseOff = myRank_ * allRankSplitData[0] * dataTypeSize_ + processedDataCount * dataTypeSize_;
        //     tempAlgParamLocalCopy.buffInfo.outBuffBaseOff = myRank_ * allRankSplitData[0] * dataTypeSize_ + processedDataCount * dataTypeSize_;
        //     // tempAlgParamLocalCopy.inputSliceStride = multiLoopAllRankSplitData[loop][0] * dataTypeSize_;
        //     // tempAlgParamLocalCopy.outputSliceStride = multiLoopAllRankSplitData[loop][0] * dataTypeSize_;
        //     // tempAlgParamLocalCopy.inputSliceStride = (myRank_ == (rankSize_ - 1)) ? allRankSplitData[myRank_-1] * dataTypeSize_: allRankSplitData[myRank_] * dataTypeSize_;//allRankSplitData[0] * dataTypeSize_
       //     // tempAlgParamLocalCopy.repeatNum = rankSize_;
        //     tempAlgParamLocalCopy.sliceSize = currDataCount * dataTypeSize_;
        //     tempAlgParamLocalCopy.localCopyFlag = 1;

        //     // templateResourceGX.threads.clear();
        //     // templateResourceGX.threads.emplace_back(threads_[0]);
        //     CHK_RET(gAlgTempX->KernelRun(param, tempAlgParamLocalCopy, templateResourceGX));
        //     CHK_RET(PostSyncInterThreads(mainThread, syncThreads, notifyIdxesSubToMain));
        //     HCCL_DEBUG("[%s] AG local copy end", __func__);
        // }   //     // tempAlgParamLocalCopy.outputSliceStride = 0;
      


        // 4.5 GATHER for内层2d
        u32 level0StepCountAG = omniPipeSliceInfoG.dataSliceLevel0.size();
        HCCL_DEBUG("[%s] level0StepCountAG %u", __func__, level0StepCountAG);
        for (u32 i = 0; i < level0StepCountAG; i++) {
            // 初始化机内template param
            // GenTemplateAlgParamsByDimData(tempGAlgParamsX, omniPipeSliceInfoAG.dataSliceLevel0[i], processedDataCount);
            // GenTemplateAlgParamsByDimData(tempGAlgParamsY, omniPipeSliceInfoAG.dataSliceLevel1[i], processedDataCount);
            // 开始前同步
            CHK_RET(PreSyncInterThreads(mainThread, syncThreads, notifyIdxesMainToSub));
            

            CHK_RET(GenTempAlgParamsIn2HCCLBuff(tempGAlgParamsX, omniPipeSliceInfoG.dataSliceLevel0[i], processedDataCount, resCtx, param));
            CHK_RET(GenTempAlgParamsIn2HCCLBuff(tempGAlgParamsY, omniPipeSliceInfoG.dataSliceLevel1[i], processedDataCount, resCtx, param));
            
            if (i == 0) { // 第一步
                HCCL_INFO("[%s][KernelRun] first start.", __func__);
                if (isRoot){ // 0 
                    HCCL_INFO("[%s][isRoot] myRank_[%d] 0.", __func__, myRank_);
                     gAlgTempY.subRoot = rootYAixs;  //0 纵向逻辑假root是0
                     gAlgTempX.subRoot = rootXAixs;  //0 横向逻辑假root是0
                }
                if (isSameXAxis && !isRoot) { // 2
                    HCCL_INFO("[%s][isSameXAxis] myRank_[%d] 0.", __func__, myRank_);  
                    gAlgTempX.subRoot = rankIdxLevel0_; //0 横向逻辑假root是0
                    gAlgTempY.subRoot = rootYAixs;      //0 纵向逻辑假root是0
                }

                if (isSameYAxis && !isRoot) { // 1;
                    HCCL_INFO("[%s][isSameYAxis] myRank_[%d] 0.", __func__, myRank_);
                    gAlgTempX.subRoot = rootXAixs;       //0 横向逻辑假root是0
                    gAlgTempY.subRoot = rankIdxLevel1_;  //0 纵向逻辑假root是0
                } 
                
                if (!isSameXAxis && !isSameYAxis && !isRoot){ // 3
                    HCCL_INFO("[%s][isDiagnol] myRank_[%d] 0.", __func__, myRank_);
                    gAlgTempX.subRoot = GetYRoot(); //0 横向逻辑假root是0
                    gAlgTempY.subRoot = GetXRoot(); //0 纵向逻辑假root是0

                }
            }else if (i == level0StepCountAG - 1) {  // 最后一步
                HCCL_INFO("[%s][KernelRun] lastStep.", __func__);
            // ----------------第n步----------------
            // 如果当前卡是root的同x轴节点 nhr ccl->usrOut
            // 如果当前卡是root的同y轴节点 mesh ccl->usrOut
                if (isSameXAxis && !isRoot) { // 2
                    HCCL_INFO("[%s][isSameXAxis] myRank_[%d] 2.", __func__, myRank_);
                    CHK_RET(GenTempAlgParamsHCCLBuff2HCCLBuff(tempGAlgParamsY, omniPipeSliceInfoG.dataSliceLevel1[i], processedDataCount, resCtx, param));
                    gAlgTempY.subRoot = rootYAixs;
                    gAlgTempX.subRoot = 999;
                } else if (isSameYAxis && !isRoot) { // 1
                    HCCL_INFO("[%s][isSameYAxis] myRank_[%d] 2.", __func__, myRank_);
                    CHK_RET(GenTempAlgParamsHCCLBuff2HCCLBuff(tempGAlgParamsX, omniPipeSliceInfoG.dataSliceLevel0[i], processedDataCount, resCtx, param));
                    gAlgTempX.subRoot = rootXAixs;
                    gAlgTempY.subRoot = 999;
                } else if(isRoot){//0
                    HCCL_INFO("[%s][isRoot] myRank_[%d] 2.", __func__, myRank_);
                    gAlgTempY.subRoot = rootYAixs;//0
                    gAlgTempX.subRoot = rootXAixs;
                } else{//3
                    HCCL_INFO("[%s][isDiagnol] myRank_[%d] 2.", __func__, myRank_);
                    gAlgTempY.subRoot = 999;
                    gAlgTempX.subRoot = 999;
                }
            } else {  // 中间的所有步
                HCCL_INFO("[%s][KernelRun] middlestep start.", __func__);
            // ----------------第2 ~ n-1步----------------
            // 如果当前卡是root的同x轴节点 nhr ccl->usrOut
            // 如果当前卡是root的同y轴节点 mesh usrOut->usrOut
            // 如果当前卡是斜对角节点 mesh usrOut->ccl 
                if(isRoot){
                    HCCL_INFO("[%s][isRoot] myRank_[%d] 1.", __func__, myRank_); 
                    gAlgTempY.subRoot = rootYAixs; //0 纵向逻辑假root是0
                    gAlgTempX.subRoot = rootXAixs; //0 横向逻辑假root是0
                } else if (isSameXAxis && !isRoot) { // 0 2
                HCCL_INFO("[%s][isSameXAxis] myRank_[%d] 1.", __func__, myRank_);
                    CHK_RET(GenTempAlgParamsHCCLBuff2HCCLBuff(tempGAlgParamsY, omniPipeSliceInfoG.dataSliceLevel1[i], processedDataCount, resCtx, param));
                    gAlgTempX.subRoot = rankIdxLevel0_; //0 横向逻辑假root是0
                    gAlgTempY.subRoot = rootYAixs;      //0 纵向逻辑假root是0

                } else if (isSameYAxis && !isRoot) {
                    HCCL_INFO("[%s][isSameYAxis] myRank_[%d] 1.", __func__, myRank_);
                    CHK_RET(GenTempAlgParamsIn2HCCLBuff(tempGAlgParamsX, omniPipeSliceInfoG.dataSliceLevel0[i], processedDataCount, resCtx, param));
                    gAlgTempX.subRoot = rootXAixs;//TODO:root的地方待修改
                    gAlgTempY.subRoot = 999;
                } else {
                    HCCL_INFO("[%s][isDiagnol] myRank_[%d] 1.", __func__, myRank_);
                    CHK_RET(GenTempAlgParamsIn2HCCLBuff(tempGAlgParamsX, omniPipeSliceInfoG.dataSliceLevel0[i], processedDataCount, resCtx, param));
                    gAlgTempX.subRoot = GetYRoot();
                    gAlgTempY.subRoot = 999;
                }
                HCCL_INFO("[%s][KernelRun] middlestep.", __func__);
            }
            HCCL_INFO("[%s][KernelRun] start.", __func__);
            gAlgTempX.isStepOne_ = (i == 0);
            gAlgTempX.isloopOne_ = (loop == 0);
            gAlgTempX.isLastStep_ = (i == level0StepCountAG - 1);
 	        CHK_RET(gAlgTempX.KernelRun(param, tempGAlgParamsX, templateResourceGX));
            
            gAlgTempY.isStepOne_ = (i == 0);
	        gAlgTempY.isloopOne_ = (loop == 0);
            gAlgTempY.isLastStep_ = (i == level0StepCountAG - 1);
            CHK_RET(gAlgTempY.KernelRun(param, tempGAlgParamsY, templateResourceGY));
            //第一步做完后回到主流做尾同步
            CHK_RET(PostSyncInterThreads(mainThread, syncThreads, notifyIdxesSubToMain));
        }
        
        // if (myRank_ == param.root) {
        //     // 4.4 G本地拷贝 (TODO:待修改)
        //     HCCL_DEBUG("[%s] Gather local copy start, myRank[%d], currDataCount %llu, processedDataCount %llu dataSize_ %llu",
        //                     __func__, myRank_, dataCount_, processedDataCount, dataSize_);
        //     // ThreadHandle mainThread = threads_[0];
        //     // std::vector<ThreadHandle> syncThreads{threads_[1]};
        //     // std::vector<u32> notifyIdxesMainToSub{0};
        //     // std::vector<u32> notifyIdxesSubToMain{0};
        //     CHK_RET(PreSyncInterThreads(mainThread, syncThreads, notifyIdxesMainToSub));
        //     for (u32 i = 0; i < rankSize_; i++) {
                
        //         // templateResourceGX.threads.clear();
        //         // templateResourceGX.threads.emplace_back(threads_[1]);

        //         TemplateDataParams tempAlgParamLocalCopy;
        //         tempAlgParamLocalCopy.localCopyFlag = 1;
        //         tempAlgParamLocalCopy.buffInfo.outputPtr = param.outputPtr;
        //         tempAlgParamLocalCopy.buffInfo.hcclBuff = resCtx.cclMem;
        //         tempAlgParamLocalCopy.buffInfo.outBuffType = BufferType::OUTPUT;

        //         tempAlgParamLocalCopy.count = allRankSplitData[i]; // 128
        //         tempAlgParamLocalCopy.sliceSize = allRankSplitData[i] *dataTypeSize_ ; // 128*4
        //         tempAlgParamLocalCopy.buffInfo.outBuffBaseOff = i * allRankSplitData[i] * dataTypeSize_; // i * 512
        //         tempAlgParamLocalCopy.buffInfo.inBuffBaseOff = i * allRankSplitData[i] * dataTypeSize_;  // i * 512
        //         if (i == param.root) {
        //             tempAlgParamLocalCopy.buffInfo.inputPtr = param.inputPtr;
        //             tempAlgParamLocalCopy.buffInfo.inBuffType = BufferType::INPUT;
        //         } else {
        //             tempAlgParamLocalCopy.buffInfo.inputPtr = resCtx.cclMem.addr;
        //             tempAlgParamLocalCopy.buffInfo.inBuffType = BufferType::HCCL_BUFFER;
        //         }

        //         HCCL_DEBUG("[%s] tempAlgParamLocalCopy.buffInfo.inputPtr[%u] ",&(param.inputPtr));
        //         HCCL_DEBUG("[%s] myRank[%u] localCopy inBuffBaseOff[%lu] outBuffBaseOff[%lu] sliceSize[%lu]", __func__,
        //         myRank_, tempAlgParamLocalCopy.buffInfo.inBuffBaseOff, tempAlgParamLocalCopy.buffInfo.outBuffBaseOff,
        //         tempAlgParamLocalCopy.sliceSize);
        //         CHK_RET(gAlgTempX->KernelRun(param, tempAlgParamLocalCopy, templateResourceGX));
                
        //         }
        //     CHK_RET(PostSyncInterThreads(mainThread, syncThreads, notifyIdxesSubToMain));
        //     HCCL_DEBUG("[%s] AG local copy end", __func__);
        // }

        processedDataCount += currDataCount;
    }
    if (myRank_ == param.root) {
        // 4.4 G本地拷贝 (TODO:待修改)
        HCCL_DEBUG("[%s] Gather local copy start, myRank[%d], currDataCount %llu, processedDataCount %llu dataSize_ %llu",
                        __func__, myRank_, dataCount_, processedDataCount, dataSize_);
        ThreadHandle mainThread = threads_[0];
        std::vector<ThreadHandle> syncThreads{threads_[1]};
        std::vector<u32> notifyIdxesMainToSub{0};
        std::vector<u32> notifyIdxesSubToMain{0};
        u64 rankOffset = 0;
        CHK_RET(PreSyncInterThreads(mainThread, syncThreads, notifyIdxesMainToSub));
        for (u32 i = 0; i < rankSize_; i++) {
            TemplateDataParams tempAlgParamLocalCopy;
            tempAlgParamLocalCopy.localCopyFlag = 1;
            tempAlgParamLocalCopy.buffInfo.outputPtr = param.outputPtr;
            tempAlgParamLocalCopy.buffInfo.hcclBuff = resCtx.cclMem;
            tempAlgParamLocalCopy.buffInfo.outBuffType = BufferType::OUTPUT;

            tempAlgParamLocalCopy.count = allRankSplitData[i]; // 128
            tempAlgParamLocalCopy.sliceSize = allRankSplitData[i] *dataTypeSize_ ; // 128*4
            tempAlgParamLocalCopy.buffInfo.outBuffBaseOff = rankOffset; // i * 512
            tempAlgParamLocalCopy.buffInfo.inBuffBaseOff = rankOffset;  // i * 512
            if (i == param.root) {
                tempAlgParamLocalCopy.buffInfo.inputPtr = param.inputPtr;
                tempAlgParamLocalCopy.buffInfo.inBuffType = BufferType::INPUT;
            } else {
                tempAlgParamLocalCopy.buffInfo.inputPtr = resCtx.cclMem.addr;
                tempAlgParamLocalCopy.buffInfo.inBuffType = BufferType::HCCL_BUFFER;
            }

            HCCL_DEBUG("[%s] tempAlgParamLocalCopy.buffInfo.inputPtr[%u] ",&(param.inputPtr));
            HCCL_DEBUG("[%s] myRank[%u] localCopy inBuffBaseOff[%lu] outBuffBaseOff[%lu] sliceSize[%lu]", __func__,
            myRank_, tempAlgParamLocalCopy.buffInfo.inBuffBaseOff, tempAlgParamLocalCopy.buffInfo.outBuffBaseOff,
            tempAlgParamLocalCopy.sliceSize);
            CHK_RET(gAlgTempX.KernelRun(param, tempAlgParamLocalCopy, templateResourceGX));
            rankOffset += allRankSplitData[i] * dataTypeSize_;
        }
        CHK_RET(PostSyncInterThreads(mainThread, syncThreads, notifyIdxesSubToMain));
        HCCL_DEBUG("[%s] AG local copy end", __func__);
    }
    HCCL_INFO("[%s][OrchestrateLoop] End.", __func__);
    return HCCL_SUCCESS;
}

// #ifndef AICPU_COMPILE // [jjy][todo] 这里的TopoMatchUBX还是TopoMatchMultilevel？这里需要写ifndef吗？
REGISTER_EXEC_V2_MULTI(HcclCMDType::HCCL_CMD_REDUCE, 
                                CcuV2ReduceOmniPipe2D,
                                CcuV2ReduceOmniPipeExecutor, 
                                TopoMatchUBX, 
                                CcuTempReduceScatterOmniPipeMesh1DMem2Mem, 
                                CcuTempReduceScatterOmniPipeMesh1DMem2Mem, 
                                CcuTempGatherOmniPipeMesh1DMem2Mem,
                                CcuTempGatherOmniPipeMesh1DMem2MemY);
// #endif
}
