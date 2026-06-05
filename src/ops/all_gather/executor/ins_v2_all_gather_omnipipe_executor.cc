/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "ins_v2_all_gather_omnipipe_executor.h"
#include <algorithm>
#include <sstream>
#include "alg_data_trans_wrapper.h"
#include "alg_param.h"
#include "topo_match_ubx.h"
#include "topo_match_multilevel.h"
#include "topo_match_pcie_mix.h"
#include "ins_temp_all_gather_omnipipe_mesh_1D.h"
#include "ins_temp_all_gather_omnipipe_nhr_dpu.h"
#include "ins_temp_all_gather_omnipipe_nhr.h"
#include "topo_match_3_level.h"

namespace ops_hccl {
template <typename AlgTopoMatch, typename InsAlgTemplate0, typename InsAlgTemplate1, typename InsAlgTemplate2>
InsV2AllGatherOmniPipeExecutor<AlgTopoMatch, InsAlgTemplate0, InsAlgTemplate1,
                               InsAlgTemplate2>::InsV2AllGatherOmniPipeExecutor()
{
}

template <typename AlgTopoMatch, typename InsAlgTemplate0, typename InsAlgTemplate1, typename InsAlgTemplate2>
HcclResult InsV2AllGatherOmniPipeExecutor<AlgTopoMatch, InsAlgTemplate0, InsAlgTemplate1, InsAlgTemplate2>::InitCommInfo(
    const OpParam& param, const TopoInfoWithNetLayerDetails* topoInfo,
    const AlgHierarchyInfoForAllLevel& algHierarchyInfo)
{
    myRank_ = topoInfo->userRank;
    rankSize_ = topoInfo->userRankSize;
    devType_ = topoInfo->deviceType;
    dataType_ = param.DataDes.dataType;
    dataCount_ = param.DataDes.count;
    opMode_ = param.opMode;
    dataTypeSize_ = SIZE_TABLE[param.DataDes.dataType];
    algHierarchyInfo_ = algHierarchyInfo;
    
    // ✅ 新增维测日志：基础信息
    HCCL_INFO("[InsV2AllGatherOmniPipeExecutor][InitCommInfo][rank:%u] 初始化完成, rankSize:%u, devType:%u, "
              "dataType:%u, dataTypeSize:%u, dataCount:%lu, opMode:%u",
              myRank_, rankSize_, devType_, dataType_, dataTypeSize_, dataCount_, opMode_);
    
    // ✅ 新增维测日志：拓扑层级详细信息
    HCCL_INFO("[InsV2AllGatherOmniPipeExecutor][InitCommInfo][rank:%u] algHierarchyInfo.infos.size():%zu",
              myRank_, algHierarchyInfo_.infos.size());
    for (size_t i = 0; i < algHierarchyInfo_.infos.size(); i++) {
        HCCL_INFO("[InsV2AllGatherOmniPipeExecutor][InitCommInfo][rank:%u] 层级[%zu] 组数:%zu",
                  myRank_, i, algHierarchyInfo_.infos[i].size());
        for (size_t j = 0; j < algHierarchyInfo_.infos[i].size(); j++) {
            HCCL_INFO("[InsV2AllGatherOmniPipeExecutor][InitCommInfo][rank:%u] 层级[%zu] 组[%zu] 大小:%zu",
                      myRank_, i, j, algHierarchyInfo_.infos[i][j].size());
        }
    }
    
    return HCCL_SUCCESS;
}

template <typename AlgTopoMatch, typename InsAlgTemplate0, typename InsAlgTemplate1, typename InsAlgTemplate2>
HcclResult InsV2AllGatherOmniPipeExecutor<AlgTopoMatch, InsAlgTemplate0, InsAlgTemplate1, InsAlgTemplate2>::BuildSubCommAndTempMap(
    const OpParam& param,
    const AlgHierarchyInfoForAllLevel& algHierarchyInfo,
    std::vector<std::vector<u32>>& subCommRanks0,
    std::vector<std::vector<u32>>& subCommRanks1,
    std::vector<std::vector<u32>>& subCommRanks2,
    std::map<u32, std::shared_ptr<InsAlgTemplateBase>>& tempMap,
    const TopoInfoWithNetLayerDetails* topoInfo)
{
    // ✅ 新增维测日志：函数入口
    HCCL_INFO("[InsV2AllGatherOmniPipeExecutor][BuildSubCommAndTempMap][rank:%u] 开始构建子通信域, "
              "level0Topo:%u, level0PcieMix:%d, topoType_:%u",
              myRank_, topoInfo->level0Topo, topoInfo->level0PcieMix, topoType_);

    // if(topoInfo->level0Topo == Level0Shape::MESH_1D_CLOS && !topoInfo->level0PcieMix) {
    //     // ✅ 新增维测日志：进入UBX拓扑分支
    //     HCCL_INFO("[InsV2AllGatherOmniPipeExecutor][BuildSubCommAndTempMap][rank:%u] 进入【UBX MESH_1D_CLOS】拓扑分支", myRank_);
        
    //     subCommRanks0 = {algHierarchyInfo_.infos[0][0]};
    //     // ✅ 新增维测日志：L0子通信域
    //     HCCL_INFO("[InsV2AllGatherOmniPipeExecutor][BuildSubCommAndTempMap][rank:%u] L0子通信域大小:%zu",
    //               myRank_, subCommRanks0[0].size());
        
    //     std::vector<u32> closRanks;
    //     u32 meshSize = algHierarchyInfo_.infos[0][0].size();
    //     // ✅ 新增维测日志：机内Mesh大小
    //     HCCL_INFO("[InsV2AllGatherOmniPipeExecutor][BuildSubCommAndTempMap][rank:%u] 单台服务器卡数(meshSize):%u", myRank_, meshSize);
        
    //     for(auto rank : algHierarchyInfo_.infos[0][1]) {
    //         if(rank % meshSize == topoInfo->userRank % meshSize) {
    //             closRanks.push_back(rank);
    //         }
    //     }
    //     // ✅ 新增维测日志：L1 CLOS子通信域
    //     HCCL_INFO("[InsV2AllGatherOmniPipeExecutor][BuildSubCommAndTempMap][rank:%u] L1 CLOS子通信域大小:%zu",
    //               myRank_, closRanks.size());
        
    //     subCommRanks1 = {closRanks};
    //     subCommRanks2 = algHierarchyInfo_.infos[1];
    //     // ✅ 新增维测日志：L2子通信域
    //     HCCL_INFO("[InsV2AllGatherOmniPipeExecutor][BuildSubCommAndTempMap][rank:%u] L2子通信域组数:%zu",
    //               myRank_, subCommRanks2.size());
    // } 
    if(topoType_ == TopoType::THREE_LEVEL) {
        // ✅ 新增维测日志：进入三层拓扑分支
        HCCL_INFO("[InsV2AllGatherOmniPipeExecutor][BuildSubCommAndTempMap][rank:%u] 进入【THREE_LEVEL】三层拓扑分支", myRank_);
        
        if (!algHierarchyInfo.infos[0].empty() && !algHierarchyInfo.infos[0][0].empty()) {
                subCommRanks0.push_back(algHierarchyInfo.infos[0][0]);
            } else {
                subCommRanks0.emplace_back(std::vector<u32>{myRank_});
            }
        // ✅ 新增维测日志：L0子通信域
        HCCL_INFO("[InsV2AllGatherOmniPipeExecutor][BuildSubCommAndTempMap][rank:%u] L0子通信域大小:%zu",
                  myRank_, subCommRanks0[0].size());
        
            if (!algHierarchyInfo.infos[1].empty() && !algHierarchyInfo.infos[1][0].empty()) {
                subCommRanks1.push_back(algHierarchyInfo.infos[1][0]);
            } else {
                subCommRanks1.emplace_back(std::vector<u32>{myRank_});
            }
        // ✅ 新增维测日志：L1子通信域
        HCCL_INFO("[InsV2AllGatherOmniPipeExecutor][BuildSubCommAndTempMap][rank:%u] L1子通信域大小:%zu",
                  myRank_, subCommRanks1[0].size());
        
            if (!algHierarchyInfo.infos[2].empty() && !algHierarchyInfo.infos[2][0].empty()) {
                subCommRanks2.push_back(algHierarchyInfo.infos[2][0]);
            } else {
                subCommRanks2.emplace_back(std::vector<u32>{myRank_});
            }
        // ✅ 新增维测日志：L2子通信域
        HCCL_INFO("[InsV2AllGatherOmniPipeExecutor][BuildSubCommAndTempMap][rank:%u] L2子通信域大小:%zu",
                  myRank_, subCommRanks2[0].size());
    }
    else {
        // ✅ 新增维测日志：进入通用拓扑分支
        HCCL_INFO("[InsV2AllGatherOmniPipeExecutor][BuildSubCommAndTempMap][rank:%u] 进入【通用】拓扑分支", myRank_);
        
        subCommRanks0 = algHierarchyInfo_.infos[0];
        subCommRanks1 = algHierarchyInfo_.infos[1];
        subCommRanks2.emplace_back(std::vector<u32>{myRank_});
        
        // ✅ 新增维测日志：各层子通信域
        HCCL_INFO("[InsV2AllGatherOmniPipeExecutor][BuildSubCommAndTempMap][rank:%u] L0子通信域组数:%zu, L1组数:%zu, L2组数:%zu",
                  myRank_, subCommRanks0.size(), subCommRanks1.size(), subCommRanks2.size());
    }
    rankSizeLevel_[OMNIPIPE_LEVEL0] = subCommRanks0[0].size();
    rankSizeLevel_[OMNIPIPE_LEVEL1] = subCommRanks1[0].size();
    rankSizeLevel_[OMNIPIPE_LEVEL2] = subCommRanks2[0].size();
    tempMap.clear();
    if (rankSizeLevel_[OMNIPIPE_LEVEL0] > 1) {
        tempMap[OMNIPIPE_LEVEL0] = std::make_shared<InsAlgTemplate0>(param, myRank_, subCommRanks0);
        // ✅ 新增维测日志：创建L0模板实例
        HCCL_INFO("[InsV2AllGatherOmniPipeExecutor][BuildSubCommAndTempMap][rank:%u] 创建L0模板实例成功", myRank_);
    }
    if (rankSizeLevel_[OMNIPIPE_LEVEL1] > 1) {
        tempMap[OMNIPIPE_LEVEL1] = std::make_shared<InsAlgTemplate1>(param, myRank_, subCommRanks1);
        // ✅ 新增维测日志：创建L1模板实例
        HCCL_INFO("[InsV2AllGatherOmniPipeExecutor][BuildSubCommAndTempMap][rank:%u] 创建L1模板实例成功", myRank_);
    }
    if (rankSizeLevel_[OMNIPIPE_LEVEL2] > 1) {
        tempMap[OMNIPIPE_LEVEL2] = std::make_shared<InsAlgTemplate2>(param, myRank_, subCommRanks2);
        // ✅ 新增维测日志：创建L2模板实例
        HCCL_INFO("[InsV2AllGatherOmniPipeExecutor][BuildSubCommAndTempMap][rank:%u] 创建L2模板实例成功", myRank_);
    }
    
    // ✅ 新增维测日志：函数出口
    HCCL_INFO("[InsV2AllGatherOmniPipeExecutor][BuildSubCommAndTempMap][rank:%u] 子通信域构建完成, tempMap大小:%zu",
              myRank_, tempMap.size());
    
    return HCCL_SUCCESS;
}

template <typename AlgTopoMatch, typename InsAlgTemplate0, typename InsAlgTemplate1, typename InsAlgTemplate2>
HcclResult
InsV2AllGatherOmniPipeExecutor<AlgTopoMatch, InsAlgTemplate0, InsAlgTemplate1, InsAlgTemplate2>::CalcAlgHierarchyInfo(
    HcclComm comm, TopoInfoWithNetLayerDetails* topoInfo, AlgHierarchyInfoForAllLevel& algHierarchyInfo)
{
    // ✅ 新增维测日志：函数入口
    HCCL_INFO("[InsV2AllGatherOmniPipeExecutor][CalcAlgHierarchyInfo][rank:%u] 开始拓扑匹配", myRank_);
    
    AlgTopoMatch topoMatch;
    CHK_RET(topoMatch.MatchTopo(comm, topoInfo, algHierarchyInfo));
    
    // ✅ 新增维测日志：拓扑匹配完成
    HCCL_INFO("[InsV2AllGatherOmniPipeExecutor][CalcAlgHierarchyInfo][rank:%u] 拓扑匹配完成", myRank_);
    
    return HCCL_SUCCESS;
}

template <typename AlgTopoMatch, typename InsAlgTemplate0, typename InsAlgTemplate1, typename InsAlgTemplate2>
HcclResult InsV2AllGatherOmniPipeExecutor<AlgTopoMatch, InsAlgTemplate0, InsAlgTemplate1, InsAlgTemplate2>::CalcRes(
    HcclComm comm, const OpParam& param, const TopoInfoWithNetLayerDetails* topoInfo,
    const AlgHierarchyInfoForAllLevel& algHierarchyInfo, AlgResourceRequest& resourceRequest)
{
    // ✅ 新增维测日志：函数入口
    HCCL_INFO("[InsV2AllGatherOmniPipeExecutor][CalcRes][rank:%u] 开始资源计算", myRank_);
    
    // 初始化一些基本成员变量
    InitCommInfo(param, topoInfo, algHierarchyInfo);
     if (algHierarchyInfo_.infos.size() == 3 &&
 	         !algHierarchyInfo_.infos[2].empty() && !algHierarchyInfo_.infos[2][0].empty()) {
 	         topoType_ = TopoType::THREE_LEVEL;
             // ✅ 新增维测日志：拓扑类型判断
             HCCL_INFO("[InsV2AllGatherOmniPipeExecutor][CalcRes][rank:%u] 检测到三层拓扑", myRank_);
 	     } else {
 	         topoType_ = TopoType::UBX_2LEVEL;
             // ✅ 新增维测日志：拓扑类型判断
             HCCL_INFO("[InsV2AllGatherOmniPipeExecutor][CalcRes][rank:%u] 检测到UBX两层拓扑", myRank_);
 	}
    // 计算subCommRanks
    std::vector<std::vector<u32>> subCommRanks0;
    std::vector<std::vector<u32>> subCommRanks1;
    std::vector<std::vector<u32>> subCommRanks2;    
    std::map<u32, std::shared_ptr<InsAlgTemplateBase>> tempMap;


    rankSizeLevel_.resize(OMNIPIPE_LEVEL_NUM);
    rankIdxLevel_.resize(OMNIPIPE_LEVEL_NUM);
    
    CHK_RET(BuildSubCommAndTempMap(param, algHierarchyInfo,
            subCommRanks0, subCommRanks1, subCommRanks2, tempMap, topoInfo));

    

    // ✅ 新增维测日志：各层rank大小
    HCCL_INFO("[InsV2AllGatherOmniPipeExecutor][CalcRes][rank:%u] 各层rank大小: L0=%u, L1=%u, L2=%u",
              myRank_, rankSizeLevel_[OMNIPIPE_LEVEL0], rankSizeLevel_[OMNIPIPE_LEVEL1], rankSizeLevel_[OMNIPIPE_LEVEL2]);

    rankIdxLevel_[OMNIPIPE_LEVEL0] = myRank_ % rankSizeLevel_[OMNIPIPE_LEVEL0];
    rankIdxLevel_[OMNIPIPE_LEVEL1] = myRank_ % (rankSizeLevel_[OMNIPIPE_LEVEL0] * rankSizeLevel_[OMNIPIPE_LEVEL1]) /
                                      rankSizeLevel_[OMNIPIPE_LEVEL0];
    rankIdxLevel_[OMNIPIPE_LEVEL2] = myRank_ / (rankSizeLevel_[OMNIPIPE_LEVEL0] * rankSizeLevel_[OMNIPIPE_LEVEL1]);
    
    // ✅ 新增维测日志：各层rank索引
    HCCL_INFO("[InsV2AllGatherOmniPipeExecutor][CalcRes][rank:%u] 各层rank索引: L0=%u, L1=%u, L2=%u",
              myRank_, rankIdxLevel_[OMNIPIPE_LEVEL0], rankIdxLevel_[OMNIPIPE_LEVEL1], rankIdxLevel_[OMNIPIPE_LEVEL2]);

    // if (rankSizeLevel_[OMNIPIPE_LEVEL0] > 1) {
    //     tempMap[OMNIPIPE_LEVEL0] = std::make_shared<InsAlgTemplate0>(param, myRank_, subCommRanks0);
    // }
    // if (rankSizeLevel_[OMNIPIPE_LEVEL1] > 1) {
    //     tempMap[OMNIPIPE_LEVEL1] = std::make_shared<InsAlgTemplate1>(param, myRank_, subCommRanks1);
    // }
    // if (rankSizeLevel_[OMNIPIPE_LEVEL2] > 1) {
    //     tempMap[OMNIPIPE_LEVEL2] = std::make_shared<InsAlgTemplate2>(param, myRank_, subCommRanks2);
    // }

    for (auto& temp : tempMap) {
        CHK_RET(CalcResLevel(comm, param, topoInfo, temp.second, resourceRequest));
    }
    HCCL_DEBUG("[InInsV2AllGatherOmniPipeExecutor][CalcRes] myRank[%u], notifyNumOnMainThread[%u], slaveThreadNum[%u], "
               "channels[%u]",
        myRank_, resourceRequest.notifyNumOnMainThread, resourceRequest.slaveThreadNum,
        resourceRequest.channels.size());
    
    // ✅ 新增维测日志：资源计算结果
    HCCL_INFO("[InsV2AllGatherOmniPipeExecutor][CalcRes][rank:%u] 资源计算完成, 主线程notify数:%u, 从线程数:%u, 通道数:%zu",
              myRank_, resourceRequest.notifyNumOnMainThread, resourceRequest.slaveThreadNum, resourceRequest.channels.size());
    
    return HCCL_SUCCESS;
}

template <typename AlgTopoMatch, typename InsAlgTemplate0, typename InsAlgTemplate1, typename InsAlgTemplate2>
HcclResult
InsV2AllGatherOmniPipeExecutor<AlgTopoMatch, InsAlgTemplate0, InsAlgTemplate1, InsAlgTemplate2>::CalcResLevel(
    HcclComm comm, const OpParam& param, const TopoInfoWithNetLayerDetails* topoInfo,
    std::shared_ptr<InsAlgTemplateBase> tempAlg, AlgResourceRequest& resourceRequest) const
{
    // ✅ 新增维测日志：函数入口
    HCCL_INFO("[InsV2AllGatherOmniPipeExecutor][CalcResLevel][rank:%u] 开始层级资源计算", myRank_);
    
    AlgResourceRequest resReqlevel;
    CHK_RET(tempAlg->CalcRes(comm, param, topoInfo, resReqlevel));
    
    // ✅ 新增维测日志：层级资源结果
    HCCL_INFO("[InsV2AllGatherOmniPipeExecutor][CalcResLevel][rank:%u] 层级资源: 从线程数:%u, 主线程notify数:%u, 通道数:%zu",
              myRank_, resReqlevel.slaveThreadNum, resReqlevel.notifyNumOnMainThread, resReqlevel.channels.size());
    
    resourceRequest.slaveThreadNum += resReqlevel.slaveThreadNum + 1;
    resourceRequest.notifyNumOnMainThread += 1;
    resourceRequest.notifyNumPerThread.emplace_back(resReqlevel.notifyNumOnMainThread +
                                                    1);  // temp2控制流：从流数量+主控制流
    resourceRequest.notifyNumPerThread.insert(resourceRequest.notifyNumPerThread.end(),
                                              resReqlevel.notifyNumPerThread.begin(),
                                              resReqlevel.notifyNumPerThread.end());
    resourceRequest.channels.emplace_back(resReqlevel.channels[0]);
    
    return HCCL_SUCCESS;
}

// 该函数必须按照level0、level1、level2的顺序调用
template <typename AlgTopoMatch, typename InsAlgTemplate0, typename InsAlgTemplate1, typename InsAlgTemplate2>
HcclResult InsV2AllGatherOmniPipeExecutor<AlgTopoMatch, InsAlgTemplate0, InsAlgTemplate1, InsAlgTemplate2>::
    PrepareResForTemplateLevel(u32 level, std::shared_ptr<InsAlgTemplateBase>& tempBase)
{
    // ✅ 新增维测日志：函数入口
    HCCL_INFO("[InsV2AllGatherOmniPipeExecutor][PrepareResForTemplateLevel][rank:%u] 开始为层级[%u]准备资源", myRank_, level);
    
    u32 levelThreadNum = tempBase->GetThreadNum();
    // ✅ 新增维测日志：层级线程数
    HCCL_INFO("[InsV2AllGatherOmniPipeExecutor][PrepareResForTemplateLevel][rank:%u] 层级[%u]线程数:%u", myRank_, level, levelThreadNum);
    
    if (level == OMNIPIPE_LEVEL0) {
        levelThreads_[OMNIPIPE_LEVEL0].assign(threads_.begin() + 1, threads_.begin() + 1 + levelThreadNum);
        tempMainThreadsXY_.push_back(levelThreads_[OMNIPIPE_LEVEL0].at(0));
        // ✅ 新增维测日志：L0线程分配
        HCCL_INFO("[InsV2AllGatherOmniPipeExecutor][PrepareResForTemplateLevel][rank:%u] L0主线程ID:%u",
                  myRank_, levelThreads_[OMNIPIPE_LEVEL0].at(0));
    } else if (level == OMNIPIPE_LEVEL1) {
        levelThreads_[OMNIPIPE_LEVEL1].assign(threads_.begin() + 1 + levelThreads_[OMNIPIPE_LEVEL0].size(),
                                              threads_.begin() + 1 + levelThreads_[0].size() + levelThreadNum);
        tempMainThreadsXY_.push_back(levelThreads_[OMNIPIPE_LEVEL1].at(0));
        // ✅ 新增维测日志：L1线程分配
        HCCL_INFO("[InsV2AllGatherOmniPipeExecutor][PrepareResForTemplateLevel][rank:%u] L1主线程ID:%u",
                  myRank_, levelThreads_[OMNIPIPE_LEVEL1].at(0));
    } else if (level == OMNIPIPE_LEVEL2) {
        levelThreads_[OMNIPIPE_LEVEL2].assign(
            threads_.begin() + 1 + levelThreads_[OMNIPIPE_LEVEL0].size() + levelThreads_[OMNIPIPE_LEVEL1].size(),
            threads_.end());
        tempMainThreadsZ_.push_back(levelThreads_[OMNIPIPE_LEVEL2].at(0));
        // ✅ 新增维测日志：L2线程分配
        HCCL_INFO("[InsV2AllGatherOmniPipeExecutor][PrepareResForTemplateLevel][rank:%u] L2主线程ID:%u",
                  myRank_, levelThreads_[OMNIPIPE_LEVEL2].at(0));
    }

    // 获取当前template各自的主thread上有多少notify
    AlgResourceRequest levelTempRequest;
    CHK_RET(tempBase->GetRes(levelTempRequest));
    if (level < OMNIPIPE_LEVEL2) {
        ntfIdxCtrlToTempXY_.push_back(levelTempRequest.notifyNumOnMainThread);
        ntfIdxTempToCtrlXY_.push_back(tempMainThreadsXY_.size() + tempMainThreadsZ_.size() - 1);
        // ✅ 新增维测日志：XY层notify索引
        HCCL_INFO("[InsV2AllGatherOmniPipeExecutor][PrepareResForTemplateLevel][rank:%u] XY层notify索引: ctrlToTemp=%u, tempToCtrl=%u",
                  myRank_, levelTempRequest.notifyNumOnMainThread, tempMainThreadsXY_.size() + tempMainThreadsZ_.size() - 1);
    } else {
        ntfIdxCtrlToTempZ_.push_back(levelTempRequest.notifyNumOnMainThread);
        ntfIdxTempToCtrlZ_.push_back(tempMainThreadsXY_.size() + tempMainThreadsZ_.size() - 1);
        // ✅ 新增维测日志：Z层notify索引
        HCCL_INFO("[InsV2AllGatherOmniPipeExecutor][PrepareResForTemplateLevel][rank:%u] Z层notify索引: ctrlToTemp=%u, tempToCtrl=%u",
                  myRank_, levelTempRequest.notifyNumOnMainThread, tempMainThreadsXY_.size() + tempMainThreadsZ_.size() - 1);
    }
    
    // ✅ 新增维测日志：函数出口
    HCCL_INFO("[InsV2AllGatherOmniPipeExecutor][PrepareResForTemplateLevel][rank:%u] 层级[%u]资源准备完成", myRank_, level);
    
    return HCCL_SUCCESS;
}

template <typename AlgTopoMatch, typename InsAlgTemplate0, typename InsAlgTemplate1, typename InsAlgTemplate2>
HcclResult InsV2AllGatherOmniPipeExecutor<AlgTopoMatch, InsAlgTemplate0, InsAlgTemplate1, InsAlgTemplate2>::Orchestrate(
    const OpParam& param, const AlgResourceCtxSerializable& resCtx)
{
    myRank_ = resCtx.topoInfo.userRank;
    rankSize_ = resCtx.topoInfo.userRankSize;
    dataCount_ = param.DataDes.count;
    dataTypeSize_ = SIZE_TABLE[param.DataDes.dataType];
    dataSize_ = dataCount_ * dataTypeSize_;
    dataType_ = param.DataDes.dataType;
    reduceOp_ = param.reduceType;
    algHierarchyInfo_ = resCtx.algHierarchyInfo;
    
    // ✅ 新增维测日志：编排入口
    HCCL_INFO("[InsV2AllGatherOmniPipeExecutor][Orchestrate][rank:%u] 开始算法编排, 总数据量:%lu字节(%lu个元素)",
              myRank_, dataSize_, dataCount_);

    maxTmpMemSize_ = resCtx.cclMem.size;  // maxTmpMemSize_设定为cclIn的大小，op中将申请的HcclBuff全给了cclIn
    // ✅ 新增维测日志：临时内存大小
    HCCL_INFO("[InsV2AllGatherOmniPipeExecutor][Orchestrate][rank:%u] CCL临时内存大小:%lu字节", myRank_, maxTmpMemSize_);

    // 计算subCommRanks
    std::vector<std::vector<u32>> subCommRanks0;
    std::vector<std::vector<u32>> subCommRanks1;
    std::vector<std::vector<u32>> subCommRanks2;    
    std::map<u32, std::shared_ptr<InsAlgTemplateBase>> tempMap;

    rankSizeLevel_.resize(OMNIPIPE_LEVEL_NUM);
    rankIdxLevel_.resize(OMNIPIPE_LEVEL_NUM);

    CHK_RET(BuildSubCommAndTempMap(param, algHierarchyInfo_,
            subCommRanks0, subCommRanks1, subCommRanks2, tempMap, &resCtx.topoInfo));

    
    // ✅ 新增维测日志：各层rank大小
    HCCL_INFO("[InsV2AllGatherOmniPipeExecutor][Orchestrate][rank:%u] 各层rank大小: L0=%u, L1=%u, L2=%u",
              myRank_, rankSizeLevel_[OMNIPIPE_LEVEL0], rankSizeLevel_[OMNIPIPE_LEVEL1], rankSizeLevel_[OMNIPIPE_LEVEL2]);

    rankIdxLevel_[OMNIPIPE_LEVEL0] = myRank_ % rankSizeLevel_[OMNIPIPE_LEVEL0];
    rankIdxLevel_[OMNIPIPE_LEVEL1] = myRank_ % (rankSizeLevel_[OMNIPIPE_LEVEL0] * rankSizeLevel_[OMNIPIPE_LEVEL1]) /
                                      rankSizeLevel_[OMNIPIPE_LEVEL0];
    rankIdxLevel_[OMNIPIPE_LEVEL2] = myRank_ / (rankSizeLevel_[OMNIPIPE_LEVEL0] * rankSizeLevel_[OMNIPIPE_LEVEL1]);
    
    // ✅ 新增维测日志：各层rank索引
    HCCL_INFO("[InsV2AllGatherOmniPipeExecutor][Orchestrate][rank:%u] 各层rank索引: L0=%u, L1=%u, L2=%u",
              myRank_, rankIdxLevel_[OMNIPIPE_LEVEL0], rankIdxLevel_[OMNIPIPE_LEVEL1], rankIdxLevel_[OMNIPIPE_LEVEL2]);

    // if (rankSizeLevel_[OMNIPIPE_LEVEL0] > 1) {
    //     tempMap[OMNIPIPE_LEVEL0] = std::make_shared<InsAlgTemplate0>(param, myRank_, subCommRanks0);
    // }
    // if (rankSizeLevel_[OMNIPIPE_LEVEL1] > 1) {
    //     tempMap[OMNIPIPE_LEVEL1] = std::make_shared<InsAlgTemplate1>(param, myRank_, subCommRanks1);
    // }
    // if (rankSizeLevel_[OMNIPIPE_LEVEL2] > 1) {
    //     tempMap[OMNIPIPE_LEVEL2] = std::make_shared<InsAlgTemplate2>(param, myRank_, subCommRanks2);
    // }

    // 为temp分配thread
    threads_ = resCtx.threads;
    controlThread_ = threads_.at(0);
    // ✅ 新增维测日志：控制线程ID
    HCCL_INFO("[InsV2AllGatherOmniPipeExecutor][Orchestrate][rank:%u] 控制线程ID:%u, 总线程数:%zu",
              myRank_, controlThread_, threads_.size());
    
    levelThreads_.resize(OMNIPIPE_LEVEL_NUM);
    for (auto& temp : tempMap) {
        CHK_RET(PrepareResForTemplateLevel(temp.first, temp.second));
    }

    CHK_RET(RestoreChannelMap(resCtx, remoteRankToChannelInfo_));

    // 算法展开
    HcclResult ret = OrchestrateLoop(param, resCtx, tempMap);
    CHK_PRT_RET(
        ret != HCCL_SUCCESS,
        HCCL_ERROR("[InsV2AllGatherOmniPipeExecutor][Orchestrate][rank:%u] errNo[0x%016llx] AllGather excutor kernel run failed",
                   myRank_, HCCL_ERROR_CODE(ret)),
        ret);
    
    // ✅ 新增维测日志：编排完成
    HCCL_INFO("[InsV2AllGatherOmniPipeExecutor][Orchestrate][rank:%u] 算法编排执行完成", myRank_);
    
    return HCCL_SUCCESS;
}


template <typename AlgTopoMatch, typename InsAlgTemplate0, typename InsAlgTemplate1, typename InsAlgTemplate2>
HcclResult
InsV2AllGatherOmniPipeExecutor<AlgTopoMatch, InsAlgTemplate0, InsAlgTemplate1,
                               InsAlgTemplate2>::GenTemplateAlgParamsByDimData(TemplateDataParams& tempAlgParams,
                                                                               StepSliceInfo& stepSliceInfo) const
{
    // tempAlgParams.buffInfo.hcclBuff 已在外部赋值
    tempAlgParams.buffInfo.inBuffType = BufferType::HCCL_BUFFER;
    tempAlgParams.buffInfo.outBuffType = BufferType::HCCL_BUFFER;

    tempAlgParams.buffInfo.inBuffBaseOff = stepSliceInfo.buffInfo.inBuffBaseOff;
    tempAlgParams.buffInfo.outBuffBaseOff = stepSliceInfo.buffInfo.outBuffBaseOff;
    tempAlgParams.buffInfo.hcclBuffBaseOff = stepSliceInfo.buffInfo.hcclBuffBaseOff;  // 实际上是空值
    tempAlgParams.stepSliceInfo = stepSliceInfo;

    // ✅ 新增维测日志：参数生成结果
    HCCL_INFO("[InsV2AllGatherOmniPipeExecutor][GenTemplateAlgParamsByDimData][rank:%u] 生成模板参数, "
              "输入偏移:%u, 输出偏移:%u",
              myRank_, tempAlgParams.buffInfo.inBuffBaseOff, tempAlgParams.buffInfo.outBuffBaseOff);
    
    return HCCL_SUCCESS;
}

template <typename AlgTopoMatch, typename InsAlgTemplate0, typename InsAlgTemplate1, typename InsAlgTemplate2>
HcclResult
InsV2AllGatherOmniPipeExecutor<AlgTopoMatch, InsAlgTemplate0, InsAlgTemplate1, InsAlgTemplate2>::OrchestrateLoop(
    const OpParam& param, const AlgResourceCtxSerializable& resCtx,
    std::map<u32, std::shared_ptr<InsAlgTemplateBase>>& tempMap)
{
    // ✅ 新增维测日志：循环入口
    HCCL_INFO("[InsV2AllGatherOmniPipeExecutor][OrchestrateLoop][rank:%u] 开始流水线循环执行", myRank_);
    
    //带宽赋值
    double bw_ag_l0=BW_OMNI_DEFAULT;
    double bw_ag_l1=BW_OMNI_DEFAULT;
    double bw_ag_l2=BW_OMNI_DEFAULT;
    double bw_rs_l0=BW_OMNI_DEFAULT;
    double bw_rs_l1=BW_OMNI_DEFAULT;
    double bw_rs_l2=BW_OMNI_DEFAULT;

    if (resCtx.topoInfo.level0PcieMix) {
        if (rankSizeLevel_[OMNIPIPE_LEVEL1]==2) {
            bw_ag_l1=BW_OMNI_PCIE_EIGHT_AG_CLOS;
            bw_rs_l1=BW_OMNI_PCIE_EIGHT_RS_CLOS;
        } else if (rankSizeLevel_[OMNIPIPE_LEVEL1]==4) {
            bw_ag_l1=BW_OMNI_PCIE_SIXTEEN_AG_CLOS;
            bw_rs_l1=BW_OMNI_PCIE_SIXTEEN_RS_CLOS;
        }
    }
    std::vector<double> endpointAttrBw{bw_ag_l0, bw_ag_l1, bw_ag_l2};
    
    // ✅ 新增维测日志：原始带宽
    HCCL_INFO("[InsV2AllGatherOmniPipeExecutor][OrchestrateLoop][rank:%u] 原始带宽: L0=%.2f, L1=%.2f, L2=%.2f",
              myRank_, bw_ag_l0, bw_ag_l1, bw_ag_l2);

    //计算等价带宽
    double eqBw0 = endpointAttrBw[0];//L0 mesh
    double eqBw1 = endpointAttrBw[1];//L1 NHR
    double eqBw2 = endpointAttrBw[2];//L2 NHR

    //level0为mesh,等价mesh为其本身
    //level1为nhr
    //level2, ranksize = 1
    eqBw1 = rankSizeLevel_[OMNIPIPE_LEVEL1] > 1 ? eqBw1 / (rankSizeLevel_[OMNIPIPE_LEVEL1] - 1) : eqBw1;
    eqBw2 = rankSizeLevel_[OMNIPIPE_LEVEL2] > 1 ? eqBw2 / (rankSizeLevel_[OMNIPIPE_LEVEL2] - 1) : eqBw2;

    std::vector<double> endpointAttrBwNew{eqBw0, eqBw1, eqBw2};
    // ✅ 新增维测日志：等价带宽
    HCCL_INFO("[InsV2AllGatherOmniPipeExecutor][OrchestrateLoop][rank:%u] 等价带宽: L0=%.2f, L1=%.2f, L2=%.2f",
              myRank_, eqBw0, eqBw1, eqBw2);
    
    u64 scratchBoundDataSize = maxTmpMemSize_ / rankSize_ / HCCL_MIN_SLICE_ALIGN * HCCL_MIN_SLICE_ALIGN / dataTypeSize_;
    // ✅ 新增维测日志：内存限制数据量
    HCCL_INFO("[InsV2AllGatherOmniPipeExecutor][OrchestrateLoop][rank:%u] 内存限制单循环最大数据量:%lu个元素",
              myRank_, scratchBoundDataSize);

    u64 transportBoundDataSize = UB_MAX_DATA_SIZE;
    u64 maxCountPerLoop = std::min(scratchBoundDataSize, transportBoundDataSize);
    u64 loopTimes = dataCount_ / maxCountPerLoop + static_cast<u64>(dataCount_ % maxCountPerLoop != 0);
    
    // ✅ 新增维测日志：循环参数
    HCCL_INFO("[InsV2AllGatherOmniPipeExecutor][OrchestrateLoop][rank:%u] 单循环最大数据量:%lu, 总循环次数:%lu",
              myRank_, maxCountPerLoop, loopTimes);

    u64 perLoopSize = maxCountPerLoop * dataTypeSize_;
    std::vector<u64> dataSizePerLoop(rankSize_, perLoopSize);
    std::vector<u64> dataWholeSize(rankSize_, perLoopSize);

    for (int i = 0; i < rankSize_; i++) {
        dataSizePerLoop.push_back(perLoopSize);
        dataWholeSize.push_back(perLoopSize);
    }

    OmniPipeSliceParam omniPipeSliceParam;
    HCCL_INFO("[InsV2AllGatherOmniPipeExecutor][OrchestrateLoop][rank:%u] 准备设置 levelRankSize: L0=%u, L1=%u, L2=%u",
            myRank_,
            rankSizeLevel_[OMNIPIPE_LEVEL0],
            rankSizeLevel_[OMNIPIPE_LEVEL1],
            rankSizeLevel_[OMNIPIPE_LEVEL2]);
    omniPipeSliceParam.levelRankSize = {rankSizeLevel_[OMNIPIPE_LEVEL0], rankSizeLevel_[OMNIPIPE_LEVEL1],
                                        rankSizeLevel_[OMNIPIPE_LEVEL2]};
    HCCL_INFO("[InsV2AllGatherOmniPipeExecutor][OrchestrateLoop][rank:%u] levelRankSize 设置完成: [%u, %u, %u]",
            myRank_,
            omniPipeSliceParam.levelRankSize[0],
            omniPipeSliceParam.levelRankSize[1],
            omniPipeSliceParam.levelRankSize[2]);
    omniPipeSliceParam.endpointAttrBw = endpointAttrBwNew;
    omniPipeSliceParam.dataSizePerLoop = dataSizePerLoop;
    omniPipeSliceParam.dataTypeSize = dataTypeSize_;
    omniPipeSliceParam.levelRankId = {rankIdxLevel_[OMNIPIPE_LEVEL0], rankIdxLevel_[OMNIPIPE_LEVEL1],
                                      rankIdxLevel_[OMNIPIPE_LEVEL2]};
    omniPipeSliceParam.opMode = opMode_;
    omniPipeSliceParam.engine = CommEngine::COMM_ENGINE_AICPU_TS;
    omniPipeSliceParam.dataWholeSize = dataWholeSize;

    OmniPipeSliceInfo alignSliceInfo = CalcAGOmniPipeSliceInfo(omniPipeSliceParam);
    // ✅ 新增维测日志：切片信息
    HCCL_INFO("[InsV2AllGatherOmniPipeExecutor][OrchestrateLoop][rank:%u] 对齐切片信息: L0切片数:%zu, L1切片数:%zu, L2切片数:%zu",
              myRank_, alignSliceInfo.dataSliceLevel0.size(), alignSliceInfo.dataSliceLevel1.size(), alignSliceInfo.dataSliceLevel2.size());
    
    // 4、计算第n次的loop的slice信息
    OmniPipeSliceInfo tailSliceInfo;
    if (dataCount_ % maxCountPerLoop != 0) {
        u64 perLoopSize = (dataCount_ % maxCountPerLoop) * dataTypeSize_;
        std::vector<u64> dataSizePerLoop(rankSize_, perLoopSize);
        std::vector<u64> dataWholeSize(rankSize_, perLoopSize);
        omniPipeSliceParam.dataSizePerLoop = dataSizePerLoop;
        omniPipeSliceParam.dataWholeSize = dataWholeSize;
        tailSliceInfo = CalcAGOmniPipeSliceInfo(omniPipeSliceParam);
        // ✅ 新增维测日志：尾切片信息
        HCCL_INFO("[InsV2AllGatherOmniPipeExecutor][OrchestrateLoop][rank:%u] 尾切片信息: L0切片数:%zu, L1切片数:%zu, L2切片数:%zu",
                  myRank_, tailSliceInfo.dataSliceLevel0.size(), tailSliceInfo.dataSliceLevel1.size(), tailSliceInfo.dataSliceLevel2.size());
    }

    u64 processedDataCount = 0;
    OmniPipeSliceInfo omniPipeSliceInfo;

    std::map<u32, TemplateResource> tempResMap;
    std::map<u32, TemplateDataParams> tempAlgParamMap;

    for (auto& temp : tempMap) {
        tempResMap[temp.first].channels = remoteRankToChannelInfo_[temp.first];
        tempResMap[temp.first].threads = levelThreads_[temp.first];
        tempAlgParamMap[temp.first].buffInfo.hcclBuff = resCtx.cclMem;
    }
    
    // ✅ 新增维测日志：开始循环执行
    HCCL_INFO("[InsV2AllGatherOmniPipeExecutor][OrchestrateLoop][rank:%u] 开始执行%lu次循环", myRank_, loopTimes);
    
    for (u64 loop = 0; loop < loopTimes; loop++) {
        u64 currDataCount = (loop == loopTimes - 1) ? dataCount_ - processedDataCount : maxCountPerLoop;
        // ✅ 新增维测日志：当前循环信息
        HCCL_INFO("[InsV2AllGatherOmniPipeExecutor][OrchestrateLoop][rank:%u] 第%lu次循环, 处理数据量:%lu个元素",
                  myRank_, loop, currDataCount);
        
        DataSlice src(param.inputPtr, processedDataCount * dataTypeSize_, currDataCount * dataTypeSize_, currDataCount);
        DataSlice dst(resCtx.cclMem.addr, myRank_ * currDataCount * dataTypeSize_, currDataCount * dataTypeSize_,
                        currDataCount);
        CHK_RET(LocalCopy(controlThread_, src, dst));
        // ✅ 新增维测日志：本地拷贝完成
        HCCL_INFO("[InsV2AllGatherOmniPipeExecutor][OrchestrateLoop][rank:%u] 第%lu次循环 输入数据拷贝完成", myRank_, loop);

        if (loop == loopTimes - 1 && dataCount_ % maxCountPerLoop != 0) {
            omniPipeSliceInfo = tailSliceInfo;
        } else {
            omniPipeSliceInfo = alignSliceInfo;
        }

        CHK_PRT_RET(omniPipeSliceInfo.dataSliceLevel2.size() == 0,
                    HCCL_ERROR("[InsV2AllGatherOmniPipeExecutor][OrchestrateLoop][rank:%u] omniPipeSliceInfo Level2 slice size is 0.", myRank_),
                    HCCL_E_PARA);

        u32 level2StepCount = omniPipeSliceInfo.dataSliceLevel2.size();
        u32 level0StepCount = omniPipeSliceInfo.dataSliceLevel0.size() / omniPipeSliceInfo.dataSliceLevel2.size();
        // ✅ 新增维测日志：step数
        HCCL_INFO("[InsV2AllGatherOmniPipeExecutor][OrchestrateLoop][rank:%u] 第%lu次循环 L2步数:%u, L0步数:%u",
                  myRank_, loop, level2StepCount, level0StepCount);

        for (int i = 0; i < level2StepCount; i++) {
            // ✅ 新增维测日志：L2 step开始
            HCCL_INFO("[InsV2AllGatherOmniPipeExecutor][OrchestrateLoop][rank:%u] 第%lu次循环 L2 step[%d]开始", myRank_, loop, i);
            
            if (rankSizeLevel_[OMNIPIPE_LEVEL2] > 1) {
                CHK_RET(GenTemplateAlgParamsByDimData(tempAlgParamMap[OMNIPIPE_LEVEL2],
                                                      omniPipeSliceInfo.dataSliceLevel2[i]));
                CHK_RET(PreSyncInterThreads(controlThread_, tempMainThreadsZ_, ntfIdxCtrlToTempZ_));
                CHK_RET(tempMap[OMNIPIPE_LEVEL2]->KernelRun(param, tempAlgParamMap[OMNIPIPE_LEVEL2],
                                                             tempResMap[OMNIPIPE_LEVEL2]));
                // ✅ 新增维测日志：L2 kernel执行完成
                HCCL_INFO("[InsV2AllGatherOmniPipeExecutor][OrchestrateLoop][rank:%u] 第%lu次循环 L2 step[%d] kernel执行完成", myRank_, loop, i);
            }
            for (int j = 0; j < level0StepCount; j++) {
                // ✅ 新增维测日志：L0 step开始
                HCCL_INFO("[InsV2AllGatherOmniPipeExecutor][OrchestrateLoop][rank:%u] 第%lu次循环 L0 step[%d]开始", myRank_, loop, j);
                
                CHK_RET(PreSyncInterThreads(controlThread_, tempMainThreadsXY_, ntfIdxCtrlToTempXY_));
                if (rankSizeLevel_[OMNIPIPE_LEVEL0] > 1) {
                    CHK_RET(GenTemplateAlgParamsByDimData(tempAlgParamMap[OMNIPIPE_LEVEL0],
                                                          omniPipeSliceInfo.dataSliceLevel0[i * level0StepCount + j]));
                    CHK_RET(tempMap[OMNIPIPE_LEVEL0]->KernelRun(param, tempAlgParamMap[OMNIPIPE_LEVEL0],
                                                                 tempResMap[OMNIPIPE_LEVEL0]));
                    // ✅ 新增维测日志：L0 kernel执行完成
                    HCCL_INFO("[InsV2AllGatherOmniPipeExecutor][OrchestrateLoop][rank:%u] 第%lu次循环 L0 step[%d] kernel执行完成", myRank_, loop, j);
                }
                if (rankSizeLevel_[OMNIPIPE_LEVEL1] > 1) {
                    CHK_RET(GenTemplateAlgParamsByDimData(tempAlgParamMap[OMNIPIPE_LEVEL1],
                                                          omniPipeSliceInfo.dataSliceLevel1[i * level0StepCount + j]));
                    CHK_RET(tempMap[OMNIPIPE_LEVEL1]->KernelRun(param, tempAlgParamMap[OMNIPIPE_LEVEL1],
                                                                 tempResMap[OMNIPIPE_LEVEL1]));
                    // ✅ 新增维测日志：L1 kernel执行完成
                    HCCL_INFO("[InsV2AllGatherOmniPipeExecutor][OrchestrateLoop][rank:%u] 第%lu次循环 L1 step[%d] kernel执行完成", myRank_, loop, j);
                }
                CHK_RET(PostSyncInterThreads(controlThread_, tempMainThreadsXY_, ntfIdxTempToCtrlXY_));
            }
            if (rankSizeLevel_[OMNIPIPE_LEVEL2] > 1) {
                CHK_RET(PostSyncInterThreads(controlThread_, tempMainThreadsZ_, ntfIdxTempToCtrlZ_));
            }
        }

        for (u32 rank = 0; rank < rankSize_; rank++) {
            DataSlice dst(param.outputPtr, (rank * dataCount_ + processedDataCount) * dataTypeSize_,
                            currDataCount * dataTypeSize_, currDataCount);
            DataSlice src(resCtx.cclMem.addr, rank * currDataCount * dataTypeSize_, currDataCount * dataTypeSize_,
                            currDataCount);
            CHK_RET(LocalCopy(controlThread_, src, dst));
        }
        // ✅ 新增维测日志：输出数据拷贝完成
        HCCL_INFO("[InsV2AllGatherOmniPipeExecutor][OrchestrateLoop][rank:%u] 第%lu次循环 输出数据拷贝完成", myRank_, loop);
        
        processedDataCount += currDataCount;
        // ✅ 新增维测日志：已处理数据量
        HCCL_INFO("[InsV2AllGatherOmniPipeExecutor][OrchestrateLoop][rank:%u] 已处理数据量:%lu/%lu",
                  myRank_, processedDataCount, dataCount_);
    }
    
    // ✅ 新增维测日志：循环执行完成
    HCCL_INFO("[InsV2AllGatherOmniPipeExecutor][OrchestrateLoop][rank:%u] 所有循环执行完成", myRank_);
    
    return HCCL_SUCCESS;
}
template <typename AlgTopoMatch, typename InsAlgTemplate0, typename InsAlgTemplate1, typename InsAlgTemplate2>
HcclResult
InsV2AllGatherOmniPipeExecutor<AlgTopoMatch, InsAlgTemplate0, InsAlgTemplate1, InsAlgTemplate2>::RestoreChannelMap(
    const AlgResourceCtxSerializable& resCtx,
    std::vector<std::map<u32, std::vector<ChannelInfo>>>& rankIdToChannelInfo) const
{
    // ✅ 新增维测日志：函数入口
    HCCL_INFO("[InsV2AllGatherOmniPipeExecutor][RestoreChannelMap][rank:%u] 开始恢复通道映射", myRank_);
    
    rankIdToChannelInfo.resize(OMNIPIPE_LEVEL_NUM);
    u32 level = 0;
    for (u32 i = 0; i < OMNIPIPE_LEVEL_NUM; i++) {
        if (rankSizeLevel_[i] > 1) {
            for (auto& channel : resCtx.channels[level]) {
                u32 remoteRank = channel.remoteRank;
                rankIdToChannelInfo[i][remoteRank].push_back(channel);
            }
            // ✅ 新增维测日志：层级通道数
            HCCL_INFO("[InsV2AllGatherOmniPipeExecutor][RestoreChannelMap][rank:%u] 层级[%u] 通道数:%zu",
                      myRank_, i, rankIdToChannelInfo[i].size());
            level++;
        }
    }
    
    // ✅ 新增维测日志：函数出口
    HCCL_INFO("[InsV2AllGatherOmniPipeExecutor][RestoreChannelMap][rank:%u] 通道映射恢复完成", myRank_);
    
    return HCCL_SUCCESS;
}

REGISTER_EXEC_V2_MULTI(HcclCMDType::HCCL_CMD_ALLGATHER, InsV2AllGatherOmniPipeMultilevel,
                       InsV2AllGatherOmniPipeExecutor, TopoMatchMultilevel, InsTempAllGatherOmniPipeMesh1D,
                       InsTempAllGatherOmniPipeNHR, InsTempAllGatherOmniPipeNHRDPU);
REGISTER_EXEC_V2_MULTI(HcclCMDType::HCCL_CMD_ALLGATHER, InsV2AllGatherOmniPipePcie,
                       InsV2AllGatherOmniPipeExecutor, TopoMatchPcieMix, InsTempAllGatherOmniPipeMesh1D,
                       InsTempAllGatherOmniPipeNHR, InsTempAllGatherOmniPipeNHRDPU);
 REGISTER_EXEC_V2_MULTI(HcclCMDType::HCCL_CMD_ALLGATHER, InsV2AllGatherOmniPipe,	 
                        InsV2AllGatherOmniPipeExecutor, TopoMatchUBX, InsTempAllGatherOmniPipeMesh1D,	 
                        InsTempAllGatherOmniPipeNHR, InsTempAllGatherOmniPipeNHRDPU);

REGISTER_EXEC_V2_MULTI(HcclCMDType::HCCL_CMD_ALLGATHER, InsTestUboeAlgorithm,
                       InsV2AllGatherOmniPipeExecutor, TopoMatch3Level, InsTempAllGatherOmniPipeMesh1D,
                       InsTempAllGatherOmniPipeNHR, InsTempAllGatherOmniPipeNHR);
}  // namespace ops_hccl