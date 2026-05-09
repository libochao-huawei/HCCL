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
#include "../../../reduce_scatter/template/ccu/ccu_temp_reduce_scatter_mesh_1D.h"
#include "ccu_temp_gather_omnipipe_mesh_1d.h"
#include "omnipipe_data_slice_calc.h"

namespace ops_hccl {

template <typename AlgTopoMatch, typename InsRsAlgTemplateX, typename InsRsAlgTemplateY, 
          typename InsGatherAlgTemplateX, typename InsGatherAlgTemplateY>
CcuV2ReduceOmniPipeExecutor<AlgTopoMatch, InsRsAlgTemplateX, InsRsAlgTemplateY, 
                           InsGatherAlgTemplateX, InsGatherAlgTemplateY>::CcuV2ReduceOmniPipeExecutor()
{
}

template <typename AlgTopoMatch, typename InsRsAlgTemplateX, typename InsRsAlgTemplateY, 
          typename InsGatherAlgTemplateX, typename InsGatherAlgTemplateY>
HcclResult CcuV2ReduceOmniPipeExecutor<AlgTopoMatch, InsRsAlgTemplateX, InsRsAlgTemplateY,
                                      InsGatherAlgTemplateX, InsGatherAlgTemplateY>::CalcAlgHierarchyInfo(
    HcclComm comm, TopoInfoWithNetLayerDetails *topoInfo, AlgHierarchyInfoForAllLevel &algHierarchyInfo)
{
    CHK_PTR_NULL(topoInfo);
    AlgTopoMatch topoMatch;
    CHK_RET(topoMatch.MatchTopo(comm, topoInfo, algHierarchyInfo));
    return HCCL_SUCCESS;
}

template <typename AlgTopoMatch, typename InsRsAlgTemplateX, typename InsRsAlgTemplateY, 
          typename InsGatherAlgTemplateX, typename InsGatherAlgTemplateY>
HcclResult CcuV2ReduceOmniPipeExecutor<AlgTopoMatch, InsRsAlgTemplateX, InsRsAlgTemplateY,
                                      InsGatherAlgTemplateX, InsGatherAlgTemplateY>::InitCommInfo(
    const OpParam &param, const TopoInfoWithNetLayerDetails *topoInfo, 
    const AlgHierarchyInfoForAllLevel &algHierarchyInfo)
{
    myRank_ = topoInfo->userRank;
    rankSize_ = topoInfo->userRankSize;
    devType_ = topoInfo->deviceType;
    reduceOp_ = param.reduceType;
    dataType_ = param.DataDes.dataType;
    dataCount_ = param.DataDes.count;
    dataTypeSize_ = SIZE_TABLE[param.DataDes.dataType];
    rootRank_ = param.root;
    return HCCL_SUCCESS;
}

template <typename AlgTopoMatch, typename InsRsAlgTemplateX, typename InsRsAlgTemplateY, 
          typename InsGatherAlgTemplateX, typename InsGatherAlgTemplateY>
HcclResult CcuV2ReduceOmniPipeExecutor<AlgTopoMatch, InsRsAlgTemplateX, InsRsAlgTemplateY,
                                      InsGatherAlgTemplateX, InsGatherAlgTemplateY>::CalcResLevel(
    HcclComm comm, const OpParam& param, const TopoInfoWithNetLayerDetails* topoInfo,
    std::shared_ptr<CcuAlgTemplateBase> tempAlg, AlgResourceRequest& resourceReq, const int& curLevel)
{
    AlgResourceRequest resReqlevel;
    CHK_RET(tempAlg->CalcRes(comm, param, topoInfo, resReqlevel));
    resourceReq.slaveThreadNum += resReqlevel.slaveThreadNum + 1;
    resourceReq.notifyNumOnMainThread += 1;
    resourceReq.notifyNumPerThread.emplace_back(resReqlevel.notifyNumOnMainThread + 1);
    resourceReq.notifyNumPerThread.insert(resourceReq.notifyNumPerThread.end(),
                                          resReqlevel.notifyNumPerThread.begin(),
                                          resReqlevel.notifyNumPerThread.end());
    
    if (curLevel < OMNIPIPE_GATHER_LEVEL0) {
        resourceReq.channels.emplace_back(resReqlevel.channels[0]);
    }
    
    return HCCL_SUCCESS;
}

template <typename AlgTopoMatch, typename InsRsAlgTemplateX, typename InsRsAlgTemplateY, 
          typename InsGatherAlgTemplateX, typename InsGatherAlgTemplateY>
HcclResult CcuV2ReduceOmniPipeExecutor<AlgTopoMatch, InsRsAlgTemplateX, InsRsAlgTemplateY,
                                      InsGatherAlgTemplateX, InsGatherAlgTemplateY>::InitSubCommRanks(
    std::vector<std::vector<u32>>& subCommRanks0, std::vector<std::vector<u32>>& subCommRanks1,
    const AlgHierarchyInfoForAllLevel& algHierarchyInfo)
{
    subCommRanks0.clear();
    subCommRanks1.clear();
    
    subCommRanks0.push_back(algHierarchyInfo.infos[0][0]);
    
    u32 intraLevel0DeviceNum = algHierarchyInfo.infos[0][0].size();
    u32 intraSuperPodDeviceNum = algHierarchyInfo.infos[0][1].size();
    
    subCommRanks1.resize(1);
    for (u32 i = myRank_ % intraLevel0DeviceNum; i < intraSuperPodDeviceNum; i += intraLevel0DeviceNum) {
        subCommRanks1[0].push_back(algHierarchyInfo.infos[0][1][i]);
    }
    
    return HCCL_SUCCESS;
}

template <typename AlgTopoMatch, typename InsRsAlgTemplateX, typename InsRsAlgTemplateY, 
          typename InsGatherAlgTemplateX, typename InsGatherAlgTemplateY>
HcclResult CcuV2ReduceOmniPipeExecutor<AlgTopoMatch, InsRsAlgTemplateX, InsRsAlgTemplateY,
                                      InsGatherAlgTemplateX, InsGatherAlgTemplateY>::InitTemplate(
    const OpParam& param, std::map<u32, std::shared_ptr<CcuAlgTemplateBase>>& tempMap,
    const std::vector<std::vector<u32>>& subCommRanks0, const std::vector<std::vector<u32>>& subCommRanks1)
{
    if (rankSizeLevel0_ > 1) {
        tempMap[OMNIPIPE_RS_LEVEL0] = std::make_shared<InsRsAlgTemplateX>(param, myRank_, subCommRanks0);
        tempMap[OMNIPIPE_GATHER_LEVEL0] = std::make_shared<InsGatherAlgTemplateX>(param, myRank_, subCommRanks0);
    }
    
    if (rankSizeLevel1_ > 1) {
        tempMap[OMNIPIPE_RS_LEVEL1] = std::make_shared<InsRsAlgTemplateY>(param, myRank_, subCommRanks1);
        tempMap[OMNIPIPE_GATHER_LEVEL1] = std::make_shared<InsGatherAlgTemplateY>(param, myRank_, subCommRanks1);
    }
    
    HCCL_DEBUG("[CcuV2ReduceOmniPipeExecutor][InitTemplate] tempMap.size[%u]", tempMap.size());
    
    return HCCL_SUCCESS;
}

template <typename AlgTopoMatch, typename InsRsAlgTemplateX, typename InsRsAlgTemplateY, 
          typename InsGatherAlgTemplateX, typename InsGatherAlgTemplateY>
HcclResult CcuV2ReduceOmniPipeExecutor<AlgTopoMatch, InsRsAlgTemplateX, InsRsAlgTemplateY,
                                      InsGatherAlgTemplateX, InsGatherAlgTemplateY>::CalcRes(
    HcclComm comm, const OpParam &param, const TopoInfoWithNetLayerDetails *topoInfo,
    const AlgHierarchyInfoForAllLevel &algHierarchyInfo, AlgResourceRequest &resourceRequest)
{
    auto topoinfo_local = const_cast<TopoInfoWithNetLayerDetails*>(topoInfo);
    auto& algHierarchyInfo_local = const_cast<AlgHierarchyInfoForAllLevel&>(algHierarchyInfo);
    
    InitCommInfo(param, topoinfo_local, algHierarchyInfo_local);
    
    rankSizeLevel0_ = algHierarchyInfo.infos[0][0].size();
    rankSizeLevel1_ = algHierarchyInfo.infos[0][1].size() / rankSizeLevel0_;
    
    HCCL_DEBUG("[CcuV2ReduceOmniPipeExecutor] L0[%u], L1[%u], root[%u]", 
        rankSizeLevel0_, rankSizeLevel1_, rootRank_);
    
    uint32_t intraSuperPodDeviceNum = rankSizeLevel0_ * rankSizeLevel1_;
    
    rankIdxLevel0_ = (myRank_ % intraSuperPodDeviceNum) % rankSizeLevel0_;
    rankIdxLevel1_ = (myRank_ % intraSuperPodDeviceNum) / rankSizeLevel0_;
    
    rootIdxLevel0_ = (rootRank_ % intraSuperPodDeviceNum) % rankSizeLevel0_;
    rootIdxLevel1_ = (rootRank_ % intraSuperPodDeviceNum) / rankSizeLevel0_;
    
    std::vector<std::vector<u32>> subCommRanks0;
    std::vector<std::vector<u32>> subCommRanks1;
    InitSubCommRanks(subCommRanks0, subCommRanks1, algHierarchyInfo);
    
    std::map<u32, std::shared_ptr<CcuAlgTemplateBase>> tempMap;
    InitTemplate(param, tempMap, subCommRanks0, subCommRanks1);
    
    resourceRequest.slaveThreadNum = 0;
    resourceRequest.notifyNumOnMainThread = 0;
    
    for (int level = 0; level < OMNIPIPE_REDUCE_LEVEL_NUM; level++) {
        if (tempMap.count(level) > 0) {
            CHK_RET(CalcResLevel(comm, param, topoInfo, tempMap[level], resourceRequest, level));
        }
    }
    
    HCCL_DEBUG("[CcuV2ReduceOmniPipeExecutor][CalcRes] slaveThreadNum[%u], channels[%u]",
        resourceRequest.slaveThreadNum, resourceRequest.channels.size());
    
    return HCCL_SUCCESS;
}

template <typename AlgTopoMatch, typename InsRsAlgTemplateX, typename InsRsAlgTemplateY, 
          typename InsGatherAlgTemplateX, typename InsGatherAlgTemplateY>
HcclResult CcuV2ReduceOmniPipeExecutor<AlgTopoMatch, InsRsAlgTemplateX, InsRsAlgTemplateY,
                                      InsGatherAlgTemplateX, InsGatherAlgTemplateY>::Orchestrate(
    const OpParam &param, const AlgResourceCtxSerializable &resCtx)
{
    HCCL_INFO("[CcuV2ReduceOmniPipeExecutor][Orchestrate] Start");
    
    threads_ = resCtx.threads;
    controlThread_ = threads_.at(0);
    myRank_ = resCtx.topoInfo.userRank;
    rankSize_ = resCtx.topoInfo.userRankSize;
    rootRank_ = param.root;
    dataCount_ = param.DataDes.count;
    dataTypeSize_ = SIZE_TABLE[param.DataDes.dataType];
    dataSize_ = dataCount_ * dataTypeSize_;
    
    rankSizeLevel0_ = resCtx.algHierarchyInfo.infos[0][0].size();
    rankSizeLevel1_ = resCtx.algHierarchyInfo.infos[0][1].size() / rankSizeLevel0_;
    
    uint32_t intraSuperPodDeviceNum = rankSizeLevel0_ * rankSizeLevel1_;
    rankIdxLevel0_ = (myRank_ % intraSuperPodDeviceNum) % rankSizeLevel0_;
    rankIdxLevel1_ = (myRank_ % intraSuperPodDeviceNum) / rankSizeLevel0_;
    
    rootIdxLevel0_ = (rootRank_ % intraSuperPodDeviceNum) % rankSizeLevel0_;
    rootIdxLevel1_ = (rootRank_ % intraSuperPodDeviceNum) / rankSizeLevel0_;
    
    HCCL_ERROR("[CcuV2ReduceOmniPipeExecutor] threads[%u], dataTypeSize[%u], dataCount[%u], "
        "L0[%u], L1[%u], root[%u]",
        threads_.size(), dataTypeSize_, dataCount_, rankSizeLevel0_, rankSizeLevel1_, rootRank_);
    
    HcclResult ret = OrchestrateLoop(param, resCtx);
    CHK_PRT_RET(ret != HCCL_SUCCESS,
        HCCL_ERROR("[CcuV2ReduceOmniPipeExecutor][Orchestrate] errNo[0x%016llx] kernel run failed",
            HCCL_ERROR_CODE(ret)), ret);
    
    HCCL_INFO("[CcuV2ReduceOmniPipeExecutor][Orchestrate] End");
    return HCCL_SUCCESS;
}

template <typename AlgTopoMatch, typename InsRsAlgTemplateX, typename InsRsAlgTemplateY, 
          typename InsGatherAlgTemplateX, typename InsGatherAlgTemplateY>
HcclResult CcuV2ReduceOmniPipeExecutor<AlgTopoMatch, InsRsAlgTemplateX, InsRsAlgTemplateY,
                                      InsGatherAlgTemplateX, InsGatherAlgTemplateY>::GenTemplateAlgParamsByDimData(
    TemplateDataParams &tempAlgParams, StepSliceInfo &stepSliceInfo, u64 processedDataCount)
{
    tempAlgParams.buffInfo.inBuffType = BufferType::HCCL_BUFFER;
    tempAlgParams.buffInfo.outBuffType = BufferType::HCCL_BUFFER;
    
    tempAlgParams.buffInfo.inBuffBaseOff = stepSliceInfo.buffInfo.inBuffBaseOff;
    tempAlgParams.buffInfo.outBuffBaseOff = stepSliceInfo.buffInfo.outBuffBaseOff;
    tempAlgParams.buffInfo.hcclBuffBaseOff = stepSliceInfo.buffInfo.hcclBuffBaseOff;
    
    tempAlgParams.stepSliceInfo = stepSliceInfo;
    return HCCL_SUCCESS;
}

template <typename AlgTopoMatch, typename InsRsAlgTemplateX, typename InsRsAlgTemplateY, 
          typename InsGatherAlgTemplateX, typename InsGatherAlgTemplateY>
HcclResult CcuV2ReduceOmniPipeExecutor<AlgTopoMatch, InsRsAlgTemplateX, InsRsAlgTemplateY,
                                      InsGatherAlgTemplateX, InsGatherAlgTemplateY>::InitOmniPipeScratchParam(
    OmniPipeScratchParam& scratchParam, const OpParam& param,
    const std::vector<double>& endpointAttrBwAvg,
    std::map<u32, std::shared_ptr<CcuAlgTemplateBase>>& tempMap)
{
    std::vector<u64> levelRankSizeVec = {rankSizeLevel0_, rankSizeLevel1_};
    scratchParam.levelRankSize = levelRankSizeVec;
    scratchParam.endpointAttrBw = endpointAttrBwAvg;
    scratchParam.dataTypeSize = dataTypeSize_;
    scratchParam.opMode = param.opMode;
    scratchParam.engine = param.engine;
    return HCCL_SUCCESS;
}

template <typename AlgTopoMatch, typename InsRsAlgTemplateX, typename InsRsAlgTemplateY, 
          typename InsGatherAlgTemplateX, typename InsGatherAlgTemplateY>
HcclResult CcuV2ReduceOmniPipeExecutor<AlgTopoMatch, InsRsAlgTemplateX, InsRsAlgTemplateY,
                                      InsGatherAlgTemplateX, InsGatherAlgTemplateY>::InitOmniPipeSliceParam(
    OmniPipeSliceParam& sliceParam, const OpParam& param,
    const std::vector<double>& endpointAttrBwAvg,
    std::map<u32, std::shared_ptr<CcuAlgTemplateBase>>& tempMap)
{
    std::vector<u64> levelRankSizeVec = {rankSizeLevel0_, rankSizeLevel1_};
    std::vector<u64> levelRankIdVec = {rankIdxLevel0_, rankIdxLevel1_};
    
    sliceParam.endpointAttrBw = endpointAttrBwAvg;
    sliceParam.levelRankSize = levelRankSizeVec;
    sliceParam.levelRankId = levelRankIdVec;
    sliceParam.dataTypeSize = dataTypeSize_;
    sliceParam.opMode = param.opMode;
    sliceParam.engine = param.engine;
    return HCCL_SUCCESS;
}

template <typename AlgTopoMatch, typename InsRsAlgTemplateX, typename InsRsAlgTemplateY, 
          typename InsGatherAlgTemplateX, typename InsGatherAlgTemplateY>
HcclResult CcuV2ReduceOmniPipeExecutor<AlgTopoMatch, InsRsAlgTemplateX, InsRsAlgTemplateY,
                                      InsGatherAlgTemplateX, InsGatherAlgTemplateY>::InitTemplateParams(
    const OpParam &param, const AlgResourceCtxSerializable &resCtx,
    const std::map<u32, std::shared_ptr<CcuAlgTemplateBase>>& tempMap,
    std::map<u32, TemplateResource>& tempResMap,
    std::map<u32, TemplateDataParams>& tempAlgParamMap)
{
    for (int level = 0; level < OMNIPIPE_REDUCE_LEVEL_NUM; level++) {
        if (tempMap.count(level) > 0) {
            if (level < OMNIPIPE_GATHER_LEVEL0) {
                tempResMap[level].threads = levelThreads_[level];
                tempResMap[level].channels = remoteRankToChannelInfo_[level];
            } else {
                tempResMap[level].threads = levelThreadsGather_[level - OMNIPIPE_GATHER_LEVEL0];
                tempResMap[level].channels = remoteRankToChannelInfo_[level - OMNIPIPE_GATHER_LEVEL0];
            }
            
            tempAlgParamMap[level].buffInfo.inputPtr = param.inputPtr;
            tempAlgParamMap[level].buffInfo.outputPtr = param.outputPtr;
            tempAlgParamMap[level].buffInfo.hcclBuff = resCtx.cclMem;
        }
    }
    return HCCL_SUCCESS;
}

template <typename AlgTopoMatch, typename InsRsAlgTemplateX, typename InsRsAlgTemplateY, 
          typename InsGatherAlgTemplateX, typename InsGatherAlgTemplateY>
HcclResult CcuV2ReduceOmniPipeExecutor<AlgTopoMatch, InsRsAlgTemplateX, InsRsAlgTemplateY,
                                      InsGatherAlgTemplateX, InsGatherAlgTemplateY>::OrchestrateLoop(
    const OpParam &param, const AlgResourceCtxSerializable& resCtx)
{
    std::vector<std::vector<u32>> subCommRanks0;
    std::vector<std::vector<u32>> subCommRanks1;
    InitSubCommRanks(subCommRanks0, subCommRanks1, resCtx.algHierarchyInfo);
    
    std::map<u32, std::shared_ptr<CcuAlgTemplateBase>> tempMap;
    InitTemplate(param, tempMap, subCommRanks0, subCommRanks1);
    
    OmniPipeScratchParam scratchParam;
    InitOmniPipeScratchParam(scratchParam, param, {}, tempMap);
    scratchParam.maxTmpMemSize = resCtx.cclMem.size;
    
    auto allRankSplitData = OmniPipeSplitData(rankSize_, dataCount_, dataTypeSize_);
    scratchParam.dataSize = CalcCountToDataSize(allRankSplitData, dataTypeSize_);
    
    std::vector<u64> loopInfo = CalcOmniPipeScratchInfo(scratchParam);
    u64 maxCountPerLoop = loopInfo[0];
    u64 loopTimes = loopInfo[1];
    
    OmniPipeSliceParam sliceParam;
    InitOmniPipeSliceParam(sliceParam, param, {}, tempMap);
    
    OmniPipeSliceInfo OmniPipeSliceInfoRS;
    OmniPipeSliceInfo OmniPipeSliceInfoGather;
    
    for (u64 loop = 0; loop < loopTimes; loop++) {
        sliceParam.dataSizePerLoop = dataSize_;
        sliceParam.dataWholeSize = dataSize_;
        OmniPipeSliceInfoRS = CalcRSOmniPipeSliceInfo(sliceParam);
        OmniPipeSliceInfoGather = CalcGatherOmniPipeSliceInfo(sliceParam);
        
        TemplateDataParams tempParamLocalCopy;
        tempParamLocalCopy.buffInfo.hcclBuff = resCtx.cclMem;
        tempParamLocalCopy.buffInfo.inputPtr = param.inputPtr;
        tempParamLocalCopy.buffInfo.outputPtr = param.outputPtr;
        tempParamLocalCopy.localCopyFlag = 1;
        
        u32 intraPodStepNum = OmniPipeSliceInfoRS.dataSliceLevel0.size();
        
        for (u32 stepXY = 0; stepXY < intraPodStepNum; stepXY++) {
            if (rankSizeLevel0_ > 1) {
                GenTemplateAlgParamsByDimData(tempParamLocalCopy, 
                    OmniPipeSliceInfoRS.dataSliceLevel0[stepXY], loop * maxCountPerLoop);
                CHK_RET(tempMap[OMNIPIPE_RS_LEVEL0]->KernelRun(param, tempParamLocalCopy, 
                    TemplateResource()));
            }
            
            if (rankSizeLevel1_ > 1) {
                GenTemplateAlgParamsByDimData(tempParamLocalCopy,
                    OmniPipeSliceInfoRS.dataSliceLevel1[stepXY], loop * maxCountPerLoop);
                CHK_RET(tempMap[OMNIPIPE_RS_LEVEL1]->KernelRun(param, tempParamLocalCopy,
                    TemplateResource()));
            }
        }
        
        intraPodStepNum = OmniPipeSliceInfoGather.dataSliceLevel0.size();
        
        for (u32 stepXY = 0; stepXY < intraPodStepNum; stepXY++) {
            if (rankSizeLevel0_ > 1) {
                GenTemplateAlgParamsByDimData(tempParamLocalCopy,
                    OmniPipeSliceInfoGather.dataSliceLevel0[stepXY], loop * maxCountPerLoop);
                CHK_RET(tempMap[OMNIPIPE_GATHER_LEVEL0]->KernelRun(param, tempParamLocalCopy,
                    TemplateResource()));
            }
            
            if (rankSizeLevel1_ > 1) {
                GenTemplateAlgParamsByDimData(tempParamLocalCopy,
                    OmniPipeSliceInfoGather.dataSliceLevel1[stepXY], loop * maxCountPerLoop);
                CHK_RET(tempMap[OMNIPIPE_GATHER_LEVEL1]->KernelRun(param, tempParamLocalCopy,
                    TemplateResource()));
            }
        }
        
        if (myRank_ == rootRank_) {
            TemplateDataParams finalParams;
            finalParams.buffInfo.hcclBuff = resCtx.cclMem;
            finalParams.buffInfo.inputPtr = param.inputPtr;
            finalParams.buffInfo.outputPtr = param.outputPtr;
            finalParams.buffInfo.inBuffType = BufferType::HCCL_BUFFER;
            finalParams.buffInfo.outBuffType = BufferType::OUTPUT;
            finalParams.buffInfo.inBuffBaseOff = 0;
            finalParams.buffInfo.outBuffBaseOff = 0;
            finalParams.localCopyFlag = 1;
            finalParams.sliceSize = dataSize_;
            
            HCCL_DEBUG("[CcuV2ReduceOmniPipeExecutor] Root final copy to output");
        }
    }
    
    HCCL_INFO("[CcuV2ReduceOmniPipeExecutor][OrchestrateLoop] End");
    return HCCL_SUCCESS;
}

REGISTER_EXEC_V2(HcclCMDType::HCCL_CMD_REDUCE, CcuV2ReduceOmniPipe2D, CcuV2ReduceOmniPipeExecutor,
    TopoMatchUBX, CcuTempReduceScatterMesh1D, CcuTempReduceScatterMesh1D, 
    CcuTempGatherOmniPipeMesh1D, CcuTempGatherOmniPipeMesh1D);

}