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
#include "ins_temp_all_gather_mesh_1D_opt.h"
#include "ins_temp_all_gather_mesh_clos_opt.h"

#include "topo_match_ubx_v3.h"

namespace ops_hccl {

template <typename AlgTopoMatch, typename InsAlgTemplate0, typename InsAlgTemplate1>
InsV2AllGatherParallelOptExecutor<AlgTopoMatch, InsAlgTemplate0, InsAlgTemplate1>::InsV2AllGatherParallelOptExecutor()
{
}

template <typename AlgTopoMatch, typename InsAlgTemplate0, typename InsAlgTemplate1>
HcclResult InsV2AllGatherParallelOptExecutor<AlgTopoMatch, InsAlgTemplate0, InsAlgTemplate1>::CalcAlgHierarchyInfo(
    HcclComm comm, TopoInfoWithNetLayerDetails *topoInfo, AlgHierarchyInfoForAllLevel &algHierarchyInfo)
{
    AlgTopoMatch topoMatch;
    CHK_RET(topoMatch.MatchTopo(comm, topoInfo, algHierarchyInfo));
    return HCCL_SUCCESS;
}

template <typename AlgTopoMatch, typename InsAlgTemplate0, typename InsAlgTemplate1>
HcclResult InsV2AllGatherParallelOptExecutor<AlgTopoMatch, InsAlgTemplate0, InsAlgTemplate1>::CalcRes(
    HcclComm comm, const OpParam &param, const TopoInfoWithNetLayerDetails *topoInfo, const AlgHierarchyInfoForAllLevel &algHierarchyInfo,
    AlgResourceRequest &resourceRequest)
{
    myRank_ = topoInfo->userRank;
    // 构建template
    std::vector<std::vector<u32>> intraHierarchyInfo;
    std::vector<std::vector<u32>> interHierarchyInfo;
    if(topoInfo->level0Topo == Level0Shape::MESH_1D_CLOS && !topoInfo->level0PcieMix) {
        intraHierarchyInfo = {algHierarchyInfo.infos[0][0]};
        std::vector<u32> closRanks;
        u32 meshSize = algHierarchyInfo.infos[0][0].size();
        for(auto rank : algHierarchyInfo.infos[0][1]) {
            closRanks.push_back(rank);
        }
        interHierarchyInfo = {closRanks};
    } else {
        constexpr u32 TOPO_NUM = 2;
        CHK_PRT_RET(algHierarchyInfo.infos.size() < TOPO_NUM || algHierarchyInfo.infos[0].empty() || algHierarchyInfo.infos[1].empty(),
                     HCCL_ERROR("[InsAllGatherParallelExecutor][CalcRes] Invalid topoInfo"),
                     HcclResult::HCCL_E_INTERNAL);
        intraHierarchyInfo = algHierarchyInfo.infos[0];
        interHierarchyInfo = algHierarchyInfo.infos[1];
    }

    InsAlgTemplate0 intraTempAlg(param, topoInfo->userRank, intraHierarchyInfo);
    InsAlgTemplate1 interTempAlg(param, topoInfo->userRank, interHierarchyInfo);
    
    // 调用计算资源的函数
    AlgResourceRequest intraTempRequest;
    AlgResourceRequest interTempRequest;
    CHK_RET(intraTempAlg.CalcRes(comm, param, topoInfo, intraTempRequest));
    CHK_RET(interTempAlg.CalcRes(comm, param, topoInfo, interTempRequest));
    constexpr u32 SUB_MAIN_THREAD_NUM = 2;
    resourceRequest.notifyNumOnMainThread = SUB_MAIN_THREAD_NUM;  // 用于两个template间同步
    resourceRequest.slaveThreadNum = intraTempRequest.slaveThreadNum + interTempRequest.slaveThreadNum + SUB_MAIN_THREAD_NUM;
    resourceRequest.notifyNumPerThread.emplace_back(intraTempRequest.notifyNumOnMainThread + 1);
    resourceRequest.notifyNumPerThread.insert(resourceRequest.notifyNumPerThread.end(),
                                              intraTempRequest.notifyNumPerThread.begin(),
                                              intraTempRequest.notifyNumPerThread.end());
    resourceRequest.notifyNumPerThread.emplace_back(interTempRequest.notifyNumOnMainThread + 1);
    resourceRequest.notifyNumPerThread.insert(resourceRequest.notifyNumPerThread.end(),
                                              interTempRequest.notifyNumPerThread.begin(),
                                              interTempRequest.notifyNumPerThread.end());
    if (param.engine != COMM_ENGINE_CCU) {
        CHK_PRT_RET(intraTempRequest.channels.empty() || interTempRequest.channels.empty(),
                     HCCL_ERROR("[InsAllGatherParallelExecutor][CalcRes] intraTemplate or interTemplate has empty channels."),
                     HcclResult::HCCL_E_INTERNAL);
        resourceRequest.channels.emplace_back(intraTempRequest.channels[0]);
        resourceRequest.channels.emplace_back(interTempRequest.channels[0]);
    } else {
        // ccu
        HCCL_INFO("[InsAllGatherParallelExecutor][CalcRes] intraTemplate has [%d] kernels.", intraTempRequest.ccuKernelNum[0]);
        resourceRequest.ccuKernelInfos.insert(resourceRequest.ccuKernelInfos.end(),
                                            intraTempRequest.ccuKernelInfos.begin(),
                                            intraTempRequest.ccuKernelInfos.end());
        resourceRequest.ccuKernelNum.emplace_back(intraTempRequest.ccuKernelNum[0]);
        HCCL_INFO("[InsAllGatherParallelExecutor][CalcRes] interTemplate has [%d] kernels.", interTempRequest.ccuKernelNum[0]);
        resourceRequest.ccuKernelInfos.insert(resourceRequest.ccuKernelInfos.end(),
                                            interTempRequest.ccuKernelInfos.begin(),
                                            interTempRequest.ccuKernelInfos.end());
        resourceRequest.ccuKernelNum.emplace_back(interTempRequest.ccuKernelNum[0]);
    }
    HCCL_DEBUG("[InsV2AllGatherParallelOptExecutor][CalcRes] myRank[%u], notifyNumOnMainThread[%u], slaveThreadNum[%u], "
               "channels[%u]",
               myRank_, resourceRequest.notifyNumOnMainThread, resourceRequest.slaveThreadNum,
               resourceRequest.channels.size());
    for (auto i = 0; i < resourceRequest.notifyNumPerThread.size(); i++) {
        HCCL_DEBUG("[InsV2AllGatherParallelOptExecutor][CalcRes] myRank[%u], notifyNumPerThread[%u]=[%u]", myRank_, i,
                   resourceRequest.notifyNumPerThread[i]);
    }

    return HCCL_SUCCESS;
}

template <typename AlgTopoMatch, typename InsAlgTemplate0, typename InsAlgTemplate1>
uint64_t InsV2AllGatherParallelOptExecutor<AlgTopoMatch, InsAlgTemplate0, InsAlgTemplate1>::GetRankSize(
    const std::vector<std::vector<u32>> &vTopo) const
{
    uint64_t count = 1;
    for (const auto &i : vTopo) {
        count *= i.size();
    }
    return count;
}

template <typename AlgTopoMatch, typename InsAlgTemplate0, typename InsAlgTemplate1>
HcclResult InsV2AllGatherParallelOptExecutor<AlgTopoMatch, InsAlgTemplate0, InsAlgTemplate1>::Orchestrate(
    const OpParam &param, const AlgResourceCtxSerializable &resCtx)
{
    HCCL_INFO("[InsV2AllGatherParallelOptExecutor][Orchestrate] Orchestrate Start");
    maxTmpMemSize_ = resCtx.cclMem.size;  // maxTmpMemSize_设定为cclIn的大小，op中将申请的HcclBuff全给了cclIn
    myRank_ = resCtx.topoInfo.userRank;
    // 给channels_和threads_赋值
    threads_ = resCtx.threads;
    if (param.engine != CommEngine::COMM_ENGINE_AIV && param.engine != CommEngine::COMM_ENGINE_CCU) {
        CHK_RET(RestoreChannelMap(resCtx, remoteRankToChannelInfo_));
        intraLinkMap_ = remoteRankToChannelInfo_[0];
        interLinkMap_ = remoteRankToChannelInfo_[1];

        // ========== 新增日志 ==========
        HCCL_INFO("[Orchestrate] interLinkMap_ size=%zu", interLinkMap_.size());
        for (auto &kv : interLinkMap_) {
            HCCL_INFO("[Orchestrate] interLinkMap_ size=%zu", interLinkMap_.size());
        }
        for (auto &kv : interLinkMap_) {
            HCCL_INFO("[Orchestrate] interLinkMap_ rank=%u channels=%zu", kv.first, kv.second.size());
        }
        // ============================
    }
    dataCount_ = param.DataDes.count;
    dataType_ = param.DataDes.dataType;
    dataTypeSize_ = DATATYPE_SIZE_TABLE[param.DataDes.dataType];
    dataSize_ = dataCount_ * dataTypeSize_;

    if(resCtx.topoInfo.level0Topo == Level0Shape::MESH_1D_CLOS && !resCtx.topoInfo.level0PcieMix) {
        intraHierarchyInfo_ = {resCtx.algHierarchyInfo.infos[0][0]};
        std::vector<u32> closRanks;
        u32 meshSize = resCtx.algHierarchyInfo.infos[0][0].size();
        for(auto rank : resCtx.algHierarchyInfo.infos[0][1]) { 
            closRanks.push_back(rank);   
        }
        interHierarchyInfo_ = {closRanks};
    } else {
        intraHierarchyInfo_ = resCtx.algHierarchyInfo.infos[0];
        interHierarchyInfo_ = resCtx.algHierarchyInfo.infos[1];
    }
    rankSizeLevel0_ = GetRankSize(intraHierarchyInfo_);
    rankSizeLevel1_ = GetRankSize(interHierarchyInfo_);
    
    rankIdxLevel0_ = myRank_ % rankSizeLevel0_;
    rankIdxLevel1_ = myRank_;
    // 实例化算法模板类
    // 构建template
    InsAlgTemplate0 intraTempAlg(param, resCtx.topoInfo.userRank, intraHierarchyInfo_);
    InsAlgTemplate1 interTempAlg(param, resCtx.topoInfo.userRank, interHierarchyInfo_);
    if (param.engine == CommEngine::COMM_ENGINE_AICPU_TS) {
        interTempAlg.SetchannelsPerRank(interLinkMap_);
    }
    // 将计算资源分配个每个算法
    PrepareResForTemplate(intraTempAlg, interTempAlg);
    // 算法展开

    HcclResult ret = OrchestrateLoop(param, resCtx, intraTempAlg, interTempAlg);
    CHK_PRT_RET(
        ret != HCCL_SUCCESS,
        HCCL_ERROR("[InsV2AllGatherParallelOptExecutor][Orchestrate]errNo[0x%016llx] All Gather excutor kernel run failed",
                   HCCL_ERROR_CODE(ret)),
        ret);
    return HCCL_SUCCESS;
}

template <typename AlgTopoMatch, typename InsAlgTemplate0, typename InsAlgTemplate1>
HcclResult InsV2AllGatherParallelOptExecutor<AlgTopoMatch, InsAlgTemplate0, InsAlgTemplate1>::PrepareResForTemplate(
    InsAlgTemplate0 &tempAlgIntra, InsAlgTemplate1 &tempAlgInter)
{
    AlgResourceRequest intraTempRequest;
    AlgResourceRequest interTempRequest;
    tempAlgIntra.GetRes(intraTempRequest);
    tempAlgInter.GetRes(interTempRequest);
    auto intraThreadsNum = intraTempRequest.slaveThreadNum + 1;
    auto interThreadsNum = intraTempRequest.slaveThreadNum + 1;
    auto intraNotifyOnMainThread = intraTempRequest.notifyNumOnMainThread;
    auto interNotifyOnMainThread = interTempRequest.notifyNumOnMainThread;

    intraThreads_.assign(threads_.begin() + 1, threads_.begin() + intraThreadsNum + 1);
    interThreads_.assign(threads_.begin() + intraThreadsNum + 1, threads_.end());
    // 用于两个算法同步
    mainThread_ = threads_.at(0);
    templateMainThreads_.emplace_back(intraThreads_.at(0));
    templateMainThreads_.emplace_back(interThreads_.at(0));
    syncNotifyOnTemplates_ = {intraNotifyOnMainThread, interNotifyOnMainThread};
    syncNotifyOnMain_ = {0, 1};
    return HCCL_SUCCESS;
}

template <typename AlgTopoMatch, typename InsAlgTemplate0, typename InsAlgTemplate1>
void InsV2AllGatherParallelOptExecutor<AlgTopoMatch, InsAlgTemplate0, InsAlgTemplate1>::GetParallelDataSplit(
    std::vector<float> &splitDataSize) const
{
    // double splitData = multipleDimensionSplitRatio_;
    double splitData = 0.44;
    splitDataSize.push_back(splitData);
    splitDataSize.push_back(splitData);
    splitDataSize.push_back(1 - splitData - splitData);
    HCCL_INFO("[InsV2AllGatherParallelOptExecutor] splitDataSize is %f, %f, %f", splitDataSize[0], splitDataSize[1], splitDataSize[2]);
    return;
}

template <typename AlgTopoMatch, typename InsAlgTemplate0, typename InsAlgTemplate1>
HcclResult InsV2AllGatherParallelOptExecutor<AlgTopoMatch, InsAlgTemplate0, InsAlgTemplate1>::OrchestrateLoop(
    const OpParam &param, const AlgResourceCtxSerializable &resCtx, InsAlgTemplate0 &tempAlgIntra,
    InsAlgTemplate1 &tempAlgInter)
{
    HCCL_INFO("[InsV2AllGatherParallelOptExecutor] AlgTemplate intra server is [%s]", tempAlgIntra.Describe().c_str());
    HCCL_INFO("[InsV2AllGatherParallelOptExecutor] AlgTemplate inter server is [%s]", tempAlgInter.Describe().c_str());
    multipleDimensionSplitRatio_ = param.opConfig.multipleDimensionSplitRatio;
    std::vector<float> dataSplitSize;
    GetParallelDataSplit(dataSplitSize);

    u64 dataCountAxis0 = static_cast<u64>(dataSplitSize[0] * dataCount_);
    u64 dataCountAxis1 = static_cast<u64>(dataSplitSize[1] * dataCount_);
    u64 dataCountAxis2 = static_cast<u64>(dataCount_ - dataCountAxis0 - dataCountAxis1);

    u64 dataSizeAxis0 = dataCountAxis0 * dataTypeSize_;
    u64 dataSizeAxis1 = dataCountAxis1 * dataTypeSize_;
    u64 dataSizeAxis2 = dataCountAxis2 * dataTypeSize_;

    u64 rankSize = rankIdxLevel1_;

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
    TemplateDataParams tempAlgParamsAll0;

    // Stage 1
    TemplateDataParams tempAlgParamsIntra1;
    TemplateDataParams tempAlgParamsInter1;
    TemplateDataParams tempAlgParamsAll1;

    // Stage 0 远端写模式
    tempAlgIntra.SetRemoteWrite(true);
    tempAlgInter.SetRemoteWrite(true);

    // tempAlgParamsIntra0
    {
        tempAlgParamsIntra0.enableRemoteMemAccess = param.opMode == OpMode::OFFLOAD;

        tempAlgParamsIntra0.buffInfo.inputPtr = param.inputPtr;
        tempAlgParamsIntra0.buffInfo.inBuffType = BufferType::INPUT;
        tempAlgParamsIntra0.buffInfo.inputSize = dataSizeAxis0;
        tempAlgParamsIntra0.buffInfo.inBuffBaseOff = 0;
        tempAlgParamsIntra0.inputSliceStride = dataSize_;

        tempAlgParamsIntra0.repeatNum = 1;

        tempAlgParamsIntra0.buffInfo.hcclBuff = resCtx.cclMem;
        tempAlgParamsIntra0.buffInfo.hcclBuffType = BufferType::HCCL_BUFFER;
        tempAlgParamsIntra0.buffInfo.hcclBuffSize = dataSize_ * rankSize;
        tempAlgParamsIntra0.buffInfo.hcclBuffBaseOff = 0;
        tempAlgParamsIntra0.outputSliceStride = dataSize_;

        // 不需要 copy out
        tempAlgParamsIntra0.buffInfo.outputPtr = nullptr;

        tempAlgParamsIntra0.sliceSize = dataSizeAxis0;
        tempAlgParamsIntra0.count = dataCountAxis0;
    }

    {
        tempAlgParamsInter0.enableRemoteMemAccess = param.opMode == OpMode::OFFLOAD;

        tempAlgParamsInter0.buffInfo.inputPtr = param.inputPtr;
        tempAlgParamsInter0.buffInfo.inBuffType = BufferType::INPUT;
        tempAlgParamsInter0.buffInfo.inputSize = dataSizeAxis1;
        tempAlgParamsInter0.buffInfo.inBuffBaseOff = dataSizeAxis0;
        tempAlgParamsInter0.inputSliceStride = dataSize_;

        tempAlgParamsInter0.repeatNum = 1;

        tempAlgParamsInter0.buffInfo.hcclBuff = resCtx.cclMem;
        tempAlgParamsInter0.buffInfo.hcclBuffType = BufferType::HCCL_BUFFER;
        tempAlgParamsInter0.buffInfo.hcclBuffSize = dataSize_ * rankSize;
        tempAlgParamsInter0.buffInfo.hcclBuffBaseOff = dataSizeAxis0;
        tempAlgParamsIntra0.outputSliceStride = dataSize_;

        // 不需要 copy out
        tempAlgParamsInter0.buffInfo.outputPtr = nullptr;

        tempAlgParamsInter0.sliceSize = dataSizeAxis1;
        tempAlgParamsInter0.count = dataCountAxis1;
    }

    {
        tempAlgParamsAll0.enableRemoteMemAccess = param.opMode == OpMode::OFFLOAD;

        tempAlgParamsAll0.buffInfo.inputPtr = param.inputPtr;
        tempAlgParamsAll0.buffInfo.inBuffType = BufferType::INPUT;
        tempAlgParamsAll0.buffInfo.inputSize = dataSizeAxis2;
        tempAlgParamsAll0.buffInfo.inBuffBaseOff = dataSizeAxis0 + dataSizeAxis1;
        tempAlgParamsAll0.inputSliceStride = dataSize_;

        tempAlgParamsAll0.repeatNum = 4; // 先发4份

        tempAlgParamsAll0.buffInfo.hcclBuff = resCtx.cclMem;
        tempAlgParamsAll0.buffInfo.hcclBuffType = BufferType::HCCL_BUFFER;
        tempAlgParamsAll0.buffInfo.hcclBuffSize = dataSize_ * rankSize;
        tempAlgParamsAll0.buffInfo.hcclBuffBaseOff = dataSizeAxis0 + dataSizeAxis1;

        // 不需要 copy out
        tempAlgParamsAll0.buffInfo.outputPtr = param.outputPtr;
        tempAlgParamsAll0.buffInfo.outBuffType = BufferType::OUTPUT;
        tempAlgParamsAll0.buffInfo.outputSize = param.outputSize;
        tempAlgParamsAll0.buffInfo.outBuffBaseOff = dataSizeAxis0 + dataSizeAxis1;
        tempAlgParamsAll0.outputSliceStride = dataSize_;

        tempAlgParamsAll0.sliceSize = dataSizeAxis2;
        tempAlgParamsAll0.count = dataCountAxis2;
    }

    CHK_RET(PreSyncInterThreads(mainThread_, templateMainThreads_, syncNotifyOnTemplates_));

    CHK_RET(tempAlgIntra.KernelRun(param, tempAlgParamsIntra0, intraTempAlgRes));

    tempAlgInter.SetTemplateDataParams1(tempAlgParamsAll0);
    CHK_RET(tempAlgInter.KernelRun(param, tempAlgParamsInter0, interTempAlgRes));
    
    CHK_RET(PostSyncInterThreads(mainThread_, templateMainThreads_, syncNotifyOnMain_));


    // Stage 1 远端读模式
    tempAlgIntra.SetRemoteWrite(false);
    tempAlgInter.SetRemoteWrite(false);

    // tempAlgParamsIntra0
    {
        tempAlgParamsIntra1.enableRemoteMemAccess = param.opMode == OpMode::OFFLOAD;

        tempAlgParamsIntra1.buffInfo.inputPtr = param.inputPtr;
        tempAlgParamsIntra1.buffInfo.inBuffType = BufferType::INPUT;
        tempAlgParamsIntra1.buffInfo.inputSize = dataSizeAxis0;
        tempAlgParamsIntra1.buffInfo.inBuffBaseOff = dataSizeAxis0;
        tempAlgParamsIntra1.inputSliceStride = dataSize_;

        tempAlgParamsIntra1.repeatNum = rankIdxLevel1_ / rankIdxLevel0_;

        tempAlgParamsIntra1.buffInfo.hcclBuff = resCtx.cclMem;
        tempAlgParamsIntra1.buffInfo.hcclBuffType = BufferType::HCCL_BUFFER;
        tempAlgParamsIntra1.buffInfo.hcclBuffSize = dataSize_ * rankSize;
        tempAlgParamsIntra1.buffInfo.hcclBuffBaseOff = 0;

        // 不需要 copy out
        tempAlgParamsIntra1.buffInfo.outputPtr = param.outputPtr;
        tempAlgParamsIntra1.buffInfo.outBuffType = BufferType::OUTPUT;
        tempAlgParamsIntra1.buffInfo.outputSize = param.outputSize;
        tempAlgParamsIntra1.buffInfo.outBuffBaseOff = dataSizeAxis0;
        tempAlgParamsIntra1.outputSliceStride = dataSize_;

        tempAlgParamsIntra1.sliceSize = dataSizeAxis0;
        tempAlgParamsIntra1.count = dataCountAxis0;
    }

    {
        tempAlgParamsInter1.enableRemoteMemAccess = param.opMode == OpMode::OFFLOAD;

        tempAlgParamsInter1.buffInfo.inputPtr = param.inputPtr;
        tempAlgParamsInter1.buffInfo.inBuffType = BufferType::INPUT;
        tempAlgParamsInter1.buffInfo.inputSize = dataSizeAxis1;
        tempAlgParamsInter1.buffInfo.inBuffBaseOff = 0;
        tempAlgParamsInter1.inputSliceStride = dataSize_;

        tempAlgParamsInter1.repeatNum = rankIdxLevel0_;

        tempAlgParamsInter1.buffInfo.hcclBuff = resCtx.cclMem;
        tempAlgParamsInter1.buffInfo.hcclBuffType = BufferType::HCCL_BUFFER;
        tempAlgParamsInter1.buffInfo.hcclBuffSize = dataSize_ * rankSize;
        tempAlgParamsInter1.buffInfo.hcclBuffBaseOff = 0;

        // 不需要 copy out
        tempAlgParamsInter1.buffInfo.outputPtr = param.outputPtr;
        tempAlgParamsInter1.buffInfo.outBuffType = BufferType::OUTPUT;
        tempAlgParamsInter1.buffInfo.outputSize = param.outputSize;
        tempAlgParamsInter1.buffInfo.outBuffBaseOff = 0;
        tempAlgParamsInter1.outputSliceStride = dataSize_;

        tempAlgParamsInter1.sliceSize = dataSizeAxis1;
        tempAlgParamsInter1.count = dataCountAxis1;
    }

    {
        tempAlgParamsAll1.enableRemoteMemAccess = param.opMode == OpMode::OFFLOAD;

        tempAlgParamsAll1.buffInfo.inputPtr = param.inputPtr;
        tempAlgParamsAll1.buffInfo.inBuffType = BufferType::INPUT;
        tempAlgParamsAll1.buffInfo.inputSize = dataSizeAxis2;
        tempAlgParamsAll1.buffInfo.inBuffBaseOff = dataSizeAxis0 + dataSizeAxis1;
        tempAlgParamsAll1.inputSliceStride = dataSize_;

        tempAlgParamsAll1.repeatNum = rankIdxLevel1_ - 4; // 先发4份

        tempAlgParamsAll1.buffInfo.hcclBuff = resCtx.cclMem;
        tempAlgParamsAll1.buffInfo.hcclBuffType = BufferType::HCCL_BUFFER;
        tempAlgParamsAll1.buffInfo.hcclBuffSize = dataSize_ * rankSize;
        tempAlgParamsAll1.buffInfo.hcclBuffBaseOff = 0;

        tempAlgParamsAll1.buffInfo.outputPtr = param.outputPtr;
        tempAlgParamsAll1.buffInfo.outBuffType = BufferType::OUTPUT;
        tempAlgParamsAll1.buffInfo.outputSize = param.outputSize;
        tempAlgParamsAll1.buffInfo.outBuffBaseOff = dataSizeAxis0 + dataSizeAxis1;
        tempAlgParamsAll1.outputSliceStride = dataSize_;

        tempAlgParamsAll1.sliceSize = dataSizeAxis2;
        tempAlgParamsAll1.count = dataCountAxis2;
    }

    CHK_RET(PreSyncInterThreads(mainThread_, templateMainThreads_, syncNotifyOnTemplates_));

    CHK_RET(tempAlgIntra.KernelRun(param, tempAlgParamsIntra1, intraTempAlgRes));

    tempAlgInter.SetTemplateDataParams1(tempAlgParamsAll1);
    CHK_RET(tempAlgInter.KernelRun(param, tempAlgParamsInter1, interTempAlgRes));
    
    CHK_RET(PostSyncInterThreads(mainThread_, templateMainThreads_, syncNotifyOnMain_));

    return HcclResult::HCCL_SUCCESS;
};

REGISTER_EXECUTOR_BY_TWO_TEMPS(HcclCMDType::HCCL_CMD_ALLGATHER, InsAllGatherParallelMesh1DMeshClosOptMultiJetty,
                               InsV2AllGatherParallelOptExecutor, TopoMatchUBX,
                               InsTempAllGatherMesh1DOpt, InsTempAllGatherMeshClosOpt);

}
// 算法注册
