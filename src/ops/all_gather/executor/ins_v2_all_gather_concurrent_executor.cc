/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "ins_v2_all_gather_concurrent_executor.h"
#include "math.h"
#include "alg_data_trans_wrapper.h"
#include "hccl_res.h"
#include "ccu_alg_template_base.h"

// AICPU template 头文件
#include "ins_temp_all_gather_mesh_1D.h"
#include "ins_temp_all_gather_nhr.h"

#ifndef AICPU_COMPILE
// CCU template 头文件
#include "ccu_temp_all_gather_mesh_1D.h"
#include "ccu_temp_all_gather_nhr_1D_multijetty_mem2mem.h"
#include "ccu_temp_all_gather_mesh_1D_mem2mem.h"

#endif

constexpr u32 CLOS_PORT_NUM = 4;

namespace ops_hccl {

template <typename AlgTopoMatch, typename InsAlgTemplate0, typename InsAlgTemplate1>
InsAllGatherConcurrentExecutor<AlgTopoMatch, InsAlgTemplate0, InsAlgTemplate1>::InsAllGatherConcurrentExecutor()
{
}

template <typename AlgTopoMatch, typename InsAlgTemplate0, typename InsAlgTemplate1>
HcclResult InsAllGatherConcurrentExecutor<AlgTopoMatch, InsAlgTemplate0, InsAlgTemplate1>::InitCommInfo(
    const OpParam &param, const TopoInfoWithNetLayerDetails *topoInfo, const AlgHierarchyInfoForAllLevel &algHierarchyInfo)
{
    myRank_ = topoInfo->userRank;
    rankSize_ = topoInfo->userRankSize;
    devType_ = topoInfo->deviceType;
    dataType_ = param.DataDes.dataType;
    dataCount_ = param.DataDes.count;
    dataTypeSize_ = SIZE_TABLE[param.DataDes.dataType];

    algHierarchyInfo_ = algHierarchyInfo;
    HCCL_INFO("[InsAllGatherConcurrentExecutor][InitCommInfo] myRank [%u], rankSize [%u], devType [%u], "
              "dataType [%u] dataTypeSize [%u], dataCount_ [%u]",
              myRank_, rankSize_, devType_, dataType_, dataTypeSize_, dataCount_);
    return HCCL_SUCCESS;
}

template <typename AlgTopoMatch, typename InsAlgTemplate0, typename InsAlgTemplate1>
HcclResult InsAllGatherConcurrentExecutor<AlgTopoMatch, InsAlgTemplate0, InsAlgTemplate1>::CalcAlgHierarchyInfo(
    HcclComm comm, TopoInfoWithNetLayerDetails *topoInfo, AlgHierarchyInfoForAllLevel &algHierarchyInfo)
{
    // 使用topo match计算AlgHierarchyInfoForAllLevel
    AlgTopoMatch topoMatch;
    CHK_RET(topoMatch.MatchTopo(comm, topoInfo, algHierarchyInfo));
    return HCCL_SUCCESS;
}

template <typename AlgTopoMatch, typename InsAlgTemplate0, typename InsAlgTemplate1>
HcclResult InsAllGatherConcurrentExecutor<AlgTopoMatch, InsAlgTemplate0, InsAlgTemplate1>::CalcRes(
    HcclComm comm, const OpParam &param, const TopoInfoWithNetLayerDetails *topoInfo, const AlgHierarchyInfoForAllLevel &algHierarchyInfo,
    AlgResourceRequest &resourceRequest)
{
    // 初始化一些基本成员变量
    InitCommInfo(param, topoInfo, algHierarchyInfo);
    // 拆分algHierarchyInfo
    std::vector<std::vector<u32>> temp0HierarchyInfo = {algHierarchyInfo.infos[0][0]};
    std::vector<std::vector<u32>> temp1HierarchyInfo = {algHierarchyInfo.infos[0][1]};
    // 构建template
    std::shared_ptr<InsAlgTemplate0> temp0Alg = std::make_shared<InsAlgTemplate0>(param, myRank_, temp0HierarchyInfo);
    std::shared_ptr<InsAlgTemplate1> temp1Alg = std::make_shared<InsAlgTemplate1>(param, myRank_, temp1HierarchyInfo);
    // 调用计算资源的函数
    AlgResourceRequest temp0ResReq;
    AlgResourceRequest temp1ResReq;
    temp0Alg->CalcRes(comm, param, topoInfo, temp0ResReq);
    temp1Alg->CalcRes(comm, param, topoInfo, temp1ResReq);

    std::vector<HcclChannelDesc> temp0Channels;
    std::vector<HcclChannelDesc> temp1Channels;
    CommTopo temp0PriorityTopo = COMM_TOPO_1DMESH;
    CHK_RET(CalcChannelRequestMesh1DWithPriorityTopo(comm, param, topoInfo, temp0HierarchyInfo, temp0Channels, temp0PriorityTopo));
    CommTopo temp1PriorityTopo = COMM_TOPO_CLOS;
    CHK_RET(CalcChannelRequestNHRWithPriorityTopo(comm, param, topoInfo, temp1HierarchyInfo, temp1Channels, temp1PriorityTopo));

    CHK_PRT_RET(temp0Channels.size() != temp1Channels.size(),
            HCCL_ERROR("[InsAllGatherConcurrentExecutor][CalcRes] temp0Channels.size()[%zu] is not equal to temp1Channels.size()[%zu]",
                    temp0Channels.size(), temp1Channels.size()),
            HcclResult::HCCL_E_INTERNAL);

    // 两个模板并行，资源累加
    resourceRequest.slaveThreadNum = temp0ResReq.slaveThreadNum + temp1ResReq.slaveThreadNum + 1;
    resourceRequest.notifyNumOnMainThread = temp0ResReq.notifyNumOnMainThread + 1;
    resourceRequest.notifyNumPerThread.insert(resourceRequest.notifyNumPerThread.end(),
                                              temp0ResReq.notifyNumPerThread.begin(),
                                              temp0ResReq.notifyNumPerThread.end());  // 每个从流所需Notify数量
    resourceRequest.notifyNumPerThread.emplace_back(temp1ResReq.notifyNumOnMainThread + 1);  // 需要与主流通信
    resourceRequest.notifyNumPerThread.insert(resourceRequest.notifyNumPerThread.end(),
                                              temp1ResReq.notifyNumPerThread.begin(),
                                              temp1ResReq.notifyNumPerThread.end());
    
    if (param.engine == CommEngine::COMM_ENGINE_CCU) {
        for (auto &kernelInfo : temp0ResReq.ccuKernelInfos) {
            kernelInfo.channels = temp0Channels;
        }
        resourceRequest.ccuKernelNum.emplace_back(temp0ResReq.ccuKernelNum[0]);
        resourceRequest.ccuKernelNum.emplace_back(temp1ResReq.ccuKernelNum[0]);
        resourceRequest.ccuKernelInfos.insert(resourceRequest.ccuKernelInfos.end(), temp0ResReq.ccuKernelInfos.begin(),
                                              temp0ResReq.ccuKernelInfos.end());
        resourceRequest.ccuKernelInfos.insert(resourceRequest.ccuKernelInfos.end(), temp1ResReq.ccuKernelInfos.begin(),
                                              temp1ResReq.ccuKernelInfos.end());
    } else {
        resourceRequest.channels[0].insert(resourceRequest.channels[0].end(), temp0Channels.begin(),
                                           temp0Channels.end());
        resourceRequest.channels[0].insert(resourceRequest.channels[0].end(), temp1Channels.begin(),
            temp1Channels.end());
    }
    return HCCL_SUCCESS;
}

template <typename AlgTopoMatch, typename InsAlgTemplate0, typename InsAlgTemplate1>
void InsAllGatherConcurrentExecutor<AlgTopoMatch, InsAlgTemplate0, InsAlgTemplate1>::GenAlgParamsforTemplate0(
    const OpParam &param, const AlgResourceCtxSerializable &resCtx, const u64 dataOffset,
    const u64 dataCountPerLoopforTemp0, const u64 scratchOffset, TemplateDataParams &temp0AlgParams) const
{
    temp0AlgParams.buffInfo.inputPtr = param.inputPtr;
    temp0AlgParams.buffInfo.outputPtr = param.outputPtr;
    temp0AlgParams.buffInfo.hcclBuff = resCtx.cclMem;
    temp0AlgParams.buffInfo.inputSize = param.inputSize;
    temp0AlgParams.buffInfo.outputSize = param.outputSize;
    temp0AlgParams.buffInfo.inBuffType = BufferType::INPUT;
    temp0AlgParams.buffInfo.outBuffType = BufferType::OUTPUT;
    temp0AlgParams.count = dataCountPerLoopforTemp0;
    temp0AlgParams.sliceSize = dataCountPerLoopforTemp0 * dataTypeSize_;
    temp0AlgParams.buffInfo.inBuffBaseOff = dataOffset;
    temp0AlgParams.buffInfo.outBuffBaseOff = dataOffset;
    temp0AlgParams.buffInfo.hcclBuffBaseOff = scratchOffset;

    temp0AlgParams.inputSliceStride = 0;
    temp0AlgParams.outputSliceStride = dataSize_;
    temp0AlgParams.repeatNum = 1;
    temp0AlgParams.inputRepeatStride = 0;
    temp0AlgParams.outputRepeatStride = 0;

    HCCL_DEBUG(
        "[InsAllGatherConcurrentExecutor][GenTemplate0AlgParams] rank[%d] inBuffBaseOff[%llu] "
        "outBuffBaseOff[%llu] hcclBuffBaseOff[%llu] sliceSize[%llu] inputSliceStride[%llu] outputSliceStride[%llu]",
        myRank_, temp0AlgParams.buffInfo.inBuffBaseOff, temp0AlgParams.buffInfo.outBuffBaseOff,
        temp0AlgParams.buffInfo.hcclBuffBaseOff, temp0AlgParams.sliceSize, temp0AlgParams.inputSliceStride,
        temp0AlgParams.outputSliceStride);
    return;
}

template <typename AlgTopoMatch, typename InsAlgTemplate0, typename InsAlgTemplate1>
void InsAllGatherConcurrentExecutor<AlgTopoMatch, InsAlgTemplate0, InsAlgTemplate1>::GenAlgParamsforTemplate1(
    const OpParam &param, const AlgResourceCtxSerializable &resCtx, const u64 dataOffset,
    const u64 dataCountPerLoopforTemp1, const u64 scratchOffset, TemplateDataParams &temp1AlgParams) const
{
    temp1AlgParams.buffInfo.inputPtr = param.inputPtr;
    temp1AlgParams.buffInfo.outputPtr = param.outputPtr;
    temp1AlgParams.buffInfo.hcclBuff = resCtx.cclMem;
    temp1AlgParams.buffInfo.inputSize = param.inputSize;
    temp1AlgParams.buffInfo.outputSize = param.outputSize;
    temp1AlgParams.buffInfo.inBuffType = BufferType::INPUT;
    temp1AlgParams.buffInfo.outBuffType = BufferType::OUTPUT;
    temp1AlgParams.count = dataCountPerLoopforTemp1;
    temp1AlgParams.sliceSize = dataCountPerLoopforTemp1 * dataTypeSize_;
    temp1AlgParams.buffInfo.inBuffBaseOff = dataOffset;
    temp1AlgParams.buffInfo.outBuffBaseOff = dataOffset;
    temp1AlgParams.buffInfo.hcclBuffBaseOff = scratchOffset;

    temp1AlgParams.inputSliceStride = 0; 
    temp1AlgParams.outputSliceStride = dataSize_;
    temp1AlgParams.repeatNum = 1;
    temp1AlgParams.inputRepeatStride = 0;
    temp1AlgParams.outputRepeatStride = 0;

    HCCL_DEBUG(
        "[InsAllGatherConcurrentExecutor][GenTemplate1AlgParams] rank[%d] inBuffBaseOff[%llu] "
        "outBuffBaseOff[%llu] hcclBuffBaseOff[%llu] sliceSize[%llu] inputSliceStride[%llu] outputSliceStride[%llu]",
        myRank_, temp1AlgParams.buffInfo.inBuffBaseOff, temp1AlgParams.buffInfo.outBuffBaseOff,
        temp1AlgParams.buffInfo.hcclBuffBaseOff, temp1AlgParams.sliceSize, temp1AlgParams.inputSliceStride,
        temp1AlgParams.outputSliceStride);
    return;
}

template <typename AlgTopoMatch, typename InsAlgTemplate0, typename InsAlgTemplate1>
void InsAllGatherConcurrentExecutor<AlgTopoMatch, InsAlgTemplate0, InsAlgTemplate1>::GetParallelDataSplit(
    std::vector<float> &splitDataSize) const
{
    const u32 portNum0 = rankSize_ - 1;  // mesh端口数为rank size - 1
    const u32 portNum1 = CLOS_PORT_NUM;
    double splitData = portNum0 / (portNum0 + portNum1);
    splitDataSize.push_back(splitData);  
    splitDataSize.push_back(1 - splitData);
    HCCL_INFO("[InsAllGatherConcurrentExecutor][GenTemplate1AlgParams] portNum0[%u], portNum1[%u], splitData[%.4f], ",
            portNum0, portNum1, splitData);
    return;
}

template <typename AlgTopoMatch, typename InsAlgTemplate0, typename InsAlgTemplate1>
HcclResult InsAllGatherConcurrentExecutor<AlgTopoMatch, InsAlgTemplate0, InsAlgTemplate1>::PrepareResForTemplate(
    const OpParam &param, const AlgResourceCtxSerializable &resCtx, InsAlgTemplate0 &algTemplate0,
    InsAlgTemplate1 &algTemplate1)
{
    AlgResourceRequest temp0Request;
    AlgResourceRequest temp1Request;
    algTemplate0.GetRes(temp0Request);  // 算法0需要的资源
    algTemplate1.GetRes(temp1Request);  // 算法1需要的资源

    auto tmp0ThreadsNum = temp0Request.slaveThreadNum + 1;
    auto tmp1ThreadsNum = temp1Request.slaveThreadNum + 1;
    auto tmp0NotifyOnMainThread = temp0Request.notifyNumOnMainThread;
    auto tmp1NotifyOnMainThread = temp1Request.notifyNumOnMainThread;

    tmp0Threads_.assign(threads_.begin(), threads_.begin() + tmp0ThreadsNum);
    tmp1Threads_.assign(threads_.begin() + tmp0ThreadsNum, threads_.end());
    // 用于两个算法同步
    mainThread_ = tmp0Threads_.at(0);
    templateMainThreads_.emplace_back(tmp1Threads_.at(0));
    syncNotifyOnTemplates_ = {tmp1NotifyOnMainThread}; // 算法1需要从流数
    syncNotifyOnMain_ = {tmp0NotifyOnMainThread}; // 算法0需要从流数

    return HCCL_SUCCESS;
}

template <typename AlgTopoMatch, typename InsAlgTemplate0, typename InsAlgTemplate1>
HcclResult InsAllGatherConcurrentExecutor<AlgTopoMatch, InsAlgTemplate0, InsAlgTemplate1>::Orchestrate(
    const OpParam &param, const AlgResourceCtxSerializable &resCtx)
{
    HCCL_INFO("[InsAllGatherConcurrentExecutor][Orchestrate] Orchestrate Start");
    algHierarchyInfo_ = resCtx.algHierarchyInfo;
    threads_ = resCtx.threads;

    maxTmpMemSize_ = resCtx.cclMem.size;
    myRank_ = resCtx.topoInfo.userRank;
    rankSize_ = resCtx.topoInfo.userRankSize;
    dataCount_ = param.DataDes.count;
    dataType_ = param.DataDes.dataType;
    dataTypeSize_ = DATATYPE_SIZE_TABLE[param.DataDes.dataType];
    dataSize_ = dataCount_ * dataTypeSize_;
     
    // 拆分algHierarchyInfo
    std::vector<std::vector<u32>> temp0HierarchyInfo = {algHierarchyInfo_.infos[0][0]};
    std::vector<std::vector<u32>> temp1HierarchyInfo = {algHierarchyInfo_.infos[0][1]};

    // 构建template
    InsAlgTemplate0 algTemplate0(param, myRank_, temp0HierarchyInfo);
    InsAlgTemplate1 algTemplate1(param, myRank_, temp1HierarchyInfo);

    // 分配threads
    PrepareResForTemplate(param, resCtx, algTemplate0, algTemplate1);

    // 分配channels或者ccuKernels
    if (param.engine == CommEngine::COMM_ENGINE_CCU) {
        tmp0CcuKernels_.assign(resCtx.ccuKernels.begin(), resCtx.ccuKernels.begin() + resCtx.ccuKernelNum[0]);
        tmp1CcuKernels_.assign(resCtx.ccuKernels.begin() + resCtx.ccuKernelNum[0], resCtx.ccuKernels.begin() + resCtx.ccuKernelNum[0] + resCtx.ccuKernelNum[1]);
    } else {
        const u64 splitIndex = rankSize_ - 1; // 默认第一个算法为mesh，使用 rankSize_ - 1 个 link
        for (u64 i = 0; i < resCtx.channels[0].size(); ++i) {
            const auto& channel = resCtx.channels[0][i];
            auto& targetMap = (i < splitIndex) ? tmp0LinkMap_ : tmp1LinkMap_;
            targetMap[channel.remoteRank].push_back(channel);
        }
    }

    // 算法展开
    HcclResult ret = OrchestrateLoop(param, resCtx, algTemplate0, algTemplate1);
    CHK_PRT_RET(
        ret != HCCL_SUCCESS,
        HCCL_ERROR("[InsAllGatherConcurrentExecutor][Orchestrate]errNo[0x%016llx] All gather executor kernel run failed",
                   HCCL_ERROR_CODE(ret)),
        ret);
    return HCCL_SUCCESS;
}

template <typename AlgTopoMatch, typename InsAlgTemplate0, typename InsAlgTemplate1>
HcclResult InsAllGatherConcurrentExecutor<AlgTopoMatch, InsAlgTemplate0, InsAlgTemplate1>::OrchestrateLoop(
    const OpParam &param, const AlgResourceCtxSerializable &resCtx, InsAlgTemplate0 &algTemplate0,
    InsAlgTemplate1 &algTemplate1)
{
    HCCL_INFO("[InsAllGatherConcurrentExecutor][OrchestrateLoop] Start");
    // 构造Mesh拓扑template资源
    TemplateResource templateAlgResforTemp0;
    templateAlgResforTemp0.threads = tmp0Threads_;
    // 构造Nhr拓扑template资源
    TemplateResource templateAlgResforTemp1;
    templateAlgResforTemp1.threads = tmp1Threads_;

    if (param.engine == CommEngine::COMM_ENGINE_CCU) {
        templateAlgResforTemp0.ccuKernels = tmp0CcuKernels_;
        templateAlgResforTemp1.ccuKernels = tmp1CcuKernels_;
    } else {
        templateAlgResforTemp0.channels = tmp0LinkMap_;
        templateAlgResforTemp1.channels = tmp1LinkMap_;
    }

    TemplateDataParams AlgParamsforTemp0;
    TemplateDataParams AlgParamsforTemp1;

    // 数据切分
    std::vector<float> dataSplitSize;
    GetParallelDataSplit(dataSplitSize);

    // 缓存切分
    u32 scratchMultiplierforTemp0 = algTemplate0.CalcScratchMultiple(AlgParamsforTemp0.buffInfo.inBuffType,
                                                                      AlgParamsforTemp0.buffInfo.outBuffType);
    u32 scratchMultiplierforTemp1 = algTemplate1.CalcScratchMultiple(AlgParamsforTemp1.buffInfo.inBuffType,
                                                                      AlgParamsforTemp1.buffInfo.outBuffType);
    u32 ScratchMultiplier0 =
        static_cast<u32>(std::ceil(dataSplitSize[0] * scratchMultiplierforTemp0));
    u32 ScratchMultiplier1 = static_cast<u32>(std::ceil(dataSplitSize[1] * scratchMultiplierforTemp1));
    u32 totalScratchMultiple = ScratchMultiplier0 + ScratchMultiplier1;
    u64 scratchMemBlockSize = maxTmpMemSize_;
    HCCL_INFO("[InsAllGatherConcurrentExecutor]maxTmpMemSize_ [%u]", maxTmpMemSize_);
    if (totalScratchMultiple > 0) {
        scratchMemBlockSize = (maxTmpMemSize_ / HCCL_MIN_SLICE_ALIGN / totalScratchMultiple) * HCCL_MIN_SLICE_ALIGN;
    }
    u64 scratchOffsetforTemp0 = 0;
    u64 scratchOffsetforTemp1 = ScratchMultiplier0 * scratchMemBlockSize;

    // 计算单次传输最大数据条目数
    u64 transportBoundDataSize = UB_MAX_DATA_SIZE;
    u64 maxDataCountPerLoop =
        (std::min(static_cast<u64>(scratchMemBlockSize), static_cast<u64>(UB_MAX_DATA_SIZE)) / dataTypeSize_);
    u64 maxDataSizePerLoop = maxDataCountPerLoop * dataTypeSize_;
    HCCL_INFO("[InsAllGatherConcurrentExecutor][OrchestrateOpbase] maxDataCountPerLoop[%llu], maxDataSizePerLoop[%llu], "
              "transportBoundDataSize[%llu], totalScratchMultiple[%llu]",
              maxDataCountPerLoop, maxDataSizePerLoop, transportBoundDataSize, totalScratchMultiple);
    CHK_PRT_RET(maxDataCountPerLoop == 0,
                HCCL_ERROR("[InsAllGatherConcurrentExecutor][OrchestrateOpbase] maxDataCountPerLoop is 0"),
                HCCL_E_INTERNAL);

    // 计算loopTimes
    u64 loopTimes = dataCount_ / maxDataCountPerLoop + static_cast<u64>(dataCount_ % maxDataCountPerLoop != 0);
    u64 processedDataCount = 0;
    // 前同步
    CHK_RET(PreSyncInterThreads(mainThread_, templateMainThreads_, syncNotifyOnTemplates_));
    for (u64 loopIndex = 0; loopIndex < loopTimes; loopIndex++) {
        u64 currDataCount = (loopIndex == loopTimes - 1) ? dataCount_ - processedDataCount : maxDataCountPerLoop;
        u64 dataCountPerLoopforTemp0 = static_cast<u64>(dataSplitSize[0] * currDataCount);
        u64 dataCountPerLoopforTemp1 = currDataCount - dataCountPerLoopforTemp0;
        u64 dataOffsetforTemp0 = processedDataCount * dataTypeSize_;
        u64 dataOffsetforTemp1 = dataOffsetforTemp0 + dataCountPerLoopforTemp0 * dataTypeSize_;
        // 第一个算法
        GenAlgParamsforTemplate0(param, resCtx, dataOffsetforTemp0, dataCountPerLoopforTemp0, scratchOffsetforTemp0,
                                 AlgParamsforTemp0);
        CHK_RET(algTemplate0.KernelRun(param, AlgParamsforTemp0, templateAlgResforTemp0));
        // 第二个算法
        GenAlgParamsforTemplate1(param, resCtx, dataOffsetforTemp1, dataCountPerLoopforTemp1, scratchOffsetforTemp1,
                                 AlgParamsforTemp1);
        CHK_RET(algTemplate1.KernelRun(param, AlgParamsforTemp1, templateAlgResforTemp1));
        processedDataCount += currDataCount;
    }
    // 尾同步
    CHK_RET(PostSyncInterThreads(mainThread_, templateMainThreads_, syncNotifyOnMain_));
    HCCL_INFO("[InsAllGatherConcurrentExecutor][OrchestrateLoop] End.");
    return HCCL_SUCCESS;
}

// 算法注册
REGISTER_EXECUTOR_BY_TWO_TEMPS(HcclCMDType::HCCL_CMD_ALLGATHER, InsAllGatherConcurrentMesh1DNHR, InsAllGatherConcurrentExecutor,
                              TopoMatchUBX, InsTempAllGatherMesh1D, InsTempAllGatherNHR);

#ifndef AICPU_COMPILE
REGISTER_EXECUTOR_BY_TWO_TEMPS(HcclCMDType::HCCL_CMD_ALLGATHER, CcuAllGatherConcurrentMesh1DNHRMem, InsAllGatherConcurrentExecutor,
    TopoMatchUBX, CcuTempAllGatherMesh1DMem2Mem, CcuTemAllGatherNHR1DMultiJettyMem2Mem);

REGISTER_EXECUTOR_BY_TWO_TEMPS(HcclCMDType::HCCL_CMD_ALLGATHER, CcuAllGatherConcurrentMesh1DNHR,
                               InsAllGatherConcurrentExecutor, TopoMatchUBX, CcuTempAllGatherMesh1D,
                               CcuTemAllGatherNHR1DMultiJettyMem2Mem);
#endif

}  // namespace