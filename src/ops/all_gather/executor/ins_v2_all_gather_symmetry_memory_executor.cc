/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#include "ins_v2_all_gather_symmetry_memory_executor.h"
#include "topo_match_1d.h"
#include "ins_temp_all_gather_symmetry_memory_mesh_1D.h"

namespace ops_hccl {

template <typename AlgTopoMatch, typename InsAlgTemplate>
InsV2AllGatherSymmetryMemoryExecutor<AlgTopoMatch, InsAlgTemplate>::InsV2AllGatherSymmetryMemoryExecutor()
{
}

template <typename AlgTopoMatch, typename InsAlgTemplate>
HcclResult InsV2AllGatherSymmetryMemoryExecutor<AlgTopoMatch, InsAlgTemplate>::CalcAlgHierarchyInfo(
    HcclComm comm, TopoInfoWithNetLayerDetails *topoInfo, AlgHierarchyInfoForAllLevel &algHierarchyInfo)
{
    // 使用topo match计算AlgHierarchyInfoForAllLevel
    AlgTopoMatch topoMatch;
    CHK_RET(topoMatch.MatchTopo(comm, topoInfo, algHierarchyInfo));
    return HCCL_SUCCESS;
}

template <typename AlgTopoMatch, typename InsAlgTemplate>
HcclResult InsV2AllGatherSymmetryMemoryExecutor<AlgTopoMatch, InsAlgTemplate>::CalcRes(
    HcclComm comm, const OpParam &param, const TopoInfoWithNetLayerDetails *topoInfo, const AlgHierarchyInfoForAllLevel &algHierarchyInfo,
    AlgResourceRequest &resourceRequest)
{
    // 构建template
    std::shared_ptr<InsAlgTemplate> algTemplate =
        std::make_shared<InsAlgTemplate>(param, topoInfo->userRank, algHierarchyInfo.infos[0]);

    CHK_RET(algTemplate->CalcRes(comm, param, topoInfo, resourceRequest));
    myRank_ = topoInfo->userRank;
    HCCL_DEBUG("[InsV2AllGatherSymmetryMemoryExecutor][CalcRes] myRank[%u], notifyNumOnMainThread[%u], slaveThreadNum[%u], "
               "channels[%u]",
               myRank_, resourceRequest.notifyNumOnMainThread, resourceRequest.slaveThreadNum,
               resourceRequest.channels.size());
    for (auto i = 0; i < resourceRequest.notifyNumPerThread.size(); i++) {
        HCCL_DEBUG("[InsV2AllGatherSymmetryMemoryExecutor][CalcRes] myRank[%u], notifyNumPerThread[%u]=[%u]", myRank_, i,
                   resourceRequest.notifyNumPerThread[i]);
    }
    return HCCL_SUCCESS;
}

template <typename AlgTopoMatch, typename InsAlgTemplate>
HcclResult InsV2AllGatherSymmetryMemoryExecutor<AlgTopoMatch, InsAlgTemplate>::Orchestrate(
    const OpParam &param, const AlgResourceCtxSerializable &resCtx)
{
    HCCL_INFO("[InsV2AllGatherSymmetryMemoryExecutor][Orchestrate] Orchestrate Start");
    myRank_ = resCtx.topoInfo.userRank;

    threads_ = resCtx.threads;
    CHK_RET(RestoreChannelMap(resCtx, remoteRankToChannelInfo_));
    dataCount_ = param.DataDes.count;
    dataType_ = param.DataDes.dataType;
    dataTypeSize_ = DATATYPE_SIZE_TABLE[param.DataDes.dataType];
    dataSize_ = dataCount_ * dataTypeSize_;
    HCCL_DEBUG("[InsV2AllGatherSymmetryMemoryExecutor][Orchestrate] myRank[%u], threadsSize[%lu], "
               "dataCount[%llu], dataTypeSize[%lu]",
               myRank_, threads_.size(), dataCount_, dataTypeSize_);
    HcclResult ret = OrchestrateLoop(param, resCtx);
    CHK_PRT_RET(
        ret != HCCL_SUCCESS,
        HCCL_ERROR("[InsV2AllGatherSymmetryMemoryExecutor][Orchestrate]errNo[0x%016llx] All Gather excutor kernel run failed",
                   HCCL_ERROR_CODE(ret)),
        ret);
    HCCL_INFO("[InsV2AllGatherSymmetryMemoryExecutor][Orchestrate] Orchestrate End");
    return HCCL_SUCCESS;
}

template <typename AlgTopoMatch, typename InsAlgTemplate>
HcclResult InsV2AllGatherSymmetryMemoryExecutor<AlgTopoMatch, InsAlgTemplate>::OrchestrateLoop(
    const OpParam &param, const AlgResourceCtxSerializable &resCtx)
{
    HCCL_INFO("[InsV2AllGatherSymmetryMemoryExecutor][OrchestrateLoop] Start");

    // 准备资源
    TemplateResource templateAlgRes;

    if (remoteRankToChannelInfo_.size() > 0) {
        templateAlgRes.channels = remoteRankToChannelInfo_[0];
    }
    templateAlgRes.threads = resCtx.threads;
    // 准备数据
    TemplateDataParams tempAlgParams;
    tempAlgParams.buffInfo.inputPtr = param.inputPtr;
    tempAlgParams.buffInfo.outputPtr = param.outputPtr;

    // 类型改为HCCL_BUFFER?
    tempAlgParams.buffInfo.inBuffType = BufferType::INPUT;
    tempAlgParams.buffInfo.outBuffType = BufferType::OUTPUT;

    tempAlgParams.buffInfo.inputSize = param.inputSize;
    tempAlgParams.buffInfo.outputSize = param.outputSize;
    // 不需要重复
    tempAlgParams.repeatNum = 1;
    tempAlgParams.inputRepeatStride = 0;
    tempAlgParams.outputRepeatStride = 0;
    HCCL_INFO("[InsV2AllGatherSymmetryMemoryExecutor][OrchestrateLoop] myRank[%u], inputPtr[%#llx] outputPtr[%#llx], "
              "cclAddr[%#llx], cclSize[%llu], channelSize[%lu], threadSize[%lu], ",
              myRank_, param.inputPtr, param.outputPtr, resCtx.cclMem.addr, resCtx.cclMem.size,
              templateAlgRes.channels.size(), templateAlgRes.threads.size());
    // 构建template
    InsAlgTemplate algTemplate(param, resCtx.topoInfo.userRank, resCtx.algHierarchyInfo.infos[0]);

    if (param.engine == COMM_ENGINE_AICPU_TS && std::string(param.algName) != "InsAllGatherNHR") {
        algTemplate.SetchannelsPerRank(templateAlgRes.channels);
    }

    // 对称内存一次性传整个data
    tempAlgParams.buffInfo.inBuffBaseOff = 0;
    tempAlgParams.buffInfo.outBuffBaseOff = 0;

    tempAlgParams.count = dataCount_;
    tempAlgParams.sliceSize = dataSize_;
    tempAlgParams.inputSliceStride = 0;
    tempAlgParams.outputSliceStride = dataSize_;

    HCCL_DEBUG("[InsV2AllGatherSymmetryMemoryExecutor] myRank[%u], loop [%u] tempAlgParams.inputSliceStride [%u],"
                "tempAlgParams.outputSliceStride [%u] tempAlgParams.sliceSize [%u]",
                myRank_, loop, tempAlgParams.inputSliceStride, tempAlgParams.outputSliceStride,
                tempAlgParams.sliceSize);
    HCCL_DEBUG("[InsV2AllGatherSymmetryMemoryExecutor] myRank[%u], loop [%u] tempAlgParams.buffInfo.inBuffBaseOff [%u],"
                "tempAlgParams.buffInfo.outBuffBaseOff [%u]",
                myRank_, loop, tempAlgParams.buffInfo.inBuffBaseOff, tempAlgParams.buffInfo.outBuffBaseOff);

    CHK_RET(algTemplate.KernelRun(param, tempAlgParams, templateAlgRes));
    processedDataCount += currDataCount;

    HCCL_INFO("[InsV2AllGatherSymmetryMemoryExecutor][OrchestrateLoop] End.");
    return HCCL_SUCCESS;
}

REGISTER_EXEC_V2(HcclCMDType::HCCL_CMD_ALLGATHER, InsAllGatherMesh1D, InsV2AllGatherSymmetryMemoryExecutor, TopoMatch1D,
                 InsTempAllGatherSymmetryMemoryMesh1D);
}  // namespace ops_hccl
