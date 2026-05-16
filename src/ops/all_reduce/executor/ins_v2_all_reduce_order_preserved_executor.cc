/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "ins_v2_all_reduce_order_preserved_executor.h"
#include "ins_temp_reduce_scatter_order_preserved.h"
#include "ins_temp_all_gather_nhr.h"
#include <cmath>
#include <algorithm>

namespace ops_hccl {

template <typename AlgTopoMatch, typename InsAlgTemplateRS, typename InsAlgTemplateAG>
InsV2AllReduceOrderPreservedExecutor<AlgTopoMatch, InsAlgTemplateRS, InsAlgTemplateAG>::InsV2AllReduceOrderPreservedExecutor()
{
    deterministicStrict_ = true;
}

template <typename AlgTopoMatch, typename InsAlgTemplateRS, typename InsAlgTemplateAG>
HcclResult InsV2AllReduceOrderPreservedExecutor<AlgTopoMatch, InsAlgTemplateRS, InsAlgTemplateAG>::CalcAlgHierarchyInfo(
    HcclComm comm, TopoInfoWithNetLayerDetails *topoInfo, AlgHierarchyInfoForAllLevel &algHierarchyInfo)
{
    myRank_ = topoInfo->userRank;
    rankSize_ = topoInfo->userRankSize;
    devType_ = topoInfo->deviceType;
    AlgTopoMatch topoMatch;
    CHK_RET(topoMatch.MatchTopo(comm, topoInfo, algHierarchyInfo));
    HCCL_INFO("[InsV2AllReduceOrderPreservedExecutor][CalcAlgHierarchyInfo] myRank[%u], rankSize[%u] (flat level1 only)",
        myRank_, rankSize_);
    return HCCL_SUCCESS;
}

template <typename AlgTopoMatch, typename InsAlgTemplateRS, typename InsAlgTemplateAG>
HcclResult InsV2AllReduceOrderPreservedExecutor<AlgTopoMatch, InsAlgTemplateRS, InsAlgTemplateAG>::CalcRes(
    HcclComm comm, const OpParam &param, const TopoInfoWithNetLayerDetails *topoInfo,
    const AlgHierarchyInfoForAllLevel &algHierarchyInfo, AlgResourceRequest &resourceRequest)
{
    myRank_ = topoInfo->userRank;
    rankSize_ = topoInfo->userRankSize;
    devType_ = topoInfo->deviceType;
    reduceOp_ = param.reduceType;
    dataType_ = param.DataDes.dataType;
    dataCount_ = param.DataDes.count;
    dataTypeSize_ = SIZE_TABLE[param.DataDes.dataType];
    aicpuUnfoldMode_ = param.aicpuUnfoldMode;

    algHierarchyInfo_ = algHierarchyInfo;

    InitExecutorInfo(param);
    CalcSizePerBlock(param);
    CalcGroupSlices(param);

    std::shared_ptr<InsAlgTemplateRS> rsTempAlg =
        std::make_shared<InsAlgTemplateRS>(param, myRank_, algHierarchyInfo.infos[0]);

    std::shared_ptr<InsAlgTemplateAG> agTempAlg =
        std::make_shared<InsAlgTemplateAG>(param, myRank_, algHierarchyInfo.infos[0]);

    AlgResourceRequest resReqRS;
    AlgResourceRequest resReqAG;

    CHK_RET(rsTempAlg->CalcRes(comm, param, topoInfo, resReqRS));
    CHK_RET(agTempAlg->CalcRes(comm, param, topoInfo, resReqAG));

    resourceRequest.slaveThreadNum = std::max(resReqRS.slaveThreadNum, resReqAG.slaveThreadNum);

    resourceRequest.notifyNumPerThread.clear();
    resourceRequest.notifyNumPerThread.resize(resourceRequest.slaveThreadNum);
    for (u32 i = 0; i < resourceRequest.slaveThreadNum; ++i) {
        if (i < resReqRS.notifyNumPerThread.size()) {
            resourceRequest.notifyNumPerThread[i] = std::max(resourceRequest.notifyNumPerThread[i],
                resReqRS.notifyNumPerThread[i]);
        }
        if (i < resReqAG.notifyNumPerThread.size()) {
            resourceRequest.notifyNumPerThread[i] = std::max(resourceRequest.notifyNumPerThread[i],
                resReqAG.notifyNumPerThread[i]);
        }
    }

    resourceRequest.notifyNumOnMainThread = std::max(resReqRS.notifyNumOnMainThread,
        resReqAG.notifyNumOnMainThread);

    resourceRequest.channels.clear();
    if (resReqRS.channels.size() > 0) {
        resourceRequest.channels.push_back(resReqRS.channels[0]);
    }
    if (resReqAG.channels.size() > 0) {
        resourceRequest.channels.push_back(resReqAG.channels[0]);
    }

    HCCL_INFO("[InsV2AllReduceOrderPreservedExecutor][CalcRes] slaveThreadNum[%u], notifyNumOnMainThread[%u]",
        resourceRequest.slaveThreadNum, resourceRequest.notifyNumOnMainThread);
    return HCCL_SUCCESS;
}

template <typename AlgTopoMatch, typename InsAlgTemplateRS, typename InsAlgTemplateAG>
HcclResult InsV2AllReduceOrderPreservedExecutor<AlgTopoMatch, InsAlgTemplateRS, InsAlgTemplateAG>::Orchestrate(
    const OpParam &param, const AlgResourceCtxSerializable& resCtx)
{
    HCCL_INFO("[InsV2AllReduceOrderPreservedExecutor][Orchestrate] Start");

    myRank_ = resCtx.topoInfo.userRank;
    rankSize_ = resCtx.topoInfo.userRankSize;
    dataCount_ = param.DataDes.count;
    dataTypeSize_ = SIZE_TABLE[param.DataDes.dataType];
    dataSize_ = dataCount_ * dataTypeSize_;
    dataType_ = param.DataDes.dataType;
    reduceOp_ = param.reduceType;
    algHierarchyInfo_ = resCtx.algHierarchyInfo;
    threads_ = resCtx.threads;
    aicpuUnfoldMode_ = param.aicpuUnfoldMode;

    CHK_RET(RestoreChannelMap(resCtx, remoteRankToChannelInfo_));

    InitExecutorInfo(param);
    CalcSizePerBlock(param);
    CalcGroupSlices(param);

    HcclResult ret = OrchestrateLoop(param, resCtx);
    CHK_PRT_RET(ret != HCCL_SUCCESS,
        HCCL_ERROR("[InsV2AllReduceOrderPreservedExecutor][Orchestrate] kernel run failed, err[0x%016llx]",
            HCCL_ERROR_CODE(ret)), ret);
    return HCCL_SUCCESS;
}

template <typename AlgTopoMatch, typename InsAlgTemplateRS, typename InsAlgTemplateAG>
HcclResult InsV2AllReduceOrderPreservedExecutor<AlgTopoMatch, InsAlgTemplateRS, InsAlgTemplateAG>::OrchestrateLoop(
    const OpParam &param, const AlgResourceCtxSerializable &resCtx)
{
    HCCL_INFO("[InsV2AllReduceOrderPreservedExecutor][OrchestrateLoop] Start, deterministicStrict[%d] (flat level1)",
        deterministicStrict_);

    CHK_RET(RunReduceScatter(param, resCtx));

    CHK_RET(RunAllGather(param, resCtx));

    HCCL_INFO("[InsV2AllReduceOrderPreservedExecutor][OrchestrateLoop] Success");
    return HCCL_SUCCESS;
}

template <typename AlgTopoMatch, typename InsAlgTemplateRS, typename InsAlgTemplateAG>
HcclResult InsV2AllReduceOrderPreservedExecutor<AlgTopoMatch, InsAlgTemplateRS, InsAlgTemplateAG>::InitExecutorInfo(const OpParam &param)
{
    deterministicStrict_ = IsNeedStrictMode(param);
    if (deterministicStrict_) {
        CHK_PRT_RET(!CheckStrictCondition(param),
            HCCL_ERROR("[InsV2AllReduceOrderPreservedExecutor] not support DETERMINISTIC_STRICT mode."),
            HCCL_E_NOT_SUPPORT);
    }
    HCCL_INFO("[InsV2AllReduceOrderPreservedExecutor][InitExecutorInfo] deterministicStrict[%d]",
        deterministicStrict_);
    return HCCL_SUCCESS;
}

template <typename AlgTopoMatch, typename InsAlgTemplateRS, typename InsAlgTemplateAG>
u64 InsV2AllReduceOrderPreservedExecutor<AlgTopoMatch, InsAlgTemplateRS, InsAlgTemplateAG>::RoundUpWithDivisor(
    u64 value, u64 divisor) const
{
    if (value == 0 || divisor == 0) {
        return divisor;
    }
    return ((value + (divisor - 1)) / divisor) * divisor;
}

template <typename AlgTopoMatch, typename InsAlgTemplateRS, typename InsAlgTemplateAG>
HcclResult InsV2AllReduceOrderPreservedExecutor<AlgTopoMatch, InsAlgTemplateRS, InsAlgTemplateAG>::CalcSizePerBlock(const OpParam &param)
{
    u64 sizePerBlock = (dataCount_ + rankSize_ - 1) / rankSize_ * dataTypeSize_;
    memInfo_.sizePerBlock = RoundUpWithDivisor(sizePerBlock, HCCL_MIN_SLICE_ALIGN_ORDER_PRESERVED);
    memInfo_.all2allOffset = 0;
    memInfo_.scratchMemFlag = false;
    memInfo_.totalSize = 0;
    HCCL_INFO("[CalcSizePerBlock] sizePerBlock[%llu], dataCount[%llu], rankSize[%u]",
        memInfo_.sizePerBlock, dataCount_, rankSize_);
    return HCCL_SUCCESS;
}

template <typename AlgTopoMatch, typename InsAlgTemplateRS, typename InsAlgTemplateAG>
HcclResult InsV2AllReduceOrderPreservedExecutor<AlgTopoMatch, InsAlgTemplateRS, InsAlgTemplateAG>::CalcGroupSlices(const OpParam &param)
{
    memInfo_.groupSize.clear();
    u64 sizeRemain = dataSize_;
    for (u32 rankId = 0; rankId < rankSize_; rankId++) {
        u64 size = (sizeRemain > memInfo_.sizePerBlock) ? memInfo_.sizePerBlock : sizeRemain;
        memInfo_.groupSize.push_back(size);
        sizeRemain -= size;
    }
    memInfo_.totalSize = std::max(memInfo_.sizePerBlock * rankSize_, dataSize_);
    HCCL_INFO("[CalcGroupSlices] groupSize.size[%u], totalSize[%llu]",
        memInfo_.groupSize.size(), memInfo_.totalSize);
    return HCCL_SUCCESS;
}

template <typename AlgTopoMatch, typename InsAlgTemplateRS, typename InsAlgTemplateAG>
bool InsV2AllReduceOrderPreservedExecutor<AlgTopoMatch, InsAlgTemplateRS, InsAlgTemplateAG>::IsNeedStrictMode(const OpParam &param) const
{
    bool isStrictMode = (param.deterministicConfig == DETERMINISTIC_STRICT)
        && (param.DataDes.dataType == HCCL_DATA_TYPE_FP16 || param.DataDes.dataType == HCCL_DATA_TYPE_FP32 ||
            param.DataDes.dataType == HCCL_DATA_TYPE_BFP16)
        && (param.reduceType == HCCL_REDUCE_SUM)
        && rankSize_ >= MIN_STRICT_RANK_NUM_ORDER_PRESERVED;
    return isStrictMode;
}

template <typename AlgTopoMatch, typename InsAlgTemplateRS, typename InsAlgTemplateAG>
bool InsV2AllReduceOrderPreservedExecutor<AlgTopoMatch, InsAlgTemplateRS, InsAlgTemplateAG>::CheckStrictCondition(const OpParam &param) const
{
    CHK_PRT_RET(param.reduceType == HCCL_REDUCE_PROD,
        HCCL_ERROR("[CheckStrictCondition] DETERMINISTIC_STRICT mode not support PROD."), false);
    CHK_PRT_RET(param.DataDes.dataType == HCCL_DATA_TYPE_FP64,
        HCCL_ERROR("[CheckStrictCondition] DETERMINISTIC_STRICT mode not support FP64."), false);
    return true;
}

template <typename AlgTopoMatch, typename InsAlgTemplateRS, typename InsAlgTemplateAG>
HcclResult InsV2AllReduceOrderPreservedExecutor<AlgTopoMatch, InsAlgTemplateRS, InsAlgTemplateAG>::RunReduceScatter(
    const OpParam &param, const AlgResourceCtxSerializable &resCtx)
{
    HCCL_INFO("[RunReduceScatter] Start (flat level1)");

    if (rankSize_ == 1) {
        CHK_RET(RunReduceScatterSingleRank(param, resCtx));
        return HCCL_SUCCESS;
    }

    TemplateDataParams tempAlgParams;
    tempAlgParams.count = dataCount_;
    tempAlgParams.allRankSliceSize = memInfo_.groupSize;
    tempAlgParams.sliceOffset.clear();
    for (u32 i = 0; i < rankSize_; i++) {
        tempAlgParams.sliceOffset.push_back(i * memInfo_.sizePerBlock);
    }

    TemplateResource templateResource;
    templateResource.threads = threads_;
    if (remoteRankToChannelInfo_.size() > 0) {
        templateResource.channels = remoteRankToChannelInfo_[0];
    }

    std::shared_ptr<InsAlgTemplateRS> rsTempAlg =
        std::make_shared<InsAlgTemplateRS>(param, myRank_, algHierarchyInfo_.infos[0]);

    CHK_RET(rsTempAlg->KernelRun(param, tempAlgParams, templateResource));

    HCCL_INFO("[RunReduceScatter] Success");
    return HCCL_SUCCESS;
}

template <typename AlgTopoMatch, typename InsAlgTemplateRS, typename InsAlgTemplateAG>
HcclResult InsV2AllReduceOrderPreservedExecutor<AlgTopoMatch, InsAlgTemplateRS, InsAlgTemplateAG>::RunReduceScatterSingleRank(
    const OpParam &param, const AlgResourceCtxSerializable &resCtx)
{
    HCCL_INFO("[RunReduceScatterSingleRank] Skip for single rank");
    memInfo_.all2allOffset = 0;
    return HCCL_SUCCESS;
}

template <typename AlgTopoMatch, typename InsAlgTemplateRS, typename InsAlgTemplateAG>
HcclResult InsV2AllReduceOrderPreservedExecutor<AlgTopoMatch, InsAlgTemplateRS, InsAlgTemplateAG>::RunAllGather(
    const OpParam &param, const AlgResourceCtxSerializable &resCtx)
{
    HCCL_INFO("[RunAllGather] Start (using NHR template, flat level1)");

    TemplateDataParams tempAlgParams;
    tempAlgParams.count = dataCount_ / rankSize_;
    tempAlgParams.allRankSliceSize = memInfo_.groupSize;
    tempAlgParams.sliceOffset.clear();
    for (u32 i = 0; i < rankSize_; i++) {
        tempAlgParams.sliceOffset.push_back(i * memInfo_.sizePerBlock);
    }

    TemplateResource templateResource;
    templateResource.threads = threads_;
    if (remoteRankToChannelInfo_.size() > 0) {
        templateResource.channels = remoteRankToChannelInfo_[0];
    }

    std::shared_ptr<InsAlgTemplateAG> agTempAlg =
        std::make_shared<InsAlgTemplateAG>(param, myRank_, algHierarchyInfo_.infos[0]);

    CHK_RET(agTempAlg->KernelRun(param, tempAlgParams, templateResource));

    HCCL_INFO("[RunAllGather] Success");
    return HCCL_SUCCESS;
}

REGISTER_EXECUTOR_BY_TWO_TEMPS(HcclCMDType::HCCL_CMD_ALLREDUCE, AllReduceOrderPreserved,
    InsV2AllReduceOrderPreservedExecutor, TopoMatch1D,
    InsTempReduceScatterOrderPreserved, InsTempAllGatherNHR);

}